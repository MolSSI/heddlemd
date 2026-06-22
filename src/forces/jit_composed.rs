// rq-9f309378 rq-2d2eaf72
//! JIT-composed kernel infrastructure.
//!
//! Every fast-class slot exposes a CUDA source fragment via the
//! appropriate `PotentialBuilder` method (`pair_force_fragment`,
//! `bonded_force_fragment`, `angle_force_fragment`). The framework
//! collects the active fragments at `ForceField::new` time, grouped by
//! parallelism shape, concatenates each shape's fragments with a
//! shared preamble and a generated outer-loop body, JIT-compiles the
//! result via `cudarc::nvrtc::compile_ptx_with_opts`, and loads the
//! resulting PTX as a CUDA module per shape. At step time the framework
//! launches one composed kernel per active fast-class pair-force slot
//! plus one composed entry point per active bonded / angle slot per
//! `ForceField::step` / `step_class(Fast, …)` invocation in place of
//! the per-slot standalone kernels.
//!
//! See `rqm/forces/jit-composed-pair-force.md` (pair-force composer)
//! and `rqm/forces/jit-composed-intramolecular.md` (bonded / angle
//! composer).

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

use crate::gpu::{GpuError, ParticleBuffers};
use crate::pbc::SimulationBox;

use super::{ForceFieldError, NeighborListState};

const MODULE_NAME: &str = "heddle_jit_composed_pair_force";
const F_ENTRY: &str = "heddle_jit_composed_pair_force_f";
const FEV_ENTRY: &str = "heddle_jit_composed_pair_force_fev";

const WARPS_PER_BLOCK: u32 = 8;
const BLOCK_SIZE: u32 = WARPS_PER_BLOCK * 32;

/// Self-contained CUDA C++ source fragment plus identifying metadata,
/// returned by `PotentialBuilder::pair_force_fragment(cx)`. All four
/// source fields are concatenated by the composer into one nvrtc
/// translation unit.
///
/// See `rqm/forces/jit-composed-pair-force.md` for the contract on
/// what each piece must contain.
#[derive(Debug, Clone)]
pub struct PairForceFragment {
    /// The slot's stable label; matches the constructed slot's
    /// `Potential::label()`.
    pub label: &'static str,
    /// The name of the `__device__` functor struct the fragment
    /// defines (e.g. `"LjPairFunctor"`).
    pub functor_struct_name: &'static str,
    /// CUDA source for the functor struct plus any helper functions
    /// it depends on. Concatenated verbatim into the composed source
    /// above the composite-functor definition.
    pub functor_source: String,
    /// CUDA source for the fragment's contribution to the entry-point
    /// argument list. Each line declares one `extern "C"` kernel
    /// parameter, comma-terminated (newline after each comma is
    /// conventional). The composer concatenates these between the
    /// common args and the trailing `unsigned int n` parameter; the
    /// owning slot's `bind_pair_force_args` pushes one argument per
    /// declared parameter onto the builder in the same order.
    pub entry_point_args: String,
    /// CUDA source for the entry-point body's functor-field
    /// initialisation. The composer emits this once per launch
    /// invocation right after declaring the composite functor
    /// variable. The fragment is responsible for assigning every
    /// member of its functor instance from the entry-point args
    /// declared in `entry_point_args`.
    pub functor_init_source: String,
}

/// Context passed to every active fast-class pair-force slot's
/// `Potential::bind_pair_force_args(...)` call. Exposes references to
/// the per-step shared inputs every slot may need (positions, charges,
/// type indices live on `ParticleBuffers`; the lattice lives on
/// `SimulationBox`; the neighbour-list buffers live on
/// `NeighborListState`).
pub struct PairForceBindContext<'a> {
    pub buffers: &'a ParticleBuffers,
    pub sim_box: &'a SimulationBox,
    pub neighbor_list: &'a NeighborListState,
}

/// Self-contained CUDA C++ source fragment plus identifying metadata,
/// returned by `PotentialBuilder::bonded_force_fragment(cx)`. Same
/// field shape as `PairForceFragment`; the functor's contract differs
/// (per-bond evaluation, not per-pair). See
/// `rqm/forces/jit-composed-intramolecular.md`.
#[derive(Debug, Clone)]
pub struct BondedForceFragment {
    pub label: &'static str,
    pub functor_struct_name: &'static str,
    pub functor_source: String,
    pub entry_point_args: String,
    pub functor_init_source: String,
}

/// Self-contained CUDA C++ source fragment plus identifying metadata,
/// returned by `PotentialBuilder::angle_force_fragment(cx)`. Same
/// field shape as `BondedForceFragment`; the functor's contract is
/// the angle shape (per-angle evaluation taking displacements of
/// `r_ij` and `r_kj`).
#[derive(Debug, Clone)]
pub struct AngleForceFragment {
    pub label: &'static str,
    pub functor_struct_name: &'static str,
    pub functor_source: String,
    pub entry_point_args: String,
    pub functor_init_source: String,
}

/// Context passed to every active fast-class bonded slot's
/// `Potential::bind_bonded_force_args(...)` call and every active
/// fast-class angle slot's `Potential::bind_angle_force_args(...)`
/// call. Exposes references to the per-step shared inputs the slot
/// may need (positions / lattice are reached through `buffers` and
/// `sim_box`; the slot's bond / angle list and scratch buffer are
/// stored on the slot itself).
pub struct ForceLaunchContext<'a> {
    pub buffers: &'a ParticleBuffers,
    pub sim_box: &'a SimulationBox,
}

/// Bonded slot's per-launch scratch buffers exposed to the framework
/// so it can construct the composed-bonded-kernel argument list. The
/// slot owns the bond list and the bond-pair scratch buffer; the
/// framework needs read access to wire the common kernel args
/// (`bonds`, `bond_pair_x/y/z[, _energy, _virial]`).
pub struct BondedScratchView<'a> {
    pub bonds: &'a CudaSlice<u32>,
    pub bond_pair_x: &'a CudaSlice<crate::precision::Real>,
    pub bond_pair_y: &'a CudaSlice<crate::precision::Real>,
    pub bond_pair_z: &'a CudaSlice<crate::precision::Real>,
    pub bond_pair_energy: &'a CudaSlice<crate::precision::Real>,
    pub bond_pair_virial: &'a CudaSlice<crate::precision::Real>,
    pub bond_count: usize,
}

/// Angle slot's per-launch scratch buffers exposed to the framework
/// for the composed-angle-kernel argument list.
pub struct AngleScratchView<'a> {
    pub angles: &'a CudaSlice<u32>,
    pub angle_triple_x: &'a CudaSlice<crate::precision::Real>,
    pub angle_triple_y: &'a CudaSlice<crate::precision::Real>,
    pub angle_triple_z: &'a CudaSlice<crate::precision::Real>,
    pub angle_triple_energy: &'a CudaSlice<crate::precision::Real>,
    pub angle_triple_virial: &'a CudaSlice<crate::precision::Real>,
    pub angle_count: usize,
}

/// Self-contained CUDA C++ source fragment plus identifying metadata,
/// returned by `Integrator::post_force_per_particle_fragment(...)`,
/// `Thermostat::post_force_per_particle_fragment(...)`, and
/// `Barostat::post_force_per_particle_fragment(...)`. The composer
/// concatenates each fragment's per-thread body into the composed
/// post-force per-particle kernel in canonical slot order.
///
/// See `rqm/integration/jit-composed-post-force.md` for the contract
/// each fragment must satisfy.
#[derive(Debug, Clone)]
pub struct PerParticleFragment {
    pub label: &'static str,
    /// CUDA C++ source declaring helper `__device__` functions,
    /// structs, or constants the fragment's `per_thread_body` depends
    /// on. Concatenated verbatim into the composed source above the
    /// entry point. Empty for fragments that need no helpers.
    pub helper_source: String,
    /// CUDA C++ source declaring the fragment's contribution to the
    /// composed entry point's argument list. Each line declares one
    /// `extern "C"` kernel parameter, comma-terminated.
    pub entry_point_args: String,
    /// CUDA C++ source for the fragment's per-thread work. Variables
    /// in scope at the inlining point: `unsigned int i` (particle
    /// index, validated `i < n`), `Real lx, ly, lz, xy, xz, yz`
    /// (the lattice).
    pub per_thread_body: String,
}

/// Context passed to every active slot's
/// `bind_post_force_per_particle_args(...)` call.
pub struct PostForceBindContext<'a> {
    pub buffers: &'a ParticleBuffers,
    pub sim_box: &'a SimulationBox,
    pub dt: crate::precision::Real,
}

const POST_FORCE_MODULE_NAME: &str = "heddle_jit_composed_post_force_per_particle";
const POST_FORCE_ENTRY: &str = "heddle_jit_composed_post_force_per_particle";

/// JIT-composed post-force per-particle kernel module + entry-point
/// handle. Built by the runner when at least one integrator /
/// thermostat / barostat slot exposes a post-force fragment;
/// otherwise the runner carries `None` for this field and no
/// composed-kernel launch is attempted at step time.
///
/// See `rqm/integration/jit-composed-post-force.md`.
#[derive(Debug)]
pub struct JitComposedPostForcePerParticle {
    pub fragment_labels: Vec<&'static str>,
    pub entry_point: CudaFunction,
}

impl JitComposedPostForcePerParticle {
    pub fn compile_and_load(
        device: &Arc<CudaDevice>,
        fragments: &[PerParticleFragment],
    ) -> Result<Self, super::ForceFieldError> {
        let source = compose_post_force_source(fragments);
        let ptx = jit_compile(device, &source, |log| {
            super::ForceFieldError::FragmentCompileFailed {
                log: format_post_force_compile_failure(fragments, log, &source),
            }
        })?;
        device
            .load_ptx(ptx, POST_FORCE_MODULE_NAME, &[POST_FORCE_ENTRY])
            .map_err(|e| {
                super::ForceFieldError::FragmentLoadFailed(GpuError::from(e))
            })?;
        let entry_point = device
            .get_func(POST_FORCE_MODULE_NAME, POST_FORCE_ENTRY)
            .expect("composed post-force kernel entry was just loaded");
        Ok(JitComposedPostForcePerParticle {
            fragment_labels: fragments.iter().map(|f| f.label).collect(),
            entry_point,
        })
    }

    /// Launch the composed post-force per-particle kernel. `builder`
    /// must have been pre-populated with the common args (positions,
    /// images, velocities, forces, masses, lattice) followed by each
    /// active slot's per-fragment args in canonical slot order
    /// (integrator → thermostat → barostat), followed by the trailing
    /// `n` arg.
    ///
    /// # Safety
    /// The argument list must match the composed entry-point signature
    /// exactly. The runner is responsible for that invariant.
    pub unsafe fn launch(
        &self,
        n: u32,
        mut builder: PairForceLaunchBuilder,
    ) -> Result<(), GpuError> {
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = self.entry_point.clone();
        unsafe {
            func.launch(cfg, &mut builder.kernel_params)
                .map_err(GpuError::from)?;
        }
        drop(builder.storage);
        Ok(())
    }
}

fn compose_post_force_source(fragments: &[PerParticleFragment]) -> String {
    let mut s = String::with_capacity(
        8192
            + fragments
                .iter()
                .map(|f| f.helper_source.len() + f.per_thread_body.len())
                .sum::<usize>(),
    );
    s.push_str(PREAMBLE);
    // Each fragment's helper source above the entry point, with the
    // slot's label noted so nvrtc compile errors can be traced.
    for f in fragments {
        s.push_str("// ---- post-force helper source: ");
        s.push_str(f.label);
        s.push_str(" ----\n");
        s.push_str(&f.helper_source);
        s.push_str("\n// ---- end post-force helper source: ");
        s.push_str(f.label);
        s.push_str(" ----\n");
    }

    // Entry-point signature: common args + per-fragment args + n.
    s.push_str("\nextern \"C\" __global__ void ");
    s.push_str(POST_FORCE_ENTRY);
    s.push_str("(\n");
    s.push_str("    Real *positions_x,\n");
    s.push_str("    Real *positions_y,\n");
    s.push_str("    Real *positions_z,\n");
    s.push_str("    int *images_x,\n");
    s.push_str("    int *images_y,\n");
    s.push_str("    int *images_z,\n");
    s.push_str("    Real *velocities_x,\n");
    s.push_str("    Real *velocities_y,\n");
    s.push_str("    Real *velocities_z,\n");
    s.push_str("    const Real *forces_x,\n");
    s.push_str("    const Real *forces_y,\n");
    s.push_str("    const Real *forces_z,\n");
    s.push_str("    const Real *masses,\n");
    s.push_str("    const Real *lattice,\n");
    for f in fragments {
        s.push_str(&f.entry_point_args);
    }
    s.push_str("    unsigned int n)\n");
    s.push_str("{\n");
    s.push_str("    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str("    if (i >= n) return;\n");
    s.push_str("    Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];\n");
    s.push_str("    Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];\n");
    for f in fragments {
        s.push_str("    // ---- per-thread body: ");
        s.push_str(f.label);
        s.push_str(" ----\n    {\n");
        s.push_str(&f.per_thread_body);
        s.push_str("\n    }\n");
    }
    s.push_str("}\n");
    s
}

fn format_post_force_compile_failure(
    fragments: &[PerParticleFragment],
    log: &str,
    source: &str,
) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "nvrtc failed to compile the JIT-composed post-force per-particle kernel."
    );
    let _ = writeln!(s, "Active fragments (canonical slot order):");
    for f in fragments {
        let _ = writeln!(s, "  - {}", f.label);
    }
    let _ = writeln!(s, "nvrtc compile log:");
    let _ = writeln!(s, "{}", log);
    let _ = writeln!(s, "Composed source (line-numbered):");
    for (i, line) in source.lines().enumerate() {
        let _ = writeln!(s, "{:5}: {}", i + 1, line);
    }
    s
}

/// Shape-agnostic alias for `PairForceLaunchBuilder` — the binding
/// mechanism is the same across the pair-force, bonded, and angle
/// composers, so the launch builder is one type with multiple names.
pub type ForceLaunchBuilder = PairForceLaunchBuilder;

/// Argument-builder threaded through every active fast-class pair-force
/// slot's `Potential::bind_pair_force_args(...)` call. Pre-populated by
/// the framework with the composed kernel's common arguments; each slot
/// then pushes its parameter buffers and scalars in the order its
/// fragment expects them.
pub struct PairForceLaunchBuilder {
    /// Owned storage for each argument's bytes. Pointers in
    /// `kernel_params` point into the `Box<[u8]>` heap allocations.
    /// Box ensures the allocation address is stable across pushes onto
    /// the outer Vec.
    storage: Vec<Box<[u8]>>,
    kernel_params: Vec<*mut c_void>,
}

impl Default for PairForceLaunchBuilder {
    fn default() -> Self {
        PairForceLaunchBuilder {
            storage: Vec::new(),
            kernel_params: Vec::new(),
        }
    }
}

impl PairForceLaunchBuilder {
    pub fn new() -> Self {
        PairForceLaunchBuilder::default()
    }

    /// Push a CUDA device buffer's device pointer as a kernel
    /// argument. The kernel will see a `T*` parameter.
    pub fn push_device_buffer<T>(&mut self, buf: &CudaSlice<T>) {
        let dev_ptr: u64 = *buf.device_ptr();
        self.push_scalar(dev_ptr);
    }

    /// Push a `Copy` scalar value as a kernel argument. The kernel
    /// will see a `T` parameter (passed by value).
    pub fn push_scalar<T: Copy>(&mut self, value: T) {
        let size = std::mem::size_of::<T>();
        let mut bytes: Box<[u8]> = vec![0u8; size].into_boxed_slice();
        unsafe {
            std::ptr::copy_nonoverlapping(
                &value as *const T as *const u8,
                bytes.as_mut_ptr(),
                size,
            );
        }
        let ptr = bytes.as_mut_ptr() as *mut c_void;
        self.storage.push(bytes);
        self.kernel_params.push(ptr);
    }
}

/// JIT-composed pair-force kernel module + entry-point handles. Built
/// by `ForceField::new` when at least one fast-class pair-force slot is
/// active; otherwise the `ForceField` carries `None` for this field and
/// no composed-kernel launch is attempted at step time.
#[derive(Debug)]
pub struct JitComposedPairForce {
    pub fragment_labels: Vec<&'static str>,
    pub pair_force_f: CudaFunction,
    pub pair_force_fev: CudaFunction,
}

impl JitComposedPairForce {
    /// Compose, compile, and load the composed kernel from the active
    /// fragments. `fragments` is the active fast-class pair-force
    /// fragment list in canonical slot order.
    pub fn compile_and_load(
        device: &Arc<CudaDevice>,
        fragments: &[PairForceFragment],
    ) -> Result<Self, ForceFieldError> {
        let source = compose_source(fragments);

        let arch_arg = detect_arch_option(device);
        let mut options = vec!["--std=c++17".to_string()];
        if let Some(a) = arch_arg {
            options.push(a);
        }
        #[cfg(feature = "f64")]
        options.push("--define-macro=HEDDLE_REAL_F64".to_string());
        let opts = CompileOptions {
            options,
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(&source, opts).map_err(|e| {
            let log = match e {
                cudarc::nvrtc::CompileError::CompileError { ref log, .. } => log
                    .to_str()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|_| format!("{e:?}")),
                _ => format!("{e:?}"),
            };
            ForceFieldError::FragmentCompileFailed {
                log: format_compile_failure(fragments, &log, &source),
            }
        })?;

        device
            .load_ptx(ptx, MODULE_NAME, &[F_ENTRY, FEV_ENTRY])
            .map_err(|e| ForceFieldError::FragmentLoadFailed(GpuError::from(e)))?;
        let pair_force_f = device
            .get_func(MODULE_NAME, F_ENTRY)
            .expect("composed pair-force kernel _f entry was just loaded");
        let pair_force_fev = device
            .get_func(MODULE_NAME, FEV_ENTRY)
            .expect("composed pair-force kernel _fev entry was just loaded");

        Ok(JitComposedPairForce {
            fragment_labels: fragments.iter().map(|f| f.label).collect(),
            pair_force_f,
            pair_force_fev,
        })
    }

    /// Launch the composed kernel for `n` particles. `use_fev`
    /// selects the `_fev` entry point when true and the `_f` entry
    /// point when false. `builder` must have been pre-populated with
    /// the common args, the per-fragment args (in canonical slot
    /// order), and the final `n` arg.
    ///
    /// # Safety
    /// `builder`'s argument list must match the composed kernel's
    /// entry-point signature exactly. The framework's per-step
    /// dispatch is responsible for that invariant.
    pub unsafe fn launch(
        &self,
        n: u32,
        use_fev: bool,
        mut builder: PairForceLaunchBuilder,
    ) -> Result<(), GpuError> {
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(WARPS_PER_BLOCK), 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = if use_fev {
            self.pair_force_fev.clone()
        } else {
            self.pair_force_f.clone()
        };
        unsafe {
            func.launch(cfg, &mut builder.kernel_params)
                .map_err(GpuError::from)?;
        }
        // Keep `builder.storage` alive across the launch so the
        // pointers in `kernel_params` remain valid until cuLaunchKernel
        // returns.
        drop(builder.storage);
        Ok(())
    }
}

fn detect_arch_option(device: &Arc<CudaDevice>) -> Option<String> {
    use cudarc::driver::sys;
    let mut major: i32 = 0;
    let mut minor: i32 = 0;
    let dev_ord = device.ordinal();
    unsafe {
        let lib = sys::lib();
        let mut cuda_device: sys::CUdevice = 0;
        if lib.cuDeviceGet(&mut cuda_device, dev_ord as i32)
            != sys::cudaError_enum::CUDA_SUCCESS
        {
            return None;
        }
        if lib.cuDeviceGetAttribute(
            &mut major,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            cuda_device,
        ) != sys::cudaError_enum::CUDA_SUCCESS
        {
            return None;
        }
        if lib.cuDeviceGetAttribute(
            &mut minor,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            cuda_device,
        ) != sys::cudaError_enum::CUDA_SUCCESS
        {
            return None;
        }
    }
    Some(format!("--gpu-architecture=compute_{}{}", major, minor))
}

fn format_compile_failure(
    fragments: &[PairForceFragment],
    log: &str,
    source: &str,
) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "nvrtc failed to compile the JIT-composed pair-force kernel."
    );
    let _ = writeln!(s, "Active fragments (canonical slot order):");
    for f in fragments {
        let _ = writeln!(s, "  - {} (functor: {})", f.label, f.functor_struct_name);
    }
    let _ = writeln!(s, "nvrtc compile log:");
    let _ = writeln!(s, "{}", log);
    // Append numbered source lines for easier inspection of nvrtc
    // line:column references in the log.
    let _ = writeln!(s, "Composed source (line-numbered):");
    for (i, line) in source.lines().enumerate() {
        let _ = writeln!(s, "{:5}: {}", i + 1, line);
    }
    s
}

fn functor_field_name(label: &str) -> String {
    let mut out = String::with_capacity(label.len() + 1);
    out.push_str("functor_");
    for c in label.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    out
}

fn compose_source(fragments: &[PairForceFragment]) -> String {
    let mut s = String::with_capacity(
        8192 + fragments.iter().map(|f| f.functor_source.len()).sum::<usize>(),
    );
    s.push_str(PREAMBLE);
    for f in fragments {
        s.push_str("// ---- fragment functor source: ");
        s.push_str(f.label);
        s.push_str(" ----\n");
        s.push_str(&f.functor_source);
        s.push_str("\n// ---- end fragment functor source: ");
        s.push_str(f.label);
        s.push_str(" ----\n");
    }

    // Composite-functor struct: one field per active fragment, each
    // typed as the fragment's declared functor struct name.
    s.push_str("\nstruct HeddleJitComposedPairFunc {\n");
    for f in fragments {
        s.push_str("    ");
        s.push_str(f.functor_struct_name);
        s.push(' ');
        s.push_str(&functor_field_name(f.label));
        s.push_str(";\n");
    }
    s.push_str("};\n");

    // Per-pair inner-loop body (called from the outer loop, splice in
    // each fragment's evaluation in canonical slot order).
    s.push_str("\ntemplate <bool WriteEv>\n");
    s.push_str("__device__ static inline void heddle_jit_eval_pair(\n");
    s.push_str("    const HeddleJitComposedPairFunc &composite,\n");
    s.push_str("    Real r2, unsigned int i, unsigned int j,\n");
    s.push_str("    Real dx, Real dy, Real dz,\n");
    s.push_str("    Real &p_x, Real &p_y, Real &p_z,\n");
    s.push_str("    Real &p_e, Real &p_w)\n");
    s.push_str("{\n");
    s.push_str("    Real factor = R(0.0), energy = R(0.0), virial = R(0.0);\n");
    for f in fragments {
        let field = functor_field_name(f.label);
        s.push_str(&format!(
            "    {{\n        Real cut2 = composite.{f}.cutoff_squared(i, j);\n        \
             if (r2 <= cut2) {{\n            Real s_factor, s_energy, s_virial;\n            \
             composite.{f}.evaluate(r2, i, j, s_factor, s_energy, s_virial);\n            \
             Real scale = composite.{f}.exclusion_scale(i, j);\n            \
             factor += s_factor * scale;\n            \
             if (WriteEv) {{ energy += s_energy * scale; virial += s_virial * scale; }}\n        \
             }}\n    }}\n",
            f = field
        ));
    }
    s.push_str("    p_x += factor * dx;\n");
    s.push_str("    p_y += factor * dy;\n");
    s.push_str("    p_z += factor * dz;\n");
    s.push_str("    if (WriteEv) {\n");
    s.push_str("        p_e += energy * R(0.5);\n");
    s.push_str("        p_w += virial * R(0.5);\n");
    s.push_str("    }\n");
    s.push_str("}\n");

    s.push_str(OUTER_LOOP_TEMPLATE);

    // _f entry point
    emit_entry_point(&mut s, fragments, F_ENTRY, false);
    // _fev entry point
    emit_entry_point(&mut s, fragments, FEV_ENTRY, true);

    s
}

fn emit_entry_point(
    s: &mut String,
    fragments: &[PairForceFragment],
    entry_name: &str,
    write_ev: bool,
) {
    s.push_str("\nextern \"C\" __global__ void ");
    s.push_str(entry_name);
    s.push_str("(\n");
    s.push_str("    const Real *positions_x,\n");
    s.push_str("    const Real *positions_y,\n");
    s.push_str("    const Real *positions_z,\n");
    s.push_str("    const unsigned int *neighbor_list,\n");
    s.push_str("    const unsigned int *neighbor_counts,\n");
    s.push_str("    unsigned int max_neighbors,\n");
    s.push_str("    const Real *lattice,\n");
    s.push_str("    Real *slot_force_x,\n");
    s.push_str("    Real *slot_force_y,\n");
    s.push_str("    Real *slot_force_z,\n");
    if write_ev {
        s.push_str("    Real *slot_energy,\n");
        s.push_str("    Real *slot_virial,\n");
    }
    for f in fragments {
        s.push_str(&f.entry_point_args);
    }
    s.push_str("    unsigned int n)\n");
    s.push_str("{\n");
    s.push_str("    HeddleJitComposedPairFunc composite;\n");
    for f in fragments {
        s.push_str(&f.functor_init_source);
    }
    s.push_str("    heddle_jit_outer_loop<");
    s.push_str(if write_ev { "true" } else { "false" });
    s.push_str(">(\n");
    s.push_str("        composite, n, max_neighbors,\n");
    s.push_str("        positions_x, positions_y, positions_z,\n");
    s.push_str("        neighbor_list, neighbor_counts,\n");
    s.push_str("        lattice,\n");
    s.push_str("        slot_force_x, slot_force_y, slot_force_z,\n");
    if write_ev {
        s.push_str("        slot_energy, slot_virial);\n");
    } else {
        s.push_str("        nullptr, nullptr);\n");
    }
    s.push_str("}\n");
}

/// Inlined preamble: precision shim, PBC minimum-image helpers,
/// exclusion-scale generic helper, warp-reduce helper, block-size
/// constants. Held verbatim as a single `&'static str` so the same
/// preamble compiles into every composed source regardless of which
/// fragments are active.
const PREAMBLE: &str = r#"// Heddle JIT-composed pair-force kernel preamble.
#ifdef HEDDLE_REAL_F64
typedef double Real;
#define R(x) ((Real)(x))
__device__ __forceinline__ Real Real_sqrt(Real x) { return sqrt(x); }
__device__ __forceinline__ Real Real_rsqrt(Real x) { return rsqrt(x); }
__device__ __forceinline__ Real Real_exp(Real x) { return exp(x); }
__device__ __forceinline__ Real Real_log(Real x) { return log(x); }
__device__ __forceinline__ Real Real_floor(Real x) { return floor(x); }
__device__ __forceinline__ Real Real_fma(Real a, Real b, Real c) { return fma(a, b, c); }
__device__ __forceinline__ Real Real_erfc(Real x) { return erfc(x); }
__device__ __forceinline__ Real Real_atan2(Real y, Real x) { return atan2(y, x); }
#else
typedef float Real;
#define R(x) ((Real)(x))
__device__ __forceinline__ Real Real_sqrt(Real x) { return sqrtf(x); }
__device__ __forceinline__ Real Real_rsqrt(Real x) { return rsqrtf(x); }
__device__ __forceinline__ Real Real_exp(Real x) { return expf(x); }
__device__ __forceinline__ Real Real_log(Real x) { return logf(x); }
__device__ __forceinline__ Real Real_floor(Real x) { return floorf(x); }
__device__ __forceinline__ Real Real_fma(Real a, Real b, Real c) { return fmaf(a, b, c); }
__device__ __forceinline__ Real Real_erfc(Real x) { return erfcf(x); }
__device__ __forceinline__ Real Real_atan2(Real y, Real x) { return atan2f(y, x); }
#endif

#define HEDDLE_JIT_WARP_SIZE 32
#define HEDDLE_JIT_WARPS_PER_BLOCK 8

__device__ __forceinline__ Real heddle_jit_warp_reduce_sum(Real v) {
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v;
}

__device__ static inline void heddle_jit_triclinic_cart_to_frac(
    Real x, Real y, Real z,
    Real lx, Real ly, Real lz,
    Real xy, Real xz, Real yz,
    Real &s_a, Real &s_b, Real &s_c)
{
  s_c = z / lz;
  s_b = (y - s_c * yz) / ly;
  s_a = (x - s_b * xy - s_c * xz) / lx;
}

__device__ static inline void heddle_jit_triclinic_min_image(
    Real &dx, Real &dy, Real &dz,
    Real lx, Real ly, Real lz,
    Real xy, Real xz, Real yz)
{
  Real s_a, s_b, s_c;
  heddle_jit_triclinic_cart_to_frac(dx, dy, dz, lx, ly, lz, xy, xz, yz, s_a, s_b, s_c);
  Real ka = Real_floor(s_a + R(0.5));
  Real kb = Real_floor(s_b + R(0.5));
  Real kc = Real_floor(s_c + R(0.5));
  dx -= ka * lx + kb * xy + kc * xz;
  dy -= kb * ly + kc * yz;
  dz -= kc * lz;
}

// Generic exclusion-scale lookup used by every fragment's
// `exclusion_scale(i, j)` method when it indexes into a per-pair
// scale table.
__device__ static inline Real heddle_jit_exclusion_scale(
    unsigned int i, unsigned int j,
    const unsigned int *offsets,
    const unsigned int *partners,
    const Real *scales)
{
  unsigned int start = offsets[i];
  unsigned int end = offsets[i + 1];
  for (unsigned int m = start; m < end; ++m) {
    if (partners[m] == j) return scales[m];
  }
  return R(1.0);
}
"#;

const OUTER_LOOP_TEMPLATE: &str = r#"
template <bool WriteEv>
__device__ static inline void heddle_jit_outer_loop(
    const HeddleJitComposedPairFunc &composite,
    unsigned int n,
    unsigned int max_neighbors,
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    const Real *lattice,
    Real *slot_force_x,
    Real *slot_force_y,
    Real *slot_force_z,
    Real *slot_energy,
    Real *slot_virial)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];

  unsigned int warp_id_in_block = threadIdx.x / HEDDLE_JIT_WARP_SIZE;
  unsigned int lane = threadIdx.x & (HEDDLE_JIT_WARP_SIZE - 1u);
  unsigned int i = blockIdx.x * HEDDLE_JIT_WARPS_PER_BLOCK + warp_id_in_block;
  if (i >= n) return;

  unsigned int count = neighbor_counts[i];
  unsigned int row_base = i * max_neighbors;
  unsigned int sweep_end =
      ((count + HEDDLE_JIT_WARP_SIZE - 1u) / HEDDLE_JIT_WARP_SIZE) * HEDDLE_JIT_WARP_SIZE;

  Real p_x = R(0.0), p_y = R(0.0), p_z = R(0.0);
  Real p_e = R(0.0), p_w = R(0.0);

  Real pi_x = positions_x[i];
  Real pi_y = positions_y[i];
  Real pi_z = positions_z[i];

  for (unsigned int s = 0u; s < sweep_end; s += HEDDLE_JIT_WARP_SIZE) {
    unsigned int k = s + lane;
    if (k < count) {
      unsigned int j = neighbor_list[row_base + k];
      if (i != j) {
        Real dx = pi_x - positions_x[j];
        Real dy = pi_y - positions_y[j];
        Real dz = pi_z - positions_z[j];
        heddle_jit_triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);
        Real r2 = dx * dx + dy * dy + dz * dz;
        heddle_jit_eval_pair<WriteEv>(composite, r2, i, j, dx, dy, dz,
                                       p_x, p_y, p_z, p_e, p_w);
      }
    }
  }

  p_x = heddle_jit_warp_reduce_sum(p_x);
  p_y = heddle_jit_warp_reduce_sum(p_y);
  p_z = heddle_jit_warp_reduce_sum(p_z);
  if (WriteEv) {
    p_e = heddle_jit_warp_reduce_sum(p_e);
    p_w = heddle_jit_warp_reduce_sum(p_w);
  }

  if (lane == 0u) {
    slot_force_x[i] += p_x;
    slot_force_y[i] += p_y;
    slot_force_z[i] += p_z;
    if (WriteEv) {
      slot_energy[i] += p_e;
      slot_virial[i] += p_w;
    }
  }
}
"#;

// ============================================================
// Bonded composer
// ============================================================

const BONDED_MODULE_NAME: &str = "heddle_jit_composed_bonded";

/// JIT-composed bonded contribution module + per-slot entry-point
/// handles. Built by `ForceField::new` when at least one fast-class
/// bonded slot is active; otherwise the `ForceField` carries `None`
/// for this field and no composed-bonded launch is attempted at step
/// time. Each active slot contributes one `_f` entry point and one
/// `_fev` entry point, indexed by canonical slot order among active
/// bonded slots.
#[derive(Debug)]
pub struct JitComposedBondedForce {
    pub fragment_labels: Vec<&'static str>,
    /// Per-slot `_f` entry points, indexed by canonical slot order
    /// among active bonded slots (zero-based).
    pub entry_points_f: Vec<CudaFunction>,
    /// Per-slot `_fev` entry points, indexed identically to
    /// `entry_points_f`.
    pub entry_points_fev: Vec<CudaFunction>,
}

impl JitComposedBondedForce {
    pub fn compile_and_load(
        device: &Arc<CudaDevice>,
        fragments: &[BondedForceFragment],
    ) -> Result<Self, ForceFieldError> {
        let source = compose_bonded_source(fragments);
        let ptx = jit_compile(device, &source, |log| {
            ForceFieldError::FragmentCompileFailed {
                log: format_bonded_compile_failure(fragments, log, &source),
            }
        })?;

        // cudarc's load_ptx requires `&[&'static str]`; the per-slot
        // entry names are dynamic. Leak each name to satisfy the
        // 'static bound. The leak is bounded by the slot count and
        // is paid once per `ForceField::new`.
        let mut entry_name_refs: Vec<&'static str> = Vec::with_capacity(2 * fragments.len());
        for i in 0..fragments.len() {
            entry_name_refs.push(Box::leak(
                format!("heddle_jit_composed_bonded_{}_f", i).into_boxed_str(),
            ));
            entry_name_refs.push(Box::leak(
                format!("heddle_jit_composed_bonded_{}_fev", i).into_boxed_str(),
            ));
        }

        device
            .load_ptx(ptx, BONDED_MODULE_NAME, &entry_name_refs)
            .map_err(|e| ForceFieldError::FragmentLoadFailed(GpuError::from(e)))?;

        let mut entry_points_f: Vec<CudaFunction> = Vec::with_capacity(fragments.len());
        let mut entry_points_fev: Vec<CudaFunction> = Vec::with_capacity(fragments.len());
        for i in 0..fragments.len() {
            entry_points_f.push(
                device
                    .get_func(BONDED_MODULE_NAME, entry_name_refs[2 * i])
                    .expect("composed bonded kernel _f entry was just loaded"),
            );
            entry_points_fev.push(
                device
                    .get_func(BONDED_MODULE_NAME, entry_name_refs[2 * i + 1])
                    .expect("composed bonded kernel _fev entry was just loaded"),
            );
        }

        Ok(JitComposedBondedForce {
            fragment_labels: fragments.iter().map(|f| f.label).collect(),
            entry_points_f,
            entry_points_fev,
        })
    }

    /// Launch one slot's composed bonded entry point.
    ///
    /// # Safety
    /// `builder`'s argument list must match the entry point's
    /// signature: common args (positions_x/y/z, bonds, lattice,
    /// bond_pair_x/y/z[, bond_pair_energy, bond_pair_virial when
    /// `use_fev`], per-fragment args, n_bonds). The framework's
    /// per-step dispatch is responsible for that invariant.
    pub unsafe fn launch_slot(
        &self,
        slot_index: usize,
        n_bonds: u32,
        use_fev: bool,
        mut builder: ForceLaunchBuilder,
    ) -> Result<(), GpuError> {
        let cfg = LaunchConfig {
            grid_dim: (n_bonds.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = if use_fev {
            self.entry_points_fev[slot_index].clone()
        } else {
            self.entry_points_f[slot_index].clone()
        };
        unsafe {
            func.launch(cfg, &mut builder.kernel_params)
                .map_err(GpuError::from)?;
        }
        drop(builder.storage);
        Ok(())
    }
}

fn compose_bonded_source(fragments: &[BondedForceFragment]) -> String {
    let mut s = String::with_capacity(
        8192 + fragments.iter().map(|f| f.functor_source.len()).sum::<usize>(),
    );
    s.push_str(PREAMBLE);
    for f in fragments {
        s.push_str("// ---- bonded fragment functor source: ");
        s.push_str(f.label);
        s.push_str(" ----\n");
        s.push_str(&f.functor_source);
        s.push_str("\n// ---- end bonded fragment functor source: ");
        s.push_str(f.label);
        s.push_str(" ----\n");
    }
    for (i, f) in fragments.iter().enumerate() {
        emit_bonded_entry_point(&mut s, f, i, false);
        emit_bonded_entry_point(&mut s, f, i, true);
    }
    s
}

fn emit_bonded_entry_point(
    s: &mut String,
    fragment: &BondedForceFragment,
    slot_index: usize,
    write_ev: bool,
) {
    let entry_name = format!(
        "heddle_jit_composed_bonded_{}_{}",
        slot_index,
        if write_ev { "fev" } else { "f" }
    );
    s.push_str("\nextern \"C\" __global__ void ");
    s.push_str(&entry_name);
    s.push_str("(\n");
    s.push_str("    const Real *positions_x,\n");
    s.push_str("    const Real *positions_y,\n");
    s.push_str("    const Real *positions_z,\n");
    s.push_str("    const unsigned int *bonds,\n");
    s.push_str("    const Real *lattice,\n");
    s.push_str("    Real *bond_pair_x,\n");
    s.push_str("    Real *bond_pair_y,\n");
    s.push_str("    Real *bond_pair_z,\n");
    if write_ev {
        s.push_str("    Real *bond_pair_energy,\n");
        s.push_str("    Real *bond_pair_virial,\n");
    }
    s.push_str(&fragment.entry_point_args);
    s.push_str("    unsigned int n_bonds)\n");
    s.push_str("{\n");
    s.push_str(&format!(
        "    {} functor;\n",
        fragment.functor_struct_name
    ));
    s.push_str(&fragment.functor_init_source);
    s.push_str("    Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];\n");
    s.push_str("    Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];\n");
    s.push_str("    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str("    if (k >= n_bonds) return;\n");
    s.push_str("    unsigned int atom_i = bonds[3u * k + 0u];\n");
    s.push_str("    unsigned int atom_j = bonds[3u * k + 1u];\n");
    s.push_str("    unsigned int type_idx = bonds[3u * k + 2u];\n");
    s.push_str("    Real dx = positions_x[atom_i] - positions_x[atom_j];\n");
    s.push_str("    Real dy = positions_y[atom_i] - positions_y[atom_j];\n");
    s.push_str("    Real dz = positions_z[atom_i] - positions_z[atom_j];\n");
    s.push_str("    heddle_jit_triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);\n");
    s.push_str("    Real r2 = dx * dx + dy * dy + dz * dz;\n");
    s.push_str("    if (r2 == R(0.0)) {\n");
    s.push_str("        bond_pair_x[2u * k]      = R(0.0);\n");
    s.push_str("        bond_pair_y[2u * k]      = R(0.0);\n");
    s.push_str("        bond_pair_z[2u * k]      = R(0.0);\n");
    s.push_str("        bond_pair_x[2u * k + 1u] = R(0.0);\n");
    s.push_str("        bond_pair_y[2u * k + 1u] = R(0.0);\n");
    s.push_str("        bond_pair_z[2u * k + 1u] = R(0.0);\n");
    if write_ev {
        s.push_str("        bond_pair_energy[2u * k]      = R(0.0);\n");
        s.push_str("        bond_pair_energy[2u * k + 1u] = R(0.0);\n");
        s.push_str("        bond_pair_virial[2u * k]      = R(0.0);\n");
        s.push_str("        bond_pair_virial[2u * k + 1u] = R(0.0);\n");
    }
    s.push_str("        return;\n");
    s.push_str("    }\n");
    s.push_str("    Real r = Real_sqrt(r2);\n");
    s.push_str("    Real fmag, u_k, w_k;\n");
    s.push_str("    functor.evaluate(r2, r, type_idx, dx, dy, dz, fmag, u_k, w_k);\n");
    s.push_str("    bond_pair_x[2u * k]      =  fmag * dx;\n");
    s.push_str("    bond_pair_y[2u * k]      =  fmag * dy;\n");
    s.push_str("    bond_pair_z[2u * k]      =  fmag * dz;\n");
    s.push_str("    bond_pair_x[2u * k + 1u] = -fmag * dx;\n");
    s.push_str("    bond_pair_y[2u * k + 1u] = -fmag * dy;\n");
    s.push_str("    bond_pair_z[2u * k + 1u] = -fmag * dz;\n");
    if write_ev {
        s.push_str("    bond_pair_energy[2u * k]      = u_k * R(0.5);\n");
        s.push_str("    bond_pair_energy[2u * k + 1u] = u_k * R(0.5);\n");
        s.push_str("    bond_pair_virial[2u * k]      = w_k * R(0.5);\n");
        s.push_str("    bond_pair_virial[2u * k + 1u] = w_k * R(0.5);\n");
    }
    s.push_str("}\n");
}

fn format_bonded_compile_failure(
    fragments: &[BondedForceFragment],
    log: &str,
    source: &str,
) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "nvrtc failed to compile the JIT-composed bonded kernel."
    );
    let _ = writeln!(s, "Active bonded fragments (canonical slot order):");
    for f in fragments {
        let _ = writeln!(s, "  - {} (functor: {})", f.label, f.functor_struct_name);
    }
    let _ = writeln!(s, "nvrtc compile log:");
    let _ = writeln!(s, "{}", log);
    let _ = writeln!(s, "Composed bonded source (line-numbered):");
    for (i, line) in source.lines().enumerate() {
        let _ = writeln!(s, "{:5}: {}", i + 1, line);
    }
    s
}

// ============================================================
// Angle composer
// ============================================================

const ANGLE_MODULE_NAME: &str = "heddle_jit_composed_angle";

/// JIT-composed angle contribution module + per-slot entry-point
/// handles. Built by `ForceField::new` when at least one fast-class
/// angle slot is active.
#[derive(Debug)]
pub struct JitComposedAngleForce {
    pub fragment_labels: Vec<&'static str>,
    pub entry_points_f: Vec<CudaFunction>,
    pub entry_points_fev: Vec<CudaFunction>,
}

impl JitComposedAngleForce {
    pub fn compile_and_load(
        device: &Arc<CudaDevice>,
        fragments: &[AngleForceFragment],
    ) -> Result<Self, ForceFieldError> {
        let source = compose_angle_source(fragments);
        let ptx = jit_compile(device, &source, |log| {
            ForceFieldError::FragmentCompileFailed {
                log: format_angle_compile_failure(fragments, log, &source),
            }
        })?;

        let mut entry_name_refs: Vec<&'static str> = Vec::with_capacity(2 * fragments.len());
        for i in 0..fragments.len() {
            entry_name_refs.push(Box::leak(
                format!("heddle_jit_composed_angle_{}_f", i).into_boxed_str(),
            ));
            entry_name_refs.push(Box::leak(
                format!("heddle_jit_composed_angle_{}_fev", i).into_boxed_str(),
            ));
        }

        device
            .load_ptx(ptx, ANGLE_MODULE_NAME, &entry_name_refs)
            .map_err(|e| ForceFieldError::FragmentLoadFailed(GpuError::from(e)))?;

        let mut entry_points_f: Vec<CudaFunction> = Vec::with_capacity(fragments.len());
        let mut entry_points_fev: Vec<CudaFunction> = Vec::with_capacity(fragments.len());
        for i in 0..fragments.len() {
            entry_points_f.push(
                device
                    .get_func(ANGLE_MODULE_NAME, entry_name_refs[2 * i])
                    .expect("composed angle kernel _f entry was just loaded"),
            );
            entry_points_fev.push(
                device
                    .get_func(ANGLE_MODULE_NAME, entry_name_refs[2 * i + 1])
                    .expect("composed angle kernel _fev entry was just loaded"),
            );
        }

        Ok(JitComposedAngleForce {
            fragment_labels: fragments.iter().map(|f| f.label).collect(),
            entry_points_f,
            entry_points_fev,
        })
    }

    /// Launch one slot's composed angle entry point.
    pub unsafe fn launch_slot(
        &self,
        slot_index: usize,
        n_angles: u32,
        use_fev: bool,
        mut builder: ForceLaunchBuilder,
    ) -> Result<(), GpuError> {
        let cfg = LaunchConfig {
            grid_dim: (n_angles.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = if use_fev {
            self.entry_points_fev[slot_index].clone()
        } else {
            self.entry_points_f[slot_index].clone()
        };
        unsafe {
            func.launch(cfg, &mut builder.kernel_params)
                .map_err(GpuError::from)?;
        }
        drop(builder.storage);
        Ok(())
    }
}

fn compose_angle_source(fragments: &[AngleForceFragment]) -> String {
    let mut s = String::with_capacity(
        8192 + fragments.iter().map(|f| f.functor_source.len()).sum::<usize>(),
    );
    s.push_str(PREAMBLE);
    for f in fragments {
        s.push_str("// ---- angle fragment functor source: ");
        s.push_str(f.label);
        s.push_str(" ----\n");
        s.push_str(&f.functor_source);
        s.push_str("\n// ---- end angle fragment functor source: ");
        s.push_str(f.label);
        s.push_str(" ----\n");
    }
    for (i, f) in fragments.iter().enumerate() {
        emit_angle_entry_point(&mut s, f, i, false);
        emit_angle_entry_point(&mut s, f, i, true);
    }
    s
}

fn emit_angle_entry_point(
    s: &mut String,
    fragment: &AngleForceFragment,
    slot_index: usize,
    write_ev: bool,
) {
    let entry_name = format!(
        "heddle_jit_composed_angle_{}_{}",
        slot_index,
        if write_ev { "fev" } else { "f" }
    );
    s.push_str("\nextern \"C\" __global__ void ");
    s.push_str(&entry_name);
    s.push_str("(\n");
    s.push_str("    const Real *positions_x,\n");
    s.push_str("    const Real *positions_y,\n");
    s.push_str("    const Real *positions_z,\n");
    s.push_str("    const unsigned int *angles,\n");
    s.push_str("    const Real *lattice,\n");
    s.push_str("    Real *angle_triple_x,\n");
    s.push_str("    Real *angle_triple_y,\n");
    s.push_str("    Real *angle_triple_z,\n");
    if write_ev {
        s.push_str("    Real *angle_triple_energy,\n");
        s.push_str("    Real *angle_triple_virial,\n");
    }
    s.push_str(&fragment.entry_point_args);
    s.push_str("    unsigned int n_angles)\n");
    s.push_str("{\n");
    s.push_str(&format!(
        "    {} functor;\n",
        fragment.functor_struct_name
    ));
    s.push_str(&fragment.functor_init_source);
    s.push_str("    Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];\n");
    s.push_str("    Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];\n");
    s.push_str("    unsigned int m = blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str("    if (m >= n_angles) return;\n");
    s.push_str("    unsigned int atom_i = angles[4u * m + 0u];\n");
    s.push_str("    unsigned int atom_j = angles[4u * m + 1u];\n");
    s.push_str("    unsigned int atom_k = angles[4u * m + 2u];\n");
    s.push_str("    unsigned int type_idx = angles[4u * m + 3u];\n");
    s.push_str("    Real dx_ij = positions_x[atom_i] - positions_x[atom_j];\n");
    s.push_str("    Real dy_ij = positions_y[atom_i] - positions_y[atom_j];\n");
    s.push_str("    Real dz_ij = positions_z[atom_i] - positions_z[atom_j];\n");
    s.push_str("    Real dx_kj = positions_x[atom_k] - positions_x[atom_j];\n");
    s.push_str("    Real dy_kj = positions_y[atom_k] - positions_y[atom_j];\n");
    s.push_str("    Real dz_kj = positions_z[atom_k] - positions_z[atom_j];\n");
    s.push_str("    heddle_jit_triclinic_min_image(dx_ij, dy_ij, dz_ij, lx, ly, lz, xy, xz, yz);\n");
    s.push_str("    heddle_jit_triclinic_min_image(dx_kj, dy_kj, dz_kj, lx, ly, lz, xy, xz, yz);\n");
    s.push_str("    Real fix, fiy, fiz, fkx, fky, fkz, u_m, w_m;\n");
    s.push_str("    functor.evaluate(dx_ij, dy_ij, dz_ij, dx_kj, dy_kj, dz_kj, type_idx,\n");
    s.push_str("                     fix, fiy, fiz, fkx, fky, fkz, u_m, w_m);\n");
    s.push_str("    Real fjx = -(fix + fkx);\n");
    s.push_str("    Real fjy = -(fiy + fky);\n");
    s.push_str("    Real fjz = -(fiz + fkz);\n");
    s.push_str("    angle_triple_x[3u * m + 0u] = fix;\n");
    s.push_str("    angle_triple_y[3u * m + 0u] = fiy;\n");
    s.push_str("    angle_triple_z[3u * m + 0u] = fiz;\n");
    s.push_str("    angle_triple_x[3u * m + 1u] = fjx;\n");
    s.push_str("    angle_triple_y[3u * m + 1u] = fjy;\n");
    s.push_str("    angle_triple_z[3u * m + 1u] = fjz;\n");
    s.push_str("    angle_triple_x[3u * m + 2u] = fkx;\n");
    s.push_str("    angle_triple_y[3u * m + 2u] = fky;\n");
    s.push_str("    angle_triple_z[3u * m + 2u] = fkz;\n");
    if write_ev {
        s.push_str("    Real e_share = u_m * (R(1.0) / R(3.0));\n");
        s.push_str("    Real w_share = w_m * (R(1.0) / R(3.0));\n");
        s.push_str("    angle_triple_energy[3u * m + 0u] = e_share;\n");
        s.push_str("    angle_triple_energy[3u * m + 1u] = e_share;\n");
        s.push_str("    angle_triple_energy[3u * m + 2u] = e_share;\n");
        s.push_str("    angle_triple_virial[3u * m + 0u] = w_share;\n");
        s.push_str("    angle_triple_virial[3u * m + 1u] = w_share;\n");
        s.push_str("    angle_triple_virial[3u * m + 2u] = w_share;\n");
    }
    s.push_str("}\n");
}

fn format_angle_compile_failure(
    fragments: &[AngleForceFragment],
    log: &str,
    source: &str,
) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "nvrtc failed to compile the JIT-composed angle kernel."
    );
    let _ = writeln!(s, "Active angle fragments (canonical slot order):");
    for f in fragments {
        let _ = writeln!(s, "  - {} (functor: {})", f.label, f.functor_struct_name);
    }
    let _ = writeln!(s, "nvrtc compile log:");
    let _ = writeln!(s, "{}", log);
    let _ = writeln!(s, "Composed angle source (line-numbered):");
    for (i, line) in source.lines().enumerate() {
        let _ = writeln!(s, "{:5}: {}", i + 1, line);
    }
    s
}

// ============================================================
// Shared compile helper
// ============================================================

fn jit_compile<F>(
    device: &Arc<CudaDevice>,
    source: &str,
    on_fail: F,
) -> Result<cudarc::nvrtc::Ptx, ForceFieldError>
where
    F: FnOnce(&str) -> ForceFieldError,
{
    let arch_arg = detect_arch_option(device);
    let mut options = vec!["--std=c++17".to_string()];
    if let Some(a) = arch_arg {
        options.push(a);
    }
    #[cfg(feature = "f64")]
    options.push("--define-macro=HEDDLE_REAL_F64".to_string());
    let opts = CompileOptions {
        options,
        ..Default::default()
    };
    compile_ptx_with_opts(source, opts).map_err(|e| {
        let log = match e {
            cudarc::nvrtc::CompileError::CompileError { ref log, .. } => log
                .to_str()
                .map(|s| s.to_string())
                .unwrap_or_else(|_| format!("{e:?}")),
            _ => format!("{e:?}"),
        };
        on_fail(&log)
    })
}
