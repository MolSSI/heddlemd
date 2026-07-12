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
use crate::precision::Real;

use super::{ForceFieldError, NeighborListState};

use std::sync::atomic::{AtomicBool, Ordering};

/// Whether JIT-compiled CUDA kernels are built with `--use_fast_math`.
/// Set once at startup from `[simulation].fast_math` (default `true`) and
/// read by every JIT compile site (pair force, bonded, angle, post-force,
/// and the SPME reciprocal). Defaults to `true` so an embedder that never
/// calls `set_jit_fast_math` gets the production default. rq-a84e1c76
static JIT_FAST_MATH: AtomicBool = AtomicBool::new(true);

/// Select whether subsequent JIT compilations use `--use_fast_math`.
/// Called once during startup from the parsed configuration, before any
/// kernel is compiled.
pub fn set_jit_fast_math(enabled: bool) {
    JIT_FAST_MATH.store(enabled, Ordering::Relaxed);
}

/// Whether JIT compilations currently use `--use_fast_math`.
pub fn jit_fast_math_enabled() -> bool {
    JIT_FAST_MATH.load(Ordering::Relaxed)
}

/// Append `--use_fast_math` to a JIT option list when fast-math is on.
/// Shared by every JIT compile site so the flag is applied uniformly.
pub(crate) fn push_jit_fast_math(options: &mut Vec<String>) {
    if jit_fast_math_enabled() {
        options.push("--use_fast_math".to_string());
    }
}

/// Declares whether a pair-force fragment uses a single cutoff for
/// every pair (and what that cutoff is) or a per-pair cutoff. The
/// composer uses this to elide the per-fragment
/// `r² <= cutoff_squared(i, j)` guard when the fragment's cutoff
/// matches the outer max-cutoff mask, and to emit a JIT-compile-time
/// constant guard when the fragment's cutoff is strictly less than
/// the outer max.
///
/// See `rqm/forces/jit-composed-pair-force.md` *Feature API*.
#[derive(Debug, Clone, Copy)]
pub enum CutoffHandling {
    /// Every pair this fragment evaluates uses the same cutoff `c`.
    /// The fragment must implement `cutoff_squared(i, j) == c²` for
    /// every `(i, j)`.
    Uniform(Real),
    /// The fragment's `cutoff_squared(i, j)` may vary per pair; the
    /// composer emits the runtime guard around the fragment's
    /// `evaluate` call.
    PerPair,
}

const MODULE_NAME: &str = "heddle_jit_composed_pair_force";
const F_ENTRY: &str = "heddle_jit_composed_pair_force_f";
const FEV_ENTRY: &str = "heddle_jit_composed_pair_force_fev";
const F_SINGLE_ENTRY: &str = "heddle_jit_composed_pair_force_single_f";
const FEV_SINGLE_ENTRY: &str = "heddle_jit_composed_pair_force_single_fev";
// Exclusion-tile pass entry points (CellList mode only). rq-fa0b3d10
const F_EXCL_ENTRY: &str = "heddle_jit_composed_pair_force_excl_f";
const FEV_EXCL_ENTRY: &str = "heddle_jit_composed_pair_force_excl_fev";

const WARPS_PER_BLOCK: u32 = 8;
const BLOCK_SIZE: u32 = WARPS_PER_BLOCK * 32;
// rq-51209811 rq-3d7e3ff7
// Minimum resident BLOCK_SIZE-thread blocks per SM requested through
// `__launch_bounds__` on the packed-neighbour pass entry points. Caps
// the per-thread register count so the scheduler can keep at least this
// many blocks resident. Value 4 is the spill-free occupancy ceiling on
// SM 8.6: the `_fev` kernel fits in 63 registers (0 spill, 67% theoretical
// occupancy) at this bound, and tightening further (>=5) forces register
// spilling that lowers throughput. The kernel is not occupancy-limited —
// 4 is throughput-neutral versus an unbounded launch and serves as a
// register-creep guard. See `rqm/forces/jit-composed-pair-force.md`
// *Launch Bounds*.
const PACKED_MIN_BLOCKS_PER_SM: u32 = 4;

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
    /// Per-pair cutoff structure. Drives the composer's
    /// cutoff-collapse optimisation (omit the per-fragment guard
    /// when `Uniform(c)` matches the outer max-cutoff mask; emit a
    /// compile-time-constant guard when `Uniform(c)` is strictly
    /// less; emit the runtime guard when `PerPair`).
    pub cutoff: CutoffHandling,
    /// Whether this fragment's `evaluate` / `cutoff_squared` read the
    /// per-atom `i_type` / `j_type` parameters. The composer ORs this
    /// across active fragments: if any sets it, the outer loop emits
    /// the per-atom `type_indices` load and j-side shuffle and passes
    /// the live indices; if none do, the load is elided and
    /// `i_type` / `j_type` are `0`. rq-b10f28d7 rq-61fa8b93
    pub consumes_type_index: bool,
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
/// returned by `DihedralPotential::dihedral_force_fragment()`. Same
/// field shape as `AngleForceFragment`; the functor's contract is the
/// dihedral shape (per-dihedral evaluation taking displacements of
/// `r_ij = r_i − r_j`, `r_kj = r_k − r_j`, and `r_lk = r_l − r_k`).
#[derive(Debug, Clone)]
pub struct DihedralForceFragment {
    pub label: &'static str,
    pub functor_struct_name: &'static str,
    pub functor_source: String,
    pub entry_point_args: String,
    pub functor_init_source: String,
}

/// Dihedral slot's per-launch scratch buffers exposed to the framework
/// for the composed-dihedral-kernel argument list.
pub struct DihedralScratchView<'a> {
    pub dihedrals: &'a CudaSlice<u32>,
    pub dihedral_quadruple_x: &'a CudaSlice<crate::precision::Real>,
    pub dihedral_quadruple_y: &'a CudaSlice<crate::precision::Real>,
    pub dihedral_quadruple_z: &'a CudaSlice<crate::precision::Real>,
    pub dihedral_quadruple_energy: &'a CudaSlice<crate::precision::Real>,
    pub dihedral_quadruple_virial: &'a CudaSlice<crate::precision::Real>,
    pub dihedral_count: usize,
}

/// Argument-builder threaded through every active fast-class slot's
/// bind method (`bind_pair_force_args`, `bind_bonded_force_args`,
/// `bind_angle_force_args`). Pre-populated by the framework with the
/// composed kernel's common arguments; each slot then pushes its
/// parameter buffers and scalars in the order its fragment expects them.
/// Shape-neutral: the binding mechanism is the same across the
/// pair-force, bonded, and angle composers.
pub struct ForceLaunchBuilder {
    /// Owned storage for each argument's bytes. Pointers in
    /// `kernel_params` point into the `Box<[u8]>` heap allocations.
    /// Box ensures the allocation address is stable across pushes onto
    /// the outer Vec.
    storage: Vec<Box<[u8]>>,
    kernel_params: Vec<*mut c_void>,
}

impl Default for ForceLaunchBuilder {
    fn default() -> Self {
        ForceLaunchBuilder {
            storage: Vec::new(),
            kernel_params: Vec::new(),
        }
    }
}

impl ForceLaunchBuilder {
    pub fn new() -> Self {
        ForceLaunchBuilder::default()
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

// ---------------------------------------------------------------------
// Typed pair-force argument schema (prototype — see `cleanup.md` Tier 1).
//
// A fast-class pair-force slot historically hand-maintained THREE
// positionally-coupled pieces that the compiler could not cross-check:
//
//   1. `PairForceFragment::entry_point_args`   — the CUDA `extern "C"`
//      parameter declarations.
//   2. `PairForceFragment::functor_init_source` — the assignments that
//      copy each kernel parameter into the composite functor's field.
//   3. `Potential::bind_pair_force_args`        — the positional pushes
//      of the matching device buffers / scalars at launch time.
//
// A drift in order, count, or type between (1) and (3) is silent: the
// kernel reads the wrong bytes for an argument and produces wrong forces
// (or crashes) only at runtime. The single guard was a `// Order MUST
// match` comment.
//
// `KernelArgSchema` makes one ordered, typed list the single source
// of truth. (1) and (2) are GENERATED from it, so they cannot diverge.
// (3) is routed through `KernelArgBinder`, which VALIDATES every push
// against the schema by name and kind — turning a silent argument-order
// corruption into a located panic at the bind call site.
// ---------------------------------------------------------------------

/// Element type of a kernel pointer parameter / scalar parameter, used
/// to validate that a bound buffer or scalar matches the declared
/// schema entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemTy {
    U32,
    I32,
    Real,
}

/// Compile-time mapping from a Rust buffer/scalar element type to its
/// CUDA `ElemTy`, so `KernelArgBinder::buffer` can check a
/// `CudaSlice<T>` against the schema's declared pointer element type.
pub trait KernelElem {
    const ELEM: ElemTy;
}
impl KernelElem for u32 {
    const ELEM: ElemTy = ElemTy::U32;
}
impl KernelElem for i32 {
    const ELEM: ElemTy = ElemTy::I32;
}
impl KernelElem for Real {
    const ELEM: ElemTy = ElemTy::Real;
}

/// Whether a kernel parameter is a pointer (bound from a `CudaSlice`) or
/// a by-value scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgKind {
    Buffer,
    Scalar,
}

/// The CUDA type of one JIT kernel parameter. Knows how to emit its
/// declaration and which binder push is valid for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelArgType {
    ConstPtrU32,
    ConstPtrI32,
    ConstPtrReal,
    MutPtrReal,
    ScalarU32,
    ScalarReal,
}

impl KernelArgType {
    /// The CUDA `extern "C"` declaration for a parameter of this type
    /// with the given name (e.g. `const Real *lj_type_sigma`).
    pub fn cuda_decl(self, name: &str) -> String {
        let prefix = match self {
            KernelArgType::ConstPtrU32 => "const unsigned int *",
            KernelArgType::ConstPtrI32 => "const int *",
            KernelArgType::ConstPtrReal => "const Real *",
            KernelArgType::MutPtrReal => "Real *",
            KernelArgType::ScalarU32 => "unsigned int ",
            KernelArgType::ScalarReal => "Real ",
        };
        format!("{prefix}{name}")
    }

    pub fn kind(self) -> ArgKind {
        match self {
            KernelArgType::ScalarU32 | KernelArgType::ScalarReal => ArgKind::Scalar,
            _ => ArgKind::Buffer,
        }
    }

    /// Element type for a pointer parameter; `None` for scalars.
    pub fn elem(self) -> Option<ElemTy> {
        match self {
            KernelArgType::ConstPtrU32 => Some(ElemTy::U32),
            KernelArgType::ConstPtrI32 => Some(ElemTy::I32),
            KernelArgType::ConstPtrReal | KernelArgType::MutPtrReal => Some(ElemTy::Real),
            KernelArgType::ScalarU32 | KernelArgType::ScalarReal => None,
        }
    }
}

/// One entry in a slot's pair-force argument schema: the CUDA parameter
/// name, its type, and the composite-functor field it initialises.
#[derive(Debug, Clone)]
pub struct KernelArg {
    /// CUDA kernel parameter name, e.g. `"lj_type_sigma"`.
    pub name: &'static str,
    pub ty: KernelArgType,
    /// Functor struct field this parameter is copied into, e.g.
    /// `"type_sigma"` (assigned as
    /// `composite.<slot-functor>.type_sigma = lj_type_sigma;`).
    pub functor_field: &'static str,
}

impl KernelArg {
    pub const fn new(
        name: &'static str,
        ty: KernelArgType,
        functor_field: &'static str,
    ) -> Self {
        KernelArg { name, ty, functor_field }
    }
}

/// Selects the form `KernelArgSchema::functor_init_source` generates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FunctorInit {
    /// Pair-force: each slot is a member of a shared composite functor.
    /// Init lines read `composite.<composite_member>.<field> = <name>;`.
    CompositeMember,
    /// Bonded/angle: each per-slot entry point declares one local
    /// functor named `functor`. Init lines read
    /// `functor.<field> = <name>;`.
    LocalFunctor,
}

/// The single source of truth for one JIT slot's kernel arguments.
/// Shape-neutral: pair-force, bonded, and angle slots all declare their
/// arguments through it. Generates the fragment's `entry_point_args` and
/// `functor_init_source`, and validates the slot's launch-time binding.
#[derive(Debug, Clone)]
pub struct KernelArgSchema {
    /// Composite-functor member for the owning slot, derived from its
    /// label (e.g. `"functor_lennard_jones"`). Read only by the
    /// `CompositeMember` init style.
    composite_member: String,
    init: FunctorInit,
    args: Vec<KernelArg>,
}

impl KernelArgSchema {
    /// Build a schema for a pair-force slot. `functor_init_source`
    /// generates composite-member assignments
    /// (`composite.functor_<label>.<field> = <name>;`); `label` must
    /// equal the slot's `Potential::label()` so the generated member
    /// name matches the one the composer emits.
    pub fn pair_force(label: &str, args: Vec<KernelArg>) -> Self {
        KernelArgSchema {
            composite_member: functor_field_name(label),
            init: FunctorInit::CompositeMember,
            args,
        }
    }

    /// Build a schema for a bonded or angle slot. `functor_init_source`
    /// generates local-functor assignments (`functor.<field> = <name>;`)
    /// targeting the `functor` local each per-slot entry point declares.
    pub fn intramolecular(label: &str, args: Vec<KernelArg>) -> Self {
        KernelArgSchema {
            composite_member: functor_field_name(label),
            init: FunctorInit::LocalFunctor,
            args,
        }
    }

    /// CUDA parameter declarations for the fragment's `entry_point_args`,
    /// one per line, indented and comma-terminated to match the composer's
    /// concatenation contract. Identical for both constructors.
    pub fn entry_point_args(&self) -> String {
        let mut s = String::new();
        for a in &self.args {
            s.push_str("    ");
            s.push_str(&a.ty.cuda_decl(a.name));
            s.push_str(",\n");
        }
        s
    }

    /// Functor-field initialisation for the fragment's
    /// `functor_init_source`, in the form selected by the constructor.
    pub fn functor_init_source(&self) -> String {
        let mut s = String::new();
        for a in &self.args {
            match self.init {
                FunctorInit::CompositeMember => {
                    s.push_str("    composite.");
                    s.push_str(&self.composite_member);
                    s.push('.');
                }
                FunctorInit::LocalFunctor => {
                    s.push_str("    functor.");
                }
            }
            s.push_str(a.functor_field);
            s.push_str(" = ");
            s.push_str(a.name);
            s.push_str(";\n");
        }
        s
    }
}

/// Schema-checked wrapper over `ForceLaunchBuilder`. Each named push
/// is validated against the slot's `KernelArgSchema` in declaration
/// order: a wrong name, wrong push-kind (buffer vs scalar), wrong
/// element type, or wrong count panics with a located message at bind
/// time, instead of silently corrupting the kernel's argument list.
pub struct KernelArgBinder<'a> {
    schema: &'a KernelArgSchema,
    builder: &'a mut ForceLaunchBuilder,
    slot_label: &'static str,
    cursor: usize,
}

impl<'a> KernelArgBinder<'a> {
    pub fn new(
        schema: &'a KernelArgSchema,
        slot_label: &'static str,
        builder: &'a mut ForceLaunchBuilder,
    ) -> Self {
        KernelArgBinder {
            schema,
            builder,
            slot_label,
            cursor: 0,
        }
    }

    /// Validate the next expected argument against `(name, kind, elem)`
    /// and return its declared type (Copy, so no borrow of `self`
    /// outlives the call).
    fn expect(&self, name: &str, kind: ArgKind, elem: Option<ElemTy>) -> KernelArgType {
        let a = self.schema.args.get(self.cursor).unwrap_or_else(|| {
            panic!(
                "slot `{}`: binding pushed more arguments than the schema declares \
                 ({} declared); extra argument `{}`",
                self.slot_label,
                self.schema.args.len(),
                name,
            )
        });
        if a.name != name {
            panic!(
                "slot `{}` argument #{}: schema declares `{}` but binding pushed `{}` \
                 — order/name drift between the fragment signature and the binding",
                self.slot_label, self.cursor, a.name, name,
            );
        }
        if a.ty.kind() != kind {
            panic!(
                "slot `{}` argument `{}`: schema declares a {:?} parameter but binding \
                 pushed a {:?}",
                self.slot_label, name, a.ty.kind(), kind,
            );
        }
        if kind == ArgKind::Buffer && a.ty.elem() != elem {
            panic!(
                "slot `{}` argument `{}`: schema declares `{}` (element {:?}) but binding \
                 pushed a buffer of element {:?}",
                self.slot_label,
                name,
                a.ty.cuda_decl(name),
                a.ty.elem(),
                elem,
            );
        }
        a.ty
    }

    /// Push a device buffer for the named pointer parameter.
    pub fn buffer<T: KernelElem>(&mut self, name: &'static str, buf: &CudaSlice<T>) {
        self.expect(name, ArgKind::Buffer, Some(T::ELEM));
        self.builder.push_device_buffer(buf);
        self.cursor += 1;
    }

    /// Push a `u32` scalar for the named `unsigned int` parameter.
    pub fn scalar_u32(&mut self, name: &'static str, value: u32) {
        let ty = self.expect(name, ArgKind::Scalar, None);
        assert_eq!(
            ty,
            KernelArgType::ScalarU32,
            "slot `{}` argument `{}`: scalar_u32 push does not match declared {:?}",
            self.slot_label, name, ty,
        );
        self.builder.push_scalar(value);
        self.cursor += 1;
    }

    /// Push a `Real` scalar for the named `Real` parameter.
    pub fn scalar_real(&mut self, name: &'static str, value: Real) {
        let ty = self.expect(name, ArgKind::Scalar, None);
        assert_eq!(
            ty,
            KernelArgType::ScalarReal,
            "slot `{}` argument `{}`: scalar_real push does not match declared {:?}",
            self.slot_label, name, ty,
        );
        self.builder.push_scalar(value);
        self.cursor += 1;
    }

    /// Assert every declared argument was bound exactly once.
    pub fn finish(self) {
        if self.cursor != self.schema.args.len() {
            panic!(
                "slot `{}`: binding pushed {} arguments but the schema declares {}",
                self.slot_label,
                self.cursor,
                self.schema.args.len(),
            );
        }
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
    /// Per-pair single-pair entry point (`AggregateLevel::ForcesOnly`).
    /// Launched after the main pair-force kernel when the neighbour
    /// list's `single_pairs_count` is non-zero; each thread handles
    /// one sparse-tile-extracted pair and contributes the fragment's
    /// scale-multiplied per-pair contribution to both atoms'
    /// fixed-point slots (Newton's 3rd via `±`).
    pub single_pair_f: CudaFunction,
    /// `AggregateLevel::ForcesAndScalars` variant of `single_pair_f`.
    pub single_pair_fev: CudaFunction,
    /// Exclusion-tile pass entry points, present only when the kernel was
    /// composed with the per-tile exclusion bitmask (CellList mode). In
    /// all-pairs mode exclusions fold into the evaluator's per-pair scale
    /// and there is no separate pass. rq-fa0b3d10
    pub excl_tile_f: Option<CudaFunction>,
    pub excl_tile_fev: Option<CudaFunction>,
}

impl JitComposedPairForce {
    /// Compose, compile, and load the composed kernel from the active
    /// fragments. `fragments` is the active fast-class pair-force
    /// fragment list in canonical slot order.
    pub fn compile_and_load(
        device: &Arc<CudaDevice>,
        fragments: &[PairForceFragment],
        max_cutoff: crate::precision::Real,
        use_exclusion_bitmask: bool,
    ) -> Result<Self, ForceFieldError> {
        let source = compose_source(fragments, max_cutoff, use_exclusion_bitmask);

        let arch_arg = detect_arch_option(device);
        let mut options = vec!["--std=c++17".to_string()];
        if let Some(a) = arch_arg {
            options.push(a);
        }
        #[cfg(feature = "f64")]
        options.push("--define-macro=HEDDLE_REAL_F64".to_string());
        push_jit_fast_math(&mut options);
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

        let mut entry_points = vec![F_ENTRY, FEV_ENTRY, F_SINGLE_ENTRY, FEV_SINGLE_ENTRY];
        if use_exclusion_bitmask {
            entry_points.push(F_EXCL_ENTRY);
            entry_points.push(FEV_EXCL_ENTRY);
        }
        device
            .load_ptx(ptx, MODULE_NAME, &entry_points)
            .map_err(|e| ForceFieldError::FragmentLoadFailed(GpuError::from(e)))?;
        let pair_force_f = device
            .get_func(MODULE_NAME, F_ENTRY)
            .expect("composed pair-force kernel _f entry was just loaded");
        let pair_force_fev = device
            .get_func(MODULE_NAME, FEV_ENTRY)
            .expect("composed pair-force kernel _fev entry was just loaded");
        let single_pair_f = device
            .get_func(MODULE_NAME, F_SINGLE_ENTRY)
            .expect("composed pair-force single-pair _f entry was just loaded");
        let single_pair_fev = device
            .get_func(MODULE_NAME, FEV_SINGLE_ENTRY)
            .expect("composed pair-force single-pair _fev entry was just loaded");
        let (excl_tile_f, excl_tile_fev) = if use_exclusion_bitmask {
            (
                Some(
                    device
                        .get_func(MODULE_NAME, F_EXCL_ENTRY)
                        .expect("composed pair-force excl-tile _f entry was just loaded"),
                ),
                Some(
                    device
                        .get_func(MODULE_NAME, FEV_EXCL_ENTRY)
                        .expect("composed pair-force excl-tile _fev entry was just loaded"),
                ),
            )
        } else {
            (None, None)
        };

        Ok(JitComposedPairForce {
            fragment_labels: fragments.iter().map(|f| f.label).collect(),
            pair_force_f,
            pair_force_fev,
            single_pair_f,
            single_pair_fev,
            excl_tile_f,
            excl_tile_fev,
        })
    }

    /// Launch the composed pair-force kernel over the interacting
    /// tiles list. `interacting_tiles_count` is the number of entries
    /// (one warp per entry). `use_fev` selects between the `_f` and
    /// `_fev` entry points. `builder` must have been pre-populated
    /// with the common args (including `block_centre` and
    /// `block_bbox`), per-fragment args (in canonical slot order),
    /// and the trailing `n` arg.
    ///
    /// # Safety
    /// `builder`'s argument list must match the composed kernel's
    /// entry-point signature exactly.
    pub unsafe fn launch(
        &self,
        n_iblocks: u32,
        use_fev: bool,
        mut builder: ForceLaunchBuilder,
    ) -> Result<(), GpuError> {
        if n_iblocks == 0 {
            drop(builder.storage);
            return Ok(());
        }
        let cfg = LaunchConfig {
            grid_dim: (n_iblocks, 1, 1),
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

    /// Launch the per-pair single-pair kernel. The grid is sized to
    /// `single_pairs_capacity` so the captured kernel covers any
    /// post-rebuild count; each thread reads the live count from
    /// device memory (via the `interaction_count` pointer in the
    /// builder) and returns early past the live boundary. `builder`
    /// must be pre-populated with the single-pair kernel's common
    /// args (positions, single_pair_atoms, interaction_count_ptr,
    /// lattice, fixed-point accumulators), the per-fragment args in
    /// canonical slot order, and the trailing `n` arg.
    ///
    /// # Safety
    /// `builder`'s argument list must match the single-pair kernel's
    /// entry-point signature exactly.
    pub unsafe fn launch_single_pair(
        &self,
        single_pairs_capacity: u32,
        use_fev: bool,
        mut builder: ForceLaunchBuilder,
    ) -> Result<(), GpuError> {
        if single_pairs_capacity == 0 {
            drop(builder.storage);
            return Ok(());
        }
        let block_size: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: (single_pairs_capacity.div_ceil(block_size), 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = if use_fev {
            self.single_pair_fev.clone()
        } else {
            self.single_pair_f.clone()
        };
        unsafe {
            func.launch(cfg, &mut builder.kernel_params)
                .map_err(GpuError::from)?;
        }
        drop(builder.storage);
        Ok(())
    }

    /// Launch the exclusion-tile pass (one warp per tile). The grid is
    /// sized to `exclusion_tiles_capacity` warps so the captured kernel
    /// covers any post-rebuild tile count; each warp reads the live count
    /// from `interaction_count[2]` (via the pointer in the builder) and
    /// returns early past the live boundary. `builder` must be
    /// pre-populated with the exclusion-tile common args (exclusion-tile
    /// buffers, `interaction_count`, `tile_sorted_posq`,
    /// `sorted_particle_ids`, `type_indices`, `lattice`, and the
    /// fixed-point accumulators), the per-fragment args in canonical slot
    /// order, and the trailing `n`.
    ///
    /// # Safety
    /// `builder`'s argument list must match the exclusion-tile entry
    /// point's signature exactly. rq-fa0b3d10
    pub unsafe fn launch_excl_tile(
        &self,
        exclusion_tiles_capacity: u32,
        use_fev: bool,
        mut builder: ForceLaunchBuilder,
    ) -> Result<(), GpuError> {
        let func = if use_fev {
            self.excl_tile_fev.as_ref()
        } else {
            self.excl_tile_f.as_ref()
        };
        let func = match func {
            Some(f) if exclusion_tiles_capacity > 0 => f.clone(),
            // No bitmask pass compiled (all-pairs mode) or empty capacity:
            // nothing to launch.
            _ => {
                drop(builder.storage);
                return Ok(());
            }
        };
        // 8 warps per block (256 threads); one warp per tile.
        const WARPS_PER_BLOCK: u32 = 8;
        let block_size: u32 = WARPS_PER_BLOCK * 32;
        let cfg = LaunchConfig {
            grid_dim: (exclusion_tiles_capacity.div_ceil(WARPS_PER_BLOCK), 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.launch(cfg, &mut builder.kernel_params)
                .map_err(GpuError::from)?;
        }
        drop(builder.storage);
        Ok(())
    }
}

pub(crate) fn detect_arch_option(device: &Arc<CudaDevice>) -> Option<String> {
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

/// Emit a `template <bool WriteEv>` per-pair evaluator named `fn_name`
/// that sums each fragment's `(factor, energy)` at pair (i, j). When
/// `apply_scale` is true, each fragment's contribution is multiplied by
/// its per-pair `exclusion_scale(i, j)` (fully-excluded pairs contribute
/// zero, 1-4 pairs contribute a scaled amount); when false, the
/// contribution is added at full strength with no exclusion lookup.
/// Per-fragment cutoff guards are emitted per each fragment's
/// `CutoffHandling`; the outer-loop max-cutoff mask is applied by the
/// caller. rq-7d64da58 rq-a4b9e702 rq-fa0b3d10
fn emit_eval_pair_sum(
    s: &mut String,
    fragments: &[PairForceFragment],
    max_cutoff: Real,
    fn_name: &str,
    apply_scale: bool,
) {
    s.push_str("\ntemplate <bool WriteEv>\n");
    s.push_str(&format!("__device__ static inline void {fn_name}(\n"));
    s.push_str("    const HeddleJitComposedPairFunc &composite,\n");
    s.push_str(
        "    Real r2, Real inv_r, Real r, Real qi, Real qj, unsigned int i_type, unsigned int j_type, unsigned int i, unsigned int j,\n",
    );
    s.push_str("    Real &factor, Real &energy)\n");
    s.push_str("{\n");
    s.push_str("    factor = R(0.0); energy = R(0.0);\n");
    for f in fragments {
        let field = functor_field_name(f.label);
        let body = if apply_scale {
            format!(
                "Real s_factor, s_energy;\n            \
                 composite.{f}.evaluate(r2, inv_r, r, qi, qj, i_type, j_type, i, j, s_factor, s_energy);\n            \
                 Real ex_scale = composite.{f}.exclusion_scale(i, j);\n            \
                 factor += s_factor * ex_scale;\n            \
                 if (WriteEv) {{ energy += s_energy * ex_scale; }}",
                f = field
            )
        } else {
            format!(
                "Real s_factor, s_energy;\n            \
                 composite.{f}.evaluate(r2, inv_r, r, qi, qj, i_type, j_type, i, j, s_factor, s_energy);\n            \
                 factor += s_factor;\n            \
                 if (WriteEv) {{ energy += s_energy; }}",
                f = field
            )
        };
        match f.cutoff {
            // Uniform cutoff matching the outer max: the outer mask
            // already covers it; omit the per-fragment guard.
            CutoffHandling::Uniform(c) if c == max_cutoff => {
                s.push_str(&format!("    {{\n        {body}\n    }}\n", body = body));
            }
            // Uniform cutoff strictly less than the outer max: emit a
            // compile-time-constant guard against c² (no per-pair load
            // of `cutoff_squared(i, j)`).
            CutoffHandling::Uniform(c) => {
                let c_sq = (c as f64) * (c as f64);
                s.push_str(&format!(
                    "    {{\n        if (r2 <= R({c_sq:.17e})) {{\n            \
                     {body}\n        }}\n    }}\n",
                    c_sq = c_sq,
                    body = body,
                ));
            }
            // Per-pair cutoff: emit the runtime guard around the
            // fragment's evaluate.
            CutoffHandling::PerPair => {
                s.push_str(&format!(
                    "    {{\n        Real cut2 = composite.{f}.cutoff_squared(i_type, j_type, i, j);\n        \
                     if (r2 <= cut2) {{\n            {body}\n        }}\n    }}\n",
                    f = field,
                    body = body,
                ));
            }
        }
    }
    s.push_str("}\n");
}

fn compose_source(
    fragments: &[PairForceFragment],
    max_cutoff: Real,
    use_exclusion_bitmask: bool,
) -> String {
    let mut s = String::with_capacity(
        8192 + fragments.iter().map(|f| f.functor_source.len()).sum::<usize>(),
    );
    s.push_str(PREAMBLE);
    // Per-pair early-exit threshold. The composer embeds the maximum
    // squared cutoff across all active fast-class pair-force slots as
    // a `#define` constant in the JIT source. The outer loop applies
    // this as a branchless mask: pair math runs unconditionally and
    // the mask zeroes contributions for pairs past this threshold.
    let max_cutoff_squared = (max_cutoff as f64) * (max_cutoff as f64);
    s.push_str(&format!(
        "\n#define HEDDLE_JIT_MAX_CUTOFF R({:.17e})\n\n",
        max_cutoff as f64
    ));
    s.push_str(&format!(
        "\n#define HEDDLE_JIT_MAX_CUTOFF_SQUARED R({:.17e})\n\n",
        max_cutoff_squared
    ));
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

    // Per-pair functor sum: sums each active slot's `(factor, energy)`
    // at pair (i, j). Where a fragment's per-pair `exclusion_scale(i, j)`
    // is applied depends on the neighbour-list mode (see
    // `packed-neighbour-pair-force.md` and `jit-composed-pair-force.md`
    // *Exclusion Handling*):
    //
    // - CellList mode emits two evaluators: `heddle_jit_eval_pair_sum`
    //   (scale-free) for the bulk and single-pair passes, which never see
    //   a modified pair, and `heddle_jit_eval_pair_sum_scaled`, used by
    //   the exclusion-tile pass on its flagged (modified) pairs.
    // - All-pairs mode emits one scale-aware `heddle_jit_eval_pair_sum`
    //   applied to every pair.
    //
    // rq-7d64da58 — the functor emits only (factor, energy); the per-pair
    // scalar virial is derived by the caller as factor * r2.
    // rq-a4b9e702 rq-b28a6d96 rq-fa0b3d10 rq-8ae4a9f1
    if use_exclusion_bitmask {
        emit_eval_pair_sum(&mut s, fragments, max_cutoff, "heddle_jit_eval_pair_sum", false);
        emit_eval_pair_sum(
            &mut s,
            fragments,
            max_cutoff,
            "heddle_jit_eval_pair_sum_scaled",
            true,
        );
    } else {
        emit_eval_pair_sum(&mut s, fragments, max_cutoff, "heddle_jit_eval_pair_sum", true);
    }

    let _ = max_cutoff;

    s.push_str(OUTER_LOOP_TEMPLATE);
    s.push_str(SINGLE_PAIR_LOOP_TEMPLATE);
    if use_exclusion_bitmask {
        s.push_str(EXCL_TILE_LOOP_TEMPLATE);
    }

    // _f / _fev entry points. Each evaluates the single-periodic-copy
    // eligibility once per i-block at runtime and branches inside the
    // outer loop; the branch is uniform across the warp, so there is
    // no per-pair warp divergence. See
    // `rqm/forces/packed-neighbour-pair-force.md` *Single-Periodic-
    // Copy Fast Path*.
    emit_entry_point(&mut s, fragments, F_ENTRY, false);
    emit_entry_point(&mut s, fragments, FEV_ENTRY, true);
    // Per-pair single-pair entry points
    emit_single_pair_entry_point(&mut s, fragments, F_SINGLE_ENTRY, false);
    emit_single_pair_entry_point(&mut s, fragments, FEV_SINGLE_ENTRY, true);
    // Exclusion-tile pass entry points (CellList mode only). rq-fa0b3d10
    if use_exclusion_bitmask {
        emit_excl_tile_entry_point(&mut s, fragments, F_EXCL_ENTRY, false);
        emit_excl_tile_entry_point(&mut s, fragments, FEV_EXCL_ENTRY, true);
    }

    // Resolve the per-atom type-index load markers left in the loop
    // templates. The per-atom `type_indices` buffer is always a common
    // kernel argument, but the dereference is only emitted when at
    // least one active fragment consumes the index — otherwise the
    // markers expand to nothing and `i_type` / `j_type` stay 0.
    // rq-b10f28d7 rq-61fa8b93
    let any_consumes_type_index = fragments.iter().any(|f| f.consumes_type_index);
    let (itype_load, jtype_load, jtype_shuffle, perpair_load) = if any_consumes_type_index {
        (
            "i_type = i_valid ? type_indices[i_atom_id] : 0u;",
            "j_type = j_valid ? type_indices[j_atom_id] : 0u;",
            "j_type = __shfl_sync(0xFFFFFFFFu, j_type, src_lane);",
            "i_type = type_indices[atom_i]; j_type = type_indices[atom_j];",
        )
    } else {
        ("", "", "", "")
    };
    let s = s
        .replace("/*HEDDLE_JIT_ITYPE_LOAD*/", itype_load)
        .replace("/*HEDDLE_JIT_JTYPE_LOAD*/", jtype_load)
        .replace("/*HEDDLE_JIT_JTYPE_SHUFFLE*/", jtype_shuffle)
        .replace("/*HEDDLE_JIT_TYPE_LOAD_PERPAIR*/", perpair_load);

    s
}

/// Emit the per-pair single-pair entry point. Common args take the
/// `single_pair_atoms` / `single_pair_count` pair (in place of the
/// packed-neighbour list inputs the main entry point uses) and
/// dispatch `heddle_jit_single_pair_loop`.
fn emit_single_pair_entry_point(
    s: &mut String,
    fragments: &[PairForceFragment],
    entry_name: &str,
    write_ev: bool,
) {
    s.push_str("\nextern \"C\" __global__ void ");
    s.push_str(entry_name);
    s.push_str("(\n");
    s.push_str("    const Real4 *posq,\n");
    s.push_str("    const unsigned int *type_indices,\n");
    s.push_str("    const unsigned int *single_pair_atoms,\n");
    s.push_str("    const unsigned int *interaction_count_ptr,\n");
    s.push_str("    const Real *lattice,\n");
    s.push_str("    unsigned long long *fast_force_x_fp,\n");
    s.push_str("    unsigned long long *fast_force_y_fp,\n");
    s.push_str("    unsigned long long *fast_force_z_fp,\n");
    s.push_str("    unsigned long long *fast_energy_fp,\n");
    s.push_str("    unsigned long long *fast_virial_fp,\n");
    for f in fragments {
        s.push_str(&f.entry_point_args);
    }
    s.push_str("    unsigned int n)\n");
    s.push_str("{\n");
    s.push_str("    HeddleJitComposedPairFunc composite;\n");
    for f in fragments {
        s.push_str(&f.functor_init_source);
    }
    s.push_str("    heddle_jit_single_pair_loop<");
    s.push_str(if write_ev { "true" } else { "false" });
    s.push_str(">(\n");
    s.push_str("        composite, single_pair_atoms, interaction_count_ptr,\n");
    s.push_str("        posq,\n");
    s.push_str("        type_indices,\n");
    s.push_str("        lattice,\n");
    s.push_str("        fast_force_x_fp, fast_force_y_fp, fast_force_z_fp,\n");
    s.push_str("        fast_energy_fp, fast_virial_fp,\n");
    s.push_str("        n);\n");
    s.push_str("}\n");
}

fn emit_entry_point(
    s: &mut String,
    fragments: &[PairForceFragment],
    entry_name: &str,
    write_ev: bool,
) {
    // rq-3d7e3ff7 — `__launch_bounds__` caps registers on the
    // packed-neighbour pass so the SM keeps at least
    // PACKED_MIN_BLOCKS_PER_SM resident blocks, raising the
    // latency-hiding warp count. The single-pair pass (one thread
    // per pair, not occupancy-limited) carries no such bound.
    s.push_str("\nextern \"C\" __global__ void __launch_bounds__(");
    s.push_str(&BLOCK_SIZE.to_string());
    s.push_str(", ");
    s.push_str(&PACKED_MIN_BLOCKS_PER_SM.to_string());
    s.push_str(") ");
    s.push_str(entry_name);
    s.push_str("(\n");
    s.push_str("    const Real4 *posq,\n");
    s.push_str("    const unsigned int *type_indices,\n");
    s.push_str("    const Real4 *tile_sorted_posq,\n");
    s.push_str("    const Real *block_centre,\n");
    s.push_str("    const Real *block_bbox,\n");
    s.push_str("    const unsigned int *sorted_particle_ids,\n");
    s.push_str("    const unsigned int *iblock_offset,\n");
    s.push_str("    const unsigned int *sorted_interacting_atoms,\n");
    s.push_str("    unsigned int n_iblocks,\n");
    s.push_str("    const Real *lattice,\n");
    s.push_str("    unsigned long long *fast_force_x_fp,\n");
    s.push_str("    unsigned long long *fast_force_y_fp,\n");
    s.push_str("    unsigned long long *fast_force_z_fp,\n");
    s.push_str("    unsigned long long *fast_energy_fp,\n");
    s.push_str("    unsigned long long *fast_virial_fp,\n");
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
    s.push_str("        composite, iblock_offset, n_iblocks,\n");
    s.push_str("        posq,\n");
    s.push_str("        type_indices,\n");
    s.push_str("        tile_sorted_posq,\n");
    s.push_str("        block_centre,\n");
    s.push_str("        block_bbox,\n");
    s.push_str("        sorted_particle_ids,\n");
    s.push_str("        sorted_interacting_atoms,\n");
    s.push_str("        lattice,\n");
    s.push_str(
        "        fast_force_x_fp, fast_force_y_fp, fast_force_z_fp,\n",
    );
    s.push_str("        fast_energy_fp, fast_virial_fp,\n");
    s.push_str("        n);\n");
    s.push_str("}\n");
}

// rq-fa0b3d10 — exclusion-tile pass entry point. Common args are the
// exclusion-tile buffers + interaction_count (for the live tile count),
// the tile-sorted positions, sorted ids, type indices, lattice, and the
// fixed-point accumulators; per-fragment args follow in canonical slot
// order (the functor still needs its parameter buffers even though no
// exclusion scale is applied here). One warp per tile.
fn emit_excl_tile_entry_point(
    s: &mut String,
    fragments: &[PairForceFragment],
    entry_name: &str,
    write_ev: bool,
) {
    s.push_str("\nextern \"C\" __global__ void ");
    s.push_str(entry_name);
    s.push_str("(\n");
    s.push_str("    const unsigned int *exclusion_tile_iblocks,\n");
    s.push_str("    const unsigned int *exclusion_tile_jblocks,\n");
    s.push_str("    const unsigned int *exclusion_tile_masks,\n");
    s.push_str("    const unsigned int *interaction_count,\n");
    s.push_str("    const Real4 *tile_sorted_posq,\n");
    s.push_str("    const unsigned int *sorted_particle_ids,\n");
    s.push_str("    const unsigned int *type_indices,\n");
    s.push_str("    const Real *lattice,\n");
    s.push_str("    unsigned long long *fast_force_x_fp,\n");
    s.push_str("    unsigned long long *fast_force_y_fp,\n");
    s.push_str("    unsigned long long *fast_force_z_fp,\n");
    s.push_str("    unsigned long long *fast_energy_fp,\n");
    s.push_str("    unsigned long long *fast_virial_fp,\n");
    for f in fragments {
        s.push_str(&f.entry_point_args);
    }
    s.push_str("    unsigned int n)\n");
    s.push_str("{\n");
    s.push_str("    HeddleJitComposedPairFunc composite;\n");
    for f in fragments {
        s.push_str(&f.functor_init_source);
    }
    s.push_str("    heddle_jit_excl_tile_loop<");
    s.push_str(if write_ev { "true" } else { "false" });
    s.push_str(">(\n");
    s.push_str("        composite, exclusion_tile_iblocks, exclusion_tile_jblocks,\n");
    s.push_str("        exclusion_tile_masks, interaction_count,\n");
    s.push_str("        tile_sorted_posq,\n");
    s.push_str("        sorted_particle_ids,\n");
    s.push_str("        type_indices,\n");
    s.push_str("        lattice,\n");
    s.push_str("        fast_force_x_fp, fast_force_y_fp, fast_force_z_fp,\n");
    s.push_str("        fast_energy_fp, fast_virial_fp,\n");
    s.push_str("        n);\n");
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
typedef double4 Real4;
#define R(x) ((Real)(x))
__device__ __forceinline__ Real Real_sqrt(Real x) { return sqrt(x); }
__device__ __forceinline__ Real Real_rsqrt(Real x) { return rsqrt(x); }
__device__ __forceinline__ Real Real_exp(Real x) { return exp(x); }
__device__ __forceinline__ Real Real_log(Real x) { return log(x); }
__device__ __forceinline__ Real Real_floor(Real x) { return floor(x); }
__device__ __forceinline__ Real Real_fma(Real a, Real b, Real c) { return fma(a, b, c); }
__device__ __forceinline__ Real Real_erfc(Real x) { return erfc(x); }
__device__ __forceinline__ Real Real_atan2(Real y, Real x) { return atan2(y, x); }
__device__ __forceinline__ Real Real_sin(Real x) { return sin(x); }
__device__ __forceinline__ Real Real_cos(Real x) { return cos(x); }
#else
typedef float Real;
typedef float4 Real4;
#define R(x) ((Real)(x))
__device__ __forceinline__ Real Real_sqrt(Real x) { return sqrtf(x); }
__device__ __forceinline__ Real Real_rsqrt(Real x) { return rsqrtf(x); }
__device__ __forceinline__ Real Real_exp(Real x) { return expf(x); }
__device__ __forceinline__ Real Real_log(Real x) { return logf(x); }
__device__ __forceinline__ Real Real_floor(Real x) { return floorf(x); }
__device__ __forceinline__ Real Real_fma(Real a, Real b, Real c) { return fmaf(a, b, c); }
__device__ __forceinline__ Real Real_erfc(Real x) { return erfcf(x); }
__device__ __forceinline__ Real Real_atan2(Real y, Real x) { return atan2f(y, x); }
__device__ __forceinline__ Real Real_sin(Real x) { return sinf(x); }
__device__ __forceinline__ Real Real_cos(Real x) { return cosf(x); }
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

// In-place shift of (x, y, z) to the periodic image closest to
// (cx, cy, cz). Mirror of `triclinic_wrap_against_center` in
// `kernels/pbc.cuh`; redeclared in the JIT preamble because the JIT
// translation unit does not include the project headers. Used by the
// SPC fast-path entry points to wrap pi and pj against the i-block
// centre once outside the inner loop, so the per-pair dx = pi - pj
// is the canonical min-image displacement without further wrapping.
__device__ static inline void triclinic_wrap_against_center(
    Real &x, Real &y, Real &z,
    Real cx, Real cy, Real cz,
    Real lx, Real ly, Real lz,
    Real xy, Real xz, Real yz)
{
  Real dx = x - cx;
  Real dy = y - cy;
  Real dz = z - cz;
  Real s_a, s_b, s_c;
  heddle_jit_triclinic_cart_to_frac(dx, dy, dz, lx, ly, lz, xy, xz, yz, s_a, s_b, s_c);
  Real ka = Real_floor(s_a + R(0.5));
  Real kb = Real_floor(s_b + R(0.5));
  Real kc = Real_floor(s_c + R(0.5));
  x -= ka * lx + kb * xy + kc * xz;
  y -= kb * ly + kc * yz;
  z -= kc * lz;
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

// Fixed-point conversion for atomic force/energy/virial accumulation.
// Integer addition is associative regardless of arrival order, so the
// per-atom sum is bit-exact across runs. Scale 2^48 gives ~3.6e-15
// precision in atomic units — adequate for SD convergence to typical
// 1e-10 Ha/Bohr force tolerances and well below f32's quantization
// for typical MD value ranges. Max representable: ~2^15 in atomic
// units, large enough for any reasonable per-atom force.
__device__ static inline long long heddle_jit_real_to_fixed(Real f) {
  // Multiply by 2^24 twice to apply scale 2^48 without overflowing
  // the f32 intermediate for moderately-sized inputs.
  Real scaled = f * (Real) (1u << 24);
  scaled *= (Real) (1u << 24);
  return (long long) scaled;
}

// AtomicAdd in fixed-point. `buf` is the per-atom fixed-point buffer
// reinterpreted as `unsigned long long`. The 64-bit atomic preserves
// the two's-complement integer interpretation.
__device__ static inline void heddle_jit_atomic_add_fp(
    unsigned long long *buf, unsigned int atom, Real f)
{
  if (f != R(0.0)) {
    unsigned long long delta = (unsigned long long) heddle_jit_real_to_fixed(f);
    atomicAdd(&buf[atom], delta);
  }
}
"#;

const OUTER_LOOP_TEMPLATE: &str = r#"
// Packed-neighbour pair-force outer loop. One warp per
// interacting_tiles entry. Each lane owns one i-atom of the entry's
// i-block; j-atoms come from interacting_atoms[pos*32 + lane], one
// individual j-atom ID per lane (pre-filtered real neighbours from
// possibly different j-blocks).
//
// Inner loop runs 32 lock-step iterations with a diagonal shuffle:
// at iteration t, lane k pairs with j_lane (k + t) mod 32 via warp
// shuffles of the j-side state. Each lane accumulates the force on
// BOTH its i-atom and the current j-atom in per-lane registers
// (Newton's 3rd). At the end the per-lane (i_*, j_*) accumulators
// are atomicAdded — in fixed-point — to the per-class accumulator
// buffer.
//
// I-block-cooperative layout: one threadblock per i-block. Eight warps
// (HEDDLE_JIT_WARPS_PER_BLOCK) share the same 32 i-atoms and stride
// across that i-block's entries via warp_id. Each warp accumulates its
// slice's i-side contributions in registers across every entry it
// touches, then atomic-adds into a shared-mem fixed-point accumulator
// once at the end. A single warp finally atomic-flushes the shared
// accumulator to the global per-atom slots. Net effect: i-side global
// atomics drop from one per (entry, lane) to one per (i-atom).
// Determinism is preserved because the shared and global accumulators
// are i64 fixed-point — integer addition is associative regardless of
// the order in which warps and blocks contribute.
template <bool WriteEv>
__device__ static inline void heddle_jit_outer_loop(
    const HeddleJitComposedPairFunc &composite,
    const unsigned int *iblock_offset,
    unsigned int n_iblocks,
    const Real4 *posq,
    const unsigned int *type_indices,
    const Real4 *tile_sorted_posq,
    const Real *block_centre,
    const Real *block_bbox,
    const unsigned int *sorted_particle_ids,
    const unsigned int *sorted_interacting_atoms,
    const Real *lattice,
    unsigned long long *fast_force_x_fp,
    unsigned long long *fast_force_y_fp,
    unsigned long long *fast_force_z_fp,
    unsigned long long *fast_energy_fp,
    unsigned long long *fast_virial_fp,
    unsigned int n)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];

  unsigned int i_block = blockIdx.x;
  if (i_block >= n_iblocks) return;
  unsigned int warp_id_in_block = threadIdx.x / HEDDLE_JIT_WARP_SIZE;
  unsigned int lane = threadIdx.x & (HEDDLE_JIT_WARP_SIZE - 1u);

  unsigned int range_begin = iblock_offset[i_block];
  unsigned int range_end = iblock_offset[i_block + 1];

  __shared__ unsigned long long shared_fx[HEDDLE_JIT_WARP_SIZE];
  __shared__ unsigned long long shared_fy[HEDDLE_JIT_WARP_SIZE];
  __shared__ unsigned long long shared_fz[HEDDLE_JIT_WARP_SIZE];
  __shared__ unsigned long long shared_e[HEDDLE_JIT_WARP_SIZE];
  __shared__ unsigned long long shared_w[HEDDLE_JIT_WARP_SIZE];

  // Initialize shared accumulators (one slot per i-atom in the block).
  if (warp_id_in_block == 0u) {
    shared_fx[lane] = 0ull;
    shared_fy[lane] = 0ull;
    shared_fz[lane] = 0ull;
    if (WriteEv) {
      shared_e[lane] = 0ull;
      shared_w[lane] = 0ull;
    }
  }

  // Single-periodic-copy fast-path eligibility, decided per-block at
  // runtime from the block's own bounding box. The check is
  //   0.5 * L_axis - bbox_axis >= MAX_CUTOFF
  // for every axis. When this holds, the periodic image of every
  // in-cutoff j-atom relative to the i-block centre is the same as
  // its min-image relative to any i-atom in the block — so wrapping
  // both pi and pj against the centre once outside the inner loop
  // makes the per-pair `dx = pi - pj` already the canonical min-image
  // displacement. The branch is uniform across the warp (all 32 lanes
  // compute the same predicate from i_block, the lattice constants
  // and `MAX_CUTOFF`), so there is no per-pair warp divergence.
  // Triclinic boxes (any of xy, xz, yz non-zero) are conservatively
  // ineligible; they fall through to the per-pair min-image branch.
  Real bbox_x = block_bbox[i_block * 3u + 0u];
  Real bbox_y = block_bbox[i_block * 3u + 1u];
  Real bbox_z = block_bbox[i_block * 3u + 2u];
  bool orthorhombic = (xy == R(0.0)) && (xz == R(0.0)) && (yz == R(0.0));
  bool spc = orthorhombic
          && (R(0.5) * lx - bbox_x >= HEDDLE_JIT_MAX_CUTOFF)
          && (R(0.5) * ly - bbox_y >= HEDDLE_JIT_MAX_CUTOFF)
          && (R(0.5) * lz - bbox_z >= HEDDLE_JIT_MAX_CUTOFF);

  Real cx = R(0.0), cy = R(0.0), cz = R(0.0);
  if (spc) {
    cx = block_centre[i_block * 4u + 0u];
    cy = block_centre[i_block * 4u + 1u];
    cz = block_centre[i_block * 4u + 2u];
  }

  // Each lane owns one i-atom of i_block. Load its original atom ID
  // and position from the tile-sorted view (coalesced). Lanes past
  // n_atoms are inactive sentinels — gated by `i_valid`.
  unsigned int i_slot = i_block * 32u + lane;
  bool i_valid = i_slot < n;
  unsigned int i_atom_id = i_valid ? sorted_particle_ids[i_slot] : n;
  Real4 pq_i_load = tile_sorted_posq[i_slot];
  Real pi_x = pq_i_load.x;
  Real pi_y = pq_i_load.y;
  Real pi_z = pq_i_load.z;
  if (spc && i_valid) {
    triclinic_wrap_against_center(pi_x, pi_y, pi_z,
                                  cx, cy, cz,
                                  lx, ly, lz, xy, xz, yz);
  }
  Real qi   = pq_i_load.w;
  // Per-atom particle-type index for the i-atom, loaded once per atom
  // (amortized across all of this lane's pairs). Stays 0u when no
  // active fragment consumes the type index. rq-b10f28d7 rq-61fa8b93
  unsigned int i_type = 0u;
  /*HEDDLE_JIT_ITYPE_LOAD*/

  // Per-warp register accumulator persists across every entry this
  // warp processes — this is the register-staging optimization that
  // collapses one i-side global atomic per (entry, lane) down to one
  // per (warp, lane). The accumulators are i64 fixed-point: a warp's
  // tile entries arrive in a non-deterministic (atomic-built) order, and
  // integer addition is associative, so the per-i-atom total is bit-exact
  // across runs. Float accumulation here would make the sum depend on
  // entry order and break run-to-run reproducibility. rq-693544f8
  unsigned long long warp_i_fx = 0u, warp_i_fy = 0u, warp_i_fz = 0u;
  unsigned long long warp_i_e  = 0u, warp_i_w  = 0u;

  __syncthreads();

  for (unsigned int e = range_begin + warp_id_in_block;
       e < range_end;
       e += HEDDLE_JIT_WARPS_PER_BLOCK) {
    // Each lane reads its j-atom ID (one per lane) and j-position
    // from the canonical particle-id-ordered positions array.
    unsigned int j_atom_id = sorted_interacting_atoms[e * 32u + lane];
    bool j_valid = j_atom_id < n;
    Real4 pq_j_load;
    if (j_valid) {
      pq_j_load = posq[j_atom_id];
    } else {
      pq_j_load.x = R(0.0); pq_j_load.y = R(0.0); pq_j_load.z = R(0.0); pq_j_load.w = R(0.0);
    }
    Real pj_x = pq_j_load.x;
    Real pj_y = pq_j_load.y;
    Real pj_z = pq_j_load.z;
    if (spc && j_valid) {
      triclinic_wrap_against_center(pj_x, pj_y, pj_z,
                                    cx, cy, cz,
                                    lx, ly, lz, xy, xz, yz);
    }
    Real qj   = pq_j_load.w;
    // Per-atom particle-type index for the j-atom, loaded once per
    // atom and rotated with the j-side state through the diagonal
    // shuffle below. rq-b10f28d7
    unsigned int j_type = 0u;
    /*HEDDLE_JIT_JTYPE_LOAD*/

    // Per-lane `j_in_iblock` flag: does this lane's current j-atom
    // sit anywhere in the i-block's 32-atom set? If yes, the pair
    // (i-atom_of_some_lane, j-atom) will be visited from BOTH sides
    // of the 32-rotation sweep — once as (this-lane's i, that j)
    // and once as (that-lane's i, this j) — so applying Newton's
    // 3rd via `j_fx -= fx` would double-count. Suppressing j-side
    // for `j_in_iblock` pairs leaves each atom's contribution on
    // its own i-side (`ei_fx`) accumulator exactly once, matching
    // the "self-block, no Newton 3rd" convention but done per pair
    // instead of per entry. This is what makes the packed kernel
    // robust against mixed entries where a self-block-like set of
    // j-atoms co-inhabits with cross-block j-atoms.
    //
    // Detection: iterate over each initial-lane m in [0, 32),
    // broadcast that lane's `j_atom_id` to the whole warp, ballot
    // whether any lane's `i_atom_id` matches, and record bit `m` in
    // `j_in_iblock_ballot`. Then each lane extracts its own bit.
    // The mask rotates alongside the j-side state through the
    // 32-iteration loop, so at rotation `r` this lane's flag
    // corresponds to the j-atom currently in this lane's registers.
    unsigned int j_in_iblock_ballot = 0u;
    #pragma unroll
    for (unsigned int m = 0u; m < 32u; ++m) {
      unsigned int j_m = __shfl_sync(0xFFFFFFFFu, j_atom_id, m);
      bool match = i_valid && (j_m < n) && (i_atom_id == j_m);
      unsigned int b = __ballot_sync(0xFFFFFFFFu, match ? 1u : 0u);
      if (b != 0u) {
        j_in_iblock_ballot |= (1u << m);
      }
    }
    unsigned int my_j_in_iblock = (j_in_iblock_ballot >> lane) & 1u;

    // Per-entry j-side accumulator (reset every entry — different
    // j-atoms each time).
    Real j_fx = R(0.0), j_fy = R(0.0), j_fz = R(0.0);
    Real j_e  = R(0.0), j_w  = R(0.0);
    // Per-entry i-side accumulator (float). Summed over this entry's
    // fixed 32-rotation order — deterministic — then converted to
    // fixed-point once per entry and folded into the i64 warp
    // accumulator. One conversion per entry (not per pair). rq-693544f8
    Real ei_fx = R(0.0), ei_fy = R(0.0), ei_fz = R(0.0);
    Real ei_e  = R(0.0), ei_w  = R(0.0);

    for (unsigned int t = 0u; t < 32u; ++t) {
      if (i_valid && j_valid && i_atom_id != j_atom_id) {
        Real dx = pi_x - pj_x;
        Real dy = pi_y - pj_y;
        Real dz = pi_z - pj_z;
        if (!spc) {
          heddle_jit_triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);
        }
        Real r2 = dx * dx + dy * dy + dz * dz;

        // Shared scalar intermediates: one rsqrt + one multiply
        // computes `1/r` and `r` for the warp once per pair. Every
        // fragment's `evaluate` consumes these instead of recomputing
        // `1/r²`, `sqrt(1/r²)`, or `1/r` from `r²` itself.
        Real inv_r = Real_rsqrt(r2);
        Real r = r2 * inv_r;

        // Branchless max-cutoff mask. Fragment math runs
        // unconditionally; the mask zeroes contributions for pairs
        // past HEDDLE_JIT_MAX_CUTOFF_SQUARED. Multiplying a finite
        // value by +0.0f yields +0.0f in IEEE-754, so accumulators
        // are bit-exact zero for out-of-cutoff pairs.
        Real cutoff_mask = (r2 <= HEDDLE_JIT_MAX_CUTOFF_SQUARED) ? R(1.0) : R(0.0);

        Real factor = R(0.0), energy = R(0.0);
        heddle_jit_eval_pair_sum<WriteEv>(composite, r2, inv_r, r,
                                           qi, qj,
                                           i_type, j_type,
                                           i_atom_id, j_atom_id,
                                           factor, energy);
        factor *= cutoff_mask;
        if (WriteEv) {
          energy *= cutoff_mask;
        }
        Real fx = factor * dx;
        Real fy = factor * dy;
        Real fz = factor * dz;
        // Both sides accumulate in float within the entry (fixed
        // 32-rotation order, deterministic). The i-side per-entry sum is
        // converted to fixed-point once after the loop; the j-side is
        // flushed per entry.
        ei_fx += fx;  ei_fy += fy;  ei_fz += fz;
        if (my_j_in_iblock == 0u) {
          j_fx -= fx;  j_fy -= fy;  j_fz -= fz;
        }
        if (WriteEv) {
          Real he = energy * R(0.5);
          // rq-7d64da58 — derive the per-pair scalar virial from the
          // masked, exclusion-scaled force factor: W = factor * r2.
          Real hw = (factor * r2) * R(0.5);
          ei_e += he;  ei_w += hw;
          if (my_j_in_iblock == 0u) {
            j_e += he;  j_w += hw;
          }
        }
      }
      // Rotate j-side state by one lane. The `my_j_in_iblock`
      // per-lane flag rotates alongside because it belongs to the
      // j-atom currently at this lane.
      unsigned int src_lane = (lane + 1u) & 31u;
      pj_x = __shfl_sync(0xFFFFFFFFu, pj_x, src_lane);
      pj_y = __shfl_sync(0xFFFFFFFFu, pj_y, src_lane);
      pj_z = __shfl_sync(0xFFFFFFFFu, pj_z, src_lane);
      qj   = __shfl_sync(0xFFFFFFFFu, qj,   src_lane);
      j_atom_id = __shfl_sync(0xFFFFFFFFu, j_atom_id, src_lane);
      j_valid = j_atom_id < n;
      /*HEDDLE_JIT_JTYPE_SHUFFLE*/
      my_j_in_iblock = __shfl_sync(0xFFFFFFFFu, my_j_in_iblock, src_lane);
      j_fx = __shfl_sync(0xFFFFFFFFu, j_fx, src_lane);
      j_fy = __shfl_sync(0xFFFFFFFFu, j_fy, src_lane);
      j_fz = __shfl_sync(0xFFFFFFFFu, j_fz, src_lane);
      if (WriteEv) {
        j_e = __shfl_sync(0xFFFFFFFFu, j_e, src_lane);
        j_w = __shfl_sync(0xFFFFFFFFu, j_w, src_lane);
      }
    }

    // Fold this entry's float i-side sum into the i64 warp accumulator —
    // one fixed-point conversion per entry. The cross-entry sum is i64
    // (associative), so it is bit-exact regardless of entry order.
    if (i_valid) {
      warp_i_fx += (unsigned long long) heddle_jit_real_to_fixed(ei_fx);
      warp_i_fy += (unsigned long long) heddle_jit_real_to_fixed(ei_fy);
      warp_i_fz += (unsigned long long) heddle_jit_real_to_fixed(ei_fz);
      if (WriteEv) {
        warp_i_e += (unsigned long long) heddle_jit_real_to_fixed(ei_e);
        warp_i_w += (unsigned long long) heddle_jit_real_to_fixed(ei_w);
      }
    }

    // j-side global atomic, one per (entry, lane). j-atoms change
    // every entry, so we have to flush per entry — the register
    // staging only helps the i-side.
    if (j_valid) {
      heddle_jit_atomic_add_fp(fast_force_x_fp, j_atom_id, j_fx);
      heddle_jit_atomic_add_fp(fast_force_y_fp, j_atom_id, j_fy);
      heddle_jit_atomic_add_fp(fast_force_z_fp, j_atom_id, j_fz);
      if (WriteEv) {
        heddle_jit_atomic_add_fp(fast_energy_fp, j_atom_id, j_e);
        heddle_jit_atomic_add_fp(fast_virial_fp, j_atom_id, j_w);
      }
    }
  }

  // Each warp adds its warp-resident i-side sum to the block's shared
  // accumulator. Shared atomicAdd on u64 is cheap (no global L2 hop)
  // and integer addition is associative — ordering across warps is
  // irrelevant to the final value.
  if (i_valid) {
    // warp_i_* are already i64 fixed-point — add them straight in.
    atomicAdd(&shared_fx[lane], warp_i_fx);
    atomicAdd(&shared_fy[lane], warp_i_fy);
    atomicAdd(&shared_fz[lane], warp_i_fz);
    if (WriteEv) {
      atomicAdd(&shared_e[lane], warp_i_e);
      atomicAdd(&shared_w[lane], warp_i_w);
    }
  }
  __syncthreads();

  // First warp flushes the shared accumulator to global — one global
  // atomic per (i_block, i-atom) for the whole block, regardless of
  // how many entries this i-block had.
  if (warp_id_in_block == 0u && i_valid) {
    atomicAdd(&fast_force_x_fp[i_atom_id], shared_fx[lane]);
    atomicAdd(&fast_force_y_fp[i_atom_id], shared_fy[lane]);
    atomicAdd(&fast_force_z_fp[i_atom_id], shared_fz[lane]);
    if (WriteEv) {
      atomicAdd(&fast_energy_fp[i_atom_id], shared_e[lane]);
      atomicAdd(&fast_virial_fp[i_atom_id], shared_w[lane]);
    }
  }
}
"#;

// Per-pair sparse-tile outer loop. One thread per entry in
// `single_pair_atoms`. Reads the canonical (i, j) atom IDs, computes
// (dx, dy, dz, r2, inv_r, r), invokes the per-pair evaluator
// (`heddle_jit_eval_pair_sum`, which multiplies by the fragment's
// `exclusion_scale(i, j)` inline), applies the branchless max-cutoff
// mask, and atomic-adds the per-fragment Newton's-3rd-law pair
// contribution to both atoms' fixed-point slots.
const SINGLE_PAIR_LOOP_TEMPLATE: &str = r#"
template <bool WriteEv>
__device__ static inline void heddle_jit_single_pair_loop(
    const HeddleJitComposedPairFunc &composite,
    const unsigned int *single_pair_atoms,
    const unsigned int *interaction_count_ptr,
    const Real4 *posq,
    const unsigned int *type_indices,
    const Real *lattice,
    unsigned long long *fast_force_x_fp,
    unsigned long long *fast_force_y_fp,
    unsigned long long *fast_force_z_fp,
    unsigned long long *fast_energy_fp,
    unsigned long long *fast_virial_fp,
    unsigned int n)
{
  // Read the live single-pair count from device memory at kernel
  // entry. Passing a device pointer (rather than a scalar value) is
  // load-bearing under CUDA graph capture: every neighbour-list
  // rebuild updates `interaction_count[1]` in place, and the
  // captured kernel reads the fresh value at each replay.
  unsigned int single_pair_count = interaction_count_ptr[1];
  unsigned int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair_idx >= single_pair_count) return;
  unsigned int atom_i = single_pair_atoms[2u * pair_idx];
  unsigned int atom_j = single_pair_atoms[2u * pair_idx + 1u];
  if (atom_i >= n || atom_j >= n) return;

  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];

  Real4 pq_i = posq[atom_i];
  Real4 pq_j = posq[atom_j];
  Real qi = pq_i.w;
  Real qj = pq_j.w;
  // Per-atom particle-type indices for this sparse pair. One thread
  // per pair, so the indices are loaded directly (no amortization).
  // rq-b10f28d7 rq-61fa8b93
  unsigned int i_type = 0u;
  unsigned int j_type = 0u;
  /*HEDDLE_JIT_TYPE_LOAD_PERPAIR*/

  Real dx = pq_i.x - pq_j.x;
  Real dy = pq_i.y - pq_j.y;
  Real dz = pq_i.z - pq_j.z;
  heddle_jit_triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);
  Real r2 = dx * dx + dy * dy + dz * dz;
  Real inv_r = Real_rsqrt(r2);
  Real r = r2 * inv_r;

  Real cutoff_mask = (r2 <= HEDDLE_JIT_MAX_CUTOFF_SQUARED) ? R(1.0) : R(0.0);

  Real factor = R(0.0), energy = R(0.0);
  heddle_jit_eval_pair_sum<WriteEv>(
      composite, r2, inv_r, r, qi, qj, i_type, j_type, atom_i, atom_j, factor, energy);
  factor *= cutoff_mask;
  if (WriteEv) {
    energy *= cutoff_mask;
  }

  Real fx = factor * dx;
  Real fy = factor * dy;
  Real fz = factor * dz;

  heddle_jit_atomic_add_fp(fast_force_x_fp, atom_i,  fx);
  heddle_jit_atomic_add_fp(fast_force_y_fp, atom_i,  fy);
  heddle_jit_atomic_add_fp(fast_force_z_fp, atom_i,  fz);
  heddle_jit_atomic_add_fp(fast_force_x_fp, atom_j, -fx);
  heddle_jit_atomic_add_fp(fast_force_y_fp, atom_j, -fy);
  heddle_jit_atomic_add_fp(fast_force_z_fp, atom_j, -fz);

  if (WriteEv) {
    Real he = energy * R(0.5);
    // rq-7d64da58 — derive the per-pair scalar virial: W = factor * r2.
    Real hw = (factor * r2) * R(0.5);
    heddle_jit_atomic_add_fp(fast_energy_fp, atom_i, he);
    heddle_jit_atomic_add_fp(fast_energy_fp, atom_j, he);
    heddle_jit_atomic_add_fp(fast_virial_fp, atom_i, hw);
    heddle_jit_atomic_add_fp(fast_virial_fp, atom_j, hw);
  }
}
"#;

// Exclusion-tile pair-force pass. One warp per exclusion tile — a block
// pair (bi <= bj) that contains at least one modified atom pair (a pair
// whose scale differs from 1 in some fragment). The whole block pair
// (modified and unmodified pairs alike) is computed here so the bulk and
// single-pair passes can skip it and do zero exclusion work. Lane `l`
// owns i-atom slot bi*32+l and initially j-atom slot bj*32+l; the
// 32-iteration diagonal shuffle visits every (i_lane, j_lane) pair. A
// pair flagged in the modified-pair bitmask is summed through the
// scale-aware evaluator (applying each fragment's per-pair scale — a
// full exclusion contributes nothing, a 1-4 pair a scaled force); an
// unflagged pair is evaluated at full strength. The self-block (bi==bj)
// diagonal is skipped. A self-block tile applies the i-side only — each
// intra-block pair is visited from both orderings and the bitmask is
// symmetric — and a cross-block tile applies both sides (Newton's 3rd).
// The j-side diagonal-shuffle bookkeeping mirrors the packed-neighbour
// pass. Accumulation is i64 fixed-point, so it is order-independent and
// bit-exact run to run. rq-fa0b3d10
const EXCL_TILE_LOOP_TEMPLATE: &str = r#"
template <bool WriteEv>
__device__ static inline void heddle_jit_excl_tile_loop(
    const HeddleJitComposedPairFunc &composite,
    const unsigned int *exclusion_tile_iblocks,
    const unsigned int *exclusion_tile_jblocks,
    const unsigned int *exclusion_tile_masks,
    const unsigned int *interaction_count_ptr,
    const Real4 *tile_sorted_posq,
    const unsigned int *sorted_particle_ids,
    const unsigned int *type_indices,
    const Real *lattice,
    unsigned long long *fast_force_x_fp,
    unsigned long long *fast_force_y_fp,
    unsigned long long *fast_force_z_fp,
    unsigned long long *fast_energy_fp,
    unsigned long long *fast_virial_fp,
    unsigned int n)
{
  // Live exclusion-tile count read from device memory (graph-capture
  // safe: the host rebuild refreshes interaction_count[2] in place).
  unsigned int n_excl_tiles = interaction_count_ptr[2];
  unsigned int warp_global =
      (blockIdx.x * blockDim.x + threadIdx.x) / HEDDLE_JIT_WARP_SIZE;
  if (warp_global >= n_excl_tiles) return;
  unsigned int lane = threadIdx.x & (HEDDLE_JIT_WARP_SIZE - 1u);
  unsigned int t = warp_global;

  unsigned int bi = exclusion_tile_iblocks[t];
  unsigned int bj = exclusion_tile_jblocks[t];
  bool self_block = (bi == bj);

  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];

  // i-atom: this lane owns slot bi*32 + lane.
  unsigned int i_slot = bi * 32u + lane;
  bool i_valid = i_slot < n;
  unsigned int i_atom_id = i_valid ? sorted_particle_ids[i_slot] : n;
  Real4 pq_i = tile_sorted_posq[i_slot];
  Real pi_x = pq_i.x, pi_y = pq_i.y, pi_z = pq_i.z, qi = pq_i.w;
  unsigned int i_type = 0u;
  /*HEDDLE_JIT_ITYPE_LOAD*/

  // j-atom: this lane owns slot bj*32 + lane initially; it rotates.
  unsigned int j_slot = bj * 32u + lane;
  bool j_valid = j_slot < n;
  unsigned int j_atom_id = j_valid ? sorted_particle_ids[j_slot] : n;
  Real4 pq_j = tile_sorted_posq[j_slot];
  Real pj_x = pq_j.x, pj_y = pq_j.y, pj_z = pq_j.z, qj = pq_j.w;
  unsigned int j_type = 0u;
  /*HEDDLE_JIT_JTYPE_LOAD*/

  // This lane's modified-pair mask row (i_lane == lane): bit j_lane set
  // => pair (lane, j_lane) is a modified pair whose per-fragment scale is
  // applied. The mask row is fixed for this lane, so no rotation is
  // needed — at iteration tt the partner is j_lane = (lane + tt) mod 32.
  unsigned int my_mask_row = exclusion_tile_masks[t * 32u + lane];

  // i-side accumulator (this lane's i-atom, fixed). j-side accumulator
  // travels with the j-atom currently held by this lane.
  Real i_fx = R(0.0), i_fy = R(0.0), i_fz = R(0.0);
  Real i_e  = R(0.0), i_w  = R(0.0);
  Real j_fx = R(0.0), j_fy = R(0.0), j_fz = R(0.0);
  Real j_e  = R(0.0), j_w  = R(0.0);

  for (unsigned int tt = 0u; tt < 32u; ++tt) {
    unsigned int j_lane = (lane + tt) & 31u;
    bool modified = ((my_mask_row >> j_lane) & 1u) != 0u;
    if (i_valid && j_valid && i_atom_id != j_atom_id) {
      Real dx = pi_x - pj_x;
      Real dy = pi_y - pj_y;
      Real dz = pi_z - pj_z;
      heddle_jit_triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);
      Real r2 = dx * dx + dy * dy + dz * dz;
      Real inv_r = Real_rsqrt(r2);
      Real r = r2 * inv_r;
      Real cutoff_mask = (r2 <= HEDDLE_JIT_MAX_CUTOFF_SQUARED) ? R(1.0) : R(0.0);
      Real factor = R(0.0), energy = R(0.0);
      // A flagged (modified) pair applies its per-fragment exclusion
      // scale; an unflagged pair is evaluated at full strength with no
      // scale lookup. A fully-excluded pair (scale 0) contributes
      // nothing; a 1-4 pair contributes a scaled force. rq-fa0b3d10
      if (modified) {
        heddle_jit_eval_pair_sum_scaled<WriteEv>(composite, r2, inv_r, r,
                                           qi, qj, i_type, j_type,
                                           i_atom_id, j_atom_id,
                                           factor, energy);
      } else {
        heddle_jit_eval_pair_sum<WriteEv>(composite, r2, inv_r, r,
                                           qi, qj, i_type, j_type,
                                           i_atom_id, j_atom_id,
                                           factor, energy);
      }
      factor *= cutoff_mask;
      if (WriteEv) energy *= cutoff_mask;
      Real fx = factor * dx, fy = factor * dy, fz = factor * dz;
      i_fx += fx; i_fy += fy; i_fz += fz;
      if (!self_block) { j_fx -= fx; j_fy -= fy; j_fz -= fz; }
      if (WriteEv) {
        Real he = energy * R(0.5);
        // rq-7d64da58 — per-pair scalar virial: W = factor * r2.
        Real hw = (factor * r2) * R(0.5);
        i_e += he; i_w += hw;
        if (!self_block) { j_e += he; j_w += hw; }
      }
    }
    unsigned int src_lane = (lane + 1u) & 31u;
    pj_x = __shfl_sync(0xFFFFFFFFu, pj_x, src_lane);
    pj_y = __shfl_sync(0xFFFFFFFFu, pj_y, src_lane);
    pj_z = __shfl_sync(0xFFFFFFFFu, pj_z, src_lane);
    qj   = __shfl_sync(0xFFFFFFFFu, qj,   src_lane);
    j_atom_id = __shfl_sync(0xFFFFFFFFu, j_atom_id, src_lane);
    j_valid = j_atom_id < n;
    /*HEDDLE_JIT_JTYPE_SHUFFLE*/
    j_fx = __shfl_sync(0xFFFFFFFFu, j_fx, src_lane);
    j_fy = __shfl_sync(0xFFFFFFFFu, j_fy, src_lane);
    j_fz = __shfl_sync(0xFFFFFFFFu, j_fz, src_lane);
    if (WriteEv) {
      j_e = __shfl_sync(0xFFFFFFFFu, j_e, src_lane);
      j_w = __shfl_sync(0xFFFFFFFFu, j_w, src_lane);
    }
  }

  // Flush: i-side always; j-side only for cross-block tiles (a
  // self-block tile's Newton's-3rd contributions are already covered by
  // the symmetric sweep). After 32 rotations the j-side registers and
  // j_atom_id have returned to their starting lane.
  if (i_valid) {
    heddle_jit_atomic_add_fp(fast_force_x_fp, i_atom_id, i_fx);
    heddle_jit_atomic_add_fp(fast_force_y_fp, i_atom_id, i_fy);
    heddle_jit_atomic_add_fp(fast_force_z_fp, i_atom_id, i_fz);
    if (WriteEv) {
      heddle_jit_atomic_add_fp(fast_energy_fp, i_atom_id, i_e);
      heddle_jit_atomic_add_fp(fast_virial_fp, i_atom_id, i_w);
    }
  }
  if (!self_block && j_valid) {
    heddle_jit_atomic_add_fp(fast_force_x_fp, j_atom_id, j_fx);
    heddle_jit_atomic_add_fp(fast_force_y_fp, j_atom_id, j_fy);
    heddle_jit_atomic_add_fp(fast_force_z_fp, j_atom_id, j_fz);
    if (WriteEv) {
      heddle_jit_atomic_add_fp(fast_energy_fp, j_atom_id, j_e);
      heddle_jit_atomic_add_fp(fast_virial_fp, j_atom_id, j_w);
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
    /// signature: common args (posq, bonds, lattice,
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
    s.push_str("    const Real4 *posq,\n");
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
    s.push_str("    Real4 pq_i = posq[atom_i];\n");
    s.push_str("    Real4 pq_j = posq[atom_j];\n");
    s.push_str("    Real dx = pq_i.x - pq_j.x;\n");
    s.push_str("    Real dy = pq_i.y - pq_j.y;\n");
    s.push_str("    Real dz = pq_i.z - pq_j.z;\n");
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
    s.push_str("    Real fmag, u_k;\n");
    s.push_str("    functor.evaluate(r2, r, type_idx, dx, dy, dz, fmag, u_k);\n");
    // rq-ff5a04bc — the functor emits no virial; derive the bond virial
    // W_k = fmag * r2 from the force factor and the separation.
    s.push_str("    Real w_k = fmag * r2;\n");
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
    s.push_str("    const Real4 *posq,\n");
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
    s.push_str("    Real4 pq_i = posq[atom_i];\n");
    s.push_str("    Real4 pq_j = posq[atom_j];\n");
    s.push_str("    Real4 pq_k = posq[atom_k];\n");
    s.push_str("    Real dx_ij = pq_i.x - pq_j.x;\n");
    s.push_str("    Real dy_ij = pq_i.y - pq_j.y;\n");
    s.push_str("    Real dz_ij = pq_i.z - pq_j.z;\n");
    s.push_str("    Real dx_kj = pq_k.x - pq_j.x;\n");
    s.push_str("    Real dy_kj = pq_k.y - pq_j.y;\n");
    s.push_str("    Real dz_kj = pq_k.z - pq_j.z;\n");
    s.push_str("    heddle_jit_triclinic_min_image(dx_ij, dy_ij, dz_ij, lx, ly, lz, xy, xz, yz);\n");
    s.push_str("    heddle_jit_triclinic_min_image(dx_kj, dy_kj, dz_kj, lx, ly, lz, xy, xz, yz);\n");
    s.push_str("    Real fix, fiy, fiz, fkx, fky, fkz, u_m;\n");
    s.push_str("    functor.evaluate(dx_ij, dy_ij, dz_ij, dx_kj, dy_kj, dz_kj, type_idx,\n");
    s.push_str("                     fix, fiy, fiz, fkx, fky, fkz, u_m);\n");
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
        // rq-a997a963 — the functor emits no virial; derive the angle
        // virial W_m = r_ij·F_i + r_kj·F_k from the returned forces and
        // the leg displacements.
        s.push_str("    Real w_m = dx_ij * fix + dy_ij * fiy + dz_ij * fiz\n");
        s.push_str("             + dx_kj * fkx + dy_kj * fky + dz_kj * fkz;\n");
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
// Dihedral composer
// ============================================================

const DIHEDRAL_MODULE_NAME: &str = "heddle_jit_composed_dihedral";

/// JIT-composed dihedral contribution module + per-slot entry-point
/// handles. Built by `ForceField::new` when at least one fast-class
/// dihedral slot is active.
#[derive(Debug)]
pub struct JitComposedDihedralForce {
    pub fragment_labels: Vec<&'static str>,
    pub entry_points_f: Vec<CudaFunction>,
    pub entry_points_fev: Vec<CudaFunction>,
}

impl JitComposedDihedralForce {
    pub fn compile_and_load(
        device: &Arc<CudaDevice>,
        fragments: &[DihedralForceFragment],
    ) -> Result<Self, ForceFieldError> {
        let source = compose_dihedral_source(fragments);
        let ptx = jit_compile(device, &source, |log| {
            ForceFieldError::FragmentCompileFailed {
                log: format_dihedral_compile_failure(fragments, log, &source),
            }
        })?;

        let mut entry_name_refs: Vec<&'static str> = Vec::with_capacity(2 * fragments.len());
        for i in 0..fragments.len() {
            entry_name_refs.push(Box::leak(
                format!("heddle_jit_composed_dihedral_{}_f", i).into_boxed_str(),
            ));
            entry_name_refs.push(Box::leak(
                format!("heddle_jit_composed_dihedral_{}_fev", i).into_boxed_str(),
            ));
        }

        device
            .load_ptx(ptx, DIHEDRAL_MODULE_NAME, &entry_name_refs)
            .map_err(|e| ForceFieldError::FragmentLoadFailed(GpuError::from(e)))?;

        let mut entry_points_f: Vec<CudaFunction> = Vec::with_capacity(fragments.len());
        let mut entry_points_fev: Vec<CudaFunction> = Vec::with_capacity(fragments.len());
        for i in 0..fragments.len() {
            entry_points_f.push(
                device
                    .get_func(DIHEDRAL_MODULE_NAME, entry_name_refs[2 * i])
                    .expect("composed dihedral kernel _f entry was just loaded"),
            );
            entry_points_fev.push(
                device
                    .get_func(DIHEDRAL_MODULE_NAME, entry_name_refs[2 * i + 1])
                    .expect("composed dihedral kernel _fev entry was just loaded"),
            );
        }

        Ok(JitComposedDihedralForce {
            fragment_labels: fragments.iter().map(|f| f.label).collect(),
            entry_points_f,
            entry_points_fev,
        })
    }

    /// Launch one slot's composed dihedral entry point.
    pub unsafe fn launch_slot(
        &self,
        slot_index: usize,
        n_dihedrals: u32,
        use_fev: bool,
        mut builder: ForceLaunchBuilder,
    ) -> Result<(), GpuError> {
        let cfg = LaunchConfig {
            grid_dim: (n_dihedrals.div_ceil(256), 1, 1),
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

fn compose_dihedral_source(fragments: &[DihedralForceFragment]) -> String {
    let mut s = String::with_capacity(
        8192 + fragments.iter().map(|f| f.functor_source.len()).sum::<usize>(),
    );
    s.push_str(PREAMBLE);
    for f in fragments {
        s.push_str("// ---- dihedral fragment functor source: ");
        s.push_str(f.label);
        s.push_str(" ----\n");
        s.push_str(&f.functor_source);
        s.push_str("\n// ---- end dihedral fragment functor source: ");
        s.push_str(f.label);
        s.push_str(" ----\n");
    }
    for (i, f) in fragments.iter().enumerate() {
        emit_dihedral_entry_point(&mut s, f, i, false);
        emit_dihedral_entry_point(&mut s, f, i, true);
    }
    s
}

fn emit_dihedral_entry_point(
    s: &mut String,
    fragment: &DihedralForceFragment,
    slot_index: usize,
    write_ev: bool,
) {
    let entry_name = format!(
        "heddle_jit_composed_dihedral_{}_{}",
        slot_index,
        if write_ev { "fev" } else { "f" }
    );
    s.push_str("\nextern \"C\" __global__ void ");
    s.push_str(&entry_name);
    s.push_str("(\n");
    s.push_str("    const Real4 *posq,\n");
    s.push_str("    const unsigned int *dihedrals,\n");
    s.push_str("    const Real *lattice,\n");
    s.push_str("    Real *dihedral_quadruple_x,\n");
    s.push_str("    Real *dihedral_quadruple_y,\n");
    s.push_str("    Real *dihedral_quadruple_z,\n");
    if write_ev {
        s.push_str("    Real *dihedral_quadruple_energy,\n");
        s.push_str("    Real *dihedral_quadruple_virial,\n");
    }
    s.push_str(&fragment.entry_point_args);
    s.push_str("    unsigned int n_dihedrals)\n");
    s.push_str("{\n");
    s.push_str(&format!(
        "    {} functor;\n",
        fragment.functor_struct_name
    ));
    s.push_str(&fragment.functor_init_source);
    s.push_str("    Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];\n");
    s.push_str("    Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];\n");
    s.push_str("    unsigned int m = blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str("    if (m >= n_dihedrals) return;\n");
    s.push_str("    unsigned int atom_i = dihedrals[5u * m + 0u];\n");
    s.push_str("    unsigned int atom_j = dihedrals[5u * m + 1u];\n");
    s.push_str("    unsigned int atom_k = dihedrals[5u * m + 2u];\n");
    s.push_str("    unsigned int atom_l = dihedrals[5u * m + 3u];\n");
    s.push_str("    unsigned int type_idx = dihedrals[5u * m + 4u];\n");
    s.push_str("    Real4 pq_i = posq[atom_i];\n");
    s.push_str("    Real4 pq_j = posq[atom_j];\n");
    s.push_str("    Real4 pq_k = posq[atom_k];\n");
    s.push_str("    Real4 pq_l = posq[atom_l];\n");
    s.push_str("    Real dx_ij = pq_i.x - pq_j.x;\n");
    s.push_str("    Real dy_ij = pq_i.y - pq_j.y;\n");
    s.push_str("    Real dz_ij = pq_i.z - pq_j.z;\n");
    s.push_str("    Real dx_kj = pq_k.x - pq_j.x;\n");
    s.push_str("    Real dy_kj = pq_k.y - pq_j.y;\n");
    s.push_str("    Real dz_kj = pq_k.z - pq_j.z;\n");
    s.push_str("    Real dx_kl = pq_k.x - pq_l.x;\n");
    s.push_str("    Real dy_kl = pq_k.y - pq_l.y;\n");
    s.push_str("    Real dz_kl = pq_k.z - pq_l.z;\n");
    s.push_str("    heddle_jit_triclinic_min_image(dx_ij, dy_ij, dz_ij, lx, ly, lz, xy, xz, yz);\n");
    s.push_str("    heddle_jit_triclinic_min_image(dx_kj, dy_kj, dz_kj, lx, ly, lz, xy, xz, yz);\n");
    s.push_str("    heddle_jit_triclinic_min_image(dx_kl, dy_kl, dz_kl, lx, ly, lz, xy, xz, yz);\n");
    s.push_str("    Real fix, fiy, fiz, fjx, fjy, fjz, fkx, fky, fkz, flx, fly, flz, u_m;\n");
    s.push_str("    functor.evaluate(dx_ij, dy_ij, dz_ij, dx_kj, dy_kj, dz_kj, dx_kl, dy_kl, dz_kl, type_idx,\n");
    s.push_str("                     fix, fiy, fiz, fjx, fjy, fjz, fkx, fky, fkz, flx, fly, flz, u_m);\n");
    s.push_str("    dihedral_quadruple_x[4u * m + 0u] = fix;\n");
    s.push_str("    dihedral_quadruple_y[4u * m + 0u] = fiy;\n");
    s.push_str("    dihedral_quadruple_z[4u * m + 0u] = fiz;\n");
    s.push_str("    dihedral_quadruple_x[4u * m + 1u] = fjx;\n");
    s.push_str("    dihedral_quadruple_y[4u * m + 1u] = fjy;\n");
    s.push_str("    dihedral_quadruple_z[4u * m + 1u] = fjz;\n");
    s.push_str("    dihedral_quadruple_x[4u * m + 2u] = fkx;\n");
    s.push_str("    dihedral_quadruple_y[4u * m + 2u] = fky;\n");
    s.push_str("    dihedral_quadruple_z[4u * m + 2u] = fkz;\n");
    s.push_str("    dihedral_quadruple_x[4u * m + 3u] = flx;\n");
    s.push_str("    dihedral_quadruple_y[4u * m + 3u] = fly;\n");
    s.push_str("    dihedral_quadruple_z[4u * m + 3u] = flz;\n");
    if write_ev {
        s.push_str("    Real e_share = u_m * R(0.25);\n");
        // rq-932d37a2 — the functor emits no virial; derive the dihedral
        // virial W_m = Σ_a (r_a − r_j)·F_a from the returned forces and
        // the bond displacements. r_l − r_j = (r_k − r_j) − (r_k − r_l)
        // = dx_kj − dx_kl.
        s.push_str("    Real dx_lj = dx_kj - dx_kl;\n");
        s.push_str("    Real dy_lj = dy_kj - dy_kl;\n");
        s.push_str("    Real dz_lj = dz_kj - dz_kl;\n");
        s.push_str("    Real w_m = dx_ij * fix + dy_ij * fiy + dz_ij * fiz\n");
        s.push_str("             + dx_kj * fkx + dy_kj * fky + dz_kj * fkz\n");
        s.push_str("             + dx_lj * flx + dy_lj * fly + dz_lj * flz;\n");
        s.push_str("    Real w_share = w_m * R(0.25);\n");
        s.push_str("    dihedral_quadruple_energy[4u * m + 0u] = e_share;\n");
        s.push_str("    dihedral_quadruple_energy[4u * m + 1u] = e_share;\n");
        s.push_str("    dihedral_quadruple_energy[4u * m + 2u] = e_share;\n");
        s.push_str("    dihedral_quadruple_energy[4u * m + 3u] = e_share;\n");
        s.push_str("    dihedral_quadruple_virial[4u * m + 0u] = w_share;\n");
        s.push_str("    dihedral_quadruple_virial[4u * m + 1u] = w_share;\n");
        s.push_str("    dihedral_quadruple_virial[4u * m + 2u] = w_share;\n");
        s.push_str("    dihedral_quadruple_virial[4u * m + 3u] = w_share;\n");
    }
    s.push_str("}\n");
}

fn format_dihedral_compile_failure(
    fragments: &[DihedralForceFragment],
    log: &str,
    source: &str,
) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "nvrtc failed to compile the JIT-composed dihedral kernel."
    );
    let _ = writeln!(s, "Active dihedral fragments (canonical slot order):");
    for f in fragments {
        let _ = writeln!(s, "  - {} (functor: {})", f.label, f.functor_struct_name);
    }
    let _ = writeln!(s, "nvrtc compile log:");
    let _ = writeln!(s, "{}", log);
    let _ = writeln!(s, "Composed dihedral source (line-numbered):");
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
    push_jit_fast_math(&mut options);
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

#[cfg(test)]
mod launch_bounds_tests {
    use super::*;

    // The entry-point scaffolding (including launch bounds) is emitted
    // regardless of the fragment list, so a fragment-free composed source
    // is sufficient to inspect the kernel declarations.
    fn composed_source_for_inspection() -> String {
        compose_source(&[], 1.0 as Real, false)
    }

    // rq-20febc65
    #[test]
    fn packed_neighbour_entry_points_declare_launch_bounds() {
        let src = composed_source_for_inspection();
        let lb = format!(
            "__launch_bounds__({}, {})",
            BLOCK_SIZE, PACKED_MIN_BLOCKS_PER_SM
        );
        assert!(
            src.contains(&format!("{lb} {F_ENTRY}(")),
            "packed _f entry point missing launch bounds"
        );
        assert!(
            src.contains(&format!("{lb} {FEV_ENTRY}(")),
            "packed _fev entry point missing launch bounds"
        );
    }

    // rq-139cffbe
    #[test]
    fn launch_bounds_arguments_match_constants() {
        assert_eq!(BLOCK_SIZE, 256);
        let src = composed_source_for_inspection();
        let lb = format!(
            "__launch_bounds__({}, {})",
            BLOCK_SIZE, PACKED_MIN_BLOCKS_PER_SM
        );
        assert!(src.contains(&format!("{lb} {F_ENTRY}(")));
        assert!(src.contains(&format!("{lb} {FEV_ENTRY}(")));
        // Every `__launch_bounds__` occurrence uses exactly these arguments.
        assert_eq!(
            src.matches("__launch_bounds__").count(),
            src.matches(&lb).count(),
            "a __launch_bounds__ occurrence uses arguments other than \
             (BLOCK_SIZE, PACKED_MIN_BLOCKS_PER_SM)"
        );
    }

    // rq-0314caab
    #[test]
    fn single_pair_entry_points_have_no_launch_bounds() {
        let src = composed_source_for_inspection();
        // Only the two packed-neighbour entry points carry launch bounds.
        assert_eq!(
            src.matches("__launch_bounds__").count(),
            2,
            "exactly the two packed-neighbour entry points carry launch bounds"
        );
        for name in [F_SINGLE_ENTRY, FEV_SINGLE_ENTRY] {
            assert!(
                src.contains(&format!("void {name}(")),
                "entry point {name} should be declared `void {name}(` with no launch bounds"
            );
        }
    }

    // rq-5214bef3
    #[test]
    fn composer_emits_no_correction_entry_points() {
        let src = composed_source_for_inspection();
        // No _correct_f / _correct_fev entry points appear anywhere in
        // the emitted source, and no per-pair `heddle_jit_eval_pair_correction`
        // function is emitted.
        for needle in ["_correct_f", "_correct_fev", "heddle_jit_eval_pair_correction"] {
            assert!(
                !src.contains(needle),
                "composed source unexpectedly contains `{needle}` — the correction pass is retired"
            );
        }
        // The composed module exposes exactly four extern "C" entry
        // points: the packed-neighbour `_f`/`_fev` and the single-pair
        // `_single_f`/`_single_fev`.
        let extern_c_count = src.matches("extern \"C\"").count();
        assert_eq!(
            extern_c_count, 4,
            "composed source must expose exactly four extern \"C\" entry points; got {extern_c_count}",
        );
        for name in [F_ENTRY, FEV_ENTRY, F_SINGLE_ENTRY, FEV_SINGLE_ENTRY] {
            assert!(
                src.contains(&format!("{name}(")),
                "expected entry point `{name}` in composed source"
            );
        }
    }

    fn minimal_fragment(label: &'static str) -> PairForceFragment {
        let name: &'static str = Box::leak(format!("XSF_{label}").into_boxed_str());
        let functor_source = format!(
            r#"
struct {n} {{
    __device__ inline Real cutoff_squared(unsigned int, unsigned int, unsigned int, unsigned int) const {{ return R(0.0); }}
    __device__ inline void evaluate(Real, Real, Real, Real, Real, unsigned int, unsigned int, unsigned int, unsigned int,
                                     Real &factor, Real &energy) const {{
        factor = R(0.0); energy = R(0.0);
    }}
    __device__ inline Real exclusion_scale(unsigned int, unsigned int) const {{ return R(1.0); }}
}};
"#,
            n = name
        );
        PairForceFragment {
            label,
            functor_struct_name: name,
            functor_source,
            entry_point_args: String::new(),
            functor_init_source: String::new(),
            cutoff: CutoffHandling::Uniform(1.0 as Real),
            consumes_type_index: false,
        }
    }

    // rq-b099ff28
    #[test]
    fn composer_emits_exclusion_scaled_evaluator() {
        // In all-pairs (Trivial) mode — no per-tile bitmask — the
        // per-pair evaluator calls every fragment's `exclusion_scale(i,
        // j)` and multiplies its `(factor, energy)` by that scale inline.
        // A single fragment is enough to observe one such call.
        let src = compose_source(&[minimal_fragment("a")], 1.0 as Real, false);
        assert!(
            src.contains("heddle_jit_eval_pair_sum"),
            "composer must emit the `heddle_jit_eval_pair_sum` evaluator"
        );
        assert!(
            src.contains(".exclusion_scale("),
            "evaluator must call `.exclusion_scale(i, j)` on the fragment"
        );
        assert!(
            src.contains("ex_scale"),
            "evaluator must multiply the fragment's contribution by the returned scale"
        );
    }

    // rq-a4b9e702 rq-b28a6d96 rq-8ae4a9f1 — in CellList mode the composer
    // emits a scale-free evaluator for the bulk and single-pair passes
    // (which never see a modified pair) plus a scale-aware evaluator used
    // by the exclusion-tile pass.
    #[test]
    fn bitmask_composer_emits_scale_free_and_scaled_evaluators() {
        let src = compose_source(&[minimal_fragment("a")], 1.0 as Real, true);
        // The scale-free evaluator's accumulate has no per-pair scale.
        assert!(
            src.contains("factor += s_factor;"),
            "CellList mode must emit a scale-free evaluator (bulk/single)"
        );
        // The scale-aware evaluator (for the exclusion-tile pass) applies
        // each fragment's exclusion_scale.
        assert!(
            src.contains("heddle_jit_eval_pair_sum_scaled"),
            "CellList mode must emit the scale-aware evaluator"
        );
        assert!(
            src.contains("factor += s_factor * ex_scale;"),
            "the scale-aware evaluator must multiply by exclusion_scale"
        );
        // The exclusion-tile pass exists and dispatches to the scale-aware
        // evaluator for its flagged pairs.
        assert!(
            src.contains("heddle_jit_excl_tile_loop")
                && src.contains("heddle_jit_eval_pair_sum_scaled<WriteEv>"),
            "the exclusion-tile pass must dispatch to the scale-aware evaluator"
        );
    }

    // rq-fa0b3d10 — the exclusion-tile pass and its entry points are
    // emitted only when the bitmask is active (CellList mode).
    #[test]
    fn excl_tile_pass_emitted_only_with_bitmask() {
        let with_mask = compose_source(&[minimal_fragment("a")], 1.0 as Real, true);
        assert!(
            with_mask.contains("heddle_jit_excl_tile_loop"),
            "bitmask mode must emit the exclusion-tile loop"
        );
        assert!(
            with_mask.contains("heddle_jit_composed_pair_force_excl_f")
                && with_mask.contains("heddle_jit_composed_pair_force_excl_fev"),
            "bitmask mode must emit both exclusion-tile entry points"
        );
        let without = compose_source(&[minimal_fragment("a")], 1.0 as Real, false);
        assert!(
            !without.contains("heddle_jit_excl_tile_loop"),
            "all-pairs mode must not emit the exclusion-tile loop"
        );
    }

    // rq-54aec894
    #[test]
    fn packed_neighbour_pass_dispatches_to_exclusion_scaled_evaluator() {
        let src = composed_source_for_inspection();
        // The packed-neighbour outer loop's inner body calls
        // `heddle_jit_eval_pair_sum`. Since the composer templates this
        // as `heddle_jit_eval_pair_sum<WriteEv>(...)`, look for that
        // dispatch substring.
        assert!(
            src.contains("heddle_jit_eval_pair_sum<WriteEv>"),
            "packed-neighbour outer loop must dispatch to `heddle_jit_eval_pair_sum<WriteEv>`"
        );
    }

    // rq-95f0812c
    #[test]
    fn single_pair_pass_dispatches_to_exclusion_scaled_evaluator() {
        let src = composed_source_for_inspection();
        // The single-pair outer loop lives in `SINGLE_PAIR_LOOP_TEMPLATE`
        // and dispatches the same evaluator. Confirm the evaluator name
        // appears in the file (both outer loops share it) and the
        // single-pair loop function is present.
        assert!(
            src.contains("heddle_jit_single_pair_loop"),
            "composer must emit `heddle_jit_single_pair_loop`"
        );
        assert!(
            src.contains("heddle_jit_eval_pair_sum<WriteEv>"),
            "single-pair loop must dispatch to `heddle_jit_eval_pair_sum<WriteEv>`"
        );
    }

    // rq-44385733
    #[test]
    fn force_field_has_no_excluded_pair_state() {
        // The `ForceField` struct exposes no field named
        // `excluded_pair_atoms` or `excluded_pair_count`. This test
        // enforces the invariant at compile time via a source
        // inspection: the shape check is not GPU-dependent so it can
        // live in a pure unit test.
        let src = std::fs::read_to_string("src/forces/mod.rs")
            .expect("src/forces/mod.rs is present for inspection");
        assert!(
            !src.contains("excluded_pair_atoms"),
            "src/forces/mod.rs unexpectedly references `excluded_pair_atoms`"
        );
        assert!(
            !src.contains("excluded_pair_count"),
            "src/forces/mod.rs unexpectedly references `excluded_pair_count`"
        );
    }

    // rq-c406ffcd
    #[test]
    fn book_documents_launch_configuration_constants() {
        let doc = std::fs::read_to_string("book/dev-notes/compile-time-constants.md")
            .expect("compile-time-constants book page exists");
        for needle in ["PACKED_MIN_BLOCKS_PER_SM", "BLOCK_SIZE", "WARPS_PER_BLOCK"] {
            assert!(doc.contains(needle), "book page must document {needle}");
        }
        assert!(
            doc.contains(&PACKED_MIN_BLOCKS_PER_SM.to_string()),
            "book page must state the PACKED_MIN_BLOCKS_PER_SM value"
        );
        let lower = doc.to_lowercase();
        assert!(
            lower.contains("build time") && lower.contains("not"),
            "book page must state the constants are build-time and not TOML-exposed"
        );
    }
}

#[cfg(test)]
mod type_index_amortization_tests {
    use super::*;

    // Minimal inspectable fragment. `consumes` drives whether the
    // composer emits the per-atom type-index load. The functor body
    // reads no per-atom data (so any `type_indices[...]` in the composed
    // source must come from the outer loop, not the functor).
    fn frag(label: &'static str, consumes: bool) -> PairForceFragment {
        let name: &'static str = Box::leak(format!("TF_{label}").into_boxed_str());
        let functor_source = format!(
            r#"
struct {n} {{
    __device__ inline Real cutoff_squared(unsigned int, unsigned int, unsigned int, unsigned int) const {{ return R(0.0); }}
    __device__ inline void evaluate(Real, Real, Real, Real, Real, unsigned int, unsigned int, unsigned int, unsigned int,
                                     Real &factor, Real &energy) const {{
        factor = R(0.0); energy = R(0.0);
    }}
    __device__ inline Real exclusion_scale(unsigned int, unsigned int) const {{ return R(1.0); }}
}};
"#,
            n = name
        );
        PairForceFragment {
            label,
            functor_struct_name: name,
            functor_source,
            entry_point_args: String::new(),
            functor_init_source: String::new(),
            cutoff: CutoffHandling::Uniform(1.0 as Real),
            consumes_type_index: consumes,
        }
    }

    // rq-b125bd5c
    #[test]
    fn evaluate_signature_carries_per_atom_types() {
        let src = compose_source(&[frag("a", true)], 1.0 as Real, false);
        // The shared evaluator helpers take i_type / j_type alongside qi / qj.
        assert!(src.contains(
            "Real qi, Real qj, unsigned int i_type, unsigned int j_type, \
             unsigned int i, unsigned int j,"
        ));
        // The per-fragment evaluate call threads the same i_type / j_type.
        assert!(src.contains(".evaluate(r2, inv_r, r, qi, qj, i_type, j_type, i, j,"));
    }

    // rq-b10f28d7
    #[test]
    fn consuming_fragment_loads_type_index_once_per_atom() {
        let src = compose_source(&[frag("a", true)], 1.0 as Real, false);
        // Outer loop loads both atoms' type index from the common buffer.
        assert!(src.contains("type_indices[i_atom_id]"));
        assert!(src.contains("type_indices[j_atom_id]"));
        // The j-side index is rotated through the diagonal shuffle.
        assert!(src.contains("__shfl_sync(0xFFFFFFFFu, j_type, src_lane)"));
        // All injection markers are resolved.
        for marker in [
            "/*HEDDLE_JIT_ITYPE_LOAD*/",
            "/*HEDDLE_JIT_JTYPE_LOAD*/",
            "/*HEDDLE_JIT_JTYPE_SHUFFLE*/",
            "/*HEDDLE_JIT_TYPE_LOAD_PERPAIR*/",
        ] {
            assert!(!src.contains(marker), "unresolved marker {marker}");
        }
    }

    // rq-61fa8b93
    #[test]
    fn type_index_load_elided_when_unconsumed() {
        let src = compose_source(&[frag("a", false)], 1.0 as Real, false);
        // No dereference of type_indices anywhere when no fragment consumes it.
        assert!(
            !src.contains("type_indices["),
            "type-index load must be elided when unconsumed"
        );
        // i_type / j_type are still declared and default to 0.
        assert!(src.contains("unsigned int i_type = 0u;"));
        assert!(src.contains("unsigned int j_type = 0u;"));
        // Markers are still resolved (to nothing).
        assert!(!src.contains("/*HEDDLE_JIT_ITYPE_LOAD*/"));
        assert!(!src.contains("/*HEDDLE_JIT_TYPE_LOAD_PERPAIR*/"));
    }

    // rq-c2b26c0c
    #[test]
    fn type_indices_is_a_common_argument() {
        // Present in the entry-point signatures whether or not a
        // fragment consumes it (it is a framework common argument).
        let consuming = compose_source(&[frag("a", true)], 1.0 as Real, false);
        let inert = compose_source(&[frag("a", false)], 1.0 as Real, false);
        let n_consuming = consuming.matches("const unsigned int *type_indices,").count();
        let n_inert = inert.matches("const unsigned int *type_indices,").count();
        // Four entry-point signatures (packed / single-pair, each _f
        // and _fev) plus their two loop-function signatures all
        // declare the common arg.
        assert!(
            n_consuming >= 4,
            "type_indices must be a common arg on every pair-force entry point"
        );
        // It is a common argument regardless of whether a fragment
        // consumes it — the count does not depend on consumption.
        assert_eq!(n_consuming, n_inert);
    }
}

#[cfg(test)]
mod virial_derivation_tests {
    use super::*;

    fn pair_frag(label: &'static str, name: &'static str) -> PairForceFragment {
        let functor_source = format!(
            r#"
struct {name} {{
    __device__ inline Real cutoff_squared(unsigned int, unsigned int, unsigned int, unsigned int) const {{ return R(0.0); }}
    __device__ inline void evaluate(Real, Real, Real, Real, Real, unsigned int, unsigned int, unsigned int, unsigned int,
                                     Real &factor, Real &energy) const {{ factor = R(0.0); energy = R(0.0); }}
    __device__ inline Real exclusion_scale(unsigned int, unsigned int) const {{ return R(1.0); }}
}};
"#
        );
        PairForceFragment {
            label,
            functor_struct_name: name,
            functor_source,
            entry_point_args: String::new(),
            functor_init_source: String::new(),
            cutoff: CutoffHandling::Uniform(1.0 as Real),
            consumes_type_index: false,
        }
    }

    // rq-27add068 rq-e7fc1920 — the scale-aware evaluator applies each
    // fragment's own exclusion_scale independently, so per-fragment
    // combinations (e.g. one fragment scaled 0, another 1) act
    // fragment-by-fragment rather than as a single shared scale.
    #[test] // rq-27add068 rq-e7fc1920
    fn scale_aware_evaluator_scales_each_fragment_independently() {
        // All-pairs mode emits the scale-aware `heddle_jit_eval_pair_sum`.
        let src = compose_source(
            &[pair_frag("a", "VFa"), pair_frag("b", "VFb")],
            1.0 as Real,
            false,
        );
        // Each fragment's contribution is multiplied by *its own*
        // functor's exclusion_scale — both appear, one per fragment.
        assert!(
            src.contains("composite.functor_a.exclusion_scale(i, j)"),
            "fragment a must be scaled by functor_a's exclusion_scale"
        );
        assert!(
            src.contains("composite.functor_b.exclusion_scale(i, j)"),
            "fragment b must be scaled by functor_b's exclusion_scale"
        );
        // And each fragment's evaluate feeds its own scaled accumulate
        // (no cross-fragment sharing of the scale).
        assert!(
            src.contains("composite.functor_a.evaluate(")
                && src.contains("composite.functor_b.evaluate("),
            "each fragment evaluates and scales through its own functor"
        );
    }

    // rq-7d64da58 — the pair composer derives the per-pair scalar virial
    // from the force factor; the functor emits only (factor, energy).
    #[test] // rq-7d64da58
    fn pair_composer_derives_virial_from_factor_and_r2() {
        let src = compose_source(&[pair_frag("a", "VFa")], 1.0 as Real, false);
        // The per-pair evaluator takes only (factor, energy) — no virial
        // out-parameter anywhere in the composed source.
        assert!(
            !src.contains("Real &virial"),
            "no fragment/evaluator may declare a virial out-parameter"
        );
        assert!(
            src.contains("Real &factor, Real &energy)"),
            "the per-pair evaluator must take (factor, energy)"
        );
        // The _fev accumulation derives the per-pair scalar virial as
        // factor * r2 from the masked, exclusion-scaled factor.
        assert!(
            src.contains("(factor * r2)"),
            "composer must derive the per-pair virial as factor * r2"
        );
    }

    // rq-ef17db0f — with several fragments, each adds into the shared
    // factor and the virial is derived once from the summed factor, so it
    // is the sum of the fragments' individual virials.
    #[test] // rq-ef17db0f
    fn pair_virial_is_derived_once_from_summed_factor() {
        let src = compose_source(&[pair_frag("a", "VFa"), pair_frag("b", "VFb")], 1.0 as Real, false);
        assert!(
            src.matches("factor += s_factor * ex_scale;").count() >= 2,
            "each active fragment must add its factor into the shared per-pair factor"
        );
        assert!(
            src.contains("(factor * r2)"),
            "the per-pair virial is derived once from the summed factor"
        );
    }

    fn bonded_frag() -> BondedForceFragment {
        BondedForceFragment {
            label: "bf",
            functor_struct_name: "BF",
            functor_source: r#"
struct BF {
    __device__ inline void evaluate(Real, Real, unsigned int, Real, Real, Real, Real &fmag, Real &u_k) const { fmag = R(0.0); u_k = R(0.0); }
};
"#
            .to_string(),
            entry_point_args: String::new(),
            functor_init_source: String::new(),
        }
    }

    // rq-ff5a04bc — a bonded functor emits no virial; the composer derives
    // W_k = fmag * r2.
    #[test] // rq-ff5a04bc
    fn bonded_composer_derives_virial_from_fmag_and_r2() {
        let src = compose_bonded_source(&[bonded_frag()]);
        assert!(
            src.contains("functor.evaluate(r2, r, type_idx, dx, dy, dz, fmag, u_k);"),
            "the bonded functor call must take (fmag, u_k) with no virial out-parameter"
        );
        assert!(
            src.contains("Real w_k = fmag * r2;"),
            "the composer must derive the bond virial W_k = fmag * r2"
        );
    }

    fn angle_frag() -> AngleForceFragment {
        AngleForceFragment {
            label: "af",
            functor_struct_name: "AF",
            functor_source: r#"
struct AF {
    __device__ inline void evaluate(Real, Real, Real, Real, Real, Real, unsigned int,
        Real &fix, Real &fiy, Real &fiz, Real &fkx, Real &fky, Real &fkz, Real &u_m) const {
        fix = R(0.0); fiy = R(0.0); fiz = R(0.0); fkx = R(0.0); fky = R(0.0); fkz = R(0.0); u_m = R(0.0);
    }
};
"#
            .to_string(),
            entry_point_args: String::new(),
            functor_init_source: String::new(),
        }
    }

    // rq-a997a963 — an angle functor emits no virial; the composer derives
    // W_m = r_ij·F_i + r_kj·F_k from the returned forces.
    #[test] // rq-a997a963
    fn angle_composer_derives_virial_from_forces() {
        let src = compose_angle_source(&[angle_frag()]);
        assert!(
            !src.contains(", u_m, w_m);"),
            "the angle functor call must take u_m with no virial out-parameter"
        );
        assert!(
            src.contains("Real w_m = dx_ij * fix + dy_ij * fiy + dz_ij * fiz"),
            "the composer must derive the angle virial from the leg displacements and forces"
        );
    }

    fn dihedral_frag() -> DihedralForceFragment {
        DihedralForceFragment {
            label: "df",
            functor_struct_name: "DF",
            functor_source: r#"
struct DF {
    __device__ inline void evaluate(Real, Real, Real, Real, Real, Real, Real, Real, Real, unsigned int,
        Real &fix, Real &fiy, Real &fiz, Real &fjx, Real &fjy, Real &fjz,
        Real &fkx, Real &fky, Real &fkz, Real &flx, Real &fly, Real &flz, Real &u_m) const {
        fix = R(0.0); fiy = R(0.0); fiz = R(0.0); fjx = R(0.0); fjy = R(0.0); fjz = R(0.0);
        fkx = R(0.0); fky = R(0.0); fkz = R(0.0); flx = R(0.0); fly = R(0.0); flz = R(0.0); u_m = R(0.0);
    }
};
"#
            .to_string(),
            entry_point_args: String::new(),
            functor_init_source: String::new(),
        }
    }

    // rq-932d37a2 — a dihedral functor emits no virial; the composer
    // derives W_m from the four forces and the bond displacements.
    #[test] // rq-932d37a2
    fn dihedral_composer_derives_virial_from_forces() {
        let src = compose_dihedral_source(&[dihedral_frag()]);
        assert!(
            !src.contains(", u_m, w_m);"),
            "the dihedral functor call must take u_m with no virial out-parameter"
        );
        assert!(
            src.contains("Real dx_lj = dx_kj - dx_kl;"),
            "the composer must form r_l - r_j = dx_kj - dx_kl"
        );
        assert!(
            src.contains("Real w_m = dx_ij * fix + dy_ij * fiy + dz_ij * fiz"),
            "the composer must derive the dihedral virial from forces and displacements"
        );
    }
}
