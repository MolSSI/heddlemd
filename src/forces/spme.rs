// rq-202493a5 rq-9ca00d25
//! SPME reciprocal-space pipeline (PR 1 scope): charge spreading,
//! forward FFT, influence-function multiply, inverse FFT. Owned grid
//! buffers, cuFFT plans, precomputed influence function, and a bin-only
//! `NeighborListState` used by the spread kernel.
//!
//! Force gather, real-space `erfc` slot, self-energy, and ForceField
//! integration are out of scope for this module; they land in a
//! subsequent PR.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtrMut, DeviceSlice};
use cudarc::nvrtc::Ptx;

use crate::gpu::cufft::{CuFftError, Plan3dC2R, Plan3dR2C};
use crate::gpu::device::get_func;
use crate::gpu::{
    GpuContext, GpuError, K_COULOMB_F32, ParticleBuffers,
    spme_atom_sort, spme_force_gather, spme_real_pair_force,
};
use crate::kernels;
use crate::io::config::SpmeConfig;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::topology::{DeviceExclusionList, ExclusionList};
use super::neighbor_list::{alloc_scan_block_totals, NeighborListError};
use super::{
    AggregateLevel, ForceFieldContext, ForceFieldError, PairForceBindContext,
    PairForceFragment, PairForceLaunchBuilder, Potential, PotentialBuildContext,
    PotentialBuilder, SlotOutputView,
};
use crate::precision::Real;

// rq-7bd2d9ca
#[derive(Debug, Clone, Copy)]
pub struct SpmeParameters {
    pub alpha: Real,
    pub r_cut_real: Real,
    pub grid: [u32; 3],
    pub spline_order: u32,
}

impl From<&SpmeConfig> for SpmeParameters {
    fn from(c: &SpmeConfig) -> Self {
        SpmeParameters {
            alpha: c.alpha as Real,
            r_cut_real: c.r_cut_real as Real,
            grid: c.grid,
            spline_order: c.spline_order,
        }
    }
}

// rq-846bdb8b rq-ebfa6e1f
#[derive(Debug, thiserror::Error)]
pub enum SpmeError {
    #[error("{0}")]
    NeighborList(#[from] NeighborListError),
    #[error("{0}")]
    CuFft(#[from] CuFftError),
    #[error("SPME grid[{axis}] = {n} is less than 2 * spline_order = {required}")]
    InvalidGrid {
        axis: &'static str,
        n: u32,
        required: u32,
    },
    #[error("{0}")]
    Gpu(#[from] GpuError),
}

/// PR 1 scope: owns the FFT grid buffers, cuFFT plans, precomputed
/// influence function, and a bin-only NeighborListState. Per-step
/// `compute()` runs spread → forward FFT → influence multiply →
/// inverse FFT, leaving `V[g]` on the device for downstream gather
/// (PR 2).
#[derive(Debug)]
pub struct SpmeReciprocalGrid {
    #[allow(dead_code)]
    pub device: Arc<CudaDevice>,
    pub params: SpmeParameters,
    pub particle_count: usize,
    pub m: usize,           // n_a * n_b * n_c (real-grid size)
    pub m_complex: usize,   // n_a * n_b * (n_c/2 + 1)
    /// Fixed-point charge-density grid. `spme_spread_fixed_point`
    /// accumulates per-particle contributions via `atomicAdd<i64>`
    /// using the scale `2^32`; `spme_spread_finish` converts back to
    /// f32 `rho` via the inverse scale. Zeroed before each step's
    /// spread.
    pub rho_fixed: CudaSlice<i64>,
    pub rho: CudaSlice<Real>,
    pub rho_hat_interleaved: CudaSlice<Real>,
    pub v: CudaSlice<Real>,
    pub influence_g: CudaSlice<Real>,
    /// Per-cell factor `G[k] · (1 − K²/(2α²))`. Read per-thread by
    /// `spme_recip_apply_influence`, multiplied by `|rho_hat[k]|²` and
    /// the Hermitian weight, and fed into the per-block reduction that
    /// writes `virial_partials`.
    pub virial_factor: CudaSlice<Real>,
    /// Per-block partial sums of the reciprocal-space virial
    /// contribution, written one entry per launch block of
    /// `spme_recip_apply_influence`. Length equals the launch's grid
    /// size (`ceil(m_complex / 256)`). Reduced on device to the scalar
    /// `w_per_particle_virial = W_recip / N` by
    /// `spme_recip_reduce_partials`; never copied to host on the
    /// per-step path.
    pub virial_partials: CudaSlice<Real>,
    /// Per-axis B-spline correction factors. Depend only on
    /// `(grid, spline_order)`, populated once at construction from the
    /// host-side Cox-de Boor recursion, never re-uploaded.
    pub b_factors_a: CudaSlice<Real>,
    pub b_factors_b: CudaSlice<Real>,
    pub b_factors_c: CudaSlice<Real>,
    /// Box generation the influence function was computed against.
    /// Refreshed when the sim_box generation changes.
    pub cached_box_generation: u64,
    pub forward_plan: Plan3dR2C,
    pub inverse_plan: Plan3dC2R,
    /// Device-resident work area shared by the R2C and C2R cuFFT plans.
    /// Sized to the larger of the two plans' `work_size()` requests;
    /// both plans bind to this pointer at construction via
    /// `cufftSetWorkArea`. The fixed pointer makes captured
    /// `cufftExec*` calls safe to replay inside a CUDA graph.
    pub workspace: CudaSlice<u8>,
    /// SPME primary-bin index per atom; written by
    /// `spme_compute_bin_key` and consumed by the scatter stage.
    pub atom_bin_key: CudaSlice<u32>,
    /// Per-bin atom histogram. Zeroed before each sort, accumulated
    /// via `atomicAdd` inside `spme_compute_bin_key`.
    pub bin_atom_counts: CudaSlice<u32>,
    /// Exclusive prefix scan of `bin_atom_counts`; bin `b` occupies
    /// sorted-index positions `[bin_atom_offsets[b], bin_atom_offsets[b + 1])`.
    pub bin_atom_offsets: CudaSlice<u32>,
    /// Per-bin scatter cursor. Zeroed before each sort, incremented
    /// atomically inside `scatter_atoms_into_cells` (reused).
    pub bin_atom_cursor: CudaSlice<u32>,
    /// The sorted permutation. Entry `t` names the original atom index
    /// processed at sorted slot `t`. Consumed by
    /// `spme_spread_fixed_point` and `spme_force_gather`. Initialised
    /// to the identity permutation at construction so the very first
    /// `compute()` works even before the first sort runs.
    pub sorted_atom_index: CudaSlice<u32>,
    /// Multi-level scan-stack buffers consumed by
    /// `prefix_scan_cell_counts` when operating on a histogram of
    /// length `M`.
    pub sort_scan_block_totals: Vec<CudaSlice<u32>>,
    /// The neighbour-list rebuild generation observed at the last
    /// sort. The slot re-runs the sort pipeline when the framework
    /// reports a generation strictly greater than this value.
    pub cached_neighbor_list_generation: u64,
}

impl SpmeReciprocalGrid {
    pub fn new(
        gpu: &GpuContext,
        sim_box: &SimulationBox,
        particle_count: usize,
        params: SpmeParameters,
    ) -> Result<Self, SpmeError> {
        let n_a = params.grid[0];
        let n_b = params.grid[1];
        let n_c = params.grid[2];
        let p = params.spline_order;
        let required = 2 * p;
        let axis_names = ["a", "b", "c"];
        for (i, &n) in params.grid.iter().enumerate() {
            if n < required {
                return Err(SpmeError::InvalidGrid {
                    axis: axis_names[i],
                    n,
                    required,
                });
            }
        }

        let m = n_a as usize * n_b as usize * n_c as usize;
        let m_complex = n_a as usize * n_b as usize * (n_c as usize / 2 + 1);
        let device = gpu.device.clone();

        // Fixed-point charge-density grid. Zeroed at construction and
        // cleared via `memset_zeros` before each step's spread.
        let rho_fixed = device.alloc_zeros::<i64>(m).map_err(GpuError::from)?;

        let rho = device.alloc_zeros::<Real>(m).map_err(GpuError::from)?;
        let v = device.alloc_zeros::<Real>(m).map_err(GpuError::from)?;
        let rho_hat_interleaved = device
            .alloc_zeros::<Real>(2 * m_complex)
            .map_err(GpuError::from)?;

        // B-spline correction factors depend only on (grid, spline_order)
        // — compute on host, upload once, never refresh.
        let b_factors_a_host = compute_b_factors(n_a, p);
        let b_factors_b_host = compute_b_factors(n_b, p);
        let b_factors_c_host = compute_b_factors(n_c, p);
        let b_factors_a = device
            .htod_sync_copy(&b_factors_a_host)
            .map_err(GpuError::from)?;
        let b_factors_b = device
            .htod_sync_copy(&b_factors_b_host)
            .map_err(GpuError::from)?;
        let b_factors_c = device
            .htod_sync_copy(&b_factors_c_host)
            .map_err(GpuError::from)?;
        let mut influence_g = device
            .alloc_zeros::<Real>(m_complex)
            .map_err(GpuError::from)?;
        let mut virial_factor = device
            .alloc_zeros::<Real>(m_complex)
            .map_err(GpuError::from)?;
        // Sized to the per-step `spme_recip_apply_influence` grid: one
        // partial per launch block. Block size 256 matches the
        // shared-memory tree shape inside the kernel.
        let num_partial_blocks = m_complex.div_ceil(256);
        let virial_partials = device
            .alloc_zeros::<Real>(num_partial_blocks)
            .map_err(GpuError::from)?;

        // Atom spatial pre-sort scratch. `sorted_atom_index` is
        // initialised to the identity permutation so the very first
        // `compute()` call can run the spread / gather kernels even
        // before the first sort completes.
        let n = particle_count;
        let atom_bin_key = device.alloc_zeros::<u32>(n.max(1)).map_err(GpuError::from)?;
        let bin_atom_counts = device.alloc_zeros::<u32>(m).map_err(GpuError::from)?;
        let bin_atom_offsets = device.alloc_zeros::<u32>(m + 1).map_err(GpuError::from)?;
        let bin_atom_cursor = device.alloc_zeros::<u32>(m).map_err(GpuError::from)?;
        let sorted_atom_index = if n == 0 {
            device.alloc_zeros::<u32>(0).map_err(GpuError::from)?
        } else {
            let identity: Vec<u32> = (0..n as u32).collect();
            device.htod_sync_copy(&identity).map_err(GpuError::from)?
        };
        let sort_scan_block_totals = alloc_scan_block_totals(&device, m)?;

        let forward_plan = Plan3dR2C::new_unallocated(&device, n_a, n_b, n_c)?;
        let inverse_plan = Plan3dC2R::new_unallocated(&device, n_a, n_b, n_c)?;

        // Bind both cuFFT plans to the device's default stream. Without
        // this, cuFFT runs on the legacy NULL stream — but when
        // `init_device` uses `CudaDevice::new_with_stream` the device's
        // default stream is non-NULL, and kernels launching on it
        // would not be visible to cuFFT (and vice versa).
        let dev_stream = *device.cu_stream();
        forward_plan.set_stream(dev_stream)?;
        inverse_plan.set_stream(dev_stream)?;

        // Allocate a single device-resident work area sized to the
        // larger of the two plans' requested sizes; bind both plans to
        // it. The plans share the buffer because their executions are
        // strictly serialised on the default stream. The pinned
        // pointer is a prerequisite for the captured `cufftExec*`
        // calls to be safe across CUDA graph replays.
        let work_size = forward_plan.work_size()?.max(inverse_plan.work_size()?);
        let workspace_len = work_size.max(1);
        let mut workspace = device
            .alloc_zeros::<u8>(workspace_len)
            .map_err(GpuError::from)?;
        let work_ptr = (*workspace.device_ptr_mut()) as *mut std::ffi::c_void;
        forward_plan.set_work_area(work_ptr)?;
        inverse_plan.set_work_area(work_ptr)?;

        // Populate `influence_g` and `virial_factor` for the initial
        // box. The launch is async on the default stream; downstream
        // consumers in `compute()` read from the same stream and
        // observe the writes without additional synchronisation.
        crate::gpu::spme_recip_compute_influence(
            &gpu.kernels,
            &b_factors_a,
            &b_factors_b,
            &b_factors_c,
            &mut influence_g,
            &mut virial_factor,
            sim_box,
            params.grid,
            K_COULOMB_F32,
            params.alpha,
            m_complex as u32,
        )?;

        Ok(SpmeReciprocalGrid {
            device,
            params,
            particle_count,
            m,
            m_complex,
            rho_fixed,
            rho,
            rho_hat_interleaved,
            v,
            influence_g,
            virial_factor,
            virial_partials,
            b_factors_a,
            b_factors_b,
            b_factors_c,
            cached_box_generation: sim_box.generation(),
            forward_plan,
            inverse_plan,
            workspace,
            atom_bin_key,
            bin_atom_counts,
            bin_atom_offsets,
            bin_atom_cursor,
            sorted_atom_index,
            sort_scan_block_totals,
            cached_neighbor_list_generation: 0,
        })
    }

    /// Run the per-step reciprocal-space pipeline:
    ///   sort (when triggered) → spread → forward FFT → influence
    ///   multiply → inverse FFT.
    /// On return, `self.v` holds the smoothed potential V[g] (with
    /// `k_C/V_box` and `|b|²` baked in via the influence function);
    /// `self.rho` holds the charge density rho[g].
    ///
    /// `neighbor_list_generation` is the framework's monotonic
    /// rebuild-counter value (`NeighborListState::rebuild_generation()`).
    /// The slot re-runs the atom spatial pre-sort when this value
    /// strictly exceeds the slot's cached generation; otherwise the
    /// prior sort permutation is reused.
    pub fn compute(
        &mut self,
        sim_box: &SimulationBox,
        particle_buffers: &ParticleBuffers,
        neighbor_list_generation: u64,
        timings: &mut Timings,
    ) -> Result<(), SpmeError> {
        // Refresh influence function (and the virial factor that
        // tracks it) on device when the box has changed. All recip
        // launches share the default stream, so downstream consumers
        // see the refreshed buffers without additional synchronisation.
        if sim_box.generation() != self.cached_box_generation {
            crate::gpu::spme_recip_compute_influence(
                &particle_buffers.kernels,
                &self.b_factors_a,
                &self.b_factors_b,
                &self.b_factors_c,
                &mut self.influence_g,
                &mut self.virial_factor,
                sim_box,
                self.params.grid,
                K_COULOMB_F32,
                self.params.alpha,
                self.m_complex as u32,
            )?;
            self.cached_box_generation = sim_box.generation();
        }

        // Atom spatial pre-sort. Triggered when the framework's
        // neighbour-list rebuild generation advances past the slot's
        // cached value. Concentrates the spread's atomicAdd writes and
        // the gather's V[g] reads on neighbouring cache lines.
        if neighbor_list_generation > self.cached_neighbor_list_generation
            && self.particle_count > 0
        {
            spme_atom_sort(
                particle_buffers,
                sim_box,
                self.params.grid,
                &mut self.atom_bin_key,
                &mut self.bin_atom_counts,
                &mut self.bin_atom_offsets,
                &mut self.bin_atom_cursor,
                &mut self.sorted_atom_index,
                &mut self.sort_scan_block_totals,
            )?;
            self.cached_neighbor_list_generation = neighbor_list_generation;
        }

        // Charge spread (fixed-point atomic-add pipeline).
        let _ = timings;
        // Always zero rho_fixed so the per-step atomicAdd<i64> accumulates
        // a clean state. Particle-count == 0 still produces a correct
        // all-zero rho via spme_spread_finish.
        self.device
            .memset_zeros(&mut self.rho_fixed)
            .map_err(GpuError::from)?;
        if self.particle_count > 0 {
            crate::gpu::spme_spread_fixed_point(
                particle_buffers,
                &self.sorted_atom_index,
                sim_box,
                self.params.grid,
                self.params.spline_order,
                &mut self.rho_fixed,
            )?;
        }
        crate::gpu::spme_spread_finish(
            &particle_buffers.kernels,
            &self.rho_fixed,
            &mut self.rho,
            self.m as u32,
        )?;
        self.forward_plan
            .execute(&self.rho, &mut self.rho_hat_interleaved)?;

        let n_c = self.params.grid[2];
        let n_c_complex = (n_c / 2 + 1) as u32;
        crate::gpu::spme_recip_apply_influence(
            &particle_buffers.kernels,
            &self.influence_g,
            &self.virial_factor,
            &mut self.rho_hat_interleaved,
            &mut self.virial_partials,
            n_c,
            n_c_complex,
            self.m_complex as u32,
        )?;

        self.inverse_plan
            .execute(&self.rho_hat_interleaved, &mut self.v)?;

        Ok(())
    }

    /// Synchronises the device. Production paths do not need to call
    /// this — the per-stream ordering of the default stream makes
    /// `rho` / `v` / `virial_partials` visible to any subsequent
    /// kernel or dtoh on the same stream. Tests that read these
    /// buffers via the host call this for clarity (a subsequent
    /// `dtoh_sync_copy` already synchronises, so the call is logically
    /// redundant but kept as a no-cost diagnostic).
    pub fn sync_recip(&self) -> Result<(), GpuError> {
        self.device.synchronize().map_err(GpuError::from)
    }
}

// rq-9ca00d25
//
// SPME B-spline structure-factor correction. For axis with grid size N
// and B-spline order p, `b_factors[k] = |b(k)|²` where
//   b(k) = exp(2π i (p-1) k / N) / Σ_{j=0..p-2} M_p(j+1) · exp(2π i j k / N)
// and |b(k)|² = 1 / |denominator|².
pub fn compute_b_factors(n: u32, p: u32) -> Vec<Real> {
    let n = n as usize;
    let p = p as usize;
    let mut out = vec![0.0; n];
    let two_pi = 2.0 * std::f64::consts::PI;
    let m_p_samples: Vec<f64> = (1..p).map(|j| cardinal_bspline(p, j as f64)).collect();
    for k in 0..n {
        let theta = two_pi * (k as f64) / (n as f64);
        let mut sum_re = 0.0_f64;
        let mut sum_im = 0.0_f64;
        for (j, &m_val) in m_p_samples.iter().enumerate() {
            let angle = theta * (j as f64);
            sum_re += m_val * angle.cos();
            sum_im += m_val * angle.sin();
        }
        let denom2 = sum_re * sum_re + sum_im * sum_im;
        out[k] = if denom2 > 0.0 {
            (1.0 / denom2) as Real
        } else {
            0.0
        };
    }
    out
}

// Cardinal B-spline M_p(x) via the Cox-de Boor recursion, host-side.
fn cardinal_bspline(p: usize, x: f64) -> f64 {
    let mut vals: Vec<f64> = (0..p)
        .map(|i| {
            let xi = x - (i as f64);
            if xi >= 0.0 && xi < 1.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    for order in 2..=p {
        let inv = 1.0 / (order as f64 - 1.0);
        for i in 0..(p - order + 1) {
            let xi = x - i as f64;
            vals[i] =
                xi * inv * vals[i] + ((order as f64) - xi) * inv * vals[i + 1];
        }
    }
    vals[0]
}

// rq-f6d45062
//
// Real-space `erfc`-screened pair-force slot. Structurally analogous to
// Holds the per-pair-type parameters and exclusion list; the fused
// pair-force kernel runs directly inside `compute()` and writes the
// per-particle output to the SlotOutputView.
// rq-22171569
#[derive(Debug)]
pub struct SpmeRealSpaceState {
    #[allow(dead_code)]
    device: Arc<CudaDevice>,
    exclusions: DeviceExclusionList,
    alpha: Real,
    r_cut_real: Real,
    particle_count: usize,
    max_neighbors: u32,
}

impl SpmeRealSpaceState {
    pub fn new(
        gpu: &GpuContext,
        particle_count: usize,
        alpha: Real,
        r_cut_real: Real,
        max_neighbors: u32,
        exclusion_list: &ExclusionList,
    ) -> Result<Self, NeighborListError> {
        let device = gpu.device.clone();
        let exclusions = DeviceExclusionList::from_host(&device, exclusion_list)?;
        Ok(SpmeRealSpaceState {
            device,
            exclusions,
            alpha,
            r_cut_real,
            particle_count,
            max_neighbors,
        })
    }
}

impl Potential for SpmeRealSpaceState {
    fn label(&self) -> &'static str {
        "spme_real"
    }

    fn max_cutoff(&self) -> Option<Real> {
        Some(self.r_cut_real)
    }

    fn compute(
        &mut self,
        buffers: &ParticleBuffers,
        sim_box: &SimulationBox,
        mut output: SlotOutputView<'_>,
        cx: &ForceFieldContext<'_>,
        timings: &mut Timings,
        level: AggregateLevel,
    ) -> Result<(), ForceFieldError> {
        if self.particle_count == 0 {
            return Ok(());
        }
        let nl = cx
            .neighbor_list
            .expect("SpmeRealSpaceState requires a shared neighbor list");
        timings.kernel_start(KernelStage::SPME_REAL_PAIR_FORCE)?;
        spme_real_pair_force(
            buffers,
            &mut output,
            sim_box,
            self.alpha,
            self.r_cut_real,
            &self.exclusions.atom_excl_offsets,
            &self.exclusions.atom_excl_partners,
            &self.exclusions.atom_excl_coul_scales,
            &nl.neighbor_list,
            &nl.neighbor_counts,
            self.max_neighbors,
            level,
        )?;
        timings.kernel_stop(KernelStage::SPME_REAL_PAIR_FORCE)?;
        Ok(())
    }

    fn bind_pair_force_args(
        &self,
        ctx: &PairForceBindContext<'_>,
        builder: &mut PairForceLaunchBuilder,
    ) {
        builder.push_device_buffer(&ctx.buffers.charges);
        builder.push_scalar(K_COULOMB_F32);
        builder.push_scalar(self.alpha);
        builder.push_scalar(self.r_cut_real);
        builder.push_device_buffer(&self.exclusions.atom_excl_offsets);
        builder.push_device_buffer(&self.exclusions.atom_excl_partners);
        builder.push_device_buffer(&self.exclusions.atom_excl_coul_scales);
    }
}

/// SPME real-space `erfc`-screened pair force fragment for the
/// JIT-composed pair-force kernel.
pub fn spme_real_pair_force_fragment() -> PairForceFragment {
    let functor_source = r#"
struct SpmeRealPairFunctor {
    const Real *charges;
    Real k_coulomb;
    Real alpha;
    Real r_cut_real;
    const unsigned int *excl_offsets;
    const unsigned int *excl_partners;
    const Real *excl_scales;

    __device__ inline Real cutoff_squared(unsigned int, unsigned int) const {
        return r_cut_real * r_cut_real;
    }

    __device__ inline void evaluate(
        Real r2, unsigned int i, unsigned int j,
        Real &factor, Real &energy, Real &virial) const
    {
        Real qi = charges[i];
        Real qj = charges[j];
        Real qq = qi * qj;
        Real inv_r2 = R(1.0) / r2;
        Real inv_r  = Real_sqrt(inv_r2);
        Real r      = R(1.0) / inv_r;
        Real ar = alpha * r;
        Real erfc_ar = Real_erfc(ar);
        Real gauss = Real_exp(-(ar * ar));
        Real one_over_sqrt_pi = R(0.5641895835477563);
        energy = k_coulomb * qq * erfc_ar * inv_r;
        factor = k_coulomb * qq * inv_r2
                 * (erfc_ar * inv_r + R(2.0) * alpha * one_over_sqrt_pi * gauss);
        virial = factor * r2;
    }

    __device__ inline Real exclusion_scale(unsigned int i, unsigned int j) const {
        return heddle_jit_exclusion_scale(i, j, excl_offsets, excl_partners, excl_scales);
    }
};
"#;
    let entry_point_args = r#"    const Real *spme_real_charges,
    Real spme_real_k_coulomb,
    Real spme_real_alpha,
    Real spme_real_r_cut,
    const unsigned int *spme_real_excl_offsets,
    const unsigned int *spme_real_excl_partners,
    const Real *spme_real_excl_scales,
"#;
    let functor_init_source = r#"    composite.functor_spme_real.charges = spme_real_charges;
    composite.functor_spme_real.k_coulomb = spme_real_k_coulomb;
    composite.functor_spme_real.alpha = spme_real_alpha;
    composite.functor_spme_real.r_cut_real = spme_real_r_cut;
    composite.functor_spme_real.excl_offsets = spme_real_excl_offsets;
    composite.functor_spme_real.excl_partners = spme_real_excl_partners;
    composite.functor_spme_real.excl_scales = spme_real_excl_scales;
"#;
    PairForceFragment {
        label: "spme_real",
        functor_struct_name: "SpmeRealPairFunctor",
        functor_source: functor_source.to_string(),
        entry_point_args: entry_point_args.to_string(),
        functor_init_source: functor_init_source.to_string(),
    }
}

// rq-202493a5 rq-9ca00d25
//
// Reciprocal-space slot wrapping `SpmeReciprocalGrid` plus a per-
// particle self-energy buffer. `contribute()` runs the spread / FFT /
// multiply / inverse-FFT pipeline and reduces the per-cell virial to a
// scalar; `reduce()` runs the force-gather kernel which writes per-
// particle force, energy (with self-energy subtracted), and the
// uniform per-particle virial share.
// rq-b1148667
#[derive(Debug)]
pub struct SpmeReciprocalState {
    grid: SpmeReciprocalGrid,
    // `u_self_per_particle[i] = k_C · (α/√π) · q_i²`. Subtracted from
    // the per-particle reciprocal energy inside the gather kernel.
    u_self_per_particle: CudaSlice<Real>,
    // Reduced per-particle reciprocal virial share, computed by
    // `spme_recip_reduce_partials` from `SpmeReciprocalGrid::virial_partials`
    // and read by the force-gather kernel on the same stream.
    w_per_particle_virial: CudaSlice<Real>,
}

impl SpmeReciprocalState {
    pub fn new(
        gpu: &GpuContext,
        sim_box: &SimulationBox,
        particle_count: usize,
        charges: &[Real],
        params: SpmeParameters,
    ) -> Result<Self, SpmeError> {
        let grid = SpmeReciprocalGrid::new(gpu, sim_box, particle_count, params)?;
        // Precompute per-particle self-energy:
        //   u_self_i = k_C · (α/√π) · q_i²
        let inv_sqrt_pi = 1.0_f64 / std::f64::consts::PI.sqrt();
        let prefactor = (K_COULOMB_F32 as f64) * (params.alpha as f64) * inv_sqrt_pi;
        let u_self_host: Vec<Real> = (0..particle_count)
            .map(|i| {
                let q = charges.get(i).copied().unwrap_or(0.0);
                (prefactor * (q as f64) * (q as f64)) as Real
            })
            .collect();
        let u_self_per_particle = if particle_count == 0 {
            grid.device.alloc_zeros::<Real>(0).map_err(GpuError::from)?
        } else {
            grid.device
                .htod_sync_copy(&u_self_host)
                .map_err(GpuError::from)?
        };
        let w_per_particle_virial =
            grid.device.alloc_zeros::<Real>(1).map_err(GpuError::from)?;
        Ok(SpmeReciprocalState {
            grid,
            u_self_per_particle,
            w_per_particle_virial,
        })
    }

    // Test access to the underlying grid (rho/V buffers, influence_g,
    // etc.). Used by `tests/spme_pipeline.rs`.
    pub fn grid(&self) -> &SpmeReciprocalGrid {
        &self.grid
    }
}

impl Potential for SpmeReciprocalState {
    fn label(&self) -> &'static str {
        "spme_reciprocal"
    }

    fn max_cutoff(&self) -> Option<Real> {
        // The reciprocal-space slot does not consume the shared
        // neighbor list; it owns its own bin structure internally.
        None
    }

    // rq-df6d79a1
    fn frequency_class(&self) -> super::ForceClass {
        super::ForceClass::Slow
    }

    fn compute(
        &mut self,
        buffers: &ParticleBuffers,
        sim_box: &SimulationBox,
        mut output: SlotOutputView<'_>,
        cx: &ForceFieldContext<'_>,
        timings: &mut Timings,
        _level: AggregateLevel,
    ) -> Result<(), ForceFieldError> {
        let n = self.grid.particle_count;
        if n == 0 {
            return Ok(());
        }
        // When the framework has a shared neighbour list (the common
        // case — SPME requires the real-space slot whose r_cut sizes
        // it), use its rebuild generation as the sort trigger. When
        // no neighbour list exists, keep the construction-time
        // identity permutation forever (correct, but no cache-locality
        // benefit).
        let neighbor_list_generation = cx
            .neighbor_list
            .map(|nl| nl.rebuild_generation())
            .unwrap_or(0);
        timings.kernel_start(KernelStage::SPME_RECIP_PIPELINE)?;
        self.grid
            .compute(sim_box, buffers, neighbor_list_generation, timings)
            .map_err(map_spme_err)?;
        crate::gpu::spme_recip_reduce_partials(
            &buffers.kernels,
            &self.grid.virial_partials,
            &mut self.w_per_particle_virial,
            self.grid.virial_partials.len() as u32,
            n as u32,
        )?;
        timings.kernel_stop(KernelStage::SPME_RECIP_PIPELINE)?;

        timings.kernel_start(KernelStage::SPME_FORCE_GATHER)?;
        spme_force_gather(
            buffers,
            &self.grid.sorted_atom_index,
            sim_box,
            &self.grid.v,
            &self.u_self_per_particle,
            &self.w_per_particle_virial,
            self.grid.params.grid,
            self.grid.params.spline_order,
            &mut output.force_x,
            &mut output.force_y,
            &mut output.force_z,
            &mut output.energy,
            &mut output.virial,
        )?;
        timings.kernel_stop(KernelStage::SPME_FORCE_GATHER)?;
        Ok(())
    }
}

fn map_spme_err(e: SpmeError) -> ForceFieldError {
    match e {
        SpmeError::Gpu(g) => ForceFieldError::Gpu(g),
        SpmeError::NeighborList(n) => ForceFieldError::NeighborList(n),
        // The other variants are construction-time errors that should
        // not surface during a step. If one ever does, panic loudly
        // rather than silently mapping to a generic GPU error: it
        // indicates a config-layer invariant has been violated after
        // setup.
        SpmeError::CuFft(other) => {
            panic!("SPME step encountered cuFFT error: {other:?}")
        }
        SpmeError::InvalidGrid { axis, n, required } => panic!(
            "SPME step encountered invalid grid (axis {axis}, n {n}, required {required})"
        ),
    }
}

// rq-e8550f96
#[derive(Debug, Clone)]
pub struct SpmeRealBuilder;

impl PotentialBuilder for SpmeRealBuilder {
    fn build(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        let Some(spme_cfg) = cx.spme_config else {
            return Ok(None);
        };
        let params = SpmeParameters::from(spme_cfg);
        let max_neighbors = super::max_neighbors_from(cx.neighbor_list_config, cx.particle_count);
        let state = SpmeRealSpaceState::new(
            cx.gpu,
            cx.particle_count,
            params.alpha,
            params.r_cut_real,
            max_neighbors,
            cx.exclusion_list,
        )?;
        Ok(Some(Box::new(state)))
    }

    fn box_clone(&self) -> Box<dyn PotentialBuilder> {
        Box::new(self.clone())
    }

    fn pair_force_fragment(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<PairForceFragment>, ForceFieldError> {
        if cx.spme_config.is_none() {
            return Ok(None);
        }
        Ok(Some(spme_real_pair_force_fragment()))
    }
}

// rq-e8550f96
#[derive(Debug, Clone)]
pub struct SpmeReciprocalBuilder;

impl PotentialBuilder for SpmeReciprocalBuilder {
    fn build(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        let Some(spme_cfg) = cx.spme_config else {
            return Ok(None);
        };
        let params = SpmeParameters::from(spme_cfg);
        let state = SpmeReciprocalState::new(
            cx.gpu,
            cx.sim_box,
            cx.particle_count,
            cx.charges,
            params,
        )
        .map_err(map_spme_err)?;
        Ok(Some(Box::new(state)))
    }

    fn box_clone(&self) -> Box<dyn PotentialBuilder> {
        Box::new(self.clone())
    }
}

// rq-2093594f rq-9a512ed1
#[derive(Debug, Clone)]
pub struct SpmeRealKernels {
    pub spme_real_pair_force_f: CudaFunction,
    pub spme_real_pair_force_fev: CudaFunction,
}

impl SpmeRealKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::SPME_REAL),
            "spme_real",
            &["spme_real_pair_force_f", "spme_real_pair_force_fev"],
        )?;
        Ok(SpmeRealKernels {
            spme_real_pair_force_f: get_func(device, "spme_real", "spme_real_pair_force_f")?,
            spme_real_pair_force_fev: get_func(device, "spme_real", "spme_real_pair_force_fev")?,
        })
    }
}

// rq-2093594f rq-9ca00d25
#[derive(Debug, Clone)]
pub struct SpmeRecipKernels {
    pub spme_recip_compute_influence: CudaFunction,
    pub spme_compute_bin_key: CudaFunction,
    pub spme_spread_fixed_point: CudaFunction,
    pub spme_spread_finish: CudaFunction,
    pub spme_recip_apply_influence: CudaFunction,
    pub spme_recip_reduce_partials: CudaFunction,
    pub spme_force_gather: CudaFunction,
}

impl SpmeRecipKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::SPME_RECIP),
            "spme_recip",
            &[
                "spme_recip_compute_influence",
                "spme_compute_bin_key",
                "spme_spread_fixed_point",
                "spme_spread_finish",
                "spme_recip_apply_influence",
                "spme_recip_reduce_partials",
                "spme_force_gather",
            ],
        )?;
        Ok(SpmeRecipKernels {
            spme_recip_compute_influence: get_func(
                device,
                "spme_recip",
                "spme_recip_compute_influence",
            )?,
            spme_compute_bin_key: get_func(device, "spme_recip", "spme_compute_bin_key")?,
            spme_spread_fixed_point: get_func(device, "spme_recip", "spme_spread_fixed_point")?,
            spme_spread_finish: get_func(device, "spme_recip", "spme_spread_finish")?,
            spme_recip_apply_influence: get_func(
                device,
                "spme_recip",
                "spme_recip_apply_influence",
            )?,
            spme_recip_reduce_partials: get_func(
                device,
                "spme_recip",
                "spme_recip_reduce_partials",
            )?,
            spme_force_gather: get_func(device, "spme_recip", "spme_force_gather")?,
        })
    }
}
