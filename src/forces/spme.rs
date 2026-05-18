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

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice};
use cudarc::nvrtc::Ptx;

use crate::gpu::cufft::{CuFftError, Plan3dC2R, Plan3dR2C};
use crate::gpu::device::get_func;
use crate::gpu::{
    GpuContext, GpuError, K_COULOMB_F32, PairBuffer, ParticleBuffers, reduce_pair_forces,
    spme_charge_spread, spme_force_gather, spme_influence_multiply, spme_real_pair_force,
};
use crate::kernels;
use crate::io::config::SpmeConfig;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::topology::{DeviceExclusionList, ExclusionList};
use super::neighbor_list::{NeighborListError, NeighborListState};
use super::{
    ForceFieldContext, ForceFieldError, Potential, PotentialBuildContext, PotentialBuilder,
    SlotOutputView,
};

// rq-7bd2d9ca
#[derive(Debug, Clone, Copy)]
pub struct SpmeParameters {
    pub alpha: f32,
    pub r_cut_real: f32,
    pub grid: [u32; 3],
    pub spline_order: u32,
}

impl From<&SpmeConfig> for SpmeParameters {
    fn from(c: &SpmeConfig) -> Self {
        SpmeParameters {
            alpha: c.alpha as f32,
            r_cut_real: c.r_cut_real as f32,
            grid: c.grid,
            spline_order: c.spline_order,
        }
    }
}

// rq-846bdb8b
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
    pub bin_list: NeighborListState,
    pub rho: CudaSlice<f32>,
    pub rho_hat_interleaved: CudaSlice<f32>,
    pub v: CudaSlice<f32>,
    pub influence_g: CudaSlice<f32>,
    /// Per-cell factor `G[k] · (1 − K²/(2α²))`. Multiplied by
    /// `|rho_hat[k]|²` and the Hermitian weight by the kernel to produce
    /// `virial_per_cell`.
    pub virial_factor: CudaSlice<f32>,
    /// Scratch buffer holding the per-cell virial contribution after
    /// `spme_influence_multiply`. Reduced host-side to the scalar
    /// W_recip during the gather/reduce step.
    pub virial_per_cell: CudaSlice<f32>,
    /// Box generation the influence function was computed against.
    /// Refreshed when the sim_box generation changes.
    pub cached_box_generation: u64,
    pub forward_plan: Plan3dR2C,
    pub inverse_plan: Plan3dC2R,
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

        let bin_list = NeighborListState::new_cell_list_only(
            gpu,
            sim_box,
            particle_count,
            params.grid,
        )?;

        let rho = device.alloc_zeros::<f32>(m).map_err(GpuError::from)?;
        let v = device.alloc_zeros::<f32>(m).map_err(GpuError::from)?;
        let rho_hat_interleaved = device
            .alloc_zeros::<f32>(2 * m_complex)
            .map_err(GpuError::from)?;

        // Compute b-factors, influence function, and virial factor on
        // the host, then upload.
        let b_factors_a = compute_b_factors(n_a, p);
        let b_factors_b = compute_b_factors(n_b, p);
        let b_factors_c = compute_b_factors(n_c, p);
        let (influence_host, virial_host) = compute_influence_and_virial(
            sim_box,
            params,
            &b_factors_a,
            &b_factors_b,
            &b_factors_c,
        );
        debug_assert_eq!(influence_host.len(), m_complex);
        debug_assert_eq!(virial_host.len(), m_complex);
        let influence_g = device.htod_sync_copy(&influence_host).map_err(GpuError::from)?;
        let virial_factor =
            device.htod_sync_copy(&virial_host).map_err(GpuError::from)?;
        let virial_per_cell = device
            .alloc_zeros::<f32>(m_complex)
            .map_err(GpuError::from)?;

        let forward_plan = Plan3dR2C::new(&device, n_a, n_b, n_c)?;
        let inverse_plan = Plan3dC2R::new(&device, n_a, n_b, n_c)?;

        Ok(SpmeReciprocalGrid {
            device,
            params,
            particle_count,
            m,
            m_complex,
            bin_list,
            rho,
            rho_hat_interleaved,
            v,
            influence_g,
            virial_factor,
            virial_per_cell,
            cached_box_generation: sim_box.generation(),
            forward_plan,
            inverse_plan,
        })
    }

    /// Run the per-step reciprocal-space pipeline:
    ///   spread → forward FFT → influence multiply → inverse FFT.
    /// On return, `self.v` holds the smoothed potential V[g] (with
    /// `k_C/V_box` and `|b|²` baked in via the influence function);
    /// `self.rho` holds the charge density rho[g].
    pub fn compute(
        &mut self,
        sim_box: &SimulationBox,
        particle_buffers: &ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<(), SpmeError> {
        // Refresh influence function (and the virial factor that
        // tracks it) if the box has changed.
        if sim_box.generation() != self.cached_box_generation {
            let n_a = self.params.grid[0];
            let n_b = self.params.grid[1];
            let n_c = self.params.grid[2];
            let p = self.params.spline_order;
            let b_factors_a = compute_b_factors(n_a, p);
            let b_factors_b = compute_b_factors(n_b, p);
            let b_factors_c = compute_b_factors(n_c, p);
            let (g_host, vf_host) = compute_influence_and_virial(
                sim_box,
                self.params,
                &b_factors_a,
                &b_factors_b,
                &b_factors_c,
            );
            self.device
                .htod_sync_copy_into(&g_host, &mut self.influence_g)
                .map_err(GpuError::from)?;
            self.device
                .htod_sync_copy_into(&vf_host, &mut self.virial_factor)
                .map_err(GpuError::from)?;
            self.cached_box_generation = sim_box.generation();
        }

        // 1. Rebuild bin structure.
        self.bin_list.pre_step(sim_box, particle_buffers, timings)?;

        // 2. Charge spreading (writes rho).
        let cl = self
            .bin_list
            .cell_list_data()
            .expect("SpmeReciprocalGrid bin list must be in cell-list-only mode");
        spme_charge_spread(
            particle_buffers,
            sim_box,
            &cl.sorted_particle_ids,
            &cl.cell_offsets,
            self.params.grid,
            self.params.spline_order,
            &mut self.rho,
        )?;

        // 3. Forward FFT (rho → rho_hat).
        self.forward_plan
            .execute(&self.rho, &mut self.rho_hat_interleaved)?;

        // 4. Influence multiply (rho_hat *= G; also writes per-cell virial).
        let n_c = self.params.grid[2];
        let n_c_complex = (n_c / 2 + 1) as u32;
        spme_influence_multiply(
            &particle_buffers.kernels,
            &self.influence_g,
            &self.virial_factor,
            &mut self.rho_hat_interleaved,
            &mut self.virial_per_cell,
            n_c,
            n_c_complex,
            self.m_complex as u32,
        )?;

        // 5. Inverse FFT (rho_hat → V).
        self.inverse_plan
            .execute(&self.rho_hat_interleaved, &mut self.v)?;

        Ok(())
    }
}

// rq-9ca00d25
//
// SPME B-spline structure-factor correction. For axis with grid size N
// and B-spline order p, `b_factors[k] = |b(k)|²` where
//   b(k) = exp(2π i (p-1) k / N) / Σ_{j=0..p-2} M_p(j+1) · exp(2π i j k / N)
// and |b(k)|² = 1 / |denominator|².
pub fn compute_b_factors(n: u32, p: u32) -> Vec<f32> {
    let n = n as usize;
    let p = p as usize;
    let mut out = vec![0.0_f32; n];
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
            (1.0 / denom2) as f32
        } else {
            0.0
        };
    }
    out
}

// rq-9ca00d25
//
// Influence function G[k] for the SPME reciprocal pipeline, computed
// on the host and uploaded once per box-generation refresh. The
// formula is
//   G[k] = (k_C / V_box) · (4π / |K|²) · exp(-|K|²/(4α²))
//          · b_factors_a[k_a] · b_factors_b[k_b] · b_factors_c[k_c]
// with G[0] = 0 (tinfoil boundary conditions). The reciprocal-lattice
// wave vector K = 2π · (m_a · b_a_vec + m_b · b_b_vec + m_c · b_c_vec)
// where b_*_vec are the rows of H^{-T} (the inverse-transpose of the
// lattice matrix; see `simulation-box.md`).
//
// The companion `virial_factor[k] = G[k] · (1 − K²/(2α²))` is
// precomputed alongside G to support the reciprocal-space scalar
// virial reduction. virial_factor[0] = 0.
pub fn compute_influence_and_virial(
    sim_box: &SimulationBox,
    params: SpmeParameters,
    b_factors_a: &[f32],
    b_factors_b: &[f32],
    b_factors_c: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let n_a = params.grid[0] as usize;
    let n_b = params.grid[1] as usize;
    let n_c = params.grid[2] as usize;
    let n_c_complex = n_c / 2 + 1;
    let m_complex = n_a * n_b * n_c_complex;
    let mut out = vec![0.0_f32; m_complex];
    let mut vf = vec![0.0_f32; m_complex];

    let lx = sim_box.lx() as f64;
    let ly = sim_box.ly() as f64;
    let lz = sim_box.lz() as f64;
    let xy = sim_box.xy() as f64;
    let xz = sim_box.xz() as f64;
    let yz = sim_box.yz() as f64;
    let v_box = (lx * ly * lz) as f64;
    let alpha = params.alpha as f64;
    let four_alpha2 = 4.0 * alpha * alpha;
    let four_pi = 4.0 * std::f64::consts::PI;
    let two_pi = 2.0 * std::f64::consts::PI;
    let k_c = K_COULOMB_F32 as f64;
    let prefactor = k_c / v_box;

    // Rows of H^{-T} (the reciprocal lattice).
    // For our lower-triangular H with rows = (a, b, c), the transpose is
    // upper triangular with the same elements; its inverse is again
    // upper triangular and has closed-form entries below.
    let recip_a = [
        1.0 / lx,
        -xy / (lx * ly),
        (xy * yz - xz * ly) / (lx * ly * lz),
    ];
    let recip_b = [0.0, 1.0 / ly, -yz / (ly * lz)];
    let recip_c = [0.0, 0.0, 1.0 / lz];

    for ka in 0..n_a {
        let ma: f64 = if ka <= n_a / 2 {
            ka as f64
        } else {
            ka as f64 - n_a as f64
        };
        let b_a = b_factors_a[ka] as f64;
        for kb in 0..n_b {
            let mb: f64 = if kb <= n_b / 2 {
                kb as f64
            } else {
                kb as f64 - n_b as f64
            };
            let b_b = b_factors_b[kb] as f64;
            for kc in 0..n_c_complex {
                let mc: f64 = if kc <= n_c / 2 {
                    kc as f64
                } else {
                    kc as f64 - n_c as f64
                };
                let b_c = b_factors_c[kc] as f64;
                let kx = two_pi
                    * (ma * recip_a[0] + mb * recip_b[0] + mc * recip_c[0]);
                let ky = two_pi
                    * (ma * recip_a[1] + mb * recip_b[1] + mc * recip_c[1]);
                let kz = two_pi
                    * (ma * recip_a[2] + mb * recip_b[2] + mc * recip_c[2]);
                let k2 = kx * kx + ky * ky + kz * kz;
                let idx = (ka * n_b + kb) * n_c_complex + kc;
                if k2 == 0.0 {
                    out[idx] = 0.0;
                    vf[idx] = 0.0;
                } else {
                    let g = prefactor * (four_pi / k2) * (-k2 / four_alpha2).exp()
                        * b_a
                        * b_b
                        * b_c;
                    out[idx] = g as f32;
                    let factor = 1.0 - k2 / (2.0 * alpha * alpha);
                    vf[idx] = (g * factor) as f32;
                }
            }
        }
    }
    (out, vf)
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
// `LennardJonesState` and `CoulombState`: owns a `PairBuffer` and a
// `DeviceExclusionList`, contributes via the `spme_real_pair_force`
// kernel, and reduces via `reduce_pair_forces`.
#[derive(Debug)]
pub struct SpmeRealSpaceState {
    #[allow(dead_code)]
    device: Arc<CudaDevice>,
    pair_buffer: PairBuffer,
    exclusions: DeviceExclusionList,
    alpha: f32,
    r_cut_real: f32,
    particle_count: usize,
}

impl SpmeRealSpaceState {
    pub fn new(
        gpu: &GpuContext,
        particle_count: usize,
        alpha: f32,
        r_cut_real: f32,
        max_neighbors: u32,
        exclusion_list: &ExclusionList,
    ) -> Result<Self, NeighborListError> {
        let device = gpu.device.clone();
        let pair_buffer = PairBuffer::new(gpu, particle_count, max_neighbors)?;
        let exclusions = DeviceExclusionList::from_host(&device, exclusion_list)?;
        Ok(SpmeRealSpaceState {
            device,
            pair_buffer,
            exclusions,
            alpha,
            r_cut_real,
            particle_count,
        })
    }
}

impl Potential for SpmeRealSpaceState {
    fn label(&self) -> &'static str {
        "spme_real"
    }

    fn max_cutoff(&self) -> Option<f32> {
        Some(self.r_cut_real)
    }

    fn contribute(
        &mut self,
        buffers: &ParticleBuffers,
        sim_box: &SimulationBox,
        cx: &ForceFieldContext<'_>,
        timings: &mut Timings,
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
            &mut self.pair_buffer,
            sim_box,
            self.alpha,
            self.r_cut_real,
            &self.exclusions.atom_excl_offsets,
            &self.exclusions.atom_excl_partners,
            &self.exclusions.atom_excl_coul_scales,
            &nl.neighbor_list,
            &nl.neighbor_counts,
        )?;
        timings.kernel_stop(KernelStage::SPME_REAL_PAIR_FORCE)?;
        Ok(())
    }

    fn reduce(
        &mut self,
        mut output: SlotOutputView<'_>,
        cx: &ForceFieldContext<'_>,
        timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        if self.particle_count == 0 {
            return Ok(());
        }
        let nl = cx
            .neighbor_list
            .expect("SpmeRealSpaceState requires a shared neighbor list");
        timings.kernel_start(KernelStage::REDUCE_PAIR_FORCES)?;
        reduce_pair_forces(
            &self.pair_buffer,
            &nl.neighbor_counts,
            &mut output.force_x,
            &mut output.force_y,
            &mut output.force_z,
            &mut output.energy,
            &mut output.virial,
            self.particle_count,
        )?;
        timings.kernel_stop(KernelStage::REDUCE_PAIR_FORCES)?;
        Ok(())
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
#[derive(Debug)]
pub struct SpmeReciprocalState {
    grid: SpmeReciprocalGrid,
    // `u_self_per_particle[i] = k_C · (α/√π) · q_i²`. Subtracted from
    // the per-particle reciprocal energy inside the gather kernel.
    u_self_per_particle: CudaSlice<f32>,
    // Host scratch for the reciprocal-virial reduction. Reused across
    // steps to avoid per-step allocation.
    virial_host_scratch: Vec<f32>,
    // Reduced per-particle reciprocal virial share, set by `contribute()`
    // from `virial_per_cell` and consumed by `reduce()` via the gather
    // kernel argument.
    w_per_particle_virial: f32,
}

impl SpmeReciprocalState {
    pub fn new(
        gpu: &GpuContext,
        sim_box: &SimulationBox,
        particle_count: usize,
        charges: &[f32],
        params: SpmeParameters,
    ) -> Result<Self, SpmeError> {
        let grid = SpmeReciprocalGrid::new(gpu, sim_box, particle_count, params)?;
        // Precompute per-particle self-energy:
        //   u_self_i = k_C · (α/√π) · q_i²
        let inv_sqrt_pi = 1.0_f64 / std::f64::consts::PI.sqrt();
        let prefactor = (K_COULOMB_F32 as f64) * (params.alpha as f64) * inv_sqrt_pi;
        let u_self_host: Vec<f32> = (0..particle_count)
            .map(|i| {
                let q = charges.get(i).copied().unwrap_or(0.0);
                (prefactor * (q as f64) * (q as f64)) as f32
            })
            .collect();
        let u_self_per_particle = if particle_count == 0 {
            grid.device.alloc_zeros::<f32>(0).map_err(GpuError::from)?
        } else {
            grid.device
                .htod_sync_copy(&u_self_host)
                .map_err(GpuError::from)?
        };
        Ok(SpmeReciprocalState {
            grid,
            u_self_per_particle,
            virial_host_scratch: vec![0.0_f32; grid_m_complex_or_zero(&params)],
            w_per_particle_virial: 0.0,
        })
    }

    // Test access to the underlying grid (rho/V buffers, influence_g,
    // etc.). Used by `tests/spme_pipeline.rs`.
    pub fn grid(&self) -> &SpmeReciprocalGrid {
        &self.grid
    }
}

fn grid_m_complex_or_zero(params: &SpmeParameters) -> usize {
    let n_a = params.grid[0] as usize;
    let n_b = params.grid[1] as usize;
    let n_c = params.grid[2] as usize;
    n_a * n_b * (n_c / 2 + 1)
}

impl Potential for SpmeReciprocalState {
    fn label(&self) -> &'static str {
        "spme_reciprocal"
    }

    fn max_cutoff(&self) -> Option<f32> {
        // The reciprocal-space slot does not consume the shared
        // neighbor list; it owns its own bin structure internally.
        None
    }

    // rq-df6d79a1
    fn frequency_class(&self) -> super::ForceClass {
        super::ForceClass::Slow
    }

    fn contribute(
        &mut self,
        buffers: &ParticleBuffers,
        sim_box: &SimulationBox,
        _cx: &ForceFieldContext<'_>,
        timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        let n = self.grid.particle_count;
        if n == 0 {
            self.w_per_particle_virial = 0.0;
            return Ok(());
        }
        timings.kernel_start(KernelStage::SPME_RECIP_PIPELINE)?;
        self.grid
            .compute(sim_box, buffers, timings)
            .map_err(map_spme_err)?;
        timings.kernel_stop(KernelStage::SPME_RECIP_PIPELINE)?;

        // Reduce `virial_per_cell` host-side. The 0.5 factor matches the
        // Ewald half-sum that defines U_recip in
        // `docs/long-range-electrostatics.md`.
        if self.virial_host_scratch.len() != self.grid.m_complex {
            self.virial_host_scratch.resize(self.grid.m_complex, 0.0);
        }
        self.grid
            .device
            .dtoh_sync_copy_into(&self.grid.virial_per_cell, &mut self.virial_host_scratch)
            .map_err(GpuError::from)?;
        let mut w_recip = 0.0_f64;
        for &v in &self.virial_host_scratch {
            w_recip += v as f64;
        }
        self.w_per_particle_virial = (0.5 * w_recip / n as f64) as f32;
        Ok(())
    }

    fn reduce(
        &mut self,
        mut output: SlotOutputView<'_>,
        cx: &ForceFieldContext<'_>,
        timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        let n = self.grid.particle_count;
        if n == 0 {
            return Ok(());
        }
        timings.kernel_start(KernelStage::SPME_FORCE_GATHER)?;
        spme_force_gather(
            cx.buffers,
            cx.sim_box,
            &self.grid.v,
            &self.u_self_per_particle,
            self.w_per_particle_virial,
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
#[derive(Debug)]
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
}

// rq-e8550f96
#[derive(Debug)]
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
}

// rq-2093594f rq-9a512ed1
#[derive(Debug, Clone)]
pub struct SpmeRealKernels {
    pub spme_real_pair_force: CudaFunction,
}

impl SpmeRealKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::SPME_REAL),
            "spme_real",
            &["spme_real_pair_force"],
        )?;
        Ok(SpmeRealKernels {
            spme_real_pair_force: get_func(device, "spme_real", "spme_real_pair_force")?,
        })
    }
}

// rq-2093594f rq-9ca00d25
#[derive(Debug, Clone)]
pub struct SpmeRecipKernels {
    pub spme_charge_spread: CudaFunction,
    pub spme_influence_multiply: CudaFunction,
    pub spme_force_gather: CudaFunction,
}

impl SpmeRecipKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::SPME_RECIP),
            "spme_recip",
            &["spme_charge_spread", "spme_influence_multiply", "spme_force_gather"],
        )?;
        Ok(SpmeRecipKernels {
            spme_charge_spread: get_func(device, "spme_recip", "spme_charge_spread")?,
            spme_influence_multiply: get_func(device, "spme_recip", "spme_influence_multiply")?,
            spme_force_gather: get_func(device, "spme_recip", "spme_force_gather")?,
        })
    }
}
