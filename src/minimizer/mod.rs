// Minimizer slot framework. See `rqm/minimization/steepest-descent.md`.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceSlice};
use cudarc::nvrtc::Ptx;

use crate::forces::{ForceField, ForceFieldError};
use crate::gpu::device::get_func;
use crate::gpu::{GpuContext, GpuError, ParticleBuffers};
use crate::integrator::{Constraint, ConstraintError};
use crate::io::config::{ConfigError, SlotConfig};
use crate::kernels;
use crate::pbc::SimulationBox;
use crate::timings::{Timings, TimingsError};

pub mod steepest_descent;

pub use steepest_descent::{
    SteepestDescentBuilder, SteepestDescentMinimizer, SteepestDescentParams,
};

#[derive(Debug, thiserror::Error)]
pub enum MinimizerError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("{0}")]
    Timings(#[from] TimingsError),
    #[error("{0}")]
    ForceField(#[from] ForceFieldError),
    #[error("{0}")]
    Constraint(#[from] ConstraintError),
    #[error("unknown minimizer kind `{0}`")]
    UnknownKind(String),
}

/// Why a minimizer's outer loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinimizerConvergence {
    /// `F_max ≤ force_tolerance` at the accepted positions.
    ForceTolerance,
    /// `|ΔE| / max(|E_prev|, |E_curr|, ε) ≤ energy_tolerance` between
    /// two consecutive accepted iterations.
    EnergyTolerance,
    /// `F_max == 0.0` at the accepted positions (already at minimum,
    /// or `particle_count == 0`).
    ForceZero,
    /// Iteration cap reached without any physical criterion firing.
    MaxIterations,
}

impl MinimizerConvergence {
    /// Short token used in stdout/diagnostics output.
    pub fn token(self) -> &'static str {
        match self {
            MinimizerConvergence::ForceTolerance => "force_tolerance",
            MinimizerConvergence::EnergyTolerance => "energy_tolerance",
            MinimizerConvergence::ForceZero => "force_zero",
            MinimizerConvergence::MaxIterations => "max_iterations",
        }
    }
}

/// Outcome of one `Minimizer::step` call.
#[derive(Debug, Clone, Copy)]
pub struct MinimizerStepReport {
    /// True iff the trial step was accepted.
    pub accepted: bool,
    /// Total potential energy at the post-call accepted positions (J).
    /// For a rejected step this equals `prev_energy` since positions
    /// are restored from the snapshot.
    pub energy: f64,
    /// `F_max` at the post-call accepted positions (N).
    pub max_force: f64,
    /// The step size used by this iteration's trial (m).
    pub step_size: f64,
    /// The accepted energy entering this iteration (J). Used by the
    /// runner's energy-tolerance convergence check.
    pub prev_energy: f64,
}

/// One concrete minimizer slot. Implementations own per-phase device
/// allocations (snapshot buffers, reductions scratches) and the
/// per-phase adaptive state (current step size).
pub trait Minimizer: std::fmt::Debug + Send {
    /// Execute one outer iteration. The runner supplies the global
    /// force field (already populated with `forces_*` and
    /// `potential_energies` at the current accepted positions) and
    /// the optional constraint slot. Returns the post-iteration
    /// accepted state along with the accept/reject decision.
    fn step(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        force_field: &mut ForceField,
        constraint: Option<&mut dyn Constraint>,
        timings: &mut Timings,
    ) -> Result<MinimizerStepReport, MinimizerError>;

    /// Compute the initial accepted state at the runner-supplied warmed-up
    /// forces. Returns `(energy, max_force)` at the current positions.
    fn initial_state(
        &mut self,
        buffers: &mut ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<(f64, f64), MinimizerError>;

    /// Test the most-recent accepted state against the configured
    /// physical convergence criteria. Returns `Some(reason)` on
    /// convergence, `None` to continue. Pure: no kernel launches, no
    /// buffer mutations.
    fn check_convergence(
        &self,
        report: &MinimizerStepReport,
    ) -> Option<MinimizerConvergence>;

    /// Convergence reason when the runner exhausts `max_iterations`.
    /// Always `MaxIterations` in the SD case; reserved for variants
    /// that might short-circuit differently.
    fn max_iterations_reason(&self) -> MinimizerConvergence {
        MinimizerConvergence::MaxIterations
    }

    /// Iteration cap configured for this minimizer.
    fn max_iterations(&self) -> u64;
}

pub trait MinimizerBuilder: std::fmt::Debug + Send + Sync {
    fn kind_name(&self) -> &'static str;

    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError>;

    /// `true` iff this minimizer can drive a constraint slot's
    /// position-projection hook. Default `true`; future minimizers
    /// that cannot project positions in isolation override.
    fn supports_constraints(&self, _params: &toml::Value) -> bool {
        true
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
        params: &toml::Value,
    ) -> Result<Box<dyn Minimizer>, MinimizerError>;
}

#[derive(Debug)]
pub struct MinimizerRegistry {
    pub builders: Vec<Box<dyn MinimizerBuilder>>,
}

impl MinimizerRegistry {
    pub fn new() -> Self {
        MinimizerRegistry {
            builders: Vec::new(),
        }
    }

    pub fn with_builtins() -> Self {
        MinimizerRegistry {
            builders: vec![Box::new(SteepestDescentBuilder)],
        }
    }

    pub fn register(&mut self, builder: Box<dyn MinimizerBuilder>) {
        self.builders.push(builder);
    }

    pub fn lookup(&self, kind: &str) -> Option<&dyn MinimizerBuilder> {
        for b in &self.builders {
            if b.kind_name() == kind {
                return Some(b.as_ref());
            }
        }
        None
    }

    pub fn build(
        &self,
        slot: &SlotConfig,
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
    ) -> Result<Box<dyn Minimizer>, MinimizerError> {
        let b = self
            .lookup(&slot.kind)
            .ok_or_else(|| MinimizerError::UnknownKind(slot.kind.clone()))?;
        b.build(gpu, particle_count, n_constraints, &slot.params)
    }
}

impl Default for MinimizerRegistry {
    fn default() -> Self {
        MinimizerRegistry::with_builtins()
    }
}

// CUDA kernel handle for the minimizer's per-step kernels. Loaded
// alongside the other kernel modules at `init_device`.
#[derive(Debug, Clone)]
pub struct MinimizeKernels {
    pub sd_compute_step: CudaFunction,
    pub sd_snapshot: CudaFunction,
    pub sd_restore: CudaFunction,
    pub sd_f_max_reduction: CudaFunction,
}

impl MinimizeKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::MINIMIZE),
            "minimize",
            &[
                "sd_compute_step",
                "sd_snapshot",
                "sd_restore",
                "sd_f_max_reduction",
            ],
        )?;
        Ok(MinimizeKernels {
            sd_compute_step: get_func(device, "minimize", "sd_compute_step")?,
            sd_snapshot: get_func(device, "minimize", "sd_snapshot")?,
            sd_restore: get_func(device, "minimize", "sd_restore")?,
            sd_f_max_reduction: get_func(device, "minimize", "sd_f_max_reduction")?,
        })
    }
}

// Host-launch wrappers for the minimizer kernels.

const BLOCK: u32 = 256;

fn ceil_div_block(n: u32) -> u32 {
    n.div_ceil(BLOCK)
}

pub(crate) fn sd_compute_step(
    buffers: &mut ParticleBuffers,
    step_size: f32,
    inv_f_max: f32,
) -> Result<(), GpuError> {
    let n = buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    let func = buffers.kernels.minimize.sd_compute_step.clone();
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (ceil_div_block(n_u32), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    use cudarc::driver::LaunchAsync;
    unsafe {
        func.launch(
            cfg,
            (
                &mut buffers.positions_x,
                &mut buffers.positions_y,
                &mut buffers.positions_z,
                &buffers.forces_x,
                &buffers.forces_y,
                &buffers.forces_z,
                step_size,
                inv_f_max,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

pub(crate) fn sd_snapshot(
    buffers: &ParticleBuffers,
    snapshot_x: &mut CudaSlice<f32>,
    snapshot_y: &mut CudaSlice<f32>,
    snapshot_z: &mut CudaSlice<f32>,
) -> Result<(), GpuError> {
    let n = buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    let func = buffers.kernels.minimize.sd_snapshot.clone();
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (ceil_div_block(n_u32), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    use cudarc::driver::LaunchAsync;
    unsafe {
        func.launch(
            cfg,
            (
                &buffers.positions_x,
                &buffers.positions_y,
                &buffers.positions_z,
                snapshot_x,
                snapshot_y,
                snapshot_z,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

pub(crate) fn sd_restore(
    buffers: &mut ParticleBuffers,
    snapshot_x: &CudaSlice<f32>,
    snapshot_y: &CudaSlice<f32>,
    snapshot_z: &CudaSlice<f32>,
) -> Result<(), GpuError> {
    let n = buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    let func = buffers.kernels.minimize.sd_restore.clone();
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (ceil_div_block(n_u32), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    use cudarc::driver::LaunchAsync;
    unsafe {
        func.launch(
            cfg,
            (
                &mut buffers.positions_x,
                &mut buffers.positions_y,
                &mut buffers.positions_z,
                snapshot_x,
                snapshot_y,
                snapshot_z,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

pub(crate) fn sd_f_max_reduction(
    buffers: &ParticleBuffers,
    scratch: &mut CudaSlice<f32>,
) -> Result<f32, GpuError> {
    let n = buffers.particle_count();
    if n == 0 {
        return Ok(0.0_f32);
    }
    debug_assert_eq!(scratch.len(), 1);
    let n_u32 = n as u32;
    let func = buffers.kernels.minimize.sd_f_max_reduction.clone();
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    use cudarc::driver::LaunchAsync;
    unsafe {
        func.launch(
            cfg,
            (
                &buffers.forces_x,
                &buffers.forces_y,
                &buffers.forces_z,
                &mut *scratch,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    let mut out = [0.0_f32; 1];
    buffers
        .device
        .dtoh_sync_copy_into(scratch, &mut out)
        .map_err(GpuError::from)?;
    Ok(out[0])
}
