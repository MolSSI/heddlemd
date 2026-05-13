// rq-4f386df8 rq-5cb33196 rq-67414c32
use std::sync::Arc;

use cudarc::driver::CudaDevice;

use crate::forces::{ForceField, ForceFieldError};
use crate::gpu::{
    GpuError, LosslessBuffers, ParticleBuffers, lan_drift_half, lan_ou_step, vv_kick,
    vv_kick_drift, vv_kick_drift_lossless, vv_kick_lossless,
};
use crate::io::config::IntegratorKind;
use crate::io::log_output::BOLTZMANN_J_PER_K;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings, TimingsError};

// rq-a5069572
#[derive(Debug)]
pub enum IntegratorError {
    Gpu(GpuError),
    Timings(TimingsError),
    ForceField(ForceFieldError),
    UnknownKind(String),
}

impl std::fmt::Display for IntegratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntegratorError::Gpu(e) => write!(f, "Gpu({e})"),
            IntegratorError::Timings(e) => write!(f, "Timings({e})"),
            IntegratorError::ForceField(e) => write!(f, "ForceField({e})"),
            IntegratorError::UnknownKind(name) => {
                write!(f, "UnknownKind({name:?})")
            }
        }
    }
}

impl std::error::Error for IntegratorError {}

impl From<GpuError> for IntegratorError {
    fn from(e: GpuError) -> Self {
        IntegratorError::Gpu(e)
    }
}

impl From<TimingsError> for IntegratorError {
    fn from(e: TimingsError) -> Self {
        IntegratorError::Timings(e)
    }
}

impl From<ForceFieldError> for IntegratorError {
    fn from(e: ForceFieldError) -> Self {
        IntegratorError::ForceField(e)
    }
}

// rq-e4c4ff61
pub trait Integrator: std::fmt::Debug + Send {
    fn step(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        force_field: &mut ForceField,
        dt: f32,
        step_index: u64,
        timings: &mut Timings,
    ) -> Result<(), IntegratorError>;
}

// rq-87fdd9b1
pub trait IntegratorBuilder: std::fmt::Debug + Send + Sync {
    fn kind_name(&self) -> &'static str;
    fn build(
        &self,
        device: Arc<CudaDevice>,
        particle_count: usize,
        kind: &IntegratorKind,
    ) -> Result<Box<dyn Integrator>, IntegratorError>;
}

// rq-1d5b5e35
#[derive(Debug)]
pub struct IntegratorRegistry {
    pub builders: Vec<Box<dyn IntegratorBuilder>>,
}

impl IntegratorRegistry {
    pub fn new() -> Self {
        IntegratorRegistry { builders: Vec::new() }
    }

    pub fn with_builtins() -> Self {
        IntegratorRegistry {
            builders: vec![
                Box::new(VelocityVerletBuilder),
                Box::new(LangevinBaoabBuilder),
            ],
        }
    }

    pub fn register(&mut self, builder: Box<dyn IntegratorBuilder>) {
        self.builders.push(builder);
    }

    // rq-df39d15b
    pub fn build(
        &self,
        kind: &IntegratorKind,
        device: Arc<CudaDevice>,
        particle_count: usize,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        let target = kind.name();
        for b in &self.builders {
            if b.kind_name() == target {
                return b.build(device, particle_count, kind);
            }
        }
        Err(IntegratorError::UnknownKind(target.to_string()))
    }
}

impl Default for IntegratorRegistry {
    fn default() -> Self {
        IntegratorRegistry::with_builtins()
    }
}

// --- Velocity Verlet ---

#[derive(Debug)]
pub struct VelocityVerletState {
    lossless: Option<LosslessBuffers>,
}

impl Integrator for VelocityVerletState {
    // rq-cf361ff5
    fn step(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        force_field: &mut ForceField,
        dt: f32,
        _step_index: u64,
        timings: &mut Timings,
    ) -> Result<(), IntegratorError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }

        if let Some(ll) = self.lossless.as_mut() {
            timings.kernel_start(KernelStage::VvKickDriftLossless)?;
            vv_kick_drift_lossless(buffers, ll, dt)?;
            timings.kernel_stop(KernelStage::VvKickDriftLossless)?;
        } else {
            timings.kernel_start(KernelStage::VvKickDrift)?;
            vv_kick_drift(buffers, dt)?;
            timings.kernel_stop(KernelStage::VvKickDrift)?;
        }

        force_field.step(buffers, sim_box, timings)?;

        if let Some(ll) = self.lossless.as_mut() {
            timings.kernel_start(KernelStage::VvKickLossless)?;
            vv_kick_lossless(buffers, ll, dt)?;
            timings.kernel_stop(KernelStage::VvKickLossless)?;
        } else {
            timings.kernel_start(KernelStage::VvKick)?;
            vv_kick(buffers, dt)?;
            timings.kernel_stop(KernelStage::VvKick)?;
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct VelocityVerletBuilder;

impl IntegratorBuilder for VelocityVerletBuilder {
    fn kind_name(&self) -> &'static str {
        "velocity-verlet"
    }

    fn build(
        &self,
        device: Arc<CudaDevice>,
        particle_count: usize,
        kind: &IntegratorKind,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        match kind {
            IntegratorKind::VelocityVerlet { lossless } => {
                let buffers = if *lossless {
                    Some(LosslessBuffers::new(device, particle_count)?)
                } else {
                    None
                };
                Ok(Box::new(VelocityVerletState { lossless: buffers }))
            }
            other => Err(IntegratorError::UnknownKind(other.name().to_string())),
        }
    }
}

// --- Langevin BAOAB ---

#[derive(Debug)]
pub struct LangevinBaoabState {
    pub friction: f64,
    pub temperature: f64,
    pub seed: u64,
}

impl Integrator for LangevinBaoabState {
    fn step(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        force_field: &mut ForceField,
        dt: f32,
        step_index: u64,
        timings: &mut Timings,
    ) -> Result<(), IntegratorError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }

        // BAOAB pre-force: B(dt/2), A(dt/2), O(dt), A(dt/2)
        timings.kernel_start(KernelStage::LangevinKickHalf)?;
        vv_kick(buffers, dt)?;
        timings.kernel_stop(KernelStage::LangevinKickHalf)?;

        timings.kernel_start(KernelStage::LangevinDriftHalf)?;
        lan_drift_half(buffers, dt)?;
        timings.kernel_stop(KernelStage::LangevinDriftHalf)?;

        let alpha = (-(self.friction as f32) * dt).exp();
        let kt = (BOLTZMANN_J_PER_K * self.temperature) as f32;
        timings.kernel_start(KernelStage::LangevinOuStep)?;
        lan_ou_step(buffers, self.seed, step_index, alpha, kt)?;
        timings.kernel_stop(KernelStage::LangevinOuStep)?;

        timings.kernel_start(KernelStage::LangevinDriftHalf)?;
        lan_drift_half(buffers, dt)?;
        timings.kernel_stop(KernelStage::LangevinDriftHalf)?;

        // Force evaluation at the new positions.
        force_field.step(buffers, sim_box, timings)?;

        // BAOAB post-force: B(dt/2)
        timings.kernel_start(KernelStage::LangevinKickHalf)?;
        vv_kick(buffers, dt)?;
        timings.kernel_stop(KernelStage::LangevinKickHalf)?;

        Ok(())
    }
}

#[derive(Debug)]
pub struct LangevinBaoabBuilder;

impl IntegratorBuilder for LangevinBaoabBuilder {
    fn kind_name(&self) -> &'static str {
        "langevin-baoab"
    }

    fn build(
        &self,
        device: Arc<CudaDevice>,
        particle_count: usize,
        kind: &IntegratorKind,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        let _ = device;
        let _ = particle_count;
        match kind {
            IntegratorKind::LangevinBaoab {
                friction,
                temperature,
                seed,
            } => Ok(Box::new(LangevinBaoabState {
                friction: *friction,
                temperature: *temperature,
                seed: *seed,
            })),
            other => Err(IntegratorError::UnknownKind(other.name().to_string())),
        }
    }
}
