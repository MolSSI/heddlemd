// rq-4f386df8 rq-5cb33196 rq-67414c32
use std::sync::Arc;

use cudarc::driver::CudaDevice;

use crate::gpu::{
    GpuError, LosslessBuffers, ParticleBuffers, lan_drift_half, lan_ou_step, vv_kick,
    vv_kick_drift, vv_kick_drift_lossless, vv_kick_lossless,
};
use crate::io::log_output::BOLTZMANN_J_PER_K;
use crate::io::config::IntegratorKind;
use crate::timings::{KernelStage, Timings, TimingsError};

// rq-a5069572
#[derive(Debug)]
pub enum IntegratorError {
    Gpu(GpuError),
    Timings(TimingsError),
}

impl std::fmt::Display for IntegratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntegratorError::Gpu(e) => write!(f, "Gpu({e})"),
            IntegratorError::Timings(e) => write!(f, "Timings({e})"),
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

#[derive(Debug)]
pub struct VelocityVerletState {
    lossless: Option<LosslessBuffers>,
}

impl VelocityVerletState {
    fn pre_force_step(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: f32,
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
        Ok(())
    }

    fn post_force_step(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), IntegratorError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }
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

// rq-bcb0f58a
#[derive(Debug)]
pub struct LangevinBaoabState {
    pub friction: f64,
    pub temperature: f64,
    pub seed: u64,
}

impl LangevinBaoabState {
    fn pre_force_step(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: f32,
        step_index: u64,
        timings: &mut Timings,
    ) -> Result<(), IntegratorError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }

        // BAOAB pre: B(dt/2), A(dt/2), O(dt), A(dt/2)
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

        Ok(())
    }

    fn post_force_step(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), IntegratorError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }
        timings.kernel_start(KernelStage::LangevinKickHalf)?;
        vv_kick(buffers, dt)?;
        timings.kernel_stop(KernelStage::LangevinKickHalf)?;
        Ok(())
    }
}

// rq-e4c4ff61
#[derive(Debug)]
pub enum Integrator {
    VelocityVerlet(VelocityVerletState),
    LangevinBaoab(LangevinBaoabState),
}

impl Integrator {
    // rq-df39d15b rq-ad27732e
    pub fn new(
        device: Arc<CudaDevice>,
        particle_count: usize,
        kind: &IntegratorKind,
    ) -> Result<Self, IntegratorError> {
        match kind {
            IntegratorKind::VelocityVerlet { lossless } => {
                let buffers = if *lossless {
                    Some(LosslessBuffers::new(device, particle_count)?)
                } else {
                    None
                };
                Ok(Integrator::VelocityVerlet(VelocityVerletState {
                    lossless: buffers,
                }))
            }
            IntegratorKind::LangevinBaoab {
                friction,
                temperature,
                seed,
            } => {
                let _ = device;
                let _ = particle_count;
                Ok(Integrator::LangevinBaoab(LangevinBaoabState {
                    friction: *friction,
                    temperature: *temperature,
                    seed: *seed,
                }))
            }
        }
    }

    // rq-cf361ff5
    pub fn pre_force_step(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: f32,
        step_index: u64,
        timings: &mut Timings,
    ) -> Result<(), IntegratorError> {
        match self {
            Integrator::VelocityVerlet(state) => state.pre_force_step(buffers, dt, timings),
            Integrator::LangevinBaoab(state) => {
                state.pre_force_step(buffers, dt, step_index, timings)
            }
        }
    }

    // rq-700c7729
    pub fn post_force_step(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: f32,
        _step_index: u64,
        timings: &mut Timings,
    ) -> Result<(), IntegratorError> {
        match self {
            Integrator::VelocityVerlet(state) => state.post_force_step(buffers, dt, timings),
            Integrator::LangevinBaoab(state) => state.post_force_step(buffers, dt, timings),
        }
    }
}
