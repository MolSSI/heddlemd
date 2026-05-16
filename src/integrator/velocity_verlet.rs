// rq-09a2e15f

use crate::gpu::{
    GpuContext, LosslessBuffers, ParticleBuffers, vv_kick, vv_kick_drift,
    vv_kick_drift_lossless, vv_kick_lossless,
};
use crate::forces::ForceField;
use crate::io::config::IntegratorKind;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::{Integrator, IntegratorBuilder, IntegratorError};

#[derive(Debug)]
pub struct VelocityVerletState {
    lossless: Option<LosslessBuffers>,
}

impl Integrator for VelocityVerletState {
    fn step(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        force_field: &mut ForceField,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), IntegratorError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }

        if let Some(ll) = self.lossless.as_mut() {
            timings.kernel_start(KernelStage::VV_KICK_DRIFT_LOSSLESS)?;
            vv_kick_drift_lossless(buffers, ll, sim_box, dt)?;
            timings.kernel_stop(KernelStage::VV_KICK_DRIFT_LOSSLESS)?;
        } else {
            timings.kernel_start(KernelStage::VV_KICK_DRIFT)?;
            vv_kick_drift(buffers, sim_box, dt)?;
            timings.kernel_stop(KernelStage::VV_KICK_DRIFT)?;
        }

        force_field.step(buffers, sim_box, timings)?;

        if let Some(ll) = self.lossless.as_mut() {
            timings.kernel_start(KernelStage::VV_KICK_LOSSLESS)?;
            vv_kick_lossless(buffers, ll, dt)?;
            timings.kernel_stop(KernelStage::VV_KICK_LOSSLESS)?;
        } else {
            timings.kernel_start(KernelStage::VV_KICK)?;
            vv_kick(buffers, dt)?;
            timings.kernel_stop(KernelStage::VV_KICK)?;
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
        gpu: &GpuContext,
        particle_count: usize,
        kind: &IntegratorKind,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        match kind {
            IntegratorKind::VelocityVerlet { lossless } => {
                let buffers = if *lossless {
                    Some(LosslessBuffers::new(gpu, particle_count)?)
                } else {
                    None
                };
                Ok(Box::new(VelocityVerletState { lossless: buffers }))
            }
            other => Err(IntegratorError::UnknownKind(other.name().to_string())),
        }
    }
}
