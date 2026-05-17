// rq-d5a4f220

use crate::gpu::{GpuContext, ParticleBuffers, lan_drift_half, lan_ou_step, vv_kick};
use crate::io::config::IntegratorKind;
use crate::io::log_output::BOLTZMANN_J_PER_K;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::{Integrator, IntegratorBuilder, IntegratorError, StepPlan, SubStep};

#[derive(Debug)]
pub struct LangevinBaoabState {
    pub friction: f64,
    pub temperature: f64,
    pub seed: u64,
    pub draw_counter: u64,
}

impl Integrator for LangevinBaoabState {
    // rq-aa68f468
    fn plan(&self, dt: f32) -> StepPlan {
        // The two Drift sub-steps internally use dt/2; the integrator's
        // execute() reads `dt` from the SubStep and applies the
        // appropriate factor inside the `lan_drift_half` kernel.
        StepPlan {
            steps: vec![
                SubStep::KickHalf { dt, label: "B" },
                SubStep::Drift { dt, label: "A_pre" },
                SubStep::Custom { dt, label: "O" },
                SubStep::Drift { dt, label: "A_post" },
                SubStep::ForceEval,
                SubStep::KickHalf { dt, label: "B" },
            ],
        }
    }

    fn execute(
        &mut self,
        substep: &SubStep,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        timings: &mut Timings,
    ) -> Result<(), IntegratorError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }
        match substep {
            SubStep::KickHalf { dt, .. } => {
                timings.kernel_start(KernelStage::LANGEVIN_KICK_HALF)?;
                vv_kick(buffers, *dt)?;
                timings.kernel_stop(KernelStage::LANGEVIN_KICK_HALF)?;
                Ok(())
            }
            SubStep::Drift { dt, .. } => {
                timings.kernel_start(KernelStage::LANGEVIN_DRIFT_HALF)?;
                lan_drift_half(buffers, sim_box, *dt)?;
                timings.kernel_stop(KernelStage::LANGEVIN_DRIFT_HALF)?;
                Ok(())
            }
            SubStep::Custom { dt, label } if *label == "O" => {
                let alpha = (-(self.friction as f32) * *dt).exp();
                let kt = (BOLTZMANN_J_PER_K * self.temperature) as f32;
                self.draw_counter += 1;
                timings.kernel_start(KernelStage::LANGEVIN_OU_STEP)?;
                lan_ou_step(buffers, self.seed, self.draw_counter, alpha, kt)?;
                timings.kernel_stop(KernelStage::LANGEVIN_OU_STEP)?;
                Ok(())
            }
            other => Err(IntegratorError::UnexpectedSubStep {
                variant: other.variant_name(),
            }),
        }
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
        gpu: &GpuContext,
        particle_count: usize,
        kind: &IntegratorKind,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        let _ = gpu;
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
                draw_counter: 0,
            })),
            other => Err(IntegratorError::UnknownKind(other.name().to_string())),
        }
    }
}
