// rq-09a2e15f

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction};
use cudarc::nvrtc::Ptx;
use serde::Deserialize;

use crate::gpu::device::get_func;
use crate::gpu::{
    GpuContext, GpuError, LosslessBuffers, ParticleBuffers, vv_kick, vv_kick_drift,
    vv_kick_drift_lossless, vv_kick_lossless,
};
use crate::kernels;
use crate::io::config::ConfigError;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::{Integrator, IntegratorBuilder, IntegratorError, StepPlan, SubStep};

// rq-1f87880c — typed parameter struct for the "velocity-verlet"
// builder, deserialised from the `[integrator]` section's
// `SlotConfig::params`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VelocityVerletParams {
    #[serde(default)]
    pub lossless: bool,
}

fn deserialize_params(params: &toml::Value) -> Result<VelocityVerletParams, ConfigError> {
    params
        .clone()
        .try_into::<VelocityVerletParams>()
        .map_err(|e| crate::io::config::translate_params_error("integrator", e))
}

#[derive(Debug)]
pub struct VelocityVerletState {
    lossless: Option<LosslessBuffers>,
}

impl Integrator for VelocityVerletState {
    // rq-aa68f468
    fn plan(&self, dt: f32) -> StepPlan {
        StepPlan {
            steps: vec![
                SubStep::KickDrift { dt, label: "vv_kick_drift" },
                SubStep::ForceEval {
                    class: None,
                    level: Some(crate::forces::AggregateLevel::ForcesOnly),
                },
                SubStep::KickHalf { dt, label: "vv_kick" },
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
            SubStep::KickDrift { dt, .. } => {
                if let Some(ll) = self.lossless.as_mut() {
                    timings.kernel_start(KernelStage::VV_KICK_DRIFT_LOSSLESS)?;
                    vv_kick_drift_lossless(buffers, ll, sim_box, *dt)?;
                    timings.kernel_stop(KernelStage::VV_KICK_DRIFT_LOSSLESS)?;
                } else {
                    timings.kernel_start(KernelStage::VV_KICK_DRIFT)?;
                    vv_kick_drift(buffers, sim_box, *dt)?;
                    timings.kernel_stop(KernelStage::VV_KICK_DRIFT)?;
                }
                Ok(())
            }
            SubStep::KickHalf { dt, .. } => {
                if let Some(ll) = self.lossless.as_mut() {
                    timings.kernel_start(KernelStage::VV_KICK_LOSSLESS)?;
                    vv_kick_lossless(buffers, ll, *dt)?;
                    timings.kernel_stop(KernelStage::VV_KICK_LOSSLESS)?;
                } else {
                    timings.kernel_start(KernelStage::VV_KICK)?;
                    vv_kick(buffers, *dt)?;
                    timings.kernel_stop(KernelStage::VV_KICK)?;
                }
                Ok(())
            }
            other => Err(IntegratorError::UnexpectedSubStep {
                variant: other.variant_name(),
            }),
        }
    }
}

#[derive(Debug)]
pub struct VelocityVerletBuilder;

impl IntegratorBuilder for VelocityVerletBuilder {
    fn kind_name(&self) -> &'static str {
        "velocity-verlet"
    }

    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> {
        deserialize_params(params).map(|_| ())
    }

    fn supports_constraints(&self, params: &toml::Value) -> bool {
        // Reads `params.lossless`; falls back to `false` if params are
        // malformed (callers should run validate_params first).
        let lossless = deserialize_params(params).map(|p| p.lossless).unwrap_or(false);
        !lossless
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        _n_constraints: usize,
        params: &toml::Value,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        let params = deserialize_params(params)
            .map_err(|_| IntegratorError::UnknownKind("velocity-verlet (malformed params)".into()))?;
        let buffers = if params.lossless {
            Some(LosslessBuffers::new(gpu, particle_count)?)
        } else {
            None
        };
        Ok(Box::new(VelocityVerletState { lossless: buffers }))
    }
}

// rq-2093594f rq-e20b2f39
#[derive(Debug, Clone)]
pub struct IntegrateKernels {
    pub vv_kick_drift: CudaFunction,
    pub vv_kick: CudaFunction,
    pub vv_kick_drift_lossless: CudaFunction,
    pub vv_kick_lossless: CudaFunction,
}

impl IntegrateKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::INTEGRATE),
            "integrate",
            &[
                "vv_kick_drift",
                "vv_kick",
                "vv_kick_drift_lossless",
                "vv_kick_lossless",
            ],
        )?;
        Ok(IntegrateKernels {
            vv_kick_drift: get_func(device, "integrate", "vv_kick_drift")?,
            vv_kick: get_func(device, "integrate", "vv_kick")?,
            vv_kick_drift_lossless: get_func(device, "integrate", "vv_kick_drift_lossless")?,
            vv_kick_lossless: get_func(device, "integrate", "vv_kick_lossless")?,
        })
    }
}
