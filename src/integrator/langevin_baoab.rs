// rq-d5a4f220

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction};
use cudarc::nvrtc::Ptx;
use serde::Deserialize;

use cudarc::driver::CudaSlice;

use crate::gpu::device::get_func;
use crate::gpu::{GpuContext, GpuError, ParticleBuffers, lan_drift_half, lan_ou_step, vv_kick};
use crate::kernels;
use crate::io::config::ConfigError;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::{Integrator, IntegratorBuilder, IntegratorError, StepPlan, SubStep};
use crate::precision::Real;

// rq-1f87880c — typed parameter struct for the "langevin-baoab"
// builder, deserialised from the `[integrator]` section's
// `SlotConfig::params`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LangevinBaoabParams {
    pub friction: f64,
    pub temperature: f64,
    pub seed: u64,
}

fn deserialize_params(params: &toml::Value) -> Result<LangevinBaoabParams, ConfigError> {
    params
        .clone()
        .try_into::<LangevinBaoabParams>()
        .map_err(|e| crate::io::config::translate_params_error("integrator", e))
}

fn require_finite_positive(field: &str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(ConfigError::InvalidValue {
            field: field.to_string(),
            reason: format!("value must be finite and strictly positive, got {value}"),
        });
    }
    Ok(())
}

// rq-bcb0f58a
#[derive(Debug)]
pub struct LangevinBaoabState {
    pub friction: f64,
    pub temperature: f64,
    pub seed: u64,
    pub draw_counter: u64,
    /// Device-resident Philox counter. `lan_ou_step` reads from it; the
    /// launcher follows with `increment_u64_device`. Allows the
    /// integrator's O-step to be safely captured in a CUDA graph.
    draw_counter_device: CudaSlice<u64>,
}

impl LangevinBaoabState {
    /// Construct a `LangevinBaoabState` with the given parameters and a
    /// fresh device-resident draw counter pre-initialised to
    /// `draw_counter`. Used by tests that exercise the integrator
    /// directly.
    pub fn new(
        gpu: &GpuContext,
        friction: f64,
        temperature: f64,
        seed: u64,
        draw_counter: u64,
    ) -> Result<Self, GpuError> {
        let mut draw_counter_device = gpu.device.alloc_zeros::<u64>(1)?;
        if draw_counter != 0 {
            gpu.device
                .htod_sync_copy_into(&[draw_counter], &mut draw_counter_device)?;
        }
        Ok(LangevinBaoabState {
            friction,
            temperature,
            seed,
            draw_counter,
            draw_counter_device,
        })
    }
}

impl crate::integrator::PostForcePerParticle for LangevinBaoabState {
    fn post_force_per_particle_fragment(
        &self,
    ) -> crate::forces::PerParticleFragment {
        crate::forces::PerParticleFragment {
            label: "langevin_baoab",
            helper_source: String::new(),
            entry_point_args: String::from("    Real langevin_dt,\n"),
            per_thread_body: String::from(
                "        Real m_l = masses[i];\n\
                 \x20       Real ax_l = forces_x[i] / m_l;\n\
                 \x20       Real ay_l = forces_y[i] / m_l;\n\
                 \x20       Real az_l = forces_z[i] / m_l;\n\
                 \x20       Real half_dt_l = langevin_dt * R(0.5);\n\
                 \x20       velocities_x[i] += ax_l * half_dt_l;\n\
                 \x20       velocities_y[i] += ay_l * half_dt_l;\n\
                 \x20       velocities_z[i] += az_l * half_dt_l;",
            ),
        }
    }

    fn bind_post_force_per_particle_args(
        &self,
        ctx: &crate::forces::PostForceBindContext<'_>,
        builder: &mut crate::forces::ForceLaunchBuilder,
    ) {
        builder.push_scalar::<Real>(ctx.dt);
    }}

impl Integrator for LangevinBaoabState {
    // rq-aa68f468
    fn plan(&self, dt: Real) -> StepPlan {
        // The two Drift sub-steps internally use dt/2; the integrator's
        // execute() reads `dt` from the SubStep and applies the
        // appropriate factor inside the `lan_drift_half` kernel.
        StepPlan {
            steps: vec![
                SubStep::KickHalf { dt, label: "B" },
                SubStep::Drift { dt, label: "A_pre" },
                SubStep::Custom { dt, label: "O" },
                SubStep::Drift { dt, label: "A_post" },
                SubStep::ForceEval {
                    class: None,
                    level: Some(crate::forces::AggregateLevel::ForcesOnly),
                },
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
                let alpha = (-(self.friction as Real) * *dt).exp();
                // k_B = 1 in atomic units; temperature is already k_B · T.
                let kt = self.temperature as Real;
                timings.kernel_start(KernelStage::LANGEVIN_OU_STEP)?;
                lan_ou_step(buffers, &mut self.draw_counter_device, self.seed, alpha, kt)?;
                timings.kernel_stop(KernelStage::LANGEVIN_OU_STEP)?;
                self.draw_counter += 1;
                Ok(())
            }
            other => Err(IntegratorError::UnexpectedSubStep {
                variant: other.variant_name(),
            }),
        }
    }

    // rq-9c5226e5 — Langevin-BAOAB's post-force phase is the trailing
    // `B` half-kick (`vv_kick`-equivalent). Mid-plan `O` / `A` work
    // remains as separate kernel launches via `execute(...)`.
    fn post_force_per_particle(&self) -> Option<&dyn crate::integrator::PostForcePerParticle> {
        Some(self)
    }

}

#[derive(Debug, Clone)]
pub struct LangevinBaoabBuilder;

use crate::registry::KindedBuilder;

impl KindedBuilder for LangevinBaoabBuilder {
    fn kind_name(&self) -> &'static str {
        "langevin-baoab"
    }}

impl IntegratorBuilder for LangevinBaoabBuilder {
    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> {
        let p = deserialize_params(params)?;
        require_finite_positive("integrator.friction", p.friction)?;
        require_finite_positive("integrator.temperature", p.temperature)?;
        Ok(())
    }

    fn owns_thermostat(&self, _params: &toml::Value) -> bool {
        true
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        _n_constraints: usize,
        params: &toml::Value,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        let _ = particle_count;
        let p = deserialize_params(params)
            .map_err(|_| IntegratorError::UnknownKind("langevin-baoab (malformed params)".into()))?;
        let draw_counter_device = gpu
            .device
            .alloc_zeros::<u64>(1)
            .map_err(|e| IntegratorError::Gpu(GpuError::from(e)))?;
        Ok(Box::new(LangevinBaoabState {
            friction: p.friction,
            temperature: p.temperature,
            seed: p.seed,
            draw_counter: 0,
            draw_counter_device,
        }))
    }
}

// rq-2093594f
#[derive(Debug, Clone)]
pub struct LangevinKernels {
    pub lan_drift_half: CudaFunction,
    pub lan_ou_step: CudaFunction,
}

impl LangevinKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::LANGEVIN),
            "langevin",
            &["lan_drift_half", "lan_ou_step"],
        )?;
        Ok(LangevinKernels {
            lan_drift_half: get_func(device, "langevin", "lan_drift_half")?,
            lan_ou_step: get_func(device, "langevin", "lan_ou_step")?,
        })
    }
}
