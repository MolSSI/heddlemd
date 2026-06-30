// rq-09a2e15f


use serde::Deserialize;

#[cfg(not(feature = "f64"))]
use crate::gpu::{LosslessBuffers, vv_kick_drift_lossless, vv_kick_lossless};
use crate::gpu::{GpuContext, ParticleBuffers, vv_kick, vv_kick_drift};
use crate::io::config::ConfigError;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::{
    ConstraintCapableIntegrator, Integrator, IntegratorBuilder, IntegratorError, StepPlan,
    SubStep,
};
use crate::precision::Real;

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
    #[cfg(not(feature = "f64"))]
    lossless: Option<LosslessBuffers>,
}

impl VelocityVerletState {
    pub fn new(
        gpu: &GpuContext,
        particle_count: usize,
        lossless: bool,
    ) -> Result<Self, IntegratorError> {
        #[cfg(not(feature = "f64"))]
        {
            let buffers = if lossless {
                Some(LosslessBuffers::new(gpu, particle_count)?)
            } else {
                None
            };
            return Ok(VelocityVerletState { lossless: buffers });
        }
        #[cfg(feature = "f64")]
        {
            let _ = (gpu, particle_count, lossless);
            return Ok(VelocityVerletState {});
        }
    }
}

impl VelocityVerletState {
    #[cfg(not(feature = "f64"))]
    fn is_lossless(&self) -> bool {
        self.lossless.is_some()
    }
    #[cfg(feature = "f64")]
    fn is_lossless(&self) -> bool {
        false
    }
}

impl crate::integrator::PostForcePerParticle for VelocityVerletState {
    fn post_force_per_particle_fragment(
        &self,
    ) -> crate::forces::PerParticleFragment {
        let lossless = self.is_lossless();
        let (entry_point_args, per_thread_body) = if lossless {
            (
                String::from(
                    "    double *vv_velocities_x_lo,\n\
                     \x20   double *vv_velocities_y_lo,\n\
                     \x20   double *vv_velocities_z_lo,\n\
                     \x20   Real vv_dt,\n",
                ),
                String::from(
                    "        Real m = masses[i];\n\
                     \x20       Real ax = forces_x[i] / m;\n\
                     \x20       Real ay = forces_y[i] / m;\n\
                     \x20       Real az = forces_z[i] / m;\n\
                     \x20       Real half_dt = vv_dt * R(0.5);\n\
                     \x20       double dvx = (double)(ax * half_dt);\n\
                     \x20       double dvy = (double)(ay * half_dt);\n\
                     \x20       double dvz = (double)(az * half_dt);\n\
                     \x20       double ext_vx = (double)velocities_x[i] + vv_velocities_x_lo[i] + dvx;\n\
                     \x20       double ext_vy = (double)velocities_y[i] + vv_velocities_y_lo[i] + dvy;\n\
                     \x20       double ext_vz = (double)velocities_z[i] + vv_velocities_z_lo[i] + dvz;\n\
                     \x20       Real new_vx = (Real)ext_vx;\n\
                     \x20       Real new_vy = (Real)ext_vy;\n\
                     \x20       Real new_vz = (Real)ext_vz;\n\
                     \x20       vv_velocities_x_lo[i] = ext_vx - (double)new_vx;\n\
                     \x20       vv_velocities_y_lo[i] = ext_vy - (double)new_vy;\n\
                     \x20       vv_velocities_z_lo[i] = ext_vz - (double)new_vz;\n\
                     \x20       velocities_x[i] = new_vx;\n\
                     \x20       velocities_y[i] = new_vy;\n\
                     \x20       velocities_z[i] = new_vz;",
                ),
            )
        } else {
            (
                String::from("    Real vv_dt,\n"),
                String::from(
                    "        Real m = masses[i];\n\
                     \x20       Real ax = forces_x[i] / m;\n\
                     \x20       Real ay = forces_y[i] / m;\n\
                     \x20       Real az = forces_z[i] / m;\n\
                     \x20       Real half_dt = vv_dt * R(0.5);\n\
                     \x20       velocities_x[i] += ax * half_dt;\n\
                     \x20       velocities_y[i] += ay * half_dt;\n\
                     \x20       velocities_z[i] += az * half_dt;",
                ),
            )
        };
        crate::forces::PerParticleFragment {
            label: "velocity_verlet",
            helper_source: String::new(),
            entry_point_args,
            per_thread_body,
        }
    }

    fn bind_post_force_per_particle_args(
        &self,
        ctx: &crate::forces::PostForceBindContext<'_>,
        builder: &mut crate::forces::ForceLaunchBuilder,
    ) {
        #[cfg(not(feature = "f64"))]
        if let Some(ll) = self.lossless.as_ref() {
            builder.push_device_buffer(&ll.velocities_x_lo);
            builder.push_device_buffer(&ll.velocities_y_lo);
            builder.push_device_buffer(&ll.velocities_z_lo);
        }
        builder.push_scalar::<Real>(ctx.dt);
    }}

impl Integrator for VelocityVerletState {
    // rq-aa68f468
    fn plan(&self, dt: Real) -> StepPlan {
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

    // rq-9c5226e5 — VV half-kick fragment for the composed post-force
    // per-particle kernel. The fragment writes velocities (and the
    // matching `_lo` compensation buffers when lossless mode is
    // active). The standalone `vv_kick` / `vv_kick_lossless`
    // entry points are not launched on the JIT path; the runner's
    // composed-kernel dispatch handles the trailing half-kick.
    fn post_force_per_particle(&self) -> Option<&dyn crate::integrator::PostForcePerParticle> {
        Some(self)
    }


    // rq-83e752cd
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
                #[cfg(not(feature = "f64"))]
                if let Some(ll) = self.lossless.as_mut() {
                    timings.kernel_start(KernelStage::VV_KICK_DRIFT_LOSSLESS)?;
                    vv_kick_drift_lossless(buffers, ll, sim_box, *dt)?;
                    timings.kernel_stop(KernelStage::VV_KICK_DRIFT_LOSSLESS)?;
                    return Ok(());
                }
                timings.kernel_start(KernelStage::VV_KICK_DRIFT)?;
                vv_kick_drift(buffers, sim_box, *dt)?;
                timings.kernel_stop(KernelStage::VV_KICK_DRIFT)?;
                Ok(())
            }
            SubStep::KickHalf { dt, .. } => {
                #[cfg(not(feature = "f64"))]
                if let Some(ll) = self.lossless.as_mut() {
                    timings.kernel_start(KernelStage::VV_KICK_LOSSLESS)?;
                    vv_kick_lossless(buffers, ll, *dt)?;
                    timings.kernel_stop(KernelStage::VV_KICK_LOSSLESS)?;
                    return Ok(());
                }
                timings.kernel_start(KernelStage::VV_KICK)?;
                vv_kick(buffers, *dt)?;
                timings.kernel_stop(KernelStage::VV_KICK)?;
                Ok(())
            }
            other => Err(IntegratorError::UnexpectedSubStep {
                variant: other.variant_name(),
            }),
        }
    }
}

// rq-0e26dde0 — Velocity-Verlet's plan `[KickDrift, ForceEval, KickHalf]`
// lines up with the constraint slot's hook positions (one drift bracket
// plus a terminal kick). The lossless mode rejects hook installation at
// runtime — its compensated-sum bookkeeping doesn't yet account for
// constraint corrections.
impl ConstraintCapableIntegrator for VelocityVerletState {
    fn check_accepts_constraints_now(&self) -> Result<(), &'static str> {
        #[cfg(not(feature = "f64"))]
        if self.lossless.is_some() {
            return Err("velocity-Verlet in lossless mode does not yet support constraints");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct VelocityVerletBuilder;

use crate::registry::KindedBuilder;

impl KindedBuilder for VelocityVerletBuilder {
    fn kind_name(&self) -> &'static str {
        "velocity-verlet"
    }}

impl IntegratorBuilder for VelocityVerletBuilder {
    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> {
        let p = deserialize_params(params)?;
        #[cfg(feature = "f64")]
        if p.lossless {
            return Err(ConfigError::LosslessUnsupportedInF64Build);
        }
        let _ = p;
        Ok(())
    }

    // rq-9331ede2
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
        #[cfg(not(feature = "f64"))]
        {
            let buffers = if params.lossless {
                Some(LosslessBuffers::new(gpu, particle_count)?)
            } else {
                None
            };
            return Ok(Box::new(VelocityVerletState { lossless: buffers }));
        }
        #[cfg(feature = "f64")]
        {
            let _ = (gpu, particle_count, params);
            return Ok(Box::new(VelocityVerletState {}));
        }
    }
}

// rq-2093594f rq-e20b2f39
crate::gpu_kernels! {
    module: "integrate",
    ptx: crate::kernels::INTEGRATE,
    struct: IntegrateKernels,
    kernels: [
        vv_kick_drift,
        vv_kick,
        #[cfg(not(feature = "f64"))] vv_kick_drift_lossless,
        #[cfg(not(feature = "f64"))] vv_kick_lossless,
    ],
    stages: {
        VV_KICK_DRIFT          = "vv_kick_drift",
        VV_KICK                = "vv_kick",
        VV_KICK_DRIFT_LOSSLESS = "vv_kick_drift_lossless",
        VV_KICK_LOSSLESS       = "vv_kick_lossless",
    },
}
