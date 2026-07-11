//! RESPA multiple-timestep integrator (Tuckerman-Berne-Martyna 1992).
//! Impulse splitting over the Fast/Slow force classes: fast-class
//! forces kick every inner step, slow-class forces kick as
//! half-impulses at the outer boundaries. See
//! `rqm/integration/respa.md`.

// rq-564d914c

use serde::Deserialize;

use crate::gpu::ParticleBuffers;
use crate::forces::ForceClass;
use crate::io::config::ConfigError;
use crate::pbc::SimulationBox;
use crate::timings::Timings;

use super::{
    Integrator, IntegratorBuilder, IntegratorError, KickSource, StepPlan, SubStep,
};
use crate::precision::Real;

// rq-9f8480ae
/// Typed parameter struct for the "respa" builder, deserialised from
/// the `[integrator]` section's `SlotConfig::params`. `lossless` is
/// accepted by the schema so its rejection can carry a targeted error
/// (see `validate_params`); compensated inner-kick accumulation is not
/// implemented.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RespaParams {
    pub n_inner: u32,
    #[serde(default)]
    pub lossless: bool,
}

fn deserialize_params(params: &toml::Value) -> Result<RespaParams, ConfigError> {
    params
        .clone()
        .try_into::<RespaParams>()
        .map_err(|e| crate::io::config::translate_params_error("integrator", e))
}

// rq-6922b0be
#[derive(Debug)]
pub struct RespaIntegrator {
    n_inner: u32,
}

impl RespaIntegrator {
    pub fn new(n_inner: u32) -> Self {
        RespaIntegrator { n_inner }
    }
}

impl Integrator for RespaIntegrator {
    // rq-0af322d7 — the inner loop unrolled: 3ν + 3 sub-steps whose
    // shape depends only on `dt` and the construction-time `n_inner`.
    fn plan(&self, dt: Real) -> StepPlan {
        let nu = self.n_inner;
        let delta = dt / nu as Real;
        let mut steps = Vec::with_capacity(3 * nu as usize + 3);
        steps.push(SubStep::KickHalf {
            dt,
            label: "respa_outer_kick",
            source: KickSource::Class(ForceClass::Slow),
        });
        for _ in 0..nu {
            steps.push(SubStep::KickDrift {
                dt: delta,
                label: "respa_inner_kick_drift",
                source: KickSource::Class(ForceClass::Fast),
            });
            steps.push(SubStep::ForceEval {
                class: Some(ForceClass::Fast),
                level: None,
            });
            steps.push(SubStep::KickHalf {
                dt: delta,
                label: "respa_inner_kick",
                source: KickSource::Class(ForceClass::Fast),
            });
        }
        steps.push(SubStep::ForceEval {
            class: Some(ForceClass::Slow),
            level: None,
        });
        steps.push(SubStep::KickHalf {
            dt,
            label: "respa_outer_kick",
            source: KickSource::Class(ForceClass::Slow),
        });
        StepPlan { steps }
    }

    // Every RESPA sub-step is runner-dispatched (force evaluations,
    // class-sourced kicks including the trailing outer kick); `execute`
    // receives nothing in normal operation.
    fn execute(
        &mut self,
        substep: &SubStep,
        _buffers: &mut ParticleBuffers,
        _sim_box: &mut SimulationBox,
        _timings: &mut Timings,
    ) -> Result<(), IntegratorError> {
        Err(IntegratorError::UnexpectedSubStep {
            variant: substep.variant_name(),
        })
    }

}

// rq-e8550f96
#[derive(Debug, Clone)]
pub struct RespaBuilder;

impl crate::registry::KindedBuilder for RespaBuilder {
    fn kind_name(&self) -> &'static str {
        "respa"
    }
}

impl IntegratorBuilder for RespaBuilder {
    // rq-9f8480ae rq-cb480c95
    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> {
        let p = deserialize_params(params)?;
        if p.n_inner == 0 {
            return Err(ConfigError::InvalidValue {
                field: "integrator.n_inner".to_string(),
                reason: "must be >= 1".to_string(),
            });
        }
        if p.lossless {
            return Err(ConfigError::InvalidValue {
                field: "integrator.lossless".to_string(),
                reason: "kind \"respa\" does not support lossless mode \
                         (compensated inner-kick accumulation is not implemented)"
                    .to_string(),
            });
        }
        Ok(())
    }

    // rq-cb480c95 — correct constrained RESPA requires RATTLE after
    // every inner velocity update, which the constraint hook contract
    // does not express.
    fn supports_constraints(&self, _params: &toml::Value) -> bool {
        false
    }

    // rq-cb480c95 — RESPA-NPT splittings are a separate design.
    fn supports_barostat(&self, _params: &toml::Value) -> bool {
        false
    }

    fn build(
        &self,
        _gpu: &crate::gpu::GpuContext,
        _particle_count: usize,
        _n_constraints: usize,
        params: &toml::Value,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        let p = deserialize_params(params)
            .map_err(|_| IntegratorError::UnknownKind("respa (malformed params)".into()))?;
        Ok(Box::new(RespaIntegrator::new(p.n_inner.max(1))))
    }
}
