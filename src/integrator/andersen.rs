// rq-5e059f6b

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use serde::Deserialize;

use crate::gpu::{GpuContext, GpuError, ParticleBuffers, andersen_resample};
use crate::io::config::ConfigError;
use crate::timings::{KernelStage, Timings};

use super::{Thermostat, ThermostatBuilder, ThermostatError};
use crate::precision::Real;

// rq-1f87880c
#[derive(Debug, Clone, Deserialize, serde::Serialize, crate::units::Convert)]
#[serde(deny_unknown_fields)]
pub struct AndersenParams {
    pub temperature: crate::units::Temperature,
    pub collision_rate: crate::units::InverseTime,
    pub seed: u64,
}

fn deserialize_params(params: &toml::Value) -> Result<AndersenParams, ConfigError> {
    params
        .clone()
        .try_into::<AndersenParams>()
        .map_err(|e| crate::io::config::translate_params_error("thermostat", e))
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

fn require_finite_non_negative(field: &str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() || value < 0.0 {
        return Err(ConfigError::InvalidValue {
            field: field.to_string(),
            reason: format!("value must be finite and >= 0, got {value}"),
        });
    }
    Ok(())
}

// rq-feba0a88
#[derive(Debug)]
pub struct AndersenThermostat {
    pub temperature: f64,
    pub collision_rate: f64,
    pub seed: u64,
    pub draw_counter: u64,
    pub kt: f64,
    /// Legacy field retained for diagnostic compatibility. Always
    /// zero; the per-step `(K_new − K_old)` accounting that the
    /// standalone path tracked is not reproduced inside the
    /// JIT-composed post-force per-particle kernel.
    pub cumulative_injection: f64,
    /// Device-resident Philox counter. `andersen_resample` reads it and
    /// increments it (one lane in the first block) after the per-thread
    /// draws, so successive launches — including across runs — draw a
    /// fresh Philox sequence. Persists across the run.
    pub draw_counter_device: cudarc::driver::CudaSlice<u64>,
}

impl AndersenThermostat {
    fn new(
        gpu: &GpuContext,
        _particle_count: usize,
        temperature: f64,
        collision_rate: f64,
        seed: u64,
    ) -> Result<Self, GpuError> {
        // k_B = 1 in atomic units; temperature is already k_B · T.
        let kt = temperature;
        let draw_counter_device =
            gpu.device.alloc_zeros::<u64>(1).map_err(GpuError::from)?;
        Ok(AndersenThermostat {
            temperature,
            collision_rate,
            seed,
            draw_counter: 0,
            kt,
            cumulative_injection: 0.0,
            draw_counter_device,
        })
    }

    pub fn flush_pending_injection(
        &mut self,
        device: &Arc<CudaDevice>,
    ) -> Result<(), GpuError> {
        // Refresh the host-side draw_counter cache for diagnostics.
        let mut host_counter = [0_u64; 1];
        device
            .dtoh_sync_copy_into(&self.draw_counter_device, &mut host_counter)
            .map_err(GpuError::from)?;
        self.draw_counter = host_counter[0];
        Ok(())
    }
}

impl Thermostat for AndersenThermostat {
    // rq-7a124d43 rq-a060db3f — Andersen contributes no composed fragment;
    // on a coupling step `apply_post` launches the standalone
    // `andersen_resample` kernel, which performs the per-particle Bernoulli
    // draw and conditional Maxwell-Boltzmann resample and increments the
    // device draw counter so successive launches draw fresh Philox streams.
    // `dt` is the effective coupling timestep, so the collision probability
    // `1 - exp`-style clamp uses the elapsed interval.
    fn apply_post(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: Real,
        timings: &mut Timings,
    ) -> Result<(), ThermostatError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }
        let p_collision =
            ((self.collision_rate as f64) * (dt as f64)).clamp(0.0, 1.0) as Real;
        timings.kernel_start(KernelStage::ANDERSEN_RESAMPLE)?;
        andersen_resample(
            buffers,
            &mut self.draw_counter_device,
            self.seed,
            p_collision,
            self.kt as Real,
        )?;
        timings.kernel_stop(KernelStage::ANDERSEN_RESAMPLE)?;
        Ok(())
    }

    fn flush_pending_injection(
        &mut self,
        device: &Arc<CudaDevice>,
    ) -> Result<(), ThermostatError> {
        AndersenThermostat::flush_pending_injection(self, device)
            .map_err(ThermostatError::from)
    }

    // rq-1163481e
    fn log_column_names(&self) -> &'static [(&'static str, crate::units::Dimension)] {
        &[("andersen_conserved", crate::units::Dimension::Energy)]
    }

    // rq-6d2daea0 — The cumulative-injection accounting the conserved
    // column would need requires a kinetic-energy measurement before AND
    // after the resample. Those reductions are not performed, so the
    // conserved column reports `K + U` without the injection correction;
    // users running long Andersen trajectories should expect
    // detailed-balance drift in this column.
    fn log_column_values(
        &self,
        kinetic_energy: f64,
        potential_energy: f64,
    ) -> Vec<f64> {
        vec![kinetic_energy + potential_energy]
    }
}

// rq-fd0cef60
#[derive(Debug, Clone)]
pub struct AndersenBuilder;

use crate::registry::KindedBuilder;

impl KindedBuilder for AndersenBuilder {
    fn kind_name(&self) -> &'static str {
        "andersen"
    }
    fn convert_params(
        &self,
        units: crate::units::UnitSystem,
        params: &mut toml::Value,
    ) -> Result<(), crate::io::config::ConfigError> {
        crate::registry::convert_params_in_place::<AndersenParams>(units, params)
    }
}

impl ThermostatBuilder for AndersenBuilder {
    fn graph_compatible(&self, _params: &toml::Value) -> bool {
        // A thermostat's coupling never runs inside a captured graph:
        // the graph captures non-coupling steps only, and coupling steps
        // (where `andersen_resample` launches) run on the per-step path.
        // So thermostat graph-compatibility does not gate eligibility.
        true
    }

    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> {
        let p = deserialize_params(params)?;
        require_finite_positive("thermostat.temperature", p.temperature.0)?;
        require_finite_non_negative("thermostat.collision_rate", p.collision_rate.0)?;
        Ok(())
    }

    // `_n_constraints` is deliberately unused: Andersen needs no DOF
    // count. Collisions resample a particle's full velocity from the
    // Maxwell-Boltzmann distribution independently per particle, so
    // momentum is NOT conserved and no `3N − n_constraints − 3` target
    // enters the dynamics (same reported-temperature convention note
    // as Langevin; see `rqm/integration/andersen.md`).
    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        _n_constraints: usize,
        params: &toml::Value,
    ) -> Result<Box<dyn Thermostat>, ThermostatError> {
        let p = deserialize_params(params)
            .map_err(|_| ThermostatError::UnknownKind("andersen (malformed params)".into()))?;
        let state = AndersenThermostat::new(
            gpu,
            particle_count,
            p.temperature.0,
            p.collision_rate.0,
            p.seed,
        )?;
        Ok(Box::new(state))
    }
}

// rq-2093594f rq-5e059f6b — Andersen's per-particle resample kernel.
// `apply_post` launches it directly on coupling steps (a thermostat
// contributes no composed post-force fragment).
crate::gpu_kernels! {
    module: "andersen",
    ptx: crate::kernels::ANDERSEN,
    struct: AndersenKernels,
    kernels: [andersen_resample],
    stages: {
        ANDERSEN_RESAMPLE = "andersen_resample",
    },
}
