// rq-5e059f6b

use cudarc::driver::CudaSlice;

use serde::Deserialize;

use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, andersen_resample, compute_kinetic_energy,
};
use crate::io::config::ConfigError;
use crate::io::log_output::BOLTZMANN_J_PER_K;
use crate::timings::{KernelStage, Timings};

use super::{Thermostat, ThermostatBuilder, ThermostatError};

// rq-1f87880c
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AndersenParams {
    pub temperature: f64,
    pub collision_rate: f64,
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
    pub cumulative_injection: f64,
    ke_scratch: CudaSlice<f32>,
    most_recent_ke: f64,
}

impl AndersenThermostat {
    fn new(
        gpu: &GpuContext,
        _particle_count: usize,
        temperature: f64,
        collision_rate: f64,
        seed: u64,
    ) -> Result<Self, GpuError> {
        let kt = BOLTZMANN_J_PER_K * temperature;
        let ke_scratch = gpu.device.alloc_zeros::<f32>(1).map_err(GpuError::from)?;
        Ok(AndersenThermostat {
            temperature,
            collision_rate,
            seed,
            draw_counter: 0,
            kt,
            cumulative_injection: 0.0,
            ke_scratch,
            most_recent_ke: 0.0,
        })
    }
}

impl Thermostat for AndersenThermostat {
    // rq-7a124d43
    fn apply_post(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), ThermostatError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }

        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k_old = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;

        self.draw_counter += 1;
        let p_collision = ((self.collision_rate as f64) * (dt as f64))
            .clamp(0.0, 1.0) as f32;
        let kt = self.kt as f32;
        timings.kernel_start(KernelStage::ANDERSEN_RESAMPLE)?;
        andersen_resample(buffers, self.seed, self.draw_counter, p_collision, kt)?;
        timings.kernel_stop(KernelStage::ANDERSEN_RESAMPLE)?;

        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k_new = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;

        self.cumulative_injection += k_new - k_old;
        self.most_recent_ke = k_new;
        Ok(())
    }

    // rq-1163481e
    fn log_column_names(&self) -> &'static [&'static str] {
        &["andersen_conserved"]
    }

    // rq-6d2daea0
    fn log_column_values(
        &self,
        kinetic_energy: f64,
        potential_energy: f64,
    ) -> Vec<f64> {
        vec![kinetic_energy + potential_energy - self.cumulative_injection]
    }
}

// rq-fd0cef60
#[derive(Debug)]
pub struct AndersenBuilder;

impl ThermostatBuilder for AndersenBuilder {
    fn kind_name(&self) -> &'static str {
        "andersen"
    }

    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> {
        let p = deserialize_params(params)?;
        require_finite_positive("thermostat.temperature", p.temperature)?;
        require_finite_non_negative("thermostat.collision_rate", p.collision_rate)?;
        Ok(())
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        params: &toml::Value,
    ) -> Result<Box<dyn Thermostat>, ThermostatError> {
        let p = deserialize_params(params)
            .map_err(|_| ThermostatError::UnknownKind("andersen (malformed params)".into()))?;
        let state = AndersenThermostat::new(
            gpu,
            particle_count,
            p.temperature,
            p.collision_rate,
            p.seed,
        )?;
        Ok(Box::new(state))
    }
}
