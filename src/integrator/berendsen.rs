// rq-25f24b26

use cudarc::driver::CudaSlice;

use serde::Deserialize;

use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, compute_kinetic_energy, rescale_velocities,
};
use crate::io::config::ConfigError;
use crate::io::log_output::BOLTZMANN_J_PER_K;
use crate::timings::{KernelStage, Timings};

use super::{Thermostat, ThermostatBuilder, ThermostatError};

// rq-1f87880c
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BerendsenParams {
    pub temperature: f64,
    pub tau: f64,
}

fn deserialize_params(params: &toml::Value) -> Result<BerendsenParams, ConfigError> {
    params
        .clone()
        .try_into::<BerendsenParams>()
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

// rq-f856f666
#[derive(Debug)]
pub struct BerendsenThermostat {
    pub temperature: f64,
    pub tau: f64,
    pub g_dof: u32,
    pub kt_target: f64,
    pub cumulative_injection: f64,
    ke_scratch: CudaSlice<f32>,
    most_recent_ke: f64,
}

impl BerendsenThermostat {
    fn new(
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
        temperature: f64,
        tau: f64,
    ) -> Result<Self, GpuError> {
        let g_dof =
            ((3 * particle_count) as i64 - n_constraints as i64 - 3).max(1) as u32;
        let kt_target = BOLTZMANN_J_PER_K * temperature;
        let ke_scratch = gpu.device.alloc_zeros::<f32>(1).map_err(GpuError::from)?;
        Ok(BerendsenThermostat {
            temperature,
            tau,
            g_dof,
            kt_target,
            cumulative_injection: 0.0,
            ke_scratch,
            most_recent_ke: 0.0,
        })
    }
}

impl Thermostat for BerendsenThermostat {
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

        if k_old <= 0.0 {
            self.most_recent_ke = 0.0;
            return Ok(());
        }

        let nf = self.g_dof as f64;
        let k_target = (nf / 2.0) * self.kt_target;
        let lambda_sq = (1.0 + ((dt as f64) / self.tau) * (k_target / k_old - 1.0)).max(0.0);
        let lambda = lambda_sq.sqrt();
        let factor = lambda as f32;

        timings.kernel_start(KernelStage::BERENDSEN_RESCALE_VELOCITIES)?;
        rescale_velocities(buffers, factor)?;
        timings.kernel_stop(KernelStage::BERENDSEN_RESCALE_VELOCITIES)?;

        let k_new = lambda_sq * k_old;
        self.cumulative_injection += k_new - k_old;
        self.most_recent_ke = k_new;
        Ok(())
    }

    // rq-c908bbf1
    fn log_column_names(&self) -> &'static [&'static str] {
        &["berendsen_conserved"]
    }

    // rq-3589910b
    fn log_column_values(
        &self,
        kinetic_energy: f64,
        potential_energy: f64,
    ) -> Vec<f64> {
        vec![kinetic_energy + potential_energy - self.cumulative_injection]
    }
}

// rq-6c9037a4
#[derive(Debug)]
pub struct BerendsenBuilder;

impl ThermostatBuilder for BerendsenBuilder {
    fn kind_name(&self) -> &'static str {
        "berendsen"
    }

    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> {
        let p = deserialize_params(params)?;
        require_finite_positive("thermostat.temperature", p.temperature)?;
        require_finite_positive("thermostat.tau", p.tau)?;
        Ok(())
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
        params: &toml::Value,
    ) -> Result<Box<dyn Thermostat>, ThermostatError> {
        let p = deserialize_params(params)
            .map_err(|_| ThermostatError::UnknownKind("berendsen (malformed params)".into()))?;
        let state = BerendsenThermostat::new(
            gpu,
            particle_count,
            n_constraints,
            p.temperature,
            p.tau,
        )?;
        Ok(Box::new(state))
    }
}
