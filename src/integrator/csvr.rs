// rq-891232bf

use cudarc::driver::CudaSlice;

use serde::Deserialize;

use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, compute_kinetic_energy, rescale_velocities,
};
use crate::io::config::ConfigError;
use crate::timings::{KernelStage, Timings};

use super::philox::philox_normal;
use super::{Thermostat, ThermostatBuilder, ThermostatError};
use crate::precision::Real;

// rq-1f87880c
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CsvrParams {
    pub temperature: f64,
    pub tau: f64,
    pub seed: u64,
}

fn deserialize_params(params: &toml::Value) -> Result<CsvrParams, ConfigError> {
    params
        .clone()
        .try_into::<CsvrParams>()
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

// rq-47d91c7d
#[derive(Debug)]
pub struct CsvrThermostat {
    pub temperature: f64,
    pub tau: f64,
    pub seed: u64,
    pub draw_counter: u64,
    pub g_dof: u32,
    pub kt_target: f64,
    pub cumulative_injection: f64,
    ke_scratch: CudaSlice<Real>,
    most_recent_ke: f64,
}

impl CsvrThermostat {
    fn new(
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
        temperature: f64,
        tau: f64,
        seed: u64,
    ) -> Result<Self, GpuError> {
        let g_dof =
            ((3 * particle_count) as i64 - n_constraints as i64 - 3).max(1) as u32;
        // k_B = 1 in atomic units; the temperature parameter is already
        // `k_B · T` in Hartrees, so `kt_target` is just the temperature.
        let kt_target = temperature;
        let ke_scratch = gpu.device.alloc_zeros::<Real>(1).map_err(GpuError::from)?;
        Ok(CsvrThermostat {
            temperature,
            tau,
            seed,
            draw_counter: 0,
            g_dof,
            kt_target,
            cumulative_injection: 0.0,
            ke_scratch,
            most_recent_ke: 0.0,
        })
    }

    fn draw_new_kinetic_energy(&self, k_old: f64, dt: Real) -> f64 {
        let c = (-(dt as f64) / self.tau).exp();
        let nf = self.g_dof as f64;
        let k_target = (nf / 2.0) * self.kt_target;
        let one_minus_c = 1.0 - c;

        let seed_lo = self.seed as u32;
        let seed_hi = (self.seed >> 32) as u32;
        let ctr_lo = self.draw_counter as u32;
        let ctr_hi = (self.draw_counter >> 32) as u32;

        let r = philox_normal(seed_lo, seed_hi, ctr_lo, ctr_hi, 0, 0);
        let mut s = 0.0_f64;
        for sample_index in 1..self.g_dof {
            let xi = philox_normal(seed_lo, seed_hi, ctr_lo, ctr_hi, sample_index, 0);
            s += xi * xi;
        }

        let cross = if k_old > 0.0 {
            2.0 * r * (c * one_minus_c * k_old * k_target / nf).sqrt()
        } else {
            0.0
        };
        let k_new = c * k_old + (k_target / nf) * one_minus_c * (s + r * r) + cross;
        if k_new.is_finite() && k_new > 0.0 {
            k_new
        } else {
            k_old
        }
    }
}

impl Thermostat for CsvrThermostat {
    // rq-7a124d43
    fn apply_post(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: Real,
        timings: &mut Timings,
    ) -> Result<(), ThermostatError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }

        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k_old = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;

        self.draw_counter += 1;
        let k_new = self.draw_new_kinetic_energy(k_old, dt);
        self.cumulative_injection += k_new - k_old;
        self.most_recent_ke = k_new;

        if k_old > 0.0 && (k_new - k_old).abs() > 0.0 {
            let factor = (k_new / k_old).sqrt() as Real;
            timings.kernel_start(KernelStage::CSVR_RESCALE_VELOCITIES)?;
            rescale_velocities(buffers, factor)?;
            timings.kernel_stop(KernelStage::CSVR_RESCALE_VELOCITIES)?;
        }

        Ok(())
    }

    // rq-8ee58ec1
    fn log_column_names(&self) -> &'static [(&'static str, crate::units::Dimension)] {
        // csvr_conserved is a conserved Hamiltonian-like scalar in Hartrees.
        &[("csvr_conserved", crate::units::Dimension::Energy)]
    }

    // rq-2a5de2ab
    fn log_column_values(
        &self,
        kinetic_energy: f64,
        potential_energy: f64,
    ) -> Vec<f64> {
        vec![kinetic_energy + potential_energy - self.cumulative_injection]
    }
}

// rq-750b828f
#[derive(Debug, Clone)]
pub struct CsvrBuilder;

impl ThermostatBuilder for CsvrBuilder {
    fn kind_name(&self) -> &'static str {
        "csvr"
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
            .map_err(|_| ThermostatError::UnknownKind("csvr (malformed params)".into()))?;
        let state =
            CsvrThermostat::new(gpu, particle_count, n_constraints, p.temperature, p.tau, p.seed)?;
        Ok(Box::new(state))
    }

    fn box_clone(&self) -> Box<dyn ThermostatBuilder> {
        Box::new(self.clone())
    }
}
