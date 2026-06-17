// rq-5e059f6b

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice};
use cudarc::nvrtc::Ptx;
use serde::Deserialize;

use crate::gpu::device::get_func;
use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, andersen_resample, compute_kinetic_energy,
};
use crate::kernels;
use crate::io::config::ConfigError;
use crate::timings::{KernelStage, Timings};

use super::{Thermostat, ThermostatBuilder, ThermostatError};
use crate::precision::Real;

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
    ke_scratch: CudaSlice<Real>,
    most_recent_ke: f64,
    /// Device-resident Philox counter. The kernel reads from it; the
    /// launcher follows with `increment_u64_device` so every launch
    /// advances the counter by one. Captured as graph nodes when the
    /// slot runs in graph mode.
    draw_counter_device: CudaSlice<u64>,
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
        let ke_scratch = gpu.device.alloc_zeros::<Real>(1).map_err(GpuError::from)?;
        let draw_counter_device =
            gpu.device.alloc_zeros::<u64>(1).map_err(GpuError::from)?;
        Ok(AndersenThermostat {
            temperature,
            collision_rate,
            seed,
            draw_counter: 0,
            kt,
            cumulative_injection: 0.0,
            ke_scratch,
            most_recent_ke: 0.0,
            draw_counter_device,
        })
    }
}

impl Thermostat for AndersenThermostat {
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

        let p_collision = ((self.collision_rate as f64) * (dt as f64))
            .clamp(0.0, 1.0) as Real;
        let kt = self.kt as Real;
        timings.kernel_start(KernelStage::ANDERSEN_RESAMPLE)?;
        andersen_resample(
            buffers,
            &mut self.draw_counter_device,
            self.seed,
            p_collision,
            kt,
        )?;
        timings.kernel_stop(KernelStage::ANDERSEN_RESAMPLE)?;
        self.draw_counter += 1;

        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k_new = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;

        self.cumulative_injection += k_new - k_old;
        self.most_recent_ke = k_new;
        Ok(())
    }

    // rq-1163481e
    fn log_column_names(&self) -> &'static [(&'static str, crate::units::Dimension)] {
        // andersen_conserved is a conserved Hamiltonian-like scalar in Hartrees.
        &[("andersen_conserved", crate::units::Dimension::Energy)]
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
#[derive(Debug, Clone)]
pub struct AndersenBuilder;

impl ThermostatBuilder for AndersenBuilder {
    fn kind_name(&self) -> &'static str {
        "andersen"
    }

    fn graph_compatible(&self, _params: &toml::Value) -> bool {
        // Andersen's `apply_post` runs `compute_kinetic_energy` before
        // and after the resample kernel; both calls perform a host
        // dtoh of the kinetic-energy scalar to update the
        // `cumulative_injection` log column. Until that bookkeeping
        // moves to device, Andersen runs on the per-step launch path.
        false
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
        _n_constraints: usize,
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

    fn box_clone(&self) -> Box<dyn ThermostatBuilder> {
        Box::new(self.clone())
    }
}

// rq-2093594f rq-5e059f6b
#[derive(Debug, Clone)]
pub struct AndersenKernels {
    pub andersen_resample: CudaFunction,
}

impl AndersenKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::ANDERSEN),
            "andersen",
            &["andersen_resample"],
        )?;
        Ok(AndersenKernels {
            andersen_resample: get_func(device, "andersen", "andersen_resample")?,
        })
    }
}
