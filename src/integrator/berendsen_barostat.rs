// rq-0d8c8688

use cudarc::driver::CudaSlice;

use serde::Deserialize;

use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, compute_kinetic_energy, compute_total_virial,
    rescale_positions,
};
use crate::io::config::ConfigError;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::{Barostat, BarostatBuilder, BarostatError};

// rq-1f87880c
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BerendsenBarostatParams {
    pub pressure: f64,
    pub tau: f64,
    pub compressibility: f64,
}

fn deserialize_params(params: &toml::Value) -> Result<BerendsenBarostatParams, ConfigError> {
    params
        .clone()
        .try_into::<BerendsenBarostatParams>()
        .map_err(|e| crate::io::config::translate_params_error("barostat", e))
}

fn require_finite(field: &str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() {
        return Err(ConfigError::InvalidValue {
            field: field.to_string(),
            reason: format!("value must be finite, got {value}"),
        });
    }
    Ok(())
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

// rq-0d8c8688 rq-5c758681
#[derive(Debug)]
pub struct BerendsenBarostat {
    pub pressure: f64,
    pub tau: f64,
    pub compressibility: f64,
    pub most_recent_pressure: f64,
    pub most_recent_volume: f64,
    ke_scratch: CudaSlice<f32>,
    virial_scratch: CudaSlice<f32>,
}

impl BerendsenBarostat {
    fn new(
        gpu: &GpuContext,
        _particle_count: usize,
        pressure: f64,
        tau: f64,
        compressibility: f64,
    ) -> Result<Self, GpuError> {
        let ke_scratch = gpu.device.alloc_zeros::<f32>(1).map_err(GpuError::from)?;
        let virial_scratch = gpu.device.alloc_zeros::<f32>(1).map_err(GpuError::from)?;
        Ok(BerendsenBarostat {
            pressure,
            tau,
            compressibility,
            most_recent_pressure: 0.0,
            most_recent_volume: 0.0,
            ke_scratch,
            virial_scratch,
        })
    }
}

// Host-side safety floor on μ. Sensible parameters never approach it;
// the floor only triggers under pathological combinations
// (β · dt/τ · (P_target − P) > 1).
const MU_MIN: f64 = 1.0e-6;

impl Barostat for BerendsenBarostat {
    // rq-1179e42f rq-29dda250
    fn apply(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), BarostatError> {
        if buffers.particle_count() == 0 {
            // Pre-populate the diagnostic fields so log_column_values
            // still returns finite numbers for an empty run.
            self.most_recent_pressure = 0.0;
            self.most_recent_volume = sim_box.volume() as f64;
            return Ok(());
        }

        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;

        timings.kernel_start(KernelStage::VIRIAL_SUM_REDUCE)?;
        let w = compute_total_virial(buffers, &mut self.virial_scratch)? as f64;
        timings.kernel_stop(KernelStage::VIRIAL_SUM_REDUCE)?;

        let v_pre = sim_box.volume() as f64;
        let pressure = (2.0 * k + w) / (3.0 * v_pre);

        let mu_cubed = 1.0
            - self.compressibility * ((dt as f64) / self.tau) * (self.pressure - pressure);
        let mu_cubed_clamped = mu_cubed.max(MU_MIN * MU_MIN * MU_MIN);
        let mu = mu_cubed_clamped.cbrt();
        let mu_f32 = mu as f32;

        timings.kernel_start(KernelStage::BERENDSEN_BAROSTAT_RESCALE_POSITIONS)?;
        rescale_positions(buffers, mu_f32)?;
        timings.kernel_stop(KernelStage::BERENDSEN_BAROSTAT_RESCALE_POSITIONS)?;

        // Bumps generation; downstream consumers refresh on next force step.
        sim_box
            .rescale_isotropic(mu_f32)
            .map_err(|_| BarostatError::Gpu(GpuError(
                cudarc::driver::DriverError(
                    cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
                ),
            )))?;

        self.most_recent_pressure = pressure;
        self.most_recent_volume = sim_box.volume() as f64;
        Ok(())
    }

    // rq-62b44dc9 rq-b6728f3c
    fn log_column_names(&self) -> &'static [(&'static str, crate::units::Dimension)] {
        use crate::units::Dimension;
        &[
            ("pressure", Dimension::Pressure),
            ("box_volume", Dimension::Dimensionless),
        ]
    }

    // rq-62b44dc9 rq-82baba1a
    fn log_column_values(
        &self,
        _kinetic_energy: f64,
        _potential_energy: f64,
    ) -> Vec<f64> {
        vec![self.most_recent_pressure, self.most_recent_volume]
    }
}

// rq-4ef89c50
#[derive(Debug)]
pub struct BerendsenBarostatBuilder;

impl BarostatBuilder for BerendsenBarostatBuilder {
    fn kind_name(&self) -> &'static str {
        "berendsen"
    }

    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> {
        let p = deserialize_params(params)?;
        require_finite("barostat.pressure", p.pressure)?;
        require_finite_positive("barostat.tau", p.tau)?;
        require_finite_positive("barostat.compressibility", p.compressibility)?;
        Ok(())
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        _n_constraints: usize,
        params: &toml::Value,
    ) -> Result<Box<dyn Barostat>, BarostatError> {
        let p = deserialize_params(params).map_err(|_| {
            BarostatError::UnknownKind("berendsen (malformed params)".into())
        })?;
        let state = BerendsenBarostat::new(
            gpu,
            particle_count,
            p.pressure,
            p.tau,
            p.compressibility,
        )?;
        Ok(Box::new(state))
    }
}
