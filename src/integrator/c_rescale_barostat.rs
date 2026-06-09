// rq-11f5dfd1

use cudarc::driver::CudaSlice;

use serde::Deserialize;

use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, compute_kinetic_energy, compute_total_virial,
    rescale_positions,
};
use crate::io::config::ConfigError;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::philox::philox_normal;
use super::{Barostat, BarostatBuilder, BarostatError};

// rq-1f87880c
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CRescaleBarostatParams {
    pub pressure: f64,
    pub temperature: f64,
    pub tau: f64,
    pub compressibility: f64,
    pub seed: u64,
}

fn deserialize_params(params: &toml::Value) -> Result<CRescaleBarostatParams, ConfigError> {
    params
        .clone()
        .try_into::<CRescaleBarostatParams>()
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

// Host-side safety floor on μ. Sensible parameters never approach it;
// the floor only triggers under pathological combinations.
const MU_MIN: f64 = 1.0e-6;

// rq-11f5dfd1
#[derive(Debug)]
pub struct CRescaleBarostat {
    pub pressure: f64,
    pub temperature: f64,
    pub tau: f64,
    pub compressibility: f64,
    pub seed: u64,
    pub draw_counter: u64,
    pub cumulative_barostat_injection: f64,
    pub most_recent_pressure: f64,
    pub most_recent_volume: f64,
    ke_scratch: CudaSlice<f32>,
    virial_scratch: CudaSlice<f32>,
}

impl CRescaleBarostat {
    #[allow(clippy::too_many_arguments)]
    fn new(
        gpu: &GpuContext,
        _particle_count: usize,
        pressure: f64,
        temperature: f64,
        tau: f64,
        compressibility: f64,
        seed: u64,
    ) -> Result<Self, GpuError> {
        let ke_scratch = gpu.device.alloc_zeros::<f32>(1).map_err(GpuError::from)?;
        let virial_scratch = gpu.device.alloc_zeros::<f32>(1).map_err(GpuError::from)?;
        Ok(CRescaleBarostat {
            pressure,
            temperature,
            tau,
            compressibility,
            seed,
            draw_counter: 0,
            cumulative_barostat_injection: 0.0,
            most_recent_pressure: 0.0,
            most_recent_volume: 0.0,
            ke_scratch,
            virial_scratch,
        })
    }
}

impl Barostat for CRescaleBarostat {
    // rq-1179e42f
    fn apply(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), BarostatError> {
        if buffers.particle_count() == 0 {
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

        self.draw_counter += 1;
        let seed_lo = self.seed as u32;
        let seed_hi = (self.seed >> 32) as u32;
        let ctr_lo = self.draw_counter as u32;
        let ctr_hi = (self.draw_counter >> 32) as u32;
        let r = philox_normal(seed_lo, seed_hi, ctr_lo, ctr_hi, 0, 0);

        // k_B = 1 in atomic units; temperature is already k_B · T.
        let kt = self.temperature;
        let dt_f64 = dt as f64;
        let deterministic = -self.compressibility * (dt_f64 / self.tau)
            * (self.pressure - pressure);
        let noise_amplitude =
            (2.0 * self.compressibility * kt * dt_f64 / (self.tau * v_pre)).sqrt();
        let mu_cubed = 1.0 + deterministic + noise_amplitude * r;
        let mu_cubed_clamped = mu_cubed.max(MU_MIN * MU_MIN * MU_MIN);
        let mu = mu_cubed_clamped.cbrt();
        let mu_f32 = mu as f32;

        timings.kernel_start(KernelStage::C_RESCALE_BAROSTAT_RESCALE_POSITIONS)?;
        rescale_positions(buffers, mu_f32)?;
        timings.kernel_stop(KernelStage::C_RESCALE_BAROSTAT_RESCALE_POSITIONS)?;

        sim_box
            .rescale_isotropic(mu_f32)
            .map_err(|_| BarostatError::Gpu(GpuError(
                cudarc::driver::DriverError(
                    cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
                ),
            )))?;

        let v_post = sim_box.volume() as f64;
        self.cumulative_barostat_injection += self.pressure * (v_post - v_pre);
        self.most_recent_pressure = pressure;
        self.most_recent_volume = v_post;
        Ok(())
    }

    // rq-11f5dfd1
    fn log_column_names(&self) -> &'static [(&'static str, crate::units::Dimension)] {
        use crate::units::Dimension;
        &[
            ("pressure", Dimension::Pressure),
            // box_volume has no Dimension::Volume variant; report in Bohr^3 and
            // pass through unchanged (dimensionless w.r.t. the unit converter).
            ("box_volume", Dimension::Dimensionless),
            ("c_rescale_conserved", Dimension::Energy),
        ]
    }

    // rq-11f5dfd1
    fn log_column_values(
        &self,
        kinetic_energy: f64,
        potential_energy: f64,
    ) -> Vec<f64> {
        let conserved = kinetic_energy
            + potential_energy
            + self.pressure * self.most_recent_volume
            - self.cumulative_barostat_injection;
        vec![
            self.most_recent_pressure,
            self.most_recent_volume,
            conserved,
        ]
    }
}

#[derive(Debug)]
pub struct CRescaleBarostatBuilder;

impl BarostatBuilder for CRescaleBarostatBuilder {
    fn kind_name(&self) -> &'static str {
        "c-rescale"
    }

    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> {
        let p = deserialize_params(params)?;
        require_finite("barostat.pressure", p.pressure)?;
        require_finite_positive("barostat.temperature", p.temperature)?;
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
            BarostatError::UnknownKind("c-rescale (malformed params)".into())
        })?;
        let state = CRescaleBarostat::new(
            gpu,
            particle_count,
            p.pressure,
            p.temperature,
            p.tau,
            p.compressibility,
            p.seed,
        )?;
        Ok(Box::new(state))
    }
}
