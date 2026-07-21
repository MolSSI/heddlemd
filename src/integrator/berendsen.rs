// rq-25f24b26

use cudarc::driver::CudaSlice;

use serde::Deserialize;

use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, berendsen_compute_factor,
    compute_kinetic_energy_on_device, rescale_velocities_device_factor,
};
use crate::io::config::ConfigError;
use crate::timings::{KernelStage, Timings};

use super::{Thermostat, ThermostatBuilder, ThermostatError};
use crate::precision::Real;

// rq-1f87880c
#[derive(Debug, Clone, Deserialize, serde::Serialize, crate::units::Convert)]
#[serde(deny_unknown_fields)]
pub struct BerendsenParams {
    pub temperature: crate::units::Temperature,
    pub tau: crate::units::Time,
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
    ke_scratch: CudaSlice<Real>,
    /// Single-element device buffer holding the per-step rescale
    /// factor λ written by `berendsen_compute_factor` and consumed by
    /// the standalone `rescale_velocities_device_factor` launch in
    /// `apply_post`. Public so tests can dispatch that rescale against
    /// it directly.
    pub factor_device: CudaSlice<Real>,
    /// Single-element device buffer accumulating
    /// `K_old · (λ² − 1)` per step. `flush_pending_injection`
    /// drains and zeroes it before each log row.
    cumulative_injection_delta: CudaSlice<f64>,
}

impl BerendsenThermostat {
    fn new(
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
        temperature: f64,
        tau: f64,
    ) -> Result<Self, GpuError> {
        // Thermal DOF: `3N − n_constraints − 3` (COM momentum removed
        // at init and preserved by the uniform velocity rescale).
        // Floored to 1 — the rescale factor divides by `N_f` via
        // `T_inst = 2K / N_f`. The floor only engages for systems with
        // no thermal DOF, which should not be thermostatted anyway
        // (see `rqm/integration/berendsen.md`, degenerate cases).
        let g_dof =
            ((3 * particle_count) as i64 - n_constraints as i64 - 3).max(1) as u32;
        // k_B = 1 in atomic units; temperature is already k_B · T.
        let kt_target = temperature;
        let ke_scratch = gpu.device.alloc_zeros::<Real>(1).map_err(GpuError::from)?;
        let factor_device = gpu.device.alloc_zeros::<Real>(1).map_err(GpuError::from)?;
        let cumulative_injection_delta =
            gpu.device.alloc_zeros::<f64>(1).map_err(GpuError::from)?;
        Ok(BerendsenThermostat {
            temperature,
            tau,
            g_dof,
            kt_target,
            cumulative_injection: 0.0,
            ke_scratch,
            factor_device,
            cumulative_injection_delta,
        })
    }

    pub fn flush_pending_injection(
        &mut self,
        device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(), GpuError> {
        let mut host_delta = [0.0_f64; 1];
        device
            .dtoh_sync_copy_into(&self.cumulative_injection_delta, &mut host_delta)
            .map_err(GpuError::from)?;
        self.cumulative_injection += host_delta[0];
        let zero = [0.0_f64; 1];
        device
            .htod_sync_copy_into(&zero, &mut self.cumulative_injection_delta)
            .map_err(GpuError::from)?;
        Ok(())
    }
}

impl Thermostat for BerendsenThermostat {
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
        compute_kinetic_energy_on_device(buffers, &mut self.ke_scratch)?;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;

        let nf = self.g_dof as f64;
        // Equipartition: `K_target = (N_f / 2) · k_B·T` over the
        // constraint- and COM-removed DOF computed at construction.
        let k_target = (nf / 2.0) * self.kt_target;
        let dt_over_tau = (dt as f64) / self.tau;
        timings.kernel_start(KernelStage::BERENDSEN_COMPUTE_FACTOR)?;
        berendsen_compute_factor(
            buffers,
            &self.ke_scratch,
            &mut self.factor_device,
            &mut self.cumulative_injection_delta,
            k_target,
            dt_over_tau,
        )?;
        timings.kernel_stop(KernelStage::BERENDSEN_COMPUTE_FACTOR)?;

        // Per-particle rescale `v ← λ · v`, reading `factor_device` on
        // the device. The full-step kinetic-energy reduction above is a
        // fusion barrier, so Berendsen applies the rescale as its own
        // launch rather than a composed fragment.
        timings.kernel_start(KernelStage::BERENDSEN_RESCALE_VELOCITIES)?;
        rescale_velocities_device_factor(buffers, &self.factor_device)?;
        timings.kernel_stop(KernelStage::BERENDSEN_RESCALE_VELOCITIES)?;
        Ok(())
    }

    fn flush_pending_injection(
        &mut self,
        device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(), ThermostatError> {
        BerendsenThermostat::flush_pending_injection(self, device)
            .map_err(ThermostatError::from)
    }


    // rq-c908bbf1
    fn log_column_names(&self) -> &'static [(&'static str, crate::units::Dimension)] {
        // berendsen_conserved is a conserved Hamiltonian-like scalar in Hartrees.
        &[("berendsen_conserved", crate::units::Dimension::Energy)]
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
#[derive(Debug, Clone)]
pub struct BerendsenBuilder;

use crate::registry::KindedBuilder;

impl KindedBuilder for BerendsenBuilder {
    fn kind_name(&self) -> &'static str {
        "berendsen"
    }
    fn convert_params(
        &self,
        units: crate::units::UnitSystem,
        params: &mut toml::Value,
    ) -> Result<(), crate::io::config::ConfigError> {
        crate::registry::convert_params_in_place::<BerendsenParams>(units, params)
    }
}

impl ThermostatBuilder for BerendsenBuilder {
    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> {
        let p = deserialize_params(params)?;
        require_finite_positive("thermostat.temperature", p.temperature.0)?;
        require_finite_positive("thermostat.tau", p.tau.0)?;
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
            p.temperature.0,
            p.tau.0,
        )?;
        Ok(Box::new(state))
    }
}
