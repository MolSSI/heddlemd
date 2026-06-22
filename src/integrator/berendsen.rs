// rq-25f24b26

use cudarc::driver::CudaSlice;

use serde::Deserialize;

use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, berendsen_compute_factor,
    compute_kinetic_energy_on_device,
};
use crate::io::config::ConfigError;
use crate::timings::{KernelStage, Timings};

use super::{Thermostat, ThermostatBuilder, ThermostatError};
use crate::precision::Real;

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
    ke_scratch: CudaSlice<Real>,
    /// Single-element device buffer holding the per-step rescale
    /// factor λ written by `berendsen_compute_factor`. The
    /// JIT-composed post-force per-particle kernel reads it.
    /// Public so tests that bypass the composed kernel can dispatch
    /// the standalone `rescale_velocities_device_factor` against it.
    pub factor_device: CudaSlice<Real>,
    /// Single-element device buffer accumulating
    /// `K_old · (λ² − 1)` per step. `flush_pending_injection`
    /// drains and zeroes it before each log row.
    cumulative_injection_delta: CudaSlice<f64>,
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
            most_recent_ke: 0.0,
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
        let k_target = (nf / 2.0) * self.kt_target;
        let dt_over_tau = (dt as f64) / self.tau;
        berendsen_compute_factor(
            buffers,
            &self.ke_scratch,
            &mut self.factor_device,
            &mut self.cumulative_injection_delta,
            k_target,
            dt_over_tau,
        )?;
        // The composed post-force per-particle kernel applies the
        // rescale via this slot's source fragment.
        Ok(())
    }

    fn flush_pending_injection(
        &mut self,
        device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(), ThermostatError> {
        BerendsenThermostat::flush_pending_injection(self, device)
            .map_err(ThermostatError::from)
    }

    fn post_force_per_particle_fragment(
        &self,
    ) -> Option<crate::forces::PerParticleFragment> {
        Some(crate::forces::PerParticleFragment {
            label: "berendsen",
            helper_source: String::new(),
            entry_point_args: String::from(
                "    const Real *berendsen_factor_device,\n",
            ),
            per_thread_body: String::from(
                "        Real berendsen_factor = berendsen_factor_device[0];\n\
                 \x20       velocities_x[i] *= berendsen_factor;\n\
                 \x20       velocities_y[i] *= berendsen_factor;\n\
                 \x20       velocities_z[i] *= berendsen_factor;",
            ),
        })
    }

    fn bind_post_force_per_particle_args(
        &self,
        _ctx: &crate::forces::PostForceBindContext<'_>,
        builder: &mut crate::forces::ForceLaunchBuilder,
    ) {
        builder.push_device_buffer(&self.factor_device);
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

    fn box_clone(&self) -> Box<dyn ThermostatBuilder> {
        Box::new(self.clone())
    }
}
