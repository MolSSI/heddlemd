// rq-0d8c8688

use cudarc::driver::CudaSlice;

use serde::Deserialize;

use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, berendsen_compute_mu,
    compute_kinetic_energy_on_device, compute_total_virial_on_device,
};
use crate::io::config::ConfigError;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::{Barostat, BarostatBuilder, BarostatError};
use crate::precision::Real;

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
    ke_scratch: CudaSlice<Real>,
    virial_scratch: CudaSlice<Real>,
    /// Single-element device buffer holding the latest rescale factor
    /// µ. Written by `berendsen_compute_mu` and consumed by
    /// `rescale_positions_device_factor` on the same stream.
    mu_device: CudaSlice<Real>,
    /// Two-element device diagnostic buffer
    /// `[pressure_latest, volume_latest]`. Overwritten every step;
    /// drained into host fields by `flush_pending_injection`.
    diagnostics_device: CudaSlice<f64>,
}

impl BerendsenBarostat {
    fn new(
        gpu: &GpuContext,
        _particle_count: usize,
        pressure: f64,
        tau: f64,
        compressibility: f64,
    ) -> Result<Self, GpuError> {
        let ke_scratch = gpu.device.alloc_zeros::<Real>(1).map_err(GpuError::from)?;
        let virial_scratch = gpu.device.alloc_zeros::<Real>(1).map_err(GpuError::from)?;
        let mu_device = gpu.device.alloc_zeros::<Real>(1).map_err(GpuError::from)?;
        let diagnostics_device = gpu.device.alloc_zeros::<f64>(2).map_err(GpuError::from)?;
        Ok(BerendsenBarostat {
            pressure,
            tau,
            compressibility,
            most_recent_pressure: 0.0,
            most_recent_volume: 0.0,
            ke_scratch,
            virial_scratch,
            mu_device,
            diagnostics_device,
        })
    }

    /// Drains the device-side `[pressure, volume]` diagnostic into
    /// `most_recent_pressure` and `most_recent_volume`. Idempotent.
    /// The runner calls this once before each log row.
    pub fn flush_pending_injection(
        &mut self,
        device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(), GpuError> {
        let mut host = [0.0_f64; 2];
        device
            .dtoh_sync_copy_into(&self.diagnostics_device, &mut host)
            .map_err(GpuError::from)?;
        self.most_recent_pressure = host[0];
        self.most_recent_volume = host[1];
        Ok(())
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
        dt: Real,
        timings: &mut Timings,
    ) -> Result<(), BarostatError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }

        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        compute_kinetic_energy_on_device(buffers, &mut self.ke_scratch)?;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;

        timings.kernel_start(KernelStage::VIRIAL_SUM_REDUCE)?;
        compute_total_virial_on_device(buffers, &mut self.virial_scratch)?;
        timings.kernel_stop(KernelStage::VIRIAL_SUM_REDUCE)?;

        // Combined µ + lattice mutation + diagnostics write. No
        // per-step dtoh; lattice mutated in place via
        // `sim_box.lattice_device_mut()`.
        let dt_f64 = dt as f64;
        berendsen_compute_mu(
            buffers,
            &self.ke_scratch,
            &self.virial_scratch,
            sim_box.lattice_device_mut(),
            &mut self.mu_device,
            &mut self.diagnostics_device,
            self.pressure,
            self.tau,
            self.compressibility,
            dt_f64,
            MU_MIN * MU_MIN * MU_MIN,
        )?;
        // The per-particle position rescale `x ← μ · x` is dispatched
        // by the JIT-composed post-force per-particle kernel via this
        // slot's source fragment.

        Ok(())
    }

    fn flush_pending_injection(
        &mut self,
        device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(), BarostatError> {
        BerendsenBarostat::flush_pending_injection(self, device).map_err(BarostatError::from)
    }

    fn post_force_per_particle_fragment(
        &self,
    ) -> Option<crate::forces::PerParticleFragment> {
        Some(crate::forces::PerParticleFragment {
            label: "berendsen_barostat",
            helper_source: String::new(),
            entry_point_args: String::from(
                "    const Real *berendsen_baro_mu_device,\n",
            ),
            per_thread_body: String::from(
                "        Real berendsen_baro_mu = berendsen_baro_mu_device[0];\n\
                 \x20       Real4 pq = posq[i];\n\
                 \x20       pq.x *= berendsen_baro_mu;\n\
                 \x20       pq.y *= berendsen_baro_mu;\n\
                 \x20       pq.z *= berendsen_baro_mu;\n\
                 \x20       posq[i] = pq;",
            ),
        })
    }

    fn bind_post_force_per_particle_args(
        &self,
        _ctx: &crate::forces::PostForceBindContext<'_>,
        builder: &mut crate::forces::ForceLaunchBuilder,
    ) {
        builder.push_device_buffer(&self.mu_device);
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
#[derive(Debug, Clone)]
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

    fn box_clone(&self) -> Box<dyn BarostatBuilder> {
        Box::new(self.clone())
    }
}
