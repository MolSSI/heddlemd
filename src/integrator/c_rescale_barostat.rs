// rq-11f5dfd1

use cudarc::driver::CudaSlice;

use serde::Deserialize;

use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, c_rescale_compute_mu,
    compute_kinetic_energy_on_device, compute_total_virial_on_device, rescale_positions,
};
use crate::io::config::ConfigError;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::{Barostat, BarostatBuilder, BarostatError};
use crate::precision::Real;

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

// rq-11f5dfd1 rq-e38993a3
#[derive(Debug)]
pub struct CRescaleBarostat {
    pub pressure: f64,
    pub temperature: f64,
    pub tau: f64,
    pub compressibility: f64,
    pub seed: u64,
    pub draw_counter: u64,
    /// Conserved-quantity correction term. Accumulates
    /// `P_target · (v_post - v_pre)` across every `apply`. Updated
    /// lazily: the GPU-side delta buffer accumulates per step, and
    /// `flush_pending_injection` downloads + zeroes it. The runner
    /// calls `flush_pending_injection` before reading
    /// `log_column_values`.
    pub cumulative_barostat_injection: f64,
    pub most_recent_pressure: f64,
    pub most_recent_volume: f64,
    ke_scratch: CudaSlice<Real>,
    virial_scratch: CudaSlice<Real>,
    /// Two-element device buffer: `[mu, pressure]`. Written by
    /// `c_rescale_compute_mu`, downloaded by the host to update the
    /// simulation box and emit the diagnostic pressure column.
    mu_pressure_device: CudaSlice<Real>,
    /// Single-element device accumulator for
    /// `P_target · (v_post - v_pre)` across steps since the last flush.
    /// `f64` to preserve precision across many steps before a host
    /// download. Reset to zero after every flush.
    cumulative_injection_delta: CudaSlice<f64>,
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
        let ke_scratch = gpu.device.alloc_zeros::<Real>(1).map_err(GpuError::from)?;
        let virial_scratch = gpu.device.alloc_zeros::<Real>(1).map_err(GpuError::from)?;
        let mu_pressure_device =
            gpu.device.alloc_zeros::<Real>(2).map_err(GpuError::from)?;
        let cumulative_injection_delta =
            gpu.device.alloc_zeros::<f64>(1).map_err(GpuError::from)?;
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
            mu_pressure_device,
            cumulative_injection_delta,
        })
    }

    /// Drains the device-side `P_target · (v_post - v_pre)` accumulator
    /// into `cumulative_barostat_injection`, then zeroes the device
    /// buffer. Idempotent across consecutive calls. The runner calls
    /// this once before each log row.
    pub fn flush_pending_injection(
        &mut self,
        device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(), GpuError> {
        let mut host_delta = [0.0_f64; 1];
        device
            .dtoh_sync_copy_into(&self.cumulative_injection_delta, &mut host_delta)
            .map_err(GpuError::from)?;
        self.cumulative_barostat_injection += host_delta[0];
        let zero = [0.0_f64; 1];
        device
            .htod_sync_copy_into(&zero, &mut self.cumulative_injection_delta)
            .map_err(GpuError::from)?;
        Ok(())
    }
}

impl Barostat for CRescaleBarostat {
    // rq-1179e42f rq-2b405d23
    fn apply(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        dt: Real,
        timings: &mut Timings,
    ) -> Result<(), BarostatError> {
        if buffers.particle_count() == 0 {
            self.most_recent_pressure = 0.0;
            self.most_recent_volume = sim_box.volume() as f64;
            return Ok(());
        }

        // 1. Reduce KE and total virial into device buffers — no
        //    blocking dtoh. The pipeline stays GPU-side from here
        //    through the mu computation.
        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        compute_kinetic_energy_on_device(buffers, &mut self.ke_scratch)?;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;

        timings.kernel_start(KernelStage::VIRIAL_SUM_REDUCE)?;
        compute_total_virial_on_device(buffers, &mut self.virial_scratch)?;
        timings.kernel_stop(KernelStage::VIRIAL_SUM_REDUCE)?;

        // 2. Compute mu and pressure on device. The conserved-quantity
        //    injection delta is accumulated into
        //    `cumulative_injection_delta` for later flush at log time.
        self.draw_counter += 1;
        let v_pre = sim_box.volume() as f64;
        let kt = self.temperature;
        let dt_f64 = dt as f64;
        c_rescale_compute_mu(
            buffers,
            &self.ke_scratch,
            &self.virial_scratch,
            &mut self.mu_pressure_device,
            &mut self.cumulative_injection_delta,
            self.seed,
            self.draw_counter,
            v_pre,
            self.pressure,
            self.tau,
            self.compressibility,
            kt,
            dt_f64,
            MU_MIN * MU_MIN * MU_MIN,
        )?;

        // 3. Download [mu, pressure]. This is the *only* per-step
        //    barostat sync; mu is needed host-side to update the
        //    simulation box, and the diagnostic pressure feeds the
        //    log column.
        let mut mu_pressure_host = [0.0 as Real; 2];
        buffers
            .device
            .dtoh_sync_copy_into(&self.mu_pressure_device, &mut mu_pressure_host)
            .map_err(GpuError::from)?;
        let mu = mu_pressure_host[0];
        let pressure = mu_pressure_host[1] as f64;

        // 4. Apply the rescale. Positions go through the existing
        //    kernel (scalar mu arg), box is updated host-side.
        timings.kernel_start(KernelStage::C_RESCALE_BAROSTAT_RESCALE_POSITIONS)?;
        rescale_positions(buffers, mu)?;
        timings.kernel_stop(KernelStage::C_RESCALE_BAROSTAT_RESCALE_POSITIONS)?;

        sim_box
            .rescale_isotropic(mu)
            .map_err(|_| BarostatError::Gpu(GpuError(
                cudarc::driver::DriverError(
                    cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
                ),
            )))?;

        let v_post = sim_box.volume() as f64;
        self.most_recent_pressure = pressure;
        self.most_recent_volume = v_post;
        Ok(())
    }

    fn flush_pending_injection(
        &mut self,
        device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(), BarostatError> {
        CRescaleBarostat::flush_pending_injection(self, device)
            .map_err(BarostatError::from)
    }

    // rq-11f5dfd1 rq-b8e6dfb3
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

    // rq-11f5dfd1 rq-9a88810a
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

// rq-d521381a
#[derive(Debug, Clone)]
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

    fn box_clone(&self) -> Box<dyn BarostatBuilder> {
        Box::new(self.clone())
    }
}
