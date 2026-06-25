// rq-11f5dfd1

use cudarc::driver::CudaSlice;

use serde::Deserialize;

use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, c_rescale_compute_mu,
    compute_kinetic_energy_on_device, compute_total_virial_on_device,
    rescale_positions_device_factor,
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
    /// Single-element device buffer holding the latest rescale factor
    /// µ. Written by `c_rescale_compute_mu`; the JIT-composed
    /// post-force per-particle kernel reads `mu_device[0]` in
    /// c-rescale's fragment body. Public so tests that bypass the
    /// composed-kernel path can dispatch the standalone
    /// `rescale_positions_device_factor` against it.
    pub mu_device: CudaSlice<Real>,
    /// Three-element device buffer laid out as
    /// `[pressure_latest, volume_latest, injection_delta]`. Slots 0 and
    /// 1 are overwritten every step; slot 2 accumulates
    /// `P_target · (v_post - v_pre)` and is drained / zeroed by
    /// `flush_pending_injection`. `f64` for precision across many
    /// steps before a host download.
    diagnostics_device: CudaSlice<f64>,
    /// Single-element device buffer holding the Philox draw counter.
    /// `c_rescale_compute_mu` reads it at entry and writes back
    /// `counter + 1` at exit; safe to capture in a CUDA graph because
    /// the kernel runs on a single thread.
    draw_counter_device: CudaSlice<u64>,
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
        let mu_device = gpu.device.alloc_zeros::<Real>(1).map_err(GpuError::from)?;
        let diagnostics_device = gpu.device.alloc_zeros::<f64>(3).map_err(GpuError::from)?;
        let draw_counter_device = gpu.device.alloc_zeros::<u64>(1).map_err(GpuError::from)?;
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
            mu_device,
            diagnostics_device,
            draw_counter_device,
        })
    }

    /// Drains the device-side diagnostic buffer
    /// `[pressure_latest, volume_latest, injection_delta]` into the
    /// host fields `most_recent_pressure`, `most_recent_volume`, and
    /// `cumulative_barostat_injection` (the delta is added, then the
    /// device slot is zeroed). Idempotent across consecutive calls:
    /// a second call immediately after the first finds delta = 0 and
    /// reads the same pressure / volume values.
    pub fn flush_pending_injection(
        &mut self,
        device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(), GpuError> {
        let mut host = [0.0_f64; 3];
        device
            .dtoh_sync_copy_into(&self.diagnostics_device, &mut host)
            .map_err(GpuError::from)?;
        self.most_recent_pressure = host[0];
        self.most_recent_volume = host[1];
        self.cumulative_barostat_injection += host[2];
        // Zero only the injection delta; pressure / volume slots can
        // remain — they'll be overwritten by the next apply.
        let zeroed = [host[0], host[1], 0.0_f64];
        device
            .htod_sync_copy_into(&zeroed, &mut self.diagnostics_device)
            .map_err(GpuError::from)?;
        // Refresh the host-side draw_counter cache for diagnostics.
        let mut host_counter = [0_u64; 1];
        device
            .dtoh_sync_copy_into(&self.draw_counter_device, &mut host_counter)
            .map_err(GpuError::from)?;
        self.draw_counter = host_counter[0];
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
            return Ok(());
        }

        // 1. KE + virial reduce into device scratch buffers (no dtoh).
        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        compute_kinetic_energy_on_device(buffers, &mut self.ke_scratch)?;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;

        timings.kernel_start(KernelStage::VIRIAL_SUM_REDUCE)?;
        compute_total_virial_on_device(buffers, &mut self.virial_scratch)?;
        timings.kernel_stop(KernelStage::VIRIAL_SUM_REDUCE)?;

        // 2. Combined compute_mu + lattice mutation + diagnostics
        //    write. No per-step dtoh; the µ, diagnostics, and Philox
        //    counter all live on device; the lattice is mutated in
        //    place through `sim_box.lattice_device_mut()` (which
        //    bumps the box generation counter). The kernel reads and
        //    increments `draw_counter_device` in place, so safely
        //    captured by a CUDA graph.
        let kt = self.temperature;
        let dt_f64 = dt as f64;
        timings.kernel_start(KernelStage::C_RESCALE_COMPUTE_MU)?;
        c_rescale_compute_mu(
            buffers,
            &self.ke_scratch,
            &self.virial_scratch,
            sim_box.lattice_device_mut(),
            &mut self.mu_device,
            &mut self.diagnostics_device,
            &mut self.draw_counter_device,
            self.seed,
            self.pressure,
            self.tau,
            self.compressibility,
            kt,
            dt_f64,
            MU_MIN * MU_MIN * MU_MIN,
        )?;
        timings.kernel_stop(KernelStage::C_RESCALE_COMPUTE_MU)?;
        self.draw_counter += 1;

        // The per-particle position rescale `x ← μ · x` is dispatched
        // by the JIT-composed post-force per-particle kernel via this
        // slot's source fragment. `apply` produces the device-resident
        // `mu_device` scalar; the composed kernel reads it.

        Ok(())
    }

    // rq-56044cc3 — c-rescale's per-particle position rescale fragment
    // for the JIT-composed post-force kernel. `apply` still computes
    // µ via `c_rescale_compute_mu`; the composed kernel reads
    // `mu_device` and applies the rescale to positions.
    fn post_force_per_particle_fragment(
        &self,
    ) -> Option<crate::forces::PerParticleFragment> {
        Some(crate::forces::PerParticleFragment {
            label: "c_rescale_barostat",
            helper_source: String::new(),
            entry_point_args: String::from(
                "    const Real *c_rescale_mu_device,\n",
            ),
            per_thread_body: String::from(
                "        Real c_rescale_mu = c_rescale_mu_device[0];\n\
                 \x20       Real4 pq = posq[i];\n\
                 \x20       pq.x *= c_rescale_mu;\n\
                 \x20       pq.y *= c_rescale_mu;\n\
                 \x20       pq.z *= c_rescale_mu;\n\
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
