// rq-891232bf

use cudarc::driver::CudaSlice;

use serde::Deserialize;

use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, compute_kinetic_energy_on_device,
    csvr_sample_and_factor, rescale_velocities_device_factor,
};
use crate::io::config::ConfigError;
use crate::timings::{KernelStage, Timings};

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
    /// Conserved-quantity correction term. Accumulates `k_new - k_old`
    /// across every CSVR `apply_post`. Updated lazily: the GPU-side
    /// delta buffer accumulates per step, and `flush_pending_injection`
    /// downloads + zeroes it. The runner calls
    /// `flush_pending_injection` before reading `log_column_values`.
    pub cumulative_injection: f64,
    /// Single-element device buffer holding the most recent KE
    /// (`kinetic_energy_reduce` output). Read by
    /// `csvr_sample_and_factor` on the same stream — never copied to
    /// host on the per-step path.
    ke_scratch: CudaSlice<Real>,
    /// Single-element device buffer holding the rescale factor
    /// computed by `csvr_sample_and_factor`. The JIT-composed
    /// post-force per-particle kernel reads `factor_device[0]` in
    /// CSVR's fragment body. Public so tests that bypass the
    /// composed kernel can dispatch `rescale_velocities_device_factor`
    /// against it.
    pub factor_device: CudaSlice<Real>,
    /// Single-element device buffer accumulating `(k_new - k_old)`
    /// across CSVR steps since the last `flush_pending_injection`.
    /// `f64` to preserve precision across many steps before a host
    /// download. Reset to zero after every flush.
    cumulative_injection_delta: CudaSlice<f64>,
    /// Single-element device buffer holding the current Philox draw
    /// counter. `csvr_sample_and_factor` reads the value at entry and
    /// writes back `counter + 1` at exit. Living on the device makes
    /// the kernel safe to capture in a CUDA graph: every replay
    /// observes the post-increment counter from the previous replay
    /// and draws a distinct Philox sequence.
    draw_counter_device: CudaSlice<u64>,
    /// Per-block partial sums of `Σ xi_i²` for the multi-block CSVR
    /// sample (length `CSVR_PARTIAL_BLOCKS`). Written by
    /// `csvr_sample_partials` and reduced by `csvr_finish_from_partials`
    /// when `g_dof` exceeds the single-block threshold. rq-5f59fa80
    csvr_partials: CudaSlice<f64>,
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
        let factor_device = gpu.device.alloc_zeros::<Real>(1).map_err(GpuError::from)?;
        let cumulative_injection_delta =
            gpu.device.alloc_zeros::<f64>(1).map_err(GpuError::from)?;
        let draw_counter_device =
            gpu.device.alloc_zeros::<u64>(1).map_err(GpuError::from)?;
        let csvr_partials = gpu
            .device
            .alloc_zeros::<f64>(crate::gpu::CSVR_PARTIAL_BLOCKS as usize)
            .map_err(GpuError::from)?;
        Ok(CsvrThermostat {
            temperature,
            tau,
            seed,
            draw_counter: 0,
            g_dof,
            kt_target,
            cumulative_injection: 0.0,
            ke_scratch,
            factor_device,
            cumulative_injection_delta,
            draw_counter_device,
            csvr_partials,
        })
    }

    /// Downloads the device-side `(k_new - k_old)` accumulator into
    /// `cumulative_injection`, then zeroes the device buffer. Idempotent
    /// when called twice in a row (the device delta is zero after the
    /// first flush). The runner calls this once before each log-write
    /// so the conserved-quantity column reflects every step since the
    /// last flush; per-step callers (apply_post) never call it.
    pub fn flush_pending_injection(
        &mut self,
        device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(), GpuError> {
        let mut host_delta = [0.0_f64; 1];
        device
            .dtoh_sync_copy_into(&self.cumulative_injection_delta, &mut host_delta)
            .map_err(GpuError::from)?;
        self.cumulative_injection += host_delta[0];
        // Zero the device delta so the next flush sees only fresh injection.
        let zero = [0.0_f64; 1];
        device
            .htod_sync_copy_into(&zero, &mut self.cumulative_injection_delta)
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

        // 1. Kinetic-energy reduction into device buffer `ke_scratch`.
        //    No host download — the value never leaves the GPU.
        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        compute_kinetic_energy_on_device(buffers, &mut self.ke_scratch)?;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;

        // 2. CSVR chain math on device: reads `k_old` from
        //    `ke_scratch`, samples Philox in parallel, writes the
        //    rescale factor to `factor_device`, and accumulates
        //    `(k_new - k_old)` into `cumulative_injection_delta`. The
        //    kernel reads + increments `draw_counter_device` in place
        //    so every launch — including graph replays — uses a fresh
        //    Philox counter.
        let c = (-(dt as f64) / self.tau).exp();
        let one_minus_c = 1.0 - c;
        let nf = self.g_dof as f64;
        let k_target = (nf / 2.0) * self.kt_target;
        let k_target_over_nf = k_target / nf;
        csvr_sample_and_factor(
            buffers,
            &self.ke_scratch,
            &mut self.factor_device,
            &mut self.cumulative_injection_delta,
            &mut self.draw_counter_device,
            &mut self.csvr_partials,
            self.seed,
            self.g_dof,
            c,
            one_minus_c,
            k_target_over_nf,
        )?;

        // The per-particle rescale `v ← α · v` is dispatched by the
        // JIT-composed post-force per-particle kernel via this slot's
        // source fragment (see `rqm/integration/jit-composed-post-force.md`).
        // `apply_post` produces the device-resident `factor_device`
        // scalar; the composed kernel reads it.

        Ok(())
    }

    // rq-86dea9a1 — CSVR's per-particle rescale fragment for the
    // JIT-composed post-force kernel. The chain math
    // (`csvr_sample_and_factor`) still runs as part of `apply_post`
    // and produces `factor_device`; the composed kernel reads that
    // device-resident scalar in this fragment's per-thread body.
    fn post_force_per_particle_fragment(
        &self,
    ) -> Option<crate::forces::PerParticleFragment> {
        Some(crate::forces::PerParticleFragment {
            label: "csvr",
            helper_source: String::new(),
            entry_point_args: String::from(
                "    const Real *csvr_factor_device,\n",
            ),
            per_thread_body: String::from(
                "        Real csvr_factor = csvr_factor_device[0];\n\
                 \x20       velocities_x[i] *= csvr_factor;\n\
                 \x20       velocities_y[i] *= csvr_factor;\n\
                 \x20       velocities_z[i] *= csvr_factor;",
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

    fn flush_pending_injection(
        &mut self,
        device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> Result<(), ThermostatError> {
        CsvrThermostat::flush_pending_injection(self, device)
            .map_err(ThermostatError::from)
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
