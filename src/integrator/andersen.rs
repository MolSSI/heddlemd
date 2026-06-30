// rq-5e059f6b

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use serde::Deserialize;

use crate::gpu::{GpuContext, GpuError, ParticleBuffers};
use crate::io::config::ConfigError;
use crate::timings::Timings;

use super::{Thermostat, ThermostatBuilder, ThermostatError};
use crate::precision::Real;

// rq-1f87880c
#[derive(Debug, Clone, Deserialize, serde::Serialize, crate::units::Convert)]
#[serde(deny_unknown_fields)]
pub struct AndersenParams {
    pub temperature: crate::units::Temperature,
    pub collision_rate: crate::units::InverseTime,
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
    /// Legacy field retained for diagnostic compatibility. Always
    /// zero; the per-step `(K_new − K_old)` accounting that the
    /// standalone path tracked is not reproduced inside the
    /// JIT-composed post-force per-particle kernel.
    pub cumulative_injection: f64,
    /// Per-step probability `p = clamp(collision_rate · dt, 0, 1)`,
    /// pushed onto the composed-kernel launch builder. Cached on
    /// `apply_post` from the current `dt`.
    cached_p_collision: Real,
    /// Device-resident Philox counter. The fragment reads from it
    /// (one lane in the first block increments it after the per-thread
    /// draws). Persists across runs / graph replays. Public so tests
    /// that bypass the composed-kernel path can dispatch the
    /// standalone `andersen_resample` against it.
    pub draw_counter_device: cudarc::driver::CudaSlice<u64>,
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
        let draw_counter_device =
            gpu.device.alloc_zeros::<u64>(1).map_err(GpuError::from)?;
        Ok(AndersenThermostat {
            temperature,
            collision_rate,
            seed,
            draw_counter: 0,
            kt,
            cumulative_injection: 0.0,
            cached_p_collision: 0.0,
            draw_counter_device,
        })
    }

    pub fn flush_pending_injection(
        &mut self,
        device: &Arc<CudaDevice>,
    ) -> Result<(), GpuError> {
        // Refresh the host-side draw_counter cache for diagnostics.
        let mut host_counter = [0_u64; 1];
        device
            .dtoh_sync_copy_into(&self.draw_counter_device, &mut host_counter)
            .map_err(GpuError::from)?;
        self.draw_counter = host_counter[0];
        Ok(())
    }
}

impl crate::integrator::PostForcePerParticle for AndersenThermostat {
    fn post_force_per_particle_fragment(
        &self,
    ) -> crate::forces::PerParticleFragment {
        crate::forces::PerParticleFragment {
            label: "andersen",
            helper_source: String::from(ANDERSEN_PHILOX_HELPER_SOURCE),
            entry_point_args: String::from(
                "    const unsigned int *andersen_particle_ids,\n\
                 \x20   unsigned long long *andersen_draw_counter_device,\n\
                 \x20   unsigned int andersen_seed_lo,\n\
                 \x20   unsigned int andersen_seed_hi,\n\
                 \x20   Real andersen_p_collision,\n\
                 \x20   Real andersen_kt,\n",
            ),
            per_thread_body: String::from(ANDERSEN_PER_THREAD_BODY),
        }
    }

    fn bind_post_force_per_particle_args(
        &self,
        ctx: &crate::forces::PostForceBindContext<'_>,
        builder: &mut crate::forces::ForceLaunchBuilder,
    ) {
        builder.push_device_buffer(&ctx.buffers.particle_ids);
        builder.push_device_buffer(&self.draw_counter_device);
        let seed_lo = self.seed as u32;
        let seed_hi = (self.seed >> 32) as u32;
        builder.push_scalar::<u32>(seed_lo);
        builder.push_scalar::<u32>(seed_hi);
        builder.push_scalar::<Real>(self.cached_p_collision);
        builder.push_scalar::<Real>(self.kt as Real);
    }}

impl Thermostat for AndersenThermostat {
    // rq-7a124d43 — Andersen does no scalar-prep work; the per-particle
    // Bernoulli + Maxwell-Boltzmann resample is performed entirely
    // inside the JIT-composed post-force per-particle kernel via this
    // slot's source fragment. `apply_post` only caches the current
    // step's `p_collision` for the bind method to pass to the kernel.
    fn apply_post(
        &mut self,
        _buffers: &mut ParticleBuffers,
        dt: Real,
        _timings: &mut Timings,
    ) -> Result<(), ThermostatError> {
        self.cached_p_collision = ((self.collision_rate as f64) * (dt as f64))
            .clamp(0.0, 1.0) as Real;
        Ok(())
    }

    fn flush_pending_injection(
        &mut self,
        device: &Arc<CudaDevice>,
    ) -> Result<(), ThermostatError> {
        AndersenThermostat::flush_pending_injection(self, device)
            .map_err(ThermostatError::from)
    }

    // rq-1163481e
    fn log_column_names(&self) -> &'static [(&'static str, crate::units::Dimension)] {
        &[("andersen_conserved", crate::units::Dimension::Energy)]
    }

    // rq-6d2daea0 — The cumulative-injection accounting that the
    // historical standalone path tracked requires a kinetic-energy
    // measurement before AND after the resample. Both measurements
    // are needed in the same `apply_post` window, which is incompatible
    // with the J3 contract (apply_post does scalar prep only; the
    // resample is dispatched by the composed kernel after apply_post
    // returns). The conserved column reports `K + U` without the
    // injection correction; users running long Andersen trajectories
    // should expect detailed-balance drift in this column.
    fn log_column_values(
        &self,
        kinetic_energy: f64,
        potential_energy: f64,
    ) -> Vec<f64> {
        vec![kinetic_energy + potential_energy]
    }

    // rq-a060db3f — Andersen's post-force fragment performs the
    // per-particle Bernoulli draw and conditional Maxwell-Boltzmann
    // resample. The fragment carries its own inline Philox-4×32-10
    // implementation (slot-prefixed `andersen_*` symbols) so the
    // composed kernel needs no external `#include`. The first thread
    // of the first block increments the device draw counter at the
    // end of the per-thread body so subsequent kernel launches draw
    // a fresh Philox sequence.
    fn post_force_per_particle(&self) -> Option<&dyn crate::integrator::PostForcePerParticle> {
        Some(self)
    }

}

const ANDERSEN_PHILOX_HELPER_SOURCE: &str = r#"
__device__ inline unsigned int andersen_mulhi32(unsigned int a, unsigned int b)
{
  return __umulhi(a, b);
}

__device__ inline void andersen_philox4x32_10(
    unsigned int key_lo, unsigned int key_hi,
    unsigned int ctr0, unsigned int ctr1, unsigned int ctr2, unsigned int ctr3,
    unsigned int *out0, unsigned int *out1, unsigned int *out2, unsigned int *out3)
{
  unsigned int c0 = ctr0;
  unsigned int c1 = ctr1;
  unsigned int c2 = ctr2;
  unsigned int c3 = ctr3;
  unsigned int k0 = key_lo;
  unsigned int k1 = key_hi;
  const unsigned int M0 = 0xD2511F53u;
  const unsigned int M1 = 0xCD9E8D57u;
  const unsigned int W0 = 0x9E3779B9u;
  const unsigned int W1 = 0xBB67AE85u;
  for (int round = 0; round < 10; ++round) {
    unsigned int hi0 = andersen_mulhi32(c0, M0);
    unsigned int lo0 = c0 * M0;
    unsigned int hi2 = andersen_mulhi32(c2, M1);
    unsigned int lo2 = c2 * M1;
    c0 = hi2 ^ c1 ^ k0;
    c1 = lo2;
    c2 = hi0 ^ c3 ^ k1;
    c3 = lo0;
    k0 += W0;
    k1 += W1;
  }
  *out0 = c0;
  *out1 = c1;
  *out2 = c2;
  *out3 = c3;
}

__device__ inline double andersen_u32_to_uniform_open(unsigned int x)
{
  const double scale = 1.0 / 4294967296.0;
  return ((double)x + 0.5) * scale;
}

// Box-Muller transform: one Gaussian per call from a (uniform_hi,
// uniform_lo) pair drawn from the same Philox block. Matches the
// standalone `philox_gaussian` semantics in `kernels/philox.cuh`
// closely enough that the Andersen invariant — canonical
// distribution sampling — is preserved.
__device__ inline Real andersen_philox_gaussian(
    unsigned int seed_lo, unsigned int seed_hi,
    unsigned int draw_counter_lo, unsigned int draw_counter_hi,
    unsigned int pid, unsigned int draw_kind)
{
  unsigned int o0, o1, o2, o3;
  andersen_philox4x32_10(seed_lo, seed_hi,
                         draw_counter_lo, draw_counter_hi, pid, draw_kind,
                         &o0, &o1, &o2, &o3);
  double u1 = andersen_u32_to_uniform_open(o0);
  double u2 = andersen_u32_to_uniform_open(o1);
  double r = sqrt(-2.0 * log(u1));
  double theta = 6.283185307179586 * u2;
  return (Real)(r * cos(theta));
}
"#;

const ANDERSEN_PER_THREAD_BODY: &str = r#"
        unsigned long long andersen_counter = *andersen_draw_counter_device;
        unsigned int andersen_dc_lo = (unsigned int)(andersen_counter & 0xFFFFFFFFULL);
        unsigned int andersen_dc_hi = (unsigned int)(andersen_counter >> 32);
        unsigned int andersen_pid = andersen_particle_ids[i];

        unsigned int andersen_o0, andersen_o1, andersen_o2, andersen_o3;
        andersen_philox4x32_10(
            andersen_seed_lo, andersen_seed_hi,
            andersen_dc_lo, andersen_dc_hi, andersen_pid, 3u,
            &andersen_o0, &andersen_o1, &andersen_o2, &andersen_o3);
        double andersen_u = andersen_u32_to_uniform_open(andersen_o0);

        if (andersen_u < (double)andersen_p_collision) {
            Real andersen_m = masses[i];
            Real andersen_sigma = Real_sqrt(andersen_kt / andersen_m);
            Real andersen_xi_x = andersen_philox_gaussian(
                andersen_seed_lo, andersen_seed_hi,
                andersen_dc_lo, andersen_dc_hi, andersen_pid, 0u);
            Real andersen_xi_y = andersen_philox_gaussian(
                andersen_seed_lo, andersen_seed_hi,
                andersen_dc_lo, andersen_dc_hi, andersen_pid, 1u);
            Real andersen_xi_z = andersen_philox_gaussian(
                andersen_seed_lo, andersen_seed_hi,
                andersen_dc_lo, andersen_dc_hi, andersen_pid, 2u);
            velocities_x[i] = andersen_sigma * andersen_xi_x;
            velocities_y[i] = andersen_sigma * andersen_xi_y;
            velocities_z[i] = andersen_sigma * andersen_xi_z;
        }

        if (i == 0u) {
            *andersen_draw_counter_device = andersen_counter + 1ULL;
        }
"#;

// rq-fd0cef60
#[derive(Debug, Clone)]
pub struct AndersenBuilder;

use crate::registry::KindedBuilder;

impl KindedBuilder for AndersenBuilder {
    fn kind_name(&self) -> &'static str {
        "andersen"
    }
    fn convert_params(
        &self,
        units: crate::units::UnitSystem,
        params: &mut toml::Value,
    ) -> Result<(), crate::io::config::ConfigError> {
        crate::registry::convert_params_in_place::<AndersenParams>(units, params)
    }
}

impl ThermostatBuilder for AndersenBuilder {
    fn graph_compatible(&self, _params: &toml::Value) -> bool {
        // Andersen now does no host-side work in `apply_post` (no KE
        // dtoh; the resample is dispatched by the composed kernel).
        // Eligible for graph capture.
        true
    }

    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> {
        let p = deserialize_params(params)?;
        require_finite_positive("thermostat.temperature", p.temperature.0)?;
        require_finite_non_negative("thermostat.collision_rate", p.collision_rate.0)?;
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
            p.temperature.0,
            p.collision_rate.0,
            p.seed,
        )?;
        Ok(Box::new(state))
    }
}

// rq-2093594f rq-5e059f6b — Kept temporarily so existing PTX-load
// scaffolding keeps compiling. The standalone `andersen_resample`
// kernel is retired by K; the slot dispatches the resample via the
// JIT-composed post-force per-particle kernel.
crate::gpu_kernels! {
    module: "andersen",
    ptx: crate::kernels::ANDERSEN,
    struct: AndersenKernels,
    kernels: [andersen_resample],
    stages: {
        ANDERSEN_RESAMPLE = "andersen_resample",
    },
}
