// rq-f606ff6f


use cudarc::driver::CudaSlice;
use serde::Deserialize;

use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, compute_kinetic_energy, rescale_velocities,
};
use crate::io::config::ConfigError;
use crate::timings::{KernelStage, Timings};

use super::{Thermostat, ThermostatBuilder, ThermostatError};
use crate::precision::Real;

// rq-1f87880c
#[derive(Debug, Clone, Deserialize, serde::Serialize, crate::units::Convert)]
#[serde(deny_unknown_fields)]
pub struct NoseHooverChainParams {
    pub temperature: crate::units::Temperature,
    pub tau: crate::units::Time,
    #[serde(default = "default_chain_length")]
    pub chain_length: u32,
    #[serde(default = "default_yoshida_order")]
    pub yoshida_order: u32,
    #[serde(default = "default_n_resp")]
    pub n_resp: u32,
}

fn default_chain_length() -> u32 { 3 }
fn default_yoshida_order() -> u32 { 3 }
fn default_n_resp() -> u32 { 1 }

fn deserialize_params(params: &toml::Value) -> Result<NoseHooverChainParams, ConfigError> {
    params
        .clone()
        .try_into::<NoseHooverChainParams>()
        .map_err(|e| crate::io::config::translate_params_error("thermostat", e))
}

fn invalid(field: impl Into<String>, reason: impl Into<String>) -> ConfigError {
    ConfigError::InvalidValue {
        field: field.into(),
        reason: reason.into(),
    }
}

fn require_finite_positive(field: &str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(
            field,
            format!("value must be finite and strictly positive, got {value}"),
        ));
    }
    Ok(())
}

// Suzuki-Yoshida sub-step weights. The arrays are exposed as `&'static`
// slices via `yoshida_weights`.
static YOSHIDA_1: [f64; 1] = [1.0];
static YOSHIDA_3: [f64; 3] = [
    1.3512071919596577,
    -1.7024143839193155,
    1.3512071919596577,
];
static YOSHIDA_5: [f64; 5] = [
    0.41449077179437574,
    0.41449077179437574,
    -0.6579630871775030,
    0.41449077179437574,
    0.41449077179437574,
];
static YOSHIDA_7: [f64; 7] = [
    0.7845136104775573,
    0.2355732133593582,
    -1.1776799841788710,
    1.3151863206839023,
    -1.1776799841788710,
    0.2355732133593582,
    0.7845136104775573,
];

pub(super) fn yoshida_weights(n: u32) -> &'static [f64] {
    match n {
        1 => &YOSHIDA_1,
        3 => &YOSHIDA_3,
        5 => &YOSHIDA_5,
        7 => &YOSHIDA_7,
        _ => panic!("invalid yoshida_order {n}: must be 1, 3, 5, or 7"),
    }
}

/// Pure host-side Nosé-Hoover chain sub-step. Mutates `xi` and `p_xi`
/// in place; returns the multiplicative rescale factor the caller must
/// apply to the chain's thermalized DOF. Shared by the NHC thermostat
/// (which applies the factor via `rescale_velocities`) and the MTK NPT
/// integrator (which applies it to the particle velocities for the
/// particle chain, and to `p_eps` host-side for the cell chain).
///
/// - `dt` — sub-step length (already divided by `2·n_resp` and
///   weighted by the Yoshida coefficient).
/// - `k_thermalized` — kinetic energy of the thermalized DOF:
///   `2K` for an `N_f`-DOF particle chain; `p_eps²/W` for the
///   1-DOF MTK cell chain.
/// - `g_dof` — number of DOFs this chain thermostats (`N_f` for the
///   particle chain; `1.0` for the cell chain).
/// - `kt` — `k_B · T`.
// rq-3b6d5001
pub fn nhc_chain_sub_step(
    xi: &mut [f64],
    p_xi: &mut [f64],
    q_mass: &[f64],
    dt: f64,
    k_thermalized: f64,
    g_dof: f64,
    kt: f64,
) -> f64 {
    let m = xi.len();
    debug_assert_eq!(p_xi.len(), m);
    debug_assert_eq!(q_mass.len(), m);
    if m == 0 {
        return 1.0;
    }
    let mut k = k_thermalized;

    // High-to-low cascade.
    for j in (0..m).rev() {
        let s = if j == m - 1 {
            1.0
        } else {
            (-dt / 8.0 * p_xi[j + 1] / q_mass[j + 1]).exp()
        };
        p_xi[j] *= s;
        let g_j = if j == 0 {
            k - g_dof * kt
        } else {
            p_xi[j - 1].powi(2) / q_mass[j - 1] - kt
        };
        p_xi[j] += dt / 4.0 * g_j;
        p_xi[j] *= s;
    }

    // Multiplicative rescale factor for the thermalized DOF. The
    // caller applies it (particle chain: via rescale_velocities; cell
    // chain: by multiplying p_eps host-side).
    let factor = (-dt / 2.0 * p_xi[0] / q_mass[0]).exp();
    k *= factor * factor;

    // Chain position update.
    for j in 0..m {
        xi[j] += dt / 2.0 * p_xi[j] / q_mass[j];
    }

    // Low-to-high cascade.
    for j in 0..m {
        let s = if j == m - 1 {
            1.0
        } else {
            (-dt / 8.0 * p_xi[j + 1] / q_mass[j + 1]).exp()
        };
        p_xi[j] *= s;
        let g_j = if j == 0 {
            k - g_dof * kt
        } else {
            p_xi[j - 1].powi(2) / q_mass[j - 1] - kt
        };
        p_xi[j] += dt / 4.0 * g_j;
        p_xi[j] *= s;
    }

    factor
}

// rq-62e2bef5
#[derive(Debug)]
pub struct NoseHooverChainThermostat {
    pub temperature: f64,
    pub tau: f64,
    pub chain_length: u32,
    pub yoshida_order: u32,
    pub n_resp: u32,
    pub g_dof: u32,
    pub kt: f64,
    pub q_mass: Vec<f64>,
    pub xi: Vec<f64>,
    pub p_xi: Vec<f64>,
    yoshida: &'static [f64],
    ke_scratch: CudaSlice<Real>,
    most_recent_ke: f64,
}

impl NoseHooverChainThermostat {
    fn new(
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
        temperature: f64,
        tau: f64,
        chain_length: u32,
        yoshida_order: u32,
        n_resp: u32,
    ) -> Result<Self, GpuError> {
        let m = chain_length as usize;
        // Thermal DOF: `3N − n_constraints − 3` (COM momentum removed
        // at init and preserved by NHC's uniform velocity rescale).
        // Clamped to 0, NOT floored to 1 like CSVR/Berendsen/MTK: NHC
        // keeps the honest count and instead goes inert when it is 0
        // (`q_mass[0]` below would be 0, making the chain equations
        // singular — see the `g_dof == 0` guards in
        // `thermostat_half_step_cumulative` and `log_column_values`).
        let g_dof =
            ((3 * particle_count) as i64 - n_constraints as i64 - 3).max(0) as u32;
        // k_B = 1 in atomic units; temperature is already k_B · T.
        let kt = temperature;
        let tau2 = tau * tau;
        // Chain masses: `Q_1 = g · kT · τ²` (link 1 thermostats all g
        // particle DOF), `Q_j = kT · τ²` for j > 1 (each higher link
        // thermostats the single chain DOF below it).
        let mut q_mass = vec![0.0_f64; m];
        if m > 0 {
            q_mass[0] = (g_dof as f64) * kt * tau2;
            for j in 1..m {
                q_mass[j] = kt * tau2;
            }
        }
        let ke_scratch = gpu.device.alloc_zeros::<Real>(1).map_err(GpuError::from)?;
        Ok(NoseHooverChainThermostat {
            temperature,
            tau,
            chain_length,
            yoshida_order,
            n_resp,
            g_dof,
            kt,
            q_mass,
            xi: vec![0.0_f64; m],
            p_xi: vec![0.0_f64; m],
            yoshida: yoshida_weights(yoshida_order),
            ke_scratch,
            most_recent_ke: 0.0,
        })
    }

    /// Yoshida × `n_resp` chain integration, accumulating the
    /// per-iteration rescale factors into a single cumulative
    /// scalar. The cumulative factor is returned alongside the
    /// updated kinetic energy. No `rescale_velocities` launches
    /// happen inside this function; both `apply_pre` and `apply_post`
    /// apply the returned cumulative factor via `rescale_velocities`.
    fn thermostat_half_step_cumulative(
        &mut self,
        dt: Real,
        mut k: f64,
    ) -> (f64, f64) {
        // rq-fd90fab3 — zero thermal DOF makes q_mass[0] = 0 and the
        // chain equations singular; the thermostat is inert.
        if self.g_dof == 0 {
            return (1.0, k);
        }
        let dt = dt as f64;
        let n_resp = self.n_resp as f64;
        let g_dof = self.g_dof as f64;
        let kt = self.kt;
        let mut cumulative: f64 = 1.0;
        for w in self.yoshida.to_vec() {
            for _ in 0..self.n_resp {
                let delta_t = w * dt / (2.0 * n_resp);
                let factor = nhc_chain_sub_step(
                    &mut self.xi,
                    &mut self.p_xi,
                    &self.q_mass,
                    delta_t,
                    2.0 * k,
                    g_dof,
                    kt,
                );
                cumulative *= factor;
                k *= factor * factor;
            }
        }
        (cumulative, k)
    }
}

impl Thermostat for NoseHooverChainThermostat {
    // rq-2fe47a86 rq-a9c46f51
    fn apply_pre(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: Real,
        timings: &mut Timings,
    ) -> Result<(), ThermostatError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }
        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let (cumulative, _k_final) = self.thermostat_half_step_cumulative(dt, k);
        // rq-a9c46f51 — inert when g_dof == 0: no rescale launch.
        if self.g_dof != 0 {
            timings.kernel_start(KernelStage::NHC_RESCALE_VELOCITIES)?;
            rescale_velocities(buffers, cumulative as Real)?;
            timings.kernel_stop(KernelStage::NHC_RESCALE_VELOCITIES)?;
        }
        Ok(())
    }

    // rq-7a124d43 rq-370bf3a8
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
        let k = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let (cumulative, k_final) = self.thermostat_half_step_cumulative(dt, k);
        // Trailing coupling half. `k` is reduced from the full-step
        // (post-trailing-kick) velocities — a fusion barrier — so NHC
        // applies its cumulative rescale as its own launch here rather
        // than through a composed fragment. rq-a9c46f51 — inert when
        // g_dof == 0: no rescale launch.
        if self.g_dof != 0 {
            timings.kernel_start(KernelStage::NHC_RESCALE_VELOCITIES)?;
            rescale_velocities(buffers, cumulative as Real)?;
            timings.kernel_stop(KernelStage::NHC_RESCALE_VELOCITIES)?;
        }
        self.most_recent_ke = k_final;
        Ok(())
    }

    // rq-8a571737
    fn log_column_names(&self) -> &'static [(&'static str, crate::units::Dimension)] {
        // nhc_conserved is a conserved Hamiltonian-like scalar in Hartrees.
        &[("nhc_conserved", crate::units::Dimension::Energy)]
    }

    // rq-f94f6bac
    fn log_column_values(
        &self,
        kinetic_energy: f64,
        potential_energy: f64,
    ) -> Vec<f64> {
        // rq-a8dc62ec — inert thermostat (g_dof == 0): the chain never
        // propagates and p_xi[0]²/(2·q_mass[0]) would be 0/0; report
        // H' = K + U with a zero chain term.
        if self.g_dof == 0 {
            return vec![kinetic_energy + potential_energy];
        }
        let mut chain_term = 0.0_f64;
        for (p, q) in self.p_xi.iter().zip(self.q_mass.iter()) {
            chain_term += (*p) * (*p) / (2.0 * (*q));
        }
        if !self.xi.is_empty() {
            chain_term += (self.g_dof as f64) * self.kt * self.xi[0];
            for &xi_j in self.xi.iter().skip(1) {
                chain_term += self.kt * xi_j;
            }
        }
        vec![kinetic_energy + potential_energy + chain_term]
    }
}

// rq-4bd6ff2b
#[derive(Debug, Clone)]
pub struct NoseHooverChainBuilder;

use crate::registry::KindedBuilder;

impl KindedBuilder for NoseHooverChainBuilder {
    fn kind_name(&self) -> &'static str {
        "nose-hoover-chain"
    }
    fn convert_params(
        &self,
        units: crate::units::UnitSystem,
        params: &mut toml::Value,
    ) -> Result<(), crate::io::config::ConfigError> {
        crate::registry::convert_params_in_place::<NoseHooverChainParams>(units, params)
    }
}

impl ThermostatBuilder for NoseHooverChainBuilder {
    fn graph_compatible(&self, _params: &toml::Value) -> bool {
        // NHC iterates a Yoshida chain integration over `xi` / `p_xi`
        // host scalars between sub-step kernel launches and uses
        // `compute_kinetic_energy` which performs a per-step dtoh of
        // the kinetic-energy scalar. Neither can be captured in a CUDA
        // graph.
        false
    }

    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> {
        let p = deserialize_params(params)?;
        require_finite_positive("thermostat.temperature", p.temperature.0)?;
        require_finite_positive("thermostat.tau", p.tau.0)?;
        if p.chain_length < 1 {
            return Err(invalid(
                "thermostat.chain_length",
                "chain_length must be a positive integer",
            ));
        }
        if !matches!(p.yoshida_order, 1 | 3 | 5 | 7) {
            return Err(invalid(
                "thermostat.yoshida_order",
                "yoshida_order must be one of 1, 3, 5, 7",
            ));
        }
        if p.n_resp < 1 {
            return Err(invalid(
                "thermostat.n_resp",
                "n_resp must be a positive integer",
            ));
        }
        Ok(())
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
        params: &toml::Value,
    ) -> Result<Box<dyn Thermostat>, ThermostatError> {
        let p = deserialize_params(params).map_err(|_| {
            ThermostatError::UnknownKind("nose-hoover-chain (malformed params)".into())
        })?;
        let state = NoseHooverChainThermostat::new(
            gpu,
            particle_count,
            n_constraints,
            p.temperature.0,
            p.tau.0,
            p.chain_length,
            p.yoshida_order,
            p.n_resp,
        )?;
        Ok(Box::new(state))
    }
}

// rq-2093594f rq-f606ff6f
//
// The `nose_hoover` PTX module backs the kinetic-energy reduction shared
// by every thermostat and both barostats, and the per-thermostat scalar
// kernels for the Nosé-Hoover-chain, CSVR, and Berendsen thermostats; the
// thermostat timing stages are owned here accordingly. rq-5f59fa80
crate::gpu_kernels! {
    module: "nose_hoover",
    ptx: crate::kernels::NOSE_HOOVER,
    struct: NoseHooverKernels,
    kernels: [
        kinetic_energy_reduce,
        kinetic_energy_reduce_partials,
        rescale_velocities,
        rescale_velocities_device_factor,
        csvr_sample_and_factor,
        csvr_sample_partials,
        csvr_finish_from_partials,
        berendsen_compute_factor,
        increment_u64,
    ],
    stages: {
        KINETIC_ENERGY_REDUCE        = "kinetic_energy_reduce",
        NHC_RESCALE_VELOCITIES       = "nhc_rescale_velocities",
        CSVR_RESCALE_VELOCITIES      = "csvr_rescale_velocities",
        BERENDSEN_RESCALE_VELOCITIES = "berendsen_rescale_velocities",
        CSVR_SAMPLE_AND_FACTOR       = "csvr_sample_and_factor",
        BERENDSEN_COMPUTE_FACTOR     = "berendsen_compute_factor",
    },
}
