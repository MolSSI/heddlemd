// rq-3b6d5001
//
// MTK NPT integrator (isotropic, fused thermostat + barostat).

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice};
use cudarc::nvrtc::Ptx;

use crate::gpu::device::get_func;
use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, compute_kinetic_energy, compute_total_virial,
    mtk_position_drift, mtk_velocity_half_kick, rescale_velocities,
};
use crate::kernels;
use crate::io::config::ConfigError;
use serde::Deserialize;
use crate::precision::Real;

// rq-1f87880c — typed parameter struct for the "mtk-npt" builder.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MtkNptParams {
    pub temperature: f64,
    pub pressure: f64,
    pub tau_t: f64,
    pub tau_p: f64,
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

fn deserialize_params(params: &toml::Value) -> Result<MtkNptParams, ConfigError> {
    params
        .clone()
        .try_into::<MtkNptParams>()
        .map_err(|e| crate::io::config::translate_params_error("integrator", e))
}

fn invalid(field: impl Into<String>, reason: impl Into<String>) -> ConfigError {
    ConfigError::InvalidValue {
        field: field.into(),
        reason: reason.into(),
    }
}

fn require_finite(field: &str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() {
        return Err(invalid(field, format!("value must be finite, got {value}")));
    }
    Ok(())
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
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::nose_hoover_chain::{nhc_chain_sub_step, yoshida_weights};
use super::{Integrator, IntegratorBuilder, IntegratorError, StepPlan, SubStep};

// Host-side Φ_v / Φ_x factor. Computes sinh(α)/α with a Taylor
// fallback when |α| < TAYLOR_THRESHOLD so the result stays finite and
// well-conditioned near α ≈ 0.
const SINH_OVER_X_TAYLOR_THRESHOLD: f64 = 1.0e-6;

#[inline]
fn sinh_over_x(alpha: f64) -> f64 {
    if alpha.abs() < SINH_OVER_X_TAYLOR_THRESHOLD {
        1.0 + alpha * alpha / 6.0
    } else {
        alpha.sinh() / alpha
    }
}

// rq-3b6d5001 rq-508680c7
#[derive(Debug)]
pub struct MtkNptIntegrator {
    pub temperature: f64,
    pub pressure: f64,
    pub tau_t: f64,
    pub tau_p: f64,
    pub chain_length: u32,
    pub yoshida_order: u32,
    pub n_resp: u32,
    pub g_dof: u32,
    pub kt: f64,
    pub w_cell: f64,
    pub p_eps: f64,
    pub eps: f64,
    pub q_mass_part: Vec<f64>,
    pub xi_part: Vec<f64>,
    pub p_xi_part: Vec<f64>,
    pub q_mass_cell: Vec<f64>,
    pub xi_cell: Vec<f64>,
    pub p_xi_cell: Vec<f64>,
    yoshida: &'static [f64],
    ke_scratch: CudaSlice<Real>,
    virial_scratch: CudaSlice<Real>,
    pub most_recent_pressure: f64,
    pub most_recent_volume: f64,
    pub most_recent_ke: f64,
    // Per-plan-walk scratch values that flow between sub-steps. Set
    // and consumed within a single plan walk; values left over from
    // one walk are harmlessly overwritten at the start of the next.
    scratch_k: f64,
    scratch_volume: f64,
    scratch_pressure: f64,
    scratch_k_post_kick: f64,
}

impl MtkNptIntegrator {
    #[allow(clippy::too_many_arguments)]
    fn new(
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
        temperature: f64,
        pressure: f64,
        tau_t: f64,
        tau_p: f64,
        chain_length: u32,
        yoshida_order: u32,
        n_resp: u32,
    ) -> Result<Self, GpuError> {
        let m = chain_length as usize;
        let g_dof =
            ((3 * particle_count) as i64 - n_constraints as i64 - 3).max(1) as u32;
        // k_B = 1 in atomic units; temperature is already k_B · T.
        let kt = temperature;
        let tau_t2 = tau_t * tau_t;
        let tau_p2 = tau_p * tau_p;

        let mut q_mass_part = vec![0.0_f64; m];
        if m > 0 {
            q_mass_part[0] = (g_dof as f64) * kt * tau_t2;
            for j in 1..m {
                q_mass_part[j] = kt * tau_t2;
            }
        }
        let q_mass_cell = vec![kt * tau_t2; m];
        let w_cell = (g_dof as f64 + 3.0) * kt * tau_p2;

        let ke_scratch = gpu.device.alloc_zeros::<Real>(1).map_err(GpuError::from)?;
        let virial_scratch = gpu.device.alloc_zeros::<Real>(1).map_err(GpuError::from)?;

        Ok(MtkNptIntegrator {
            temperature,
            pressure,
            tau_t,
            tau_p,
            chain_length,
            yoshida_order,
            n_resp,
            g_dof,
            kt,
            w_cell,
            p_eps: 0.0,
            eps: 0.0,
            q_mass_part,
            xi_part: vec![0.0_f64; m],
            p_xi_part: vec![0.0_f64; m],
            q_mass_cell,
            xi_cell: vec![0.0_f64; m],
            p_xi_cell: vec![0.0_f64; m],
            yoshida: yoshida_weights(yoshida_order),
            ke_scratch,
            virial_scratch,
            most_recent_pressure: 0.0,
            most_recent_volume: 0.0,
            most_recent_ke: 0.0,
            scratch_k: 0.0,
            scratch_volume: 0.0,
            scratch_pressure: 0.0,
            scratch_k_post_kick: 0.0,
        })
    }

    // Particle-chain half-step using the shared NHC helper. Mutates
    // particle velocities via rescale_velocities (one kernel launch
    // per Yoshida sub-step) and updates the particle-chain state.
    fn particle_chain_half_step(
        &mut self,
        dt: Real,
        buffers: &mut ParticleBuffers,
        mut k: f64,
        timings: &mut Timings,
    ) -> Result<f64, IntegratorError> {
        let dt = dt as f64;
        let n_resp = self.n_resp as f64;
        let g_dof = self.g_dof as f64;
        for w in self.yoshida.to_vec() {
            for _ in 0..self.n_resp {
                let delta_t = w * dt / (2.0 * n_resp);
                let factor = nhc_chain_sub_step(
                    &mut self.xi_part,
                    &mut self.p_xi_part,
                    &self.q_mass_part,
                    delta_t,
                    2.0 * k,
                    g_dof,
                    self.kt,
                );
                let factor = factor as Real;
                timings.kernel_start(KernelStage::MTK_NPT_RESCALE_VELOCITIES)?;
                rescale_velocities(buffers, factor)?;
                timings.kernel_stop(KernelStage::MTK_NPT_RESCALE_VELOCITIES)?;
                let factor_f64 = factor as f64;
                k *= factor_f64 * factor_f64;
            }
        }
        Ok(k)
    }

    fn cell_chain_half_step(&mut self, dt: Real) {
        let dt = dt as f64;
        let n_resp = self.n_resp as f64;
        for w in self.yoshida.to_vec() {
            for _ in 0..self.n_resp {
                let delta_t = w * dt / (2.0 * n_resp);
                let k_thermalized = self.p_eps * self.p_eps / self.w_cell;
                let factor = nhc_chain_sub_step(
                    &mut self.xi_cell,
                    &mut self.p_xi_cell,
                    &self.q_mass_cell,
                    delta_t,
                    k_thermalized,
                    1.0,
                    self.kt,
                );
                self.p_eps *= factor;
            }
        }
    }

    fn conserved_hamiltonian(&self, ke: f64, pe: f64) -> f64 {
        let mut h = ke + pe;
        h += self.pressure * self.most_recent_volume;
        h += 0.5 * self.p_eps * self.p_eps / self.w_cell;
        for (p, q) in self.p_xi_part.iter().zip(self.q_mass_part.iter()) {
            h += (*p) * (*p) / (2.0 * (*q));
        }
        for (p, q) in self.p_xi_cell.iter().zip(self.q_mass_cell.iter()) {
            h += (*p) * (*p) / (2.0 * (*q));
        }
        if !self.xi_part.is_empty() {
            h += (self.g_dof as f64) * self.kt * self.xi_part[0];
            for &xi_j in self.xi_part.iter().skip(1) {
                h += self.kt * xi_j;
            }
        }
        for &xi_j in &self.xi_cell {
            h += self.kt * xi_j;
        }
        h
    }
}

impl Integrator for MtkNptIntegrator {
    // rq-aa68f468 rq-8cda2c89
    fn plan(&self, dt: Real) -> StepPlan {
        // The MTK symmetric Trotter splitting. Most sub-steps are
        // integrator-private (`Custom`) because they involve host-side
        // chain arithmetic or KE / virial reductions that the constraint
        // framework's hook insertion machinery (Drift / KickDrift /
        // final KickHalf) does not want to interleave. The KickHalf
        // sub-step is the cell-coupled velocity update; the Drift
        // sub-step is the cell-coupled position update plus the
        // SimulationBox rescale.
        StepPlan {
            steps: vec![
                SubStep::Custom { dt, label: "ke_reduce_pre" },
                SubStep::Custom { dt, label: "vir_reduce_pre" },
                SubStep::Custom { dt, label: "cell_chain_pre" },
                SubStep::Custom { dt, label: "particle_chain_pre" },
                SubStep::Custom { dt, label: "baro_kick_pre" },
                SubStep::KickHalf { dt, label: "vel_kick_pre" },
                SubStep::Drift { dt, label: "drift_box" },
                SubStep::ForceEval {
                    class: None,
                    level: Some(crate::forces::AggregateLevel::ForcesAndScalars),
                },
                SubStep::Custom { dt, label: "ke_reduce_post" },
                SubStep::Custom { dt, label: "vir_reduce_post" },
                SubStep::KickHalf { dt, label: "vel_kick_post" },
                SubStep::Custom { dt, label: "ke_reduce_post_kick" },
                SubStep::Custom { dt, label: "baro_kick_post" },
                SubStep::Custom { dt, label: "particle_chain_post" },
                SubStep::Custom { dt, label: "cell_chain_post" },
            ],
        }
    }

    fn execute(
        &mut self,
        substep: &SubStep,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        timings: &mut Timings,
    ) -> Result<(), IntegratorError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }
        let dt = match substep {
            SubStep::KickHalf { dt, .. }
            | SubStep::Drift { dt, .. }
            | SubStep::Custom { dt, .. } => *dt,
            other => {
                return Err(IntegratorError::UnexpectedSubStep {
                    variant: other.variant_name(),
                });
            }
        };
        let dt_f64 = dt as f64;
        let nf = self.g_dof as f64;

        match substep {
            SubStep::Custom { label: "ke_reduce_pre", .. } => {
                timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
                self.scratch_k = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
                timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;
                Ok(())
            }
            SubStep::Custom { label: "vir_reduce_pre", .. } => {
                timings.kernel_start(KernelStage::VIRIAL_SUM_REDUCE)?;
                let w_vir = compute_total_virial(buffers, &mut self.virial_scratch)? as f64;
                timings.kernel_stop(KernelStage::VIRIAL_SUM_REDUCE)?;
                self.scratch_volume = sim_box.volume() as f64;
                self.scratch_pressure =
                    (2.0 * self.scratch_k + w_vir) / (3.0 * self.scratch_volume);
                Ok(())
            }
            SubStep::Custom { label: "cell_chain_pre", .. } => {
                self.cell_chain_half_step(dt);
                Ok(())
            }
            SubStep::Custom { label: "particle_chain_pre", .. } => {
                self.scratch_k =
                    self.particle_chain_half_step(dt, buffers, self.scratch_k, timings)?;
                Ok(())
            }
            SubStep::Custom { label: "baro_kick_pre", .. } => {
                // p_eps ← p_eps + (dt/2) · (3V(P − P_ext) + (6/N_f) · K)
                self.p_eps += 0.5
                    * dt_f64
                    * (3.0 * self.scratch_volume * (self.scratch_pressure - self.pressure)
                        + 6.0 / nf * self.scratch_k);
                Ok(())
            }
            SubStep::KickHalf { label: "vel_kick_pre", .. }
            | SubStep::KickHalf { label: "vel_kick_post", .. } => {
                let alpha_v = (1.0 + 3.0 / nf) * (self.p_eps / self.w_cell);
                let exp_ma_half = (-alpha_v * dt_f64 / 2.0).exp();
                let phi_v_dt_half = 0.5
                    * dt_f64
                    * sinh_over_x(alpha_v * dt_f64 / 4.0)
                    * (-alpha_v * dt_f64 / 4.0).exp();
                timings.kernel_start(KernelStage::MTK_NPT_VELOCITY_HALF_KICK)?;
                mtk_velocity_half_kick(buffers, exp_ma_half as Real, phi_v_dt_half as Real)?;
                timings.kernel_stop(KernelStage::MTK_NPT_VELOCITY_HALF_KICK)?;
                Ok(())
            }
            SubStep::Drift { label: "drift_box", .. } => {
                let beta = self.p_eps / self.w_cell;
                let exp_b_dt = (beta * dt_f64).exp();
                let phi_x_dt =
                    dt_f64 * sinh_over_x(beta * dt_f64 / 2.0) * (beta * dt_f64 / 2.0).exp();
                timings.kernel_start(KernelStage::MTK_NPT_POSITION_DRIFT)?;
                mtk_position_drift(buffers, exp_b_dt as Real, phi_x_dt as Real)?;
                timings.kernel_stop(KernelStage::MTK_NPT_POSITION_DRIFT)?;
                self.eps += beta * dt_f64;
                let mu_box = exp_b_dt as Real;
                sim_box.multiply_lattice_isotropic(mu_box).map_err(|_| {
                    IntegratorError::Gpu(GpuError(cudarc::driver::DriverError(
                        cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
                    )))
                })?;
                Ok(())
            }
            SubStep::Custom { label: "ke_reduce_post", .. } => {
                timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
                self.scratch_k = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
                timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;
                Ok(())
            }
            SubStep::Custom { label: "vir_reduce_post", .. } => {
                timings.kernel_start(KernelStage::VIRIAL_SUM_REDUCE)?;
                let w_vir = compute_total_virial(buffers, &mut self.virial_scratch)? as f64;
                timings.kernel_stop(KernelStage::VIRIAL_SUM_REDUCE)?;
                // Host volume is stale after drift_box's
                // `multiply_lattice_isotropic`; refresh from device
                // so `scratch_volume` reflects the post-drift box.
                sim_box.flush_from_device().map_err(|_| {
                    IntegratorError::Gpu(GpuError(cudarc::driver::DriverError(
                        cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
                    )))
                })?;
                self.scratch_volume = sim_box.volume() as f64;
                self.scratch_pressure =
                    (2.0 * self.scratch_k + w_vir) / (3.0 * self.scratch_volume);
                Ok(())
            }
            SubStep::Custom { label: "ke_reduce_post_kick", .. } => {
                timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
                self.scratch_k_post_kick =
                    compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
                timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;
                Ok(())
            }
            SubStep::Custom { label: "baro_kick_post", .. } => {
                self.p_eps += 0.5
                    * dt_f64
                    * (3.0 * self.scratch_volume * (self.scratch_pressure - self.pressure)
                        + 6.0 / nf * self.scratch_k_post_kick);
                Ok(())
            }
            SubStep::Custom { label: "particle_chain_post", .. } => {
                let k_after_part = self.particle_chain_half_step(
                    dt,
                    buffers,
                    self.scratch_k_post_kick,
                    timings,
                )?;
                self.most_recent_pressure = self.scratch_pressure;
                self.most_recent_volume = self.scratch_volume;
                self.most_recent_ke = k_after_part;
                Ok(())
            }
            SubStep::Custom { label: "cell_chain_post", .. } => {
                self.cell_chain_half_step(dt);
                Ok(())
            }
            other => Err(IntegratorError::UnexpectedSubStep {
                variant: other.variant_name(),
            }),
        }
    }

    // rq-3b6d5001 rq-14a7685e
    fn log_column_names(&self) -> &'static [(&'static str, crate::units::Dimension)] {
        use crate::units::Dimension;
        &[
            ("pressure", Dimension::Pressure),
            ("box_volume", Dimension::Dimensionless),
            ("mtk_npt_conserved", Dimension::Energy),
        ]
    }

    // rq-f9ebe53f
    fn log_column_values(&self, kinetic_energy: f64, potential_energy: f64) -> Vec<f64> {
        let h = self.conserved_hamiltonian(kinetic_energy, potential_energy);
        vec![self.most_recent_pressure, self.most_recent_volume, h]
    }
}

// rq-0b7f7023
#[derive(Debug, Clone)]
pub struct MtkNptBuilder;

impl IntegratorBuilder for MtkNptBuilder {
    fn kind_name(&self) -> &'static str {
        "mtk-npt"
    }

    fn graph_compatible(&self, _params: &toml::Value) -> bool {
        // MTK-NPT's sub-step executor mutates `self.eps` between
        // sub-steps and reads `sim_box.volume()` post-drift — both are
        // host-side scalar operations that a CUDA graph cannot
        // capture.
        false
    }

    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> {
        let p = deserialize_params(params)?;
        require_finite_positive("integrator.temperature", p.temperature)?;
        require_finite("integrator.pressure", p.pressure)?;
        require_finite_positive("integrator.tau_t", p.tau_t)?;
        require_finite_positive("integrator.tau_p", p.tau_p)?;
        if p.chain_length < 1 {
            return Err(invalid(
                "integrator.chain_length",
                "chain_length must be a positive integer",
            ));
        }
        if !matches!(p.yoshida_order, 1 | 3 | 5 | 7) {
            return Err(invalid(
                "integrator.yoshida_order",
                "yoshida_order must be one of 1, 3, 5, 7",
            ));
        }
        if p.n_resp < 1 {
            return Err(invalid(
                "integrator.n_resp",
                "n_resp must be a positive integer",
            ));
        }
        Ok(())
    }

    fn owns_thermostat(&self, _params: &toml::Value) -> bool { true }
    fn owns_barostat(&self, _params: &toml::Value) -> bool { true }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        n_constraints: usize,
        params: &toml::Value,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        let p = deserialize_params(params)
            .map_err(|_| IntegratorError::UnknownKind("mtk-npt (malformed params)".into()))?;
        let state = MtkNptIntegrator::new(
            gpu,
            particle_count,
            n_constraints,
            p.temperature,
            p.pressure,
            p.tau_t,
            p.tau_p,
            p.chain_length,
            p.yoshida_order,
            p.n_resp,
        )?;
        Ok(Box::new(state))
    }

    fn box_clone(&self) -> Box<dyn IntegratorBuilder> {
        Box::new(self.clone())
    }
}

// rq-2093594f rq-3b6d5001
#[derive(Debug, Clone)]
pub struct MtkKernels {
    pub mtk_velocity_half_kick: CudaFunction,
    pub mtk_position_drift: CudaFunction,
}

impl MtkKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::MTK),
            "mtk",
            &["mtk_velocity_half_kick", "mtk_position_drift"],
        )?;
        Ok(MtkKernels {
            mtk_velocity_half_kick: get_func(device, "mtk", "mtk_velocity_half_kick")?,
            mtk_position_drift: get_func(device, "mtk", "mtk_position_drift")?,
        })
    }
}
