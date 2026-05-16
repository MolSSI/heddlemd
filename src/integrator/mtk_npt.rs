// rq-3b6d5001
//
// MTK NPT integrator (isotropic, fused thermostat + barostat).

use cudarc::driver::CudaSlice;

use crate::forces::ForceField;
use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, compute_kinetic_energy, compute_total_virial,
    mtk_position_drift, mtk_velocity_half_kick, rescale_velocities,
};
use crate::io::config::IntegratorKind;
use crate::io::log_output::BOLTZMANN_J_PER_K;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::nose_hoover_chain::{nhc_chain_sub_step, yoshida_weights};
use super::{Integrator, IntegratorBuilder, IntegratorError};

// Host-side Φ_v / Φ_x factor. Computes sinh(α)/α with a Taylor
// fallback when |α| < TAYLOR_THRESHOLD so the result stays finite and
// well-conditioned near α ≈ 0.
const SINH_OVER_X_TAYLOR_THRESHOLD: f64 = 1.0e-6;

#[inline]
fn sinh_over_x(alpha: f64) -> f64 {
    if alpha.abs() < SINH_OVER_X_TAYLOR_THRESHOLD {
        // sinh(α)/α ≈ 1 + α²/6 + O(α⁴); the linear term is zero by
        // symmetry. f64 precision suffices for α down to ~1e-308.
        1.0 + alpha * alpha / 6.0
    } else {
        alpha.sinh() / alpha
    }
}

// rq-3b6d5001
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
    ke_scratch: CudaSlice<f32>,
    virial_scratch: CudaSlice<f32>,
    pub most_recent_pressure: f64,
    pub most_recent_volume: f64,
    pub most_recent_ke: f64,
}

impl MtkNptIntegrator {
    #[allow(clippy::too_many_arguments)]
    fn new(
        gpu: &GpuContext,
        particle_count: usize,
        temperature: f64,
        pressure: f64,
        tau_t: f64,
        tau_p: f64,
        chain_length: u32,
        yoshida_order: u32,
        n_resp: u32,
    ) -> Result<Self, GpuError> {
        let m = chain_length as usize;
        let g_dof = ((3 * particle_count) as i64 - 3).max(1) as u32;
        let kt = BOLTZMANN_J_PER_K * temperature;
        let tau_t2 = tau_t * tau_t;
        let tau_p2 = tau_p * tau_p;

        // Particle chain masses: Q_1 = g · k_B · T · τ_t², Q_j = k_B · T · τ_t² for j > 1.
        let mut q_mass_part = vec![0.0_f64; m];
        if m > 0 {
            q_mass_part[0] = (g_dof as f64) * kt * tau_t2;
            for j in 1..m {
                q_mass_part[j] = kt * tau_t2;
            }
        }
        // Cell chain masses: Q'_j = k_B · T · τ_t² for all j (1-DOF chain).
        let q_mass_cell = vec![kt * tau_t2; m];
        // Cell mass: W = (g + 3) · k_B · T · τ_p².
        let w_cell = (g_dof as f64 + 3.0) * kt * tau_p2;

        let ke_scratch = gpu.device.alloc_zeros::<f32>(1).map_err(GpuError::from)?;
        let virial_scratch = gpu.device.alloc_zeros::<f32>(1).map_err(GpuError::from)?;

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
        })
    }

    // Particle-chain half-step using the shared NHC helper. Mutates
    // particle velocities via rescale_velocities (one kernel launch per
    // Yoshida sub-step) and updates the particle-chain state.
    // Threads `k` through Yoshida sub-steps host-side (factor² update)
    // to avoid re-launching kinetic_energy_reduce.
    fn particle_chain_half_step(
        &mut self,
        dt: f32,
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
                let factor_f32 = factor as f32;
                timings.kernel_start(KernelStage::MTK_NPT_RESCALE_VELOCITIES)?;
                rescale_velocities(buffers, factor_f32)?;
                timings.kernel_stop(KernelStage::MTK_NPT_RESCALE_VELOCITIES)?;
                let factor_f64 = factor_f32 as f64;
                k *= factor_f64 * factor_f64;
            }
        }
        Ok(k)
    }

    // Cell-chain half-step using the shared NHC helper. Pure host
    // arithmetic; mutates the cell-chain state and `p_eps`. The "DOF"
    // it thermostats is the single scalar cell momentum, so g_dof = 1.
    fn cell_chain_half_step(&mut self, dt: f32) {
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

    // Conserved Hamiltonian for the diagnostic column.
    fn conserved_hamiltonian(&self, ke: f64, pe: f64) -> f64 {
        let mut h = ke + pe;
        h += self.pressure * self.most_recent_volume;
        h += 0.5 * self.p_eps * self.p_eps / self.w_cell;
        // Particle chain kinetic terms.
        for (p, q) in self.p_xi_part.iter().zip(self.q_mass_part.iter()) {
            h += (*p) * (*p) / (2.0 * (*q));
        }
        // Cell chain kinetic terms.
        for (p, q) in self.p_xi_cell.iter().zip(self.q_mass_cell.iter()) {
            h += (*p) * (*p) / (2.0 * (*q));
        }
        // Particle chain potential terms.
        if !self.xi_part.is_empty() {
            h += (self.g_dof as f64) * self.kt * self.xi_part[0];
            for &xi_j in self.xi_part.iter().skip(1) {
                h += self.kt * xi_j;
            }
        }
        // Cell chain potential terms (each DOF carries one k_B T).
        for &xi_j in &self.xi_cell {
            h += self.kt * xi_j;
        }
        h
    }
}

impl Integrator for MtkNptIntegrator {
    // rq-aa68f468
    fn step(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        force_field: &mut ForceField,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), IntegratorError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }

        let dt_f64 = dt as f64;
        let nf = self.g_dof as f64;

        // --- Pre: KE + virial + pressure -----------------------------
        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let mut k = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;

        timings.kernel_start(KernelStage::VIRIAL_SUM_REDUCE)?;
        let w_vir = compute_total_virial(buffers, &mut self.virial_scratch)? as f64;
        timings.kernel_stop(KernelStage::VIRIAL_SUM_REDUCE)?;

        let mut volume = sim_box.volume() as f64;
        let mut pressure = (2.0 * k + w_vir) / (3.0 * volume);

        // --- 1: cell chain ½ (host-only) -----------------------------
        self.cell_chain_half_step(dt);

        // --- 2: particle chain ½ -------------------------------------
        k = self.particle_chain_half_step(dt, buffers, k, timings)?;

        // --- 3: baro kick ½ ------------------------------------------
        // p_eps ← p_eps + (dt/2) · (3V(P − P_ext) + (3/N_f) · 2K)
        self.p_eps += 0.5 * dt_f64
            * (3.0 * volume * (pressure - self.pressure) + 6.0 / nf * k);

        // --- 4: vel kick ½ (cell-coupled half-kick from F) -----------
        // α_v = (1 + 3/N_f) · (p_eps / W); v solves dv/dt = F/m - α_v · v
        // over dt/2: v ← exp(-α_v·dt/2) · v + (dt/2) · Φ_v · F/m
        // where Φ_v = sinh(α_v·dt/4)/(α_v·dt/4) · exp(-α_v·dt/4) · 2
        // (standard MTTK form). We package the two coefficients into
        // the kernel arguments exp_minus_alpha and phi_v_dt_half so the
        // device just does v ← A·v + B·(F/m).
        let alpha_v = (1.0 + 3.0 / nf) * (self.p_eps / self.w_cell);
        let exp_ma_half = (-alpha_v * dt_f64 / 2.0).exp();
        let phi_v_dt_half = 0.5 * dt_f64
            * sinh_over_x(alpha_v * dt_f64 / 4.0)
            * (-alpha_v * dt_f64 / 4.0).exp();
        timings.kernel_start(KernelStage::MTK_NPT_VELOCITY_HALF_KICK)?;
        mtk_velocity_half_kick(buffers, exp_ma_half as f32, phi_v_dt_half as f32)?;
        timings.kernel_stop(KernelStage::MTK_NPT_VELOCITY_HALF_KICK)?;

        // --- 5: drift + box -----------------------------------------
        // β = p_eps / W; x solves dx/dt = v + β·x over dt:
        //   x ← exp(β·dt) · x + dt · Φ_x · exp(β·dt/2) · v
        // ε ← ε + β·dt; V ← V · exp(3β·dt); μ_box = exp(β·dt).
        let beta = self.p_eps / self.w_cell;
        let exp_b_dt = (beta * dt_f64).exp();
        let phi_x_dt = dt_f64 * sinh_over_x(beta * dt_f64 / 2.0) * (beta * dt_f64 / 2.0).exp();
        timings.kernel_start(KernelStage::MTK_NPT_POSITION_DRIFT)?;
        mtk_position_drift(buffers, exp_b_dt as f32, phi_x_dt as f32)?;
        timings.kernel_stop(KernelStage::MTK_NPT_POSITION_DRIFT)?;
        self.eps += beta * dt_f64;
        let mu_box = exp_b_dt as f32;
        sim_box
            .rescale_isotropic(mu_box)
            .map_err(|_| IntegratorError::Gpu(GpuError(
                cudarc::driver::DriverError(
                    cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
                ),
            )))?;

        // --- 6: force eval ------------------------------------------
        force_field.step(buffers, sim_box, timings)?;

        // --- Refresh K, W_vir, V, P at the post-drift state ---------
        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        k = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;
        timings.kernel_start(KernelStage::VIRIAL_SUM_REDUCE)?;
        let w_vir = compute_total_virial(buffers, &mut self.virial_scratch)? as f64;
        timings.kernel_stop(KernelStage::VIRIAL_SUM_REDUCE)?;
        volume = sim_box.volume() as f64;
        pressure = (2.0 * k + w_vir) / (3.0 * volume);

        // --- 7: vel kick ½ (mirror) ----------------------------------
        let alpha_v = (1.0 + 3.0 / nf) * (self.p_eps / self.w_cell);
        let exp_ma_half = (-alpha_v * dt_f64 / 2.0).exp();
        let phi_v_dt_half = 0.5 * dt_f64
            * sinh_over_x(alpha_v * dt_f64 / 4.0)
            * (-alpha_v * dt_f64 / 4.0).exp();
        timings.kernel_start(KernelStage::MTK_NPT_VELOCITY_HALF_KICK)?;
        mtk_velocity_half_kick(buffers, exp_ma_half as f32, phi_v_dt_half as f32)?;
        timings.kernel_stop(KernelStage::MTK_NPT_VELOCITY_HALF_KICK)?;

        // Refresh K after the closing velocity half-kick so the closing
        // particle chain uses the right value.
        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k_post_kick = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;

        // --- 8: baro kick ½ (mirror) ---------------------------------
        self.p_eps += 0.5 * dt_f64
            * (3.0 * volume * (pressure - self.pressure) + 6.0 / nf * k_post_kick);

        // --- 9: particle chain ½ (mirror) ----------------------------
        let k_after_part = self.particle_chain_half_step(dt, buffers, k_post_kick, timings)?;

        // --- 10: cell chain ½ (mirror; host-only) --------------------
        self.cell_chain_half_step(dt);

        self.most_recent_pressure = pressure;
        self.most_recent_volume = volume;
        self.most_recent_ke = k_after_part;
        Ok(())
    }

    // rq-3b6d5001
    fn log_column_names(&self) -> &'static [&'static str] {
        &["pressure", "box_volume", "mtk_npt_conserved"]
    }

    fn log_column_values(&self, kinetic_energy: f64, potential_energy: f64) -> Vec<f64> {
        let h = self.conserved_hamiltonian(kinetic_energy, potential_energy);
        vec![self.most_recent_pressure, self.most_recent_volume, h]
    }
}

#[derive(Debug)]
pub struct MtkNptBuilder;

impl IntegratorBuilder for MtkNptBuilder {
    fn kind_name(&self) -> &'static str {
        "mtk-npt"
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        kind: &IntegratorKind,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        match kind {
            IntegratorKind::MtkNpt {
                temperature,
                pressure,
                tau_t,
                tau_p,
                chain_length,
                yoshida_order,
                n_resp,
            } => {
                let state = MtkNptIntegrator::new(
                    gpu,
                    particle_count,
                    *temperature,
                    *pressure,
                    *tau_t,
                    *tau_p,
                    *chain_length,
                    *yoshida_order,
                    *n_resp,
                )?;
                Ok(Box::new(state))
            }
            other => Err(IntegratorError::UnknownKind(other.name().to_string())),
        }
    }
}
