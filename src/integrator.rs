// rq-4f386df8 rq-5cb33196 rq-67414c32
use cudarc::driver::CudaSlice;

use crate::forces::{ForceField, ForceFieldError};
use crate::gpu::{
    GpuContext, GpuError, LosslessBuffers, ParticleBuffers, compute_kinetic_energy,
    lan_drift_half, lan_ou_step, rescale_velocities, vv_kick, vv_kick_drift,
    vv_kick_drift_lossless, vv_kick_lossless,
};
use crate::io::config::IntegratorKind;
use crate::io::log_output::BOLTZMANN_J_PER_K;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings, TimingsError};

// rq-a5069572 rq-e1ceb5c0 rq-6cf916af
#[derive(Debug, thiserror::Error)]
pub enum IntegratorError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("{0}")]
    Timings(#[from] TimingsError),
    #[error("{0}")]
    ForceField(#[from] ForceFieldError),
    #[error("unknown integrator kind `{0}`")]
    UnknownKind(String),
}

// rq-e4c4ff61
pub trait Integrator: std::fmt::Debug + Send {
    fn step(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        force_field: &mut ForceField,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), IntegratorError>;

    /// Diagnostic column names this integrator wants the runner to
    /// include in the CSV log (`io/log-output.md`). Returned slice has
    /// `'static` lifetime so the runner can pass it to `LogWriter::open`
    /// without copying. Default: empty.
    fn log_column_names(&self) -> &'static [&'static str] {
        &[]
    }

    /// Current values of those columns. The runner supplies the total
    /// kinetic and potential energies (in joules) it has just computed
    /// for the log row; the integrator combines them with its own state
    /// to produce the requested values. Returned `Vec` length must
    /// equal `log_column_names().len()`. Default: empty.
    fn log_column_values(
        &self,
        _kinetic_energy: f64,
        _potential_energy: f64,
    ) -> Vec<f64> {
        Vec::new()
    }
}

// rq-87fdd9b1
pub trait IntegratorBuilder: std::fmt::Debug + Send + Sync {
    fn kind_name(&self) -> &'static str;
    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        kind: &IntegratorKind,
    ) -> Result<Box<dyn Integrator>, IntegratorError>;
}

// rq-1d5b5e35
#[derive(Debug)]
pub struct IntegratorRegistry {
    pub builders: Vec<Box<dyn IntegratorBuilder>>,
}

impl IntegratorRegistry {
    pub fn new() -> Self {
        IntegratorRegistry { builders: Vec::new() }
    }

    pub fn with_builtins() -> Self {
        IntegratorRegistry {
            builders: vec![
                Box::new(VelocityVerletBuilder),
                Box::new(LangevinBaoabBuilder),
                Box::new(NoseHooverChainBuilder),
            ],
        }
    }

    pub fn register(&mut self, builder: Box<dyn IntegratorBuilder>) {
        self.builders.push(builder);
    }

    // rq-df39d15b
    pub fn build(
        &self,
        kind: &IntegratorKind,
        gpu: &GpuContext,
        particle_count: usize,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        let target = kind.name();
        for b in &self.builders {
            if b.kind_name() == target {
                return b.build(gpu, particle_count, kind);
            }
        }
        Err(IntegratorError::UnknownKind(target.to_string()))
    }
}

impl Default for IntegratorRegistry {
    fn default() -> Self {
        IntegratorRegistry::with_builtins()
    }
}

// --- Velocity Verlet ---

#[derive(Debug)]
pub struct VelocityVerletState {
    lossless: Option<LosslessBuffers>,
}

impl Integrator for VelocityVerletState {
    // rq-cf361ff5
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

        if let Some(ll) = self.lossless.as_mut() {
            timings.kernel_start(KernelStage::VV_KICK_DRIFT_LOSSLESS)?;
            vv_kick_drift_lossless(buffers, ll, sim_box, dt)?;
            timings.kernel_stop(KernelStage::VV_KICK_DRIFT_LOSSLESS)?;
        } else {
            timings.kernel_start(KernelStage::VV_KICK_DRIFT)?;
            vv_kick_drift(buffers, sim_box, dt)?;
            timings.kernel_stop(KernelStage::VV_KICK_DRIFT)?;
        }

        force_field.step(buffers, sim_box, timings)?;

        if let Some(ll) = self.lossless.as_mut() {
            timings.kernel_start(KernelStage::VV_KICK_LOSSLESS)?;
            vv_kick_lossless(buffers, ll, dt)?;
            timings.kernel_stop(KernelStage::VV_KICK_LOSSLESS)?;
        } else {
            timings.kernel_start(KernelStage::VV_KICK)?;
            vv_kick(buffers, dt)?;
            timings.kernel_stop(KernelStage::VV_KICK)?;
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct VelocityVerletBuilder;

impl IntegratorBuilder for VelocityVerletBuilder {
    fn kind_name(&self) -> &'static str {
        "velocity-verlet"
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        kind: &IntegratorKind,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        match kind {
            IntegratorKind::VelocityVerlet { lossless } => {
                let buffers = if *lossless {
                    Some(LosslessBuffers::new(gpu, particle_count)?)
                } else {
                    None
                };
                Ok(Box::new(VelocityVerletState { lossless: buffers }))
            }
            other => Err(IntegratorError::UnknownKind(other.name().to_string())),
        }
    }
}

// --- Langevin BAOAB ---

#[derive(Debug)]
pub struct LangevinBaoabState {
    pub friction: f64,
    pub temperature: f64,
    pub seed: u64,
    pub draw_counter: u64,
}

impl Integrator for LangevinBaoabState {
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

        // BAOAB pre-force: B(dt/2), A(dt/2), O(dt), A(dt/2)
        timings.kernel_start(KernelStage::LANGEVIN_KICK_HALF)?;
        vv_kick(buffers, dt)?;
        timings.kernel_stop(KernelStage::LANGEVIN_KICK_HALF)?;

        timings.kernel_start(KernelStage::LANGEVIN_DRIFT_HALF)?;
        lan_drift_half(buffers, sim_box, dt)?;
        timings.kernel_stop(KernelStage::LANGEVIN_DRIFT_HALF)?;

        let alpha = (-(self.friction as f32) * dt).exp();
        let kt = (BOLTZMANN_J_PER_K * self.temperature) as f32;
        self.draw_counter += 1;
        timings.kernel_start(KernelStage::LANGEVIN_OU_STEP)?;
        lan_ou_step(buffers, self.seed, self.draw_counter, alpha, kt)?;
        timings.kernel_stop(KernelStage::LANGEVIN_OU_STEP)?;

        timings.kernel_start(KernelStage::LANGEVIN_DRIFT_HALF)?;
        lan_drift_half(buffers, sim_box, dt)?;
        timings.kernel_stop(KernelStage::LANGEVIN_DRIFT_HALF)?;

        // Force evaluation at the new positions.
        force_field.step(buffers, sim_box, timings)?;

        // BAOAB post-force: B(dt/2)
        timings.kernel_start(KernelStage::LANGEVIN_KICK_HALF)?;
        vv_kick(buffers, dt)?;
        timings.kernel_stop(KernelStage::LANGEVIN_KICK_HALF)?;

        Ok(())
    }
}

#[derive(Debug)]
pub struct LangevinBaoabBuilder;

impl IntegratorBuilder for LangevinBaoabBuilder {
    fn kind_name(&self) -> &'static str {
        "langevin-baoab"
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        kind: &IntegratorKind,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        let _ = gpu;
        let _ = particle_count;
        match kind {
            IntegratorKind::LangevinBaoab {
                friction,
                temperature,
                seed,
            } => Ok(Box::new(LangevinBaoabState {
                friction: *friction,
                temperature: *temperature,
                seed: *seed,
                draw_counter: 0,
            })),
            other => Err(IntegratorError::UnknownKind(other.name().to_string())),
        }
    }
}

// --- Nosé-Hoover chain (NHC) ---
// rq-f606ff6f

// Suzuki-Yoshida sub-step weights. The arrays are exposed as `&'static`
// slices via `yoshida_weights`. The n=3 and n=5 values come from
// `1/(2 − 2^(1/3))` and `1/(4 − 4^(1/3))`, precomputed in `f64`. The
// n=7 sequence is the standard Yoshida 1990 / Suzuki 1990 7-stage
// symmetric splitting.
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

fn yoshida_weights(n: u32) -> &'static [f64] {
    match n {
        1 => &YOSHIDA_1,
        3 => &YOSHIDA_3,
        5 => &YOSHIDA_5,
        7 => &YOSHIDA_7,
        _ => panic!("invalid yoshida_order {n}: must be 1, 3, 5, or 7"),
    }
}

#[derive(Debug)]
pub struct NoseHooverChainState {
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
    ke_scratch: CudaSlice<f32>,
    most_recent_ke: f64,
}

impl NoseHooverChainState {
    fn new(
        gpu: &GpuContext,
        particle_count: usize,
        temperature: f64,
        tau: f64,
        chain_length: u32,
        yoshida_order: u32,
        n_resp: u32,
    ) -> Result<Self, GpuError> {
        let m = chain_length as usize;
        let g_dof = ((3 * particle_count) as i64 - 3).max(0) as u32;
        let kt = BOLTZMANN_J_PER_K * temperature;
        let tau2 = tau * tau;
        let mut q_mass = vec![0.0_f64; m];
        if m > 0 {
            q_mass[0] = (g_dof as f64) * kt * tau2;
            for j in 1..m {
                q_mass[j] = kt * tau2;
            }
        }
        let ke_scratch = gpu.device.alloc_zeros::<f32>(1).map_err(GpuError::from)?;
        Ok(NoseHooverChainState {
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

    // One MKT chain sub-step of length `dt`. `k` is the current kinetic
    // energy in joules; the returned value is `k` after the velocity
    // rescale this sub-step performed. Mutates `self.xi`, `self.p_xi`,
    // and `buffers.velocities_*`.
    fn chain_sub_step(
        &mut self,
        dt: f64,
        buffers: &mut ParticleBuffers,
        mut k: f64,
        timings: &mut Timings,
    ) -> Result<f64, IntegratorError> {
        let m = self.chain_length as usize;
        if m == 0 {
            return Ok(k);
        }
        let kt = self.kt;
        let g = self.g_dof as f64;

        // High-to-low cascade.
        for j in (0..m).rev() {
            let s = if j == m - 1 {
                1.0
            } else {
                (-dt / 8.0 * self.p_xi[j + 1] / self.q_mass[j + 1]).exp()
            };
            self.p_xi[j] *= s;
            let g_j = if j == 0 {
                2.0 * k - g * kt
            } else {
                self.p_xi[j - 1].powi(2) / self.q_mass[j - 1] - kt
            };
            self.p_xi[j] += dt / 4.0 * g_j;
            self.p_xi[j] *= s;
        }

        // Particle velocity rescale.
        let factor = (-dt / 2.0 * self.p_xi[0] / self.q_mass[0]).exp() as f32;
        timings.kernel_start(KernelStage::NHC_RESCALE_VELOCITIES)?;
        rescale_velocities(buffers, factor)?;
        timings.kernel_stop(KernelStage::NHC_RESCALE_VELOCITIES)?;
        let factor_f64 = factor as f64;
        k *= factor_f64 * factor_f64;

        // Chain position update.
        for j in 0..m {
            self.xi[j] += dt / 2.0 * self.p_xi[j] / self.q_mass[j];
        }

        // Low-to-high cascade.
        for j in 0..m {
            let s = if j == m - 1 {
                1.0
            } else {
                (-dt / 8.0 * self.p_xi[j + 1] / self.q_mass[j + 1]).exp()
            };
            self.p_xi[j] *= s;
            let g_j = if j == 0 {
                2.0 * k - g * kt
            } else {
                self.p_xi[j - 1].powi(2) / self.q_mass[j - 1] - kt
            };
            self.p_xi[j] += dt / 4.0 * g_j;
            self.p_xi[j] *= s;
        }

        Ok(k)
    }

    fn thermostat_half_step(
        &mut self,
        dt: f32,
        buffers: &mut ParticleBuffers,
        mut k: f64,
        timings: &mut Timings,
    ) -> Result<f64, IntegratorError> {
        let dt = dt as f64;
        let n_resp = self.n_resp as f64;
        for w in self.yoshida.to_vec() {
            for _ in 0..self.n_resp {
                let delta_t = w * dt / (2.0 * n_resp);
                k = self.chain_sub_step(delta_t, buffers, k, timings)?;
            }
        }
        Ok(k)
    }
}

impl Integrator for NoseHooverChainState {
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

        // First KE reduce + thermostat half-step.
        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k = self.thermostat_half_step(dt, buffers, k, timings)?;
        let _ = k;

        // Velocity-Verlet core.
        timings.kernel_start(KernelStage::VV_KICK_DRIFT)?;
        vv_kick_drift(buffers, sim_box, dt)?;
        timings.kernel_stop(KernelStage::VV_KICK_DRIFT)?;

        force_field.step(buffers, sim_box, timings)?;

        timings.kernel_start(KernelStage::VV_KICK)?;
        vv_kick(buffers, dt)?;
        timings.kernel_stop(KernelStage::VV_KICK)?;

        // Second KE reduce + thermostat half-step.
        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k = self.thermostat_half_step(dt, buffers, k, timings)?;
        self.most_recent_ke = k;

        Ok(())
    }

    fn log_column_names(&self) -> &'static [&'static str] {
        &["nhc_conserved"]
    }

    fn log_column_values(
        &self,
        kinetic_energy: f64,
        potential_energy: f64,
    ) -> Vec<f64> {
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

#[derive(Debug)]
pub struct NoseHooverChainBuilder;

impl IntegratorBuilder for NoseHooverChainBuilder {
    fn kind_name(&self) -> &'static str {
        "nose-hoover-chain"
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        kind: &IntegratorKind,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        match kind {
            IntegratorKind::NoseHooverChain {
                temperature,
                tau,
                chain_length,
                yoshida_order,
                n_resp,
            } => {
                let state = NoseHooverChainState::new(
                    gpu,
                    particle_count,
                    *temperature,
                    *tau,
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

