// rq-e0a0553d rq-6cd635cd rq-6c5b4246
//
// Three orthogonal slot frameworks: integrator, thermostat, barostat.
// The runner chains the slots `apply_pre → step → apply_post → apply`
// per timestep (see `simulation-runner.md` and `framework.md`).
use cudarc::driver::CudaSlice;

use crate::forces::{ForceField, ForceFieldError};
use crate::gpu::{
    GpuContext, GpuError, LosslessBuffers, ParticleBuffers, andersen_resample,
    compute_kinetic_energy, lan_drift_half, lan_ou_step, rescale_velocities, vv_kick,
    vv_kick_drift, vv_kick_drift_lossless, vv_kick_lossless,
};
use crate::io::config::{BarostatKind, IntegratorKind, ThermostatKind};
use crate::io::log_output::BOLTZMANN_J_PER_K;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings, TimingsError};

// rq-2ccf40de
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

// rq-2ccf40de
#[derive(Debug, thiserror::Error)]
pub enum ThermostatError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("{0}")]
    Timings(#[from] TimingsError),
    #[error("unknown thermostat kind `{0}`")]
    UnknownKind(String),
}

// rq-2ccf40de
#[derive(Debug, thiserror::Error)]
pub enum BarostatError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("{0}")]
    Timings(#[from] TimingsError),
    #[error("unknown barostat kind `{0}`")]
    UnknownKind(String),
}

// --- Integrator trait, builder, registry ------------------------------

// rq-78f484d9
pub trait Integrator: std::fmt::Debug + Send {
    // rq-aa68f468
    fn step(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        force_field: &mut ForceField,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), IntegratorError>;

    fn log_column_names(&self) -> &'static [&'static str] {
        &[]
    }

    fn log_column_values(
        &self,
        _kinetic_energy: f64,
        _potential_energy: f64,
    ) -> Vec<f64> {
        Vec::new()
    }
}

// rq-29e08cb5
pub trait IntegratorBuilder: std::fmt::Debug + Send + Sync {
    fn kind_name(&self) -> &'static str;
    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        kind: &IntegratorKind,
    ) -> Result<Box<dyn Integrator>, IntegratorError>;
}

// rq-4901507f
#[derive(Debug)]
pub struct IntegratorRegistry {
    pub builders: Vec<Box<dyn IntegratorBuilder>>,
}

impl IntegratorRegistry {
    pub fn new() -> Self {
        IntegratorRegistry { builders: Vec::new() }
    }

    // rq-4901507f
    pub fn with_builtins() -> Self {
        IntegratorRegistry {
            builders: vec![
                Box::new(VelocityVerletBuilder),
                Box::new(LangevinBaoabBuilder),
            ],
        }
    }

    pub fn register(&mut self, builder: Box<dyn IntegratorBuilder>) {
        self.builders.push(builder);
    }

    // rq-24f6b8b9
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

// --- Thermostat trait, builder, registry ------------------------------

// rq-5d9ed248
pub trait Thermostat: std::fmt::Debug + Send {
    // rq-2fe47a86
    fn apply_pre(
        &mut self,
        _buffers: &mut ParticleBuffers,
        _dt: f32,
        _timings: &mut Timings,
    ) -> Result<(), ThermostatError> {
        Ok(())
    }

    // rq-7a124d43
    fn apply_post(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), ThermostatError>;

    fn log_column_names(&self) -> &'static [&'static str] {
        &[]
    }

    fn log_column_values(
        &self,
        _kinetic_energy: f64,
        _potential_energy: f64,
    ) -> Vec<f64> {
        Vec::new()
    }
}

// rq-29e08cb5
pub trait ThermostatBuilder: std::fmt::Debug + Send + Sync {
    fn kind_name(&self) -> &'static str;
    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        kind: &ThermostatKind,
    ) -> Result<Box<dyn Thermostat>, ThermostatError>;
}

// rq-4901507f
#[derive(Debug)]
pub struct ThermostatRegistry {
    pub builders: Vec<Box<dyn ThermostatBuilder>>,
}

impl ThermostatRegistry {
    pub fn new() -> Self {
        ThermostatRegistry { builders: Vec::new() }
    }

    // rq-4901507f
    pub fn with_builtins() -> Self {
        ThermostatRegistry {
            builders: vec![
                Box::new(NoseHooverChainBuilder),
                Box::new(CsvrBuilder),
                Box::new(AndersenBuilder),
                Box::new(BerendsenBuilder),
            ],
        }
    }

    pub fn register(&mut self, builder: Box<dyn ThermostatBuilder>) {
        self.builders.push(builder);
    }

    // rq-678c233d
    pub fn build_optional(
        &self,
        kind: Option<&ThermostatKind>,
        gpu: &GpuContext,
        particle_count: usize,
    ) -> Result<Option<Box<dyn Thermostat>>, ThermostatError> {
        let Some(kind) = kind else { return Ok(None) };
        let target = kind.name();
        for b in &self.builders {
            if b.kind_name() == target {
                return Ok(Some(b.build(gpu, particle_count, kind)?));
            }
        }
        Err(ThermostatError::UnknownKind(target.to_string()))
    }
}

impl Default for ThermostatRegistry {
    fn default() -> Self {
        ThermostatRegistry::with_builtins()
    }
}

// --- Barostat trait, builder, registry --------------------------------

// rq-076617ab
pub trait Barostat: std::fmt::Debug + Send {
    // rq-1179e42f
    fn apply(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), BarostatError>;

    fn log_column_names(&self) -> &'static [&'static str] {
        &[]
    }

    fn log_column_values(
        &self,
        _kinetic_energy: f64,
        _potential_energy: f64,
    ) -> Vec<f64> {
        Vec::new()
    }
}

// rq-29e08cb5
pub trait BarostatBuilder: std::fmt::Debug + Send + Sync {
    fn kind_name(&self) -> &'static str;
    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        kind: &BarostatKind,
    ) -> Result<Box<dyn Barostat>, BarostatError>;
}

// rq-4901507f
#[derive(Debug)]
pub struct BarostatRegistry {
    pub builders: Vec<Box<dyn BarostatBuilder>>,
}

impl BarostatRegistry {
    pub fn new() -> Self {
        BarostatRegistry { builders: Vec::new() }
    }

    // rq-4901507f
    // The default registry registers no barostat implementations. The
    // trait surface and config slot exist so that the Berendsen barostat
    // (and future barostats) can be added without further framework
    // changes; see `framework.md`.
    pub fn with_builtins() -> Self {
        BarostatRegistry { builders: Vec::new() }
    }

    pub fn register(&mut self, builder: Box<dyn BarostatBuilder>) {
        self.builders.push(builder);
    }

    // rq-9548bc1a
    pub fn build_optional(
        &self,
        kind: Option<&BarostatKind>,
        gpu: &GpuContext,
        particle_count: usize,
    ) -> Result<Option<Box<dyn Barostat>>, BarostatError> {
        let Some(kind) = kind else { return Ok(None) };
        let target = kind.name();
        for b in &self.builders {
            if b.kind_name() == target {
                return Ok(Some(b.build(gpu, particle_count, kind)?));
            }
        }
        Err(BarostatError::UnknownKind(target.to_string()))
    }
}

impl Default for BarostatRegistry {
    fn default() -> Self {
        BarostatRegistry::with_builtins()
    }
}

// =====================================================================
// Concrete integrators
// =====================================================================

// --- Velocity Verlet --------------------------------------------------
// rq-09a2e15f

#[derive(Debug)]
pub struct VelocityVerletState {
    lossless: Option<LosslessBuffers>,
}

impl Integrator for VelocityVerletState {
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

// --- Langevin BAOAB ---------------------------------------------------
// rq-d5a4f220

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

// =====================================================================
// Host-side Philox-4×32-10 RNG (shared utility for stochastic thermostats)
// =====================================================================
// rq-3d7c8e53
//
// Byte-for-byte equivalent to the device-side `philox4x32_10` in
// `kernels/langevin.cu`. Reusable from any host-side stochastic
// integrator that needs reproducible random draws (CSVR; future
// Andersen / stochastic barostat / etc.).

const PHILOX_M0: u32 = 0xD2511F53;
const PHILOX_M1: u32 = 0xCD9E8D57;
const PHILOX_W0: u32 = 0x9E3779B9;
const PHILOX_W1: u32 = 0xBB67AE85;

#[inline]
fn mulhi32(a: u32, b: u32) -> u32 {
    ((a as u64).wrapping_mul(b as u64) >> 32) as u32
}

/// Counter-based Philox-4×32-10. Inputs: 2-word key, 4-word counter.
/// Output: 4-word block. Pure function; matches the device-side helper
/// in `kernels/langevin.cu` byte-for-byte.
pub fn philox_4x32_10(
    key_lo: u32,
    key_hi: u32,
    ctr0: u32,
    ctr1: u32,
    ctr2: u32,
    ctr3: u32,
) -> [u32; 4] {
    let mut c0 = ctr0;
    let mut c1 = ctr1;
    let mut c2 = ctr2;
    let mut c3 = ctr3;
    let mut k0 = key_lo;
    let mut k1 = key_hi;
    for _ in 0..10 {
        let hi0 = mulhi32(c0, PHILOX_M0);
        let lo0 = c0.wrapping_mul(PHILOX_M0);
        let hi2 = mulhi32(c2, PHILOX_M1);
        let lo2 = c2.wrapping_mul(PHILOX_M1);
        let nc0 = hi2 ^ c1 ^ k0;
        let nc1 = lo2;
        let nc2 = hi0 ^ c3 ^ k1;
        let nc3 = lo0;
        c0 = nc0;
        c1 = nc1;
        c2 = nc2;
        c3 = nc3;
        k0 = k0.wrapping_add(PHILOX_W0);
        k1 = k1.wrapping_add(PHILOX_W1);
    }
    [c0, c1, c2, c3]
}

/// One standard-normal draw via Box-Muller (cos branch), matching the
/// device-side `philox_gaussian` formula exactly. Returns `f64` (the
/// device-side helper truncates to `f32` for its on-device use; CSVR
/// keeps the full `f64` because its chain math benefits from it).
pub fn philox_normal(
    key_lo: u32,
    key_hi: u32,
    ctr0: u32,
    ctr1: u32,
    ctr2: u32,
    ctr3: u32,
) -> f64 {
    let out = philox_4x32_10(key_lo, key_hi, ctr0, ctr1, ctr2, ctr3);
    let scale = 1.0_f64 / 4_294_967_296.0;
    let u1 = (out[0] as f64 + 0.5) * scale;
    let u2 = (out[1] as f64 + 0.5) * scale;
    let r = (-2.0_f64 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    r * theta.cos()
}

// =====================================================================
// Concrete thermostats
// =====================================================================

// --- Nosé-Hoover chain (NHC) -----------------------------------------
// rq-f606ff6f

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

fn yoshida_weights(n: u32) -> &'static [f64] {
    match n {
        1 => &YOSHIDA_1,
        3 => &YOSHIDA_3,
        5 => &YOSHIDA_5,
        7 => &YOSHIDA_7,
        _ => panic!("invalid yoshida_order {n}: must be 1, 3, 5, or 7"),
    }
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
    ke_scratch: CudaSlice<f32>,
    most_recent_ke: f64,
}

impl NoseHooverChainThermostat {
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

    fn chain_sub_step(
        &mut self,
        dt: f64,
        buffers: &mut ParticleBuffers,
        mut k: f64,
        timings: &mut Timings,
    ) -> Result<f64, ThermostatError> {
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
    ) -> Result<f64, ThermostatError> {
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

impl Thermostat for NoseHooverChainThermostat {
    // rq-2fe47a86
    fn apply_pre(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), ThermostatError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }
        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;
        self.thermostat_half_step(dt, buffers, k, timings)?;
        Ok(())
    }

    // rq-7a124d43
    fn apply_post(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), ThermostatError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }
        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k = self.thermostat_half_step(dt, buffers, k, timings)?;
        self.most_recent_ke = k;
        Ok(())
    }

    // rq-8a571737
    fn log_column_names(&self) -> &'static [&'static str] {
        &["nhc_conserved"]
    }

    // rq-f94f6bac
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

// rq-4bd6ff2b
#[derive(Debug)]
pub struct NoseHooverChainBuilder;

impl ThermostatBuilder for NoseHooverChainBuilder {
    fn kind_name(&self) -> &'static str {
        "nose-hoover-chain"
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        kind: &ThermostatKind,
    ) -> Result<Box<dyn Thermostat>, ThermostatError> {
        match kind {
            ThermostatKind::NoseHooverChain {
                temperature,
                tau,
                chain_length,
                yoshida_order,
                n_resp,
            } => {
                let state = NoseHooverChainThermostat::new(
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
            other => Err(ThermostatError::UnknownKind(other.name().to_string())),
        }
    }
}

// --- CSVR (Bussi-Donadio-Parrinello canonical sampling) --------------
// rq-891232bf

// rq-47d91c7d
#[derive(Debug)]
pub struct CsvrThermostat {
    pub temperature: f64,
    pub tau: f64,
    pub seed: u64,
    pub draw_counter: u64,
    pub g_dof: u32,
    pub kt_target: f64,
    pub cumulative_injection: f64,
    ke_scratch: CudaSlice<f32>,
    most_recent_ke: f64,
}

impl CsvrThermostat {
    fn new(
        gpu: &GpuContext,
        particle_count: usize,
        temperature: f64,
        tau: f64,
        seed: u64,
    ) -> Result<Self, GpuError> {
        let g_dof = ((3 * particle_count) as i64 - 3).max(1) as u32;
        let kt_target = BOLTZMANN_J_PER_K * temperature;
        let ke_scratch = gpu.device.alloc_zeros::<f32>(1).map_err(GpuError::from)?;
        Ok(CsvrThermostat {
            temperature,
            tau,
            seed,
            draw_counter: 0,
            g_dof,
            kt_target,
            cumulative_injection: 0.0,
            ke_scratch,
            most_recent_ke: 0.0,
        })
    }

    fn draw_new_kinetic_energy(&self, k_old: f64, dt: f32) -> f64 {
        let c = (-(dt as f64) / self.tau).exp();
        let nf = self.g_dof as f64;
        let k_target = (nf / 2.0) * self.kt_target;
        let one_minus_c = 1.0 - c;

        let seed_lo = self.seed as u32;
        let seed_hi = (self.seed >> 32) as u32;
        let ctr_lo = self.draw_counter as u32;
        let ctr_hi = (self.draw_counter >> 32) as u32;

        let r = philox_normal(seed_lo, seed_hi, ctr_lo, ctr_hi, 0, 0);
        let mut s = 0.0_f64;
        for sample_index in 1..self.g_dof {
            let xi = philox_normal(seed_lo, seed_hi, ctr_lo, ctr_hi, sample_index, 0);
            s += xi * xi;
        }

        let cross = if k_old > 0.0 {
            2.0 * r * (c * one_minus_c * k_old * k_target / nf).sqrt()
        } else {
            0.0
        };
        let k_new = c * k_old + (k_target / nf) * one_minus_c * (s + r * r) + cross;
        if k_new.is_finite() && k_new > 0.0 {
            k_new
        } else {
            k_old
        }
    }
}

impl Thermostat for CsvrThermostat {
    // rq-7a124d43
    fn apply_post(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), ThermostatError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }

        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k_old = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;

        self.draw_counter += 1;
        let k_new = self.draw_new_kinetic_energy(k_old, dt);
        self.cumulative_injection += k_new - k_old;
        self.most_recent_ke = k_new;

        if k_old > 0.0 && (k_new - k_old).abs() > 0.0 {
            let factor = (k_new / k_old).sqrt() as f32;
            timings.kernel_start(KernelStage::CSVR_RESCALE_VELOCITIES)?;
            rescale_velocities(buffers, factor)?;
            timings.kernel_stop(KernelStage::CSVR_RESCALE_VELOCITIES)?;
        }

        Ok(())
    }

    // rq-8ee58ec1
    fn log_column_names(&self) -> &'static [&'static str] {
        &["csvr_conserved"]
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
#[derive(Debug)]
pub struct CsvrBuilder;

impl ThermostatBuilder for CsvrBuilder {
    fn kind_name(&self) -> &'static str {
        "csvr"
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        kind: &ThermostatKind,
    ) -> Result<Box<dyn Thermostat>, ThermostatError> {
        match kind {
            ThermostatKind::Csvr {
                temperature,
                tau,
                seed,
            } => {
                let state = CsvrThermostat::new(gpu, particle_count, *temperature, *tau, *seed)?;
                Ok(Box::new(state))
            }
            other => Err(ThermostatError::UnknownKind(other.name().to_string())),
        }
    }
}

// --- Andersen stochastic thermostat ----------------------------------
// rq-5e059f6b

// rq-feba0a88
#[derive(Debug)]
pub struct AndersenThermostat {
    pub temperature: f64,
    pub collision_rate: f64,
    pub seed: u64,
    pub draw_counter: u64,
    pub kt: f64,
    pub cumulative_injection: f64,
    ke_scratch: CudaSlice<f32>,
    most_recent_ke: f64,
}

impl AndersenThermostat {
    fn new(
        gpu: &GpuContext,
        _particle_count: usize,
        temperature: f64,
        collision_rate: f64,
        seed: u64,
    ) -> Result<Self, GpuError> {
        let kt = BOLTZMANN_J_PER_K * temperature;
        let ke_scratch = gpu.device.alloc_zeros::<f32>(1).map_err(GpuError::from)?;
        Ok(AndersenThermostat {
            temperature,
            collision_rate,
            seed,
            draw_counter: 0,
            kt,
            cumulative_injection: 0.0,
            ke_scratch,
            most_recent_ke: 0.0,
        })
    }
}

impl Thermostat for AndersenThermostat {
    // rq-7a124d43
    fn apply_post(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), ThermostatError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }

        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k_old = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;

        self.draw_counter += 1;
        let p_collision = ((self.collision_rate as f64) * (dt as f64))
            .clamp(0.0, 1.0) as f32;
        let kt = self.kt as f32;
        timings.kernel_start(KernelStage::ANDERSEN_RESAMPLE)?;
        andersen_resample(buffers, self.seed, self.draw_counter, p_collision, kt)?;
        timings.kernel_stop(KernelStage::ANDERSEN_RESAMPLE)?;

        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k_new = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;

        self.cumulative_injection += k_new - k_old;
        self.most_recent_ke = k_new;
        Ok(())
    }

    // rq-1163481e
    fn log_column_names(&self) -> &'static [&'static str] {
        &["andersen_conserved"]
    }

    // rq-6d2daea0
    fn log_column_values(
        &self,
        kinetic_energy: f64,
        potential_energy: f64,
    ) -> Vec<f64> {
        vec![kinetic_energy + potential_energy - self.cumulative_injection]
    }
}

// rq-fd0cef60
#[derive(Debug)]
pub struct AndersenBuilder;

impl ThermostatBuilder for AndersenBuilder {
    fn kind_name(&self) -> &'static str {
        "andersen"
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        kind: &ThermostatKind,
    ) -> Result<Box<dyn Thermostat>, ThermostatError> {
        match kind {
            ThermostatKind::Andersen {
                temperature,
                collision_rate,
                seed,
            } => {
                let state = AndersenThermostat::new(
                    gpu,
                    particle_count,
                    *temperature,
                    *collision_rate,
                    *seed,
                )?;
                Ok(Box::new(state))
            }
            other => Err(ThermostatError::UnknownKind(other.name().to_string())),
        }
    }
}

// --- Berendsen weak-coupling thermostat -------------------------------
// rq-25f24b26

// rq-f856f666
#[derive(Debug)]
pub struct BerendsenThermostat {
    pub temperature: f64,
    pub tau: f64,
    pub g_dof: u32,
    pub kt_target: f64,
    pub cumulative_injection: f64,
    ke_scratch: CudaSlice<f32>,
    most_recent_ke: f64,
}

impl BerendsenThermostat {
    fn new(
        gpu: &GpuContext,
        particle_count: usize,
        temperature: f64,
        tau: f64,
    ) -> Result<Self, GpuError> {
        let g_dof = ((3 * particle_count) as i64 - 3).max(1) as u32;
        let kt_target = BOLTZMANN_J_PER_K * temperature;
        let ke_scratch = gpu.device.alloc_zeros::<f32>(1).map_err(GpuError::from)?;
        Ok(BerendsenThermostat {
            temperature,
            tau,
            g_dof,
            kt_target,
            cumulative_injection: 0.0,
            ke_scratch,
            most_recent_ke: 0.0,
        })
    }
}

impl Thermostat for BerendsenThermostat {
    // rq-7a124d43
    fn apply_post(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), ThermostatError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }

        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k_old = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;

        if k_old <= 0.0 {
            self.most_recent_ke = 0.0;
            return Ok(());
        }

        let nf = self.g_dof as f64;
        let k_target = (nf / 2.0) * self.kt_target;
        let lambda_sq = (1.0 + ((dt as f64) / self.tau) * (k_target / k_old - 1.0)).max(0.0);
        let lambda = lambda_sq.sqrt();
        let factor = lambda as f32;

        timings.kernel_start(KernelStage::BERENDSEN_RESCALE_VELOCITIES)?;
        rescale_velocities(buffers, factor)?;
        timings.kernel_stop(KernelStage::BERENDSEN_RESCALE_VELOCITIES)?;

        let k_new = lambda_sq * k_old;
        self.cumulative_injection += k_new - k_old;
        self.most_recent_ke = k_new;
        Ok(())
    }

    // rq-c908bbf1
    fn log_column_names(&self) -> &'static [&'static str] {
        &["berendsen_conserved"]
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
#[derive(Debug)]
pub struct BerendsenBuilder;

impl ThermostatBuilder for BerendsenBuilder {
    fn kind_name(&self) -> &'static str {
        "berendsen"
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        kind: &ThermostatKind,
    ) -> Result<Box<dyn Thermostat>, ThermostatError> {
        match kind {
            ThermostatKind::Berendsen { temperature, tau } => {
                let state = BerendsenThermostat::new(gpu, particle_count, *temperature, *tau)?;
                Ok(Box::new(state))
            }
            other => Err(ThermostatError::UnknownKind(other.name().to_string())),
        }
    }
}
