// rq-f606ff6f
//
// Nose-Hoover chain (NHC) thermostat tests. The thermostat is exercised
// in isolation through its `apply_pre` and `apply_post` hooks; the
// shared kinetic_energy_reduce / rescale_velocities helpers documented
// in `nose-hoover-chain.md` are also exercised directly.

use dynamics::forces::{AggregateLevel, AngleList, BondList, ExclusionList, ForceField, PotentialRegistry};
use dynamics::gpu::{
    GpuContext, ParticleBuffers, compute_kinetic_energy, init_device, rescale_velocities,
};
use dynamics::integrator::IntegratorStepExt;
use dynamics::integrator::{
    NoseHooverChainThermostat, Thermostat, ThermostatRegistry,
};
use dynamics::io::SlotConfig;
use dynamics::io::config::NeighborListConfig;
use dynamics::pbc::SimulationBox;
use dynamics::state::ParticleState;
use dynamics::timings::{KernelStage, Timings};

#[allow(dead_code)]
const KB: f64 = 1.380649e-23;
const LEN_F: f64 = 5.29177210903e-11;
const MASS_F: f64 = 9.1093837015e-31;
const TIME_F: f64 = 2.4188843265857195e-17;
const TEMP_F: f64 = 315775.0248040668;
const VEL_F: f64 = 2187691.2636411153;

fn small_state(n: usize, mass: f32) -> ParticleState {
    let pos: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let zero = vec![0.0_f32; n];
    ParticleState::new(
        pos,
        zero.clone(),
        zero.clone(),
        zero.clone(),
        zero.clone(),
        zero,
        vec![mass; n],
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap()
}

fn box_large() -> SimulationBox {
    let l = (1.0e6 / LEN_F) as f32;
    SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap()
}

fn empty_force_field(gpu: &GpuContext, n: usize) -> ForceField {
    ForceField::new(
        &PotentialRegistry::with_builtins(),
        gpu,
        n,
        &box_large(),
        &[],
        &[],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(n),
        &AngleList::empty(0),
        &ExclusionList::empty(n),
        &NeighborListConfig::AllPairs,
    )
    .unwrap()
}

fn nhc_kind(
    temperature: f64,
    tau: f64,
    chain_length: u32,
    yoshida_order: u32,
    n_resp: u32,
) -> SlotConfig {
    // Convert SI inputs (K, s) to atomic units.
    let temperature = temperature / TEMP_F;
    let tau = tau / TIME_F;
    SlotConfig::from_params_str(
        "nose-hoover-chain",
        &format!(
            "temperature = {temperature:e}\ntau = {tau:e}\nchain_length = {chain_length}\nyoshida_order = {yoshida_order}\nn_resp = {n_resp}\n"
        ),
    )
}

fn build_nhc(gpu: &GpuContext, n: usize, slot: &SlotConfig) -> Box<dyn Thermostat> {
    ThermostatRegistry::with_builtins()
        .build_optional(Some(slot), gpu, n, 0)
        .unwrap()
        .unwrap()
}

fn unbox_nhc(boxed: Box<dyn Thermostat>) -> NoseHooverChainThermostat {
    unsafe { *Box::from_raw(Box::into_raw(boxed) as *mut NoseHooverChainThermostat) }
}

// --- Construction ---

// rq-b43bf21c
#[test]
fn registry_builds_nhc_with_defaults() {
    let gpu = init_device().unwrap();
    let kind = nhc_kind(300.0, 1.0e-13, 3, 3, 1);
    let state = unbox_nhc(build_nhc(&gpu, 4, &kind));
    assert_eq!(state.chain_length, 3);
    assert_eq!(state.xi, vec![0.0, 0.0, 0.0]);
    assert_eq!(state.p_xi, vec![0.0, 0.0, 0.0]);
    // nhc_kind converts SI inputs; the engine stores atomic-unit values.
    let kt = 300.0 / TEMP_F;
    let tau2 = (1.0e-13_f64 / TIME_F).powi(2);
    let tol = (state.g_dof as f64) * kt * tau2 * 1.0e-14;
    assert!((state.q_mass[0] - (state.g_dof as f64) * kt * tau2).abs() < tol);
    assert!((state.q_mass[1] - kt * tau2).abs() < kt * tau2 * 1.0e-14);
    assert!((state.q_mass[2] - kt * tau2).abs() < kt * tau2 * 1.0e-14);
    assert_eq!(state.g_dof, 9);
}

// rq-12d7c3fe
#[test]
fn registry_builds_nhc_with_chain_length_1() {
    let gpu = init_device().unwrap();
    let kind = nhc_kind(300.0, 1.0e-13, 1, 3, 1);
    let state = unbox_nhc(build_nhc(&gpu, 4, &kind));
    assert_eq!(state.xi.len(), 1);
    assert_eq!(state.p_xi.len(), 1);
    assert_eq!(state.q_mass.len(), 1);
}

// rq-5f21bfd8
#[test]
fn registry_builds_nhc_with_particle_count_zero() {
    let gpu = init_device().unwrap();
    let kind = nhc_kind(300.0, 1.0e-13, 3, 3, 1);
    let state = unbox_nhc(build_nhc(&gpu, 0, &kind));
    assert_eq!(state.g_dof, 0);
}

// --- compute_kinetic_energy helper ---

// rq-25e0208d
#[test]
fn kinetic_energy_reduce_zero_velocity_returns_zero() {
    let gpu = init_device().unwrap();
    let state = small_state(4, 1.0);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut scratch = gpu.device.alloc_zeros::<f32>(1).unwrap();
    let ke = compute_kinetic_energy(&buffers, &mut scratch).unwrap();
    assert_eq!(ke, 0.0);
}

// rq-74e42489
#[test]
fn kinetic_energy_reduce_matches_host_formula() {
    let gpu = init_device().unwrap();
    let n = 8usize;
    let masses: Vec<f32> = (0..n).map(|i| 1.0 + 0.5 * i as f32).collect();
    let vx: Vec<f32> = (0..n).map(|i| 0.1 + i as f32).collect();
    let vy: Vec<f32> = (0..n).map(|i| -0.2 * i as f32).collect();
    let vz: Vec<f32> = (0..n).map(|i| 0.3 - i as f32).collect();
    let state = ParticleState::new(
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vx.clone(),
        vy.clone(),
        vz.clone(),
        masses.clone(),
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap();
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut scratch = gpu.device.alloc_zeros::<f32>(1).unwrap();
    let ke = compute_kinetic_energy(&buffers, &mut scratch).unwrap();
    let mut expected = 0.0_f32;
    for i in 0..n {
        expected += 0.5 * masses[i] * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    }
    let rel = (ke - expected).abs() / expected.abs();
    assert!(rel < 5.0e-5);
}

// rq-5c197a37
#[test]
fn kinetic_energy_reduce_is_deterministic() {
    let gpu = init_device().unwrap();
    let n = 1000usize;
    let masses: Vec<f32> = (0..n).map(|i| 1.0 + 0.001 * i as f32).collect();
    let vx: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
    let vy: Vec<f32> = (0..n).map(|i| (i as f32).cos()).collect();
    let vz: Vec<f32> = (0..n).map(|i| 0.5 - 0.001 * i as f32).collect();
    let state = ParticleState::new(
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vx,
        vy,
        vz,
        masses,
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap();
    let buffers_a = ParticleBuffers::new(&gpu, &state).unwrap();
    let buffers_b = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut scratch_a = gpu.device.alloc_zeros::<f32>(1).unwrap();
    let mut scratch_b = gpu.device.alloc_zeros::<f32>(1).unwrap();
    let ke_a = compute_kinetic_energy(&buffers_a, &mut scratch_a).unwrap();
    let ke_b = compute_kinetic_energy(&buffers_b, &mut scratch_b).unwrap();
    assert_eq!(ke_a.to_bits(), ke_b.to_bits());
}

// rq-96f71d13
#[test]
fn kinetic_energy_reduce_empty_state_returns_zero() {
    let gpu = init_device().unwrap();
    let state = small_state(0, 1.0);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut scratch = gpu.device.alloc_zeros::<f32>(1).unwrap();
    let ke = compute_kinetic_energy(&buffers, &mut scratch).unwrap();
    assert_eq!(ke, 0.0);
}

// --- rescale_velocities helper ---

// rq-6966fd4f
#[test]
fn rescale_velocities_multiplies_components_by_factor() {
    let gpu = init_device().unwrap();
    let state = ParticleState::new(
        vec![0.0_f32; 2],
        vec![0.0_f32; 2],
        vec![0.0_f32; 2],
        vec![1.0_f32, -4.0_f32],
        vec![2.0_f32, 5.0_f32],
        vec![3.0_f32, -6.0_f32],
        vec![1.0_f32; 2],
        vec![0.0_f32; 2],
        vec![0u32; 2],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    rescale_velocities(&mut buffers, 0.5).unwrap();
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vz = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
    assert_eq!(vx, vec![0.5_f32, -2.0_f32]);
    assert_eq!(vy, vec![1.0_f32, 2.5_f32]);
    assert_eq!(vz, vec![1.5_f32, -3.0_f32]);
}

// rq-393a7932
#[test]
fn rescale_velocities_factor_one_is_identity() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let vx: Vec<f32> = (0..n).map(|i| 0.5 - i as f32).collect();
    let vy: Vec<f32> = (0..n).map(|i| 1.0 + 0.3 * i as f32).collect();
    let vz: Vec<f32> = (0..n).map(|i| -0.2 * i as f32).collect();
    let state = ParticleState::new(
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vx.clone(),
        vy.clone(),
        vz.clone(),
        vec![1.0_f32; n],
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    rescale_velocities(&mut buffers, 1.0).unwrap();
    let rvx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    assert_eq!(rvx, vx);
}

// rq-bef900e1
#[test]
fn rescale_velocities_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = small_state(0, 1.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    rescale_velocities(&mut buffers, 0.5).unwrap();
}

// --- NHC slot per-step kernel sequence ---

// rq-76069102
#[test]
fn nhc_apply_pre_and_apply_post_launch_expected_kernels() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = build_nhc(&gpu, n, &nhc_kind(300.0, 1.0e-13, 3, 3, 1));
    therm
        .apply_pre(&mut buffers, (1.0e-15 / TIME_F) as f32, &mut timings)
        .unwrap();
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as f32, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    let count_for = |stage: KernelStage| -> u64 {
        report
            .stages
            .iter()
            .find(|r| r.name == stage.name())
            .map(|r| r.count)
            .unwrap_or(0)
    };
    // Two KE reductions per step (one per half-step).
    assert_eq!(count_for(KernelStage::KINETIC_ENERGY_REDUCE), 2);
    // Yoshida 3 × n_resp 1 × 2 halves = 6 rescale launches per step.
    assert_eq!(count_for(KernelStage::NHC_RESCALE_VELOCITIES), 6);
    // The thermostat does NOT launch VV kernels.
    assert_eq!(count_for(KernelStage::VV_KICK_DRIFT), 0);
    assert_eq!(count_for(KernelStage::VV_KICK), 0);
}

// rq-e9a5474f
#[test]
fn nhc_apply_pre_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = small_state(0, 1.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = build_nhc(&gpu, 0, &nhc_kind(300.0, 1.0e-13, 3, 3, 1));
    therm
        .apply_pre(&mut buffers, (1.0e-15 / TIME_F) as f32, &mut timings)
        .unwrap();
}

// rq-9b3e0e89
#[test]
fn nhc_apply_post_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = small_state(0, 1.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = build_nhc(&gpu, 0, &nhc_kind(300.0, 1.0e-13, 3, 3, 1));
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as f32, &mut timings)
        .unwrap();
}

// --- Log columns ---

// rq-17d3ddfe rq-a16c37bd
#[test]
fn nhc_log_column_names_returns_nhc_conserved() {
    let gpu = init_device().unwrap();
    let kind = nhc_kind(300.0, 1.0e-13, 3, 3, 1);
    let therm = build_nhc(&gpu, 4, &kind);
    let names: Vec<&str> = therm.log_column_names().iter().map(|(n, _)| *n).collect();
    assert_eq!(names, vec!["nhc_conserved"]);
}

// rq-7909b92c rq-ded81a4a
#[test]
fn vv_and_langevin_log_column_names_are_empty() {
    use dynamics::integrator::IntegratorRegistry;
    let gpu = init_device().unwrap();
    let vv = IntegratorRegistry::with_builtins()
        .build(
            &SlotConfig::from_params_str("velocity-verlet", "lossless = false"),
            &gpu,
            4, 0)
        .unwrap();
    assert!(vv.log_column_names().is_empty());
    let lan = IntegratorRegistry::with_builtins()
        .build(
            &SlotConfig::from_params_str(
                "langevin-baoab",
                "friction = 1.0e12\ntemperature = 300.0\nseed = 1\n",
            ),
            &gpu,
            4, 0)
        .unwrap();
    assert!(lan.log_column_names().is_empty());
}

// rq-07a18814
#[test]
fn nhc_log_column_values_combines_ke_pe_and_chain_term() {
    let gpu = init_device().unwrap();
    let mut s = unbox_nhc(build_nhc(&gpu, 4, &nhc_kind(300.0, 1.0e-13, 2, 3, 1)));
    // All values are in atomic units. k_B = 1 inside the engine.
    let kt = 300.0 / TEMP_F;
    let tau_au = 1.0e-13_f64 / TIME_F;
    let q1 = (s.g_dof as f64) * kt * tau_au.powi(2);
    let q2 = kt * tau_au.powi(2);
    s.xi[0] = 0.1;
    s.xi[1] = 0.2;
    s.p_xi[0] = 0.5e-30;
    s.p_xi[1] = -0.3e-30;
    let ke = 1.0e-20_f64;
    let pe = 2.0e-20_f64;
    let expected_chain = s.p_xi[0].powi(2) / (2.0 * q1)
        + s.p_xi[1].powi(2) / (2.0 * q2)
        + (s.g_dof as f64) * kt * s.xi[0]
        + kt * s.xi[1];
    let expected: f64 = ke + pe + expected_chain;
    let extras = s.log_column_values(ke, pe);
    assert_eq!(extras.len(), 1);
    let rel = (extras[0] - expected).abs() / expected.abs();
    assert!(rel < 1.0e-12);
}

// --- End-to-end determinism + COM conservation ---

fn atomic_state(n: usize) -> ParticleState {
    let mass: f32 = (1.66e-27 / MASS_F) as f32;
    let mut vx: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n / 2 {
        let v = (500.0 / VEL_F) as f32 * ((i as f32) + 1.0);
        vx.push(v);
        vx.push(-v);
    }
    if vx.len() < n {
        vx.push(0.0);
    }
    let zero = vec![0.0_f32; n];
    ParticleState::new(
        (0..n).map(|i| (i as f32) * (1.0e-10 / LEN_F) as f32).collect(),
        zero.clone(),
        zero.clone(),
        vx,
        zero.clone(),
        zero,
        vec![mass; n],
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap()
}

// rq-6faf6fba
#[test]
fn nhc_two_runs_with_identical_inputs_match() {
    let gpu = init_device().unwrap();
    let state = atomic_state(8);

    fn run_once(gpu: &GpuContext, state: &ParticleState) -> Vec<f32> {
        let n = state.particle_count();
        let mut buffers = ParticleBuffers::new(gpu, state).unwrap();
        let mut timings = Timings::new(gpu).unwrap();
        let mut therm = build_nhc(gpu, n, &nhc_kind(300.0, 1.0e-13, 3, 3, 1));
        for _ in 0..5 {
            therm
                .apply_pre(&mut buffers, (1.0e-15 / TIME_F) as f32, &mut timings)
                .unwrap();
            therm
                .apply_post(&mut buffers, (1.0e-15 / TIME_F) as f32, &mut timings)
                .unwrap();
        }
        gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap()
    }

    let a = run_once(&gpu, &state);
    let b = run_once(&gpu, &state);
    assert_eq!(a, b);
    for v in &a {
        assert!(v.is_finite());
    }
}

// rq-6a4016ac
#[test]
fn nhc_preserves_com_momentum_to_round_off() {
    let gpu = init_device().unwrap();
    let n = 16usize;
    let state = atomic_state(n);
    let mass = 1.66e-27_f32;
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integ = dynamics::integrator::IntegratorRegistry::with_builtins()
        .build(
            &SlotConfig::from_params_str("velocity-verlet", "lossless = false"),
            &gpu,
            n, 0)
        .unwrap();
    let mut therm = build_nhc(&gpu, n, &nhc_kind(300.0, 1.0e-13, 3, 3, 1));
    ff.step(&mut buffers, &sim_box, &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    for _ in 0..20 {
        therm
            .apply_pre(&mut buffers, (1.0e-15 / TIME_F) as f32, &mut timings)
            .unwrap();
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, (1.0e-15 / TIME_F) as f32, &mut timings)
            .unwrap();
        therm
            .apply_post(&mut buffers, (1.0e-15 / TIME_F) as f32, &mut timings)
            .unwrap();
    }
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let p_com: f64 = vx.iter().map(|&v| (mass as f64) * (v as f64)).sum();
    let scale: f32 = vx.iter().map(|v| v.abs()).fold(0.0, f32::max);
    let tol = (mass as f64) * (scale as f64) * 1.0e-3;
    assert!(
        p_com.abs() < tol,
        "p_com = {p_com} (tol {tol}), velocities = {vx:?}"
    );
}

// rq-572a0431
#[test]
fn init_device_exposes_nhc_kernels() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    assert!(device.has_func("nose_hoover", "rescale_velocities"));
    assert!(device.has_func("nose_hoover", "kinetic_energy_reduce"));
    let _ = gpu.kernels.nose_hoover.rescale_velocities.clone();
    let _ = gpu.kernels.nose_hoover.kinetic_energy_reduce.clone();
}

// rq-5c799ac6
#[test]
fn rescale_velocities_does_not_modify_positions_masses_or_forces() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let snap_px = state.positions_x.clone();
    let snap_py = state.positions_y.clone();
    let snap_pz = state.positions_z.clone();
    let snap_masses = state.masses.clone();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    rescale_velocities(&mut buffers, 0.5).unwrap();
    let px = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let py = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let pz = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    let masses = gpu.device.dtoh_sync_copy(&buffers.masses).unwrap();
    let fx = gpu.device.dtoh_sync_copy(&buffers.forces_x).unwrap();
    let fy = gpu.device.dtoh_sync_copy(&buffers.forces_y).unwrap();
    let fz = gpu.device.dtoh_sync_copy(&buffers.forces_z).unwrap();
    assert_eq!(px, snap_px);
    assert_eq!(py, snap_py);
    assert_eq!(pz, snap_pz);
    assert_eq!(masses, snap_masses);
    for v in fx.iter().chain(fy.iter()).chain(fz.iter()) {
        assert_eq!(*v, 0.0_f32, "forces should remain at their pre-rescale value (zero here)");
    }
}

// rq-1aa67999
#[test]
fn nhc_constructs_for_a_settled_water_system() {
    use dynamics::integrator::ThermostatRegistry;
    let gpu = init_device().unwrap();
    let kind = nhc_kind(300.0, 1.0e-13, 3, 3, 1);
    // 24 particles (8 waters) with 24 constraints; g_dof = 45.
    let therm = ThermostatRegistry::with_builtins()
        .build_optional(Some(&kind), &gpu, 24, 24)
        .unwrap()
        .unwrap();
    let state = unbox_nhc(therm);
    assert_eq!(state.g_dof, 45);
}
