// rq-5e059f6b
//
// Andersen thermostat tests. The thermostat is exercised in isolation
// through its `apply_post` hook; the `andersen_resample` kernel is
// also exercised directly.

use heddle_md::forces::{AggregateLevel, AngleList, BondList, ExclusionList, ForceField, PotentialRegistry};
use heddle_md::gpu::{
    GpuContext, ParticleBuffers, andersen_resample, compute_kinetic_energy, init_device,
};
use heddle_md::integrator::IntegratorStepExt;
use heddle_md::integrator::{
    AndersenThermostat, Thermostat, ThermostatRegistry,
};
use heddle_md::io::SlotConfig;
use heddle_md::io::config::NeighborListConfig;
use heddle_md::pbc::SimulationBox;
use heddle_md::state::ParticleState;
use heddle_md::timings::{KernelStage, Timings};
use heddle_md::precision::Real;

#[allow(dead_code)]
const KB: f64 = 1.380649e-23;
const LEN_F: f64 = 5.29177210903e-11;
const MASS_F: f64 = 9.1093837015e-31;
const TIME_F: f64 = 2.4188843265857195e-17;
const TEMP_F: f64 = 315775.0248040668;
const VEL_F: f64 = 2187691.2636411153;

fn box_large(gpu: &heddle_md::gpu::GpuContext) -> SimulationBox {
    let l = (1.0e6 / LEN_F) as Real;
    SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap()
}

fn empty_force_field(gpu: &GpuContext, n: usize) -> ForceField {
    ForceField::new(
        &PotentialRegistry::with_builtins(),
        gpu,
        n,
        &box_large(&gpu),
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

fn andersen_kind(temperature: f64, collision_rate: f64, seed: u64) -> SlotConfig {
    // Convert SI inputs (K, 1/s) to atomic units.
    let temperature = temperature / TEMP_F;
    let collision_rate = collision_rate * TIME_F;
    SlotConfig::from_params_str(
        "andersen",
        &format!(
            "temperature = {temperature:e}\ncollision_rate = {collision_rate:e}\nseed = {seed}\n"
        ),
    )
}

fn build_andersen(gpu: &GpuContext, n: usize, slot: &SlotConfig) -> Box<dyn Thermostat> {
    ThermostatRegistry::with_builtins()
        .build_optional(Some(slot), gpu, n, 0)
        .unwrap()
        .unwrap()
}

fn unbox_andersen(boxed: Box<dyn Thermostat>) -> AndersenThermostat {
    unsafe { *Box::from_raw(Box::into_raw(boxed) as *mut AndersenThermostat) }
}

fn atomic_state(n: usize) -> ParticleState {
    let mass: Real = (1.66e-27 / MASS_F) as Real;
    let mut vx: Vec<Real> = Vec::with_capacity(n);
    for i in 0..n / 2 {
        let v = (500.0 / VEL_F) as Real * ((i as Real) + 1.0);
        vx.push(v);
        vx.push(-v);
    }
    if vx.len() < n {
        vx.push(0.0);
    }
    let zero = vec![0.0; n];
    ParticleState::new(
        (0..n).map(|i| (i as Real) * (1.0e-10 / LEN_F) as Real).collect(),
        zero.clone(),
        zero.clone(),
        vx,
        zero.clone(),
        zero,
        vec![mass; n],
        vec![0.0; n],
        (0..n as u32).collect(),
        None,
        None,
    )
    .unwrap()
}

// --- Construction ---

// rq-dc3c616a rq-3396b95f
#[test]
fn registry_builds_andersen() {
    let gpu = init_device().unwrap();
    let kind = andersen_kind(300.0, 1.0e12, 42);
    let therm = unbox_andersen(build_andersen(&gpu, 4, &kind));
    assert_eq!(therm.draw_counter, 0);
    assert_eq!(therm.cumulative_injection, 0.0);
    // andersen_kind converts SI; the engine stores kt = temperature (k_B = 1).
    assert!((therm.kt - 300.0 / TEMP_F).abs() < 1.0e-30);
}

// rq-abcae430
#[test]
fn registry_builds_andersen_particle_count_zero() {
    let gpu = init_device().unwrap();
    let kind = andersen_kind(300.0, 1.0e12, 1);
    let _therm = build_andersen(&gpu, 0, &kind);
}

// rq-62ad70f8
#[test]
fn registry_builds_andersen_collision_rate_zero() {
    let gpu = init_device().unwrap();
    let kind = andersen_kind(300.0, 0.0, 1);
    let _therm = build_andersen(&gpu, 4, &kind);
}

// --- andersen_resample kernel ---

// rq-6ac1c0f2
#[test]
fn andersen_resample_p_zero_is_identity() {
    let gpu = init_device().unwrap();
    let n = 8usize;
    let state = atomic_state(n);
    let snap_vx: Vec<Real> = state.velocities_x.clone();
    let snap_vy: Vec<Real> = state.velocities_y.clone();
    let snap_vz: Vec<Real> = state.velocities_z.clone();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    andersen_resample(&mut buffers, 1, 1, 0.0, ((300.0 / TEMP_F)) as Real).unwrap();
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vz = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
    assert_eq!(vx, snap_vx);
    assert_eq!(vy, snap_vy);
    assert_eq!(vz, snap_vz);
}

// rq-4254b707 rq-299112e9
#[test]
fn andersen_resample_p_one_replaces_every_particle() {
    let gpu = init_device().unwrap();
    let n = 1024usize;
    let mass: Real = 1.66e-27;
    let vx_init = vec![1000.0; n];
    let vy_init = vec![1000.0; n];
    let vz_init = vec![1000.0; n];
    let state = ParticleState::new(
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vx_init.clone(),
        vy_init.clone(),
        vz_init.clone(),
        vec![mass; n],
        vec![0.0; n],
        (0..n as u32).collect(),
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let kt = (300.0 / TEMP_F);
    andersen_resample(&mut buffers, 42, 1, 1.0, kt as Real).unwrap();
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vz = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
    for i in 0..n {
        assert_ne!(vx[i], vx_init[i], "vx[{i}] unchanged");
        assert_ne!(vy[i], vy_init[i], "vy[{i}] unchanged");
        assert_ne!(vz[i], vz_init[i], "vz[{i}] unchanged");
    }
    let sigma2_target = (kt / mass as f64) as f64;
    for (label, comp) in [("vx", &vx), ("vy", &vy), ("vz", &vz)] {
        let mean: f64 = comp.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var: f64 = comp.iter().map(|&v| ((v as f64) - mean).powi(2)).sum::<f64>() / n as f64;
        let rel = (var - sigma2_target).abs() / sigma2_target;
        assert!(
            rel < 0.1,
            "{label} variance {var:e} vs expected {sigma2_target:e} (rel {rel})"
        );
    }
}

// rq-5ac172fc
#[test]
fn andersen_resample_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = atomic_state(0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    andersen_resample(&mut buffers, 1, 1, 0.5, 1.0).unwrap();
}

// rq-bacbf7d2
#[test]
fn andersen_resample_deterministic_across_runs() {
    let gpu = init_device().unwrap();
    let n = 64usize;
    let state = atomic_state(n);
    let mut a = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut b = ParticleBuffers::new(&gpu, &state).unwrap();
    let kt = ((300.0 / TEMP_F)) as Real;
    andersen_resample(&mut a, 7, 3, 1.0, kt).unwrap();
    andersen_resample(&mut b, 7, 3, 1.0, kt).unwrap();
    let va = gpu.device.dtoh_sync_copy(&a.velocities_x).unwrap();
    let vb = gpu.device.dtoh_sync_copy(&b.velocities_x).unwrap();
    assert_eq!(va, vb);
}

// rq-c3564e8a
#[test]
fn andersen_resample_different_seeds_differ() {
    let gpu = init_device().unwrap();
    let n = 64usize;
    let state = atomic_state(n);
    let mut a = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut b = ParticleBuffers::new(&gpu, &state).unwrap();
    let kt = ((300.0 / TEMP_F)) as Real;
    andersen_resample(&mut a, 1, 1, 1.0, kt).unwrap();
    andersen_resample(&mut b, 2, 1, 1.0, kt).unwrap();
    let va = gpu.device.dtoh_sync_copy(&a.velocities_x).unwrap();
    let vb = gpu.device.dtoh_sync_copy(&b.velocities_x).unwrap();
    let differs = va.iter().zip(vb.iter()).filter(|(x, y)| x != y).count();
    assert!(differs as f64 / n as f64 > 0.9);
}

// rq-8040ce8a
#[test]
fn andersen_resample_different_draw_counters_differ() {
    let gpu = init_device().unwrap();
    let n = 64usize;
    let state = atomic_state(n);
    let mut a = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut b = ParticleBuffers::new(&gpu, &state).unwrap();
    let kt = ((300.0 / TEMP_F)) as Real;
    andersen_resample(&mut a, 1, 1, 1.0, kt).unwrap();
    andersen_resample(&mut b, 1, 2, 1.0, kt).unwrap();
    let va = gpu.device.dtoh_sync_copy(&a.velocities_x).unwrap();
    let vb = gpu.device.dtoh_sync_copy(&b.velocities_x).unwrap();
    let differs = va.iter().zip(vb.iter()).filter(|(x, y)| x != y).count();
    assert!(differs as f64 / n as f64 > 0.9);
}

// --- Per-step kernel sequence ---

// rq-cef43ff0
#[test]
fn andersen_apply_post_launches_expected_kernels() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = build_andersen(&gpu, n, &andersen_kind(300.0, 1.0e12, 1));
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
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
    assert_eq!(count_for(KernelStage::KINETIC_ENERGY_REDUCE), 2);
    assert_eq!(count_for(KernelStage::ANDERSEN_RESAMPLE), 1);
    assert_eq!(count_for(KernelStage::VV_KICK_DRIFT), 0);
    assert_eq!(count_for(KernelStage::VV_KICK), 0);
}

// rq-8fdfc981 rq-e60481e9
#[test]
fn andersen_apply_post_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = atomic_state(0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = build_andersen(&gpu, 0, &andersen_kind(300.0, 1.0e12, 1));
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
        .unwrap();
}

// rq-15e44a1b rq-167867a2
#[test]
fn andersen_apply_pre_is_trait_default_noop() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let snap_vx = state.velocities_x.clone();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = build_andersen(&gpu, n, &andersen_kind(300.0, 1.0e12, 1));
    therm
        .apply_pre(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
        .unwrap();
    let vx_after = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    assert_eq!(vx_after, snap_vx);
    let report = timings.finalize().unwrap();
    for s in &report.stages {
        assert_eq!(s.count, 0);
    }
}

// --- draw_counter and cumulative_injection ---

// rq-c814659f
#[test]
fn andersen_draw_counter_increments_per_apply_post() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = unbox_andersen(build_andersen(&gpu, n, &andersen_kind(300.0, 1.0e12, 1)));
    assert_eq!(therm.draw_counter, 0);
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
        .unwrap();
    assert_eq!(therm.draw_counter, 1);
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
        .unwrap();
    assert_eq!(therm.draw_counter, 2);
}

// rq-b1e87ce4
#[test]
fn andersen_cumulative_injection_tracks_ke_change() {
    let gpu = init_device().unwrap();
    let n = 32usize;
    let state = atomic_state(n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = unbox_andersen(build_andersen(&gpu, n, &andersen_kind(300.0, 1.0e16, 1)));
    let mut scratch = gpu.device.alloc_zeros::<Real>(1).unwrap();
    let k_before = compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
        .unwrap();
    let k_after = compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    let expected = k_after - k_before;
    let rel = (therm.cumulative_injection - expected).abs() / expected.abs().max(1.0e-30);
    assert!(rel < 1.0e-4);
}

// --- Log columns ---

// rq-8eb14902 rq-c50c6f84
#[test]
fn andersen_log_column_names_returns_andersen_conserved() {
    let gpu = init_device().unwrap();
    let kind = andersen_kind(300.0, 1.0e12, 1);
    let therm = build_andersen(&gpu, 4, &kind);
    let names: Vec<&str> = therm.log_column_names().iter().map(|(n, _)| *n).collect();
    assert_eq!(names, vec!["andersen_conserved"]);
}

// rq-26ff4aea
#[test]
fn andersen_log_column_values_subtracts_cumulative_injection() {
    let gpu = init_device().unwrap();
    let mut therm = unbox_andersen(build_andersen(&gpu, 4, &andersen_kind(300.0, 1.0e12, 1)));
    therm.cumulative_injection = 1.0e-20;
    let extras = therm.log_column_values(2.5e-20, 3.0e-20);
    assert_eq!(extras.len(), 1);
    let expected: f64 = 2.5e-20 + 3.0e-20 - 1.0e-20;
    assert!((extras[0] - expected).abs() < 1.0e-30);
}

// --- Massive Andersen (p clamped to 1) ---

// rq-c9865e4c
#[test]
fn andersen_collision_rate_above_one_clamped_to_full_resample() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let snap_vx = state.velocities_x.clone();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    // collision_rate · dt = 10
    let mut therm = build_andersen(&gpu, n, &andersen_kind(300.0, 1.0e16, 1));
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
        .unwrap();
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    for (i, (a, b)) in vx.iter().zip(snap_vx.iter()).enumerate() {
        assert_ne!(a, b, "vx[{i}] unchanged ({a})");
    }
}

// rq-1eaff437
#[test]
fn andersen_collision_rate_zero_leaves_velocities_unchanged() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let snap_vx = state.velocities_x.clone();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = build_andersen(&gpu, n, &andersen_kind(300.0, 0.0, 1));
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
        .unwrap();
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    assert_eq!(vx, snap_vx);
}

// --- Determinism + temperature tracking ---

// rq-f062ec85
#[test]
fn andersen_two_runs_with_identical_inputs_match() {
    let gpu = init_device().unwrap();
    let state = atomic_state(8);

    fn run_once(gpu: &GpuContext, state: &ParticleState) -> Vec<Real> {
        let n = state.particle_count();
        let mut buffers = ParticleBuffers::new(gpu, state).unwrap();
        let mut timings = Timings::new(gpu).unwrap();
        let mut therm = build_andersen(gpu, n, &andersen_kind(300.0, 1.0e12, 42));
        for _ in 0..5 {
            therm
                .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
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

// rq-20daf925
#[test]
fn andersen_different_seeds_diverge() {
    let gpu = init_device().unwrap();
    let state = atomic_state(8);

    fn run_once(gpu: &GpuContext, state: &ParticleState, seed: u64) -> Vec<Real> {
        let n = state.particle_count();
        let mut buffers = ParticleBuffers::new(gpu, state).unwrap();
        let mut timings = Timings::new(gpu).unwrap();
        let mut therm = build_andersen(gpu, n, &andersen_kind(300.0, 1.0e16, seed));
        for _ in 0..3 {
            therm
                .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
                .unwrap();
        }
        gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap()
    }

    let a = run_once(&gpu, &state, 1);
    let b = run_once(&gpu, &state, 2);
    assert_ne!(a, b);
}

// rq-536457be
#[test]
fn andersen_time_averaged_ke_tracks_target() {
    let gpu = init_device().unwrap();
    let n = 64usize;
    // Atomic-unit values; k_B = 1.
    let mass: Real = (1.66e-27 / MASS_F) as Real;
    let temperature = 300.0_f64;
    let kt = temperature / TEMP_F;
    let k_target = (3.0 * n as f64 / 2.0) * kt;
    let zero = vec![0.0; n];
    let state = ParticleState::new(
        (0..n).map(|i| (i as Real) * (1.0e-10 / LEN_F) as Real).collect(),
        zero.clone(),
        zero.clone(),
        vec![(100.0 / VEL_F) as Real; n],
        zero.clone(),
        vec![0.0; n],
        vec![mass; n],
        vec![0.0; n],
        (0..n as u32).collect(),
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large(&gpu);
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integ = heddle_md::integrator::IntegratorRegistry::with_builtins()
        .build(
            &SlotConfig::from_params_str("velocity-verlet", "lossless = false"),
            &gpu,
            n, 0)
        .unwrap();
    let mut therm = build_andersen(&gpu, n, &andersen_kind(temperature, 5.0e14, 11));
    ff.step(&mut buffers, &sim_box, &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let mut scratch = gpu.device.alloc_zeros::<Real>(1).unwrap();
    for _ in 0..200 {
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, (1.0e-15 / TIME_F) as Real, &mut timings)
            .unwrap();
        therm
            .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
            .unwrap();
    }
    let mut sum = 0.0_f64;
    let n_samples = 500;
    for _ in 0..n_samples {
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, (1.0e-15 / TIME_F) as Real, &mut timings)
            .unwrap();
        therm
            .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
            .unwrap();
        sum += compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    }
    let k_avg = sum / n_samples as f64;
    let rel = (k_avg - k_target).abs() / k_target;
    assert!(rel < 0.15, "k_avg = {k_avg:e}, target {k_target:e}, rel {rel}");
}
