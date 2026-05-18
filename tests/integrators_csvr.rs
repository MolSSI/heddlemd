// rq-891232bf
//
// CSVR (Bussi-Donadio-Parrinello) thermostat tests. The thermostat is
// exercised in isolation through its `apply_post` hook.

use dynamics::forces::{AngleList, BondList, ExclusionList, ForceField};
use dynamics::gpu::{
    GpuContext, ParticleBuffers, compute_kinetic_energy, init_device, lan_ou_step,
};
use dynamics::integrator::IntegratorStepExt;
use dynamics::integrator::{
    CsvrThermostat, Thermostat, ThermostatRegistry, philox_4x32_10, philox_normal,
};
use dynamics::io::SlotConfig;
use dynamics::io::config::NeighborListConfig;
use dynamics::pbc::SimulationBox;
use dynamics::state::ParticleState;
use dynamics::timings::{KernelStage, Timings};

const KB: f64 = 1.380649e-23;

fn box_large() -> SimulationBox {
    SimulationBox::new(1.0e6, 1.0e6, 1.0e6, 0.0, 0.0, 0.0).unwrap()
}

fn empty_force_field(gpu: &GpuContext, n: usize) -> ForceField {
    ForceField::new(
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

fn csvr_kind(temperature: f64, tau: f64, seed: u64) -> SlotConfig {
    SlotConfig::from_params_str(
        "csvr",
        &format!("temperature = {temperature:e}\ntau = {tau:e}\nseed = {seed}\n"),
    )
}

fn build_csvr(gpu: &GpuContext, n: usize, slot: &SlotConfig) -> Box<dyn Thermostat> {
    ThermostatRegistry::with_builtins()
        .build_optional(Some(slot), gpu, n)
        .unwrap()
        .unwrap()
}

fn unbox_csvr(boxed: Box<dyn Thermostat>) -> CsvrThermostat {
    unsafe { *Box::from_raw(Box::into_raw(boxed) as *mut CsvrThermostat) }
}

fn atomic_state(n: usize) -> ParticleState {
    let mass: f32 = 1.66e-27;
    let mut vx: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n / 2 {
        let v = 500.0 * ((i as f32) + 1.0);
        vx.push(v);
        vx.push(-v);
    }
    if vx.len() < n {
        vx.push(0.0);
    }
    let zero = vec![0.0_f32; n];
    ParticleState::new(
        (0..n).map(|i| (i as f32) * 1.0e-10).collect(),
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

// --- Construction ---

// rq-9e1142aa
#[test]
fn registry_builds_csvr() {
    let gpu = init_device().unwrap();
    let kind = csvr_kind(300.0, 1.0e-13, 42);
    let therm = unbox_csvr(build_csvr(&gpu, 4, &kind));
    assert_eq!(therm.draw_counter, 0);
    assert_eq!(therm.cumulative_injection, 0.0);
    assert_eq!(therm.g_dof, 9);
    assert!((therm.kt_target - KB * 300.0).abs() < 1.0e-30);
}

// rq-cf008c68
#[test]
fn registry_builds_csvr_particle_count_zero() {
    let gpu = init_device().unwrap();
    let kind = csvr_kind(300.0, 1.0e-13, 1);
    let _therm = build_csvr(&gpu, 0, &kind);
}

// rq-c4872e7e
#[test]
fn registry_builds_csvr_particle_count_one() {
    let gpu = init_device().unwrap();
    let kind = csvr_kind(300.0, 1.0e-13, 1);
    let therm = unbox_csvr(build_csvr(&gpu, 1, &kind));
    assert_eq!(therm.g_dof, 1);
}

// --- Host-side Philox parity with the device-side kernel ---

// rq-11a953dc
#[test]
fn host_philox_matches_device_philox() {
    let gpu = init_device().unwrap();
    let n = 1usize;
    let mass: f32 = 1.0;
    let kt: f32 = 1.0;
    let seed: u64 = 0x1234_5678_9ABC_DEF0;
    let draw: u64 = 7;
    let state = ParticleState::new(
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![mass; n],
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    lan_ou_step(&mut buffers, seed, draw, 0.0_f32, kt).unwrap();
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vz = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();

    let seed_lo = seed as u32;
    let seed_hi = (seed >> 32) as u32;
    let draw_lo = draw as u32;
    let draw_hi = (draw >> 32) as u32;
    let host_xi = |axis: u32| -> f32 {
        philox_normal(seed_lo, seed_hi, draw_lo, draw_hi, 0, axis) as f32
    };
    let sigma = (kt / mass).sqrt();
    let expected_vx = sigma * host_xi(0);
    let expected_vy = sigma * host_xi(1);
    let expected_vz = sigma * host_xi(2);
    assert_eq!(vx[0].to_bits(), expected_vx.to_bits());
    assert_eq!(vy[0].to_bits(), expected_vy.to_bits());
    assert_eq!(vz[0].to_bits(), expected_vz.to_bits());
}

// rq-db1298bd
#[test]
fn philox_is_pure_function() {
    let a = philox_4x32_10(1, 2, 3, 4, 5, 6);
    let b = philox_4x32_10(1, 2, 3, 4, 5, 6);
    assert_eq!(a, b);
    let c = philox_4x32_10(1, 2, 3, 4, 5, 7);
    assert_ne!(a, c);
}

// --- Per-step kernel sequence ---

// rq-4e9e09f0
#[test]
fn csvr_apply_post_launches_expected_kernels() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = build_csvr(&gpu, n, &csvr_kind(300.0, 1.0e-13, 1));
    therm
        .apply_post(&mut buffers, 1.0e-15, &mut timings)
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
    assert_eq!(count_for(KernelStage::KINETIC_ENERGY_REDUCE), 1);
    assert_eq!(count_for(KernelStage::CSVR_RESCALE_VELOCITIES), 1);
    assert_eq!(count_for(KernelStage::VV_KICK_DRIFT), 0);
    assert_eq!(count_for(KernelStage::VV_KICK), 0);
}

// rq-a2454a72
#[test]
fn csvr_apply_post_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = atomic_state(0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = build_csvr(&gpu, 0, &csvr_kind(300.0, 1.0e-13, 1));
    therm
        .apply_post(&mut buffers, 1.0e-15, &mut timings)
        .unwrap();
}

// rq-d1f1b53e
#[test]
fn csvr_apply_pre_is_trait_default_noop() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let snap_vx = state.velocities_x.clone();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = build_csvr(&gpu, n, &csvr_kind(300.0, 1.0e-13, 1));
    therm
        .apply_pre(&mut buffers, 1.0e-15, &mut timings)
        .unwrap();
    let vx_after = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    assert_eq!(vx_after, snap_vx);
    let report = timings.finalize().unwrap();
    for s in &report.stages {
        assert_eq!(s.count, 0, "apply_pre launched kernel {:?}", s.name);
    }
}

// --- draw_counter advances ---

// rq-1e5dcdc9
#[test]
fn csvr_draw_counter_increments_per_apply_post() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = unbox_csvr(build_csvr(&gpu, n, &csvr_kind(300.0, 1.0e-13, 1)));
    assert_eq!(therm.draw_counter, 0);
    therm
        .apply_post(&mut buffers, 1.0e-15, &mut timings)
        .unwrap();
    assert_eq!(therm.draw_counter, 1);
    therm
        .apply_post(&mut buffers, 1.0e-15, &mut timings)
        .unwrap();
    assert_eq!(therm.draw_counter, 2);
}

// rq-dc95802b
#[test]
fn csvr_two_thermostats_at_same_counter_produce_identical_velocities() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let mut buffers_a = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut buffers_b = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings_a = Timings::new(&gpu).unwrap();
    let mut timings_b = Timings::new(&gpu).unwrap();
    let mut therm_a = build_csvr(&gpu, n, &csvr_kind(300.0, 1.0e-13, 7));
    let mut therm_b = build_csvr(&gpu, n, &csvr_kind(300.0, 1.0e-13, 7));
    therm_a
        .apply_post(&mut buffers_a, 1.0e-15, &mut timings_a)
        .unwrap();
    therm_b
        .apply_post(&mut buffers_b, 1.0e-15, &mut timings_b)
        .unwrap();
    let va = gpu.device.dtoh_sync_copy(&buffers_a.velocities_x).unwrap();
    let vb = gpu.device.dtoh_sync_copy(&buffers_b.velocities_x).unwrap();
    assert_eq!(va, vb);
}

// --- Log columns ---

// rq-2c1bb918
#[test]
fn csvr_log_column_names_returns_csvr_conserved() {
    let gpu = init_device().unwrap();
    let kind = csvr_kind(300.0, 1.0e-13, 1);
    let therm = build_csvr(&gpu, 4, &kind);
    assert_eq!(therm.log_column_names(), &["csvr_conserved"]);
}

// rq-ca0b98cb
#[test]
fn csvr_log_column_values_subtracts_cumulative_injection() {
    let gpu = init_device().unwrap();
    let mut therm = unbox_csvr(build_csvr(&gpu, 4, &csvr_kind(300.0, 1.0e-13, 1)));
    therm.cumulative_injection = 1.0e-20;
    let extras = therm.log_column_values(2.5e-20, 3.0e-20);
    assert_eq!(extras.len(), 1);
    let expected: f64 = 2.5e-20 + 3.0e-20 - 1.0e-20;
    assert!((extras[0] - expected).abs() < 1.0e-30);
}

// --- Cumulative injection updates during apply_post ---

// rq-11b0deff
#[test]
fn csvr_cumulative_injection_tracks_kinetic_energy_changes() {
    let gpu = init_device().unwrap();
    let n = 8usize;
    let state = atomic_state(n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = unbox_csvr(build_csvr(&gpu, n, &csvr_kind(300.0, 1.0e-13, 1)));
    let mut scratch = gpu.device.alloc_zeros::<f32>(1).unwrap();
    let k_before = compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    therm
        .apply_post(&mut buffers, 1.0e-15, &mut timings)
        .unwrap();
    let k_after = compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    let expected = k_after - k_before;
    let rel = (therm.cumulative_injection - expected).abs() / expected.abs().max(1.0e-30);
    assert!(rel < 1.0e-4);
}

// --- End-to-end determinism + COM preservation ---

// rq-dc51e1c3
#[test]
fn csvr_two_runs_with_identical_inputs_match() {
    let gpu = init_device().unwrap();
    let state = atomic_state(8);

    fn run_once(gpu: &GpuContext, state: &ParticleState) -> Vec<f32> {
        let n = state.particle_count();
        let mut buffers = ParticleBuffers::new(gpu, state).unwrap();
        let mut timings = Timings::new(gpu).unwrap();
        let mut therm = build_csvr(gpu, n, &csvr_kind(300.0, 1.0e-13, 42));
        for _ in 0..5 {
            therm
                .apply_post(&mut buffers, 1.0e-15, &mut timings)
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

// rq-94a43204
#[test]
fn csvr_different_seeds_produce_different_trajectories() {
    let gpu = init_device().unwrap();
    let state = atomic_state(8);

    fn run_once(gpu: &GpuContext, state: &ParticleState, seed: u64) -> Vec<f32> {
        let n = state.particle_count();
        let mut buffers = ParticleBuffers::new(gpu, state).unwrap();
        let mut timings = Timings::new(gpu).unwrap();
        let mut therm = build_csvr(gpu, n, &csvr_kind(300.0, 1.0e-13, seed));
        for _ in 0..3 {
            therm
                .apply_post(&mut buffers, 1.0e-15, &mut timings)
                .unwrap();
        }
        gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap()
    }

    let a = run_once(&gpu, &state, 1);
    let b = run_once(&gpu, &state, 2);
    assert_ne!(a, b);
}

// rq-287e8d41
#[test]
fn csvr_preserves_com_momentum() {
    let gpu = init_device().unwrap();
    let n = 16usize;
    let state = atomic_state(n);
    let mass = 1.66e-27_f32;
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = build_csvr(&gpu, n, &csvr_kind(300.0, 1.0e-13, 7));
    for _ in 0..20 {
        therm
            .apply_post(&mut buffers, 1.0e-15, &mut timings)
            .unwrap();
    }
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let p_com: f64 = vx.iter().map(|&v| (mass as f64) * (v as f64)).sum();
    let scale: f32 = vx.iter().map(|v| v.abs()).fold(0.0, f32::max);
    let tol = (mass as f64) * (scale as f64) * 1.0e-3;
    assert!(p_com.abs() < tol, "p_com = {p_com} (tol {tol})");
}

// rq-f70f7c1e
#[test]
fn csvr_time_averaged_ke_tracks_k_target() {
    let gpu = init_device().unwrap();
    let n = 32usize;
    let mass: f32 = 1.66e-27;
    let temperature = 300.0_f64;
    let kt = KB * temperature;
    let g_dof = (3 * n - 3) as f64;
    let k_target = (g_dof / 2.0) * kt;
    let v_each = ((k_target / ((n as f64) * 0.5 * (mass as f64))) as f64).sqrt() as f32;
    let mut vx: Vec<f32> = Vec::with_capacity(n);
    for _ in 0..n / 2 {
        vx.push(v_each);
        vx.push(-v_each);
    }
    let zero = vec![0.0_f32; n];
    let state = ParticleState::new(
        (0..n).map(|i| (i as f32) * 1.0e-10).collect(),
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
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integ = dynamics::integrator::IntegratorRegistry::with_builtins()
        .build(
            &SlotConfig::from_params_str("velocity-verlet", "lossless = false"),
            &gpu,
            n,
        )
        .unwrap();
    let mut therm = build_csvr(&gpu, n, &csvr_kind(temperature, 1.0e-14, 11));
    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
    let mut scratch = gpu.device.alloc_zeros::<f32>(1).unwrap();
    for _ in 0..100 {
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, None, 1.0e-15, &mut timings)
            .unwrap();
        therm
            .apply_post(&mut buffers, 1.0e-15, &mut timings)
            .unwrap();
    }
    let mut sum = 0.0_f64;
    let n_samples = 250;
    for _ in 0..n_samples {
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, None, 1.0e-15, &mut timings)
            .unwrap();
        therm
            .apply_post(&mut buffers, 1.0e-15, &mut timings)
            .unwrap();
        sum += compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    }
    let k_avg = sum / (n_samples as f64);
    let rel = (k_avg - k_target).abs() / k_target;
    assert!(rel < 0.15, "k_avg = {k_avg:e}, target {k_target:e}, rel {rel}");
}
