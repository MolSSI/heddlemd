// rq-25f24b26
//
// Berendsen weak-coupling thermostat tests. The thermostat is exercised
// in isolation through its `apply_post` hook (and `apply_pre` for the
// trait-default no-op scenario); composition with velocity-Verlet is
// covered by the integrator framework tests in
// `tests/integrator_framework.rs`.

use dynamics::forces::{AngleList, BondList, ExclusionList, ForceField, PotentialRegistry};
use dynamics::gpu::{
    GpuContext, ParticleBuffers, compute_kinetic_energy, init_device,
};
use dynamics::integrator::IntegratorStepExt;
use dynamics::integrator::{
    BerendsenThermostat, Thermostat, ThermostatRegistry,
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

fn berendsen_kind(temperature: f64, tau: f64) -> SlotConfig {
    // Convert SI inputs (K, s) to atomic units.
    let temperature = temperature / TEMP_F;
    let tau = tau / TIME_F;
    SlotConfig::from_params_str(
        "berendsen",
        &format!("temperature = {temperature:e}\ntau = {tau:e}\n"),
    )
}

fn build_berendsen(gpu: &GpuContext, n: usize, slot: &SlotConfig) -> Box<dyn Thermostat> {
    ThermostatRegistry::with_builtins()
        .build_optional(Some(slot), gpu, n, 0)
        .unwrap()
        .unwrap()
}

// `Box::into_raw` on a `Box<dyn Thermostat>` returns a fat pointer; the
// data pointer aliases the underlying `BerendsenThermostat` allocation,
// so the cast is safe as long as the registry actually built a Berendsen
// thermostat.
fn unbox_berendsen(boxed: Box<dyn Thermostat>) -> BerendsenThermostat {
    unsafe { *Box::from_raw(Box::into_raw(boxed) as *mut BerendsenThermostat) }
}

fn symmetric_state(n: usize, mass_si: f32, v_mag_si: f32) -> ParticleState {
    assert!(n.is_multiple_of(2));
    // Convert SI inputs (kg, m/s) to atomic units (m_e, Bohr/au-time).
    let mass = (mass_si as f64 / MASS_F) as f32;
    let v_mag = (v_mag_si as f64 / VEL_F) as f32;
    let mut vx: Vec<f32> = Vec::with_capacity(n);
    for _ in 0..n / 2 {
        vx.push(v_mag);
        vx.push(-v_mag);
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
        (0..n as u32).collect(),
        None,
        None,
    )
    .unwrap()
}

// --- Construction ---

// rq-e8252f10
#[test]
fn registry_builds_berendsen() {
    let gpu = init_device().unwrap();
    let kind = berendsen_kind(300.0, 1.0e-13);
    let _therm = build_berendsen(&gpu, 4, &kind);
}

// rq-e3a8d87c
#[test]
fn registry_builds_berendsen_particle_count_zero() {
    let gpu = init_device().unwrap();
    let kind = berendsen_kind(300.0, 1.0e-13);
    let _therm = build_berendsen(&gpu, 0, &kind);
}

// rq-470019c7
#[test]
fn berendsen_state_precomputes_g_dof_one_for_single_particle() {
    let gpu = init_device().unwrap();
    let kind = berendsen_kind(300.0, 1.0e-13);
    let state = unbox_berendsen(build_berendsen(&gpu, 1, &kind));
    assert_eq!(state.g_dof, 1);
}

#[test]
fn berendsen_state_precomputes_g_dof_and_kt_target() {
    let gpu = init_device().unwrap();
    let kind = berendsen_kind(300.0, 1.0e-13);
    let state = unbox_berendsen(build_berendsen(&gpu, 4, &kind));
    assert_eq!(state.g_dof, 9); // max(1, 3*4 - 3)
    let expected_kt = 300.0 / TEMP_F;
    assert!((state.kt_target - expected_kt).abs() < 1.0e-30);
    assert_eq!(state.cumulative_injection, 0.0);
}

// --- Per-step kernel sequence ---

// rq-0e85eebc
#[test]
fn berendsen_apply_post_launches_expected_kernels() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = symmetric_state(n, 1.66e-27, 500.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let kind = berendsen_kind(300.0, 1.0e-13);
    let mut therm = build_berendsen(&gpu, n, &kind);
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
    assert_eq!(count_for(KernelStage::KINETIC_ENERGY_REDUCE), 1);
    assert_eq!(count_for(KernelStage::BERENDSEN_RESCALE_VELOCITIES), 1);
    // The thermostat does NOT launch the VV kernels (those belong to the
    // integrator).
    assert_eq!(count_for(KernelStage::VV_KICK_DRIFT), 0);
    assert_eq!(count_for(KernelStage::VV_KICK), 0);
}

// rq-b296ab12
#[test]
fn berendsen_apply_post_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = symmetric_state(0, 1.66e-27, 0.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let kind = berendsen_kind(300.0, 1.0e-13);
    let mut therm = build_berendsen(&gpu, 0, &kind);
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as f32, &mut timings)
        .unwrap();
}

// rq-92fc3091
#[test]
fn berendsen_apply_pre_is_trait_default_noop() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = symmetric_state(n, 1.66e-27, 500.0);
    let snap_vx = state.velocities_x.clone();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let kind = berendsen_kind(300.0, 1.0e-13);
    let mut therm = build_berendsen(&gpu, n, &kind);
    therm
        .apply_pre(&mut buffers, (1.0e-15 / TIME_F) as f32, &mut timings)
        .unwrap();
    let vx_after = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    assert_eq!(vx_after, snap_vx);
    let report = timings.finalize().unwrap();
    for s in &report.stages {
        assert_eq!(s.count, 0, "apply_pre launched kernel {:?}", s.name);
    }
}

// --- λ formula correctness ---

// rq-1c1022f0
#[test]
fn berendsen_lambda_one_when_k_equals_target() {
    let gpu = init_device().unwrap();
    let n = 8usize;
    let mass_si: f32 = 1.66e-27;
    let temperature = 300.0_f64;
    let g_dof = (3 * n - 3) as f64;
    // k_B = 1 in the engine; the helper supplies SI temperature so divide
    // by TEMP_F to get the atomic-unit kT value (k_B · T in Hartrees).
    let k_target = (g_dof / 2.0) * (temperature / TEMP_F);
    let mass_au = mass_si as f64 / MASS_F;
    let v_each_au = (k_target / ((n as f64) * 0.5 * mass_au)).sqrt();
    // symmetric_state itself converts SI → atomic. Supply the
    // SI-equivalent velocity that matches the atomic v_each.
    let v_each_si = (v_each_au * VEL_F) as f32;
    let state = symmetric_state(n, mass_si, v_each_si);
    let snap_vx = state.velocities_x.clone();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = build_berendsen(&gpu, n, &berendsen_kind(temperature, 1.0e-13));
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as f32, &mut timings)
        .unwrap();
    let vx_after = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    for (a, b) in vx_after.iter().zip(snap_vx.iter()) {
        let rel = (a - b).abs() / b.abs().max(1.0);
        assert!(rel < 1.0e-4, "vx after = {a}, before = {b}");
    }
}

// rq-c6adba60 / rq-b6c8867f
#[test]
fn berendsen_lambda_squared_matches_analytical_when_cooling() {
    let gpu = init_device().unwrap();
    let n = 8usize;
    let mass_si: f32 = 1.66e-27;
    let temperature = 300.0_f64;
    let g_dof = (3 * n - 3) as f64;
    let k_target = (g_dof / 2.0) * (temperature / TEMP_F);
    let k_old_desired = 2.0 * k_target;
    let mass_au = mass_si as f64 / MASS_F;
    let v_each_au = (k_old_desired / ((n as f64) * 0.5 * mass_au)).sqrt();
    let v_each_si = (v_each_au * VEL_F) as f32;
    let state = symmetric_state(n, mass_si, v_each_si);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let dt = (1.0e-14 / TIME_F) as f32;
    let tau = 1.0e-13_f64;
    let mut therm = build_berendsen(&gpu, n, &berendsen_kind(temperature, tau));
    let mut scratch = gpu.device.alloc_zeros::<f32>(1).unwrap();
    let k_before = compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    therm.apply_post(&mut buffers, dt, &mut timings).unwrap();
    let k_after = compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    // dt is in atomic time; tau supplied as SI seconds → convert.
    let expected_lambda_sq =
        1.0 + ((dt as f64) / (tau / TIME_F)) * (k_target / k_before - 1.0);
    let expected_k_after = expected_lambda_sq * k_before;
    let rel = (k_after - expected_k_after).abs() / expected_k_after.abs();
    assert!(
        rel < 1.0e-4,
        "k_after = {k_after}, expected ≈ {expected_k_after} (λ² = {expected_lambda_sq})"
    );
}

// rq-4fe2658a
#[test]
fn berendsen_lambda_squared_clamped_to_zero_when_runaway() {
    let gpu = init_device().unwrap();
    let n = 8usize;
    let mass_si: f32 = 1.66e-27;
    let temperature = 300.0_f64;
    let g_dof = (3 * n - 3) as f64;
    let k_target = (g_dof / 2.0) * (temperature / TEMP_F);
    let mass_au = mass_si as f64 / MASS_F;
    let v_each_au = (100.0 * k_target / ((n as f64) * 0.5 * mass_au)).sqrt();
    let v_each_si = (v_each_au * VEL_F) as f32;
    let state = symmetric_state(n, mass_si, v_each_si);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let dt = (2.0e-13 / TIME_F) as f32; // dt/τ = 2.0 in SI
    let tau = 1.0e-13_f64;
    let mut therm = build_berendsen(&gpu, n, &berendsen_kind(temperature, tau));
    therm.apply_post(&mut buffers, dt, &mut timings).unwrap();
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    for v in &vx {
        assert_eq!(*v, 0.0_f32, "all velocities should be quenched to zero, got {v}");
    }
}

// rq-7d9f2da7
#[test]
fn berendsen_skips_rescale_when_k_zero() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = symmetric_state(n, 1.66e-27, 0.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let boxed = build_berendsen(&gpu, n, &berendsen_kind(300.0, 1.0e-13));
    // Apply once with the trait object so apply_post runs on the box.
    let mut therm = boxed;
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as f32, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    let count = report
        .stages
        .iter()
        .find(|r| r.name == KernelStage::BERENDSEN_RESCALE_VELOCITIES.name())
        .map(|r| r.count)
        .unwrap_or(0);
    assert_eq!(count, 0);
}

// --- COM-momentum preservation ---

// rq-5541d869
#[test]
fn berendsen_preserves_com_momentum() {
    let gpu = init_device().unwrap();
    let n = 16usize;
    let mass: f32 = 1.66e-27;
    let state = symmetric_state(n, mass, 500.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = build_berendsen(&gpu, n, &berendsen_kind(300.0, 1.0e-13));
    for _ in 0..20 {
        therm
            .apply_post(&mut buffers, (1.0e-15 / TIME_F) as f32, &mut timings)
            .unwrap();
    }
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let p_com: f64 = vx.iter().map(|&v| (mass as f64) * (v as f64)).sum();
    let scale: f32 = vx.iter().map(|v| v.abs()).fold(0.0, f32::max);
    let tol = (mass as f64) * (scale as f64) * 1.0e-3;
    assert!(p_com.abs() < tol, "p_com = {p_com} (tol {tol})");
}

// --- cumulative_injection bookkeeping ---

// rq-cfcce369
#[test]
fn berendsen_cumulative_injection_matches_kinetic_change() {
    let gpu = init_device().unwrap();
    let n = 16usize;
    let mass: f32 = 1.66e-27;
    let state = symmetric_state(n, mass, 1000.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = unbox_berendsen(build_berendsen(&gpu, n, &berendsen_kind(300.0, 1.0e-13)));
    let mut scratch = gpu.device.alloc_zeros::<f32>(1).unwrap();
    let k_before = compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as f32, &mut timings)
        .unwrap();
    let k_after = compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    let expected = k_after - k_before;
    let rel = (therm.cumulative_injection - expected).abs() / expected.abs().max(1.0e-30);
    assert!(
        rel < 1.0e-4,
        "cumulative_injection = {}, ΔK = {}",
        therm.cumulative_injection, expected
    );
}

// --- Log columns ---

// rq-a0b746c6
#[test]
fn berendsen_log_column_names_returns_berendsen_conserved() {
    let gpu = init_device().unwrap();
    let kind = berendsen_kind(300.0, 1.0e-13);
    let therm = build_berendsen(&gpu, 4, &kind);
    let names: Vec<&str> = therm.log_column_names().iter().map(|(n, _)| *n).collect();
    assert_eq!(names, vec!["berendsen_conserved"]);
}

// rq-2b114f41
#[test]
fn berendsen_log_column_values_subtracts_cumulative_injection() {
    let gpu = init_device().unwrap();
    let mut therm = unbox_berendsen(build_berendsen(&gpu, 4, &berendsen_kind(300.0, 1.0e-13)));
    therm.cumulative_injection = 1.0e-20;
    let extras = therm.log_column_values(2.5e-20, 3.0e-20);
    assert_eq!(extras.len(), 1);
    let expected: f64 = 2.5e-20 + 3.0e-20 - 1.0e-20;
    assert!((extras[0] - expected).abs() < 1.0e-30);
}

// --- Determinism ---

// rq-102e58cf
#[test]
fn berendsen_two_runs_with_identical_inputs_match() {
    let gpu = init_device().unwrap();
    let state = symmetric_state(8, 1.66e-27, 500.0);

    fn run_once(gpu: &GpuContext, state: &ParticleState) -> Vec<f32> {
        let n = state.particle_count();
        let mut buffers = ParticleBuffers::new(gpu, state).unwrap();
        let mut timings = Timings::new(gpu).unwrap();
        let mut therm = build_berendsen(gpu, n, &berendsen_kind(300.0, 1.0e-13));
        for _ in 0..5 {
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
        assert!(v.is_finite(), "non-finite velocity: {a:?}");
    }
}

// --- Temperature relaxation ---

// rq-314d24cf
#[test]
fn berendsen_temperature_relaxes_toward_target() {
    // Composed runner of VV (empty force-field) + Berendsen via the
    // dispatch order documented in `framework.md`: integrator.step
    // followed by thermostat.apply_post each step. With no forces, the
    // VV step is a pure drift and K is held constant by VV alone; the
    // thermostat drives K to K_target on the time scale τ.
    let gpu = init_device().unwrap();
    let n = 64usize;
    let mass_si: f32 = 1.66e-27;
    let temperature = 300.0_f64;
    let g_dof = (3 * n - 3) as f64;
    let k_target = (g_dof / 2.0) * (temperature / TEMP_F);
    let mass_au = mass_si as f64 / MASS_F;
    let v_each_au = (2.0 * k_target / ((n as f64) * 0.5 * mass_au)).sqrt();
    let v_each_si = (v_each_au * VEL_F) as f32;
    let state = symmetric_state(n, mass_si, v_each_si);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let dt = (1.0e-15 / TIME_F) as f32;
    let tau = 1.0e-13_f64;

    let mut integrator = dynamics::integrator::IntegratorRegistry::with_builtins()
        .build(
            &SlotConfig::from_params_str("velocity-verlet", "lossless = false"),
            &gpu,
            n, 0)
        .unwrap();
    let mut therm = build_berendsen(&gpu, n, &berendsen_kind(temperature, tau));

    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
    let n_steps = 500usize;
    for _ in 0..n_steps {
        integrator
            .step(&mut buffers, &mut sim_box, &mut ff, None, dt, &mut timings)
            .unwrap();
        therm
            .apply_post(&mut buffers, dt, &mut timings)
            .unwrap();
    }
    let mut scratch = gpu.device.alloc_zeros::<f32>(1).unwrap();
    let k_final = compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    let rel = (k_final - k_target).abs() / k_target;
    assert!(
        rel < 0.05,
        "after 5τ, K_final = {k_final:e} vs K_target = {k_target:e} (rel {rel})"
    );
}
