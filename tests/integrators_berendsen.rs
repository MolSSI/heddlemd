// rq-25f24b26
//
// Berendsen weak-coupling thermostat integrator tests.

use dynamics::forces::{AngleList, BondList, ExclusionList, ForceField};
use dynamics::gpu::{
    GpuContext, ParticleBuffers, compute_kinetic_energy, init_device,
};
use dynamics::integrator::{BerendsenState, Integrator, IntegratorRegistry};
use dynamics::io::IntegratorKind;
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

fn berendsen_kind(temperature: f64, tau: f64) -> IntegratorKind {
    IntegratorKind::Berendsen { temperature, tau }
}

fn unbox_berendsen(integ: Box<dyn Integrator>) -> BerendsenState {
    unsafe { *Box::from_raw(Box::into_raw(integ) as *mut BerendsenState) }
}

// Build a state with prescribed mass and per-particle velocities whose
// COM-x is exactly zero (symmetric ±v pairs).
fn symmetric_state(n: usize, mass: f32, v_mag: f32) -> ParticleState {
    assert!(n.is_multiple_of(2));
    let mut vx: Vec<f32> = Vec::with_capacity(n);
    for _ in 0..n / 2 {
        vx.push(v_mag);
        vx.push(-v_mag);
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
        (0..n as u32).collect(),
        None,
        None,
    )
    .unwrap()
}

// --- Construction ---

#[test]
fn registry_builds_berendsen() {
    let gpu = init_device().unwrap();
    let kind = berendsen_kind(300.0, 1.0e-13);
    let _integrator = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 4)
        .unwrap();
}

#[test]
fn registry_builds_berendsen_particle_count_zero() {
    let gpu = init_device().unwrap();
    let kind = berendsen_kind(300.0, 1.0e-13);
    let _integrator = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 0)
        .unwrap();
}

#[test]
fn berendsen_state_precomputes_g_dof_and_kt_target() {
    let gpu = init_device().unwrap();
    let kind = berendsen_kind(300.0, 1.0e-13);
    let state =
        unbox_berendsen(IntegratorRegistry::with_builtins().build(&kind, &gpu, 4).unwrap());
    assert_eq!(state.g_dof, 9); // max(1, 3*4 - 3)
    let expected_kt = KB * 300.0;
    assert!((state.kt_target - expected_kt).abs() < 1.0e-30);
    assert_eq!(state.cumulative_injection, 0.0);
}

// --- Per-step kernel sequence ---

#[test]
fn berendsen_step_launches_expected_kernels() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = symmetric_state(n, 1.66e-27, 500.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut force_field = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let kind = berendsen_kind(300.0, 1.0e-13);
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, n)
        .unwrap();
    force_field
        .step(&mut buffers, &sim_box, &mut timings)
        .unwrap();
    integrator
        .step(
            &mut buffers,
            &mut sim_box,
            &mut force_field,
            1.0e-15,
            &mut timings,
        )
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
    assert_eq!(count_for(KernelStage::VV_KICK_DRIFT), 1);
    assert_eq!(count_for(KernelStage::VV_KICK), 1);
    assert_eq!(count_for(KernelStage::KINETIC_ENERGY_REDUCE), 1);
    assert_eq!(count_for(KernelStage::BERENDSEN_RESCALE_VELOCITIES), 1);
}

#[test]
fn berendsen_step_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = symmetric_state(0, 1.66e-27, 0.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut force_field = empty_force_field(&gpu, 0);
    let mut timings = Timings::new(&gpu).unwrap();
    let kind = berendsen_kind(300.0, 1.0e-13);
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 0)
        .unwrap();
    integrator
        .step(
            &mut buffers,
            &mut sim_box,
            &mut force_field,
            1.0e-15,
            &mut timings,
        )
        .unwrap();
}

// --- λ formula correctness ---

#[test]
fn berendsen_lambda_one_when_k_equals_target() {
    let gpu = init_device().unwrap();
    let n = 8usize;
    let mass: f32 = 1.66e-27;
    // Compute v such that K_old = K_target = (g_dof/2) · k_B · T.
    let temperature = 300.0_f64;
    let g_dof = (3 * n - 3) as f64;
    let k_target = (g_dof / 2.0) * KB * temperature;
    let v_each = (k_target / ((n as f64) * 0.5 * (mass as f64))).sqrt() as f32;
    let state = symmetric_state(n, mass, v_each);
    let snap_vx = state.velocities_x.clone();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integ = unbox_berendsen(
        IntegratorRegistry::with_builtins()
            .build(&berendsen_kind(temperature, 1.0e-13), &gpu, n)
            .unwrap(),
    );
    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
    integ
        .step(
            &mut buffers,
            &mut sim_box,
            &mut ff,
            1.0e-15,
            &mut timings,
        )
        .unwrap();
    let vx_after = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    // Velocities should equal the snapshot to within f32 round-off because λ ≈ 1.
    for (a, b) in vx_after.iter().zip(snap_vx.iter()) {
        let rel = (a - b).abs() / b.abs().max(1.0);
        assert!(rel < 1.0e-4, "vx after = {a}, before = {b}");
    }
}

#[test]
fn berendsen_lambda_squared_matches_analytical_when_cooling() {
    // K_old = 2 · K_target, dt/τ = 0.1 → λ² = 1 + 0.1·(0.5 - 1) = 0.95.
    let gpu = init_device().unwrap();
    let n = 8usize;
    let mass: f32 = 1.66e-27;
    let temperature = 300.0_f64;
    let g_dof = (3 * n - 3) as f64;
    let k_target = (g_dof / 2.0) * KB * temperature;
    let k_old_desired = 2.0 * k_target;
    let v_each = (k_old_desired / ((n as f64) * 0.5 * (mass as f64))).sqrt() as f32;
    let state = symmetric_state(n, mass, v_each);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let dt = 1.0e-14_f32;
    let tau = 1.0e-13_f64;
    let mut integ = unbox_berendsen(
        IntegratorRegistry::with_builtins()
            .build(&berendsen_kind(temperature, tau), &gpu, n)
            .unwrap(),
    );
    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
    let mut scratch = gpu.device.alloc_zeros::<f32>(1).unwrap();
    let k_before = compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    integ
        .step(&mut buffers, &mut sim_box, &mut ff, dt, &mut timings)
        .unwrap();
    let k_after = compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    // Expected λ² = 1 + (dt/τ) · (K_target/K_old − 1) computed with the
    // actual K_old measured by the kernel (slight f32 round-off vs the
    // analytical 2·K_target).
    let expected_lambda_sq = 1.0 + ((dt as f64) / tau) * (k_target / k_before - 1.0);
    let expected_k_after = expected_lambda_sq * k_before;
    let rel = (k_after - expected_k_after).abs() / expected_k_after.abs();
    assert!(
        rel < 1.0e-4,
        "k_after = {k_after}, expected ≈ {expected_k_after} (λ² = {expected_lambda_sq})"
    );
}

#[test]
fn berendsen_lambda_squared_clamped_to_zero_when_runaway() {
    // K_old = 100 · K_target, dt/τ = 2.0 → naive λ² = 1 + 2.0·(0.01-1) = -0.98
    // → clamped to 0. Post-step velocities are all zero.
    let gpu = init_device().unwrap();
    let n = 8usize;
    let mass: f32 = 1.66e-27;
    let temperature = 300.0_f64;
    let g_dof = (3 * n - 3) as f64;
    let k_target = (g_dof / 2.0) * KB * temperature;
    let v_each = (100.0 * k_target / ((n as f64) * 0.5 * (mass as f64))).sqrt() as f32;
    let state = symmetric_state(n, mass, v_each);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let dt = 2.0e-13_f32; // dt/τ = 2.0
    let tau = 1.0e-13_f64;
    let mut integ = unbox_berendsen(
        IntegratorRegistry::with_builtins()
            .build(&berendsen_kind(temperature, tau), &gpu, n)
            .unwrap(),
    );
    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
    integ
        .step(&mut buffers, &mut sim_box, &mut ff, dt, &mut timings)
        .unwrap();
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    for v in &vx {
        assert_eq!(*v, 0.0_f32, "all velocities should be quenched to zero, got {v}");
    }
}

#[test]
fn berendsen_skips_rescale_when_k_zero() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = symmetric_state(n, 1.66e-27, 0.0); // all velocities zero
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integ = unbox_berendsen(
        IntegratorRegistry::with_builtins()
            .build(&berendsen_kind(300.0, 1.0e-13), &gpu, n)
            .unwrap(),
    );
    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
    integ
        .step(
            &mut buffers,
            &mut sim_box,
            &mut ff,
            1.0e-15,
            &mut timings,
        )
        .unwrap();
    assert_eq!(integ.cumulative_injection, 0.0);
    // Verify the rescale kernel was NOT launched.
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

#[test]
fn berendsen_preserves_com_momentum() {
    let gpu = init_device().unwrap();
    let n = 16usize;
    let mass: f32 = 1.66e-27;
    let state = symmetric_state(n, mass, 500.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integ = IntegratorRegistry::with_builtins()
        .build(&berendsen_kind(300.0, 1.0e-13), &gpu, n)
        .unwrap();
    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
    for _ in 0..20 {
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, 1.0e-15, &mut timings)
            .unwrap();
    }
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let p_com: f64 = vx.iter().map(|&v| (mass as f64) * (v as f64)).sum();
    let scale: f32 = vx.iter().map(|v| v.abs()).fold(0.0, f32::max);
    let tol = (mass as f64) * (scale as f64) * 1.0e-3;
    assert!(p_com.abs() < tol, "p_com = {p_com} (tol {tol})");
}

// --- cumulative_injection bookkeeping ---

#[test]
fn berendsen_cumulative_injection_matches_kinetic_change() {
    let gpu = init_device().unwrap();
    let n = 16usize;
    let mass: f32 = 1.66e-27;
    let state = symmetric_state(n, mass, 1000.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integ = unbox_berendsen(
        IntegratorRegistry::with_builtins()
            .build(&berendsen_kind(300.0, 1.0e-13), &gpu, n)
            .unwrap(),
    );
    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
    let mut scratch = gpu.device.alloc_zeros::<f32>(1).unwrap();
    let k_before = compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    integ
        .step(
            &mut buffers,
            &mut sim_box,
            &mut ff,
            1.0e-15,
            &mut timings,
        )
        .unwrap();
    let k_after = compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    let expected = k_after - k_before;
    let rel = (integ.cumulative_injection - expected).abs() / expected.abs().max(1.0e-30);
    assert!(rel < 1.0e-4, "cumulative_injection = {}, ΔK = {}", integ.cumulative_injection, expected);
}

// --- Log columns ---

#[test]
fn berendsen_log_column_names_returns_berendsen_conserved() {
    let gpu = init_device().unwrap();
    let kind = berendsen_kind(300.0, 1.0e-13);
    let integ = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 4)
        .unwrap();
    assert_eq!(integ.log_column_names(), &["berendsen_conserved"]);
}

#[test]
fn berendsen_log_column_values_subtracts_cumulative_injection() {
    let gpu = init_device().unwrap();
    let mut integ = unbox_berendsen(
        IntegratorRegistry::with_builtins()
            .build(&berendsen_kind(300.0, 1.0e-13), &gpu, 4)
            .unwrap(),
    );
    integ.cumulative_injection = 1.0e-20;
    let extras = integ.log_column_values(2.5e-20, 3.0e-20);
    assert_eq!(extras.len(), 1);
    let expected = 2.5e-20 + 3.0e-20 - 1.0e-20;
    assert!((extras[0] - expected).abs() < 1.0e-30);
}

// --- Determinism ---

#[test]
fn berendsen_two_runs_with_identical_inputs_match() {
    let gpu = init_device().unwrap();
    let state = symmetric_state(8, 1.66e-27, 500.0);

    fn run_once(gpu: &GpuContext, state: &ParticleState) -> Vec<f32> {
        let n = state.particle_count();
        let mut buffers = ParticleBuffers::new(gpu, state).unwrap();
        let mut sim_box = box_large();
        let mut ff = empty_force_field(gpu, n);
        let mut timings = Timings::new(gpu).unwrap();
        let mut integ = IntegratorRegistry::with_builtins()
            .build(&berendsen_kind(300.0, 1.0e-13), gpu, n)
            .unwrap();
        ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
        for _ in 0..5 {
            integ
                .step(
                    &mut buffers,
                    &mut sim_box,
                    &mut ff,
                    1.0e-15,
                    &mut timings,
                )
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

#[test]
fn berendsen_temperature_relaxes_toward_target() {
    // Start at T = 2 · T_target; after several τ the temperature should
    // approach T_target. Exponential relaxation rate 1/τ; after n · τ
    // the offset is reduced by exp(-n).
    let gpu = init_device().unwrap();
    let n = 64usize;
    let mass: f32 = 1.66e-27;
    let temperature = 300.0_f64;
    let g_dof = (3 * n - 3) as f64;
    let k_target = (g_dof / 2.0) * KB * temperature;
    // K_initial = 2 · K_target
    let v_each = (2.0 * k_target / ((n as f64) * 0.5 * (mass as f64))).sqrt() as f32;
    let state = symmetric_state(n, mass, v_each);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let dt = 1.0e-15_f32;
    let tau = 1.0e-13_f64;
    let mut integ = IntegratorRegistry::with_builtins()
        .build(&berendsen_kind(temperature, tau), &gpu, n)
        .unwrap();
    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
    // Run 5τ ≈ 500 steps. Exponential decay factor exp(-5) ≈ 0.0067,
    // so K should be within 1% of K_target.
    let n_steps = 500usize;
    for _ in 0..n_steps {
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, dt, &mut timings)
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
