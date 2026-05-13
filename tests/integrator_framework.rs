use dynamics::gpu::{ParticleBuffers, init_device};
use dynamics::integrator::{Integrator, LangevinBaoabState};
use dynamics::io::IntegratorKind;
use dynamics::state::ParticleState;
use dynamics::timings::{KernelStage, Timings};

fn small_state(n: usize) -> ParticleState {
    let mut pos: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    if pos.is_empty() {
        pos = Vec::new();
    }
    let zero = vec![0.0_f32; n];
    let m = vec![1.0_f32; n];
    ParticleState::new(pos, zero.clone(), zero.clone(), zero.clone(), zero.clone(), zero, m, vec![0u32; n], None)
        .unwrap()
}

fn vv_kind(lossless: bool) -> IntegratorKind {
    IntegratorKind::VelocityVerlet { lossless }
}

fn langevin_kind(seed: u64) -> IntegratorKind {
    IntegratorKind::LangevinBaoab {
        friction: 1.0e12,
        temperature: 300.0,
        seed,
    }
}

// rq-e02917c3
#[test]
fn construct_vv_lossy() {
    let device = init_device().unwrap();
    let integrator = Integrator::new(device, 4, &vv_kind(false)).unwrap();
    assert!(matches!(integrator, Integrator::VelocityVerlet(_)));
}

// rq-db78448e
#[test]
fn construct_vv_lossless() {
    let device = init_device().unwrap();
    let integrator = Integrator::new(device, 4, &vv_kind(true)).unwrap();
    assert!(matches!(integrator, Integrator::VelocityVerlet(_)));
}

// rq-47877631
#[test]
fn construct_langevin() {
    let device = init_device().unwrap();
    let integrator = Integrator::new(device, 4, &langevin_kind(42)).unwrap();
    match integrator {
        Integrator::LangevinBaoab(LangevinBaoabState {
            friction,
            temperature,
            seed,
        }) => {
            assert_eq!(friction, 1.0e12);
            assert_eq!(temperature, 300.0);
            assert_eq!(seed, 42);
        }
        _ => panic!("expected LangevinBaoab variant"),
    }
}

// rq-48fd88ed
#[test]
fn construct_empty() {
    let device = init_device().unwrap();
    let integrator = Integrator::new(device, 0, &vv_kind(true)).unwrap();
    assert!(matches!(integrator, Integrator::VelocityVerlet(_)));
}

// rq-171b99f5
#[test]
fn pre_force_step_noop_on_empty() {
    let device = init_device().unwrap();
    let state = ParticleState::new(
        Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    let mut integrator = Integrator::new(device, 0, &vv_kind(false)).unwrap();
    integrator
        .pre_force_step(&mut buffers, 0.1, 1, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    assert!(report.stages.is_empty());
}

// rq-a49bb176
#[test]
fn post_force_step_noop_on_empty() {
    let device = init_device().unwrap();
    let state = ParticleState::new(
        Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    let mut integrator = Integrator::new(device, 0, &langevin_kind(1)).unwrap();
    integrator
        .post_force_step(&mut buffers, 0.1, 1, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    assert!(report.stages.is_empty());
}

// rq-2980a672
#[test]
fn vv_pre_force_step_launches_vv_kick_drift() {
    let device = init_device().unwrap();
    let state = small_state(4);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    let mut integrator = Integrator::new(device, 4, &vv_kind(false)).unwrap();
    integrator
        .pre_force_step(&mut buffers, 0.1, 1, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    let row = report.stages.iter().find(|s| s.name == "vv_kick_drift").unwrap();
    assert_eq!(row.count, 1);
}

// rq-36382434
#[test]
fn vv_post_force_step_launches_vv_kick() {
    let device = init_device().unwrap();
    let state = small_state(4);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    let mut integrator = Integrator::new(device, 4, &vv_kind(false)).unwrap();
    integrator
        .post_force_step(&mut buffers, 0.1, 1, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    let row = report.stages.iter().find(|s| s.name == "vv_kick").unwrap();
    assert_eq!(row.count, 1);
}

// rq-7b9aada4
#[test]
fn lossless_vv_uses_lossless_kernels() {
    let device = init_device().unwrap();
    let state = small_state(4);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    let mut integrator = Integrator::new(device, 4, &vv_kind(true)).unwrap();
    integrator.pre_force_step(&mut buffers, 0.1, 1, &mut timings).unwrap();
    integrator.post_force_step(&mut buffers, 0.1, 1, &mut timings).unwrap();
    let report = timings.finalize().unwrap();
    let names: Vec<&str> = report.stages.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"vv_kick_drift_lossless"));
    assert!(names.contains(&"vv_kick_lossless"));
    assert!(!names.contains(&"vv_kick_drift"));
    assert!(!names.contains(&"vv_kick"));
}

// rq-d12c24f0
#[test]
fn step_index_propagates_to_langevin() {
    let device = init_device().unwrap();
    let state = small_state(2);
    let mut buffers_a = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut buffers_b = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings_a = Timings::new(device.clone()).unwrap();
    let mut timings_b = Timings::new(device.clone()).unwrap();
    let mut integrator_a = Integrator::new(device.clone(), 2, &langevin_kind(1)).unwrap();
    let mut integrator_b = Integrator::new(device.clone(), 2, &langevin_kind(1)).unwrap();
    integrator_a.pre_force_step(&mut buffers_a, 1.0e-15, 1, &mut timings_a).unwrap();
    integrator_b.pre_force_step(&mut buffers_b, 1.0e-15, 2, &mut timings_b).unwrap();
    let mut state_a = state.clone();
    let mut state_b = state.clone();
    state_a.download_from(&buffers_a).unwrap();
    state_b.download_from(&buffers_b).unwrap();
    assert_ne!(state_a.velocities_x, state_b.velocities_x);
}

// rq-706001ec
#[test]
fn two_independent_runs_byte_identical() {
    let device = init_device().unwrap();
    let state = small_state(4);

    let mut buffers_a = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut buffers_b = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings_a = Timings::new(device.clone()).unwrap();
    let mut timings_b = Timings::new(device.clone()).unwrap();
    let mut integrator_a = Integrator::new(device.clone(), 4, &vv_kind(false)).unwrap();
    let mut integrator_b = Integrator::new(device.clone(), 4, &vv_kind(false)).unwrap();

    for step in 1..=10 {
        integrator_a.pre_force_step(&mut buffers_a, 0.001, step, &mut timings_a).unwrap();
        integrator_a.post_force_step(&mut buffers_a, 0.001, step, &mut timings_a).unwrap();
        integrator_b.pre_force_step(&mut buffers_b, 0.001, step, &mut timings_b).unwrap();
        integrator_b.post_force_step(&mut buffers_b, 0.001, step, &mut timings_b).unwrap();
    }

    let mut state_a = state.clone();
    let mut state_b = state.clone();
    state_a.download_from(&buffers_a).unwrap();
    state_b.download_from(&buffers_b).unwrap();
    assert_eq!(state_a.positions_x, state_b.positions_x);
    assert_eq!(state_a.velocities_x, state_b.velocities_x);
}

// Silence the unused-name lint for the imported KernelStage variant set.
#[test]
fn _imports_used() {
    let _ = KernelStage::LangevinKickHalf;
}
