use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use dynamics::gpu::{
    LosslessBuffers, ParticleBuffers, init_device, vv_kick_drift_lossless, vv_kick_lossless,
};
use dynamics::state::ParticleState;

// --- Helpers ---

fn empty_state(n: usize) -> ParticleState {
    ParticleState::new(
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        None,
    )
    .expect("empty_state")
}

fn diverse_state(n: usize) -> ParticleState {
    let positions_x: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.5).collect();
    let positions_y: Vec<f32> = (0..n).map(|i| 2.0 - i as f32 * 0.3).collect();
    let positions_z: Vec<f32> = (0..n).map(|i| -0.5 + i as f32 * 0.2).collect();
    let velocities_x: Vec<f32> = (0..n).map(|i| 0.1 * (i as f32 + 1.0)).collect();
    let velocities_y: Vec<f32> = (0..n).map(|i| -0.07 * (i as f32 + 1.0)).collect();
    let velocities_z: Vec<f32> = (0..n).map(|i| 0.04 * (i as f32 + 1.0)).collect();
    let masses: Vec<f32> = (0..n).map(|i| 1.0 + 0.25 * i as f32).collect();
    let mut state = ParticleState::new(
        positions_x,
        positions_y,
        positions_z,
        velocities_x,
        velocities_y,
        velocities_z,
        masses,
        None,
    )
    .unwrap();
    state.forces_x = (0..n).map(|i| 0.05 + i as f32 * 0.02).collect();
    state.forces_y = (0..n).map(|i| -0.03 + i as f32 * 0.015).collect();
    state.forces_z = (0..n).map(|i| 0.07 - i as f32 * 0.005).collect();
    state
}

fn neg_in_place_f32(device: &Arc<CudaDevice>, slice: &mut CudaSlice<f32>) {
    let host: Vec<f32> = device.dtoh_sync_copy(slice).unwrap();
    let neg: Vec<f32> = host.into_iter().map(|x| -x).collect();
    device.htod_sync_copy_into(&neg, slice).unwrap();
}

fn neg_in_place_f64(device: &Arc<CudaDevice>, slice: &mut CudaSlice<f64>) {
    let host: Vec<f64> = device.dtoh_sync_copy(slice).unwrap();
    let neg: Vec<f64> = host.into_iter().map(|x| -x).collect();
    device.htod_sync_copy_into(&neg, slice).unwrap();
}

fn negate_velocities(buffers: &mut ParticleBuffers, lossless: &mut LosslessBuffers) {
    let device = buffers.device.clone();
    neg_in_place_f32(&device, &mut buffers.velocities_x);
    neg_in_place_f32(&device, &mut buffers.velocities_y);
    neg_in_place_f32(&device, &mut buffers.velocities_z);
    neg_in_place_f64(&device, &mut lossless.velocities_x_lo);
    neg_in_place_f64(&device, &mut lossless.velocities_y_lo);
    neg_in_place_f64(&device, &mut lossless.velocities_z_lo);
}

#[derive(Clone, PartialEq, Debug)]
struct FullSnapshot {
    positions_x: Vec<f32>,
    positions_y: Vec<f32>,
    positions_z: Vec<f32>,
    velocities_x: Vec<f32>,
    velocities_y: Vec<f32>,
    velocities_z: Vec<f32>,
    forces_x: Vec<f32>,
    forces_y: Vec<f32>,
    forces_z: Vec<f32>,
    masses: Vec<f32>,
    particle_ids: Vec<u32>,
    positions_x_lo: Vec<f64>,
    positions_y_lo: Vec<f64>,
    positions_z_lo: Vec<f64>,
    velocities_x_lo: Vec<f64>,
    velocities_y_lo: Vec<f64>,
    velocities_z_lo: Vec<f64>,
}

fn capture_snapshot(buffers: &ParticleBuffers, lossless: &LosslessBuffers) -> FullSnapshot {
    let device = buffers.device.clone();
    FullSnapshot {
        positions_x: device.dtoh_sync_copy(&buffers.positions_x).unwrap(),
        positions_y: device.dtoh_sync_copy(&buffers.positions_y).unwrap(),
        positions_z: device.dtoh_sync_copy(&buffers.positions_z).unwrap(),
        velocities_x: device.dtoh_sync_copy(&buffers.velocities_x).unwrap(),
        velocities_y: device.dtoh_sync_copy(&buffers.velocities_y).unwrap(),
        velocities_z: device.dtoh_sync_copy(&buffers.velocities_z).unwrap(),
        forces_x: device.dtoh_sync_copy(&buffers.forces_x).unwrap(),
        forces_y: device.dtoh_sync_copy(&buffers.forces_y).unwrap(),
        forces_z: device.dtoh_sync_copy(&buffers.forces_z).unwrap(),
        masses: device.dtoh_sync_copy(&buffers.masses).unwrap(),
        particle_ids: device.dtoh_sync_copy(&buffers.particle_ids).unwrap(),
        positions_x_lo: device.dtoh_sync_copy(&lossless.positions_x_lo).unwrap(),
        positions_y_lo: device.dtoh_sync_copy(&lossless.positions_y_lo).unwrap(),
        positions_z_lo: device.dtoh_sync_copy(&lossless.positions_z_lo).unwrap(),
        velocities_x_lo: device.dtoh_sync_copy(&lossless.velocities_x_lo).unwrap(),
        velocities_y_lo: device.dtoh_sync_copy(&lossless.velocities_y_lo).unwrap(),
        velocities_z_lo: device.dtoh_sync_copy(&lossless.velocities_z_lo).unwrap(),
    }
}

// --- Module loading ---

#[test] // rq-70fe268a
fn init_device_loads_lossless_kernels() {
    let device = init_device().expect("init_device");
    assert!(device.has_func("integrate", "vv_kick_drift_lossless"));
    assert!(device.has_func("integrate", "vv_kick_lossless"));
}

// --- LosslessBuffers construction ---

#[test] // rq-5bfa5b37
fn lossless_buffers_new_allocates_zero_initialised() {
    let device = init_device().expect("init_device");
    let lossless = LosslessBuffers::new(device.clone(), 4).expect("new");
    assert_eq!(lossless.particle_count(), 4);
    assert_eq!(lossless.positions_x_lo.len(), 4);
    assert_eq!(lossless.positions_y_lo.len(), 4);
    assert_eq!(lossless.positions_z_lo.len(), 4);
    assert_eq!(lossless.velocities_x_lo.len(), 4);
    assert_eq!(lossless.velocities_y_lo.len(), 4);
    assert_eq!(lossless.velocities_z_lo.len(), 4);
    assert_eq!(device.dtoh_sync_copy(&lossless.positions_x_lo).unwrap(), vec![0.0_f64; 4]);
    assert_eq!(device.dtoh_sync_copy(&lossless.positions_y_lo).unwrap(), vec![0.0_f64; 4]);
    assert_eq!(device.dtoh_sync_copy(&lossless.positions_z_lo).unwrap(), vec![0.0_f64; 4]);
    assert_eq!(device.dtoh_sync_copy(&lossless.velocities_x_lo).unwrap(), vec![0.0_f64; 4]);
    assert_eq!(device.dtoh_sync_copy(&lossless.velocities_y_lo).unwrap(), vec![0.0_f64; 4]);
    assert_eq!(device.dtoh_sync_copy(&lossless.velocities_z_lo).unwrap(), vec![0.0_f64; 4]);
}

#[test] // rq-b96ce51d
fn lossless_buffers_new_with_zero_particle_count() {
    let device = init_device().expect("init_device");
    let lossless = LosslessBuffers::new(device.clone(), 0).expect("new");
    assert_eq!(lossless.particle_count(), 0);
    assert_eq!(lossless.positions_x_lo.len(), 0);
    assert_eq!(lossless.positions_y_lo.len(), 0);
    assert_eq!(lossless.positions_z_lo.len(), 0);
    assert_eq!(lossless.velocities_x_lo.len(), 0);
    assert_eq!(lossless.velocities_y_lo.len(), 0);
    assert_eq!(lossless.velocities_z_lo.len(), 0);
}

// --- Lossless empty-state ---

#[test] // rq-58cac735
fn vv_kick_drift_lossless_empty_noop() {
    let device = init_device().expect("init_device");
    let mut buffers = ParticleBuffers::new(device.clone(), &empty_state(0)).unwrap();
    let mut lossless = LosslessBuffers::new(device.clone(), 0).unwrap();
    vv_kick_drift_lossless(&mut buffers, &mut lossless, 0.1).expect("kick_drift_lossless");
}

#[test] // rq-5626dfc6
fn vv_kick_lossless_empty_noop() {
    let device = init_device().expect("init_device");
    let mut buffers = ParticleBuffers::new(device.clone(), &empty_state(0)).unwrap();
    let mut lossless = LosslessBuffers::new(device.clone(), 0).unwrap();
    vv_kick_lossless(&mut buffers, &mut lossless, 0.1).expect("kick_lossless");
}

// --- Block-non-aligned ---

#[test] // rq-5a6d5e9e
fn vv_kick_drift_lossless_block_non_aligned() {
    let device = init_device().expect("init_device");
    let n = 1000;
    let mut state = empty_state(n);
    state.velocities_x = vec![1.0_f32; n];
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut lossless = LosslessBuffers::new(device.clone(), n).unwrap();

    let initial_positions_x = state.positions_x.clone();
    vv_kick_drift_lossless(&mut buffers, &mut lossless, 0.1).expect("kick_drift_lossless");

    let final_positions_x: Vec<f32> = device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let final_velocities_x: Vec<f32> = device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    for i in 0..n {
        assert_eq!(
            final_positions_x[i],
            initial_positions_x[i] + 0.1_f32,
            "positions_x[{i}]"
        );
        assert_eq!(final_velocities_x[i], 1.0_f32, "velocities_x[{i}]");
    }
}

// --- Lossless side effects ---

#[test] // rq-bb075030
fn vv_kick_drift_lossless_does_not_modify_forces_masses_ids() {
    let device = init_device().expect("init_device");
    let state = diverse_state(4);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut lossless = LosslessBuffers::new(device.clone(), 4).unwrap();
    let snapshot = capture_snapshot(&buffers, &lossless);

    vv_kick_drift_lossless(&mut buffers, &mut lossless, 0.1).expect("kick_drift_lossless");

    let after = capture_snapshot(&buffers, &lossless);
    assert_eq!(after.forces_x, snapshot.forces_x);
    assert_eq!(after.forces_y, snapshot.forces_y);
    assert_eq!(after.forces_z, snapshot.forces_z);
    assert_eq!(after.masses, snapshot.masses);
    assert_eq!(after.particle_ids, snapshot.particle_ids);
}

#[test] // rq-acafdfe4
fn vv_kick_lossless_does_not_modify_positions_forces_masses_ids() {
    let device = init_device().expect("init_device");
    let state = diverse_state(4);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut lossless = LosslessBuffers::new(device.clone(), 4).unwrap();
    let snapshot = capture_snapshot(&buffers, &lossless);

    vv_kick_lossless(&mut buffers, &mut lossless, 0.1).expect("kick_lossless");

    let after = capture_snapshot(&buffers, &lossless);
    assert_eq!(after.positions_x, snapshot.positions_x);
    assert_eq!(after.positions_y, snapshot.positions_y);
    assert_eq!(after.positions_z, snapshot.positions_z);
    assert_eq!(after.forces_x, snapshot.forces_x);
    assert_eq!(after.forces_y, snapshot.forces_y);
    assert_eq!(after.forces_z, snapshot.forces_z);
    assert_eq!(after.masses, snapshot.masses);
    assert_eq!(after.particle_ids, snapshot.particle_ids);
}

// --- Lossless bit-reversibility ---

fn assert_observables_match(a: &FullSnapshot, b: &FullSnapshot) {
    assert_eq!(a.positions_x, b.positions_x, "positions_x");
    assert_eq!(a.positions_y, b.positions_y, "positions_y");
    assert_eq!(a.positions_z, b.positions_z, "positions_z");
    assert_eq!(a.velocities_x, b.velocities_x, "velocities_x");
    assert_eq!(a.velocities_y, b.velocities_y, "velocities_y");
    assert_eq!(a.velocities_z, b.velocities_z, "velocities_z");
    assert_eq!(a.forces_x, b.forces_x, "forces_x");
    assert_eq!(a.forces_y, b.forces_y, "forces_y");
    assert_eq!(a.forces_z, b.forces_z, "forces_z");
    assert_eq!(a.masses, b.masses, "masses");
    assert_eq!(a.particle_ids, b.particle_ids, "particle_ids");
}

fn assert_residuals_close(a: &FullSnapshot, b: &FullSnapshot, tol: f64) {
    let pairs = [
        (&a.positions_x_lo, &b.positions_x_lo, "positions_x_lo"),
        (&a.positions_y_lo, &b.positions_y_lo, "positions_y_lo"),
        (&a.positions_z_lo, &b.positions_z_lo, "positions_z_lo"),
        (&a.velocities_x_lo, &b.velocities_x_lo, "velocities_x_lo"),
        (&a.velocities_y_lo, &b.velocities_y_lo, "velocities_y_lo"),
        (&a.velocities_z_lo, &b.velocities_z_lo, "velocities_z_lo"),
    ];
    for (left, right, label) in pairs {
        assert_eq!(left.len(), right.len(), "{label} length");
        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            let diff = (l - r).abs();
            assert!(
                diff <= tol,
                "{label}[{i}] differs by {diff} (left={l}, right={r}, tol={tol})"
            );
        }
    }
}

#[test] // rq-1a504311
fn single_step_round_trip_zero_force() {
    let device = init_device().expect("init_device");
    let n = 8;
    let positions_x: Vec<f32> = (0..n).map(|i| 0.5 + i as f32 * 0.3).collect();
    let positions_y: Vec<f32> = (0..n).map(|i| -1.0 + i as f32 * 0.2).collect();
    let positions_z: Vec<f32> = (0..n).map(|i| 0.7 + i as f32 * 0.15).collect();
    let velocities_x: Vec<f32> = (0..n).map(|i| 0.1 * (i as f32 + 1.0)).collect();
    let velocities_y: Vec<f32> = (0..n).map(|i| -0.05 * (i as f32 + 1.0)).collect();
    let velocities_z: Vec<f32> = (0..n).map(|i| 0.07 * (i as f32 + 1.0)).collect();
    let state = ParticleState::new(
        positions_x,
        positions_y,
        positions_z,
        velocities_x,
        velocities_y,
        velocities_z,
        vec![1.0_f32; n],
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut lossless = LosslessBuffers::new(device.clone(), n).unwrap();
    let snapshot = capture_snapshot(&buffers, &lossless);

    vv_kick_drift_lossless(&mut buffers, &mut lossless, 0.1).expect("forward");
    negate_velocities(&mut buffers, &mut lossless);
    vv_kick_drift_lossless(&mut buffers, &mut lossless, 0.1).expect("reverse");
    negate_velocities(&mut buffers, &mut lossless);

    let after = capture_snapshot(&buffers, &lossless);
    assert_observables_match(&after, &snapshot);
    // Residuals carry compensation bookkeeping that may drift by f64 ULPs
    // after a round-trip; the visible f32 state stays bit-exact.
    assert_residuals_close(&after, &snapshot, 1e-12);
}

#[test] // rq-b73316ed
fn multi_step_round_trip_constant_force() {
    let device = init_device().expect("init_device");
    let n = 16;
    let state = diverse_state(n);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut lossless = LosslessBuffers::new(device.clone(), n).unwrap();
    let snapshot = capture_snapshot(&buffers, &lossless);

    let dt = 0.001_f32;
    let n_steps = 50;
    for _ in 0..n_steps {
        vv_kick_drift_lossless(&mut buffers, &mut lossless, dt).expect("forward kd");
        vv_kick_lossless(&mut buffers, &mut lossless, dt).expect("forward k");
    }
    negate_velocities(&mut buffers, &mut lossless);
    for _ in 0..n_steps {
        vv_kick_drift_lossless(&mut buffers, &mut lossless, dt).expect("reverse kd");
        vv_kick_lossless(&mut buffers, &mut lossless, dt).expect("reverse k");
    }
    negate_velocities(&mut buffers, &mut lossless);

    let after = capture_snapshot(&buffers, &lossless);
    assert_observables_match(&after, &snapshot);
    // 50 forward + 50 reverse steps accumulate at most a few f64 ULPs of
    // residual drift; the visible f32 state stays bit-exact.
    assert_residuals_close(&after, &snapshot, 1e-10);
}

#[test] // rq-2a0e97f5
fn two_independent_lossless_runs_byte_identical() {
    let device = init_device().expect("init_device");
    let n = 64;
    let state = diverse_state(n);
    let dt = 0.001_f32;
    let n_steps = 10;

    let run = |state: &ParticleState| -> FullSnapshot {
        let mut buffers = ParticleBuffers::new(device.clone(), state).unwrap();
        let mut lossless = LosslessBuffers::new(device.clone(), n).unwrap();
        for _ in 0..n_steps {
            vv_kick_drift_lossless(&mut buffers, &mut lossless, dt).unwrap();
            vv_kick_lossless(&mut buffers, &mut lossless, dt).unwrap();
        }
        capture_snapshot(&buffers, &lossless)
    };

    let a = run(&state);
    let b = run(&state);
    assert_eq!(a, b);
}
