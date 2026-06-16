// Reversibility relies on the lossless compensated-summation kernels,
// which are only compiled into the default (f32) build.
#![cfg(not(feature = "f64"))]

mod common;
use common::*;

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};
use heddle_md::gpu::{
    GpuContext, LennardJonesParameterTable, LosslessBuffers, ParticleBuffers,
    init_device, vv_kick, vv_kick_drift, vv_kick_drift_lossless, vv_kick_lossless,
};
use heddle_md::pbc::SimulationBox;
use heddle_md::state::ParticleState;
use heddle_md::precision::Real;

const N: usize = 64;
const BOX_L: Real = 8.0;
const LATTICE_SPACING: Real = 2.0;
const LATTICE_ORIGIN: Real = -3.0;
const DT: Real = 0.001;
const SIGMA: Real = 1.0;
const EPSILON: Real = 1.0;
const CUTOFF: Real = 2.5;
const N_STEPS: usize = 100;
const RESIDUAL_TOL: f64 = 1e-10;

// rq-7600786b
fn build_initial_state() -> ParticleState {
    let mut positions_x = Vec::with_capacity(N);
    let mut positions_y = Vec::with_capacity(N);
    let mut positions_z = Vec::with_capacity(N);
    for ix in 0..4 {
        for iy in 0..4 {
            for iz in 0..4 {
                let i = ix * 16 + iy * 4 + iz;
                let fi = i as Real;
                let perturb_x = 0.2 * (fi * 0.7 + 0.0).sin();
                let perturb_y = 0.2 * (fi * 0.7 + 1.1).sin();
                let perturb_z = 0.2 * (fi * 0.7 + 2.3).sin();
                positions_x.push(ix as Real * LATTICE_SPACING + LATTICE_ORIGIN + perturb_x);
                positions_y.push(iy as Real * LATTICE_SPACING + LATTICE_ORIGIN + perturb_y);
                positions_z.push(iz as Real * LATTICE_SPACING + LATTICE_ORIGIN + perturb_z);
            }
        }
    }
    ParticleState::new(
        positions_x,
        positions_y,
        positions_z,
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![1.0; N],
        vec![0.0; N],
        vec![0u32; N],
        None,
            None,
    )
    .expect("build_initial_state")
}

#[derive(Clone, Debug)]
struct FullSnapshot {
    positions_x: Vec<Real>,
    positions_y: Vec<Real>,
    positions_z: Vec<Real>,
    velocities_x: Vec<Real>,
    velocities_y: Vec<Real>,
    velocities_z: Vec<Real>,
    forces_x: Vec<Real>,
    forces_y: Vec<Real>,
    forces_z: Vec<Real>,
    masses: Vec<Real>,
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

fn neg_in_place(device: &Arc<CudaDevice>, slice: &mut CudaSlice<Real>) {
    let host: Vec<Real> = device.dtoh_sync_copy(slice).unwrap();
    let neg: Vec<Real> = host.into_iter().map(|x| -x).collect();
    device.htod_sync_copy_into(&neg, slice).unwrap();
}

fn neg_in_place_f64(device: &Arc<CudaDevice>, slice: &mut CudaSlice<f64>) {
    let host: Vec<f64> = device.dtoh_sync_copy(slice).unwrap();
    let neg: Vec<f64> = host.into_iter().map(|x| -x).collect();
    device.htod_sync_copy_into(&neg, slice).unwrap();
}

fn negate_velocities_lossless(buffers: &mut ParticleBuffers, lossless: &mut LosslessBuffers) {
    let device = buffers.device.clone();
    neg_in_place(&device, &mut buffers.velocities_x);
    neg_in_place(&device, &mut buffers.velocities_y);
    neg_in_place(&device, &mut buffers.velocities_z);
    neg_in_place_f64(&device, &mut lossless.velocities_x_lo);
    neg_in_place_f64(&device, &mut lossless.velocities_y_lo);
    neg_in_place_f64(&device, &mut lossless.velocities_z_lo);
}

fn negate_velocities_lossy(buffers: &mut ParticleBuffers) {
    let device = buffers.device.clone();
    neg_in_place(&device, &mut buffers.velocities_x);
    neg_in_place(&device, &mut buffers.velocities_y);
    neg_in_place(&device, &mut buffers.velocities_z);
}

struct PipelineFixture {
    buffers: ParticleBuffers,
    sim_box: SimulationBox,
    params: LennardJonesParameterTable,
}

impl PipelineFixture {
    fn build(gpu: &GpuContext, state: &ParticleState) -> Self {
        let buffers = ParticleBuffers::new(gpu, state).unwrap();
        let sim_box = SimulationBox::new(BOX_L, BOX_L, BOX_L, 0.0, 0.0, 0.0).unwrap();
        let params = single_type_lj_table(&gpu.device, SIGMA, EPSILON, CUTOFF);
        Self {
            buffers,
            sim_box,
            params,
        }
    }

    fn warm_up(&mut self) {
        lj_pair_force_into_buffers(&mut self.buffers, &self.sim_box, &self.params).unwrap();
    }
}

// rq-7b5eef8c
fn lossless_step(fixture: &mut PipelineFixture, lossless: &mut LosslessBuffers, dt: Real) {
    vv_kick_drift_lossless(&mut fixture.buffers, lossless, &fixture.sim_box, dt).unwrap();
    lj_pair_force_into_buffers(&mut fixture.buffers, &fixture.sim_box, &fixture.params).unwrap();
    vv_kick_lossless(&mut fixture.buffers, lossless, dt).unwrap();
}

fn lossy_step(fixture: &mut PipelineFixture, dt: Real) {
    vv_kick_drift(&mut fixture.buffers, &fixture.sim_box, dt).unwrap();
    lj_pair_force_into_buffers(&mut fixture.buffers, &fixture.sim_box, &fixture.params).unwrap();
    vv_kick(&mut fixture.buffers, dt).unwrap();
}

// rq-090b70bb
fn assert_observables_match(after: &FullSnapshot, before: &FullSnapshot) {
    assert_eq!(after.positions_x, before.positions_x, "positions_x");
    assert_eq!(after.positions_y, before.positions_y, "positions_y");
    assert_eq!(after.positions_z, before.positions_z, "positions_z");
    assert_eq!(after.velocities_x, before.velocities_x, "velocities_x");
    assert_eq!(after.velocities_y, before.velocities_y, "velocities_y");
    assert_eq!(after.velocities_z, before.velocities_z, "velocities_z");
    assert_eq!(after.forces_x, before.forces_x, "forces_x");
    assert_eq!(after.forces_y, before.forces_y, "forces_y");
    assert_eq!(after.forces_z, before.forces_z, "forces_z");
    assert_eq!(after.masses, before.masses, "masses");
    assert_eq!(after.particle_ids, before.particle_ids, "particle_ids");
}

fn assert_residuals_close(after: &FullSnapshot, before: &FullSnapshot, tol: f64) {
    let pairs = [
        (&after.positions_x_lo, &before.positions_x_lo, "positions_x_lo"),
        (&after.positions_y_lo, &before.positions_y_lo, "positions_y_lo"),
        (&after.positions_z_lo, &before.positions_z_lo, "positions_z_lo"),
        (&after.velocities_x_lo, &before.velocities_x_lo, "velocities_x_lo"),
        (&after.velocities_y_lo, &before.velocities_y_lo, "velocities_y_lo"),
        (&after.velocities_z_lo, &before.velocities_z_lo, "velocities_z_lo"),
    ];
    for (left, right, label) in pairs {
        assert_eq!(left.len(), right.len(), "{label} length");
        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            let diff = (l - r).abs();
            assert!(
                diff <= tol,
                "{label}[{i}] differs by {diff} (after={l}, before={r}, tol={tol})"
            );
        }
    }
}

// rq-1d618f18
fn lossless_round_trip(n_steps: usize) -> (FullSnapshot, FullSnapshot) {
    let gpu = init_device().expect("init_device");
    let state = build_initial_state();
    let mut fixture = PipelineFixture::build(&gpu, &state);
    let mut lossless = LosslessBuffers::new(&gpu, N).unwrap();

    fixture.warm_up();
    let snapshot = capture_snapshot(&fixture.buffers, &lossless);

    for _ in 0..n_steps {
        lossless_step(&mut fixture, &mut lossless, DT);
    }
    negate_velocities_lossless(&mut fixture.buffers, &mut lossless);
    for _ in 0..n_steps {
        lossless_step(&mut fixture, &mut lossless, DT);
    }
    negate_velocities_lossless(&mut fixture.buffers, &mut lossless);

    let after = capture_snapshot(&fixture.buffers, &lossless);
    (snapshot, after)
}

#[test] // rq-0099ef65
fn single_step_lossless_round_trip_restores_observables_bit_exactly() {
    let (before, after) = lossless_round_trip(1);
    assert_observables_match(&after, &before);
    assert_residuals_close(&after, &before, RESIDUAL_TOL);
}

#[test] // rq-7822b88b
fn hundred_step_lossless_round_trip_restores_observables_bit_exactly() {
    let (before, after) = lossless_round_trip(N_STEPS);
    assert_observables_match(&after, &before);
    assert_residuals_close(&after, &before, RESIDUAL_TOL);
}

#[test] // rq-b87fd5e8
fn positions_visibly_evolve_over_lossless_forward_run() {
    let gpu = init_device().expect("init_device");
    let state = build_initial_state();
    let mut fixture = PipelineFixture::build(&gpu, &state);
    let mut lossless = LosslessBuffers::new(&gpu, N).unwrap();

    fixture.warm_up();
    let initial = capture_snapshot(&fixture.buffers, &lossless);

    for _ in 0..N_STEPS {
        lossless_step(&mut fixture, &mut lossless, DT);
    }

    let after = capture_snapshot(&fixture.buffers, &lossless);

    let max_disp = (0..N)
        .map(|i| {
            let dx = after.positions_x[i] - initial.positions_x[i];
            let dy = after.positions_y[i] - initial.positions_y[i];
            let dz = after.positions_z[i] - initial.positions_z[i];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(0.0, Real::max);
    assert!(
        max_disp > 0.001,
        "max displacement after {N_STEPS} steps was {max_disp} (should exceed 0.001)"
    );
}

#[test] // rq-ed048159
fn all_observables_finite_after_lossless_forward_run() {
    let gpu = init_device().expect("init_device");
    let state = build_initial_state();
    let mut fixture = PipelineFixture::build(&gpu, &state);
    let mut lossless = LosslessBuffers::new(&gpu, N).unwrap();

    fixture.warm_up();
    for _ in 0..N_STEPS {
        lossless_step(&mut fixture, &mut lossless, DT);
    }

    let after = capture_snapshot(&fixture.buffers, &lossless);
    let arrays: [(&str, &[Real]); 9] = [
        ("positions_x", &after.positions_x),
        ("positions_y", &after.positions_y),
        ("positions_z", &after.positions_z),
        ("velocities_x", &after.velocities_x),
        ("velocities_y", &after.velocities_y),
        ("velocities_z", &after.velocities_z),
        ("forces_x", &after.forces_x),
        ("forces_y", &after.forces_y),
        ("forces_z", &after.forces_z),
    ];
    for (label, arr) in arrays {
        for (i, &v) in arr.iter().enumerate() {
            assert!(v.is_finite(), "{label}[{i}] = {v} is not finite");
        }
    }
}

#[test] // rq-1b44b5da
fn lossy_round_trip_does_not_restore_observables() {
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let state = build_initial_state();
    let mut fixture = PipelineFixture::build(&gpu, &state);

    fixture.warm_up();
    let before_positions_x: Vec<Real> = device.dtoh_sync_copy(&fixture.buffers.positions_x).unwrap();
    let before_positions_y: Vec<Real> = device.dtoh_sync_copy(&fixture.buffers.positions_y).unwrap();
    let before_positions_z: Vec<Real> = device.dtoh_sync_copy(&fixture.buffers.positions_z).unwrap();
    let before_velocities_x: Vec<Real> = device.dtoh_sync_copy(&fixture.buffers.velocities_x).unwrap();
    let before_velocities_y: Vec<Real> = device.dtoh_sync_copy(&fixture.buffers.velocities_y).unwrap();
    let before_velocities_z: Vec<Real> = device.dtoh_sync_copy(&fixture.buffers.velocities_z).unwrap();

    for _ in 0..N_STEPS {
        lossy_step(&mut fixture, DT);
    }
    negate_velocities_lossy(&mut fixture.buffers);
    for _ in 0..N_STEPS {
        lossy_step(&mut fixture, DT);
    }
    negate_velocities_lossy(&mut fixture.buffers);

    let after_positions_x: Vec<Real> = device.dtoh_sync_copy(&fixture.buffers.positions_x).unwrap();
    let after_positions_y: Vec<Real> = device.dtoh_sync_copy(&fixture.buffers.positions_y).unwrap();
    let after_positions_z: Vec<Real> = device.dtoh_sync_copy(&fixture.buffers.positions_z).unwrap();
    let after_velocities_x: Vec<Real> = device.dtoh_sync_copy(&fixture.buffers.velocities_x).unwrap();
    let after_velocities_y: Vec<Real> = device.dtoh_sync_copy(&fixture.buffers.velocities_y).unwrap();
    let after_velocities_z: Vec<Real> = device.dtoh_sync_copy(&fixture.buffers.velocities_z).unwrap();

    let differs = before_positions_x != after_positions_x
        || before_positions_y != after_positions_y
        || before_positions_z != after_positions_z
        || before_velocities_x != after_velocities_x
        || before_velocities_y != after_velocities_y
        || before_velocities_z != after_velocities_z;
    assert!(
        differs,
        "lossy round trip unexpectedly restored every observable bit-exactly"
    );
}
