use cudarc::driver::DeviceSlice;
use dynamics::gpu::{ParticleBuffers, init_device, vv_kick, vv_kick_drift};
use dynamics::state::ParticleState;
use dynamics::pbc::SimulationBox;

fn empty_state(n: usize) -> ParticleState {
    ParticleState::new(
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        vec![0u32; n],
        None,
            None,
    )
    .expect("empty_state: ParticleState::new should succeed")
}

fn snapshot_via_download(buffers: &ParticleBuffers) -> ParticleState {
    let n = buffers.particle_count();
    let mut state = empty_state(n);
    state.download_from(buffers).expect("download_from failed");
    state
}

fn diverse_state(n: usize) -> ParticleState {
    // Distinct nonzero values for every field so byte-identical comparisons
    // are meaningful.
    let positions_x: Vec<f32> = (0..n).map(|i| 1.0 + i as f32).collect();
    let positions_y: Vec<f32> = (0..n).map(|i| 2.0 + i as f32).collect();
    let positions_z: Vec<f32> = (0..n).map(|i| 3.0 + i as f32).collect();
    let velocities_x: Vec<f32> = (0..n).map(|i| 0.1 * (i as f32 + 1.0)).collect();
    let velocities_y: Vec<f32> = (0..n).map(|i| -0.2 * (i as f32 + 1.0)).collect();
    let velocities_z: Vec<f32> = (0..n).map(|i| 0.05 * (i as f32 + 1.0)).collect();
    let masses: Vec<f32> = (0..n).map(|i| 1.0 + 0.5 * i as f32).collect();
    let mut state = ParticleState::new(
        positions_x,
        positions_y,
        positions_z,
        velocities_x,
        velocities_y,
        velocities_z,
        masses,
        vec![0u32; n],
        None,
            None,
    )
    .unwrap();
    state.forces_x = (0..n).map(|i| 0.5 + i as f32 * 0.1).collect();
    state.forces_y = (0..n).map(|i| -0.3 + i as f32 * 0.2).collect();
    state.forces_z = (0..n).map(|i| 0.7 - i as f32 * 0.05).collect();
    state
}

// rq-bc8375ce
#[test]
fn init_device_loads_integrate_module_with_both_kernels() {
    let device = init_device().expect("init_device");
    assert!(device.has_func("integrate", "vv_kick_drift"));
    assert!(device.has_func("integrate", "vv_kick"));
}

// rq-4f7dc024
#[test]
fn vv_kick_drift_advances_position_when_force_zero() {
    let device = init_device().expect("init_device");
    let sim_box = SimulationBox::new_orthorhombic(1.0e6, 1.0e6, 1.0e6).unwrap();
    let state = ParticleState::new(
        vec![1.0],
        vec![2.0],
        vec![3.0],
        vec![0.5],
        vec![-0.25],
        vec![0.125],
        vec![1.0],
        vec![0u32; 1],
        None,
            None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).expect("buffers");

    vv_kick_drift(&mut buffers, &sim_box, 0.1).expect("kick_drift");
    let result = snapshot_via_download(&buffers);

    // Force is zero, so velocities_* are unchanged and positions_* advance by v*dt.
    assert_eq!(result.velocities_x, vec![0.5_f32]);
    assert_eq!(result.velocities_y, vec![-0.25_f32]);
    assert_eq!(result.velocities_z, vec![0.125_f32]);
    assert_eq!(result.positions_x, vec![1.0_f32 + 0.5_f32 * 0.1_f32]);
    assert_eq!(result.positions_y, vec![2.0_f32 + (-0.25_f32) * 0.1_f32]);
    assert_eq!(result.positions_z, vec![3.0_f32 + 0.125_f32 * 0.1_f32]);
}

// rq-d25000c5
#[test]
fn vv_kick_drift_exact_half_step_under_constant_force() {
    let device = init_device().expect("init_device");
    let sim_box = SimulationBox::new_orthorhombic(1.0e6, 1.0e6, 1.0e6).unwrap();
    let mut state = ParticleState::new(
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![1.0],
        vec![0u32; 1],
        None,
            None,
    )
    .unwrap();
    state.forces_x = vec![2.0];
    state.forces_y = vec![-4.0];
    state.forces_z = vec![1.0];

    let mut buffers = ParticleBuffers::new(device.clone(), &state).expect("buffers");
    vv_kick_drift(&mut buffers, &sim_box, 0.1).expect("kick_drift");
    let result = snapshot_via_download(&buffers);

    let half_dt = 0.1_f32 * 0.5_f32;
    let vx = 2.0_f32 * half_dt;
    let vy = -4.0_f32 * half_dt;
    let vz = 1.0_f32 * half_dt;
    assert_eq!(result.velocities_x[0], vx);
    assert_eq!(result.velocities_y[0], vy);
    assert_eq!(result.velocities_z[0], vz);
    assert_eq!(result.positions_x[0], vx * 0.1_f32);
    assert_eq!(result.positions_y[0], vy * 0.1_f32);
    assert_eq!(result.positions_z[0], vz * 0.1_f32);
}

// rq-2f52c25e
#[test]
fn vv_kick_leaves_velocity_unchanged_when_force_zero() {
    let device = init_device().expect("init_device");
    let state = ParticleState::new(
        vec![1.0],
        vec![2.0],
        vec![3.0],
        vec![0.5],
        vec![-0.25],
        vec![0.125],
        vec![1.0],
        vec![0u32; 1],
        None,
            None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).expect("buffers");

    vv_kick(&mut buffers, 0.1).expect("kick");
    let result = snapshot_via_download(&buffers);

    assert_eq!(result.velocities_x, vec![0.5_f32]);
    assert_eq!(result.velocities_y, vec![-0.25_f32]);
    assert_eq!(result.velocities_z, vec![0.125_f32]);
    assert_eq!(result.positions_x, vec![1.0_f32]);
    assert_eq!(result.positions_y, vec![2.0_f32]);
    assert_eq!(result.positions_z, vec![3.0_f32]);
}

// rq-29718dcf
#[test]
fn full_step_matches_constant_acceleration_kinematics() {
    let device = init_device().expect("init_device");
    let sim_box = SimulationBox::new_orthorhombic(1.0e6, 1.0e6, 1.0e6).unwrap();
    let mut state = ParticleState::new(
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![1.0],
        vec![0u32; 1],
        None,
            None,
    )
    .unwrap();
    state.forces_x = vec![2.0];
    let mut buffers = ParticleBuffers::new(device.clone(), &state).expect("buffers");

    vv_kick_drift(&mut buffers, &sim_box, 0.1).expect("kick_drift");
    vv_kick(&mut buffers, 0.1).expect("kick");
    let result = snapshot_via_download(&buffers);

    let dt = 0.1_f32;
    let half_dt = dt * 0.5_f32;
    let a = 2.0_f32 / 1.0_f32;
    // The kernel produces v_half = a*half_dt, then x += v_half*dt, then
    // v_final = v_half + a*half_dt. Match its arithmetic.
    let v_half = a * half_dt;
    let expected_x = v_half * dt;
    let expected_v = v_half + a * half_dt;
    assert_eq!(result.positions_x[0], expected_x);
    assert_eq!(result.velocities_x[0], expected_v);
    assert_eq!(result.positions_y[0], 0.0);
    assert_eq!(result.positions_z[0], 0.0);
    assert_eq!(result.velocities_y[0], 0.0);
    assert_eq!(result.velocities_z[0], 0.0);
}

// rq-bd149f52
#[test]
fn acceleration_scales_inversely_with_mass() {
    let device = init_device().expect("init_device");
    let sim_box = SimulationBox::new_orthorhombic(1.0e6, 1.0e6, 1.0e6).unwrap();
    let mut state = ParticleState::new(
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![1.0, 4.0],
        vec![0u32; 2],
        None,
            None,
    )
    .unwrap();
    state.forces_x = vec![1.0, 1.0];
    let mut buffers = ParticleBuffers::new(device.clone(), &state).expect("buffers");

    vv_kick_drift(&mut buffers, &sim_box, 0.2).expect("kick_drift");
    let result = snapshot_via_download(&buffers);

    let half_dt = 0.2_f32 * 0.5_f32;
    assert_eq!(result.velocities_x[0], (1.0_f32 / 1.0_f32) * half_dt);
    assert_eq!(result.velocities_x[1], (1.0_f32 / 4.0_f32) * half_dt);
}

// rq-b13eba96
#[test]
fn particles_evolve_independently() {
    let device = init_device().expect("init_device");
    let sim_box = SimulationBox::new_orthorhombic(1.0e6, 1.0e6, 1.0e6).unwrap();
    let mut state = ParticleState::new(
        vec![0.0, 1.0, -2.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![0u32; 3],
        None,
            None,
    )
    .unwrap();
    state.forces_x = vec![1.0, -2.0, 0.5];
    let initial_positions_x = state.positions_x.clone();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).expect("buffers");

    vv_kick_drift(&mut buffers, &sim_box, 0.1).expect("kick_drift");
    vv_kick(&mut buffers, 0.1).expect("kick");
    let result = snapshot_via_download(&buffers);

    let dt = 0.1_f32;
    let half_dt = dt * 0.5_f32;
    for i in 0..3 {
        let f = state.forces_x[i];
        let m = state.masses[i];
        let a = f / m;
        let v_half = a * half_dt;
        let expected_x = initial_positions_x[i] + v_half * dt;
        let expected_v = v_half + a * half_dt;
        assert_eq!(result.positions_x[i], expected_x, "particle {i} position");
        assert_eq!(result.velocities_x[i], expected_v, "particle {i} velocity");
    }
}

// rq-e8c21a03
#[test]
fn dt_zero_leaves_state_unchanged_for_kick_drift() {
    let device = init_device().expect("init_device");
    let sim_box = SimulationBox::new_orthorhombic(1.0e6, 1.0e6, 1.0e6).unwrap();
    let state = diverse_state(8);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).expect("buffers");
    let snapshot = snapshot_via_download(&buffers);

    vv_kick_drift(&mut buffers, &sim_box, 0.0).expect("kick_drift");
    let result = snapshot_via_download(&buffers);

    assert_eq!(result.positions_x, snapshot.positions_x);
    assert_eq!(result.positions_y, snapshot.positions_y);
    assert_eq!(result.positions_z, snapshot.positions_z);
    assert_eq!(result.velocities_x, snapshot.velocities_x);
    assert_eq!(result.velocities_y, snapshot.velocities_y);
    assert_eq!(result.velocities_z, snapshot.velocities_z);
    assert_eq!(result.forces_x, snapshot.forces_x);
    assert_eq!(result.forces_y, snapshot.forces_y);
    assert_eq!(result.forces_z, snapshot.forces_z);
    assert_eq!(result.masses, snapshot.masses);
    assert_eq!(result.particle_ids, snapshot.particle_ids);
}

// rq-d28737dd
#[test]
fn dt_zero_leaves_state_unchanged_for_kick() {
    let device = init_device().expect("init_device");
    let state = diverse_state(8);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).expect("buffers");
    let snapshot = snapshot_via_download(&buffers);

    vv_kick(&mut buffers, 0.0).expect("kick");
    let result = snapshot_via_download(&buffers);

    assert_eq!(result.positions_x, snapshot.positions_x);
    assert_eq!(result.positions_y, snapshot.positions_y);
    assert_eq!(result.positions_z, snapshot.positions_z);
    assert_eq!(result.velocities_x, snapshot.velocities_x);
    assert_eq!(result.velocities_y, snapshot.velocities_y);
    assert_eq!(result.velocities_z, snapshot.velocities_z);
    assert_eq!(result.forces_x, snapshot.forces_x);
    assert_eq!(result.forces_y, snapshot.forces_y);
    assert_eq!(result.forces_z, snapshot.forces_z);
    assert_eq!(result.masses, snapshot.masses);
    assert_eq!(result.particle_ids, snapshot.particle_ids);
}

// rq-1e2d749d
#[test]
fn vv_kick_drift_on_empty_state_is_noop() {
    let device = init_device().expect("init_device");
    let sim_box = SimulationBox::new_orthorhombic(1.0e6, 1.0e6, 1.0e6).unwrap();
    let state = ParticleState::new(
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![0u32; 0],
        None,
            None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).expect("buffers");
    vv_kick_drift(&mut buffers, &sim_box, 0.1).expect("kick_drift");
    assert_eq!(buffers.particle_count(), 0);
    assert_eq!(buffers.positions_x.len(), 0);
}

// rq-386cfae3
#[test]
fn vv_kick_on_empty_state_is_noop() {
    let device = init_device().expect("init_device");
    let state = ParticleState::new(
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![0u32; 0],
        None,
            None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).expect("buffers");
    vv_kick(&mut buffers, 0.1).expect("kick");
    assert_eq!(buffers.particle_count(), 0);
    assert_eq!(buffers.velocities_x.len(), 0);
}

// rq-a93a5b14
#[test]
fn block_non_aligned_particle_count_is_handled() {
    let device = init_device().expect("init_device");
    let sim_box = SimulationBox::new_orthorhombic(1.0e6, 1.0e6, 1.0e6).unwrap();
    let n = 1000;
    let positions_x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let state = ParticleState::new(
        positions_x.clone(),
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        vec![0u32; n],
        None,
            None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).expect("buffers");
    let snapshot = snapshot_via_download(&buffers);

    vv_kick_drift(&mut buffers, &sim_box, 0.1).expect("kick_drift");
    let result = snapshot_via_download(&buffers);

    for i in 0..n {
        assert_eq!(
            result.positions_x[i],
            positions_x[i] + 0.1_f32,
            "positions_x[{i}]"
        );
    }
    assert_eq!(result.positions_y, snapshot.positions_y);
    assert_eq!(result.positions_z, snapshot.positions_z);
    assert_eq!(result.velocities_x, snapshot.velocities_x);
    assert_eq!(result.velocities_y, snapshot.velocities_y);
    assert_eq!(result.velocities_z, snapshot.velocities_z);
    assert_eq!(result.forces_x, snapshot.forces_x);
    assert_eq!(result.forces_y, snapshot.forces_y);
    assert_eq!(result.forces_z, snapshot.forces_z);
    assert_eq!(result.masses, snapshot.masses);
    assert_eq!(result.particle_ids, snapshot.particle_ids);
}

// rq-7dfa14cf
#[test]
fn vv_kick_drift_does_not_modify_forces_or_masses() {
    let device = init_device().expect("init_device");
    let sim_box = SimulationBox::new_orthorhombic(1.0e6, 1.0e6, 1.0e6).unwrap();
    let state = diverse_state(4);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).expect("buffers");
    let snapshot = snapshot_via_download(&buffers);

    vv_kick_drift(&mut buffers, &sim_box, 0.1).expect("kick_drift");
    let result = snapshot_via_download(&buffers);

    assert_eq!(result.forces_x, snapshot.forces_x);
    assert_eq!(result.forces_y, snapshot.forces_y);
    assert_eq!(result.forces_z, snapshot.forces_z);
    assert_eq!(result.masses, snapshot.masses);
}

// rq-f721b7a1
#[test]
fn vv_kick_does_not_modify_forces_masses_or_positions() {
    let device = init_device().expect("init_device");
    let state = diverse_state(4);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).expect("buffers");
    let snapshot = snapshot_via_download(&buffers);

    vv_kick(&mut buffers, 0.1).expect("kick");
    let result = snapshot_via_download(&buffers);

    assert_eq!(result.forces_x, snapshot.forces_x);
    assert_eq!(result.forces_y, snapshot.forces_y);
    assert_eq!(result.forces_z, snapshot.forces_z);
    assert_eq!(result.masses, snapshot.masses);
    assert_eq!(result.positions_x, snapshot.positions_x);
    assert_eq!(result.positions_y, snapshot.positions_y);
    assert_eq!(result.positions_z, snapshot.positions_z);
}

// rq-37e6f318
#[test]
fn two_independent_runs_produce_byte_identical_outputs() {
    let device = init_device().expect("init_device");
    let sim_box = SimulationBox::new_orthorhombic(1.0e6, 1.0e6, 1.0e6).unwrap();
    let state = diverse_state(128);

    let mut buffers_a = ParticleBuffers::new(device.clone(), &state).expect("buffers a");
    vv_kick_drift(&mut buffers_a, &sim_box, 0.01).expect("kick_drift a");
    vv_kick(&mut buffers_a, 0.01).expect("kick a");
    let result_a = snapshot_via_download(&buffers_a);

    let mut buffers_b = ParticleBuffers::new(device.clone(), &state).expect("buffers b");
    vv_kick_drift(&mut buffers_b, &sim_box, 0.01).expect("kick_drift b");
    vv_kick(&mut buffers_b, 0.01).expect("kick b");
    let result_b = snapshot_via_download(&buffers_b);

    assert_eq!(result_a.positions_x, result_b.positions_x);
    assert_eq!(result_a.positions_y, result_b.positions_y);
    assert_eq!(result_a.positions_z, result_b.positions_z);
    assert_eq!(result_a.velocities_x, result_b.velocities_x);
    assert_eq!(result_a.velocities_y, result_b.velocities_y);
    assert_eq!(result_a.velocities_z, result_b.velocities_z);
    assert_eq!(result_a.forces_x, result_b.forces_x);
    assert_eq!(result_a.forces_y, result_b.forces_y);
    assert_eq!(result_a.forces_z, result_b.forces_z);
    assert_eq!(result_a.masses, result_b.masses);
    assert_eq!(result_a.particle_ids, result_b.particle_ids);
}

// rq-8e47334c
#[test]
fn nan_force_propagates_to_velocity_and_position() {
    let device = init_device().expect("init_device");
    let sim_box = SimulationBox::new_orthorhombic(1.0e6, 1.0e6, 1.0e6).unwrap();
    let mut state = ParticleState::new(
        vec![1.0],
        vec![2.0],
        vec![3.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![1.0],
        vec![0u32; 1],
        None,
            None,
    )
    .unwrap();
    state.forces_x = vec![f32::NAN];
    state.forces_y = vec![0.0];
    state.forces_z = vec![0.0];
    let mut buffers = ParticleBuffers::new(device.clone(), &state).expect("buffers");

    vv_kick_drift(&mut buffers, &sim_box, 0.1).expect("kick_drift");
    let result = snapshot_via_download(&buffers);

    assert!(result.velocities_x[0].is_nan());
    assert!(result.positions_x[0].is_nan());
    assert_eq!(result.velocities_y[0], 0.0);
    assert_eq!(result.velocities_z[0], 0.0);
    assert_eq!(result.positions_y[0], 2.0);
    assert_eq!(result.positions_z[0], 3.0);
}

// rq-b2d67b57
#[test]
fn forward_then_negated_kick_drift_returns_free_particle_to_origin() {
    let device = init_device().expect("init_device");
    let sim_box = SimulationBox::new_orthorhombic(1.0e6, 1.0e6, 1.0e6).unwrap();
    // Free particles: F=0, m=1, arbitrary nonzero positions and velocities.
    let positions_x = vec![1.0_f32, -2.0, 3.5, 0.25];
    let positions_y = vec![0.5_f32, 1.25, -0.75, 2.0];
    let positions_z = vec![-1.5_f32, 0.0, 0.5, -0.25];
    let velocities_x = vec![0.1_f32, -0.2, 0.05, 0.3];
    let velocities_y = vec![0.4_f32, 0.05, -0.15, 0.0];
    let velocities_z = vec![-0.1_f32, 0.2, 0.1, -0.05];
    let state = ParticleState::new(
        positions_x.clone(),
        positions_y.clone(),
        positions_z.clone(),
        velocities_x.clone(),
        velocities_y.clone(),
        velocities_z.clone(),
        vec![1.0; 4],
        vec![0u32; 4],
        None,
            None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).expect("buffers");

    vv_kick_drift(&mut buffers, &sim_box, 0.1).expect("forward");
    vv_kick_drift(&mut buffers, &sim_box, -0.1).expect("reverse");
    let result = snapshot_via_download(&buffers);

    let tol = 1e-6_f32;
    for i in 0..4 {
        assert!((result.positions_x[i] - positions_x[i]).abs() <= tol);
        assert!((result.positions_y[i] - positions_y[i]).abs() <= tol);
        assert!((result.positions_z[i] - positions_z[i]).abs() <= tol);
        assert!((result.velocities_x[i] - velocities_x[i]).abs() <= tol);
        assert!((result.velocities_y[i] - velocities_y[i]).abs() <= tol);
        assert!((result.velocities_z[i] - velocities_z[i]).abs() <= tol);
    }
}

// --- Image-flag wrap ---

#[test]
fn vv_kick_drift_wraps_positions_back_into_primary_image() {
    let device = init_device().expect("init_device");
    let sim_box = SimulationBox::new_orthorhombic(10.0, 10.0, 10.0).unwrap();
    let state = ParticleState::new(
        vec![4.9_f32],
        vec![0.0],
        vec![0.0],
        vec![2.0_f32],
        vec![0.0],
        vec![0.0],
        vec![1.0],
        vec![0u32; 1],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    vv_kick_drift(&mut buffers, &sim_box, 0.1).unwrap();
    let r = snapshot_via_download(&buffers);
    assert!((r.positions_x[0] - (-4.9_f32)).abs() < 1e-5);
    assert_eq!(r.images_x[0], 1);
    assert_eq!(r.images_y[0], 0);
    assert_eq!(r.images_z[0], 0);
}

#[test]
fn vv_kick_drift_wraps_in_negative_x() {
    let device = init_device().expect("init_device");
    let sim_box = SimulationBox::new_orthorhombic(10.0, 10.0, 10.0).unwrap();
    let state = ParticleState::new(
        vec![-4.9_f32],
        vec![0.0],
        vec![0.0],
        vec![-2.0_f32],
        vec![0.0],
        vec![0.0],
        vec![1.0],
        vec![0u32; 1],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    vv_kick_drift(&mut buffers, &sim_box, 0.1).unwrap();
    let r = snapshot_via_download(&buffers);
    assert!((r.positions_x[0] - 4.9_f32).abs() < 1e-5);
    assert_eq!(r.images_x[0], -1);
}

#[test]
fn vv_kick_drift_handles_multi_period_crossings() {
    let device = init_device().expect("init_device");
    let sim_box = SimulationBox::new_orthorhombic(10.0, 10.0, 10.0).unwrap();
    let state = ParticleState::new(
        vec![0.0_f32],
        vec![0.0],
        vec![0.0],
        vec![250.0_f32],
        vec![0.0],
        vec![0.0],
        vec![1.0],
        vec![0u32; 1],
        None,
        Some((vec![7], vec![0], vec![0])),
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    vv_kick_drift(&mut buffers, &sim_box, 0.1).unwrap();
    let r = snapshot_via_download(&buffers);
    assert!(r.positions_x[0] >= -5.0 && r.positions_x[0] < 5.0);
    // Started at image 7, position 0 (unwrapped: 70). After drift +25, unwrapped
    // = 95. Wrap into [-5, 5): floor((25 + 5) / 10) = 3, so position = -5.0 and
    // image = 7 + 3 = 10.
    assert_eq!(r.images_x[0], 10);
    let unwrapped = r.positions_x[0] + (r.images_x[0] as f32) * 10.0;
    assert!((unwrapped - 95.0_f32).abs() < 1e-4);
}

#[test]
fn vv_kick_does_not_modify_image_flags() {
    let device = init_device().expect("init_device");
    let n = 4;
    let state = ParticleState::new(
        vec![0.1_f32, 0.2, 0.3, 0.4],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.5; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        vec![0u32; n],
        None,
        Some((vec![1, -2, 3, 0], vec![0, 1, -1, 4], vec![2, 0, -3, 1])),
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    vv_kick(&mut buffers, 0.1).unwrap();
    let r = snapshot_via_download(&buffers);
    assert_eq!(r.images_x, vec![1, -2, 3, 0]);
    assert_eq!(r.images_y, vec![0, 1, -1, 4]);
    assert_eq!(r.images_z, vec![2, 0, -3, 1]);
}
