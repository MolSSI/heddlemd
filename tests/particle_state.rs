use cudarc::driver::DeviceSlice;
use heddle_md::gpu::{ParticleBuffers, init_device};
use heddle_md::state::{ParticleState, ParticleStateError};

fn make_state_with_values(
    n: usize,
    seed: f32,
    ids: Option<Vec<u32>>,
) -> ParticleState {
    let positions_x: Vec<f32> = (0..n).map(|i| seed + i as f32 + 0.10).collect();
    let positions_y: Vec<f32> = (0..n).map(|i| seed + i as f32 + 0.20).collect();
    let positions_z: Vec<f32> = (0..n).map(|i| seed + i as f32 + 0.30).collect();
    let velocities_x: Vec<f32> = (0..n).map(|i| seed + i as f32 + 0.40).collect();
    let velocities_y: Vec<f32> = (0..n).map(|i| seed + i as f32 + 0.50).collect();
    let velocities_z: Vec<f32> = (0..n).map(|i| seed + i as f32 + 0.60).collect();
    let masses: Vec<f32> = (0..n).map(|i| seed + i as f32 + 0.70).collect();
    ParticleState::new(
        positions_x,
        positions_y,
        positions_z,
        velocities_x,
        velocities_y,
        velocities_z,
        masses,
        vec![0.0_f32; n],
        vec![0u32; n],
        ids,
            None,
    )
    .expect("make_state_with_values: ParticleState::new should succeed")
}

fn assert_states_equal(a: &ParticleState, b: &ParticleState) {
    assert_eq!(a.positions_x, b.positions_x, "positions_x mismatch");
    assert_eq!(a.positions_y, b.positions_y, "positions_y mismatch");
    assert_eq!(a.positions_z, b.positions_z, "positions_z mismatch");
    assert_eq!(a.velocities_x, b.velocities_x, "velocities_x mismatch");
    assert_eq!(a.velocities_y, b.velocities_y, "velocities_y mismatch");
    assert_eq!(a.velocities_z, b.velocities_z, "velocities_z mismatch");
    assert_eq!(a.forces_x, b.forces_x, "forces_x mismatch");
    assert_eq!(a.forces_y, b.forces_y, "forces_y mismatch");
    assert_eq!(a.forces_z, b.forces_z, "forces_z mismatch");
    assert_eq!(a.masses, b.masses, "masses mismatch");
    assert_eq!(a.particle_ids, b.particle_ids, "particle_ids mismatch");
}

fn fill_with_garbage(state: &mut ParticleState) {
    let n = state.particle_count();
    for i in 0..n {
        state.positions_x[i] = -1234.5;
        state.positions_y[i] = -2345.6;
        state.positions_z[i] = -3456.7;
        state.velocities_x[i] = -4567.8;
        state.velocities_y[i] = -5678.9;
        state.velocities_z[i] = -6789.0;
        state.forces_x[i] = -7890.1;
        state.forces_y[i] = -8901.2;
        state.forces_z[i] = -9012.3;
        state.masses[i] = -1023.4;
        state.particle_ids[i] = 999_000 + i as u32;
    }
}

// --- Construction ---

#[test] // rq-81f4ec9d
fn construct_with_matching_arrays_and_default_ids() {
    let n = 4;
    let positions_x: Vec<f32> = vec![0.0; n];
    let positions_y: Vec<f32> = vec![0.0; n];
    let positions_z: Vec<f32> = vec![0.0; n];
    let velocities_x: Vec<f32> = vec![0.0; n];
    let velocities_y: Vec<f32> = vec![0.0; n];
    let velocities_z: Vec<f32> = vec![0.0; n];
    let masses: Vec<f32> = vec![0.0; n];

    let state = ParticleState::new(
        positions_x,
        positions_y,
        positions_z,
        velocities_x,
        velocities_y,
        velocities_z,
        masses,
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
            None,
    )
    .expect("construction should succeed");

    assert_eq!(state.particle_count(), 4);
    assert_eq!(state.particle_ids, vec![0u32, 1, 2, 3]);
    assert_eq!(state.forces_x, vec![0.0f32; 4]);
    assert_eq!(state.forces_y, vec![0.0f32; 4]);
    assert_eq!(state.forces_z, vec![0.0f32; 4]);
}

#[test] // rq-2bbc4121
fn construct_with_matching_arrays_and_explicit_unique_ids() {
    let n = 3;
    let state = ParticleState::new(
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0_f32; n],
        vec![0u32; n],
        Some(vec![10, 20, 30]),
            None,
    )
    .expect("construction should succeed");
    assert_eq!(state.particle_ids, vec![10u32, 20, 30]);
}

#[test] // rq-c22483b4
fn construct_an_empty_state() {
    let state = ParticleState::new(
        vec![],
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
    .expect("construction should succeed");
    assert_eq!(state.particle_count(), 0);
    assert!(state.positions_x.is_empty());
    assert!(state.positions_y.is_empty());
    assert!(state.positions_z.is_empty());
    assert!(state.velocities_x.is_empty());
    assert!(state.velocities_y.is_empty());
    assert!(state.velocities_z.is_empty());
    assert!(state.forces_x.is_empty());
    assert!(state.forces_y.is_empty());
    assert!(state.forces_z.is_empty());
    assert!(state.masses.is_empty());
    assert!(state.particle_ids.is_empty());
}

#[test] // rq-91aa1f1c
fn construct_an_empty_state_with_explicit_empty_ids() {
    let state = ParticleState::new(
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![0u32; 0],
        Some(vec![]),
            None,
    )
    .expect("construction should succeed");
    assert_eq!(state.particle_count(), 0);
}

#[test] // rq-1e8b3c79
fn reject_when_positions_y_has_wrong_length() {
    let err = ParticleState::new(
        vec![0.0; 4],
        vec![0.0; 3],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0_f32; 4],
        vec![0u32; 4],
        None,
            None,
    )
    .expect_err("expected LengthMismatch");
    match err {
        ParticleStateError::LengthMismatch {
            array,
            expected,
            actual,
        } => {
            assert_eq!(array, "positions_y");
            assert_eq!(expected, 4);
            assert_eq!(actual, 3);
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test] // rq-ce89d4a4
fn reject_when_masses_has_wrong_length() {
    let err = ParticleState::new(
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 5],
        vec![0.0_f32; 5],
        vec![0u32; 5],
        None,
            None,
    )
    .expect_err("expected LengthMismatch");
    match err {
        ParticleStateError::LengthMismatch {
            array,
            expected,
            actual,
        } => {
            assert_eq!(array, "masses");
            assert_eq!(expected, 4);
            assert_eq!(actual, 5);
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test] // rq-391cb266
fn reject_when_explicit_ids_have_wrong_length() {
    let err = ParticleState::new(
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0_f32; 4],
        vec![0u32; 4],
        Some(vec![0, 1]),
            None,
    )
    .expect_err("expected LengthMismatch");
    match err {
        ParticleStateError::LengthMismatch {
            array,
            expected,
            actual,
        } => {
            assert_eq!(array, "particle_ids");
            assert_eq!(expected, 4);
            assert_eq!(actual, 2);
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test] // rq-4b38148c
fn reject_duplicate_explicit_ids() {
    let err = ParticleState::new(
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0_f32; 4],
        vec![0u32; 4],
        Some(vec![7, 1, 7, 3]),
            None,
    )
    .expect_err("expected DuplicateParticleId");
    match err {
        ParticleStateError::DuplicateParticleId(id) => assert_eq!(id, 7),
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test] // rq-9447dfcf
fn nan_values_accepted_at_construction() {
    let mut positions_x = vec![0.0f32; 4];
    positions_x[0] = f32::NAN;
    let state = ParticleState::new(
        positions_x,
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0_f32; 4],
        vec![0u32; 4],
        None,
            None,
    )
    .expect("construction should succeed even with NaN");
    assert!(state.positions_x[0].is_nan());
}

// --- ParticleBuffers allocation and upload ---

#[test] // rq-0ffccee0
fn allocate_device_buffers_and_perform_initial_upload() {
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let state = make_state_with_values(4, 1.0, None);
    let buffers = ParticleBuffers::new(&gpu, &state)
        .expect("ParticleBuffers::new should succeed");
    assert_eq!(buffers.particle_count(), 4);
    assert_eq!(buffers.positions_x.len(), 4);
    assert_eq!(buffers.positions_y.len(), 4);
    assert_eq!(buffers.positions_z.len(), 4);
    assert_eq!(buffers.velocities_x.len(), 4);
    assert_eq!(buffers.velocities_y.len(), 4);
    assert_eq!(buffers.velocities_z.len(), 4);
    assert_eq!(buffers.forces_x.len(), 4);
    assert_eq!(buffers.forces_y.len(), 4);
    assert_eq!(buffers.forces_z.len(), 4);
    assert_eq!(buffers.masses.len(), 4);
    assert_eq!(buffers.particle_ids.len(), 4);

    assert_eq!(device.dtoh_sync_copy(&buffers.positions_x).unwrap(), state.positions_x);
    assert_eq!(device.dtoh_sync_copy(&buffers.positions_y).unwrap(), state.positions_y);
    assert_eq!(device.dtoh_sync_copy(&buffers.positions_z).unwrap(), state.positions_z);
    assert_eq!(device.dtoh_sync_copy(&buffers.velocities_x).unwrap(), state.velocities_x);
    assert_eq!(device.dtoh_sync_copy(&buffers.velocities_y).unwrap(), state.velocities_y);
    assert_eq!(device.dtoh_sync_copy(&buffers.velocities_z).unwrap(), state.velocities_z);
    assert_eq!(device.dtoh_sync_copy(&buffers.forces_x).unwrap(), state.forces_x);
    assert_eq!(device.dtoh_sync_copy(&buffers.forces_y).unwrap(), state.forces_y);
    assert_eq!(device.dtoh_sync_copy(&buffers.forces_z).unwrap(), state.forces_z);
    assert_eq!(device.dtoh_sync_copy(&buffers.masses).unwrap(), state.masses);
    assert_eq!(device.dtoh_sync_copy(&buffers.particle_ids).unwrap(), state.particle_ids);
}

#[test] // rq-c8aa7417
fn allocate_device_buffers_from_an_empty_state() {
    let gpu = init_device().expect("init_device");
    let state = ParticleState::new(
        vec![],
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
    let buffers = ParticleBuffers::new(&gpu, &state)
        .expect("ParticleBuffers::new should succeed");
    assert_eq!(buffers.particle_count(), 0);
    assert_eq!(buffers.positions_x.len(), 0);
    assert_eq!(buffers.positions_y.len(), 0);
    assert_eq!(buffers.positions_z.len(), 0);
    assert_eq!(buffers.velocities_x.len(), 0);
    assert_eq!(buffers.velocities_y.len(), 0);
    assert_eq!(buffers.velocities_z.len(), 0);
    assert_eq!(buffers.forces_x.len(), 0);
    assert_eq!(buffers.forces_y.len(), 0);
    assert_eq!(buffers.forces_z.len(), 0);
    assert_eq!(buffers.masses.len(), 0);
    assert_eq!(buffers.particle_ids.len(), 0);
}

#[test] // rq-780d68ea
fn reject_particle_buffers_new_when_host_array_length_inconsistent() {
    let gpu = init_device().expect("init_device");
    let mut state = make_state_with_values(4, 1.0, None);
    state.velocities_z.truncate(3);
    let err = ParticleBuffers::new(&gpu, &state)
        .expect_err("expected LengthMismatch on velocities_z");
    match err {
        ParticleStateError::LengthMismatch {
            array,
            expected,
            actual,
        } => {
            assert_eq!(array, "velocities_z");
            assert_eq!(expected, 4);
            assert_eq!(actual, 3);
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test] // rq-4d226dff
fn re_upload_after_host_side_mutation() {
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let mut state = make_state_with_values(4, 1.0, None);
    let mut buffers = ParticleBuffers::new(&gpu, &state)
        .expect("ParticleBuffers::new");

    let new_positions_x: Vec<f32> = vec![100.0, 101.0, 102.0, 103.0];
    state.positions_x = new_positions_x.clone();

    buffers.upload(&state).expect("upload should succeed");

    assert_eq!(device.dtoh_sync_copy(&buffers.positions_x).unwrap(), new_positions_x);
    assert_eq!(device.dtoh_sync_copy(&buffers.positions_y).unwrap(), state.positions_y);
    assert_eq!(device.dtoh_sync_copy(&buffers.positions_z).unwrap(), state.positions_z);
    assert_eq!(device.dtoh_sync_copy(&buffers.velocities_x).unwrap(), state.velocities_x);
    assert_eq!(device.dtoh_sync_copy(&buffers.velocities_y).unwrap(), state.velocities_y);
    assert_eq!(device.dtoh_sync_copy(&buffers.velocities_z).unwrap(), state.velocities_z);
    assert_eq!(device.dtoh_sync_copy(&buffers.forces_x).unwrap(), state.forces_x);
    assert_eq!(device.dtoh_sync_copy(&buffers.forces_y).unwrap(), state.forces_y);
    assert_eq!(device.dtoh_sync_copy(&buffers.forces_z).unwrap(), state.forces_z);
    assert_eq!(device.dtoh_sync_copy(&buffers.masses).unwrap(), state.masses);
    assert_eq!(device.dtoh_sync_copy(&buffers.particle_ids).unwrap(), state.particle_ids);
}

#[test] // rq-9ff4fd10
fn reject_upload_when_host_particle_count_differs_from_buffers() {
    let gpu = init_device().expect("init_device");
    let state_a = make_state_with_values(4, 1.0, None);
    let state_b = make_state_with_values(5, 2.0, None);
    let mut buffers = ParticleBuffers::new(&gpu, &state_a)
        .expect("ParticleBuffers::new");
    let err = buffers.upload(&state_b).expect_err("expected LengthMismatch");
    match err {
        ParticleStateError::LengthMismatch {
            array,
            expected,
            actual,
        } => {
            assert_eq!(array, "positions_x");
            assert_eq!(expected, 4);
            assert_eq!(actual, 5);
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test] // rq-b4bb7096
fn reject_upload_when_host_array_drifts_in_length() {
    let gpu = init_device().expect("init_device");
    let mut state = make_state_with_values(4, 1.0, None);
    let mut buffers = ParticleBuffers::new(&gpu, &state)
        .expect("ParticleBuffers::new");
    state.forces_y.truncate(3);
    let err = buffers.upload(&state).expect_err("expected LengthMismatch");
    match err {
        ParticleStateError::LengthMismatch {
            array,
            expected,
            actual,
        } => {
            assert_eq!(array, "forces_y");
            assert_eq!(expected, 4);
            assert_eq!(actual, 3);
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

// --- Download ---

#[test] // rq-39a260b5
fn download_into_source_state_preserves_values() {
    let gpu = init_device().expect("init_device");
    let mut state = make_state_with_values(4, 1.0, None);
    let snapshot = state.clone();
    let buffers = ParticleBuffers::new(&gpu, &state).expect("ParticleBuffers::new");
    state.download_from(&buffers).expect("download should succeed");
    assert_states_equal(&state, &snapshot);
}

#[test] // rq-9594de53
fn download_overwrites_in_place_after_host_mutation() {
    let gpu = init_device().expect("init_device");
    let mut state = make_state_with_values(4, 1.0, None);
    let original = state.clone();
    let buffers = ParticleBuffers::new(&gpu, &state).expect("ParticleBuffers::new");

    fill_with_garbage(&mut state);

    state.download_from(&buffers).expect("download should succeed");
    assert_states_equal(&state, &original);
}

#[test] // rq-f4ebf12a
fn download_reflects_device_side_changes_pushed_by_interim_re_upload() {
    let gpu = init_device().expect("init_device");
    let mut state_a = make_state_with_values(4, 1.0, None);
    let mut buffers = ParticleBuffers::new(&gpu, &state_a)
        .expect("ParticleBuffers::new");

    let state_b = make_state_with_values(4, 100.0, Some(vec![100, 101, 102, 103]));
    let snapshot_b = state_b.clone();
    buffers.upload(&state_b).expect("upload should succeed");

    fill_with_garbage(&mut state_a);

    state_a.download_from(&buffers).expect("download should succeed");
    assert_states_equal(&state_a, &snapshot_b);
}

#[test] // rq-7ab80063
fn reject_download_when_host_particle_count_differs_from_buffers() {
    let gpu = init_device().expect("init_device");
    let state_a = make_state_with_values(4, 1.0, None);
    let mut state_b = make_state_with_values(5, 2.0, None);
    let buffers = ParticleBuffers::new(&gpu, &state_a)
        .expect("ParticleBuffers::new");
    let err = state_b.download_from(&buffers).expect_err("expected LengthMismatch");
    match err {
        ParticleStateError::LengthMismatch {
            array,
            expected,
            actual,
        } => {
            assert_eq!(array, "positions_x");
            assert_eq!(expected, 4);
            assert_eq!(actual, 5);
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test] // rq-1bdbd71e
fn reject_download_when_host_array_drifts_in_length() {
    let gpu = init_device().expect("init_device");
    let state = make_state_with_values(4, 1.0, None);
    let buffers = ParticleBuffers::new(&gpu, &state)
        .expect("ParticleBuffers::new");
    let mut other = make_state_with_values(4, 5.0, None);
    other.masses = vec![0.0; 6];
    let err = other.download_from(&buffers).expect_err("expected LengthMismatch");
    match err {
        ParticleStateError::LengthMismatch {
            array,
            expected,
            actual,
        } => {
            assert_eq!(array, "masses");
            assert_eq!(expected, 4);
            assert_eq!(actual, 6);
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test] // rq-58254790
fn download_from_empty_buffers_into_empty_state() {
    let gpu = init_device().expect("init_device");
    let source = ParticleState::new(
        vec![],
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
    let buffers = ParticleBuffers::new(&gpu, &source)
        .expect("ParticleBuffers::new");
    let mut sink = ParticleState::new(
        vec![],
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
    sink.download_from(&buffers).expect("download should succeed");
    assert_eq!(sink.particle_count(), 0);
}

#[test] // rq-790c1f86
fn reject_when_type_indices_has_wrong_length() {
    let err = ParticleState::new(
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![1.0; 4],
        vec![0.0_f32; 4],
        vec![0u32; 3],
        None,
            None,
    )
    .expect_err("expected LengthMismatch on type_indices");
    match err {
        ParticleStateError::LengthMismatch {
            array,
            expected,
            actual,
        } => {
            assert_eq!(array, "type_indices");
            assert_eq!(expected, 4);
            assert_eq!(actual, 3);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn type_indices_round_trip_through_particle_buffers() {
    let gpu = init_device().expect("init_device");
    let state = ParticleState::new(
        vec![0.0_f32, 1.0, 2.0],
        vec![0.0_f32; 3],
        vec![0.0_f32; 3],
        vec![0.0_f32; 3],
        vec![0.0_f32; 3],
        vec![0.0_f32; 3],
        vec![1.0_f32; 3],
        vec![0.0_f32; 3],
        vec![0u32, 2, 1],
        None,
            None,
    )
    .unwrap();
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let host = gpu.device.dtoh_sync_copy(&buffers.type_indices).unwrap();
    assert_eq!(host, vec![0u32, 2, 1]);
    // Download path mirrors values back into a sink state.
    let mut sink = ParticleState::new(
        vec![0.0_f32; 3],
        vec![0.0_f32; 3],
        vec![0.0_f32; 3],
        vec![0.0_f32; 3],
        vec![0.0_f32; 3],
        vec![0.0_f32; 3],
        vec![1.0_f32; 3],
        vec![0.0_f32; 3],
        vec![0u32; 3],
        None,
            None,
    )
    .unwrap();
    sink.download_from(&buffers).unwrap();
    assert_eq!(sink.type_indices, vec![0u32, 2, 1]);
}

#[test] // rq-0519e35c
fn new_state_zero_inits_potential_energies_and_virials() {
    let n = 4;
    let state = ParticleState::new(
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![1.0_f32; n],
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
            None,
    )
    .unwrap();
    assert_eq!(state.potential_energies, vec![0.0_f32; n]);
    assert_eq!(state.virials, vec![0.0_f32; n]);
}

#[test] // rq-9504346c
fn potential_energies_and_virials_round_trip_through_buffers() {
    let gpu = init_device().expect("init_device");
    let mut state = ParticleState::new(
        vec![0.0_f32, 1.0, 2.0, 3.0],
        vec![0.0_f32; 4],
        vec![0.0_f32; 4],
        vec![0.0_f32; 4],
        vec![0.0_f32; 4],
        vec![0.0_f32; 4],
        vec![1.0_f32; 4],
        vec![0.0_f32; 4],
        vec![0u32; 4],
        None,
            None,
    )
    .unwrap();
    state.potential_energies = vec![1.0_f32, -2.0, 3.5, 0.25];
    state.virials = vec![10.0_f32, 20.0, -30.0, 40.0];
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    state.potential_energies = vec![0.0_f32; 4];
    state.virials = vec![0.0_f32; 4];
    state.download_from(&buffers).unwrap();
    assert_eq!(state.potential_energies, vec![1.0_f32, -2.0, 3.5, 0.25]);
    assert_eq!(state.virials, vec![10.0_f32, 20.0, -30.0, 40.0]);
}

// --- Image flags ---

#[test] // rq-6f897168
fn images_default_to_zero_when_none_passed() {
    let n = 4;
    let state = ParticleState::new(
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![1.0_f32; n],
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap();
    assert_eq!(state.images_x, vec![0_i32; n]);
    assert_eq!(state.images_y, vec![0_i32; n]);
    assert_eq!(state.images_z, vec![0_i32; n]);
}

#[test] // rq-2315b501
fn explicit_nonzero_images_stored_as_supplied() {
    let n = 3;
    let state = ParticleState::new(
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![1.0_f32; n],
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
        Some((vec![1, -2, 0], vec![0, 3, -1], vec![-4, 0, 5])),
    )
    .unwrap();
    assert_eq!(state.images_x, vec![1, -2, 0]);
    assert_eq!(state.images_y, vec![0, 3, -1]);
    assert_eq!(state.images_z, vec![-4, 0, 5]);
}

#[test] // rq-0705e380
fn reject_explicit_images_y_wrong_length() {
    let n = 4;
    let err = ParticleState::new(
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![1.0_f32; n],
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
        Some((vec![0; n], vec![0; 3], vec![0; n])),
    )
    .expect_err("expected LengthMismatch");
    match err {
        heddle_md::state::ParticleStateError::LengthMismatch {
            array,
            expected,
            actual,
        } => {
            assert_eq!(array, "images_y");
            assert_eq!(expected, 4);
            assert_eq!(actual, 3);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test] // rq-2c31aa6e
fn particle_buffers_carry_image_buffers() {
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let n = 4;
    let images_x = vec![1, -2, 3, 0];
    let images_y = vec![-7, 4, 0, 8];
    let images_z = vec![5, -3, 2, -1];
    let state = ParticleState::new(
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![1.0_f32; n],
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
        Some((images_x.clone(), images_y.clone(), images_z.clone())),
    )
    .unwrap();
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    assert_eq!(buffers.images_x.len(), n);
    assert_eq!(buffers.images_y.len(), n);
    assert_eq!(buffers.images_z.len(), n);
    let dx: Vec<i32> = device.dtoh_sync_copy(&buffers.images_x).unwrap();
    let dy: Vec<i32> = device.dtoh_sync_copy(&buffers.images_y).unwrap();
    let dz: Vec<i32> = device.dtoh_sync_copy(&buffers.images_z).unwrap();
    assert_eq!(dx, images_x);
    assert_eq!(dy, images_y);
    assert_eq!(dz, images_z);
}

#[test] // rq-cef2288f
fn reject_upload_when_images_x_has_wrong_length() {
    let gpu = init_device().expect("init_device");
    let n = 4;
    let state = ParticleState::new(
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![1.0_f32; n],
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut bad = state.clone();
    bad.images_x.truncate(3);
    let err = buffers.upload(&bad).expect_err("expected LengthMismatch");
    match err {
        heddle_md::state::ParticleStateError::LengthMismatch {
            array,
            expected,
            actual,
        } => {
            assert_eq!(array, "images_x");
            assert_eq!(expected, 4);
            assert_eq!(actual, 3);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test] // rq-7fd19f00
fn reject_when_charges_has_wrong_length() {
    let n = 4;
    let positions_x = vec![0.0_f32; n];
    let err = ParticleState::new(
        positions_x.clone(),
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![1.0_f32; n],
        vec![0.0_f32; 5], // charges wrong length
        vec![0u32; n],
        None,
        None,
    )
    .expect_err("expected LengthMismatch on charges");
    match err {
        ParticleStateError::LengthMismatch {
            array,
            expected,
            actual,
        } => {
            assert_eq!(array, "charges");
            assert_eq!(expected, 4);
            assert_eq!(actual, 5);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test] // rq-fdc02bdb
fn charges_round_trip_through_particle_buffers() {
    let gpu = init_device().unwrap();
    let n = 4;
    let charges = vec![1.602e-19_f32, -1.602e-19, 0.0, 3.2e-19];
    let mut state = ParticleState::new(
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![1.0_f32; n],
        charges.clone(),
        vec![0u32; n],
        None,
        None,
    )
    .unwrap();
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    // Zero on the host, download, and check it matches the device side.
    state.charges = vec![0.0; n];
    state.download_from(&buffers).unwrap();
    assert_eq!(state.charges, charges);
}
