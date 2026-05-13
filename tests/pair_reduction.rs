mod common;
use common::*;

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use dynamics::gpu::{PairBuffer, ParticleBuffers, init_device};
use dynamics::state::ParticleState;

fn zero_state(n: usize) -> ParticleState {
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
    )
    .expect("ParticleState::new")
}

fn zero_particle_buffers(device: Arc<CudaDevice>, n: usize) -> ParticleBuffers {
    ParticleBuffers::new(device, &zero_state(n)).expect("ParticleBuffers::new")
}

fn upload_counts(device: &Arc<CudaDevice>, counts: &[u32]) -> CudaSlice<u32> {
    device.htod_sync_copy(counts).expect("upload counts")
}

fn upload_pair_x(pair: &mut PairBuffer, data: &[f32]) {
    let device = pair.device.clone();
    device
        .htod_sync_copy_into(data, &mut pair.pair_forces_x)
        .expect("upload pair_forces_x");
}

fn upload_pair_y(pair: &mut PairBuffer, data: &[f32]) {
    let device = pair.device.clone();
    device
        .htod_sync_copy_into(data, &mut pair.pair_forces_y)
        .expect("upload pair_forces_y");
}

fn upload_pair_z(pair: &mut PairBuffer, data: &[f32]) {
    let device = pair.device.clone();
    device
        .htod_sync_copy_into(data, &mut pair.pair_forces_z)
        .expect("upload pair_forces_z");
}

fn download_pair_x(pair: &PairBuffer) -> Vec<f32> {
    pair.device.dtoh_sync_copy(&pair.pair_forces_x).unwrap()
}

fn download_pair_y(pair: &PairBuffer) -> Vec<f32> {
    pair.device.dtoh_sync_copy(&pair.pair_forces_y).unwrap()
}

fn download_pair_z(pair: &PairBuffer) -> Vec<f32> {
    pair.device.dtoh_sync_copy(&pair.pair_forces_z).unwrap()
}

fn download_forces(buffers: &ParticleBuffers) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let device = buffers.device.clone();
    (
        device.dtoh_sync_copy(&buffers.forces_x).unwrap(),
        device.dtoh_sync_copy(&buffers.forces_y).unwrap(),
        device.dtoh_sync_copy(&buffers.forces_z).unwrap(),
    )
}

// --- PairBuffer construction ---

#[test] // rq-6fdefca0
fn pair_buffer_new_allocates_zero_initialised_buffers() {
    let device = init_device().expect("init_device");
    let pair = PairBuffer::new(device.clone(), 4, 8).expect("PairBuffer::new");
    assert_eq!(pair.particle_count(), 4);
    assert_eq!(pair.max_neighbors(), 8);
    assert_eq!(pair.pair_forces_x.len(), 32);
    assert_eq!(pair.pair_forces_y.len(), 32);
    assert_eq!(pair.pair_forces_z.len(), 32);
    assert_eq!(download_pair_x(&pair), vec![0.0_f32; 32]);
    assert_eq!(download_pair_y(&pair), vec![0.0_f32; 32]);
    assert_eq!(download_pair_z(&pair), vec![0.0_f32; 32]);
}

#[test] // rq-74e4bd02
fn pair_buffer_new_with_zero_particle_count() {
    let device = init_device().expect("init_device");
    let pair = PairBuffer::new(device.clone(), 0, 8).expect("PairBuffer::new");
    assert_eq!(pair.particle_count(), 0);
    assert_eq!(pair.pair_forces_x.len(), 0);
    assert_eq!(pair.pair_forces_y.len(), 0);
    assert_eq!(pair.pair_forces_z.len(), 0);
}

#[test] // rq-15e1e995
fn pair_buffer_new_with_zero_max_neighbors() {
    let device = init_device().expect("init_device");
    let pair = PairBuffer::new(device.clone(), 4, 0).expect("PairBuffer::new");
    assert_eq!(pair.max_neighbors(), 0);
    assert_eq!(pair.pair_forces_x.len(), 0);
    assert_eq!(pair.pair_forces_y.len(), 0);
    assert_eq!(pair.pair_forces_z.len(), 0);
}

// --- Module loading ---

#[test] // rq-a43552d5
fn init_device_loads_reduce_module() {
    let device = init_device().expect("init_device");
    assert!(device.has_func("reduce", "reduce_pair_forces"));
}

// --- Reduction correctness: trivial cases ---

#[test] // rq-2d051b0c
fn reduction_with_all_zero_counts_zeroes_forces() {
    let device = init_device().expect("init_device");
    let mut pair = PairBuffer::new(device.clone(), 4, 8).expect("PairBuffer::new");
    upload_pair_x(&mut pair, &vec![3.14_f32; 32]);
    upload_pair_y(&mut pair, &vec![2.71_f32; 32]);
    upload_pair_z(&mut pair, &vec![1.41_f32; 32]);

    let mut state = zero_state(4);
    state.forces_x = vec![10.0, 20.0, 30.0, 40.0];
    state.forces_y = vec![-1.0, -2.0, -3.0, -4.0];
    state.forces_z = vec![5.0, 6.0, 7.0, 8.0];
    let mut particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();

    let counts = upload_counts(&device, &[0u32, 0, 0, 0]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, fy, fz) = download_forces(&particle_buffers);
    assert_eq!(fx, vec![0.0_f32; 4]);
    assert_eq!(fy, vec![0.0_f32; 4]);
    assert_eq!(fz, vec![0.0_f32; 4]);
}

#[test] // rq-8ee33aa0
fn reduction_with_single_particle_single_neighbor() {
    let device = init_device().expect("init_device");
    let mut pair = PairBuffer::new(device.clone(), 1, 4).expect("PairBuffer::new");
    upload_pair_x(&mut pair, &[1.5_f32, 0.0, 0.0, 0.0]);
    upload_pair_y(&mut pair, &[-2.5_f32, 0.0, 0.0, 0.0]);
    upload_pair_z(&mut pair, &[0.75_f32, 0.0, 0.0, 0.0]);

    let mut particle_buffers = zero_particle_buffers(device.clone(), 1);
    let counts = upload_counts(&device, &[1u32]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, fy, fz) = download_forces(&particle_buffers);
    assert_eq!(fx, vec![1.5_f32]);
    assert_eq!(fy, vec![-2.5_f32]);
    assert_eq!(fz, vec![0.75_f32]);
}

// --- Reduction correctness: order and bounds ---

#[test] // rq-e950f4e6
fn reduction_sums_entries_left_to_right() {
    let device = init_device().expect("init_device");
    let mut pair = PairBuffer::new(device.clone(), 1, 4).expect("PairBuffer::new");
    upload_pair_x(&mut pair, &[1.0_f32, 2.0, 4.0, 999.0]);

    let mut particle_buffers = zero_particle_buffers(device.clone(), 1);
    let counts = upload_counts(&device, &[3u32]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, _, _) = download_forces(&particle_buffers);
    let expected = (1.0_f32 + 2.0_f32) + 4.0_f32;
    assert_eq!(fx[0], expected);
    // The slot at index 3 (value 999.0) must not be included.
    assert!(fx[0] != expected + 999.0);
}

#[test] // rq-78fc2fbb
fn reduction_ignores_slots_beyond_count() {
    let device = init_device().expect("init_device");
    let mut pair = PairBuffer::new(device.clone(), 1, 8).expect("PairBuffer::new");
    let mut data = vec![f32::INFINITY; 8];
    data[0] = 10.0;
    data[1] = 20.0;
    upload_pair_x(&mut pair, &data);

    let mut particle_buffers = zero_particle_buffers(device.clone(), 1);
    let counts = upload_counts(&device, &[2u32]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, _, _) = download_forces(&particle_buffers);
    assert_eq!(fx[0], 30.0_f32);
    assert!(fx[0].is_finite());
}

#[test] // rq-590dcd7e
fn reduction_at_full_max_neighbors_capacity() {
    let device = init_device().expect("init_device");
    let mut pair = PairBuffer::new(device.clone(), 1, 4).expect("PairBuffer::new");
    upload_pair_x(&mut pair, &[1.0_f32, 2.0, 3.0, 4.0]);

    let mut particle_buffers = zero_particle_buffers(device.clone(), 1);
    let counts = upload_counts(&device, &[4u32]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, _, _) = download_forces(&particle_buffers);
    let expected = ((1.0_f32 + 2.0_f32) + 3.0_f32) + 4.0_f32;
    assert_eq!(fx[0], expected);
}

// --- Reduction correctness: multiple particles ---

#[test] // rq-6808532e
fn per_particle_reduction_with_varying_counts() {
    let device = init_device().expect("init_device");
    let mut pair = PairBuffer::new(device.clone(), 3, 4).expect("PairBuffer::new");
    let pair_x: Vec<f32> = vec![
        // particle 0: count = 2
        1.0, 2.0, 100.0, 100.0,
        // particle 1: count = 1
        10.0, 100.0, 100.0, 100.0,
        // particle 2: count = 4
        0.5, 0.5, 0.5, 0.5,
    ];
    upload_pair_x(&mut pair, &pair_x);

    let mut particle_buffers = zero_particle_buffers(device.clone(), 3);
    let counts = upload_counts(&device, &[2u32, 1, 4]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, _, _) = download_forces(&particle_buffers);
    assert_eq!(fx[0], 3.0_f32);
    assert_eq!(fx[1], 10.0_f32);
    assert_eq!(fx[2], 2.0_f32);
}

// --- Empty state ---

#[test] // rq-493caf32
fn reduce_pair_forces_on_empty_state_is_noop() {
    let device = init_device().expect("init_device");
    let pair = PairBuffer::new(device.clone(), 0, 8).expect("PairBuffer::new");
    let mut particle_buffers = zero_particle_buffers(device.clone(), 0);
    let counts = upload_counts(&device, &[]);
    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
}

// --- Bounds handling ---

#[test] // rq-77e88745
fn block_non_aligned_particle_count_is_handled() {
    let device = init_device().expect("init_device");
    let n: usize = 1000;
    let max_neighbors: u32 = 2;
    let mut pair = PairBuffer::new(device.clone(), n, max_neighbors).expect("PairBuffer::new");
    let mut x_data = vec![0.0_f32; n * 2];
    for i in 0..n {
        x_data[i * 2] = i as f32;
        x_data[i * 2 + 1] = -(i as f32);
    }
    upload_pair_x(&mut pair, &x_data);

    let mut particle_buffers = zero_particle_buffers(device.clone(), n);
    let counts = upload_counts(&device, &vec![2u32; n]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, fy, fz) = download_forces(&particle_buffers);
    for i in 0..n {
        assert_eq!(fx[i], 0.0_f32, "fx[{i}]");
        assert_eq!(fy[i], 0.0_f32, "fy[{i}]");
        assert_eq!(fz[i], 0.0_f32, "fz[{i}]");
    }
}

// --- Side effects ---

#[test] // rq-f5299d6e
fn reduction_overwrites_prior_force_values() {
    let device = init_device().expect("init_device");
    let pair = PairBuffer::new(device.clone(), 4, 2).expect("PairBuffer::new");

    let mut state = zero_state(4);
    state.forces_x = vec![99.0, 88.0, 77.0, 66.0];
    let mut particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();

    let counts = upload_counts(&device, &[0u32, 0, 0, 0]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, _, _) = download_forces(&particle_buffers);
    assert_eq!(fx, vec![0.0_f32, 0.0, 0.0, 0.0]);
}

#[test] // rq-9b794cff
fn reduction_does_not_modify_pair_buffer() {
    let device = init_device().expect("init_device");
    let mut pair = PairBuffer::new(device.clone(), 4, 4).expect("PairBuffer::new");
    let pair_x: Vec<f32> = (0..16).map(|i| 0.1 + i as f32 * 0.5).collect();
    let pair_y: Vec<f32> = (0..16).map(|i| -0.2 + i as f32 * 0.25).collect();
    let pair_z: Vec<f32> = (0..16).map(|i| 0.3 + i as f32 * 0.125).collect();
    upload_pair_x(&mut pair, &pair_x);
    upload_pair_y(&mut pair, &pair_y);
    upload_pair_z(&mut pair, &pair_z);

    let snapshot_x = download_pair_x(&pair);
    let snapshot_y = download_pair_y(&pair);
    let snapshot_z = download_pair_z(&pair);

    let mut particle_buffers = zero_particle_buffers(device.clone(), 4);
    let counts = upload_counts(&device, &[3u32, 3, 3, 3]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");

    assert_eq!(download_pair_x(&pair), snapshot_x);
    assert_eq!(download_pair_y(&pair), snapshot_y);
    assert_eq!(download_pair_z(&pair), snapshot_z);
}

#[test] // rq-c9da25a7
fn reduction_does_not_modify_positions_velocities_masses() {
    let device = init_device().expect("init_device");
    let pair = PairBuffer::new(device.clone(), 4, 2).expect("PairBuffer::new");

    let state = ParticleState::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
        vec![0.1, 0.2, 0.3, 0.4],
        vec![-0.1, -0.2, -0.3, -0.4],
        vec![0.05, 0.1, 0.15, 0.2],
        vec![1.5, 2.5, 3.5, 4.5],
        vec![0u32; 4],
        Some(vec![100, 200, 300, 400]),
    )
    .unwrap();
    let mut particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();

    let counts = upload_counts(&device, &[0u32, 0, 0, 0]);
    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");

    let mut downloaded = state.clone();
    downloaded.download_from(&particle_buffers).unwrap();
    assert_eq!(downloaded.positions_x, state.positions_x);
    assert_eq!(downloaded.positions_y, state.positions_y);
    assert_eq!(downloaded.positions_z, state.positions_z);
    assert_eq!(downloaded.velocities_x, state.velocities_x);
    assert_eq!(downloaded.velocities_y, state.velocities_y);
    assert_eq!(downloaded.velocities_z, state.velocities_z);
    assert_eq!(downloaded.masses, state.masses);
    assert_eq!(downloaded.particle_ids, state.particle_ids);
}

#[test] // rq-eb3a65df
fn reduction_does_not_modify_neighbor_counts() {
    let device = init_device().expect("init_device");
    let pair = PairBuffer::new(device.clone(), 4, 2).expect("PairBuffer::new");
    let mut particle_buffers = zero_particle_buffers(device.clone(), 4);
    let counts = upload_counts(&device, &[0u32, 1, 2, 0]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");

    let downloaded: Vec<u32> = device.dtoh_sync_copy(&counts).unwrap();
    assert_eq!(downloaded, vec![0u32, 1, 2, 0]);
}

// --- Reproducibility ---

#[test] // rq-b4f18ea1
fn two_independent_runs_produce_byte_identical_net_forces() {
    let device = init_device().expect("init_device");
    let n: usize = 128;
    let max_neighbors: u32 = 16;
    let total = n * max_neighbors as usize;
    let pair_x: Vec<f32> = (0..total).map(|i| (i as f32) * 0.001 - 0.5).collect();
    let pair_y: Vec<f32> = (0..total).map(|i| (i as f32) * -0.002 + 0.25).collect();
    let pair_z: Vec<f32> = (0..total).map(|i| (i as f32) * 0.0005).collect();
    let counts: Vec<u32> = (0..n).map(|i| (i as u32) % (max_neighbors + 1)).collect();

    fn run(
        device: &Arc<CudaDevice>,
        n: usize,
        max_neighbors: u32,
        pair_x: &[f32],
        pair_y: &[f32],
        pair_z: &[f32],
        counts: &[u32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut pair =
            PairBuffer::new(device.clone(), n, max_neighbors).expect("PairBuffer::new");
        upload_pair_x(&mut pair, pair_x);
        upload_pair_y(&mut pair, pair_y);
        upload_pair_z(&mut pair, pair_z);
        let mut particle_buffers = zero_particle_buffers(device.clone(), n);
        let counts_dev = upload_counts(device, counts);
        reduce_pair_forces_into_buffers(&pair, &counts_dev, &mut particle_buffers).expect("reduce");
        download_forces(&particle_buffers)
    }

    let (ax, ay, az) = run(&device, n, max_neighbors, &pair_x, &pair_y, &pair_z, &counts);
    let (bx, by, bz) = run(&device, n, max_neighbors, &pair_x, &pair_y, &pair_z, &counts);
    assert_eq!(ax, bx);
    assert_eq!(ay, by);
    assert_eq!(az, bz);
}

// --- Numerical edge cases ---

#[test] // rq-5cb58365
fn nan_pair_contribution_propagates_to_nan() {
    let device = init_device().expect("init_device");
    let mut pair = PairBuffer::new(device.clone(), 1, 4).expect("PairBuffer::new");
    upload_pair_x(&mut pair, &[1.0_f32, f32::NAN, 3.0, 0.0]);

    let mut particle_buffers = zero_particle_buffers(device.clone(), 1);
    let counts = upload_counts(&device, &[3u32]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, _, _) = download_forces(&particle_buffers);
    assert!(fx[0].is_nan());
}

#[test] // rq-a1c567b3
fn infinite_pair_contribution_propagates_to_infinity() {
    let device = init_device().expect("init_device");
    let mut pair = PairBuffer::new(device.clone(), 1, 4).expect("PairBuffer::new");
    upload_pair_x(&mut pair, &[1.0_f32, f32::INFINITY, 3.0, 0.0]);

    let mut particle_buffers = zero_particle_buffers(device.clone(), 1);
    let counts = upload_counts(&device, &[3u32]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, _, _) = download_forces(&particle_buffers);
    assert!(fx[0].is_infinite());
    assert!(fx[0] > 0.0);
}
