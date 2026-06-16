mod common;
use common::*;

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use heddle_md::gpu::{GpuContext, PairBuffer, ParticleBuffers, init_device};
use heddle_md::state::ParticleState;
use heddle_md::precision::Real;

fn zero_state(n: usize) -> ParticleState {
    ParticleState::new(
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        vec![0.0; n],
        vec![0u32; n],
        None,
            None,
    )
    .expect("ParticleState::new")
}

fn zero_particle_buffers(gpu: &GpuContext, n: usize) -> ParticleBuffers {
    ParticleBuffers::new(gpu, &zero_state(n)).expect("ParticleBuffers::new")
}

fn upload_counts(device: &Arc<CudaDevice>, counts: &[u32]) -> CudaSlice<u32> {
    device.htod_sync_copy(counts).expect("upload counts")
}

fn upload_pair_x(pair: &mut PairBuffer, data: &[Real]) {
    let device = pair.device.clone();
    device
        .htod_sync_copy_into(data, &mut pair.pair_forces_x)
        .expect("upload pair_forces_x");
}

fn upload_pair_y(pair: &mut PairBuffer, data: &[Real]) {
    let device = pair.device.clone();
    device
        .htod_sync_copy_into(data, &mut pair.pair_forces_y)
        .expect("upload pair_forces_y");
}

fn upload_pair_z(pair: &mut PairBuffer, data: &[Real]) {
    let device = pair.device.clone();
    device
        .htod_sync_copy_into(data, &mut pair.pair_forces_z)
        .expect("upload pair_forces_z");
}

fn download_pair_x(pair: &PairBuffer) -> Vec<Real> {
    pair.device.dtoh_sync_copy(&pair.pair_forces_x).unwrap()
}

fn download_pair_y(pair: &PairBuffer) -> Vec<Real> {
    pair.device.dtoh_sync_copy(&pair.pair_forces_y).unwrap()
}

fn download_pair_z(pair: &PairBuffer) -> Vec<Real> {
    pair.device.dtoh_sync_copy(&pair.pair_forces_z).unwrap()
}

fn download_forces(buffers: &ParticleBuffers) -> (Vec<Real>, Vec<Real>, Vec<Real>) {
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
    let gpu = init_device().expect("init_device");
    let pair = PairBuffer::new(&gpu, 4, 8).expect("PairBuffer::new");
    assert_eq!(pair.particle_count(), 4);
    assert_eq!(pair.max_neighbors(), 8);
    assert_eq!(pair.pair_forces_x.len(), 32);
    assert_eq!(pair.pair_forces_y.len(), 32);
    assert_eq!(pair.pair_forces_z.len(), 32);
    assert_eq!(download_pair_x(&pair), vec![0.0; 32]);
    assert_eq!(download_pair_y(&pair), vec![0.0; 32]);
    assert_eq!(download_pair_z(&pair), vec![0.0; 32]);
}

#[test] // rq-74e4bd02
fn pair_buffer_new_with_zero_particle_count() {
    let gpu = init_device().expect("init_device");
    let pair = PairBuffer::new(&gpu, 0, 8).expect("PairBuffer::new");
    assert_eq!(pair.particle_count(), 0);
    assert_eq!(pair.pair_forces_x.len(), 0);
    assert_eq!(pair.pair_forces_y.len(), 0);
    assert_eq!(pair.pair_forces_z.len(), 0);
}

#[test] // rq-15e1e995
fn pair_buffer_new_with_zero_max_neighbors() {
    let gpu = init_device().expect("init_device");
    let pair = PairBuffer::new(&gpu, 4, 0).expect("PairBuffer::new");
    assert_eq!(pair.max_neighbors(), 0);
    assert_eq!(pair.pair_forces_x.len(), 0);
    assert_eq!(pair.pair_forces_y.len(), 0);
    assert_eq!(pair.pair_forces_z.len(), 0);
}

// --- Module loading ---

#[test] // rq-a43552d5
fn init_device_loads_reduce_module() {
    let gpu = init_device().expect("init_device");
    assert!(gpu.device.has_func("reduce", "reduce_pair_forces"));
    let _ = gpu.kernels.reduce.reduce_pair_forces.clone();
}

// --- Reduction correctness: trivial cases ---

#[test] // rq-2d051b0c
fn reduction_with_all_zero_counts_zeroes_forces() {
    let gpu = init_device().expect("init_device");
    let mut pair = PairBuffer::new(&gpu, 4, 8).expect("PairBuffer::new");
    upload_pair_x(&mut pair, &vec![3.14; 32]);
    upload_pair_y(&mut pair, &vec![2.71; 32]);
    upload_pair_z(&mut pair, &vec![1.41; 32]);

    let mut state = zero_state(4);
    state.forces_x = vec![10.0, 20.0, 30.0, 40.0];
    state.forces_y = vec![-1.0, -2.0, -3.0, -4.0];
    state.forces_z = vec![5.0, 6.0, 7.0, 8.0];
    let mut particle_buffers = ParticleBuffers::new(&gpu, &state).unwrap();

    let counts = upload_counts(&gpu.device, &[0u32, 0, 0, 0]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, fy, fz) = download_forces(&particle_buffers);
    assert_eq!(fx, vec![0.0; 4]);
    assert_eq!(fy, vec![0.0; 4]);
    assert_eq!(fz, vec![0.0; 4]);
}

#[test] // rq-8ee33aa0
fn reduction_with_single_particle_single_neighbor() {
    let gpu = init_device().expect("init_device");
    let mut pair = PairBuffer::new(&gpu, 1, 4).expect("PairBuffer::new");
    upload_pair_x(&mut pair, &[1.5, 0.0, 0.0, 0.0]);
    upload_pair_y(&mut pair, &[-2.5, 0.0, 0.0, 0.0]);
    upload_pair_z(&mut pair, &[0.75, 0.0, 0.0, 0.0]);

    let mut particle_buffers = zero_particle_buffers(&gpu, 1);
    let counts = upload_counts(&gpu.device, &[1u32]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, fy, fz) = download_forces(&particle_buffers);
    assert_eq!(fx, vec![1.5]);
    assert_eq!(fy, vec![-2.5]);
    assert_eq!(fz, vec![0.75]);
}

// --- Reduction correctness: order and bounds ---

#[test] // rq-e950f4e6
fn reduction_sums_entries_left_to_right() {
    let gpu = init_device().expect("init_device");
    let mut pair = PairBuffer::new(&gpu, 1, 4).expect("PairBuffer::new");
    upload_pair_x(&mut pair, &[1.0, 2.0, 4.0, 999.0]);

    let mut particle_buffers = zero_particle_buffers(&gpu, 1);
    let counts = upload_counts(&gpu.device, &[3u32]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, _, _) = download_forces(&particle_buffers);
    let expected = (1.0 + 2.0) + 4.0;
    assert_eq!(fx[0], expected);
    // The slot at index 3 (value 999.0) must not be included.
    assert!(fx[0] != expected + 999.0);
}

#[test] // rq-78fc2fbb
fn reduction_ignores_slots_beyond_count() {
    let gpu = init_device().expect("init_device");
    let mut pair = PairBuffer::new(&gpu, 1, 8).expect("PairBuffer::new");
    let mut data = vec![Real::INFINITY; 8];
    data[0] = 10.0;
    data[1] = 20.0;
    upload_pair_x(&mut pair, &data);

    let mut particle_buffers = zero_particle_buffers(&gpu, 1);
    let counts = upload_counts(&gpu.device, &[2u32]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, _, _) = download_forces(&particle_buffers);
    assert_eq!(fx[0], 30.0);
    assert!(fx[0].is_finite());
}

#[test] // rq-590dcd7e
fn reduction_at_full_max_neighbors_capacity() {
    let gpu = init_device().expect("init_device");
    let mut pair = PairBuffer::new(&gpu, 1, 4).expect("PairBuffer::new");
    upload_pair_x(&mut pair, &[1.0, 2.0, 3.0, 4.0]);

    let mut particle_buffers = zero_particle_buffers(&gpu, 1);
    let counts = upload_counts(&gpu.device, &[4u32]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, _, _) = download_forces(&particle_buffers);
    let expected = ((1.0 + 2.0) + 3.0) + 4.0;
    assert_eq!(fx[0], expected);
}

// --- Reduction correctness: multiple particles ---

#[test] // rq-6808532e
fn per_particle_reduction_with_varying_counts() {
    let gpu = init_device().expect("init_device");
    let mut pair = PairBuffer::new(&gpu, 3, 4).expect("PairBuffer::new");
    let pair_x: Vec<Real> = vec![
        // particle 0: count = 2
        1.0, 2.0, 100.0, 100.0,
        // particle 1: count = 1
        10.0, 100.0, 100.0, 100.0,
        // particle 2: count = 4
        0.5, 0.5, 0.5, 0.5,
    ];
    upload_pair_x(&mut pair, &pair_x);

    let mut particle_buffers = zero_particle_buffers(&gpu, 3);
    let counts = upload_counts(&gpu.device, &[2u32, 1, 4]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, _, _) = download_forces(&particle_buffers);
    assert_eq!(fx[0], 3.0);
    assert_eq!(fx[1], 10.0);
    assert_eq!(fx[2], 2.0);
}

// --- Empty state ---

#[test] // rq-493caf32
fn reduce_pair_forces_on_empty_state_is_noop() {
    let gpu = init_device().expect("init_device");
    let pair = PairBuffer::new(&gpu, 0, 8).expect("PairBuffer::new");
    let mut particle_buffers = zero_particle_buffers(&gpu, 0);
    let counts = upload_counts(&gpu.device, &[]);
    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
}

// --- Bounds handling ---

#[test] // rq-77e88745
fn block_non_aligned_particle_count_is_handled() {
    let gpu = init_device().expect("init_device");
    let n: usize = 1000;
    let max_neighbors: u32 = 2;
    let mut pair = PairBuffer::new(&gpu, n, max_neighbors).expect("PairBuffer::new");
    let mut x_data = vec![0.0; n * 2];
    for i in 0..n {
        x_data[i * 2] = i as Real;
        x_data[i * 2 + 1] = -(i as Real);
    }
    upload_pair_x(&mut pair, &x_data);

    let mut particle_buffers = zero_particle_buffers(&gpu, n);
    let counts = upload_counts(&gpu.device, &vec![2u32; n]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, fy, fz) = download_forces(&particle_buffers);
    for i in 0..n {
        assert_eq!(fx[i], 0.0, "fx[{i}]");
        assert_eq!(fy[i], 0.0, "fy[{i}]");
        assert_eq!(fz[i], 0.0, "fz[{i}]");
    }
}

// --- Side effects ---

#[test] // rq-f5299d6e
fn reduction_overwrites_prior_force_values() {
    let gpu = init_device().expect("init_device");
    let pair = PairBuffer::new(&gpu, 4, 2).expect("PairBuffer::new");

    let mut state = zero_state(4);
    state.forces_x = vec![99.0, 88.0, 77.0, 66.0];
    let mut particle_buffers = ParticleBuffers::new(&gpu, &state).unwrap();

    let counts = upload_counts(&gpu.device, &[0u32, 0, 0, 0]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, _, _) = download_forces(&particle_buffers);
    assert_eq!(fx, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test] // rq-9b794cff
fn reduction_does_not_modify_pair_buffer() {
    let gpu = init_device().expect("init_device");
    let mut pair = PairBuffer::new(&gpu, 4, 4).expect("PairBuffer::new");
    let pair_x: Vec<Real> = (0..16).map(|i| 0.1 + i as Real * 0.5).collect();
    let pair_y: Vec<Real> = (0..16).map(|i| -0.2 + i as Real * 0.25).collect();
    let pair_z: Vec<Real> = (0..16).map(|i| 0.3 + i as Real * 0.125).collect();
    upload_pair_x(&mut pair, &pair_x);
    upload_pair_y(&mut pair, &pair_y);
    upload_pair_z(&mut pair, &pair_z);

    let snapshot_x = download_pair_x(&pair);
    let snapshot_y = download_pair_y(&pair);
    let snapshot_z = download_pair_z(&pair);

    let mut particle_buffers = zero_particle_buffers(&gpu, 4);
    let counts = upload_counts(&gpu.device, &[3u32, 3, 3, 3]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");

    assert_eq!(download_pair_x(&pair), snapshot_x);
    assert_eq!(download_pair_y(&pair), snapshot_y);
    assert_eq!(download_pair_z(&pair), snapshot_z);
}

#[test] // rq-c9da25a7
fn reduction_does_not_modify_positions_velocities_masses() {
    let gpu = init_device().expect("init_device");
    let pair = PairBuffer::new(&gpu, 4, 2).expect("PairBuffer::new");

    let state = ParticleState::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
        vec![0.1, 0.2, 0.3, 0.4],
        vec![-0.1, -0.2, -0.3, -0.4],
        vec![0.05, 0.1, 0.15, 0.2],
        vec![1.5, 2.5, 3.5, 4.5],
        vec![0.0; vec![1.5, 2.5, 3.5, 4.5].len()],
        vec![0u32; 4],
        Some(vec![100, 200, 300, 400]),
            None,
    )
    .unwrap();
    let mut particle_buffers = ParticleBuffers::new(&gpu, &state).unwrap();

    let counts = upload_counts(&gpu.device, &[0u32, 0, 0, 0]);
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
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let pair = PairBuffer::new(&gpu, 4, 2).expect("PairBuffer::new");
    let mut particle_buffers = zero_particle_buffers(&gpu, 4);
    let counts = upload_counts(&device, &[0u32, 1, 2, 0]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");

    let downloaded: Vec<u32> = device.dtoh_sync_copy(&counts).unwrap();
    assert_eq!(downloaded, vec![0u32, 1, 2, 0]);
}

// --- Reproducibility ---

#[test] // rq-b4f18ea1
fn two_independent_runs_produce_byte_identical_net_forces() {
    let gpu = init_device().expect("init_device");
    let n: usize = 128;
    let max_neighbors: u32 = 16;
    let total = n * max_neighbors as usize;
    let pair_x: Vec<Real> = (0..total).map(|i| (i as Real) * 0.001 - 0.5).collect();
    let pair_y: Vec<Real> = (0..total).map(|i| (i as Real) * -0.002 + 0.25).collect();
    let pair_z: Vec<Real> = (0..total).map(|i| (i as Real) * 0.0005).collect();
    let counts: Vec<u32> = (0..n).map(|i| (i as u32) % (max_neighbors + 1)).collect();

    fn run(
        gpu: &GpuContext,
        n: usize,
        max_neighbors: u32,
        pair_x: &[Real],
        pair_y: &[Real],
        pair_z: &[Real],
        counts: &[u32],
    ) -> (Vec<Real>, Vec<Real>, Vec<Real>) {
        let mut pair =
            PairBuffer::new(gpu, n, max_neighbors).expect("PairBuffer::new");
        upload_pair_x(&mut pair, pair_x);
        upload_pair_y(&mut pair, pair_y);
        upload_pair_z(&mut pair, pair_z);
        let mut particle_buffers = zero_particle_buffers(gpu, n);
        let counts_dev = upload_counts(&gpu.device, counts);
        reduce_pair_forces_into_buffers(&pair, &counts_dev, &mut particle_buffers).expect("reduce");
        download_forces(&particle_buffers)
    }

    let (ax, ay, az) = run(&gpu, n, max_neighbors, &pair_x, &pair_y, &pair_z, &counts);
    let (bx, by, bz) = run(&gpu, n, max_neighbors, &pair_x, &pair_y, &pair_z, &counts);
    assert_eq!(ax, bx);
    assert_eq!(ay, by);
    assert_eq!(az, bz);
}

// --- Numerical edge cases ---

#[test] // rq-5cb58365
fn nan_pair_contribution_propagates_to_nan() {
    let gpu = init_device().expect("init_device");
    let mut pair = PairBuffer::new(&gpu, 1, 4).expect("PairBuffer::new");
    upload_pair_x(&mut pair, &[1.0, Real::NAN, 3.0, 0.0]);

    let mut particle_buffers = zero_particle_buffers(&gpu, 1);
    let counts = upload_counts(&gpu.device, &[3u32]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, _, _) = download_forces(&particle_buffers);
    assert!(fx[0].is_nan());
}

#[test] // rq-a1c567b3
fn infinite_pair_contribution_propagates_to_infinity() {
    let gpu = init_device().expect("init_device");
    let mut pair = PairBuffer::new(&gpu, 1, 4).expect("PairBuffer::new");
    upload_pair_x(&mut pair, &[1.0, Real::INFINITY, 3.0, 0.0]);

    let mut particle_buffers = zero_particle_buffers(&gpu, 1);
    let counts = upload_counts(&gpu.device, &[3u32]);

    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");
    let (fx, _, _) = download_forces(&particle_buffers);
    assert!(fx[0].is_infinite());
    assert!(fx[0] > 0.0);
}

// --- Energy and virial reduction ---

#[test] // rq-9e487c80
fn reduction_sums_pair_energies_left_to_right() {
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let mut pair = PairBuffer::new(&gpu, 1, 4).unwrap();
    let mut state = ParticleState::new(
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![1.0],
        vec![0.0; vec![1.0].len()],
        vec![0u32],
        None,
            None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    device
        .htod_sync_copy_into(&vec![0.5, 1.5, 2.0, 999.0], &mut pair.pair_energies)
        .unwrap();
    device
        .htod_sync_copy_into(&vec![-1.0, 2.0, 3.0, 0.0], &mut pair.pair_virials)
        .unwrap();
    let counts = device.htod_sync_copy(&[3u32]).unwrap();
    reduce_pair_forces_into_buffers(&pair, &counts, &mut buffers).unwrap();
    state.download_from(&buffers).unwrap();
    assert_eq!(state.potential_energies[0], (0.5 + 1.5) + 2.0);
    assert_eq!(state.virials[0], (-1.0 + 2.0) + 3.0);
}

#[test] // rq-961c2ee6
fn reduction_zero_count_writes_zero_to_energy_and_virial() {
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let mut pair = PairBuffer::new(&gpu, 2, 4).unwrap();
    let mut state = ParticleState::new(
        vec![0.0, 1.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0; vec![1.0, 1.0].len()],
        vec![0u32, 0],
        None,
            None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    // Pre-fill energies and virials with non-zero junk to prove they get overwritten.
    device
        .htod_sync_copy_into(&vec![7.0; 8], &mut pair.pair_energies)
        .unwrap();
    device
        .htod_sync_copy_into(&vec![-3.0; 8], &mut pair.pair_virials)
        .unwrap();
    let counts = device.htod_sync_copy(&[0u32, 0]).unwrap();
    reduce_pair_forces_into_buffers(&pair, &counts, &mut buffers).unwrap();
    state.download_from(&buffers).unwrap();
    assert_eq!(state.potential_energies, vec![0.0, 0.0]);
    assert_eq!(state.virials, vec![0.0, 0.0]);
}

#[test] // rq-41d9e514
fn energy_and_virial_share_force_indexing() {
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let mut pair = PairBuffer::new(&gpu, 2, 2).unwrap();
    let mut state = ParticleState::new(
        vec![0.0, 1.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0; vec![1.0, 1.0].len()],
        vec![0u32, 0],
        None,
            None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    device
        .htod_sync_copy_into(&vec![1.0, 2.0, 3.0, 4.0], &mut pair.pair_forces_x)
        .unwrap();
    device
        .htod_sync_copy_into(&vec![10.0, 20.0, 30.0, 40.0], &mut pair.pair_energies)
        .unwrap();
    device
        .htod_sync_copy_into(
            &vec![100.0, 200.0, 300.0, 400.0],
            &mut pair.pair_virials,
        )
        .unwrap();
    let counts = device.htod_sync_copy(&[2u32, 2]).unwrap();
    reduce_pair_forces_into_buffers(&pair, &counts, &mut buffers).unwrap();
    state.download_from(&buffers).unwrap();
    assert_eq!(state.forces_x, vec![3.0, 7.0]);
    assert_eq!(state.potential_energies, vec![30.0, 70.0]);
    assert_eq!(state.virials, vec![300.0, 700.0]);
}

// --- Warp-tree reduction shape ---

#[test] // rq-b2221a99
fn reduction_with_count_larger_than_warp_size_sweeps_multiple_iterations() {
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let mut pair = PairBuffer::new(&gpu, 1, 128).unwrap();
    let mut data = vec![0.0; 128];
    for slot in data.iter_mut().take(96) {
        *slot = 1.0;
    }
    upload_pair_x(&mut pair, &data);
    let mut buffers = zero_particle_buffers(&gpu, 1);
    let counts = device.htod_sync_copy(&[96u32]).unwrap();
    reduce_pair_forces_into_buffers(&pair, &counts, &mut buffers).unwrap();
    let (fx, _, _) = download_forces(&buffers);
    assert_eq!(fx[0], 96.0);
}

#[test] // rq-c009903e
fn reduction_with_count_larger_than_one_warp_sweep_accumulates_across_sweeps() {
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let mut pair = PairBuffer::new(&gpu, 1, 1024).unwrap();
    let mut data = vec![0.0; 1024];
    for slot in data.iter_mut().take(600) {
        *slot = 1.0;
    }
    for slot in data.iter_mut().skip(600) {
        *slot = 99.0;
    }
    upload_pair_x(&mut pair, &data);
    let mut buffers = zero_particle_buffers(&gpu, 1);
    let counts = device.htod_sync_copy(&[600u32]).unwrap();
    reduce_pair_forces_into_buffers(&pair, &counts, &mut buffers).unwrap();
    let (fx, _, _) = download_forces(&buffers);
    assert_eq!(fx[0], 600.0);
    assert!(fx[0].is_finite());
}

/// Replicates the device-side warp tree reduction shape so a CPU
/// reference can be compared bit-for-bit with the GPU output. Matches
/// the per-warp algorithm documented in `rqm/pair-reduction.md`:
/// 32 lanes accumulate strided partial sums, then combine via a
/// 5-step pairwise XOR butterfly. Lane 0's value is the result.
fn cpu_warp_tree_sum(slots: &[Real], count: usize, max_neighbors: usize) -> Real {
    const WARP_SIZE: usize = 32;
    let mut lanes = [0.0 as Real; WARP_SIZE];
    let count = count.min(max_neighbors);
    let sweep_end = count.div_ceil(WARP_SIZE) * WARP_SIZE;
    let mut s = 0;
    while s < sweep_end {
        for lane in 0..WARP_SIZE {
            let k = s + lane;
            if k < count {
                lanes[lane] += slots[k];
            }
        }
        s += WARP_SIZE;
    }
    for &stride in &[16usize, 8, 4, 2, 1] {
        let mut next = lanes;
        for lane in 0..WARP_SIZE {
            next[lane] = lanes[lane] + lanes[lane ^ stride];
        }
        lanes = next;
    }
    lanes[0]
}

#[test] // rq-aee2bfb2
fn reduction_tree_result_agrees_with_cpu_warp_tree_reference() {
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let max_neighbors: u32 = 1024;
    let count: usize = 800;
    let mut data = vec![0.0; max_neighbors as usize];
    for k in 0..count {
        data[k] = (k as Real * 0.1).sin();
    }
    let mut pair = PairBuffer::new(&gpu, 1, max_neighbors).unwrap();
    upload_pair_x(&mut pair, &data);
    let mut buffers = zero_particle_buffers(&gpu, 1);
    let counts = device.htod_sync_copy(&[count as u32]).unwrap();
    reduce_pair_forces_into_buffers(&pair, &counts, &mut buffers).unwrap();
    let (fx, _, _) = download_forces(&buffers);

    let cpu_reference = cpu_warp_tree_sum(&data, count, max_neighbors as usize);
    assert_eq!(fx[0].to_bits(), cpu_reference.to_bits());

    // Sequential left-to-right f32 sum is close but not bit-exact.
    let seq: Real = data.iter().take(count).copied().sum();
    let rel = 1e-5;
    assert!(
        (fx[0] - seq).abs() <= rel * seq.abs().max(Real::MIN_POSITIVE),
        "GPU tree-sum {} vs sequential left-to-right sum {} differ outside 1e-5",
        fx[0],
        seq
    );
}

#[test] // rq-46d24bfb
fn one_warp_per_particle_warps_are_independent() {
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let n: usize = 512;
    let max_neighbors: u32 = 64;
    let total = n * max_neighbors as usize;
    // Deterministic pseudo-random: linear congruential.
    let mut state: u32 = 0xDEAD_BEEF;
    let mut next = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        // Map to [-1.0, 1.0].
        ((state >> 8) as Real) / ((1u32 << 23) as Real) - 1.0
    };
    let data: Vec<Real> = (0..total).map(|_| next()).collect();
    let counts: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2654435761).wrapping_rem(max_neighbors + 1))
        .collect();
    let mut pair = PairBuffer::new(&gpu, n, max_neighbors).unwrap();
    upload_pair_x(&mut pair, &data);
    let mut buffers = zero_particle_buffers(&gpu, n);
    let counts_dev = device.htod_sync_copy(&counts).unwrap();
    reduce_pair_forces_into_buffers(&pair, &counts_dev, &mut buffers).unwrap();
    let (fx, _, _) = download_forces(&buffers);

    for i in 0..n {
        let slot_base = i * max_neighbors as usize;
        let slots = &data[slot_base..slot_base + max_neighbors as usize];
        let expected = cpu_warp_tree_sum(slots, counts[i] as usize, max_neighbors as usize);
        assert_eq!(
            fx[i].to_bits(),
            expected.to_bits(),
            "particle {i}: GPU {} (bits {:x}) != CPU warp-tree {} (bits {:x})",
            fx[i],
            fx[i].to_bits(),
            expected,
            expected.to_bits()
        );
    }
}

// --- Warp-per-particle grid layout ---

#[test]
fn particle_count_not_multiple_of_warps_per_block_uses_ragged_final_block() {
    // 10 particles, WARPS_PER_BLOCK = 8 → 2 blocks. Block 1's warps
    // 2..8 carry no particle and must return without writing.
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let n: usize = 10;
    let max_neighbors: u32 = 4;
    let mut pair = PairBuffer::new(&gpu, n, max_neighbors).unwrap();
    let mut data = vec![0.0 as Real; n * max_neighbors as usize];
    for i in 0..n {
        data[i * max_neighbors as usize] = 1.0;
    }
    upload_pair_x(&mut pair, &data);
    let counts = device.htod_sync_copy(&vec![1u32; n]).unwrap();
    let mut buffers = zero_particle_buffers(&gpu, n);
    reduce_pair_forces_into_buffers(&pair, &counts, &mut buffers).unwrap();
    let (fx, _, _) = download_forces(&buffers);
    for i in 0..n {
        assert_eq!(fx[i], 1.0, "particle {i}");
    }
}

#[test]
fn particle_count_below_warps_per_block_uses_a_single_under_full_block() {
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let n: usize = 3;
    let max_neighbors: u32 = 4;
    let mut pair = PairBuffer::new(&gpu, n, max_neighbors).unwrap();
    let mut data = vec![0.0 as Real; n * max_neighbors as usize];
    for i in 0..n {
        data[i * max_neighbors as usize] = 2.0;
    }
    upload_pair_x(&mut pair, &data);
    let counts = device.htod_sync_copy(&vec![1u32; n]).unwrap();
    let mut buffers = zero_particle_buffers(&gpu, n);
    reduce_pair_forces_into_buffers(&pair, &counts, &mut buffers).unwrap();
    let (fx, _, _) = download_forces(&buffers);
    for i in 0..n {
        assert_eq!(fx[i], 2.0, "particle {i}");
    }
}

#[test]
fn reduce_kernels_declare_no_shared_memory() {
    // The compiled reduce.cu PTX should contain no .shared directive.
    // The warp-per-particle topology uses only register accumulators
    // and __shfl_xor_sync within each warp.
    let ptx = heddle_md::kernels::REDUCE;
    assert!(
        !ptx.contains(".shared"),
        "reduce.cu PTX contains .shared directive (shared memory declared)"
    );
}

// --- Split independence (force vs. energy-virial launcher) ---

#[test] // rq-9f3a36aa
fn reduce_pair_forces_does_not_touch_energy_or_virial_targets() {
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let n: usize = 4;
    let max_neighbors: u32 = 2;
    let mut pair = PairBuffer::new(&gpu, n, max_neighbors).unwrap();
    upload_pair_x(&mut pair, &[1.0; 8]);
    device
        .htod_sync_copy_into(&[99.0; 8], &mut pair.pair_energies)
        .unwrap();
    device
        .htod_sync_copy_into(&[123.0; 8], &mut pair.pair_virials)
        .unwrap();

    let mut buffers = zero_particle_buffers(&gpu, n);
    // Seed the energy/virial targets with known nonzero patterns.
    let pattern_e: Vec<Real> = vec![11.0, 22.0, 33.0, 44.0];
    let pattern_v: Vec<Real> = vec![-1.0, -2.0, -3.0, -4.0];
    device
        .htod_sync_copy_into(&pattern_e, &mut buffers.potential_energies)
        .unwrap();
    device
        .htod_sync_copy_into(&pattern_v, &mut buffers.virials)
        .unwrap();

    let counts = device.htod_sync_copy(&[2u32; 4]).unwrap();
    let mut vx = buffers.forces_x.slice_mut(..);
    let mut vy = buffers.forces_y.slice_mut(..);
    let mut vz = buffers.forces_z.slice_mut(..);
    heddle_md::gpu::reduce_pair_forces(&pair, &counts, &mut vx, &mut vy, &mut vz, n).unwrap();

    let read_e = device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
    let read_v = device.dtoh_sync_copy(&buffers.virials).unwrap();
    assert_eq!(read_e, pattern_e);
    assert_eq!(read_v, pattern_v);
}

#[test] // rq-75ee70dd
fn reduce_pair_energy_virial_does_not_touch_force_targets() {
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let n: usize = 4;
    let max_neighbors: u32 = 2;
    let mut pair = PairBuffer::new(&gpu, n, max_neighbors).unwrap();
    device
        .htod_sync_copy_into(&[1.0; 8], &mut pair.pair_energies)
        .unwrap();
    device
        .htod_sync_copy_into(&[1.0; 8], &mut pair.pair_virials)
        .unwrap();

    let mut buffers = zero_particle_buffers(&gpu, n);
    let pattern_fx: Vec<Real> = vec![10.0, 20.0, 30.0, 40.0];
    let pattern_fy: Vec<Real> = vec![-10.0, -20.0, -30.0, -40.0];
    let pattern_fz: Vec<Real> = vec![1.5, 2.5, 3.5, 4.5];
    device
        .htod_sync_copy_into(&pattern_fx, &mut buffers.forces_x)
        .unwrap();
    device
        .htod_sync_copy_into(&pattern_fy, &mut buffers.forces_y)
        .unwrap();
    device
        .htod_sync_copy_into(&pattern_fz, &mut buffers.forces_z)
        .unwrap();

    let counts = device.htod_sync_copy(&[2u32; 4]).unwrap();
    let mut ve = buffers.potential_energies.slice_mut(..);
    let mut vw = buffers.virials.slice_mut(..);
    heddle_md::gpu::reduce_pair_energy_virial(&pair, &counts, &mut ve, &mut vw, n).unwrap();

    let read_fx = device.dtoh_sync_copy(&buffers.forces_x).unwrap();
    let read_fy = device.dtoh_sync_copy(&buffers.forces_y).unwrap();
    let read_fz = device.dtoh_sync_copy(&buffers.forces_z).unwrap();
    assert_eq!(read_fx, pattern_fx);
    assert_eq!(read_fy, pattern_fy);
    assert_eq!(read_fz, pattern_fz);
}

#[test] // rq-803ee7e5
fn skipping_reduce_pair_energy_virial_leaves_stale_energy_targets() {
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let n: usize = 2;
    let max_neighbors: u32 = 2;
    let mut pair = PairBuffer::new(&gpu, n, max_neighbors).unwrap();
    device
        .htod_sync_copy_into(&[10.0, 20.0, 30.0, 40.0], &mut pair.pair_energies)
        .unwrap();
    device
        .htod_sync_copy_into(&[1.0, 1.0, 1.0, 1.0], &mut pair.pair_virials)
        .unwrap();

    let mut buffers = zero_particle_buffers(&gpu, n);
    let counts = device.htod_sync_copy(&[2u32, 2]).unwrap();

    // Run the full reduction once: energy targets land at [30.0, 70.0].
    {
        let mut ve = buffers.potential_energies.slice_mut(..);
        let mut vw = buffers.virials.slice_mut(..);
        heddle_md::gpu::reduce_pair_energy_virial(&pair, &counts, &mut ve, &mut vw, n).unwrap();
    }
    let pre = device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
    assert_eq!(pre, vec![30.0, 70.0]);

    // Overwrite pair_energies; the new sums would be [2.0, 2.0]. Then
    // run only reduce_pair_forces. The energy targets must stay stale.
    device
        .htod_sync_copy_into(&[1.0, 1.0, 1.0, 1.0], &mut pair.pair_energies)
        .unwrap();
    let mut vx = buffers.forces_x.slice_mut(..);
    let mut vy = buffers.forces_y.slice_mut(..);
    let mut vz = buffers.forces_z.slice_mut(..);
    heddle_md::gpu::reduce_pair_forces(&pair, &counts, &mut vx, &mut vy, &mut vz, n).unwrap();
    let post = device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
    assert_eq!(post, vec![30.0, 70.0]);
}

#[test] // rq-b4c772c9
fn two_independent_runs_of_reduce_pair_energy_virial_produce_byte_identical_scalars() {
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let n: usize = 128;
    let max_neighbors: u32 = 16;
    let total = n * max_neighbors as usize;
    let pair_e: Vec<Real> = (0..total).map(|i| (i as Real) * 0.001 - 0.5).collect();
    let pair_v: Vec<Real> = (0..total).map(|i| (i as Real) * -0.002 + 0.25).collect();
    let counts: Vec<u32> = (0..n).map(|i| (i as u32) % (max_neighbors + 1)).collect();

    let run = || -> (Vec<Real>, Vec<Real>) {
        let mut pair = PairBuffer::new(&gpu, n, max_neighbors).unwrap();
        device.htod_sync_copy_into(&pair_e, &mut pair.pair_energies).unwrap();
        device.htod_sync_copy_into(&pair_v, &mut pair.pair_virials).unwrap();
        let mut buffers = zero_particle_buffers(&gpu, n);
        let counts_dev = device.htod_sync_copy(&counts).unwrap();
        let mut ve = buffers.potential_energies.slice_mut(..);
        let mut vw = buffers.virials.slice_mut(..);
        heddle_md::gpu::reduce_pair_energy_virial(&pair, &counts_dev, &mut ve, &mut vw, n).unwrap();
        drop(ve);
        drop(vw);
        (
            device.dtoh_sync_copy(&buffers.potential_energies).unwrap(),
            device.dtoh_sync_copy(&buffers.virials).unwrap(),
        )
    };
    let (a_e, a_v) = run();
    let (b_e, b_v) = run();
    assert_eq!(a_e, b_e);
    assert_eq!(a_v, b_v);
}

#[test] // rq-41453204
fn reduce_pair_energy_virial_on_empty_state_is_noop() {
    let gpu = init_device().expect("init_device");
    let device = gpu.device.clone();
    let pair = PairBuffer::new(&gpu, 0, 8).unwrap();
    let counts: CudaSlice<u32> = device.htod_sync_copy(&Vec::<u32>::new()).unwrap();
    let mut buffers = zero_particle_buffers(&gpu, 0);
    let mut ve = buffers.potential_energies.slice_mut(..);
    let mut vw = buffers.virials.slice_mut(..);
    // particle_count=0 must early-return Ok without launching a kernel.
    heddle_md::gpu::reduce_pair_energy_virial(&pair, &counts, &mut ve, &mut vw, 0).unwrap();
}
