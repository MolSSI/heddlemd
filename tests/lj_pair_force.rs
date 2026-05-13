mod common;
use common::*;

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use dynamics::gpu::{LennardJonesParameterTable, PairBuffer, ParticleBuffers, init_device};
use dynamics::pbc::SimulationBox;
use dynamics::state::ParticleState;

// --- Helpers ---

#[derive(Debug, Clone, Copy)]
struct LjScalarParams {
    sigma: f32,
    epsilon: f32,
    cutoff: f32,
}

fn default_box() -> SimulationBox {
    SimulationBox::new_orthorhombic(20.0, 20.0, 20.0).expect("default box")
}

fn default_params() -> LjScalarParams {
    LjScalarParams {
        sigma: 1.0,
        epsilon: 1.0,
        cutoff: 5.0,
    }
}

fn table_from_scalar(device: &Arc<CudaDevice>, p: LjScalarParams) -> LennardJonesParameterTable {
    single_type_lj_table(device, p.sigma, p.epsilon, p.cutoff)
}

fn build_state_xyz(positions: &[[f32; 3]]) -> ParticleState {
    let n = positions.len();
    let px: Vec<f32> = positions.iter().map(|p| p[0]).collect();
    let py: Vec<f32> = positions.iter().map(|p| p[1]).collect();
    let pz: Vec<f32> = positions.iter().map(|p| p[2]).collect();
    ParticleState::new(
        px,
        py,
        pz,
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        vec![0u32; n],
        None,
    )
    .expect("build_state_xyz: ParticleState::new")
}

fn min_image_axis(dx: f32, l: f32) -> f32 {
    dx - l * ((dx + l * 0.5) / l).floor()
}

fn lj_force_components(
    pi: [f32; 3],
    pj: [f32; 3],
    lengths: [f32; 3],
    params: LjScalarParams,
) -> [f32; 3] {
    let dx = min_image_axis(pi[0] - pj[0], lengths[0]);
    let dy = min_image_axis(pi[1] - pj[1], lengths[1]);
    let dz = min_image_axis(pi[2] - pj[2], lengths[2]);
    let r2 = dx * dx + dy * dy + dz * dz;
    if r2 > params.cutoff * params.cutoff {
        return [0.0, 0.0, 0.0];
    }
    let inv_r2 = 1.0 / r2;
    let sigma2 = params.sigma * params.sigma;
    let sr2 = sigma2 * inv_r2;
    let sr6 = sr2 * sr2 * sr2;
    let sr12 = sr6 * sr6;
    let factor = 24.0 * params.epsilon * inv_r2 * (2.0 * sr12 - sr6);
    [factor * dx, factor * dy, factor * dz]
}

fn fill_pair_forces_with(pair: &mut PairBuffer, value: f32) {
    let device = pair.device.clone();
    let len = pair.pair_forces_x.len();
    let data = vec![value; len];
    device
        .htod_sync_copy_into(&data, &mut pair.pair_forces_x)
        .unwrap();
    device
        .htod_sync_copy_into(&data, &mut pair.pair_forces_y)
        .unwrap();
    device
        .htod_sync_copy_into(&data, &mut pair.pair_forces_z)
        .unwrap();
}

fn download_pair_forces(pair: &PairBuffer) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let device = pair.device.clone();
    (
        device.dtoh_sync_copy(&pair.pair_forces_x).unwrap(),
        device.dtoh_sync_copy(&pair.pair_forces_y).unwrap(),
        device.dtoh_sync_copy(&pair.pair_forces_z).unwrap(),
    )
}

fn upload_counts(device: &Arc<CudaDevice>, counts: &[u32]) -> CudaSlice<u32> {
    device.htod_sync_copy(counts).unwrap()
}

// --- Module loading ---

#[test] // rq-06058b71
fn init_device_loads_pair_force_module() {
    let device = init_device().expect("init_device");
    assert!(device.has_func("pair_force", "lj_pair_force"));
}

// --- Two-particle correctness ---

#[test] // rq-c538b29d
fn two_particles_at_fixed_separation_produce_closed_form_force() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let params = default_params();
    let table = table_from_scalar(&device, params);
    let positions = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]];
    let state = build_state_xyz(&positions);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 2, 2).unwrap();

    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).expect("lj");
    let (px, py, pz) = download_pair_forces(&pair);

    let expected = lj_force_components(positions[0], positions[1], sim_box.lengths(), params);
    assert_eq!(px[0 * 2 + 1], expected[0]);
    assert_eq!(py[0 * 2 + 1], expected[1]);
    assert_eq!(pz[0 * 2 + 1], expected[2]);
}

#[test] // rq-975b5ae0
fn newtons_third_law_is_bit_exact_for_non_boundary_displacements() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let params = default_params();
    let table = table_from_scalar(&device, params);
    let positions = [[0.0, 0.0, 0.0], [1.3, 0.4, -0.2]];
    let state = build_state_xyz(&positions);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 2, 2).unwrap();

    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).expect("lj");
    let (px, py, pz) = download_pair_forces(&pair);

    assert_eq!(px[0 * 2 + 1], -px[1 * 2 + 0]);
    assert_eq!(py[0 * 2 + 1], -py[1 * 2 + 0]);
    assert_eq!(pz[0 * 2 + 1], -pz[1 * 2 + 0]);
}

// --- Self slot ---

#[test] // rq-cc87744c
fn self_interaction_slots_are_zero() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let params = default_params();
    let table = table_from_scalar(&device, params);
    let positions = [
        [0.0, 0.0, 0.0],
        [1.5, 0.5, -0.3],
        [-2.0, 1.0, 0.7],
        [0.8, -1.5, 2.5],
    ];
    let state = build_state_xyz(&positions);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 4, 4).unwrap();
    fill_pair_forces_with(&mut pair, 999.0);

    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).expect("lj");
    let (px, py, pz) = download_pair_forces(&pair);
    for i in 0..4 {
        let slot = i * 4 + i;
        assert_eq!(px[slot], 0.0_f32, "px self slot for i={i}");
        assert_eq!(py[slot], 0.0_f32, "py self slot for i={i}");
        assert_eq!(pz[slot], 0.0_f32, "pz self slot for i={i}");
    }
}

// --- Cutoff handling ---

#[test] // rq-96fadc6f
fn slot_for_pair_beyond_cutoff_is_zero() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let params = default_params();
    let table = table_from_scalar(&device, params);
    let positions = [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]];
    let state = build_state_xyz(&positions);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 2, 2).unwrap();
    fill_pair_forces_with(&mut pair, 999.0);

    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).expect("lj");
    let (px, py, pz) = download_pair_forces(&pair);
    assert_eq!(px[0 * 2 + 1], 0.0_f32);
    assert_eq!(py[0 * 2 + 1], 0.0_f32);
    assert_eq!(pz[0 * 2 + 1], 0.0_f32);
    assert_eq!(px[1 * 2 + 0], 0.0_f32);
    assert_eq!(py[1 * 2 + 0], 0.0_f32);
    assert_eq!(pz[1 * 2 + 0], 0.0_f32);
}

#[test] // rq-d6bd915a
fn pair_exactly_at_cutoff_is_included() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let params = default_params();
    let table = table_from_scalar(&device, params);
    let positions = [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]];
    let state = build_state_xyz(&positions);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 2, 2).unwrap();

    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).expect("lj");
    let (px, _, _) = download_pair_forces(&pair);

    let expected = lj_force_components(positions[0], positions[1], sim_box.lengths(), params);
    assert_eq!(px[0 * 2 + 1], expected[0]);
    // At r=cutoff the LJ force is non-zero (a small attractive value).
    assert!(expected[0] != 0.0);
}

// --- Force-zero point ---

#[test] // rq-85192a05
fn at_lj_minimum_force_is_near_zero() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let params = default_params();
    let table = table_from_scalar(&device, params);
    let r_min = 2.0_f32.powf(1.0 / 6.0);
    let positions = [[0.0, 0.0, 0.0], [r_min, 0.0, 0.0]];
    let state = build_state_xyz(&positions);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 2, 2).unwrap();

    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).expect("lj");
    let (px, py, pz) = download_pair_forces(&pair);

    let expected = lj_force_components(positions[0], positions[1], sim_box.lengths(), params);
    assert_eq!(px[0 * 2 + 1], expected[0]);
    assert_eq!(py[0 * 2 + 1], expected[1]);
    assert_eq!(pz[0 * 2 + 1], expected[2]);
    // The closed-form force at the LJ minimum is zero up to f32 round-off.
    assert!(px[0 * 2 + 1].abs() < 1e-4);
    assert_eq!(py[0 * 2 + 1], 0.0);
    assert_eq!(pz[0 * 2 + 1], 0.0);
}

// --- Parameter scaling ---

#[test] // rq-26ffa053
fn doubling_epsilon_doubles_force() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let positions = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]];
    let state = build_state_xyz(&positions);

    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 2, 2).unwrap();
    let table1 = single_type_lj_table(&device, 1.0, 1.0, 5.0);
    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table1).expect("lj1");
    let (px1, _, _) = download_pair_forces(&pair);
    let f1 = px1[0 * 2 + 1];

    let mut pair2 = PairBuffer::new(device.clone(), 2, 2).unwrap();
    let table2 = single_type_lj_table(&device, 1.0, 2.0, 5.0);
    lj_pair_force_no_excl(&particle_buffers, &mut pair2, &sim_box, &table2).expect("lj2");
    let (px2, _, _) = download_pair_forces(&pair2);
    let f2 = px2[0 * 2 + 1];

    // Doubling epsilon doubles the LJ factor, which doubles the force component.
    assert!((f2 - 2.0 * f1).abs() <= 1e-5 * f1.abs().max(1.0));
}

// --- PBC minimum-image ---

#[test] // rq-8626ec3c
fn pbc_minimum_image_used_across_box_boundary() {
    let device = init_device().expect("init_device");
    let sim_box = SimulationBox::new_orthorhombic(10.0, 10.0, 10.0).unwrap();
    let params = LjScalarParams {
        sigma: 1.0,
        epsilon: 1.0,
        cutoff: 2.0,
    };
    let table = table_from_scalar(&device, params);
    let positions = [[-4.5, 0.0, 0.0], [4.5, 0.0, 0.0]];
    let state = build_state_xyz(&positions);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 2, 2).unwrap();

    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).expect("lj");
    let (px, _, _) = download_pair_forces(&pair);

    let expected = lj_force_components(positions[0], positions[1], sim_box.lengths(), params);
    assert_eq!(px[0 * 2 + 1], expected[0]);
    // The minimum-image displacement from particle 0 to particle 1 is dx=+1.0, so
    // the repulsive force on particle 0 points in the +x direction.
    assert!(expected[0] > 0.0);
}

// --- N=1 and N=0 ---

#[test] // rq-681afa90
fn single_particle_state_only_self_slot() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let params = default_params();
    let table = table_from_scalar(&device, params);
    let state = build_state_xyz(&[[1.0, 2.0, 3.0]]);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 1, 1).unwrap();
    fill_pair_forces_with(&mut pair, 999.0);

    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).expect("lj");
    let (px, py, pz) = download_pair_forces(&pair);
    assert_eq!(px, vec![0.0_f32]);
    assert_eq!(py, vec![0.0_f32]);
    assert_eq!(pz, vec![0.0_f32]);
}

#[test] // rq-fc220d87
fn empty_state_is_noop() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let params = default_params();
    let table = table_from_scalar(&device, params);
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
    )
    .unwrap();
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 0, 0).unwrap();
    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).expect("lj");
}

// --- Block-non-aligned ---

#[test] // rq-d1e7cb57
fn block_non_aligned_particle_count() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let params = default_params();
    let table = table_from_scalar(&device, params);
    let n = 17;
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let fi = i as f32;
            [fi * 0.5 - 4.0, (fi * 0.5).sin(), (fi * 0.3).cos() * 0.7]
        })
        .collect();
    let state = build_state_xyz(&positions);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), n, n as u32).unwrap();

    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).expect("lj");
    let (px, py, pz) = download_pair_forces(&pair);

    // The kernel uses FMA where nvcc chooses to; the host helper performs
    // separate multiplies and adds. The two agree to within a few ULP per
    // operation, so closed-form comparison uses a small tolerance. Self
    // slots use exact equality because no FMA is involved.
    let close_enough = |a: f32, b: f32| {
        let scale = a.abs().max(b.abs()).max(1e-5);
        (a - b).abs() <= 1e-5 * scale
    };
    for i in 0..n {
        for k in 0..n {
            let slot = i * n + k;
            if i == k {
                assert_eq!(px[slot], 0.0_f32, "px self slot i={i}");
                assert_eq!(py[slot], 0.0_f32, "py self slot i={i}");
                assert_eq!(pz[slot], 0.0_f32, "pz self slot i={i}");
            } else {
                let expected =
                    lj_force_components(positions[i], positions[k], sim_box.lengths(), params);
                assert!(
                    close_enough(px[slot], expected[0]),
                    "px[{i},{k}] kernel={} host={}",
                    px[slot],
                    expected[0]
                );
                assert!(
                    close_enough(py[slot], expected[1]),
                    "py[{i},{k}] kernel={} host={}",
                    py[slot],
                    expected[1]
                );
                assert!(
                    close_enough(pz[slot], expected[2]),
                    "pz[{i},{k}] kernel={} host={}",
                    pz[slot],
                    expected[2]
                );
            }
        }
    }
}

// --- Reproducibility ---

#[test] // rq-dfca62d2
fn two_independent_runs_byte_identical() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let params = default_params();
    let table = table_from_scalar(&device, params);
    let n = 64;
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let fi = i as f32;
            [fi * 0.2 - 6.4, (fi * 0.3).sin() * 2.0, (fi * 0.7).cos() * 1.5]
        })
        .collect();
    let state = build_state_xyz(&positions);

    let particle_buffers_a = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair_a = PairBuffer::new(device.clone(), n, n as u32).unwrap();
    lj_pair_force_no_excl(&particle_buffers_a, &mut pair_a, &sim_box, &table).expect("a");
    let (ax, ay, az) = download_pair_forces(&pair_a);

    let particle_buffers_b = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair_b = PairBuffer::new(device.clone(), n, n as u32).unwrap();
    lj_pair_force_no_excl(&particle_buffers_b, &mut pair_b, &sim_box, &table).expect("b");
    let (bx, by, bz) = download_pair_forces(&pair_b);

    assert_eq!(ax, bx);
    assert_eq!(ay, by);
    assert_eq!(az, bz);
}

// --- Slots beyond N untouched ---

#[test] // rq-e564f8e2
fn slots_beyond_neighbor_counts_are_zeroed() {
    // Under the unified-kernel design, pair_buffer.max_neighbors equals the
    // shared neighbor list's max_neighbors. With a cell-list neighbor list
    // where max_neighbors exceeds the actual neighbor count, the kernel
    // zeros the unused slots so the segmented reduction sees a clean sum.
    let device = init_device().expect("init_device");
    let sim_box = SimulationBox::new_orthorhombic(20.0, 20.0, 20.0).unwrap();
    let params = default_params();
    let table = table_from_scalar(&device, params);
    let n = 4;
    let positions = [
        [0.0, 0.0, 0.0],
        [1.5, 0.5, -0.3],
        [-2.0, 1.0, 0.7],
        [0.8, -1.5, 2.5],
    ];
    let state = build_state_xyz(&positions);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    // Cell-list neighbor list with max_neighbors=8 (exceeding any plausible
    // per-particle neighbor count for N=4).
    let max_neighbors: u32 = 8;
    let mut nl = dynamics::forces::NeighborListState::new_cell_list(
        device.clone(),
        sim_box,
        n,
        params.cutoff,
        max_neighbors,
        0.3,
    )
    .unwrap();
    let mut timings = dynamics::timings::Timings::new(device.clone()).unwrap();
    nl.rebuild(&particle_buffers, &mut timings).unwrap();
    let counts: Vec<u32> = device.dtoh_sync_copy(&nl.neighbor_counts).unwrap();

    let mut pair = PairBuffer::new(device.clone(), n, max_neighbors).unwrap();
    fill_pair_forces_with(&mut pair, 13.5);
    let excl = empty_exclusions(&device, n);
    dynamics::gpu::lj_pair_force(
        &particle_buffers,
        &mut pair,
        &sim_box,
        &table,
        &excl.atom_excl_offsets,
        &excl.atom_excl_partners,
        &excl.atom_excl_scales,
        &nl.neighbor_list,
        &nl.neighbor_counts,
    )
    .unwrap();
    let (px, py, pz) = download_pair_forces(&pair);

    // Slots in [neighbor_counts[i], max_neighbors) are zeroed by the kernel.
    for i in 0..n {
        for k in counts[i] as usize..max_neighbors as usize {
            let slot = i * max_neighbors as usize + k;
            assert_eq!(px[slot], 0.0_f32, "px[{i},{k}] should be 0 (beyond count)");
            assert_eq!(py[slot], 0.0_f32, "py[{i},{k}] should be 0 (beyond count)");
            assert_eq!(pz[slot], 0.0_f32, "pz[{i},{k}] should be 0 (beyond count)");
        }
    }
}

// --- Side effects ---

#[test] // rq-14d7a940
fn does_not_modify_positions_velocities_masses_or_forces() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let params = default_params();
    let table = table_from_scalar(&device, params);
    let mut state = ParticleState::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 0.5, 1.5, 2.5],
        vec![0.1, 0.2, 0.3, 0.4],
        vec![-0.1, -0.2, -0.3, -0.4],
        vec![0.05, 0.1, 0.15, 0.2],
        vec![1.5, 2.5, 3.5, 4.5],
        vec![0u32; 4],
        Some(vec![100, 200, 300, 400]),
    )
    .unwrap();
    state.forces_x = vec![0.7, 0.8, 0.9, 1.0];
    state.forces_y = vec![-0.7, -0.8, -0.9, -1.0];
    state.forces_z = vec![0.5, 0.6, 0.7, 0.8];
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 4, 4).unwrap();

    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).expect("lj");

    let mut downloaded = state.clone();
    downloaded.download_from(&particle_buffers).unwrap();
    assert_eq!(downloaded.positions_x, state.positions_x);
    assert_eq!(downloaded.positions_y, state.positions_y);
    assert_eq!(downloaded.positions_z, state.positions_z);
    assert_eq!(downloaded.velocities_x, state.velocities_x);
    assert_eq!(downloaded.velocities_y, state.velocities_y);
    assert_eq!(downloaded.velocities_z, state.velocities_z);
    assert_eq!(downloaded.forces_x, state.forces_x);
    assert_eq!(downloaded.forces_y, state.forces_y);
    assert_eq!(downloaded.forces_z, state.forces_z);
    assert_eq!(downloaded.masses, state.masses);
    assert_eq!(downloaded.particle_ids, state.particle_ids);
}

// --- End-to-end with reduction ---

#[test] // rq-ec53799e
fn lj_then_reduce_produces_correct_net_forces() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let params = default_params();
    let table = table_from_scalar(&device, params);
    let positions = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]];
    let state = build_state_xyz(&positions);
    let mut particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 2, 2).unwrap();
    let counts = upload_counts(&device, &[2u32, 2]);

    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).expect("lj");
    reduce_pair_forces_into_buffers(&pair, &counts, &mut particle_buffers).expect("reduce");

    let mut downloaded = state.clone();
    downloaded.download_from(&particle_buffers).unwrap();

    let expected_on_0 =
        lj_force_components(positions[0], positions[1], sim_box.lengths(), params);
    assert_eq!(downloaded.forces_x[0], expected_on_0[0]);
    assert_eq!(downloaded.forces_x[1], -downloaded.forces_x[0]);
    assert_eq!(downloaded.forces_y[0], expected_on_0[1]);
    assert_eq!(downloaded.forces_z[0], expected_on_0[2]);
}

// --- NaN propagation ---

#[test] // rq-daf7550b
fn nan_positions_propagate_to_nan_pair_forces() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let params = default_params();
    let table = table_from_scalar(&device, params);
    let state = ParticleState::new(
        vec![f32::NAN, 1.5],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![0u32; 2],
        None,
    )
    .unwrap();
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 2, 2).unwrap();

    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).expect("lj");
    let (px, _, _) = download_pair_forces(&pair);
    assert!(px[0 * 2 + 1].is_nan());
    assert!(px[1 * 2 + 0].is_nan());
}

// --- Multi-type parameter dispatch ---

/// Build a `LennardJonesParameterTable` from explicit n_types and three
/// length-n_types² host arrays.
fn build_table(
    device: &Arc<CudaDevice>,
    n_types: u32,
    sigma: &[f32],
    epsilon: &[f32],
    cutoff: &[f32],
) -> LennardJonesParameterTable {
    let len = (n_types as usize) * (n_types as usize);
    assert_eq!(sigma.len(), len);
    assert_eq!(epsilon.len(), len);
    assert_eq!(cutoff.len(), len);
    LennardJonesParameterTable {
        n_types,
        sigma: device.htod_sync_copy(sigma).unwrap(),
        epsilon: device.htod_sync_copy(epsilon).unwrap(),
        cutoff: device.htod_sync_copy(cutoff).unwrap(),
    }
}

/// State builder with explicit type_indices.
fn build_state_with_types(positions: &[[f32; 3]], type_indices: Vec<u32>) -> ParticleState {
    let n = positions.len();
    assert_eq!(type_indices.len(), n);
    let px: Vec<f32> = positions.iter().map(|p| p[0]).collect();
    let py: Vec<f32> = positions.iter().map(|p| p[1]).collect();
    let pz: Vec<f32> = positions.iter().map(|p| p[2]).collect();
    ParticleState::new(
        px,
        py,
        pz,
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        type_indices,
        None,
    )
    .expect("ParticleState::new")
}

#[test]
fn multi_type_same_type_pair_uses_diagonal_slot() {
    let device = init_device().unwrap();
    let sim_box = default_box();
    // n_types=2: σ_00=1.0, σ_01=σ_10=2.0, σ_11=3.0; ε=1.0 across the
    // diagonal, ε=0.5 off-diagonal; all cutoffs = 5.0.
    let table = build_table(
        &device,
        2,
        &[1.0, 2.0, 2.0, 3.0],
        &[1.0, 0.5, 0.5, 2.0],
        &[5.0, 5.0, 5.0, 5.0],
    );
    let state = build_state_with_types(&[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], vec![0, 0]);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 2, 2).unwrap();
    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).unwrap();
    let (px, _, _) = download_pair_forces(&pair);
    // Both particles are type 0 → diagonal slot σ=1, ε=1.
    let expected = lj_force_components(
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        sim_box.lengths(),
        LjScalarParams { sigma: 1.0, epsilon: 1.0, cutoff: 5.0 },
    );
    assert!((px[0 * 2 + 1] - expected[0]).abs() < 1e-5);
}

#[test]
fn multi_type_mixed_pair_uses_off_diagonal_slot() {
    let device = init_device().unwrap();
    let sim_box = default_box();
    let table = build_table(
        &device,
        2,
        &[1.0, 2.0, 2.0, 3.0],
        &[1.0, 0.5, 0.5, 2.0],
        &[5.0, 5.0, 5.0, 5.0],
    );
    let state = build_state_with_types(&[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], vec![0, 1]);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 2, 2).unwrap();
    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).unwrap();
    let (px, _, _) = download_pair_forces(&pair);
    // Mixed pair (0,1) → slot [0,1] = off-diagonal: σ=2.0, ε=0.5.
    let expected = lj_force_components(
        [0.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        sim_box.lengths(),
        LjScalarParams { sigma: 2.0, epsilon: 0.5, cutoff: 5.0 },
    );
    assert!((px[0 * 2 + 1] - expected[0]).abs() < 1e-5);
}

#[test]
fn multi_type_newtons_third_law_symmetric_table() {
    let device = init_device().unwrap();
    let sim_box = default_box();
    // Off-diagonal entries equal; symmetric by construction.
    let table = build_table(
        &device,
        2,
        &[1.0, 2.0, 2.0, 3.0],
        &[1.0, 0.5, 0.5, 2.0],
        &[5.0, 5.0, 5.0, 5.0],
    );
    let state = build_state_with_types(
        &[[0.0, 0.0, 0.0], [1.3, 0.4, -0.2]],
        vec![0, 1],
    );
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 2, 2).unwrap();
    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).unwrap();
    let (px, py, pz) = download_pair_forces(&pair);
    assert_eq!(px[0 * 2 + 1], -px[1 * 2 + 0]);
    assert_eq!(py[0 * 2 + 1], -py[1 * 2 + 0]);
    assert_eq!(pz[0 * 2 + 1], -pz[1 * 2 + 0]);
}

#[test]
fn multi_type_per_pair_cutoff_zeros_only_the_exceeded_pair() {
    let device = init_device().unwrap();
    let sim_box = default_box();
    // cutoff_00 = 5.0, cutoff_01 = cutoff_10 = 1.0, cutoff_11 = 5.0
    let table = build_table(
        &device,
        2,
        &[1.0, 1.0, 1.0, 1.0],
        &[1.0, 1.0, 1.0, 1.0],
        &[5.0, 1.0, 1.0, 5.0],
    );
    // p0 (type 0), p1 (type 0) at r=1.5  → within 5.0  → non-zero.
    // p0 (type 0), p2 (type 1) at r=2.0  → exceeds 1.0 → zero.
    // p1 (type 0), p2 (type 1) at r=0.5  → within 1.0  → non-zero.
    let state = build_state_with_types(
        &[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
        vec![0, 0, 1],
    );
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 3, 3).unwrap();
    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).unwrap();
    let (px, _, _) = download_pair_forces(&pair);
    assert!(px[0 * 3 + 1] != 0.0, "(0,0)-type pair at r=1.5 should be non-zero");
    assert_eq!(px[0 * 3 + 2], 0.0, "(0,1)-type pair at r=2.0 should be zero");
    assert!(px[1 * 3 + 2] != 0.0, "(0,1)-type pair at r=0.5 should be non-zero");
}

#[test]
fn multi_type_three_type_dispatch() {
    let device = init_device().unwrap();
    let sim_box = default_box();
    // 3x3 table with distinct σ per pair (kept symmetric).
    let sigma = [
        1.0, 1.5, 2.0,
        1.5, 2.0, 2.5,
        2.0, 2.5, 3.0,
    ];
    let epsilon = [1.0; 9];
    let cutoff = [5.0; 9];
    let table = build_table(&device, 3, &sigma, &epsilon, &cutoff);
    // One atom of each type, placed so all pairs are within cutoff.
    let positions = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]];
    let state = build_state_with_types(&positions, vec![0, 1, 2]);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 3, 3).unwrap();
    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).unwrap();
    let (px, _, _) = download_pair_forces(&pair);
    // Verify each (i, k) slot matches its closed-form prediction using the
    // sigma at type_indices[i] * 3 + type_indices[k].
    let lengths = sim_box.lengths();
    for i in 0..3 {
        for k in 0..3 {
            if i == k {
                continue;
            }
            let s = sigma[i * 3 + k];
            let expected = lj_force_components(
                positions[i],
                positions[k],
                lengths,
                LjScalarParams { sigma: s, epsilon: 1.0, cutoff: 5.0 },
            );
            assert!(
                (px[i * 3 + k] - expected[0]).abs() < 1e-4,
                "i={i} k={k}: got {} expected {}",
                px[i * 3 + k],
                expected[0]
            );
        }
    }
}

#[test]
fn lj_param_table_from_config_builds_symmetric_table() {
    use dynamics::io::config::{PairInteractionConfig, ParticleTypeConfig};
    let device = init_device().unwrap();
    let particle_types = vec![
        ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0 },
        ParticleTypeConfig { name: "Kr".to_string(), mass: 2.0 },
    ];
    let pair_interactions = vec![
        PairInteractionConfig {
            between: ("Ar".to_string(), "Ar".to_string()),
            potential: "lennard-jones".to_string(),
            sigma: 1.0,
            epsilon: 1.0,
            cutoff: 5.0,
        },
        PairInteractionConfig {
            between: ("Ar".to_string(), "Kr".to_string()),
            potential: "lennard-jones".to_string(),
            sigma: 2.0,
            epsilon: 0.5,
            cutoff: 5.0,
        },
        PairInteractionConfig {
            between: ("Kr".to_string(), "Kr".to_string()),
            potential: "lennard-jones".to_string(),
            sigma: 3.0,
            epsilon: 2.0,
            cutoff: 5.0,
        },
    ];
    let table =
        LennardJonesParameterTable::from_config(&device, &particle_types, &pair_interactions)
            .expect("from_config");
    assert_eq!(table.n_types, 2);
    let sigma = device.dtoh_sync_copy(&table.sigma).unwrap();
    let epsilon = device.dtoh_sync_copy(&table.epsilon).unwrap();
    let cutoff = device.dtoh_sync_copy(&table.cutoff).unwrap();
    assert_eq!(sigma, vec![1.0_f32, 2.0, 2.0, 3.0]);
    assert_eq!(epsilon, vec![1.0_f32, 0.5, 0.5, 2.0]);
    assert_eq!(cutoff, vec![5.0_f32, 5.0, 5.0, 5.0]);
}

// --- Shared neighbor-list integration ---

#[test] // rq-9004fd7a
fn lennard_jones_state_reports_its_max_cutoff_to_framework() {
    // Build a ForceField with one LJ slot whose largest cutoff is 4.0.
    use dynamics::forces::{
        BondList, ExclusionList, ForceField, Potential,
    };
    use dynamics::io::config::{
        NeighborListConfig, PairInteractionConfig, ParticleTypeConfig,
    };
    let device = init_device().unwrap();
    let particle_types = vec![ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0 }];
    let pair_interactions = vec![PairInteractionConfig {
        between: ("Ar".to_string(), "Ar".to_string()),
        potential: "lennard-jones".to_string(),
        sigma: 1.0,
        epsilon: 1.0,
        cutoff: 4.0,
    }];
    let sim_box = SimulationBox::new_orthorhombic(20.0, 20.0, 20.0).unwrap();
    let ff = ForceField::new(
        device,
        4,
        &sim_box,
        &particle_types,
        &pair_interactions,
        &[],
        &BondList::empty(4),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let lj_slot = ff.slots[0].as_ref();
    assert_eq!(lj_slot.max_cutoff(), Some(4.0_f32));
}

#[test] // rq-e90c6feb
fn trivial_mode_and_cell_list_mode_forces_agree() {
    use dynamics::forces::{
        BondList, ExclusionList, ForceField,
    };
    use dynamics::io::config::{
        NeighborListConfig, PairInteractionConfig, ParticleTypeConfig,
    };
    let device = init_device().unwrap();
    let sim_box = SimulationBox::new_orthorhombic(20.0, 20.0, 20.0).unwrap();
    let particle_types = vec![ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0 }];
    let pair_interactions = vec![PairInteractionConfig {
        between: ("Ar".to_string(), "Ar".to_string()),
        potential: "lennard-jones".to_string(),
        sigma: 1.0,
        epsilon: 1.0,
        cutoff: 3.0,
    }];
    // 4 particles in a small cluster
    let positions = [
        [0.0_f32, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [0.0, 1.3, 0.0],
        [0.4, 0.5, 0.7],
    ];
    let state_a = build_state_xyz(&positions);
    let state_b = state_a.clone();
    let mut buffers_a = ParticleBuffers::new(device.clone(), &state_a).unwrap();
    let mut buffers_b = ParticleBuffers::new(device.clone(), &state_b).unwrap();
    let mut t_a = dynamics::timings::Timings::new(device.clone()).unwrap();
    let mut t_b = dynamics::timings::Timings::new(device.clone()).unwrap();

    let mut ff_trivial = ForceField::new(
        device.clone(),
        4,
        &sim_box,
        &particle_types,
        &pair_interactions,
        &[],
        &BondList::empty(4),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let mut ff_cell = ForceField::new(
        device.clone(),
        4,
        &sim_box,
        &particle_types,
        &pair_interactions,
        &[],
        &BondList::empty(4),
        &ExclusionList::empty(4),
        &NeighborListConfig::CellList { max_neighbors: 32, r_skin: 0.4 },
    )
    .unwrap();
    ff_trivial.step(&mut buffers_a, &sim_box, &mut t_a).unwrap();
    ff_cell.step(&mut buffers_b, &sim_box, &mut t_b).unwrap();

    let fx_a = device.dtoh_sync_copy(&buffers_a.forces_x).unwrap();
    let fx_b = device.dtoh_sync_copy(&buffers_b.forces_x).unwrap();
    for i in 0..4 {
        let denom = fx_a[i].abs().max(1.0);
        assert!(
            (fx_a[i] - fx_b[i]).abs() / denom < 1.0e-4,
            "i={i}: trivial {} vs cell {}",
            fx_a[i],
            fx_b[i]
        );
    }
}
