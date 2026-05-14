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
        &sim_box,
        n,
        params.cutoff,
        max_neighbors,
        0.3,
    )
    .unwrap();
    let mut timings = dynamics::timings::Timings::new(device.clone()).unwrap();
    nl.rebuild(&sim_box, &particle_buffers, &mut timings).unwrap();
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
            None,
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
/// length-n_types² host arrays. `r_switch` is set to `cutoff` on every
/// slot, selecting the hard-cutoff degenerate case in the kernel so
/// multi-type tests written against the unmodified Lennard-Jones
/// expression are unaffected.
fn build_table(
    device: &Arc<CudaDevice>,
    n_types: u32,
    sigma: &[f32],
    epsilon: &[f32],
    cutoff: &[f32],
) -> LennardJonesParameterTable {
    let switch: Vec<f32> = cutoff.to_vec();
    build_table_with_switch(device, n_types, sigma, epsilon, cutoff, &switch)
}

/// Build a `LennardJonesParameterTable` from explicit n_types and four
/// length-n_types² host arrays. Tests that exercise the switching
/// function with multi-type parameters use this directly.
fn build_table_with_switch(
    device: &Arc<CudaDevice>,
    n_types: u32,
    sigma: &[f32],
    epsilon: &[f32],
    cutoff: &[f32],
    switch: &[f32],
) -> LennardJonesParameterTable {
    let len = (n_types as usize) * (n_types as usize);
    assert_eq!(sigma.len(), len);
    assert_eq!(epsilon.len(), len);
    assert_eq!(cutoff.len(), len);
    assert_eq!(switch.len(), len);
    LennardJonesParameterTable {
        n_types,
        sigma: device.htod_sync_copy(sigma).unwrap(),
        epsilon: device.htod_sync_copy(epsilon).unwrap(),
        cutoff: device.htod_sync_copy(cutoff).unwrap(),
        switch: device.htod_sync_copy(switch).unwrap(),
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
    use dynamics::io::config::{PairInteractionConfig, PairPotentialParams, ParticleTypeConfig};
    let device = init_device().unwrap();
    let particle_types = vec![
        ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0 },
        ParticleTypeConfig { name: "Kr".to_string(), mass: 2.0 },
    ];
    let pair_interactions = vec![
        PairInteractionConfig {
            between: ("Ar".to_string(), "Ar".to_string()),
            cutoff: 5.0,
            r_switch: 5.0,
            potential: PairPotentialParams::LennardJones { sigma: 1.0, epsilon: 1.0 },
        },
        PairInteractionConfig {
            between: ("Ar".to_string(), "Kr".to_string()),
            cutoff: 5.0,
            r_switch: 5.0,
            potential: PairPotentialParams::LennardJones { sigma: 2.0, epsilon: 0.5 },
        },
        PairInteractionConfig {
            between: ("Kr".to_string(), "Kr".to_string()),
            cutoff: 5.0,
            r_switch: 5.0,
            potential: PairPotentialParams::LennardJones { sigma: 3.0, epsilon: 2.0 },
        },
    ];
    let table =
        LennardJonesParameterTable::from_config(&device, &particle_types, &pair_interactions)
            .expect("from_config");
    assert_eq!(table.n_types, 2);
    let sigma = device.dtoh_sync_copy(&table.sigma).unwrap();
    let epsilon = device.dtoh_sync_copy(&table.epsilon).unwrap();
    let cutoff = device.dtoh_sync_copy(&table.cutoff).unwrap();
    let switch = device.dtoh_sync_copy(&table.switch).unwrap();
    assert_eq!(sigma, vec![1.0_f32, 2.0, 2.0, 3.0]);
    assert_eq!(epsilon, vec![1.0_f32, 0.5, 0.5, 2.0]);
    assert_eq!(cutoff, vec![5.0_f32, 5.0, 5.0, 5.0]);
    assert_eq!(switch, vec![5.0_f32, 5.0, 5.0, 5.0]);
}

// --- Shared neighbor-list integration ---

#[test] // rq-9004fd7a
fn lennard_jones_state_reports_its_max_cutoff_to_framework() {
    // Build a ForceField with one LJ slot whose largest cutoff is 4.0.
    use dynamics::forces::{
        BondList, ExclusionList, ForceField, Potential,
    };
    use dynamics::io::config::{
        NeighborListConfig, PairInteractionConfig, PairPotentialParams, ParticleTypeConfig,
    };
    let device = init_device().unwrap();
    let particle_types = vec![ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0 }];
    let pair_interactions = vec![PairInteractionConfig {
        between: ("Ar".to_string(), "Ar".to_string()),
        cutoff: 4.0,
        r_switch: 4.0,
        potential: PairPotentialParams::LennardJones { sigma: 1.0, epsilon: 1.0 },
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
        NeighborListConfig, PairInteractionConfig, PairPotentialParams, ParticleTypeConfig,
    };
    let device = init_device().unwrap();
    let sim_box = SimulationBox::new_orthorhombic(20.0, 20.0, 20.0).unwrap();
    let particle_types = vec![ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0 }];
    let pair_interactions = vec![PairInteractionConfig {
        between: ("Ar".to_string(), "Ar".to_string()),
        cutoff: 3.0,
        r_switch: 3.0,
        potential: PairPotentialParams::LennardJones { sigma: 1.0, epsilon: 1.0 },
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

// --- Energy and virial outputs ---

fn download_pair_energies(pair: &PairBuffer) -> Vec<f32> {
    let device = pair.device.clone();
    device.dtoh_sync_copy(&pair.pair_energies).unwrap()
}

fn download_pair_virials(pair: &PairBuffer) -> Vec<f32> {
    let device = pair.device.clone();
    device.dtoh_sync_copy(&pair.pair_virials).unwrap()
}

#[test] // rq-b68b3445
fn two_particle_pair_energy_matches_closed_form() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let params = default_params();
    let table = table_from_scalar(&device, params);
    let positions = [[0.0_f32, 0.0, 0.0], [1.5, 0.0, 0.0]];
    let state = build_state_xyz(&positions);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 2, 2).unwrap();
    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).unwrap();
    let pe = download_pair_energies(&pair);
    let sigma = params.sigma;
    let epsilon = params.epsilon;
    let r = 1.5_f32;
    let sr2 = (sigma / r).powi(2);
    let sr6 = sr2.powi(3);
    let sr12 = sr6 * sr6;
    let expected = 4.0_f32 * epsilon * (sr12 - sr6);
    // Slots (0,1) and (1,0) each hold half the energy.
    assert!((pe[0 * 2 + 1] + pe[1 * 2 + 0] - expected).abs() < 1.0e-5);
}

#[test] // rq-0b71c50a
fn two_particle_pair_virial_matches_r_dot_f() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let params = default_params();
    let table = table_from_scalar(&device, params);
    let positions = [[0.0_f32, 0.0, 0.0], [1.5, 0.0, 0.0]];
    let state = build_state_xyz(&positions);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 2, 2).unwrap();
    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).unwrap();
    let (px, _, _) = download_pair_forces(&pair);
    let pv = download_pair_virials(&pair);
    // r_ij = (1.5, 0, 0); F_ij = (px[0*2+1], 0, 0). w = r · F.
    let r_dot_f = 1.5_f32 * px[0 * 2 + 1];
    // The kernel computes w = (r_i - r_j) · F_i = (-1.5) * (-F_x_on_0) = 1.5 * F_x_on_0.
    // But our convention: slot (i, j) holds force on i due to j, so F_x_on_0 due to 1 is px[0*2+1].
    // Displacement vector used in kernel: r_i - r_j. For i=0, j=1: dx = -1.5.
    // r_ij · F_ij = dx * fx + dy * fy + dz * fz where (fx,fy,fz) is force on 0 due to 1.
    // Here dx=-1.5, fx=px[0*2+1]. So w = -1.5 * px[0*2+1].
    // Symmetrically for slot (1,0): dx=+1.5, fx=-px[0*2+1]. Same w.
    let expected = -1.5_f32 * px[0 * 2 + 1];
    let total = pv[0 * 2 + 1] + pv[1 * 2 + 0];
    assert!((total - expected).abs() < 1.0e-5, "got {total} expected {expected}");
    // Also: the slot's value should be exactly the (i==0,j==1) virial = -1.5*F.
    // px[0*2+1] = px is the same px from `pair`'s force buffer; we verified it.
    let _ = r_dot_f; // silence
}

#[test] // rq-a50cb6a1
fn pair_beyond_cutoff_yields_zero_energy_and_virial() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let params = default_params();
    let table = table_from_scalar(&device, params);
    let positions = [[0.0_f32, 0.0, 0.0], [6.0, 0.0, 0.0]];
    let state = build_state_xyz(&positions);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 2, 2).unwrap();
    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).unwrap();
    let pe = download_pair_energies(&pair);
    let pv = download_pair_virials(&pair);
    assert_eq!(pe[0 * 2 + 1], 0.0);
    assert_eq!(pv[0 * 2 + 1], 0.0);
    assert_eq!(pe[1 * 2 + 0], 0.0);
    assert_eq!(pv[1 * 2 + 0], 0.0);
}

#[test] // rq-82f8d168
fn self_slots_carry_zero_energy_and_virial() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let params = default_params();
    let table = table_from_scalar(&device, params);
    let positions = [
        [0.0_f32, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 1.5],
    ];
    let state = build_state_xyz(&positions);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 4, 4).unwrap();
    lj_pair_force_no_excl(&particle_buffers, &mut pair, &sim_box, &table).unwrap();
    let pe = download_pair_energies(&pair);
    let pv = download_pair_virials(&pair);
    for i in 0..4 {
        let slot = i * 4 + i;
        assert_eq!(pe[slot], 0.0_f32);
        assert_eq!(pv[slot], 0.0_f32);
    }
}

#[test] // rq-95c2f543
fn exclusion_scaling_applies_uniformly_to_force_energy_virial() {
    use dynamics::forces::{DeviceExclusionList, ExclusionList, Exclusion};
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let params = default_params();
    let table = table_from_scalar(&device, params);
    let positions = [[0.0_f32, 0.0, 0.0], [1.5, 0.0, 0.0]];
    let state = build_state_xyz(&positions);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();

    // Unscaled run.
    let mut pair_full = PairBuffer::new(device.clone(), 2, 2).unwrap();
    lj_pair_force_no_excl(&particle_buffers, &mut pair_full, &sim_box, &table).unwrap();
    let pe_full = download_pair_energies(&pair_full);
    let pv_full = download_pair_virials(&pair_full);
    let (px_full, _, _) = download_pair_forces(&pair_full);

    // Half-strength exclusion run. Build the host-side ExclusionList by
    // hand since the only public constructor is `empty(n)`.
    let host = ExclusionList {
        entries: vec![
            Exclusion { atom_i: 0, atom_j: 1, scale: 0.5 },
            Exclusion { atom_i: 1, atom_j: 0, scale: 0.5 },
        ],
        atom_excl_offsets: vec![0u32, 1, 2],
        atom_excl_partners: vec![1u32, 0],
        atom_excl_scales: vec![0.5_f32, 0.5],
        particle_count: 2,
    };
    let excl = DeviceExclusionList::from_host(&device, &host).unwrap();
    let mut pair_half = PairBuffer::new(device.clone(), 2, 2).unwrap();
    dynamics::gpu::lj_pair_force(
        &particle_buffers,
        &mut pair_half,
        &sim_box,
        &table,
        &excl.atom_excl_offsets,
        &excl.atom_excl_partners,
        &excl.atom_excl_scales,
        &trivial_neighbor_list(&device, &sim_box, 2).neighbor_list,
        &trivial_neighbor_list(&device, &sim_box, 2).neighbor_counts,
    )
    .unwrap();
    let pe_half = download_pair_energies(&pair_half);
    let pv_half = download_pair_virials(&pair_half);
    let (px_half, _, _) = download_pair_forces(&pair_half);

    // Every (0,1) and (1,0) slot scaled by 0.5.
    assert!((px_half[0 * 2 + 1] - 0.5 * px_full[0 * 2 + 1]).abs() < 1.0e-5);
    assert!((pe_half[0 * 2 + 1] - 0.5 * pe_full[0 * 2 + 1]).abs() < 1.0e-5);
    assert!((pv_half[0 * 2 + 1] - 0.5 * pv_full[0 * 2 + 1]).abs() < 1.0e-5);
}

// --- Switching function ---

/// Closed-form switched LJ pair force on particle 0 from particle 1
/// along x when the two particles lie on the x axis at separation `r`,
/// computed in f32 with the same arithmetic order the kernel uses
/// (normalised-tau form to avoid the (r_c²−r_s²)³ underflow at SI
/// scales).
fn switched_fx_on_0(r: f32, sigma: f32, epsilon: f32, cutoff: f32, r_switch: f32) -> f32 {
    let r2 = r * r;
    let r_c2 = cutoff * cutoff;
    if r2 > r_c2 {
        return 0.0;
    }
    let inv_r2 = 1.0 / r2;
    let sigma2 = sigma * sigma;
    let sr2 = sigma2 * inv_r2;
    let sr6 = sr2 * sr2 * sr2;
    let sr12 = sr6 * sr6;
    let mut factor = 24.0 * epsilon * inv_r2 * (2.0 * sr12 - sr6);
    let mut energy = 4.0 * epsilon * (sr12 - sr6);
    let r_s2 = r_switch * r_switch;
    if r2 > r_s2 {
        let delta = r_c2 - r_s2;
        let inv_delta = 1.0 / delta;
        let tau = (r2 - r_s2) * inv_delta;
        let one_minus_tau = 1.0 - tau;
        let s = one_minus_tau * one_minus_tau * (1.0 + 2.0 * tau);
        let chain_coeff = 12.0 * tau * one_minus_tau * inv_delta;
        factor = s * factor + chain_coeff * energy;
        energy = s * energy;
    }
    let _ = energy;
    factor * (-r)
}

/// Closed-form switched LJ pair energy at separation `r` (full pair
/// energy, not the half stored per slot).
fn switched_pair_energy(r: f32, sigma: f32, epsilon: f32, cutoff: f32, r_switch: f32) -> f32 {
    let r2 = r * r;
    let r_c2 = cutoff * cutoff;
    if r2 > r_c2 {
        return 0.0;
    }
    let inv_r2 = 1.0 / r2;
    let sigma2 = sigma * sigma;
    let sr2 = sigma2 * inv_r2;
    let sr6 = sr2 * sr2 * sr2;
    let sr12 = sr6 * sr6;
    let mut energy = 4.0 * epsilon * (sr12 - sr6);
    let r_s2 = r_switch * r_switch;
    if r2 > r_s2 {
        let delta = r_c2 - r_s2;
        let inv_delta = 1.0 / delta;
        let tau = (r2 - r_s2) * inv_delta;
        let one_minus_tau = 1.0 - tau;
        let s = one_minus_tau * one_minus_tau * (1.0 + 2.0 * tau);
        energy = s * energy;
    }
    energy
}

fn two_particle_pair(r: f32) -> [[f32; 3]; 2] {
    [[0.0, 0.0, 0.0], [r, 0.0, 0.0]]
}

fn run_lj_two_particle(
    device: &Arc<CudaDevice>,
    sim_box: &SimulationBox,
    positions: [[f32; 3]; 2],
    table: &LennardJonesParameterTable,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let state = build_state_xyz(&positions);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair = PairBuffer::new(device.clone(), 2, 2).unwrap();
    lj_pair_force_no_excl(&particle_buffers, &mut pair, sim_box, table).unwrap();
    let (px, py, pz) = download_pair_forces(&pair);
    let pe = download_pair_energies(&pair);
    let pv = download_pair_virials(&pair);
    (px, py, pz, pe, pv)
}

#[test] // rq-0c4f8da8
fn switching_pair_inside_r_switch_sees_unmodified_lj() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    // cutoff=5, switch=4: r=1.5 is well inside the inner plateau.
    let table = single_type_lj_table_with_switch(&device, 1.0, 1.0, 5.0, 4.0);
    let (px, _, _, pe, _) = run_lj_two_particle(&device, &sim_box, two_particle_pair(1.5), &table);
    let expected_fx = switched_fx_on_0(1.5, 1.0, 1.0, 5.0, 4.0);
    assert!((px[0 * 2 + 1] - expected_fx).abs() < 1.0e-6);
    // Inside the plateau, the switched energy equals the unswitched LJ
    // energy. (Both slots hold half the pair energy.)
    let r = 1.5_f32;
    let sr2 = (1.0_f32 / r).powi(2);
    let sr6 = sr2.powi(3);
    let sr12 = sr6 * sr6;
    let lj_energy = 4.0_f32 * (sr12 - sr6);
    assert!((pe[0 * 2 + 1] + pe[1 * 2 + 0] - lj_energy).abs() < 1.0e-5);
}

#[test] // rq-38441c15
fn switching_pair_exactly_at_r_switch_sees_unmodified_lj() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let table = single_type_lj_table_with_switch(&device, 1.0, 1.0, 5.0, 4.0);
    let (px, _, _, pe, _) = run_lj_two_particle(&device, &sim_box, two_particle_pair(4.0), &table);
    // At r == r_switch the kernel takes the r2 <= r_s2 branch (S=1) and
    // returns the unswitched LJ force/energy.
    let expected_fx = switched_fx_on_0(4.0, 1.0, 1.0, 5.0, 4.0);
    assert!((px[0 * 2 + 1] - expected_fx).abs() < 1.0e-6);
    let r = 4.0_f32;
    let sr2 = (1.0_f32 / r).powi(2);
    let sr6 = sr2.powi(3);
    let sr12 = sr6 * sr6;
    let lj_energy = 4.0_f32 * (sr12 - sr6);
    assert!((pe[0 * 2 + 1] + pe[1 * 2 + 0] - lj_energy).abs() < 1.0e-7);
}

#[test] // rq-f93d278e
fn switching_pair_exactly_at_r_cut_yields_zero_when_switch_less_than_cutoff() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let table = single_type_lj_table_with_switch(&device, 1.0, 1.0, 5.0, 4.0);
    let (px, py, pz, pe, pv) =
        run_lj_two_particle(&device, &sim_box, two_particle_pair(5.0), &table);
    // S(r_c²) = 0 by polynomial form, so all five slots are zero.
    assert_eq!(px[0 * 2 + 1], 0.0);
    assert_eq!(py[0 * 2 + 1], 0.0);
    assert_eq!(pz[0 * 2 + 1], 0.0);
    assert_eq!(pe[0 * 2 + 1], 0.0);
    assert_eq!(pv[0 * 2 + 1], 0.0);
}

#[test] // rq-cb85cf61
fn switching_pair_inside_window_matches_closed_form_switched_value() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let table = single_type_lj_table_with_switch(&device, 1.0, 1.0, 5.0, 4.0);
    let r = 4.5_f32;
    let (px, _, _, pe, _) = run_lj_two_particle(&device, &sim_box, two_particle_pair(r), &table);
    let expected_fx = switched_fx_on_0(r, 1.0, 1.0, 5.0, 4.0);
    let rel = (px[0 * 2 + 1] - expected_fx).abs() / expected_fx.abs().max(1.0e-30);
    assert!(
        rel < 1.0e-4,
        "got {} expected {} (rel {rel})",
        px[0 * 2 + 1],
        expected_fx
    );
    let expected_energy = switched_pair_energy(r, 1.0, 1.0, 5.0, 4.0);
    let got_energy = pe[0 * 2 + 1] + pe[1 * 2 + 0];
    assert!(
        (got_energy - expected_energy).abs() < 1.0e-6,
        "got {got_energy} expected {expected_energy}"
    );
}

#[test] // rq-ae20ddac
fn switching_force_is_c1_continuous_at_r_switch() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let table = single_type_lj_table_with_switch(&device, 1.0, 1.0, 5.0, 4.0);
    let eps = 1.0e-3_f32;
    let (px_below, _, _, _, _) =
        run_lj_two_particle(&device, &sim_box, two_particle_pair(4.0 - eps), &table);
    let (px_above, _, _, _, _) =
        run_lj_two_particle(&device, &sim_box, two_particle_pair(4.0 + eps), &table);
    let f_below = px_below[0 * 2 + 1];
    let f_above = px_above[0 * 2 + 1];
    let denom = f_below.abs().max(1.0e-10);
    let rel = (f_below - f_above).abs() / denom;
    assert!(
        rel < 1.0e-2,
        "force jump across r_switch: {f_below} vs {f_above} (rel {rel})"
    );
}

#[test] // rq-e5e3443f
fn switching_force_is_c1_continuous_at_r_cut() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let table = single_type_lj_table_with_switch(&device, 1.0, 1.0, 5.0, 4.0);
    let eps = 1.0e-3_f32;
    let (px_inside, _, _, _, _) =
        run_lj_two_particle(&device, &sim_box, two_particle_pair(5.0 - eps), &table);
    let f_inside = px_inside[0 * 2 + 1].abs();
    let f_at_rs = switched_fx_on_0(4.0, 1.0, 1.0, 5.0, 4.0).abs();
    assert!(
        f_inside < 1.0e-2 * f_at_rs,
        "expected force near r_cut to be small relative to force at r_switch: \
         f_inside={f_inside} f_at_rs={f_at_rs}"
    );
}

#[test] // rq-916f99f3
fn switching_degenerate_reproduces_hard_cutoff_everywhere_inside() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let table = single_type_lj_table_with_switch(&device, 1.0, 1.0, 5.0, 5.0);
    for r in [1.5_f32, 3.0, 4.5] {
        let (px, _, _, pe, _) =
            run_lj_two_particle(&device, &sim_box, two_particle_pair(r), &table);
        let expected_fx = lj_force_components(
            [0.0, 0.0, 0.0],
            [r, 0.0, 0.0],
            sim_box.lengths(),
            LjScalarParams { sigma: 1.0, epsilon: 1.0, cutoff: 5.0 },
        )[0];
        assert_eq!(px[0 * 2 + 1], expected_fx, "force at r={r}");
        let sr2 = (1.0_f32 / r).powi(2);
        let sr6 = sr2.powi(3);
        let sr12 = sr6 * sr6;
        let lj_energy = 4.0_f32 * (sr12 - sr6);
        assert!(
            (pe[0 * 2 + 1] + pe[1 * 2 + 0] - lj_energy).abs() < 1.0e-6,
            "energy at r={r}"
        );
    }
}

#[test] // rq-531afe39
fn switching_pair_beyond_r_cut_yields_zero_independent_of_r_switch() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let table = single_type_lj_table_with_switch(&device, 1.0, 1.0, 5.0, 4.0);
    let (px, py, pz, pe, pv) =
        run_lj_two_particle(&device, &sim_box, two_particle_pair(6.0), &table);
    assert_eq!(px[0 * 2 + 1], 0.0);
    assert_eq!(py[0 * 2 + 1], 0.0);
    assert_eq!(pz[0 * 2 + 1], 0.0);
    assert_eq!(pe[0 * 2 + 1], 0.0);
    assert_eq!(pv[0 * 2 + 1], 0.0);
}

#[test] // rq-d0f489d7
fn switching_pair_virial_inside_window_equals_factor_switched_times_r2() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let table = single_type_lj_table_with_switch(&device, 1.0, 1.0, 5.0, 4.0);
    let r = 4.5_f32;
    let (px, _, _, _, pv) =
        run_lj_two_particle(&device, &sim_box, two_particle_pair(r), &table);
    // The pair virial is r_ij · F_ij = factor_new * r². For two particles
    // on the x axis at separation r, that equals -r * fx_on_0 (dx = -r
    // for particle 0, F on 0 has component +factor*(-r) = px[0*2+1]).
    // Equivalently the kernel writes w = fx*dx + 0 + 0 = -r * px[0*2+1].
    let expected_w = -r * px[0 * 2 + 1];
    let got_total = pv[0 * 2 + 1] + pv[1 * 2 + 0];
    assert!(
        (got_total - expected_w).abs() < 1.0e-6,
        "got {got_total} expected {expected_w}"
    );
}

#[test] // rq-ef8013be
fn switching_newtons_third_law_holds_bitwise_across_window() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let table = single_type_lj_table_with_switch(&device, 1.0, 1.0, 5.0, 4.0);
    let positions = [[0.0_f32, 0.0, 0.0], [4.5, 0.4, -0.2]];
    let (px, py, pz, _, _) = run_lj_two_particle(&device, &sim_box, positions, &table);
    assert_eq!(px[0 * 2 + 1], -px[1 * 2 + 0]);
    assert_eq!(py[0 * 2 + 1], -py[1 * 2 + 0]);
    assert_eq!(pz[0 * 2 + 1], -pz[1 * 2 + 0]);
}

#[test] // rq-fb55af77
fn switching_exclusion_scaling_multiplies_switched_quantities() {
    use dynamics::forces::{DeviceExclusionList, Exclusion, ExclusionList};
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let table = single_type_lj_table_with_switch(&device, 1.0, 1.0, 5.0, 4.0);
    let positions = two_particle_pair(4.5);
    let state = build_state_xyz(&positions);
    let particle_buffers = ParticleBuffers::new(device.clone(), &state).unwrap();

    // Unscaled (empty exclusion list).
    let mut pair_full = PairBuffer::new(device.clone(), 2, 2).unwrap();
    lj_pair_force_no_excl(&particle_buffers, &mut pair_full, &sim_box, &table).unwrap();
    let (px_full, _, _) = download_pair_forces(&pair_full);
    let pe_full = download_pair_energies(&pair_full);
    let pv_full = download_pair_virials(&pair_full);

    // Half-strength exclusion: scale = 0.5 on the (0,1) pair.
    let host = ExclusionList {
        entries: vec![
            Exclusion { atom_i: 0, atom_j: 1, scale: 0.5 },
            Exclusion { atom_i: 1, atom_j: 0, scale: 0.5 },
        ],
        atom_excl_offsets: vec![0u32, 1, 2],
        atom_excl_partners: vec![1u32, 0],
        atom_excl_scales: vec![0.5_f32, 0.5],
        particle_count: 2,
    };
    let excl = DeviceExclusionList::from_host(&device, &host).unwrap();
    let mut pair_half = PairBuffer::new(device.clone(), 2, 2).unwrap();
    let nl = trivial_neighbor_list(&device, &sim_box, 2);
    dynamics::gpu::lj_pair_force(
        &particle_buffers,
        &mut pair_half,
        &sim_box,
        &table,
        &excl.atom_excl_offsets,
        &excl.atom_excl_partners,
        &excl.atom_excl_scales,
        &nl.neighbor_list,
        &nl.neighbor_counts,
    )
    .unwrap();
    let (px_half, _, _) = download_pair_forces(&pair_half);
    let pe_half = download_pair_energies(&pair_half);
    let pv_half = download_pair_virials(&pair_half);

    assert!((px_half[0 * 2 + 1] - 0.5 * px_full[0 * 2 + 1]).abs() < 1.0e-6);
    assert!((pe_half[0 * 2 + 1] - 0.5 * pe_full[0 * 2 + 1]).abs() < 1.0e-6);
    assert!((pv_half[0 * 2 + 1] - 0.5 * pv_full[0 * 2 + 1]).abs() < 1.0e-6);
}

#[test] // rq-37f8c017
fn switching_per_pair_type_r_switch_dispatches_correctly() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    // n_types = 2. Diagonal slots (0,0) and (1,1) have r_switch = 5.0
    // (no switching); off-diagonal (0,1) and (1,0) have r_switch = 4.0
    // (active switching at r = 4.5).
    let sigma = vec![1.0_f32, 1.0, 1.0, 1.0];
    let epsilon = vec![1.0_f32, 1.0, 1.0, 1.0];
    let cutoff = vec![5.0_f32, 5.0, 5.0, 5.0];
    let switch = vec![5.0_f32, 4.0, 4.0, 5.0];
    let table = build_table_with_switch(&device, 2, &sigma, &epsilon, &cutoff, &switch);
    let r = 4.5_f32;
    let positions = [[0.0_f32, 0.0, 0.0], [r, 0.0, 0.0]];

    // Mixed-type pair: uses the off-diagonal switch = 4.0.
    let state_mixed = build_state_with_types(&positions, vec![0, 1]);
    let buffers_mixed = ParticleBuffers::new(device.clone(), &state_mixed).unwrap();
    let mut pair_mixed = PairBuffer::new(device.clone(), 2, 2).unwrap();
    lj_pair_force_no_excl(&buffers_mixed, &mut pair_mixed, &sim_box, &table).unwrap();
    let (px_mixed, _, _) = download_pair_forces(&pair_mixed);
    let expected_mixed = switched_fx_on_0(r, 1.0, 1.0, 5.0, 4.0);
    let rel_mixed = (px_mixed[0 * 2 + 1] - expected_mixed).abs() / expected_mixed.abs().max(1.0e-30);
    assert!(rel_mixed < 1.0e-4, "mixed: got {} expected {}", px_mixed[0 * 2 + 1], expected_mixed);

    // Same-type pair: uses the diagonal switch = 5.0 (hard-cutoff
    // degenerate; returns the unswitched LJ value).
    let state_same = build_state_with_types(&positions, vec![0, 0]);
    let buffers_same = ParticleBuffers::new(device.clone(), &state_same).unwrap();
    let mut pair_same = PairBuffer::new(device.clone(), 2, 2).unwrap();
    lj_pair_force_no_excl(&buffers_same, &mut pair_same, &sim_box, &table).unwrap();
    let (px_same, _, _) = download_pair_forces(&pair_same);
    let expected_same = lj_force_components(
        positions[0],
        positions[1],
        sim_box.lengths(),
        LjScalarParams { sigma: 1.0, epsilon: 1.0, cutoff: 5.0 },
    )[0];
    assert_eq!(px_same[0 * 2 + 1], expected_same);
}

#[test] // rq-dbd3c689
fn switching_bit_exact_reproducibility_across_runs() {
    let device = init_device().expect("init_device");
    let sim_box = default_box();
    let table = single_type_lj_table_with_switch(&device, 1.0, 1.0, 5.0, 4.0);
    let n = 64;
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let fi = i as f32;
            // Place atoms over a range that crosses the switching window
            // [4.0, 5.0] so the kernel exercises both branches.
            [fi * 0.15 - 4.8, (fi * 0.3).sin() * 1.5, (fi * 0.7).cos() * 1.2]
        })
        .collect();
    let state = build_state_xyz(&positions);

    let bufs_a = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair_a = PairBuffer::new(device.clone(), n, n as u32).unwrap();
    lj_pair_force_no_excl(&bufs_a, &mut pair_a, &sim_box, &table).unwrap();
    let (ax, ay, az) = download_pair_forces(&pair_a);
    let ae = download_pair_energies(&pair_a);
    let av = download_pair_virials(&pair_a);

    let bufs_b = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair_b = PairBuffer::new(device.clone(), n, n as u32).unwrap();
    lj_pair_force_no_excl(&bufs_b, &mut pair_b, &sim_box, &table).unwrap();
    let (bx, by, bz) = download_pair_forces(&pair_b);
    let be = download_pair_energies(&pair_b);
    let bv = download_pair_virials(&pair_b);

    assert_eq!(ax, bx);
    assert_eq!(ay, by);
    assert_eq!(az, bz);
    assert_eq!(ae, be);
    assert_eq!(av, bv);
}

#[test] // rq-214639c9
fn switching_from_config_populates_user_supplied_r_switch() {
    use dynamics::io::config::{PairInteractionConfig, PairPotentialParams, ParticleTypeConfig};
    let device = init_device().expect("init_device");
    let particle_types = vec![ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0 }];
    let pair_interactions = vec![PairInteractionConfig {
        between: ("Ar".to_string(), "Ar".to_string()),
        cutoff: 5.0,
        r_switch: 4.0,
        potential: PairPotentialParams::LennardJones { sigma: 1.0, epsilon: 1.0 },
    }];
    let table =
        LennardJonesParameterTable::from_config(&device, &particle_types, &pair_interactions)
            .expect("from_config");
    let switch = device.dtoh_sync_copy(&table.switch).unwrap();
    assert_eq!(switch, vec![4.0_f32]);
}

#[test] // rq-6a542a0a
fn switching_from_config_receives_default_r_switch_when_omitted() {
    // Round-trip through load_config so the default r_switch = 0.9 *
    // cutoff is exercised end-to-end, then assert the parameter table
    // built from that config carries the default value on every slot.
    use dynamics::io::load_config;
    let device = init_device().expect("init_device");
    let dir = {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "dynamics-switching-default-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).expect("create temp dir");
        p
    };
    let config_path = dir.join("sim.toml");
    std::fs::write(
        &config_path,
        r#"
schema_version = 1
init = "init.xyz"

[simulation]
seed = 1
n_steps = 0
dt = 1.0e-15
temperature = 300.0

[integrator]
kind = "velocity-verlet"

[[particle_types]]
name = "Ar"
mass = 1.0

[[particle_types]]
name = "Kr"
mass = 2.0

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 5.0

[[pair_interactions]]
between = ["Ar", "Kr"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 5.0

[[pair_interactions]]
between = ["Kr", "Kr"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 5.0
"#,
    )
    .unwrap();
    let config = load_config(&config_path).expect("load_config");
    for pi in &config.pair_interactions {
        assert!((pi.r_switch - 4.5).abs() < 1.0e-12);
    }
    let table = LennardJonesParameterTable::from_config(
        &device,
        &config.particle_types,
        &config.pair_interactions,
    )
    .expect("from_config");
    let switch = device.dtoh_sync_copy(&table.switch).unwrap();
    assert_eq!(switch, vec![4.5_f32, 4.5, 4.5, 4.5]);
}
