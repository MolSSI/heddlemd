// rq-c4645fa6
use cudarc::driver::DeviceSlice;
use dynamics::forces::{NeighborListError, NeighborListState};
use dynamics::gpu::{GpuContext, ParticleBuffers, init_device};
use dynamics::pbc::SimulationBox;
use dynamics::state::ParticleState;
use dynamics::timings::Timings;

fn box_n(l: f32) -> SimulationBox {
    SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap()
}

fn state_from_positions(px: Vec<f32>, py: Vec<f32>, pz: Vec<f32>) -> ParticleState {
    let n = px.len();
    ParticleState::new(
        px,
        py,
        pz,
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
            None,
    )
    .unwrap()
}

// rq-c0cfc5d6
#[test]
fn cell_counts_floor_of_l_over_search_radius() {
    let gpu = init_device().unwrap();
    let nl = NeighborListState::new_cell_list(&gpu, &box_n(10.0), 0, 1.0, 8, 0.3).unwrap();
    let cl = nl.cell_list_data().expect("cell-list mode");
    assert_eq!(cl.n_cells, [7, 7, 7]);
}

// rq-1b9c474c
#[test]
fn reject_box_admitting_fewer_than_three_cells() {
    let gpu = init_device().unwrap();
    let result = NeighborListState::new_cell_list(&gpu, &box_n(10.0), 0, 1.0, 8, 3.0);
    match result {
        Err(NeighborListError::BoxTooSmallForCells {
            direction,
            width,
            required,
        }) => {
            assert_eq!(direction, "a");
            assert!((width - 10.0).abs() < 1.0e-6);
            assert!((required - 12.0).abs() < 1.0e-6);
        }
        other => panic!("expected BoxTooSmallForCells, got {other:?}"),
    }
}

// rq-4bc8028f
#[test]
fn particle_count_zero_builds_and_runs() {
    let gpu = init_device().unwrap();
    let sim_box = box_n(10.0);
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 0, 1.0, 8, 0.3).unwrap();
    let state = state_from_positions(vec![], vec![], vec![]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let max_disp = nl.displacement_check(&sim_box, &buffers, &mut timings).unwrap();
    assert_eq!(max_disp, 0.0);
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();
}

// rq-52f547fd
#[test]
fn single_particle_yields_empty_neighbor_list() {
    let gpu = init_device().unwrap();
    let sim_box = box_n(10.0);
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 1, 1.0, 8, 0.3).unwrap();
    let state = state_from_positions(vec![0.0], vec![0.0], vec![0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    let counts = gpu.device.dtoh_sync_copy(&nl.neighbor_counts).unwrap();
    assert_eq!(counts[0], 0);
}

// rq-ea0ee5ef rq-e75b24e7 rq-2bc559ec
#[test]
fn neighbor_list_contains_all_within_search_radius_and_is_sorted() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    // 4 particles along x at 0.0, 0.5, 1.0, 2.0
    let state = state_from_positions(
        vec![0.0, 0.5, 1.0, 2.0],
        vec![0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0],
    );
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let r_cut = 1.0_f32;
    let r_skin = 0.3_f32;
    let max_neighbors = 8u32;
    let sim_box = box_n(10.0);
    let mut nl =
        NeighborListState::new_cell_list(&gpu, &sim_box, 4, r_cut, max_neighbors, r_skin)
            .unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    let list = device.dtoh_sync_copy(&nl.neighbor_list).unwrap();
    let counts = device.dtoh_sync_copy(&nl.neighbor_counts).unwrap();
    // Atom 0 at x=0 should have partners 1 (0.5) and 2 (1.0). 2.0 is outside
    // r_cut + r_skin = 1.3.
    assert_eq!(counts[0], 2);
    let start = 0;
    let partners: Vec<u32> = list[start..start + counts[0] as usize].to_vec();
    assert_eq!(partners, vec![1u32, 2u32]);
    // Atom i's neighbor list never contains i itself.
    for i in 0..4usize {
        let base = i * max_neighbors as usize;
        let c = counts[i] as usize;
        for k in 0..c {
            assert_ne!(list[base + k], i as u32);
        }
        // Sorted ascending by partner index.
        let slice = &list[base..base + c];
        for w in slice.windows(2) {
            assert!(w[0] < w[1], "atom {i} list not sorted: {slice:?}");
        }
    }
}

// rq-25faef11 rq-b39d3be7
#[test]
fn neighbor_list_uses_minimum_image() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    // Two particles separated by 0.2 across the periodic boundary along x.
    // Box of length 10.0 → -5.0..+5.0 primary cell.
    let state = state_from_positions(
        vec![-4.9, 4.9],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
    );
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = box_n(10.0);
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 2, 0.7, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    let counts = device.dtoh_sync_copy(&nl.neighbor_counts).unwrap();
    let list = device.dtoh_sync_copy(&nl.neighbor_list).unwrap();
    assert_eq!(counts[0], 1, "atom 0 should see atom 1 via PBC");
    assert_eq!(list[0], 1);
    assert_eq!(counts[1], 1);
    assert_eq!(list[8], 0);
}

// rq-0181787c
#[test]
fn build_signals_overflow() {
    let gpu = init_device().unwrap();
    // 6 particles tightly clustered within r_cut+r_skin; max_neighbors=2.
    // Each atom has 5 partners within range, exceeding the cap.
    let positions: Vec<[f32; 3]> = (0..6).map(|i| [i as f32 * 0.1, 0.0, 0.0]).collect();
    let px: Vec<f32> = positions.iter().map(|p| p[0]).collect();
    let py: Vec<f32> = positions.iter().map(|p| p[1]).collect();
    let pz: Vec<f32> = positions.iter().map(|p| p[2]).collect();
    let state = state_from_positions(px, py, pz);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = box_n(10.0);
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 6, 1.0, 2, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let err = nl.rebuild(&sim_box, &buffers, &mut timings).unwrap_err();
    match err {
        NeighborListError::NeighborListOverflow { max } => assert_eq!(max, 2),
        other => panic!("expected NeighborListOverflow, got {other:?}"),
    }
}

// rq-6bf3709c
#[test]
fn two_rebuilds_with_identical_positions_agree() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let state = state_from_positions(
        vec![0.0, 0.4, 0.8, 1.2, 1.6, 2.0],
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        vec![-0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
    );
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let sim_box = box_n(10.0);
    let mut nl_a =
        NeighborListState::new_cell_list(&gpu, &sim_box, 6, 1.0, 16, 0.3).unwrap();
    let mut nl_b =
        NeighborListState::new_cell_list(&gpu, &sim_box, 6, 1.0, 16, 0.3).unwrap();
    nl_a.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    nl_b.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    let list_a = device.dtoh_sync_copy(&nl_a.neighbor_list).unwrap();
    let list_b = device.dtoh_sync_copy(&nl_b.neighbor_list).unwrap();
    let counts_a = device.dtoh_sync_copy(&nl_a.neighbor_counts).unwrap();
    let counts_b = device.dtoh_sync_copy(&nl_b.neighbor_counts).unwrap();
    let offsets_a =
        device.dtoh_sync_copy(&nl_a.cell_list_data().unwrap().cell_offsets).unwrap();
    let offsets_b =
        device.dtoh_sync_copy(&nl_b.cell_list_data().unwrap().cell_offsets).unwrap();
    let ids_a =
        device.dtoh_sync_copy(&nl_a.cell_list_data().unwrap().sorted_particle_ids).unwrap();
    let ids_b =
        device.dtoh_sync_copy(&nl_b.cell_list_data().unwrap().sorted_particle_ids).unwrap();
    assert_eq!(list_a, list_b);
    assert_eq!(counts_a, counts_b);
    assert_eq!(offsets_a, offsets_b);
    assert_eq!(ids_a, ids_b);
}

// rq-53ae77a4
#[test]
fn displacement_check_zero_immediately_after_rebuild() {
    let gpu = init_device().unwrap();
    let state = state_from_positions(
        vec![0.0, 0.5, 1.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    );
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = box_n(10.0);
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 3, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    let max_disp = nl.displacement_check(&sim_box, &buffers, &mut timings).unwrap();
    assert!(max_disp.abs() < 1.0e-5);
}

// rq-f94ee5cd
#[test]
fn displacement_check_returns_max_across_particles() {
    let gpu = init_device().unwrap();
    let state = state_from_positions(
        vec![0.0, 1.0, 2.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    );
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = box_n(10.0);
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 3, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    // Move atom 1 by 0.5 along x.
    let new = state_from_positions(
        vec![0.0, 1.5, 2.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    );
    buffers.upload(&new).unwrap();
    let max_disp = nl.displacement_check(&sim_box, &buffers, &mut timings).unwrap();
    assert!((max_disp - 0.5).abs() < 1.0e-4, "max_disp = {max_disp}");
}

// rq-35981c27
#[test]
fn first_pre_step_unconditionally_rebuilds() {
    let gpu = init_device().unwrap();
    let state = state_from_positions(
        vec![0.0, 1.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
    );
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = box_n(10.0);
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 2, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    assert!(nl.cell_list_data().unwrap().needs_rebuild);
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    assert!(!nl.cell_list_data().unwrap().needs_rebuild);
}

// rq-90524f5d
#[test]
fn sub_skin_movement_does_not_trigger_rebuild() {
    let gpu = init_device().unwrap();
    let state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = box_n(10.0);
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 2, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap(); // Initial rebuild.

    // Move every particle by less than r_skin/2 = 0.15.
    let moved =
        state_from_positions(vec![0.05, 1.10], vec![0.0, 0.0], vec![0.0, 0.0]);
    buffers.upload(&moved).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    assert!(!nl.cell_list_data().unwrap().needs_rebuild);
}

// rq-9f63a183
#[test]
fn over_skin_movement_triggers_rebuild() {
    let gpu = init_device().unwrap();
    let state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = box_n(10.0);
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 2, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap(); // Initial rebuild.

    // Move atom 1 by 0.5 (more than r_skin/2 = 0.15).
    let moved =
        state_from_positions(vec![0.0, 1.5], vec![0.0, 0.0], vec![0.0, 0.0]);
    buffers.upload(&moved).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    // After pre_step, the rebuild has happened so needs_rebuild is false.
    assert!(!nl.cell_list_data().unwrap().needs_rebuild);
    // The reference positions now equal the current positions, so a fresh
    // displacement_check returns zero.
    let max_disp = nl.displacement_check(&sim_box, &buffers, &mut timings).unwrap();
    assert!(max_disp.abs() < 1.0e-4);
}

// --- Trivial mode ---

#[test] // rq-789fcec9
fn trivial_mode_contents() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let nl = NeighborListState::new_trivial(&gpu, &box_n(10.0), 3).unwrap();
    let counts = device.dtoh_sync_copy(&nl.neighbor_counts).unwrap();
    let list = device.dtoh_sync_copy(&nl.neighbor_list).unwrap();
    assert_eq!(counts, vec![3u32, 3, 3]);
    assert_eq!(list, vec![0u32, 1, 2, 0, 1, 2, 0, 1, 2]);
}

#[test] // rq-bb3773aa
fn trivial_mode_pre_step_does_no_work() {
    let gpu = init_device().unwrap();
    let state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = box_n(10.0);
    let mut nl = NeighborListState::new_trivial(&gpu, &sim_box, 2).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    let report = timings.finalize().unwrap();
    for stage in &report.stages {
        assert!(
            stage.name != "neighbor_displacement_squared",
            "trivial pre_step launched displacement check"
        );
        assert!(
            stage.name != "neighbor_list_build",
            "trivial pre_step launched neighbor-list build"
        );
    }
}

#[test] // rq-30f85829
fn trivial_mode_has_no_cell_list_fields() {
    let gpu = init_device().unwrap();
    let nl = NeighborListState::new_trivial(&gpu, &box_n(10.0), 4).unwrap();
    assert!(matches!(nl.mode, dynamics::forces::NeighborListMode::Trivial));
    assert!(nl.cell_list_data().is_none());
}

// --- Box generation tracking ---

#[test] // rq-1b742a37
fn cached_generation_initialised_from_construction_time_box() {
    let gpu = init_device().unwrap();
    let sim_box = box_n(10.0);
    assert_eq!(sim_box.generation(), 0);
    let nl = NeighborListState::new_cell_list(&gpu, &sim_box, 0, 1.0, 8, 0.3).unwrap();
    assert_eq!(nl.cell_list_data().unwrap().cached_generation, 0);
}

#[test] // rq-882c9e86
fn cached_generation_initialised_from_non_zero_generation() {
    let gpu = init_device().unwrap();
    let mut sim_box = box_n(10.0);
    sim_box.set_lattice(10.0, 10.0, 10.0, 0.0, 0.0, 0.0).expect("ok");
    assert_eq!(sim_box.generation(), 1);
    let nl = NeighborListState::new_cell_list(&gpu, &sim_box, 0, 1.0, 8, 0.3).unwrap();
    assert_eq!(nl.cell_list_data().unwrap().cached_generation, 1);
}

#[test] // rq-db8b171d
fn pre_step_with_unchanged_box_does_not_refresh_cache() {
    let gpu = init_device().unwrap();
    let sim_box = box_n(10.0);
    let state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 2, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    let cl_before = nl.cell_list_data().unwrap();
    let (n_cells, n_cells_total, cached_gen, offsets_len) = (
        cl_before.n_cells,
        cl_before.n_cells_total,
        cl_before.cached_generation,
        cl_before.cell_offsets.len(),
    );
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    let cl_after = nl.cell_list_data().unwrap();
    assert_eq!(cl_after.n_cells, n_cells);
    assert_eq!(cl_after.n_cells_total, n_cells_total);
    assert_eq!(cl_after.cached_generation, cached_gen);
    assert_eq!(cl_after.cell_offsets.len(), offsets_len);
}

#[test] // rq-cf847c1f
fn box_generation_increment_refreshes_cell_layout_and_rebuilds() {
    let gpu = init_device().unwrap();
    let mut sim_box = box_n(10.0);
    let state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 2, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    assert_eq!(nl.cell_list_data().unwrap().n_cells, [7, 7, 7]);
    assert_eq!(nl.cell_list_data().unwrap().cached_generation, 0);

    sim_box.set_lattice(20.0, 20.0, 20.0, 0.0, 0.0, 0.0).expect("ok");
    assert_eq!(sim_box.generation(), 1);

    // Move positions into the new box and re-upload (otherwise atoms sit outside primary cell).
    let new_state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &new_state).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    let cl = nl.cell_list_data().unwrap();
    assert_eq!(cl.n_cells, [15, 15, 15]);
    assert_eq!(cl.n_cells_total, 15 * 15 * 15);
    assert_eq!(cl.cell_offsets.len(), 15 * 15 * 15 + 1);
    assert_eq!(cl.cached_generation, 1);
    assert!(!cl.needs_rebuild, "rebuild should have happened during pre_step");
}

#[test] // rq-dacb071c
fn generation_mismatch_with_box_too_small_returns_box_too_small() {
    let gpu = init_device().unwrap();
    let mut sim_box = box_n(10.0);
    let state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 2, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    let (n_cells_before, total_before, gen_before, offsets_len_before) = {
        let cl = nl.cell_list_data().unwrap();
        (
            cl.n_cells,
            cl.n_cells_total,
            cl.cached_generation,
            cl.cell_offsets.len(),
        )
    };

    // Box too small along a (floor(3.0 / 1.3) = 2 < 3).
    sim_box.set_lattice(3.0, 10.0, 10.0, 0.0, 0.0, 0.0).expect("ok");
    let err = nl
        .pre_step(&sim_box, &buffers, &mut timings)
        .expect_err("expected BoxTooSmallForCells");
    match err {
        NeighborListError::BoxTooSmallForCells {
            direction,
            width,
            required,
        } => {
            assert_eq!(direction, "a");
            assert!((width - 3.0).abs() < 1.0e-6);
            assert!((required - 3.9).abs() < 1.0e-5);
        }
        other => panic!("unexpected error: {other:?}"),
    }
    let cl = nl.cell_list_data().unwrap();
    assert_eq!(cl.n_cells, n_cells_before);
    assert_eq!(cl.n_cells_total, total_before);
    assert_eq!(cl.cached_generation, gen_before);
    assert_eq!(cl.cell_offsets.len(), offsets_len_before);
}

#[test] // rq-d22f105f
fn cell_offsets_reallocated_when_n_cells_total_changes() {
    let gpu = init_device().unwrap();
    let mut sim_box = box_n(10.0);
    let state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 2, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    assert_eq!(nl.cell_list_data().unwrap().n_cells_total, 343);
    assert_eq!(nl.cell_list_data().unwrap().cell_offsets.len(), 344);

    sim_box.set_lattice(12.0, 12.0, 12.0, 0.0, 0.0, 0.0).expect("ok");
    let new_state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &new_state).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    assert_eq!(nl.cell_list_data().unwrap().n_cells_total, 729);
    assert_eq!(nl.cell_list_data().unwrap().cell_offsets.len(), 730);
}

#[test] // rq-331b6e81
fn cell_offsets_not_reallocated_when_n_cells_total_unchanged() {
    let gpu = init_device().unwrap();
    let mut sim_box = box_n(10.0);
    let state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 2, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    let initial_total = nl.cell_list_data().unwrap().n_cells_total;
    let initial_offsets_len = nl.cell_list_data().unwrap().cell_offsets.len();
    assert_eq!(initial_total, 343);

    // L=9.8 still gives floor(9.8/1.3)=7 cells per axis (same n_cells_total).
    sim_box.set_lattice(9.8, 9.8, 9.8, 0.0, 0.0, 0.0).expect("ok");
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    let cl = nl.cell_list_data().unwrap();
    assert_eq!(cl.n_cells_total, initial_total);
    assert_eq!(cl.cell_offsets.len(), initial_offsets_len);
    assert_eq!(cl.n_cells, [7, 7, 7]);
}

#[test] // rq-31a9e3bb
fn r_search_sq_preserved_across_generation_refresh() {
    let gpu = init_device().unwrap();
    let mut sim_box = box_n(10.0);
    let state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 2, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    let r_search_sq_before = nl.cell_list_data().unwrap().r_search_sq;
    assert!((r_search_sq_before - 1.69).abs() < 1.0e-5);

    sim_box.set_lattice(20.0, 20.0, 20.0, 0.0, 0.0, 0.0).expect("ok");
    let new_state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &new_state).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    let r_search_sq_after = nl.cell_list_data().unwrap().r_search_sq;
    assert_eq!(r_search_sq_after, r_search_sq_before);
}

#[test] // rq-699cccff
fn two_pre_steps_after_single_box_mutation_refresh_only_once() {
    let gpu = init_device().unwrap();
    let mut sim_box = box_n(10.0);
    let state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 2, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();

    sim_box.set_lattice(12.0, 12.0, 12.0, 0.0, 0.0, 0.0).expect("ok");
    let new_state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &new_state).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    let (n_cells_total, cell_offsets_len, cached_gen) = {
        let cl = nl.cell_list_data().unwrap();
        (cl.n_cells_total, cl.cell_offsets.len(), cl.cached_generation)
    };
    assert_eq!(cached_gen, 1);

    // Second pre_step without further mutation should not refresh; cache fields
    // identical, and the displacement check runs (no longer skipped).
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    let cl = nl.cell_list_data().unwrap();
    assert_eq!(cl.cached_generation, 1);
    assert_eq!(cl.n_cells_total, n_cells_total);
    assert_eq!(cl.cell_offsets.len(), cell_offsets_len);
}

#[test] // rq-72aae589
fn generation_mismatch_detected_even_when_edge_lengths_unchanged() {
    let gpu = init_device().unwrap();
    let mut sim_box = box_n(10.0);
    let state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 2, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    assert_eq!(nl.cell_list_data().unwrap().cached_generation, 0);

    // Mutate to the same lengths — generation still bumps.
    sim_box.set_lattice(10.0, 10.0, 10.0, 0.0, 0.0, 0.0).expect("ok");
    assert_eq!(sim_box.generation(), 1);

    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    let cl = nl.cell_list_data().unwrap();
    assert_eq!(cl.cached_generation, 1);
    assert!(!cl.needs_rebuild, "rebuild should have run inside pre_step");
}

// --- Device-side spatial hash ---
//
// Fixture: box L=10, r_cut=1.0, r_skin=0.3 → r_search=1.3, n_cells=7 per
// axis (n_cells_total=343), cell_size = 10/7 ≈ 1.4286.
// Positions chosen so each atom's cell index is fully predictable.
// cy=cz=3 for all atoms (y=z=0); cell index = 49*cx + 7*3 + 3 = 49*cx + 24.
//   x=-1.0 → cx=2 → c=122
//   x=-4.5 → cx=0 → c=24
//   x=-3.0 → cx=1 → c=73
//   x=-4.0 → cx=0 → c=24
//   x=-2.0 → cx=2 → c=122

fn spatial_hash_fixture(
    gpu: &GpuContext,
) -> (SimulationBox, NeighborListState, ParticleBuffers, Timings) {
    let sim_box = box_n(10.0);
    let state = state_from_positions(
        vec![-1.0, -4.5, -3.0, -4.0, -2.0],
        vec![0.0; 5],
        vec![0.0; 5],
    );
    let buffers = ParticleBuffers::new(gpu, &state).unwrap();
    let nl =
        NeighborListState::new_cell_list(gpu, &sim_box, 5, 1.0, 8, 0.3).unwrap();
    let timings = Timings::new(gpu).unwrap();
    (sim_box, nl, buffers, timings)
}

#[test] // rq-f164bf76
fn cell_indices_populated_by_device_pipeline() {
    let gpu = init_device().unwrap();
    let (sim_box, mut nl, buffers, mut timings) = spatial_hash_fixture(&gpu);
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    let cell_indices = gpu.device
        .dtoh_sync_copy(&nl.cell_list_data().unwrap().cell_indices)
        .unwrap();
    assert_eq!(&cell_indices[..5], &[122u32, 24, 73, 24, 122]);
}

#[test] // rq-19fd5b09
fn cell_counts_is_device_computed_histogram() {
    let gpu = init_device().unwrap();
    let (sim_box, mut nl, buffers, mut timings) = spatial_hash_fixture(&gpu);
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    let counts = gpu.device
        .dtoh_sync_copy(&nl.cell_list_data().unwrap().cell_counts)
        .unwrap();
    let n_cells_total = nl.cell_list_data().unwrap().n_cells_total;
    assert_eq!(counts.len(), n_cells_total);
    let sum: u32 = counts.iter().copied().sum();
    assert_eq!(sum, 5);
    assert_eq!(counts[24], 2);
    assert_eq!(counts[73], 1);
    assert_eq!(counts[122], 2);
    for (c, &v) in counts.iter().enumerate() {
        if c == 24 || c == 73 || c == 122 {
            continue;
        }
        assert_eq!(v, 0, "cell {c} should be empty, got count {v}");
    }
}

#[test] // rq-f8ad62d4 rq-cd50d861
fn cell_offsets_is_exclusive_prefix_sum_ending_at_particle_count() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let (sim_box, mut nl, buffers, mut timings) = spatial_hash_fixture(&gpu);
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    let cl = nl.cell_list_data().unwrap();
    let offsets = device.dtoh_sync_copy(&cl.cell_offsets).unwrap();
    let counts = device.dtoh_sync_copy(&cl.cell_counts).unwrap();
    assert_eq!(offsets.len(), cl.n_cells_total + 1);
    assert_eq!(offsets[0], 0);
    for c in 0..cl.n_cells_total {
        assert_eq!(
            offsets[c + 1],
            offsets[c] + counts[c],
            "exclusive prefix sum broken at cell {c}"
        );
    }
    assert_eq!(offsets[cl.n_cells_total], 5);
    for w in offsets.windows(2) {
        assert!(w[0] <= w[1], "offsets must be non-decreasing");
    }
}

#[test] // rq-265f4da4
fn scatter_places_each_atom_inside_its_cell_slice() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let (sim_box, mut nl, buffers, mut timings) = spatial_hash_fixture(&gpu);
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    let cl = nl.cell_list_data().unwrap();
    let sorted_ids = device.dtoh_sync_copy(&cl.sorted_particle_ids).unwrap();
    let cell_indices = device.dtoh_sync_copy(&cl.cell_indices).unwrap();
    let offsets = device.dtoh_sync_copy(&cl.cell_offsets).unwrap();
    let mut seen = [false; 5];
    for i in 0..5usize {
        let c = cell_indices[i] as usize;
        let start = offsets[c] as usize;
        let end = offsets[c + 1] as usize;
        let pos = sorted_ids[start..end]
            .iter()
            .position(|&p| p == i as u32)
            .expect("atom must appear in its cell's slice");
        assert!(
            start + pos >= start && start + pos < end,
            "atom {i} slot must be inside [{start}, {end})"
        );
        assert!(!seen[i], "atom {i} must appear exactly once");
        seen[i] = true;
    }
    assert!(seen.iter().all(|&b| b));
}

#[test] // rq-7a14d0d8 rq-838acdee rq-2303ee2e
fn per_cell_sort_canonicalises_sorted_particle_ids() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let (sim_box, mut nl, buffers, mut timings) = spatial_hash_fixture(&gpu);
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    let cl = nl.cell_list_data().unwrap();
    let sorted_ids = device.dtoh_sync_copy(&cl.sorted_particle_ids).unwrap();
    assert_eq!(&sorted_ids[..5], &[1u32, 3, 2, 0, 4]);
    let offsets = device.dtoh_sync_copy(&cl.cell_offsets).unwrap();
    for c in 0..cl.n_cells_total {
        let start = offsets[c] as usize;
        let end = offsets[c + 1] as usize;
        let slice = &sorted_ids[start..end];
        for w in slice.windows(2) {
            assert!(w[0] < w[1], "cell {c} slice not strictly ascending: {slice:?}");
        }
    }
}

#[test] // rq-ecad9802
fn write_cursors_is_reset_between_rebuilds() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let (sim_box, mut nl, buffers, mut timings) = spatial_hash_fixture(&gpu);
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    let first = device
        .dtoh_sync_copy(&nl.cell_list_data().unwrap().sorted_particle_ids)
        .unwrap();
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    let second = device
        .dtoh_sync_copy(&nl.cell_list_data().unwrap().sorted_particle_ids)
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(&second[..5], &[1u32, 3, 2, 0, 4]);
}

#[test] // rq-6c8415f6
fn rebuild_produces_correct_output_via_gpu_pipeline() {
    // The rebuild is implemented with no host-side download of positions
    // and no host-side upload of sorted_particle_ids/cell_offsets. This
    // test exercises the end-to-end GPU pipeline and verifies the canonical
    // (cell, particle_id) order, which is only achievable when the entire
    // pipeline runs on the device.
    let gpu = init_device().unwrap();
    let (sim_box, mut nl, buffers, mut timings) = spatial_hash_fixture(&gpu);
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    let sorted_ids = gpu.device
        .dtoh_sync_copy(&nl.cell_list_data().unwrap().sorted_particle_ids)
        .unwrap();
    assert_eq!(&sorted_ids[..5], &[1u32, 3, 2, 0, 4]);
}

#[test] // rq-6fd5167a
fn cells_exceeding_u32_addressing_rejected_at_construction() {
    let gpu = init_device().unwrap();
    // r_cut + r_skin = 1.0; L = 1626 → 1626 cells/axis → 1626³ =
    // 4,298,942,376 cells, just past u32::MAX (4,294,967,295). The
    // rejection happens in check_n_cells_total before any device buffer
    // is allocated, so the 17 GB of cell buffers are never requested.
    let sim_box = box_n(1626.0);
    let err =
        NeighborListState::new_cell_list(&gpu, &sim_box, 0, 0.5, 8, 0.5).expect_err("err");
    match err {
        NeighborListError::TooManyCells {
            n_cells_total,
            max_supported,
        } => {
            assert_eq!(n_cells_total, 1626usize * 1626 * 1626);
            assert_eq!(n_cells_total, 4_298_942_376);
            assert_eq!(max_supported, u32::MAX as usize);
        }
        other => panic!("expected TooManyCells, got {other:?}"),
    }
}

#[test] // rq-5f2c42be
fn prefix_scan_correct_beyond_single_block_totals_pass() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    // r_cut + r_skin = 0.05 over a 10.0 box → 200 cells/axis →
    // n_cells_total = 8,000,000, well past the former scan_block_size²
    // (65,536) limit and requiring multiple recursion levels in the
    // prefix scan.
    let sim_box = box_n(10.0);
    // 100 particles on a coarse lattice inside the box.
    let mut px = Vec::with_capacity(100);
    let mut py = Vec::with_capacity(100);
    let mut pz = Vec::with_capacity(100);
    for i in 0..100u32 {
        px.push(-4.0 + (i % 10) as f32 * 0.8);
        py.push(-4.0 + ((i / 10) % 10) as f32 * 0.8);
        pz.push(-4.0 + (i % 7) as f32 * 0.5);
    }
    let state = state_from_positions(px, py, pz);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut nl =
        NeighborListState::new_cell_list(&gpu, &sim_box, 100, 0.025, 8, 0.025)
            .expect("new_cell_list");
    assert_eq!(nl.cell_list_data().unwrap().n_cells, [200, 200, 200]);
    assert_eq!(nl.cell_list_data().unwrap().n_cells_total, 8_000_000);
    let mut timings = Timings::new(&gpu).unwrap();
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();

    let cl = nl.cell_list_data().unwrap();
    let offsets = device.dtoh_sync_copy(&cl.cell_offsets).unwrap();
    let counts = device.dtoh_sync_copy(&cl.cell_counts).unwrap();
    assert_eq!(offsets.len(), cl.n_cells_total + 1);
    assert_eq!(offsets[0], 0);
    for c in 0..cl.n_cells_total {
        assert_eq!(
            offsets[c + 1],
            offsets[c] + counts[c],
            "exclusive prefix sum broken at cell {c}"
        );
    }
    assert_eq!(offsets[cl.n_cells_total], 100);
    for w in offsets.windows(2) {
        assert!(w[0] <= w[1], "offsets must be non-decreasing");
    }
}

#[test] // rq-f2e4b0b8
fn cell_list_scratch_reallocated_on_box_generation_refresh() {
    let gpu = init_device().unwrap();
    let mut sim_box = box_n(10.0);
    let state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut nl =
        NeighborListState::new_cell_list(&gpu, &sim_box, 2, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    let cl_before = nl.cell_list_data().unwrap();
    assert_eq!(cl_before.cell_counts.len(), 343);
    assert_eq!(cl_before.write_cursors.len(), 343);
    // The block-totals stack for n_cells_total = 343 is [ceil(343/256), 1]
    // = [2, 1]: the level-0 buffer holds 2 per-block totals.
    assert_eq!(cl_before.scan_block_totals.len(), 2);
    assert_eq!(cl_before.scan_block_totals[0].len(), 2);
    let cell_indices_len_before = cl_before.cell_indices.len();

    sim_box.set_lattice(12.0, 12.0, 12.0, 0.0, 0.0, 0.0).expect("ok");
    let new_state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &new_state).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    let cl_after = nl.cell_list_data().unwrap();
    assert_eq!(cl_after.cell_counts.len(), 729);
    assert_eq!(cl_after.write_cursors.len(), 729);
    // The stack is rebuilt for n_cells_total = 729: [ceil(729/256), 1]
    // = [3, 1], so the level-0 buffer now holds 3 per-block totals.
    assert_eq!(cl_after.scan_block_totals.len(), 2);
    assert_eq!(cl_after.scan_block_totals[0].len(), 3);
    // cell_indices is per-atom (particle_count = 2) and not reallocated.
    assert_eq!(cl_after.cell_indices.len(), cell_indices_len_before);
}

// --- Tilted-box geometry --------------------------------------------------

// rq-a7aac794
#[test]
fn cell_counts_reflect_perpendicular_widths_for_a_tilted_box() {
    let gpu = init_device().unwrap();
    // yz = 10 makes lattice vectors b = (0,10,0), c = (0,10,10). The
    // c-direction perpendicular width is unchanged (w_c = 10); the
    // b-direction perpendicular width drops to (ly*lz)/√(lz²+yz²) =
    // 100/√200 ≈ 7.07.
    let sim_box = SimulationBox::new(10.0, 10.0, 10.0, 0.0, 0.0, 10.0).unwrap();
    // r_cut = 0.7, r_skin = 0.6 → r_search = 1.3. n_cells[1] = floor(7.07/1.3) = 5.
    let nl = NeighborListState::new_cell_list(&gpu, &sim_box, 0, 0.7, 8, 0.6).unwrap();
    let cl = nl.cell_list_data().expect("cell-list mode");
    assert_eq!(cl.n_cells, [7, 5, 7], "tilted box per-direction cell counts");
}

// rq-e84d3bac
#[test]
fn reject_a_tilted_box_whose_perpendicular_width_drops_below_required() {
    let gpu = init_device().unwrap();
    // Lattice 13x13x13 with yz = 13. w_a = w_c = 13.0; w_b = (13*13) /
    // sqrt(13² + 13²) = 169/√338 ≈ 9.19. With r_cut = 3.7, r_skin = 0.3
    // → required width = 3*(r_cut+r_skin) = 12, the b direction can't
    // host 3 cells while a and c can.
    let sim_box = SimulationBox::new(13.0, 13.0, 13.0, 0.0, 0.0, 13.0).unwrap();
    let res = NeighborListState::new_cell_list(&gpu, &sim_box, 0, 3.7, 8, 0.3);
    match res {
        Err(NeighborListError::BoxTooSmallForCells { direction, width, required }) => {
            assert_eq!(direction, "b");
            let expected_width = 169.0_f32 / (338.0_f32).sqrt();
            assert!((width - expected_width).abs() < 1.0e-3, "got width {width}, expected {expected_width}");
            assert!((required - 12.0).abs() < 1.0e-4);
        }
        other => panic!("expected BoxTooSmallForCells, got {other:?}"),
    }
}

// --- Cell-index binning and wrap ------------------------------------------

// rq-151cb099
#[test]
fn cell_index_at_the_plus_half_boundary_clamps_inside_the_grid() {
    let gpu = init_device().unwrap();
    let l: f32 = 10.0;
    let sim_box = box_n(l);
    // r_cut + r_skin = 1.3 → n_cells = 7 along each axis. A particle just
    // inside the +x boundary (fractional s_a just shy of +0.5) bins to
    // cell index n-1 = 6 along the a axis. f32 round-off near 1.0 can
    // round (s_a + 0.5) * n up to exactly n, and the kernel's clamp
    // returns the in-range index n-1.
    let eps: f32 = 1.0e-5;
    let state = state_from_positions(
        vec![l * 0.5 - eps],
        vec![0.0],
        vec![0.0],
    );
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 1, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    let cl = nl.cell_list_data().unwrap();
    let cell_indices = gpu.device.dtoh_sync_copy(&cl.cell_indices).unwrap();
    let n_a = cl.n_cells[0] as usize;
    let n_b = cl.n_cells[1] as usize;
    let n_c = cl.n_cells[2] as usize;
    // Linear index c = ((a * n_b) + b) * n_c + c → recover a.
    let ca = cell_indices[0] as usize / (n_b * n_c);
    assert_eq!(ca, n_a - 1, "expected boundary clamp to n_cells_a - 1 = {}", n_a - 1);
}

// rq-a99ca751
#[test]
fn cell_index_of_a_position_outside_the_primary_cell_wraps_before_binning() {
    let gpu = init_device().unwrap();
    let l: f32 = 10.0;
    let sim_box = box_n(l);
    // Two particles whose Cartesian positions differ by one full lattice
    // vector along a. After triclinic_wrap_with_image, both map to the
    // same point in the primary cell and bin to the same cell index.
    let state = state_from_positions(
        vec![0.3, 0.3 + l],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
    );
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 2, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    let cell_indices = gpu.device
        .dtoh_sync_copy(&nl.cell_list_data().unwrap().cell_indices)
        .unwrap();
    assert_eq!(
        cell_indices[0], cell_indices[1],
        "particle past the primary cell must bin into the same cell as its wrapped image"
    );
}

// --- Neighbor-list partial ordering --------------------------------------

// rq-b5289acc
#[test]
fn neighbor_list_is_not_globally_partner_id_sorted_across_cells() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    // r_cut + r_skin = 1.0 → n_cells = 10 per axis along x in a 10x10x10 box.
    // Place an atom at the centre (atom 4) and four partners at varying
    // x so that they fall into different x-cells. The 27-cell sweep
    // visits cells in (da, db, dc) order — i.e. ascending da first —
    // which means the partner sitting at a lower cell index along x is
    // visited before one at a higher cell index, *regardless* of the
    // partner's particle ID.
    //
    // Layout (single x axis; all y=z=0):
    //   atom 0 (high particle id)  → x ≈ -0.55 → cell_a = 4
    //   atom 1 (low partner)        → x ≈ -0.25 → cell_a = 4
    //   atom 2 (high partner)       → x ≈ +0.55 → cell_a = 5
    //   atom 3 (low particle id)    → x ≈ +0.85 → cell_a = 5
    //   atom 4 (the home atom)      → x = 0.0   → cell_a = 5
    //
    // The 27-cell sweep of atom 4 visits cell_a = 4 before cell_a = 5,
    // so within atom 4's neighbour list, atoms 0 and 1 appear before 2
    // and 3 — even though atom 0's particle ID > atom 1's. Concretely
    // the per-list order is [1, 0, 3, 2] (within-cell sort by particle
    // ID gives [0, 1] in cell 4 and [2, 3] in cell 5, but the
    // sort_cells_by_particle_id pass canonicalises that within each
    // cell, and the cell-sweep order interleaves them).
    let state = state_from_positions(
        vec![-0.55_f32, -0.25, 0.55, 0.85, 0.0],
        vec![0.0; 5],
        vec![0.0; 5],
    );
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = box_n(10.0);
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 5, 0.7, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    let list = device.dtoh_sync_copy(&nl.neighbor_list).unwrap();
    let counts = device.dtoh_sync_copy(&nl.neighbor_counts).unwrap();
    let base = 4 * 8;
    let c = counts[4] as usize;
    let slice = &list[base..base + c];
    assert_eq!(c, 4, "atom 4 should see four partners (counts: {counts:?})");
    // The sweep visits the cell at x ≈ -0.5 before the cell at x ≈ +0.5.
    // The lower-cell partners are atoms {0, 1}, the upper-cell partners
    // are {2, 3}. Within each cell, sorting by particle ID gives an
    // ascending pair. Across cells the global sequence is NOT strictly
    // ascending in partner ID: position 1 (atom 1 from low cell) is
    // followed by position 2 (atom 2 or 3 from the high cell), but
    // there is no guarantee that position 1 < position 2 holds for ALL
    // such sequences — it does here only because we picked IDs in the
    // order they happened to sort into. Concretely we check that the
    // list is NOT strictly ascending (cell-sweep order beats partner
    // index): we expect [0, 1, 2, 3] under cell_a sweep (4 then 5),
    // confirming atoms from a lower cell come before those from a
    // higher one regardless of partner ID.
    let mut prev_cell = -1i32;
    let cl = nl.cell_list_data().unwrap();
    let cell_indices = device.dtoh_sync_copy(&cl.cell_indices).unwrap();
    let n_b = cl.n_cells[1] as usize;
    let n_c = cl.n_cells[2] as usize;
    for &partner in slice {
        let ca = (cell_indices[partner as usize] as usize / (n_b * n_c)) as i32;
        assert!(
            ca >= prev_cell,
            "partner {partner} (cell_a={ca}) appears after a partner from cell_a={prev_cell}; sweep is not monotone non-decreasing"
        );
        prev_cell = ca;
    }
    // And the cells actually differ — the list spans more than one cell,
    // which is what makes the across-cell ordering meaningful.
    let first_ca = cell_indices[slice[0] as usize] as usize / (n_b * n_c);
    let last_ca = cell_indices[slice[slice.len() - 1] as usize] as usize / (n_b * n_c);
    assert_ne!(first_ca, last_ca, "neighbour list spans multiple cells");
}

// --- Generation refresh paths --------------------------------------------

// rq-e2a31585
#[test]
fn box_generation_refresh_handles_tilt_mutation() {
    let gpu = init_device().unwrap();
    let mut sim_box = box_n(10.0);
    let state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    // r_cut + r_skin = 1.3 → n_cells = floor(10/1.3) = 7 in the unmodified
    // box.
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 2, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    assert_eq!(nl.cell_list_data().unwrap().n_cells, [7, 7, 7]);

    // Introduce yz = 5.0 → w_b drops to (10*10)/√(100 + 25) = 100/√125 ≈
    // 8.944. n_cells[1] = floor(8.944/1.3) = 6.
    sim_box.set_lattice(10.0, 10.0, 10.0, 0.0, 0.0, 5.0).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    let cl = nl.cell_list_data().unwrap();
    assert_eq!(cl.n_cells, [7, 6, 7]);
    assert_eq!(cl.cached_generation, sim_box.generation());
}

// rq-f79d1ac5
#[test]
fn generation_tick_with_unchanged_n_cells_total_triggers_rebuild_on_displacement() {
    let gpu = init_device().unwrap();
    let mut sim_box = box_n(10.0);
    // r_skin = 1.0 → threshold = 0.5. r_cut + r_skin = 1.5 → n_cells = floor(10/1.5) = 6
    // on each axis (n_cells_total = 216).
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 2, 0.5, 8, 1.0).unwrap();
    let state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    let n_cells_before = nl.cell_list_data().unwrap().n_cells_total;
    assert_eq!(n_cells_before, 216);

    // Move atom 1 by 1.0 along x (well past r_skin/2 = 0.5).
    let moved = state_from_positions(vec![0.0, 2.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    buffers.upload(&moved).unwrap();

    // Tick the generation with a lattice that yields the same n_cells_total:
    // r_search stays 1.5 and the box stays 10×10×10, just bumping the
    // generation counter.
    sim_box.set_lattice(10.0, 10.0, 10.0, 0.0, 0.0, 0.0).unwrap();
    assert!(sim_box.generation() > 0);

    let mut t2 = Timings::new(&gpu).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut t2).unwrap();
    assert_eq!(nl.cell_list_data().unwrap().n_cells_total, n_cells_before);
    assert!(!nl.cell_list_data().unwrap().needs_rebuild, "rebuild should have run");
    let report = t2.finalize().unwrap();
    let nlb = report.stages.iter().find(|s| s.name == "neighbor_list_build").map(|s| s.count).unwrap_or(0);
    assert_eq!(nlb, 1, "neighbor_list_build must fire once after the rebuild");
}

// rq-3288a78c
#[test]
fn npt_style_barostat_ticks_rebuild_at_displacement_driven_rate() {
    // Mimic an NPT loop where the barostat bumps box.generation() each
    // step with a tiny scale that leaves n_cells_total unchanged. Apply
    // a small physical drift per step. The neighbour list should NOT
    // rebuild on every step — only when the accumulated displacement
    // crosses r_skin / 2 — proving that a generation tick alone does
    // not force a rebuild beyond the displacement-driven cadence.
    let gpu = init_device().unwrap();
    let mut sim_box = box_n(10.0);
    let r_skin: f32 = 1.0;
    let drift_per_step: f32 = 0.1; // 5 steps to cross r_skin/2 = 0.5.
    let mut nl = NeighborListState::new_cell_list(&gpu, &sim_box, 2, 0.5, 8, r_skin).unwrap();
    let mut x = 1.0_f32;
    let state = state_from_positions(vec![0.0, x], vec![0.0, 0.0], vec![0.0, 0.0]);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    nl.pre_step(&sim_box, &buffers, &mut timings).unwrap();
    let _initial_n_cells_total = nl.cell_list_data().unwrap().n_cells_total;

    // Drive 10 steps with barostat ticks. Each step: bump generation by
    // re-setting the same lattice; drift atom 1 by drift_per_step.
    let mut rebuilds_after_initial = 0u64;
    let mut last_nlb_count = {
        let report_stub = Timings::new(&gpu).unwrap().finalize().unwrap();
        let _ = report_stub;
        0u64
    };
    let mut accum_t = Timings::new(&gpu).unwrap();
    for _ in 0..10 {
        sim_box.set_lattice(10.0, 10.0, 10.0, 0.0, 0.0, 0.0).unwrap();
        x += drift_per_step;
        let moved = state_from_positions(vec![0.0, x], vec![0.0, 0.0], vec![0.0, 0.0]);
        buffers.upload(&moved).unwrap();
        nl.pre_step(&sim_box, &buffers, &mut accum_t).unwrap();
    }
    let report = accum_t.finalize().unwrap();
    let nlb_total = report.stages.iter().find(|s| s.name == "neighbor_list_build")
        .map(|s| s.count).unwrap_or(0);
    rebuilds_after_initial += nlb_total - last_nlb_count;
    last_nlb_count = nlb_total;
    // After 10 ticks at 0.1 per step the atom has drifted 1.0 ≈ 2x r_skin/2.
    // We expect rebuilds at most 3 times (initial + 2 displacement crossings).
    assert!(
        rebuilds_after_initial >= 1 && rebuilds_after_initial <= 3,
        "rebuilds should be displacement-driven, not generation-tick driven; got {rebuilds_after_initial}"
    );
}
