// rq-c4645fa6
use dynamics::forces::{NeighborListError, NeighborListState};
use dynamics::gpu::{ParticleBuffers, init_device};
use dynamics::pbc::SimulationBox;
use dynamics::state::ParticleState;
use dynamics::timings::Timings;

fn box_n(l: f32) -> SimulationBox {
    SimulationBox::new_orthorhombic(l, l, l).unwrap()
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
        vec![0u32; n],
        None,
    )
    .unwrap()
}

// rq-c0cfc5d6
#[test]
fn cell_counts_floor_of_l_over_search_radius() {
    let device = init_device().unwrap();
    let nl = NeighborListState::new_cell_list(device, box_n(10.0), 0, 1.0, 8, 0.3).unwrap();
    let cl = nl.cell_list_data().expect("cell-list mode");
    assert_eq!(cl.n_cells, [7, 7, 7]);
    let expected = 10.0_f32 / 7.0;
    for a in 0..3 {
        assert!(
            (cl.cell_size[a] - expected).abs() < 1.0e-6,
            "cell_size[{}] = {}, expected {}",
            a,
            cl.cell_size[a],
            expected
        );
    }
}

// rq-1b9c474c
#[test]
fn reject_box_admitting_fewer_than_three_cells() {
    let device = init_device().unwrap();
    let result = NeighborListState::new_cell_list(device, box_n(10.0), 0, 1.0, 8, 3.0);
    match result {
        Err(NeighborListError::BoxTooSmallForCells {
            axis,
            length,
            required,
        }) => {
            assert_eq!(axis, "x");
            assert!((length - 10.0).abs() < 1.0e-6);
            assert!((required - 12.0).abs() < 1.0e-6);
        }
        other => panic!("expected BoxTooSmallForCells, got {other:?}"),
    }
}

// rq-4bc8028f
#[test]
fn particle_count_zero_builds_and_runs() {
    let device = init_device().unwrap();
    let mut nl = NeighborListState::new_cell_list(device.clone(), box_n(10.0), 0, 1.0, 8, 0.3).unwrap();
    let state = state_from_positions(vec![], vec![], vec![]);
    let buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings = Timings::new(device).unwrap();
    let max_disp = nl.displacement_check(&buffers, &mut timings).unwrap();
    assert_eq!(max_disp, 0.0);
    nl.rebuild(&buffers, &mut timings).unwrap();
}

// rq-52f547fd
#[test]
fn single_particle_yields_empty_neighbor_list() {
    let device = init_device().unwrap();
    let mut nl = NeighborListState::new_cell_list(device.clone(), box_n(10.0), 1, 1.0, 8, 0.3).unwrap();
    let state = state_from_positions(vec![0.0], vec![0.0], vec![0.0]);
    let buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    nl.rebuild(&buffers, &mut timings).unwrap();
    let counts = device.dtoh_sync_copy(&nl.neighbor_counts).unwrap();
    assert_eq!(counts[0], 0);
}

// rq-ea0ee5ef rq-e75b24e7 rq-2bc559ec
#[test]
fn neighbor_list_contains_all_within_search_radius_and_is_sorted() {
    let device = init_device().unwrap();
    // 4 particles along x at 0.0, 0.5, 1.0, 2.0
    let state = state_from_positions(
        vec![0.0, 0.5, 1.0, 2.0],
        vec![0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0],
    );
    let buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let r_cut = 1.0_f32;
    let r_skin = 0.3_f32;
    let max_neighbors = 8u32;
    let mut nl =
        NeighborListState::new_cell_list(device.clone(), box_n(10.0), 4, r_cut, max_neighbors, r_skin)
            .unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    nl.rebuild(&buffers, &mut timings).unwrap();
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

// rq-25faef11
#[test]
fn neighbor_list_uses_minimum_image() {
    let device = init_device().unwrap();
    // Two particles separated by 0.2 across the periodic boundary along x.
    // Box of length 10.0 → -5.0..+5.0 primary cell.
    let state = state_from_positions(
        vec![-4.9, 4.9],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
    );
    let buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut nl = NeighborListState::new_cell_list(device.clone(), box_n(10.0), 2, 0.7, 8, 0.3).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    nl.rebuild(&buffers, &mut timings).unwrap();
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
    let device = init_device().unwrap();
    // 6 particles tightly clustered within r_cut+r_skin; max_neighbors=2.
    // Each atom has 5 partners within range, exceeding the cap.
    let positions: Vec<[f32; 3]> = (0..6).map(|i| [i as f32 * 0.1, 0.0, 0.0]).collect();
    let px: Vec<f32> = positions.iter().map(|p| p[0]).collect();
    let py: Vec<f32> = positions.iter().map(|p| p[1]).collect();
    let pz: Vec<f32> = positions.iter().map(|p| p[2]).collect();
    let state = state_from_positions(px, py, pz);
    let buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut nl = NeighborListState::new_cell_list(device.clone(), box_n(10.0), 6, 1.0, 2, 0.3).unwrap();
    let mut timings = Timings::new(device).unwrap();
    let err = nl.rebuild(&buffers, &mut timings).unwrap_err();
    match err {
        NeighborListError::NeighborListOverflow { max } => assert_eq!(max, 2),
        other => panic!("expected NeighborListOverflow, got {other:?}"),
    }
}

// rq-6bf3709c
#[test]
fn two_rebuilds_with_identical_positions_agree() {
    let device = init_device().unwrap();
    let state = state_from_positions(
        vec![0.0, 0.4, 0.8, 1.2, 1.6, 2.0],
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        vec![-0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
    );
    let buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    let mut nl_a =
        NeighborListState::new_cell_list(device.clone(), box_n(10.0), 6, 1.0, 16, 0.3).unwrap();
    let mut nl_b =
        NeighborListState::new_cell_list(device.clone(), box_n(10.0), 6, 1.0, 16, 0.3).unwrap();
    nl_a.rebuild(&buffers, &mut timings).unwrap();
    nl_b.rebuild(&buffers, &mut timings).unwrap();
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
    let device = init_device().unwrap();
    let state = state_from_positions(
        vec![0.0, 0.5, 1.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    );
    let buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut nl = NeighborListState::new_cell_list(device.clone(), box_n(10.0), 3, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(device).unwrap();
    nl.rebuild(&buffers, &mut timings).unwrap();
    let max_disp = nl.displacement_check(&buffers, &mut timings).unwrap();
    assert!(max_disp.abs() < 1.0e-5);
}

// rq-f94ee5cd
#[test]
fn displacement_check_returns_max_across_particles() {
    let device = init_device().unwrap();
    let state = state_from_positions(
        vec![0.0, 1.0, 2.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    );
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut nl = NeighborListState::new_cell_list(device.clone(), box_n(10.0), 3, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    nl.rebuild(&buffers, &mut timings).unwrap();
    // Move atom 1 by 0.5 along x.
    let new = state_from_positions(
        vec![0.0, 1.5, 2.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    );
    buffers.upload(&new).unwrap();
    let max_disp = nl.displacement_check(&buffers, &mut timings).unwrap();
    assert!((max_disp - 0.5).abs() < 1.0e-4, "max_disp = {max_disp}");
}

// rq-35981c27
#[test]
fn first_pre_step_unconditionally_rebuilds() {
    let device = init_device().unwrap();
    let state = state_from_positions(
        vec![0.0, 1.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
    );
    let buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut nl = NeighborListState::new_cell_list(device.clone(), box_n(10.0), 2, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(device).unwrap();
    assert!(nl.cell_list_data().unwrap().needs_rebuild);
    nl.pre_step(&buffers, &mut timings).unwrap();
    assert!(!nl.cell_list_data().unwrap().needs_rebuild);
}

// rq-90524f5d
#[test]
fn sub_skin_movement_does_not_trigger_rebuild() {
    let device = init_device().unwrap();
    let state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut nl = NeighborListState::new_cell_list(device.clone(), box_n(10.0), 2, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    nl.pre_step(&buffers, &mut timings).unwrap(); // Initial rebuild.

    // Move every particle by less than r_skin/2 = 0.15.
    let moved =
        state_from_positions(vec![0.05, 1.10], vec![0.0, 0.0], vec![0.0, 0.0]);
    buffers.upload(&moved).unwrap();
    nl.pre_step(&buffers, &mut timings).unwrap();
    assert!(!nl.cell_list_data().unwrap().needs_rebuild);
}

// rq-9f63a183
#[test]
fn over_skin_movement_triggers_rebuild() {
    let device = init_device().unwrap();
    let state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut nl = NeighborListState::new_cell_list(device.clone(), box_n(10.0), 2, 1.0, 8, 0.3).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    nl.pre_step(&buffers, &mut timings).unwrap(); // Initial rebuild.

    // Move atom 1 by 0.5 (more than r_skin/2 = 0.15).
    let moved =
        state_from_positions(vec![0.0, 1.5], vec![0.0, 0.0], vec![0.0, 0.0]);
    buffers.upload(&moved).unwrap();
    nl.pre_step(&buffers, &mut timings).unwrap();
    // After pre_step, the rebuild has happened so needs_rebuild is false.
    assert!(!nl.cell_list_data().unwrap().needs_rebuild);
    // The reference positions now equal the current positions, so a fresh
    // displacement_check returns zero.
    let max_disp = nl.displacement_check(&buffers, &mut timings).unwrap();
    assert!(max_disp.abs() < 1.0e-4);
}

// --- Trivial mode ---

#[test] // rq-789fcec9
fn trivial_mode_contents() {
    let device = init_device().unwrap();
    let nl = NeighborListState::new_trivial(device.clone(), box_n(10.0), 3).unwrap();
    let counts = device.dtoh_sync_copy(&nl.neighbor_counts).unwrap();
    let list = device.dtoh_sync_copy(&nl.neighbor_list).unwrap();
    assert_eq!(counts, vec![3u32, 3, 3]);
    assert_eq!(list, vec![0u32, 1, 2, 0, 1, 2, 0, 1, 2]);
}

#[test] // rq-bb3773aa
fn trivial_mode_pre_step_does_no_work() {
    let device = init_device().unwrap();
    let state = state_from_positions(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0]);
    let buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut nl = NeighborListState::new_trivial(device.clone(), box_n(10.0), 2).unwrap();
    let mut timings = Timings::new(device).unwrap();
    nl.pre_step(&buffers, &mut timings).unwrap();
    nl.pre_step(&buffers, &mut timings).unwrap();
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
    let device = init_device().unwrap();
    let nl = NeighborListState::new_trivial(device, box_n(10.0), 4).unwrap();
    assert!(matches!(nl.mode, dynamics::forces::NeighborListMode::Trivial));
    assert!(nl.cell_list_data().is_none());
}
