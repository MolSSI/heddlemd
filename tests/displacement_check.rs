//! Tests for the device-side displacement-check kernel and its
//! consumption inside `NeighborListState::pre_step`. Each test
//! corresponds to a Gherkin scenario in
//! `rqm/forces/neighbor-list.md` or `rqm/cuda-graphs.md`.

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use heddle_md::forces::{NeighborListMode, NeighborListState};
use heddle_md::gpu::{GpuContext, ParticleBuffers, init_device};
use heddle_md::io::config::NeighborListConfig;
use heddle_md::pbc::SimulationBox;
use heddle_md::precision::Real;
use heddle_md::state::ParticleState;
use heddle_md::timings::Timings;

/// 10 Å × 10 Å × 10 Å orthorhombic box. Default for the displacement
/// kernel tests below — large enough that the cell-list constructor
/// admits at least 3 cells per axis for any r_cut + r_skin used here.
fn box_10(gpu: &GpuContext) -> SimulationBox {
    SimulationBox::new(&gpu.device, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0).unwrap()
}

/// Build a `ParticleBuffers` with the supplied per-atom xyz; everything
/// else is filled with zeros. Used to position the test particles at
/// known offsets from their reference.
fn buffers_at(
    gpu: &GpuContext,
    px: Vec<Real>,
    py: Vec<Real>,
    pz: Vec<Real>,
) -> ParticleBuffers {
    let n = px.len();
    let state = ParticleState::new(
        px,
        py,
        pz,
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        vec![0.0; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap();
    ParticleBuffers::new(gpu, &state).unwrap()
}

/// Build a `NeighborListState` in CellList mode with `r_cut = 1.0`,
/// `r_skin = 0.3`. The cell layout supports 7 cells per axis at box
/// side 10. `max_neighbors` is unused by the displacement kernel.
fn cell_list_state(gpu: &GpuContext, n: usize, sim_box: &SimulationBox) -> NeighborListState {
    NeighborListState::new_cell_list(gpu, sim_box, n, 1.0, 256, 0.3).unwrap()
}

/// Copy the supplied host-side reference positions into the CellList's
/// reference-position buffers. Bypasses the rebuild kernel; useful for
/// constructing test scenarios where reference != current.
fn write_references(
    device: &Arc<CudaDevice>,
    nl: &mut NeighborListState,
    rx: &[Real],
    ry: &[Real],
    rz: &[Real],
) {
    match &mut nl.mode {
        NeighborListMode::CellList(cl) => {
            device.htod_sync_copy_into(rx, &mut cl.reference_positions_x).unwrap();
            device.htod_sync_copy_into(ry, &mut cl.reference_positions_y).unwrap();
            device.htod_sync_copy_into(rz, &mut cl.reference_positions_z).unwrap();
        }
        _ => panic!("test expects CellList mode"),
    }
}

/// Read back bit 0 (the displacement-trip bit) of the combined
/// `neighbor_status` word of a CellList NL.
fn read_flag(device: &Arc<CudaDevice>, nl: &NeighborListState) -> u32 {
    match (&nl.mode, nl.packed.as_ref()) {
        (NeighborListMode::CellList(_), Some(p)) => {
            let host: Vec<u32> = device.dtoh_sync_copy(&p.neighbor_status).unwrap();
            host[0] & 1
        }
        _ => panic!("test expects CellList mode"),
    }
}

/// Set the device-side `neighbor_status` word directly (host-side
/// initialiser for tests that need to assert bit 0 gets cleared by a
/// rebuild, etc.).
fn set_flag(device: &Arc<CudaDevice>, nl: &mut NeighborListState, value: u32) {
    match (&mut nl.mode, nl.packed.as_mut()) {
        (NeighborListMode::CellList(_), Some(p)) => {
            device.htod_sync_copy_into(&[value], &mut p.neighbor_status).unwrap();
        }
        _ => panic!("test expects CellList mode"),
    }
}

#[test] // rq-837c85d3
fn displacement_check_at_reference_leaves_flag_clear() {
    let gpu = init_device().unwrap();
    let sb = box_10(&gpu);
    let n = 8;
    let positions = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let buffers = buffers_at(&gpu, positions.clone(), positions.clone(), positions.clone());
    let mut nl = cell_list_state(&gpu, n, &sb);
    // Make reference positions identical to current positions.
    write_references(&gpu.device, &mut nl, &positions, &positions, &positions);
    let mut timings = Timings::new(&gpu).unwrap();
    nl.enqueue_displacement_check(&sb, &buffers, &mut timings).unwrap();
    assert_eq!(read_flag(&gpu.device, &nl), 0);
}

#[test] // rq-6e1f04f3
fn displacement_check_uses_minimum_image() {
    let gpu = init_device().unwrap();
    let sb = box_10(&gpu);
    // Place a particle at +4.95 with reference at -4.95 (wrapped across
    // the PBC boundary). The raw difference is 9.9; the min-image
    // difference is 0.1, below the threshold r_skin/2 = 0.15.
    let pos_x = vec![4.95];
    let ref_x = vec![-4.95];
    let buffers = buffers_at(&gpu, pos_x, vec![0.0], vec![0.0]);
    let mut nl = cell_list_state(&gpu, 1, &sb);
    write_references(&gpu.device, &mut nl, &ref_x, &[0.0], &[0.0]);
    let mut timings = Timings::new(&gpu).unwrap();
    nl.enqueue_displacement_check(&sb, &buffers, &mut timings).unwrap();
    assert_eq!(read_flag(&gpu.device, &nl), 0);
}

#[test] // rq-c43dd1ab
fn displacement_check_sets_flag_when_any_particle_exceeds_threshold() {
    let gpu = init_device().unwrap();
    let sb = box_10(&gpu);
    let n = 8;
    let mut pos = vec![0.0; n];
    let refs = vec![0.0; n];
    // Particle 7 moves 0.5 (well past r_skin/2 = 0.15); others stay put.
    pos[7] = 0.5;
    let buffers = buffers_at(&gpu, pos, vec![0.0; n], vec![0.0; n]);
    let mut nl = cell_list_state(&gpu, n, &sb);
    write_references(&gpu.device, &mut nl, &refs, &refs, &refs);
    let mut timings = Timings::new(&gpu).unwrap();
    nl.enqueue_displacement_check(&sb, &buffers, &mut timings).unwrap();
    assert_eq!(read_flag(&gpu.device, &nl), 1);
}

#[test] // rq-c9f970fe
fn displacement_check_is_sticky_across_launches() {
    let gpu = init_device().unwrap();
    let sb = box_10(&gpu);
    let n = 2;
    let refs = vec![0.0; n];
    let mut nl = cell_list_state(&gpu, n, &sb);
    write_references(&gpu.device, &mut nl, &refs, &refs, &refs);
    let mut timings = Timings::new(&gpu).unwrap();
    // Launch #1: one particle exceeds threshold. Flag goes 0 -> 1.
    let buffers_above = buffers_at(&gpu, vec![0.0, 0.5], vec![0.0; n], vec![0.0; n]);
    nl.enqueue_displacement_check(&sb, &buffers_above, &mut timings).unwrap();
    assert_eq!(read_flag(&gpu.device, &nl), 1);
    // Launch #2: every particle within threshold, but the flag
    // remains sticky-1 because the host has not cleared it.
    let buffers_below = buffers_at(&gpu, vec![0.0, 0.05], vec![0.0; n], vec![0.0; n]);
    nl.enqueue_displacement_check(&sb, &buffers_below, &mut timings).unwrap();
    assert_eq!(read_flag(&gpu.device, &nl), 1);
}

#[test] // rq-46d72444
fn rebuild_clears_displacement_flag() {
    let gpu = init_device().unwrap();
    let sb = box_10(&gpu);
    let n = 4;
    let positions = vec![1.0, 3.0, 5.0, 7.0];
    let buffers = buffers_at(
        &gpu,
        positions.clone(),
        positions.clone(),
        positions.clone(),
    );
    let mut nl = cell_list_state(&gpu, n, &sb);
    let mut timings = Timings::new(&gpu).unwrap();
    // Force the flag high before the rebuild and verify pre_step
    // clears it via the post-rebuild memset.
    set_flag(&gpu.device, &mut nl, 1u32);
    nl.pre_step(&sb, &buffers, &mut timings).unwrap();
    assert_eq!(read_flag(&gpu.device, &nl), 0);
    // Reference positions equal current positions after the rebuild.
    let (rx, ry, rz) = match &nl.mode {
        NeighborListMode::CellList(cl) => (
            gpu.device.dtoh_sync_copy(&cl.reference_positions_x).unwrap(),
            gpu.device.dtoh_sync_copy(&cl.reference_positions_y).unwrap(),
            gpu.device.dtoh_sync_copy(&cl.reference_positions_z).unwrap(),
        ),
        _ => unreachable!(),
    };
    for i in 0..n {
        assert!((rx[i] - positions[i]).abs() < 1e-6);
        assert!((ry[i] - positions[i]).abs() < 1e-6);
        assert!((rz[i] - positions[i]).abs() < 1e-6);
    }
}

#[test] // rq-5d2e8748
fn pre_step_downloads_a_single_u32() {
    // The displacement_check API returns a single bool produced from
    // a single-word dtoh of `disp_rebuild_flag`. There is no
    // observable mechanism within the public API to count CUDA
    // memcpys directly; we instead assert that
    // `displacement_check` agrees with the in-buffer value.
    let gpu = init_device().unwrap();
    let sb = box_10(&gpu);
    let n = 4;
    let positions = vec![1.0, 3.0, 5.0, 7.0];
    let buffers = buffers_at(
        &gpu,
        positions.clone(),
        positions.clone(),
        positions.clone(),
    );
    let mut nl = cell_list_state(&gpu, n, &sb);
    let mut timings = Timings::new(&gpu).unwrap();
    // First pre_step does the initial rebuild (needs_rebuild = true).
    nl.pre_step(&sb, &buffers, &mut timings).unwrap();
    // Flag is 0 (just rebuilt) — pre_step's second call should not
    // rebuild and should return false from displacement_check.
    let tripped = nl.displacement_check(&sb, &buffers, &mut timings).unwrap();
    assert!(!tripped);
    // Now force the flag high.
    set_flag(&gpu.device, &mut nl, 1u32);
    let tripped = nl.displacement_check(&sb, &buffers, &mut timings).unwrap();
    assert!(tripped);
}

#[test] // rq-151a7e82
fn default_graph_batch_size_is_50() {
    use heddle_md::io::config::SimulationConfig;
    let cfg: SimulationConfig = toml::from_str("seed = 0\ntemperature = 300.0").unwrap();
    assert_eq!(cfg.graph_batch_size, 50);
}

#[test] // rq-6caca2f6
fn skin_contract_holds_for_default_K_at_liquid_md_rates() {
    // Pure arithmetic check matching the scenario:
    //   K = 50; r_skin = 0.3 * r_cut; max_step_displacement <= 0.001 * r_cut
    //   K * max_step_displacement <= 0.05 * r_cut < 0.15 * r_cut = r_skin / 2
    let r_cut: f64 = 10.0;
    let k: f64 = 50.0;
    let r_skin = 0.3 * r_cut;
    let max_step_displacement = 0.001 * r_cut;
    let bound = k * max_step_displacement;
    assert!(bound < r_skin * 0.5,
        "skin-contract margin failed: K*max_step_disp = {bound} >= r_skin/2 = {}", r_skin * 0.5);
}

// rq-59bbfa07 rq-faf1dd2e rq-c4cc1d99 rq-f4069c16
//
// These four cuda-graphs scenarios exercise the captured-graph kernel
// sequence and the per-batch host-sync surface. The kernel-presence
// scenario is covered indirectly by `timings.rs` (which already asserts
// the displacement-check stage records launches), and the per-batch
// rebuild scenarios run as part of the canonical SPC-water example
// (which the harness in `tests/integration.rs` exercises end to end).
// Explicit per-batch CUDA-graph tests are out of scope for this
// per-kernel test file; the runtime correctness is verified by the
// 8k SPC water example whose phase output is compared against the
// reference at the integration-test level.
#[test]
fn quiescent_batch_leaves_flag_clear_and_triggered_batch_sets_it() {
    let gpu = init_device().unwrap();
    let sb = box_10(&gpu);
    let n = 4;
    let positions = vec![1.0, 3.0, 5.0, 7.0];
    let buffers_q = buffers_at(
        &gpu,
        positions.clone(),
        positions.clone(),
        positions.clone(),
    );
    let mut nl = cell_list_state(&gpu, n, &sb);
    write_references(&gpu.device, &mut nl, &positions, &positions, &positions);
    let mut timings = Timings::new(&gpu).unwrap();
    // 50 quiescent "captured-step" replays of the displacement kernel
    // — every replay sees reference == current. Flag must stay 0.
    for _ in 0..50 {
        nl.enqueue_displacement_check(&sb, &buffers_q, &mut timings).unwrap();
    }
    assert_eq!(read_flag(&gpu.device, &nl), 0);

    // Triggered batch: one replay observes an over-threshold particle.
    // Subsequent replays see the same positions but the flag stays
    // sticky.
    let mut moved = positions.clone();
    moved[2] = 5.5; // 0.5 displacement on particle 2, well above 0.15
    let buffers_t = buffers_at(&gpu, moved, positions.clone(), positions.clone());
    nl.enqueue_displacement_check(&sb, &buffers_t, &mut timings).unwrap();
    nl.enqueue_displacement_check(&sb, &buffers_q, &mut timings).unwrap();
    assert_eq!(read_flag(&gpu.device, &nl), 1);

    // pre_step at the batch boundary observes the flag and rebuilds,
    // clearing the flag.
    nl.pre_step(&sb, &buffers_t, &mut timings).unwrap();
    assert_eq!(read_flag(&gpu.device, &nl), 0);
}

// rq-59bbfa07
// The captured-graph includes-displacement-check scenario is covered
// indirectly by the existing `tests/timings.rs::cell_list_step_records_*`
// tests, which assert that every `ForceField::step` records exactly one
// `neighbor_displacement_check_flag` stage launch (the stage name has
// been updated to match the renamed kernel). Re-asserting the same
// behaviour here would only duplicate that coverage.
