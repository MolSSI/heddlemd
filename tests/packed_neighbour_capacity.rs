//! Tests for O(N) packed-neighbour buffer sizing, the all-pairs growth
//! ceiling, and `pre_step` reallocation reporting. Each test corresponds
//! to a Gherkin scenario in `rqm/forces/packed-neighbour-pair-force.md`
//! or `rqm/forces/neighbor-list.md`.

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use heddle_md::forces::{
    NeighborListMode, NeighborListState, all_pairs_tile_capacity,
    default_interacting_tiles_capacity,
};
use heddle_md::gpu::{GpuContext, ParticleBuffers, init_device};
use heddle_md::pbc::SimulationBox;
use heddle_md::precision::Real;
use heddle_md::state::ParticleState;
use heddle_md::timings::Timings;

// --- Pure capacity-function tests (no GPU required) ---

// rq-36026b97
#[test]
fn seed_capacity_is_far_below_all_pairs_bound() {
    // 196,608 atoms (the spc-water-65536 case) => 6144 atom-blocks.
    let n_blocks = 6144u32;
    let seed = default_interacting_tiles_capacity(n_blocks);
    let all_pairs = all_pairs_tile_capacity(n_blocks);
    assert_eq!(all_pairs, n_blocks * n_blocks);
    // Seed is O(N): a small multiple of n_blocks, orders of magnitude
    // below the O(n_blocks^2) all-pairs bound that used to be allocated.
    assert!(seed <= 256 * n_blocks);
    assert!((seed as u64) * 16 < all_pairs as u64);
}

// rq-8d7e376d
#[test]
fn seed_capacity_scales_linearly_with_n() {
    let n_blocks_a = 768u32; // 24,576 atoms
    let n_blocks_b = 768u32 * 8; // 8x the atoms
    let seed_a = default_interacting_tiles_capacity(n_blocks_a) as u64;
    let seed_b = default_interacting_tiles_capacity(n_blocks_b) as u64;
    // Linear: 8x the blocks => exactly 8x the seed (both below ceiling).
    assert_eq!(seed_b, 8 * seed_a);
    // And not quadratic: the seed is nowhere near n_blocks^2.
    assert!(seed_a < (n_blocks_a as u64) * (n_blocks_a as u64));
}

// rq-25f8dd1d
#[test]
fn seed_clamped_to_all_pairs_ceiling_for_tiny_systems() {
    // n_blocks = 4 => all-pairs ceiling 16, far below the 128*4 = 512 seed.
    assert_eq!(all_pairs_tile_capacity(4), 16);
    assert_eq!(default_interacting_tiles_capacity(4), 16);
    // n_blocks = 0 guards return 1.
    assert_eq!(default_interacting_tiles_capacity(0), 1);
    assert_eq!(all_pairs_tile_capacity(0), 1);
}

// --- GPU rebuild / pre_step tests ---

fn buffers_at(gpu: &GpuContext, px: Vec<Real>, py: Vec<Real>, pz: Vec<Real>) -> ParticleBuffers {
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

/// A simple-cubic lattice of `side^3` atoms at unit spacing in a box of
/// edge `side`. With `r_cut = 1.0`, `r_skin = 0.3` every atom has its
/// six axial neighbours inside the search radius, so a rebuild produces
/// a non-trivial packed neighbour list.
fn grid_system(gpu: &GpuContext, side: usize) -> (SimulationBox, ParticleBuffers, usize) {
    let n = side * side * side;
    let l = side as Real;
    let sb = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    let (mut px, mut py, mut pz) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                px.push(i as Real);
                py.push(j as Real);
                pz.push(k as Real);
            }
        }
    }
    (sb, buffers_at(gpu, px, py, pz), n)
}

fn cell_list_state(gpu: &GpuContext, n: usize, sim_box: &SimulationBox) -> NeighborListState {
    NeighborListState::new_cell_list(gpu, sim_box, n, 1.0, 256, 0.3).unwrap()
}

// Status-word bit layout (mirrors the private constants in
// `src/forces/neighbor_list.rs`).
const STATUS_DISPLACEMENT_TRIPPED: u32 = 1 << 0;
const STATUS_TILES_HIGH_WATER: u32 = 1 << 1;
const STATUS_TILES_OVERFLOW: u32 = 1 << 3;

/// Overwrite the combined `neighbor_status` word on the device.
fn set_status(device: &Arc<CudaDevice>, nl: &mut NeighborListState, value: u32) {
    let status = &mut nl
        .packed
        .as_mut()
        .expect("packed data present in CellList mode")
        .neighbor_status;
    device.htod_sync_copy_into(&[value], status).unwrap();
}

/// Read the live `[tiles, single_pairs]` interaction counts off the
/// device. The steady-state pipeline never does this; tests do it only
/// to assert on the sizing the device-resident counts produced.
fn read_counts(device: &Arc<CudaDevice>, nl: &NeighborListState) -> [u32; 2] {
    let host = device
        .dtoh_sync_copy(&nl.packed.as_ref().unwrap().interaction_count)
        .unwrap();
    [host[0], host[1]]
}

// rq-ea8640f5
#[test]
fn probe_rebuild_sizes_capacity_with_headroom_below_fill_threshold() {
    let gpu = init_device().unwrap();
    let (sb, buffers, n) = grid_system(&gpu, 8); // 512 atoms
    let mut nl = cell_list_state(&gpu, n, &sb);
    let mut timings = Timings::new(&gpu).unwrap();

    // The first rebuild is the synchronous probe.
    nl.rebuild(&sb, &buffers, &mut timings).unwrap();

    let count = read_counts(&gpu.device, &nl)[0] as u64;
    let packed = nl.packed.as_ref().unwrap();
    let capacity = packed.interacting_tiles_capacity as u64;
    let fill = packed.tile_pair_fill_threshold;
    let growth = packed.tile_pair_growth_factor;
    assert!(count > 0, "expected interactions");
    // Capacity holds the build with headroom: the live count is at or
    // below the high-water mark (capacity * fill_threshold), so the
    // probe left the tiles_high_water bit clear.
    assert!(
        count <= (capacity as f64 * fill).floor() as u64,
        "count {count} must be <= high-water mark of capacity {capacity}",
    );
    // ...and the probe did not over-allocate beyond one growth step past
    // that lower bound (or the O(N) seed for tiny systems). A small
    // additive slack absorbs per-step ceiling rounding.
    let seed = default_interacting_tiles_capacity(packed.n_blocks) as u64;
    let upper = seed.max((count as f64 / fill * growth).ceil() as u64 + 2);
    assert!(capacity <= upper, "capacity {capacity} exceeded headroom upper bound {upper}");
}

// rq-1ca7df49 rq-88175d6f
#[test]
fn pre_step_grows_geometrically_on_high_water_and_reports_reallocation() {
    let gpu = init_device().unwrap();
    let (sb, buffers, n) = grid_system(&gpu, 8);
    let mut nl = cell_list_state(&gpu, n, &sb);
    let mut timings = Timings::new(&gpu).unwrap();

    // Probe rebuild sizes capacity; record it.
    nl.pre_step(&sb, &buffers, &mut timings).unwrap();
    let cap_before = nl.packed.as_ref().unwrap().interacting_tiles_capacity;
    let growth = nl.packed.as_ref().unwrap().tile_pair_growth_factor;

    // Simulate the previous build having tripped the tiles high-water
    // mark (build complete, nothing dropped).
    set_status(&gpu.device, &mut nl, STATUS_TILES_HIGH_WATER);

    let outcome = nl.pre_step(&sb, &buffers, &mut timings).unwrap();
    assert!(outcome.rebuilt);
    assert!(outcome.reallocated, "a high-water grow must report reallocation");
    let cap_after = nl.packed.as_ref().unwrap().interacting_tiles_capacity;
    assert_eq!(
        cap_after,
        (cap_before as f64 * growth).ceil() as u32,
        "capacity must grow by exactly one geometric step",
    );
}

// rq-f867ab96
#[test]
fn high_water_bit_forces_rebuild_even_when_displacement_clear() {
    let gpu = init_device().unwrap();
    let (sb, buffers, n) = grid_system(&gpu, 8);
    let mut nl = cell_list_state(&gpu, n, &sb);
    let mut timings = Timings::new(&gpu).unwrap();

    nl.pre_step(&sb, &buffers, &mut timings).unwrap();
    // Only the high-water bit is set; the displacement bit is clear.
    set_status(&gpu.device, &mut nl, STATUS_TILES_HIGH_WATER);

    let outcome = nl.pre_step(&sb, &buffers, &mut timings).unwrap();
    assert!(outcome.rebuilt, "high-water alone must trigger a rebuild");
    assert!(outcome.reallocated);
}

// rq-8142fff7 rq-a5bd8157 rq-2dda3169
#[test]
fn overflow_bit_halts_with_packed_neighbor_overflow() {
    use heddle_md::forces::NeighborListError;
    let gpu = init_device().unwrap();
    let (sb, buffers, n) = grid_system(&gpu, 8);
    let mut nl = cell_list_state(&gpu, n, &sb);
    let mut timings = Timings::new(&gpu).unwrap();

    nl.pre_step(&sb, &buffers, &mut timings).unwrap();
    set_status(&gpu.device, &mut nl, STATUS_TILES_OVERFLOW);

    let err = nl.pre_step(&sb, &buffers, &mut timings).unwrap_err();
    match err {
        NeighborListError::PackedNeighborOverflow { buffer } => {
            assert_eq!(buffer, "interacting_tiles");
        }
        other => panic!("expected PackedNeighborOverflow, got {other:?}"),
    }
}

// rq-75f86ce3
#[test]
fn pre_step_that_reuses_buffers_reports_no_reallocation() {
    let gpu = init_device().unwrap();
    let (sb, buffers, n) = grid_system(&gpu, 8);
    let mut nl = cell_list_state(&gpu, n, &sb);
    let mut timings = Timings::new(&gpu).unwrap();

    // Probe sizes capacity with headroom; force a displacement-only
    // rebuild that fits the existing capacity.
    nl.pre_step(&sb, &buffers, &mut timings).unwrap();
    set_status(&gpu.device, &mut nl, STATUS_DISPLACEMENT_TRIPPED);

    let outcome = nl.pre_step(&sb, &buffers, &mut timings).unwrap();
    assert!(outcome.rebuilt);
    assert!(!outcome.reallocated, "a rebuild that reuses the buffers must not report reallocation");
}

// rq-623447db rq-a39234ba
#[test]
fn pre_step_without_rebuild_reports_neither_flag() {
    let gpu = init_device().unwrap();
    let (sb, buffers, n) = grid_system(&gpu, 8);
    let mut nl = cell_list_state(&gpu, n, &sb);
    let mut timings = Timings::new(&gpu).unwrap();

    // First pre_step runs the probe rebuild and sets the reference
    // positions. With the particles unmoved and the status word clean,
    // the second pre_step's status read sees no trip and does not rebuild.
    let first = nl.pre_step(&sb, &buffers, &mut timings).unwrap();
    assert!(first.rebuilt);

    let second = nl.pre_step(&sb, &buffers, &mut timings).unwrap();
    assert!(!second.rebuilt);
    assert!(!second.reallocated);
}

// rq-8b6d0c41 rq-b8504fa1
#[test]
fn steady_state_rebuild_produces_a_correct_list_from_device_counts() {
    let gpu = init_device().unwrap();
    let (sb, buffers, n) = grid_system(&gpu, 8);
    let mut nl = cell_list_state(&gpu, n, &sb);
    let mut timings = Timings::new(&gpu).unwrap();

    // Probe rebuild (synchronous sizing).
    nl.rebuild(&sb, &buffers, &mut timings).unwrap();
    let probe_count = read_counts(&gpu.device, &nl);

    // A second, steady-state rebuild reads its counts only on the device
    // (no host count is consulted to size launches); it must reproduce
    // the same interaction counts and leave the high-water bit clear.
    let reallocated = nl.rebuild(&sb, &buffers, &mut timings).unwrap();
    assert!(!reallocated);
    let steady_count = read_counts(&gpu.device, &nl);
    assert_eq!(probe_count, steady_count, "steady rebuild must reproduce the build");
    let status = gpu
        .device
        .dtoh_sync_copy(&nl.packed.as_ref().unwrap().neighbor_status)
        .unwrap();
    assert_eq!(status[0], 0, "a build within capacity leaves the status word clean");
}
