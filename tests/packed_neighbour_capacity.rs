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

fn force_capacity_to_one(nl: &mut NeighborListState) {
    nl.packed
        .as_mut()
        .expect("packed data present in CellList mode")
        .interacting_tiles_capacity = 1;
}

fn set_disp_flag(device: &Arc<CudaDevice>, nl: &mut NeighborListState, value: u32) {
    match &mut nl.mode {
        NeighborListMode::CellList(cl) => {
            device
                .htod_sync_copy_into(&[value], &mut cl.disp_rebuild_flag)
                .unwrap();
        }
        _ => panic!("test expects CellList mode"),
    }
}

// rq-ea8640f5
#[test]
fn rebuild_sizes_capacity_to_an_on_order_n_value() {
    let gpu = init_device().unwrap();
    let (sb, buffers, n) = grid_system(&gpu, 8); // 512 atoms
    let mut nl = cell_list_state(&gpu, n, &sb);
    let mut timings = Timings::new(&gpu).unwrap();

    nl.rebuild(&sb, &buffers, &mut timings).unwrap();

    let packed = nl.packed.as_ref().unwrap();
    let n_blocks = packed.n_blocks;
    let count = packed.interacting_tiles_count;
    let capacity = packed.interacting_tiles_capacity;
    assert!(count > 0, "expected interactions");
    // The probe rebuild sizes the capacity to the actual interaction
    // count (with at most one growth_factor of headroom) or the O(N)
    // seed — it is never pre-allocated to the all-pairs maximum unless
    // the system genuinely needs it.
    let seed = default_interacting_tiles_capacity(n_blocks) as u64;
    let count_with_headroom = (count as f64 * packed.tile_pair_growth_factor).ceil() as u64;
    let upper = seed.max(count_with_headroom);
    assert!(
        (count as u64) <= capacity as u64 && (capacity as u64) <= upper,
        "capacity {capacity} must lie in [{count}, {upper}] (count {count}, seed {seed})",
    );
}

// rq-8b6d0c41
#[test]
fn rebuild_that_overflows_capacity_reports_reallocation() {
    let gpu = init_device().unwrap();
    let (sb, buffers, n) = grid_system(&gpu, 8);
    let mut nl = cell_list_state(&gpu, n, &sb);
    let mut timings = Timings::new(&gpu).unwrap();

    // Settle the list, then artificially shrink the capacity so the next
    // rebuild's true count overflows it and the buffers must grow.
    nl.rebuild(&sb, &buffers, &mut timings).unwrap();
    force_capacity_to_one(&mut nl);

    let reallocated = nl.rebuild(&sb, &buffers, &mut timings).unwrap();
    assert!(reallocated, "a rebuild that grew the buffers must report reallocation");
}

// rq-75f86ce3
#[test]
fn rebuild_that_fits_reports_no_reallocation() {
    let gpu = init_device().unwrap();
    let (sb, buffers, n) = grid_system(&gpu, 8);
    let mut nl = cell_list_state(&gpu, n, &sb);
    let mut timings = Timings::new(&gpu).unwrap();

    // First rebuild grows the seed to fit; the second reuses that
    // capacity unchanged and reports no reallocation.
    nl.rebuild(&sb, &buffers, &mut timings).unwrap();
    let reallocated = nl.rebuild(&sb, &buffers, &mut timings).unwrap();
    assert!(!reallocated, "a rebuild that reuses the buffers must report no reallocation");
}

// rq-1ca7df49
#[test]
fn pre_step_reports_reallocation_when_rebuild_grows() {
    let gpu = init_device().unwrap();
    let (sb, buffers, n) = grid_system(&gpu, 8);
    let mut nl = cell_list_state(&gpu, n, &sb);
    let mut timings = Timings::new(&gpu).unwrap();

    // Settle, shrink capacity, and trip the displacement flag so the
    // next pre_step rebuilds and is forced to grow.
    nl.pre_step(&sb, &buffers, &mut timings).unwrap();
    force_capacity_to_one(&mut nl);
    set_disp_flag(&gpu.device, &mut nl, 1);

    let outcome = nl.pre_step(&sb, &buffers, &mut timings).unwrap();
    assert!(outcome.rebuilt);
    assert!(outcome.reallocated);
}

// rq-623447db
#[test]
fn pre_step_without_rebuild_reports_neither_flag() {
    let gpu = init_device().unwrap();
    let (sb, buffers, n) = grid_system(&gpu, 8);
    let mut nl = cell_list_state(&gpu, n, &sb);
    let mut timings = Timings::new(&gpu).unwrap();

    // First pre_step rebuilds (needs_rebuild starts true) and sets the
    // reference positions. With the particles unmoved, the second
    // pre_step's displacement check sees zero motion and does not rebuild.
    let first = nl.pre_step(&sb, &buffers, &mut timings).unwrap();
    assert!(first.rebuilt);

    let second = nl.pre_step(&sb, &buffers, &mut timings).unwrap();
    assert!(!second.rebuilt);
    assert!(!second.reallocated);
}
