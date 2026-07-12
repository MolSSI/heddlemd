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
    NeighborListState::new_cell_list(gpu, sim_box, n, 1.0, 0.3).unwrap()
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

// --- Exclusion-tile build ---

/// Read the live exclusion tiles off the device: `(count, [(bi, bj)],
/// [mask row])` where each tile contributes 32 consecutive mask words.
fn read_exclusion_tiles(
    device: &Arc<CudaDevice>,
    nl: &NeighborListState,
) -> (u32, Vec<(u32, u32)>, Vec<u32>) {
    let packed = nl.packed.as_ref().unwrap();
    let count = packed.exclusion_tiles_count;
    let ib = device.dtoh_sync_copy(&packed.exclusion_tile_iblocks).unwrap();
    let jb = device.dtoh_sync_copy(&packed.exclusion_tile_jblocks).unwrap();
    let masks = device.dtoh_sync_copy(&packed.exclusion_tile_masks).unwrap();
    let pairs: Vec<(u32, u32)> = (0..count as usize).map(|t| (ib[t], jb[t])).collect();
    let mask_rows = masks[..count as usize * 32].to_vec();
    (count, pairs, mask_rows)
}

#[test]
fn exclusion_tiles_empty_when_no_exclusions() {
    let gpu = init_device().unwrap();
    let (sb, buffers, n) = grid_system(&gpu, 4); // 64 atoms, 2 blocks
    let mut nl = cell_list_state(&gpu, n, &sb);
    let mut timings = Timings::new(&gpu).unwrap();
    // No excluded pairs set.
    nl.rebuild(&sb, &buffers, &mut timings).unwrap();
    let (count, _, _) = read_exclusion_tiles(&gpu.device, &nl);
    assert_eq!(count, 0, "no exclusions => no exclusion tiles");
    // interaction_count[2] mirrors the host count.
    let ic = gpu
        .device
        .dtoh_sync_copy(&nl.packed.as_ref().unwrap().interaction_count)
        .unwrap();
    assert_eq!(ic[2], 0);
    // The skip-list CSR is all zero (no block pair is skipped).
    let offs = gpu
        .device
        .dtoh_sync_copy(&nl.packed.as_ref().unwrap().excl_jblock_offsets)
        .unwrap();
    assert!(offs.iter().all(|&o| o == 0));
}

// rq-acfb3375 (tile-presence half; the "routed away from the bulk list"
// half is covered once find_blocks skips exclusion block pairs).
#[test]
fn exclusion_tile_appears_for_excluded_block_pair() {
    let gpu = init_device().unwrap();
    let (sb, buffers, n) = grid_system(&gpu, 4); // 64 atoms, 2 blocks
    let mut nl = cell_list_state(&gpu, n, &sb);
    let mut timings = Timings::new(&gpu).unwrap();
    let excluded = vec![(0u32, 1u32), (2u32, 3u32), (10u32, 40u32)];
    nl.set_excluded_pairs(excluded.clone());
    nl.rebuild(&sb, &buffers, &mut timings).unwrap();

    // Replicate the inversion to know each pair's expected block pair.
    let sorted = gpu
        .device
        .dtoh_sync_copy(nl.sorted_particle_ids_for_packed().unwrap())
        .unwrap();
    let mut atom_slot = vec![u32::MAX; n];
    for (slot, &pid) in sorted.iter().take(n).enumerate() {
        atom_slot[pid as usize] = slot as u32;
    }
    let (count, tiles, masks) = read_exclusion_tiles(&gpu.device, &nl);
    assert!(count > 0, "excluded pairs must produce tiles");
    for &(a, b) in &excluded {
        let (sa, sb2) = (atom_slot[a as usize], atom_slot[b as usize]);
        let (si, sj) = if sa <= sb2 { (sa, sb2) } else { (sb2, sa) };
        let (bi, bj) = (si / 32, sj / 32);
        let (il, jl) = (si % 32, sj % 32);
        let t = tiles
            .iter()
            .position(|&(x, y)| x == bi && y == bj)
            .unwrap_or_else(|| panic!("no exclusion tile for block pair ({bi},{bj})"));
        let bit = masks[t * 32 + il as usize] & (1u32 << jl);
        assert!(bit != 0, "excluded pair ({a},{b}) bit not set in tile ({bi},{bj})");
    }
}

// rq-10443d06
#[test]
fn exclusion_tiles_byte_identical_across_rebuilds() {
    let gpu = init_device().unwrap();
    let (sb, buffers, n) = grid_system(&gpu, 4);
    let mut nl = cell_list_state(&gpu, n, &sb);
    let mut timings = Timings::new(&gpu).unwrap();
    nl.set_excluded_pairs(vec![(0u32, 1u32), (5u32, 33u32), (20u32, 21u32)]);

    nl.rebuild(&sb, &buffers, &mut timings).unwrap();
    let (c1, tiles1, masks1) = read_exclusion_tiles(&gpu.device, &nl);
    nl.rebuild(&sb, &buffers, &mut timings).unwrap();
    let (c2, tiles2, masks2) = read_exclusion_tiles(&gpu.device, &nl);

    assert_eq!(c1, c2, "exclusion-tile count must be identical across rebuilds");
    assert_eq!(tiles1, tiles2, "tile (bi, bj) list must be byte-identical");
    assert_eq!(masks1, masks2, "tile bitmasks must be byte-identical");
    // Tiles are emitted in ascending (bi, bj) order.
    assert!(tiles1.windows(2).all(|w| w[0] < w[1]), "tiles must be sorted and unique");
}

// rq-acfb3375 — construction routes an exclusion block pair away from
// the bulk list: after a rebuild, no packed or single-pair entry pairs
// an atom of bi with an atom of bj for any exclusion tile (bi, bj).
#[test]
fn exclusion_block_pair_absent_from_bulk_list() {
    let gpu = init_device().unwrap();
    let (sb, buffers, n) = grid_system(&gpu, 8); // 512 atoms, 16 blocks
    let mut nl = cell_list_state(&gpu, n, &sb);
    let mut timings = Timings::new(&gpu).unwrap();
    // Exclude a spread of pairs so several block pairs become exclusion
    // tiles (including at least one cross-block pair).
    nl.set_excluded_pairs(vec![
        (0u32, 1u32),
        (2u32, 3u32),
        (5u32, 40u32),
        (33u32, 130u32),
        (200u32, 201u32),
    ]);
    nl.rebuild(&sb, &buffers, &mut timings).unwrap();

    // slot(atom) -> block via the inverse sort.
    let sorted = gpu
        .device
        .dtoh_sync_copy(nl.sorted_particle_ids_for_packed().unwrap())
        .unwrap();
    let mut block_of = vec![u32::MAX; n];
    for (slot, &pid) in sorted.iter().take(n).enumerate() {
        block_of[pid as usize] = (slot / 32) as u32;
    }

    let (n_tiles, tiles, _masks) = read_exclusion_tiles(&gpu.device, &nl);
    assert!(n_tiles > 0, "excluded pairs must produce tiles");
    let excl_set: std::collections::HashSet<(u32, u32)> = tiles.into_iter().collect();

    let packed = nl.packed.as_ref().unwrap();
    let counts = read_counts(&gpu.device, &nl);
    let (n_bulk, n_single) = (counts[0] as usize, counts[1] as usize);

    // Bulk packed entries: i-block = interacting_tiles[pos]; each
    // interacting_atoms[pos*32 + lane] is a j-atom id.
    let itiles = gpu.device.dtoh_sync_copy(&packed.interacting_tiles).unwrap();
    let iatoms = gpu.device.dtoh_sync_copy(&packed.interacting_atoms).unwrap();
    for pos in 0..n_bulk {
        let bi = itiles[pos];
        for lane in 0..32usize {
            let ja = iatoms[pos * 32 + lane];
            if (ja as usize) >= n {
                continue; // padding sentinel
            }
            let bj = block_of[ja as usize];
            let key = if bi <= bj { (bi, bj) } else { (bj, bi) };
            assert!(
                !excl_set.contains(&key),
                "bulk entry pos {pos} pairs blocks {bi},{bj} which form exclusion tile {key:?}"
            );
        }
    }

    // Single-pair entries: (single_pair_atoms[2k], [2k+1]).
    let sp = gpu.device.dtoh_sync_copy(&packed.single_pair_atoms).unwrap();
    for k in 0..n_single {
        let (ai, aj) = (sp[2 * k], sp[2 * k + 1]);
        if (ai as usize) >= n || (aj as usize) >= n {
            continue;
        }
        let (bi, bj) = (block_of[ai as usize], block_of[aj as usize]);
        let key = if bi <= bj { (bi, bj) } else { (bj, bi) };
        assert!(
            !excl_set.contains(&key),
            "single pair ({ai},{aj}) sits in exclusion tile {key:?}"
        );
    }
}
