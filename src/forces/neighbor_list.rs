// rq-693ce6fa rq-b2d68288
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaDevice, CudaSlice};

use crate::gpu::{
    GpuContext, GpuError, Kernels, ParticleBuffers, SPATIAL_HASH_SCAN_BLOCK_SIZE,
    compute_cell_indices_and_histogram, copy_positions_into_reference,
    neighbor_displacement_check_flag, prefix_scan_cell_counts,
    scatter_atoms_into_cells, sort_cells_by_particle_id,
};
use crate::pbc::SimulationBox;
use crate::timings::{HostStage, KernelStage, Timings};
use crate::precision::{Real, Real4};

// rq-d8e4407a rq-e1ceb5c0 rq-6cf916af rq-1bbcf3b7
#[derive(Debug, thiserror::Error)]
pub enum NeighborListError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("an atom has more than {max} neighbors")]
    NeighborListOverflow { max: u32 },
    // rq-2dda3169
    #[error("packed-neighbour buffer `{buffer}` overflowed: a build would have dropped interactions within the search radius")]
    PackedNeighborOverflow { buffer: &'static str },
    #[error("simulation box perpendicular width along lattice direction `{direction}` is {width}, below the required {required}")]
    BoxTooSmallForCells {
        direction: &'static str,
        width: Real,
        required: Real,
    },
    #[error("cell grid has {n_cells_total} cells, exceeding the device limit of {max_supported}")]
    TooManyCells {
        n_cells_total: usize,
        max_supported: usize,
    },
}

// rq-ff424773
#[derive(Debug)]
pub enum NeighborListMode {
    Trivial,
    CellList(CellListData),
    // CellListOnly produces the cell-list output (sorted particle IDs +
    // per-cell offsets) without building a neighbor list. Used by the
    // SPME reciprocal-space slot; see `rqm/forces/spme.md`.
    CellListOnly(CellListData),
}

// rq-ff424773
#[derive(Debug)]
pub struct CellListData {
    pub n_cells: [u32; 3],
    pub n_cells_total: usize,
    pub r_cut: Real,
    pub r_skin: Real,
    pub r_search_sq: Real,
    pub cached_generation: u64,
    pub cell_indices: CudaSlice<u32>,
    pub cell_counts: CudaSlice<u32>,
    pub write_cursors: CudaSlice<u32>,
    pub scan_block_totals: Vec<CudaSlice<u32>>,
    pub sorted_particle_ids: CudaSlice<u32>,
    pub cell_offsets: CudaSlice<u32>,
    pub reference_positions_x: CudaSlice<Real>,
    pub reference_positions_y: CudaSlice<Real>,
    pub reference_positions_z: CudaSlice<Real>,
    /// `(r_skin / 2)²` cached as a host-side scalar so it can be
    /// passed by value to the displacement-check kernel without an
    /// extra device read. Set once at construction and again only
    /// when `r_skin` itself changes (it does not change across a
    /// phase).
    pub threshold_sq: Real,
    pub overflow_flag: CudaSlice<u32>,
    pub needs_rebuild: bool,
}

/// Packed-neighbour data held by `NeighborListState`. See
/// `rqm/forces/packed-neighbour-pair-force.md`.
#[derive(Debug)]
pub struct PackedNeighborData {
    /// Number of 32-atom blocks: `ceil(particle_count / 32)`.
    pub n_blocks: u32,
    /// Per-block real-atom count (`32` except possibly for the last
    /// block).
    pub tile_atom_count: CudaSlice<u32>,
    /// Per-block active-lane bitmask.
    pub tile_lane_mask: CudaSlice<u32>,
    /// Block-order interleaved `(x, y, z, q)` positions+charges
    /// (refreshed every step by
    /// `scatter_positions_to_tile_order`). Length `n_blocks * 32`.
    pub tile_sorted_posq: CudaSlice<Real4>,
    /// Per-block centre `(x, y, z, max_disp_sq)` packed as 4 `Real`.
    pub block_centre: CudaSlice<Real>,
    /// Per-block bbox half-extents `(dx, dy, dz)` packed as 3 `Real`.
    pub block_bbox: CudaSlice<Real>,
    /// Sorted-particle-ids view used by the construction kernel.
    /// In CellList mode this aliases `CellListData::sorted_particle_ids`
    /// via `Option::None` here (the construction kernel reads the
    /// cell-list buffer directly); in Trivial mode this carries an
    /// identity permutation.
    pub trivial_sorted_particle_ids: Option<CudaSlice<u32>>,
    /// Per-entry i-block index. Length `interacting_tiles_capacity`.
    pub interacting_tiles: CudaSlice<u32>,
    /// Per-entry packed 32 individual j-atom IDs. Length
    /// `interacting_tiles_capacity * 32`.
    pub interacting_atoms: CudaSlice<u32>,
    /// Live interaction counts: `[interacting_tiles_count,
    /// single_pairs_count]`. Read on the device by every downstream
    /// kernel (histogram, scan, scatter, force passes); never copied to
    /// the host on a steady-state rebuild. rq-67a09135
    pub interaction_count: CudaSlice<u32>,
    /// Combined rebuild status word. Bit 0 (`displacement_tripped`) is
    /// set by the per-step displacement-check kernel; bits 1-4
    /// (`tiles_high_water`, `single_pairs_high_water`, `tiles_overflow`,
    /// `single_pairs_overflow`) by the packed construction's
    /// `set_neighbor_status_bits`. Zeroed at the start of every rebuild
    /// and read once per batch boundary by `pre_step`. See
    /// `rqm/forces/neighbor-list.md` *Displacement Check*. rq-67a09135 rq-1f38d78a
    pub neighbor_status: CudaSlice<u32>,
    /// Current allocated capacity.
    pub interacting_tiles_capacity: u32,
    /// Live entry count after the most recent rebuild.
    pub interacting_tiles_count: u32,
    /// Geometric multiplier applied to a capacity when it is grown.
    pub tile_pair_growth_factor: f64,
    /// Fraction of a capacity at which a build is treated as near-full
    /// and the capacity is grown ahead of any dropped entry. In `(0, 1)`.
    /// rq-67a09135
    pub tile_pair_fill_threshold: f64,
    /// Per-i-block count of entries belonging to that i-block.
    /// Length `n_blocks`. Filled by `histogram_entries_by_iblock` at
    /// each rebuild from the live `interacting_tiles` array.
    pub iblock_count: CudaSlice<u32>,
    /// Prefix-scan of `iblock_count`. Length `n_blocks + 1`. Slot
    /// `iblock_offset[b]` is the start of i-block `b`'s entries inside
    /// the sorted view; `iblock_offset[n_blocks]` is the total entry
    /// count.
    pub iblock_offset: CudaSlice<u32>,
    /// Per-rebuild scratch used by `scatter_entries_by_iblock` to
    /// claim destination slots inside each i-block's contiguous range
    /// via `atomicAdd`. Zeroed before every scatter call.
    pub iblock_cursor: CudaSlice<u32>,
    /// Prefix-scan ladder for `iblock_count → iblock_offset`. Same
    /// shape as `CellListData::scan_block_totals`.
    pub iblock_scan_block_totals: Vec<CudaSlice<u32>>,
    /// `interacting_atoms` re-arranged so that all entries for i-block
    /// `b` lie contiguously in `[iblock_offset[b], iblock_offset[b+1])`.
    /// Length `interacting_tiles_capacity * 32`. Produced by the
    /// scatter pass at every rebuild; consumed by the JIT pair-force
    /// kernel in place of `interacting_atoms`.
    pub sorted_interacting_atoms: CudaSlice<u32>,
    /// Sparse-tile (i_atom, j_atom) pairs extracted at neighbour-list
    /// build time. Length `2 * single_pairs_capacity`. Interleaved
    /// `[i0, j0, i1, j1, …]` of original atom IDs. Consumed by the
    /// JIT single-pair entry point (one thread per pair). See
    /// `rqm/forces/packed-neighbour-pair-force.md` *Neighbour List*.
    pub single_pair_atoms: CudaSlice<u32>,
    /// Allocated capacity of `single_pair_atoms` measured in pairs.
    /// The underlying `CudaSlice<u32>` has `2 * single_pairs_capacity`
    /// slots.
    pub single_pairs_capacity: u32,
    /// Live single-pair count after the most recent rebuild
    /// (`interaction_count[1]` on the device).
    pub single_pairs_count: u32,
}

/// Outcome of a `NeighborListState::pre_step` call. `rebuilt` is `true`
/// when the call ran a rebuild; `reallocated` is `true` when that
/// rebuild grew (and therefore reallocated) a packed-neighbour buffer
/// (`interacting_tiles`, `interacting_atoms`, or `single_pair_atoms`).
/// The batched graph-replay loop re-captures the phase graph when
/// `reallocated` is set (see `rqm/cuda-graphs.md`).
///
/// rq-1217c816
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PreStepOutcome {
    pub rebuilt: bool,
    pub reallocated: bool,
}

// rq-b2d68288
#[derive(Debug)]
pub struct NeighborListState {
    pub device: Arc<CudaDevice>,
    pub kernels: Arc<Kernels>,
    pub particle_count: usize,
    pub mode: NeighborListMode,
    /// Packed-neighbour data populated in `CellList` and `Trivial`
    /// modes; `None` for `CellListOnly` (SPME bin-only mode).
    pub packed: Option<PackedNeighborData>,
    // Monotonically-increasing counter incremented at the end of every
    // successful `rebuild_impl`. Downstream consumers (e.g. the SPME
    // reciprocal-space slot's atom spatial pre-sort) cache the last
    // observed value and re-run their per-rebuild work when the
    // generation advances.
    rebuild_generation: u64,
    /// `false` until the first (probe) rebuild of the state has run. The
    /// probe sizes the packed-neighbour capacity by reading
    /// `neighbor_status` synchronously and growing geometrically until no
    /// high-water or overflow bit is set; every later rebuild is
    /// dtoh-free and relies on the per-batch status read for growth.
    /// rq-67a09135
    has_probed: bool,
}

/// Default fraction of a packed-neighbour capacity at which a build is
/// treated as near-full and the capacity is grown ahead of any dropped
/// entry. See `rqm/forces/packed-neighbour-pair-force.md` *Capacity*.
/// rq-67a09135
pub const DEFAULT_TILE_PAIR_FILL_THRESHOLD: f64 = 0.8;

/// Default geometric growth multiplier for packed-neighbour capacities.
pub const DEFAULT_TILE_PAIR_GROWTH_FACTOR: f64 = 1.5;

// Bit layout of the combined `neighbor_status` word. See
// `rqm/forces/neighbor-list.md` *Displacement Check*. rq-1f38d78a rq-67a09135
const STATUS_DISPLACEMENT_TRIPPED: u32 = 1 << 0;
const STATUS_TILES_HIGH_WATER: u32 = 1 << 1;
const STATUS_SINGLE_PAIRS_HIGH_WATER: u32 = 1 << 2;
const STATUS_TILES_OVERFLOW: u32 = 1 << 3;
const STATUS_SINGLE_PAIRS_OVERFLOW: u32 = 1 << 4;

/// O(N) seed capacity for `interacting_tiles`. The buffers are sized
/// to the *actual* interaction count, never to the `O(n_blocks^2)`
/// all-pairs bound: this seed is just a reasonable starting point, and
/// the first (probe) rebuild grows it to the true count via
/// overflow-driven growth. For a cutoff system the working capacity is
/// `O(N)`. Clamped down to the all-pairs maximum for tiny systems where
/// that is smaller than the seed.
///
/// rq-67a09135
pub fn default_interacting_tiles_capacity(n_blocks: u32) -> u32 {
    if n_blocks == 0 {
        return 1;
    }
    // ~TILES_PER_BLOCK_SEED packed entries per i-block. A dense liquid
    // i-block sees on the order of this many 32-j-atom entries; an
    // under- or over-estimate only changes how many times the probe
    // rebuild grows, never correctness.
    const TILES_PER_BLOCK_SEED: u64 = 128;
    let seed = (n_blocks as u64).saturating_mul(TILES_PER_BLOCK_SEED);
    let ceiling = all_pairs_tile_capacity(n_blocks) as u64;
    seed.min(ceiling).max(1).min(u32::MAX as u64) as u32
}

/// All-pairs maximum number of packed `interacting_tiles` entries:
/// every one of the `n_blocks` i-blocks can pack at most `N` j-atoms,
/// i.e. `n_blocks` entries of 32, giving `n_blocks^2` entries total.
/// Growth never exceeds this ceiling; a build whose true count would
/// exceed it is a pathology and surfaces as `NeighborListOverflow`.
/// Saturated to `u32::MAX`.
///
/// rq-67a09135
pub fn all_pairs_tile_capacity(n_blocks: u32) -> u32 {
    let nb = n_blocks as u64;
    nb.saturating_mul(nb).max(1).min(u32::MAX as u64) as u32
}

fn alloc_packed_neighbor_data(
    device: &Arc<CudaDevice>,
    particle_count: usize,
    interacting_tiles_capacity: u32,
    tile_pair_growth_factor: f64,
    tile_pair_fill_threshold: f64,
    trivial_mode: bool,
) -> Result<PackedNeighborData, NeighborListError> {
    let n_blocks = ((particle_count as u32) + 31) / 32;
    let n_blocks_alloc = (n_blocks.max(1)) as usize;
    let padded_n = (n_blocks_alloc * 32).max(1);
    let cap = interacting_tiles_capacity.max(1);
    let cap_alloc = cap as usize;

    let mut atom_count_host = vec![32u32; n_blocks_alloc];
    let mut lane_mask_host = vec![0xFFFF_FFFFu32; n_blocks_alloc];
    if n_blocks > 0 && (particle_count as u32) % 32 != 0 {
        let last = (n_blocks - 1) as usize;
        let r = (particle_count as u32) % 32;
        atom_count_host[last] = r;
        lane_mask_host[last] = (1u32 << r) - 1;
    }
    if n_blocks == 0 {
        atom_count_host[0] = 0;
        lane_mask_host[0] = 0;
    }
    let tile_atom_count = device
        .htod_sync_copy(&atom_count_host)
        .map_err(GpuError::from)?;
    let tile_lane_mask = device
        .htod_sync_copy(&lane_mask_host)
        .map_err(GpuError::from)?;
    let tile_sorted_posq = device
        .alloc_zeros::<Real4>(padded_n)
        .map_err(GpuError::from)?;
    let block_centre = device
        .alloc_zeros::<Real>(n_blocks_alloc * 4)
        .map_err(GpuError::from)?;
    let block_bbox = device
        .alloc_zeros::<Real>(n_blocks_alloc * 3)
        .map_err(GpuError::from)?;
    let trivial_sorted_particle_ids = if trivial_mode {
        let identity: Vec<u32> = (0..particle_count as u32).collect();
        if identity.is_empty() {
            Some(device.alloc_zeros::<u32>(1).map_err(GpuError::from)?)
        } else {
            Some(device.htod_sync_copy(&identity).map_err(GpuError::from)?)
        }
    } else {
        None
    };
    let interacting_tiles = device
        .alloc_zeros::<u32>(cap_alloc)
        .map_err(GpuError::from)?;
    let interacting_atoms = device
        .alloc_zeros::<u32>(cap_alloc * 32)
        .map_err(GpuError::from)?;
    let interaction_count = device.alloc_zeros::<u32>(2).map_err(GpuError::from)?;
    let neighbor_status = device.alloc_zeros::<u32>(1).map_err(GpuError::from)?;
    let iblock_count = device
        .alloc_zeros::<u32>(n_blocks_alloc)
        .map_err(GpuError::from)?;
    let iblock_offset = device
        .alloc_zeros::<u32>(n_blocks_alloc + 1)
        .map_err(GpuError::from)?;
    let iblock_cursor = device
        .alloc_zeros::<u32>(n_blocks_alloc)
        .map_err(GpuError::from)?;
    let iblock_scan_block_totals = alloc_scan_block_totals(&device, n_blocks_alloc)?;
    let sorted_interacting_atoms = device
        .alloc_zeros::<u32>(cap_alloc * 32)
        .map_err(GpuError::from)?;
    // Single-pair initial capacity. An O(N) seed sharing the same
    // density heuristic as the tile list; the probe rebuild grows it to
    // the true sparse-pair count via overflow-driven growth. A mid-phase
    // growth that reallocates this buffer inside a captured graph is
    // handled by re-capturing the phase graph (see
    // `rqm/forces/packed-neighbour-pair-force.md` *Capacity* and
    // `rqm/cuda-graphs.md`), so pre-sizing to the all-pairs bound is no
    // longer required.
    let single_pairs_capacity = default_interacting_tiles_capacity(n_blocks).max(1);
    let single_pair_atoms = device
        .alloc_zeros::<u32>(2 * single_pairs_capacity as usize)
        .map_err(GpuError::from)?;

    Ok(PackedNeighborData {
        n_blocks,
        tile_atom_count,
        tile_lane_mask,
        tile_sorted_posq,
        block_centre,
        block_bbox,
        trivial_sorted_particle_ids,
        interacting_tiles,
        interacting_atoms,
        interaction_count,
        neighbor_status,
        interacting_tiles_capacity: cap,
        interacting_tiles_count: 0,
        tile_pair_growth_factor,
        tile_pair_fill_threshold,
        iblock_count,
        iblock_offset,
        iblock_cursor,
        iblock_scan_block_totals,
        sorted_interacting_atoms,
        single_pair_atoms,
        single_pairs_capacity,
        single_pairs_count: 0,
    })
}

impl PackedNeighborData {
    /// High-water mark for the tile list: `floor(capacity * fill_threshold)`.
    /// A build whose live tile count exceeds this is grown ahead of an
    /// actual overflow. rq-67a09135
    pub fn tiles_high_water_mark(&self) -> u32 {
        ((self.interacting_tiles_capacity as f64) * self.tile_pair_fill_threshold)
            .floor() as u32
    }

    /// High-water mark for the single-pair list. rq-67a09135
    pub fn single_pairs_high_water_mark(&self) -> u32 {
        ((self.single_pairs_capacity as f64) * self.tile_pair_fill_threshold)
            .floor() as u32
    }

    /// Grow the entry-list buffers geometrically: the new capacity is
    /// `ceil(capacity * tile_pair_growth_factor)`. Count-free — the host
    /// never reads the live interaction count — so it is usable both in
    /// the synchronous probe loop and at a steady-state batch boundary.
    ///
    /// rq-67a09135
    pub fn grow_tiles(&mut self, device: &Arc<CudaDevice>) -> Result<(), GpuError> {
        let new_cap_f = (self.interacting_tiles_capacity as f64)
            * self.tile_pair_growth_factor;
        let new_cap = (new_cap_f.ceil() as u32)
            .max(self.interacting_tiles_capacity + 1)
            .max(1);
        let new_alloc = new_cap as usize;
        self.interacting_tiles = device.alloc_zeros::<u32>(new_alloc)?;
        self.interacting_atoms = device.alloc_zeros::<u32>(new_alloc * 32)?;
        self.sorted_interacting_atoms = device.alloc_zeros::<u32>(new_alloc * 32)?;
        self.interacting_tiles_capacity = new_cap;
        Ok(())
    }

    /// Grow `single_pair_atoms` geometrically. rq-67a09135
    pub fn grow_single_pairs(&mut self, device: &Arc<CudaDevice>) -> Result<(), GpuError> {
        let new_cap_f = (self.single_pairs_capacity as f64) * self.tile_pair_growth_factor;
        let new_cap = (new_cap_f.ceil() as u32)
            .max(self.single_pairs_capacity + 1)
            .max(1);
        self.single_pair_atoms =
            device.alloc_zeros::<u32>(2 * new_cap as usize)?;
        self.single_pairs_capacity = new_cap;
        Ok(())
    }
}

impl NeighborListState {
    /// Borrow the cell-list-specific data; returns `None` when the state is
    /// in `Trivial` mode.
    pub fn cell_list_data(&self) -> Option<&CellListData> {
        match &self.mode {
            NeighborListMode::Trivial => None,
            NeighborListMode::CellList(cl) | NeighborListMode::CellListOnly(cl) => Some(cl),
        }
    }

    /// Mutable borrow of the cell-list-specific data; returns `None` when
    /// the state is in `Trivial` mode.
    pub fn cell_list_data_mut(&mut self) -> Option<&mut CellListData> {
        match &mut self.mode {
            NeighborListMode::Trivial => None,
            NeighborListMode::CellList(cl) | NeighborListMode::CellListOnly(cl) => Some(cl),
        }
    }

    /// `true` when the state is in CellListOnly mode (bin structure only,
    /// no neighbor list).
    pub fn is_bin_only(&self) -> bool {
        matches!(self.mode, NeighborListMode::CellListOnly(_))
    }

    /// Monotonically-increasing rebuild counter. Bumped once every time
    /// `rebuild` completes successfully. Downstream consumers cache the
    /// observed value to detect when a rebuild has occurred.
    pub fn rebuild_generation(&self) -> u64 {
        self.rebuild_generation
    }

    // rq-14033af1
    pub fn new_cell_list(
        gpu: &GpuContext,
        sim_box: &SimulationBox,
        particle_count: usize,
        r_cut: Real,
        r_skin: Real,
    ) -> Result<Self, NeighborListError> {
        let device = gpu.device.clone();
        let kernels = gpu.kernels.clone();
        debug_assert!(r_cut > 0.0);
        // r_skin == 0 is permitted (degenerate but valid: every
        // step's displacement check triggers a rebuild because
        // max_disp > r_skin * 0.5 = 0 unless the system is fully
        // stationary). Negative is malformed.
        debug_assert!(r_skin >= 0.0);
        let r_search = r_cut + r_skin;
        let r_search_sq = r_search * r_search;
        let n_cells = compute_cell_layout(sim_box, r_search)?;
        let n_cells_total =
            n_cells[0] as usize * n_cells[1] as usize * n_cells[2] as usize;
        check_n_cells_total(n_cells_total)?;

        let cell_indices = device
            .alloc_zeros::<u32>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let cell_counts = device
            .alloc_zeros::<u32>(n_cells_total)
            .map_err(GpuError::from)?;
        let write_cursors = device
            .alloc_zeros::<u32>(n_cells_total)
            .map_err(GpuError::from)?;
        let scan_block_totals = alloc_scan_block_totals(&device, n_cells_total)?;
        let sorted_particle_ids = device
            .alloc_zeros::<u32>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let cell_offsets = device
            .alloc_zeros::<u32>(n_cells_total + 1)
            .map_err(GpuError::from)?;
        let reference_positions_x = device
            .alloc_zeros::<Real>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let reference_positions_y = device
            .alloc_zeros::<Real>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let reference_positions_z = device
            .alloc_zeros::<Real>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let half_skin = (r_skin as f64) * 0.5;
        let threshold_sq: Real = (half_skin * half_skin) as Real;
        let overflow_flag = device.alloc_zeros::<u32>(1).map_err(GpuError::from)?;

        let n_blocks = ((particle_count as u32) + 31) / 32;
        let initial_cap = default_interacting_tiles_capacity(n_blocks);
        let packed = alloc_packed_neighbor_data(
            &device,
            particle_count,
            initial_cap,
            DEFAULT_TILE_PAIR_GROWTH_FACTOR,
            DEFAULT_TILE_PAIR_FILL_THRESHOLD,
            false, // CellList mode uses cell-list's sorted_particle_ids
        )?;

        Ok(NeighborListState {
            device,
            kernels,
            particle_count,
            packed: Some(packed),
            mode: NeighborListMode::CellList(CellListData {
                n_cells,
                n_cells_total,
                r_cut,
                r_skin,
                r_search_sq,
                cached_generation: sim_box.generation(),
                cell_indices,
                cell_counts,
                write_cursors,
                scan_block_totals,
                sorted_particle_ids,
                cell_offsets,
                reference_positions_x,
                reference_positions_y,
                reference_positions_z,
                threshold_sq,
                overflow_flag,
                needs_rebuild: true,
            }),
            rebuild_generation: 0,
            has_probed: false,
        })
    }

    // rq-9ca00d25 rq-202493a5
    //
    // Build a bin-only cell-list state with explicit grid dimensions.
    // Used by the SPME reciprocal-space slot (see rqm/forces/spme.md).
    // The state produces sorted particle IDs and per-cell offsets but
    // no neighbor list; the neighbor-list-build and displacement-check
    // kernels are never launched.
    // rq-d47caa3d
    pub fn new_cell_list_only(
        gpu: &GpuContext,
        sim_box: &SimulationBox,
        particle_count: usize,
        n_cells_per_direction: [u32; 3],
    ) -> Result<Self, NeighborListError> {
        let device = gpu.device.clone();
        let kernels = gpu.kernels.clone();

        let direction_names: [&'static str; 3] = ["a", "b", "c"];
        for d in 0..3 {
            if n_cells_per_direction[d] < 3 {
                return Err(NeighborListError::BoxTooSmallForCells {
                    direction: direction_names[d],
                    width: 0.0,
                    required: 3.0,
                });
            }
        }
        let n_cells = n_cells_per_direction;
        let n_cells_total =
            n_cells[0] as usize * n_cells[1] as usize * n_cells[2] as usize;
        check_n_cells_total(n_cells_total)?;

        // Cell-list scratch + outputs.
        let cell_indices = device
            .alloc_zeros::<u32>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let cell_counts = device
            .alloc_zeros::<u32>(n_cells_total)
            .map_err(GpuError::from)?;
        let write_cursors = device
            .alloc_zeros::<u32>(n_cells_total)
            .map_err(GpuError::from)?;
        let scan_block_totals = alloc_scan_block_totals(&device, n_cells_total)?;
        let sorted_particle_ids = device
            .alloc_zeros::<u32>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let cell_offsets = device
            .alloc_zeros::<u32>(n_cells_total + 1)
            .map_err(GpuError::from)?;

        let reference_positions_x = device.alloc_zeros::<Real>(0).map_err(GpuError::from)?;
        let reference_positions_y = device.alloc_zeros::<Real>(0).map_err(GpuError::from)?;
        let reference_positions_z = device.alloc_zeros::<Real>(0).map_err(GpuError::from)?;
        let overflow_flag = device.alloc_zeros::<u32>(0).map_err(GpuError::from)?;

        Ok(NeighborListState {
            device,
            kernels,
            particle_count,
            packed: None,
            mode: NeighborListMode::CellListOnly(CellListData {
                n_cells,
                n_cells_total,
                r_cut: 0.0,
                r_skin: 0.0,
                r_search_sq: 0.0,
                cached_generation: sim_box.generation(),
                cell_indices,
                cell_counts,
                write_cursors,
                scan_block_totals,
                sorted_particle_ids,
                cell_offsets,
                reference_positions_x,
                reference_positions_y,
                reference_positions_z,
                threshold_sq: 0.0,
                overflow_flag,
                needs_rebuild: true,
            }),
            rebuild_generation: 0,
            has_probed: false,
        })
    }

    // rq-c96fd9d2
    pub fn new_trivial(
        gpu: &GpuContext,
        _sim_box: &SimulationBox,
        particle_count: usize,
    ) -> Result<Self, NeighborListError> {
        let device = gpu.device.clone();
        let kernels = gpu.kernels.clone();

        let packed = if particle_count == 0 {
            None
        } else {
            let n_blocks = ((particle_count as u32) + 31) / 32;
            // Trivial mode: enumerate the upper-triangular set of
            // (i_block, j_block) tile pairs with j_block >= i_block.
            // That totals n_blocks * (n_blocks + 1) / 2 entries, which
            // is the emission count of the loop below and therefore
            // the exact size the packed buffers must be sized to
            // (cudarc's `htod_sync_copy_into` requires the host and
            // device slices to have equal length).
            let total_entries = (n_blocks as u64) * (n_blocks as u64 + 1) / 2;
            let cap = total_entries.min(u32::MAX as u64).max(1) as u32;
            let mut packed = alloc_packed_neighbor_data(
                &device,
                particle_count,
                cap,
                DEFAULT_TILE_PAIR_GROWTH_FACTOR,
                DEFAULT_TILE_PAIR_FILL_THRESHOLD,
                true,
            )?;

            // Populate `interacting_tiles` and `interacting_atoms` for
            // all-pairs enumeration. For each i_block and each j_block
            // with j_block >= i_block, emit one entry with 32 packed
            // j-atom IDs (drawn from j_block * 32 .. j_block * 32 + 32,
            // sentinel-padded if past N).
            let mut tiles_host: Vec<u32> = Vec::with_capacity(cap as usize);
            let mut atoms_host: Vec<u32> = Vec::with_capacity((cap as usize) * 32);
            let sentinel = particle_count as u32;
            for i_block in 0..n_blocks {
                for j_block in i_block..n_blocks {
                    tiles_host.push(i_block);
                    for lane in 0..32u32 {
                        let atom = j_block * 32 + lane;
                        if atom < sentinel {
                            atoms_host.push(atom);
                        } else {
                            atoms_host.push(sentinel);
                        }
                    }
                }
            }
            packed.interacting_tiles_count = tiles_host.len() as u32;
            if !tiles_host.is_empty() {
                device
                    .htod_sync_copy_into(&tiles_host, &mut packed.interacting_tiles)
                    .map_err(GpuError::from)?;
                device
                    .htod_sync_copy_into(&atoms_host, &mut packed.interacting_atoms)
                    .map_err(GpuError::from)?;
                // The all-pairs enumeration above already emits entries
                // in i-block-sorted order, so `sorted_interacting_atoms`
                // is the same content as `interacting_atoms`.
                device
                    .htod_sync_copy_into(&atoms_host, &mut packed.sorted_interacting_atoms)
                    .map_err(GpuError::from)?;
            }
            // The JIT pair-force kernel reads the entry count from device
            // memory (so a captured CUDA graph picks up the live value);
            // mirror the host-side count to interaction_count[0].
            let count_host = [packed.interacting_tiles_count, 0u32];
            device
                .htod_sync_copy_into(&count_host, &mut packed.interaction_count)
                .map_err(GpuError::from)?;
            // Populate per-i-block count + prefix-scan offsets from the
            // host-built sorted layout: each i-block emits `(n_blocks - b)`
            // entries (the j_block ∈ [b, n_blocks) range).
            let mut iblock_count_host: Vec<u32> = vec![0u32; n_blocks as usize];
            for &t in &tiles_host {
                iblock_count_host[t as usize] += 1;
            }
            let mut iblock_offset_host: Vec<u32> = vec![0u32; (n_blocks as usize) + 1];
            let mut acc: u32 = 0;
            for b in 0..n_blocks as usize {
                iblock_offset_host[b] = acc;
                acc += iblock_count_host[b];
            }
            iblock_offset_host[n_blocks as usize] = acc;
            if n_blocks > 0 {
                device
                    .htod_sync_copy_into(&iblock_count_host, &mut packed.iblock_count)
                    .map_err(GpuError::from)?;
                device
                    .htod_sync_copy_into(&iblock_offset_host, &mut packed.iblock_offset)
                    .map_err(GpuError::from)?;
            }

            // Populate the tile-sorted positions view's padding lanes
            // with +inf so the force kernel treats them as inactive.
            let pos_inf = 3.4e38 as Real;
            let padded_n = (n_blocks as usize) * 32;
            if padded_n > particle_count {
                let pad_count = padded_n - particle_count;
                let pad = vec![
                    Real4 { x: pos_inf, y: pos_inf, z: pos_inf, w: 0.0 };
                    pad_count
                ];
                let mut view = packed.tile_sorted_posq.slice_mut(particle_count..);
                device.htod_sync_copy_into(&pad, &mut view).map_err(GpuError::from)?;
            }
            Some(packed)
        };

        Ok(NeighborListState {
            device,
            kernels,
            particle_count,
            packed,
            mode: NeighborListMode::Trivial,
            rebuild_generation: 0,
            // Trivial mode pre-populates the packed list on the host and
            // never runs the construction probe.
            has_probed: true,
        })
    }

    /// Returns the sorted-particle-ids buffer the packed-neighbour
    /// pipeline should read for block-to-atom-ID translation. CellList
    /// mode uses the cell-list's sort; Trivial mode uses the identity
    /// permutation owned by `PackedNeighborData`.
    pub fn sorted_particle_ids_for_packed(&self) -> Option<&CudaSlice<u32>> {
        match (&self.mode, &self.packed) {
            (NeighborListMode::CellList(cl), Some(_)) => Some(&cl.sorted_particle_ids),
            (NeighborListMode::Trivial, Some(p)) => p.trivial_sorted_particle_ids.as_ref(),
            _ => None,
        }
    }

    // rq-282af621
    fn refresh_cell_layout_if_box_changed(
        &mut self,
        sim_box: &SimulationBox,
    ) -> Result<bool, NeighborListError> {
        let device = self.device.clone();
        let cl = match &mut self.mode {
            NeighborListMode::Trivial => return Ok(false),
            // Bin-only mode has a fixed n_cells_per_direction (the FFT
            // grid resolution); the box-generation refresh re-records
            // the generation but does not re-derive n_cells from
            // r_cut/r_skin.
            NeighborListMode::CellListOnly(cl) => {
                if sim_box.generation() != cl.cached_generation {
                    cl.cached_generation = sim_box.generation();
                    cl.needs_rebuild = true;
                    return Ok(true);
                }
                return Ok(false);
            }
            NeighborListMode::CellList(cl) => cl,
        };
        if sim_box.generation() == cl.cached_generation {
            return Ok(false);
        }
        let r_search = cl.r_cut + cl.r_skin;
        let new_n_cells = compute_cell_layout(sim_box, r_search)?;
        let new_n_cells_total =
            new_n_cells[0] as usize * new_n_cells[1] as usize * new_n_cells[2] as usize;
        check_n_cells_total(new_n_cells_total)?;
        let n_cells_total_changed = new_n_cells_total != cl.n_cells_total;
        if n_cells_total_changed {
            cl.cell_offsets = device
                .alloc_zeros::<u32>(new_n_cells_total + 1)
                .map_err(GpuError::from)?;
            cl.cell_counts = device
                .alloc_zeros::<u32>(new_n_cells_total)
                .map_err(GpuError::from)?;
            cl.write_cursors = device
                .alloc_zeros::<u32>(new_n_cells_total)
                .map_err(GpuError::from)?;
            cl.scan_block_totals =
                alloc_scan_block_totals(&device, new_n_cells_total)?;
        }
        cl.n_cells = new_n_cells;
        cl.n_cells_total = new_n_cells_total;
        cl.cached_generation = sim_box.generation();
        if n_cells_total_changed {
            cl.needs_rebuild = true;
        }
        Ok(n_cells_total_changed)
    }

    // rq-1f38d78a
    /// Queue the per-step displacement-check kernel on the device's
    /// default stream. One thread per atom computes the min-image
    /// displacement from the rebuild-time reference position and sets
    /// `disp_rebuild_flag = 1u` via `atomicOr` when the squared length
    /// exceeds `threshold_sq = (r_skin / 2)²`. The flag is sticky:
    /// it is otherwise cleared only by `pre_step` after a rebuild.
    /// Called as the last device-visible action of every physical
    /// step from `ForceField::step` and
    /// `ForceField::step_no_neighbor_check`, so the launch sits inside
    /// any captured graph that includes the per-step force sequence.
    pub fn enqueue_displacement_check(
        &mut self,
        sim_box: &SimulationBox,
        buffers: &ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<(), NeighborListError> {
        if self.particle_count == 0 {
            return Ok(());
        }
        // Disjoint field borrows: reference positions live in `self.mode`
        // (CellListData); the status word lives in `self.packed`.
        let cl = match &self.mode {
            NeighborListMode::Trivial => return Ok(()),
            // Bin-only mode rebuilds unconditionally each pre_step; no
            // displacement check is queued.
            NeighborListMode::CellListOnly(_) => return Ok(()),
            NeighborListMode::CellList(cl) => cl,
        };
        let threshold_sq = cl.threshold_sq;
        let (ref_x, ref_y, ref_z) = (
            &cl.reference_positions_x,
            &cl.reference_positions_y,
            &cl.reference_positions_z,
        );
        let status = match self.packed.as_mut() {
            Some(p) => &mut p.neighbor_status,
            None => return Ok(()),
        };
        timings
            .kernel_start(KernelStage::NEIGHBOR_DISPLACEMENT_SQUARED)
            .map_err(map_timings_err)?;
        // rq-1f38d78a — sets bit 0 of the shared status word via atomicOr.
        neighbor_displacement_check_flag(
            buffers,
            ref_x,
            ref_y,
            ref_z,
            sim_box,
            threshold_sq,
            status,
        )?;
        timings
            .kernel_stop(KernelStage::NEIGHBOR_DISPLACEMENT_SQUARED)
            .map_err(map_timings_err)?;
        Ok(())
    }

    /// Read the combined `neighbor_status` word with a single 4-byte
    /// `dtoh_sync_copy`. Returns `0` in `Trivial` / `CellListOnly` modes
    /// (no status word is consulted). This is the only device-to-host
    /// transfer the neighbour path performs per batch boundary.
    /// rq-1f38d78a rq-67a09135
    fn read_neighbor_status(&self) -> Result<u32, NeighborListError> {
        let status_buf = match (&self.mode, self.packed.as_ref()) {
            (NeighborListMode::CellList(_), Some(p)) => &p.neighbor_status,
            _ => return Ok(0),
        };
        let host: Vec<u32> = self
            .device
            .dtoh_sync_copy(status_buf)
            .map_err(GpuError::from)?;
        Ok(host[0])
    }

    // rq-c49b2fe6
    /// Host-side consumer of bit 0 of `neighbor_status`: issues a single
    /// 4-byte `dtoh_sync_copy` against the word, returning `true` if the
    /// displacement-check kernel has signalled at any captured step since
    /// the last rebuild that an atom's displacement exceeded `r_skin / 2`.
    pub fn displacement_check(
        &mut self,
        _sim_box: &SimulationBox,
        _buffers: &ParticleBuffers,
        _timings: &mut Timings,
    ) -> Result<bool, NeighborListError> {
        if self.particle_count == 0 {
            return Ok(false);
        }
        Ok((self.read_neighbor_status()? & STATUS_DISPLACEMENT_TRIPPED) != 0)
    }

    // rq-7db97132
    // Returns `true` when a packed-neighbour buffer was reallocated
    // during the rebuild (so a captured CUDA graph holding stale device
    // pointers must be re-captured). rq-7db97132
    pub fn rebuild(
        &mut self,
        sim_box: &SimulationBox,
        buffers: &ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<bool, NeighborListError> {
        if self.particle_count == 0 {
            match &mut self.mode {
                NeighborListMode::CellList(cl) | NeighborListMode::CellListOnly(cl) => {
                    cl.needs_rebuild = false;
                }
                NeighborListMode::Trivial => {}
            }
            return Ok(false);
        }
        if matches!(self.mode, NeighborListMode::Trivial) {
            return Ok(false);
        }
        let started = Instant::now();
        // rq-67a09135 rq-1f38d78a — Zero the combined status word at the
        // start of the rebuild so the construction kernel's high-water /
        // overflow bits (and the next batch's displacement bit) start
        // clean. The first rebuild of the state runs the synchronous
        // sizing probe; every later rebuild is dtoh-free.
        let probe = !self.has_probed;
        if let Some(p) = self.packed.as_mut() {
            self.device
                .memset_zeros(&mut p.neighbor_status)
                .map_err(GpuError::from)?;
        }
        let result = self.rebuild_impl(sim_box, buffers, timings, probe);
        if result.is_ok() {
            self.has_probed = true;
        }
        timings.record_host(HostStage::NEIGHBOR_LIST_REBUILD, started.elapsed());
        result
    }

    fn rebuild_impl(
        &mut self,
        sim_box: &SimulationBox,
        buffers: &ParticleBuffers,
        timings: &mut Timings,
        probe: bool,
    ) -> Result<bool, NeighborListError> {
        let device = self.device.clone();
        let kernels = self.kernels.clone();
        let particle_count = self.particle_count;
        let bin_only = matches!(self.mode, NeighborListMode::CellListOnly(_));

        // Pull out parameters we need outside the cell-list borrow.
        let r_search_sq = match &self.mode {
            NeighborListMode::Trivial => return Ok(false),
            NeighborListMode::CellList(cl) | NeighborListMode::CellListOnly(cl) => cl.r_search_sq,
        };

        // Cell-list pre-step.
        {
            let cl = self.cell_list_data_mut().expect("non-Trivial mode");
            compute_cell_indices_and_histogram(
                buffers,
                sim_box,
                cl.n_cells,
                &mut cl.cell_indices,
                &mut cl.cell_counts,
            )?;

            prefix_scan_cell_counts(
                &kernels,
                &cl.cell_counts,
                &mut cl.cell_offsets,
                &mut cl.scan_block_totals,
                cl.n_cells_total,
                crate::gpu::PrefixScanSentinel::Host(particle_count as u32),
            )?;

            scatter_atoms_into_cells(
                &device,
                &kernels,
                &cl.cell_indices,
                &cl.cell_offsets,
                &mut cl.write_cursors,
                &mut cl.sorted_particle_ids,
                particle_count,
            )?;

            sort_cells_by_particle_id(
                &kernels,
                &cl.cell_offsets,
                &mut cl.sorted_particle_ids,
                cl.n_cells_total,
            )?;
        }

        if bin_only {
            let cl = self.cell_list_data_mut().expect("CellListOnly");
            cl.needs_rebuild = false;
            self.rebuild_generation = self.rebuild_generation.wrapping_add(1);
            return Ok(false);
        }

        // Packed-neighbour construction. `reallocated` is `true` when a
        // packed buffer grew during this rebuild (probe path only — a
        // steady-state rebuild never grows; growth happens in `pre_step`
        // before the rebuild). rq-67a09135
        let reallocated = self.rebuild_packed_neighbour(buffers, sim_box, r_search_sq, probe)?;

        {
            let cl = self.cell_list_data_mut().expect("non-Trivial mode");
            timings
                .kernel_start(KernelStage::COPY_POSITIONS_INTO_REFERENCE)
                .map_err(map_timings_err)?;
            // The reference positions are refreshed so the next batch's
            // displacement check measures from this rebuild's positions.
            // The status word (including bit 0) was already zeroed at the
            // start of the rebuild, so no end-of-rebuild reset is needed.
            copy_positions_into_reference(
                buffers,
                &mut cl.reference_positions_x,
                &mut cl.reference_positions_y,
                &mut cl.reference_positions_z,
            )?;
            timings
                .kernel_stop(KernelStage::COPY_POSITIONS_INTO_REFERENCE)
                .map_err(map_timings_err)?;

            cl.needs_rebuild = false;
        }
        self.rebuild_generation = self.rebuild_generation.wrapping_add(1);
        Ok(reallocated)
    }

    /// Packed-neighbour construction pipeline (see
    /// `rqm/forces/packed-neighbour-pair-force.md`). Called from
    /// `rebuild_impl` after the cell-list sort completes.
    fn rebuild_packed_neighbour(
        &mut self,
        buffers: &ParticleBuffers,
        sim_box: &SimulationBox,
        r_search_sq: Real,
        probe: bool,
    ) -> Result<bool, NeighborListError> {
        let device = self.device.clone();
        let kernels = self.kernels.clone();
        let particle_count = self.particle_count;
        if particle_count == 0 {
            return Ok(false);
        }
        let n_blocks = self
            .packed
            .as_ref()
            .map(|p| p.n_blocks)
            .unwrap_or(0);
        if n_blocks == 0 {
            return Ok(false);
        }

        // Split borrow: cell-list's sorted_particle_ids (immutable) and
        // self.packed (mutable) live on disjoint fields of self.
        let sorted_view: *const CudaSlice<u32> = match &self.mode {
            NeighborListMode::CellList(cl) => &cl.sorted_particle_ids,
            _ => unreachable!("rebuild_packed_neighbour is for CellList only"),
        };
        let packed = self.packed.as_mut().expect("packed data present");

        // 1. Scatter positions into tile-sorted view (block order).
        crate::gpu::scatter_positions_to_tile_order(
            &kernels,
            buffers,
            unsafe { &*sorted_view },
            &mut packed.tile_sorted_posq,
        )?;

        // 2. Fill partial-block padding lanes with +infinity so they
        //    trivially fail every distance check.
        let padded_n = n_blocks * 32;
        crate::gpu::fill_tile_position_padding(
            &kernels,
            &mut packed.tile_sorted_posq,
            particle_count as u32,
            padded_n,
        )?;

        // 3. Per-block bounding boxes.
        crate::gpu::compute_block_bbox(
            &kernels,
            &packed.tile_sorted_posq,
            &packed.tile_atom_count,
            &mut packed.block_centre,
            &mut packed.block_bbox,
            n_blocks,
        )?;

        // 4. Find blocks with interactions, then record the high-water /
        //    overflow state in `neighbor_status` from the device-resident
        //    counts. No interaction count is copied to the host.
        //
        //    Steady-state (`probe == false`): run exactly once, no dtoh.
        //    Growth, when needed, was already applied by `pre_step` before
        //    this rebuild, so `reallocated` is always `false` here.
        //
        //    Probe (`probe == true`): read `neighbor_status` synchronously
        //    and grow geometrically until neither a high-water nor an
        //    overflow bit is set, sizing capacity with headroom. Runs once
        //    per state, before CUDA-graph capture.
        //    rq-67a09135
        let mut reallocated = false;
        loop {
            device
                .memset_zeros(&mut packed.interaction_count)
                .map_err(GpuError::from)?;
            if probe {
                // A retry's set_neighbor_status_bits must start from a
                // clean word; the first iteration's zero is redundant with
                // the one `rebuild` issued, which is harmless.
                device
                    .memset_zeros(&mut packed.neighbor_status)
                    .map_err(GpuError::from)?;
            }

            let max_entries = packed.interacting_tiles_capacity;
            let max_single_pairs = packed.single_pairs_capacity;
            crate::gpu::find_blocks_with_interactions(
                &kernels,
                &packed.tile_sorted_posq,
                unsafe { &*sorted_view },
                &packed.block_centre,
                &packed.block_bbox,
                sim_box,
                r_search_sq,
                n_blocks,
                particle_count as u32,
                max_entries,
                max_single_pairs,
                &mut packed.interacting_tiles,
                &mut packed.interacting_atoms,
                &mut packed.single_pair_atoms,
                &mut packed.interaction_count,
            )?;
            // rq-67a09135 — set bits 1-4 of neighbor_status on the device.
            let tiles_hw = packed.tiles_high_water_mark();
            let sp_hw = packed.single_pairs_high_water_mark();
            crate::gpu::set_neighbor_status_bits(
                &kernels,
                &packed.interaction_count,
                packed.interacting_tiles_capacity,
                packed.single_pairs_capacity,
                tiles_hw,
                sp_hw,
                &mut packed.neighbor_status,
            )?;

            if !probe {
                break;
            }
            let status = device
                .dtoh_sync_copy(&packed.neighbor_status)
                .map_err(GpuError::from)?[0];
            let grow_tiles =
                (status & (STATUS_TILES_HIGH_WATER | STATUS_TILES_OVERFLOW)) != 0;
            let grow_sp = (status
                & (STATUS_SINGLE_PAIRS_HIGH_WATER | STATUS_SINGLE_PAIRS_OVERFLOW))
                != 0;
            if !grow_tiles && !grow_sp {
                break;
            }
            if grow_tiles {
                packed.grow_tiles(&device).map_err(NeighborListError::Gpu)?;
                reallocated = true;
            }
            if grow_sp {
                packed
                    .grow_single_pairs(&device)
                    .map_err(NeighborListError::Gpu)?;
                reallocated = true;
            }
        }

        // 5. Sort entries by i-block so the force kernel can process
        //    consecutive same-i-block entries with register carryover
        //    on the i-side accumulator. The scan's trailing sentinel comes
        //    from the device-resident interaction count (rq-67a09135).
        device
            .memset_zeros(&mut packed.iblock_count)
            .map_err(GpuError::from)?;
        crate::gpu::histogram_entries_by_iblock(
            &kernels,
            &packed.interacting_tiles,
            &packed.interaction_count,
            &mut packed.iblock_count,
            n_blocks,
            packed.interacting_tiles_capacity,
        )?;
        crate::gpu::prefix_scan_cell_counts(
            &kernels,
            &packed.iblock_count,
            &mut packed.iblock_offset,
            &mut packed.iblock_scan_block_totals,
            n_blocks as usize,
            crate::gpu::PrefixScanSentinel::Device(&packed.interaction_count),
        )?;
        device
            .memset_zeros(&mut packed.iblock_cursor)
            .map_err(GpuError::from)?;
        crate::gpu::scatter_entries_by_iblock(
            &kernels,
            &packed.interacting_tiles,
            &packed.interacting_atoms,
            &packed.interaction_count,
            &packed.iblock_offset,
            &mut packed.iblock_cursor,
            &mut packed.sorted_interacting_atoms,
            n_blocks,
            packed.interacting_tiles_capacity,
        )?;

        Ok(reallocated)
    }

    // rq-1217c816
    pub fn pre_step(
        &mut self,
        sim_box: &SimulationBox,
        buffers: &ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<PreStepOutcome, NeighborListError> {
        if self.particle_count == 0 {
            return Ok(PreStepOutcome::default());
        }
        if matches!(self.mode, NeighborListMode::Trivial) {
            return Ok(PreStepOutcome::default());
        }

        // CellListOnly mode rebuilds every step unconditionally; the
        // displacement-check + r_skin machinery is bypassed entirely.
        if matches!(self.mode, NeighborListMode::CellListOnly(_)) {
            if let NeighborListMode::CellListOnly(cl) = &mut self.mode {
                cl.needs_rebuild = true;
            }
            let reallocated = self.rebuild(sim_box, buffers, timings)?;
            return Ok(PreStepOutcome {
                rebuilt: true,
                reallocated,
            });
        }

        let n_cells_changed = self.refresh_cell_layout_if_box_changed(sim_box)?;

        let mut rebuild_required = match &self.mode {
            NeighborListMode::Trivial | NeighborListMode::CellListOnly(_) => unreachable!(),
            NeighborListMode::CellList(cl) => cl.needs_rebuild,
        };
        // `true` when this call grows a packed buffer (high-water), which
        // — like a probe-time grow — forces a phase-graph re-capture.
        let mut grew = false;

        if !n_cells_changed && !rebuild_required {
            // rq-1f38d78a rq-67a09135 rq-2dda3169
            // Single-word `dtoh_sync_copy` of the combined status word —
            // the only device-to-host transfer the neighbour path performs
            // per batch. It surfaces the displacement (bit 0), high-water
            // (bits 1-2), and overflow (bits 3-4) signals together; the
            // rebuild itself copies nothing.
            let status = self.read_neighbor_status()?;
            // Overflow: a build dropped within-`r_search` entries, so the
            // no-silent-drop guarantee is violated. Halt.
            if (status & STATUS_TILES_OVERFLOW) != 0 {
                return Err(NeighborListError::PackedNeighborOverflow {
                    buffer: "interacting_tiles",
                });
            }
            if (status & STATUS_SINGLE_PAIRS_OVERFLOW) != 0 {
                return Err(NeighborListError::PackedNeighborOverflow {
                    buffer: "single_pair_atoms",
                });
            }
            // High-water: the build came within `tile_pair_fill_threshold`
            // of capacity while dropping nothing. Grow geometrically before
            // the rebuild so the resized buffers are populated this call.
            if (status & STATUS_TILES_HIGH_WATER) != 0 {
                if let Some(p) = self.packed.as_mut() {
                    p.grow_tiles(&self.device).map_err(NeighborListError::Gpu)?;
                    grew = true;
                }
                rebuild_required = true;
            }
            if (status & STATUS_SINGLE_PAIRS_HIGH_WATER) != 0 {
                if let Some(p) = self.packed.as_mut() {
                    p.grow_single_pairs(&self.device)
                        .map_err(NeighborListError::Gpu)?;
                    grew = true;
                }
                rebuild_required = true;
            }
            if (status & STATUS_DISPLACEMENT_TRIPPED) != 0 {
                rebuild_required = true;
            }
        }
        if rebuild_required {
            if let NeighborListMode::CellList(cl) = &mut self.mode {
                cl.needs_rebuild = true;
            }
            let reallocated = self.rebuild(sim_box, buffers, timings)?;
            return Ok(PreStepOutcome {
                rebuilt: true,
                reallocated: reallocated || grew,
            });
        }
        Ok(PreStepOutcome::default())
    }
}

// rq-dfad7218
//
// n_cells_d = floor(w_d / (r_cut + r_skin)) per lattice direction
// d ∈ {a, b, c}, where w_d is the box's perpendicular width along that
// direction (see simulation-box.md). Rejects with BoxTooSmallForCells if
// any direction admits fewer than 3 cells.
fn compute_cell_layout(
    sim_box: &SimulationBox,
    r_search: Real,
) -> Result<[u32; 3], NeighborListError> {
    let widths = sim_box.perpendicular_widths();
    let direction_names: [&'static str; 3] = ["a", "b", "c"];
    let mut n_cells = [0u32; 3];
    for d in 0..3 {
        let w = widths[d];
        let nc = (w / r_search).floor() as i64;
        if nc < 3 {
            return Err(NeighborListError::BoxTooSmallForCells {
                direction: direction_names[d],
                width: w,
                required: 3.0 * r_search,
            });
        }
        n_cells[d] = nc as u32;
    }
    Ok(n_cells)
}

// rq-d8e4407a
//
// The device addresses cells with a `u32` cell index (the cell-index
// arithmetic and every scan kernel argument are `u32`), so the cell
// grid may hold at most `u32::MAX` cells. Checked before any device
// buffer is allocated so an over-fine grid fails loud rather than
// overflowing the cell-index arithmetic.
fn check_n_cells_total(n_cells_total: usize) -> Result<(), NeighborListError> {
    let max_supported = u32::MAX as usize;
    if n_cells_total > max_supported {
        return Err(NeighborListError::TooManyCells {
            n_cells_total,
            max_supported,
        });
    }
    Ok(())
}

// rq-a060036e
//
// Lengths of the recursive prefix scan's block-totals stack for a grid
// of `n_cells_total` cells: level 0 holds `ceil(n_cells_total / B)`
// per-block totals, each subsequent level holds `ceil(prev / B)`, and
// the stack ends with the first level of length 1. Every
// `prefix_scan_local_blocks` call needs a block-totals output buffer,
// including the terminal single-block one, so the last length is 1.
pub(crate) fn scan_stack_lengths(n_cells_total: usize) -> Vec<usize> {
    let block = SPATIAL_HASH_SCAN_BLOCK_SIZE as usize;
    let mut lengths = Vec::new();
    let mut len = n_cells_total;
    loop {
        let blocks = len.div_ceil(block);
        lengths.push(blocks);
        if blocks <= 1 {
            break;
        }
        len = blocks;
    }
    lengths
}

pub(crate) fn alloc_scan_block_totals(
    device: &Arc<CudaDevice>,
    n_cells_total: usize,
) -> Result<Vec<CudaSlice<u32>>, NeighborListError> {
    scan_stack_lengths(n_cells_total)
        .into_iter()
        .map(|len| {
            device
                .alloc_zeros::<u32>(len.max(1))
                .map_err(|e| NeighborListError::Gpu(GpuError::from(e)))
        })
        .collect()
}

fn map_timings_err(e: crate::timings::TimingsError) -> NeighborListError {
    match e {
        crate::timings::TimingsError::Gpu(g) => NeighborListError::Gpu(g),
    }
}

// rq-2093594f rq-0469400b rq-67a09135
crate::gpu_kernels! {
    module: "neighbor",
    ptx: crate::kernels::NEIGHBOR,
    struct: NeighborKernels,
    kernels: [
        neighbor_displacement_check_flag,
        copy_positions_into_reference,
        compute_cell_indices_and_histogram,
        prefix_scan_local_blocks,
        prefix_scan_apply_block_totals,
        prefix_scan_finalize_offsets,
        prefix_scan_finalize_offsets_dev,
        scatter_atoms_into_cells,
        sort_cells_by_particle_id,
        scatter_positions_to_tile_order,
        fill_tile_position_padding,
        compute_block_bbox,
        find_blocks_with_interactions,
        set_neighbor_status_bits,
        finalize_packed_forces,
        histogram_entries_by_iblock,
        scatter_entries_by_iblock,
    ],
    stages: {
        NEIGHBOR_DISPLACEMENT_SQUARED   = "neighbor_displacement_check_flag",
        COPY_POSITIONS_INTO_REFERENCE   = "copy_positions_into_reference",
        SCATTER_POSITIONS_TO_TILE_ORDER = "scatter_positions_to_tile_order",
        FINALIZE_PACKED_FORCES          = "finalize_packed_forces",
    },
}
