// rq-693ce6fa rq-b2d68288
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice};
use cudarc::nvrtc::Ptx;

use crate::gpu::device::get_func;
use crate::gpu::{
    GpuContext, GpuError, Kernels, ParticleBuffers, SPATIAL_HASH_SCAN_BLOCK_SIZE,
    compute_cell_indices_and_histogram, copy_positions_into_reference,
    neighbor_displacement_squared, neighbor_list_build, prefix_scan_cell_counts,
    scatter_atoms_into_cells, sort_cells_by_particle_id,
};
use crate::kernels;
use crate::pbc::SimulationBox;
use crate::timings::{HostStage, KernelStage, Timings};
use crate::precision::Real;

// rq-d8e4407a rq-e1ceb5c0 rq-6cf916af rq-1bbcf3b7
#[derive(Debug, thiserror::Error)]
pub enum NeighborListError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("an atom has more than {max} neighbors")]
    NeighborListOverflow { max: u32 },
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
    pub disp_sq: CudaSlice<Real>,
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
    /// Block-order positions (refreshed every step by
    /// `scatter_positions_to_tile_order`). Length `n_blocks * 32`.
    pub tile_sorted_positions_x: CudaSlice<Real>,
    pub tile_sorted_positions_y: CudaSlice<Real>,
    pub tile_sorted_positions_z: CudaSlice<Real>,
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
    /// single_pairs_count]`.
    pub interaction_count: CudaSlice<u32>,
    /// Set non-zero by the construction kernel when capacity overflows.
    pub overflow_flag: CudaSlice<u32>,
    /// Current allocated capacity.
    pub interacting_tiles_capacity: u32,
    /// Live entry count after the most recent rebuild.
    pub interacting_tiles_count: u32,
    /// Multiplier applied to the capacity on overflow.
    pub tile_pair_growth_factor: f64,
}

// rq-b2d68288
#[derive(Debug)]
pub struct NeighborListState {
    pub device: Arc<CudaDevice>,
    pub kernels: Arc<Kernels>,
    pub particle_count: usize,
    pub max_neighbors: u32,
    pub neighbor_list: CudaSlice<u32>,
    pub neighbor_counts: CudaSlice<u32>,
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
}

/// Default initial capacity for `interacting_tiles`. Sized to the
/// all-pairs upper bound `n_blocks^2`, so a first rebuild never needs
/// to grow the buffer. Memory cost at N=24576: 768^2 = 589,824 entries
/// = ~2.4 MB for tiles, 75 MB for atoms — acceptable.
pub fn default_interacting_tiles_capacity(n_blocks: u32) -> u32 {
    if n_blocks == 0 {
        return 1;
    }
    let sq = (n_blocks as u64).saturating_mul(n_blocks as u64);
    sq.min(u32::MAX as u64) as u32
}

fn alloc_packed_neighbor_data(
    device: &Arc<CudaDevice>,
    particle_count: usize,
    interacting_tiles_capacity: u32,
    tile_pair_growth_factor: f64,
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
    let tile_sorted_positions_x = device
        .alloc_zeros::<Real>(padded_n)
        .map_err(GpuError::from)?;
    let tile_sorted_positions_y = device
        .alloc_zeros::<Real>(padded_n)
        .map_err(GpuError::from)?;
    let tile_sorted_positions_z = device
        .alloc_zeros::<Real>(padded_n)
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
    let overflow_flag = device.alloc_zeros::<u32>(1).map_err(GpuError::from)?;

    Ok(PackedNeighborData {
        n_blocks,
        tile_atom_count,
        tile_lane_mask,
        tile_sorted_positions_x,
        tile_sorted_positions_y,
        tile_sorted_positions_z,
        block_centre,
        block_bbox,
        trivial_sorted_particle_ids,
        interacting_tiles,
        interacting_atoms,
        interaction_count,
        overflow_flag,
        interacting_tiles_capacity: cap,
        interacting_tiles_count: 0,
        tile_pair_growth_factor,
    })
}

impl PackedNeighborData {
    /// Grow the entry-list buffers to at least `required` entries.
    pub fn grow_to(
        &mut self,
        device: &Arc<CudaDevice>,
        required: u32,
    ) -> Result<(), GpuError> {
        let new_cap_f = (required as f64) * self.tile_pair_growth_factor;
        let new_cap = (new_cap_f.ceil() as u32).max(required).max(1);
        let new_alloc = new_cap as usize;
        self.interacting_tiles = device.alloc_zeros::<u32>(new_alloc)?;
        self.interacting_atoms = device.alloc_zeros::<u32>(new_alloc * 32)?;
        self.interacting_tiles_capacity = new_cap;
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
        max_neighbors: u32,
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
        debug_assert!(max_neighbors > 0);
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
        let nl_len = particle_count * max_neighbors as usize;
        let neighbor_list = device
            .alloc_zeros::<u32>(nl_len.max(1))
            .map_err(GpuError::from)?;
        let neighbor_counts = device
            .alloc_zeros::<u32>(particle_count.max(1))
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
        let disp_sq = device
            .alloc_zeros::<Real>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let overflow_flag = device.alloc_zeros::<u32>(1).map_err(GpuError::from)?;

        let n_blocks = ((particle_count as u32) + 31) / 32;
        let initial_cap = default_interacting_tiles_capacity(n_blocks);
        let packed = alloc_packed_neighbor_data(
            &device,
            particle_count,
            initial_cap,
            1.5,
            false, // CellList mode uses cell-list's sorted_particle_ids
        )?;

        Ok(NeighborListState {
            device,
            kernels,
            particle_count,
            max_neighbors,
            neighbor_list,
            neighbor_counts,
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
                disp_sq,
                overflow_flag,
                needs_rebuild: true,
            }),
            rebuild_generation: 0,
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

        // Neighbor-list-only buffers are zero-length in bin-only mode.
        let neighbor_list = device.alloc_zeros::<u32>(0).map_err(GpuError::from)?;
        let neighbor_counts = device.alloc_zeros::<u32>(0).map_err(GpuError::from)?;
        let reference_positions_x = device.alloc_zeros::<Real>(0).map_err(GpuError::from)?;
        let reference_positions_y = device.alloc_zeros::<Real>(0).map_err(GpuError::from)?;
        let reference_positions_z = device.alloc_zeros::<Real>(0).map_err(GpuError::from)?;
        let disp_sq = device.alloc_zeros::<Real>(0).map_err(GpuError::from)?;
        let overflow_flag = device.alloc_zeros::<u32>(0).map_err(GpuError::from)?;

        Ok(NeighborListState {
            device,
            kernels,
            particle_count,
            max_neighbors: 0,
            neighbor_list,
            neighbor_counts,
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
                disp_sq,
                overflow_flag,
                needs_rebuild: true,
            }),
            rebuild_generation: 0,
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
        let max_neighbors = particle_count as u32;
        let nl_len = particle_count * particle_count;
        let nl_host: Vec<u32> = if nl_len == 0 {
            Vec::new()
        } else {
            let mut v = Vec::with_capacity(nl_len);
            for _ in 0..particle_count {
                for k in 0..particle_count {
                    v.push(k as u32);
                }
            }
            v
        };
        let counts_host: Vec<u32> = vec![particle_count as u32; particle_count];

        let neighbor_list = if nl_host.is_empty() {
            device.alloc_zeros::<u32>(0).map_err(GpuError::from)?
        } else {
            device.htod_sync_copy(&nl_host).map_err(GpuError::from)?
        };
        let neighbor_counts = if counts_host.is_empty() {
            device.alloc_zeros::<u32>(0).map_err(GpuError::from)?
        } else {
            device.htod_sync_copy(&counts_host).map_err(GpuError::from)?
        };

        let packed = if particle_count == 0 {
            None
        } else {
            let n_blocks = ((particle_count as u32) + 31) / 32;
            // Trivial mode: every i-block is interacting with every
            // j-block (including itself). Each i-block sees ceil(N/32)
            // packed entries, each containing up to 32 j-atoms drawn
            // from j-blocks j_block = (entry / 1) (one j-block per
            // entry; we just pack atom IDs sequentially).
            let entries_per_block = n_blocks;
            let total_entries = (entries_per_block as u64) * (n_blocks as u64);
            let cap = total_entries.min(u32::MAX as u64).max(1) as u32;
            let mut packed = alloc_packed_neighbor_data(
                &device,
                particle_count,
                cap,
                1.5,
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
            }

            // Populate the tile-sorted positions view's padding lanes
            // with +inf so the force kernel treats them as inactive.
            let pos_inf = (3.4e38 as Real, 3.4e38 as Real, 3.4e38 as Real);
            let padded_n = (n_blocks as usize) * 32;
            if padded_n > particle_count {
                let pad_count = padded_n - particle_count;
                let pad_x = vec![pos_inf.0; pad_count];
                let pad_y = vec![pos_inf.1; pad_count];
                let pad_z = vec![pos_inf.2; pad_count];
                let mut view_x = packed.tile_sorted_positions_x.slice_mut(particle_count..);
                let mut view_y = packed.tile_sorted_positions_y.slice_mut(particle_count..);
                let mut view_z = packed.tile_sorted_positions_z.slice_mut(particle_count..);
                device.htod_sync_copy_into(&pad_x, &mut view_x).map_err(GpuError::from)?;
                device.htod_sync_copy_into(&pad_y, &mut view_y).map_err(GpuError::from)?;
                device.htod_sync_copy_into(&pad_z, &mut view_z).map_err(GpuError::from)?;
            }
            Some(packed)
        };

        Ok(NeighborListState {
            device,
            kernels,
            particle_count,
            max_neighbors,
            neighbor_list,
            neighbor_counts,
            packed,
            mode: NeighborListMode::Trivial,
            rebuild_generation: 0,
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

    // rq-c49b2fe6
    pub fn displacement_check(
        &mut self,
        sim_box: &SimulationBox,
        buffers: &ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<Real, NeighborListError> {
        if self.particle_count == 0 {
            return Ok(0.0);
        }
        let cl = match &mut self.mode {
            NeighborListMode::Trivial => return Ok(0.0),
            // Bin-only mode has no displacement check; rebuild every step.
            NeighborListMode::CellListOnly(_) => return Ok(0.0),
            NeighborListMode::CellList(cl) => cl,
        };
        timings
            .kernel_start(KernelStage::NEIGHBOR_DISPLACEMENT_SQUARED)
            .map_err(map_timings_err)?;
        neighbor_displacement_squared(
            buffers,
            &cl.reference_positions_x,
            &cl.reference_positions_y,
            &cl.reference_positions_z,
            sim_box,
            &mut cl.disp_sq,
        )?;
        timings
            .kernel_stop(KernelStage::NEIGHBOR_DISPLACEMENT_SQUARED)
            .map_err(map_timings_err)?;

        let host: Vec<Real> = self
            .device
            .dtoh_sync_copy(&cl.disp_sq)
            .map_err(GpuError::from)?;
        let mut max_sq: f64 = 0.0;
        for &v in host[..self.particle_count].iter() {
            let v64 = v as f64;
            if v64 > max_sq {
                max_sq = v64;
            }
        }
        Ok(max_sq.sqrt() as Real)
    }

    // rq-7db97132
    pub fn rebuild(
        &mut self,
        sim_box: &SimulationBox,
        buffers: &ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<(), NeighborListError> {
        if self.particle_count == 0 {
            match &mut self.mode {
                NeighborListMode::CellList(cl) | NeighborListMode::CellListOnly(cl) => {
                    cl.needs_rebuild = false;
                }
                NeighborListMode::Trivial => {}
            }
            return Ok(());
        }
        if matches!(self.mode, NeighborListMode::Trivial) {
            return Ok(());
        }
        let started = Instant::now();
        let result = self.rebuild_impl(sim_box, buffers, timings);
        timings.record_host(HostStage::NEIGHBOR_LIST_REBUILD, started.elapsed());
        result
    }

    fn rebuild_impl(
        &mut self,
        sim_box: &SimulationBox,
        buffers: &ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<(), NeighborListError> {
        let device = self.device.clone();
        let kernels = self.kernels.clone();
        let particle_count = self.particle_count;
        let bin_only = matches!(self.mode, NeighborListMode::CellListOnly(_));

        // Pull out parameters we need outside the cell-list borrow.
        let r_search_sq = match &self.mode {
            NeighborListMode::Trivial => return Ok(()),
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
                particle_count,
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
            return Ok(());
        }

        // Packed-neighbour construction.
        self.rebuild_packed_neighbour(buffers, sim_box, r_search_sq)?;

        {
            let cl = self.cell_list_data_mut().expect("non-Trivial mode");
            timings
                .kernel_start(KernelStage::COPY_POSITIONS_INTO_REFERENCE)
                .map_err(map_timings_err)?;
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
        Ok(())
    }

    /// Packed-neighbour construction pipeline (see
    /// `rqm/forces/packed-neighbour-pair-force.md`). Called from
    /// `rebuild_impl` after the cell-list sort completes.
    fn rebuild_packed_neighbour(
        &mut self,
        buffers: &ParticleBuffers,
        sim_box: &SimulationBox,
        r_search_sq: Real,
    ) -> Result<(), NeighborListError> {
        let device = self.device.clone();
        let kernels = self.kernels.clone();
        let particle_count = self.particle_count;
        if particle_count == 0 {
            return Ok(());
        }
        let n_blocks = self
            .packed
            .as_ref()
            .map(|p| p.n_blocks)
            .unwrap_or(0);
        if n_blocks == 0 {
            return Ok(());
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
            &mut packed.tile_sorted_positions_x,
            &mut packed.tile_sorted_positions_y,
            &mut packed.tile_sorted_positions_z,
        )?;

        // 2. Fill partial-block padding lanes with +infinity so they
        //    trivially fail every distance check.
        let padded_n = n_blocks * 32;
        crate::gpu::fill_tile_position_padding(
            &kernels,
            &mut packed.tile_sorted_positions_x,
            &mut packed.tile_sorted_positions_y,
            &mut packed.tile_sorted_positions_z,
            particle_count as u32,
            padded_n,
        )?;

        // 3. Per-block bounding boxes.
        crate::gpu::compute_block_bbox(
            &kernels,
            &packed.tile_sorted_positions_x,
            &packed.tile_sorted_positions_y,
            &packed.tile_sorted_positions_z,
            &packed.tile_atom_count,
            &mut packed.block_centre,
            &mut packed.block_bbox,
            n_blocks,
        )?;

        // 4. Find blocks with interactions (with grow+retry on overflow).
        loop {
            // Zero counters and overflow flag.
            device
                .memset_zeros(&mut packed.interaction_count)
                .map_err(GpuError::from)?;
            device
                .memset_zeros(&mut packed.overflow_flag)
                .map_err(GpuError::from)?;

            let max_entries = packed.interacting_tiles_capacity;
            crate::gpu::find_blocks_with_interactions(
                &kernels,
                &packed.tile_sorted_positions_x,
                &packed.tile_sorted_positions_y,
                &packed.tile_sorted_positions_z,
                unsafe { &*sorted_view },
                &packed.block_centre,
                &packed.block_bbox,
                sim_box,
                r_search_sq,
                n_blocks,
                particle_count as u32,
                max_entries,
                &mut packed.interacting_tiles,
                &mut packed.interacting_atoms,
                &mut packed.interaction_count,
                &mut packed.overflow_flag,
            )?;

            let flag: Vec<u32> = device
                .dtoh_sync_copy(&packed.overflow_flag)
                .map_err(GpuError::from)?;
            let counts: Vec<u32> = device
                .dtoh_sync_copy(&packed.interaction_count)
                .map_err(GpuError::from)?;
            if flag[0] == 0 {
                packed.interacting_tiles_count = counts[0];
                break;
            }
            // Grow capacity and retry.
            let required = counts[0].max(packed.interacting_tiles_capacity + 1);
            packed.grow_to(&device, required).map_err(NeighborListError::Gpu)?;
        }

        Ok(())
    }

    // rq-1217c816
    pub fn pre_step(
        &mut self,
        sim_box: &SimulationBox,
        buffers: &ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<(), NeighborListError> {
        if self.particle_count == 0 {
            return Ok(());
        }
        if matches!(self.mode, NeighborListMode::Trivial) {
            return Ok(());
        }

        // CellListOnly mode rebuilds every step unconditionally; the
        // displacement-check + r_skin machinery is bypassed entirely.
        if matches!(self.mode, NeighborListMode::CellListOnly(_)) {
            if let NeighborListMode::CellListOnly(cl) = &mut self.mode {
                cl.needs_rebuild = true;
            }
            return self.rebuild(sim_box, buffers, timings);
        }

        let refreshed = self.refresh_cell_layout_if_box_changed(sim_box)?;

        let (mut rebuild_required, r_skin) = match &self.mode {
            NeighborListMode::Trivial | NeighborListMode::CellListOnly(_) => unreachable!(),
            NeighborListMode::CellList(cl) => (cl.needs_rebuild, cl.r_skin),
        };

        if !refreshed && !rebuild_required {
            let max_disp = self.displacement_check(sim_box, buffers, timings)?;
            if max_disp > r_skin * 0.5 {
                rebuild_required = true;
            }
        }
        if rebuild_required {
            if let NeighborListMode::CellList(cl) = &mut self.mode {
                cl.needs_rebuild = true;
            }
            self.rebuild(sim_box, buffers, timings)?;
        }
        Ok(())
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

// rq-2093594f rq-0469400b
#[derive(Debug, Clone)]
pub struct NeighborKernels {
    pub neighbor_displacement_squared: CudaFunction,
    pub neighbor_list_build: CudaFunction,
    pub copy_positions_into_reference: CudaFunction,
    pub compute_cell_indices_and_histogram: CudaFunction,
    pub prefix_scan_local_blocks: CudaFunction,
    pub prefix_scan_apply_block_totals: CudaFunction,
    pub prefix_scan_finalize_offsets: CudaFunction,
    pub scatter_atoms_into_cells: CudaFunction,
    pub sort_cells_by_particle_id: CudaFunction,
    pub scatter_positions_to_tile_order: CudaFunction,
    pub fill_tile_position_padding: CudaFunction,
    pub compute_block_bbox: CudaFunction,
    pub find_blocks_with_interactions: CudaFunction,
    pub finalize_packed_forces: CudaFunction,
}

impl NeighborKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::NEIGHBOR),
            "neighbor",
            &[
                "neighbor_displacement_squared",
                "neighbor_list_build",
                "copy_positions_into_reference",
                "compute_cell_indices_and_histogram",
                "prefix_scan_local_blocks",
                "prefix_scan_apply_block_totals",
                "prefix_scan_finalize_offsets",
                "scatter_atoms_into_cells",
                "sort_cells_by_particle_id",
                "scatter_positions_to_tile_order",
                "fill_tile_position_padding",
                "compute_block_bbox",
                "find_blocks_with_interactions",
                "finalize_packed_forces",
            ],
        )?;
        Ok(NeighborKernels {
            neighbor_displacement_squared: get_func(
                device,
                "neighbor",
                "neighbor_displacement_squared",
            )?,
            neighbor_list_build: get_func(device, "neighbor", "neighbor_list_build")?,
            copy_positions_into_reference: get_func(
                device,
                "neighbor",
                "copy_positions_into_reference",
            )?,
            compute_cell_indices_and_histogram: get_func(
                device,
                "neighbor",
                "compute_cell_indices_and_histogram",
            )?,
            prefix_scan_local_blocks: get_func(device, "neighbor", "prefix_scan_local_blocks")?,
            prefix_scan_apply_block_totals: get_func(
                device,
                "neighbor",
                "prefix_scan_apply_block_totals",
            )?,
            prefix_scan_finalize_offsets: get_func(
                device,
                "neighbor",
                "prefix_scan_finalize_offsets",
            )?,
            scatter_atoms_into_cells: get_func(device, "neighbor", "scatter_atoms_into_cells")?,
            sort_cells_by_particle_id: get_func(device, "neighbor", "sort_cells_by_particle_id")?,
            scatter_positions_to_tile_order: get_func(
                device,
                "neighbor",
                "scatter_positions_to_tile_order",
            )?,
            fill_tile_position_padding: get_func(
                device,
                "neighbor",
                "fill_tile_position_padding",
            )?,
            compute_block_bbox: get_func(device, "neighbor", "compute_block_bbox")?,
            find_blocks_with_interactions: get_func(
                device,
                "neighbor",
                "find_blocks_with_interactions",
            )?,
            finalize_packed_forces: get_func(device, "neighbor", "finalize_packed_forces")?,
        })
    }
}
