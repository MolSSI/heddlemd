// rq-693ce6fa rq-b2d68288
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaDevice, CudaSlice};

use crate::gpu::{
    GpuError, ParticleBuffers, SPATIAL_HASH_MAX_CELLS, SPATIAL_HASH_SCAN_BLOCK_SIZE,
    compute_cell_indices_and_histogram, copy_positions_into_reference,
    neighbor_displacement_squared, neighbor_list_build, prefix_scan_cell_counts,
    scatter_atoms_into_cells, sort_cells_by_particle_id,
};
use crate::pbc::SimulationBox;
use crate::timings::{HostStage, KernelStage, Timings};

// rq-d8e4407a
#[derive(Debug)]
pub enum NeighborListError {
    Gpu(GpuError),
    NeighborListOverflow {
        max: u32,
    },
    BoxTooSmallForCells {
        axis: &'static str,
        length: f32,
        required: f32,
    },
    TooManyCells {
        n_cells_total: usize,
        max_supported: usize,
    },
}

impl std::fmt::Display for NeighborListError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NeighborListError::Gpu(e) => write!(f, "Gpu({e})"),
            NeighborListError::NeighborListOverflow { max } => {
                write!(f, "NeighborListOverflow {{ max: {max} }}")
            }
            NeighborListError::BoxTooSmallForCells {
                axis,
                length,
                required,
            } => write!(
                f,
                "BoxTooSmallForCells {{ axis: {axis:?}, length: {length}, required: {required} }}"
            ),
            NeighborListError::TooManyCells {
                n_cells_total,
                max_supported,
            } => write!(
                f,
                "TooManyCells {{ n_cells_total: {n_cells_total}, max_supported: {max_supported} }}"
            ),
        }
    }
}

impl std::error::Error for NeighborListError {}

impl From<GpuError> for NeighborListError {
    fn from(e: GpuError) -> Self {
        NeighborListError::Gpu(e)
    }
}

// rq-77754ad1
#[derive(Debug)]
pub enum NeighborListMode {
    Trivial,
    CellList(CellListData),
}

// rq-77754ad1
#[derive(Debug)]
pub struct CellListData {
    pub n_cells: [u32; 3],
    pub cell_size: [f32; 3],
    pub n_cells_total: usize,
    pub r_cut: f32,
    pub r_skin: f32,
    pub r_search_sq: f32,
    pub cached_generation: u64,
    pub cell_indices: CudaSlice<u32>,
    pub cell_counts: CudaSlice<u32>,
    pub write_cursors: CudaSlice<u32>,
    pub scan_block_totals: CudaSlice<u32>,
    pub sorted_particle_ids: CudaSlice<u32>,
    pub cell_offsets: CudaSlice<u32>,
    pub reference_positions_x: CudaSlice<f32>,
    pub reference_positions_y: CudaSlice<f32>,
    pub reference_positions_z: CudaSlice<f32>,
    pub disp_sq: CudaSlice<f32>,
    pub overflow_flag: CudaSlice<u32>,
    pub needs_rebuild: bool,
}

// rq-b2d68288
#[derive(Debug)]
pub struct NeighborListState {
    pub device: Arc<CudaDevice>,
    pub particle_count: usize,
    pub max_neighbors: u32,
    pub neighbor_list: CudaSlice<u32>,
    pub neighbor_counts: CudaSlice<u32>,
    pub mode: NeighborListMode,
}

impl NeighborListState {
    /// Borrow the cell-list-specific data; returns `None` when the state is
    /// in `Trivial` mode.
    pub fn cell_list_data(&self) -> Option<&CellListData> {
        match &self.mode {
            NeighborListMode::Trivial => None,
            NeighborListMode::CellList(cl) => Some(cl),
        }
    }

    /// Mutable borrow of the cell-list-specific data; returns `None` when
    /// the state is in `Trivial` mode.
    pub fn cell_list_data_mut(&mut self) -> Option<&mut CellListData> {
        match &mut self.mode {
            NeighborListMode::Trivial => None,
            NeighborListMode::CellList(cl) => Some(cl),
        }
    }

    // rq-14033af1
    pub fn new_cell_list(
        device: Arc<CudaDevice>,
        sim_box: &SimulationBox,
        particle_count: usize,
        r_cut: f32,
        max_neighbors: u32,
        r_skin: f32,
    ) -> Result<Self, NeighborListError> {
        debug_assert!(r_cut > 0.0);
        debug_assert!(r_skin > 0.0);
        debug_assert!(max_neighbors > 0);
        let r_search = r_cut + r_skin;
        let r_search_sq = r_search * r_search;
        let (n_cells, cell_size) = compute_cell_layout(sim_box, r_search)?;
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
        let scan_block_totals = device
            .alloc_zeros::<u32>(scan_blocks_for(n_cells_total))
            .map_err(GpuError::from)?;
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
            .alloc_zeros::<f32>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let reference_positions_y = device
            .alloc_zeros::<f32>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let reference_positions_z = device
            .alloc_zeros::<f32>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let disp_sq = device
            .alloc_zeros::<f32>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let overflow_flag = device.alloc_zeros::<u32>(1).map_err(GpuError::from)?;

        Ok(NeighborListState {
            device,
            particle_count,
            max_neighbors,
            neighbor_list,
            neighbor_counts,
            mode: NeighborListMode::CellList(CellListData {
                n_cells,
                cell_size,
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
        })
    }

    // rq-c96fd9d2
    pub fn new_trivial(
        device: Arc<CudaDevice>,
        _sim_box: &SimulationBox,
        particle_count: usize,
    ) -> Result<Self, NeighborListError> {
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

        Ok(NeighborListState {
            device,
            particle_count,
            max_neighbors,
            neighbor_list,
            neighbor_counts,
            mode: NeighborListMode::Trivial,
        })
    }

    // rq-282af621
    fn refresh_cell_layout_if_box_changed(
        &mut self,
        sim_box: &SimulationBox,
    ) -> Result<bool, NeighborListError> {
        let device = self.device.clone();
        let cl = match &mut self.mode {
            NeighborListMode::Trivial => return Ok(false),
            NeighborListMode::CellList(cl) => cl,
        };
        if sim_box.generation() == cl.cached_generation {
            return Ok(false);
        }
        let r_search = cl.r_cut + cl.r_skin;
        let (new_n_cells, new_cell_size) = compute_cell_layout(sim_box, r_search)?;
        let new_n_cells_total =
            new_n_cells[0] as usize * new_n_cells[1] as usize * new_n_cells[2] as usize;
        check_n_cells_total(new_n_cells_total)?;
        if new_n_cells_total != cl.n_cells_total {
            cl.cell_offsets = device
                .alloc_zeros::<u32>(new_n_cells_total + 1)
                .map_err(GpuError::from)?;
            cl.cell_counts = device
                .alloc_zeros::<u32>(new_n_cells_total)
                .map_err(GpuError::from)?;
            cl.write_cursors = device
                .alloc_zeros::<u32>(new_n_cells_total)
                .map_err(GpuError::from)?;
            cl.scan_block_totals = device
                .alloc_zeros::<u32>(scan_blocks_for(new_n_cells_total))
                .map_err(GpuError::from)?;
        }
        cl.n_cells = new_n_cells;
        cl.cell_size = new_cell_size;
        cl.n_cells_total = new_n_cells_total;
        cl.cached_generation = sim_box.generation();
        cl.needs_rebuild = true;
        Ok(true)
    }

    // rq-c49b2fe6
    pub fn displacement_check(
        &mut self,
        sim_box: &SimulationBox,
        buffers: &ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<f32, NeighborListError> {
        if self.particle_count == 0 {
            return Ok(0.0);
        }
        let cl = match &mut self.mode {
            NeighborListMode::Trivial => return Ok(0.0),
            NeighborListMode::CellList(cl) => cl,
        };
        timings
            .kernel_start(KernelStage::NeighborDisplacementSquared)
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
            .kernel_stop(KernelStage::NeighborDisplacementSquared)
            .map_err(map_timings_err)?;

        let host: Vec<f32> = self
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
        Ok(max_sq.sqrt() as f32)
    }

    // rq-7db97132
    pub fn rebuild(
        &mut self,
        sim_box: &SimulationBox,
        buffers: &ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<(), NeighborListError> {
        if self.particle_count == 0 {
            if let NeighborListMode::CellList(cl) = &mut self.mode {
                cl.needs_rebuild = false;
            }
            return Ok(());
        }
        if matches!(self.mode, NeighborListMode::Trivial) {
            return Ok(());
        }
        let started = Instant::now();
        let result = self.rebuild_impl(sim_box, buffers, timings);
        timings.record_host(HostStage::NeighborListRebuild, started.elapsed());
        result
    }

    fn rebuild_impl(
        &mut self,
        sim_box: &SimulationBox,
        buffers: &ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<(), NeighborListError> {
        let device = self.device.clone();
        let max_neighbors = self.max_neighbors;
        let particle_count = self.particle_count;

        let cl = match &mut self.mode {
            NeighborListMode::Trivial => return Ok(()),
            NeighborListMode::CellList(cl) => cl,
        };

        compute_cell_indices_and_histogram(
            buffers,
            sim_box,
            cl.n_cells,
            cl.cell_size,
            &mut cl.cell_indices,
            &mut cl.cell_counts,
        )?;

        prefix_scan_cell_counts(
            &device,
            &cl.cell_counts,
            &mut cl.cell_offsets,
            &mut cl.scan_block_totals,
            cl.n_cells_total,
            particle_count,
        )?;

        scatter_atoms_into_cells(
            &device,
            &cl.cell_indices,
            &cl.cell_offsets,
            &mut cl.write_cursors,
            &mut cl.sorted_particle_ids,
            particle_count,
        )?;

        sort_cells_by_particle_id(
            &device,
            &cl.cell_offsets,
            &mut cl.sorted_particle_ids,
            cl.n_cells_total,
        )?;

        device
            .memset_zeros(&mut cl.overflow_flag)
            .map_err(GpuError::from)?;

        timings
            .kernel_start(KernelStage::NeighborListBuild)
            .map_err(map_timings_err)?;
        neighbor_list_build(
            buffers,
            &cl.sorted_particle_ids,
            &cl.cell_offsets,
            sim_box,
            cl.n_cells,
            cl.cell_size,
            cl.r_search_sq,
            max_neighbors,
            &mut self.neighbor_list,
            &mut self.neighbor_counts,
            &mut cl.overflow_flag,
        )?;
        timings
            .kernel_stop(KernelStage::NeighborListBuild)
            .map_err(map_timings_err)?;

        let flag: Vec<u32> = self
            .device
            .dtoh_sync_copy(&cl.overflow_flag)
            .map_err(GpuError::from)?;
        if flag[0] != 0 {
            return Err(NeighborListError::NeighborListOverflow {
                max: max_neighbors,
            });
        }

        timings
            .kernel_start(KernelStage::CopyPositionsIntoReference)
            .map_err(map_timings_err)?;
        copy_positions_into_reference(
            buffers,
            &mut cl.reference_positions_x,
            &mut cl.reference_positions_y,
            &mut cl.reference_positions_z,
        )?;
        timings
            .kernel_stop(KernelStage::CopyPositionsIntoReference)
            .map_err(map_timings_err)?;

        cl.needs_rebuild = false;
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

        let refreshed = self.refresh_cell_layout_if_box_changed(sim_box)?;

        let (mut rebuild_required, r_skin) = match &self.mode {
            NeighborListMode::Trivial => unreachable!(),
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

fn compute_cell_layout(
    sim_box: &SimulationBox,
    r_search: f32,
) -> Result<([u32; 3], [f32; 3]), NeighborListError> {
    let lengths = sim_box.lengths();
    let axis_names: [&'static str; 3] = ["x", "y", "z"];
    let mut n_cells = [0u32; 3];
    let mut cell_size = [0.0f32; 3];
    for a in 0..3 {
        let l = lengths[a];
        let nc = (l / r_search).floor() as i64;
        if nc < 3 {
            return Err(NeighborListError::BoxTooSmallForCells {
                axis: axis_names[a],
                length: l,
                required: 3.0 * r_search,
            });
        }
        n_cells[a] = nc as u32;
        cell_size[a] = l / nc as f32;
    }
    Ok((n_cells, cell_size))
}

fn check_n_cells_total(n_cells_total: usize) -> Result<(), NeighborListError> {
    if n_cells_total > SPATIAL_HASH_MAX_CELLS {
        return Err(NeighborListError::TooManyCells {
            n_cells_total,
            max_supported: SPATIAL_HASH_MAX_CELLS,
        });
    }
    Ok(())
}

fn scan_blocks_for(n_cells_total: usize) -> usize {
    let block = SPATIAL_HASH_SCAN_BLOCK_SIZE as usize;
    (n_cells_total + block - 1) / block
}

fn map_timings_err(e: crate::timings::TimingsError) -> NeighborListError {
    match e {
        crate::timings::TimingsError::Gpu(g) => NeighborListError::Gpu(g),
    }
}
