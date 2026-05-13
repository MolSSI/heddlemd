// rq-693ce6fa rq-b2d68288
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaDevice, CudaSlice};

use crate::gpu::{
    GpuError, ParticleBuffers, copy_positions_into_reference, neighbor_displacement_squared,
    neighbor_list_build,
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
    pub r_skin: f32,
    pub r_search_sq: f32,
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
    pub sim_box: SimulationBox,
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
        sim_box: SimulationBox,
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
        let lengths = sim_box.lengths();
        let mut n_cells = [0u32; 3];
        let mut cell_size = [0.0f32; 3];
        let axis_names: [&'static str; 3] = ["x", "y", "z"];
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
        let n_cells_total =
            n_cells[0] as usize * n_cells[1] as usize * n_cells[2] as usize;

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
            sim_box,
            particle_count,
            max_neighbors,
            neighbor_list,
            neighbor_counts,
            mode: NeighborListMode::CellList(CellListData {
                n_cells,
                cell_size,
                n_cells_total,
                r_skin,
                r_search_sq,
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

    // rq-77754ad1
    pub fn new_trivial(
        device: Arc<CudaDevice>,
        sim_box: SimulationBox,
        particle_count: usize,
    ) -> Result<Self, NeighborListError> {
        let max_neighbors = particle_count as u32;
        // Populate neighbor_list[i * N + k] = k and neighbor_counts[i] = N.
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
            sim_box,
            particle_count,
            max_neighbors,
            neighbor_list,
            neighbor_counts,
            mode: NeighborListMode::Trivial,
        })
    }

    // rq-c49b2fe6
    pub fn displacement_check(
        &mut self,
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
            &self.sim_box,
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
        let result = self.rebuild_impl(buffers, timings);
        timings.record_host(HostStage::NeighborListRebuild, started.elapsed());
        result
    }

    fn rebuild_impl(
        &mut self,
        buffers: &ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<(), NeighborListError> {
        let n = self.particle_count;
        // Download current positions.
        let pos_x: Vec<f32> = self
            .device
            .dtoh_sync_copy(&buffers.positions_x)
            .map_err(GpuError::from)?;
        let pos_y: Vec<f32> = self
            .device
            .dtoh_sync_copy(&buffers.positions_y)
            .map_err(GpuError::from)?;
        let pos_z: Vec<f32> = self
            .device
            .dtoh_sync_copy(&buffers.positions_z)
            .map_err(GpuError::from)?;

        let lengths = self.sim_box.lengths();
        let sim_box = self.sim_box;
        let max_neighbors = self.max_neighbors;

        let cl = match &mut self.mode {
            NeighborListMode::Trivial => return Ok(()),
            NeighborListMode::CellList(cl) => cl,
        };

        let mut entries: Vec<(u32, u32)> = Vec::with_capacity(n);
        for i in 0..n {
            let cx = cell_index_axis(pos_x[i], lengths[0], cl.cell_size[0], cl.n_cells[0]);
            let cy = cell_index_axis(pos_y[i], lengths[1], cl.cell_size[1], cl.n_cells[1]);
            let cz = cell_index_axis(pos_z[i], lengths[2], cl.cell_size[2], cl.n_cells[2]);
            let c = (cx * cl.n_cells[1] + cy) * cl.n_cells[2] + cz;
            entries.push((c, i as u32));
        }
        entries.sort_by_key(|&(c, pid)| (c, pid));

        let sorted_ids: Vec<u32> = entries.iter().map(|&(_, pid)| pid).collect();
        let mut counts: Vec<u32> = vec![0u32; cl.n_cells_total];
        for &(c, _) in &entries {
            counts[c as usize] += 1;
        }
        let mut offsets: Vec<u32> = vec![0u32; cl.n_cells_total + 1];
        for c in 0..cl.n_cells_total {
            offsets[c + 1] = offsets[c] + counts[c];
        }

        self.device
            .htod_sync_copy_into(&sorted_ids, &mut cl.sorted_particle_ids)
            .map_err(GpuError::from)?;
        self.device
            .htod_sync_copy_into(&offsets, &mut cl.cell_offsets)
            .map_err(GpuError::from)?;

        let zero: [u32; 1] = [0];
        self.device
            .htod_sync_copy_into(&zero, &mut cl.overflow_flag)
            .map_err(GpuError::from)?;

        timings
            .kernel_start(KernelStage::NeighborListBuild)
            .map_err(map_timings_err)?;
        neighbor_list_build(
            buffers,
            &cl.sorted_particle_ids,
            &cl.cell_offsets,
            &sim_box,
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
        buffers: &ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<(), NeighborListError> {
        if self.particle_count == 0 {
            return Ok(());
        }
        let (needs_rebuild, r_skin) = match &self.mode {
            NeighborListMode::Trivial => return Ok(()),
            NeighborListMode::CellList(cl) => (cl.needs_rebuild, cl.r_skin),
        };
        let mut rebuild_required = needs_rebuild;
        if !rebuild_required {
            let max_disp = self.displacement_check(buffers, timings)?;
            if max_disp > r_skin * 0.5 {
                rebuild_required = true;
            }
        }
        if rebuild_required {
            if let NeighborListMode::CellList(cl) = &mut self.mode {
                cl.needs_rebuild = true;
            }
            self.rebuild(buffers, timings)?;
        }
        Ok(())
    }
}

fn cell_index_axis(x: f32, length: f32, cell_size: f32, n_cells: u32) -> u32 {
    let wrapped = x - length * ((x + length * 0.5) / length).floor();
    let mut idx = ((wrapped + length * 0.5) / cell_size).floor() as i64;
    if idx < 0 {
        idx = 0;
    }
    let max_idx = (n_cells as i64) - 1;
    if idx > max_idx {
        idx = max_idx;
    }
    idx as u32
}

fn map_timings_err(e: crate::timings::TimingsError) -> NeighborListError {
    match e {
        crate::timings::TimingsError::Gpu(g) => NeighborListError::Gpu(g),
    }
}
