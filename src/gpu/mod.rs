pub mod barostat_kernels;
pub mod buffers;
pub mod cufft;
pub mod device;
pub mod fill;
pub mod graph;
pub mod kernels;
pub mod lossless_buffers;

pub use barostat_kernels::BarostatKernels;
pub use buffers::ParticleBuffers;
pub use device::{GpuContext, GpuError, Kernels, init_device};
pub use fill::FillKernels;
pub use graph::{
    CaptureMode, CudaGraph, CudaGraphExec, GraphError, GraphNodeSummary, begin_stream_capture,
    end_stream_capture,
};
pub use kernels::{
    CSVR_PARTIAL_BLOCKS, K_COULOMB_F32, LennardJonesParameterTable, SPATIAL_HASH_SCAN_BLOCK_SIZE,
    andersen_resample, berendsen_compute_factor, berendsen_compute_mu,
    c_rescale_compute_mu, combine_class_totals,
    compute_block_bbox,
    compute_cell_indices_and_histogram,
    compute_kinetic_energy, compute_kinetic_energy_on_device,
    compute_total_potential_energy, compute_total_virial,
    compute_total_virial_on_device,
    copy_positions_into_reference,
    csvr_sample_and_factor,
    fill_tile_position_padding,
    finalize_packed_forces,
    find_blocks_with_interactions,
    set_neighbor_status_bits,
    PrefixScanSentinel,
    histogram_entries_by_iblock,
    scatter_entries_by_iblock,
    lan_drift_half, lan_ou_step,
    mtk_position_drift, mtk_velocity_half_kick,
    neighbor_displacement_check_flag,
    prefix_scan_cell_counts, reduce_angle_forces, reduce_bond_forces,
    rescale_positions, rescale_positions_device_factor, rescale_velocities,
    rescale_velocities_device_factor,
    constraint_virial_scatter, increment_u64_device, rattle_velocities, scatter_atoms_into_cells,
    scatter_positions_to_tile_order,
    shake_positions, shake_positions_no_velocity, shake_snapshot,
    sort_cells_by_particle_id,
    spme_atom_sort, spme_compute_bin_key,
    spme_spread_finish, spme_spread_fixed_point,
    spme_force_gather, spme_recip_apply_influence,
    spme_recip_compute_influence,
    spme_recip_reduce_partials, vv_kick,
    vv_kick_drift,
};
#[cfg(not(feature = "f64"))]
pub use kernels::{vv_kick_drift_lossless, vv_kick_lossless};
pub use lossless_buffers::LosslessBuffers;

use std::sync::Arc;
use cudarc::driver::CudaSlice;
use crate::precision::Real;

/// Helper that owns the five per-particle output buffers a pair-force
/// kernel writes through `SlotOutputView`. Provided for test scaffolding
/// and for callers that want a one-shot output target outside the
/// per-class slot-output buffers managed by `ForceField`.
#[derive(Debug)]
pub struct SlotOutputBuffers {
    pub force_x: CudaSlice<Real>,
    pub force_y: CudaSlice<Real>,
    pub force_z: CudaSlice<Real>,
    pub energy: CudaSlice<Real>,
    pub virial: CudaSlice<Real>,
}

impl SlotOutputBuffers {
    pub fn new(device: &Arc<cudarc::driver::CudaDevice>, n: usize) -> Result<Self, GpuError> {
        Ok(SlotOutputBuffers {
            force_x: device.alloc_zeros::<Real>(n).map_err(GpuError::from)?,
            force_y: device.alloc_zeros::<Real>(n).map_err(GpuError::from)?,
            force_z: device.alloc_zeros::<Real>(n).map_err(GpuError::from)?,
            energy: device.alloc_zeros::<Real>(n).map_err(GpuError::from)?,
            virial: device.alloc_zeros::<Real>(n).map_err(GpuError::from)?,
        })
    }

    pub fn view(&mut self) -> crate::forces::SlotOutputView<'_> {
        crate::forces::SlotOutputView {
            force_x: self.force_x.slice_mut(..),
            force_y: self.force_y.slice_mut(..),
            force_z: self.force_z.slice_mut(..),
            energy: self.energy.slice_mut(..),
            virial: self.virial.slice_mut(..),
        }
    }
}
