pub mod barostat_kernels;
pub mod buffers;
pub mod cufft;
pub mod device;
pub mod fill;
pub mod kernels;
pub mod lossless_buffers;

pub use barostat_kernels::BarostatKernels;
pub use buffers::ParticleBuffers;
pub use device::{GpuContext, GpuError, Kernels, init_device};
pub use fill::FillKernels;
pub use kernels::{
    K_COULOMB_F32, LennardJonesParameterTable, SPATIAL_HASH_SCAN_BLOCK_SIZE,
    accumulate_forces, andersen_resample, compute_cell_indices_and_histogram,
    compute_kinetic_energy, compute_total_potential_energy, compute_total_virial,
    copy_positions_into_reference,
    coulomb_pair_force, harmonic_angle_force, lan_drift_half, lan_ou_step, lj_pair_force,
    morse_bond_force, mtk_position_drift, mtk_velocity_half_kick,
    neighbor_displacement_squared, neighbor_list_build,
    prefix_scan_cell_counts, reduce_angle_forces, reduce_bond_forces,
    rescale_positions, rescale_velocities,
    constraint_virial_scatter, rattle_velocities, scatter_atoms_into_cells,
    shake_positions, shake_positions_no_velocity, shake_snapshot,
    sort_cells_by_particle_id, spme_charge_spread, spme_charge_spread_on_stream,
    spme_force_gather, spme_influence_multiply, spme_influence_multiply_on_stream,
    spme_real_pair_force, spme_recip_virial_finalize_on_stream, vv_kick,
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
