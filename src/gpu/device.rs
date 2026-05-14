use std::sync::Arc;

use cudarc::driver::{CudaDevice, DriverError};
use cudarc::nvrtc::Ptx;

use crate::kernels;

// rq-91ac50d0 rq-e1ceb5c0
#[derive(Debug, thiserror::Error)]
#[error("GPU error: {0}")]
pub struct GpuError(#[from] pub DriverError);

pub fn init_device() -> Result<Arc<CudaDevice>, GpuError> {
    let device = CudaDevice::new(0)?;
    device.load_ptx(Ptx::from_src(kernels::FILL), "fill", &["fill"])?;
    // rq-e20b2f39
    device.load_ptx(
        Ptx::from_src(kernels::INTEGRATE),
        "integrate",
        &[
            "vv_kick_drift",
            "vv_kick",
            "vv_kick_drift_lossless",
            "vv_kick_lossless",
        ],
    )?;
    // rq-56d8375d
    device.load_ptx(
        Ptx::from_src(kernels::REDUCE),
        "reduce",
        &["reduce_pair_forces"],
    )?;
    // rq-78d9fd1c
    device.load_ptx(
        Ptx::from_src(kernels::PAIR_FORCE),
        "pair_force",
        &["lj_pair_force"],
    )?;
    device.load_ptx(
        Ptx::from_src(kernels::LANGEVIN),
        "langevin",
        &["lan_drift_half", "lan_ou_step"],
    )?;
    device.load_ptx(
        Ptx::from_src(kernels::MORSE),
        "morse",
        &["morse_bond_force", "reduce_bond_forces"],
    )?;
    device.load_ptx(
        Ptx::from_src(kernels::FORCES),
        "forces",
        &["accumulate_forces"],
    )?;
    // rq-0469400b
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
        ],
    )?;
    Ok(device)
}
