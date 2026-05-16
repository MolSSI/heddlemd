use std::sync::Arc;

use cudarc::driver::sys::CUresult;
use cudarc::driver::{CudaDevice, CudaFunction, DriverError};
use cudarc::nvrtc::Ptx;

use crate::kernels;

// rq-91ac50d0 rq-e1ceb5c0
#[derive(Debug, thiserror::Error)]
#[error("GPU error: {0}")]
pub struct GpuError(#[from] pub DriverError);

// rq-2093594f
//
// Typed handle to every CUDA kernel compiled from `kernels/*.cu`. One
// field per kernel, each named after the kernel. `init_device` populates
// every field once, so launch sites obtain their function through typed
// field access rather than a string-keyed lookup.
#[derive(Debug, Clone)]
pub struct Kernels {
    pub fill: CudaFunction,
    pub vv_kick_drift: CudaFunction,
    pub vv_kick: CudaFunction,
    pub vv_kick_drift_lossless: CudaFunction,
    pub vv_kick_lossless: CudaFunction,
    pub reduce_pair_forces: CudaFunction,
    pub lj_pair_force: CudaFunction,
    pub coulomb_pair_force: CudaFunction,
    pub spme_real_pair_force: CudaFunction,
    pub spme_charge_spread: CudaFunction,
    pub spme_influence_multiply: CudaFunction,
    pub spme_force_gather: CudaFunction,
    pub lan_drift_half: CudaFunction,
    pub lan_ou_step: CudaFunction,
    pub morse_bond_force: CudaFunction,
    pub reduce_bond_forces: CudaFunction,
    pub harmonic_angle_force: CudaFunction,
    pub reduce_angle_forces: CudaFunction,
    pub kinetic_energy_reduce: CudaFunction,
    pub rescale_velocities: CudaFunction,
    pub andersen_resample: CudaFunction,
    pub accumulate_forces: CudaFunction,
    pub neighbor_displacement_squared: CudaFunction,
    pub neighbor_list_build: CudaFunction,
    pub copy_positions_into_reference: CudaFunction,
    pub compute_cell_indices_and_histogram: CudaFunction,
    pub prefix_scan_local_blocks: CudaFunction,
    pub prefix_scan_apply_block_totals: CudaFunction,
    pub prefix_scan_finalize_offsets: CudaFunction,
    pub scatter_atoms_into_cells: CudaFunction,
    pub sort_cells_by_particle_id: CudaFunction,
}

// rq-2093594f
//
// The value returned by `init_device`: a plain struct bundling the
// initialized device and the typed kernel handle. Both fields are cheap
// to clone, so each GPU-resource struct stores its own clones rather than
// borrowing the `GpuContext`.
#[derive(Debug, Clone)]
pub struct GpuContext {
    pub device: Arc<CudaDevice>,
    pub kernels: Arc<Kernels>,
}

// rq-c38c8f3b
pub fn init_device() -> Result<GpuContext, GpuError> {
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
    // rq-846bdb8b
    device.load_ptx(
        Ptx::from_src(kernels::COULOMB),
        "coulomb",
        &["coulomb_pair_force"],
    )?;
    // rq-9a512ed1
    device.load_ptx(
        Ptx::from_src(kernels::SPME_REAL),
        "spme_real",
        &["spme_real_pair_force"],
    )?;
    // rq-9ca00d25
    device.load_ptx(
        Ptx::from_src(kernels::SPME_RECIP),
        "spme_recip",
        &["spme_charge_spread", "spme_influence_multiply", "spme_force_gather"],
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
        Ptx::from_src(kernels::ANGLE),
        "angle",
        &["harmonic_angle_force", "reduce_angle_forces"],
    )?;
    // rq-f606ff6f
    device.load_ptx(
        Ptx::from_src(kernels::NOSE_HOOVER),
        "nose_hoover",
        &["kinetic_energy_reduce", "rescale_velocities"],
    )?;
    // rq-5e059f6b
    device.load_ptx(
        Ptx::from_src(kernels::ANDERSEN),
        "andersen",
        &["andersen_resample"],
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

    // A kernel that is missing from its loaded module surfaces here, once,
    // as a `GpuError` rather than on first launch.
    let get = |module: &str, name: &str| -> Result<CudaFunction, GpuError> {
        device
            .get_func(module, name)
            .ok_or(GpuError(DriverError(CUresult::CUDA_ERROR_NOT_FOUND)))
    };

    let kernels = Kernels {
        fill: get("fill", "fill")?,
        vv_kick_drift: get("integrate", "vv_kick_drift")?,
        vv_kick: get("integrate", "vv_kick")?,
        vv_kick_drift_lossless: get("integrate", "vv_kick_drift_lossless")?,
        vv_kick_lossless: get("integrate", "vv_kick_lossless")?,
        reduce_pair_forces: get("reduce", "reduce_pair_forces")?,
        lj_pair_force: get("pair_force", "lj_pair_force")?,
        coulomb_pair_force: get("coulomb", "coulomb_pair_force")?,
        spme_real_pair_force: get("spme_real", "spme_real_pair_force")?,
        spme_charge_spread: get("spme_recip", "spme_charge_spread")?,
        spme_influence_multiply: get("spme_recip", "spme_influence_multiply")?,
        spme_force_gather: get("spme_recip", "spme_force_gather")?,
        lan_drift_half: get("langevin", "lan_drift_half")?,
        lan_ou_step: get("langevin", "lan_ou_step")?,
        morse_bond_force: get("morse", "morse_bond_force")?,
        reduce_bond_forces: get("morse", "reduce_bond_forces")?,
        harmonic_angle_force: get("angle", "harmonic_angle_force")?,
        reduce_angle_forces: get("angle", "reduce_angle_forces")?,
        kinetic_energy_reduce: get("nose_hoover", "kinetic_energy_reduce")?,
        rescale_velocities: get("nose_hoover", "rescale_velocities")?,
        andersen_resample: get("andersen", "andersen_resample")?,
        accumulate_forces: get("forces", "accumulate_forces")?,
        neighbor_displacement_squared: get("neighbor", "neighbor_displacement_squared")?,
        neighbor_list_build: get("neighbor", "neighbor_list_build")?,
        copy_positions_into_reference: get("neighbor", "copy_positions_into_reference")?,
        compute_cell_indices_and_histogram: get(
            "neighbor",
            "compute_cell_indices_and_histogram",
        )?,
        prefix_scan_local_blocks: get("neighbor", "prefix_scan_local_blocks")?,
        prefix_scan_apply_block_totals: get("neighbor", "prefix_scan_apply_block_totals")?,
        prefix_scan_finalize_offsets: get("neighbor", "prefix_scan_finalize_offsets")?,
        scatter_atoms_into_cells: get("neighbor", "scatter_atoms_into_cells")?,
        sort_cells_by_particle_id: get("neighbor", "sort_cells_by_particle_id")?,
    };

    Ok(GpuContext {
        device,
        kernels: Arc::new(kernels),
    })
}
