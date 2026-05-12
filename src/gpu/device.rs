use std::sync::Arc;

use cudarc::driver::{CudaDevice, DriverError};
use cudarc::nvrtc::Ptx;

use crate::kernels;

#[derive(Debug)]
pub struct GpuError(pub DriverError);

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GPU error: {}", self.0)
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

impl From<DriverError> for GpuError {
    fn from(e: DriverError) -> Self {
        GpuError(e)
    }
}

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
    Ok(device)
}
