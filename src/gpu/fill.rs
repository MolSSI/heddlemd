// rq-2093594f
//
// Kernel handles for the `fill` PTX module (`kernels/fill.cu`). The
// smoke-test kernel that validates the full toolchain.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction};
use cudarc::nvrtc::Ptx;

use super::device::{GpuError, get_func};
use crate::kernels;

#[derive(Debug, Clone)]
pub struct FillKernels {
    pub fill: CudaFunction,
}

impl FillKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(Ptx::from_src(kernels::FILL), "fill", &["fill"])?;
        Ok(FillKernels {
            fill: get_func(device, "fill", "fill")?,
        })
    }
}
