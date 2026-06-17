// rq-2093594f
//
// Kernel handles for the `barostat` PTX module (`kernels/barostat.cu`).
// Two kernels (`virial_sum_reduce`, `rescale_positions`) shared by the
// Berendsen and c-rescale barostats and by the MTK barostat substep.
// Lives in `src/gpu/` (rather than inside one barostat's module)
// because no single consumer is its natural owner.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction};
use cudarc::nvrtc::Ptx;

use super::device::{GpuError, get_func};
use crate::kernels;

#[derive(Debug, Clone)]
pub struct BarostatKernels {
    pub virial_sum_reduce: CudaFunction,
    pub rescale_positions: CudaFunction,
    pub rescale_positions_device_factor: CudaFunction,
    pub multiply_lattice_isotropic: CudaFunction,
    pub c_rescale_compute_mu: CudaFunction,
    pub berendsen_compute_mu: CudaFunction,
}

impl BarostatKernels {
    // rq-0d8c8688
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::BAROSTAT),
            "barostat",
            &[
                "virial_sum_reduce",
                "rescale_positions",
                "rescale_positions_device_factor",
                "multiply_lattice_isotropic",
                "c_rescale_compute_mu",
                "berendsen_compute_mu",
            ],
        )?;
        Ok(BarostatKernels {
            virial_sum_reduce: get_func(device, "barostat", "virial_sum_reduce")?,
            rescale_positions: get_func(device, "barostat", "rescale_positions")?,
            rescale_positions_device_factor: get_func(
                device,
                "barostat",
                "rescale_positions_device_factor",
            )?,
            multiply_lattice_isotropic: get_func(
                device,
                "barostat",
                "multiply_lattice_isotropic",
            )?,
            c_rescale_compute_mu: get_func(device, "barostat", "c_rescale_compute_mu")?,
            berendsen_compute_mu: get_func(device, "barostat", "berendsen_compute_mu")?,
        })
    }
}
