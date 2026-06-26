use std::sync::Arc;

use cudarc::driver::sys::CUresult;
use cudarc::driver::{CudaDevice, CudaFunction, DriverError};

// rq-91ac50d0 rq-e1ceb5c0
#[derive(Debug, thiserror::Error)]
#[error("GPU error: {0}")]
pub struct GpuError(#[from] pub DriverError);

// Shared helper used by every subsystem's `XKernels::load`. Looks up a
// kernel function on the device; absence surfaces as `GpuError` rather
// than `Option::None`. Kernels missing from a loaded PTX module are
// caught once at startup, not on first launch.
pub(crate) fn get_func(
    device: &Arc<CudaDevice>,
    module: &str,
    name: &str,
) -> Result<CudaFunction, GpuError> {
    device
        .get_func(module, name)
        .ok_or(GpuError(DriverError(CUresult::CUDA_ERROR_NOT_FOUND)))
}

// rq-2093594f
//
// Typed handle to every CUDA kernel compiled from `kernels/*.cu`,
// organized as one field per subsystem (one field per `.cu` file).
// Each subsystem owns its own kernel struct, its loader, and the
// names of its kernels. `device.rs` does not name any individual
// kernel — adding one is a one-line edit to the relevant
// subsystem's `XKernels::load`. Adding a whole new subsystem is a
// one-line edit here.
#[derive(Debug, Clone)]
pub struct Kernels {
    pub fill: crate::gpu::fill::FillKernels,
    pub integrate: crate::integrator::velocity_verlet::IntegrateKernels,
    pub spme_recip: crate::forces::spme::SpmeRecipKernels,
    pub langevin: crate::integrator::langevin_baoab::LangevinKernels,
    pub morse: crate::forces::morse::MorseKernels,
    pub angle: crate::forces::angle::AngleKernels,
    pub nose_hoover: crate::integrator::nose_hoover_chain::NoseHooverKernels,
    pub andersen: crate::integrator::andersen::AndersenKernels,
    pub barostat: crate::gpu::barostat_kernels::BarostatKernels,
    pub mtk: crate::integrator::mtk_npt::MtkKernels,
    pub shake: crate::integrator::shake::ShakeKernels,
    pub forces: crate::forces::ForcesKernels,
    pub neighbor: crate::forces::neighbor_list::NeighborKernels,
    pub minimize: crate::minimizer::MinimizeKernels,
}

impl Kernels {
    // rq-2093594f — composes every subsystem's `XKernels::load` in
    // registration order. The first failing subsystem short-circuits
    // the rest.
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        Ok(Kernels {
            fill: crate::gpu::fill::FillKernels::load(device)?,
            integrate: crate::integrator::velocity_verlet::IntegrateKernels::load(device)?,
            spme_recip: crate::forces::spme::SpmeRecipKernels::load(device)?,
            langevin: crate::integrator::langevin_baoab::LangevinKernels::load(device)?,
            morse: crate::forces::morse::MorseKernels::load(device)?,
            angle: crate::forces::angle::AngleKernels::load(device)?,
            nose_hoover: crate::integrator::nose_hoover_chain::NoseHooverKernels::load(device)?,
            andersen: crate::integrator::andersen::AndersenKernels::load(device)?,
            barostat: crate::gpu::barostat_kernels::BarostatKernels::load(device)?,
            mtk: crate::integrator::mtk_npt::MtkKernels::load(device)?,
            shake: crate::integrator::shake::ShakeKernels::load(device)?,
            forces: crate::forces::ForcesKernels::load(device)?,
            neighbor: crate::forces::neighbor_list::NeighborKernels::load(device)?,
            minimize: crate::minimizer::MinimizeKernels::load(device)?,
        })
    }
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
    let device = CudaDevice::new_with_stream(0)?;
    let kernels = Kernels::load(&device)?;
    Ok(GpuContext {
        device,
        kernels: Arc::new(kernels),
    })
}
