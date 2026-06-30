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
// The `Kernels` aggregate (one typed field per subsystem), `Kernels::load`
// (each subsystem's loader in manifest order, short-circuiting on the
// first failure), and the `KernelStage::ORDER` registry (the manifest-order
// concatenation of every subsystem's `STAGES`) are all generated from this
// single manifest. `device.rs` names no individual kernel and carries no
// per-subsystem stage list; adding a subsystem is one line here plus its
// own `gpu_kernels!` invocation. See `rqm/build-pipeline.md`.
crate::define_kernels! {
    fill:        crate::gpu::fill::FillKernels,
    integrate:   crate::integrator::velocity_verlet::IntegrateKernels,
    spme_recip:  crate::forces::spme::SpmeRecipKernels,
    langevin:    crate::integrator::langevin_baoab::LangevinKernels,
    morse:       crate::forces::morse::MorseKernels,
    angle:       crate::forces::angle::AngleKernels,
    nose_hoover: crate::integrator::nose_hoover_chain::NoseHooverKernels,
    andersen:    crate::integrator::andersen::AndersenKernels,
    barostat:    crate::gpu::barostat_kernels::BarostatKernels,
    mc_barostat: crate::gpu::mc_barostat_kernels::McBarostatKernels,
    mtk:         crate::integrator::mtk_npt::MtkKernels,
    shake:       crate::integrator::shake::ShakeKernels,
    settle:      crate::integrator::settle::SettleKernels,
    forces:      crate::forces::ForcesKernels,
    neighbor:    crate::forces::neighbor_list::NeighborKernels,
    minimize:    crate::minimizer::MinimizeKernels,
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
