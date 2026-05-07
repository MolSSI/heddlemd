use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};

use crate::gpu::GpuError;

// rq-303211d9
#[derive(Debug)]
pub struct LosslessBuffers {
    pub device: Arc<CudaDevice>,
    pub positions_x_lo: CudaSlice<f64>,
    pub positions_y_lo: CudaSlice<f64>,
    pub positions_z_lo: CudaSlice<f64>,
    pub velocities_x_lo: CudaSlice<f64>,
    pub velocities_y_lo: CudaSlice<f64>,
    pub velocities_z_lo: CudaSlice<f64>,
    particle_count: usize,
}

impl LosslessBuffers {
    pub fn new(
        device: Arc<CudaDevice>,
        particle_count: usize,
    ) -> Result<Self, GpuError> {
        let positions_x_lo = device.alloc_zeros::<f64>(particle_count).map_err(GpuError::from)?;
        let positions_y_lo = device.alloc_zeros::<f64>(particle_count).map_err(GpuError::from)?;
        let positions_z_lo = device.alloc_zeros::<f64>(particle_count).map_err(GpuError::from)?;
        let velocities_x_lo = device.alloc_zeros::<f64>(particle_count).map_err(GpuError::from)?;
        let velocities_y_lo = device.alloc_zeros::<f64>(particle_count).map_err(GpuError::from)?;
        let velocities_z_lo = device.alloc_zeros::<f64>(particle_count).map_err(GpuError::from)?;
        Ok(LosslessBuffers {
            device,
            positions_x_lo,
            positions_y_lo,
            positions_z_lo,
            velocities_x_lo,
            velocities_y_lo,
            velocities_z_lo,
            particle_count,
        })
    }

    pub fn particle_count(&self) -> usize {
        self.particle_count
    }
}
