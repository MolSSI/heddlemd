use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};

use crate::gpu::GpuError;

// rq-a0c0992f
#[derive(Debug)]
pub struct PairBuffer {
    pub device: Arc<CudaDevice>,
    pub pair_forces_x: CudaSlice<f32>,
    pub pair_forces_y: CudaSlice<f32>,
    pub pair_forces_z: CudaSlice<f32>,
    pub pair_energies: CudaSlice<f32>,
    pub pair_virials: CudaSlice<f32>,
    particle_count: usize,
    max_neighbors: u32,
}

impl PairBuffer {
    // rq-79048663
    pub fn new(
        device: Arc<CudaDevice>,
        particle_count: usize,
        max_neighbors: u32,
    ) -> Result<Self, GpuError> {
        let len = particle_count * max_neighbors as usize;
        let pair_forces_x = device.alloc_zeros::<f32>(len).map_err(GpuError::from)?;
        let pair_forces_y = device.alloc_zeros::<f32>(len).map_err(GpuError::from)?;
        let pair_forces_z = device.alloc_zeros::<f32>(len).map_err(GpuError::from)?;
        let pair_energies = device.alloc_zeros::<f32>(len).map_err(GpuError::from)?;
        let pair_virials = device.alloc_zeros::<f32>(len).map_err(GpuError::from)?;
        Ok(PairBuffer {
            device,
            pair_forces_x,
            pair_forces_y,
            pair_forces_z,
            pair_energies,
            pair_virials,
            particle_count,
            max_neighbors,
        })
    }

    // rq-3c42e6bd
    pub fn particle_count(&self) -> usize {
        self.particle_count
    }

    // rq-12657190
    pub fn max_neighbors(&self) -> u32 {
        self.max_neighbors
    }
}
