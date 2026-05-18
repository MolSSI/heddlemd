use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice};
use cudarc::nvrtc::Ptx;

use crate::gpu::device::get_func;
use crate::gpu::{GpuContext, GpuError, Kernels};
use crate::kernels;

// rq-2093594f rq-56d8375d
#[derive(Debug, Clone)]
pub struct ReduceKernels {
    pub reduce_pair_forces: CudaFunction,
}

impl ReduceKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::REDUCE),
            "reduce",
            &["reduce_pair_forces"],
        )?;
        Ok(ReduceKernels {
            reduce_pair_forces: get_func(device, "reduce", "reduce_pair_forces")?,
        })
    }
}

// rq-a0c0992f
#[derive(Debug)]
pub struct PairBuffer {
    pub device: Arc<CudaDevice>,
    pub kernels: Arc<Kernels>,
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
        gpu: &GpuContext,
        particle_count: usize,
        max_neighbors: u32,
    ) -> Result<Self, GpuError> {
        let device = gpu.device.clone();
        let kernels = gpu.kernels.clone();
        let len = particle_count * max_neighbors as usize;
        let pair_forces_x = device.alloc_zeros::<f32>(len).map_err(GpuError::from)?;
        let pair_forces_y = device.alloc_zeros::<f32>(len).map_err(GpuError::from)?;
        let pair_forces_z = device.alloc_zeros::<f32>(len).map_err(GpuError::from)?;
        let pair_energies = device.alloc_zeros::<f32>(len).map_err(GpuError::from)?;
        let pair_virials = device.alloc_zeros::<f32>(len).map_err(GpuError::from)?;
        Ok(PairBuffer {
            device,
            kernels,
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
