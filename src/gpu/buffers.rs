use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};

use crate::gpu::{GpuContext, GpuError, Kernels};
use crate::state::{ParticleState, ParticleStateError, check_len};

// rq-4a8de06c
#[derive(Debug)]
pub struct ParticleBuffers {
    pub device: Arc<CudaDevice>,
    pub kernels: Arc<Kernels>,
    pub positions_x: CudaSlice<f32>,
    pub positions_y: CudaSlice<f32>,
    pub positions_z: CudaSlice<f32>,
    pub images_x: CudaSlice<i32>,
    pub images_y: CudaSlice<i32>,
    pub images_z: CudaSlice<i32>,
    pub velocities_x: CudaSlice<f32>,
    pub velocities_y: CudaSlice<f32>,
    pub velocities_z: CudaSlice<f32>,
    pub forces_x: CudaSlice<f32>,
    pub forces_y: CudaSlice<f32>,
    pub forces_z: CudaSlice<f32>,
    pub potential_energies: CudaSlice<f32>,
    pub virials: CudaSlice<f32>,
    pub masses: CudaSlice<f32>,
    pub charges: CudaSlice<f32>,
    pub type_indices: CudaSlice<u32>,
    pub particle_ids: CudaSlice<u32>,
}

impl ParticleBuffers {
    // rq-b09032cb
    pub fn new(
        gpu: &GpuContext,
        state: &ParticleState,
    ) -> Result<Self, ParticleStateError> {
        let device = gpu.device.clone();
        let kernels = gpu.kernels.clone();
        let n = state.particle_count();
        check_len("positions_y", n, state.positions_y.len())?;
        check_len("positions_z", n, state.positions_z.len())?;
        check_len("images_x", n, state.images_x.len())?;
        check_len("images_y", n, state.images_y.len())?;
        check_len("images_z", n, state.images_z.len())?;
        check_len("velocities_x", n, state.velocities_x.len())?;
        check_len("velocities_y", n, state.velocities_y.len())?;
        check_len("velocities_z", n, state.velocities_z.len())?;
        check_len("forces_x", n, state.forces_x.len())?;
        check_len("forces_y", n, state.forces_y.len())?;
        check_len("forces_z", n, state.forces_z.len())?;
        check_len("potential_energies", n, state.potential_energies.len())?;
        check_len("virials", n, state.virials.len())?;
        check_len("masses", n, state.masses.len())?;
        check_len("charges", n, state.charges.len())?;
        check_len("type_indices", n, state.type_indices.len())?;
        check_len("particle_ids", n, state.particle_ids.len())?;

        let positions_x = device.htod_sync_copy(&state.positions_x).map_err(GpuError::from)?;
        let positions_y = device.htod_sync_copy(&state.positions_y).map_err(GpuError::from)?;
        let positions_z = device.htod_sync_copy(&state.positions_z).map_err(GpuError::from)?;
        let images_x = device.htod_sync_copy(&state.images_x).map_err(GpuError::from)?;
        let images_y = device.htod_sync_copy(&state.images_y).map_err(GpuError::from)?;
        let images_z = device.htod_sync_copy(&state.images_z).map_err(GpuError::from)?;
        let velocities_x = device.htod_sync_copy(&state.velocities_x).map_err(GpuError::from)?;
        let velocities_y = device.htod_sync_copy(&state.velocities_y).map_err(GpuError::from)?;
        let velocities_z = device.htod_sync_copy(&state.velocities_z).map_err(GpuError::from)?;
        let forces_x = device.htod_sync_copy(&state.forces_x).map_err(GpuError::from)?;
        let forces_y = device.htod_sync_copy(&state.forces_y).map_err(GpuError::from)?;
        let forces_z = device.htod_sync_copy(&state.forces_z).map_err(GpuError::from)?;
        let potential_energies = device
            .htod_sync_copy(&state.potential_energies)
            .map_err(GpuError::from)?;
        let virials = device.htod_sync_copy(&state.virials).map_err(GpuError::from)?;
        let masses = device.htod_sync_copy(&state.masses).map_err(GpuError::from)?;
        let charges = device.htod_sync_copy(&state.charges).map_err(GpuError::from)?;
        let type_indices = device.htod_sync_copy(&state.type_indices).map_err(GpuError::from)?;
        let particle_ids = device.htod_sync_copy(&state.particle_ids).map_err(GpuError::from)?;

        Ok(ParticleBuffers {
            device,
            kernels,
            positions_x,
            positions_y,
            positions_z,
            images_x,
            images_y,
            images_z,
            velocities_x,
            velocities_y,
            velocities_z,
            forces_x,
            forces_y,
            forces_z,
            potential_energies,
            virials,
            masses,
            charges,
            type_indices,
            particle_ids,
        })
    }

    // rq-18411920
    pub fn particle_count(&self) -> usize {
        self.positions_x.len()
    }

    // rq-179ed985
    pub fn upload(&mut self, state: &ParticleState) -> Result<(), ParticleStateError> {
        let n = self.particle_count();
        check_len("positions_x", n, state.positions_x.len())?;
        check_len("positions_y", n, state.positions_y.len())?;
        check_len("positions_z", n, state.positions_z.len())?;
        check_len("images_x", n, state.images_x.len())?;
        check_len("images_y", n, state.images_y.len())?;
        check_len("images_z", n, state.images_z.len())?;
        check_len("velocities_x", n, state.velocities_x.len())?;
        check_len("velocities_y", n, state.velocities_y.len())?;
        check_len("velocities_z", n, state.velocities_z.len())?;
        check_len("forces_x", n, state.forces_x.len())?;
        check_len("forces_y", n, state.forces_y.len())?;
        check_len("forces_z", n, state.forces_z.len())?;
        check_len("potential_energies", n, state.potential_energies.len())?;
        check_len("virials", n, state.virials.len())?;
        check_len("masses", n, state.masses.len())?;
        check_len("charges", n, state.charges.len())?;
        check_len("type_indices", n, state.type_indices.len())?;
        check_len("particle_ids", n, state.particle_ids.len())?;

        let device = &self.device;
        device
            .htod_sync_copy_into(&state.positions_x, &mut self.positions_x)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.positions_y, &mut self.positions_y)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.positions_z, &mut self.positions_z)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.images_x, &mut self.images_x)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.images_y, &mut self.images_y)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.images_z, &mut self.images_z)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.velocities_x, &mut self.velocities_x)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.velocities_y, &mut self.velocities_y)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.velocities_z, &mut self.velocities_z)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.forces_x, &mut self.forces_x)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.forces_y, &mut self.forces_y)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.forces_z, &mut self.forces_z)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.potential_energies, &mut self.potential_energies)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.virials, &mut self.virials)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.masses, &mut self.masses)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.charges, &mut self.charges)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.type_indices, &mut self.type_indices)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.particle_ids, &mut self.particle_ids)
            .map_err(GpuError::from)?;
        Ok(())
    }
}
