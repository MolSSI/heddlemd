use cudarc::driver::{CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};

use crate::gpu::{GpuError, PairBuffer, ParticleBuffers};

const BLOCK_SIZE: u32 = 256;

fn launch_config(n: u32) -> LaunchConfig {
    let grid = n.div_ceil(BLOCK_SIZE);
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

// rq-f1ba909b
pub fn vv_kick_drift(
    buffers: &mut ParticleBuffers,
    dt: f32,
) -> Result<(), GpuError> {
    let n = buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    let func = buffers
        .device
        .get_func("integrate", "vv_kick_drift")
        .expect("integrate module is not loaded; init_device() must be called first");
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(
            cfg,
            (
                &mut buffers.positions_x,
                &mut buffers.positions_y,
                &mut buffers.positions_z,
                &mut buffers.velocities_x,
                &mut buffers.velocities_y,
                &mut buffers.velocities_z,
                &buffers.forces_x,
                &buffers.forces_y,
                &buffers.forces_z,
                &buffers.masses,
                dt,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-f2e3fa58
pub fn vv_kick(buffers: &mut ParticleBuffers, dt: f32) -> Result<(), GpuError> {
    let n = buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    let func = buffers
        .device
        .get_func("integrate", "vv_kick")
        .expect("integrate module is not loaded; init_device() must be called first");
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(
            cfg,
            (
                &mut buffers.velocities_x,
                &mut buffers.velocities_y,
                &mut buffers.velocities_z,
                &buffers.forces_x,
                &buffers.forces_y,
                &buffers.forces_z,
                &buffers.masses,
                dt,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-6690fae9
pub fn reduce_pair_forces(
    pair_buffer: &PairBuffer,
    neighbor_counts: &CudaSlice<u32>,
    particle_buffers: &mut ParticleBuffers,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let max_neighbors = pair_buffer.max_neighbors();
    debug_assert_eq!(pair_buffer.particle_count(), n);
    debug_assert_eq!(neighbor_counts.len(), n);
    debug_assert_eq!(
        pair_buffer.pair_forces_x.len(),
        n * max_neighbors as usize
    );

    let n_u32 = n as u32;
    let func = particle_buffers
        .device
        .get_func("reduce", "reduce_pair_forces")
        .expect("reduce module is not loaded; init_device() must be called first");
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(
            cfg,
            (
                &pair_buffer.pair_forces_x,
                &pair_buffer.pair_forces_y,
                &pair_buffer.pair_forces_z,
                neighbor_counts,
                max_neighbors,
                &mut particle_buffers.forces_x,
                &mut particle_buffers.forces_y,
                &mut particle_buffers.forces_z,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}
