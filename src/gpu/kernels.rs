use cudarc::driver::{CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};

use crate::gpu::{GpuError, LosslessBuffers, PairBuffer, ParticleBuffers};
use crate::pbc::SimulationBox;

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

// rq-dafe0fcb
#[derive(Debug, Clone, Copy)]
pub struct LennardJonesParameters {
    pub sigma: f32,
    pub epsilon: f32,
    pub cutoff: f32,
}

// rq-d3a14184
pub fn lj_pair_force(
    particle_buffers: &ParticleBuffers,
    pair_buffer: &mut PairBuffer,
    sim_box: &SimulationBox,
    params: &LennardJonesParameters,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(pair_buffer.particle_count(), n);
    debug_assert!(pair_buffer.max_neighbors() as usize >= n);

    let n_u32 = n as u32;
    let max_neighbors = pair_buffer.max_neighbors();
    let func = particle_buffers
        .device
        .get_func("pair_force", "lj_pair_force")
        .expect("pair_force module is not loaded; init_device() must be called first");

    let grid_side = n_u32.div_ceil(16);
    let cfg = LaunchConfig {
        grid_dim: (grid_side, grid_side, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    let lengths = sim_box.lengths();
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.positions_x,
                &particle_buffers.positions_y,
                &particle_buffers.positions_z,
                &mut pair_buffer.pair_forces_x,
                &mut pair_buffer.pair_forces_y,
                &mut pair_buffer.pair_forces_z,
                max_neighbors,
                lengths[0],
                lengths[1],
                lengths[2],
                params.sigma,
                params.epsilon,
                params.cutoff,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

pub fn vv_kick_drift_lossless(
    buffers: &mut ParticleBuffers,
    lossless: &mut LosslessBuffers,
    dt: f32,
) -> Result<(), GpuError> {
    let n = buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(lossless.particle_count(), n);
    let n_u32 = n as u32;
    let func = buffers
        .device
        .get_func("integrate", "vv_kick_drift_lossless")
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
                &mut lossless.positions_x_lo,
                &mut lossless.positions_y_lo,
                &mut lossless.positions_z_lo,
                &mut lossless.velocities_x_lo,
                &mut lossless.velocities_y_lo,
                &mut lossless.velocities_z_lo,
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

pub fn vv_kick_lossless(
    buffers: &mut ParticleBuffers,
    lossless: &mut LosslessBuffers,
    dt: f32,
) -> Result<(), GpuError> {
    let n = buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(lossless.particle_count(), n);
    let n_u32 = n as u32;
    let func = buffers
        .device
        .get_func("integrate", "vv_kick_lossless")
        .expect("integrate module is not loaded; init_device() must be called first");
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(
            cfg,
            (
                &mut buffers.velocities_x,
                &mut buffers.velocities_y,
                &mut buffers.velocities_z,
                &mut lossless.velocities_x_lo,
                &mut lossless.velocities_y_lo,
                &mut lossless.velocities_z_lo,
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
