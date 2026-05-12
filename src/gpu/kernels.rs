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
    target_x: &mut CudaSlice<f32>,
    target_y: &mut CudaSlice<f32>,
    target_z: &mut CudaSlice<f32>,
    particle_count: usize,
) -> Result<(), GpuError> {
    let n = particle_count;
    if n == 0 {
        return Ok(());
    }
    let max_neighbors = pair_buffer.max_neighbors();
    debug_assert_eq!(pair_buffer.particle_count(), n);
    debug_assert_eq!(neighbor_counts.len(), n);
    debug_assert_eq!(target_x.len(), n);
    debug_assert_eq!(target_y.len(), n);
    debug_assert_eq!(target_z.len(), n);
    debug_assert_eq!(
        pair_buffer.pair_forces_x.len(),
        n * max_neighbors as usize
    );

    let n_u32 = n as u32;
    let func = pair_buffer
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
                target_x,
                target_y,
                target_z,
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
    atom_excl_offsets: &CudaSlice<u32>,
    atom_excl_partners: &CudaSlice<u32>,
    atom_excl_scales: &CudaSlice<f32>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(pair_buffer.particle_count(), n);
    debug_assert!(pair_buffer.max_neighbors() as usize >= n);
    debug_assert_eq!(atom_excl_offsets.len(), n + 1);
    debug_assert_eq!(atom_excl_partners.len(), atom_excl_scales.len());

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
                atom_excl_offsets,
                atom_excl_partners,
                atom_excl_scales,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-d46a89d5
#[allow(clippy::too_many_arguments)]
pub fn lj_pair_force_neighbor(
    particle_buffers: &ParticleBuffers,
    pair_buffer: &mut PairBuffer,
    sim_box: &SimulationBox,
    params: &LennardJonesParameters,
    atom_excl_offsets: &CudaSlice<u32>,
    atom_excl_partners: &CudaSlice<u32>,
    atom_excl_scales: &CudaSlice<f32>,
    neighbor_list: &CudaSlice<u32>,
    neighbor_counts: &CudaSlice<u32>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(pair_buffer.particle_count(), n);
    let max_neighbors = pair_buffer.max_neighbors();
    debug_assert_eq!(neighbor_list.len(), n * max_neighbors as usize);
    debug_assert_eq!(neighbor_counts.len(), n);
    debug_assert_eq!(atom_excl_offsets.len(), n + 1);
    debug_assert_eq!(atom_excl_partners.len(), atom_excl_scales.len());

    let n_u32 = n as u32;
    let func = particle_buffers
        .device
        .get_func("pair_force", "lj_pair_force_neighbor")
        .expect("pair_force module is not loaded; init_device() must be called first");

    let grid_y = n_u32.div_ceil(16);
    let grid_x = max_neighbors.div_ceil(16).max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
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
                atom_excl_offsets,
                atom_excl_partners,
                atom_excl_scales,
                neighbor_list,
                neighbor_counts,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-f00f729e (morse_bond_force launcher mirroring the `gpu` convention)
#[allow(clippy::too_many_arguments)]
pub fn morse_bond_force(
    particle_buffers: &ParticleBuffers,
    bonds: &CudaSlice<u32>,
    bond_de: &CudaSlice<f32>,
    bond_a: &CudaSlice<f32>,
    bond_re: &CudaSlice<f32>,
    sim_box: &SimulationBox,
    bond_pair_x: &mut CudaSlice<f32>,
    bond_pair_y: &mut CudaSlice<f32>,
    bond_pair_z: &mut CudaSlice<f32>,
    n_bonds: usize,
) -> Result<(), GpuError> {
    if n_bonds == 0 {
        return Ok(());
    }
    let n_u32 = n_bonds as u32;
    let func = particle_buffers
        .device
        .get_func("morse", "morse_bond_force")
        .expect("morse module is not loaded; init_device() must be called first");
    let cfg = launch_config(n_u32);
    let lengths = sim_box.lengths();
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.positions_x,
                &particle_buffers.positions_y,
                &particle_buffers.positions_z,
                bonds,
                bond_de,
                bond_a,
                bond_re,
                lengths[0],
                lengths[1],
                lengths[2],
                bond_pair_x,
                bond_pair_y,
                bond_pair_z,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-6435723d (well, that was Langevin's id; this is the bond reduction —
// using a fresh id-comment in the spec rqm-bond reduction declaration.)
#[allow(clippy::too_many_arguments)]
pub fn reduce_bond_forces(
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    bond_pair_x: &CudaSlice<f32>,
    bond_pair_y: &CudaSlice<f32>,
    bond_pair_z: &CudaSlice<f32>,
    atom_bond_offsets: &CudaSlice<u32>,
    atom_bond_indices: &CudaSlice<u32>,
    accumulator_x: &mut CudaSlice<f32>,
    accumulator_y: &mut CudaSlice<f32>,
    accumulator_z: &mut CudaSlice<f32>,
    particle_count: usize,
) -> Result<(), GpuError> {
    if particle_count == 0 {
        return Ok(());
    }
    let n_u32 = particle_count as u32;
    let func = device
        .get_func("morse", "reduce_bond_forces")
        .expect("morse module is not loaded; init_device() must be called first");
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(
            cfg,
            (
                bond_pair_x,
                bond_pair_y,
                bond_pair_z,
                atom_bond_offsets,
                atom_bond_indices,
                accumulator_x,
                accumulator_y,
                accumulator_z,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn accumulate_forces(
    particle_buffers: &mut ParticleBuffers,
    slot0: Option<(&CudaSlice<f32>, &CudaSlice<f32>, &CudaSlice<f32>)>,
    slot1: Option<(&CudaSlice<f32>, &CudaSlice<f32>, &CudaSlice<f32>)>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    let func = particle_buffers
        .device
        .get_func("forces", "accumulate_forces")
        .expect("forces module is not loaded; init_device() must be called first");
    let cfg = launch_config(n_u32);

    // Slot pointers default to slot0 (Lennard-Jones) when absent, since the
    // bitmask blocks dereferencing of empty slots anyway. cudarc requires
    // valid CudaSlice references for kernel arguments.
    let (s0x, s0y, s0z) = slot0.ok_or_else(|| GpuError::from(cudarc::driver::DriverError(
        cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
    )))?;
    let (s1x, s1y, s1z) = slot1.unwrap_or((s0x, s0y, s0z));
    let mut bitmask: u32 = 0;
    bitmask |= 1;
    if slot1.is_some() {
        bitmask |= 2;
    }
    let n_slots: u32 = if slot1.is_some() { 2 } else { 1 };

    unsafe {
        func.launch(
            cfg,
            (
                s0x,
                s0y,
                s0z,
                s1x,
                s1y,
                s1z,
                n_slots,
                bitmask,
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

// rq-884b5cd6
pub fn neighbor_displacement_squared(
    particle_buffers: &ParticleBuffers,
    reference_x: &CudaSlice<f32>,
    reference_y: &CudaSlice<f32>,
    reference_z: &CudaSlice<f32>,
    sim_box: &SimulationBox,
    disp_sq: &mut CudaSlice<f32>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(reference_x.len(), n);
    debug_assert_eq!(reference_y.len(), n);
    debug_assert_eq!(reference_z.len(), n);
    debug_assert_eq!(disp_sq.len(), n);
    let n_u32 = n as u32;
    let func = particle_buffers
        .device
        .get_func("neighbor", "neighbor_displacement_squared")
        .expect("neighbor module is not loaded; init_device() must be called first");
    let cfg = launch_config(n_u32);
    let lengths = sim_box.lengths();
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.positions_x,
                &particle_buffers.positions_y,
                &particle_buffers.positions_z,
                reference_x,
                reference_y,
                reference_z,
                lengths[0],
                lengths[1],
                lengths[2],
                disp_sq,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-a1262872
#[allow(clippy::too_many_arguments)]
pub fn neighbor_list_build(
    particle_buffers: &ParticleBuffers,
    sorted_particle_ids: &CudaSlice<u32>,
    cell_offsets: &CudaSlice<u32>,
    sim_box: &SimulationBox,
    n_cells: [u32; 3],
    cell_size: [f32; 3],
    r_search_sq: f32,
    max_neighbors: u32,
    neighbor_list: &mut CudaSlice<u32>,
    neighbor_counts: &mut CudaSlice<u32>,
    overflow_flag: &mut CudaSlice<u32>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(sorted_particle_ids.len(), n);
    debug_assert_eq!(
        cell_offsets.len(),
        (n_cells[0] * n_cells[1] * n_cells[2]) as usize + 1
    );
    debug_assert_eq!(neighbor_list.len(), n * max_neighbors as usize);
    debug_assert_eq!(neighbor_counts.len(), n);
    debug_assert_eq!(overflow_flag.len(), 1);

    let n_u32 = n as u32;
    let func = particle_buffers
        .device
        .get_func("neighbor", "neighbor_list_build")
        .expect("neighbor module is not loaded; init_device() must be called first");
    let cfg = launch_config(n_u32);
    let lengths = sim_box.lengths();
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.positions_x,
                &particle_buffers.positions_y,
                &particle_buffers.positions_z,
                sorted_particle_ids,
                cell_offsets,
                lengths[0],
                lengths[1],
                lengths[2],
                cell_size[0],
                cell_size[1],
                cell_size[2],
                n_cells[0],
                n_cells[1],
                n_cells[2],
                r_search_sq,
                max_neighbors,
                neighbor_list,
                neighbor_counts,
                overflow_flag,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-344f7af0
pub fn copy_positions_into_reference(
    particle_buffers: &ParticleBuffers,
    reference_x: &mut CudaSlice<f32>,
    reference_y: &mut CudaSlice<f32>,
    reference_z: &mut CudaSlice<f32>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(reference_x.len(), n);
    debug_assert_eq!(reference_y.len(), n);
    debug_assert_eq!(reference_z.len(), n);
    let n_u32 = n as u32;
    let func = particle_buffers
        .device
        .get_func("neighbor", "copy_positions_into_reference")
        .expect("neighbor module is not loaded; init_device() must be called first");
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.positions_x,
                &particle_buffers.positions_y,
                &particle_buffers.positions_z,
                reference_x,
                reference_y,
                reference_z,
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

// rq-f00f729e
pub fn lan_drift_half(
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
        .get_func("langevin", "lan_drift_half")
        .expect("langevin module is not loaded; init_device() must be called first");
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(
            cfg,
            (
                &mut buffers.positions_x,
                &mut buffers.positions_y,
                &mut buffers.positions_z,
                &buffers.velocities_x,
                &buffers.velocities_y,
                &buffers.velocities_z,
                dt,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-6435723d
pub fn lan_ou_step(
    buffers: &mut ParticleBuffers,
    seed: u64,
    step_index: u64,
    alpha: f32,
    kt: f32,
) -> Result<(), GpuError> {
    let n = buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    let func = buffers
        .device
        .get_func("langevin", "lan_ou_step")
        .expect("langevin module is not loaded; init_device() must be called first");
    let cfg = launch_config(n_u32);
    let seed_lo = (seed & 0xFFFF_FFFF) as u32;
    let seed_hi = (seed >> 32) as u32;
    let step_lo = (step_index & 0xFFFF_FFFF) as u32;
    let step_hi = (step_index >> 32) as u32;
    unsafe {
        func.launch(
            cfg,
            (
                &mut buffers.velocities_x,
                &mut buffers.velocities_y,
                &mut buffers.velocities_z,
                &buffers.masses,
                &buffers.particle_ids,
                seed_lo,
                seed_hi,
                step_lo,
                step_hi,
                alpha,
                kt,
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
