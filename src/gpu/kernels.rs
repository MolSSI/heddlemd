use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, CudaViewMut, DeviceSlice, LaunchAsync, LaunchConfig};

use crate::gpu::{GpuError, LosslessBuffers, PairBuffer, ParticleBuffers};
use crate::io::config::{PairInteractionConfig, PairPotentialParams, ParticleTypeConfig};
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
    sim_box: &SimulationBox,
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
    let lengths = sim_box.lengths();
    unsafe {
        func.launch(
            cfg,
            (
                &mut buffers.positions_x,
                &mut buffers.positions_y,
                &mut buffers.positions_z,
                &mut buffers.images_x,
                &mut buffers.images_y,
                &mut buffers.images_z,
                &mut buffers.velocities_x,
                &mut buffers.velocities_y,
                &mut buffers.velocities_z,
                &buffers.forces_x,
                &buffers.forces_y,
                &buffers.forces_z,
                &buffers.masses,
                lengths[0],
                lengths[1],
                lengths[2],
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
#[allow(clippy::too_many_arguments)]
pub fn reduce_pair_forces(
    pair_buffer: &PairBuffer,
    neighbor_counts: &CudaSlice<u32>,
    target_force_x: &mut CudaViewMut<'_, f32>,
    target_force_y: &mut CudaViewMut<'_, f32>,
    target_force_z: &mut CudaViewMut<'_, f32>,
    target_energy: &mut CudaViewMut<'_, f32>,
    target_virial: &mut CudaViewMut<'_, f32>,
    particle_count: usize,
) -> Result<(), GpuError> {
    let n = particle_count;
    if n == 0 {
        return Ok(());
    }
    let max_neighbors = pair_buffer.max_neighbors();
    debug_assert_eq!(pair_buffer.particle_count(), n);
    debug_assert_eq!(neighbor_counts.len(), n);
    debug_assert_eq!(target_force_x.len(), n);
    debug_assert_eq!(target_force_y.len(), n);
    debug_assert_eq!(target_force_z.len(), n);
    debug_assert_eq!(target_energy.len(), n);
    debug_assert_eq!(target_virial.len(), n);
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
                &pair_buffer.pair_energies,
                &pair_buffer.pair_virials,
                neighbor_counts,
                max_neighbors,
                target_force_x,
                target_force_y,
                target_force_z,
                target_energy,
                target_virial,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-dafe0fcb
#[derive(Debug)]
pub struct LennardJonesParameterTable {
    pub n_types: u32,
    pub sigma: CudaSlice<f32>,
    pub epsilon: CudaSlice<f32>,
    pub cutoff: CudaSlice<f32>,
    pub switch: CudaSlice<f32>,
}

impl LennardJonesParameterTable {
    // rq-1adf5954
    pub fn from_config(
        device: &Arc<CudaDevice>,
        particle_types: &[ParticleTypeConfig],
        pair_interactions: &[PairInteractionConfig],
    ) -> Result<Self, GpuError> {
        let n_types = particle_types.len();
        let len = n_types * n_types;
        let mut sigma_host: Vec<f32> = vec![0.0; len];
        let mut epsilon_host: Vec<f32> = vec![0.0; len];
        let mut cutoff_host: Vec<f32> = vec![0.0; len];
        let mut switch_host: Vec<f32> = vec![0.0; len];

        for pi in pair_interactions {
            let ti = particle_types
                .iter()
                .position(|pt| pt.name == pi.between.0)
                .expect("pair_interactions type name absent from particle_types (config-layer invariant)");
            let tj = particle_types
                .iter()
                .position(|pt| pt.name == pi.between.1)
                .expect("pair_interactions type name absent from particle_types (config-layer invariant)");
            let PairPotentialParams::LennardJones { sigma, epsilon } = pi.potential;
            let s = sigma as f32;
            let e = epsilon as f32;
            let c = pi.cutoff as f32;
            let rs = pi.r_switch as f32;
            sigma_host[ti * n_types + tj] = s;
            sigma_host[tj * n_types + ti] = s;
            epsilon_host[ti * n_types + tj] = e;
            epsilon_host[tj * n_types + ti] = e;
            cutoff_host[ti * n_types + tj] = c;
            cutoff_host[tj * n_types + ti] = c;
            switch_host[ti * n_types + tj] = rs;
            switch_host[tj * n_types + ti] = rs;
        }

        let sigma = htod_or_empty_f32(device, &sigma_host)?;
        let epsilon = htod_or_empty_f32(device, &epsilon_host)?;
        let cutoff = htod_or_empty_f32(device, &cutoff_host)?;
        let switch = htod_or_empty_f32(device, &switch_host)?;

        Ok(LennardJonesParameterTable {
            n_types: n_types as u32,
            sigma,
            epsilon,
            cutoff,
            switch,
        })
    }
}

fn htod_or_empty_f32(
    device: &Arc<CudaDevice>,
    data: &[f32],
) -> Result<CudaSlice<f32>, GpuError> {
    if data.is_empty() {
        device.alloc_zeros::<f32>(0).map_err(GpuError::from)
    } else {
        device.htod_sync_copy(data).map_err(GpuError::from)
    }
}

// rq-d3a14184
#[allow(clippy::too_many_arguments)]
pub fn lj_pair_force(
    particle_buffers: &ParticleBuffers,
    pair_buffer: &mut PairBuffer,
    sim_box: &SimulationBox,
    params: &LennardJonesParameterTable,
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
    let table_len = params.n_types as usize * params.n_types as usize;
    debug_assert_eq!(params.sigma.len(), table_len);
    debug_assert_eq!(params.epsilon.len(), table_len);
    debug_assert_eq!(params.cutoff.len(), table_len);
    debug_assert_eq!(params.switch.len(), table_len);

    let n_u32 = n as u32;
    let func = particle_buffers
        .device
        .get_func("pair_force", "lj_pair_force")
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
                &particle_buffers.type_indices,
                &mut pair_buffer.pair_forces_x,
                &mut pair_buffer.pair_forces_y,
                &mut pair_buffer.pair_forces_z,
                &mut pair_buffer.pair_energies,
                &mut pair_buffer.pair_virials,
                max_neighbors,
                lengths[0],
                lengths[1],
                lengths[2],
                params.n_types,
                &params.sigma,
                &params.epsilon,
                &params.cutoff,
                &params.switch,
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
    bond_pair_energy: &mut CudaSlice<f32>,
    bond_pair_virial: &mut CudaSlice<f32>,
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
                bond_pair_energy,
                bond_pair_virial,
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
    bond_pair_energy: &CudaSlice<f32>,
    bond_pair_virial: &CudaSlice<f32>,
    atom_bond_offsets: &CudaSlice<u32>,
    atom_bond_indices: &CudaSlice<u32>,
    slot_force_x: &mut CudaViewMut<'_, f32>,
    slot_force_y: &mut CudaViewMut<'_, f32>,
    slot_force_z: &mut CudaViewMut<'_, f32>,
    slot_energy: &mut CudaViewMut<'_, f32>,
    slot_virial: &mut CudaViewMut<'_, f32>,
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
                bond_pair_energy,
                bond_pair_virial,
                atom_bond_offsets,
                atom_bond_indices,
                slot_force_x,
                slot_force_y,
                slot_force_z,
                slot_energy,
                slot_virial,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-c0f98145
#[allow(clippy::too_many_arguments)]
pub fn accumulate_forces(
    particle_buffers: &mut ParticleBuffers,
    slot_forces_x: &CudaSlice<f32>,
    slot_forces_y: &CudaSlice<f32>,
    slot_forces_z: &CudaSlice<f32>,
    slot_energies: &CudaSlice<f32>,
    slot_virials: &CudaSlice<f32>,
    num_slots: u32,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    debug_assert_eq!(slot_forces_x.len(), num_slots as usize * n);
    debug_assert_eq!(slot_forces_y.len(), num_slots as usize * n);
    debug_assert_eq!(slot_forces_z.len(), num_slots as usize * n);
    debug_assert_eq!(slot_energies.len(), num_slots as usize * n);
    debug_assert_eq!(slot_virials.len(), num_slots as usize * n);

    let func = particle_buffers
        .device
        .get_func("forces", "accumulate_forces")
        .expect("forces module is not loaded; init_device() must be called first");
    let cfg = launch_config(n_u32);

    unsafe {
        func.launch(
            cfg,
            (
                slot_forces_x,
                slot_forces_y,
                slot_forces_z,
                slot_energies,
                slot_virials,
                num_slots,
                &mut particle_buffers.forces_x,
                &mut particle_buffers.forces_y,
                &mut particle_buffers.forces_z,
                &mut particle_buffers.potential_energies,
                &mut particle_buffers.virials,
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

pub const SPATIAL_HASH_SCAN_BLOCK_SIZE: u32 = 256;
pub const SPATIAL_HASH_MAX_CELLS: usize =
    (SPATIAL_HASH_SCAN_BLOCK_SIZE as usize) * (SPATIAL_HASH_SCAN_BLOCK_SIZE as usize);

#[allow(clippy::too_many_arguments)]
pub fn compute_cell_indices_and_histogram(
    particle_buffers: &ParticleBuffers,
    sim_box: &SimulationBox,
    n_cells: [u32; 3],
    cell_size: [f32; 3],
    cell_indices: &mut CudaSlice<u32>,
    cell_counts: &mut CudaSlice<u32>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_cells_total = n_cells[0] as usize * n_cells[1] as usize * n_cells[2] as usize;
    debug_assert_eq!(cell_indices.len(), n);
    debug_assert_eq!(cell_counts.len(), n_cells_total);
    particle_buffers
        .device
        .memset_zeros(cell_counts)
        .map_err(GpuError::from)?;
    let n_u32 = n as u32;
    let func = particle_buffers
        .device
        .get_func("neighbor", "compute_cell_indices_and_histogram")
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
                lengths[0],
                lengths[1],
                lengths[2],
                cell_size[0],
                cell_size[1],
                cell_size[2],
                n_cells[0],
                n_cells[1],
                n_cells[2],
                cell_indices,
                cell_counts,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

pub fn prefix_scan_cell_counts(
    device: &Arc<CudaDevice>,
    cell_counts: &CudaSlice<u32>,
    cell_offsets: &mut CudaSlice<u32>,
    scan_block_totals: &mut CudaSlice<u32>,
    n_cells_total: usize,
    particle_count: usize,
) -> Result<(), GpuError> {
    if n_cells_total == 0 {
        return Ok(());
    }
    let scan_block_size = SPATIAL_HASH_SCAN_BLOCK_SIZE;
    let n_blocks =
        ((n_cells_total + scan_block_size as usize - 1) / scan_block_size as usize) as u32;
    debug_assert_eq!(cell_counts.len(), n_cells_total);
    debug_assert_eq!(cell_offsets.len(), n_cells_total + 1);
    debug_assert_eq!(scan_block_totals.len(), n_blocks as usize);
    debug_assert!(n_blocks <= scan_block_size);

    let local_cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (scan_block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    let local_func = device
        .get_func("neighbor", "prefix_scan_local_blocks")
        .expect("neighbor module is not loaded; init_device() must be called first");
    let n_cells_total_u32 = n_cells_total as u32;
    unsafe {
        local_func
            .launch(
                local_cfg,
                (cell_counts, &mut *cell_offsets, &mut *scan_block_totals, n_cells_total_u32),
            )
            .map_err(GpuError::from)?;
    }

    let totals_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (scan_block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    let totals_func = device
        .get_func("neighbor", "prefix_scan_block_totals")
        .expect("neighbor module is not loaded; init_device() must be called first");
    unsafe {
        totals_func
            .launch(totals_cfg, (&mut *scan_block_totals, n_blocks))
            .map_err(GpuError::from)?;
    }

    let apply_cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (scan_block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    let apply_func = device
        .get_func("neighbor", "prefix_scan_apply_block_totals")
        .expect("neighbor module is not loaded; init_device() must be called first");
    let particle_count_u32 = particle_count as u32;
    unsafe {
        apply_func
            .launch(
                apply_cfg,
                (
                    &*scan_block_totals,
                    &mut *cell_offsets,
                    n_cells_total_u32,
                    particle_count_u32,
                ),
            )
            .map_err(GpuError::from)?;
    }
    Ok(())
}

pub fn scatter_atoms_into_cells(
    device: &Arc<CudaDevice>,
    cell_indices: &CudaSlice<u32>,
    cell_offsets: &CudaSlice<u32>,
    write_cursors: &mut CudaSlice<u32>,
    sorted_particle_ids: &mut CudaSlice<u32>,
    particle_count: usize,
) -> Result<(), GpuError> {
    if particle_count == 0 {
        return Ok(());
    }
    debug_assert_eq!(cell_indices.len(), particle_count);
    debug_assert_eq!(sorted_particle_ids.len(), particle_count);
    device.memset_zeros(write_cursors).map_err(GpuError::from)?;
    let n_u32 = particle_count as u32;
    let func = device
        .get_func("neighbor", "scatter_atoms_into_cells")
        .expect("neighbor module is not loaded; init_device() must be called first");
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(cfg, (cell_indices, cell_offsets, write_cursors, sorted_particle_ids, n_u32))
            .map_err(GpuError::from)?;
    }
    Ok(())
}

pub fn sort_cells_by_particle_id(
    device: &Arc<CudaDevice>,
    cell_offsets: &CudaSlice<u32>,
    sorted_particle_ids: &mut CudaSlice<u32>,
    n_cells_total: usize,
) -> Result<(), GpuError> {
    if n_cells_total == 0 {
        return Ok(());
    }
    debug_assert_eq!(cell_offsets.len(), n_cells_total + 1);
    let n_u32 = n_cells_total as u32;
    let func = device
        .get_func("neighbor", "sort_cells_by_particle_id")
        .expect("neighbor module is not loaded; init_device() must be called first");
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(cfg, (cell_offsets, sorted_particle_ids, n_u32))
            .map_err(GpuError::from)?;
    }
    Ok(())
}

pub fn vv_kick_drift_lossless(
    buffers: &mut ParticleBuffers,
    lossless: &mut LosslessBuffers,
    sim_box: &SimulationBox,
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
    let lengths = sim_box.lengths();
    unsafe {
        func.launch(
            cfg,
            (
                &mut buffers.positions_x,
                &mut buffers.positions_y,
                &mut buffers.positions_z,
                &mut buffers.images_x,
                &mut buffers.images_y,
                &mut buffers.images_z,
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
                lengths[0],
                lengths[1],
                lengths[2],
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
    sim_box: &SimulationBox,
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
    let lengths = sim_box.lengths();
    unsafe {
        func.launch(
            cfg,
            (
                &mut buffers.positions_x,
                &mut buffers.positions_y,
                &mut buffers.positions_z,
                &mut buffers.images_x,
                &mut buffers.images_y,
                &mut buffers.images_z,
                &buffers.velocities_x,
                &buffers.velocities_y,
                &buffers.velocities_z,
                lengths[0],
                lengths[1],
                lengths[2],
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
    draw_counter: u64,
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
    let draw_lo = (draw_counter & 0xFFFF_FFFF) as u32;
    let draw_hi = (draw_counter >> 32) as u32;
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
                draw_lo,
                draw_hi,
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
