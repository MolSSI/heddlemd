use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, CudaViewMut, DeviceSlice, LaunchAsync, LaunchConfig};

use crate::gpu::{GpuError, Kernels, LosslessBuffers, PairBuffer, ParticleBuffers};
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
    let func = buffers.kernels.vv_kick_drift.clone();
    let cfg = launch_config(n_u32);
    let lat = sim_box.lattice();
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
                lat[0],
                lat[1],
                lat[2],
                lat[3],
                lat[4],
                lat[5],
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
    let func = buffers.kernels.vv_kick.clone();
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
    let func = pair_buffer.kernels.reduce_pair_forces.clone();
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
    atom_excl_lj_scales: &CudaSlice<f32>,
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
    debug_assert_eq!(atom_excl_partners.len(), atom_excl_lj_scales.len());
    let table_len = params.n_types as usize * params.n_types as usize;
    debug_assert_eq!(params.sigma.len(), table_len);
    debug_assert_eq!(params.epsilon.len(), table_len);
    debug_assert_eq!(params.cutoff.len(), table_len);
    debug_assert_eq!(params.switch.len(), table_len);

    let n_u32 = n as u32;
    let func = particle_buffers.kernels.lj_pair_force.clone();

    let grid_y = n_u32.div_ceil(16);
    let grid_x = max_neighbors.div_ceil(16).max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    let lat = sim_box.lattice();
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
                lat[0],
                lat[1],
                lat[2],
                lat[3],
                lat[4],
                lat[5],
                params.n_types,
                &params.sigma,
                &params.epsilon,
                &params.cutoff,
                &params.switch,
                atom_excl_offsets,
                atom_excl_partners,
                atom_excl_lj_scales,
                neighbor_list,
                neighbor_counts,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

/// Coulomb constant `k_C = 1 / (4 π ε₀)` in SI units (N·m²/C²),
/// rounded to f32. See `forces/coulomb-pair-force.md`. rq-bfd7004c
pub const K_COULOMB_F32: f32 = 8.987_551_787e9_f32;

// rq-846bdb8b
#[allow(clippy::too_many_arguments)]
pub fn coulomb_pair_force(
    particle_buffers: &ParticleBuffers,
    pair_buffer: &mut PairBuffer,
    sim_box: &SimulationBox,
    cutoff: f32,
    r_switch: f32,
    atom_excl_offsets: &CudaSlice<u32>,
    atom_excl_partners: &CudaSlice<u32>,
    atom_excl_coul_scales: &CudaSlice<f32>,
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
    debug_assert_eq!(atom_excl_partners.len(), atom_excl_coul_scales.len());
    debug_assert_eq!(particle_buffers.charges.len(), n);

    let n_u32 = n as u32;
    let func = particle_buffers.kernels.coulomb_pair_force.clone();

    let grid_y = n_u32.div_ceil(16);
    let grid_x = max_neighbors.div_ceil(16).max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    let lat = sim_box.lattice();
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.positions_x,
                &particle_buffers.positions_y,
                &particle_buffers.positions_z,
                &particle_buffers.charges,
                &mut pair_buffer.pair_forces_x,
                &mut pair_buffer.pair_forces_y,
                &mut pair_buffer.pair_forces_z,
                &mut pair_buffer.pair_energies,
                &mut pair_buffer.pair_virials,
                max_neighbors,
                lat[0],
                lat[1],
                lat[2],
                lat[3],
                lat[4],
                lat[5],
                K_COULOMB_F32,
                cutoff,
                r_switch,
                atom_excl_offsets,
                atom_excl_partners,
                atom_excl_coul_scales,
                neighbor_list,
                neighbor_counts,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-9a512ed1 rq-f6d45062
#[allow(clippy::too_many_arguments)]
pub fn spme_real_pair_force(
    particle_buffers: &ParticleBuffers,
    pair_buffer: &mut PairBuffer,
    sim_box: &SimulationBox,
    alpha: f32,
    r_cut_real: f32,
    atom_excl_offsets: &CudaSlice<u32>,
    atom_excl_partners: &CudaSlice<u32>,
    atom_excl_coul_scales: &CudaSlice<f32>,
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
    debug_assert_eq!(atom_excl_partners.len(), atom_excl_coul_scales.len());
    debug_assert_eq!(particle_buffers.charges.len(), n);

    let n_u32 = n as u32;
    let func = particle_buffers.kernels.spme_real_pair_force.clone();

    let grid_y = n_u32.div_ceil(16);
    let grid_x = max_neighbors.div_ceil(16).max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    let lat = sim_box.lattice();
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.positions_x,
                &particle_buffers.positions_y,
                &particle_buffers.positions_z,
                &particle_buffers.charges,
                &mut pair_buffer.pair_forces_x,
                &mut pair_buffer.pair_forces_y,
                &mut pair_buffer.pair_forces_z,
                &mut pair_buffer.pair_energies,
                &mut pair_buffer.pair_virials,
                max_neighbors,
                lat[0],
                lat[1],
                lat[2],
                lat[3],
                lat[4],
                lat[5],
                K_COULOMB_F32,
                alpha,
                r_cut_real,
                atom_excl_offsets,
                atom_excl_partners,
                atom_excl_coul_scales,
                neighbor_list,
                neighbor_counts,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-9ca00d25 rq-202493a5
#[allow(clippy::too_many_arguments)]
pub fn spme_charge_spread(
    particle_buffers: &ParticleBuffers,
    sim_box: &SimulationBox,
    sorted_particle_ids: &CudaSlice<u32>,
    cell_offsets: &CudaSlice<u32>,
    grid: [u32; 3],
    spline_order: u32,
    rho: &mut CudaSlice<f32>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    let n_a = grid[0];
    let n_b = grid[1];
    let n_c = grid[2];
    let m = n_a as usize * n_b as usize * n_c as usize;
    debug_assert_eq!(rho.len(), m);
    debug_assert_eq!(cell_offsets.len(), m + 1);
    debug_assert_eq!(sorted_particle_ids.len(), n.max(1));
    debug_assert_eq!(particle_buffers.charges.len(), n);

    let m_u32 = m as u32;
    let func = particle_buffers.kernels.spme_charge_spread.clone();
    let cfg = launch_config(m_u32);
    let lat = sim_box.lattice();
    let n_u32 = n as u32;
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.positions_x,
                &particle_buffers.positions_y,
                &particle_buffers.positions_z,
                &particle_buffers.charges,
                sorted_particle_ids,
                cell_offsets,
                lat[0],
                lat[1],
                lat[2],
                lat[3],
                lat[4],
                lat[5],
                n_a,
                n_b,
                n_c,
                spline_order,
                rho,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-9ca00d25
#[allow(clippy::too_many_arguments)]
pub fn spme_influence_multiply(
    kernels: &Kernels,
    influence_g: &CudaSlice<f32>,
    virial_factor: &CudaSlice<f32>,
    rho_hat_interleaved: &mut CudaSlice<f32>,
    virial_per_cell: &mut CudaSlice<f32>,
    n_c: u32,
    n_c_complex: u32,
    n_complex: u32,
) -> Result<(), GpuError> {
    if n_complex == 0 {
        return Ok(());
    }
    debug_assert_eq!(influence_g.len(), n_complex as usize);
    debug_assert_eq!(virial_factor.len(), n_complex as usize);
    debug_assert_eq!(rho_hat_interleaved.len(), 2 * n_complex as usize);
    debug_assert_eq!(virial_per_cell.len(), n_complex as usize);
    let func = kernels.spme_influence_multiply.clone();
    let cfg = launch_config(n_complex);
    unsafe {
        func.launch(
            cfg,
            (
                influence_g,
                virial_factor,
                rho_hat_interleaved,
                virial_per_cell,
                n_c,
                n_c_complex,
                n_complex,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-9ca00d25
#[allow(clippy::too_many_arguments)]
pub fn spme_force_gather(
    particle_buffers: &ParticleBuffers,
    sim_box: &SimulationBox,
    v: &CudaSlice<f32>,
    u_self_per_particle: &CudaSlice<f32>,
    w_per_particle_virial: f32,
    grid: [u32; 3],
    spline_order: u32,
    slot_force_x: &mut CudaViewMut<'_, f32>,
    slot_force_y: &mut CudaViewMut<'_, f32>,
    slot_force_z: &mut CudaViewMut<'_, f32>,
    slot_energy: &mut CudaViewMut<'_, f32>,
    slot_virial: &mut CudaViewMut<'_, f32>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let m =
        grid[0] as usize * grid[1] as usize * grid[2] as usize;
    debug_assert_eq!(v.len(), m);
    debug_assert_eq!(particle_buffers.charges.len(), n);
    debug_assert_eq!(u_self_per_particle.len(), n);
    debug_assert_eq!(slot_force_x.len(), n);
    debug_assert_eq!(slot_force_y.len(), n);
    debug_assert_eq!(slot_force_z.len(), n);
    debug_assert_eq!(slot_energy.len(), n);
    debug_assert_eq!(slot_virial.len(), n);

    let n_u32 = n as u32;
    let func = particle_buffers.kernels.spme_force_gather.clone();
    let cfg = launch_config(n_u32);
    let lat = sim_box.lattice();
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.positions_x,
                &particle_buffers.positions_y,
                &particle_buffers.positions_z,
                &particle_buffers.charges,
                v,
                u_self_per_particle,
                w_per_particle_virial,
                lat[0],
                lat[1],
                lat[2],
                lat[3],
                lat[4],
                lat[5],
                grid[0],
                grid[1],
                grid[2],
                spline_order,
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
    let func = particle_buffers.kernels.morse_bond_force.clone();
    let cfg = launch_config(n_u32);
    let lat = sim_box.lattice();
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
                lat[0],
                lat[1],
                lat[2],
                lat[3],
                lat[4],
                lat[5],
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
    kernels: &Kernels,
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
    let func = kernels.reduce_bond_forces.clone();
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

    let func = particle_buffers.kernels.accumulate_forces.clone();
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
    let func = particle_buffers.kernels.neighbor_displacement_squared.clone();
    let cfg = launch_config(n_u32);
    let lat = sim_box.lattice();
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
                lat[0],
                lat[1],
                lat[2],
                lat[3],
                lat[4],
                lat[5],
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
    let func = particle_buffers.kernels.neighbor_list_build.clone();
    let cfg = launch_config(n_u32);
    let lat = sim_box.lattice();
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.positions_x,
                &particle_buffers.positions_y,
                &particle_buffers.positions_z,
                sorted_particle_ids,
                cell_offsets,
                lat[0],
                lat[1],
                lat[2],
                lat[3],
                lat[4],
                lat[5],
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
    let func = particle_buffers.kernels.copy_positions_into_reference.clone();
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

#[allow(clippy::too_many_arguments)]
pub fn compute_cell_indices_and_histogram(
    particle_buffers: &ParticleBuffers,
    sim_box: &SimulationBox,
    n_cells: [u32; 3],
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
        .kernels
        .compute_cell_indices_and_histogram
        .clone();
    let cfg = launch_config(n_u32);
    let lat = sim_box.lattice();
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.positions_x,
                &particle_buffers.positions_y,
                &particle_buffers.positions_z,
                lat[0],
                lat[1],
                lat[2],
                lat[3],
                lat[4],
                lat[5],
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

// rq-2ef5e222
//
// Drives the recursive multi-level exclusive prefix scan of `cell_counts`
// into `cell_offsets`. `scan_block_totals` is the block-totals stack:
// buffer `l` holds the per-block inclusive totals produced at recursion
// level `l`, with length `ceil(n_cells_total / B^(l + 1))`; the last
// buffer has length 1. The driver:
//   1. scans `cell_counts` into `cell_offsets` with a per-block local
//      scan, emitting `scan_block_totals[0]`;
//   2. descends, scanning each stack level in place (input aliases
//      output) and emitting the next level's totals;
//   3. ascends, adding each level's scanned totals back into the level
//      below;
//   4. writes the `cell_offsets[n_cells_total] = particle_count`
//      sentinel.
// Issues O(log(n_cells_total)) kernel launches.
pub fn prefix_scan_cell_counts(
    kernels: &Kernels,
    cell_counts: &CudaSlice<u32>,
    cell_offsets: &mut CudaSlice<u32>,
    scan_block_totals: &mut [CudaSlice<u32>],
    n_cells_total: usize,
    particle_count: usize,
) -> Result<(), GpuError> {
    if n_cells_total == 0 {
        return Ok(());
    }
    debug_assert_eq!(cell_counts.len(), n_cells_total);
    debug_assert_eq!(cell_offsets.len(), n_cells_total + 1);
    debug_assert!(!scan_block_totals.is_empty());

    let n_cells_total_u32 = n_cells_total as u32;

    // Phase 1: per-block local scan of cell_counts into cell_offsets,
    // emitting the level-0 block totals.
    {
        let func = kernels.prefix_scan_local_blocks.clone();
        unsafe {
            func.launch(
                launch_config(n_cells_total_u32),
                (
                    cell_counts,
                    &mut *cell_offsets,
                    &mut scan_block_totals[0],
                    n_cells_total_u32,
                ),
            )
            .map_err(GpuError::from)?;
        }
    }

    // The stack's last buffer has length 1 and is never itself scanned;
    // every earlier level spans more than one block and is scanned in
    // place during the descent.
    let descent_levels = scan_block_totals.len() - 1;

    // Phase 2: descend — scan each stack level in place. Level `l` reads
    // and writes `scan_block_totals[l]` (the kernel reads each input
    // element before any write, so aliasing is safe) and emits the
    // level-`l + 1` totals.
    for l in 0..descent_levels {
        let len = scan_block_totals[l].len() as u32;
        let (head, tail) = scan_block_totals.split_at_mut(l + 1);
        let level = &head[l];
        let totals = &mut tail[0];
        let func = kernels.prefix_scan_local_blocks.clone();
        unsafe {
            func.launch(launch_config(len), (level, level, totals, len))
                .map_err(GpuError::from)?;
        }
    }

    // Phase 3: ascend — add each scanned level's totals back into the
    // level below (level 0's target is `cell_offsets`).
    for l in (0..descent_levels).rev() {
        let func = kernels.prefix_scan_apply_block_totals.clone();
        if l == 0 {
            unsafe {
                func.launch(
                    launch_config(n_cells_total_u32),
                    (&scan_block_totals[0], &mut *cell_offsets, n_cells_total_u32),
                )
                .map_err(GpuError::from)?;
            }
        } else {
            let len = scan_block_totals[l - 1].len() as u32;
            let (head, tail) = scan_block_totals.split_at_mut(l);
            let output = &mut head[l - 1];
            let block_offsets = &tail[0];
            unsafe {
                func.launch(launch_config(len), (block_offsets, output, len))
                    .map_err(GpuError::from)?;
            }
        }
    }

    // Phase 4: write the trailing cell_offsets[n_cells_total] sentinel.
    {
        let func = kernels.prefix_scan_finalize_offsets.clone();
        unsafe {
            func.launch(
                launch_config(1),
                (&mut *cell_offsets, n_cells_total_u32, particle_count as u32),
            )
            .map_err(GpuError::from)?;
        }
    }
    Ok(())
}

pub fn scatter_atoms_into_cells(
    device: &Arc<CudaDevice>,
    kernels: &Kernels,
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
    let func = kernels.scatter_atoms_into_cells.clone();
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(cfg, (cell_indices, cell_offsets, write_cursors, sorted_particle_ids, n_u32))
            .map_err(GpuError::from)?;
    }
    Ok(())
}

pub fn sort_cells_by_particle_id(
    kernels: &Kernels,
    cell_offsets: &CudaSlice<u32>,
    sorted_particle_ids: &mut CudaSlice<u32>,
    n_cells_total: usize,
) -> Result<(), GpuError> {
    if n_cells_total == 0 {
        return Ok(());
    }
    debug_assert_eq!(cell_offsets.len(), n_cells_total + 1);
    let n_u32 = n_cells_total as u32;
    let func = kernels.sort_cells_by_particle_id.clone();
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
    let func = buffers.kernels.vv_kick_drift_lossless.clone();
    let cfg = launch_config(n_u32);
    let lat = sim_box.lattice();
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
                lat[0],
                lat[1],
                lat[2],
                lat[3],
                lat[4],
                lat[5],
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
    let func = buffers.kernels.lan_drift_half.clone();
    let cfg = launch_config(n_u32);
    let lat = sim_box.lattice();
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
                lat[0],
                lat[1],
                lat[2],
                lat[3],
                lat[4],
                lat[5],
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
    let func = buffers.kernels.lan_ou_step.clone();
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
    let func = buffers.kernels.vv_kick_lossless.clone();
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
