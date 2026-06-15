use std::sync::Arc;

use cudarc::driver::{
    CudaDevice, CudaSlice, CudaStream, CudaViewMut, DeviceSlice, LaunchAsync, LaunchConfig,
};

#[cfg(not(feature = "f64"))]
use crate::gpu::LosslessBuffers;
use crate::gpu::{GpuError, Kernels, PairBuffer, ParticleBuffers};
use crate::io::config::{PairInteractionConfig, PairPotentialParams, ParticleTypeConfig};
use crate::pbc::SimulationBox;
use crate::precision::Real;

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
    dt: Real,
) -> Result<(), GpuError> {
    let n = buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    let func = buffers.kernels.integrate.vv_kick_drift.clone();
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
pub fn vv_kick(buffers: &mut ParticleBuffers, dt: Real) -> Result<(), GpuError> {
    let n = buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    let func = buffers.kernels.integrate.vv_kick.clone();
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
    target_force_x: &mut CudaViewMut<'_, Real>,
    target_force_y: &mut CudaViewMut<'_, Real>,
    target_force_z: &mut CudaViewMut<'_, Real>,
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
    debug_assert_eq!(
        pair_buffer.pair_forces_x.len(),
        n * max_neighbors as usize
    );

    let n_u32 = n as u32;
    let func = pair_buffer.kernels.reduce.reduce_pair_forces.clone();
    let cfg = LaunchConfig {
        grid_dim: (n_u32, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(
            cfg,
            (
                &pair_buffer.pair_forces_x,
                &pair_buffer.pair_forces_y,
                &pair_buffer.pair_forces_z,
                neighbor_counts,
                max_neighbors,
                target_force_x,
                target_force_y,
                target_force_z,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-c9240ed4
pub fn reduce_pair_energy_virial(
    pair_buffer: &PairBuffer,
    neighbor_counts: &CudaSlice<u32>,
    target_energy: &mut CudaViewMut<'_, Real>,
    target_virial: &mut CudaViewMut<'_, Real>,
    particle_count: usize,
) -> Result<(), GpuError> {
    let n = particle_count;
    if n == 0 {
        return Ok(());
    }
    let max_neighbors = pair_buffer.max_neighbors();
    debug_assert_eq!(pair_buffer.particle_count(), n);
    debug_assert_eq!(neighbor_counts.len(), n);
    debug_assert_eq!(target_energy.len(), n);
    debug_assert_eq!(target_virial.len(), n);
    debug_assert_eq!(
        pair_buffer.pair_energies.len(),
        n * max_neighbors as usize
    );

    let n_u32 = n as u32;
    let func = pair_buffer
        .kernels
        .reduce
        .reduce_pair_energy_virial
        .clone();
    let cfg = LaunchConfig {
        grid_dim: (n_u32, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(
            cfg,
            (
                &pair_buffer.pair_energies,
                &pair_buffer.pair_virials,
                neighbor_counts,
                max_neighbors,
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
    pub sigma: CudaSlice<Real>,
    pub epsilon: CudaSlice<Real>,
    pub cutoff: CudaSlice<Real>,
    pub switch: CudaSlice<Real>,
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
        let mut sigma_host: Vec<Real> = vec![0.0; len];
        let mut epsilon_host: Vec<Real> = vec![0.0; len];
        let mut cutoff_host: Vec<Real> = vec![0.0; len];
        let mut switch_host: Vec<Real> = vec![0.0; len];

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
            let s = sigma as Real;
            let e = epsilon as Real;
            let c = pi.cutoff as Real;
            let rs = pi.r_switch as Real;
            sigma_host[ti * n_types + tj] = s;
            sigma_host[tj * n_types + ti] = s;
            epsilon_host[ti * n_types + tj] = e;
            epsilon_host[tj * n_types + ti] = e;
            cutoff_host[ti * n_types + tj] = c;
            cutoff_host[tj * n_types + ti] = c;
            switch_host[ti * n_types + tj] = rs;
            switch_host[tj * n_types + ti] = rs;
        }

        let sigma = htod_or_empty(device, &sigma_host)?;
        let epsilon = htod_or_empty(device, &epsilon_host)?;
        let cutoff = htod_or_empty(device, &cutoff_host)?;
        let switch = htod_or_empty(device, &switch_host)?;

        Ok(LennardJonesParameterTable {
            n_types: n_types as u32,
            sigma,
            epsilon,
            cutoff,
            switch,
        })
    }
}

fn htod_or_empty(
    device: &Arc<CudaDevice>,
    data: &[Real],
) -> Result<CudaSlice<Real>, GpuError> {
    if data.is_empty() {
        device.alloc_zeros::<Real>(0).map_err(GpuError::from)
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
    atom_excl_lj_scales: &CudaSlice<Real>,
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
    let func = particle_buffers.kernels.lj.pair_force.clone();

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

/// Coulomb prefactor `k_C = 1 / (4 π ε₀)`. In the engine's internal
/// Hartree atomic units `k_C = 1` exactly, so no permittivity factor
/// appears in the pair-force or SPME kernels. See
/// `forces/coulomb-pair-force.md`. rq-bfd7004c
pub const K_COULOMB_F32: Real = 1.0;

// rq-846bdb8b rq-38676211
#[allow(clippy::too_many_arguments)]
pub fn coulomb_pair_force(
    particle_buffers: &ParticleBuffers,
    pair_buffer: &mut PairBuffer,
    sim_box: &SimulationBox,
    cutoff: Real,
    r_switch: Real,
    atom_excl_offsets: &CudaSlice<u32>,
    atom_excl_partners: &CudaSlice<u32>,
    atom_excl_coul_scales: &CudaSlice<Real>,
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
    let func = particle_buffers.kernels.coulomb.coulomb_pair_force.clone();

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

// rq-9a512ed1 rq-f6d45062 rq-44cce069 rq-eb9e5cc3 rq-f735ea05
#[allow(clippy::too_many_arguments)]
pub fn spme_real_pair_force(
    particle_buffers: &ParticleBuffers,
    pair_buffer: &mut PairBuffer,
    sim_box: &SimulationBox,
    alpha: Real,
    r_cut_real: Real,
    atom_excl_offsets: &CudaSlice<u32>,
    atom_excl_partners: &CudaSlice<u32>,
    atom_excl_coul_scales: &CudaSlice<Real>,
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
    let func = particle_buffers.kernels.spme_real.spme_real_pair_force.clone();

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

// rq-9ca00d25 rq-202493a5 rq-16f6c7dc rq-f69698b8
#[allow(clippy::too_many_arguments)]
pub fn spme_charge_spread(
    particle_buffers: &ParticleBuffers,
    sim_box: &SimulationBox,
    sorted_particle_ids: &CudaSlice<u32>,
    cell_offsets: &CudaSlice<u32>,
    grid: [u32; 3],
    spline_order: u32,
    rho: &mut CudaSlice<Real>,
) -> Result<(), GpuError> {
    spme_charge_spread_impl(
        particle_buffers,
        sim_box,
        sorted_particle_ids,
        cell_offsets,
        grid,
        spline_order,
        rho,
        None,
    )
}

// rq-9ca00d25
#[allow(clippy::too_many_arguments)]
pub fn spme_charge_spread_on_stream(
    particle_buffers: &ParticleBuffers,
    sim_box: &SimulationBox,
    sorted_particle_ids: &CudaSlice<u32>,
    cell_offsets: &CudaSlice<u32>,
    grid: [u32; 3],
    spline_order: u32,
    rho: &mut CudaSlice<Real>,
    stream: &CudaStream,
) -> Result<(), GpuError> {
    spme_charge_spread_impl(
        particle_buffers,
        sim_box,
        sorted_particle_ids,
        cell_offsets,
        grid,
        spline_order,
        rho,
        Some(stream),
    )
}

#[allow(clippy::too_many_arguments)]
fn spme_charge_spread_impl(
    particle_buffers: &ParticleBuffers,
    sim_box: &SimulationBox,
    sorted_particle_ids: &CudaSlice<u32>,
    cell_offsets: &CudaSlice<u32>,
    grid: [u32; 3],
    spline_order: u32,
    rho: &mut CudaSlice<Real>,
    stream: Option<&CudaStream>,
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
    let func = particle_buffers.kernels.spme_recip.spme_charge_spread.clone();
    let cfg = launch_config(m_u32);
    let lat = sim_box.lattice();
    let n_u32 = n as u32;
    let args = (
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
    );
    unsafe {
        match stream {
            Some(s) => func.launch_on_stream(s, cfg, args).map_err(GpuError::from)?,
            None => func.launch(cfg, args).map_err(GpuError::from)?,
        }
    }
    Ok(())
}

// rq-9ca00d25 rq-127df3d6 rq-8326d2d1
#[allow(clippy::too_many_arguments)]
pub fn spme_influence_multiply(
    kernels: &Kernels,
    influence_g: &CudaSlice<Real>,
    virial_factor: &CudaSlice<Real>,
    rho_hat_interleaved: &mut CudaSlice<Real>,
    virial_per_cell: &mut CudaSlice<Real>,
    n_c: u32,
    n_c_complex: u32,
    n_complex: u32,
) -> Result<(), GpuError> {
    spme_influence_multiply_impl(
        kernels,
        influence_g,
        virial_factor,
        rho_hat_interleaved,
        virial_per_cell,
        n_c,
        n_c_complex,
        n_complex,
        None,
    )
}

// rq-9ca00d25
#[allow(clippy::too_many_arguments)]
pub fn spme_influence_multiply_on_stream(
    kernels: &Kernels,
    influence_g: &CudaSlice<Real>,
    virial_factor: &CudaSlice<Real>,
    rho_hat_interleaved: &mut CudaSlice<Real>,
    virial_per_cell: &mut CudaSlice<Real>,
    n_c: u32,
    n_c_complex: u32,
    n_complex: u32,
    stream: &CudaStream,
) -> Result<(), GpuError> {
    spme_influence_multiply_impl(
        kernels,
        influence_g,
        virial_factor,
        rho_hat_interleaved,
        virial_per_cell,
        n_c,
        n_c_complex,
        n_complex,
        Some(stream),
    )
}

#[allow(clippy::too_many_arguments)]
fn spme_influence_multiply_impl(
    kernels: &Kernels,
    influence_g: &CudaSlice<Real>,
    virial_factor: &CudaSlice<Real>,
    rho_hat_interleaved: &mut CudaSlice<Real>,
    virial_per_cell: &mut CudaSlice<Real>,
    n_c: u32,
    n_c_complex: u32,
    n_complex: u32,
    stream: Option<&CudaStream>,
) -> Result<(), GpuError> {
    if n_complex == 0 {
        return Ok(());
    }
    debug_assert_eq!(influence_g.len(), n_complex as usize);
    debug_assert_eq!(virial_factor.len(), n_complex as usize);
    debug_assert_eq!(rho_hat_interleaved.len(), 2 * n_complex as usize);
    debug_assert_eq!(virial_per_cell.len(), n_complex as usize);
    let func = kernels.spme_recip.spme_influence_multiply.clone();
    let cfg = launch_config(n_complex);
    let args = (
        influence_g,
        virial_factor,
        rho_hat_interleaved,
        virial_per_cell,
        n_c,
        n_c_complex,
        n_complex,
    );
    unsafe {
        match stream {
            Some(s) => func.launch_on_stream(s, cfg, args).map_err(GpuError::from)?,
            None => func.launch(cfg, args).map_err(GpuError::from)?,
        }
    }
    Ok(())
}

// rq-9ca00d25 rq-35b76155 rq-c6f6a13c
#[allow(clippy::too_many_arguments)]
pub fn spme_force_gather(
    particle_buffers: &ParticleBuffers,
    sim_box: &SimulationBox,
    v: &CudaSlice<Real>,
    u_self_per_particle: &CudaSlice<Real>,
    w_per_particle_virial: Real,
    grid: [u32; 3],
    spline_order: u32,
    slot_force_x: &mut CudaViewMut<'_, Real>,
    slot_force_y: &mut CudaViewMut<'_, Real>,
    slot_force_z: &mut CudaViewMut<'_, Real>,
    slot_energy: &mut CudaViewMut<'_, Real>,
    slot_virial: &mut CudaViewMut<'_, Real>,
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
    let func = particle_buffers.kernels.spme_recip.spme_force_gather.clone();
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

// rq-f00f729e rq-66d80d54 (morse_bond_force launcher mirroring the `gpu` convention)
#[allow(clippy::too_many_arguments)]
pub fn morse_bond_force(
    particle_buffers: &ParticleBuffers,
    bonds: &CudaSlice<u32>,
    bond_de: &CudaSlice<Real>,
    bond_a: &CudaSlice<Real>,
    bond_re: &CudaSlice<Real>,
    sim_box: &SimulationBox,
    bond_pair_x: &mut CudaSlice<Real>,
    bond_pair_y: &mut CudaSlice<Real>,
    bond_pair_z: &mut CudaSlice<Real>,
    bond_pair_energy: &mut CudaSlice<Real>,
    bond_pair_virial: &mut CudaSlice<Real>,
    n_bonds: usize,
) -> Result<(), GpuError> {
    if n_bonds == 0 {
        return Ok(());
    }
    let n_u32 = n_bonds as u32;
    let func = particle_buffers.kernels.morse.morse_bond_force.clone();
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
// rq-10adebc4
#[allow(clippy::too_many_arguments)]
pub fn reduce_bond_forces(
    kernels: &Kernels,
    bond_pair_x: &CudaSlice<Real>,
    bond_pair_y: &CudaSlice<Real>,
    bond_pair_z: &CudaSlice<Real>,
    bond_pair_energy: &CudaSlice<Real>,
    bond_pair_virial: &CudaSlice<Real>,
    atom_bond_offsets: &CudaSlice<u32>,
    atom_bond_indices: &CudaSlice<u32>,
    slot_force_x: &mut CudaViewMut<'_, Real>,
    slot_force_y: &mut CudaViewMut<'_, Real>,
    slot_force_z: &mut CudaViewMut<'_, Real>,
    slot_energy: &mut CudaViewMut<'_, Real>,
    slot_virial: &mut CudaViewMut<'_, Real>,
    particle_count: usize,
) -> Result<(), GpuError> {
    if particle_count == 0 {
        return Ok(());
    }
    let n_u32 = particle_count as u32;
    let func = kernels.morse.reduce_bond_forces.clone();
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

// Launch helper for the harmonic-angle kernel. One thread per angle.
// rq-db5924d8
#[allow(clippy::too_many_arguments)]
pub fn harmonic_angle_force(
    particle_buffers: &ParticleBuffers,
    angles: &CudaSlice<u32>,
    angle_k_theta: &CudaSlice<Real>,
    angle_theta_0: &CudaSlice<Real>,
    sim_box: &SimulationBox,
    angle_triple_x: &mut CudaSlice<Real>,
    angle_triple_y: &mut CudaSlice<Real>,
    angle_triple_z: &mut CudaSlice<Real>,
    angle_triple_energy: &mut CudaSlice<Real>,
    angle_triple_virial: &mut CudaSlice<Real>,
    n_angles: usize,
) -> Result<(), GpuError> {
    if n_angles == 0 {
        return Ok(());
    }
    let n_u32 = n_angles as u32;
    let func = particle_buffers.kernels.angle.harmonic_angle_force.clone();
    let cfg = launch_config(n_u32);
    let lat = sim_box.lattice();
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.positions_x,
                &particle_buffers.positions_y,
                &particle_buffers.positions_z,
                angles,
                angle_k_theta,
                angle_theta_0,
                lat[0],
                lat[1],
                lat[2],
                lat[3],
                lat[4],
                lat[5],
                angle_triple_x,
                angle_triple_y,
                angle_triple_z,
                angle_triple_energy,
                angle_triple_virial,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// Launch helper for the per-atom angle reduction. One thread per atom.
// rq-34bfe79a
#[allow(clippy::too_many_arguments)]
pub fn reduce_angle_forces(
    kernels: &Kernels,
    angle_triple_x: &CudaSlice<Real>,
    angle_triple_y: &CudaSlice<Real>,
    angle_triple_z: &CudaSlice<Real>,
    angle_triple_energy: &CudaSlice<Real>,
    angle_triple_virial: &CudaSlice<Real>,
    atom_angle_offsets: &CudaSlice<u32>,
    atom_angle_indices: &CudaSlice<u32>,
    slot_force_x: &mut CudaViewMut<'_, Real>,
    slot_force_y: &mut CudaViewMut<'_, Real>,
    slot_force_z: &mut CudaViewMut<'_, Real>,
    slot_energy: &mut CudaViewMut<'_, Real>,
    slot_virial: &mut CudaViewMut<'_, Real>,
    particle_count: usize,
) -> Result<(), GpuError> {
    if particle_count == 0 {
        return Ok(());
    }
    let n_u32 = particle_count as u32;
    let func = kernels.angle.reduce_angle_forces.clone();
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(
            cfg,
            (
                angle_triple_x,
                angle_triple_y,
                angle_triple_z,
                angle_triple_energy,
                angle_triple_virial,
                atom_angle_offsets,
                atom_angle_indices,
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

// Launch helper for the NHC kinetic-energy reduction. Single-block,
// 256 threads. Output goes to a length-1 device buffer the caller owns
// (typically reused across calls to avoid per-step allocation). The
// helper synchronously downloads the value and returns it as Real.
// rq-f606ff6f
pub fn compute_kinetic_energy(
    particle_buffers: &ParticleBuffers,
    scratch: &mut CudaSlice<Real>,
) -> Result<Real, GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(0.0);
    }
    debug_assert_eq!(scratch.len(), 1);
    let n_u32 = n as u32;
    let func = particle_buffers.kernels.nose_hoover.kinetic_energy_reduce.clone();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0, // shared array is __shared__ static, not dynamic
    };
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.velocities_x,
                &particle_buffers.velocities_y,
                &particle_buffers.velocities_z,
                &particle_buffers.masses,
                &mut *scratch,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    let mut out = [0.0; 1];
    particle_buffers
        .device
        .dtoh_sync_copy_into(scratch, &mut out)
        .map_err(GpuError::from)?;
    Ok(out[0])
}

// Uniform per-particle velocity rescale. Block size 256, grid
// ceil(n / 256). When n == 0 returns Ok(()) without launching.
// rq-f606ff6f rq-09e04194
pub fn rescale_velocities(
    particle_buffers: &mut ParticleBuffers,
    factor: Real,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    let func = particle_buffers.kernels.nose_hoover.rescale_velocities.clone();
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(
            cfg,
            (
                &mut particle_buffers.velocities_x,
                &mut particle_buffers.velocities_y,
                &mut particle_buffers.velocities_z,
                factor,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// Launch helper for the Andersen per-particle resample kernel. Block
// size 256, grid `ceil(n / 256)`. When `n == 0` returns Ok(()) without
// launching. Debug-asserts `p_collision ∈ [0, 1]` (caller clamps).
// rq-5e059f6b rq-da36d746
#[allow(clippy::too_many_arguments)]
pub fn andersen_resample(
    buffers: &mut ParticleBuffers,
    seed: u64,
    draw_counter: u64,
    p_collision: Real,
    kt: Real,
) -> Result<(), GpuError> {
    let n = buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert!((0.0..=1.0).contains(&p_collision));
    let n_u32 = n as u32;
    let func = buffers.kernels.andersen.andersen_resample.clone();
    let cfg = launch_config(n_u32);
    let seed_lo = seed as u32;
    let seed_hi = (seed >> 32) as u32;
    let draw_lo = draw_counter as u32;
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
                p_collision,
                kt,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// Launch helper for the per-particle virial-sum reduction used by the
// Berendsen barostat. Single-block, 256 threads. Output goes to a
// length-1 device buffer the caller owns (typically reused across calls
// to avoid per-step allocation). The helper synchronously downloads the
// value and returns it as Real.
// rq-0d8c8688 rq-0f50dade
pub fn compute_total_virial(
    particle_buffers: &ParticleBuffers,
    scratch: &mut CudaSlice<Real>,
) -> Result<Real, GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(0.0);
    }
    debug_assert_eq!(scratch.len(), 1);
    let n_u32 = n as u32;
    let func = particle_buffers.kernels.barostat.virial_sum_reduce.clone();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.virials,
                &mut *scratch,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    let mut out = [0.0; 1];
    particle_buffers
        .device
        .dtoh_sync_copy_into(scratch, &mut out)
        .map_err(GpuError::from)?;
    Ok(out[0])
}

// rq-fc6859df
//
// Reuses `virial_sum_reduce` (the generic single-block deterministic
// f32 sum-reduction kernel) against `particle_buffers.potential_energies`.
// Runner-side helper for assembling integrator/thermostat log columns
// that need the total potential energy without downloading the
// per-particle buffer.
pub fn compute_total_potential_energy(
    particle_buffers: &ParticleBuffers,
    scratch: &mut CudaSlice<Real>,
) -> Result<Real, GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(0.0);
    }
    debug_assert_eq!(scratch.len(), 1);
    let n_u32 = n as u32;
    let func = particle_buffers.kernels.barostat.virial_sum_reduce.clone();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.potential_energies,
                &mut *scratch,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    let mut out = [0.0; 1];
    particle_buffers
        .device
        .dtoh_sync_copy_into(scratch, &mut out)
        .map_err(GpuError::from)?;
    Ok(out[0])
}

// Uniform per-particle position rescale used by the Berendsen barostat.
// Block size 256, grid ceil(n / 256). When n == 0 returns Ok(()) without
// launching. Does NOT touch velocities, forces, image flags, or any
// neighbor-list reference positions.
// rq-0d8c8688 rq-19916fb0
pub fn rescale_positions(
    particle_buffers: &mut ParticleBuffers,
    factor: Real,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    let func = particle_buffers.kernels.barostat.rescale_positions.clone();
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(
            cfg,
            (
                &mut particle_buffers.positions_x,
                &mut particle_buffers.positions_y,
                &mut particle_buffers.positions_z,
                factor,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// Launch helper for the MTK cell-coupled velocity half-kick. Block
// size 256, grid ceil(n / 256). When n == 0 returns Ok(()) without
// launching. The host pre-computes both scalar arguments in f64 and
// passes them as Real.
// rq-3b6d5001 rq-cadfb824
pub fn mtk_velocity_half_kick(
    particle_buffers: &mut ParticleBuffers,
    exp_minus_alpha: Real,
    phi_v_dt_half: Real,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    let func = particle_buffers.kernels.mtk.mtk_velocity_half_kick.clone();
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(
            cfg,
            (
                &mut particle_buffers.velocities_x,
                &mut particle_buffers.velocities_y,
                &mut particle_buffers.velocities_z,
                &particle_buffers.forces_x,
                &particle_buffers.forces_y,
                &particle_buffers.forces_z,
                &particle_buffers.masses,
                exp_minus_alpha,
                phi_v_dt_half,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// Launch helper for the MTK cell-coupled position drift. Block size
// 256, grid ceil(n / 256). When n == 0 returns Ok(()) without
// launching.
// rq-3b6d5001 rq-f1c96a3f
pub fn mtk_position_drift(
    particle_buffers: &mut ParticleBuffers,
    exp_b_dt: Real,
    phi_x_dt: Real,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    let func = particle_buffers.kernels.mtk.mtk_position_drift.clone();
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(
            cfg,
            (
                &mut particle_buffers.positions_x,
                &mut particle_buffers.positions_y,
                &mut particle_buffers.positions_z,
                &particle_buffers.velocities_x,
                &particle_buffers.velocities_y,
                &particle_buffers.velocities_z,
                exp_b_dt,
                phi_x_dt,
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
    fast_slot_forces_x: &CudaSlice<Real>,
    fast_slot_forces_y: &CudaSlice<Real>,
    fast_slot_forces_z: &CudaSlice<Real>,
    fast_slot_energies: &CudaSlice<Real>,
    fast_slot_virials: &CudaSlice<Real>,
    num_fast_slots: u32,
    slow_slot_forces_x: &CudaSlice<Real>,
    slow_slot_forces_y: &CudaSlice<Real>,
    slow_slot_forces_z: &CudaSlice<Real>,
    slow_slot_energies: &CudaSlice<Real>,
    slow_slot_virials: &CudaSlice<Real>,
    num_slow_slots: u32,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    debug_assert_eq!(fast_slot_forces_x.len(), num_fast_slots as usize * n);
    debug_assert_eq!(fast_slot_forces_y.len(), num_fast_slots as usize * n);
    debug_assert_eq!(fast_slot_forces_z.len(), num_fast_slots as usize * n);
    debug_assert_eq!(fast_slot_energies.len(), num_fast_slots as usize * n);
    debug_assert_eq!(fast_slot_virials.len(), num_fast_slots as usize * n);
    debug_assert_eq!(slow_slot_forces_x.len(), num_slow_slots as usize * n);
    debug_assert_eq!(slow_slot_forces_y.len(), num_slow_slots as usize * n);
    debug_assert_eq!(slow_slot_forces_z.len(), num_slow_slots as usize * n);
    debug_assert_eq!(slow_slot_energies.len(), num_slow_slots as usize * n);
    debug_assert_eq!(slow_slot_virials.len(), num_slow_slots as usize * n);

    let func = particle_buffers.kernels.forces.accumulate_forces.clone();
    let cfg = launch_config(n_u32);

    unsafe {
        func.launch(
            cfg,
            (
                fast_slot_forces_x,
                fast_slot_forces_y,
                fast_slot_forces_z,
                fast_slot_energies,
                fast_slot_virials,
                num_fast_slots,
                slow_slot_forces_x,
                slow_slot_forces_y,
                slow_slot_forces_z,
                slow_slot_energies,
                slow_slot_virials,
                num_slow_slots,
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
    reference_x: &CudaSlice<Real>,
    reference_y: &CudaSlice<Real>,
    reference_z: &CudaSlice<Real>,
    sim_box: &SimulationBox,
    disp_sq: &mut CudaSlice<Real>,
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
    let func = particle_buffers.kernels.neighbor.neighbor_displacement_squared.clone();
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
    r_search_sq: Real,
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
    let n_cells_total = n_cells[0] * n_cells[1] * n_cells[2];
    // One block per home cell, BLOCK_SIZE threads per block. Each block
    // tiles candidate positions for one neighbour cell at a time into
    // dynamic shared memory, sized as four BLOCK_SIZE-wide arrays
    // (x, y, z, particle_id).
    let func = particle_buffers.kernels.neighbor.neighbor_list_build.clone();
    let cfg = LaunchConfig {
        grid_dim: (n_cells_total, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: BLOCK_SIZE * 4 * std::mem::size_of::<u32>() as u32,
    };
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
    reference_x: &mut CudaSlice<Real>,
    reference_y: &mut CudaSlice<Real>,
    reference_z: &mut CudaSlice<Real>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(reference_x.len(), n);
    debug_assert_eq!(reference_y.len(), n);
    debug_assert_eq!(reference_z.len(), n);
    let n_u32 = n as u32;
    let func = particle_buffers.kernels.neighbor.copy_positions_into_reference.clone();
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

// rq-10f6f831
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
        .neighbor
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
        let func = kernels.neighbor.prefix_scan_local_blocks.clone();
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
        let func = kernels.neighbor.prefix_scan_local_blocks.clone();
        unsafe {
            func.launch(launch_config(len), (level, level, totals, len))
                .map_err(GpuError::from)?;
        }
    }

    // Phase 3: ascend — add each scanned level's totals back into the
    // level below (level 0's target is `cell_offsets`).
    for l in (0..descent_levels).rev() {
        let func = kernels.neighbor.prefix_scan_apply_block_totals.clone();
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
        let func = kernels.neighbor.prefix_scan_finalize_offsets.clone();
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

// rq-9d0cb192
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
    let func = kernels.neighbor.scatter_atoms_into_cells.clone();
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(cfg, (cell_indices, cell_offsets, write_cursors, sorted_particle_ids, n_u32))
            .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-165c4422
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
    let func = kernels.neighbor.sort_cells_by_particle_id.clone();
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(cfg, (cell_offsets, sorted_particle_ids, n_u32))
            .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-7d5e87ee
#[cfg(not(feature = "f64"))]
pub fn vv_kick_drift_lossless(
    buffers: &mut ParticleBuffers,
    lossless: &mut LosslessBuffers,
    sim_box: &SimulationBox,
    dt: Real,
) -> Result<(), GpuError> {
    let n = buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(lossless.particle_count(), n);
    let n_u32 = n as u32;
    let func = buffers.kernels.integrate.vv_kick_drift_lossless.clone();
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
    dt: Real,
) -> Result<(), GpuError> {
    let n = buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    let func = buffers.kernels.langevin.lan_drift_half.clone();
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
    alpha: Real,
    kt: Real,
) -> Result<(), GpuError> {
    let n = buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    let func = buffers.kernels.langevin.lan_ou_step.clone();
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

// rq-4ea8bbb2
#[cfg(not(feature = "f64"))]
pub fn vv_kick_lossless(
    buffers: &mut ParticleBuffers,
    lossless: &mut LosslessBuffers,
    dt: Real,
) -> Result<(), GpuError> {
    let n = buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(lossless.particle_count(), n);
    let n_u32 = n as u32;
    let func = buffers.kernels.integrate.vv_kick_lossless.clone();
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

// rq-53800cef — SHAKE / RATTLE / constraint-virial-scatter launchers

#[allow(clippy::too_many_arguments)]
pub fn shake_snapshot(
    particle_buffers: &ParticleBuffers,
    group_atoms: &CudaSlice<u32>,
    group_atom_offset: &CudaSlice<u32>,
    group_atom_count: &CudaSlice<u32>,
    snapshot_x: &mut CudaSlice<Real>,
    snapshot_y: &mut CudaSlice<Real>,
    snapshot_z: &mut CudaSlice<Real>,
    n_groups: usize,
) -> Result<(), GpuError> {
    if n_groups == 0 {
        return Ok(());
    }
    let n_u32 = n_groups as u32;
    let func = particle_buffers.kernels.shake.shake_snapshot.clone();
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.positions_x,
                &particle_buffers.positions_y,
                &particle_buffers.positions_z,
                group_atoms,
                group_atom_offset,
                group_atom_count,
                snapshot_x,
                snapshot_y,
                snapshot_z,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn shake_positions(
    particle_buffers: &mut ParticleBuffers,
    snapshot_x: &CudaSlice<Real>,
    snapshot_y: &CudaSlice<Real>,
    snapshot_z: &CudaSlice<Real>,
    group_atoms: &CudaSlice<u32>,
    group_atom_offset: &CudaSlice<u32>,
    group_atom_count: &CudaSlice<u32>,
    group_constraint_offset: &CudaSlice<u32>,
    group_constraint_count: &CudaSlice<u32>,
    group_constraints_local_i: &CudaSlice<u8>,
    group_constraints_local_j: &CudaSlice<u8>,
    group_constraints_r2: &CudaSlice<Real>,
    atom_mass: &CudaSlice<Real>,
    sim_box: &SimulationBox,
    dt: Real,
    constraint_virial: &mut CudaSlice<Real>,
    n_groups: usize,
) -> Result<(), GpuError> {
    if n_groups == 0 {
        return Ok(());
    }
    let n_u32 = n_groups as u32;
    let func = particle_buffers.kernels.shake.shake_positions.clone();
    let cfg = launch_config(n_u32);
    let lat = sim_box.lattice();
    unsafe {
        func.launch(
            cfg,
            (
                &mut particle_buffers.positions_x,
                &mut particle_buffers.positions_y,
                &mut particle_buffers.positions_z,
                &mut particle_buffers.velocities_x,
                &mut particle_buffers.velocities_y,
                &mut particle_buffers.velocities_z,
                snapshot_x,
                snapshot_y,
                snapshot_z,
                group_atoms,
                group_atom_offset,
                group_atom_count,
                group_constraint_offset,
                group_constraint_count,
                group_constraints_local_i,
                group_constraints_local_j,
                group_constraints_r2,
                atom_mass,
                lat[0],
                lat[1],
                lat[2],
                lat[3],
                lat[4],
                lat[5],
                dt,
                &mut *constraint_virial,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn constraint_virial_scatter(
    particle_buffers: &mut ParticleBuffers,
    constraint_virial: &CudaSlice<Real>,
    group_atoms: &CudaSlice<u32>,
    n_atom_slots: usize,
) -> Result<(), GpuError> {
    if n_atom_slots == 0 {
        return Ok(());
    }
    let n_atom_slots_u32 = n_atom_slots as u32;
    let func = particle_buffers
        .kernels
        .shake
        .constraint_virial_scatter
        .clone();
    let cfg = launch_config(n_atom_slots_u32);
    unsafe {
        func.launch(
            cfg,
            (
                constraint_virial,
                group_atoms,
                &mut particle_buffers.virials,
                n_atom_slots_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn shake_positions_no_velocity(
    particle_buffers: &mut ParticleBuffers,
    group_atoms: &CudaSlice<u32>,
    group_atom_offset: &CudaSlice<u32>,
    group_atom_count: &CudaSlice<u32>,
    group_constraint_offset: &CudaSlice<u32>,
    group_constraint_count: &CudaSlice<u32>,
    group_constraints_local_i: &CudaSlice<u8>,
    group_constraints_local_j: &CudaSlice<u8>,
    group_constraints_r2: &CudaSlice<Real>,
    atom_mass: &CudaSlice<Real>,
    sim_box: &SimulationBox,
    n_groups: usize,
) -> Result<(), GpuError> {
    if n_groups == 0 {
        return Ok(());
    }
    let n_u32 = n_groups as u32;
    let func = particle_buffers
        .kernels
        .shake
        .shake_positions_no_velocity
        .clone();
    let cfg = launch_config(n_u32);
    let lat = sim_box.lattice();
    unsafe {
        func.launch(
            cfg,
            (
                &mut particle_buffers.positions_x,
                &mut particle_buffers.positions_y,
                &mut particle_buffers.positions_z,
                group_atoms,
                group_atom_offset,
                group_atom_count,
                group_constraint_offset,
                group_constraint_count,
                group_constraints_local_i,
                group_constraints_local_j,
                group_constraints_r2,
                atom_mass,
                lat[0],
                lat[1],
                lat[2],
                lat[3],
                lat[4],
                lat[5],
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn rattle_velocities(
    particle_buffers: &mut ParticleBuffers,
    group_atoms: &CudaSlice<u32>,
    group_atom_offset: &CudaSlice<u32>,
    group_atom_count: &CudaSlice<u32>,
    group_constraint_offset: &CudaSlice<u32>,
    group_constraint_count: &CudaSlice<u32>,
    group_constraints_local_i: &CudaSlice<u8>,
    group_constraints_local_j: &CudaSlice<u8>,
    atom_mass: &CudaSlice<Real>,
    sim_box: &SimulationBox,
    dt: Real,
    constraint_virial: &mut CudaSlice<Real>,
    n_groups: usize,
) -> Result<(), GpuError> {
    if n_groups == 0 {
        return Ok(());
    }
    let n_u32 = n_groups as u32;
    let func = particle_buffers.kernels.shake.rattle_velocities.clone();
    let cfg = launch_config(n_u32);
    let lat = sim_box.lattice();
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.positions_x,
                &particle_buffers.positions_y,
                &particle_buffers.positions_z,
                &mut particle_buffers.velocities_x,
                &mut particle_buffers.velocities_y,
                &mut particle_buffers.velocities_z,
                group_atoms,
                group_atom_offset,
                group_atom_count,
                group_constraint_offset,
                group_constraint_count,
                group_constraints_local_i,
                group_constraints_local_j,
                atom_mass,
                lat[0],
                lat[1],
                lat[2],
                lat[3],
                lat[4],
                lat[5],
                dt,
                &mut *constraint_virial,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}
