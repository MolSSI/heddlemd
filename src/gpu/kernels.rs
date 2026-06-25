use std::sync::Arc;

use cudarc::driver::{
    CudaDevice, CudaSlice, CudaViewMut, DeviceSlice, LaunchAsync, LaunchConfig,
};

#[cfg(not(feature = "f64"))]
use crate::gpu::LosslessBuffers;
use crate::gpu::{GpuError, Kernels, ParticleBuffers};
use crate::io::config::{PairInteractionConfig, PairPotentialParams, ParticleTypeConfig};
use crate::pbc::SimulationBox;
use crate::precision::{Real, Real4};

const BLOCK_SIZE: u32 = 256;

/// Warps per block in the fused pair-force warp-per-particle topology.
/// Must match the `PAIR_FORCE_WARPS_PER_BLOCK` constant in
/// `kernels/pair_compute.cuh`.
pub const PAIR_FORCE_WARPS_PER_BLOCK: u32 = 8;
pub const PAIR_FORCE_BLOCK_SIZE: u32 = 256;

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
    let lattice = sim_box.lattice_device();
    unsafe {
        func.launch(
            cfg,
            (
                &mut buffers.posq,
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
                lattice,
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
    output: &mut crate::forces::SlotOutputView<'_>,
    sim_box: &SimulationBox,
    params: &LennardJonesParameterTable,
    atom_excl_offsets: &CudaSlice<u32>,
    atom_excl_partners: &CudaSlice<u32>,
    atom_excl_lj_scales: &CudaSlice<Real>,
    neighbor_list: &CudaSlice<u32>,
    neighbor_counts: &CudaSlice<u32>,
    max_neighbors: u32,
    level: crate::forces::AggregateLevel,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(neighbor_list.len(), n * max_neighbors as usize);
    debug_assert_eq!(neighbor_counts.len(), n);
    debug_assert_eq!(atom_excl_offsets.len(), n + 1);
    debug_assert_eq!(atom_excl_partners.len(), atom_excl_lj_scales.len());
    debug_assert_eq!(output.force_x.len(), n);
    debug_assert_eq!(output.force_y.len(), n);
    debug_assert_eq!(output.force_z.len(), n);
    let table_len = params.n_types as usize * params.n_types as usize;
    debug_assert_eq!(params.sigma.len(), table_len);
    debug_assert_eq!(params.epsilon.len(), table_len);
    debug_assert_eq!(params.cutoff.len(), table_len);
    debug_assert_eq!(params.switch.len(), table_len);

    let n_u32 = n as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_u32.div_ceil(PAIR_FORCE_WARPS_PER_BLOCK), 1, 1),
        block_dim: (PAIR_FORCE_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let lattice = sim_box.lattice_device();
    match level {
        crate::forces::AggregateLevel::ForcesOnly => unsafe {
            let func = particle_buffers.kernels.lj.pair_force_f.clone();
            func.launch(
                cfg,
                (
                    &particle_buffers.posq,
                    &particle_buffers.type_indices,
                    max_neighbors,
                    lattice,
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
                    &mut output.force_x,
                    &mut output.force_y,
                    &mut output.force_z,
                    n_u32,
                ),
            ).map_err(GpuError::from)?;
        },
        crate::forces::AggregateLevel::ForcesAndScalars => {
            debug_assert_eq!(output.energy.len(), n);
            debug_assert_eq!(output.virial.len(), n);
            unsafe {
                let func = particle_buffers.kernels.lj.pair_force_fev.clone();
                func.launch(
                    cfg,
                    (
                        &particle_buffers.posq,
                        &particle_buffers.type_indices,
                        max_neighbors,
                        lattice,
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
                        &mut output.force_x,
                        &mut output.force_y,
                        &mut output.force_z,
                        &mut output.energy,
                        &mut output.virial,
                        n_u32,
                    ),
                ).map_err(GpuError::from)?;
            }
        }
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
    output: &mut crate::forces::SlotOutputView<'_>,
    sim_box: &SimulationBox,
    cutoff: Real,
    r_switch: Real,
    atom_excl_offsets: &CudaSlice<u32>,
    atom_excl_partners: &CudaSlice<u32>,
    atom_excl_coul_scales: &CudaSlice<Real>,
    neighbor_list: &CudaSlice<u32>,
    neighbor_counts: &CudaSlice<u32>,
    max_neighbors: u32,
    level: crate::forces::AggregateLevel,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(neighbor_list.len(), n * max_neighbors as usize);
    debug_assert_eq!(neighbor_counts.len(), n);
    debug_assert_eq!(atom_excl_offsets.len(), n + 1);
    debug_assert_eq!(atom_excl_partners.len(), atom_excl_coul_scales.len());
    debug_assert_eq!(output.force_x.len(), n);

    let n_u32 = n as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_u32.div_ceil(PAIR_FORCE_WARPS_PER_BLOCK), 1, 1),
        block_dim: (PAIR_FORCE_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let lattice = sim_box.lattice_device();
    match level {
        crate::forces::AggregateLevel::ForcesOnly => unsafe {
            let func = particle_buffers.kernels.coulomb.coulomb_pair_force_f.clone();
            func.launch(
                cfg,
                (
                    &particle_buffers.posq,
                    max_neighbors,
                    lattice,
                    K_COULOMB_F32,
                    cutoff,
                    r_switch,
                    atom_excl_offsets,
                    atom_excl_partners,
                    atom_excl_coul_scales,
                    neighbor_list,
                    neighbor_counts,
                    &mut output.force_x,
                    &mut output.force_y,
                    &mut output.force_z,
                    n_u32,
                ),
            ).map_err(GpuError::from)?;
        },
        crate::forces::AggregateLevel::ForcesAndScalars => {
            debug_assert_eq!(output.energy.len(), n);
            debug_assert_eq!(output.virial.len(), n);
            unsafe {
                let func = particle_buffers.kernels.coulomb.coulomb_pair_force_fev.clone();
                func.launch(
                    cfg,
                    (
                        &particle_buffers.posq,
                        max_neighbors,
                        lattice,
                        K_COULOMB_F32,
                        cutoff,
                        r_switch,
                        atom_excl_offsets,
                        atom_excl_partners,
                        atom_excl_coul_scales,
                        neighbor_list,
                        neighbor_counts,
                        &mut output.force_x,
                        &mut output.force_y,
                        &mut output.force_z,
                        &mut output.energy,
                        &mut output.virial,
                        n_u32,
                    ),
                ).map_err(GpuError::from)?;
            }
        }
    }
    Ok(())
}

// rq-9a512ed1 rq-f6d45062 rq-44cce069 rq-eb9e5cc3 rq-f735ea05
#[allow(clippy::too_many_arguments)]
pub fn spme_real_pair_force(
    particle_buffers: &ParticleBuffers,
    output: &mut crate::forces::SlotOutputView<'_>,
    sim_box: &SimulationBox,
    alpha: Real,
    r_cut_real: Real,
    atom_excl_offsets: &CudaSlice<u32>,
    atom_excl_partners: &CudaSlice<u32>,
    atom_excl_coul_scales: &CudaSlice<Real>,
    neighbor_list: &CudaSlice<u32>,
    neighbor_counts: &CudaSlice<u32>,
    max_neighbors: u32,
    level: crate::forces::AggregateLevel,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(neighbor_list.len(), n * max_neighbors as usize);
    debug_assert_eq!(neighbor_counts.len(), n);
    debug_assert_eq!(atom_excl_offsets.len(), n + 1);
    debug_assert_eq!(atom_excl_partners.len(), atom_excl_coul_scales.len());
    debug_assert_eq!(output.force_x.len(), n);

    let n_u32 = n as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_u32.div_ceil(PAIR_FORCE_WARPS_PER_BLOCK), 1, 1),
        block_dim: (PAIR_FORCE_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let lattice = sim_box.lattice_device();
    match level {
        crate::forces::AggregateLevel::ForcesOnly => unsafe {
            let func = particle_buffers.kernels.spme_real.spme_real_pair_force_f.clone();
            func.launch(
                cfg,
                (
                    &particle_buffers.posq,
                    max_neighbors,
                    lattice,
                    K_COULOMB_F32,
                    alpha,
                    r_cut_real,
                    atom_excl_offsets,
                    atom_excl_partners,
                    atom_excl_coul_scales,
                    neighbor_list,
                    neighbor_counts,
                    &mut output.force_x,
                    &mut output.force_y,
                    &mut output.force_z,
                    n_u32,
                ),
            ).map_err(GpuError::from)?;
        },
        crate::forces::AggregateLevel::ForcesAndScalars => {
            debug_assert_eq!(output.energy.len(), n);
            debug_assert_eq!(output.virial.len(), n);
            unsafe {
                let func = particle_buffers.kernels.spme_real.spme_real_pair_force_fev.clone();
                func.launch(
                    cfg,
                    (
                        &particle_buffers.posq,
                        max_neighbors,
                        lattice,
                        K_COULOMB_F32,
                        alpha,
                        r_cut_real,
                        atom_excl_offsets,
                        atom_excl_partners,
                        atom_excl_coul_scales,
                        neighbor_list,
                        neighbor_counts,
                        &mut output.force_x,
                        &mut output.force_y,
                        &mut output.force_z,
                        &mut output.energy,
                        &mut output.virial,
                        n_u32,
                    ),
                ).map_err(GpuError::from)?;
            }
        }
    }
    Ok(())
}

// Fixed-point charge spread. One warp per sorted slot (8 warps per
// 256-thread block, grid ceil(N / 8)). Lane 0 reads
// `i = sorted_atom_index[t]` to resolve its assigned atom index before
// reading the atom's position and charge; each lane then issues
// `ceil(p^3 / 32)` `atomicAdd<i64>` operations into `rho_fixed`. The
// caller is responsible for zeroing `rho_fixed` before this kernel
// runs (via `memset_zeros` on the device).
pub fn spme_spread_fixed_point(
    particle_buffers: &ParticleBuffers,
    sorted_atom_index: &CudaSlice<u32>,
    sim_box: &SimulationBox,
    grid: [u32; 3],
    spline_order: u32,
    rho_fixed: &mut CudaSlice<i64>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_a = grid[0];
    let n_b = grid[1];
    let n_c = grid[2];
    let m = n_a as usize * n_b as usize * n_c as usize;
    debug_assert_eq!(rho_fixed.len(), m);
    debug_assert_eq!(sorted_atom_index.len(), n);

    let n_u32 = n as u32;
    // PME_ORDER (= spline_order) threads per atom, each owning one
    // z-slice of the p^3 spline support and looping over the
    // p^2 (d_a, d_b) cells in that slice.
    let n_threads = n_u32.checked_mul(spline_order).expect(
        "n * spline_order overflows u32 — particle_count too large for this kernel",
    );
    let cfg = LaunchConfig {
        grid_dim: (n_threads.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let lattice = sim_box.lattice_device();
    let func = particle_buffers.kernels.spme_recip.spme_spread_fixed_point.clone();
    let args = (
        &particle_buffers.posq,
        sorted_atom_index,
        lattice,
        n_a,
        n_b,
        n_c,
        spline_order,
        rho_fixed,
        n_u32,
    );
    unsafe {
        func.launch(cfg, args).map_err(GpuError::from)?;
    }
    Ok(())
}

// Fixed-point -> f32 conversion. One thread per real grid cell.
pub fn spme_spread_finish(
    kernels: &Kernels,
    rho_fixed: &CudaSlice<i64>,
    rho: &mut CudaSlice<Real>,
    m: u32,
) -> Result<(), GpuError> {
    if m == 0 {
        return Ok(());
    }
    debug_assert_eq!(rho_fixed.len(), m as usize);
    debug_assert_eq!(rho.len(), m as usize);
    let func = kernels.spme_recip.spme_spread_finish.clone();
    let cfg = launch_config(m);
    let args = (rho_fixed, rho, m);
    unsafe {
        func.launch(cfg, args).map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-9ca00d25 rq-95385a9d
//
// Launches the fused `spme_recip_apply_influence` kernel:
// one thread per complex grid cell writes V_hat = G * rho_hat in place
// and accumulates the per-thread virial contribution into a per-block
// partial sum written to `virial_partials`.
//
// `virial_partials.len()` must equal the launch's grid size
// (`ceil(n_complex / 256)`); each block writes exactly one entry.
#[allow(clippy::too_many_arguments)]
pub fn spme_recip_apply_influence(
    kernels: &Kernels,
    influence_g: &CudaSlice<Real>,
    virial_factor: &CudaSlice<Real>,
    rho_hat_interleaved: &mut CudaSlice<Real>,
    virial_partials: &mut CudaSlice<Real>,
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
    let cfg = launch_config(n_complex);
    debug_assert_eq!(virial_partials.len(), cfg.grid_dim.0 as usize);
    let func = kernels.spme_recip.spme_recip_apply_influence.clone();
    let args = (
        influence_g,
        virial_factor,
        rho_hat_interleaved,
        virial_partials,
        n_c,
        n_c_complex,
        n_complex,
    );
    unsafe {
        func.launch(cfg, args).map_err(GpuError::from)?;
    }
    Ok(())
}

/// Launches `spme_recip_compute_influence` on the supplied stream:
/// recomputes both `influence_G` and `virial_factor` cell-by-cell from
/// the current lattice. One thread per complex grid cell, block size
/// 256, grid `ceil(M_complex / 256)`. Inner arithmetic is `double`
/// precision; the device store casts to the storage `Real`.
///
/// Returns `Ok(())` immediately (no host wait); the launch is
/// asynchronous on the supplied stream. The caller is expected to
/// schedule downstream consumers (`spme_charge_spread`,
/// `spme_recip_apply_influence`) on the same stream so the writes are
/// visible without additional synchronization.
#[allow(clippy::too_many_arguments)]
pub fn spme_recip_compute_influence(
    kernels: &Kernels,
    b_factors_a: &CudaSlice<Real>,
    b_factors_b: &CudaSlice<Real>,
    b_factors_c: &CudaSlice<Real>,
    influence_g: &mut CudaSlice<Real>,
    virial_factor: &mut CudaSlice<Real>,
    sim_box: &SimulationBox,
    grid: [u32; 3],
    k_coulomb: Real,
    alpha: Real,
    m_complex: u32,
) -> Result<(), GpuError> {
    if m_complex == 0 {
        return Ok(());
    }
    debug_assert_eq!(b_factors_a.len(), grid[0] as usize);
    debug_assert_eq!(b_factors_b.len(), grid[1] as usize);
    debug_assert_eq!(b_factors_c.len(), grid[2] as usize);
    debug_assert_eq!(influence_g.len(), m_complex as usize);
    debug_assert_eq!(virial_factor.len(), m_complex as usize);
    let func = kernels.spme_recip.spme_recip_compute_influence.clone();
    let cfg = launch_config(m_complex);
    let lattice = sim_box.lattice_device();
    let args = (
        b_factors_a,
        b_factors_b,
        b_factors_c,
        &mut *influence_g,
        &mut *virial_factor,
        lattice,
        grid[0],
        grid[1],
        grid[2],
        k_coulomb,
        alpha,
        m_complex,
    );
    unsafe {
        func.launch(cfg, args).map_err(GpuError::from)?;
    }
    Ok(())
}

/// Launches the single-block deterministic reduction of
/// `virial_partials` followed by the Ewald half-sum / per-particle
/// scale. Writes
///   w_per_particle_virial[0] = (0.5 / n) * Σ virial_partials[b]
/// on the supplied stream. Block size 256 (matches the kernel's
/// `__shared__ Real partial[256]`).
pub fn spme_recip_reduce_partials(
    kernels: &Kernels,
    virial_partials: &CudaSlice<Real>,
    w_per_particle_virial: &mut CudaSlice<Real>,
    num_blocks: u32,
    n_particles: u32,
) -> Result<(), GpuError> {
    if num_blocks == 0 || n_particles == 0 {
        return Ok(());
    }
    debug_assert_eq!(virial_partials.len(), num_blocks as usize);
    debug_assert_eq!(w_per_particle_virial.len(), 1);
    let func = kernels.spme_recip.spme_recip_reduce_partials.clone();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let scale: Real = 0.5 / (n_particles as Real);
    let args = (virial_partials, w_per_particle_virial, num_blocks, scale);
    unsafe {
        func.launch(cfg, args).map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-9ca00d25 rq-35b76155 rq-c6f6a13c
#[allow(clippy::too_many_arguments)]
pub fn spme_force_gather(
    particle_buffers: &ParticleBuffers,
    sorted_atom_index: &CudaSlice<u32>,
    sim_box: &SimulationBox,
    v: &CudaSlice<Real>,
    u_self_per_particle: &CudaSlice<Real>,
    w_per_particle_virial: &CudaSlice<Real>,
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
    debug_assert_eq!(w_per_particle_virial.len(), 1);
    debug_assert_eq!(u_self_per_particle.len(), n);
    debug_assert_eq!(sorted_atom_index.len(), n);
    debug_assert_eq!(slot_force_x.len(), n);
    debug_assert_eq!(slot_force_y.len(), n);
    debug_assert_eq!(slot_force_z.len(), n);
    debug_assert_eq!(slot_energy.len(), n);
    debug_assert_eq!(slot_virial.len(), n);

    let n_u32 = n as u32;
    let func = particle_buffers.kernels.spme_recip.spme_force_gather.clone();
    let cfg = launch_config(n_u32);
    let lattice = sim_box.lattice_device();
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.posq,
                v,
                u_self_per_particle,
                w_per_particle_virial,
                sorted_atom_index,
                lattice,
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

// rq-06f1edf2 rq-7594b1fc
//
// Atom spatial pre-sort key computation. One thread per particle.
// Each thread computes the SPME primary-bin index from the same
// `spread_per_particle_setup` geometry the spread and gather kernels
// use, writes the bin to `atom_bin_key[i]`, and atomically increments
// `bin_atom_counts[bin]`. The caller is responsible for zeroing
// `bin_atom_counts` (length M) via `memset_zeros` before launch.
#[allow(clippy::too_many_arguments)]
pub fn spme_compute_bin_key(
    particle_buffers: &ParticleBuffers,
    sim_box: &SimulationBox,
    grid: [u32; 3],
    atom_bin_key: &mut CudaSlice<u32>,
    bin_atom_counts: &mut CudaSlice<u32>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_a = grid[0];
    let n_b = grid[1];
    let n_c = grid[2];
    let m = n_a as usize * n_b as usize * n_c as usize;
    debug_assert_eq!(atom_bin_key.len(), n);
    debug_assert_eq!(bin_atom_counts.len(), m);

    let n_u32 = n as u32;
    let cfg = launch_config(n_u32);
    let lattice = sim_box.lattice_device();
    let func = particle_buffers.kernels.spme_recip.spme_compute_bin_key.clone();
    let args = (
        &particle_buffers.posq,
        lattice,
        n_a,
        n_b,
        n_c,
        &mut *atom_bin_key,
        &mut *bin_atom_counts,
        n_u32,
    );
    unsafe {
        func.launch(cfg, args).map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-a1b761fc
//
// Atom spatial pre-sort orchestrator. Runs the five-stage count-sort
// pipeline on the default stream when the framework's neighbour-list
// rebuild generation has advanced past the slot's cached value.
//
// Returns `Ok(())` immediately when the generation is unchanged — no
// kernel launches.
#[allow(clippy::too_many_arguments)]
pub fn spme_atom_sort(
    particle_buffers: &ParticleBuffers,
    sim_box: &SimulationBox,
    grid: [u32; 3],
    atom_bin_key: &mut CudaSlice<u32>,
    bin_atom_counts: &mut CudaSlice<u32>,
    bin_atom_offsets: &mut CudaSlice<u32>,
    bin_atom_cursor: &mut CudaSlice<u32>,
    sorted_atom_index: &mut CudaSlice<u32>,
    sort_scan_block_totals: &mut [CudaSlice<u32>],
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let device = particle_buffers.device.clone();
    let kernels = particle_buffers.kernels.clone();
    let n_a = grid[0];
    let n_b = grid[1];
    let n_c = grid[2];
    let m = n_a as usize * n_b as usize * n_c as usize;
    debug_assert_eq!(atom_bin_key.len(), n);
    debug_assert_eq!(bin_atom_counts.len(), m);
    debug_assert_eq!(bin_atom_offsets.len(), m + 1);
    debug_assert_eq!(bin_atom_cursor.len(), m);
    debug_assert_eq!(sorted_atom_index.len(), n);

    // Stage 1: zero the histogram and the scatter cursor.
    device.memset_zeros(bin_atom_counts).map_err(GpuError::from)?;
    device.memset_zeros(bin_atom_cursor).map_err(GpuError::from)?;

    // Stage 2: per-atom primary-bin computation + histogram.
    spme_compute_bin_key(
        particle_buffers,
        sim_box,
        grid,
        atom_bin_key,
        bin_atom_counts,
    )?;

    // Stage 3: exclusive prefix scan of the histogram into offsets.
    prefix_scan_cell_counts(
        &kernels,
        bin_atom_counts,
        bin_atom_offsets,
        sort_scan_block_totals,
        m,
        n,
    )?;

    // Stage 4: scatter atoms into their sorted slots using a per-bin
    // atomic cursor. The cursor was zeroed in stage 1; the scatter
    // launcher does not re-zero it.
    {
        let n_u32 = n as u32;
        let func = kernels.neighbor.scatter_atoms_into_cells.clone();
        let cfg = launch_config(n_u32);
        unsafe {
            func.launch(
                cfg,
                (
                    &*atom_bin_key,
                    &*bin_atom_offsets,
                    &mut *bin_atom_cursor,
                    &mut *sorted_atom_index,
                    n_u32,
                ),
            )
            .map_err(GpuError::from)?;
        }
    }

    // Stage 5: per-bin insertion sort over sorted_atom_index in
    // strictly ascending atom-index order. After this pass, two runs
    // with byte-identical positions produce a byte-identical
    // sorted_atom_index regardless of the non-deterministic scatter
    // cursor.
    sort_cells_by_particle_id(&kernels, bin_atom_offsets, sorted_atom_index, m)?;
    Ok(())
}

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
    write_scalars: bool,
) -> Result<(), GpuError> {
    if particle_count == 0 {
        return Ok(());
    }
    let n_u32 = particle_count as u32;
    let func = kernels.morse.reduce_bond_forces.clone();
    let cfg = launch_config(n_u32);
    let write_scalars_u32: u32 = if write_scalars { 1 } else { 0 };
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
                write_scalars_u32,
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
    write_scalars: bool,
) -> Result<(), GpuError> {
    if particle_count == 0 {
        return Ok(());
    }
    let n_u32 = particle_count as u32;
    let func = kernels.angle.reduce_angle_forces.clone();
    let cfg = launch_config(n_u32);
    let write_scalars_u32: u32 = if write_scalars { 1 } else { 0 };
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
                write_scalars_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-1727d6bd
// Largest particle count handled by the single-block scalar reductions.
// Above this the deterministic two-pass multi-block path is used (the
// whole GPU instead of one SM). The single-block path's summation order
// is bit-identical to the historical reduction, so values for systems at
// or below this size are unchanged. See `rqm/pipeline-reproducibility.md`.
const SINGLE_BLOCK_REDUCE_MAX: u32 = 8192;

fn single_block_reduce_cfg() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0, // shared array is __shared__ static, not dynamic
    }
}

// rq-1727d6bd
// Pass 2 of the multi-block reduction: deterministically sum
// `reduction_partials[0..REDUCE_PARTIAL_BLOCKS]` into `scratch[0]` with
// the single-block `virial_sum_reduce`.
fn reduce_partials_into(
    buffers: &ParticleBuffers,
    scratch: &mut CudaSlice<Real>,
) -> Result<(), GpuError> {
    let func = buffers.kernels.barostat.virial_sum_reduce.clone();
    unsafe {
        func.launch(
            single_block_reduce_cfg(),
            (
                &buffers.reduction_partials,
                &mut *scratch,
                crate::gpu::buffers::REDUCE_PARTIAL_BLOCKS,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-1727d6bd
// Deterministic kinetic-energy reduction into `scratch[0]`. Single-block
// for `n <= SINGLE_BLOCK_REDUCE_MAX`, two-pass multi-block otherwise.
fn reduce_kinetic_energy_into(
    buffers: &mut ParticleBuffers,
    scratch: &mut CudaSlice<Real>,
) -> Result<(), GpuError> {
    let n_u32 = buffers.particle_count() as u32;
    if n_u32 <= SINGLE_BLOCK_REDUCE_MAX {
        let func = buffers.kernels.nose_hoover.kinetic_energy_reduce.clone();
        unsafe {
            func.launch(
                single_block_reduce_cfg(),
                (
                    &buffers.velocities_x,
                    &buffers.velocities_y,
                    &buffers.velocities_z,
                    &buffers.masses,
                    &mut *scratch,
                    n_u32,
                ),
            )
            .map_err(GpuError::from)?;
        }
    } else {
        let func = buffers
            .kernels
            .nose_hoover
            .kinetic_energy_reduce_partials
            .clone();
        let cfg = LaunchConfig {
            grid_dim: (crate::gpu::buffers::REDUCE_PARTIAL_BLOCKS, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.launch(
                cfg,
                (
                    &buffers.velocities_x,
                    &buffers.velocities_y,
                    &buffers.velocities_z,
                    &buffers.masses,
                    &mut buffers.reduction_partials,
                    n_u32,
                ),
            )
            .map_err(GpuError::from)?;
        }
        reduce_partials_into(buffers, scratch)?;
    }
    Ok(())
}

// rq-1727d6bd
// Deterministic sum of a per-particle `Real` array (`buffers.virials` or
// `buffers.potential_energies`) into `scratch[0]`. `is_pe` selects the
// input field; both follow the same single-block / multi-block split.
fn reduce_particle_array_into(
    buffers: &mut ParticleBuffers,
    scratch: &mut CudaSlice<Real>,
    is_pe: bool,
) -> Result<(), GpuError> {
    let n_u32 = buffers.particle_count() as u32;
    if n_u32 <= SINGLE_BLOCK_REDUCE_MAX {
        let func = buffers.kernels.barostat.virial_sum_reduce.clone();
        let cfg = single_block_reduce_cfg();
        unsafe {
            if is_pe {
                func.launch(cfg, (&buffers.potential_energies, &mut *scratch, n_u32))
                    .map_err(GpuError::from)?;
            } else {
                func.launch(cfg, (&buffers.virials, &mut *scratch, n_u32))
                    .map_err(GpuError::from)?;
            }
        }
    } else {
        let func = buffers
            .kernels
            .barostat
            .virial_sum_reduce_partials
            .clone();
        let cfg = LaunchConfig {
            grid_dim: (crate::gpu::buffers::REDUCE_PARTIAL_BLOCKS, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            if is_pe {
                func.launch(
                    cfg,
                    (&buffers.potential_energies, &mut buffers.reduction_partials, n_u32),
                )
                .map_err(GpuError::from)?;
            } else {
                func.launch(cfg, (&buffers.virials, &mut buffers.reduction_partials, n_u32))
                    .map_err(GpuError::from)?;
            }
        }
        reduce_partials_into(buffers, scratch)?;
    }
    Ok(())
}

// Launch helper for the kinetic-energy reduction. Deterministic
// (single-block for small N, two-pass multi-block for large N). Output
// goes to a length-1 device buffer the caller owns; the helper
// synchronously downloads the value and returns it as Real.
// rq-f606ff6f rq-1727d6bd
pub fn compute_kinetic_energy(
    particle_buffers: &mut ParticleBuffers,
    scratch: &mut CudaSlice<Real>,
) -> Result<Real, GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(0.0);
    }
    debug_assert_eq!(scratch.len(), 1);
    reduce_kinetic_energy_into(particle_buffers, scratch)?;
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

/// Launches `kinetic_energy_reduce` and leaves the scalar result in
/// `scratch[0]` on device — no dtoh. Same kernel as
/// `compute_kinetic_energy`, just without the blocking host transfer;
/// the caller is expected to consume the value either via another
/// device-side kernel (e.g. `csvr_sample_and_factor`) or via a later
/// host dtoh when one is unavoidable.
pub fn compute_kinetic_energy_on_device(
    particle_buffers: &mut ParticleBuffers,
    scratch: &mut CudaSlice<Real>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(scratch.len(), 1);
    reduce_kinetic_energy_into(particle_buffers, scratch)
}

/// Launches `rescale_velocities_device_factor`: applies a velocity
/// rescale `v_i ← factor[0] · v_i` where `factor` is a single-element
/// device buffer (typically the output of `csvr_sample_and_factor`).
pub fn rescale_velocities_device_factor(
    particle_buffers: &mut ParticleBuffers,
    factor: &CudaSlice<Real>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(factor.len(), 1);
    let n_u32 = n as u32;
    let func = particle_buffers
        .kernels
        .nose_hoover
        .rescale_velocities_device_factor
        .clone();
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

/// Launches `berendsen_compute_factor`: reads the kinetic-energy
/// scalar from `k_old`, computes λ on the device, writes
/// `factor_out[0]`, and accumulates the per-step injection delta
/// `K_old · (λ² − 1)` into `cumulative_injection_delta[0]`. Single
/// 1-thread launch. Stays graph-capturable.
pub fn berendsen_compute_factor(
    particle_buffers: &ParticleBuffers,
    k_old: &CudaSlice<Real>,
    factor_out: &mut CudaSlice<Real>,
    cumulative_injection_delta: &mut CudaSlice<f64>,
    k_target: f64,
    dt_over_tau: f64,
) -> Result<(), GpuError> {
    debug_assert_eq!(k_old.len(), 1);
    debug_assert_eq!(factor_out.len(), 1);
    debug_assert_eq!(cumulative_injection_delta.len(), 1);
    let func = particle_buffers
        .kernels
        .nose_hoover
        .berendsen_compute_factor
        .clone();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(
            cfg,
            (
                k_old,
                &mut *factor_out,
                &mut *cumulative_injection_delta,
                k_target,
                dt_over_tau,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

/// Launches `csvr_sample_and_factor`: reads `k_old` from device,
/// generates `g_dof` Philox samples in parallel, evaluates the CSVR
/// chain, and writes `factor_out[0] = sqrt(k_new/k_old)` plus accumulates
/// `(k_new - k_old)` into `cumulative_injection_delta[0]`. Single
/// 256-thread block; no host sync required.
#[allow(clippy::too_many_arguments)]
pub fn csvr_sample_and_factor(
    particle_buffers: &ParticleBuffers,
    k_old: &CudaSlice<Real>,
    factor_out: &mut CudaSlice<Real>,
    cumulative_injection_delta: &mut CudaSlice<f64>,
    draw_counter_device: &mut CudaSlice<u64>,
    seed: u64,
    g_dof: u32,
    c: f64,
    one_minus_c: f64,
    k_target_over_nf: f64,
) -> Result<(), GpuError> {
    if g_dof == 0 {
        return Ok(());
    }
    debug_assert_eq!(k_old.len(), 1);
    debug_assert_eq!(factor_out.len(), 1);
    debug_assert_eq!(cumulative_injection_delta.len(), 1);
    debug_assert_eq!(draw_counter_device.len(), 1);
    let func = particle_buffers
        .kernels
        .nose_hoover
        .csvr_sample_and_factor
        .clone();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let seed_lo = seed as u32;
    let seed_hi = (seed >> 32) as u32;
    unsafe {
        func.launch(
            cfg,
            (
                k_old,
                &mut *factor_out,
                &mut *cumulative_injection_delta,
                &mut *draw_counter_device,
                seed_lo,
                seed_hi,
                g_dof,
                c,
                one_minus_c,
                k_target_over_nf,
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
    draw_counter_device: &mut CudaSlice<u64>,
    seed: u64,
    p_collision: Real,
    kt: Real,
) -> Result<(), GpuError> {
    let n = buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert!((0.0..=1.0).contains(&p_collision));
    debug_assert_eq!(draw_counter_device.len(), 1);
    let n_u32 = n as u32;
    let func = buffers.kernels.andersen.andersen_resample.clone();
    let cfg = launch_config(n_u32);
    let seed_lo = seed as u32;
    let seed_hi = (seed >> 32) as u32;
    unsafe {
        func.launch(
            cfg,
            (
                &mut buffers.velocities_x,
                &mut buffers.velocities_y,
                &mut buffers.velocities_z,
                &buffers.masses,
                &buffers.particle_ids,
                &*draw_counter_device,
                seed_lo,
                seed_hi,
                p_collision,
                kt,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    // Bump the counter on device for the next launch.
    increment_u64_device(buffers, draw_counter_device)?;
    Ok(())
}

/// Launches the trivial `increment_u64` kernel, which adds 1 to a
/// single u64 device counter. Used after multi-block Philox kernels
/// (where reading and writing the counter inside the same kernel is
/// not safe across blocks) to advance the counter by exactly one per
/// graph node. Captured as a graph node so a replayed graph advances
/// the counter on each replay.
pub fn increment_u64_device(
    buffers: &ParticleBuffers,
    counter: &mut CudaSlice<u64>,
) -> Result<(), GpuError> {
    debug_assert_eq!(counter.len(), 1);
    let func = buffers.kernels.nose_hoover.increment_u64.clone();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(cfg, (&mut *counter,))
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
    particle_buffers: &mut ParticleBuffers,
    scratch: &mut CudaSlice<Real>,
) -> Result<Real, GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(0.0);
    }
    debug_assert_eq!(scratch.len(), 1);
    reduce_particle_array_into(particle_buffers, scratch, false)?;
    let mut out = [0.0; 1];
    particle_buffers
        .device
        .dtoh_sync_copy_into(scratch, &mut out)
        .map_err(GpuError::from)?;
    Ok(out[0])
}

/// Same as `compute_total_virial` but leaves the result in `scratch[0]`
/// on device — no host download. Used by the c-rescale barostat's
/// device-resident pipeline.
pub fn compute_total_virial_on_device(
    particle_buffers: &mut ParticleBuffers,
    scratch: &mut CudaSlice<Real>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(scratch.len(), 1);
    reduce_particle_array_into(particle_buffers, scratch, false)
}

/// Launches `c_rescale_compute_mu`: reads `k`, `w`, and the device
/// lattice (mutated in place by this kernel), samples one Philox normal,
/// computes µ + pressure + post-rescale volume in double precision,
/// mutates the lattice by µ, writes µ to `mu_out`, and writes
/// `[pressure, v_post, injection_delta]` into `diagnostics`. Single
/// thread.
#[allow(clippy::too_many_arguments)]
pub fn c_rescale_compute_mu(
    particle_buffers: &ParticleBuffers,
    k: &CudaSlice<Real>,
    w: &CudaSlice<Real>,
    lattice: &mut CudaSlice<Real>,
    mu_out: &mut CudaSlice<Real>,
    diagnostics: &mut CudaSlice<f64>,
    draw_counter_device: &mut CudaSlice<u64>,
    seed: u64,
    pressure_target: f64,
    tau: f64,
    compressibility: f64,
    kt: f64,
    dt: f64,
    mu_cubed_min: f64,
) -> Result<(), GpuError> {
    debug_assert_eq!(k.len(), 1);
    debug_assert_eq!(w.len(), 1);
    debug_assert_eq!(lattice.len(), 6);
    debug_assert_eq!(mu_out.len(), 1);
    debug_assert_eq!(diagnostics.len(), 3);
    debug_assert_eq!(draw_counter_device.len(), 1);
    let func = particle_buffers
        .kernels
        .barostat
        .c_rescale_compute_mu
        .clone();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let seed_lo = seed as u32;
    let seed_hi = (seed >> 32) as u32;
    unsafe {
        func.launch(
            cfg,
            (
                k,
                w,
                &mut *lattice,
                &mut *mu_out,
                &mut *diagnostics,
                &mut *draw_counter_device,
                seed_lo,
                seed_hi,
                pressure_target,
                tau,
                compressibility,
                kt,
                dt,
                mu_cubed_min,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

/// Launches `berendsen_compute_mu`: deterministic isotropic rescale
/// variant of `c_rescale_compute_mu` (no Philox). Mutates the lattice
/// in place by µ; writes µ to `mu_out`; writes `[pressure, v_post]`
/// to `diagnostics` (length 2).
#[allow(clippy::too_many_arguments)]
pub fn berendsen_compute_mu(
    particle_buffers: &ParticleBuffers,
    k: &CudaSlice<Real>,
    w: &CudaSlice<Real>,
    lattice: &mut CudaSlice<Real>,
    mu_out: &mut CudaSlice<Real>,
    diagnostics: &mut CudaSlice<f64>,
    pressure_target: f64,
    tau: f64,
    compressibility: f64,
    dt: f64,
    mu_cubed_min: f64,
) -> Result<(), GpuError> {
    debug_assert_eq!(k.len(), 1);
    debug_assert_eq!(w.len(), 1);
    debug_assert_eq!(lattice.len(), 6);
    debug_assert_eq!(mu_out.len(), 1);
    debug_assert_eq!(diagnostics.len(), 2);
    let func = particle_buffers
        .kernels
        .barostat
        .berendsen_compute_mu
        .clone();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(
            cfg,
            (
                k,
                w,
                &mut *lattice,
                &mut *mu_out,
                &mut *diagnostics,
                pressure_target,
                tau,
                compressibility,
                dt,
                mu_cubed_min,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

/// Launches `rescale_positions_device_factor`: applies a uniform
/// per-particle position rescale `x_i ← factor[0] · x_i` reading the
/// rescale factor from a 1-element device buffer instead of taking it
/// as a host scalar.
pub fn rescale_positions_device_factor(
    particle_buffers: &mut ParticleBuffers,
    factor: &CudaSlice<Real>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(factor.len(), 1);
    let n_u32 = n as u32;
    let func = particle_buffers
        .kernels
        .barostat
        .rescale_positions_device_factor
        .clone();
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(
            cfg,
            (
                &mut particle_buffers.posq,
                factor,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

// rq-fc6859df rq-1727d6bd
//
// Deterministically sums `particle_buffers.potential_energies`
// (single-block for small N, two-pass multi-block for large N).
// Runner-side helper for assembling integrator/thermostat log columns
// that need the total potential energy without downloading the
// per-particle buffer.
pub fn compute_total_potential_energy(
    particle_buffers: &mut ParticleBuffers,
    scratch: &mut CudaSlice<Real>,
) -> Result<Real, GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(0.0);
    }
    debug_assert_eq!(scratch.len(), 1);
    reduce_particle_array_into(particle_buffers, scratch, true)?;
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
                &mut particle_buffers.posq,
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
                &mut particle_buffers.posq,
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
pub fn combine_class_totals(
    particle_buffers: &mut ParticleBuffers,
    fast_total_forces_x: &CudaSlice<Real>,
    fast_total_forces_y: &CudaSlice<Real>,
    fast_total_forces_z: &CudaSlice<Real>,
    fast_total_potential_energies: &CudaSlice<Real>,
    fast_total_virials: &CudaSlice<Real>,
    slow_total_forces_x: &CudaSlice<Real>,
    slow_total_forces_y: &CudaSlice<Real>,
    slow_total_forces_z: &CudaSlice<Real>,
    slow_total_potential_energies: &CudaSlice<Real>,
    slow_total_virials: &CudaSlice<Real>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    debug_assert_eq!(fast_total_forces_x.len(), n);
    debug_assert_eq!(fast_total_forces_y.len(), n);
    debug_assert_eq!(fast_total_forces_z.len(), n);
    debug_assert_eq!(fast_total_potential_energies.len(), n);
    debug_assert_eq!(fast_total_virials.len(), n);
    debug_assert_eq!(slow_total_forces_x.len(), n);
    debug_assert_eq!(slow_total_forces_y.len(), n);
    debug_assert_eq!(slow_total_forces_z.len(), n);
    debug_assert_eq!(slow_total_potential_energies.len(), n);
    debug_assert_eq!(slow_total_virials.len(), n);

    let func = particle_buffers.kernels.forces.combine_class_totals.clone();
    let cfg = launch_config(n_u32);

    unsafe {
        func.launch(
            cfg,
            (
                fast_total_forces_x,
                fast_total_forces_y,
                fast_total_forces_z,
                fast_total_potential_energies,
                fast_total_virials,
                slow_total_forces_x,
                slow_total_forces_y,
                slow_total_forces_z,
                slow_total_potential_energies,
                slow_total_virials,
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
pub fn neighbor_displacement_check_flag(
    particle_buffers: &ParticleBuffers,
    reference_x: &CudaSlice<Real>,
    reference_y: &CudaSlice<Real>,
    reference_z: &CudaSlice<Real>,
    sim_box: &SimulationBox,
    threshold_sq: Real,
    disp_rebuild_flag: &mut CudaSlice<u32>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(reference_x.len(), n);
    debug_assert_eq!(reference_y.len(), n);
    debug_assert_eq!(reference_z.len(), n);
    debug_assert!(disp_rebuild_flag.len() >= 1);
    let n_u32 = n as u32;
    let func = particle_buffers
        .kernels
        .neighbor
        .neighbor_displacement_check_flag
        .clone();
    let cfg = launch_config(n_u32);
    let lattice = sim_box.lattice_device();
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.posq,
                reference_x,
                reference_y,
                reference_z,
                lattice,
                threshold_sq,
                disp_rebuild_flag,
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
    // dynamic shared memory: three Real arrays (x, y, z) and one u32
    // array (particle_id), each BLOCK_SIZE wide. Per-element bytes
    // therefore depend on `Real`: 16 bytes in the f32 build, 28 bytes
    // in the f64 build.
    let func = particle_buffers.kernels.neighbor.neighbor_list_build.clone();
    let per_elem_bytes =
        3 * std::mem::size_of::<Real>() as u32 + std::mem::size_of::<u32>() as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_cells_total, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: BLOCK_SIZE * per_elem_bytes,
    };
    let lattice = sim_box.lattice_device();
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.posq,
                sorted_particle_ids,
                cell_offsets,
                lattice,
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
                &particle_buffers.posq,
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
    let lattice = sim_box.lattice_device();
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.posq,
                lattice,
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

// =====================================================================
// Packed-neighbour pair-force pipeline launchers
// (rqm/forces/packed-neighbour-pair-force.md)
// =====================================================================

const PACKED_NL_WARPS_PER_BLOCK: u32 = 4;
const PACKED_NL_BLOCK_SIZE: u32 = PACKED_NL_WARPS_PER_BLOCK * 32;
const PACKED_BBOX_WARPS_PER_BLOCK: u32 = 8;
const PACKED_BBOX_BLOCK_SIZE: u32 = PACKED_BBOX_WARPS_PER_BLOCK * 32;

pub fn scatter_positions_to_tile_order(
    kernels: &Kernels,
    particle_buffers: &ParticleBuffers,
    sorted_particle_ids: &CudaSlice<u32>,
    tile_sorted_posq: &mut CudaSlice<Real4>,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = n as u32;
    let func = kernels.neighbor.scatter_positions_to_tile_order.clone();
    let cfg = launch_config(n_u32);
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.posq,
                sorted_particle_ids,
                &mut *tile_sorted_posq,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

pub fn fill_tile_position_padding(
    kernels: &Kernels,
    tile_sorted_posq: &mut CudaSlice<Real4>,
    n: u32,
    padded_n: u32,
) -> Result<(), GpuError> {
    if padded_n <= n {
        return Ok(());
    }
    let padding = padded_n - n;
    let func = kernels.neighbor.fill_tile_position_padding.clone();
    let cfg = launch_config(padding);
    unsafe {
        func.launch(
            cfg,
            (
                &mut *tile_sorted_posq,
                n,
                padded_n,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

pub fn compute_block_bbox(
    kernels: &Kernels,
    tile_sorted_posq: &CudaSlice<Real4>,
    tile_atom_count: &CudaSlice<u32>,
    block_centre: &mut CudaSlice<Real>,
    block_bbox: &mut CudaSlice<Real>,
    n_blocks: u32,
) -> Result<(), GpuError> {
    if n_blocks == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig {
        grid_dim: (n_blocks.div_ceil(PACKED_BBOX_WARPS_PER_BLOCK).max(1), 1, 1),
        block_dim: (PACKED_BBOX_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let func = kernels.neighbor.compute_block_bbox.clone();
    unsafe {
        func.launch(
            cfg,
            (
                tile_sorted_posq,
                tile_atom_count,
                &mut *block_centre,
                &mut *block_bbox,
                n_blocks,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn find_blocks_with_interactions(
    kernels: &Kernels,
    tile_sorted_posq: &CudaSlice<Real4>,
    sorted_particle_ids: &CudaSlice<u32>,
    block_centre: &CudaSlice<Real>,
    block_bbox: &CudaSlice<Real>,
    sim_box: &SimulationBox,
    r_search_sq: Real,
    n_blocks: u32,
    n_atoms: u32,
    max_entries: u32,
    max_single_pairs: u32,
    interacting_tiles: &mut CudaSlice<u32>,
    interacting_atoms: &mut CudaSlice<u32>,
    single_pair_atoms: &mut CudaSlice<u32>,
    interaction_count: &mut CudaSlice<u32>,
    overflow_flag: &mut CudaSlice<u32>,
) -> Result<(), GpuError> {
    if n_blocks == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig {
        grid_dim: (n_blocks.div_ceil(PACKED_NL_WARPS_PER_BLOCK).max(1), 1, 1),
        block_dim: (PACKED_NL_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let func = kernels.neighbor.find_blocks_with_interactions.clone();
    unsafe {
        func.launch(
            cfg,
            (
                tile_sorted_posq,
                sorted_particle_ids,
                block_centre,
                block_bbox,
                sim_box.lattice_device(),
                r_search_sq,
                n_blocks,
                n_atoms,
                max_entries,
                max_single_pairs,
                &mut *interacting_tiles,
                &mut *interacting_atoms,
                &mut *single_pair_atoms,
                &mut *interaction_count,
                &mut *overflow_flag,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn histogram_entries_by_iblock(
    kernels: &Kernels,
    interacting_tiles: &CudaSlice<u32>,
    entry_count_ptr: &CudaSlice<u32>,
    iblock_count: &mut CudaSlice<u32>,
    n_blocks: u32,
    max_entry_count: u32,
) -> Result<(), GpuError> {
    if max_entry_count == 0 || n_blocks == 0 {
        return Ok(());
    }
    let cfg = launch_config(max_entry_count);
    let func = kernels.neighbor.histogram_entries_by_iblock.clone();
    unsafe {
        func.launch(
            cfg,
            (
                interacting_tiles,
                entry_count_ptr,
                &mut *iblock_count,
                n_blocks,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn scatter_entries_by_iblock(
    kernels: &Kernels,
    interacting_tiles: &CudaSlice<u32>,
    interacting_atoms: &CudaSlice<u32>,
    entry_count_ptr: &CudaSlice<u32>,
    iblock_offset: &CudaSlice<u32>,
    iblock_cursor: &mut CudaSlice<u32>,
    sorted_interacting_atoms: &mut CudaSlice<u32>,
    n_blocks: u32,
    max_entry_count: u32,
) -> Result<(), GpuError> {
    if max_entry_count == 0 || n_blocks == 0 {
        return Ok(());
    }
    // One warp per entry; 8 warps per block = 256 threads.
    let warps_per_block: u32 = 8;
    let block_dim: u32 = warps_per_block * 32;
    let grid_x = max_entry_count.div_ceil(warps_per_block).max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let func = kernels.neighbor.scatter_entries_by_iblock.clone();
    unsafe {
        func.launch(
            cfg,
            (
                interacting_tiles,
                interacting_atoms,
                entry_count_ptr,
                iblock_offset,
                &mut *iblock_cursor,
                &mut *sorted_interacting_atoms,
                n_blocks,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn finalize_packed_forces(
    kernels: &Kernels,
    fp_fx: &CudaSlice<u64>,
    fp_fy: &CudaSlice<u64>,
    fp_fz: &CudaSlice<u64>,
    fp_e: &CudaSlice<u64>,
    fp_w: &CudaSlice<u64>,
    out_fx: &mut cudarc::driver::CudaViewMut<'_, Real>,
    out_fy: &mut cudarc::driver::CudaViewMut<'_, Real>,
    out_fz: &mut cudarc::driver::CudaViewMut<'_, Real>,
    out_e: &mut cudarc::driver::CudaViewMut<'_, Real>,
    out_w: &mut cudarc::driver::CudaViewMut<'_, Real>,
    n: u32,
    write_ev: bool,
) -> Result<(), GpuError> {
    if n == 0 {
        return Ok(());
    }
    let cfg = launch_config(n);
    let func = kernels.neighbor.finalize_packed_forces.clone();
    let ev_u32: u32 = if write_ev { 1 } else { 0 };
    unsafe {
        func.launch(
            cfg,
            (
                fp_fx,
                fp_fy,
                fp_fz,
                fp_e,
                fp_w,
                &mut *out_fx,
                &mut *out_fy,
                &mut *out_fz,
                &mut *out_e,
                &mut *out_w,
                n,
                ev_u32,
            ),
        )
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
    let lattice = sim_box.lattice_device();
    unsafe {
        func.launch(
            cfg,
            (
                &mut buffers.posq,
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
                lattice,
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
    let lattice = sim_box.lattice_device();
    unsafe {
        func.launch(
            cfg,
            (
                &mut buffers.posq,
                &mut buffers.images_x,
                &mut buffers.images_y,
                &mut buffers.images_z,
                &buffers.velocities_x,
                &buffers.velocities_y,
                &buffers.velocities_z,
                lattice,
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
    draw_counter_device: &mut CudaSlice<u64>,
    seed: u64,
    alpha: Real,
    kt: Real,
) -> Result<(), GpuError> {
    let n = buffers.particle_count();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(draw_counter_device.len(), 1);
    let n_u32 = n as u32;
    let func = buffers.kernels.langevin.lan_ou_step.clone();
    let cfg = launch_config(n_u32);
    let seed_lo = (seed & 0xFFFF_FFFF) as u32;
    let seed_hi = (seed >> 32) as u32;
    unsafe {
        func.launch(
            cfg,
            (
                &mut buffers.velocities_x,
                &mut buffers.velocities_y,
                &mut buffers.velocities_z,
                &buffers.masses,
                &buffers.particle_ids,
                &*draw_counter_device,
                seed_lo,
                seed_hi,
                alpha,
                kt,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    increment_u64_device(buffers, draw_counter_device)?;
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
                &particle_buffers.posq,
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
    max_group_atoms: u32,
) -> Result<(), GpuError> {
    if n_groups == 0 {
        return Ok(());
    }
    let n_u32 = n_groups as u32;
    let func = particle_buffers.kernels.shake.shake_positions.clone();
    // rq-115e5926
    // One thread per group, in blocks of SHAKE_POS_BLOCK_SIZE. Each block
    // stages its groups' atoms into dynamic shared memory (16 Reals per
    // atom). The block size is kept at 64 so the reservation
    // `block * max_group_atoms * 16 * sizeof(Real)` stays within the 48 KB
    // default shared budget even at MAX_GROUP_ATOMS = 8.
    const SHAKE_POS_BLOCK_SIZE: u32 = 64;
    let block = SHAKE_POS_BLOCK_SIZE;
    let grid = n_u32.div_ceil(block);
    let shared_atoms = block * max_group_atoms.max(1);
    let shared_mem_bytes = shared_atoms * 16 * (std::mem::size_of::<Real>() as u32);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes,
    };
    let lattice = sim_box.lattice_device();
    unsafe {
        func.launch(
            cfg,
            (
                &mut particle_buffers.posq,
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
                lattice,
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
    let lattice = sim_box.lattice_device();
    unsafe {
        func.launch(
            cfg,
            (
                &mut particle_buffers.posq,
                group_atoms,
                group_atom_offset,
                group_atom_count,
                group_constraint_offset,
                group_constraint_count,
                group_constraints_local_i,
                group_constraints_local_j,
                group_constraints_r2,
                atom_mass,
                lattice,
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
    max_group_atoms: u32,
) -> Result<(), GpuError> {
    if n_groups == 0 {
        return Ok(());
    }
    let n_u32 = n_groups as u32;
    let func = particle_buffers.kernels.shake.rattle_velocities.clone();
    // rq-53800cef rq-115e5926
    // One thread per group, in blocks of RATTLE_BLOCK_SIZE. The block
    // stages its groups' atoms into dynamic shared memory (11 Reals per
    // atom), so the launch reserves
    // `block * max_group_atoms * 11 * sizeof(Real)` bytes — the worst-case
    // atom count a block can own. See `kernels/shake.cu` rattle_velocities.
    const RATTLE_BLOCK_SIZE: u32 = 128;
    let block = RATTLE_BLOCK_SIZE;
    let grid = n_u32.div_ceil(block);
    let shared_atoms = block * max_group_atoms.max(1);
    let shared_mem_bytes = shared_atoms * 11 * (std::mem::size_of::<Real>() as u32);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes,
    };
    let lattice = sim_box.lattice_device();
    unsafe {
        func.launch(
            cfg,
            (
                &particle_buffers.posq,
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
                lattice,
                dt,
                &mut *constraint_virial,
                n_u32,
            ),
        )
        .map_err(GpuError::from)?;
    }
    Ok(())
}
