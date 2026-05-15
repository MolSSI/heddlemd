//! Shared test helpers. Not a test binary itself; included from individual
//! test files via `mod common;`.

#![allow(dead_code)]

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};
use dynamics::forces::{DeviceExclusionList, ExclusionList, NeighborListState};
use dynamics::gpu::{
    GpuContext, GpuError, LennardJonesParameterTable, PairBuffer, ParticleBuffers, lj_pair_force,
    reduce_pair_forces,
};
use dynamics::pbc::SimulationBox;

/// Build a `DeviceExclusionList` representing zero exclusions on `n`
/// particles. The LJ kernel reads the buffers and never finds a match,
/// applying scale `1.0` to every pair.
pub fn empty_exclusions(device: &Arc<CudaDevice>, n: usize) -> DeviceExclusionList {
    let host = ExclusionList::empty(n);
    DeviceExclusionList::from_host(device, &host).expect("empty exclusion buffers")
}

/// Build a `LennardJonesParameterTable` for a single-type system using one
/// (σ, ε, cutoff) triple. Equivalent to the n_types=1 case with a single
/// table entry that every particle pair looks up. `r_switch` is set equal
/// to `cutoff`, which selects the hard-cutoff degenerate case in the LJ
/// kernel so that tests written against the unmodified Lennard-Jones
/// expression are unaffected.
pub fn single_type_lj_table(
    device: &Arc<CudaDevice>,
    sigma: f32,
    epsilon: f32,
    cutoff: f32,
) -> LennardJonesParameterTable {
    single_type_lj_table_with_switch(device, sigma, epsilon, cutoff, cutoff)
}

/// Build a `LennardJonesParameterTable` for a single-type system with an
/// explicit `r_switch < cutoff`. Tests that exercise the switching
/// function use this helper.
pub fn single_type_lj_table_with_switch(
    device: &Arc<CudaDevice>,
    sigma: f32,
    epsilon: f32,
    cutoff: f32,
    r_switch: f32,
) -> LennardJonesParameterTable {
    LennardJonesParameterTable {
        n_types: 1,
        sigma: device.htod_sync_copy(&[sigma]).expect("upload sigma"),
        epsilon: device.htod_sync_copy(&[epsilon]).expect("upload epsilon"),
        cutoff: device.htod_sync_copy(&[cutoff]).expect("upload cutoff"),
        switch: device.htod_sync_copy(&[r_switch]).expect("upload switch"),
    }
}

/// Build a trivial-mode neighbor list (every particle's list = [0..N)).
/// Used by tests that exercise the LJ kernel directly without going through
/// the ForceField pipeline.
pub fn trivial_neighbor_list(
    gpu: &GpuContext,
    sim_box: &SimulationBox,
    particle_count: usize,
) -> NeighborListState {
    NeighborListState::new_trivial(gpu, sim_box, particle_count)
        .expect("trivial neighbor list")
}

/// Wrapper around `lj_pair_force` that constructs an empty exclusion list
/// and a trivial neighbor list on the fly. Kernel-correctness tests use
/// this to exercise the kernel without standing up a full ForceField.
pub fn lj_pair_force_no_excl(
    particle_buffers: &ParticleBuffers,
    pair: &mut PairBuffer,
    sim_box: &SimulationBox,
    params: &LennardJonesParameterTable,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    let gpu = GpuContext {
        device: particle_buffers.device.clone(),
        kernels: particle_buffers.kernels.clone(),
    };
    let excl = empty_exclusions(&gpu.device, n);
    let nl = trivial_neighbor_list(&gpu, sim_box, n);
    lj_pair_force(
        particle_buffers,
        pair,
        sim_box,
        params,
        &excl.atom_excl_offsets,
        &excl.atom_excl_partners,
        &excl.atom_excl_lj_scales,
        &nl.neighbor_list,
        &nl.neighbor_counts,
    )
}

/// Backward-compatible wrapper that calls the new parameterised
/// `reduce_pair_forces` launcher against `particle_buffers.forces_*`,
/// `potential_energies`, and `virials`.
pub fn reduce_pair_forces_into_buffers(
    pair: &PairBuffer,
    counts: &CudaSlice<u32>,
    particle_buffers: &mut ParticleBuffers,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    let mut vx = particle_buffers.forces_x.slice_mut(..);
    let mut vy = particle_buffers.forces_y.slice_mut(..);
    let mut vz = particle_buffers.forces_z.slice_mut(..);
    let mut ve = particle_buffers.potential_energies.slice_mut(..);
    let mut vw = particle_buffers.virials.slice_mut(..);
    reduce_pair_forces(pair, counts, &mut vx, &mut vy, &mut vz, &mut ve, &mut vw, n)
}
