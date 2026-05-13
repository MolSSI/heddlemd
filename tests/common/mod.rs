//! Shared test helpers. Not a test binary itself; included from individual
//! test files via `mod common;`.

#![allow(dead_code)]

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};
use dynamics::forces::{DeviceExclusionList, ExclusionList};
use dynamics::gpu::{
    GpuError, LennardJonesParameterTable, PairBuffer, ParticleBuffers, lj_pair_force,
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
/// table entry that every particle pair looks up.
pub fn single_type_lj_table(
    device: &Arc<CudaDevice>,
    sigma: f32,
    epsilon: f32,
    cutoff: f32,
) -> LennardJonesParameterTable {
    LennardJonesParameterTable {
        n_types: 1,
        sigma: device.htod_sync_copy(&[sigma]).expect("upload sigma"),
        epsilon: device.htod_sync_copy(&[epsilon]).expect("upload epsilon"),
        cutoff: device.htod_sync_copy(&[cutoff]).expect("upload cutoff"),
    }
}

/// Backward-compatible wrapper around `lj_pair_force` that constructs an
/// empty exclusion list on the fly. Mirrors the function's pre-framework
/// signature so existing kernel-correctness tests can call it unchanged.
pub fn lj_pair_force_no_excl(
    particle_buffers: &ParticleBuffers,
    pair: &mut PairBuffer,
    sim_box: &SimulationBox,
    params: &LennardJonesParameterTable,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    let excl = empty_exclusions(&particle_buffers.device, n);
    lj_pair_force(
        particle_buffers,
        pair,
        sim_box,
        params,
        &excl.atom_excl_offsets,
        &excl.atom_excl_partners,
        &excl.atom_excl_scales,
    )
}

/// Backward-compatible wrapper that calls the new parameterised
/// `reduce_pair_forces` launcher against `particle_buffers.forces_*`.
pub fn reduce_pair_forces_into_buffers(
    pair: &PairBuffer,
    counts: &CudaSlice<u32>,
    particle_buffers: &mut ParticleBuffers,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    let mut vx = particle_buffers.forces_x.slice_mut(..);
    let mut vy = particle_buffers.forces_y.slice_mut(..);
    let mut vz = particle_buffers.forces_z.slice_mut(..);
    reduce_pair_forces(pair, counts, &mut vx, &mut vy, &mut vz, n)
}
