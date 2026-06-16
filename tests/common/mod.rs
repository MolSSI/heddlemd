//! Shared test helpers. Not a test binary itself; included from individual
//! test files via `mod common;`.

#![allow(dead_code)]

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use heddle_md::forces::{
    AggregateLevel, DeviceExclusionList, ExclusionList, NeighborListState, SlotOutputView,
};
use heddle_md::gpu::{
    GpuContext, GpuError, LennardJonesParameterTable, ParticleBuffers, lj_pair_force,
};
use heddle_md::pbc::SimulationBox;
use heddle_md::precision::Real;

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
    sigma: Real,
    epsilon: Real,
    cutoff: Real,
) -> LennardJonesParameterTable {
    single_type_lj_table_with_switch(device, sigma, epsilon, cutoff, cutoff)
}

/// Build a `LennardJonesParameterTable` for a single-type system with an
/// explicit `r_switch < cutoff`. Tests that exercise the switching
/// function use this helper.
pub fn single_type_lj_table_with_switch(
    device: &Arc<CudaDevice>,
    sigma: Real,
    epsilon: Real,
    cutoff: Real,
    r_switch: Real,
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

/// Allocates fresh slot-output buffers sized for `n` particles. The
/// individual CudaSlice fields are zero-initialised.
pub fn alloc_slot_output(
    device: &Arc<CudaDevice>,
    n: usize,
) -> heddle_md::gpu::SlotOutputBuffers {
    heddle_md::gpu::SlotOutputBuffers::new(device, n).expect("alloc_slot_output")
}

/// Wrapper around `lj_pair_force` that constructs an empty exclusion list
/// and a trivial neighbor list on the fly. Kernel-correctness tests use
/// this to exercise the kernel without standing up a full ForceField.
/// The per-particle output is written into `output`'s buffers; the caller
/// is responsible for downloading them for assertions.
pub fn lj_pair_force_no_excl(
    particle_buffers: &ParticleBuffers,
    output: &mut heddle_md::gpu::SlotOutputBuffers,
    sim_box: &SimulationBox,
    params: &LennardJonesParameterTable,
    level: AggregateLevel,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    let gpu = GpuContext {
        device: particle_buffers.device.clone(),
        kernels: particle_buffers.kernels.clone(),
    };
    let excl = empty_exclusions(&gpu.device, n);
    let nl = trivial_neighbor_list(&gpu, sim_box, n);
    let max_neighbors = n as u32;
    let mut view = output.view();
    lj_pair_force(
        particle_buffers,
        &mut view,
        sim_box,
        params,
        &excl.atom_excl_offsets,
        &excl.atom_excl_partners,
        &excl.atom_excl_lj_scales,
        &nl.neighbor_list,
        &nl.neighbor_counts,
        max_neighbors,
        level,
    )
}

/// One-shot wrapper that runs the fused LJ pair-force kernel (forces +
/// energy + virial), then copies the slot-output into `particle_buffers`'s
/// `forces_*`, `potential_energies`, and `virials` fields. Replicates the
/// behaviour the old `lj_pair_force` + `reduce_pair_forces` +
/// `reduce_pair_energy_virial` sequence had, so tests that drive a single
/// LJ slot end-to-end can call one helper.
pub fn lj_pair_force_into_buffers(
    particle_buffers: &mut ParticleBuffers,
    sim_box: &SimulationBox,
    params: &LennardJonesParameterTable,
) -> Result<(), GpuError> {
    let n = particle_buffers.particle_count();
    let mut slot_out = heddle_md::gpu::SlotOutputBuffers::new(&particle_buffers.device, n)?;
    lj_pair_force_no_excl(
        particle_buffers,
        &mut slot_out,
        sim_box,
        params,
        AggregateLevel::ForcesAndScalars,
    )?;
    // Copy slot_out -> particle_buffers.forces_*, potential_energies, virials.
    let device = particle_buffers.device.clone();
    device
        .dtod_copy(&slot_out.force_x, &mut particle_buffers.forces_x)
        .map_err(GpuError::from)?;
    device
        .dtod_copy(&slot_out.force_y, &mut particle_buffers.forces_y)
        .map_err(GpuError::from)?;
    device
        .dtod_copy(&slot_out.force_z, &mut particle_buffers.forces_z)
        .map_err(GpuError::from)?;
    device
        .dtod_copy(&slot_out.energy, &mut particle_buffers.potential_energies)
        .map_err(GpuError::from)?;
    device
        .dtod_copy(&slot_out.virial, &mut particle_buffers.virials)
        .map_err(GpuError::from)?;
    Ok(())
}

