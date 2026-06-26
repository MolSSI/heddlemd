//! Shared test helpers. Not a test binary itself; included from individual
//! test files via `mod common;`.

#![allow(dead_code)]

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use heddle_md::forces::{DeviceExclusionList, Exclusion, ExclusionList};
use heddle_md::gpu::LennardJonesParameterTable;
use heddle_md::precision::Real;

/// Build a `DeviceExclusionList` representing zero exclusions on `n`
/// particles. The LJ kernel reads the buffers and never finds a match,
/// applying scale `1.0` to every pair.
pub fn empty_exclusions(device: &Arc<CudaDevice>, n: usize) -> DeviceExclusionList {
    let host = ExclusionList::empty(n);
    DeviceExclusionList::from_host(device, &host).expect("empty exclusion buffers")
}

/// Build a host-side `ExclusionList` over `n` particles holding the
/// given symmetric `(i, j, scale_lj, scale_coul)` entries. The
/// per-atom offset / partner / scale arrays are populated for both
/// directions of every entry.
pub fn host_exclusions_from_entries(
    n: usize,
    entries: &[(u32, u32, Real, Real)],
) -> ExclusionList {
    let exclusion_entries: Vec<Exclusion> = entries
        .iter()
        .map(|&(i, j, lj, coul)| Exclusion {
            atom_i: i,
            atom_j: j,
            scale_lj: lj,
            scale_coul: coul,
        })
        .collect();
    let mut per_atom: Vec<Vec<(u32, Real, Real)>> = vec![Vec::new(); n];
    for &(i, j, lj, coul) in entries {
        per_atom[i as usize].push((j, lj, coul));
        per_atom[j as usize].push((i, lj, coul));
    }
    let mut atom_excl_offsets = Vec::with_capacity(n + 1);
    let mut atom_excl_partners = Vec::new();
    let mut atom_excl_lj_scales = Vec::new();
    let mut atom_excl_coul_scales = Vec::new();
    atom_excl_offsets.push(0u32);
    for partners in &per_atom {
        for &(partner, lj, coul) in partners {
            atom_excl_partners.push(partner);
            atom_excl_lj_scales.push(lj);
            atom_excl_coul_scales.push(coul);
        }
        atom_excl_offsets.push(atom_excl_partners.len() as u32);
    }
    ExclusionList {
        entries: exclusion_entries,
        atom_excl_offsets,
        atom_excl_partners,
        atom_excl_lj_scales,
        atom_excl_coul_scales,
        particle_count: n,
    }
}

/// Build a `DeviceExclusionList` from the given symmetric entries. Each
/// tuple is `(atom_i, atom_j, scale_lj, scale_coul)`.
pub fn exclusions_from_entries(
    device: &Arc<CudaDevice>,
    n: usize,
    entries: &[(u32, u32, Real, Real)],
) -> DeviceExclusionList {
    let host = host_exclusions_from_entries(n, entries);
    DeviceExclusionList::from_host(device, &host).expect("upload exclusion buffers")
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

/// Allocates fresh slot-output buffers sized for `n` particles. The
/// individual CudaSlice fields are zero-initialised.
pub fn alloc_slot_output(
    device: &Arc<CudaDevice>,
    n: usize,
) -> heddle_md::gpu::SlotOutputBuffers {
    heddle_md::gpu::SlotOutputBuffers::new(device, n).expect("alloc_slot_output")
}

