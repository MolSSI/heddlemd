//! Harmonic-bonded slot physics tests.
//!
//! Like the Morse slot, the per-bond harmonic contribution kernel is
//! JIT-composed at `ForceField::new` time and dispatched from the bonded
//! composed module; the per-atom reduction reuses the universal
//! `reduce_bond_forces` kernel. These tests drive the slot through a
//! `ForceField` instance and assert on the per-particle force / energy /
//! virial outputs. See `rqm/forces/harmonic-bond.md`.

use heddle_md::forces::{
    AggregateLevel, AngleList, Bond, BondList, DihedralList, ExclusionList, ForceField,
    HarmonicBondBuilder, HarmonicBondState, MorseBondedBuilder, PotentialRegistry,
};
use heddle_md::gpu::{GpuContext, ParticleBuffers, init_device};
use heddle_md::io::config::{BondTypeConfig, NeighborListConfig};
use heddle_md::pbc::SimulationBox;
use heddle_md::precision::Real;
use heddle_md::state::ParticleState;
use heddle_md::timings::Timings;

fn box_10(gpu: &GpuContext) -> SimulationBox {
    SimulationBox::new(&gpu.device, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0).unwrap()
}

fn harmonic_type(k: f64, r0: f64) -> BondTypeConfig {
    BondTypeConfig::Harmonic {
        name: "CT-CT".to_string(),
        k,
        r0,
    }
}

fn morse_type(de: f64, a: f64, re: f64) -> BondTypeConfig {
    BondTypeConfig::Morse {
        name: "MM".to_string(),
        de,
        a,
        re,
    }
}

fn state_from(positions: &[[Real; 3]]) -> ParticleState {
    let n = positions.len();
    ParticleState::new(
        positions.iter().map(|p| p[0]).collect(),
        positions.iter().map(|p| p[1]).collect(),
        positions.iter().map(|p| p[2]).collect(),
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        vec![0.0; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap()
}

/// Build a `BondList` with a correct per-atom reduction map for `bonds`.
fn bond_list_from(n: usize, bonds: Vec<Bond>) -> BondList {
    let mut atom_bond_offsets = vec![0u32; n + 1];
    for b in &bonds {
        atom_bond_offsets[b.atom_i as usize + 1] += 1;
        atom_bond_offsets[b.atom_j as usize + 1] += 1;
    }
    for i in 1..=n {
        atom_bond_offsets[i] += atom_bond_offsets[i - 1];
    }
    let mut atom_bond_indices = vec![0u32; bonds.len() * 2];
    let mut cursor: Vec<u32> = atom_bond_offsets[..n].to_vec();
    for (k, b) in bonds.iter().enumerate() {
        atom_bond_indices[cursor[b.atom_i as usize] as usize] = (2 * k) as u32;
        cursor[b.atom_i as usize] += 1;
        atom_bond_indices[cursor[b.atom_j as usize] as usize] = (2 * k + 1) as u32;
        cursor[b.atom_j as usize] += 1;
    }
    BondList {
        bonds,
        atom_bond_offsets,
        atom_bond_indices,
        particle_count: n,
    }
}

fn bond(i: u32, j: u32, ti: u32) -> Bond {
    Bond {
        atom_i: i,
        atom_j: j,
        bond_type_index: ti,
    }
}

struct Result {
    forces_x: Vec<Real>,
    forces_y: Vec<Real>,
    forces_z: Vec<Real>,
    energies: Vec<Real>,
    virials: Vec<Real>,
}

fn build_ff(
    gpu: &GpuContext,
    n: usize,
    bonds: &BondList,
    bond_types: &[BondTypeConfig],
    with_morse: bool,
) -> ForceField {
    let mut registry = PotentialRegistry::new();
    if with_morse {
        registry.register(Box::new(MorseBondedBuilder));
    }
    registry.register(Box::new(HarmonicBondBuilder));
    ForceField::new(
        &registry,
        gpu,
        n,
        &box_10(gpu),
        &[],
        &[],
        bond_types,
        &[],
        &[],
        None,
        &[],
        bonds,
        &AngleList::empty(0),
        &DihedralList::empty(0),
        &ExclusionList::empty(n),
        &NeighborListConfig::AllPairs,
    )
    .unwrap()
}

fn run_ff(ff: &mut ForceField, gpu: &GpuContext, state: &ParticleState) -> Result {
    let mut buffers = ParticleBuffers::new(gpu, state).unwrap();
    let mut timings = Timings::new(gpu).unwrap();
    ff.step(
        &mut buffers,
        &box_10(gpu),
        &mut timings,
        AggregateLevel::ForcesAndScalars,
    )
    .unwrap();
    Result {
        forces_x: gpu.device.dtoh_sync_copy(&buffers.forces_x).unwrap(),
        forces_y: gpu.device.dtoh_sync_copy(&buffers.forces_y).unwrap(),
        forces_z: gpu.device.dtoh_sync_copy(&buffers.forces_z).unwrap(),
        energies: gpu.device.dtoh_sync_copy(&buffers.potential_energies).unwrap(),
        virials: gpu.device.dtoh_sync_copy(&buffers.virials).unwrap(),
    }
}

/// Harmonic-only run over a two-particle single-bond system.
fn run_harmonic(gpu: &GpuContext, state: &ParticleState, bonds: &BondList, k: f64, r0: f64) -> Result {
    let n = state.positions_x.len();
    let bt = vec![harmonic_type(k, r0)];
    let mut ff = build_ff(gpu, n, bonds, &bt, false);
    run_ff(&mut ff, gpu, state)
}

fn single_bond(n: usize) -> BondList {
    bond_list_from(n, vec![bond(0, 1, 0)])
}

// ---- Construction and per-potential selection ----

// rq-e2b77f30
#[test]
fn construct_harmonic_bond_state_from_all_harmonic_list() {
    let gpu = init_device().unwrap();
    // 3 bonds among 4 atoms, two harmonic types.
    let bl = bond_list_from(4, vec![bond(0, 1, 0), bond(1, 2, 1), bond(2, 3, 0)]);
    let bt = vec![harmonic_type(2.0, 1.0), harmonic_type(4.0, 1.5)];
    let state = HarmonicBondState::new(&gpu, &bl, &bt).unwrap();
    assert_eq!(state.bond_count, 3);
    assert_eq!(state.particle_count, 4);
    let k = gpu.device.dtoh_sync_copy(&state.bond_k).unwrap();
    let r0 = gpu.device.dtoh_sync_copy(&state.bond_r0).unwrap();
    assert_eq!(k, vec![2.0 as Real, 4.0 as Real]);
    assert_eq!(r0, vec![1.0 as Real, 1.5 as Real]);
}

// rq-990d287d rq-c82ac64c
#[test]
fn harmonic_slot_selects_only_harmonic_bonds_from_mixed_list() {
    let gpu = init_device().unwrap();
    // type 0 = morse, type 1 = harmonic. bond A (harmonic), bond B (morse).
    let bt = vec![morse_type(1.0, 2.0, 1.0), harmonic_type(2.0, 1.0)];
    // bond 0: (2,3) harmonic; bond 1: (0,1) morse.
    let bl = bond_list_from(4, vec![bond(2, 3, 1), bond(0, 1, 0)]);
    let state = HarmonicBondState::new(&gpu, &bl, &bt).unwrap();
    assert_eq!(state.bond_count, 1);
    // The single selected bond is the harmonic one, (2, 3), type 1.
    let bonds = gpu.device.dtoh_sync_copy(&state.bonds).unwrap();
    assert_eq!(bonds, vec![2u32, 3u32, 1u32]);
    // The reduction map covers only that bond's two atom shares.
    let offsets = gpu.device.dtoh_sync_copy(&state.atom_bond_offsets).unwrap();
    let indices = gpu.device.dtoh_sync_copy(&state.atom_bond_indices).unwrap();
    assert_eq!(offsets, vec![0u32, 0, 0, 1, 2]); // only atoms 2 and 3 participate
    assert_eq!(indices, vec![0u32, 1u32]);
}

// ---- Force kernel correctness ----

// rq-8eb932b7
#[test]
fn equilibrium_distance_produces_zero_force() {
    let gpu = init_device().unwrap();
    let state = state_from(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
    let r = run_harmonic(&gpu, &state, &single_bond(2), 2.0, 1.0);
    assert!(r.forces_x[0].abs() < 1e-5, "F_x[0]={}", r.forces_x[0]);
    assert!(r.forces_x[1].abs() < 1e-5, "F_x[1]={}", r.forces_x[1]);
}

// rq-c3d498ad
#[test]
fn compressed_bond_repulsive() {
    let gpu = init_device().unwrap();
    // r = 0.5 < r0 = 1.0 → repulsive: atom 0 pushed in -x, atom 1 in +x.
    let state = state_from(&[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]);
    let r = run_harmonic(&gpu, &state, &single_bond(2), 2.0, 1.0);
    assert!(r.forces_x[0] < 0.0, "F_x[0]={}", r.forces_x[0]);
    assert!(r.forces_x[1] > 0.0, "F_x[1]={}", r.forces_x[1]);
    assert!((r.forces_x[0] + r.forces_x[1]).abs() < 1e-5, "Newton's third law");
}

// rq-86fb06cd
#[test]
fn stretched_bond_attractive() {
    let gpu = init_device().unwrap();
    // r = 2.0 > r0 = 1.0 → attractive: atom 0 pulled toward +x.
    let state = state_from(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    let r = run_harmonic(&gpu, &state, &single_bond(2), 2.0, 1.0);
    assert!(r.forces_x[0] > 0.0, "F_x[0]={}", r.forces_x[0]);
    assert!(r.forces_x[1] < 0.0, "F_x[1]={}", r.forces_x[1]);
}

// rq-8a7c1b6a
#[test]
fn force_magnitude_matches_closed_form() {
    let gpu = init_device().unwrap();
    let k: Real = 2.0;
    let r0: Real = 1.0;
    let r_val: Real = 1.3;
    let state = state_from(&[[0.0, 0.0, 0.0], [r_val, 0.0, 0.0]]);
    let r = run_harmonic(&gpu, &state, &single_bond(2), k as f64, r0 as f64);
    // |F on atom 0| = k*|r - r0|; direction is +x (attractive, stretched).
    let expected = k * (r_val - r0);
    let rel = (r.forces_x[0] - expected).abs() / expected.abs().max(1e-6);
    assert!(rel < 1e-4, "expected F_x[0]={}, got {}", expected, r.forces_x[0]);
}

// rq-2d7a2cab
#[test]
fn minimum_image_is_applied() {
    let gpu = init_device().unwrap();
    // Separation 9.5 in a box of 10 → minimum image r = 0.5, displacement
    // (r_i - r_j) wraps to +0.5 along x. Compressed (r < r0=1) → F_x[0] > 0.
    let state = state_from(&[[0.0, 0.0, 0.0], [9.5, 0.0, 0.0]]);
    let r = run_harmonic(&gpu, &state, &single_bond(2), 2.0, 1.0);
    assert!(r.forces_x[0] > 0.0, "min-image should make F_x[0] positive: {}", r.forces_x[0]);
    assert!((r.forces_x[0] + r.forces_x[1]).abs() < 1e-5, "Newton's third law");
    // Magnitude uses the wrapped distance 0.5, not 9.5.
    let expected: Real = 2.0 * (0.5 - 1.0); // k*(r - r0) = -1.0 → F_x[0] = +1.0
    assert!((r.forces_x[0] - expected.abs()).abs() < 1e-4, "F_x[0]={}", r.forces_x[0]);
}

// rq-eff785d2
#[test]
fn degenerate_distance_produces_zero_force_not_nan() {
    let gpu = init_device().unwrap();
    let state = state_from(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let r = run_harmonic(&gpu, &state, &single_bond(2), 2.0, 1.0);
    for v in r.forces_x.iter().chain(r.forces_y.iter()).chain(r.forces_z.iter()) {
        assert!(v.abs() < 1e-12, "expected zero, got {v}");
    }
}

// ---- Energy and virial ----

// rq-1286e57c
#[test]
fn stretched_bond_energy_matches_half_convention() {
    let gpu = init_device().unwrap();
    let k: Real = 2.0;
    let r0: Real = 1.0;
    let r_val: Real = 1.5;
    let state = state_from(&[[0.0, 0.0, 0.0], [r_val, 0.0, 0.0]]);
    let r = run_harmonic(&gpu, &state, &single_bond(2), k as f64, r0 as f64);
    let dr = r_val - r0;
    let u_full = 0.5 * k * dr * dr;
    let total: Real = r.energies.iter().sum();
    let rel = (total - u_full).abs() / u_full.abs().max(1e-6);
    assert!(rel < 1e-4, "expected U={}, got {}", u_full, total);
}

// rq-1b14dbf1
#[test]
fn stretched_bond_virial_matches_r_dot_f() {
    let gpu = init_device().unwrap();
    let k: Real = 2.0;
    let r0: Real = 1.0;
    let r_val: Real = 1.5;
    let state = state_from(&[[0.0, 0.0, 0.0], [r_val, 0.0, 0.0]]);
    let r = run_harmonic(&gpu, &state, &single_bond(2), k as f64, r0 as f64);
    // W = r * F_radial where F_radial = -k*(r - r0).
    let f_radial = -k * (r_val - r0);
    let w_expected = f_radial * r_val;
    let total: Real = r.virials.iter().sum();
    let rel = (total - w_expected).abs() / w_expected.abs().max(1e-6);
    assert!(rel < 1e-4, "expected W={}, got {}", w_expected, total);
}

// rq-5c4d167d
#[test]
fn degenerate_geometry_produces_zero_energy_and_virial() {
    let gpu = init_device().unwrap();
    let state = state_from(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let r = run_harmonic(&gpu, &state, &single_bond(2), 2.0, 1.0);
    for v in r.energies.iter().chain(r.virials.iter()) {
        assert!(v.abs() < 1e-12, "expected zero, got {v}");
    }
}

// ---- Reduction ----

// rq-6d5b52f0
#[test]
fn atom_with_two_harmonic_bonds_receives_sum() {
    let gpu = init_device().unwrap();
    let k: Real = 2.0;
    let r0: Real = 1.0;
    // atom0 at origin; atom1 at (1.5,0,0); atom2 at (2.5,0,0). Both bonds
    // (0,1) and (0,2) stretched along +x, so both pull atom0 in +x.
    let state = state_from(&[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 0.0, 0.0]]);
    let bl = bond_list_from(3, vec![bond(0, 1, 0), bond(0, 2, 0)]);
    let r = run_harmonic(&gpu, &state, &bl, k as f64, r0 as f64);
    // F_x[0] = k*(1.5-1) + k*(2.5-1) = 1.0 + 3.0 = 4.0.
    let expected = k * (1.5 - r0) + k * (2.5 - r0);
    assert!((r.forces_x[0] - expected).abs() < 1e-4, "F_x[0]={}, expected {}", r.forces_x[0], expected);
}

// ---- Empty states / slot presence ----

// rq-5728ee79
#[test]
fn only_morse_bonds_does_not_construct_harmonic_slot() {
    let gpu = init_device().unwrap();
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let bl = bond_list_from(2, vec![bond(0, 1, 0)]); // type 0 = morse
    let ff = build_ff(&gpu, 2, &bl, &bt, true);
    let labels: Vec<&str> = ff.slots.iter().map(|s| s.label()).collect();
    assert!(labels.contains(&"morse_bonded"), "labels={:?}", labels);
    assert!(!labels.contains(&"harmonic_bond"), "labels={:?}", labels);
}

// rq-46ef4201
#[test]
fn zero_harmonic_bonds_is_a_no_op() {
    let gpu = init_device().unwrap();
    // A bond list with only a morse-typed bond → the harmonic selection
    // is empty, so the constructed state has bond_count == 0.
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let bl = bond_list_from(2, vec![bond(0, 1, 0)]);
    let state = HarmonicBondState::new(&gpu, &bl, &bt).unwrap();
    assert_eq!(state.bond_count, 0);
    assert_eq!(state.particle_count, 2);
}

// rq-5e8cef9e
#[test]
fn reduce_on_zero_particles_is_a_no_op() {
    let gpu = init_device().unwrap();
    let bl = BondList::empty(0);
    let bt = vec![harmonic_type(2.0, 1.0)];
    let state = HarmonicBondState::new(&gpu, &bl, &bt).unwrap();
    assert_eq!(state.particle_count, 0);
    assert_eq!(state.bond_count, 0);
}

// ---- Reproducibility ----

// rq-65d5cf1c
#[test]
fn two_independent_runs_are_byte_identical() {
    let gpu = init_device().unwrap();
    let state = state_from(&[[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]]);
    let a = run_harmonic(&gpu, &state, &single_bond(2), 2.0, 1.0);
    let b = run_harmonic(&gpu, &state, &single_bond(2), 2.0, 1.0);
    assert_eq!(a.forces_x, b.forces_x);
    assert_eq!(a.forces_y, b.forces_y);
    assert_eq!(a.forces_z, b.forces_z);
    assert_eq!(a.energies, b.energies);
    assert_eq!(a.virials, b.virials);
}

// ---- End-to-end ----

// rq-f523bca1
#[test]
fn diatomic_equilibrium_produces_zero_net_force() {
    let gpu = init_device().unwrap();
    let state = state_from(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
    let r = run_harmonic(&gpu, &state, &single_bond(2), 2.0, 1.0);
    let sum_x: Real = r.forces_x.iter().sum();
    let sum_y: Real = r.forces_y.iter().sum();
    let sum_z: Real = r.forces_z.iter().sum();
    assert!(sum_x.abs() < 1e-5);
    assert!(sum_y.abs() < 1e-5);
    assert!(sum_z.abs() < 1e-5);
}

// rq-80eff754
#[test]
fn newtons_third_law_holds_for_combined_force() {
    let gpu = init_device().unwrap();
    let state = state_from(&[[0.0, 0.0, 0.0], [1.4, 0.3, 0.5]]);
    let r = run_harmonic(&gpu, &state, &single_bond(2), 2.0, 1.0);
    assert!((r.forces_x[0] + r.forces_x[1]).abs() < 1e-5);
    assert!((r.forces_y[0] + r.forces_y[1]).abs() < 1e-5);
    assert!((r.forces_z[0] + r.forces_z[1]).abs() < 1e-5);
}

// rq-d757f7a1
#[test]
fn mixed_morse_and_harmonic_routes_each_bond_to_one_slot() {
    let gpu = init_device().unwrap();
    // 4 atoms: bond (0,1) morse (type 0), bond (2,3) harmonic (type 1),
    // sharing no atoms. Positions: morse pair stretched, harmonic pair
    // stretched.
    let positions = [
        [0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0], // morse pair, r = 1.2
        [5.0, 0.0, 0.0],
        [7.0, 0.0, 0.0], // harmonic pair, r = 2.0
    ];
    let state = state_from(&positions);
    let bt = vec![morse_type(1.0, 2.0, 1.0), harmonic_type(2.0, 1.0)];
    let bl = bond_list_from(4, vec![bond(0, 1, 0), bond(2, 3, 1)]);

    // Combined run (both slots active).
    let mut ff_both = build_ff(&gpu, 4, &bl, &bt, true);
    let combined = run_ff(&mut ff_both, &gpu, &state);

    // Isolated harmonic-only run over the same system: the morse bond is
    // ignored (its builder is not registered), so only atoms 2,3 move.
    let mut ff_h = build_ff(&gpu, 4, &bl, &bt, false);
    let harmonic_only = run_ff(&mut ff_h, &gpu, &state);

    // The harmonic pair's forces in the combined run must equal the
    // harmonic-only run (harmonic bond not double-counted, morse bond not
    // contributing to atoms 2,3).
    for a in [2usize, 3] {
        assert!((combined.forces_x[a] - harmonic_only.forces_x[a]).abs() < 1e-5,
            "atom {a} F_x combined={} harmonic-only={}", combined.forces_x[a], harmonic_only.forces_x[a]);
    }
    // The morse pair (atoms 0,1) gets no contribution in the harmonic-only
    // run, and a nonzero, Newton-balanced contribution in the combined run.
    assert!(harmonic_only.forces_x[0].abs() < 1e-12);
    assert!(harmonic_only.forces_x[1].abs() < 1e-12);
    assert!(combined.forces_x[0].abs() > 1e-4);
    assert!((combined.forces_x[0] + combined.forces_x[1]).abs() < 1e-5);
    // Harmonic pair balanced too.
    assert!((combined.forces_x[2] + combined.forces_x[3]).abs() < 1e-5);
}
