//! Morse-bonded slot physics tests.
//!
//! Under J2, the per-bond Morse contribution kernel is JIT-composed
//! at `ForceField::new` time and dispatched from the bonded composed
//! module; there is no standalone `morse_bond_force` launcher.
//! These tests drive the slot through a `ForceField` instance and
//! assert on the per-particle force / energy / virial outputs.

use heddle_md::forces::{
    AggregateLevel, AngleList, Bond, BondList, ExclusionList, ForceField, MorseBondedBuilder,
    PotentialRegistry,
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

fn morse_type(de: f64, a: f64, re: f64) -> BondTypeConfig {
    BondTypeConfig::Morse {
        name: "CC".to_string(),
        de,
        a,
        re,
    }
}

fn two_particle_state(p0: [Real; 3], p1: [Real; 3]) -> ParticleState {
    ParticleState::new(
        vec![p0[0], p1[0]],
        vec![p0[1], p1[1]],
        vec![p0[2], p1[2]],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![1.0; 2],
        vec![0.0; 2],
        vec![0u32; 2],
        None,
        None,
    )
    .unwrap()
}

fn single_bond_list(n: usize) -> BondList {
    let bonds = vec![Bond {
        atom_i: 0,
        atom_j: 1,
        bond_type_index: 0,
    }];
    let mut atom_bond_offsets = vec![0u32; n + 1];
    atom_bond_offsets[1] = 1;
    atom_bond_offsets[2] = 2;
    for i in 3..=n {
        atom_bond_offsets[i] = 2;
    }
    let atom_bond_indices = vec![0u32, 1u32];
    BondList {
        bonds,
        atom_bond_offsets,
        atom_bond_indices,
        particle_count: n,
    }
}

struct MorseResult {
    forces_x: Vec<Real>,
    forces_y: Vec<Real>,
    forces_z: Vec<Real>,
    energies: Vec<Real>,
    virials: Vec<Real>,
}

/// Build a one-slot (Morse only) ForceField, run one step at
/// `AggregateLevel::ForcesAndScalars`, and download per-particle
/// outputs.
fn run_morse(
    gpu: &GpuContext,
    state: &ParticleState,
    bonds: &BondList,
    bond_types: &[BondTypeConfig],
) -> MorseResult {
    let n = state.positions_x.len();
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(MorseBondedBuilder));
    let mut ff = ForceField::new(
        &registry,
        gpu,
        n,
        &box_10(gpu),
        &[],
        &[],
        bond_types,
        &[],
        None,
        None,
        &[],
        bonds,
        &AngleList::empty(0),
        &ExclusionList::empty(n),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(gpu, state).unwrap();
    let mut timings = Timings::new(gpu).unwrap();
    ff.step(
        &mut buffers,
        &box_10(gpu),
        &mut timings,
        AggregateLevel::ForcesAndScalars,
    )
    .unwrap();
    MorseResult {
        forces_x: gpu.device.dtoh_sync_copy(&buffers.forces_x).unwrap(),
        forces_y: gpu.device.dtoh_sync_copy(&buffers.forces_y).unwrap(),
        forces_z: gpu.device.dtoh_sync_copy(&buffers.forces_z).unwrap(),
        energies: gpu.device.dtoh_sync_copy(&buffers.potential_energies).unwrap(),
        virials: gpu.device.dtoh_sync_copy(&buffers.virials).unwrap(),
    }
}

// rq-679282f5
#[test]
fn init_device_loads_morse_module() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    assert!(device.get_func("morse", "reduce_bond_forces").is_some());
    let _ = gpu.kernels.morse.reduce_bond_forces.clone();
}

// rq-2e4e70b4
#[test]
fn equilibrium_distance_produces_zero_force() {
    let gpu = init_device().unwrap();
    let state = two_particle_state([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let r = run_morse(&gpu, &state, &bl, &bt);
    assert!(r.forces_x[0].abs() < 1e-5, "F_x[0]={}", r.forces_x[0]);
    assert!(r.forces_x[1].abs() < 1e-5, "F_x[1]={}", r.forces_x[1]);
}

#[test]
fn compressed_bond_repulsive() {
    let gpu = init_device().unwrap();
    // r = 0.8 < re = 1.0 → repulsive: F on atom_i (at x=0) pushes
    // away from atom_j (at x=0.8), i.e. F_x[0] < 0 and F_x[1] > 0.
    let state = two_particle_state([0.0, 0.0, 0.0], [0.8, 0.0, 0.0]);
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let r = run_morse(&gpu, &state, &bl, &bt);
    assert!(r.forces_x[0] < 0.0, "F_x[0]={}", r.forces_x[0]);
    assert!(r.forces_x[1] > 0.0, "F_x[1]={}", r.forces_x[1]);
    assert!(
        (r.forces_x[0] + r.forces_x[1]).abs() < 1e-5,
        "Newton's third law"
    );
}

#[test]
fn stretched_bond_attractive() {
    let gpu = init_device().unwrap();
    // r = 1.2 > re = 1.0 → attractive: F pulls atoms together.
    let state = two_particle_state([0.0, 0.0, 0.0], [1.2, 0.0, 0.0]);
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let r = run_morse(&gpu, &state, &bl, &bt);
    assert!(r.forces_x[0] > 0.0, "F_x[0]={}", r.forces_x[0]);
    assert!(r.forces_x[1] < 0.0, "F_x[1]={}", r.forces_x[1]);
}

#[test]
fn force_magnitude_matches_closed_form() {
    let gpu = init_device().unwrap();
    let de: Real = 1.0;
    let a: Real = 2.0;
    let re: Real = 1.0;
    let r_val: Real = 1.3;
    let state = two_particle_state([0.0, 0.0, 0.0], [r_val, 0.0, 0.0]);
    let bl = single_bond_list(2);
    let bt = vec![morse_type(de as f64, a as f64, re as f64)];
    let result = run_morse(&gpu, &state, &bl, &bt);
    // F_radial = -dU/dr = -2 De a (1-e) e where e = exp(-a(r - re)).
    // Force on atom_i along +d_hat where d_hat = (r_i - r_j)/r.
    // F_x_on_i = F_radial * (r_i - r_j) / r = F_radial * (-1.0).
    let e = (-a * (r_val - re)).exp();
    let f_radial = -2.0 * de * a * (1.0 - e) * e;
    let f_x_on_i_expected = f_radial * (-1.0);
    let rel =
        (result.forces_x[0] - f_x_on_i_expected).abs() / f_x_on_i_expected.abs().max(1e-6);
    assert!(rel < 1e-4, "expected F_x[0]={}, got {}", f_x_on_i_expected, result.forces_x[0]);
}

#[test]
fn r_zero_produces_zero_force_and_energy() {
    let gpu = init_device().unwrap();
    let state = two_particle_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let r = run_morse(&gpu, &state, &bl, &bt);
    for v in r.forces_x.iter().chain(r.forces_y.iter()).chain(r.forces_z.iter()) {
        assert!(v.abs() < 1e-12, "expected zero, got {v}");
    }
    for v in r.energies.iter().chain(r.virials.iter()) {
        assert!(v.abs() < 1e-12, "expected zero, got {v}");
    }
}

#[test]
fn diatomic_equilibrium_produces_zero_net_force() {
    let gpu = init_device().unwrap();
    let state = two_particle_state([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let r = run_morse(&gpu, &state, &bl, &bt);
    let sum_x: Real = r.forces_x.iter().sum();
    let sum_y: Real = r.forces_y.iter().sum();
    let sum_z: Real = r.forces_z.iter().sum();
    assert!(sum_x.abs() < 1e-5);
    assert!(sum_y.abs() < 1e-5);
    assert!(sum_z.abs() < 1e-5);
}

#[test]
fn newtons_third_law_holds_for_combined_force() {
    let gpu = init_device().unwrap();
    let state = two_particle_state([0.0, 0.0, 0.0], [1.4, 0.3, 0.5]);
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let r = run_morse(&gpu, &state, &bl, &bt);
    assert!((r.forces_x[0] + r.forces_x[1]).abs() < 1e-5);
    assert!((r.forces_y[0] + r.forces_y[1]).abs() < 1e-5);
    assert!((r.forces_z[0] + r.forces_z[1]).abs() < 1e-5);
}

// rq-7ba4f321
#[test]
fn stretched_bond_energy_matches_closed_form() {
    let gpu = init_device().unwrap();
    let de: Real = 1.0;
    let a: Real = 2.0;
    let re: Real = 1.0;
    let r_val: Real = 1.3;
    let state = two_particle_state([0.0, 0.0, 0.0], [r_val, 0.0, 0.0]);
    let bl = single_bond_list(2);
    let bt = vec![morse_type(de as f64, a as f64, re as f64)];
    let r = run_morse(&gpu, &state, &bl, &bt);
    // U = De * (1 - e)^2 where e = exp(-a(r - re)). The framework
    // distributes the energy half/half between the two atoms.
    let e = (-a * (r_val - re)).exp();
    let one_minus_e = 1.0 - e;
    let u_full = de * one_minus_e * one_minus_e;
    let total_energy: Real = r.energies.iter().sum();
    let rel = (total_energy - u_full).abs() / u_full.abs().max(1e-6);
    assert!(rel < 1e-4, "expected U={}, got {}", u_full, total_energy);
}

// rq-ca49d49a
#[test]
fn stretched_bond_virial_matches_r_dot_f() {
    let gpu = init_device().unwrap();
    let de: Real = 1.0;
    let a: Real = 2.0;
    let re: Real = 1.0;
    let r_val: Real = 1.3;
    let state = two_particle_state([0.0, 0.0, 0.0], [r_val, 0.0, 0.0]);
    let bl = single_bond_list(2);
    let bt = vec![morse_type(de as f64, a as f64, re as f64)];
    let r = run_morse(&gpu, &state, &bl, &bt);
    // W = fmag * r^2; F_radial / r = fmag (force factor along d_hat
    // scaled by 1/r). Compute the expected scalar virial as
    // F_radial * r_radial where F_radial = -2*De*a*(1-e)*e.
    let e = (-a * (r_val - re)).exp();
    let f_radial = -2.0 * de * a * (1.0 - e) * e;
    // W convention: each atom carries W_k/2 → summing both gives W_k.
    let w_expected = f_radial * r_val;
    let total_virial: Real = r.virials.iter().sum();
    let rel = (total_virial - w_expected).abs() / w_expected.abs().max(1e-6);
    assert!(rel < 1e-4, "expected W={}, got {}", w_expected, total_virial);
}

#[test]
fn minimum_image_is_applied_to_bond_displacement() {
    let gpu = init_device().unwrap();
    // Two atoms separated by 9.5 along x in a box of length 10:
    // minimum image gives r = 0.5, with displacement direction
    // along -x. The bond's equilibrium length is 1.0, so the bond
    // is compressed → atoms repel along the minimum-image direction.
    let state = two_particle_state([0.0, 0.0, 0.0], [9.5, 0.0, 0.0]);
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let r = run_morse(&gpu, &state, &bl, &bt);
    // After minimum image, the displacement (r_i - r_j) wraps to
    // +0.5 along x (since 9.5 wraps to -0.5; r_i - r_j = -(-0.5) =
    // +0.5). Compressed → atom_i feels a force away from atom_j,
    // i.e. F_x[0] > 0.
    assert!(
        r.forces_x[0] > 0.0,
        "minimum image wrap should make F_x[0] positive: {}",
        r.forces_x[0]
    );
    assert!(
        (r.forces_x[0] + r.forces_x[1]).abs() < 1e-5,
        "Newton's third law"
    );
}

#[test]
fn two_independent_runs_produce_byte_identical_forces() {
    let gpu = init_device().unwrap();
    let state = two_particle_state([0.0, 0.0, 0.0], [1.2, 0.0, 0.0]);
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let a = run_morse(&gpu, &state, &bl, &bt);
    let b = run_morse(&gpu, &state, &bl, &bt);
    assert_eq!(a.forces_x, b.forces_x);
    assert_eq!(a.forces_y, b.forces_y);
    assert_eq!(a.forces_z, b.forces_z);
    assert_eq!(a.energies, b.energies);
    assert_eq!(a.virials, b.virials);
}
