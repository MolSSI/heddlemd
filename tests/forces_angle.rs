//! Harmonic-angle slot physics tests.
//!
//! Under J2, the per-angle harmonic contribution kernel is
//! JIT-composed at `ForceField::new` time and dispatched from the
//! angle composed module; there is no standalone
//! `harmonic_angle_force` launcher. These tests drive the slot
//! through a `ForceField` instance and assert on the per-particle
//! force / energy / virial outputs.

use heddle_md::forces::topology::Angle;
use heddle_md::forces::{AggregateLevel, AngleList, BondList, DihedralList, ExclusionList, ForceField, HarmonicAngleBuilder, PotentialRegistry};
use heddle_md::gpu::{GpuContext, ParticleBuffers, init_device};
use heddle_md::io::config::{AngleTypeConfig, NeighborListConfig};
use heddle_md::pbc::SimulationBox;
use heddle_md::precision::Real;
use heddle_md::state::ParticleState;
use heddle_md::timings::Timings;

fn box_10(gpu: &GpuContext) -> SimulationBox {
    SimulationBox::new(&gpu.device, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0).unwrap()
}

fn harmonic_type(k_theta: f64, theta_0: f64) -> AngleTypeConfig {
    AngleTypeConfig::Harmonic {
        name: "AAA".to_string(),
        k_theta,
        theta_0,
    }
}

fn three_particle_state(positions: [[Real; 3]; 3]) -> ParticleState {
    ParticleState::new(
        positions.iter().map(|p| p[0]).collect(),
        positions.iter().map(|p| p[1]).collect(),
        positions.iter().map(|p| p[2]).collect(),
        vec![0.0; 3],
        vec![0.0; 3],
        vec![0.0; 3],
        vec![1.0; 3],
        vec![0.0; 3],
        vec![0u32; 3],
        None,
        None,
    )
    .unwrap()
}

fn single_angle_list(n: usize, i: u32, j: u32, k: u32) -> AngleList {
    let angles = vec![Angle {
        atom_i: i,
        atom_j: j,
        atom_k: k,
        angle_type_index: 0,
    }];
    let mut counts = vec![0u32; n];
    counts[i as usize] += 1;
    counts[j as usize] += 1;
    counts[k as usize] += 1;
    let mut atom_angle_offsets = vec![0u32; n + 1];
    let mut running = 0u32;
    for a in 0..n {
        atom_angle_offsets[a] = running;
        running += counts[a];
    }
    atom_angle_offsets[n] = running;
    // For one angle (m = 0), the per-atom-buffer slots are:
    //   slot 0 → atom_i, slot 1 → atom_j, slot 2 → atom_k.
    let mut per_atom: Vec<Vec<u32>> = vec![Vec::new(); n];
    per_atom[i as usize].push(0);
    per_atom[j as usize].push(1);
    per_atom[k as usize].push(2);
    let mut atom_angle_indices = Vec::new();
    for a in 0..n {
        for &idx in &per_atom[a] {
            atom_angle_indices.push(idx);
        }
    }
    AngleList {
        angles,
        atom_angle_offsets,
        atom_angle_indices,
        particle_count: n,
    }
}

fn place_isoceles(d: Real, theta: Real) -> [[Real; 3]; 3] {
    // atom_j at origin; atom_i along +x at distance d; atom_k at
    // distance d, angle theta from atom_i around atom_j.
    [
        [d, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [d * theta.cos(), d * theta.sin(), 0.0],
    ]
}

struct AngleResult {
    forces_x: Vec<Real>,
    forces_y: Vec<Real>,
    forces_z: Vec<Real>,
    energies: Vec<Real>,
    virials: Vec<Real>,
}

fn run_angle(
    gpu: &GpuContext,
    state: &ParticleState,
    angles: &AngleList,
    angle_types: &[AngleTypeConfig],
) -> AngleResult {
    let n = state.positions_x.len();
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(HarmonicAngleBuilder));
    let mut ff = ForceField::new(
        &registry,
        gpu,
        n,
        &box_10(gpu),
        &[],
        &[],
        &[],
        angle_types,
        &[],
        None,
        None,
        &[],
        &BondList::empty(n),
        angles,
        &DihedralList::empty(n),
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
    AngleResult {
        forces_x: gpu.device.dtoh_sync_copy(&buffers.forces_x).unwrap(),
        forces_y: gpu.device.dtoh_sync_copy(&buffers.forces_y).unwrap(),
        forces_z: gpu.device.dtoh_sync_copy(&buffers.forces_z).unwrap(),
        energies: gpu.device.dtoh_sync_copy(&buffers.potential_energies).unwrap(),
        virials: gpu.device.dtoh_sync_copy(&buffers.virials).unwrap(),
    }
}

#[test]
fn init_device_loads_angle_module() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    assert!(device.get_func("angle", "reduce_angle_forces").is_some());
    let _ = gpu.kernels.angle.reduce_angle_forces.clone();
}

#[test]
fn equilibrium_angle_produces_zero_force() {
    let gpu = init_device().unwrap();
    let theta_0: Real = std::f32::consts::FRAC_PI_2 as Real;
    let state = three_particle_state(place_isoceles(1.0, theta_0));
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(1.0, theta_0 as f64)];
    let r = run_angle(&gpu, &state, &al, &at);
    for v in r.forces_x.iter().chain(r.forces_y.iter()).chain(r.forces_z.iter()) {
        assert!(v.abs() < 1e-4, "expected zero force, got {v}");
    }
}

#[test]
fn energy_matches_closed_form() {
    let gpu = init_device().unwrap();
    let theta_0: Real = std::f32::consts::FRAC_PI_2 as Real;
    let theta: Real = theta_0 + 0.2;
    let k_theta: Real = 5.0;
    let state = three_particle_state(place_isoceles(1.0, theta));
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(k_theta as f64, theta_0 as f64)];
    let r = run_angle(&gpu, &state, &al, &at);
    let u_expected: Real = 0.5 * k_theta * (theta - theta_0) * (theta - theta_0);
    let total: Real = r.energies.iter().sum();
    let rel = (total - u_expected).abs() / u_expected.abs().max(1e-6);
    assert!(rel < 1e-4, "expected U={}, got {}", u_expected, total);
}

#[test]
fn perturbed_angle_force_satisfies_newtons_third_law() {
    let gpu = init_device().unwrap();
    let theta_0: Real = std::f32::consts::FRAC_PI_2 as Real;
    let theta: Real = theta_0 + 0.3;
    let state = three_particle_state(place_isoceles(1.0, theta));
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(1.0, theta_0 as f64)];
    let r = run_angle(&gpu, &state, &al, &at);
    let sum_x: Real = r.forces_x.iter().sum();
    let sum_y: Real = r.forces_y.iter().sum();
    let sum_z: Real = r.forces_z.iter().sum();
    assert!(sum_x.abs() < 1e-4);
    assert!(sum_y.abs() < 1e-4);
    assert!(sum_z.abs() < 1e-4);
}

#[test]
fn degenerate_geometry_produces_zero_force() {
    let gpu = init_device().unwrap();
    let state = three_particle_state([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(1.0, 1.0)];
    let r = run_angle(&gpu, &state, &al, &at);
    for v in r.forces_x.iter().chain(r.forces_y.iter()).chain(r.forces_z.iter()) {
        assert!(v.abs() < 1e-12, "expected zero, got {v}");
    }
}

#[test]
fn near_collinear_geometry_safety_guard_zeros() {
    let gpu = init_device().unwrap();
    let state = three_particle_state([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]);
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(1.0, std::f32::consts::FRAC_PI_2 as f64)];
    let r = run_angle(&gpu, &state, &al, &at);
    for v in r.forces_x.iter().chain(r.forces_y.iter()).chain(r.forces_z.iter()) {
        assert!(v.abs() < 1e-4, "expected zero, got {v}");
    }
}

#[test]
fn two_independent_runs_byte_identical() {
    let gpu = init_device().unwrap();
    let theta_0: Real = std::f32::consts::FRAC_PI_2 as Real;
    let state = three_particle_state(place_isoceles(1.0, theta_0 + 0.1));
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(1.0, theta_0 as f64)];
    let a = run_angle(&gpu, &state, &al, &at);
    let b = run_angle(&gpu, &state, &al, &at);
    assert_eq!(a.forces_x, b.forces_x);
    assert_eq!(a.forces_y, b.forces_y);
    assert_eq!(a.forces_z, b.forces_z);
    assert_eq!(a.energies, b.energies);
    assert_eq!(a.virials, b.virials);
}
