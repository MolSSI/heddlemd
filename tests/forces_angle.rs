// rq-d9adc4cb Feature: Harmonic Angle Bonded Potential
use std::f64::consts::PI;

use heddle_md::forces::{AggregateLevel, AngleList, BondList, ExclusionList, ForceField, HarmonicAngleState, PotentialRegistry};
use heddle_md::forces::topology::Angle;
use heddle_md::gpu::{ParticleBuffers, harmonic_angle_force, init_device};
use heddle_md::io::config::{
    AngleTypeConfig, BondTypeConfig, NeighborListConfig, PairInteractionConfig,
    PairPotentialParams, ParticleTypeConfig,
};
use heddle_md::pbc::SimulationBox;
use heddle_md::state::ParticleState;
use heddle_md::timings::Timings;
use heddle_md::precision::Real;

fn box_10() -> SimulationBox {
    SimulationBox::new(10.0, 10.0, 10.0, 0.0, 0.0, 0.0).unwrap()
}

fn box_nm() -> SimulationBox {
    let l = 2.0e-9;
    SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap()
}

fn three_particle_state(positions: [[Real; 3]; 3], charges: [Real; 3]) -> ParticleState {
    let mut px = Vec::with_capacity(3);
    let mut py = Vec::with_capacity(3);
    let mut pz = Vec::with_capacity(3);
    for p in &positions {
        px.push(p[0]);
        py.push(p[1]);
        pz.push(p[2]);
    }
    ParticleState::new(
        px,
        py,
        pz,
        vec![0.0; 3],
        vec![0.0; 3],
        vec![0.0; 3],
        vec![1.0; 3],
        charges.to_vec(),
        vec![0u32; 3],
        None,
        None,
    )
    .expect("ParticleState::new")
}

fn harmonic_type(k_theta: f64, theta_0: f64) -> AngleTypeConfig {
    AngleTypeConfig::Harmonic {
        name: "HOH".to_string(),
        k_theta,
        theta_0,
    }
}

fn single_angle_list(n: usize, i: u32, j: u32, k: u32) -> AngleList {
    let angles = vec![Angle {
        atom_i: i.min(k),
        atom_j: j,
        atom_k: i.max(k),
        angle_type_index: 0,
    }];
    let mut atom_angle_offsets = vec![0u32; n + 1];
    for a in &angles {
        atom_angle_offsets[a.atom_i as usize + 1] += 1;
        atom_angle_offsets[a.atom_j as usize + 1] += 1;
        atom_angle_offsets[a.atom_k as usize + 1] += 1;
    }
    for x in 1..=n {
        atom_angle_offsets[x] += atom_angle_offsets[x - 1];
    }
    let mut atom_angle_indices = vec![0u32; angles.len() * 3];
    let mut cursor: Vec<u32> = atom_angle_offsets[..n].to_vec();
    for (m, a) in angles.iter().enumerate() {
        let s_i = (3 * m) as u32;
        let s_j = (3 * m + 1) as u32;
        let s_k = (3 * m + 2) as u32;
        atom_angle_indices[cursor[a.atom_i as usize] as usize] = s_i;
        cursor[a.atom_i as usize] += 1;
        atom_angle_indices[cursor[a.atom_j as usize] as usize] = s_j;
        cursor[a.atom_j as usize] += 1;
        atom_angle_indices[cursor[a.atom_k as usize] as usize] = s_k;
        cursor[a.atom_k as usize] += 1;
    }
    AngleList {
        angles,
        atom_angle_offsets,
        atom_angle_indices,
        particle_count: n,
    }
}

// rq-f7a71238
#[test]
fn init_device_loads_angle_module() {
    // Loading the device exposes the angle kernels on the Kernels handle.
    let _gpu = init_device().unwrap();
    // No assertion: a missing kernel would have panicked inside `init_device`.
}

// rq-dbee9f45
#[test]
fn construct_harmonic_angle_state() {
    let gpu = init_device().unwrap();
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(383.0, 1.911)];
    let state = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    assert_eq!(state.angle_count, 1);
    assert_eq!(state.particle_count, 3);
    let k: Vec<Real> = gpu.device.dtoh_sync_copy(&state.angle_k_theta).unwrap();
    let t0: Vec<Real> = gpu.device.dtoh_sync_copy(&state.angle_theta_0).unwrap();
    assert_eq!(k.len(), 1);
    assert!((k[0] - 383.0).abs() < 1.0e-3);
    assert!((t0[0] - 1.911).abs() < 1.0e-6);
}

// Place atom i and atom k symmetrically about atom j (the centre) so the
// included angle at j equals `theta`. Edge length d_ij = d_kj = d.
fn place_isoceles(d: Real, theta: Real) -> [[Real; 3]; 3] {
    let half = 0.5 * theta;
    let dx = d * half.sin();
    let dy = d * half.cos();
    // atom_j at origin; atom_i and atom_k symmetric about +y axis at
    // (-dx, dy, 0) and (+dx, dy, 0). Geometric angle subtended at j is θ.
    [[-dx, dy, 0.0], [0.0, 0.0, 0.0], [dx, dy, 0.0]]
}

fn launch_angle_force(
    gpu: &heddle_md::gpu::GpuContext,
    state: &mut HarmonicAngleState,
    buffers: &ParticleBuffers,
    sim_box: &SimulationBox,
) {
    harmonic_angle_force(
        buffers,
        &state.angles,
        &state.angle_k_theta,
        &state.angle_theta_0,
        sim_box,
        &mut state.angle_triple_x,
        &mut state.angle_triple_y,
        &mut state.angle_triple_z,
        &mut state.angle_triple_energy,
        &mut state.angle_triple_virial,
        state.angle_count,
    )
    .unwrap();
    let _ = gpu;
}

// rq-a57bcebe rq-9bb3094c
#[test]
fn equilibrium_angle_produces_zero_force() {
    let gpu = init_device().unwrap();
    let theta_0 = 1.911;
    let positions = place_isoceles(1.0, theta_0);
    let state = three_particle_state(positions, [0.0; 3]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(383.0, theta_0 as f64)];
    let mut s = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    launch_angle_force(&gpu, &mut s, &buffers, &box_10());
    let fx: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_x).unwrap();
    let fy: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_y).unwrap();
    let fz: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_z).unwrap();
    // f32 trig rounding gives δθ of order 1e-7 at equilibrium; with k=383
    // and a unit-magnitude geometric prefactor, the residual force is
    // ~k·δθ ≈ a few × 10⁻⁴. Use a tolerance scaled by k.
    let tol = 383.0 * 1.0e-6;
    for slot in 0..3 {
        assert!(
            fx[slot].abs() < tol && fy[slot].abs() < tol && fz[slot].abs() < tol,
            "slot {slot}: |F| should be ~0 (tol {tol}), got ({}, {}, {})",
            fx[slot], fy[slot], fz[slot]
        );
    }
}

// rq-e60a2781
#[test]
fn compressed_angle_pushes_wings_apart() {
    let gpu = init_device().unwrap();
    let theta_0 = 1.911;
    let theta = theta_0 - 0.1;
    let positions = place_isoceles(1.0, theta);
    let state = three_particle_state(positions, [0.0; 3]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(383.0, theta_0 as f64)];
    let mut s = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    launch_angle_force(&gpu, &mut s, &buffers, &box_10());
    let fx: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_x).unwrap();
    let fy: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_y).unwrap();
    let fz: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_z).unwrap();
    // Bisector along +y; compressed (θ < θ₀) means restoring torque opens
    // the angle, pushing each wing further along its outward x direction.
    // Slot 0 holds atom_i (left wing, dx negative); its fx should be more
    // negative. Slot 2 holds atom_k (right wing); its fx should be positive.
    assert!(fx[0] < 0.0, "compressed: fx on left wing should be negative, got {}", fx[0]);
    assert!(fx[2] > 0.0, "compressed: fx on right wing should be positive, got {}", fx[2]);
    // Newton's third law over the three atoms (sum to zero).
    let sx = fx[0] + fx[1] + fx[2];
    let sy = fy[0] + fy[1] + fy[2];
    let sz = fz[0] + fz[1] + fz[2];
    assert!(sx.abs() < 1.0e-4, "Σfx = {sx}");
    assert!(sy.abs() < 1.0e-4, "Σfy = {sy}");
    assert!(sz.abs() < 1.0e-4, "Σfz = {sz}");
}

// rq-98fd2e40
#[test]
fn stretched_angle_pulls_wings_together() {
    let gpu = init_device().unwrap();
    let theta_0 = 1.911;
    let theta = theta_0 + 0.1;
    let positions = place_isoceles(1.0, theta);
    let state = three_particle_state(positions, [0.0; 3]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(383.0, theta_0 as f64)];
    let mut s = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    launch_angle_force(&gpu, &mut s, &buffers, &box_10());
    let fx: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_x).unwrap();
    // Stretched (θ > θ₀): restoring torque closes the angle, pulling each
    // wing inward. Slot 0 holds left wing → fx should be positive (toward
    // centre). Slot 2 holds right wing → fx should be negative.
    assert!(fx[0] > 0.0, "stretched: fx on left wing should be positive, got {}", fx[0]);
    assert!(fx[2] < 0.0, "stretched: fx on right wing should be negative, got {}", fx[2]);
}

// rq-ee7566b4
#[test]
fn energy_matches_closed_form() {
    let gpu = init_device().unwrap();
    let theta_0 = 1.911;
    let theta = theta_0 + 0.2;
    let positions = place_isoceles(1.0, theta);
    let state = three_particle_state(positions, [0.0; 3]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let al = single_angle_list(3, 0, 1, 2);
    let k = 383.0_f64;
    let at = vec![harmonic_type(k, theta_0 as f64)];
    let mut s = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    launch_angle_force(&gpu, &mut s, &buffers, &box_10());
    let e: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_energy).unwrap();
    let total: f64 = e.iter().map(|&v| v as f64).sum();
    let dtheta = (theta - theta_0) as f64;
    let expected = 0.5 * k * dtheta * dtheta;
    let rel = (total - expected).abs() / expected.abs();
    assert!(rel < 1.0e-4, "energy {} vs expected {} (rel {})", total, expected, rel);
}

// rq-4ffdad62 rq-a587753e
#[test]
fn degenerate_geometry_produces_zero() {
    let gpu = init_device().unwrap();
    let positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let state = three_particle_state(positions, [0.0; 3]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(383.0, 1.911)];
    let mut s = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    launch_angle_force(&gpu, &mut s, &buffers, &box_10());
    let fx: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_x).unwrap();
    let fy: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_y).unwrap();
    let fz: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_z).unwrap();
    let e: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_energy).unwrap();
    for slot in 0..3 {
        assert_eq!(fx[slot], 0.0, "slot {slot} fx not zero on degenerate input");
        assert_eq!(fy[slot], 0.0, "slot {slot} fy not zero on degenerate input");
        assert_eq!(fz[slot], 0.0, "slot {slot} fz not zero on degenerate input");
        assert_eq!(e[slot], 0.0, "slot {slot} energy not zero on degenerate input");
    }
}

// rq-bd367201
#[test]
fn near_collinear_geometry_safety_guard_zeros() {
    let gpu = init_device().unwrap();
    // theta very close to π — sin θ < 1e-7.
    let positions = [
        [-1.0, 1.0e-9, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0e-9, 0.0],
    ];
    let state = three_particle_state(positions, [0.0; 3]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(383.0, 1.911)];
    let mut s = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    launch_angle_force(&gpu, &mut s, &buffers, &box_10());
    let fx: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_x).unwrap();
    let fy: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_y).unwrap();
    let fz: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_z).unwrap();
    for slot in 0..3 {
        assert!(fx[slot].is_finite() && fy[slot].is_finite() && fz[slot].is_finite());
        assert_eq!(fx[slot], 0.0, "slot {slot} should be zero near collinear");
        assert_eq!(fy[slot], 0.0);
        assert_eq!(fz[slot], 0.0);
    }
}

// rq-cf50db39
#[test]
fn harmonic_angle_force_zero_angles_is_noop() {
    let gpu = init_device().unwrap();
    let al = AngleList::empty(3);
    let at: Vec<AngleTypeConfig> = vec![];
    let mut s = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    let state = three_particle_state([[0.0; 3]; 3], [0.0; 3]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    launch_angle_force(&gpu, &mut s, &buffers, &box_10());
}

// rq-9120ab3c
#[test]
fn two_independent_constructs_produce_byte_identical_outputs() {
    let gpu = init_device().unwrap();
    let theta_0 = 1.911;
    let positions = place_isoceles(1.0, theta_0 + 0.05);
    let state = three_particle_state(positions, [0.0; 3]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(383.0, theta_0 as f64)];
    let mut s1 = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    let mut s2 = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    launch_angle_force(&gpu, &mut s1, &buffers, &box_10());
    launch_angle_force(&gpu, &mut s2, &buffers, &box_10());
    let a: Vec<Real> = gpu.device.dtoh_sync_copy(&s1.angle_triple_x).unwrap();
    let b: Vec<Real> = gpu.device.dtoh_sync_copy(&s2.angle_triple_x).unwrap();
    assert_eq!(a, b, "two independent runs must agree byte-for-byte");
}

// --- SPC water single-step smoke test ---
//
// Single SPC water molecule, slightly off-equilibrium. The angle slot
// produces non-trivial forces; Morse bonds tuned so 2·D_e·a² matches
// the SPC harmonic stiffness 4.515e5 J/m² at r_e = 1.0 Å are also
// active. The test runs one ForceField::step and checks Newton's third
// law (force sum over the three atoms is zero per axis) — the strongest
// test that doesn't require redundantly reimplementing the bond+angle
// force formulas in the test.

fn spc_morse_tuned() -> BondTypeConfig {
    // d_OH = 1.0e-10 m (SPC). harmonic k_b = 4.515e5 J/m² (flexible SPC).
    // For Morse, U''(r_e) = 2 D_e a². Pick d_e = 2.0e-19 J (≈ 0.5 eV) and
    // a = sqrt(k / (2 d_e)) ≈ sqrt(4.515e5 / 4.0e-19) ≈ 1.063e12 1/m.
    let de: f64 = 2.0e-19;
    let k_b: f64 = 4.515e5;
    let a: f64 = (k_b / (2.0 * de)).sqrt();
    BondTypeConfig::Morse {
        name: "OH".to_string(),
        de,
        a,
        re: 1.0e-10,
    }
}

// rq-501bce66 rq-b19189c2
#[test]
fn spc_single_step_satisfies_newtons_third_law() {
    let gpu = init_device().unwrap();
    let theta_0 = 1.911;
    let theta = theta_0 + 0.05; // small displacement so angle force is non-trivial
    let d_oh = 1.0e-10;
    let positions = place_isoceles(d_oh, theta);

    let e_charge: Real = 1.602176634e-19;
    // Build an O-H-H state: atom 0 = H (i), 1 = O (j, centre), 2 = H (k).
    let mut state = three_particle_state(
        positions,
        [0.41 * e_charge, -0.82 * e_charge, 0.41 * e_charge],
    );
    state.type_indices = vec![1, 0, 1]; // H, O, H
    state.masses = vec![1.6735e-27, 2.6566e-26, 1.6735e-27];

    let sim_box = box_nm();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();

    let particle_types = vec![
        ParticleTypeConfig {
            name: "O".to_string(),
            mass: 2.6566e-26,
            charge: -0.82 * e_charge as f64,
        },
        ParticleTypeConfig {
            name: "H".to_string(),
            mass: 1.6735e-27,
            charge: 0.41 * e_charge as f64,
        },
    ];
    // LJ on O-O only; cross terms zero. Cutoff = 0.5 nm.
    let pairs = vec![
        PairInteractionConfig {
            between: ("O".to_string(), "O".to_string()),
            cutoff: 0.5e-9,
            r_switch: 0.5e-9,
            potential: PairPotentialParams::LennardJones {
                sigma: 3.166e-10,
                epsilon: 6.502e-22,
            },
        },
        PairInteractionConfig {
            between: ("H".to_string(), "H".to_string()),
            cutoff: 0.5e-9,
            r_switch: 0.5e-9,
            potential: PairPotentialParams::LennardJones {
                sigma: 1.0e-10,
                epsilon: 0.0,
            },
        },
        PairInteractionConfig {
            between: ("H".to_string(), "O".to_string()),
            cutoff: 0.5e-9,
            r_switch: 0.5e-9,
            potential: PairPotentialParams::LennardJones {
                sigma: 1.0e-10,
                epsilon: 0.0,
            },
        },
    ];
    let bond_types = vec![spc_morse_tuned()];
    // Realistic flexible-SPC HOH bend stiffness (75.9 kcal/mol/rad²).
    let angle_types = vec![harmonic_type(5.27e-19, theta_0 as f64)];

    // Two OH bonds (H0-O1, H2-O1); one HOH angle at centre O1.
    use heddle_md::forces::topology::Bond;
    let bond_list = BondList {
        bonds: vec![
            Bond {
                atom_i: 0,
                atom_j: 1,
                bond_type_index: 0,
            },
            Bond {
                atom_i: 1,
                atom_j: 2,
                bond_type_index: 0,
            },
        ],
        atom_bond_offsets: vec![0, 1, 3, 4],
        atom_bond_indices: vec![0, 1, 2, 3],
        particle_count: 3,
    };
    let angle_list = single_angle_list(3, 0, 1, 2);
    // 1-2 exclusions (0,1) and (1,2) plus 1-3 exclusion (0,2). The angle
    // slot kernels handle the bonded forces directly; the LJ/Coulomb
    // kernels see all three pairs as fully excluded.
    let exclusion_list = ExclusionList {
        entries: vec![
            heddle_md::forces::Exclusion { atom_i: 0, atom_j: 1, scale_lj: 0.0, scale_coul: 0.0 },
            heddle_md::forces::Exclusion { atom_i: 0, atom_j: 2, scale_lj: 0.0, scale_coul: 0.0 },
            heddle_md::forces::Exclusion { atom_i: 1, atom_j: 2, scale_lj: 0.0, scale_coul: 0.0 },
        ],
        atom_excl_offsets: vec![0, 2, 4, 6],
        atom_excl_partners: vec![1, 2, 0, 2, 0, 1],
        atom_excl_lj_scales: vec![0.0; 6],
        atom_excl_coul_scales: vec![0.0; 6],
        particle_count: 3,
    };

    let charges = vec![0.41 * e_charge, -0.82 * e_charge, 0.41 * e_charge];

    let mut ff = ForceField::new(
        &PotentialRegistry::with_builtins(),
        &gpu,
        3,
        &sim_box,
        &particle_types,
        &pairs,
        &bond_types,
        &angle_types,
        None,
        None,
        &charges,
        &bond_list,
        &angle_list,
        &exclusion_list,
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &sim_box, &mut timings, AggregateLevel::ForcesAndScalars).unwrap();

    let fx = gpu.device.dtoh_sync_copy(&buffers.forces_x).unwrap();
    let fy = gpu.device.dtoh_sync_copy(&buffers.forces_y).unwrap();
    let fz = gpu.device.dtoh_sync_copy(&buffers.forces_z).unwrap();

    // Newton's third law on the closed three-atom system: total force is
    // zero on every axis (well within f32 round-off).
    let sx = fx[0] + fx[1] + fx[2];
    let sy = fy[0] + fy[1] + fy[2];
    let sz = fz[0] + fz[1] + fz[2];
    let scale = fx.iter().chain(&fy).chain(&fz).map(|v| v.abs()).fold(0.0, Real::max);
    let tol = (scale * 1.0e-4).max(1.0e-25);
    assert!(sx.abs() < tol, "Σfx = {sx}, scale = {scale}");
    assert!(sy.abs() < tol, "Σfy = {sy}, scale = {scale}");
    assert!(sz.abs() < tol, "Σfz = {sz}, scale = {scale}");

    // Per-atom forces are finite and non-zero (the system is off-eq).
    for atom in 0..3 {
        assert!(fx[atom].is_finite() && fy[atom].is_finite() && fz[atom].is_finite());
    }
    let mag_total: Real = (0..3)
        .map(|i| (fx[i] * fx[i] + fy[i] * fy[i] + fz[i] * fz[i]).sqrt())
        .sum();
    assert!(mag_total > 0.0, "off-equilibrium system should produce non-zero net forces");

    let _ = PI; // silence import if unused on this path
}

// --- Closed-form magnitude, virial, and minimum image -------------------

// rq-922e1683
#[test]
fn force_magnitude_matches_closed_form_for_an_isolated_angle() {
    let gpu = init_device().unwrap();
    let theta_0 = 1.911;
    let theta = theta_0 + 0.1;
    let d = 1.0e-10;
    let k_theta = 5.27e-19_f64;
    let positions = place_isoceles(d, theta);
    let state = three_particle_state(positions, [0.0; 3]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(k_theta, theta_0 as f64)];
    let mut s = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    launch_angle_force(&gpu, &mut s, &buffers, &box_nm());
    let fx: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_x).unwrap();
    let fy: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_y).unwrap();
    let fz: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_z).unwrap();
    let mag = |i: usize| -> f64 {
        let v = ((fx[i] as f64).powi(2) + (fy[i] as f64).powi(2) + (fz[i] as f64).powi(2)).sqrt();
        v
    };
    let total = mag(0) + mag(1) + mag(2);
    // Closed-form: for symmetric isosceles geometry with |r_ij|=|r_kj|=d,
    // |F_i| = |F_k| = k*Δθ/d and |F_j| = 2 k*Δθ*sin(θ/2)/d, so the sum
    // is (2 k Δθ / d) * (1 + sin(θ/2)).
    let dtheta = (theta - theta_0) as f64;
    let expected = (2.0 * k_theta * dtheta / d as f64) * (1.0 + (theta as f64 * 0.5).sin());
    let rel = (total - expected).abs() / expected.abs();
    assert!(rel < 5.0e-3, "force-magnitude sum {total:e} vs expected {expected:e} (rel {rel:e})");
}

// rq-fbdf08ff
#[test]
fn minimum_image_is_applied_to_displacements() {
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap();
    let theta_0 = 1.911;
    let at = vec![harmonic_type(5.27e-19, theta_0 as f64)];
    let al = single_angle_list(3, 0, 1, 2);

    // Setup A: triangle near origin, no minimum-image wrap.
    let positions_a = [
        [-0.15 * l, 0.05 * l, 0.0],
        [0.0, 0.0, 0.0],
        [0.10 * l, 0.05 * l, 0.0],
    ];
    // Setup B: same triangle shifted by -0.45*l. p_i lands at +0.40*l so
    // r_ij raw = p_i - p_j = +0.85*l (> l/2) and must wrap to -0.15*l for
    // the kernel to recover the same geometry.
    let positions_b = [
        [0.40 * l, 0.05 * l, 0.0],
        [-0.45 * l, 0.0, 0.0],
        [-0.35 * l, 0.05 * l, 0.0],
    ];

    let run = |positions: [[Real; 3]; 3]| -> (Vec<Real>, Vec<Real>, Vec<Real>) {
        let state = three_particle_state(positions, [0.0; 3]);
        let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
        let mut s = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
        launch_angle_force(&gpu, &mut s, &buffers, &sim_box);
        (
            gpu.device.dtoh_sync_copy(&s.angle_triple_x).unwrap(),
            gpu.device.dtoh_sync_copy(&s.angle_triple_y).unwrap(),
            gpu.device.dtoh_sync_copy(&s.angle_triple_z).unwrap(),
        )
    };
    let (ax, ay, az) = run(positions_a);
    let (bx, by, bz) = run(positions_b);
    let scale = ax.iter().chain(&ay).chain(&az).map(|v| v.abs()).fold(0.0, Real::max);
    assert!(scale > 0.0, "expected non-zero forces in baseline run");
    let tol = scale * 1.0e-4;
    for slot in 0..3 {
        assert!((ax[slot] - bx[slot]).abs() < tol, "fx[{slot}]: A={} B={}", ax[slot], bx[slot]);
        assert!((ay[slot] - by[slot]).abs() < tol, "fy[{slot}]: A={} B={}", ay[slot], by[slot]);
        assert!((az[slot] - bz[slot]).abs() < tol, "fz[{slot}]: A={} B={}", az[slot], bz[slot]);
    }
}

// rq-fe95ff5f
#[test]
fn angle_virial_equals_r_ij_dot_f_i_plus_r_kj_dot_f_k() {
    // A harmonic-angle force is purely tangential to both r_ij and
    // r_kj, so r_ij·F_i = 0 and r_kj·F_k = 0 in continuous math; both
    // the kernel-computed w_m and the host-side cross-check land at
    // f32 round-off after near-perfect cancellation. The scenario's
    // claim therefore reduces to: both quantities are small *at the
    // same scale*. We verify that |Σ virial slots| and |r_ij·F_i +
    // r_kj·F_k| are each bounded by a small multiple of the natural
    // pairwise scale |F_i|·|r_ij|.
    let gpu = init_device().unwrap();
    let positions = [
        [-1.2, 0.8, 0.0],
        [0.0, 0.0, 0.0],
        [0.6, 1.4, 0.0],
    ];
    let state = three_particle_state(positions, [0.0; 3]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(383.0, 1.911)];
    let mut s = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    launch_angle_force(&gpu, &mut s, &buffers, &box_10());
    let fx: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_x).unwrap();
    let fy: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_y).unwrap();
    let fz: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_z).unwrap();
    let virial: Vec<Real> = gpu.device.dtoh_sync_copy(&s.angle_triple_virial).unwrap();
    let r_ij = [
        positions[0][0] - positions[1][0],
        positions[0][1] - positions[1][1],
        positions[0][2] - positions[1][2],
    ];
    let r_kj = [
        positions[2][0] - positions[1][0],
        positions[2][1] - positions[1][1],
        positions[2][2] - positions[1][2],
    ];
    let host_w = r_ij[0] * fx[0] + r_ij[1] * fy[0] + r_ij[2] * fz[0]
        + r_kj[0] * fx[2] + r_kj[1] * fy[2] + r_kj[2] * fz[2];
    let summed_w = virial[0] + virial[1] + virial[2];
    let f_mag = ((fx[0] * fx[0] + fy[0] * fy[0] + fz[0] * fz[0]).sqrt()
        + (fx[2] * fx[2] + fy[2] * fy[2] + fz[2] * fz[2]).sqrt()) as f64;
    let r_mag = ((r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2]).sqrt()
        + (r_kj[0] * r_kj[0] + r_kj[1] * r_kj[1] + r_kj[2] * r_kj[2]).sqrt()) as f64;
    let scale = f_mag * r_mag;
    assert!(scale > 1.0, "expected a non-trivial r·F scale (got {scale})");
    // 32-bit cancellation noise sits well below 1e-3 * scale.
    let tol = (scale * 1.0e-3) as Real;
    assert!(summed_w.abs() < tol, "Σ virial slots {summed_w:e} exceeds noise tol {tol:e}");
    assert!(host_w.abs() < tol, "host r_ij·F_i + r_kj·F_k {host_w:e} exceeds noise tol {tol:e}");
}

// --- Reduction kernel (atom → angle-triple summation) -------------------

fn alloc_and_run_reduce(
    gpu: &heddle_md::gpu::GpuContext,
    angle_triple_x: &[Real],
    angle_triple_y: &[Real],
    angle_triple_z: &[Real],
    angle_triple_energy: &[Real],
    angle_triple_virial: &[Real],
    atom_angle_offsets: &[u32],
    atom_angle_indices: &[u32],
    particle_count: usize,
) -> (Vec<Real>, Vec<Real>, Vec<Real>) {
    use cudarc::driver::DeviceSlice;
    let device = gpu.device.clone();
    let triple_x = device.htod_sync_copy(angle_triple_x).unwrap();
    let triple_y = device.htod_sync_copy(angle_triple_y).unwrap();
    let triple_z = device.htod_sync_copy(angle_triple_z).unwrap();
    let triple_e = device.htod_sync_copy(angle_triple_energy).unwrap();
    let triple_v = device.htod_sync_copy(angle_triple_virial).unwrap();
    let offsets = device.htod_sync_copy(atom_angle_offsets).unwrap();
    let indices = if atom_angle_indices.is_empty() {
        device.alloc_zeros::<u32>(1).unwrap()
    } else {
        device.htod_sync_copy(atom_angle_indices).unwrap()
    };
    let mut sx = device.alloc_zeros::<Real>(particle_count.max(1)).unwrap();
    let mut sy = device.alloc_zeros::<Real>(particle_count.max(1)).unwrap();
    let mut sz = device.alloc_zeros::<Real>(particle_count.max(1)).unwrap();
    let mut se = device.alloc_zeros::<Real>(particle_count.max(1)).unwrap();
    let mut sv = device.alloc_zeros::<Real>(particle_count.max(1)).unwrap();
    let upper_sx = sx.len();
    let upper_sy = sy.len();
    let upper_sz = sz.len();
    let upper_se = se.len();
    let upper_sv = sv.len();
    {
        let mut vx = sx.slice_mut(0..upper_sx);
        let mut vy = sy.slice_mut(0..upper_sy);
        let mut vz = sz.slice_mut(0..upper_sz);
        let mut ve = se.slice_mut(0..upper_se);
        let mut vv = sv.slice_mut(0..upper_sv);
        heddle_md::gpu::reduce_angle_forces(
            &gpu.kernels,
            &triple_x,
            &triple_y,
            &triple_z,
            &triple_e,
            &triple_v,
            &offsets,
            &indices,
            &mut vx, &mut vy, &mut vz, &mut ve, &mut vv,
            particle_count,
        ).unwrap();
    }
    (
        device.dtoh_sync_copy(&sx).unwrap(),
        device.dtoh_sync_copy(&sy).unwrap(),
        device.dtoh_sync_copy(&sz).unwrap(),
    )
}

// rq-27efd6a0
#[test]
fn atom_appearing_in_one_angle_receives_that_angles_force() {
    let gpu = init_device().unwrap();
    // One angle: slot 0 = atom_i contribution, slot 1 = atom_j, slot 2 = atom_k.
    let triple_x = vec![0.5, -1.0, 0.5];
    let triple_zeros = vec![0.0; 3];
    let atom_angle_offsets = vec![0u32, 1, 2, 3];
    let atom_angle_indices = vec![0u32, 1, 2];
    let (sx, _, _) = alloc_and_run_reduce(
        &gpu,
        &triple_x, &triple_zeros, &triple_zeros, &triple_zeros, &triple_zeros,
        &atom_angle_offsets,
        &atom_angle_indices,
        3,
    );
    assert_eq!(sx[0], 0.5);
    assert_eq!(sx[1], -1.0);
    assert_eq!(sx[2], 0.5);
}

// rq-ca76fc02 rq-699192b2
#[test]
fn atom_in_centre_and_wing_receives_sum_in_sorted_angle_index_order() {
    let gpu = init_device().unwrap();
    // Two angles. Layout: 6 slots, [F_i0, F_j0, F_k0, F_i1, F_j1, F_k1].
    // Atom 0 is the centre of angle 0 (slot 1) and a wing of angle 1
    // (slot 3 — i of angle 1).
    let triple_x = vec![2.5, 7.0, -1.5, 13.0, -2.0, 4.0];
    let triple_zeros = vec![0.0; 6];
    // Particle layout: 5 atoms, only atom 0 carries two angle slots.
    // atom_angle_indices for atom 0 carries [1, 3] (its centre slot for
    // angle 0 then its wing slot for angle 1) — sorted by angle index.
    let atom_angle_offsets = vec![0u32, 2, 2, 2, 2, 2];
    let atom_angle_indices = vec![1u32, 3];
    let (sx, _, _) = alloc_and_run_reduce(
        &gpu,
        &triple_x, &triple_zeros, &triple_zeros, &triple_zeros, &triple_zeros,
        &atom_angle_offsets,
        &atom_angle_indices,
        5,
    );
    // Reduction sums triple_x[1] + triple_x[3] = 7.0 + 13.0 = 20.0
    // left-to-right.
    assert_eq!(sx[0], 7.0 + 13.0);
}

// rq-5fcdc437
#[test]
fn atom_with_no_angles_gets_zero_accumulator() {
    let gpu = init_device().unwrap();
    // Two angles touching atoms 0..3 only; atom 4 has no angle.
    // Layout: 6 angle-triple slots.
    let triple_x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let triple_zeros = vec![0.0; 6];
    // Angle 0: (i=0, j=1, k=2) → slots [0, 1, 2].
    // Angle 1: (i=1, j=2, k=3) → slots [3, 4, 5].
    // atom_angle_offsets (length 6): cumulative per-atom slot count.
    //  atom 0 → 1 slot, atom 1 → 2, atom 2 → 2, atom 3 → 1, atom 4 → 0.
    let atom_angle_offsets = vec![0u32, 1, 3, 5, 6, 6];
    let atom_angle_indices = vec![0u32, 1, 3, 2, 4, 5];
    let (sx, sy, sz) = alloc_and_run_reduce(
        &gpu,
        &triple_x, &triple_zeros, &triple_zeros, &triple_zeros, &triple_zeros,
        &atom_angle_offsets,
        &atom_angle_indices,
        5,
    );
    assert_eq!(sx[4], 0.0, "atom 4 has no angles → fx must be 0");
    assert_eq!(sy[4], 0.0);
    assert_eq!(sz[4], 0.0);
}

// rq-8d5a8d9c
#[test]
fn reduce_angle_forces_on_zero_particles_is_a_noop() {
    let gpu = init_device().unwrap();
    let triple = Vec::<Real>::new();
    let offsets = vec![0u32];
    let indices = Vec::<u32>::new();
    // particle_count == 0 → reduce_angle_forces returns Ok(()) without
    // launching a kernel (verified by reaching the assertions below).
    let _ = alloc_and_run_reduce(
        &gpu, &triple, &triple, &triple, &triple, &triple,
        &offsets, &indices, 0,
    );
}
