use std::f64::consts::PI;

use dynamics::forces::{
    AngleList, BondList, ExclusionList, ForceField, HarmonicAngleState,
};
use dynamics::forces::topology::Angle;
use dynamics::gpu::{ParticleBuffers, harmonic_angle_force, init_device};
use dynamics::io::config::{
    AngleTypeConfig, BondTypeConfig, NeighborListConfig, PairInteractionConfig,
    PairPotentialParams, ParticleTypeConfig,
};
use dynamics::pbc::SimulationBox;
use dynamics::state::ParticleState;
use dynamics::timings::Timings;

fn box_10() -> SimulationBox {
    SimulationBox::new(10.0, 10.0, 10.0, 0.0, 0.0, 0.0).unwrap()
}

fn box_nm() -> SimulationBox {
    let l = 2.0e-9_f32;
    SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap()
}

fn three_particle_state(positions: [[f32; 3]; 3], charges: [f32; 3]) -> ParticleState {
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
        vec![0.0_f32; 3],
        vec![0.0_f32; 3],
        vec![0.0_f32; 3],
        vec![1.0_f32; 3],
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

#[test]
fn init_device_loads_angle_module() {
    // Loading the device exposes the angle kernels on the Kernels handle.
    let _gpu = init_device().unwrap();
    // No assertion: a missing kernel would have panicked inside `init_device`.
}

#[test]
fn construct_harmonic_angle_state() {
    let gpu = init_device().unwrap();
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(383.0, 1.911)];
    let state = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    assert_eq!(state.angle_count, 1);
    assert_eq!(state.particle_count, 3);
    let k: Vec<f32> = gpu.device.dtoh_sync_copy(&state.angle_k_theta).unwrap();
    let t0: Vec<f32> = gpu.device.dtoh_sync_copy(&state.angle_theta_0).unwrap();
    assert_eq!(k.len(), 1);
    assert!((k[0] - 383.0_f32).abs() < 1.0e-3);
    assert!((t0[0] - 1.911_f32).abs() < 1.0e-6);
}

// Place atom i and atom k symmetrically about atom j (the centre) so the
// included angle at j equals `theta`. Edge length d_ij = d_kj = d.
fn place_isoceles(d: f32, theta: f32) -> [[f32; 3]; 3] {
    let half = 0.5 * theta;
    let dx = d * half.sin();
    let dy = d * half.cos();
    // atom_j at origin; atom_i and atom_k symmetric about +y axis at
    // (-dx, dy, 0) and (+dx, dy, 0). Geometric angle subtended at j is θ.
    [[-dx, dy, 0.0], [0.0, 0.0, 0.0], [dx, dy, 0.0]]
}

fn launch_angle_force(
    gpu: &dynamics::gpu::GpuContext,
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

#[test]
fn equilibrium_angle_produces_zero_force() {
    let gpu = init_device().unwrap();
    let theta_0 = 1.911_f32;
    let positions = place_isoceles(1.0, theta_0);
    let state = three_particle_state(positions, [0.0; 3]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(383.0, theta_0 as f64)];
    let mut s = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    launch_angle_force(&gpu, &mut s, &buffers, &box_10());
    let fx: Vec<f32> = gpu.device.dtoh_sync_copy(&s.angle_triple_x).unwrap();
    let fy: Vec<f32> = gpu.device.dtoh_sync_copy(&s.angle_triple_y).unwrap();
    let fz: Vec<f32> = gpu.device.dtoh_sync_copy(&s.angle_triple_z).unwrap();
    // f32 trig rounding gives δθ of order 1e-7 at equilibrium; with k=383
    // and a unit-magnitude geometric prefactor, the residual force is
    // ~k·δθ ≈ a few × 10⁻⁴. Use a tolerance scaled by k.
    let tol = 383.0_f32 * 1.0e-6;
    for slot in 0..3 {
        assert!(
            fx[slot].abs() < tol && fy[slot].abs() < tol && fz[slot].abs() < tol,
            "slot {slot}: |F| should be ~0 (tol {tol}), got ({}, {}, {})",
            fx[slot], fy[slot], fz[slot]
        );
    }
}

#[test]
fn compressed_angle_pushes_wings_apart() {
    let gpu = init_device().unwrap();
    let theta_0 = 1.911_f32;
    let theta = theta_0 - 0.1_f32;
    let positions = place_isoceles(1.0, theta);
    let state = three_particle_state(positions, [0.0; 3]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(383.0, theta_0 as f64)];
    let mut s = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    launch_angle_force(&gpu, &mut s, &buffers, &box_10());
    let fx: Vec<f32> = gpu.device.dtoh_sync_copy(&s.angle_triple_x).unwrap();
    let fy: Vec<f32> = gpu.device.dtoh_sync_copy(&s.angle_triple_y).unwrap();
    let fz: Vec<f32> = gpu.device.dtoh_sync_copy(&s.angle_triple_z).unwrap();
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

#[test]
fn stretched_angle_pulls_wings_together() {
    let gpu = init_device().unwrap();
    let theta_0 = 1.911_f32;
    let theta = theta_0 + 0.1_f32;
    let positions = place_isoceles(1.0, theta);
    let state = three_particle_state(positions, [0.0; 3]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(383.0, theta_0 as f64)];
    let mut s = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    launch_angle_force(&gpu, &mut s, &buffers, &box_10());
    let fx: Vec<f32> = gpu.device.dtoh_sync_copy(&s.angle_triple_x).unwrap();
    // Stretched (θ > θ₀): restoring torque closes the angle, pulling each
    // wing inward. Slot 0 holds left wing → fx should be positive (toward
    // centre). Slot 2 holds right wing → fx should be negative.
    assert!(fx[0] > 0.0, "stretched: fx on left wing should be positive, got {}", fx[0]);
    assert!(fx[2] < 0.0, "stretched: fx on right wing should be negative, got {}", fx[2]);
}

#[test]
fn energy_matches_closed_form() {
    let gpu = init_device().unwrap();
    let theta_0 = 1.911_f32;
    let theta = theta_0 + 0.2_f32;
    let positions = place_isoceles(1.0, theta);
    let state = three_particle_state(positions, [0.0; 3]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let al = single_angle_list(3, 0, 1, 2);
    let k = 383.0_f64;
    let at = vec![harmonic_type(k, theta_0 as f64)];
    let mut s = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    launch_angle_force(&gpu, &mut s, &buffers, &box_10());
    let e: Vec<f32> = gpu.device.dtoh_sync_copy(&s.angle_triple_energy).unwrap();
    let total: f64 = e.iter().map(|&v| v as f64).sum();
    let dtheta = (theta - theta_0) as f64;
    let expected = 0.5 * k * dtheta * dtheta;
    let rel = (total - expected).abs() / expected.abs();
    assert!(rel < 1.0e-4, "energy {} vs expected {} (rel {})", total, expected, rel);
}

#[test]
fn degenerate_geometry_produces_zero() {
    let gpu = init_device().unwrap();
    let positions = [[0.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let state = three_particle_state(positions, [0.0; 3]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(383.0, 1.911)];
    let mut s = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    launch_angle_force(&gpu, &mut s, &buffers, &box_10());
    let fx: Vec<f32> = gpu.device.dtoh_sync_copy(&s.angle_triple_x).unwrap();
    let fy: Vec<f32> = gpu.device.dtoh_sync_copy(&s.angle_triple_y).unwrap();
    let fz: Vec<f32> = gpu.device.dtoh_sync_copy(&s.angle_triple_z).unwrap();
    let e: Vec<f32> = gpu.device.dtoh_sync_copy(&s.angle_triple_energy).unwrap();
    for slot in 0..3 {
        assert_eq!(fx[slot], 0.0, "slot {slot} fx not zero on degenerate input");
        assert_eq!(fy[slot], 0.0, "slot {slot} fy not zero on degenerate input");
        assert_eq!(fz[slot], 0.0, "slot {slot} fz not zero on degenerate input");
        assert_eq!(e[slot], 0.0, "slot {slot} energy not zero on degenerate input");
    }
}

#[test]
fn near_collinear_geometry_safety_guard_zeros() {
    let gpu = init_device().unwrap();
    // theta very close to π — sin θ < 1e-7.
    let positions = [
        [-1.0_f32, 1.0e-9, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0e-9, 0.0],
    ];
    let state = three_particle_state(positions, [0.0; 3]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(383.0, 1.911)];
    let mut s = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    launch_angle_force(&gpu, &mut s, &buffers, &box_10());
    let fx: Vec<f32> = gpu.device.dtoh_sync_copy(&s.angle_triple_x).unwrap();
    let fy: Vec<f32> = gpu.device.dtoh_sync_copy(&s.angle_triple_y).unwrap();
    let fz: Vec<f32> = gpu.device.dtoh_sync_copy(&s.angle_triple_z).unwrap();
    for slot in 0..3 {
        assert!(fx[slot].is_finite() && fy[slot].is_finite() && fz[slot].is_finite());
        assert_eq!(fx[slot], 0.0, "slot {slot} should be zero near collinear");
        assert_eq!(fy[slot], 0.0);
        assert_eq!(fz[slot], 0.0);
    }
}

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

#[test]
fn two_independent_constructs_produce_byte_identical_outputs() {
    let gpu = init_device().unwrap();
    let theta_0 = 1.911_f32;
    let positions = place_isoceles(1.0, theta_0 + 0.05);
    let state = three_particle_state(positions, [0.0; 3]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let al = single_angle_list(3, 0, 1, 2);
    let at = vec![harmonic_type(383.0, theta_0 as f64)];
    let mut s1 = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    let mut s2 = HarmonicAngleState::new(&gpu, &al, &at).unwrap();
    launch_angle_force(&gpu, &mut s1, &buffers, &box_10());
    launch_angle_force(&gpu, &mut s2, &buffers, &box_10());
    let a: Vec<f32> = gpu.device.dtoh_sync_copy(&s1.angle_triple_x).unwrap();
    let b: Vec<f32> = gpu.device.dtoh_sync_copy(&s2.angle_triple_x).unwrap();
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

#[test]
fn spc_single_step_satisfies_newtons_third_law() {
    let gpu = init_device().unwrap();
    let theta_0 = 1.911_f32;
    let theta = theta_0 + 0.05_f32; // small displacement so angle force is non-trivial
    let d_oh = 1.0e-10_f32;
    let positions = place_isoceles(d_oh, theta);

    let e_charge: f32 = 1.602176634e-19;
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
    use dynamics::forces::topology::Bond;
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
            dynamics::forces::Exclusion { atom_i: 0, atom_j: 1, scale_lj: 0.0, scale_coul: 0.0 },
            dynamics::forces::Exclusion { atom_i: 0, atom_j: 2, scale_lj: 0.0, scale_coul: 0.0 },
            dynamics::forces::Exclusion { atom_i: 1, atom_j: 2, scale_lj: 0.0, scale_coul: 0.0 },
        ],
        atom_excl_offsets: vec![0, 2, 4, 6],
        atom_excl_partners: vec![1, 2, 0, 2, 0, 1],
        atom_excl_lj_scales: vec![0.0; 6],
        atom_excl_coul_scales: vec![0.0; 6],
        particle_count: 3,
    };

    let charges = vec![0.41 * e_charge, -0.82 * e_charge, 0.41 * e_charge];

    let mut ff = ForceField::new(
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
    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();

    let fx = gpu.device.dtoh_sync_copy(&buffers.forces_x).unwrap();
    let fy = gpu.device.dtoh_sync_copy(&buffers.forces_y).unwrap();
    let fz = gpu.device.dtoh_sync_copy(&buffers.forces_z).unwrap();

    // Newton's third law on the closed three-atom system: total force is
    // zero on every axis (well within f32 round-off).
    let sx = fx[0] + fx[1] + fx[2];
    let sy = fy[0] + fy[1] + fy[2];
    let sz = fz[0] + fz[1] + fz[2];
    let scale = fx.iter().chain(&fy).chain(&fz).map(|v| v.abs()).fold(0.0_f32, f32::max);
    let tol = (scale * 1.0e-4).max(1.0e-25);
    assert!(sx.abs() < tol, "Σfx = {sx}, scale = {scale}");
    assert!(sy.abs() < tol, "Σfy = {sy}, scale = {scale}");
    assert!(sz.abs() < tol, "Σfz = {sz}, scale = {scale}");

    // Per-atom forces are finite and non-zero (the system is off-eq).
    for atom in 0..3 {
        assert!(fx[atom].is_finite() && fy[atom].is_finite() && fz[atom].is_finite());
    }
    let mag_total: f32 = (0..3)
        .map(|i| (fx[i] * fx[i] + fy[i] * fy[i] + fz[i] * fz[i]).sqrt())
        .sum();
    assert!(mag_total > 0.0, "off-equilibrium system should produce non-zero net forces");

    let _ = PI; // silence import if unused on this path
}
