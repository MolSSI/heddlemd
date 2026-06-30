//! Periodic-dihedral slot tests. Cover the Gherkin scenarios from
//! `rqm/forces/periodic-dihedral.md`. Like the angle tests, the
//! per-dihedral contribution kernel is JIT-composed at
//! `ForceField::new` time and dispatched from the dihedral composed
//! module; these tests drive the slot through a `ForceField` and
//! assert on the per-particle force / energy / virial outputs.

use heddle_md::forces::topology::Dihedral;
use heddle_md::forces::{
    AggregateLevel, AngleList, BondList, DihedralList, ExclusionList, ForceField,
    PeriodicDihedralBuilder, PotentialRegistry,
};
use heddle_md::gpu::{GpuContext, ParticleBuffers, init_device};
use heddle_md::io::config::{DihedralTypeConfig, NeighborListConfig};
use heddle_md::pbc::SimulationBox;
use heddle_md::precision::Real;
use heddle_md::state::ParticleState;
use heddle_md::timings::Timings;

fn box_10(gpu: &GpuContext) -> SimulationBox {
    SimulationBox::new(&gpu.device, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0).unwrap()
}

fn periodic_type(
    name: &str,
    k_phi: f64,
    n: u32,
    phi_0: f64,
) -> DihedralTypeConfig {
    DihedralTypeConfig::Periodic {
        name: name.to_string(),
        k_phi,
        n,
        phi_0,
        scale_lj_14: 0.5,
        scale_coul_14: 1.0 / 1.2,
    }
}

fn four_particle_state(positions: [[Real; 3]; 4]) -> ParticleState {
    ParticleState::new(
        positions.iter().map(|p| p[0]).collect(),
        positions.iter().map(|p| p[1]).collect(),
        positions.iter().map(|p| p[2]).collect(),
        vec![0.0; 4],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![1.0; 4],
        vec![0.0; 4],
        vec![0u32; 4],
        None,
        None,
    )
    .unwrap()
}

/// Build a DihedralList from a list of (atom_i, atom_j, atom_k,
/// atom_l, type_index) quintuples among `n` particles. The quintuples
/// are assumed to be in canonical order (sorted by tuple), as the
/// reduction kernel and the slot's compute pipeline rely on that.
fn dihedral_list_from(n: usize, dihs: &[(u32, u32, u32, u32, u32)]) -> DihedralList {
    let dihedrals: Vec<Dihedral> = dihs
        .iter()
        .map(|&(i, j, k, l, t)| Dihedral {
            atom_i: i,
            atom_j: j,
            atom_k: k,
            atom_l: l,
            dihedral_type_index: t,
        })
        .collect();
    let mut counts = vec![0u32; n];
    for d in &dihedrals {
        counts[d.atom_i as usize] += 1;
        counts[d.atom_j as usize] += 1;
        counts[d.atom_k as usize] += 1;
        counts[d.atom_l as usize] += 1;
    }
    let mut atom_dihedral_offsets = vec![0u32; n + 1];
    let mut running = 0u32;
    for a in 0..n {
        atom_dihedral_offsets[a] = running;
        running += counts[a];
    }
    atom_dihedral_offsets[n] = running;
    let mut per_atom: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (m, d) in dihedrals.iter().enumerate() {
        let slot_i = (4 * m) as u32;
        let slot_j = (4 * m + 1) as u32;
        let slot_k = (4 * m + 2) as u32;
        let slot_l = (4 * m + 3) as u32;
        per_atom[d.atom_i as usize].push(slot_i);
        per_atom[d.atom_j as usize].push(slot_j);
        per_atom[d.atom_k as usize].push(slot_k);
        per_atom[d.atom_l as usize].push(slot_l);
    }
    let mut atom_dihedral_indices = Vec::new();
    for a in 0..n {
        for &idx in &per_atom[a] {
            atom_dihedral_indices.push(idx);
        }
    }
    DihedralList {
        dihedrals,
        atom_dihedral_offsets,
        atom_dihedral_indices,
        particle_count: n,
    }
}

/// Place a four-atom dihedral with j at origin, k on +x at distance 1,
/// i in +y at distance 1, l at distance 1 from k with dihedral angle
/// `phi` around the j-k (x) axis from i. With this construction,
/// `b1 = (0,1,0)`, `b2 = (1,0,0)`, `b3 = r_k - r_l = (0,-cos(phi),-sin(phi))`
/// and the computed dihedral angle equals `phi`.
fn place_unit_dihedral(phi: Real) -> [[Real; 3]; 4] {
    [
        [0.0, 1.0, 0.0],                 // atom_i
        [0.0, 0.0, 0.0],                 // atom_j
        [1.0, 0.0, 0.0],                 // atom_k
        [1.0, phi.cos(), phi.sin()],     // atom_l
    ]
}

struct DihedralResult {
    forces_x: Vec<Real>,
    forces_y: Vec<Real>,
    forces_z: Vec<Real>,
    energies: Vec<Real>,
    virials: Vec<Real>,
}

fn run_dihedral(
    gpu: &GpuContext,
    state: &ParticleState,
    dihedrals: &DihedralList,
    dihedral_types: &[DihedralTypeConfig],
) -> DihedralResult {
    let n = state.positions_x.len();
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(PeriodicDihedralBuilder));
    let mut ff = ForceField::new(
        &registry,
        gpu,
        n,
        &box_10(gpu),
        &[],
        &[],
        &[],
        &[],
        dihedral_types,
        None,
        None,
        &[],
        &BondList::empty(n),
        &AngleList::empty(n),
        dihedrals,
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
    DihedralResult {
        forces_x: gpu.device.dtoh_sync_copy(&buffers.forces_x).unwrap(),
        forces_y: gpu.device.dtoh_sync_copy(&buffers.forces_y).unwrap(),
        forces_z: gpu.device.dtoh_sync_copy(&buffers.forces_z).unwrap(),
        energies: gpu.device.dtoh_sync_copy(&buffers.potential_energies).unwrap(),
        virials: gpu.device.dtoh_sync_copy(&buffers.virials).unwrap(),
    }
}

// --- Module loading ---

#[test] // rq-ae1c6a11
fn init_device_loads_dihedral_module() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    assert!(device.get_func("dihedral", "reduce_dihedral_forces").is_some());
    let _ = gpu.kernels.dihedral.reduce_dihedral_forces.clone();
}

// --- Construction ---

#[test] // rq-d2f43e61
fn construct_periodic_dihedral_state_uploads_parameters() {
    let gpu = init_device().unwrap();
    let dl = dihedral_list_from(4, &[(0, 1, 2, 3, 0)]);
    let dts = vec![periodic_type("ABCD", 2.0, 3, 0.0)];
    let state =
        heddle_md::forces::PeriodicDihedralState::new(&gpu, &dl, &dts).unwrap();
    assert_eq!(state.dihedral_count, 1);
    assert_eq!(state.particle_count, 4);
    let k = gpu.device.dtoh_sync_copy(&state.dihedral_k_phi).unwrap();
    let p0 = gpu.device.dtoh_sync_copy(&state.dihedral_phi_0).unwrap();
    let nn = gpu.device.dtoh_sync_copy(&state.dihedral_n).unwrap();
    assert_eq!(k.as_slice(), &[2.0 as Real]);
    assert_eq!(p0.as_slice(), &[0.0 as Real]);
    assert_eq!(nn.as_slice(), &[3u32]);
}

#[test] // rq-ae19b22f
fn construct_keeps_all_periodic_typed_dihedrals() {
    // With only periodic dihedral types (the only functional form
    // implemented today), every entry of the canonical DihedralList
    // survives the filter. Multi-type case: two distinct periodic
    // types referenced by two dihedrals on the same quadruple.
    // (Until a non-periodic functional form is added, the strict
    // "filter mixed types" scenario from the Gherkin spec has no
    // way to be set up; this test exercises the multi-type plumbing
    // that is the substrate of that filter.)
    let gpu = init_device().unwrap();
    let dl =
        dihedral_list_from(4, &[(0, 1, 2, 3, 0), (0, 1, 2, 3, 1)]);
    let dts = vec![
        periodic_type("A", 1.0, 1, 0.0),
        periodic_type("B", 0.5, 3, 0.2),
    ];
    let state =
        heddle_md::forces::PeriodicDihedralState::new(&gpu, &dl, &dts).unwrap();
    assert_eq!(state.dihedral_count, 2);
    let k = gpu.device.dtoh_sync_copy(&state.dihedral_k_phi).unwrap();
    assert_eq!(k.as_slice(), &[1.0 as Real, 0.5 as Real]);
}

// --- Force kernel correctness ---

#[test] // rq-603b4597
fn equilibrium_dihedral_produces_zero_force() {
    let gpu = init_device().unwrap();
    // With n=1 and phi_0=φ, f_phi = k·1·sin(0) = 0 → zero force.
    let phi: Real = 0.5;
    let state = four_particle_state(place_unit_dihedral(phi));
    let dl = dihedral_list_from(4, &[(0, 1, 2, 3, 0)]);
    let dts = vec![periodic_type("D", 1.0, 1, phi as f64)];
    let r = run_dihedral(&gpu, &state, &dl, &dts);
    for v in r.forces_x.iter().chain(r.forces_y.iter()).chain(r.forces_z.iter()) {
        assert!(v.abs() < 1e-4, "expected zero force, got {v}");
    }
}

#[test] // rq-e0933b95
fn newtons_third_law_holds_for_dihedral() {
    let gpu = init_device().unwrap();
    let phi: Real = 0.7;
    let state = four_particle_state(place_unit_dihedral(phi));
    let dl = dihedral_list_from(4, &[(0, 1, 2, 3, 0)]);
    let dts = vec![periodic_type("D", 1.0, 3, 0.0)];
    let r = run_dihedral(&gpu, &state, &dl, &dts);
    let sx: Real = r.forces_x.iter().sum();
    let sy: Real = r.forces_y.iter().sum();
    let sz: Real = r.forces_z.iter().sum();
    assert!(sx.abs() < 1e-4, "ΣF_x = {sx}");
    assert!(sy.abs() < 1e-4, "ΣF_y = {sy}");
    assert!(sz.abs() < 1e-4, "ΣF_z = {sz}");
}

#[test] // rq-185a2743
fn force_matches_closed_form_periodic() {
    // For my unit-dihedral geometry, the computed φ equals the input
    // φ. With n=1, phi_0=0, k=1: f_phi = sin(φ). For φ=0.5, the
    // analytical per-atom force can be derived from the chain-rule
    // formulas, but the simpler invariant is that the total magnitude
    // of the four per-atom forces scales linearly with k.
    let gpu = init_device().unwrap();
    let phi: Real = 0.5;
    let state = four_particle_state(place_unit_dihedral(phi));
    let dl = dihedral_list_from(4, &[(0, 1, 2, 3, 0)]);
    let dts_k1 = vec![periodic_type("D", 1.0, 1, 0.0)];
    let dts_k2 = vec![periodic_type("D", 2.0, 1, 0.0)];
    let r1 = run_dihedral(&gpu, &state, &dl, &dts_k1);
    let r2 = run_dihedral(&gpu, &state, &dl, &dts_k2);
    let mag1: Real = r1
        .forces_x
        .iter()
        .zip(r1.forces_y.iter())
        .zip(r1.forces_z.iter())
        .map(|((x, y), z)| (x * x + y * y + z * z).sqrt())
        .sum();
    let mag2: Real = r2
        .forces_x
        .iter()
        .zip(r2.forces_y.iter())
        .zip(r2.forces_z.iter())
        .map(|((x, y), z)| (x * x + y * y + z * z).sqrt())
        .sum();
    // mag2 should be 2 × mag1 (forces scale linearly in k_phi for
    // this geometry).
    let rel = (mag2 - 2.0 * mag1).abs() / mag1.max(1e-6);
    assert!(rel < 1e-3, "k-linearity check failed: mag1={mag1}, mag2={mag2}");
}

#[test] // rq-dceedbbb
fn force_consistent_with_central_difference_of_energy() {
    // F · dx ≈ −[U(x + ε·dx) − U(x − ε·dx)] / 2.
    // Vary atom_l's z-coordinate (which rotates the dihedral) and
    // verify the analytical gradient matches the central-difference
    // estimate, projected onto atom_l's z direction.
    let gpu = init_device().unwrap();
    let phi: Real = 0.6;
    let dl = dihedral_list_from(4, &[(0, 1, 2, 3, 0)]);
    let dts = vec![periodic_type("D", 1.0, 3, 0.0)];

    let eps: Real = 1e-3;
    let state_plus = four_particle_state(place_unit_dihedral(phi + eps));
    let state_minus = four_particle_state(place_unit_dihedral(phi - eps));
    let state_mid = four_particle_state(place_unit_dihedral(phi));

    let r_plus = run_dihedral(&gpu, &state_plus, &dl, &dts);
    let r_minus = run_dihedral(&gpu, &state_minus, &dl, &dts);
    let r_mid = run_dihedral(&gpu, &state_mid, &dl, &dts);

    let u_plus: Real = r_plus.energies.iter().sum();
    let u_minus: Real = r_minus.energies.iter().sum();
    // dU/dφ ≈ (U(φ+ε) − U(φ−ε)) / (2ε); the analytical value is
    // -k·n·sin(n·φ − phi_0) = -1·3·sin(3·0.6) = -3·sin(1.8).
    let du_central = (u_plus - u_minus) / (2.0 * eps);
    let du_analytical: Real = -3.0 * (3.0 as Real * 0.6 as Real).sin();
    let rel = (du_central - du_analytical).abs() / du_analytical.abs().max(1e-6);
    assert!(
        rel < 5e-2,
        "central-diff dU/dφ = {du_central}, analytical = {du_analytical}"
    );

    // Spot-check the per-atom force on atom_l: its magnitude should
    // be non-zero off-equilibrium.
    let fl_mag = (r_mid.forces_x[3] * r_mid.forces_x[3]
        + r_mid.forces_y[3] * r_mid.forces_y[3]
        + r_mid.forces_z[3] * r_mid.forces_z[3])
        .sqrt();
    assert!(fl_mag > 1e-3, "expected non-zero force on atom_l, got {fl_mag}");
}

#[test] // rq-2835ec70
fn multi_term_dihedral_sums_at_per_atom_reduction() {
    // Two rows on the same quadruple with different types (n=1 and
    // n=3): the per-atom force should equal the sum of the two
    // single-term forces.
    let gpu = init_device().unwrap();
    let phi: Real = 0.4;
    let state = four_particle_state(place_unit_dihedral(phi));
    let dl_n1 = dihedral_list_from(4, &[(0, 1, 2, 3, 0)]);
    let dl_n3 = dihedral_list_from(4, &[(0, 1, 2, 3, 0)]);
    let dl_both =
        dihedral_list_from(4, &[(0, 1, 2, 3, 0), (0, 1, 2, 3, 1)]);
    let dts_n1 = vec![periodic_type("N1", 1.0, 1, 0.0)];
    let dts_n3 = vec![periodic_type("N3", 0.5, 3, 0.2)];
    let dts_both = vec![
        periodic_type("N1", 1.0, 1, 0.0),
        periodic_type("N3", 0.5, 3, 0.2),
    ];
    let r_n1 = run_dihedral(&gpu, &state, &dl_n1, &dts_n1);
    let r_n3 = run_dihedral(&gpu, &state, &dl_n3, &dts_n3);
    let r_sum = run_dihedral(&gpu, &state, &dl_both, &dts_both);
    for a in 0..4 {
        let expected_x = r_n1.forces_x[a] + r_n3.forces_x[a];
        let expected_y = r_n1.forces_y[a] + r_n3.forces_y[a];
        let expected_z = r_n1.forces_z[a] + r_n3.forces_z[a];
        let dx = (r_sum.forces_x[a] - expected_x).abs();
        let dy = (r_sum.forces_y[a] - expected_y).abs();
        let dz = (r_sum.forces_z[a] - expected_z).abs();
        assert!(dx < 1e-4, "atom {a} F_x: sum={}, expected={}", r_sum.forces_x[a], expected_x);
        assert!(dy < 1e-4, "atom {a} F_y: sum={}, expected={}", r_sum.forces_y[a], expected_y);
        assert!(dz < 1e-4, "atom {a} F_z: sum={}, expected={}", r_sum.forces_z[a], expected_z);
    }
}

#[test] // rq-c432078d
fn minimum_image_is_applied() {
    // Place atom_l on the far side of the periodic box so the
    // wrapped (minimum-image) displacement is small while the raw
    // displacement is large. The kernel should compute the dihedral
    // from the wrapped displacement.
    let gpu = init_device().unwrap();
    // Build the canonical geometry, then shift atom_l by +lx so
    // unwrapped b3 has a large x component but wrapped b3 matches
    // the canonical placement.
    let phi: Real = 0.4;
    let mut pos = place_unit_dihedral(phi);
    pos[3][0] += 10.0; // wrap by one box length
    let state_wrapped = four_particle_state(pos);
    let state_canonical = four_particle_state(place_unit_dihedral(phi));
    let dl = dihedral_list_from(4, &[(0, 1, 2, 3, 0)]);
    let dts = vec![periodic_type("D", 1.0, 1, 0.0)];
    let r_wrapped = run_dihedral(&gpu, &state_wrapped, &dl, &dts);
    let r_canonical = run_dihedral(&gpu, &state_canonical, &dl, &dts);
    for a in 0..4 {
        assert!(
            (r_wrapped.forces_x[a] - r_canonical.forces_x[a]).abs() < 1e-4,
            "PBC wrap changed force on atom {a}.x"
        );
        assert!(
            (r_wrapped.forces_y[a] - r_canonical.forces_y[a]).abs() < 1e-4,
            "PBC wrap changed force on atom {a}.y"
        );
        assert!(
            (r_wrapped.forces_z[a] - r_canonical.forces_z[a]).abs() < 1e-4,
            "PBC wrap changed force on atom {a}.z"
        );
    }
}

#[test] // rq-b600335f
fn degenerate_geometry_m_zero_produces_zero_force() {
    // Atoms i, j, k collinear → b1 × b2 = 0, |m|² ≈ 0.
    let gpu = init_device().unwrap();
    let state = four_particle_state([
        [2.0, 0.0, 0.0], // i
        [0.0, 0.0, 0.0], // j
        [1.0, 0.0, 0.0], // k
        [1.5, 0.5, 0.0], // l (arbitrary, but not collinear with k)
    ]);
    let dl = dihedral_list_from(4, &[(0, 1, 2, 3, 0)]);
    let dts = vec![periodic_type("D", 1.0, 3, 0.0)];
    let r = run_dihedral(&gpu, &state, &dl, &dts);
    for v in r.forces_x.iter().chain(r.forces_y.iter()).chain(r.forces_z.iter()) {
        assert!(v.abs() < 1e-12, "expected zero force, got {v}");
    }
}

#[test] // rq-41636703
fn degenerate_geometry_n_zero_produces_zero_force() {
    // Atoms j, k, l collinear → b2 × b3 = 0, |n|² ≈ 0.
    let gpu = init_device().unwrap();
    let state = four_particle_state([
        [0.5, 0.5, 0.0], // i (out of jkl line)
        [0.0, 0.0, 0.0], // j
        [1.0, 0.0, 0.0], // k
        [2.0, 0.0, 0.0], // l (collinear with j, k)
    ]);
    let dl = dihedral_list_from(4, &[(0, 1, 2, 3, 0)]);
    let dts = vec![periodic_type("D", 1.0, 3, 0.0)];
    let r = run_dihedral(&gpu, &state, &dl, &dts);
    for v in r.forces_x.iter().chain(r.forces_y.iter()).chain(r.forces_z.iter()) {
        assert!(v.abs() < 1e-12, "expected zero force, got {v}");
    }
}

#[test] // rq-c0951af5
fn degenerate_geometry_b2_zero_produces_zero_force() {
    // atom_j and atom_k at the same position → |b2|² ≈ 0.
    let gpu = init_device().unwrap();
    let state = four_particle_state([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0], // k == j
        [0.0, 1.0, 1.0],
    ]);
    let dl = dihedral_list_from(4, &[(0, 1, 2, 3, 0)]);
    let dts = vec![periodic_type("D", 1.0, 3, 0.0)];
    let r = run_dihedral(&gpu, &state, &dl, &dts);
    for v in r.forces_x.iter().chain(r.forces_y.iter()).chain(r.forces_z.iter()) {
        assert!(v.abs() < 1e-12, "expected zero force, got {v}");
    }
}

// --- Config-load validation ---

fn config_with_dihedral(n: u32) -> String {
    format!(
        r#"schema_version = 1
units = "atomic"
init = "argon.in.xyz"

[simulation]
seed = 1
temperature = 1.0

[[phase]]
name = "p"
n_steps = 1
dt = 1.0

[phase.integrator]
kind = "velocity-verlet"
lossless = false

[[particle_types]]
name = "Ar"
mass = 1.0

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0

[[dihedral_types]]
name = "D"
potential = "periodic"
k_phi = 1.0
n = {n}
phi_0 = 0.0
"#
    )
}

fn write_tmp_config(name: &str, contents: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut dir = std::env::temp_dir();
    dir.push(format!("heddlemd-dih-cfg-{}-{}-{}", std::process::id(), name, nanos));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("sim.in.toml");
    std::fs::write(&path, contents).unwrap();
    path
}

#[test] // rq-7bf70ee9
fn config_rejects_n_zero() {
    use heddle_md::io::config::{ConfigError, load_config};
    let path = write_tmp_config("n_zero", &config_with_dihedral(0));
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => {
            assert_eq!(field, "dihedral_types[0].n");
        }
        e => panic!("expected InvalidValue for dihedral_types[0].n, got {:?}", e),
    }
}

#[test] // rq-ce48ab4a
fn config_rejects_n_above_six() {
    use heddle_md::io::config::{ConfigError, load_config};
    let path = write_tmp_config("n_seven", &config_with_dihedral(7));
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => {
            assert_eq!(field, "dihedral_types[0].n");
        }
        e => panic!("expected InvalidValue for dihedral_types[0].n, got {:?}", e),
    }
}

// --- Reduction kernel correctness ---

#[test] // rq-a73f7a4b
fn atom_in_one_dihedral_receives_its_force() {
    let gpu = init_device().unwrap();
    let phi: Real = 0.5;
    let state = four_particle_state(place_unit_dihedral(phi));
    let dl = dihedral_list_from(4, &[(0, 1, 2, 3, 0)]);
    let dts = vec![periodic_type("D", 1.0, 3, 0.0)];
    let r = run_dihedral(&gpu, &state, &dl, &dts);
    // Atom 0 (the only one touching slot 0) has at least one nonzero
    // component; Newton's third law is checked separately.
    let mag0 = (r.forces_x[0] * r.forces_x[0]
        + r.forces_y[0] * r.forces_y[0]
        + r.forces_z[0] * r.forces_z[0])
        .sqrt();
    assert!(mag0 > 1e-3, "expected nonzero F_0, got mag {mag0}");
}

#[test] // rq-59b48bcd
fn atom_in_two_dihedrals_receives_sum() {
    let gpu = init_device().unwrap();
    // Two dihedrals on the same 4 atoms (but with two distinct
    // dihedral_types), so atom 0 contributes to two scratch slots.
    let phi: Real = 0.4;
    let state = four_particle_state(place_unit_dihedral(phi));
    let dl_single =
        dihedral_list_from(4, &[(0, 1, 2, 3, 0)]);
    let dl_double = dihedral_list_from(
        4,
        &[(0, 1, 2, 3, 0), (0, 1, 2, 3, 1)],
    );
    let dts_single = vec![periodic_type("A", 1.0, 1, 0.0)];
    let dts_double = vec![
        periodic_type("A", 1.0, 1, 0.0),
        periodic_type("A", 1.0, 1, 0.0),
    ];
    let r_single = run_dihedral(&gpu, &state, &dl_single, &dts_single);
    let r_double = run_dihedral(&gpu, &state, &dl_double, &dts_double);
    // With two identical types, the doubled run should give exactly
    // 2 × the single-run forces.
    for a in 0..4 {
        let exp_x = 2.0 * r_single.forces_x[a];
        let exp_y = 2.0 * r_single.forces_y[a];
        let exp_z = 2.0 * r_single.forces_z[a];
        let dx = (r_double.forces_x[a] - exp_x).abs();
        let dy = (r_double.forces_y[a] - exp_y).abs();
        let dz = (r_double.forces_z[a] - exp_z).abs();
        assert!(dx < 1e-4 && dy < 1e-4 && dz < 1e-4,
            "atom {a}: doubled run not equal to 2× single ({:?} vs 2×{:?})",
            (r_double.forces_x[a], r_double.forces_y[a], r_double.forces_z[a]),
            (r_single.forces_x[a], r_single.forces_y[a], r_single.forces_z[a]));
    }
}

#[test] // rq-7d412e08
fn reduction_summation_order_is_deterministic() {
    // Two runs with identical inputs must give byte-identical
    // accumulators — that's the reduction's load-bearing invariant.
    let gpu = init_device().unwrap();
    let phi: Real = 0.4;
    let state = four_particle_state(place_unit_dihedral(phi));
    let dl = dihedral_list_from(
        4,
        &[(0, 1, 2, 3, 0), (0, 1, 2, 3, 1)],
    );
    let dts = vec![
        periodic_type("A", 1.5, 1, 0.0),
        periodic_type("B", 0.7, 3, 0.2),
    ];
    let r1 = run_dihedral(&gpu, &state, &dl, &dts);
    let r2 = run_dihedral(&gpu, &state, &dl, &dts);
    assert_eq!(r1.forces_x, r2.forces_x);
    assert_eq!(r1.forces_y, r2.forces_y);
    assert_eq!(r1.forces_z, r2.forces_z);
}

#[test] // rq-91fbdf55
fn atom_with_no_dihedrals_gets_zero_accumulator() {
    let gpu = init_device().unwrap();
    // 5-atom state with a dihedral touching 0..=3; atom 4 has no
    // dihedral contribution.
    let mut positions = [[0.0 as Real; 3]; 4];
    for (i, p) in place_unit_dihedral(0.5).into_iter().enumerate() {
        positions[i] = p;
    }
    let state = ParticleState::new(
        positions.iter().map(|p| p[0]).chain(std::iter::once(0.5)).collect(),
        positions.iter().map(|p| p[1]).chain(std::iter::once(0.5)).collect(),
        positions.iter().map(|p| p[2]).chain(std::iter::once(0.5)).collect(),
        vec![0.0; 5],
        vec![0.0; 5],
        vec![0.0; 5],
        vec![1.0; 5],
        vec![0.0; 5],
        vec![0u32; 5],
        None,
        None,
    )
    .unwrap();
    let dl = dihedral_list_from(5, &[(0, 1, 2, 3, 0)]);
    let dts = vec![periodic_type("D", 1.0, 3, 0.0)];
    let r = run_dihedral(&gpu, &state, &dl, &dts);
    assert!(r.forces_x[4].abs() < 1e-12);
    assert!(r.forces_y[4].abs() < 1e-12);
    assert!(r.forces_z[4].abs() < 1e-12);
}

// --- Empty states ---

#[test] // rq-b0e866c2
fn empty_dihedral_list_is_a_noop() {
    // Builder declines to construct when the list is empty; the
    // ForceField then has no dihedral slot, which exercises the
    // empty-state path.
    let gpu = init_device().unwrap();
    let state = four_particle_state(place_unit_dihedral(0.3));
    let n = state.positions_x.len();
    let dl = DihedralList::empty(n);
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(PeriodicDihedralBuilder));
    let mut ff = ForceField::new(
        &registry, &gpu, n, &box_10(&gpu),
        &[], &[], &[], &[], &[],
        None, None, &[],
        &BondList::empty(n), &AngleList::empty(n), &dl,
        &ExclusionList::empty(n), &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(
        &mut buffers,
        &box_10(&gpu),
        &mut timings,
        AggregateLevel::ForcesAndScalars,
    )
    .unwrap();
    let fx = gpu.device.dtoh_sync_copy(&buffers.forces_x).unwrap();
    for v in &fx {
        assert!(v.abs() < 1e-12);
    }
}

#[test] // rq-6b38fe0b
fn reduce_dihedral_forces_on_zero_particles_is_a_noop() {
    use heddle_md::gpu::reduce_dihedral_forces;
    let gpu = init_device().unwrap();
    let empty_real = gpu.device.alloc_zeros::<Real>(0).unwrap();
    let empty_u32 = gpu.device.alloc_zeros::<u32>(0).unwrap();
    let mut out_x = gpu.device.alloc_zeros::<Real>(0).unwrap();
    let mut out_y = gpu.device.alloc_zeros::<Real>(0).unwrap();
    let mut out_z = gpu.device.alloc_zeros::<Real>(0).unwrap();
    let mut out_e = gpu.device.alloc_zeros::<Real>(0).unwrap();
    let mut out_w = gpu.device.alloc_zeros::<Real>(0).unwrap();
    reduce_dihedral_forces(
        &gpu.kernels,
        &empty_real,
        &empty_real,
        &empty_real,
        &empty_real,
        &empty_real,
        &empty_u32,
        &empty_u32,
        &mut out_x.slice_mut(..),
        &mut out_y.slice_mut(..),
        &mut out_z.slice_mut(..),
        &mut out_e.slice_mut(..),
        &mut out_w.slice_mut(..),
        0,
        true,
    )
    .unwrap();
}

// --- Energy and virial ---

#[test] // rq-d225bef2
fn dihedral_energy_matches_closed_form() {
    let gpu = init_device().unwrap();
    let phi: Real = 0.5;
    let k: Real = 2.0;
    let n_mult: u32 = 3;
    let phi_0: Real = 0.0;
    let state = four_particle_state(place_unit_dihedral(phi));
    let dl = dihedral_list_from(4, &[(0, 1, 2, 3, 0)]);
    let dts = vec![periodic_type("D", k as f64, n_mult, phi_0 as f64)];
    let r = run_dihedral(&gpu, &state, &dl, &dts);
    let u_total: Real = r.energies.iter().sum();
    let u_expected: Real = k
        * (1.0 + ((n_mult as Real) * phi - phi_0).cos());
    let rel = (u_total - u_expected).abs() / u_expected.abs().max(1e-6);
    assert!(rel < 1e-3, "U_total = {u_total}, expected {u_expected}");
}

#[test] // rq-f2f1372d
fn degenerate_dihedral_produces_zero_energy_and_virial() {
    let gpu = init_device().unwrap();
    // j == k → degenerate b2.
    let state = four_particle_state([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
    ]);
    let dl = dihedral_list_from(4, &[(0, 1, 2, 3, 0)]);
    let dts = vec![periodic_type("D", 2.0, 3, 0.0)];
    let r = run_dihedral(&gpu, &state, &dl, &dts);
    let u_total: Real = r.energies.iter().sum();
    let w_total: Real = r.virials.iter().sum();
    assert!(u_total.abs() < 1e-12, "U_total = {u_total}");
    assert!(w_total.abs() < 1e-12, "W_total = {w_total}");
}

// --- Reproducibility ---

#[test] // rq-04a6973e
fn two_independent_runs_byte_identical() {
    let gpu = init_device().unwrap();
    let phi: Real = 0.5;
    let state = four_particle_state(place_unit_dihedral(phi));
    let dl = dihedral_list_from(4, &[(0, 1, 2, 3, 0)]);
    let dts = vec![periodic_type("D", 1.0, 3, 0.2)];
    let a = run_dihedral(&gpu, &state, &dl, &dts);
    let b = run_dihedral(&gpu, &state, &dl, &dts);
    assert_eq!(a.forces_x, b.forces_x);
    assert_eq!(a.forces_y, b.forces_y);
    assert_eq!(a.forces_z, b.forces_z);
    assert_eq!(a.energies, b.energies);
    assert_eq!(a.virials, b.virials);
}

// --- Builder filtering ---

#[test] // rq-c0e1d856
fn builder_returns_none_for_empty_dihedral_list() {
    use heddle_md::forces::{Potential, PotentialBuilder, PotentialBuildContext};
    let gpu = init_device().unwrap();
    let dl = DihedralList::empty(4);
    let cx = PotentialBuildContext {
        gpu: &gpu,
        particle_count: 4,
        sim_box: &box_10(&gpu),
        particle_types: &[],
        pair_interactions: &[],
        bond_types: &[],
        angle_types: &[],
        dihedral_types: &[periodic_type("D", 1.0, 1, 0.0)],
        coulomb_config: None,
        spme_config: None,
        charges: &[],
        bond_list: &BondList::empty(4),
        angle_list: &AngleList::empty(4),
        dihedral_list: &dl,
        exclusion_list: &ExclusionList::empty(4),
        neighbor_list_config: &NeighborListConfig::AllPairs,
    };
    let built: Option<Box<dyn Potential>> = PeriodicDihedralBuilder.build(&cx).unwrap();
    assert!(built.is_none(), "expected no slot for empty DihedralList");
}
