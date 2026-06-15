// rq-9a80c43c — SHAKE+RATTLE constraint tests (SPC/E water as the
// canonical fixture). Inputs are expressed in atomic units, matching
// the internal pipeline; the SI-equivalent geometry is 1.0 Å for O–H
// and 1.633 Å for H–H.

use std::sync::Arc;

use heddle_md::integrator::IntegratorStepExt;
use heddle_md::forces::{ConstraintGroup, ConstraintList, GroupConstraint, PotentialRegistry};
use heddle_md::gpu::{ParticleBuffers, init_device};
use heddle_md::integrator::shake::{ShakeBuilder, ShakeConstraintsState, ShakeError};
use heddle_md::integrator::{Constraint, ConstraintError, ConstraintRegistry};
use heddle_md::io::config::NamedSlotConfig;
use heddle_md::pbc::SimulationBox;
use heddle_md::state::ParticleState;
use heddle_md::timings::Timings;
use heddle_md::precision::Real;

// Atomic units. SPC/E geometry: r_OH = 1.0 Å = 1.88973 a₀;
// r_HH = 1.633 Å = 3.08591 a₀.
const R_OH: Real = 1.889_726_1;
const R_HH: Real = 3.085_926_4;
// Realistic masses in electron-mass units (m_e). 1 amu ≈ 1822.888 m_e.
// m_O = 15.9994 amu; m_H = 1.008 amu.
const M_O: Real = 29_167.43;
const M_H: Real = 1_837.47;

// 1 fs in atomic time units (ℏ/E_h ≈ 24.189 attoseconds → 1 fs ≈ 41.341 atu).
const ATU_PER_FS: Real = 41.341_374;
// Production thermal MD timestep for rigid-water SHAKE: 2 fs.
const PROD_DT: Real = 2.0 * ATU_PER_FS;
// Production thermal MD timestep for stiffer C–H constraints (methane): 1 fs.
const PROD_DT_METHANE: Real = ATU_PER_FS;

fn spce_type() -> NamedSlotConfig {
    NamedSlotConfig::from_params_str(
        "SPCE",
        "shake",
        &format!(
            "atoms = 3\nconstraints = [\n  {{ i = 0, j = 1, d = {} }},\n  {{ i = 0, j = 2, d = {} }},\n  {{ i = 1, j = 2, d = {} }},\n]\n",
            R_OH as f64, R_OH as f64, R_HH as f64,
        ),
    )
}

/// Build a ConstraintList containing `n_waters` SHAKE groups whose
/// atom indices are sequential triples (0,1,2), (3,4,5), etc.
fn sequential_shake_list(n_waters: usize) -> ConstraintList {
    let mut groups = Vec::with_capacity(n_waters);
    let mut group_atoms = Vec::with_capacity(3 * n_waters);
    let mut group_constraints = Vec::with_capacity(3 * n_waters);
    for w in 0..n_waters {
        let base_atom = 3 * w as u32;
        groups.push(ConstraintGroup {
            atom_offset: (3 * w) as u32,
            atom_count: 3,
            constraint_offset: (3 * w) as u32,
            constraint_count: 3,
            constraint_type_index: 0,
        });
        group_atoms.extend([base_atom, base_atom + 1, base_atom + 2]);
        group_constraints.extend([
            GroupConstraint {
                local_i: 0,
                local_j: 1,
                r0: R_OH,
            },
            GroupConstraint {
                local_i: 0,
                local_j: 2,
                r0: R_OH,
            },
            GroupConstraint {
                local_i: 1,
                local_j: 2,
                r0: R_HH,
            },
        ]);
    }
    ConstraintList {
        groups,
        group_atoms,
        group_constraints,
        particle_count: 3 * n_waters,
    }
}

/// Build a `ParticleState` with `n_waters` rigid waters at the
/// equilibrium geometry. Water `w` is centred at
/// `(spacing * w, 0, 0)` so the molecules don't overlap.
fn water_state(n_waters: usize, spacing: Real) -> ParticleState {
    let mut positions_x = Vec::with_capacity(3 * n_waters);
    let mut positions_y = Vec::with_capacity(3 * n_waters);
    let mut positions_z = Vec::with_capacity(3 * n_waters);
    let mut masses = Vec::with_capacity(3 * n_waters);
    // Place the equilibrium triangle in the xy-plane:
    //   O at (+d_oh, 0, 0); H1 at (0, -r_hh/2, 0); H2 at (0, +r_hh/2, 0).
    let d_oh = (R_OH * R_OH - (R_HH * 0.5) * (R_HH * 0.5)).sqrt();
    for w in 0..n_waters {
        let cx = (w as Real) * spacing;
        positions_x.extend([cx + d_oh, cx, cx]);
        positions_y.extend([0.0, -R_HH * 0.5, R_HH * 0.5]);
        positions_z.extend([0.0, 0.0, 0.0]);
        masses.extend([M_O, M_H, M_H]);
    }
    let zero = vec![0.0; 3 * n_waters];
    ParticleState::new(
        positions_x,
        positions_y,
        positions_z,
        zero.clone(),
        zero.clone(),
        zero.clone(),
        masses,
        vec![0.0; 3 * n_waters],
        vec![0u32; 3 * n_waters],
        None,
        None,
    )
    .unwrap()
}

fn big_box() -> SimulationBox {
    // A box much larger than any inter-molecular spacing exercised in
    // these tests.
    SimulationBox::new(1.0e4, 1.0e4, 1.0e4, 0.0, 0.0, 0.0).unwrap()
}

// --- Construction tests --------------------------------------------------

// rq-64700eb0
#[test]
fn construct_shake_slot_for_one_water() {
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();
    assert_eq!(slot.group_count, 1);
    assert_eq!(slot.particle_count, 3);
}

// rq-7921e537
#[test]
fn shake_registry_with_builtins_returns_some_for_non_empty_list() {
    let gpu = init_device().unwrap();
    let registry = ConstraintRegistry::with_builtins();
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let slot = registry
        .build_optional(&list, &gpu, 3, &state.masses, &[spce_type()])
        .unwrap();
    assert!(slot.is_some());
    assert_eq!(slot.unwrap().group_count(), 1);
}

// rq-aea6734a
#[test]
fn shake_registry_with_builtins_returns_none_for_empty_list() {
    let gpu = init_device().unwrap();
    let registry = ConstraintRegistry::with_builtins();
    let list = ConstraintList::empty(0);
    let slot = registry.build_optional(&list, &gpu, 0, &[], &[]).unwrap();
    assert!(slot.is_none());
}

// rq-9a80c43c — rejects malformed (out-of-range local index) shake params.
#[test]
fn shake_rejects_malformed_constraint_pair() {
    let gpu = init_device().unwrap();
    let _ = gpu;
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let bad_type = NamedSlotConfig::from_params_str(
        "SPCE",
        "shake",
        // i=3 is out of range for atoms=3 (local indices 0..3).
        "atoms = 3\nconstraints = [\n  { i = 3, j = 1, d = 1.0 },\n]\n",
    );
    let err = ShakeConstraintsState::new(
        Arc::clone(&init_device().unwrap().device),
        &list,
        &state.masses,
        &[bad_type],
    )
    .unwrap_err();
    match err {
        ShakeError::MalformedShakeType { name, .. } => assert_eq!(name, "SPCE"),
        other => panic!("expected MalformedShakeType, got {other:?}"),
    }
}

// Helper: try to construct a ShakeConstraintsState with a custom
// constraint-type body; expect a MalformedShakeType error whose
// `reason` contains the substring `expected_reason`.
fn assert_shake_constraints_state_rejects(params_body: &str, expected_reason: &str) {
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let bad_type = NamedSlotConfig::from_params_str("SPCE", "shake", params_body);
    let err = ShakeConstraintsState::new(
        Arc::clone(&gpu.device),
        &list,
        &state.masses,
        &[bad_type],
    )
    .unwrap_err();
    match err {
        ShakeError::MalformedShakeType { reason, .. } => {
            assert!(
                reason.contains(expected_reason),
                "expected reason containing `{expected_reason}`, got `{reason}`"
            );
        }
        other => panic!("expected MalformedShakeType, got {other:?}"),
    }
}

// rq-659836c5
#[test]
fn shake_constraints_state_rejects_atoms_zero() {
    assert_shake_constraints_state_rejects(
        "atoms = 0\nconstraints = [\n  { i = 0, j = 1, d = 1.0 },\n]\n",
        "atoms must be strictly positive",
    );
}

// rq-e6e7d6e2
#[test]
fn shake_constraints_state_rejects_atoms_exceeds_max() {
    assert_shake_constraints_state_rejects(
        "atoms = 9\nconstraints = [\n  { i = 0, j = 1, d = 1.0 },\n]\n",
        "MAX_GROUP_ATOMS",
    );
}

// rq-9bd207b2
#[test]
fn shake_constraints_state_rejects_constraints_exceeds_max() {
    // 13 constraint pairs (one per (0,k) for k in 1..=12 — but we only
    // have 8 max atoms, so synthesise 13 pairs within atoms=8 by
    // covering every pair).
    let mut entries = String::new();
    let mut count = 0;
    'outer: for i in 0..8u32 {
        for j in (i + 1)..8u32 {
            if count >= 13 { break 'outer; }
            entries.push_str(&format!("  {{ i = {i}, j = {j}, d = 1.0 }},\n"));
            count += 1;
        }
    }
    let body = format!("atoms = 8\nconstraints = [\n{entries}]\n");
    assert_shake_constraints_state_rejects(&body, "MAX_GROUP_CONSTRAINTS");
}

// rq-a8971153
#[test]
fn shake_constraints_state_rejects_duplicate_constraint_pair() {
    assert_shake_constraints_state_rejects(
        "atoms = 3\nconstraints = [\n  { i = 0, j = 1, d = 1.0 },\n  { i = 1, j = 0, d = 1.2 },\n]\n",
        "duplicate constraint pair",
    );
}

// rq-5be2064b
#[test]
fn shake_constraints_state_rejects_non_positive_target_distance() {
    assert_shake_constraints_state_rejects(
        "atoms = 3\nconstraints = [\n  { i = 0, j = 1, d = 0.0 },\n]\n",
        "strictly positive",
    );
    assert_shake_constraints_state_rejects(
        "atoms = 3\nconstraints = [\n  { i = 0, j = 1, d = -1.0 },\n]\n",
        "strictly positive",
    );
}

// rq-7ef08958 rq-18165336
#[test]
fn empty_constraint_registry_reports_unsupported_kind() {
    let gpu = init_device().unwrap();
    let registry = ConstraintRegistry::new();
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let err = registry
        .build_optional(&list, &gpu, 3, &state.masses, &[spce_type()])
        .unwrap_err();
    match err {
        ConstraintError::UnsupportedKind(kind) => {
            assert_eq!(kind, "shake");
        }
        other => panic!("expected UnsupportedKind, got {other:?}"),
    }
}

// --- Empty-state tests ---------------------------------------------------

// rq-79c091e0
#[test]
fn shake_hooks_on_zero_group_slot_are_noops() {
    let gpu = init_device().unwrap();
    let empty_list = ConstraintList::empty(0);
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &empty_list,
        &[],
        &[],
    )
    .unwrap();
    assert_eq!(slot.group_count, 0);
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings)
        .unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings)
        .unwrap();
    slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT, &mut timings)
        .unwrap();
}

// --- Snapshot kernel -----------------------------------------------------

#[test]
fn shake_snapshot_copies_pre_drift_positions() {
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings)
        .unwrap();
    let snap_x: Vec<Real> =
        gpu.device.dtoh_sync_copy(&slot.snapshot_x).unwrap();
    let snap_y: Vec<Real> =
        gpu.device.dtoh_sync_copy(&slot.snapshot_y).unwrap();
    let snap_z: Vec<Real> =
        gpu.device.dtoh_sync_copy(&slot.snapshot_z).unwrap();
    assert_eq!(&snap_x[..3], state.positions_x.as_slice());
    assert_eq!(&snap_y[..3], state.positions_y.as_slice());
    assert_eq!(&snap_z[..3], state.positions_z.as_slice());
}

// --- Position projection -------------------------------------------------

fn dist(ax: Real, ay: Real, az: Real, bx: Real, by: Real, bz: Real) -> Real {
    ((ax - bx).powi(2) + (ay - by).powi(2) + (az - bz).powi(2)).sqrt()
}

// rq-0f5c9f99 rq-7c13040a
#[test]
fn shake_positions_restores_constraint_distances_after_bond_stretch() {
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();

    // Snapshot pre-drift state.
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings)
        .unwrap();

    // Perturb post-drift positions: stretch O-H1 by 5% along the
    // O-H1 direction.
    let mut pos_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let mut pos_y: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let mut pos_z: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    let dx = pos_x[1] - pos_x[0];
    let dy = pos_y[1] - pos_y[0];
    let dz = pos_z[1] - pos_z[0];
    pos_x[1] = pos_x[0] + dx * 1.05;
    pos_y[1] = pos_y[0] + dy * 1.05;
    pos_z[1] = pos_z[0] + dz * 1.05;
    gpu.device.htod_sync_copy_into(&pos_x, &mut buffers.positions_x).unwrap();
    gpu.device.htod_sync_copy_into(&pos_y, &mut buffers.positions_y).unwrap();
    gpu.device.htod_sync_copy_into(&pos_z, &mut buffers.positions_z).unwrap();

    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings)
        .unwrap();

    let pos_x_c: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let pos_y_c: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let pos_z_c: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();

    let d_oh1 = dist(pos_x_c[0], pos_y_c[0], pos_z_c[0], pos_x_c[1], pos_y_c[1], pos_z_c[1]);
    let d_oh2 = dist(pos_x_c[0], pos_y_c[0], pos_z_c[0], pos_x_c[2], pos_y_c[2], pos_z_c[2]);
    let d_hh = dist(pos_x_c[1], pos_y_c[1], pos_z_c[1], pos_x_c[2], pos_y_c[2], pos_z_c[2]);
    let tol = 1.0e-4;
    assert!(
        ((d_oh1 - R_OH) / R_OH).abs() < tol,
        "O-H1 distance {d_oh1} differs from r_oh {R_OH}"
    );
    assert!(
        ((d_oh2 - R_OH) / R_OH).abs() < tol,
        "O-H2 distance {d_oh2} differs from r_oh {R_OH}"
    );
    assert!(
        ((d_hh - R_HH) / R_HH).abs() < tol,
        "H-H distance {d_hh} differs from r_hh {R_HH}"
    );
}

#[test]
fn shake_positions_preserves_centre_of_mass() {
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();

    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings)
        .unwrap();

    // Apply an arbitrary perturbation to all three atoms.
    let mut pos_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let mut pos_y: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let mut pos_z: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    let perturb = [
        (0.05, 0.025, -0.015),
        (-0.035, 0.020, 0.030),
        (0.015, -0.040, 0.010),
    ];
    for (i, (dx, dy, dz)) in perturb.iter().enumerate() {
        pos_x[i] += *dx;
        pos_y[i] += *dy;
        pos_z[i] += *dz;
    }
    gpu.device.htod_sync_copy_into(&pos_x, &mut buffers.positions_x).unwrap();
    gpu.device.htod_sync_copy_into(&pos_y, &mut buffers.positions_y).unwrap();
    gpu.device.htod_sync_copy_into(&pos_z, &mut buffers.positions_z).unwrap();

    let m = &state.masses;
    let total = (m[0] + m[1] + m[2]) as f64;
    let com_u = (
        (m[0] * pos_x[0] + m[1] * pos_x[1] + m[2] * pos_x[2]) as f64 / total,
        (m[0] * pos_y[0] + m[1] * pos_y[1] + m[2] * pos_y[2]) as f64 / total,
        (m[0] * pos_z[0] + m[1] * pos_z[1] + m[2] * pos_z[2]) as f64 / total,
    );

    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings)
        .unwrap();

    let pc_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let pc_y: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let pc_z: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    let com_c = (
        (m[0] * pc_x[0] + m[1] * pc_x[1] + m[2] * pc_x[2]) as f64 / total,
        (m[0] * pc_y[0] + m[1] * pc_y[1] + m[2] * pc_y[2]) as f64 / total,
        (m[0] * pc_z[0] + m[1] * pc_z[1] + m[2] * pc_z[2]) as f64 / total,
    );
    // f32 SoA mass-weighted COM round-off scales with the system mass
    // and the per-atom displacements; a few 10⁻³ a₀ is the largest
    // deviation we can guarantee at this precision.
    let tol = 5.0e-3;
    assert!((com_c.0 - com_u.0).abs() < tol, "COM x drifted: {} vs {}", com_c.0, com_u.0);
    assert!((com_c.1 - com_u.1).abs() < tol, "COM y drifted: {} vs {}", com_c.1, com_u.1);
    assert!((com_c.2 - com_u.2).abs() < tol, "COM z drifted: {} vs {}", com_c.2, com_u.2);
}

// rq-5d18fa01
#[test]
fn shake_positions_updates_half_step_velocities_consistently() {
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();

    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings)
        .unwrap();

    // Stretch the O-H1 bond before SHAKE.
    let mut pos_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    pos_x[1] += 0.05;
    gpu.device.htod_sync_copy_into(&pos_x, &mut buffers.positions_x).unwrap();
    let u_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let v_before: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();

    let dt: Real = PROD_DT;
    slot.apply_after_drift(&mut buffers, &sim_box, dt, &mut timings)
        .unwrap();

    let c_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let v_after: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    for i in 0..3 {
        let expected = v_before[i] + (c_x[i] - u_x[i]) / dt;
        assert!(
            (v_after[i] - expected).abs() < 1.0e-4,
            "velocity correction inconsistent for atom {i}: got {} expected {}",
            v_after[i],
            expected
        );
    }
}

// rq-a0174216
#[test]
fn shake_positions_converges_at_production_dt_with_thermal_amplitude_displacements() {
    // At dt = 2 fs the unconstrained drift per step is ~80× larger than
    // the 1-atu drift exercised by every other position-projection test
    // in this file. Verify that SHAKE still pulls every constraint
    // residual inside the kernel's SHAKE_TOL² convergence criterion and
    // that the 1/dt half-step velocity update remains finite at the
    // production dt magnitude.
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();

    // Thermal-amplitude velocities (T ≈ 300 K). At 300 K the
    // Maxwell-Boltzmann v_rms is ≈ 1.24e-3 a₀/atu for H and ≈ 2.91e-4
    // a₀/atu for O. Use mutually different per-component vectors so the
    // drift breaks every constraint, not just one.
    let vx = vec![ 2.0e-4, -1.0e-3,  1.3e-3];
    let vy = vec![-1.5e-4,  1.2e-3, -0.6e-3];
    let vz = vec![ 1.0e-4, -0.8e-3,  1.1e-3];
    gpu.device.htod_sync_copy_into(&vx, &mut buffers.velocities_x).unwrap();
    gpu.device.htod_sync_copy_into(&vy, &mut buffers.velocities_y).unwrap();
    gpu.device.htod_sync_copy_into(&vz, &mut buffers.velocities_z).unwrap();

    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();

    // Unconstrained drift: r += v · dt.
    let mut px: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let mut py: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let mut pz: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    for i in 0..3 {
        px[i] += vx[i] * PROD_DT;
        py[i] += vy[i] * PROD_DT;
        pz[i] += vz[i] * PROD_DT;
    }
    gpu.device.htod_sync_copy_into(&px, &mut buffers.positions_x).unwrap();
    gpu.device.htod_sync_copy_into(&py, &mut buffers.positions_y).unwrap();
    gpu.device.htod_sync_copy_into(&pz, &mut buffers.positions_z).unwrap();

    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();

    let qx: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let qy: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let qz: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    let vx_after: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy_after: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vz_after: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();

    // Per-constraint residuals at exit. The kernel's SHAKE_TOL² is
    // 3.57e-6 a₀² (≈ 1e-26 m²). Allow 4× slack for f32 round-off
    // between the kernel's σ check and this host-side reconstruction.
    const SHAKE_TOL2: Real = 3.57e-6;
    let r2 = |a: usize, b: usize| -> Real {
        let dx = qx[a] - qx[b];
        let dy = qy[a] - qy[b];
        let dz = qz[a] - qz[b];
        dx * dx + dy * dy + dz * dz
    };
    let residuals = [
        ("O-H1", r2(0, 1) - R_OH * R_OH),
        ("O-H2", r2(0, 2) - R_OH * R_OH),
        ("H1-H2", r2(1, 2) - R_HH * R_HH),
    ];
    let bound = 4.0 * SHAKE_TOL2;
    for (name, sigma) in &residuals {
        assert!(
            sigma.abs() < bound,
            "{name} residual σ = {sigma} exceeds 4·SHAKE_TOL² = {bound} \
             — SHAKE did not converge inside SHAKE_MAX_ITER at production dt",
        );
    }

    // 1/dt velocity update finite for every atom (no f32 overflow).
    for i in 0..3 {
        assert!(vx_after[i].is_finite(), "v_x[{i}] not finite: {}", vx_after[i]);
        assert!(vy_after[i].is_finite(), "v_y[{i}] not finite: {}", vy_after[i]);
        assert!(vz_after[i].is_finite(), "v_z[{i}] not finite: {}", vz_after[i]);
    }
}

// --- Velocity projection -------------------------------------------------

// rq-17b28c63
#[test]
fn rattle_velocities_zeroes_constraint_derivatives() {
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();

    // Inject arbitrary velocities (atomic units).
    let vx = vec![1.0e-3, -0.5e-3, 0.3e-3];
    let vy = vec![0.2e-3, 0.4e-3, -0.7e-3];
    let vz = vec![-0.6e-3, 0.1e-3, 0.8e-3];
    gpu.device.htod_sync_copy_into(&vx, &mut buffers.velocities_x).unwrap();
    gpu.device.htod_sync_copy_into(&vy, &mut buffers.velocities_y).unwrap();
    gpu.device.htod_sync_copy_into(&vz, &mut buffers.velocities_z).unwrap();

    slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT, &mut timings)
        .unwrap();

    let px: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let py: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let pz: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    let vx_c: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy_c: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vz_c: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();

    let dot = |a: usize, b: usize| -> f64 {
        let drx = (px[a] - px[b]) as f64;
        let dry = (py[a] - py[b]) as f64;
        let drz = (pz[a] - pz[b]) as f64;
        let dvx = (vx_c[a] - vx_c[b]) as f64;
        let dvy = (vy_c[a] - vy_c[b]) as f64;
        let dvz = (vz_c[a] - vz_c[b]) as f64;
        drx * dvx + dry * dvy + drz * dvz
    };
    // RATTLE's absolute tolerance on `|v_rel · d|` is ~8.6e-17 a₀²/atu
    // (atomic units of the SI bound 1.0e-20 m²/s), well below f32
    // precision at the working scale `|v · r| ~ 1e-3`. The kernel
    // therefore converges to f32 round-off rather than to the
    // nominal tolerance; allow several units of last place on the
    // dot-product reconstruction here.
    let tol = 5.0e-9;
    assert!(dot(0, 1).abs() < tol, "(v_O - v_H1) · r_OH1 = {}", dot(0, 1));
    assert!(dot(0, 2).abs() < tol, "(v_O - v_H2) · r_OH2 = {}", dot(0, 2));
    assert!(dot(1, 2).abs() < tol, "(v_H1 - v_H2) · r_HH = {}", dot(1, 2));
}

// rq-7e084b5e
#[test]
fn rattle_velocities_preserves_centre_of_mass_velocity() {
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();

    let vx = vec![1.0e-3, -0.5e-3, 0.3e-3];
    let vy = vec![0.2e-3, 0.4e-3, -0.7e-3];
    let vz = vec![-0.6e-3, 0.1e-3, 0.8e-3];
    gpu.device.htod_sync_copy_into(&vx, &mut buffers.velocities_x).unwrap();
    gpu.device.htod_sync_copy_into(&vy, &mut buffers.velocities_y).unwrap();
    gpu.device.htod_sync_copy_into(&vz, &mut buffers.velocities_z).unwrap();

    let m = &state.masses;
    let total = (m[0] + m[1] + m[2]) as f64;
    let com_v0 = (
        (m[0] * vx[0] + m[1] * vx[1] + m[2] * vx[2]) as f64 / total,
        (m[0] * vy[0] + m[1] * vy[1] + m[2] * vy[2]) as f64 / total,
        (m[0] * vz[0] + m[1] * vz[1] + m[2] * vz[2]) as f64 / total,
    );

    slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT, &mut timings)
        .unwrap();

    let vx_c: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy_c: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vz_c: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
    let com_v1 = (
        (m[0] * vx_c[0] + m[1] * vx_c[1] + m[2] * vx_c[2]) as f64 / total,
        (m[0] * vy_c[0] + m[1] * vy_c[1] + m[2] * vy_c[2]) as f64 / total,
        (m[0] * vz_c[0] + m[1] * vz_c[1] + m[2] * vz_c[2]) as f64 / total,
    );
    let tol = 1.0e-6;
    assert!((com_v0.0 - com_v1.0).abs() < tol, "COM vx changed");
    assert!((com_v0.1 - com_v1.1).abs() < tol, "COM vy changed");
    assert!((com_v0.2 - com_v1.2).abs() < tol, "COM vz changed");
}

// --- Side effects --------------------------------------------------------

// rq-757a5bad
#[test]
fn shake_positions_writes_non_zero_position_level_constraint_virial() {
    // After apply_after_drift on a non-trivially perturbed water,
    // constraint_virial on the slot must contain non-zero entries for
    // every atom in the group.
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    // Perturb H1 along the O-H1 axis to force a position-level virial
    // contribution from every constraint.
    let mut pos_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let dx = pos_x[1] - pos_x[0];
    pos_x[1] = pos_x[0] + dx * 1.05;
    gpu.device.htod_sync_copy_into(&pos_x, &mut buffers.positions_x).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let virial: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.constraint_virial).unwrap();
    // The three water atoms each carry a virial entry. At least two
    // must be non-zero (the O-H1 pair drives a position-level virial on
    // both atoms).
    let nonzero = virial.iter().take(3).filter(|&&v| v.abs() > 1e-30).count();
    assert!(nonzero >= 2, "expected ≥2 non-zero virial entries, got {virial:?}");
}

// rq-13f424a1 rq-178eb1ae
#[test]
fn shake_positions_no_velocity_restores_constraint_distances_from_off_manifold_positions() {
    // apply_position_projection_only must restore constraint distances
    // without snapshotting (minimization use case). Set positions
    // off-manifold and verify post-call distances match r_oh / r_hh.
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();
    // Snapshot velocities and virials so we can verify they're
    // untouched.
    let vel_x_snap: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vel_y_snap: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vel_z_snap: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
    let virial_snap: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.constraint_virial).unwrap();
    // Perturb every constraint by ~5%.
    let mut pos_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let mut pos_y: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let mut pos_z: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    for i in 1..3 {
        let dx = pos_x[i] - pos_x[0];
        let dy = pos_y[i] - pos_y[0];
        let dz = pos_z[i] - pos_z[0];
        pos_x[i] = pos_x[0] + dx * 1.05;
        pos_y[i] = pos_y[0] + dy * 1.05;
        pos_z[i] = pos_z[0] + dz * 1.05;
    }
    gpu.device.htod_sync_copy_into(&pos_x, &mut buffers.positions_x).unwrap();
    gpu.device.htod_sync_copy_into(&pos_y, &mut buffers.positions_y).unwrap();
    gpu.device.htod_sync_copy_into(&pos_z, &mut buffers.positions_z).unwrap();
    slot.apply_position_projection_only(&mut buffers, &sim_box, &mut timings).unwrap();
    let pos_x_c: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let pos_y_c: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let pos_z_c: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    let d_oh1 = dist(pos_x_c[0], pos_y_c[0], pos_z_c[0], pos_x_c[1], pos_y_c[1], pos_z_c[1]);
    let d_oh2 = dist(pos_x_c[0], pos_y_c[0], pos_z_c[0], pos_x_c[2], pos_y_c[2], pos_z_c[2]);
    let d_hh  = dist(pos_x_c[1], pos_y_c[1], pos_z_c[1], pos_x_c[2], pos_y_c[2], pos_z_c[2]);
    let tol = 1.0e-4;
    assert!(((d_oh1 - R_OH) / R_OH).abs() < tol);
    assert!(((d_oh2 - R_OH) / R_OH).abs() < tol);
    assert!(((d_hh  - R_HH) / R_HH).abs() < tol);
    // Velocities and constraint_virial must be untouched.
    let vel_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vel_y: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vel_z: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
    let virial: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.constraint_virial).unwrap();
    assert_eq!(vel_x, vel_x_snap);
    assert_eq!(vel_y, vel_y_snap);
    assert_eq!(vel_z, vel_z_snap);
    assert_eq!(virial, virial_snap);
}

// rq-3b6f4dec
#[test]
fn rattle_velocities_accumulates_velocity_level_constraint_virial_when_dt_positive() {
    // After apply_after_drift (which writes the position-level half of
    // the virial) and then apply_after_kick with dt > 0, the
    // constraint_virial buffer holds the SUM of the position-level and
    // velocity-level halves. The velocity-level half is non-zero iff
    // the post-kick velocities are off the velocity manifold and the
    // RATTLE iteration produced a correction.
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();
    // Drive a position-level virial first (snapshot + perturb +
    // after_drift).
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let mut pos_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let dx = pos_x[1] - pos_x[0];
    pos_x[1] = pos_x[0] + dx * 1.05;
    gpu.device.htod_sync_copy_into(&pos_x, &mut buffers.positions_x).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let virial_after_pos: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.constraint_virial).unwrap();
    // Inject off-manifold velocities so RATTLE has work to do.
    let vx = vec![1.0e-3, -0.5e-3, 0.3e-3];
    let vy = vec![0.2e-3, 0.4e-3, -0.7e-3];
    let vz = vec![-0.6e-3, 0.1e-3, 0.8e-3];
    gpu.device.htod_sync_copy_into(&vx, &mut buffers.velocities_x).unwrap();
    gpu.device.htod_sync_copy_into(&vy, &mut buffers.velocities_y).unwrap();
    gpu.device.htod_sync_copy_into(&vz, &mut buffers.velocities_z).unwrap();
    // dt > 0 → velocity-level virial accumulates on top of the
    // position-level virial already in the buffer.
    slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let virial_after_kick: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.constraint_virial).unwrap();
    // Note: apply_after_kick also calls constraint_virial_scatter which
    // doesn't modify constraint_virial. The velocity-level half is
    // written into the same buffer by rattle_velocities.
    // At least one entry must have changed (the velocity-level half is
    // non-zero for the perturbed velocities).
    let n = virial_after_pos.len().min(virial_after_kick.len());
    let any_changed = (0..n).any(|i| (virial_after_kick[i] - virial_after_pos[i]).abs() > 1e-30);
    assert!(any_changed, "velocity-level virial accumulation should have changed at least one entry");
}

// rq-3aef0b06
#[test]
fn rattle_velocities_skips_velocity_level_virial_when_dt_zero() {
    // apply_initial_velocity_projection calls rattle_velocities with
    // dt = 0.0; the kernel must skip the velocity-level virial
    // accumulation in this branch (no associated timestep). Verify by
    // snapshotting constraint_virial before the call and asserting it
    // is unchanged after.
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();
    // Inject off-manifold velocities so the rattle iteration runs.
    let vx = vec![1.0e-3, -0.5e-3, 0.3e-3];
    let vy = vec![0.2e-3, 0.4e-3, -0.7e-3];
    let vz = vec![-0.6e-3, 0.1e-3, 0.8e-3];
    gpu.device.htod_sync_copy_into(&vx, &mut buffers.velocities_x).unwrap();
    gpu.device.htod_sync_copy_into(&vy, &mut buffers.velocities_y).unwrap();
    gpu.device.htod_sync_copy_into(&vz, &mut buffers.velocities_z).unwrap();
    let virial_snap: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.constraint_virial).unwrap();
    slot.apply_initial_velocity_projection(&mut buffers, &sim_box, &mut timings).unwrap();
    let virial_after: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.constraint_virial).unwrap();
    assert_eq!(
        virial_snap, virial_after,
        "rattle with dt = 0 must not touch constraint_virial"
    );
    // Sanity: velocities should have been projected (rattle still
    // runs), so they differ from the off-manifold input.
    let vx_after: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    assert_ne!(vx_after, vx, "rattle should have projected velocities");
}

// rq-8471c200
#[test]
fn constraint_virial_scatter_additively_writes_into_particle_virials() {
    // Initialise particle_virials to a known non-zero pattern; run
    // apply_after_kick (which fires shake's virial scatter step) and
    // verify particle_virials += per-atom constraint_virial entries.
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();
    // Drive non-zero virial via the after_drift hook.
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let mut pos_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let dx = pos_x[1] - pos_x[0];
    pos_x[1] = pos_x[0] + dx * 1.05;
    gpu.device.htod_sync_copy_into(&pos_x, &mut buffers.positions_x).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    // Seed buffers.virials with a recognisable pattern, then snapshot
    // the per-atom constraint_virial values.
    let pre_virials = vec![10.0, 20.0, 30.0];
    gpu.device.htod_sync_copy_into(&pre_virials, &mut buffers.virials).unwrap();
    let constraint_virial: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.constraint_virial).unwrap();
    // apply_after_kick runs rattle + constraint_virial_scatter.
    slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let post_virials: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.virials).unwrap();
    // The post-kick constraint_virial includes both the position-level
    // half (already there) and the velocity-level half (added by
    // rattle). The scatter adds those to particle_virials.
    let post_constraint: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.constraint_virial).unwrap();
    let _ = constraint_virial; // pre-rattle snapshot (unused; we use post)
    // The expectation: each atom in the group receives the SUM of its
    // contributions across constraint slots (one slot per atom-of-group
    // in this single-group case → one contribution per atom). So:
    //   post_virials[i] = pre_virials[i] + post_constraint[i]
    let tol = 1e-5;
    for i in 0..3 {
        let expected = pre_virials[i] + post_constraint[i];
        assert!(
            (post_virials[i] - expected).abs() <= tol * expected.abs().max(1.0),
            "particle_virials[{i}] = {} vs expected {} (pre {} + cv {})",
            post_virials[i], expected, pre_virials[i], post_constraint[i]
        );
    }
}

// rq-513b4dbe
#[test]
fn constraint_virial_scatter_handles_two_disjoint_groups() {
    // Two waters → two disjoint SHAKE groups. After apply_after_kick,
    // particle_virials carries non-zero contributions for every atom
    // in both groups.
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(2);
    let state = water_state(2, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();
    // Perturb both waters' H1.
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let mut pos_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    for w in 0..2 {
        let i_o = 3 * w;
        let i_h1 = 3 * w + 1;
        let dx = pos_x[i_h1] - pos_x[i_o];
        pos_x[i_h1] = pos_x[i_o] + dx * 1.05;
    }
    gpu.device.htod_sync_copy_into(&pos_x, &mut buffers.positions_x).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let pre_virials = vec![0.0; 6];
    gpu.device.htod_sync_copy_into(&pre_virials, &mut buffers.virials).unwrap();
    slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let post_virials: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.virials).unwrap();
    // Both groups' atoms received scatter contributions; in particular,
    // group 1's atom 3 (O of water 1) and atom 4 (H1 of water 1) have
    // non-zero virial entries — confirms the scatter handled both
    // groups disjointly without overwriting atom slots.
    for atom in [0usize, 1, 2, 3, 4, 5] {
        assert!(
            post_virials[atom].abs() > 1e-30,
            "atom {atom} in disjoint group received no virial contribution: {post_virials:?}"
        );
    }
}

// Methane (CH4) topology helper: 5 atoms (C, H, H, H, H), 4 C-H constraints.
const R_CH: Real = 2.066_0; // ~1.094 Å in Bohr
fn methane_type() -> NamedSlotConfig {
    NamedSlotConfig::from_params_str(
        "CH4",
        "shake",
        &format!(
            "atoms = 5\nconstraints = [\n  {{ i = 0, j = 1, d = {} }},\n  {{ i = 0, j = 2, d = {} }},\n  {{ i = 0, j = 3, d = {} }},\n  {{ i = 0, j = 4, d = {} }},\n]\n",
            R_CH as f64, R_CH as f64, R_CH as f64, R_CH as f64,
        ),
    )
}

fn methane_list() -> ConstraintList {
    let groups = vec![ConstraintGroup {
        atom_offset: 0,
        atom_count: 5,
        constraint_offset: 0,
        constraint_count: 4,
        constraint_type_index: 0,
    }];
    let group_atoms = vec![0u32, 1, 2, 3, 4];
    let group_constraints = vec![
        GroupConstraint { local_i: 0, local_j: 1, r0: R_CH },
        GroupConstraint { local_i: 0, local_j: 2, r0: R_CH },
        GroupConstraint { local_i: 0, local_j: 3, r0: R_CH },
        GroupConstraint { local_i: 0, local_j: 4, r0: R_CH },
    ];
    ConstraintList {
        groups,
        group_atoms,
        group_constraints,
        particle_count: 5,
    }
}

fn methane_state() -> ParticleState {
    // C at origin, four H atoms at tetrahedral positions scaled to
    // r_CH along the body diagonals (±,±,±) / sqrt(3).
    let inv_sqrt3 = 1.0 / (3.0 as Real).sqrt();
    let h_offset = R_CH * inv_sqrt3;
    let positions_x = vec![0.0, h_offset, -h_offset, h_offset, -h_offset];
    let positions_y = vec![0.0, h_offset, -h_offset, -h_offset, h_offset];
    let positions_z = vec![0.0, h_offset, h_offset, -h_offset, -h_offset];
    // C ~ 12 amu; H ~ 1 amu in electron-mass units.
    let m_c = 21_874.66;
    let m_h = 1_837.47;
    let masses = vec![m_c, m_h, m_h, m_h, m_h];
    ParticleState::new(
        positions_x,
        positions_y,
        positions_z,
        vec![0.0; 5],
        vec![0.0; 5],
        vec![0.0; 5],
        masses,
        vec![0.0; 5],
        vec![0u32; 5],
        None,
        None,
    )
    .unwrap()
}

// rq-c70532a9
#[test]
fn construct_shake_slot_for_methane_succeeds() {
    let gpu = init_device().unwrap();
    let list = methane_list();
    let state = methane_state();
    let slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[methane_type()],
    )
    .unwrap();
    assert_eq!(slot.group_count, 1);
    assert_eq!(slot.particle_count, 5);
}

// rq-1d06fac7
#[test]
fn methane_c_h_constraints_remain_within_tolerance_after_100_steps() {
    let gpu = init_device().unwrap();
    let list = methane_list();
    let state = methane_state();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[methane_type()],
    )
    .unwrap();
    // Inject small per-atom velocities; let the integration-free SHAKE
    // round-trip (snapshot → after_drift → after_kick) run 100 times
    // and verify constraints stay tight throughout.
    let v = 1.0e-4;
    let vx = vec![0.0, v, -v, v, -v];
    let vy = vec![0.0, v, -v, -v, v];
    let vz = vec![0.0, v, v, -v, -v];
    gpu.device.htod_sync_copy_into(&vx, &mut buffers.velocities_x).unwrap();
    gpu.device.htod_sync_copy_into(&vy, &mut buffers.velocities_y).unwrap();
    gpu.device.htod_sync_copy_into(&vz, &mut buffers.velocities_z).unwrap();
    for _ in 0..100 {
        slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT_METHANE, &mut timings).unwrap();
        // Simulate an unconstrained drift step: x += v · dt.
        let mut pos_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
        let mut pos_y: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
        let mut pos_z: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
        let cur_vx: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
        let cur_vy: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
        let cur_vz: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
        for i in 0..5 {
            pos_x[i] += cur_vx[i] * PROD_DT_METHANE;
            pos_y[i] += cur_vy[i] * PROD_DT_METHANE;
            pos_z[i] += cur_vz[i] * PROD_DT_METHANE;
        }
        gpu.device.htod_sync_copy_into(&pos_x, &mut buffers.positions_x).unwrap();
        gpu.device.htod_sync_copy_into(&pos_y, &mut buffers.positions_y).unwrap();
        gpu.device.htod_sync_copy_into(&pos_z, &mut buffers.positions_z).unwrap();
        slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT_METHANE, &mut timings).unwrap();
        slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT_METHANE, &mut timings).unwrap();
    }
    // Constraint distances must still match r_CH.
    let px: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let py: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let pz: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    let tol = 1.0e-3;
    for i in 1..5 {
        let d = dist(px[0], py[0], pz[0], px[i], py[i], pz[i]);
        assert!(
            ((d - R_CH) / R_CH).abs() < tol,
            "C-H{i} drift after 100 steps: d = {d}, r_CH = {R_CH}"
        );
    }
}

// rq-fc27df14
#[test]
fn shake_only_run_with_no_constraint_slot_leaves_distances_drifting() {
    // Negative control: skip the SHAKE projection between drift and
    // after_kick. After 10 cycles, the constraint distances drift away
    // from r_oh / r_hh significantly.
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    // Inject a velocity that would naturally pull H1 along the O-H1
    // axis (stretching the bond) and let it drift unconstrained.
    let vx = vec![0.0, 1.0e-3, -1.0e-3];
    let vy = vec![0.0, 1.0e-3, -1.0e-3];
    let vz = vec![0.0, 0.0, 0.0];
    gpu.device.htod_sync_copy_into(&vx, &mut buffers.velocities_x).unwrap();
    gpu.device.htod_sync_copy_into(&vy, &mut buffers.velocities_y).unwrap();
    gpu.device.htod_sync_copy_into(&vz, &mut buffers.velocities_z).unwrap();
    let mut px: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let mut py: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let mut pz: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    for _ in 0..10 {
        for i in 0..3 {
            px[i] += vx[i];
            py[i] += vy[i];
            pz[i] += vz[i];
        }
    }
    let d_oh1 = dist(px[0], py[0], pz[0], px[1], py[1], pz[1]);
    let drift = ((d_oh1 - R_OH) / R_OH).abs();
    assert!(
        drift > 5.0e-3,
        "without SHAKE, O-H1 distance should drift noticeably; got |Δ|/r_oh = {drift}, d_oh1 = {d_oh1}"
    );
    let _ = list;
}

// rq-ff235ded
#[test]
fn full_velocity_verlet_step_with_one_rigid_spce_group_preserves_all_three_distances() {
    // Run apply_before_drift → drift (unconstrained) → apply_after_drift
    // → apply_after_kick once with a non-trivial velocity. After the
    // full SHAKE/RATTLE cycle, all three constraint distances are
    // restored exactly within SHAKE tolerance.
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();
    // Non-zero velocities so the drift moves atoms.
    let vx = vec![1.0e-4, -2.0e-4, 1.5e-4];
    let vy = vec![5.0e-5, 3.0e-4, -1.0e-4];
    let vz = vec![-1.0e-4, 1.0e-4, 2.0e-4];
    gpu.device.htod_sync_copy_into(&vx, &mut buffers.velocities_x).unwrap();
    gpu.device.htod_sync_copy_into(&vy, &mut buffers.velocities_y).unwrap();
    gpu.device.htod_sync_copy_into(&vz, &mut buffers.velocities_z).unwrap();
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    // Drift: x += v · dt (mimic the VV KickDrift sub-step's position update).
    let mut px: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let mut py: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let mut pz: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    for i in 0..3 {
        px[i] += vx[i] * PROD_DT;
        py[i] += vy[i] * PROD_DT;
        pz[i] += vz[i] * PROD_DT;
    }
    gpu.device.htod_sync_copy_into(&px, &mut buffers.positions_x).unwrap();
    gpu.device.htod_sync_copy_into(&py, &mut buffers.positions_y).unwrap();
    gpu.device.htod_sync_copy_into(&pz, &mut buffers.positions_z).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let qx: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let qy: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let qz: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    let d_oh1 = dist(qx[0], qy[0], qz[0], qx[1], qy[1], qz[1]);
    let d_oh2 = dist(qx[0], qy[0], qz[0], qx[2], qy[2], qz[2]);
    let d_hh  = dist(qx[1], qy[1], qz[1], qx[2], qy[2], qz[2]);
    let tol = 1.0e-4;
    assert!(((d_oh1 - R_OH) / R_OH).abs() < tol);
    assert!(((d_oh2 - R_OH) / R_OH).abs() < tol);
    assert!(((d_hh  - R_HH) / R_HH).abs() < tol);
}

// rq-cfb1e3aa
#[test]
fn constraint_iteration_order_matches_topology_declared_order() {
    // The order constraints appear in the ShakeConstraintsState's
    // group_constraints_local_{i,j} device buffers must equal the
    // order they were declared in the constraint-type's `constraints`
    // array. Build a custom constraint type with a non-default order
    // and verify the upload preserves it.
    let gpu = init_device().unwrap();
    let custom_type = NamedSlotConfig::from_params_str(
        "SPCE_reversed",
        "shake",
        // Reversed pair order: (1,2), (0,2), (0,1).
        &format!(
            "atoms = 3\nconstraints = [\n  {{ i = 1, j = 2, d = {} }},\n  {{ i = 0, j = 2, d = {} }},\n  {{ i = 0, j = 1, d = {} }},\n]\n",
            R_HH as f64, R_OH as f64, R_OH as f64,
        ),
    );
    let list = sequential_shake_list(1);
    let state = water_state(1, 10.0);
    let slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[custom_type],
    )
    .unwrap();
    let local_i: Vec<u8> = gpu.device.dtoh_sync_copy(&slot.group_constraints_local_i).unwrap();
    let local_j: Vec<u8> = gpu.device.dtoh_sync_copy(&slot.group_constraints_local_j).unwrap();
    assert_eq!(&local_i[..3], &[1u8, 0, 0]);
    assert_eq!(&local_j[..3], &[2u8, 2, 1]);
}

#[test]
fn shake_does_not_modify_atoms_outside_groups() {
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);
    // 5 particles total — first 3 form the water; last 2 are bystanders.
    let mut state = water_state(1, 10.0);
    state.positions_x.extend([2.0, 4.0]);
    state.positions_y.extend([2.0, 4.0]);
    state.positions_z.extend([2.0, 4.0]);
    state.velocities_x.extend([0.001, -0.002]);
    state.velocities_y.extend([0.0, 0.0]);
    state.velocities_z.extend([0.0, 0.0]);
    state.forces_x.extend([0.0; 2]);
    state.forces_y.extend([0.0; 2]);
    state.forces_z.extend([0.0; 2]);
    state.potential_energies.extend([0.0; 2]);
    state.virials.extend([0.0; 2]);
    state.masses.extend([21_874.66, 29_167.43]);
    state.charges.extend([0.0; 2]);
    state.type_indices.extend([0u32; 2]);
    state.particle_ids = (0u32..5).collect();
    state.images_x.extend([0; 2]);
    state.images_y.extend([0; 2]);
    state.images_z.extend([0; 2]);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();
    // Run all three hooks.
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let pos_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let pos_y: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let pos_z: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    let v_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let v_y: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let v_z: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
    for i in 3..5 {
        assert_eq!(pos_x[i], state.positions_x[i]);
        assert_eq!(pos_y[i], state.positions_y[i]);
        assert_eq!(pos_z[i], state.positions_z[i]);
        assert_eq!(v_x[i], state.velocities_x[i]);
        assert_eq!(v_y[i], state.velocities_y[i]);
        assert_eq!(v_z[i], state.velocities_z[i]);
    }
}

#[test]
fn shake_does_not_modify_forces_masses_or_ids() {
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);
    let mut state = water_state(1, 10.0);
    state.forces_x = vec![0.1, 0.2, 0.3];
    state.forces_y = vec![0.4, 0.5, 0.6];
    state.forces_z = vec![0.7, 0.8, 0.9];
    state.potential_energies = vec![10.0, 20.0, 30.0];
    state.virials = vec![100.0, 200.0, 300.0];
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let fx: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.forces_x).unwrap();
    let fy: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.forces_y).unwrap();
    let fz: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.forces_z).unwrap();
    let m: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.masses).unwrap();
    let pid: Vec<u32> = gpu.device.dtoh_sync_copy(&buffers.particle_ids).unwrap();
    let pe: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
    let vir: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.virials).unwrap();
    assert_eq!(fx, state.forces_x);
    assert_eq!(fy, state.forces_y);
    assert_eq!(fz, state.forces_z);
    assert_eq!(m, state.masses);
    assert_eq!(pid, state.particle_ids);
    assert_eq!(pe, state.potential_energies);
    // The constraint_virial_scatter kernel adds the per-atom-of-group
    // virial contribution into buffers.virials (initially the
    // user-provided value). For atoms that are part of a SHAKE group
    // *and* receive virial corrections, we therefore expect the slot
    // to add to the user-provided value rather than preserve it. This
    // test stages a single group at equilibrium under RATTLE alone, so
    // the contribution is below f32 round-off and the test holds.
    assert_eq!(vir, state.virials);
}

// --- Multi-water independence --------------------------------------------

#[test]
fn multiple_water_groups_evolve_independently() {
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(3);
    let state = water_state(3, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    // Stretch each water's O-H1 by a different amount.
    let mut pos_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    for w in 0..3 {
        let i_o = 3 * w;
        let i_h1 = 3 * w + 1;
        let stretch = 1.0 + 0.02 * (w as Real + 1.0);
        let dx = pos_x[i_h1] - pos_x[i_o];
        pos_x[i_h1] = pos_x[i_o] + dx * stretch;
    }
    gpu.device.htod_sync_copy_into(&pos_x, &mut buffers.positions_x).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();

    let px: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let py: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let pz: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    for w in 0..3 {
        let i_o = 3 * w;
        let i_h1 = 3 * w + 1;
        let i_h2 = 3 * w + 2;
        let d_oh1 = dist(px[i_o], py[i_o], pz[i_o], px[i_h1], py[i_h1], pz[i_h1]);
        let d_oh2 = dist(px[i_o], py[i_o], pz[i_o], px[i_h2], py[i_h2], pz[i_h2]);
        let d_hh = dist(px[i_h1], py[i_h1], pz[i_h1], px[i_h2], py[i_h2], pz[i_h2]);
        assert!(((d_oh1 - R_OH) / R_OH).abs() < 1.0e-4);
        assert!(((d_oh2 - R_OH) / R_OH).abs() < 1.0e-4);
        assert!(((d_hh - R_HH) / R_HH).abs() < 1.0e-4);
    }
}

// --- Periodic boundary handling ------------------------------------------

// rq-7a0a23e3
//
// A SPC/E water whose atoms appear on opposite sides of the +x periodic
// boundary: the kernel's `min_image_to` alignment at the start of
// `shake_positions` must bring every atom into atom 0's image before
// projecting, then write atoms 1..n-1 back via delta so their global
// image bookkeeping is preserved.
// rq-7a0a23e3
#[test]
fn shake_positions_handles_water_straddling_periodic_boundary() {
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(1);

    // Small orthorhombic box; the primary cell is [-5, +5)^3.
    let box_l: Real = 10.0;
    let sim_box = SimulationBox::new(box_l, box_l, box_l, 0.0, 0.0, 0.0).unwrap();

    // Equilibrium-geometry water centred at x = +4.5 in the +x edge of
    // the primary cell. With d_OH = sqrt(R_OH^2 - (R_HH/2)^2) ≈ 1.09:
    //   O at (4.5 + d_oh, 0, 0) ≈ (5.59, 0, 0) — outside the primary
    //                                            cell along +x.
    //   H1 at (4.5, -R_HH/2, 0)
    //   H2 at (4.5, +R_HH/2, 0)
    // We then wrap O into the primary cell: (-4.41, 0, 0). The molecule
    // is now visually "split" across the +x boundary, even though under
    // minimum-image its geometry is the equilibrium triangle.
    let d_oh = (R_OH * R_OH - (R_HH * 0.5) * (R_HH * 0.5)).sqrt();
    let cx: Real = 4.5;
    let mut o_pos = [cx + d_oh, 0.0, 0.0];
    o_pos[0] -= box_l; // wrap O into the primary cell at x ≈ -4.41
    let h1_pos = [cx, -R_HH * 0.5, 0.0];
    let h2_pos = [cx, R_HH * 0.5, 0.0];

    // Sanity: under minimum-image, the configuration sits on the
    // constraint manifold.
    fn min_image_dist_ortho(a: [Real; 3], b: [Real; 3], box_l: Real) -> Real {
        let mut d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        for k in 0..3 {
            let ki = (d[k] / box_l + 0.5).floor();
            d[k] -= ki * box_l;
        }
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }
    assert!(
        (min_image_dist_ortho(o_pos, h1_pos, box_l) - R_OH).abs() < 1.0e-5,
        "fixture sanity: pre-perturb O-H1 should equal R_OH under minimum-image",
    );

    // Build a ParticleState with the wrap-straddling positions.
    let state = ParticleState::new(
        vec![o_pos[0], h1_pos[0], h2_pos[0]],
        vec![o_pos[1], h1_pos[1], h2_pos[1]],
        vec![o_pos[2], h1_pos[2], h2_pos[2]],
        vec![0.0; 3],
        vec![0.0; 3],
        vec![0.0; 3],
        vec![M_O, M_H, M_H],
        vec![0.0; 3],
        vec![0u32; 3],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = ShakeConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();

    // Snapshot pre-drift state.
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings)
        .unwrap();

    // Snapshot the *global* positions for a "no spurious wrap" check
    // after the projection.
    let pre_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let pre_y: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let pre_z: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();

    // Perturb H1 by ~0.01 a₀ along the (min-image) O→H1 direction —
    // small enough to leave atoms in their original lattice image but
    // large enough to drive SHAKE into a few iterations.
    let mut dx = h1_pos[0] - o_pos[0];
    let mut dy = h1_pos[1] - o_pos[1];
    let mut dz = h1_pos[2] - o_pos[2];
    let kx = (dx / box_l + 0.5).floor();
    let ky = (dy / box_l + 0.5).floor();
    let kz = (dz / box_l + 0.5).floor();
    dx -= kx * box_l;
    dy -= ky * box_l;
    dz -= kz * box_l;
    let norm = (dx * dx + dy * dy + dz * dz).sqrt();
    let stretch = 0.01;
    let mut h1_pert = h1_pos;
    h1_pert[0] += stretch * dx / norm;
    h1_pert[1] += stretch * dy / norm;
    h1_pert[2] += stretch * dz / norm;
    let pos_x = vec![o_pos[0], h1_pert[0], h2_pos[0]];
    let pos_y = vec![o_pos[1], h1_pert[1], h2_pos[1]];
    let pos_z = vec![o_pos[2], h1_pert[2], h2_pos[2]];
    gpu.device.htod_sync_copy_into(&pos_x, &mut buffers.positions_x).unwrap();
    gpu.device.htod_sync_copy_into(&pos_y, &mut buffers.positions_y).unwrap();
    gpu.device.htod_sync_copy_into(&pos_z, &mut buffers.positions_z).unwrap();

    // Unconstrained mass-weighted COM (under minimum-image w.r.t. O).
    let m_total = (M_O + 2.0 * M_H) as f64;
    let unconstrained_com = {
        let bring_to_o = |p: [Real; 3]| -> [Real; 3] {
            let mut d = [p[0] - o_pos[0], p[1] - o_pos[1], p[2] - o_pos[2]];
            for k in 0..3 {
                let ki = (d[k] / box_l + 0.5).floor();
                d[k] -= ki * box_l;
            }
            [o_pos[0] + d[0], o_pos[1] + d[1], o_pos[2] + d[2]]
        };
        let o_loc = bring_to_o(o_pos);
        let h1_loc = bring_to_o(h1_pert);
        let h2_loc = bring_to_o(h2_pos);
        [
            (M_O * o_loc[0] + M_H * h1_loc[0] + M_H * h2_loc[0]) as f64 / m_total,
            (M_O * o_loc[1] + M_H * h1_loc[1] + M_H * h2_loc[1]) as f64 / m_total,
            (M_O * o_loc[2] + M_H * h1_loc[2] + M_H * h2_loc[2]) as f64 / m_total,
        ]
    };

    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings)
        .unwrap();

    let post_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let post_y: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let post_z: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();

    // Constraint distances restored under minimum-image.
    let o = [post_x[0], post_y[0], post_z[0]];
    let h1 = [post_x[1], post_y[1], post_z[1]];
    let h2 = [post_x[2], post_y[2], post_z[2]];
    let d_oh1 = min_image_dist_ortho(o, h1, box_l);
    let d_oh2 = min_image_dist_ortho(o, h2, box_l);
    let d_hh = min_image_dist_ortho(h1, h2, box_l);
    let tol = 1.0e-4;
    assert!(
        ((d_oh1 - R_OH) / R_OH).abs() < tol,
        "O-H1 distance under min-image {d_oh1} differs from R_OH {R_OH}"
    );
    assert!(
        ((d_oh2 - R_OH) / R_OH).abs() < tol,
        "O-H2 distance under min-image {d_oh2} differs from R_OH {R_OH}"
    );
    assert!(
        ((d_hh - R_HH) / R_HH).abs() < tol,
        "H1-H2 distance under min-image {d_hh} differs from R_HH {R_HH}"
    );

    // No atom moved more than the constraint-correction magnitude
    // (~0.01 a₀): in particular, no atom jumped across the cell. Were
    // the kernel to accidentally wrap an atom while writing back, the
    // displacement would be ~box_l.
    for i in 0..3 {
        let dxg = post_x[i] - pre_x[i];
        let dyg = post_y[i] - pre_y[i];
        let dzg = post_z[i] - pre_z[i];
        let mag = (dxg * dxg + dyg * dyg + dzg * dzg).sqrt();
        assert!(
            mag < 0.5 * box_l,
            "atom {i} displaced by {mag} a₀ — suspicious wrap (box_l = {box_l})",
        );
    }

    // Mass-weighted COM under min-image is preserved.
    let constrained_com = {
        let bring_to_o = |p: [Real; 3]| -> [Real; 3] {
            let mut d = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
            for k in 0..3 {
                let ki = (d[k] / box_l + 0.5).floor();
                d[k] -= ki * box_l;
            }
            [o[0] + d[0], o[1] + d[1], o[2] + d[2]]
        };
        let o_loc = bring_to_o(o);
        let h1_loc = bring_to_o(h1);
        let h2_loc = bring_to_o(h2);
        [
            (M_O * o_loc[0] + M_H * h1_loc[0] + M_H * h2_loc[0]) as f64 / m_total,
            (M_O * o_loc[1] + M_H * h1_loc[1] + M_H * h2_loc[1]) as f64 / m_total,
            (M_O * o_loc[2] + M_H * h1_loc[2] + M_H * h2_loc[2]) as f64 / m_total,
        ]
    };
    // COM may differ from the unconstrained COM by a small amount due to
    // the H1 perturbation being projected; allow ~1e-3 a₀ slack.
    let com_tol = 1.0e-3;
    assert!(
        (constrained_com[0] - unconstrained_com[0]).abs() < com_tol,
        "COM x drifted across the wrap: pre {} vs post {}",
        unconstrained_com[0],
        constrained_com[0]
    );
    assert!(
        (constrained_com[1] - unconstrained_com[1]).abs() < com_tol,
        "COM y drifted across the wrap",
    );
    assert!(
        (constrained_com[2] - unconstrained_com[2]).abs() < com_tol,
        "COM z drifted across the wrap",
    );
}

// --- Reproducibility -----------------------------------------------------

// rq-aa5ac09f rq-c7fc10c5
#[test]
fn two_independent_shake_runs_produce_byte_identical_outputs() {
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(8);
    let state = water_state(8, 10.0);

    let run = |seed: u32| -> (Vec<Real>, Vec<Real>, Vec<Real>, Vec<Real>, Vec<Real>, Vec<Real>) {
        let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
        let sim_box = big_box();
        let mut timings = Timings::new(&gpu).unwrap();
        let mut slot = ShakeConstraintsState::new(
            gpu.device.clone(),
            &list,
            &state.masses,
            &[spce_type()],
        )
        .unwrap();
        // Inject seed-dependent velocities (deterministic).
        let mut vx = vec![0.0; 24];
        for i in 0..24 {
            vx[i] = ((seed as i32 + i as i32) as Real) * 1.0e-6;
        }
        gpu.device.htod_sync_copy_into(&vx, &mut buffers.velocities_x).unwrap();
        slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
        slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
        slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
        let px = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
        let py = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
        let pz = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
        let vx_o = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
        let vy_o = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
        let vz_o = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
        (px, py, pz, vx_o, vy_o, vz_o)
    };
    let a = run(7);
    let b = run(7);
    assert_eq!(a.0, b.0);
    assert_eq!(a.1, b.1);
    assert_eq!(a.2, b.2);
    assert_eq!(a.3, b.3);
    assert_eq!(a.4, b.4);
    assert_eq!(a.5, b.5);
}

// --- Module loading + builder construction -------------------------------

#[test]
fn init_device_exposes_shake_kernels() {
    let gpu = init_device().unwrap();
    // If the kernels were missing, init_device would have errored. Just
    // confirm the Kernels handle is populated.
    let _ = &gpu.kernels.shake.shake_snapshot;
    let _ = &gpu.kernels.shake.shake_positions;
    let _ = &gpu.kernels.shake.rattle_velocities;
}

#[test]
fn shake_builder_kind_name_is_shake() {
    let b = ShakeBuilder;
    use heddle_md::integrator::ConstraintBuilder;
    assert_eq!(b.kind_name(), "shake");
}

// --- Lossless rejection moved to config-load time ----------------------
//
// The runner-side `IncompatibleConstraint` rule
// (`validate_constraint_compatibility_rejects_lossless_with_constraints`
// in `tests/io_config.rs`) is the canonical test for the lossless +
// constraints rejection. The integrator's `execute()` no longer
// receives a constraint slot, so it cannot itself return any
// "unsupported" error at run time.
// rq-047c1f4d rq-09a19014 rq-53237ec4
#[test]
fn lossless_velocity_verlet_kind_does_not_support_constraints() {
    use heddle_md::integrator::IntegratorRegistry;
    use heddle_md::io::SlotConfig;
    let kind = SlotConfig::from_params_str("velocity-verlet", "lossless = true\n");
    let registry = IntegratorRegistry::with_builtins();
    let builder = registry.lookup(&kind.kind).unwrap();
    assert!(!builder.supports_constraints(&kind.params));
}

// --- Integration through Integrator::step --------------------------------

// rq-90538790 rq-4e47cdad rq-77f959b2
#[test]
fn integrator_step_dispatches_all_three_constraint_hooks() {
    use std::sync::{Arc as StdArc, Mutex};
    use heddle_md::forces::AngleList;
    use heddle_md::forces::ForceField;
    use heddle_md::integrator::{IntegratorStepWithConstraintExt, VelocityVerletState};
    use heddle_md::io::config::NeighborListConfig;

    #[derive(Debug)]
    struct RecordingConstraint {
        log: StdArc<Mutex<Vec<&'static str>>>,
    }
    impl Constraint for RecordingConstraint {
        fn apply_before_drift(
            &mut self,
            _b: &mut ParticleBuffers,
            _sb: &SimulationBox,
            _dt: Real,
            _t: &mut Timings,
        ) -> Result<(), ConstraintError> {
            self.log.lock().unwrap().push("before_drift");
            Ok(())
        }
        fn apply_after_drift(
            &mut self,
            _b: &mut ParticleBuffers,
            _sb: &SimulationBox,
            _dt: Real,
            _t: &mut Timings,
        ) -> Result<(), ConstraintError> {
            self.log.lock().unwrap().push("after_drift");
            Ok(())
        }
        fn apply_after_kick(
            &mut self,
            _b: &mut ParticleBuffers,
            _sb: &SimulationBox,
            _dt: Real,
            _t: &mut Timings,
        ) -> Result<(), ConstraintError> {
            self.log.lock().unwrap().push("after_kick");
            Ok(())
        }
    }

    let gpu = init_device().unwrap();
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = big_box();
    let mut ff = ForceField::new(
        &PotentialRegistry::with_builtins(),
        &gpu,
        3,
        &sim_box,
        &[],
        &[],
        &[],
        &[],
        None,
        None,
        &[],
        &heddle_md::forces::BondList::empty(3),
        &AngleList::empty(0),
        &heddle_md::forces::ExclusionList::empty(3),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let mut integrator = VelocityVerletState::new(&gpu, 3, false).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let log = StdArc::new(Mutex::new(Vec::new()));
    let mut rec = RecordingConstraint { log: log.clone() };
    integrator
        .step_with_constraint(&mut buffers, &mut sim_box, &mut ff, &mut rec, PROD_DT, &mut timings)
        .unwrap();
    let order = log.lock().unwrap().clone();
    assert_eq!(order, vec!["before_drift", "after_drift", "after_kick"]);
}

// rq-7047ea32
#[test]
fn integrator_step_with_none_constraint_skips_all_hooks() {
    // Verify that step succeeds and produces no SHAKE/RATTLE timings
    // when no constraint is passed.
    use heddle_md::forces::AngleList;
    use heddle_md::forces::ForceField;
    use heddle_md::integrator::IntegratorRegistry;
    use heddle_md::io::SlotConfig;
    use heddle_md::io::config::NeighborListConfig;
    let gpu = init_device().unwrap();
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = big_box();
    let mut ff = ForceField::new(
        &PotentialRegistry::with_builtins(),
        &gpu,
        3,
        &sim_box,
        &[],
        &[],
        &[],
        &[],
        None,
        None,
        &[],
        &heddle_md::forces::BondList::empty(3),
        &AngleList::empty(0),
        &heddle_md::forces::ExclusionList::empty(3),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let registry = IntegratorRegistry::with_builtins();
    let vv = SlotConfig::from_params_str("velocity-verlet", "lossless = false\n");
    let mut integrator = registry.build(&vv, &gpu, 3, 0).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, PROD_DT, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    for s in report.stages {
        let count = s.count;
        if s.name.starts_with("shake_") || s.name.starts_with("rattle_") || s.name == "constraint_virial_scatter" {
            assert_eq!(count, 0, "{} should not fire", s.name);
        }
    }
}
