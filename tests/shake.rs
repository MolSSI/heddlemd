// rq-9a80c43c — SHAKE+RATTLE constraint tests (SPC/E water as the
// canonical fixture). Inputs are expressed in atomic units, matching
// the internal pipeline; the SI-equivalent geometry is 1.0 Å for O–H
// and 1.633 Å for H–H.

use std::sync::Arc;

use dynamics::integrator::IntegratorStepExt;
use dynamics::forces::{ConstraintGroup, ConstraintList, GroupConstraint, PotentialRegistry};
use dynamics::gpu::{ParticleBuffers, init_device};
use dynamics::integrator::shake::{ShakeBuilder, ShakeConstraintsState, ShakeError};
use dynamics::integrator::{Constraint, ConstraintError, ConstraintRegistry};
use dynamics::io::config::NamedSlotConfig;
use dynamics::pbc::SimulationBox;
use dynamics::state::ParticleState;
use dynamics::timings::Timings;

// Atomic units. SPC/E geometry: r_OH = 1.0 Å = 1.88973 a₀;
// r_HH = 1.633 Å = 3.08591 a₀.
const R_OH: f32 = 1.889_726_1;
const R_HH: f32 = 3.085_926_4;
// Realistic masses in electron-mass units (m_e). 1 amu ≈ 1822.888 m_e.
// m_O = 15.9994 amu; m_H = 1.008 amu.
const M_O: f32 = 29_167.43;
const M_H: f32 = 1_837.47;

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
fn water_state(n_waters: usize, spacing: f32) -> ParticleState {
    let mut positions_x = Vec::with_capacity(3 * n_waters);
    let mut positions_y = Vec::with_capacity(3 * n_waters);
    let mut positions_z = Vec::with_capacity(3 * n_waters);
    let mut masses = Vec::with_capacity(3 * n_waters);
    // Place the equilibrium triangle in the xy-plane:
    //   O at (+d_oh, 0, 0); H1 at (0, -r_hh/2, 0); H2 at (0, +r_hh/2, 0).
    let d_oh = (R_OH * R_OH - (R_HH * 0.5) * (R_HH * 0.5)).sqrt();
    for w in 0..n_waters {
        let cx = (w as f32) * spacing;
        positions_x.extend([cx + d_oh, cx, cx]);
        positions_y.extend([0.0, -R_HH * 0.5, R_HH * 0.5]);
        positions_z.extend([0.0, 0.0, 0.0]);
        masses.extend([M_O, M_H, M_H]);
    }
    let zero = vec![0.0_f32; 3 * n_waters];
    ParticleState::new(
        positions_x,
        positions_y,
        positions_z,
        zero.clone(),
        zero.clone(),
        zero.clone(),
        masses,
        vec![0.0_f32; 3 * n_waters],
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

// rq-3abb71cd
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

// rq-3abb71cd rq-7921e537
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

// rq-fd0add61
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

// rq-7ef08958
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

// rq-5d972f15
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
    slot.apply_before_drift(&mut buffers, &sim_box, 1.0, &mut timings)
        .unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, 1.0, &mut timings)
        .unwrap();
    slot.apply_after_kick(&mut buffers, &sim_box, 1.0, &mut timings)
        .unwrap();
}

// --- Snapshot kernel -----------------------------------------------------

// rq-4ec4d1d6
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
    slot.apply_before_drift(&mut buffers, &sim_box, 1.0, &mut timings)
        .unwrap();
    let snap_x: Vec<f32> =
        gpu.device.dtoh_sync_copy(&slot.snapshot_x).unwrap();
    let snap_y: Vec<f32> =
        gpu.device.dtoh_sync_copy(&slot.snapshot_y).unwrap();
    let snap_z: Vec<f32> =
        gpu.device.dtoh_sync_copy(&slot.snapshot_z).unwrap();
    assert_eq!(&snap_x[..3], state.positions_x.as_slice());
    assert_eq!(&snap_y[..3], state.positions_y.as_slice());
    assert_eq!(&snap_z[..3], state.positions_z.as_slice());
}

// --- Position projection -------------------------------------------------

fn dist(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> f32 {
    ((ax - bx).powi(2) + (ay - by).powi(2) + (az - bz).powi(2)).sqrt()
}

// rq-a8b68f59 rq-0f5c9f99 rq-7c13040a
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
    slot.apply_before_drift(&mut buffers, &sim_box, 1.0, &mut timings)
        .unwrap();

    // Perturb post-drift positions: stretch O-H1 by 5% along the
    // O-H1 direction.
    let mut pos_x: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let mut pos_y: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let mut pos_z: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    let dx = pos_x[1] - pos_x[0];
    let dy = pos_y[1] - pos_y[0];
    let dz = pos_z[1] - pos_z[0];
    pos_x[1] = pos_x[0] + dx * 1.05;
    pos_y[1] = pos_y[0] + dy * 1.05;
    pos_z[1] = pos_z[0] + dz * 1.05;
    gpu.device.htod_sync_copy_into(&pos_x, &mut buffers.positions_x).unwrap();
    gpu.device.htod_sync_copy_into(&pos_y, &mut buffers.positions_y).unwrap();
    gpu.device.htod_sync_copy_into(&pos_z, &mut buffers.positions_z).unwrap();

    slot.apply_after_drift(&mut buffers, &sim_box, 1.0, &mut timings)
        .unwrap();

    let pos_x_c: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let pos_y_c: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let pos_z_c: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();

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

// rq-f26ae0cc
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

    slot.apply_before_drift(&mut buffers, &sim_box, 1.0, &mut timings)
        .unwrap();

    // Apply an arbitrary perturbation to all three atoms.
    let mut pos_x: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let mut pos_y: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let mut pos_z: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    let perturb = [
        (0.05_f32, 0.025, -0.015),
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

    slot.apply_after_drift(&mut buffers, &sim_box, 1.0, &mut timings)
        .unwrap();

    let pc_x: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let pc_y: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let pc_z: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
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

// rq-25acc667 rq-5d18fa01
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

    slot.apply_before_drift(&mut buffers, &sim_box, 1.0, &mut timings)
        .unwrap();

    // Stretch the O-H1 bond before SHAKE.
    let mut pos_x: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    pos_x[1] += 0.05;
    gpu.device.htod_sync_copy_into(&pos_x, &mut buffers.positions_x).unwrap();
    let u_x: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let v_before: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();

    let dt: f32 = 1.0;
    slot.apply_after_drift(&mut buffers, &sim_box, dt, &mut timings)
        .unwrap();

    let c_x: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let v_after: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
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

// --- Velocity projection -------------------------------------------------

// rq-66e657bf rq-17b28c63
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
    let vx = vec![1.0e-3_f32, -0.5e-3, 0.3e-3];
    let vy = vec![0.2e-3_f32, 0.4e-3, -0.7e-3];
    let vz = vec![-0.6e-3_f32, 0.1e-3, 0.8e-3];
    gpu.device.htod_sync_copy_into(&vx, &mut buffers.velocities_x).unwrap();
    gpu.device.htod_sync_copy_into(&vy, &mut buffers.velocities_y).unwrap();
    gpu.device.htod_sync_copy_into(&vz, &mut buffers.velocities_z).unwrap();

    slot.apply_after_kick(&mut buffers, &sim_box, 1.0, &mut timings)
        .unwrap();

    let px: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let py: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let pz: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    let vx_c: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy_c: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vz_c: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();

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

// rq-13af93b9 rq-7e084b5e
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

    let vx = vec![1.0e-3_f32, -0.5e-3, 0.3e-3];
    let vy = vec![0.2e-3_f32, 0.4e-3, -0.7e-3];
    let vz = vec![-0.6e-3_f32, 0.1e-3, 0.8e-3];
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

    slot.apply_after_kick(&mut buffers, &sim_box, 1.0, &mut timings)
        .unwrap();

    let vx_c: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy_c: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vz_c: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
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

// rq-fc6ec19e
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
    state.masses.extend([21_874.66_f32, 29_167.43]);
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
    slot.apply_before_drift(&mut buffers, &sim_box, 1.0, &mut timings).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, 1.0, &mut timings).unwrap();
    slot.apply_after_kick(&mut buffers, &sim_box, 1.0, &mut timings).unwrap();
    let pos_x: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let pos_y: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let pos_z: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    let v_x: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let v_y: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let v_z: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
    for i in 3..5 {
        assert_eq!(pos_x[i], state.positions_x[i]);
        assert_eq!(pos_y[i], state.positions_y[i]);
        assert_eq!(pos_z[i], state.positions_z[i]);
        assert_eq!(v_x[i], state.velocities_x[i]);
        assert_eq!(v_y[i], state.velocities_y[i]);
        assert_eq!(v_z[i], state.velocities_z[i]);
    }
}

// rq-fd498605
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
    slot.apply_before_drift(&mut buffers, &sim_box, 1.0, &mut timings).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, 1.0, &mut timings).unwrap();
    slot.apply_after_kick(&mut buffers, &sim_box, 1.0, &mut timings).unwrap();
    let fx: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.forces_x).unwrap();
    let fy: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.forces_y).unwrap();
    let fz: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.forces_z).unwrap();
    let m: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.masses).unwrap();
    let pid: Vec<u32> = gpu.device.dtoh_sync_copy(&buffers.particle_ids).unwrap();
    let pe: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
    let vir: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.virials).unwrap();
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

// rq-d5790d66
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
    slot.apply_before_drift(&mut buffers, &sim_box, 1.0, &mut timings).unwrap();
    // Stretch each water's O-H1 by a different amount.
    let mut pos_x: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    for w in 0..3 {
        let i_o = 3 * w;
        let i_h1 = 3 * w + 1;
        let stretch = 1.0 + 0.02 * (w as f32 + 1.0);
        let dx = pos_x[i_h1] - pos_x[i_o];
        pos_x[i_h1] = pos_x[i_o] + dx * stretch;
    }
    gpu.device.htod_sync_copy_into(&pos_x, &mut buffers.positions_x).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, 1.0, &mut timings).unwrap();

    let px: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let py: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let pz: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
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
    let box_l: f32 = 10.0;
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
    let cx: f32 = 4.5;
    let mut o_pos = [cx + d_oh, 0.0_f32, 0.0_f32];
    o_pos[0] -= box_l; // wrap O into the primary cell at x ≈ -4.41
    let h1_pos = [cx, -R_HH * 0.5, 0.0];
    let h2_pos = [cx, R_HH * 0.5, 0.0];

    // Sanity: under minimum-image, the configuration sits on the
    // constraint manifold.
    fn min_image_dist_ortho(a: [f32; 3], b: [f32; 3], box_l: f32) -> f32 {
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
        vec![0.0_f32; 3],
        vec![0.0_f32; 3],
        vec![0.0_f32; 3],
        vec![M_O, M_H, M_H],
        vec![0.0_f32; 3],
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
    slot.apply_before_drift(&mut buffers, &sim_box, 1.0, &mut timings)
        .unwrap();

    // Snapshot the *global* positions for a "no spurious wrap" check
    // after the projection.
    let pre_x: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let pre_y: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let pre_z: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();

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
    let stretch = 0.01_f32;
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
        let bring_to_o = |p: [f32; 3]| -> [f32; 3] {
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

    slot.apply_after_drift(&mut buffers, &sim_box, 1.0, &mut timings)
        .unwrap();

    let post_x: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let post_y: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let post_z: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();

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
        let bring_to_o = |p: [f32; 3]| -> [f32; 3] {
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

// rq-99ee814d rq-aa5ac09f rq-c7fc10c5
#[test]
fn two_independent_shake_runs_produce_byte_identical_outputs() {
    let gpu = init_device().unwrap();
    let list = sequential_shake_list(8);
    let state = water_state(8, 10.0);

    let run = |seed: u32| -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
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
        let mut vx = vec![0.0_f32; 24];
        for i in 0..24 {
            vx[i] = ((seed as i32 + i as i32) as f32) * 1.0e-6;
        }
        gpu.device.htod_sync_copy_into(&vx, &mut buffers.velocities_x).unwrap();
        slot.apply_before_drift(&mut buffers, &sim_box, 1.0, &mut timings).unwrap();
        slot.apply_after_drift(&mut buffers, &sim_box, 1.0, &mut timings).unwrap();
        slot.apply_after_kick(&mut buffers, &sim_box, 1.0, &mut timings).unwrap();
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

// rq-bdb4af60
#[test]
fn init_device_exposes_shake_kernels() {
    let gpu = init_device().unwrap();
    // If the kernels were missing, init_device would have errored. Just
    // confirm the Kernels handle is populated.
    let _ = &gpu.kernels.shake.shake_snapshot;
    let _ = &gpu.kernels.shake.shake_positions;
    let _ = &gpu.kernels.shake.rattle_velocities;
}

// rq-278cb574
#[test]
fn shake_builder_kind_name_is_shake() {
    let b = ShakeBuilder;
    use dynamics::integrator::ConstraintBuilder;
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
    use dynamics::integrator::IntegratorRegistry;
    use dynamics::io::SlotConfig;
    let kind = SlotConfig::from_params_str("velocity-verlet", "lossless = true\n");
    let registry = IntegratorRegistry::with_builtins();
    let builder = registry.lookup(&kind.kind).unwrap();
    assert!(!builder.supports_constraints(&kind.params));
}

// --- Integration through Integrator::step --------------------------------

// rq-90538790
#[test]
fn integrator_step_dispatches_all_three_constraint_hooks() {
    use std::sync::{Arc as StdArc, Mutex};
    use dynamics::forces::AngleList;
    use dynamics::forces::ForceField;
    use dynamics::integrator::{IntegratorStepWithConstraintExt, VelocityVerletState};
    use dynamics::io::config::NeighborListConfig;

    #[derive(Debug)]
    struct RecordingConstraint {
        log: StdArc<Mutex<Vec<&'static str>>>,
    }
    impl Constraint for RecordingConstraint {
        fn apply_before_drift(
            &mut self,
            _b: &mut ParticleBuffers,
            _sb: &SimulationBox,
            _dt: f32,
            _t: &mut Timings,
        ) -> Result<(), ConstraintError> {
            self.log.lock().unwrap().push("before_drift");
            Ok(())
        }
        fn apply_after_drift(
            &mut self,
            _b: &mut ParticleBuffers,
            _sb: &SimulationBox,
            _dt: f32,
            _t: &mut Timings,
        ) -> Result<(), ConstraintError> {
            self.log.lock().unwrap().push("after_drift");
            Ok(())
        }
        fn apply_after_kick(
            &mut self,
            _b: &mut ParticleBuffers,
            _sb: &SimulationBox,
            _dt: f32,
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
        &dynamics::forces::BondList::empty(3),
        &AngleList::empty(0),
        &dynamics::forces::ExclusionList::empty(3),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let mut integrator = VelocityVerletState::new(&gpu, 3, false).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let log = StdArc::new(Mutex::new(Vec::new()));
    let mut rec = RecordingConstraint { log: log.clone() };
    integrator
        .step_with_constraint(&mut buffers, &mut sim_box, &mut ff, &mut rec, 1.0, &mut timings)
        .unwrap();
    let order = log.lock().unwrap().clone();
    assert_eq!(order, vec!["before_drift", "after_drift", "after_kick"]);
}

#[test]
fn integrator_step_with_none_constraint_skips_all_hooks() {
    // Verify that step succeeds and produces no SHAKE/RATTLE timings
    // when no constraint is passed.
    use dynamics::forces::AngleList;
    use dynamics::forces::ForceField;
    use dynamics::integrator::IntegratorRegistry;
    use dynamics::io::SlotConfig;
    use dynamics::io::config::NeighborListConfig;
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
        &dynamics::forces::BondList::empty(3),
        &AngleList::empty(0),
        &dynamics::forces::ExclusionList::empty(3),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let registry = IntegratorRegistry::with_builtins();
    let vv = SlotConfig::from_params_str("velocity-verlet", "lossless = false\n");
    let mut integrator = registry.build(&vv, &gpu, 3, 0).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, 1.0, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    for s in report.stages {
        let count = s.count;
        if s.name.starts_with("shake_") || s.name.starts_with("rattle_") || s.name == "constraint_virial_scatter" {
            assert_eq!(count, 0, "{} should not fire", s.name);
        }
    }
}
