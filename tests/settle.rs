// rq-67e62f4b — SETTLE rigid-water constraint tests.

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use dynamics::integrator::IntegratorStepExt;
use dynamics::forces::{ConstraintGroup, ConstraintList, GroupConstraint, PotentialRegistry};
use dynamics::gpu::{GpuContext, ParticleBuffers, init_device};
use dynamics::integrator::settle::{SettleBuilder, SettleConstraintsState, SettleError};
use dynamics::integrator::{Constraint, ConstraintError, ConstraintRegistry};
use dynamics::io::config::NamedSlotConfig;
use dynamics::pbc::SimulationBox;
use dynamics::state::ParticleState;
use dynamics::timings::Timings;

const R_OH: f32 = 1.0e-10;
const R_HH: f32 = 1.633e-10;

fn spce_type() -> NamedSlotConfig {
    NamedSlotConfig::from_params_str(
        "SPCE",
        "settle-water",
        &format!("r_oh = {}\nr_hh = {}\n", R_OH as f64, R_HH as f64),
    )
}

/// Build a ConstraintList containing `n_waters` SETTLE groups whose
/// atom indices are sequential triples (0,1,2), (3,4,5), etc.
fn sequential_settle_list(n_waters: usize) -> ConstraintList {
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
                r0: 0.0,
            },
            GroupConstraint {
                local_i: 0,
                local_j: 2,
                r0: 0.0,
            },
            GroupConstraint {
                local_i: 1,
                local_j: 2,
                r0: 0.0,
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
    //   O at (+d_oh, 0, 0); H1 at (0, -r_hh/2, 0); H2 at (0, +r_hh/2, 0)
    let d_oh = (R_OH * R_OH - (R_HH * 0.5) * (R_HH * 0.5)).sqrt();
    for w in 0..n_waters {
        let cx = (w as f32) * spacing;
        positions_x.extend([cx + d_oh, cx, cx]);
        positions_y.extend([0.0, -R_HH * 0.5, R_HH * 0.5]);
        positions_z.extend([0.0, 0.0, 0.0]);
        // Realistic masses (kg) so the SETTLE mass check accepts them.
        masses.extend([15.999_4e-27_f32, 1.008e-27_f32, 1.008e-27_f32]);
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
    SimulationBox::new(1.0e-6, 1.0e-6, 1.0e-6, 0.0, 0.0, 0.0).unwrap()
}

// --- Construction tests --------------------------------------------------

#[test]
fn construct_settle_slot_for_one_water() {
    let gpu = init_device().unwrap();
    let list = sequential_settle_list(1);
    let state = water_state(1, 5.0e-10);
    let slot = SettleConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();
    assert_eq!(slot.group_count, 1);
    assert_eq!(slot.particle_count, 3);
}

#[test]
fn settle_registry_with_builtins_returns_some_for_non_empty_list() {
    let gpu = init_device().unwrap();
    let registry = ConstraintRegistry::with_builtins();
    let list = sequential_settle_list(1);
    let state = water_state(1, 5.0e-10);
    let slot = registry
        .build_optional(&list, &gpu, 3, &state.masses, &[spce_type()])
        .unwrap();
    assert!(slot.is_some());
    assert_eq!(slot.unwrap().group_count(), 1);
}

#[test]
fn settle_registry_with_builtins_returns_none_for_empty_list() {
    let gpu = init_device().unwrap();
    let registry = ConstraintRegistry::with_builtins();
    let list = ConstraintList::empty(0);
    let slot = registry.build_optional(&list, &gpu, 0, &[], &[]).unwrap();
    assert!(slot.is_none());
}

#[test]
fn settle_rejects_inconsistent_h_masses() {
    let gpu = init_device().unwrap();
    let _ = gpu;
    let list = sequential_settle_list(1);
    // Custom state where the two H masses differ.
    let mut state = water_state(1, 5.0e-10);
    state.masses[2] = state.masses[1] * 1.5;
    let err = SettleConstraintsState::new(
        Arc::clone(&init_device().unwrap().device),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap_err();
    match err {
        SettleError::InconsistentMasses { .. } => {}
        other => panic!("expected InconsistentMasses, got {other:?}"),
    }
}

#[test]
fn empty_constraint_registry_reports_unsupported_kind() {
    let gpu = init_device().unwrap();
    let registry = ConstraintRegistry::new();
    let list = sequential_settle_list(1);
    let state = water_state(1, 5.0e-10);
    let err = registry
        .build_optional(&list, &gpu, 3, &state.masses, &[spce_type()])
        .unwrap_err();
    match err {
        ConstraintError::UnsupportedKind(kind) => {
            assert_eq!(kind, "settle-water");
        }
        other => panic!("expected UnsupportedKind, got {other:?}"),
    }
}

// --- Empty-state tests ---------------------------------------------------

#[test]
fn settle_hooks_on_zero_group_slot_are_noops() {
    let gpu = init_device().unwrap();
    let empty_list = ConstraintList::empty(0);
    let mut slot = SettleConstraintsState::new(
        gpu.device.clone(),
        &empty_list,
        &[],
        &[],
    )
    .unwrap();
    assert_eq!(slot.group_count, 0);
    let state = water_state(1, 5.0e-10);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    slot.apply_before_drift(&mut buffers, &sim_box, 1.0e-15, &mut timings)
        .unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, 1.0e-15, &mut timings)
        .unwrap();
    slot.apply_after_kick(&mut buffers, &sim_box, 1.0e-15, &mut timings)
        .unwrap();
}

// --- Snapshot kernel -----------------------------------------------------

#[test]
fn settle_snapshot_copies_pre_drift_positions() {
    let gpu = init_device().unwrap();
    let list = sequential_settle_list(1);
    let state = water_state(1, 5.0e-10);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = SettleConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();
    slot.apply_before_drift(&mut buffers, &sim_box, 1.0e-15, &mut timings)
        .unwrap();
    let snap_x: Vec<f32> =
        gpu.device.dtoh_sync_copy(&slot.snapshot_x).unwrap();
    let snap_y: Vec<f32> =
        gpu.device.dtoh_sync_copy(&slot.snapshot_y).unwrap();
    let snap_z: Vec<f32> =
        gpu.device.dtoh_sync_copy(&slot.snapshot_z).unwrap();
    assert_eq!(snap_x, state.positions_x);
    assert_eq!(snap_y, state.positions_y);
    assert_eq!(snap_z, state.positions_z);
}

// --- Position projection -------------------------------------------------

fn dist(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> f32 {
    ((ax - bx).powi(2) + (ay - by).powi(2) + (az - bz).powi(2)).sqrt()
}

#[test]
fn settle_positions_restores_constraint_distances_after_bond_stretch() {
    let gpu = init_device().unwrap();
    let list = sequential_settle_list(1);
    let state = water_state(1, 5.0e-10);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = SettleConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();

    // Snapshot pre-drift state.
    slot.apply_before_drift(&mut buffers, &sim_box, 1.0e-15, &mut timings)
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

    slot.apply_after_drift(&mut buffers, &sim_box, 1.0e-15, &mut timings)
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

#[test]
fn settle_positions_preserves_centre_of_mass() {
    let gpu = init_device().unwrap();
    let list = sequential_settle_list(1);
    let state = water_state(1, 5.0e-10);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = SettleConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();

    slot.apply_before_drift(&mut buffers, &sim_box, 1.0e-15, &mut timings)
        .unwrap();

    // Apply an arbitrary perturbation to all three atoms.
    let mut pos_x: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let mut pos_y: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let mut pos_z: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    let perturb = [
        (1.0e-12, 0.5e-12, -0.3e-12),
        (-0.7e-12, 0.4e-12, 0.6e-12),
        (0.3e-12, -0.8e-12, 0.2e-12),
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

    slot.apply_after_drift(&mut buffers, &sim_box, 1.0e-15, &mut timings)
        .unwrap();

    let pc_x: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let pc_y: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let pc_z: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    let com_c = (
        (m[0] * pc_x[0] + m[1] * pc_x[1] + m[2] * pc_x[2]) as f64 / total,
        (m[0] * pc_y[0] + m[1] * pc_y[1] + m[2] * pc_y[2]) as f64 / total,
        (m[0] * pc_z[0] + m[1] * pc_z[1] + m[2] * pc_z[2]) as f64 / total,
    );
    let tol = 1.0e-13;
    assert!((com_c.0 - com_u.0).abs() < tol, "COM x drifted");
    assert!((com_c.1 - com_u.1).abs() < tol, "COM y drifted");
    assert!((com_c.2 - com_u.2).abs() < tol, "COM z drifted");
}

#[test]
fn settle_positions_updates_half_step_velocities_consistently() {
    let gpu = init_device().unwrap();
    let list = sequential_settle_list(1);
    let state = water_state(1, 5.0e-10);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = SettleConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();

    slot.apply_before_drift(&mut buffers, &sim_box, 1.0e-15, &mut timings)
        .unwrap();

    // Stretch the O-H1 bond before SETTLE.
    let mut pos_x: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    pos_x[1] += 5.0e-12;
    gpu.device.htod_sync_copy_into(&pos_x, &mut buffers.positions_x).unwrap();
    let u_x: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let v_before: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();

    let dt: f32 = 1.0e-15;
    slot.apply_after_drift(&mut buffers, &sim_box, dt, &mut timings)
        .unwrap();

    let c_x: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let v_after: Vec<f32> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    for i in 0..3 {
        let expected = v_before[i] + (c_x[i] - u_x[i]) / dt;
        assert!(
            (v_after[i] - expected).abs() < 1.0e-2,
            "velocity correction inconsistent for atom {i}: got {} expected {}",
            v_after[i],
            expected
        );
    }
}

// --- Velocity projection -------------------------------------------------

#[test]
fn settle_velocities_zeroes_constraint_derivatives() {
    let gpu = init_device().unwrap();
    let list = sequential_settle_list(1);
    let state = water_state(1, 5.0e-10);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = SettleConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();

    // Inject arbitrary velocities.
    let vx = vec![1.0_f32, -0.5, 0.3];
    let vy = vec![0.2_f32, 0.4, -0.7];
    let vz = vec![-0.6_f32, 0.1, 0.8];
    gpu.device.htod_sync_copy_into(&vx, &mut buffers.velocities_x).unwrap();
    gpu.device.htod_sync_copy_into(&vy, &mut buffers.velocities_y).unwrap();
    gpu.device.htod_sync_copy_into(&vz, &mut buffers.velocities_z).unwrap();

    slot.apply_after_kick(&mut buffers, &sim_box, 1.0e-15, &mut timings)
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
    let tol = 1.0e-12;
    assert!(dot(0, 1).abs() < tol, "(v_O - v_H1) · r_OH1 = {}", dot(0, 1));
    assert!(dot(0, 2).abs() < tol, "(v_O - v_H2) · r_OH2 = {}", dot(0, 2));
    assert!(dot(1, 2).abs() < tol, "(v_H1 - v_H2) · r_HH = {}", dot(1, 2));
}

#[test]
fn settle_velocities_preserves_centre_of_mass_velocity() {
    let gpu = init_device().unwrap();
    let list = sequential_settle_list(1);
    let state = water_state(1, 5.0e-10);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = SettleConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();

    let vx = vec![1.0_f32, -0.5, 0.3];
    let vy = vec![0.2_f32, 0.4, -0.7];
    let vz = vec![-0.6_f32, 0.1, 0.8];
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

    slot.apply_after_kick(&mut buffers, &sim_box, 1.0e-15, &mut timings)
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

#[test]
fn settle_does_not_modify_atoms_outside_groups() {
    let gpu = init_device().unwrap();
    let list = sequential_settle_list(1);
    // 5 particles total — first 3 form the water; last 2 are bystanders.
    let mut state = water_state(1, 5.0e-10);
    state.positions_x.extend([1.0e-10, 2.0e-10]);
    state.positions_y.extend([1.0e-10, 2.0e-10]);
    state.positions_z.extend([1.0e-10, 2.0e-10]);
    state.velocities_x.extend([0.1, -0.2]);
    state.velocities_y.extend([0.0, 0.0]);
    state.velocities_z.extend([0.0, 0.0]);
    state.forces_x.extend([0.0; 2]);
    state.forces_y.extend([0.0; 2]);
    state.forces_z.extend([0.0; 2]);
    state.potential_energies.extend([0.0; 2]);
    state.virials.extend([0.0; 2]);
    state.masses.extend([12.0e-27_f32, 16.0e-27]);
    state.charges.extend([0.0; 2]);
    state.type_indices.extend([0u32; 2]);
    state.particle_ids = (0u32..5).collect();
    state.images_x.extend([0; 2]);
    state.images_y.extend([0; 2]);
    state.images_z.extend([0; 2]);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = SettleConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();
    // Run all three hooks.
    slot.apply_before_drift(&mut buffers, &sim_box, 1.0e-15, &mut timings).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, 1.0e-15, &mut timings).unwrap();
    slot.apply_after_kick(&mut buffers, &sim_box, 1.0e-15, &mut timings).unwrap();
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

#[test]
fn settle_does_not_modify_forces_masses_or_ids() {
    let gpu = init_device().unwrap();
    let list = sequential_settle_list(1);
    let mut state = water_state(1, 5.0e-10);
    state.forces_x = vec![0.1, 0.2, 0.3];
    state.forces_y = vec![0.4, 0.5, 0.6];
    state.forces_z = vec![0.7, 0.8, 0.9];
    state.potential_energies = vec![10.0, 20.0, 30.0];
    state.virials = vec![100.0, 200.0, 300.0];
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = SettleConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();
    slot.apply_before_drift(&mut buffers, &sim_box, 1.0e-15, &mut timings).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, 1.0e-15, &mut timings).unwrap();
    slot.apply_after_kick(&mut buffers, &sim_box, 1.0e-15, &mut timings).unwrap();
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
    assert_eq!(vir, state.virials);
}

// --- Multi-water independence --------------------------------------------

#[test]
fn multiple_water_groups_evolve_independently() {
    let gpu = init_device().unwrap();
    let list = sequential_settle_list(3);
    let state = water_state(3, 5.0e-10);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot = SettleConstraintsState::new(
        gpu.device.clone(),
        &list,
        &state.masses,
        &[spce_type()],
    )
    .unwrap();
    slot.apply_before_drift(&mut buffers, &sim_box, 1.0e-15, &mut timings).unwrap();
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
    slot.apply_after_drift(&mut buffers, &sim_box, 1.0e-15, &mut timings).unwrap();

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

// --- Reproducibility -----------------------------------------------------

#[test]
fn two_independent_settle_runs_produce_byte_identical_outputs() {
    let gpu = init_device().unwrap();
    let list = sequential_settle_list(8);
    let state = water_state(8, 5.0e-10);

    let mut run = |seed: u32| -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
        let sim_box = big_box();
        let mut timings = Timings::new(&gpu).unwrap();
        let mut slot = SettleConstraintsState::new(
            gpu.device.clone(),
            &list,
            &state.masses,
            &[spce_type()],
        )
        .unwrap();
        // Inject seed-dependent velocities (deterministic).
        let mut vx = vec![0.0_f32; 24];
        for i in 0..24 {
            vx[i] = ((seed as i32 + i as i32) as f32) * 1.0e-4;
        }
        gpu.device.htod_sync_copy_into(&vx, &mut buffers.velocities_x).unwrap();
        slot.apply_before_drift(&mut buffers, &sim_box, 1.0e-15, &mut timings).unwrap();
        slot.apply_after_drift(&mut buffers, &sim_box, 1.0e-15, &mut timings).unwrap();
        slot.apply_after_kick(&mut buffers, &sim_box, 1.0e-15, &mut timings).unwrap();
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
fn init_device_exposes_settle_kernels() {
    let gpu = init_device().unwrap();
    // If the kernels were missing, init_device would have errored. Just
    // confirm the Kernels handle is populated.
    let _ = &gpu.kernels.settle.settle_snapshot;
    let _ = &gpu.kernels.settle.settle_positions;
    let _ = &gpu.kernels.settle.settle_velocities;
}

#[test]
fn settle_builder_kind_name_is_settle_water() {
    let b = SettleBuilder;
    use dynamics::integrator::ConstraintBuilder;
    assert_eq!(b.kind_name(), "settle-water");
}

// --- Lossless rejection moved to config-load time ----------------------
//
// The runner-side `IncompatibleConstraint` rule
// (`validate_constraint_compatibility_rejects_lossless_with_constraints`
// in `tests/io_config.rs`) is the canonical test for the lossless +
// constraints rejection. The integrator's `execute()` no longer
// receives a constraint slot, so it cannot itself return any
// "unsupported" error at run time.
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

#[test]
fn integrator_step_dispatches_all_three_constraint_hooks() {
    use std::sync::{Arc as StdArc, Mutex};
    use dynamics::forces::AngleList;
    use dynamics::forces::ForceField;
    use dynamics::integrator::{Integrator, IntegratorRegistry};
    use dynamics::io::SlotConfig;
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
    let state = water_state(1, 5.0e-10);
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
    let log = StdArc::new(Mutex::new(Vec::new()));
    let mut rec = RecordingConstraint { log: log.clone() };
    let arg: Option<&mut dyn Constraint> = Some(&mut rec);
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, arg, 1.0e-15, &mut timings)
        .unwrap();
    let order = log.lock().unwrap().clone();
    assert_eq!(order, vec!["before_drift", "after_drift", "after_kick"]);
}

#[test]
fn integrator_step_with_none_constraint_skips_all_hooks() {
    // Just verify that step succeeds and produces no SETTLE timings
    // when no constraint is passed. (Implicitly tested by every
    // existing integrator test that uses None.)
    use dynamics::forces::AngleList;
    use dynamics::forces::ForceField;
    use dynamics::integrator::{Integrator, IntegratorRegistry};
    use dynamics::io::SlotConfig;
    use dynamics::io::config::NeighborListConfig;
    let gpu = init_device().unwrap();
    let state = water_state(1, 5.0e-10);
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
        .step(&mut buffers, &mut sim_box, &mut ff, None, 1.0e-15, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    for s in report.stages {
        let count = s.count;
        if s.name.starts_with("settle_") {
            assert_eq!(count, 0, "settle stage {} should not fire", s.name);
        }
    }
}
