// rq-709c8eb5 — SETTLE constraint tests (SPC/E water as the canonical
// fixture). Inputs are in atomic units, matching the internal pipeline;
// the SI-equivalent geometry is 1.0 Å for O–H and 1.633 Å for H–H.

use std::path::PathBuf;
use std::sync::Arc;

use heddle_md::forces::{
    ConstraintGroup, ConstraintList, GroupConstraint, TopologyFileError, load_topology_file,
};
use heddle_md::gpu::{ParticleBuffers, init_device};
use heddle_md::integrator::settle::{SettleConstraintsState, SettleError};
use heddle_md::integrator::shake::ShakeConstraintsState;
use heddle_md::integrator::{Constraint, ConstraintRegistry};
use heddle_md::io::config::NamedSlotConfig;
use heddle_md::pbc::SimulationBox;
use heddle_md::state::ParticleState;
use heddle_md::timings::Timings;
use heddle_md::units::UnitSystem;
use heddle_md::precision::Real;

// Atomic units. SPC/E geometry: r_OH = 1.0 Å = 1.88973 a₀;
// r_HH = 1.633 Å = 3.08591 a₀.
const R_OH: Real = 1.889_726_1;
const R_HH: Real = 3.085_926_4;
const M_O: Real = 29_167.43;
const M_H: Real = 1_837.47;

const ATU_PER_FS: Real = 41.341_374;
const PROD_DT: Real = 2.0 * ATU_PER_FS;

fn spce_type() -> NamedSlotConfig {
    NamedSlotConfig::from_params_str(
        "SPCE",
        "settle",
        &format!("d_OH = {}\nd_HH = {}\n", R_OH as f64, R_HH as f64),
    )
}

/// Build a ConstraintList with `n_waters` SETTLE groups whose atom
/// indices are sequential triples (0,1,2), (3,4,5), … The per-group
/// constraints carry the canonical water pattern (the framework
/// populates these; SETTLE's kernels do not read them, but a realistic
/// list carries them).
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
            GroupConstraint { local_i: 0, local_j: 1, r0: R_OH },
            GroupConstraint { local_i: 0, local_j: 2, r0: R_OH },
            GroupConstraint { local_i: 1, local_j: 2, r0: R_HH },
        ]);
    }
    ConstraintList {
        groups,
        group_atoms,
        group_constraints,
        particle_count: 3 * n_waters,
    }
}

/// `n_waters` rigid waters at the equilibrium geometry, water `w`
/// centred at `(spacing * w, 0, 0)`.
fn water_state(n_waters: usize, spacing: Real) -> ParticleState {
    let mut positions_x = Vec::with_capacity(3 * n_waters);
    let mut positions_y = Vec::with_capacity(3 * n_waters);
    let mut positions_z = Vec::with_capacity(3 * n_waters);
    let mut masses = Vec::with_capacity(3 * n_waters);
    // O at (+d_oh, 0, 0); H1 at (0, -r_hh/2, 0); H2 at (0, +r_hh/2, 0).
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

fn big_box(gpu: &heddle_md::gpu::GpuContext) -> SimulationBox {
    SimulationBox::new(&gpu.device, 1.0e4, 1.0e4, 1.0e4, 0.0, 0.0, 0.0).unwrap()
}

fn dist(ax: Real, ay: Real, az: Real, bx: Real, by: Real, bz: Real) -> Real {
    ((ax - bx).powi(2) + (ay - by).powi(2) + (az - bz).powi(2)).sqrt()
}

fn assert_rigid(px: &[Real], py: &[Real], pz: &[Real], base: usize, ctx: &str) {
    let d_oh1 = dist(px[base], py[base], pz[base], px[base + 1], py[base + 1], pz[base + 1]);
    let d_oh2 = dist(px[base], py[base], pz[base], px[base + 2], py[base + 2], pz[base + 2]);
    let d_hh = dist(px[base + 1], py[base + 1], pz[base + 1], px[base + 2], py[base + 2], pz[base + 2]);
    let tol = 1.0e-4;
    assert!(((d_oh1 - R_OH) / R_OH).abs() < tol, "{ctx}: O-H1 = {d_oh1} vs {R_OH}");
    assert!(((d_oh2 - R_OH) / R_OH).abs() < tol, "{ctx}: O-H2 = {d_oh2} vs {R_OH}");
    assert!(((d_hh - R_HH) / R_HH).abs() < tol, "{ctx}: H-H = {d_hh} vs {R_HH}");
}

// --- Construction & parameter validation ---------------------------------

// rq-a8a99082
#[test]
fn construct_settle_slot_for_one_water() {
    let gpu = init_device().unwrap();
    let list = sequential_settle_list(1);
    let state = water_state(1, 10.0);
    let slot =
        SettleConstraintsState::new(gpu.device.clone(), &list, &state.masses, &[spce_type()])
            .unwrap();
    assert_eq!(slot.group_count, 1);
    assert_eq!(slot.particle_count, 3);
    let atoms: Vec<u32> = gpu.device.dtoh_sync_copy(&slot.group_atoms).unwrap();
    assert_eq!(&atoms[..3], &[0, 1, 2]);
}

// rq-9f211910
#[test]
fn settle_new_computes_canonical_geometry() {
    let gpu = init_device().unwrap();
    let list = sequential_settle_list(1);
    let state = water_state(1, 10.0);
    let slot =
        SettleConstraintsState::new(gpu.device.clone(), &list, &state.masses, &[spce_type()])
            .unwrap();
    let ra: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.group_ra).unwrap();
    let rb: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.group_rb).unwrap();
    let rc: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.group_rc).unwrap();

    let total = (M_O + 2.0 * M_H) as f64;
    let h = ((R_OH as f64).powi(2) - ((R_HH as f64) * 0.5).powi(2)).sqrt();
    let exp_ra = (2.0 * M_H as f64 / total) * h;
    let exp_rb = (M_O as f64 / total) * h;
    let exp_rc = (R_HH as f64) * 0.5;
    let tol = 1.0e-4;
    assert!(((ra[0] as f64 - exp_ra) / exp_ra).abs() < tol, "ra {} vs {}", ra[0], exp_ra);
    assert!(((rb[0] as f64 - exp_rb) / exp_rb).abs() < tol, "rb {} vs {}", rb[0], exp_rb);
    assert!(((rc[0] as f64 - exp_rc) / exp_rc).abs() < tol, "rc {} vs {}", rc[0], exp_rc);
    // Canonical COM is the origin: m_O·ra = 2·m_H·rb.
    assert!(
        ((M_O as f64 * ra[0] as f64 - 2.0 * M_H as f64 * rb[0] as f64) / (M_O as f64 * ra[0] as f64))
            .abs()
            < tol,
        "m_O·ra != 2·m_H·rb"
    );
}

// rq-6bef53e7
#[test]
fn settle_empty_list_is_noop() {
    let gpu = init_device().unwrap();
    let empty = ConstraintList::empty(0);
    let mut slot =
        SettleConstraintsState::new(gpu.device.clone(), &empty, &[], &[]).unwrap();
    assert_eq!(slot.group_count, 0);
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = big_box(&gpu);
    let mut timings = Timings::new(&gpu).unwrap();
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    slot.apply_position_projection_only(&mut buffers, &sim_box, &mut timings).unwrap();
}

fn assert_settle_rejects(params_body: &str, expected_reason: &str) {
    let gpu = init_device().unwrap();
    let list = sequential_settle_list(1);
    let state = water_state(1, 10.0);
    let bad = NamedSlotConfig::from_params_str("SPCE", "settle", params_body);
    let err = SettleConstraintsState::new(Arc::clone(&gpu.device), &list, &state.masses, &[bad])
        .unwrap_err();
    match err {
        SettleError::MalformedSettleType { reason, .. } => assert!(
            reason.contains(expected_reason),
            "expected `{expected_reason}`, got `{reason}`"
        ),
        other => panic!("expected MalformedSettleType, got {other:?}"),
    }
}

// rq-73c1173f
#[test]
fn settle_rejects_non_positive_d_oh() {
    assert_settle_rejects("d_OH = 0.0\nd_HH = 3.0\n", "d_OH must be strictly positive");
    assert_settle_rejects("d_OH = -1.0\nd_HH = 3.0\n", "d_OH must be strictly positive");
}

// rq-d2976a76
#[test]
fn settle_rejects_degenerate_geometry() {
    // d_HH >= 2·d_OH → collinear / imaginary apex height.
    assert_settle_rejects("d_OH = 1.0\nd_HH = 2.0\n", "d_HH must be less than 2 * d_OH");
}

// rq-b98dcdd2
#[test]
fn settle_rejects_wrong_atom_count() {
    let gpu = init_device().unwrap();
    // A group declaring 4 atoms.
    let groups = vec![ConstraintGroup {
        atom_offset: 0,
        atom_count: 4,
        constraint_offset: 0,
        constraint_count: 3,
        constraint_type_index: 0,
    }];
    let list = ConstraintList {
        groups,
        group_atoms: vec![0, 1, 2, 3],
        group_constraints: vec![
            GroupConstraint { local_i: 0, local_j: 1, r0: R_OH },
            GroupConstraint { local_i: 0, local_j: 2, r0: R_OH },
            GroupConstraint { local_i: 1, local_j: 2, r0: R_HH },
        ],
        particle_count: 4,
    };
    let masses = vec![M_O, M_H, M_H, M_H];
    let err = SettleConstraintsState::new(gpu.device.clone(), &list, &masses, &[spce_type()])
        .unwrap_err();
    match err {
        SettleError::InvalidGroupShape { group_index, reason } => {
            assert_eq!(group_index, 0);
            assert!(reason.contains("atom count"), "reason: {reason}");
        }
        other => panic!("expected InvalidGroupShape, got {other:?}"),
    }
}

// rq-56aeb344
#[test]
fn settle_rejects_unequal_hydrogen_masses() {
    let gpu = init_device().unwrap();
    let list = sequential_settle_list(1);
    // H1 and H2 masses differ.
    let masses = vec![M_O, M_H, M_H * 2.0];
    let err = SettleConstraintsState::new(gpu.device.clone(), &list, &masses, &[spce_type()])
        .unwrap_err();
    match err {
        SettleError::InvalidGroupShape { reason, .. } => {
            assert!(reason.contains("equal"), "reason: {reason}");
        }
        other => panic!("expected InvalidGroupShape, got {other:?}"),
    }
}

// --- Topology expansion --------------------------------------------------

fn tmp_topology(name: &str, body: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut dir = std::env::temp_dir();
    dir.push(format!("heddlemd-settle-{}-{}-{}", std::process::id(), name, nanos));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("topology.top");
    std::fs::write(&p, body).unwrap();
    p
}

// rq-0d91d811
#[test]
fn settle_row_expands_to_three_canonical_constraints() {
    let path = tmp_topology("expand", "[constraints]\n0 1 2 SPCE\n");
    let registry = ConstraintRegistry::with_builtins();
    let (_b, _a, _dl, _e, cl, _ql) =
        load_topology_file(&path, 3, &[], &[], &[], &[spce_type()], &registry, UnitSystem::Atomic).unwrap();
    assert_eq!(cl.groups.len(), 1);
    assert_eq!(cl.groups[0].constraint_count, 3);
    let cs = &cl.group_constraints;
    assert_eq!(cs.len(), 3);
    assert_eq!((cs[0].local_i, cs[0].local_j), (0, 1));
    assert_eq!((cs[1].local_i, cs[1].local_j), (0, 2));
    assert_eq!((cs[2].local_i, cs[2].local_j), (1, 2));
    assert!((cs[0].r0 - R_OH).abs() < 1.0e-4);
    assert!((cs[1].r0 - R_OH).abs() < 1.0e-4);
    assert!((cs[2].r0 - R_HH).abs() < 1.0e-4);
}

// rq-6de1f8d5
#[test]
fn settle_group_adds_implicit_exclusions() {
    let path = tmp_topology("excl", "[constraints]\n0 1 2 SPCE\n");
    let registry = ConstraintRegistry::with_builtins();
    let (_b, _a, _dl, el, _cl, _ql) =
        load_topology_file(&path, 3, &[], &[], &[], &[spce_type()], &registry, UnitSystem::Atomic).unwrap();
    let has = |i: u32, j: u32| {
        el.entries
            .iter()
            .any(|e| e.atom_i == i && e.atom_j == j && e.scale_lj == 0.0 && e.scale_coul == 0.0)
    };
    assert!(has(0, 1), "missing (0,1) exclusion: {:?}", el.entries);
    assert!(has(0, 2), "missing (0,2) exclusion");
    assert!(has(1, 2), "missing (1,2) exclusion");
}

// rq-1ea7342a
#[test]
fn settle_row_wrong_atom_count_rejected_by_parser() {
    let path = tmp_topology("badrow", "[constraints]\n0 1 SPCE\n");
    let registry = ConstraintRegistry::with_builtins();
    let err = load_topology_file(&path, 3, &[], &[], &[], &[spce_type()], &registry, UnitSystem::Atomic).unwrap_err();
    match err {
        TopologyFileError::InvalidConstraintRow { reason, .. } => {
            assert!(reason.contains('3'), "reason should name expected count 3: {reason}");
        }
        other => panic!("expected InvalidConstraintRow, got {other:?}"),
    }
}

// --- Position reset (SETTLE) ---------------------------------------------

fn make_slot_and_buffers(
    gpu: &heddle_md::gpu::GpuContext,
    n_waters: usize,
) -> (SettleConstraintsState, ParticleBuffers, SimulationBox, Timings, ParticleState) {
    let list = sequential_settle_list(n_waters);
    let state = water_state(n_waters, 10.0);
    let buffers = ParticleBuffers::new(gpu, &state).unwrap();
    let sim_box = big_box(gpu);
    let timings = Timings::new(gpu).unwrap();
    let slot =
        SettleConstraintsState::new(gpu.device.clone(), &list, &state.masses, &[spce_type()])
            .unwrap();
    (slot, buffers, sim_box, timings, state)
}

// rq-d5b31775
#[test]
fn settle_positions_restores_after_uniform_translation() {
    let gpu = init_device().unwrap();
    let (mut slot, mut buffers, sim_box, mut timings, state) = make_slot_and_buffers(&gpu, 1);
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    // Shift all atoms uniformly along x (a pure translation keeps it rigid;
    // SETTLE must reproduce the translated rigid geometry).
    let (mut px, py, pz) = buffers.download_positions().unwrap();
    for v in px.iter_mut() {
        *v += 0.02;
    }
    buffers.upload_positions(&px, &py, &pz).unwrap();
    let m = &state.masses;
    let total = (m[0] + m[1] + m[2]) as f64;
    let com_u = (m[0] * px[0] + m[1] * px[1] + m[2] * px[2]) as f64 / total;
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let (px, py, pz) = buffers.download_positions().unwrap();
    assert_rigid(&px, &py, &pz, 0, "after uniform translation");
    let com_c = (m[0] * px[0] + m[1] * px[1] + m[2] * px[2]) as f64 / total;
    assert!((com_c - com_u).abs() < 5.0e-3, "COM x drifted: {com_c} vs {com_u}");
}

// rq-603bd03b
#[test]
fn settle_positions_restores_after_per_atom_kick() {
    let gpu = init_device().unwrap();
    let (mut slot, mut buffers, sim_box, mut timings, _state) = make_slot_and_buffers(&gpu, 1);
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let (mut px, mut py, mut pz) = buffers.download_positions().unwrap();
    let perturb = [(0.03, 0.02, -0.01), (-0.02, 0.015, 0.02), (0.01, -0.025, 0.012)];
    for (i, (dx, dy, dz)) in perturb.iter().enumerate() {
        px[i] += dx;
        py[i] += dy;
        pz[i] += dz;
    }
    buffers.upload_positions(&px, &py, &pz).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let (px, py, pz) = buffers.download_positions().unwrap();
    assert_rigid(&px, &py, &pz, 0, "after per-atom kick");
}

// rq-3bd5ee23
#[test]
fn settle_positions_updates_half_step_velocities_consistently() {
    let gpu = init_device().unwrap();
    let (mut slot, mut buffers, sim_box, mut timings, _state) = make_slot_and_buffers(&gpu, 1);
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    // Perturb (unconstrained positions) and record them.
    let (mut px, py, pz) = buffers.download_positions().unwrap();
    px[1] += 0.05;
    buffers.upload_positions(&px, &py, &pz).unwrap();
    let u_x: Vec<Real> = buffers.download_positions().unwrap().0;
    let v_before: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let dt = PROD_DT;
    slot.apply_after_drift(&mut buffers, &sim_box, dt, &mut timings).unwrap();
    let c_x: Vec<Real> = buffers.download_positions().unwrap().0;
    let v_after: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    for i in 0..3 {
        let expected = v_before[i] + (c_x[i] - u_x[i]) / dt;
        assert!(
            (v_after[i] - expected).abs() < 1.0e-4,
            "velocity correction inconsistent for atom {i}: {} vs {}",
            v_after[i],
            expected
        );
    }
}

// rq-95be04ff
#[test]
fn settle_positions_writes_non_zero_position_level_virial() {
    let gpu = init_device().unwrap();
    let (mut slot, mut buffers, sim_box, mut timings, _state) = make_slot_and_buffers(&gpu, 1);
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let (mut px, py, pz) = buffers.download_positions().unwrap();
    let dx = px[1] - px[0];
    px[1] = px[0] + dx * 1.05;
    buffers.upload_positions(&px, &py, &pz).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let virial: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.constraint_virial).unwrap();
    let nonzero = virial.iter().take(3).filter(|&&v| v.abs() > 1e-30).count();
    assert!(nonzero >= 2, "expected >=2 non-zero virial entries, got {virial:?}");
}

// rq-eba5c2ff
#[test]
fn settle_positions_handles_water_straddling_periodic_boundary() {
    let gpu = init_device().unwrap();
    let box_l: Real = 10.0;
    let sim_box = SimulationBox::new(&gpu.device, box_l, box_l, box_l, 0.0, 0.0, 0.0).unwrap();
    let list = sequential_settle_list(1);

    // Equilibrium triangle centred so it straddles the +x boundary: shift
    // by +box_l/2 along x and wrap into [-box_l/2, box_l/2).
    let d_oh = (R_OH * R_OH - (R_HH * 0.5) * (R_HH * 0.5)).sqrt();
    let wrap = |x: Real| -> Real {
        let mut v = x;
        while v >= box_l * 0.5 {
            v -= box_l;
        }
        while v < -box_l * 0.5 {
            v += box_l;
        }
        v
    };
    let shift = box_l * 0.5;
    let px0 = vec![wrap(d_oh + shift), wrap(shift), wrap(shift)];
    let py0 = vec![0.0, -R_HH * 0.5, R_HH * 0.5];
    let pz0 = vec![0.0, 0.0, 0.0];
    let masses = vec![M_O, M_H, M_H];
    let state = ParticleState::new(
        px0.clone(),
        py0.clone(),
        pz0.clone(),
        vec![0.0; 3],
        vec![0.0; 3],
        vec![0.0; 3],
        masses,
        vec![0.0; 3],
        vec![0u32; 3],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut slot =
        SettleConstraintsState::new(gpu.device.clone(), &list, &state.masses, &[spce_type()])
            .unwrap();

    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    // Perturb O along +x (still straddling).
    let (mut px, py, pz) = buffers.download_positions().unwrap();
    px[0] = wrap(px[0] + 0.03);
    buffers.upload_positions(&px, &py, &pz).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();

    let (px, py, pz) = buffers.download_positions().unwrap();
    let mid = |a: usize, b: usize, p: &[Real]| -> Real {
        let mut d = p[a] - p[b];
        while d > box_l * 0.5 {
            d -= box_l;
        }
        while d < -box_l * 0.5 {
            d += box_l;
        }
        d
    };
    let mi_dist = |a: usize, b: usize| -> Real {
        let dx = mid(a, b, &px);
        let dy = mid(a, b, &py);
        let dz = mid(a, b, &pz);
        (dx * dx + dy * dy + dz * dz).sqrt()
    };
    let tol = 1.0e-4;
    assert!(((mi_dist(0, 1) - R_OH) / R_OH).abs() < tol, "O-H1 (min-image) = {}", mi_dist(0, 1));
    assert!(((mi_dist(0, 2) - R_OH) / R_OH).abs() < tol, "O-H2 (min-image) = {}", mi_dist(0, 2));
    assert!(((mi_dist(1, 2) - R_HH) / R_HH).abs() < tol, "H-H (min-image) = {}", mi_dist(1, 2));
}

// rq-9638eee5
#[test]
fn settle_positions_handles_production_dt_thermal_amplitude() {
    let gpu = init_device().unwrap();
    let (mut slot, mut buffers, sim_box, mut timings, _state) = make_slot_and_buffers(&gpu, 1);
    let vx = vec![2.0e-4, -1.0e-3, 1.3e-3];
    let vy = vec![-1.5e-4, 1.2e-3, -0.6e-3];
    let vz = vec![1.0e-4, -0.8e-3, 1.1e-3];
    gpu.device.htod_sync_copy_into(&vx, &mut buffers.velocities_x).unwrap();
    gpu.device.htod_sync_copy_into(&vy, &mut buffers.velocities_y).unwrap();
    gpu.device.htod_sync_copy_into(&vz, &mut buffers.velocities_z).unwrap();
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let (mut px, mut py, mut pz) = buffers.download_positions().unwrap();
    for i in 0..3 {
        px[i] += vx[i] * PROD_DT;
        py[i] += vy[i] * PROD_DT;
        pz[i] += vz[i] * PROD_DT;
    }
    buffers.upload_positions(&px, &py, &pz).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let (px, py, pz) = buffers.download_positions().unwrap();
    assert_rigid(&px, &py, &pz, 0, "production dt thermal");
    let vx_after: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    for i in 0..3 {
        assert!(vx_after[i].is_finite(), "v_x[{i}] not finite");
    }
}

// --- Velocity reset (SETTLE) ---------------------------------------------

fn inject_velocities(gpu: &heddle_md::gpu::GpuContext, buffers: &mut ParticleBuffers) {
    let vx = vec![1.0e-3, -0.5e-3, 0.3e-3];
    let vy = vec![0.2e-3, 0.4e-3, -0.7e-3];
    let vz = vec![-0.6e-3, 0.1e-3, 0.8e-3];
    gpu.device.htod_sync_copy_into(&vx, &mut buffers.velocities_x).unwrap();
    gpu.device.htod_sync_copy_into(&vy, &mut buffers.velocities_y).unwrap();
    gpu.device.htod_sync_copy_into(&vz, &mut buffers.velocities_z).unwrap();
}

// rq-9dd716cf
#[test]
fn settle_velocities_zeroes_constraint_derivatives() {
    let gpu = init_device().unwrap();
    let (mut slot, mut buffers, sim_box, mut timings, _state) = make_slot_and_buffers(&gpu, 1);
    inject_velocities(&gpu, &mut buffers);
    slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let (px, py, pz) = buffers.download_positions().unwrap();
    let vx: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vz: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
    let dot = |a: usize, b: usize| -> f64 {
        let drx = (px[a] - px[b]) as f64;
        let dry = (py[a] - py[b]) as f64;
        let drz = (pz[a] - pz[b]) as f64;
        let dvx = (vx[a] - vx[b]) as f64;
        let dvy = (vy[a] - vy[b]) as f64;
        let dvz = (vz[a] - vz[b]) as f64;
        drx * dvx + dry * dvy + drz * dvz
    };
    let tol = 5.0e-9;
    assert!(dot(0, 1).abs() < tol, "(v_O-v_H1)·r = {}", dot(0, 1));
    assert!(dot(0, 2).abs() < tol, "(v_O-v_H2)·r = {}", dot(0, 2));
    assert!(dot(1, 2).abs() < tol, "(v_H1-v_H2)·r = {}", dot(1, 2));
}

// rq-38d09177
#[test]
fn settle_velocities_preserves_com_velocity() {
    let gpu = init_device().unwrap();
    let (mut slot, mut buffers, sim_box, mut timings, state) = make_slot_and_buffers(&gpu, 1);
    let vx = vec![1.0e-3, -0.5e-3, 0.3e-3];
    let vy = vec![0.2e-3, 0.4e-3, -0.7e-3];
    let vz = vec![-0.6e-3, 0.1e-3, 0.8e-3];
    gpu.device.htod_sync_copy_into(&vx, &mut buffers.velocities_x).unwrap();
    gpu.device.htod_sync_copy_into(&vy, &mut buffers.velocities_y).unwrap();
    gpu.device.htod_sync_copy_into(&vz, &mut buffers.velocities_z).unwrap();
    let m = &state.masses;
    let total = (m[0] + m[1] + m[2]) as f64;
    let com = |v: &[Real]| (m[0] * v[0] + m[1] * v[1] + m[2] * v[2]) as f64 / total;
    let cv0 = (com(&vx), com(&vy), com(&vz));
    slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let vx_c: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy_c: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vz_c: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
    let cv1 = (com(&vx_c), com(&vy_c), com(&vz_c));
    let tol = 1.0e-6;
    assert!((cv0.0 - cv1.0).abs() < tol, "COM vx changed");
    assert!((cv0.1 - cv1.1).abs() < tol, "COM vy changed");
    assert!((cv0.2 - cv1.2).abs() < tol, "COM vz changed");
}

// rq-b317db56
#[test]
fn settle_velocities_accumulates_virial_when_dt_positive() {
    let gpu = init_device().unwrap();
    let (mut slot, mut buffers, sim_box, mut timings, _state) = make_slot_and_buffers(&gpu, 1);
    // Seed a known position-level virial pattern, then run after_kick.
    let seed = vec![1.0, -2.0, 3.0];
    gpu.device.htod_sync_copy_into(&seed, &mut slot.constraint_virial).unwrap();
    inject_velocities(&gpu, &mut buffers);
    slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let after: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.constraint_virial).unwrap();
    let changed = (0..3).any(|i| (after[i] - seed[i]).abs() > 1e-30);
    assert!(changed, "velocity-level virial should have accumulated on top of the seed");
}

// rq-b7ed6d52
#[test]
fn settle_velocities_skips_virial_when_dt_non_positive() {
    let gpu = init_device().unwrap();
    let (mut slot, mut buffers, sim_box, mut timings, _state) = make_slot_and_buffers(&gpu, 1);
    inject_velocities(&gpu, &mut buffers);
    let seed = vec![1.0, -2.0, 3.0];
    gpu.device.htod_sync_copy_into(&seed, &mut slot.constraint_virial).unwrap();
    let vx_in: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    slot.apply_initial_velocity_projection(&mut buffers, &sim_box, &mut timings).unwrap();
    let after: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.constraint_virial).unwrap();
    assert_eq!(seed, after, "dt<=0 must not touch constraint_virial");
    // Sanity: velocities were still projected.
    let vx_out: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    assert_ne!(vx_in, vx_out, "velocities should have been projected");
}

// --- Virial scatter ------------------------------------------------------

// rq-9883ee42
#[test]
fn settle_virial_scatter_additive() {
    let gpu = init_device().unwrap();
    let (mut slot, mut buffers, sim_box, mut timings, _state) = make_slot_and_buffers(&gpu, 1);
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let (mut px, py, pz) = buffers.download_positions().unwrap();
    let dx = px[1] - px[0];
    px[1] = px[0] + dx * 1.05;
    buffers.upload_positions(&px, &py, &pz).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let pre = vec![10.0, 20.0, 30.0];
    gpu.device.htod_sync_copy_into(&pre, &mut buffers.virials).unwrap();
    slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let post: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.virials).unwrap();
    let cv: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.constraint_virial).unwrap();
    let tol = 1e-5;
    for i in 0..3 {
        let expected = pre[i] + cv[i];
        assert!(
            (post[i] - expected).abs() <= tol * expected.abs().max(1.0),
            "virials[{i}] = {} vs expected {}",
            post[i],
            expected
        );
    }
}

// rq-a8d63610
#[test]
fn settle_virial_scatter_two_disjoint_groups() {
    let gpu = init_device().unwrap();
    let (mut slot, mut buffers, sim_box, mut timings, _state) = make_slot_and_buffers(&gpu, 2);
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let (mut px, py, pz) = buffers.download_positions().unwrap();
    for w in 0..2 {
        let i_o = 3 * w;
        let i_h1 = 3 * w + 1;
        let dx = px[i_h1] - px[i_o];
        px[i_h1] = px[i_o] + dx * 1.05;
    }
    buffers.upload_positions(&px, &py, &pz).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let pre = vec![0.0; 6];
    gpu.device.htod_sync_copy_into(&pre, &mut buffers.virials).unwrap();
    slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let post: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.virials).unwrap();
    for atom in 0..6 {
        assert!(post[atom].abs() > 1e-30, "atom {atom} got no virial: {post:?}");
    }
}

// --- Position-only projection (minimization) -----------------------------

// rq-bab6638a
#[test]
fn settle_positions_no_velocity_restores_from_off_manifold() {
    let gpu = init_device().unwrap();
    let (mut slot, mut buffers, sim_box, mut timings, _state) = make_slot_and_buffers(&gpu, 1);
    let vx_snap: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let virial_snap: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.constraint_virial).unwrap();
    let (mut px, mut py, mut pz) = buffers.download_positions().unwrap();
    for i in 1..3 {
        let dx = px[i] - px[0];
        let dy = py[i] - py[0];
        let dz = pz[i] - pz[0];
        px[i] = px[0] + dx * 1.05;
        py[i] = py[0] + dy * 1.05;
        pz[i] = pz[0] + dz * 1.05;
    }
    buffers.upload_positions(&px, &py, &pz).unwrap();
    slot.apply_position_projection_only(&mut buffers, &sim_box, &mut timings).unwrap();
    let (px, py, pz) = buffers.download_positions().unwrap();
    assert_rigid(&px, &py, &pz, 0, "position-only projection");
    let vx: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let virial: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.constraint_virial).unwrap();
    assert_eq!(vx, vx_snap, "velocities must be untouched");
    assert_eq!(virial, virial_snap, "constraint_virial must be untouched");
}

// --- Reproducibility -----------------------------------------------------

// rq-cc3f19a3
#[test]
fn two_runs_of_after_drift_are_byte_identical() {
    let gpu = init_device().unwrap();
    let run = || -> (Vec<Real>, Vec<Real>, Vec<Real>, Vec<Real>) {
        let (mut slot, mut buffers, sim_box, mut timings, _s) = make_slot_and_buffers(&gpu, 2);
        slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
        let (mut px, mut py, mut pz) = buffers.download_positions().unwrap();
        for i in 0..px.len() {
            px[i] += 0.013 * (i as Real + 1.0);
            py[i] -= 0.007 * (i as Real + 1.0);
            pz[i] += 0.004 * (i as Real + 1.0);
        }
        buffers.upload_positions(&px, &py, &pz).unwrap();
        slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
        let (px, _py, _pz) = buffers.download_positions().unwrap();
        let vx: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
        let cv: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.constraint_virial).unwrap();
        (px, vx, cv, _py)
    };
    let a = run();
    let b = run();
    assert_eq!(a.0, b.0, "positions differ");
    assert_eq!(a.1, b.1, "velocities differ");
    assert_eq!(a.2, b.2, "constraint_virial differs");
}

// rq-c90e3e46
#[test]
fn two_runs_of_after_kick_are_byte_identical() {
    let gpu = init_device().unwrap();
    let run = || -> (Vec<Real>, Vec<Real>) {
        let (mut slot, mut buffers, sim_box, mut timings, _s) = make_slot_and_buffers(&gpu, 2);
        inject_velocities_n(&gpu, &mut buffers, 2);
        slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
        let vx: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
        let cv: Vec<Real> = gpu.device.dtoh_sync_copy(&slot.constraint_virial).unwrap();
        (vx, cv)
    };
    let a = run();
    let b = run();
    assert_eq!(a.0, b.0, "velocities differ");
    assert_eq!(a.1, b.1, "constraint_virial differs");
}

fn inject_velocities_n(gpu: &heddle_md::gpu::GpuContext, buffers: &mut ParticleBuffers, n: usize) {
    let mut vx = Vec::new();
    let mut vy = Vec::new();
    let mut vz = Vec::new();
    for w in 0..n {
        let s = (w as Real + 1.0) * 1.0e-4;
        vx.extend([s, -0.5 * s, 0.3 * s]);
        vy.extend([0.2 * s, 0.4 * s, -0.7 * s]);
        vz.extend([-0.6 * s, 0.1 * s, 0.8 * s]);
    }
    gpu.device.htod_sync_copy_into(&vx, &mut buffers.velocities_x).unwrap();
    gpu.device.htod_sync_copy_into(&vy, &mut buffers.velocities_y).unwrap();
    gpu.device.htod_sync_copy_into(&vz, &mut buffers.velocities_z).unwrap();
}

// --- Composition with the integrator framework ---------------------------

// rq-102b4b02
#[test]
fn full_round_trip_preserves_all_distances_over_100_steps() {
    let gpu = init_device().unwrap();
    let (mut slot, mut buffers, sim_box, mut timings, _state) = make_slot_and_buffers(&gpu, 1);
    inject_velocities(&gpu, &mut buffers);
    for step in 0..100 {
        slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
        // Unconstrained drift: x += v · dt.
        let (mut px, mut py, mut pz) = buffers.download_positions().unwrap();
        let vx: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
        let vy: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
        let vz: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
        for i in 0..px.len() {
            px[i] += vx[i] * PROD_DT;
            py[i] += vy[i] * PROD_DT;
            pz[i] += vz[i] * PROD_DT;
        }
        buffers.upload_positions(&px, &py, &pz).unwrap();
        slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
        slot.apply_after_kick(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
        let (px, py, pz) = buffers.download_positions().unwrap();
        assert_rigid(&px, &py, &pz, 0, &format!("step {step}"));
    }
}

// rq-c129167f
#[test]
fn no_constraint_slot_leaves_distances_drifting() {
    let gpu = init_device().unwrap();
    let state = water_state(1, 10.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    // Inject velocities and drift unconstrained for 20 steps with NO
    // projection.
    inject_velocities(&gpu, &mut buffers);
    for _ in 0..20 {
        let (mut px, mut py, mut pz) = buffers.download_positions().unwrap();
        let vx: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
        let vy: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
        let vz: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
        for i in 0..px.len() {
            px[i] += vx[i] * PROD_DT;
            py[i] += vy[i] * PROD_DT;
            pz[i] += vz[i] * PROD_DT;
        }
        buffers.upload_positions(&px, &py, &pz).unwrap();
    }
    let (px, py, pz) = buffers.download_positions().unwrap();
    let d_oh1 = dist(px[0], py[0], pz[0], px[1], py[1], pz[1]);
    let drift = ((d_oh1 - R_OH) / R_OH).abs();
    assert!(drift > 5.0e-3, "without SETTLE, O-H1 should drift; got |Δ|/r = {drift}");
}

// --- Graph / minimization compatibility & registry ----------------------

// rq-63926477
#[test]
fn settle_builder_reports_graph_compatibility() {
    use heddle_md::integrator::settle::SettleBuilder;
    use heddle_md::integrator::ConstraintBuilder;
    let b = SettleBuilder;
    assert!(b.graph_compatible(&spce_type().params));
}

// rq-c5898418
#[test]
fn settle_builder_supports_position_projection_only() {
    use heddle_md::integrator::settle::SettleBuilder;
    use heddle_md::integrator::ConstraintBuilder;
    let b = SettleBuilder;
    assert!(b.supports_position_projection_only(&spce_type().params));
}

// rq-f4acb881
#[test]
fn registry_with_builtins_registers_settle_and_shake() {
    let registry = ConstraintRegistry::with_builtins();
    assert!(registry.lookup("settle").is_some(), "settle not registered");
    assert!(registry.lookup("shake").is_some(), "shake not registered");
    let b = registry.lookup("settle").unwrap();
    assert_eq!(b.kind_name(), "settle");
}

// =====================================================================
// Miyamoto-Kollman closed-form position reset
// =====================================================================

fn shake_spce_type() -> NamedSlotConfig {
    NamedSlotConfig::from_params_str(
        "SPCE",
        "shake",
        &format!(
            "atoms = 3\nconstraints = [\n  {{ i = 0, j = 1, d = {} }},\n  {{ i = 0, j = 2, d = {} }},\n  {{ i = 1, j = 2, d = {} }},\n]\n",
            R_OH as f64, R_OH as f64, R_HH as f64,
        ),
    )
}

fn shake_list_1() -> ConstraintList {
    ConstraintList {
        groups: vec![ConstraintGroup {
            atom_offset: 0,
            atom_count: 3,
            constraint_offset: 0,
            constraint_count: 3,
            constraint_type_index: 0,
        }],
        group_atoms: vec![0, 1, 2],
        group_constraints: vec![
            GroupConstraint { local_i: 0, local_j: 1, r0: R_OH },
            GroupConstraint { local_i: 0, local_j: 2, r0: R_OH },
            GroupConstraint { local_i: 1, local_j: 2, r0: R_HH },
        ],
        particle_count: 3,
    }
}

// rq-76abf8d7
#[test]
fn settle_positions_restores_exact_rigidity_single_pass() {
    let gpu = init_device().unwrap();
    let (mut slot, mut buffers, sim_box, mut timings, _state) = make_slot_and_buffers(&gpu, 1);
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    // Break every constraint by ~1e-2 a_0 with an arbitrary per-atom kick.
    let (mut px, mut py, mut pz) = buffers.download_positions().unwrap();
    let kick = [(0.011, -0.007, 0.004), (-0.009, 0.012, -0.006), (0.006, -0.010, 0.013)];
    for (i, (dx, dy, dz)) in kick.iter().enumerate() {
        px[i] += dx;
        py[i] += dy;
        pz[i] += dz;
    }
    buffers.upload_positions(&px, &py, &pz).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let (px, py, pz) = buffers.download_positions().unwrap();
    // The closed-form rotation restores rigidity far tighter than the
    // iterative tolerance (1e-4); a single pass reaches f32 round-off.
    let d_oh1 = dist(px[0], py[0], pz[0], px[1], py[1], pz[1]);
    let d_oh2 = dist(px[0], py[0], pz[0], px[2], py[2], pz[2]);
    let d_hh = dist(px[1], py[1], pz[1], px[2], py[2], pz[2]);
    let tol = 1.0e-5;
    assert!(((d_oh1 - R_OH) / R_OH).abs() < tol, "O-H1 {d_oh1} vs {R_OH}");
    assert!(((d_oh2 - R_OH) / R_OH).abs() < tol, "O-H2 {d_oh2} vs {R_OH}");
    assert!(((d_hh - R_HH) / R_HH).abs() < tol, "H-H {d_hh} vs {R_HH}");
}

// rq-3163a55d
#[test]
fn mk_reset_agrees_with_converged_shake_projection() {
    let gpu = init_device().unwrap();
    let state = water_state(1, 10.0);
    let sim_box = big_box(&gpu);
    let mut timings = Timings::new(&gpu).unwrap();

    // SETTLE (M-K) and SHAKE (converged minimal-displacement) over the same
    // water, same snapshot, same unconstrained perturbation.
    let mut settle = SettleConstraintsState::new(
        gpu.device.clone(), &sequential_settle_list(1), &state.masses, &[spce_type()],
    ).unwrap();
    let mut shake = ShakeConstraintsState::new(
        gpu.device.clone(), &shake_list_1(), &state.masses, &[shake_spce_type()],
    ).unwrap();

    let mut bs = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut bh = ParticleBuffers::new(&gpu, &state).unwrap();
    settle.apply_before_drift(&mut bs, &sim_box, PROD_DT, &mut timings).unwrap();
    shake.apply_before_drift(&mut bh, &sim_box, PROD_DT, &mut timings).unwrap();

    // Identical unconstrained perturbation (~1e-2 a_0) applied to both.
    let kick = [(0.012, -0.008, 0.005), (-0.007, 0.011, -0.006), (0.004, -0.009, 0.010)];
    for b in [&mut bs, &mut bh] {
        let (mut px, mut py, mut pz) = b.download_positions().unwrap();
        for (i, (dx, dy, dz)) in kick.iter().enumerate() {
            px[i] += dx;
            py[i] += dy;
            pz[i] += dz;
        }
        b.upload_positions(&px, &py, &pz).unwrap();
    }
    settle.apply_after_drift(&mut bs, &sim_box, PROD_DT, &mut timings).unwrap();
    shake.apply_after_drift(&mut bh, &sim_box, PROD_DT, &mut timings).unwrap();

    let (sx, sy, sz) = bs.download_positions().unwrap();
    let (hx, hy, hz) = bh.download_positions().unwrap();
    // The two algorithms reach the same constrained configuration; they differ
    // only at the order of the small per-step orientation correction.
    let tol = 1.0e-3;
    for i in 0..3 {
        let d = dist(sx[i], sy[i], sz[i], hx[i], hy[i], hz[i]);
        assert!(d < tol, "atom {i}: M-K vs SHAKE differ by {d} a_0");
    }
}

// rq-28cc7d41
#[test]
fn settle_positions_guards_near_degenerate_geometry() {
    let gpu = init_device().unwrap();
    let (mut slot, mut buffers, sim_box, mut timings, _state) = make_slot_and_buffers(&gpu, 1);
    slot.apply_before_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    // Drive the oxygen far out of the molecular plane (>> ra ≈ 0.12 a_0), so
    // sinphi = za1d/ra would exceed 1 without clamping, and squash the two
    // hydrogens toward the symmetry axis (near-collinear unconstrained input).
    let (mut px, mut py, mut pz) = buffers.download_positions().unwrap();
    pz[0] += 0.8;             // O far out of plane
    px[1] = px[0]; py[1] = py[0] + 0.05;
    px[2] = px[0]; py[2] = py[0] - 0.05;
    buffers.upload_positions(&px, &py, &pz).unwrap();
    slot.apply_after_drift(&mut buffers, &sim_box, PROD_DT, &mut timings).unwrap();
    let (px, py, pz) = buffers.download_positions().unwrap();
    for i in 0..3 {
        assert!(px[i].is_finite() && py[i].is_finite() && pz[i].is_finite(),
            "atom {i} produced a non-finite coordinate");
    }
    // Still a rigid water afterwards (clamped, not NaN).
    assert_rigid(&px, &py, &pz, 0, "near-degenerate input");
}

// rq-9e22adb3
#[test]
fn settle_positions_no_velocity_is_noop_for_rigid_molecule() {
    let gpu = init_device().unwrap();
    let (mut slot, mut buffers, sim_box, mut timings, _state) = make_slot_and_buffers(&gpu, 1);
    // The water_state geometry already satisfies the canonical constraints
    // exactly, so the projection must leave positions byte-identical.
    let (bx, by, bz) = buffers.download_positions().unwrap();
    slot.apply_position_projection_only(&mut buffers, &sim_box, &mut timings).unwrap();
    let (ax, ay, az) = buffers.download_positions().unwrap();
    assert_eq!(bx, ax, "x positions perturbed by projection of a rigid molecule");
    assert_eq!(by, ay, "y positions perturbed");
    assert_eq!(bz, az, "z positions perturbed");
}

// rq-261b5b46 — End-to-end: steepest-descent minimization with a SETTLE
// constraint slot converges (the line search does not collapse to a zero
// step, the regression that the identity-for-rigid projection prevents).
#[test]
fn sd_minimization_with_settle_converges() {
    use std::fs;
    let gpu = init_device().unwrap();
    let _ = gpu; // ensure a GPU is present for this end-to-end run
    let dir = std::env::temp_dir().join(format!("heddlemd-settle-min-{}", std::process::id()));
    if dir.exists() {
        let _ = fs::remove_dir_all(&dir);
    }
    fs::create_dir_all(&dir).unwrap();

    // Two rigid SPC/E waters with their oxygens 3.0 Å apart (repulsive O–O
    // LJ → a real force to minimize). Each water is at the exact canonical
    // geometry, so every trial-step projection starts from a rigid molecule.
    let init = "6\n\
        Lattice=\"5.0e-9 0 0 0 5.0e-9 0 0 0 5.0e-9\" Properties=species:S:1:pos:R:3\n\
        O  -1.500000000e-10 0.000000000e0 0.000000000e0\n\
        H  -0.500000000e-10 0.000000000e0 0.000000000e0\n\
        H  -1.833800000e-10 9.426400000e-11 0.000000000e0\n\
        O   1.500000000e-10 0.000000000e0 0.000000000e0\n\
        H   2.500000000e-10 0.000000000e0 0.000000000e0\n\
        H   1.166200000e-10 9.426400000e-11 0.000000000e0\n";
    fs::write(dir.join("water.in.xyz"), init).unwrap();
    fs::write(
        dir.join("water.in.topology"),
        "[constraints]\n0 1 2 SPCE\n3 4 5 SPCE\n",
    )
    .unwrap();

    let cfg = r#"schema_version = 1
units = "si"
init = "water.in.xyz"
topology = "water.in.topology"

[simulation]
seed = 1
temperature = 0.0

[[minimization]]
name = "min"

[minimization.algorithm]
kind = "steepest-descent"
initial_step = 1.0e-13
max_step = 1.0e-11
force_tolerance = 1.0e-11
energy_tolerance = 1.0e-6
max_iterations = 500

[minimization.output]
minlog_every = 1
trajectory_every = 0

[[particle_types]]
name = "O"
mass = 2.6566e-26
charge = 0.0

[[particle_types]]
name = "H"
mass = 1.6735e-27
charge = 0.0

[[pair_interactions]]
between = ["O", "O"]
potential = "lennard-jones"
sigma = 3.166e-10
epsilon = 1.080e-21
cutoff = 1.0e-9
r_switch = 1.0e-9

[[pair_interactions]]
between = ["H", "H"]
potential = "lennard-jones"
sigma = 1.0e-10
epsilon = 1.0e-30
cutoff = 1.0e-9
r_switch = 1.0e-9

[[pair_interactions]]
between = ["H", "O"]
potential = "lennard-jones"
sigma = 1.0e-10
epsilon = 1.0e-30
cutoff = 1.0e-9
r_switch = 1.0e-9

[[constraint_types]]
name = "SPCE"
kind = "settle"
d_OH = 1.0e-10
d_HH = 1.633e-10

[neighbor_list]
mode = "all-pairs"
"#;
    let cfg_path = dir.join("water.in.toml");
    fs::write(&cfg_path, cfg).unwrap();

    let summary = match heddle_md::runner::run_simulation(&cfg_path) {
        Ok(s) => s,
        Err(e) => panic!("run_simulation failed (line search likely collapsed): {e:?}"),
    };
    assert_eq!(summary.phases.len(), 1);
    let ps = &summary.phases[0];
    assert_eq!(ps.kind, "minimization");
    assert!(ps.n_steps > 0 && ps.n_steps <= 500, "iters = {}", ps.n_steps);
    assert!(ps.convergence.is_some(), "minimization did not converge");
}
