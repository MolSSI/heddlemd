//! Potential consistency harness tests. Implements the Gherkin scenarios of
//! `rqm/forces/potential-consistency-harness.md`.
//!
//! Positive scenarios drive built-in fixtures through the real GPU force
//! pipeline. Negative scenarios drive the same invariant checks against a CPU
//! `Evaluator` carrying an injected defect, modelling the fragment bug each
//! scenario describes — this exercises the detection logic directly without
//! hand-authoring a deliberately broken GPU kernel.

use heddle_md::gpu::init_device;

mod common;
use common::consistency::{
    assert_all_builtin_potentials_consistent, assert_fixture_coverage, assert_potential_consistent,
    builtin_consistency_fixtures, builtin_fragment_labels, check_force_energy, check_newton,
    check_pair_continuity, check_reference_points, check_virial, ConsistencyFixture, Eval,
    Evaluator, PotentialShape, ReferencePoint, Tolerance,
};
use std::collections::HashSet;

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

fn fixture(label: &str) -> ConsistencyFixture {
    builtin_consistency_fixtures()
        .into_iter()
        .find(|f| f.label == label)
        .unwrap_or_else(|| panic!("no built-in fixture labelled {label}"))
}

fn pair_geometry(r: f64) -> Vec<[f64; 3]> {
    vec![[0.0, 0.0, 0.0], [r, 0.0, 0.0]]
}

/// A CPU pairwise evaluator over a decaying potential `U(r) = A/r²`, with an
/// injectable defect. Used only for the negative scenarios: each `Bug` models
/// the described fragment mistake at the evaluator level.
#[derive(Clone, Copy)]
enum Bug {
    None,
    SignFlip,
    MissingInvR,
    HalvedEnergy,
    ScaledEnergy(f64),
    AsymmetricForce,
    InconsistentVirial,
    DiscontinuousSwitch,
}

struct CpuPair {
    cutoff: f64,
    truncate: bool,
    r_switch: f64,
    step: f64,
    bug: Bug,
}

impl CpuPair {
    fn new(bug: Bug) -> Self {
        CpuPair { cutoff: 4.0, truncate: false, r_switch: 0.0, step: 0.0, bug }
    }
}

impl Evaluator for CpuPair {
    fn eval(&mut self, pos: &[[f64; 3]]) -> Eval {
        let a = 1.0_f64;
        let d = [pos[0][0] - pos[1][0], pos[0][1] - pos[1][1], pos[0][2] - pos[1][2]];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let r = r2.sqrt();
        let (mut u, dudr) = if self.truncate && r >= self.cutoff {
            (0.0, 0.0)
        } else if self.truncate {
            (a * (1.0 / r.powi(2) - 1.0 / self.cutoff.powi(2)), -2.0 * a / r.powi(3))
        } else {
            (a / r.powi(2), -2.0 * a / r.powi(3))
        };
        if matches!(self.bug, Bug::DiscontinuousSwitch) && r < self.r_switch {
            u += self.step;
        }
        // Correct per-atom force: F0[k] = −dU/dr · (Δk / r); F1 = −F0.
        let mut f0 = [0.0; 3];
        if r > 0.0 {
            for k in 0..3 {
                f0[k] = -dudr * (d[k] / r);
            }
        }
        let mut f1 = [-f0[0], -f0[1], -f0[2]];
        let mut w = -r * dudr; // scalar virial for a central pair force
        match self.bug {
            Bug::SignFlip => {
                for k in 0..3 {
                    f0[k] = -f0[k];
                    f1[k] = -f1[k];
                }
            }
            Bug::MissingInvR => {
                for k in 0..3 {
                    f0[k] = -dudr * d[k]; // omit the 1/r
                    f1[k] = -f0[k];
                }
            }
            Bug::HalvedEnergy => u *= 0.5,
            Bug::ScaledEnergy(kf) => {
                u *= kf;
                for k in 0..3 {
                    f0[k] *= kf;
                    f1[k] *= kf;
                }
                w *= kf;
            }
            Bug::AsymmetricForce => f1 = f0, // both atoms pushed the same way
            Bug::InconsistentVirial => w *= 2.0,
            Bug::None | Bug::DiscontinuousSwitch => {}
        }
        Eval { energy: u, forces: vec![f0, f1], virial: w }
    }
}

/// A bond-shaped, GPU-free fixture for driving the per-invariant checks against
/// a `CpuPair` (no continuity/cutoff obligations).
fn cpu_bond_fixture(label: &'static str) -> ConsistencyFixture {
    ConsistencyFixture::for_checks(label, PotentialShape::Bond, vec![1.2, 1.6], pair_geometry, 0.0, 0.0)
}

// =====================================================================
// Positive scenarios (built-in fixtures, real GPU pipeline)
// =====================================================================

#[test] // rq-2d06b5d2
fn correct_pair_passes_finite_difference() {
    let gpu = init_device().unwrap();
    let lj = fixture("lennard_jones");
    let mut ev = lj.gpu_evaluator(&gpu);
    check_force_energy(&lj, &mut ev);
}

#[test] // rq-fd3ba41e
fn equal_and_opposite_pair_forces_pass_newton() {
    let gpu = init_device().unwrap();
    let lj = fixture("lennard_jones");
    let mut ev = lj.gpu_evaluator(&gpu);
    check_newton(&lj, &mut ev);
}

#[test] // rq-0e8b48bd
fn correct_pair_virial_matches_scaling_fd() {
    let gpu = init_device().unwrap();
    let lj = fixture("lennard_jones");
    let mut ev = lj.gpu_evaluator(&gpu);
    check_virial(&lj, &mut ev);
}

#[test] // rq-fad67bb5
fn c1_switch_joins_smoothly_and_vanishes_at_cutoff() {
    let gpu = init_device().unwrap();
    let lj = fixture("lennard_jones");
    let mut ev = lj.gpu_evaluator(&gpu);
    check_pair_continuity(&lj, &mut ev);
}

#[test] // rq-bd34fa41
fn lennard_jones_reference_points_hold() {
    let gpu = init_device().unwrap();
    let lj = fixture("lennard_jones");
    let mut ev = lj.gpu_evaluator(&gpu);
    check_reference_points(&lj, &mut ev);
}

#[test] // rq-e4ad8b2c
fn angle_potential_reports_zero_virial() {
    let gpu = init_device().unwrap();
    let angle = fixture("harmonic_angle");
    // The virial check compares the reported W to the isotropic-scaling FD,
    // which is zero for an angle term.
    let mut ev = angle.gpu_evaluator(&gpu);
    check_virial(&angle, &mut ev);
    // The reported virial itself is zero within tolerance at every sample.
    for &theta in &angle.samples {
        let e = ev.eval(&(vec![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [theta.cos(), theta.sin(), 0.0]]));
        assert!(e.virial.abs() < 1.0e-2, "angle reported virial not zero: {}", e.virial);
    }
}

#[test] // rq-9db48a6d
fn dihedral_passes_all_applicable_invariants() {
    let gpu = init_device().unwrap();
    let dih = fixture("periodic_dihedral");
    assert_eq!(dih.shape, PotentialShape::Dihedral);
    // assert_potential_consistent runs FD, Newton, virial, and reference
    // points; continuity is skipped for a non-pair shape.
    assert_potential_consistent(&dih, &gpu);
}

#[test] // rq-cfa07a3c
fn every_builtin_fragment_potential_passes_the_sweep() {
    let gpu = init_device().unwrap();
    assert_all_builtin_potentials_consistent(&gpu);
}

#[test] // rq-7700c3b8
fn fixture_system_carries_exactly_one_slot() {
    let gpu = init_device().unwrap();
    for f in builtin_consistency_fixtures() {
        let (ff, sim_box, _m, _q, _t) = f.build_system(&gpu);
        // A CorrectionOnly fixture carries a second slot: a zero-epsilon
        // Lennard-Jones that supplies the neighbour list's cutoff (the fragment
        // under test declares none) and contributes exactly zero energy and
        // force, so the measurement stays attributable to the fragment.
        let expected_slots = if f.unbounded { 2 } else { 1 };
        assert_eq!(
            ff.slots.len(),
            expected_slots,
            "fixture {} built {} slots, expected {}",
            f.label,
            ff.slots.len(),
            expected_slots
        );
        let max_reach = f.cutoff.max(f.samples.iter().cloned().fold(0.0, f64::max));
        assert!(
            (sim_box.min_perpendicular_width() as f64) > 2.0 * max_reach,
            "fixture {} box too small: width {} vs reach {}",
            f.label,
            sim_box.min_perpendicular_width(),
            max_reach
        );
    }
}

// =====================================================================
// Negative scenarios (CPU evaluator with an injected defect)
// =====================================================================

#[test] // rq-0d01f64a
#[should_panic(expected = "force-energy")]
fn sign_flipped_force_fails_finite_difference() {
    let fx = cpu_bond_fixture("sign_flip");
    let mut ev = CpuPair::new(Bug::SignFlip);
    check_force_energy(&fx, &mut ev);
}

#[test] // rq-3654fa69
#[should_panic(expected = "force-energy")]
fn force_missing_inv_r_fails_finite_difference() {
    let fx = cpu_bond_fixture("missing_inv_r");
    let mut ev = CpuPair::new(Bug::MissingInvR);
    check_force_energy(&fx, &mut ev);
}

#[test] // rq-f417a0f5
#[should_panic(expected = "force-energy")]
fn bonded_halved_energy_fails_finite_difference() {
    let fx = cpu_bond_fixture("halved_energy");
    let mut ev = CpuPair::new(Bug::HalvedEnergy);
    check_force_energy(&fx, &mut ev);
}

#[test] // rq-9d9747ed
#[should_panic(expected = "Newton")]
fn asymmetric_force_fails_newton() {
    let fx = cpu_bond_fixture("asymmetric");
    let mut ev = CpuPair::new(Bug::AsymmetricForce);
    check_newton(&fx, &mut ev);
}

#[test] // rq-24c9bf90
#[should_panic(expected = "virial")]
fn inconsistent_virial_fails_virial_check() {
    let fx = cpu_bond_fixture("bad_virial");
    let mut ev = CpuPair::new(Bug::InconsistentVirial);
    check_virial(&fx, &mut ev);
}

#[test] // rq-6d39e77d
#[should_panic(expected = "continuity")]
fn discontinuous_switch_fails_continuity() {
    // A pair fixture with a genuine switching region (r_switch < cutoff) whose
    // energy jumps at r_switch.
    let fx = ConsistencyFixture::for_checks(
        "discontinuous",
        PotentialShape::Pair,
        vec![1.5, 2.0],
        pair_geometry,
        3.0, // r_switch
        4.0, // cutoff
    );
    let mut ev = CpuPair { cutoff: 4.0, truncate: true, r_switch: 3.0, step: 0.5, bug: Bug::DiscontinuousSwitch };
    check_pair_continuity(&fx, &mut ev);
}

#[test] // rq-1b1e47ee
#[should_panic(expected = "continuity")]
fn force_nonzero_beyond_cutoff_fails_continuity() {
    // r_switch == cutoff → no smoothness sweep; only the force-vanishes-beyond-
    // cutoff check runs. The untruncated potential keeps a nonzero force there.
    let fx = ConsistencyFixture::for_checks(
        "leaky_cutoff",
        PotentialShape::Pair,
        vec![1.5, 2.0],
        pair_geometry,
        4.0, // r_switch == cutoff
        4.0, // cutoff
    );
    let mut ev = CpuPair { cutoff: 4.0, truncate: false, r_switch: 4.0, step: 0.0, bug: Bug::None };
    check_pair_continuity(&fx, &mut ev);
}

#[test] // rq-525bf72b
#[should_panic(expected = "reference-point")]
fn uniformly_scaled_form_passes_fd_but_fails_reference_point() {
    // Energy and force both scaled by a constant: self-consistent (FD passes)
    // but wrong absolute energy (reference point fails).
    let fx = cpu_bond_fixture("scaled").with_reference_points(vec![ReferencePoint {
        coordinate: 1.0,
        energy: Some(1.0), // correct U(1) = A/1² = 1
        coord_force: None,
        tol: Tolerance::new(2.0e-2, 2.0e-2),
    }]);
    // FD is self-consistent and must NOT panic.
    let mut ev = CpuPair::new(Bug::ScaledEnergy(2.0));
    check_force_energy(&fx, &mut ev);
    // The reference-point energy is wrong → this panics.
    check_reference_points(&fx, &mut ev);
}

#[test] // rq-75cb6f7d
#[should_panic(expected = "coverage")]
fn builtin_without_fixture_fails_coverage() {
    let gpu = init_device().unwrap();
    // A fixture set missing one real built-in fragment label.
    let mut labels: HashSet<String> = builtin_fragment_labels(&gpu);
    let dropped = labels.iter().next().cloned().expect("at least one built-in fragment");
    labels.remove(&dropped);
    assert_fixture_coverage(&labels, &gpu);
}
