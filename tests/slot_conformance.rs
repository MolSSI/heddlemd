//! rq-c7f4d96b — Slot conformance: every registered thermostat and barostat,
//! run on dense rigid SPC/E water and asserted against the quantity it controls.
//!
//! See `rqm/integration/slot-conformance-harness.md` (rq-c7f4d96b). The harness lives in
//! `tests/common/slot_conformance.rs`.
//!
//! The `spce_water` e2e preset is two molecules in a 5 nm box — vacuum — so the
//! quantities a thermostat and a barostat control barely move there, and a
//! broken slot is indistinguishable from a working one. These tests run the same
//! slots on 729 molecules at liquid density, where they are not.

mod common;

use common::e2e::{Case, SystemBuilder};
use common::slot_conformance::{
    assert_barostat_coverage, assert_thermostat_coverage, builtin_barostat_cases,
    builtin_thermostat_cases, check_mean_density_value, check_mean_temperature_value,
    check_no_nan, kelvin_to_au, mean_density, mean_temperature_k, registered_barostat_kinds,
    registered_thermostat_kinds, run_barostat_case, run_thermostat_case, synthetic_phase,
    volume_for_density, Expect, SlotCase, CONFORMANCE_N_MOL, CONFORMANCE_SIDE,
};

// ---- Coverage -----------------------------------------------------------

/// Every registered thermostat must have a conformance case. A new thermostat
/// cannot be merged without declaring what it is supposed to do on dense water.
#[test] // rq-17119008
fn every_registered_thermostat_has_a_conformance_case() {
    assert_thermostat_coverage(&builtin_thermostat_cases());
}

/// Every registered barostat must have a conformance case.
#[test] // rq-6c125bac
fn every_registered_barostat_has_a_conformance_case() {
    assert_barostat_coverage(&builtin_barostat_cases());
}

/// The coverage check must actually detect an uncovered slot — otherwise it is
/// decoration. Drop a row and assert it fires. (The potential-consistency
/// harness has the same guard, `builtin_without_fixture_fails_coverage`.)
#[test] // rq-a22870b8
#[should_panic(expected = "has no conformance case")]
fn a_thermostat_without_a_case_fails_coverage() {
    let mut cases = builtin_thermostat_cases();
    cases.pop();
    assert_thermostat_coverage(&cases);
}

#[test] // rq-f797fd2c
#[should_panic(expected = "has no conformance case")]
fn a_barostat_without_a_case_fails_coverage() {
    let mut cases = builtin_barostat_cases();
    cases.pop();
    assert_barostat_coverage(&cases);
}

/// A case naming a kind nobody registered would silently stop testing anything.
#[test] // rq-754c7ddb
#[should_panic(expected = "is not registered")]
fn a_case_for_an_unregistered_kind_fails_coverage() {
    let mut cases = builtin_thermostat_cases();
    cases.push(SlotCase {
        kind: "does-not-exist",
        expect: Expect::HoldsTemperature { rel_tol: 0.03 },
        note: "synthetic",
    });
    assert_thermostat_coverage(&cases);
}

// ---- The checks have teeth ----------------------------------------------
//
// A conformance suite is only worth its tolerances. These feed each check the
// value the bug it exists to catch actually produced, and assert it panics. If
// someone loosens a tolerance far enough to have missed the real defect, these
// go red.

/// The pre-fix engine settled at 266 K against a 298.15 K setpoint under SETTLE
/// — the thermostat was coupling to kinetic energy the constraint projection was
/// about to delete. The temperature check must reject that.
#[test] // rq-4240aedb
#[should_panic(expected = "mean temperature")]
fn the_temperature_check_rejects_the_historical_settle_deficit() {
    check_mean_temperature_value(266.5, 298.15, 0.03, "historical", "project-before-couple bug");
}

/// The pre-fix engine's barostat could not see the constraint virial, read
/// +27,000 atm instead of +59 atm, and expanded the box until the density
/// reached 0.034 g/cm3. The density check must reject that.
#[test] // rq-6d2c1446
#[should_panic(expected = "mean density")]
fn the_density_check_rejects_the_historical_barostat_runaway() {
    check_mean_density_value(0.034, 0.92, 1.04, "historical", "publish-before-barostat bug");
}

/// ...and the same check must still ACCEPT a conforming mean, or the tolerance is
/// too tight to be useful.
#[test] // rq-8c06ce9c
fn the_temperature_check_accepts_a_conforming_mean() {
    for (t, k) in [(300.01, "csvr"), (297.97, "berendsen")] {
        check_mean_temperature_value(t, 298.15, 0.03, k, "conforming");
    }
}

#[test] // rq-1eb2642b
fn the_density_check_accepts_a_conforming_density() {
    for (d, k) in [(0.9669, "c-rescale"), (0.9680, "berendsen"), (0.9820, "monte-carlo")] {
        check_mean_density_value(d, 0.92, 1.04, k, "conforming");
    }
}

// ---- Units --------------------------------------------------------------
//
// `PhysicsSample` is in Hartree atomic units throughout: `temperature` carries
// k_B*T in Hartrees, `volume` is in Bohr^3. The `units` selector in a config
// governs the FILES, not this in-memory series. Read it raw and every check is
// silently wrong by a constant factor.

#[test] // rq-610c023c
fn temperature_is_converted_out_of_atomic_units() {
    let phase = synthetic_phase(&[
        (kelvin_to_au(300.0), 1.0),
        (kelvin_to_au(300.0), 1.0),
        (kelvin_to_au(296.0), 1.0),
        (kelvin_to_au(300.0), 1.0),
    ]);
    // The mean is taken over the second half.
    let mean = mean_temperature_k(&phase);
    assert!(
        (mean - 298.0).abs() < 1e-6,
        "expected 298 K from the second half, got {mean}"
    );
}

#[test] // rq-efb5ebb3
fn density_is_derived_from_the_box_volume_in_bohr3() {
    let v = volume_for_density(CONFORMANCE_N_MOL, 0.997);
    let phase = synthetic_phase(&[(0.0, v), (0.0, v), (0.0, v), (0.0, v)]);
    let rho = mean_density(&phase, CONFORMANCE_N_MOL);
    assert!(
        (rho - 0.997).abs() < 1e-6,
        "expected 0.997 g/cm3, got {rho}"
    );
}

#[test] // rq-c0c0f112
#[should_panic(expected = "non-positive mean box volume")]
fn a_non_positive_mean_volume_is_an_error_not_a_division() {
    let phase = synthetic_phase(&[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
    let _ = mean_density(&phase, CONFORMANCE_N_MOL);
}

#[test] // rq-1987903e
#[should_panic(expected = "diverged")]
fn a_diverging_slot_is_reported_as_divergence_not_as_a_bad_mean() {
    // A NaN must be named as a divergence at the step it appeared, not folded
    // into a mean that then fails some unrelated tolerance.
    let phase = synthetic_phase(&[
        (kelvin_to_au(298.0), 1.0),
        (f64::NAN, 1.0),
        (kelvin_to_au(298.0), 1.0),
        (kelvin_to_au(298.0), 1.0),
    ]);
    check_no_nan(&phase, "synthetic");
}

// ---- The preset ---------------------------------------------------------

#[test] // rq-5f3cf6c9
#[should_panic(expected = "below the cell-list minimum")]
fn the_preset_refuses_a_box_narrower_than_the_cell_list_allows() {
    let dir = Case::new("conf_preset_too_small");
    // 3^3 = 27 waters at liquid density gives L ~ 0.90 nm, far below the
    // 3 * (r_cut + r_skin) = 2.55 nm floor. Fail here, with the widths named,
    // rather than deep inside a cell-list build.
    let _ = SystemBuilder::dense_spce_water(3).write(&dir);
}

#[test] // rq-805728c8
fn the_presets_input_files_are_byte_identical_across_calls() {
    // The orientations come from a seeded LCG, not the run's RNG, so the
    // generated inputs are reproducible and the byte-identity assertions over
    // this preset stay meaningful.
    let a = Case::new("conf_preset_bytes_a");
    let b = Case::new("conf_preset_bytes_b");
    SystemBuilder::dense_spce_water(CONFORMANCE_SIDE).write(&a);
    SystemBuilder::dense_spce_water(CONFORMANCE_SIDE).write(&b);
    for f in ["sim.in.xyz", "sim.in.topology"] {
        assert_eq!(
            std::fs::read(a.dir().join(f)).unwrap(),
            std::fs::read(b.dir().join(f)).unwrap(),
            "{f} differs between two writes of the same preset"
        );
    }
}

// ---- The sweep ----------------------------------------------------------
//
// One test per slot rather than one loop over all of them, so a failure names
// the slot and the others still run.

#[test] // rq-eb7c5c08
fn thermostat_csvr_holds_its_setpoint_on_dense_water() {
    run_thermostat_case(&case(&builtin_thermostat_cases(), "csvr"));
}

#[test] // rq-eb7c5c08
fn thermostat_berendsen_holds_its_setpoint_on_dense_water() {
    run_thermostat_case(&case(&builtin_thermostat_cases(), "berendsen"));
}

/// KNOWN BROKEN — diverges to NaN under SETTLE at dt = 2 fs. Kept red on
/// purpose: the slot is registered, so a user can select it, so it must work.
#[test] // rq-eb7c5c08
fn thermostat_nose_hoover_chain_holds_its_setpoint_on_dense_water() {
    run_thermostat_case(&case(&builtin_thermostat_cases(), "nose-hoover-chain"));
}

/// KNOWN BROKEN — the per-atom Maxwell-Boltzmann resample injects energy into
/// the constrained degrees of freedom, which SETTLE then projects out, so the
/// run settles ~5% cold. Kept red on purpose.
#[test] // rq-eb7c5c08
fn thermostat_andersen_holds_its_setpoint_on_dense_water() {
    run_thermostat_case(&case(&builtin_thermostat_cases(), "andersen"));
}

#[test] // rq-716752a7
fn barostat_c_rescale_holds_liquid_density_on_dense_water() {
    run_barostat_case(&case(&builtin_barostat_cases(), "c-rescale"));
}

#[test] // rq-716752a7
fn barostat_berendsen_holds_liquid_density_on_dense_water() {
    run_barostat_case(&case(&builtin_barostat_cases(), "berendsen"));
}

#[test] // rq-716752a7
fn barostat_monte_carlo_holds_liquid_density_on_dense_water() {
    run_barostat_case(&case(&builtin_barostat_cases(), "monte-carlo"));
}

/// Look a case up by kind. Panics rather than silently skipping, so a renamed
/// kind cannot quietly drop a slot out of the sweep.
fn case(cases: &[SlotCase], kind: &str) -> SlotCase {
    *cases
        .iter()
        .find(|c| c.kind == kind)
        .unwrap_or_else(|| panic!("no conformance case for `{kind}`"))
}

/// Guard against the sweep and the tables drifting apart: every case in the
/// table must have a `#[test]` above driving it. Checked by count, which is
/// crude but catches the common failure (a slot is registered, a row is added
/// to satisfy coverage, and nobody writes the test that runs it).
#[test] // rq-af8c5af7
fn every_case_in_the_tables_is_driven_by_a_test() {
    const THERMOSTAT_SWEEP_TESTS: usize = 4;
    const BAROSTAT_SWEEP_TESTS: usize = 3;
    assert_eq!(
        builtin_thermostat_cases().len(),
        THERMOSTAT_SWEEP_TESTS,
        "a thermostat case was added or removed without updating the sweep tests in this file \
         (registered: {:?})",
        registered_thermostat_kinds()
    );
    assert_eq!(
        builtin_barostat_cases().len(),
        BAROSTAT_SWEEP_TESTS,
        "a barostat case was added or removed without updating the sweep tests in this file \
         (registered: {:?})",
        registered_barostat_kinds()
    );
}

