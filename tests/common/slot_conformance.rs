//! rq-c7f4d96b — Slot Conformance Harness.
//!
//! A battery of physical invariants run against **every registered thermostat
//! and barostat**, on a system where those invariants actually bite: dense,
//! rigid SPC/E water (`SystemBuilder::dense_spce_water`).
//!
//! This mirrors the Potential Consistency Harness (`tests/common/consistency.rs`,
//! `rqm/forces/potential-consistency-harness.md`) and exists for the same reason:
//! a per-slot test file proves the slot does *something*, but nothing checks that
//! every registered slot is covered, or that it is exercised on a system where a
//! defect would show. Two bugs lived through the whole suite because of that:
//!
//!   * a thermostat that settled 32 K below its setpoint under SETTLE — it was
//!     coupling to kinetic energy the constraint projection was about to delete;
//!   * a barostat that could not see the constraint virial and expanded the box
//!     without bound (density 0.997 -> 0.034 g/cm3).
//!
//! Neither was visible on the `spce_water` preset, which is two molecules in a
//! 5 nm box with the charges zeroed. Both are glaring on this one.
//!
//! Two properties make the harness worth having, and both are load-bearing:
//!
//! 1. **Coverage is enforced against the registry.** `assert_thermostat_coverage`
//!    / `assert_barostat_coverage` enumerate `Registry::with_builtins()` and
//!    panic if any registered kind has no case. A new slot cannot be added
//!    without also declaring what it is supposed to do.
//! 2. **The checks are proven to discriminate.** The negative tests in
//!    `tests/slot_conformance.rs` feed each check the *historical* value of the
//!    bug it is meant to catch (266 K; 0.034 g/cm3) and assert it panics. A
//!    tolerance loose enough to have missed the real bug fails those tests.

use heddle_md::integrator::{BarostatRegistry, ThermostatRegistry};
use heddle_md::runner::PhaseSummary;

use super::e2e::{run_case, Case, SystemBuilder};

// `PhysicsSample` is entirely in Hartree ATOMIC UNITS (see `simulation-runner.md`),
// including its `temperature` field, which carries `k_B * T` in Hartrees rather
// than kelvin. The `units = "si"` selector in the config governs the *files*, not
// this in-memory series. Convert on read.
//
// (The pre-existing `assert_mean_temperature_near` in `e2e.rs` compares
// `p.temperature` directly against a kelvin target and so is off by this factor;
// it has no callers, which is why nobody noticed.)
const K_PER_AU: f64 = 315_775.024_804_066_8;

/// Bohr^3 -> cm^3. `PhysicsSample::volume` is in Hartree atomic units.
const BOHR_CM: f64 = 5.291_772_109_03e-9;
/// Molar mass of water, g/mol.
const M_WATER: f64 = 18.015_28;
const N_AVOGADRO: f64 = 6.022_140_76e23;

/// `side` of the dense-water preset used by every case: 9^3 = 729 molecules,
/// 2187 atoms. The smallest cubic box that still clears the cell list's
/// `width >= 3 * (r_cut + r_skin)` floor at this preset's 7 A cutoff.
pub const CONFORMANCE_SIDE: usize = 9;
pub const CONFORMANCE_N_MOL: usize = CONFORMANCE_SIDE.pow(3);

pub const SETPOINT_K: f64 = 298.15;
pub const SETPOINT_PA: f64 = 1.013_25e5;

/// What a slot is required to do on the dense-water system.
#[derive(Debug, Clone, Copy)]
pub enum Expect {
    /// A thermostat must hold the phase's mean temperature within `rel_tol` of
    /// its setpoint.
    HoldsTemperature { rel_tol: f64 },
    /// A barostat must hold the phase's mean density inside `[lo, hi]`, in
    /// g/cm^3.
    HoldsDensity { lo: f64, hi: f64 },
}

/// One row of the conformance table.
#[derive(Debug, Clone, Copy)]
pub struct SlotCase {
    /// The registered `kind` string. Must match a builder in the registry.
    pub kind: &'static str,
    pub expect: Expect,
    /// Why these bounds, in one line. Read by whoever a failure lands on.
    pub note: &'static str,
}

// ---- The tables ---------------------------------------------------------

/// Every registered thermostat, and the temperature it must hold on dense
/// rigid SPC/E water under SETTLE.
///
/// `rel_tol = 0.03` is deliberately tight enough to catch the historical
/// project-before-couple defect, which settled the run at 266 K against a
/// 298.15 K setpoint — a 10.7% deficit. A tolerance that would have let that
/// through is not worth having; see the negative tests.
pub fn builtin_thermostat_cases() -> Vec<SlotCase> {
    vec![
        SlotCase {
            kind: "csvr",
            expect: Expect::HoldsTemperature { rel_tol: 0.03 },
            note: "stochastic velocity rescaling; samples the canonical distribution",
        },
        SlotCase {
            kind: "berendsen",
            expect: Expect::HoldsTemperature { rel_tol: 0.03 },
            note: "exponential relaxation to the setpoint; not canonical, but must hit the mean",
        },
        SlotCase {
            kind: "nose-hoover-chain",
            expect: Expect::HoldsTemperature { rel_tol: 0.03 },
            note: "deterministic chain; needs an equilibrated start (harness minimizes) and \
                   the default n_resp=5 RESPA sub-cycling to integrate the chain stably",
        },
        SlotCase {
            kind: "andersen",
            expect: Expect::HoldsTemperature { rel_tol: 0.03 },
            note: "stochastic collisions resample whole rigid molecules (per-group), \
                   so the projection onto the constraint manifold is a correct constrained-MB draw",
        },
    ]
}

/// Every registered barostat, and the density it must hold on dense rigid SPC/E
/// water at 1 atm and 298.15 K, thermostatted by CSVR.
///
/// The band is `[0.92, 1.04]` g/cm^3 rather than the SPC/E literature
/// 0.994-1.001, because this preset is deliberately small and short-ranged: a
/// 2.8 nm box, a 7 A cutoff, and no long-range dispersion correction together
/// pull the equilibrium density 2-3% low (measured: c-rescale 0.967, berendsen
/// 0.968, monte-carlo 0.982). The band is not the physics tolerance, it is the
/// *preset's* tolerance — and it is still two orders of magnitude tighter than
/// the failure it exists to catch, which collapsed the box to 0.034 g/cm3.
pub fn builtin_barostat_cases() -> Vec<SlotCase> {
    vec![
        SlotCase {
            kind: "c-rescale",
            expect: Expect::HoldsDensity { lo: 0.92, hi: 1.04 },
            note: "stochastic cell rescaling; reduces the virial in its own apply",
        },
        SlotCase {
            kind: "berendsen",
            expect: Expect::HoldsDensity { lo: 0.92, hi: 1.04 },
            note: "exponential box relaxation; reduces the virial in its own apply",
        },
        SlotCase {
            kind: "monte-carlo",
            expect: Expect::HoldsDensity { lo: 0.92, hi: 1.04 },
            note: "Metropolis volume moves on dU; the only barostat that never reads the virial",
        },
    ]
}

// ---- Coverage -----------------------------------------------------------

fn assert_coverage(registered: Vec<&'static str>, cases: &[SlotCase], slot: &str) {
    for kind in &registered {
        assert!(
            cases.iter().any(|c| c.kind == *kind),
            "coverage: registered {slot} `{kind}` has no conformance case. Every slot in \
             the registry must declare what it is supposed to do on dense water — add a \
             row to builtin_{slot}_cases(). (Registered: {registered:?})"
        );
    }
    // The converse: a row naming a kind nobody registered is dead weight and
    // would silently stop testing anything.
    for c in cases {
        assert!(
            registered.contains(&c.kind),
            "conformance case names {slot} `{}`, which is not registered. Registered: {registered:?}",
            c.kind
        );
    }
}

pub fn registered_thermostat_kinds() -> Vec<&'static str> {
    ThermostatRegistry::with_builtins()
        .builders()
        .iter()
        .map(|b| b.kind_name())
        .collect()
}

pub fn registered_barostat_kinds() -> Vec<&'static str> {
    BarostatRegistry::with_builtins()
        .builders()
        .iter()
        .map(|b| b.kind_name())
        .collect()
}

pub fn assert_thermostat_coverage(cases: &[SlotCase]) {
    assert_coverage(registered_thermostat_kinds(), cases, "thermostat");
}

pub fn assert_barostat_coverage(cases: &[SlotCase]) {
    assert_coverage(registered_barostat_kinds(), cases, "barostat");
}

// ---- Checks -------------------------------------------------------------
//
// Each takes a bare value as well as a phase, so the negative tests can feed a
// check the historical value of the bug it exists to catch without paying for a
// simulation. Each panics with a message that names the invariant, so
// `#[should_panic(expected = ...)]` can discriminate.

/// Mean of `f` over the samples after an initial equilibration fraction.
fn tail_mean(phase: &PhaseSummary, equil_frac: f64, f: impl Fn(&heddle_md::runner::PhysicsSample) -> f64) -> f64 {
    let n = phase.physics.len();
    assert!(n >= 4, "expected several physics samples, got {n}");
    let skip = ((n as f64) * equil_frac) as usize;
    let tail = &phase.physics[skip..];
    tail.iter().map(&f).sum::<f64>() / tail.len() as f64
}

pub fn check_no_nan(phase: &PhaseSummary, label: &str) {
    for s in &phase.physics {
        assert!(
            s.temperature.is_finite() && s.total_energy.is_finite() && s.volume.is_finite(),
            "[{label}] diverged: non-finite physics at step {} (T = {}, E = {}, V = {})",
            s.step,
            s.temperature,
            s.total_energy,
            s.volume
        );
    }
}

pub fn check_mean_temperature_value(mean: f64, target: f64, rel_tol: f64, label: &str, note: &str) {
    let err = (mean - target).abs() / target;
    assert!(
        err <= rel_tol,
        "[{label}] mean temperature {mean:.2} K is {:.1}% from the {target:.2} K setpoint \
         (tolerance {:.1}%).\n  note: {note}\n  A thermostat that cannot hold its setpoint on \
         dense rigid water is not thermostatting the physical degrees of freedom.",
        100.0 * err,
        100.0 * rel_tol
    );
}

/// Mean temperature in KELVIN over the phase's second half.
pub fn mean_temperature_k(phase: &PhaseSummary) -> f64 {
    tail_mean(phase, 0.5, |s| s.temperature) * K_PER_AU
}

pub fn check_mean_temperature(phase: &PhaseSummary, target: f64, rel_tol: f64, label: &str, note: &str) {
    check_no_nan(phase, label);
    check_mean_temperature_value(mean_temperature_k(phase), target, rel_tol, label, note);
}

pub fn check_mean_density_value(mean: f64, lo: f64, hi: f64, label: &str, note: &str) {
    assert!(
        (lo..=hi).contains(&mean),
        "[{label}] mean density {mean:.4} g/cm3 is outside [{lo}, {hi}].\n  note: {note}\n  \
         A barostat that cannot hold liquid water at its own setpoint is not reading the \
         pressure correctly — check that every virial contribution has been published before \
         the barostat reduces it."
    );
}

/// Density in g/cm^3 from the phase's mean box volume (atomic units).
pub fn mean_density(phase: &PhaseSummary, n_mol: usize) -> f64 {
    let v_bohr3 = tail_mean(phase, 0.5, |s| s.volume);
    assert!(
        v_bohr3 > 0.0,
        "phase reported a non-positive mean box volume ({v_bohr3})"
    );
    let v_cm3 = v_bohr3 * BOHR_CM.powi(3);
    let mass_g = n_mol as f64 * M_WATER / N_AVOGADRO;
    mass_g / v_cm3
}

pub fn check_mean_density(phase: &PhaseSummary, n_mol: usize, lo: f64, hi: f64, label: &str, note: &str) {
    check_no_nan(phase, label);
    check_mean_density_value(mean_density(phase, n_mol), lo, hi, label, note);
}

/// Build a `PhaseSummary` carrying exactly the given samples, for exercising the
/// checks without paying for a simulation. `temperature` is in atomic units and
/// `volume` in Bohr^3, matching what the runner actually produces.
pub fn synthetic_phase(samples: &[(f64, f64)]) -> PhaseSummary {
    PhaseSummary {
        name: "synthetic".to_string(),
        n_steps: samples.len() as u64,
        frames_written: 0,
        log_rows_written: samples.len() as u64,
        elapsed_micros: 0,
        kind: "md",
        convergence: None,
        min_final_max_force: None,
        physics: samples
            .iter()
            .enumerate()
            .map(|(i, &(t_au, v_bohr3))| heddle_md::runner::PhysicsSample {
                step: i as u64,
                time: i as f64,
                kinetic_energy: 0.0,
                potential_energy: 0.0,
                total_energy: 0.0,
                temperature: t_au,
                pressure: 0.0,
                volume: v_bohr3,
            })
            .collect(),
    }
}

/// The atomic-unit temperature corresponding to `k` kelvin. The inverse of the
/// conversion the checks apply on read.
pub fn kelvin_to_au(k: f64) -> f64 {
    k / K_PER_AU
}

/// The box volume, in Bohr^3, at which `n_mol` waters have density `rho`
/// (g/cm^3). The inverse of `mean_density`.
pub fn volume_for_density(n_mol: usize, rho: f64) -> f64 {
    let mass_g = n_mol as f64 * M_WATER / N_AVOGADRO;
    (mass_g / rho) / BOHR_CM.powi(3)
}

// ---- Runners ------------------------------------------------------------

/// Run one thermostat case: dense SPC/E water, SETTLE, constant box (NVT).
///
/// 10,000 steps at 2 fs = 20 ps, with the mean taken over the second half.
///
/// The length is set by the slowest slot, not the fastest. The preset starts from
/// a lattice, which carries a real relaxation transient, and the weakest coupling
/// in the table needs ~10 ps to shed it. A shorter run would fail a working
/// thermostat for not having equilibrated yet — a test that fails for the wrong
/// reason is worse than no test.
/// The MD phase of a run that may be preceded by a minimization phase. The
/// conformance runs minimize first (see `run_thermostat_case`), so the physics
/// series lives in the last phase, not `phases[0]`.
fn md_phase(summary: &heddle_md::runner::RunSummary) -> &PhaseSummary {
    summary
        .phases
        .iter()
        .rev()
        .find(|p| !p.physics.is_empty())
        .expect("no MD phase with a physics series")
}

pub fn run_thermostat_case(case: &SlotCase) {
    let Expect::HoldsTemperature { rel_tol } = case.expect else {
        panic!("thermostat case `{}` must declare HoldsTemperature", case.kind);
    };
    let dir = Case::new(&format!("conf_therm_{}", case.kind));
    let cfg = SystemBuilder::dense_spce_water(CONFORMANCE_SIDE)
        .constraints("settle")
        .minimize(true)
        .thermostat(case.kind, SETPOINT_K)
        .n_steps(10_000)
        .log_every(100)
        .trajectory_every(0)
        .write(&dir);
    let summary = run_case(&cfg);
    check_mean_temperature(md_phase(&summary), SETPOINT_K, rel_tol, case.kind, case.note);
}

/// Run one barostat case: dense SPC/E water, SETTLE, CSVR, NPT at 1 atm.
///
/// 25,000 steps at 2 fs = 50 ps. The box needs ~20 ps to settle from the
/// lattice initial state; the mean is taken over the second half.
///
/// The thermostat is CSVR, deliberately: it is the one whose conformance case
/// passes, so a failure here is attributable to the barostat.
pub fn run_barostat_case(case: &SlotCase) {
    let Expect::HoldsDensity { lo, hi } = case.expect else {
        panic!("barostat case `{}` must declare HoldsDensity", case.kind);
    };
    let dir = Case::new(&format!("conf_baro_{}", case.kind));
    let cfg = SystemBuilder::dense_spce_water(CONFORMANCE_SIDE)
        .constraints("settle")
        .minimize(true)
        .thermostat("csvr", SETPOINT_K)
        .barostat(case.kind, SETPOINT_PA)
        .n_steps(25_000)
        .log_every(250)
        .trajectory_every(0)
        .write(&dir);
    let summary = run_case(&cfg);
    check_mean_density(
        md_phase(&summary),
        CONFORMANCE_N_MOL,
        lo,
        hi,
        case.kind,
        case.note,
    );
}
