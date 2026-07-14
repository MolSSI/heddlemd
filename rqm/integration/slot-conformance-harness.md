# Feature: Slot Conformance Harness <!-- rq-c7f4d96b -->

The slot conformance harness runs a battery of physical invariants against
**every registered thermostat and barostat**, on a system where those invariants
are observable: dense, rigid SPC/E water.

It is the ensemble-slot counterpart to the potential consistency harness
(`rqm/forces/potential-consistency-harness.md`), and it answers the same two
questions that a per-slot test file cannot:

- **Is every registered slot covered?** A slot is selectable by any user the
  moment it is in the registry. The harness derives its ground truth from the
  registry itself, so a slot cannot be registered without also declaring what it
  is required to do.
- **Is it exercised where a defect is visible?** A thermostat controls the mean
  temperature and a barostat controls the mean density. Both quantities are
  nearly inert in a dilute system: in a box that is mostly vacuum, the constraint
  virial is negligible against the force virial, and the kinetic energy that
  rigid constraints remove is a small share of the total. A slot that mishandles
  either produces the same trajectory as one that does not. The harness therefore
  runs on a liquid, where the two contributions are the same order as the terms
  they correct.

The harness lives in `tests/common/slot_conformance.rs`; the tests that drive it
are `tests/slot_conformance.rs`.

## The Preset <!-- rq-9254d227 -->

`SystemBuilder::dense_spce_water(side)` builds `side³` rigid SPC/E water
molecules on a simple-cubic lattice at **0.997 g/cm³**, each carrying a
deterministic pseudo-random orientation, with the SPC/E charges and SPME
electrostatics.

The orientations are load-bearing. An aligned lattice of water dipoles is
ferroelectric: it carries a large net polarisation and a pressure unlike liquid
water's. The orientations are drawn from a seeded LCG rather than from the run's
RNG, so the generated input files are byte-identical on every call and the
reproducibility assertions over this preset remain meaningful.

`side = 9` (729 molecules, 2187 atoms) is the size the conformance cases use. It
is the smallest cubic box that satisfies the cell list's
`width ≥ 3 · (r_cut + r_skin)` requirement at the preset's 7 Å cutoff:
`L = 2.797 nm` against a 2.55 nm floor.

The preset is deliberately small and short-ranged, and that has a physical
consequence the tolerances absorb: a 2.8 nm box, a 7 Å cutoff, and the absence of
a long-range dispersion correction together place the equilibrium density 2–3%
below the SPC/E value of 0.994–1.001 g/cm³. The density band the harness asserts
is a property of the preset, not of the force field.

## Cases <!-- rq-63e9ef08 -->

The harness is table-driven, one `SlotCase` per registered kind:

- **Thermostat cases** run in NVT — the preset at a constant box, under SETTLE —
  and require the slot to hold the mean temperature within a relative tolerance
  of its setpoint.
- **Barostat cases** run in NPT — the preset under SETTLE at 1 atm, thermostatted
  by CSVR — and require the slot to hold the mean density inside a band.

Barostat cases pair with CSVR specifically because CSVR's own conformance case
passes, which makes a failing barostat case attributable to the barostat.

Means are taken over the second half of each run. The preset starts from a
lattice and carries a relaxation transient, and the run length is set by the
slowest-coupling slot in the table rather than the fastest: a case that fails a
conforming slot for not having equilibrated yet reports a defect that does not
exist.

## Coverage <!-- rq-af8c5af7 -->

Coverage is derived from the registry, not from the table. `assert_coverage`
takes the registered kinds from `Registry::with_builtins().builders()` via
`KindedBuilder::kind_name()` and asserts in both directions:

- every registered kind has a case — a slot that no case exercises is a slot that
  a user can select and nothing checks;
- every case names a registered kind — a case for a kind that is not registered
  silently exercises nothing.

## Conformance Is Not Waivable <!-- rq-12d901cf -->

No slot is exempt from its case. A registered slot that cannot meet its
requirement is a defect in that slot, and the harness reports it as a failing
test until it is fixed. The harness carries no mechanism for marking a case as
expected-to-fail, deliberately: a waived case rots, because nothing signals that
it should be tightened once the underlying defect is repaired, and a suite that
reports a non-conforming slot as passing is the failure mode the harness exists
to prevent.

## Discriminating Power <!-- rq-cc3206c5 -->

A conformance suite is worth exactly its tolerances, so the tolerances are
themselves under test. Each check has a paired negative test that feeds it a
value characteristic of the class of defect the check exists to catch and asserts
that the check rejects it, and a positive test that feeds it a value
characteristic of a conforming slot and asserts that the check accepts it. A
tolerance loosened far enough to admit a real defect, or tightened far enough to
reject a conforming slot, turns one of those tests red.

The checks accept a bare measured value as well as a `PhaseSummary`, so the
negative tests cost no simulation time.

## Units <!-- rq-8c5337af -->

`PhysicsSample` is expressed entirely in Hartree atomic units
(`rqm/simulation-runner.md`). Its `temperature` field carries `k_B · T` in
Hartrees rather than kelvin, and its `volume` is in Bohr³. The `units` selector
in a configuration governs the unit system of the **files**, not of this
in-memory series. The harness converts on read: temperatures to kelvin,
volumes to a density in g/cm³.

## Feature API <!-- rq-2aea3450 -->

### Types <!-- rq-4ef62aca -->

- `Expect` — what a slot is required to do on the dense-water system. <!-- rq-bf4b0145 -->
  - `HoldsTemperature { rel_tol: f64 }` — the phase's mean temperature must be
    within `rel_tol` (relative) of the thermostat's setpoint.
  - `HoldsDensity { lo: f64, hi: f64 }` — the phase's mean density, in g/cm³,
    must lie in `[lo, hi]`.

- `SlotCase` — one row of a conformance table. <!-- rq-04d04426 -->
  - `kind: &'static str` — the registered `kind` string. Must name a builder in
    the corresponding registry.
  - `expect: Expect` — the requirement. A thermostat case carries
    `HoldsTemperature`; a barostat case carries `HoldsDensity`. A case whose
    `Expect` variant does not match its slot type is a programming error and
    panics when run.
  - `note: &'static str` — why these bounds, in one line. Reproduced in the
    failure message.

### Functions <!-- rq-85c252a7 -->

- `SystemBuilder::dense_spce_water(side: usize) -> SystemBuilder` <!-- rq-6028de4d -->
  - Selects the dense-water preset (`rqm/e2e-testing.md`, *Test Harness*).
  - Writes `side³` rigid SPC/E molecules at 0.997 g/cm³ with deterministic
    pseudo-random orientations, the SPC/E charges, SPME, and a SETTLE-compatible
    constraint topology.
  - Panics if `side` yields a box narrower than `3 · (r_cut + r_skin)`, rather
    than deferring to a cell-list build failure at run time.

- `builtin_thermostat_cases() -> Vec<SlotCase>` — the conformance table for <!-- rq-99fc9152 -->
  thermostats. One row per registered thermostat kind, each carrying
  `Expect::HoldsTemperature`.

- `builtin_barostat_cases() -> Vec<SlotCase>` — the conformance table for <!-- rq-70cf9aa8 -->
  barostats. One row per registered barostat kind, each carrying
  `Expect::HoldsDensity`.

- `registered_thermostat_kinds() -> Vec<&'static str>` and <!-- rq-69354a0e -->
  `registered_barostat_kinds() -> Vec<&'static str>` — the `kind_name()` of every
  builder in the corresponding `Registry::with_builtins()`. This is the coverage
  ground truth.

- `assert_thermostat_coverage(cases: &[SlotCase])` and <!-- rq-550acba3 -->
  `assert_barostat_coverage(cases: &[SlotCase])`
  - Panic if a registered kind has no case, naming the kind and the registered
    set.
  - Panic if a case names a kind that is not registered.

- `check_no_nan(phase: &PhaseSummary, label: &str)` — panics if any physics sample <!-- rq-6bfa82f7 -->
  carries a non-finite temperature, total energy, or volume, naming the step. A
  diverging slot fails here rather than producing a meaningless mean.

- `mean_temperature_k(phase: &PhaseSummary) -> f64` — the mean of the phase's <!-- rq-c3b0c28f -->
  temperature samples over its second half, converted from atomic units to
  kelvin.

- `mean_density(phase: &PhaseSummary, n_mol: usize) -> f64` — the mean of the <!-- rq-a828920c -->
  phase's box-volume samples over its second half, converted from Bohr³ to a
  water density in g/cm³. Panics on a non-positive mean volume.

- `check_mean_temperature_value(mean: f64, target: f64, rel_tol: f64, label: &str, note: &str)` <!-- rq-9fe2a0cf -->
  - Panics, with a message containing `"mean temperature"`, unless
    `|mean − target| / target ≤ rel_tol`.
  - Takes a bare value so a negative test can exercise the tolerance without
    running a simulation.

- `check_mean_density_value(mean: f64, lo: f64, hi: f64, label: &str, note: &str)` <!-- rq-d00435b1 -->
  - Panics, with a message containing `"mean density"`, unless
    `lo ≤ mean ≤ hi`.

- `check_mean_temperature(phase: &PhaseSummary, target: f64, rel_tol: f64, label: &str, note: &str)` <!-- rq-5a708d61 -->
  — `check_no_nan`, then `check_mean_temperature_value` on `mean_temperature_k`.

- `check_mean_density(phase: &PhaseSummary, n_mol: usize, lo: f64, hi: f64, label: &str, note: &str)` <!-- rq-ed7c24da -->
  — `check_no_nan`, then `check_mean_density_value` on `mean_density`.

- `run_thermostat_case(case: &SlotCase)` — writes and runs the dense-water preset <!-- rq-6e2b074e -->
  under SETTLE at a constant box with `case.kind` installed as the thermostat,
  then applies `check_mean_temperature`. Panics if `case.expect` is not
  `HoldsTemperature`.

- `run_barostat_case(case: &SlotCase)` — writes and runs the dense-water preset <!-- rq-14b1bb15 -->
  under SETTLE with a CSVR thermostat and `case.kind` installed as the barostat
  at 1 atm, then applies `check_mean_density`. Panics if `case.expect` is not
  `HoldsDensity`.

## Out of Scope <!-- rq-1a32807d -->

- **Integrators and constraints.** The registry enumeration and coverage
  machinery generalise, but the invariants do not: an integrator is characterised
  by energy conservation (`rqm/e2e-testing.md`) and a constraint by its manifold
  residual, neither of which is a controlled ensemble variable.
- **The thermostat × barostat cross product.** Every barostat case runs against
  CSVR. Other combinations exercise the same `run_step` ordering, at a
  multiplicative cost in runtime.
- **Ensemble distribution shape.** The harness asserts the mean of the controlled
  variable, not that its fluctuations follow the correct distribution. A
  thermostat that holds the correct mean temperature with an incorrect
  kinetic-energy variance conforms.
- **An unconstrained control arm.** Every case runs under SETTLE. Separating "a
  slot that fails only under constraints" from "a slot that fails everywhere" is
  the business of the per-slot unit tests.

## Gherkin Scenarios <!-- rq-601f9a1f -->

```gherkin
Feature: Slot conformance on dense liquid water

  Background:
    Given the dense_spce_water preset with side = 9, i.e. 729 rigid SPC/E molecules at 0.997 g/cm^3
    And SETTLE constraints over the preset's rigid-geometry metadata

  # --- Coverage ---

  @rq-17119008
  Scenario: Every registered thermostat has a conformance case
    Given the kind names of the thermostat registry's built-in builders
    When they are compared against builtin_thermostat_cases()
    Then every registered kind has a case
    And every case names a registered kind

  @rq-6c125bac
  Scenario: Every registered barostat has a conformance case
    Given the kind names of the barostat registry's built-in builders
    When they are compared against builtin_barostat_cases()
    Then every registered kind has a case
    And every case names a registered kind

  @rq-a22870b8
  Scenario: A registered thermostat with no case fails coverage
    Given builtin_thermostat_cases() with one row removed
    When assert_thermostat_coverage is called
    Then it panics with a message naming the uncovered kind

  @rq-f797fd2c
  Scenario: A registered barostat with no case fails coverage
    Given builtin_barostat_cases() with one row removed
    When assert_barostat_coverage is called
    Then it panics with a message naming the uncovered kind

  @rq-754c7ddb
  Scenario: A case for an unregistered kind fails coverage
    Given builtin_thermostat_cases() with a row naming a kind that is not registered
    When assert_thermostat_coverage is called
    Then it panics with a message reporting that the kind is not registered

  # --- The preset ---

  @rq-5f3cf6c9
  Scenario: The preset refuses a box narrower than the cell list allows
    Given a side small enough that L < 3 * (r_cut + r_skin)
    When SystemBuilder::dense_spce_water(side) is written
    Then it panics naming the box width and the cell-list minimum

  @rq-805728c8
  Scenario: The preset's input files are byte-identical across calls
    Given two Cases
    When dense_spce_water(9) is written into each
    Then the two sim.in.xyz files are byte-identical
    And the two sim.in.topology files are byte-identical

  # --- Conformance ---

  @rq-eb7c5c08
  Scenario: A thermostat holds its setpoint
    Given a registered thermostat at a 298.15 K setpoint and a constant box
    When the conformance case is run
    Then the mean temperature over the run's second half is within its rel_tol of the setpoint

  @rq-716752a7
  Scenario: A barostat holds liquid density
    Given a registered barostat at 1 atm and a CSVR thermostat at 298.15 K
    When the conformance case is run
    Then the mean density over the run's second half lies within its [lo, hi] band

  @rq-1987903e
  Scenario: A diverging slot is reported as divergence, not as a bad mean
    Given a phase whose physics samples contain a non-finite temperature
    When check_no_nan is called
    Then it panics naming the step at which the sample went non-finite

  # --- Discriminating power ---

  @rq-4240aedb
  Scenario: The temperature check rejects a mean far from the setpoint
    Given a measured mean temperature 10% below a 298.15 K setpoint
    When check_mean_temperature_value is called with rel_tol = 0.03
    Then it panics with a message containing "mean temperature"

  @rq-8c06ce9c
  Scenario: The temperature check accepts a conforming mean
    Given a measured mean temperature within 1% of a 298.15 K setpoint
    When check_mean_temperature_value is called with rel_tol = 0.03
    Then it returns without panicking

  @rq-6d2c1446
  Scenario: The density check rejects a collapsed box
    Given a measured mean density an order of magnitude below liquid water's
    When check_mean_density_value is called with the conformance band
    Then it panics with a message containing "mean density"

  @rq-1eb2642b
  Scenario: The density check accepts a conforming density
    Given a measured mean density inside the conformance band
    When check_mean_density_value is called with that band
    Then it returns without panicking

  # --- Units ---

  @rq-610c023c
  Scenario: Temperature is converted out of atomic units
    Given a phase whose PhysicsSample temperature samples carry k_B * T in Hartrees
    When mean_temperature_k is called
    Then the result is that mean in kelvin

  @rq-efb5ebb3
  Scenario: Density is derived from the box volume in Bohr^3
    Given a phase whose PhysicsSample volume samples are in Bohr^3
    When mean_density is called with the preset's molecule count
    Then the result is the water density in g/cm^3

  @rq-c0c0f112
  Scenario: A non-positive mean volume is an error, not a division
    Given a phase whose mean box volume is zero
    When mean_density is called
    Then it panics naming the non-positive volume
```
