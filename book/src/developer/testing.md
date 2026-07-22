# Testing extension components

HeddleMD tests an extension at several levels. A focused test can establish
that one equation or kernel is correct, while a shared harness establishes
that every registered implementation satisfies the same physical contract.
Both are necessary: a component can pass an isolated test and still interact
incorrectly with constraints, force aggregation, or the timestep runner.

This page introduces the two principal shared harnesses:

- the **potential-consistency harness** for pair and bonded potentials;
- the **slot-conformance harness** for thermostats and barostats.

It also explains how these harnesses fit with the feature-specific tests found
under `tests/`.

## Test layers

A new extension should normally be tested at four layers:

| Layer | Main question | Typical location |
| --- | --- | --- |
| Configuration | Are parameters parsed, converted, and validated correctly? | `tests/io_config.rs`, `tests/unit_conversion.rs`, or the feature test file |
| Component | Does the builder construct the component, and do its direct methods handle normal and empty inputs? | A feature file such as `tests/barostats_berendsen.rs` |
| Shared physical contract | Does the component satisfy invariants common to every component of its kind? | `tests/potential_consistency.rs` or `tests/slot_conformance.rs` |
| End to end | Does a complete simulation compose the component correctly and remain deterministic? | The feature test file or `tests/determinism.rs` |

Do not use a long simulation to replace a small analytical test. A trajectory
may reveal that something is wrong without identifying whether the cause is a
sign error, unit conversion, reduction, placement, or random-number sequence.
Conversely, do not rely only on direct kernel tests: they bypass the registry,
runner, and other components with which the extension must compose.

## Running the tests

The shared positive-path harnesses execute the real GPU pipeline and therefore
require a supported NVIDIA GPU and CUDA environment. Run one integration-test
target with Cargo:

```text
cargo test --test potential_consistency
cargo test --test slot_conformance
cargo test --test barostats_berendsen
```

To run one named test while developing, add its name as a filter:

```text
cargo test --test potential_consistency correct_pair_passes_finite_difference
```

Some helper and negative-control tests are CPU-only, but Cargo still builds the
test target and its GPU-facing code. Treat a missing GPU or CUDA installation
as an environment failure, not as evidence about the physical implementation.

## Potential-consistency harness

The implementation lives in `tests/common/consistency.rs`, and
`tests/potential_consistency.rs` drives its positive and negative scenarios.
Its detailed requirements are recorded in
`rqm/forces/potential-consistency-harness.md`.

The harness evaluates one isolated potential at a series of geometries. For
each geometry it obtains the total energy, force on every atom, and scalar
virial from the real force-field pipeline. It then checks invariants that do
not depend on the potential's particular functional form:

- **Force–energy consistency:** a central finite difference of the energy
  agrees with each Cartesian force component (`F = -grad U`).
- **Newton's third law:** the forces sum to zero for the isolated interaction.
- **Virial consistency:** the reported scalar virial agrees with the energy
  change under an infinitesimal isotropic coordinate scaling.
- **Reference values:** optional analytical energies or generalized forces are
  reproduced at selected coordinates.
- **Cutoff behavior for pair potentials:** the switched interaction joins
  smoothly and vanishes at its cutoff. This check does not apply to fragments
  explicitly declared unbounded.

These checks catch common CUDA-fragment errors such as reversing a force sign,
omitting the division by distance, splitting an energy twice, producing
asymmetric forces, or reporting an inconsistent virial.

### The fixture and evaluator

A `ConsistencyFixture` describes what the harness should evaluate. Its most
important fields are:

- `label`, used in failure messages and coverage checks;
- `shape`, one of `Pair`, `Bond`, `Angle`, or `Dihedral`;
- `samples`, the distances, angles, or dihedrals at which to test;
- a geometry function that turns each sample coordinate into atom positions;
- a builder for an isolated `ForceField` containing the potential;
- tolerances and finite-difference step sizes;
- optional analytical `ReferencePoint` values;
- switching and cutoff information for pair potentials.

The shape constructors—`ConsistencyFixture::pair`, `bond`, `angle`, and
`dihedral`—supply the standard geometry and topology for their interaction
type. `pair_correction` supports an unbounded correction-only pair fragment.

`GpuEvaluator` is the bridge to the production code. For every requested
geometry it creates fresh `ParticleBuffers`, runs
`ForceField::step(..., AggregateLevel::ForcesAndScalars)`, copies the forces,
per-particle energies, and virials back to the host, and returns their totals
as an `Eval`.

The checking functions are generic over the `Evaluator` trait. Negative tests
can therefore use a small CPU evaluator containing a deliberate defect. This
proves that the check fails for the mistake it is intended to detect without
adding a broken CUDA kernel to the project.

### Adding a pair-potential fixture

Add the new fixture to `builtin_consistency_fixtures()` in
`tests/common/consistency.rs`. A pair fixture has the following conceptual
form; the exact configuration types should match the new potential:

```rust
ConsistencyFixture::pair(
    "buckingham",
    |registry| registry.register(Box::new(BuckinghamBuilder)),
    particle_types,
    pair_interactions,
    charges,
    None,                 // optional SPME configuration
    cutoff,
    r_switch,
    vec![r1, r2, r3],     // informative, non-singular sample distances
)
.with_reference_points(vec![
    ReferencePoint {
        coordinate: r_ref,
        energy: Some(expected_energy),
        coord_force: Some(expected_radial_force),
        tol: Tolerance::new(relative_tolerance, absolute_tolerance),
    },
])
```

Choose samples that exercise the repulsive and attractive regions and, when
applicable, the switched region. Avoid singular points and configurations
whose expected force is so close to zero that a sign or scale error becomes
invisible. Reference points are especially useful because finite-difference
consistency alone cannot detect an energy and force that are both wrong by the
same constant factor.

The fixture should construct a registry containing only the builder under
test. This isolates its contribution from other potentials. The fixture
constructors already provide a sufficiently large box and an all-pairs
neighbor list for the small test system.

### Coverage is enforced

`assert_fixture_coverage` compares fixture labels with the fragment labels
reported by the built-in potential registry. Adding a built-in fragment
without a fixture therefore fails the coverage test. A fixture whose label no
longer corresponds to a registered fragment also fails.

`assert_all_builtin_potentials_consistent` runs the complete invariant suite
over every built-in fixture. During development, an individual test such as
`check_force_energy` can provide a narrower failure, but the complete sweep is
the final shared-contract check.

The consistency harness does not replace tests for parameter claims, unit
conversion, argument-schema strings, fragment metadata, exclusions, mixed
particle types, or an empty interaction list. Keep those focused tests in the
feature file and in `tests/potential_claims.rs` or
`tests/forces_framework.rs` as appropriate.

## Barostat tests

Barostats have two complementary test styles:

1. Direct feature tests exercise construction, reductions, scale factors,
   box mutation, random counters, and empty-system behavior.
2. The shared slot-conformance harness checks whether every registered
   barostat maintains a physically meaningful density in a complete
   constrained-water simulation.

The direct tests are grouped by implementation:

- `tests/barostats_berendsen.rs` for deterministic per-step coupling;
- `tests/barostats_c_rescale.rs` for stochastic per-step coupling;
- `tests/barostats_mc.rs` for periodic Monte Carlo volume moves.

### Direct per-step tests

A typical direct fixture constructs:

- a `GpuContext` from `init_device()`;
- a `SlotConfig` from a `kind` and TOML parameter string;
- a barostat through `BarostatRegistry::with_builtins().build_optional(...)`;
- a small `ParticleState` with prescribed positions, velocities, masses, and
  per-particle virials;
- `ParticleBuffers`, a `SimulationBox`, and `Timings`.

Call the public `Barostat` trait method when testing behavior shared by the
framework. Tests may inspect a concrete built-in type when verifying
method-specific diagnostics, but prefer safe public accessors where available.
An unsafe cast from `Box<dyn Barostat>` to a concrete type is tightly coupled
to the registry roster and should not be copied into ordinary downstream code.

For a per-step barostat, focused tests should cover:

- registry discovery and construction, including zero particles;
- parameter validation and conversion;
- the deterministic kinetic-energy and virial reductions;
- the analytical scale factor for controlled kinetic energy, virial, box volume, and
  timestep values;
- changes to the box, positions, and box-generation counter;
- no changes to velocities unless the method explicitly requires them;
- no kernel launch or box mutation for an empty particle set;
- diagnostic and conserved-quantity bookkeeping;
- identical results from identical seeds and device-counter states for a
  stochastic method.

Tests should populate the per-particle virials deliberately. Calling
`barostat.apply` with an arbitrary zeroed buffer does not verify whether the
barostat consumes the pressure information produced by the force pipeline.
At least one integration test should run a real force evaluation with
`AggregateLevel::ForcesAndScalars` before applying the barostat.

### Direct periodic-move tests

A periodic barostat is host-orchestrated and uses `apply_move` rather than the
per-step `apply` hook. Its fixture also needs a mutable `ForceField`, molecule
tables, and a topology-compatible state.

In addition to configuration and construction, test:

- the declared `BarostatPeriodicity` and move frequency;
- the proposal counter advancing once per attempted move;
- acceptance and rejection paths;
- exact restoration of positions, forces, and lattice after rejection;
- no velocity changes;
- rigid intramolecular geometry under molecular center-of-mass scaling;
- cutoff/box-width guards;
- deterministic decisions for identical seeds and counters;
- graph-mode and ordinary-launch behavior at host batch boundaries.

## Slot-conformance harness for barostats

The shared implementation is in `tests/common/slot_conformance.rs`, and
`tests/slot_conformance.rs` contains the test sweep. The harness runs each
built-in barostat on dense, rigid SPC/E water with SETTLE constraints and a
CSVR thermostat. It then checks that the mean density over the equilibrated
part of the phase lies within the case's declared interval.

The system is intentionally more demanding than a minimal smoke test. It is
dense enough for pressure and constraint-virial errors to be visible. The
barostat run uses a thermostat that has its own conformance case, helping to
attribute a density failure to the barostat rather than uncontrolled thermal
dynamics.

Each built-in barostat has a `SlotCase` in `builtin_barostat_cases()`:

```rust
SlotCase {
    kind: "my-barostat",
    expect: Expect::HoldsDensity { lo, hi },
    note: "brief explanation of the method and the chosen bounds",
}
```

To add a built-in barostat to the harness:

1. Add a `SlotCase` with a scientifically justified density interval and a
   note explaining it.
2. Add a named `#[test]` in `tests/slot_conformance.rs` that finds the case and
   passes it to `run_barostat_case`.
3. Update the barostat sweep count checked by
   `every_case_in_the_tables_is_driven_by_a_test`.
4. Run the new test and the complete slot-conformance target.

`assert_barostat_coverage` compares the case table with
`BarostatRegistry::with_builtins()`. A registered barostat without a case, or
a stale case without a registered builder, fails immediately. The separate
sweep-count test ensures that adding a table row is not enough: a named test
must actually execute it.

The density bounds are regression and conformance bounds for the harness's
finite system, cutoff, and run length. They are not a claim that every valid
barostat must reproduce an experimental bulk density within the same narrow
range under arbitrary settings. Record the reason for the interval in the
case's `note` so future changes do not widen it without understanding what
defect it was designed to catch.

The harness also contains negative controls that feed historically incorrect
temperature or density values directly to its checking functions and assert
that they fail. These tests demonstrate that the tolerances can discriminate
the targeted defect; they do not require a GPU simulation.

## Choosing tolerances

A useful tolerance is wide enough to accommodate the known numerical noise of
the tested precision and finite system, but narrow enough to fail for a
plausible implementation error.

Use these principles:

- Base analytical-test tolerances on the precision and reduction path, not on
  the size of the observed error after the test fails.
- Include an absolute floor when the expected value can approach zero.
- Use several sample coordinates so a cancellation at one point cannot hide a
  defect.
- Give every broad statistical interval a physical and finite-size rationale.
- Add a negative control when a tolerance protects against a known class of
  error. A check that has never been shown to reject a representative defect
  may provide false confidence.

## Checklist for a new extension

- [ ] Configuration parsing, domain validation, unknown fields, and unit
      conversion are tested.
- [ ] Registry construction and an empty particle or interaction set are
      tested.
- [ ] Small analytical tests isolate the method's equations and kernel
      outputs.
- [ ] The appropriate shared harness contains a fixture or conformance case.
- [ ] Registry-to-harness coverage checks pass.
- [ ] Tolerances are justified and discriminate meaningful defects.
- [ ] Stochastic tests control both the seed and counter state.
- [ ] At least one end-to-end test exercises composition with the runner and
      relevant constraints, thermostat, or force field.
- [ ] Deterministic repeated runs are compared where the extension changes
      per-step numerical work.
