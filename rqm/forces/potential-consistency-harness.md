# Feature: Potential Consistency Harness <!-- rq-ec1452c6 -->

Every force-field potential is defined by a CUDA source fragment that must
satisfy relationships the fragment's author writes by hand: the force must be
the negative gradient of the energy, the forces must sum to zero on an isolated
cluster, the reported scalar virial must match the force, and a switched pair
interaction must join smoothly and vanish at its cutoff. A fragment that
violates one of these compiles cleanly and produces a subtly wrong trajectory
rather than an error.

The potential consistency harness verifies these relationships numerically for
each potential by driving the real force pipeline on a minimal isolated system
and reading back per-particle forces, potential energy, and scalar virial. A
registry of canonical fixtures — one per built-in potential — lets a single
test sweep every built-in, so a potential added without its own consistency
fixture fails the sweep rather than going untested.

The harness targets the fragment-composed potentials: the fast-class pair-force
slots (Lennard-Jones, SPME real-space, and any additional pairwise van-der-Waals
form) and the intramolecular slots (bond, angle, dihedral). Each is exercised
through `ForceField::step` at `AggregateLevel::ForcesAndScalars` on the GPU, the
same evaluation path a production run uses. The harness lives in the shared test
module `tests/common/`; the built-in sweep is a test in `tests/potential_consistency.rs`.

This harness checks a potential against *itself and against known analytic
points*; it is complementary to the end-to-end layer (`e2e-testing.md`), which
checks whole-simulation invariants, and to the SPME-versus-Ewald reference
comparison, which checks the reciprocal-space pipeline the fragment model does
not cover.

## Minimal Systems <!-- rq-19c1c994 -->

Each fixture builds an isolated system carrying exactly one active potential
slot — a `ForceField` constructed from a registry holding only the potential
under test, so the read-back energy, forces, and virial are that potential's
contribution alone. Every system is placed in a box wide enough that only the
primary image interacts (all separations below the cutoff and below half the
minimum box width) and carries no exclusions unless the fixture declares them.

- **Pair** — two atoms separated by a distance `r` along one axis. The
  coordinate is `r`.
- **Bond** — two atoms joined by one bond of the type under test. The
  coordinate is the bond length `r`.
- **Angle** — three atoms `i–j–k` forming one angle of the type under test,
  with fixed leg lengths. The coordinate is the angle `θ`.
- **Dihedral** — four atoms `i–j–k–l` forming one dihedral of the type under
  test, with fixed bond lengths and bend angles. The coordinate is the torsion
  `φ`.

A fixture supplies a set of **sample coordinate values** spanning the
interaction's meaningful range. For a pair fixture the samples include points
inside the switching region `[r_switch, cutoff]` and just beyond the cutoff.

## Invariants <!-- rq-b9fde79c -->

At each sample geometry, the harness evaluates the total potential energy `U`,
the per-particle forces `F_a`, and the total scalar virial `W` of the isolated
system by one force-and-scalars evaluation, and asserts:

- **Force–energy finite-difference consistency.** For every atom `a` and every
  Cartesian axis `d`, the central finite difference of the total energy with
  respect to that coordinate equals the negative of the corresponding force
  component:
  `−(U(x_{a,d}+h) − U(x_{a,d}−h)) / 2h ≈ F_{a,d}`.
  This is `F = −∇U` evaluated component-by-component; it is shape-uniform and
  requires no coordinate-gradient bookkeeping. It catches a missing `1/r`
  factor, a sign error, and an energy or force scaled by the wrong constant.

- **Newton's third law.** The vector sum of the per-particle forces on the
  isolated cluster is zero: `Σ_a F_a ≈ 0`. For the two-atom pair and bond
  systems this is equivalent to `F_i = −F_j`. It catches asymmetric force
  assignment in the fragment.

- **Virial–force consistency.** The reported total scalar virial equals the
  value obtained by central finite difference of the energy under isotropic
  scaling of the isolated system about its centroid:
  `W ≈ −dU/d(ln λ)|_{λ=1}`, where `U(λ)` is the energy with every position
  scaled by `λ`. For a central pair or bond force this equals `Σ factor·r²` over
  the interacting pairs; for a pure angle or dihedral force, whose energy is
  invariant under isotropic scaling, it is zero. It catches a fragment whose
  reported virial does not correspond to the force it applies (the failure mode
  the composer-derived-virial simplification is designed to remove).

- **Switch/cutoff continuity** (pair only). Sampling the pair energy and force
  across `[r_switch − δ, cutoff + δ]`, the energy and force are continuous
  across `r_switch` and across the cutoff (no jump beyond tolerance), and both
  are exactly zero for `r > cutoff`. It catches a discontinuous or
  non-vanishing switching function.

In addition, a fixture may declare **analytic reference points**: coordinate
values with an expected total energy and/or an expected coordinate-conjugate
force `−dU/dq`. The harness evaluates the potential at each and asserts the
match. Reference points catch a functional form that is uniformly wrong yet
internally self-consistent — which the finite-difference check alone cannot,
since a wrong `U` and its matching wrong `F` satisfy `F = −∇U`.

## Fixture Registry and Coverage <!-- rq-6dbbbccc -->

`builtin_consistency_fixtures()` returns one fixture per built-in
fragment-composed potential. The built-in sweep
`assert_all_builtin_potentials_consistent` runs every fixture and additionally
asserts **coverage**: the set of potential labels the fixtures target includes
the `Potential::label()` of every fast-class fragment-composed slot that
`PotentialRegistry::with_builtins()` can produce. A built-in potential without a
matching fixture fails this coverage assertion, so a newly added potential
cannot be silently omitted from the harness.

## Tolerances and f32 <!-- rq-61232d64 -->

The engine stores and computes forces, energies, and the fixed-point force
accumulation in single precision, so the harness's tolerances accommodate f32
round-off and the finite-difference truncation error. Each fixture carries
per-invariant `Tolerance` values with defaults tuned for the f32 pipeline and a
finite-difference step chosen to balance truncation against f32 cancellation;
fixtures and individual reference points may override them. Tolerances are
relative with an absolute floor, so a check against a near-zero expected value
(a force at an energy minimum, a virial for an angle term) does not demand
impossible relative precision.

The harness runs on the GPU through the real force pipeline and requires a
device, exactly as the per-potential kernel tests do. It performs no I/O and is
independent of the reproducibility guarantee (it is a correctness check on a
single evaluation, not a run-to-run comparison).

## Feature API <!-- rq-4a1699c4 -->

### Types <!-- rq-e67ef8c3 -->

- `PotentialShape` — the four fragment-composed shapes the harness drives. <!-- rq-45ceac42 -->

  ```rust
  pub enum PotentialShape { Pair, Bond, Angle, Dihedral }
  ```

- `Tolerance` — a relative tolerance with an absolute floor. `Default` supplies <!-- rq-7c83a420 -->
  values tuned for the f32 pipeline.

  ```rust
  pub struct Tolerance { pub rel: f64, pub abs: f64 }
  ```

- `ReferencePoint` — an analytic check at one coordinate value. <!-- rq-96809580 -->

  ```rust
  pub struct ReferencePoint {
      pub coordinate: f64,          // r, θ, or φ per the fixture's shape
      pub energy: Option<f64>,      // expected total potential energy, if checked
      pub coord_force: Option<f64>, // expected generalized force −dU/dq, if checked
      pub tol: Tolerance,
  }
  ```

- `ConsistencyFixture` — everything needed to drive one potential. It carries <!-- rq-e8b786b3 -->
  the target `Potential::label()`, the shape, a builder that constructs the
  isolated single-slot system at a given coordinate value, the sample coordinate
  values, the reference points, per-invariant tolerances, and (for a pair
  fixture) the `r_switch` and `cutoff` bounding the continuity sweep. Shape
  constructors build the common cases:
  - `ConsistencyFixture::pair(label, entry, samples) -> Self`
  - `ConsistencyFixture::bond(label, entry, samples) -> Self`
  - `ConsistencyFixture::angle(label, entry, samples) -> Self`
  - `ConsistencyFixture::dihedral(label, entry, samples) -> Self`
  - `.with_reference_points(points) -> Self`, `.with_tolerance(...) -> Self`,
    and `.with_charges(...) -> Self` (for a charged pair slot such as SPME
    real-space) refine a fixture.

### Functions <!-- rq-9c0e9e48 -->

- `assert_potential_consistent(fixture: &ConsistencyFixture, gpu: &GpuContext)` <!-- rq-9cd51562 -->
  - Builds the fixture's isolated single-slot system.
  - Runs, at every sample coordinate, the force–energy finite-difference check
    (all atoms, all axes), the Newton's-third-law check, and the virial check;
    and, for a `Pair` fixture, the switch/cutoff continuity check across the
    fixture's `[r_switch, cutoff]`.
  - Evaluates and asserts every declared reference point.
  - On the first failed assertion, panics with a message naming the invariant,
    the sample coordinate, the offending atom and axis where applicable, and
    both the expected and actual values.

- `builtin_consistency_fixtures() -> Vec<ConsistencyFixture>` <!-- rq-da0e909a -->
  - Returns one fixture per built-in fragment-composed potential: Lennard-Jones,
    SPME real-space, Morse bond, harmonic bond, harmonic angle, and periodic
    dihedral. Each carries samples spanning its range and at least one analytic
    reference point.

- `assert_all_builtin_potentials_consistent(gpu: &GpuContext)` <!-- rq-2e64e2c1 -->
  - Runs `assert_potential_consistent` for every fixture in
    `builtin_consistency_fixtures()`.
  - Asserts coverage: every fast-class fragment-composed slot label producible
    by `PotentialRegistry::with_builtins()` is targeted by some fixture. Panics
    naming any uncovered label.

## Out of Scope <!-- rq-be32cfe0 -->

- **The SPME reciprocal-space slot.** It is a slow-class FFT pipeline evaluated
  through `Potential::compute`, not a per-pair or per-bond fragment functor, so
  the finite-difference, Newton, virial, and continuity invariants over a
  minimal fragment system do not apply. Its correctness is checked by the
  SPME-versus-Ewald reference comparison and the end-to-end SPME runs
  (`e2e-testing.md`).
- **Multi-slot interaction.** The harness isolates one potential per system;
  the composition of several active slots is covered by the force-field
  framework tests (`framework.md`) and the end-to-end layer.
- **Cross-hardware or run-to-run reproducibility.** The harness asserts physical
  consistency of a single evaluation, not byte-identity; reproducibility is
  covered by `pipeline-reproducibility.md` and `e2e-testing.md`.
- **Rotational-invariance (torque) checks.** The harness asserts translational
  invariance (`Σ F = 0`); it does not separately assert `Σ r × F = 0`.

---

## Gherkin Scenarios <!-- rq-4dccf720 -->

```gherkin
Feature: Potential consistency harness

  Background:
    Given a GPU device is available
    And a fixture builds an isolated system carrying exactly one potential slot

  # --- Force–energy finite-difference consistency ---

  @rq-2d06b5d2
  Scenario: A correct pair potential passes the finite-difference check
    Given the Lennard-Jones fixture with samples spanning [0.9σ, cutoff]
    When assert_potential_consistent is called
    Then for every atom and axis the central finite difference of energy equals the negated force component within tolerance
    And the call does not panic

  @rq-0d01f64a
  Scenario: A sign-flipped force fails the finite-difference check
    Given a pair fixture whose fragment returns the negative of the correct force
    When assert_potential_consistent is called
    Then the call panics naming the force–energy invariant, a sample coordinate, and the offending atom and axis

  @rq-3654fa69
  Scenario: A force missing the 1/r factor fails the finite-difference check
    Given a pair fixture whose fragment returns −dU/dr instead of −(1/r)·dU/dr
    When assert_potential_consistent is called
    Then the call panics naming the force–energy invariant

  @rq-f417a0f5
  Scenario: A bonded potential that halves its own energy fails the finite-difference check
    Given a bond fixture whose fragment returns half the bond energy
    When assert_potential_consistent is called
    Then the call panics naming the force–energy or reference-point invariant

  # --- Newton's third law ---

  @rq-fd3ba41e
  Scenario: Equal and opposite pair forces pass
    Given the Lennard-Jones fixture
    When assert_potential_consistent is called
    Then the vector sum of the per-particle forces is zero within tolerance at every sample

  @rq-9d9747ed
  Scenario: Asymmetric force assignment fails Newton's third law
    Given a pair fixture whose fragment applies unequal forces to the two atoms
    When assert_potential_consistent is called
    Then the call panics naming the Newton's-third-law invariant

  # --- Virial–force consistency ---

  @rq-0e8b48bd
  Scenario: A correct pair virial matches the scaling finite difference
    Given the Lennard-Jones fixture
    When assert_potential_consistent is called
    Then the reported scalar virial equals −dU/d(ln λ) under isotropic scaling within tolerance at every sample

  @rq-24c9bf90
  Scenario: An inconsistent virial fails the virial check
    Given a pair fixture whose fragment reports a virial that is not factor·r²
    When assert_potential_consistent is called
    Then the call panics naming the virial invariant

  @rq-e4ad8b2c
  Scenario: An angle potential reports zero scalar virial
    Given the harmonic-angle fixture
    When assert_potential_consistent is called
    Then the reported scalar virial is zero within tolerance at every sample
    And the scaling finite difference of energy is zero within tolerance

  # --- Switch/cutoff continuity (pair only) ---

  @rq-fad67bb5
  Scenario: A C1 switching function joins smoothly and vanishes at the cutoff
    Given the Lennard-Jones fixture with a switching function over [r_switch, cutoff]
    When assert_potential_consistent is called
    Then the energy and force are continuous across r_switch and across the cutoff within tolerance
    And the energy and force are exactly zero for separations beyond the cutoff

  @rq-6d39e77d
  Scenario: A discontinuous switch fails the continuity check
    Given a pair fixture whose energy jumps at r_switch
    When assert_potential_consistent is called
    Then the call panics naming the continuity invariant

  @rq-1b1e47ee
  Scenario: A force that does not vanish beyond the cutoff fails
    Given a pair fixture whose force is nonzero for r greater than the cutoff
    When assert_potential_consistent is called
    Then the call panics naming the continuity invariant

  # --- Analytic reference points ---

  @rq-bd34fa41
  Scenario: Lennard-Jones reference points hold
    Given the Lennard-Jones fixture with reference points at r = σ (U = 0) and r = 2^(1/6)·σ (coord_force = 0, U = −ε)
    When assert_potential_consistent is called
    Then the evaluated energy and coordinate force match each reference point within its tolerance

  @rq-525bf72b
  Scenario: A uniformly wrong but self-consistent form fails a reference point
    Given a pair fixture whose energy is scaled by a constant factor, with a correct matching force
    When assert_potential_consistent is called
    Then the finite-difference check passes
    But the call panics naming the reference-point invariant

  # --- Shapes ---

  @rq-9db48a6d
  Scenario: A dihedral potential passes all applicable invariants
    Given the periodic-dihedral fixture with samples spanning [0, 2π]
    When assert_potential_consistent is called
    Then the force–energy, Newton, and virial checks pass at every sample
    And the continuity check is not applied

  # --- Built-in sweep and coverage ---

  @rq-cfa07a3c
  Scenario: Every built-in fragment potential passes the sweep
    When assert_all_builtin_potentials_consistent is called
    Then every fixture in builtin_consistency_fixtures passes
    And the call does not panic

  @rq-75cb6f7d
  Scenario: A built-in potential without a fixture fails coverage
    Given a fast-class fragment-composed built-in slot with no matching fixture in builtin_consistency_fixtures
    When assert_all_builtin_potentials_consistent is called
    Then the call panics naming the uncovered potential label

  # --- Isolation ---

  @rq-7700c3b8
  Scenario: A fixture system carries exactly one active slot
    Given any built-in fixture
    When its isolated system is built
    Then the ForceField contains exactly one potential slot
    And the box is wide enough that only the primary image interacts
```
