# Feature: Op Schedule Dependency Model <!-- rq-6b625df1 -->

A timestep is one canonical schedule of operations, and every operation in
it declares the simulation state it reads and writes. This declaration —
the operation's **resource footprint** — lets a validation pass confirm
the schedule is dependency-correct before the run starts: every operation
observes fresh state, and no operation consumes force-derived state
(forces, per-class force accumulators) that a preceding position or box
mutation has invalidated without an intervening force evaluation.

The schedule is the `StepPlan` an integrator returns from
`Integrator::plan` (see `framework.md`); its operations are `SubStep`
values. This model adds, to each operation, the footprint that makes the
schedule's data dependencies explicit, and a pass that checks them. It is
the structural foundation of the principle stated in `docs/architecture.md`
(*Step orchestration and kernel fusion*): the schedule is the single
authoritative definition of a timestep, and an operation's placement in it
determines which state that operation observes. The validator makes that
principle machine-checked rather than a convention.

Each operation runs as its own kernel launch; the model coalesces nothing.
Its value is correctness and extensibility — a schedule with an
ill-ordered force dependency is rejected at phase setup rather than
silently producing wrong dynamics — not performance.

## Resources <!-- rq-38931487 -->

A `Resource` names one distinct component of per-particle or global
simulation state that the schedule tracks for dependency reasoning:

- `Positions` — the per-particle position arrays.
- `Velocities` — the per-particle velocity arrays.
- `Images` — the per-particle periodic image flags.
- `Box` — the simulation box / lattice.
- `Forces` — the combined per-particle net force arrays
  (`ParticleBuffers.forces_*`).
- `ClassForces(ForceClass)` — a single force class's accumulator
  (`Fast` or `Slow`; see `framework.md` and `respa.md`).

`Positions`, `Velocities`, `Images`, and `Box` are **base** resources:
they hold state directly and are always readable. `Forces` and
`ClassForces(_)` are **derived** resources: they are a function of the
current positions and box, produced by a force evaluation. A write to
`Positions` or `Box` makes every derived resource stale — the cached
forces no longer correspond to the current configuration — until a force
evaluation reproduces them.

A `ResourceSet` is a set of `Resource` values (the reads or writes of one
operation).

Kinetic energy, virial, and potential energy are **not** tracked
resources in this model. Their reductions are performed inside the
composite thermostat / barostat operations, whose footprints declare only
the base and force resources they touch (see *Out of Scope*).

## Operation footprints <!-- rq-c04f856f -->

An operation's footprint is an `OpFootprint { reads: ResourceSet, writes:
ResourceSet }`. The built-in `SubStep` variants have fixed footprints,
determined by the variant and its fields:

| Operation                                   | reads                          | writes                          |
| ------------------------------------------- | ------------------------------ | ------------------------------- |
| `KickHalf { source: Total }`                | Velocities, Forces             | Velocities                      |
| `KickHalf { source: Class(c) }`             | Velocities, ClassForces(c)     | Velocities                      |
| `Drift`                                      | Velocities                     | Positions, Images               |
| `KickDrift { source: Total }`               | Velocities, Forces             | Velocities, Positions, Images   |
| `KickDrift { source: Class(c) }`            | Velocities, ClassForces(c)     | Velocities, Positions, Images   |
| `ForceEval { class: None }`                 | Positions, Box                 | Forces, ClassForces(Fast), ClassForces(Slow) |
| `ForceEval { class: Some(c) }`              | Positions, Box                 | Forces, ClassForces(c)          |
| `ThermostatHalf`                            | Velocities                     | Velocities                      |
| `ConstraintPoint { phase: BeforeDrift }`    | Positions, Velocities, Box     | Velocities                      |
| `ConstraintPoint { phase: AfterDrift }`     | Positions, Velocities, Box     | Positions, Velocities           |
| `ConstraintPoint { phase: AfterKick }`      | Positions, Velocities, Box     | Velocities                      |
| `BarostatPoint`                             | Velocities, Box                | Positions, Velocities, Box      |

`ForceEval`'s footprint is independent of its `AggregateLevel`: whether or
not it computes scalar aggregates, it reads the current configuration and
(re)produces the force resources.

A `ForceEval { class: Some(c) }` re-validates only `ClassForces(c)` (and
the combined `Forces` the combiner refreshes from every class), leaving
the other class's accumulator as it was. This is what lets the validator
distinguish a correctly re-evaluated multiple-timestep schedule from one
that kicks a class on stale accumulators.

### Custom operation footprints <!-- rq-c50ff880 -->

A `Custom` operation carries its footprint explicitly, because an
integrator-private sub-step's reads and writes are known only to the
integrator that emits it: `Custom { dt, label, reads: ResourceSet, writes:
ResourceSet }`. An integrator declares the footprint at the point it
builds the operation. A `Custom` operation that reads `Forces` or
`ClassForces(c)` is subject to the same freshness rule as any built-in
operation.

## Dependency validation <!-- rq-c5c17dcd -->

Validation walks a `StepPlan`'s operations in order against a set `V` of
currently-valid resources, checking each operation's reads and applying
its writes:

1. `V` is seeded with the **carried-in** set — every base resource plus
   `Forces` and both `ClassForces` — reflecting that an integrator enters
   a step with valid cached forces from the previous step's (or the
   warm-up's) final evaluation. The scalar-free force resources are the
   only derived state, and they are carried across the step boundary.
2. For each operation, in schedule order:
   a. **Read check.** Every resource in the operation's `reads` must be
      in `V`. A read of a resource not in `V` is a validation error.
   b. **Invalidation.** If the operation's `writes` intersects
      `{ Positions, Box }`, remove every derived resource (`Forces` and
      all `ClassForces`) from `V` — the configuration changed, so the
      cached forces are stale.
   c. **Write.** Insert the operation's `writes` into `V`.

The read check uses `V` as it stands *before* the operation's own
invalidation and writes, so an operation that both reads `Forces` and
writes `Positions` (a `KickDrift`) reads the still-valid forces first and
only then invalidates them.

Each step is validated independently with the carried-in seed; the model
does not carry invalidation across the step boundary. A terminal
`BarostatPoint` that mutates the box invalidates the force resources at
the very end of the step, but the next step is validated afresh with
forces valid — the cross-step force-caching contract
(`velocity-verlet.md`, symplectic-with-cached-forces) is assumed, not
re-derived here. What validation catches is **intra-step** staleness: a
force consumer placed after a configuration change with no intervening
(re-)evaluation.

### When validation runs <!-- rq-89908fc6 -->

The runner validates an integrator's plan once per phase, at phase setup,
against the phase's probe plan. A validation failure aborts phase setup
with a runner error naming the offending operation and resource; the run
does not enter the timestep loop. Validation is a pure function of the
plan and performs no device work, so it is also directly unit-testable on
hand-built plans.

## Feature API <!-- rq-21d945b9 -->

### Types <!-- rq-0bc28d40 -->

- `Resource` — closed enum: `Positions`, `Velocities`, `Images`, `Box`, <!-- rq-f44776cb -->
  `Forces`, `ClassForces(ForceClass)`. `Copy`, `Eq`, `Hash`.
- `ResourceSet` — a set of `Resource`. Supports construction from a slice <!-- rq-5cf3694c -->
  / iterator, membership test (`contains`), single-element `insert` /
  `remove`, and iteration (`iter`) — enough to express the read check and
  the `{ Positions, Box }` invalidation test, both of which the validator
  performs element-wise over an operation's footprint.
- `OpFootprint` — `{ reads: ResourceSet, writes: ResourceSet }`. <!-- rq-5414f115 -->
- `ScheduleError` — error returned by validation. Includes at minimum: <!-- rq-3fd3777d -->
  - `ReadsStaleResource { index: usize, op: &'static str, resource: Resource }`
    — the operation at `index` (named by `SubStep::variant_name()`) reads
    `resource`, which is not valid at that point in the schedule (stale
    force-derived state, or a resource never produced this step).

### Functions and methods <!-- rq-025c93eb -->

- `SubStep::footprint(&self) -> OpFootprint` <!-- rq-d20fe5ce -->
  - Returns the operation's declared reads and writes per the table above.
    For `Custom`, returns the `reads` / `writes` the variant carries.
- `StepPlan::validate(&self) -> Result<(), ScheduleError>` <!-- rq-129c5de9 -->
  - Walks the plan applying the validation algorithm above with the
    carried-in seed. Returns `Ok(())` for a dependency-correct schedule
    (including the empty plan) and the first `ScheduleError` otherwise.
  - Pure: launches no kernels and touches no device buffers.

### Runner integration <!-- rq-77f1e6ef -->

- The runner calls `StepPlan::validate` on the phase's probe plan during
  phase setup, before CUDA-graph eligibility and the timestep loop. A
  returned `ScheduleError` is surfaced as a runner setup error that aborts
  the phase.

## Relationship to fusion and determinism <!-- rq-c8783edf -->

This model makes the `docs/architecture.md` fusion principle structural:
the schedule declares each operation's reads and writes, and validation
confirms no operation observes stale state. Because every operation still
runs as its own kernel launch — the model performs no coalescing — the
per-step launch sequence, and therefore the engine's bit-wise
reproducibility, is the schedule walked in order plus the small, fixed set
of dispatches the runner wraps around that walk (`framework.md`,
*Determinism Guarantees*). Those wrapped dispatches are the documented
default-topology thermostat halves (`docs/architecture.md`, *One deliberate
exception*) and the pre-coupling velocity projection that precedes the
thermostat's `apply_post` on a constrained coupling step (see *Out of
Scope*). Neither is a schedule operation, and neither reorders operations
within the walk or changes which state an operation observes. Validation
adds a setup-time check; it changes no per-step execution.

## Out of Scope <!-- rq-506baad0 -->

- **No coalescing fuser.** The model declares and validates footprints; it
  does not merge adjacent pointwise operations into a single launch. Each
  operation is one launch.
- **Composite thermostat / barostat operations.** `ThermostatHalf`,
  `ConstraintPoint`, and `BarostatPoint` are single operations whose
  footprints declare the aggregate base and force resources they touch.
  The reductions performed inside a thermostat's `apply_post` or a
  barostat's `apply` are not themselves schedule operations, and kinetic
  energy, virial, and potential energy are not tracked resources. Making
  those reductions first-class barrier operations — with scalar resources
  and their own producer/consumer validation — is a separate layer of the
  Op model, not part of this foundation.
- **The pre-coupling velocity projection.** On a coupling step of a
  constrained run the runner projects the velocities onto the constraint
  manifold between the trailing kick and the wrapped thermostat's
  `apply_post`, so the thermostat couples to the on-manifold kinetic energy
  (`framework.md`, *Per-Step Interface*; `constraint-framework.md`). It is
  **not** a schedule operation: it declares no footprint, appears in no
  `StepPlan`, and is not validated. It carries no ordering rules of its own
  either — it re-uses the `dt` of the plan's terminal `ConstraintPoint {
  AfterKick }` (and fires only when the plan has one), and that marker still
  executes in place in the post-force tail. Its footprint would in any case
  be identical to the `AfterKick` marker's, so it can invalidate nothing the
  validator tracks.

## Gherkin Scenarios <!-- rq-f3467a4e -->

```gherkin
Feature: Op schedule dependency validation

  @rq-1c8baf7d
  Scenario: A velocity-Verlet plan validates
    Given a plan [ConstraintPoint{BeforeDrift}, KickDrift{Total}, ConstraintPoint{AfterDrift},
      ForceEval{None}, KickHalf{Total}, ConstraintPoint{AfterKick}, BarostatPoint]
    When StepPlan::validate is called
    Then it returns Ok(())

  @rq-ab2607c7
  Scenario: The empty plan validates
    Given a plan with no operations
    When StepPlan::validate is called
    Then it returns Ok(())

  @rq-0625c5d4
  Scenario: Base resources are readable at the start of a step
    Given a plan [ThermostatHalf{Pre}, ForceEval{None}, KickHalf{Total}]
    When StepPlan::validate is called
    Then it returns Ok(())
    And no ReadsStaleResource is raised for the ThermostatHalf reading Velocities

  @rq-7fde3409
  Scenario: Cached forces are readable at the start of a step
    Given a plan [KickDrift{Total}, ForceEval{None}, KickHalf{Total}]
    When StepPlan::validate is called
    Then it returns Ok(())
    And the leading KickDrift reads the carried-in Forces without error

  @rq-df53eb91
  Scenario: A force read after a drift with no intervening force evaluation is stale
    Given a plan [KickDrift{Total}, KickHalf{Total}]
    When StepPlan::validate is called
    Then it returns Err(ScheduleError::ReadsStaleResource) for the KickHalf at index 1 reading Forces

  @rq-b577c9bd
  Scenario: A force evaluation re-validates forces after a position update
    Given a plan [KickDrift{Total}, ForceEval{None}, KickHalf{Total}]
    When StepPlan::validate is called
    Then it returns Ok(())

  @rq-a53f18a6
  Scenario: A bare Drift then a total kick is stale
    Given a plan [Drift, KickHalf{Total}]
    When StepPlan::validate is called
    Then it returns Err(ScheduleError::ReadsStaleResource) for the KickHalf reading Forces

  @rq-41dab871
  Scenario: A RESPA plan validates
    Given a RESPA plan [KickHalf{Class(Slow)},
      (KickDrift{Class(Fast)}, ForceEval{Some(Fast)}, KickHalf{Class(Fast)}) x n_inner,
      ForceEval{Some(Slow)}, KickHalf{Class(Slow)}]
    When StepPlan::validate is called
    Then it returns Ok(())

  @rq-95031d97
  Scenario: A class kick on an accumulator invalidated by a drift is stale
    Given a plan [KickDrift{Class(Fast)}, KickHalf{Class(Fast)}]
    When StepPlan::validate is called
    Then it returns Err(ScheduleError::ReadsStaleResource) for the KickHalf reading ClassForces(Fast)

  @rq-12a0b0f8
  Scenario: A class-specific force evaluation re-validates only its own class
    Given a plan [KickDrift{Class(Fast)}, ForceEval{Some(Fast)}, KickHalf{Class(Slow)}]
    When StepPlan::validate is called
    Then it returns Err(ScheduleError::ReadsStaleResource) for the KickHalf reading ClassForces(Slow)

  @rq-ce497f66
  Scenario: A KickDrift reads valid forces before invalidating them
    Given a plan [ForceEval{None}, KickDrift{Total}, ForceEval{None}, KickHalf{Total}]
    When StepPlan::validate is called
    Then it returns Ok(())
    And the KickDrift at index 1 reads Forces without error

  @rq-cf364916
  Scenario: A Custom operation that reads forces after a drift is stale
    Given a plan [KickDrift{Total}, Custom{reads: {Velocities, Forces}, writes: {Velocities}}]
    When StepPlan::validate is called
    Then it returns Err(ScheduleError::ReadsStaleResource) for the Custom operation reading Forces

  @rq-4ab1c94e
  Scenario: A Custom operation reading only base resources always validates
    Given a plan [KickDrift{Total}, Custom{reads: {Velocities}, writes: {Velocities}}, ForceEval{None}, KickHalf{Total}]
    When StepPlan::validate is called
    Then it returns Ok(())

  @rq-1fecec44
  Scenario: SubStep::footprint reports the declared reads and writes
    Given a KickHalf { source: Class(Slow) }
    When footprint() is called
    Then reads == {Velocities, ClassForces(Slow)} and writes == {Velocities}

  @rq-7734703f
  Scenario: The runner rejects a plan that fails validation at phase setup
    Given an integrator whose plan reads Forces after a Drift with no intervening ForceEval
    When the runner enters the phase
    Then phase setup returns a runner error carrying the ScheduleError
    And the timestep loop does not run
```
