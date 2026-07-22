# Feature: Op Schedule Dependency Model <!-- rq-6b625df1 -->

A timestep is one canonical schedule of operations, and every operation in
it declares the simulation state it reads and writes. This declaration —
the operation's **resource footprint** — lets a validation pass confirm
the schedule is dependency-correct before the run starts. The pass enforces
two guarantees:

- **Intra-step freshness.** No operation consumes force-derived state
  (forces, per-class force accumulators) that a preceding position or box
  mutation *within the same step* has invalidated without an intervening
  force evaluation.
- **Cross-step freshness.** Because the same schedule is replayed every
  step, the force-derived state an operation reads at the *start* of a step
  must be state the schedule's previous iteration actually leaves valid at
  its *end* — not merely assumed to be valid. A schedule whose terminal
  operations invalidate the forces its own leading operations then read is
  rejected, unless an active weak-coupling pressure-coupling slot explicitly
  declares that the resulting staleness is an accepted approximation.

The two guarantees together make the schedule sound as a repeating loop, not
only as a single isolated walk.

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

### The inert `BarostatPoint` footprint <!-- rq-4fed3c90 -->

A constraint-capable integrator emits a `BarostatPoint` marker
unconditionally, and the marker is a no-op when no per-step barostat is
configured (see `framework.md`). The footprint above — reading and writing
the box and positions — is the marker's footprint *when a per-step barostat
is active*. When the validation context reports no active per-step barostat,
the marker is **inert**: it carries the empty footprint (no reads, no
writes) and therefore invalidates nothing. This is what keeps an NVE or NVT
schedule — which carries the marker but never scales the box — from being
treated as though it mutated the configuration at the end of every step.

### Custom operation footprints <!-- rq-c50ff880 -->

A `Custom` operation carries its footprint explicitly, because an
integrator-private sub-step's reads and writes are known only to the
integrator that emits it: `Custom { dt, label, reads: ResourceSet, writes:
ResourceSet }`. An integrator declares the footprint at the point it
builds the operation. A `Custom` operation that reads `Forces` or
`ClassForces(c)` is subject to the same freshness rule as any built-in
operation.

## Dependency validation <!-- rq-c5c17dcd -->

Validation is a walk over a `StepPlan`'s operations in schedule order,
against a set `V` of currently-valid resources. The core walk checks each
operation's reads and applies its writes:

- **Read check.** Every resource in the operation's `reads` must be in `V`.
  A read of a resource not in `V` is a validation error.
- **Invalidation.** If the operation's `writes` intersects
  `{ Positions, Box }`, remove every derived resource (`Forces` and all
  `ClassForces`) from `V` — the configuration changed, so the cached forces
  are stale.
- **Write.** Insert the operation's `writes` into `V`.

The read check uses `V` as it stands *before* the operation's own
invalidation and writes, so an operation that both reads `Forces` and
writes `Positions` (a `KickDrift`) reads the still-valid forces first and
only then invalidates them. Throughout the walk each operation contributes
its **effective** footprint: an inert `BarostatPoint` (no active per-step
barostat) contributes the empty footprint (see *The inert `BarostatPoint`
footprint*).

Full validation runs this walk twice, with two different seeds — once for
intra-step freshness and once for cross-step freshness.

### Intra-step pass <!-- rq-28c539f5 -->

The intra-step walk seeds `V` with the **carried-in** set — every base
resource plus `Forces` and both `ClassForces`. This reflects the first step
of a phase, which enters with valid forces from the warm-up evaluation. A
read that fails here is an intra-step staleness error
(`ScheduleError::ReadsStaleResource`): a force consumer placed after a
configuration change with no intervening (re-)evaluation *within the same
step*.

### Cross-step pass <!-- rq-338550b2 -->

The same schedule is replayed on every step, so from the second step onward
a step does not enter with all forces valid — it enters with exactly the
force-derived state the previous iteration of the schedule left valid. The
cross-step walk validates that steady state.

Its seed is derived from the schedule itself. Let `D_end` be the set of
derived resources valid at the *end* of the intra-step walk. `D_end` is a
property of the schedule alone, independent of which resources were valid
at the start: derived resources are removed only by operations that write
`Positions` or `Box`, and restored only by `ForceEval` operations, and
neither depends on the starting set. So the derived state a step hands to
its successor is fixed by the plan's sequence of force evaluations and
configuration mutations.

The **carry set** `D_carry` is the derived state a successor step may rely
on:

- Absent a tolerance declaration, `D_carry = D_end`. A schedule that ends
  by mutating the configuration (a terminal `BarostatPoint`) and does not
  re-evaluate forces afterward leaves `D_end` without the invalidated force
  resources, so its successor starts without them.
- When the active per-step barostat declares that the staleness left by its
  terminal rescale is an accepted approximation (see *Weak-coupling
  tolerance*), `D_carry` additionally includes every derived resource that
  the tolerated terminal configuration mutation invalidated. For a schedule
  whose only configuration change after its final `ForceEval` is that
  tolerated rescale, this restores the full carried-in set.

The cross-step walk seeds `V` with the base resources plus `D_carry` and
applies the same read / invalidate / write algorithm. A read that fails
here is a cross-step staleness error
(`ScheduleError::ReadsStaleCachedForce`): the schedule's leading operations
consume force-derived state that the schedule's own terminal operations
invalidated, with neither an intervening force evaluation nor a tolerance
declaration. The fix is either to append a `ForceEval` for the affected
class after the mutation, or — for a genuine weak-coupling rescale — to
declare the tolerance on the barostat.

The cross-step seed `D_carry` is never larger than the carried-in set, so
the cross-step pass is never weaker than the intra-step pass; a schedule
that passes it also passes the intra-step pass. Both are required
nonetheless: the intra-step pass alone would accept a schedule that is
correct on its warm-up-seeded first step and stale on every step after it.

### Weak-coupling tolerance <!-- rq-fd52063f -->

A per-step barostat that rescales the box and positions at the end of the
step (the weak-coupling `berendsen` and `c-rescale` barostats) leaves the
cached forces describing the pre-rescale configuration. Recomputing forces
after the rescale would double the force evaluations per step; instead the
leading half-kick of the next step consumes those pre-rescale forces, a
bounded second-order approximation standard for weak-coupling pressure
coupling. Such a barostat reports `Barostat::tolerates_stale_cached_forces
== true`, and the runner carries that value into the validation context.

A barostat that does **not** declare the tolerance — the default — is held
to strict loop-consistency: a schedule that ends by mutating the
configuration under that barostat and reads the invalidated forces at the
next step's start is rejected. The periodic Monte-Carlo barostat reports
`false`: its `apply` is inert, it places no mutating operation in the
per-step schedule, and its host-side volume move rebuilds the neighbour
list and re-evaluates forces at the next batch boundary, so it introduces
no cross-step force staleness to tolerate.

The tolerance covers only *force-derived* staleness from the terminal
rescale. Neighbour-list validity across the box change is a separate
concern, handled by the displacement-check rebuild trigger
(`forces/neighbor-list.md`), not by this model.

### When validation runs <!-- rq-89908fc6 -->

The runner validates an integrator's plan once per phase, at phase setup,
against the phase's probe plan and a `StepValidationContext` it assembles
from the phase's configured slots — whether a per-step barostat is active
and, if so, whether that barostat tolerates stale cached forces. A
validation failure aborts phase setup with a runner error naming the
offending operation and resource; the run does not enter the timestep loop.
Validation is a pure function of the plan and the context, performs no
device work, and is therefore directly unit-testable on hand-built plans.

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
- `StepValidationContext` — the phase context validation is performed <!-- rq-b83f8ae6 -->
  against. Carries at minimum:
  - `per_step_barostat_active: bool` — whether a per-step barostat is
    configured for the phase. When `false`, a `BarostatPoint` marker is
    inert and contributes the empty footprint.
  - `tolerates_stale_cached_forces: bool` — whether the active per-step
    barostat accepts the force staleness its terminal rescale leaves for
    the next step (see *Weak-coupling tolerance*). Meaningful only when
    `per_step_barostat_active` is `true`. The runner sets it from the
    configured barostat's `Barostat::tolerates_stale_cached_forces()`.
- `ScheduleError` — error returned by validation. Includes at minimum: <!-- rq-3fd3777d -->
  - `ReadsStaleResource { index: usize, op: &'static str, resource: Resource }`
    — the operation at `index` (named by `SubStep::variant_name()`) reads
    `resource`, which is not valid at that point in the schedule (stale
    force-derived state, or a resource never produced this step). Raised by
    the intra-step pass.
  - `ReadsStaleCachedForce { index: usize, op: &'static str, resource: Resource }`
    — the operation at `index` reads force-derived `resource` at the start
    of a step that the schedule's own terminal operations invalidated, with
    no intervening force evaluation and no weak-coupling tolerance. Raised
    by the cross-step pass.

### Functions and methods <!-- rq-025c93eb -->

- `SubStep::footprint(&self) -> OpFootprint` <!-- rq-d20fe5ce -->
  - Returns the operation's declared reads and writes per the table above.
    For `Custom`, returns the `reads` / `writes` the variant carries. This
    is the static footprint; it reports the full `BarostatPoint` footprint
    regardless of whether a barostat is active.
- `SubStep::effective_footprint(&self, ctx: &StepValidationContext) -> OpFootprint` <!-- rq-2e035fd8 -->
  - Returns the footprint the validator uses for this operation under
    `ctx`. Equals `footprint()` for every operation except a
    `BarostatPoint` when `ctx.per_step_barostat_active` is `false`, for
    which it returns the empty footprint (the inert marker).
- `StepPlan::validate(&self, ctx: &StepValidationContext) -> Result<(), ScheduleError>` <!-- rq-129c5de9 -->
  - Runs the intra-step pass and the cross-step pass described above, using
    each operation's `effective_footprint(ctx)`. Returns `Ok(())` for a
    schedule that passes both (including the empty plan). Returns the first
    failure otherwise: a `ReadsStaleResource` from the intra-step pass, or,
    when the intra-step pass succeeds, a `ReadsStaleCachedForce` from the
    cross-step pass.
  - Pure: a function of the plan and `ctx` only; launches no kernels and
    touches no device buffers.
- `Barostat::tolerates_stale_cached_forces(&self) -> bool` <!-- rq-c8be316e -->
  - Whether the barostat accepts that its terminal per-step rescale leaves
    the next step's leading force consumers reading pre-rescale forces.
    Default `false`. The weak-coupling `berendsen` and `c-rescale`
    barostats return `true`; the periodic Monte-Carlo barostat returns
    `false`. The trait itself is defined in `framework.md`; the runner
    reads this method to populate `StepValidationContext`.

### Runner integration <!-- rq-77f1e6ef -->

- The runner assembles a `StepValidationContext` from the phase's
  configured slots — `per_step_barostat_active` from whether a per-step
  barostat is present, and `tolerates_stale_cached_forces` from that
  barostat's `Barostat::tolerates_stale_cached_forces()` — and calls
  `StepPlan::validate` on the phase's probe plan with it during phase
  setup, before CUDA-graph eligibility and the timestep loop. A returned
  `ScheduleError` is surfaced as a runner setup error that aborts the
  phase.
- When the integrator owns its pressure coupling rather than taking a
  separate barostat slot (`mtk-npt`), the integrator emits the box-mutating
  operations directly, and the runner sources `per_step_barostat_active`
  and `tolerates_stale_cached_forces` from the integrator's own declaration
  in place of a barostat slot's. The cross-step guarantee is otherwise
  identical: such an integrator's plan must either be loop-consistent (end
  with the forces its next step reads valid) or declare the weak-coupling
  tolerance.

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
exception*) and the leading velocity projection that precedes the
thermostat's `apply_post` and the terminal `BarostatPoint` on every step of
a constrained run (see *Out of Scope*). Neither is a schedule operation,
and neither reorders operations
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
- **The leading velocity projection.** On every step of a constrained run
  the runner projects the velocities onto the constraint manifold — and
  publishes the constraint virial — between the trailing kick and the
  wrapped thermostat's `apply_post`, so that the thermostat couples to the
  on-manifold kinetic energy and the terminal `BarostatPoint`'s virial
  reduction sees the constraint contribution (`framework.md`, *Per-Step
  Interface*; `constraint-framework.md`). It is **not** a schedule
  operation: it declares no footprint, appears in no `StepPlan`, and is not
  validated. It carries no ordering rules of its own either — it re-uses the
  `dt` of the plan's terminal `ConstraintPoint { AfterKick }` (and fires
  only when the plan has one), and that marker still executes in place in
  the post-force tail, where it dispatches the repair projection. Its
  footprint would in any case be identical to the `AfterKick` marker's, so
  it can invalidate nothing the validator tracks. (The virial is not a
  tracked resource — see the bullet above — so the model does not express
  the publish-before-reduce dependency either; that ordering is a
  `run_step` invariant.)

## Gherkin Scenarios <!-- rq-f3467a4e -->

```gherkin
Feature: Op schedule dependency validation

  Background:
    Given the validation context has no active per-step barostat
    # Unless a scenario states otherwise, a BarostatPoint marker is therefore
    # inert (empty footprint) and StepPlan::validate is called with that context.

  @rq-1c8baf7d
  Scenario: A velocity-Verlet plan with an inert BarostatPoint validates
    Given a plan [ConstraintPoint{BeforeDrift}, KickDrift{Total}, ConstraintPoint{AfterDrift},
      ForceEval{None}, KickHalf{Total}, ConstraintPoint{AfterKick}, BarostatPoint]
    When StepPlan::validate is called
    Then it returns Ok(())
    And the inert BarostatPoint invalidates no forces, so the cross-step pass passes

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

  @rq-97093c72
  Scenario: A terminal box mutation under an untolerated barostat leaves the next step's forces stale
    Given the validation context has an active per-step barostat that does not tolerate stale cached forces
    And a plan [ConstraintPoint{BeforeDrift}, KickDrift{Total}, ConstraintPoint{AfterDrift},
      ForceEval{None}, KickHalf{Total}, ConstraintPoint{AfterKick}, BarostatPoint]
    When StepPlan::validate is called
    Then the intra-step pass returns Ok(())
    And the cross-step pass returns Err(ScheduleError::ReadsStaleCachedForce) for the KickDrift reading Forces

  @rq-060b1323
  Scenario: A terminal box mutation under a tolerant barostat validates
    Given the validation context has an active per-step barostat that tolerates stale cached forces
    And a plan [ConstraintPoint{BeforeDrift}, KickDrift{Total}, ConstraintPoint{AfterDrift},
      ForceEval{None}, KickHalf{Total}, ConstraintPoint{AfterKick}, BarostatPoint]
    When StepPlan::validate is called
    Then it returns Ok(())

  @rq-df64e69a
  Scenario: A trailing force evaluation makes a terminal box mutation loop-consistent without tolerance
    Given the validation context has an active per-step barostat that does not tolerate stale cached forces
    And a plan [KickDrift{Total}, ForceEval{None}, KickHalf{Total}, BarostatPoint, ForceEval{None}]
    When StepPlan::validate is called
    Then it returns Ok(())
    And the cross-step carry set includes Forces because the plan ends with a ForceEval

  @rq-8e1ce2f8
  Scenario: A plan that begins with a force evaluation is robust to a terminal box mutation
    Given the validation context has an active per-step barostat that does not tolerate stale cached forces
    And a plan [ForceEval{None}, KickHalf{Total}, KickDrift{Total}, ForceEval{None}, KickHalf{Total}, BarostatPoint]
    When StepPlan::validate is called
    Then it returns Ok(())
    And no leading operation reads carried-in Forces before the first ForceEval

  @rq-229f0723
  Scenario: A RESPA plan with an untolerated terminal box mutation leaves the slow accumulator stale across the boundary
    Given the validation context has an active per-step barostat that does not tolerate stale cached forces
    And a RESPA plan [KickHalf{Class(Slow)},
      (KickDrift{Class(Fast)}, ForceEval{Some(Fast)}, KickHalf{Class(Fast)}) x n_inner,
      ForceEval{Some(Slow)}, KickHalf{Class(Slow)}, BarostatPoint]
    When StepPlan::validate is called
    Then the intra-step pass returns Ok(())
    And the cross-step pass returns Err(ScheduleError::ReadsStaleCachedForce) for the leading KickHalf reading ClassForces(Slow)

  @rq-009c28e2
  Scenario: The same RESPA plan validates under a tolerant barostat
    Given the validation context has an active per-step barostat that tolerates stale cached forces
    And a RESPA plan [KickHalf{Class(Slow)},
      (KickDrift{Class(Fast)}, ForceEval{Some(Fast)}, KickHalf{Class(Fast)}) x n_inner,
      ForceEval{Some(Slow)}, KickHalf{Class(Slow)}, BarostatPoint]
    When StepPlan::validate is called
    Then it returns Ok(())

  @rq-450484bb
  Scenario: effective_footprint reports an inert BarostatPoint as empty
    Given a BarostatPoint operation
    And a validation context with no active per-step barostat
    When effective_footprint(ctx) is called
    Then reads == {} and writes == {}

  @rq-13cb1367
  Scenario: effective_footprint reports an active BarostatPoint with its full footprint
    Given a BarostatPoint operation
    And a validation context with an active per-step barostat
    When effective_footprint(ctx) is called
    Then reads == {Velocities, Box} and writes == {Positions, Velocities, Box}

  @rq-011858f8
  Scenario: Weak-coupling barostats declare force-staleness tolerance
    Given the c-rescale barostat and the berendsen barostat
    Then tolerates_stale_cached_forces() returns true for each

  @rq-6814860a
  Scenario: The Monte-Carlo barostat does not declare tolerance
    Given the monte-carlo barostat
    Then tolerates_stale_cached_forces() returns false
```
