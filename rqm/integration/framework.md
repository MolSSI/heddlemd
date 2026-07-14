# Feature: Pluggable Integration Framework <!-- rq-e0a0553d -->

The runner drives time integration through four orthogonal slots that
compose at every timestep:

1. An `Integrator` — the core time-stepping algorithm. Owns the velocity
   kicks, position drifts, and the in-step call into `force_field.step`.
2. An optional `Thermostat` — temperature coupling. Fires twice per
   step, once before the integrator (`apply_pre`) and once after
   (`apply_post`), so symmetric Trotter splittings such as
   Nosé-Hoover-chain can place a half-step on each side of the
   integrator's velocity-Verlet body.
3. An optional `Barostat` — pressure coupling. A per-step barostat's
   `apply` is dispatched at a `SubStep::BarostatPoint` marker the
   integrator places in its plan, so pressure coupling is plan-declared
   rather than fixed by the runner. An integrator that hosts a per-step
   barostat emits a single terminal `BarostatPoint` after its final
   velocity update; the runner runs `apply` there, in the post-force
   tail, consuming the freshest virial / kinetic-energy data, mutating
   the box for the next step's force evaluation, and rescaling positions.
   An integrator whose splitting requires interleaved coupling instead
   places the `BarostatPoint` mid-plan, where `apply` runs during the
   walk. A periodic barostat (declared through
   `Barostat::periodicity`) performs no per-step work and runs a
   host-orchestrated move every `N` steps at a batch boundary through
   `apply_move` (the Monte-Carlo barostat; see `mc-barostat.md`); its
   `apply` is the no-op default, so a `BarostatPoint` is inert for it.
4. An optional `Constraint` — holonomic constraint projection (rigid
   bonds, rigid groups). Driven by the runner during its walk of the
   integrator's `StepPlan` (see *Per-Step Interface*): a
   constraint-capable integrator places `SubStep::ConstraintPoint`
   markers in its plan at the sub-step boundaries where the slot must
   snapshot or project, and the runner dispatches each marker to the
   corresponding `Constraint` hook. Integrators never hold a reference
   to the constraint slot; they only declare where its hooks belong.
   The slot, its trait, its data layout, and its compatibility rules
   are defined in `constraint-framework.md`; the v1 implementation
   (SETTLE for three-atom rigid water) lives in `settle.md`.

Each slot is independently registered and independently selectable
from TOML. Omitting `[thermostat]` selects NVE; omitting `[barostat]`
selects constant-volume; an empty (or absent) `[constraints]` section
of the topology file selects no constraints.

Some integrators own their own thermostat (the O step in Langevin
BAOAB *is* the Ornstein-Uhlenbeck thermostat); some additionally own
their own barostat (the MTK NPT integrator carries an extended-system
cell DOF and its own thermostat chains on both the particles and the
cell). Those integrators declare ownership through their builder's
`IntegratorBuilder::owns_thermostat(&params)` and
`IntegratorBuilder::owns_barostat(&params)` predicate methods, and
the config loader rejects co-configured `[thermostat]` /
`[barostat]` tables at load time. An analogous predicate
`IntegratorBuilder::supports_constraints(&params)` gates the
constraint slot: integrators that do not drive the constraint hooks
are incompatible with a non-empty `[constraints]` topology section.
See `constraint-framework.md` for the full rule.

## Slots <!-- rq-f8bb021a -->

### Integrator slot <!-- rq-d8c0e5b0 -->

The default registry exposes three integrators:

| `kind` value      | Owns thermostat? | Owns barostat? | Implementation                                                   | File                  |
| ----------------- | ---------------- | -------------- | ---------------------------------------------------------------- | --------------------- |
| `velocity-verlet` | no               | no             | symplectic NVE (lossy or lossless)                               | `velocity-verlet.md`  |
| `langevin-baoab`  | yes              | no             | stochastic NVT via BAOAB splitting                               | `langevin-baoab.md`   |
| `mtk-npt`         | yes              | yes            | deterministic NPT via MTK extended-system (isotropic, fused)     | `mtk-npt.md`          |

Each implementation's per-step kernels, parameter set, and timings
stages are documented in its own requirements file.

### Thermostat slot <!-- rq-0f5fba54 -->

The default registry exposes four thermostats:

| `kind` value         | Implementation                                              | File                  |
| -------------------- | ----------------------------------------------------------- | --------------------- |
| `nose-hoover-chain`  | deterministic NVT via MKT Nosé-Hoover-chain Trotter step    | `nose-hoover-chain.md`|
| `csvr`               | stochastic NVT via canonical sampling velocity rescaling    | `csvr.md`             |
| `andersen`           | stochastic NVT via per-particle Maxwell-Boltzmann resampling| `andersen.md`         |
| `berendsen`          | weak-coupling (equilibration only — not canonical)          | `berendsen.md`        |

A thermostat couples on a fixed cadence — every `coupling_interval`
steps (`io/config-schema.md`), where an interval of `1` couples every
step. On a coupling step the thermostat reduces the **full-step**
kinetic energy — the kinetic energy of the velocities *after* the
integrator's trailing kick — and applies its velocity update from that
reduction. This matches the symmetric-Trotter convention of
established MD engines (LAMMPS `fix nvt`, OpenMM `NoseHooverIntegrator`,
GROMACS `md-vv`), and preserves the time-reversal symmetry the
integrator's lossless mode relies on (`docs/architecture.md`,
*Precision policy*).

Because the kinetic-energy reduction reads post-trailing-kick
velocities, it is a **fusion barrier** between the trailing kick and the
thermostat's rescale (`docs/architecture.md`, *Step orchestration and
kernel fusion*). Its reduction and rescale run as their own standalone
kernel launches on coupling steps (see *Per-Step Interface*).

### Barostat slot <!-- rq-d898f1cd -->

The default registry exposes three barostats:

| `kind` value   | Implementation                                                       | File                      |
| -------------- | -------------------------------------------------------------------- | ------------------------- |
| `berendsen`    | weak-coupling isotropic pressure coupling (equilibration only — not canonical) | `berendsen-barostat.md`   |
| `c-rescale`    | stochastic isotropic cell-rescaling (canonical NPT)                  | `c-rescale-barostat.md`   |
| `monte-carlo`  | periodic isotropic Metropolis volume moves on molecular centres of mass (canonical NPT) | `mc-barostat.md`          |

## Per-Step Interface <!-- rq-daadfc1a -->

An integrator describes its work as an ordered sequence of typed
*sub-steps* (a `StepPlan`) and exposes a method that executes one
sub-step at a time. The runner walks the plan, dispatching each
sub-step to `integrator.execute(...)`, `force_field.step(...)`, the
configured thermostat, or the configured constraint slot depending on
the sub-step's variant (see `constraint-framework.md` for the hook
contract). Integrators never reference the constraint slot directly;
a constraint-capable integrator only declares where the slot's hooks
belong by placing `SubStep::ConstraintPoint` markers in its plan.

Each sub-step declares the simulation state it reads and writes — its
resource footprint — and the runner validates the plan's data
dependencies at phase setup, rejecting a schedule that consumes stale
force-derived state. The resource model, footprints, and validation are
defined in `op-model.md`.

The runner drives the timestep loop by calling the free function
`run_step` once per timestep. `run_step` walks the integrator's **entire**
`StepPlan` — through its post-force tail — and is the single executor of
a timestep's per-particle work. Every post-force per-particle operation —
the integrator's trailing kick, the thermostat's velocity rescale, the
barostat's position/velocity rescale, and the constraint slot's velocity
projection — runs as its own standalone kernel launch. There is no
composed post-force kernel; `run_step` executes the post-force region as
a fixed, explicit, canonical sequence
`integrator → thermostat → barostat → constraint projection`
(`docs/architecture.md`, *Step orchestration and kernel fusion*: global
reductions are fusion barriers, and the post-force pointwise operations
are separated by the reductions they consume, so there is nothing to
fuse). The plan, walked in full by `run_step`, is the single source of
truth for this ordering; the runner supplies only the coupling-step
decision (through `RunStepOptions.coupling_dt`, below) and never
re-encodes the tail.

Thermostat placement has two topologies, selected by the plan itself: a
plan with no `ThermostatHalf` sub-steps is **wrapped** by `run_step`
(`apply_pre` before the walk, `apply_post` between the trailing kick and
the terminal barostat / projection — the default topology every
single-timestep integrator uses); a plan containing `ThermostatHalf`
sub-steps owns its thermostat placement, dispatches those markers during
the walk, and receives no wrapping.

A wrapped thermostat's `apply_pre` and `apply_post` fire only on
**coupling steps** — steps where `step % coupling_interval == 0`
(`io/config-schema.md`). The runner owns the step counter and therefore
computes the cadence: on a coupling step it passes
`RunStepOptions.coupling_dt = Some(dt_couple)`, and `run_step` fires both
wrapped halves with that `dt_couple`; on the intervening steps it passes
`None` and the thermostat is inert (neither half runs and velocities are
untouched by it). The **effective coupling timestep**
`dt_couple = coupling_interval · dt` keeps a thermostat's relaxation time
`τ` physically meaningful regardless of how often it couples. `apply_post`
reduces the full-step kinetic energy — the velocities after the
integrator's trailing kick, and, when the plan carries a terminal
`ConstraintPoint { AfterKick }` and a constraint slot is installed, after
those velocities have been projected onto the constraint manifold by
`Constraint::apply_after_kick` — and applies its rescale from it. The
projection must lead the reduction: the trailing kick leaves the
velocities off the manifold, and a thermostat that reduced their kinetic
energy would be coupling to energy the projection is about to delete.
That leading projection is also the step's **single publish** of the
constraint virial, and it must lead the terminal `BarostatPoint` for the
same structural reason: a per-step barostat reduces `buffers.virials`
inside its own `apply`, and a virial without the constraint contribution
is catastrophically wrong for rigid molecules. It therefore fires on
**every** step, not only on coupling steps. The plan's terminal
`ConstraintPoint { AfterKick }` is then a *repair* projection
(`Constraint::reproject_velocities_no_publish`), which does not
re-publish (see `constraint-framework.md`, *Ordering of the velocity
projections*).

Constraint placement is fully plan-declared: a constraint hook fires
only where the plan carries a `SubStep::ConstraintPoint` marker (see
`constraint-framework.md`). No structural inference is made from
`Drift` / `KickHalf` variants. A `ConstraintPoint` is a no-op when the
run has no constraint slot, so a constraint-capable integrator emits the
markers unconditionally and the same plan drives constrained and
unconstrained runs. A plan-final `ConstraintPoint { phase: AfterKick }`
velocity projection runs last in the post-force tail — after the
trailing kick and any thermostat and barostat rescale — so a velocity
projection is the last per-particle velocity operation of the step
(RATTLE-last).

Barostat placement is likewise plan-declared: a per-step barostat's
`apply` runs only where the plan carries a `SubStep::BarostatPoint`
marker, not at a fixed position. A `BarostatPoint` is a no-op when no
per-step barostat is configured, so an integrator that hosts a barostat
emits the marker unconditionally. `apply` performs the barostat's full
per-step work — the kinetic-energy and virial reductions, the
scale-factor and box-lattice update, and the per-particle
position/velocity rescale — in one call. A **terminal** `BarostatPoint`
(in the plan's trailing run of post-force markers, alongside any
`AfterKick`) is the canonical placement: `apply` runs in the post-force
tail, after the thermostat's `apply_post` and before the terminal
velocity projection. A **non-terminal** `BarostatPoint` (mid-plan) is
dispatched during the walk, where `apply` runs the same full work.

The plan's trailing run of the trailing kick, any `BarostatPoint`, and
any `ConstraintPoint { phase: AfterKick }` is the **post-force tail**.
`run_step` dispatches it as a fixed ordered sequence — trailing kick,
then `constraint.apply_after_kick` (the leading projection-and-publish,
for a plan-final velocity projection; unconditional), then
`thermostat.apply_post` (on a coupling step, for a wrapped thermostat),
then `barostat.apply` (for a terminal `BarostatPoint`), then
`constraint.reproject_velocities_no_publish` (the terminal repair
projection). `run_step` walks the whole plan, so the tail is part of that
walk, not a separate sequence the runner re-encodes.

The runner's timestep loop computes the cadence (it owns the step
counter) and delegates the whole step to `run_step`:

```text
loop step in 1..=n_steps:
    let plan = integrator.plan(dt)
    /* Cadence is the runner's concern: a wrapped thermostat couples
       every coupling_interval steps. dt_couple = coupling_interval · dt. */
    let couples = thermostat.is_some() && !plan.has_thermostat_points()
        && step % coupling_interval == 0
    let coupling_dt = if couples { Some(coupling_interval * dt) } else { None }
    run_step(integrator, buffers, sim_box, force_field,
             constraint, thermostat, barostat, dt, timings,
             RunStepOptions { coupling_dt, runner_needs_scalars, .. })
    /* A periodic (Monte-Carlo) barostat runs its host-orchestrated volume
       move at its own batch cadence, outside the dynamics step. */
    if barostat.periodicity() == EveryNSteps(f) and step % f == 0:
        barostat.apply_move(force_field, buffers, sim_box, constraint, dt, timings)
    ...trajectory / log output...
```

`run_step` walks the entire plan and dispatches the one canonical
per-step ordering — leading wrapped half, main walk (through the trailing
kick), trailing wrapped half, then the post-force marker tail:

```text
run_step(.., opts):
    let plan = integrator.plan(dt)
    let tail_start = plan.trailing_post_force_start()
    let wrap = opts.coupling_dt.is_some() && !plan.has_thermostat_points()
    /* leading wrapped half (coupling step, plan owns no ThermostatHalf) */
    if wrap: thermostat.apply_pre(buffers, opts.coupling_dt, timings)  /* on v(t) */
    /* main region: everything up to the trailing post-force markers,
       including the trailing kick and any interleaved thermostat /
       constraint / barostat markers */
    for sub in plan.steps[0 .. tail_start]:
        match sub:
            SubStep::ForceEval { class: None, level } =>
                force_field.step(buffers, sim_box, timings, resolve_level(level))
            SubStep::ForceEval { class: Some(c), level } =>
                force_field.step_class(c, buffers, sim_box, timings, resolve_level(level))
            SubStep::ThermostatHalf { dt: sub_dt, phase } =>
                /* plan-owned placement; no-op when thermostat is None */
                match phase: Pre => thermostat.apply_pre(...); Post => thermostat.apply_post(...)
            SubStep::ConstraintPoint { phase, dt: sub_dt } =>
                constraint.dispatch(phase, buffers, sim_box, sub_dt, timings)  /* no-op if None */
            SubStep::BarostatPoint { dt: sub_dt } =>
                barostat.apply(buffers, sim_box, sub_dt, timings)  /* interleaved: full apply */
            SubStep::KickHalf | KickDrift { source: Class(c), .. } =>
                class_kick*(buffers, force_field.class_forces(c), sub_dt)
            other =>
                integrator.execute(other, buffers, sim_box, timings)  /* incl. trailing kick */
    /* leading projection-and-publish: project the post-trailing-kick velocities
       onto the manifold and publish the constraint virial. Unconditional — the
       barostat below needs the constraint virial on every step, not only on
       coupling steps. Fires iff a constraint slot is installed and the plan has
       a terminal ConstraintPoint { AfterKick }. */
    if constraint.is_some() && plan has terminal AfterKick { dt: proj_dt }:
        constraint.apply_after_kick(buffers, sim_box, proj_dt, timings)
    /* trailing wrapped half (coupling step): full-step KE reduce + rescale,
       after the leading projection, before the terminal barostat */
    if wrap: thermostat.apply_post(buffers, opts.coupling_dt, timings)
    /* post-force marker tail: terminal BarostatPoint then terminal AfterKick */
    for sub in plan.steps[tail_start ..]:
        match sub:
            SubStep::BarostatPoint { dt: sub_dt } =>
                /* reduces buffers.virials — sees the constraint virial published above */
                barostat.apply(buffers, sim_box, sub_dt, timings)  /* no-op if None / periodic */
            SubStep::ConstraintPoint { phase: AfterKick, dt: sub_dt } =>
                /* repair projection (RATTLE-last); no virial publish. Falls back to
                   apply_after_kick when no leading projection ran. */
                constraint.reproject_velocities_no_publish(buffers, sim_box, sub_dt, timings)
```

`run_step` is parameterised by a `RunStepOptions` value (see *Feature
API*). The runner builds one `RunStepOptions` per step: `coupling_dt`
carries the coupling-step decision (above); `run_neighbor_pre_step`
toggles the `force_field.step_no_neighbor_check` path used during
CUDA-graph capture, and `runner_needs_scalars` forces the
scalar-aggregating force level. There is no `skip_substep_index` and no
deferral flag: `run_step` walks the whole plan and dispatches the
post-force tail itself, in canonical order.

The runner calls `integrator.plan(dt)` once per phase as a shape probe
(`has_thermostat_points()` selects the thermostat topology and CUDA-graph
eligibility; `has_barostat_points()` drives the barostat-placement
guard), and `run_step` calls it once per timestep to walk. `plan(dt)` is
a pure function of `dt` and the integrator's static configuration; it
returns the same `StepPlan` shape every call with the same `dt` (no
per-step branching on simulation state). Plans may contain zero or more
sub-steps; an empty plan is a no-op for that timestep.

`integrator.execute(sub, buffers, sim_box, timings)` runs one sub-step.
It receives no `&mut ForceField` because force evaluation is dispatched
by the runner, not the integrator. The integrator's per-sub-step
kernel launches bracket their own timings stages.

When `constraint` is `None`, or when the plan carries no
`ConstraintPoint` markers, no constraint hooks fire and the loop
reduces to a straight plan walk. Only constraint-capable integrators
emit `ConstraintPoint` markers, and the config loader rejects a
non-empty `[constraints]` section paired with an integrator whose
builder's `supports_constraints(&params)` returns `false` (see
`constraint-framework.md`), so a constraint slot is present only
alongside a marker-emitting plan. The integrator code never mentions
the constraint slot; it only places the markers.

The runner's `step` value is local to the timestep loop; it gates
trajectory and log writes via `step % trajectory_every == 0` and
`step % log_every == 0`, and is not visible to any slot. Slots that
need a monotone counter (for example a stochastic thermostat that
needs reproducible RNG draws) maintain their own counter on their
state and increment it on every invocation.

`thermostat.apply_pre()` and `thermostat.apply_post()` mutate
velocities (and read kinetic energy). They never touch positions, box,
or forces. Thermostats that only need post-step coupling leave
`apply_pre` at its default empty implementation.

`barostat.apply()` mutates positions and the simulation box, reading
virial / kinetic data from `buffers`. The integrator has already
populated `buffers.virials` and `buffers.forces_*` during its
in-step force evaluation; the barostat consumes those without
re-launching the force pipeline. For barostats that read virial every
step, the integrator's `ForceEval` sub-step must request
`AggregateLevel::ForcesAndScalars` (either explicitly via its
`level: Some(ForcesAndScalars)` or by deferring to the runner with
`level: None` on a step the runner upgrades). The mutated box is
observed by the next iteration's plan walk through the existing
`SimulationBox::generation()` change-detection path
(`forces/neighbor-list.md`, `forces/spme.md`).

`runner.resolve_level(sub_step_level: Option<AggregateLevel>) ->
AggregateLevel` upgrades the sub-step's request to
`AggregateLevel::ForcesAndScalars` whenever the runner needs the
scalar aggregates this step for its own purposes — specifically:

- the step writes a trajectory frame (`step % trajectory_every == 0`),
- the step writes a log row (`step % log_every == 0`),
- the step is a minimization iteration (the SD minimizer reads energy
  every iteration),
- or any output / observable subsystem indicates it requires energy
  or virial at this step.

Otherwise `resolve_level` returns `sub_step_level.unwrap_or(AggregateLevel::ForcesOnly)`.
An integrator that always requires scalars (e.g. MTK-NPT) emits
`level: Some(ForcesAndScalars)` and the runner's upgrade is a no-op
for that sub-step. An integrator that has no scalar requirement
emits `level: None` or `level: Some(ForcesOnly)` and the runner picks
the cheap level on steps that don't need scalars.

The runner performs one warm-up `force_field.step(..., AggregateLevel::ForcesAndScalars)`
call before entering the timestep loop so the first iteration's plan
walk reads valid `forces_*`, `potential_energies`, and `virials`.
Integrators that follow the symplectic-with-cached-F contract place a
`KickHalf` or `KickDrift` sub-step before the `ForceEval` so they
consume `F(t)`, and place their final velocity update (a `KickHalf`
or `KickDrift`) after the `ForceEval` so it consumes `F(t+dt)`. Every
integrator in the default registry follows this contract.

## Construction and Lifetime <!-- rq-5a1771b2 -->

The runner constructs all three slots after `init_device` returns and
immediately after the `Timings` instance is created. Construction
draws from one parsed `SlotConfig` for the integrator and optional
`SlotConfig`s for the thermostat and barostat (see
`io/config-schema.md`). Each `SlotConfig` carries a `kind: String`
naming one of the registered builders plus a `params: toml::Value`
holding the kind-specific parameters in raw form. The runner consults
the corresponding `IntegratorRegistry`, `ThermostatRegistry`, or
`BarostatRegistry`, looks up the builder by name, and calls its
`build` method to obtain a `Box<dyn Integrator>`,
`Option<Box<dyn Thermostat>>`, or `Option<Box<dyn Barostat>>`.
Per-particle device buffers (when an implementation needs them — for
example, `LosslessBuffers` for the lossless velocity-Verlet mode, or
`ke_scratch` for the kinetic-energy reduction used by every
thermostat) are allocated on the runner's `Arc<CudaDevice>` inside
each builder.

Every slot's allocations persist for the lifetime of the run and are
dropped together with the rest of the runner's GPU resources at end
of run. The runner never re-creates a slot mid-run, and no slot's
state ever crosses to another `Arc<CudaDevice>`.

## Compatibility Rules <!-- rq-9913daee -->

The compatibility predicates an integrator answers — `owns_thermostat`,
`owns_barostat`, `supports_constraints` — live on the
`IntegratorBuilder` trait and take the integrator's parsed
`toml::Value` parameters as input. The runner consults the registered
builder (looked up by `kind` name) and asks the predicate after
parsing the config and before constructing any GPU state.

- An integrator whose
  `IntegratorBuilder::owns_thermostat(&params)` returns `true` is
  incompatible with any configured `[thermostat]`. `load_config` (with
  registry access via `Config::validate_against`, see
  `io/config-schema.md`) returns
  `ConfigError::IncompatibleThermostat { integrator: <kind name> }`
  when the user configures both. `langevin-baoab` and `mtk-npt` are
  the integrators in the default registry that own their thermostat.
- An integrator whose
  `IntegratorBuilder::owns_barostat(&params)` returns `true` is
  incompatible with any configured `[barostat]`.
  `ConfigError::IncompatibleBarostat { integrator: <kind name> }`
  fires for the same reason. `mtk-npt` is the only integrator in the
  default registry that owns its barostat.
- An integrator whose
  `IntegratorBuilder::supports_constraints(&params)` returns `false`
  is incompatible with a non-empty topology `[constraints]` section
  (see `constraint-framework.md`).
- The thermostat slot is optional. When `[thermostat]` is omitted,
  the runner holds `None` and skips both `apply_pre` and `apply_post`
  hooks. This is how the user expresses NVE composition (or a
  self-thermostatted integrator standing alone).
- The barostat slot is optional. When `[barostat]` is omitted, the
  runner holds `None` and every `BarostatPoint` in the plan is a no-op.
  This is how the user expresses constant-volume composition.
- A per-step `[barostat]` requires the configured integrator's plan to
  carry at least one `BarostatPoint` marker; otherwise the barostat's
  `apply` would never fire. An integrator that does not own a barostat
  and hosts a per-step barostat therefore emits a `BarostatPoint`
  (`velocity-verlet` and `langevin-baoab` both emit a terminal one). A
  registry lint test asserts this for every built-in integrator that
  accepts a per-step barostat; the runner additionally guards at phase
  setup, returning `RunnerError::BarostatPlacementMissing { integrator:
  <kind name> }` when a per-step barostat is configured with an
  integrator whose plan has no `BarostatPoint`. (A periodic barostat
  couples through `apply_move` at batch boundaries and needs no
  `BarostatPoint`.)
- The `[thermostat]` and `[barostat]` slots accept at most one entry
  each per run. Composing multiple simultaneous thermostats or
  multiple simultaneous barostats is out of scope.

The predicates may depend on the slot's parsed `params`. For example,
`velocity-verlet`'s `supports_constraints` returns `true` when
`params.lossless == false` and `false` when `params.lossless == true`.
The builder is the single authority on these predicates because it is
the only component that understands its own parameter shape.

## Empty State <!-- rq-0bb735c9 -->

When the runner has `particle_count == 0`, every slot's hooks return
`Ok(())` without launching any kernel. Each slot's allocations (if
any) may have zero-length device slices but must construct
successfully.

## Feature API <!-- rq-6cd635cd -->

### Types <!-- rq-6c5b4246 -->

- `SubStep` — closed enum describing one piece of an integrator's <!-- rq-dbbffa7d -->
  per-timestep work. Variants:

  ```rust
  pub enum SubStep {
      /// Velocity half-kick: v ← v + (F/m) · dt/2 (or the
      /// integrator-private equivalent). No position update.
      /// `source` selects which force the kick consumes and who
      /// dispatches it (see `KickSource`).
      KickHalf { dt: f32, label: &'static str, source: KickSource },

      /// Position drift: x ← x + v · dt (or the integrator-private
      /// equivalent). No velocity update.
      Drift { dt: f32, label: &'static str },

      /// Fused KickHalf + Drift in a single kernel launch (e.g. the
      /// `vv_kick_drift` kernel for velocity-Verlet): the kick part
      /// uses `dt/2`, the drift part `dt`. `source` as on `KickHalf`.
      KickDrift { dt: f32, label: &'static str, source: KickSource },

      /// Dispatch the configured thermostat's pre- or post-half at
      /// this point in the plan, with the given `dt`. Dispatched by
      /// the runner, not by the integrator's `execute()`; a no-op
      /// when the run has no thermostat. A plan containing one or
      /// more `ThermostatHalf` sub-steps takes full ownership of
      /// thermostat placement: the runner's default wrapping
      /// (`apply_pre` before the walk, `apply_post` after) is
      /// suppressed for that plan.
      ThermostatHalf { dt: f32, phase: ThermostatPhase },

      /// Dispatch the configured constraint slot's hook at this point
      /// in the plan, with the given `dt`. Dispatched by the runner,
      /// not by the integrator's `execute()`; a no-op when the run has
      /// no constraint slot. `phase` selects which hook fires (see
      /// `ConstraintPhase`). Constraint-capable integrators place these
      /// markers to declare where the slot must snapshot or project;
      /// the runner performs no structural inference from the plan's
      /// kick / drift shape. `dt` is the sub-step timestep the hook
      /// operates over (the inner timestep for a multiple-timestep
      /// integrator, the full timestep for a single-step integrator),
      /// so a projection's velocity and virial factors use the correct
      /// interval.
      ConstraintPoint { phase: ConstraintPhase, dt: f32 },

      /// Dispatch the configured per-step barostat's `apply` at this
      /// point in the plan, with the given `dt`. Dispatched by the
      /// runner, not by the integrator's `execute()`; a no-op when no
      /// per-step barostat is configured (and inert for a periodic
      /// barostat, whose `apply` is the no-op default). An integrator
      /// that hosts a barostat places a single terminal `BarostatPoint`
      /// (the canonical placement — `run_step` fires `apply` in the
      /// post-force tail) or a mid-plan `BarostatPoint` for interleaved
      /// coupling (`run_step` fires `apply` during the walk). `dt` is the
      /// sub-step timestep the coupling operates over.
      BarostatPoint { dt: f32 },

      /// Force-pipeline evaluation. Dispatched by the runner, not by
      /// the integrator's `execute()`. The `class` field selects
      /// which force class(es) to re-evaluate (see
      /// `rqm/forces/framework.md`):
      ///   - `None` → runner calls `force_field.step(...)` (every
      ///     slot, every class).
      ///   - `Some(class)` → runner calls
      ///     `force_field.step_class(class, ...)` (only slots whose
      ///     `frequency_class() == class`).
      /// In both cases the combiner re-runs across every class so
      /// `ParticleBuffers.forces_*` always holds the latest total.
      ///
      /// `level` selects the aggregation level passed through to
      /// `ForceField::step` / `step_class`:
      ///   - `Some(ForcesAndScalars)` → integrator requires fresh
      ///     potential_energies and virials at this sub-step (e.g. NPT
      ///     barostats that read virial every step).
      ///   - `Some(ForcesOnly)` → integrator only needs forces;
      ///     potential_energies / virials may stay at their previous
      ///     value.
      ///   - `None` → integrator has no preference; the runner picks
      ///     based on its own needs (logging / minimization /
      ///     observable sampling on this step).
      /// The runner's `resolve_level` upgrades any sub-step request to
      /// `ForcesAndScalars` whenever it independently needs the
      /// scalars this step (e.g. an output frame is being written),
      /// so an integrator that emits `Some(ForcesOnly)` never causes
      /// stale scalars to leak into an output.
      ForceEval {
          class: Option<ForceClass>,
          level: Option<AggregateLevel>,
      },

      /// Integrator-private sub-step that doesn't fit the
      /// kick/drift/force triad (Langevin's OU step, MTK's chain or
      /// barostat sub-steps, kinetic-energy reductions for a
      /// barostat, etc.). `dt` carries the plan timestep so
      /// `execute()` can compute sub-step factors statelessly; the
      /// `label` lets `execute()` dispatch to the right kernel. `reads`
      /// and `writes` declare the sub-step's resource footprint for
      /// schedule dependency validation (see `op-model.md`); a built-in
      /// variant's footprint is fixed by its kind, but a `Custom`
      /// sub-step's is known only to the integrator that emits it.
      Custom { dt: f32, label: &'static str, reads: ResourceSet, writes: ResourceSet },
  }
  ```

- `KickSource` — selects the force a `KickHalf` / `KickDrift` <!-- rq-8fe78f4c -->
  consumes, and thereby its dispatcher:

  ```rust
  pub enum KickSource {
      /// The combined per-particle total force
      /// (`ParticleBuffers.forces_*`, as written by the class
      /// combiner). Dispatched to `integrator.execute()`; the
      /// integrator launches its own kick kernel.
      Total,

      /// A single class accumulator (`ForceField`'s
      /// `fast_total_forces_*` or `slow_total_forces_*` — see
      /// `rqm/forces/framework.md`, *Class Output Accumulators*).
      /// Dispatched by the runner, not by `execute()`: only the
      /// runner holds the `ForceField`, so it launches the
      /// framework-owned `class_kick_half` / `class_kick_drift`
      /// kernels with the selected class's force buffers. This is
      /// the kick form used by impulse-splitting multiple-timestep
      /// integrators (RESPA, `respa.md`), whose inner kicks consume
      /// only fast-class forces and outer kicks only slow-class
      /// forces.
      Class(ForceClass),
  }
  ```

- `ThermostatPhase` — which thermostat half a `ThermostatHalf` <!-- rq-ab6c5844 -->
  sub-step dispatches:

  ```rust
  pub enum ThermostatPhase {
      /// `thermostat.apply_pre(buffers, dt, timings)`.
      Pre,
      /// `thermostat.apply_post(buffers, dt, timings)`.
      Post,
  }
  ```

- `ConstraintPhase` — which constraint hook a `ConstraintPoint` <!-- inline --> <!-- rq-eea8aa89 -->
  sub-step dispatches. Each variant maps to one `Constraint` trait
  method (see `constraint-framework.md`):

  ```rust
  pub enum ConstraintPhase {
      /// `constraint.apply_before_drift(buffers, sim_box, dt, timings)`
      /// — snapshot the pre-drift positions. Placed immediately before
      /// a position-updating sub-step.
      BeforeDrift,
      /// `constraint.apply_after_drift(buffers, sim_box, dt, timings)`
      /// — project positions onto the constraint manifold and update
      /// the corresponding half-step velocities. Placed immediately
      /// after a position-updating sub-step.
      AfterDrift,
      /// Project velocities onto the constraint manifold. Placed after
      /// a velocity-updating sub-step. A plan-final `AfterKick` marker
      /// is fired by `run_step` in the post-force tail, after the
      /// trailing kick and any thermostat / barostat rescale, so it is
      /// the last per-particle velocity operation; it dispatches
      /// `constraint.reproject_velocities_no_publish(...)`, the repair
      /// projection, because `run_step` has already fired
      /// `constraint.apply_after_kick(...)` — project *and publish the
      /// constraint virial* — immediately after the trailing kick, so
      /// that the terminal `BarostatPoint`'s virial reduction sees it.
      /// A non-terminal `AfterKick` marker (or one with no leading
      /// projection) dispatches `apply_after_kick` itself (see *Per-Step
      /// Interface*).
      AfterKick,
  }
  ```

  - `label` on every variant that carries one is integrator-private
    and exists for debugging, timings stage selection, and (for
    `Custom`) dispatch inside `execute()`. The runner does not
    interpret the label.
  - Constraint hook placement is driven entirely by `ConstraintPoint`
    markers; the runner does no structural inference from `Drift` /
    `KickHalf` / `KickDrift` variants. A `ConstraintPoint` is a no-op
    when the run has no constraint slot, exactly as `ThermostatHalf`
    is a no-op without a thermostat.
  - Single-step integrators (velocity-Verlet, Langevin BAOAB,
    NHC/CSVR/Andersen/Berendsen-paired plans) emit
    `ForceEval { class: None, level: Some(ForcesOnly) }` so the runner
    re-evaluates every slot at the cheap level. Integrators that
    require fresh scalars every step emit
    `ForceEval { class: None, level: Some(ForcesAndScalars) }` —
    MTK-NPT (its barostat reads virial every step) and the constant-
    pressure c-rescale integrator both fall in this group. An
    integrator that has no scalar requirement of its own emits
    `level: None` and defers entirely to the runner. The RESPA
    integrator (`respa.md`) emits
    `ForceEval { class: Some(Fast), level: None }` once per inner
    step and `ForceEval { class: Some(Slow), level: None }` once per
    outer step, together with `KickSource::Class`-sourced kicks.
  - `ForceClass` and `AggregateLevel` are both re-exported from
    `crate::forces` (see `rqm/forces/framework.md` for their
    definitions).

- `StepPlan` — ordered list of `SubStep`s describing one full <!-- rq-9fbba3be -->
  timestep. `Debug + Clone`.

  ```rust
  pub struct StepPlan {
      pub steps: Vec<SubStep>,
  }
  ```

  - `steps.len() == 0` is allowed and represents an integrator that
    does nothing this timestep. The runner walks an empty plan
    without launching any kernel.
  - The plan may contain zero, one, or more `ForceEval` sub-steps.
    Zero: forces stay at their previous value (suitable for inertial
    drift or analytic propagation). One: the standard symplectic
    pattern. More than one: multiple-timestep integrators (RESPA,
    `respa.md`) and predictor-corrector schemes; a RESPA plan is the
    inner loop unrolled, so the plan shape stays a pure function of
    `dt` and the integrator's static configuration.
  - `StepPlan::has_thermostat_points() -> bool` — `true` iff any
    sub-step is a `ThermostatHalf`. The runner consults this once
    per step to choose the thermostat topology (see *Per-Step
    Interface*) and at phase setup to exclude marker-bearing plans
    from CUDA-graph capture (a `ThermostatHalf` dispatches host-side
    thermostat arithmetic, which cannot be captured; such plans run
    on the eager path — see `cuda-graphs.md`).
  - The plan's **post-force tail** is the maximal trailing run of the
    `ConstraintPoint { phase: AfterKick }` and `BarostatPoint` sub-steps
    (the trailing kick, a non-marker, sits just before it in the main
    region). `run_step` dispatches this run in canonical order as the
    final part of its whole-plan walk (see *Per-Step Interface*).
  - `StepPlan::trailing_post_force_start() -> usize` — the index at which
    the post-force marker tail begins (the length of the maximal trailing
    run of `ConstraintPoint { phase: AfterKick }` and `BarostatPoint`
    subtracted from `steps.len()`; equals `steps.len()` when the last
    sub-step is not a post-force marker). `run_step` walks
    `steps[0..start]`, then fires the wrapped thermostat's `apply_post`,
    then walks `steps[start..]` (the terminal barostat / projection).
  - `StepPlan::has_barostat_points() -> bool` — `true` iff any sub-step
    is a `BarostatPoint`. The runner consults it at phase setup for the
    barostat-placement guard (a per-step barostat paired with an
    integrator whose plan carries no `BarostatPoint` is rejected). A plan
    with an interleaved (non-terminal) `BarostatPoint` runs on the eager
    path — the mid-walk barostat arithmetic cannot be captured (see
    `cuda-graphs.md`).

- `RunStepOptions` — per-call options for `run_step`. Plain `Copy` <!-- rq-1d366b88 -->
  data; a caller overrides individual fields against `Default`.

  ```rust
  #[derive(Debug, Clone, Copy)]
  pub struct RunStepOptions {
      pub run_neighbor_pre_step: bool,
      pub runner_needs_scalars: bool,
      pub coupling_dt: Option<Real>,
  }
  ```

  - `run_neighbor_pre_step` — `true` makes each `ForceEval` sub-step
    call `force_field.step(...)` (which runs the neighbour-list
    pre-step); `false` makes it call
    `force_field.step_no_neighbor_check(...)`, used by the CUDA-graph
    capture path where the neighbour pre-step runs at batch boundaries
    (see `cuda-graphs.md`).
  - `runner_needs_scalars` — `true` resolves every `ForceEval` to
    `AggregateLevel::ForcesAndScalars` regardless of the sub-step's own
    preference (see `resolve_aggregate_level`).
  - `coupling_dt` — the coupling-step decision for a runner-wrapped
    thermostat. `Some(dt_couple)` marks this step as a coupling step:
    `run_step` fires the wrapped thermostat's `apply_pre` before the walk
    and `apply_post` at the post-force-marker boundary, both with
    `dt_couple`. `None` leaves the wrapped thermostat inert this step.
    The runner computes it (`coupling_interval · dt` on a coupling step,
    else `None`); it owns the step counter, `run_step` owns the ordering.
    It applies only to the wrapped topology — `run_step` ignores it when
    the plan carries `ThermostatHalf` markers (those are dispatched
    during the walk) — and only when a thermostat slot is passed.

  `run_step` walks the **whole** plan. It dispatches every interleaved
  `ConstraintPoint` and `BarostatPoint` it encounters, the trailing kick,
  the leading `constraint.apply_after_kick` projection-and-publish at the
  post-force-marker boundary, and the post-force tail (a terminal
  `BarostatPoint`, then a plan-final `ConstraintPoint { phase: AfterKick }`
  dispatched as the repair projection), each a no-op when the
  corresponding slot is absent. There is no skip or deferral flag.

  `RunStepOptions::default()` is
  `{ run_neighbor_pre_step: true, runner_needs_scalars: false, coupling_dt: None }`.

- `Integrator` — object-safe trait implemented by every concrete <!-- rq-78f484d9 -->
  integrator. Owns the core time-stepping algorithm.

  ```rust
  pub trait Integrator: std::fmt::Debug + Send {
      /// Return the ordered sequence of sub-steps that constitute one
      /// timestep of size `dt`. Pure: must return the same shape for
      /// the same `dt` and the same integrator state across calls.
      fn plan(&self, dt: f32) -> StepPlan;

      /// Execute one sub-step from this integrator's plan. Receives
      /// every sub-step except `SubStep::ForceEval` (which the runner
      /// dispatches directly to the force field).
      fn execute(
          &mut self,
          substep: &SubStep,
          buffers: &mut ParticleBuffers,
          sim_box: &mut SimulationBox,
          timings: &mut Timings,
      ) -> Result<(), IntegratorError>;

      /// Diagnostic column names and physical dimensions this
      /// integrator wants the runner to include in the CSV log
      /// (`io/log-output.md`). Each entry is a `(name, Dimension)`
      /// pair. The writer applies the output-direction conversion to
      /// the f64 value of each extra column on every row, using the
      /// declared dimension. Columns that carry pure ratios or other
      /// already-normalized values declare `Dimension::Dimensionless`
      /// and pass through unchanged. Returned slice has `'static`
      /// lifetime so the runner can pass it to `LogWriter::open`
      /// without copying. Default: empty.
      fn log_column_names(&self) -> &'static [(&'static str, Dimension)] { &[] }

      /// Current values of those columns. The runner supplies the
      /// total kinetic and potential energies it has just computed
      /// for the log row (in Hartrees, the engine's atomic energy
      /// unit; output-direction conversion happens later in
      /// `LogWriter::write_row`). The integrator combines them with
      /// its own state to produce the requested values, themselves
      /// in atomic units of the dimension declared by
      /// `log_column_names()`. The returned `Vec` must have the same
      /// length as `log_column_names()`. Default: empty.
      fn log_column_values(
          &self,
          kinetic_energy: f64,
          potential_energy: f64,
      ) -> Vec<f64> { Vec::new() }
  }
  ```

  - `plan(dt)` is called once per timestep by the runner, plus once
    per phase as a shape probe. It does
    no I/O, launches no kernels, and may not allocate per-particle
    GPU buffers (those are constructed once at slot construction).
  - `execute(sub, ...)` is called once per non-`ForceEval` sub-step,
    in plan order, by the runner. The integrator dispatches on
    `sub`'s variant and label to choose the right kernel. Sub-steps
    are independent of each other apart from their effect on
    `buffers` and the integrator's `&mut self`.
  - `execute()` is never called with `SubStep::ForceEval`; the runner
    dispatches force evaluation directly via `force_field.step(...)`.
    An integrator that places `ForceEval` in its plan but receives a
    `ForceEval` in `execute()` (e.g. due to misuse) should return an
    `IntegratorError` describing the misuse; conforming runners
    never produce this call.
  - `sim_box` is passed mutably to `execute()` so future integrators
    that mutate the box during the step (integrated barostats) can
    do so. Integrators that do not mutate the box leave it
    unchanged.
  - On a successful return from the runner's plan walk,
    `buffers.forces_*` holds `F` evaluated at the post-step positions
    (so the next iteration's plan can begin with a `KickHalf` /
    `KickDrift` that reads `F(t)`).
  - `plan(dt)` returns an empty plan when the integrator has nothing
    to do for that timestep; this is the canonical way to express a
    no-op step.

- `Thermostat` — object-safe trait implemented by every concrete <!-- rq-5d9ed248 -->
  thermostat.

  ```rust
  pub trait Thermostat: std::fmt::Debug + Send {
      /// Apply the thermostat's pre-step modification (typically a
      /// Trotter half-step). Mutates velocities; never touches
      /// positions, box, or forces. Default: no-op.
      fn apply_pre(
          &mut self,
          buffers: &mut ParticleBuffers,
          dt: f32,
          timings: &mut Timings,
      ) -> Result<(), ThermostatError> { Ok(()) }

      /// Apply the thermostat's post-step modification: reduce the
      /// full-step kinetic energy of the current (post-trailing-kick)
      /// velocities and rescale from it. Mutates velocities; never
      /// touches positions, box, or forces. Runs its own reduction and
      /// rescale kernels as standalone launches.
      fn apply_post(
          &mut self,
          buffers: &mut ParticleBuffers,
          dt: f32,
          timings: &mut Timings,
      ) -> Result<(), ThermostatError>;

      fn log_column_names(&self) -> &'static [&'static str] { &[] }
      fn log_column_values(
          &self,
          kinetic_energy: f64,
          potential_energy: f64,
      ) -> Vec<f64> { Vec::new() }
  }
  ```

  - `apply_pre` and `apply_post` are called only on coupling steps, and
    each receives the **effective coupling timestep** `dt_couple =
    coupling_interval · dt` (not the bare integrator `dt`). Thermostats
    that internally split this into half-steps do so themselves (NHC
    takes `dt_couple/2` for each side of its symmetric chain step;
    CSVR / Berendsen / Andersen use the full `dt_couple` in their
    relaxation formula and only act on the post side).
  - `apply_post` reduces the kinetic energy of the velocities as they
    stand when it is called — after the integrator's trailing kick, and
    after the terminal velocity projection when the run has a constraint
    slot — so the coupling always sees the full-step *physical* kinetic
    energy, the one whose degrees of freedom its target is built from. It
    performs that reduction and the subsequent rescale as its own
    standalone kernel launches.
  - The thermostat never reads or writes `sim_box`, `force_field`,
    or `buffers.forces_*` / `buffers.virials`.
  - `apply_pre` returns immediately when
    `buffers.particle_count() == 0`. So does `apply_post`.

- `Barostat` — object-safe trait implemented by every concrete <!-- rq-076617ab -->
  barostat.

  ```rust
  pub trait Barostat: std::fmt::Debug + Send {
      /// How often this barostat couples to the dynamics. A per-step
      /// barostat (the default) runs `apply` every step inside the
      /// captured sequence; a periodic barostat runs `apply_move`
      /// every `N` steps at a batch boundary and leaves `apply` a
      /// no-op. See `mc-barostat.md`.
      fn periodicity(&self) -> BarostatPeriodicity {
          BarostatPeriodicity::EveryStep
      }

      /// Perform the per-step barostat's full work in one call: the
      /// kinetic-energy and virial reductions, the scale-factor and
      /// box-lattice update, and the per-particle position/velocity
      /// rescale. Reads virial and kinetic data from `buffers` (already
      /// populated by the integrator's in-step force evaluation) and
      /// mutates `buffers.positions_*` / `buffers.velocities_*` and
      /// `sim_box`. Never launches the force pipeline directly. A
      /// periodic barostat leaves this at the no-op default.
      ///
      /// Dispatched at a `SubStep::BarostatPoint` marker (see
      /// *Per-Step Interface*), not at a fixed position. At a
      /// terminal (canonical) `BarostatPoint` `run_step` fires `apply`
      /// in the post-force tail, after the thermostat's `apply_post`;
      /// at an interleaved (mid-plan) `BarostatPoint` it fires during
      /// the walk. `apply` performs the same full work in both cases.
      fn apply(
          &mut self,
          buffers: &mut ParticleBuffers,
          sim_box: &mut SimulationBox,
          dt: f32,
          timings: &mut Timings,
      ) -> Result<(), BarostatError> { Ok(()) }

      /// Perform a periodic barostat's host-orchestrated move at a
      /// batch boundary. Unlike `apply`, it receives `&mut ForceField`
      /// because the move re-evaluates the potential energy at a trial
      /// configuration (e.g. a Monte-Carlo volume move). The default
      /// is a no-op; per-step barostats do not override it.
      fn apply_move(
          &mut self,
          force_field: &mut ForceField,
          buffers: &mut ParticleBuffers,
          sim_box: &mut SimulationBox,
          constraint: Option<&mut dyn Constraint>,
          dt: f32,
          timings: &mut Timings,
      ) -> Result<(), BarostatError> { Ok(()) }

      fn log_column_names(&self) -> &'static [&'static str] { &[] }
      fn log_column_values(
          &self,
          kinetic_energy: f64,
          potential_energy: f64,
      ) -> Vec<f64> { Vec::new() }
  }
  ```

  - `apply` and `apply_move` each return immediately when
    `buffers.particle_count() == 0`.
  - A barostat overrides exactly one of `apply` (per-step) or
    `apply_move` (periodic); the choice is consistent with the
    `periodicity()` it reports.
  - Mutating `sim_box` bumps its generation counter, which the next
    iteration's force pipeline observes via the existing
    change-detection path (`forces/neighbor-list.md`,
    `forces/spme.md`).

- `BarostatPeriodicity` — closed enum: `EveryStep` (per-step coupling) <!-- inline --> <!-- rq-343a8f18 -->
  or `EveryNSteps(u32)` (a host-orchestrated move every `N` steps at a
  batch boundary). The runner reads it to bound replay batches on the
  move cadence and to keep the per-step scalar requirement off for a
  periodic barostat (see `cuda-graphs.md`).

- `SlotConfig` — open-shaped parsed slot selection. Lives alongside <!-- rq-1f87880c -->
  the rest of the config types in `crate::io::config`. Its TOML
  mapping is defined in `io/config-schema.md`.

  ```rust
  pub struct SlotConfig {
      pub kind: String,
      pub params: toml::Value,
  }
  ```

  - `kind` is the registry lookup key (e.g. `"velocity-verlet"`,
    `"csvr"`, `"berendsen"`). It is the bridge between the open
    config layer and the open registries.
  - `params` carries every TOML field of the section other than
    `kind`, flattened into a `toml::Value`. Each registered builder
    deserialises its own typed parameter struct from this value (see
    `IntegratorBuilder::validate_params` and
    `IntegratorBuilder::build`); the framework never inspects
    `params` itself.

  The same `SlotConfig` shape is used by the `[integrator]`,
  `[thermostat]`, and `[barostat]` sections. The `[[constraint_types]]`
  array uses a closely related `NamedSlotConfig { name, kind, params }`
  shape that adds the type's user-facing name (see
  `io/config-schema.md`).

- `IntegratorBuilder`, `ThermostatBuilder`, `BarostatBuilder` — <!-- rq-29e08cb5 -->
  parallel traits describing a registered slot implementation.
  Implementations are stateless and self-register at construction
  time. Each builder owns its parameter shape: it deserialises a
  typed parameter struct from the `params: toml::Value` carried by
  the `SlotConfig`, validates per-kind constraints, exposes the
  compatibility predicates the runner needs, and constructs a boxed
  trait object on demand.

  ```rust
  pub trait IntegratorBuilder:
      KindedBuilder + IntegratorBuilderClone + std::fmt::Debug + Send + Sync
  {
      // `kind_name()` (the TOML `kind` lookup key) is inherited from
      // `KindedBuilder`; cloning from the generated `…BuilderClone` helper. See
      // `registry-framework.md`.

      /// Validate the kind-specific parameters at config-load time,
      /// before any GPU setup runs. Implementations deserialise the
      /// `toml::Value` into their typed parameter struct and surface
      /// every domain check (finite, positive, in-range, allowed
      /// enum value, …) as a `ConfigError::InvalidValue` (or one of
      /// the more specific `ConfigError` variants documented in
      /// `io/config-schema.md`).
      fn validate_params(&self, params: &toml::Value)
          -> Result<(), ConfigError>;

      /// `true` iff the integrator fuses its own thermostat (so
      /// composing it with a `[thermostat]` slot is rejected at
      /// load time). The default returns `false`.
      fn owns_thermostat(&self, _params: &toml::Value) -> bool { false }

      /// `true` iff the integrator fuses its own barostat. Default
      /// `false`.
      fn owns_barostat(&self, _params: &toml::Value) -> bool { false }

      /// `true` iff the integrator drives the three `Constraint`
      /// slot hooks (see `constraint-framework.md`). Default
      /// `false`.
      fn supports_constraints(&self, _params: &toml::Value) -> bool { false }

      /// `true` iff every per-step entry point (`step`, `execute`,
      /// and any sub-step hook surfaced through `Plan`) consists of
      /// pure CUDA kernel launches with no host-side state mutation
      /// between launches and no `dtoh_sync_copy` / `htod_sync_copy`
      /// calls. Determines whether phases driven by this integrator
      /// run under CUDA graph mode; see `cuda-graphs.md`. Default
      /// `true`; integrators that read device scalars into host
      /// fields between sub-steps (e.g. `mtk-npt`) override to
      /// `false`.
      fn graph_compatible(&self, _params: &toml::Value) -> bool { true }

      /// Construct the integrator. The caller has already invoked
      /// `validate_params(&params)`, so the builder may unwrap
      /// trusted fields; any failure inside `build` surfaces as
      /// `IntegratorError::Gpu` (the typical case is a failed GPU
      /// allocation). `n_constraints` is the total holonomic
      /// constraint count of the run (sum of every constraint group's
      /// `constraint_count`, zero when the topology has no
      /// `[constraints]` section). Integrators that compute internal
      /// degrees-of-freedom counts (e.g. `mtk-npt`) consume it; others
      /// ignore it.
      fn build(
          &self,
          gpu: &GpuContext,
          particle_count: usize,
          n_constraints: usize,
          params: &toml::Value,
      ) -> Result<Box<dyn Integrator>, IntegratorError>;
  }

  pub trait ThermostatBuilder:
      KindedBuilder + ThermostatBuilderClone + std::fmt::Debug + Send + Sync
  {
      // `kind_name()` from `KindedBuilder`; cloning from the generated `…BuilderClone` helper.
      fn validate_params(&self, params: &toml::Value)
          -> Result<(), ConfigError>;

      /// Same contract as `IntegratorBuilder::graph_compatible`.
      /// Default `true`. `nose-hoover-chain` overrides to `false`.
      fn graph_compatible(&self, _params: &toml::Value) -> bool { true }

      /// `n_constraints` is the total holonomic constraint count of
      /// the run (sum of every constraint group's `constraint_count`,
      /// zero when the topology has no `[constraints]` section).
      /// Implementations use it to compute their thermostatted
      /// degrees-of-freedom count; see each thermostat's requirements
      /// file for the exact formula.
      fn build(
          &self,
          gpu: &GpuContext,
          particle_count: usize,
          n_constraints: usize,
          params: &toml::Value,
      ) -> Result<Box<dyn Thermostat>, ThermostatError>;
  }

  pub trait BarostatBuilder:
      KindedBuilder + BarostatBuilderClone + std::fmt::Debug + Send + Sync
  {
      // `kind_name()` from `KindedBuilder`; cloning from the generated `…BuilderClone` helper.
      fn validate_params(&self, params: &toml::Value)
          -> Result<(), ConfigError>;

      /// Same contract as `IntegratorBuilder::graph_compatible`.
      /// Default `true`.
      fn graph_compatible(&self, _params: &toml::Value) -> bool { true }

      /// `n_constraints` is the total holonomic constraint count of
      /// the run (sum of every constraint group's `constraint_count`,
      /// zero when the topology has no `[constraints]` section).
      /// Barostats that compute kinetic-pressure or internal
      /// degrees-of-freedom counts consume it; others ignore it.
      fn build(
          &self,
          gpu: &GpuContext,
          particle_count: usize,
          n_constraints: usize,
          params: &toml::Value,
      ) -> Result<Box<dyn Barostat>, BarostatError>;
  }
  ```

  - `kind_name` returns the registry's lookup key.
  - `validate_params` is a pure function of the supplied parameters
    and is called by `Config::validate_against` before any GPU work.
    It must not allocate device memory.
  - The integrator-specific predicates
    (`owns_thermostat`, `owns_barostat`, `supports_constraints`) are
    pure functions of the supplied parameters. Predicate
    implementations that depend on the params (e.g.
    `velocity-verlet`'s `supports_constraints` flipping on
    `lossless`) deserialise the relevant field on demand.
  - `build` constructs the concrete slot. The caller is responsible
    for having passed the same `params` through `validate_params`
    first; conforming registry helpers (`build_or_validate_first`
    below) chain the two calls.

- `IntegratorRegistry`, `ThermostatRegistry`, `BarostatRegistry` — <!-- rq-4901507f -->
  `Registry<dyn IntegratorBuilder>` / `Registry<dyn ThermostatBuilder>` /
  `Registry<dyn BarostatBuilder>` (the generic container; see
  `registry-framework.md`). All three are named-selection registries:
  their builder traits carry `KindedBuilder`, so the generic `lookup(kind)`,
  `with_builtins()`, `register`, `Clone`, and `Default` apply. Per-registry
  built-in rosters: the integrator registry carries a builder for every
  `kind` in the slot's "Slots" table; the thermostat registry carries
  `nose-hoover-chain`, `csvr`, `andersen`, `berendsen`; the barostat
  registry carries `berendsen`, `c-rescale`, and `monte-carlo`.

  Construction dispatch is subsystem-specific (the build inputs are
  integrator-side):
  - `Registry<dyn IntegratorBuilder>::build(&self, slot: &SlotConfig, gpu: &GpuContext, particle_count: usize, n_constraints: usize) -> Result<Box<dyn Integrator>, IntegratorError>`
    — looks up the builder whose `kind_name()` equals `slot.kind` and
    delegates `build(gpu, particle_count, n_constraints, &slot.params)`
    to it. Returns `IntegratorError::UnknownKind(slot.kind.clone())`
    when no builder matches. The runner also uses `lookup` directly to
    query compatibility predicates (`owns_thermostat`,
    `supports_constraints`) and to drive `validate_params`.
  - The thermostat and barostat registries expose
    `build_optional(&self, slot: Option<&SlotConfig>, gpu: &GpuContext, particle_count: usize, n_constraints: usize) -> Result<Option<Box<dyn Thermostat>>, ThermostatError>`
    (and the corresponding barostat variant): if `slot` is `None`,
    returns `Ok(None)` without consulting the builders; otherwise
    dispatches the same way as `build` and wraps the result in
    `Some(..)`.
  - The three integrator-side registries (plus `ConstraintRegistry`
    from `constraint-framework.md` and `PotentialRegistry` from
    `forces/framework.md`) are also reachable as fields of the
    runner-level `heddle_md::Registries` bundle. See
    `simulation-runner.md` for the bundle's constructors and
    convenience `register_*` methods. The inner registries can be
    constructed and composed independently of the bundle when
    callers want one-at-a-time control.

- `IntegratorError`, `ThermostatError`, `BarostatError` — error types <!-- rq-2ccf40de -->
  returned by the corresponding trait methods. Variants for each:
  - `Gpu(GpuError)` — CUDA driver / kernel-launch failure.
  - `Timings(TimingsError)` — CUDA event recording failure.
  - `UnknownKind(String)` — the registry has no builder for the
    requested `kind` name.
  - `UnexpectedSubStep { variant: &'static str }` — the integrator's
    `execute()` was called with a sub-step variant it does not
    handle (e.g., the integrator received `SubStep::ForceEval`,
    which the runner is supposed to dispatch directly). Only present
    on `IntegratorError`. Conforming runners never produce this.

  Force-field errors raised by `force_field.step(...)` surface to the
  runner as `RunnerError::ForceField(ForceFieldError)` directly; the
  call site is the runner's plan walk, not the integrator's
  `execute()`, so `IntegratorError` does not wrap `ForceFieldError`.

  The runner's `RunnerError` wraps all three via
  `RunnerError::Integrator(IntegratorError)`,
  `RunnerError::Thermostat(ThermostatError)`, and
  `RunnerError::Barostat(BarostatError)`, plus
  `RunnerError::Constraint(ConstraintError)` for constraint-slot
  construction failures and hook-invocation failures
  (`constraint-framework.md`). It additionally carries
  `RunnerError::BarostatPlacementMissing { integrator: String }`,
  returned at phase setup when a per-step barostat is configured with an
  integrator whose plan carries no `BarostatPoint` marker.

### Functions and methods <!-- rq-c8848b7f -->

- `IntegratorRegistry::lookup(&self, kind: &str) -> Option<&dyn IntegratorBuilder>` <!-- rq-24f6b8b9 -->
  - Returns the first registered builder whose `kind_name()` equals
    `kind`. The runner uses this to query the integrator's
    compatibility predicates (`owns_thermostat`,
    `owns_barostat`, `supports_constraints`) and to drive
    `validate_params` before any GPU work.

- `IntegratorRegistry::build(&self, slot: &SlotConfig, gpu: &GpuContext, particle_count: usize, n_constraints: usize) -> Result<Box<dyn Integrator>, IntegratorError>` <!-- rq-1e30bbf4 -->
  - Looks up the builder whose `kind_name()` equals `slot.kind` and
    delegates `build(gpu, particle_count, n_constraints, &slot.params)`.
  - Returns `IntegratorError::UnknownKind(slot.kind.clone())` when
    no builder matches.
  - The builder is responsible for kind-specific allocations:
    - For `velocity-verlet`, the builder reads the `lossless` bool
      from `params` and allocates `LosslessBuffers` when
      `lossless == true`.
    - For `langevin-baoab`, the builder reads `friction`,
      `temperature`, `seed` from `params` and captures them (no
      per-particle state allocations; Philox is counter-based —
      see `langevin-baoab.md`).
  - A `particle_count` of zero is permitted: any per-particle device
    allocations have length zero.

- `ThermostatRegistry::lookup(&self, kind: &str) -> Option<&dyn ThermostatBuilder>` <!-- rq-c44b25af -->
  - Parallel to `IntegratorRegistry::lookup`. Returns `None` when no
    registered builder matches.

- `ThermostatRegistry::build_optional(&self, slot: Option<&SlotConfig>, gpu: &GpuContext, particle_count: usize) -> Result<Option<Box<dyn Thermostat>>, ThermostatError>` <!-- rq-678c233d -->
  - When `slot` is `None`, returns `Ok(None)` without consulting the
    builders.
  - When `slot` is `Some`, looks up the builder whose `kind_name()`
    equals `slot.kind` and delegates
    `build(gpu, particle_count, n_constraints, &slot.params)`. Returns
    `ThermostatError::UnknownKind(slot.kind.clone())` when no builder
    matches.
  - Per-thermostat builder responsibilities are documented in each
    thermostat's requirements file (`nose-hoover-chain.md`,
    `csvr.md`, `andersen.md`, `berendsen.md`).
  - A `particle_count` of zero is permitted: any per-particle device
    allocations (such as `ke_scratch`) still allocate at their fixed
    length (length 1 for the scalar reductions).

- `BarostatRegistry::lookup(&self, kind: &str) -> Option<&dyn BarostatBuilder>` <!-- rq-acbb6d0e -->
  - Parallel to `IntegratorRegistry::lookup`.

- `BarostatRegistry::build_optional(&self, slot: Option<&SlotConfig>, gpu: &GpuContext, particle_count: usize, n_constraints: usize) -> Result<Option<Box<dyn Barostat>>, BarostatError>` <!-- rq-9548bc1a -->
  - When `slot` is `None`, returns `Ok(None)` without consulting the
    builders.
  - When `slot` is `Some`, looks up the builder whose `kind_name()`
    equals `slot.kind` and delegates
    `build(gpu, particle_count, n_constraints, &slot.params)`. Returns
    `BarostatError::UnknownKind(slot.kind.clone())` when no builder
    matches.
  - Per-barostat builder responsibilities are documented in each
    barostat's requirements file (`berendsen-barostat.md`,
    `c-rescale-barostat.md`).

- `Integrator::plan(&self, dt: f32) -> StepPlan` <!-- rq-aa68f468 -->
  - Returns the integrator's ordered sub-step sequence for a
    timestep of size `dt`. Pure: same `dt` and same integrator
    state must yield the same plan shape across calls. Allocates a
    short `Vec<SubStep>`; does not touch GPU buffers.
  - May return an empty plan; the runner walks it as a no-op.

- `Integrator::execute(&mut self, substep: &SubStep, buffers: &mut ParticleBuffers, sim_box: &mut SimulationBox, timings: &mut Timings) -> Result<(), IntegratorError>` <!-- rq-83e752cd -->
  - Executes one sub-step from the plan. Dispatches on `substep`'s
    variant and label to launch the appropriate kernel(s).
  - Never receives `SubStep::ForceEval { .. }` from a conforming
    runner regardless of the `class` payload; if it does, returns
    `IntegratorError::UnexpectedSubStep { variant: "ForceEval" }`.
  - Returns `Ok(())` without launching any kernel when
    `buffers.particle_count() == 0`.
  - `execute` is the sole launcher of the integrator's per-particle
    sub-steps, including the trailing kick; it must remain correct for
    every SubStep the plan contains.

- `run_step(integrator: &mut dyn Integrator, buffers: &mut ParticleBuffers, sim_box: &mut SimulationBox, force_field: &mut ForceField, constraint: Option<&mut dyn Constraint>, thermostat: Option<&mut dyn Thermostat>, barostat: Option<&mut dyn Barostat>, dt: f32, timings: &mut Timings, opts: RunStepOptions) -> Result<(), StepError>` <!-- rq-277dbeb2 -->
  - The single free-function executor of one timestep: it walks the
    integrator's **entire** plan, including the post-force tail, and is
    the only plan walker in the crate. Calls `integrator.plan(dt)`, then:
    1. When `opts.coupling_dt` is `Some(dt_couple)` and the plan carries
       no `ThermostatHalf` markers, fires `thermostat.apply_pre(buffers,
       dt_couple, timings)` (a no-op when `thermostat` is `None`).
    2. Walks the main region `steps[0..trailing_post_force_start()]`,
       dispatching each sub-step:
       - `SubStep::ForceEval` → `force_field.step{,_class}(...)` (or
         their `_no_neighbor_check` variants when
         `opts.run_neighbor_pre_step == false`).
       - `SubStep::KickHalf` / `KickDrift` with
         `source: KickSource::Class(c)` → the framework-owned
         `class_kick_half` / `class_kick_drift` launch helpers, reading
         class `c`'s accumulator buffers from `force_field`.
       - `SubStep::ThermostatHalf { dt, phase }` →
         `thermostat.apply_pre` / `apply_post` (plan-owned placement; a
         no-op when `thermostat` is `None`).
       - `SubStep::ConstraintPoint { phase, dt }` → the matching
         `Constraint` hook, passing the marker's `dt`. A no-op when
         `constraint` is `None`.
       - A non-terminal `SubStep::BarostatPoint { dt }` →
         `barostat.apply(buffers, sim_box, dt, timings)` (interleaved:
         full work in the walk). A no-op when `barostat` is `None` or
         periodic.
       - Every other sub-step (including `Total`-sourced kicks, i.e. the
         trailing kick) → `integrator.execute(...)`.
    3. At the post-force-marker boundary — after the trailing kick — fires
       `constraint.apply_after_kick(...)` with the terminal marker's `dt`,
       when a constraint slot is installed and the plan has a terminal
       `SubStep::ConstraintPoint { phase: AfterKick }`. This is the step's
       **leading** projection: it puts the post-kick velocities back on the
       constraint manifold *and* publishes the constraint virial into
       `buffers.virials`. It is **unconditional** — not gated on
       `opts.coupling_dt` — because the terminal `BarostatPoint` in step 5
       reduces `buffers.virials` on every step.
    4. When `opts.coupling_dt` is `Some(dt_couple)` and the plan carries
       no `ThermostatHalf` markers, fires `thermostat.apply_post(buffers,
       dt_couple, timings)` — after the leading projection, before the
       terminal barostat (a no-op when `thermostat` is `None`), so it
       reduces the on-manifold full-step kinetic energy.
    5. Walks the post-force marker tail `steps[trailing_post_force_start()..]`:
       a terminal `SubStep::BarostatPoint { dt }` → `barostat.apply(...)`
       (no-op when `barostat` is `None` or periodic), whose virial
       reduction therefore sees the constraint virial published in step 3;
       then a plan-final `SubStep::ConstraintPoint { phase: AfterKick, dt }`
       → `constraint.reproject_velocities_no_publish(...)` (no-op when
       `constraint` is `None`), the *repair* projection, which puts the
       velocities back on the manifold after the thermostat's and
       barostat's rescales without re-publishing the virial, so a velocity
       projection is the last per-particle velocity operation
       (RATTLE-last). When no leading projection ran in step 3 (no
       constraint slot), the marker would dispatch `apply_after_kick`
       itself.
  - The wrapped-thermostat halves (steps 1 and 4) fire only for the
    default topology (no `ThermostatHalf` markers). A plan that owns its
    thermostat placement receives no wrapping; its `ThermostatHalf`
    markers dispatch during the walk in step 2.
  - There is no skip or deferral flag and no separate runner-owned tail:
    the whole per-step ordering lives in `run_step`.
  - Each `ForceEval`'s aggregate level is
    `resolve_aggregate_level(sub_step_level, opts.runner_needs_scalars)`.
  - Returns `StepError` (see `constraint-framework.md`).

- `resolve_aggregate_level(sub_step_level: Option<AggregateLevel>, runner_needs_scalars: bool) -> AggregateLevel` <!-- rq-2cd403d5 -->
  - Returns `AggregateLevel::ForcesAndScalars` when `runner_needs_scalars`
    is `true` or the sub-step itself requested scalars; otherwise returns
    `sub_step_level.unwrap_or(AggregateLevel::ForcesOnly)`.

- `class_kick_half(buffers: &mut ParticleBuffers, class_forces: ClassForceViews<'_>, dt: Real) -> Result<(), GpuError>` <!-- rq-85102698 -->
  - Framework-owned launch helper for a class-sourced velocity
    half-kick: one thread per particle computing
    `v ← v + (F_c/m) · dt/2`, where `F_c` is the selected class
    accumulator (`ClassForceViews` bundles the three force-component
    views the runner extracts from `ForceField`'s
    `{fast,slow}_total_forces_{x,y,z}`). Deterministic: one thread
    per particle, no reductions, no atomics.

- `class_kick_drift(buffers: &mut ParticleBuffers, class_forces: ClassForceViews<'_>, dt: Real) -> Result<(), GpuError>` <!-- rq-e3317d5a -->
  - Fused class-sourced kick + drift, matching the `KickDrift`
    convention: `v ← v + (F_c/m) · dt/2` then `x ← x + v · dt`, with
    the same position wrapping and image-flag updates as
    `vv_kick_drift` (see `velocity-verlet.md`). Deterministic as
    above.
  - Both kernels live in `kernels/integrate.cu` alongside the
    velocity-Verlet kernels and are loaded through the standard
    `gpu_kernels!` manifest.

- `Thermostat::apply_pre(&mut self, buffers: &mut ParticleBuffers, dt: f32, timings: &mut Timings) -> Result<(), ThermostatError>` <!-- rq-2fe47a86 -->
  - Default implementation returns `Ok(())` without launching any
    kernel. Thermostats that do not need pre-step coupling (CSVR,
    Andersen, Berendsen) accept the default.
  - Called only on coupling steps, with `dt = coupling_interval · dt`.
  - Returns `Ok(())` without launching any kernel when
    `buffers.particle_count() == 0`.

- `Thermostat::apply_post(&mut self, buffers: &mut ParticleBuffers, dt: f32, timings: &mut Timings) -> Result<(), ThermostatError>` <!-- rq-7a124d43 -->
  - Every concrete `Thermostat` must implement this method.
  - Called only on coupling steps, after the integrator's trailing
    kick, with `dt = coupling_interval · dt`.
  - Returns `Ok(())` without launching any kernel when
    `buffers.particle_count() == 0`.
  - Runs **every** part of the thermostat's post-force work as its own
    standalone kernel launches — the full-step kinetic-energy reduction,
    the factor / chain computation, and the per-particle rescale or
    resample. Its reduction reads the post-trailing-kick velocities (a
    fusion barrier; `docs/architecture.md`). For CSVR: kinetic-energy
    reduction, sample-and-factor, and `rescale_velocities`. For NHC:
    chain integration and `rescale_velocities`. For Andersen: the
    `andersen_resample` kernel.

- `Barostat::apply(&mut self, buffers: &mut ParticleBuffers, sim_box: &mut SimulationBox, dt: f32, timings: &mut Timings) -> Result<(), BarostatError>` <!-- rq-1179e42f -->
  - Every concrete `Barostat` must implement this method.
  - Returns `Ok(())` without launching any kernel when
    `buffers.particle_count() == 0`.
  - Performs the barostat's full per-step work as standalone launches:
    the kinetic-energy and virial reductions, the scale-factor and
    box-lattice update, the injection-accumulator bookkeeping, and the
    per-particle position/velocity rescale. For c-rescale: virial +
    kinetic reduction, mu compute + lattice update, and the position
    rescale. For Berendsen barostat: the reductions, mu compute + box
    mutation, and the position rescale.

## Determinism Guarantees <!-- rq-a93a8dc4 -->

The framework preserves the project's bit-wise reproducibility
invariant under the same conditions each slot individually guarantees:

- All slot kernels and the force pipeline run on the default stream
  of the same `Arc<CudaDevice>` carried by `ParticleBuffers`. No
  additional streams are introduced.
- The outer dispatch order (`apply_pre`, the main walk, then the
  post-force tail: trailing kick, `apply_post`, barostat `apply`,
  terminal velocity projection) is fixed and identical across runs.
- The plan walk visits sub-steps in `StepPlan::steps` order; the plan
  is a pure function of `dt` and the integrator's static
  configuration, so identical inputs across runs produce identical
  plans and identical sub-step orderings.
- The trait surfaces carry no loop-position parameter; the runner's
  loop counter stays local to the runner. Slots that need a monotone,
  reproducible counter (such as `LangevinBaoab` for its RNG draws, or
  `Csvr` / `Andersen` for theirs) own it on their own state and
  increment it deterministically per invocation.
- Implementations that draw random numbers document the exact RNG
  scheme so two runs on the same GPU with the same seed produce
  byte-identical trajectories.
- Composing a deterministic integrator with a deterministic
  thermostat (NHC, Berendsen) and no barostat produces a deterministic
  combination. Composing with a stochastic thermostat (CSVR,
  Andersen) preserves the stochastic thermostat's bit-exact
  reproducibility under the same RNG seed.

## Out of Scope <!-- rq-8d904561 -->

- A user-supplied integrator / thermostat / barostat DSL (à la
  OpenMM's `CustomIntegrator`). New slot implementations are Rust
  source code that implement the corresponding trait and register a
  builder.
- Constrained and constant-pressure multiple-timestepping. The RESPA
  integrator (`respa.md`) covers NVE/NVT. The `ConstraintPoint` marker
  vocabulary can express RATTLE inside the inner loop (a
  velocity-projection marker after every inner kick), but RESPA does
  not emit those markers and rejects a `[constraints]` section at
  config validation; constrained RESPA and RESPA-NPT splittings are
  out of scope here (see `respa.md`, *Compatibility*).
- Constraint algorithms other than SETTLE. M-SHAKE, P-LINCS, and
  every other constraint algorithm are out of scope for this
  framework file; they share the `Constraint` slot defined in
  `constraint-framework.md` and arrive in their own feature files.
- Concrete barostat implementations. The trait, registry, and config
  schema slot exist; the default registry has no builders.
- Multiple simultaneous thermostats per run. The runner holds at most
  one `Box<dyn Thermostat>`.
- Multiple simultaneous barostats per run. The runner holds at most
  one `Box<dyn Barostat>`.
- Mid-run replacement of any slot. Each slot is fixed at construction
  and never replaced for the duration of a run.
- Dynamic loading of slot implementations from shared libraries.
- Barostat coupling timing. `coupling_interval` and the full-step
  kinetic-energy policy govern the thermostat slot only. A per-step
  barostat continues to sample its virial and kinetic energy in its own
  `apply`, and periodic barostats keep their own move cadence; the
  point at which a barostat samples kinetic energy for its pressure
  estimate is unchanged.

---

## Gherkin Scenarios <!-- rq-ee777124 -->

```gherkin
Feature: Pluggable integration framework

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called

  # --- Integrator construction ---

  @rq-444903e2
  Scenario: Construct velocity-Verlet (lossy) via the integrator registry
    Given an IntegratorRegistry::with_builtins()
    And a SlotConfig { kind: "velocity-verlet", params: { lossless: false } }
    When registry.build(&slot, &gpu, particle_count=4, n_constraints=0) is called
    Then it returns Ok(integrator)
    And integrator's underlying type implements `Integrator`

  @rq-7d4c470a
  Scenario: Construct velocity-Verlet (lossless) via the integrator registry
    Given an IntegratorRegistry::with_builtins()
    And a SlotConfig { kind: "velocity-verlet", params: { lossless: true } }
    When registry.build(&slot, &gpu, particle_count=4, n_constraints=0) is called
    Then it returns Ok(integrator)
    And the underlying integrator allocates LosslessBuffers with particle_count == 4

  @rq-706c4b80
  Scenario: Construct Langevin BAOAB via the integrator registry
    Given an IntegratorRegistry::with_builtins()
    And a SlotConfig { kind: "langevin-baoab",
      params: { friction: 1.0e12, temperature: 300.0, seed: 42 } }
    When registry.build(&slot, &gpu, particle_count=4, n_constraints=0) is called
    Then it returns Ok(integrator)

  @rq-b44769f1
  Scenario: Construct an integrator with particle_count = 0
    Given an IntegratorRegistry::with_builtins()
    And a SlotConfig { kind: "velocity-verlet", params: { lossless: true } }
    When registry.build(&slot, &gpu, particle_count=0, n_constraints=0) is called
    Then it returns Ok(integrator)
    And every per-particle device allocation has length 0

  @rq-5711d6ce
  Scenario: Empty integrator registry reports UnknownKind
    Given an empty IntegratorRegistry (no builders registered)
    And a SlotConfig { kind: "velocity-verlet", params: { lossless: false } }
    When registry.build(&slot, &gpu, particle_count=4, n_constraints=0) is called
    Then it returns Err(IntegratorError::UnknownKind("velocity-verlet"))

  @rq-79c53582
  Scenario: Unknown kind in a populated registry reports UnknownKind
    Given an IntegratorRegistry::with_builtins()
    And a SlotConfig { kind: "no-such-integrator", params: { } }
    When registry.build(&slot, &gpu, particle_count=4, n_constraints=0) is called
    Then it returns Err(IntegratorError::UnknownKind("no-such-integrator"))

  @rq-8fbdbc0c
  Scenario: lookup returns the builder for a registered kind
    Given an IntegratorRegistry::with_builtins()
    When registry.lookup("velocity-verlet") is called
    Then it returns Some(builder)
    And builder.kind_name() equals "velocity-verlet"

  @rq-e8adaa9c
  Scenario: lookup returns None for an unregistered kind
    Given an IntegratorRegistry::with_builtins()
    When registry.lookup("no-such-integrator") is called
    Then it returns None

  @rq-0d7ebeb6
  Scenario: Custom integrator builder is selectable
    Given an IntegratorRegistry::with_builtins()
    And a custom IntegratorBuilder whose kind_name() is "test-stub"
    When registry.register(custom_builder) is called
    Then registry.build(...) routes "test-stub" kind requests to the custom builder

  # --- Thermostat construction ---

  @rq-353da04c
  Scenario: Construct Nosé-Hoover chain via the thermostat registry
    Given a ThermostatRegistry::with_builtins()
    And a SlotConfig { kind: "nose-hoover-chain", params:
      { temperature: 300.0, tau: 1.0e-13,
        chain_length: 3, yoshida_order: 3, n_resp: 1 } }
    When registry.build_optional(Some(&slot), &gpu, particle_count=4) is called
    Then it returns Ok(Some(thermostat))

  @rq-69d2c5f5
  Scenario: Construct CSVR via the thermostat registry
    Given a ThermostatRegistry::with_builtins()
    And a SlotConfig { kind: "csvr", params:
      { temperature: 300.0, tau: 1.0e-13, seed: 42 } }
    When registry.build_optional(Some(&slot), &gpu, particle_count=4) is called
    Then it returns Ok(Some(thermostat))

  @rq-3396b95f
  Scenario: Construct Andersen via the thermostat registry
    Given a ThermostatRegistry::with_builtins()
    And a SlotConfig { kind: "andersen", params:
      { temperature: 300.0, collision_rate: 1.0e12, seed: 42 } }
    When registry.build_optional(Some(&slot), &gpu, particle_count=4) is called
    Then it returns Ok(Some(thermostat))

  @rq-a336b496
  Scenario: Construct Berendsen via the thermostat registry
    Given a ThermostatRegistry::with_builtins()
    And a SlotConfig { kind: "berendsen", params:
      { temperature: 300.0, tau: 1.0e-13 } }
    When registry.build_optional(Some(&slot), &gpu, particle_count=4) is called
    Then it returns Ok(Some(thermostat))

  @rq-fb3f2189
  Scenario: build_optional with None returns Ok(None)
    Given a ThermostatRegistry::with_builtins()
    When registry.build_optional(None, &gpu, particle_count=4) is called
    Then it returns Ok(None)
    And no builder is consulted

  @rq-6dffb17f
  Scenario: Empty thermostat registry reports UnknownKind
    Given an empty ThermostatRegistry (no builders registered)
    And a SlotConfig { kind: "berendsen", params:
      { temperature: 300.0, tau: 1.0e-13 } }
    When registry.build_optional(Some(&slot), &gpu, particle_count=4) is called
    Then it returns Err(ThermostatError::UnknownKind("berendsen"))

  # --- Barostat construction ---

  @rq-386e3288
  Scenario: BarostatRegistry::with_builtins() exposes the registered barostats
    Given a BarostatRegistry::with_builtins()
    Then the registry contains a builder whose kind_name() is "berendsen"
    And the registry contains a builder whose kind_name() is "c-rescale"

  @rq-82cdabba
  Scenario: build_optional with None returns Ok(None) on the barostat registry
    Given a BarostatRegistry::with_builtins()
    When registry.build_optional(None, device, particle_count=4) is called
    Then it returns Ok(None)

  # --- Per-step dispatch ---

  @rq-0a6a97f6
  Scenario: Dispatch loop calls thermostat.apply_pre, plan walk, thermostat.apply_post in that order
    Given a velocity-Verlet integrator
    And a Nosé-Hoover-chain thermostat
    And a recording wrapper that timestamps every trait call (apply_pre,
      apply_post, plan, every execute, force_field.step)
    When the runner executes one timestep
    Then the first recorded event is apply_pre
    And the last recorded event is apply_post
    And between apply_pre and apply_post the recorded sub-calls are exactly
      [plan, execute(KickDrift), force_field.step, execute(KickHalf)]
    And no barostat hook is recorded

  @rq-8fd4e3bf
  Scenario: plan walk on empty state is a no-op
    Given a ParticleBuffers with particle_count() == 0
    And any constructed Integrator
    When the runner walks integrator.plan(0.1) and dispatches each sub-step
    Then every execute(...) call returns Ok(())
    And no kernel launches are recorded for any call

  @rq-e60481e9
  Scenario: apply_post on empty state is a no-op
    Given a ParticleBuffers with particle_count() == 0
    And any constructed Thermostat
    When thermostat.apply_post(&mut buffers, dt=0.1, &mut timings) is called
    Then it returns Ok(())
    And no kernel launches are recorded for that call

  @rq-167867a2
  Scenario: Default apply_pre is a no-op for thermostats that don't override it
    Given a CSVR thermostat
    And a ParticleBuffers with particle_count == 4 and a snapshot of velocities
    When thermostat.apply_pre(&mut buffers, dt=0.1, &mut timings) is called
    Then it returns Ok(())
    And velocities are bit-identical to the snapshot
    And no kernel launches are recorded for that call

  @rq-aa624d38
  Scenario: Thermostat couples on the full-step (post-trailing-kick) kinetic energy
    Given a velocity-Verlet integrator and a CSVR thermostat with coupling_interval = 1
    And a recording wrapper that captures the kinetic energy apply_post reduces
    And a ParticleBuffers whose forces make the trailing kick change velocities
    When the runner executes one timestep
    Then the recorded kinetic energy equals the kinetic energy of the
      velocities after the trailing kick
    And it differs from the kinetic energy of the pre-trailing-kick
      (half-step) velocities

  @rq-6ec9d751
  Scenario: Thermostat is inert on a non-coupling step
    Given a velocity-Verlet integrator and a thermostat with coupling_interval = 4
    When the runner executes step 1 (1 % 4 != 0)
    Then neither apply_pre nor apply_post is called
    And velocities are unchanged by any thermostat action
    And the integrator's trailing kick still runs (via execute) in the post-force tail

  @rq-76963898
  Scenario: Thermostat couples on an interval-boundary step with the effective timestep
    Given a velocity-Verlet integrator and a thermostat with coupling_interval = 4 and base dt
    When the runner executes step 4 (4 % 4 == 0)
    Then apply_pre and apply_post are both called
    And each receives dt_couple = 4 * dt
    And the trailing kick runs before apply_post in the post-force tail

  @rq-f4d73396
  Scenario: Unit coupling interval couples every step
    Given a thermostat with coupling_interval = 1
    When the runner executes any step
    Then apply_pre and apply_post are called on that step
    And each receives dt_couple = dt

  # --- Wrapped-thermostat coupling through run_step.coupling_dt ---

  @rq-b7f9628d
  Scenario: coupling_dt Some fires the wrapped halves at their canonical positions
    Given a recording thermostat and a velocity-Verlet plan (no ThermostatHalf markers)
    When run_step is called with RunStepOptions { coupling_dt: Some(dt_couple), ..default }
    Then thermostat.apply_pre is called once before the main-region walk with dt_couple
    And thermostat.apply_post is called once after the trailing kick with dt_couple
    And no other apply_pre / apply_post call occurs in that run_step call

  @rq-888952cd
  Scenario: coupling_dt None leaves the wrapped thermostat inert
    Given a recording thermostat and a velocity-Verlet plan (no ThermostatHalf markers)
    When run_step is called with RunStepOptions { coupling_dt: None, ..default }
    Then neither apply_pre nor apply_post is called
    And the trailing kick still runs

  @rq-b7ef61f1
  Scenario: run_step ignores coupling_dt when the plan owns its thermostat
    Given a recording thermostat and a plan that contains ThermostatHalf markers
    When run_step is called with RunStepOptions { coupling_dt: Some(dt_couple), ..default }
    Then the wrapped apply_pre / apply_post are not fired by the coupling_dt path
    And the thermostat is dispatched only at its ThermostatHalf markers with their own dt

  @rq-38e9c3b5
  Scenario: coupling_dt Some is a no-op when no thermostat slot is passed
    Given a velocity-Verlet plan and thermostat = None
    When run_step is called with RunStepOptions { coupling_dt: Some(dt_couple), ..default }
    Then the step completes with Ok(()) and no thermostat method is invoked

  @rq-d3bd619e
  Scenario: Velocity-Verlet plan walk launches vv_kick_drift, force pipeline, and vv_kick
    Given a velocity-Verlet integrator (lossless=false) with particle_count=4
    And a snapshot of buffers.positions_x and buffers.velocities_x before the call
    When the runner walks integrator.plan(0.1)
    Then every dispatch returns Ok(())
    And positions_x differs from the snapshot
    And velocities_x differs from the snapshot
    And timings.finalize() reports count==1 for KernelStage::VV_KICK_DRIFT
    And timings.finalize() reports count==1 for KernelStage::VV_KICK

  @rq-17def001
  Scenario: Lossless velocity-Verlet uses the lossless kernels
    Given a velocity-Verlet integrator (lossless=true) with particle_count=4
    When the runner walks integrator.plan(0.1)
    Then timings.finalize() reports count==1 for KernelStage::VV_KICK_DRIFT_LOSSLESS
    And timings.finalize() reports count==1 for KernelStage::VV_KICK_LOSSLESS
    And KernelStage::VV_KICK_DRIFT and KernelStage::VV_KICK have count==0

  @rq-812e88d5
  Scenario: Force evaluation is dispatched by the runner, not the integrator
    Given a velocity-Verlet integrator and a ForceField with one LennardJones slot
    When the runner walks integrator.plan(0.1) once
    Then timings.finalize() reports count==1 for KernelStage::LJ_PAIR_FORCE
    And KernelStage::REDUCE_PAIR_FORCES has count==1
    And KernelStage::ACCUMULATE_FORCES has count==1
    And integrator.execute(...) is never called with SubStep::ForceEval { .. } regardless of the `class` payload

  # --- Plan structure ---

  @rq-94a67d95
  Scenario: plan(dt) returns the same StepPlan shape across repeated calls
    Given a velocity-Verlet integrator
    When integrator.plan(dt=0.1) is called twice
    Then both calls return StepPlans with identical variant + label sequences

  @rq-4300cafc
  Scenario: plan(dt) is pure; it does not launch kernels or touch buffers
    Given a velocity-Verlet integrator
    And a snapshot of buffers before the call
    When integrator.plan(0.1) is called
    Then buffers are byte-identical to the snapshot
    And no kernel launches are recorded

  @rq-384ed838
  Scenario: Empty plan walks as a no-op
    Given a stub integrator whose plan(dt) returns StepPlan { steps: vec![] }
    When the runner executes one timestep with this integrator
    Then no execute(...) call is made
    And no force_field.step(...) call is made
    And no kernel launches are recorded

  @rq-07ead62b
  Scenario: Plan with multiple ForceEval { class: None } sub-steps invokes force_field.step that many times
    Given a stub integrator whose plan(dt) returns
      [KickHalf, Drift, ForceEval { class: None }, KickHalf, Drift,
       ForceEval { class: None }, KickHalf]
    When the runner executes one timestep with this integrator
    Then force_field.step(...) is invoked exactly twice
    And force_field.step_class(...) is invoked exactly zero times

  @rq-d4d435c8
  Scenario: integrator.execute receiving ForceEval surfaces UnexpectedSubStep
    Given any concrete integrator
    When execute(&SubStep::ForceEval { class: None }, ...) is called directly (bypassing the runner)
    Then it returns Err(IntegratorError::UnexpectedSubStep { variant: "ForceEval" })

  @rq-751bbb3c
  Scenario: integrator.execute receiving ForceEval with a class also surfaces UnexpectedSubStep
    Given any concrete integrator
    When execute(&SubStep::ForceEval { class: Some(ForceClass::Fast) }, ...) is called directly
    Then it returns Err(IntegratorError::UnexpectedSubStep { variant: "ForceEval" })

  # --- Compatibility ---

  @rq-e9be025b
  Scenario: VelocityVerlet builder does not own its thermostat or its barostat
    Given the "velocity-verlet" builder from IntegratorRegistry::with_builtins()
    And params = { lossless: false }
    Then builder.owns_thermostat(&params) returns false
    And builder.owns_barostat(&params) returns false

  @rq-4dd5d2d0
  Scenario: LangevinBaoab builder owns its thermostat but not its barostat
    Given the "langevin-baoab" builder from IntegratorRegistry::with_builtins()
    And params = { friction: 1.0e12, temperature: 300.0, seed: 0 }
    Then builder.owns_thermostat(&params) returns true
    And builder.owns_barostat(&params) returns false

  @rq-95b66af0
  Scenario: MtkNpt builder owns both its thermostat and its barostat
    Given the "mtk-npt" builder from IntegratorRegistry::with_builtins()
    And params = { temperature: 85.0, pressure: 1.0e5,
      tau_t: 1.0e-13, tau_p: 1.0e-12,
      chain_length: 3, yoshida_order: 3, n_resp: 1 }
    Then builder.owns_thermostat(&params) returns true
    And builder.owns_barostat(&params) returns true

  @rq-7d37c707
  Scenario: VelocityVerlet builder's supports_constraints depends on the lossless flag
    Given the "velocity-verlet" builder from IntegratorRegistry::with_builtins()
    Then builder.supports_constraints(&{ lossless: false }) returns true
    And builder.supports_constraints(&{ lossless: true }) returns false

  @rq-084ba25b
  Scenario: Builder validate_params accepts a well-formed params object
    Given the "velocity-verlet" builder
    When builder.validate_params(&{ lossless: false }) is called
    Then it returns Ok(())

  @rq-cb52dec0
  Scenario: Builder validate_params rejects an out-of-domain field
    Given the "langevin-baoab" builder
    And params = { friction: -1.0, temperature: 300.0, seed: 1 }
    When builder.validate_params(&params) is called
    Then it returns Err(ConfigError::InvalidValue { field: "integrator.friction", .. })

  @rq-7a076bc9
  Scenario: Builder validate_params rejects an unknown field
    Given the "velocity-verlet" builder
    And params = { lossless: false, junk: true }
    When builder.validate_params(&params) is called
    Then it returns Err(ConfigError::Parse { .. })

  # --- RNG-using slot state ---

  @rq-009bbbdc
  Scenario: Two consecutive step() calls on a Langevin integrator produce different post-call velocities
    Given a Langevin-BAOAB integrator with seed=1, friction=1e12, temperature=300, particle_count=2
    When step() is called twice on the same buffers with identical inputs
    Then the two calls produce different post-call velocities
    (because the integrator's internal draw_counter advances between calls)

  @rq-b2d5886a
  Scenario: Two consecutive apply_post calls on a CSVR thermostat produce different post-call velocities
    Given a CSVR thermostat with seed=1, temperature=300, tau=1e-13, particle_count=4
    When apply_post is called twice on identical buffers
    Then the two calls produce different post-call velocities
    (because the thermostat's internal draw_counter advances between calls)

  # --- Determinism across two runs ---

  @rq-1b0504e7
  Scenario: Two independent runs of the same slot combination are byte-identical
    Given two runners constructed from identical
      (SlotConfig integrator, Option<SlotConfig> thermostat, Option<SlotConfig> barostat) tuples
    And two ParticleBuffers built from byte-identical ParticleStates
    When each runs N=10 timesteps with the same dt
    Then the two final ParticleStates agree byte-for-byte

  # --- AggregateLevel resolution ---

  @rq-5a7e597e
  Scenario: A symplectic integrator emits ForceEval with ForcesOnly by default
    Given a velocity-Verlet integrator built from its default registry
    When integrator.plan(dt) is called
    Then the returned StepPlan contains exactly one SubStep::ForceEval
    And that sub-step's `level` field equals Some(AggregateLevel::ForcesOnly)
    And the sub-step's `class` field equals None

  @rq-3a9cb990
  Scenario: MTK-NPT emits ForceEval with ForcesAndScalars
    Given an MTK-NPT integrator built from its default registry
    When integrator.plan(dt) is called
    Then the returned StepPlan contains exactly one SubStep::ForceEval
    And that sub-step's `level` field equals Some(AggregateLevel::ForcesAndScalars)

  @rq-9f551521
  Scenario: runner.resolve_level upgrades to ForcesAndScalars on a logging step
    Given a runner with log_every = 100
    And a SubStep::ForceEval with level = Some(AggregateLevel::ForcesOnly)
    When step % log_every == 0 holds at this iteration
    Then runner.resolve_level(level) returns AggregateLevel::ForcesAndScalars

  @rq-1ee2ef41
  Scenario: runner.resolve_level upgrades to ForcesAndScalars on a trajectory frame
    Given a runner with trajectory_every = 50
    And a SubStep::ForceEval with level = None
    When step % trajectory_every == 0 holds at this iteration
    Then runner.resolve_level(level) returns AggregateLevel::ForcesAndScalars

  @rq-5e5f48da
  Scenario: runner.resolve_level falls through to ForcesOnly when neither logging
    nor trajectory output is due
    Given a runner with log_every = 100 and trajectory_every = 50
    And a SubStep::ForceEval with level = Some(AggregateLevel::ForcesOnly)
    When step is not a multiple of either log_every or trajectory_every
      and no other observable subsystem requests scalars
    Then runner.resolve_level(level) returns AggregateLevel::ForcesOnly

  @rq-75a19aca
  Scenario: runner.resolve_level keeps ForcesAndScalars when the sub-step already requests it
    Given a SubStep::ForceEval with level = Some(AggregateLevel::ForcesAndScalars)
    When runner.resolve_level(level) is called
    Then it returns AggregateLevel::ForcesAndScalars regardless of step counters

  # --- run_step / RunStepOptions ---

  @rq-93c52ca4
  Scenario: RunStepOptions::default values
    When RunStepOptions::default() is constructed
    Then run_neighbor_pre_step is true
    And runner_needs_scalars is false
    And coupling_dt is None

  @rq-d0240417
  Scenario: run_step with default options walks every sub-step via the neighbour-checked force path
    Given an integrator whose plan has three sub-steps including one ForceEval
    When run_step is called with constraint = None and RunStepOptions::default()
    Then integrator.execute is invoked once for every non-ForceEval sub-step
    And the ForceEval dispatches force_field.step (the neighbour-checked variant), not step_no_neighbor_check

  @rq-5500596b
  Scenario: run_neighbor_pre_step = false uses the no-neighbor-check force path
    Given an integrator with one ForceEval sub-step
    When run_step is called with RunStepOptions { run_neighbor_pre_step: false, ..default }
    Then the ForceEval dispatches force_field.step_no_neighbor_check, not force_field.step

  @rq-d64bc1c6
  Scenario: The trailing kick runs via execute in run_step's main walk
    Given an integrator whose plan has a trailing Total-sourced KickHalf at index k
      followed only by post-force markers (or nothing)
    When run_step walks the plan
    Then integrator.execute is invoked once for the sub-step at index k
    And no composed post-force kernel is launched

  @rq-f34598ae
  Scenario: Constraint hooks fire in canonical order across the walk and the tail
    Given a recording constraint slot and an integrator whose plan is
      [ConstraintPoint { phase: BeforeDrift }, Drift, ConstraintPoint { phase: AfterDrift },
       ForceEval, KickHalf, ConstraintPoint { phase: AfterKick }]
    When run_step is called once with Some(constraint)
    Then the recorded hook order is exactly
      [apply_before_drift, apply_after_drift, apply_after_kick,
       reproject_velocities_no_publish]
    And apply_before_drift and apply_after_drift are dispatched in run_step's
      main-region walk, while apply_after_kick is dispatched at the post-force
      marker boundary immediately after the trailing kick and
      reproject_velocities_no_publish is dispatched by the plan's terminal
      AfterKick marker — all within the one run_step call
    When run_step is called once with the same plan and constraint = None
    Then no constraint hook fires and the step completes with Ok(())

  @rq-43025b6a
  Scenario: runner_needs_scalars forces ForcesAndScalars
    Given an integrator whose ForceEval sub-step requests level = Some(AggregateLevel::ForcesOnly)
    When run_step is called with RunStepOptions { runner_needs_scalars: true, ..default }
    Then the ForceEval is evaluated at AggregateLevel::ForcesAndScalars

  @rq-836404c9
  Scenario: run_step is the only plan-walk free function
    Given the public API of src/integrator/mod.rs
    Then the only free plan-walk function is run_step(.., opts: RunStepOptions)
    And no per-combination wrapper plan-walk functions exist

  # --- Plan-declared constraint points ---

  @rq-2e4f64b0
  Scenario: ConstraintPoint markers are no-ops when no constraint slot is configured
    Given a velocity-Verlet integrator whose plan contains ConstraintPoint markers
    When run_step walks the plan with constraint = None
    Then the walk completes with Ok(())
    And no constraint method is invoked

  @rq-62c54adc
  Scenario: A ConstraintPoint hook receives the marker's own dt
    Given a recording constraint slot
    And an integrator whose plan ends in ConstraintPoint { phase: AfterKick, dt: 0.5 }
    When the runner executes one timestep with Some(constraint)
    Then apply_after_kick is invoked with dt == 0.5

  @rq-195a6215
  Scenario: One run_step call produces the whole canonical post-force order
    Given a recording thermostat, a recording per-step barostat, and a
      recording constraint slot, all writing to one shared ordered log
    And an integrator whose plan ends in [KickHalf, BarostatPoint, ConstraintPoint { phase: AfterKick }]
    When run_step is called once with all three slots and
      RunStepOptions { coupling_dt: Some(dt_couple), ..default }
    Then the recorded order is exactly
      [apply_pre, trailing kick, apply_post, barostat.apply, apply_after_kick]
    And no operation runs outside that single run_step call (the runner
      dispatches no post-force tail of its own)

  # --- Plan-declared barostat points ---

  @rq-b167a309
  Scenario: A terminal BarostatPoint dispatches barostat.apply once in run_step's post-force tail
    Given a recording per-step barostat
    And a stub integrator whose plan is [KickHalf, BarostatPoint { dt }]
    When run_step is called once with Some(barostat)
    Then barostat.apply is recorded exactly once with the marker's dt,
      after the trailing kick, within that run_step call's post-force tail

  @rq-68061953
  Scenario: BarostatPoint is a no-op when no per-step barostat is configured
    Given a stub integrator whose plan contains a BarostatPoint
    When run_step walks the plan with barostat = None
    Then the walk completes with Ok(())
    And no barostat method is invoked

  @rq-63fe749a
  Scenario: BarostatPoint is inert for a periodic barostat
    Given a Monte-Carlo (periodic) barostat whose apply is the no-op default
    And a stub integrator whose plan contains a terminal BarostatPoint
    When the runner executes one timestep
    Then barostat.apply performs no work
    And the periodic move still fires through apply_move at its batch cadence

  @rq-f9d0621d
  Scenario: has_barostat_points and trailing_post_force_start reflect the plan
    Given a plan whose final sub-step is BarostatPoint { dt: 0.5 }
    Then plan.has_barostat_points() is true
    And plan.trailing_post_force_start() is less than plan.steps.len()
      (the terminal BarostatPoint is in the post-force tail)
    Given a plan with a BarostatPoint that is not in the trailing run
    Then plan.has_barostat_points() is true
    And plan.trailing_post_force_start() equals plan.steps.len()
      (no trailing post-force marker; the BarostatPoint is interleaved)

  @rq-e043b064
  Scenario: An interleaved BarostatPoint dispatches apply during the main-region walk
    Given a recording per-step barostat
    And a stub integrator whose plan is [BarostatPoint { dt }, ForceEval, KickHalf]
    When run_step is called once with Some(barostat)
    Then barostat.apply is invoked once, before the ForceEval (the point is
      interleaved, not in the post-force tail)

  @rq-1d9788b5
  Scenario: A per-step barostat with an integrator that emits no BarostatPoint is rejected at phase setup
    Given a stub integrator whose plan contains no BarostatPoint
    And a per-step barostat configured
    When the runner enters the phase
    Then it returns Err(RunnerError::BarostatPlacementMissing { integrator })

  # --- Class-sourced kicks ---

  @rq-cac5cc99
  Scenario: A Class-sourced KickHalf reads the class accumulator, not the combined force
    Given a ForceField whose Fast accumulator holds force (1, 0, 0) and Slow accumulator (2, 0, 0) for particle 0
    And an integrator whose plan is [KickHalf { dt, source: Class(Slow), .. }]
    When run_step walks the plan
    Then particle 0's velocity gains (2 / m) · dt/2 along x
    And integrator.execute is not called for that sub-step

  @rq-5edc2d8f
  Scenario: A Class-sourced KickDrift kicks with dt/2 and drifts with dt
    Given a Fast accumulator holding force (f, 0, 0) and a particle at rest at x0
    And an integrator whose plan is [KickDrift { dt, source: Class(Fast), .. }]
    When run_step walks the plan
    Then the particle's velocity is (f/m) · dt/2
    And its position is x0 + (f/m) · dt/2 · dt

  @rq-0047eebd
  Scenario: A Total-sourced kick is dispatched to integrator.execute unchanged
    Given an integrator whose plan is [KickHalf { dt, source: Total, .. }]
    When run_step walks the plan
    Then integrator.execute receives that sub-step

  # --- Plan-declared thermostat points ---

  @rq-2eac2f62
  Scenario: ThermostatHalf sub-steps dispatch the thermostat at the marker positions
    Given a recording thermostat and an integrator whose plan is
      [ThermostatHalf { phase: Pre, .. }, Drift { .. }, ThermostatHalf { phase: Post, .. }]
    When the runner executes one timestep
    Then the thermostat records exactly one apply_pre and one apply_post
    And the apply_pre precedes the Drift and the apply_post follows it

  @rq-8c0c385b
  Scenario: A marker-bearing plan suppresses the runner's default thermostat wrapping
    Given a recording thermostat and an integrator whose plan contains one ThermostatHalf { phase: Post, .. } and no Pre marker
    When the runner executes one timestep
    Then the thermostat records zero apply_pre calls and exactly one apply_post call

  @rq-177b7289
  Scenario: A marker-free plan keeps the default wrapping topology
    Given a recording thermostat and an integrator whose plan contains no ThermostatHalf
    When the runner executes one timestep
    Then the thermostat records exactly one apply_pre before the plan walk and one apply_post after it

  @rq-22755bc1
  Scenario: ThermostatHalf is a no-op when no thermostat is configured
    Given no thermostat slot and an integrator whose plan contains ThermostatHalf sub-steps
    When run_step walks the plan with thermostat = None
    Then the walk completes with Ok(())
    And no thermostat method is invoked

  @rq-72bb7b90
  Scenario: has_thermostat_points reflects the plan contents
    Given a plan with no ThermostatHalf sub-steps
    Then plan.has_thermostat_points() is false
    Given a plan containing one ThermostatHalf sub-step
    Then plan.has_thermostat_points() is true

```
