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
3. An optional `Barostat` — pressure coupling. Fires once per step,
   after the thermostat's post-step, so it consumes the freshest
   virial / kinetic-energy data and mutates the box for the next
   step's force evaluation.
4. An optional `Constraint` — holonomic constraint projection (rigid
   bonds, rigid groups). Driven by the runner during its walk of the
   integrator's `StepPlan` (see *Per-Step Interface*): the runner
   inserts the constraint's hooks at the canonical sub-step
   boundaries (before / after every `Drift` or `KickDrift`, and after
   the final velocity update). Integrators never reference the
   constraint slot. The slot, its trait, its data layout, and its
   compatibility rules are defined in `constraint-framework.md`; the
   v1 implementation (SETTLE for three-atom rigid water) lives in
   `settle.md`.

Each slot is independently registered and independently selectable
from TOML. Omitting `[thermostat]` selects NVE; omitting `[barostat]`
selects constant-volume; an empty (or absent) `[constraints]` section
of the topology file selects no constraints.

Some integrators own their own thermostat (the O step in Langevin
BAOAB *is* the Ornstein-Uhlenbeck thermostat); some additionally own
their own barostat (the MTK NPT integrator carries an extended-system
cell DOF and its own thermostat chains on both the particles and the
cell). Those integrators declare ownership through
`IntegratorKind::owns_thermostat()` and
`IntegratorKind::owns_barostat()`, and the config loader rejects
co-configured `[thermostat]` / `[barostat]` tables at load time. An
analogous predicate `IntegratorKind::supports_constraints()` gates
the constraint slot: integrators that do not drive the constraint
hooks are incompatible with a non-empty `[constraints]` topology
section. See `constraint-framework.md` for the full rule.

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

### Barostat slot <!-- rq-d898f1cd -->

The default registry exposes two barostats:

| `kind` value | Implementation                                                       | File                      |
| ------------ | -------------------------------------------------------------------- | ------------------------- |
| `berendsen`  | weak-coupling isotropic pressure coupling (equilibration only — not canonical) | `berendsen-barostat.md`   |
| `c-rescale`  | stochastic isotropic cell-rescaling (canonical NPT)                  | `c-rescale-barostat.md`   |

## Per-Step Interface <!-- rq-daadfc1a -->

An integrator describes its work as an ordered sequence of typed
*sub-steps* (a `StepPlan`) and exposes a method that executes one
sub-step at a time. The runner walks the plan, dispatching each
sub-step to either `integrator.execute(...)` or `force_field.step(...)`
depending on the sub-step's variant, and inserts the constraint slot's
hooks at the canonical sub-step boundaries (see `constraint-framework.md`
for the hook contract). Integrators never reference the constraint
slot directly.

The runner drives the timestep loop in a fixed pattern:

```text
loop step in 1..=n_steps:
    if let Some(t) = thermostat { t.apply_pre(buffers, dt, timings) }
    let plan = integrator.plan(dt)
    let install_hooks = constraint.is_some() && integrator_kind.supports_constraints()
    for (i, sub) in plan.steps.iter().enumerate():
        let is_drift = matches!(sub, SubStep::Drift{..} | SubStep::KickDrift{..})
        if install_hooks && is_drift {
            constraint.apply_before_drift(buffers, sim_box, dt, timings)
        }
        match sub:
            SubStep::ForceEval =>
                force_field.step(buffers, sim_box, timings)
            other =>
                integrator.execute(other, buffers, sim_box, timings)
        if install_hooks && is_drift {
            constraint.apply_after_drift(buffers, sim_box, dt, timings)
        }
    let last_is_kick = matches!(
        plan.steps.last(),
        Some(SubStep::KickHalf{..} | SubStep::KickDrift{..}))
    if install_hooks && last_is_kick {
        constraint.apply_after_kick(buffers, sim_box, dt, timings)
    }
    if let Some(t) = thermostat { t.apply_post(buffers, dt, timings) }
    if let Some(b) = barostat   { b.apply(buffers, sim_box, dt, timings) }
    ...trajectory / log output...
```

The runner calls `integrator.plan(dt)` once per timestep. `plan(dt)` is
a pure function of `dt` and the integrator's static configuration; it
returns the same `StepPlan` shape every call with the same `dt` (no
per-step branching on simulation state). Plans may contain zero or
more sub-steps; an empty plan is a no-op for that timestep.

`integrator.execute(sub, buffers, sim_box, timings)` runs one sub-step.
It receives no `&mut ForceField` because force evaluation is dispatched
by the runner, not the integrator. The integrator's per-sub-step
kernel launches bracket their own timings stages.

When `constraint` is `None`, when the integrator's
`IntegratorKind::supports_constraints()` returns `false`, or when the
plan has no Drift / KickDrift sub-steps, no constraint hooks fire and
the loop reduces to a straight plan walk. The integrator code never
mentions the constraint slot.

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
re-launching the force pipeline. The mutated box is observed by the
next iteration's plan walk through the existing
`SimulationBox::generation()` change-detection path
(`forces/neighbor-list.md`, `forces/spme.md`).

The runner performs one warm-up `force_field.step(...)` call before
entering the timestep loop so the first iteration's plan walk reads
valid `forces_*` and `virials`. Integrators that follow the
symplectic-with-cached-F contract place a `KickHalf` or `KickDrift`
sub-step before the `ForceEval` so they consume `F(t)`, and place
their final velocity update (a `KickHalf` or `KickDrift`) after the
`ForceEval` so it consumes `F(t+dt)`. Every integrator in the
default registry follows this contract.

## Construction and Lifetime <!-- rq-5a1771b2 -->

The runner constructs all three slots after `init_device` returns and
immediately after the `Timings` instance is created. Construction
draws from the parsed `IntegratorKind`, optional `ThermostatKind`,
and optional `BarostatKind` config variants; the runner queries an
`IntegratorRegistry`, `ThermostatRegistry`, and `BarostatRegistry`
with the parsed kind's name and payload to obtain
`Box<dyn Integrator>`, `Option<Box<dyn Thermostat>>`, and
`Option<Box<dyn Barostat>>`. Per-particle device buffers (when an
implementation needs them — for example, `LosslessBuffers` for the
lossless velocity-Verlet mode, or `ke_scratch` for the kinetic-energy
reduction used by every thermostat) are allocated on the runner's
`Arc<CudaDevice>` inside each builder.

Every slot's allocations persist for the lifetime of the run and are
dropped together with the rest of the runner's GPU resources at end
of run. The runner never re-creates a slot mid-run, and no slot's
state ever crosses to another `Arc<CudaDevice>`.

## Compatibility Rules <!-- rq-9913daee -->

- An integrator that owns its own thermostat
  (`IntegratorKind::owns_thermostat()` returns `true`) is incompatible
  with any configured `[thermostat]`. `load_config` returns
  `ConfigError::IncompatibleThermostat { integrator: <kind name> }`
  when the user configures both. `langevin-baoab` and `mtk-npt` are
  the integrators in the default registry that own their thermostat.
- An integrator that owns its own barostat
  (`IntegratorKind::owns_barostat()` returns `true`) is incompatible
  with any configured `[barostat]`. `load_config` returns
  `ConfigError::IncompatibleBarostat { integrator: <kind name> }`
  when the user configures both. `mtk-npt` is the only integrator
  in the default registry that owns its barostat.
- The thermostat slot is optional. When `[thermostat]` is omitted,
  the runner holds `None` and skips both `apply_pre` and `apply_post`
  hooks. This is how the user expresses NVE composition (or a
  self-thermostatted integrator standing alone).
- The barostat slot is optional. When `[barostat]` is omitted, the
  runner holds `None` and skips the `apply` hook. This is how the
  user expresses constant-volume composition.
- The `[thermostat]` and `[barostat]` slots accept at most one entry
  each per run. Composing multiple simultaneous thermostats or
  multiple simultaneous barostats is out of scope.

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
      KickHalf { dt: f32, label: &'static str },

      /// Position drift: x ← x + v · dt (or the integrator-private
      /// equivalent). No velocity update.
      Drift { dt: f32, label: &'static str },

      /// Fused KickHalf + Drift in a single kernel launch (e.g. the
      /// `vv_kick_drift` kernel for velocity-Verlet).
      KickDrift { dt: f32, label: &'static str },

      /// Full force-pipeline evaluation. Dispatched by the runner,
      /// not by the integrator's `execute()`; the runner calls
      /// `force_field.step(buffers, sim_box, timings)` directly.
      ForceEval,

      /// Integrator-private sub-step that doesn't fit the
      /// kick/drift/force triad (Langevin's OU step, MTK's chain or
      /// barostat sub-steps, kinetic-energy reductions for a
      /// barostat, etc.). The `label` lets the integrator's
      /// `execute()` dispatch to the right kernel.
      Custom { label: &'static str },
  }
  ```

  - `label` on every variant is integrator-private and exists for
    debugging, timings stage selection, and (for `Custom`) dispatch
    inside `execute()`. The runner does not interpret the label.
  - The constraint slot's hook insertion logic
    (`constraint-framework.md`) reads only the variant tag, not the
    label.

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
    pattern. More than one: predictor-corrector or future multi-step
    integrators.

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

      /// Diagnostic column names this integrator wants the runner to
      /// include in the CSV log (`io/log-output.md`). Returned slice
      /// has `'static` lifetime so the runner can pass it to
      /// `LogWriter::open` without copying. Default: empty.
      fn log_column_names(&self) -> &'static [&'static str] { &[] }

      /// Current values of those columns. The runner supplies the
      /// total kinetic and potential energies it has just computed
      /// for the log row (in joules); the integrator combines them
      /// with its own state to produce the requested values. The
      /// returned `Vec` must have the same length as
      /// `log_column_names()`. Default: empty.
      fn log_column_values(
          &self,
          kinetic_energy: f64,
          potential_energy: f64,
      ) -> Vec<f64> { Vec::new() }
  }
  ```

  - `plan(dt)` is called once per timestep by the runner. It does
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

      /// Apply the thermostat's post-step modification. Mutates
      /// velocities; never touches positions, box, or forces.
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

  - `apply_pre` and `apply_post` each receive the same `dt` the
    integrator received. Thermostats that internally split this into
    half-steps do so themselves (NHC takes `dt/2` for each side of
    its symmetric chain step; CSVR / Berendsen / Andersen use the
    full `dt` in their relaxation formula and only act on the post
    side).
  - The thermostat never reads or writes `sim_box`, `force_field`,
    or `buffers.forces_*` / `buffers.virials`.
  - `apply_pre` returns immediately when
    `buffers.particle_count() == 0`. So does `apply_post`.

- `Barostat` — object-safe trait implemented by every concrete <!-- rq-076617ab -->
  barostat.

  ```rust
  pub trait Barostat: std::fmt::Debug + Send {
      /// Apply the barostat's box / position rescale. Reads
      /// virial and kinetic data from `buffers` (already populated
      /// by the integrator's in-step force evaluation), mutates
      /// `buffers.positions_*` and `sim_box`. Never launches the
      /// force pipeline directly.
      fn apply(
          &mut self,
          buffers: &mut ParticleBuffers,
          sim_box: &mut SimulationBox,
          dt: f32,
          timings: &mut Timings,
      ) -> Result<(), BarostatError>;

      fn log_column_names(&self) -> &'static [&'static str] { &[] }
      fn log_column_values(
          &self,
          kinetic_energy: f64,
          potential_energy: f64,
      ) -> Vec<f64> { Vec::new() }
  }
  ```

  - `apply` returns immediately when
    `buffers.particle_count() == 0`.
  - Mutating `sim_box` bumps its generation counter, which the next
    iteration's force pipeline observes via the existing
    change-detection path (`forces/neighbor-list.md`,
    `forces/spme.md`).

- `IntegratorKind`, `ThermostatKind`, `BarostatKind` — tagged enums <!-- rq-1f87880c -->
  carrying the parsed config-level selection and per-kind parameters.
  All three live alongside the rest of the config types in
  `crate::io::config`; their variants, fields, and TOML mapping are
  defined in `io/config-schema.md`.

  Each enum exposes `name(&self) -> &'static str` returning the same
  string the corresponding registry uses as the lookup key. This is
  the bridge between the closed config-level enums and the open
  registries.

  `IntegratorKind` additionally exposes:

  ```rust
  pub fn owns_thermostat(&self) -> bool;
  pub fn owns_barostat(&self) -> bool;
  pub fn supports_constraints(&self) -> bool;
  ```

  `owns_thermostat` returns `true` for variants whose integrator
  fuses its own thermostat. `LangevinBaoab { .. }` and
  `MtkNpt { .. }` return `true`; `VelocityVerlet { .. }` returns
  `false`.

  `owns_barostat` returns `true` for variants whose integrator
  fuses its own barostat. `MtkNpt { .. }` returns `true`;
  `VelocityVerlet { .. }` and `LangevinBaoab { .. }` return `false`.

  `supports_constraints` returns `true` for variants whose
  integrator drives the three `Constraint` hooks (see
  `constraint-framework.md`). `VelocityVerlet { lossless: false }`
  returns `true`; every other variant in the default registry
  (including `VelocityVerlet { lossless: true }`, `LangevinBaoab`,
  and `MtkNpt`) returns `false`.

- `IntegratorBuilder`, `ThermostatBuilder`, `BarostatBuilder` — <!-- rq-29e08cb5 -->
  parallel traits describing a registered slot implementation.
  Implementations are stateless and self-register at construction
  time. Each builder produces a boxed trait object from the
  corresponding `*Kind` payload and the runner's device.

  ```rust
  pub trait IntegratorBuilder: std::fmt::Debug + Send + Sync {
      fn kind_name(&self) -> &'static str;
      fn build(
          &self,
          device: Arc<CudaDevice>,
          particle_count: usize,
          kind: &IntegratorKind,
      ) -> Result<Box<dyn Integrator>, IntegratorError>;
  }

  pub trait ThermostatBuilder: std::fmt::Debug + Send + Sync {
      fn kind_name(&self) -> &'static str;
      fn build(
          &self,
          device: Arc<CudaDevice>,
          particle_count: usize,
          kind: &ThermostatKind,
      ) -> Result<Box<dyn Thermostat>, ThermostatError>;
  }

  pub trait BarostatBuilder: std::fmt::Debug + Send + Sync {
      fn kind_name(&self) -> &'static str;
      fn build(
          &self,
          device: Arc<CudaDevice>,
          particle_count: usize,
          kind: &BarostatKind,
      ) -> Result<Box<dyn Barostat>, BarostatError>;
  }
  ```

  - `kind_name` returns the lookup key.
  - `build` inspects `kind` (typically by matching the relevant
    variant) and constructs the concrete slot. Returns
    `*Error::UnknownKind` if the variant does not match the expected
    one.

- `IntegratorRegistry`, `ThermostatRegistry`, `BarostatRegistry` — <!-- rq-4901507f -->
  parallel host-side registries of builders. Each holds:
  - `builders: Vec<Box<dyn *Builder>>`

  Methods (illustrated for the integrator registry; the thermostat
  and barostat registries follow the same shape with the
  corresponding types):

  - `IntegratorRegistry::with_builtins() -> IntegratorRegistry` —
    constructs a registry pre-populated with builders for every
    `kind` value in the slot's "Slots" table above.
    `ThermostatRegistry::with_builtins()` pre-populates
    `nose-hoover-chain`, `csvr`, `andersen`, `berendsen`.
    `BarostatRegistry::with_builtins()` pre-populates `berendsen`
    and `c-rescale`.
  - `IntegratorRegistry::register(&mut self, builder: Box<dyn
    IntegratorBuilder>)` — appends a builder. Two builders sharing
    the same `kind_name()` are not detected at registration; the
    lookup returns the first match.
  - `IntegratorRegistry::build(&self, kind: &IntegratorKind, device:
    Arc<CudaDevice>, particle_count: usize) -> Result<Box<dyn
    Integrator>, IntegratorError>` — looks up the builder whose
    `kind_name()` equals `kind.name()` and delegates to it. Returns
    `IntegratorError::UnknownKind(kind.name().to_string())` when no
    builder matches.
  - The thermostat and barostat registries expose
    `build_optional(&self, kind: Option<&ThermostatKind>, device,
    particle_count) -> Result<Option<Box<dyn Thermostat>>,
    ThermostatError>` (and the corresponding barostat variant): if
    `kind` is `None`, returns `Ok(None)` without consulting the
    builders; otherwise dispatches the same way as `build` and wraps
    the result in `Some(..)`.

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
  (`constraint-framework.md`).

### Functions and methods <!-- rq-c8848b7f -->

- `IntegratorRegistry::build(&self, kind: &IntegratorKind, device: Arc<CudaDevice>, particle_count: usize) -> Result<Box<dyn Integrator>, IntegratorError>` <!-- rq-24f6b8b9 -->
  - Looks up the builder whose `kind_name()` equals `kind.name()`.
    Delegates construction to that builder.
  - Returns `IntegratorError::UnknownKind(...)` when no builder
    matches.
  - The builder is responsible for kind-specific allocations:
    - For `VelocityVerlet { lossless }`, the builder constructs a
      `VelocityVerletState` that allocates `LosslessBuffers` when
      `lossless == true`.
    - For `LangevinBaoab { friction, temperature, seed }`, the
      builder constructs a `LangevinBaoabState` that captures the
      friction, temperature, and seed (no per-particle state
      allocations; Philox is counter-based — see `langevin-baoab.md`).
  - A `particle_count` of zero is permitted: any per-particle device
    allocations have length zero.

- `ThermostatRegistry::build_optional(&self, kind: Option<&ThermostatKind>, device: Arc<CudaDevice>, particle_count: usize) -> Result<Option<Box<dyn Thermostat>>, ThermostatError>` <!-- rq-678c233d -->
  - When `kind` is `None`, returns `Ok(None)` without consulting the
    builders.
  - When `kind` is `Some`, looks up the builder whose `kind_name()`
    equals `kind.name()` and delegates. Returns
    `ThermostatError::UnknownKind(...)` when no builder matches.
  - Per-thermostat builder responsibilities are documented in each
    thermostat's requirements file (`nose-hoover-chain.md`,
    `csvr.md`, `andersen.md`, `berendsen.md`).
  - A `particle_count` of zero is permitted: any per-particle device
    allocations (such as `ke_scratch`) still allocate at their fixed
    length (length 1 for the scalar reductions).

- `BarostatRegistry::build_optional(&self, kind: Option<&BarostatKind>, device: Arc<CudaDevice>, particle_count: usize) -> Result<Option<Box<dyn Barostat>>, BarostatError>` <!-- rq-9548bc1a -->
  - When `kind` is `None`, returns `Ok(None)` without consulting the
    builders.
  - When `kind` is `Some`, looks up the builder whose `kind_name()`
    equals `kind.name()` and delegates. Returns
    `BarostatError::UnknownKind(...)` when no builder matches.
  - Per-barostat builder responsibilities are documented in each
    barostat's requirements file (`berendsen-barostat.md`).

- `Integrator::plan(&self, dt: f32) -> StepPlan` <!-- rq-aa68f468 -->
  - Returns the integrator's ordered sub-step sequence for a
    timestep of size `dt`. Pure: same `dt` and same integrator
    state must yield the same plan shape across calls. Allocates a
    short `Vec<SubStep>`; does not touch GPU buffers.
  - May return an empty plan; the runner walks it as a no-op.

- `Integrator::execute(&mut self, substep: &SubStep, buffers: &mut ParticleBuffers, sim_box: &mut SimulationBox, timings: &mut Timings) -> Result<(), IntegratorError>` <!-- rq-83e752cd -->
  - Executes one sub-step from the plan. Dispatches on `substep`'s
    variant and label to launch the appropriate kernel(s).
  - Never receives `SubStep::ForceEval` from a conforming runner;
    if it does, returns
    `IntegratorError::UnexpectedSubStep { variant: "ForceEval" }`.
  - Returns `Ok(())` without launching any kernel when
    `buffers.particle_count() == 0`.

- `Thermostat::apply_pre(&mut self, buffers: &mut ParticleBuffers, dt: f32, timings: &mut Timings) -> Result<(), ThermostatError>` <!-- rq-2fe47a86 -->
  - Default implementation returns `Ok(())` without launching any
    kernel. Thermostats that do not need pre-step coupling (CSVR,
    Andersen, Berendsen) accept the default.
  - Returns `Ok(())` without launching any kernel when
    `buffers.particle_count() == 0`.

- `Thermostat::apply_post(&mut self, buffers: &mut ParticleBuffers, dt: f32, timings: &mut Timings) -> Result<(), ThermostatError>` <!-- rq-7a124d43 -->
  - Every concrete `Thermostat` must implement this method.
  - Returns `Ok(())` without launching any kernel when
    `buffers.particle_count() == 0`.

- `Barostat::apply(&mut self, buffers: &mut ParticleBuffers, sim_box: &mut SimulationBox, dt: f32, timings: &mut Timings) -> Result<(), BarostatError>` <!-- rq-1179e42f -->
  - Every concrete `Barostat` must implement this method.
  - Returns `Ok(())` without launching any kernel when
    `buffers.particle_count() == 0`.

## Determinism Guarantees <!-- rq-a93a8dc4 -->

The composed framework preserves the project's bit-wise reproducibility
invariant under the same conditions each slot individually guarantees:

- All slot kernels and the force pipeline run on the default stream
  of the same `Arc<CudaDevice>` carried by `ParticleBuffers`. No
  additional streams are introduced.
- The outer dispatch order (`apply_pre`, then the plan walk, then
  `apply_post`, then `apply`) is fixed and identical across runs.
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
- Multi-time-step / RESPA integrators that split forces by frequency.
  The integrator trait shape supports them (the integrator can call
  `force_field.step` multiple times per `step()` with different
  effective `dt`s), but no RESPA implementation ships in the default
  registry.
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
    And an IntegratorKind::VelocityVerlet { lossless: false }
    When registry.build(&kind, device, particle_count=4) is called
    Then it returns Ok(integrator)
    And integrator's underlying type implements `Integrator`

  @rq-7d4c470a
  Scenario: Construct velocity-Verlet (lossless) via the integrator registry
    Given an IntegratorRegistry::with_builtins()
    And an IntegratorKind::VelocityVerlet { lossless: true }
    When registry.build(&kind, device, particle_count=4) is called
    Then it returns Ok(integrator)
    And the underlying integrator allocates LosslessBuffers with particle_count == 4

  @rq-706c4b80
  Scenario: Construct Langevin BAOAB via the integrator registry
    Given an IntegratorRegistry::with_builtins()
    And an IntegratorKind::LangevinBaoab { friction: 1.0e12, temperature: 300.0, seed: 42 }
    When registry.build(&kind, device, particle_count=4) is called
    Then it returns Ok(integrator)

  @rq-b44769f1
  Scenario: Construct an integrator with particle_count = 0
    Given an IntegratorRegistry::with_builtins()
    And an IntegratorKind::VelocityVerlet { lossless: true }
    When registry.build(&kind, device, particle_count=0) is called
    Then it returns Ok(integrator)
    And every per-particle device allocation has length 0

  @rq-5711d6ce
  Scenario: Empty integrator registry reports UnknownKind
    Given an empty IntegratorRegistry (no builders registered)
    And an IntegratorKind::VelocityVerlet { lossless: false }
    When registry.build(&kind, device, particle_count=4) is called
    Then it returns Err(IntegratorError::UnknownKind("velocity-verlet"))

  @rq-0d7ebeb6
  Scenario: Custom integrator builder is selectable
    Given an IntegratorRegistry::with_builtins()
    And a custom IntegratorBuilder whose kind_name() is "test-stub"
    When registry.register(custom_builder) is called
    Then registry.build(...) routes test-stub-kind requests to the custom builder

  # --- Thermostat construction ---

  @rq-353da04c
  Scenario: Construct Nosé-Hoover chain via the thermostat registry
    Given a ThermostatRegistry::with_builtins()
    And a ThermostatKind::NoseHooverChain {
      temperature: 300.0, tau: 1.0e-13,
      chain_length: 3, yoshida_order: 3, n_resp: 1 }
    When registry.build_optional(Some(&kind), device, particle_count=4) is called
    Then it returns Ok(Some(thermostat))

  @rq-69d2c5f5
  Scenario: Construct CSVR via the thermostat registry
    Given a ThermostatRegistry::with_builtins()
    And a ThermostatKind::Csvr {
      temperature: 300.0, tau: 1.0e-13, seed: 42 }
    When registry.build_optional(Some(&kind), device, particle_count=4) is called
    Then it returns Ok(Some(thermostat))

  @rq-3396b95f
  Scenario: Construct Andersen via the thermostat registry
    Given a ThermostatRegistry::with_builtins()
    And a ThermostatKind::Andersen {
      temperature: 300.0, collision_rate: 1.0e12, seed: 42 }
    When registry.build_optional(Some(&kind), device, particle_count=4) is called
    Then it returns Ok(Some(thermostat))

  @rq-a336b496
  Scenario: Construct Berendsen via the thermostat registry
    Given a ThermostatRegistry::with_builtins()
    And a ThermostatKind::Berendsen { temperature: 300.0, tau: 1.0e-13 }
    When registry.build_optional(Some(&kind), device, particle_count=4) is called
    Then it returns Ok(Some(thermostat))

  @rq-fb3f2189
  Scenario: build_optional with None returns Ok(None)
    Given a ThermostatRegistry::with_builtins()
    When registry.build_optional(None, device, particle_count=4) is called
    Then it returns Ok(None)
    And no builder is consulted

  @rq-6dffb17f
  Scenario: Empty thermostat registry reports UnknownKind
    Given an empty ThermostatRegistry (no builders registered)
    And a ThermostatKind::Berendsen { temperature: 300.0, tau: 1.0e-13 }
    When registry.build_optional(Some(&kind), device, particle_count=4) is called
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
    And integrator.execute(...) is never called with SubStep::ForceEval

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
  Scenario: Plan with multiple ForceEvals invokes force_field.step that many times
    Given a stub integrator whose plan(dt) returns
      [KickHalf, Drift, ForceEval, KickHalf, Drift, ForceEval, KickHalf]
    When the runner executes one timestep with this integrator
    Then force_field.step(...) is invoked exactly twice

  @rq-d4d435c8
  Scenario: integrator.execute receiving ForceEval surfaces UnexpectedSubStep
    Given any concrete integrator
    When execute(&SubStep::ForceEval, ...) is called directly (bypassing the runner)
    Then it returns Err(IntegratorError::UnexpectedSubStep { variant: "ForceEval" })

  # --- Compatibility ---

  @rq-e9be025b
  Scenario: VelocityVerlet does not own its thermostat or its barostat
    Given an IntegratorKind::VelocityVerlet { lossless: false }
    Then kind.owns_thermostat() returns false
    And kind.owns_barostat() returns false

  @rq-4dd5d2d0
  Scenario: LangevinBaoab owns its thermostat but not its barostat
    Given an IntegratorKind::LangevinBaoab { friction: 1.0e12, temperature: 300.0, seed: 0 }
    Then kind.owns_thermostat() returns true
    And kind.owns_barostat() returns false

  @rq-95b66af0
  Scenario: MtkNpt owns both its thermostat and its barostat
    Given an IntegratorKind::MtkNpt { temperature: 85.0, pressure: 1.0e5,
      tau_t: 1.0e-13, tau_p: 1.0e-12,
      chain_length: 3, yoshida_order: 3, n_resp: 1 }
    Then kind.owns_thermostat() returns true
    And kind.owns_barostat() returns true

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
  Scenario: Two independent runs of the same composed configuration are byte-identical
    Given two runners constructed from identical (IntegratorKind, Option<ThermostatKind>, Option<BarostatKind>) tuples
    And two ParticleBuffers built from byte-identical ParticleStates
    When each runs N=10 timesteps with the same dt
    Then the two final ParticleStates agree byte-for-byte
```
