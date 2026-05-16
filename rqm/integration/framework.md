# Feature: Pluggable Integration Framework <!-- rq-e0a0553d -->

The runner drives time integration through three orthogonal slots that
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

Each slot is independently registered and independently selectable
from TOML. Omitting `[thermostat]` selects NVE; omitting `[barostat]`
selects constant-volume.

Some integrators own their own thermostat (the O step in Langevin
BAOAB *is* the Ornstein-Uhlenbeck thermostat). Those integrators
declare ownership through `IntegratorKind::owns_thermostat()` and the
config loader rejects co-configured `[thermostat]` at load time.

## Slots <!-- rq-f8bb021a -->

### Integrator slot <!-- rq-d8c0e5b0 -->

The default registry exposes two integrators:

| `kind` value      | Owns thermostat? | Implementation                                    | File                  |
| ----------------- | ---------------- | ------------------------------------------------- | --------------------- |
| `velocity-verlet` | no               | symplectic NVE (lossy or lossless)                | `velocity-verlet.md`  |
| `langevin-baoab`  | yes              | stochastic NVT via BAOAB splitting                | `langevin-baoab.md`   |

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

The runner drives the timestep loop in a fixed pattern:

```text
loop step in 1..=n_steps:
    if let Some(t) = thermostat { t.apply_pre(buffers, dt, timings) }
    integrator.step(buffers, sim_box, force_field, dt, timings)
    if let Some(t) = thermostat { t.apply_post(buffers, dt, timings) }
    if let Some(b) = barostat   { b.apply(buffers, sim_box, dt, timings) }
    ...trajectory / log output...
```

The runner's `step` value is local to the timestep loop; it gates
trajectory and log writes via `step % trajectory_every == 0` and
`step % log_every == 0`, and is not visible to any slot. Slots that
need a monotone counter (for example a stochastic thermostat that
needs reproducible RNG draws) maintain their own counter on their
state and increment it on every invocation.

`integrator.step()` is responsible for all velocity and position
updates within its splitting (the kicks, the drift, the in-step force
evaluation), plus any state the integrator itself owns. The
integrator calls `force_field.step(buffers, sim_box, timings)` at the
point(s) of its choice. For the symplectic and Leimkuhler–Matthews
integrators in the default registry this happens exactly once per
`step()`.

`thermostat.apply_pre()` and `thermostat.apply_post()` mutate
velocities (and read kinetic energy). They never touch positions, box,
or forces. Thermostats that only need post-step coupling leave
`apply_pre` at its default empty implementation.

`barostat.apply()` mutates positions and the simulation box, reading
virial / kinetic data from `buffers`. The integrator has already
populated `buffers.virials` and `buffers.forces_*` during its
in-step force evaluation; the barostat consumes those without
re-launching the force pipeline. The mutated box is observed by the
next iteration's `integrator.step()` through the existing
`SimulationBox::generation()` change-detection path
(`forces/neighbor-list.md`, `forces/spme.md`).

The runner performs one warm-up `force_field.step(...)` call before
entering the timestep loop so the first iteration's
`integrator.step()` reads valid `forces_*` and `virials`. Integrators
that follow the symplectic-with-cached-F contract assume
`buffers.forces_*` holds `F(t)` on entry and produce `F(t+dt)` on
exit; the current registry's integrators all follow this contract.

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
  when the user configures both. `langevin-baoab` is the only such
  integrator in the default registry.
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

- `Integrator` — object-safe trait implemented by every concrete <!-- rq-78f484d9 -->
  integrator. Owns the core time-stepping algorithm.

  ```rust
  pub trait Integrator: std::fmt::Debug + Send {
      fn step(
          &mut self,
          buffers: &mut ParticleBuffers,
          sim_box: &mut SimulationBox,
          force_field: &mut ForceField,
          dt: f32,
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

  - The integrator runs every sub-step the time-stepping algorithm
    requires (velocity kicks, position drifts, force evaluations).
    It calls `force_field.step(buffers, sim_box, timings)` at the
    appropriate point(s).
  - `sim_box` is passed mutably so future integrators that mutate the
    box during the step (e.g. integrated barostats) can do so.
    Integrators that do not mutate the box leave it unchanged.
  - On a successful return, `buffers.forces_*` holds `F` evaluated at
    the post-step positions (so the next iteration's pipeline can
    begin with a half-kick that reads `F(t)`).
  - Returns `Ok(())` immediately when `buffers.particle_count() == 0`.

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
  ```

  which returns `true` for variants whose integrator fuses its own
  thermostat. `LangevinBaoab { .. }` returns `true`; `VelocityVerlet
  { .. }` returns `false`.

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
  - `ForceField(ForceFieldError)` — surfaces failures from
    `force_field.step(...)` (only the integrator triggers force
    evaluations; the variant is therefore present only on
    `IntegratorError`).
  - `UnknownKind(String)` — the registry has no builder for the
    requested `kind` name.

  The runner's `RunnerError` wraps all three via
  `RunnerError::Integrator(IntegratorError)`,
  `RunnerError::Thermostat(ThermostatError)`, and
  `RunnerError::Barostat(BarostatError)`.

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

- `Integrator::step(&mut self, buffers: &mut ParticleBuffers, sim_box: &mut SimulationBox, force_field: &mut ForceField, dt: f32, timings: &mut Timings) -> Result<(), IntegratorError>` <!-- rq-aa68f468 -->
  - Runs a single timestep. Calls `force_field.step(...)` at the
    point(s) of the integrator's choice.
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
- The dispatch order (`apply_pre`, then `step`, then `apply_post`,
  then `apply`) is fixed and identical across runs.
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
- Constraint algorithms (SHAKE, RATTLE, LINCS). The integrator trait
  shape supports them (constraints fit between drift and the next
  velocity update inside `step()`), but no constraint integrator
  ships in the default registry.
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
  Scenario: Dispatch loop calls thermostat.apply_pre, integrator.step, thermostat.apply_post in that order
    Given a velocity-Verlet integrator
    And a Nosé-Hoover-chain thermostat
    And a recording wrapper that timestamps every trait call
    When the runner executes one timestep
    Then the recorded order is exactly [apply_pre, step, apply_post]
    And no barostat hook is recorded

  @rq-8fd4e3bf
  Scenario: step() on empty state is a no-op
    Given a ParticleBuffers with particle_count() == 0
    And any constructed Integrator
    When integrator.step(&mut buffers, &mut sim_box, &mut force_field, dt=0.1, &mut timings) is called
    Then it returns Ok(())
    And no kernel launches are recorded for that call

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

  @rq-64f1e2a6
  Scenario: Velocity-Verlet step() launches vv_kick_drift, force pipeline, and vv_kick
    Given a velocity-Verlet integrator (lossless=false) with particle_count=4
    And a snapshot of buffers.positions_x and buffers.velocities_x before the call
    When integrator.step(&mut buffers, &mut sim_box, &mut force_field, dt=0.1, &mut timings) is called
    Then it returns Ok(())
    And positions_x differs from the snapshot
    And velocities_x differs from the snapshot
    And timings.finalize() reports count==1 for KernelStage::VV_KICK_DRIFT
    And timings.finalize() reports count==1 for KernelStage::VV_KICK

  @rq-35eef107
  Scenario: Lossless velocity-Verlet uses the lossless kernels
    Given a velocity-Verlet integrator (lossless=true) with particle_count=4
    When integrator.step(...) is called
    Then timings.finalize() reports count==1 for KernelStage::VV_KICK_DRIFT_LOSSLESS
    And timings.finalize() reports count==1 for KernelStage::VV_KICK_LOSSLESS
    And KernelStage::VV_KICK_DRIFT and KernelStage::VV_KICK have count==0

  @rq-15e0a433
  Scenario: Integrator owns the force evaluation inside step()
    Given a velocity-Verlet integrator and a ForceField with one LennardJones slot
    When integrator.step(...) is called once
    Then timings.finalize() reports count==1 for KernelStage::LJ_PAIR_FORCE
    And KernelStage::REDUCE_PAIR_FORCES has count==1
    And KernelStage::ACCUMULATE_FORCES has count==1

  # --- Compatibility ---

  @rq-e9be025b
  Scenario: VelocityVerlet does not own its thermostat
    Given an IntegratorKind::VelocityVerlet { lossless: false }
    Then kind.owns_thermostat() returns false

  @rq-4dd5d2d0
  Scenario: LangevinBaoab owns its thermostat
    Given an IntegratorKind::LangevinBaoab { friction: 1.0e12, temperature: 300.0, seed: 0 }
    Then kind.owns_thermostat() returns true

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
