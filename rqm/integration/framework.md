# Feature: Pluggable Integrator Framework <!-- rq-4f386df8 -->

The runner drives time integration through a single `Integrator` trait
object. Each timestep, the runner calls `integrator.step(...)` once; the
integrator owns the full sub-step sequence within a step — kicks, drifts,
thermostat updates, force evaluations, and (in future implementations)
barostat box rescalings or constraint solves. The runner has no
visibility into how many force evaluations a single `step()` performs or
what intermediate substeps the integrator runs.

Construction is registry-driven. Each integrator implementation
registers a named builder at startup; the runner consults the registry
with the kind name parsed from the config and receives a
`Box<dyn Integrator>`. Adding an integrator means writing a new
`Integrator` implementation and a builder that registers under a new
name.

## Slots <!-- rq-10c79bb0 -->

The default registry exposes two integrators:

| `kind` value      | Implementation                       | File                  |
| ----------------- | ------------------------------------ | --------------------- |
| `velocity-verlet` | symplectic NVE (lossy or lossless)   | `velocity-verlet.md`  |
| `langevin-baoab`  | stochastic NVT via BAOAB splitting   | `langevin-baoab.md`   |

Each implementation's per-step kernels, parameter set, and timings
stages are documented in its own requirements file.

## Per-Step Interface <!-- rq-bc0e4fd5 -->

The runner drives the timestep loop in a fixed pattern:

```text
loop step in 1..=n_steps:
    integrator.step(buffers, sim_box, force_field, dt, step, timings)
    ...trajectory / log output...
```

`step()` is responsible for everything between the previous step's
output and the current step's output: all velocity and position
updates, all force evaluations, and (when implemented) all
thermostat / barostat / constraint substeps. The integrator calls
`force_field.step(buffers, sim_box, timings)` at the point(s) of its
choice; for the symplectic and Leimkuhler–Matthews integrators in the
default registry this happens exactly once per `step()`.

The runner performs one warm-up `force_field.step(...)` call before
entering the timestep loop so the integrator's first `step()` reads
valid `forces_*`. Integrators that follow the symplectic-with-cached-F
contract assume `buffers.forces_*` holds `F(t)` on entry and produce
`F(t+dt)` on exit. The current registry's integrators all follow this
contract.

## Construction and Lifetime <!-- rq-7a4b9cbd -->

The runner constructs the integrator after `init_device` returns and
immediately after the `Timings` instance is created. Construction
parameters come from the parsed `IntegratorKind` config variant; the
runner queries an `IntegratorRegistry` with the kind's name and the
kind's payload to obtain a `Box<dyn Integrator>`. Per-particle device
buffers (when an implementation needs them — for example,
`LosslessBuffers` for the lossless velocity-Verlet mode) are allocated
on the runner's `Arc<CudaDevice>` inside the builder.

The integrator's allocations persist for the lifetime of the run and
are dropped together with the rest of the runner's GPU resources at end
of run. The runner never re-creates the integrator mid-run, and the
integrator's state never crosses to another `Arc<CudaDevice>`.

## Empty State <!-- rq-0f7b8a57 -->

When the runner has `particle_count == 0`, `step()` returns `Ok(())`
without launching any kernel. The integrator's allocations (if any)
may have zero-length device slices but must construct successfully.

## Step Counter <!-- rq-198f5ec1 -->

The `step_index` argument to `step()` matches the `step` value the
runner uses to gate trajectory and log writes. The first call inside
the timestep loop carries `step_index = 1`. Velocity Verlet ignores it;
Langevin BAOAB consumes it as part of the Philox counter packing (see
`langevin-baoab.md`).

## Feature API <!-- rq-5cb33196 -->

### Types <!-- rq-67414c32 -->

- `Integrator` — object-safe trait implemented by every concrete <!-- rq-e4c4ff61 -->
  integrator.

  ```rust
  pub trait Integrator: std::fmt::Debug + Send {
      fn step(
          &mut self,
          buffers: &mut ParticleBuffers,
          sim_box: &mut SimulationBox,
          force_field: &mut ForceField,
          dt: f32,
          step_index: u64,
          timings: &mut Timings,
      ) -> Result<(), IntegratorError>;
  }
  ```

  - The integrator runs every sub-step the timestep requires
    (velocity kicks, position drifts, thermostat updates, force
    evaluations, etc.). It calls `force_field.step(buffers, sim_box,
    timings)` at the appropriate point(s).
  - `sim_box` is passed mutably so future barostats may rescale the
    box during the step. Integrators that do not rescale leave it
    unchanged.
  - On a successful return, `buffers.forces_*` holds `F` evaluated at
    the post-step positions (so the next `step()` can begin with a
    half-kick that reads `F(t)`).
  - Returns `Ok(())` immediately when `buffers.particle_count() == 0`.

- `IntegratorKind` — `enum` carrying the parsed config-level selection <!-- rq-686b0d37 -->
  and per-kind parameters. Lives alongside the rest of the config
  types in `crate::io::config`. Variants:

  ```rust
  pub enum IntegratorKind {
      VelocityVerlet { lossless: bool },
      LangevinBaoab { friction: f64, temperature: f64, seed: u64 },
  }
  ```

  `IntegratorKind` is a closed enum at the config layer so the config
  loader can emit specific error messages (missing field, wrong type,
  unknown kind) before the runner reaches the registry. Each variant
  carries enough information for the corresponding builder to
  construct its concrete integrator.

  `IntegratorKind::name(&self) -> &'static str` returns the same
  string the registry uses as the lookup key (`"velocity-verlet"` for
  `VelocityVerlet`, `"langevin-baoab"` for `LangevinBaoab`). This is
  the bridge between the closed config-level enum and the open
  registry.

- `IntegratorBuilder` — trait describing a registered integrator. <!-- rq-87fdd9b1 -->
  Implementations are stateless and self-register at construction
  time.

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
  ```

  - `kind_name` returns the lookup key.
  - `build` inspects `kind` (typically by matching the relevant
    variant) and constructs the concrete `Integrator`. Returns
    `IntegratorError::UnknownKind` if the variant does not match the
    expected one.

- `IntegratorRegistry` — host-side registry of `IntegratorBuilder`s. <!-- rq-1d5b5e35 -->
  Fields:
  - `builders: Vec<Box<dyn IntegratorBuilder>>`

  Methods:
  - `IntegratorRegistry::with_builtins() -> IntegratorRegistry` —
    constructs a registry pre-populated with `velocity-verlet` and
    `langevin-baoab` builders.
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

- `IntegratorError` — error type returned by every integrator method. <!-- rq-a5069572 -->
  Variants:
  - `Gpu(GpuError)` — CUDA driver / kernel-launch failure.
  - `Timings(TimingsError)` — CUDA event recording failure.
  - `ForceField(ForceFieldError)` — surfaces failures from
    `force_field.step(...)` called inside the integrator.
  - `UnknownKind(String)` — the registry has no builder for the
    requested `kind` name.

  The runner's `RunnerError::Integrator(IntegratorError)` wraps this.

### Functions and methods <!-- rq-ad27732e -->

- `IntegratorRegistry::build(&self, kind: &IntegratorKind, device: Arc<CudaDevice>, particle_count: usize) -> Result<Box<dyn Integrator>, IntegratorError>` <!-- rq-df39d15b -->
  - Looks up the builder whose `kind_name()` equals `kind.name()`.
    Delegates construction to that builder.
  - Returns `IntegratorError::UnknownKind(...)` when no builder matches.
  - The builder is responsible for kind-specific allocations:
    - For `VelocityVerlet { lossless }`, the builder constructs a
      `VelocityVerletState` that allocates `LosslessBuffers` when
      `lossless == true`.
    - For `LangevinBaoab { friction, temperature, seed }`, the builder
      constructs a `LangevinBaoabState` that captures the friction,
      temperature, and seed (no per-particle state allocations; Philox
      is counter-based — see `langevin-baoab.md`).
  - A `particle_count` of zero is permitted: any per-particle device
    allocations have length zero.

- `Integrator::step(&mut self, buffers: &mut ParticleBuffers, sim_box: &mut SimulationBox, force_field: &mut ForceField, dt: f32, step_index: u64, timings: &mut Timings) -> Result<(), IntegratorError>` <!-- rq-cf361ff5 -->
  - Runs a single timestep. Calls `force_field.step(...)` at the
    point(s) of the integrator's choice.
  - Returns `Ok(())` without launching any kernel when
    `buffers.particle_count() == 0`.

## Determinism Guarantees <!-- rq-8d44fa4d -->

The trait framework preserves the project's bit-wise reproducibility
invariant under the same conditions each integrator individually
guarantees:

- All integrator kernels and the force pipeline run on the default
  stream of the same `Arc<CudaDevice>` carried by `ParticleBuffers`.
  No additional streams are introduced.
- The integrator's `step_index` argument is the only loop-carried
  state the framework injects; it is deterministic and identical
  across runs with the same config.
- Implementations that draw random numbers (`LangevinBaoab`) document
  the exact RNG scheme so two runs on the same GPU with the same
  `seed` produce byte-identical trajectories.

## Out of Scope <!-- rq-f542b61f -->

- A user-supplied integrator DSL (à la OpenMM's `CustomIntegrator`).
  New integrators are Rust source code that implement the
  `Integrator` trait and register a builder.
- Multi-time-step / RESPA integrators that split forces by frequency.
  The trait shape supports them (the integrator can call
  `force_field.step` multiple times per `step()` with different
  effective `dt`s), but no RESPA implementation ships in the default
  registry.
- Constraint algorithms (SHAKE, RATTLE, LINCS). The trait shape
  supports them (constraints fit between drift and the next
  velocity update inside `step()`), but no constraint integrator
  ships in the default registry.
- Pressure coupling (barostats). The trait shape supports them
  (`sim_box` is `&mut`), but no barostat ships in the default
  registry. When a barostat is added, downstream consumers of the
  box (most notably `NeighborListState`) must be notified of
  changes so cached cell sizes stay consistent; that contract is
  documented in `forces/neighbor-list.md` at the time the barostat
  lands.
- Mid-run replacement of the integrator. The integrator is fixed at
  construction and never replaced for the duration of a run.
- Dynamic loading of integrators from shared libraries.

---

## Gherkin Scenarios <!-- rq-9605b930 -->

```gherkin
Feature: Pluggable integrator framework

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called

  # --- Construction ---

  @rq-e02917c3
  Scenario: Construct velocity-Verlet (lossy) via the registry
    Given an IntegratorRegistry::with_builtins()
    And an IntegratorKind::VelocityVerlet { lossless: false }
    When registry.build(&kind, device, particle_count=4) is called
    Then it returns Ok(integrator)
    And integrator's underlying type implements `Integrator`

  @rq-db78448e
  Scenario: Construct velocity-Verlet (lossless) via the registry
    Given an IntegratorRegistry::with_builtins()
    And an IntegratorKind::VelocityVerlet { lossless: true }
    When registry.build(&kind, device, particle_count=4) is called
    Then it returns Ok(integrator)
    And the underlying integrator allocates LosslessBuffers with particle_count == 4

  @rq-47877631
  Scenario: Construct Langevin BAOAB via the registry
    Given an IntegratorRegistry::with_builtins()
    And an IntegratorKind::LangevinBaoab { friction: 1.0e12, temperature: 300.0, seed: 42 }
    When registry.build(&kind, device, particle_count=4) is called
    Then it returns Ok(integrator)

  @rq-48fd88ed
  Scenario: Construct with particle_count = 0
    Given an IntegratorRegistry::with_builtins()
    And an IntegratorKind::VelocityVerlet { lossless: true }
    When registry.build(&kind, device, particle_count=0) is called
    Then it returns Ok(integrator)
    And every per-particle device allocation has length 0

  @rq-78da0ce9
  Scenario: Registry without a matching builder reports UnknownKind
    Given an empty IntegratorRegistry (no builders registered)
    And an IntegratorKind::VelocityVerlet { lossless: false }
    When registry.build(&kind, device, particle_count=4) is called
    Then it returns Err(IntegratorError::UnknownKind("velocity-verlet"))

  @rq-89b4b926
  Scenario: Custom builder registered after with_builtins() is selectable
    Given an IntegratorRegistry::with_builtins()
    And a custom IntegratorBuilder whose kind_name() is "test-stub"
    When registry.register(custom_builder) is called
    Then registry.build(...) routes test-stub-kind requests to the custom builder

  # --- Per-step dispatch ---

  @rq-171b99f5
  Scenario: step() on empty state is a no-op
    Given a ParticleBuffers with particle_count() == 0
    And any constructed Integrator
    When integrator.step(&mut buffers, &mut sim_box, &mut force_field, dt=0.1, step_index=1, &mut timings) is called
    Then it returns Ok(())
    And no kernel launches are recorded for that call

  @rq-2980a672
  Scenario: Velocity-Verlet step() launches vv_kick_drift, force pipeline, and vv_kick
    Given a velocity-Verlet integrator (lossless=false) with particle_count=4
    And a snapshot of buffers.positions_x and buffers.velocities_x before the call
    When integrator.step(&mut buffers, &mut sim_box, &mut force_field, dt=0.1, step_index=1, &mut timings) is called
    Then it returns Ok(())
    And positions_x differs from the snapshot
    And velocities_x differs from the snapshot
    And timings.finalize() reports count==1 for KernelStage::VvKickDrift
    And timings.finalize() reports count==1 for KernelStage::VvKick

  @rq-7b9aada4
  Scenario: Lossless velocity-Verlet uses the lossless kernels
    Given a velocity-Verlet integrator (lossless=true) with particle_count=4
    When integrator.step(...) is called
    Then timings.finalize() reports count==1 for KernelStage::VvKickDriftLossless
    And timings.finalize() reports count==1 for KernelStage::VvKickLossless
    And KernelStage::VvKickDrift and KernelStage::VvKick have count==0

  @rq-1b18924f
  Scenario: Integrator owns the force evaluation inside step()
    Given a velocity-Verlet integrator and a ForceField with one LennardJones slot
    When integrator.step(...) is called once
    Then timings.finalize() reports count==1 for KernelStage::LjPairForce
    And KernelStage::ReducePairForces has count==1
    And KernelStage::AccumulateForces has count==1

  # --- Step counter propagation ---

  @rq-d12c24f0
  Scenario: step_index is passed through and affects RNG draws
    Given a Langevin-BAOAB integrator with seed=1, friction=1e12, temperature=300, particle_count=2
    When step() is called twice on identical inputs with step_index=1 and step_index=2
    Then the two calls produce different post-call velocities

  # --- Determinism across two runs ---

  @rq-706001ec
  Scenario: Two independent runs with identical inputs are byte-identical
    Given two Integrator instances of the same kind built from identical IntegratorKinds
    And two ParticleBuffers built from byte-identical ParticleStates
    When each runs N=10 timesteps with the same dt and the same step_index sequence
    Then the two final ParticleStates agree byte-for-byte
```
