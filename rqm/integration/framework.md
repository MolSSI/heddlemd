# Feature: Pluggable Integrator Framework <!-- rq-4f386df8 -->

The runner selects its time-integration algorithm from a fixed, named set of
*integrator slots* via the `[integrator].kind` field of the config (see
`config-schema.md`). Every slot implements the same per-step interface so
the runner's main loop is identical regardless of which integrator is in
use. Each slot is a Rust enum variant; adding a new integrator means adding
a new variant to `Integrator`, a matching `IntegratorKind` variant in the
config layer, and a per-variant kernel set.

## Slots <!-- rq-10c79bb0 -->

The framework currently exposes two integrator slots:

| `kind` value      | Variant                             | File                  |
| ----------------- | ----------------------------------- | --------------------- |
| `velocity-verlet` | symplectic NVE (lossy or lossless)  | `velocity-verlet.md`  |
| `langevin-baoab`  | stochastic NVT via BAOAB splitting  | `langevin-baoab.md`   |

Each slot's per-step kernels, parameter set, and timings stages are
documented in its own requirements file.

## Per-Step Interface <!-- rq-bc0e4fd5 -->

The runner drives the timestep loop in a fixed pattern:

```text
loop step in 1..=n_steps:
    integrator.pre_force_step(buffers, dt, step, timings)
    lj_pair_force(...) + reduce_pair_forces(...)        # force pipeline
    integrator.post_force_step(buffers, dt, step, timings)
    ...trajectory / log output...
```

`pre_force_step` runs every operation the integrator needs *before* the
force pipeline (typically position drifts and half-kicks). `post_force_step`
runs every operation the integrator needs *after* the force pipeline
(typically the final half-kick).

The runner performs one warm-up force evaluation before entering the loop,
so the first `pre_force_step` call always reads valid forces.

## Construction and Lifetime <!-- rq-7a4b9cbd -->

The runner constructs the integrator after `init_device` returns and
immediately after the `Timings` instance is created. Construction parameters
come from the parsed `IntegratorKind` config variant; per-particle device
buffers are allocated on the runner's `Arc<CudaDevice>`. The integrator's
allocations persist for the lifetime of the run and are dropped together
with the rest of the runner's GPU resources at end of run.

The runner never re-creates the integrator mid-run, and the integrator's
state never crosses to another `Arc<CudaDevice>`.

## Empty State <!-- rq-0f7b8a57 -->

When the runner has `particle_count == 0`, `pre_force_step` and
`post_force_step` return `Ok(())` without launching any kernel. The
integrator's allocations (if any) may have zero-length device slices but
must construct successfully.

## Step Counter <!-- rq-198f5ec1 -->

The `step_index` argument to `pre_force_step` and `post_force_step` matches
the `step` value the runner uses to gate trajectory and log writes. The
first call inside the timestep loop carries `step_index = 1`. Velocity
Verlet ignores it; Langevin BAOAB consumes it as part of the Philox
counter packing (see `langevin-baoab.md`).

## Feature API <!-- rq-5cb33196 -->

### Types <!-- rq-67414c32 -->

- `Integrator` — closed `enum` with one variant per slot. Variants carry <!-- rq-e4c4ff61 -->
  the per-run state owned by each integrator implementation.

  ```rust
  pub enum Integrator {
      VelocityVerlet(VelocityVerletState),
      LangevinBaoab(LangevinBaoabState),
  }
  ```

  The concrete state types (`VelocityVerletState`, `LangevinBaoabState`)
  are public and `Debug`; their fields are private. The variants implement
  no methods themselves — all behaviour lives on `Integrator`'s inherent
  impl, which matches on `self` and forwards.

- `IntegratorKind` — `enum` carrying the parsed config-level selection <!-- rq-686b0d37 -->
  and per-kind parameters. Lives alongside the rest of the config types
  in `crate::io::config`. Variants:

  ```rust
  pub enum IntegratorKind {
      VelocityVerlet { lossless: bool },
      LangevinBaoab { friction: f64, temperature: f64, seed: u64 },
  }
  ```

- `IntegratorError` — error type returned by every integrator method. <!-- rq-a5069572 -->
  Variants:
  - `Gpu(GpuError)` — CUDA driver / kernel-launch failure.
  - `Timings(TimingsError)` — CUDA event recording failure.

  The runner's `RunnerError::Integrator(IntegratorError)` wraps this.

### Functions and methods <!-- rq-ad27732e -->

- `Integrator::new(device: Arc<CudaDevice>, particle_count: usize, kind: &IntegratorKind) -> Result<Self, IntegratorError>` <!-- rq-df39d15b -->
  - Dispatches on `kind`. For `VelocityVerlet`, constructs a
    `VelocityVerletState` that allocates `LosslessBuffers` when
    `lossless == true`. For `LangevinBaoab`, constructs a
    `LangevinBaoabState` that captures the friction, temperature, and seed
    (no per-particle state allocations; Philox is counter-based — see
    `langevin-baoab.md`).
  - A `particle_count` of zero is permitted: any per-particle device
    allocations have length zero.

- `Integrator::pre_force_step(&mut self, buffers: &mut ParticleBuffers, dt: f32, step_index: u64, timings: &mut Timings) -> Result<(), IntegratorError>` <!-- rq-cf361ff5 -->
  - Matches on `self` and forwards to the variant's implementation.
  - Returns `Ok(())` without launching any kernel when
    `buffers.particle_count() == 0`.

- `Integrator::post_force_step(&mut self, buffers: &mut ParticleBuffers, dt: f32, step_index: u64, timings: &mut Timings) -> Result<(), IntegratorError>` <!-- rq-700c7729 -->
  - Matches on `self` and forwards to the variant's implementation.
  - Returns `Ok(())` without launching any kernel when
    `buffers.particle_count() == 0`.

`Integrator::new`, `pre_force_step`, and `post_force_step` are the entire
public surface of the framework. Variant-specific helpers (e.g. the VV
launcher functions, the Langevin OU kernel) remain accessible from
`crate::gpu` for direct use in tests but are not part of the slot
interface.

## Determinism Guarantees <!-- rq-8d44fa4d -->

The slot framework preserves the project's bit-wise reproducibility
invariant under the same conditions each integrator individually
guarantees:

- All slot variants launch on the default stream of the same
  `Arc<CudaDevice>` carried by `ParticleBuffers`. No additional streams
  are introduced.
- The integrator's `step_index` argument is the only loop-carried state
  the framework injects; it is deterministic and identical across runs
  with the same config.
- Variants that draw random numbers (`LangevinBaoab`) document the exact
  RNG scheme so two runs on the same GPU with the same `seed` produce
  byte-identical trajectories.

## Out of Scope <!-- rq-f542b61f -->

- A user-supplied integrator DSL (à la OpenMM's `CustomIntegrator`).
  Slots are a closed set; new integrators land as new enum variants in
  the codebase, not as configuration.
- Multi-time-step / RESPA integrators that split forces by frequency.
- Constraint algorithms (SHAKE, RATTLE, LINCS).
- Pressure coupling (barostats); the slots only cover NVE and NVT
  ensembles.
- Mid-run changes of integrator. The integrator is fixed at construction
  and never replaced.
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
  Scenario: Construct velocity-Verlet (lossy)
    Given an IntegratorKind::VelocityVerlet { lossless: false }
    When Integrator::new(device, particle_count=4, &kind) is called
    Then it returns Ok(integrator)
    And integrator matches the VelocityVerlet variant

  @rq-db78448e
  Scenario: Construct velocity-Verlet (lossless)
    Given an IntegratorKind::VelocityVerlet { lossless: true }
    When Integrator::new(device, particle_count=4, &kind) is called
    Then it returns Ok(integrator)
    And the variant carries a LosslessBuffers with particle_count == 4

  @rq-47877631
  Scenario: Construct Langevin BAOAB
    Given an IntegratorKind::LangevinBaoab { friction: 1.0e12, temperature: 300.0, seed: 42 }
    When Integrator::new(device, particle_count=4, &kind) is called
    Then it returns Ok(integrator)
    And integrator matches the LangevinBaoab variant

  @rq-48fd88ed
  Scenario: Construct with particle_count = 0
    Given an IntegratorKind::VelocityVerlet { lossless: true }
    When Integrator::new(device, particle_count=0, &kind) is called
    Then it returns Ok(integrator)
    And the variant's LosslessBuffers has particle_count == 0

  # --- Per-step dispatch ---

  @rq-171b99f5
  Scenario: pre_force_step on empty state is a no-op
    Given a ParticleBuffers with particle_count() == 0
    And any constructed Integrator variant
    When integrator.pre_force_step(&mut buffers, dt=0.1, step_index=1, &mut timings) is called
    Then it returns Ok(())
    And no kernel launches are recorded for that call

  @rq-a49bb176
  Scenario: post_force_step on empty state is a no-op
    Given a ParticleBuffers with particle_count() == 0
    And any constructed Integrator variant
    When integrator.post_force_step(&mut buffers, dt=0.1, step_index=1, &mut timings) is called
    Then it returns Ok(())
    And no kernel launches are recorded for that call

  @rq-2980a672
  Scenario: Velocity-Verlet pre_force_step launches vv_kick_drift
    Given an Integrator::VelocityVerlet { lossless=false } with particle_count=4
    And a snapshot of buffers.positions_x before the call
    When integrator.pre_force_step(&mut buffers, dt=0.1, step_index=1, &mut timings) is called
    Then it returns Ok(())
    And positions_x differs from the snapshot
    And timings.finalize() reports count==1 for KernelStage::VvKickDrift

  @rq-36382434
  Scenario: Velocity-Verlet post_force_step launches vv_kick
    Given an Integrator::VelocityVerlet { lossless=false } with particle_count=4
    And a snapshot of buffers.velocities_x before the call
    When integrator.post_force_step(&mut buffers, dt=0.1, step_index=1, &mut timings) is called
    Then it returns Ok(())
    And velocities_x differs from the snapshot
    And timings.finalize() reports count==1 for KernelStage::VvKick

  @rq-7b9aada4
  Scenario: Lossless velocity-Verlet uses the lossless kernels
    Given an Integrator::VelocityVerlet { lossless=true } with particle_count=4
    When integrator.pre_force_step(...) and integrator.post_force_step(...) are called
    And timings.finalize() is queried
    Then KernelStage::VvKickDriftLossless has count==1
    And KernelStage::VvKickLossless has count==1
    And KernelStage::VvKickDrift and KernelStage::VvKick have count==0

  # --- Step counter propagation ---

  @rq-d12c24f0
  Scenario: step_index is passed through to the variant
    Given an Integrator::LangevinBaoab with seed=1, friction=1e12, temperature=300, particle_count=2
    When pre_force_step is called twice on identical inputs with step_index=1 and step_index=2
    Then the two calls produce different post-call velocities

  # --- Determinism across two runs ---

  @rq-706001ec
  Scenario: Two independent runs with identical inputs are byte-identical
    Given two Integrator instances of the same variant, built from identical IntegratorKinds
    And two ParticleBuffers built from byte-identical ParticleStates
    When each runs N=10 timesteps with the same dt and the same step_index sequence
    Then the two final ParticleStates agree byte-for-byte
```
