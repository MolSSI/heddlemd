# Feature: Langevin BAOAB Integrator <!-- rq-d5a4f220 -->

Stochastic NVT integrator using the Leimkuhler-Matthews BAOAB splitting of
Langevin dynamics. One of the pluggable integrator slots (see
`framework.md`); selected by `kind = "langevin-baoab"` in the config's
`[integrator]` section.

Langevin BAOAB is a **fused** integrator: the Ornstein-Uhlenbeck "O"
step inside the B-A-O-A-B splitting is itself the temperature coupling,
so the integrator owns its own thermostat. The
`"langevin-baoab"` builder's
`IntegratorBuilder::owns_thermostat(&params)` returns `true`, and
`Config::validate_against(&registries)` rejects any config that
combines `[integrator] kind = "langevin-baoab"` with a
`[thermostat]` table via `ConfigError::IncompatibleThermostat
{ integrator: "langevin-baoab" }`. To compose velocity-Verlet with a thermostat
from the registry instead, select `kind = "velocity-verlet"` and add
a `[thermostat]` section (`csvr`, `nose-hoover-chain`, `andersen`, or
`berendsen`).

The Langevin equation of motion is

```text
dx = v dt
dv = (F/m) dt - γ v dt + sqrt(2 γ T / m) dW
```

where `γ` is the friction coefficient, `T` is the bath temperature, and
`dW` is a standard Wiener increment. The BAOAB splitting writes the
single-step propagator as the symmetric product of three operators:
**B** (momentum kick from forces), **A** (position drift), and **O**
(Ornstein-Uhlenbeck step on momenta).

## Algorithm <!-- rq-0e5ac409 -->

For each timestep of size `dt` with `α = exp(-γ dt)`:

1. **B(dt/2)**: `v ← v + (F/m) (dt/2)`
2. **A(dt/2)**: `x ← x + v (dt/2)`
3. **O(dt)**:   `v ← α v + sqrt((1 - α²) T / m) ξ`
4. **A(dt/2)**: `x ← x + v (dt/2)`
5. **Force evaluation**: recompute `F(x)` using the new positions.
6. **B(dt/2)**: `v ← v + (F/m) (dt/2)`

Steps 1–4, step 5 (a `SubStep::ForceEval` that the runner dispatches
to `force_field.step(...)`), and step 6 are emitted as the
integrator's `StepPlan` and walked by the runner. The initial force
evaluation needed to seed step 1 of the first iteration is provided
by the runner's standard warm-up pass.

`ξ` is a vector of independent standard normal random variables, one per
particle per axis, generated as described in *RNG* below.

The Boltzmann constant `k_B = 1` exactly in the engine's Hartree
atomic units; the bath temperature `T` carries `k_B · T` in Hartrees
and no explicit Boltzmann factor appears in any expression above.

## Step Plan <!-- rq-ff46e833 -->

`LangevinBaoabState::plan(dt)` returns the fixed six-element BAOAB
sequence the runner walks:

```rust
StepPlan { steps: vec![
    SubStep::KickHalf { dt, label: "B"          },   // first half-kick
    SubStep::Drift    { dt, label: "A_pre"      },   // first half-drift
    SubStep::Custom   {     label: "O"          },   // Ornstein-Uhlenbeck
    SubStep::Drift    { dt, label: "A_post"     },   // second half-drift
    SubStep::ForceEval,                              // recompute F(x)
    SubStep::KickHalf { dt, label: "B"          },   // second half-kick
]}
```

The `Drift` sub-steps internally use `dt/2`; the integrator's
`execute()` reads the `dt` carried on the sub-step and applies the
appropriate factor. Likewise both `KickHalf`s apply `dt/2` internally.

`LangevinBaoabState::execute(sub, ...)` dispatches to the kernels:

| Sub-step variant         | Label    | Kernel           | Operation                              | Stage label          |
| ------------------------ | -------- | ---------------- | -------------------------------------- | -------------------- |
| `SubStep::KickHalf`      | `B`      | `vv_kick`        | `v += (F/m) (dt/2)`                    | `LangevinKickHalf`   |
| `SubStep::Drift`         | `A_pre`  | `lan_drift_half` | `x += v (dt/2)`                        | `LangevinDriftHalf`  |
| `SubStep::Custom`        | `O`      | `lan_ou_step`    | `v ← α v + sqrt((1-α²) T/m) ξ`     | `LangevinOuStep`     |
| `SubStep::Drift`         | `A_post` | `lan_drift_half` | `x += v (dt/2)`                        | `LangevinDriftHalf`  |
| (`SubStep::ForceEval`)   |          | force pipeline   | dispatched by runner                   | (force-pipeline stages) |
| `SubStep::KickHalf`      | `B`      | `vv_kick`        | `v += (F/m) (dt/2)`                    | `LangevinKickHalf`   |

`vv_kick` is reused from velocity Verlet because the half-kick
operation is identical. The reuse is reflected in the timings file
by labelling calls from the Langevin slot with the `Langevin*` stage
names — see `performance-analysis.md`.

`lan_drift_half` and `lan_ou_step` are the CUDA kernels in
`kernels/langevin.cu`.

The Custom `"O"` sub-step is also where the integrator's
`draw_counter` increments. The counter advances exactly once per
timestep because `"O"` appears exactly once in the plan.

The `"langevin-baoab"` builder's
`IntegratorBuilder::supports_constraints(&params)` returns `false`
regardless of params; the runner therefore inserts no constraint
hooks around this integrator's two `Drift` sub-steps or its final
`KickHalf`. Composing constraints with Langevin BAOAB is rejected at
config load by `ConfigError::IncompatibleConstraint` (see
`integration/constraint-framework.md`).

## Parameters <!-- rq-1b6324c3 -->

The `"langevin-baoab"` builder deserialises
`LangevinBaoabParams { friction: f64, temperature: f64, seed: u64 }`
from the `[integrator]` section's `SlotConfig::params` field:

- `friction: f64` — the damping coefficient `γ` in inverse atomic
  time units (`1 / (hbar / E_h)`). Required. Finite and strictly
  positive; `friction = 0` is rejected
  (the integrator would degenerate to velocity Verlet and users should
  select `kind = "velocity-verlet"` explicitly).
- `temperature: f64` — bath temperature `T` as `k_B · T` in Hartrees
  (the engine's internal temperature representation; `k_B = 1`).
  Required.
  Finite and strictly positive. Independent of `simulation.temperature`,
  which governs the initial-velocity sampler; users may quench or heat
  between the initial state and the run by supplying different values.
- `seed: u64` — counter-based RNG seed. Required, independent of
  `simulation.seed`. Two runs with identical configs on the same GPU
  produce byte-identical trajectories.

## RNG <!-- rq-ce445e13 -->

The OU step requires `3 N` independent Gaussian samples per timestep. The
samples are generated by a counter-based Philox-4×32-10 RNG (Salmon et al.,
SC11) consumed inline inside `lan_ou_step`. No per-thread RNG state is
stored on the device.

### Counter packing <!-- rq-98084bc0 -->

Each per-`(particle, axis)` Langevin draw is the output of one
Philox-4×32-10 invocation with:

- **Key (2 × u32)**: `(seed_lo, seed_hi)` — the low and high halves of the
  config's `integrator.seed`.
- **Counter (4 × u32)**:
  - `counter[0] = draw_counter_lo` — low 32 bits of the integrator's
    `draw_counter`.
  - `counter[1] = draw_counter_hi` — high 32 bits of `draw_counter`.
  - `counter[2] = particle_id` — the particle's `u32` ID from
    `ParticleBuffers::particle_ids` (numerically equal to its index when
    the runner uses default IDs).
  - `counter[3] = axis_id` — `0` for x, `1` for y, `2` for z.

This packing guarantees that every `(seed, draw_counter, particle_id,
axis_id)` quadruple maps to a unique Philox counter and therefore an
independent draw.

The `draw_counter` lives on `LangevinBaoabState` (see *Feature API*
below). It starts at `0` at construction. Each call to
`LangevinBaoabState::step` performs `self.draw_counter += 1` before
launching the OU kernel and passes the post-increment value to
`lan_ou_step`. The first invocation therefore uses `draw_counter = 1`,
the second uses `2`, and so on. The integrator's `step()` is the only
place that touches `draw_counter`; consumers wanting reproducible
re-runs from a known position set the field directly before any
`step()` call.

### Box-Muller transform <!-- rq-25ba1194 -->

Philox produces four `u32` values per invocation; the OU step uses the
first two:

```text
u1 = (output[0] as f64 + 0.5) * 2^-32       # uniform in (0, 1)
u2 = (output[1] as f64 + 0.5) * 2^-32       # uniform in (0, 1)
ξ  = sqrt(-2 ln u1) * cos(2 π u2)            # standard normal
```

The second Box-Muller output (the `sin` half) is discarded. `output[2]`
and `output[3]` are unused.

### Reproducibility <!-- rq-2b21f93c -->

Two runs with identical `(seed, friction, temperature)` and identical
initial particle state on the same GPU produce byte-identical trajectory
and log files. The Philox key/counter scheme is stateless, so the RNG
contributes no run-to-run drift.

## Feature API <!-- rq-dc5c70bf -->

### Types <!-- rq-b41d8d36 -->

- `LangevinBaoabState` — Langevin-BAOAB integrator state, registered <!-- rq-bcb0f58a -->
  in the integrator framework under `kind = "langevin-baoab"` (see
  `framework.md`). Fields:

  - `friction: f64`
  - `temperature: f64`
  - `seed: u64`
  - `draw_counter: u64` — Philox counter advance for the OU kernel.
    Initialised to `0` by the builder. `LangevinBaoabState::step`
    pre-increments this field on every call (`self.draw_counter += 1`)
    and passes the post-increment value to `lan_ou_step`; the first
    invocation in a run therefore uses `draw_counter == 1`. The field
    is public so future restart-from-checkpoint flows can restore it
    explicitly; in-run code does not modify it from outside `step()`.

  All fields are public for parity with the other registered
  integrators' state; construction goes through `LangevinBaoabBuilder`
  with the `SlotConfig` whose `kind == "langevin-baoab"`.

### CUDA Kernels <!-- rq-26ba73d0 -->

`kernels/langevin.cu` declares two `extern "C"` kernels and
`#include`s `kernels/philox.cuh` for the shared `__device__` Philox
primitives (`philox4x32_10`, `philox_gaussian`,
`u32_to_uniform_open`; see `andersen.md` for the header's full
declaration set). The same header is consumed by every CUDA file
that needs counter-based RNG draws (currently `langevin.cu` and
`andersen.cu`); the algorithm description (constants, round
structure, Box-Muller transform) lives in this file under *RNG*
above and is its single specification site.

```c
// in kernels/philox.cuh:
__device__ inline void philox4x32_10(
    unsigned int key_lo, unsigned int key_hi,
    unsigned int ctr0, unsigned int ctr1,
    unsigned int ctr2, unsigned int ctr3,
    unsigned int *out0, unsigned int *out1,
    unsigned int *out2, unsigned int *out3);

extern "C" __global__ void lan_drift_half(
    float *positions_x, float *positions_y, float *positions_z,
    int *images_x, int *images_y, int *images_z,
    const float *velocities_x, const float *velocities_y, const float *velocities_z,
    float lx, float ly, float lz, float xy, float xz, float yz,
    float dt,
    unsigned int n);

extern "C" __global__ void lan_ou_step(
    float *velocities_x, float *velocities_y, float *velocities_z,
    const float *masses,
    const unsigned int *particle_ids,
    unsigned int seed_lo, unsigned int seed_hi,
    unsigned int draw_counter_lo, unsigned int draw_counter_hi,
    float alpha,
    float kt,        // engine-side temperature value: k_B · T in Hartrees
    unsigned int n);
```

Each thread computes its global index as
`blockIdx.x * blockDim.x + threadIdx.x`. If the index is `>= n` the thread
returns without touching any buffer.

`lan_drift_half` performs `x[i] += v[i] * (dt * 0.5f)` per axis, then
wraps the updated position back into the primary image of the
simulation box via the same triclinic
`wrap_position_with_image_count` step used by velocity Verlet (see
`simulation-box.md`). The integer triple returned by the wrap is added
to `(images_x[i], images_y[i], images_z[i])`. The unwrapped position
`wrapped + images_x · a + images_y · b + images_z · c` is invariant
under this wrap. For an orthorhombic box the algorithm reduces to per-
axis wraps. Velocities are read-only inputs and are not touched.

`lan_ou_step` performs, for axis `a ∈ {0, 1, 2}`:

```text
philox4x32_10(seed_lo, seed_hi, draw_lo, draw_hi, particle_ids[i], a, &o0, &o1, &o2, &o3)
u1 = (o0 + 0.5) * 2^-32   // f64
u2 = (o1 + 0.5) * 2^-32   // f64
xi = sqrt(-2 ln u1) * cos(2 π u2)
sigma = sqrtf((1 - alpha*alpha) * kt / masses[i])
v[a][i] = alpha * v[a][i] + sigma * (float) xi
```

The host pre-computes `alpha = expf(- (friction as f32) * dt)` and
`kt = temperature as f32` once per timestep and passes them to the
kernel (the engine stores `k_B · T` directly; `k_B = 1`, so no
multiplication is performed).

### PTX Module Loading <!-- rq-8586776f -->

`init_device()` loads the compiled `kernels/langevin.cu` PTX as module
`"langevin"` and captures its `lan_drift_half` and `lan_ou_step`
functions into the `Kernels` handle (see `build-pipeline.md`).

### Rust Launch Helpers <!-- rq-3bf899d0 -->

Two free functions in `src/gpu/kernels.rs`, re-exported from `crate::gpu`:

- `lan_drift_half(buffers: &mut ParticleBuffers, sim_box: &SimulationBox, dt: f32) -> Result<(), GpuError>` <!-- rq-f00f729e -->
  - Launches the `lan_drift_half` kernel, reading edge lengths from
    `sim_box` for the wrap step.
  - Block size 256; grid `ceil(n / 256)`.
  - When `buffers.particle_count() == 0`, returns `Ok(())` without
    launching.
  - Invokes the kernel through the `Kernels` handle reached from its
    arguments; it performs no string-keyed kernel lookup of its own (see
    `build-pipeline.md`).

- `lan_ou_step(buffers: &mut ParticleBuffers, seed: u64, draw_counter: u64, alpha: f32, kt: f32) -> Result<(), GpuError>` <!-- rq-6435723d -->
  - Launches the `lan_ou_step` kernel with the seed and draw counter
    packed into `(seed_lo, seed_hi, draw_counter_lo, draw_counter_hi)`
    u32 pairs.
  - Block size 256; grid `ceil(n / 256)`.
  - When `buffers.particle_count() == 0`, returns `Ok(())` without
    launching.
  - Invokes the kernel through the `Kernels` handle, like
    `lan_drift_half`.

## Launch Configuration <!-- rq-9bef7cd0 -->

- Block size: 256 threads.
- Grid size: `ceil(n / 256)` blocks in the x dimension.
- Shared memory: zero bytes.
- Stream: the default stream carried by `ParticleBuffers::device`.

## Out of Scope <!-- rq-e4ace7d6 -->

- Other Langevin splittings (OBABO, ABOBA, etc.).
- Massive thermostatting (per-atom γ).
- Centre-of-mass drift removal during the O step. A
  `remove_com_drift` flag may be added as a small in-place edit to this
  feature when needed.
- Constraints (SHAKE/RATTLE) and constrained Langevin.
- A lossless `(f32, f64)` compensated mode. The OU step is stochastic;
  bit-exact reversibility is not meaningful.
- Composition with the `[thermostat]` slot. The integrator owns its
  own thermostat (the OU step); `load_config` rejects configurations
  that pair `langevin-baoab` with any `[thermostat]` entry.
- Pressure coupling. Composition with the `[barostat]` slot is
  permitted by the framework but no concrete barostat ships in the
  default registry, so the question is not yet exercised.
- The Verlet B and A steps as separate kernels with their own `_lossless`
  variants. Langevin shares `vv_kick` with velocity Verlet for the B step;
  `lan_drift_half` is its own kernel because Verlet has no half-drift
  primitive.
- cuRAND-based RNG; Philox-4×32-10 is the only supported source.
- Sub-stepping the O step for very large `γ dt`. The `α = exp(-γ dt)`
  formula is well-conditioned for all `γ dt > 0` but loses noise
  resolution as `α → 0`; users are expected to pick a reasonable `γ dt`
  themselves.

---

## Gherkin Scenarios <!-- rq-1f2f912b -->

```gherkin
Feature: Langevin BAOAB integrator

  Background:
    Given a CUDA-capable GPU available as device 0
    And a SimulationBox with lx=ly=lz=1.0e6 unless otherwise specified
      (large enough that no particle in a scenario below wraps during a
      single drift call, so image flags stay at zero and positions match
      the closed-form arithmetic).
    And init_device() has been called

  # --- Module loading and construction ---

  @rq-662fccc1
  Scenario: init_device exposes the Langevin kernels on the Kernels handle
    When init_device() is called
    Then the returned GpuContext's kernels handle exposes the lan_drift_half function
    And the kernels handle exposes the lan_ou_step function

  @rq-457b5271
  Scenario: Construct LangevinBaoabState
    Given an IntegratorKind::LangevinBaoab { friction: 1.0e12, temperature: 300.0, seed: 42 }
    When Integrator::new(device, particle_count=4, &kind) is called
    Then it returns Ok(Integrator::LangevinBaoab(state))
    And state's stored friction equals 1.0e12
    And state's stored temperature equals 300.0
    And state's stored seed equals 42

  @rq-e9994f86
  Scenario: Construct with particle_count = 0
    Given an IntegratorKind::LangevinBaoab { friction: 1.0e12, temperature: 300.0, seed: 42 }
    When Integrator::new(device, particle_count=0, &kind) is called
    Then it returns Ok(integrator)

  # --- lan_drift_half kernel ---

  @rq-358de3e6
  Scenario: lan_drift_half advances positions by v * dt/2
    Given a ParticleBuffers from a single particle at x=(1, 2, 3) with v=(0.5, -0.25, 0.125)
    When lan_drift_half(&mut buffers, dt=0.1) is called
    And the buffers are downloaded into a host ParticleState
    Then positions equal (1 + 0.5*0.05, 2 + -0.25*0.05, 3 + 0.125*0.05)
    And velocities equal (0.5, -0.25, 0.125) unchanged

  @rq-5e8125ac
  Scenario: lan_drift_half leaves velocities, forces, masses unchanged
    Given a ParticleBuffers with N=4 known nonzero values
    And a snapshot of velocities, forces, masses, particle_ids
    When lan_drift_half(&mut buffers, dt=0.1) is called
    And the buffers are downloaded
    Then velocities, forces, masses, particle_ids match the snapshot byte-for-byte

  @rq-2680918c
  Scenario: lan_drift_half on empty state is a no-op
    Given a ParticleBuffers with particle_count() == 0
    When lan_drift_half(&mut buffers, dt=0.1) is called
    Then it returns Ok(())

  @rq-247e3799
  Scenario: dt = 0 leaves positions unchanged
    Given a ParticleBuffers with N=4 known nonzero positions and velocities
    And a snapshot of positions
    When lan_drift_half(&mut buffers, dt=0.0) is called
    And buffers are downloaded
    Then positions match the snapshot byte-for-byte

  # --- lan_ou_step kernel ---

  @rq-41389685
  Scenario: lan_ou_step with friction = 0 (alpha = 1) is identity on velocities (sanity)
    Given a ParticleBuffers with N=4 known nonzero velocities
    When lan_ou_step(&mut buffers, seed=1, draw_counter=1, alpha=1.0, kt=0.0) is called
    And buffers are downloaded
    Then velocities match the snapshot byte-for-byte
    # Note: kt=0 is the testable sentinel; the runtime config rejects friction=0
    # via the framework, so this scenario is exercised through direct kernel call only.

  @rq-01813ffa
  Scenario: lan_ou_step on empty state is a no-op
    Given a ParticleBuffers with particle_count() == 0
    When lan_ou_step(&mut buffers, seed=1, draw_counter=1, alpha=0.5, kt=1.0) is called
    Then it returns Ok(())

  @rq-9922b639
  Scenario: Two identical calls produce byte-identical velocities
    Given two independent ParticleBuffers built from identical ParticleStates of N=64
    When lan_ou_step is called on each with seed=42, draw_counter=7, identical alpha and kt
    Then run A's velocities_x, velocities_y, velocities_z agree byte-for-byte with run B's

  @rq-10652c60
  Scenario: Different seeds give different velocity outputs
    Given two ParticleBuffers built from identical inputs of N=64
    When lan_ou_step is called on each with the same draw_counter and parameters
      but seed=1 and seed=2
    Then the resulting velocities differ on at least 90% of components

  @rq-e2f2de4f
  Scenario: Different draw_counter values give different velocity outputs
    Given two ParticleBuffers built from identical inputs of N=64
    When lan_ou_step is called on each with the same seed and parameters
      but draw_counter=1 and draw_counter=2
    Then the resulting velocities differ on at least 90% of components

  @rq-50baca8c
  Scenario: Variance scales with sqrt((1 - alpha^2) * kt / m)
    Given a ParticleBuffers of N=10000 particles all with v=0, m=7.28e4
      m_e (the engine's argon mass in atomic units)
    And alpha = 0.5, kt = 9.5e-4 (≈ k_B · 300 K in Hartrees)
    When lan_ou_step is called with seed=42, draw_counter=1
    And velocities are downloaded
    Then the sample variance of each axis is within 5% of (1 - alpha^2) * kt / m

  # --- Slot integration ---

  @rq-e1dd0625
  Scenario: One plan walk launches all six expected kernel calls
    Given a Langevin-BAOAB integrator with friction=1e12, temperature=300, seed=1
    And a ParticleBuffers with N=4 nonzero values
    And a warm-up force evaluation has populated forces
    When the runner walks integrator.plan(dt=1e-15) once
    And timings.finalize() is queried
    Then KernelStage::LANGEVIN_KICK_HALF has count == 2  (one before the drifts, one after the force eval)
    And KernelStage::LANGEVIN_DRIFT_HALF has count == 2
    And KernelStage::LANGEVIN_OU_STEP has count == 1
    And KernelStage::LJ_PAIR_FORCE has count == 1
    And KernelStage::REDUCE_PAIR_FORCES has count == 1

  @rq-6e98222c
  Scenario: Plan walk on empty Langevin state is a no-op
    Given a Langevin-BAOAB integrator with friction=1e12, temperature=300, seed=1
    And a ParticleBuffers with particle_count() == 0
    When the runner walks integrator.plan(dt=1e-15) once
    Then every execute(...) call returns Ok(())
    And no kernel launches are recorded

  @rq-01784049
  Scenario: draw_counter starts at 0 and increments per plan walk
    Given a freshly built LangevinBaoabState
    Then state.draw_counter equals 0
    When the runner walks integrator.plan(dt) once (executing the "O" Custom sub-step)
    Then state.draw_counter equals 1
    When the runner walks integrator.plan(dt) a second time
    Then state.draw_counter equals 2

  @rq-e70ee09e
  Scenario: Two LangevinBaoabState instances at the same draw_counter and seed produce identical OU draws
    Given LangevinBaoabState A and B built from the same IntegratorKind
    And A.draw_counter and B.draw_counter both equal 5
    And identical buffer state and dt
    When the runner walks plan(dt) once on each
    Then the post-call velocities of A and B agree byte-for-byte

  # --- End-to-end determinism ---

  @rq-874dbfec
  Scenario: Two end-to-end Langevin runs with identical configs produce byte-identical outputs
    Given two complete simulations with kind="langevin-baoab", identical friction, temperature, seed,
      identical initial particle state, n_steps=10
    When dynamics run is invoked on each
    Then the two trajectory files are byte-identical
    And the two log files are byte-identical

  @rq-2a3f0e9b
  Scenario: Different seeds produce different trajectories
    Given two complete simulations identical except integrator.seed = 1 and integrator.seed = 2
    When dynamics run is invoked on each
    Then the two trajectory files differ

  # --- Temperature target ---

  @rq-adc3a32f
  Scenario: Equilibrium kinetic energy approaches ((3N − n_constraints) / 2) · k_B · T
    Given a Langevin run with N=512, friction=1.0e13, temperature=300, seed=42,
      initial v = 0, dt = 1.0e-15, n_steps=1000, n_constraints=0
    When the run completes
    Then the kinetic energy reported by the final log row is within 15% of
      ((3 * N − n_constraints) / 2) · k_B · 300
    (Langevin couples each Cartesian velocity component to an independent
    Ornstein-Uhlenbeck noise and does not conserve total momentum, so the
    equilibrium kinetic energy corresponds to 3N − n_constraints thermal
    DOFs. The reported `temperature` log column uses the COM-removed
    convention 3N − n_constraints − 3, which differs from this DOF count
    by a factor (3N − n_constraints − 3) / (3N − n_constraints) — close
    to 1 for any N of practical interest.)

  # --- Empty system ---

  @rq-1c729f15
  Scenario: Langevin runs through the runner with N=0
    Given a config with kind="langevin-baoab" and an init file with N=0
    When dynamics run is invoked
    Then it exits with code 0
    And the timings file contains no Langevin kernel rows

  # --- Image-flag wrap in lan_drift_half ---

  @rq-7cd5fae2
  Scenario: lan_drift_half wraps positions across the +L/2 boundary
    Given a SimulationBox with lx=ly=lz=10.0
    And a ParticleBuffers from a single particle at x=(4.95, 0.0, 0.0)
      with v=(2.0, 0.0, 0.0) and zero image flags
    When lan_drift_half(&mut buffers, &sim_box, dt=0.1) is called
    And the buffers are downloaded
    Then positions_x[0] equals -4.95
      (raw position 4.95 + 2.0 * 0.05 = 5.05; wrap subtracts lx = 10.0)
    And images_x[0] equals 1

  @rq-d6e89324
  Scenario: lan_drift_half does not modify image flags when no wrap occurs
    Given a SimulationBox with lx=ly=lz=10.0
    And a ParticleBuffers from a single particle at x=(0.0, 0.0, 0.0)
      with v=(0.1, 0.1, 0.1) and images_x[0]=3, images_y[0]=-1, images_z[0]=0
    When lan_drift_half(&mut buffers, &sim_box, dt=0.1) is called
    And the buffers are downloaded
    Then images_x[0] equals 3
    And images_y[0] equals -1
    And images_z[0] equals 0
```
