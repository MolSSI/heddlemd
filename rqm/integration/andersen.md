# Feature: Andersen Stochastic Thermostat <!-- rq-5e059f6b -->

The Andersen thermostat (Andersen, *J. Chem. Phys.* **72**, 2384
(1980)) is a stochastic NVT integrator. One of the pluggable
integrator slots (see `framework.md`); selected by
`kind = "andersen"` in the config's `[integrator]` section.

Each timestep, after the velocity-Verlet propagation, every particle
is independently and randomly assigned a new velocity drawn from the
Maxwell-Boltzmann distribution at the user-specified temperature with
probability `p = collision_rate · dt`. Particles that are not selected
retain the velocity produced by the symplectic step. Repeated
application produces the canonical distribution exactly.

The thermostat preserves no momentum: per-particle resampling makes
each draw independent, so the centre-of-mass momentum drifts
stochastically. Time-correlation functions are distorted by the
collisions in proportion to `collision_rate`. Andersen is the
strongest sampler in the registry — it converges to the canonical
distribution from any starting condition — and the weakest at
preserving the underlying dynamics.

## Algorithm <!-- rq-e15cc5ac -->

For each timestep of size `dt`:

1. Run the velocity-Verlet sub-step (kick / drift / force / kick), see
   `velocity-verlet.md`.
2. Compute the instantaneous kinetic energy `K_old` via the shared
   `compute_kinetic_energy` helper (`nose-hoover-chain.md`).
3. For each particle `i`:
   - Draw a uniform `U_i ∈ (0, 1)` from a counter-based Philox-4×32-10
     stream (see *RNG* below).
   - If `U_i < p` where `p = clamp(collision_rate · dt, 0, 1)`:
     replace the particle's velocity with a fresh Maxwell-Boltzmann
     sample at temperature `T`:
     `v_i ← σ_i · (ξ_x, ξ_y, ξ_z)` with `σ_i = sqrt(k_B · T / m_i)`
     and `ξ_a ~ N(0, 1)` independently per axis.
   - Otherwise leave `v_i` unchanged.
4. Compute the instantaneous kinetic energy `K_new` via the same
   helper, and update the integrator's running
   `cumulative_injection += K_new − K_old`. Used by the
   `andersen_conserved` log column.

The per-particle resampling is a single GPU kernel launch
(`andersen_resample`). The two surrounding kinetic-energy reductions
each launch the shared `kinetic_energy_reduce` kernel.

The collision probability is clamped to `[0, 1]` so any
`collision_rate · dt ≥ 1` reduces to "always resample" (massive
Andersen). This is mathematically valid; the canonical-distribution
guarantee holds.

## Per-Step Kernel Sequence <!-- rq-7843f188 -->

Per timestep the Andersen integrator's `step()` runs the following in
fixed order:

| Order | Step          | Kernel / call          | Operation                                            | Stage label              |
| ----- | ------------- | ---------------------- | ---------------------------------------------------- | ------------------------ |
| 1     | VV kick-drift | `vv_kick_drift`        | `v += (F/m)(dt/2)`, then `x += v dt`                 | `VvKickDrift`            |
| 2     | Force eval    | `force_field.step`     | recompute `F(x)`                                     | (slot-specific)          |
| 3     | VV kick       | `vv_kick`              | `v += (F/m)(dt/2)`                                   | `VvKick`                 |
| 4     | KE reduce     | `kinetic_energy_reduce` | one f32 scalar of `K_old`                            | `KineticEnergyReduce`    |
| 5     | Resample      | `andersen_resample`    | per-particle Bernoulli + Maxwell-Boltzmann draw      | `AndersenResample`       |
| 6     | KE reduce     | `kinetic_energy_reduce` | one f32 scalar of `K_new`                            | `KineticEnergyReduce`    |

`vv_kick_drift`, `vv_kick`, and `kinetic_energy_reduce` are reused
from the existing infrastructure. `andersen_resample` is the only
new CUDA kernel introduced by this feature.

## Parameters <!-- rq-eb0bc993 -->

Config layer fields set on `IntegratorKind::Andersen`:

- `temperature: f64` — bath temperature `T` in kelvin. Required.
  Finite and strictly positive. Independent of
  `simulation.temperature`, which governs the initial-velocity
  sampler.
- `collision_rate: f64` — per-particle stochastic collision frequency
  `ν` in inverse seconds. Required. Finite and `≥ 0` (`0`
  degenerates to NVE — no resampling — and is permitted as a
  diagnostic mode). Typical values for liquid water are `10¹¹–10¹²
  s⁻¹` (collision probability `≈ 10⁻⁴–10⁻²` per fs).
- `seed: u64` — counter-based RNG seed. Required, independent of
  `simulation.seed` and any other integrator's seed.

The per-step collision probability is computed as
`p = clamp(collision_rate · dt, 0.0, 1.0)`.

## RNG <!-- rq-b3086891 -->

Andersen draws, per particle per step, one uniform sample for the
Bernoulli decision and three standard-normal samples (one per axis)
when the decision accepts. All draws come from the same Philox-4×32-10
RNG used by `langevin-baoab.md` and `csvr.md`; the device-side
helpers `philox4x32_10`, `philox_gaussian`, and `u32_to_uniform_open`
live in `kernels/philox.cuh` and are shared by every RNG-consuming
kernel.

### Counter packing <!-- rq-d6a1b8eb -->

Each per-(particle, draw-kind) Philox invocation uses:

- **Key (2 × u32)**: `(seed_lo, seed_hi)` — low and high halves of
  the integrator's `seed`.
- **Counter (4 × u32)**:
  - `counter[0] = draw_counter_lo` — low 32 bits of the integrator's
    `draw_counter`.
  - `counter[1] = draw_counter_hi` — high 32 bits.
  - `counter[2] = particle_id` — the particle's `u32` ID from
    `ParticleBuffers::particle_ids`.
  - `counter[3] = draw_kind`:
    - `0` — Gaussian for the x-axis.
    - `1` — Gaussian for the y-axis.
    - `2` — Gaussian for the z-axis.
    - `3` — uniform for the Bernoulli decision.

The `draw_counter` lives on `AndersenState`. It starts at `0` at
construction and is pre-incremented on every `step()` call before the
kernel launch; the first launch in a run therefore uses
`draw_counter == 1`.

A non-colliding particle still consumes the uniform draw (kind `3`)
but does not consume the three Gaussians (kinds `0–2`). The kernel
nonetheless uses a fixed counter packing — collision and non-collision
particles share the same enumeration of `(particle_id, draw_kind)`
pairs — so two runs with identical seeds make byte-identical
decisions and produce byte-identical post-step velocities.

### Reproducibility <!-- rq-6fb62a59 -->

Two runs with identical `(seed, temperature, collision_rate)` and
identical initial particle state on the same GPU produce
byte-identical trajectory and log files. Philox is stateless; the
per-particle Bernoulli decision is a pure function of `(seed,
draw_counter, particle_id)`; and the Maxwell-Boltzmann draws use the
same per-axis convention as `langevin-baoab.md`.

## Andersen conserved quantity <!-- rq-bfa0cc5a -->

The diagnostic for Andersen is

```text
H_andersen = K + U − Σ_{steps} (K_new − K_old)
```

i.e., the physical Hamiltonian plus a running subtraction of the
cumulative kinetic energy injected (or removed) by the resampling
over all completed steps. With a correct implementation
`{H_andersen(t)}` drifts as a martingale (zero-mean increment per
step) and is bounded in expectation; an implementation bug produces
a systematic drift.

`H_andersen` is exposed as a per-log-row diagnostic column named
`andersen_conserved` when Andersen is the configured integrator (see
`io/log-output.md`). The integrator accumulates `Σ (K_new − K_old)`
on its host-side state across every `step()` call from the two
kinetic-energy reductions in steps 4 and 6 of the per-step sequence
above.

## Empty State and degenerate cases <!-- rq-645252f1 -->

- `particle_count == 0`: `step()` returns `Ok(())` without launching
  any kernel.
- `collision_rate == 0.0`: `p == 0`. No particle is ever resampled;
  the integrator degenerates to NVE velocity-Verlet (plus two
  redundant KE reductions per step). Useful as a diagnostic baseline.
- `collision_rate · dt ≥ 1.0`: `p` is clamped to `1.0`. Every
  particle is resampled every step ("massive Andersen"). The KE
  before/after the resample have entirely independent expected
  values; `H_andersen` is still a valid diagnostic.

## Feature API <!-- rq-4fad2dd6 -->

### Types <!-- rq-46792e3e -->

- `AndersenState` — implements the `Integrator` trait declared in <!-- rq-feba0a88 -->
  `framework.md`. Registered in `IntegratorRegistry::with_builtins`
  under `kind_name() == "andersen"`. Fields:

  - `device: Arc<CudaDevice>`
  - `temperature: f64`
  - `collision_rate: f64`
  - `seed: u64`
  - `draw_counter: u64` — Philox counter advance for the per-step
    resample kernel. Initialised to `0`; pre-incremented on every
    `step()` call.
  - `kt: f64` — `BOLTZMANN_J_PER_K · temperature`.
  - `cumulative_injection: f64` — running sum of `K_new − K_old`
    across every completed `step()` call. Initialised to `0.0`. Used
    by `log_column_values`.
  - `ke_scratch: CudaSlice<f32>` — length-1 device buffer for the
    kinetic-energy reduction; reused across calls.
  - `most_recent_ke: f64` — last kinetic energy computed during the
    current `step()`.

  All fields private; the slot's public surface is the `Integrator`
  trait methods and construction via `AndersenBuilder`. The
  `draw_counter` and `cumulative_injection` fields are public for
  parity with the other registered integrators so a future
  restart-from-checkpoint flow can restore them explicitly.

- `AndersenBuilder` — implements `IntegratorBuilder` with <!-- rq-fd0cef60 -->
  `kind_name() == "andersen"`. `build(device, particle_count, kind)`
  matches against `IntegratorKind::Andersen { … }`, allocates the
  length-1 `ke_scratch` device buffer, and returns the boxed
  `AndersenState`.

### `Integrator` trait overrides <!-- rq-3ace72b0 -->

`AndersenState` overrides the diagnostic-column trait methods
declared in `framework.md`:

- `log_column_names() -> &'static ["andersen_conserved"]`. <!-- rq-1163481e -->
- `log_column_values(ke, pe) -> vec![ke + pe − cumulative_injection]`. <!-- rq-6d2daea0 -->

### CUDA Kernels <!-- rq-b7c6d7d8 -->

`kernels/andersen.cu` declares one `extern "C"` kernel and
`#include`s `kernels/philox.cuh` for the shared Philox device-side
helpers:

```c
extern "C" __global__ void andersen_resample(
    float *velocities_x, float *velocities_y, float *velocities_z,
    const float *masses,
    const unsigned int *particle_ids,
    unsigned int seed_lo, unsigned int seed_hi,
    unsigned int draw_counter_lo, unsigned int draw_counter_hi,
    float p_collision,        // clamped to [0, 1] by the host
    float kt,                 // k_B * temperature, in J
    unsigned int n);
```

Each thread maps to one particle `i = blockIdx.x · blockDim.x +
threadIdx.x`. If the index is `≥ n` the thread returns. Otherwise:

1. Read `pid = particle_ids[i]` and `m = masses[i]`.
2. Draw the uniform: invoke `philox4x32_10(seed_lo, seed_hi,
   draw_counter_lo, draw_counter_hi, pid, 3u, …)`, take the first
   output via `u32_to_uniform_open` to obtain `U ∈ (0, 1)`.
3. If `U ≥ p_collision`, return without modifying any velocity
   component.
4. Otherwise compute `sigma = sqrtf(kt / m)` and three independent
   Gaussians via `philox_gaussian(seed_lo, seed_hi, draw_counter_lo,
   draw_counter_hi, pid, axis)` for `axis ∈ {0, 1, 2}`. Write
   `velocities_{x,y,z}[i] = sigma * xi_{x,y,z}`.

No interaction between threads; trivially deterministic.

### Shared Philox header <!-- rq-4ac937ef -->

`kernels/philox.cuh` carries the device-side Philox-4×32-10
primitives:

```c
__device__ inline unsigned int mulhi32(unsigned int a, unsigned int b);
__device__ inline void philox4x32_10(
    unsigned int key_lo, unsigned int key_hi,
    unsigned int ctr0, unsigned int ctr1, unsigned int ctr2, unsigned int ctr3,
    unsigned int *out0, unsigned int *out1, unsigned int *out2, unsigned int *out3);
__device__ inline double u32_to_uniform_open(unsigned int x);
__device__ inline float philox_gaussian(
    unsigned int seed_lo, unsigned int seed_hi,
    unsigned int step_lo, unsigned int step_hi,
    unsigned int particle_id, unsigned int axis_id);
```

The header carries no PTX module of its own and `init_device`
performs no `load_ptx` call for it. `kernels/langevin.cu` and
`kernels/andersen.cu` both `#include "philox.cuh"`; nvcc inlines the
helpers into each translation unit. The algorithm description
(constants, round structure, Box-Muller transform) lives in
`langevin-baoab.md` under *RNG*; this header is its single
implementation site.

### PTX Module Loading <!-- rq-73a3b802 -->

`init_device()` loads the compiled `kernels/andersen.cu` PTX as
module `"andersen"` and captures its `andersen_resample` function
into the `Kernels` handle (see `build-pipeline.md`).

### Rust Launch Helpers <!-- rq-91a4634b -->

One free function in `src/gpu/kernels.rs`, re-exported from
`crate::gpu`:

- `andersen_resample(buffers: &mut ParticleBuffers, seed: u64, draw_counter: u64, p_collision: f32, kt: f32) -> Result<(), GpuError>` <!-- rq-da36d746 -->
  - Launches the `andersen_resample` kernel with the seed and
    draw_counter packed into `(seed_lo, seed_hi, draw_counter_lo,
    draw_counter_hi)` u32 pairs.
  - Block size 256; grid `ceil(n / 256)`.
  - When `buffers.particle_count() == 0`, returns `Ok(())` without
    launching.
  - Invokes the kernel through the `Kernels` handle, like the other
    launch helpers; performs no string-keyed kernel lookup of its own
    (see `build-pipeline.md`).
  - Debug-asserts that `p_collision ∈ [0.0, 1.0]` on entry. The
    caller is responsible for clamping `collision_rate · dt` before
    calling this helper.

## Launch Configuration <!-- rq-af2461ae -->

- `andersen_resample`: block size 256, grid `ceil(n / 256)`. Shared
  memory: 0 bytes.
- `kinetic_energy_reduce`: documented in `nose-hoover-chain.md`.
- Stream: the default stream carried by `ParticleBuffers::device`.

## Determinism <!-- rq-37b05f03 -->

- All three reused kernels (`vv_kick_drift`, `vv_kick`,
  `kinetic_energy_reduce`) are deterministic by construction (see
  their respective documentation).
- `andersen_resample` reads only `(seed, draw_counter, particle_id)`
  for its Philox input; no per-thread RNG state. Two runs on the same
  GPU with identical inputs produce byte-identical velocities.
- The `draw_counter` advances by exactly `+1` per `step()` call.
- Two end-to-end Andersen runs with identical configs on the same
  GPU produce byte-identical trajectory and log files, including the
  `andersen_conserved` column.

## Out of Scope <!-- rq-c5f66ecd -->

- A lossless `(f32, f64)` compensated mode. Andersen is stochastic;
  bit-exact time-reversibility is not a property the algorithm
  promises.
- COM-momentum-preserving variant. Andersen intrinsically breaks
  momentum conservation; a hybrid scheme that subtracts a global
  drift after each resample would be a different integrator.
- Per-type collision rates. The single `collision_rate` applies to
  every particle regardless of species.
- Massive Andersen as a distinct `kind` name. Setting
  `collision_rate · dt ≥ 1` (clamped to `p = 1`) achieves the same
  effect under the existing `"andersen"` slot.
- Constraint algorithms (SHAKE/RATTLE).
- Pressure coupling.
- An off-by-one `draw_counter` mode for restart-from-checkpoint
  alignment; the `draw_counter` is a public field for explicit
  restoration.

---

## Gherkin Scenarios <!-- rq-202a5a1a -->

```gherkin
Feature: Andersen stochastic thermostat

  Background:
    Given a CUDA-capable GPU available as device 0
    And a SimulationBox with lx=ly=lz=1.0e6 unless otherwise specified
    And init_device() has been called

  # --- Construction ---

  @rq-dc3c616a
  Scenario: Construct AndersenState via the registry
    Given an IntegratorKind::Andersen {
      temperature: 300.0, collision_rate: 1.0e12, seed: 42 }
    When registry.build(&kind, device, particle_count=4) is called
    Then it returns Ok(integrator)
    And the underlying AndersenState has draw_counter == 0
    And state.cumulative_injection == 0.0
    And state.kt == k_B * 300.0

  @rq-abcae430
  Scenario: Construct with particle_count = 0
    Given an IntegratorKind::Andersen {
      temperature: 300.0, collision_rate: 1.0e12, seed: 1 }
    When registry.build(&kind, device, particle_count=0) is called
    Then it returns Ok(integrator)

  @rq-62ad70f8
  Scenario: Construct with collision_rate = 0 (NVE-equivalent)
    Given an IntegratorKind::Andersen {
      temperature: 300.0, collision_rate: 0.0, seed: 1 }
    When registry.build(&kind, device, particle_count=4) is called
    Then it returns Ok(integrator)

  # --- Config validation ---

  @rq-b8aa57c6
  Scenario: Reject non-positive temperature
    Given a config with [integrator] kind="andersen",
      temperature=0.0, collision_rate=1.0e12, seed=1
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue {
      field: "integrator.temperature", reason: _ })

  @rq-0f3b352b
  Scenario: Reject negative collision_rate
    Given a config with [integrator] kind="andersen",
      temperature=300.0, collision_rate=-1.0, seed=1
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue {
      field: "integrator.collision_rate", reason: _ })

  @rq-c4581536
  Scenario: collision_rate = 0 accepted
    Given a config with [integrator] kind="andersen",
      temperature=300.0, collision_rate=0.0, seed=1
    When load_config is called
    Then it returns Ok(config)

  @rq-c5b42daa
  Scenario: Missing seed rejected
    Given a config with [integrator] kind="andersen",
      temperature=300.0, collision_rate=1.0e12
    When load_config is called
    Then it returns Err(ConfigError::MissingField {
      field: "integrator.seed" })

  @rq-8df9d74b
  Scenario: Reject extra fields (e.g. tau from NHC/CSVR)
    Given a config with [integrator] kind="andersen",
      temperature=300.0, collision_rate=1.0e12, seed=1, tau=1.0e-13
    When load_config is called
    Then it returns Err(ConfigError::UnknownIntegratorField {
      kind: "andersen", field: "tau" })

  # --- andersen_resample kernel ---

  @rq-6ac1c0f2
  Scenario: andersen_resample with p_collision = 0 is identity
    Given a ParticleBuffers with N=8 known non-zero velocities
    And a snapshot of velocities
    When andersen_resample(seed=1, draw_counter=1, p_collision=0.0, kt=1.0) is launched
    And the buffers are downloaded
    Then velocities match the snapshot byte-for-byte

  @rq-4254b707
  Scenario: andersen_resample with p_collision = 1 resamples every particle
    Given a ParticleBuffers with N=8 known non-zero velocities
    When andersen_resample(seed=1, draw_counter=1, p_collision=1.0, kt=k_B*300) is launched
    And the buffers are downloaded
    Then every velocity component differs from the snapshot
    And the per-axis sample variance is consistent with sigma² = kt / m_i

  @rq-5ac172fc
  Scenario: andersen_resample on empty state is a no-op
    Given a ParticleBuffers with particle_count() == 0
    When andersen_resample(...) is called
    Then it returns Ok(())

  @rq-bacbf7d2
  Scenario: andersen_resample is deterministic across two runs
    Given two ParticleBuffers built from byte-identical ParticleStates of N=64
    When andersen_resample is launched on each with identical (seed, draw_counter, p_collision, kt)
    Then post-call velocities of run A and run B agree byte-for-byte

  @rq-c3564e8a
  Scenario: Different seeds yield different post-call velocities
    Given two ParticleBuffers built from identical inputs of N=64
    When andersen_resample is launched on each with the same draw_counter and parameters
      but seed=1 and seed=2
    Then the resulting velocities differ on at least 90% of components

  @rq-8040ce8a
  Scenario: Different draw_counters yield different post-call velocities
    Given two ParticleBuffers built from identical inputs of N=64
    When andersen_resample is launched on each with the same seed
      but draw_counter=1 and draw_counter=2
    Then the resulting velocities differ on at least 90% of components

  # --- Per-step kernel sequence (slot integration) ---

  @rq-cef43ff0
  Scenario: step() launches all expected kernels
    Given an Andersen integrator with temperature=300, collision_rate=1e12,
      seed=1, particle_count=4
    And a ForceField with one LennardJones slot
    And a warm-up force evaluation has populated forces
    When integrator.step(...) is called with dt=1e-15
    And timings.finalize() is queried
    Then KernelStage::VV_KICK_DRIFT has count == 1
    And KernelStage::VV_KICK has count == 1
    And KernelStage::KINETIC_ENERGY_REDUCE has count == 2
    And KernelStage::ANDERSEN_RESAMPLE has count == 1

  @rq-8fdfc981
  Scenario: step() on empty state is a no-op
    Given an Andersen integrator with particle_count=0
    When integrator.step(...) is called
    Then it returns Ok(())

  # --- draw_counter and cumulative_injection bookkeeping ---

  @rq-c814659f
  Scenario: draw_counter starts at 0 and increments by 1 per step
    Given a freshly built AndersenState
    Then state.draw_counter == 0
    When integrator.step(...) is called once
    Then state.draw_counter == 1
    When integrator.step(...) is called a second time
    Then state.draw_counter == 2

  @rq-b1e87ce4
  Scenario: cumulative_injection records K_new − K_old per step
    Given an Andersen integrator and a ParticleBuffers with known non-zero velocities
    And a ForceField with zero slots (no force contribution)
    When integrator.step(...) is called once
    Then state.cumulative_injection equals K_new − K_old
      (with K_new and K_old measured at the two KE reductions in that step)
      to f64 round-off

  # --- Log columns ---

  @rq-8eb14902
  Scenario: log_column_names returns ["andersen_conserved"]
    Given a constructed AndersenState
    Then state.log_column_names() equals ["andersen_conserved"]

  @rq-26ff4aea
  Scenario: log_column_values returns ke + pe − cumulative_injection
    Given an AndersenState with cumulative_injection = 1.0e-20
    When state.log_column_values(ke=2.5e-20, pe=3.0e-20) is called
    Then it returns [2.5e-20 + 3.0e-20 − 1.0e-20] = [4.5e-20]

  @rq-c50c6f84
  Scenario: Log file header includes andersen_conserved when Andersen is the integrator
    Given a config with [integrator].kind = "andersen"
    And log_every > 0
    When the runner produces the log file
    Then its header line is "step,time,kinetic_energy,temperature,andersen_conserved"

  # --- p_collision clamping ---

  @rq-c9865e4c
  Scenario: collision_rate · dt > 1 is clamped to p = 1 (massive Andersen)
    Given an Andersen integrator with collision_rate=1.0e16
    And dt=1.0e-15 (so collision_rate · dt = 10)
    And a ParticleBuffers with N=4 known velocities
    When integrator.step(...) is called
    Then every particle is resampled (post-call velocities differ on every component
      with very high probability)

  @rq-1eaff437
  Scenario: collision_rate = 0 leaves velocities unchanged by the resample
    Given an Andersen integrator with collision_rate=0
    And a ParticleBuffers with N=4 known velocities
    And a ForceField with zero slots
    When integrator.step(...) is called
    Then velocities are byte-identical to the post-VV state (no resample occurred)

  # --- Determinism ---

  @rq-f062ec85
  Scenario: Two independent Andersen runs with identical seeds produce byte-identical outputs
    Given two complete simulations with kind="andersen", identical parameters
      (including identical seed), identical initial state, n_steps=10
    When dynamics run is invoked on each
    Then the trajectory files are byte-identical
    And the log files are byte-identical, including the andersen_conserved column

  @rq-20daf925
  Scenario: Different seeds produce different trajectories
    Given two complete simulations identical except integrator.seed = 1 and = 2
    When dynamics run is invoked on each
    Then the trajectory files differ

  # --- Physical correctness ---

  @rq-536457be
  Scenario: Time-averaged kinetic energy tracks (N_f / 2) k_B T
    Given an Andersen run with N=128 LJ particles, temperature=300,
      collision_rate=1.0e13, dt=1.0e-15, n_steps=2000, seed=1
    When the run completes
    Then the time-averaged kinetic energy over the last 1000 log rows
      is within 5% of (3 * N / 2) · k_B · 300

  @rq-299112e9
  Scenario: Variance of resampled velocity components matches MB
    Given a 1000-particle system with collision_rate · dt = 1 (massive)
    And dummy initial velocities (e.g. all zero)
    When integrator.step(...) is called once
    Then the sample variance of each velocity component
      is within 5% of k_B · T / m
```
