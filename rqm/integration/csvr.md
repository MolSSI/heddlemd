# Feature: CSVR Stochastic Velocity Rescaling Thermostat <!-- rq-891232bf -->

Canonical Sampling through Velocity Rescaling (CSVR;
Bussi-Donadio-Parrinello, *J. Chem. Phys.* **126**, 014101 (2007)) is a
stochastic NVT integrator. One of the pluggable integrator slots (see
`framework.md`); selected by `kind = "csvr"` in the config's
`[integrator]` section.

The thermostat applies a single global velocity rescale per step
*after* the velocity-Verlet propagation. The rescale factor is drawn
from a Markov kernel that, after equilibration, samples the canonical
distribution at the user-specified temperature exactly. CSVR has no
ergodicity caveat for stiff systems (unlike vanilla Nosé-Hoover) and
perturbs the underlying dynamics less than Langevin (no per-particle
friction term).

## Algorithm <!-- rq-062ea284 -->

For each timestep of size `dt` with `c = exp(−dt/τ)`:

1. Run the velocity-Verlet sub-step (kick / drift / force / kick), see
   `velocity-verlet.md`.
2. Compute the instantaneous kinetic energy
   `K = (1/2) Σ_i m_i |v_i|²` using the deterministic GPU reduction
   documented in `nose-hoover-chain.md`.
3. Draw `N_f` independent standard-normal samples on the host, where
   `N_f = max(1, 3·N − 3)` is the number of thermostatted degrees of
   freedom. Designate the first sample as `R` and the remaining
   `N_f − 1` samples' squared sum as `S`:

   ```text
   R = ξ_0
   S = ξ_1² + ξ_2² + … + ξ_{N_f−1}²
   ```

4. Compute the new target kinetic energy (Bussi-Donadio-Parrinello
   2007 / PLUMED `csvr.cc`):

   ```text
   K_target = (N_f / 2) · k_B · T
   K_new    = c · K
              + (K_target / N_f) · (1 − c) · (S + R²)
              + 2 · R · sqrt(c · (1 − c) · K · K_target / N_f)
   ```

   The `(S + R²)` term is `chi²(N_f)` since adding `R²` to
   `S ∼ chi²(N_f − 1)` makes a chi-squared with `N_f` degrees of
   freedom. At equilibrium (`K = K_target`), `E[S + R²] = N_f` gives
   `E[K_new] = K_target` exactly.

5. The velocity-rescale factor is `α = sqrt(K_new / K)` when `K > 0`;
   when `K = 0` (every velocity is exactly zero) the formula
   degenerates and the rescale is skipped. The CSVR step also produces
   a sign convention: the rescale factor is positive only when the
   discriminant `K_new` is non-negative. When `K_new` ends up
   negative (extremely unlikely for `N_f` of physical size; possible
   only for `N_f < 5` with adversarial draws) the kernel takes
   `K_new = K` (no rescale this step). This guard keeps the
   integrator robust for tiny systems.

6. Apply the rescale with the shared `rescale_velocities` helper
   (documented in `nose-hoover-chain.md`):
   `v_i ← α · v_i` for every particle `i` and axis.

The CSVR Markov kernel is exact in the sense that repeated
application from any non-zero initial `K` converges to the canonical
distribution `P(K) ∝ K^{(N_f/2)−1} · exp(−K / (k_B · T))`.

The instantaneous kinetic energy is the only piece of particle state
the thermostat reads; the chain has no auxiliary degrees of freedom.

## Per-Step Kernel Sequence <!-- rq-5f59fa80 -->

Per timestep the CSVR integrator's `step()` runs the following in
fixed order:

| Order | Step       | Kernel / call          | Operation                                          | Stage label              |
| ----- | ---------- | ---------------------- | -------------------------------------------------- | ------------------------ |
| 1     | VV kick-drift | `vv_kick_drift`     | `v += (F/m)(dt/2)`, then `x += v dt`               | `VvKickDrift`            |
| 2     | Force eval | `force_field.step`     | recompute `F(x)`                                   | (slot-specific)          |
| 3     | VV kick    | `vv_kick`              | `v += (F/m)(dt/2)`                                 | `VvKick`                 |
| 4     | KE reduce  | `kinetic_energy_reduce` | one f32 scalar of `K`                              | `KineticEnergyReduce`    |
| 5     | Host RNG + rescale | `rescale_velocities` (1 launch) | sample `R`, `S`, compute `α`, scale velocities | `CsvrRescaleVelocities` |

`vv_kick`, `vv_kick_drift`, `kinetic_energy_reduce`, and
`rescale_velocities` are all reused from the existing infrastructure;
the timings file labels CSVR's velocity-rescale call under
`CsvrRescaleVelocities` to distinguish it from
`NhcRescaleVelocities`. No new CUDA kernels.

The host-side work each step is `N_f` Philox-4×32-10 invocations + a
chi-squared sum + the `α` calculation; for `N_f` of order 10³ this is
< 10 µs on a modern CPU.

## Parameters <!-- rq-a8da85cf -->

Config layer fields set on `IntegratorKind::Csvr`:

- `temperature: f64` — bath temperature `T` in kelvin. Required.
  Finite and strictly positive. Independent of `simulation.temperature`,
  which governs the initial-velocity sampler.
- `tau: f64` — thermostat coupling time in seconds. Required. Finite
  and strictly positive. Larger `τ` leaves the dynamics closer to NVE
  (slow thermostat coupling); smaller `τ` enforces the target
  temperature more aggressively. Typical values for liquid water are
  100 fs – 1 ps.
- `seed: u64` — counter-based RNG seed. Required, independent of
  `simulation.seed` and any other integrator's seed. Two runs with
  identical configs on the same GPU produce byte-identical
  trajectories.

## RNG <!-- rq-54b99519 -->

CSVR draws `N_f` independent standard-normal samples per timestep. The
samples are generated by a counter-based Philox-4×32-10 RNG (Salmon
et al., SC11) on the **host**, using the same algorithm as the
device-side Philox in `langevin-baoab.md`. The host-side
implementation lives in `src/integrator.rs::philox_4x32_10`, exposed
to other host-side stochastic integrators as a stable utility.

### Counter packing <!-- rq-7c49aae5 -->

Each per-step draw of `N_f` normals consumes `N_f` Philox invocations
with:

- **Key (2 × u32)**: `(seed_lo, seed_hi)` — low and high halves of
  the integrator's `seed`.
- **Counter (4 × u32)**:
  - `counter[0] = draw_counter_lo` — low 32 bits of the integrator's
    `draw_counter`.
  - `counter[1] = draw_counter_hi` — high 32 bits.
  - `counter[2] = sample_index` — `0` for the lone `R` draw,
    `1..N_f−1` for the chi-squared components.
  - `counter[3] = 0` — reserved (matches Langevin's axis slot).

The `draw_counter` lives on `CsvrState`. It starts at `0` at
construction. Each `step()` performs `self.draw_counter += 1` before
the draws and the in-step loop uses the post-increment value across
all `N_f` Philox calls of that step. The first invocation in a run
therefore uses `draw_counter == 1`; the `N`-th step uses
`draw_counter == N`.

### Box-Muller transform <!-- rq-d4b9d831 -->

Each Philox call returns four `u32` values; CSVR uses the first two
to produce one standard normal via Box-Muller, matching Langevin's
convention exactly:

```text
u1 = (output[0] as f64 + 0.5) * 2^-32       # uniform in (0, 1)
u2 = (output[1] as f64 + 0.5) * 2^-32       # uniform in (0, 1)
ξ  = sqrt(-2 ln u1) * cos(2 π u2)            # standard normal (f64)
```

The second Box-Muller output (`sin` half) is discarded; `output[2]`
and `output[3]` are unused. The draw is computed in `f64` and used in
`f64` chain math, then converted to `f32` only for the final
`α` argument passed to `rescale_velocities`.

### Reproducibility <!-- rq-99a68e09 -->

Two runs with identical `(seed, temperature, tau)` and identical
initial particle state on the same GPU produce byte-identical
trajectory and log files. The Philox stream is stateless (no
algorithmic state to drift), and the host-side chi-squared sum
proceeds left-to-right in fixed sample-index order.

## CSVR conserved quantity <!-- rq-be551127 -->

The conserved diagnostic for CSVR is

```text
H_csvr = K + U − Σ_{steps} (K_new − K_old)
```

i.e., the physical Hamiltonian plus a running subtraction of the
cumulative kinetic energy injected (or removed) by the thermostat
over all completed steps. With a correct CSVR implementation the
sequence `{H_csvr(t)}` drifts as a martingale (zero-mean increment
per step) and is bounded in expectation; an implementation bug
produces a systematic drift instead.

`H_csvr` is exposed as a per-log-row diagnostic column named
`csvr_conserved` when CSVR is the configured integrator (see
`io/log-output.md`). The integrator accumulates `Σ (K_new − K_old)`
on its host-side state across every `step()` call and combines it
with the kinetic and potential energies supplied by the runner at
log-write time.

## Empty State and degenerate cases <!-- rq-2a95c46c -->

- `particle_count == 0`: `step()` returns `Ok(())` without launching
  any kernel.
- `particle_count == 1`: `N_f = max(1, 3 − 3) = 1`. The chi-squared
  sum is empty (`S = 0`); the lone `R` term still appears. The
  thermostat propagates a degenerate one-degree-of-freedom system as
  documented in the original paper.
- `K == 0` on entry: every velocity is exactly zero. The rescale
  formula contains `K` in a square-root term that would divide by
  zero; the kernel skips the rescale this step. The next step's
  velocities (carrying whatever forces injected during the VV kicks)
  will be non-zero and the rescale resumes.

## Feature API <!-- rq-b1a8a6ca -->

### Types <!-- rq-1c26a108 -->

- `CsvrState` — implements the `Integrator` trait declared in <!-- rq-47d91c7d -->
  `framework.md`. Registered in `IntegratorRegistry::with_builtins`
  under `kind_name() == "csvr"`. Fields:

  - `temperature: f64`
  - `tau: f64`
  - `seed: u64`
  - `draw_counter: u64` — Philox counter advance for the CSVR draws.
    Initialised to `0` by the builder; pre-incremented on every
    `step()` call.
  - `g_dof: u32` — `max(1, 3 · particle_count − 3)`.
  - `kt_target: f64` — `BOLTZMANN_J_PER_K · temperature`.
  - `cumulative_injection: f64` — running sum of `K_new − K_old` over
    every completed `step()` call. Initialised to `0.0`. Used by
    `log_column_values`.
  - `ke_scratch: CudaSlice<f32>` — length-1 device buffer for the
    kinetic-energy reduction; reused across calls.
  - `most_recent_ke: f64` — last kinetic energy computed during the
    current `step()`. Available to `log_column_values` to avoid a
    redundant download.
  - `particle_count: usize`

  All fields private; the slot's public surface is the `Integrator`
  trait methods (see `framework.md`) and the construction via
  `CsvrBuilder`. The `draw_counter` and `cumulative_injection`
  fields are public for parity with other integrators' state so a
  future restart-from-checkpoint flow can restore them explicitly.

- `CsvrBuilder` — implements `IntegratorBuilder` with <!-- rq-750b828f -->
  `kind_name() == "csvr"`. `build(device, particle_count, kind)`
  matches against `IntegratorKind::Csvr { … }`, allocates the
  length-1 `ke_scratch` device buffer, and returns the boxed
  `CsvrState`.

### `Integrator` trait overrides <!-- rq-5aae9633 -->

`CsvrState` overrides the diagnostic-column trait methods declared in
`framework.md`:

- `log_column_names() -> &'static ["csvr_conserved"]`. <!-- rq-8ee58ec1 -->
- `log_column_values(ke, pe) -> vec![ke + pe − self.cumulative_injection]`. <!-- rq-2a5de2ab -->

### Host-side RNG utility <!-- rq-3d7c8e53 -->

`src/integrator.rs::philox_4x32_10(key_lo: u32, key_hi: u32, ctr0:
u32, ctr1: u32, ctr2: u32, ctr3: u32) -> [u32; 4]` — host-side
Philox-4×32-10 implementation matching the device-side helper used by
`lan_ou_step` (see `langevin-baoab.md`). Pure function, deterministic
output for any fixed key/counter pair. Available to any other
host-side stochastic integrator that lands later (e.g. Andersen
thermostat).

The accompanying helper `philox_normal(key_lo, key_hi, ctr0, ctr1,
ctr2, ctr3) -> f64` wraps `philox_4x32_10` with the Box-Muller
transform documented under *RNG* above and returns one f64 standard
normal per call (discarding the `sin` half).

### CUDA kernels <!-- rq-2c834256 -->

CSVR introduces no new CUDA kernels. It reuses
`kinetic_energy_reduce` and `rescale_velocities` from
`nose-hoover-chain.md` through their public launch helpers
`compute_kinetic_energy` and `rescale_velocities` re-exported from
`crate::gpu`.

## Launch Configuration <!-- rq-cb0c9d24 -->

Per-step launch counts (excluding the force pipeline):

- `vv_kick_drift`: 1 launch (block 256, grid `ceil(n/256)`).
- `vv_kick`: 1 launch.
- `kinetic_energy_reduce`: 1 launch (single block of 256 threads).
- `rescale_velocities`: 1 launch (block 256, grid `ceil(n/256)`).

All launches go through the default stream of
`ParticleBuffers::device`.

## Determinism <!-- rq-72a606e3 -->

- The velocity-Verlet kernels and `rescale_velocities` are
  deterministic by construction (see their respective documentation).
- `kinetic_energy_reduce` uses a single-block deterministic reduction
  tree (see `nose-hoover-chain.md`).
- The host-side Philox stream is pure functional; the chi-squared sum
  proceeds in fixed left-to-right sample-index order in `f64`.
- The `draw_counter` advances by exactly `+1` per `step()`, so two
  runs that take the same number of steps with the same seed consume
  the same set of Philox counters.
- Two end-to-end CSVR runs on the same GPU with identical configs and
  identical initial particle state produce byte-identical trajectory
  and log files, including the `csvr_conserved` column.

## Out of Scope <!-- rq-1178e005 -->

- A lossless `(f32, f64)` compensated mode. CSVR is stochastic;
  bit-exact time-reversibility is not a property the algorithm
  promises.
- Pre/post symmetric splitting of the thermostat. The single
  post-VV variant is the canonical Bussi 2007 algorithm and is
  sufficient for correct canonical sampling.
- Massive thermostatting (per-atom independent CSVR). Single global
  thermostat only.
- User-overrideable `N_f` (degrees of freedom). The integrator
  hard-codes `N_f = max(1, 3N − 3)` to match the COM-removed
  initial-velocity convention; constraint-aware `N_f` follows the
  constraint feature.
- Pressure coupling (Bussi-Parrinello stochastic-cell NPT extension).
  A future stochastic barostat would slot in alongside CSVR; not in
  v1.
- Constraint algorithms (SHAKE/RATTLE).
- Wilson-Hilferty or Gamma-rejection approximations of the
  chi-squared distribution. The exact sum-of-squared-normals
  implementation is the only supported sampler; future opt-in
  approximations could land if `N_f` is large enough to make the
  exact sum dominate the step cost.
- A device-side RNG path for CSVR. The host-side path is fast enough
  for typical system sizes and simpler to reason about; a future
  GPU-side variant could be added if profiling demands it.
- Restart-from-checkpoint of the `draw_counter` and
  `cumulative_injection` via the config layer. The fields are public
  for direct assignment by future checkpoint code; the config layer
  carries no syntax for them today.

---

## Gherkin Scenarios <!-- rq-1775255c -->

```gherkin
Feature: CSVR stochastic velocity-rescaling thermostat

  Background:
    Given a CUDA-capable GPU available as device 0
    And a SimulationBox with lx=ly=lz=1.0e6 unless otherwise specified
    And init_device() has been called

  # --- Construction ---

  @rq-9e1142aa
  Scenario: Construct CsvrState via the registry
    Given an IntegratorKind::Csvr { temperature: 300.0, tau: 1.0e-13, seed: 42 }
    When registry.build(&kind, device, particle_count=4) is called
    Then it returns Ok(integrator)
    And the underlying CsvrState has draw_counter == 0
    And state.cumulative_injection == 0.0
    And state.g_dof == max(1, 3*4 − 3) == 9
    And state.kt_target == k_B * 300.0

  @rq-cf008c68
  Scenario: Construct with particle_count = 0
    Given an IntegratorKind::Csvr { temperature: 300.0, tau: 1.0e-13, seed: 1 }
    When registry.build(..., particle_count=0) is called
    Then it returns Ok(integrator)

  @rq-c4872e7e
  Scenario: Construct with particle_count = 1 (N_f = 1)
    Given an IntegratorKind::Csvr { temperature: 300.0, tau: 1.0e-13, seed: 1 }
    When registry.build(..., particle_count=1) is called
    Then it returns Ok(integrator)
    And state.g_dof == 1

  # --- Config validation (paired with config-schema scenarios) ---

  @rq-eba43990
  Scenario: Reject non-positive temperature
    Given a config with [integrator] kind="csvr", temperature=0.0, tau=1.0e-13, seed=1
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "integrator.temperature", reason: _ })

  @rq-e1a6bde9
  Scenario: Reject non-positive tau
    Given a config with [integrator] kind="csvr", temperature=300.0, tau=-1.0e-13, seed=1
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "integrator.tau", reason: _ })

  @rq-84927a79
  Scenario: Missing seed rejected
    Given a config with [integrator] kind="csvr", temperature=300.0, tau=1.0e-13
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "integrator.seed" })

  @rq-89d45c45
  Scenario: Reject extra fields (e.g., chain_length from NHC)
    Given a config with [integrator] kind="csvr", temperature=300.0, tau=1.0e-13, seed=1,
      chain_length=3
    When load_config is called
    Then it returns Err(ConfigError::UnknownIntegratorField { kind: "csvr", field: "chain_length" })

  # --- Host-side Philox parity with the device ---

  @rq-11a953dc
  Scenario: philox_4x32_10 produces the same outputs as the device-side helper
    Given a fixed (key, counter) quintuple (key_lo=1, key_hi=2,
      ctr0=3, ctr1=4, ctr2=5, ctr3=6)
    When philox_4x32_10(1, 2, 3, 4, 5, 6) is called on the host
    And the device-side `philox4x32_10` from langevin.cu is invoked with the
      same inputs (via the langevin OU kernel's call path, which exercises it)
    Then the two return the same four u32 values

  @rq-db1298bd
  Scenario: philox_normal returns the standard-normal Box-Muller cos branch
    Given key=(0, 0), counter=(0, 0, 0, 0)
    When philox_normal(0, 0, 0, 0, 0, 0) is called
    Then the returned f64 is the standard-normal Box-Muller cos
      computed from the same Philox output, matching the device-side draw
      formula in `langevin-baoab.md`

  # --- Per-step kernel sequence ---

  @rq-4e9e09f0
  Scenario: step() launches the expected kernel set
    Given a CSVR integrator with seed=1, temperature=300, tau=1e-13, particle_count=4
    And a ForceField with one LennardJones slot
    And a warm-up force evaluation has populated forces
    When integrator.step(...) is called with dt=1e-15
    And timings.finalize() is queried
    Then KernelStage::VV_KICK_DRIFT has count == 1
    And KernelStage::VV_KICK has count == 1
    And KernelStage::KINETIC_ENERGY_REDUCE has count == 1
    And KernelStage::CSVR_RESCALE_VELOCITIES has count == 1

  @rq-a2454a72
  Scenario: step() on empty CSVR state is a no-op
    Given a CSVR integrator with particle_count=0
    When integrator.step(...) is called
    Then it returns Ok(())

  # --- draw_counter advances per step ---

  @rq-1e5dcdc9
  Scenario: draw_counter starts at 0 and increments by 1 per step
    Given a freshly built CsvrState
    Then state.draw_counter == 0
    When integrator.step(...) is called once
    Then state.draw_counter == 1
    When integrator.step(...) is called again
    Then state.draw_counter == 2

  @rq-dc95802b
  Scenario: Two CsvrStates at the same draw_counter and seed produce identical
    velocity outputs after step()
    Given two CsvrStates A and B built from identical IntegratorKinds
    And both A.draw_counter and B.draw_counter == 5
    And identical buffer state
    When integrator.step(...) is called once on each
    Then post-call velocities of A and B are byte-identical

  # --- Log columns ---

  @rq-2c1bb918
  Scenario: log_column_names returns ["csvr_conserved"]
    Given a constructed CsvrState
    Then state.log_column_names() equals ["csvr_conserved"]

  @rq-ca0b98cb
  Scenario: log_column_values returns ke + pe − cumulative_injection
    Given a CsvrState with cumulative_injection = 1.0e-20
    When state.log_column_values(ke=2.5e-20, pe=3.0e-20) is called
    Then it returns [2.5e-20 + 3.0e-20 − 1.0e-20] = [4.5e-20]

  @rq-11b0deff
  Scenario: cumulative_injection accumulates K_new − K_old across calls
    Given a CsvrState with cumulative_injection = 0.0
    And K_old = 1.0e-20 from a synthetic buffer setup before step()
    When integrator.step(...) is called once
    Then state.cumulative_injection equals the (K_new − K_old) of that step
      to f64 round-off

  @rq-1d25a0db
  Scenario: Log file header includes csvr_conserved when CSVR is the integrator
    Given a config with [integrator].kind = "csvr"
    And log_every > 0
    When the runner produces the log file
    Then its header line is "step,time,kinetic_energy,temperature,csvr_conserved"

  @rq-0dc48f6b
  Scenario: Log file header omits csvr_conserved when VV / Langevin / NHC is the integrator
    Given a config with [integrator].kind != "csvr"
    And log_every > 0
    When the runner produces the log file
    Then the header line does not include "csvr_conserved"

  # --- Rescale correctness ---

  @rq-efea1b70
  Scenario: Rescale skipped when K == 0
    Given a CSVR integrator and a ParticleBuffers whose velocities are all 0
    When integrator.step(...) is called and no force is applied
      (force_field has zero slots)
    Then no rescale_velocities launch occurs that step
    And state.cumulative_injection equals 0.0

  @rq-287e8d41
  Scenario: Rescale preserves COM momentum
    Given a CSVR integrator with N=16 particles and initial COM-x = 0
    When integrator.step(...) is called 20 times with dt=1e-15
    Then Σ_i m_i v_x[i] evaluated on the final velocities is zero
      to f32 round-off

  # --- Physical correctness ---

  @rq-f70f7c1e
  Scenario: Equilibrium kinetic energy approaches (N_f / 2) k_B T
    Given a CSVR run with N=512 LJ particles, temperature=300, tau=1e-13,
      dt=1e-15, n_steps=5000, seed=1
    When the run completes
    Then the time-averaged kinetic energy over the last 1000 log rows is
      within 5% of (N_f / 2) · k_B · 300
    And the time-averaged temperature is within 5% of 300 K

  @rq-4ae77a72
  Scenario: csvr_conserved drifts as a zero-mean martingale (no systematic bias)
    Given a CSVR run with N=128 LJ particles, dt=1e-15, n_steps=2000, seed=1
    When the run completes
    Then |mean(csvr_conserved increments)| < 3 · stddev(increments) / sqrt(n_rows)
      (i.e., consistent with zero mean to within 3σ)

  # --- Determinism ---

  @rq-dc51e1c3
  Scenario: Two independent CSVR runs with identical seeds produce byte-identical outputs
    Given two complete simulations with kind="csvr", identical parameters
      (including identical seed), identical initial state, n_steps=10
    When dynamics run is invoked on each
    Then the trajectory files are byte-identical
    And the log files are byte-identical, including the csvr_conserved column

  @rq-94a43204
  Scenario: Different seeds produce different trajectories
    Given two complete simulations identical except integrator.seed = 1 and = 2
    When dynamics run is invoked on each
    Then the trajectory files differ
```
