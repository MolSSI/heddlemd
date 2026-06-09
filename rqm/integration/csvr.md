# Feature: CSVR Stochastic Velocity Rescaling Thermostat <!-- rq-891232bf -->

Canonical Sampling through Velocity Rescaling (CSVR;
Bussi-Donadio-Parrinello, *J. Chem. Phys.* **126**, 014101 (2007)) is
a stochastic NVT thermostat. One of the pluggable thermostat slots
(see `framework.md`); selected by `kind = "csvr"` in the config's
`[thermostat]` section.

The thermostat applies a single global velocity rescale per timestep,
as a post-only hook (`apply_post`); its `apply_pre` is the trait
default no-op. The rescale factor is drawn from a Markov kernel that,
after equilibration, samples the canonical distribution at the
user-specified temperature exactly. CSVR has no ergodicity caveat for
stiff systems (unlike vanilla Nosé-Hoover) and perturbs the underlying
dynamics less than Langevin (no per-particle friction term).

## Algorithm <!-- rq-062ea284 -->

The thermostat is invoked through `apply_post(buffers, dt, timings)`
after the integrator's `step()` returns. The integrator has already
completed its velocity-Verlet substeps; the thermostat operates on
the post-step velocities. For each invocation with timestep `dt`, let
`c = exp(−dt/τ)`:

1. Compute the instantaneous kinetic energy
   `K = (1/2) Σ_i m_i |v_i|²` using the deterministic GPU reduction
   documented in `nose-hoover-chain.md`.
2. Draw `N_f` independent standard-normal samples on the host, where
   `N_f = max(1, 3·N − n_constraints − 3)` is the number of
   thermostatted degrees of freedom. The `n_constraints` term subtracts
   the holonomic constraints carried by the run (zero when no
   `[constraints]` section is configured; `3 · n_settle_groups` for a
   SETTLE'd water system). The `−3` term subtracts the three COM
   translational degrees of freedom that CSVR's uniform velocity
   rescale preserves. Designate the first sample as `R` and the
   remaining `N_f − 1` samples' squared sum as `S`:

   ```text
   R = ξ_0
   S = ξ_1² + ξ_2² + … + ξ_{N_f−1}²
   ```

3. Compute the new target kinetic energy (Bussi-Donadio-Parrinello
   2007 / PLUMED `csvr.cc`):

   ```text
   K_target = (N_f / 2) · T          # k_B = 1 in atomic units; T carries
                                     # k_B · T (Hartrees)
   K_new    = c · K
              + (K_target / N_f) · (1 − c) · (S + R²)
              + 2 · R · sqrt(c · (1 − c) · K · K_target / N_f)
   ```

   The `(S + R²)` term is `chi²(N_f)` since adding `R²` to
   `S ∼ chi²(N_f − 1)` makes a chi-squared with `N_f` degrees of
   freedom. At equilibrium (`K = K_target`), `E[S + R²] = N_f` gives
   `E[K_new] = K_target` exactly.

4. The velocity-rescale factor is `α = sqrt(K_new / K)` when `K > 0`;
   when `K = 0` (every velocity is exactly zero) the formula
   degenerates and the rescale is skipped. The CSVR step also produces
   a sign convention: the rescale factor is positive only when the
   discriminant `K_new` is non-negative. When `K_new` ends up
   negative (extremely unlikely for `N_f` of physical size; possible
   only for `N_f < 5` with adversarial draws) the kernel takes
   `K_new = K` (no rescale this invocation). This guard keeps the
   thermostat robust for tiny systems.

5. Apply the rescale with the shared `rescale_velocities` helper
   (documented in `nose-hoover-chain.md`):
   `v_i ← α · v_i` for every particle `i` and axis.

The CSVR Markov kernel is exact in the sense that repeated
application from any non-zero initial `K` converges to the canonical
distribution `P(K) ∝ K^{(N_f/2)−1} · exp(−K / T)` (with `k_B = 1` and
all quantities in Hartree atomic units; `T` is the engine-side
`k_B · T` value).

The instantaneous kinetic energy is the only piece of particle state
the thermostat reads; it has no auxiliary degrees of freedom.

`apply_pre` is the trait default (no-op): CSVR is a post-only
single-rescale formula and never modifies velocities before the
integrator runs.

## Per-Step Kernel Sequence <!-- rq-5f59fa80 -->

Per timestep the CSVR thermostat's `apply_post` runs the following in
fixed order:

| Order | Step               | Kernel / call                   | Operation                                          | Stage label             |
| ----- | ------------------ | ------------------------------- | -------------------------------------------------- | ----------------------- |
| 1     | KE reduce          | `kinetic_energy_reduce`         | one f32 scalar of `K`                              | `KineticEnergyReduce`   |
| 2     | Host RNG + rescale | `rescale_velocities` (1 launch) | sample `R`, `S`, compute `α`, scale velocities     | `CsvrRescaleVelocities` |

`kinetic_energy_reduce` and `rescale_velocities` are reused from
`nose-hoover-chain.md`; the timings file labels CSVR's velocity-rescale
call under `CsvrRescaleVelocities` to distinguish it from
`NhcRescaleVelocities` and `BerendsenRescaleVelocities`. No new CUDA
kernels.

The host-side work each invocation is `N_f` Philox-4×32-10
invocations + a chi-squared sum + the `α` calculation; for `N_f` of
order 10³ this is < 10 µs on a modern CPU. The integrator's own
kernels (`vv_kick_drift`, `vv_kick`, the force pipeline) are launched
separately by `integrator.step()` and are not part of this slot's
per-step sequence.

## Parameters <!-- rq-a8da85cf -->

The matching builder deserialises a typed `CsvrParams` from the `[thermostat]` section's `SlotConfig::params` (see `framework.md`); the per-field reference below documents that parameter struct:

- `temperature: f64` — bath temperature `T` as `k_B · T` in Hartrees
  (the engine's internal temperature representation; `k_B = 1`).
  Required. Finite and strictly positive. Independent of
  `simulation.temperature`, which governs the initial-velocity sampler.
- `tau: f64` — thermostat coupling time in atomic time units
  (`hbar / E_h` ≈ 24.2 attoseconds). Required. Finite and strictly
  positive. Larger `τ` leaves the dynamics closer to NVE (slow
  thermostat coupling); smaller `τ` enforces the target temperature
  more aggressively. Typical values for liquid water are 100 fs – 1 ps
  (i.e. ~4.1e3 – 4.1e4 atomic time units).
- `seed: u64` — counter-based RNG seed. Required, independent of
  `simulation.seed` and any other slot's seed. Two runs with
  identical configs on the same GPU produce byte-identical
  trajectories.

## RNG <!-- rq-54b99519 -->

CSVR draws `N_f` independent standard-normal samples per timestep.
The samples are generated by a counter-based Philox-4×32-10 RNG
(Salmon et al., SC11) on the **host**, using the same algorithm as
the device-side Philox in `langevin-baoab.md`. The host-side
implementation lives in `src/integrator/philox.rs::philox_4x32_10`,
exposed to other host-side stochastic slots as a stable utility.

### Counter packing <!-- rq-7c49aae5 -->

Each per-step draw of `N_f` normals consumes `N_f` Philox invocations
with:

- **Key (2 × u32)**: `(seed_lo, seed_hi)` — low and high halves of
  the thermostat's `seed`.
- **Counter (4 × u32)**:
  - `counter[0] = draw_counter_lo` — low 32 bits of the thermostat's
    `draw_counter`.
  - `counter[1] = draw_counter_hi` — high 32 bits.
  - `counter[2] = sample_index` — `0` for the lone `R` draw,
    `1..N_f−1` for the chi-squared components.
  - `counter[3] = 0` — reserved (matches Langevin's axis slot).

The `draw_counter` lives on `CsvrThermostat`. It starts at `0` at
construction. Each `apply_post` performs `self.draw_counter += 1`
before the draws and the in-call loop uses the post-increment value
across all `N_f` Philox calls of that invocation. The first
invocation in a run therefore uses `draw_counter == 1`; the `N`-th
invocation uses `draw_counter == N`.

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
H_csvr = K + U − Σ_{invocations} (K_new − K_old)
```

i.e., the physical Hamiltonian plus a running subtraction of the
cumulative kinetic energy injected (or removed) by the thermostat
over all completed invocations. With a correct CSVR implementation
the sequence `{H_csvr(t)}` drifts as a martingale (zero-mean
increment per step) and is bounded in expectation; an implementation
bug produces a systematic drift instead.

`H_csvr` is exposed as a per-log-row diagnostic column named
`csvr_conserved` when CSVR is the configured thermostat (see
`io/log-output.md`). The thermostat accumulates `Σ (K_new − K_old)`
on its host-side state across every `apply_post` call and combines
it with the kinetic and potential energies supplied by the runner at
log-write time.

## Empty State and degenerate cases <!-- rq-2a95c46c -->

- `particle_count == 0`: `apply_pre` and `apply_post` return
  `Ok(())` without launching any kernel.
- `particle_count == 1` with `n_constraints == 0`:
  `N_f = max(1, 3 − 0 − 3) = 1`. The chi-squared sum is empty
  (`S = 0`); the lone `R` term still appears. The thermostat
  propagates a degenerate one-degree-of-freedom system as documented
  in the original paper.
- Heavily-constrained systems where `3·N − n_constraints − 3 <= 0`:
  the `max(1, …)` floor keeps `N_f = 1` so the thermostat still
  runs. Users should not pair CSVR with such systems; the convention
  is documented for completeness.
- `K == 0` on entry: every velocity is exactly zero. The rescale
  formula contains `K` in a square-root term that would divide by
  zero; the kernel skips the rescale this invocation. The next
  integrator step's velocities (carrying whatever forces injected
  during the VV kicks) will be non-zero and the rescale resumes.

## Feature API <!-- rq-b1a8a6ca -->

### Types <!-- rq-1c26a108 -->

- `CsvrThermostat` — implements the `Thermostat` trait declared in <!-- rq-47d91c7d -->
  `framework.md`. Registered in `ThermostatRegistry::with_builtins`
  under `kind_name() == "csvr"`. Fields:

  - `temperature: f64`
  - `tau: f64`
  - `seed: u64`
  - `draw_counter: u64` — Philox counter advance for the CSVR draws.
    Initialised to `0` by the builder; pre-incremented on every
    `apply_post` call.
  - `g_dof: u32` — `max(1, 3 · particle_count − n_constraints − 3)`,
    computed at construction from the `n_constraints` parameter passed
    by the runner.
  - `kt_target: f64` — equals `temperature` (the engine stores
    `k_B · T` in Hartrees directly; `k_B = 1`, so no Boltzmann constant
    appears in this field).
  - `cumulative_injection: f64` — running sum of `K_new − K_old` over
    every completed `apply_post` call. Initialised to `0.0`. Used by
    `log_column_values`.
  - `ke_scratch: CudaSlice<f32>` — length-1 device buffer for the
    kinetic-energy reduction; reused across calls.
  - `most_recent_ke: f64` — last kinetic energy computed during the
    current `apply_post`. Available to `log_column_values` to avoid a
    redundant download.
  - `particle_count: usize`

  All fields private; the slot's public surface is the `Thermostat`
  trait methods (see `framework.md`) and the construction via
  `CsvrBuilder`. The `draw_counter` and `cumulative_injection`
  fields are public for parity with other slots' state so a future
  restart-from-checkpoint flow can restore them explicitly.

- `CsvrBuilder` — implements `ThermostatBuilder` with <!-- rq-750b828f -->
  `kind_name() == "csvr"`. `build(gpu, particle_count, n_constraints,
  params)` deserialises `CsvrParams` from `params`, computes
  `g_dof = max(1, 3 · particle_count − n_constraints − 3)`, allocates
  the length-1 `ke_scratch` device buffer, and returns the boxed
  `CsvrThermostat`.

### `Thermostat` trait overrides <!-- rq-5aae9633 -->

`CsvrThermostat` overrides the diagnostic-column trait methods
declared in `framework.md`:

- `log_column_names() -> &'static ["csvr_conserved"]`. <!-- rq-8ee58ec1 -->
- `log_column_values(ke, pe) -> vec![ke + pe − self.cumulative_injection]`. <!-- rq-2a5de2ab -->

`apply_pre` is left at its trait default (no-op).

### Host-side RNG utility <!-- rq-3d7c8e53 -->

`src/integrator/philox.rs::philox_4x32_10(key_lo: u32, key_hi: u32,
ctr0: u32, ctr1: u32, ctr2: u32, ctr3: u32) -> [u32; 4]` — host-side
Philox-4×32-10 implementation matching the device-side helper used
by `lan_ou_step` (see `langevin-baoab.md`). Pure function,
deterministic output for any fixed key/counter pair. Available to
any other host-side stochastic slot (e.g. Andersen thermostat).

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

Per-step launch counts (per `apply_post` invocation):

- `kinetic_energy_reduce`: 1 launch (single block of 256 threads).
- `rescale_velocities`: 1 launch (block 256, grid `ceil(n/256)`).

All launches go through the default stream of
`ParticleBuffers::device`.

## Determinism <!-- rq-72a606e3 -->

- `rescale_velocities` is deterministic by construction (see
  `nose-hoover-chain.md`).
- `kinetic_energy_reduce` uses a single-block deterministic reduction
  tree (see `nose-hoover-chain.md`).
- The host-side Philox stream is pure functional; the chi-squared sum
  proceeds in fixed left-to-right sample-index order in `f64`.
- The `draw_counter` advances by exactly `+1` per `apply_post`, so
  two runs that take the same number of steps with the same seed
  consume the same set of Philox counters.
- Two end-to-end runs composing the same integrator with CSVR on the
  same GPU with identical configs and identical initial particle
  state produce byte-identical trajectory and log files, including
  the `csvr_conserved` column.

## Out of Scope <!-- rq-1178e005 -->

- A lossless `(f32, f64)` compensated mode. CSVR is stochastic;
  bit-exact time-reversibility is not a property the algorithm
  promises.
- Pre/post symmetric splitting of the thermostat. The single
  post-VV variant is the canonical Bussi 2007 algorithm and is
  sufficient for correct canonical sampling.
- Massive thermostatting (per-atom independent CSVR). Single global
  thermostat only.
- User-overrideable `N_f` (degrees of freedom). The thermostat
  hard-codes `N_f = max(1, 3N − n_constraints − 3)`, the
  COM-removed and constraint-aware convention shared with
  `compute_temperature` (see `io/log-output.md`).
- Pressure coupling (Bussi-Parrinello stochastic-cell NPT extension).
  A future stochastic barostat would slot in alongside CSVR via the
  `[barostat]` slot.
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

  @rq-a6cd03aa
  Scenario: Construct CsvrThermostat via the registry (unconstrained system)
    Given a ThermostatKind::Csvr { temperature: 9.5e-4, tau: 41.34, seed: 42 }
      (300 K and 1 ps expressed in atomic units)
    When registry.build_optional(Some(&kind), device, particle_count=4, n_constraints=0) is called
    Then it returns Ok(Some(thermostat))
    And the underlying CsvrThermostat has draw_counter == 0
    And state.cumulative_injection == 0.0
    And state.g_dof == max(1, 3*4 − 0 − 3) == 9
    And state.kt_target == 9.5e-4 (the engine stores k_B · T directly,
      so kt_target equals the user-supplied temperature value)

  @rq-70a46202
  Scenario: Construct CsvrThermostat for a SETTLE'd water system
    Given a ThermostatKind::Csvr { temperature: 300.0, tau: 1.0e-13, seed: 42 }
    When registry.build_optional(Some(&kind), device, particle_count=24, n_constraints=24) is called
      (8 SETTLE waters: 24 atoms, 3 constraints per molecule)
    Then it returns Ok(Some(thermostat))
    And state.g_dof == max(1, 3*24 − 24 − 3) == 45

  @rq-b5089af4
  Scenario: Construct with particle_count = 0
    Given a ThermostatKind::Csvr { temperature: 300.0, tau: 1.0e-13, seed: 1 }
    When registry.build_optional(Some(&kind), device, particle_count=0, n_constraints=0) is called
    Then it returns Ok(Some(thermostat))

  @rq-7326a2d5
  Scenario: Construct with particle_count = 1 (N_f = 1)
    Given a ThermostatKind::Csvr { temperature: 300.0, tau: 1.0e-13, seed: 1 }
    When registry.build_optional(Some(&kind), device, particle_count=1, n_constraints=0) is called
    Then it returns Ok(Some(thermostat))
    And state.g_dof == 1

  @rq-d16be675
  Scenario: Heavily-constrained system clamps g_dof to 1
    Given a ThermostatKind::Csvr { temperature: 300.0, tau: 1.0e-13, seed: 1 }
    When registry.build_optional(Some(&kind), device, particle_count=2, n_constraints=4) is called
    Then it returns Ok(Some(thermostat))
    And state.g_dof == max(1, 3*2 − 4 − 3) == max(1, -1) == 1

  # --- Config validation (paired with config-schema scenarios) ---

  @rq-eba43990
  Scenario: Reject non-positive temperature
    Given a config with [thermostat] kind="csvr", temperature=0.0, tau=1.0e-13, seed=1
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "thermostat.temperature", reason: _ })

  @rq-e1a6bde9
  Scenario: Reject non-positive tau
    Given a config with [thermostat] kind="csvr", temperature=300.0, tau=-1.0e-13, seed=1
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "thermostat.tau", reason: _ })

  @rq-84927a79
  Scenario: Missing seed rejected
    Given a config with [thermostat] kind="csvr", temperature=300.0, tau=1.0e-13
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "thermostat.seed" })

  @rq-89d45c45
  Scenario: Reject extra fields (e.g., chain_length from NHC)
    Given a config with [thermostat] kind="csvr", temperature=300.0, tau=1.0e-13, seed=1,
      chain_length=3
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, message })
    And path equals "thermostat"
    And message mentions "chain_length"

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
  Scenario: apply_post launches the expected kernel set
    Given a CSVR thermostat with seed=1, temperature=300, tau=1e-13, particle_count=4
    And buffers prepared with non-zero velocities
    When thermostat.apply_post(&mut buffers, dt=1e-15, &mut timings) is called
    Then KernelStage::KINETIC_ENERGY_REDUCE has count == 1
    And KernelStage::CSVR_RESCALE_VELOCITIES has count == 1

  @rq-a2454a72
  Scenario: apply_post on empty CSVR state is a no-op
    Given a CSVR thermostat with particle_count=0
    When thermostat.apply_post(...) is called
    Then it returns Ok(())

  @rq-d1f1b53e
  Scenario: apply_pre is the trait default (no-op)
    Given a CSVR thermostat with particle_count=4
    And a snapshot of buffers.velocities before the call
    When thermostat.apply_pre(&mut buffers, dt=1e-15, &mut timings) is called
    Then it returns Ok(())
    And velocities are bit-identical to the snapshot
    And no kernel launches are recorded for that call

  # --- draw_counter advances per invocation ---

  @rq-1e5dcdc9
  Scenario: draw_counter starts at 0 and increments by 1 per apply_post
    Given a freshly built CsvrThermostat
    Then state.draw_counter == 0
    When thermostat.apply_post(...) is called once
    Then state.draw_counter == 1
    When thermostat.apply_post(...) is called again
    Then state.draw_counter == 2

  @rq-dc95802b
  Scenario: Two CsvrThermostats at the same draw_counter and seed produce identical
    velocity outputs after apply_post
    Given two CsvrThermostats A and B built from identical ThermostatKinds
    And both A.draw_counter and B.draw_counter == 5
    And identical buffer state
    When thermostat.apply_post(...) is called once on each
    Then post-call velocities of A and B are byte-identical

  # --- Log columns ---

  @rq-2c1bb918
  Scenario: log_column_names returns ["csvr_conserved"]
    Given a constructed CsvrThermostat
    Then state.log_column_names() equals ["csvr_conserved"]

  @rq-ca0b98cb
  Scenario: log_column_values returns ke + pe − cumulative_injection
    Given a CsvrThermostat with cumulative_injection = 1.0e-20
    When state.log_column_values(ke=2.5e-20, pe=3.0e-20) is called
    Then it returns [2.5e-20 + 3.0e-20 − 1.0e-20] = [4.5e-20]

  @rq-11b0deff
  Scenario: cumulative_injection accumulates K_new − K_old across calls
    Given a CsvrThermostat with cumulative_injection = 0.0
    And K_old = 1.0e-20 from a synthetic buffer setup before apply_post
    When thermostat.apply_post(...) is called once
    Then state.cumulative_injection equals the (K_new − K_old) of that invocation
      to f64 round-off

  @rq-1d25a0db
  Scenario: Log file header includes csvr_conserved when CSVR is the thermostat
    Given a config with [thermostat].kind = "csvr"
    And log_every > 0
    When the runner produces the log file
    Then its header line is "step,time,kinetic_energy,temperature,csvr_conserved"

  @rq-0dc48f6b
  Scenario: Log file header omits csvr_conserved when CSVR is not the thermostat
    Given a config with [thermostat].kind != "csvr" or [thermostat] omitted entirely
    And log_every > 0
    When the runner produces the log file
    Then the header line does not include "csvr_conserved"

  # --- Rescale correctness ---

  @rq-efea1b70
  Scenario: Rescale skipped when K == 0
    Given a CSVR thermostat composed with velocity-Verlet
    And a ParticleBuffers whose velocities are all 0
    When the runner executes one timestep and no force is applied
      (force_field has zero slots)
    Then no rescale_velocities launch occurs that step
    And state.cumulative_injection equals 0.0

  @rq-287e8d41
  Scenario: Rescale preserves COM momentum
    Given a CSVR thermostat composed with velocity-Verlet on N=16 particles
      with initial COM-x = 0
    When the runner executes 20 timesteps with dt=1e-15
    Then Σ_i m_i v_x[i] evaluated on the final velocities is zero
      to f32 round-off

  # --- Physical correctness ---

  @rq-f70f7c1e
  Scenario: Equilibrium kinetic energy approaches (N_f / 2) · T
    Given a composed runner of velocity-Verlet + CSVR with N=512 LJ particles,
      configured via TOML `units = "si"` with temperature=300 K,
      tau=1e-13 s, dt=1e-15 s, n_steps=5000, seed=1 (the loader converts
      these to the engine's atomic-unit values T_atomic, tau_atomic,
      dt_atomic before integration)
    When the run completes
    Then the time-averaged kinetic energy over the last 1000 log rows is
      within 5% of (N_f / 2) · T_atomic (with `k_B = 1` so no Boltzmann
      factor appears)
    And the time-averaged temperature column read from the SI-mode CSV log
      is within 5% of 300 K

  @rq-4ae77a72
  Scenario: csvr_conserved drifts as a zero-mean martingale (no systematic bias)
    Given a composed runner of velocity-Verlet + CSVR with N=128 LJ particles,
      dt=1e-15, n_steps=2000, seed=1
    When the run completes
    Then |mean(csvr_conserved increments)| < 3 · stddev(increments) / sqrt(n_rows)
      (i.e., consistent with zero mean to within 3σ)

  # --- Determinism ---

  @rq-dc51e1c3
  Scenario: Two independent composed runs with identical seeds produce byte-identical outputs
    Given two complete simulations composing velocity-Verlet + CSVR with identical parameters
      (including identical seed), identical initial state, n_steps=10
    When dynamics run is invoked on each
    Then the trajectory files are byte-identical
    And the log files are byte-identical, including the csvr_conserved column

  @rq-94a43204
  Scenario: Different seeds produce different trajectories
    Given two composed runs identical except thermostat.seed = 1 and = 2
    When dynamics run is invoked on each
    Then the trajectory files differ
```
