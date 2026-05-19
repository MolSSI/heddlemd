# Feature: Berendsen Weak-Coupling Thermostat <!-- rq-25f24b26 -->

The Berendsen weak-coupling thermostat (Berendsen et al., *J. Chem.
Phys.* **81**, 3684 (1984)) is a deterministic NVT-like temperature
coupling. One of the pluggable thermostat slots (see `framework.md`);
selected by `kind = "berendsen"` in the config's `[thermostat]`
section.

The thermostat runs once per timestep, after the integrator's
post-step state is published, as a post-only hook (`apply_post`). Its
`apply_pre` is the default no-op. Every velocity is multiplied by a
global scalar `λ` chosen to relax the instantaneous temperature
toward the user-specified target temperature over a coupling time
`τ`. The rescale preserves centre-of-mass momentum exactly. The
thermostat carries no RNG state and produces byte-identical
trajectories across runs on the same GPU.

> **Caveat: Berendsen does NOT sample the canonical ensemble.** The
> coupling proportionally rescales all velocities each step, but
> uniform rescaling does not redistribute energy between the centre-
> of-mass mode and the internal modes. Over long runs this produces
> the "flying ice cube" pathology — kinetic energy concentrates in
> COM modes, and the temperature distribution narrows below the
> equipartition value. Use Berendsen for **equilibration only**.
> For canonical-ensemble production runs, use one of the
> ensemble-correct alternatives in the registry: `csvr`,
> `nose-hoover-chain`, `andersen`, or the fused `langevin-baoab`
> integrator.

## Algorithm <!-- rq-adba6f8a -->

The thermostat is invoked through `apply_post(buffers, dt, timings)`
after the integrator's `step()` returns. The integrator has already
completed its velocity-Verlet substeps; the thermostat operates on
the post-step velocities. For each invocation with timestep `dt`:

1. Compute the instantaneous kinetic energy `K_old` via the shared
   `compute_kinetic_energy` helper (`nose-hoover-chain.md`).
2. Compute the rescale factor

   ```text
   K_target = (N_f / 2) · k_B · T
   λ² = 1 + (dt / τ) · (K_target / K_old − 1)
   λ  = sqrt(max(λ², 0))
   ```

   where `N_f = max(1, 3·N − n_constraints − 3)` is the number of
   thermostatted degrees of freedom (constraint- and COM-removed; the
   same convention used by CSVR — see `csvr.md` — and by
   `compute_temperature` in `io/log-output.md`). The `max(λ², 0)`
   floor handles the rare case
   where `dt / τ > 1` and `K_target ≪ K_old` would produce a
   negative `λ²`; clamping to zero quenches the velocities rather
   than producing imaginary numbers. Sensible parameters
   (`dt / τ ≤ 0.1`) never hit the floor.

3. Apply the rescale with the shared `rescale_velocities` helper
   (`nose-hoover-chain.md`):
   `v_i ← λ · v_i` for every particle `i` and axis.

4. Update the running
   `cumulative_injection += K_old · (λ² − 1)`. The host computes
   `K_new = λ² · K_old` directly from the kernel-supplied `K_old`
   and the host-computed `λ`; no second kinetic-energy reduction is
   needed.

When `K_old = 0` on entry (every velocity exactly zero, e.g. a
freshly-initialised system before the first force-driven kick), the
formula for `λ²` divides by zero. The thermostat detects this and
skips both the rescale launch and the `cumulative_injection` update
for that invocation. The next integrator step's velocity-Verlet kick
produces non-zero `K` and the thermostat resumes.

`apply_pre` is the trait default (no-op): Berendsen is a post-only
weak-coupling formula and never modifies velocities before the
integrator runs.

## Per-Step Kernel Sequence <!-- rq-dd953328 -->

Per timestep the Berendsen thermostat's `apply_post` runs the
following in fixed order:

| Order | Step      | Kernel / call           | Operation                                  | Stage label                  |
| ----- | --------- | ----------------------- | ------------------------------------------ | ---------------------------- |
| 1     | KE reduce | `kinetic_energy_reduce` | one f32 scalar of `K_old`                  | `KineticEnergyReduce`        |
| 2     | Rescale   | `rescale_velocities`    | host computes λ, scale velocities by λ     | `BerendsenRescaleVelocities` |

`kinetic_energy_reduce` and `rescale_velocities` are reused from
`nose-hoover-chain.md`. No new CUDA kernels.

The host-side work each step is one `f64` `λ` computation; cost is
negligible. The integrator's own kernels (`vv_kick_drift`, `vv_kick`,
the force pipeline) are launched separately by `integrator.step()`
and are not part of this slot's per-step sequence.

## Parameters <!-- rq-e3243e24 -->

The matching builder deserialises a typed `BerendsenParams` from the `[thermostat]` section's `SlotConfig::params` (see `framework.md`); the per-field reference below documents that parameter struct:

- `temperature: f64` — bath temperature `T` in kelvin. Required.
  Finite and strictly positive. Independent of
  `simulation.temperature`, which governs the initial-velocity
  sampler.
- `tau: f64` — thermostat coupling time in seconds. Required. Finite
  and strictly positive. Typical values for liquid water are 100 fs
  – 1 ps. Smaller `τ` couples more strongly (faster temperature
  relaxation, larger departure from NVE); larger `τ` leaves the
  dynamics closer to NVE.

No RNG seed: Berendsen is deterministic.

## Berendsen "conserved quantity" <!-- rq-0f81f83f -->

The diagnostic for Berendsen is

```text
H_berendsen = K + U − Σ_{invocations} (K_new − K_old)
```

i.e., the physical Hamiltonian plus a running subtraction of the
cumulative kinetic energy injected (or removed) by the rescale.
Because Berendsen is deterministic, this quantity drifts only by
`O(dt²)` per step (the velocity-Verlet integrator error), in
contrast to the martingale drift of the stochastic thermostats. A
systematic non-`O(dt²)` drift indicates an implementation bug.

`H_berendsen` is exposed as a per-log-row diagnostic column named
`berendsen_conserved` when Berendsen is the configured thermostat
(see `io/log-output.md`). The thermostat accumulates
`Σ (K_new − K_old)` on its host-side state across every
`apply_post` call, computed from `K_old` (kernel-reduced) and
`K_new = λ² · K_old` (host-computed).

## Empty State and degenerate cases <!-- rq-ed2ebeca -->

- `particle_count == 0`: `apply_pre` and `apply_post` return
  `Ok(())` without launching any kernel.
- `particle_count == 1` with `n_constraints == 0`:
  `N_f = max(1, 3 − 0 − 3) = 1`. The rescale works as documented; the
  system has one thermostatted degree of freedom.
- Heavily-constrained systems where `3·N − n_constraints − 3 <= 0`:
  the `max(1, …)` floor keeps `N_f = 1` so the thermostat still runs.
- `K_old == 0` on entry: rescale and accumulation are skipped this
  invocation.
- `dt / τ > 1` combined with `K_target ≪ K_old`: `λ²` is clamped to
  zero (quench).

## Feature API <!-- rq-ec1fc04e -->

### Types <!-- rq-ef1feb38 -->

- `BerendsenThermostat` — implements the `Thermostat` trait declared <!-- rq-f856f666 -->
  in `framework.md`. Registered in `ThermostatRegistry::with_builtins`
  under `kind_name() == "berendsen"`. Fields:

  - `device: Arc<CudaDevice>`
  - `temperature: f64`
  - `tau: f64`
  - `g_dof: u32` — `max(1, 3 · particle_count − n_constraints − 3)`,
    computed at construction from the `n_constraints` parameter
    passed by the runner.
  - `kt_target: f64` — `BOLTZMANN_J_PER_K · temperature`.
  - `cumulative_injection: f64` — running sum of `K_new − K_old`
    across every completed `apply_post` invocation. Initialised to
    `0.0`. Used by `log_column_values`.
  - `ke_scratch: CudaSlice<f32>` — length-1 device buffer for the
    kinetic-energy reduction; reused across calls.
  - `most_recent_ke: f64` — last kinetic energy `K_new` computed
    during the current `apply_post` invocation.

  All fields private; the slot's public surface is the `Thermostat`
  trait methods and construction via `BerendsenBuilder`. The
  `cumulative_injection` and `most_recent_ke` fields are public for
  parity with the other registered thermostats so a future
  restart-from-checkpoint flow can restore them explicitly.

- `BerendsenBuilder` — implements `ThermostatBuilder` with <!-- rq-6c9037a4 -->
  `kind_name() == "berendsen"`. `build(device, particle_count, kind)`
  deserialises `BerendsenParams` from `params`, allocates the
  length-1 `ke_scratch` device buffer, and returns the boxed
  `BerendsenThermostat`.

### `Thermostat` trait overrides <!-- rq-46d980b2 -->

`BerendsenThermostat` overrides the diagnostic-column trait methods
declared in `framework.md`:

- `log_column_names() -> &'static ["berendsen_conserved"]`. <!-- rq-c908bbf1 -->
- `log_column_values(ke, pe) -> vec![ke + pe − cumulative_injection]`. <!-- rq-3589910b -->

`apply_pre` is left at its trait default (no-op).

### CUDA Kernels <!-- rq-2aadbec9 -->

Berendsen introduces no new CUDA kernels. It reuses
`kinetic_energy_reduce` and `rescale_velocities` from
`nose-hoover-chain.md` through their public launch helpers
`compute_kinetic_energy` and `rescale_velocities` re-exported from
`crate::gpu`.

## Launch Configuration <!-- rq-e2f03a25 -->

Per-step launch counts (per `apply_post` invocation):

- `kinetic_energy_reduce`: 1 launch (single block of 256 threads).
- `rescale_velocities`: 1 launch (block 256, grid `ceil(n/256)`).

All launches go through the default stream of
`ParticleBuffers::device`.

## Determinism <!-- rq-5e802037 -->

- Both kernels involved are deterministic by construction (see
  `nose-hoover-chain.md`).
- Berendsen carries no RNG; there are no stochastic draws to
  randomise.
- The host-side `λ` computation runs in `f64` from a single `f64`
  input (`K_old`) and the deterministic parameters; two runs produce
  byte-identical `λ` and therefore byte-identical post-step
  velocities.
- Two end-to-end runs composing the same integrator with Berendsen
  on the same GPU with identical configs and identical initial
  particle state produce byte-identical trajectory and log files,
  including the `berendsen_conserved` column.

## Out of Scope <!-- rq-0076e6b4 -->

- A lossless `(f32, f64)` compensated mode. Berendsen is
  deterministic and time-reversibility under the integrator is not a
  property the algorithm promises.
- Per-type coupling times. The single `τ` applies to every particle
  regardless of species.
- Massive Berendsen (one independent coupling per atom). Single
  global thermostat only.
- A canonical-ensemble correction (e.g. Bussi-Donadio-Parrinello's
  stochastic version is exactly that and ships separately as
  `csvr`).
- A runtime warning when Berendsen is selected. The caveat lives in
  this requirements file and in the kind-field documentation in
  `config-schema.md`; the runner does not print a warning when
  loading a Berendsen config.
- Constraint algorithms (SHAKE/RATTLE).
- Pressure coupling (the Berendsen barostat is a separate
  `[barostat]` slot; see `framework.md`).

---

## Gherkin Scenarios <!-- rq-6f7ad4ad -->

```gherkin
Feature: Berendsen weak-coupling thermostat

  Background:
    Given a CUDA-capable GPU available as device 0
    And a SimulationBox with lx=ly=lz=1.0e6 unless otherwise specified
    And init_device() has been called

  # --- Construction ---

  @rq-e8252f10
  Scenario: Construct BerendsenThermostat via the registry (unconstrained system)
    Given a ThermostatKind::Berendsen { temperature: 300.0, tau: 1.0e-13 }
    When registry.build_optional(Some(&kind), device, particle_count=4, n_constraints=0) is called
    Then it returns Ok(Some(thermostat))
    And the underlying BerendsenThermostat has cumulative_injection == 0.0
    And state.g_dof == max(1, 3·4 − 0 − 3) == 9
    And state.kt_target == k_B * 300.0

  @rq-6136714b
  Scenario: Construct for a SETTLE'd water system
    Given a ThermostatKind::Berendsen { temperature: 300.0, tau: 1.0e-13 }
    When registry.build_optional(Some(&kind), device, particle_count=24, n_constraints=24) is called
    Then state.g_dof == max(1, 3*24 − 24 − 3) == 45

  @rq-e3a8d87c
  Scenario: Construct with particle_count = 0
    Given a ThermostatKind::Berendsen { temperature: 300.0, tau: 1.0e-13 }
    When registry.build_optional(Some(&kind), device, particle_count=0, n_constraints=0) is called
    Then it returns Ok(Some(thermostat))

  @rq-470019c7
  Scenario: Construct with particle_count = 1 (N_f = 1)
    Given a ThermostatKind::Berendsen { temperature: 300.0, tau: 1.0e-13 }
    When registry.build_optional(Some(&kind), device, particle_count=1, n_constraints=0) is called
    Then state.g_dof == 1

  # --- Config validation ---

  @rq-cef1e640
  Scenario: Reject non-positive temperature
    Given a config with [thermostat] kind="berendsen",
      temperature=0.0, tau=1.0e-13
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "thermostat.temperature", reason: _ })

  @rq-14fdd6ef
  Scenario: Reject non-positive tau
    Given a config with [thermostat] kind="berendsen",
      temperature=300.0, tau=-1.0e-13
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "thermostat.tau", reason: _ })

  @rq-61a19da3
  Scenario: Missing temperature is rejected
    Given a config with [thermostat] kind="berendsen", tau=1.0e-13
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "thermostat.temperature" })

  @rq-d189e9be
  Scenario: Missing tau is rejected
    Given a config with [thermostat] kind="berendsen", temperature=300.0
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "thermostat.tau" })

  @rq-0c7d1ff5
  Scenario: Reject extra fields (e.g. seed from CSVR/Andersen)
    Given a config with [thermostat] kind="berendsen",
      temperature=300.0, tau=1.0e-13, seed=42
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, message })
    And path equals "thermostat"
    And message mentions "seed"

  @rq-fff341e9
  Scenario: Reject extra fields (e.g. collision_rate from Andersen)
    Given a config with [thermostat] kind="berendsen",
      temperature=300.0, tau=1.0e-13, collision_rate=1.0e12
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, message })
    And path equals "thermostat"
    And message mentions "collision_rate"

  # --- Per-step kernel sequence ---

  @rq-0e85eebc
  Scenario: apply_post launches the expected kernel set
    Given a Berendsen thermostat with temperature=300, tau=1e-13, particle_count=4
    And buffers prepared with non-zero velocities
    When thermostat.apply_post(&mut buffers, dt=1e-15, &mut timings) is called
    Then KernelStage::KINETIC_ENERGY_REDUCE has count == 1
    And KernelStage::BERENDSEN_RESCALE_VELOCITIES has count == 1

  @rq-b296ab12
  Scenario: apply_post on empty state is a no-op
    Given a Berendsen thermostat with particle_count=0
    When thermostat.apply_post(...) is called
    Then it returns Ok(())

  @rq-92fc3091
  Scenario: apply_pre is the trait default (no-op)
    Given a Berendsen thermostat with particle_count=4
    And a snapshot of buffers.velocities before the call
    When thermostat.apply_pre(&mut buffers, dt=1e-15, &mut timings) is called
    Then it returns Ok(())
    And velocities are bit-identical to the snapshot
    And no kernel launches are recorded for that call

  # --- λ formula ---

  @rq-1c1022f0
  Scenario: λ = 1 when K_old already equals K_target
    Given an N=8 system with velocities placed so K_old exactly equals K_target
    When thermostat.apply_post(...) is called with dt=1.0e-15, τ=1.0e-13
    Then the rescale factor used is 1.0 within f32 round-off
    And the post-step velocities equal the pre-rescale velocities byte-for-byte

  @rq-c6adba60
  Scenario: λ > 1 when K_old < K_target (system needs heating)
    Given a system with K_old = 0.5 · K_target
    When thermostat.apply_post(...) is called with dt=0.1·τ
    Then the rescale factor used satisfies λ² = 1 + 0.1 · (2.0 − 1) = 1.1
    And post-step kinetic energy equals λ² · K_old = 0.55 · K_target

  @rq-b6c8867f
  Scenario: λ < 1 when K_old > K_target (system needs cooling)
    Given a system with K_old = 2.0 · K_target
    When thermostat.apply_post(...) is called with dt=0.1·τ
    Then the rescale factor used satisfies λ² = 1 + 0.1 · (0.5 − 1) = 0.95
    And post-step kinetic energy equals 0.95 · K_old = 1.9 · K_target

  @rq-4fe2658a
  Scenario: λ² is clamped to zero when (K_target/K_old − 1) · dt/τ < −1
    Given a system with K_old = 100 · K_target
    When thermostat.apply_post(...) is called with dt = 0.1 · τ
      (so λ² would be 1 + 0.1 · (0.01 − 1) = 0.901 — not clamped here)
    And the same scenario with dt = 2.0 · τ
      (so naive λ² would be 1 + 2.0 · (0.01 − 1) = -0.98 — clamped)
    Then the post-step velocities in the clamped case are all zero

  @rq-7d9f2da7
  Scenario: Rescale is skipped when K_old = 0
    Given a system with every velocity exactly zero (K_old = 0)
    When thermostat.apply_post(...) is called
    Then no rescale_velocities launch occurs that invocation
    And state.cumulative_injection equals 0.0

  # --- COM-momentum preservation ---

  @rq-5541d869
  Scenario: Berendsen preserves centre-of-mass momentum
    Given an N=16 system with initial COM-x = 0
    And a composed runner of velocity-Verlet + Berendsen
    When the runner executes 20 timesteps with dt=1e-15
    Then Σ_i m_i v_x[i] on the final velocities is zero to f32 round-off

  # --- cumulative_injection and log columns ---

  @rq-cfcce369
  Scenario: cumulative_injection equals K_old · (λ² − 1) per invocation
    Given a Berendsen thermostat and a ParticleBuffers with known non-zero velocities
    When thermostat.apply_post(...) is called once
    Then state.cumulative_injection equals K_old · (λ² − 1) to f64 round-off
      (where K_old and λ are those used in the launch)

  @rq-a0b746c6
  Scenario: log_column_names returns ["berendsen_conserved"]
    Given a constructed BerendsenThermostat
    Then state.log_column_names() equals ["berendsen_conserved"]

  @rq-2b114f41
  Scenario: log_column_values returns ke + pe − cumulative_injection
    Given a BerendsenThermostat with cumulative_injection = 1.0e-20
    When state.log_column_values(ke=2.5e-20, pe=3.0e-20) is called
    Then it returns [4.5e-20]

  @rq-05b4c988
  Scenario: Log file header includes berendsen_conserved when Berendsen is the thermostat
    Given a config with [thermostat].kind = "berendsen"
    And log_every > 0
    When the runner produces the log file
    Then its header line is "step,time,kinetic_energy,temperature,berendsen_conserved"

  # --- Temperature relaxation ---

  @rq-314d24cf
  Scenario: Temperature relaxes toward target on the time scale τ
    Given an N=128 system initialised at T = 2 · T_target
    And a composed runner of velocity-Verlet + Berendsen with τ = 1e-13, dt = 1e-15
    When the run completes 500 steps
    Then the temperature at step 500 is between T_target and 1.1 · T_target
      (most of the relaxation completes within several τ)

  @rq-3486c5e9
  Scenario: H_berendsen drifts only by O(dt²) per step
    Given an N=64 composed run with velocity-Verlet + Berendsen, dt=1e-15, n_steps=1000
    When the run completes
    Then |H_berendsen(n_steps) − H_berendsen(0)| / |H_berendsen(0)| is < 1.0e-3
    And the drift halves when dt is halved (consistent with O(dt²))

  # --- Determinism ---

  @rq-102e58cf
  Scenario: Two independent composed runs with identical configs are byte-identical
    Given two complete simulations composing velocity-Verlet + Berendsen,
      identical parameters, identical initial state, n_steps=10
    When dynamics run is invoked on each
    Then the trajectory files are byte-identical
    And the log files are byte-identical, including the berendsen_conserved column
```
