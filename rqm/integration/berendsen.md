# Feature: Berendsen Weak-Coupling Thermostat <!-- rq-25f24b26 -->

The Berendsen weak-coupling thermostat (Berendsen et al., *J. Chem.
Phys.* **81**, 3684 (1984)) is a deterministic NVT-like integrator.
One of the pluggable integrator slots (see `framework.md`); selected
by `kind = "berendsen"` in the config's `[integrator]` section.

Each timestep, after the velocity-Verlet propagation, every velocity
is multiplied by a global scalar `λ` chosen to relax the instantaneous
temperature toward the user-specified target temperature over a
coupling time `τ`. The rescale preserves centre-of-mass momentum
exactly. The thermostat carries no RNG state and produces
byte-identical trajectories across runs on the same GPU.

> **Caveat: Berendsen does NOT sample the canonical ensemble.** The
> coupling proportionally rescales all velocities each step, but
> uniform rescaling does not redistribute energy between the centre-
> of-mass mode and the internal modes. Over long runs this produces
> the "flying ice cube" pathology — kinetic energy concentrates in
> COM modes, and the temperature distribution narrows below the
> equipartition value. Use Berendsen for **equilibration only**.
> For canonical-ensemble production runs, use one of the
> ensemble-correct alternatives in the registry: `csvr`,
> `nose-hoover-chain`, `langevin-baoab`, or `andersen`.

## Algorithm <!-- rq-adba6f8a -->

For each timestep of size `dt`:

1. Run the velocity-Verlet sub-step (kick / drift / force / kick), see
   `velocity-verlet.md`.
2. Compute the instantaneous kinetic energy `K_old` via the shared
   `compute_kinetic_energy` helper (`nose-hoover-chain.md`).
3. Compute the rescale factor

   ```text
   K_target = (N_f / 2) · k_B · T
   λ² = 1 + (dt / τ) · (K_target / K_old − 1)
   λ  = sqrt(max(λ², 0))
   ```

   where `N_f = max(1, 3·N − 3)` is the number of thermostatted
   degrees of freedom. The `max(λ², 0)` floor handles the rare case
   where `dt / τ > 1` and `K_target ≪ K_old` would produce a
   negative `λ²`; clamping to zero quenches the velocities rather
   than producing imaginary numbers. Sensible parameters
   (`dt / τ ≤ 0.1`) never hit the floor.

4. Apply the rescale with the shared `rescale_velocities` helper
   (`nose-hoover-chain.md`):
   `v_i ← λ · v_i` for every particle `i` and axis.

5. Update the integrator's running
   `cumulative_injection += K_old · (λ² − 1)`. The host computes
   `K_new = λ² · K_old` directly from the kernel-supplied K_old and
   the host-computed λ; no second kinetic-energy reduction is
   needed.

When `K_old = 0` on entry (every velocity exactly zero, e.g. a
freshly-initialised system before the first force-driven kick), the
formula for `λ²` divides by zero. The integrator detects this and
skips both the rescale launch and the `cumulative_injection` update
for that step. The next step's velocity-Verlet kick produces non-zero
K and the thermostat resumes.

## Per-Step Kernel Sequence <!-- rq-dd953328 -->

Per timestep the Berendsen integrator's `step()` runs the following
in fixed order:

| Order | Step          | Kernel / call          | Operation                                            | Stage label               |
| ----- | ------------- | ---------------------- | ---------------------------------------------------- | ------------------------- |
| 1     | VV kick-drift | `vv_kick_drift`        | `v += (F/m)(dt/2)`, then `x += v dt`                 | `VvKickDrift`             |
| 2     | Force eval    | `force_field.step`     | recompute `F(x)`                                     | (slot-specific)           |
| 3     | VV kick       | `vv_kick`              | `v += (F/m)(dt/2)`                                   | `VvKick`                  |
| 4     | KE reduce     | `kinetic_energy_reduce` | one f32 scalar of `K_old`                            | `KineticEnergyReduce`     |
| 5     | Rescale       | `rescale_velocities`   | host computes λ, scale velocities by λ               | `BerendsenRescaleVelocities` |

`vv_kick_drift`, `vv_kick`, `kinetic_energy_reduce`, and
`rescale_velocities` are all reused from the existing infrastructure.
No new CUDA kernels.

The host-side work each step is one `f64` `λ` computation; cost is
negligible.

## Parameters <!-- rq-e3243e24 -->

Config layer fields set on `IntegratorKind::Berendsen`:

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
H_berendsen = K + U − Σ_{steps} (K_new − K_old)
```

i.e., the physical Hamiltonian plus a running subtraction of the
cumulative kinetic energy injected (or removed) by the rescale.
Because Berendsen is deterministic, this quantity drifts only by
`O(dt²)` per step (the velocity-Verlet integrator error), in
contrast to the martingale drift of the stochastic thermostats. A
systematic non-`O(dt²)` drift indicates an implementation bug.

`H_berendsen` is exposed as a per-log-row diagnostic column named
`berendsen_conserved` when Berendsen is the configured integrator
(see `io/log-output.md`). The integrator accumulates
`Σ (K_new − K_old)` on its host-side state across every `step()`
call, computed from `K_old` (kernel-reduced) and
`K_new = λ² · K_old` (host-computed).

## Empty State and degenerate cases <!-- rq-ed2ebeca -->

- `particle_count == 0`: `step()` returns `Ok(())` without launching
  any kernel.
- `particle_count == 1`: `N_f = max(1, 3·1 − 3) = 1`. The rescale
  works as documented; the system has one thermostatted degree of
  freedom.
- `K_old == 0` on entry: rescale and accumulation are skipped this
  step.
- `dt / τ > 1` combined with `K_target ≪ K_old`: `λ²` is clamped to
  zero (quench).

## Feature API <!-- rq-ec1fc04e -->

### Types <!-- rq-ef1feb38 -->

- `BerendsenState` — implements the `Integrator` trait declared in <!-- rq-f856f666 -->
  `framework.md`. Registered in `IntegratorRegistry::with_builtins`
  under `kind_name() == "berendsen"`. Fields:

  - `device: Arc<CudaDevice>`
  - `temperature: f64`
  - `tau: f64`
  - `g_dof: u32` — `max(1, 3·particle_count − 3)`.
  - `kt_target: f64` — `BOLTZMANN_J_PER_K · temperature`.
  - `cumulative_injection: f64` — running sum of `K_new − K_old`
    across every completed `step()` call. Initialised to `0.0`.
    Used by `log_column_values`.
  - `ke_scratch: CudaSlice<f32>` — length-1 device buffer for the
    kinetic-energy reduction; reused across calls.
  - `most_recent_ke: f64` — last kinetic energy `K_new` computed
    during the current `step()`.

  All fields private; the slot's public surface is the `Integrator`
  trait methods and construction via `BerendsenBuilder`. The
  `cumulative_injection` and `most_recent_ke` fields are public for
  parity with the other registered integrators so a future
  restart-from-checkpoint flow can restore them explicitly.

- `BerendsenBuilder` — implements `IntegratorBuilder` with <!-- rq-6c9037a4 -->
  `kind_name() == "berendsen"`. `build(device, particle_count, kind)`
  matches against `IntegratorKind::Berendsen { … }`, allocates the
  length-1 `ke_scratch` device buffer, and returns the boxed
  `BerendsenState`.

### `Integrator` trait overrides <!-- rq-46d980b2 -->

`BerendsenState` overrides the diagnostic-column trait methods
declared in `framework.md`:

- `log_column_names() -> &'static ["berendsen_conserved"]`. <!-- rq-c908bbf1 -->
- `log_column_values(ke, pe) -> vec![ke + pe − cumulative_injection]`. <!-- rq-3589910b -->

### CUDA Kernels <!-- rq-2aadbec9 -->

Berendsen introduces no new CUDA kernels. It reuses
`kinetic_energy_reduce` and `rescale_velocities` from
`nose-hoover-chain.md` through their public launch helpers
`compute_kinetic_energy` and `rescale_velocities` re-exported from
`crate::gpu`.

## Launch Configuration <!-- rq-e2f03a25 -->

Per-step launch counts (excluding the force pipeline):

- `vv_kick_drift`: 1 launch (block 256, grid `ceil(n/256)`).
- `vv_kick`: 1 launch.
- `kinetic_energy_reduce`: 1 launch (single block of 256 threads).
- `rescale_velocities`: 1 launch (block 256, grid `ceil(n/256)`).

All launches go through the default stream of
`ParticleBuffers::device`.

## Determinism <!-- rq-5e802037 -->

- All four kernels involved are deterministic by construction (see
  their respective documentation).
- Berendsen carries no RNG; there are no stochastic draws to
  randomise.
- The host-side `λ` computation runs in `f64` from a single `f64`
  input (`K_old`) and the deterministic parameters; two runs produce
  byte-identical `λ` and therefore byte-identical post-step
  velocities.
- Two end-to-end Berendsen runs on the same GPU with identical
  configs and identical initial particle state produce byte-identical
  trajectory and log files, including the `berendsen_conserved`
  column.

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
- Pressure coupling (the Berendsen barostat is a separate feature;
  not in v1).

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
  Scenario: Construct BerendsenState via the registry
    Given an IntegratorKind::Berendsen { temperature: 300.0, tau: 1.0e-13 }
    When registry.build(&kind, device, particle_count=4) is called
    Then it returns Ok(integrator)
    And the underlying BerendsenState has cumulative_injection == 0.0
    And state.g_dof == max(1, 3·4 − 3) == 9
    And state.kt_target == k_B * 300.0

  @rq-73384fea
  Scenario: Construct with particle_count = 0
    Given an IntegratorKind::Berendsen { temperature: 300.0, tau: 1.0e-13 }
    When registry.build(&kind, device, particle_count=0) is called
    Then it returns Ok(integrator)

  @rq-079b4f9e
  Scenario: Construct with particle_count = 1 (N_f = 1)
    Given an IntegratorKind::Berendsen { temperature: 300.0, tau: 1.0e-13 }
    When registry.build(&kind, device, particle_count=1) is called
    Then state.g_dof == 1

  # --- Config validation ---

  @rq-cef1e640
  Scenario: Reject non-positive temperature
    Given a config with [integrator] kind="berendsen",
      temperature=0.0, tau=1.0e-13
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "integrator.temperature", reason: _ })

  @rq-14fdd6ef
  Scenario: Reject non-positive tau
    Given a config with [integrator] kind="berendsen",
      temperature=300.0, tau=-1.0e-13
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "integrator.tau", reason: _ })

  @rq-61a19da3
  Scenario: Missing temperature is rejected
    Given a config with [integrator] kind="berendsen", tau=1.0e-13
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "integrator.temperature" })

  @rq-d189e9be
  Scenario: Missing tau is rejected
    Given a config with [integrator] kind="berendsen", temperature=300.0
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "integrator.tau" })

  @rq-0c7d1ff5
  Scenario: Reject extra fields (e.g. seed from CSVR/Andersen)
    Given a config with [integrator] kind="berendsen",
      temperature=300.0, tau=1.0e-13, seed=42
    When load_config is called
    Then it returns Err(ConfigError::UnknownIntegratorField {
      kind: "berendsen", field: "seed" })

  @rq-fff341e9
  Scenario: Reject extra fields (e.g. collision_rate from Andersen)
    Given a config with [integrator] kind="berendsen",
      temperature=300.0, tau=1.0e-13, collision_rate=1.0e12
    When load_config is called
    Then it returns Err(ConfigError::UnknownIntegratorField {
      kind: "berendsen", field: "collision_rate" })

  # --- Per-step kernel sequence ---

  @rq-0e85eebc
  Scenario: step() launches the expected kernel set
    Given a Berendsen integrator with temperature=300, tau=1e-13, particle_count=4
    And a ForceField with one LennardJones slot
    And a warm-up force evaluation has populated forces
    When integrator.step(...) is called with dt=1e-15
    And timings.finalize() is queried
    Then KernelStage::VV_KICK_DRIFT has count == 1
    And KernelStage::VV_KICK has count == 1
    And KernelStage::KINETIC_ENERGY_REDUCE has count == 1
    And KernelStage::BERENDSEN_RESCALE_VELOCITIES has count == 1

  @rq-b296ab12
  Scenario: step() on empty state is a no-op
    Given a Berendsen integrator with particle_count=0
    When integrator.step(...) is called
    Then it returns Ok(())

  # --- λ formula ---

  @rq-1c1022f0
  Scenario: λ = 1 when K_old already equals K_target
    Given an N=8 system with velocities placed so K_old exactly equals K_target
    And no forces (empty ForceField)
    When integrator.step(...) is called with dt=1.0e-15, τ=1.0e-13
    Then the rescale factor used is 1.0 within f32 round-off
    And the post-step velocities equal the pre-rescale velocities byte-for-byte

  @rq-c6adba60
  Scenario: λ > 1 when K_old < K_target (system needs heating)
    Given a system with K_old = 0.5 · K_target
    When integrator.step(...) is called with dt=0.1·τ
    Then the rescale factor used satisfies λ² = 1 + 0.1 · (2.0 − 1) = 1.1
    And post-step kinetic energy equals λ² · K_old = 0.55 · K_target

  @rq-b6c8867f
  Scenario: λ < 1 when K_old > K_target (system needs cooling)
    Given a system with K_old = 2.0 · K_target
    When integrator.step(...) is called with dt=0.1·τ
    Then the rescale factor used satisfies λ² = 1 + 0.1 · (0.5 − 1) = 0.95
    And post-step kinetic energy equals 0.95 · K_old = 1.9 · K_target

  @rq-4fe2658a
  Scenario: λ² is clamped to zero when (K_target/K_old − 1) · dt/τ < −1
    Given a system with K_old = 100 · K_target
    When integrator.step(...) is called with dt = 0.1 · τ
      (so λ² would be 1 + 0.1 · (0.01 − 1) = 0.901 — not clamped here)
    And the same scenario with dt = 2.0 · τ
      (so naive λ² would be 1 + 2.0 · (0.01 − 1) = -0.98 — clamped)
    Then the post-step velocities in the clamped case are all zero

  @rq-7d9f2da7
  Scenario: Rescale is skipped when K_old = 0
    Given a system with every velocity exactly zero (K_old = 0)
    And no forces (empty ForceField)
    When integrator.step(...) is called
    Then no rescale_velocities launch occurs that step
    And state.cumulative_injection equals 0.0

  # --- COM-momentum preservation ---

  @rq-5541d869
  Scenario: Berendsen preserves centre-of-mass momentum
    Given an N=16 Berendsen integrator with initial COM-x = 0
    When integrator.step(...) is called 20 times with dt=1e-15
    Then Σ_i m_i v_x[i] on the final velocities is zero to f32 round-off

  # --- cumulative_injection and log columns ---

  @rq-cfcce369
  Scenario: cumulative_injection equals K_old · (λ² − 1) per step
    Given a Berendsen integrator and a ParticleBuffers with known non-zero velocities
    And a ForceField with zero slots
    When integrator.step(...) is called once
    Then state.cumulative_injection equals K_old · (λ² − 1) to f64 round-off
      (where K_old and λ are those used in the launch)

  @rq-a0b746c6
  Scenario: log_column_names returns ["berendsen_conserved"]
    Given a constructed BerendsenState
    Then state.log_column_names() equals ["berendsen_conserved"]

  @rq-2b114f41
  Scenario: log_column_values returns ke + pe − cumulative_injection
    Given a BerendsenState with cumulative_injection = 1.0e-20
    When state.log_column_values(ke=2.5e-20, pe=3.0e-20) is called
    Then it returns [4.5e-20]

  @rq-05b4c988
  Scenario: Log file header includes berendsen_conserved when Berendsen is the integrator
    Given a config with [integrator].kind = "berendsen"
    And log_every > 0
    When the runner produces the log file
    Then its header line is "step,time,kinetic_energy,temperature,berendsen_conserved"

  # --- Temperature relaxation ---

  @rq-314d24cf
  Scenario: Temperature relaxes toward target on the time scale τ
    Given an N=128 system initialised at T = 2 · T_target
    And Berendsen with τ = 1e-13, dt = 1e-15
    When the run completes 500 steps
    Then the temperature at step 500 is between T_target and 1.1 · T_target
      (most of the relaxation completes within several τ)

  @rq-3486c5e9
  Scenario: H_berendsen drifts only by O(dt²) per step
    Given an N=64 Berendsen run with dt=1e-15, n_steps=1000
    When the run completes
    Then |H_berendsen(n_steps) − H_berendsen(0)| / |H_berendsen(0)| is < 1.0e-3
    And the drift halves when dt is halved (consistent with O(dt²))

  # --- Determinism ---

  @rq-102e58cf
  Scenario: Two independent Berendsen runs with identical configs are byte-identical
    Given two complete simulations with kind="berendsen", identical parameters,
      identical initial state, n_steps=10
    When dynamics run is invoked on each
    Then the trajectory files are byte-identical
    And the log files are byte-identical, including the berendsen_conserved column
```
