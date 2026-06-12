# Feature: Stochastic Cell-Rescaling (C-Rescale) Barostat <!-- rq-11f5dfd1 -->

The stochastic cell-rescaling barostat (Bernetti & Bussi, *J. Chem.
Phys.* **153**, 114107 (2020)) is an isotropic, stochastic
pressure-coupling slot that samples the canonical NPT distribution
exactly. One of the pluggable barostat slots (see `framework.md`);
selected by `kind = "c-rescale"` in the config's `[barostat]`
section.

The barostat runs once per timestep, after the integrator's step and
after the optional thermostat's `apply_post`. Each invocation
computes the instantaneous pressure from the kinetic energy and the
total scalar virial, derives an isotropic scale factor `μ` that
combines a deterministic Berendsen-style relaxation toward the
user-specified target pressure with a Brownian noise term sized so
that the long-time stationary distribution of the box volume is the
correct NPT marginal, then rescales every particle's position and
the simulation box by `μ`. The fractional coordinates of every
particle are invariant under the rescale, so no PBC wrap is required
and image counts carry over unchanged.

The barostat preserves centre-of-mass position exactly (uniform
scaling about the origin) and uses a counter-based Philox-4×32-10
RNG with a single normal draw per step, so it produces byte-identical
trajectories across runs on the same GPU with the same seed.

Unlike the Berendsen barostat, the C-rescale barostat is suitable
for canonical-ensemble production runs: the stochastic term restores
detailed balance and the correct NPT volume fluctuations.

## Algorithm <!-- rq-9f95c5da -->

The barostat is invoked through `apply(buffers, sim_box, dt,
timings)` after `integrator.step()` and after the thermostat's
`apply_post` (when a thermostat is configured) return. Both
`buffers.virials` (per-particle scalar virials populated by the
in-step force evaluation) and `buffers.velocities_*` (post-step
velocities, possibly rescaled by the thermostat) are read by this
hook.

For each invocation with timestep `dt`:

1. Compute the instantaneous kinetic energy `K = (1/2) Σ_i m_i |v_i|²`
   via the shared `compute_kinetic_energy` helper
   (`nose-hoover-chain.md`).

2. Compute the instantaneous total scalar virial
   `W = Σ_i buffers.virials[i]` via the shared
   `compute_total_virial` helper (`berendsen-barostat.md`). The
   per-particle virials buffer carries every contribution that
   enters the pressure estimator: the force-field pair, bonded,
   angle, and SPME (real + reciprocal) terms populated by
   `force_field.step`, plus any constraint contribution added by
   the `Constraint` slot's `apply_after_kick` hook (see
   `constraint-framework.md`; for SETTLE the contribution is
   documented in `settle.md`).

3. Read the current box volume `V = sim_box.volume()`.

4. Compute the instantaneous pressure

   ```text
   P = (2 K + W) / (3 V)
   ```

   in `E_h / a_0^3` (the engine's atomic pressure unit).

5. Advance the host-side `draw_counter` (pre-increment by `+1`) and
   draw a single standard-normal sample `R` via the counter-based
   Philox-4×32-10 RNG with the integrator's `seed` (see *RNG*
   below).

6. Compute the isotropic C-rescale scale factor `μ`:

   ```text
   μ³ = 1 − β · (dt / τ_P) · (P_target − P)
       + sqrt( 2 · β · T · dt / (τ_P · V) ) · R
   μ  = max(μ_min, μ³)^(1/3)
   ```

   where, with `k_B = 1` exactly in the engine's atomic units (no
   Boltzmann factor appears in the noise amplitude):
   - `β` is the user-supplied isothermal compressibility in
     `a_0^3 / E_h`,
   - `τ_P` is the user-supplied pressure-coupling time constant in
     atomic time units (`hbar / E_h`),
   - `P_target` is the user-supplied target pressure in
     `E_h / a_0^3`,
   - `T` is the user-supplied target temperature carrying `k_B · T`
     in Hartrees,
   - `R` is the standard-normal sample from step 5,
   - `μ_min = 1.0e-6` is the host-side safety floor (shared with
     the Berendsen barostat) that prevents pathological collapse
     when `μ³` is non-positive (rare under sensible parameters).

   The first term inside `μ³` is the Berendsen-style deterministic
   drift toward `P_target`; the second is the Bernetti-Bussi noise
   that produces the correct stationary NPT volume fluctuations.
   Sign convention: when `P < P_target` (under-pressured), the
   deterministic drift contracts the box; when `P > P_target`
   (over-pressured), it expands the box. The noise term is symmetric.

7. Launch the shared `rescale_positions` kernel (described in
   `berendsen-barostat.md`) to update particle positions:
   `x_i ← μ · x_i` for every particle. The rescale is applied about
   the box origin; fractional coordinates relative to the new box
   are unchanged.

8. Rescale the simulation box: call
   `sim_box.rescale_isotropic(μ)`, which multiplies all six lattice
   parameters `(lx, ly, lz, xy, xz, yz)` by `μ` and bumps the box
   generation counter. The bumped generation triggers the existing
   refresh paths in `forces/neighbor-list.md` and `forces/spme.md`
   on the next iteration's `force_field.step`.

9. Update the cumulative barostat-work diagnostic
   `cumulative_barostat_injection += P_target · (V_post − V_pre)`,
   where `V_post = μ³ · V_pre`. Consumed by `log_column_values` to
   produce the `c_rescale_conserved` log column.

When `buffers.particle_count() == 0`, the entire hook is a no-op:
no kernel launches occur, no RNG draw is consumed, the box is not
mutated, and `cumulative_barostat_injection` is unchanged.

The user is responsible for keeping the barostat's `temperature`
parameter consistent with any configured thermostat's target
temperature. The framework performs no cross-slot validation; the
barostat reads only its own `temperature` field.

## Per-Step Kernel Sequence <!-- rq-46a60367 -->

Per timestep, the C-rescale barostat's `apply` runs the following
in fixed order:

| Order | Step             | Kernel / call           | Operation                                          | Stage label                     |
| ----- | ---------------- | ----------------------- | -------------------------------------------------- | ------------------------------- |
| 1     | KE reduce        | `kinetic_energy_reduce` | one f32 scalar of `K`                              | `KineticEnergyReduce`           |
| 2     | Virial reduce    | `virial_sum_reduce`     | one f32 scalar of `W = Σ_i buffers.virials[i]`     | `VirialSumReduce`               |
| 3     | Position rescale | `rescale_positions`     | host draws `R`, computes μ, scales positions by μ  | `CRescaleBarostatRescalePositions` |

After step 3 the barostat calls `sim_box.rescale_isotropic(μ)` on
the host (no kernel launch) and updates the host-side
`cumulative_barostat_injection`. All three reused kernels
(`kinetic_energy_reduce`, `virial_sum_reduce`, `rescale_positions`)
are shared with the Berendsen thermostat / barostat slots; the
timings file labels the C-rescale position-rescale launch under
`CRescaleBarostatRescalePositions` to distinguish it from
`BerendsenBarostatRescalePositions`.

The integrator's own kernels (`vv_kick_drift`, `vv_kick`, the force
pipeline) and the thermostat's kernels are launched separately by
their respective hooks and are not part of this slot's per-step
sequence.

## Parameters <!-- rq-c2211c85 -->

The matching builder deserialises a typed `CRescaleBarostatParams` from the `[barostat]` section's `SlotConfig::params` (see `framework.md`); the per-field reference below documents that parameter struct:

- `pressure: f64` — target pressure `P_target` in `E_h / a_0^3` (the
  engine's atomic pressure unit).
  Required. Finite. May be any sign or zero (the formula handles
  negative and zero targets identically to positive ones).
- `temperature: f64` — target temperature `T` as `k_B · T` in Hartrees
  (the engine's internal temperature representation; `k_B = 1`) used
  in the
  noise term. Required. Finite and strictly positive. Independent
  of `simulation.temperature` and of any `[thermostat].temperature`;
  the framework performs no cross-slot validation. For canonical
  NPT sampling the user must keep this value consistent with the
  thermostat (or, for langevin-baoab, with the integrator's bath
  temperature).
- `tau: f64` — pressure-coupling time constant in atomic time units
  (`hbar / E_h`). Required.
  Finite and strictly positive. Typical values for liquid water are
  1–5 ps. Smaller `τ_P` couples the barostat more strongly to the
  physical system; larger `τ_P` smooths the response.
- `compressibility: f64` — isothermal compressibility `β` in
  `a_0^3 / E_h` (= the inverse atomic pressure unit). Required. Finite
  and strictly positive. Typical values: water ≈ `4.5e-10` 1/Pa
  ≈ `1.5e-2` in atomic units; LJ argon at liquid density similar
  order. An inaccurate value produces a different effective
  relaxation rate but does not break correctness of the long-time NPT
  distribution.
- `seed: u64` — counter-based RNG seed for the per-step normal draw.
  Required, independent of `simulation.seed` and any other slot's
  seed.

## RNG <!-- rq-411906c7 -->

The C-rescale barostat draws one standard-normal sample per step.
The sample is generated by the counter-based Philox-4×32-10 RNG
(Salmon et al., SC11) on the **host**, using the same algorithm
as every other host-side stochastic slot in the engine. The
host-side helper is `src/integrator/philox.rs::philox_normal` (see
`csvr.md`).

### Counter packing <!-- rq-add8ab9e -->

Each per-step draw uses one Philox invocation with:

- **Key (2 × u32)**: `(seed_lo, seed_hi)` — low and high halves of
  the barostat's `seed`.
- **Counter (4 × u32)**:
  - `counter[0] = draw_counter_lo` — low 32 bits of the barostat's
    `draw_counter`.
  - `counter[1] = draw_counter_hi` — high 32 bits.
  - `counter[2] = 0` — reserved.
  - `counter[3] = 0` — reserved.

The `draw_counter` lives on `CRescaleBarostat`. It starts at `0`
at construction and is pre-incremented on every `apply` call
before the draw; the first invocation in a run therefore uses
`draw_counter == 1`; the `N`-th invocation uses
`draw_counter == N`.

### Reproducibility <!-- rq-ef4d2caf -->

Two runs with identical `(seed, pressure, temperature, tau,
compressibility)` and identical initial particle state on the same
GPU produce byte-identical trajectory and log files. The Philox
stream is stateless; the per-step draw is a pure function of
`(seed, draw_counter)`; the host-side `μ³`, `μ`, and post-rescale
quantities run in `f64` from `f32` inputs.

## Diagnostic log columns <!-- rq-43e44a28 -->

The barostat exposes three per-log-row diagnostic columns when it
is configured (see `io/log-output.md`):

- `pressure` — instantaneous pressure `P` in `E_h / a_0^3` as computed in
  step 4 of the algorithm. This is the value used to derive the
  step's `μ`. When `buffers.particle_count() == 0`, `pressure` is
  reported as `0.0`.
- `box_volume` — simulation-box volume `V` in cubic metres. The
  value reported is the post-rescale volume (`μ³ · V_pre`),
  matching the lattice that the trajectory frame writes at the
  same step.
- `c_rescale_conserved` — `ke + pe + P_target · V_post −
  cumulative_barostat_injection`, where the cumulative term
  accumulates `P_target · (V_post − V_pre)` over every completed
  `apply` call. With a correct implementation `c_rescale_conserved`
  drifts only by the integrator's `O(dt²)` discretization error
  plus a stochastic martingale, in contrast to the systematic drift
  produced by an implementation bug. Mirrors the CSVR thermostat's
  `csvr_conserved` diagnostic.

## Empty State and degenerate cases <!-- rq-5b004038 -->

- `buffers.particle_count() == 0`: `apply` returns `Ok(())` without
  launching any kernel, without drawing from the RNG, and without
  mutating `sim_box`. The `pressure` and `box_volume` log columns
  are populated as `0.0` and `sim_box.volume()` respectively;
  `c_rescale_conserved` reports `ke + pe + P_target · V` (the
  cumulative term is zero).
- `V == 0` (degenerate box): unreachable; `pbc.rs` refuses to
  construct or rescale to a zero-volume box.
- `μ³ ≤ 0`: clamped to `μ_min³ = 1e-18` (so `μ = 1e-6`), exactly as
  in the Berendsen barostat.
- `T == 0` (degenerate noise amplitude): the noise term vanishes
  and the algorithm reduces to the Berendsen barostat formula.
  `temperature` is required strictly positive, so this case is
  rejected at config-load time.

## Feature API <!-- rq-79609285 -->

### Types <!-- rq-3b728a4a -->

- `CRescaleBarostat` — implements the `Barostat` trait declared in <!-- rq-e38993a3 -->
  `framework.md`. Registered in `BarostatRegistry::with_builtins`
  under `kind_name() == "c-rescale"`. Fields:

  - `device: Arc<CudaDevice>`
  - `pressure: f64` — `P_target`.
  - `temperature: f64` — `T` used in the noise amplitude.
  - `tau: f64` — `τ_P`.
  - `compressibility: f64` — `β`.
  - `seed: u64`
  - `draw_counter: u64` — Philox counter advance. Initialised to
    `0` by the builder; pre-incremented on every `apply` call.
  - `cumulative_barostat_injection: f64` — running sum of
    `P_target · (V_post − V_pre)` across every completed `apply`
    call. Initialised to `0.0`. Used by `log_column_values`.
  - `ke_scratch: CudaSlice<f32>` — length-1 device buffer for the
    kinetic-energy reduction; reused across calls.
  - `virial_scratch: CudaSlice<f32>` — length-1 device buffer for
    the virial reduction; reused across calls.
  - `most_recent_pressure: f64` — `P` from the most recent
    `apply` call. Used by `log_column_values`.
  - `most_recent_volume: f64` — post-rescale `V` from the most
    recent `apply` call. Used by `log_column_values`.

  All fields private; the slot's public surface is the `Barostat`
  trait methods and construction via `CRescaleBarostatBuilder`.
  `draw_counter`, `cumulative_barostat_injection`,
  `most_recent_pressure`, and `most_recent_volume` are public for
  parity with other slots' diagnostic state so a future
  restart-from-checkpoint flow can restore them explicitly.

- `CRescaleBarostatBuilder` — implements `BarostatBuilder` with <!-- rq-d521381a -->
  `kind_name() == "c-rescale"`. `build(device, particle_count,
  kind)` deserialises `CRescaleBarostatParams` from `params`, allocates
  the two length-1 device scratch buffers, and returns the boxed
  `CRescaleBarostat`.

### `Barostat` trait overrides <!-- rq-e17a3af9 -->

`CRescaleBarostat` overrides every method on the `Barostat` trait
declared in `framework.md`:

- `apply(buffers, sim_box, dt, timings)` — runs the per-step <!-- rq-2b405d23 -->
  algorithm above.
- `log_column_names() -> &'static ["pressure", "box_volume", <!-- rq-b8e6dfb3 -->
  "c_rescale_conserved"]`.
- `log_column_values(ke, pe) -> vec![most_recent_pressure, <!-- rq-9a88810a -->
  most_recent_volume, ke + pe + pressure · most_recent_volume −
  cumulative_barostat_injection]`. The runner supplies the
  freshly-computed total kinetic and potential energies; the
  barostat combines them with its own cached state.

### CUDA Kernels and launch helpers <!-- rq-d6c4e0f5 -->

No new CUDA kernels. The slot reuses
`kinetic_energy_reduce` (`nose-hoover-chain.md`),
`virial_sum_reduce` (`berendsen-barostat.md`), and
`rescale_positions` (`berendsen-barostat.md`) through their
existing launch helpers `compute_kinetic_energy`,
`compute_total_virial`, and `rescale_positions` re-exported from
`crate::gpu`. The `SimulationBox::rescale_isotropic` convenience is
likewise reused from `berendsen-barostat.md`.

## Launch Configuration <!-- rq-84b2667f -->

Per-step launch counts (per `apply` invocation):

- `kinetic_energy_reduce`: 1 launch (single block of 256 threads).
- `virial_sum_reduce`: 1 launch (single block of 256 threads).
- `rescale_positions`: 1 launch (block 256, grid `ceil(n/256)`).

All launches go through the default stream of
`ParticleBuffers::device`.

## Determinism <!-- rq-91c4727a -->

- All three reused kernels are deterministic by construction (see
  the helper files referenced under *Feature API*).
- The host-side Philox stream is pure functional; the per-step
  normal draw is a pure function of `(seed, draw_counter)`.
- The host-side `P`, `μ³`, `μ`, and `cumulative_barostat_injection`
  computations run in `f64` from `f32` inputs (`K`, `W`, `V`) and
  the deterministic parameters; two runs with the same seed
  produce byte-identical `μ` and therefore byte-identical
  post-rescale positions and box.
- `SimulationBox::rescale_isotropic(μ)` is a pure deterministic
  multiplication; the generation counter is monotonically
  incremented in lock-step across runs.
- The `draw_counter` advances by exactly `+1` per `apply` call.
- Two end-to-end runs composing the same integrator (and optionally
  the same thermostat) with the C-rescale barostat on the same GPU
  with identical configs and identical initial particle state
  produce byte-identical trajectory and log files, including the
  `pressure`, `box_volume`, and `c_rescale_conserved` columns.

## Out of Scope <!-- rq-5915b212 -->

- Semi-isotropic and anisotropic box deformation. The slot rescales
  all six lattice parameters by a single scalar `μ`; per-axis or
  semi-isotropic (xy-coupled, z-independent) coupling is a separate
  feature requiring per-axis virial computation throughout the
  force pipeline.
- Cross-slot validation against the thermostat's `temperature`. The
  user supplies the barostat's `temperature` explicitly and is
  responsible for keeping it consistent with whatever thermostat is
  configured.
- The Berendsen barostat. Selected separately via `kind =
  "berendsen"` (see `berendsen-barostat.md`); deterministic
  weak-coupling, equilibration only.
- Extended-system barostats (Parrinello-Rahman, Martyna-Tobias-Klein).
  These integrators carry an additional dynamical box-momentum
  degree of freedom and would require an integrator with an
  augmented `step()`, not a post-step `Barostat::apply` hook.
- Constraint algorithms (SHAKE/RATTLE) and their interaction with
  the position rescale. Constraints would need to be re-projected
  after the rescale; the framework does not yet ship a constraint
  slot.
- A `μ`-clamp configurable per-run. The host-side `μ_min = 1e-6`
  floor is a fixed safety guard shared with the Berendsen barostat;
  users who hit it should tighten their parameters.

---

## Gherkin Scenarios <!-- rq-8d18d6ee -->

```gherkin
Feature: Stochastic cell-rescaling (C-rescale) barostat

  Background:
    Given a CUDA-capable GPU available as device 0
    And a SimulationBox with lx=ly=lz=1.0e-9 unless otherwise specified
    And init_device() has been called

  # --- Construction ---

  @rq-26b98781
  Scenario: Construct CRescaleBarostat via the registry
    Given a BarostatKind::CRescale {
      pressure: 1.0e5, temperature: 85.0,
      tau: 1.0e-12, compressibility: 4.5e-10, seed: 42 }
    When registry.build_optional(Some(&kind), device, particle_count=4) is called
    Then it returns Ok(Some(barostat))
    And the underlying CRescaleBarostat has draw_counter == 0
    And the underlying CRescaleBarostat has cumulative_barostat_injection == 0.0
    And state.pressure == 1.0e5
    And state.temperature == 85.0
    And state.tau == 1.0e-12
    And state.compressibility == 4.5e-10
    And state.seed == 42

  @rq-f1b57184
  Scenario: Construct with particle_count = 0
    Given a BarostatKind::CRescale {
      pressure: 1.0e5, temperature: 85.0,
      tau: 1.0e-12, compressibility: 4.5e-10, seed: 1 }
    When registry.build_optional(Some(&kind), device, particle_count=0) is called
    Then it returns Ok(Some(barostat))

  @rq-1e27ee47
  Scenario: BarostatRegistry::with_builtins() exposes c-rescale alongside berendsen
    Given a BarostatRegistry::with_builtins()
    Then the registry contains a builder whose kind_name() is "berendsen"
    And the registry contains a builder whose kind_name() is "c-rescale"

  # --- Config validation ---

  @rq-8904d7cb
  Scenario: Accept negative target pressure
    Given a config with [barostat] kind="c-rescale",
      pressure=-1.0e5, temperature=85.0, tau=1.0e-12,
      compressibility=4.5e-10, seed=1
    When load_config is called
    Then it returns Ok(config)

  @rq-d406cdc2
  Scenario: Reject non-positive temperature
    Given a config with [barostat] kind="c-rescale",
      pressure=1.0e5, temperature=0.0, tau=1.0e-12,
      compressibility=4.5e-10, seed=1
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "barostat.temperature", reason: _ })

  @rq-a3b6838f
  Scenario: Reject non-positive tau
    Given a config with [barostat] kind="c-rescale",
      pressure=1.0e5, temperature=85.0, tau=-1.0e-12,
      compressibility=4.5e-10, seed=1
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "barostat.tau", reason: _ })

  @rq-de8b9cd5
  Scenario: Reject non-positive compressibility
    Given a config with [barostat] kind="c-rescale",
      pressure=1.0e5, temperature=85.0, tau=1.0e-12,
      compressibility=0.0, seed=1
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "barostat.compressibility", reason: _ })

  @rq-1a2f0ba9
  Scenario: Missing pressure rejected
    Given a config with [barostat] kind="c-rescale",
      temperature=85.0, tau=1.0e-12, compressibility=4.5e-10, seed=1
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "barostat.pressure" })

  @rq-0a18c7c2
  Scenario: Missing temperature rejected
    Given a config with [barostat] kind="c-rescale",
      pressure=1.0e5, tau=1.0e-12, compressibility=4.5e-10, seed=1
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "barostat.temperature" })

  @rq-b4d2ac96
  Scenario: Missing tau rejected
    Given a config with [barostat] kind="c-rescale",
      pressure=1.0e5, temperature=85.0, compressibility=4.5e-10, seed=1
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "barostat.tau" })

  @rq-77f42047
  Scenario: Missing compressibility rejected
    Given a config with [barostat] kind="c-rescale",
      pressure=1.0e5, temperature=85.0, tau=1.0e-12, seed=1
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "barostat.compressibility" })

  @rq-0b5f0881
  Scenario: Missing seed rejected
    Given a config with [barostat] kind="c-rescale",
      pressure=1.0e5, temperature=85.0, tau=1.0e-12, compressibility=4.5e-10
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "barostat.seed" })

  @rq-e9caf013
  Scenario: Reject extra fields (e.g. Berendsen barostat doesn't have a seed)
    Given a config with [barostat] kind="c-rescale",
      pressure=1.0e5, temperature=85.0, tau=1.0e-12, compressibility=4.5e-10,
      seed=42, friction=1.0e12
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, message })
    And path equals "barostat"
    And message mentions "friction"

  # --- Per-step kernel sequence ---

  @rq-3e57f675
  Scenario: apply launches the expected kernel set
    Given a C-rescale barostat with pressure=1.0e5, temperature=85.0,
      tau=1.0e-12, compressibility=4.5e-10, seed=1
    And a ParticleBuffers with N=4 non-zero velocities and virials
    When barostat.apply(&mut buffers, &mut sim_box, dt=1e-15, &mut timings) is called
    Then KernelStage::KINETIC_ENERGY_REDUCE has count == 1
    And KernelStage::VIRIAL_SUM_REDUCE has count == 1
    And KernelStage::C_RESCALE_BAROSTAT_RESCALE_POSITIONS has count == 1
    And KernelStage::BERENDSEN_BAROSTAT_RESCALE_POSITIONS has count == 0
    And KernelStage::VV_KICK_DRIFT has count == 0
    And KernelStage::VV_KICK has count == 0

  @rq-4894ae09
  Scenario: apply on empty state is a no-op
    Given a C-rescale barostat with particle_count=0
    When barostat.apply(...) is called
    Then it returns Ok(())
    And sim_box.generation() is unchanged
    And state.draw_counter is unchanged
    And state.cumulative_barostat_injection equals 0.0

  # --- draw_counter advances ---

  @rq-1f2b5320
  Scenario: draw_counter starts at 0 and increments by 1 per apply
    Given a freshly built CRescaleBarostat
    Then state.draw_counter == 0
    When barostat.apply(...) is called once
    Then state.draw_counter == 1
    When barostat.apply(...) is called again
    Then state.draw_counter == 2

  @rq-cedda168
  Scenario: Two CRescaleBarostats at the same (seed, draw_counter)
    produce identical post-rescale positions and box
    Given two CRescaleBarostats A and B built from identical BarostatKinds
    And both A.draw_counter and B.draw_counter == 5
    And identical buffer state
    When barostat.apply(...) is called once on each
    Then post-call positions of A and B are byte-identical
    And post-call sim_box lattices of A and B are byte-identical

  # --- μ correctness (deterministic and stochastic parts) ---

  @rq-9e5579bb
  Scenario: μ³ matches the analytical formula for a fixed Philox draw
    Given an N=8 system with known K and W such that P = P_target / 2
    And a barostat built with seed=1, draw_counter=0
    When barostat.apply(...) is called once with known dt, τ, β, T
    Then the post-rescale V / V_pre exactly equals
      1 − β · (dt/τ) · (P_target − P) + sqrt(2·β·k_B·T·dt/(τ·V_pre)) · R
      where R = philox_normal(seed, 1, 0, 0, 0, 0)
      within f32 round-off

  @rq-ac873434
  Scenario: Setting temperature very small reduces C-rescale to the Berendsen barostat
    Given an N=8 system with known K and W
    And a C-rescale barostat with temperature=1e-30 (noise term ≈ 0)
    And a Berendsen barostat with the same pressure, tau, and compressibility
    When each barostat applies once
    Then the post-rescale volumes agree to within f32 round-off

  # --- Fractional-coord and PBC invariants ---

  @rq-3b9e9550
  Scenario: Fractional coordinates of every particle are invariant under apply
    Given an N=8 system with arbitrary positions
    And a snapshot of fractional coordinates per particle
    When barostat.apply(...) is called
    Then the post-step fractional coordinates of every particle equal the
      snapshot within f32 round-off
    And no particle moved across an image boundary (image flags unchanged)

  @rq-94d30346
  Scenario: Triclinic shape is preserved under apply
    Given a triclinic SimulationBox with non-zero xy, xz, yz
    And a snapshot of (xy/lx, xz/lx, yz/ly) ratios
    When barostat.apply(...) is called
    Then the post-step (xy/lx, xz/lx, yz/ly) ratios equal the snapshot
      within f32 round-off

  # --- Box-generation propagation ---

  @rq-9d2d90b3
  Scenario: sim_box.generation() advances after apply
    Given a C-rescale barostat and a SimulationBox at generation g
    When barostat.apply(...) is called
    Then sim_box.generation() == g + 1

  # --- Log columns ---

  @rq-e5cb5505
  Scenario: log_column_names returns ["pressure", "box_volume", "c_rescale_conserved"]
    Given a constructed CRescaleBarostat
    Then state.log_column_names() equals ["pressure", "box_volume", "c_rescale_conserved"]

  @rq-fb78338b
  Scenario: log_column_values returns the cached pressure, post-rescale volume,
    and the c_rescale_conserved combination
    Given a CRescaleBarostat with
      most_recent_pressure = 1.01e5,
      most_recent_volume = 1.0e-27,
      cumulative_barostat_injection = 3.0e-22,
      pressure = 1.0e5
    When state.log_column_values(ke=1.5e-20, pe=2.0e-20) is called
    Then it returns [
      1.01e5,
      1.0e-27,
      1.5e-20 + 2.0e-20 + 1.0e5 · 1.0e-27 − 3.0e-22 ]

  @rq-df305128
  Scenario: Log file header includes pressure, box_volume, and c_rescale_conserved when C-rescale is the configured barostat
    Given a config with [barostat].kind = "c-rescale"
    And log_every > 0
    When the runner produces the log file
    Then its header line ends with "pressure,box_volume,c_rescale_conserved"

  # --- Composition with thermostat and integrator ---

  @rq-0f3f63c8
  Scenario: C-rescale composes with velocity-Verlet + CSVR for NPT
    Given a composed runner of velocity-Verlet + CSVR + C-rescale with
      N=128 LJ argon at T_target = 85 K (set on both CSVR and the
      barostat), P_target = 1.0 bar, n_steps = 1000
    When the run completes
    Then the run finishes without error
    And the time-averaged temperature over the last 500 log rows is within 5% of 85 K
    And the time-averaged pressure over the last 500 log rows is within 20% of 1.0 bar
    And the volume fluctuations (RMS / mean) over the last 500 log rows are
      within 50% of the NPT prediction sqrt(k_B · T · β / V_mean)

  @rq-2d109b3a
  Scenario: C-rescale composes with langevin-baoab
    Given a composed runner of langevin-baoab (no [thermostat]) + C-rescale with
      N=128 LJ argon, T_target on the integrator = T_target on the barostat = 85 K
    When load_config and run are invoked
    Then the run finishes without error

  # --- Determinism ---

  @rq-6e0d6cb4
  Scenario: Two independent composed runs with identical configs and seeds are byte-identical
    Given two complete simulations composing velocity-Verlet + C-rescale
      with identical parameters (including identical barostat.seed),
      identical initial state, n_steps = 10
    When dynamics run is invoked on each
    Then the trajectory files are byte-identical
    And the log files are byte-identical, including pressure, box_volume, and
      c_rescale_conserved
    And the final SimulationBox lattices are byte-identical

  @rq-efc07f81
  Scenario: Different seeds produce different trajectories
    Given two complete simulations identical except barostat.seed = 1 and = 2
    When dynamics run is invoked on each
    Then the trajectory files differ

```
