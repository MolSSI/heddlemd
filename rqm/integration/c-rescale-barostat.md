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

1. Launch `kinetic_energy_reduce` to write the instantaneous kinetic
   energy `K = (1/2) Σ_i m_i |v_i|²` into the slot-owned
   `ke_scratch: CudaSlice<f32>` (length 1).

2. Launch `virial_sum_reduce` to write the instantaneous total
   scalar virial `W = Σ_i buffers.virials[i]` into the slot-owned
   `virial_scratch: CudaSlice<f32>` (length 1). The per-particle
   `virials` buffer carries every contribution that enters the
   pressure estimator: the force-field pair, bonded, angle, and
   SPME (real + reciprocal) terms populated by `force_field.step`,
   plus any constraint contribution added by the `Constraint`
   slot's `apply_after_kick` hook (see `constraint-framework.md`;
   for SETTLE the contribution is documented in `settle.md`).

3. Pre-increment the host-side `draw_counter` by `+1`.

4. Launch the `c_rescale_compute_mu_and_rescale_lattice` kernel
   (described under *CUDA Kernels* below). The kernel reads
   `K` and `W` from device buffers, reads the current box lattice
   `[lx, ly, lz, xy, xz, yz]` from `sim_box.lattice_device_mut()`,
   computes `V_pre = lx · ly · lz` and the pressure

   ```text
   P = (2 K + W) / (3 V_pre)
   ```

   in `E_h / a_0^3` (the engine's atomic pressure unit), draws a
   single standard-normal Philox-4×32-10 sample `R` from the
   integrator's `seed` and the just-incremented `draw_counter`
   (see *RNG*), and computes the C-rescale scale factor `μ`:

   ```text
   μ³ = 1 − β · (dt / τ_P) · (P_target − P)
       + sqrt( 2 · β · T · dt / (τ_P · V_pre) ) · R
   μ  = max(μ_min, μ³)^(1/3)
   ```

   with parameters (`β`, `τ_P`, `P_target`, `T`) passed as scalar
   kernel arguments and `μ_min = 1.0e-6` the device-side safety
   floor (shared with the Berendsen barostat) that prevents
   pathological collapse when `μ³` is non-positive (rare under
   sensible parameters).

   With `k_B = 1` in atomic units, no Boltzmann factor appears in
   the noise amplitude:
   - `β` is the user-supplied isothermal compressibility in
     `a_0^3 / E_h`,
   - `τ_P` is the user-supplied pressure-coupling time constant in
     atomic time units (`hbar / E_h`),
   - `P_target` is the user-supplied target pressure in
     `E_h / a_0^3`,
   - `T` is the user-supplied target temperature carrying `k_B · T`
     in Hartrees,
   - `R` is the standard-normal sample.

   The first term inside `μ³` is the Berendsen-style deterministic
   drift toward `P_target`; the second is the Bernetti-Bussi noise
   that produces the correct stationary NPT volume fluctuations.
   Sign convention: when `P < P_target` (under-pressured), the
   deterministic drift contracts the box; when `P > P_target`
   (over-pressured), it expands the box. The noise term is symmetric.

   The kernel then mutates the device lattice in place
   (`lattice[i] ← μ · lattice[i]` for each of the six entries),
   writes the device-side rescale-factor scalar
   `mu_device: CudaSlice<f32>` (length 1) for use by
   `rescale_positions_device_factor` in step 5, and writes the
   three-element diagnostic buffer
   `diagnostics_device: CudaSlice<f64>` (length 3):

   ```text
   diagnostics_device[0] = P                          // most_recent_pressure
   diagnostics_device[1] = V_post = μ³ · V_pre        // most_recent_volume
   diagnostics_device[2] += P_target · (V_post − V_pre)  // cumulative_injection delta
   ```

   Element `[2]` is a running accumulator: each kernel invocation
   adds the per-step contribution rather than overwriting.

   The host's `sim_box` generation counter is incremented by 1
   inside the call (via `lattice_device_mut`); the host fields
   `sim_box.lx()`, `sim_box.volume()`, etc. become stale until the
   next `sim_box.flush_from_device()` call (issued by the runner
   before each log row; see *Diagnostic log columns* below).

5. Launch `rescale_positions_device_factor` to update particle
   positions by reading `mu_device[0]` and scaling each position
   component by it. The rescale is applied about the box origin;
   fractional coordinates relative to the new box are unchanged.
   No host involvement; no per-step dtoh of `μ`.

When `buffers.particle_count() == 0`, the entire hook is a no-op:
no kernel launches occur, no RNG draw is consumed, the box is not
mutated, and the diagnostic accumulators are unchanged.

The user is responsible for keeping the barostat's `temperature`
parameter consistent with any configured thermostat's target
temperature. The framework performs no cross-slot validation; the
barostat reads only its own `temperature` field.

## Per-Step Kernel Sequence <!-- rq-46a60367 -->

Per timestep, the C-rescale barostat's `apply` runs the following
three device-side steps; the fourth (the per-particle position
rescale) is dispatched by the JIT-composed post-force per-particle
kernel (see `jit-composed-post-force.md`).

| Order | Step              | Kernel / call                              | Operation                                                                                                          | Stage label                          |
| ----- | ----------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------ |
| 1     | KE reduce         | `kinetic_energy_reduce`                    | f32 scalar of `K` into `ke_scratch`                                                                                | `KineticEnergyReduce`                |
| 2     | Virial reduce     | `virial_sum_reduce`                        | f32 scalar of `W` into `virial_scratch`                                                                            | `VirialSumReduce`                    |
| 3     | µ + lattice + diag | `c_rescale_compute_mu_and_rescale_lattice` | reads `K`, `W`, lattice; draws `R`; computes µ + P in f64; mutates lattice; writes µ + diagnostics device buffers | `CRescaleComputeMuAndRescaleLattice` |
| 4     | Position rescale  | composed post-force per-particle kernel    | reads `mu_device[0]`, scales every particle position by it                                                          | `JitComposedPostForce`               |

The first three kernels run inside `apply`; the fourth runs from
the JIT-composed post-force per-particle kernel via c-rescale's
participation through its source fragment. The standalone
`rescale_positions_device_factor` kernel and the corresponding
`CRescaleBarostatRescalePositions` timings stage no longer exist;
the per-particle rescale body lives in c-rescale's source
fragment described below.

No per-step host download occurs. The host `sim_box`'s lattice
mirror, `most_recent_pressure`, `most_recent_volume`, and
`cumulative_barostat_injection` host fields are stale between log
rows. The runner refreshes them at log-write cadence by calling
`sim_box.flush_from_device()` and `barostat.flush_pending_injection(device)`
(see *Diagnostic log columns* and `framework.md`'s flush pattern).

## Post-Force Per-Particle Fragment <!-- rq-61f241f6 -->

`CRescaleBarostatState::post_force_per_particle_fragment()` returns
the per-particle position rescale as a `PerParticleFragment` (see
`jit-composed-post-force.md`):

- The per-thread body reads `Real mu = c_rescale_mu_device[0];`
  once and applies
  `positions_x/y/z[i] *= mu`. The position update happens AFTER
  any integrator or thermostat fragment in the canonical
  per-thread evaluation order, so the rescaled position reflects
  the post-step velocity contributions.

- `entry_point_args` declares
  `const Real *c_rescale_mu_device` (the device-resident µ scalar
  populated by `c_rescale_compute_mu_and_rescale_lattice`).

- `bind_post_force_per_particle_args` pushes `&self.mu_device`
  onto the launch builder.

C-rescale operates only on positions; the velocity rescale that
some c-rescale variants apply is not part of this barostat's
fragment.

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
  - `mu_device: CudaSlice<f32>` — length-1 device buffer holding
    the latest rescale factor `μ`, written by
    `c_rescale_compute_mu_and_rescale_lattice` and consumed by
    `rescale_positions_device_factor` on the same stream.
  - `diagnostics_device: CudaSlice<f64>` — length-3 device buffer
    laid out as `[most_recent_pressure, most_recent_volume,
    cumulative_injection_delta]`. Slots `[0]` and `[1]` are
    overwritten every step; slot `[2]` accumulates
    `P_target · (V_post − V_pre)` per step since the last
    `flush_pending_injection`.
  - `most_recent_pressure: f64` — `P` from the most recent flushed
    `apply` call. Host-side mirror of `diagnostics_device[0]`,
    refreshed only by `flush_pending_injection`. Stale between
    flushes.
  - `most_recent_volume: f64` — post-rescale `V` from the most
    recent flushed `apply` call. Host-side mirror of
    `diagnostics_device[1]`, refreshed only by
    `flush_pending_injection`. Stale between flushes.

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
- `flush_pending_injection(device)` — downloads `diagnostics_device` <!-- rq-fe7f35cd -->
  into the host fields `most_recent_pressure`, `most_recent_volume`,
  and accumulates `diagnostics_device[2]` into
  `cumulative_barostat_injection` before zeroing the device-side
  delta. Idempotent across consecutive calls. The runner calls this
  once before each log row to refresh the diagnostic columns.
- `log_column_names() -> &'static ["pressure", "box_volume", <!-- rq-b8e6dfb3 -->
  "c_rescale_conserved"]`.
- `log_column_values(ke, pe) -> vec![most_recent_pressure, <!-- rq-9a88810a -->
  most_recent_volume, ke + pe + pressure · most_recent_volume −
  cumulative_barostat_injection]`. The runner supplies the
  freshly-computed total kinetic and potential energies; the
  barostat combines them with its own cached state. The caller is
  expected to have called `flush_pending_injection` before reading
  `log_column_values` so the host fields reflect the latest
  device-side state.

### CUDA Kernels and launch helpers <!-- rq-d6c4e0f5 -->

`kernels/barostat.cu` declares
`c_rescale_compute_mu_and_rescale_lattice`:

```c
extern "C" __global__ void c_rescale_compute_mu_and_rescale_lattice(
    const float *k_scratch,              // length 1, K from kinetic_energy_reduce
    const float *w_scratch,              // length 1, W from virial_sum_reduce
    float *lattice,                      // length 6: in-place mutated
    float *mu_out,                       // length 1
    double *diagnostics,                 // length 3: [P, V_post, injection_delta]
    unsigned int seed_lo,
    unsigned int seed_hi,
    unsigned int draw_lo,
    unsigned int draw_hi,
    double pressure_target,              // P_target
    double tau,
    double compressibility,
    double kt,                           // k_B · T (T parameter; k_B = 1 in atomic units)
    double dt,
    double mu_cubed_min);                // (μ_min)^3 = 1e-18 host-side floor
```

Single-block, single-thread launch. The kernel reads `K` and `W`
from the device scratch buffers, reads the six lattice values from
the `lattice` argument, computes `V_pre = lattice[0] * lattice[1] *
lattice[2]`, computes the pressure `P = (2 K + W) / (3 V_pre)` in
`double` precision, draws one Philox-4×32-10 standard-normal sample
using `(seed_lo, seed_hi, draw_lo, draw_hi)`, evaluates `μ³` and
clamps to `mu_cubed_min`, computes `μ = cbrt(μ³)` and `V_post = μ³ ·
V_pre`, writes `mu_out[0] = (float) μ`, multiplies each of the six
`lattice[i]` slots by `μ` in place, and finally writes the diagnostic
triple
`diagnostics[0] = P`, `diagnostics[1] = V_post`,
`diagnostics[2] += P_target * (V_post − V_pre)`.

`kernels/barostat.cu` also declares
`rescale_positions_device_factor` — identical to
`rescale_positions` (Berendsen) except it reads the rescale factor
from a 1-element device buffer instead of taking it as a scalar
kernel argument:

```c
extern "C" __global__ void rescale_positions_device_factor(
    float *positions_x,
    float *positions_y,
    float *positions_z,
    const float *factor,                 // length 1
    unsigned int n);
```

The Rust launchers
`c_rescale_compute_mu_and_rescale_lattice` and
`rescale_positions_device_factor` are exposed under `crate::gpu`.

The slot reuses `kinetic_energy_reduce`
(`nose-hoover-chain.md`) and `virial_sum_reduce`
(`berendsen-barostat.md`) through their existing launch helpers
`compute_kinetic_energy_on_device` and
`compute_total_virial_on_device` (the variants that leave the
result on device — no host download). The lattice mutation is
mediated by `sim_box.lattice_device_mut()`.

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
    When heddlemd run is invoked on each
    Then the trajectory files are byte-identical
    And the log files are byte-identical, including pressure, box_volume, and
      c_rescale_conserved
    And the final SimulationBox lattices are byte-identical

  @rq-efc07f81
  Scenario: Different seeds produce different trajectories
    Given two complete simulations identical except barostat.seed = 1 and = 2
    When heddlemd run is invoked on each
    Then the trajectory files differ

```
