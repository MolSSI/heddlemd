# Feature: Berendsen Weak-Coupling Barostat <!-- rq-0d8c8688 -->

The Berendsen weak-coupling barostat (Berendsen et al., *J. Chem.
Phys.* **81**, 3684 (1984)) is an isotropic, deterministic
pressure-coupling slot. One of the pluggable barostat slots (see
`framework.md`); selected by `kind = "berendsen"` in the config's
`[barostat]` section.

The barostat runs once per timestep, after the integrator's step and
after the optional thermostat's `apply_post`. Each invocation
computes the instantaneous pressure from the kinetic energy and the
total scalar virial, derives an isotropic scale factor `μ` that
relaxes the pressure toward the user-specified target on a coupling
time `τ`, then rescales every particle's position and the simulation
box by `μ`. The fractional coordinates of every particle are
invariant under the rescale, so no PBC wrap is required and image
counts carry over unchanged.

The barostat preserves centre-of-mass position exactly (uniform
scaling about the origin) and carries no RNG state, so it produces
byte-identical trajectories across runs on the same GPU.

> **Caveat: the Berendsen barostat does NOT sample the isobaric
> ensemble.** Like the Berendsen thermostat, the uniform-scaling
> coupling is consistent in the mean (the time-averaged pressure
> approaches the target) but produces incorrect volume fluctuations
> and is not detailed-balance with respect to NPT or NPH. Use it for
> **equilibration only**. Canonical-ensemble production runs should
> use a stochastic-cell or extended-system barostat once those slots
> ship.

## Algorithm <!-- rq-6fb6f0b7 -->

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
   virials buffer carries every contribution that enters the
   pressure estimator: force-field pair / bonded / angle / SPME
   (real + reciprocal) terms populated by `force_field.step`,
   plus any constraint contribution added by the `Constraint`
   slot's `apply_after_kick` hook (see `constraint-framework.md`;
   for SETTLE the contribution is documented in `settle.md`).

3. Launch the `berendsen_compute_mu_and_rescale_lattice` kernel
   (described under *CUDA Kernels* below). The kernel reads `K`
   and `W` from device buffers, reads the current box lattice from
   `sim_box.lattice_device_mut()`, computes
   `V_pre = lx · ly · lz`, the instantaneous pressure

   ```text
   P = (2 K + W) / (3 V_pre)
   ```

   in `E_h / a_0^3`, and the Berendsen scale factor

   ```text
   μ³ = 1 − β · (dt / τ) · (P_target − P)
   μ  = max(μ_min, μ³)^(1/3)
   ```

   in `double` precision with parameters (`β`, `τ`, `P_target`)
   passed as scalar kernel arguments and `μ_min = 1.0e-6` the
   device-side safety floor. The kernel then mutates the device
   lattice in place (`lattice[i] ← μ · lattice[i]`), writes the
   device-side rescale factor `mu_device: CudaSlice<f32>` (length 1)
   for the position-rescale kernel in step 4, and writes the
   two-element diagnostic buffer
   `diagnostics_device: CudaSlice<f64>` (length 2):

   ```text
   diagnostics_device[0] = P                    // most_recent_pressure
   diagnostics_device[1] = V_post = μ³ · V_pre  // most_recent_volume
   ```

   Sign convention: when `P < P_target` the system is under-pressured,
   `(P_target − P) > 0`, `μ³ < 1`, and the box contracts. When
   `P > P_target` the system is over-pressured, `μ³ > 1`, and the
   box expands.

   The host's `sim_box` generation counter is incremented by 1
   inside the call (via `lattice_device_mut`); the host fields
   become stale until the next `sim_box.flush_from_device()`.

4. The per-particle position rescale `x_i ← μ · x_i` is dispatched
   by the JIT-composed post-force per-particle kernel, not by
   `apply`. Berendsen barostat's
   `post_force_per_particle_fragment()` carries the `x *=
   berendsen_mu_device[0]` body that the composed kernel inlines
   per thread. `apply` itself does not launch a per-particle
   rescale. The rescale is applied about the box origin; fractional
   coordinates relative to the new box are unchanged. No host
   involvement; no per-step dtoh of `μ`.

When `buffers.particle_count() == 0`, the entire hook is a no-op:
no kernel launches occur, the box is not mutated, and the
diagnostic buffer is unchanged.

Berendsen's diagnostic state has no analogue to C-rescale's
`cumulative_injection_delta` — the deterministic Berendsen path
does no detailed-balance correction, so no conserved-quantity
column is published.

## Per-Step Kernel Sequence <!-- rq-03d70e2d -->

Per timestep, the Berendsen barostat's `apply` runs the following
in fixed order:

| Order | Step              | Kernel / call                              | Operation                                                                                                          | Stage label                            |
| ----- | ----------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------- |
| 1     | KE reduce         | `kinetic_energy_reduce`                    | f32 scalar of `K` into `ke_scratch`                                                                                | `KineticEnergyReduce`                  |
| 2     | Virial reduce     | `virial_sum_reduce`                        | f32 scalar of `W` into `virial_scratch`                                                                            | `VirialSumReduce`                      |
| 3     | µ + lattice + diag | `berendsen_compute_mu_and_rescale_lattice` | reads `K`, `W`, lattice; computes µ + P in f64; mutates lattice; writes µ + diagnostics device buffers              | `BerendsenComputeMuAndRescaleLattice`  |
| 4     | Position rescale  | composed post-force per-particle kernel    | reads `mu_device[0]`, scales every particle position by it                                                          | `JitComposedPostForce`                 |

Steps 1–3 run inside `apply`. Step 4 runs from the JIT-composed
post-force per-particle kernel via Berendsen barostat's source
fragment.

No per-step host download occurs. The host `sim_box`'s lattice
mirror, `most_recent_pressure`, and `most_recent_volume` host
fields are stale between log rows. The runner refreshes them at
log-write cadence by calling `sim_box.flush_from_device()` and
`barostat.flush_pending_injection(device)` (the diagnostic flush
method; see `framework.md`).

The `kinetic_energy_reduce` and `virial_sum_reduce` kernels are
launched through the on-device helpers
`compute_kinetic_energy_on_device` (`nose-hoover-chain.md`) and
`compute_total_virial_on_device` (this slot — see *Feature API*
below); the `_on_device` variants leave the result in the supplied
scratch buffer without downloading it. The
`rescale_positions_device_factor` and lattice-rescale kernels are
described under *Feature API* below.

The integrator's own kernels (`vv_kick_drift`, `vv_kick`, the force
pipeline) and the thermostat's kernels are launched separately by
their respective hooks and are not part of this slot's per-step
sequence.

## Parameters <!-- rq-3db027c2 -->

The matching builder deserialises a typed `BerendsenBarostatParams` from the `[barostat]` section's `SlotConfig::params` (see `framework.md`); the per-field reference below documents that parameter struct:

- `pressure: f64` — target pressure `P_target` in `E_h / a_0^3` (the
  engine's atomic pressure unit).
  Required. Finite. May be any sign or zero; the formula handles
  negative targets, zero target (e.g. vacuum equilibration), and
  positive targets (the common case) identically.
- `tau: f64` — pressure-coupling time constant in atomic time units
  (`hbar / E_h`). Required.
  Finite and strictly positive. Typical values for liquid water are
  1–5 ps; longer than the thermostat's `τ` so that pressure
  fluctuations average out before the barostat responds.
- `compressibility: f64` — isothermal compressibility `β` in
  `a_0^3 / E_h` (the inverse atomic pressure unit). Required. Finite
  and strictly positive. Typical values: water ≈ `4.5e-10` 1/Pa
  ≈ `1.5e-2` in atomic units; LJ argon at liquid density similar
  order. An
  inaccurate value produces a different effective relaxation rate
  (`β · 1/τ`) but does not break correctness.

No RNG seed: the Berendsen barostat is deterministic.

## Diagnostic log columns <!-- rq-62b44dc9 -->

The barostat exposes two per-log-row diagnostic columns when it is
configured (see `io/log-output.md`):

- `pressure` — instantaneous pressure `P` in `E_h / a_0^3` (the
  engine's atomic pressure unit) as computed in
  step 4 of the algorithm. This is the value used to derive the
  step's `μ`. When `buffers.particle_count() == 0`, `pressure` is
  reported as `0.0`.
- `box_volume` — simulation-box volume `V` in cubic metres. The
  value reported is the post-rescale volume (`μ³ · V_pre`), matching
  the lattice that the trajectory frame writes at the same step.

No `*_conserved` column: the Berendsen barostat is not symplectic
and does not preserve any natural extended Hamiltonian.

## Empty State and degenerate cases <!-- rq-bf3ec42e -->

- `buffers.particle_count() == 0`: `apply` returns `Ok(())` without
  launching any kernel and without mutating `sim_box`. The `pressure`
  and `box_volume` log columns are populated as `0.0` and
  `sim_box.volume()` respectively.
- `V == 0` (degenerate box): `compute_pressure` would divide by zero;
  the runner refuses to construct a `SimulationBox` with zero
  volume (`pbc.rs`), so this case is unreachable.
- `μ³ ≤ 0`: clamped to `μ_min³ = 1e-18` (so `μ = 1e-6`); the next
  step's pressure is re-evaluated against the now-tiny box.
  Recoverable only if the user notices and tightens parameters.
- `2K + W == 0` (extremely cold system with cancelling virial):
  `P = 0`; `μ³ = 1 − β · (dt/τ) · P_target`. Standard handling.

## Feature API <!-- rq-142f60e8 -->

### Types <!-- rq-25e1c7c2 -->

- `BerendsenBarostat` — implements the `Barostat` trait declared in <!-- rq-5c758681 -->
  `framework.md`. Registered in `BarostatRegistry::with_builtins`
  under `kind_name() == "berendsen"`. Fields:

  - `device: Arc<CudaDevice>`
  - `pressure: f64` — `P_target`.
  - `tau: f64` — `τ`.
  - `compressibility: f64` — `β`.
  - `ke_scratch: CudaSlice<f32>` — length-1 device buffer for the
    kinetic-energy reduction; reused across calls.
  - `virial_scratch: CudaSlice<f32>` — length-1 device buffer for
    the virial reduction; reused across calls.
  - `most_recent_pressure: f64` — `P` from the most recent
    `apply` call. Used by `log_column_values`.
  - `most_recent_volume: f64` — post-rescale `V` from the most
    recent `apply` call. Used by `log_column_values`.

  All fields private; the slot's public surface is the `Barostat`
  trait methods and construction via `BerendsenBarostatBuilder`.
  `most_recent_pressure` and `most_recent_volume` are public for
  parity with other slots' diagnostic state so a future
  restart-from-checkpoint flow can restore them explicitly.

- `BerendsenBarostatBuilder` — implements `BarostatBuilder` with <!-- rq-4ef89c50 -->
  `kind_name() == "berendsen"`. `build(device, particle_count, kind)`
  deserialises `BerendsenBarostatParams` from `params`, allocates the
  two length-1 device scratch buffers, and returns the boxed
  `BerendsenBarostat`.

### `Barostat` trait overrides <!-- rq-ac5affb7 -->

`BerendsenBarostat` overrides every method on the `Barostat` trait
declared in `framework.md`:

- `apply(buffers, sim_box, dt, timings)` — runs the per-step <!-- rq-29dda250 -->
  algorithm above.
- `log_column_names() -> &'static ["pressure", "box_volume"]`. <!-- rq-b6728f3c -->
- `log_column_values(_ke, _pe) -> vec![most_recent_pressure, <!-- rq-82baba1a -->
  most_recent_volume]`. The runner's `ke` and `pe` arguments are
  unused (the barostat already computed and cached its own `P` and
  `V` during `apply`); the trait signature still receives them for
  uniformity with the integrator and thermostat slots.

### SimulationBox API <!-- rq-168727b6 -->

- `SimulationBox::rescale_isotropic(&mut self, factor: f32) -> <!-- rq-9e2e9d4e -->
  Result<(), SimulationBoxError>`
  - Multiplies all six lattice parameters `(lx, ly, lz, xy, xz, yz)`
    by `factor` in a single mutation that bumps the generation
    counter exactly once.
  - Returns `SimulationBoxError::InvalidLattice(...)` (the existing
    variant emitted by `validate_lattice`) when `factor` is
    non-finite, zero, or produces a non-finite / non-positive edge
    length. The host-side `μ_min` clamp in the barostat keeps `μ`
    well within the valid range under sensible parameters.
  - Convenience over `set_lattice(factor*lx, factor*ly, factor*lz,
    factor*xy, factor*xz, factor*yz)`; provided so callers cannot
    accidentally apply different scale factors to the orthogonal and
    shear components and silently change the triclinic shape.

### CUDA Kernels <!-- rq-d764ea16 -->

`kernels/barostat.cu` declares two `extern "C"` kernels:

```c
extern "C" __global__ void virial_sum_reduce(
    const float *virials,
    float *partial,       // shared mem, single f32 output by thread 0
    unsigned int n);

extern "C" __global__ void rescale_positions(
    float *positions_x, float *positions_y, float *positions_z,
    float factor,
    unsigned int n);
```

#### `virial_sum_reduce` <!-- rq-2b0862c1 -->

Structurally identical to `kinetic_energy_reduce`
(`nose-hoover-chain.md`). A single-block kernel with
`blockDim.x = 256`. Each thread loops over its strided subset of
particles, accumulating `virials[i]` into a register. The
per-thread partials are then summed across the block via the same
deterministic left-to-right pairwise reduction in shared memory used
by `kinetic_energy_reduce`. The output is a length-1
`CudaSlice<f32>` (held on `BerendsenBarostat.virial_scratch`) which
the host downloads via `dtoh_sync_copy_into` and promotes to `f64`
before the pressure formula.

Single-block execution underutilises the GPU for very large `n` but
keeps the determinism analysis trivial; the cost is negligible
relative to the force pipeline.

#### `rescale_positions` <!-- rq-fece5481 -->

One thread per particle. Thread `i`:

```c
positions_x[i] *= factor;
positions_y[i] *= factor;
positions_z[i] *= factor;
```

No interaction between threads; trivially deterministic. Block size
256, grid `ceil(n / 256)`. Does **not** update image flags, velocity
buffers, force buffers, or any neighbor-list reference positions.
Image flags are invariant under uniform scaling (fractional coords
unchanged); reference positions are refreshed automatically on the
next `force_field.step` via the box-generation change-detection path
(`forces/neighbor-list.md`).

### PTX Module Loading <!-- rq-a034083a -->

`init_device()` loads the compiled `kernels/barostat.cu` PTX as
module `"barostat"` and captures `virial_sum_reduce` and
`rescale_positions` into the `Kernels` handle (see
`build-pipeline.md`).

### Rust Launch Helpers <!-- rq-fdc545da -->

Two free functions in `src/gpu/kernels.rs`, re-exported from
`crate::gpu`:

- `compute_total_virial(buffers: &ParticleBuffers, scratch: &mut CudaSlice<f32>) -> Result<f32, GpuError>` <!-- rq-0f50dade -->
  - Launches `virial_sum_reduce` over `buffers.virials` with output
    `scratch` (a length-1 device buffer the caller owns; reused
    across calls to avoid per-step allocation).
  - Downloads `scratch[0]` host-side via `dtoh_sync_copy_into` and
    returns the value in Hartrees.
  - Block size 256, single block, no shared-memory tuning beyond
    what the kernel declares.
  - When `buffers.particle_count() == 0`, returns `Ok(0.0_f32)`
    without launching.
  - General-purpose: usable by any future barostat that needs the
    instantaneous virial.

- `rescale_positions(buffers: &mut ParticleBuffers, factor: f32) -> Result<(), GpuError>` <!-- rq-19916fb0 -->
  - Launches `rescale_positions` over `buffers.positions_*`.
  - Block size 256, grid `ceil(n / 256)`.
  - When `buffers.particle_count() == 0`, returns `Ok(())` without
    launching.
  - General-purpose: usable by any future barostat that performs a
    uniform isotropic position rescale.

## Launch Configuration <!-- rq-23a93703 -->

Per-step launch counts (per `apply` invocation):

- `kinetic_energy_reduce`: 1 launch (single block of 256 threads).
- `virial_sum_reduce`: 1 launch (single block of 256 threads).
- `rescale_positions`: 1 launch (block 256, grid `ceil(n/256)`).

All launches go through the default stream of
`ParticleBuffers::device`.

## Determinism <!-- rq-2b07d8fc -->

- All three kernels involved are deterministic by construction
  (single-block deterministic reductions; trivially parallel position
  rescale).
- The Berendsen barostat carries no RNG; there are no stochastic
  draws to randomise.
- The host-side `P`, `μ³`, and `μ` computations run in `f64` from
  `f32` inputs (`K`, `W`, `V`) and the deterministic parameters; two
  runs produce byte-identical `μ` and therefore byte-identical
  post-rescale positions and box.
- `SimulationBox::rescale_isotropic(μ)` is a pure deterministic
  multiplication; the generation counter is monotonically incremented
  in lock-step across runs.
- Two end-to-end runs composing the same integrator (and optionally
  the same thermostat) with the Berendsen barostat on the same GPU
  with identical configs and identical initial particle state
  produce byte-identical trajectory and log files, including the
  `pressure` and `box_volume` columns.

## Out of Scope <!-- rq-b2ccd87a -->

- Semi-isotropic and anisotropic box deformation. The slot rescales
  all six lattice parameters by a single scalar `μ`; per-axis or
  semi-isotropic (xy-coupled, z-independent) coupling is a separate
  feature.
- Stochastic-cell variants (Bussi-Parrinello). A stochastic NPT
  barostat would slot in alongside the Berendsen barostat under a
  distinct `kind` name.
- Extended-system barostats (Parrinello-Rahman, Martyna-Tobias-Klein).
  These integrators carry an additional dynamical box-momentum
  degree of freedom and would require an integrator with an
  augmented `step()`, not a post-step `Barostat::apply` hook;
  shipping one would also require the integrator to declare
  `owns_barostat()` analogously to `owns_thermostat()`.
- A runtime warning when the Berendsen barostat is selected. The
  caveat lives in this requirements file and in the `[barostat]`
  documentation in `config-schema.md`; the runner does not print a
  warning when loading a Berendsen-barostat config.
- Compressibility auto-tuning from system properties. The user
  supplies a fixed `β`.
- A `μ`-clamp configurable per-run. The host-side `μ_min = 1e-6`
  floor is a fixed safety guard; users who hit it should tighten
  their parameters.
- Constraint algorithms (SHAKE/RATTLE) and their interaction with
  the rescale. Constraints would need to be re-projected after the
  position rescale; the framework does not yet ship a constraint
  slot.

---

## Gherkin Scenarios <!-- rq-b6324878 -->

```gherkin
Feature: Berendsen weak-coupling barostat

  Background:
    Given a CUDA-capable GPU available as device 0
    And a SimulationBox with lx=ly=lz=1.0e-9 unless otherwise specified
    And init_device() has been called

  # --- Construction ---

  @rq-52b7d30e
  Scenario: Construct BerendsenBarostat via the registry
    Given a BarostatKind::Berendsen {
      pressure: 1.0e5, tau: 1.0e-12, compressibility: 4.5e-10 }
    When registry.build_optional(Some(&kind), device, particle_count=4) is called
    Then it returns Ok(Some(barostat))
    And the underlying BerendsenBarostat has most_recent_pressure == 0.0
    And the underlying BerendsenBarostat has most_recent_volume == 0.0
    And state.pressure == 1.0e5
    And state.tau == 1.0e-12
    And state.compressibility == 4.5e-10

  @rq-abf09acf
  Scenario: Construct with particle_count = 0
    Given a BarostatKind::Berendsen {
      pressure: 1.0e5, tau: 1.0e-12, compressibility: 4.5e-10 }
    When registry.build_optional(Some(&kind), device, particle_count=0) is called
    Then it returns Ok(Some(barostat))

  @rq-05e8f300
  Scenario: build_optional with None returns Ok(None)
    Given a BarostatRegistry::with_builtins()
    When registry.build_optional(None, device, particle_count=4) is called
    Then it returns Ok(None)
    And no builder is consulted

  @rq-909e5bb4
  Scenario: BarostatRegistry::with_builtins() exposes berendsen
    Given a BarostatRegistry::with_builtins()
    Then the registry has exactly one registered builder
    And that builder's kind_name() equals "berendsen"

  # --- Config validation (paired with config-schema scenarios) ---

  @rq-125677a3
  Scenario: Reject non-positive tau
    Given a config with [barostat] kind="berendsen",
      pressure=1.0e5, tau=-1.0e-12, compressibility=4.5e-10
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "barostat.tau", reason: _ })

  @rq-06772617
  Scenario: Reject non-positive compressibility
    Given a config with [barostat] kind="berendsen",
      pressure=1.0e5, tau=1.0e-12, compressibility=0.0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "barostat.compressibility", reason: _ })

  @rq-2adf8b58
  Scenario: Accept negative target pressure
    Given a config with [barostat] kind="berendsen",
      pressure=-1.0e5, tau=1.0e-12, compressibility=4.5e-10
    When load_config is called
    Then it returns Ok(config)

  @rq-7ac01c02
  Scenario: Missing pressure rejected
    Given a config with [barostat] kind="berendsen",
      tau=1.0e-12, compressibility=4.5e-10
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "barostat.pressure" })

  @rq-5dce727f
  Scenario: Missing tau rejected
    Given a config with [barostat] kind="berendsen",
      pressure=1.0e5, compressibility=4.5e-10
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "barostat.tau" })

  @rq-414ed5ef
  Scenario: Missing compressibility rejected
    Given a config with [barostat] kind="berendsen",
      pressure=1.0e5, tau=1.0e-12
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "barostat.compressibility" })

  @rq-90eab90b
  Scenario: Reject extra fields
    Given a config with [barostat] kind="berendsen",
      pressure=1.0e5, tau=1.0e-12, compressibility=4.5e-10, seed=42
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, message })
    And path equals "barostat"
    And message mentions "seed"

  # --- compute_total_virial helper ---

  @rq-cf4d6ab4
  Scenario: compute_total_virial of a single particle with virials[0]=0 is zero
    Given a ParticleBuffers from a single particle with virials[0]=0.0
    When compute_total_virial(&buffers, &mut scratch) is called
    Then it returns Ok(0.0_f32)

  @rq-098fabd1
  Scenario: compute_total_virial matches the host-side sum on small N
    Given a ParticleBuffers from N=4 particles with virials = [1.0, -2.0, 3.0, -4.0]
    When compute_total_virial(&buffers, &mut scratch) is called
    Then the returned value equals -2.0_f32 within f32 round-off

  @rq-d801f67c
  Scenario: compute_total_virial is deterministic
    Given two ParticleBuffers built from byte-identical ParticleStates of N=1000
    When compute_total_virial is called on each with its own scratch buffer
    Then the two returned values agree byte-for-byte

  @rq-4a328491
  Scenario: compute_total_virial on empty state returns 0.0 without launching
    Given a ParticleBuffers with particle_count() == 0
    When compute_total_virial(&buffers, &mut scratch) is called
    Then it returns Ok(0.0_f32)

  # --- rescale_positions helper ---

  @rq-77292dee
  Scenario: rescale_positions multiplies every position component by the factor
    Given a ParticleBuffers from N=2 particles with x0=(1, 2, 3) and x1=(-4, 5, -6)
    When rescale_positions(&mut buffers, factor=0.5) is called
    And the buffers are downloaded
    Then positions_x equals [0.5, -2.0]
    And positions_y equals [1.0, 2.5]
    And positions_z equals [1.5, -3.0]

  @rq-00e98375
  Scenario: rescale_positions does not modify velocities, forces, masses, or images
    Given a ParticleBuffers with N=4 non-zero positions, velocities, forces, images
    And a snapshot of velocities, forces, masses, images
    When rescale_positions(&mut buffers, factor=0.7) is called
    And the buffers are downloaded
    Then velocities_*, forces_*, masses, images_* are byte-identical to the snapshot

  @rq-2fc35d61
  Scenario: rescale_positions with factor=1.0 is the identity
    Given a ParticleBuffers from N=4 non-zero positions
    And a snapshot of positions
    When rescale_positions(&mut buffers, factor=1.0) is called
    And the buffers are downloaded
    Then positions are byte-identical to the snapshot

  @rq-64c051d4
  Scenario: rescale_positions on empty state is a no-op
    Given a ParticleBuffers with particle_count() == 0
    When rescale_positions(&mut buffers, factor=0.5) is called
    Then it returns Ok(())

  # --- SimulationBox::rescale_isotropic ---

  @rq-af9257bb
  Scenario: rescale_isotropic multiplies all six lattice parameters
    Given a SimulationBox with (lx, ly, lz, xy, xz, yz) = (1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
    When sim_box.rescale_isotropic(0.5) is called
    Then sim_box.lx() == 0.5
    And sim_box.ly() == 1.0
    And sim_box.lz() == 1.5
    And sim_box.xy() == 0.05
    And sim_box.xz() == 0.10
    And sim_box.yz() == 0.15

  @rq-911d9120
  Scenario: rescale_isotropic bumps the generation counter
    Given a SimulationBox with generation g
    When sim_box.rescale_isotropic(1.01) is called
    Then sim_box.generation() == g + 1

  @rq-b0b4c220
  Scenario: rescale_isotropic rejects zero factor
    Given a SimulationBox with any valid lattice
    When sim_box.rescale_isotropic(0.0) is called
    Then it returns Err(SimulationBoxError::InvalidLattice(_))

  @rq-9ba11e1e
  Scenario: rescale_isotropic rejects non-finite factor
    Given a SimulationBox with any valid lattice
    When sim_box.rescale_isotropic(f32::NAN) is called
    Then it returns Err(SimulationBoxError::InvalidLattice(_))

  # --- Per-step kernel sequence ---

  @rq-92cecd28
  Scenario: apply launches the expected kernel set
    Given a Berendsen barostat with pressure=1.0e5, tau=1.0e-12, compressibility=4.5e-10
    And a ParticleBuffers with N=4 non-zero velocities and virials
    When barostat.apply(&mut buffers, &mut sim_box, dt=1e-15, &mut timings) is called
    Then KernelStage::KINETIC_ENERGY_REDUCE has count == 1
    And KernelStage::VIRIAL_SUM_REDUCE has count == 1
    And KernelStage::BERENDSEN_BAROSTAT_RESCALE_POSITIONS has count == 1
    And KernelStage::VV_KICK_DRIFT has count == 0
    And KernelStage::VV_KICK has count == 0

  @rq-69600add
  Scenario: apply on empty state is a no-op
    Given a Berendsen barostat with particle_count=0
    When barostat.apply(...) is called
    Then it returns Ok(())
    And sim_box.generation() is unchanged

  # --- Pressure / μ correctness ---

  @rq-a2bb55c6
  Scenario: μ equals 1 when instantaneous pressure equals target
    Given an N=8 system with velocities and virials placed so (2K + W)/(3V) exactly equals P_target
    When barostat.apply(...) is called
    Then the rescale factor μ is 1.0 within f32 round-off
    And the post-step positions equal the pre-rescale positions byte-for-byte
    And sim_box.volume() equals the pre-step volume within f32 round-off

  @rq-c9f9d550
  Scenario: μ < 1 when instantaneous pressure below target (system contracts)
    Given an N=8 system with K and W such that P = P_target / 2
    When barostat.apply(...) is called with dt = 0.1·τ
    Then the rescale factor μ satisfies μ³ = 1 − β · 0.1 · (P_target − P)
    And μ < 1
    And sim_box.volume() decreases by exactly μ³

  @rq-ed3ed814
  Scenario: μ > 1 when instantaneous pressure above target (system expands)
    Given an N=8 system with K and W such that P = 2·P_target
    When barostat.apply(...) is called with dt = 0.1·τ
    Then the rescale factor μ satisfies μ³ = 1 − β · 0.1 · (P_target − P)
    And μ > 1
    And sim_box.volume() increases by exactly μ³

  @rq-4dbe4a07
  Scenario: μ clamped to μ_min when the formula yields a non-positive μ³
    Given a system with K and W such that P_target − P is large enough that
      β · (dt/τ) · (P_target − P) > 1
    When barostat.apply(...) is called
    Then μ equals 1.0e-6 (the host-side safety floor)
    And sim_box.volume() shrinks to μ³ · V_pre = 1.0e-18 · V_pre

  # --- Fractional-coord and PBC invariants ---

  @rq-cf183b79
  Scenario: Fractional coordinates of every particle are invariant under apply
    Given an N=8 system with arbitrary positions
    And a snapshot of fractional coordinates (positions / lattice) per particle
    When barostat.apply(...) is called
    Then the post-step fractional coordinates of every particle equal the
      snapshot within f32 round-off
    And no particle moved across an image boundary (image flags unchanged)

  @rq-16252a37
  Scenario: Triclinic shape is preserved under apply
    Given a triclinic SimulationBox with non-zero xy, xz, yz
    And a snapshot of (xy/lx, xz/lx, yz/ly) ratios
    When barostat.apply(...) is called
    Then the post-step (xy/lx, xz/lx, yz/ly) ratios equal the snapshot
      within f32 round-off

  # --- Box-generation propagation ---

  @rq-136f7d15
  Scenario: sim_box.generation() advances after apply
    Given a Berendsen barostat and a SimulationBox at generation g
    When barostat.apply(...) is called
    Then sim_box.generation() == g + 1

  # --- Log columns ---

  @rq-7564b1e7
  Scenario: log_column_names returns ["pressure", "box_volume"]
    Given a constructed BerendsenBarostat
    Then state.log_column_names() equals ["pressure", "box_volume"]

  @rq-24073418
  Scenario: log_column_values returns the cached most-recent pressure and post-rescale volume
    Given a BerendsenBarostat with most_recent_pressure = 1.01e5 and
      most_recent_volume = 1.0e-27
    When state.log_column_values(_ke, _pe) is called
    Then it returns [1.01e5, 1.0e-27]

  @rq-75297a48
  Scenario: Log file header includes pressure and box_volume when the Berendsen barostat is the configured barostat
    Given a config with [barostat].kind = "berendsen"
    And log_every > 0
    When the runner produces the log file
    Then its header line is "step,time,kinetic_energy,temperature,pressure,box_volume"

  # --- Composition with thermostat and integrator ---

  @rq-ad67b3da
  Scenario: Berendsen barostat composes with velocity-Verlet integrator (NPH)
    Given a composed runner of velocity-Verlet + Berendsen barostat with N=128 LJ
      argon, pressure = 1.0 bar, tau = 1.0e-12, compressibility = 1.0e-9, n_steps = 1000
    When the run completes
    Then the run finishes without error
    And the time-averaged pressure over the last 500 log rows is within 50% of 1.0 bar

  @rq-13bf10fc
  Scenario: Berendsen barostat composes with velocity-Verlet + CSVR thermostat (NPT)
    Given a composed runner of velocity-Verlet + CSVR + Berendsen barostat with
      N=128 LJ argon at T_target = 85 K, P_target = 1.0 bar, n_steps = 1000
    When the run completes
    Then the run finishes without error
    And the time-averaged temperature over the last 500 log rows is within 5% of 85 K
    And the time-averaged pressure over the last 500 log rows is within 50% of 1.0 bar

  @rq-2d579721
  Scenario: Berendsen barostat composes with langevin-baoab integrator
    Given a composed runner of langevin-baoab (no [thermostat]) + Berendsen barostat with
      N=128 LJ argon, T_target = 85 K (set on langevin-baoab), P_target = 1.0 bar
    When load_config and run are invoked
    Then the run finishes without error
      (the integrator-owns-thermostat compatibility rule applies only to
       [thermostat]; [barostat] composes freely with langevin-baoab)

  # --- Determinism ---

  @rq-3460c38a
  Scenario: Two independent composed runs with identical configs are byte-identical
    Given two complete simulations composing velocity-Verlet + Berendsen barostat
      with identical parameters, identical initial state, n_steps = 10
    When heddlemd run is invoked on each
    Then the trajectory files are byte-identical
    And the log files are byte-identical, including the pressure and box_volume columns
    And the final SimulationBox lattices are byte-identical
```
