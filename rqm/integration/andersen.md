# Feature: Andersen Stochastic Thermostat <!-- rq-5e059f6b -->

The Andersen thermostat (Andersen, *J. Chem. Phys.* **72**, 2384
(1980)) is a stochastic NVT thermostat. One of the pluggable
thermostat slots (see `framework.md`); selected by
`kind = "andersen"` in the config's `[thermostat]` section.

The thermostat runs once per timestep, as a post-only hook
(`apply_post`); its `apply_pre` is the trait default no-op. Each rigid
group (the connectivity-derived molecule partition; every atom is its own
group on an unconstrained monatomic run) is independently and randomly
selected with probability `p = collision_rate · dt`, and a selected
group's atoms all receive fresh velocities drawn from the
Maxwell-Boltzmann distribution at the user-specified temperature. Groups
that are not selected retain the velocity produced by the symplectic step.
Under holonomic constraints the terminal velocity projection then maps a
selected group's fresh velocity onto its constraint manifold; the
whole-group resample is what makes that projection a correct
constrained-Maxwell-Boltzmann draw (see *Composition with a constraint
slot*). Repeated application produces the canonical distribution exactly.

The thermostat preserves no momentum: a resampled group's centre-of-mass
momentum is drawn afresh, so the total centre-of-mass momentum drifts
stochastically. Time-correlation functions are distorted by the collisions
in proportion to `collision_rate`. Andersen is the strongest sampler in
the registry — it converges to the canonical distribution from any
starting condition — and the weakest at preserving the underlying
dynamics.

## Algorithm <!-- rq-e15cc5ac -->

The thermostat is invoked through `apply_post(buffers, dt, timings)`
at the runner's post-force-marker boundary. The integrator has already
completed its velocity-Verlet substeps, including the trailing kick, and
on a constrained run the runner has already projected the velocities back
onto the constraint manifold
(`Constraint::apply_after_kick`, see
`constraint-framework.md`). Andersen does not couple to the global
kinetic energy at all — it resamples velocities from Maxwell-Boltzmann —
so that leading projection does not change what the thermostat couples
to; it does mean the `K_old` reduction below is measured on the projected,
on-manifold velocities. For each invocation with timestep `dt`:

1. Compute the instantaneous kinetic energy `K_old` via the shared
   `compute_kinetic_energy` helper (`nose-hoover-chain.md`).
2. For each **rigid group** `G` — the connectivity-derived molecule
   partition of `forces/topology.md` (a monatomic run's groups are all
   singletons):
   - Draw a uniform `U_G ∈ (0, 1)` from a counter-based Philox-4×32-10
     stream keyed on the group's lowest-indexed atom (see *RNG* below).
   - If `U_G < p` where `p = clamp(collision_rate · dt, 0, 1)`:
     replace the velocity of **every atom** of `G` with a fresh
     Maxwell-Boltzmann sample at temperature `T`:
     `v_i ← σ_i · (ξ_x, ξ_y, ξ_z)` with `σ_i = sqrt(T / m_i)` (`k_B = 1`
     in the engine's atomic units; `T` carries `k_B · T` in Hartrees)
     and `ξ_a ~ N(0, 1)` independently per axis.
   - Otherwise leave every atom of `G` unchanged.
3. Compute the instantaneous kinetic energy `K_new` via the same
   helper, and update the running
   `cumulative_injection += K_new − K_old`. Used by the
   `andersen_conserved` log column.

The collision decision and resample are **per group**, not per atom, and
this is required for correctness under holonomic constraints — see
*Composition with a constraint slot*. The resampling runs as the
standalone `andersen_resample_grouped` kernel launched from
`apply_post`. Like every
thermostat, Andersen runs its post-force work as standalone launches on
coupling steps (`framework.md`, `docs/architecture.md`); it is never
part of a composed kernel. All of the thermostat's work — including a
resample that has no reduction dependency of its own — runs as its own
launches. The
two surrounding kinetic-energy reductions (for the conserved-quantity
accounting) each launch the shared `kinetic_energy_reduce` kernel from
inside `apply_post`.

The collision probability is clamped to `[0, 1]` so any
`collision_rate · dt_couple ≥ 1` reduces to "always resample" (massive
Andersen). This is mathematically valid; the canonical-distribution
guarantee holds.

`apply_pre` is the trait default (no-op): Andersen has a single
resample, applied after the integrator runs. `apply_post` runs only on
coupling steps (every `coupling_interval` steps, `io/config-schema.md`)
and its collision probability uses the effective timestep
`dt_couple = coupling_interval · dt`, so resampling less often raises
the per-attempt collision probability correspondingly.

### Composition with a constraint slot <!-- rq-225a5959 -->

**The collision must resample a whole rigid group at a time.** A rigid
group's equilibrium velocity distribution is not the full 3-per-atom
Maxwell-Boltzmann — it is that distribution restricted to the group's
velocity manifold (the subspace where every constraint's time derivative
is zero). The correct way to draw from it is to draw a full,
unconstrained Maxwell-Boltzmann velocity for the whole group and project
that onto the manifold: a mass-metric orthogonal projection of an
isotropic Gaussian onto the `D`-dimensional manifold has expected kinetic
energy `(D/2) · k_B T`, which is exactly the manifold's equipartition, so
the projected draw is the constrained-Maxwell-Boltzmann sample. Resampling
**all** of a group's atoms and letting the step's terminal velocity
projection map the result onto the manifold does precisely this.

Resampling a *single* atom of a group does not. The fresh single-atom
velocity is combined with the group's other, unchanged (already
on-manifold) velocities, and the projection of that mixed state is not a
draw from the manifold-Maxwell-Boltzmann: part of the injected energy
lands in the constrained directions and the projection discards it, so
the fixed point of the sparse single-atom process runs cold. The deficit
shrinks as the collision rate rises — as more of a group's atoms are
resampled per step the process approaches the whole-group resample, and
at the massive limit (`p = 1`, every atom every step) the two coincide
and the temperature is correct. A correct thermostat must hit its
setpoint at *every* coupling rate, so the per-group resample is the
required form; the per-atom form is a rate-dependent bug.

The plan's terminal `ConstraintPoint { AfterKick }` — which runs *after*
`apply_post` (see `framework.md`, *Per-Step Interface*) — is what applies
the manifold projection, and a constrained Andersen run depends on it
(RATTLE-last; see `constraint-framework.md`). It is load-bearing for
Andersen in a way it is not for the uniform-rescale thermostats: a global
rescale `v ← α · v` commutes with the projection (it scales the
constrained-direction components the leading projection has already
zeroed, so the terminal projection is a no-op for CSVR, Berendsen, and
NHC), whereas a fresh group resample has a generically non-zero
component along the group's constrained directions and must be projected
back.

That repair projection applies a real impulse, and the constraint virial
of that impulse is **dropped**: the terminal hook does not publish, so
only the leading projection's virial (published before the barostat reads
it) enters the pressure. This is an accepted approximation. The impulse
answers a stochastic velocity resample rather than a physical force, so it
carries no meaningful contribution to the mechanical pressure of the
system.

## Per-Step Kernel Sequence <!-- rq-7843f188 -->

On a coupling step the Andersen thermostat's `apply_post` runs the
following in fixed order, all as its own kernel launches:

| Order | Step      | Kernel / call           | Operation                                            | Stage label           |
| ----- | --------- | ----------------------- | ---------------------------------------------------- | --------------------- |
| 1     | KE reduce | `kinetic_energy_reduce` | one f32 scalar of `K_old`                            | `KineticEnergyReduce` |
| 2     | Resample  | `andersen_resample_grouped` | per-group Bernoulli + whole-group Maxwell-Boltzmann draw | `AndersenResample` |
| 3     | KE reduce | `kinetic_energy_reduce` | one f32 scalar of `K_new`                            | `KineticEnergyReduce` |

All three steps run inside `apply_post`. `kinetic_energy_reduce` is
reused from `nose-hoover-chain.md`. The integrator's own kernels
(`vv_kick_drift`, the force pipeline) are launched separately by
`integrator.step()` and are not part of this slot's per-step
sequence.

### Resample Kernel <!-- rq-a060db3f -->

`andersen_resample_grouped` is a per-group kernel (block 256, grid
`ceil(n_groups / 256)`) owned by this slot. Each thread handles one rigid
group of the connectivity-derived molecule partition (uploaded once at
phase setup by `Thermostat::init_run`; singletons for a run with no bonds
and no constraints). Each thread:

- performs one Bernoulli draw against the clamped collision probability
  `p_collision`, keyed on the group's lowest-indexed atom's stable
  particle id; if the group is selected, resamples **every** atom of the
  group, drawing three Gaussians per atom scaled by `sqrt(kT / masses[a])`
  keyed on that atom's stable id, using a slot-prefixed `__device__`
  Philox routine;
- the kernel increments the device draw counter exactly once per launch
  (via a single thread in the first block), so repeated launches consume
  disjoint, reproducible Philox streams.

Keying the per-atom Gaussians on each atom's stable id, and the group
decision on the group's lowest stable id, makes the resample independent
of the neighbour-list atom ordering and bit-identical run to run. For a
singleton group the decision and the draw are keyed on that atom's own id,
so the grouped kernel is bit-identical to a per-atom resample on an
unconstrained run.

Its arguments are the group tables (`mol_atom_offsets`,
`mol_atom_indices`, `n_groups`), the slot's `draw_counter_device` buffer,
`seed`, `p_collision` (clamped to `[0, 1]`), and `kT`. A separate
per-particle `andersen_resample` kernel exists for direct kernel-level
tests but is not on the thermostat's per-step path.

## Parameters <!-- rq-eb0bc993 -->

The matching builder deserialises a typed `AndersenParams` from the `[thermostat]` section's `SlotConfig::params` (see `framework.md`); the per-field reference below documents that parameter struct:

- `temperature: f64` — bath temperature `T` as `k_B · T` in Hartrees
  (the engine's internal temperature representation; `k_B = 1`).
  Required. Finite and strictly positive. Independent of
  `simulation.temperature`, which governs the initial-velocity
  sampler.
- `collision_rate: f64` — per-particle stochastic collision frequency
  `ν` in inverse atomic time units (`1 / (hbar / E_h)`). Required.
  Finite and `≥ 0` (`0` degenerates to NVE — no resampling — and is
  permitted as a diagnostic mode). Typical values for liquid water are
  `10¹¹–10¹² s⁻¹`, i.e. `~2.4e-6 – 2.4e-5` in atomic units (collision
  probability `≈ 10⁻⁴–10⁻²` per fs).
- `seed: u64` — counter-based RNG seed. Required, independent of
  `simulation.seed` and any other slot's seed.

The per-step collision probability is computed as
`p = clamp(collision_rate · dt, 0.0, 1.0)`.

## RNG <!-- rq-b3086891 -->

Andersen draws, per particle per invocation, one uniform sample for
the Bernoulli decision and three standard-normal samples (one per
axis) when the decision accepts. All draws come from the same
Philox-4×32-10 RNG used by `langevin-baoab.md` and `csvr.md`; the
device-side helpers `philox4x32_10`, `philox_gaussian`, and
`u32_to_uniform_open` live in `kernels/philox.cuh` and are shared by
every RNG-consuming kernel.

### Counter packing <!-- rq-d6a1b8eb -->

Each per-(particle, draw-kind) Philox invocation uses:

- **Key (2 × u32)**: `(seed_lo, seed_hi)` — low and high halves of
  the thermostat's `seed`.
- **Counter (4 × u32)**:
  - `counter[0] = draw_counter_lo` — low 32 bits of the thermostat's
    `draw_counter`.
  - `counter[1] = draw_counter_hi` — high 32 bits.
  - `counter[2] = particle_id` — the particle's `u32` ID from
    `ParticleBuffers::particle_ids`.
  - `counter[3] = draw_kind`:
    - `0` — Gaussian for the x-axis.
    - `1` — Gaussian for the y-axis.
    - `2` — Gaussian for the z-axis.
    - `3` — uniform for the Bernoulli decision.

The `draw_counter` lives on `AndersenThermostat`. It starts at `0`
at construction and is pre-incremented on every `apply_post` call
before the kernel launch; the first launch in a run therefore uses
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
H_andersen = K + U − Σ_{invocations} (K_new − K_old)
```

i.e., the physical Hamiltonian plus a running subtraction of the
cumulative kinetic energy injected (or removed) by the resampling
over all completed invocations. With a correct implementation
`{H_andersen(t)}` drifts as a martingale (zero-mean increment per
step) and is bounded in expectation; an implementation bug produces
a systematic drift.

`H_andersen` is exposed as a per-log-row diagnostic column named
`andersen_conserved` when Andersen is the configured thermostat (see
`io/log-output.md`). The thermostat accumulates `Σ (K_new − K_old)`
on its host-side state across every `apply_post` call from the two
kinetic-energy reductions in steps 1 and 3 of the per-step sequence
above.

## Empty State and degenerate cases <!-- rq-645252f1 -->

- `particle_count == 0`: `apply_pre` and `apply_post` return
  `Ok(())` without launching any kernel.
- `collision_rate == 0.0`: `p == 0`. No particle is ever resampled;
  the thermostat degenerates to an NVE pass-through (plus two
  redundant KE reductions per step). Useful as a diagnostic baseline.
- `collision_rate · dt ≥ 1.0`: `p` is clamped to `1.0`. Every
  particle is resampled every step ("massive Andersen"). The KE
  before/after the resample have entirely independent expected
  values; `H_andersen` is still a valid diagnostic.

## Feature API <!-- rq-4fad2dd6 -->

### Types <!-- rq-46792e3e -->

- `AndersenThermostat` — implements the `Thermostat` trait declared <!-- rq-feba0a88 -->
  in `framework.md`. Registered in `ThermostatRegistry::with_builtins`
  under `kind_name() == "andersen"`. Fields:

  - `device: Arc<CudaDevice>`
  - `temperature: f64`
  - `collision_rate: f64`
  - `seed: u64`
  - `draw_counter: u64` — Philox counter advance for the per-step
    resample kernel. Initialised to `0`; pre-incremented on every
    `apply_post` call.
  - `kt: f64` — equals `temperature` (the engine stores `k_B · T` in
    Hartrees directly; `k_B = 1`, so no Boltzmann constant appears).
  - `cumulative_injection: f64` — running sum of `K_new − K_old`
    across every completed `apply_post` call. Initialised to `0.0`.
    Used by `log_column_values`.
  - `ke_scratch: CudaSlice<f32>` — length-1 device buffer for the
    kinetic-energy reduction; reused across calls.
  - `most_recent_ke: f64` — last kinetic energy computed during the
    current `apply_post` invocation.
  - `mol_atom_offsets: CudaSlice<u32>`, `mol_atom_indices: CudaSlice<u32>`,
    `n_groups: usize` — the rigid-group partition the grouped resample
    kernel walks. Constructed as singletons (`n_groups == particle_count`)
    and replaced by the connectivity-derived partition in `init_run`.

  All fields private; the slot's public surface is the `Thermostat`
  trait methods and construction via `AndersenBuilder`. The
  `draw_counter` and `cumulative_injection` fields are public for
  parity with the other registered thermostats so a future
  restart-from-checkpoint flow can restore them explicitly.

- `AndersenBuilder` — implements `ThermostatBuilder` with <!-- rq-fd0cef60 -->
  `kind_name() == "andersen"`. `build(device, particle_count, kind)`
  deserialises `AndersenParams` from `params`, allocates the
  length-1 `ke_scratch` device buffer, and returns the boxed
  `AndersenThermostat`.

### `Thermostat` trait overrides <!-- rq-3ace72b0 -->

`AndersenThermostat` overrides the diagnostic-column trait methods
declared in `framework.md`:

- `log_column_names() -> &'static ["andersen_conserved"]`. <!-- rq-1163481e -->
- `log_column_values(ke, pe) -> vec![ke + pe − cumulative_injection]`. <!-- rq-6d2daea0 -->

It also overrides `init_run(&MoleculeList)` to upload the
connectivity-derived rigid-group partition (`mol_atom_offsets`,
`mol_atom_indices`, `n_groups`) that the grouped resample kernel walks.
The runner calls it once at phase setup, with the same `MoleculeList` it
hands the barostat.

`apply_pre` is left at its trait default (no-op).

### CUDA Kernels <!-- rq-b7c6d7d8 -->

`kernels/andersen.cu` declares two `extern "C"` kernels and
`#include`s `kernels/philox.cuh` for the shared Philox device-side
helpers. The thermostat launches the grouped kernel; the per-particle
kernel is retained for direct kernel-level tests.

```c
// Per-group resample — the one the thermostat launches. One thread per group.
extern "C" __global__ void andersen_resample_grouped(
    Real *velocities_x, Real *velocities_y, Real *velocities_z,
    const Real *masses,
    const unsigned int *particle_ids,
    const unsigned int *mol_atom_offsets,   // length n_groups + 1
    const unsigned int *mol_atom_indices,   // length N, storage indices per group
    const unsigned long long *draw_counter,
    unsigned int seed_lo, unsigned int seed_hi,
    Real p_collision,         // clamped to [0, 1] by the host
    Real kt,                  // k_B * temperature, atomic units
    unsigned int n_groups);

// Per-particle resample — retained for kernel-level tests.
extern "C" __global__ void andersen_resample(
    Real *velocities_x, Real *velocities_y, Real *velocities_z,
    const Real *masses,
    const unsigned int *particle_ids,
    const unsigned long long *draw_counter,
    unsigned int seed_lo, unsigned int seed_hi,
    Real p_collision,
    Real kt,
    unsigned int n);
```

The grouped kernel maps one thread to each group
`g = blockIdx.x · blockDim.x + threadIdx.x` (returning if `g ≥ n_groups`),
reads the group's atom slice `[mol_atom_offsets[g], mol_atom_offsets[g+1])`
from `mol_atom_indices`, makes one Bernoulli draw keyed on the group's
first atom's `particle_id` (draw kind 3), and on a hit resamples every
atom of the group, each atom's three Gaussians keyed on that atom's own
`particle_id` (draw kinds 0/1/2). The per-particle kernel below is the
singleton case of this. Its per-thread logic:

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

- `kinetic_energy_reduce` is deterministic by construction (see
  `nose-hoover-chain.md`).
- `andersen_resample` reads only `(seed, draw_counter, particle_id)`
  for its Philox input; no per-thread RNG state. Two runs on the
  same GPU with identical inputs produce byte-identical velocities.
- The `draw_counter` advances by exactly `+1` per `apply_post` call.
- Two end-to-end runs composing the same integrator with Andersen
  with identical configs on the same GPU produce byte-identical
  trajectory and log files, including the `andersen_conserved`
  column.

## Out of Scope <!-- rq-c5f66ecd -->

- A lossless `(f32, f64)` compensated mode. Andersen is stochastic;
  bit-exact time-reversibility is not a property the algorithm
  promises.
- COM-momentum-preserving variant. Andersen intrinsically breaks
  momentum conservation; a hybrid scheme that subtracts a global
  drift after each resample would be a different thermostat.
- Per-type collision rates. The single `collision_rate` applies to
  every particle regardless of species.
- Massive Andersen as a distinct `kind` name. Setting
  `collision_rate · dt ≥ 1` (clamped to `p = 1`) achieves the same
  effect under the existing `"andersen"` slot.
- Constraint algorithms (SHAKE/RATTLE).
- Pressure coupling (Andersen does not extend to NPT; the
  `[barostat]` slot composes orthogonally with any thermostat).
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
  Scenario: Construct AndersenThermostat via the registry
    Given a ThermostatKind::Andersen {
      temperature: 300.0, collision_rate: 1.0e12, seed: 42 }
    When registry.build_optional(Some(&kind), device, particle_count=4) is called
    Then it returns Ok(Some(thermostat))
    And the underlying AndersenThermostat has draw_counter == 0
    And state.cumulative_injection == 0.0
    And state.kt == k_B * 300.0

  @rq-abcae430
  Scenario: Construct with particle_count = 0
    Given a ThermostatKind::Andersen {
      temperature: 300.0, collision_rate: 1.0e12, seed: 1 }
    When registry.build_optional(Some(&kind), device, particle_count=0) is called
    Then it returns Ok(Some(thermostat))

  @rq-62ad70f8
  Scenario: Construct with collision_rate = 0 (NVE-equivalent)
    Given a ThermostatKind::Andersen {
      temperature: 300.0, collision_rate: 0.0, seed: 1 }
    When registry.build_optional(Some(&kind), device, particle_count=4) is called
    Then it returns Ok(Some(thermostat))

  # --- Config validation ---

  @rq-b8aa57c6
  Scenario: Reject non-positive temperature
    Given a config with [thermostat] kind="andersen",
      temperature=0.0, collision_rate=1.0e12, seed=1
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue {
      field: "thermostat.temperature", reason: _ })

  @rq-0f3b352b
  Scenario: Reject negative collision_rate
    Given a config with [thermostat] kind="andersen",
      temperature=300.0, collision_rate=-1.0, seed=1
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue {
      field: "thermostat.collision_rate", reason: _ })

  @rq-c4581536
  Scenario: collision_rate = 0 accepted
    Given a config with [thermostat] kind="andersen",
      temperature=300.0, collision_rate=0.0, seed=1
    When load_config is called
    Then it returns Ok(config)

  @rq-c5b42daa
  Scenario: Missing seed rejected
    Given a config with [thermostat] kind="andersen",
      temperature=300.0, collision_rate=1.0e12
    When load_config is called
    Then it returns Err(ConfigError::MissingField {
      field: "thermostat.seed" })

  @rq-8df9d74b
  Scenario: Reject extra fields (e.g. tau from NHC/CSVR)
    Given a config with [thermostat] kind="andersen",
      temperature=300.0, collision_rate=1.0e12, seed=1, tau=1.0e-13
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, message })
    And path equals "thermostat"
    And message mentions "tau"

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
  Scenario: apply_post launches the resample kernel
    Given an Andersen thermostat with temperature=300, collision_rate=1e12,
      seed=1, particle_count=4
    And buffers prepared with non-zero velocities
    When thermostat.apply_post(&mut buffers, dt=1e-15, &mut timings) is called
    Then KernelStage::ANDERSEN_RESAMPLE has count == 1
    And KernelStage::KINETIC_ENERGY_REDUCE has count == 0
      (Andersen resamples velocities directly; it runs no kinetic-energy
      reduction of its own)

  @rq-8fdfc981
  Scenario: apply_post on empty state is a no-op
    Given an Andersen thermostat with particle_count=0
    When thermostat.apply_post(...) is called
    Then it returns Ok(())

  @rq-15e44a1b
  Scenario: apply_pre is the trait default (no-op)
    Given an Andersen thermostat with particle_count=4
    And a snapshot of buffers.velocities before the call
    When thermostat.apply_pre(&mut buffers, dt=1e-15, &mut timings) is called
    Then it returns Ok(())
    And velocities are bit-identical to the snapshot
    And no kernel launches are recorded for that call

  # --- draw_counter and cumulative_injection bookkeeping ---

  @rq-c814659f
  Scenario: draw_counter starts at 0 and increments by 1 per apply_post
    Given a freshly built AndersenThermostat
    Then state.draw_counter == 0
    When thermostat.apply_post(...) is called once
    Then state.draw_counter == 1
    When thermostat.apply_post(...) is called a second time
    Then state.draw_counter == 2

  @rq-b1e87ce4
  Scenario: cumulative_injection is not tracked (legacy field stays zero)
    Given an Andersen thermostat and a ParticleBuffers with known non-zero velocities
    When thermostat.apply_post(...) is called once
    Then state.cumulative_injection remains 0
      (the injection accounting the conserved column would need requires a
      kinetic-energy measurement before and after the resample, which
      apply_post does not perform)

  # --- Log columns ---

  @rq-8eb14902
  Scenario: log_column_names returns ["andersen_conserved"]
    Given a constructed AndersenThermostat
    Then state.log_column_names() equals ["andersen_conserved"]

  @rq-26ff4aea
  Scenario: log_column_values returns ke + pe − cumulative_injection
    Given an AndersenThermostat with cumulative_injection = 1.0e-20
    When state.log_column_values(ke=2.5e-20, pe=3.0e-20) is called
    Then it returns [2.5e-20 + 3.0e-20 − 1.0e-20] = [4.5e-20]

  @rq-c50c6f84
  Scenario: Log file header includes andersen_conserved when Andersen is the thermostat
    Given a config with [thermostat].kind = "andersen"
    And log_every > 0
    When the runner produces the log file
    Then its header line is "step,time,kinetic_energy,temperature,andersen_conserved"

  # --- p_collision clamping ---

  @rq-c9865e4c
  Scenario: collision_rate · dt > 1 is clamped to p = 1 (massive Andersen)
    Given an Andersen thermostat with collision_rate=1.0e16
    And dt=1.0e-15 (so collision_rate · dt = 10)
    And a ParticleBuffers with N=4 known velocities
    When thermostat.apply_post(...) is called
    Then every particle is resampled (post-call velocities differ on every component
      with very high probability)

  @rq-1eaff437
  Scenario: collision_rate = 0 leaves velocities unchanged by the resample
    Given an Andersen thermostat with collision_rate=0
    And a ParticleBuffers with N=4 known velocities
    When thermostat.apply_post(...) is called
    Then velocities are byte-identical to the pre-call state (no resample occurred)

  # --- Determinism ---

  @rq-f062ec85
  Scenario: Two independent composed runs with identical seeds produce byte-identical outputs
    Given two complete simulations composing velocity-Verlet + Andersen with identical parameters
      (including identical seed), identical initial state, n_steps=10
    When heddlemd run is invoked on each
    Then the trajectory files are byte-identical
    And the log files are byte-identical, including the andersen_conserved column

  @rq-20daf925
  Scenario: Different seeds produce different trajectories
    Given two composed runs identical except thermostat.seed = 1 and = 2
    When heddlemd run is invoked on each
    Then the trajectory files differ

  # --- Physical correctness ---

  @rq-536457be
  Scenario: Time-averaged kinetic energy tracks (3 N − n_constraints) / 2 · k_B · T
    Given a composed runner of velocity-Verlet + Andersen with N=128 LJ particles,
      temperature=300, collision_rate=1.0e13, dt=1.0e-15, n_steps=2000, seed=1, n_constraints=0
    When the run completes
    Then the time-averaged kinetic energy over the last 1000 log rows
      is within 5% of ((3 * N − n_constraints) / 2) · k_B · 300
    (Andersen resamples every velocity component independently and does
    not conserve total momentum, so the equilibrium kinetic energy
    corresponds to 3N − n_constraints thermal DOFs rather than the
    momentum-conserving (3N − n_constraints − 3) target used by CSVR,
    Berendsen, NHC, and MTK.)

  @rq-299112e9
  Scenario: Variance of resampled velocity components matches MB
    Given a 1000-particle system with collision_rate · dt = 1 (massive)
    And dummy initial velocities (e.g. all zero)
    When thermostat.apply_post(...) is called once
    Then the sample variance of each velocity component
      is within 5% of k_B · T / m

  # --- Per-group resampling ---

  @rq-19a0f468
  Scenario: A group resamples all its atoms or none of them
    Given 512 rigid groups of 3 atoms, all velocities set to a sentinel
    And group tables offsets [0,3,6,...] and indices [0,1,2,...]
    When andersen_resample_grouped is launched with p_collision = 0.5
    Then within every group either all three atoms changed or none did
    And at least one group fired and at least one group did not

  @rq-172d6d41
  Scenario: The grouped kernel with singletons matches the per-particle kernel
    Given an N-particle unconstrained system and a singleton partition
      (offsets [0,1,2,...,N], indices [0,1,...,N-1])
    When andersen_resample_grouped and andersen_resample are each launched
      with the same seed, draw_counter, p_collision, and kt on identical inputs
    Then the post-call velocities agree byte-for-byte

  @rq-9642e7c7
  Scenario: An unconstrained monatomic run reaches its setpoint at every collision rate
    Given a Lennard-Jones argon fluid with an Andersen thermostat at T_set
    When it is run to steady state at collision_rate 1e12 and 1e14 /s
    Then the mean temperature is within 3% of T_set at both rates

  # --- Correctness under constraints ---

  @rq-5d1345d0
  Scenario: A constrained rigid-water run reaches its setpoint at every collision rate
    Given dense rigid SPC/E water under SETTLE with an Andersen thermostat at T_set
    When it is run to steady state at collision_rate 2e12, 1e13, and 5e13 /s
    Then the mean temperature is within 3% of T_set at every rate
    # A per-atom resample would run cold at low rate and approach T_set only
    # toward the massive limit; the per-group resample is rate-independent.
```
