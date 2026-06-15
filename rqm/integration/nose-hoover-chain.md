# Feature: Nosé-Hoover Chain (NHC) Thermostat <!-- rq-f606ff6f -->

The Nosé-Hoover-chain thermostat (Martyna-Klein-Tuckerman, *J. Chem.
Phys.* **97**, 2635 (1992)) is a deterministic NVT thermostat. One of
the pluggable thermostat slots (see `framework.md`); selected by
`kind = "nose-hoover-chain"` in the config's `[thermostat]` section.

The thermostat runs twice per timestep, as a symmetric Trotter
splitting around the integrator: a half-step before the integrator
(`apply_pre`) and a mirror half-step after (`apply_post`). Each
half-step combines host-side chain arithmetic with one or more
`rescale_velocities` launches. The thermostat carries no RNG state
and produces byte-identical trajectories across runs on the same
GPU.

This file also documents two shared GPU helpers — the kinetic-energy
reduction (`kinetic_energy_reduce` / `compute_kinetic_energy`) and
the uniform velocity rescale (`rescale_velocities`) — that are
re-used by the other thermostats (CSVR, Berendsen, Andersen) and by
any future slot needing the same primitives.

## Algorithm <!-- rq-c4a07fb3 -->

The Nosé-Hoover chain equations of motion couple the `3N` physical
velocities `{v_i}` to a chain of `M` auxiliary variables
`{ξ_j, p_ξ_j}_{j=1..M}` with chain masses `{Q_j}`:

```text
ẋ_i      = v_i
v̇_i      = F_i/m_i − (p_ξ_1 / Q_1) v_i
ξ̇_j      = p_ξ_j / Q_j                          (j = 1..M)
ṗ_ξ_1    = (2K − g T) − (p_ξ_2 / Q_2) p_ξ_1
ṗ_ξ_j    = (p_ξ_{j−1}² / Q_{j−1} − T)
           − (p_ξ_{j+1} / Q_{j+1}) p_ξ_j         (1 < j < M)
ṗ_ξ_M    = (p_ξ_{M−1}² / Q_{M−1} − T)
```

where `K = (1/2) Σ_i m_i |v_i|²` is the instantaneous kinetic energy,
`g = max(0, 3N − n_constraints − 3)` is the number of thermal
degrees of freedom (the three centre-of-mass modes are removed at
initial-velocity generation, and any holonomic constraints carried by
the run are also subtracted; the same convention used by CSVR and by
`compute_temperature` in `io/log-output.md`), and `T` is the
engine-side temperature value carrying `k_B · T` in Hartrees
(`k_B = 1` exactly in atomic units, so no Boltzmann factor appears).

### Chain masses <!-- rq-d0cb986d -->

```text
Q_1 = g T τ²
Q_j = T τ²        (j > 1)
```

where `τ` is the user-supplied thermostat coupling time in atomic time
units (`hbar / E_h`). The masses are precomputed once at construction
and never updated during the run.

### MKT Liouville splitting <!-- rq-312906a6 -->

A single timestep `dt` of NVT dynamics is propagated as the symmetric
product:

```text
e^(L · dt) ≈ e^(L_chain · dt/2) · e^(L_VV · dt) · e^(L_chain · dt/2)
```

where `L_VV` is the velocity-Verlet operator owned by the integrator
slot (see `velocity-verlet.md`) and `L_chain` is the chain-thermostat
operator owned by this thermostat slot. The thermostat applies the
left `e^(L_chain · dt/2)` factor in `apply_pre` and the right factor
in `apply_post`; the integrator's `step()` fires in between.

The chain operator is itself approximated by Yoshida-Suzuki
sub-stepping: each `dt/2` thermostat half-step is split into
`n_yoshida × n_resp` sub-steps with weights
`w_1, …, w_{n_yoshida}` (Suzuki, 1990; Yoshida, 1990):

```text
for stage in 0..n_yoshida:
    for resp in 0..n_resp:
        δt = w[stage] · dt / (2 · n_resp)
        chain_sub_step(δt)
```

The Suzuki-Yoshida weights are:

- `n_yoshida = 1` — `w = [1.0]`.
- `n_yoshida = 3` — `w = [w_a, 1 − 2 w_a, w_a]` with
  `w_a = 1 / (2 − 2^{1/3}) ≈ 1.351207...`.
- `n_yoshida = 5` — `w = [w_b, w_b, 1 − 4 w_b, w_b, w_b]` with
  `w_b = 1 / (4 − 4^{1/3}) ≈ 0.414490...`.
- `n_yoshida = 7` — the standard 7-point Suzuki sequence
  `w = [0.7845136104775573, 0.2355732133593582, −1.1776799841788710,
        1.3151863206839023, −1.1776799841788710, 0.2355732133593582,
        0.7845136104775573]` (Yoshida, 1990).

Any other `n_yoshida` is rejected at config-load time.

### Chain sub-step <!-- rq-cc34e758 -->

Each Yoshida sub-step of length `δt` consists of:

1. **Outermost chain-momentum kick** (one direction):
   `p_ξ_M ← p_ξ_M + (δt/4) · (p_ξ_{M−1}²/Q_{M−1} − T)`
   *(omitted when `M = 1`)*.

2. **Inner chain-momentum updates, M−1 → 1**:
   for `j` from `M−1` down to `1`:
     `s = exp(−(δt/8) · p_ξ_{j+1}/Q_{j+1})`  *(or `s = 1` when `j = M`)*
     `p_ξ_j ← p_ξ_j · s`
     if `j == 1`:
       `p_ξ_1 ← p_ξ_1 + (δt/4) · (2K − g T)`
     else:
       `p_ξ_j ← p_ξ_j + (δt/4) · (p_ξ_{j−1}²/Q_{j−1} − T)`
     `p_ξ_j ← p_ξ_j · s`

3. **Particle velocity rescale**:
   `factor = exp(−(δt/2) · p_ξ_1 / Q_1)`
   `v_i ← v_i · factor`   (for every particle `i` and axis)
   `K ← K · factor²`

4. **Chain position update**:
   `ξ_j ← ξ_j + (δt/2) · p_ξ_j / Q_j`   (for every `j`).

5. **Inner chain-momentum updates, 1 → M−1**:
   for `j` from `1` to `M−1`:
     `s = exp(−(δt/8) · p_ξ_{j+1}/Q_{j+1})`
     `p_ξ_j ← p_ξ_j · s`
     if `j == 1`:
       `p_ξ_1 ← p_ξ_1 + (δt/4) · (2K − g T)`
     else:
       `p_ξ_j ← p_ξ_j + (δt/4) · (p_ξ_{j−1}²/Q_{j−1} − T)`
     `p_ξ_j ← p_ξ_j · s`

6. **Outermost chain-momentum kick** (closing direction):
   `p_ξ_M ← p_ξ_M + (δt/4) · (p_ξ_{M−1}²/Q_{M−1} − T)`
   *(omitted when `M = 1`)*.

Steps 1–6 are the standard MKT symmetric sub-step (Tuckerman, *Statistical
Mechanics: Theory and Molecular Simulation*, §4.10). The full half-step
applies this sub-step `n_yoshida × n_resp` times. The kinetic energy `K`
is updated host-side after each velocity rescale by multiplying by
`factor²`; the host therefore does not re-launch the kinetic-energy
reduction within a single half-step.

## Per-Step Kernel Sequence <!-- rq-f45cdfb6 -->

Per timestep the NHC thermostat fires two half-steps around the
integrator's `step()`. Both halves have the same shape:

| Hook        | Step          | Kernel / call                             | Operation                                                  | Stage label              |
| ----------- | ------------- | ----------------------------------------- | ---------------------------------------------------------- | ------------------------ |
| `apply_pre` | KE reduce     | `kinetic_energy_reduce`                   | one f32 scalar of `K = ½ Σ m_i \|v_i\|²`                   | `KineticEnergyReduce`    |
| `apply_pre` | Thermostat ½  | host chain + `rescale_velocities` × N_sub | NHC half-step, `N_sub = n_yoshida · n_resp` rescales       | `NhcRescaleVelocities`   |
| `apply_post`| KE reduce     | `kinetic_energy_reduce`                   | refresh `K` after the integrator's VV step                 | `KineticEnergyReduce`    |
| `apply_post`| Thermostat ½  | host chain + `rescale_velocities` × N_sub | NHC half-step (mirror of `apply_pre`)                      | `NhcRescaleVelocities`   |

The integrator's own kernels (`vv_kick_drift`, `vv_kick`, the force
pipeline) are launched separately by `integrator.step()` and are not
part of this slot's per-step sequence.

The total per-step launch count owned by this thermostat is
`2 + 2 · N_sub`. For the recommended defaults `M = 3, n_yoshida = 3,
n_resp = 1`, that is `2 + 6 = 8` thermostat-related launches per step.

## Parameters <!-- rq-d6cf8e86 -->

The matching builder deserialises a typed `NoseHooverChainParams` from the `[thermostat]` section's `SlotConfig::params` (see `framework.md`); the per-field reference below documents that parameter struct:

- `temperature: f64` — bath temperature `T` as `k_B · T` in Hartrees
  (the engine's internal temperature representation; `k_B = 1`).
  Required. Finite
  and strictly positive. Independent of `simulation.temperature`, which
  governs the initial-velocity sampler.
- `tau: f64` — thermostat coupling time in atomic time units
  (`hbar / E_h`). Required. Finite
  and strictly positive. Typical values for liquid water are 50–100 fs
  (≈ 10× the natural OH stretching period). Smaller `τ` couples the
  thermostat more strongly to the physical system and dampens kinetic
  fluctuations harder; larger `τ` leaves the dynamics closer to NVE.
- `chain_length: u32` — number of chain elements `M`. Optional;
  defaults to `3`. Must be `≥ 1`. `M = 1` reduces to vanilla
  Nosé-Hoover. `M ≥ 2` restores ergodicity for stiff systems.
- `yoshida_order: u32` — number of Yoshida-Suzuki sub-steps per chain
  half-step. Optional; defaults to `3`. Accepted values: `1`, `3`, `5`,
  `7`. Any other value is rejected at config-load time.
- `n_resp: u32` — number of times each Yoshida sub-step is repeated
  (the "RESP" sub-cycle count). Optional; defaults to `1`. Must be
  `≥ 1`. Larger values divide the chain step into smaller integration
  intervals; useful when the chain modes are unusually stiff relative
  to `dt`.

## Chain state <!-- rq-71f469ae -->

Host-side state on `NoseHooverChainThermostat`:

- `xi: Vec<f64>` — length `M`, chain positions. Initialised to `0.0`
  at construction.
- `p_xi: Vec<f64>` — length `M`, chain momenta. Initialised to `0.0`
  at construction.
- `q_mass: Vec<f64>` — length `M`, precomputed chain masses. Element
  `0` is `g · T · τ²`; elements `1..M` are `T · τ²` (with `k_B = 1`
  and `T` carrying `k_B · T` in atomic units).
- `g_dof: u32` — degrees of freedom thermostatted by the chain,
  computed at construction as
  `max(0, 3 · particle_count − n_constraints − 3)` from the
  `n_constraints` parameter passed by the runner. The
  centre-of-mass-removed initial-velocity convention (see
  `simulation-runner.md`) keeps the COM momentum exactly zero under
  NHC's uniform velocity rescaling, and `n_constraints` accounts for
  the holonomic constraints (e.g. SETTLE) that project out
  intramolecular velocity components. Stored once at construction
  and never updated.

All chain arithmetic runs in `f64` on the host. The chain state is
read and written exclusively inside `NoseHooverChainThermostat`'s
`apply_pre` and `apply_post` methods and in `log_column_values`
(read-only); no other code path touches it.

## NHC conserved Hamiltonian <!-- rq-6bd0b42f -->

The extended Hamiltonian conserved by the NHC equations is

```text
H' = K + U
     + Σ_{j=1..M} p_ξ_j² / (2 Q_j)
     + g T · ξ_1
     + T · Σ_{j=2..M} ξ_j
```

where `K` is the instantaneous kinetic energy and `U` is the total
potential energy. `H'` is invariant under the exact NHC dynamics; in
the discretised MKT splitting it drifts by `O(dt²)` per step. Drift
in `H'` over a run is the canonical correctness diagnostic for an
NHC implementation.

`H'` is exposed as a per-log-row diagnostic column named
`nhc_conserved` when NHC is the configured thermostat (see
`io/log-output.md`). The thermostat computes its chain term
(`Σ p_ξ²/(2Q) + g T ξ_1 + T Σ_{j>1} ξ_j`) from its host-side
state and combines it with the kinetic and potential energies supplied
by the runner at log-write time.

## Empty State and degenerate cases <!-- rq-fd90fab3 -->

- `particle_count == 0`: both `apply_pre` and `apply_post` return
  `Ok(())` without launching any kernel. The chain state arrays are
  allocated with `M` elements regardless (since `M` is a config-time
  constant); the `g_dof` is `0`.
- `particle_count == 1`: `g_dof = 0`. The chain still propagates but
  drives the kinetic energy toward zero (the target `g T` is
  zero). This is the mathematically correct behaviour for a system
  with zero thermal degrees of freedom; users should not run NHC on a
  one-particle system but the thermostat does not refuse to construct.
- `M == 1`: vanilla Nosé-Hoover. The "outermost chain-momentum kick"
  steps 1 and 6 in the sub-step are skipped (no `Q_{M-1}` exists);
  the M−1 → 1 and 1 → M−1 inner-update loops both reduce to a single
  iteration. Documented here so callers can use `M = 1` as a
  diagnostic mode without surprises.

## Feature API <!-- rq-a487d90c -->

### Types <!-- rq-3a56a657 -->

- `NoseHooverChainThermostat` — implements the `Thermostat` trait <!-- rq-62e2bef5 -->
  declared in `framework.md`. Registered in
  `ThermostatRegistry::with_builtins` under
  `kind_name() == "nose-hoover-chain"`. Fields:

  - `device: Arc<CudaDevice>`
  - `temperature: f64` — `T`, copied from the config.
  - `tau: f64` — coupling time, copied from the config.
  - `chain_length: u32` — `M`.
  - `yoshida_order: u32` — `n_yoshida`.
  - `n_resp: u32` — `n_resp`.
  - `yoshida_weights: &'static [f64]` — pre-resolved Suzuki-Yoshida
    weights for the chosen `yoshida_order`.
  - `g_dof: u32` — `max(0, 3 · particle_count − n_constraints − 3)`,
    computed at construction from the `n_constraints` parameter
    passed by the runner.
  - `q_mass: Vec<f64>` — chain masses, length `M`.
  - `xi: Vec<f64>` — chain positions, length `M`.
  - `p_xi: Vec<f64>` — chain momenta, length `M`.
  - `ke_scratch: CudaSlice<f32>` — length-1 device buffer that holds
    the kinetic energy reduction's output; reused across calls.
  - `most_recent_ke: f64` — last kinetic energy computed during the
    current `apply_post`, accurate at the end of `apply_post`. Used by
    `log_column_values` to avoid a redundant download.
  - `particle_count: usize`

  All fields private; the slot's public surface is the `Thermostat`
  trait methods (see `framework.md`) and the construction via
  `NoseHooverChainBuilder`.

- `NoseHooverChainBuilder` — implements `ThermostatBuilder` with <!-- rq-4bd6ff2b -->
  `kind_name() == "nose-hoover-chain"`. `build(device, particle_count,
  kind)` deserialises `NoseHooverChainParams` from `params`,
  allocates the chain state, precomputes `q_mass` and the
  Suzuki-Yoshida weights, allocates the length-1 `ke_scratch` device
  buffer, and returns the boxed `NoseHooverChainThermostat`.

### `Thermostat` trait overrides <!-- rq-b47589c9 -->

`NoseHooverChainThermostat` overrides every method on the
`Thermostat` trait declared in `framework.md`:

- `apply_pre(buffers, dt, timings)` — runs the left <!-- rq-a9c46f51 -->
  `e^(L_chain · dt/2)` factor: one `kinetic_energy_reduce` launch
  followed by `N_sub` chain sub-steps, each of which performs the
  host-side MKT chain math and one `rescale_velocities` launch.
- `apply_post(buffers, dt, timings)` — runs the right <!-- rq-370bf3a8 -->
  `e^(L_chain · dt/2)` factor: same shape as `apply_pre`.
- `log_column_names() -> &'static ["nhc_conserved"]`. <!-- rq-8a571737 -->
- `log_column_values(ke, pe) -> vec![H']` where `H'` follows the <!-- rq-f94f6bac -->
  formula in *NHC conserved Hamiltonian* above, with `K = ke`,
  `U = pe`, and the chain term computed from `xi`, `p_xi`, `q_mass`,
  `g_dof`, `temperature`.

### CUDA Kernels <!-- rq-a4eb7957 -->

`kernels/nose_hoover.cu` declares two `extern "C"` kernels:

```c
extern "C" __global__ void kinetic_energy_reduce(
    const float *velocities_x, const float *velocities_y, const float *velocities_z,
    const float *masses,
    float *partial,      // length blockDim.x in shared mem; single f32 output written by thread 0
    unsigned int n);

extern "C" __global__ void rescale_velocities(
    float *velocities_x, float *velocities_y, float *velocities_z,
    float factor,
    unsigned int n);
```

#### `kinetic_energy_reduce` <!-- rq-1727d6bd -->

A single-block kernel with `blockDim.x = 256`. Each thread loops over
its strided subset of particles
(`for i in threadIdx.x..n step blockDim.x`), accumulating
`½ · m_i · (v_xi² + v_yi² + v_zi²)` into a register. The per-thread
partials are then summed across the block via a deterministic
left-to-right pairwise reduction in shared memory:

```c
__shared__ float partial[256];
partial[threadIdx.x] = thread_register_sum;
__syncthreads();
for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    if (threadIdx.x % (2 * stride) == 0) {
        partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
}
if (threadIdx.x == 0) partial_out[0] = partial[0];
```

The strided per-thread accumulation order (thread `t` sees particles
`t, t + 256, t + 512, …`) is fixed by `n` and `blockDim.x`, and the
shared-memory reduction tree is fully symmetric, so the kernel returns
byte-identical `f32` results for identical inputs on the same GPU.
Single-block execution underutilises the GPU for very large `n` but
keeps the determinism analysis trivial; the cost is negligible
relative to the force-pipeline launches.

The output is a length-1 `CudaSlice<f32>` (held on
`NoseHooverChainThermostat.ke_scratch` for this slot, or on the
analogous field of any other thermostat that uses the helper) which
the host downloads via `dtoh_sync_copy_into` and promotes to `f64`
before the chain math.

#### `rescale_velocities` <!-- rq-a20bd6f3 -->

One thread per particle. Thread `i`:

```c
velocities_x[i] *= factor;
velocities_y[i] *= factor;
velocities_z[i] *= factor;
```

No interaction between threads; trivially deterministic. Block size
256, grid `ceil(n / 256)`.

### PTX Module Loading <!-- rq-c465b5f4 -->

`init_device()` loads the compiled `kernels/nose_hoover.cu` PTX as
module `"nose_hoover"` and captures `kinetic_energy_reduce` and
`rescale_velocities` into the `Kernels` handle (see
`build-pipeline.md`).

### Rust Launch Helpers <!-- rq-d028bd1f -->

Three free functions in `src/gpu/kernels.rs`, re-exported from
`crate::gpu`:

- `compute_kinetic_energy(buffers: &ParticleBuffers, scratch: &mut CudaSlice<f32>) -> Result<f32, GpuError>` <!-- rq-511f4606 -->
  - Launches `kinetic_energy_reduce` over `buffers.velocities_*` and
    `buffers.masses` with output `scratch` (a length-1 device buffer
    the caller owns; reused across calls to avoid per-step
    allocation).
  - Downloads `scratch[0]` host-side via `dtoh_sync_copy_into` and
    returns the value.
  - Block size 256, single block, no shared-memory tuning beyond what
    the kernel declares.
  - When `buffers.particle_count() == 0`, returns `Ok(0.0_f32)`
    without launching.
  - Invokes the kernel through the `Kernels` handle reached from
    `buffers`; performs no string-keyed kernel lookup of its own (see
    `build-pipeline.md`).
  - General-purpose: usable by every thermostat in the registry and
    by any future slot that needs an instantaneous kinetic energy on
    the host without downloading the full velocity state.

- `rescale_velocities(buffers: &mut ParticleBuffers, factor: f32) -> Result<(), GpuError>` <!-- rq-09e04194 -->
  - Launches `rescale_velocities` over `buffers.velocities_*`.
  - Block size 256; grid `ceil(n / 256)`.
  - When `buffers.particle_count() == 0`, returns `Ok(())` without
    launching.
  - Invokes the kernel through the `Kernels` handle, like
    `compute_kinetic_energy`.
  - General-purpose: usable by any thermostat that needs a uniform
    scalar velocity rescale.

- `compute_total_potential_energy(buffers: &ParticleBuffers, scratch: &mut CudaSlice<f32>) -> Result<f32, GpuError>` <!-- rq-fc6859df -->
  - Launches `virial_sum_reduce` (the generic single-block deterministic
    f32 sum-reduction kernel declared in `kernels/barostat.cu` and
    documented in `berendsen-barostat.md`) over
    `buffers.potential_energies`, with output `scratch` (a length-1
    device buffer the caller owns; reused across calls to avoid per-step
    allocation).
  - Downloads `scratch[0]` host-side via `dtoh_sync_copy_into` and
    returns the value in Hartrees.
  - Block size 256, single block, no shared-memory tuning beyond what
    the kernel declares. Tracked under
    `KernelStage::POTENTIAL_ENERGY_REDUCE` (distinct from
    `KernelStage::VIRIAL_SUM_REDUCE`, which counts the barostat-driven
    launches of the same kernel binary).
  - When `buffers.particle_count() == 0`, returns `Ok(0.0_f32)`
    without launching.
  - Invokes the kernel through the `Kernels` handle reached from
    `buffers`; performs no string-keyed kernel lookup of its own.
  - General-purpose: the runner calls it whenever it needs the
    instantaneous total potential energy for a slot's
    `log_column_values(ke, pe)` invocation (currently
    `NoseHooverChainThermostat::log_column_values` and
    `MtkNptIntegrator::log_column_values`); any future slot that
    declares a PE-using diagnostic column observes the same value
    through the same path.

### Shared `nhc_chain_sub_step` host-side helper <!-- rq-19496703 -->

The chain sub-step described under *Chain sub-step* above is exposed
as a pure host-side helper function in `src/integrator/nose_hoover_chain.rs`
so that slots that need a parallel NHC chain on a different DOF can
reuse the same algorithm rather than duplicating it. The
`NoseHooverChainThermostat` slot is one caller; the `mtk-npt`
integrator (`mtk-npt.md`) is the other.

```rust
pub fn nhc_chain_sub_step(
    xi: &mut [f64],
    p_xi: &mut [f64],
    q_mass: &[f64],
    dt: f64,
    k_thermalized: f64,    // 2K for the particle chain; p_eps²/W for an MTK cell chain
    g_dof: f64,            // N_f for the particle chain; 1.0 for a 1-DOF chain
    kt: f64,
) -> f64
```

- Mutates `xi` and `p_xi` in place; reads `q_mass` and the four
  scalar inputs.
- Returns the multiplicative rescale factor that the caller must
  apply to the thermalized DOF (the particle chain feeds it to
  `rescale_velocities`; the MTK cell chain multiplies `p_eps` by it
  host-side).
- Slice lengths `xi.len() == p_xi.len() == q_mass.len() == M` are
  required; the helper handles `M = 1` (no outermost-chain-momentum
  kicks) and `M ≥ 2` uniformly.
- Pure function in the f64 IEEE-754 sense: deterministic, no I/O,
  no allocation, no panics under finite inputs.

## Launch Configuration <!-- rq-aff3dafa -->

- `kinetic_energy_reduce`: block size 256, single block (grid 1).
  Shared memory: `256 * sizeof(float) = 1024 B`.
- `rescale_velocities`: block size 256, grid `ceil(n / 256)`. Shared
  memory: 0 bytes.
- Stream: the default stream carried by `ParticleBuffers::device`.

## Determinism <!-- rq-98473586 -->

- All NHC chain arithmetic runs in `f64` on the host with a fixed
  arithmetic sequence (see *Algorithm*). Two runs on the same machine
  produce byte-identical chain state per step.
- The two device-side kernels are deterministic by construction (see
  *CUDA Kernels*).
- The thermostat carries no RNG; there is no per-call stochastic
  draw to seed.
- Two end-to-end runs composing the same integrator with NHC on the
  same GPU with identical configs and identical initial particle
  state produce byte-identical trajectory and log files, including
  the `nhc_conserved` column.

## Out of Scope <!-- rq-61528c78 -->

- A lossless `(f32, f64)` compensated mode (cf. velocity-Verlet's
  `lossless` flag). The chain's `exp()` and per-sub-step velocity
  rescale operations do not have a clean compensated form; bit-exact
  time-reversibility under NHC is not a property the algorithm
  promises.
- Massive thermostatting (one independent chain per atom). Single
  global chain only.
- Pressure coupling (Martyna-Tobias-Klein NPT extension). A future
  barostat would slot in alongside NHC via the `[barostat]` slot,
  possibly sharing the chain primitive.
- Constraints (SHAKE/RATTLE). The trait shape supports them; no
  constrained-NHC thermostat ships.
- User-overrideable `g` (degrees of freedom). The thermostat
  hard-codes `g = max(0, 3N − n_constraints − 3)`, the
  COM-removed and constraint-aware convention shared with CSVR
  (`csvr.md`) and `compute_temperature` (`io/log-output.md`).
- Multi-step velocity-rescale buffering. Each Yoshida sub-step's
  velocity rescale launches its own `rescale_velocities` kernel; no
  attempt is made to fold consecutive rescales into a single launch
  because the chain's `p_ξ_1` depends on the freshly-rescaled
  kinetic energy between sub-steps.
- Sub-stepping of the integrator's velocity-Verlet portion (RESPA
  over the inner forces). The chain RESP factor `n_resp` only
  sub-steps the chain itself.
- A `nose-hoover` (M=1) variant under a distinct `kind` name. M=1 is
  exercised through `chain_length = 1` under the existing
  `nose-hoover-chain` slot.
- Cross-thermostat initial-chain-state seeding (e.g. running a short
  NHC warm-up before switching to NHC-NPT). The thermostat is fixed
  at construction; future restart-from-checkpoint flows expose
  `xi`/`p_xi` as public fields so they can be restored explicitly.

---

## Gherkin Scenarios <!-- rq-8b1827a9 -->

```gherkin
Feature: Nosé-Hoover chain (NHC) thermostat

  Background:
    Given a CUDA-capable GPU available as device 0
    And a SimulationBox with lx=ly=lz=1.0e-9 unless otherwise specified
    And init_device() has been called

  # --- Module loading and construction ---

  @rq-572a0431
  Scenario: init_device exposes the NHC kernels on the Kernels handle
    When init_device() is called
    Then the returned GpuContext's kernels handle exposes the kinetic_energy_reduce function
    And the kernels handle exposes the rescale_velocities function

  @rq-b43bf21c
  Scenario: Construct NoseHooverChainThermostat with default chain parameters
    Given a ThermostatKind::NoseHooverChain {
      temperature: 300.0, tau: 1.0e-13,
      chain_length: 3, yoshida_order: 3, n_resp: 1 }
    When registry.build_optional(Some(&kind), device, particle_count=4, n_constraints=0) is called
    Then it returns Ok(Some(thermostat))
    And the underlying NoseHooverChainThermostat has chain_length=3
    And xi equals [0.0, 0.0, 0.0]
    And p_xi equals [0.0, 0.0, 0.0]
    And q_mass[0] equals g_dof * T_atomic * tau² (the temperature parameter
      supplied to the kind, no Boltzmann factor)
    And q_mass[1] equals T_atomic * tau²
    And q_mass[2] equals T_atomic * tau²
    And g_dof equals 9 (3 * 4 − 0 − 3)

  @rq-1aa67999
  Scenario: Construct for a SETTLE'd water system
    Given an NHC kind with chain_length=3, temperature=300, tau=1e-13
    When registry.build_optional(Some(&kind), device, particle_count=24, n_constraints=24) is called
    Then it returns Ok(Some(thermostat))
    And g_dof equals 45 (3 * 24 − 24 − 3)
    And q_mass[0] equals 45 * T_atomic * tau²

  @rq-12d7c3fe
  Scenario: Construct with chain_length = 1 reduces to vanilla Nosé-Hoover
    Given a ThermostatKind::NoseHooverChain { temperature: 300, tau: 1e-13,
      chain_length: 1, yoshida_order: 3, n_resp: 1 }
    When registry.build_optional(Some(&kind), device, particle_count=4, n_constraints=0) is called
    Then it returns Ok(Some(thermostat))
    And xi has length 1
    And p_xi has length 1
    And q_mass has length 1

  @rq-5f21bfd8
  Scenario: Construct with particle_count = 0
    Given an NHC kind with chain_length=3
    When registry.build_optional(Some(&kind), device, particle_count=0, n_constraints=0) is called
    Then it returns Ok(Some(thermostat))
    And g_dof equals 0
    And the ke_scratch device buffer has length 1

  @rq-6dc8454d
  Scenario: Reject yoshida_order outside {1, 3, 5, 7}
    Given a config with [thermostat].kind="nose-hoover-chain" and yoshida_order=2
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "thermostat.yoshida_order", reason: _ })

  @rq-811c598f
  Scenario: Reject chain_length = 0
    Given a config with [thermostat].kind="nose-hoover-chain" and chain_length=0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "thermostat.chain_length", reason: _ })

  @rq-dd6fe266
  Scenario: Reject n_resp = 0
    Given a config with [thermostat].kind="nose-hoover-chain" and n_resp=0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "thermostat.n_resp", reason: _ })

  @rq-e5b63a73
  Scenario: Reject non-positive temperature
    Given a config with [thermostat].kind="nose-hoover-chain" and temperature=0.0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "thermostat.temperature", reason: _ })

  @rq-d532de58
  Scenario: Reject non-positive tau
    Given a config with [thermostat].kind="nose-hoover-chain" and tau=-1.0e-13
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "thermostat.tau", reason: _ })

  # --- kinetic_energy_reduce ---

  @rq-25e0208d
  Scenario: kinetic_energy_reduce of a single particle at v=0 is zero
    Given a ParticleBuffers from a single particle at v=(0,0,0), m=1.0
    When compute_kinetic_energy(&buffers, &mut scratch) is called
    Then it returns Ok(0.0_f32)

  @rq-74e42489
  Scenario: kinetic_energy_reduce matches the host-side formula on small N
    Given a ParticleBuffers from N=4 particles with non-trivial masses and velocities
    When compute_kinetic_energy(&buffers, &mut scratch) is called
    Then the returned value equals 0.5 * Σ_i m_i (vx_i² + vy_i² + vz_i²) within f32 round-off

  @rq-5c197a37
  Scenario: kinetic_energy_reduce is deterministic
    Given two ParticleBuffers built from byte-identical ParticleStates of N=1000
    When compute_kinetic_energy is called on each with its own scratch buffer
    Then the two returned values agree byte-for-byte

  @rq-96f71d13
  Scenario: kinetic_energy_reduce on empty state returns 0.0 without launching
    Given a ParticleBuffers with particle_count() == 0
    When compute_kinetic_energy(&buffers, &mut scratch) is called
    Then it returns Ok(0.0_f32)

  # --- rescale_velocities ---

  @rq-6966fd4f
  Scenario: rescale_velocities multiplies every velocity component by the factor
    Given a ParticleBuffers from N=2 particles with v0=(1,2,3) and v1=(-4,5,-6)
    When rescale_velocities(&mut buffers, factor=0.5) is called
    And the buffers are downloaded
    Then velocities_x equals [0.5, -2.0]
    And velocities_y equals [1.0, 2.5]
    And velocities_z equals [1.5, -3.0]

  @rq-5c799ac6
  Scenario: rescale_velocities does not modify positions, masses, or forces
    Given a ParticleBuffers with N=4 nonzero values
    And a snapshot of positions, masses, forces
    When rescale_velocities(&mut buffers, factor=0.7) is called
    And the buffers are downloaded
    Then positions_*, masses, forces_* are byte-identical to the snapshot

  @rq-393a7932
  Scenario: rescale_velocities with factor=1.0 is the identity
    Given a ParticleBuffers from N=4 nonzero velocities
    And a snapshot of velocities
    When rescale_velocities(&mut buffers, factor=1.0) is called
    And the buffers are downloaded
    Then velocities are byte-identical to the snapshot

  @rq-bef900e1
  Scenario: rescale_velocities on empty state is a no-op
    Given a ParticleBuffers with particle_count() == 0
    When rescale_velocities(&mut buffers, factor=0.5) is called
    Then it returns Ok(())

  # --- Slot integration: per-step kernel sequence ---

  @rq-76069102
  Scenario: apply_pre and apply_post launch the expected kernel calls for default chain parameters
    Given an NHC thermostat with chain_length=3, yoshida_order=3, n_resp=1, particle_count=4
    And buffers prepared with non-zero velocities
    When thermostat.apply_pre(&mut buffers, dt=1e-15, &mut timings) is called
    Then KernelStage::KINETIC_ENERGY_REDUCE has count == 1
    And KernelStage::NHC_RESCALE_VELOCITIES has count == 3  (3 Yoshida × 1 RESP × 1 half)
    When thermostat.apply_post(&mut buffers, dt=1e-15, &mut timings) is called
    Then KernelStage::KINETIC_ENERGY_REDUCE has total count == 2  (one per half)
    And KernelStage::NHC_RESCALE_VELOCITIES has total count == 6  (3 + 3)

  @rq-e9a5474f
  Scenario: apply_pre on empty NHC state is a no-op
    Given an NHC thermostat with particle_count=0
    When thermostat.apply_pre(...) is called
    Then it returns Ok(())

  @rq-9b3e0e89
  Scenario: apply_post on empty NHC state is a no-op
    Given an NHC thermostat with particle_count=0
    When thermostat.apply_post(...) is called
    Then it returns Ok(())

  # --- Integration with the runner / log column ---

  @rq-17d3ddfe
  Scenario: log_column_names returns ["nhc_conserved"] for NHC
    Given a constructed NoseHooverChainThermostat
    Then state.log_column_names() equals ["nhc_conserved"]

  @rq-7909b92c
  Scenario: log_column_names returns empty for integrators (VV / Langevin)
    Given a constructed VelocityVerletState
    Then state.log_column_names() equals []
    Given a constructed LangevinBaoabState
    Then state.log_column_names() equals []

  @rq-07a18814
  Scenario: log_column_values returns the conserved Hamiltonian
    Given an NHC thermostat with chain_length=2, temperature=300, tau=1e-13, g_dof=9,
      xi=[0.1, 0.2], p_xi=[0.5, -0.3]
    When state.log_column_values(ke=1.0e-3, pe=2.0e-3) is called (engine-
      side Hartrees)
    Then it returns a Vec with one entry equal to
      1.0e-3 + 2.0e-3
      + 0.5² / (2 * q_mass[0]) + (-0.3)² / (2 * q_mass[1])
      + 9 * T_atomic * 0.1
      + T_atomic * 0.2
      (with `k_B = 1`; no Boltzmann factor) within f64 round-off

  @rq-a16c37bd
  Scenario: Log file header includes nhc_conserved when NHC is the thermostat
    Given a config with [thermostat].kind="nose-hoover-chain"
    And log_every > 0
    When the runner produces the log file
    Then its header line is "step,time,kinetic_energy,temperature,nhc_conserved"

  @rq-ded81a4a
  Scenario: Log file header omits nhc_conserved when no thermostat is configured
    Given a config with [integrator].kind="velocity-verlet" and [thermostat] omitted
    And log_every > 0
    When the runner produces the log file
    Then its header line is "step,time,kinetic_energy,temperature"

  # --- Determinism across runs ---

  @rq-6faf6fba
  Scenario: Two independent composed runs with identical configs produce byte-identical outputs
    Given two complete simulations composing velocity-Verlet + NHC with identical parameters,
      identical initial particle state, n_steps=10
    When heddlemd run is invoked on each
    Then the two trajectory files are byte-identical
    And the two log files are byte-identical, including the nhc_conserved column

  @rq-6a4016ac
  Scenario: COM momentum is preserved under NHC velocity rescaling
    Given a composed runner of velocity-Verlet + NHC with initial COM momentum = 0
      (the runner's standard setup) and n_steps=100
    When the run completes
    Then Σ_i m_i v_i evaluated on the final velocities is zero within f32 round-off
```
