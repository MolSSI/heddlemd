# Feature: Smooth Particle-Mesh Ewald (SPME) <!-- rq-202493a5 -->

Smooth particle-mesh Ewald is the long-range electrostatics method
configured by the `[spme]` table in the simulation config. The SPME
algorithm partitions the Coulomb energy `U_total = k_C · Σ_{i<j} q_i q_j / r_ij`
into three contributions:

```
U_total = U_real + U_reciprocal − U_self
```

where
- `U_real` is a short-range pairwise sum with screening `erfc(α · r)`,
  evaluated on the shared neighbor list,
- `U_reciprocal` is a long-range smooth sum evaluated on a 3D FFT grid,
- `U_self = k_C · (α/√π) · Σ_i q_i²` corrects for each particle's
  self-interaction introduced by the charge spreading step.

`α` is the Ewald splitting parameter; it controls the partitioning of
work between real and reciprocal space. `k_C = 1/(4π ε₀) ≈ 8.987 551 787 × 10⁹
N·m²/C²` is the Coulomb constant (rounded to `f32`).

See `docs/long-range-electrostatics.md` for the architectural overview.
This file specifies the configuration interface, the two `Potential` slots
that implement SPME (the real-space `erfc` slot and the reciprocal-space
spread → FFT → multiply → IFFT → gather pipeline), the CUDA kernels they
own, and the cuFFT-determinism precondition validated at device-init time.

## Slot structure <!-- rq-3e2bcb37 -->

SPME contributes two `Potential` slots to the `ForceField`:

- `SpmeRealSpaceState` — a pair-force slot using `erfc(α · r) / r`
  screening over the shared neighbor list. Structurally similar to the
  truncated Coulomb slot (`coulomb-pair-force.md`); differs only in the
  functional form of the pair force.
- `SpmeReciprocalState` — owns the FFT grid buffers, the cuFFT plan, the
  precomputed influence function, and a dedicated bin-only cell-list
  used by the spread and gather kernels.

Both slots share the per-particle `charges` buffer on `ParticleBuffers`
(see `particle-state.md`) and the shared `DeviceExclusionList` (see
`topology.md`). The two slots are constructed together when `[spme]` is
present in the config; they share the parsed `alpha` and per-particle
charges but are otherwise independent.

The `[spme]` and `[coulomb]` tables are mutually exclusive in the config
(see `io/config-schema.md`).

## Parameters <!-- rq-7bd2d9ca -->

The `[spme]` table parses into a `SpmeConfig` carried as
`Config::spme: Option<SpmeConfig>` (see `io/config-schema.md`). Required
fields:

- `alpha: f64` — Ewald splitting parameter in inverse Bohr (`1/a_0`).
  Finite, strictly positive.
- `r_cut_real: f64` — real-space cutoff in Bohr (`a_0`). Finite,
  strictly positive.
- `grid: [u32; 3]` — FFT grid dimensions, in the lattice-direction order
  `[n_a, n_b, n_c]`. Each component must satisfy
  `n_d >= 2 · spline_order`.
- `spline_order: u32` — B-spline interpolation order. Accepted values
  are `4`, `5`, `6`, `7`, `8`. Defaults to `4` when omitted.

The schema description in `io/config-schema.md` documents the
recommended parameter relationship (`α · r_cut_real ≈ 3.5` for typical
accuracy targets; grid spacing `~1 Å` per direction) but the loader
performs no auto-derivation: every field except `spline_order` is
required when the table is present.

## Real-space slot <!-- rq-f6d45062 -->

The real-space slot is structurally analogous to `coulomb-pair-force.md`
but evaluates `erfc(α · r) / r` instead of `1/r`. The slot uses the
shared `NeighborListState` owned by `ForceField`, the per-particle
`charges` buffer, and the shared `DeviceExclusionList`'s
`atom_excl_coul_scales` array.

### Algorithm <!-- rq-39b05bc9 -->

For each ordered `(i, k)` with `0 <= i < N` and `0 <= k <
neighbor_counts[i]`:

1. The pair-buffer slot is `slot = i * max_neighbors + k`.
2. Read `j = neighbor_list[slot]`. If `i == j` or
   `k >= neighbor_counts[i]`, write `0.0_f32` to all five pair-buffer
   slots and stop.
3. Compute the displacement `(dx, dy, dz) = positions[i] − positions[j]`
   and apply the triclinic minimum-image algorithm of `simulation-box.md`.
4. Compute `r² = dx² + dy² + dz²`. If `r² > r_cut_real²`, write zeros to
   all five slots and stop.
5. Read `q_i = charges[i]`, `q_j = charges[j]`.
6. Compute the screened Coulomb factor and energy:

   ```text
   inv_r2  = 1.0f / r2
   inv_r   = sqrtf(inv_r2)
   r       = sqrtf(r2)
   qq      = q_i * q_j
   ar      = α * r
   erfc_ar = erfcf(ar)
   gauss   = expf(-(ar * ar))
   energy  = k_C * qq * erfc_ar * inv_r
   factor  = k_C * qq * inv_r * (erfc_ar * inv_r2
                                 + (2.0f * α / sqrtf(π)) * gauss * inv_r2)
   ```

   `factor · r_ij` is the screened-Coulomb force on particle `i` due to
   `j`. The derivative form combines the `1/r²` decay of `erfc(αr)/r` with
   the Gaussian term from `d(erfc)/dr = −(2α/√π) · exp(−α²r²)`.

7. Apply the per-pair Coulomb exclusion scale (see `topology.md`):
   `scale = exclusion_scale(i, j, atom_excl_offsets, atom_excl_partners,
   atom_excl_coul_scales)`. Multiply `factor`, `energy`, and the scalar
   virial `w = factor · r²` by `scale`.
8. Write the final values to the pair buffer using the half-sum
   convention:

   ```text
   pair_forces_x[slot] = factor * dx
   pair_forces_y[slot] = factor * dy
   pair_forces_z[slot] = factor * dz
   pair_energies[slot] = 0.5 * energy
   pair_virials[slot]  = 0.5 * w
   ```

The real-space slot does not apply a switching function. The `erfc`
factor decays rapidly enough that a hard cutoff is acceptable when
`α · r_cut_real >= 3.5` (the loader does not enforce this; it is a
user-tuning concern documented in `io/config-schema.md`).

### Real-space CUDA kernel <!-- rq-9a512ed1 -->

`kernels/spme_real.cu` declares one `extern "C"` kernel:

```c
extern "C" __global__ void spme_real_pair_force(
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    const float *charges,
    float *pair_forces_x,
    float *pair_forces_y,
    float *pair_forces_z,
    float *pair_energies,
    float *pair_virials,
    unsigned int max_neighbors,
    float lx, float ly, float lz, float xy, float xz, float yz,
    float k_coulomb,
    float alpha,
    float r_cut_real,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const float *atom_excl_coul_scales,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    unsigned int n);
```

Launch configuration: `(grid_x, grid_y, 1) = (ceil(max_neighbors/16),
ceil(n/16), 1)` blocks of `(16, 16, 1) = 256` threads, matching the LJ
and truncated-Coulomb kernels.

### Real-space reproducibility <!-- rq-cf6116b8 -->

Same as the truncated Coulomb pair force: each pair-buffer slot is written
by exactly one thread; no atomics; reduction is the existing
`reduce_pair_forces` kernel in deterministic slot order.

## Reciprocal-space pipeline <!-- rq-9ca00d25 -->

The reciprocal-space slot owns:

- A dedicated `NeighborListState` configured in cell-list-only mode with
  `n_cells_per_direction = grid` (see *Bin Structure* below and
  `neighbor-list.md` for the construction path). This service produces a
  sorted particle-ID list with the FFT grid's resolution.
- Real-valued grid buffers `rho: [f32; M]` (charge density) and
  `V: [f32; M]` (smoothed potential) where `M = n_a · n_b · n_c`.
- A complex-valued grid `rho_hat: [c32; M_complex]` where
  `M_complex = n_a · n_b · (n_c/2 + 1)`. cuFFT stores real-to-complex
  output in Hermitian symmetry format.
- A precomputed influence function `influence_G: [f32; M_complex]`
  rebuilt whenever the simulation box's `generation` counter changes.
- Per-axis B-spline correction factor arrays
  `b_factors_a: [f32; n_a]`, `b_factors_b: [f32; n_b]`,
  `b_factors_c: [f32; n_c]` precomputed once at slot construction.
- A cuFFT plan handle for the `(n_a, n_b, n_c)` R2C / C2R transforms.
  Both plans are bound to the slot's `recip_stream` via `cufftSetStream`
  at construction and are never rebound.
- A dedicated CUDA stream `recip_stream` on which the four reciprocal
  kernels execute concurrently with the default stream's work for the
  same timestep.
- Two CUDA events `default_ready_event` and `recip_ready_event` used to
  synchronize `recip_stream` with the device's default stream at the
  entry and exit of the reciprocal pipeline (see *Launch configuration*).
- A host-side scratch `Vec<f32>` of length `M_complex` for the
  `virial_per_cell` dtoh that runs at the start of `reduce()`.

### Bin structure <!-- rq-618bc65e -->

The spread and gather kernels both walk a bin structure with one bin per
FFT grid cell. The slot constructs this via
`NeighborListState::new_cell_list_only` (see `neighbor-list.md`): a
cell-list-mode state with explicit `n_cells_per_direction = grid` that
runs only the cell-index, prefix-scan, scatter, and in-cell sort stages
(no neighbor-list build, no displacement check). The state's
`sorted_particle_ids` and `cell_offsets` buffers carry the bin structure
the spread and gather kernels read.

The slot's `pre_step` rebuilds the bin structure every step
unconditionally (no skin tolerance): particles cross FFT-grid cell
boundaries on every reasonable timestep and the spread kernel must see
the latest binning.

### Charge spreading <!-- rq-037dd2f3 -->

The spread kernel computes the charge density `rho[g]` for each grid
point `g = (g_a, g_b, g_c)`:

```text
rho[g] = Σ_i q_i · M_p(s_a_i · n_a - g_a) · M_p(s_b_i · n_b - g_b) · M_p(s_c_i · n_c - g_c)
```

where `s_i = (s_a_i, s_b_i, s_c_i)` are particle `i`'s fractional
coordinates (computed via `SimulationBox::fractional_coords` on the
wrapped position) and `M_p` is the 1D cardinal B-spline of order `p`.
The sum runs over every particle whose support intersects `g`, i.e. every
particle whose primary bin lies within the box of `p × p × p` bins
centred on `g`.

**Iteration direction.** The kernel uses **one thread per grid point**.
Each thread:

1. Computes its own grid coordinate `(g_a, g_b, g_c)` from the
   thread/block index.
2. Iterates `(d_a, d_b, d_c) ∈ {−⌈p/2⌉ + 1, …, ⌊p/2⌋}³` (a box of
   `p³` neighbouring bins, wrapping modulo `n_d` per direction).
3. For each visited bin, walks the sorted particle IDs in
   `sorted_particle_ids[cell_offsets[c] .. cell_offsets[c+1]]` in
   ascending particle-index order.
4. For each particle `i` in the bin, computes the 1D B-spline weights
   `w_a = M_p(s_a_i · n_a − g_a)`, similarly for `b` and `c`, multiplies
   them, multiplies by `q_i`, and accumulates into a per-thread `rho`
   register.
5. Writes the final accumulator to `rho[grid_index(g)]`.

Each grid point is written by exactly one thread; no atomics are used.
The accumulator is summed in (bin-iteration, particle-iteration) order,
which is fixed across runs given identical positions on the same GPU.

The grid index uses the standard row-major mapping
`grid_index(g_a, g_b, g_c) = (g_a · n_b + g_b) · n_c + g_c`.

### Forward FFT <!-- rq-f2673343 -->

A single cuFFT R2C plan transforms `rho` into `rho_hat`:

```text
cuFFT_R2C_3D(plan, in=rho, out=rho_hat)
```

The plan is constructed once at slot init and reused every step. The
plan dimensions are `(n_a, n_b, n_c)` in cuFFT's natural ordering (the
slowest-varying axis first; consistent with our row-major grid).

cuFFT's R2C output has length `n_a · n_b · (n_c/2 + 1)` complex32
entries; the kernel reads this directly without rearrangement.

### Influence function <!-- rq-e7b74f7a -->

The precomputed influence function for grid index
`k = (k_a, k_b, k_c)` (with `k_c < n_c/2 + 1`) is

```text
m_a = (k_a < n_a / 2) ? k_a : k_a − n_a
m_b = (k_b < n_b / 2) ? k_b : k_b − n_b
m_c = (k_c < n_c / 2) ? k_c : k_c − n_c    # always k_c since k_c < n_c/2 + 1

K = 2π · (m_a · b_a + m_b · b_b + m_c · b_c)
K2 = |K|²

G[k] = (4π / K²) · exp(−K² / (4 α²))
       · b_factors_a[k_a] · b_factors_b[k_b] · b_factors_c[k_c]
```

where `b_a`, `b_b`, `b_c` are the rows of the reciprocal lattice matrix
`H^(-T)` (computed once from the simulation box) and the `b_factors_*`
are the SPME B-spline correction terms:

```text
b_factors_d[k] = |Σ_{j=0..p-1} M_p(j + 1) · exp(2π i · k · j / n_d)|⁻²
```

The `k = (0, 0, 0)` slot is set to zero, implementing tinfoil boundary
conditions and removing the (unphysical) infinite background-charge
contribution.

`influence_G` is rebuilt whenever the slot observes a different
`sim_box.generation()` from the value captured at last rebuild (the same
box-generation tracking pattern as `NeighborListState`). When the box has
not changed, the precomputed values are reused.

`b_factors_*` are independent of the box and depend only on the grid
dimensions and spline order; they are computed once at slot construction
and never rebuilt.

### Influence-function multiply and inverse FFT <!-- rq-95385a9d -->

A per-cell multiply kernel applies `V_hat[k] = G[k] · rho_hat[k]` for
every `k`, including writing zero for `k = (0, 0, 0)`. One thread per
complex grid cell; no atomics.

A cuFFT C2R plan transforms `V_hat` back into `V`:

```text
cuFFT_C2R_3D(plan, in=V_hat, out=V)
```

The R2C and C2R plans may be one combined plan handle or two; the
implementation is free to choose. Both transforms reuse the same
`(n_a, n_b, n_c)` plan dimensions.

### Force gather <!-- rq-df8766ae -->

The reciprocal-space force on particle `i` is

```text
F_i_recip = −q_i · ∇_r ( Σ_g V[g] · M_p(s_a · n_a − g_a)
                                  · M_p(s_b · n_b − g_b)
                                  · M_p(s_c · n_c − g_c) )
```

Operationally: one thread per particle. Each thread:

1. Reads particle `i`'s wrapped position and computes the fractional
   coordinates `(s_a, s_b, s_c)`.
2. Determines its primary bin and iterates the `p × p × p` neighbouring
   bins of grid points (wrapping modulo `n_d`).
3. For each grid point `g` in the support, computes the 1D B-spline
   weights `w_a, w_b, w_c` and the corresponding 1D derivatives
   `dw_a, dw_b, dw_c`.
4. Accumulates per-particle force components from the gradient
   contribution `V[g] · (dw_a · w_b · w_c, w_a · dw_b · w_c, w_a · w_b · dw_c)`
   scaled by `−q_i · n_d` (the chain-rule factor for the fractional-to-
   grid map).
5. Accumulates the per-particle reciprocal energy
   `0.5 · q_i · Σ_g V[g] · w_a · w_b · w_c`.

Each particle is written by exactly one thread; no atomics, no race
conditions. Summation order over the `p³` grid points within a particle
is fixed in `(d_a, d_b, d_c)` lexicographic order, so the contribution
ordering is deterministic.

### Reciprocal-space virial <!-- rq-ce4590c1 -->

The reciprocal-space slot computes the scalar virial trace from the
reciprocal grid:

```text
W_recip = (k_C / 2) · Σ_{k ≠ 0} G[k] · |rho_hat[k]|² · (1 − K² / (2 α²))
```

The contribution per complex-grid cell is computed by the same kernel
that does the influence-function multiply (or a separate kernel that
runs alongside; the implementation is free to fuse or split). The cell
values are summed into a single scalar `W_recip` by host-side
accumulation of the `virial_per_cell` buffer: at the start of
`SpmeReciprocalState::reduce()`, the default stream issues
`cudaStreamWaitEvent(default_stream, recip_ready_event)` so the
subsequent `dtoh_sync_copy_into(virial_per_cell, host_scratch)` sees
finalized values, the host walks the scratch in `0..M_complex` order
and accumulates into an `f64`, and the result is scaled to
`W_recip / N`. The host `f64` accumulation order is fixed across runs;
the scalar is therefore byte-identical on the same hardware.

The scalar is distributed per particle by equal division: each particle
receives `W_recip / N` in its `virials[i]` slot. Summing `virials` over
all particles yields the system total `W_recip`. The per-particle
attribution has no individual physical meaning; the convention exists
only so the SoA `virials: Vec<f32>` layout sums correctly.

The real-space slot writes the per-pair virial contribution in the
standard pair-buffer pattern (`pair_virials[slot] = 0.5 · scale · factor · r²`).

### Self-energy <!-- rq-29bdf2b2 -->

The self-energy `U_self = k_C · (α / √π) · Σ_i q_i²` is constant for the
duration of the run (charges do not change). The slot computes the
per-particle self-energy contribution

```text
u_self_i = k_C · (α / √π) · q_i²
```

once at slot construction by reading the host-side charges, and stores
the resulting per-particle constant in a device buffer
`u_self_per_particle: CudaSlice<f32>` of length `N`. Every step, the
slot's `reduce()` writes the reciprocal-space per-particle energy as

```text
energy_per_particle[i] = (per-particle reciprocal contribution) − u_self_per_particle[i]
```

Summing `energy_per_particle` over all particles yields
`U_reciprocal − U_self`, matching the Ewald decomposition.

`u_self_per_particle` is rebuilt only when per-particle charges change.
Charges are immutable for the lifetime of a run in v1, so the buffer is
computed once at construction and never updated.

## cuFFT determinism precondition <!-- rq-017a61a4 -->

`init_device` runs a smoke test that confirms cuFFT produces
byte-identical output on two consecutive R2C transforms of the same
input. The test runs only when the active simulation config selects
SPME (i.e. when `config.spme.is_some()`); on configs without SPME the
test is skipped to avoid startup overhead.

The smoke test:

1. Constructs a 16×16×16 R2C plan on the same device the simulation will
   use.
2. Uploads a fixed pseudo-random `f32` grid of 4096 entries seeded with
   a constant (independent of the simulation's RNG seeds).
3. Runs the R2C transform twice in succession on the same input buffer
   into two separate output buffers.
4. Downloads both output buffers and compares them byte-for-byte.

A byte mismatch surfaces as
`RunnerError::CuFftNonDeterministic { differences: usize }` and exits
with code `1`. The test runs once per `init_device` invocation; the
result is not cached across processes.

When SPME is not configured, `init_device` does not initialise cuFFT and
does not run the smoke test. The cuFFT plan and any cuFFT bindings exist
only inside the `SpmeReciprocalState` construction path.

## Feature API <!-- rq-47a9ef14 -->

### Types <!-- rq-66067eba -->

- `SpmeConfig` — parsed `[spme]` table. Fields: <!-- rq-61889ff1 -->
  - `alpha: f64`
  - `r_cut_real: f64`
  - `grid: [u32; 3]`
  - `spline_order: u32`

- `SpmeRealSpaceState` — implements `Potential` with <!-- rq-22171569 -->
  `label() == "spme_real"`. Reports
  `max_cutoff() = Some(r_cut_real as f32)` so the shared neighbor list
  sizes its search radius. Fields private; the slot's public surface is
  the per-step methods invoked by `ForceField::step` (see
  `framework.md`).

  Constructor:
  - `SpmeRealSpaceState::new(gpu: &GpuContext, particle_count: usize, alpha: f32, r_cut_real: f32, max_neighbors: u32, exclusion_list: &ExclusionList) -> Result<SpmeRealSpaceState, NeighborListError>`

- `SpmeReciprocalState` — implements `Potential` with <!-- rq-b1148667 -->
  `label() == "spme_reciprocal"`. Reports `max_cutoff() = None` (it does
  not contribute to the shared neighbor list's search radius). Fields
  private. The slot owns its own bin-only `NeighborListState` for the
  spread / gather kernels, plus the dedicated `recip_stream` and the
  two cross-stream events used by the reciprocal pipeline (see
  *Reciprocal-space pipeline*). `recip_stream` and the two events are
  allocated by `SpmeReciprocalState::new` and released in the slot's
  `Drop` impl.

  Constructor:
  - `SpmeReciprocalState::new(gpu: &GpuContext, sim_box: &SimulationBox, particle_count: usize, charges: &[f32], alpha: f32, grid: [u32; 3], spline_order: u32) -> Result<SpmeReciprocalState, SpmeError>`

- `SpmeError` — error type for SPME slot construction. Variants: <!-- rq-ebfa6e1f -->
  - `NeighborList(NeighborListError)` — from the bin-only neighbor-list
    construction (e.g. `BoxTooSmallForCells` if the FFT grid dims
    exceed what the box can accommodate).
  - `CuFft(CuFftError)` — cuFFT plan construction failed.
  - `InvalidGrid { axis: &'static str, n: u32, spline_order: u32 }` —
    one of the grid dimensions is less than `2 · spline_order`. Loader
    validation enforces this before construction, but the slot
    re-validates as a guard against direct API misuse.
  - `Gpu(GpuError)` — a CUDA driver operation failed during buffer
    allocation.

- `CuFftError` — wrapper around cuFFT failure codes from the underlying <!-- rq-1ad7e751 -->
  bindings. Variants follow the `cufftResult_t` enumeration as needed by
  the implementation (`InvalidPlan`, `ExecFailed`, etc.).

### Functions <!-- rq-cf82e422 -->

- `spme_real_pair_force(particle_buffers, pair_buffer, sim_box, alpha, r_cut_real, atom_excl_offsets, atom_excl_partners, atom_excl_coul_scales, neighbor_list, neighbor_counts) -> Result<(), GpuError>` <!-- rq-f735ea05 -->
  Launches the `spme_real_pair_force` kernel.

- `spme_charge_spread(particle_buffers, spme_state) -> Result<(), GpuError>` <!-- rq-f69698b8 -->
  Launches the charge-spread kernel. Writes `spme_state.rho`.

- `spme_forward_fft(spme_state) -> Result<(), CuFftError>` <!-- rq-24e36eba -->
  Runs the cuFFT R2C transform `rho → rho_hat`.

- `spme_influence_multiply(spme_state) -> Result<(), GpuError>` <!-- rq-8326d2d1 -->
  Multiplies `rho_hat[k] *= G[k]` in place, also writing the per-cell
  virial contribution to a scratch buffer for the subsequent reduction.

- `spme_inverse_fft(spme_state) -> Result<(), CuFftError>` <!-- rq-a98abc35 -->
  Runs the cuFFT C2R transform `V_hat → V`.

- `spme_force_gather(particle_buffers, spme_state, slot_output) -> Result<(), GpuError>` <!-- rq-c6f6a13c -->
  Launches the force-gather kernel. Writes per-particle force,
  reciprocal energy, and (after the deterministic virial reduction)
  per-particle virial.

- `cufft_determinism_smoke_test(device: &Arc<CudaDevice>) -> Result<(), CuFftError>` <!-- rq-d880c228 -->
  Used by `init_device` when SPME is enabled. Returns `Err` on a
  byte-difference between two consecutive R2C transforms.

### CUDA kernels <!-- rq-7225b86f -->

`kernels/spme_real.cu` declares:

```c
extern "C" __global__ void spme_real_pair_force(
    /* signature shown in the Real-space CUDA kernel section above */);
```

`kernels/spme_recip.cu` declares:

```c
extern "C" __global__ void spme_charge_spread(
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    const float *charges,
    const unsigned int *sorted_particle_ids,
    const unsigned int *cell_offsets,
    float lx, float ly, float lz, float xy, float xz, float yz,
    unsigned int n_a, unsigned int n_b, unsigned int n_c,
    unsigned int spline_order,
    float *rho,
    unsigned int n);

extern "C" __global__ void spme_influence_multiply(
    const float *influence_G,
    float *rho_hat_real,   // interleaved real and imag parts
    float *rho_hat_imag,
    float *virial_per_cell,  // scratch for reciprocal-virial reduction
    float lx, float ly, float lz, float xy, float xz, float yz,
    unsigned int n_a, unsigned int n_b, unsigned int n_c,
    float alpha,
    unsigned int n_complex);

extern "C" __global__ void spme_force_gather(
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    const float *charges,
    const float *V,
    float lx, float ly, float lz, float xy, float xz, float yz,
    unsigned int n_a, unsigned int n_b, unsigned int n_c,
    unsigned int spline_order,
    float k_coulomb,
    float *slot_force_x,
    float *slot_force_y,
    float *slot_force_z,
    float *slot_energy,
    unsigned int n);
```

Plus a deterministic reduction kernel for the per-cell virial buffer
(can reuse the existing `reduce_pair_forces` machinery or implement a
new fixed-topology tree reduction; the kernel signature is left to the
implementation).

### Launch configuration <!-- rq-03d9869d -->

- `spme_real_pair_force`: 16×16×1 = 256 threads/block, grid <!-- rq-eb9e5cc3 -->
  `(ceil(max_neighbors/16), ceil(N/16), 1)`. Matches the LJ kernel.
- `spme_charge_spread`: one thread per real grid cell, block size 256, <!-- rq-16f6c7dc -->
  grid `ceil(M / 256)` where `M = n_a · n_b · n_c`.
- `spme_influence_multiply`: one thread per complex grid cell, block <!-- rq-127df3d6 -->
  size 256, grid `ceil(M_complex / 256)`.
- `spme_force_gather`: one thread per particle, block size 256, grid <!-- rq-35b76155 -->
  `ceil(N / 256)`.

Stream assignment:

- `spme_real_pair_force` and `spme_force_gather` run on the device's <!-- rq-44cce069 -->
  default stream carried by `particle_buffers.device`.
- The four reciprocal-pipeline kernels — `spme_charge_spread`, the
  cuFFT R2C transform, `spme_influence_multiply`, and the cuFFT C2R
  transform — run on a dedicated CUDA stream `recip_stream` owned by
  `SpmeReciprocalState`. The R2C and C2R plans are bound to
  `recip_stream` via `cufftSetStream` once at slot construction and
  are never rebound.

Cross-stream synchronization uses two CUDA events owned by
`SpmeReciprocalState`:

- `default_ready_event` — recorded on the default stream at the start <!-- rq-274db88b -->
  of `SpmeReciprocalState::contribute()` and waited on by
  `recip_stream` (via `cudaStreamWaitEvent`) before the first
  reciprocal kernel enqueues. Guarantees that the integrator's writes
  to `positions_x/y/z` and the construction-time writes to `charges`
  are visible to the reciprocal pipeline.
- `recip_ready_event` — recorded on `recip_stream` immediately after <!-- rq-0d5e76ec -->
  the inverse FFT enqueues at the end of `contribute()`, and waited on
  by the default stream at the start of `SpmeReciprocalState::reduce()`.
  Guarantees that `V` and `virial_per_cell` are finalized before the
  default stream's `virial_per_cell` dtoh and the `spme_force_gather`
  launch.

Both events are reused across timesteps; they are created in
`SpmeReciprocalState::new` and released when the slot is dropped. The
host call to `SpmeReciprocalState::contribute()` returns as soon as the
four reciprocal kernels have been enqueued on `recip_stream`; the host
does not block on cuFFT or on the reciprocal kernels.

## Reproducibility <!-- rq-20530653 -->

SPME on Dynamics is bit-exact GPU-vs-GPU when run on the same hardware
with identical inputs. Six components carry the reproducibility
invariant:

1. **Bin structure.** Inherits the existing cell-list service's
   determinism: particles sorted by `(cell_index, particle_id)` with a
   fixed-topology insertion sort within each bin.
2. **Charge spread.** One thread per grid point; per-thread accumulation
   in fixed `(bin_index, particle_id)` order.
3. **cuFFT.** Deterministic for fixed plan dimensions, single-stream
   usage, and the same hardware. "Single-stream usage" is satisfied
   because both cuFFT plans are bound once at slot construction to
   `recip_stream` via `cufftSetStream` and are never rebound. The
   `cufft_determinism_smoke_test` run at `init_device` time validates
   the contract on the host's specific cuFFT version.
4. **Influence-function multiply and virial-per-cell write.** One
   thread per complex grid cell; no atomics; no inter-thread reads.
5. **Force gather.** One thread per particle; each thread reads `p³`
   grid points in fixed `(d_a, d_b, d_c)` lexicographic order. No
   atomics. The per-particle virial uses the equal-division attribution
   `W_recip / N` (the slot's `reduce()` distributes the scalar
   identically to every particle, so the SoA convention is preserved
   regardless of summation order).
6. **Two-stream model.** The reciprocal pipeline runs on
   `recip_stream`; the real-space slot, the force-gather kernel, and
   every non-SPME slot's kernel run on the default stream. The two
   streams write to disjoint device buffers — `recip_stream` writes
   `rho`, `rho_hat`, `V`, and `virial_per_cell`; the default stream
   writes the real-space `PairBuffer`, the slot-output buffers, and
   `forces_*`. Cross-stream ordering is enforced by the two events
   `default_ready_event` and `recip_ready_event` recorded at
   deterministic points in `contribute()` and waited on at
   deterministic points in `contribute()` / `reduce()`; the
   wait edges are independent of host thread scheduling. Two runs with
   identical inputs on the same GPU therefore observe byte-identical
   writes from both streams.

The reciprocal-virial scalar reduction sums `virial_per_cell` on the
host in `f64` in fixed index order at the start of `reduce()`; the
implementation must not use unordered atomic-add or any
non-deterministic device-side reduction.

## Out of Scope <!-- rq-f0038583 -->

- Particle-Particle-Particle-Mesh (P3M) variants. The implementation is
  SPME with the B-spline-corrected influence function specified above;
  P3M's optimised influence function is a separate (related) feature.
- Higher-order B-splines beyond `p = 8`. Larger orders work in
  principle but pay a `p³` work multiplier per grid point and require
  larger FFT grids; not exercised in v1.
- Auto-tuning `alpha`, `grid`, and `r_cut_real` to a requested accuracy
  budget. The loader takes the values as given; users tune them
  manually.
- Per-frame charge updates. Charges are fixed for the lifetime of a
  run; `u_self_per_particle` is computed once.
- Excluded-pair real-space correction. The 1-2 and 1-3 bonded pairs are
  zero-scaled in `atom_excl_coul_scales` (the standard convention for
  excluded pairs in PME). Codes that retain a portion of the bonded
  Coulomb (e.g. via 1-4 scaling) use the `scale_coul` field of
  `topology.md`'s exclusion entries; no special PME-only excluded-pair
  treatment is added.
- Non-tinfoil boundary conditions. The `k = 0` entry of the influence
  function is fixed at zero; conductive ("tinfoil") boundary conditions
  are the only supported convention.
- Multi-grid PME. Splitting the reciprocal sum across multiple grids of
  different resolutions is a future optimisation.
- Charge-neutrality enforcement or warnings at config-load time. Non-
  neutral systems run with the tinfoil convention; the user is
  responsible for the physical interpretation.
- Reciprocal-space virial diagonal anisotropy. The slot writes the
  scalar trace `W_recip` only; the off-diagonal pressure-tensor
  components are not computed. A future barostat that needs the full
  pressure tensor is a separate feature.
- SPME-on-CPU fallback. The slot requires CUDA + cuFFT; there is no
  CPU implementation.

---

## Gherkin Scenarios <!-- rq-c0668cb7 -->

```gherkin
Feature: Smooth particle-mesh Ewald (SPME)

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called with a config that selects SPME
    And the Coulomb constant k_C = 8.987551787e9 (rounded to f32)
    And an orthorhombic SimulationBox with lx=ly=lz=2.0e-9
    And an SpmeConfig with alpha=2.0e10, r_cut_real=8.0e-10,
        grid=[16, 16, 16], spline_order=4

  # --- Slot presence and parameter validation ---

  @rq-dd41209b
  Scenario: SPME slots present in the ForceField when [spme] is configured
    Given a Config with the Background SpmeConfig
    When ForceField::new is called
    Then the slot list contains SpmeRealSpaceState (label "spme_real")
    And the slot list contains SpmeReciprocalState (label "spme_reciprocal")
    And the slot list does not contain CoulombState (truncated Coulomb)

  @rq-aeb23925
  Scenario: Reject grid dimension below 2*spline_order
    Given an SpmeConfig with spline_order=4 and grid=[6, 16, 16]
    When SpmeReciprocalState::new is called
    Then it returns Err(SpmeError::InvalidGrid { axis: "a", n: 6, spline_order: 4 })

  @rq-ab74a666
  Scenario: Reject grid dimension below 2*spline_order along b
    Given an SpmeConfig with spline_order=6 and grid=[16, 8, 16]
    When SpmeReciprocalState::new is called
    Then it returns Err(SpmeError::InvalidGrid { axis: "b", n: 8, spline_order: 6 })

  # --- cuFFT determinism precondition ---

  @rq-637cd1a5
  Scenario: cuFFT smoke test passes on a determinism-conforming installation
    Given a config that selects SPME
    When init_device() is called
    Then it succeeds without error
    And cufft_determinism_smoke_test returned Ok(()) internally

  @rq-02f4d342
  Scenario: cuFFT smoke test fails-loud when consecutive transforms produce different output
    Given a config that selects SPME
    And cuFFT (simulated) produces byte-different output on consecutive R2C calls of the same input
    When init_device() is called
    Then it returns Err(RunnerError::CuFftNonDeterministic { differences: d }) with d > 0

  @rq-ea4205ec
  Scenario: cuFFT smoke test is skipped when SPME is not configured
    Given a config that does not declare [spme]
    When init_device() is called
    Then cufft_determinism_smoke_test is not invoked

  # --- Real-space slot: physics ---

  @rq-3726c0f1
  Scenario: Real-space slot produces zero force on an excluded pair (scale_coul=0)
    Given two oppositely charged particles within r_cut_real
    And an ExclusionList listing the pair with scale_coul=0.0
    When the real-space slot's contribute() is called
    Then the pair-buffer slot for (0, 1) has zero force, zero energy, zero virial

  @rq-af7018c0
  Scenario: Real-space slot produces zero outside r_cut_real
    Given two particles at separation greater than r_cut_real
    When the real-space slot's contribute() is called
    Then the pair-buffer slot has zero force, zero energy, zero virial

  @rq-83088c2f
  Scenario: Real-space slot matches the closed-form erfc force for an isolated pair
    Given two unit-charge particles at separation r = 4.0e-10 inside the cutoff
    And alpha = 2.0e10
    When the real-space slot's contribute() is called
    Then pair_forces_x[0*max + 1] equals the closed-form
      k_C · q_i · q_j · (erfc(α r) · inv_r2 + (2 α / √π) · exp(-α² r²) · inv_r2) · dx · inv_r
      to within 1e-5 relative

  @rq-0caebe37
  Scenario: Real-space slot obeys Newton's third law for a non-boundary pair
    Given two particles at non-boundary separation
    When the real-space slot's contribute() is called
    Then pair_forces_x[0*max + slot_j] equals −pair_forces_x[1*max + slot_i] bit-exactly

  # --- Reciprocal-space pipeline: spread ---

  @rq-881559bd
  Scenario: Spread produces zero charge density at a grid point with no particle support
    Given one particle at position s = (0.5, 0.5, 0.5) in fractional coords
    And a grid point at fractional position (0.0, 0.0, 0.0) far from the particle's support
    When spme_charge_spread is called
    Then rho[grid_index(0, 0, 0)] equals 0.0

  @rq-79291441
  Scenario: Spread sum integrates to total charge
    Given a system of N particles with total charge Q = Σ q_i
    When spme_charge_spread is called
    Then Σ_g rho[g] equals Q to within 1e-5 relative tolerance
      (B-splines are partition-of-unity normalised)

  @rq-07297467
  Scenario: Spread is independent of particle order in the bin
    Given two particles at the same primary bin, presented in two different
      input orderings (e.g. swapped IDs)
    When spme_charge_spread is called
    Then the resulting rho is byte-identical between the two runs
      (the cell-list sort canonicalises particle order within a bin)

  @rq-3c0beda9
  Scenario: Spread reduces to a single B-spline weight for one isolated particle
    Given exactly one particle with charge q at fractional position s
    When spme_charge_spread is called
    Then rho[g] equals q · M_p(s_a·n_a - g_a) · M_p(s_b·n_b - g_b) · M_p(s_c·n_c - g_c)
      for every grid point g

  # --- Reciprocal-space pipeline: FFT ---

  @rq-e3c3898a
  Scenario: Forward FFT of a zero grid produces zero
    Given a charge density `rho` of all zeros
    When spme_forward_fft is called
    Then rho_hat is all zeros (both real and imag parts)

  @rq-f02e9e0e
  Scenario: Inverse FFT round-trips the forward FFT
    Given a non-trivial rho
    When spme_forward_fft and spme_inverse_fft are called in succession
      (without the influence-function multiply in between)
    Then the round-trip output equals the input scaled by the FFT normalisation factor
      (cuFFT convention: forward+inverse = N · identity)

  # --- Reciprocal-space pipeline: multiply ---

  @rq-e5bf6fea
  Scenario: The k=0 entry is zeroed by the influence-function multiply
    Given any non-zero rho_hat
    When spme_influence_multiply is called
    Then rho_hat[k=0] equals 0.0 + 0.0i after the multiply
      (tinfoil boundary condition)

  # --- Force gather ---

  @rq-bd684a0d
  Scenario: Force gather on one isolated particle returns the gradient of its own spread
    Given a single particle with charge q and the spread + FFT + multiply + IFFT pipeline run
    When spme_force_gather is called
    Then the per-particle force F equals -q · ∇V evaluated at the particle position
      to within 1e-4 relative
    And after subtracting the self-energy gradient (zero for an isolated particle in v1),
      the result matches the closed-form short-distance limit

  @rq-2996a545
  Scenario: Force gather and forward FFT of force agree with explicit Ewald for small N
    Given 4 particles with random fractional positions and random charges summing to 0
    And explicit-Ewald reference forces computed on the host with the same parameters
    When the full SPME pipeline runs
    Then per-particle forces agree with the reference within 1e-3 relative tolerance

  # --- Self-energy ---

  @rq-ef8dee82
  Scenario: Self-energy is subtracted per particle and totals -k_C·(α/√π)·Σ q²
    Given a system with N particles and charges (+e, -e)
    When the reciprocal slot's reduce() runs
    Then Σ_i u_self_per_particle[i] equals k_C · (α/√π) · (e² + e²)
    And the slot's per-particle energy output equals
      (reciprocal-energy share) − u_self_per_particle[i] for each particle

  # --- Reciprocal-space virial ---

  @rq-0816969e
  Scenario: Reciprocal-space virial is distributed equally per particle
    Given a non-zero rho_hat producing a non-zero W_recip
    When the reciprocal slot's reduce() runs
    Then every particle's virials[i] entry equals W_recip / N (to within f32 round-off)
    And summing virials over all particles recovers W_recip

  # --- Reproducibility ---

  @rq-09d4e13f
  Scenario: Two independent SPME runs on identical inputs produce byte-identical outputs
    Given two SpmeReciprocalState instances with identical config and identical positions
    When the full pipeline (spread + FFT + multiply + IFFT + gather) runs on each
    Then the per-particle force, energy, and virial buffers are byte-identical
      between the two runs

  # --- Mutual exclusion with truncated Coulomb ---

  @rq-203ecf81
  Scenario: Reject a config that declares both [spme] and [coulomb]
    Given a Config TOML with both [spme] and [coulomb] tables
    When load_config is called
    Then it returns Err(ConfigError::ConflictingElectrostatics { .. })

  # --- Box-compat check ---

  @rq-674cc467
  Scenario: Box compatibility picks up SPME's real-space cutoff
    Given a Config with the Background SpmeConfig (r_cut_real = 8.0e-10)
    And a simulation box with min_perpendicular_width = 2.0e-9
    And neighbor_list.r_skin = 1.0e-10
    When the runner runs the cell-list compatibility check
    Then required = 3 · (8.0e-10 + 1.0e-10) = 2.7e-9 > 2.0e-9 → fails
    And the runner exits with RunnerError::CellListBoxTooSmall referencing the smallest perpendicular direction

  @rq-991b4695
  Scenario: Box compatibility ignores SPME's reciprocal grid
    Given a Config with grid=[256, 256, 256] but r_cut_real well within the box
    When the runner runs the cell-list compatibility check
    Then it passes (the grid resolution does not enter the check)

  # --- Bin-only cell-list construction ---

  @rq-2ae37ac3
  Scenario: SPME reciprocal state's internal cell list uses one bin per FFT grid cell
    Given an SpmeConfig with grid=[16, 16, 16]
    When SpmeReciprocalState::new is called
    Then the slot's internal NeighborListState has n_cells = [16, 16, 16]
    And the slot's internal NeighborListState has mode CellListOnly
    And the slot's internal state does not allocate the neighbor_list or neighbor_counts buffers

  @rq-dd829afb
  Scenario: Bin-only cell list rebuilds every step regardless of particle displacement
    Given an SpmeReciprocalState with N particles immediately after a step
    And no particle has moved more than f32 epsilon since the last rebuild
    When the slot's pre_step is called for the next step
    Then a fresh cell-list rebuild was performed
    And the displacement-check kernel was not launched (it is absent in bin-only mode)

  # --- Per-particle charge dependence ---

  @rq-ea67c26b
  Scenario: Doubling all charges quadruples the reciprocal-space energy and forces scale linearly
    Given a system where reciprocal energy U_r is recorded for charges {q_i}
    When the same system is run with charges {2 · q_i}
    Then the new reciprocal energy equals 4 · U_r to within 1e-4 relative
    And the per-particle forces double in magnitude

  # --- Triclinic box ---

  @rq-3b9611f2
  Scenario: SPME runs on a triclinic box with non-zero tilts
    Given a SimulationBox with non-zero xy, xz, yz tilts
    And an SpmeConfig with valid grid dimensions
    When the full SPME pipeline runs
    Then the influence function uses k-vectors from the reciprocal lattice H^(-T)
    And the per-particle reciprocal-space force agrees with an explicit-Ewald reference
      on the same triclinic box to within 1e-3 relative

  # --- Streams and cross-stream synchronization ---

  @rq-5f54b9b3
  Scenario: SpmeReciprocalState owns a dedicated CUDA stream and two synchronization events
    Given a constructed SpmeReciprocalState built via SpmeReciprocalState::new
    Then state.recip_stream is a CudaStream handle distinct from the device's default stream
    And state.default_ready_event and state.recip_ready_event are CudaEvent handles

  @rq-29d0e458
  Scenario: cuFFT R2C and C2R plans are bound to the recip stream at construction
    Given a constructed SpmeReciprocalState
    Then cufftSetStream was called on the R2C plan with recip_stream during SpmeReciprocalState::new
    And cufftSetStream was called on the C2R plan with recip_stream during SpmeReciprocalState::new
    And no further cufftSetStream calls are issued during step execution

  @rq-5abda6fa
  Scenario: Reciprocal pipeline kernels enqueue on the recip stream
    Given a constructed SpmeReciprocalState
    When SpmeReciprocalState::contribute is called
    Then spme_charge_spread, the forward cuFFT R2C, spme_influence_multiply, and the inverse cuFFT C2R each enqueue on recip_stream
    And the default stream's pending work queue contains no reciprocal-pipeline launches

  @rq-b1d70f2f
  Scenario: contribute() returns without blocking on the reciprocal pipeline
    Given a constructed SpmeReciprocalState
    When SpmeReciprocalState::contribute is called
    Then contribute() returns Ok(()) without invoking cudaStreamSynchronize or any dtoh on the recip stream
    And the host-side virial_host_scratch has not been read

  @rq-19bc076f
  Scenario: contribute() records default_ready_event on the default stream and the recip stream waits on it before the first kernel
    Given a constructed SpmeReciprocalState
    When SpmeReciprocalState::contribute is called
    Then a cudaEventRecord(default_ready_event, default_stream) call precedes the recip pipeline launches
    And a cudaStreamWaitEvent(recip_stream, default_ready_event) call precedes the first reciprocal kernel enqueue

  @rq-0fd9a581
  Scenario: contribute() records recip_ready_event on the recip stream after the inverse FFT
    Given a constructed SpmeReciprocalState
    When SpmeReciprocalState::contribute is called
    Then a cudaEventRecord(recip_ready_event, recip_stream) call follows the inverse cuFFT C2R enqueue

  @rq-46530505
  Scenario: reduce() waits for the recip stream before reading the per-cell virial
    Given SpmeReciprocalState::contribute has just been called and the recip pipeline is still in flight
    When SpmeReciprocalState::reduce is called
    Then a cudaStreamWaitEvent(default_stream, recip_ready_event) call precedes the dtoh_sync_copy_into of virial_per_cell
    And after the dtoh completes, the host scratch holds finalized virial_per_cell values

  @rq-7404b017
  Scenario: spme_force_gather runs on the default stream after the recip-stream wait
    Given SpmeReciprocalState::reduce has just been called
    Then spme_force_gather was enqueued on the default stream
    And the default stream's wait on recip_ready_event preceded the force_gather launch

  @rq-73efd4be
  Scenario: Two-stream pipeline preserves bit-exact reproducibility across runs
    Given two independent ForceField instances A and B, both with SPME enabled and identical inputs
    When each runs one full ForceField::step on the same GPU
    And each pipeline's ParticleBuffers.forces_x, forces_y, forces_z, potential_energies, virials are downloaded
    Then run A and run B agree byte-for-byte on every f32

  @rq-1aa9e851
  Scenario: recip_stream and the two events are released when the slot is dropped
    Given a constructed SpmeReciprocalState
    When the state is dropped
    Then recip_stream is destroyed via the cudarc Drop path
    And default_ready_event and recip_ready_event are destroyed via the cudarc Drop path
```
