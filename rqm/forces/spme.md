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
  functional form of the pair force. This slot runs only when the
  LJ + SPME-real fused composite (`lj-spme-real-fused.md`) is inactive
  — i.e. when no `[[pair_interactions]]` are configured. When LJ is
  also configured, the composite displaces this slot through the
  framework's displacement mechanism, and the standalone SPME-real
  kernel does not run for the lifetime of that `ForceField`.
- `SpmeReciprocalState` — owns the FFT grid buffers, the cuFFT plan, the
  precomputed influence function, and a dedicated bin-only cell-list
  used by the spread and gather kernels. Unaffected by displacement;
  the reciprocal-space pipeline always runs separately.

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

The kernel topology, sweep loop, warp-tree reduction, and per-particle
output write follow the common warp-per-particle pattern specified in
`pair-force-kernel.md`. The real-space-SPME contribution at each
`(i, k)` pair is computed as follows.

For lane `lane` of the warp handling particle `i` at sweep step `s`,
when `k = s * 32 + lane` satisfies `k < neighbor_counts[i]` and
`j = neighbor_list[i * max_neighbors + k]` is not equal to `i`:

1. Compute the displacement `(dx, dy, dz) = positions[i] − positions[j]`
   and apply the triclinic minimum-image algorithm of `simulation-box.md`.
2. Compute `r² = dx² + dy² + dz²`. If `r² > r_cut_real²`, the pair
   contributes nothing; the lane skips to its next assigned neighbour.
3. Read `q_i = charges[i]`, `q_j = charges[j]`.
4. Compute the screened Coulomb factor and energy:

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
   `j`. The derivative form combines the `1/r²` decay of `erfc(αr)/r`
   with the Gaussian term from `d(erfc)/dr = −(2α/√π) · exp(−α²r²)`.

5. Apply the per-pair Coulomb exclusion scale (see `topology.md`):
   `scale = exclusion_scale(i, j, atom_excl_offsets, atom_excl_partners,
   atom_excl_coul_scales)`. Multiply `factor` and `energy` by `scale`,
   and compute the scalar virial `w = factor · r²`.
6. Add `(factor * dx, factor * dy, factor * dz)` to the lane's
   `(p_x, p_y, p_z)` register accumulators. The `_fev` variant
   additionally adds `energy * 0.5f` to `p_e` and `w * 0.5f` to `p_w`.
   The `0.5` factor distributes each unordered pair's energy and virial
   across the two ordered contributions `(i, j)` and `(j, i)`.

After every lane has processed every assigned neighbour, the warp-tree
butterfly reduction collapses the 32 lane accumulators to lane 0, which
adds the particle's net force (and, in the `_fev` variant,
energy/virial) into its class accumulator via a read-modify-write at
the per-particle slot of the `SlotOutputView` it received. See
`pair-force-kernel.md` for the topology and reduction details, and
`framework.md`'s *Class Output Accumulators* for the accumulator
layout.

The real-space slot does not apply a switching function. The `erfc`
factor decays rapidly enough that a hard cutoff is acceptable when
`α · r_cut_real >= 3.5` (the loader does not enforce this; it is a
user-tuning concern documented in `io/config-schema.md`).

### Real-space CUDA kernels <!-- rq-9a512ed1 -->

`kernels/spme_real.cu` declares two `extern "C"` kernels (forces-only
and forces + energy + virial) that share the warp-per-particle pattern
documented in `pair-force-kernel.md`:

```c
extern "C" __global__ void spme_real_pair_force_f(
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    const float *charges,
    unsigned int max_neighbors,
    const float *lattice,           // length 6: [lx, ly, lz, xy, xz, yz]
    float k_coulomb,
    float alpha,
    float r_cut_real,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const float *atom_excl_coul_scales,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    float *slot_force_x,
    float *slot_force_y,
    float *slot_force_z,
    unsigned int n);

extern "C" __global__ void spme_real_pair_force_fev(
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    const float *charges,
    unsigned int max_neighbors,
    const float *lattice,           // length 6: [lx, ly, lz, xy, xz, yz]
    float k_coulomb,
    float alpha,
    float r_cut_real,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const float *atom_excl_coul_scales,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    float *slot_force_x,
    float *slot_force_y,
    float *slot_force_z,
    float *slot_energy,
    float *slot_virial,
    unsigned int n);
```

Launch configuration: `block_dim = (256, 1, 1)`,
`grid_dim = (ceil(n / 8), 1, 1)`, matching the LJ and truncated-Coulomb
kernels (see `pair-force-kernel.md`).

### Real-space reproducibility <!-- rq-cf6116b8 -->

Same as the truncated Coulomb pair force: the per-particle output is
the deterministic warp-tree sum of its per-pair contributions,
accumulated in the fixed lane-strided order specified by
`pair-force-kernel.md`. Identical runs on the same GPU with identical
inputs produce byte-identical `slot_force_*` outputs.

## Reciprocal-space pipeline <!-- rq-9ca00d25 -->

The reciprocal-space slot owns:

- A fixed-point charge-density grid `rho_fixed: CudaSlice<i64>` of
  length `M`. Each cell holds the per-particle contributions
  `q_i · w_a · w_b · w_c` accumulated via `atomicAdd<i64>` and
  represented as the fixed-point integer
  `(i64)(value × 2^32)`. The grid is zeroed before each step's
  spread and converted to f32 `rho` by `spme_spread_finish` after
  the spread completes. The fixed-point representation makes the
  per-step accumulation exactly associative across threads (integer
  atomicAdd is associative), so two runs on the same GPU with
  byte-identical inputs produce a byte-identical `rho_fixed`
  regardless of atomic-completion order.

- Real-valued grid buffers `rho: [f32; M]` (charge density,
  populated by `spme_spread_finish` from `rho_fixed`) and
  `V: [f32; M]` (smoothed potential) where `M = n_a · n_b · n_c`.
- A complex-valued grid `rho_hat: [c32; M_complex]` where
  `M_complex = n_a · n_b · (n_c/2 + 1)`. cuFFT stores real-to-complex
  output in Hermitian symmetry format.
- A device-resident influence function `influence_G: CudaSlice<f32>` of
  length `M_complex`, rebuilt whenever the simulation box's `generation`
  counter changes.
- A device-resident virial-factor table
  `virial_factor: CudaSlice<f32>` of length `M_complex` holding the
  per-cell multiplier `G[k] · (1 - K² / (2 α²))` read by
  `spme_recip_apply_influence` to compute the per-thread virial
  contribution it feeds into the per-block reduction. Rebuilt
  alongside `influence_G` on the same box-generation trigger.
- Per-axis B-spline correction factor arrays held as device buffers
  `b_factors_a: CudaSlice<f32>`, `b_factors_b: CudaSlice<f32>`, and
  `b_factors_c: CudaSlice<f32>` of length `n_a`, `n_b`, and `n_c`
  respectively. Populated once at slot construction from the host-side
  Cox-de Boor recursion and never re-uploaded; they depend only on
  `(grid, spline_order)`, not on the simulation box.
- A device-resident single-element scalar
  `w_per_particle_virial: CudaSlice<f32>` holding the per-particle
  reciprocal-virial share `W_recip / N` written by
  `spme_recip_reduce_partials` and read by `spme_force_gather`.
- A device-resident scratch buffer
  `virial_partials: CudaSlice<f32>` of length
  `ceil(M_complex / 256)` (the number of 256-thread blocks that
  cover the complex grid). Each block of `spme_recip_apply_influence`
  reduces its assigned slice of the complex grid into a single
  partial sum and writes it to its slot of `virial_partials`.
  `spme_recip_reduce_partials` then sums these partials into
  `w_per_particle_virial`. The two-stage shape avoids materialising
  a length-`M_complex` per-cell virial buffer in global memory.
- A cuFFT plan handle for the `(n_a, n_b, n_c)` R2C / C2R transforms.
  Both plans are bound to the device's default `CudaStream` via
  `cufftSetStream` at construction and are never rebound.
- A device-resident `workspace: CudaSlice<u8>` buffer of size
  `max(cufftGetSize(forward_plan), cufftGetSize(inverse_plan))`,
  allocated at slot construction. Both cuFFT plans run with
  `cufftSetAutoAllocation(plan, 0)` and have their work-area pointer
  bound to this buffer via `cufftSetWorkArea(plan, workspace)` at
  construction. The two plans share the buffer because their
  executions are strictly serialised on the default stream.

- The atom spatial pre-sort scratch (see *Atom spatial pre-sort* for
  the per-rebuild pipeline that consumes these):
  - `atom_bin_key: CudaSlice<u32>` of length `N`. Entry `i` holds
    the SPME primary bin index `g_a · n_b · n_c + g_b · n_c + g_c`
    of atom `i`.
  - `bin_atom_counts: CudaSlice<u32>` of length `M`. Per-bin atom
    histogram, zeroed before each sort.
  - `bin_atom_offsets: CudaSlice<u32>` of length `M + 1`. Exclusive
    prefix scan of `bin_atom_counts`; bin `b` occupies sorted-index
    positions `[bin_atom_offsets[b], bin_atom_offsets[b + 1])`.
  - `bin_atom_cursor: CudaSlice<u32>` of length `M`. Per-bin atomic
    cursor used during the scatter stage; zeroed before each sort.
  - `sorted_atom_index: CudaSlice<u32>` of length `N`. The result
    of the sort: `sorted_atom_index[t] = i` means the atom processed
    at sorted slot `t` was originally atom `i`. Consumed by
    `spme_spread_fixed_point` and `spme_force_gather` to walk atoms
    in spatial-bin order. Initialised at slot construction to the
    identity permutation `sorted_atom_index[t] = t`, so the first
    `compute()` call works even before the first sort runs.
  - `sort_scan_block_totals: Vec<CudaSlice<u32>>` — the multi-level
    scan-stack buffers consumed by the shared
    `prefix_scan_cell_counts` family (see `neighbor-list.md`) when
    it operates on a histogram of length `M`.
- `cached_neighbor_list_generation: u64`. The neighbour-list rebuild
  generation observed at the last sort. The slot re-runs the sort
  pipeline when the framework reports a generation strictly greater
  than this value (see *Atom spatial pre-sort* for the trigger
  protocol).

### Atom spatial pre-sort <!-- rq-06f1edf2 -->

The reciprocal-space slot walks atoms in spatial-bin order during
spread and gather. Sorting concentrates each warp's atomic writes
(spread) and grid reads (gather) on neighbouring grid cells, so the
hot grid cells stay in L1/L2 across consecutive warps.

The sorted order is materialised as a permutation
`sorted_atom_index: CudaSlice<u32>` of length `N`. Each entry
`sorted_atom_index[t] = i` names the original atom index `i` to be
processed at sorted slot `t`. Spread and gather kernels read this
permutation with their block-id and use `i` to address
`positions[i]`, `charges[i]`, and (for gather) the per-particle
slot-output cells `slot_force_*[i]` / `slot_energy[i]` /
`slot_virial[i]`.

**Trigger protocol.** The sort runs at the start of every
`SpmeReciprocalState::compute()` call where the framework's
neighbour-list rebuild generation is strictly greater than the
slot's `cached_neighbor_list_generation`. The slot updates
`cached_neighbor_list_generation` to the current value when the
sort completes. The first `compute()` call after slot construction
sees a generation > 0 (every neighbour-list build advances the
counter, including the initial build), so the first sort always
fires.

**Sort algorithm.** A count-sort over the SPME primary-bin key:

1. `spme_compute_bin_key` — one thread per atom. Reads particle
   `i`'s wrapped position, computes the primary bin
   `(g_a, g_b, g_c)` via the same `spread_per_particle_setup`
   helper the spread kernel uses, and writes
   `atom_bin_key[i] = (g_a · n_b + g_b) · n_c + g_c`. Atomically
   increments `bin_atom_counts[atom_bin_key[i]]` (each atom emits
   exactly one `+1`, so the final per-bin count is independent of
   atomic-completion order).

2. `prefix_scan_cell_counts` (the family in `neighbor-list.md`) —
   produces `bin_atom_offsets` of length `M + 1` from
   `bin_atom_counts` of length `M`.

3. `scatter_atoms_into_cells` (reused from `neighbor-list.md`) — one
   thread per atom. Thread `i` reads `b = atom_bin_key[i]`, computes
   `t = bin_atom_offsets[b] + atomicAdd(&bin_atom_cursor[b], 1)`,
   and writes `sorted_atom_index[t] = i`. The atomic-completion
   order of cursor increments within a bin is non-deterministic,
   so the in-bin order of `sorted_atom_index` is non-deterministic
   on a single sort run.

   **In-bin determinism.** A follow-up canonicalising pass
   (`sort_cells_by_particle_id`, reused from `neighbor-list.md`,
   one thread per bin) sorts each bin's `sorted_atom_index` slice
   in strictly ascending atom-index order. After this pass, two runs
   with byte-identical inputs produce a byte-identical
   `sorted_atom_index` regardless of the non-deterministic scatter
   order.

**Determinism.** Combined with the i64 atomic-add associativity of
the spread (see *Reproducibility*), the canonicalised
`sorted_atom_index` guarantees byte-identical `rho_fixed` and
`slot_force_*` across runs on the same GPU regardless of warp /
atomic completion order.

**Per-step state reset.** `bin_atom_counts` and `bin_atom_cursor`
accumulate via `atomicAdd` inside the bin-key and scatter kernels,
so they are zeroed via `memset_zeros` at the start of every sort.
`atom_bin_key`, `bin_atom_offsets`, `sorted_atom_index`, and the
scan-stack buffers are overwritten every entry, so they need no
explicit reset.

### Charge spreading <!-- rq-382a6e66 -->

The charge density on the FFT grid is the same quantity for every
implementation:

```text
rho[g] = Σ_i q_i · M_p(s_a_i · n_a - g_a) · M_p(s_b_i · n_b - g_b) · M_p(s_c_i · n_c - g_c)
```

where `s_i = (s_a_i, s_b_i, s_c_i)` are particle `i`'s fractional
coordinates (computed via `SimulationBox::fractional_coords` on the
wrapped position), `M_p` is the 1D cardinal B-spline of order `p`, and
the sum runs over every particle whose support intersects `g` —
equivalently, every particle whose primary bin lies within the box of
`p × p × p` bins centred on `g`.

The slot computes this via a two-stage **fixed-point atomic-add**
pipeline that runs every step. Both stages execute on the default
stream and are deterministic under the same-GPU run-to-run contract.

The fixed-point representation maps a real value `v` to the integer
`v_fixed = (i64)(v × 2^32)`. With charges bounded by O(1 e) and 
B-spline weights bounded by 1, a single contribution maps to a value
of magnitude at most a few × 10⁹, and the worst-case accumulated cell
sum stays well under `i64::MAX ≈ 9.2 × 10¹⁸`. Integer atomic addition
on i64 is exactly associative on the same GPU regardless of
atomic-completion order, so the accumulated fixed-point grid is
byte-identical across runs with byte-identical inputs.

The two stages:

1. **Per-step state reset.** `rho_fixed` is zeroed via the device's
   `memset_zeros` before the spread launches. The reset and every
   later kernel run on the same default stream, so the per-stream
   ordering supplies the read-after-write guarantee with no explicit
   synchronisation.

2. **Per-particle fixed-point scatter.** `spme_spread_fixed_point`
   runs one warp per sorted slot with 8 warps per block (256 threads)
   and grid `ceil(N / 8)`. Lane 0 of each warp reads
   `i = sorted_atom_index[t]` where `t` is the sorted slot for this
   warp (`t = blockIdx.x · 8 + warp_id_in_block`), then reads
   particle `i`'s wrapped position and charge `q_i`, computes the
   fractional coordinates `(s_a, s_b, s_c)`, the primary bin
   `(g_a, g_b, g_c)`, the fractional offsets `(t_a, t_b, t_c)`, and
   the per-axis 1D B-spline weights `wa[0..p]`, `wb[0..p]`,
   `wc[0..p]`; the per-axis weights, primary bin, and `q_i` are
   broadcast to every lane via `__shfl_sync`. Consecutive sorted
   slots address atoms with nearby primary bins, so the lane-stride
   `atomicAdd<i64>` writes from consecutive warps cluster on
   neighbouring `rho_fixed` cache lines.

   Each of the 32 lanes handles `⌈p³ / 32⌉` of the `p³` grid
   contributions. Lane `l` iterates the contribution index
   `k = l, l + 32, l + 64, …` while `k < p³`. For its assigned
   `k = d_a · p² + d_b · p + d_c`, the lane:
   - Computes the wrapped grid-cell index
     `g = ((g_a + d_a) mod n_a · n_b + (g_b + d_b) mod n_b) · n_c
          + (g_c + d_c) mod n_c`.
   - Computes the f32 contribution value
     `v = q_i · wa[d_a] · wb[d_b] · wc[d_c]`.
   - Converts to fixed-point
     `v_fixed = (i64) rintf(v · 2^32)` and issues
     `atomicAdd(&rho_fixed[g], v_fixed)`.

   Round-to-nearest in the f32 → i64 conversion (CUDA's
   `__float2ll_rn` or `rintf` + cast) keeps the per-contribution
   rounding direction deterministic. The atomic-completion order of
   the `N · p³` adds is non-deterministic, but i64 addition is
   associative, so the final `rho_fixed` is byte-identical across
   runs.

3. **Fixed-point → f32 conversion.** `spme_spread_finish` runs one
   thread per grid cell with block size 256 and grid `ceil(M / 256)`.
   Thread `c` reads `rho_fixed[c]` and writes
   `rho[c] = (f32) rho_fixed[c] × FIXED_POINT_SCALE_INV` where
   `FIXED_POINT_SCALE_INV = 1.0f / (float)(1ULL << 32)`. Each cell is
   written by exactly one thread; no atomics, no inter-thread
   communication.

When `particle_count == 0`, the spread kernel launch is skipped, the
fixed-point grid stays at its post-`memset_zeros` zero state, and
`spme_spread_finish` produces an all-zero `rho`.

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

The precomputed influence function for complex grid index
`k = (k_a, k_b, k_c)` (with `k_c < n_c/2 + 1`) is

```text
m_a = (k_a ≤ n_a / 2) ? k_a : k_a − n_a
m_b = (k_b ≤ n_b / 2) ? k_b : k_b − n_b
m_c = (k_c ≤ n_c / 2) ? k_c : k_c − n_c    # always k_c since k_c < n_c/2 + 1

K = 2π · (m_a · b_a + m_b · b_b + m_c · b_c)
K2 = |K|²

G[k] = (k_C / V_box) · (4π / K²) · exp(−K² / (4 α²))
       · b_factors_a[k_a] · b_factors_b[k_b] · b_factors_c[k_c]

virial_factor[k] = G[k] · (1 − K² / (2 α²))
```

where `b_a`, `b_b`, `b_c` are the rows of the reciprocal lattice matrix
`H^(-T)` (derived from the current simulation box), `V_box = lx · ly · lz`
is the box volume, `k_C` is the Coulomb prefactor (1 in atomic units),
and the `b_factors_*` are the SPME B-spline correction terms:

```text
b_factors_d[k] = |Σ_{j=0..p-1} M_p(j + 1) · exp(2π i · k · j / n_d)|⁻²
```

The `k = (0, 0, 0)` slot is set to zero in both `influence_G` and
`virial_factor`, implementing tinfoil boundary conditions and removing
the (unphysical) infinite background-charge contribution.

`b_factors_*` are independent of the box and depend only on the grid
dimensions and spline order; they are computed once on the host at slot
construction via the Cox-de Boor B-spline recursion, uploaded to the
slot's device buffers, and never rebuilt.

`influence_G` and `virial_factor` are populated on device by a
`spme_recip_compute_influence` kernel that runs on `recip_stream`. The
kernel takes the current box lattice as scalar kernel arguments
(`lx, ly, lz, xy, xz, yz`), the precomputed device-resident
`b_factors_*` buffers, the grid dimensions, and `α`. One thread per
complex grid cell evaluates the formulae above; threads do not
communicate. All inner arithmetic uses `double` precision (the
reciprocal-lattice inversion, the K-vector dot product, the exponential,
and the B-spline-correction product) and the final value is cast to the
storage `Real` at the device-store site, matching the precision policy
of every other f32 kernel that performs accuracy-sensitive transcendental
arithmetic on device.

`spme_recip_compute_influence` runs:

1. **At slot construction**, after the `b_factors_*` buffers have been
   uploaded, to populate `influence_G` and `virial_factor` for the
   initial box. The slot's `cached_box_generation` is set to
   `sim_box.generation()` at this point.
2. **At the start of every `SpmeReciprocalState::compute()` call where
   `sim_box.generation() != self.cached_box_generation`**, before
   the bin-list pre-step and before the recip-stream's wait for the
   default stream. The launch updates `cached_box_generation`. When
   the generations match, the call is skipped and the prior
   `influence_G` / `virial_factor` are reused.

The cadence of recompute therefore tracks the C-rescale and other
barostats' box updates: NVT runs recompute exactly once (at
construction); NPT runs recompute every step the barostat fires.

When the kernel runs, the strict default-stream ordering (the
influence recompute is dispatched before any later reciprocal kernel
on the same stream; see *Launch configuration*) ensures that
downstream consumers see the updated buffers before reading them.

### Influence-function multiply, virial partial reduction, and inverse FFT <!-- rq-95385a9d -->

`spme_recip_apply_influence` is a fused per-cell kernel that, in one
pass over the complex grid, both produces the smoothed reciprocal
potential `V_hat` and accumulates the per-cell virial contribution
into a small per-block partial-sums buffer.

Per thread (one thread per complex cell, `k = (k_a, k_b, k_c)`):

1. Read `rho_hat[k]`, `influence_G[k]`, `virial_factor[k]`.
2. Compute `V_hat[k] = influence_G[k] · rho_hat[k]`, including a
   write of zero for `k = (0, 0, 0)` (the `k = 0` slots of both
   `influence_G` and `virial_factor` are pre-zeroed by
   `spme_recip_compute_influence`).
3. Compute the per-thread virial contribution
   `v_t = virial_factor[k] · |rho_hat[k]|²` (zero at `k = 0`).
4. Hold `v_t` in a per-thread register accumulator and participate
   in the block-level deterministic shared-memory pairwise tree
   that reduces all 256 lane contributions to a single block partial.
5. Lane 0 of each block writes the block partial to
   `virial_partials[block_id]`.

The kernel uses one 256-thread block per `ceil(M_complex / 256)`
blocks of the complex grid and reads / writes only `rho_hat`,
`influence_G`, `virial_factor`, `V_hat`, and `virial_partials`. No
atomics, no inter-block synchronisation. The block partial-sum tree
shape depends only on the fixed block size (256), so two runs on
the same GPU produce byte-identical `V_hat` and byte-identical
`virial_partials`.

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

Operationally: one thread per sorted slot. Each thread reads its
sorted-slot index `t = blockIdx.x · blockDim.x + threadIdx.x`,
resolves the original atom index `i = sorted_atom_index[t]`, and
then:

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
6. Writes the per-particle force and energy contributions to
   `slot_force_*[i]`, `slot_energy[i]`, and (after the
   deterministic virial reduction) `slot_virial[i]` — the output
   addresses are by the *original* atom index `i`, not the sorted
   slot `t`, so the slot-output layout remains in canonical
   particle-index order regardless of how the sort permutes the
   processing order.

Each particle is written by exactly one thread; no atomics, no race
conditions. Summation order over the `p³` grid points within a particle
is fixed in `(d_a, d_b, d_c)` lexicographic order, so the contribution
ordering is deterministic.

Consecutive sorted slots address atoms with nearby primary bins, so
the grid reads from `V[g]` across consecutive threads cluster on
neighbouring cache lines.

### Reciprocal-space virial <!-- rq-ce4590c1 -->

The reciprocal-space slot computes the scalar virial trace from the
reciprocal grid:

```text
W_recip = 0.5 · Σ_{k ≠ 0} virial_factor[k] · |rho_hat[k]|²
```

where `virial_factor[k] = G[k] · (1 − K² / (2 α²))` is precomputed in
the device-resident `virial_factor` buffer alongside `influence_G` (see
*Influence function*). The Coulomb prefactor `k_C` and the `(4π/K²)`
Greens-function term are already folded into `G[k]`; the `0.5` outside
the sum is the Ewald half-sum convention.

The per-cell contributions `virial_factor[k] · |rho_hat[k]|²` are
folded into the per-block partial sums in
`virial_partials` directly inside the fused
`spme_recip_apply_influence` kernel; no length-`M_complex` per-cell
virial buffer is materialised in global memory (see
*Influence-function multiply, virial partial reduction, and inverse
FFT* above). The scalar `W_recip / N` is computed on device by the
`spme_recip_reduce_partials` kernel, dispatched on the default
stream immediately after the influence-multiply pass. A single
256-thread block sums `virial_partials[0 .. num_blocks − 1]` with
a strided per-thread accumulator followed by a deterministic
left-to-right pairwise tree in shared memory (the same shape as
`barostat::virial_sum_reduce` and
`nose_hoover::kinetic_energy_reduce`). Lane 0 of the block
multiplies the reduced sum by `0.5 / N` and writes the result to
the device-resident single-element scalar `w_per_particle_virial`.
Two runs on the same GPU with byte-identical inputs produce a
byte-identical `w_per_particle_virial[0]`.

The scalar is distributed per particle by equal division inside
`spme_force_gather`: each particle reads `w_per_particle_virial[0]`
once and writes it to its own `virials[i]` slot. Summing `virials` over
all particles yields the system total `W_recip`. The per-particle
attribution has no individual physical meaning; the convention exists
only so the SoA `virials: Vec<f32>` layout sums correctly.

The real-space slot accumulates the per-pair virial contribution
`0.5 · scale · factor · r²` into the warp's `p_w` register
accumulator, summed and written to `slot_virial[i]` by the
warp-tree reduction (see `pair-force-kernel.md`).

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
  spread / gather kernels and the device-resident cuFFT workspace
  buffer used by both transforms (see *Reciprocal-space pipeline*).
  Every kernel and cuFFT call the slot dispatches runs on the device's
  default stream; the slot owns no secondary streams or cross-stream
  events.

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

- `spme_real_pair_force(particle_buffers, output, sim_box, alpha, r_cut_real, atom_excl_offsets, atom_excl_partners, atom_excl_coul_scales, neighbor_list, neighbor_counts, max_neighbors, level) -> Result<(), GpuError>` <!-- rq-f735ea05 -->
  Selects the kernel variant based on `level`:
  `AggregateLevel::ForcesOnly` dispatches to `spme_real_pair_force_f`;
  `AggregateLevel::ForcesAndScalars` dispatches to
  `spme_real_pair_force_fev`. Writes per-particle output directly into
  `output`'s slot rows.

- `spme_recip_compute_influence(kernels, b_factors_a, b_factors_b, b_factors_c, influence_g, virial_factor, sim_box, grid, alpha, k_coulomb, m_complex) -> Result<(), GpuError>` <!-- rq-99d3d796 -->
  Launches `spme_recip_compute_influence` on the device's default
  stream. Writes `influence_g` and `virial_factor`. The host call
  returns as soon as the launch has been enqueued; no host-side
  computation of the influence function is performed.

- `spme_atom_sort(particle_buffers, sim_box, spme_state) -> Result<(), GpuError>` <!-- rq-a1b761fc -->
  Rebuilds `spme_state.sorted_atom_index` on the default stream when
  the framework's neighbour-list rebuild generation is strictly
  greater than `spme_state.cached_neighbor_list_generation`. The
  pipeline is:
  1. Device-side `memset_zeros` on `bin_atom_counts` and
     `bin_atom_cursor`.
  2. `spme_compute_bin_key` — one thread per particle; writes
     `atom_bin_key[i]` and atomically increments
     `bin_atom_counts[bin]`.
  3. The `prefix_scan_cell_counts` family (see `neighbor-list.md`) —
     produces `bin_atom_offsets` from `bin_atom_counts`.
  4. `scatter_atoms_into_cells` (reused from `neighbor-list.md`) —
     one thread per particle; writes `sorted_atom_index[t]` for the
     slot
     `t = bin_atom_offsets[bin] + atomicAdd(&bin_atom_cursor[bin], 1)`.
  5. `sort_cells_by_particle_id` (reused from `neighbor-list.md`) —
     one thread per bin; insertion-sorts the bin's slice of
     `sorted_atom_index` in strictly ascending atom-index order.

  Updates `spme_state.cached_neighbor_list_generation` to the
  framework's current value on success. Returns `Ok(())` immediately
  once every kernel has been enqueued. When the generation has not
  advanced, the function returns `Ok(())` without launching any
  kernels.

- `spme_spread_charges(particle_buffers, spme_state) -> Result<(), GpuError>` <!-- rq-a1b761fa -->
  Launches the two-stage fixed-point charge-spread pipeline on the
  default stream:
  1. Device-side `memset_zeros` on `spme_state.rho_fixed` (length
     `M`, i64) to clear the previous step's accumulation.
  2. `spme_spread_fixed_point` — one warp per sorted slot. Lane 0
     reads `i = sorted_atom_index[t]` to resolve the atom index
     before reading the atom's position and charge; each lane
     issues `⌈p³ / 32⌉` `atomicAdd<i64>` operations into
     `rho_fixed`, totalling `N · p³` atomic adds per step.
  3. `spme_spread_finish` — one thread per grid cell; converts
     `rho_fixed[c]` to `rho[c] = (f32) rho_fixed[c] · 2^-32`.

  Writes `spme_state.rho`. Returns `Ok(())` immediately once every
  kernel has been enqueued; no host-side computation is performed.
  When `particle_count == 0`, the fixed-point spread kernel is
  skipped (the `memset_zeros` and `spme_spread_finish` still run),
  so `rho` is produced as all zeros.

- The R2C forward transform `rho → rho_hat` is invoked via <!-- rq-24e36eba -->
  `SpmeReciprocalGrid::forward_plan.execute(&rho, &mut rho_hat)`, where
  `forward_plan: Plan3dR2C` is constructed in
  `SpmeReciprocalGrid::new`, has its work-area pre-bound to the slot's
  device-resident workspace via `cufftSetWorkArea`, and is pre-bound
  to the default stream via `cufftSetStream`. The call returns
  `Result<(), CuFftError>`.

- `spme_recip_apply_influence(spme_state) -> Result<(), GpuError>` <!-- rq-5ec02591 -->
  Multiplies `rho_hat[k] *= G[k]` in place on the default stream,
  computes the per-thread virial contribution
  `virial_factor[k] · |rho_hat[k]|²`, reduces those contributions
  block-by-block via a deterministic shared-memory pairwise tree, and
  writes the per-block partial sums to `virial_partials`. Operates
  in a single kernel pass over the complex grid; no length-`M_complex`
  per-cell virial buffer is materialised in global memory.

- `spme_recip_reduce_partials(kernels, virial_partials, w_per_particle_virial, num_blocks, n_particles) -> Result<(), GpuError>` <!-- rq-e0d010c0 -->
  Launches `spme_recip_reduce_partials` on the default stream. Writes
  `w_per_particle_virial[0] = (0.5 / N) · Σ virial_partials[b]` via
  the single-block deterministic pairwise tree described in
  *Reciprocal-space virial*. No host download.

- The C2R inverse transform `rho_hat → V` is invoked via <!-- rq-a98abc35 -->
  `SpmeReciprocalGrid::inverse_plan.execute(&rho_hat, &mut V)`, where
  `inverse_plan: Plan3dC2R` is constructed in
  `SpmeReciprocalGrid::new`, has its work-area pre-bound to the slot's
  device-resident workspace via `cufftSetWorkArea`, and is pre-bound
  to the default stream via `cufftSetStream`. The call returns
  `Result<(), CuFftError>`.

- `spme_force_gather(particle_buffers, spme_state, slot_output) -> Result<(), GpuError>` <!-- rq-c6f6a13c -->
  Launches the force-gather kernel. Threads address atoms via
  `i = sorted_atom_index[t]` for `t = blockIdx.x · blockDim.x +
  threadIdx.x` so consecutive threads work on spatially-clustered
  atoms; per-atom output is written to the canonical particle-index
  positions `slot_force_*[i]`, `slot_energy[i]`, `slot_virial[i]`.
  Writes per-particle force, reciprocal energy, and (after the
  deterministic virial reduction) per-particle virial.

- `cufft_determinism_smoke_test(device: &Arc<CudaDevice>) -> Result<(), CuFftError>` <!-- rq-d880c228 -->
  Used by `init_device` when SPME is enabled. Returns `Err` on a
  byte-difference between two consecutive R2C transforms.

### CUDA kernels <!-- rq-7225b86f -->

`kernels/spme_real.cu` declares two `extern "C"` kernels (signatures
shown in the *Real-space CUDA kernels* section above):

```c
extern "C" __global__ void spme_real_pair_force_f(/* ... */);
extern "C" __global__ void spme_real_pair_force_fev(/* ... */);
```

`kernels/spme_recip.cu` declares:

```c
extern "C" __global__ void spme_recip_compute_influence(
    const float *b_factors_a,           // length n_a
    const float *b_factors_b,           // length n_b
    const float *b_factors_c,           // length n_c
    float *influence_G,                 // length M_complex
    float *virial_factor,               // length M_complex
    const float *lattice,           // length 6: [lx, ly, lz, xy, xz, yz]
    unsigned int n_a, unsigned int n_b, unsigned int n_c,
    float k_coulomb,
    float alpha,
    unsigned int m_complex);

extern "C" __global__ void spme_recip_reduce_partials(
    const float *virial_partials,       // length num_blocks
    float *w_per_particle_virial,       // length 1
    unsigned int num_blocks,
    float scale);                       // 0.5 / N

extern "C" __global__ void spme_spread_fixed_point(
    const float        *positions_x,
    const float        *positions_y,
    const float        *positions_z,
    const float        *charges,
    const unsigned int *sorted_atom_index,  // length n
    const float        *lattice,            // length 6
    unsigned int n_a, unsigned int n_b, unsigned int n_c,
    unsigned int spline_order,
    long long   *rho_fixed,                 // length M (i64)
    unsigned int n);

extern "C" __global__ void spme_spread_finish(
    const long long *rho_fixed,     // length M
    float           *rho,           // length M
    unsigned int M);

extern "C" __global__ void spme_compute_bin_key(
    const float  *positions_x,
    const float  *positions_y,
    const float  *positions_z,
    const float  *lattice,           // length 6
    unsigned int  n_a, unsigned int n_b, unsigned int n_c,
    unsigned int *atom_bin_key,      // length n
    unsigned int *bin_atom_counts,   // length M (atomicAdd)
    unsigned int  n);

// The bin-key scatter (step 3) reuses `scatter_atoms_into_cells` from
// `kernels/neighbor.cu`: thread `i` reads its bin index, atomically
// increments the per-bin write cursor, and writes the resulting sorted
// slot. The in-bin canonicalisation pass (step 4) reuses
// `sort_cells_by_particle_id`: one thread per bin runs an insertion
// sort over its slice in strictly ascending atom-index order. See
// `rqm/forces/neighbor-list.md` for both kernel signatures.

extern "C" __global__ void spme_recip_apply_influence(
    const float *influence_G,           // length M_complex
    const float *virial_factor,         // length M_complex
    float *rho_hat_real,   // interleaved real and imag parts
    float *rho_hat_imag,
    float *virial_partials,             // length num_blocks
    unsigned int m_complex);

extern "C" __global__ void spme_force_gather(
    const float        *positions_x,
    const float        *positions_y,
    const float        *positions_z,
    const float        *charges,
    const float        *V,
    const unsigned int *sorted_atom_index,  // length n
    const float        *lattice,            // length 6
    unsigned int n_a, unsigned int n_b, unsigned int n_c,
    unsigned int spline_order,
    float k_coulomb,
    float *slot_force_x,
    float *slot_force_y,
    float *slot_force_z,
    float *slot_energy,
    unsigned int n);
```

No length-`M_complex` per-cell virial buffer exists; the per-block
partial sums produced by `spme_recip_apply_influence` flow directly
into `spme_recip_reduce_partials`. Both reductions use deterministic
fixed-topology shared-memory pairwise trees whose shape depends only
on the launch block size (256), not on thread scheduling.

### Launch configuration <!-- rq-03d9869d -->

- `spme_real_pair_force_f` / `spme_real_pair_force_fev`: 256 threads <!-- rq-eb9e5cc3 -->
  per block (8 warps × 32 lanes), grid `ceil(N / 8)`. Matches the
  common pair-force pattern (see `pair-force-kernel.md`).
- `spme_recip_compute_influence`: one thread per complex grid cell, <!-- rq-17b52850 -->
  block size 256, grid `ceil(M_complex / 256)`. No shared memory.
- `spme_spread_fixed_point`: one warp per sorted slot, block size 256 <!-- rq-996edcaf -->
  (8 warps × 32 lanes), grid `ceil(N / 8)`. No static shared memory
  (per-axis B-spline weights, primary bin, and `q_i` are broadcast
  lane-to-lane via `__shfl_sync`). Lane 0 reads
  `i = sorted_atom_index[t]` before reading the atom's position and
  charge. Each of the 32 lanes performs `⌈p³ / 32⌉` `atomicAdd<i64>`
  operations into `rho_fixed`.
- `spme_spread_finish`: one thread per grid cell, block size 256, <!-- rq-d9403350 -->
  grid `ceil(M / 256)`. No shared memory. Reads `rho_fixed[c]`,
  writes `rho[c] = (f32) rho_fixed[c] · 2^-32`.
- `spme_compute_bin_key`: one thread per particle, block size 256, <!-- rq-7594b1fc -->
  grid `ceil(N / 256)`. No shared memory. Computes the SPME primary
  bin index per atom and atomically increments `bin_atom_counts`.
- `scatter_atoms_into_cells` (reused): one thread per particle, block <!-- rq-ab1063b3 -->
  size 256, grid `ceil(N / 256)`. No shared memory. Atomic cursor
  increment into `bin_atom_cursor` to determine the sorted slot.
- `sort_cells_by_particle_id` (reused): one thread per bin, block size <!-- rq-1b747c20 -->
  256, grid `ceil(M / 256)`. No shared memory. Per-bin insertion sort
  over the bin's slice of `sorted_atom_index` in strictly ascending
  atom-index order; the slot's per-bin slice length is bounded by the
  worst-case number of atoms whose primary bin coincides with any
  given grid cell, which is small under SPME's grid-density convention
  (~ a few atoms per cell).
- `spme_recip_apply_influence`: one thread per complex grid cell, <!-- rq-b82694ec -->
  block size 256, grid `ceil(M_complex / 256)`,
  `__shared__ Real partial[256]` for the per-block virial reduction.
  Per thread: read `rho_hat[k]`, `influence_G[k]`, `virial_factor[k]`,
  write `V_hat[k]`, accumulate the per-thread virial contribution into
  shared memory, participate in the deterministic pairwise tree.
  Lane 0 of each block writes one entry of `virial_partials`.
- `spme_recip_reduce_partials`: single block of 256 threads, <!-- rq-3a41a142 -->
  `__shared__ Real partial[256]`. Strided per-thread accumulator
  followed by deterministic left-to-right pairwise tree. Only lane 0
  writes the output (one scalar to `w_per_particle_virial`).
- `spme_force_gather`: one thread per particle, block size 256, grid <!-- rq-35b76155 -->
  `ceil(N / 256)`.

Stream assignment:

- Every SPME kernel and cuFFT call dispatched by either SPME slot — <!-- rq-44cce069 -->
  `spme_real_pair_force_*`, `spme_force_gather`,
  `spme_recip_compute_influence` (when triggered by a box-generation
  change), the atom-sort pipeline (`spme_compute_bin_key`,
  the prefix-scan family, `scatter_atoms_into_cells`,
  `sort_cells_by_particle_id`, plus the device-side `memset_zeros`
  on `bin_atom_counts` and `bin_atom_cursor` — fired when triggered
  by a neighbour-list rebuild generation change),
  the `spme_spread_fixed_point` and `spme_spread_finish`
  kernels (plus the device-side `memset_zeros` on `rho_fixed`),
  the cuFFT R2C transform, `spme_recip_apply_influence`, the cuFFT
  C2R transform, and `spme_recip_reduce_partials` — runs on the
  device's default stream carried by `particle_buffers.device`.
  Both cuFFT plans are bound to the default stream via
  `cufftSetStream` once at slot construction and are never rebound.

The slot owns no secondary CUDA streams and no CUDA events. The
ordering of writes and reads within a step's reciprocal pipeline
(spread → R2C → influence-multiply → virial-finalize → C2R →
force-gather) is the natural launch order on the default stream:
every later kernel reads only buffers written by an earlier kernel
on the same stream, so CUDA's implicit per-stream ordering supplies
the producer-consumer guarantee with no explicit synchronisation.

## Reproducibility <!-- rq-20530653 -->

SPME on HeddleMD is bit-exact GPU-vs-GPU when run on the same hardware
with identical inputs. Eight components carry the reproducibility
invariant:

1. **Atom spatial pre-sort.** `spme_compute_bin_key` evaluates the
   primary bin per atom from positions alone — no atomics in the
   key computation, no inter-thread communication. The histogram
   atomicAdd on `bin_atom_counts` reduces to integer addition of
   `+1`s, whose result is independent of completion order. The
   `scatter_atoms_into_cells` stage uses non-deterministic
   `atomicAdd` cursors into `bin_atom_cursor`, but the subsequent
   `sort_cells_by_particle_id` insertion sort fixes each bin's slice
   of `sorted_atom_index` to strictly ascending atom-index order.
   Two runs on the same GPU with byte-identical positions therefore
   produce a byte-identical `sorted_atom_index`.
2. **Charge spread.** The per-particle B-spline weight computation
   inside `spme_spread_fixed_point` is fully per-lane and
   deterministic — lane 0 reads `i = sorted_atom_index[t]` and
   broadcasts the resolved atom index and per-axis weights via
   `__shfl_sync`, every lane evaluates the same expression for its
   assigned `(d_a, d_b, d_c)`, and the f32 → i64 conversion uses
   round-to-nearest. The atomic adds into `rho_fixed` are on i64,
   which is an exactly associative operation: the final cell value
   `rho_fixed[g] = Σ v_fixed` is independent of the order in which
   the `atomicAdd<i64>` operations complete. Two runs on the same
   GPU with byte-identical inputs therefore produce a byte-identical
   `rho_fixed`. The `spme_spread_finish` pass writes
   `rho[c] = (f32) rho_fixed[c] · 2^-32` per cell with no
   inter-thread communication, so `rho` is byte-identical too.
3. **cuFFT.** Deterministic for fixed plan dimensions, single-stream
   usage, and the same hardware. "Single-stream usage" is satisfied
   because both cuFFT plans are bound once at slot construction to
   the device's default stream via `cufftSetStream` and are never
   rebound, and because their work-area pointer is fixed at
   construction via `cufftSetAutoAllocation(plan, 0)` +
   `cufftSetWorkArea(plan, workspace)` so cuFFT does not
   transparently reallocate scratch memory at execution time. The
   `cufft_determinism_smoke_test` run at `init_device` time validates
   the contract on the host's specific cuFFT version.
4. **Influence function recompute.** `spme_recip_compute_influence`
   runs one thread per complex grid cell with no inter-thread
   communication. Inner arithmetic in `double` precision. The kernel
   fires whenever the slot observes a changed `sim_box.generation()`,
   producing byte-identical `influence_G` and `virial_factor` for
   byte-identical box lattices on the same GPU.
5. **Influence-function multiply and virial partial reduction.**
   `spme_recip_apply_influence` runs one thread per complex grid
   cell; no atomics; no inter-thread reads on the multiply portion.
   The per-block partial-sum reduction over the per-thread virial
   contributions uses a deterministic fixed-topology shared-memory
   pairwise tree whose shape depends only on the launch block size
   (256), so two runs on the same GPU produce byte-identical
   `V_hat` and byte-identical `virial_partials`.
6. **Virial partial-sums reduction.** `spme_recip_reduce_partials`
   runs a single block of 256 threads with a strided per-thread
   accumulator and a deterministic left-to-right pairwise tree in
   shared memory. Two runs with byte-identical `virial_partials`
   produce a byte-identical `w_per_particle_virial[0]`.
7. **Force gather.** One thread per sorted slot; each thread reads `p³`
   grid points in fixed `(d_a, d_b, d_c)` lexicographic order. No
   atomics. The per-particle virial uses the equal-division attribution
   `W_recip / N` (the slot's `compute()` distributes the scalar
   identically to every particle, so the SoA convention is preserved
   regardless of summation order).
8. **Two-stream model.** The reciprocal pipeline runs on
   `recip_stream`; the real-space slot, the force-gather kernel, and
   every non-SPME slot's kernel run on the default stream. The two
   streams write to disjoint device buffers — `recip_stream` writes
   `rho`, `rho_hat`, `V`, and `virial_per_cell`; the default stream
   writes the slot-output buffers and `forces_*`. Cross-stream ordering
   is enforced by the two events
   `default_ready_event` and `recip_ready_event` recorded at
   deterministic points in `contribute()` and waited on at
   deterministic points in `contribute()` / `reduce()`; the
   wait edges are independent of host thread scheduling. Two runs with
   identical inputs on the same GPU therefore observe byte-identical
   writes from both streams.

The reciprocal-virial scalar reduction must use deterministic
fixed-topology tree reductions (`spme_recip_apply_influence`'s
per-block tree plus `spme_recip_reduce_partials`' final-block tree
in shared memory). The implementation must not use unordered
atomic-add or any non-deterministic device-side reduction.

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
- Softening the bit-exact GPU-vs-GPU determinism guarantee on the
  reciprocal-space pipeline in exchange for additional wall-clock
  savings. The spread and gather kernels run one thread per grid
  point (spread) and one thread per particle (gather) with no
  atomics, in fixed iteration order over their per-thread input
  ranges, and the per-block reduction trees in
  `spme_recip_apply_influence` and `spme_recip_reduce_partials`
  use fixed shared-memory shapes. A faster implementation that
  relaxed any of these properties — for example one-thread-per-particle
  spread with `atomicAdd` into `rho`, or a non-deterministic
  reduction shape that depends on block-completion order — could
  reach materially lower wall-clock per step, at the cost of breaking
  the run-to-run byte-equality contract that `architecture.md`
  guarantees for the engine. That tradeoff is intentionally not
  taken here; the determinism guarantee is treated as load-bearing
  for the long-running production runs the engine is designed for.
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
    When the _fev variant of spme_real_pair_force is called
    Then slot_force_x[0], slot_force_y[0], slot_force_z[0], slot_energy[0], slot_virial[0] are all 0.0

  @rq-d150a16b
  Scenario: Real-space slot scales contribution by 0.5 when scale_coul = 0.5
    Given two oppositely charged particles at separation r = 3e-10 (< r_cut_real)
    And an ExclusionList listing the pair with scale_coul = 0.5
    When the _fev variant of spme_real_pair_force is called to obtain values_scaled
    And the _fev variant of spme_real_pair_force is called with an empty exclusion list to obtain values_unscaled
    Then values_scaled.slot_force_x[0] equals 0.5 * values_unscaled.slot_force_x[0] bit-for-bit
    And values_scaled.slot_energy[0] equals 0.5 * values_unscaled.slot_energy[0] bit-for-bit
    And values_scaled.slot_virial[0] equals 0.5 * values_unscaled.slot_virial[0] bit-for-bit

  @rq-675080c4
  Scenario: Real-space slot with scale_coul = 1.0 reproduces the un-excluded value
    Given two oppositely charged particles at separation r = 3e-10
    And an ExclusionList listing the pair with scale_coul = 1.0
    When the _fev variant of spme_real_pair_force is called to obtain values_explicit
    And the _fev variant of spme_real_pair_force is called with an empty exclusion list to obtain values_implicit
    Then values_explicit.slot_force_x[0] equals values_implicit.slot_force_x[0] bit-for-bit
    And values_explicit.slot_energy[0] equals values_implicit.slot_energy[0] bit-for-bit
    And values_explicit.slot_virial[0] equals values_implicit.slot_virial[0] bit-for-bit

  @rq-34329bda
  Scenario: A real-space exclusion entry on one pair does not attenuate other pairs
    Given a ParticleState of N=3 with positions p0=(0,0,0), p1=(2e-10,0,0), p2=(4e-10,0,0)
      and charges (+1e, -1e, +1e)
    And an ExclusionList listing only pair (0, 1) with scale_coul = 0.0
    When the _fev variant of spme_real_pair_force is called
    Then slot_force_x[0] equals the SPME real-space force on particle 0 due to particle 2 only
      (the (0, 1) contribution is suppressed; the (0, 2) contribution is unscaled)
    And slot_force_x[2] equals the SPME real-space force on particle 2 due to particles 0 and 1
      (no exclusion entry attenuates particle 2's contributions)

  @rq-af7018c0
  Scenario: Real-space slot produces zero outside r_cut_real
    Given two particles at separation greater than r_cut_real
    When the _fev variant of spme_real_pair_force is called
    Then slot_force_x[0], slot_force_y[0], slot_force_z[0], slot_energy[0], slot_virial[0] are all 0.0

  @rq-83088c2f
  Scenario: Real-space slot matches the closed-form erfc force for an isolated pair
    Given two unit-charge particles at separation r = 4.0e-10 inside the cutoff
    And alpha = 2.0e10
    When the _f variant of spme_real_pair_force is called
    Then slot_force_x[0] equals the closed-form
      k_C · q_i · q_j · (erfc(α r) · inv_r2 + (2 α / √π) · exp(-α² r²) · inv_r2) · dx · inv_r
      to within 1e-5 relative

  @rq-0caebe37
  Scenario: Real-space slot obeys Newton's third law for a non-boundary pair
    Given two particles at non-boundary separation
    When the _f variant of spme_real_pair_force is called
    Then slot_force_x[0] equals -slot_force_x[1] bit-exactly (Newton's third law for an isolated pair)

  # --- Reciprocal-space pipeline: atom spatial pre-sort ---

  @rq-0f592ab6
  Scenario: Sort fires on the first compute() call
    Given an SpmeReciprocalState immediately after construction
      with cached_neighbor_list_generation == 0
    And the framework's neighbour-list rebuild generation is 1 (or any value > 0)
    When SpmeReciprocalState::compute() is called
    Then the five-stage sort pipeline (memset_zeros, spme_compute_bin_key,
      prefix_scan_cell_counts family, scatter_atoms_into_cells, sort_cells_by_particle_id)
      launches exactly once on the default stream
    And the slot's cached_neighbor_list_generation equals the framework's
      generation after the call

  @rq-9ffb2eb3
  Scenario: Sort fires when the neighbour-list rebuild generation advances
    Given an SpmeReciprocalState whose cached_neighbor_list_generation matches
      the framework's current generation
    When the framework rebuilds the neighbour list (advancing its rebuild generation)
    And SpmeReciprocalState::compute() is called
    Then the sort pipeline launches exactly once during this call
    And the slot's cached_neighbor_list_generation equals the new value

  @rq-be54d0f6
  Scenario: Sort does not fire when generation is unchanged
    Given an SpmeReciprocalState whose cached_neighbor_list_generation matches
      the framework's current generation
    When SpmeReciprocalState::compute() is called twice in succession with
      no intervening neighbour-list rebuild
    Then the sort pipeline launches zero times during the second call
    And sorted_atom_index is byte-identical to its pre-second-call contents

  @rq-03cbd438
  Scenario: sorted_atom_index is a permutation of [0, N)
    Given an SpmeReciprocalState with N particles
    When the sort pipeline runs
    Then sorted_atom_index has length N
    And the set of values in sorted_atom_index is exactly {0, 1, ..., N - 1}
      (every original atom index appears exactly once)

  @rq-e59ee965
  Scenario: sorted_atom_index orders atoms by primary bin
    Given N atoms with known fractional positions
    When the sort pipeline runs
    Then for every t in 0..N - 1, atom_bin_key[sorted_atom_index[t]] is
      less than or equal to atom_bin_key[sorted_atom_index[t + 1]]
      (consecutive sorted slots have monotonically non-decreasing primary bin)

  @rq-c88ec28e
  Scenario: In-bin order is canonical (strictly ascending atom index)
    Given N atoms with at least one primary bin holding multiple atoms
    When the sort pipeline runs
    Then within each bin's slice of sorted_atom_index, the entries are in
      strictly ascending atom-index order (the canonicalising pass has
      fixed the order regardless of the non-deterministic scatter cursor)

  @rq-99cb2bd3
  Scenario: Two runs of the sort pipeline produce byte-identical sorted_atom_index
    Given two SpmeReciprocalState instances with identical configuration,
      identical particle positions, and identical particle charges
    When the sort pipeline runs on each
    Then dtoh(sorted_atom_index) is byte-identical between the two runs

  @rq-d4d66d02
  Scenario: Initial sorted_atom_index is the identity permutation
    Given an SpmeReciprocalState immediately after construction
    When sorted_atom_index is read before any compute() call
    Then sorted_atom_index[t] == t for every t in [0, N)
      (so the very first compute() call can run the spread / gather kernels
      even before the first sort completes, processing atoms in particle-index order)

  # --- Reciprocal-space pipeline: spread (fixed-point atomic-add) ---

  @rq-881559bd
  Scenario: Spread produces zero charge density at a grid point with no particle support
    Given one particle at position s = (0.5, 0.5, 0.5) in fractional coords
    And a grid point at fractional position (0.0, 0.0, 0.0) far from the particle's support
    When the spread pipeline runs
    Then rho[grid_index(0, 0, 0)] equals 0.0

  @rq-79291441
  Scenario: Spread sum integrates to total charge
    Given a system of N particles with total charge Q = Σ q_i
    When the spread pipeline runs
    Then Σ_g rho[g] equals Q to within 1e-5 relative tolerance
      (B-splines are partition-of-unity normalised)

  @rq-07297467
  Scenario: Spread is byte-identical regardless of input particle ordering
    Given two particles within each other's p-bin support, presented in two
      different input orderings (e.g. swapped IDs)
    When the spread pipeline runs
    Then the resulting `rho` is byte-identical between the two runs
      (`atomicAdd<i64>` is exactly associative, so the order in which
      atomic adds complete does not affect the final fixed-point grid)

  @rq-3c0beda9
  Scenario: Spread reduces to a single B-spline weight for one isolated particle
    Given exactly one particle with charge q at fractional position s
    When the spread pipeline runs
    Then rho[g] equals q · M_p(s_a·n_a - g_a) · M_p(s_b·n_b - g_b) · M_p(s_c·n_c - g_c)
      for every grid point g to within f32 round-off of the
      `rho_fixed[g] · 2^-32` conversion

  @rq-5b4519e9
  Scenario: rho_fixed is zeroed at the start of every step
    Given an SpmeReciprocalState whose `rho_fixed` holds non-zero values
      from a prior step
    When `spme_spread_charges` runs
    Then `rho_fixed` is observed to be zeroed by the device's
      `memset_zeros` before `spme_spread_fixed_point` issues any
      `atomicAdd<i64>`

  @rq-3dc94856
  Scenario: spme_spread_fixed_point issues exactly N · p³ atomicAdd<i64> operations
    Given N particles, spline order p, and an instrumented atomic counter
    When `spme_spread_fixed_point` runs
    Then the device-side counter records exactly `N · p³` `atomicAdd<i64>`
      invocations on `rho_fixed`

  @rq-6098e0c1
  Scenario: One particle's contribution at a single cell matches the fixed-point B-spline value
    Given exactly one particle with charge q at fractional position s,
      whose support uniquely covers grid cell g (no other particle
      contributes to g)
    When `spme_spread_fixed_point` runs
    Then `rho_fixed[g]` equals `(i64) rintf(q · w_a · w_b · w_c · 2^32)`
      bit-exactly, where `(w_a, w_b, w_c)` are the per-axis B-spline
      weights for the offset of g from the particle's primary bin

  @rq-09b6b539
  Scenario: spme_spread_finish converts rho_fixed to rho via the inverse scale
    Given a `rho_fixed[c]` populated with a known i64 value X
    When `spme_spread_finish` runs
    Then `rho[c]` equals `(f32)((double)X * 2^-32)` to within f32
      round-off, with no inter-thread communication during the
      conversion

  @rq-723b40a0
  Scenario: Two runs of the full spread pipeline produce byte-identical rho_fixed and rho
    Given two independently-constructed SpmeReciprocalState instances with
      identical configurations and identical particle positions and charges
    When the spread pipeline runs on each
    Then dtoh(rho_fixed) is byte-identical between the two runs
    And dtoh(rho) is byte-identical between the two runs

  @rq-c60c1d5f
  Scenario: Spread output is independent of sorted_atom_index permutation
    Given two SpmeReciprocalState instances A and B with identical
      configuration, positions, and charges
    And A's sorted_atom_index is the identity permutation
      (the slot's construction default before any sort runs)
    And B's sorted_atom_index is the canonical bin-sorted permutation
      (the post-sort permutation)
    When spread runs on each instance
    Then dtoh(rho_fixed) is byte-identical between the two runs
      (i64 atomic-add associativity guarantees that the final cell
      values do not depend on the warp processing order)

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

  # --- Reciprocal-space pipeline: influence function recompute ---

  @rq-9cee9bfd
  Scenario: Influence buffers are populated at slot construction
    Given an SpmeReciprocalState constructed with a sim_box B0 and parameters (alpha, grid, p)
    When the slot's construction finishes
    Then influence_G and virial_factor are device buffers of length M_complex
    And dtoh(influence_G) and dtoh(virial_factor) agree cell-by-cell with the
      analytical formula for B0 to within f32 round-off
    And the slot's cached_box_generation equals B0.generation()

  @rq-c8954d3e
  Scenario: Influence buffers are rebuilt when sim_box generation changes
    Given an SpmeReciprocalState whose cached_box_generation matches the current sim_box
    And a new sim_box B1 with a different generation counter
    When SpmeReciprocalState::compute() is called with B1
    Then spme_recip_compute_influence launches exactly once on recip_stream
      during this call
    And after the call, dtoh(influence_G) agrees with the analytical formula for B1
    And the slot's cached_box_generation equals B1.generation()

  @rq-c4d04411
  Scenario: Influence buffers are reused when sim_box generation is unchanged
    Given an SpmeReciprocalState whose cached_box_generation matches the current sim_box
    When SpmeReciprocalState::compute() is called with the same sim_box twice in succession
    Then spme_recip_compute_influence launches zero times during the second call
    And influence_G and virial_factor are byte-identical to their pre-second-call contents

  @rq-ff16a2c9
  Scenario: k=0 cell is zero in both influence_G and virial_factor after recompute
    Given any sim_box and any SpmeReciprocalState
    When spme_recip_compute_influence runs (at construction or on generation change)
    Then influence_G[grid_index(0, 0, 0)] equals 0.0
    And virial_factor[grid_index(0, 0, 0)] equals 0.0

  @rq-f8a66553
  Scenario: virial_factor[k] = G[k] · (1 - K² / (2 α²)) cell-by-cell
    Given an SpmeReciprocalState with parameters (alpha, grid, p) and a sim_box B
    When spme_recip_compute_influence has run for B
    Then for every cell k with k ≠ (0, 0, 0),
      virial_factor[k] equals influence_G[k] · (1 - K(k)² / (2 α²))
      to within f32 round-off

  @rq-4bd0b129
  Scenario: Influence recompute is deterministic across two independent slots
    Given two SpmeReciprocalState instances with identical (alpha, grid, p) and identical sim_box
    When spme_recip_compute_influence runs on each
    Then dtoh(influence_G) and dtoh(virial_factor) are byte-identical between the two slots

  @rq-ce68c21a
  Scenario: Influence recompute under a C-rescale barostat fires every step
    Given a 100-step NPT run with the C-rescale barostat enabled
    When the run completes
    Then spme_recip_compute_influence has launched 100 + 1 times
      (one at slot construction plus one per step's box update)

  @rq-8e6a1933
  Scenario: Influence recompute under NVT fires once
    Given a 100-step NVT run with no barostat
    When the run completes
    Then spme_recip_compute_influence has launched exactly once (at slot construction)

  # --- Reciprocal-space pipeline: fused apply-influence ---

  @rq-e5bf6fea
  Scenario: The k=0 entry is zeroed by the fused apply-influence kernel
    Given any non-zero rho_hat
    When spme_recip_apply_influence is called
    Then rho_hat[k=0] equals 0.0 + 0.0i after the kernel
      (tinfoil boundary condition)

  @rq-35af4d98
  Scenario: apply_influence produces V_hat[k] = G[k] * rho_hat[k] for k != 0
    Given a complex grid of known rho_hat values and a known influence_G
    When spme_recip_apply_influence runs
    Then for every k != 0, V_hat[k] equals influence_G[k] * rho_hat[k]
      to within f32 round-off

  @rq-d4d54057
  Scenario: apply_influence writes one virial partial per block
    Given a complex grid of M_complex cells and block size 256
    And num_blocks = ceil(M_complex / 256)
    When spme_recip_apply_influence runs
    Then exactly num_blocks entries of virial_partials are written
    And each entry equals the sum of virial_factor[k] * |rho_hat[k]|² over the cells assigned to its block,
      reduced via the deterministic shared-memory pairwise tree

  @rq-191d86df
  Scenario: apply_influence does not materialise a length-M_complex virial buffer
    Given the SpmeReciprocalState is constructed
    When the slot's owned device allocations are enumerated
    Then no allocation of length M_complex is reserved for per-cell virial staging
    And virial_partials has length ceil(M_complex / 256)

  @rq-70442b56
  Scenario: Two runs of apply_influence on identical inputs are byte-identical
    Given two device-resident copies of identical rho_hat, influence_G, virial_factor inputs
    When spme_recip_apply_influence runs on each
    Then dtoh(V_hat) is byte-identical between the two runs
    And dtoh(virial_partials) is byte-identical between the two runs

  # --- Force gather ---

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

  @rq-ede87154
  Scenario: spme_recip_reduce_partials writes (0.5 / N) · Σ virial_partials on device
    Given a virial_partials device buffer of length num_blocks with known per-block values
    When spme_recip_reduce_partials is launched with scale = 0.5 / N
    Then dtoh(w_per_particle_virial)[0] equals 0.5 / N · Σ_b virial_partials[b]
      to within f32 round-off
    And no host download of virial_partials occurs

  @rq-9d344eb9
  Scenario: Two runs of spme_recip_reduce_partials on identical inputs are byte-identical
    Given two device-resident copies of an identical virial_partials buffer
    When spme_recip_reduce_partials runs on each
    Then dtoh(w_per_particle_virial) produces byte-identical scalars between the two runs

  @rq-65ba517f
  Scenario: End-to-end recip-virial reduction equals (0.5 / N) · Σ_k virial_factor[k] · |rho_hat[k]|²
    Given a known rho_hat, influence_G, and virial_factor on device
    And reference total W = Σ_k virial_factor[k] · |rho_hat[k]|² computed on host in f64
    When spme_recip_apply_influence followed by spme_recip_reduce_partials runs
    Then dtoh(w_per_particle_virial)[0] equals (0.5 / N) · W to within f32 round-off

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

  # --- Cross-stream synchronization (observable behavior only) ---

  @rq-73efd4be
  Scenario: Two-stream pipeline preserves bit-exact reproducibility across runs
    Given two independent ForceField instances A and B, both with SPME enabled and identical inputs
    When each runs one full ForceField::step on the same GPU
    And each pipeline's ParticleBuffers.forces_x, forces_y, forces_z, potential_energies, virials are downloaded
    Then run A and run B agree byte-for-byte on every f32
```
