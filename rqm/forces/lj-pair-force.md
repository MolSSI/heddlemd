# Feature: Lennard-Jones Pair Force <!-- rq-13c02457 -->

Lennard-Jones is the non-bonded pairwise potential slot in the pluggable
potential framework (`framework.md`). The slot is present when the config
declares at least one `[[pair_interactions]]` entry. Its parameters come
from the per-pair-type table built from the full `[[pair_interactions]]`
array (see `io/config-schema.md`).

The slot contributes its per-pair functional form to the JIT-composed
pair-force pipeline as a `PairForceFragment` (see
`jit-composed-pair-force.md`). The composed kernel runs the
packed-neighbour data model and force-accumulation pattern specified
in `packed-neighbour-pair-force.md`: the inner loop evaluates only
real neighbours; per-particle contributions accumulate via
`atomicAdd` on the class's fixed-point force buffer. The slot does
not launch its own per-potential kernel.

This file specifies the Lennard-Jones functional form (with optional
inner-switching radius), the per-pair-type parameter tables the slot
exposes (`type_sigma`, `type_epsilon`, `type_cutoff`, `type_switch`,
`type_indices`), and the per-pair `evaluate(r², i, j) -> (factor,
energy, virial)` contract its fragment implements. The kernel
topology, launch configuration, neighbour-list contract, and
reproducibility mechanism are specified in
`packed-neighbour-pair-force.md` and not restated here.

Work is O(N · average neighbour count) when the shared list is in
cell-list mode and O(N²) when it is in trivial mode (every particle's
list contains every particle). The kernel is the same in both cases;
only the contents and size of the shared list differ.

The kernel reads the per-particle `type_indices` buffer (see
`particle-state.md`) and the per-pair-type parameter arrays, applies
the `ExclusionList` (see `topology.md`) per-pair scaling, and adds
into the `SlotOutputView` (a view onto the slot's class accumulator)
handed to it through `Potential::compute` — no per-pair intermediate
buffer is materialised on the device.

This file specifies `LennardJonesParameterTable` (the device-resident
per-pair-type parameter table), the two `lj_pair_force_*` CUDA kernels
in `kernels/pair_force.cu`, and the Rust launch helper that drives them.

Each pair contribution is multiplied by a CHARMM-style C¹ switching
function `S(r²)` over the interval `[r_switch, r_cut]` so that both
energy and force go smoothly to zero at the cutoff. Below `r_switch`
the switching factor is exactly `1` and the kernel evaluates the
unmodified Lennard-Jones force and energy.

## Algorithm <!-- rq-6d209943 -->

The kernel topology, sweep loop, warp-tree reduction, exclusion-scale
lookup, and per-particle output write all follow the common
warp-per-particle pattern specified in `pair-force-kernel.md`. The
LJ-specific contribution at each `(i, k)` pair is computed as follows.

For lane `lane` of the warp handling particle `i` at sweep step `s`,
when `k = s * 32 + lane` satisfies `k < neighbor_counts[i]` and
`j = neighbor_list[i * max_neighbors + k]` is not equal to `i`:

1. Resolve the per-pair-type parameter slot:

   ```
   ti = type_indices[i]
   tj = type_indices[j]
   p  = ti * n_types + tj
   sigma   = type_sigma[p]
   epsilon = type_epsilon[p]
   cutoff  = type_cutoff[p]
   switch  = type_switch[p]
   ```

2. Compute the displacement `(dx, dy, dz) = positions[i] - positions[j]`
   and apply the triclinic minimum-image convention using the six lattice
   parameters `(lx, ly, lz, xy, xz, yz)` and the tilt-subtraction
   algorithm defined in `simulation-box.md` (z-then-y-then-x wraps with
   tilt subtraction). For an orthorhombic box (all tilts zero) this
   reduces to three independent per-axis wraps.
3. Compute `r2 = dx*dx + dy*dy + dz*dz`. If `r2 > cutoff * cutoff`, the
   pair contributes nothing; the lane skips to its next assigned
   neighbour.
4. Compute the unswitched Lennard-Jones factor and energy in this order:

   ```
   inv_r2  = 1.0f / r2
   sigma2  = sigma * sigma
   sr2     = sigma2 * inv_r2
   sr6     = sr2 * sr2 * sr2
   sr12    = sr6 * sr6
   factor  = 24.0f * epsilon * inv_r2 * (2.0f * sr12 - sr6)
   energy  = 4.0f * epsilon * (sr12 - sr6)
   ```

   `factor` is the scalar such that the radial Lennard-Jones force
   vector is `factor * (dx, dy, dz)`; `energy` is the closed-form pair
   potential `U_lj(r)`.

5. Apply the CHARMM-style C¹ switching function `S(r²)` defined over
   `[r_switch, r_cut]`. Let `r_s2 = switch * switch` and
   `r_c2 = cutoff * cutoff`. The polynomial is evaluated in normalised
   form so the only place the cutoff-width factor appears is `1/delta`
   (not `1/delta³`), which keeps the arithmetic in range for f32 even
   at SI-scale lengths where `delta ≈ 10⁻¹⁹ m²` and `delta³` underflows
   the normal range. Two branches:

   - If `r2 <= r_s2`, the unmodified pair is fully inside the inner
     plateau:

     ```
     // S = 1, dS/d(r²) = 0; factor and energy are unchanged.
     ```

   - Otherwise (`r_s2 < r2`, with `r2 <= r_c2` guaranteed by step 3):

     ```
     delta        = r_c2 - r_s2
     inv_delta    = 1.0f / delta
     tau          = (r2 - r_s2) * inv_delta            // in [0, 1]
     one_minus    = 1.0f - tau
     S            = one_minus * one_minus * (1.0f + 2.0f * tau)
     chain_coeff  = 12.0f * tau * one_minus * inv_delta  // = -2 * dS/d(r²)
     factor       = S * factor + chain_coeff * energy
     energy       = S * energy
     ```

     With `tau = (r² − r_s2) / delta` the polynomial reduces to
     `S(tau) = (1 − tau)² (1 + 2 tau)` and
     `dS/d(r²) = −6 tau (1 − tau) / delta`. `S` satisfies
     `S(tau=0) = 1`, `S(tau=1) = 0`, and has zero derivative at both
     endpoints, making the resulting force C¹ continuous at `r_switch`
     and at `r_cut`. The `factor` update is the chain-rule consequence
     of multiplying the unswitched potential by `S(r²)`:
     `F_new = S · F_lj − (dS/dr) · U_lj · r̂`, which when expressed via
     `r²` collapses to `factor_new = S · factor + (−2 · dS/d(r²)) · U_lj`.

6. Compute the per-component contribution `(fx, fy, fz) = factor *
   (dx, dy, dz)` using the post-switching `factor`, and the scalar pair
   virial `w = fx * dx + fy * dy + fz * dz`. Apply the per-pair
   exclusion scale `scale = exclusion_scale(i, j, ...,
   atom_excl_lj_scales)` to `fx`, `fy`, `fz`, and to the post-switching
   `energy` and `w`. Switching is always applied before the exclusion
   scale.
7. Add `(fx, fy, fz)` to the lane's `(p_x, p_y, p_z)` register
   accumulators. The `_fev` variant additionally adds `energy * 0.5f` to
   `p_e` and `w * 0.5f` to `p_w`. The `0.5f` factor distributes each
   unordered pair's energy and virial across the two ordered
   contributions `(i, j)` and `(j, i)` so the framework's combiner
   counts each pair exactly once when summing over all particles.

After every lane has processed every assigned neighbour, the warp-tree
butterfly reduction collapses the 32 lane accumulators to lane 0, which
writes the particle's net force into `slot_force_x[i]`, `slot_force_y[i]`,
`slot_force_z[i]`; in the `_fev` variant lane 0 additionally writes
`slot_energy[i]` and `slot_virial[i]`. See `pair-force-kernel.md` for
the topology and reduction details.

### Switching-function degenerate case <!-- inline --> <!-- rq-78d2ad15 -->

When `switch == cutoff` the interval `[r_switch, r_cut]` is empty.
`r_s2 == r_c2` and the second branch of step 5 is unreachable: step 3
already gated out `r2 > r_c2`, and the remaining `r2 <= r_c2 == r_s2`
satisfies the first branch (`r2 <= r_s2`). The unmodified Lennard-Jones
expression is therefore used everywhere up to and including `r2 == r_c2`.
This degenerate case is the hard-cutoff behaviour: `S = 1` over
`[0, r_cut]` and step 3 produces the cliff at `r_cut`. No division by
zero ever occurs.

### JIT fragment behaviour <!-- rq-lj-jit-frag --> <!-- rq-9cd4085e -->

When `LennardJonesBuilder::pair_force_fragment(cx)` participates in
the JIT-composed pair-force kernel (see
`jit-composed-pair-force.md`), the fragment differs from the
standalone kernel above in three ways:

1. **Shared `(inv_r, r, qi, qj)` inputs.** The fragment's
   `evaluate` signature is
   `evaluate(Real r2, Real inv_r, Real r, Real qi, Real qj,
   unsigned int i, unsigned int j, Real &factor, Real &energy,
   Real &virial)`. `inv_r = rsqrtf(r²)`, `r = r² · inv_r`,
   `qi = posq[i].w`, and `qj = posq[j].w` are computed once per
   pair by the composer's outer loop and threaded into every
   active fragment. The LJ fragment does not compute `1.0 / r2`;
   it derives `inv_r2 = inv_r · inv_r` from `inv_r` directly. It
   ignores `qi` and `qj` (the Lennard-Jones functional form does
   not depend on charges; the parameters come from the per-pair-
   type `type_sigma` and `type_epsilon` arrays instead).

2. **Compile-time elision of the degenerate switch.** At fragment
   construction the builder inspects the
   `LennardJonesParameterTable`'s `switch` and `cutoff` arrays. If
   every `switch[p]` equals the corresponding `cutoff[p]` for
   every pair-type slot `p` (equivalently, every configured
   `[[pair_interactions]]` entry sets `r_switch == cutoff`,
   including the loader's `0.9 * cutoff` default when no entry
   sets it that way), the builder emits a *no-switch* fragment
   source whose `evaluate` body is exactly the unmodified
   Lennard-Jones recipe of step 4 in *Algorithm* (no `r_s2 =
   r_switch · r_switch`, no `r² > r_s²` test, no chain-rule
   correction). The switching code is not emitted into the
   generated PTX. If any `switch[p] != cutoff[p]`, the builder
   emits the full switch-aware fragment source covering the
   C¹ polynomial of step 5.

3. **CutoffHandling.** The fragment reports its cutoff structure
   to the composer based on the same parameter table:

   - When every `cutoff[p]` for `p ∈ [0, n_types²)` equals the
     same value `c`, the fragment reports
     `cutoff: CutoffHandling::Uniform(c)`. The composer omits the
     per-fragment `r² <= cutoff_squared(i, j)` guard when
     `c² == HEDDLE_JIT_MAX_CUTOFF_SQUARED`; otherwise it emits a
     single `if (r² <= c²)` guard with `c²` as a JIT-compile-time
     constant.
   - When at least two `cutoff[p]` entries differ, the fragment
     reports `cutoff: CutoffHandling::PerPair` and the composer
     emits the runtime
     `if (r² <= functor.cutoff_squared(i, j))` guard. The
     functor's `cutoff_squared(i, j)` indexes
     `type_cutoff[slot(i, j)]` and squares it.

   `evaluate` is invoked unconditionally for every pair the outer
   loop visits; the outer max-cutoff mask described in
   `jit-composed-pair-force.md` zeroes the contribution for
   `r² > HEDDLE_JIT_MAX_CUTOFF_SQUARED`. The fragment is safe to
   call at any positive `r²` because `inv_r` and `r` are
   well-defined for all positive `r²` and the LJ functional form
   `(σ·inv_r)¹² − (σ·inv_r)⁶` is finite for all positive `r²`.

The standalone `lj_pair_force_*` kernels documented below keep
the per-pair recipe in *Algorithm*; the JIT-fragment changes are
scoped to the JIT path.

### Parameter-table symmetry <!-- rq-7d92b551 -->

The kernel reads `type_sigma`, `type_epsilon`, `type_cutoff`, and
`type_switch` at index `ti * n_types + tj` without enforcing symmetry
between `(ti, tj)` and `(tj, ti)`. The expected use is for the host to
fill both `table[ti * n_types + tj]` and `table[tj * n_types + ti]`
from the same unordered `[[pair_interactions]]` entry, so the table is
symmetric by construction. Asymmetric tables yield asymmetric pair
forces and break Newton's third law.

### Reproducibility <!-- rq-a1abedca -->

The arithmetic is performed in the documented order, on identical inputs,
on every run. Each particle `i`'s warp accumulates its contributions in
a fixed lane-strided order (see `pair-force-kernel.md`); the warp-tree
butterfly is a fixed-shape reduction. Two runs with identical inputs
produce byte-identical per-particle outputs.

### Newton's third law <!-- rq-b7bbabd0 -->

Warps `W_i` and `W_j` independently compute the per-pair LJ contribution
to particles `i` and `j`. The displacements differ only in sign, the
wrap formula respects sign symmetry for displacements not equal to
exactly `±L/2`, and the LJ factor depends only on `r²` (which is
identical in both warps) and on the per-pair parameters at slot
`ti * n_types + tj` versus slot `tj * n_types + ti`. Provided the
parameter table is symmetric in `(ti, tj)` (the standard construction
from unordered `[[pair_interactions]]`), particle `i`'s net force from
the pair `{i, j}` and particle `j`'s net force from the same pair are
exact negatives bit-for-bit, except for the measure-zero exact-boundary
case `dx = ±L/2` (and similarly for `dy`, `dz`), where the asymmetric
wrap formula causes both warps to compute the same magnitude rather
than opposites.

## Feature API <!-- rq-61207d82 -->

### Types <!-- rq-20e97464 -->

- `LennardJonesParameterTable` — device-resident per-pair-type parameter <!-- rq-dafe0fcb -->
  table. Fields:
  - `n_types: u32` — number of distinct particle types referenced by the
    parameter table.
  - `sigma: CudaSlice<f32>` — length `n_types * n_types`. Entry at index
    `ti * n_types + tj` holds the σ value for the unordered pair
    `(ti, tj)`.
  - `epsilon: CudaSlice<f32>` — length `n_types * n_types`. Same indexing
    rule.
  - `cutoff: CudaSlice<f32>` — length `n_types * n_types`. Same indexing
    rule.
  - `switch: CudaSlice<f32>` — length `n_types * n_types`. Entry at
    index `ti * n_types + tj` holds the inner switching radius
    `r_switch` (metres) for the unordered pair `(ti, tj)`. Always
    satisfies `0 < r_switch <= cutoff` for that pair-type. Equality
    `r_switch == cutoff` selects the hard-cutoff degenerate case
    described in *Switching-function degenerate case*.

  Construction loads the four host-side `Vec<f32>` parameter tables onto
  the device with `htod_sync_copy`. The host-side construction is the
  responsibility of the caller; the standard production path is the
  associated function described below, which builds the table from a
  parsed `Config`.

  The table does not validate the σ/ε/cutoff/r_switch values themselves:
  non-finite, zero, or negative entries propagate to the kernel and
  yield non-finite or numerically meaningless forces.

- `LennardJonesParameterTable::from_config(device: &Arc<CudaDevice>, <!-- inline --> <!-- rq-1adf5954 -->
  particle_types: &[ParticleTypeConfig], pair_interactions:
  &[PairInteractionConfig]) -> Result<LennardJonesParameterTable, GpuError>`
  - `n_types = particle_types.len()`.
  - Allocates four host-side `Vec<f32>` of length `n_types * n_types`,
    initially zero. For each entry in `pair_interactions`, resolves the
    two type names to indices `ti`, `tj` via `particle_types` (matched
    by `name`), reads σ and ε from the entry's
    `PairPotentialParams::LennardJones` variant and `cutoff` /
    `r_switch` from the entry's common fields, and writes σ, ε,
    `cutoff`, and `r_switch` at both `[ti * n_types + tj]` and
    `[tj * n_types + ti]`. The caller is responsible for having
    validated that every unordered pair is covered and that
    `r_switch <= cutoff` for every entry (the config loader enforces
    both; see `io/config-schema.md`). `r_switch` is taken directly from
    `PairInteractionConfig::r_switch`, which the config loader
    populates with the user-supplied value when present and with
    `0.9 * cutoff` when omitted.
  - Uploads each host array to a fresh `CudaSlice<f32>` and returns the
    populated `LennardJonesParameterTable`.
  - When `n_types == 0` (no particle types declared), all four slices
    have length zero.

### CUDA Kernels <!-- rq-4ddab3c7 -->

`kernels/pair_force.cu` declares two `extern "C"` kernels (forces-only
and forces + energy + virial) that share the warp-per-particle pattern
documented in `pair-force-kernel.md`.

#### `lj_pair_force_f`, `lj_pair_force_fev` <!-- rq-7b13d9cf -->

```c
extern "C" __global__ void lj_pair_force_f(
    const float4 *posq,
    const unsigned int *type_indices,
    unsigned int max_neighbors,
    const float *lattice,           // length 6: [lx, ly, lz, xy, xz, yz]
    unsigned int n_types,
    const float *type_sigma,
    const float *type_epsilon,
    const float *type_cutoff,
    const float *type_switch,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const float *atom_excl_lj_scales,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    float *slot_force_x,
    float *slot_force_y,
    float *slot_force_z,
    unsigned int n);

extern "C" __global__ void lj_pair_force_fev(
    const float4 *posq,
    const unsigned int *type_indices,
    unsigned int max_neighbors,
    const float *lattice,           // length 6: [lx, ly, lz, xy, xz, yz]
    unsigned int n_types,
    const float *type_sigma,
    const float *type_epsilon,
    const float *type_cutoff,
    const float *type_switch,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const float *atom_excl_lj_scales,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    float *slot_force_x,
    float *slot_force_y,
    float *slot_force_z,
    float *slot_energy,
    float *slot_virial,
    unsigned int n);
```

The LJ kernels read positions from `posq[i].xyz` and ignore
`posq[i].w`. Charges enter the LJ functional form only via the
σ/ε parameter table; they are not consulted directly.

`lj_pair_force_f` writes only the three per-particle force-component
slot outputs; `lj_pair_force_fev` writes the three force components
plus the per-particle potential-energy share and scalar-virial share.
Both variants share the same warp-per-particle launch geometry,
per-pair-type parameter lookup, displacement and minimum-image, cutoff
test, CHARMM-style C¹ switching function over `[r_switch, r_cut]`,
exclusion-list scaling, and warp-tree reduction defined in
`pair-force-kernel.md` and the *Algorithm* section above.

`neighbor_list` and `neighbor_counts` are owned by the shared
`NeighborListState` on `ForceField` (see `neighbor-list.md` and
`framework.md`). The framework keeps the list current via its
`pre_step` invocation before any slot's `compute` runs.

### Exclusion scaling <!-- inline-edit --> <!-- rq-dddcbf07 -->

After computing the closed-form Lennard-Jones force `(fx, fy, fz)`,
energy, and virial for pair `(i, k)` and before adding them to the
lane's register accumulators, the kernel scales all five quantities by
the factor returned by the shared device helper `exclusion_scale`
declared in `kernels/exclusions.cuh` (see `topology.md` for the helper's
API and semantics):

```
float s = exclusion_scale(
    i, j, atom_excl_offsets, atom_excl_partners, atom_excl_lj_scales);
fx *= s; fy *= s; fz *= s;
energy *= s;
w *= s;
```

The helper returns the matching scale when `j` appears in atom `i`'s
exclusion-partner range and `1.0f` otherwise (including when the range
is empty), so the unscaled LJ contribution flows through unchanged for
pairs that are not on the exclusion list.

The kernel must be launched with an exclusion list shaped consistently
with the particle count: `atom_excl_offsets` has length `N + 1` (where
the final entry equals the total number of partner entries), and
`atom_excl_partners` and `atom_excl_lj_scales` have the same length as
each other. Empty lists are represented by `atom_excl_offsets` of
length `N + 1` filled with zeros and zero-length partner / scale
buffers; the kernel handles this case without a separate code path.

### PTX Module Loading <!-- rq-78d9fd1c -->

`init_device()` loads the compiled `kernels/pair_force.cu` PTX as module
`"pair_force"` and captures both the `lj_pair_force_f` and
`lj_pair_force_fev` functions into the `Kernels` handle (see
`build-pipeline.md`).

### Rust Launcher <!-- rq-d6beaed7 -->

A free function in `src/gpu/kernels.rs`, re-exported from `crate::gpu`:

- `lj_pair_force(particle_buffers: &ParticleBuffers, output: &mut SlotOutputView<'_>, sim_box: &SimulationBox, params: &LennardJonesParameterTable, exclusions: &DeviceExclusionList, neighbor_list: &CudaSlice<u32>, neighbor_counts: &CudaSlice<u32>, max_neighbors: u32, level: AggregateLevel) -> Result<(), GpuError>` <!-- rq-d3a14184 -->
  - Selects the kernel variant based on `level`:
    `AggregateLevel::ForcesOnly` dispatches to `lj_pair_force_f` and
    writes only the three force-component slot output rows;
    `AggregateLevel::ForcesAndScalars` dispatches to `lj_pair_force_fev`
    and additionally writes the energy and virial slot output rows.
  - Launches with the warp-per-particle geometry documented in
    `pair-force-kernel.md`: `block_dim = (256, 1, 1)`,
    `grid_dim = (ceil(n / 8), 1, 1)`.
  - When `particle_buffers.particle_count() == 0`, returns `Ok(())`
    without launching a kernel.
  - Returns the underlying `GpuError` if the kernel launch fails.
  - Invokes the kernel through the `lj_pair_force_f` or
    `lj_pair_force_fev` field of the `Kernels` handle reached from its
    arguments; it performs no string-keyed kernel lookup of its own (see
    `build-pipeline.md`).

  The `DeviceExclusionList` argument is a host-side handle holding the
  four device buffers `atom_excl_offsets`, `atom_excl_partners`,
  `atom_excl_lj_scales`, and `atom_excl_coul_scales`. It is constructed
  from the host-side `ExclusionList` (see `topology.md`) once at force-field
  construction time and shared between the LJ and Coulomb slots; each
  slot consumes the scale array appropriate to itself. An empty exclusion
  list is represented by a `DeviceExclusionList` whose offsets buffer has
  length `N + 1` filled with zeros and whose partner / scale buffers have
  length zero.

  The launcher trusts the caller for shape consistency: in debug builds
  it asserts `output.force_x.len() == particle_buffers.particle_count()`
  (and similarly for `force_y`, `force_z`, and — under
  `ForcesAndScalars` — `energy` and `virial`),
  `neighbor_list.len() == particle_buffers.particle_count() *
  max_neighbors as usize`,
  `neighbor_counts.len() == particle_buffers.particle_count()`,
  `exclusions.particle_count() == particle_buffers.particle_count()`,
  and `params.sigma.len() == params.epsilon.len() == params.cutoff.len()
  == params.switch.len() == params.n_types as usize *
  params.n_types as usize`. Release builds skip the asserts for parity
  with the other kernel launchers. The launcher does not validate the
  σ/ε/cutoff/r_switch entries themselves and does not check any cutoff
  against `sim_box.min_perpendicular_width() / 2` (that gating happens
  in the neighbor list; see `forces/neighbor-list.md`).

## Launch Configuration <!-- rq-4fd872f5 -->

- Block size: 256 threads (8 warps × 32 lanes) for both variants.
- Grid size: `ceil(n / 8)` blocks in the x dimension.
- Shared memory: zero bytes.
- Stream: the default stream carried by `ParticleBuffers.device`.

Both numbers match the common pair-force pattern (`pair-force-kernel.md`).

## Practical Bounds <!-- rq-4a902e65 -->

- `n` is `u32` on the device side. Particle counts up to `u32::MAX` are
  representable.
- When the shared `NeighborListState` is in `Trivial` mode, work is
  O(N²) and `max_neighbors == N`; intended for systems of at most a few
  thousand particles.
- When the shared list is in `CellList` mode, work is O(N · avg_neighbors).
  `max_neighbors` is a user-supplied bound (typically 64–256).

## Slot Integration <!-- rq-a5a919df -->

`LennardJonesState` implements the `Potential` trait declared in
`framework.md` with `label() == "lennard_jones"`. It is a single struct
carrying the `LennardJonesParameterTable` and the `DeviceExclusionList`.
It does not own a neighbor list: the framework's shared
`NeighborListState` is passed in through `ForceFieldContext` at each
`compute` call.

The slot's `Potential` methods:

- `max_cutoff` returns `Some(max_cutoff)` where `max_cutoff` is the
  largest cutoff across the slot's pair-interaction configuration,
  captured at construction time as a plain `f32` field. The trait call
  requires no device download.
- `compute(buffers, sim_box, output, cx, timings, level)` invokes the
  `lj_pair_force` launcher (defined above), forwarding `output`,
  `level`, the shared `NeighborListState`'s `neighbor_list` and
  `neighbor_counts`, and `max_neighbors`. The launcher selects the
  `_f` or `_fev` kernel variant based on `level` and writes directly
  into the `SlotOutputView` slices, bracketed by the `LjPairForce`
  `KernelStage` labels. The framework has already called `pre_step` on
  the shared neighbor list before this method runs.

## Out of Scope <!-- rq-9d7966f4 -->

- Other interaction potentials (Buckingham, Morse, Coulomb, bonded terms).
- Combining rules (Lorentz–Berthelot, geometric, …) inside the kernel;
  the config supplies per-pair `(σ, ε, cutoff)` directly and any combining
  rule is the user's responsibility at config-authoring time.
- Energy and virial tensor computation; this feature computes forces only.
- Long-range tail corrections.
- Truncated-and-shifted (energy-shift-only) potential variants. The
  switching function specified in the *Algorithm* section is the only
  in-scope smoothing scheme.
- Numerical validation of inputs (cutoff vs. box size, σ > 0, ε > 0).
- The `f64` precision feature flag.
- Multi-stream or multi-GPU launches.

---

## Gherkin Scenarios <!-- rq-3c98d7a9 -->

```gherkin
Feature: Lennard-Jones O(N²) pair force kernel

  Background:
    Given a SimulationBox constructed with lx=20.0, ly=20.0, lz=20.0
    And a LennardJonesParameterTable with n_types=1, sigma=[1.0],
      epsilon=[1.0], cutoff=[5.0], switch=[5.0]
    And every particle has type_indices[i] = 0
    Note: Background fixes switch == cutoff so that scenarios written
    against the unmodified Lennard-Jones expression remain valid.
    Scenarios that exercise the switching region override `switch`
    explicitly in their `Given` clauses.

  # --- Module loading ---

  @rq-06058b71
  Scenario: init_device exposes the LJ kernel on the Kernels handle
    Given a CUDA-capable GPU is available as device 0
    When init_device() is called
    Then the returned GpuContext's kernels handle exposes the lj_pair_force function

  # --- Two-particle correctness ---

  @rq-c538b29d
  Scenario: Two particles at a fixed separation produce the closed-form LJ force
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5,0,0)
    When the _f variant of lj_pair_force is called with sigma=1.0, epsilon=1.0, cutoff=5.0
    Then slot_force_x[0] equals the closed-form LJ force on particle 0 due to particle 1 at r=1.5
    And slot_force_y[0] equals 0
    And slot_force_z[0] equals 0

  @rq-975b5ae0
  Scenario: Newton's third law is bit-exact for non-boundary displacements
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.3, 0.4, -0.2)
    When the _f variant of lj_pair_force is called
    Then slot_force_x[0] equals -slot_force_x[1] bitwise
    And slot_force_y[0] equals -slot_force_y[1] bitwise
    And slot_force_z[0] equals -slot_force_z[1] bitwise

  # --- Self slot ---

  @rq-cc87744c
  Scenario: Self-interaction slots are zero
    Given a ParticleState of N=4 with arbitrary positions
    When the _f variant of lj_pair_force is called
    Then for every i in 0..4, slot_force_x[i], slot_force_y[i], slot_force_z[i] are all 0.0_f32

  # --- Cutoff handling ---

  @rq-96fadc6f
  Scenario: Slot for a pair beyond cutoff is zero
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(6.0, 0, 0)
    And cutoff=5.0
    When the _f variant of lj_pair_force is called
    Then slot_force_x[0], slot_force_y[0], slot_force_z[0] are all 0.0_f32
    And slot_force_x[1], slot_force_y[1], slot_force_z[1] are all 0.0_f32

  @rq-d6bd915a
  Scenario: Pair exactly at cutoff yields the hard-cutoff value when switch == cutoff
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(5.0, 0, 0)
    And cutoff=5.0 and switch=5.0
    When the _f variant of lj_pair_force is called
    Then slot_force_x[0] equals the closed-form LJ force at r=5.0

  # --- Switching function ---

  @rq-0c4f8da8
  Scenario: Pair inside r_switch sees the unmodified Lennard-Jones force
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5, 0, 0)
    And cutoff=5.0 and switch=4.0
    When the _fev variant of lj_pair_force is called
    Then slot_force_x[0] equals the closed-form LJ force at r=1.5
    And slot_energy[0] + slot_energy[1] equals 4.0 * ε * (sr12 - sr6) at r=1.5 within f32 round-off
      (half-sum convention: each particle's slot carries 0.5 · U_ij; the sum recovers U_ij)

  @rq-38441c15
  Scenario: Pair exactly at r_switch sees the unmodified Lennard-Jones force
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(4.0, 0, 0)
    And cutoff=5.0 and switch=4.0
    When the _fev variant of lj_pair_force is called
    Then slot_force_x[0] equals the closed-form LJ force at r=4.0
    And slot_energy[0] + slot_energy[1] equals 4.0 * ε * (sr12 - sr6) at r=4.0 within f32 round-off

  @rq-f93d278e
  Scenario: Pair exactly at r_cut yields zero force and zero energy when switch < cutoff
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(5.0, 0, 0)
    And cutoff=5.0 and switch=4.0
    When the _fev variant of lj_pair_force is called
    Then slot_force_x[0], slot_force_y[0], slot_force_z[0] are all 0.0_f32
    And slot_energy[0] and slot_virial[0] are both 0.0_f32
    And slot_energy[1] and slot_virial[1] are both 0.0_f32

  @rq-cb85cf61
  Scenario: Pair near r_cut inside the switching window has force smaller than the unmodified value
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(4.95, 0, 0)
    And cutoff=5.0 and switch=4.0
    When the _fev variant of lj_pair_force is called to obtain f_switched (slot_force, slot_energy)
    And lj_pair_force is called with switch=cutoff to obtain f_unmodified
    Then |f_switched.slot_force_x[0]| is strictly less than |f_unmodified.slot_force_x[0]|
      (the switching function drives the force toward zero at r = r_cut, dominating the
      chain-rule correction term once r is well inside the inner switching window)
    And f_switched.slot_force_x[0] has the same sign as f_unmodified.slot_force_x[0] (or both are 0)
    And f_switched.slot_energy[0] + f_switched.slot_energy[1]
      equals S(r²) * 4.0 * ε * (sr12 - sr6) at r=4.95 within f32 round-off,
      where S(r²) = (1−τ)²(1+2τ) with τ = (r²−r_s²)/(r_c²−r_s²)

  @rq-ae20ddac
  Scenario: Force is C¹ continuous at r_switch
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(r, 0, 0)
    And cutoff=5.0 and switch=4.0
    When the _f variant of lj_pair_force is called at r = 4.0 - 1e-3 to obtain f_below
    And lj_pair_force is called at r = 4.0 + 1e-3 to obtain f_above
    Then |f_below.x - f_above.x| is bounded by 1e-2 * |f_below.x|

  @rq-e5e3443f
  Scenario: Force is C¹ continuous at r_cut
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(r, 0, 0)
    And cutoff=5.0 and switch=4.0
    When the _f variant of lj_pair_force is called at r = 5.0 - 1e-3 to obtain f_inside
    Then |f_inside.x| is bounded by 1e-2 * |closed-form LJ force at r = 4.0|

  @rq-916f99f3
  Scenario: switch == cutoff reproduces the hard-cutoff behaviour everywhere inside the cutoff
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(r, 0, 0) for r in {1.5, 3.0, 4.5}
    And cutoff=5.0 and switch=5.0
    When the _fev variant of lj_pair_force is called for each r
    Then slot_force_x[0] equals the closed-form LJ force at that r
    And slot_energy[0] + slot_energy[1]
      equals 4.0 * ε * (sr12 - sr6) at that r within f32 round-off

  @rq-531afe39
  Scenario: Pair beyond r_cut yields zero independent of r_switch
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(6.0, 0, 0)
    And cutoff=5.0 and switch=4.0
    When the _fev variant of lj_pair_force is called
    Then slot_force_x[0], slot_force_y[0], slot_force_z[0] are all 0.0_f32
    And slot_energy[0] and slot_virial[0] are both 0.0_f32
    And slot_energy[1] and slot_virial[1] are both 0.0_f32

  @rq-d0f489d7
  Scenario: Pair virial inside the switching window equals factor_switched * r²
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(4.5, 0, 0)
    And cutoff=5.0 and switch=4.0
    When the _fev variant of lj_pair_force is called
    Then slot_virial[0] + slot_virial[1]
      equals factor_switched * r² at r=4.5 within f32 round-off,
      where factor_switched = S(r²) · factor_lj − 2 · dS/d(r²) · U_lj

  @rq-ef8013be
  Scenario: Newton's third law holds bitwise across the switching window
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(4.5, 0.4, -0.2)
    And cutoff=5.0 and switch=4.0
    When the _f variant of lj_pair_force is called
    Then slot_force_x[0] equals -slot_force_x[1] bitwise
    And slot_force_y[0] equals -slot_force_y[1] bitwise
    And slot_force_z[0] equals -slot_force_z[1] bitwise

  @rq-fb55af77
  Scenario: Exclusion scaling multiplies the switched force, energy, and virial
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(4.5, 0, 0)
    And cutoff=5.0 and switch=4.0
    And a DeviceExclusionList containing the entry (0, 1, 0.5)
    When the _f variant of lj_pair_force is called to obtain slot_force_scaled
    And lj_pair_force is called with an empty DeviceExclusionList
      to obtain slot_force_unscaled
    Then slot_force_scaled_x[0] equals 0.5 * slot_force_unscaled_x[0]
      within f32 round-off
    And the analogous relation holds for pair_energies and pair_virials

  @rq-37f8c017
  Scenario: Per-pair-type r_switch dispatches correctly
    Given a LennardJonesParameterTable with n_types=2,
      sigma=[1.0, 1.0, 1.0, 1.0], epsilon=[1.0, 1.0, 1.0, 1.0],
      cutoff=[5.0, 5.0, 5.0, 5.0], switch=[5.0, 4.0, 4.0, 5.0]
    And a ParticleState of N=2 with positions p0=(0,0,0) and p1=(4.5, 0, 0)
    When the _f variant of lj_pair_force is called with type_indices = [0, 1]
    Then slot_force_x[0] equals S(r²) * closed-form LJ force at r=4.5
      using the off-diagonal switch=4.0
    And lj_pair_force called with type_indices = [0, 0] under the same positions
      yields slot_force_x[0] equal to the unswitched closed-form LJ force at r=4.5
      using the diagonal switch=5.0

  @rq-dbd3c689
  Scenario: Bit-exact reproducibility across runs with switching active
    Given two ParticleBuffers built from byte-identical ParticleState inputs of N=64
    And the LennardJonesParameterTable for each run has switch < cutoff
    When lj_pair_force is launched on each with identical parameters
    Then run A's slot_force_x, slot_force_y, slot_force_z, slot_energy, and slot_virial agree byte-for-byte with run B's

  @rq-214639c9
  Scenario: from_config populates r_switch from the parsed PairInteractionConfig
    Given a Config with particle_types ["Ar"] and a single pair_interactions
      entry between=["Ar","Ar"] with cutoff=5.0 and r_switch=4.0
    When LennardJonesParameterTable::from_config(device, &config.particle_types,
      &config.pair_interactions) is called
    Then the returned table has switch downloaded to the host equal to [4.0]

  @rq-6a542a0a
  Scenario: from_config receives the default r_switch when the config omitted it
    Given a Config in which load_config has populated PairInteractionConfig.r_switch
      with 0.9 * cutoff because the [[pair_interactions]] entry omitted r_switch
    When LennardJonesParameterTable::from_config is called
    Then the returned table has switch downloaded to the host equal to
      0.9 * cutoff for every entry, with both diagonal and off-diagonal slots
      filled symmetrically

  # --- Force-zero point ---

  @rq-85192a05
  Scenario: At the LJ minimum r = sigma * 2^(1/6), the force is zero
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(2^(1/6), 0, 0)
    And sigma=1.0
    When the _f variant of lj_pair_force is called
    Then slot_force_x[0], slot_force_y[0], slot_force_z[0] are all 0.0_f32 to within f32 round-off

  # --- Parameter scaling ---

  @rq-26ffa053
  Scenario: Doubling epsilon doubles the force at the same separation
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5, 0, 0)
    When the _f variant of lj_pair_force is called with epsilon=1.0 to obtain f1
    And lj_pair_force is called with epsilon=2.0 to obtain f2
    Then f2 equals 2.0 * f1 within f32 round-off

  # --- PBC minimum-image ---

  @rq-8626ec3c
  Scenario: Two particles across the box boundary interact via the minimum image
    Given a SimulationBox with lx=10.0, ly=10.0, lz=10.0
    And a ParticleState of N=2 with positions p0=(-4.5, 0, 0) and p1=(4.5, 0, 0)
    And cutoff=2.0
    When the _f variant of lj_pair_force is called
    Then slot_force_x[0] equals the closed-form LJ force at r=1.0 (computed via minimum-image dx=-1.0)

  # --- N=1 and N=0 ---

  @rq-681afa90
  Scenario: Single-particle state produces only a zero self slot
    Given a ParticleState of N=1
    When the _f variant of lj_pair_force is called
    Then slot_force_x[0], slot_force_y[0], slot_force_z[0] are all 0.0_f32

  @rq-fc220d87
  Scenario: Empty state is a no-op
    Given a ParticleState of N=0
    When the _f variant of lj_pair_force is called
    Then it returns Ok(())

  # --- Block-non-aligned ---

  @rq-d1e7cb57
  Scenario: Block-non-aligned particle count is handled by the bounds check
    Given a ParticleState of N=17 with positions distributed within the box
    When the _f variant of lj_pair_force is called
    Then for every i in 0..17, slot_force_x[i] equals 0
    And for every i in 0..17, k in 0..17, k != i, the slot equals the closed-form LJ force on i due to k

  # --- Reproducibility ---

  @rq-dfca62d2
  Scenario: Two independent runs produce byte-identical pair-force buffers
    Given two ParticleBuffers built from identical ParticleState inputs of N=64
    When lj_pair_force is launched on each with identical parameters
    Then run A's slot_force_x, slot_force_y, slot_force_z agree byte-for-byte with run B's

  # --- Slots beyond N are untouched ---

  @rq-e564f8e2
  Scenario: Kernel does not write slots with k >= n
    Given a ParticleState of N=4
    And every slot_force_* entry pre-loaded with the sentinel value 13.5_f32
    When the _f variant of lj_pair_force is called
    Then for every i in 0..4 and k in 4..8, slot_force_x[i] for every i in 0..4 equals only the in-cutoff contributions and the sentinel-loaded scratch is left untouched

  # --- Side effects ---

  @rq-14d7a940
  Scenario: Kernel does not modify positions, velocities, masses, or net forces
    Given a ParticleBuffers built from a ParticleState with N=4 known nonzero values
    And a snapshot of positions_*, velocities_*, masses, forces_*, particle_ids before launch
    When the _f variant of lj_pair_force is called
    And particle_buffers is downloaded to a host ParticleState
    Then every snapshot field is byte-identical to the corresponding downloaded field

  # --- End-to-end through the framework ---

  @rq-ec53799e
  Scenario: lj_pair_force writes the correct per-particle net force directly into the slot output
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5, 0, 0)
    And neighbor_counts on the device equal to [2, 2]
    When the _f variant of lj_pair_force is called against a SlotOutputView
    And the SlotOutputView is downloaded to the host
    Then slot_force_x[0] equals the closed-form LJ force on particle 0 due to particle 1 at r=1.5
    And slot_force_x[1] equals -slot_force_x[0] bitwise

  # --- Exclusion list ---

  @rq-e80653f1
  Scenario: Empty exclusion list leaves all pair forces unchanged
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5, 0, 0)
    And an empty DeviceExclusionList
    When the _f variant of lj_pair_force is called
    Then slot_force_x[0] equals the closed-form LJ force at r=1.5

  @rq-80dcfa97
  Scenario: Full exclusion (scale=0) zeros the LJ contribution for the excluded pair
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5, 0, 0)
    And a DeviceExclusionList containing the entry (0, 1, 0.0)
    When the _f variant of lj_pair_force is called
    Then slot_force_x[0], slot_force_y[0], slot_force_z[0] are all 0.0_f32
    And slot_force_x[1], slot_force_y[1], slot_force_z[1] are all 0.0_f32

  @rq-31430003
  Scenario: Half-strength exclusion (scale=0.5) halves the LJ contribution
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5, 0, 0)
    And a DeviceExclusionList containing the entry (0, 1, 0.5)
    When the _f variant of lj_pair_force is called
    Then slot_force_x[0] equals 0.5 * closed-form LJ force at r=1.5 within f32 round-off
    And slot_force_x[1] equals -slot_force_x[0]

  @rq-8c786f79
  Scenario: Exclusion only applies to the listed pair
    Given a ParticleState of N=3 with positions p0=(0,0,0), p1=(1.5,0,0), p2=(3.0,0,0)
    And a DeviceExclusionList containing only the entry (0, 1, 0.0)
    When the _f variant of lj_pair_force is called
    Then slot_force_x[0] equals the contribution of every in-cutoff neighbour other than particle 1 (the (0,1) pair is scaled by 0.0 and contributes nothing)
    And particle 0's slot_force_x reflects the unscaled (0,2) pair contribution
    And particle 1's slot_force_x reflects the unscaled (1,2) pair contribution

  @rq-3a1eea58
  Scenario: Scale = 1.0 is equivalent to no exclusion
    Given a ParticleState of N=2 and an exclusion (0, 1, 1.0)
    When the _f variant of lj_pair_force is called
    Then slot_force_x[0] equals the closed-form LJ force at the pair distance

  # --- NaN propagation ---

  @rq-daf7550b
  Scenario: NaN positions propagate to NaN pair forces
    Given a ParticleState of N=2 with positions_x[0] = f32::NAN and otherwise valid finite values
    When the _f variant of lj_pair_force is called
    Then slot_force_x[0] is NaN
    And slot_force_x[1] is NaN

  # --- Multi-type parameter dispatch ---

  @rq-4a14aec3
  Scenario: Same-type pair uses the diagonal parameter slot
    Given a LennardJonesParameterTable with n_types=2,
      sigma=[1.0, 2.0, 2.0, 3.0], epsilon=[1.0, 0.5, 0.5, 2.0], cutoff=[5.0, 5.0, 5.0, 5.0]
    And a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5,0,0)
    And type_indices = [0, 0]
    When the _f variant of lj_pair_force is called
    Then slot_force_x[0] equals the closed-form LJ force at r=1.5
      using sigma=1.0 and epsilon=1.0

  @rq-23fc870b
  Scenario: Mixed-type pair uses the off-diagonal parameter slot
    Given a LennardJonesParameterTable with n_types=2,
      sigma=[1.0, 2.0, 2.0, 3.0], epsilon=[1.0, 0.5, 0.5, 2.0], cutoff=[5.0, 5.0, 5.0, 5.0]
    And a ParticleState of N=2 with positions p0=(0,0,0) and p1=(2.5,0,0)
    And type_indices = [0, 1]
    When the _f variant of lj_pair_force is called
    Then slot_force_x[0] equals the closed-form LJ force at r=2.5
      using sigma=2.0 and epsilon=0.5

  @rq-55640f03
  Scenario: Different-type same-pair Newton's third law holds for symmetric tables
    Given a LennardJonesParameterTable with n_types=2 filled symmetrically
      from one [[pair_interactions]] entry per unordered pair
    And a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.3, 0.4, -0.2)
    And type_indices = [0, 1]
    When the _f variant of lj_pair_force is called
    Then slot_force_x[0] equals -slot_force_x[1] bitwise
    And slot_force_y[0] equals -slot_force_y[1] bitwise
    And slot_force_z[0] equals -slot_force_z[1] bitwise

  @rq-244fe033
  Scenario: Per-pair-type cutoff zeroes only the pair whose cutoff it exceeds
    Given a LennardJonesParameterTable with n_types=2 where
      cutoff[(0,0)] = 5.0 and cutoff[(0,1)] = cutoff[(1,0)] = 1.0 and cutoff[(1,1)] = 5.0
    And a ParticleState of N=3 with positions p0=(0,0,0), p1=(1.5,0,0), p2=(2.0,0,0)
    And type_indices = [0, 0, 1]
    When the _f variant of lj_pair_force is called
    Then particle 0's slot_force_x reflects the (0,0)-type pair at r=1.5 (inside cutoff 5.0)
    And particle 0's slot_force_x carries no contribution from the (0,1)-type pair at r=2.0 (exceeds cutoff 1.0)
    And particle 1's slot_force_x reflects the (0,1)-type pair at r=0.5 (inside cutoff 1.0)

  @rq-1e7e6aa4
  Scenario: Three-type table dispatches correctly per pair
    Given a LennardJonesParameterTable with n_types=3 whose σ entries differ for every (ti,tj)
    And a ParticleState of N=3 with one atom of each type and fixed positions
    When the _f variant of lj_pair_force is called
    Then for every particle i, slot_force_x[i] matches the sum over k of the closed-form LJ contributions from each in-cutoff partner k
      force computed using sigma[type_indices[i] * 3 + type_indices[k]] and the
      corresponding epsilon and cutoff entries

  @rq-75446ddd
  Scenario: from_config builds a symmetric parameter table from an unordered pair_interactions
    Given a Config with particle_types ["Ar", "Kr"] and pair_interactions
      ("Ar","Ar"){σ=1, ε=1, cut=5}, ("Ar","Kr"){σ=2, ε=0.5, cut=5}, ("Kr","Kr"){σ=3, ε=2, cut=5}
    When LennardJonesParameterTable::from_config(device, &config.particle_types, &config.pair_interactions) is called
    Then the returned table has n_types=2
    And sigma downloaded to the host equals [1.0, 2.0, 2.0, 3.0]
    And epsilon downloaded to the host equals [1.0, 0.5, 0.5, 2.0]
    And cutoff downloaded to the host equals [5.0, 5.0, 5.0, 5.0]

  # --- Shared neighbor list ---

  @rq-9004fd7a
  Scenario: LennardJonesState reports its max cutoff to the framework
    Given a LennardJonesState constructed from pair_interactions with cutoffs [5.0, 3.0, 4.0]
    When max_cutoff() is queried
    Then it returns Some(5.0)

  @rq-535c2b1e
  Scenario: LJ kernel reads the shared neighbor list from ForceFieldContext
    Given a ForceField with one LennardJones slot in CellList mode
    And a particle configuration that places two atoms within the LJ cutoff
    When ForceField::step is called
    Then ForceField::neighbor_list has been pre_step'd once before contribute runs
    And the LJ kernel reads neighbor_list / neighbor_counts from the shared state

  @rq-e90c6feb
  Scenario: Trivial-mode and cell-list-mode forces agree within tolerance
    Given two ForceField instances built from byte-identical particle states
      one with NeighborListConfig::AllPairs (Trivial shared list)
      and the other with NeighborListConfig::CellList { max_neighbors, r_skin }
    When ForceField::step is called on each
    Then forces_* agree componentwise within 1e-4 relative error

  # --- Energy and virial outputs ---

  @rq-b68b3445
  Scenario: Two-particle pair energy matches the closed-form Lennard-Jones expression
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5,0,0)
    And a LennardJonesParameterTable with σ=1.0, ε=1.0, cutoff=5.0
    When the _fev variant of lj_pair_force is called
    Then slot_energy[0] + slot_energy[1]
      equals 4.0 * ε * (sr12 - sr6) within f32 round-off
      where sr2 = (σ/r)^2, sr6 = sr2^3, sr12 = sr6^2, r = 1.5

  @rq-0b71c50a
  Scenario: Two-particle pair virial matches r_ij · F_ij
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5,0,0)
    And a LennardJonesParameterTable with σ=1.0, ε=1.0, cutoff=5.0
    When the _fev variant of lj_pair_force is called
    Then slot_virial[0] + slot_virial[1]
      equals r_ij · F_ij within f32 round-off
      where F_ij is the force on particle 0 due to particle 1

  @rq-a50cb6a1
  Scenario: Pair beyond cutoff yields zero per-particle energy and virial
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(6.0,0,0)
    And cutoff=5.0
    When the _fev variant of lj_pair_force is called
    Then slot_energy[0] and slot_virial[0] are both 0.0_f32
    And slot_energy[1] and slot_virial[1] are both 0.0_f32

  @rq-82f8d168
  Scenario: A particle whose only listed neighbour is itself carries zero energy and virial
    Given a ParticleState of N=4 with arbitrary positions
    And the neighbour list for each particle i lists only itself (count=1, neighbor_list[i*4 + 0] = i)
    When the _fev variant of lj_pair_force is called
    Then for every i in 0..4, slot_energy[i] and slot_virial[i] are both 0.0_f32

  @rq-95c2f543
  Scenario: Exclusion scaling applies uniformly to force, energy, and virial
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5,0,0)
    And a DeviceExclusionList containing the entry (0, 1, 0.5)
    When the _fev variant of lj_pair_force is called
    Then slot_energy[0] equals 0.5 * (un-excluded LJ energy / 2) within f32 round-off
      (the 0.5 is the exclusion scale; the 1/2 is the half-sum convention; the slot accumulates one pair contribution)
    And slot_virial[0] equals 0.5 * (un-excluded virial / 2) within f32 round-off

  # --- JIT fragment behaviour ---

  @rq-7e0df64f
  Scenario: LJ JIT fragment uses the composer-supplied inv_r and r
    Given a ForceField with at least one [[pair_interactions]] entry and the JIT-composed kernel active
    And the composed kernel source captured for inspection
    Then the LJ fragment's evaluate signature is `evaluate(Real r2, Real inv_r, Real r, unsigned int i, unsigned int j, Real &factor, Real &energy, Real &virial)`
    And the LJ fragment body does not contain `1.0 / r2`, `1.0f / r2`, `R(1.0) / r2`, or any other expression that computes inv_r2 from r2

  @rq-6060a744
  Scenario: LJ JIT fragment with uniform cutoff reports CutoffHandling::Uniform
    Given a config with three [[pair_interactions]] entries all setting cutoff = 1.0e-9
    When LennardJonesBuilder::pair_force_fragment(cx) is called
    Then it returns Ok(Some(fragment)) with `fragment.cutoff == CutoffHandling::Uniform(1.0e-9)`

  @rq-6f6fe328
  Scenario: LJ JIT fragment with mixed cutoffs reports CutoffHandling::PerPair
    Given a config with two [[pair_interactions]] entries setting cutoff = 1.0e-9 and one setting cutoff = 8.0e-10
    When LennardJonesBuilder::pair_force_fragment(cx) is called
    Then it returns Ok(Some(fragment)) with `fragment.cutoff == CutoffHandling::PerPair`

  @rq-5c26c1ff
  Scenario: LJ JIT fragment elides the switching code when every pair-type sets r_switch = cutoff
    Given a config whose every [[pair_interactions]] entry sets r_switch = cutoff (the loader's input or the explicit user value)
    When LennardJonesBuilder::pair_force_fragment(cx) is called
    And the composed kernel source is captured for inspection
    Then the LJ fragment body does not contain the literal substring `r_switch`
    And the LJ fragment body does not contain the literal substring `if (r2 > r_s2)`
    And the LJ fragment body does not contain the chain-rule coefficient `12.0`

  @rq-e4628324
  Scenario: LJ JIT fragment emits the switching code when at least one pair-type has r_switch < cutoff
    Given a config with two [[pair_interactions]] entries where one has r_switch = 0.9 * cutoff and the other has r_switch = cutoff
    When LennardJonesBuilder::pair_force_fragment(cx) is called
    And the composed kernel source is captured for inspection
    Then the LJ fragment body contains the C¹ switching polynomial of step 5 (the `r_s2`, `tau`, `S = one_minus * one_minus * (1.0 + 2.0 * tau)`, and chain-rule terms)

  @rq-828e1ea2
  Scenario: LJ JIT fragment with uniform cutoff produces the closed-form LJ force within f32 round-off
    Given a config with one [[pair_interactions]] entry between Ar-Ar with sigma=1.0, epsilon=1.0, cutoff=5.0, r_switch=5.0
    And a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5,0,0)
    And the JIT-composed kernel active
    When ForceField::step(...) is called
    Then the per-particle force on particle 0 equals the closed-form LJ force at r=1.5 within 1e-5 relative tolerance
    And two runs of the same configuration produce byte-identical per-particle forces
```
