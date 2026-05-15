# Feature: Lennard-Jones Pair Force Kernel <!-- rq-13c02457 -->

Lennard-Jones is the non-bonded pairwise potential slot in the pluggable
potential framework (`framework.md`). The slot is present when the config
declares at least one `[[pair_interactions]]` entry. Its parameters come
from the per-pair-type table built from the full
`[[pair_interactions]]` array (see `io/config-schema.md`).

The slot evaluates pair forces with a single kernel that reads the
shared `NeighborListState` owned by `ForceField` (see
`neighbor-list.md`). Each `(i, k)` thread reads
`neighbor_list[i * max_neighbors + k]` to find its partner `j`, looks
up the per-pair-type parameters at `type_indices[i] * n_types +
type_indices[j]`, applies the per-pair cutoff, and writes the force
into the `PairBuffer` at slot `i * max_neighbors + k`. Work is O(N ·
average neighbour count) when the shared list is in cell-list mode and
O(N²) when it is in trivial mode (every particle's list contains every
particle). The kernel is the same in both cases; only the contents and
size of the shared list differ.

The kernel reads the per-particle `type_indices` buffer (see
`particle-state.md`) and the per-pair-type parameter arrays, applies
the `ExclusionList` (see `bonds.md`) per-pair scaling, and writes into
the `PairBuffer`. Reduction uses `reduce_pair_forces` (see
`pair-reduction.md`).

This file specifies `LennardJonesParameterTable` (the device-resident
per-pair-type parameter table), the `lj_pair_force` CUDA kernel in
`kernels/pair_force.cu`, and the Rust launch helper that drives it.

Each pair contribution is multiplied by a CHARMM-style C¹ switching
function `S(r²)` over the interval `[r_switch, r_cut]` so that both
energy and force go smoothly to zero at the cutoff. Below `r_switch`
the switching factor is exactly `1` and the kernel evaluates the
unmodified Lennard-Jones force and energy.

## Algorithm <!-- rq-6d209943 -->

For each ordered `(i, k)` with `0 <= i < N` and `0 <= k <
neighbor_counts[i]`:

1. The pair-buffer slot is `slot = i * max_neighbors + k`.
2. Read `j = neighbor_list[slot]`. If `i == j`, write `0.0_f32` to
   `pair_forces_x[slot]`, `pair_forces_y[slot]`, `pair_forces_z[slot]`,
   `pair_energies[slot]`, and `pair_virials[slot]` and stop. (The kernel
   encounters `i == j` only when the shared neighbor list is in trivial
   mode, which lists every particle including self.)
3. Resolve the per-pair-type parameter slot:

   ```
   ti = type_indices[i]
   tj = type_indices[j]
   p  = ti * n_types + tj
   sigma   = type_sigma[p]
   epsilon = type_epsilon[p]
   cutoff  = type_cutoff[p]
   switch  = type_switch[p]
   ```

4. Compute the displacement `dx = positions_x[i] - positions_x[j]` (and
   similarly `dy`, `dz`).
5. Apply the triclinic minimum-image convention to `(dx, dy, dz)` using
   the six lattice parameters `(lx, ly, lz, xy, xz, yz)` and the tilt-
   subtraction algorithm defined in `simulation-box.md` (z-then-y-then-x
   wraps with tilt subtraction). For an orthorhombic box (all tilts
   zero) this reduces to three independent per-axis wraps.
6. Compute `r2 = dx*dx + dy*dy + dz*dz`.
7. If `r2 > cutoff * cutoff`, write `0.0_f32` to all five slots
   (force x/y/z, energy, virial) and stop.
8. Compute the unswitched Lennard-Jones factor and energy in this order:

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

9. Apply the CHARMM-style C¹ switching function `S(r²)` defined over
   `[r_switch, r_cut]`. Let `r_s2 = switch * switch` and
   `r_c2 = cutoff * cutoff`. The polynomial is evaluated in
   normalised form so the only place the cutoff-width factor appears
   is `1/delta` (not `1/delta³`), which keeps the arithmetic in range
   for f32 even at SI-scale lengths where `delta ≈ 10⁻¹⁹ m²` and
   `delta³` underflows the normal range. Two branches:

   - If `r2 <= r_s2`, the unmodified pair is fully inside the inner
     plateau:

     ```
     // S = 1, dS/d(r²) = 0; factor and energy are unchanged.
     ```

   - Otherwise (`r_s2 < r2`, with `r2 <= r_c2` guaranteed by step 7):

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
     endpoints, making the resulting force C¹ continuous at
     `r_switch` and at `r_cut`. The `factor` update is the chain-rule
     consequence of multiplying the unswitched potential by `S(r²)`:
     `F_new = S · F_lj − (dS/dr) · U_lj · r̂`, which when expressed via
     `r²` collapses to
     `factor_new = S · factor + (−2 · dS/d(r²)) · U_lj`.

10. Compute the per-component force `(fx, fy, fz) = factor * (dx, dy,
    dz)` using the post-switching `factor`, and the scalar pair virial
    `w = fx * dx + fy * dy + fz * dz`. Apply the exclusion scaling (see
    *Exclusion scaling*) to `fx`, `fy`, `fz`, and to the post-switching
    `energy` and `w`. Switching is always applied before the exclusion
    scale.

11. Write the slot's share of each per-pair quantity:

    ```
    pair_forces_x[slot] = fx
    pair_forces_y[slot] = fy
    pair_forces_z[slot] = fz
    pair_energies[slot] = energy * 0.5f
    pair_virials[slot]  = w * 0.5f
    ```

    The `0.5f` factor distributes the pair's energy and virial across its
    two slots, slot `(i, j)` and slot `(j, i)`, so the segmented reduction
    counts each pair exactly once when summed over all particles.

The `(i, k)` slot holds the force on particle `i` due to partner
`j = neighbor_list[i * max_neighbors + k]`, plus particle `i`'s shares
of the pair potential energy and scalar virial. The segmented reduction
kernel sums `neighbor_counts[i]` slots per particle, including the self
slot (which contributes zero) when present in trivial mode, and produces
the correct per-particle net force, potential-energy share, and
virial share.

### Switching-function degenerate case <!-- inline --> <!-- rq-78d2ad15 -->

When `switch == cutoff` the interval `[r_switch, r_cut]` is empty.
`r_s2 == r_c2` and the second branch of step 9 is unreachable: step 7
already gated out `r2 > r_c2`, and the remaining `r2 <= r_c2 == r_s2`
satisfies the first branch (`r2 <= r_s2`). The unmodified
Lennard-Jones expression is therefore used everywhere up to and
including `r2 == r_c2`. This degenerate case is the hard-cutoff
behaviour: `S = 1` over `[0, r_cut]` and step 7 produces the cliff at
`r_cut`. No division by zero ever occurs.

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
on every run. Each `(i, k)` slot is written by exactly one thread; there
are no atomics and no race conditions. Two runs with identical inputs
produce byte-identical outputs.

### Newton's third law <!-- rq-b7bbabd0 -->

Threads `(i, j)` and `(j, i)` independently compute `F_ij` and `F_ji`. The
displacements differ only in sign, the wrap formula respects sign symmetry
for displacements not equal to exactly `±L/2`, and the LJ factor depends
only on `r²` (which is identical in both threads) and on the per-pair
parameters at slot `ti * n_types + tj` versus slot `tj * n_types + ti`.
Provided the parameter table is symmetric in `(ti, tj)` (the standard
construction from unordered `[[pair_interactions]]`),
`pair_forces_*[i*max_neighbors + j] == -pair_forces_*[j*max_neighbors + i]`
bit-exactly for all displacements except the measure-zero exact-boundary
case `dx = ±L/2` (and similarly for `dy`, `dz`), where the asymmetric wrap
formula causes both threads to compute the same value rather than
opposites.

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

`kernels/pair_force.cu` declares one `extern "C"` kernel.

#### `lj_pair_force` <!-- rq-7b13d9cf -->

```c
extern "C" __global__ void lj_pair_force(
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    const unsigned int *type_indices,
    float *pair_forces_x,
    float *pair_forces_y,
    float *pair_forces_z,
    float *pair_energies,
    float *pair_virials,
    unsigned int max_neighbors,
    float lx, float ly, float lz, float xy, float xz, float yz,
    unsigned int n_types,
    const float *type_sigma,
    const float *type_epsilon,
    const float *type_cutoff,
    const float *type_switch,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const float *atom_excl_scales,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    unsigned int n);
```

Each thread maps to one `(i, k)` pair where `i` is an atom and `k` is
its k-th neighbour:

```
i = blockIdx.y * blockDim.y + threadIdx.y;
k = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= n || k >= neighbor_counts[i]) {
    // Slots beyond the actual neighbour count are zeroed so the
    // segmented reduction sees a clean sum even when max_neighbors
    // exceeds the realised count.
    pair_forces_x[i * max_neighbors + k] = 0.0f;  // (when i < n)
    pair_forces_y[i * max_neighbors + k] = 0.0f;
    pair_forces_z[i * max_neighbors + k] = 0.0f;
    pair_energies[i * max_neighbors + k] = 0.0f;
    pair_virials[i * max_neighbors + k]  = 0.0f;
    return;
}
unsigned int j = neighbor_list[i * max_neighbors + k];
```

After resolving `j`, the per-pair-type parameter lookup at
`type_indices[i] * n_types + type_indices[j]`, the per-pair force,
energy, and virial calculation, minimum-image, cutoff check, the
CHARMM-style C¹ switching function over `[r_switch, r_cut]`, and
exclusion-list scaling proceed as in the *Algorithm* section above.
The five quantities (force x/y/z, half-energy, half-virial) are written
to slot `i * max_neighbors + k`.

`neighbor_list` and `neighbor_counts` are owned by the shared
`NeighborListState` on `ForceField` (see `neighbor-list.md` and
`framework.md`). The framework keeps the list current via its
`pre_step` invocation before any slot's `contribute` runs.

### Exclusion scaling <!-- inline-edit --> <!-- rq-dddcbf07 -->

After computing the closed-form Lennard-Jones force `(fx, fy, fz)`,
energy, and virial for pair `(i, k)` and before writing the results to
the pair-buffer slots, the kernel scales all five quantities by the
factor returned by the shared device helper `exclusion_scale` declared
in `kernels/exclusions.cuh` (see `bonds.md` for the helper's API and
semantics):

```
float s = exclusion_scale(
    i, j, atom_excl_offsets, atom_excl_partners, atom_excl_scales);
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
`atom_excl_partners` and `atom_excl_scales` have the same length as
each other. Empty lists are represented by `atom_excl_offsets` of
length `N + 1` filled with zeros and zero-length partner / scale
buffers; the kernel handles this case without a separate code path.

### PTX Module Loading <!-- rq-78d9fd1c -->

`init_device()` loads the compiled `kernels/pair_force.cu` PTX as module
`"pair_force"` and captures its `lj_pair_force` function into the
`Kernels` handle (see `build-pipeline.md`).

### Rust Launcher <!-- rq-d6beaed7 -->

A free function in `src/gpu/kernels.rs`, re-exported from `crate::gpu`:

- `lj_pair_force(particle_buffers: &ParticleBuffers, pair_buffer: &mut PairBuffer, sim_box: &SimulationBox, params: &LennardJonesParameterTable, exclusions: &DeviceExclusionList, neighbor_list: &CudaSlice<u32>, neighbor_counts: &CudaSlice<u32>) -> Result<(), GpuError>` <!-- rq-d3a14184 -->
  - Launches the `lj_pair_force` kernel with the per-pair force, energy,
    virial, simulation box, parameter-table, type-index, exclusion-list,
    and shared neighbor-list arguments described above. The kernel writes
    the five per-pair quantities into the corresponding fields of
    `pair_buffer`.
  - 2D launch: `block_dim = (16, 16, 1)`, `grid_dim = (ceil(max_neighbors
    / 16), ceil(n / 16), 1)` — the y dimension covers atoms, the x
    dimension covers per-atom neighbour slots.
  - When `particle_buffers.particle_count() == 0`, returns `Ok(())`
    without launching a kernel.
  - Returns the underlying `GpuError` if the kernel launch fails.
  - Invokes the kernel through the `lj_pair_force` field of the
    `Kernels` handle reached from its arguments; it performs no
    string-keyed kernel lookup of its own (see `build-pipeline.md`).

  The `DeviceExclusionList` argument is a host-side handle holding the
  three device buffers `atom_excl_offsets`, `atom_excl_partners`, and
  `atom_excl_scales`. It is constructed from the host-side
  `ExclusionList` (see `bonds.md`) when the `MorseBonded` slot (or the
  `LennardJones` slot on its own) is built. An empty exclusion list is
  represented by a `DeviceExclusionList` whose offsets buffer has length
  `N + 1` filled with zeros and whose partner / scale buffers have
  length zero.

  The launcher trusts the caller for shape consistency: in debug builds
  it asserts `pair_buffer.particle_count() ==
  particle_buffers.particle_count()`,
  `neighbor_list.len() == particle_buffers.particle_count() *
  pair_buffer.max_neighbors() as usize`,
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

- Block size: 16 × 16 × 1 = 256 threads per block.
- Grid size: `(ceil(max_neighbors / 16), ceil(n / 16), 1)` blocks.
- Shared memory: zero bytes.
- Stream: the default stream carried by `pair_buffer.device`.

## Practical Bounds <!-- rq-4a902e65 -->

- `n` is `u32` on the device side. Particle counts up to `u32::MAX` are
  representable.
- When the shared `NeighborListState` is in `Trivial` mode, work is
  O(N²) and `max_neighbors == N`; intended for systems of at most a few
  thousand particles.
- When the shared list is in `CellList` mode, work is O(N · avg_neighbors).
  `max_neighbors` is a user-supplied bound (typically 64–256); the
  launcher's debug assert ensures the pair buffer is large enough.

## Slot Integration <!-- rq-a5a919df -->

`LennardJonesState` implements the `Potential` trait declared in
`framework.md` with `label() == "lennard_jones"`. It is a single struct
carrying the `PairBuffer` (sized to `max_neighbors` set by the shared
`NeighborListState`), the `LennardJonesParameterTable`, and the
`DeviceExclusionList`. It does not own a neighbor list: the framework's
shared `NeighborListState` is passed in through `ForceFieldContext` at
each `contribute` call.

The slot's `Potential` methods:

- `max_cutoff` returns `Some(max_cutoff)` where `max_cutoff` is the
  largest cutoff across the slot's pair-interaction configuration,
  captured at construction time as a plain `f32` field. The trait call
  requires no device download.
- `contribute(buffers, sim_box, cx, timings)` launches
  `lj_pair_force(particle_buffers, &mut pair_buffer, sim_box,
  &params, &exclusions, &cx.neighbor_list.expect("LJ requires shared
  neighbor list").neighbor_list, &cx.neighbor_list.unwrap().neighbor_counts)`,
  bracketed by the `LjPairForce` `KernelStage` labels. The framework
  has already called `pre_step` on the shared neighbor list before
  this method runs.
- `reduce` launches `reduce_pair_forces(&pair_buffer, &neighbor_counts,
  &mut output.x, &mut output.y, &mut output.z, N)` (see
  `pair-reduction.md`), where `neighbor_counts` comes from the shared
  `NeighborListState`. Writes into the `SlotForceView` supplied by the
  framework, bracketed by the `ReducePairForces` stage labels.

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
    And a PairBuffer with particle_count=2 and max_neighbors=2
    When lj_pair_force is called with sigma=1.0, epsilon=1.0, cutoff=5.0
    Then pair_forces_x[0*2 + 1] equals the closed-form LJ force on particle 0 due to particle 1 at r=1.5
    And pair_forces_y[0*2 + 1] equals 0
    And pair_forces_z[0*2 + 1] equals 0

  @rq-975b5ae0
  Scenario: Newton's third law is bit-exact for non-boundary displacements
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.3, 0.4, -0.2)
    And a PairBuffer with particle_count=2 and max_neighbors=2
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1] equals -pair_forces_x[1*2 + 0] bitwise
    And pair_forces_y[0*2 + 1] equals -pair_forces_y[1*2 + 0] bitwise
    And pair_forces_z[0*2 + 1] equals -pair_forces_z[1*2 + 0] bitwise

  # --- Self slot ---

  @rq-cc87744c
  Scenario: Self-interaction slots are zero
    Given a ParticleState of N=4 with arbitrary positions
    And a PairBuffer with particle_count=4 and max_neighbors=4
    When lj_pair_force is called
    Then for every i in 0..4, pair_forces_x[i*4 + i], pair_forces_y[i*4 + i], pair_forces_z[i*4 + i] are all 0.0_f32

  # --- Cutoff handling ---

  @rq-96fadc6f
  Scenario: Slot for a pair beyond cutoff is zero
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(6.0, 0, 0)
    And cutoff=5.0
    And a PairBuffer with particle_count=2 and max_neighbors=2
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1], pair_forces_y[0*2 + 1], pair_forces_z[0*2 + 1] are all 0.0_f32
    And pair_forces_x[1*2 + 0], pair_forces_y[1*2 + 0], pair_forces_z[1*2 + 0] are all 0.0_f32

  @rq-d6bd915a
  Scenario: Pair exactly at cutoff yields the hard-cutoff value when switch == cutoff
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(5.0, 0, 0)
    And cutoff=5.0 and switch=5.0
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1] equals the closed-form LJ force at r=5.0

  # --- Switching function ---

  @rq-0c4f8da8
  Scenario: Pair inside r_switch sees the unmodified Lennard-Jones force
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5, 0, 0)
    And cutoff=5.0 and switch=4.0
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1] equals the closed-form LJ force at r=1.5
    And pair_energies[0*2 + 1] + pair_energies[1*2 + 0]
      equals 4.0 * ε * (sr12 - sr6) at r=1.5 within f32 round-off

  @rq-38441c15
  Scenario: Pair exactly at r_switch sees the unmodified Lennard-Jones force
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(4.0, 0, 0)
    And cutoff=5.0 and switch=4.0
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1] equals the closed-form LJ force at r=4.0
    And pair_energies[0*2 + 1] + pair_energies[1*2 + 0]
      equals 4.0 * ε * (sr12 - sr6) at r=4.0 within f32 round-off

  @rq-f93d278e
  Scenario: Pair exactly at r_cut yields zero force and zero energy when switch < cutoff
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(5.0, 0, 0)
    And cutoff=5.0 and switch=4.0
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1], pair_forces_y[0*2 + 1], pair_forces_z[0*2 + 1] are all 0.0_f32
    And pair_energies[0*2 + 1] and pair_virials[0*2 + 1] are both 0.0_f32

  @rq-cb85cf61
  Scenario: Pair inside the switching window has force smaller than the unmodified value
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(4.5, 0, 0)
    And cutoff=5.0 and switch=4.0
    When lj_pair_force is called to obtain f_switched
    And lj_pair_force is called with switch=cutoff to obtain f_unmodified
    Then |f_switched.x| is strictly less than |f_unmodified.x|
    And f_switched.x has the same sign as f_unmodified.x (or both are 0)
    And the switched pair_energies[0*2 + 1] + pair_energies[1*2 + 0]
      equals S(r²) * 4.0 * ε * (sr12 - sr6) at r=4.5 within f32 round-off,
      where S(r²) = (r_c²-r²)²(r_c²+2r²-3r_s²)/(r_c²-r_s²)³

  @rq-ae20ddac
  Scenario: Force is C¹ continuous at r_switch
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(r, 0, 0)
    And cutoff=5.0 and switch=4.0
    When lj_pair_force is called at r = 4.0 - 1e-3 to obtain f_below
    And lj_pair_force is called at r = 4.0 + 1e-3 to obtain f_above
    Then |f_below.x - f_above.x| is bounded by 1e-2 * |f_below.x|

  @rq-e5e3443f
  Scenario: Force is C¹ continuous at r_cut
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(r, 0, 0)
    And cutoff=5.0 and switch=4.0
    When lj_pair_force is called at r = 5.0 - 1e-3 to obtain f_inside
    Then |f_inside.x| is bounded by 1e-2 * |closed-form LJ force at r = 4.0|

  @rq-916f99f3
  Scenario: switch == cutoff reproduces the hard-cutoff behaviour everywhere inside the cutoff
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(r, 0, 0) for r in {1.5, 3.0, 4.5}
    And cutoff=5.0 and switch=5.0
    When lj_pair_force is called for each r
    Then pair_forces_x[0*2 + 1] equals the closed-form LJ force at that r
    And pair_energies[0*2 + 1] + pair_energies[1*2 + 0]
      equals 4.0 * ε * (sr12 - sr6) at that r within f32 round-off

  @rq-531afe39
  Scenario: Pair beyond r_cut yields zero independent of r_switch
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(6.0, 0, 0)
    And cutoff=5.0 and switch=4.0
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1], pair_forces_y[0*2 + 1], pair_forces_z[0*2 + 1] are all 0.0_f32
    And pair_energies[0*2 + 1] and pair_virials[0*2 + 1] are both 0.0_f32

  @rq-d0f489d7
  Scenario: Pair virial inside the switching window equals factor_switched * r²
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(4.5, 0, 0)
    And cutoff=5.0 and switch=4.0
    When lj_pair_force is called
    Then pair_virials[0*2 + 1] + pair_virials[1*2 + 0]
      equals factor_switched * r² at r=4.5 within f32 round-off,
      where factor_switched = S(r²) · factor_lj − 2 · dS/d(r²) · U_lj

  @rq-ef8013be
  Scenario: Newton's third law holds bitwise across the switching window
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(4.5, 0.4, -0.2)
    And cutoff=5.0 and switch=4.0
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1] equals -pair_forces_x[1*2 + 0] bitwise
    And pair_forces_y[0*2 + 1] equals -pair_forces_y[1*2 + 0] bitwise
    And pair_forces_z[0*2 + 1] equals -pair_forces_z[1*2 + 0] bitwise

  @rq-fb55af77
  Scenario: Exclusion scaling multiplies the switched force, energy, and virial
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(4.5, 0, 0)
    And cutoff=5.0 and switch=4.0
    And a DeviceExclusionList containing the entry (0, 1, 0.5)
    When lj_pair_force is called to obtain pair_forces_scaled
    And lj_pair_force is called with an empty DeviceExclusionList
      to obtain pair_forces_unscaled
    Then pair_forces_scaled.x[0*2 + 1] equals 0.5 * pair_forces_unscaled.x[0*2 + 1]
      within f32 round-off
    And the analogous relation holds for pair_energies and pair_virials

  @rq-37f8c017
  Scenario: Per-pair-type r_switch dispatches correctly
    Given a LennardJonesParameterTable with n_types=2,
      sigma=[1.0, 1.0, 1.0, 1.0], epsilon=[1.0, 1.0, 1.0, 1.0],
      cutoff=[5.0, 5.0, 5.0, 5.0], switch=[5.0, 4.0, 4.0, 5.0]
    And a ParticleState of N=2 with positions p0=(0,0,0) and p1=(4.5, 0, 0)
    When lj_pair_force is called with type_indices = [0, 1]
    Then pair_forces_x[0*2 + 1] equals S(r²) * closed-form LJ force at r=4.5
      using the off-diagonal switch=4.0
    And lj_pair_force called with type_indices = [0, 0] under the same positions
      yields pair_forces_x[0*2 + 1] equal to the unswitched closed-form LJ force at r=4.5
      using the diagonal switch=5.0

  @rq-dbd3c689
  Scenario: Bit-exact reproducibility across runs with switching active
    Given two ParticleBuffers built from byte-identical ParticleState inputs of N=64
    And the LennardJonesParameterTable for each run has switch < cutoff
    When lj_pair_force is launched on each with identical parameters
    Then run A's pair_forces_x, pair_forces_y, pair_forces_z, pair_energies,
      and pair_virials agree byte-for-byte with run B's

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
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1], pair_forces_y[0*2 + 1], pair_forces_z[0*2 + 1] are all 0.0_f32 to within f32 round-off

  # --- Parameter scaling ---

  @rq-26ffa053
  Scenario: Doubling epsilon doubles the force at the same separation
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5, 0, 0)
    When lj_pair_force is called with epsilon=1.0 to obtain f1
    And lj_pair_force is called with epsilon=2.0 to obtain f2
    Then f2 equals 2.0 * f1 within f32 round-off

  # --- PBC minimum-image ---

  @rq-8626ec3c
  Scenario: Two particles across the box boundary interact via the minimum image
    Given a SimulationBox with lx=10.0, ly=10.0, lz=10.0
    And a ParticleState of N=2 with positions p0=(-4.5, 0, 0) and p1=(4.5, 0, 0)
    And cutoff=2.0
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1] equals the closed-form LJ force at r=1.0 (computed via minimum-image dx=-1.0)

  # --- N=1 and N=0 ---

  @rq-681afa90
  Scenario: Single-particle state produces only a zero self slot
    Given a ParticleState of N=1
    And a PairBuffer with particle_count=1 and max_neighbors=1
    When lj_pair_force is called
    Then pair_forces_x[0], pair_forces_y[0], pair_forces_z[0] are all 0.0_f32

  @rq-fc220d87
  Scenario: Empty state is a no-op
    Given a ParticleState of N=0
    And a PairBuffer with particle_count=0 and max_neighbors=0
    When lj_pair_force is called
    Then it returns Ok(())

  # --- Block-non-aligned ---

  @rq-d1e7cb57
  Scenario: Block-non-aligned particle count is handled by the bounds check
    Given a ParticleState of N=17 with positions distributed within the box
    And a PairBuffer with particle_count=17 and max_neighbors=17
    When lj_pair_force is called
    Then for every i in 0..17, pair_forces_x[i*17 + i] equals 0
    And for every i in 0..17, k in 0..17, k != i, the slot equals the closed-form LJ force on i due to k

  # --- Reproducibility ---

  @rq-dfca62d2
  Scenario: Two independent runs produce byte-identical pair-force buffers
    Given two PairBuffers and two ParticleBuffers built from identical ParticleState inputs of N=64
    When lj_pair_force is launched on each with identical parameters
    Then run A's pair_forces_x, pair_forces_y, pair_forces_z agree byte-for-byte with run B's

  # --- Slots beyond N are untouched ---

  @rq-e564f8e2
  Scenario: Kernel does not write slots with k >= n
    Given a ParticleState of N=4
    And a PairBuffer with particle_count=4 and max_neighbors=8
    And every pair_forces_* slot pre-loaded with the sentinel value 13.5_f32
    When lj_pair_force is called
    Then for every i in 0..4 and k in 4..8, pair_forces_x[i*8 + k], pair_forces_y[i*8 + k], pair_forces_z[i*8 + k] still equal 13.5_f32

  # --- Side effects ---

  @rq-14d7a940
  Scenario: Kernel does not modify positions, velocities, masses, or net forces
    Given a ParticleBuffers built from a ParticleState with N=4 known nonzero values
    And a snapshot of positions_*, velocities_*, masses, forces_*, particle_ids before launch
    When lj_pair_force is called
    And particle_buffers is downloaded to a host ParticleState
    Then every snapshot field is byte-identical to the corresponding downloaded field

  # --- End-to-end with reduction ---

  @rq-ec53799e
  Scenario: lj_pair_force followed by reduce_pair_forces produces the correct net force
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5, 0, 0)
    And a PairBuffer with particle_count=2 and max_neighbors=2
    And neighbor_counts on the device equal to [2, 2]
    When lj_pair_force is called
    And reduce_pair_forces is called
    And particle_buffers is downloaded to a host ParticleState
    Then forces_x[0] equals the closed-form LJ force on particle 0 due to particle 1 at r=1.5
    And forces_x[1] equals -forces_x[0] bitwise

  # --- Exclusion list ---

  @rq-e80653f1
  Scenario: Empty exclusion list leaves all pair forces unchanged
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5, 0, 0)
    And an empty DeviceExclusionList
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1] equals the closed-form LJ force at r=1.5

  @rq-80dcfa97
  Scenario: Full exclusion (scale=0) zeros the LJ contribution for the excluded pair
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5, 0, 0)
    And a DeviceExclusionList containing the entry (0, 1, 0.0)
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1], pair_forces_y[0*2 + 1], pair_forces_z[0*2 + 1] are all 0.0_f32
    And pair_forces_x[1*2 + 0], pair_forces_y[1*2 + 0], pair_forces_z[1*2 + 0] are all 0.0_f32

  @rq-31430003
  Scenario: Half-strength exclusion (scale=0.5) halves the LJ contribution
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5, 0, 0)
    And a DeviceExclusionList containing the entry (0, 1, 0.5)
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1] equals 0.5 * closed-form LJ force at r=1.5 within f32 round-off
    And pair_forces_x[1*2 + 0] equals -pair_forces_x[0*2 + 1]

  @rq-8c786f79
  Scenario: Exclusion only applies to the listed pair
    Given a ParticleState of N=3 with positions p0=(0,0,0), p1=(1.5,0,0), p2=(3.0,0,0)
    And a DeviceExclusionList containing only the entry (0, 1, 0.0)
    When lj_pair_force is called
    Then pair_forces_x[0*3 + 1] is 0 (the (0,1) pair is scaled by 0.0)
    And pair_forces_x[0*3 + 2] is non-zero (the (0,2) pair is unscaled)
    And pair_forces_x[1*3 + 2] is non-zero (the (1,2) pair is unscaled)

  @rq-3a1eea58
  Scenario: Scale = 1.0 is equivalent to no exclusion
    Given a ParticleState of N=2 and an exclusion (0, 1, 1.0)
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1] equals the closed-form LJ force at the pair distance

  # --- NaN propagation ---

  @rq-daf7550b
  Scenario: NaN positions propagate to NaN pair forces
    Given a ParticleState of N=2 with positions_x[0] = f32::NAN and otherwise valid finite values
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1] is NaN
    And pair_forces_x[1*2 + 0] is NaN

  # --- Multi-type parameter dispatch ---

  @rq-4a14aec3
  Scenario: Same-type pair uses the diagonal parameter slot
    Given a LennardJonesParameterTable with n_types=2,
      sigma=[1.0, 2.0, 2.0, 3.0], epsilon=[1.0, 0.5, 0.5, 2.0], cutoff=[5.0, 5.0, 5.0, 5.0]
    And a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5,0,0)
    And type_indices = [0, 0]
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1] equals the closed-form LJ force at r=1.5
      using sigma=1.0 and epsilon=1.0

  @rq-23fc870b
  Scenario: Mixed-type pair uses the off-diagonal parameter slot
    Given a LennardJonesParameterTable with n_types=2,
      sigma=[1.0, 2.0, 2.0, 3.0], epsilon=[1.0, 0.5, 0.5, 2.0], cutoff=[5.0, 5.0, 5.0, 5.0]
    And a ParticleState of N=2 with positions p0=(0,0,0) and p1=(2.5,0,0)
    And type_indices = [0, 1]
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1] equals the closed-form LJ force at r=2.5
      using sigma=2.0 and epsilon=0.5

  @rq-55640f03
  Scenario: Different-type same-pair Newton's third law holds for symmetric tables
    Given a LennardJonesParameterTable with n_types=2 filled symmetrically
      from one [[pair_interactions]] entry per unordered pair
    And a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.3, 0.4, -0.2)
    And type_indices = [0, 1]
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1] equals -pair_forces_x[1*2 + 0] bitwise
    And pair_forces_y[0*2 + 1] equals -pair_forces_y[1*2 + 0] bitwise
    And pair_forces_z[0*2 + 1] equals -pair_forces_z[1*2 + 0] bitwise

  @rq-244fe033
  Scenario: Per-pair-type cutoff zeroes only the pair whose cutoff it exceeds
    Given a LennardJonesParameterTable with n_types=2 where
      cutoff[(0,0)] = 5.0 and cutoff[(0,1)] = cutoff[(1,0)] = 1.0 and cutoff[(1,1)] = 5.0
    And a ParticleState of N=3 with positions p0=(0,0,0), p1=(1.5,0,0), p2=(2.0,0,0)
    And type_indices = [0, 0, 1]
    When lj_pair_force is called
    Then pair_forces_x[0*3 + 1] is non-zero (the (0,0)-type pair at r=1.5 is inside cutoff 5.0)
    And pair_forces_x[0*3 + 2] is 0 (the (0,1)-type pair at r=2.0 exceeds cutoff 1.0)
    And pair_forces_x[1*3 + 2] is 0 (the (0,1)-type pair at r=0.5 is inside cutoff 1.0 — non-zero)

  @rq-1e7e6aa4
  Scenario: Three-type table dispatches correctly per pair
    Given a LennardJonesParameterTable with n_types=3 whose σ entries differ for every (ti,tj)
    And a ParticleState of N=3 with one atom of each type and fixed positions
    When lj_pair_force is called
    Then for every (i, k) with i != k, pair_forces_x[i*3 + k] matches the closed-form LJ
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
    When lj_pair_force is called
    Then pair_energies[0*2 + 1] + pair_energies[1*2 + 0]
      equals 4.0 * ε * (sr12 - sr6) within f32 round-off
      where sr2 = (σ/r)^2, sr6 = sr2^3, sr12 = sr6^2, r = 1.5

  @rq-0b71c50a
  Scenario: Two-particle pair virial matches r_ij · F_ij
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5,0,0)
    And a LennardJonesParameterTable with σ=1.0, ε=1.0, cutoff=5.0
    When lj_pair_force is called
    Then pair_virials[0*2 + 1] + pair_virials[1*2 + 0]
      equals r_ij · F_ij within f32 round-off
      where F_ij is the force on particle 0 due to particle 1

  @rq-a50cb6a1
  Scenario: Pair beyond cutoff yields zero energy and virial slots
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(6.0,0,0)
    And cutoff=5.0
    When lj_pair_force is called
    Then pair_energies[0*2 + 1] and pair_virials[0*2 + 1] are both 0.0_f32
    And pair_energies[1*2 + 0] and pair_virials[1*2 + 0] are both 0.0_f32

  @rq-82f8d168
  Scenario: Self slots carry zero energy and virial
    Given a ParticleState of N=4 with arbitrary positions
    When lj_pair_force is called with a trivial neighbor list
    Then for every i in 0..4, pair_energies[i*4 + i] and pair_virials[i*4 + i] are both 0.0_f32

  @rq-95c2f543
  Scenario: Exclusion scaling applies uniformly to force, energy, and virial
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(1.5,0,0)
    And a DeviceExclusionList containing the entry (0, 1, 0.5)
    When lj_pair_force is called
    Then pair_energies[0*2 + 1] equals 0.5 * (un-excluded LJ energy / 2) within f32 round-off
    And pair_virials[0*2 + 1] equals 0.5 * (un-excluded virial / 2) within f32 round-off
```
