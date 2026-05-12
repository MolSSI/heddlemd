# Feature: Lennard-Jones O(N²) Pair Force Kernel <!-- rq-13c02457 -->

Lennard-Jones is the non-bonded pairwise potential slot in the pluggable
potential framework (`framework.md`). The slot is always present; its
parameters come from the config's `[[pair_interactions]]` array. The
contribution kernel pairs every `(i, k)` thread against every other particle
directly (O(N²)), reads any matching scaling factor from the
`ExclusionList` (see `bonds.md`), and writes per-pair force contributions
into a `PairBuffer` at deterministic offsets. The slot's reduction kernel
(`reduce_pair_forces`, see `pair-reduction.md`) sums those contributions
into the slot's private per-atom accumulator.

This file specifies `LennardJonesParameters` (the host-side parameter
struct), `kernels/pair_force.cu` (the CUDA kernel including its
exclusion-list query), and the Rust launch helper that drives it.

## Algorithm <!-- rq-6d209943 -->

For each ordered pair `(i, k)` with `0 <= i < N` and `0 <= k < N`:

1. The pair-buffer slot is `slot = i * max_neighbors + k`.
2. If `i == k`, write `0.0_f32` to `pair_forces_x[slot]`,
   `pair_forces_y[slot]`, and `pair_forces_z[slot]` and stop.
3. Compute the displacement `dx = positions_x[i] - positions_x[k]` (and
   similarly `dy`, `dz`).
4. Apply the minimum-image convention along each Cartesian axis using the
   simulation box edge lengths `(lx, ly, lz)`:
   `dx <- dx - lx * floor((dx + lx * 0.5f) / lx)` (and similarly for `dy`,
   `dz`).
5. Compute `r2 = dx*dx + dy*dy + dz*dz`.
6. If `r2 > cutoff * cutoff`, write `0.0_f32` to the three slots and stop.
7. Otherwise, compute the LJ factor in this order:

   ```
   inv_r2  = 1.0f / r2
   sigma2  = sigma * sigma
   sr2     = sigma2 * inv_r2
   sr6     = sr2 * sr2 * sr2
   sr12    = sr6 * sr6
   factor  = 24.0f * epsilon * inv_r2 * (2.0f * sr12 - sr6)
   ```

8. Write the per-component force to the slot:

   ```
   pair_forces_x[slot] = factor * dx
   pair_forces_y[slot] = factor * dy
   pair_forces_z[slot] = factor * dz
   ```

The `(i, k)` slot holds the force on particle `i` due to particle `j = k`.
With `neighbor_counts[i] = N` for every `i`, the segmented reduction kernel
sums all `N` slots per particle, including the self slot which contributes
zero, and produces the correct net force.

### Reproducibility <!-- rq-a1abedca -->

The arithmetic is performed in the documented order, on identical inputs,
on every run. Each `(i, k)` slot is written by exactly one thread; there
are no atomics and no race conditions. Two runs with identical inputs
produce byte-identical outputs.

### Newton's third law <!-- rq-b7bbabd0 -->

Threads `(i, j)` and `(j, i)` independently compute `F_ij` and `F_ji`. The
displacements differ only in sign, the wrap formula respects sign symmetry
for displacements not equal to exactly `±L/2`, and the LJ factor depends
only on `r²` (which is identical in both threads). Therefore
`pair_forces_*[i*max_neighbors + j] == -pair_forces_*[j*max_neighbors + i]`
bit-exactly for all displacements except the measure-zero exact-boundary
case `dx = ±L/2` (and similarly for `dy`, `dz`), where the asymmetric wrap
formula causes both threads to compute the same value rather than
opposites.

## Feature API <!-- rq-61207d82 -->

### Types <!-- rq-20e97464 -->

- `LennardJonesParameters` — host-side `Copy` struct carrying the three <!-- rq-dafe0fcb -->
  scalar parameters of the LJ pair interaction:
  - `sigma: f32` — distance at which the pair potential is zero.
  - `epsilon: f32` — depth of the potential well.
  - `cutoff: f32` — pair distance above which the force is treated as
    exactly zero.

  Construction is direct field access: `LennardJonesParameters { sigma,
  epsilon, cutoff }`. There is no validating constructor; non-finite, zero,
  or negative parameters propagate to the kernel and yield non-finite or
  numerically meaningless forces.

### CUDA Kernel <!-- rq-4ddab3c7 -->

`kernels/pair_force.cu` declares one `extern "C"` kernel:

```c
extern "C" __global__ void lj_pair_force(
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    float *pair_forces_x,
    float *pair_forces_y,
    float *pair_forces_z,
    unsigned int max_neighbors,
    float lx, float ly, float lz,
    float sigma,
    float epsilon,
    float cutoff,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const float *atom_excl_scales,
    unsigned int n);
```

Each thread maps to one ordered `(i, k)` pair via:

```
i = blockIdx.y * blockDim.y + threadIdx.y;
k = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= n || k >= n) return;
```

The thread executes the algorithm above for its `(i, k)` pair. The kernel
reads `positions_*` and writes only `pair_forces_*` at the indices
`i * max_neighbors + k` for `0 <= i < n` and `0 <= k < n`. Slots with
`k >= n` are not written.

### Exclusion scaling <!-- inline-edit --> <!-- rq-dddcbf07 -->

After computing the closed-form Lennard-Jones force `(fx, fy, fz)` for pair
`(i, k)` and before writing the result to `pair_forces_*[slot]`, the kernel
queries the exclusion list to scale the force:

```
start = atom_excl_offsets[i]
end   = atom_excl_offsets[i + 1]
scale = 1.0f
for m in start .. end:
    if atom_excl_partners[m] == k:
        scale = atom_excl_scales[m]
        break
fx *= scale; fy *= scale; fz *= scale
```

The lookup is a linear scan over atom `i`'s exclusion partners. The
partner list is short for typical bonded systems (≤ 12 entries per
atom). When the exclusion list is empty, every atom's offset range is
`[k, k]` for some `k`, the loop runs zero iterations, and `scale`
remains `1.0`, leaving the unscaled LJ force intact.

The kernel must be launched with an exclusion list shaped consistently
with the particle count: `atom_excl_offsets` has length `N + 1` (where
the final entry equals the total number of partner entries), and
`atom_excl_partners` and `atom_excl_scales` have the same length as
each other. Empty lists are represented by `atom_excl_offsets` of
length `N + 1` filled with zeros and zero-length partner / scale
buffers; the kernel handles this case without a separate code path.

### PTX Module Loading <!-- rq-78d9fd1c -->

`init_device()` loads the compiled `kernels/pair_force.cu` PTX with module
name `"pair_force"` and registers function name `"lj_pair_force"`,
alongside the existing `fill`, `integrate`, and `reduce` modules.

### Rust Launcher <!-- rq-d6beaed7 -->

A free function in `src/gpu/kernels.rs`, re-exported from `crate::gpu`:

- `lj_pair_force(particle_buffers: &ParticleBuffers, pair_buffer: &mut PairBuffer, sim_box: &SimulationBox, params: &LennardJonesParameters, exclusions: &DeviceExclusionList) -> Result<(), GpuError>` <!-- rq-d3a14184 -->
  - Launches the `lj_pair_force` kernel with the per-pair force, simulation
    box, parameter, and exclusion-list arguments described above.
  - 2D launch: `block_dim = (16, 16, 1)`, `grid_dim = (ceil(n / 16), ceil(n / 16), 1)`.
  - When `particle_buffers.particle_count() == 0`, returns `Ok(())` without
    launching a kernel.
  - Returns the underlying `GpuError` if the kernel launch fails.
  - Panics if the `"pair_force"` module is not loaded on the device, since
    this indicates a programming error in `init_device()`.

  The `DeviceExclusionList` argument is a host-side handle holding the three
  device buffers `atom_excl_offsets`, `atom_excl_partners`, and
  `atom_excl_scales`. It is constructed from the host-side `ExclusionList`
  (see `bonds.md`) when the `MorseBonded` slot (or the `LennardJones` slot
  on its own) is built. An empty exclusion list is represented by a
  `DeviceExclusionList` whose offsets buffer has length `N + 1` filled with
  zeros and whose partner / scale buffers have length zero.

  The launcher trusts the caller for shape consistency: in debug builds it
  asserts `pair_buffer.particle_count() == particle_buffers.particle_count()`,
  `pair_buffer.max_neighbors() as usize >= particle_buffers.particle_count()`,
  and `exclusions.particle_count() == particle_buffers.particle_count()`.
  Release builds skip the asserts for parity with the other kernel
  launchers. The launcher does not validate `params.sigma`, `params.epsilon`,
  or `params.cutoff` and does not check `params.cutoff <= min(lx, ly, lz) / 2`.

## Launch Configuration <!-- rq-4fd872f5 -->

- Block size: 16 × 16 × 1 = 256 threads per block.
- Grid size: `(ceil(n / 16), ceil(n / 16), 1)` blocks.
- Shared memory: zero bytes.
- Stream: the default stream carried by `pair_buffer.device`.

## Practical Bounds <!-- rq-4a902e65 -->

- `n` is `u32` on the device side. Particle counts up to `u32::MAX` are
  representable but the per-step work is O(N²); this kernel is intended
  for systems of at most a few thousand particles.
- `max_neighbors` must be at least `n`; otherwise the kernel writes outside
  the buffer for `k >= max_neighbors`. The launcher's debug assert catches
  this in development.

## Slot Integration <!-- rq-a5a919df -->

`LennardJonesState` is one variant of the `PotentialSlot` enum declared in
`framework.md`. Construction allocates the per-slot `PairBuffer`, the
`neighbor_counts` device slice (all entries equal to `N`), the
`DeviceExclusionList` (uploaded from the host `ExclusionList` produced by
the bonds parser), and the slot's three private per-atom accumulator
slices. The slot's per-step methods invoked by `ForceField::step`:

- Contribution: launch `lj_pair_force(particle_buffers, &mut pair_buffer,
  sim_box, &params, &exclusions)`, bracketed by
  `timings.kernel_start(KernelStage::LjPairForce) /
  kernel_stop(KernelStage::LjPairForce)`.
- Reduction: launch `reduce_pair_forces(&pair_buffer, &neighbor_counts,
  &mut accumulator_x, &mut accumulator_y, &mut accumulator_z, N)` (see
  the parameterised launcher in `pair-reduction.md`), bracketed by the
  `ReducePairForces` stage labels.

## Out of Scope <!-- rq-9d7966f4 -->

- Other interaction potentials (Buckingham, Morse, Coulomb, bonded terms).
- Combining rules (Lorentz–Berthelot or geometric) for multi-species
  systems; this feature treats every pair with the same `(σ, ε)`.
- Neighbor-list-driven pair force kernels (a future feature replaces the
  implicit `k = j` mapping with an explicit neighbor index table).
- Energy and virial tensor computation; this feature computes forces only.
- Long-range tail corrections and shifted/truncated potential variants.
- Numerical validation of inputs (cutoff vs. box size, σ > 0, ε > 0).
- The `f64` precision feature flag.
- Multi-stream or multi-GPU launches.

---

## Gherkin Scenarios <!-- rq-3c98d7a9 -->

```gherkin
Feature: Lennard-Jones O(N²) pair force kernel

  Background:
    Given a SimulationBox constructed with lx=20.0, ly=20.0, lz=20.0
    And LennardJonesParameters { sigma: 1.0, epsilon: 1.0, cutoff: 5.0 }

  # --- Module loading ---

  @rq-06058b71
  Scenario: init_device loads the pair_force module with the LJ kernel
    Given a CUDA-capable GPU is available as device 0
    When init_device() is called
    Then the device exposes a function named "lj_pair_force" in module "pair_force"

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
  Scenario: Pair exactly at cutoff is included
    Given a ParticleState of N=2 with positions p0=(0,0,0) and p1=(5.0, 0, 0)
    And cutoff=5.0
    When lj_pair_force is called
    Then pair_forces_x[0*2 + 1] equals the closed-form LJ force at r=5.0

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
```
