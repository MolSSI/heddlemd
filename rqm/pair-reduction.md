# Feature: Pair Buffer and Deterministic Segmented Reduction <!-- rq-a406a35b -->

The simulation accumulates per-pair contributions to force, potential
energy, and the scalar virial into a fixed-shape device buffer and reduces
that buffer to per-particle aggregates in a deterministic order. This file
specifies the device data structure (`PairBuffer`), the CUDA kernel that
performs the reduction (`kernels/reduce.cu`), and the Rust launch helper
that drives it.

The reduction is the keystone of the project's bit-wise reproducibility
claim: every floating-point sum is performed on the same inputs in the same
order on every run, regardless of GPU thread scheduling.

## Data Layout <!-- rq-e435f271 -->

The pair buffer is a 2D array of shape `[particle_count, max_neighbors]`
for each per-pair quantity, stored row-major with row stride
`max_neighbors`:

```
pair_forces_x:  CudaSlice<f32>  (length = particle_count * max_neighbors)
pair_forces_y:  CudaSlice<f32>  (length = particle_count * max_neighbors)
pair_forces_z:  CudaSlice<f32>  (length = particle_count * max_neighbors)
pair_energies:  CudaSlice<f32>  (length = particle_count * max_neighbors)
pair_virials:   CudaSlice<f32>  (length = particle_count * max_neighbors)
```

Slot `i * max_neighbors + k` holds the contribution to particle `i` from
its `k`-th neighbor. The slot is written by exactly one thread of the
upstream pair-force kernel; no atomics are involved at any layer.

`pair_forces_*[slot]` carries the Cartesian force on particle `i` due to
the partner `j` of slot `k`. By Newton's third law the (i, j) and (j, i)
force slots hold opposite signs.

`pair_energies[slot]` carries particle `i`'s share of the pair potential
energy `u_ij`. The contribution kernel writes `u_ij / 2` so that summing
slot (i, j) plus slot (j, i) recovers `u_ij` exactly once per pair.

`pair_virials[slot]` carries particle `i`'s share of the scalar virial
`r_ij · F_ij`. The contribution kernel writes `(r_ij · F_ij) / 2` so that
summing slot (i, j) plus slot (j, i) recovers the per-pair virial exactly
once per pair.

A separate `CudaSlice<u32>` of length `particle_count`, named
`neighbor_counts`, records the number of populated slots in each row. It is
owned by the shared `NeighborListState` (see `forces/neighbor-list.md`).
Tests allocate a synthetic counts buffer when exercising the reduction in
isolation.

`max_neighbors` and `particle_count` are fixed at `PairBuffer` construction.

## Reduction Algorithm <!-- rq-b0913965 -->

Each particle `i` is reduced by one CUDA block of `BLOCK_SIZE = 256`
threads. The block computes five per-particle sums (three force
components, energy, virial) cooperatively using a fixed-shape
register-level partial accumulation followed by a fixed-shape pairwise
reduction tree.

The algorithm is identical for all five sums; the description below uses
a single quantity `q` standing for any of `pair_forces_x`, `pair_forces_y`,
`pair_forces_z`, `pair_energies`, `pair_virials`.

```text
count = neighbor_counts[i]
// Phase 1: strided sweep across the full row width.
// Every thread accumulates a register-resident partial sum p_t.
SWEEPS = (max_neighbors + BLOCK_SIZE - 1) / BLOCK_SIZE
p_t = 0.0f
for s = 0 .. SWEEPS:
    k = s * BLOCK_SIZE + threadIdx.x
    v = (k < count) ? q[i * max_neighbors + k] : 0.0f
    p_t = p_t + v
// Phase 2: warp-level pairwise tree reduction.
// Inside each 32-lane warp, the partial sums are combined by
// pairwise butterfly shuffles in log2(32) = 5 steps:
for stride in [16, 8, 4, 2, 1]:
    p_t = p_t + __shfl_xor_sync(0xffffffff, p_t, stride)
// Phase 3: inter-warp reduction.
// Lane 0 of each warp writes its warp total to a shared-memory array
// warp_q[warp_id]. After __syncthreads(), the first warp pulls its
// eight per-warp partials back into registers (lanes >= NUM_WARPS
// read 0.0f) and runs the same pairwise butterfly tree across them.
// Thread (warp 0, lane 0) writes the block total.
```

`BLOCK_SIZE = 256` and `NUM_WARPS = BLOCK_SIZE / 32 = 8` are kernel
compile-time constants.

The Phase 1 sweep width is `max_neighbors` for every particle regardless
of `count`: lanes whose `k >= count` contribute `0.0f` to their partial
sum but still issue the strided load (the load address is in range
because the pair-buffer width is `max_neighbors`). This keeps the
reduction tree shape identical across every particle in a launch.

The reduction is run-to-run bit-exact: every floating-point addition in
Phases 1–3 takes the same operands in the same order on every launch, on
the same GPU. Two runs of `reduce_pair_forces` with identical inputs
produce byte-identical output buffers.

The sum is **not** the sequential left-to-right sum
`((q[0] + q[1]) + q[2]) + ...`; it is the deterministic block tree-sum
defined above. The two values agree to within a small relative tolerance
governed by IEEE-754 round-off but generally differ in the low bits of
the f32 mantissa. The architecture explicitly permits this in
`docs/architecture.md` ("a fixed tree reduction with a deterministic
topology … as long as the tree shape depends only on the neighbor
count").

When `count == 0`, every thread's `p_t` stays at `0.0f` through Phase 1,
and the warp- and inter-warp trees propagate zero, so the five output
slices receive `0.0_f32` at index `i`.

The final writes overwrite whatever was previously in the five output
slices — the reduction does not accumulate.

## Device-side Pair-Force Frame Helper <!-- rq-73c4d574 -->

Every pair-force kernel writes into the pair buffer through a small
shared device-side helper, declared in `kernels/pair_frame.cuh` and
included by `kernels/pair_force.cu`, `kernels/coulomb.cu`, and
`kernels/spme_real.cu`. The helper centralises the universal pair-buffer
write protocol — thread→slot mapping, the three skip-the-pair guards,
the displacement and minimum-image reduction, the exclusion-scale apply,
and the per-pair `* 0.5f` halving — so each pair-force kernel reduces
to (1) a setup call, (2) the per-potential cutoff test and pair
functional form, and (3) a write call. The header declares no kernels of
its own; `init_device()` performs no `load_ptx` call for it.

The helper has three `__device__` entry points:

```c
struct PairFrame {
    bool active;
    unsigned int i;
    unsigned int j;
    unsigned int slot;
    float dx;
    float dy;
    float dz;
    float r2;
};

__device__ static inline PairFrame pair_frame_setup(
    unsigned int n,
    unsigned int max_neighbors,
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    float lx, float ly, float lz,
    float xy, float xz, float yz,
    float *pair_forces_x,
    float *pair_forces_y,
    float *pair_forces_z,
    float *pair_energies,
    float *pair_virials);

__device__ static inline void pair_frame_write_zero(
    unsigned int slot,
    float *pair_forces_x,
    float *pair_forces_y,
    float *pair_forces_z,
    float *pair_energies,
    float *pair_virials);

__device__ static inline void pair_frame_write(
    unsigned int slot,
    float fx, float fy, float fz,
    float energy,
    float virial,
    float scale,
    float *pair_forces_x,
    float *pair_forces_y,
    float *pair_forces_z,
    float *pair_energies,
    float *pair_virials);
```

`pair_frame_setup` performs the steps a pair-force kernel needs before
the per-potential math runs:

1. Computes the thread indices
   `i = blockIdx.y * blockDim.y + threadIdx.y` and
   `k = blockIdx.x * blockDim.x + threadIdx.x`. If `i >= n` or
   `k >= max_neighbors`, returns a `PairFrame` with `active == false`
   and performs no buffer writes. The thread has no assigned slot in
   this case.
2. Computes `slot = i * max_neighbors + k`.
3. If `k >= neighbor_counts[i]`, writes `0.0_f32` to all five pair-buffer
   slots at index `slot` and returns a `PairFrame` with `active == false`
   and `slot` populated.
4. Reads `j = neighbor_list[slot]`. If `i == j` (the trivial-mode
   self-pair), writes `0.0_f32` to all five pair-buffer slots and
   returns a `PairFrame` with `active == false` and `slot` populated.
5. Computes the displacement `dx = positions_x[i] - positions_x[j]` and
   similarly `dy`, `dz`. Applies the triclinic minimum-image convention
   using the six lattice parameters `(lx, ly, lz, xy, xz, yz)` defined
   in `simulation-box.md`.
6. Computes `r2 = dx*dx + dy*dy + dz*dz`.
7. Returns a `PairFrame` with `active == true`, the populated `i`, `j`,
   `slot`, `dx`, `dy`, `dz`, and `r2`. The cutoff test itself is the
   caller's responsibility: the cutoff value differs per pair potential
   (per-pair-type table for Lennard-Jones, a single global scalar for
   the Coulomb and SPME real-space slots), so the helper does not
   attempt to share it.

`pair_frame_write_zero(slot, ...)` writes `0.0_f32` to
`pair_forces_x[slot]`, `pair_forces_y[slot]`, `pair_forces_z[slot]`,
`pair_energies[slot]`, and `pair_virials[slot]`. The caller invokes it
on the cutoff-exceeded branch (after `pair_frame_setup` returned
`active == true` but the kernel's per-potential cutoff test failed).

`pair_frame_write(slot, fx, fy, fz, energy, virial, scale, ...)`
multiplies all five inputs by `scale`, multiplies `energy` and `virial`
by an additional `0.5f`, and writes the results into the five pair-buffer
slots at index `slot`. The caller computes `fx = factor * dx`,
`fy = factor * dy`, `fz = factor * dz`, and the scalar virial
`virial = fx * dx + fy * dy + fz * dz` before invoking write; the
exclusion scale comes from `exclusion_scale(...)` declared in
`kernels/exclusions.cuh` (see `forces/topology.md`). The `0.5f` factor
is what distributes each pair's energy and virial across its two slots
`(i, j)` and `(j, i)` so the segmented reduction counts each pair
exactly once when summed over all particles.

A typical pair-force kernel using the frame is structured as:

```c
extern "C" __global__ void some_pair_force(
    const float *positions_x, const float *positions_y, const float *positions_z,
    /* per-potential parameter arrays */,
    float *pair_forces_x, float *pair_forces_y, float *pair_forces_z,
    float *pair_energies, float *pair_virials,
    unsigned int max_neighbors,
    float lx, float ly, float lz, float xy, float xz, float yz,
    /* per-potential cutoff inputs and exclusion arrays */,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    unsigned int n)
{
    PairFrame f = pair_frame_setup(
        n, max_neighbors,
        positions_x, positions_y, positions_z,
        neighbor_list, neighbor_counts,
        lx, ly, lz, xy, xz, yz,
        pair_forces_x, pair_forces_y, pair_forces_z,
        pair_energies, pair_virials);
    if (!f.active) {
        return;
    }
    /* Per-potential cutoff lookup + test. */
    float cutoff = /* per-potential */;
    if (f.r2 > cutoff * cutoff) {
        pair_frame_write_zero(
            f.slot,
            pair_forces_x, pair_forces_y, pair_forces_z,
            pair_energies, pair_virials);
        return;
    }
    /* Per-potential pair functional: produces `factor` and `energy`. */
    float factor = /* per-potential */;
    float energy = /* per-potential */;
    /* Optional: per-potential switching function adjusts (factor, energy). */
    float fx = factor * f.dx;
    float fy = factor * f.dy;
    float fz = factor * f.dz;
    float virial = fx * f.dx + fy * f.dy + fz * f.dz;
    float scale = exclusion_scale(
        f.i, f.j, atom_excl_offsets, atom_excl_partners, /* lj_scales | coul_scales */);
    pair_frame_write(
        f.slot, fx, fy, fz, energy, virial, scale,
        pair_forces_x, pair_forces_y, pair_forces_z,
        pair_energies, pair_virials);
}
```

Adding a new pair potential (Buckingham, tabulated, ...) consists of
writing a new `extern "C"` kernel that follows this shape and supplies
its own cutoff source, pair functional form, switching policy, and
exclusion-scale array. The frame holds the universal protocol invariant
across every such addition.

### Determinism <!-- rq-d8a08c4a -->

Each pair-buffer slot is written by exactly one thread, whether through
`pair_frame_setup`'s skip-write path, the caller's
`pair_frame_write_zero` call, or `pair_frame_write`. There are no
atomics. The arithmetic inside `pair_frame_setup` (displacement,
minimum-image, `r2`) and inside `pair_frame_write` (scale apply, halving)
is performed in the documented order on identical inputs on every run.
Two runs of any kernel that uses the frame, with identical inputs and on
the same GPU, produce byte-identical pair-buffer contents.

### Empty state <!-- rq-efc6f7f7 -->

When `n == 0` or `max_neighbors == 0`, every thread's index check in
step 1 returns `active == false` and the kernel returns without
launching any per-slot writes. The pair-buffer slices that are length
zero in this case receive no writes.

## Feature API <!-- rq-c7420b98 -->

### Types <!-- rq-9197d752 -->

- `PairBuffer` — host-side wrapper around the five device per-pair <!-- rq-a0c0992f -->
  contribution buffers (three force components, energy, and scalar
  virial). All five `CudaSlice<f32>` fields are `pub` so contribution
  kernels can write into them at deterministic offsets. Also carries an
  `Arc<CudaDevice>` for allocation bookkeeping.

  Fields:
  - `device: Arc<CudaDevice>`
  - `pair_forces_x: CudaSlice<f32>`
  - `pair_forces_y: CudaSlice<f32>`
  - `pair_forces_z: CudaSlice<f32>`
  - `pair_energies: CudaSlice<f32>`
  - `pair_virials: CudaSlice<f32>`
  - private `particle_count: usize`
  - private `max_neighbors: u32`

### CUDA Kernel <!-- rq-31bd2eee -->

`kernels/reduce.cu` declares one `extern "C"` kernel:

```c
extern "C" __global__ void reduce_pair_forces(
    const float *pair_forces_x,
    const float *pair_forces_y,
    const float *pair_forces_z,
    const float *pair_energies,
    const float *pair_virials,
    const unsigned int *neighbor_counts,
    unsigned int max_neighbors,
    float *net_forces_x,
    float *net_forces_y,
    float *net_forces_z,
    float *net_energy,
    float *net_virial,
    unsigned int n);
```

Each block reduces one particle: `i = blockIdx.x`. If `i >= n` every
thread in the block returns without touching any buffer. Otherwise, the
block's 256 threads cooperatively execute the reduction algorithm above
for particle `i`.

Phase-3 inter-warp partial sums are exchanged through a static
shared-memory buffer declared inside the kernel as
`__shared__ float warp_partials[NUM_WARPS][5]` (one slot per warp per
quantity, `NUM_WARPS = 8`). The kernel declares no dynamic shared
memory.

The kernel reads the five input pair-contribution arrays and
`neighbor_counts`, and writes the five output per-particle arrays. It
does not modify the pair-contribution buffers, the neighbor counts, or
any other particle state.

### PTX Module Loading <!-- rq-56d8375d -->

`init_device()` loads the compiled `kernels/reduce.cu` PTX as module
`"reduce"` and captures its `reduce_pair_forces` function into the
`Kernels` handle (see `build-pipeline.md`).

### Constructor and Accessors <!-- rq-be5fe064 -->

- `PairBuffer::new(device: Arc<CudaDevice>, particle_count: usize, max_neighbors: u32) -> Result<PairBuffer, GpuError>` <!-- rq-79048663 -->
  - Allocates five `CudaSlice<f32>` of length `particle_count * max_neighbors`
    via `CudaDevice::alloc_zeros` (three forces, energy, virial). Every
    slot starts at `0.0_f32`.
  - Returns the populated `PairBuffer`.
  - Returns `Err(GpuError)` on a CUDA driver allocation failure.
  - A `particle_count` of zero or a `max_neighbors` of zero is permitted and
    yields a buffer whose five device allocations have length zero.

- `PairBuffer::particle_count(&self) -> usize` <!-- rq-3c42e6bd -->
  - Returns the value supplied at construction.

- `PairBuffer::max_neighbors(&self) -> u32` <!-- rq-12657190 -->
  - Returns the value supplied at construction.

### Reduction Launcher <!-- rq-6f2452d1 -->

A free function in `src/gpu/kernels.rs`, re-exported from `crate::gpu`:

- `reduce_pair_forces(pair_buffer: &PairBuffer, neighbor_counts: &CudaSlice<u32>, target_force_x: &mut CudaViewMut<'_, f32>, target_force_y: &mut CudaViewMut<'_, f32>, target_force_z: &mut CudaViewMut<'_, f32>, target_energy: &mut CudaViewMut<'_, f32>, target_virial: &mut CudaViewMut<'_, f32>, particle_count: usize) -> Result<(), GpuError>` <!-- rq-6690fae9 -->
  - Launches the `reduce_pair_forces` kernel against the five caller-
    supplied target buffers. The targets receive the per-particle net
    force (three components), per-particle potential-energy share, and
    per-particle scalar-virial share from the pair-buffer reduction
    (overwriting any prior contents).
  - Block size is 256; grid size is `particle_count` blocks (one block
    per particle).
  - When `particle_count == 0`, returns `Ok(())` without launching a
    kernel.
  - Returns the underlying `GpuError` if the kernel launch fails.
  - Invokes the kernel through the `reduce_pair_forces` field of the
    `Kernels` handle reached from its arguments; it performs no
    string-keyed kernel lookup of its own (see `build-pipeline.md`).

  The launcher trusts the caller for shape consistency: it asserts (debug
  builds only) that `pair_buffer.particle_count() == particle_count`,
  that `neighbor_counts.len() == particle_count`, that each of the five
  target views has length `particle_count`, and that the pair-buffer
  slices have length `particle_count * pair_buffer.max_neighbors()`.
  Release builds skip the asserts for parity with the other kernel
  launchers.

  Within the pluggable potential framework, the `LennardJonesState` slot
  passes its assigned rows of the framework's flat slot-output buffers
  as the five targets (see `forces/framework.md`).

## Launch Configuration <!-- rq-9be271aa -->

- Block size: 256 threads (8 warps of 32).
- Grid size: `n` blocks in the x dimension — one block per particle.
- Shared memory: static, `NUM_WARPS * 5 * sizeof(f32) = 160` bytes per
  block (a `[NUM_WARPS][5]` array of per-warp partials, where
  `NUM_WARPS = blockDim.x / 32 = 8`). The kernel requests no dynamic
  shared memory at launch.
- Stream: the default stream carried by `pair_buffer.device`.

## Out of Scope <!-- rq-2b7cfbaf -->

- The pair-force kernel that fills `pair_forces_*` (a future feature).
- Construction of `neighbor_counts` (owned by the future neighbor-list
  feature).
- Bonded force terms, electrostatics, accumulation across multiple force
  kernels into the same `forces_*` buffer.
- Sub-block parallelism: multiple particles per block, multi-block
  cooperative reductions over a single particle, persistent-kernel
  schemes. One block reduces exactly one particle.
- Bit-exact equality with a sequential left-to-right CPU reference sum:
  the kernel uses a deterministic block tree reduction whose f32 result
  differs from the scalar serial sum in the low mantissa bits.
- Numerical validation of pair contributions (NaN/Inf propagate).
- The `f64` precision feature flag.

---

## Gherkin Scenarios <!-- rq-9561f753 -->

```gherkin
Feature: Pair buffer and deterministic segmented reduction

  # --- PairBuffer construction ---

  @rq-6fdefca0
  Scenario: PairBuffer::new allocates zero-initialised pair-contribution buffers
    Given a GpuContext obtained from init_device()
    When PairBuffer::new(device, particle_count=4, max_neighbors=8) is called
    Then it returns Ok(buffer)
    And buffer.particle_count() is 4
    And buffer.max_neighbors() is 8
    And each of buffer.pair_forces_x, pair_forces_y, pair_forces_z, pair_energies, pair_virials has length 32
    And every element of each device buffer equals 0.0_f32 when downloaded to the host

  @rq-74e4bd02
  Scenario: PairBuffer::new with particle_count = 0
    Given a GpuContext obtained from init_device()
    When PairBuffer::new(device, particle_count=0, max_neighbors=8) is called
    Then it returns Ok(buffer)
    And buffer.particle_count() is 0
    And each pair-force device buffer has length 0

  @rq-15e1e995
  Scenario: PairBuffer::new with max_neighbors = 0
    Given a GpuContext obtained from init_device()
    When PairBuffer::new(device, particle_count=4, max_neighbors=0) is called
    Then it returns Ok(buffer)
    And buffer.max_neighbors() is 0
    And each pair-force device buffer has length 0

  # --- Module loading ---

  @rq-a43552d5
  Scenario: init_device exposes the reduction kernel on the Kernels handle
    Given a CUDA-capable GPU is available as device 0
    When init_device() is called
    Then the returned GpuContext's kernels handle exposes the reduce_pair_forces function

  # --- Reduction correctness: trivial cases ---

  @rq-2d051b0c
  Scenario: Reduction with all zero neighbor counts produces zero net forces
    Given a PairBuffer with particle_count=4 and max_neighbors=8
    And the pair_forces_* buffers contain arbitrary nonzero values
    And neighbor_counts is [0, 0, 0, 0] on the device
    And ParticleBuffers built from a state of 4 particles with nonzero forces
    When reduce_pair_forces(&pair_buffer, &counts, &mut particle_buffers) is called
    And particle_buffers is downloaded to a host ParticleState
    Then forces_x, forces_y, forces_z are each [0.0, 0.0, 0.0, 0.0]

  @rq-8ee33aa0
  Scenario: Reduction with a single particle and a single neighbor
    Given a PairBuffer with particle_count=1 and max_neighbors=4
    And pair_forces_x[0] = 1.5, pair_forces_y[0] = -2.5, pair_forces_z[0] = 0.75
    And neighbor_counts is [1] on the device
    When reduce_pair_forces is called
    Then forces_x[0] equals 1.5
    And forces_y[0] equals -2.5
    And forces_z[0] equals 0.75

  # --- Reduction correctness: order and bounds ---

  @rq-e950f4e6
  Scenario: Reduction sums slots 0..count and excludes slots beyond count
    Given a PairBuffer with particle_count=1 and max_neighbors=4
    And pair_forces_x = [1.0, 2.0, 4.0, 999.0]
    And neighbor_counts is [3] on the device
    When reduce_pair_forces is called
    Then forces_x[0] equals 7.0_f32
    And the slot at index 3 (value 999.0) is not included

  @rq-78fc2fbb
  Scenario: Slots beyond neighbor_counts[i] are not summed
    Given a PairBuffer with particle_count=1 and max_neighbors=8
    And pair_forces_x[0..2] = [10.0, 20.0]
    And pair_forces_x[2..8] = [f32::INFINITY; 6]
    And neighbor_counts is [2] on the device
    When reduce_pair_forces is called
    Then forces_x[0] equals 30.0_f32
    And forces_x[0] is finite

  @rq-590dcd7e
  Scenario: Reduction at full max_neighbors capacity
    Given a PairBuffer with particle_count=1 and max_neighbors=4
    And pair_forces_x = [1.0, 2.0, 3.0, 4.0]
    And neighbor_counts is [4] on the device
    When reduce_pair_forces is called
    Then forces_x[0] equals 10.0_f32

  # --- Reduction correctness: multiple particles ---

  @rq-6808532e
  Scenario: Per-particle reduction with varying counts
    Given a PairBuffer with particle_count=3 and max_neighbors=4
    And pair_forces_x for particle 0 (slots 0..4) = [1.0, 2.0, 100.0, 100.0]
    And pair_forces_x for particle 1 (slots 4..8) = [10.0, 100.0, 100.0, 100.0]
    And pair_forces_x for particle 2 (slots 8..12) = [0.5, 0.5, 0.5, 0.5]
    And neighbor_counts is [2, 1, 4] on the device
    When reduce_pair_forces is called
    Then forces_x[0] equals 3.0_f32
    And forces_x[1] equals 10.0_f32
    And forces_x[2] equals 2.0_f32

  # --- Empty state ---

  @rq-493caf32
  Scenario: reduce_pair_forces on an empty state is a no-op
    Given a PairBuffer with particle_count=0 and max_neighbors=8
    And ParticleBuffers built from an empty ParticleState
    And an empty neighbor_counts CudaSlice<u32>
    When reduce_pair_forces is called
    Then it returns Ok(())

  # --- Bounds handling ---

  @rq-77e88745
  Scenario: Every particle is reduced exactly once across the full grid
    Given a PairBuffer with particle_count=1000 and max_neighbors=2
    And every pair_forces_x[i*2] = i and pair_forces_x[i*2+1] = -i
    And every pair_forces_y, pair_forces_z slot is 0.0
    And neighbor_counts is [2; 1000] on the device
    When reduce_pair_forces is called
    Then forces_x[i] equals 0.0 for every i in 0..1000
    And forces_y[i] equals 0.0 for every i in 0..1000
    And forces_z[i] equals 0.0 for every i in 0..1000

  # --- Side effects ---

  @rq-f5299d6e
  Scenario: Reduction overwrites prior net force values
    Given a ParticleBuffers whose forces_x is [99.0, 88.0, 77.0, 66.0]
    And a PairBuffer with particle_count=4 and max_neighbors=2 holding zeroes
    And neighbor_counts is [0, 0, 0, 0] on the device
    When reduce_pair_forces is called
    Then forces_x equals [0.0, 0.0, 0.0, 0.0]

  @rq-9b794cff
  Scenario: Reduction does not modify the pair buffer
    Given a PairBuffer with particle_count=4 and max_neighbors=4 with known nonzero values
    And neighbor_counts is [3, 3, 3, 3] on the device
    And a snapshot of pair_forces_x, pair_forces_y, pair_forces_z before launch
    When reduce_pair_forces is called
    And each pair-force slice is downloaded to the host
    Then the downloaded slices are byte-identical to the snapshot

  @rq-c9da25a7
  Scenario: Reduction does not modify positions, velocities, or masses
    Given a ParticleBuffers built from a known ParticleState with particle_count=4
    And a PairBuffer with all-zero pair forces
    And neighbor_counts is [0, 0, 0, 0]
    And a snapshot of positions_*, velocities_*, masses, particle_ids before launch
    When reduce_pair_forces is called
    And particle_buffers is downloaded to a host ParticleState
    Then positions_x, positions_y, positions_z, velocities_x, velocities_y, velocities_z, masses, and particle_ids are byte-identical to the snapshot

  @rq-eb3a65df
  Scenario: Reduction does not modify neighbor_counts
    Given a PairBuffer with particle_count=4 and max_neighbors=2
    And a neighbor_counts CudaSlice<u32> initialised to [0, 1, 2, 0]
    When reduce_pair_forces is called
    And neighbor_counts is downloaded to the host
    Then the downloaded values are [0, 1, 2, 0]

  # --- Reproducibility ---

  @rq-b4f18ea1
  Scenario: Two independent runs produce byte-identical net forces
    Given two independently-constructed PairBuffers and ParticleBuffers populated with identical contents at particle_count=128 and max_neighbors=16
    And identical neighbor_counts in both runs
    When reduce_pair_forces is launched on each
    And both forces_x, forces_y, forces_z are downloaded to the host
    Then run A and run B agree byte-for-byte on every f32

  # --- Block tree reduction shape ---

  @rq-b2221a99
  Scenario: Reduction with count_i larger than one warp uses the inter-warp tree
    Given a PairBuffer with particle_count=1 and max_neighbors=128
    And pair_forces_x[0..96] = [1.0; 96]
    And pair_forces_x[96..128] = [0.0; 32]
    And neighbor_counts is [96] on the device
    When reduce_pair_forces is called
    Then forces_x[0] equals 96.0_f32

  @rq-c009903e
  Scenario: Reduction with count_i larger than one block sweep accumulates across sweeps
    Given a PairBuffer with particle_count=1 and max_neighbors=1024
    And pair_forces_x[0..600] = [1.0; 600]
    And pair_forces_x[600..1024] = [99.0; 424]
    And neighbor_counts is [600] on the device
    When reduce_pair_forces is called
    Then forces_x[0] equals 600.0_f32
    And forces_x[0] is finite

  @rq-aee2bfb2
  Scenario: Reduction tree result agrees with a CPU pairwise tree reference to within a small relative tolerance
    Given a PairBuffer with particle_count=1 and max_neighbors=1024
    And pair_forces_x[k] = sin(k * 0.1) for k in 0..800
    And neighbor_counts is [800] on the device
    When reduce_pair_forces is called
    And a CPU reference sum is computed using the same block tree reduction shape
    Then forces_x[0] equals the CPU reference exactly
    And forces_x[0] is within 1e-5 relative tolerance of the sequential left-to-right f32 sum

  @rq-46d24bfb
  Scenario: One block per particle, blocks are independent
    Given a PairBuffer with particle_count=512 and max_neighbors=64
    And neighbor_counts populated with a deterministic pseudo-random distribution in 0..=64
    And pair_forces_x populated with deterministic pseudo-random f32 in [-1.0, 1.0]
    When reduce_pair_forces is called
    And the same reduction is performed independently on the host for each particle using the same block tree shape
    Then every forces_x[i] equals the host reference value exactly

  # --- Numerical edge cases ---

  @rq-5cb58365
  Scenario: NaN pair contributions propagate to NaN net forces
    Given a PairBuffer with particle_count=1 and max_neighbors=4
    And pair_forces_x = [1.0, f32::NAN, 3.0, 0.0]
    And neighbor_counts is [3]
    When reduce_pair_forces is called
    Then forces_x[0] is NaN

  @rq-a1c567b3
  Scenario: Infinite pair contributions propagate to infinite net forces
    Given a PairBuffer with particle_count=1 and max_neighbors=4
    And pair_forces_x = [1.0, f32::INFINITY, 3.0, 0.0]
    And neighbor_counts is [3]
    When reduce_pair_forces is called
    Then forces_x[0] is positive infinity

  # --- Energy and virial reduction ---

  @rq-9e487c80
  Scenario: Reduction sums pair energies and virials over slots 0..count
    Given a PairBuffer with particle_count=1 and max_neighbors=4
    And pair_energies = [0.5, 1.5, 2.0, 999.0]
    And pair_virials = [-1.0, 2.0, 3.0, 0.0]
    And neighbor_counts is [3]
    When reduce_pair_forces is called
    Then net_energy[0] equals 4.0_f32
    And net_virial[0] equals 4.0_f32

  @rq-961c2ee6
  Scenario: Reduction with zero count writes zero to energy and virial targets
    Given a PairBuffer with particle_count=2 and max_neighbors=4
    And pair_energies and pair_virials contain arbitrary nonzero values
    And neighbor_counts is [0, 0]
    When reduce_pair_forces is called
    Then net_energy and net_virial are each [0.0, 0.0]

  @rq-41d9e514
  Scenario: Energy and virial reductions share the same indexing as force
    Given a PairBuffer with particle_count=2 and max_neighbors=2
    And pair_forces_x = [1.0, 2.0, 3.0, 4.0]
    And pair_energies = [10.0, 20.0, 30.0, 40.0]
    And pair_virials  = [100.0, 200.0, 300.0, 400.0]
    And neighbor_counts is [2, 2]
    When reduce_pair_forces is called
    Then net_forces_x[0] equals 3.0 and net_forces_x[1] equals 7.0
    And net_energy[0] equals 30.0 and net_energy[1] equals 70.0
    And net_virial[0] equals 300.0 and net_virial[1] equals 700.0
```
