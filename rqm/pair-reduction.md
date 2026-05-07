# Feature: Pair Buffer and Deterministic Segmented Reduction <!-- rq-a406a35b -->

The simulation accumulates per-pair force contributions into a fixed-shape
device buffer and reduces that buffer to a per-particle net force in a
deterministic order. This file specifies the device data structure
(`PairBuffer`), the CUDA kernel that performs the reduction
(`kernels/reduce.cu`), and the Rust launch helper that drives it.

The reduction is the keystone of the project's bit-wise reproducibility
claim: every floating-point sum is performed on the same inputs in the same
order on every run, regardless of GPU thread scheduling.

## Data Layout <!-- rq-e435f271 -->

The pair buffer is a 2D array of shape `[particle_count, max_neighbors]` for
each Cartesian force component, stored row-major with row stride
`max_neighbors`:

```
pair_forces_x: CudaSlice<f32>  (length = particle_count * max_neighbors)
pair_forces_y: CudaSlice<f32>  (length = particle_count * max_neighbors)
pair_forces_z: CudaSlice<f32>  (length = particle_count * max_neighbors)
```

Slot `i * max_neighbors + k` holds the contribution to particle `i` from its
`k`-th neighbor. The slot is written by exactly one thread of the upstream
pair-force kernel (defined in a separate, future feature); no atomics are
involved at any layer.

A separate `CudaSlice<u32>` of length `particle_count`, named
`neighbor_counts`, records the number of populated slots in each row. It is
not part of `PairBuffer`; the future neighbor-list feature owns it. Tests
allocate a synthetic counts buffer.

`max_neighbors` and `particle_count` are fixed at `PairBuffer` construction.

## Reduction Algorithm <!-- rq-b0913965 -->

For each particle `i`, the kernel computes:

```
count = neighbor_counts[i]
sum_x = 0.0f
sum_y = 0.0f
sum_z = 0.0f
for k = 0 .. count:
    idx = i * max_neighbors + k
    sum_x = sum_x + pair_forces_x[idx]
    sum_y = sum_y + pair_forces_y[idx]
    sum_z = sum_z + pair_forces_z[idx]
net_forces_x[i] = sum_x
net_forces_y[i] = sum_y
net_forces_z[i] = sum_z
```

The summation is sequential left-to-right with one thread per particle. The
final write overwrites whatever was previously in `net_forces_*[i]` — the
reduction does not accumulate.

When `count == 0`, the loop runs zero iterations and `net_forces_*[i]` is
written as `0.0_f32`.

## Feature API <!-- rq-c7420b98 -->

### Types <!-- rq-9197d752 -->

- `PairBuffer` — host-side wrapper around the three device pair-force <!-- rq-a0c0992f -->
  buffers. All three `CudaSlice<f32>` fields are `pub` so future force
  kernels can write into them at deterministic offsets. Also carries an
  `Arc<CudaDevice>` for allocation bookkeeping.

  Fields:
  - `device: Arc<CudaDevice>`
  - `pair_forces_x: CudaSlice<f32>`
  - `pair_forces_y: CudaSlice<f32>`
  - `pair_forces_z: CudaSlice<f32>`
  - private `particle_count: usize`
  - private `max_neighbors: u32`

### CUDA Kernel <!-- rq-31bd2eee -->

`kernels/reduce.cu` declares one `extern "C"` kernel:

```c
extern "C" __global__ void reduce_pair_forces(
    const float *pair_forces_x,
    const float *pair_forces_y,
    const float *pair_forces_z,
    const unsigned int *neighbor_counts,
    unsigned int max_neighbors,
    float *net_forces_x,
    float *net_forces_y,
    float *net_forces_z,
    unsigned int n);
```

Each thread computes its global index as `blockIdx.x * blockDim.x + threadIdx.x`.
If the index is `>= n` the thread returns without touching any buffer.
Otherwise, the thread executes the reduction algorithm above for its
assigned particle.

The kernel reads `pair_forces_*` and `neighbor_counts` and writes
`net_forces_*`. It does not modify the pair-force buffers, the neighbor
counts, or any other particle state.

### PTX Module Loading <!-- rq-56d8375d -->

`init_device()` loads the compiled `kernels/reduce.cu` PTX with module name
`"reduce"` and registers function name `"reduce_pair_forces"`, alongside
the existing `fill` and `integrate` modules.

### Constructor and Accessors <!-- rq-be5fe064 -->

- `PairBuffer::new(device: Arc<CudaDevice>, particle_count: usize, max_neighbors: u32) -> Result<PairBuffer, GpuError>` <!-- rq-79048663 -->
  - Allocates three `CudaSlice<f32>` of length `particle_count * max_neighbors`
    via `CudaDevice::alloc_zeros`. Every slot starts at `0.0_f32`.
  - Returns the populated `PairBuffer`.
  - Returns `Err(GpuError)` on a CUDA driver allocation failure.
  - A `particle_count` of zero or a `max_neighbors` of zero is permitted and
    yields a buffer whose three device allocations have length zero.

- `PairBuffer::particle_count(&self) -> usize` <!-- rq-3c42e6bd -->
  - Returns the value supplied at construction.

- `PairBuffer::max_neighbors(&self) -> u32` <!-- rq-12657190 -->
  - Returns the value supplied at construction.

### Reduction Launcher <!-- rq-6f2452d1 -->

A free function in `src/gpu/kernels.rs`, re-exported from `crate::gpu`:

- `reduce_pair_forces(pair_buffer: &PairBuffer, neighbor_counts: &CudaSlice<u32>, particle_buffers: &mut ParticleBuffers) -> Result<(), GpuError>` <!-- rq-6690fae9 -->
  - Launches the `reduce_pair_forces` kernel.
  - Block size is 256; grid size is `ceil(particle_buffers.particle_count() / 256)`.
  - When `particle_buffers.particle_count() == 0`, returns `Ok(())` without
    launching a kernel.
  - Returns the underlying `GpuError` if the kernel launch fails.
  - Panics if the `"reduce"` module is not loaded on the device, since this
    indicates a programming error in `init_device()`.

  The launcher trusts the caller for shape consistency: it asserts (debug
  builds only) that `pair_buffer.particle_count() == particle_buffers.particle_count()`,
  that `neighbor_counts.len() == particle_buffers.particle_count()`, and that
  the pair-force slices have length
  `particle_buffers.particle_count() * pair_buffer.max_neighbors()`. Release
  builds skip the asserts for parity with the other kernel launchers.

## Launch Configuration <!-- rq-9be271aa -->

- Block size: 256 threads.
- Grid size: `ceil(n / 256)` blocks in the x dimension.
- Shared memory: zero bytes.
- Stream: the default stream carried by `pair_buffer.device`.

## Out of Scope <!-- rq-2b7cfbaf -->

- The pair-force kernel that fills `pair_forces_*` (a future feature).
- Construction of `neighbor_counts` (owned by the future neighbor-list
  feature).
- Bonded force terms, electrostatics, accumulation across multiple force
  kernels into the same `forces_*` buffer.
- Warp-parallel or tree-shaped reductions; this feature uses one
  thread per particle.
- Numerical validation of pair contributions (NaN/Inf propagate).
- The `f64` precision feature flag.

---

## Gherkin Scenarios <!-- rq-9561f753 -->

```gherkin
Feature: Pair buffer and deterministic segmented reduction

  # --- PairBuffer construction ---

  @rq-6fdefca0
  Scenario: PairBuffer::new allocates zero-initialised pair-force buffers
    Given a CudaDevice obtained from init_device()
    When PairBuffer::new(device, particle_count=4, max_neighbors=8) is called
    Then it returns Ok(buffer)
    And buffer.particle_count() is 4
    And buffer.max_neighbors() is 8
    And each of buffer.pair_forces_x, pair_forces_y, pair_forces_z has length 32
    And every element of each device buffer equals 0.0_f32 when downloaded to the host

  @rq-74e4bd02
  Scenario: PairBuffer::new with particle_count = 0
    Given a CudaDevice obtained from init_device()
    When PairBuffer::new(device, particle_count=0, max_neighbors=8) is called
    Then it returns Ok(buffer)
    And buffer.particle_count() is 0
    And each pair-force device buffer has length 0

  @rq-15e1e995
  Scenario: PairBuffer::new with max_neighbors = 0
    Given a CudaDevice obtained from init_device()
    When PairBuffer::new(device, particle_count=4, max_neighbors=0) is called
    Then it returns Ok(buffer)
    And buffer.max_neighbors() is 0
    And each pair-force device buffer has length 0

  # --- Module loading ---

  @rq-a43552d5
  Scenario: init_device loads the reduce module with the reduction kernel
    Given a CUDA-capable GPU is available as device 0
    When init_device() is called
    Then the device exposes a function named "reduce_pair_forces" in module "reduce"

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
  Scenario: Reduction sums entries 0..count in left-to-right order
    Given a PairBuffer with particle_count=1 and max_neighbors=4
    And pair_forces_x = [1.0, 2.0, 4.0, 999.0]
    And neighbor_counts is [3] on the device
    When reduce_pair_forces is called
    Then forces_x[0] equals (1.0_f32 + 2.0_f32) + 4.0_f32 in IEEE arithmetic
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
    Then forces_x[0] equals ((1.0_f32 + 2.0_f32) + 3.0_f32) + 4.0_f32

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
  Scenario: Block-non-aligned particle counts are handled by the bounds check
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
```
