# Feature: Fused Pair-Force Kernel Pattern <!-- rq-d7a35317 -->

Every short-range pairwise potential in the engine (Lennard-Jones,
truncated Coulomb, SPME real space, and any future addition) evaluates
its per-pair contribution and accumulates the per-particle net force
in a single CUDA kernel per (potential, AggregateLevel) variant. The
kernel walks each particle's neighbour list, computes the per-pair
force in registers, and reduces it to a per-particle total through a
warp-tree butterfly — no per-pair intermediate buffer is materialised
on the device.

The deterministic warp-strided reduction is the keystone of the
project's bit-wise reproducibility claim: every floating-point
addition that contributes to particle `i`'s net force takes the same
operands in the same lane order on every run regardless of GPU thread
scheduling.

This file specifies the kernel topology, the shared `kernels/pair_compute.cuh`
device-side helper, the per-potential kernel-variant pattern, the
launch configuration, and the reproducibility argument. The per-pair
functional form (Lennard-Jones, Coulomb, erfc-screened Coulomb, ...)
is specified in the per-potential files (`lj-pair-force.md`,
`coulomb-pair-force.md`, `spme.md`).

## Kernel Topology <!-- rq-b73c3b0b -->

Each particle `i` is processed by one CUDA warp — 32 lanes — running
inside a 256-thread block (`WARPS_PER_BLOCK = 8`). A block carries
8 warps, each handling a different particle; warps within a block
share no data. With `n` particles, the grid is `ceil(n / 8)` blocks.
Trailing warps in the last block whose assigned particle index lies
past `n` return without writing, and the early-return is taken on the
uniform condition `i >= n` so all 32 lanes of the warp return
together.

Per warp (handling particle `i`):

```text
warp_id_in_block = threadIdx.x / WARP_SIZE
lane             = threadIdx.x & (WARP_SIZE - 1)
i                = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block
if i >= n: return                              // entire warp returns

count = neighbor_counts[i]
SWEEPS = (count + WARP_SIZE - 1) / WARP_SIZE
p_x = p_y = p_z = 0.0f                         // forces variant + fev
p_e = p_w = 0.0f                               // fev variant only

for s = 0 .. SWEEPS:
    k = s * WARP_SIZE + lane
    if k < count:
        j = neighbor_list[i * max_neighbors + k]
        if i == j:                             // trivial-mode self-pair
            // contribute zero; nothing to accumulate
        else:
            (dx, dy, dz) = positions[i] - positions[j]
            triclinic_min_image(dx, dy, dz, box)
            r2 = dx*dx + dy*dy + dz*dz
            if r2 <= r_cut * r_cut:
                (factor, energy, virial) = per_potential_form(r2, params)
                scale = exclusion_scale(i, j, ...)
                fx = factor * dx * scale
                fy = factor * dy * scale
                fz = factor * dz * scale
                p_x += fx; p_y += fy; p_z += fz
                // fev variant additionally:
                p_e += energy * scale * 0.5
                p_w += virial * scale * 0.5

// Warp-pairwise butterfly tree, 5 steps:
for stride in [16, 8, 4, 2, 1]:
    p_x += __shfl_xor_sync(0xffffffff, p_x, stride)
    p_y += __shfl_xor_sync(0xffffffff, p_y, stride)
    p_z += __shfl_xor_sync(0xffffffff, p_z, stride)
    // fev variant additionally reduces p_e and p_w through the same tree

if lane == 0:
    slot_force_x[i] = p_x
    slot_force_y[i] = p_y
    slot_force_z[i] = p_z
    // fev variant additionally:
    slot_energy[i]  = p_e
    slot_virial[i]  = p_w
```

`WARPS_PER_BLOCK = 8`, `WARP_SIZE = 32`, and the resulting
`BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK = 256` are kernel
compile-time constants shared by every fused pair-force kernel.

The `0.5f` factor on the energy and virial accumulators distributes
each unordered pair's energy and virial across its two ordered slots
`(i, j)` and `(j, i)`. Warp `W_i` adds `0.5 * U_ij` to particle
`i`'s energy share; warp `W_j` independently adds `0.5 * U_ij` to
particle `j`'s. Summing over every particle recovers the unordered
pair sum exactly once.

The Phase-1 sweep loop runs `(count + WARP_SIZE - 1) / WARP_SIZE`
sweeps, scaling with the populated row width rather than
`max_neighbors`. For `count == 0` the loop runs zero sweeps. The
kernel relies on the contract `count <= max_neighbors` (see
`neighbor-list.md`) for its in-row bound — the active mask
`k < count` covers it without a separate `k < max_neighbors` guard.

The warp tree always executes its fixed 5-step butterfly regardless
of `count`, so the reduction tree shape applied to the 32 lanes'
partial sums is identical across every particle in a launch.

## Device-side Pair Compute Helper <!-- rq-2adca0ab -->

`kernels/pair_compute.cuh` declares a small `__device__` helper that
centralises the per-warp scaffolding — thread → particle mapping,
sweep-loop bookkeeping, j load, displacement and minimum-image,
self-pair guard, exclusion-scale lookup, the per-component register
accumulation, the warp-tree butterfly, and the lane-0 slot-output
write. Each pair-force kernel calls into this helper, supplying a
per-potential `__device__` functor (or inline-able function) that
takes `(r2, params)` and returns the triple `(factor, energy, virial)`
together with a cutoff predicate.

The helper has one entry point per AggregateLevel variant:

```c
template <typename PairFunc>
__device__ static inline void pair_compute_f(
    PairFunc per_pair,                         // returns (factor) on cutoff-passing pair
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    unsigned int max_neighbors,
    const Real *lattice,            // length 6: [lx, ly, lz, xy, xz, yz]
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const Real *atom_excl_scales,
    Real *slot_force_x,
    Real *slot_force_y,
    Real *slot_force_z,
    unsigned int n);

template <typename PairFunc>
__device__ static inline void pair_compute_fev(
    PairFunc per_pair,                         // returns (factor, energy, virial)
    /* same kernel-parameter inputs as pair_compute_f */
    Real *slot_force_x,
    Real *slot_force_y,
    Real *slot_force_z,
    Real *slot_energy,
    Real *slot_virial,
    unsigned int n);
```

`PairFunc` is a stateless device functor whose `operator()(r2, params)`
returns either a `Real` factor (for `pair_compute_f`) or a triple of
`Real` values (for `pair_compute_fev`). It also reports whether the
pair survives the per-potential cutoff test; the helper treats a
non-survivor identically to an unpopulated slot (no accumulation).

The helper does not declare static or dynamic shared memory and uses
only register accumulators and `__shfl_xor_sync`. The per-potential
kernel reduces to:

```c
extern "C" __global__ void <potential>_pair_force_f(
    /* per-potential parameter arrays */,
    /* shared kernel-parameter inputs */,
    Real *slot_force_x,
    Real *slot_force_y,
    Real *slot_force_z,
    unsigned int n)
{
    pair_compute_f(<PerPotentialPairFunc>{ /* per-potential params */ }, ...);
}

extern "C" __global__ void <potential>_pair_force_fev(
    /* per-potential parameter arrays */,
    /* shared kernel-parameter inputs */,
    Real *slot_force_x,
    Real *slot_force_y,
    Real *slot_force_z,
    Real *slot_energy,
    Real *slot_virial,
    unsigned int n)
{
    pair_compute_fev(<PerPotentialPairFunc>{ /* per-potential params */ }, ...);
}
```

Adding a new pair potential (Buckingham, tabulated, ...) consists of
writing a new `PerPotentialPairFunc` device functor and two thin
`extern "C"` kernels that invoke `pair_compute_f` / `pair_compute_fev`
with it. The helper holds the universal reduction protocol invariant
across every such addition.

The per-potential `<potential>_pair_force_{f,fev}` kernels described
in this file are the standalone-testing entry points: every unit
test of a per-pair functional form launches the matching
per-potential kernel directly. The framework's per-step force
evaluation does *not* dispatch the per-potential kernels — fast-class
pair-force slots participate in the JIT-composed pair-force kernel
described in `jit-composed-pair-force.md`, which walks the neighbour
list once and accumulates every active slot's per-pair contribution
in registers before the warp-tree reduction. Both paths use the same
sweep order and the same reduction shape, so a per-potential kernel's
output equals one-slot composed-kernel output bit-for-bit.

## Kernel Variants <!-- rq-7b27c75b -->

Each pair-force potential exposes two `extern "C"` kernel entry
points:

- `<potential>_pair_force_f` — writes the three per-particle force
  components into the slot output buffer; does not write energy or
  virial. Selected when the framework call site has
  `AggregateLevel::ForcesOnly`.
- `<potential>_pair_force_fev` — writes the three per-particle force
  components plus the per-particle potential-energy share and the
  per-particle scalar-virial share. Selected when the framework call
  site has `AggregateLevel::ForcesAndScalars`.

The two variants share their force computation and the warp-tree
butterfly; the fev variant additionally accumulates the energy and
virial partial sums and writes them to the corresponding slot output
rows. The compiler dead-code-eliminates the unused accumulation in
the f variant.

## Launch Configuration <!-- rq-63726e30 -->

- Block size: 256 threads (8 warps × 32 lanes) for both variants of
  every pair-force kernel.
- Grid size: `ceil(n / 8)` blocks in the x dimension.
- Shared memory: none. Neither variant declares static or dynamic
  shared memory; the warp-pairwise tree runs entirely through
  `__shfl_xor_sync`.
- Stream: the default stream carried by the calling Potential's
  `device` for the truncated-Coulomb and LJ potentials. The SPME
  real-space variant runs on the default stream while the reciprocal
  pipeline owns its own stream (see `spme.md`).

## Reproducibility <!-- rq-6e191fd0 -->

The reduction is run-to-run bit-exact: every floating-point addition
in the sweep and the warp tree takes the same operands in the same
order on every launch, on the same GPU. Two runs of any
`<potential>_pair_force_*` kernel with identical inputs produce
byte-identical slot output buffers.

The sum is **not** the sequential left-to-right sum
`((c[0] + c[1]) + c[2]) + ...` over contributions in neighbour-list
order; it is the deterministic warp tree-sum defined above. The two
values agree to within a small relative tolerance governed by
IEEE-754 round-off but generally differ in the low bits of the f32
mantissa. The architecture explicitly permits this in
`docs/architecture.md` ("a fixed tree reduction with a deterministic
topology … as long as the tree shape depends only on the neighbor
count").

When `count == 0`, every lane's accumulators stay at `0.0f` through
the sweep, the warp tree propagates zero, and lane 0 adds `0.0_f32`
to every output slice at index `i` (a no-op load-store).

The final writes are read-modify-writes into class-accumulator slices
(see `framework.md`'s *Class Output Accumulators*): lane 0 loads the
current value at index `i`, adds the warp's per-particle reduced
contribution, and stores the sum. Each `(class, particle)`
accumulator cell is written by exactly one warp per slot per launch,
so no atomics are needed.

## Newton's Third Law <!-- rq-f5522d5f -->

For an unordered pair `{i, j}`, warp `W_i` computes `F_ij` and adds
it to particle `i`'s register accumulators while warp `W_j`
independently computes `F_ji = -F_ij` and adds it to particle `j`'s.
The two computations share no device state. By Newton's third law
their per-component results are exact negatives (modulo the
measure-zero exact-boundary minimum-image case `dx = ±L/2`).
Summing the kernel's output over all particles therefore yields zero
to within floating-point round-off for an isolated system.

The kernel does NOT exploit pair symmetry to skip the j-side
computation; doing so would force an atomic or shared-memory write
across warps and violate the per-particle deterministic-order
invariant. Both directions are computed independently, and the
arithmetic in each warp is identical to its mirror in the other.

## Feature API <!-- rq-e1345339 -->

### CUDA Kernel Entry Points <!-- rq-05e4b5da -->

Two `extern "C"` kernels per pair-force potential, both following the
common pattern documented above. Per-potential parameter lists are
specified in the per-potential files:

- `lj_pair_force_f`, `lj_pair_force_fev` — see `lj-pair-force.md`. <!-- rq-a783727a -->
- `coulomb_pair_force_f`, `coulomb_pair_force_fev` — see <!-- rq-28815c89 -->
  `coulomb-pair-force.md`.
- `spme_real_pair_force_f`, `spme_real_pair_force_fev` — see <!-- rq-7c59a571 -->
  `spme.md`.

### PTX Module Loading <!-- rq-f0753132 -->

`init_device()` loads the compiled PTX of each pair-force `.cu`
translation unit as its respective module and captures both kernel
variants into the `Kernels` handle (see `build-pipeline.md`).

### Launchers <!-- rq-8e8eef43 -->

Per-potential launcher functions in `src/gpu/kernels.rs` accept the
`AggregateLevel` value handed to them by the slot's `Potential::compute`
implementation and dispatch to the `_f` or `_fev` variant accordingly.
Per-potential launcher signatures are specified in the per-potential
files.

Every launcher reads the slot's parameter tables, the shared
`NeighborListState`, the shared `DeviceExclusionList`, and the
`ParticleBuffers` positions as immutable inputs; the only mutable
outputs are the slot output buffer slices handed to the launcher by
the framework (see `framework.md`).

When `particle_count == 0`, every launcher returns `Ok(())` without
launching a kernel.

## Launch Configuration Constants <!-- rq-5946ee09 -->

`src/gpu/kernels.rs` exposes `PAIR_FORCE_WARPS_PER_BLOCK = 8` and
`PAIR_FORCE_BLOCK_SIZE = 256` as `pub const u32`. Both pair-force and
reduce kernels use these constants; they match the
`WARPS_PER_BLOCK` / `BLOCK_SIZE` compile-time values declared in
`kernels/pair_compute.cuh`.

## Out of Scope <!-- rq-605f2b43 -->

- Per-pair intermediate device buffers. The fused kernels accumulate
  in registers; no `PairBuffer` type exists in the engine.
- Sub-warp parallelism: multi-warp cooperation on a single particle,
  multi-block cooperative reductions over a single particle,
  persistent-kernel schemes. One warp reduces exactly one particle.
- Newton's-third-law symmetry optimisation that visits each unordered
  pair only once. Both directions are computed redundantly to keep
  the per-particle accumulation deterministic and atomic-free.
- Bit-exact equality with a sequential left-to-right CPU reference
  sum over neighbour-list order: the kernel uses a deterministic warp
  tree reduction whose f32 result differs from the scalar serial sum
  in the low mantissa bits.
- Numerical validation of pair contributions (NaN/Inf propagate to
  the per-particle output).
- Bonded and angle potentials. Those slots use their own bond/angle
  index tables and per-bond / per-angle intermediates and do not
  follow this pattern; see `morse-bonded.md` and `harmonic-angle.md`.

---

## Gherkin Scenarios <!-- rq-5c12bfee -->

The scenarios below cover the warp-per-particle pattern itself —
topology, sweep semantics, reduction shape, Newton's third law,
variant selection. Per-potential functional-form scenarios (closed-
form factor agreement, switching-function behaviour, exclusion-table
attenuation) live in the per-potential files.

Scenarios prefaced with "any pair-force potential P" are
implementation-parameterised: every such scenario runs three times,
once each with P = Lennard-Jones, P = truncated Coulomb, and
P = SPME real-space. Each instantiation supplies the per-pair
functional form, parameter table, and exclusion-scale array
appropriate to that potential. The expected results are computed on
the host using the same per-pair functional form so the assertion is
the same shape across all three runs.

```gherkin
Feature: Fused warp-per-particle pair-force kernel pattern

  # --- Grid layout ---

  @rq-1e135c50
  Scenario: Particle count that is not a multiple of WARPS_PER_BLOCK uses a ragged final block
    Given any pair-force potential P
    And a ParticleState with particle_count = 10
    And neighbor_counts is [1; 10] on the device, with each particle's first neighbour set so the per-pair force is 1.0 in the x component
    When the _f variant of P is launched
    Then the launch uses ceil(10 / 8) = 2 blocks of 256 threads each
    And slot_force_x[i] equals 1.0 for every i in 0..10
    And the warps 2..8 of the second block (with particle index >= 10) return without writing past index 9

  @rq-ae1b567a
  Scenario: Particle count below WARPS_PER_BLOCK uses a single under-full block
    Given any pair-force potential P
    And a ParticleState with particle_count = 3
    And neighbor_counts is [1, 1, 1] on the device with each pair contributing 2.0 in the x component
    When the _f variant of P is launched
    Then the launch uses 1 block of 256 threads
    And slot_force_x[i] equals 2.0 for every i in 0..3
    And warps 3..8 of the single block return without writing

  @rq-1708fc0a
  Scenario: Pair-force kernels declare no shared memory
    Given a CUDA-capable GPU is available as device 0
    When init_device() is called
    Then the lj_pair_force_f kernel's sharedSizeBytes attribute is 0
    And the lj_pair_force_fev kernel's sharedSizeBytes attribute is 0
    And the coulomb_pair_force_f kernel's sharedSizeBytes attribute is 0
    And the coulomb_pair_force_fev kernel's sharedSizeBytes attribute is 0
    And the spme_real_pair_force_f kernel's sharedSizeBytes attribute is 0
    And the spme_real_pair_force_fev kernel's sharedSizeBytes attribute is 0

  # --- Sweep semantics ---

  @rq-5568bd21
  Scenario: count == 0 yields zero per-particle output
    Given any pair-force potential P
    And a ParticleState with particle_count = 4
    And neighbor_counts is [0, 0, 0, 0] on the device
    And the slot_force_x/y/z target buffers initialised to known nonzero patterns
    When the _f variant of P is launched
    Then slot_force_x, slot_force_y, slot_force_z are all [0.0, 0.0, 0.0, 0.0]

  @rq-5e2ee310
  Scenario: Sweep reads only slots [0, count) and ignores trailing junk
    Given any pair-force potential P
    And a ParticleState with particle_count = 1
    And neighbor_counts is [3] on the device
    And the neighbor_list at slot 3 is populated with a particle ID that, if read, would induce an NaN in the per-pair functional form
    When the _f variant of P is launched
    Then slot_force_x[0], slot_force_y[0], slot_force_z[0] are all finite

  @rq-dce96f48
  Scenario: Sweep loop runs zero iterations when count == 0
    Given a ParticleState with particle_count = 1 and a single particle whose neighbor_counts entry is 0
    And the per-pair functional form for P would write NaN to its outputs if evaluated on any neighbour
    When the _f variant of P is launched
    Then slot_force_x[0], slot_force_y[0], slot_force_z[0] are all 0.0_f32

  @rq-ab8046b4
  Scenario: Self-pair (i == j) in a trivial-mode neighbour list contributes zero
    Given any pair-force potential P
    And a ParticleState with particle_count = 2 and positions p0 = (0, 0, 0), p1 = (1.5, 0, 0)
    And neighbor_counts is [2, 2] on the device
    And neighbor_list is [0, 1, 1, 0] — each particle's list contains itself plus the other particle
    When the _f variant of P is launched
    Then slot_force_x[0], slot_force_y[0], slot_force_z[0] are all finite
    And slot_force_x[0] equals the closed-form per-pair x-force for the (0, 1) pair only
    And slot_force_x[1] equals the closed-form per-pair x-force for the (1, 0) pair only

  # --- Reduction shape ---

  @rq-3982aff8
  Scenario: Result of any pair-force kernel agrees with a CPU warp-tree reference exactly
    Given any pair-force potential P with deterministic per-pair functional form per_pair(r2)
    And a ParticleState of N = 64 with deterministic pseudo-random positions
    And a deterministic neighbor list with neighbor_counts in [0, max_neighbors]
    When the _f variant of P is launched
    And a CPU reference sum is computed using the same warp tree reduction shape (32 strided partial sums combined by a 5-step pairwise butterfly)
    Then every slot_force_x[i], slot_force_y[i], slot_force_z[i] equals the CPU reference exactly

  @rq-d7cf2ae0
  Scenario: Sweep handles the iteration boundary exactly at WARP_SIZE
    Given any pair-force potential P
    And a ParticleState with particle_count = 2 placed so the per-pair force on particle 0 due to particle 1 is finite and nonzero
    And neighbor_counts is [32, 0] on the device
    And the first 32 slots of particle 0's neighbour list all hold the partner index 1 (the same partner 32 times)
    When the _f variant of P is launched
    Then slot_force_x[0] equals 32 times the closed-form per-pair x-force for the (0, 1) pair, summed via the warp tree
    And the sweep loop executes exactly one full iteration (SWEEPS = ceil(32 / 32) = 1)

  @rq-97db6cec
  Scenario: Sweep loop accumulates correctly when count is one more than WARP_SIZE
    Given any pair-force potential P
    And a ParticleState with particle_count = 2 placed so the per-pair force on particle 0 due to particle 1 is finite and nonzero
    And neighbor_counts is [33, 0] on the device
    And the first 33 slots of particle 0's neighbour list all hold the partner index 1
    When the _f variant of P is launched
    Then slot_force_x[0] equals 33 times the closed-form per-pair x-force for the (0, 1) pair, summed via the warp tree (lane 0 of the second sweep iteration contributes the 33rd copy)

  @rq-63608bb9
  Scenario: Sweep loop runs three full iterations when count is an exact multiple of WARP_SIZE
    Given any pair-force potential P
    And a ParticleState with particle_count = 2 placed so the per-pair force on particle 0 due to particle 1 is finite and nonzero
    And neighbor_counts is [96, 0] on the device
    And the first 96 slots of particle 0's neighbour list all hold the partner index 1
    When the _f variant of P is launched
    Then slot_force_x[0] equals 96 times the closed-form per-pair x-force for the (0, 1) pair, summed via the warp tree (3 full sweep iterations, no partial fourth)

  # --- Variant selection ---

  @rq-43b88a11
  Scenario: init_device exposes both kernel variants on every pair-force Kernels field
    Given a CUDA-capable GPU is available as device 0
    When init_device() is called
    Then gpu.kernels.lj.pair_force_f is a populated CudaFunction
    And gpu.kernels.lj.pair_force_fev is a populated CudaFunction
    And gpu.kernels.coulomb.coulomb_pair_force_f is a populated CudaFunction
    And gpu.kernels.coulomb.coulomb_pair_force_fev is a populated CudaFunction
    And gpu.kernels.spme_real.spme_real_pair_force_f is a populated CudaFunction
    And gpu.kernels.spme_real.spme_real_pair_force_fev is a populated CudaFunction

  @rq-b6c1c681
  Scenario: _f variant does not write the energy or virial slot output rows
    Given any pair-force potential P
    And the slot_energy and slot_virial target buffers initialised to known nonzero patterns A_energy and A_virial
    When the _f variant of P is launched
    Then slot_energy is byte-identical to A_energy
    And slot_virial is byte-identical to A_virial

  @rq-3f01c5c9
  Scenario: _fev variant writes the energy and virial slot output rows
    Given any pair-force potential P
    And the slot_energy and slot_virial target buffers initialised to known nonzero patterns A_energy and A_virial
    And at least one pair-interacting particle pair exists
    When the _fev variant of P is launched
    Then slot_energy differs from A_energy
    And slot_virial differs from A_virial

  @rq-9bb18c44
  Scenario: _f and _fev variants agree on the force output for the same inputs
    Given any pair-force potential P
    And a ParticleState of N = 16 with deterministic positions producing nonzero pair forces
    When the _f variant of P is launched against output buffer A
    And the _fev variant of P is launched against output buffer B
    Then A.slot_force_x, A.slot_force_y, A.slot_force_z are byte-identical to B.slot_force_x, B.slot_force_y, B.slot_force_z

  # --- Reproducibility ---

  @rq-49cdbd88
  Scenario: Two independent runs of any pair-force kernel produce byte-identical output
    Given any pair-force potential P
    And two independently-constructed ParticleStates with identical contents at N = 128
    And identical neighbor lists in both runs
    When the same variant of P is launched on each
    And slot_force_x, slot_force_y, slot_force_z (and slot_energy, slot_virial for the fev variant) are downloaded to the host
    Then run A and run B agree byte-for-byte on every Real value

  # --- Newton's third law ---

  @rq-bd067911
  Scenario: Per-particle forces sum to zero for an isolated system
    Given any pair-force potential P
    And a ParticleState of N particles with arbitrary positions inside the box
    And a neighbor list covering every unordered pair
    When the _f variant of P is launched
    And slot_force_x, slot_force_y, slot_force_z are downloaded and summed over all particles
    Then each component sum is within N · eps_f32 of zero

  @rq-4864c156
  Scenario: Pair (i, j) and (j, i) contributions are opposite for non-boundary displacements
    Given any pair-force potential P
    And a ParticleState of N = 2 with positions p0 = (0, 0, 0) and p1 = (1.3, 0.4, -0.2)
    And both particles appear in each other's neighbor list
    When the _f variant of P is launched
    Then slot_force_x[0] equals -slot_force_x[1] bit-exactly
    And slot_force_y[0] equals -slot_force_y[1] bit-exactly
    And slot_force_z[0] equals -slot_force_z[1] bit-exactly

  # --- Empty state ---

  @rq-68b8b92a
  Scenario: Launcher with particle_count = 0 is a no-op
    Given any pair-force potential P
    And a ParticleState with particle_count = 0
    When the launcher for either variant of P is called
    Then it returns Ok(()) without launching a kernel

  # --- Side effects ---

  @rq-5241686b
  Scenario: Kernel does not modify positions, velocities, or masses
    Given a ParticleBuffers built from a known ParticleState
    And a snapshot of positions_*, velocities_*, masses, particle_ids before launch
    When any variant of any pair-force kernel is launched
    And particle_buffers is downloaded to a host ParticleState
    Then positions_x, positions_y, positions_z, velocities_x, velocities_y, velocities_z, masses, and particle_ids are byte-identical to the snapshot

  @rq-df61d939
  Scenario: Kernel does not modify neighbor_counts or neighbor_list
    Given a NeighborListState built deterministically
    And a snapshot of neighbor_counts and neighbor_list before launch
    When any variant of any pair-force kernel is launched
    And both buffers are downloaded to the host
    Then they are byte-identical to the snapshot

  # --- Numerical edge cases ---

  @rq-2d299ffa
  Scenario: NaN per-pair contribution propagates to NaN per-particle output
    Given a pair-force potential P with parameters chosen so the per-pair functional form returns NaN for a specific pair
    And the affected pair is the only entry within the cutoff for particle 0
    When the _f variant of P is launched
    Then slot_force_x[0] is NaN
```
