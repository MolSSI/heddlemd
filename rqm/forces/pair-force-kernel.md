# Feature: Pair-Force Kernel Conventions <!-- rq-d7a35317 -->

Every short-range pair-force kernel in the engine is the
JIT-composed kernel described in
`jit-composed-pair-force.md`, whose topology, neighbour-list
contract, and force-accumulation pattern are specified by the
packed-neighbour architecture in
`packed-neighbour-pair-force.md`. There is one force-evaluation
path per fast-class pair-force pipeline, JIT-composed at
`ForceField::new` time across all active fast-class pair-force
slots.

This file specifies only the conventions shared across that one
kernel and the per-potential source fragments it ingests: the
two `AggregateLevel` variants (`_f` / `_fev`), the launch
constants (`WARPS_PER_BLOCK = 8`, `BLOCK_SIZE = 256`), and the
absence of per-potential standalone kernels.

## Kernel Variants <!-- rq-7b27c75b -->

The JIT-composed pair-force module exposes two `extern "C"`
entry points per pipeline:

- `heddle_jit_composed_pair_force_f` — writes per-particle force
  components (`fx`, `fy`, `fz`) only. Selected when the framework
  call site has `AggregateLevel::ForcesOnly`. The composed source
  emits the energy and virial accumulations as dead code that the
  CUDA compiler eliminates.
- `heddle_jit_composed_pair_force_fev` — writes per-particle
  force components plus the per-particle potential-energy share
  and the per-particle scalar-virial share. Selected when the
  framework call site has `AggregateLevel::ForcesAndScalars`.

Both variants accumulate into the same per-class fixed-point
buffers and run the same inner-loop structure; they differ only in
whether the per-pair `energy` and `virial` contributions are
incremented (see `packed-neighbour-pair-force.md` *Fixed-Point
Force Buffers*).

A second pair of entry points covers the single-pair list:

- `heddle_jit_composed_pair_force_f_single`
- `heddle_jit_composed_pair_force_fev_single`

These follow the same `_f` / `_fev` naming convention. See
`packed-neighbour-pair-force.md` *Force Kernel* for their
structure.

## Launch Configuration Constants <!-- rq-63726e30 -->

- `WARPS_PER_BLOCK = 8`, `BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK
  = 256` for the main JIT-composed kernel.
- The single-pair kernel uses `BLOCK_SIZE = 256` with one thread
  per pair (`WARPS_PER_BLOCK` is not used).
- `src/gpu/kernels.rs` exposes `PAIR_FORCE_WARPS_PER_BLOCK = 8`
  and `PAIR_FORCE_BLOCK_SIZE = 256` as `pub const u32`. The
  packed-neighbour construction pipeline reuses the same
  constants where appropriate.
- Shared memory: the main JIT-composed kernel declares per-block
  shared memory only for the per-warp staging required by the
  diagonal-shuffle inner loop (one warp's worth of intermediate
  positions and forces); see
  `packed-neighbour-pair-force.md` *Diagonal Shuffle*.
- Stream: the default stream carried by the calling potential's
  device. The SPME reciprocal pipeline owns its own stream (see
  `spme.md`); the SPME real-space slot's fragment runs inside the
  JIT-composed pair-force kernel on the default stream like every
  other fast-class pair-force slot.

## Per-Potential Standalone Kernels <!-- rq-7c59a571 -->

The engine does not carry per-potential standalone pair-force
kernels (`lj_pair_force_*`, `coulomb_pair_force_*`,
`spme_real_pair_force_*`). Each fast-class pair-force slot
contributes a CUDA source fragment to the JIT composer (see
`jit-composed-pair-force.md`); the composer produces the single
entry-point kernel that processes every active slot's
contribution per pair.

## Reproducibility <!-- rq-6e191fd0 -->

Per-particle force accumulation is bit-exact run-to-run on the
same GPU via fixed-point integer atomics (see
`packed-neighbour-pair-force.md` *Fixed-Point Force Buffers* and
*Determinism*). Integer addition is associative regardless of
operand arrival order, so the per-atom sum is invariant under
warp scheduling.

The fixed-point representation is `i64` interpreted from
`unsigned long long` with scale `2^32`. The `Real` value returned
by the finalisation kernel agrees with the f32 Kahan sum of the
underlying contributions to within a few ULPs of the fixed-point
quantisation step; this small deviation from a notional
"reference f32 sum" is explicitly permitted by
`docs/architecture.md`.

## Out of Scope <!-- rq-605f2b43 -->

- Per-potential standalone pair-force kernels. The packed-
  neighbour JIT-composed kernel is the only force-evaluation
  path; no fallback `<potential>_pair_force_*` kernel exists.
- Warp-tree butterfly reduction of per-particle accumulators.
  Forces accumulate via fixed-point atomicAdd; no warp-tree
  reduction runs over the per-particle outputs.
- Per-particle padded neighbour lists indexed by `i * max_neighbors
  + k`. The packed-neighbour list (see
  `packed-neighbour-pair-force.md`) replaces this structure.
- Bonded and angle potentials. Those slots use their own bond /
  angle index tables and per-bond / per-angle intermediates; see
  `morse-bonded.md` and `harmonic-angle.md`.
- `f64` builds of the JIT-composed kernel; see
  `packed-neighbour-pair-force.md` *Out of Scope*.

## Gherkin Scenarios <!-- rq-5c12bfee -->

```gherkin
Feature: Pair-Force Kernel Conventions

  @rq-72de7711
  Scenario: AggregateLevel::ForcesOnly selects the _f entry point
    Given the framework calls ForceField::step with
      AggregateLevel::ForcesOnly
    When the JIT-composed pair-force kernel launches
    Then the _f entry point is invoked
    And neither energy nor virial is incremented in any per-atom
      fixed-point slot

  @rq-2b605fa8
  Scenario: AggregateLevel::ForcesAndScalars selects the _fev entry point
    Given the framework calls ForceField::step with
      AggregateLevel::ForcesAndScalars
    When the JIT-composed pair-force kernel launches
    Then the _fev entry point is invoked
    And the per-atom fixed-point energy and virial slots receive
      contributions for every evaluated pair

  @rq-c74e6546
  Scenario: No per-potential standalone kernel is loaded
    Given a fast-class pair-force pipeline with Lennard-Jones,
      Coulomb, and SPME-real slots active
    When ForceField::new completes
    Then the Kernels handle does not carry an lj_pair_force_*,
      coulomb_pair_force_*, or spme_real_pair_force_* CudaFunction
    And only the JIT-composed module's entry points are present

  @rq-f34a4f43
  Scenario: Single-pair kernel observes the same _f/_fev convention
    Given AggregateLevel::ForcesOnly is in effect
    When the single-pair kernel launches
    Then the heddle_jit_composed_pair_force_f_single entry point is
      invoked
```
