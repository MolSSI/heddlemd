# Feature: Packed-Neighbour Pair-Force Architecture <!-- rq-bce26a14 -->

The fast-class pair-force pipeline partitions atoms into 32-atom
**blocks** in cell-sort order and evaluates pair interactions through
a kernel whose neighbour-list element is one **i-block paired with
32 individual j-atoms**. The 32 j-atoms in one entry are not
required to come from the same j-block; they are 32 distinct atoms
that the construction kernel has verified to be within
`r_cut + r_skin` of at least one atom in the i-block. The force
kernel's inner loop runs 32 iterations with a diagonal shuffle so
each of the 32 i-atoms is paired with each of the 32 j-atoms — all
1024 inner iterations evaluate a real pair (the construction has
already filtered candidates).

Forces, potential energies, and virials accumulate via
`atomicAdd` on a per-particle **64-bit fixed-point** buffer.
Integer addition is associative regardless of arrival order, so the
per-particle sum is bit-exact across runs without an ordered
reduction stage. A single conversion kernel translates the
fixed-point sum to `Real` at the end of each step's force
evaluation.

Pairs that include any exclusion go through a separate, smaller
list of (i-block, j-block) **exclusion tiles** processed with the
same diagonal shuffle and explicit per-pair exclusion-scale lookup.
The neighbour list itself never contains entries that involve
excluded pairs.

This file specifies the data model, the block layout, the
neighbour-list construction pipeline, the force kernel, the
fixed-point force buffers, the JIT composer integration, the
determinism invariants, the configuration surface, and the Feature
API. Adjacent files: `rqm/forces/neighbor-list.md` (the cell-list
spatial pre-step that supplies `sorted_particle_ids`),
`rqm/forces/jit-composed-pair-force.md` (the JIT composer's
per-slot source-fragment contract, with the outer-loop template
specified here), `rqm/forces/framework.md` (the slot framework and
class output buffers — fast-class pair-force slots use the
fixed-point class accumulator described below), `architecture.md`
(the GPU-vs-GPU bit-exact reproducibility invariant, preserved here
by integer associativity).

This architecture applies to fast-class pair-force slots
(`lennard_jones`, `coulomb`, `spme_real`, and any user-registered
fast-class pair-force slot). It does not apply to bonded or angle
slots (which retain their per-bond / per-angle scratch-buffer
architecture), the SPME reciprocal pipeline (unaffected), or any
slow-class slot.

## Scope <!-- rq-1bab5b76 -->

The packed-neighbour architecture is the only force-evaluation
path for fast-class pair-force slots. The JIT-composed pair-force
kernel (`heddle_jit_composed_pair_force_f` /
`heddle_jit_composed_pair_force_fev`) is its entry point and is
launched once per step per `ForceField::step` /
`step_class(Fast, …)` call when at least one fast-class pair-force
slot is active.

`max_neighbors` is not a configuration field. There is no
per-particle padded neighbour list. The neighbour-list buffers are
sized to the actual number of packed entries plus a growth margin
(see *Capacity* below).

## Atom Blocks <!-- rq-0206fbeb -->

Atoms are partitioned into blocks of 32 in the post-cell-sort
ordering supplied by `sorted_particle_ids` (see
`rqm/forces/neighbor-list.md` for the cell-list construction). The
number of blocks is `n_blocks = ⌈N / 32⌉`.

- **Block-index `b` contains 32 sort-positions** `b·32, b·32+1, …,
  b·32+31`. The atom at block position `(b, ℓ)` (block `b`, lane
  `ℓ`) has the original atom ID `sorted_particle_ids[b·32 + ℓ]`.
- For `N` not divisible by 32, the last block has `N mod 32` real
  atoms and `32 − (N mod 32)` inactive lanes. The padding atoms
  are treated as positioned at infinity so they fail every cutoff
  test; the kernel additionally gates per-atom writes through
  `tile_atom_count[b]` (described below).

Block membership is stable across pair-force kernel launches and
changes only at neighbour-list rebuild time.

Per-block metadata held on `NeighborListState`:

- `tile_atom_count: CudaSlice<u32>` of length `n_blocks` — number of
  real atoms in each block. `32` for every block except possibly
  the last.
- `tile_lane_mask: CudaSlice<u32>` of length `n_blocks` — active-lane
  bitmask `(1u << tile_atom_count[b]) − 1`. `0xFFFFFFFF` for full
  blocks; the partial last block (if any) has zeros in the high
  bits.

The mapping from block position to original atom ID is
`sorted_particle_ids[b · 32 + ℓ]`; no separate block-to-particle
table is allocated.

## Tile-Sorted Position View <!-- rq-512037ca -->

`NeighborListState` carries a per-step-refreshed position view
indexed by block position:

- `tile_sorted_positions_x: CudaSlice<Real>` of length `N` (or
  `1` when `N = 0`).
- `tile_sorted_positions_y: CudaSlice<Real>` similarly.
- `tile_sorted_positions_z: CudaSlice<Real>` similarly.

The semantics are
`tile_sorted_positions_*[k] = positions_*[sorted_particle_ids[k]]`
for every `k` in `[0, N)`. The kernel
`scatter_positions_to_tile_order` writes these buffers once at
the start of every `ForceField::step()` from the current
`ParticleBuffers.positions_*` arrays. Inactive lanes in the
partial last block receive `+∞` for all three coordinates so
they trivially fail every cutoff test in the force kernel and
the construction kernel.

The i-side position loads in the force kernel and the i-side and
j-side position loads in the construction kernel read from this
view (coalesced 32-element spans per warp). The j-side position
loads in the force kernel read `positions_*[j_atom_id]` directly,
where `j_atom_id` is the original atom ID stored in
`interacting_atoms` (see below); these reads are uncoalesced but
benefit from L1 cache locality when packed j-atoms are spatially
clustered.

## Block Bounding Boxes <!-- rq-bd3f4707 -->

Per-block axis-aligned bounding boxes are computed at rebuild
time:

- `block_centre: CudaSlice<Real4>` of length `n_blocks` — block
  centre `(x, y, z)` plus the maximum atom-to-centre distance in
  `w`. The `w` term is used to tighten the centre-to-centre
  bounding-sphere prune.
- `block_bbox: CudaSlice<half3>` of length `n_blocks` — bounding-box
  half-extents `(dx, dy, dz)`, stored as half-precision floats so
  the inner pre-filter loop fits more entries in cache. Half-up
  rounding is used so the stored extents never undercount and
  interactions are never missed.

For the partial last block, padding atoms (block positions ≥
`tile_atom_count[b]`) are not allowed to widen the bbox; they
contribute `±∞` sentinel values during the reduction so they are
ignored by the `min`/`max` reduction.

## Sorted-Blocks-by-Volume <!-- rq-4f6fbfcb -->

Blocks are reordered for the pre-filter sweep so the small (dense)
blocks are visited before the large (sparse) ones. The sort key is
`log(dx + dy + dz)` quantised into 20 bins; the sorted array is
held as `sorted_blocks: CudaSlice<u32>` of length `n_blocks`, with
each entry packing `(bin << 22) | block_index` so the radix sort
sees a 32-bit key.

`sorted_blocks` is rebuilt at every neighbour-list rebuild. The
construction-kernel sweep iterates `sorted_blocks` rather than the
identity block ordering.

A second pair of arrays carries bbox data in the sorted order so
the pre-filter avoids one indirection per access:

- `sorted_block_centre: CudaSlice<Real4>` — same shape as
  `block_centre`, permuted by `sorted_blocks`.
- `sorted_block_bbox: CudaSlice<half3>` — same shape as
  `block_bbox`, permuted by `sorted_blocks`.

## Neighbour List <!-- rq-21131a60 -->

The packed neighbour list is a flat array of entries. Each entry
binds one i-block to 32 j-atoms:

- `interacting_tiles: CudaSlice<u32>` of length
  `interacting_tiles_capacity` — `interacting_tiles[pos]` is the
  i-block index for entry `pos`.
- `interacting_atoms: CudaSlice<u32>` of length
  `interacting_tiles_capacity · 32` — `interacting_atoms[pos·32 + ℓ]`
  is one **original atom ID** (not a block position) of a j-atom
  that interacts with the i-block `interacting_tiles[pos]`. The 32
  j-atoms in one entry may come from any combination of j-blocks;
  the only guarantee is that every j-atom has passed the per-atom
  cutoff test against at least one i-atom in
  `interacting_tiles[pos]`. No padding, no mask.

Padding values are reserved: a j-atom slot may carry `N` (i.e., one
past the largest valid atom ID) to indicate "no atom" in the tail
of the final partial entry. The kernel skips slots with value
`>= N`.

When the population of interacting atoms for an i-block X is less
than a threshold (`MAX_BITS_FOR_PAIRS`, default 4 j-atoms across
the same y-block), the surviving atoms are written to a
**single-pair list** instead of `interacting_atoms`:

- `single_pairs: CudaSlice<Int2>` of length `single_pairs_capacity`
  — `single_pairs[k]` is `(atom_i, atom_j)`, two **original atom
  IDs**, exactly one pair per entry. This avoids spending a 32-slot
  packed entry on a sparse contribution.

The interaction counts are held on a small device counter array:

- `interaction_count: CudaSlice<u32>` of length 2 —
  `interaction_count[0]` is the live entry count of
  `interacting_tiles` / `interacting_atoms`;
  `interaction_count[1]` is the live entry count of `single_pairs`.

`interaction_count` is reset to `(0, 0)` at the start of every
rebuild.

## Exclusion Tiles <!-- rq-03faaf24 -->

Exclusion handling is split off from the cutoff neighbour list:

- `exclusion_tiles: CudaSlice<Int2>` of length `n_exclusion_tiles`
  — `(x_block, y_block)` pairs. Every pair-of-blocks whose 32×32
  cross-product contains at least one excluded pair-of-atoms (from
  topology) is enumerated in this list **once**, at
  `ForceField::new` time. The list is sorted by `(x_block,
  y_block)` lex and is constant across the simulation (it changes
  only if the exclusion topology changes).
- `exclusion_tiles_count: u32` — number of live entries; equals
  `exclusion_tiles.len()`.

The construction kernel for the cutoff neighbour list explicitly
**skips** any candidate j-atom whose original atom ID appears in
the i-block's exclusion table. Cutoff neighbour-list entries
therefore never contain excluded pairs; the force kernel can
process them without any per-pair exclusion check.

The exclusion-tile table is built once at `ForceField::new` and
held immutable. Its size grows linearly with the topology
exclusion count, not with `N`.

Per-pair exclusion scaling for pairs visited inside an exclusion
tile is the responsibility of each pair-force fragment, through
its `exclusion_scale(i, j)` functor method (see
`jit-composed-pair-force.md`'s *Source-Fragment Contract*). Each
fragment carries its own per-atom `excl_offsets` / `excl_partners`
/ `excl_scales` arrays (built from the topology's `ExclusionList`)
and looks up the scale per pair by atom ID, not by lane index.
This preserves per-fragment scales — for example, an OPLS-style
1-4 exclusion where the Lennard-Jones contribution scales by
`0.5` while the Coulomb contribution scales by `0.833` — without
the packed-neighbour data model holding any centralised scale
table. Fragments that have no per-pair scaling (`exclusion_scale`
always returns `1.0`) pay only the call cost in Loop 1; in Loop 2
they pay nothing.

## Construction Pipeline <!-- rq-dbffee81 -->

The neighbour-list rebuild runs the following kernels in sequence
on the device's default stream, after the cell-list pre-step has
produced `sorted_particle_ids`:

1. **`scatter_positions_to_tile_order`** — refresh
   `tile_sorted_positions_*` from `positions_*` via
   `sorted_particle_ids`. One thread per atom; block size 256.
   (This kernel also runs every step, not only on rebuild — see
   *Per-Step Pipeline* below.)
2. **`compute_block_bbox`** — one warp per block; each warp's 32
   lanes load 32 atom positions from `tile_sorted_positions_*` and
   reduce min/max via `__shfl_xor_sync`. Writes `block_centre`,
   `block_bbox`, and the per-block bounding-sphere radius
   `block_centre[b].w`. Inactive lanes contribute `±∞` so they do
   not widen the box.
3. **`compute_sort_keys`** — one thread per block; reads
   `block_bbox`, computes the volume-bin key `(bin << 22) |
   block_index`, writes `sorted_blocks`. The bin range is computed
   from the global min/max of `block_bbox` sums via a single-block
   reduction.
4. **`sort_blocks_by_volume`** — radix sort by the high bits of
   `sorted_blocks`. Stable; ties broken by block index. Either an
   inline radix sort (preferred) or a CUB-backed sort (acceptable
   if reproducible).
5. **`sort_block_data`** — one thread per block; permutes
   `block_centre` and `block_bbox` through `sorted_blocks` into
   `sorted_block_centre` and `sorted_block_bbox`. Half-precision
   conversion happens here.
6. **`find_blocks_with_interactions`** — the main construction
   kernel. One warp per i-block (iterated through
   `sorted_blocks`). For each candidate j-block (from inner
   sweep), the warp:
   - Computes the centre-to-centre bbox-prune test (`Δ² ≤
     (r_search + radius_i + radius_j)²`) using
     `sorted_block_centre` and `sorted_block_bbox`.
   - For surviving candidates, loads the j-block's 32 atoms and
     tests each j-atom against all 32 i-atoms (via warp-shuffled
     i-positions). Records per-j-atom interaction bits.
   - Excludes any j-atom that is on the i-block's exclusion list
     (those pairs go through `exclusion_tiles`, not the cutoff
     list).
   - Compacts surviving j-atoms (whose interaction bit is set)
     into a per-warp staging buffer of size `BUFFER_SIZE` (default
     256). When the buffer reaches 32, the warp flushes 32
     j-atoms into the global `interacting_tiles[pos]` /
     `interacting_atoms[pos·32 + ℓ]` arrays, advancing
     `interaction_count[0]` via `atomicAdd`.
   - If a candidate j-block contributes fewer than
     `MAX_BITS_FOR_PAIRS` surviving j-atoms (default 4), the
     warp diverts the contribution to `single_pairs`, advancing
     `interaction_count[1]` via `atomicAdd`.
   - At the end of the i-block sweep, the partial tail of the
     staging buffer is flushed; the final entry pads unused
     slots with `N` (the sentinel "no atom" value) so the force
     kernel's bounds check skips them.

The kernel observes a `force_rebuild` flag from the displacement
check (see *Per-Step Pipeline*) and returns immediately if no
rebuild is needed.

The construction kernel reads the cutoff and skin distance from a
device-side scalar (`r_search_sq`) updated only when the
simulation box generation changes (see
`rqm/forces/neighbor-list.md` for the box-generation
mechanism).

### Capacity <!-- rq-67a09135 -->

`NeighborListState` carries:

- `interacting_tiles_capacity: u32` — current allocated capacity
  of `interacting_tiles` and `interacting_atoms` (the latter is
  sized `interacting_tiles_capacity · 32`).
- `single_pairs_capacity: u32` — current allocated capacity of
  `single_pairs`.
- `tile_pair_growth_factor: f64` — multiplier applied on overflow.
  Must be greater than 1.0. Default 1.5.

The construction kernel observes both capacities and writes
overflow flags to a small status buffer when a flush would exceed
capacity. The host reads the status after the kernel completes; if
either overflowed, the host grows the respective buffer to
`required * tile_pair_growth_factor` and re-runs the construction
from step 6. Steps 1–5 are not repeated.

Initial capacities are set at `NeighborListState` construction
time from the configuration (see *Configuration* below). When a
phase begins, an explicit one-shot rebuild runs before any
CUDA-graph capture so the captured graph holds the post-grow
device pointers; subsequent rebuilds within the captured loop
must therefore not grow. Implementation must guarantee this
either by tuning the initial capacity from the configuration or
by pre-running a probe rebuild whose result is used to size the
buffers before capture begins.

## Fixed-Point Force Buffers <!-- rq-a2f419db -->

Forces, potential energies, and per-particle virials are
accumulated in 64-bit fixed-point integer buffers held on
`ForceField`:

- `fast_total_forces_fp_x: CudaSlice<u64>` of length `N`
- `fast_total_forces_fp_y: CudaSlice<u64>`
- `fast_total_forces_fp_z: CudaSlice<u64>`
- `fast_total_potential_energies_fp: CudaSlice<u64>`
- `fast_total_virials_fp: CudaSlice<u64>`

(and analogously `slow_total_*_fp` for slow-class slots that opt
into the same accumulator, e.g. SPME-real; SPME-recip continues
to write to its own `Real` buffer because it is not pair-force
shaped.)

The fixed-point scale is `2^32` (i.e., `0x100000000`):

```text
real_to_fixed(f: Real) -> i64 = (i64) (f * 2^32)
fixed_to_real(s: u64) -> Real = ((i64) s) / 2^32 as Real
```

The conversion uses signed-integer-cast-of-bit-pattern so positive
and negative `Real` values both round to the nearest int64 when
multiplied by `2^32`. The buffers are typed `u64` because CUDA's
`atomicAdd(unsigned long long*, unsigned long long)` is the
available 64-bit atomic; the values are interpreted as i64
two's-complement when reading. Two's-complement integer addition is
associative regardless of operand signedness, so the accumulator is
bit-exact across runs.

At the start of every `ForceField::step` / `step_class(Fast, …)`
call that re-evaluates the fast class, the fast fixed-point
buffers are zeroed via `cudaMemsetAsync`. (`memset` to zero is a
deterministic equivalent of writing all-zero u64 values.) Slow-
class buffers similarly when the slow class re-evaluates.

Per-pair force/energy/virial contributions are accumulated as
follows inside the inner loop of the force kernel:

```text
// per-pair contribution to atom i and atom j:
delta_fx, delta_fy, delta_fz = (factor * dx, factor * dy, factor * dz)
delta_energy = energy_share
delta_virial = virial_share

// per-lane registers accumulate over the 32 inner iterations;
// no atomic in the inner loop:
i_fx += delta_fx;   i_fy += delta_fy;   i_fz += delta_fz;
j_fx -= delta_fx;   j_fy -= delta_fy;   j_fz -= delta_fz;
i_e  += delta_energy * 0.5; j_e += delta_energy * 0.5;
i_w  += delta_virial * 0.5; j_w += delta_virial * 0.5;
```

At the end of the 32-iteration inner loop, each lane atomicAdds
its accumulated `i_*` to the i-atom's fixed-point slot and its
accumulated `j_*` to the j-atom's fixed-point slot. Newton's 3rd
law is satisfied because each pair's contribution is applied once
to each atom (via the same `delta_*` with opposite signs).

After all fast-class pair-force evaluation is complete (one
JIT-composed kernel launch per step), a finalization kernel
`finalize_fast_class_forces` converts the fixed-point buffers to
`Real` and adds them into `ParticleBuffers.forces_*`,
`ParticleBuffers.potential_energies`, and
`ParticleBuffers.virials`. The conversion uses
`fixed_to_real(s)` and the add is performed in `Real`. The
fixed-point buffer is not zeroed by the finalizer (the next step's
class-zero kernel does that).

## Force Kernel <!-- rq-6083409b -->

The JIT-composed pair-force kernel has two top-level loops, each
matching one neighbour-list source:

### Loop 1: Exclusion Tiles <!-- rq-a177c7ee -->

For every entry `t` in `exclusion_tiles`:

- `x = exclusion_tiles[t].x_block`, `y = exclusion_tiles[t].y_block`.
- Load the 32 i-atoms of `x` (positions from
  `tile_sorted_positions_*[x·32 + lane]`, original atom IDs from
  `sorted_particle_ids[x·32 + lane]`).
- Load the 32 j-atoms of `y` similarly.
- Run the 32-iteration diagonal shuffle (described in *Diagonal
  Shuffle* below). For each `(i, j)` pair, the composer invokes
  the with-exclusions variant of the per-pair evaluator (see
  `jit-composed-pair-force.md`): each active fragment's
  `evaluate` produces its raw `(factor, energy, virial)`, the
  fragment's `exclusion_scale(i, j)` produces its per-fragment
  scale, and the lane's pair accumulator adds `(factor·scale,
  energy·scale·0.5, virial·scale·0.5)` for that fragment.
  Fragments with no exclusion partners for atom `i` see
  `exclusion_scale(i, j) == 1.0` and contribute their full pair
  value; fragments where `(i, j)` is fully excluded see
  `exclusion_scale(i, j) == 0.0` and contribute zero.
- AtomicAdd per-lane `i_*` and `j_*` accumulators to the
  fixed-point buffer using i-atom and j-atom original IDs.

For the diagonal case `x == y`, the lane `i_lane == j_lane` is
the self-pair; the inner loop skips it explicitly via `if
(i_lane == j_lane) continue`.

### Loop 2: Cutoff Entries <!-- rq-a4b9e702 -->

For every entry `pos` in `[0, interaction_count[0])`:

- `x = interacting_tiles[pos]`.
- Load 32 i-atoms of `x` as above.
- Load 32 j-atoms from `interacting_atoms[pos·32 + lane]`. Each
  is an original atom ID; load its position from
  `positions_*[j_atom_id]`. If `j_atom_id >= N`, the slot is the
  partial-tail padding; treat as inactive.
- Run the 32-iteration diagonal shuffle. The composer invokes
  the without-exclusions variant of the per-pair evaluator:
  each active fragment's `evaluate` produces `(factor, energy,
  virial)` which the lane adds directly into the pair
  accumulator (no `exclusion_scale` call, no per-pair
  partner-list memory load). The construction kernel has already
  removed excluded pairs from `interacting_atoms`, so every pair
  visited here is implicitly scale `1.0`.
- AtomicAdd per-lane accumulators to the fixed-point buffer.

### Loop 3: Single Pairs <!-- rq-b28a6d96 -->

For every entry `k` in `[0, interaction_count[1])`:

- `(atom_i, atom_j) = single_pairs[k]`.
- One thread per pair; load i-position, j-position, compute the
  pair contribution.
- AtomicAdd directly to the fixed-point buffer for both atoms.

### Diagonal Shuffle <!-- rq-18847c46 -->

The 32-iteration inner loop pattern:

```text
// initial j-side state per lane: lane ℓ holds j-atom ℓ
i_lane = lane
tj = lane
for t in 0..32:
    if (x == y && tj == i_lane) {
        // self pair on a diagonal exclusion tile; skip
    } else if (j_atom_id_at(tj) < N) {
        // evaluate pair (i_atom = lane, j_atom_lane = tj):
        compute delta, factor, energy, virial
        i_fx += factor * dx;   i_fy += factor * dy;   i_fz += factor * dz
        j_fx -= factor * dx;   j_fy -= factor * dy;   j_fz -= factor * dz
        i_e  += energy * 0.5;  j_e  += energy * 0.5
        i_w  += virial * 0.5;  j_w  += virial * 0.5
    }
    // shuffle j-side state by one lane:
    j_atom_id, j_x, j_y, j_z, j_fx, j_fy, j_fz, j_e, j_w =
        __shfl_sync(0xFFFFFFFFu, ..., (lane + 1) & 31)
    tj = (tj + 1) & 31
```

After 32 iterations the j-side accumulators have rotated 32 times
and are back at their starting lane. Lane ℓ's `j_*` accumulators
hold the total contribution to the j-atom that started in lane ℓ
(i.e., the j-atom at j-lane `ℓ` of the entry).

The 32-iteration loop count is constant — no early exit. Per-pair
gating is by `j_atom_id < N` (and by the optional
`r2 <= cutoff_squared` check inside each per-slot fragment's
`evaluate`).

### Launch Configuration <!-- rq-8db4fbff -->

Loops 1 and 2 launch as a single CUDA kernel. The grid covers
both loops by total warp count, partitioned at host-side:

- `n_warps_loop1 = exclusion_tiles_count`
- `n_warps_loop2 = interaction_count[0]` (read once at launch time)
- `total_warps = n_warps_loop1 + n_warps_loop2`
- `grid_dim = ⌈total_warps / WARPS_PER_BLOCK⌉` with
  `WARPS_PER_BLOCK = 8`, `BLOCK_SIZE = 256`.

Each warp determines from its global warp index whether it
processes an exclusion tile (index `< n_warps_loop1`) or a
cutoff entry (index `≥ n_warps_loop1`).

Loop 3 launches as a separate kernel with grid `⌈n_single_pairs /
256⌉` blocks of 256 threads, one thread per pair.

## JIT Composer Integration <!-- rq-ffbee244 -->

The per-slot `PairForceFragment` source contract is documented in
`rqm/forces/jit-composed-pair-force.md`: each fragment provides
`functor_struct`, `functor_source`, `entry_point_args`,
`functor_init_source`, and a `cutoff: CutoffHandling`
declaration. The functor's interface is `cutoff_squared(i, j) ->
Real`, `evaluate(r2, inv_r, r, i, j, factor, energy, virial)`,
and `exclusion_scale(i, j) -> Real`. The composer's per-pair
evaluator has two variants:

- `heddle_jit_eval_pair_sum_excl<WriteEv>(composite, r2, inv_r,
  r, i, j, factor, energy, virial)` — sums each fragment's
  contribution multiplied by that fragment's
  `exclusion_scale(i, j)`. Used by Loop 1's diagonal-shuffle
  inner loop.
- `heddle_jit_eval_pair_sum<WriteEv>(composite, r2, inv_r, r, i,
  j, factor, energy, virial)` — sums each fragment's contribution
  with no `exclusion_scale` call. Used by Loop 2's diagonal-
  shuffle inner loop. Loop 2 never sees an excluded pair, so the
  scale is implicitly `1.0`.

Both variants apply the cutoff handling described in
`jit-composed-pair-force.md` (the per-fragment cutoff guard,
collapsed when `CutoffHandling::Uniform(c)` matches the outer
max-cutoff mask).

### Common Entry-Point Arguments <!-- rq-e8ba1aff -->

The composer emits the following common arguments for both
`heddle_jit_composed_pair_force_f` and
`heddle_jit_composed_pair_force_fev`, in this order:

```text
const Real *positions_x,
const Real *positions_y,
const Real *positions_z,
const Real *tile_sorted_positions_x,
const Real *tile_sorted_positions_y,
const Real *tile_sorted_positions_z,
const unsigned int *sorted_particle_ids,
const Int2 *exclusion_tiles,
unsigned int exclusion_tiles_count,
const unsigned int *interacting_tiles,
const unsigned int *interacting_atoms,
const unsigned int *interaction_count,
const Real *lattice,
unsigned long long *fast_force_x_fp,
unsigned long long *fast_force_y_fp,
unsigned long long *fast_force_z_fp,
unsigned long long *fast_energy_fp,
unsigned long long *fast_virial_fp,
unsigned int n,
```

The `_fev` entry point uses the same argument list (energy and
virial buffers are always passed; the `_f` entry point's emitted
inner loop simply does not increment `fast_energy_fp` /
`fast_virial_fp`).

Per-fragment arguments (per-slot parameter tables, including each
fragment's `excl_offsets` / `excl_partners` / `excl_scales`
arrays consumed by its `exclusion_scale(i, j)` method) are
appended after the common arguments in canonical slot order. The
trailing `unsigned int n` is the final argument.

### Single-Pairs Kernel <!-- rq-f119bc11 -->

A second JIT-composed entry point covers Loop 3:

```text
extern "C" __global__ void heddle_jit_composed_pair_force_f_single(...)
extern "C" __global__ void heddle_jit_composed_pair_force_fev_single(...)
```

Its common arguments add `const Int2 *single_pairs` and
`unsigned int n_single_pairs` in place of `interacting_tiles` /
`interacting_atoms` / `interaction_count`. Per-fragment arguments
are identical to the main kernel.

## Per-Step Pipeline <!-- rq-0acba2a0 -->

The fast-class pair-force pipeline runs the following sequence
every step:

| # | Stage | When |
|---|---|---|
| 1 | `scatter_positions_to_tile_order` | Every step |
| 2 | Pre-step neighbour-list check | Every step; rebuild may run if displacement exceeds `r_skin / 2` |
| 3 | `cudaMemsetAsync` zeroing fast fixed-point buffers | Every step (or only when re-evaluating the Fast class) |
| 4 | `heddle_jit_composed_pair_force_{f,fev}` | Once per step |
| 5 | `heddle_jit_composed_pair_force_{f,fev}_single` | Once per step (only if `n_single_pairs > 0`) |
| 6 | `finalize_fast_class_forces` | Once per step; converts fixed-point to Real and adds into `ParticleBuffers.forces_*` |

When step 2 triggers a rebuild, the rebuild pipeline (steps 1–6 of
*Construction Pipeline*, including the second `scatter` since
`sorted_particle_ids` has changed) runs before step 3.

## Configuration <!-- rq-9527bd2d -->

The `[neighbor_list]` section of the simulation config carries:

- `mode: "cell-list" | "all-pairs"` — unchanged from the existing
  `NeighborListConfig`.
- `r_skin: f64` (length, in active unit system) — unchanged from
  the existing `NeighborListConfig`.
- `tile_pair_growth_factor: f64` — multiplier applied to the
  `interacting_tiles` and `single_pairs` capacities on overflow.
  Must be greater than 1.0. Default 1.5.
- `interacting_tiles_initial_capacity: u32` (optional) — initial
  allocated entry count for the cutoff list. Default
  `⌈n_blocks² / 16⌉ + n_blocks` (a generous starting point that
  rarely requires growth on the first build). May be overridden
  to a smaller value for systems where the natural neighbour
  count is known.
- `single_pairs_initial_capacity: u32` (optional) — initial
  allocated entry count for the single-pair list. Default
  `⌈n_blocks · 32⌉` (one single pair per i-block on average).

The field `max_neighbors` is **not** part of `NeighborListConfig`.
Specifying it in the configuration file is a load-time error with
an explanatory message that the field is no longer used.

## Determinism <!-- rq-db98d977 -->

Two independent runs starting from byte-identical inputs produce
byte-identical outputs (forces, energies, virials). The invariants
supporting this:

1. **Deterministic cell-list pre-step.** `sorted_particle_ids` is
   a pure function of the input particle state (see
   `rqm/forces/neighbor-list.md`).
2. **Deterministic block layout and bbox.** Block membership and
   block bounding boxes are pure functions of
   `sorted_particle_ids` and the current positions.
3. **Deterministic volume sort.** The block-volume sort uses a
   stable radix sort keyed on `(volume_bin << 22) | block_index`,
   so equal-volume blocks tie-break on block index — deterministic.
4. **Deterministic construction sweep.** The construction kernel
   sweeps `sorted_blocks` in fixed index order. The per-warp
   staging buffer flushes in fixed order (the warp's lanes contribute
   in lane order via `__ballot_sync` + prefix-sum packing). The
   global `atomicAdd(&interaction_count[0], …)` assigns a
   deterministic *count*, but the per-entry write order is
   determined by warp-scheduling. **This is acceptable because the
   force kernel processes entries via the fixed-point atomic
   accumulator, whose final per-particle sum is invariant under
   permutation of entries.**
5. **Deterministic exclusion-tile enumeration.** `exclusion_tiles`
   is enumerated at `ForceField::new` time in lex order of
   `(x_block, y_block)` and is constant across the simulation.
6. **Bit-exact accumulation via integer atomics.** Per-particle
   force, energy, and virial accumulators are 64-bit unsigned
   integers interpreted as two's-complement signed integers.
   Integer addition is associative under two's-complement
   arithmetic regardless of operand order. `atomicAdd` on
   `unsigned long long` is the load-bearing primitive; its
   per-atom result is independent of how many warps wrote to that
   atom and in what order.
7. **Deterministic conversion.** The fixed-point-to-real
   conversion `(int64) sum / 2^32` is a deterministic function of
   the integer sum; the final `Real` addition into
   `ParticleBuffers.forces_*` happens in a kernel that reads each
   atom's slot exactly once.

The reproducibility scope is GPU-vs-GPU on the same hardware,
matching `architecture.md`. CPU-vs-GPU is not promised.

## Feature API <!-- rq-e1643cc5 -->

### Types <!-- rq-86f757e0 -->

- `NeighborListState` adds the fields documented under *Atom <!-- rq-b5c18e6b -->
  Blocks*, *Tile-Sorted Position View*, *Block Bounding Boxes*,
  *Sorted-Blocks-by-Volume*, and *Neighbour List* above. The
  per-particle padded `neighbor_list` and `neighbor_counts` fields
  are not present.
- `ForceField` adds the fixed-point class accumulator fields <!-- rq-e6b37d18 -->
  documented under *Fixed-Point Force Buffers*. The per-class
  `fast_total_forces_x/y/z` (and the `_potential_energies` /
  `_virials`) become `unsigned long long` fixed-point arrays; the
  per-slot `SlotOutputView` framework no longer issues 5-array
  bind targets for fast-class pair-force slots (it remains for
  bonded and angle slots, and for slow-class slots not using
  packed-neighbour).
- `NeighborListConfig` carries the fields documented under <!-- rq-3fbb8dea -->
  *Configuration*. The `max_neighbors` field is absent; the
  TOML parser reports an explicit error if `max_neighbors`
  appears in the config text.

### CUDA Kernels <!-- rq-2647cb7e -->

- `scatter_positions_to_tile_order(positions_x, positions_y, <!-- rq-89245537 -->
  positions_z, sorted_particle_ids, tile_sorted_positions_x,
  tile_sorted_positions_y, tile_sorted_positions_z, n)` — one
  thread per atom; block 256.
- `compute_block_bbox(tile_sorted_positions_x, tile_sorted_positions_y, <!-- rq-9f947525 -->
  tile_sorted_positions_z, tile_atom_count, block_centre,
  block_bbox, n_blocks)` — one warp per block.
- `compute_sort_keys(block_bbox, sorted_blocks, n_blocks)` — one <!-- rq-e9ed7617 -->
  thread per block; preceded by a single-block reduction for the
  global bbox-sum range.
- `sort_blocks_by_volume(sorted_blocks, n_blocks)` — radix sort. <!-- rq-62f00166 -->
- `sort_block_data(sorted_blocks, block_centre, block_bbox, <!-- rq-c1afd2f3 -->
  sorted_block_centre, sorted_block_bbox, n_blocks)` — one
  thread per block.
- `find_blocks_with_interactions(sorted_blocks, <!-- rq-169e1d84 -->
  sorted_block_centre, sorted_block_bbox, tile_sorted_positions_*,
  exclusion_indices, exclusion_row_indices,
  interacting_tiles, interacting_atoms, single_pairs,
  interaction_count, interacting_tiles_capacity,
  single_pairs_capacity, max_bits_for_pairs, r_search_sq,
  lattice, force_rebuild_flag, n_blocks, n_atoms,
  overflow_flag)` — main construction kernel. One warp per
  i-block, iterating through `sorted_blocks`.
- `heddle_jit_composed_pair_force_f` / <!-- rq-42e29605 -->
  `heddle_jit_composed_pair_force_fev` — JIT-composed force
  kernels; argument list documented under *JIT Composer
  Integration*.
- `heddle_jit_composed_pair_force_f_single` / <!-- rq-3ddf259b -->
  `heddle_jit_composed_pair_force_fev_single` — JIT-composed
  single-pair kernels.
- `finalize_fast_class_forces(fast_force_*_fp, fast_energy_fp, <!-- rq-9c80a966 -->
  fast_virial_fp, particle_forces_x, particle_forces_y,
  particle_forces_z, particle_potential_energies,
  particle_virials, n)` — one thread per atom; converts
  fixed-point sums to Real and `+=` into the particle buffers.
- `finalize_slow_class_forces` (analogous, when slow class uses <!-- rq-2c323eaa -->
  the fixed-point accumulator).

### Functions <!-- rq-c85fa8d1 -->

- `crate::gpu::scatter_positions_to_tile_order(kernels, buffers, <!-- rq-595e7ea4 -->
  sorted_particle_ids, tile_sorted_positions_x, tile_sorted_positions_y,
  tile_sorted_positions_z) -> Result<(), GpuError>` — launches
  the scatter kernel.
- `crate::gpu::compute_block_bbox(kernels, <!-- rq-3a31b3f0 -->
  tile_sorted_positions_x, tile_sorted_positions_y,
  tile_sorted_positions_z, tile_atom_count, block_centre,
  block_bbox, n_blocks) -> Result<(), GpuError>`.
- `crate::gpu::sort_blocks_by_volume(kernels, block_bbox, <!-- rq-ad6f0de0 -->
  sorted_blocks, sorted_block_centre, sorted_block_bbox,
  n_blocks) -> Result<(), GpuError>` — runs the three sub-steps
  (compute keys, radix sort, sort_block_data) as a single
  callable.
- `crate::gpu::find_blocks_with_interactions(kernels, …) -> <!-- rq-68a9602b -->
  Result<u32, GpuError>` — returns the post-build live count of
  `interacting_tiles`, or an overflow error indicating which
  capacity to grow.
- `crate::gpu::finalize_fast_class_forces(kernels, fp_buffers, <!-- rq-5a7f78c4 -->
  particle_buffers, n) -> Result<(), GpuError>`.
- `NeighborListState::rebuild` (existing) extends to call the <!-- rq-4896a257 -->
  construction pipeline above instead of `neighbor_list_build`.

### Helper Functions in the JIT-Composed Source <!-- rq-49f2304c -->

The composer emits the following helpers into the kernel preamble
(in addition to the existing `heddle_jit_triclinic_min_image`
and the per-slot helpers):

- `__device__ static inline long long heddle_jit_real_to_fixed(Real f) <!-- rq-8de805b0 -->
   { return (long long) (f * (Real) (1ull << 32)); }`
- `__device__ static inline Real heddle_jit_fixed_to_real(unsigned long long s) <!-- rq-d244267b -->
   { return ((Real) (long long) s) / (Real) (1ull << 32); }`
- A 32-iteration diagonal-shuffle macro `HEDDLE_JIT_SHUFFLE_PAIR_LOOP`
  that emits the inner loop including the per-lane `i_*` / `j_*`
  accumulators and the trailing `__shfl_sync` rotation.

## Out of Scope <!-- rq-ee43a3fe -->

- **`f64` compile-time builds.** The fixed-point representation
  assumes `Real == f32` so the scale `2^32` fits in `long long`
  with adequate range for typical MD force magnitudes. The `f64`
  build path returns an explicit error at `ForceField::new` if
  this architecture is in use; an `f64` extension would require a
  wider fixed-point type (e.g., split-int128 atomics or a
  software-emulated 128-bit accumulator) and is deferred.
- **The `USE_LARGE_BLOCKS` optimisation.** Optionally pre-filters
  with the bbox of 32 consecutive blocks ("large blocks") before
  the per-block sweep. The optimisation is omitted from this
  architecture; if the per-block sweep becomes a measurable
  bottleneck for very large `N` it can be added without changing
  the data model.
- **Position reordering of `posq` into block order.** Consider
  permuting `posq` into block order so the force kernel's
  `posq[j]` reads are sequential within the warp. This
  architecture keeps `positions_*` in original atom-ID order and
  pays a per-pair indirect load for j-positions. The optimisation
  is deferred; it requires moving every per-atom parameter table
  (charges, type indices, exclusion offsets, etc.) into block
  order at every rebuild.
- **The `singlePeriodicCopy` fast-path.** Consider checking
  whether the box is large enough relative to the cutoff that all
  positions can be wrapped into one periodic image, allowing the
  inner loop to skip the per-pair minimum-image wrap. The
  optimisation is omitted from this architecture; the existing
  triclinic min-image helper runs on every pair.
- **Mixed-precision energy/virial accumulation.** Energy and
  virial use the same `2^32` fixed-point scale as forces. The
  shared scale is adequate for typical MD energy ranges but may
  saturate for unusually large systems; a separate finer scale
  for energies is deferred.

## Gherkin Scenarios <!-- rq-9333127d -->

```gherkin
Feature: Packed-Neighbour Pair-Force Architecture

  Background:
    Given a fast-class pair-force pipeline with at least one
      Lennard-Jones, Coulomb, or SPME-real slot active
    And N particles laid out in cell-list order

  # --- Block layout ---

  @rq-c5e23ba8
  Scenario: Block count derives from ceil(N / 32)
    Given N = 100
    When NeighborListState::rebuild completes
    Then n_blocks equals 4
    And tile_atom_count holds [32, 32, 32, 4]
    And tile_lane_mask holds [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0000000F]

  @rq-48a014ed
  Scenario: Full block carries lane mask 0xFFFFFFFF
    Given a block b with tile_atom_count[b] = 32
    When NeighborListState::rebuild completes
    Then tile_lane_mask[b] equals 0xFFFFFFFF

  @rq-5631e383
  Scenario: Partial last block carries truncated lane mask
    Given N = 100 so block 3 has 4 real atoms
    When NeighborListState::rebuild completes
    Then tile_lane_mask[3] equals 0x0000000F
    And the high 28 bits of tile_lane_mask[3] are zero

  # --- Tile-sorted position view ---

  @rq-d9197bb9
  Scenario: tile_sorted_positions reflects current positions after the scatter
    Given a simulation step that has just completed integration
    When scatter_positions_to_tile_order runs at the start of
      ForceField::step
    Then tile_sorted_positions_x[k] equals
      positions_x[sorted_particle_ids[k]]
      for every k in [0, N)
    And the same equality holds for y and z

  @rq-146221cb
  Scenario: Scatter runs every step regardless of rebuild
    Given a simulation that runs 10 steps with the displacement-driven
      rebuild firing every 5 steps
    When the simulation completes
    Then KernelStage::SCATTER_POSITIONS_TO_TILE_ORDER records exactly
      10 launches

  @rq-ede1fc4c
  Scenario: Partial-block inactive lanes carry +infinity
    Given N = 100 so block 3 has 4 real atoms
    When scatter_positions_to_tile_order completes
    Then tile_sorted_positions_x[100..128] hold positive infinity
    And tile_sorted_positions_y[100..128] hold positive infinity
    And tile_sorted_positions_z[100..128] hold positive infinity

  # --- Block bounding boxes ---

  @rq-2f77077b
  Scenario: Full block bbox spans all 32 atoms
    Given block b is full
    When compute_block_bbox runs
    Then block_centre[b].x equals the mean of the 32 atoms' x positions
    And block_bbox[b].dx covers the maximum atom-from-centre displacement in x

  @rq-6ca8a9c4
  Scenario: Partial block bbox ignores inactive lanes
    Given block 3 has 4 real atoms at positions p0..p3
    When compute_block_bbox runs
    Then block_centre[3] reflects only p0..p3
    And no inactive lane widens block_bbox[3]

  # --- Volume sort ---

  @rq-7fcd6e99
  Scenario: sorted_blocks is sorted by volume bin ascending
    When sort_blocks_by_volume completes
    Then for every consecutive pair sorted_blocks[i], sorted_blocks[i+1],
      the high 10 bits of sorted_blocks[i] are <= the high 10 bits of
      sorted_blocks[i+1]

  @rq-e6de7839
  Scenario: Equal-volume blocks tie-break on block index
    Given two blocks b1 < b2 with identical volume bins
    When sort_blocks_by_volume completes
    Then b1 appears before b2 in sorted_blocks

  # --- Neighbour-list construction ---

  @rq-1470ef9f
  Scenario: interacting_atoms entries contain only real neighbours
    Given the construction kernel has completed
    When for every entry pos in [0, interaction_count[0])
      and every lane L in [0, 32)
    Then either interacting_atoms[pos*32 + L] >= N (the no-atom sentinel)
    Or interacting_atoms[pos*32 + L] is an atom whose position is within
      r_cut + r_skin of at least one atom in
      sorted_particle_ids[interacting_tiles[pos] * 32 .. (interacting_tiles[pos]+1) * 32]
      under minimum image

  @rq-351b2539
  Scenario: Excluded pairs never appear in interacting_atoms
    Given atoms i, j with an exclusion entry in the topology
    When NeighborListState::rebuild completes
    Then there is no entry pos and lane L such that
      interacting_atoms[pos*32 + L] == j AND
      i is in sorted_particle_ids[interacting_tiles[pos] * 32 .. (...)+32]

  @rq-e8667000
  Scenario: Sparse contributions go to single_pairs
    Given a candidate j-block that contributes 3 interacting atoms
      with the same i-block (max_bits_for_pairs default 4)
    When the construction kernel processes this candidate
    Then 3 entries are written to single_pairs (one per surviving atom)
    And no entry is written to interacting_atoms for this candidate

  @rq-7646dd13
  Scenario: Construction returns overflow when capacity exceeded
    Given interacting_tiles_capacity = 100
    And a system that produces 150 packed entries
    When find_blocks_with_interactions runs
    Then the overflow_flag indicates "interacting_tiles overflow"
    And NeighborListState::rebuild reallocates the buffer to >= 150 * 1.5
    And the construction is re-run from find_blocks_with_interactions

  @rq-8253d3c4
  Scenario: interaction_count is reset at the start of every rebuild
    Given a prior rebuild left interaction_count = [50, 12]
    When NeighborListState::rebuild begins a new rebuild
    Then interaction_count is reset to [0, 0] before
      find_blocks_with_interactions launches

  # --- Force kernel ---

  @rq-00d11a87
  Scenario: Force kernel processes both exclusion tiles and cutoff entries
    Given exclusion_tiles_count = 10 and interaction_count[0] = 100
    When the JIT-composed pair-force kernel launches
    Then 10 warps process exclusion tiles
    And 100 warps process cutoff entries

  @rq-a6ecc598
  Scenario: Cutoff entries skip the no-atom sentinel
    Given an entry pos with interacting_atoms[pos*32 + lane] == N for some
      lane
    When the force kernel processes entry pos
    Then no pair contribution is accumulated for that lane

  @rq-0606e887
  Scenario: Self-pair on diagonal exclusion tile is skipped
    Given an exclusion tile with x_block == y_block
    When the force kernel processes the tile
    Then no pair contribution is accumulated for the iteration where
      i_lane == j_lane

  @rq-80c6a964
  Scenario: Fully-excluded pair contributes zero
    Given a pair (i, j) inside an exclusion tile where every active
      fragment's exclusion_scale(i, j) returns 0.0 (full exclusion)
    When the force kernel processes the tile
    Then no contribution from (i, j) is accumulated to either atom's
      fixed-point slot

  @rq-8840662f
  Scenario: Per-fragment exclusion scale multiplies that fragment's contribution
    Given a pair (i, j) inside an exclusion tile where the LJ fragment's
      exclusion_scale(i, j) returns 0.5 and the Coulomb fragment's
      exclusion_scale(i, j) returns 0.833
    When the force kernel processes the tile
    Then the LJ contribution to atoms i and j is exactly half of an
      otherwise-identical unexcluded LJ pair
    And the Coulomb contribution to atoms i and j is 0.833 times an
      otherwise-identical unexcluded Coulomb pair

  @rq-a7c08202
  Scenario: Loop 2 never calls exclusion_scale
    Given a ForceField with at least one fast-class pair-force fragment
    And the JIT-composed kernel source captured for inspection
    Then the Loop 2 inner-loop body (the cutoff-entries path) contains
      no call to `composite.<any>.exclusion_scale`
    And the Loop 1 inner-loop body (the exclusion-tiles path) calls
      every active fragment's `exclusion_scale` exactly once per pair

  # --- Single-pair kernel ---

  @rq-86cc8e5d
  Scenario: Single-pair kernel runs only when single_pair count is nonzero
    Given interaction_count[1] == 0
    When the force-evaluation pipeline runs
    Then heddle_jit_composed_pair_force_f_single is not launched

  @rq-8d86bfa6
  Scenario: Single-pair kernel accumulates to both atoms
    Given single_pairs[k] = (i, j)
    When the single-pair kernel processes k
    Then the contribution is atomicAdded to the fixed-point slot of i
    And the opposite-sign contribution is atomicAdded to the fixed-point slot of j

  # --- Fixed-point accumulation ---

  @rq-5b560577
  Scenario: Fixed-point buffers are zeroed at the start of every force evaluation
    Given a prior step left fast_total_forces_fp_x with arbitrary values
    When ForceField::step(Fast) begins re-evaluation
    Then fast_total_forces_fp_x is cuda-memset to zero before the
      composed pair-force kernel launches

  @rq-578a4020
  Scenario: Conversion preserves f32 precision in the typical range
    Given a fixed-point sum produced by accumulating 1000 random f32
      values in range [-100, 100]
    When fixed_to_real is applied
    Then the result equals the corresponding f32 Kahan sum to within
      f32 round-off

  @rq-b2894e9e
  Scenario: Saturation triggers an error if force exceeds the fixed-point range
    Given a Real value f such that f * 2^32 exceeds i64::MAX
    When real_to_fixed is invoked
    Then the conversion saturates and the finalization kernel sets an
      overflow flag

  @rq-40b8dfb2
  Scenario: Newton's 3rd is observed
    Given a pair (i, j) evaluated in the force kernel
    When the kernel completes
    Then the contribution to atom i is exactly the negation of the
      contribution to atom j (per coordinate)

  # --- Reproducibility ---

  @rq-f9e0aa11
  Scenario: Two GPU runs produce byte-identical packed neighbour lists
    Given two simulations started from byte-identical initial state
    When both run a single neighbour-list rebuild
    Then interaction_count is byte-identical across the two runs
    And the sorted contents of (interacting_tiles, interacting_atoms)
      are byte-identical across the two runs
    And the sorted contents of single_pairs are byte-identical across
      the two runs

  @rq-5aee0b50
  Scenario: Two GPU runs produce byte-identical per-particle forces
    Given two simulations started from byte-identical initial state
    When both run ForceField::step(Fast) once
    Then ParticleBuffers.forces_x, forces_y, forces_z compare
      byte-identical across the two runs after finalize_fast_class_forces

  @rq-0fbfc03f
  Scenario: Force result is invariant under construction-write-order
    Given a system where two runs produce the same set of packed entries
      but in different per-position write order
    When ForceField::step(Fast) runs in each
    Then the per-particle force sums are byte-identical (because integer
      atomicAdd is associative)

  # --- Configuration ---

  @rq-111eb26d
  Scenario: max_neighbors in config is rejected at load
    Given a config file with [neighbor_list].max_neighbors = 1024
    When the simulation loads the config
    Then a configuration error is reported indicating max_neighbors is
      no longer a supported field

  @rq-38096442
  Scenario: tile_pair_growth_factor below 1.0 is rejected at load
    Given a config file with [neighbor_list].tile_pair_growth_factor = 0.9
    When the simulation loads the config
    Then a configuration error is reported indicating the growth factor
      must be greater than 1.0

  @rq-043c6862
  Scenario: Default initial capacity sized to avoid first-build growth
    Given a SPC water benchmark with 24,576 atoms
    When NeighborListState::new_cell_list constructs with default config
    Then interacting_tiles_capacity is at least the count produced by the
      first rebuild

  # --- Determinism of exclusion tiles ---

  @rq-c123c82c
  Scenario: exclusion_tiles is enumerated in lex order of (x_block, y_block)
    Given the topology contains exclusions
    When ForceField::new constructs the exclusion-tile table
    Then exclusion_tiles is sorted by (x_block, y_block) ascending

  @rq-30a85bc9
  Scenario: exclusion_tiles is constant across the simulation
    Given the topology does not change during the simulation
    When 1000 steps run
    Then exclusion_tiles is not modified after ForceField::new

  # --- Out of scope ---

  @rq-92b63da0
  Scenario: f64 build refuses to instantiate the packed-neighbour kernel
    Given the project is compiled with feature "f64"
    When ForceField::new is called for a fast-class pair-force pipeline
    Then construction returns an error indicating that the packed-
      neighbour kernel does not yet support f64
```
