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

The pipeline has three pair-evaluation passes, each contributing to
the same fixed-point accumulators:

- The **packed-neighbour pass** (the main pair-force kernel) walks
  every entry of `interacting_tiles` / `interacting_atoms` and
  evaluates each pair at full strength. It contains no modified pairs
  (the construction routes them elsewhere), so it does no exclusion
  work at all.
- The **single-pair pass** walks `single_pairs` — individual
  (atom_i, atom_j) pairs extracted at neighbour-list build time from
  sparse (i-block, j-block) candidates. One thread per pair; also
  free of modified pairs by construction.
- The **exclusion-tile pass** walks the small set of block pairs that
  contain modified pairs. Each such tile carries a 1024-bit
  modified-pair bitmask; the pass computes the full 32×32 block-pair
  interaction and, for each flagged lane pair, multiplies each
  fragment's contribution by that fragment's per-pair scale.

Exclusion scales are **per fragment**: each fragment's contribution to a
pair is scaled by a value in `[0, 1]` (`0` a full exclusion, `1` full
strength, intermediate for OPLS/AMBER 1-4). Scaling is applied only in
the exclusion-tile pass, through the per-tile bitmask plus a per-pair
scale lookup confined to the flagged pairs; the bulk and single-pair
passes never read the exclusion tables. See *Exclusion Handling*.

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

## Tile-Sorted Posq View <!-- rq-512037ca -->

`NeighborListState` carries a per-step-refreshed posq view
indexed by block position:

- `tile_sorted_posq: CudaSlice<Real4>` of length `N` (or `1`
  when `N = 0`).

The semantics are
`tile_sorted_posq[k] = posq[sorted_particle_ids[k]]` for every
`k` in `[0, N)`, so `tile_sorted_posq[k].xyz` is the wrapped
position of the atom at block position `k` and
`tile_sorted_posq[k].w` is that atom's charge. The kernel
`scatter_positions_to_tile_order` writes this buffer once at the
start of every `ForceField::step()` from the current
`ParticleBuffers.posq` array. Inactive lanes in the partial last
block receive `Real4 { x: +∞, y: +∞, z: +∞, w: 0 }` so they
trivially fail every cutoff test in the force kernel and the
construction kernel.

The i-side posq loads in the force kernel and the i-side and
j-side posq loads in the construction kernel read from this view
(one 16-byte coalesced load per atom replaces what would
otherwise be four scalar loads). The j-side posq loads in the
force kernel read `posq[j_atom_id]` directly, where `j_atom_id`
is the original atom ID stored in `interacting_atoms` (see
below); these reads are uncoalesced but benefit from L1 cache
locality when packed j-atoms are spatially clustered, and they
still carry the j-atom's charge through `.w` so the fragment's
`evaluate(…, qi, qj, …)` does not need a separate charges load.

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

When a candidate (i-block, j-block) pair produces no more than
`MAX_BITS_FOR_PAIRS` surviving j-atom hits, the construction
kernel writes the individual `(i_atom, j_atom)` pairs to a
**single-pair list** instead of merging the j-atoms into the
packed-neighbour buffer:

- `single_pair_atoms: CudaSlice<u32>` of length
  `2 · single_pairs_capacity` — interleaved
  `[i_atom_0, j_atom_0, i_atom_1, j_atom_1, …]` original atom IDs.
  Adjacent pairs `(single_pair_atoms[2k], single_pair_atoms[2k+1])`
  are the two ends of one extracted pair. The construction kernel
  emits one entry per surviving `(i_lane, j_lane)` pair within a
  sparse (i-block, j-block) candidate; a single sparse candidate
  with three j-atom hits, each hitting two i-atoms, produces six
  entries.

`MAX_BITS_FOR_PAIRS` is a compile-time `#define` in
`kernels/neighbor.cu` set to `3`. Below this threshold the
single-pair pass amortises a full 32×32 = 1024-pair tile loop over
just a handful of true interactions; above it the packed-
neighbour pass is cheaper.

The bulk list never contains a modified pair; modified pairs are
carried by a parallel **exclusion-tile** list (see *Exclusion
Handling*):

- `exclusion_tile_iblocks: CudaSlice<u32>` and
  `exclusion_tile_jblocks: CudaSlice<u32>`, each of length
  `exclusion_tiles_capacity` — the i-block and j-block (`bi ≤ bj`) of
  each exclusion tile.
- `exclusion_tile_masks: CudaSlice<u32>` of length
  `exclusion_tiles_capacity · 32` — `exclusion_tile_masks[t·32 + ℓ]` is
  the 32-bit row of tile `t`'s 1024-bit bitmask for i-lane `ℓ`; bit
  `j_lane` is set when the two atoms are a modified pair (some fragment
  scale differs from `1`), and the force pass reads that pair's
  per-fragment scale at evaluation time.

The interaction counts are held on a small device counter array:

- `interaction_count: CudaSlice<u32>` of length 3 —
  `interaction_count[0]` is the live entry count of
  `interacting_tiles` / `interacting_atoms`;
  `interaction_count[1]` is the live entry count of
  `single_pair_atoms`; `interaction_count[2]` is the live count of
  exclusion tiles.

`interaction_count` is reset to `(0, 0, 0)` at the start of every
rebuild.

## Exclusion Handling <!-- rq-03faaf24 -->

Nonbonded interactions between certain intramolecular atom pairs are
**scaled per fragment**. Every exclusion topology entry carries a scale
in `[0, 1]` for each fast-class fragment independently (a Lennard-Jones
scale and a Coulomb scale): `0` removes that fragment's contribution
entirely (a full exclusion, as for 1-2 bonded and 1-3 angle pairs), `1`
leaves it at full strength, and an intermediate value scales it (as for
OPLS/AMBER 1-4 pairs — typically a Lennard-Jones scale of `0.5` and a
Coulomb scale of `0.833`). The two fragment scales are independent, so a
pair may, for example, drop its Lennard-Jones contribution (scale `0`)
while keeping its Coulomb contribution (scale `1`).

A **modified pair** is an atom pair whose scale differs from `1` in at
least one fragment. A pair whose every fragment scale is `1` is not
modified and carries no exclusion entry. Modified pairs are handled
entirely by a dedicated exclusion-tile pass; the bulk neighbour-list and
single-pair passes apply no per-pair scale and read no exclusion data.
Because every modified pair is short-range (intramolecular), modified
pairs concentrate in a small set of block pairs, materialised each
rebuild as the **exclusion tiles**:

- An **exclusion tile** is an ordered block pair `(bi, bj)` with
  `bi ≤ bj` such that at least one atom of block `bi` is a modified pair
  with at least one atom of block `bj` under the current block
  membership (`sorted_particle_ids`). It carries a 1024-bit modified-pair
  bitmask (32 `u32` words, one per i-lane): bit `(i_lane, j_lane)` is set
  when the atom at slot `bi·32 + i_lane` is a modified pair with the atom
  at slot `bj·32 + j_lane`.
- The set of exclusion tiles and their bitmasks is a deterministic
  function of the exclusion topology and the current
  `sorted_particle_ids`, recomputed at every neighbour-list rebuild
  (block membership changes with the cell sort). Building it is
  amortised over the many steps between rebuilds.

The force kernel is split so that only the exclusion-tile pass ever
consults exclusion data:

- The **exclusion-tile pass** walks the exclusion tiles. It loads the
  full 32 atoms of `bi` and the full 32 atoms of `bj` from
  `tile_sorted_posq` (two coalesced loads) and runs the 32-iteration
  diagonal shuffle over all 32×32 lane pairs. For a lane pair whose
  modified-pair bit is set, it looks up each fragment's per-pair scale
  (`exclusion_scale(i, j)`) and multiplies that fragment's contribution
  by it — a full exclusion's `0` contributes nothing, a 1-4 pair's
  fraction contributes a scaled force. A lane pair whose bit is clear
  (and is not the self-block diagonal) is evaluated at full strength with
  **no scale lookup**. The scale lookup is thus confined to the flagged
  pairs of the (few) exclusion tiles.
- The **bulk neighbour-list pass** and the **single-pair pass** do
  **zero** exclusion work: they carry no `exclusion_scale` call, no
  `atom_excl_*` reads, and no per-pair scale multiply. The construction
  guarantees they never contain a modified pair (below).

The neighbour-list construction keeps modified pairs out of the bulk
list by **skipping exclusion-tile block pairs**: when the construction
sweeps candidate j-blocks for i-block `bi`, any candidate `bj` for which
`(bi, bj)` is an exclusion tile is skipped entirely — all of that block
pair's interactions (modified and unmodified alike) are computed by the
exclusion-tile pass instead. Because a modified pair's two atoms always
share such a block pair, no modified pair can reach the bulk or
single-pair outputs, and no unmodified pair is dropped or double-counted
(each block pair is processed by exactly one pass).

This confines the per-pair, per-fragment exclusion scan to the small set
of flagged pairs inside the exclusion tiles; the bulk pass — the vast
majority of pair evaluations — carries no exclusion cost at all. The
exclusion-topology index tables `atom_excl_offsets` / `atom_excl_partners`
(see `topology.md`) are read by the per-rebuild on-device tile build to
enumerate the modified pairs and place their bits; the per-fragment scale
arrays are read only by the exclusion-tile force pass, at evaluation time.
The bulk and single-pair passes read neither.

## Construction Pipeline <!-- rq-dbffee81 -->

The neighbour-list rebuild runs the following kernels in sequence
on the device's default stream, after the cell-list pre-step has
produced `sorted_particle_ids`:

1. **`scatter_positions_to_tile_order`** — refresh
   `tile_sorted_posq` from `posq` via `sorted_particle_ids`. One
   thread per atom; block size 256. Each thread writes one
   `Real4` value `posq[sorted_particle_ids[k]]` to
   `tile_sorted_posq[k]`, carrying both the position and the
   charge through in a single coalesced store. (This kernel also
   runs every step, not only on rebuild — see *Per-Step Pipeline*
   below.)
2. **`compute_block_bbox`** — one warp per block; each warp's 32
   lanes load 32 atom positions from `tile_sorted_posq` (reading
   only the `.x/.y/.z` components) and reduce min/max via
   `__shfl_xor_sync`. Writes `block_centre`,
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
6. **`build_exclusion_tiles`** — materialise the exclusion tiles, their
   bitmasks, and the `excl_jblock_offsets` / `excl_jblocks` CSR skip-list
   from the exclusion topology under the current block membership,
   entirely on the device. It runs as a short kernel sub-pipeline on the
   default stream and copies nothing to and from the host:

   a. **Invert the sort.** `build_atom_slot` writes the
      slot-for-each-atom permutation `atom_slot[sorted_particle_ids[s]] = s`
      (one thread per slot) into a per-rebuild device scratch of length
      `N`, so a canonical atom id maps to its current slot without a host
      copy.
   b. **Emit per-pair keys.** `emit_exclusion_tile_keys` enumerates every
      canonical modified pair `(a, b)` (`a < b`) directly from the
      device-resident exclusion topology (`atom_excl_offsets` /
      `atom_excl_partners`; a modified pair is any partner entry, and the
      `a < b` filter yields each unordered pair exactly once). For each
      pair it reads slots `(sa, sb) = (atom_slot[a], atom_slot[b])`, orders
      them so `si ≤ sj`, and forms blocks `(bi, bj) = (si/32, sj/32)` and
      lanes `(li, lj) = (si%32, sj%32)`. It writes one record — the
      block-pair key `(bi, bj)` together with the lane pair — at the pair's
      own fixed index in a per-rebuild scratch of length `E` (the
      modified-pair count, fixed for the run), so placement needs no
      atomics.
   c. **Sort by block pair.** `sort_exclusion_keys` stably radix-sorts the
      records by their `(bi, bj)` key (the reproducible radix sort of
      *Sorted-Blocks-by-Volume*, ties broken by record index), grouping all
      records that share a block pair into one contiguous run in ascending
      `(bi, bj)` order.
   d. **Reduce to tiles.** `reduce_exclusion_tiles` marks each run head (a
      record whose key differs from its predecessor), prefix-scans the head
      flags to assign tile indices, and writes the tile count to
      `interaction_count[2]`. Each tile `t` records
      `exclusion_tile_iblocks[t] = bi` and `exclusion_tile_jblocks[t] = bj`,
      and every record in the run ORs its lane bit into
      `exclusion_tile_masks[t·32 + li]` (bit `lj`) — plus the symmetric bit
      `exclusion_tile_masks[t·32 + lj]` (bit `li`) for a self-block tile
      (`bi == bj`). Because the mask is an OR, it is independent of
      intra-run record order.
   e. **Build the CSR skip-list.** `build_exclusion_csr` scans the sorted,
      unique tiles into `excl_jblock_offsets` (length `n_blocks + 1`) and
      `excl_jblocks` so that
      `excl_jblocks[excl_jblock_offsets[bi] .. excl_jblock_offsets[bi+1]]`
      lists the ascending `bj` of every exclusion tile whose i-block is
      `bi`. `find_blocks_with_interactions` (step 7) consults this CSR to
      skip exclusion-tile candidate block pairs.

   The tile set, their ascending `(bi, bj)` order, their bitmasks, and the
   CSR are a pure function of the exclusion topology and
   `sorted_particle_ids` — a permutation inversion, a fixed-index emit, a
   stable sort, and an order-independent OR — so they are byte-identical
   across runs. The mask marks *which* pairs are modified; the force pass
   reads each flagged pair's per-fragment scale at evaluation time, so a
   full exclusion (`scale == 0`) and a 1-4 pair (fractional scale) are
   flagged identically and distinguished only by the scale the force pass
   applies. As its final action a single designated thread compares
   `interaction_count[2]` against `exclusion_tiles_capacity` and sets the
   `exclusion_tiles_high_water` / `exclusion_tiles_overflow` bits of
   `neighbor_status` (see *Capacity*); the count and status stay
   device-resident.
7. **`find_blocks_with_interactions`** — the main construction
   kernel. One warp per i-block (iterated through
   `sorted_blocks`). For each candidate j-block (from inner
   sweep), the warp:
   - **Skips the candidate when `(bi, bj)` is an exclusion tile** (a
     lookup against the exclusion-tile table from step 6). That block
     pair's interactions — excluded and non-excluded alike — are
     computed by the exclusion-tile pass, so the bulk list omits them
     entirely, keeping every bulk and single-pair output exclusion-free
     without dropping or double-counting any pair.
   - Computes the centre-to-centre bbox-prune test (`Δ² ≤
     (r_search + radius_i + radius_j)²`) using
     `sorted_block_centre` and `sorted_block_bbox`.
   - For surviving candidates, loads the j-block's 32 atoms and
     tests each j-atom against all 32 i-atoms (via warp-shuffled
     i-positions). Each lane computes a 32-bit
     `i_hit_mask` for its j-atom: bit `m` is set when i-atom `m`
     is within `r_search` of this j-atom (and is not the same
     atom). The lane records `any_hit = (i_hit_mask != 0)`.
   - Computes `hit_ballot = __ballot_sync(any_hit)` and
     `n_hits = __popc(hit_ballot)` — the count of j-atoms in
     this (i-block, j-block) candidate that hit at least one
     i-atom.
   - **Sparse-tile path.** When `n_hits <= MAX_BITS_FOR_PAIRS`,
     the warp emits one entry per `(i_atom, j_atom)` hit to
     `single_pair_atoms`. Each lane with `any_hit` iterates the
     set bits of its `i_hit_mask`; for each set bit `m`, the
     lane atomically claims a slot via
     `atomicAdd(&interaction_count[1], 1u)` and writes
     `(warp_iid[m], jid)` to
     `single_pair_atoms[2 · slot]` and
     `single_pair_atoms[2 · slot + 1]`. The hits do NOT enter the
     packed-neighbour staging buffer.
   - **Dense-tile path.** When `n_hits > MAX_BITS_FOR_PAIRS`,
     the j-atoms with `any_hit` are compacted into a per-warp
     staging buffer of size `BUFFER_SIZE` (default 256). When the
     buffer reaches 32, the warp flushes 32 j-atoms into the
     global `interacting_tiles[pos]` /
     `interacting_atoms[pos · 32 + ℓ]` arrays, advancing
     `interaction_count[0]` via `atomicAdd`.
   - At the end of the i-block sweep, the partial tail of the
     dense-path staging buffer is flushed; the final entry pads
     unused slots with `N` (the sentinel "no atom" value) so the
     force kernel's bounds check skips them.

   `MAX_BITS_FOR_PAIRS` is `3` (compile-time constant in
   `kernels/neighbor.cu`).

   As its final action the kernel writes the live counts to
   `interaction_count[0]` and `interaction_count[1]` on the device and,
   from a single designated thread, sets the `*_high_water` and
   `*_overflow` bits of `neighbor_status` by comparing each count
   against its capacity and high-water mark (see *Capacity*). The
   counts and status are left device-resident; the kernel returns no
   value to the host.

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
  `single_pair_atoms` measured in pairs (the `CudaSlice<u32>` has
  `2 · single_pairs_capacity` slots).
- `tile_pair_growth_factor: f64` — geometric multiplier applied to
  a capacity when it is grown. Greater than 1.0. Default 1.5.
- `tile_pair_fill_threshold: f64` — fraction of a capacity at which
  a build is treated as near-full and the capacity is grown ahead of
  any dropped entry. In the open interval `(0, 1)`. Default 0.8.

Capacities are sized to the *actual* interaction count, never to the
`O(n_blocks²)` all-pairs upper bound. For a cutoff system the live
entry count is `O(N)` — proportional to the number of interacting
atom pairs, not to `n_blocks²`. The initial seed is `O(N)`, a small
multiple of `n_blocks`, clamped down to the all-pairs reference
`n_blocks²` for tiny systems. There is no configuration knob for the
initial capacity; the probe rebuild (below) determines it.

The exclusion-tile buffers (`exclusion_tile_iblocks`,
`exclusion_tile_jblocks`, `exclusion_tile_masks`, `excl_jblocks`) carry
their own capacity `exclusion_tiles_capacity`, seeded `O(n_blocks)` and
grown by the same count-free geometric high-water mechanism as the tile
and single-pair buffers. The exclusion-tile count has a hard static upper
bound — the fixed modified-pair count `E`, since each modified pair joins
exactly one tile — so growth converges after the probe rebuild and a
steady run never regrows them; the per-rebuild build scratch
(`atom_slot`, length `N`, and the sort records, length `E`) is sized once
from `N` and `E` and never grows.

**Device-resident counts.** A steady-state rebuild copies no count
to the host. The construction kernel writes the live counts to the
two-element device buffer `interaction_count` (`[0]` = packed-tile
entries, `[1]` = single pairs). Every downstream consumer launches a
grid sized by a host-known capacity and reads the live count from
`interaction_count` on the device:

- `histogram_entries_by_iblock`, `scatter_entries_by_iblock`, and the
  i-block prefix scan launch over `interacting_tiles_capacity` (and
  `n_blocks`); threads past `interaction_count[0]` exit early.
- The i-block prefix scan's trailing offset sentinel
  `iblock_offset[n_blocks]` is written by a device thread that reads
  `interaction_count[0]`, not from a host value.
- The packed-neighbour force pass launches over `n_blocks`; the
  single-pair force pass launches over `single_pairs_capacity` and
  reads `interaction_count[1]` on the device (see *Single-Pair Pass*).
- The exclusion-tile force pass launches over `exclusion_tiles_capacity`
  and reads the live tile count `interaction_count[2]` on the device;
  warps past that count exit early (see *Exclusion-Tile Pass*). The
  exclusion-tile build (step 6) writes the count and sets the
  exclusion-tile high-water / overflow bits, so this pass never needs a
  host-known tile count.

**Status word.** `CellListData` carries the single-`u32` device
buffer `neighbor_status` (see `neighbor-list.md` *Displacement Check*)
whose bits are:

| Bit | Name | Writer |
|---|---|---|
| 0 | `displacement_tripped` | displacement-check kernel, every step |
| 1 | `tiles_high_water` | construction kernel |
| 2 | `single_pairs_high_water` | construction kernel |
| 3 | `tiles_overflow` | construction kernel |
| 4 | `single_pairs_overflow` | construction kernel |
| 5 | `exclusion_tiles_high_water` | exclusion-tile build (step 6) |
| 6 | `exclusion_tiles_overflow` | exclusion-tile build (step 6) |

Each list's producing stage sets its own bits from a single designated
device thread that compares the live count `interaction_count[c]` against
the capacity `capacity_c` via `atomicOr(neighbor_status, …)`: the
construction sweep (step 7) sets the tile and single-pair bits, and the
exclusion-tile build (step 6) sets the exclusion-tile bits. In every case:

- `interaction_count[c] > floor(capacity_c · tile_pair_fill_threshold)`
  sets the matching `*_high_water` bit. The build is **complete** — no
  entry was dropped — but the capacity is nearly full.
- A flush that would have exceeded `capacity_c` (an entry would be
  dropped) sets the matching `*_overflow` bit. The construction stops
  writing that buffer past capacity while `interaction_count[c]` keeps
  accumulating the true required count via `atomicAdd`.

**Host response.** The host reads `neighbor_status` exactly once per
batch boundary, folded into the displacement-check read it issues
anyway (see `neighbor-list.md` *Rebuild Policy*); a steady-state
rebuild therefore issues **no** device-to-host transfer of its own.
The host acts on the bits as follows:

- **High-water, no overflow:** grow each flagged capacity geometrically
  to `ceil(capacity_c · tile_pair_growth_factor)`, reallocate the
  buffer, and run a fresh rebuild into the resized buffers so the
  populated list matches the new allocation. `pre_step` reports
  `reallocated = true` and the runner recaptures the phase graph (see
  `cuda-graphs.md` *Neighbor-List Pre-Step Decomposition*). Because
  high-water fires below capacity, the build that raised it dropped
  nothing, so the in-flight list stays correct until the grow-and-
  rebuild completes. Geometric growth is count-free — the host never
  reads the count — and converges in `O(log)` steps for any density.
- **Overflow:** a build dropped within-`r_search` entries, violating
  the no-silent-drop guarantee (`architecture.md`). `pre_step` returns
  `Err(NeighborListError::PackedNeighborOverflow { buffer })`, halting
  the run. With `tile_pair_fill_threshold < 1` and
  `tile_pair_growth_factor > 1`, and atom motion between rebuilds
  bounded by `r_skin / 2`, the per-rebuild count cannot climb from
  below the high-water mark to past capacity between two consecutive
  rebuilds, so this state is unreachable in a well-behaved run and
  exists only as a guard against pathology.

Growth is permitted at *any* rebuild, including one that runs at a
batch boundary inside a CUDA-graph-captured phase. A captured graph's
buffer pointers and launch dimensions must be stable only for the
lifetime of one graph instance, not for the whole phase, so the
buffer-sizing strategy is decoupled from the capture lifetime.

**Probe rebuild.** The first rebuild of a phase runs before CUDA-graph
capture (see `cuda-graphs.md` *Capture Lifecycle*). It reads
`neighbor_status` synchronously and grows-and-retries — growing each
flagged capacity geometrically and re-running the construction from
step 6 (steps 1–5 are not repeated) — until neither a high-water nor
an overflow bit is set, sizing the initial capacities with headroom
below `tile_pair_fill_threshold`. The probe runs once per phase and is
not part of the captured replay loop, so its blocking read does not
appear in the steady-state per-step cost.

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
delta_virial = factor * r2      // composer-derived pair virial (factor is masked, exclusion-scaled)

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

The fast-class pair-force pipeline has three JIT-composed pair-force
passes — the exclusion-tile pass, the packed-neighbour (bulk) pass, and
the single-pair pass. Each writes into the same per-particle fixed-point
accumulators (`fast_total_forces_fp_*`,
`fast_total_potential_energies_fp`, `fast_total_virials_fp`); the
finaliser converts to `Real` after all passes have run. Only the
exclusion-tile pass consults exclusion data; the other two evaluate
`heddle_jit_eval_pair_sum` with no exclusion lookup or scale multiply
(see *Exclusion Handling*).

### Exclusion-Tile Pass <!-- rq-fa0b3d10 -->

For every exclusion tile `t` in `[0, interaction_count[2])`:

- `bi = exclusion_tile_iblocks[t]`, `bj = exclusion_tile_jblocks[t]`
  (`bi ≤ bj`).
- Load the full 32 i-atoms of `bi` and 32 j-atoms of `bj` from
  `tile_sorted_posq[bi · 32 + lane]` / `tile_sorted_posq[bj · 32 + lane]`
  (two coalesced loads). Inactive slots (`≥ N`) carry `+∞` and fail the
  cutoff.
- Read this lane's modified-pair mask row `exclusion_tile_masks[t · 32 + lane]`.
- Evaluate the per-tile single-periodic-copy (SPC) predicate on block `bi`
  (the tile's i-block) — the same predicate the packed-neighbour pass uses
  (see *Single-Periodic-Copy Fast Path*), read from `block_bbox[bi]` and the
  lattice, warp-uniform across the tile's 32 lanes. When it holds, each lane
  wraps its own `bi` atom and its own `bj` atom once against the i-block
  centre `block_centre[bi]` with `triclinic_wrap_against_center`, and the
  inner loop computes `dx = pi − pj` directly with **no** per-pair
  `heddle_jit_triclinic_min_image` call; otherwise the inner loop takes the
  per-pair min-image path. The branch is warp-uniform, so exactly one path
  runs per tile per launch.
- Run the 32-iteration diagonal shuffle over all 32×32 lane pairs,
  skipping only the self-block (`bi == bj`) diagonal. For each surviving
  lane pair `(i_lane, j_lane)`:
  - If its modified-pair mask bit is set, the pair's contribution is
    summed through the scale-aware evaluator: each fragment's
    `(factor, energy)` is multiplied by that fragment's per-pair scale
    `exclusion_scale(i, j)`. A fully-excluded pair (`scale == 0` in every
    fragment) therefore contributes nothing, and a 1-4 pair contributes a
    scaled force and energy.
  - If its mask bit is clear, the pair is evaluated at full strength with
    **no scale lookup** (the scale is `1`).
  The scaled contribution is added to the i-atom and j-atom fixed-point
  slots (Newton's 3rd via `±`). A self-block tile applies only the i-side
  (each intra-block pair is visited from both orderings and the mask is
  symmetric, so both atoms receive their scaled force exactly once); a
  cross-block tile applies both sides. The bulk max-cutoff mask applies
  as elsewhere, so a modified pair beyond the cutoff contributes nothing.

### Packed-Neighbour Pass <!-- rq-a4b9e702 -->

For every entry `pos` in `[0, interaction_count[0])`:

- `x = interacting_tiles[pos]`.
- Load 32 i-atoms of `x` (positions from
  `tile_sorted_positions_*[x · 32 + lane]`, original atom IDs from
  `sorted_particle_ids[x · 32 + lane]`).
- Load 32 j-atoms from `interacting_atoms[pos · 32 + lane]`. Each
  is an original atom ID; load its position from
  `positions_*[j_atom_id]`. If `j_atom_id >= N`, the slot is the
  partial-tail padding; treat as inactive.
- Run the 32-iteration diagonal shuffle (described in *Diagonal
  Shuffle* below). For each pair the lane invokes the composer's
  per-pair evaluator `heddle_jit_eval_pair_sum` (see
  `jit-composed-pair-force.md`), which sums each fragment's
  `evaluate(r², inv_r, r, i, j, …)`. **No `exclusion_scale` is applied**:
  the construction guarantees no bulk entry contains a modified pair,
  so every pair here is a full-strength interaction.
- AtomicAdd per-lane accumulators to the fixed-point buffer using
  i-atom and j-atom original IDs.

### Single-Pair Pass <!-- rq-b28a6d96 -->

For every entry `k` in `[0, interaction_count[1])`:

- `atom_i = single_pair_atoms[2k]`,
  `atom_j = single_pair_atoms[2k + 1]`.
- One thread per pair. The thread loads both positions, computes
  `(dx, dy, dz, r², inv_r, r)`, invokes the same
  `heddle_jit_eval_pair_sum` evaluator (again with **no exclusion
  scale** — single pairs are exclusion-free by construction), and
  atomicAdds the per-fragment contribution to both atoms' fixed-point
  slots. Newton's 3rd is observed by adding `+(factor · d)` to atom `i`
  and `−(factor · d)` to atom `j` per component.

### Diagonal Shuffle <!-- rq-18847c46 -->

The packed-neighbour pass's 32-iteration inner loop pattern:

```text
// initial j-side state per lane: lane ℓ holds j-atom ℓ
i_lane = lane
tj = lane
for t in 0..32:
    if (i_atom_id_at(i_lane) != j_atom_id_at(tj)
        && j_atom_id_at(tj) < N) {
        // evaluate pair (i_atom = lane, j_atom_lane = tj):
        compute delta, factor, energy
        virial = factor * r2   // composer-derived pair virial
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

The pair-skip predicate `i_atom_id != j_atom_id` covers the
self-pair case when an i-block's atoms happen to appear in the
same entry's j-atom list (e.g., when a sparse-tile boundary
straddles the i-block); the single-pair pass uses the same
predicate at the per-thread level.

After 32 iterations the j-side accumulators have rotated 32 times
and are back at their starting lane. Lane ℓ's `j_*` accumulators
hold the total contribution to the j-atom that started in lane ℓ
(i.e., the j-atom at j-lane `ℓ` of the entry).

The 32-iteration loop count is constant — no early exit. Per-pair
gating is by `j_atom_id < N` (and by the optional
`r2 <= cutoff_squared` check inside each per-slot fragment's
`evaluate`).

### Single-Periodic-Copy Fast Path <!-- rq-5ce17997 -->

The packed-neighbour pass and the exclusion-tile pass each evaluate
a per-i-block single-periodic-copy (SPC) predicate at the top of the
outer loop and branch on the result. The predicate is uniform across
the warp (all 32 lanes processing one i-block — or, for the
exclusion-tile pass, one tile — compute the same value from the
block's bbox, the lattice constants, and the compile-time max
cutoff), so there is no per-pair warp divergence and only one
warp-wide control flow path executes per i-block / per tile per
launch. The exclusion-tile pass uses the tile's i-block `bi` as its
predicate-and-centre block: the same predicate on `block_bbox[bi]`,
and both the `bi` atoms and the `bj` atoms wrap against
`block_centre[bi]`.

The two code paths are:

- **Min-image path.** For every pair the inner loop calls
  `heddle_jit_triclinic_min_image(dx, dy, dz, lattice…)` to wrap
  `dx = pi - pj` into the canonical `[-L/2, L/2)` displacement
  per lattice direction before the `r²` evaluation.
- **SPC path.** Before entering the inner loop, each lane wraps
  its own `pi` and its own `pj` into the periodic image closest to
  the i-block centre `block_centre[i_block]` (for the exclusion-tile
  pass, `block_centre[bi]`), using
  `triclinic_wrap_against_center(pos, centre, lattice)`. After
  both wraps, the inner loop computes `dx = pi - pj` and **does
  not call** `heddle_jit_triclinic_min_image` — `dx` is already
  the canonical min-image displacement. (In the packed-neighbour
  pass `pi` comes from `tile_sorted_posq` and `pj` from `posq`; in
  the exclusion-tile pass both come from `tile_sorted_posq`.)

The wrap helper `triclinic_wrap_against_center(pos, centre,
lattice)` shifts `pos` to the periodic image closest to `centre`:

```text
delta = pos - centre
(s_a, s_b, s_c) = triclinic_cart_to_frac(delta, lattice)
(k_a, k_b, k_c) = (-round(s_a), -round(s_b), -round(s_c))
pos += k_a · a + k_b · b + k_c · c
```

where `(a, b, c)` are the lattice vectors. The result satisfies
`|pos − centre|_axis ≤ L_axis / 2` in each lattice direction.

#### Per-Block Eligibility Predicate <!-- rq-4b20b449 -->

The SPC predicate is

```text
spc =     orthorhombic
      AND (0.5 · L_x − block_bbox[i_block].x ≥ MAX_CUTOFF)
      AND (0.5 · L_y − block_bbox[i_block].y ≥ MAX_CUTOFF)
      AND (0.5 · L_z − block_bbox[i_block].z ≥ MAX_CUTOFF)
```

where:

- `orthorhombic` is `(xy == 0 AND xz == 0 AND yz == 0)`, read
  from the lattice constants the kernel already loads at entry.
- `block_bbox[i_block].{x, y, z}` are the i-block's per-axis bbox
  half-extents in Cartesian coordinates, populated by
  `compute_block_bbox` at every rebuild.
- `MAX_CUTOFF` is the aggregated maximum interaction cutoff
  across all active fast-class pair-force slots. The composer
  embeds it as a `#define HEDDLE_JIT_MAX_CUTOFF R(…)` in the
  generated source alongside `HEDDLE_JIT_MAX_CUTOFF_SQUARED`.

Correctness rationale. For an i-block whose bbox half-extent
along axis `d` is `B_d`, every i-atom is within `B_d` of the
centre along that axis. After wrapping `pj` against the centre,
`|pj − centre|_d ≤ L_d / 2`. The predicate only needs to be safe
for **in-cutoff** pairs — an in-cutoff pair is one whose true
min-image separation is `≤ MAX_CUTOFF`, so its j-atom is within
`MAX_CUTOFF` of some i-atom and therefore within `MAX_CUTOFF + B_d`
of the centre. When `0.5 · L_d − B_d ≥ MAX_CUTOFF`, the
centre-image wrap and the min-image wrap select the same periodic
copy for such a j-atom, so `pi − pj` is already the canonical
min-image displacement. Out-of-cutoff pairs are zeroed by the
existing `cutoff_mask` regardless of which image the wrap selected:
the centre-wrapped separation is never smaller than the true
min-image separation (the min-image is the closest of all images,
so any other image is at least as far), so a pair whose min-image
separation exceeds the cutoff cannot be pulled below it by the
centre wrap. This holds for both passes. The packed-neighbour
pass's j-atoms reach the kernel already filtered to within-search
candidates by the construction sweep; the exclusion-tile pass
instead evaluates every one of its block pair's `32×32` lane pairs
(the block pair is skipped by the construction sweep), but the same
in-cutoff / out-of-cutoff split applies, so the predicate is safe
there too.

#### Triclinic Boxes <!-- rq-412fea28 -->

The predicate gates SPC on `orthorhombic`. Triclinic boxes (any
of `xy`, `xz`, `yz` non-zero) take the min-image path on every
i-block and every exclusion tile regardless of bbox extent.
Extending SPC to triclinic boxes is future work that would replace
the per-axis box-length check with a projection of the i-block
bbox onto each face normal; the kernel helper
`triclinic_wrap_against_center` already handles arbitrary lattice
geometry, so the change would be confined to the eligibility
predicate.

#### Box-Geometry Transitions <!-- rq-1ccb6e53 -->

Under NPT or NPH the box and the per-block bbox both change
across a step. The predicate is evaluated freshly at every
i-block / tile of every launch (of both the packed-neighbour and
the exclusion-tile pass), reading the current lattice constants
(passed as a kernel argument) and the current `block_bbox` (one
of the buffers populated by the per-rebuild
`compute_block_bbox`). No host-side cache or CUDA-graph
re-capture is required when eligibility flips. The kernel
arguments and the `block_centre` / `block_bbox` device pointers
remain valid across a captured graph replay; their contents are
read at every invocation.

#### Single-Pair Pass <!-- rq-02abafdd -->

The single-pair pass calls `heddle_jit_triclinic_min_image`
inside its per-pair evaluation. It does not partition atoms into
blocks (it launches one thread per pair), so there is no
per-block centre to wrap against. The marginal cost of the
per-pair min-image call here is small because the pass handles
at most a few thousand pairs per step, compared with millions
in the packed-neighbour pass.

### Launch Configuration <!-- rq-8db4fbff -->

Each pass launches as a separate CUDA kernel on the
`particle_buffers.device`'s default stream:

- **Packed-neighbour pass.** Grid
  `⌈n_iblocks / WARPS_PER_BLOCK⌉ × WARPS_PER_BLOCK = n_iblocks`
  blocks of `BLOCK_SIZE = 256` threads (`WARPS_PER_BLOCK = 8`).
  One block per i-block; warps within a block share that
  i-block's i-side accumulators via shared memory. Launched
  unconditionally when at least one fast-class pair-force slot
  is active. The SPC fast-path predicate is evaluated inside the
  kernel at every i-block; the host always passes
  `block_centre` and `block_bbox` and dispatches a single entry
  point per write-EV variant.
- **Single-pair pass.** Grid
  `⌈interaction_count[1] / 256⌉` blocks of `256` threads, one
  thread per single-pair entry. Launched only when
  `interaction_count[1] > 0`.

The two launches run in this order so the single-pair pass
observes the packed-neighbour pass's writes via the per-stream
ordering of the default stream.

## JIT Composer Integration <!-- rq-ffbee244 -->

The per-slot `PairForceFragment` source contract is documented in
`rqm/forces/jit-composed-pair-force.md`: each fragment provides
`functor_struct`, `functor_source`, `entry_point_args`,
`functor_init_source`, and a `cutoff: CutoffHandling`
declaration. The functor's interface is
`cutoff_squared(i_type, j_type, i, j) -> Real` and
`evaluate(r2, inv_r, r, qi, qj, i_type, j_type, i, j, factor,
energy)`. The fragment carries no `exclusion_scale` method:
exclusions are applied by the exclusion-tile bitmask, not by the
per-pair evaluator. The composer emits one per-pair evaluator:

- `heddle_jit_eval_pair_sum<WriteEv>(composite, r2, inv_r, r, i,
  j, factor, energy)` — sums each fragment's contribution. Used by
  all three passes (exclusion-tile, packed-neighbour, single-pair);
  none applies an exclusion scale. The caller derives the per-pair
  scalar virial as `factor * r²` from the summed `factor`.

The evaluator applies the cutoff handling described in
`jit-composed-pair-force.md` (the per-fragment cutoff guard,
collapsed when `CutoffHandling::Uniform(c)` matches the outer
max-cutoff mask).

### Packed-Neighbour Entry-Point Arguments <!-- rq-e8ba1aff -->

The composer emits two packed-neighbour entry points per
JIT-composed module:

- `heddle_jit_composed_pair_force_f`
- `heddle_jit_composed_pair_force_fev`

Both share the same argument list, in this order:

```text
const Real4 *posq,
const Real4 *tile_sorted_posq,
const Real *block_centre,
const Real *block_bbox,
const unsigned int *sorted_particle_ids,
const unsigned int *iblock_offset,
const unsigned int *sorted_interacting_atoms,
unsigned int n_iblocks,
const Real *lattice,
unsigned long long *fast_force_x_fp,
unsigned long long *fast_force_y_fp,
unsigned long long *fast_force_z_fp,
unsigned long long *fast_energy_fp,
unsigned long long *fast_virial_fp,
```

`block_centre` is the per-block centre buffer populated by
`compute_block_bbox` (`Real` array of length `4 · n_blocks`,
packed as `(cx, cy, cz, max_disp_sq)`; the SPC wrap reads
`.xyz`). `block_bbox` is the per-block bbox half-extent buffer
populated by the same kernel (`Real` array of length
`3 · n_blocks`, packed as `(dx, dy, dz)`; the SPC predicate reads
all three). Both pointers are always present regardless of
whether any i-block actually takes the SPC path.

Per-fragment arguments are appended after the common arguments in
canonical slot order. The trailing `unsigned int n` is the final
argument.

The `_fev` entry point uses the same argument list as `_f`; the
`_f` entry point's emitted inner loop simply does not increment
`fast_energy_fp` / `fast_virial_fp`.

### Single-Pair Entry-Point Arguments <!-- rq-f119bc11 -->

The single-pair pass has its own pair of entry points:

```text
extern "C" __global__ void heddle_jit_composed_pair_force_single_f(...)
extern "C" __global__ void heddle_jit_composed_pair_force_single_fev(...)
```

Common arguments, in order:

```text
const Real4 *posq,
const unsigned int *single_pair_atoms,
unsigned int single_pair_count,
const Real *lattice,
unsigned long long *fast_force_x_fp,
unsigned long long *fast_force_y_fp,
unsigned long long *fast_force_z_fp,
unsigned long long *fast_energy_fp,
unsigned long long *fast_virial_fp,
```

Per-fragment arguments are appended in canonical slot order
followed by the trailing `unsigned int n`. The per-fragment list
is identical to the packed-neighbour entry point's per-fragment
list (the same `bind_pair_force_args` invocations populate it).

## Per-Step Pipeline <!-- rq-0acba2a0 -->

The fast-class pair-force pipeline runs the following sequence
every step:

| # | Stage | When |
|---|---|---|
| 1 | `scatter_positions_to_tile_order` | Every step |
| 2 | Pre-step neighbour-list check | Every step; rebuild may run if displacement exceeds `r_skin / 2` |
| 3 | `cudaMemsetAsync` zeroing fast fixed-point buffers | Every step (or only when re-evaluating the Fast class) |
| 4 | `heddle_jit_composed_pair_force_excl_{f,fev}` (exclusion-tile pass) | Once per step (only if `interaction_count[2] > 0`) |
| 5 | `heddle_jit_composed_pair_force_{f,fev}` (packed-neighbour pass) | Once per step; per-i-block SPC predicate inside the kernel selects between the min-image branch and the centre-wrap branch |
| 6 | `heddle_jit_composed_pair_force_single_{f,fev}` (single-pair pass) | Once per step (only if `interaction_count[1] > 0`) |
| 7 | `finalize_fast_class_forces` | Once per step; converts fixed-point to Real and adds into `ParticleBuffers.forces_*` |

The three pair passes (4–6) all accumulate into the same fixed-point
buffers and may run in any order; the finaliser (7) follows them.

When step 2 triggers a rebuild, the rebuild pipeline (the *Construction
Pipeline* steps, including `build_exclusion_tiles` and the second
`scatter` since `sorted_particle_ids` has changed) runs before step 3.

## Configuration <!-- rq-9527bd2d -->

The `[neighbor_list]` section of the simulation config carries:

- `mode: "cell-list" | "all-pairs"` — unchanged from the existing
  `NeighborListConfig`.
- `r_skin: f64` (length, in active unit system) — unchanged from
  the existing `NeighborListConfig`.
- `tile_pair_growth_factor: f64` — geometric multiplier applied to
  the `interacting_tiles` and `single_pair_atoms` capacities when one
  is grown. Must be greater than 1.0. Default 1.5.
- `tile_pair_fill_threshold: f64` — fraction of a capacity at which a
  build is treated as near-full and the capacity is grown ahead of any
  dropped entry (see *Capacity*). Must lie in the open interval
  `(0, 1)`. Default 0.8.

The `interacting_tiles` and `single_pair_atoms` capacities themselves
are not configurable. They are sized automatically from the first
rebuild's true interaction count (see *Capacity*) and grown on demand,
so there is no initial-capacity field to tune.

The field `max_neighbors` is **not** part of `NeighborListConfig`.
Specifying it in the configuration file is a load-time error with
an explanatory message that the field is no longer used.

Exclusion scales are per fragment (see *Exclusion Handling*). Each
fragment scale is a value in `[0, 1]`; the two fragment scales of a pair
are independent, so full exclusions (`0`), fractional 1-4 scaling
(e.g. a Lennard-Jones scale of `0.5` with a Coulomb scale of `0.833`),
and per-fragment combinations (e.g. `0` Lennard-Jones with `1` Coulomb)
all load. Scale-range validation lives in the topology parser (see
`topology.md`): a scale outside `[0, 1]` or a NaN is a load-time error
naming the offending pair, and the run does not start.

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
5. **Bit-exact accumulation via integer atomics.** Per-particle
   force, energy, and virial accumulators are 64-bit unsigned
   integers interpreted as two's-complement signed integers.
   Integer addition is associative under two's-complement
   arithmetic regardless of operand order. `atomicAdd` on
   `unsigned long long` is the load-bearing primitive; its
   per-atom result is independent of how many warps wrote to that
   atom and in what order.
6. **Deterministic conversion.** The fixed-point-to-real
   conversion `(int64) sum / 2^32` is a deterministic function of
   the integer sum; the final `Real` addition into
   `ParticleBuffers.forces_*` happens in a kernel that reads each
   atom's slot exactly once.
7. **Deterministic exclusion tiles.** The set of exclusion tiles, their
   order, and their bitmasks are a pure function of the exclusion
   topology and `sorted_particle_ids`. The on-device build inverts the
   sort (a permutation), emits one record per modified pair at that pair's
   fixed index, stably radix-sorts the records by `(bi, bj)` key (ties
   broken by record index), and segment-reduces each run into one tile in
   ascending `(bi, bj)` order; each tile's mask is the OR of its run's
   lane bits, which is order-independent. The tile set, its order, and the
   bitmasks are therefore fixed rather than scheduling-dependent, and the
   per-fragment scale the force pass applies to each flagged pair is
   identical across runs. (Slot order would in any case be irrelevant to
   the forces, because the exclusion-tile pass, like the bulk pass,
   accumulates through the per-particle fixed-point accumulators.)

The reproducibility scope is GPU-vs-GPU on the same hardware,
matching `architecture.md`. CPU-vs-GPU is not promised.

## Feature API <!-- rq-e1643cc5 -->

### Types <!-- rq-86f757e0 -->

- `NeighborListState` adds the fields documented under *Atom <!-- rq-b5c18e6b -->
  Blocks*, *Tile-Sorted Position View*, *Block Bounding Boxes*,
  *Sorted-Blocks-by-Volume*, and *Neighbour List* above — including the
  exclusion-tile buffers `exclusion_tile_iblocks`, `exclusion_tile_jblocks`,
  and `exclusion_tile_masks`, their `exclusion_tiles_capacity`, and the
  `interaction_count[2]` live exclusion-tile count. The per-particle
  padded `neighbor_list` and `neighbor_counts` fields are not present.
- `ForceField` adds the fixed-point class accumulator fields <!-- rq-e6b37d18 -->
  documented under *Fixed-Point Force Buffers*. The per-class
  `fast_total_forces_x/y/z` (and the `_potential_energies` /
  `_virials`) become `unsigned long long` fixed-point arrays; the
  per-slot `SlotOutputView` framework no longer issues 5-array
  bind targets for fast-class pair-force slots (it remains for
  bonded and angle slots, and for slow-class slots not using
  packed-neighbour).
- `NeighborListConfig` carries the fields documented under <!-- rq-3fbb8dea -->
  *Configuration*, including `tile_pair_growth_factor` (default 1.5,
  `> 1.0`) and `tile_pair_fill_threshold` (default 0.8, in `(0, 1)`).
  The `max_neighbors` field is absent; the TOML parser reports an
  explicit error if `max_neighbors` appears in the config text.
- `NeighborListError` carries the variant <!-- rq-2dda3169 -->
  `PackedNeighborOverflow { buffer: &'static str }` — returned by
  `NeighborListState::pre_step` when a steady-state build set a
  `*_overflow` bit of `neighbor_status`, meaning a packed-neighbour
  buffer would have dropped within-`r_search` entries, or the
  exclusion-tile build produced more tiles than `exclusion_tiles_capacity`.
  `buffer` names the buffer that overflowed (`"interacting_tiles"`,
  `"single_pair_atoms"`, or `"exclusion_tiles"`). This is the only
  neighbour-overflow error; there is no per-particle neighbour cap to
  exceed.
- `PairSnapshot` — host-side owned enumeration of the pair set <!-- rq-a0ab0088 -->
  represented by a `NeighborListState` at the time the snapshot
  was taken. Constructed by
  `NeighborListState::pair_snapshot(&self, device)`; see
  *Functions* below. The snapshot copies four device buffers to
  the host (`interacting_tiles`, `sorted_interacting_atoms`,
  `single_pair_atoms`, and the cell-list `sorted_particle_ids`)
  after first reading the live interaction counts. It owns those
  `Vec<u32>` buffers; iterating it involves no further device
  access. A single snapshot may be iterated any number of times.
  The snapshot is scoped to `NeighborListMode::CellList` neighbour
  lists; `NeighborListMode::Trivial` is out of scope (see *Out of
  Scope*).

### CUDA Kernels <!-- rq-2647cb7e -->

- `scatter_positions_to_tile_order(posq, sorted_particle_ids, <!-- rq-89245537 -->
  tile_sorted_posq, n)` — one thread per atom; block 256. Reads
  one `Real4` from `posq[sorted_particle_ids[k]]` per thread and
  writes it to `tile_sorted_posq[k]`.
- `compute_block_bbox(tile_sorted_posq, tile_atom_count, <!-- rq-9f947525 -->
  block_centre, block_bbox, n_blocks)` — one warp per block.
  Reads `.x/.y/.z` from `tile_sorted_posq` and ignores `.w`.
- `compute_sort_keys(block_bbox, sorted_blocks, n_blocks)` — one <!-- rq-e9ed7617 -->
  thread per block; preceded by a single-block reduction for the
  global bbox-sum range.
- `sort_blocks_by_volume(sorted_blocks, n_blocks)` — radix sort. <!-- rq-62f00166 -->
- `sort_block_data(sorted_blocks, block_centre, block_bbox, <!-- rq-c1afd2f3 -->
  sorted_block_centre, sorted_block_bbox, n_blocks)` — one
  thread per block.
- `find_blocks_with_interactions(sorted_blocks, <!-- rq-169e1d84 -->
  sorted_block_centre, sorted_block_bbox, tile_sorted_positions_*,
  interacting_tiles, interacting_atoms, single_pair_atoms,
  interaction_count, interacting_tiles_capacity,
  single_pairs_capacity, tiles_high_water_mark,
  single_pairs_high_water_mark, r_search_sq, lattice,
  force_rebuild_flag, n_blocks, n_atoms, neighbor_status)` — main
  construction kernel. One warp per i-block iterating through
  `sorted_blocks`; skips candidate j-blocks that form an exclusion tile
  with the i-block. Writes the live counts to `interaction_count` and,
  from a single designated thread, sets the `*_high_water` and
  `*_overflow` bits of `neighbor_status` (see *Capacity*); it returns
  no count or status to the host. The `*_high_water_mark` arguments are
  `floor(capacity · tile_pair_fill_threshold)`, computed on the host
  from the current capacities. The `MAX_BITS_FOR_PAIRS = 3` threshold
  is a compile-time `#define`.
- `build_atom_slot(sorted_particle_ids, atom_slot, n_atoms)` — device <!-- rq-5f871bd5 -->
  kernel; one thread per slot writes the slot-for-each-atom permutation
  `atom_slot[sorted_particle_ids[s]] = s` into a per-rebuild device
  scratch, so the exclusion-tile build maps a canonical atom id to its
  current slot without a host copy.
- `build_exclusion_tiles(...)` — the on-device exclusion-tile <!-- rq-8945395f -->
  sub-pipeline launched each rebuild (see *Construction Pipeline* step 6):
  the kernels `build_atom_slot`, `emit_exclusion_tile_keys`,
  `sort_exclusion_keys` (the reproducible radix sort), `reduce_exclusion_tiles`,
  and `build_exclusion_csr`, run in sequence on the default stream. It
  reads the device-resident exclusion topology (`atom_excl_offsets` /
  `atom_excl_partners`) and `sorted_particle_ids`, and writes
  `exclusion_tile_iblocks`, `exclusion_tile_jblocks`,
  `exclusion_tile_masks`, `excl_jblock_offsets`, `excl_jblocks`, and the
  tile count `interaction_count[2]` — all on the device — then sets the
  exclusion-tile high-water / overflow bits of `neighbor_status`. It maps
  each canonical modified pair (`a < b`, every partner entry of the
  exclusion topology) to its ordered block pair `(bi, bj)` (`bi ≤ bj`) and
  lanes under the current sort, sorts the per-pair records by `(bi, bj)`,
  and segment-reduces each block pair into one exclusion tile whose
  1024-bit mask ORs the pair's bit — plus the symmetric bit on a
  self-block tile. It uses per-rebuild device scratch sized once from
  `n_atoms` (the `atom_slot` permutation) and the fixed modified-pair
  count `E` (the sort records); it performs no host round-trip. The
  per-fragment scale each flagged pair receives is not stored in the tile;
  the force pass reads it from the per-fragment scale tables via
  `exclusion_scale(i, j)` at evaluation time.
- `heddle_jit_composed_pair_force_f` / <!-- rq-42e29605 -->
  `heddle_jit_composed_pair_force_fev` — JIT-composed packed-
  neighbour entry points; argument list documented under
  *Packed-Neighbour Entry-Point Arguments*. Each evaluates the
  per-i-block SPC predicate at runtime and branches between the
  centre-wrap fast path and the per-pair min-image path; the
  branch is uniform across the warp. They apply no exclusion scale.
- `heddle_jit_composed_pair_force_excl_f` / <!-- rq-9d34c158 -->
  `heddle_jit_composed_pair_force_excl_fev` — JIT-composed
  exclusion-tile entry points; one warp per exclusion tile. Load the
  full `bi` and `bj` blocks from `tile_sorted_posq`, take the tile's
  `exclusion_tile_masks` row, apply per-fragment scale to masked lane
  pairs, and skip the self-block diagonal (see *Exclusion-Tile Pass*).
  Receive `block_centre` and `block_bbox` (alongside `lattice`) and
  evaluate the per-tile SPC predicate on the tile's i-block `bi`,
  taking the centre-wrap fast path against `block_centre[bi]` when it
  holds and the per-pair min-image path otherwise (see *Single-Periodic-Copy
  Fast Path*). The predicate is warp-uniform, so the branch is taken
  once per tile with no per-pair divergence.
- `heddle_jit_composed_pair_force_single_f` / <!-- rq-3ddf259b -->
  `heddle_jit_composed_pair_force_single_fev` — JIT-composed
  single-pair entry points; argument list documented under
  *Single-Pair Entry-Point Arguments*.
- `finalize_fast_class_forces(fast_force_*_fp, fast_energy_fp, <!-- rq-9c80a966 -->
  fast_virial_fp, particle_forces_x, particle_forces_y,
  particle_forces_z, particle_potential_energies,
  particle_virials, n)` — one thread per atom; converts
  fixed-point sums to Real and `+=` into the particle buffers.
- `finalize_slow_class_forces` (analogous, when slow class uses <!-- rq-2c323eaa -->
  the fixed-point accumulator).

### Functions <!-- rq-c85fa8d1 -->

- `crate::gpu::scatter_positions_to_tile_order(kernels, buffers, <!-- rq-595e7ea4 -->
  sorted_particle_ids, tile_sorted_posq) -> Result<(), GpuError>`
  — launches the scatter kernel. Reads from `buffers.posq`.
- `crate::gpu::compute_block_bbox(kernels, tile_sorted_posq, <!-- rq-3a31b3f0 -->
  tile_atom_count, block_centre, block_bbox, n_blocks) ->
  Result<(), GpuError>`.
- `crate::gpu::sort_blocks_by_volume(kernels, block_bbox, <!-- rq-ad6f0de0 -->
  sorted_blocks, sorted_block_centre, sorted_block_bbox,
  n_blocks) -> Result<(), GpuError>` — runs the three sub-steps
  (compute keys, radix sort, sort_block_data) as a single
  callable.
- `crate::gpu::find_blocks_with_interactions(kernels, …) -> <!-- rq-68a9602b -->
  Result<(), GpuError>` — launches the construction kernel. The live
  counts stay in `interaction_count` and the near-full / overflow
  state stays in `neighbor_status`, both on the device; the wrapper
  copies nothing to the host. The host learns of high-water or overflow
  only through the once-per-batch `neighbor_status` read (see
  `neighbor-list.md` *Rebuild Policy*).
- `crate::gpu::finalize_fast_class_forces(kernels, fp_buffers, <!-- rq-5a7f78c4 -->
  particle_buffers, n) -> Result<(), GpuError>`.
- `NeighborListState::rebuild` runs the cell-list pre-step followed by <!-- rq-4896a257 -->
  the construction pipeline above. It zeros
  `neighbor_status` before the cell-list and construction kernels run,
  copies no interaction count to the host, and reports buffer growth
  through `PreStepOutcome.reallocated`. A steady-state (non-probe)
  rebuild issues no device-to-host transfer.
- `NeighborListState::pair_snapshot(&self, device: &Arc<CudaDevice>) <!-- rq-b5b33e00 -->
  -> Result<PairSnapshot, GpuError>` — first reads
  `interaction_count` to obtain the live `interacting_tiles_count`
  and `single_pairs_count`, then copies `interacting_tiles`
  (length `interacting_tiles_count`), `sorted_interacting_atoms`
  (length `interacting_tiles_count * 32`), `single_pair_atoms`
  (length `2 * single_pairs_count`), and the cell-list
  `sorted_particle_ids` (length `n_blocks * 32`) to the host, one
  synchronous transfer per buffer, and wraps the results in a
  `PairSnapshot`. Issues five host-facing `dtoh_sync_copy` calls
  in total (one for the counts, one for each of the four owned
  buffers) and does not read `neighbor_status`. Returns
  `Err(GpuError)` if any transfer fails; on `Ok` the snapshot
  owns every buffer it needs and requires no further device
  access. Panics if the state is in `NeighborListMode::Trivial`
  or `NeighborListMode::CellListOnly` (see *Out of Scope*).
- `PairSnapshot::iter(&self) -> impl Iterator<Item = (u32, u32)>` <!-- rq-6eaf3e99 -->
  — yields every unordered pair the pipeline processes for the
  neighbour list captured in the snapshot, exactly once, as a
  canonical `(u32, u32)` with `min < max`. Pairs enumerated from
  the packed-neighbour representation are decoded from
  `interacting_tiles` and `sorted_interacting_atoms` (see
  *Iteration Semantics* below); pairs enumerated from the
  sparse representation are decoded from
  `single_pair_atoms`. Duplicates (a pair reachable from both
  representations, or a pair reachable from more than one
  entry / lane rotation of the packed representation) are
  deduplicated so that the yielded set is exactly the pair set
  the pipeline evaluates. Iteration is total: every yielded
  pair carries two distinct in-range atom IDs
  (`i < j < n_atoms`) and every such pair the neighbour list
  contains appears in the iterator's output.
- `PairSnapshot::len(&self) -> usize` — returns the number of <!-- --> <!-- rq-e79cb5b5 -->
  unordered pairs the iterator would yield, computed on the
  host from the snapshot's buffers. Consistent with
  `iter().count()` by construction.

### Iteration Semantics <!-- --> <!-- rq-f6b6f93f -->

`PairSnapshot::iter` decodes the packed-neighbour representation
in the same order the pair-force kernel evaluates it, so
downstream diagnostic code observes the same pair set:

- For each packed entry `e` in `[0, interacting_tiles_count)`,
  the entry's i-block is `interacting_tiles[e]`. For each lane
  `l` in `[0, 32)`, the lane's i-atom ID is
  `sorted_particle_ids[i_block * 32 + l]` when that slot holds a
  real atom (i.e., the value is less than `n_atoms`) and is
  absent otherwise.
- Within an entry, each lane pairs against 32 j-atoms via the
  diagonal shuffle `j_lane = (l + r) & 31` for `r` in
  `[0, 32)`. The j-atom ID is
  `sorted_interacting_atoms[e * 32 + j_lane]`; a value equal
  to the sentinel (`n_atoms`) or larger is treated as absent.
- A `(i_atom, j_atom)` candidate is yielded only when both IDs
  are real (`< n_atoms`) and distinct. The pair is normalised
  to `(min, max)` before deduplication.
- For each sparse entry `k` in `[0, single_pairs_count)`, the
  atom IDs are `single_pair_atoms[2 * k]` and
  `single_pair_atoms[2 * k + 1]`. The pair is normalised to
  `(min, max)` and joined with the packed set through the same
  dedup filter.

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
- `__device__ static inline void <!-- rq-25515b9a -->
  triclinic_wrap_against_center(Real &x, Real &y, Real &z,
  Real cx, Real cy, Real cz, Real lx, Real ly, Real lz,
  Real xy, Real xz, Real yz)` — shifts `(x, y, z)` to the
  periodic image of itself that is closest to `(cx, cy, cz)`.
  Computes the fractional displacement `(δx, δy, δz) =
  (x - cx, y - cy, z - cz)` via `triclinic_cart_to_frac`, rounds
  each component to the nearest integer to obtain
  `(k_a, k_b, k_c)`, and subtracts `k_a · a + k_b · b + k_c · c`
  from `(x, y, z)`. After the call,
  `|x − cx|_axis ≤ L_axis / 2`. Lives in `kernels/pbc.cuh`
  alongside `triclinic_min_image` and is reused by both SPC
  entry-point variants.

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
- **Mixed-precision energy/virial accumulation.** Energy and
  virial use the same `2^32` fixed-point scale as forces. The
  shared scale is adequate for typical MD energy ranges but may
  saturate for unusually large systems; a separate finer scale
  for energies is deferred.
- **`PairSnapshot` coverage of `NeighborListMode::Trivial`.** The
  snapshot API enumerates the packed-neighbour representation
  (packed dense entries plus sparse single-pairs). A
  trivial-mode neighbour list writes an all-pairs list into the
  same buffers via a distinct construction path; the snapshot
  API is not required to reproduce that enumeration. Callers
  that want an all-pairs oracle enumerate `(i, j)` directly for
  `i < j < n_atoms` and do not need to walk `PackedNeighborData`.

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

  @rq-e8667000
  Scenario: Sparse (i-block, j-block) candidate routes its pairs to single_pair_atoms
    Given a candidate (i-block, j-block) pair that produces n_hits j-atoms with any_hit
    And n_hits <= MAX_BITS_FOR_PAIRS (= 3)
    When the construction kernel processes this candidate
    Then for every j-atom with any_hit and every i-atom in its i_hit_mask, one (atom_i, atom_j) entry is written to single_pair_atoms (advancing interaction_count[1] by one per (i, j) hit)
    And no entry is written to interacting_atoms for this candidate

  @rq-560d3be9
  Scenario: Dense (i-block, j-block) candidate routes its hits to interacting_atoms
    Given a candidate (i-block, j-block) pair that produces n_hits j-atoms with any_hit
    And n_hits > MAX_BITS_FOR_PAIRS (= 3)
    When the construction kernel processes this candidate
    Then the surviving j-atoms enter the per-warp staging buffer and contribute to interacting_tiles / interacting_atoms when the buffer reaches 32
    And no entry is written to single_pair_atoms for this candidate

  @rq-b1060817
  Scenario: MAX_BITS_FOR_PAIRS equals 3
    Given the construction kernel's compiled source
    Then `MAX_BITS_FOR_PAIRS` resolves to the literal 3

  # --- Uniqueness: no duplicate pair emission ---
  #
  # The pair-force pipeline is designed on the invariant that each
  # unordered (i, j) pair appears at most once in the union of
  # interacting_atoms (packed dense entries) and single_pair_atoms
  # (sparse entries). Any duplicate causes the pair's contribution
  # to be double-counted downstream. The following scenarios assert
  # the invariant directly rather than only through its downstream
  # effect on forces.

  @rq-bebff0e9
  Scenario: Packed and sparse outputs together list each unordered pair at most once
    Given a NeighborListState rebuild has completed
    When the host dtohs interacting_atoms (over
      [0, interaction_count[0])) and single_pair_atoms (over
      [0, interaction_count[1])), decomposes each packed entry into
      its 32 (i_atom, j_atom) rotations, and normalises every pair
      to canonical (min, max) order
    Then no canonical unordered pair appears more than once across
      the union of the two outputs

  @rq-7711b39b
  Scenario: Self-block sparse candidates do not double-emit intramolecular pairs
    Given a two-particle-block system with a molecule whose 8 atoms
      sit in a single block and whose intramolecular pairs are all
      within r_search
    When find_blocks_with_interactions has completed
    Then for the self-block (i-block == j-block) tile-pair that
      produces <= MAX_BITS_FOR_PAIRS hits, each unordered (a, b)
      intramolecular pair is written to single_pair_atoms exactly once
    # The self-block sparse-tile path is the natural place for
    # duplicate emission: bit b of lane a's i_hit_mask and bit a of
    # lane b's i_hit_mask both flag the same unordered pair. The
    # construction kernel must dedupe (e.g. by aid < jid) before the
    # atomicAdd that claims the single_pair_atoms slot.

  @rq-d3b31d79
  Scenario: Molecule straddling a cell boundary does not double-emit its bonded pair
    Given a two-molecule system arranged so that the C-C bond of one
      molecule straddles a cell boundary along the a-axis, placing
      C1 in one 32-block and C2 in the adjacent 32-block
    When find_blocks_with_interactions has completed
    Then the (C1, C2) pair appears exactly once across the union of
      interacting_atoms and single_pair_atoms

  @rq-efaec906
  Scenario: r_skin values that shift n_cells preserve pair-emission uniqueness
    Given a fixed particle state whose r_cut + r_skin lands at
      different floor((box_axis) / (r_cut + r_skin)) values across
      an r_skin sweep
    When each r_skin rebuild produces its packed / sparse output
    Then every rebuild lists each unordered (i, j) pair at most once
      across the union of the two outputs

  @rq-7646dd13
  Scenario: Probe rebuild grows interacting_tiles on a near-full build
    Given the phase has not yet been CUDA-graph captured
    And interacting_tiles_capacity = 100 with tile_pair_fill_threshold = 0.8
    And a system that produces 90 packed entries
    When the probe rebuild runs find_blocks_with_interactions
    Then the tiles_high_water bit of neighbor_status is set
    And no entry was dropped from the 90-entry build
    And the probe reallocates interacting_tiles to ceil(100 * 1.5)
    And the construction is re-run from find_blocks_with_interactions

  @rq-e7839cee
  Scenario: Probe rebuild grows single_pair_atoms on a near-full build
    Given the phase has not yet been CUDA-graph captured
    And single_pairs_capacity = 50 with tile_pair_fill_threshold = 0.8
    And a system that produces 45 single-pair entries
    When the probe rebuild runs find_blocks_with_interactions
    Then the single_pairs_high_water bit of neighbor_status is set
    And the probe reallocates single_pair_atoms to ceil(50 * 1.5) pairs
    And the construction is re-run from find_blocks_with_interactions

  @rq-8253d3c4
  Scenario: interaction_count and neighbor_status are reset at the start of every rebuild
    Given a prior rebuild left interaction_count = [50, 12]
      and neighbor_status with the tiles_high_water bit set
    When NeighborListState::rebuild begins a new rebuild
    Then interaction_count is reset to [0, 0]
      and neighbor_status is reset to 0 before
      find_blocks_with_interactions launches

  @rq-b8504fa1
  Scenario: A steady-state rebuild copies no count or status to the host
    Given a CUDA-graph-captured phase past its pre-capture probe rebuild
    And a batch boundary on which the displacement bit triggers a rebuild
    When NeighborListState::rebuild runs the construction pipeline
    Then no dtoh_sync_copy of interaction_count is issued
    And no dtoh_sync_copy reads find_blocks_with_interactions output other
      than the single combined neighbor_status word read by the batch loop

  @rq-e1258ceb
  Scenario: Downstream kernels launch over capacity and read the live count from the device
    Given a build that produced interaction_count[0] = C tile entries
      into a buffer of capacity interacting_tiles_capacity = K with C < K
    When histogram_entries_by_iblock, the i-block prefix scan, and
      scatter_entries_by_iblock run
    Then each is launched with a grid sized by K (and n_blocks), not by C
    And threads whose entry index is >= interaction_count[0] exit early
    And iblock_offset[n_blocks] equals interaction_count[0] without any host
      value being supplied

  @rq-88175d6f
  Scenario: High-water at a batch boundary grows geometrically and recaptures
    Given a CUDA-graph-captured phase under a densifying barostat
    And a batch-boundary build that sets the tiles_high_water bit without
      setting any overflow bit
    When the batch loop reads neighbor_status
    Then interacting_tiles_capacity grows to ceil(capacity * tile_pair_growth_factor)
    And a fresh rebuild populates the resized buffers
    And pre_step reports reallocated = true
    And the runner recaptures the phase graph

  @rq-a5bd8157
  Scenario: An overflow bit halts the run rather than dropping interactions
    Given a batch-boundary build whose true count exceeds capacity so that an
      entry would be dropped
    When the batch loop reads neighbor_status and observes a tiles_overflow bit
    Then pre_step returns Err(NeighborListError::PackedNeighborOverflow { buffer: "interacting_tiles" })
    And the run halts without presenting forces from the incomplete list as final

  @rq-3b18d9db
  Scenario: Probe rebuild grows exclusion-tile capacity on a near-full build
    Given the phase has not yet been CUDA-graph captured
    And exclusion_tiles_capacity = 100 with tile_pair_fill_threshold = 0.8
    And a topology whose current sort produces 90 exclusion tiles
    When the probe rebuild runs build_exclusion_tiles
    Then the exclusion_tiles_high_water bit of neighbor_status is set
    And no exclusion tile was dropped from the 90-tile build
    And the probe reallocates the exclusion-tile buffers to ceil(100 * 1.5)
    And the construction is re-run from build_exclusion_tiles

  @rq-51d6a942
  Scenario: An exclusion-tile overflow halts the run
    Given a build whose exclusion-tile count exceeds exclusion_tiles_capacity
      so a tile would be dropped
    When the batch loop reads neighbor_status and observes an exclusion_tiles_overflow bit
    Then pre_step returns Err(NeighborListError::PackedNeighborOverflow { buffer: "exclusion_tiles" })
    And the run halts without presenting forces from the incomplete tile set as final

  # --- Force kernel ---

  @rq-a786df3a
  Scenario: Packed-neighbour pass runs one block per i-block
    Given n_iblocks i-blocks with any fast-class pair-force slot active
    When the JIT-composed pair-force kernel launches
    Then the grid has n_iblocks blocks of BLOCK_SIZE = 256 threads each

  @rq-a6ecc598
  Scenario: Packed-neighbour pass skips the no-atom sentinel
    Given an entry pos with interacting_atoms[pos*32 + lane] == N for some lane
    When the packed-neighbour kernel processes entry pos
    Then no pair contribution is accumulated for that lane

  @rq-b49bfdff
  Scenario: Bulk and single-pair passes do no exclusion work
    Given a ForceField with at least one fast-class pair-force fragment
    And the JIT-composed packed-neighbour and single-pair kernel sources captured for inspection
    Then neither source contains a call to exclusion_scale or a read of atom_excl_offsets / atom_excl_partners
    And the packed-neighbour pass's inner loop dispatches to heddle_jit_eval_pair_sum with no per-pair scale multiply

  @rq-80c6a964
  Scenario: A fully-excluded pair contributes nothing via the exclusion-tile pass
    Given atoms a and b are a modified pair with every fragment scale 0, sharing exclusion tile t
    And their lane positions set bit (a_lane, b_lane) in exclusion_tile_masks[t]
    When the exclusion-tile pass processes tile t
    Then the (a, b) pair is evaluated with per-fragment scale 0
    And neither atom receives any force, energy, or virial contribution from the (a, b) pair

  @rq-8840662f
  Scenario: A non-modified pair inside an exclusion tile is computed at full strength
    Given atoms a and c share exclusion tile t but are NOT a modified pair
      (their mask bit is clear) and lie within the cutoff
    When the exclusion-tile pass processes tile t
    Then the (a, c) interaction is evaluated at full strength with no scale lookup
    And it is not also computed by the bulk or single-pair passes

  @rq-4aad39c8
  Scenario: A fractional 1-4 pair contributes a scaled force
    Given atoms a and d are a modified pair with a Lennard-Jones scale of 0.5, sharing exclusion tile t, within the cutoff
    When the exclusion-tile pass processes tile t
    Then each atom's Lennard-Jones force contribution from (a, d) is half the unscaled pair force
    And the cell-list total forces match an all-pairs run that applies the same per-fragment scales

  @rq-27add068
  Scenario: A per-fragment exclusion applies each fragment's scale independently
    Given atoms a and e are a modified pair with a Lennard-Jones scale of 0 and a Coulomb scale of 1, sharing exclusion tile t
    When the exclusion-tile pass processes tile t
    Then the (a, e) pair contributes no Lennard-Jones force
    And it contributes its full Coulomb force

  @rq-acfb3375
  Scenario: Construction routes a modified-pair block pair away from the bulk list
    Given block pair (bi, bj) contains at least one modified atom pair
    When NeighborListState::rebuild completes
    Then (bi, bj) appears in the exclusion-tile list
    And no interacting_atoms / single_pair_atoms entry pairs an atom of bi with an atom of bj

  @rq-ea4617e1
  Scenario: A fractional exclusion scale loads without error
    Given a topology whose exclusion for pair (i, j) has a Lennard-Jones scale of 0.5 and a Coulomb scale of 0.833
    When the simulation loads
    Then loading succeeds
    And (i, j) is recorded as a modified pair carrying those per-fragment scales

  @rq-10443d06
  Scenario: Two GPU runs build byte-identical exclusion tiles
    Given two simulations started from byte-identical initial state
    When both run a single neighbour-list rebuild
    Then interaction_count[2] is identical across the two runs
    And the exclusion tiles sorted by (bi, bj) carry identical bitmasks across the two runs

  # --- Exclusion-tile build (on device) ---

  @rq-e4a333d2
  Scenario: A steady-state rebuild builds exclusion tiles without a host round-trip
    Given a CUDA-graph-captured phase past its pre-capture probe rebuild
    And a batch boundary on which the displacement bit triggers a rebuild
    When NeighborListState::rebuild runs the construction pipeline including
      build_exclusion_tiles
    Then no dtoh_sync_copy of sorted_particle_ids is issued
    And the exclusion tiles, their bitmasks, the excl_jblock_offsets /
      excl_jblocks CSR, and interaction_count[2] are all written on the device
    And the only device-to-host transfer the rebuild performs is the single
      combined neighbor_status word read by the batch loop

  @rq-61d3e1a7
  Scenario: build_atom_slot inverts the sort permutation
    Given sorted_particle_ids holds a permutation of [0, N)
    When build_atom_slot runs
    Then atom_slot[sorted_particle_ids[s]] equals s for every slot s in [0, N)

  @rq-94d51613
  Scenario: Two modified pairs sharing a block pair merge into one exclusion tile
    Given modified pairs (a, b) and (c, d) whose atoms all map, under the
      current sort, into the same ordered block pair (bi, bj)
    When build_exclusion_tiles completes
    Then exactly one exclusion tile carries the block pair (bi, bj)
    And that tile's 1024-bit mask has the bit set for each of (a, b) and (c, d)
      at their respective lanes
    And interaction_count[2] counts that block pair exactly once

  @rq-3dd31c3d
  Scenario: A self-block exclusion tile sets the symmetric mask bit
    Given a modified pair (a, b) whose atoms map to the same block bi
      (bi == bj) at lanes (li, lj)
    When build_exclusion_tiles completes
    Then the tile for (bi, bi) sets bit lj of mask row li
    And it also sets bit li of mask row lj

  @rq-8631baa6
  Scenario: Exclusion tiles are emitted in ascending (bi, bj) order with an ascending CSR
    Given a topology producing exclusion tiles for several distinct block pairs
    When build_exclusion_tiles completes
    Then for every consecutive pair of tiles t, t+1 the key
      (exclusion_tile_iblocks[t], exclusion_tile_jblocks[t]) is strictly less
      than (exclusion_tile_iblocks[t+1], exclusion_tile_jblocks[t+1])
    And for every i-block bi,
      excl_jblocks[excl_jblock_offsets[bi] .. excl_jblock_offsets[bi+1]] lists
      the bj of that i-block's tiles in ascending order

  @rq-62cf6290
  Scenario: An empty modified-pair set produces no exclusion tiles
    Given a topology with no modified pairs (every fragment scale is 1)
    When NeighborListState::rebuild completes
    Then interaction_count[2] equals 0
    And the exclusion-tile force pass is not launched

  # --- Single-pair pass ---

  @rq-86cc8e5d
  Scenario: Single-pair pass is skipped when its count is zero
    Given interaction_count[1] == 0
    When the force-evaluation pipeline runs
    Then heddle_jit_composed_pair_force_single_f is not launched

  @rq-8d86bfa6
  Scenario: Single-pair pass accumulates Newton's-3rd pair forces
    Given single_pair_atoms[2k] = i and single_pair_atoms[2k+1] = j with the pair inside cutoff
    When the single-pair kernel processes pair index k
    Then the contribution to atom i is `+factor·(dx, dy, dz)` in fixed point
    And the contribution to atom j is `-factor·(dx, dy, dz)` in fixed point
    And both contributions are atomicAdded to the same fixed-point accumulator the packed-neighbour pass writes

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

  @rq-ea8640f5
  Scenario: Probe rebuild sizes capacity with headroom below the fill threshold
    Given a SPC water benchmark with 24,576 atoms
    When NeighborListState::new_cell_list constructs and the runner runs the
      pre-capture probe rebuild
    Then interacting_tiles_capacity is at least the count produced by that
      rebuild divided by tile_pair_fill_threshold
    And the tiles_high_water bit of neighbor_status is clear after the probe
    And interacting_tiles_capacity does not exceed that lower bound times
      tile_pair_growth_factor

  @rq-8d7e376d
  Scenario: Buffer footprint scales linearly with N, not quadratically
    Given two SPC water systems A with N_A atoms and B with N_B = 8 * N_A atoms
      at the same density
    When the pre-capture probe rebuild completes for each
    Then interacting_tiles_capacity for B is within a small constant factor of
      8 times interacting_tiles_capacity for A
    And no buffer is allocated with a length proportional to n_blocks squared

  @rq-36026b97
  Scenario: Capacity never preallocates the all-pairs upper bound
    Given a system with n_blocks atom-blocks
    When NeighborListState::new_cell_list constructs
    Then the seed length of interacting_tiles is far below
      n_blocks * n_blocks

  @rq-8b6d0c41
  Scenario: A rebuild grows the buffers when the interaction count rises
    Given a phase whose density increases under a barostat
    And a build whose entry count first crosses
      floor(interacting_tiles_capacity * tile_pair_fill_threshold)
    When the batch loop reads the tiles_high_water bit of neighbor_status
    Then interacting_tiles is reallocated to ceil(interacting_tiles_capacity * tile_pair_growth_factor)
    And a fresh rebuild populates the resized buffer
    And the rebuild's pre_step outcome reports reallocated = true

  @rq-25f8dd1d
  Scenario: Seed capacity is clamped to the all-pairs reference for tiny systems
    Given a system with n_blocks = 4 atom-blocks
    When NeighborListState::new_cell_list constructs
    Then default_interacting_tiles_capacity(4) equals 16 (= 4 * 4)
    And the seed is not the 128 * n_blocks density heuristic

  # --- Single-periodic-copy fast path ---

  @rq-7369ded0
  Scenario: Per-block SPC predicate is true when every axis margin exceeds MAX_CUTOFF
    Given an orthorhombic box with lattice lengths (L_x, L_y, L_z) = (100, 100, 100)
    And MAX_CUTOFF = 10
    And an i-block whose bbox half-extents (block_bbox[b]) are (5, 5, 5)
    When the packed-neighbour kernel processes i-block b
    Then the SPC predicate evaluates to true
    And the warp takes the centre-wrap fast path

  @rq-527d3e7c
  Scenario: Per-block SPC predicate is false when any axis margin drops below MAX_CUTOFF
    Given an orthorhombic box with lattice lengths (L_x, L_y, L_z) = (100, 100, 100)
    And MAX_CUTOFF = 10
    And an i-block whose bbox half-extents (block_bbox[b]) are (5, 45, 5)
      so 0.5*L_y - block_bbox.y = 5 < MAX_CUTOFF
    When the packed-neighbour kernel processes i-block b
    Then the SPC predicate evaluates to false
    And the warp takes the per-pair min-image path

  @rq-fb310fbe
  Scenario: Per-block SPC predicate is boundary-true at exactly MAX_CUTOFF margin
    Given an orthorhombic box with lattice lengths (L_x, L_y, L_z) = (100, 100, 100)
    And MAX_CUTOFF = 10
    And an i-block whose bbox half-extents are (40, 40, 40)
      so 0.5*L_d - block_bbox.d = 10 = MAX_CUTOFF for every axis
    When the packed-neighbour kernel processes i-block b
    Then the SPC predicate evaluates to true

  @rq-432c52f5
  Scenario: Triclinic boxes are SPC-ineligible regardless of bbox extent
    Given a SimulationBox whose lattice has any of xy, xz, yz non-zero
    When the packed-neighbour kernel processes any i-block
    Then the SPC predicate evaluates to false
    And the warp takes the per-pair min-image path

  @rq-fe6b054a
  Scenario: SPC predicate is uniform across the warp
    Given the packed-neighbour kernel processes any i-block
    When all 32 lanes of the warp evaluate the SPC predicate
    Then every lane observes the same boolean value
    And the kernel branches once warp-wide without per-lane divergence

  @rq-df787c57
  Scenario: SPC fast path wraps pi and pj against the i-block centre
    Given the SPC predicate is true for i-block b with centre c = block_centre[b]
    When the warp loads pi for any lane from tile_sorted_posq
    Then pi is shifted in-register to the periodic image satisfying |pi - c|_axis <= L_axis / 2
    And every per-entry j-atom's pj loaded from posq is shifted to satisfy |pj - c|_axis <= L_axis / 2
    And no further shift is applied to pi or pj during the 32-iteration inner loop

  @rq-e39a87c1
  Scenario: SPC fast path inner loop skips triclinic_min_image
    Given the SPC predicate is true for the i-block currently being processed
    When the inner loop iteration evaluates dx = pi - pj
    Then heddle_jit_triclinic_min_image is not invoked for this iteration
    And r2 = dx*dx + dy*dy + dz*dz is computed from dx, dy, dz directly

  @rq-e1717060
  Scenario: Min-image branch is taken when the predicate is false
    Given the SPC predicate is false for the i-block currently being processed
    When the inner loop iteration evaluates dx = pi - pj
    Then heddle_jit_triclinic_min_image is invoked once for this iteration before r2 is computed
    And neither pi nor pj is centre-wrapped

  @rq-05b5b488
  Scenario: A box where every i-block qualifies matches the all-min-image run bit-for-bit
    Given a simulation in which the SPC predicate is true for every i-block of every step
    And a comparator run on the same hardware that disables the SPC branch and always takes min-image
    When both runs perform one ForceField::step(Fast)
    Then ParticleBuffers.forces_x, forces_y, forces_z compare byte-identical across the two runs after finalize_fast_class_forces

  @rq-c469b2e8
  Scenario: NPT box transition flips SPC eligibility without graph re-capture
    Given a CUDA-graph-captured phase in which every i-block had SPC true at capture time
    And a subsequent step in which the box shrinks so 0.5*L_d - block_bbox.d < MAX_CUTOFF for some block on some axis
    When the captured graph replays the same packed-neighbour kernel call
    Then the kernel evaluates the SPC predicate from the current lattice and block_bbox values at every i-block
    And every affected i-block takes the per-pair min-image path on this replay
    And no graph invalidation or re-capture is triggered

  @rq-9e08c8b0
  Scenario: Single-pair pass unaffected by SPC
    Given any simulation step
    When the single-pair pass runs
    Then it invokes heddle_jit_triclinic_min_image for every pair it evaluates
    And it does not read block_centre or block_bbox

  # --- Single-periodic-copy fast path: exclusion-tile pass ---

  @rq-e6620f2c
  Scenario: Exclusion-tile pass takes the SPC fast path when its i-block qualifies
    Given an orthorhombic box in which the SPC predicate on block_bbox[bi] is true
    And an exclusion tile t with i-block bi and j-block bj
    When the exclusion-tile kernel processes tile t
    Then every lane wraps its bi atom and its bj atom once against block_centre[bi]
    And the inner loop computes dx = pi - pj without calling heddle_jit_triclinic_min_image

  @rq-dc44a114
  Scenario: Exclusion-tile pass takes the min-image path when its i-block does not qualify
    Given an orthorhombic box in which the SPC predicate on block_bbox[bi] is false
      for exclusion tile t's i-block bi
    When the exclusion-tile kernel processes tile t
    Then the inner loop calls heddle_jit_triclinic_min_image once per lane pair
    And neither pi nor pj is centre-wrapped

  @rq-1e9bb643
  Scenario: Exclusion-tile SPC predicate uses the tile's i-block and is warp-uniform
    Given the exclusion-tile kernel processes tile t with i-block bi
    When all 32 lanes of the tile's warp evaluate the SPC predicate
    Then every lane reads block_bbox[bi] and observes the same boolean value
    And the kernel branches once warp-wide without per-lane divergence

  @rq-ba3ad34b
  Scenario: Triclinic box forces the exclusion-tile pass onto the min-image path
    Given a SimulationBox whose lattice has any of xy, xz, yz non-zero
    When the exclusion-tile kernel processes any tile
    Then the SPC predicate evaluates to false
    And the inner loop takes the per-pair min-image path

  @rq-ea68a7aa
  Scenario: Exclusion-tile SPC path is bit-identical to the min-image path
    Given a simulation whose exclusion tiles all have an SPC-eligible i-block
    And a comparator run on the same hardware that disables the exclusion-tile SPC branch
      and always takes min-image
    When both runs perform one ForceField::step(Fast)
    Then ParticleBuffers.forces_x, forces_y, forces_z compare byte-identical across
      the two runs after finalize_fast_class_forces

  @rq-b72ac355
  Scenario: A cross-block exclusion tile with an out-of-cutoff pair contributes nothing under SPC
    Given an SPC-eligible exclusion tile t whose i-block bi and j-block bj contain a
      lane pair whose true min-image separation exceeds MAX_CUTOFF
    When the exclusion-tile kernel processes tile t on the SPC path
    Then that lane pair's centre-wrapped separation is at least its min-image separation
    And the cutoff mask zeroes its contribution, matching the min-image path

  # --- PairSnapshot: canonical pair enumeration ---

  @rq-88b287d5
  Scenario: pair_snapshot returns all packed and sparse pairs, deduplicated, as (min, max)
    Given a completed NeighborListState::rebuild in cell-list mode with
      interacting_tiles_count > 0 and single_pairs_count > 0
    When NeighborListState::pair_snapshot(&self, device) is called
    And the returned PairSnapshot is iterated
    Then every yielded (i, j) has i < j and both indices are less than n_atoms
    And every unordered pair appears in the yielded set at most once
    And the yielded set equals the union of the pairs decoded from
      (interacting_tiles, sorted_interacting_atoms) via the diagonal
      shuffle and the pairs decoded from single_pair_atoms, with the
      sentinel value n_atoms filtered out

  @rq-26204244
  Scenario: Snapshot's device readback is bounded to the counts and the four owned buffers
    Given a NeighborListState in cell-list mode with a completed rebuild
    When NeighborListState::pair_snapshot is called
    Then exactly five host-facing device-to-host copies occur — the
      interaction_count word, the truncated interacting_tiles, the
      truncated sorted_interacting_atoms, the truncated single_pair_atoms,
      and the sorted-particle-ID view
    And no read of neighbor_status is issued

  @rq-edbd7063
  Scenario: Iterating the same snapshot twice yields the same pair sequence
    Given a PairSnapshot returned by NeighborListState::pair_snapshot
    When .iter() is called twice and each iterator is fully consumed
    Then both iterators yield the same sequence of (u32, u32) values in the same order

  @rq-751f726a
  Scenario: len equals the number of pairs the iterator yields
    Given a PairSnapshot returned by NeighborListState::pair_snapshot
    Then snapshot.len() equals snapshot.iter().count()

  @rq-c9a8df0f
  Scenario: Empty neighbour list produces an empty snapshot
    Given a NeighborListState in cell-list mode with interacting_tiles_count = 0 and single_pairs_count = 0
    When NeighborListState::pair_snapshot is called
    Then snapshot.len() equals 0
    And snapshot.iter().next() returns None

  @rq-372f4581
  Scenario: Packed-only neighbour list is enumerable
    Given a NeighborListState in cell-list mode whose only pairs live in the packed dense entries
      (interacting_tiles_count > 0, single_pairs_count = 0)
    When the snapshot is iterated
    Then the yielded set matches the diagonal-shuffle decoding of the packed entries
      with sentinel and (i == j) values filtered

  @rq-5f067d09
  Scenario: Sparse-only neighbour list is enumerable
    Given a NeighborListState in cell-list mode whose only pairs live in single_pair_atoms
      (interacting_tiles_count = 0, single_pairs_count > 0)
    When the snapshot is iterated
    Then the yielded set equals the (min, max) canonicalisation of the pairs
      encoded in single_pair_atoms

  @rq-f2641eaa
  Scenario: A pair reachable from both representations is yielded exactly once
    Given a NeighborListState in cell-list mode in which a canonical pair (a, b) appears in
      the packed dense entries and also in single_pair_atoms
    When the snapshot is iterated
    Then (a, b) is yielded exactly once

  # --- Out of scope ---

  @rq-92b63da0
  Scenario: f64 build refuses to instantiate the packed-neighbour kernel
    Given the project is compiled with feature "f64"
    When ForceField::new is called for a fast-class pair-force pipeline
    Then construction returns an error indicating that the packed-
      neighbour kernel does not yet support f64
```
