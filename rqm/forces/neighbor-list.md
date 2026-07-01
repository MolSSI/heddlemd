# Feature: Cell-List Spatial Pre-Step <!-- rq-693ce6fa -->

The `ForceField` owns a single `NeighborListState` that provides the
spatial pre-step consumed by the fast-class pair-force pipeline (the
packed-neighbour architecture described in
`packed-neighbour-pair-force.md`) and by the SPME reciprocal-space
slot's atom binning (see `spme.md`). Its job is to produce a
deterministic spatial sort of the atoms — `sorted_particle_ids` and
per-cell `cell_offsets` — plus the displacement-driven rebuild
trigger that controls when the packed-neighbour list is reconstructed.

The packed-neighbour pair-force kernel reads `sorted_particle_ids` to
partition atoms into 32-atom blocks and to translate block-position
indices into original atom IDs; it does not consume any other output
of this pre-step. The per-particle padded neighbour list and the
`max_neighbors` configuration knob are not part of this architecture.

The state exists in one of three construction modes determined by the
parsed `NeighborListConfig` and by which slots are active:

- **`CellList`** — the spatial-hash sort described in this file. Used
  by the packed-neighbour pair-force pipeline whenever
  `[neighbor_list].mode = "cell-list"` and at least one fast-class
  pair-force slot is active. Built lazily on the first step and
  rebuilt on demand when an atom's reference displacement exceeds
  `r_skin / 2`. The cell layout (number of cells per lattice
  direction, total cell count) is cached at construction from the
  simulation box's lattice parameters and refreshed whenever the
  box's `generation` counter changes; a refresh forces a rebuild only
  when the refreshed `n_cells_total` differs from the cached value
  (see *Box Generation Tracking* below). Plain barostat-driven
  generation ticks that leave `n_cells_total` unchanged let the
  displacement check govern rebuild timing.
- **`CellListOnly`** — the cell-list bin output (sorted IDs +
  per-cell offsets) only, without driving the displacement-check
  rebuild trigger. Used by the SPME reciprocal-space slot when SPME
  is active without any fast-class pair-force slot consuming the
  cell list. Rebuilt every step unconditionally.
- **`Trivial`** — `sorted_particle_ids` is the identity permutation;
  no cell list, no displacement check, no rebuild. Used when the
  config selects `[neighbor_list].mode = "all-pairs"`. The
  packed-neighbour pair-force pipeline enumerates every interacting
  block pair in this mode.

When no slot in the `ForceField` reports a `max_cutoff` and SPME is
inactive (a bonded-only or zero-slot configuration), no
`NeighborListState` is built and no spatial-sort or displacement-check
kernel runs.

This file specifies the three modes, the cell-list construction
pipeline that produces `sorted_particle_ids` and `cell_offsets`, the
per-step displacement check and rebuild policy that governs
`CellList`, and the `NeighborListState` API the framework drives.
The packed-neighbour list itself — the entries the force kernel
consumes — is specified by `packed-neighbour-pair-force.md`.

## Cell Layout <!-- rq-dfad7218 -->

Cells partition the parallelepiped box into a 3D grid of smaller
parallelepipeds aligned with the lattice vectors. Per lattice direction
`d ∈ {a, b, c}`:

```
n_cells_d            = floor(w_d / (r_cut + r_skin))
fractional_cell_size = 1 / n_cells_d
```

where `w_d` is the perpendicular width along lattice direction `d` (see
`simulation-box.md`). The cell thickness perpendicular to the opposite
face is `w_d / n_cells_d`, which is always `>= r_cut + r_skin` because
`n_cells_d` is rounded down. The 27-cell PBC search visits every adjacent
cell exactly once iff `n_cells_d >= 3` along every direction. Configurations
that violate this on any direction are rejected at neighbor-list
construction time and again on every box-generation refresh (see *Box
Generation Tracking*).

`n_cells_total = n_cells_a * n_cells_b * n_cells_c`. The cell index of a
particle at Cartesian position `(x, y, z)` is computed in fractional
coordinates:

```
s = sim_box.fractional_coords(sim_box.wrap_position((x, y, z)))
cell_d = floor((s_d + 0.5) * n_cells_d)   for d in {a, b, c}
```

The total row-major cell index is `cell_a * n_cells_b * n_cells_c +
cell_b * n_cells_c + cell_c`. Per-direction indices are clamped to
`[0, n_cells_d - 1]` to handle the `s_d = +0.5` boundary case (mapped to
`cell_d = n_cells_d - 1` rather than `n_cells_d`).

For an orthorhombic box (`xy = xz = yz = 0`), `w_a = lx`, `w_b = ly`,
`w_c = lz`, and `s_d` reduces to per-axis `pos_d / L_d` so the cell-index
formula collapses to the per-axis formula `floor((wrapped + L/2) /
cell_size)` with `cell_size_d = L_d / n_cells_d`.

## Cell List Construction <!-- rq-a060036e -->

A cell-list rebuild is performed entirely on the device. The pipeline
operates on five device buffers — three persistent outputs and two
persistent scratch buffers — without round-tripping through the host:

- **`cell_indices: CudaSlice<u32>` of length `N`** (scratch). Populated
  by the cell-indexing stage; entry `i` is atom `i`'s cell index.
- **`cell_counts: CudaSlice<u32>` of length `n_cells_total`** (scratch).
  Populated by the histogram stage; entry `c` is the number of atoms in
  cell `c`.
- **`cell_offsets: CudaSlice<u32>` of length `n_cells_total + 1`**
  (output). Populated by the exclusive-scan stage from `cell_counts`.
  `cell_offsets[c]` is the index in `sorted_particle_ids` where cell
  `c`'s atoms begin; `cell_offsets[c+1] - cell_offsets[c]` is the cell's
  occupancy; `cell_offsets[n_cells_total]` is `particle_count`.
- **`write_cursors: CudaSlice<u32>` of length `n_cells_total`** (scratch).
  Reset to zero at the start of each rebuild; the scatter stage
  `atomicAdd`s into it to claim per-cell write positions.
- **`sorted_particle_ids: CudaSlice<u32>` of length `N`** (output). The
  particle IDs sorted lexicographically by `(cell_index, particle_id)`;
  the secondary sort key gives a deterministic in-cell ordering.

The pipeline runs as a sequence of device kernels:

1. **Cell-index + histogram.** Each thread handles one atom: computes
   the atom's cell index (using the wrap formula in *Cell Layout*),
   writes it to `cell_indices[i]`, and performs an `atomicAdd` on
   `cell_counts[cell_indices[i]]`. Integer `atomicAdd` is deterministic
   in value because addition is associative; the final `cell_counts`
   array is identical across runs even though the order of atomic
   updates is not.
2. **Exclusive prefix scan.** A recursive multi-level device scan writes
   the exclusive prefix sum of `cell_counts` into `cell_offsets`. Each
   level performs a per-block local exclusive scan and emits one
   inclusive total per block; the array of per-block totals is scanned
   by the same procedure applied recursively, and the scanned totals are
   added back into the level below. The recursion bottoms out at a level
   whose input fits in a single block. The output invariant is
   `cell_offsets[c] = sum_{c' < c} cell_counts[c']` for `c` in
   `[0, n_cells_total]`, with `cell_offsets[n_cells_total] =
   particle_count`. The scan imposes no cap on `n_cells_total`; the only
   ceiling is the device's `u32` cell addressing (see
   `NeighborListError::TooManyCells`). Integer addition is associative,
   so the scan result is bit-exact run-to-run regardless of block
   scheduling.
3. **Scatter.** `write_cursors` is reset to zero. Each thread handles
   one atom and computes
   `slot = atomicAdd(&write_cursors[cell_indices[i]], 1)`, then writes
   `sorted_particle_ids[cell_offsets[cell_indices[i]] + slot] = i`. The
   within-cell ordering at this stage is non-deterministic (depends on
   the order in which threads execute their atomic).
4. **Per-cell sort.** Each thread handles one cell and insertion-sorts
   the `sorted_particle_ids` slice
   `[cell_offsets[c] .. cell_offsets[c+1]]` ascending by particle index.
   This canonicalises the non-deterministic scatter order: after this
   stage `sorted_particle_ids` is identical across runs given identical
   inputs on the same GPU, and matches the canonical
   `(cell_index, particle_id)` lex order.

No host download or upload of particle data occurs in the rebuild. The
host work per rebuild is the three per-atom / per-cell kernel launches
(cell-index + histogram, scatter, per-cell sort), the prefix scan's
`O(log(n_cells_total))` launches, and one device memset (zeroing
`write_cursors`).

## Packed-Neighbour List Construction <!-- rq-33aa3e1d -->

The packed neighbour list consumed by the fast-class pair-force
kernel is constructed by the pipeline specified in
`packed-neighbour-pair-force.md` (the `scatter_positions_to_tile_order`,
`compute_block_bbox`, volume-sort, and
`find_blocks_with_interactions` kernels). That pipeline is invoked
from `NeighborListState::rebuild` after the cell-list sort completes
and before `copy_positions_into_reference` records the
displacement-check baseline. The output is `interacting_tiles`,
`interacting_atoms`, `single_pairs`, and `interaction_count`, all
described in detail in that file.

`NeighborListState` does not allocate a per-particle padded neighbour
list and does not run a per-particle neighbour-build kernel. The
`max_neighbors` configuration knob is not present.

## Displacement Check <!-- rq-1f38d78a -->

`CellListData` carries a one-element device buffer
`neighbor_status: CudaSlice<u32>` and three reference-position
buffers `reference_positions_{x,y,z}: CudaSlice<Real>` of length `N`.
The reference positions are written immediately after every rebuild,
recording the positions used during that rebuild. `neighbor_status`
is a packed status word whose bits are reset to `0` at the start of
every rebuild and are otherwise set only by the displacement-check
kernel (bit 0) and the packed-neighbour construction kernel (bits
1–4, see `packed-neighbour-pair-force.md` *Capacity*):

| Bit | Name | Writer |
|---|---|---|
| 0 | `displacement_tripped` | displacement-check kernel, every step |
| 1 | `tiles_high_water` | construction kernel |
| 2 | `single_pairs_high_water` | construction kernel |
| 3 | `tiles_overflow` | construction kernel |
| 4 | `single_pairs_overflow` | construction kernel |

The displacement-check kernel `neighbor_displacement_check_flag(posq,
reference_x, reference_y, reference_z, lattice, threshold_sq,
neighbor_status, n)` runs once per physical timestep as the last
device-visible action in the per-step force-evaluation pipeline. One
thread per atom:

1. Computes `disp² = dx² + dy² + dz²` from
   `(posq[i].xyz − reference_*[i])` with the minimum-image wrap
   applied to the displacement vector (so an atom that crossed a PBC
   boundary still reports a small `disp²` rather than ≈ `L²`).
2. When `disp² > threshold_sq`, issues `atomicOr(neighbor_status, 1u)`,
   setting bit 0 only.

`threshold_sq` is `(r_skin / 2)²`, a per-`NeighborListState` scalar set
once at construction time and re-derived only if `r_skin` changes (it
does not change across the lifetime of a phase). The kernel is
launched on the device's default stream from
`ForceField::step` and `ForceField::step_no_neighbor_check` so the
launch sits in the same default-stream sequence as the force kernels
and is recorded into the captured graph when capture is active.

`NeighborListState::pre_step(sim_box, buffers, timings)` consumes the
word as follows:

1. If the cell-layout cache reports a box-generation mismatch (see
   *Box Generation Tracking*) and the refresh changes `n_cells_total`,
   the status read is skipped and a rebuild is forced.
2. Otherwise, the host issues `dtoh_sync_copy(&neighbor_status)` —
   a single-word blocking download on the default stream that surfaces
   the displacement, high-water, and overflow bits together. The
   download ordering against earlier kernel writes is guaranteed by the
   default-stream sequence. This one read is the only device-to-host
   transfer the neighbour path performs per batch; the rebuild itself
   copies nothing.
3. Decode the word:
   - An overflow bit (`tiles_overflow` or `single_pairs_overflow`)
     means the most recent build dropped within-`r_search` entries.
     `pre_step` returns
     `Err(NeighborListError::PackedNeighborOverflow { buffer })` and the
     run halts (see `packed-neighbour-pair-force.md` *Capacity*).
   - A high-water bit (`tiles_high_water` or `single_pairs_high_water`)
     requests a geometric capacity grow and forces a rebuild this call,
     even when bit 0 is clear. The build that set the bit was complete,
     so the current list stays correct until the grow-and-rebuild runs.
   - Bit 0 (`displacement_tripped`) alone forces a rebuild with no
     capacity change.
4. If a rebuild is required, the host zeros `neighbor_status` via
   `device.memset_zeros(&mut cl.neighbor_status)` **before** the
   cell-list and construction kernels run, grows any high-water-flagged
   capacity to `ceil(capacity · tile_pair_growth_factor)` and
   reallocates it, runs the rebuild (cell-list pipeline plus
   packed-neighbour construction), and writes the current positions
   into `reference_positions_*`. When the rebuild reallocates a
   packed-neighbour buffer, `pre_step` returns
   `PreStepOutcome.reallocated = true` so the batched graph-replay loop
   re-captures the phase graph (see `cuda-graphs.md` *Neighbor-List
   Pre-Step Decomposition*).

Inside a `pre_step` that does not rebuild, `neighbor_status` is left in
whatever state the kernels wrote it (every bit is sticky across
multiple captured steps; only the start of the next rebuild clears the
word). The high-water and overflow bits set by the most recent
construction therefore persist until the next batch boundary reads
them, which is what lets a near-full build observed at one boundary
drive a proactive grow at the next.

Because bit 0 is written every step but cleared only on rebuild, the
value the host reads at the end of a batched replay reflects "any
step in the batch tripped `r_skin / 2`" rather than "the current
state at the end of the last step". This is the conservative
direction: a rebuild fires whenever at least one step inside the
window saw an over-threshold particle, even if subsequent particle
motion would have brought the maximum back under the threshold.

When a phase runs under CUDA graph mode (`cuda-graphs.md`), the
runner moves the `NeighborListState::pre_step` call out of
`ForceField::step` and into the batched-replay outer loop. The
displacement-check *kernel* still runs every step (it is part of the
captured graph), but the host's per-batch consumption of the flag
runs once per `graph_batch_size` physical steps rather than every
step. The skin-distance contract holds as long as
`graph_batch_size * max_step_displacement < r_skin / 2`. See
`cuda-graphs.md`'s *Skin-distance contract under batched replay* for
the analysis.

## Box Generation Tracking <!-- rq-282af621 -->

In `CellList` mode the cell layout (`n_cells`, `n_cells_total`, and the
cached lattice parameters used by the spatial-hash and build kernels)
depends on the simulation box's lattice and is therefore cached. The
state records the box's `generation()` value at the moment the cache was
populated; this is stored as the field `cached_generation: u64`. The state
also stores the `r_cut` value used to derive that cache so the cache can be
refreshed without re-querying every consumer.

Every call to `NeighborListState::pre_step` receives the runner's current
`&SimulationBox` and compares `sim_box.generation()` against
`cached_generation`. On a match no cache work is done. On a mismatch the
state:

1. Recomputes `n_cells_d = floor(w_d / (r_cut + r_skin))` per lattice
   direction `d ∈ {a, b, c}` from the current box's perpendicular widths
   `w_d` (see `simulation-box.md`).
2. Re-validates that `n_cells_d >= 3` on every direction. If any direction
   fails, returns `Err(NeighborListError::BoxTooSmallForCells { direction,
   width, required })` without mutating any cached field; the caller can
   rerun the previous step's force evaluation with the prior box or abort.
3. Recomputes the scalar `n_cells_total = n_cells_a * n_cells_b * n_cells_c`.
4. Reallocates the device buffers sized by `n_cells_total` (`cell_offsets`
   to `n_cells_total + 1`; `cell_counts` and `write_cursors` to
   `n_cells_total`) if `n_cells_total` differs from the previous value,
   and rebuilds the prefix scan's block-totals stack to match the new
   `n_cells_total`. Other device buffers (`reference_positions_*`,
   `neighbor_status`, `cell_indices`) are sized by `particle_count` or
   are scalar and are not reallocated.
5. Stores the new `n_cells`, `n_cells_total`, and the cached lattice
   parameters used by the spatial-hash and build kernels, and replaces
   `cached_generation` with `sim_box.generation()`.
6. Sets `needs_rebuild = true` **only when `n_cells_total` differed
   from the prior cached value**. In that case the existing cell-list
   contents are stale (they were sized and indexed for the previous
   `n_cells_total`) and the rebuild that follows uses the new cell
   layout; the displacement check is skipped on this `pre_step` because
   the next rebuild is already required. When `n_cells_total` is
   unchanged across the refresh, `needs_rebuild` is **not** set by the
   refresh; the existing cell-list contents remain valid (cell indices
   in fractional space are preserved under uniform box scaling that
   keeps the same `n_cells`) and the displacement check governs whether
   a rebuild fires on this `pre_step` exactly as in the no-generation-
   tick case.

`r_search_sq` (`(r_cut + r_skin)²`) does not depend on the box and is left
in place across refreshes.

This policy is what makes NPT runs efficient: a barostat that ticks the
box generation every step but leaves `n_cells_total` unchanged (the
typical case, since per-step volume changes are <0.1 %) consults the
displacement check on every `pre_step`, and the neighbor list is
rebuilt at roughly the same rate as in NVT rather than every step. The
displacement check operates in physical coordinates and therefore
captures barostat-induced atom motion at the box edge — the worst-case
contributor — so correctness is preserved without an additional
box-scale-factor guard.

In `Trivial` mode the cell layout is not used; `pre_step` ignores the box's
generation and does no per-step work.

## Rebuild Policy <!-- rq-6e11554f -->

The runner holds one host-side `bool` flag `needs_rebuild`. Its initial
value is `true` so the warm-up force evaluation triggers the first
rebuild.

Per `pre_step` call (every timestep in non-graph mode; once per
`graph_batch_size` physical steps in graph mode):

1. If `sim_box.generation() != cached_generation`, refresh the cell-layout
   cache (see *Box Generation Tracking*). The refresh sets
   `needs_rebuild = true` and bypasses the displacement-flag download
   for this call only when the refreshed `n_cells_total` differs from
   the prior cached value. When `n_cells_total` is unchanged the
   refresh updates the cached lattice parameters and
   `cached_generation` only and falls through to the flag-download step
   below. The refresh may return `BoxTooSmallForCells`, in which case
   `pre_step` aborts and the error propagates.
2. Download `neighbor_status` (a single `u32`) via `dtoh_sync_copy` —
   the only device-to-host transfer the neighbour path performs this
   call. Decode the bits:
   - If `tiles_overflow` or `single_pairs_overflow` is set, return
     `Err(NeighborListError::PackedNeighborOverflow { buffer })`; the
     run halts (a build dropped within-`r_search` entries — see
     `packed-neighbour-pair-force.md` *Capacity*).
   - If `tiles_high_water` or `single_pairs_high_water` is set, mark
     the matching capacity for a geometric grow and set
     `needs_rebuild = true`.
   - If bit 0 (`displacement_tripped`) is set, set
     `needs_rebuild = true`.
   The download is skipped when `needs_rebuild` is already true.
3. If `needs_rebuild`:
   a. Zero `neighbor_status` via `device.memset_zeros(&mut
      cl.neighbor_status)` so the construction kernel's high-water and
      overflow bits for this build start clean.
   b. Grow and reallocate any capacity marked in step 2 to
      `ceil(capacity · tile_pair_growth_factor)`; record that a buffer
      was reallocated.
   c. Run the on-device cell-list pipeline (see *Cell List Construction*)
      to repopulate `cell_indices`, `cell_counts`, `cell_offsets`, and
      `sorted_particle_ids` from the current positions.
   d. Run the packed-neighbour construction (see
      `packed-neighbour-pair-force.md`), which leaves the live counts
      and the high-water/overflow bits device-resident for the next
      batch boundary to observe. No count is copied to the host.
   e. Copy current positions into the reference-position buffers.
   f. Set `needs_rebuild = false`; report `reallocated = true` through
      `PreStepOutcome` if step 3b reallocated a buffer.
4. Run downstream contribution kernels (see `framework.md`), which read
   the neighbor list.

The probe rebuild — the warm-up rebuild the runner performs before
CUDA-graph capture — is the one exception to the dtoh-free steady
state: it reads `neighbor_status` synchronously after construction and
repeats step 3b–3d, growing geometrically until neither a high-water
nor an overflow bit is set, so it sizes the initial capacities with
headroom. It runs once per phase and is outside the captured replay
loop.

The displacement-check *kernel* itself runs every physical timestep
(it is queued by `ForceField::step` and `ForceField::step_no_neighbor_check`
as the last device-visible action of the step). Bit 0 is sticky across
steps until the next rebuild clears the word, so the value read in
step 2 above reflects "any timestep since the last rebuild tripped
`r_skin / 2`".

In `CellListOnly` mode, `pre_step` skips the displacement check
entirely and runs the cell-list pipeline (cell indexing, prefix scan,
scatter, in-cell sort) on every call, regardless of how far particles
have moved. The neighbor-list-build kernel does not run in
`CellListOnly` mode.

## Configuration <!-- rq-267941a2 -->

Selected from the config's optional `[neighbor_list]` table; see
`io/config-schema.md` for the schema. Summary:

- `mode: String` — `"cell-list"` (default) or `"all-pairs"`.
- For `mode = "cell-list"`:
  - `r_skin: f64` — optional, defaults to `0.3 * max_cutoff` where
    `max_cutoff` is the largest cutoff reported by any
    neighbor-list-consuming potential. Strictly positive.
- For both modes (packed-neighbour list sizing — see
  `packed-neighbour-pair-force.md`):
  - `tile_pair_growth_factor: f64` — optional, defaults to `1.5`.
    Strictly greater than `1.0`.
  - `interacting_tiles_initial_capacity: u32` — optional; defaults
    derived from `n_blocks`.
  - `single_pairs_initial_capacity: u32` — optional; defaults
    derived from `n_blocks`.

Validation at config load:

- `r_skin > 0` and finite.
- `tile_pair_growth_factor > 1.0` and finite.
- `max_neighbors` is not a valid field. If the config text contains
  `max_neighbors`, the loader reports an explicit error explaining
  that the packed-neighbour pair-force architecture (see
  `packed-neighbour-pair-force.md`) sizes the neighbour-list buffers
  to the actual interaction count plus a growth margin, with no
  user-supplied per-atom cap.
- For every cutoff `c` reported by a consuming potential, the box's
  minimum perpendicular width satisfies `min_perpendicular_width >= 3 *
  (c + r_skin)` (see `simulation-box.md`). For an orthorhombic box this
  reduces to `min(lx, ly, lz) >= 3 * (c + r_skin)`. The init file holds
  the box; validation therefore happens after the init file is loaded
  and the effective `r_skin` and `max_cutoff` are known.
- In `mode = "all-pairs"`, `r_skin` is rejected; the packed-neighbour
  sizing fields are accepted.

The maximum cutoff used to size the cell-list search radius is the
largest `max_cutoff` reported by any consuming potential in the
`ForceField` slot list (see `framework.md`). With one or more
consumers, the cell-list search radius is `max_cutoff + r_skin` in
cell-list mode. The trivial mode does not use a search radius.

## Empty State <!-- rq-5cbab27f -->

When `particle_count == 0`:

- Cell-list construction produces empty `sorted_particle_ids` and
  `cell_offsets` of length `n_cells_total + 1` filled with zeros.
- The packed-neighbour construction pipeline does not launch.
- The displacement-check kernel does not launch.
- `needs_rebuild` stays `true` forever but no rebuild work happens.
- Trivial construction produces an empty `sorted_particle_ids` and
  empty packed-neighbour buffers.

When `particle_count == 1`:

- Cell-list construction works trivially.
- The packed-neighbour construction pipeline runs and produces zero
  interacting tiles and zero single pairs (a single atom has no
  partners under any cutoff).
- Trivial construction produces a single-element
  `sorted_particle_ids` containing `[0]`. No pair-force kernel work
  runs because no partners exist; the force kernel's diagonal
  exclusion-tile path covers the self-self case as a skip.

When the `ForceField` has zero pair-force consumers and SPME is
inactive, no `NeighborListState` is built; the framework's
`Option<NeighborListState>` is `None` and no spatial-sort or
displacement-check kernel runs over the lifetime of the run.

## Feature API <!-- rq-3e744fed -->

### Types <!-- rq-ad7eb40f -->

- `NeighborListConfig` — value of the parsed `[neighbor_list]` table. <!-- rq-060b1fab -->
  Variants:
  - `AllPairs`
  - `CellList { r_skin: f64 }`

- `PreStepOutcome` — value returned by `NeighborListState::pre_step`. <!-- rq-b22e871e -->
  Fields:
  - `rebuilt: bool` — `true` when the call ran a rebuild.
  - `reallocated: bool` — `true` when that rebuild grew (reallocated)
    a packed-neighbour buffer (`interacting_tiles`,
    `interacting_atoms`, or `single_pair_atoms`). The batched
    graph-replay loop re-captures the phase graph when this is `true`
    (see `cuda-graphs.md` *Neighbor-List Pre-Step Decomposition*).
    Both fields are `false` in `Trivial` mode and on a `pre_step`
    that performs no rebuild.

- `NeighborListState` — host-side wrapper carrying the device buffers <!-- rq-b2d68288 -->
  and parameters that make up the shared neighbor list. The state is in
  one of two modes, fixed at construction.

  Fields present in both modes:
  - `device: Arc<CudaDevice>`
  - `particle_count: usize`
  - `mode: NeighborListMode` — discriminator (`Trivial` or `CellList`).

  There is no per-particle `neighbor_list` / `neighbor_counts` buffer
  and no `max_neighbors` field; the partner data lives entirely in the
  packed-neighbour buffers enumerated below.

  Fields present only in `CellList` mode:
  - `n_cells: [u32; 3]` — number of cells along each lattice direction
    `(n_a, n_b, n_c)`.
  - `n_cells_total: usize`
  - `r_cut: f32` — the largest `Potential::max_cutoff()` value reported by
    a consumer, captured at construction. Stored so the cache-refresh
    path can recompute `n_cells` from a mutated box.
  - `r_skin: f32`
  - `r_search_sq: f32` — pre-computed `(r_cut + r_skin)²` for the build
    kernel. Independent of the box; not refreshed on generation change.
  - `cached_generation: u64` — the box's `generation()` value at the time
    the cell-layout cache was populated. Compared against the live box's
    generation on every `pre_step`; a mismatch refreshes the cache (see
    *Box Generation Tracking*).
  - `cell_indices: CudaSlice<u32>` (length `N`) — per-atom scratch
    populated by the cell-indexing stage. Sized by `particle_count`; not
    reallocated on box-generation refresh.
  - `cell_counts: CudaSlice<u32>` (length `n_cells_total`) — per-cell
    occupancy scratch populated by the histogram stage and consumed by
    the prefix scan. Reallocated when `n_cells_total` changes.
  - `cell_offsets: CudaSlice<u32>` (length `n_cells_total + 1`) — output
    of the exclusive prefix scan over `cell_counts`. Reallocated when
    `n_cells_total` changes.
  - `write_cursors: CudaSlice<u32>` (length `n_cells_total`) — per-cell
    atomic write cursors used by the scatter stage. Reset to zero at the
    start of every rebuild. Reallocated when `n_cells_total` changes.
  - `scan_block_totals: Vec<CudaSlice<u32>>` — the block-totals stack
    for the recursive prefix scan. Buffer `l` holds the per-block
    inclusive totals produced at recursion level `l` and has length
    `ceil(n_cells_total / scan_block_size^(l + 1))`; the stack carries
    one buffer per recursion level — `O(log(n_cells_total))` buffers, at
    most four at the 256-thread block size since the buffer count is
    bounded by the `u32` cell-addressing limit. Rebuilt as a whole when
    `n_cells_total` changes.
  - `sorted_particle_ids: CudaSlice<u32>` (length `N`)
  - `reference_positions_x/y/z: CudaSlice<f32>` (length `N`)
  - `neighbor_status: CudaSlice<u32>` (length `1`) — single-word
    packed status: bit 0 (`displacement_tripped`) written by the
    displacement-check kernel, bits 1–4 (`tiles_high_water`,
    `single_pairs_high_water`, `tiles_overflow`,
    `single_pairs_overflow`) written by the packed-neighbour
    construction kernel, read once per batch by `pre_step`. See
    *Displacement Check*.
  - `threshold_sq: f32` — host-side cache of `(r_skin / 2)²`, passed
    as a kernel argument by the displacement-check kernel launch.
  - `needs_rebuild: bool` — initial value `true`.

  The packed-neighbour buffers (`tile_sorted_positions_*`,
  `block_centre`, `block_bbox`, `sorted_blocks`, `interacting_tiles`,
  `interacting_atoms`, `single_pairs`, `interaction_count`,
  `tile_atom_count`, `tile_lane_mask`) are part of `NeighborListState`
  in both `CellList` and `Trivial` modes; their schema is specified
  in `packed-neighbour-pair-force.md`.

  In `Trivial` mode the cell-list-specific fields are absent;
  `sorted_particle_ids` is the identity permutation, populated once at
  construction.

- `NeighborListMode` — discriminator. Variants: <!-- inline --> <!-- inline --> <!-- rq-ff424773 -->
  - `Trivial`
  - `CellList` — the cell-list-mode state described above; produces
    the cell-list output (sorted particle IDs and per-cell offsets)
    that feeds the packed-neighbour construction pipeline (see
    `packed-neighbour-pair-force.md`).
  - `CellListOnly` — the same cell-list output, without driving the
    displacement-check rebuild trigger. Used by the SPME
    reciprocal-space slot (see `spme.md`); the spread and gather
    kernels read `sorted_particle_ids` and `cell_offsets` only.

- `NeighborListError` — error type. Variants: <!-- rq-d8e4407a -->
  - `Gpu(GpuError)` — CUDA driver / kernel-launch failure.
  - `BoxTooSmallForCells { direction: &'static str, width: f32, required: f32 }`
    — the simulation box's perpendicular width is smaller than
    `3 * (r_cut + r_skin)` along the named lattice direction.
    `direction` is one of `"a"`, `"b"`, `"c"`. `width` is the box's
    perpendicular width along that direction; `required` is
    `3 * (r_cut + r_skin)`. Detected at construction and on
    box-generation refresh.
  - `TooManyCells { n_cells_total: usize, max_supported: usize }` — the
    cell layout would produce more cells than the device can address
    with a `u32` cell index. `max_supported` is `u32::MAX as usize`.
    Detected at construction and on box-generation refresh, before any
    device buffer is allocated. In practice GPU memory is exhausted by
    the `n_cells_total`-sized buffers long before this ceiling is
    reached, but the check makes that case an explicit error rather
    than silent integer overflow in the cell-index arithmetic.

### Functions <!-- rq-3553aab2 -->

- `NeighborListState::new_cell_list(device: Arc<CudaDevice>, sim_box: &SimulationBox, particle_count: usize, r_cut: f32, r_skin: f32, tile_pair_config: TilePairCapacityConfig) -> Result<NeighborListState, NeighborListError>` <!-- rq-14033af1 -->
  - Constructs a `CellList`-mode state.
  - Computes `n_cells` per lattice direction from
    `floor(w_d / (r_cut + r_skin))` where `w_d` is the box's perpendicular
    width along direction `d` (see `simulation-box.md`).
  - Returns `BoxTooSmallForCells` if any direction has `n_cells < 3`.
  - Returns `TooManyCells` if `n_cells_total` exceeds `u32::MAX`.
  - Allocates every device buffer described in the *CellList*-mode field
    list, including the persistent scratch buffers (`cell_indices`,
    `cell_counts`, `write_cursors`) and the block-totals stack
    (`scan_block_totals`). Reference positions start at zero;
    `needs_rebuild` starts at `true`.
  - Stores `r_cut` so the cache-refresh path (see *Box Generation
    Tracking*) can recompute `n_cells` from a mutated box. `r_cut` is the
    largest cutoff across every consumer of the shared list; the framework
    computes this as the maximum of every `Potential::max_cutoff()` value
    it observes.
  - `tile_pair_config: TilePairCapacityConfig` carries
    `tile_pair_growth_factor` only. The packed-neighbour buffers are
    allocated at a small seed length here; their working capacity is
    set by the first (probe) rebuild and grown on demand, so there is
    no initial-capacity argument (see
    `forces/packed-neighbour-pair-force.md` *Capacity*).
  - Records `cached_generation = sim_box.generation()`.

- `NeighborListState::new_cell_list_only(device: Arc<CudaDevice>, sim_box: &SimulationBox, particle_count: usize, n_cells_per_direction: [u32; 3]) -> Result<NeighborListState, NeighborListError>` <!-- rq-d47caa3d -->
  - Constructs a `CellListOnly`-mode state with explicit grid
    dimensions, bypassing the `r_cut + r_skin` derivation used by
    `new_cell_list`.
  - Stores `n_cells = n_cells_per_direction` directly; the constructor
    rejects any direction with `n_cells_per_direction[d] < 3` via
    `BoxTooSmallForCells { direction, width: 0.0, required: 3.0 }`
    (the `width: 0.0` field reflects that the rejection comes from the
    grid spec, not from a measured box width).
  - Allocates the cell-list scratch buffers (`cell_indices`,
    `cell_counts`, `write_cursors`, `scan_block_totals`,
    `sorted_particle_ids`, `cell_offsets`) sized to `n_cells_total`.
    Does **not** allocate `reference_positions_*` or `neighbor_status`
    (these fields are absent from `CellListOnly`-mode states).
  - `r_cut`, `r_skin`, `r_search_sq` are not stored.
  - The state's `pre_step` rebuilds the cell list on every call,
    unconditionally; the displacement-check kernel is never launched
    in `CellListOnly` mode.
  - Records `cached_generation = sim_box.generation()`. A box-generation
    mismatch on subsequent `pre_step` calls triggers a layout refresh
    that reverifies the grid against the new box (the `n_cells` value
    is fixed; the box-generation refresh re-derives the cell sizes
    implicitly through the kernels' fractional-coordinate math).

- `NeighborListState::new_trivial(device: Arc<CudaDevice>, sim_box: &SimulationBox, particle_count: usize) -> Result<NeighborListState, NeighborListError>` <!-- inline --> <!-- rq-c96fd9d2 -->
  - Constructs a `Trivial`-mode state. The `sim_box` argument is accepted
    for API uniformity; `Trivial` mode does not consult it.
  - Sets `sorted_particle_ids` to the identity permutation `[0, 1, …,
    particle_count)` so every atom-block is dense in particle-id order,
    then runs the packed-neighbour construction (see *Packed-Neighbour
    List Construction*) over the whole system. No per-particle
    `neighbor_list` / `neighbor_counts` buffer is allocated; the partner
    data lives in the packed-neighbour buffers.
  - When `particle_count == 0`, `sorted_particle_ids` and the
    packed-neighbour buffers have length zero.

- `NeighborListState::displacement_check(&mut self, sim_box: &SimulationBox, buffers: &ParticleBuffers, timings: &mut Timings) -> Result<bool, NeighborListError>` <!-- rq-c49b2fe6 -->
  - Issues a single-word `dtoh_sync_copy` of `neighbor_status` and
    returns `true` when bit 0 (`displacement_tripped`) is set — i.e. the
    displacement-check kernel has signalled, on any step since the last
    rebuild, that an atom moved more than `r_skin / 2` (see
    *Displacement Check*). The per-step displacement-check kernel itself
    is launched from the force pipeline, not from this method.
  - Returns `false` when `particle_count == 0`.
  - Returns `false` when the state is in `Trivial` mode (no rebuild
    machinery exists).

- `NeighborListState::rebuild(&mut self, sim_box: &SimulationBox, buffers: &ParticleBuffers, timings: &mut Timings) -> Result<bool, NeighborListError>` <!-- rq-7db97132 -->
  - Runs the device-side cell-list pipeline (see *Cell List
    Construction*) followed by the packed-neighbour construction
    pipeline (see *Packed-Neighbour List Construction*), using
    `(lx, ly, lz)` from `sim_box` throughout. Updates reference
    positions. Returns `true` when the construction grew (reallocated)
    a packed-neighbour buffer.
  - Performs no host-device transfers of particle data; all
    intermediates (`cell_indices`, `cell_counts`, `cell_offsets`,
    `write_cursors`, `sorted_particle_ids`) are populated on the device.
  - Grows the packed-neighbour buffers and re-runs the construction
    when the build kernel reports an overflow, as described in
    `forces/packed-neighbour-pair-force.md` *Capacity*, retrying until
    the build fits.
  - Records whether any packed-neighbour buffer was reallocated during
    the rebuild so the caller (`pre_step`) can surface it through
    `PreStepOutcome.reallocated`.
  - Returns `Ok(())` immediately when `particle_count == 0` or when the
    state is in `Trivial` mode.

- `NeighborListState::pre_step(&mut self, sim_box: &SimulationBox, buffers: &ParticleBuffers, timings: &mut Timings) -> Result<PreStepOutcome, NeighborListError>` <!-- rq-1217c816 -->
  - Called by `ForceField::step` once per timestep before any slot's
    `contribute` runs. In `CellList` mode:
    1. Compares `sim_box.generation()` against `cached_generation`. On
       mismatch refreshes the cell-layout cache (see *Box Generation
       Tracking*); sets `needs_rebuild = true` and skips the displacement
       check **only when the refreshed `n_cells_total` differs from the
       prior cached value**. When `n_cells_total` is unchanged the
       refresh updates `cached_generation` and the cached lattice
       parameters and falls through to step 2. May return
       `BoxTooSmallForCells`.
    2. If `!needs_rebuild`, downloads `neighbor_status` (a single
       `u32`) and sets `needs_rebuild = true` when the value is
       non-zero.
    3. If `needs_rebuild`, runs the rebuild, refreshes the reference
       positions, and zeros `neighbor_status`.
    In `Trivial` mode this is a no-op.
  - Returns a `PreStepOutcome { rebuilt: bool, reallocated: bool }`.
    `rebuilt` is `true` when step 3 ran. `reallocated` is `true` when
    that rebuild grew (and therefore reallocated) any packed-neighbour
    buffer (`interacting_tiles`, `interacting_atoms`, or
    `single_pair_atoms`); the batched graph-replay loop consumes this
    flag to decide whether to re-capture the phase graph (see
    `cuda-graphs.md` *Neighbor-List Pre-Step Decomposition*). Both
    fields are `false` in `Trivial` mode and on a `pre_step` that does
    not rebuild.

### CUDA Kernels <!-- rq-0469400b -->

`kernels/neighbor.cu` declares the following `extern "C"` kernels.

Displacement-check and reference kernels:

```c
extern "C" __global__ void neighbor_displacement_check_flag(
    const float4 *posq,
    const float *reference_x, const float *reference_y, const float *reference_z,
    const float *lattice,           // length 6: [lx, ly, lz, xy, xz, yz]
    float threshold_sq,             // (r_skin / 2)²
    unsigned int *neighbor_status,// length 1; bit 0 set via atomicOr on first
                                    //   atom that exceeds threshold
    unsigned int n);

extern "C" __global__ void copy_positions_into_reference(
    const float4 *posq,
    float *reference_x, float *reference_y, float *reference_z,
    unsigned int n);
```

Both kernels above read only `.xyz` from `posq`; they ignore the
per-atom charge in `.w`. The reference-position arrays remain SoA:
they store only positions (no charges), and the displacement-check
kernel only consults them as scalar x/y/z.
`copy_positions_into_reference` splits `posq.xyz` into the three
scalar reference buffers at every neighbour-list rebuild.

`neighbor_displacement_check_flag` writes nothing when every atom is
within `r_skin / 2` of its reference position. When at least one
atom exceeds the threshold, the first such thread issues
`atomicOr(neighbor_status, 1u)`, setting bit 0. The word's bits are
otherwise set by the packed-neighbour construction kernel (bits 1–4)
and cleared only by the host-side `memset_zeros` at the start of a
rebuild (see *Displacement Check*).

The six lattice parameters `(lx, ly, lz, xy, xz, yz)` carry the box's
lower-triangular form (see `simulation-box.md`).
`neighbor_displacement_check_flag` computes its minimum-image
displacements via the triclinic *Wrap Algorithm* defined in
`simulation-box.md`.

Spatial-hash pipeline (cell-list construction):

```c
extern "C" __global__ void compute_cell_indices_and_histogram(
    const float4 *posq,
    const float *lattice,           // length 6: [lx, ly, lz, xy, xz, yz]
    unsigned int n_cells_a, unsigned int n_cells_b, unsigned int n_cells_c,
    unsigned int *cell_indices,
    unsigned int *cell_counts,
    unsigned int n);

extern "C" __global__ void prefix_scan_local_blocks(
    const unsigned int *input,
    unsigned int *output,
    unsigned int *block_totals,
    unsigned int len);

extern "C" __global__ void prefix_scan_apply_block_totals(
    const unsigned int *block_offsets,
    unsigned int *output,
    unsigned int len);

extern "C" __global__ void prefix_scan_finalize_offsets(
    unsigned int *cell_offsets,
    unsigned int n_cells_total,
    unsigned int particle_count);

extern "C" __global__ void scatter_atoms_into_cells(
    const unsigned int *cell_indices,
    const unsigned int *cell_offsets,
    unsigned int *write_cursors,
    unsigned int *sorted_particle_ids,
    unsigned int n);

extern "C" __global__ void sort_cells_by_particle_id(
    const unsigned int *cell_offsets,
    unsigned int *sorted_particle_ids,
    unsigned int n_cells_total);
```

`compute_cell_indices_and_histogram` writes each atom's cell index to
`cell_indices` and increments `cell_counts[cell_idx]` via `atomicAdd`.

`prefix_scan_local_blocks` performs a per-block exclusive scan over
`input[0 .. len]`, writing the local scan to `output[0 .. len]` and each
block's inclusive total to `block_totals[blockIdx]`. Each thread reads
its input element before any write and blocks write disjoint output
ranges, so `input` and `output` may alias the same buffer — the
recursive driver scans each block-totals level of the stack in place.
`prefix_scan_apply_block_totals` adds `block_offsets[gid / scan_block_size]`
into `output[gid]` for every `gid < len`. `prefix_scan_finalize_offsets`
writes the trailing `cell_offsets[n_cells_total] = particle_count`
sentinel with a single thread.

`scatter_atoms_into_cells` writes
`sorted_particle_ids[cell_offsets[cell_indices[i]] + atomicAdd(&write_cursors[cell_indices[i]], 1)] = i`.

`sort_cells_by_particle_id` runs one thread per cell; each thread
insertion-sorts the slice `[cell_offsets[c], cell_offsets[c+1])` of
`sorted_particle_ids` ascending by particle index.

### Rust Launch Helpers <!-- rq-fec7ae1c -->

Free functions in `src/gpu/kernels.rs`, re-exported from `crate::gpu`.
Each is a no-op when `particle_count == 0`.

Neighbor-list-build pipeline:

- `neighbor_displacement_check_flag(particle_buffers, reference_x, reference_y, reference_z, sim_box, threshold_sq, neighbor_status) -> Result<(), GpuError>` <!-- rq-884b5cd6 -->
  — launches the per-atom displacement-check kernel; sets bit 0 of
  `neighbor_status` via `atomicOr(_, 1u)` if any atom's minimum-image
  displacement from its reference position exceeds `sqrt(threshold_sq)`,
  leaving the construction kernel's bits 1–4 untouched. Called
  once per physical timestep from `ForceField::step` /
  `ForceField::step_no_neighbor_check` so the launch sits inside the
  captured graph when capture is active.
- `copy_positions_into_reference(particle_buffers, reference_x, reference_y, reference_z) -> Result<(), GpuError>` <!-- rq-344f7af0 -->

The packed-neighbour construction kernels (`scatter_positions_to_tile_order`,
`compute_block_bbox`, the volume sort, `find_blocks_with_interactions`,
and the i-block histogram / scan / scatter) are specified in
`packed-neighbour-pair-force.md`.

Spatial-hash pipeline:

- `compute_cell_indices_and_histogram(particle_buffers, sim_box, n_cells, cell_indices, cell_counts) -> Result<(), GpuError>` <!-- rq-10f6f831 -->
- `prefix_scan_cell_counts(cell_counts, cell_offsets, scan_block_totals, n_cells_total, particle_count) -> Result<(), GpuError>` — <!-- rq-2ef5e222 -->
  drives the recursive multi-level exclusive scan. Scans `cell_counts`
  into `cell_offsets` with `prefix_scan_local_blocks`, emitting
  `scan_block_totals[0]`; recursively scans each block-totals level of
  the stack in place; applies each level's scanned totals back into the
  level below with `prefix_scan_apply_block_totals`; finishes with
  `prefix_scan_finalize_offsets` to write the
  `cell_offsets[n_cells_total] = particle_count` sentinel. Issues
  `O(log(n_cells_total))` kernel launches.
- `scatter_atoms_into_cells(cell_indices, cell_offsets, write_cursors, sorted_particle_ids, particle_count) -> Result<(), GpuError>` — <!-- rq-9d0cb192 -->
  zeroes `write_cursors` before launching the scatter kernel.
- `sort_cells_by_particle_id(cell_offsets, sorted_particle_ids, n_cells_total) -> Result<(), GpuError>` <!-- rq-165c4422 -->

## Launch Configuration <!-- rq-2e15fed7 -->

- Block size: 256 threads for the per-atom kernels
  (`neighbor_displacement_squared`, `copy_positions_into_reference`,
  `compute_cell_indices_and_histogram`, `scatter_atoms_into_cells`),
  with grid `ceil(n / 256)`.
- Block size: 256 threads for the per-cell kernel
  (`sort_cells_by_particle_id`), with grid `ceil(n_cells_total / 256)`.
- Block size: 256 threads for the scan kernels
  (`prefix_scan_local_blocks`, `prefix_scan_apply_block_totals`). At
  recursion level `l` both are launched with grid `ceil(len_l / 256)`,
  where `len_0 = n_cells_total` and `len_{l+1} = ceil(len_l / 256)`; the
  recursion terminates at the level whose input fits in a single block.
  `prefix_scan_finalize_offsets` runs a single thread.
- Shared memory: `prefix_scan_local_blocks` uses one
  `unsigned int[2 * block_size]` for the double-buffered local scan.
  Other cell-list kernels use no shared memory. The packed-neighbour
  construction kernels' launch configuration is specified in
  `packed-neighbour-pair-force.md`.
- Stream: the default stream carried by `particle_buffers.device`.

## Determinism <!-- rq-c62bb861 -->

The cell-list build canonicalises within-cell order, producing a
`sorted_particle_ids` that is bit-identical across runs given identical
inputs on the same GPU. The packed-neighbour construction consumes that
ordering, so its output is likewise deterministic (the construction's
own determinism invariants are in `packed-neighbour-pair-force.md`
*Determinism*).

**Per-cell sort within `sorted_particle_ids`.** The spatial-hash
pipeline places atoms into cells with an `atomicAdd`-based scatter
whose within-cell order is non-deterministic, then runs a per-cell
insertion sort over `sorted_particle_ids` keyed on particle index.
Atomic integer addition is associative so the histogram and the
write-cursor counts are run-to-run identical even though atomic
ordering is not; the per-cell sort canonicalises the scatter output.
The end-to-end result is identical to a stable lexicographic sort on
`(cell_index, particle_id)`. **Required for run-to-run
reproducibility.**

## Performance Notes <!-- rq-54a28837 -->

- Cell-list rebuild cost is the per-atom and per-cell kernels plus the
  prefix scan and one device memset, with no host-device transfers of
  particle data. The prefix scan issues `O(log(n_cells_total))` kernel
  launches — a small constant, at most six up to ~16 M cells. Total
  work is `O(N)` for the per-atom kernels (cell index, histogram,
  scatter), `O(N · d)` for the per-cell sort at average cell density
  `d`, and `O(n_cells_total)` for the prefix scan. The packed-neighbour
  construction that follows is analysed in
  `packed-neighbour-pair-force.md`.
- Atomic-add contention: each cell sees on the order of `d` serialised
  `atomicAdd`s in the histogram and `d` in the scatter. Negligible at
  liquid density (`d` ≈ 5–20).
- Displacement check: one `u32` status word downloaded once per batch
  boundary (see *Displacement Check*), not a per-atom download.

## Out of Scope <!-- rq-58acf788 -->

- Coulomb-style long-range force, which would require a non-cell-list
  decomposition (Ewald / PME). The neighbor-list framework here is
  short-range only.
- Half-neighbor-list (Newton's-third-law optimisation that lists each
  pair once instead of twice).
- Constant or adaptive `r_skin`. v1 is constant.
- Per-pair-of-consumers cutoff filtering. The shared packed list is
  built once at the maximum cutoff across consumers; each consumer
  applies its own per-pair cutoff at force-evaluation time, reading the
  list but discarding entries beyond its own cutoff.
- Detecting a box mutation that bypasses `SimulationBox::set_lattice`
  (e.g. two `SimulationBox` values constructed independently that happen
  to share a `generation`, or a future API that mutates the lattice
  without bumping the counter). The generation counter is the contract;
  consumers trust the runner to own one canonical box and to use only
  the documented mutator.

### Known packed-neighbour double-emit

Two mechanisms have historically caused a pair to be visited
more than once by the packed pair-force kernel. Each has a
separate status:

1. **Mixed-entry Newton's-3rd double-count (FIXED).** When a
   packed entry contained both self-block-like j-atoms (atoms of
   the same i-block, expected to route through self-block
   detection) and cross-block j-atoms in the same 32-slot row,
   the coarse `self_block` detection failed for the entry and
   `j_fx -= fx` was applied to every pair. For self-block-like
   pairs inside the mixed entry this doubled the atom's force
   because both sides of the 32-rotation sweep contributed. The
   fix, in `heddle_jit_outer_loop` (see
   `jit-composed-pair-force.md` *Composed-Kernel Structure*),
   is a per-lane `my_j_in_iblock` flag computed once per entry
   and rotated alongside the j-side state; Newton's 3rd is
   suppressed per-pair whenever the current j-atom sits anywhere
   in the entry's i-block set, subsuming the old warp-wide
   `self_block` detection.

2. **Duplicate-entry emission (OPEN).**
   `find_blocks_with_interactions` produces two or more packed
   entries with the same i-block index whose j-atom rows contain
   overlapping ranges of atoms. Diagnostic dumps of the packed
   output show, for example, two entries for the same i-block
   that both list a contiguous 16-atom block among their j-atoms
   but positioned at different slot indices, indicating that the
   same tile-pair's dense output is being flushed to the buffer
   twice. The mechanism has not been isolated to a specific
   code path in `find_blocks_with_interactions`; the sort
   (`scatter_atoms_into_cells` + `sort_cells_by_particle_id`)
   and the scatter (`scatter_entries_by_iblock`) both look
   correct in isolation.

   The residual is masked for excluded pairs by the
   "apply `exclusion_scale` inline in `heddle_jit_eval_pair_sum`"
   design documented in `jit-composed-pair-force.md`: for a
   fully excluded pair (`scale = 0`) both visits contribute
   `0 × pair_force = 0`, and for a 1-4 pair (`scale = 0.5`)
   the two visits contribute `1.0 × pair_force` instead of the
   correct `0.5 × pair_force` — a bounded ~1 × 10⁻⁴ atomic-unit
   residual at reference intramolecular geometry. Non-excluded
   pairs (`scale = 1`) are still doubled, which is a real
   physics error for dense systems and remains open.

The regression test suite in `tests/neighbor_list_correctness.rs`
carries `#[ignore]`-marked scenarios that surface the defect
under `cargo test -- --ignored`. The residual failures target
the equilibrium-force and translation-invariance checks and
turn GREEN when the underlying duplicate-entry emission is
fixed.

---

## Gherkin Scenarios <!-- rq-c4645fa6 -->

```gherkin
Feature: Cell-list neighbor list

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called
    And an orthorhombic SimulationBox with lx=ly=lz=10.0
    And a particle count of 100

  # --- Cell layout ---

  @rq-c0cfc5d6
  Scenario: Cell counts are floor(w / (r_cut + r_skin)) for an orthorhombic box
    Given r_cut = 1.0, r_skin = 0.3
    And an orthorhombic SimulationBox with lx=ly=lz=10.0 (so w_a=w_b=w_c=10.0)
    When NeighborListState::new_cell_list is called
    Then n_cells equals [7, 7, 7]

  @rq-a7aac794
  Scenario: Cell counts reflect perpendicular widths for a tilted box
    Given r_cut + r_skin = 1.3
    And a SimulationBox::new(10.0, 10.0, 10.0, 0.0, 0.0, 10.0)
      (so w_c = 10.0 but w_b = (ly*lz)/sqrt(lz² + yz²) = 100/sqrt(200) ≈ 7.07)
    When NeighborListState::new_cell_list is called
    Then n_cells[1] equals floor(7.07 / 1.3) = 5
    And n_cells[2] equals floor(10.0 / 1.3) = 7

  @rq-1b9c474c
  Scenario: Reject configurations whose box admits fewer than 3 cells along any direction
    Given r_cut = 1.0, r_skin = 3.0 (so r_cut + r_skin = 4.0)
    And an orthorhombic SimulationBox with lx=10.0 (so w_a = 10.0, giving floor(10/4) = 2 < 3)
    When NeighborListState::new_cell_list is called
    Then it returns Err(NeighborListError::BoxTooSmallForCells { direction: "a", width: 10.0, required: 12.0 })

  @rq-e84d3bac
  Scenario: Reject a tilted box whose perpendicular width drops below 3*(r_cut+r_skin)
    Given r_cut + r_skin = 4.0 (required perpendicular width = 12.0)
    And a SimulationBox::new(10.0, 10.0, 10.0, 0.0, 0.0, 8.0) so that
      w_b = (10*10)/sqrt(100 + 64) ≈ 7.81 < 12.0
    When NeighborListState::new_cell_list is called
    Then it returns Err(NeighborListError::BoxTooSmallForCells { direction: "b", width: w, required: 12.0 })
      where w is the actual computed perpendicular width

  @rq-151cb099
  Scenario: Cell index of a position at the +1/2 fractional-coordinate boundary clamps inside the grid
    Given a particle whose fractional coordinate along a equals +0.5 (boundary case)
    When its cell index is computed
    Then cell_a equals n_cells_a - 1 (no out-of-bounds index)

  @rq-a99ca751
  Scenario: Cell index of a position outside the primary cell wraps before binning
    Given a particle whose Cartesian position is exactly one full lattice vector
      past the primary cell along direction a
    When its cell index is computed
    Then it equals the cell index of the corresponding particle inside the primary cell

  # --- Cell list construction ---

  @rq-838acdee
  Scenario: sorted_particle_ids sorts particles by (cell_index, particle_id)
    Given particles placed at positions producing cells [2, 0, 1, 0, 2]
    When NeighborListState::rebuild is called
    Then sorted_particle_ids equals [1, 3, 2, 0, 4]
      (cell 0: particles 1 and 3; cell 1: particle 2; cell 2: particles 0 and 4;
       within a cell, sorted ascending by particle_id)

  @rq-cd50d861
  Scenario: cell_offsets contains the prefix sum of cell occupancies
    Given the same particles as above
    Then cell_offsets[0] = 0, cell_offsets[1] = 2, cell_offsets[2] = 3, cell_offsets[3] = 5
    And cell_offsets has length n_cells_total + 1

  # --- Neighbor list construction ---

  @rq-ea0ee5ef
  Scenario: Neighbor list contains every particle within r_cut + r_skin
    Given particles in a regular grid spaced 0.5 apart, r_cut = 1.0, r_skin = 0.3
    When NeighborListState::rebuild is called
    Then atom 0's neighbor list contains every particle within 1.3 of atom 0
    And it excludes every particle farther than 1.3

  @rq-e75b24e7
  Scenario: Neighbor list excludes the self atom
    Given any non-empty system
    When NeighborListState::rebuild is called
    Then atom i's neighbor list does not contain i

  @rq-25faef11
  Scenario: Neighbor list applies minimum-image PBC
    Given particles at x = -lx/2 + 0.1 and x = +lx/2 - 0.1 (separated by 0.2
      across the periodic boundary)
    And r_cut + r_skin = 1.0
    When NeighborListState::rebuild is called
    Then atom 0's neighbor list contains atom 1
    And the displacement used was the minimum-image dx = +0.2 (not lx - 0.2)

  @rq-6bf3709c
  Scenario: Two independent rebuilds with identical positions produce identical lists
    Given two NeighborListStates built from identical configurations and identical positions
    When each is rebuilt
    Then their sorted_particle_ids, cell_offsets, interacting_tiles,
      interacting_atoms, and interaction_count agree byte-for-byte

  # --- Displacement check ---

  @rq-837c85d3
  Scenario: Displacement-check kernel on reference positions equal to current leaves the flag clear
    Given a NeighborListState immediately after a rebuild
    When neighbor_displacement_check_flag runs
    Then neighbor_status remains 0u

  @rq-6e1f04f3
  Scenario: Displacement-check kernel uses minimum-image displacement
    Given a particle whose reference position is x = lx/2 - 0.05
    And whose current position is x = -lx/2 + 0.05 (wrapped across the boundary)
    And threshold_sq = 0.15² (i.e., r_skin / 2 = 0.15)
    When neighbor_displacement_check_flag runs
    Then neighbor_status remains 0u (the wrapped displacement is ≈ 0.1, below threshold)

  @rq-c43dd1ab
  Scenario: Displacement-check kernel sets the flag when any particle exceeds threshold
    Given particle 7 has moved 0.5 from its reference and all other particles have moved less than r_skin / 2
    When neighbor_displacement_check_flag runs
    Then neighbor_status equals 1u

  @rq-c9f970fe
  Scenario: Displacement-check kernel is sticky across captured replays
    Given a captured graph that runs neighbor_displacement_check_flag every step
    And the graph is replayed for K = 50 steps in a single batch
    And on the third replay particle 7 exceeds threshold but on the fiftieth replay every particle is back within threshold
    When the host downloads neighbor_status at the batch boundary
    Then the downloaded value equals 1u

  @rq-46d72444
  Scenario: A rebuild clears the displacement flag
    Given neighbor_status holds 1u and pre_step decides to rebuild
    When the rebuild completes
    Then reference_positions_{x,y,z} equal the current posq.xyz componentwise
    And neighbor_status has been zeroed via memset_zeros before pre_step returns

  @rq-5d2e8748
  Scenario: pre_step downloads exactly one u32 per call
    Given a NeighborListState in CellList mode with N = 24576 particles
    When pre_step is called
    Then exactly one dtoh_sync_copy of length 1 (u32) is issued against neighbor_status
    And no other host-device particle transfer is issued by pre_step

  @rq-75f86ce3
  Scenario: pre_step reports no reallocation when a rebuild reuses the buffers
    Given a NeighborListState in CellList mode whose rebuild fits the current
      packed-neighbour capacity
    When pre_step runs a rebuild
    Then pre_step returns PreStepOutcome { rebuilt: true, reallocated: false }

  @rq-1ca7df49
  Scenario: pre_step reports reallocation when a high-water build grows a buffer
    Given a NeighborListState in CellList mode whose previous build set the
      tiles_high_water bit of neighbor_status
    When pre_step reads neighbor_status, grows interacting_tiles geometrically,
      and runs the rebuild
    Then pre_step returns PreStepOutcome { rebuilt: true, reallocated: true }

  @rq-f867ab96
  Scenario: A high-water bit forces a rebuild even when the displacement bit is clear
    Given a NeighborListState in CellList mode with no box-generation change
    And neighbor_status with the tiles_high_water bit set and bit 0 clear
    When pre_step is called
    Then pre_step grows interacting_tiles to ceil(capacity * tile_pair_growth_factor)
    And pre_step runs a rebuild and returns PreStepOutcome { rebuilt: true, reallocated: true }

  @rq-8142fff7
  Scenario: An overflow bit makes pre_step halt with PackedNeighborOverflow
    Given a NeighborListState in CellList mode past its pre-capture probe rebuild
    And neighbor_status with the tiles_overflow bit set
    When pre_step reads neighbor_status
    Then pre_step returns Err(NeighborListError::PackedNeighborOverflow { buffer: "interacting_tiles" })
    And no further rebuild is attempted on this call

  @rq-a39234ba
  Scenario: A steady-state rebuild copies no interaction count to the host
    Given a NeighborListState in CellList mode past its pre-capture probe rebuild
    And a pre_step that runs a rebuild
    Then the only dtoh_sync_copy issued by pre_step is the single neighbor_status word
    And interaction_count is never copied to the host

  @rq-623447db
  Scenario: pre_step that performs no rebuild reports neither flag
    Given a NeighborListState in CellList mode with no box-generation change
      and neighbor_status holding 0u
    When pre_step is called
    Then pre_step returns PreStepOutcome { rebuilt: false, reallocated: false }

  # --- Rebuild policy ---

  @rq-35981c27
  Scenario: First displacement_check call always rebuilds (needs_rebuild starts true)
    Given a freshly-constructed NeighborListState
    When pre_step is called for the first time
    Then a rebuild occurs unconditionally
    And needs_rebuild is false afterwards

  @rq-90524f5d
  Scenario: Sub-skin movement does not trigger a rebuild
    Given a NeighborListState immediately after a rebuild with r_skin = 0.3
    When every particle has moved less than r_skin/2 = 0.15 from its reference position
    And pre_step is called
    Then no rebuild occurs

  @rq-9f63a183
  Scenario: Over-skin movement triggers a rebuild
    Given a NeighborListState immediately after a rebuild with r_skin = 0.3
    When at least one particle has moved more than r_skin/2 = 0.15
    And pre_step is called
    Then a rebuild occurs
    And reference positions equal the current positions afterwards

  # --- Empty and tiny states ---

  @rq-4bc8028f
  Scenario: NeighborListState with particle_count = 0 builds successfully
    When NeighborListState::new_cell_list is called with particle_count = 0
    Then it returns Ok(state)
    And rebuild is a no-op
    And displacement_check returns 0.0

  @rq-52f547fd
  Scenario: NeighborListState with particle_count = 1 produces an empty neighbor list
    When a single particle is in the system
    And NeighborListState::rebuild is called
    Then interaction_count equals [0, 0] (no interacting tiles, no single pairs)

  # --- Determinism ---

  @rq-4b40604b
  Scenario: Cell-list mode and trivial mode produce identical forces (within f32 tolerance)
    Given two ForceField instances with identical particle positions and parameters,
      one in mode = "cell-list" with r_skin = 0.3, the other in mode = "all-pairs"
    When both run a single force evaluation
    Then the resulting forces_* agree componentwise within 1e-4 relative error

  # --- Cross-validation with the all-pairs oracle ---
  #
  # The all-pairs (trivial) neighbour list enumerates every pair by
  # construction, so it serves as an oracle for the cell-list build.
  # Any packed-neighbour defect — dropped pairs, doubled pairs, wrong
  # sort order, off-by-one in a boundary check — surfaces as a
  # per-atom force disagreement between the two modes.

  @rq-6eca8f0e
  Scenario: Cell-list forces match all-pairs forces on a multi-molecule intramolecular-exclusion system
    Given a ForceField configuration with at least 500 molecules of a
      polyatomic species (>= 3 atoms per molecule) with intramolecular
      bond, angle, and (if declared by the species) dihedral exclusions
    And two ForceField instances built from byte-identical inputs, one
      in mode = "cell-list" with the default r_skin and one in
      mode = "all-pairs"
    When both run a single force evaluation on identical particle positions
    Then per-atom forces_* agree componentwise within 1e-4 relative error
    And in particular F_max in the cell-list run is not more than 1e-6
      absolute in any component that is zero (to within f32 rounding) in
      the all-pairs run

  @rq-d991a151
  Scenario: Cell-list forces match all-pairs forces across an r_skin sweep
    Given a ForceField configuration whose box, particle_count, and
      cutoffs admit r_skin values in some non-degenerate range
      [r_skin_min, r_skin_max]
    And an all-pairs reference run over the same particle state
    When ForceField::new is constructed with mode = "cell-list" for
      each r_skin in a sweep sampling at least eight values covering
      the range (including at least one value that lands the r_search
      floor between two integer cell counts along each box axis)
    Then every cell-list run's per-atom forces_* agree with the
      all-pairs reference within 1e-4 relative error
    # r_skin is a rebuild-frequency parameter, not a physics parameter:
    # any observable difference in forces across r_skin values is a bug
    # in the packed-neighbour build or its downstream consumers.

  @rq-c90fb1bd
  Scenario: r_skin-invariance under a repeated force evaluation
    Given two ForceField instances with mode = "cell-list", identical
      particle state, identical parameters, and r_skin values
      r_skin_a and r_skin_b whose r_search rounds to different
      n_cells layouts along at least one box axis
    When both instances run a single force evaluation
    Then their per-atom forces_* agree componentwise within 1e-4
      relative error
    And F_max in each run is at thermal-scale (below the intramolecular
      LJ-at-bond-distance magnitude by at least six orders of magnitude)
      when the initial state places every intramolecular bond at its
      reference geometry

  # --- Trivial mode ---

  @rq-789fcec9
  Scenario: Trivial-mode contents
    Given a NeighborListState built via new_trivial with particle_count = 3
    When the state is built
    Then sorted_particle_ids equals [0, 1, 2] (the identity permutation)
    And the packed-neighbour buffers list every distinct atom pair among
      the three atoms exactly once (across interacting_tiles /
      single_pair_atoms)

  @rq-bb3773aa
  Scenario: Trivial-mode pre_step does no work
    Given a NeighborListState in Trivial mode
    When pre_step is called
    Then timings report zero samples for KernelStage::NeighborDisplacementSquared
    And pre_step returns PreStepOutcome { rebuilt: false, reallocated: false }

  @rq-30f85829
  Scenario: Trivial-mode state has no cell-list fields
    Given a NeighborListState built via new_trivial
    Then state.mode equals NeighborListMode::Trivial
    And the cell-list-specific buffers (cell_indices, cell_counts,
      cell_offsets, write_cursors, reference_positions_*, neighbor_status)
      are absent

  # --- Shared-service ownership ---

  @rq-2ed643ad
  Scenario: Two consumers share one neighbor list
    Given a ForceField containing two short-range Potential implementations
      with max_cutoff() reporting 5.0 and 3.0 respectively
    When ForceField::new builds the shared neighbor list in cell-list mode
    Then the neighbor list is built with r_search = 5.0 + r_skin
    And both potentials' contribute() receive the same NeighborListState reference

  @rq-83312d09
  Scenario: Bonded-only ForceField builds no neighbor list
    Given a ForceField whose only slot returns max_cutoff() = None
    When ForceField::new completes
    Then ForceField::neighbor_list is None
    And ForceField::step launches no displacement-check kernel and no neighbor-list-build kernel

  @rq-3bc18e1a
  Scenario: Max cutoff is the largest reported by any consumer
    Given a ForceField with three short-range slots reporting max_cutoffs 2.0, 4.5, 4.5
    When the framework computes the neighbor-list search radius
    Then r_search equals 4.5 + r_skin

  # --- Box generation tracking ---

  @rq-1b742a37
  Scenario: cached_generation initialised from the construction-time box
    Given a SimulationBox with generation 0
    When NeighborListState::new_cell_list is called with that box
    Then state.cached_generation equals 0

  @rq-882c9e86
  Scenario: cached_generation initialised from a non-zero construction-time generation
    Given a SimulationBox that has been mutated once (generation == 1)
    When NeighborListState::new_cell_list is called with that box
    Then state.cached_generation equals 1

  @rq-db8b171d
  Scenario: pre_step with unchanged box does not refresh the cell-layout cache
    Given a NeighborListState in CellList mode immediately after its first pre_step
    And the simulation box has not been mutated since construction
    When pre_step is called again with the same box
    Then n_cells and n_cells_total are unchanged
    And cell_offsets is not reallocated
    And state.cached_generation is unchanged

  @rq-cf847c1f
  Scenario: Box generation tick that changes n_cells_total forces a rebuild
    Given a NeighborListState in CellList mode immediately after its first pre_step
      with an orthorhombic box lx=ly=lz=10.0 and r_cut + r_skin = 1.3
      (so n_cells = [7, 7, 7], n_cells_total = 343)
    When box.set_lattice(20.0, 20.0, 20.0, 0.0, 0.0, 0.0) is called (generation 0 → 1)
    And pre_step is called with the updated box
    Then state.n_cells equals [15, 15, 15] (floor(20.0 / 1.3) = 15)
    And state.n_cells_total equals 3375 (differs from prior 343)
    And state.cached_generation equals box.generation() after the call
    And state.needs_rebuild was set to true and a rebuild was performed in
      the same pre_step call
    And the displacement-check kernel was not launched during this pre_step

  @rq-e2a31585
  Scenario: Box-generation refresh handles tilt mutation
    Given a NeighborListState in CellList mode with an orthorhombic box
      lx=ly=lz=10.0 and r_cut + r_skin = 1.3
    When box.set_lattice(10.0, 10.0, 10.0, 0.0, 0.0, 5.0) is called
      (introducing yz=5.0; w_b drops to (10*10)/sqrt(100 + 25) = 100/sqrt(125) ≈ 8.94)
    And pre_step is called with the updated box
    Then state.n_cells[1] equals floor(8.94 / 1.3) = 6
    And state.n_cells[0] and state.n_cells[2] equal floor(10.0 / 1.3) = 7

  @rq-dacb071c
  Scenario: Generation mismatch with new box too small returns BoxTooSmallForCells
    Given a NeighborListState in CellList mode with r_cut + r_skin = 1.3
    When box.set_lattice(3.0, 10.0, 10.0, 0.0, 0.0, 0.0) is called
      (so w_a = 3.0, giving floor(3.0 / 1.3) = 2 < 3)
    And pre_step is called with the updated box
    Then pre_step returns Err(NeighborListError::BoxTooSmallForCells { direction: "a", width: 3.0, required: 3.9 })
    And state.cached_generation is left unchanged
    And state.n_cells and state.n_cells_total are left unchanged
    And cell_offsets is not reallocated

  @rq-d22f105f
  Scenario: cell_offsets is reallocated when n_cells_total changes
    Given a NeighborListState in CellList mode with n_cells = [10, 10, 10]
      (n_cells_total = 1000, cell_offsets length 1001)
    When box.set_lattice is called producing n_cells = [11, 11, 11]
      (n_cells_total = 1331)
    And pre_step is called with the updated box
    Then cell_offsets is reallocated to length 1332

  @rq-331b6e81
  Scenario: cell_offsets is not reallocated when n_cells_total is unchanged
    Given a NeighborListState in CellList mode with n_cells = [10, 10, 10]
    When box.set_lattice is called producing n_cells = [10, 10, 10]
      (different lattice parameters but same n_cells_total)
    And pre_step is called with the updated box
    Then cell_offsets retains its previous device allocation (length 1001)

  @rq-31a9e3bb
  Scenario: r_search_sq is preserved across a generation refresh
    Given a NeighborListState in CellList mode with r_cut = 1.0 and r_skin = 0.3
    When box.set_lattice is called bumping the generation
    And pre_step is called with the updated box
    Then state.r_search_sq still equals 1.69 (i.e. (1.0 + 0.3)²)

  @rq-699cccff
  Scenario: Two pre_steps after a single box mutation refresh only once
    Given a NeighborListState in CellList mode
    When box.set_lattice bumps the generation once
    And pre_step is called, refreshing the cache and rebuilding
    And pre_step is called again without any further box mutation
    Then the second pre_step performs no cell-layout recompute
    And the second pre_step runs the displacement check (no longer skipped)

  @rq-f79d1ac5
  Scenario: Generation tick with unchanged n_cells_total still triggers rebuild when displacement exceeds threshold
    Given a NeighborListState in CellList mode just past its first pre_step
      with r_skin = 1.0 (so r_skin / 2 = 0.5)
    And reference positions captured at the last rebuild
    And at least one atom whose distance from its reference position exceeds 0.5
    When box.set_lattice is called bumping the generation, with the new lattice
      yielding the same n_cells_total as before
    And pre_step is called with the updated box
    Then pre_step downloads neighbor_status and observes 1u
    And state.needs_rebuild was set to true and a rebuild was performed
    And state.cached_generation equals the new box generation after the call

  @rq-3288a78c
  Scenario: NPT-style sequence of small barostat ticks rebuilds at the displacement-driven rate
    Given a NeighborListState in CellList mode with r_skin = 1.0 just past its
      first pre_step
    And a fixed reference set of atom positions
    When a sequence of K pre_step calls is issued, each preceded by a barostat
      that ticks the generation and scales the box by 1.0e-4 (so n_cells_total
      stays constant) and a small physical-coord atom drift
    Then the number of rebuilds performed across the K pre_step calls equals
      the number of pre_step calls on which the downloaded neighbor_status
      first reads 1u (the same set that would have triggered in a no-barostat
      NVT run with the same atom drifts)
    And no pre_step call triggers a rebuild solely from the generation tick

  @rq-72aae589
  Scenario: Generation tick with unchanged n_cells_total updates the cache without forcing rebuild
    Given a NeighborListState in CellList mode with an orthorhombic box
      lx=ly=lz=10.0 just past its first pre_step
      (n_cells_total = 343)
    And reference positions captured at the last rebuild
    And no atom has moved more than r_skin / 2 since the last rebuild
    When box.set_lattice(10.001, 10.001, 10.001, 0.0, 0.0, 0.0) is called
      (lattice barely changed; floor((10.001) / (r_cut + r_skin)) still equals the prior n_cells; generation bumps from 0 to 1)
    And pre_step is called with the updated box
    Then state.cached_generation equals 1 after the call
    And state.n_cells_total equals its prior value (343)
    And cell_offsets retains its prior device allocation
    And the displacement-check kernel was launched
    And state.needs_rebuild is false after pre_step returns
    And no rebuild was performed

  # --- Device-side spatial hash ---

  @rq-f164bf76
  Scenario: cell_indices is populated by the device pipeline
    Given a NeighborListState in CellList mode with n_cells_x=n_cells_y=n_cells_z=7
    And particles at positions that map to known cell indices c0, c1, c2, ...
    When NeighborListState::rebuild is called
    Then downloading cell_indices yields [c0, c1, c2, ...] for atoms [0, 1, 2, ...]

  @rq-19fd5b09
  Scenario: cell_counts is the device-computed per-cell histogram
    Given particles placed at positions producing cells [2, 0, 1, 0, 2]
    When NeighborListState::rebuild is called
    Then downloading cell_counts yields counts that sum to particle_count
    And cell_counts[0] equals 2 (particles 1 and 3)
    And cell_counts[1] equals 1 (particle 2)
    And cell_counts[2] equals 2 (particles 0 and 4)

  @rq-f8ad62d4
  Scenario: cell_offsets is the exclusive prefix sum and ends at particle_count
    Given a NeighborListState in CellList mode with N particles and
      arbitrary positions
    When NeighborListState::rebuild is called
    Then downloading cell_offsets yields a strictly non-decreasing sequence
    And cell_offsets[0] equals 0
    And cell_offsets[c+1] equals cell_offsets[c] + cell_counts[c] for every c
    And cell_offsets[n_cells_total] equals N

  @rq-265f4da4
  Scenario: scatter places each atom inside its cell's slice
    Given particles at positions producing cells [2, 0, 1, 0, 2]
    When NeighborListState::rebuild is called
    Then for every atom i, sorted_particle_ids contains i exactly once
    And the slot at which i appears falls in
      [cell_offsets[cell_indices[i]], cell_offsets[cell_indices[i]+1])

  @rq-7a14d0d8
  Scenario: per-cell sort canonicalises sorted_particle_ids
    Given particles placed at positions producing cells [2, 0, 1, 0, 2]
    When NeighborListState::rebuild is called
    Then sorted_particle_ids equals [1, 3, 2, 0, 4]
      (cell 0: particles 1 then 3; cell 1: particle 2; cell 2: particles 0 then 4)

  @rq-ecad9802
  Scenario: write_cursors is reset to zero before each rebuild
    Given a NeighborListState in CellList mode just past a rebuild whose
      write_cursors are populated (one count per cell that received atoms)
    When NeighborListState::rebuild is called a second time on identical
      positions
    Then sorted_particle_ids matches the first rebuild's output exactly
      (write_cursors did not accumulate across rebuilds)

  @rq-6c8415f6
  Scenario: NeighborListState rebuild performs no host-device particle transfers
    Given a NeighborListState in CellList mode
    When NeighborListState::rebuild is called
    Then no host-side download of posq occurs
    And no host-side upload of sorted_particle_ids or cell_offsets occurs

  @rq-6fd5167a
  Scenario: Configuration exceeding the u32 cell-addressing limit is rejected at construction
    Given r_cut + r_skin = 0.1 and lx=ly=lz=162.6 yielding
      n_cells_per_axis = 1626 (n_cells_total = 4298942376, just past u32::MAX)
    When NeighborListState::new_cell_list is called
    Then it returns Err(NeighborListError::TooManyCells { n_cells_total: 4298942376, max_supported })
      where max_supported equals u32::MAX as usize (4294967295)
    And no device buffer was allocated before the error was returned

  @rq-5f2c42be
  Scenario: Prefix scan is correct for cell counts beyond a single block-totals pass
    Given r_cut + r_skin = 0.05 yielding n_cells_per_axis = 200
      (n_cells_total = 8000000, well past scan_block_size² and requiring
      multiple recursion levels in the prefix scan)
    When NeighborListState::new_cell_list is called
    Then it returns Ok(state)
    When NeighborListState::rebuild is called
    Then downloading cell_offsets yields a non-decreasing sequence
    And cell_offsets[0] equals 0
    And cell_offsets[c+1] equals cell_offsets[c] + cell_counts[c] for every c
    And cell_offsets[n_cells_total] equals the particle count

  @rq-f2e4b0b8
  Scenario: Cell-list scratch is reallocated alongside cell_offsets on box generation refresh
    Given a NeighborListState in CellList mode with n_cells_total = 343
      (cell_counts length 343, write_cursors length 343,
       a scan_block_totals stack sized for n_cells_total = 343)
    When box.set_lattice is called producing n_cells_total = 729
    And pre_step is called with the updated box
    Then cell_counts is reallocated to length 729
    And write_cursors is reallocated to length 729
    And the scan_block_totals stack is rebuilt for n_cells_total = 729
    And cell_indices is NOT reallocated (its length particle_count is unchanged)

  @rq-2303ee2e
  Scenario: Per-cell sort yields ascending particle indices inside every cell
    Given any non-empty system
    When NeighborListState::rebuild is called
    Then for every cell c with occupancy k,
      sorted_particle_ids[cell_offsets[c] .. cell_offsets[c+1]] is a
      strictly ascending sequence of u32 particle indices
```
