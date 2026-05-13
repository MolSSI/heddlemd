# Feature: Shared Neighbor List Service <!-- rq-693ce6fa -->

The `ForceField` owns a single `NeighborListState` shared by every
short-range pair potential in the slot list. Each consuming potential
declares its maximum interaction cutoff through the `Potential` trait
(see `framework.md`); the framework aggregates these into one search
radius and builds a single neighbor list once per timestep cluster.
Per-pair cutoff filtering happens inside each consumer's kernel — the
shared list is sized to the largest cutoff so it captures every
candidate for every consumer.

The shared state exists in one of two construction modes determined by
the parsed `NeighborListConfig`:

- **`CellList`** — the spatial-hash-filtered list described in this
  file. Built lazily on the first step and rebuilt on demand when an
  atom's reference displacement exceeds `r_skin / 2`. The cell layout
  (number of cells per axis, cell size, total cell count) is cached at
  construction from the simulation box's edge lengths and refreshed
  whenever the box's `generation` counter changes (see *Box Generation
  Tracking* below).
- **`Trivial`** — every particle's neighbor list contains every
  particle (including itself; consumers handle the self slot). The list
  is materialised once at construction and never rebuilt. Used when the
  config selects `[neighbor_list].mode = "all-pairs"`.

When no slot in the `ForceField` reports a `max_cutoff` (a bonded-only
or zero-slot configuration), no `NeighborListState` is built at all and
no displacement-check kernel runs.

This file specifies both modes, the per-step displacement check and
rebuild policy that governs `CellList`, the trivial construction path
used by `Trivial`, and the `NeighborListState` API the framework drives.

## Cell Layout <!-- rq-dfad7218 -->

Cells partition the simulation box into a 3D grid. Per axis `a`:

```
target_cell_size_a = r_cut + r_skin
n_cells_a           = floor(L_a / target_cell_size_a)
actual_cell_size_a  = L_a / n_cells_a
```

The actual cell size is always `>= r_cut + r_skin` because `n_cells_a` is
rounded down. The 27-cell PBC search visits every adjacent cell exactly
once iff `n_cells_a >= 3`. Configurations that violate this on any axis
are rejected at config-load time (see `io/config-schema.md`).

`n_cells_total = n_cells_x * n_cells_y * n_cells_z`. The cell index of
position `(x, y, z)` is `cell_x * n_cells_y * n_cells_z + cell_y *
n_cells_z + cell_z` (row-major). Per-axis cell indices are computed from
the minimum-image-wrapped position:

```
wrapped_x = x - L_x * floor((x + L_x / 2) / L_x)
cell_x    = floor((wrapped_x + L_x / 2) / actual_cell_size_x)
```

with `cell_x` clamped to `[0, n_cells_x - 1]` to handle the
`wrapped_x = +L_x / 2` boundary case (mapped to `cell_x = n_cells_x - 1`
rather than `n_cells_x`).

## Cell List Construction <!-- rq-a060036e -->

A cell-list rebuild produces three host-side and three device-side
artifacts:

1. **Per-particle cell indices** (host-side, transient).
2. **`sorted_particle_ids: Vec<u32>` of length N** (device-uploaded). The
   particle IDs sorted lexicographically by `(cell_index, particle_id)`;
   the secondary sort key gives a deterministic in-cell ordering.
3. **`cell_offsets: Vec<u32>` of length `n_cells_total + 1`** (device-
   uploaded). `cell_offsets[c]` is the index in `sorted_particle_ids`
   where cell `c`'s particles begin; `cell_offsets[c+1] -
   cell_offsets[c]` is the cell's occupancy.

The sort is performed on the host using `slice::sort_by_key`, which is
stable. After sorting, the host downloads particle positions, walks them
in order, and tallies cell occupancy into `cell_offsets`. Total cost:
one position download (3 N floats), an O(N log N) sort over u32 indices,
and one upload (N + n_cells_total + 1 u32 values).

## Neighbor List Construction <!-- rq-33aa3e1d -->

After the cell list is uploaded, one device kernel builds the per-atom
neighbor list. The kernel maps one thread to each atom `i`:

1. Compute atom `i`'s cell `(cx, cy, cz)` from its current position.
2. Iterate the 27 adjacent cells `(cx + dx, cy + dy, cz + dz)` for
   `dx, dy, dz` in `{-1, 0, +1}`, wrapping each per-axis cell index modulo
   `n_cells_a`.
3. For each visited cell `c'`, walk `sorted_particle_ids[cell_offsets[c']
   .. cell_offsets[c' + 1]]` in order. For each candidate partner `j`,
   skip if `j == i`, otherwise compute the minimum-image displacement
   between `i` and `j`. If `r² <= (r_cut + r_skin)²`, append `j` to
   atom `i`'s neighbor list.
4. After all 27 cells are walked, sort atom `i`'s neighbor list in place
   by partner index (an insertion sort over `<= max_neighbors` entries).

The cell iteration order, the in-cell particle order, and the per-atom
insertion sort are all deterministic given the inputs. No atomics are
used.

The neighbor list is stored as:

- `neighbor_list: CudaSlice<u32>` of length `N * max_neighbors`. Slot
  `i * max_neighbors + k` is the `k`-th partner of atom `i`.
- `neighbor_counts: CudaSlice<u32>` of length `N`. Entry `i` is atom
  `i`'s neighbor count (`<= max_neighbors`).

When an atom's neighbor count would exceed `max_neighbors`, the kernel
sets an overflow flag in a device-side scalar and ceases appending for
that atom. The host detects the flag after the kernel returns and
returns `NeighborListOverflow { max: max_neighbors }` from
`NeighborListState::rebuild` (see *Feature API* below).

## Displacement Check <!-- rq-1f38d78a -->

Every timestep (after the integrator's pre-force step has updated
positions), the runner runs a *displacement-check* kernel:

1. Per atom `i`, compute `disp² = (x_i - x_i_ref)² + (y_i - y_i_ref)² +
   (z_i - z_i_ref)²` using the minimum-image-wrapped displacement (so an
   atom that moved through a PBC boundary still reports a small
   displacement rather than `L - epsilon`).
2. Write `disp²` to a per-atom buffer of length `N`.

The host downloads the per-atom buffer (one f32 each), computes
`max_disp = sqrt(max_i disp²_i)` in f64, and compares against
`r_skin / 2`. If `max_disp > r_skin / 2`, the host sets a
`needs_rebuild` flag.

`x_i_ref` / `y_i_ref` / `z_i_ref` are reference positions captured at
the time of the last rebuild and held in three device buffers of
length `N`. They are updated immediately after each rebuild to the
positions used during that rebuild.

Host-side max instead of a device max-reduction keeps the implementation
small for v1; the per-step download is one f32 per particle (40 KB at
N = 10000, ~4 µs over PCIe), well below the rebuild interval cost the
displacement check is amortising.

## Box Generation Tracking <!-- rq-282af621 -->

In `CellList` mode the cell layout (`n_cells`, `cell_size`, `n_cells_total`)
depends on the simulation box's edge lengths and is therefore cached. The
state records the box's `generation()` value at the moment the cache was
populated; this is stored as the field `cached_generation: u64`. The state
also stores the `r_cut` value used to derive that cache so the cache can be
refreshed without re-querying every consumer.

Every call to `NeighborListState::pre_step` receives the runner's current
`&SimulationBox` and compares `sim_box.generation()` against
`cached_generation`. On a match no cache work is done. On a mismatch the
state:

1. Recomputes `n_cells_a = floor(L_a / (r_cut + r_skin))` per axis from the
   current box's edge lengths.
2. Re-validates that `n_cells_a >= 3` on every axis. If any axis fails,
   returns `Err(NeighborListError::BoxTooSmallForCells { axis, length,
   required })` without mutating any cached field; the caller can rerun
   the previous step's force evaluation with the prior box or abort.
3. Recomputes `cell_size_a = L_a / n_cells_a` per axis and the scalar
   `n_cells_total`.
4. Reallocates the device `cell_offsets` buffer to length
   `n_cells_total + 1` if `n_cells_total` differs from the previous value.
   Other device buffers (`neighbor_list`, `neighbor_counts`,
   `reference_positions_*`, `disp_sq`, `overflow_flag`) are sized by
   `particle_count` or are scalar and are not reallocated.
5. Stores the new `n_cells`, `cell_size`, `n_cells_total`, and replaces
   `cached_generation` with `sim_box.generation()`.
6. Sets `needs_rebuild = true`. The rebuild that fires after the cache
   refresh uses the new cell layout, so the displacement check is skipped
   on a generation-mismatch step (current cell bins are stale by
   construction).

`r_search_sq` (`(r_cut + r_skin)²`) does not depend on the box and is left
in place across refreshes.

In `Trivial` mode the cell layout is not used; `pre_step` ignores the box's
generation and does no per-step work.

## Rebuild Policy <!-- rq-6e11554f -->

The runner holds one host-side `bool` flag `needs_rebuild`. Its initial
value is `true` so the warm-up force evaluation triggers the first
rebuild.

Per timestep, after the integrator's pre-force step:

1. If `sim_box.generation() != cached_generation`, refresh the cell-layout
   cache (see *Box Generation Tracking*). This sets `needs_rebuild = true`
   and skips the displacement-check kernel for this step. The refresh may
   return `BoxTooSmallForCells`, in which case `pre_step` aborts and the
   error propagates.
2. Otherwise, run the displacement-check kernel, download the per-atom
   buffer, compute `max_disp`, and set `needs_rebuild = true` if
   `max_disp > r_skin / 2`. The displacement check is skipped when
   `needs_rebuild` is already true.
3. If `needs_rebuild`:
   a. Download current positions to host.
   b. Compute cell indices and sort.
   c. Upload `sorted_particle_ids` and `cell_offsets`.
   d. Run the neighbor-list-build kernel.
   e. Check the overflow flag; fail-loud if set.
   f. Copy current positions into the reference-positions buffers.
   g. Set `needs_rebuild = false`.
4. Run downstream contribution kernels (see `framework.md`), which read
   the neighbor list.

## Configuration <!-- rq-267941a2 -->

Selected from the config's optional `[neighbor_list]` table; see
`io/config-schema.md` for the schema. Summary:

- `mode: String` — `"cell-list"` (default) or `"all-pairs"`.
- For `mode = "cell-list"`:
  - `max_neighbors: u64` — required. Strictly positive.
  - `r_skin: f64` — optional, defaults to `0.3 * max_cutoff` where
    `max_cutoff` is the largest cutoff reported by any
    neighbor-list-consuming potential. Strictly positive.

Validation at config load:

- `r_skin > 0` and finite.
- `max_neighbors > 0`.
- For every cutoff `c` reported by a consuming potential, the smallest
  box edge satisfies `L_min >= 3 * (c + r_skin)`. The init file holds
  the box; validation therefore happens after the init file is loaded
  and the effective `r_skin` and `max_cutoff` are known.
- In `mode = "all-pairs"`, `max_neighbors` and `r_skin` are rejected.

The maximum cutoff used to size the neighbor list is the largest
`max_cutoff` reported by any consuming potential in the `ForceField`
slot list (see `framework.md`). With one or more consumers, the
neighbor search radius is `max_cutoff + r_skin` in cell-list mode. The
trivial mode does not use a search radius.

## Empty State <!-- rq-5cbab27f -->

When `particle_count == 0`:

- Cell-list construction produces empty `sorted_particle_ids` and
  `cell_offsets` of length `n_cells_total + 1` filled with zeros.
- The neighbor-list build kernel does not launch.
- The displacement-check kernel does not launch.
- `needs_rebuild` stays `true` forever but no rebuild work happens.
- Trivial construction produces empty `neighbor_list` and
  `neighbor_counts` buffers.

When `particle_count == 1`:

- Cell-list construction works trivially.
- The neighbor-list build kernel runs for one atom and finds zero
  partners.
- Trivial construction produces a single-element `neighbor_counts`
  buffer with value `1`, and a single-element `neighbor_list` buffer
  containing the self index `0`. Consumers handle the self slot at
  evaluation time (the LJ kernel zeroes self slots).

When the `ForceField` has zero neighbor-list-consuming potentials, no
`NeighborListState` is built; the framework's `Option<NeighborListState>`
is `None` and no displacement-check or build kernel runs over the
lifetime of the run.

## Feature API <!-- rq-3e744fed -->

### Types <!-- rq-ad7eb40f -->

- `NeighborListConfig` — value of the parsed `[neighbor_list]` table. <!-- rq-060b1fab -->
  Variants:
  - `AllPairs`
  - `CellList { max_neighbors: u32, r_skin: f64 }`

- `NeighborListState` — host-side wrapper carrying the device buffers <!-- rq-b2d68288 -->
  and parameters that make up the shared neighbor list. The state is in
  one of two modes, fixed at construction.

  Fields present in both modes:
  - `device: Arc<CudaDevice>`
  - `particle_count: usize`
  - `max_neighbors: u32`
  - `neighbor_list: CudaSlice<u32>` (length `N * max_neighbors`)
  - `neighbor_counts: CudaSlice<u32>` (length `N`)
  - `mode: NeighborListMode` — discriminator (`Trivial` or `CellList`).

  Fields present only in `CellList` mode:
  - `n_cells: [u32; 3]`
  - `cell_size: [f32; 3]`
  - `n_cells_total: usize`
  - `r_cut: f32` — the largest `Potential::max_cutoff()` value reported by
    a consumer, captured at construction. Stored so the cache-refresh
    path can recompute `n_cells` and `cell_size` from a mutated box.
  - `r_skin: f32`
  - `r_search_sq: f32` — pre-computed `(r_cut + r_skin)²` for the build
    kernel. Independent of the box; not refreshed on generation change.
  - `cached_generation: u64` — the box's `generation()` value at the time
    the cell-layout cache was populated. Compared against the live box's
    generation on every `pre_step`; a mismatch refreshes the cache (see
    *Box Generation Tracking*).
  - `sorted_particle_ids: CudaSlice<u32>` (length `N`)
  - `cell_offsets: CudaSlice<u32>` (length `n_cells_total + 1`)
  - `reference_positions_x/y/z: CudaSlice<f32>` (length `N`)
  - `disp_sq: CudaSlice<f32>` (length `N`) — scratch for the
    displacement-check kernel.
  - `overflow_flag: CudaSlice<u32>` (length `1`) — set non-zero by the
    build kernel when any atom exceeds `max_neighbors`.
  - `needs_rebuild: bool` — initial value `true`.

  In `Trivial` mode the cell-list-specific fields are absent; the
  `neighbor_list` and `neighbor_counts` buffers are populated once at
  construction and never modified.

- `NeighborListMode` — discriminator. Variants: <!-- inline --> <!-- inline --> <!-- rq-ff424773 -->
  - `Trivial`
  - `CellList`

- `NeighborListError` — error type. Variants: <!-- rq-d8e4407a -->
  - `Gpu(GpuError)` — CUDA driver / kernel-launch failure.
  - `NeighborListOverflow { max: u32 }` — the build kernel detected an
    atom whose neighbor count would exceed `max_neighbors`. The
    simulation halts; the user must raise `max_neighbors` or reduce
    density.
  - `BoxTooSmallForCells { axis: &'static str, length: f32, required: f32 }`
    — the simulation box is smaller than `3 * (r_cut + r_skin)` along
    `axis`. Detected at `NeighborListState::new` time.

### Functions <!-- rq-3553aab2 -->

- `NeighborListState::new_cell_list(device: Arc<CudaDevice>, sim_box: &SimulationBox, particle_count: usize, r_cut: f32, max_neighbors: u32, r_skin: f32) -> Result<NeighborListState, NeighborListError>` <!-- rq-14033af1 -->
  - Constructs a `CellList`-mode state.
  - Computes `n_cells` per axis from `floor(L_axis / (r_cut + r_skin))`.
  - Returns `BoxTooSmallForCells` if any axis has `n_cells < 3`.
  - Allocates every device buffer described in the *CellList*-mode field
    list. Reference positions start at zero; `needs_rebuild` starts at
    `true`.
  - Stores `r_cut` so the cache-refresh path (see *Box Generation
    Tracking*) can recompute `n_cells` and `cell_size` from a mutated
    box. `r_cut` is the largest cutoff across every consumer of the
    shared list; the framework computes this as the maximum of every
    `Potential::max_cutoff()` value it observes.
  - Records `cached_generation = sim_box.generation()`.

- `NeighborListState::new_trivial(device: Arc<CudaDevice>, sim_box: &SimulationBox, particle_count: usize) -> Result<NeighborListState, NeighborListError>` <!-- inline --> <!-- rq-c96fd9d2 -->
  - Constructs a `Trivial`-mode state. The `sim_box` argument is accepted
    for API uniformity; `Trivial` mode does not consult it.
  - `max_neighbors = particle_count`.
  - Allocates `neighbor_list` of length `particle_count *
    particle_count` and `neighbor_counts` of length `particle_count`.
  - Fills the buffers on the host so that
    `neighbor_list[i * particle_count + k] == k` for every `(i, k)` in
    `[0, particle_count) × [0, particle_count)`, and
    `neighbor_counts[i] == particle_count` for every `i`. Uploads both
    buffers once.
  - When `particle_count == 0`, both buffers have length zero.

- `NeighborListState::displacement_check(&mut self, sim_box: &SimulationBox, buffers: &ParticleBuffers, timings: &mut Timings) -> Result<f32, NeighborListError>` <!-- rq-c49b2fe6 -->
  - Launches the displacement-check kernel against current positions and
    the stored reference positions, using `(lx, ly, lz)` from `sim_box`
    for the minimum-image PBC wrap.
  - Downloads the per-atom buffer and returns the maximum displacement.
  - Returns `0.0` when `particle_count == 0`.
  - Returns `0.0` when the state is in `Trivial` mode (no rebuild
    machinery exists).

- `NeighborListState::rebuild(&mut self, sim_box: &SimulationBox, buffers: &ParticleBuffers, timings: &mut Timings) -> Result<(), NeighborListError>` <!-- rq-7db97132 -->
  - Performs the rebuild pipeline described in *Cell List Construction*
    and *Neighbor List Construction*, using `(lx, ly, lz)` from `sim_box`
    for cell-binning and the build kernel's PBC parameters. Updates
    reference positions.
  - Returns `NeighborListOverflow` when the build kernel set the
    overflow flag.
  - Returns `Ok(())` immediately when `particle_count == 0` or when the
    state is in `Trivial` mode.

- `NeighborListState::pre_step(&mut self, sim_box: &SimulationBox, buffers: &ParticleBuffers, timings: &mut Timings) -> Result<(), NeighborListError>` <!-- rq-1217c816 -->
  - Called by `ForceField::step` once per timestep before any slot's
    `contribute` runs. In `CellList` mode:
    1. Compares `sim_box.generation()` against `cached_generation`. On
       mismatch refreshes the cell-layout cache (see *Box Generation
       Tracking*), sets `needs_rebuild = true`, and skips the
       displacement check for this step. May return
       `BoxTooSmallForCells`.
    2. Otherwise, if `!needs_rebuild`, runs the displacement check and
       sets `needs_rebuild = true` if `max_disp > r_skin / 2`.
    3. If `needs_rebuild`, runs the rebuild and clears the flag.
    In `Trivial` mode this is a no-op.

### CUDA Kernels <!-- rq-0469400b -->

`kernels/neighbor.cu` declares three `extern "C"` kernels:

```c
extern "C" __global__ void neighbor_displacement_squared(
    const float *positions_x, const float *positions_y, const float *positions_z,
    const float *reference_x, const float *reference_y, const float *reference_z,
    float lx, float ly, float lz,
    float *disp_sq,
    unsigned int n);

extern "C" __global__ void neighbor_list_build(
    const float *positions_x, const float *positions_y, const float *positions_z,
    const unsigned int *sorted_particle_ids,
    const unsigned int *cell_offsets,
    float lx, float ly, float lz,
    float cell_size_x, float cell_size_y, float cell_size_z,
    unsigned int n_cells_x, unsigned int n_cells_y, unsigned int n_cells_z,
    float r_search_sq,
    unsigned int max_neighbors,
    unsigned int *neighbor_list,
    unsigned int *neighbor_counts,
    unsigned int *overflow_flag,
    unsigned int n);

extern "C" __global__ void copy_positions_into_reference(
    const float *positions_x, const float *positions_y, const float *positions_z,
    float *reference_x, float *reference_y, float *reference_z,
    unsigned int n);
```

The first writes per-atom minimum-image squared displacements. The
second walks 27 cells, builds the per-atom neighbor list, and
in-place-sorts it by partner index using insertion sort. The third
copies positions to reference positions; called after every rebuild.

### Rust Launch Helpers <!-- rq-fec7ae1c -->

Three free functions in `src/gpu/kernels.rs`, re-exported from
`crate::gpu`:

- `neighbor_displacement_squared(particle_buffers, reference_x, reference_y, reference_z, sim_box, disp_sq) -> Result<(), GpuError>` <!-- rq-884b5cd6 -->
- `neighbor_list_build(particle_buffers, sorted_particle_ids, cell_offsets, sim_box, n_cells, cell_size, r_search_sq, max_neighbors, neighbor_list, neighbor_counts, overflow_flag) -> Result<(), GpuError>` <!-- rq-a1262872 -->
- `copy_positions_into_reference(particle_buffers, reference_x, reference_y, reference_z) -> Result<(), GpuError>` <!-- rq-344f7af0 -->

Each is a no-op when `particle_count == 0`.

## Launch Configuration <!-- rq-2e15fed7 -->

- Block size: 256 threads (all three kernels).
- Grid: `ceil(n / 256)`.
- Shared memory: zero.
- Stream: the default stream carried by `particle_buffers.device`.

## Determinism <!-- rq-c62bb861 -->

Two-sort approach:

1. **Sort 1 — particles within each cell.** The host-side stable sort on
   `(cell_index, particle_id)` ensures that, when the build kernel
   walks a cell's particles, the order is independent of any
   non-determinism in cell assignment. **Required for run-to-run
   reproducibility.**

2. **Sort 2 — per-atom neighbor list by partner index.** The build
   kernel's trailing insertion sort imposes a canonical ascending order
   on each atom's neighbor list. **Not required** for run-to-run
   reproducibility (sort 1 already guarantees identical orderings
   across runs with identical inputs on the same GPU), but provides:
   - A canonical neighbor-list contents/order independent of cell
     decomposition, useful for testing.
   - Stability under future cell-layout changes (different `r_skin`,
     different cell size).
   - Insurance against subtle regressions in sort 1.

Future feature work may drop sort 2 if it becomes a measurable cost at
very large N. Doing so does not weaken the project's bit-exact
guarantee; it only forfeits the canonical-ordering testability.

## Performance Notes <!-- rq-54a28837 -->

- Cell-list rebuild dominant cost: one position download (3 N f32
  values), one host sort of N u32 indices, one upload of `N + n_cells +
  1` u32 values, one kernel launch. Typical cost at N = 10⁴ is below
  10 ms; rebuild interval is typically 10–100 timesteps so per-step
  amortised cost is well under 1 ms.
- Displacement check: one f32 per atom downloaded each step, a host max
  reduction. Sub-ms at N = 10⁴.
- The neighbor-list build kernel walks ~27 × density particles per atom
  (about 100–200 per atom for liquid-density systems); per-step force
  evaluation drops from `O(N²)` to `O(N · avg_neighbors)`.

## Out of Scope <!-- rq-58acf788 -->

- Device-side parallel cell sort (radix sort on GPU). Future work when
  host-side sort time becomes a bottleneck; the algorithm is identical
  so the spec moves over unchanged.
- Device-side parallel displacement-max reduction. Future work when the
  N-length per-step download becomes a bottleneck.
- Auto-growing `max_neighbors` on overflow. The current v1 contract is
  fail-loud and require the user to raise the value.
- Coulomb-style long-range force, which would require a non-cell-list
  decomposition (Ewald / PME). The neighbor-list framework here is
  short-range only.
- Half-neighbor-list (Newton's-third-law optimisation that lists each
  pair once instead of twice). Doubles the build complexity and would
  also force a different reduction strategy.
- Constant or adaptive `r_skin`. v1 is constant.
- Triclinic boxes. v1 cell layout assumes orthorhombic.
- Sort 2 (per-atom neighbor-list sort) being optional in v1 — the v1
  implementation always sorts. A future feature may make it
  conditional, but doing so is its own decision-point and is not
  included here.
- Per-pair-of-consumers cutoff filtering inside the neighbor-list build
  itself. The shared list is built once at the maximum cutoff across
  consumers; each consumer applies its own per-pair cutoff at force
  evaluation time, reading the list but discarding entries beyond its
  own cutoff.
- Detecting a box mutation that bypasses `SimulationBox::set_lengths`
  (e.g. two `SimulationBox` values constructed independently that happen
  to share a `generation`, or a future API that mutates lengths without
  bumping the counter). The generation counter is the contract;
  consumers trust the runner to own one canonical box and to use only
  the documented mutator.

---

## Gherkin Scenarios <!-- rq-c4645fa6 -->

```gherkin
Feature: Cell-list neighbor list

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called
    And a SimulationBox lx=ly=lz=10.0
    And a particle count of 100

  # --- Cell layout ---

  @rq-c0cfc5d6
  Scenario: Cell counts are floor(L / (r_cut + r_skin))
    Given r_cut = 1.0, r_skin = 0.3
    When NeighborListState::new is called with lx=ly=lz=10.0
    Then n_cells equals [7, 7, 7]
    And actual cell sizes are 10.0 / 7 along each axis

  @rq-1b9c474c
  Scenario: Reject configurations whose box admits fewer than 3 cells per axis
    Given r_cut = 1.0, r_skin = 3.0 (so r_cut + r_skin = 4.0)
    And lx = 10.0 (giving floor(10/4) = 2 < 3)
    When NeighborListState::new is called
    Then it returns Err(NeighborListError::BoxTooSmallForCells { axis: "x", length: 10.0, required: 12.0 })

  @rq-151cb099
  Scenario: Cell index of a position at the +L/2 boundary clamps inside the grid
    Given a particle at x = +lx/2 (boundary case)
    When its cell index is computed
    Then cell_x equals n_cells_x - 1 (no out-of-bounds index)

  @rq-a99ca751
  Scenario: Cell index of a position outside the primary cell wraps before binning
    Given a particle at x = lx (one full period past the primary cell)
    When its cell index is computed
    Then it equals the cell index of x = 0

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

  @rq-2bc559ec
  Scenario: Each atom's neighbor list is sorted by partner index after build
    Given any non-empty system
    When NeighborListState::rebuild is called
    Then for every atom i, neighbor_list[i * max_neighbors .. i * max_neighbors + neighbor_counts[i]]
      is a strictly ascending sequence of u32 partner indices

  @rq-0181787c
  Scenario: Build kernel signals overflow when an atom exceeds max_neighbors
    Given a dense configuration where atom 0 has 257 partners within r_cut + r_skin
    And max_neighbors = 256
    When NeighborListState::rebuild is called
    Then it returns Err(NeighborListError::NeighborListOverflow { max: 256 })

  @rq-6bf3709c
  Scenario: Two independent rebuilds with identical positions produce identical lists
    Given two NeighborListStates built from identical configurations and identical positions
    When each is rebuilt
    Then their neighbor_list, neighbor_counts, sorted_particle_ids, and cell_offsets agree byte-for-byte

  # --- Displacement check ---

  @rq-53ae77a4
  Scenario: Displacement check on reference positions equal to current returns 0
    Given a NeighborListState immediately after a rebuild
    When displacement_check is called
    Then it returns 0.0 (to within f32 round-off)

  @rq-b39d3be7
  Scenario: Displacement check uses minimum-image displacement
    Given a particle whose reference position is x = lx/2 - 0.05
    And whose current position is x = -lx/2 + 0.05 (wrapped across the boundary)
    When displacement_check is called
    Then the reported max displacement is approximately 0.1, not lx - 0.1

  @rq-f94ee5cd
  Scenario: Displacement check returns the maximum across all particles
    Given particle 7 has moved 0.5 from its reference and all others have moved less
    When displacement_check is called
    Then the result equals 0.5

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
    Then neighbor_counts[0] equals 0

  # --- Determinism ---

  @rq-4b40604b
  Scenario: Cell-list mode and trivial mode produce identical forces (within f32 tolerance)
    Given two ForceField instances with identical particle positions and parameters,
      one in mode = "cell-list" with r_skin = 0.3, the other in mode = "all-pairs"
    When both run a single force evaluation
    Then the resulting forces_* agree componentwise within 1e-4 relative error

  # --- Trivial mode ---

  @rq-789fcec9
  Scenario: Trivial-mode contents
    Given a NeighborListState built via new_trivial with particle_count = 3
    When neighbor_list and neighbor_counts are downloaded
    Then neighbor_counts equals [3, 3, 3]
    And neighbor_list equals [0, 1, 2, 0, 1, 2, 0, 1, 2]

  @rq-bb3773aa
  Scenario: Trivial-mode pre_step does no work
    Given a NeighborListState in Trivial mode
    When pre_step is called
    Then timings report zero samples for KernelStage::NeighborDisplacementSquared
    And timings report zero samples for KernelStage::NeighborListBuild

  @rq-30f85829
  Scenario: Trivial-mode state has no cell-list fields
    Given a NeighborListState built via new_trivial
    Then state.mode equals NeighborListMode::Trivial
    And the cell-list-specific buffers (sorted_particle_ids, cell_offsets,
      reference_positions_*, disp_sq, overflow_flag) are absent

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
    Then n_cells, cell_size, n_cells_total are unchanged
    And cell_offsets is not reallocated
    And state.cached_generation is unchanged

  @rq-cf847c1f
  Scenario: Box generation increment refreshes cell layout and forces a rebuild
    Given a NeighborListState in CellList mode immediately after its first pre_step
      with lx=ly=lz=10.0 and r_cut + r_skin = 1.3 (so n_cells = [7, 7, 7])
    When box.set_lengths(20.0, 20.0, 20.0) is called (generation 0 → 1)
    And pre_step is called with the updated box
    Then state.n_cells equals [15, 15, 15] (floor(20.0 / 1.3) = 15)
    And state.cached_generation equals box.generation() after the call
    And state.needs_rebuild was set to true and a rebuild was performed in
      the same pre_step call
    And the displacement-check kernel was not launched during this pre_step

  @rq-dacb071c
  Scenario: Generation mismatch with new box too small returns BoxTooSmallForCells
    Given a NeighborListState in CellList mode with r_cut + r_skin = 1.3
    When box.set_lengths(3.0, 10.0, 10.0) is called (floor(3.0 / 1.3) = 2 < 3)
    And pre_step is called with the updated box
    Then pre_step returns Err(NeighborListError::BoxTooSmallForCells { axis: "x", length: 3.0, required: 3.9 })
    And state.cached_generation is left unchanged
    And state.n_cells, state.cell_size, state.n_cells_total are left unchanged
    And cell_offsets is not reallocated

  @rq-d22f105f
  Scenario: cell_offsets is reallocated when n_cells_total changes
    Given a NeighborListState in CellList mode with n_cells = [10, 10, 10]
      (n_cells_total = 1000, cell_offsets length 1001)
    When box.set_lengths is called producing n_cells = [11, 11, 11]
      (n_cells_total = 1331)
    And pre_step is called with the updated box
    Then cell_offsets is reallocated to length 1332

  @rq-331b6e81
  Scenario: cell_offsets is not reallocated when n_cells_total is unchanged
    Given a NeighborListState in CellList mode with n_cells = [10, 10, 10]
    When box.set_lengths is called producing n_cells = [10, 10, 10]
      (different lengths but same n_cells_total)
    And pre_step is called with the updated box
    Then cell_offsets retains its previous device allocation (length 1001)
    And state.cell_size is updated to the new per-axis values

  @rq-31a9e3bb
  Scenario: r_search_sq is preserved across a generation refresh
    Given a NeighborListState in CellList mode with r_cut = 1.0 and r_skin = 0.3
    When box.set_lengths is called bumping the generation
    And pre_step is called with the updated box
    Then state.r_search_sq still equals 1.69 (i.e. (1.0 + 0.3)²)

  @rq-699cccff
  Scenario: Two pre_steps after a single box mutation refresh only once
    Given a NeighborListState in CellList mode
    When box.set_lengths bumps the generation once
    And pre_step is called, refreshing the cache and rebuilding
    And pre_step is called again without any further box mutation
    Then the second pre_step performs no cell-layout recompute
    And the second pre_step runs the displacement check (no longer skipped)

  @rq-72aae589
  Scenario: Generation mismatch is detected even when the box edge lengths
    are unchanged
    Given a NeighborListState in CellList mode with lx=ly=lz=10.0 just past
      its first pre_step
    When box.set_lengths(10.0, 10.0, 10.0) is called (same lengths but
      generation bumps from 0 to 1)
    And pre_step is called with the updated box
    Then state.cached_generation equals 1 after the call
    And state.needs_rebuild was set to true and a rebuild was performed
```
