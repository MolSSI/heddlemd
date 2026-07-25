# HeddleMD code review

_Date: 2026-07-25_

This is a carefully engineered, unusually well-documented codebase. Dependency
hygiene is clean, error handling is mostly `Result`-threaded with `thiserror`,
determinism is genuinely preserved everywhere checked (RNG keyed on stable
particle IDs, associative integer accumulation, fixed rotation orders), and the
registry/kernel-macro layer is real composition rather than special-casing.
There is **no crash-in-normal-use defect**. The issues worth attention are
silent-wrong-physics gaps, one architectural guarantee that is weaker than
advertised, and a documentation divergence that will actively mislead future
contributors.

The review covered six subsystems (CUDA kernels, forces, GPU host layer,
integrators/op-model, runner+IO+config, cross-cutting infra). Findings are
ranked by impact. Every headline item was verified against the source.

## Correctness / silent-wrong-physics (fix first)

**1. Constraints silently accept non-convergence — HIGH.**
`kernels/shake.cu:263-294` (and `rattle_velocities`, `shake_positions_no_velocity`,
iterative `settle.cu`) run a fixed `MAX_ITER = 32` loop with a local `converged`
flag, then **fall through and write back positions/velocities unconditionally**
(line 291 `break`s on convergence but line 294+ proceeds regardless). No status
is returned to the host; the wrappers never read anything back. A stiff cluster,
overlapping geometry, or too-large `dt` fails to converge, the step accepts
off-manifold positions, and the run continues producing wrong-but-non-NaN
dynamics with zero diagnostic. Violates the project's "do not assume happy
paths" rule.
_Fix: kernel writes a per-group/global non-convergence flag; host reduces it and
surfaces a `ConstraintError` (at least at log cadence)._

**2. Top-level config typos are silently dropped — HIGH.**
`src/io/config.rs:779` `RawConfig` lacks `#[serde(deny_unknown_fields)]`, though
every sibling struct right below it (e.g. `RawMinimizationConfig` at :814) has
it. So `[spmee]` silently disables electrostatics, `[neighbour_list]` drops a
custom `r_skin`, a misspelled `[[pair_interaction]]` runs with **no pair
forces** — all silently, all wrong physics, exit 0.
_Fix: one line, add `deny_unknown_fields` to `RawConfig`._

**3. JIT CUDA module names collide across `ForceField`s — HIGH (latent).**
`src/forces/jit_composed.rs:113,2023,2260,2480` load PTX under constant module
names (`heddle_jit_composed_pair_force`, ...). Two `ForceField`s on one device
(multi-replica, ensemble tooling, field rebuild) — the second `load_ptx` aliases
the first, and the first field's function handles silently point at the wrong
PTX -> wrong forces, no error. The SPME sibling in the same tree already solves
this (`src/forces/spme.rs:213-215`, `AtomicU64` -> `spme_jit_{id}`); these four
composers did not adopt it.
_Fix: mirror the SPME per-instance counter._

**4. Minimizer reports success on blown-up / no-op runs — MEDIUM-HIGH.**
`src/minimizer/steepest_descent.rs:204` uses `if !(f_max > 0.0)` — since
`NaN > 0.0` is false, a NaN max-force takes the "already at minimum" branch and
exits **converged (exit 0)**. Separately, atomic-unit runs that do not set
`initial_step` start at the SI-magnitude default `1e-12 a_0 << STEP_FLOOR
(1e-6)`, so minimization no-ops after one step and still reports success.
_Fix: branch on `f_max.is_finite()` first; make step defaults dimensionally sane
or validate against `STEP_FLOOR`._

**5. `r^2 -> 0` produces a UB fixed-point cast — MEDIUM.**
In the JIT pair loop `src/forces/jit_composed.rs:1716-1723`, two distinct
coincident atoms give `r2=0` -> `inv_r=rsqrt(0)=+inf` -> `r=0*inf=NaN` (the
`i_atom_id != j_atom_id` guard blocks self-interaction, not coincidence). That
NaN flows into `heddle_jit_real_to_fixed` (:1478) whose `(long long) scaled`
cast is **unsaturated** — `(long long)inf` is UB and corrupts the atom's
accumulator. The bonded/harmonic functors guard `r<1e-7`; the pair path does
not.
_Fix: floor `r2` and/or saturate before the cast._

**6. RDF mis-normalizes on variable-cell (NPT) trajectories — MEDIUM.**
`src/analysis/rdf.rs` fixes `self.volume` and the `r_max <= min_perp/2`
min-image guard from the **first frame only**, while `consume_frame` uses each
frame's own box. So `g(r)` is scaled by `V_frame/V_first`, and a later shrinking
box silently corrupts the histogram.
_Fix: accumulate per-frame volume; re-assert the min-image bound per frame._

**7. Missing minimum-image box-size validation for `mode="all-pairs"`.**
`src/runner/setup.rs` only runs the box-vs-cutoff check for `CellList`;
all-pairs still applies min-image, so a box `< 2*r_cut` silently yields wrong
forces (`src/runner/lint.rs` reports "not applicable"). Also:
`mtk_position_drift` (`kernels/mtk.cu:52-70`) is the lone position-mutating
kernel that does not `wrap_and_count_triclinic`, breaking the primary-image
invariant and MSD image counters for unconstrained MTK-NPT.

## The op-model guarantees less than architecture.md claims — MEDIUM (architecturally important)

The architecture positions the op-model's dependency validator as the thing that
guarantees "no operation may observe force-derived state that a preceding
position/box mutation invalidated." But **virial and potential-energy buffers
are not tracked resources** (`src/integrator/op_model.rs:16-24` tracks only
`Forces`/`ClassForces`). MTK's `VirReducePre/Post` declare
`OpFootprint::new(&[], &[])` (`src/integrator/mtk_npt.rs:357-362`) with a comment
admitting they read "the (untracked) per-particle virial buffer," and
`BarostatPoint` reads only `[Velocities, Box]`. So a plan that reduces a **stale
virial** after a drift but before the force eval would pass validation. The
built-in plans are ordered correctly, so it is latent — but it is exactly the
class of bug the validator exists to catch, and it gets more dangerous as new
pressure-coupled integrators are added. Relatedly, `Drift` writes
`[Positions, Images]` but MTK's `drift_box` also mutates `Box` (footprint
under-declaration, safe only by coincidence today).
_Fix: add `Virial`/`Energy` as derived resources and declare them honestly in
the reducing ops' footprints._

## Documentation divergence (will mislead every future contributor) — MEDIUM

`docs/architecture.md` is loaded as project instructions, so its accuracy is
load-bearing. The **central reproducibility mechanism is described
inaccurately**:

- Invariants #2/#3 state "**no `atomicAdd` is used**" and "fixed-topology
  warp-tree reduction ... **no atomics, no shared memory**." The actual pair
  path (`src/forces/jit_composed.rs:1489-1494`, `kernels/neighbor.cu:705,731`)
  packs neighbors into **unordered 32-atom tiles** and reduces via **`atomicAdd`
  on i64 fixed-point accumulators** scaled `2^48`, finalized by
  `finalize_packed_forces`. Determinism still holds — integer addition is
  associative regardless of atomic order — but the documented "no atomics /
  register-only warp butterfly" is not what runs. Anyone adding a potential per
  these docs will be badly misled.
- `architecture.md`'s file listing (`pair_force.cu`, `spme_real.cu`) does not
  match the tree (`forces.cu` is only the class-combine kernel; SPME is
  `spme_recip.cu` + `spme_spread_gather.cuh`).

_Fix: rewrite the reproducibility section to describe the packed-tile +
fixed-point-atomic scheme that actually provides determinism, and correct the
file map._ This is arguably the highest-leverage single change: the docs are
excellent everywhere else, which makes the one inaccurate section especially
trap-like.

## Scales worse as the codebase grows

- **Quadratic exclusion-list build — `src/forces/topology.rs:1178-1258`.** Each
  implicit-exclusion insertion does `push` then a full `effective.sort_by_key(...)`
  (sorts at lines 1190, 1206, 1227, 1256) to keep the `binary_search` valid ->
  O(M^2 * log M). A 100k-atom topology takes minutes-to-hours _at load_.
  _Fix: track covered pairs in a `HashSet`, collect into a `Vec`, sort once._
- **`src/io/config.rs` (2595 lines) has 5x-per-field change amplification** — the
  `Raw*`/public struct duplication means each new option touches ~5 hand-synced
  sites. `src/gpu/kernels.rs` (2891 lines) is a god-module with **53 hand-rolled
  `unsafe launch` blocks vs 17 macro uses**; each new kernel adds another ~40-line
  copy. `src/forces/jit_composed.rs` (3287) triplicates the bonded/angle/dihedral
  composers, and `morse/harmonic_bond/angle/dihedral.rs` are near-verbatim
  ~400-line copies. Adding a _new potential within an existing shape_ is clean;
  adding a _new shape_ (impropers, CMAP) touches ~6 places. The
  segmented-reduction CUDA body is copy-pasted ~8x, and `min_image`/`weighted_com`
  are duplicated verbatim between `shake.cu`/`settle.cu`.
  _These are the "neglect now, pay later" items: extract a generic intramolecular
  slot + composer, a shared reduction map/header, and widen the launch macro._
- **Registry silently shadows same-named builders — `src/registry.rs:152`.**
  `lookup` returns the first `kind_name` match and built-ins register first, so a
  user override with a colliding kind is unreachable with no diagnostic. Fine
  today; a footgun once plugins/multiple registrations appear.
  _Fix: reject duplicate kinds, or define + document last-wins._

## Robustness / smaller items

- Reductions hardcode `__shared__ Real partial[256]` but index off `blockDim.x`
  (`nose_hoover.cu`, `barostat.cu`, `minimize.cu`, `spme_recip.cu`) — a launch
  with `blockDim.x > 256` silently writes OOB. Add a static guard. Same shape:
  SHAKE/RATTLE dynamic-shared-mem budget is asserted only in a comment
  (`src/gpu/kernels.rs:2567,2695`).
- Trajectory reader silently defaults malformed `Step`/`Time` to 0
  (`src/io/trajectory.rs:471-481`); init-state parses an unbounded header count
  into ~11 `Vec::with_capacity` before reading rows
  (`src/io/init_state.rs:272-312`) -> OOM on a fat-fingered header.
  `validate_spme` computes `2*spline_order` before range-checking it
  (`src/io/config.rs:2443`).
- Dead code with latent hazards: `use_exclusion_bitmask` param is `let _ =`'d and
  its surrounding comments describe a design that does not exist
  (`src/forces/jit_composed.rs:1117-1137`); `philox_normal_real_pair`
  (`kernels/philox.cuh:97`) is uncalled and its rejection fallback would bias the
  distribution if wired up; the `ThermostatHalf` marker path is currently dead
  (no built-in emits it). Several stale comments (Andersen graph-compat
  `src/integrator/andersen.rs:239`, `coupling_interval` default doc
  `src/io/config.rs:1158`).
- `Box::leak` of entry-point names on every `ForceField::new`
  (`src/forces/jit_composed.rs:2061+`); pervasive unchecked `particle_count as
  u32` (unreachable N, but silent truncation rather than a checked error);
  `panic!`/`expect` in a few `Drop` and per-step hot paths
  (`src/gpu/graph.rs:177,261`; `src/forces/mod.rs` dispatch; `src/forces/spme.rs:1105`)
  that abort the process instead of propagating.

## Verified clean (for confidence)

Energy/virial share accounting (1/2, 1/3, 1/4), Newton's-3rd double-count
suppression, exclusion-correction math, determinism under neighbor reordering,
thermostat wrapping mutual-exclusion (double-guarded — no double-thermostatting),
constraint virial published exactly once, GPU kernel argument order/types (~15
kernels cross-checked against `.cu` signatures), empty-N guards, `div_ceil` grid
math, `f64<->double` ABI, packed-neighbor staging-buffer bounds, unit
round-trips, and no `TODO`/`FIXME`/unsafe-lint-suppression anywhere.

## Recommended priority

1. **Silent constraint non-convergence (#1)** and **`deny_unknown_fields` on
   `RawConfig` (#2)** — real wrong-physics, both cheap.
2. **Correct the architecture.md reproducibility section** — it is the contract,
   and it is wrong about atomics.
3. **JIT module-name collision (#3)** before any multi-field usage lands.
4. **Track virial in the op-model** — closes the gap between what the validator
   promises and enforces.
5. **Topology quadratic** and the **config/kernels/jit duplication** — the items
   that compound as systems and the potential catalog grow.
