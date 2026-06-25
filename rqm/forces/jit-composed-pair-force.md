# Feature: JIT-Composed Pair-Force Kernel <!-- rq-9f309378 -->

Every fast-class pair-force slot in a `ForceField` contributes its
per-pair functional form as a CUDA source fragment. At
`ForceField::new` time the framework concatenates the active
fragments, JIT-compiles the result via `nvrtc`, and obtains a single
composed pair-force kernel that walks each particle's neighbour list
once and accumulates every active slot's contribution into one
register set per particle. The per-step force-evaluation pipeline
launches this single composed kernel in place of one kernel per
fast-class pair-force slot.

A *fast-class pair-force slot* is any slot whose `frequency_class()`
returns `ForceClass::Fast` AND whose `max_cutoff()` returns
`Some(_)` — that is, a slot whose contribution is evaluated by
walking the shared `NeighborListState`'s neighbour list with the
warp-per-particle pattern of `pair-force-kernel.md`. Bonded slots
(`max_cutoff() == None`), the slow-class SPME reciprocal slot, and
any other non-pair-force slot continue to dispatch via their own
`Potential::compute` kernel call at step time and are not visible to
the JIT path.

The composed kernel is per-`ForceField` and outlives a step. It is
JIT-compiled once at construction, loaded as a CUDA module, and held
for the `ForceField`'s lifetime. The `ForceField`'s slot list still
contains the constituent fast-class pair-force slots in canonical
order; the framework simply bypasses each such slot's
`Potential::compute` at step time and runs the composed kernel
instead.

This file specifies the composition mechanism, the source-fragment
contract a slot must satisfy to participate, the composed kernel's
structure, the framework's compile + launch contract, and the
determinism guarantees the composed kernel preserves.

## Slot Participation <!-- rq-ed6e49c5 -->

Every fast-class pair-force slot exposes a CUDA source fragment via
its builder's `pair_force_fragment(cx)` method (see
`framework.md`'s *Feature API*). The framework collects every
`Some(_)` return in canonical slot order during `ForceField::new`'s
construction pass and feeds the collected fragments to the composer.

A slot whose `frequency_class()` is `Fast` and whose `max_cutoff()`
is `Some(_)` but whose `pair_force_fragment(cx)` returns `None` is
rejected at construction with
`ForceFieldError::MissingPairForceFragment { label }`. A slot that
reports `max_cutoff() == None` (a bonded slot, an angle slot, an
intramolecular slot, any slot that does not consume the neighbour
list) does not participate in the composition and is not consulted
for a fragment.

The slow-class `SpmeReciprocal` slot does not participate. Its
compute path is the SPME reciprocal pipeline (`spme.md`), which is
unaffected by the JIT-composition mechanism.

## Source-Fragment Contract <!-- rq-18382cc2 -->

A `PairForceFragment` carries a self-contained CUDA C++ snippet plus
identifying metadata. The snippet:

1. Defines a stateless `__device__` functor whose name is given by
   the fragment's `functor_struct_name`. The functor exposes three
   member functions:

   ```c
   struct <functor_struct_name> {
       // Per-pair cutoff^2 test. The composed kernel calls this
       // only for fragments whose cutoff is declared
       // CutoffHandling::PerPair; Uniform fragments never have
       // this function called from the composed kernel (the outer
       // loop's max-cutoff mask covers them).
       __device__ inline Real cutoff_squared(unsigned int i, unsigned int j) const;

       // Per-pair functional form. Writes the three pair outputs.
       // `r2`, `inv_r = 1/r`, and `r` are computed once per pair
       // by the outer loop and threaded into every fragment's
       // evaluate so they share work across fragments. `qi` and
       // `qj` are the per-pair charges, extracted from the outer
       // loop's `posq_i.w` and `posq_j.w` loads so that fragments
       // needing charges (SPME-real, truncated Coulomb) don't
       // re-load them from device memory and fragments that don't
       // (Lennard-Jones) simply ignore them.
       __device__ inline void evaluate(
           Real r2, Real inv_r, Real r,
           Real qi, Real qj,
           unsigned int i, unsigned int j,
           Real &factor, Real &energy, Real &virial) const;

       // Per-pair exclusion-scale lookup. The composed kernel
       // multiplies factor / energy / virial by this scale before
       // adding the contribution to the per-particle accumulator.
       __device__ inline Real exclusion_scale(unsigned int i, unsigned int j) const;

       // Plain-old-data parameter members (pointers to device
       // buffers, scalars). The fragment declares whatever fields
       // it needs; the framework wires them at launch time via the
       // slot's bind_args() method (see below).
   };
   ```

   `evaluate` is invoked unconditionally for every pair the outer
   loop visits, including pairs whose `r²` exceeds
   `HEDDLE_JIT_MAX_CUTOFF_SQUARED`. The outer loop multiplies the
   fragment's `(factor, energy, virial)` by a max-cutoff mask
   before accumulating, so out-of-cutoff pairs contribute exactly
   zero by bit-exact equality. Fragments must therefore tolerate
   any `r² > 0` without dividing by zero, taking the square root of
   a negative number, or otherwise faulting; `inv_r` and `r` come
   from `rsqrtf(r²)` and `r² · inv_r`, which are well-defined for
   all positive `r²`.

   The fragment is free to declare helper `__device__` functions
   above the functor struct (the LJ-12-6 implementation today
   factors a `lj_pair_evaluate(...)` helper called from the functor's
   `evaluate`; the same shape is permitted in a fragment). Helper
   names must be unique across active fragments; the framework
   prefixes the slot's label onto every emitted helper name to make
   collisions across fragments impossible.

2. May include additional `#include` directives only when the
   included file appears in the framework's allow-list of preamble
   headers (see *Compilation* below). External include paths are
   not supported.

3. Uses only the precision-policy shims from `precision.cuh`
   (`Real`, `R(x)`, `Real_sqrt`, `Real_exp`, `Real_floor`, `Real_fma`,
   etc.). A fragment that references `float` or `double` directly,
   or that uses a precision-suffixed intrinsic (`sqrtf`, `sqrt`,
   `expf`, `exp`, etc.), is malformed and the framework's nvrtc
   compile fails loudly. The shim layer carries the precision
   feature flag (`Real == float` by default, `Real == double` under
   `--features f64`).

4. Carries no static state. The fragment must be a pure function
   from `(r2, i, j, parameters)` to `(factor, energy, virial)`. A
   functor that reads or writes mutable globals — beyond the slot's
   read-only parameter buffers — is malformed.

The framework supplies the rest of the composed kernel — the
warp-per-particle outer loop, the sweep iteration, the minimum-image
arithmetic, the per-pair self-skip, the warp-tree butterfly
reduction, and the per-particle write — so the fragment only
specifies the per-pair physics for one potential.

## Composed-Kernel Structure <!-- rq-693544f8 -->

The composed kernel ships three JIT-compiled passes that all write
into the same per-particle fixed-point accumulators (see
`packed-neighbour-pair-force.md` *Fixed-Point Force Buffers*):

- **Packed-neighbour pass** (`heddle_jit_composed_pair_force_f` /
  `_fev`). One block per i-block, eight warps per block. Each
  warp iterates the entries assigned to its i-block from
  `interacting_tiles` / `sorted_interacting_atoms`, runs the
  32-iteration diagonal shuffle, and invokes the composer's
  `heddle_jit_eval_pair_sum` evaluator (no `exclusion_scale`
  call) for each pair. Treats every pair as scale 1.0.
- **Single-pair pass**
  (`heddle_jit_composed_pair_force_single_f` / `_single_fev`).
  One thread per `single_pair_atoms` entry. Loads the pair's
  positions, invokes the same `heddle_jit_eval_pair_sum`
  evaluator, atomicAdds the per-pair `±factor·(dx, dy, dz)` to
  both atoms' fixed-point slots. Same implicit scale-1.0
  semantics as the packed-neighbour pass.
- **Exclusion-correction pass**
  (`heddle_jit_composed_pair_force_correct_f` / `_correct_fev`).
  One thread per `ForceField.excluded_pair_atoms` entry. Loads
  the pair's positions, invokes the composer's
  `heddle_jit_eval_pair_correction` evaluator (which multiplies
  each fragment's `(factor, energy, virial)` by
  `(exclusion_scale(i, j) − 1.0)` and sums), atomicAdds the
  result with `±` on j-side. Net per excluded pair after all
  three passes: `scale × evaluate`. Full exclusion (`scale = 0`)
  nets to zero; OPLS-style fractional (`scale = 0.5`) nets to half.

The packed-neighbour pass is launched unconditionally when at
least one fast-class pair-force slot is active. The single-pair
and exclusion-correction passes are launched only when their
respective pair counts are non-zero.

Inside each inner iteration the per-pair scaffolding runs
unconditionally — there is no warp-divergent branch on the cutoff.
The lane:

1. Computes `(dx, dy, dz, r²)` once via the minimum-image
   displacement, and (for exclusion-tile entries) the per-pair
   exclusion scale once.
2. Computes the shared scalar intermediates once. The outer
   loop has already loaded `posq_i` and `posq_j` once each (one
   16-byte coalesced `Real4` load per warp per atom, replacing
   four separate scalar loads for the position + charge); the
   per-pair scaffolding extracts the distance components from
   `posq_i.xyz − posq_j.xyz` and the charges from `posq_i.w`
   and `posq_j.w`:
   ```
   inv_r = rsqrtf(r²)
   r     = r² · inv_r
   qi    = posq_i.w
   qj    = posq_j.w
   ```
   `inv_r`, `r`, `qi`, and `qj` are threaded into every
   fragment's `evaluate`, so each fragment reuses them instead of
   recomputing `1/r²`, `sqrt(1/r²)`, `1/r` from `r²`, or
   re-loading the charges from a separate `charges` array.
   `rsqrtf` is the hardware reciprocal-square-root intrinsic.
3. Computes the max-cutoff mask once:
   ```
   mask = (r² <= HEDDLE_JIT_MAX_CUTOFF_SQUARED) ? R(1.0) : R(0.0)
   ```
   The composer embeds `HEDDLE_JIT_MAX_CUTOFF_SQUARED` as a
   `#define` constant in the generated source at JIT-composition
   time; its value is the maximum squared cutoff across all active
   fast-class pair-force slots (`max_slot s.max_cutoff()² over
   active s`). The mask is computed branchlessly so every lane in
   the warp keeps full SM utilization through the fragment math
   regardless of which pairs are in cutoff.
4. Initialises a per-pair accumulator
   `factor = 0`, `energy = 0`, `virial = 0`.
5. For each active fragment, in canonical slot order:
   - The functor's
     `evaluate(r², inv_r, r, qi, qj, i, j, …)` produces its
     `(factor_slot, energy_slot, virial_slot)`. `evaluate` is
     called unconditionally for every pair; out-of-cutoff
     contributions are zeroed by the mask at step 7.
   - When the fragment's cutoff handling
     (`CutoffHandling`, see *Feature API*) is
     `CutoffHandling::Uniform(c)` and `c² ==
     HEDDLE_JIT_MAX_CUTOFF_SQUARED`, the composer omits the
     per-fragment `r² <= cutoff_squared(i, j)` guard entirely —
     the outer max-cutoff mask covers it. When the handling is
     `CutoffHandling::Uniform(c)` and `c² <
     HEDDLE_JIT_MAX_CUTOFF_SQUARED`, the composer emits the
     guard once as `if (r² <= c²)` with `c²` as a JIT-compile-time
     constant (no per-pair load). When the handling is
     `CutoffHandling::PerPair`, the composer emits
     `if (r² <= functor.cutoff_squared(i, j))` and only adds the
     fragment's contribution when the test passes.
   - The packed-neighbour pass and the single-pair pass add
     `(factor_slot, energy_slot · 0.5, virial_slot · 0.5)`
     directly to the pair accumulator. They never call
     `exclusion_scale`; every pair is implicitly scale `1.0`.
     The `0.5` distributes each unordered pair's energy and
     virial across the two ordered slots.
   - The exclusion-correction pass multiplies the lane's per-
     fragment contribution by
     `(exclusion_scale(i, j) − 1.0)` and adds
     `((factor_slot − factor_slot) · scale_correction,
     energy_slot · scale_correction · 0.5,
     virial_slot · scale_correction · 0.5)` (where
     `scale_correction = exclusion_scale(i, j) − 1.0`) to the
     pair accumulator. Summed with the corresponding +1.0 ×
     evaluate contribution from the packed-neighbour or single-
     pair pass, the net is `scale × evaluate`.
6. The lane multiplies the pair accumulator's `(factor, energy,
   virial)` by the max-cutoff `mask`. Pairs with `r² >
   HEDDLE_JIT_MAX_CUTOFF_SQUARED` contribute zero by bit-exact
   equality (multiplying any finite value by `0.0f` yields `+0.0f`
   in IEEE-754, and subsequent adds with `+0.0f` are identity).
7. The lane forms `(fx, fy, fz) = factor * (dx, dy, dz)` and adds
   them to per-lane, per-entry `i_*` accumulators; it also subtracts
   `(fx, fy, fz)` from per-lane, per-entry `j_*` accumulators
   (Newton's 3rd, both directions computed inside the same
   iteration). The `_fev` variant additionally adds `energy` and
   `virial` to per-scalar accumulators on both sides.

The per-entry accumulators are floating-point and are summed over
the entry's fixed 32-iteration diagonal-shuffle order, which is
deterministic. They reach the per-class fixed-point buffers
(`fast_total_forces_fp_x/y/z` and, for `_fev`,
`fast_total_potential_energies_fp` and `fast_total_virials_fp`) by
two routes:

- **j-side.** At the end of each entry the per-lane `j_*` float sum
  is converted to fixed-point and atomicAdded to the j-atom's slot —
  one atomic per (entry, lane). The j-atoms differ every entry, so
  the j-side cannot be staged.
- **i-side.** Each entry's per-lane `i_*` float sum is converted to
  fixed-point and added into a **warp-resident i64 accumulator that
  persists across every entry the warp processes**; that accumulator
  is reduced through a block shared-memory i64 accumulator and
  atomicAdded to the i-atom's slot once per (i-block, i-atom). A
  warp's tile entries arrive in a non-deterministic, atomic-built
  order (see `packed-neighbour-pair-force.md`); because the
  cross-entry accumulation is i64 integer addition, which is
  associative, the per-atom sum is **bit-exact across runs
  regardless of entry order**. A floating-point i-side accumulator
  across entries would make the result depend on that order and
  break run-to-run reproducibility.

Integer addition is also associative across warps, so cross-warp
contributions combine bit-exactly regardless of order. The
conversion to `Real` happens once per step in a separate
finalisation kernel; see `packed-neighbour-pair-force.md`.

When zero fast-class pair-force slots are active, the framework does
not compose or load a composed kernel and does not launch one at
step time. The Fast-class fixed-point accumulator stays at its
post-`cudaMemsetAsync` zero state.

The composed kernel exposes six `extern "C"` entry points
generated by the composer — three passes, each with an `_f` /
`_fev` pair:

```c
// Packed-neighbour pass.
extern "C" __global__ void heddle_jit_composed_pair_force_f(...);
extern "C" __global__ void heddle_jit_composed_pair_force_fev(...);

// Single-pair pass.
extern "C" __global__ void heddle_jit_composed_pair_force_single_f(...);
extern "C" __global__ void heddle_jit_composed_pair_force_single_fev(...);

// Exclusion-correction pass.
extern "C" __global__ void heddle_jit_composed_pair_force_correct_f(...);
extern "C" __global__ void heddle_jit_composed_pair_force_correct_fev(...);
```

The argument lists are documented in
`packed-neighbour-pair-force.md` (*Packed-Neighbour Entry-Point
Arguments* and *Single-Pair Entry-Point Arguments*) and in
*Correction-Pass Design* below. Grid sizing per pass is in
`packed-neighbour-pair-force.md` *Launch Configuration*. All
three passes run on the calling `particle_buffers.device`'s
default stream so per-stream ordering serialises their writes
into the fixed-point accumulators.

### Correction-Pass Design <!-- rq-dbf9c9cf -->

The exclusion-correction pass takes the following common
arguments before the per-fragment args:

```text
const Real4 *posq,
const unsigned int *excluded_pair_atoms,
unsigned int excluded_pair_count,
const Real *lattice,
unsigned long long *fast_force_x_fp,
unsigned long long *fast_force_y_fp,
unsigned long long *fast_force_z_fp,
unsigned long long *fast_energy_fp,
unsigned long long *fast_virial_fp,
```

Per-fragment arguments are appended in canonical slot order
followed by the trailing `unsigned int n`. The per-fragment list
is identical to the packed-neighbour and single-pair entry
points'.

`excluded_pair_atoms` is interleaved
`[i0, j0, i1, j1, …]`. Pair index `k` reads `i = excluded_pair_atoms[2k]`,
`j = excluded_pair_atoms[2k + 1]`. The list is canonical
(`i < j`) so each excluded pair contributes its correction
exactly once.

## displaces() Under JIT Composition <!-- rq-11f45908 -->

The `PotentialBuilder::displaces()` mechanism (see `framework.md`)
operates at the fragment level: a builder whose `displaces()` list
names one or more other slot labels has those slots' fragments
suppressed from the composed kernel and substitutes its own
fragment in their place. The semantics are identical to the
existing slot-displacement model except that the unit of
displacement is the per-pair functor fragment, not a separate kernel
launch.

A typical use is a hand-optimised LJ-plus-SPME-real combined
fragment that performs both pair contributions inside one
`evaluate()` and packs them into a single functor — the source
fragment plays the role today played by
`kernels/lj_spme_real_fused.cu`. The framework's resolution pass
(see `framework.md`'s *Feature API*) suppresses the displaced
constituent fragments before composition.

## Compilation <!-- rq-d5a2f76a -->

`ForceField::new` performs the following at construction, after the
slot list has been determined and displacement resolution has run:

1. Collect every active fast-class pair-force slot's fragment, in
   canonical slot order, by calling its builder's
   `pair_force_fragment(cx)`. Slots whose `frequency_class()` is
   `Fast` and whose `max_cutoff()` is `Some(_)` MUST return
   `Some(_)`; a `None` return is the
   `ForceFieldError::MissingPairForceFragment { label }` rejection
   path.

2. Build the composed-kernel source by concatenating, in order:
   1. The framework preamble. The preamble includes the project's
      `precision.cuh` shim, the `pbc.cuh` minimum-image helper, a
      shared `exclusion_scale` helper, and the warp-reduce
      definition. The composer holds the preamble verbatim — it is
      not slot-specific. The preamble's source comes from a single
      `&'static str` constant compiled into the framework.
   2. Each fragment's source text, in canonical slot order, with
      the slot's label prefixed onto every emitted helper symbol
      to avoid cross-fragment collisions.
   3. A generated `HeddleJitComposedPairFunc` struct that holds one
      member of each fragment's functor type, plus a generated
      outer-loop kernel body that calls each functor in canonical
      slot order under its own `cutoff_squared` gate, accumulates
      into the per-pair `(factor, energy, virial)`, and feeds into
      the warp-per-particle outer loop described in *Composed-
      Kernel Structure*.
   4. The two `extern "C"` entry points
      `heddle_jit_composed_pair_force_f` and
      `heddle_jit_composed_pair_force_fev`, both of which set up
      the `HeddleJitComposedPairFunc` instance from the entry
      point's argument list and dispatch the outer-loop body.

3. Compile the composed source via
   `cudarc::nvrtc::compile_ptx_with_opts`. Compilation flags are
   `--std=c++17`; the device's compute capability is detected via
   `cuDeviceGetAttribute` at session start and passed as
   `--gpu-architecture=compute_<cc_major><cc_minor>` for that GPU.
   Compilation that succeeds returns a `Ptx` value; compilation
   that fails returns
   `ForceFieldError::FragmentCompileFailed { log }`, with `log`
   containing the full nvrtc compile log verbatim. The framework
   does not attempt to recover from a compile failure.

4. Load the compiled PTX into the device with `load_ptx` under the
   module name `"heddle_jit_composed_pair_force"`. Resolve the two
   `extern "C"` entry points into `CudaFunction` handles and store
   them on the `ForceField`.

The composed-kernel source is *not* cached to disk. Every
`ForceField::new` call recompiles. Typical compile + load wall-clock
is ~100 ms on RTX 30-series hardware; this is paid once per
`ForceField` lifetime, never per step.

When zero fast-class pair-force slots are active the composer is
skipped; no PTX is generated, no module is loaded, and no entry
points exist on the `ForceField`. The per-step pipeline detects this
state and never launches a composed-kernel call.

## Parameter Binding and Launch <!-- rq-92fec152 -->

Each pair-force slot's `Potential` implementation exposes a
`bind_pair_force_args(&self, builder: &mut PairForceLaunchBuilder)`
method (see `framework.md`'s *Feature API*) that pushes the slot's
parameter buffers and scalars onto a launch-argument builder in the
same order the slot's fragment expects them. The framework owns the
builder, initialises it with the common arguments documented in
`packed-neighbour-pair-force.md` *JIT Composer Integration*
(`positions_*`, `tile_sorted_positions_*`, `sorted_particle_ids`,
exclusion-tile arrays, `interacting_tiles`, `interacting_atoms`,
`interaction_count`, `lattice`, per-class fixed-point accumulator
slices, `n`), invokes `bind_pair_force_args` on each participating
slot in canonical order, and dispatches the final argument list to
the composed-kernel launch.

The framework's call site is one of:
- `heddle_jit_composed_pair_force_f` when the framework's force
  evaluation has `AggregateLevel::ForcesOnly`.
- `heddle_jit_composed_pair_force_fev` when the framework's force
  evaluation has `AggregateLevel::ForcesAndScalars`.

`bind_pair_force_args` is called exactly once per `step()` /
`step_class(Fast, ...)` invocation, in canonical slot order.
Implementations may not allocate on the GPU during the call; they
read pointers from the slot's already-owned device buffers and push
them onto the builder.

## Determinism <!-- rq-f1cced93 -->

The composed kernel preserves the same-GPU bit-exactness invariant
of `pair-force-kernel.md`. Two runs of the same `ForceField`
configuration on the same GPU with byte-identical inputs produce
byte-identical per-particle outputs. The reproducibility argument
follows directly from the per-potential argument:

1. *Composition order is deterministic.* The composed-kernel source
   is generated by walking the active slot list in canonical slot
   order, which itself is determined by the registry's registration
   order. Two runs of `ForceField::new` from byte-identical
   configurations produce byte-identical composed source.

2. *nvrtc compilation is deterministic.* nvrtc with fixed flags
   (including `--gpu-architecture=compute_<NN>`) produces
   byte-identical PTX from byte-identical input for the same
   driver / nvrtc version. The framework holds the flag set
   constant for the lifetime of the process.

3. *The warp-per-particle pattern is unchanged.* The composed
   kernel uses the same sweep order, the same lane-strided
   accumulation, the same fixed-shape five-step `__shfl_xor_sync`
   butterfly, and the same lane-0 read-modify-write into the
   per-class accumulator. The sequence of floating-point adds into
   each `(particle, axis)` accumulator cell is identical across
   runs.

4. *Per-fragment evaluation order is deterministic.* Within a pair
   visit, fragments are evaluated in canonical slot order. The
   sequence of adds into the lane's per-pair `(factor, energy,
   virial)` accumulator is fixed by the slot order, not by thread
   scheduling.

Cross-configuration equality is not a property: a JIT-composed run
with the (LJ + SPME-real) slot set and a standalone-only run with
the LJ slot followed by the SPME-real slot agree only within f32
round-off, because the JIT-composed configuration sums LJ +
SPME-real contributions into a single per-lane register before the
warp tree, while the standalone configuration combines the two
per-particle totals through the class accumulator. Both
configurations individually preserve run-to-run byte
reproducibility on the same GPU; only cross-configuration equality
is sacrificed. This is the same invariant `framework.md`'s
*Determinism Guarantees* documents for the Mod-B composite versus
standalone case.

## Feature API <!-- rq-81cc2f2b -->

### Types <!-- rq-d603cae5 -->

- `PairForceFragment` — self-contained CUDA C++ source fragment plus <!-- rq-1ff4c7cb -->
  identifying metadata. Constructed by a slot's builder via
  `pair_force_fragment(cx)`.

  ```rust
  pub struct PairForceFragment {
      pub label: &'static str,
      pub functor_struct_name: &'static str,
      pub source: &'static str,
      pub cutoff: CutoffHandling,
  }
  ```

  - `label` matches the constructed slot's `Potential::label()` and
    is used to namespace the fragment's emitted helper symbols.
  - `functor_struct_name` is the name of the `__device__` functor
    type the fragment defines (e.g. `"LjPairFunctor"`).
  - `source` is the CUDA C++ text of the fragment: zero or more
    helper `__device__` functions plus exactly one `struct
    <functor_struct_name>` definition.
  - `cutoff` declares the fragment's per-pair cutoff structure for
    the composer's cutoff-collapse optimisation; see
    `CutoffHandling` below.

- `CutoffHandling` — declares whether a fragment uses one cutoff <!-- rq-cuthand --> <!-- rq-37c40c51 -->
  for every pair (and what that cutoff is) or a per-pair cutoff.
  The composer uses this to decide whether to emit a per-fragment
  `r² <= cutoff_squared(i, j)` guard in the inner loop, and to
  compute the global `HEDDLE_JIT_MAX_CUTOFF_SQUARED` constant.

  ```rust
  pub enum CutoffHandling {
      Uniform(Real),
      PerPair,
  }
  ```

  - `Uniform(c)` — every pair this fragment evaluates uses the
    same cutoff `c`. The composer reads `c` to set
    `HEDDLE_JIT_MAX_CUTOFF_SQUARED = max_fragment c²`. When `c² ==
    HEDDLE_JIT_MAX_CUTOFF_SQUARED` the composer omits the
    per-fragment cutoff guard entirely; the outer max-cutoff mask
    covers it. When `c² < HEDDLE_JIT_MAX_CUTOFF_SQUARED` the
    composer emits a single `if (r² <= c²)` guard with `c²` as a
    JIT-compile-time constant — no per-pair load.
  - `PerPair` — the fragment's `cutoff_squared(i, j)` may vary per
    pair. The composer emits `if (r² <= functor.cutoff_squared(i,
    j))` in the inner loop and the fragment evaluates only when
    the test passes. The composer uses the fragment's
    `max_cutoff()` (reported by its `Potential::max_cutoff()`) to
    contribute to `HEDDLE_JIT_MAX_CUTOFF_SQUARED`.

  A fragment that reports `Uniform(c)` MUST have `cutoff_squared(i,
  j) == c²` for every `(i, j)`; the composer trusts the
  declaration and skips the per-pair guard. A fragment whose
  per-pair-type cutoff table happens to have every entry equal
  reports `Uniform(c)`; a table with mixed entries reports
  `PerPair`. The decision is made once at fragment construction
  time.

- `PairForceLaunchBuilder` — opaque argument-builder threaded <!-- rq-86691f43 -->
  through every active slot's `bind_pair_force_args(...)` call. The
  framework constructs it once per launch, common arguments
  pre-populated. Implementations interact with it through pushing
  methods:

  ```rust
  impl PairForceLaunchBuilder {
      pub fn push_device_buffer<T>(&mut self, buf: &CudaSlice<T>);
      pub fn push_scalar<T: Copy>(&mut self, value: T);
  }
  ```

  Each method appends the named argument to the builder's growing
  list, in the slot's required order. The framework calls the
  compiled kernel via cudarc's raw-argument launch path once every
  slot has bound its arguments.

### Error variants <!-- rq-c011e5e2 -->

`ForceFieldError` carries the following variants for the
JIT-composition mechanism (added alongside the existing variants in
`framework.md`):

- `MissingPairForceFragment { label: &'static str }` — a slot whose <!-- rq-4d1fa71c -->
  `frequency_class()` is `Fast` and whose `max_cutoff()` is
  `Some(_)` was registered with a builder whose
  `pair_force_fragment(cx)` returned `None`. Rejection is at
  `ForceField::new` time before the composed source is built.

- `FragmentCompileFailed { log: String }` — nvrtc rejected the <!-- rq-4289075c -->
  composed source. `log` carries the full nvrtc compile log. The
  framework includes the labels of every contributing fragment in
  the surrounding error context (e.g. via a `Display` impl) so the
  caller can identify which slot's fragment is the likely culprit.

- `FragmentLoadFailed(GpuError)` — `load_ptx` rejected the compiled <!-- rq-28ceccdf -->
  PTX. Includes the underlying cudarc driver error.

### Trait methods <!-- rq-139a8a17 -->

The `PotentialBuilder` trait (see `framework.md`'s *Feature API*)
exposes the following methods to participate in JIT composition:

- `pair_force_fragment(&self, cx: &PotentialBuildContext<'_>) -> <!-- rq-b5c52011 -->
  Result<Option<PairForceFragment>, ForceFieldError>` — the
  framework calls this on every registered builder during
  `ForceField::new`, after the builder's `build(cx)` has returned
  `Ok(Some(slot))` and the displacement-resolution pass has
  determined the slot survives. Return values:

  - `Ok(Some(fragment))` — the builder produces a pair-force slot,
    and the fragment is to be composed into the composed kernel.
    Required for every slot whose `frequency_class()` is `Fast`
    AND whose `max_cutoff()` is `Some(_)`.
  - `Ok(None)` — the builder does not participate in JIT
    composition. Permitted for slots whose `frequency_class()` is
    `Slow`, or whose `max_cutoff()` is `None`, or both. A `None`
    return for a fast-class pair-force slot is the
    `MissingPairForceFragment` rejection.
  - `Err(_)` — fragment construction failed. The error surfaces
    from `ForceField::new` unchanged.

  The default implementation returns `Ok(None)`. Built-in pair-force
  builders override.

The `Potential` trait exposes the following method to bind
per-launch arguments to the composed kernel:

- `bind_pair_force_args(&self, builder: &mut PairForceLaunchBuilder)` <!-- rq-a8da1cf0 -->
  — pushes the slot's parameter buffers and scalars onto `builder`
  in the order the slot's fragment expects them. The framework
  calls this on every active fast-class pair-force slot, in
  canonical slot order, once per composed-kernel launch.

  The default implementation panics. Built-in pair-force slots
  override.

  Slots whose `frequency_class()` is not `Fast`, or whose
  `max_cutoff()` is `None`, are never asked to bind; the default
  panic surfaces a programmer error rather than silently producing
  bad launches.

### Composed-kernel module name and entry points <!-- rq-52b69f09 -->

The CUDA module loaded into the device carries the name
`heddle_jit_composed_pair_force`. It exposes two `extern "C"`
kernels:

- `heddle_jit_composed_pair_force_f` — `AggregateLevel::ForcesOnly` <!-- rq-0b8b0db9 -->
  variant. Writes the three per-particle force-component additions
  into the Fast-class accumulator's force slices.

- `heddle_jit_composed_pair_force_fev` — <!-- rq-56ddc98d -->
  `AggregateLevel::ForcesAndScalars` variant. Writes the three
  per-particle force-component additions plus the per-particle
  energy and virial additions.

Both kernels are launched with block size 256, grid `ceil(n / 8)`,
no shared memory.

### Framework integration <!-- rq-b67882f0 -->

`ForceField` owns the composed-kernel module and the two
`CudaFunction` handles. They are set on the `ForceField` exactly
when the slot list contains at least one fast-class pair-force
slot; otherwise the `Option` fields are `None`.

`ForceField::step` and `ForceField::step_class(Fast, ...)`'s
per-class compute phase (step 4 of *Force Evaluation Pipeline* in
`framework.md`) is replaced for the fast-class pair-force slot
range as follows:

1. The framework still iterates the active fast-class slot list in
   canonical order.
2. When the iteration reaches the first fast-class pair-force slot
   (a slot whose builder produced a fragment), the framework
   constructs a `PairForceLaunchBuilder`, calls
   `bind_pair_force_args` on every such slot in canonical order,
   and dispatches one composed-kernel launch.
3. The framework skips each of those slots' `Potential::compute`
   calls; the composed kernel has already added their contributions
   to the Fast-class accumulator.
4. Non-pair-force fast-class slots (`max_cutoff() == None`) still
   dispatch via `Potential::compute` as today.

The composed-kernel launch is recorded in `timings` under
`KernelStage::JitComposedPairForce`. The per-slot
`KernelStage::LjPairForce`, `KernelStage::CoulombPairForce`,
`KernelStage::SpmeRealPairForce`, and
`KernelStage::LjSpmeRealFusedPairForce` stages no longer appear in
runs where the composed kernel covers their contribution.

## Out of Scope <!-- rq-e7fd0804 -->

- Composition of bonded slots (Morse, future bond potentials) and
  angle slots (HarmonicAngle, future angle potentials) into the
  same composed kernel or into a separate composed bond / angle
  kernel. The bonded and angle parallelism shapes (per-bond,
  per-angle) differ from the warp-per-particle pair-force shape;
  their JIT composition is a separate feature.

- Composition of the slow-class SPME-reciprocal pipeline. The
  reciprocal pipeline is bound to a different parallelism shape
  (one thread per grid cell), uses cuFFT (which cannot be
  JIT-composed into other kernels), and stays out of the
  composition path.

- On-disk PTX caching keyed by `(active-slot list, source-fragment
  digest, compute capability, cudarc version, driver version)`. The
  ~100 ms nvrtc compile + load is paid once per `ForceField`
  lifetime; a process that constructs many `ForceField`s in
  succession pays it that many times. Cache support is a separate
  feature whose key-management and invalidation policies warrant
  their own design.

- A user-supplied DSL for source fragments. Fragments are CUDA C++
  text constructed by builders at construction time; the framework
  does not interpret a higher-level language.

- Hot-reload of the composed kernel mid-run. The slot list is fixed
  at `ForceField::new`; the composed kernel is loaded once and
  never recompiled or replaced.

- Mixed-mode dispatch where some fast-class pair-force slots are
  composed and others run via their own `Potential::compute`. Every
  fast-class pair-force slot in the slot list participates in the
  composed kernel; a slot that cannot provide a fragment is
  rejected at construction.

- Cross-`ForceField` sharing of the composed-kernel module. Each
  `ForceField` owns its own composed module; modules are not
  globally cached.

- Independent control of which `AggregateLevel` variant each slot's
  fragment supports. Every fragment supports both `_f` and `_fev`
  paths; the framework's outer-loop body emits the energy/virial
  accumulation under a compile-time `WriteEv` template parameter,
  identical to `pair-force-kernel.md`.

- Selective recompilation when a single slot's parameters change at
  runtime (e.g. a barostat rescaling). Parameter changes flow
  through the per-step `bind_pair_force_args` path; the kernel
  source itself does not depend on parameter values and is never
  recompiled mid-run.

## Gherkin Scenarios <!-- rq-4a49e804 -->

```gherkin
Feature: JIT-composed pair-force kernel

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called

  # --- Construction ---

  @rq-dd995812
  Scenario: Composed kernel is compiled at ForceField::new when at least one fast-class pair-force slot is present
    Given a config with one [[pair_interactions]] entry for ("Ar","Ar")
    And PotentialRegistry::with_builtins()
    When ForceField::new(...) is called
    Then it returns Ok(force_field)
    And force_field exposes a CudaFunction handle for "heddle_jit_composed_pair_force_f"
    And force_field exposes a CudaFunction handle for "heddle_jit_composed_pair_force_fev"

  @rq-044f47ec
  Scenario: Composed kernel is not compiled when no fast-class pair-force slot is present
    Given a config with no [[pair_interactions]] entries and no [coulomb] / [spme] tables
    And PotentialRegistry::with_builtins()
    When ForceField::new(...) is called
    Then it returns Ok(force_field)
    And force_field has no composed-kernel module loaded
    And no nvrtc compile is invoked during construction

  @rq-c9b26a08
  Scenario: Composed source is generated by concatenating fragments in canonical slot order
    Given two active fast-class pair-force slots A (label "lennard_jones") and B (label "spme_real") with canonical order [A, B]
    When ForceField::new(...) is called
    And the framework's pre-compile composed source is captured for inspection
    Then the source contains slot A's fragment text before slot B's fragment text
    And the source contains a HeddleJitComposedPairFunc struct that holds an instance of A's functor type before B's functor type

  @rq-28bf79fa
  Scenario: Two ForceFields constructed from the same config produce byte-identical composed PTX
    Given two independent ForceField instances built from byte-identical configurations
    When the composed-kernel PTX of each is downloaded as a byte slice
    Then the two byte slices are equal

  @rq-de8aef8c
  Scenario: A fast-class pair-force builder that returns None from pair_force_fragment is rejected
    Given a custom PotentialBuilder whose build(cx) returns Ok(Some(slot)) with frequency_class() == Fast and max_cutoff() == Some(1.0)
    And whose pair_force_fragment(cx) returns Ok(None)
    When ForceField::new(...) is called
    Then it returns Err(ForceFieldError::MissingPairForceFragment { label: <slot's label> })

  @rq-af254840
  Scenario: A bonded slot (max_cutoff == None) is not consulted for a fragment
    Given a config with no [[pair_interactions]] entries and a non-empty bond list activating the Morse bonded slot
    When ForceField::new(...) is called
    Then the framework does not invoke pair_force_fragment on MorseBondedBuilder
    And no composed-kernel module is loaded

  @rq-033fc52a
  Scenario: A slow-class slot (frequency_class == Slow) is not consulted for a fragment
    Given a config that activates SpmeReciprocal
    When ForceField::new(...) is called
    Then the framework does not invoke pair_force_fragment on SpmeReciprocalBuilder

  @rq-fa17b283
  Scenario: A malformed fragment surfaces FragmentCompileFailed with the nvrtc log
    Given a custom builder whose pair_force_fragment returns a fragment whose source contains a deliberate syntax error
    When ForceField::new(...) is called
    Then it returns Err(ForceFieldError::FragmentCompileFailed { log })
    And log contains the substring "error" (verbatim from nvrtc)

  @rq-66c2bbe0
  Scenario: A fragment that references float directly is rejected at compile time
    Given a custom builder whose fragment defines a functor using `float` instead of `Real`
    When ForceField::new(...) is called
    Then it returns Err(ForceFieldError::FragmentCompileFailed { log })

  @rq-724a053c
  Scenario: Composed-kernel module name is "heddle_jit_composed_pair_force"
    Given a ForceField with at least one fast-class pair-force slot
    When the composed-kernel module name is queried
    Then it equals "heddle_jit_composed_pair_force"

  # --- Per-step launch ---

  @rq-a704e8b4
  Scenario: step() with one fast-class pair-force slot launches the composed kernel once
    Given a ForceField with exactly one fast-class pair-force slot (LennardJones only)
    When force_field.step(...) is called with AggregateLevel::ForcesAndScalars
    Then timings records exactly one sample for KernelStage::JitComposedPairForce
    And timings records zero samples for KernelStage::LjPairForce

  @rq-8443c065
  Scenario: step() with two fast-class pair-force slots launches the composed kernel once, not twice
    Given a ForceField with LennardJones and SpmeReal both active
    When force_field.step(...) is called
    Then timings records exactly one sample for KernelStage::JitComposedPairForce
    And timings records zero samples for KernelStage::LjPairForce
    And timings records zero samples for KernelStage::SpmeRealPairForce
    And timings records zero samples for KernelStage::LjSpmeRealFusedPairForce

  @rq-6ef2d116
  Scenario: Each participating slot's bind_pair_force_args is invoked once per launch in canonical order
    Given a ForceField with LennardJones (slot order index 0) and SpmeReal (slot order index 1) both active
    And both slots' bind_pair_force_args are instrumented to record their invocation order
    When force_field.step(...) is called
    Then LennardJones's bind_pair_force_args is recorded before SpmeReal's bind_pair_force_args
    And each slot's bind_pair_force_args is recorded exactly once

  @rq-16dc6d77
  Scenario: step(ForcesOnly) launches the _f variant; step(ForcesAndScalars) launches the _fev variant
    Given a ForceField with at least one fast-class pair-force slot
    When force_field.step(..., AggregateLevel::ForcesOnly) is called
    Then the composed-kernel _f entry point is dispatched
    And the _fev entry point is not dispatched
    When force_field.step(..., AggregateLevel::ForcesAndScalars) is called
    Then the composed-kernel _fev entry point is dispatched
    And the _f entry point is not dispatched

  @rq-87a8d51c
  Scenario: step_class(Slow) does not launch the composed kernel
    Given a ForceField with Fast and Slow slots both active
    When force_field.step_class(ForceClass::Slow, ...) is called
    Then timings records zero samples for KernelStage::JitComposedPairForce

  @rq-dbe00ae4
  Scenario: step() with zero fast-class pair-force slots does not attempt a composed-kernel launch
    Given a ForceField with only a Morse bonded slot active (no pair-force slots)
    When force_field.step(...) is called
    Then it returns Ok(())
    And timings records zero samples for KernelStage::JitComposedPairForce
    And the Morse bonded slot's compute kernel is invoked exactly once

  # --- Correctness ---

  @rq-847450dd
  Scenario: Composed-kernel output equals standalone-kernel output within f32 round-off
    Given a ForceField configuration with LennardJones, Coulomb (no SPME), particle_count = 64
    And the same physical state evaluated two ways: (a) via the JIT-composed pair-force kernel, (b) via per-slot kernels followed by class-combine
    When each path runs to completion
    And the per-particle forces, energies, and virials are downloaded from each
    Then every per-particle quantity from (a) agrees with (b) within a relative tolerance of 1e-4
    But the per-particle quantities from (a) are NOT byte-identical to (b)

  @rq-3460d807
  Scenario: Two independent runs of the composed kernel on identical inputs are byte-identical
    Given two ForceField instances built from byte-identical configurations
    And two ParticleBuffers built from byte-identical ParticleStates of N = 64
    When force_field.step(...) is called on each
    Then run A's forces_x, forces_y, forces_z, potential_energies, and virials agree byte-for-byte with run B's

  @rq-89d54dfc
  Scenario: LJ + SPME-real composition matches the hand-fused reference kernel within f32 round-off
    Given a ForceField configuration with LJ and SPME-real both active and exclusion tables populated
    And a reference computation using the hand-fused reference kernel at tests/reference_kernels/lj_spme_real_fused.cu against the same physical state
    When the per-particle outputs from both paths are downloaded
    Then every per-particle quantity agrees within a relative tolerance of 1e-5

  # --- Pair-functor evaluation ---

  @rq-79de753a
  Scenario: A pair contributes only to fragments whose cutoff_squared passes
    Given a ForceField with LJ (r_cut = 1.0) and SPME-real (r_cut = 1.5) both active
    And a pair at separation r = 1.2 (inside SPME-real cutoff, outside LJ cutoff)
    When force_field.step(...) is called
    Then the per-particle force on the pair equals the SPME-real-only contribution within f32 round-off
    And the LJ contribution to that pair is zero by bit-exact equality

  @rq-a304e8cb
  Scenario: A pair past the maximum slot cutoff contributes zero by bit-exact equality
    Given a ForceField with LJ (r_cut = 1.0) and SPME-real (r_cut = 1.5) both active
    And a pair at separation r = 1.7 (outside both cutoffs and outside HEDDLE_JIT_MAX_CUTOFF_SQUARED = 1.5²)
    When force_field.step(...) is called
    Then the per-particle force / energy / virial contribution from this pair is zero by bit-exact equality
    And both fragments' evaluate() are invoked (the kernel runs unconditionally and the outer mask zeroes the contribution; calls are an implementation detail)

  @rq-1babd195
  Scenario: Per-pair contributions are summed in canonical slot order before the warp tree
    Given a ForceField with LJ (slot order index 0) and SPME-real (slot order index 1) both active
    And a pair inside both cutoffs
    When force_field.step(...) is called
    Then the per-particle force equals (LJ contribution + SPME-real contribution) where the in-register sum order is LJ-then-SPME-real
    And two runs with the slot order swapped produce per-particle forces that agree only within f32 round-off

  @rq-e7fc1920
  Scenario: Per-fragment exclusion tables apply independently to that fragment's contribution
    Given a ForceField with LJ and SPME-real both active
    And a pair (i, j) whose LJ exclusion scale is 0.5 and Coulomb exclusion scale is 0.0
    When force_field.step(...) is called
    Then the LJ contribution to the pair force is 0.5 * the unscaled LJ pair force
    And the SPME-real contribution to the pair force is zero by bit-exact equality

  # --- Shared per-pair intermediates ---

  @rq-ecc1241f
  Scenario: inv_r, r, qi, qj are computed once per pair and threaded into every fragment's evaluate
    Given a ForceField with LJ and SPME-real both active
    And the composed kernel source captured for inspection
    Then the inner loop computes inv_r = rsqrtf(r2) and r = r2 * inv_r exactly once per pair
    And the inner loop extracts qi = posq_i.w and qj = posq_j.w exactly once per pair
    And every fragment's evaluate signature is `evaluate(Real r2, Real inv_r, Real r, Real qi, Real qj, unsigned int i, unsigned int j, Real &factor, Real &energy, Real &virial)`
    And no fragment's evaluate body contains a call to Real_sqrt(r2) or computes 1.0 / r2
    And no fragment's evaluate body reads from a per-fragment `charges` array (charges flow through qi/qj only)

  @rq-03ec91b0
  Scenario: Pair-force outer loop loads posq once per atom and reuses x/y/z/w
    Given a ForceField with at least one fast-class pair-force fragment active
    And the composed kernel source captured for inspection
    Then the inner loop performs exactly one Real4 load from posq[i_atom_id] and one from posq[j_atom_id] per pair
    And the inner loop does not perform separate loads from positions_x, positions_y, positions_z, or charges arrays
    And the displacement components (dx, dy, dz) are computed as posq_i.xyz − posq_j.xyz

  @rq-15a42b50
  Scenario: SPME-real and LJ in-cutoff pair force matches the closed form within f32 round-off
    Given a ForceField with LJ and SPME-real both active
    And a pair (i, j) at separation r = 0.4 (inside both cutoffs)
    When force_field.step(...) is called
    Then the per-particle force on i agrees within 1e-5 relative tolerance with the sum of (a) the closed-form LJ pair force using sigma/epsilon for the pair's types and (b) the closed-form erfc-screened Coulomb pair force using k_C, q_i, q_j, alpha, r
    And the result is byte-identical to a second run of the same kernel on the same inputs

  # --- Cutoff-handling and collapse ---

  @rq-b19d4365
  Scenario: A Uniform-cutoff fragment whose c² equals HEDDLE_JIT_MAX_CUTOFF_SQUARED has its per-fragment guard omitted
    Given a ForceField with one fragment whose builder reports CutoffHandling::Uniform(c) with c² == max_cutoff² across all active fragments
    And the composed kernel source captured for inspection
    Then the composed source does not contain a call to that fragment's `cutoff_squared(i, j)`
    And the composed source does not contain an `if (r2 <= …)` guard around that fragment's evaluate

  @rq-e8699851
  Scenario: A Uniform-cutoff fragment whose c² is strictly less than HEDDLE_JIT_MAX_CUTOFF_SQUARED gets a compile-time-constant guard
    Given two fragments F1 with CutoffHandling::Uniform(1.0) and F2 with CutoffHandling::Uniform(1.5)
    And the composed kernel source captured for inspection
    Then HEDDLE_JIT_MAX_CUTOFF_SQUARED equals 1.5² (= 2.25) at JIT compile time
    And the inner loop guards F1's evaluate with `if (r2 <= 1.0)` as a literal compile-time constant
    And the inner loop does not call F1's `cutoff_squared(i, j)` at runtime

  @rq-b3b12764
  Scenario: A PerPair-cutoff fragment has its runtime cutoff_squared guard emitted
    Given a fragment F with CutoffHandling::PerPair (e.g., LJ with mixed type_cutoff entries)
    And the composed kernel source captured for inspection
    Then the inner loop guards F's evaluate with `if (r2 <= functor.cutoff_squared(i, j))` and F's `cutoff_squared` is invoked at runtime

  @rq-118a47d5
  Scenario: LJ with a uniform type_cutoff table reports CutoffHandling::Uniform
    Given a config whose [[pair_interactions]] entries all have the same cutoff
    When LennardJonesBuilder::pair_force_fragment(cx) is called
    Then it returns Ok(Some(fragment)) with `fragment.cutoff == CutoffHandling::Uniform(c)` where c is the common cutoff

  @rq-059aff56
  Scenario: LJ with mixed cutoffs across pair types reports CutoffHandling::PerPair
    Given a config whose [[pair_interactions]] entries have at least two distinct cutoff values
    When LennardJonesBuilder::pair_force_fragment(cx) is called
    Then it returns Ok(Some(fragment)) with `fragment.cutoff == CutoffHandling::PerPair`

  @rq-b7ed60ff
  Scenario: SPME-real always reports CutoffHandling::Uniform
    Given any [spme] configuration with r_cut_real = c
    When SpmeRealBuilder::pair_force_fragment(cx) is called
    Then it returns Ok(Some(fragment)) with `fragment.cutoff == CutoffHandling::Uniform(c)`

  # --- Three-pass structure ---

  @rq-b099ff28
  Scenario: Composer emits a no-exclusion evaluator and a correction evaluator
    Given a ForceField with at least one fast-class pair-force fragment
    And the composed kernel source captured for inspection
    Then the source contains a function `heddle_jit_eval_pair_sum` whose body contains zero calls to `composite.<any>.exclusion_scale`
    And the source contains a function `heddle_jit_eval_pair_correction` whose body calls every fragment's `exclusion_scale(i, j)` once per pair

  @rq-54aec894
  Scenario: Packed-neighbour pass dispatches to the no-exclusion evaluator
    Given a ForceField with at least one fast-class pair-force fragment
    And the composed kernel source captured for inspection
    Then the packed-neighbour outer loop's inner body dispatches to `heddle_jit_eval_pair_sum<WriteEv>`
    And the packed-neighbour outer loop's inner body contains no `composite.<any>.exclusion_scale` calls

  @rq-95f0812c
  Scenario: Single-pair pass dispatches to the no-exclusion evaluator
    Given a ForceField with at least one fast-class pair-force fragment
    And the composed kernel source captured for inspection
    Then the single-pair kernel's per-thread body dispatches to `heddle_jit_eval_pair_sum<WriteEv>`
    And the single-pair kernel's per-thread body contains no `composite.<any>.exclusion_scale` calls

  @rq-0dc4e38e
  Scenario: Exclusion-correction pass dispatches to the correction evaluator
    Given a ForceField with at least one fast-class pair-force fragment
    And the composed kernel source captured for inspection
    Then the correction kernel's per-thread body dispatches to `heddle_jit_eval_pair_correction<WriteEv>`
    And the correction kernel's per-thread body calls every active fragment's `exclusion_scale` exactly once per pair

  @rq-f1a44df1
  Scenario: Single-pair pass and packed-neighbour pass produce bit-exact results for the same pair routed either way
    Given a ForceField configuration with one (i-block, j-block) pair just below the MAX_BITS_FOR_PAIRS threshold and an otherwise-identical run with the same pair just above the threshold
    When ForceField::step(...) is called on each
    Then the per-particle forces, energies, and virials are byte-identical between the two runs

  @rq-c156295f
  Scenario: Three-pass composition is run-to-run byte-identical
    Given a ForceField configuration with LJ + SPME-real + topology exclusions + sparse-tile candidates
    When ForceField::step(...) is called on two independent ForceField instances built from byte-identical inputs
    Then run A's per-particle forces, energies, and virials are byte-identical to run B's

  # --- displaces() under JIT composition ---

  @rq-e469657f
  Scenario: A composite-fragment builder displaces its constituents' fragments
    Given a custom builder C with displaces() == &["lennard_jones", "spme_real"] that produces a fragment whose single functor evaluates both LJ and SPME-real per pair
    And a config that would otherwise activate LennardJonesBuilder, SpmeRealBuilder, and C
    When ForceField::new(...) is called
    Then the composed source includes C's fragment exactly once
    And the composed source includes neither LennardJonesBuilder's fragment nor SpmeRealBuilder's fragment
    And exactly one KernelStage::JitComposedPairForce sample is recorded per step()

  @rq-28445f4e
  Scenario: A composite-fragment builder that builds without its constituents falls back gracefully
    Given a custom builder C with displaces() == &["lennard_jones", "spme_real"]
    And a config where only LennardJones is configured (no [spme])
    And C's build(cx) returns Ok(None) when its [spme] activation condition is unmet
    When ForceField::new(...) is called
    Then the composed source includes LennardJonesBuilder's fragment
    And the composed source does not include any fragment from C
    And no slot in the surviving slot list has label() == C's label

  # --- Error reporting ---

  @rq-5fd76c7f
  Scenario: FragmentCompileFailed surfaces the slot labels of every contributing fragment
    Given two active fragments A (label "alpha") and B (label "beta") where B's fragment contains a syntax error
    When ForceField::new(...) is called
    Then the returned Err's Display contains the substrings "alpha" and "beta"
    And the underlying FragmentCompileFailed::log carries the nvrtc compile log verbatim

  # --- Mod B retirement ---

  @rq-33861244
  Scenario: LjSpmeRealFusedBuilder is not part of PotentialRegistry::with_builtins
    Given PotentialRegistry::with_builtins()
    Then no builder in registry.builders has type LjSpmeRealFusedBuilder
    And kernels/lj_spme_real_fused.cu does not appear in the build-time kernel module list
```
