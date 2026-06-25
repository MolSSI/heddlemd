# Feature: JIT-Composed Intramolecular Kernels <!-- rq-2d2eaf72 -->

Every fast-class intramolecular slot — bonded slots that iterate the
bond list (Morse, future cosine bonds, …) and angle slots that
iterate the angle list (HarmonicAngle, future Urey-Bradley angles,
…) — exposes its per-bond or per-angle physics as a CUDA source
fragment. At `ForceField::new` time the framework collects the
active fragments grouped by parallelism shape, JIT-compiles a
per-shape composed module via `cudarc::nvrtc::compile_ptx_with_opts`,
and loads each module on the device. The per-step force-evaluation
pipeline dispatches one composed contribution kernel per active
bonded slot from the bonded module and one per active angle slot
from the angle module, in canonical slot order.

The mechanism mirrors the pair-force composer described in
`jit-composed-pair-force.md`. Two parallelism shapes have their own
composed modules:

- **Bonded shape** — one thread per bond, walks the slot's bond
  list (`(atom_i, atom_j, bond_type_index)` triples). Writes
  per-atom force, half-energy, and half-virial contributions into
  the slot's bond-pair scratch buffer; the standalone
  `reduce_bond_forces` kernel sums each atom's contributions into
  the slot's per-particle accumulator.
- **Angle shape** — one thread per angle, walks the slot's angle
  list (`(atom_i, atom_j, atom_k, angle_type_index)` tuples).
  Writes per-atom force, third-energy, and third-virial
  contributions into the slot's angle-triple scratch buffer; the
  standalone `reduce_angle_forces` kernel sums each atom's
  contributions into the slot's per-particle accumulator.

Each composed module holds the union of every active slot's
fragment source in that shape. Per-slot entry points address one
slot's bond/angle list and write into one slot's scratch buffer.
The framework launches each entry point in canonical slot order
once per `step()` / `step_class(Fast, ...)` invocation. Each slot
still owns its own bond / angle list and its own bond-pair /
angle-triple scratch buffer; bond and angle lists are not merged
across slots.

The reduction stage (per-atom summation of scratch contributions
into the per-particle accumulator) runs as a separate launch per
active slot using the universal `reduce_bond_forces` /
`reduce_angle_forces` kernels documented in `morse-bonded.md` and
`harmonic-angle.md`. These reduction kernels are shape-universal
across slots and are not part of the JIT module; they are compiled
at build time via `nvcc` and loaded as PTX.

This file specifies the composition mechanism, the source-fragment
contract a slot must satisfy for each shape, the composed kernel
structure, the framework's compile + launch contract, and the
determinism guarantees the composed modules preserve.

## Slot Participation <!-- rq-10313445 -->

A *fast-class bonded slot* is any slot whose `frequency_class()`
returns `ForceClass::Fast`, whose `max_cutoff()` returns `None`,
and whose builder exposes a `bonded_force_fragment(cx)` returning
`Some(_)`. A *fast-class angle slot* is the analogous slot whose
builder exposes `angle_force_fragment(cx)` returning `Some(_)`.

The framework calls `bonded_force_fragment(cx)` and
`angle_force_fragment(cx)` on every registered builder during
`ForceField::new`, after `build(cx)` has returned `Ok(Some(slot))`
and the displacement-resolution pass has determined the slot
survives. Slots whose builders return `Ok(Some(slot))` from
`build(cx)` but `Ok(None)` from both `bonded_force_fragment(cx)`
and `angle_force_fragment(cx)` are dispatched via their own
`Potential::compute` kernel call at step time (the bonded /
angle JIT path does not see them).

A slot whose builder returns `Some(_)` from
`bonded_force_fragment(cx)` and reports `frequency_class() == Fast`
participates in the bonded composed module. A slot whose builder
returns `Some(_)` from `angle_force_fragment(cx)` and reports
`frequency_class() == Fast` participates in the angle composed
module. A single builder may participate in at most one shape's
composer; a builder that returns `Some(_)` from more than one
fragment method causes `ForceField::new` to return
`ForceFieldError::SlotMultipleShapes { label }`.

A builder that names itself a bonded slot via the canonical
`MorseBondedBuilder` / future bonded builders contract but whose
`bonded_force_fragment(cx)` returns `None` is rejected at
construction with `ForceFieldError::MissingBondedFragment { label }`.
The analogous rejection for angle slots is
`ForceFieldError::MissingAngleFragment { label }`. The framework
recognises the canonical bonded / angle slot list at compile time
through the canonical builders the registry exposes; slots from
custom builders are not auto-classified as bonded / angle —
participation is determined exclusively by which fragment method
returns `Some(_)`.

## Source-Fragment Contract — Bonded Shape <!-- rq-892e8856 -->

A bonded fragment carries the CUDA source for one slot's per-bond
functor plus identifying metadata. The snippet:

1. Defines a stateless `__device__` functor whose name is given by
   the fragment's `functor_struct_name`. The functor exposes:

   ```c
   struct <functor_struct_name> {
       // Plain-old-data parameter members (device-buffer pointers,
       // scalars). The fragment declares whatever fields it needs;
       // the framework wires them at launch time via the slot's
       // bind_bonded_force_args() method.

       __device__ inline void evaluate(
           Real r2, Real r,
           unsigned int bond_type_index,
           Real dx, Real dy, Real dz,
           Real &fmag,
           Real &u_k,
           Real &w_k) const;
   };
   ```

   `evaluate` is called once per surviving bond. Inputs:
   - `r2` — the squared minimum-image displacement magnitude.
   - `r` — `sqrt(r2)` (precomputed by the outer-loop body so the
     functor avoids a redundant `sqrt`).
   - `bond_type_index` — the bond's type tag from the bond list.
   - `(dx, dy, dz)` — the minimum-image displacement
     `(r_i - r_j)`.

   Outputs:
   - `fmag` — the scalar force factor along the displacement
     direction (so that `F_on_i = fmag · (dx, dy, dz)`).
   - `u_k` — the bond's full potential energy `U_k`.
   - `w_k` — the bond's scalar virial `W_k = fmag · r²`.

   The outer-loop body computes the per-atom triples
   `± fmag · (dx, dy, dz)` and writes them along with `0.5 · u_k` /
   `0.5 · w_k` into the slot's scratch buffer.

2. Uses only the precision-policy shims from `precision.cuh`
   (`Real`, `R(x)`, `Real_sqrt`, `Real_exp`, etc.). A fragment
   that references `float` or `double` directly is malformed.

3. Carries no static state. The functor is a pure function from
   `(r2, r, bond_type_index, dx, dy, dz, parameters)` to
   `(fmag, u_k, w_k)`.

The fragment also carries `entry_point_args` and `functor_init_source`,
both generated from the slot's `KernelArgSchema` (see *Argument
Schema*). `entry_point_args` declares the slot's kernel parameters
(per-bond-type parameter table pointers, scalars), concatenated into the
composed entry point's signature; `functor_init_source` assigns those
parameters to the entry point's local `functor`, emitted at the start of
the entry-point body.

## Source-Fragment Contract — Angle Shape <!-- rq-c8fb9600 -->

An angle fragment's functor exposes a different evaluator:

```c
struct <functor_struct_name> {
    __device__ inline void evaluate(
        Real dx_ij, Real dy_ij, Real dz_ij,
        Real dx_kj, Real dy_kj, Real dz_kj,
        unsigned int angle_type_index,
        Real &fix, Real &fiy, Real &fiz,
        Real &fkx, Real &fky, Real &fkz,
        Real &u_m,
        Real &w_m) const;
};
```

Inputs:
- `(dx_ij, dy_ij, dz_ij)` — minimum-image displacement
  `(r_i - r_j)`.
- `(dx_kj, dy_kj, dz_kj)` — minimum-image displacement
  `(r_k - r_j)`.
- `angle_type_index` — the angle's type tag from the angle list.

Outputs:
- `(fix, fiy, fiz)` — Cartesian force on atom `i`.
- `(fkx, fky, fkz)` — Cartesian force on atom `k`.
  Force on atom `j` is computed by the outer-loop body as
  `-(F_i + F_k)`.
- `u_m` — the angle's full potential energy `U_m`.
- `w_m` — the angle's scalar virial
  `W_m = r_ij · F_i + r_kj · F_k`.

The outer-loop body writes the three per-atom triples (one each
for `i`, `j`, `k`) along with `u_m / 3` / `w_m / 3` (one-third
shares; see `harmonic-angle.md`'s *Force Accumulation*) into the
slot's scratch buffer.

The fragment's `entry_point_args` and `functor_init_source` are
generated from the slot's `KernelArgSchema` exactly as in the bonded
shape (see *Argument Schema*).

## Argument Schema <!-- rq-13d8e659 -->

A bonded or angle slot's kernel parameters are declared once as a
`KernelArgSchema` — an ordered list of typed `KernelArg` entries that is
the single source of truth for the slot's contribution to the composed
module. The schema type and its companions (`KernelArg`,
`KernelArgType`, `KernelArgBinder`, `ElemTy`, `ArgKind`, `KernelElem`)
are shape-neutral and shared with the pair-force composer, where they
are canonically defined (`jit-composed-pair-force.md`'s *Feature API*).
A bonded or angle slot constructs its schema with
`KernelArgSchema::intramolecular(label, args)`.

From this one list the framework derives the three artefacts that must
stay in agreement:

- The fragment's `entry_point_args` — the slot's CUDA `extern "C"`
  parameter declarations, concatenated into the per-slot entry point's
  signature — is produced by `KernelArgSchema::entry_point_args()`.
- The fragment's `functor_init_source` — the assignments that copy each
  kernel parameter into the entry point's local functor — is produced by
  `KernelArgSchema::functor_init_source()`. For an `intramolecular`
  schema each line reads `functor.<functor_field> = <name>;`, targeting
  the `functor` local the per-slot entry-point body declares (one functor
  per entry point, not a shared composite functor as in the pair-force
  shape).
- The slot's `Potential::bind_bonded_force_args` /
  `bind_angle_force_args` pushes one launch argument per schema entry
  through a `KernelArgBinder`, which validates each push against the
  same schema.

Because all three derive from one ordered list, the parameter order in
the entry-point signature, the field-initialisation order, and the
launch-time binding order are identical by construction. A bonded or
angle slot does not hand-write `entry_point_args` or
`functor_init_source`, and its bind method does not push arguments
directly onto the `ForceLaunchBuilder`.

Each `KernelArg` carries `name` (the CUDA parameter name), `ty` (a
`KernelArgType` fixing the declaration and the accepted push kind), and
`functor_field` (the functor struct field it initialises). The functor
struct declared in the fragment's `functor_source` must declare a field
of each `functor_field` name and a compatible type; a mismatch is an
nvrtc compile error (`FragmentCompileFailed`), not a silent fault.

### Schema-Checked Binding <!-- rq-30d233fd -->

`bind_bonded_force_args` / `bind_angle_force_args` constructs a
`KernelArgBinder` over the slot's schema and the framework's
`ForceLaunchBuilder`, then pushes one value per declared argument by
name. The binder validates every push, in declaration order, on every
launch: the pushed name must equal the next schema entry's `name`; the
push kind (buffer vs scalar) and, for buffers, the `CudaSlice<T>`
element type must match the schema entry's `KernelArgType`; and the
total number of pushes must equal the schema's length. A name, kind,
element-type, or count mismatch panics with a message naming the slot
and the offending argument, instead of silently corrupting the entry
point's argument list. The validation runs on every bind call (once per
composed-entry-point launch per step); it is not gated to debug builds.

## Composed-Module Structure <!-- rq-d90f3107 -->

Each shape's composed module contains:

1. The framework's preamble (precision shims, PBC minimum-image
   helpers, block-size constants — identical to
   `jit-composed-pair-force.md`'s preamble).
2. Each active slot's `functor_source`, in canonical slot order,
   with the slot's label prefixed onto every emitted helper
   symbol so cross-fragment collisions are impossible.
3. A generated `extern "C" __global__` entry point per active
   slot per `AggregateLevel` variant. Naming:
   `heddle_jit_composed_bonded_<slot_index>_<f|fev>` for the
   bonded shape and `heddle_jit_composed_angle_<slot_index>_<f|fev>`
   for the angle shape, where `<slot_index>` is the slot's
   zero-based index among the shape's active slots in canonical
   order. Each entry point takes the common args (positions,
   lattice, bond / angle list, scratch buffer slices, n_bonds /
   n_angles) plus the slot's per-fragment args.

The bonded entry point's per-thread body is the standalone
`morse_bond_force` body abstracted to a generic functor call: read
the bond, compute the minimum-image displacement, compute
`r2` and `r`, call the slot's `evaluate(...)`, write the
per-atom contributions to slots `2·k` and `2·k + 1` of the scratch
buffer. The angle entry point's body is the analogous abstraction
of the standalone `harmonic_angle_force` body.

When zero fast-class bonded slots are active, the bonded module is
not compiled and not loaded. Same for the angle module. The
per-step pipeline detects each empty state and dispatches no
composed-module launches for that shape.

## Compilation <!-- rq-e5f8b6fc -->

`ForceField::new` performs the following at construction for each
shape independently, after the slot list has been determined and
displacement resolution has run:

1. Collect every active fast-class slot's fragment for that
   shape via `bonded_force_fragment(cx)` /
   `angle_force_fragment(cx)`, in canonical slot order.
2. Build the composed-module source by concatenating, in order:
   the shape's preamble, each fragment's `functor_source`, and
   the generated `extern "C"` entry points (one per slot per
   `AggregateLevel` variant).
3. Compile via `cudarc::nvrtc::compile_ptx_with_opts` with
   `--std=c++17` and `--gpu-architecture=compute_<NN>` for the
   detected device. A compile failure returns
   `ForceFieldError::FragmentCompileFailed { log }` with the
   fragment labels prepended for diagnostic clarity.
4. Load the compiled PTX into the device under the module name
   `"heddle_jit_composed_bonded"` (or
   `"heddle_jit_composed_angle"`). Resolve every entry point into
   a `CudaFunction` handle and store them on the `ForceField`
   keyed by `(slot_index, AggregateLevel)`.

The bonded and angle composed modules are loaded independently;
either may be present without the other. Both are held for the
`ForceField`'s lifetime. PTX is not cached to disk; every
`ForceField::new` call recompiles.

## Parameter Binding and Launch <!-- rq-8bf40375 -->

Each fast-class bonded slot's `Potential` implementation exposes a
`bind_bonded_force_args(&self, ctx: &ForceLaunchContext<'_>, builder:
&mut ForceLaunchBuilder)` method, and each fast-class angle slot
exposes `bind_angle_force_args(&self, ctx, builder)` with the same
signature. The methods supply the slot's parameter buffers and scalars
through a `KernelArgBinder` validated against the slot's
`KernelArgSchema` (see *Argument Schema*), in the order the schema
declares them — the same order the slot's fragment's `entry_point_args`
were generated in.

The `ForceLaunchBuilder` is the same type used by the pair-force
composer (see `framework.md`'s *Feature API* and
`jit-composed-pair-force.md`); the binding mechanism is
shape-agnostic. `ForceLaunchContext<'a>` carries `&ParticleBuffers`,
`&SimulationBox`, and the slot's bond-pair-buffer or
angle-triple-buffer slices (see `morse-bonded.md` /
`harmonic-angle.md` for the buffer layouts).

The framework's per-step pipeline launches each active slot's
composed entry point in canonical slot order. For each launch:

1. Construct a fresh `ForceLaunchBuilder`.
2. Push the common args: positions_x/y/z, lattice, the slot's
   bond / angle list, the slot's scratch-buffer slices, the slot's
   bond / angle count.
3. Invoke the slot's `bind_bonded_force_args` /
   `bind_angle_force_args` to push the slot's per-fragment args.
4. Dispatch the composed entry point with block size 256, grid
   `ceil(n_bonds / 256)` (or `ceil(n_angles / 256)`), no shared
   memory.
5. Run the slot's `reduce_bond_forces` / `reduce_angle_forces`
   kernel (standalone, build-time-compiled) over the scratch
   buffer to sum into the Fast-class accumulator. The reduction
   kernel is not part of the composed module.

The composed-bonded and composed-angle launches are recorded in
`timings` under `KernelStage::JitComposedBondedForce` and
`KernelStage::JitComposedAngleForce`. The per-slot standalone
`KernelStage::MorseBondForce` and `KernelStage::HarmonicAngleForce`
stages do not appear in runs that go through the JIT path.

## Determinism <!-- rq-627ed4b9 -->

Same-GPU bit-exact reproducibility is preserved for two runs of
the same `ForceField` configuration with byte-identical inputs:

1. *Composition order is deterministic.* Source is generated by
   walking active slots in canonical slot order; nvrtc with fixed
   flags produces byte-identical PTX from byte-identical input.
2. *Per-thread evaluation is independent.* Each thread is the sole
   writer of two (bonded) or three (angle) scratch-buffer slots
   keyed by its bond / angle index. No atomics, no shared memory,
   no inter-thread communication during contribution.
3. *Per-atom reduction is deterministic.* `reduce_bond_forces` and
   `reduce_angle_forces` (unchanged from `morse-bonded.md` and
   `harmonic-angle.md`) sum the slot's scratch buffer's atom-keyed
   slots in fixed `atom_bond_indices` / `atom_angle_indices`
   order, which is sorted at load time.

Cross-configuration equality (one slot's per-particle output via
the JIT path vs the same physics evaluated through the standalone
per-potential kernel) is not a property: the JIT-composed
contribution kernel may produce different low-bit results than
the standalone kernel because nvrtc's code generation differs
from build-time `nvcc`'s. Both paths preserve the same-GPU
reproducibility invariant individually.

## Feature API <!-- rq-6789ce57 -->

### Types <!-- rq-28611487 -->

- `BondedForceFragment` — self-contained CUDA C++ source fragment <!-- rq-7773e9e8 -->
  plus identifying metadata for a bonded slot, returned by
  `PotentialBuilder::bonded_force_fragment(cx)`. Same field set as
  `PairForceFragment` (see `framework.md`):

  ```rust
  pub struct BondedForceFragment {
      pub label: &'static str,
      pub functor_struct_name: &'static str,
      pub functor_source: String,
      pub entry_point_args: String,
      pub functor_init_source: String,
  }
  ```

  The `entry_point_args` and `functor_init_source` fields are generated
  from the slot's `KernelArgSchema`, constructed via
  `KernelArgSchema::intramolecular`; see *Argument Schema*.

- `AngleForceFragment` — same shape as `BondedForceFragment`, <!-- rq-565ffbcc -->
  returned by `PotentialBuilder::angle_force_fragment(cx)`. Its
  `entry_point_args` and `functor_init_source` are likewise generated
  from the slot's `KernelArgSchema`.

- `KernelArgSchema`, `KernelArg`, `KernelArgType`, `KernelArgBinder`, <!-- rq-402b55fc -->
  `ElemTy`, `ArgKind`, `KernelElem` — the shape-neutral kernel-argument
  schema types, defined canonically in `jit-composed-pair-force.md`'s
  *Feature API*. A bonded or angle slot builds its schema with
  `KernelArgSchema::intramolecular(label, args)` (local-functor
  `functor_init_source`) and validates its launch-time binding with
  `KernelArgBinder` over a `ForceLaunchBuilder`.

- `ForceLaunchBuilder` — opaque argument-builder threaded through <!-- rq-61a784a9 -->
  every active fast-class slot's bind method (`bind_pair_force_args`,
  `bind_bonded_force_args`, `bind_angle_force_args`). The same
  type the pair-force composer uses; the bonded and angle
  composers reuse it unchanged.

- `ForceLaunchContext<'a>` — shape-specific per-launch context <!-- rq-1e1cd9c7 -->
  carrying read-only references to `ParticleBuffers`,
  `SimulationBox`, and the slot's scratch-buffer slices when
  applicable. The pair-force shape carries the neighbour list;
  the bonded shape carries the slot's bond list + bond-pair
  scratch slices; the angle shape carries the slot's angle list +
  angle-triple scratch slices. (See `framework.md` for the full
  field set.)

### Error variants <!-- rq-0f8a60e2 -->

`ForceFieldError` carries the following variants for the bonded /
angle JIT mechanism, alongside the pair-force composer's existing
variants (see `framework.md`):

- `MissingBondedFragment { label: &'static str }` — a slot whose <!-- rq-e045f2ef -->
  builder names itself bonded but whose `bonded_force_fragment(cx)`
  returned `None`. Reported from `ForceField::new` before the
  bonded composed source is built.
- `MissingAngleFragment { label: &'static str }` — the angle <!-- rq-c6e8020e -->
  analogue.
- `SlotMultipleShapes { label: &'static str }` — a builder whose <!-- rq-311c9f01 -->
  `pair_force_fragment(cx)`, `bonded_force_fragment(cx)`, and
  `angle_force_fragment(cx)` together return `Some(_)` from more
  than one method. A slot participates in at most one shape's
  composer.

`FragmentCompileFailed` and `FragmentLoadFailed` are reused for
bonded / angle module compile / load errors; the error's
`Display` impl additionally names every contributing fragment's
slot label so the caller can identify which slot's fragment is
the likely culprit.

### Trait methods <!-- rq-0b1918e0 -->

The `PotentialBuilder` trait (see `framework.md`'s *Feature API*)
exposes the following methods for participation:

- `bonded_force_fragment(&self, cx: &PotentialBuildContext<'_>) <!-- rq-ac1403d3 -->
  -> Result<Option<BondedForceFragment>, ForceFieldError>` —
  default `Ok(None)`. Built-in bonded builders override.
- `angle_force_fragment(&self, cx: &PotentialBuildContext<'_>) <!-- rq-c9ac8000 -->
  -> Result<Option<AngleForceFragment>, ForceFieldError>` —
  default `Ok(None)`. Built-in angle builders override.

The `Potential` trait exposes parallel binding methods to the
existing `bind_pair_force_args`:

- `bind_bonded_force_args(&self, ctx: &ForceLaunchContext<'_>, <!-- rq-b08937a3 -->
  builder: &mut ForceLaunchBuilder)` — supplies the slot's parameter
  buffers through a `KernelArgBinder` over the slot's `KernelArgSchema`
  and `builder`, pushing one value per declared argument by name and
  calling `finish()`. The binder validates every push against the
  schema; a name, kind, element-type, or count mismatch panics, naming
  the slot and the offending argument. Default panics so an
  unimplemented override surfaces a programmer error.
- `bind_angle_force_args(&self, ctx, builder)` — angle analogue. <!-- rq-9bd9ccd4 -->

### Composed-module name and entry points <!-- rq-529b2c5f -->

The bonded composed module is loaded under the name
`"heddle_jit_composed_bonded"`. For each active bonded slot with
slot index `i` (zero-based among active bonded slots in canonical
order), the module exposes:

- `heddle_jit_composed_bonded_<i>_f` — `AggregateLevel::ForcesOnly` <!-- rq-eb943277 -->
  variant. Writes only the per-atom force-component additions
  into the slot's bond-pair scratch buffer.
- `heddle_jit_composed_bonded_<i>_fev` — <!-- rq-74fce41f -->
  `AggregateLevel::ForcesAndScalars` variant. Writes per-atom
  force, half-energy, and half-virial additions.

The angle composed module is loaded under
`"heddle_jit_composed_angle"`; entry points follow the same
convention substituting `angle` for `bonded`.

Both kernels launch with block size 256, grid `ceil(n_bonds / 256)`
or `ceil(n_angles / 256)`, no shared memory.

### Framework integration <!-- rq-b677d7a4 -->

`ForceField` owns two new optional fields holding the bonded and
angle composed modules and their per-slot entry-point handles
(typed analogously to the pair-force `JitComposedPairForce` field).
They are `Some(_)` exactly when the slot list contains at least
one active bonded / angle slot.

`ForceField::step` and `ForceField::step_class(Fast, ...)`'s
per-class compute phase dispatches the bonded and angle composed
launches in canonical slot order. For each fast-class bonded slot,
the framework:

1. Constructs a `ForceLaunchBuilder`, pushes the common args,
   invokes `bind_bonded_force_args`, dispatches the slot's
   `_f` or `_fev` entry point (depending on `AggregateLevel`).
2. Launches the slot's `reduce_bond_forces` over the slot's
   scratch buffer.

The slot's `Potential::compute` is bypassed at step time for
every fast-class bonded slot that participates in the bonded
composed module. The analogous dispatch handles fast-class angle
slots.

## Out of Scope <!-- rq-bbbcfdc3 -->

- Composition of the per-atom reduction kernels
  (`reduce_bond_forces`, `reduce_angle_forces`). These are
  shape-universal across slots; they stay as standalone PTX
  modules compiled at build time. Folding them into the
  JIT-composed module is a separate feature gated on the K
  multi-step-persistent-loop work that needs every device-side
  function in one module.

- Merging multiple bonded slots' bond lists into one tagged list
  walked by a single launch. With one launch per active slot
  sharing the JIT module, the launch count scales with the active
  slot count. Workloads with multiple bonded slots remain rare;
  merging is a follow-up if and when that changes.

- Composition of bonded fragments with pair-force fragments into a
  single kernel. The parallelism shapes (warp-per-particle over
  neighbour list vs one-thread-per-bond) are incompatible.

- A user-supplied DSL for bonded / angle source fragments. Like
  the pair-force composer, fragments are CUDA C++ text constructed
  by builders at construction time; the framework does not
  interpret a higher-level language.

- On-disk PTX caching of the bonded / angle modules. Same policy
  as the pair-force composer: every `ForceField::new` recompiles.

- Hot-reload of the composed modules mid-run. The slot list is
  fixed at `ForceField::new`; modules are loaded once and never
  replaced.

## Gherkin Scenarios <!-- rq-fd955f5e -->

```gherkin
Feature: JIT-composed intramolecular kernels

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called

  # --- Bonded composition: construction ---

  @rq-69a69fef
  Scenario: Bonded composed module is compiled at ForceField::new when at least one bonded slot is active
    Given a config with one [[bond_types]] entry "morse" and a non-empty BondList
    And PotentialRegistry::with_builtins()
    When ForceField::new(...) is called
    Then it returns Ok(force_field)
    And force_field exposes a CudaFunction handle for "heddle_jit_composed_bonded_0_f"
    And force_field exposes a CudaFunction handle for "heddle_jit_composed_bonded_0_fev"

  @rq-2e3bda9a
  Scenario: Bonded composed module is not compiled when no bonded slot is active
    Given a config with no bond list
    And PotentialRegistry::with_builtins()
    When ForceField::new(...) is called
    Then no composed-bonded module is loaded
    And no nvrtc compile is invoked for the bonded shape

  @rq-3e69b82c
  Scenario: A bonded builder that returns None from bonded_force_fragment is rejected
    Given a custom builder whose build(cx) returns Ok(Some(slot)) with frequency_class() == Fast and max_cutoff() == None
    And whose bonded_force_fragment(cx) returns Ok(None)
    And which the framework classifies as a bonded slot via the canonical builders contract
    When ForceField::new(...) is called
    Then it returns Err(ForceFieldError::MissingBondedFragment { label: <slot's label> })

  @rq-182549b3
  Scenario: A builder that returns Some(_) from both pair_force_fragment and bonded_force_fragment is rejected
    Given a custom builder whose pair_force_fragment(cx) returns Ok(Some(_))
    And whose bonded_force_fragment(cx) also returns Ok(Some(_))
    When ForceField::new(...) is called
    Then it returns Err(ForceFieldError::SlotMultipleShapes { label: <slot's label> })

  # --- Angle composition: construction ---

  @rq-911ec046
  Scenario: Angle composed module is compiled at ForceField::new when at least one angle slot is active
    Given a config with one [[angle_types]] entry "harmonic" and a non-empty AngleList
    And PotentialRegistry::with_builtins()
    When ForceField::new(...) is called
    Then force_field exposes a CudaFunction handle for "heddle_jit_composed_angle_0_f"
    And force_field exposes a CudaFunction handle for "heddle_jit_composed_angle_0_fev"

  @rq-6cd9aa19
  Scenario: Angle composed module is not compiled when no angle slot is active
    Given a config with no angle list
    When ForceField::new(...) is called
    Then no composed-angle module is loaded

  # --- Per-step launch ---

  @rq-9b913320
  Scenario: step() with one bonded slot launches the bonded composed kernel once and the reduction once
    Given a ForceField with exactly one bonded slot (Morse) active
    When force_field.step(...) is called with AggregateLevel::ForcesAndScalars
    Then timings records exactly one sample for KernelStage::JitComposedBondedForce
    And timings records exactly one sample for KernelStage::ReduceBondForces
    And timings records zero samples for KernelStage::MorseBondForce

  @rq-16ec7e24
  Scenario: step() with two bonded slots launches the bonded composed kernel twice
    Given a ForceField with two distinct bonded slots active in canonical order [A, B]
    When force_field.step(...) is called
    Then heddle_jit_composed_bonded_0_fev is dispatched once with A's args
    And heddle_jit_composed_bonded_1_fev is dispatched once with B's args
    And each slot's reduce_bond_forces is launched once

  @rq-10ae4fe6
  Scenario: step() with one angle slot launches the angle composed kernel once and the reduction once
    Given a ForceField with exactly one angle slot (HarmonicAngle) active
    When force_field.step(...) is called with AggregateLevel::ForcesAndScalars
    Then timings records exactly one sample for KernelStage::JitComposedAngleForce
    And timings records exactly one sample for KernelStage::ReduceAngleForces
    And timings records zero samples for KernelStage::HarmonicAngleForce

  @rq-c90f4105
  Scenario: step(ForcesOnly) launches the _f variant; step(ForcesAndScalars) launches the _fev variant
    Given a ForceField with at least one bonded slot
    When force_field.step(..., AggregateLevel::ForcesOnly) is called
    Then the bonded composed kernel's _f entry point is dispatched
    When force_field.step(..., AggregateLevel::ForcesAndScalars) is called
    Then the bonded composed kernel's _fev entry point is dispatched

  @rq-92b0af0b
  Scenario: step_class(Slow) does not launch the bonded or angle composed kernels
    Given a ForceField with both Fast and Slow slots active
    When force_field.step_class(ForceClass::Slow, ...) is called
    Then timings records zero samples for KernelStage::JitComposedBondedForce
    And timings records zero samples for KernelStage::JitComposedAngleForce

  # --- Correctness ---

  @rq-6d6e3553
  Scenario: Bonded composed-kernel output matches the closed-form Morse force for a non-boundary bond
    Given a ForceField with one Morse slot
    And two atoms in a non-boundary configuration whose Morse force is known a priori
    When force_field.step(...) is called
    Then forces_x[0] equals f_morse_x_on_0 within a relative tolerance of 1e-5
    And forces_x[1] equals -f_morse_x_on_0 within a relative tolerance of 1e-5

  @rq-3ad303e1
  Scenario: Two independent runs of the bonded composed kernel on identical inputs are byte-identical
    Given two ForceField instances built from byte-identical configurations with one Morse slot
    And two ParticleBuffers built from byte-identical ParticleStates
    When force_field.step(...) is called on each
    Then run A's forces_x, forces_y, forces_z, potential_energies, and virials agree
      byte-for-byte with run B's

  @rq-0b75dd43
  Scenario: Two independent runs of the angle composed kernel on identical inputs are byte-identical
    Given two ForceField instances built from byte-identical configurations with one HarmonicAngle slot
    And two ParticleBuffers built from byte-identical ParticleStates
    When force_field.step(...) is called on each
    Then run A's forces_x, forces_y, forces_z, potential_energies, and virials agree
      byte-for-byte with run B's

  # --- Error reporting ---

  @rq-ad105510
  Scenario: FragmentCompileFailed surfaces the bonded slot labels of every contributing fragment
    Given two active bonded fragments where one fragment's source contains a deliberate syntax error
    When ForceField::new(...) is called
    Then the returned Err's Display contains every active bonded slot's label
    And the underlying FragmentCompileFailed::log carries the nvrtc compile log verbatim

  # --- Argument schema ---

  @rq-790edb52
  Scenario: A bonded slot's entry_point_args are generated from its argument schema
    Given a bonded slot whose KernelArgSchema, built with KernelArgSchema::intramolecular, declares ("morse_bond_de", ConstPtrReal), ("morse_bond_a", ConstPtrReal), and ("morse_bond_re", ConstPtrReal) in order
    When the slot's BondedForceFragment is constructed
    Then fragment.entry_point_args equals "    const Real *morse_bond_de,\n    const Real *morse_bond_a,\n    const Real *morse_bond_re,\n"

  @rq-c4f93cfa
  Scenario: A bonded slot's functor_init_source uses local-functor initialisation
    Given a bonded slot whose intramolecular schema includes ("morse_bond_de", ConstPtrReal, functor_field "bond_de")
    When the slot's BondedForceFragment is constructed
    Then fragment.functor_init_source contains the line "    functor.bond_de = morse_bond_de;"
    And no line in fragment.functor_init_source contains "composite."
    And it contains exactly one assignment line per schema entry, in schema order

  @rq-075663ff
  Scenario: An angle slot's functor_init_source uses local-functor initialisation
    Given an angle slot whose intramolecular schema includes ("harmonic_angle_k_theta", ConstPtrReal, functor_field "angle_k_theta")
    When the slot's AngleForceFragment is constructed
    Then fragment.functor_init_source contains the line "    functor.angle_k_theta = harmonic_angle_k_theta;"
    And no line in fragment.functor_init_source contains "composite."

  @rq-7763d1ce
  Scenario: A bonded binding that pushes arguments out of order panics
    Given a bonded slot's KernelArgSchema declaring "morse_bond_de" then "morse_bond_a"
    And a KernelArgBinder over that schema and a fresh ForceLaunchBuilder
    When bind_bonded_force_args pushes "morse_bond_a" before "morse_bond_de"
    Then the binder panics with a message naming the slot and the expected argument "morse_bond_de"

  @rq-f7cc0a56
  Scenario: A bonded binding whose buffer element type disagrees with the schema panics
    Given a bonded slot's KernelArgSchema whose first argument is "morse_bond_de" (ConstPtrReal)
    And a KernelArgBinder over that schema
    When the binding pushes a CudaSlice<u32> for "morse_bond_de"
    Then the binder panics naming the slot and the argument

  @rq-4710429f
  Scenario: The schema-generated bonded signature compiles and binds consistently
    Given a fast-class bonded slot with a populated KernelArgSchema active in a ForceField
    When ForceField::new(...) is called
    Then nvrtc compiles the bonded composed module without error
    And at step time the slot's bind_bonded_force_args, validated against the same schema, supplies exactly one launch argument per generated parameter

  # --- Standalone-kernel retirement ---

  @rq-fa85466d
  Scenario: kernels/morse.cu does not declare a morse_bond_force entry point
    Given the project's kernel source tree
    When the bonded-shape standalone kernel symbols are enumerated
    Then no extern "C" kernel named morse_bond_force exists
    And reduce_bond_forces is declared as the only standalone bonded-shape kernel

  @rq-44554f9c
  Scenario: kernels/angle.cu does not declare a harmonic_angle_force entry point
    Given the project's kernel source tree
    When the angle-shape standalone kernel symbols are enumerated
    Then no extern "C" kernel named harmonic_angle_force exists
    And reduce_angle_forces is declared as the only standalone angle-shape kernel
```
