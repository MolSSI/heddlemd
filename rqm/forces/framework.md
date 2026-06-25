# Feature: Pluggable Potential Slot Framework <!-- rq-c1ea073b -->

The runner evaluates inter-particle forces through an ordered collection of
*potential slots* assembled into a `ForceField`. Each slot implements the
`Potential` trait, which exposes a single `compute` method that adds the
slot's per-particle contribution into its `ForceClass`'s accumulator
buffers (length `particle_count` per quantity). The `ForceField`'s
`step()` method runs every slot's `compute` in slot order and combines
the two class accumulators into the particle state's `forces_*`,
`potential_energies`, and `virials` arrays via a deterministic
class-combine kernel. The runner's force-evaluation step calls
`force_field.step(...)` once; it has no visibility into which potentials
participated.

A `PotentialRegistry` of `PotentialBuilder`s drives slot construction.
Each builder is an open-extensible factory that decides, from a parsed-config
+ topology context, whether it contributes a slot and (if so) constructs
one. `ForceField::new` is a fixed loop over the registry: it iterates the
builders in registration order, calls each builder's `build(cx)`, and
appends the returned slot when `Some(_)`. Adding a new built-in potential
is a one-line edit to `PotentialRegistry::with_builtins()`. Implementing
a new `Potential` requires no edits to `ForceField::new`, to the
class-combine kernel, or to any other potential's code.

Every slot belongs to one *frequency class* (`ForceClass::Fast` or
`ForceClass::Slow`) reported by its `Potential::frequency_class()`
method. The framework exposes two force-evaluation entry points:
`ForceField::step(...)` re-evaluates every slot regardless of class, and
`ForceField::step_class(class, ...)` re-evaluates only slots whose class
matches. Both entry points re-run the class-combine kernel across both
class accumulators so
`ParticleBuffers.forces_*`, `potential_energies`, and `virials` always
hold the most recent total. The class system carries the framework's
support for multi-time-step (RESPA-style) integrators that want to
evaluate slow forces (e.g. SPME reciprocal-space) less often than fast
forces (e.g. short-range pair).

## Slots <!-- rq-cc73f184 -->

`PotentialRegistry::with_builtins()` registers six built-in `PotentialBuilder`s,
each of which contributes at most one slot to the `ForceField` when its activation
condition is met. The registry's registration order is the slot evaluation
order; the registry is the single canonical source of slot ordering, and
`ForceField::new` reads from it without making its own decisions.

| Builder | `label()` of the slot it builds | Activation condition (`build(cx)` returns `Some(_)` iff …) | `frequency_class()` | `displaces()` | Implementation file |
| --- | --- | --- | --- | --- | --- |
| `LennardJonesBuilder` | `"lennard_jones"` | `cx.pair_interactions` is non-empty | `Fast` | `&[]` | `lj-pair-force.md` |
| `CoulombBuilder` | `"coulomb"` | `cx.coulomb_config.is_some()` | `Fast` | `&[]` | `coulomb-pair-force.md` |
| `SpmeRealBuilder` | `"spme_real"` | `cx.spme_config.is_some()` | `Fast` | `&[]` | `spme.md` |
| `SpmeReciprocalBuilder` | `"spme_reciprocal"` | `cx.spme_config.is_some()` | `Slow` | `&[]` | `spme.md` |
| `MorseBondedBuilder` | `"morse_bonded"` | `!cx.bond_list.is_empty()` | `Fast` | `&[]` | `morse-bonded.md` |
| `HarmonicAngleBuilder` | `"harmonic_angle"` | `!cx.angle_list.is_empty()` | `Fast` | `&[]` | `harmonic-angle.md` |

The two SPME builders share the same activation condition; they always
appear together because the Ewald split is exact only when both halves
are evaluated. The `[coulomb]` and `[spme]` tables are mutually exclusive
at config load (see `io/config-schema.md`); a `ForceField` therefore
contains at most one electrostatics path.

A `ForceField` with zero slots is a valid configuration. `step()` writes
zeros into `particle_buffers.forces_*` and returns without launching any
slot kernels.

When multiple slots are present, they appear in the `ForceField`'s slot
list in the order their builders are registered (after displacement
resolution suppresses any constituents claimed by another active
builder). The canonical built-in order is the order of the six rows
above:

1. `LennardJones`
2. `Coulomb`
3. `SpmeRealSpace`
4. `SpmeReciprocal`
5. `MorseBonded`
6. `HarmonicAngle`

A built-in potential is added by writing a `PotentialBuilder` and inserting
it at the appropriate position in `PotentialRegistry::with_builtins()`.
The `ForceField::new` body does not change.

Fast-class slots participate in one of three JIT-composed kernels
selected by parallelism shape:

- **Pair-force shape** — a slot whose `jit_participant()` returns
  `Some(JitParticipant::PairForce(_))` participates in the
  JIT-composed pair-force kernel (see `jit-composed-pair-force.md`).
  Such a slot also reports `max_cutoff() == Some(_)`, since it
  consumes the shared neighbour list.
- **Bonded shape** — a slot whose `jit_participant()` returns
  `Some(JitParticipant::Bonded(_))` participates in the JIT-composed
  bonded module (see `jit-composed-intramolecular.md`). One launch
  per active bonded slot from the shared module per step.
- **Angle shape** — a slot whose `jit_participant()` returns
  `Some(JitParticipant::Angle(_))` participates in the JIT-composed
  angle module (also `jit-composed-intramolecular.md`). One launch
  per active angle slot from the shared module per step.

The framework collects each shape's active fragments at
`ForceField::new` time, compiles a per-shape composed module via
nvrtc, and dispatches the per-slot launches from the composed
modules at step time. A slot whose `jit_participant()` returns
`None` — the slow-class SPME reciprocal slot, and any slot whose
contribution is not JIT-composed — is not covered by any composer
and dispatches via its own `Potential::compute` kernel call at step
time. Because `jit_participant()` returns a single `JitParticipant`
variant, a slot belongs to at most one shape by construction.

A composite fragment is built as a regular pair-force builder whose
`displaces()` list names the constituent slot labels whose fragments
it absorbs. The framework's resolution pass (see *Feature API*)
suppresses the named constituents' fragments from the composed
kernel and substitutes the composite's fragment in their place. The
composite's `build()` must inspect the same activation inputs as
each constituent and return `Ok(None)` whenever any constituent's
own activation condition is not met; the constituents' standalone
fragments then participate in the composed kernel unchanged, as if
no composite were registered.

## Force Classes <!-- rq-df6d79a1 -->

`ForceClass` is a two-variant enum that partitions every slot into one of
two evaluation cadences:

- `Fast` — short-range and inexpensive contributions: short-range pair
  forces (LJ, Coulomb, SPME real-space), bonded pair forces, three-body
  angle forces. A RESPA-style integrator evaluates Fast slots once per
  inner step.
- `Slow` — long-range and expensive contributions: SPME reciprocal-space
  (the FFT pipeline). A RESPA-style integrator evaluates Slow slots once
  per outer step.

Every concrete `Potential` reports its class via
`Potential::frequency_class()`. The default implementation returns
`ForceClass::Fast`; only `SpmeReciprocalState` overrides to `Slow`. A
user-defined `Potential` that does not override `frequency_class()` is
treated as `Fast`, which is the right default for short-range / bonded /
intramolecular contributions.

A `ForceField` refreshes a per-class accumulator buffer set when its
matching evaluation entry point runs:

- `ForceField::step(...)` re-evaluates every slot in slot order,
  refreshing every class's accumulator, then runs the class-combine
  kernel.
- `ForceField::step_class(class, ...)` re-evaluates only slots whose
  class matches, refreshing that class's accumulator, then runs the
  class-combine kernel. The non-matching class's accumulator retains
  its last-written contents.

In both cases the class-combine kernel sums the two per-class
accumulators into `ParticleBuffers.forces_*`, `potential_energies`,
and `virials`. Single-step integrators emit a `SubStep::ForceEval`
with `class: None` and consume the total; multi-step (RESPA)
integrators emit `class: Some(Fast)` many times and `class:
Some(Slow)` once per outer step, and the per-particle total visible at
each kick reflects the most recent evaluation of every class — Slow
contributions are stale by up to `n − 1` inner steps, exactly the
RESPA approximation.

The two SPME builders contribute slots in different classes (Real → Fast,
Reciprocal → Slow). The Ewald split remains exact only when both classes
have been evaluated at the same simulation time; integrators that mix
`step_class(Fast)` calls with stale Slow contributions are using the
RESPA approximation, not a different splitting of the Ewald sum.

## Force Evaluation Pipeline <!-- rq-7bab5c1e -->

Each `ForceField::step(...)` or `ForceField::step_class(class, ...)`
call performs the following, in order:

1. **Shared neighbor-list update.** If `ForceField::neighbor_list` is
   `Some`, call its `pre_step` method (see `neighbor-list.md`). In
   cell-list mode this runs the displacement-check kernel and rebuilds
   the neighbor list when an atom's reference displacement exceeds
   `r_skin / 2`. In trivial mode and when `neighbor_list` is `None`,
   this step launches no kernels. The update runs at the cadence of
   whichever entry point is called: every `step` / `step_class` call
   may trigger a rebuild.
2. **Class filter.** Restrict the slot iteration to slots whose
   `frequency_class()` matches the entry-point's class selector:
   - `step(...)` and `step_class(None, ...)` (when offered as `step`'s
     default) iterate every slot.
   - `step_class(class, ...)` iterates only slots whose
     `frequency_class() == class`.
3. **Per-class accumulator reset.** For each class `C` being evaluated
   in this call, zero the buffers in `C`'s accumulator that this call
   will refresh:
   - When `level == AggregateLevel::ForcesAndScalars`: zero all five
     buffers (`{C}_total_forces_x/y/z`, `{C}_total_potential_energies`,
     `{C}_total_virials`).
   - When `level == AggregateLevel::ForcesOnly`: zero only the three
     force buffers (`{C}_total_forces_x/y/z`). The class's energy and
     virial accumulators retain whatever the most recent
     `ForcesAndScalars` evaluation of class `C` wrote into them.

   The non-evaluated class's accumulator buffers are not touched: a
   `step_class(Fast, ...)` call leaves the `slow_total_*` buffers
   unchanged.
4. **Per-slot compute.** For each selected slot, in canonical slot
   order, invoke `Potential::compute`, passing a `ForceFieldContext`
   that carries a reference to the shared `NeighborListState` (when
   present) and any other shared services, a `SlotOutputView` that
   points to its class's accumulator buffers (force-x, force-y,
   force-z, energy, virial — each a slice of length `particle_count`),
   and the call's `AggregateLevel`. The implementation runs whatever
   kernel(s) it needs to evaluate its contribution and **adds** to
   the `SlotOutputView`: it adds the per-particle force-component
   contributions on every call regardless of level, and additionally
   adds the per-particle energy and virial contributions when
   `level == AggregateLevel::ForcesAndScalars`. When `level ==
   AggregateLevel::ForcesOnly` no slot kernel writes to the energy or
   virial slices; the class's energy / virial accumulator retains the
   value from the most recent `ForcesAndScalars` evaluation.
   Determinism of the per-class total is preserved because the slot
   dispatch order is fixed and each slot's add to a given particle is
   performed by exactly one warp per particle on the default stream,
   so the sequence of adds into each `(class, axis, particle)`
   accumulator cell matches across runs.
5. **Class combine.** Run `combine_class_totals` once. The kernel
   writes, for each per-particle quantity `Q` in `{force_x, force_y,
   force_z, potential_energy, virial}`:
   `particle_buffers.Q[i] = fast_total_Q[i] + slow_total_Q[i]`. One
   thread per particle; each thread reads ten floats, performs five
   additions, and writes five floats. The kernel runs on every
   `step` / `step_class` call regardless of level; the level only
   affects which class-accumulator buffers steps 3 and 4 refreshed.

Identical runs on the same GPU with the same config and the same
sequence of `step` / `step_class` calls — including the same
`AggregateLevel` value at each call site — produce byte-identical
`particle_buffers.forces_*`, `potential_energies`, and `virials`, and
therefore byte-identical trajectories. A change in the cadence at
which `ForcesAndScalars` versus `ForcesOnly` is requested is a
configuration change, not a non-determinism: two runs that issue the
same sequence of (call kind, level) pairs are reproducible; two runs
that differ in that sequence produce different `potential_energies`
and `virials` at the steps where they diverge, exactly as expected.

## Class Output Accumulators <!-- rq-cd28340e -->

`ForceField` owns two parallel sets of per-class accumulator buffers
per `ForceClass`, one for slots that write `Real` values via
read-modify-write and one for slots that write integer atomics in
fixed-point:

**Real accumulators** (consumed by bonded reduce kernels and any
slot whose contribution is naturally additive in `Real`):

- `Fast`: `fast_total_forces_x`, `fast_total_forces_y`,
  `fast_total_forces_z`, `fast_total_potential_energies`,
  `fast_total_virials` — each a `CudaSlice<Real>` of length
  `particle_count`.
- `Slow`: `slow_total_forces_x`, `slow_total_forces_y`,
  `slow_total_forces_z`, `slow_total_potential_energies`,
  `slow_total_virials` — each a `CudaSlice<Real>` of length
  `particle_count`.

**Fixed-point accumulators** (consumed by the packed-neighbour
JIT-composed pair-force kernel via `atomicAdd`; see
`packed-neighbour-pair-force.md` *Fixed-Point Force Buffers*):

- `Fast`: `fast_total_forces_fp_x`, `fast_total_forces_fp_y`,
  `fast_total_forces_fp_z`, `fast_total_potential_energies_fp`,
  `fast_total_virials_fp` — each a `CudaSlice<u64>` of length
  `particle_count`. Interpreted as two's-complement `i64` with
  scale `2^32`.
- `Slow`: analogous, for slow-class slots that opt into the
  fixed-point accumulator (e.g. an SPME-real slot in slow mode).
  SPME-recip continues to use the `Real` accumulator because its
  output is not pair-force shaped.

Each `Real` accumulator holds the per-particle sum across every
slot in its class that writes via `Real` read-modify-write, as
written by the most recent evaluation entry point that refreshed
it. The `SlotOutputView` passed to such a slot's `compute(...)`
points to the slot's class's `Real` accumulator slices (the
`{class}_total_forces_x/y/z`, plus `potential_energies` and
`virials`). The slot kernel reads the existing value at each
particle, adds its own contribution, and writes the sum back.

Each fixed-point accumulator holds the per-particle sum across
every pair-force slot in its class. Pair-force slots do not receive
a `SlotOutputView`; they are bound into the JIT-composed
pair-force kernel (see `packed-neighbour-pair-force.md` *JIT
Composer Integration*), which `atomicAdd`s into the class's
fixed-point buffers directly. Integer addition is associative so
the per-atom sum is bit-exact across runs regardless of how many
warps contributed.

The framework zeroes the accumulator buffers being refreshed before
the first slot in a class adds to them (see step 3 of the *Force
Evaluation Pipeline*). The `Real` accumulator is zeroed via
`cudaMemsetAsync(0)`; the fixed-point accumulator is zeroed via
`cudaMemsetAsync(0)` (an all-zero `u64` value is the integer
representation of fixed-point `0.0`). The per-class accumulator
after step 4 contains exactly the sum of just-evaluated slot
contributions for that class, partitioned across the two parallel
sets.

The class-combine step reads BOTH sets of accumulators, converts
the fixed-point sum to `Real` via the
`packed-neighbour-pair-force.md` *Fixed-Point Force Buffers*
conversion (`fixed_to_real(s) = ((i64) s) / 2^32`), and writes the
combined `Real` total into `ParticleBuffers.forces_*`,
`ParticleBuffers.potential_energies`, and `ParticleBuffers.virials`.

Memory cost: `10 · particle_count · 4` bytes for the `Real` set
plus `10 · particle_count · 8` bytes for the fixed-point set =
`120 · particle_count` bytes, independent of slot count. For
`particle_count = 10⁴` this is ~1.2 MB; for `particle_count = 10⁵`,
~12 MB.

When `particle_count == 0` every accumulator buffer has length zero.
Accumulator buffers are zero-initialised at construction so that the
class-combine kernel reads valid zero contributions for any class
that has not yet been evaluated.

## Empty State <!-- rq-aa52268c -->

When `particle_count == 0`, every slot's `compute` method
early-returns without launching, and the class-combine kernel returns
without launching. `ForceField::step` and `ForceField::step_class`
return `Ok(())` having done no GPU work.

When the slot list is empty (across every class), the framework's
accumulator-zeroing memsets and the `combine_class_totals` kernel
still run: the memsets zero both classes' accumulators, and the
class-combine kernel sums the two zero buffers and writes zeros to
every entry of `particle_buffers.forces_*`, `potential_energies`,
and `virials`. `ForceField::step` returns `Ok(())`.

When `step_class(class, ...)` is called and the `ForceField` contains
zero slots in that class, the call is a no-op: it launches no kernels
(no accumulator memset, no slot compute, no class combine) and leaves
`ParticleBuffers.forces_*` untouched. The semantics are correct
because nothing to recompute means the existing total is already
current.

When a slot's *input list* is empty (e.g. a `MorseBonded` slot
constructed with `bonds.is_empty()`), the slot's `compute` returns
without launching any kernel — its add to the class accumulator is
the additive identity. The rest of the pipeline runs normally.

## Feature API <!-- rq-0da87ca1 -->

### Types <!-- rq-e4960f89 -->

- `ForceClass` — two-variant enum partitioning every slot into one of <!-- rq-c4861786 -->
  two evaluation cadences.

  ```rust
  #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
  pub enum ForceClass {
      Fast,
      Slow,
  }
  ```

  - `Fast` is the default class returned by `Potential::frequency_class()`.
  - `Slow` is for long-range / FFT-driven contributions (today: SPME
    reciprocal-space). RESPA integrators evaluate `Slow` slots less
    frequently than `Fast` slots.
  - The set of variants is closed. Extending to a third class (e.g.
    RESPA-3's "extra-slow") is a deliberate API change, not a default
    extension point.

- `AggregateLevel` — two-variant enum that selects whether the <!-- inline --> <!-- rq-81ac7d6a -->
  framework's per-step force-evaluation pipeline aggregates only the
  force components or also the scalar quantities (energy, virial).

  ```rust
  #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
  pub enum AggregateLevel {
      ForcesOnly,
      ForcesAndScalars,
  }
  ```

  - `ForcesOnly` (the cheap case, ~3/5 of the per-call reduction work)
    runs only the force-component reductions. The energy and virial
    entries of every per-class accumulator retain whatever values the
    most recent `ForcesAndScalars` call wrote into them.
  - `ForcesAndScalars` runs both the force-component reductions and
    the scalar (energy + virial) reductions.
  - The variant set is closed. Adding a third level requires editing
    every consumer; it is not an open extension point.

- `Potential` — object-safe trait implemented by every slot. <!-- rq-67ebf3b1 -->

  ```rust
  pub trait Potential: std::fmt::Debug + Send {
      fn label(&self) -> &'static str;

      fn max_cutoff(&self) -> Option<f32>;

      fn frequency_class(&self) -> ForceClass {
          ForceClass::Fast
      }

      fn compute(
          &mut self,
          buffers: &ParticleBuffers,
          sim_box: &SimulationBox,
          output: SlotOutputView<'_>,
          cx: &ForceFieldContext<'_>,
          timings: &mut Timings,
          level: AggregateLevel,
      ) -> Result<(), ForceFieldError>;

      fn jit_participant(&self) -> Option<JitParticipant<'_>> {
          None
      }
  }
  ```

  - `label` returns a short, stable, lower-snake-case identifier (for
    example `"lennard_jones"`, `"morse_bonded"`) used in error messages
    and diagnostic output. Two slots in the same `ForceField` must have
    distinct labels.
  - `max_cutoff` returns the maximum short-range interaction cutoff the
    potential needs the shared neighbor list to cover, in the same units
    as `SimulationBox`. Returns `None` if the potential does not consume
    the neighbor list (bonded potentials, intramolecular potentials
    keyed by index, etc.). `ForceField::new` aggregates these values to
    size the shared `NeighborListState`.
  - `frequency_class` reports the class at which the framework should
    evaluate this slot. Provided default returns `ForceClass::Fast`;
    only slots whose contribution is expensive enough to justify a
    different cadence override. Today only `SpmeReciprocalState`
    overrides, returning `ForceClass::Slow`.
  - `compute` runs the slot's evaluation kernel(s) against the current
    `ParticleBuffers` and `SimulationBox` and writes the per-particle
    result directly into `output`. The slot reads any shared resources
    it needs from `cx`. Implementations may read from `buffers` but
    must not write to it. Implementations that report
    `max_cutoff() == Some(_)` may assume `cx.neighbor_list` is
    `Some(_)`; implementations that report `None` are free to ignore
    `cx.neighbor_list` (it may be `None` or `Some` depending on whether
    any other slot needs the list).

    `compute` **adds** its per-particle contribution into the slices
    referenced by `output`. The framework has already zeroed the
    relevant slices for the first slot of each class to add to (see
    *Force Evaluation Pipeline* step 3), so the slot kernel may treat
    its add as a read-existing-add-write of one per-particle value
    per slice. The slot kernel adds to all three force-component
    slices on every call regardless of `level`; it additionally adds
    to the energy and virial slices only when `level ==
    AggregateLevel::ForcesAndScalars`. When `level ==
    AggregateLevel::ForcesOnly` the slot's compute must not write to
    the energy or virial slices — those slices hold the class's
    accumulated value from the most recent `ForcesAndScalars`
    evaluation and must be preserved across the `ForcesOnly` call.
    Slots whose underlying kernel cannot cheaply split along the
    force / scalar boundary (every bonded slot, e.g. `MorseBondedState`,
    `HarmonicAngleState`) take an explicit `level` parameter into
    their kernel and gate the energy / virial writes on it; the
    force-and-virial computation in registers is the same in both
    paths, only the device store is gated.

  Implementations are responsible for emitting their own
  `KernelStage` start/stop events through `timings`.

  For slots whose `frequency_class()` is `Fast` AND whose
  `max_cutoff()` is `Some(_)`, the framework bypasses `compute` at
  step time and dispatches the JIT-composed pair-force kernel
  instead (see `jit-composed-pair-force.md`). `compute` on such
  slots is reserved for standalone testing (e.g. test fixtures
  that drive the per-potential kernel directly without going
  through the framework's composed-kernel pipeline).

  `jit_participant` declares whether the slot contributes to a
  JIT-composed kernel and, if so, in which shape. It returns
  `Some(JitParticipant::PairForce(self))`,
  `Some(JitParticipant::Bonded(self))`, or
  `Some(JitParticipant::Angle(self))` from a slot that implements the
  corresponding capability trait, and `None` (the default) from a slot
  that runs only through `compute` (the slow-class SPME reciprocal
  slot, and any slot whose contribution is not JIT-composed). Because
  the return type is a single enum, a slot participates in at most one
  shape by construction. The framework collects each participant's
  fragment at `ForceField::new` and dispatches its bind at step time
  through the capability trait the variant carries; a slot that returns
  `None` has its `compute` called instead. See
  `jit-composed-pair-force.md` and `jit-composed-intramolecular.md`
  for the per-shape contracts.

- `JitParticipant<'a>` — single-shape tag a slot returns from <!-- rq-8571bd3e -->
  `Potential::jit_participant` to declare its JIT-composed contribution.

  ```rust
  pub enum JitParticipant<'a> {
      PairForce(&'a dyn PairForcePotential),
      Bonded(&'a dyn BondedPotential),
      Angle(&'a dyn AnglePotential),
  }
  ```

  Each variant borrows the slot itself as the matching capability trait
  object. A slot returns at most one variant, so participation in more
  than one shape is structurally impossible — there is no runtime
  multi-shape check.

- `PairForcePotential` — capability trait a pair-force slot implements <!-- rq-e533174d -->
  to contribute to the JIT-composed pair-force kernel. Carries both the
  slot's source fragment and its launch-time argument binding, so a slot
  cannot provide one without the other.

  ```rust
  pub trait PairForcePotential {
      fn pair_force_fragment(&self) -> PairForceFragment;
      fn bind_pair_force_args(
          &self,
          ctx: &PairForceBindContext<'_>,
          builder: &mut ForceLaunchBuilder,
      );
  }
  ```

  Neither method has a default: a type that implements
  `PairForcePotential` must supply both. `pair_force_fragment` returns
  the fragment the composer concatenates at `ForceField::new`;
  `bind_pair_force_args` pushes the slot's parameters through a
  `KernelArgBinder` at each launch. See `jit-composed-pair-force.md`.

- `BondedPotential` — capability trait a bonded slot implements, <!-- rq-d7ddc1ac -->
  carrying the slot's fragment, its per-bond scratch view, and its
  argument binding.

  ```rust
  pub trait BondedPotential {
      fn bonded_force_fragment(&self) -> BondedForceFragment;
      fn bonded_scratch(&self) -> BondedScratchView<'_>;
      fn bind_bonded_force_args(
          &self,
          ctx: &ForceLaunchContext<'_>,
          builder: &mut ForceLaunchBuilder,
      );
  }
  ```

  No method has a default. See `jit-composed-intramolecular.md`.

- `AnglePotential` — capability trait an angle slot implements, <!-- rq-da327920 -->
  carrying the slot's fragment, its per-angle scratch view, and its
  argument binding.

  ```rust
  pub trait AnglePotential {
      fn angle_force_fragment(&self) -> AngleForceFragment;
      fn angle_scratch(&self) -> AngleScratchView<'_>;
      fn bind_angle_force_args(
          &self,
          ctx: &ForceLaunchContext<'_>,
          builder: &mut ForceLaunchBuilder,
      );
  }
  ```

  No method has a default. See `jit-composed-intramolecular.md`.

- `PairForceFragment` — self-contained CUDA C++ source fragment plus <!-- rq-aa6efe11 -->
  identifying metadata, returned by
  `PairForcePotential::pair_force_fragment()`. Fields:

  ```rust
  pub struct PairForceFragment {
      pub label: &'static str,
      pub functor_struct_name: &'static str,
      pub functor_source: String,
      pub entry_point_args: String,
      pub functor_init_source: String,
  }
  ```

  The full contract on what a fragment must contain (functor
  struct shape, precision-shim usage, helper-name namespacing) is
  in `jit-composed-pair-force.md`.

- `BondedForceFragment` — self-contained CUDA C++ source fragment <!-- rq-b77f10a0 -->
  plus identifying metadata, returned by
  `BondedPotential::bonded_force_fragment()`. Same field set as
  `PairForceFragment`. The functor's contract differs (per-bond
  evaluation vs per-pair); see `jit-composed-intramolecular.md`.

- `AngleForceFragment` — same shape as `BondedForceFragment`, <!-- rq-6024a35a -->
  returned by `AnglePotential::angle_force_fragment()`. The
  functor's contract is the angle shape from
  `jit-composed-intramolecular.md`.

- `ForceLaunchBuilder` — opaque argument-builder threaded through <!-- rq-3aa5f5b8 -->
  every active fast-class slot's bind method
  (`bind_pair_force_args`, `bind_bonded_force_args`,
  `bind_angle_force_args`). Shape-agnostic — the binding mechanism
  is the same across the pair-force, bonded, and angle composers.
  Constructed by the framework once per composed-kernel launch and
  pre-populated with the launch's common arguments. Slots push
  their own arguments via:

  ```rust
  impl ForceLaunchBuilder {
      pub fn push_device_buffer<T>(&mut self, buf: &CudaSlice<T>);
      pub fn push_scalar<T: Copy>(&mut self, value: T);
  }
  ```

  See `jit-composed-pair-force.md` and
  `jit-composed-intramolecular.md` for the per-launch contracts.

- `ForceLaunchContext<'a>` — per-launch context passed to <!-- rq-538707ad -->
  `BondedPotential::bind_bonded_force_args` and
  `AnglePotential::bind_angle_force_args`. The pair-force bind takes
  `PairForceBindContext<'a>` instead (defined in
  `jit-composed-pair-force.md`); both carry the read-only per-launch
  inputs the slot needs. See the per-shape composer files for the
  exact fields.

- `ForceFieldContext<'a>` — bundle of shared services that the framework <!-- inline --> <!-- rq-559783fe -->
  exposes to every `compute` call. Constructed by `ForceField::step`
  for the duration of one step's compute phase. Fields:

  ```rust
  pub struct ForceFieldContext<'a> {
      pub neighbor_list: Option<&'a NeighborListState>,
  }
  ```

  - `neighbor_list` is `Some(_)` when at least one slot reports
    `max_cutoff() == Some(_)` at construction; otherwise `None`. New
    shared services land here as additional fields without changing the
    `Potential::compute` signature.

- `SlotOutputView<'a>` — five exclusive references to per-particle <!-- rq-304b191b -->
  accumulator slices, each of length `particle_count`. Constructed by
  `ForceField` and passed into `Potential::compute`. Implementations
  treat the slices as read-modify-write accumulators: the slot's
  kernel adds its per-particle contribution to the current value
  rather than overwriting it.

  ```rust
  pub struct SlotOutputView<'a> {
      pub force_x: CudaViewMut<'a, f32>,
      pub force_y: CudaViewMut<'a, f32>,
      pub force_z: CudaViewMut<'a, f32>,
      pub energy: CudaViewMut<'a, f32>,
      pub virial: CudaViewMut<'a, f32>,
  }
  ```

  Each field is a `CudaViewMut` of length `particle_count` onto the
  corresponding per-class accumulator buffer of the framework — the
  class is determined by the slot's `frequency_class()`. Fast-class
  slots receive views onto `fast_total_*`; Slow-class slots receive
  views onto `slow_total_*`. The view borrows the framework's storage
  for the duration of the `compute` call.

- `LennardJonesState` — implements `Potential` with `label() == "lennard_jones"` and `frequency_class() == ForceClass::Fast` (the trait default). <!-- rq-af2d1628 -->
  Owns the slot's `LennardJonesParameters` and the
  `DeviceExclusionList` (see `topology.md`). The slot consumes the
  shared `NeighborListState` (see `neighbor-list.md`); its on-host
  state carries only the parameter tables. The packed-neighbour
  data structures (`interacting_tiles`, `interacting_atoms`,
  `single_pairs`, fixed-point accumulators) are owned by
  `NeighborListState` / `ForceField` and shared across every
  fast-class pair-force slot — see
  `packed-neighbour-pair-force.md`.

  The slot does not launch its own pair-force kernel. It contributes
  a `PairForceFragment` to the JIT-composed pair-force pipeline
  (see `jit-composed-pair-force.md`), which evaluates every active
  fast-class pair-force slot's contribution per pair in a single
  composed kernel launch and atomicAdds the result to the class's
  fixed-point accumulator. The slot's `compute` method is therefore
  a no-op at step time; the per-step force-evaluation work happens
  in the composed kernel.
  directly into the `SlotOutputView` it receives. See
  `pair-force-kernel.md` for the warp-per-particle pattern and
  `lj-pair-force.md` for the per-pair functional form.

- `MorseBondedState` — implements `Potential` with `label() == "morse_bonded"` and `frequency_class() == ForceClass::Fast` (the trait default). <!-- rq-2361f2b8 -->
  Owns the slot's `BondPairBuffer`, the bond index/offset tables, and
  the per-bond-type parameter table. Construction requires a non-empty
  bond list; see `morse-bonded.md`. Its `compute` runs the bonded
  contribution kernel followed by the bonded reduction kernel and
  writes its per-particle output into the `SlotOutputView` it receives.

- `HarmonicAngleState` — implements `Potential` with `label() == "harmonic_angle"` and `frequency_class() == ForceClass::Fast` (the trait default). <!-- rq-454ad2cf -->
  Owns the slot's `AnglePairBuffer`, the angle index/offset tables,
  and the per-angle-type parameter table. Construction requires a
  non-empty angle list; see `harmonic-angle.md`. Its `compute` runs
  the angle contribution kernel followed by the angle reduction kernel
  and writes its per-particle output into the `SlotOutputView` it
  receives.

- `ForceField` — handle owning the slot collection, the per-class accumulator buffers, and the shared neighbor list. <!-- rq-684a29f1 -->

  Fields:
  - `device: Arc<CudaDevice>`
  - `slots: Vec<Box<dyn Potential>>` — in canonical evaluation order
    (the order produced by `PotentialRegistry::with_builtins()`). May
    be empty.
  - `fast_total_forces_x: CudaSlice<f32>` — length `N`.
  - `fast_total_forces_y: CudaSlice<f32>` — length `N`.
  - `fast_total_forces_z: CudaSlice<f32>` — length `N`.
  - `fast_total_potential_energies: CudaSlice<f32>` — length `N`.
  - `fast_total_virials: CudaSlice<f32>` — length `N`.
  - `slow_total_forces_x: CudaSlice<f32>` — length `N`.
  - `slow_total_forces_y: CudaSlice<f32>` — length `N`.
  - `slow_total_forces_z: CudaSlice<f32>` — length `N`.
  - `slow_total_potential_energies: CudaSlice<f32>` — length `N`.
  - `slow_total_virials: CudaSlice<f32>` — length `N`.
  - `neighbor_list: Option<NeighborListState>` — `Some(_)` when at
    least one slot returns `max_cutoff() == Some(_)`, `None`
    otherwise (bonded-only and zero-slot configurations).

  Buffers are of length `N` regardless of how many slots populate
  them: every slot in a class adds into the same accumulator. The
  canonical slot order within a class determines the order in which
  adds happen, which fixes the float-summation order and preserves
  byte-reproducibility.

- `ForceFieldError` — error type. Variants: <!-- rq-a2e20b02 -->
  - `Gpu(GpuError)` — CUDA driver / kernel-launch failure from any
    slot's kernel, an accumulator memset, or the class-combine kernel.
  - `Timings(TimingsError)` — CUDA event recording failure.
  - `NeighborList(NeighborListError)` — surfaces failures from the
    cell-list pipeline (see `neighbor-list.md`), including the
    `NeighborListOverflow` and `BoxTooSmallForCells` cases.
  - `DuplicateLabel(&'static str)` — two slots constructed with the
    same `label()`. Reported from `ForceField::new`.
  - `DisplaceConflict { label: &'static str, by: Vec<&'static str> }`
    — two or more built slots' `displaces()` lists both claim the same
    constituent `label`, and that constituent label is itself present
    among the built slots. `by` names the labels of the claimers (in
    registration order). Reported from `ForceField::new`'s
    displacement-resolution pass.
  - `FragmentCompileFailed { log: String }` — nvrtc rejected one of
    the JIT-composed module sources (pair-force, bonded, or angle).
    `log` carries the full nvrtc compile log; the error's `Display`
    impl additionally names every contributing fragment's slot
    label so the caller can identify which fragment is the likely
    culprit.
  - `FragmentLoadFailed(GpuError)` — `load_ptx` rejected the
    compiled PTX of one of the JIT-composed modules.

- `PotentialBuildContext<'a>` — bundle of borrowed references to every <!-- rq-d116af5f -->
  parsed-config and topology input a built-in `PotentialBuilder` might
  read. Built once by `ForceField::new` and passed by reference to each
  builder's `build(cx)` call. Fields:

  ```rust
  pub struct PotentialBuildContext<'a> {
      pub gpu: &'a GpuContext,
      pub particle_count: usize,
      pub sim_box: &'a SimulationBox,
      pub particle_types: &'a [ParticleTypeConfig],
      pub pair_interactions: &'a [PairInteractionConfig],
      pub bond_types: &'a [BondTypeConfig],
      pub angle_types: &'a [AngleTypeConfig],
      pub coulomb_config: Option<&'a CoulombConfig>,
      pub spme_config: Option<&'a SpmeConfig>,
      pub charges: &'a [f32],
      pub bond_list: &'a BondList,
      pub angle_list: &'a AngleList,
      pub exclusion_list: &'a ExclusionList,
      pub neighbor_list_config: &'a NeighborListConfig,
  }
  ```

  Each builder reads only the fields it needs. The context is distinct
  from `ForceFieldContext`, which is the per-step context handed to
  `Potential::compute`.

- `PotentialBuilder` — object-safe trait implemented by every potential's <!-- rq-e8550f96 -->
  factory. Each builder is responsible for at most one slot.

  ```rust
  pub trait PotentialBuilder:
      PotentialBuilderClone + std::fmt::Debug + Send + Sync
  {
      fn build(
          &self,
          cx: &PotentialBuildContext<'_>,
      ) -> Result<Option<Box<dyn Potential>>, ForceFieldError>;

      fn displaces(&self) -> &'static [&'static str] {
          &[]
      }
  }
  ```

  A builder produces a slot from the build context; the slot itself
  declares its JIT-composed participation and carries its source
  fragment (see `Potential::jit_participant` and the capability traits
  above). The builder has no fragment methods. `PotentialBuilder` does
  **not** carry the `KindedBuilder` bound — potentials are activated
  compositionally by configuration presence, not selected by a `kind`
  key — so `PotentialRegistry` has no `lookup`. The generated
  `PotentialBuilderClone` supertrait provides boxed-trait-object cloning;
  a builder needs only `#[derive(Clone)]`. See `registry-framework.md`.

  - `build` inspects `cx` and returns `Ok(Some(slot))` if this builder's
    activation condition (see *Slots*) is satisfied, or `Ok(None)` if
    not. `Err` is reserved for genuine construction failures (GPU
    allocation, malformed inputs that survived config validation, etc.).
  - Two distinct builders may not produce slots with the same
    `Potential::label()`. The framework enforces this in
    `ForceField::new`; builders themselves do not need to check.
  - `displaces` returns a list of constituent slot labels whose work
    this builder absorbs. The default implementation returns `&[]`,
    meaning the builder is a standalone potential that does not
    displace anything. A composite builder overrides this to name the
    constituent labels it replaces. The names are matched against
    `Potential::label()` of every built slot. Naming a label that no
    builder ends up producing is harmless — the displacement claim
    has no effect. Naming a label that *is* produced suppresses the
    constituent slot from the final slot list, and additionally
    suppresses the constituent's source fragment from the
    JIT-composed pair-force kernel when both the composite and the
    constituent are fast-class pair-force slots. The composite is
    responsible for inspecting `cx` and returning `Ok(None)` from
    `build()` whenever any of its constituents' activation
    conditions are not met, so the lone constituent's standalone
    slot continues to run; the framework does not silently fall back
    on the composite's behalf.

- `PotentialRegistry` — `Registry<dyn PotentialBuilder>` (the generic <!-- rq-50f0a96a -->
  container; see `registry-framework.md`). A compositional-activation
  registry: it carries no keyed `lookup`, and registration order is the
  slot evaluation order. `with_builtins()` pre-populates the six built-in
  builders in canonical evaluation order — `LennardJonesBuilder`,
  `CoulombBuilder`, `SpmeRealBuilder`, `SpmeReciprocalBuilder`,
  `MorseBondedBuilder`, `HarmonicAngleBuilder`. `ForceField::new`
  iterates `registry.builders()` and builds each against the
  `PotentialBuildContext`, collecting every builder that activates.

  `PotentialRegistry` is also reachable as the `potentials` field of
  the runner-level `heddle_md::Registries` bundle (see
  `simulation-runner.md`). The runner's
  `run_simulation_with_registries` entry point reads the bundle's
  `potentials` field instead of constructing
  `PotentialRegistry::with_builtins()` internally, so custom
  potential builders flow through the same path as built-ins.

### Functions and methods <!-- rq-17abcb76 -->

- `ForceField::new(registry: &PotentialRegistry, gpu: &GpuContext, particle_count: usize, sim_box: &SimulationBox, particle_types: &[ParticleTypeConfig], pair_interactions: &[PairInteractionConfig], bond_types: &[BondTypeConfig], angle_types: &[AngleTypeConfig], coulomb_config: Option<&CoulombConfig>, spme_config: Option<&SpmeConfig>, charges: &[f32], bond_list: &BondList, angle_list: &AngleList, exclusion_list: &ExclusionList, neighbor_list_config: &NeighborListConfig) -> Result<ForceField, ForceFieldError>` <!-- rq-79938dbf -->
  - Builds a `PotentialBuildContext` populated from every parameter
    listed above (apart from `registry`).
  - Iterates `registry.builders` in registration order. For each builder,
    calls `builder.build(&cx)`. When the call returns `Ok(Some(slot))`,
    records the slot together with the builder's `displaces()` list
    and the builder's registration index. `Ok(None)` is the no-op skip
    path (this builder's activation condition was not met). Any
    `Err(_)` short-circuits and is returned unchanged.
  - After every builder has been consulted, runs the displacement
    resolution pass:
    1. Collects the set of constituent labels claimed by at least one
       built slot's `displaces()` list. For each such label, also
       records the list of *built* slot labels whose builders claimed
       it.
    2. For every claimed label whose set of claimers has size > 1
       *and* whose label is itself present among the built slots,
       returns
       `ForceFieldError::DisplaceConflict { label, by: <claimer
       labels> }`. A claim against a label that no built slot carries
       does not count toward the conflict — the displacement is
       informational and harmless.
    3. Filters the built slot list, removing any slot whose label
       appears in the claimed set produced by step 1. Slots that are
       suppressed this way are dropped permanently for this
       `ForceField`; the framework launches no kernels on their behalf
       and allocates no per-slot state for them.
  - Appends the surviving slots to the `ForceField`'s slot list in
    their registration order. A composite slot whose constituents
    have all been suppressed appears at the position determined by its
    own builder's registration index, *not* at any constituent's
    position.
  - When no builder produces a slot, returns a `ForceField` with
    `slots.len() == 0`.
  - Allocates the ten per-class accumulator buffers
    (`fast_total_forces_x/y/z`, `fast_total_potential_energies`,
    `fast_total_virials`, and the matching `slow_total_*` set) of
    length `particle_count` on `gpu.device`. Each is zero-initialised
    so the class-combine kernel reads valid zero contributions for
    any class that has not yet been evaluated. When
    `particle_count == 0`, the allocations are length-zero.
  - Builds the shared `NeighborListState`:
    - Computes `r_cut = max(slot.max_cutoff() for slot in slots if
      slot.max_cutoff().is_some())`. If no slot reports a cutoff,
      `neighbor_list` is set to `None` and the framework launches no
      neighbor-list kernels for the lifetime of the run.
    - Otherwise consults `neighbor_list_config`:
      - `CellList { r_skin, tile_pair_capacity }`: calls
        `NeighborListState::new_cell_list(gpu, sim_box,
        particle_count, r_cut, r_skin as f32, tile_pair_capacity)`.
        May return `ForceFieldError::NeighborList(_)` (e.g.
        `BoxTooSmallForCells`).
      - `AllPairs`: calls `NeighborListState::new_trivial(gpu,
        sim_box, particle_count)`.
  - Returns `ForceFieldError::DuplicateLabel(_)` if two slots end up
    with the same `label()`.

- `ForceField::step(&mut self, buffers: &mut ParticleBuffers, sim_box: &SimulationBox, timings: &mut Timings, level: AggregateLevel) -> Result<(), ForceFieldError>` <!-- rq-3579df3b -->
  - Evaluates every slot regardless of class. Equivalent to invoking
    `step_class` once for each class present, except that the
    neighbor-list update and the class-combine kernel each run
    exactly once.
  - When `self.neighbor_list` is `Some(nl)`, calls `nl.pre_step(sim_box,
    buffers, timings)` to run the generation-cache check, the
    displacement check, and the rebuild as needed. When `None`, this
    step is skipped.
  - Constructs a `ForceFieldContext { neighbor_list:
    self.neighbor_list.as_ref() }` valid for the duration of the
    compute phase.
  - For each class `C` in `{Fast, Slow}` that has at least one slot,
    in that order:
    1. Zeros the relevant accumulator buffers for class `C` (the
       three force buffers always; additionally the energy and
       virial buffers when `level == AggregateLevel::ForcesAndScalars`)
       via async memsets on the default stream.
    2. For each slot in `self.slots` whose class is `C`, in canonical
       slot order, calls `slot.compute(buffers, sim_box, view, &cx,
       timings, level)`, where `view` is a `SlotOutputView` whose
       five fields are length-`N` slices onto class `C`'s
       accumulator buffers. The slot adds into the force-component
       fields on every call and into the energy and virial fields
       only when `level == AggregateLevel::ForcesAndScalars`.
  - Launches `combine_class_totals` once (with
    `KernelStage::CombineClassTotals`). The kernel writes
    `particle_buffers.{forces_*, potential_energies, virials}[i] =
    fast_total_{forces_*, potential_energies, virials}[i] +
    slow_total_{forces_*, potential_energies, virials}[i]` for each
    `i`. The kernel runs regardless of `level`. When the slot list is
    empty across every class, the per-class memsets and the
    class-combine kernel still run, producing zeros in every entry of
    `buffers.forces_*`, `buffers.potential_energies`, and
    `buffers.virials`. When `level == AggregateLevel::ForcesOnly`,
    `buffers.potential_energies` and `buffers.virials` reflect the
    sum of each class's accumulator energy / virial as left by the
    most recent `ForcesAndScalars` evaluation of that class — they
    are *not* refreshed by a `ForcesOnly` call. The caller is
    responsible for issuing a `ForcesAndScalars` call before reading
    those fields (see *Force Evaluation Pipeline* above).
  - Returns `Ok(())` on success.
  - Empty-state contract per *Empty State* above.

- `ForceField::step_class(&mut self, class: ForceClass, buffers: &mut ParticleBuffers, sim_box: &SimulationBox, timings: &mut Timings, level: AggregateLevel) -> Result<(), ForceFieldError>` <!-- rq-be1eb548 -->
  - Re-evaluates only slots whose `frequency_class() == class`.
  - When the framework contains no slots of `class`, returns `Ok(())`
    immediately, launching no kernels and leaving
    `ParticleBuffers.forces_*` untouched (no-op semantics; see
    *Empty State*).
  - Otherwise, performs the same neighbor-list-update sequence as
    `step`, then zeros and re-adds only the `class` accumulator (the
    same per-class memset + per-slot add sequence from `step`,
    restricted to slots in `class`). The other class's accumulator is
    not touched. Finally launches `combine_class_totals`, which sums
    both classes' accumulators into `ParticleBuffers.forces_*` so
    that the result reflects the just-evaluated class plus the
    other class's last-evaluated state.
  - Returns `Ok(())` when `particle_count == 0`, launching no kernels.

### Class-Combine Kernel <!-- rq-c0f98145 -->

`kernels/forces.cu` declares one `extern "C"` kernel:

```c
extern "C" __global__ void combine_class_totals(
    const float *fast_total_forces_x,             // length n
    const float *fast_total_forces_y,             // length n
    const float *fast_total_forces_z,             // length n
    const float *fast_total_potential_energies,   // length n
    const float *fast_total_virials,              // length n
    const float *slow_total_forces_x,             // length n
    const float *slow_total_forces_y,             // length n
    const float *slow_total_forces_z,             // length n
    const float *slow_total_potential_energies,   // length n
    const float *slow_total_virials,              // length n
    float *forces_x,                              // length n
    float *forces_y,                              // length n
    float *forces_z,                              // length n
    float *potential_energies,                    // length n
    float *virials,                               // length n
    unsigned int n);
```

Each thread maps to one particle index
`i = blockIdx.x * blockDim.x + threadIdx.x` (block size 256, grid
`ceil(n / 256)`, no shared memory, default stream of
`particle_buffers.device`). The thread reads ten floats, performs five
additions, and writes five floats:

```text
forces_x[i]           = fast_total_forces_x[i]           + slow_total_forces_x[i]
forces_y[i]           = fast_total_forces_y[i]           + slow_total_forces_y[i]
forces_z[i]           = fast_total_forces_z[i]           + slow_total_forces_z[i]
potential_energies[i] = fast_total_potential_energies[i] + slow_total_potential_energies[i]
virials[i]            = fast_total_virials[i]            + slow_total_virials[i]
```

Per-class accumulator buffers are zero-initialised at construction and
are zeroed by the framework before the first slot in a class adds to
them (see *Force Evaluation Pipeline* step 3), so the kernel sees
valid contributions from every class regardless of which classes were
re-evaluated this step. Two runs with byte-identical accumulator
inputs produce byte-identical outputs.

When `n == 0` the kernel does not launch. The kernel does not branch
on slot count and its signature does not change as slots are added or
removed.

## Determinism Guarantees <!-- rq-76cb9922 -->

- The per-class memsets, every slot's `compute`, and the
  class-combine kernel run on the default stream of the same
  `Arc<CudaDevice>` carried by `ParticleBuffers`. CUDA's implicit
  per-stream ordering guarantees that any buffer written by an
  earlier launch is visible to later launches without explicit
  synchronisation; the class-combine kernel sees the accumulators in
  their post-add state. A slot that introduces a secondary CUDA
  stream must guarantee that, by the time its `compute` returns, every
  device buffer it has written is visible to subsequent default-stream
  launches, and that every device buffer it reads has been written by
  a preceding default-stream launch the slot waited on. No in-tree
  slot uses a secondary stream.
- The slot order produced by `ForceField::new` is deterministic and
  identical across runs with the same config.
- Within each class, slots in canonical slot order each add their
  per-particle contribution into the class accumulator via a
  read-modify-write performed by a single warp per particle. The
  sequence of adds into each `(class, axis, particle)` accumulator
  cell is fixed across runs.
- The class-combine kernel adds Fast and then Slow into the
  per-particle output in a fixed order. The order is fixed across
  runs, so per-atom force, potential-energy, and virial values are
  byte-reproducible.
- Two runs that issue the same sequence of `step` / `step_class` calls
  with the same arguments produce byte-identical
  `ParticleBuffers.forces_*`, `potential_energies`, and `virials` at
  every step. The class system does not introduce non-determinism;
  RESPA's staleness of Slow contributions between Slow-class
  evaluations is deterministic.
- Two `ForceField` configurations that differ only in *which
  built-in builders are registered* — for example, one with
  `LennardJonesBuilder` and `SpmeRealBuilder` (which the framework
  fuses into a single composed pair-force kernel via the
  JIT-composition mechanism in `jit-composed-pair-force.md`) and
  another containing the LJ builder alone followed by a custom
  builder that produces SPME-real as a separate per-slot kernel
  call — produce per-particle results that agree only within f32
  round-off, not bit-for-bit. The composed configuration visits
  each pair once and accumulates LJ and SPME-real contributions
  into the same register pair before the warp-tree reduction,
  while the per-slot-kernel configuration visits each pair twice
  and combines the two slots' per-particle totals through the
  class accumulator. Both configurations individually preserve
  run-to-run byte reproducibility on the same
  GPU; only cross-configuration equality is sacrificed.

## Out of Scope <!-- rq-e448909a -->

- A user-supplied DSL for custom potentials. Implementing `Potential`
  is a Rust source-code change; potentials are not loaded from
  configuration or shared libraries.
- Concrete RESPA-style integrators. The framework exposes the
  per-class evaluation surface that a RESPA integrator would consume,
  but no in-tree integrator splits its plan by `ForceClass` today.
- A per-class read API on `ForceField` (e.g. `class_force_view(class)`)
  that lets an integrator kick by class-only force. RESPA integrators
  that need this land alongside their own dedicated read API; v1 of
  the class system only decomposes evaluation and aggregates back into
  the single `ParticleBuffers.forces_*` total.
- A third force class beyond `Fast` and `Slow` (e.g. for RESPA-3).
  Adding a variant is a deliberate API change rather than an
  open-extension point.
- Per-slot streams or async overlap of contribution and reduction
  kernels.
- Mid-run reconfiguration of slot membership. The slot list is fixed
  at `ForceField::new` and never modified.
- Slot ordering being user-configurable. The order is fixed in
  `ForceField::new`.
- Wiring the per-particle `potential_energies` and `virials` aggregates
  into log output, trajectory output, or pressure-coupling barostats.
  The framework produces the per-particle aggregates each step; the
  consumers of those aggregates (log writer, pressure logger, NPT
  barostat) are documented in their own files.
- Full virial-tensor accumulation. The framework's per-pair virial is
  the scalar trace `r_ij · F_ij`; per-component virial accumulation
  (xx, yy, zz, xy, xz, yz) is not in scope.

---

## Gherkin Scenarios <!-- rq-37ccfc1f -->

```gherkin
Feature: Pluggable potential slot framework

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called

  # --- Construction ---

  @rq-56c8a238
  Scenario: Construct a ForceField with LennardJones only
    Given a particle_count of 4
    And one [[pair_interactions]] entry for ("Ar","Ar")
    And no bond list and no bond types
    When ForceField::new(device, 4, sim_box, &pair_interactions, &[], &empty_bonds, &empty_excl, &nl_config) is called
    Then it returns Ok(force_field)
    And force_field.slots has length 1
    And force_field.slots[0].label() == "lennard_jones"

  @rq-3de16ce0
  Scenario: Construct a ForceField with LennardJones and MorseBonded
    Given a particle_count of 4
    And one [[pair_interactions]] entry for ("Ar","Ar")
    And one [[bond_types]] entry "CC" with potential="morse" and valid Morse parameters
    And a BondList with at least one bond of type "CC"
    And an ExclusionList consistent with the bonds
    When ForceField::new(...) is called
    Then it returns Ok(force_field)
    And force_field.slots has length 2
    And force_field.slots[0].label() == "lennard_jones"
    And force_field.slots[1].label() == "morse_bonded"

  @rq-0f34d11b
  Scenario: Construct a ForceField with bond_types declared but no bonds
    Given a particle_count of 4
    And one [[pair_interactions]] entry
    And one [[bond_types]] entry
    And an empty BondList
    When ForceField::new(...) is called
    Then it returns Ok(force_field)
    And force_field.slots has length 1
    And force_field.slots[0].label() == "lennard_jones"

  @rq-60f445b2
  Scenario: Construct a ForceField with zero slots
    Given a particle_count of 4
    And no [[pair_interactions]] entries
    And no bonds
    When ForceField::new(device, 4, sim_box, &[], &[], &empty_bonds, &empty_excl, &nl_config) is called
    Then it returns Ok(force_field)
    And force_field.slots is empty
    And every fast_total_* and slow_total_* accumulator has length 4

  @rq-59e60e89
  Scenario: Per-class accumulator buffers are sized particle_count regardless of slot count
    Given a particle_count of 8
    And a config producing 4 Fast slots and 1 Slow slot
    When ForceField::new(...) is called
    Then force_field.fast_total_forces_x has length 8
    And force_field.fast_total_forces_y has length 8
    And force_field.fast_total_forces_z has length 8
    And force_field.fast_total_potential_energies has length 8
    And force_field.fast_total_virials has length 8
    And force_field.slow_total_forces_x has length 8
    And force_field.slow_total_potential_energies has length 8

  @rq-c525ee79
  Scenario: Construct an empty (N=0) ForceField with potentials configured
    Given a particle_count of 0
    And a config producing 2 slots
    When ForceField::new(...) is called
    Then it returns Ok(force_field)
    And every per-class accumulator buffer has length 0

  @rq-b850b5be
  Scenario: Per-class accumulator buffers are zero-initialised at construction
    Given a particle_count of 4 and a config producing 1 Fast and 1 Slow slot
    When ForceField::new(...) is called
    And the accumulator buffers are downloaded
    Then every entry of fast_total_forces_x, fast_total_forces_y, fast_total_forces_z,
      fast_total_potential_energies, fast_total_virials,
      slow_total_forces_x, slow_total_forces_y, slow_total_forces_z,
      slow_total_potential_energies, slow_total_virials is exactly 0.0

  @rq-c170c0b7
  Scenario: Reject construction when two slots share the same label
    Given a constructed ForceField scenario in which two slots would return
      the same label() value
    When ForceField::new(...) is called
    Then it returns Err(ForceFieldError::DuplicateLabel(_))

  # --- Force evaluation pipeline ---

  @rq-32e981cc
  Scenario: step() on a ForceField with only LennardJones writes LJ forces to forces_*
    Given a constructed ForceField with the LennardJones slot only
    And a ParticleBuffers with particle_count() == 2 placed so the LJ force is non-zero
    When force_field.step(&mut buffers, &sim_box, &mut timings) is called
    And particle_buffers is downloaded
    Then forces_x is non-zero in the expected pattern
    And forces_y, forces_z are consistent with the closed-form LJ result
    And timings reports counts for KernelStage::LjPairForce, ReducePairForces,
      and CombineClassTotals

  @rq-df3a50f6
  Scenario: step() on a ForceField with both slots sums LJ and Morse
    Given a constructed ForceField with both slots
    And a ParticleBuffers and bond configuration where the LJ force and the Morse
      force on atom 0 are known a priori
    When force_field.step(&mut buffers, &sim_box, &mut timings) is called
    And particle_buffers is downloaded
    Then forces_x[0] equals lj_force_x_on_0 + morse_force_x_on_0 within f32 round-off

  @rq-fc7b1565
  Scenario: step() on a ForceField with zero slots writes zeros to forces_*
    Given a constructed ForceField with force_field.slots.is_empty()
    And a ParticleBuffers with particle_count() == 4 and arbitrary prior contents in forces_*
    When force_field.step(&mut buffers, &sim_box, &mut timings) is called
    And particle_buffers is downloaded
    Then forces_x, forces_y, forces_z are all zero
    And timings reports count==1 for KernelStage::CombineClassTotals
    And timings reports count==0 for every slot-specific KernelStage

  @rq-de47c1ac
  Scenario: step() with N=0 launches no kernels
    Given a ForceField constructed with particle_count == 0
    When force_field.step(...) is called
    Then it returns Ok(())
    And timings.finalize() reports zero samples for every KernelStage

  @rq-86741731
  Scenario: Each slot's contribution is reflected in its class accumulator after step()
    Given a constructed ForceField with two Fast slots `LJ` and `Morse` and particle_count == 3
    And a configuration where `LJ` and `Morse` each contribute known per-particle force_x values
    When force_field.step(...) is called with ForcesAndScalars
    And fast_total_forces_x is downloaded
    Then fast_total_forces_x[i] equals lj_force_x[i] + morse_force_x[i] for every i in 0..3
      within f32 round-off

  @rq-d93afc32
  Scenario: First slot in class adds into a zeroed accumulator, subsequent slots add on top
    Given a constructed ForceField with three Fast slots in canonical order LJ, Coulomb, Morse
    And a particle_count of 1 with a known per-particle force_x contribution per slot
    When force_field.step(...) is called with ForcesAndScalars
    And fast_total_forces_x is downloaded
    Then fast_total_forces_x[0] equals lj_force_x + coulomb_force_x + morse_force_x
      within f32 round-off, in that addition order

  @rq-7b99234a
  Scenario: Framework zeros the class accumulator before the first slot adds
    Given a constructed ForceField with one Fast slot and particle_count == 4
    And the first step has run with ForcesAndScalars, leaving non-zero values in fast_total_forces_x
    When force_field.step(...) is called a second time with ForcesAndScalars
    And fast_total_forces_x is downloaded
    Then fast_total_forces_x[i] equals the slot's per-particle force_x contribution
      from the second call only (not the sum of both calls), for every i in 0..4

  # --- Trait dispatch ---

  @rq-a9642241
  Scenario: Adding a new Potential implementation requires no edits to ForceField or combine_class_totals
    Given a third Potential implementation `Buckingham` with label() == "buckingham" and frequency_class() == Fast
    And a `BuckinghamBuilder` registered after `MorseBondedBuilder` in `PotentialRegistry::with_builtins()`
    When ForceField::new(...) is called with a config that activates all three slots
    Then force_field.slots has length 3
    And the combine_class_totals kernel binary is unchanged
    And the SlotOutputView passed to Buckingham's compute points at the fast_total_* accumulators

  # --- PotentialRegistry-driven construction ---

  @rq-053a026c
  Scenario: PotentialRegistry::with_builtins exposes the six built-in builders in evaluation order
    Given a PotentialRegistry constructed via PotentialRegistry::with_builtins()
    Then registry.builders has length 6
    And the builders' debug type names (or kind tags) are, in order,
      LennardJonesBuilder, CoulombBuilder, SpmeRealBuilder,
      SpmeReciprocalBuilder, MorseBondedBuilder, HarmonicAngleBuilder

  @rq-78ad9477
  Scenario: PotentialRegistry::new starts empty
    Given a registry constructed via PotentialRegistry::new()
    Then registry.builders.is_empty() returns true

  @rq-51af5f97
  Scenario: register(...) appends a builder at the end
    Given a PotentialRegistry::with_builtins()
    And a custom PotentialBuilder whose build(cx) always returns Ok(None)
    When registry.register(Box::new(custom_builder)) is called
    Then registry.builders has length 7
    And registry.builders[6] is the custom builder

  @rq-b1a132b5
  Scenario: ForceField::new iterates the registry in registration order
    Given a PotentialRegistry::with_builtins()
    And a context that satisfies both the LennardJones activation condition
      and the MorseBonded activation condition
    When ForceField::new(&registry, ...) is called
    Then force_field.slots[0].label() == "lennard_jones"
    And force_field.slots[1].label() == "morse_bonded"

  @rq-ccf4dc3f
  Scenario: Builder returning Ok(None) is skipped without erroring
    Given a PotentialRegistry containing exactly one custom builder whose
      build(cx) returns Ok(None)
    When ForceField::new(&registry, ...) is called
    Then it returns Ok(force_field)
    And force_field.slots is empty

  @rq-6ed7e318
  Scenario: Builder Err short-circuits ForceField::new
    Given a PotentialRegistry containing a custom builder whose build(cx)
      returns Err(ForceFieldError::Gpu(_))
    And a second custom builder whose build(cx), if reached, would record a call
    When ForceField::new(&registry, ...) is called
    Then it returns Err(ForceFieldError::Gpu(_))
    And the second builder's build is not invoked

  @rq-24c36f8d
  Scenario: Two builders producing slots with the same label fail construction
    Given a PotentialRegistry containing two custom builders that both build
      a Potential whose label() == "duplicate"
    And a context that satisfies both builders' activation conditions
    When ForceField::new(&registry, ...) is called
    Then it returns Err(ForceFieldError::DuplicateLabel("duplicate"))

  @rq-028f5f8e
  Scenario: Empty registry produces a zero-slot ForceField
    Given a registry constructed via PotentialRegistry::new()
    When ForceField::new(&registry, ...) is called
    Then it returns Ok(force_field)
    And force_field.slots is empty
    And force_field.neighbor_list is None

  @rq-b75ce71a
  Scenario: PotentialBuildContext exposes every parsed-config input by reference
    Given a custom builder whose build(cx) records pointer identity for
      cx.particle_types, cx.pair_interactions, cx.bond_types, cx.angle_types,
      cx.coulomb_config, cx.spme_config, cx.charges, cx.bond_list,
      cx.angle_list, cx.exclusion_list, cx.neighbor_list_config
    When ForceField::new(&registry, gpu, n, sim_box, pts, pairs, bts, ats,
      coul, spme, charges, bonds, angles, excl, nl_config) is called
    Then the recorded pointers match the addresses of the function arguments
      passed in by the caller

  # --- Composite slot displacement ---

  @rq-1b6985ab
  Scenario: PotentialBuilder::displaces default returns an empty list
    Given a custom PotentialBuilder implementation that does not override displaces()
    Then builder.displaces() returns &[]

  @rq-9e6a7b7e
  Scenario: Active LJ + SPME-real configuration fuses both fragments into one composed kernel
    Given a ForceField config that activates both LennardJones and SpmeReal
      (one [[pair_interactions]] entry plus [spme] configured with non-zero charges)
    And PotentialRegistry::with_builtins()
    When ForceField::new(...) is called
    Then force_field.slots contains slots with label() == "lennard_jones" and label() == "spme_real"
    And force_field exposes one JIT-composed pair-force kernel
    And per-step force evaluation launches the composed kernel exactly once and does not launch any per-slot pair-force kernel

  @rq-fea21f61
  Scenario: A custom composite-fragment builder with active constituents displaces both fragments
    Given a custom builder C with displaces() == &["lennard_jones", "spme_real"] that produces a fragment fusing both contributions
    And a config that activates LennardJones, SpmeReal, and C
    When ForceField::new(...) is called
    Then the JIT-composed source includes C's fragment exactly once
    And the JIT-composed source includes neither LennardJonesBuilder's fragment nor SpmeRealBuilder's fragment

  @rq-83298c86
  Scenario: Composite displacement claim against unbuilt label is a no-op
    Given a registry containing a custom composite whose displaces() == &["never_built"]
    And whose build(cx) returns Ok(Some(slot)) labelled "custom_composite"
    And a config in which no built-in builder produces a "never_built" slot
    When ForceField::new(...) is called
    Then it returns Ok(force_field)
    And force_field.slots contains exactly one slot with label() == "custom_composite"

  @rq-f10530c8
  Scenario: Two built composites claiming the same constituent error at construction
    Given a custom builder A that builds successfully and displaces() == &["lennard_jones"]
    And a custom builder B that builds successfully and displaces() == &["lennard_jones"]
    And a config that also activates LennardJonesBuilder
    When ForceField::new(...) is called
    Then it returns Err(ForceFieldError::DisplaceConflict { label: "lennard_jones", by: <both labels> })

  @rq-aa33e39f
  Scenario: Two builders both claiming a label nobody built do not error
    Given two custom builders A and B that each build successfully and each have
      displaces() == &["nobody_built_this"]
    And no other builder produces a slot with that label
    When ForceField::new(...) is called
    Then it returns Ok(force_field)
    And force_field.slots contains both A's and B's slots

  @rq-e55779e2
  Scenario: Slot list is the surviving registration order
    Given PotentialRegistry::with_builtins() and a config activating LJ + SPME + Morse
    When ForceField::new(...) is called
    Then force_field.slots in order is:
      [lennard_jones, spme_real, spme_reciprocal, morse_bonded]

  @rq-c19ea1ca
  Scenario: Two runs of the LJ + SPME composed-kernel configuration agree byte-for-byte
    Given two independently-constructed ForceFields, each built from a config that
      activates LJ and SPME so the JIT-composed pair-force kernel covers both
    And two ParticleBuffers built from byte-identical ParticleStates of N=64
    When force_field.step(...) is called on each
    Then run A's forces_x, forces_y, forces_z agree byte-for-byte with run B's

  # --- Force classes and per-class evaluation ---

  @rq-db2253db
  Scenario: Potential::frequency_class default returns Fast
    Given a custom Potential implementation that does not override frequency_class
    Then potential.frequency_class() returns ForceClass::Fast

  @rq-2dbda7ec
  Scenario: Built-in potentials report their canonical class
    Given a ForceField with every built-in slot present
    Then slot "lennard_jones"   reports frequency_class() == Fast
    And  slot "coulomb"         reports frequency_class() == Fast
    And  slot "spme_real"       reports frequency_class() == Fast
    And  slot "spme_reciprocal" reports frequency_class() == Slow
    And  slot "morse_bonded"    reports frequency_class() == Fast
    And  slot "harmonic_angle"  reports frequency_class() == Fast

  @rq-57fd217e
  Scenario: step() evaluates every class and produces the total in ParticleBuffers
    Given a ForceField with one Fast slot (LennardJones) and one Slow stub
      whose compute writes a known per-particle pattern S into its row
    When force_field.step(&mut buffers, &sim_box, &mut timings) is called
    Then ParticleBuffers.forces_x[i] equals lj_force_x[i] + S[i] for every i
    And  the LJ slot's compute kernels each fire exactly once
    And  the Slow stub's compute each fire exactly once

  @rq-1a996f5d
  Scenario: step_class(Fast) refreshes only Fast slots' contributions
    Given a ForceField with one Fast slot (LennardJones) and one Slow stub
      whose compute writes a known per-particle pattern S into its row
    And  force_field.step(...) has been called once so every class buffer is populated
    When ParticleBuffers.positions are advanced (e.g. by a drift)
    And  force_field.step_class(ForceClass::Fast, ...) is called
    Then the LJ slot's compute each fire exactly once (new LJ values)
    And  the Slow stub's compute do NOT fire (stale S contributions)
    And  ParticleBuffers.forces_x[i] equals new_lj_force_x[i] + S[i] for every i

  @rq-33cfb9fc
  Scenario: step_class(Slow) refreshes only Slow slots' contributions
    Given a ForceField with one Fast slot whose compute writes a known
      per-particle pattern F into its row
    And  one Slow stub whose compute writes a known per-particle pattern S
      that changes between successive calls (e.g. via an internal counter)
    And  force_field.step(...) has been called once so every class buffer is populated
    When force_field.step_class(ForceClass::Slow, ...) is called
    Then the Slow stub's compute each fire exactly once (new S values)
    And  the Fast slot's compute do NOT fire (stale F contributions)
    And  ParticleBuffers.forces_x[i] equals F[i] + new_S[i] for every i

  @rq-cc66d208
  Scenario: step_class(Slow) on a ForceField with no Slow slots is a no-op
    Given a ForceField with only Fast slots (e.g. one LennardJones slot)
    And  ParticleBuffers.forces_* snapshot S_before captured after a prior step()
    When force_field.step_class(ForceClass::Slow, ...) is called
    Then it returns Ok(())
    And  ParticleBuffers.forces_* equal S_before byte-for-byte
    And  timings reports zero samples for every KernelStage that any slot would launch
    And  timings reports zero samples for KernelStage::AccumulateForces

  @rq-b80f2ddb
  Scenario: step_class(Fast) on a ForceField with no Fast slots is a no-op
    Given a ForceField with only Slow slots (e.g. one stub Slow potential)
    And  ParticleBuffers.forces_* snapshot S_before captured after a prior step()
    When force_field.step_class(ForceClass::Fast, ...) is called
    Then it returns Ok(())
    And  ParticleBuffers.forces_* equal S_before byte-for-byte

  @rq-8eb7a546
  Scenario: step_class with N=0 launches no kernels
    Given a ForceField constructed with particle_count == 0 and any registry
    When force_field.step_class(ForceClass::Fast, ...) is called
    Then it returns Ok(())
    And  timings.finalize() reports zero samples for every KernelStage

  @rq-79068d4d
  Scenario: Per-class accumulator buffers have length particle_count regardless of slot count
    Given a ForceField with particle_count = 8 and a registry whose
      with_builtins() registers all six builders, with config that activates
      LennardJones, SpmeReal (Fast) and SpmeReciprocal (Slow)
    Then force_field.fast_total_forces_x.len() == 8
    And  force_field.slow_total_forces_x.len() == 8
    And  force_field.fast_total_potential_energies.len() == 8
    And  force_field.slow_total_virials.len()  == 8

  @rq-52d4b245
  Scenario: Per-class accumulator buffers are zero-initialised
    Given a freshly-constructed ForceField with at least one Fast and one Slow slot
    When ParticleBuffers.forces_* is downloaded immediately after construction
      (no step_* call yet)
    Then every entry is 0.0
    And  force_field.slow_total_forces_x downloads to all zeros
    And  force_field.fast_total_forces_x downloads to all zeros

  @rq-40f9d35a
  Scenario: Two RESPA-style call sequences with the same plan produce identical state
    Given two ForceFields constructed from identical registries and inputs,
      each holding one Fast slot and one Slow slot
    And  two ParticleBuffers built from byte-identical ParticleStates
    When each runner issues the call sequence
      [step(), step_class(Fast), step_class(Fast), step_class(Slow)]
      with identical inputs to each call
    Then run A's ParticleBuffers and run B's ParticleBuffers agree
      byte-for-byte after every call

  @rq-5855473b
  Scenario: SubStep::ForceEval { class: None } dispatches to step()
    Given a runner walking a StepPlan containing SubStep::ForceEval { class: None }
    When the runner reaches the ForceEval sub-step
    Then force_field.step(...) is invoked (every slot's kernels fire)

  @rq-256287cb
  Scenario: SubStep::ForceEval { class: Some(Fast) } dispatches to step_class(Fast)
    Given a runner walking a StepPlan containing
      SubStep::ForceEval { class: Some(ForceClass::Fast) }
    When the runner reaches the ForceEval sub-step
    Then force_field.step_class(ForceClass::Fast, ...) is invoked
    And  no Slow slot's compute kernels fire during this sub-step

  # --- Reproducibility ---

  @rq-c8e5b14e
  Scenario: Two independent runs with identical inputs are byte-identical
    Given two independently-constructed ForceFields with identical parameters
    And two ParticleBuffers built from byte-identical ParticleStates of N=64
    When force_field.step(...) is called on each
    And the two ParticleBuffers are downloaded
    Then run A's forces_x, forces_y, forces_z agree byte-for-byte with run B's

  # --- Class-combine kernel correctness ---

  @rq-a5aa743e
  Scenario: combine_class_totals sums fast + slow into ParticleBuffers
    Given a ForceField where fast_total_forces_x = [1.0, 2.0] and slow_total_forces_x = [10.0, 20.0]
    When combine_class_totals runs with n = 2
    Then particle_buffers.forces_x equals [11.0, 22.0]

  @rq-3e9217e2
  Scenario: combine_class_totals with zero-initialised accumulators writes zeros
    Given a freshly-constructed ForceField with zero slots and particle_count == 4
    When force_field.step(...) is called
    Then particle_buffers.forces_x, forces_y, forces_z are all zero
    And  particle_buffers.potential_energies, virials are all zero

  @rq-82acb52f
  Scenario: combine_class_totals is a single-threaded write per output element
    Given a ForceField with known accumulator contents
    When force_field.step(...) is called twice on identical inputs
    Then the resulting particle_buffers.forces_* agree byte-for-byte across the two calls

  # --- Shared neighbor list ---

  @rq-b33cf896
  Scenario: ForceField with a short-range potential owns a shared neighbor list
    Given a ForceField with one LennardJones slot in CellList mode
    When ForceField::new completes
    Then ForceField::neighbor_list is Some(_)
    And the shared NeighborListState's tile-pair buffers are allocated
      with capacity at least tile_pair_capacity from the config

  @rq-433c972f
  Scenario: ForceField with only a bonded potential owns no neighbor list
    Given a ForceField with one MorseBonded slot (and no pair_interactions)
    When ForceField::new completes
    Then ForceField::neighbor_list is None

  @rq-81e84c73
  Scenario: ForceFieldContext exposes the shared neighbor list to compute
    Given a ForceField with a LennardJones slot in any mode
    And a stub Potential whose compute() records the value of cx.neighbor_list
    When ForceField::step is called
    Then the stub records `Some(_)` (the same NeighborListState reference the LJ slot uses)

  @rq-e39d0ed8
  Scenario: max_cutoff aggregation determines the neighbor-list radius
    Given two short-range Potential implementations reporting max_cutoff() = Some(2.0) and Some(5.0)
    And NeighborListConfig::CellList { r_skin, tile_pair_capacity }
    When ForceField::new constructs the shared neighbor list
    Then the neighbor list's r_search equals 5.0 + r_skin

  @rq-47540d14
  Scenario: A bonded-only ForceField step launches no neighbor-list kernels
    Given a ForceField whose only slot returns max_cutoff() = None
    When ForceField::step is called
    Then timings reports zero samples for KernelStage::NeighborDisplacementSquared
    And timings reports zero samples for KernelStage::NeighborListBuild

  # --- Per-particle energy and virial outputs ---

  @rq-531faea9
  Scenario: ForceField with N=4 LJ-only step populates potential_energies and virials
    Given a constructed ForceField with one LennardJones slot and N=4
    And a ParticleState placed so the LJ contributions are non-zero
    When ForceField::step is called
    And ParticleBuffers is downloaded
    Then potential_energies is finite and non-zero in the expected pattern
    And virials is finite and non-zero in the expected pattern

  @rq-a85e8216
  Scenario: ForceField owns ten per-class accumulator buffers each of length particle_count
    Given a ForceField with mixed Fast and Slow slots and particle_count = 8
    Then force_field.fast_total_forces_x, fast_total_forces_y, fast_total_forces_z,
      fast_total_potential_energies, fast_total_virials each have length 8
    And force_field.slow_total_forces_x, slow_total_forces_y, slow_total_forces_z,
      slow_total_potential_energies, slow_total_virials each have length 8

  @rq-3d38868e
  Scenario: combine_class_totals sums fast + slow energies and virials
    Given a ForceField where fast_total_potential_energies = [1.0, 2.0] and slow_total_potential_energies = [10.0, 20.0]
    And where fast_total_virials = [0.5, 1.0] and slow_total_virials = [5.0, 10.0]
    When combine_class_totals runs with n = 2
    Then particle_buffers.potential_energies equals [11.0, 22.0]
    And particle_buffers.virials equals [5.5, 11.0]

  @rq-27c19b67
  Scenario: ForcesOnly call preserves the energy and virial accumulator entries from the previous ForcesAndScalars call
    Given a ForceField with one Fast slot and particle_count == 2
    And a first call to force_field.step(ForcesAndScalars) leaving
      fast_total_potential_energies = [u0, u1] and fast_total_virials = [w0, w1]
    When force_field.step(ForcesOnly) is called next
    And fast_total_potential_energies and fast_total_virials are downloaded
    Then they still equal [u0, u1] and [w0, w1] respectively (byte-identical)

  @rq-95cc1d55
  Scenario: step_class(Fast, ForcesAndScalars) does not touch slow_total_* accumulators
    Given a ForceField with one Fast slot and one Slow slot and particle_count == 4
    And a first call to force_field.step(ForcesAndScalars) populating both accumulators
    When force_field.step_class(ForceClass::Fast, ForcesAndScalars) is called next
    And slow_total_forces_x, slow_total_potential_energies, slow_total_virials are downloaded
    Then they hold the same byte values as after the first call

  @rq-c0f2daca
  Scenario: Zero-slot step writes zeros to potential_energies and virials
    Given a constructed ForceField with force_field.slots.is_empty()
    And a ParticleBuffers with particle_count() == 4 and arbitrary prior contents in
      potential_energies and virials
    When ForceField::step is called
    And ParticleBuffers is downloaded
    Then potential_energies and virials are each [0.0, 0.0, 0.0, 0.0]

  @rq-db3b3d5e
  Scenario: System total potential energy equals sum of particle shares
    Given a constructed ForceField with one LennardJones slot and N atoms
    When ForceField::step is called
    And the per-particle potential_energies are downloaded
    Then their sum equals the expected total LJ energy of the configuration within f32 round-off

  @rq-7fe57a77
  Scenario: System total scalar virial equals sum of particle shares
    Given a constructed ForceField with one LennardJones slot and N atoms
    When ForceField::step is called with AggregateLevel::ForcesAndScalars
    And the per-particle virials are downloaded
    Then their sum equals Σ_{i<j within cutoff} r_ij · F_ij within f32 round-off

  # --- AggregateLevel ---

  @rq-5985846f
  Scenario: step(ForcesOnly) updates forces and leaves potential_energies / virials stale
    Given a constructed ForceField with one LennardJones slot and N atoms
    And ForceField::step has just been called with AggregateLevel::ForcesAndScalars,
      producing potential_energies = E_0 and virials = W_0 on the device
    When particle positions are changed and ForceField::step is called with
      AggregateLevel::ForcesOnly
    Then forces_x, forces_y, forces_z reflect the LJ contribution at the new positions
    And potential_energies on the device is byte-identical to E_0
    And virials on the device is byte-identical to W_0
    And fast_total_potential_energies and fast_total_virials are byte-identical to
      what the prior ForcesAndScalars call left there

  @rq-beccac31
  Scenario: step(ForcesAndScalars) refreshes potential_energies and virials
    Given a constructed ForceField with one LennardJones slot and N atoms
    And ForceField::step has just been called with AggregateLevel::ForcesAndScalars
      at positions P_0, producing potential_energies = E_0 on the device
    When particle positions are changed to P_1
    And ForceField::step is called with AggregateLevel::ForcesAndScalars
    Then potential_energies on the device is the LJ potential energy share evaluated
      at P_1, differing from E_0 by the position change
    And forces_x, forces_y, forces_z reflect the LJ contribution at P_1

  @rq-55d441ee
  Scenario: Two runs with identical (call, level) sequences are byte-identical
    Given two independent ForceField instances A and B with identical configs and
      identical initial ParticleBuffers
    When each runs the same sequence of K force evaluations, each step at the same
      AggregateLevel value (a mix of ForcesOnly and ForcesAndScalars)
    Then forces_x, forces_y, forces_z, potential_energies, and virials on the
      device agree byte-for-byte between A and B

  @rq-fcc5cea5
  Scenario: A bonded slot honours ForcesOnly via its level parameter
    Given a constructed ForceField with one MorseBonded slot and particle_count == 2
    And a first call to force_field.step(ForcesAndScalars) leaving the slot's
      class accumulator fast_total_potential_energies = [u0, u1] and
      fast_total_virials = [w0, w1]
    When force_field.step(ForcesOnly) is called next
    And fast_total_potential_energies and fast_total_virials are downloaded
    Then they still equal [u0, u1] and [w0, w1] respectively
    And forces_x, forces_y, forces_z on the device reflect the bonded contribution

  @rq-d2bf331b
  Scenario: A pair-force slot honours ForcesOnly
    Given a constructed ForceField with one LennardJones slot
    And a first call to force_field.step(ForcesAndScalars) leaving the slot's
      class accumulator fast_total_potential_energies = E_class and
      fast_total_virials = W_class
    When force_field.step(ForcesOnly) is called next
    Then fast_total_potential_energies and fast_total_virials are byte-identical
      to E_class and W_class (the pair-force slot's compute skipped them)
    And fast_total_forces_x, fast_total_forces_y, fast_total_forces_z reflect the
      LJ contribution at the new positions

  @rq-82822681
  Scenario: combine_class_totals always runs regardless of level
    Given a constructed ForceField with at least one slot
    When ForceField::step is called with AggregateLevel::ForcesOnly
    Then combine_class_totals is launched exactly once
    And the timings record a single CombineClassTotals stage tick for this call
```
