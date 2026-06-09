# Feature: Constraint Slot Framework <!-- rq-3d5f2e98 -->

The runner exposes a fourth orthogonal slot — `Constraint` — alongside the
`Integrator`, `Thermostat`, and `Barostat` slots described in
`framework.md`. The constraint slot projects per-particle positions and
velocities onto a holonomic constraint manifold (rigid bonds, rigid
groups, etc.) at fixed sub-step boundaries inside the integrator. The
slot is independently registered, independently selectable from the
config, and composes with the other three slots without affecting their
interfaces beyond the integrator's `step()` signature.

The default registry exposes one constraint algorithm:

| `kind` value     | Implementation                                                                            | File         |
| ---------------- | ----------------------------------------------------------------------------------------- | ------------ |
| `settle-water`   | non-standard hybrid: iterative SHAKE + closed-form RATTLE on three-atom rigid water (see `settle.md` *Status*; expected to be deprecated by separate analytical SETTLE and general M-SHAKE features) | `settle.md`  |

The slot is selectable in any simulation whose declared integrator
returns `IntegratorBuilder::supports_constraints(&params) == true`. In the default
registry, only `velocity-verlet { lossless = false }` supports
constraints. Composing constraints with any other integrator — or with
`velocity-verlet { lossless = true }` — is rejected at config load.

## Sources of Constraint Topology <!-- rq-31518680 -->

Constraints have two declaration sites and one runtime artefact:

1. **`[[constraint_types]]` in the TOML config** — declares each named
   constraint geometry and the algorithm that consumes it (see
   `io/config-schema.md`).
2. **`[constraints]` section in the `.topology` file** — declares one
   row per constraint group (one rigid molecule or one rigid cluster),
   each row naming the atom indices it covers and the constraint-type
   name from the config (see `forces/topology.md`).
3. **`ConstraintList`** — the host-side parsed artefact produced by
   `load_topology_file` from the two sources above. Carries the
   per-group SoA documented in *Constraint Data Layout* below.

A simulation has a `Constraint` slot iff `ConstraintList::is_empty()` is
`false`. The slot's lifecycle is identical to the other three slots:
constructed once at runner start, owned by the runner for the lifetime
of the run, dropped at end of run.

## Per-Step Interface <!-- rq-f08d7a33 -->

The constraint slot's hooks are fired by the runner during its walk of
the integrator's `StepPlan` (see `framework.md`). The integrator
describes its work as an ordered sequence of `SubStep`s; the runner
identifies the canonical hook positions by inspecting the variant of
each sub-step and inserts the corresponding hook call. Integrators
never reference the constraint slot.

The four hook positions are:

| Hook                              | When the runner fires it                                                                | What the slot does                                                                                                                            |
| --------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `apply_before_drift`              | Immediately before every `SubStep::Drift` or `SubStep::KickDrift` the integrator executes | Snapshots the pre-drift positions of every atom the slot owns into the slot's internal buffer.                                                |
| `apply_after_drift`               | Immediately after every `SubStep::Drift` or `SubStep::KickDrift` the integrator executes | Projects positions back onto the constraint manifold; updates the corresponding half-step velocities so they remain consistent with the projected displacement. |
| `apply_after_kick`                | Once per timestep, after the final sub-step in the plan, iff that sub-step is a `SubStep::KickHalf` or `SubStep::KickDrift` | Projects velocities onto the constraint manifold so the time-derivative of every constraint is zero at the new positions.                     |
| `apply_position_projection_only`  | After each trial position update inside an energy-minimization phase (see `rqm/minimization/steepest-descent.md`) | Projects positions back onto the constraint manifold without modifying velocities or virials. Does not consume `dt`. |

The first three hooks fire only during the integrator's plan walk in
MD phases. The fourth hook fires only during minimization phases and
is driven by the runner's minimization loop (see
`rqm/minimization/steepest-descent.md`).

For the registered `velocity-verlet` (lossy) integrator, whose plan is
`[KickDrift, ForceEval, KickHalf]` (see `velocity-verlet.md`), the
runner's interleaved sequence becomes:

```text
constraint.apply_before_drift(buffers, sim_box, dt, timings)
integrator.execute(SubStep::KickDrift { .. }, buffers, sim_box, timings)
constraint.apply_after_drift(buffers, sim_box, dt, timings)
force_field.step(buffers, sim_box, timings)
integrator.execute(SubStep::KickHalf { .. }, buffers, sim_box, timings)
constraint.apply_after_kick(buffers, sim_box, dt, timings)
```

When the constraint slot is `None`, the runner skips hook insertion
entirely; the plan walk reduces to the bare sub-step dispatch. When
`IntegratorBuilder::supports_constraints(&params)` returns `false`
the runner also skips hook insertion (and the config loader rejects this
combination with a non-empty `[constraints]` section before the run
starts). When the plan contains no `Drift` or `KickDrift` sub-steps,
only the final-velocity `apply_after_kick` hook can fire — and only
if the plan's last sub-step is a `KickHalf` or `KickDrift`.

The runner brackets each hook call with its own timings stage
(`CONSTRAINT_BEFORE_DRIFT`, `CONSTRAINT_AFTER_DRIFT`,
`CONSTRAINT_AFTER_KICK`) using `timings.kernel_start` /
`timings.kernel_stop`; the slot's internal kernels record their own
finer-grained stages within those brackets.

`apply_before_drift`, `apply_after_drift`, and `apply_after_kick` each
receive `&SimulationBox` (immutably) to read lattice parameters for
minimum-image distance evaluation; none mutates the box.

## Compatibility Rules <!-- rq-acfda5d4 -->

- An integrator whose builder's
  `IntegratorBuilder::supports_constraints(&params)` returns `false`
  is incompatible with a non-empty `[constraints]` section of the
  topology file. `Config::validate_against(&registries)` (see
  `io/config-schema.md`) returns
  `ConfigError::IncompatibleConstraint { integrator: <kind name> }`
  at config load.
- The `"velocity-verlet"` builder returns `false` from
  `supports_constraints(&{ lossless: true })`. Combining lossless
  velocity-Verlet with constraints is therefore rejected with the
  same error; lossless support is deferred to a future feature.
- The `"langevin-baoab"` and `"mtk-npt"` builders return `false`
  from `supports_constraints(&params)` regardless of the params they
  receive.
- When the chosen integrator's builder's
  `supports_constraints(&params)` returns `true` but the topology has
  no `[constraints]` section, the runner holds `None` for the
  constraint slot and the integrator's hooks are skipped.

## Constraint Data Layout <!-- rq-be5d5b33 -->

The host-side `ConstraintList` is the canonical parsed-and-validated
view of every constraint declared by the topology file. It is shaped to
accommodate algorithms that operate on arbitrarily-sized rigid clusters
(future M-SHAKE, LINCS) as well as the fixed-shape three-atom SETTLE
case in v1. The layout follows the project SoA convention.

```text
groups:              Vec<ConstraintGroup>
group_atoms:         Vec<u32>             # length = sum of group.atom_count
group_constraints:   Vec<GroupConstraint> # length = sum of group.constraint_count
particle_count:      usize
```

Each `ConstraintGroup` is one connected component of the constraint
graph — a set of atoms made rigid by a set of pairwise distance
constraints. Each group carries:

- `atom_offset: u32` — start index into `group_atoms`.
- `atom_count: u32` — number of atoms in the group.
- `constraint_offset: u32` — start index into `group_constraints`.
- `constraint_count: u32` — number of constraints in the group.
- `constraint_type_index: u32` — index into the config's
  `[[constraint_types]]` array.

The atoms of group `g` are
`group_atoms[g.atom_offset .. g.atom_offset + g.atom_count]`. The atom
order within a group is the order in which atoms appear in the
`[constraints]` row that declared the group. Algorithm-specific
positional conventions are encoded by that order — for `settle-water`,
the first atom is the heavy atom (oxygen) and the next two are the
hydrogens. The constraint-row order is preserved verbatim so that
algorithms can rely on it; the per-group sort within `groups` (see
*Group Ordering* below) is separate.

The constraints of group `g` are
`group_constraints[g.constraint_offset .. g.constraint_offset + g.constraint_count]`.
Each `GroupConstraint` is `{ local_i: u8, local_j: u8, r0: f32 }`,
where `local_i` and `local_j` are indices into the group's atom slice
(so `0..g.atom_count`) and `r0` is the constraint distance in Bohr
(`a_0`).
Local indices restrict any single group to at most 256 atoms — sufficient
for all known SETTLE and M-SHAKE clusters; LINCS-class systems use a
different layout outside this framework.

Each group's algorithm is determined by looking up
`constraint_types[group.constraint_type_index]` (a `NamedSlotConfig`
from `io/config-schema.md`), then resolving the entry's `kind`
string against the `ConstraintRegistry` to obtain the registered
builder. The framework never tracks the algorithm tag separately —
the kind string carried by the config entry is the single source of
truth, and consumers (the topology parser, the `Constraint` slot
constructor, the registry) all resolve it the same way.

### Group Ordering <!-- rq-38199bd4 -->

`groups` is sorted by the minimum particle index each group contains.
The sort is total: every group's atom set is disjoint from every other
group's atom set (the parser rejects overlap), so the minimum-index
key is unique. Group ordering is therefore identical across runs from
identical inputs — required for the bit-wise reproducibility invariant.

`group_atoms` and `group_constraints` are stored in the same `groups`
order, so a kernel that assigns one thread per group reads consecutive
slices of both flat buffers.

### Implicit Exclusions from Constraint Groups <!-- rq-fdac1afb -->

Every intra-group atom pair is added to the effective `ExclusionList`
as `scale_lj = scale_coul = 0.0` unless an explicit `[exclusions]` row
covers that pair. The intra-group pairs include:

- Every constraint pair `(local_i, local_j)` listed in the group's
  `group_constraints` slice (the 1-2 pairs).
- Every other pair drawn from the group's atom set (1-3 and
  higher-order pairs reachable through chains of constraints inside
  the group).

For a SETTLE water group with atoms `(O, H1, H2)`, the implicit
exclusions are `(O, H1)`, `(O, H2)`, and `(H1, H2)`.

These implicit exclusions are merged into the `ExclusionList` returned
by `load_topology_file` (see `forces/topology.md`). The precedence
rules match the existing bond- and angle-derived implicit-exclusion
behaviour: an explicit `[exclusions]` entry overrides the implicit
default for that pair.

## Constraint-Type Validation <!-- rq-402cea33 -->

Per-group shape validation lives on the `ConstraintBuilder` trait
(see *Feature API* below). When `ConstraintRegistry::build_optional`
constructs the slot, it iterates the `ConstraintList`'s groups, looks
up each group's algorithm via
`constraint_types[group.constraint_type_index].kind`, and calls the
matching builder's `validate_group_shape(...)` before delegating to
`build(...)`. The topology parser also calls the builder's
`expected_atom_count(&params)` to size-check `[constraints]` rows.

Per-builder shape rules are documented in each algorithm's
requirements file. For SETTLE (see `settle.md`):

- `expected_atom_count(&params)` returns `3`.
- `validate_group_shape(...)` requires exactly 3 atoms and exactly 3
  constraints. The constraint pairs (after local re-indexing) must be
  `(0, 1)`, `(0, 2)`, and `(1, 2)`. The two constraints incident to
  local atom 0 must have the same `r0` (the O–H distance); the third
  constraint's `r0` is the H–H distance. Both distances are read from
  the named `[[constraint_types]]` entry's `params` (`r_oh` and
  `r_hh` fields), and the populated `r0` values come from those
  config fields, not from the topology file.

Shape-mismatch failures surface as
`ConstraintError::InvalidGroupShape { group_index, kind, reason }`
where `kind` is the algorithm's kind string.

## Feature API <!-- rq-6f8f049b -->

### Types <!-- rq-289ab328 -->

- `Constraint` — object-safe trait implemented by every concrete <!-- rq-ca6db610 -->
  constraint slot.

  ```rust
  pub trait Constraint: std::fmt::Debug + Send {
      fn apply_before_drift(
          &mut self,
          buffers: &mut ParticleBuffers,
          sim_box: &SimulationBox,
          dt: f32,
          timings: &mut Timings,
      ) -> Result<(), ConstraintError>;

      fn apply_after_drift(
          &mut self,
          buffers: &mut ParticleBuffers,
          sim_box: &SimulationBox,
          dt: f32,
          timings: &mut Timings,
      ) -> Result<(), ConstraintError>;

      fn apply_after_kick(
          &mut self,
          buffers: &mut ParticleBuffers,
          sim_box: &SimulationBox,
          dt: f32,
          timings: &mut Timings,
      ) -> Result<(), ConstraintError>;

      /// Project positions back onto the constraint manifold without
      /// modifying velocities, virials, or any other buffer.
      /// Driven by the minimization runner (see
      /// `rqm/minimization/steepest-descent.md`); never called from the
      /// integration plan walk.
      fn apply_position_projection_only(
          &mut self,
          buffers: &mut ParticleBuffers,
          sim_box: &SimulationBox,
          timings: &mut Timings,
      ) -> Result<(), ConstraintError>;
  }
  ```

  - Every hook returns `Ok(())` immediately when the slot owns zero
    constraint groups.
  - Hooks never modify `force_field` or `sim_box` and never read or
    write `buffers.forces_*` or `buffers.potential_energies`.
  - `apply_before_drift` is permitted to mutate slot-private buffers
    only; `buffers` is borrowed mutably for symmetry with the other
    hooks and to allow future hooks that need to mutate state.
  - `apply_after_kick` is permitted to mutate `buffers.virials` to
    add the constraint slot's contribution to the total scalar
    virial used by the barostat's pressure estimate. The
    contribution is *added* (not assigned): each constraint slot
    adds in-place, so the force evaluation that runs between
    `apply_after_drift` and `apply_after_kick` populates virials
    first and the constraint contribution is folded in afterward.
    Concrete algorithms document the exact form of their
    contribution (see `settle.md` for the `settle-water` two-half
    decomposition: a position-level `m · Δr · r / dt²` part from
    the SHAKE projection plus a velocity-level `m · Δv · r / dt`
    part from the RATTLE projection).
  - `apply_position_projection_only` mutates only
    `buffers.positions_*`. It does not touch `buffers.velocities_*`,
    `buffers.virials`, `buffers.forces_*`, or `sim_box`, and it does
    not consume `dt` (minimization has no physical time scale). It
    is permitted to read and write slot-private scratch buffers. The
    runner invokes it after each trial position update inside the
    SD loop (see `rqm/minimization/steepest-descent.md`'s *Algorithm*
    step 2); the integrator plan walk never calls it.

- `ConstraintGroup` — `Debug, Clone, Copy`. Fields: `atom_offset: u32`, <!-- rq-0faddd62 -->
  `atom_count: u32`, `constraint_offset: u32`, `constraint_count: u32`,
  `constraint_type_index: u32`.

- `GroupConstraint` — `Debug, Clone, Copy`. Fields: `local_i: u8`, <!-- rq-f28b82a7 -->
  `local_j: u8`, `r0: f32`. Invariant: `local_i < local_j`.

- `ConstraintList` — host-side parsed-and-validated constraint data. <!-- rq-b6f167a4 -->
  Fields:
  - `groups: Vec<ConstraintGroup>`
  - `group_atoms: Vec<u32>`
  - `group_constraints: Vec<GroupConstraint>`
  - `particle_count: usize`

  Method `ConstraintList::is_empty(&self) -> bool` —
  `self.groups.is_empty()`.

  Each group's algorithm is resolved at consume time from
  `constraint_types[group.constraint_type_index].kind` (the
  `NamedSlotConfig`'s `kind` string), not stored on the list itself.

- `ConstraintRegistry` — host-side registry of constraint builders. <!-- rq-3cca2cb1 -->
  Holds `builders: Vec<Box<dyn ConstraintBuilder>>`.

  Methods:

  - `ConstraintRegistry::with_builtins() -> ConstraintRegistry` —
    constructs a registry pre-populated with the builders for every
    `kind` value in the table above. In v1 this is the single `settle`
    builder.
  - `ConstraintRegistry::register(&mut self, builder: Box<dyn ConstraintBuilder>)`
    — appends a builder. Two builders sharing the same `kind_name()`
    are not detected at registration; the lookup returns the first
    match.
  - `ConstraintRegistry::lookup(&self, kind: &str) -> Option<&dyn ConstraintBuilder>`
    — returns the first registered builder whose `kind_name()` equals
    `kind`. The topology parser uses this to call
    `expected_atom_count(&params)` per constraint-type entry; the
    runner uses it to drive `validate_params` and
    `validate_group_shape` at config-validation time.
  - `ConstraintRegistry::build_optional(&self, list: &ConstraintList, gpu: &GpuContext, particle_count: usize, masses: &[f32], constraint_types: &[NamedSlotConfig]) -> Result<Option<Box<dyn Constraint>>, ConstraintError>`
    — when `list.is_empty()`, returns `Ok(None)`. Otherwise, for every
    group in `list`, looks up the algorithm via
    `constraint_types[group.constraint_type_index].kind`, finds the
    matching builder, calls `validate_group_shape(...)`, and finally
    delegates to the slot constructor. Returns
    `ConstraintError::UnsupportedKind(kind)` when any group references
    a kind not present in the registry.

- `ConstraintBuilder` — trait describing a registered constraint <!-- rq-7896e33a -->
  implementation. Implementations are stateless and self-register at
  construction time.

  ```rust
  pub trait ConstraintBuilder: std::fmt::Debug + Send + Sync {
      /// Lookup key used by ConstraintRegistry to dispatch a
      /// `NamedSlotConfig`'s `kind` string to this builder.
      fn kind_name(&self) -> &'static str;

      /// Validate the kind-specific parameters of a
      /// `[[constraint_types]]` entry at config-load time. Called by
      /// `Config::validate_against(&registries)` before any GPU work.
      fn validate_params(&self, params: &toml::Value)
          -> Result<(), ConfigError>;

      /// `true` iff the algorithm implements
      /// `Constraint::apply_position_projection_only` non-trivially
      /// (i.e., can participate in minimization phases). The default
      /// returns `true`. Algorithms that cannot project positions
      /// without a paired velocity / virial update override this to
      /// return `false`; configs that pair such an algorithm with a
      /// `[[minimization]]` phase are rejected at config load via
      /// `Config::validate_constraint_compatibility`.
      fn supports_position_projection_only(
          &self,
          _params: &toml::Value,
      ) -> bool { true }

      /// Number of atoms a single `[constraints]` topology row of
      /// this kind must declare. The topology parser uses this value
      /// to validate row column counts.
      fn expected_atom_count(&self, params: &toml::Value) -> usize;

      /// Validate the cluster shape of a single constraint group
      /// against this algorithm's requirements (atom count,
      /// constraint-pair pattern, mass consistency, etc.). Called by
      /// `ConstraintRegistry::build_optional` for every group whose
      /// algorithm matches this builder.
      fn validate_group_shape(
          &self,
          group_index: usize,
          atoms: &[u32],
          constraints: &[GroupConstraint],
          params: &toml::Value,
          masses: &[f32],
      ) -> Result<(), ConstraintError>;

      /// Construct the slot implementation. Receives the full
      /// `ConstraintList` and the `NamedSlotConfig` array; the
      /// builder filters internally to the groups whose algorithm
      /// matches its `kind_name()`. In v1 the single registered
      /// builder consumes every group.
      fn build(
          &self,
          gpu: &GpuContext,
          particle_count: usize,
          list: &ConstraintList,
          masses: &[f32],
          constraint_types: &[NamedSlotConfig],
      ) -> Result<Box<dyn Constraint>, ConstraintError>;
  }
  ```

  - `validate_params` is a pure function of the supplied parameters
    and must not allocate device memory.
  - `expected_atom_count` is also a pure function of the supplied
    parameters. SETTLE returns `3` regardless of `params`.
  - `validate_group_shape` runs before `build` and surfaces
    algorithm-specific cluster-shape errors as
    `ConstraintError::InvalidGroupShape`.
  - `build` constructs the device-side slot.

- `ConstraintError` — error type returned by every trait method. <!-- rq-feb0501c -->
  Variants:
  - `Gpu(GpuError)` — CUDA driver / kernel-launch failure.
  - `Timings(TimingsError)` — CUDA event recording failure.
  - `UnsupportedKind(String)` — the registry has no builder for the
    requested algorithm kind. The string is the unresolved `kind`
    from the corresponding `NamedSlotConfig`.
  - `UnknownConstraintType(String)` — the `[constraints]` row
    references a constraint-type name that does not appear in the
    config's `[[constraint_types]]` array.
  - `DuplicateConstraintAtom { atom: u32 }` — two constraint rows
    share an atom (forbidden in v1; clusters are disjoint).
  - `InvalidGroupShape { group_index: usize, kind: String, reason: String }`
    — the parsed group does not satisfy the algorithm's required
    cluster shape (atom count, constraint pattern, etc.).
  - `ConstraintBondPairOverlap { atom_i: u32, atom_j: u32 }` — an
    `(atom_i, atom_j)` pair is named both in `[bonds]` and (after
    expansion of constraint groups into their pairwise constraints)
    in `[constraints]`.

  The runner's `RunnerError` wraps the type via
  `RunnerError::Constraint(ConstraintError)`.

### Functions and Methods <!-- rq-dfd47225 -->

- `IntegratorBuilder::supports_constraints(&self, params: &toml::Value) -> bool` <!-- rq-9331ede2 -->
  — defined on the `IntegratorBuilder` trait (see `framework.md`).
  Returns `true` for builders whose integrator drives the three
  `Constraint` hooks. Default-registry returns:
  - `"velocity-verlet"` builder with `params.lossless == false` →
    `true`.
  - `"velocity-verlet"` builder with `params.lossless == true` →
    `false`.
  - `"langevin-baoab"` builder → `false` (regardless of `params`).
  - `"mtk-npt"` builder → `false` (regardless of `params`).

- `ConstraintRegistry::build_optional(&self, list: &ConstraintList, gpu: &GpuContext, particle_count: usize, masses: &[f32], constraint_types: &[NamedSlotConfig]) -> Result<Option<Box<dyn Constraint>>, ConstraintError>` <!-- rq-b004196f -->
  — as described above.

- `Constraint::apply_before_drift`, `Constraint::apply_after_drift`, <!-- rq-e538c545 -->
  `Constraint::apply_after_kick` — as described above. Each returns
  `Ok(())` without launching any kernel when the slot's group count
  is zero.

## Construction and Lifetime <!-- rq-2523992c -->

The runner constructs the constraint slot immediately after building
the integrator, thermostat, and barostat slots. Construction draws
from the `ConstraintList` returned by `load_topology_file`:

```rust
let constraint = ConstraintRegistry::with_builtins()
    .build_optional(
        &constraint_list,
        &gpu,
        particle_count,
        &masses,
        &config.constraint_types,
    )?;
```

The slot's per-group device buffers are allocated on the runner's
`Arc<CudaDevice>` inside the builder. All allocations persist for the
lifetime of the run and are dropped together with the rest of the
runner's GPU resources at end of run.

## Empty State <!-- rq-c491802b -->

When the runner has `particle_count == 0`, every constraint hook
returns `Ok(())` without launching any kernel. The slot's allocations
may have zero-length device slices but the builder must construct
successfully. An empty `ConstraintList` always yields `None` from
`build_optional`.

## Determinism Guarantees <!-- rq-9fe1d656 -->

The constraint framework preserves the project's bit-wise
reproducibility invariant on the same GPU under the conditions each
algorithm individually guarantees:

- All constraint kernels run on the default stream of the same
  `Arc<CudaDevice>` carried by `ParticleBuffers`. No additional
  streams are introduced.
- The intra-step hook order (`apply_before_drift`, then drift, then
  `apply_after_drift`, then force evaluation, then kick, then
  `apply_after_kick`) is fixed and identical across runs.
- Group order in the device-side group buffers matches `groups` in
  the host-side `ConstraintList`, which is sorted by each group's
  minimum particle index. Two independently-constructed
  `ConstraintList`s from identical inputs are byte-identical, and
  every kernel that consumes them processes groups in the same order.
- Concrete algorithms (SETTLE in v1) document the fixed-order
  per-group computation that produces bit-identical outputs across
  runs on the same GPU.

## Out of Scope <!-- rq-acb86c9b -->

- Concrete constraint algorithms other than SETTLE. M-SHAKE is the
  target of the next constraint feature and shares this framework's
  data layout, trait, and dispatch. LINCS-class global constraint
  solvers require a different layout and are not anticipated by this
  framework.
- Composing constraints with `velocity-verlet { lossless: true }`,
  with `langevin-baoab`, or with `mtk-npt`. Each is rejected at
  config load via the
  `IntegratorBuilder::supports_constraints(&params)` predicate.
  Lossless support is the target of a follow-up feature that designs
  how constraint corrections fold into the `(f32 high, f64 low)`
  compensated sum.
- Cluster building that merges constraint rows sharing atoms. Each
  row in `[constraints]` is its own group in v1; the parser rejects
  overlap. The connected-components algorithm needed to merge shared
  atoms across rows arrives with M-SHAKE.
- Constraint diagnostics beyond config-load validation (per-step
  residual reporting, per-group iteration counts). SETTLE is
  non-iterative; M-SHAKE will introduce its own diagnostics.
- Mid-run replacement of the constraint slot. The slot is fixed at
  construction and never replaced for the duration of a run.
- User-defined constraint algorithms via a DSL. New algorithms are
  added as Rust source files implementing the `Constraint` trait and
  registering a `ConstraintBuilder`.

---

## Gherkin Scenarios <!-- rq-a0cb32bf -->

```gherkin
Feature: Constraint slot framework

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called

  # --- Construction ---

  @rq-7921e537
  Scenario: Construct a constraint slot for a non-empty constraint list
    Given a ConstraintRegistry::with_builtins()
    And a ConstraintList containing one SettleWater group with atoms [0, 1, 2]
    When registry.build_optional(&list, device, particle_count=3) is called
    Then it returns Ok(Some(slot))

  @rq-aea6734a
  Scenario: Empty constraint list yields None
    Given a ConstraintRegistry::with_builtins()
    And an empty ConstraintList
    When registry.build_optional(&list, device, particle_count=3) is called
    Then it returns Ok(None)

  @rq-7ef08958
  Scenario: Unsupported constraint kind reports UnsupportedKind
    Given a ConstraintRegistry whose builders cover only "settle"
    And a ConstraintList referencing a constraint type with kind = "m-shake"
    When registry.build_optional(&list, &gpu, particle_count=4, &masses, &constraint_types) is called
    Then it returns Err(ConstraintError::UnsupportedKind("m-shake"))

  @rq-18165336
  Scenario: Empty ConstraintRegistry on a non-empty list reports UnsupportedKind
    Given an empty ConstraintRegistry (no builders registered)
    And a ConstraintList referencing one group with constraint type kind = "settle-water"
    When registry.build_optional(&list, &gpu, particle_count=3, &masses, &constraint_types) is called
    Then it returns Err(ConstraintError::UnsupportedKind("settle-water"))

  @rq-744ddd67
  Scenario: Construct on particle_count == 0 with an empty list
    Given a ConstraintRegistry::with_builtins()
    And an empty ConstraintList
    When registry.build_optional(&list, device, particle_count=0) is called
    Then it returns Ok(None)

  # --- IntegratorBuilder::supports_constraints(&params) ---

  @rq-fd07b4dc
  Scenario: velocity-verlet (lossy) supports constraints
    Given the "velocity-verlet" builder from IntegratorRegistry::with_builtins()
    And params = { lossless: false }
    Then builder.supports_constraints(&params) returns true

  @rq-53237ec4
  Scenario: velocity-verlet (lossless) does not support constraints
    Given the "velocity-verlet" builder from IntegratorRegistry::with_builtins()
    And params = { lossless: true }
    Then builder.supports_constraints(&params) returns false

  @rq-047c1f4d
  Scenario: langevin-baoab does not support constraints
    Given the "langevin-baoab" builder from IntegratorRegistry::with_builtins()
    And params = { friction: 1.0e12, temperature: 300.0, seed: 0 }
    Then builder.supports_constraints(&params) returns false

  @rq-09a19014
  Scenario: mtk-npt does not support constraints
    Given the "mtk-npt" builder from IntegratorRegistry::with_builtins()
    And params = { temperature: 85.0, pressure: 1.0e5,
      tau_t: 1.0e-13, tau_p: 1.0e-12,
      chain_length: 3, yoshida_order: 3, n_resp: 1 }
    Then builder.supports_constraints(&params) returns false

  # --- Compatibility rules at config load ---

  @rq-064c9df1
  Scenario: Reject [constraints] with an integrator that does not support them
    Given a config with kind = "langevin-baoab" and a topology file whose [constraints] section is non-empty
    When load_config(&path) is called
    Then it returns Err(ConfigError::IncompatibleConstraint { integrator: "langevin-baoab" })

  @rq-58476106
  Scenario: Reject [constraints] with lossless velocity-verlet
    Given a config with kind = "velocity-verlet", lossless = true, and a topology file whose [constraints] section is non-empty
    When load_config(&path) is called
    Then it returns Err(ConfigError::IncompatibleConstraint { integrator: "velocity-verlet" })

  @rq-fe2cb7ee
  Scenario: Empty [constraints] with a non-supporting integrator is permitted
    Given a config with kind = "langevin-baoab" and a topology file whose [constraints] section is empty
    When load_config(&path) is called
    Then it returns Ok(config)
    And the runner holds None for the constraint slot

  # --- Per-step dispatch ---

  @rq-90538790
  Scenario: Dispatch loop calls all three constraint hooks in order
    Given a velocity-Verlet integrator (lossless=false) with constraints
    And a recording wrapper that timestamps every Constraint hook call
    When the runner executes one timestep
    Then the recorded order is exactly
      [apply_before_drift, apply_after_drift, apply_after_kick]
    And each hook fires exactly once

  @rq-7047ea32
  Scenario: Integrator without constraints skips all three hooks
    Given a velocity-Verlet integrator (lossless=false) with constraint slot = None
    And a recording stub Constraint that records every call
    When the runner executes one timestep
    Then no constraint hooks are recorded

  @rq-99034e90
  Scenario: Plan with a single Drift sub-step fires before_drift and after_drift around it
    Given a stub integrator whose plan(dt) returns [Drift, ForceEval, KickHalf]
      and supports_constraints() returns true
    And a recording Constraint slot
    When the runner executes one timestep
    Then the recorded order is exactly
      [apply_before_drift, apply_after_drift, apply_after_kick]

  @rq-3b42c2ff
  Scenario: Plan with two Drift sub-steps fires before/after_drift twice
    Given a stub integrator whose plan(dt) returns
      [KickHalf, Drift, Custom("ou"), Drift, ForceEval, KickHalf]
      and supports_constraints() returns true
    And a recording Constraint slot
    When the runner executes one timestep
    Then apply_before_drift fires exactly twice (once before each Drift)
    And apply_after_drift fires exactly twice (once after each Drift)
    And apply_after_kick fires exactly once after the final KickHalf

  @rq-a90e4189
  Scenario: Plan whose final sub-step is not a Kick does not fire after_kick
    Given a stub integrator whose plan(dt) returns [KickHalf, ForceEval, Custom("post")]
      and supports_constraints() returns true
    And a recording Constraint slot
    When the runner executes one timestep
    Then apply_after_kick is not recorded

  @rq-c3b3ec99
  Scenario: Custom sub-step alone fires no constraint hooks
    Given a stub integrator whose plan(dt) returns [Custom("ou")]
      and supports_constraints() returns true
    And a recording Constraint slot
    When the runner executes one timestep
    Then no constraint hooks are recorded

  @rq-77f959b2
  Scenario: apply_position_projection_only is not fired during the MD plan walk
    Given a velocity-Verlet integrator (lossless=false) with a SETTLE constraint slot
    And a recording wrapper that timestamps every Constraint hook call
    When the runner executes one MD timestep
    Then apply_position_projection_only is not recorded
    And the recorded hooks are exactly [apply_before_drift, apply_after_drift, apply_after_kick]

  @rq-178eb1ae
  Scenario: apply_position_projection_only mutates positions but not velocities or virials
    Given a Constraint slot constructed from a non-empty list
    And a ParticleBuffers with off-manifold positions and snapshots of velocities_*, virials, and forces_*
    When constraint.apply_position_projection_only(&mut buffers, &sim_box, &mut timings) is called
    Then positions_x, positions_y, positions_z lie on the constraint manifold (every constraint distance equals its r0 within relative tolerance 1e-6)
    And velocities_x, velocities_y, velocities_z are byte-identical to the snapshot
    And virials and forces_x, forces_y, forces_z are byte-identical to their snapshots

  @rq-833d83a9
  Scenario: apply_position_projection_only on empty state is a no-op
    Given a ParticleBuffers with particle_count() == 0
    And a Constraint slot constructed with an empty ConstraintList
    When constraint.apply_position_projection_only(&mut buffers, &sim_box, &mut timings) is called
    Then it returns Ok(())
    And no kernel launches are recorded for that call

  @rq-fd51fccb
  Scenario: ConstraintBuilder default supports_position_projection_only returns true
    Given the registered SETTLE builder
    And any well-formed settle-water params
    Then builder.supports_position_projection_only(&params) returns true

  @rq-309d8d50
  Scenario: supports_constraints == false suppresses all hook insertion
    Given a stub integrator whose plan(dt) returns [KickDrift, ForceEval, KickHalf]
      and supports_constraints() returns false
    And a recording Constraint slot
    When the runner executes one timestep
    Then no constraint hooks are recorded
    (the config loader would normally reject this combination before reaching the loop;
     this scenario exercises the runner-side safety check.)

  @rq-03329010
  Scenario: apply_before_drift on empty state is a no-op
    Given a ParticleBuffers with particle_count() == 0
    And a Constraint slot constructed with an empty ConstraintList
    When constraint.apply_before_drift(&mut buffers, &sim_box, dt=0.1, &mut timings) is called
    Then it returns Ok(())
    And no kernel launches are recorded for that call

  @rq-129cb281
  Scenario: apply_after_drift on empty state is a no-op
    Given a ParticleBuffers with particle_count() == 0
    And a Constraint slot constructed with an empty ConstraintList
    When constraint.apply_after_drift(&mut buffers, &sim_box, dt=0.1, &mut timings) is called
    Then it returns Ok(())
    And no kernel launches are recorded for that call

  @rq-375aba37
  Scenario: apply_after_kick on empty state is a no-op
    Given a ParticleBuffers with particle_count() == 0
    And a Constraint slot constructed with an empty ConstraintList
    When constraint.apply_after_kick(&mut buffers, &sim_box, dt=0.1, &mut timings) is called
    Then it returns Ok(())
    And no kernel launches are recorded for that call

  # --- Group ordering & determinism ---

  @rq-930121d6
  Scenario: Group order is determined by minimum particle index
    Given a topology file declaring three SETTLE waters with O atoms at indices 100, 4, 50
    When load_topology_file(...) is called
    Then constraint_list.groups[0]'s minimum atom index is 4
    And constraint_list.groups[1]'s minimum atom index is 50
    And constraint_list.groups[2]'s minimum atom index is 100

  @rq-4f88e13c
  Scenario: Two independently constructed ConstraintLists are byte-identical
    Given two ConstraintList instances built from byte-identical topology + config inputs
    Then every field of list A equals the corresponding field of list B byte-for-byte

  # --- Constraint group shape validation ---

  @rq-2fbcb56c
  Scenario: SettleWater group with the wrong atom count is rejected
    Given a ConstraintList in which a SettleWater group has 4 atoms
    When the SETTLE builder is invoked
    Then it returns Err(ConstraintError::InvalidGroupShape { kind: SettleWater, reason: contains "atom count", .. })

  @rq-c2e0c1fa
  Scenario: SettleWater group with a missing constraint pair is rejected
    Given a ConstraintList in which a SettleWater group declares only the (0,1) and (0,2) constraint pairs (no (1,2) constraint)
    When the SETTLE builder is invoked
    Then it returns Err(ConstraintError::InvalidGroupShape { kind: SettleWater, reason: contains "constraint pattern", .. })

  # --- Implicit exclusions ---

  @rq-18f5ef7a
  Scenario: Constraint group adds implicit exclusions for every intra-group pair
    Given a topology file with no [exclusions] section and one SETTLE row "0 1 2 SPCE"
    When load_topology_file(...) is called
    Then exclusion_list.entries contains (0, 1, 0.0, 0.0)
    And exclusion_list.entries contains (0, 2, 0.0, 0.0)
    And exclusion_list.entries contains (1, 2, 0.0, 0.0)

  @rq-413d9c2b
  Scenario: Explicit [exclusions] entry overrides constraint-derived default
    Given a topology file with one SETTLE row "0 1 2 SPCE" and an explicit exclusion "1 2 0.5 0.5"
    When load_topology_file(...) is called
    Then exclusion_list.entries contains (1, 2, 0.5, 0.5)
    And exclusion_list.entries does not contain (1, 2, 0.0, 0.0)

  # --- Atom-overlap validation ---

  @rq-7f5b3a74
  Scenario: Two constraint rows sharing an atom are rejected
    Given a topology file with rows "0 1 2 SPCE" and "2 3 4 SPCE" (atom 2 appears in both)
    When load_topology_file(...) is called
    Then it returns Err(ConstraintError::DuplicateConstraintAtom { atom: 2 })

  @rq-d05a8f16
  Scenario: Pair appearing in both [bonds] and [constraints] is rejected
    Given a topology file with bond "0 1 OH" and constraint row "0 1 2 SPCE" (constraint expands to (0,1), (0,2), (1,2))
    When load_topology_file(...) is called
    Then it returns Err(ConstraintError::ConstraintBondPairOverlap { atom_i: 0, atom_j: 1 })
```
