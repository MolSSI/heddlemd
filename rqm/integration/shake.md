# Feature: SHAKE + RATTLE Constraint Algorithm <!-- rq-f17b858f -->

SHAKE is the iterative Gauss-Seidel projection of a set of pair-distance
holonomic constraints onto the constraint manifold; RATTLE is its
velocity-level counterpart that zeroes the time-derivative of each
constraint. Together they enforce rigid pair-distance constraints in
arbitrary connected atom groups at every timestep.

A `ShakeConstraintsState` implements the `Constraint` trait (see
`integration/constraint-framework.md`). Its `apply_before_drift` hook
snapshots pre-drift positions; its `apply_after_drift` hook runs the
SHAKE projection of the unconstrained post-drift positions, updates the
half-step velocities to be consistent with the position correction, and
writes the position-level half of the per-atom constraint-virial
contribution; its `apply_after_kick` hook runs the RATTLE velocity
projection and accumulates the velocity-level half of the
constraint-virial contribution; its `apply_position_projection_only`
hook runs a position-only SHAKE used by the minimizer.

This feature handles arbitrary rigid constraint groups whose
constraint-graph component sizes fit the compile-time caps
`MAX_GROUP_ATOMS = 8` and `MAX_GROUP_CONSTRAINTS = 12`. Three-atom rigid
water is the smallest and most common special case; methanol (6 atoms,
5 constraints), methane (5 atoms, 4 constraints), and small fluorinated
or hydroxylated organics all fit within the caps.

## Algorithm <!-- rq-9a80c43c -->

For one constraint group `g` with atoms `[a_0, a_1, ..., a_{n-1}]`
(local indices `0..n`, global indices stored in the topology's
`group_atoms`), masses `m_i = masses[a_i]`, and constraint set
`{(i_k, j_k, d_k) : k ∈ 0..K}` (local atom pair indices and target
distance):

1. `apply_before_drift` invokes `shake_snapshot`, which copies every
   atom of every group from `positions_*` into per-group snapshot
   buffers `snapshot_*`. The snapshot is used as the constraint-gradient
   reference frame for the iteration that follows the drift.

2. `apply_after_drift` invokes `shake_positions`, which iteratively
   solves the K pair-distance constraints for group `g`:

   ```text
   r_i ← positions[a_i]   for i = 0..n     // unconstrained post-drift positions
   r_i^{(0)} ← snapshot at base + i        // pre-drift, used as constraint-gradient frame
   Bring every r_i into the same lattice image as r_0 via min-image fix-up.
   Bring every r_i^{(0)} into the same lattice image as r_0^{(0)}.
   for k in 0..K:
       g_k ← r_{i_k}^{(0)} − r_{j_k}^{(0)}       # constraint-gradient direction
   for iter in 0..SHAKE_MAX_ITER (= 32):
       converged ← true
       for k in 0..K:
           σ_k ← |r_{i_k} − r_{j_k}|² − d_k²
           if |σ_k| > SHAKE_TOL² (= 1.0e-26 m²):
               converged ← false
               ddot ← (r_{i_k} − r_{j_k}) · g_k
               inv_m ← 1/m_{i_k} + 1/m_{j_k}
               λ_k ← σ_k / (2 · ddot · inv_m)
               r_{i_k} ← r_{i_k} − λ_k · g_k / m_{i_k}
               r_{j_k} ← r_{j_k} + λ_k · g_k / m_{j_k}
       if converged: break
   ```

   After the loop, the half-step velocity for every atom in the group
   is updated by `v_i ← v_i + (r_i − r_i^{unconstrained}) / dt`, the
   constrained positions are written back to `positions_*`, and the
   per-atom position-level constraint-virial contribution is written
   into `constraint_virial[3 * group_offset + i]`. The virial formula
   is `(m_i / dt²) · ((r_i − r_i^{unconstrained}) · r_i^{COM})` where
   `r_i^{COM} = r_i − r_COM` and `r_COM` is the group's mass-weighted
   centre of mass (preserved by SHAKE).

3. `apply_after_kick` invokes `rattle_velocities`, which iteratively
   projects post-kick velocities onto the velocity manifold of the
   constrained positions:

   ```text
   r_i ← positions[a_i] (already constrained, on the position manifold)
   Bring every r_i into the same lattice image as r_0.
   for k in 0..K:
       d_k ← r_{i_k} − r_{j_k}                    # constraint-gradient direction at current r
   v_i ← velocities[a_i]
   for iter in 0..RATTLE_MAX_ITER (= 32):
       converged ← true
       for k in 0..K:
           v_rel_k ← (v_{i_k} − v_{j_k}) · d_k
           if |v_rel_k| > RATTLE_TOL (= 1.0e-20 m²/s):
               converged ← false
               inv_m ← 1/m_{i_k} + 1/m_{j_k}
               μ_k ← v_rel_k / (|d_k|² · inv_m)
               v_{i_k} ← v_{i_k} − μ_k · d_k / m_{i_k}
               v_{j_k} ← v_{j_k} + μ_k · d_k / m_{j_k}
       if converged: break
   ```

   When `dt > 0` (the standard call from the integrator's post-kick
   hook), the kernel additionally accumulates the velocity-level
   constraint-virial contribution into `constraint_virial[base + i]`
   (additive on top of the position-level half already written by
   `shake_positions`). The per-atom velocity-level virial is
   `m_i · Δv_i · r_i^{COM} / dt`, where `Δv_i` is the cumulative
   velocity correction applied to atom `i` during the RATTLE
   iteration. When `dt ≤ 0` (the runner's initial-velocity projection
   at setup, where there is no associated timestep), the virial
   accumulation is skipped.

4. `apply_position_projection_only` invokes
   `shake_positions_no_velocity`, used by the minimizer after each
   trial position update. This kernel runs the same SHAKE iteration as
   `shake_positions` but with two differences: the constraint-gradient
   direction `g_k` is evaluated at the *current* (off-manifold)
   positions rather than at a snapshot (minimization has no notion of
   a pre-drift frame), and velocities and the constraint-virial buffer
   are not modified.

### Iteration count and convergence <!-- rq-2d336703 -->

For thermal MD step sizes (`dt = 1–4 fs`), the inner displacement
`|r' − r^{(0)}|` is small relative to the constraint distances, the
linearised Newton step inside the loop converges quadratically, and
typical groups converge in 1–3 sweeps. `SHAKE_MAX_ITER = 32` is the
guaranteed upper bound; iteration count past 8 indicates pathologically
large unconstrained displacements (e.g. minimization with an oversized
trial step) and is rare in equilibrated dynamics. The constraint-virial
buffer is written even if the loop exits at `SHAKE_MAX_ITER` without
strict convergence — the projection is still bit-reproducible across
runs because the iteration is deterministic, and the per-step residual
is bounded by `SHAKE_TOL²`.

The RATTLE iteration is structurally identical and converges on the
same scale of sweeps. Its tolerance `RATTLE_TOL = 1.0e-20 m²/s` is a
constant absolute bound on the residual `v_rel · d_k`. At
representative scales (`d_k ≈ 1 Å = 10⁻¹⁰ m`, thermal `v_rel ≈ 10³ m/s`),
the residual settles below `RATTLE_TOL` in 1–3 sweeps; the f32
representation of the per-iteration update is more than 6 decimal
digits above the threshold.

## Per-Step Kernel Sequence <!-- rq-157e59ad -->

For each `Constraint` hook called by the runner on a step where the
slot contains at least one group:

| Hook | Kernels launched (in order) | Notes |
|---|---|---|
| `apply_before_drift` | `shake_snapshot` | Reads current `positions_*`; writes `snapshot_*`. |
| `apply_after_drift` | `shake_positions` | Reads `snapshot_*`, current `positions_*`, current `velocities_*`. Writes constrained `positions_*`, updated `velocities_*`, position-level half of `constraint_virial`. |
| `apply_after_kick` | `rattle_velocities`, `constraint_virial_scatter` | Reads current constrained `positions_*` and post-kick `velocities_*`. Writes RATTLE-projected `velocities_*`, accumulates velocity-level half of `constraint_virial`, scatters `constraint_virial` into `particle_virials` for the barostat to consume. |
| `apply_position_projection_only` | `shake_positions_no_velocity` | Reads and writes `positions_*` only; does not touch velocities or virials. |

When the slot has zero groups (e.g. a rigid-water topology with no
SETTLE-shaped groups, or a config that omits `[constraints]`
entirely), every hook is a no-op and no kernel is launched.

## Constraint Virial <!-- rq-4617c285 -->

The per-atom constraint-virial contribution for atom `i` of group `g`
is the sum of a position-level and a velocity-level half:

```
W_i^position = (m_i / dt²) · ((r_i^constrained − r_i^unconstrained) · r_i^COM)
W_i^velocity = m_i · Δv_i^RATTLE · r_i^COM / dt
W_i         = W_i^position + W_i^velocity
```

where `r_i^COM = r_i^constrained − r^COM_group`, `r^COM_group` is the
group's mass-weighted centre of mass (preserved by both SHAKE and
RATTLE), and `Δv_i^RATTLE` is the cumulative velocity correction the
RATTLE iteration applied to atom `i`. The two halves are summed in the
`constraint_virial` buffer (position-level written by
`shake_positions`, velocity-level accumulated by `rattle_velocities`)
and then scattered into the global `particle_virials` array by
`constraint_virial_scatter`. The barostat's scalar-virial reduction
then sums `particle_virials` across all atoms; the result is the
analytic `−2 K_rot` of a rigid rotor for each group, as required by
the velocity-Verlet + SHAKE + RATTLE virial decomposition.

The arithmetic uses centre-of-mass-relative positions (rather than
lab-frame absolute positions) for f32 stability. At molecular-cluster
scales (`|r_i| ≈ 10⁻⁹ m` from the origin in large boxes, `|Δr_i| ≈
10⁻¹² m`, `1/dt² ≈ 10³⁰ s⁻²`), a direct `m · Δr · r` evaluation
underflows in f32 (the product is `≈ 10⁻⁵⁰` for water masses, below
the smallest denormal `≈ 1.4 · 10⁻⁴⁵`). The COM-relative product
`(m · Δr · r^COM) / dt²` keeps every intermediate well inside f32
normal range: the `(m / dt²) · (Δr · r^COM)` regrouping gives
intermediate values `≈ 10³` and `≈ 10⁻²³` and a product `≈ 10⁻²⁰ J`.

## Reproducibility <!-- rq-f410dd7b -->

The SHAKE iteration is deterministic in (group, constraint, iteration)
order: every group's iteration loop walks its `K` constraints in the
order declared by the topology (the `group_constraints[group.offset ..
group.offset + group.count]` slice), and every iteration starts from
the previous iteration's positions. Two independent runs on the same
GPU with identical inputs produce byte-identical `positions_*`,
`velocities_*`, and `constraint_virial` after every hook. The same
holds for RATTLE.

The integration framework runs constraint hooks on the device's
default stream, with all kernel launches in a single fixed order; this
matches `forces/framework.md`'s default-stream convention and means
the constraint slot is composable with the SPME `recip_stream` without
extra synchronisation (the constraint slot writes only to
particle-state buffers that the default stream owns).

## Group-Size Caps <!-- rq-81ce46b3 -->

`MAX_GROUP_ATOMS = 8` and `MAX_GROUP_CONSTRAINTS = 12` are kernel
compile-time constants. Per-thread storage for one group fits in
registers: each atom carries six floats (post-drift constrained
`(x, y, z)` and pre-drift snapshot `(x, y, z)`, plus inverse mass
`1/m`); each constraint carries an integer pair `(local_i, local_j)`
and a target squared distance `d_k²`. Total per-thread state at the
caps is `8 × 7 × 4 + 12 × (2 + 4) = 296 B`, well inside the per-thread
register budget of contemporary GPUs.

Groups whose atom count exceeds `MAX_GROUP_ATOMS` or whose constraint
count exceeds `MAX_GROUP_CONSTRAINTS` are rejected at slot
construction time with
`ShakeError::UnsupportedGroupSize { group_index, atoms, constraints }`.
The error message directs the user to a future M-SHAKE feature for
arbitrarily large groups.

## Parameters <!-- rq-eecd4961 -->

Per-constraint-type parameters are declared in the config's
`[[constraint_types]]` table with `kind = "shake"` (see
`io/config-schema.md`). One entry per distinct rigid molecular shape:

```toml
[[constraint_types]]
name = "SPCE"
kind = "shake"
atoms = 3
constraints = [
    { i = 0, j = 1, d = 1.0e-10 },     # O-H1
    { i = 0, j = 2, d = 1.0e-10 },     # O-H2
    { i = 1, j = 2, d = 1.633e-10 },   # H1-H2
]
```

Fields:

- `atoms: u32` — the number of atoms in every group of this type.
  Strictly positive, at most `MAX_GROUP_ATOMS = 8`.
- `constraints: Vec<{ i: u32, j: u32, d: f64 }>` — one entry per
  pair-distance constraint within the group.
  - `i` and `j` are local atom indices in `0..atoms`. The pair
    `(min(i, j), max(i, j))` must be unique across constraints; both
    `i` and `j` must be in range; `i != j`.
  - `d` is the target distance in metres (SI), strictly positive.
  - The list's length must be at most `MAX_GROUP_CONSTRAINTS = 12`,
    at least 1.

Per-row `[constraints]` entries reference the constraint-type by name
and list the global atom indices for the group's local-slot order.
For the SPC/E example above, a `[constraints]` row
`atom_O atom_H1 atom_H2 SPCE` declares one group with atom 0 = oxygen,
atom 1 = first hydrogen, atom 2 = second hydrogen.

The topology parser expands one row into a `ConstraintGroup` with
`atom_count = atoms`, `constraint_count = constraints.len()`, and one
`GroupConstraint { local_i, local_j, r0 }` per declared constraint
entry. `r0` is the value of `d` in metres, stored as `f32` after the
config-load atomic-units conversion (`io/unit-system.md`); kernels
square it once when computing `r0²`.

## Feature API <!-- rq-02727a63 -->

### Types <!-- rq-4861b9b9 -->

- `ShakeParams` — typed deserialiser for the `[[constraint_types]]` <!-- rq-55f60603 -->
  entry whose `kind == "shake"`. Fields:
  - `atoms: u32`
  - `constraints: Vec<ShakeConstraintSpec>`

- `ShakeConstraintSpec` — one row of the `constraints` array. Fields: <!-- rq-811ba2a0 -->
  - `i: u32`
  - `j: u32`
  - `d: f64` — target distance in metres.

- `ShakeConstraintsState` — implements the `Constraint` trait. Fields <!-- rq-d9a47c62 -->
  private; public surface is the `Constraint` trait methods plus the
  constructor.

  Constructor:
  - `ShakeConstraintsState::new(device: Arc<CudaDevice>, list: &ConstraintList, masses: &[f32], constraint_types: &[NamedSlotConfig]) -> Result<ShakeConstraintsState, ShakeError>`
    - Filters `constraint_types` to entries with `kind == "shake"`.
      For each such entry, deserialises `ShakeParams` from
      `entry.params` and verifies internal consistency (the
      `atoms`/`constraints` bounds described in *Parameters*).
    - Walks `list.groups`. For each group whose constraint type
      resolves to `"shake"`, copies the group's atom indices into a
      flat device array `group_atoms`, the per-constraint local-index
      pairs into `group_constraints_local_i / local_j`, and the squared
      target distances into `group_constraints_r2`. Per-group offsets
      and counts go into `group_atom_offset / atom_count /
      constraint_offset / constraint_count`.
    - Reads `masses[a]` for every atom referenced by any group,
      computes `1/masses[a]`, and stores the per-atom inverse mass in
      a device array `atom_inv_mass` of length `particle_count`. Atoms
      not referenced by any group keep `atom_inv_mass[a] = 1/masses[a]`
      (the array is populated for every atom from the topology's mass
      list; constraint-untouched atoms still appear there, harmlessly).
    - Allocates `snapshot_*` and `constraint_virial` buffers sized at
      `Σ_g atoms_g` (the same as `group_atoms.len()`).
    - Returns `Ok(state)` on success; the slot's hooks then write
      kernels on every per-step call.
    - When `list.is_empty()` or the list contains no groups whose
      constraint type resolves to `"shake"`, returns an empty slot
      whose hooks are all no-ops.
  - Returns `ShakeError` on a CUDA driver failure, on a malformed
    `[[constraint_types]]` entry, or on a group whose shape exceeds
    the per-group caps.

- `ShakeError` — error type returned by `ShakeConstraintsState::new`. <!-- rq-0b4600e2 -->
  Variants:
  - `Gpu(GpuError)` — a CUDA driver / kernel-launch failure during
    setup.
  - `Timings(TimingsError)` — a timings-system failure during setup.
  - `MalformedShakeType { name: String, reason: String }` — the named
    constraint type's `params` table failed to deserialise as
    `ShakeParams`, or violated one of the per-field bounds
    (`atoms == 0`, `atoms > MAX_GROUP_ATOMS`, empty `constraints`
    list, `constraints.len() > MAX_GROUP_CONSTRAINTS`, an out-of-range
    `i` or `j`, a non-positive or non-finite `d`, or a duplicated
    pair `(min(i, j), max(i, j))`).
  - `UnsupportedGroupSize { group_index: usize, atoms: u32, constraints: u32 }`
    — the group's `(atom_count, constraint_count)` exceeds
    `(MAX_GROUP_ATOMS, MAX_GROUP_CONSTRAINTS)`. The message directs
    the user to a future M-SHAKE feature for larger groups.
  - `GroupShapeMismatch { group_index: usize, expected_atoms: u32, actual_atoms: u32 }`
    — the `[constraints]` row for this group declared an atom count
    that disagrees with the constraint type's declared `atoms`. The
    topology parser also surfaces this via its own
    `InvalidConstraintRow`; the slot's pass is a defence-in-depth
    re-check.

  Converts into `ConstraintError` via a `From` impl.

### CUDA Kernels <!-- rq-53800cef -->

`kernels/shake.cu` declares five `extern "C"` kernels:

```c
extern "C" __global__ void shake_snapshot(
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    const unsigned int *group_atoms,
    const unsigned int *group_atom_offset,
    const unsigned int *group_atom_count,
    float *snapshot_x,
    float *snapshot_y,
    float *snapshot_z,
    unsigned int n_groups);

extern "C" __global__ void shake_positions(
    float *positions_x,
    float *positions_y,
    float *positions_z,
    float *velocities_x,
    float *velocities_y,
    float *velocities_z,
    const float *snapshot_x,
    const float *snapshot_y,
    const float *snapshot_z,
    const unsigned int *group_atoms,
    const unsigned int *group_atom_offset,
    const unsigned int *group_atom_count,
    const unsigned int *group_constraint_offset,
    const unsigned int *group_constraint_count,
    const unsigned char *group_constraints_local_i,
    const unsigned char *group_constraints_local_j,
    const float *group_constraints_r2,
    const float *atom_inv_mass,
    float lx, float ly, float lz, float xy, float xz, float yz,
    float dt,
    float *constraint_virial,
    unsigned int n_groups);

extern "C" __global__ void rattle_velocities(
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    float *velocities_x,
    float *velocities_y,
    float *velocities_z,
    const unsigned int *group_atoms,
    const unsigned int *group_atom_offset,
    const unsigned int *group_atom_count,
    const unsigned int *group_constraint_offset,
    const unsigned int *group_constraint_count,
    const unsigned char *group_constraints_local_i,
    const unsigned char *group_constraints_local_j,
    const float *atom_inv_mass,
    float lx, float ly, float lz, float xy, float xz, float yz,
    float dt,
    float *constraint_virial,
    unsigned int n_groups);

extern "C" __global__ void constraint_virial_scatter(
    const float *constraint_virial,
    const unsigned int *group_atoms,
    float *particle_virials,
    unsigned int n_atom_slots);

extern "C" __global__ void shake_positions_no_velocity(
    float *positions_x,
    float *positions_y,
    float *positions_z,
    const unsigned int *group_atoms,
    const unsigned int *group_atom_offset,
    const unsigned int *group_atom_count,
    const unsigned int *group_constraint_offset,
    const unsigned int *group_constraint_count,
    const unsigned char *group_constraints_local_i,
    const unsigned char *group_constraints_local_j,
    const float *group_constraints_r2,
    const float *atom_inv_mass,
    float lx, float ly, float lz, float xy, float xz, float yz,
    unsigned int n_groups);
```

Each kernel runs one thread per group: `g = blockIdx.x * blockDim.x +
threadIdx.x`. Threads past `n_groups` return without touching any
buffer. Block size 256; grid size `ceil(n_groups / 256)`. Shared
memory: zero bytes for all five kernels. Stream: the default stream
carried by `device`.

Per-thread storage holds up to `MAX_GROUP_ATOMS = 8` position triples
(plus snapshot triples, plus per-atom inverse mass) and
`MAX_GROUP_CONSTRAINTS = 12` constraint records. The thread reads its
group's atom count from `group_atom_count[g]` and constraint count
from `group_constraint_count[g]` and loops within those bounds. The
caps are checked at slot construction time, so kernels assume both
counts are within bounds.

### PTX Module Loading <!-- rq-11c3339a -->

`init_device()` loads the compiled `kernels/shake.cu` PTX as module
`"shake"` and captures all five kernel functions into the `Kernels`
handle (see `build-pipeline.md`).

### Builder <!-- rq-a0ddb391 -->

- `ShakeConstraintBuilder` — implements `ConstraintBuilder` (see <!-- rq-c623013e -->
  `integration/constraint-framework.md`). Methods:
  - `kind_name() -> &'static str` returns `"shake"`.
  - `expected_atom_count(&params: &toml::Value) -> usize`
    deserialises the `params` table as `ShakeParams` and returns
    `params.atoms as usize`. On a deserialisation failure or an
    out-of-range `atoms`, returns `0` (the topology parser then
    surfaces an `InvalidConstraintRow` against the `[constraints]`
    row that referenced the malformed type; the builder's `build`
    later surfaces the same condition as a `MalformedShakeType`).
  - `validate_group_shape(&self, ...) -> Result<(), ConstraintError>`
    re-verifies that the group's atom count matches the type's
    declared `atoms` and that the group's atom count and constraint
    count both fit within `MAX_GROUP_ATOMS / MAX_GROUP_CONSTRAINTS`.
    On failure, returns `ConstraintError::InvalidGroupShape { ... }`
    with the kind `"shake"`.
  - `build(&self, cx: &ConstraintBuildContext<'_>) -> Result<Box<dyn Constraint>, ConstraintError>`
    constructs a `ShakeConstraintsState` from `cx.list`,
    `cx.particle_masses`, and `cx.constraint_types`, wraps it in a
    `Box`, and returns it. Forwards any `ShakeError` through
    `From<ShakeError> for ConstraintError`.

  `ConstraintRegistry::with_builtins()` registers this builder under
  `"shake"`.

## Empty State <!-- rq-4656e089 -->

When the slot has zero groups, every hook returns `Ok(())` without
launching any kernel and without modifying any buffer. The
constraint-virial buffer has length zero. The runner's barostat
machinery reads `particle_virials` regardless and sees the
constraint slot's zero contribution naturally.

## Out of Scope <!-- rq-a48d5bbe -->

- Multi-row constraint groups (overlapping connected components). The
  topology parser still requires one `[constraints]` row per
  connected component; a future feature lifts this restriction via
  connected-component graph construction, at which point the SoA
  layout described here accommodates it without further kernel
  changes.
- The analytical SETTLE algorithm (Miyamoto-Kollman 1992) for
  three-atom rigid water. Lives in a separate feature with kind
  `"settle"`. SHAKE remains the general fallback.
- M-SHAKE for arbitrarily large rigid groups beyond the
  `MAX_GROUP_ATOMS / MAX_GROUP_CONSTRAINTS` caps. Future feature.
- LINCS (Linear Constraint Solver) and other non-iterative
  constraint algorithms.
- Flexible / bonded constraints (treated by the bonded-force kernel,
  not by `Constraint`).
- Constraint forces resolved into per-atom Cartesian components for
  visualisation; the only constraint-quantity consumed downstream is
  the scalar `constraint_virial` per group.
- Performance specialisation for very small groups (e.g. a separate
  3-atom code path). The general kernel handles 3-atom rigid water
  at the same per-group cost as the previous water-specific kernel
  to within the iteration-count noise.

---

## Gherkin Scenarios <!-- rq-5811603e -->

```gherkin
Feature: SHAKE + RATTLE rigid constraints

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called

  # --- Slot construction and parameter validation ---

  @rq-64700eb0
  Scenario: ShakeConstraintsState::new with a single 3-atom water group succeeds
    Given a ConstraintList with one group of 3 atoms and 3 constraints
      (local pairs (0,1), (0,2), (1,2) with target distances r_oh, r_oh, r_hh)
    And a [[constraint_types]] entry with name "SPCE", kind "shake", atoms 3,
      and a constraints list containing the three pairs above
    And per-atom masses (m_O, m_H, m_H) for the group's three atoms
    When ShakeConstraintsState::new is called
    Then it returns Ok(state)
    And state.group_count is 1
    And state.group_atoms on the device equals [atom_O, atom_H1, atom_H2]

  @rq-79c091e0
  Scenario: ShakeConstraintsState::new with empty constraint list succeeds and is a no-op
    Given a ConstraintList with zero groups
    When ShakeConstraintsState::new is called
    Then it returns Ok(state)
    And state.group_count is 0
    And every subsequent hook invocation launches no kernels and returns Ok(())

  @rq-c70532a9
  Scenario: ShakeConstraintsState::new with a 5-atom methane group succeeds
    Given a ConstraintList with one group of 5 atoms (C, H1, H2, H3, H4) and 4
      constraints (C-H1, C-H2, C-H3, C-H4) at the methane C-H bond length
    And a [[constraint_types]] entry with name "methane", kind "shake",
      atoms 5, and the four C-H constraints
    When ShakeConstraintsState::new is called
    Then it returns Ok(state)
    And state.group_count is 1
    And state.group_atom_count on the device equals [5]
    And state.group_constraint_count on the device equals [4]

  @rq-e6e7d6e2
  Scenario: ShakeConstraintsState::new rejects atoms > MAX_GROUP_ATOMS
    Given a ConstraintList with one group of 10 atoms and 9 constraints
    And a [[constraint_types]] entry with name "BIG", kind "shake",
      atoms 10, and 9 constraints
    When ShakeConstraintsState::new is called
    Then it returns Err(ShakeError::UnsupportedGroupSize { group_index: 0, atoms: 10, constraints: 9 })

  @rq-9bd207b2
  Scenario: ShakeConstraintsState::new rejects constraints > MAX_GROUP_CONSTRAINTS
    Given a ConstraintList with one group whose constraint type declares
      atoms = 6 and 14 pair-distance constraints
    When ShakeConstraintsState::new is called
    Then it returns Err(ShakeError::UnsupportedGroupSize { group_index: 0, atoms: 6, constraints: 14 })

  @rq-659836c5
  Scenario: ShakeConstraintsState::new rejects a constraint type with atoms == 0
    Given a [[constraint_types]] entry with name "bad", kind "shake",
      atoms 0, and a non-empty constraints list
    When ShakeConstraintsState::new is called
    Then it returns Err(ShakeError::MalformedShakeType { name: "bad", reason: r })
      where r mentions `atoms must be strictly positive`

  @rq-09f2d9c2
  Scenario: ShakeConstraintsState::new rejects a constraint with i == j
    Given a [[constraint_types]] entry whose constraints list contains
      an entry with { i: 1, j: 1, d: 1.0e-10 }
    When ShakeConstraintsState::new is called
    Then it returns Err(ShakeError::MalformedShakeType { name: _, reason: r })
      where r mentions `constraint atoms must differ`

  @rq-a8971153
  Scenario: ShakeConstraintsState::new rejects a duplicate constraint pair
    Given a [[constraint_types]] entry whose constraints list contains
      both { i: 0, j: 1, d: 1.0e-10 } and { i: 1, j: 0, d: 1.2e-10 }
    When ShakeConstraintsState::new is called
    Then it returns Err(ShakeError::MalformedShakeType { name: _, reason: r })
      where r mentions `duplicate constraint pair`

  @rq-5be2064b
  Scenario: ShakeConstraintsState::new rejects a non-positive target distance
    Given a [[constraint_types]] entry whose constraints list contains
      an entry with { i: 0, j: 1, d: 0.0 }
    When ShakeConstraintsState::new is called
    Then it returns Err(ShakeError::MalformedShakeType { name: _, reason: r })
      where r mentions `target distance must be strictly positive`

  # --- Position projection (SHAKE) ---

  @rq-0f5c9f99
  Scenario: shake_positions restores constraint distances after a small uniform translation
    Given a constructed ShakeConstraintsState with one SPC/E water group at equilibrium
    And the unconstrained post-drift positions are the equilibrium positions shifted
      uniformly by 1.0e-3 nm along x
    When apply_after_drift is called with dt = 2.0e-15 s
    Then for each constraint (i, j, r) the post-call |r_i - r_j| equals r to within 1.0e-13 m
    And the centre of mass of the three atoms equals the unconstrained centre of mass to
      within f32 round-off

  @rq-7c13040a
  Scenario: shake_positions restores constraint distances after a small per-atom kick
    Given a constructed ShakeConstraintsState with one SPC/E water group at equilibrium
    And the unconstrained post-drift positions perturb each atom independently by ~1.0e-12 m
    When apply_after_drift is called with dt = 2.0e-15 s
    Then every constraint distance is within 1.0e-13 m of its target
    And the SHAKE iteration converged in fewer than 8 sweeps

  @rq-5d18fa01
  Scenario: shake_positions updates half-step velocities consistently with the position correction
    Given a constructed ShakeConstraintsState with one SPC/E water group
    And initial velocities v_i^pre and unconstrained post-drift positions r_i^unconstrained
    When apply_after_drift is called with dt = 2.0e-15 s
    Then the post-call velocity for every atom equals v_i^pre + (r_i^constrained − r_i^unconstrained) / dt
      to within f32 round-off

  @rq-757a5bad
  Scenario: shake_positions writes a non-zero position-level constraint virial
    Given a constructed ShakeConstraintsState with one SPC/E water group whose
      unconstrained post-drift positions break every constraint by ~1.0e-12 m
    When apply_after_drift is called with dt = 2.0e-15 s
    Then constraint_virial on the device contains three nonzero entries for this group

  @rq-7a0a23e3
  Scenario: shake_positions handles a water group straddling a periodic boundary
    Given a constructed ShakeConstraintsState with one SPC/E water group
    And a small orthorhombic simulation box (Lx = Ly = Lz = 10.0 a₀)
    And pre-drift positions placing the O atom near +Lx/2 and the two H atoms near −Lx/2,
      so that the molecule straddles the +x periodic boundary
    And unconstrained post-drift positions that perturb the O–H1 bond by ~1.0e-2 a₀
      along the O→H1 direction (computed under minimum-image)
    When apply_after_drift is called with dt = 1.0 atu
    Then every constraint distance (O–H1, O–H2, H1–H2), computed under minimum-image,
      equals its target r₀ to within 1.0e-4 a₀ relative
    And the per-atom global positions remain in the same lattice image they were in
      before the call (no spurious wrap of any atom)
    And the mass-weighted centre of mass, computed by bringing every atom into atom 0's
      image, equals the same COM of the unconstrained post-drift positions to within
      1.0e-3 a₀
    And the per-group sum Σ_i constraint_virial[i] is finite

  # --- Velocity projection (RATTLE) ---

  @rq-17b28c63
  Scenario: rattle_velocities zeroes the constraint-distance time-derivative
    Given a constructed ShakeConstraintsState with one SPC/E water group at equilibrium
    And post-kick velocities with non-trivial v_rel · d for every constraint
    When apply_after_kick is called with dt = 2.0e-15 s
    Then for each constraint (i, j) the post-call (v_i − v_j) · (r_i − r_j) is within
      1.0e-20 m²/s of zero
    And the RATTLE iteration converged in fewer than 8 sweeps

  @rq-7e084b5e
  Scenario: rattle_velocities preserves the centre-of-mass velocity
    Given a constructed ShakeConstraintsState with one SPC/E water group
    And the post-kick velocities have a known mass-weighted COM velocity v_COM
    When apply_after_kick is called with dt = 2.0e-15 s
    Then the post-call mass-weighted COM velocity equals v_COM byte-identically (no
      mass-weighted force was applied to the COM)

  @rq-3b6f4dec
  Scenario: rattle_velocities accumulates a velocity-level constraint virial when dt > 0
    Given a constructed ShakeConstraintsState with one SPC/E water group
    And constraint_virial entries containing the position-level half (a known
      non-zero pattern from a prior shake_positions call)
    When apply_after_kick is called with dt = 2.0e-15 s
    Then constraint_virial after the call equals the position-level half plus the
      velocity-level half, with the velocity-level half computed as
      m_i · Δv_i · r_i^COM / dt

  @rq-3aef0b06
  Scenario: rattle_velocities skips the velocity-level virial accumulation when dt <= 0
    Given a constructed ShakeConstraintsState with one SPC/E water group
    And a stale constraint_virial pattern X on the device
    When apply_after_kick is called with dt = 0.0
    Then constraint_virial after the call is byte-identical to X

  # --- Virial scatter ---

  @rq-8471c200
  Scenario: constraint_virial_scatter additively writes per-atom virial into particle_virials
    Given a constructed ShakeConstraintsState with one SPC/E water group
    And constraint_virial contains [w0, w1, w2] for the three atoms of the group
    And particle_virials on the device is initialised to [0; N]
    When constraint_virial_scatter is launched
    Then particle_virials[atom_O] equals w0
    And particle_virials[atom_H1] equals w1
    And particle_virials[atom_H2] equals w2
    And every other particle_virials[a] is unchanged

  @rq-513b4dbe
  Scenario: constraint_virial_scatter handles two disjoint groups
    Given two SPC/E water groups with disjoint atom sets
    And constraint_virial contains six entries [w0_g0, w1_g0, w2_g0, w0_g1, w1_g1, w2_g1]
    When constraint_virial_scatter is launched
    Then particle_virials accumulates each group's three contributions into the
      corresponding three atom slots, with no cross-group interference

  # --- Position-only projection (minimization) ---

  @rq-13f424a1
  Scenario: shake_positions_no_velocity restores constraint distances from off-manifold positions
    Given a constructed ShakeConstraintsState with one SPC/E water group whose
      current positions break every constraint by ~5.0e-12 m
    When apply_position_projection_only is called
    Then every constraint distance is within 1.0e-13 m of its target
    And velocities on the device are unchanged byte-for-byte from before the call
    And constraint_virial on the device is unchanged byte-for-byte from before the call

  # --- Reproducibility ---

  @rq-aa5ac09f
  Scenario: Two independent runs of apply_after_drift produce byte-identical state
    Given two ShakeConstraintsState instances A and B with identical inputs and
      identical pre-drift snapshots
    And identical unconstrained post-drift positions in both
    When apply_after_drift is called on each with dt = 2.0e-15 s
    Then run A and run B agree byte-for-byte on positions_x, positions_y, positions_z,
      velocities_x, velocities_y, velocities_z, and constraint_virial

  @rq-c7fc10c5
  Scenario: Two independent runs of apply_after_kick produce byte-identical state
    Given two ShakeConstraintsState instances A and B with identical inputs
    And identical post-kick velocities and identical constrained positions in both
    When apply_after_kick is called on each with dt = 2.0e-15 s
    Then run A and run B agree byte-for-byte on velocities_x, velocities_y, velocities_z,
      and constraint_virial

  # --- Composition with the integrator framework ---

  @rq-ff235ded
  Scenario: A full velocity-Verlet timestep with one rigid SPC/E group preserves all three
    constraint distances
    Given a runner with a velocity-verlet integrator and a ShakeConstraintsState slot
      containing one SPC/E water group at equilibrium with thermal velocities
    When the runner runs N = 100 timesteps at dt = 2.0e-15 s
    Then for every step n in 0..N the post-step constraint distances are within
      1.0e-13 m of their targets

  @rq-fc27df14
  Scenario: A SHAKE-only run with no Constraint slot active leaves constraint distances drifting
    Given a runner with a velocity-verlet integrator and NO constraint slot
    And a SPC/E water-shaped topology
    When the runner runs N = 100 timesteps at dt = 2.0e-15 s
    Then the constraint distances drift away from their targets (this scenario
      pins the contract that the constraint slot is what enforces rigidity)

  # --- Multi-atom-type group: methane ---

  @rq-1d06fac7
  Scenario: A 5-atom methane group's C-H constraints remain within tolerance after 100 steps
    Given a runner with a velocity-verlet integrator and a ShakeConstraintsState slot
      containing one methane group (1 C, 4 H) with target r_CH = 1.09e-10 m
    When the runner runs N = 100 timesteps at dt = 1.0e-15 s
    Then every C-H constraint distance is within 1.0e-13 m of 1.09e-10 m after each step

  # --- Group-size cap ---

  @rq-2f9b0e01
  Scenario: A group whose declared atoms field exceeds MAX_GROUP_ATOMS is rejected at config load
    Given a [[constraint_types]] entry with name "TooBig", kind "shake", atoms 9
    When config validation runs
    Then it returns ConfigError pointing at the entry "TooBig" with a reason mentioning
      `atoms <= 8`

  @rq-ad2f7a73
  Scenario: A group whose constraints list exceeds MAX_GROUP_CONSTRAINTS is rejected at config load
    Given a [[constraint_types]] entry with kind "shake", atoms 6, and 13 constraints
    When config validation runs
    Then it returns ConfigError pointing at the entry with a reason mentioning
      `constraints <= 12`

  # --- Determinism of the SHAKE iteration order ---

  @rq-cfb1e3aa
  Scenario: The constraint-iteration order matches the topology's declared order
    Given a ShakeConstraintsState whose group's constraints were declared as
      [(0,1,d_a), (1,2,d_b), (0,2,d_c)] in that order
    When apply_after_drift is called
    Then the in-kernel sweep visits (0,1,d_a) first, (1,2,d_b) second, (0,2,d_c) third,
      and the post-call positions match the iteration-order-dependent SHAKE result
    And the same call with the constraints declared in a different order produces
      a deterministically different (but still constraint-satisfying) result
```
