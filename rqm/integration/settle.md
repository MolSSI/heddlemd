# Feature: SETTLE Analytic Three-Atom Rigid-Water Constraint <!-- rq-67e62f4b -->

SETTLE is the constraint algorithm of Miyamoto & Kollman (*J. Comput.
Chem.* **13**, 952 (1992)) for projecting a three-atom rigid water
molecule's positions and velocities back onto its constraint manifold
in closed form, with no iteration. It is the v1 implementation of the
pluggable `Constraint` slot defined in `constraint-framework.md`,
selected by `kind = "settle-water"` on a `[[constraint_types]]` entry
in the TOML config.

SETTLE consumes constraint groups whose `constraint_type_kind` is
`SettleWater`: exactly three atoms with exactly three constraints
forming the rigid triangle `(O, H1, H2)`. The two O–H distances must
be equal (parameter `r_oh`) and the H–H distance is the parameter
`r_hh`. The atom listed first in each `[constraints]` row is the
oxygen; the next two are the hydrogens, in either order.

The implementation provides two CUDA kernels — `settle_positions` and
`settle_velocities` — and a single host-side slot
(`SettleConstraintsState`) that owns the device-side per-group buffers
and implements the `Constraint` trait. The slot's `apply_before_drift`
hook snapshots pre-drift positions; `apply_after_drift` runs
`settle_positions` to project the post-drift positions back onto the
manifold and to correct the half-step velocities; `apply_after_kick`
runs `settle_velocities` to project the final velocities onto the
manifold.

## Algorithm <!-- rq-ce77d9fb -->

For one water group `g` with atoms `(O, H1, H2)`, masses
`(m_O, m_H, m_H)`, pre-drift positions `(r_O0, r_H10, r_H20)`, and
unconstrained post-drift positions `(r_O', r_H1', r_H2')`, the
`settle_positions` step computes constrained post-drift positions
`(r_O, r_H1, r_H2)` and updates the half-step velocities `(v_O, v_H1,
v_H2)` to reflect the position correction.

The algorithm is the closed-form projection from Miyamoto & Kollman
1992; the present description is a per-thread restatement of it for
the per-group dispatch this slot uses.

1. **Centre-of-mass invariance.** Compute the centre of mass of the
   unconstrained post-drift positions:
   `r_C = (m_O r_O' + m_H r_H1' + m_H r_H2') / M`
   where `M = m_O + 2 m_H`. The COM motion is unaffected by the
   constraint (constraint forces are internal); the constrained
   positions share this COM exactly.
2. **Build the body frame on the pre-drift positions.** Translate the
   pre-drift positions to their own COM
   `r_C0 = (m_O r_O0 + m_H r_H10 + m_H r_H20) / M`, giving
   `a0 = r_O0 - r_C0`, `b0 = r_H10 - r_C0`, `c0 = r_H20 - r_C0`. Build
   an orthonormal frame `(X0, Y0, Z0)` from these three vectors: `Z0`
   is normal to the molecular plane (proportional to
   `(b0 - a0) × (c0 - a0)`); `X0` lies along the symmetry axis of the
   canonical molecule, in the molecular plane and pointing from the
   H–H midpoint toward O; `Y0` completes the right-handed frame.
3. **Build the trial frame on the unconstrained post-drift COM-relative
   positions.** Translate to the new COM:
   `a1 = r_O' - r_C`, `b1 = r_H1' - r_C`, `c1 = r_H2' - r_C`. Express
   them in the body frame `(X0, Y0, Z0)`:
   `a1' = (a1 · X0, a1 · Y0, a1 · Z0)`, and similarly for `b1'`,
   `c1'`.
4. **Solve for the rotation.** SETTLE solves three quadratic equations
   (two for the rotation about `Z0` aligning the H–H midpoint to the
   new H–H midpoint, plus the constraint-distance condition) in
   closed form, giving two trigonometric quantities
   `(sin φ, cos φ)` and `(sin ψ, cos ψ)` plus a third small rotation
   `(sin θ, cos θ)` for the out-of-plane tilt. The full derivation
   appears in Miyamoto & Kollman 1992 §III.A; the implementation
   follows that derivation step by step using single-precision
   arithmetic throughout. No square root of a negative number is
   reachable from physical inputs; should one occur (a degenerate
   drift configuration), the kernel clamps the radicand to zero,
   which yields the constrained positions on the boundary of the
   feasible manifold.
5. **Reconstruct constrained positions.** Apply the rotation to the
   canonical body-frame positions of the rigid triangle and translate
   back to `r_C`. The canonical body-frame positions are computed
   once per constraint type at slot construction from `r_oh` and
   `r_hh`:
   - `O_body = (+r_oh cos(θ/2 - π/2), 0, 0)` along the symmetry axis,
     with `θ = H–O–H angle = 2 asin((r_hh / 2) / r_oh)` (a derived
     quantity).
   - `H1_body = (-r_oh sin(θ/2 - π/2) - r_oh cos(α), -r_hh/2, 0)`.
   - `H2_body = (-r_oh sin(θ/2 - π/2) - r_oh cos(α), +r_hh/2, 0)`.
   - The expressions above are illustrative; the implementation uses
     the equivalent COM-anchored form so the canonical body-frame
     positions sum (mass-weighted) to zero exactly.
6. **Update half-step velocities for position consistency.** For each
   atom `i ∈ {O, H1, H2}` in the group:
   `v_i ← v_i + (r_i_constrained - r_i_unconstrained) / dt`. This is
   the classical SHAKE velocity correction. It preserves the
   integrator's leapfrog half-step semantics: the corrected
   `v(t + dt/2)` is consistent with the constrained
   `x(t + dt)` having been reached from `x(t)` by free streaming.

The `settle_velocities` step projects post-second-kick velocities
`(v_O', v_H1', v_H2')` onto the constraint manifold of the
constrained positions `(r_O, r_H1, r_H2)`. For three constraints on a
rigid triangle, the projection is the solution of a 3×3 linear system
in three Lagrange multipliers `(λ_OH1, λ_OH2, λ_HH)`. The system is
constructed from the constraint Jacobian rows
`∇_i C_k = (r_i - r_j) for constraint k coupling atoms (i, j)`, and
solved in closed form because the 3×3 matrix has a known structure
for a rigid water (every off-diagonal entry is half the dot product
of two intra-group displacement vectors). The corrected velocities
satisfy `(v_i - v_j) · (r_i - r_j) = 0` for every constraint `(i, j)`.

The algorithm has no iteration. Every per-group computation completes
in a fixed number of arithmetic operations.

## Per-Step Kernel Sequence <!-- rq-de7601cd -->

| Order | Hook                 | Kernel              | Operation                                                                          | Stage label              |
| ----- | -------------------- | ------------------- | ---------------------------------------------------------------------------------- | ------------------------ |
| 1     | `apply_before_drift` | `settle_snapshot`   | copy pre-drift positions of every group's atoms into the slot's snapshot buffer    | `SettleSnapshot`         |
| 2     | `apply_after_drift`  | `settle_positions`  | per-group analytic projection of positions; per-group half-step velocity correction | `SettlePositions`        |
| 3     | `apply_after_kick`   | `settle_velocities` | per-group analytic projection of final velocities                                  | `SettleVelocities`       |

All three kernels run with one thread per constraint group. Block
size is 256; grid size is `ceil(group_count / 256)`. Shared memory is
zero bytes. The stream is the default stream carried by
`ParticleBuffers::device`.

## Reproducibility <!-- rq-52412463 -->

Per-group order is fixed at slot construction (the
`ConstraintList`'s `groups` are already sorted by minimum particle
index; the slot uploads them in that order). Each thread reads and
writes only its own group's slots in the device buffers, so there are
no atomics and no race conditions. Within a group, the per-thread
arithmetic is a fixed straight-line sequence of `f32` operations.
Two runs on the same GPU with identical inputs produce byte-identical
outputs.

## Parameters <!-- rq-f0d44c8f -->

Each `[[constraint_types]]` entry in the config with `kind =
"settle-water"` contributes one row to the per-type parameter table
uploaded to the device. Each row carries:

- `r_oh: f64` — O–H constraint distance, metres. Required. Finite
  and strictly positive.
- `r_hh: f64` — H–H constraint distance, metres. Required. Finite
  and strictly positive. Must satisfy `r_hh < 2 · r_oh` (geometric
  feasibility of the triangle); the config loader rejects out-of-range
  values.

The host-side slot stores `r_oh`, `r_hh`, the derived H–O–H angle
`theta = 2 · asin((r_hh / 2) / r_oh)` (radians), the canonical
body-frame positions for the three atoms (nine `f32` values per type),
and the per-atom masses (three `f32` values per type, drawn from
`config.particle_types` via the atoms' `type_indices`). At kernel
launch the slot passes pointers to these per-type tables; each thread
reads the row indexed by its group's `constraint_type_index`.

Mass values for the three atoms of every group must be identical
across groups that share a constraint type: the SETTLE algorithm
assumes a single oxygen mass `m_O` and a single hydrogen mass `m_H`
per constraint type. The slot constructor reads
`config.particle_types[t].mass` for every distinct `t` that appears
among any group's atoms and verifies the (O, H, H) mass pattern is
consistent per constraint type; mismatch is rejected with
`SettleError::InconsistentMasses { constraint_type_index, expected, actual }`.

## Feature API <!-- rq-2941eebb -->

### Types <!-- rq-2220db6a -->

- `SettleConstraintsState` — implements the `Constraint` trait declared <!-- rq-dd7065b9 -->
  in `constraint-framework.md`. Fields:
  - `device: Arc<CudaDevice>`
  - `group_count: usize`
  - `particle_count: usize`
  - `group_atoms: CudaSlice<u32>` — flat array of `[atom_O, atom_H1,
    atom_H2]` triples, length `3 * group_count`, in `groups` order.
  - `group_type_index: CudaSlice<u32>` — length `group_count`. Maps
    each group to its row in the per-type parameter tables.
  - `type_r_oh: CudaSlice<f32>` — length `n_settle_types`.
  - `type_r_hh: CudaSlice<f32>` — length `n_settle_types`.
  - `type_canonical_x: CudaSlice<f32>` — length `3 * n_settle_types`,
    body-frame x components of `[O_body, H1_body, H2_body]`.
  - `type_canonical_y: CudaSlice<f32>` — length `3 * n_settle_types`.
  - `type_canonical_z: CudaSlice<f32>` — length `3 * n_settle_types`.
  - `type_mass_o: CudaSlice<f32>` — length `n_settle_types`.
  - `type_mass_h: CudaSlice<f32>` — length `n_settle_types`.
  - `snapshot_x: CudaSlice<f32>` — length `3 * group_count`,
    pre-drift positions of every group's atoms (refreshed each step
    by `settle_snapshot`).
  - `snapshot_y: CudaSlice<f32>` — length `3 * group_count`.
  - `snapshot_z: CudaSlice<f32>` — length `3 * group_count`.

  All fields are private; the slot's public surface is the
  `Constraint` trait methods.

  Constructor:

  - `SettleConstraintsState::new(device: Arc<CudaDevice>, list: &ConstraintList, particle_count: usize, masses: &[f32], constraint_types: &[NamedSlotConfig]) -> Result<SettleConstraintsState, SettleError>`
    - Filters `constraint_types` to entries with
      `kind == "settle-water"`. For each, deserialises
      `SettleWaterParams { r_oh, r_hh }` from the entry's `params`
      field and computes the derived H–O–H angle and the canonical
      body-frame positions.
    - Iterates `list.groups`. For each group whose constraint type
      resolves (via
      `constraint_types[group.constraint_type_index].kind`) to
      `"settle-water"`, packs the three atom indices in the order
      declared by the original `[constraints]` row (oxygen first)
      into `group_atoms`.
    - Reads `masses[atom_O]`, `masses[atom_H1]`, `masses[atom_H2]`
      for each group and verifies they match the constraint type's
      expected `(m_O, m_H, m_H)` pattern. Returns
      `SettleError::InconsistentMasses { .. }` on mismatch.
    - Uploads all device buffers.
    - When `list.is_empty()` or the list contains no `settle-water`
      groups, allocates zero-length device buffers and returns a
      slot whose hooks are all no-ops.

- `SettleError` — error type returned by `SettleConstraintsState::new`. <!-- rq-16b29d27 -->
  Wraps `GpuError` via `Gpu(GpuError)` and `TimingsError` via
  `Timings(TimingsError)`. Algorithm-specific variants:
  - `InconsistentMasses { constraint_type_index: usize, expected: (f32, f32, f32), actual: (f32, f32, f32) }`
    — the three atoms in a group do not match the constraint type's
    expected mass pattern.
  - `MalformedSettleType { name: String, reason: String }` — the
    config's `[[constraint_types]]` entry has `kind = "settle-water"`
    but its `r_oh` / `r_hh` fields violate the geometric feasibility
    condition `r_hh < 2 · r_oh`.

  Converted into `ConstraintError` via a `From` impl so the trait
  surface stays unchanged.

### CUDA Kernels <!-- rq-744ae79d -->

`kernels/settle.cu` declares three `extern "C"` kernels:

```c
extern "C" __global__ void settle_snapshot(
    const float *positions_x, const float *positions_y, const float *positions_z,
    const unsigned int *group_atoms,
    float *snapshot_x, float *snapshot_y, float *snapshot_z,
    unsigned int n_groups);

extern "C" __global__ void settle_positions(
    float *positions_x, float *positions_y, float *positions_z,
    float *velocities_x, float *velocities_y, float *velocities_z,
    const float *snapshot_x, const float *snapshot_y, const float *snapshot_z,
    const unsigned int *group_atoms,
    const unsigned int *group_type_index,
    const float *type_canonical_x, const float *type_canonical_y, const float *type_canonical_z,
    const float *type_mass_o, const float *type_mass_h,
    float lx, float ly, float lz, float xy, float xz, float yz,
    float dt,
    unsigned int n_groups);

extern "C" __global__ void settle_velocities(
    const float *positions_x, const float *positions_y, const float *positions_z,
    float *velocities_x, float *velocities_y, float *velocities_z,
    const unsigned int *group_atoms,
    const unsigned int *group_type_index,
    const float *type_mass_o, const float *type_mass_h,
    float lx, float ly, float lz, float xy, float xz, float yz,
    unsigned int n_groups);
```

Each thread computes its global group index as
`blockIdx.x * blockDim.x + threadIdx.x`. If the index is `>= n_groups`
the thread returns without touching any buffer.

#### `settle_snapshot` <!-- rq-68847ade -->

For group `g`, the thread reads the three atom indices `(a_O, a_H1,
a_H2)` from `group_atoms[3*g .. 3*g+3]`, then copies the three
particles' `(positions_x, positions_y, positions_z)` into
`snapshot_{x,y,z}[3*g .. 3*g+3]`. The snapshot buffer is overwritten
each step; no previous-step state is retained.

#### `settle_positions` <!-- rq-db2321aa -->

For group `g`:

1. Reads atom indices `(a_O, a_H1, a_H2)` from
   `group_atoms[3*g .. 3*g+3]` and type index `t = group_type_index[g]`.
2. Reads pre-drift positions from
   `snapshot_{x,y,z}[3*g .. 3*g+3]` and unconstrained post-drift
   positions from `positions_{x,y,z}[a_*]`.
3. Reads half-step velocities from `velocities_{x,y,z}[a_*]`.
4. Reads canonical body-frame positions from
   `type_canonical_{x,y,z}[3*t .. 3*t+3]`.
5. Reads masses `m_O = type_mass_o[t]`, `m_H = type_mass_h[t]`.
6. Computes minimum-image displacements between every pair of atoms
   in both the snapshot and the unconstrained sets using the
   triclinic tilt-subtraction algorithm defined in `simulation-box.md`
   (parameters `lx, ly, lz, xy, xz, yz`). Periodic-image fix-up is
   applied to the unconstrained positions so all three atoms of the
   group lie in the same image: the algorithm wraps `a_H1` and `a_H2`
   relative to `a_O` before forming the body frame, then unwraps the
   final constrained positions back to the primary image after the
   projection.
7. Executes the closed-form algorithm in §III.A of Miyamoto & Kollman
   1992 to produce constrained positions in the same frame as the
   unconstrained positions.
8. Writes constrained positions back to `positions_{x,y,z}[a_*]`.
9. Writes corrected half-step velocities back to
   `velocities_{x,y,z}[a_*]` using `v_i ← v_i + (r_constrained - r_unconstrained) / dt`.

The kernel writes only the nine `f32` slots `positions_{x,y,z}[a_O]`,
`positions_{x,y,z}[a_H1]`, `positions_{x,y,z}[a_H2]` and the nine
`f32` slots `velocities_{x,y,z}[a_O]`, `velocities_{x,y,z}[a_H1]`,
`velocities_{x,y,z}[a_H2]`. Every other particle's state is untouched.

The kernel does not advance `images_{x,y,z}` for the corrected
atoms. SETTLE's position projection produces displacements at most
`O(dt · v_thermal)` in magnitude — well below a half-image of any
non-pathological simulation box — so the corrected position remains
in the same image as the unconstrained position. The next
`vv_kick_drift` will perform the canonical wrap-and-image-count
update.

#### `settle_velocities` <!-- rq-6c9357db -->

For group `g`:

1. Reads atom indices and type index as above.
2. Reads constrained positions from `positions_{x,y,z}[a_*]` (already
   on the constraint manifold after `settle_positions`).
3. Reads post-kick velocities from `velocities_{x,y,z}[a_*]`.
4. Reads masses `m_O = type_mass_o[t]`, `m_H = type_mass_h[t]`.
5. Computes minimum-image displacements `r_OH1`, `r_OH2`, `r_H1H2`
   between the three atoms.
6. Computes relative velocities along the three constraint directions.
7. Solves the 3×3 linear system for the three Lagrange multipliers
   `(λ_OH1, λ_OH2, λ_HH)` in closed form (the matrix entries are
   inner products of the three constraint-direction vectors weighted
   by inverse masses).
8. Updates velocities:
   `v_O ← v_O - (λ_OH1 · r_OH1 + λ_OH2 · r_OH2) / m_O`,
   `v_H1 ← v_H1 + (λ_OH1 · r_OH1 - λ_HH · r_H1H2) / m_H`,
   `v_H2 ← v_H2 + (λ_OH2 · r_OH2 + λ_HH · r_H1H2) / m_H`.

The kernel writes only the nine `f32` velocity slots of the group.
Positions are not modified.

### PTX Module Loading <!-- rq-ba4d55e6 -->

`init_device()` loads the compiled `kernels/settle.cu` PTX as module
`"settle"` and captures its `settle_snapshot`, `settle_positions`, and
`settle_velocities` functions into the `Kernels` handle (see
`build-pipeline.md`).

### Builder <!-- rq-a0f7c746 -->

- `SettleBuilder` — implements `ConstraintBuilder` with <!-- rq-278cb574 -->
  `kind_name() == "settle-water"`. The `kind_name` matches the user's
  `kind = "settle-water"` field in a `[[constraint_types]]` entry.
  - `validate_params(&params)` deserialises a
    `SettleWaterParams { r_oh: f64, r_hh: f64 }` from `params`,
    requires both to be finite and strictly positive, and requires
    `r_hh < 2 · r_oh` (the rigid-triangle feasibility condition).
    Returns `ConfigError::SettleGeometryInfeasible { name, r_oh,
    r_hh }` for the feasibility violation and
    `ConfigError::InvalidValue { field, reason }` for the per-field
    finiteness / positivity check failures. Unknown fields under
    `[[constraint_types]]` (e.g. a stray `theta_0`) surface as
    `ConfigError::Parse { path, message }`.
  - `expected_atom_count(&params)` returns `3` regardless of the
    parameter values. Used by the topology parser to size-check
    `[constraints]` rows that reference a `settle-water` type.
  - `validate_group_shape(group_index, atoms, constraints, params,
    masses)` verifies the cluster shape required by the SETTLE
    algorithm:
    - `atoms.len() == 3`.
    - `constraints.len() == 3` with local pairs `(0, 1)`, `(0, 2)`,
      `(1, 2)`.
    - The two H masses `masses[atoms[1]]` and `masses[atoms[2]]`
      agree to within a tight relative tolerance (the SETTLE
      analytic solution assumes a single hydrogen mass).
    - Per-constraint-type mass consistency: two groups that share
      the same `constraint_type_index` must agree on
      `(masses[atom_O], masses[atom_H])`.
    Failures surface as `ConstraintError::InvalidGroupShape` (or
    `SettleError::InconsistentMasses`, which converts via `From`
    into `ConstraintError::InvalidGroupShape`).
  - `build(gpu, particle_count, list, masses, constraint_types)`
    constructs a `SettleConstraintsState` from the subset of
    `list.groups` whose constraint type resolves (via
    `constraint_types[group.constraint_type_index].kind`) to
    `"settle-water"`. Returns `ConstraintError::UnsupportedKind(kind)`
    if any group references a different algorithm; the v1 framework
    expects every constraint to be SETTLE, so the builder is the only
    constraint builder registered.

## Empty State <!-- rq-1a3e432c -->

When the slot's `group_count == 0`, all three trait methods return
`Ok(())` without launching any kernel. The slot still allocates
zero-length device buffers; this happens only when the topology had a
`[constraints]` section that was empty (the v1 framework otherwise
hands the runner `None` for the slot — see `constraint-framework.md`).

When `particle_count == 0`, every constraint row's atom index would
be out of range and the topology parser rejects the file before
SETTLE construction is reached. The slot is therefore never
constructed with `particle_count == 0` and a non-empty group list.

## Out of Scope <!-- rq-5adb53ee -->

- M-SHAKE, P-LINCS, LINCS, and every other constraint algorithm.
  Each is the target of its own future feature; this file describes
  only SETTLE.
- Composition with `velocity-verlet { lossless = true }`. The slot's
  builder rejects the combination via the framework-level
  `IntegratorBuilder::supports_constraints(&params)` check; lossless
  compensated summation for SETTLE corrections is a follow-up
  feature.
- Composition with `langevin-baoab` or `mtk-npt`. Same reason.
- Constraint virial contributions to the system pressure. SETTLE's
  position corrections do work on the system; that work is not added
  to `buffers.virials` in v1. Barostats compose with the SETTLE slot
  but do not see the constraint pressure contribution.
- Per-step diagnostics (max residual, per-group iteration count).
  SETTLE is non-iterative and the residual is zero by construction
  (up to floating-point rounding); a diagnostic is unnecessary in
  v1.
- Flexible water models (SPC/Flex, etc.). Those use the harmonic
  bond and angle slots already in the registry; SETTLE is for rigid
  water only.
- Mixed rigid/flexible bonds inside the same molecule. v1 rejects
  any pair that appears in both `[bonds]` and `[constraints]`.
- Cross-cluster constraints (atoms appearing in more than one
  constraint group). v1 rejects this at topology load.
- Multi-stream or multi-GPU launches.

---

## Gherkin Scenarios <!-- rq-10ee978e -->

```gherkin
Feature: SETTLE analytic three-atom rigid-water constraint

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called
    And a SettleBuilder registered in ConstraintRegistry::with_builtins()
    And a SimulationBox with lx=ly=lz=1.0e6 (large enough that no atom in any scenario crosses a box boundary in the steps below)
    And an "SPCE" SettleWater constraint type with r_oh = 1.0e-10 and r_hh = 1.633e-10

  # --- Module loading ---

  @rq-bdb4af60
  Scenario: init_device exposes the SETTLE kernels
    When init_device() is called
    Then the returned GpuContext's kernels handle exposes the settle_snapshot function
    And the kernels handle exposes the settle_positions function
    And the kernels handle exposes the settle_velocities function

  # --- Slot construction ---

  @rq-3abb71cd
  Scenario: Construct a SETTLE slot for one water
    Given a ParticleBuffers built from three atoms (O, H1, H2) at the SPCE equilibrium geometry
    And a ConstraintList containing one SettleWater group [0, 1, 2] referencing "SPCE"
    When SettleBuilder::build(device, particle_count=3, &list) is called
    Then it returns Ok(slot)
    And slot.group_count() is 1

  @rq-fd0add61
  Scenario: Construct on an empty ConstraintList yields a zero-group slot
    Given an empty ConstraintList
    When SettleBuilder::build(device, particle_count=4, &list) is called
    Then it returns Ok(slot)
    And slot.group_count() is 0

  @rq-7e0437ab
  Scenario: Reject inconsistent masses
    Given a ConstraintList declaring one SettleWater group whose three atoms have masses (15.0, 1.5, 1.0) (the two H atoms have different masses)
    When SettleBuilder::build(...) is called
    Then it returns Err(SettleError::InconsistentMasses { .. })

  @rq-535abb5b
  Scenario: Reject infeasible SETTLE geometry at config load
    Given a config with a constraint_types entry r_oh = 1.0e-10, r_hh = 3.0e-10 (r_hh > 2 r_oh)
    When load_config(&path) is called
    Then it returns Err(ConfigError::SettleGeometryInfeasible { name: _, r_oh: _, r_hh: _ })

  # --- settle_snapshot ---

  @rq-4ec4d1d6
  Scenario: settle_snapshot copies pre-drift positions verbatim
    Given a SETTLE slot with one water group [5, 7, 9] and arbitrary nonzero positions
    When constraint.apply_before_drift(&mut buffers, &sim_box, dt=0.001, &mut timings) is called
    Then slot.snapshot_x, snapshot_y, snapshot_z each hold the pre-drift positions of atoms 5, 7, 9 in that order
    And no other buffer is mutated

  # --- settle_positions on a known displacement ---

  @rq-4b375476
  Scenario: settle_positions restores the equilibrium geometry after a small uniform translation
    Given one SPCE water at its equilibrium geometry centred at the origin (pre-drift positions)
    And the post-drift positions are the pre-drift positions plus a uniform translation (1e-12, 0, 0) (no rotation, no internal stretch)
    When apply_after_drift is called with dt = 1e-15
    Then every constrained position equals (pre-drift + translation) within an absolute tolerance of 1e-14 m
    And every constraint distance (|r_O - r_H1|, |r_O - r_H2|, |r_H1 - r_H2|) equals its reference value within a relative tolerance of 1e-6

  @rq-a8b68f59
  Scenario: settle_positions restores constraint distances after a small bond stretch
    Given one SPCE water at its equilibrium geometry (pre-drift)
    And the post-drift positions stretch the O-H1 bond by +5% while leaving the other atoms unchanged
    When apply_after_drift is called with dt = 1e-15
    Then |r_O_constrained - r_H1_constrained| equals r_oh within a relative tolerance of 1e-6
    And |r_O_constrained - r_H2_constrained| equals r_oh within a relative tolerance of 1e-6
    And |r_H1_constrained - r_H2_constrained| equals r_hh within a relative tolerance of 1e-6

  @rq-f26ae0cc
  Scenario: settle_positions preserves the centre of mass of the unconstrained positions
    Given one SPCE water with arbitrary pre-drift positions and arbitrary post-drift positions
    When apply_after_drift is called
    Then the constrained centre of mass equals the unconstrained centre of mass to within an absolute tolerance of 1e-14 m

  @rq-25acc667
  Scenario: settle_positions updates the half-step velocities consistently with the position correction
    Given one SPCE water with pre-drift positions, half-step velocities v, and unconstrained post-drift positions
    When apply_after_drift is called with dt
    Then for each atom i, v_corrected[i] - v[i] equals (r_constrained[i] - r_unconstrained[i]) / dt within an absolute tolerance of 1e-9 m/s

  # --- settle_velocities ---

  @rq-66e657bf
  Scenario: settle_velocities zeroes the time-derivative of every constraint distance
    Given one SPCE water at the constrained geometry with arbitrary post-kick velocities
    When apply_after_kick is called
    Then (v_O - v_H1) · (r_O - r_H1) equals 0 to within an absolute tolerance of 1e-12 m²/s
    And (v_O - v_H2) · (r_O - r_H2) equals 0 to within an absolute tolerance of 1e-12 m²/s
    And (v_H1 - v_H2) · (r_H1 - r_H2) equals 0 to within an absolute tolerance of 1e-12 m²/s

  @rq-13af93b9
  Scenario: settle_velocities preserves the centre-of-mass velocity
    Given one SPCE water at the constrained geometry with arbitrary post-kick velocities
    And the pre-correction COM velocity v_C = (m_O v_O + m_H v_H1 + m_H v_H2) / M
    When apply_after_kick is called
    Then the post-correction COM velocity equals v_C within an absolute tolerance of 1e-12 m/s

  # --- Buffer side effects ---

  @rq-fc6ec19e
  Scenario: SETTLE kernels do not modify atoms outside any constraint group
    Given a ParticleBuffers with N = 16 particles, of which atoms [4, 5, 6] form the only SETTLE group
    And a snapshot of positions and velocities for every non-group atom before the hook
    When apply_before_drift, apply_after_drift, and apply_after_kick are all invoked
    Then positions and velocities for every atom not in {4, 5, 6} are byte-identical to the snapshot

  @rq-87330eaa
  Scenario: SETTLE kernels do not modify forces, masses, particle_ids, type_indices, potential_energies, or virials
    Given a ParticleBuffers with one SETTLE water and snapshots of all six arrays before any hook
    When all three hooks are invoked
    Then forces_x, forces_y, forces_z, masses, particle_ids, type_indices, potential_energies, and virials are byte-identical to the snapshot

  @rq-17c0a358
  Scenario: SETTLE kernels do not modify image flags
    Given a ParticleBuffers with one SETTLE water and a snapshot of images_x, images_y, images_z before any hook
    When all three hooks are invoked
    Then images_x, images_y, images_z are byte-identical to the snapshot

  # --- Multi-group independence ---

  @rq-d5790d66
  Scenario: Multiple water groups evolve independently
    Given a ParticleBuffers with N = 9 particles arranged as three SPCE waters with disjoint atom sets [0,1,2], [3,4,5], [6,7,8]
    And each water has a distinct pre-drift geometry and post-drift displacement
    When apply_before_drift then apply_after_drift then apply_after_kick are invoked
    Then each water's constrained geometry matches the single-water reference for its own pre/post inputs to within tolerance

  # --- Empty-state ---

  @rq-5d972f15
  Scenario: SETTLE hooks on a zero-group slot are no-ops
    Given a SETTLE slot with group_count() == 0
    When apply_before_drift, apply_after_drift, and apply_after_kick are each called
    Then each returns Ok(())
    And no kernel launches are recorded for any call

  @rq-02b066b6
  Scenario: SETTLE hooks on an empty ParticleBuffers return Ok(())
    Given a ParticleBuffers with particle_count() == 0
    And a SETTLE slot whose group_count() is also 0
    When each hook is called
    Then each returns Ok(())

  # --- Reproducibility ---

  @rq-99ee814d
  Scenario: Two independent SETTLE runs produce byte-identical outputs
    Given two SETTLE slots constructed from byte-identical inputs (16 SPCE waters)
    And two ParticleBuffers built from byte-identical ParticleStates
    When apply_before_drift, a vv_kick_drift, apply_after_drift, a vv_kick, then apply_after_kick are run on each
    And both buffers are downloaded
    Then every f32 and u32 array of run A is byte-identical to run B

  # --- End-to-end constrained dynamics on one water ---

  @rq-025702ac
  Scenario: A single SETTLE water under free streaming (no forces) maintains rigid geometry indefinitely
    Given one SPCE water at the equilibrium geometry with non-zero initial velocities that include both translation and rotation
    And a velocity-Verlet integrator (lossy) with constraint slot containing this water and zero force evaluations
    When the runner executes 1000 timesteps of dt = 1e-15 s
    Then at every step, all three constraint distances remain within 1e-5 relative of their reference values
    And the centre-of-mass velocity at step 1000 equals the initial centre-of-mass velocity within an absolute tolerance of 1e-9 m/s
```
