# Feature: Rigid Three-Atom Water Constraint (`settle-water`) <!-- rq-67e62f4b -->

The `settle-water` constraint kind projects a three-atom rigid water
molecule's positions and velocities back onto its constraint manifold
at fixed sub-step boundaries inside the integrator. It is the v1
implementation of the pluggable `Constraint` slot defined in
`constraint-framework.md`, selected by `kind = "settle-water"` on a
`[[constraint_types]]` entry in the TOML config.

A `settle-water` constraint group has exactly three atoms with
exactly three constraints forming the rigid triangle `(O, H1, H2)`.
The two O–H distances must be equal (parameter `r_oh`) and the H–H
distance is the parameter `r_hh`. The atom listed first in each
`[constraints]` row is the oxygen; the next two are the hydrogens,
in either order.

The implementation provides four CUDA kernels — `settle_positions`,
`settle_velocities`, `settle_virial_scatter`, and
`settle_positions_no_velocity` — and a single host-side slot
(`SettleConstraintsState`) that owns the device-side per-group buffers
and implements the `Constraint` trait. The slot's `apply_before_drift`
hook snapshots pre-drift positions; `apply_after_drift` runs
`settle_positions` to project the post-drift positions back onto the
manifold, to correct the half-step velocities, and to write the
position-level half of each atom's constraint-virial contribution
into a slot-owned buffer; `apply_after_kick` runs `settle_velocities`
to project the final velocities onto the manifold and accumulate the
velocity-level half of the constraint virial into the same buffer,
followed by `settle_virial_scatter` to fold the combined contribution
into `buffers.virials` so the barostat sees both halves on the same
timestep. The slot's `apply_position_projection_only` hook runs
`settle_positions_no_velocity` to perform the same position projection
as `settle_positions` but without the half-step velocity correction
and without writing the constraint-virial scratch.

## Status: non-standard hybrid implementation, pending migration <!-- rq-176bca9c -->

**The kernels are named `settle_*` for historical reasons, but the
implementation is *not* the analytical SETTLE algorithm of Miyamoto
& Kollman (*J. Comput. Chem.* **13**, 952 (1992)).** It is a
non-standard hybrid:

- `settle_positions` and `settle_positions_no_velocity` are
  **iterative SHAKE** on the three pair-distance constraints
  (O–H1, O–H2, H1–H2), up to 32 sweeps per group with absolute
  tolerance `|σ| < 10⁻²⁶ m²` on each constraint.
- `settle_velocities` is **closed-form RATTLE**: a single
  Cramer's-rule solve of the 3×3 Lagrange-multiplier linear
  system that the rigid-water topology produces for the
  velocity-level projection.

Only the velocity-level kernel retains anything specifically
SETTLE-like, and even that's because the constraint matrix
happens to be 3×3 — there is no use of the body-frame /
closed-form rotation that the Miyamoto-Kollman algorithm is
named for. The position-level kernel is generic SHAKE applied
to a fixed three-constraint cluster.

This hybrid is the result of a correctness-preserving substitution:
an earlier implementation followed the Miyamoto-Kollman analytical
derivation but parameterised the body-frame rotation by only the
oxygen's in-plane angle and out-of-plane tilt, capturing 2 of 3
rotational DOFs and silently zeroing the dipole-axis rotation.
Replacing the analytical rotation with iterative SHAKE — whose `Δr`
is guaranteed to lie in the constraint-gradient subspace required
for RATTLE-consistent velocity coupling — was easier to verify
correct than fixing the parameterisation.

**Migration plan.** This `settle-water` kind is expected to be
deprecated by two follow-on features:

1. A faithful **analytical SETTLE** (`kind = "settle-water-analytic"`
   or similar) implementing the closed-form Miyamoto-Kollman 1992
   projection — non-iterative, deterministic per-group cost, and the
   algorithm that production MD codes mean when they say "SETTLE".
2. A general **M-SHAKE / SHAKE+RATTLE** (`kind = "m-shake"` or
   similar) for arbitrary rigid constraint clusters — of which
   three-atom rigid water is the simplest special case, and which
   would obsolete the present hardcoded 3-atom kernel.

When both arrive, `settle-water` as documented here will be removed.
Until then this file documents the current implementation honestly:
its kernels, its iteration counts, its tolerances, and the
constraint-virial decomposition it uses. References to "SETTLE" in
this file refer to the *kind string* `"settle-water"`, not to the
Miyamoto-Kollman algorithm.

## Algorithm <!-- rq-ce77d9fb -->

For one water group `g` with atoms `(O, H1, H2)`, masses
`(m_O, m_H, m_H)`, pre-drift positions `(r_O0, r_H10, r_H20)`, and
unconstrained post-drift positions `(r_O', r_H1', r_H2')`, the
`settle_positions` step computes constrained post-drift positions
`(r_O, r_H1, r_H2)` and updates the half-step velocities `(v_O, v_H1,
v_H2)` to reflect the position correction. `settle_velocities`
subsequently projects the post-kick velocities onto the velocity
manifold of the constrained positions. The two kernels together
implement a SHAKE position projection followed by a closed-form
RATTLE velocity projection on the three pair-distance constraints
of rigid water; see *Status* above for the algorithm-naming
disclaimer.

### Position projection: iterative SHAKE <!-- rq-c1f9a5b3 -->

`settle_positions` solves the three pair-distance constraints
iteratively. Index the constraints `k = 1, 2, 3` as `(O–H1)`,
`(O–H2)`, `(H1–H2)` with target distances `d_1 = d_2 = r_oh`,
`d_3 = r_hh`. Denote the pre-drift relative-position vectors as
`g_k = r_i^{(0)} − r_j^{(0)}` (where `(i, j)` is the atom pair of
constraint `k`); these are the constraint-gradient directions used
by SHAKE.

Per group, starting from `r_i = r_i'` (the unconstrained post-drift
positions, after minimum-image fix-up so all three atoms of the
group lie in the same lattice image):

```text
for iter in 0..MAX_ITER (= 32):
    converged = true
    for each constraint k = (i, j) with target d_k:
        σ_k = |r_i − r_j|² − d_k²
        if |σ_k| > tol²:                     # tol² = 1.0e-26 m²
            converged = false
            ddot   = (r_i − r_j) · g_k
            inv_m  = 1/m_i + 1/m_j
            λ_k    = σ_k / (2 · ddot · inv_m)
            r_i   -= λ_k · g_k / m_i
            r_j   += λ_k · g_k / m_j
    if converged: break
```

The constraint-gradient direction `g_k` is fixed at the pre-drift
geometry and reused across iterations; the linearised Newton step
above brings each `|σ_k|` below `tol²` in O(1) sweeps for thermal
MD step sizes (typically 1–2 sweeps per group at `dt = 2 fs`,
ramping with the magnitude of the per-step unconstrained
displacement). The cumulative position correction
`Δr_i = r_i_constrained − r_i_unconstrained` lies in the
constraint-gradient subspace by construction (it is a linear
combination of the `g_k`), which is what makes the corresponding
velocity correction `v_i ← v_i + Δr_i / dt` RATTLE-consistent at
the leapfrog half-step.

After the SHAKE loop, `settle_positions` writes the constrained
positions back to global memory and applies the half-step
velocity correction `v_i ← v_i + Δr_i / dt` for every atom in the
group. It then writes the **position-level** half of the per-atom
constraint-virial contribution; see *Constraint Virial → Decomposition*
for the formula and the velocity-level half added later by
`settle_velocities`.

### Velocity projection: closed-form RATTLE <!-- rq-77f3ae21 -->

`settle_velocities` projects the post-second-kick velocities
`(v_O', v_H1', v_H2')` onto the velocity manifold of the
constrained positions `(r_O, r_H1, r_H2)`. For the three pair
constraints `k ∈ {OH1, OH2, HH}`, the velocity manifold is the
joint kernel of the constraint Jacobian rows
`∇_i C_k = r_i − r_j` (for atom `i` ∈ constraint `k` coupling
atoms `(i, j)`):

```text
(v_i_corrected − v_j_corrected) · (r_i − r_j) = 0    for each k
```

Writing `v_i_corrected = v_i' − Σ_k s_k_i · λ_k · (r_i − r_j) / m_i`
(where `s_k_i = ±1` is the sign of atom `i` in constraint `k`),
the manifold conditions reduce to a 3×3 symmetric linear system
`M λ = b`, with

```text
M_{kl} = Σ_atoms s_k_atom · s_l_atom · (d_k · d_l) / m_atom
b_k    = (v_i' − v_j') · (r_i − r_j)            for constraint k = (i,j)
```

where `d_k = r_i − r_j` is the constraint-direction vector
(post-SHAKE). The 3×3 matrix is symmetric positive-definite under
the rigid-water geometry; `settle_velocities` solves it in
**closed form via Cramer's rule** (a fixed number of arithmetic
operations per group, no iteration). The Lagrange multipliers
`λ = (λ_OH1, λ_OH2, λ_HH)` then drive a single per-atom velocity
update:

```text
v_O   ← v_O   − (λ_OH1 · d_OH1 + λ_OH2 · d_OH2) / m_O
v_H1  ← v_H1  + (λ_OH1 · d_OH1 − λ_HH  · d_HH ) / m_H
v_H2  ← v_H2  + (λ_OH2 · d_OH2 + λ_HH  · d_HH ) / m_H
```

after which `(v_i − v_j) · (r_i − r_j) = 0` holds for every
constraint to within f32 rounding.

The same kernel additionally accumulates the **velocity-level**
half of the per-atom constraint-virial contribution; see
*Constraint Virial → Decomposition*.

### Iteration count and convergence <!-- rq-7e63e0f4 -->

The position-level SHAKE loop's iteration count is bounded by
`SHAKE_MAX_ITER = 32` per group. The `tol² = 1.0e-26 m²` absolute
tolerance corresponds to a relative tolerance of ~10⁻⁶ on a
(10⁻¹⁰ m)² distance — well below thermal noise at MD step sizes.
At `dt = 2 fs` and 300 K rigid water, σ_k for the H–H constraint
is of order `(v_thermal · dt)² ≈ 10⁻²³ m²`, three orders of
magnitude above tolerance, and the loop converges in 1–2 sweeps.
Larger step sizes or near-pathological initial conditions raise
the iteration count toward the 32-sweep cap; reaching the cap
without converging is unexpected and is not currently treated as
a hard error (the kernel exits with whatever residual remains and
the caller observes a small constraint violation in
`buffers.positions_*`).

## Per-Step Kernel Sequence <!-- rq-de7601cd -->

| Order | Hook                                | Kernel                            | Operation                                                                                                                                                                                       | Stage label                   |
| ----- | ----------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| 1     | `apply_before_drift`                | `settle_snapshot`                 | copy pre-drift positions of every group's atoms into the slot's snapshot buffer                                                                                                                 | `SettleSnapshot`              |
| 2     | `apply_after_drift`                 | `settle_positions`                | per-group analytic projection of positions; per-group half-step velocity correction; per-atom position-level constraint-virial contribution **written** to the slot buffer                      | `SettlePositions`             |
| 3     | `apply_after_kick`                  | `settle_velocities`               | per-group analytic projection of final velocities; per-atom velocity-level constraint-virial contribution **added** to the same slot buffer (or skipped when `dt == 0`)                         | `SettleVelocities`            |
| 4     | `apply_after_kick`                  | `settle_virial_scatter`           | per-atom-of-group scatter of the cached constraint-virial values (position + velocity halves summed) into `buffers.virials` for the barostat                                                    | `SettleVirialScatter`         |
| 5     | `apply_position_projection_only`    | `settle_positions_no_velocity`    | per-group analytic projection of positions only (no velocity correction, no virial scratch write); used by minimization phases                                                                  | `SettlePositionsNoVelocity`   |

All five kernels run with one thread per constraint group (or per
constraint-group-atom for the scatter, with one thread per atom slot
`3 * group_count`). Block size is 256; grid size is
`ceil(work_items / 256)`. Shared memory is zero bytes. The stream is
the default stream carried by `ParticleBuffers::device`.

Hooks 1–4 fire during the MD plan walk only. Hook 5 fires during
minimization phases only (see
`rqm/minimization/steepest-descent.md`); the snapshot buffer is not
consulted because the minimization projection is per-trial and does
not depend on a pre-drift reference frame.

## Constraint Virial <!-- rq-b5baee1d -->

The SETTLE constraint forces do work on the system that the
barostat (`c-rescale-barostat.md`, `berendsen-barostat.md`) must
see in its instantaneous pressure estimate `P = (2K + W) / (3V)`.
The per-atom contribution has two pieces — one delivered with the
SHAKE position projection and one delivered with the RATTLE
velocity projection — and the slot accumulates both into a single
per-atom scalar before scattering into `buffers.virials`.

### Decomposition <!-- rq-89b7da40 -->

In velocity-Verlet + SHAKE + RATTLE, the per-step constraint
impulse is split across two half-steps:

- The position projection in `apply_after_drift` (SHAKE) snaps
  positions onto the constraint manifold and, by the standard
  RATTLE-consistent half-step velocity update, also delivers an
  impulse. The corresponding time-averaged constraint force is
  `m_i · Δr_i / dt²`, where
  `Δr_i = r_constrained_i − r_unconstrained_i`.
- The velocity projection in `apply_after_kick` (RATTLE) snaps the
  freshly-kicked velocities back onto the velocity manifold,
  delivering an additional impulse. The corresponding
  time-averaged constraint force is `m_i · Δv_i / dt`, where
  `Δv_i` is the change applied by the RATTLE Lagrange-multiplier
  solve.

The per-atom virial entries written into the slot-owned
`constraint_virial: CudaSlice<f32>` buffer therefore have two
additive parts:

```text
position-level: w_i_pos = m_i · (Δr_i · r_i_COM) / dt²
velocity-level: w_i_vel = m_i · (Δv_i · r_i_COM) / dt
```

where `r_i_COM = r_i − r_COM_mol` is the atom's COM-relative
constrained position (used both for f32 numerical stability — see
*Numerical stability* below — and because the within-molecule sum
of constraint forces vanishes, so lab-frame and COM-relative forms
of `Σ F · r` agree).

Recovering both halves is required for the engine to compute the
physically correct pressure when constraint forces balance
intermolecular forces along bond directions (the dominant case for
SPC/E water under electrostatic cohesion). Either half alone is
incomplete and biases the reported pressure.

### Computation pipeline <!-- rq-1ceb334f -->

1. **Position-level compute** — inside `settle_positions` during
   `apply_after_drift`. After solving for `r_constrained_i`, the
   kernel writes
   `constraint_virial[3*g+k] = m_i · (Δr_i · r_i_COM) / dt²` for
   every atom in the group (assignment, not addition).
2. **Velocity-level accumulate** — inside `settle_velocities` during
   `apply_after_kick`. After applying the RATTLE velocity
   correction, the kernel adds
   `m_i · (Δv_i · r_i_COM) / dt` to the same
   `constraint_virial[3*g+k]` slot (addition, not assignment, so
   the position-level value written in step 1 is preserved).
3. **Scatter** — `settle_virial_scatter` runs at the end of
   `apply_after_kick`, after the force evaluation has populated
   `buffers.virials` with its force-field contributions and after
   `settle_velocities` has finished. For each
   `(group g, local atom k)` slot it adds
   `constraint_virial[3*g+k]` to
   `buffers.virials[group_atoms[3*g+k]]`.

The split between steps 1 and 3 is required by the per-step order
in `constraint-framework.md`:
`apply_before_drift → KickDrift → apply_after_drift →
ForceEval → KickHalf → apply_after_kick → barostat`. The force
evaluation between `apply_after_drift` and `apply_after_kick`
overwrites `buffers.virials`, so the scatter happens *after* it.

The scatter writes only to the `3 * group_count` atom slots actually
owned by SETTLE; every other entry of `buffers.virials` is left
unchanged. No atomics are used: each constraint group is exclusive
to one thread in `settle_positions` / `settle_velocities`, and the
scatter is a per-slot write to a unique atom index because
constraint groups have disjoint atom sets (the topology parser
rejects overlap).

### Numerical stability <!-- rq-6c3a4f88 -->

The per-atom expression `m · (Δ · r) / dt^k` (k = 2 for the
position-level half, k = 1 for the velocity-level half) is
implemented with the scale factor evaluated first:

```c
float scale = m / dt^k;          // ≈ O(10³)
contribution = scale * (Δ · r);  // (Δ · r) ≈ O(10⁻²³ m²)
```

The left-associative grouping `(m · (Δ · r)) / dt^k` would form an
intermediate `m · (Δ · r) ≈ O(10⁻⁵⁰)` that **underflows in f32**
(smallest denormal ≈ 1.4·10⁻⁴⁵) and silently rounds to zero,
collapsing the per-atom contribution to zero regardless of how
much SHAKE / RATTLE actually does. The scale-first grouping keeps
every intermediate in f32 normal range and is the only
associativity-preserving order that does so for typical MD
magnitudes.

The empty-state and reproducibility properties documented for
`settle_positions` and `settle_velocities` apply unchanged to the
scatter: a zero-group slot performs no launches; identical inputs on
the same GPU produce byte-identical virial buffers across runs.

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

- `r_oh: f64` — O–H constraint distance, Bohr (`a_0`). Required.
  Finite and strictly positive.
- `r_hh: f64` — H–H constraint distance, Bohr (`a_0`). Required.
  Finite
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
across groups that share a constraint type: the SHAKE + RATTLE
kernels assume a single oxygen mass `m_O` and a single hydrogen
mass `m_H` per constraint type (the inverse masses are hard-coded
into the constraint-pair Jacobian structure). The slot constructor reads
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
  - `constraint_virial: CudaSlice<f32>` — length `3 * group_count`,
    per-atom-of-group scalar contribution `m_i · ((r_constrained_i −
    r_unconstrained_i) · r_constrained_i) / dt²` written by
    `settle_positions` and consumed by `settle_virial_scatter`. See
    *Constraint Virial* above. Refreshed each step.

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

`kernels/settle.cu` declares five `extern "C"` kernels:

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
    float *constraint_virial,
    unsigned int n_groups);

extern "C" __global__ void settle_velocities(
    const float *positions_x, const float *positions_y, const float *positions_z,
    float *velocities_x, float *velocities_y, float *velocities_z,
    const unsigned int *group_atoms,
    const unsigned int *group_type_index,
    const float *type_mass_o, const float *type_mass_h,
    float lx, float ly, float lz, float xy, float xz, float yz,
    float dt,
    float *constraint_virial,
    unsigned int n_groups);

extern "C" __global__ void settle_virial_scatter(
    const float *constraint_virial,
    const unsigned int *group_atoms,
    float *particle_virials,
    unsigned int n_atom_slots);

extern "C" __global__ void settle_positions_no_velocity(
    float *positions_x, float *positions_y, float *positions_z,
    const unsigned int *group_atoms,
    const unsigned int *group_type_index,
    const float *type_canonical_x, const float *type_canonical_y, const float *type_canonical_z,
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
7. Executes the iterative SHAKE loop documented in
   *Algorithm → Position projection: iterative SHAKE* — up to
   `SHAKE_MAX_ITER = 32` sweeps over the three pair constraints,
   each sweep applying the linearised Lagrange-multiplier update
   `r_i -= λ_k · g_k / m_i` for any constraint whose residual
   `|σ_k| > 10⁻²⁶ m²`. The pre-drift snapshot supplies the
   constraint-gradient directions `g_k` reused across iterations.
8. Writes constrained positions back to `positions_{x,y,z}[a_*]`.
9. Writes corrected half-step velocities back to
   `velocities_{x,y,z}[a_*]` using `v_i ← v_i + (r_constrained - r_unconstrained) / dt`.
10. For each of the three atoms `i ∈ {O, H1, H2}`, writes the per-atom
    constraint-virial contribution
    `w_i = m_i · ((r_constrained_i − r_unconstrained_i) · r_constrained_i) / dt²`
    into `constraint_virial[3*g .. 3*g+3]`. See *Constraint Virial*.

The kernel writes only the nine `f32` slots `positions_{x,y,z}[a_O]`,
`positions_{x,y,z}[a_H1]`, `positions_{x,y,z}[a_H2]`, the nine
`f32` slots `velocities_{x,y,z}[a_O]`, `velocities_{x,y,z}[a_H1]`,
`velocities_{x,y,z}[a_H2]`, and the three slot-owned virial slots
`constraint_virial[3*g .. 3*g+3]`. Every other particle's state and
every other entry of `constraint_virial` is untouched.

The kernel does not advance `images_{x,y,z}` for the corrected
atoms. The SHAKE position correction produces displacements at most
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
9. When `dt > 0.0f`, accumulates the per-atom velocity-level
   constraint-virial contribution `m_i · (Δv_i · r_i_COM) / dt`
   into `constraint_virial[3*g + k]` (additive; the position-level
   contribution from `settle_positions` is already in this buffer
   and must be preserved). Because the velocity correction
   `Δv_i = ±λ_k · r_k / m_i` carries `1 / m_i`, the per-atom mass
   cancels and the contributions reduce to
   `(λ_OH1 / dt) · r_OH1 · r_i_COM ± …`, using the per-mass
   COM-relative constrained position `r_i_COM = r_i − r_COM_mol`.
   See *Constraint Virial → Decomposition* for the full formula
   and *Numerical stability* for the operation ordering. When
   `dt == 0.0f` (initial-velocity projection at runner setup, where
   no associated timestep exists), this step is skipped — the
   `constraint_virial` argument is treated as scratch and may carry
   stale values; the next `settle_positions` overwrites it before
   any consumer reads.

The kernel writes only the nine `f32` velocity slots of the group
and (when `dt > 0`) the three slot-owned `constraint_virial` slots
of the group. Positions are not modified.

#### `settle_virial_scatter` <!-- rq-a3f15f82 -->

One thread per atom slot in `group_atoms` (`n_atom_slots = 3 *
group_count`). For each thread index `s`:

1. Reads `w = constraint_virial[s]` and `atom_index =
   group_atoms[s]`.
2. Executes
   `particle_virials[atom_index] = particle_virials[atom_index] + w`.

The write is a plain addition with no atomics: the topology parser
guarantees every constraint group has a disjoint atom set, so each
`atom_index` appears in exactly one slot of `group_atoms`.

The kernel writes only the `group_count` distinct atom slots covered
by SETTLE groups. Every other entry of `particle_virials` is left
unchanged.

#### `settle_positions_no_velocity` <!-- rq-fb83923b -->

For group `g`:

1. Reads atom indices `(a_O, a_H1, a_H2)` from
   `group_atoms[3*g .. 3*g+3]` and type index `t = group_type_index[g]`.
2. Reads unconstrained positions from `positions_{x,y,z}[a_*]`.
3. Reads canonical body-frame positions from
   `type_canonical_{x,y,z}[3*t .. 3*t+3]`.
4. Reads masses `m_O = type_mass_o[t]`, `m_H = type_mass_h[t]`.
5. Computes the centre of mass of the unconstrained positions and
   minimum-image displacements between every pair of atoms using the
   triclinic tilt-subtraction algorithm in `simulation-box.md`.
6. Builds the body frame directly from the unconstrained positions
   (rather than from a separate pre-drift snapshot). Because
   minimization has no pre-drift / post-drift distinction — there is
   only one position state per trial — the projection target is
   the closest point on the manifold to the unconstrained
   positions, with the body frame oriented to minimise the rigid-
   body rotation required to satisfy the constraint distances. This
   is the same projection logic as `settle_positions` with the
   "pre-drift" frame replaced by the "unconstrained" frame; the
   closed-form solution of Miyamoto & Kollman 1992 applies
   unchanged.
7. Writes constrained positions back to `positions_{x,y,z}[a_*]`.

The kernel writes only the nine `f32` slots
`positions_{x,y,z}[a_O]`, `positions_{x,y,z}[a_H1]`,
`positions_{x,y,z}[a_H2]`. It does **not** read or write
`velocities_*`, `constraint_virial`, `forces_*`, or any other buffer.
It does not consume `dt` (no parameter for it; minimization has no
time scale).

### PTX Module Loading <!-- rq-ba4d55e6 -->

`init_device()` loads the compiled `kernels/settle.cu` PTX as module
`"settle"` and captures its `settle_snapshot`, `settle_positions`,
`settle_velocities`, `settle_virial_scatter`, and
`settle_positions_no_velocity` functions into the `Kernels` handle
(see `build-pipeline.md`).

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

- The analytical Miyamoto-Kollman 1992 SETTLE algorithm and a
  general M-SHAKE / P-LINCS / LINCS solver. These are the two
  follow-on features the current `settle-water` kind is expected
  to be deprecated by (see *Status*); they will arrive in their
  own feature files. This file describes the current SHAKE +
  closed-form RATTLE hybrid only.
- Composition with `velocity-verlet { lossless = true }`. The slot's
  builder rejects the combination via the framework-level
  `IntegratorBuilder::supports_constraints(&params)` check;
  lossless compensated summation for constraint corrections is a
  follow-up feature.
- Composition with `langevin-baoab` or `mtk-npt`. Same reason.
- Per-step diagnostics (max constraint residual after the SHAKE
  loop, per-group iteration count). The kernel does not currently
  surface the per-group `iter` value or the final `σ_k` residual
  to host. At thermal MD step sizes the loop converges in 1–2
  sweeps and the residual is at the SHAKE tolerance
  (~10⁻²⁶ m²), well below physical relevance; a diagnostic
  surface for unusual step sizes or pathological initial
  conditions is a follow-up feature.
- Hitting the iteration cap. `SHAKE_MAX_ITER = 32` is sized
  generously for thermal MD; reaching the cap without converging
  is not currently treated as a hard error (the kernel exits with
  whatever residual remains). A strict failure mode for
  non-convergence is a follow-up feature.
- Flexible water models (SPC/Flex, etc.). Those use the harmonic
  bond and angle slots already in the registry; `settle-water` is
  for rigid water only.
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

  @rq-fd498605
  Scenario: SETTLE kernels do not modify forces, masses, particle_ids, type_indices, or potential_energies
    Given a ParticleBuffers with one SETTLE water and snapshots of forces_*, masses, particle_ids, type_indices, and potential_energies before any hook
    When all three hooks are invoked
    Then forces_x, forces_y, forces_z, masses, particle_ids, type_indices, and potential_energies are byte-identical to the snapshot

  @rq-28bba228
  Scenario: SETTLE adds constraint virial to buffers.virials in apply_after_kick
    Given a ParticleBuffers with one SETTLE water at atoms [a_O, a_H1, a_H2]
      and an initial buffers.virials[a_O] = 7.0, buffers.virials[a_H1] = 11.0, buffers.virials[a_H2] = 13.0
    And a non-trivial pre-drift → post-drift displacement
    When apply_before_drift, apply_after_drift, and apply_after_kick are all invoked
    Then buffers.virials[a_O] equals 7.0 + w_O within absolute tolerance 1e-12
    And buffers.virials[a_H1] equals 11.0 + w_H1 within absolute tolerance 1e-12
    And buffers.virials[a_H2] equals 13.0 + w_H2 within absolute tolerance 1e-12
    And buffers.virials at every other index is unchanged
    Where w_i = m_i · ((r_constrained_i − r_unconstrained_i) · r_constrained_i) / dt²
      computed from the same inputs

  @rq-72dceb28
  Scenario: settle_virial_scatter is a no-op for a zero-group slot
    Given a SETTLE slot with group_count() == 0 and a snapshot of buffers.virials
    When apply_after_kick is called
    Then buffers.virials is byte-identical to the snapshot
    And no kernel launches are recorded for the scatter step

  @rq-9cf9ece2
  Scenario: Adding constraint virial completes the pressure cancellation for an equilibrated rigid water gas
    Given a composed runner of velocity-Verlet + CSVR + c-rescale-barostat with
      N_mol SPCE waters (well-equilibrated rigid water at 1 g/cm³, T = 298.15 K)
    When the run completes
    Then the time-averaged pressure reported on the c-rescale-barostat's log column
      is within 200 bar of the configured target P (1.013e5 Pa)
    (Without the SETTLE constraint virial the time-averaged pressure
    differs from P_target by O(N_mol · k_B · T / V), which for SPC/E
    liquid at standard conditions is several kbar.)

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

  # --- apply_position_projection_only (minimization hook) ---

  @rq-57d0aebf
  Scenario: settle_positions_no_velocity restores constraint distances from off-manifold positions
    Given one SPCE water at the equilibrium geometry stretched by +5% on the O-H1 bond
    When apply_position_projection_only is called
    Then |r_O - r_H1| equals r_oh within relative tolerance 1.0e-6
    And |r_O - r_H2| equals r_oh within relative tolerance 1.0e-6
    And |r_H1 - r_H2| equals r_hh within relative tolerance 1.0e-6

  @rq-5a3bb763
  Scenario: settle_positions_no_velocity does not modify velocities or virials
    Given one SPCE water with arbitrary off-manifold positions, non-zero velocities, and a snapshot of buffers.virials
    When apply_position_projection_only is called
    Then velocities_x, velocities_y, velocities_z are byte-identical to their pre-call values
    And buffers.virials is byte-identical to the snapshot
    And forces_x, forces_y, forces_z are byte-identical to their pre-call values

  @rq-bc7a8950
  Scenario: settle_positions_no_velocity preserves the centre of mass of the unconstrained positions
    Given one SPCE water with arbitrary off-manifold positions
    When apply_position_projection_only is called
    Then the constrained centre of mass equals the unconstrained centre of mass to within absolute tolerance 1e-14 m

  @rq-173bfc40
  Scenario: SettleBuilder reports supports_position_projection_only as true
    Given the SettleBuilder from ConstraintRegistry::with_builtins()
    And any well-formed settle-water params
    Then builder.supports_position_projection_only(&params) returns true

  # --- End-to-end constrained dynamics on one water ---

  @rq-025702ac
  Scenario: A single SETTLE water under free streaming (no forces) maintains rigid geometry indefinitely
    Given one SPCE water at the equilibrium geometry with non-zero initial velocities that include both translation and rotation
    And a velocity-Verlet integrator (lossy) with constraint slot containing this water and zero force evaluations
    When the runner executes 1000 timesteps of dt = 1e-15 s
    Then at every step, all three constraint distances remain within 1e-5 relative of their reference values
    And the centre-of-mass velocity at step 1000 equals the initial centre-of-mass velocity within an absolute tolerance of 1e-9 m/s
```
