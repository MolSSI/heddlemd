# Feature: SETTLE Constraint Algorithm <!-- rq-83321065 -->

SETTLE is the analytical constraint algorithm of Miyamoto & Kollman
(J. Comput. Chem. 13(8):952–962, 1992) for a symmetric three-atom rigid water
molecule. It holds the three intramolecular distances (O–H1, O–H2, H1–H2)
rigid at every timestep, specialised to the three water constraints and
solved with the per-group working set resident in registers (one thread per
molecule, no shared-memory staging). Both per-step resets are closed-form and
non-iterative: the position reset is the Miyamoto-Kollman rigid-body rotation
of the canonical triangle onto the unconstrained positions, oriented by the
pre-drift reference frame; the velocity reset drives the relative velocity
along every bond to zero by a single direct solve of the 3×3 bond-impulse
system.

A `SettleConstraintsState` implements the `Constraint` trait (see
`integration/constraint-framework.md`). Its `apply_before_drift` hook
snapshots the pre-drift positions that supply the rotation's reference frame;
its `apply_after_drift` hook performs the M-K position reset of the
unconstrained post-drift positions, updates the half-step velocities to be
consistent with the position correction, and writes the position-level half
of the per-atom constraint-virial contribution; its `apply_after_kick` hook
performs the analytical velocity reset and accumulates the velocity-level
half of the constraint-virial contribution; its
`apply_position_projection_only` hook performs a position-only reset used by
the minimizer. The minimizer hook does not use the M-K rotation — it has no
pre-drift reference frame, and a rigid snap is not a descent-compatible
retraction — so it instead performs the minimal-displacement projection onto
the rigid manifold (see *Algorithm* step 4).

This algorithm handles exactly one cluster shape: a symmetric rigid water
group of three atoms — one apex atom (the oxygen) and two equivalent
hydrogens with equal mass and equal apex-bond length. Each group carries
three holonomic constraints, so a run of `n_settle_groups` water molecules
contributes `3 · n_settle_groups` constraints to the thermal
degree-of-freedom count consumed by `csvr.md` and `log-output.md`. Clusters
that are not symmetric three-atom water are handled by SHAKE (`shake.md`),
the general iterative fallback.

## Canonical Geometry <!-- rq-57db9db2 -->

Each constraint type fixes one rigid water shape through two distances:
`d_OH` (apex–hydrogen bond length) and `d_HH` (hydrogen–hydrogen distance),
both in Bohr (`a_0`) after the config-load atomic-units conversion. With the
apex mass `m_O = masses[atom 0]` and the common hydrogen mass
`m_H = masses[atom 1] = masses[atom 2]`, the canonical molecular geometry is
computed once per group at slot construction:

```text
M  = m_O + 2 · m_H
h  = sqrt(d_OH² − (d_HH / 2)²)      # apex height above the H–H midpoint
ra = (2 · m_H / M) · h              # oxygen distance from COM, +symmetry axis
rb = (m_O / M) · h                  # hydrogen distance from COM, −symmetry axis
rc = d_HH / 2                       # half the H–H distance
```

The canonical positions in the molecular frame (centre of mass at the
origin, molecule in the xy-plane) are `O = (0, ra, 0)`,
`H1 = (−rc, −rb, 0)`, `H2 = (+rc, −rb, 0)`. By construction `ra + rb = h`
and `m_O · ra = 2 · m_H · rb`, so the canonical centre of mass is at the
origin. `h` is real and strictly positive iff `d_OH > d_HH / 2`; a
constraint type that violates this is rejected at config load (see
*Parameters*).

## Algorithm <!-- rq-709c8eb5 -->

For one rigid water group with apex atom `A` (local index 0, the oxygen) and
hydrogens `B`, `C` (local indices 1, 2), masses `m_O`, `m_H`, and canonical
geometry `(ra, rb, rc)`:

1. `apply_before_drift` invokes `settle_snapshot`, which copies every atom of
   every group from `positions_*` into per-group snapshot buffers
   `snapshot_*`. The snapshot is the on-manifold (pre-drift) configuration
   that defines the reference orientation frame for the M-K rotation that
   follows the drift; orienting the reset by the pre-drift frame is what makes
   the constraint forces act along the pre-drift bonds (energy-conserving).

2. `apply_after_drift` invokes `settle_positions`, which resets the
   unconstrained post-drift positions to the exact rigid configuration by the
   analytical Miyamoto-Kollman rotation. The pre-drift snapshot supplies the
   reference orientation; the reset computes, in closed form, the rigid
   placement of the canonical triangle that has the centre of mass of the
   unconstrained positions and is reached from them by per-atom displacements
   directed along the reference (snapshot) bonds — the configuration
   constrained dynamics reaches, evaluated directly rather than by iteration:

   ```text
   A1,B1,C1 ← positions[group atoms]   (unconstrained post-drift; A=O, B=H1, C=H2)
   A0,B0,C0 ← snapshot[group atoms]    (pre-drift, on the constraint manifold)
   Bring B1,C1 into A1's lattice image and B0,C0 into A0's image (minimum image).
   com ← (m_O·A1 + m_H·B1 + m_H·C1) / M          # constrained COM = unconstrained COM

   # Reference orthonormal frame (X', Y', Z') from the snapshot triangle about
   # its mass-weighted COM, oriented so the canonical atoms sit at
   # O=(0,ra,0), H1=(−rc,−rb,0), H2=(rc,−rb,0) in this frame.
   Build (X', Y', Z') from A0,B0,C0.

   # Express the COM-relative unconstrained positions in the primed frame and
   # solve, in closed form, for the three rotation angles — ψ and φ about the
   # in-plane axes (tilting the rigid plane onto the displaced positions) and
   # θ about Z' (the in-plane rotation that conserves the molecule's
   # orientation about its normal). See Miyamoto & Kollman 1992, Eqs. A1–A15;
   # the GROMACS `csettle` formulation is the numerically-stable reference.

   A3,B3,C3 ← canonical triangle rotated by (ψ, φ, θ), transformed back to the
              lab frame, and translated to com.
   ```

   Every `sqrt`/`asin`/`acos` argument in the closed form is clamped to its
   valid domain (`≥ 0`, or `[−1, 1]`) so that f32 round-off near a planar or
   extreme instantaneous geometry cannot produce a `NaN`; no iterative
   fall-back is used. The reset preserves the group's mass-weighted centre of
   mass exactly and restores all three intramolecular distances to f32
   round-off in a single evaluation — there is no iteration and no convergence
   tolerance. After the reset, the half-step velocity of every atom is updated
   by `v_i ← v_i + (r_i^constrained − r_i^unconstrained) / dt`, the constrained
   positions are written back to `positions_*` in the same lattice image the
   atom occupied before the call, and the per-atom position-level
   constraint-virial contribution is written into
   `constraint_virial[base + i]` as
   `(m_i / dt²) · ((r_i^constrained − r_i^unconstrained) · r_i^COM)`, where
   `r_i^COM = r_i^constrained − com`.

3. `apply_after_kick` invokes `settle_velocities`, which projects the
   post-kick velocities onto the velocity manifold of the constrained
   positions in closed form:

   ```text
   A, B, C ← positions[group atoms]   (already constrained, on the manifold)
   Bring B, C into A's lattice image.
   r_OH1 ← A − B ;  r_OH2 ← A − C ;  r_HH ← B − C    # current bond vectors
   The relative velocity along every bond must vanish:
     (v_A − v_B) · r_OH1 = 0
     (v_A − v_C) · r_OH2 = 0
     (v_B − v_C) · r_HH  = 0
   Corrections apply equal-and-opposite impulses along the bond directions
   with multipliers (g1, g2, g3):
     v_A += ( g1 · r_OH1 + g2 · r_OH2) / m_O
     v_B += (−g1 · r_OH1 + g3 · r_HH ) / m_H
     v_C += (−g2 · r_OH2 − g3 · r_HH ) / m_H
   Substituting into the three constraint equations yields a 3×3 linear
   system in (g1, g2, g3); SETTLE solves it directly (Miyamoto & Kollman
   1992, velocity section) — no iteration.
   ```

   When `dt > 0` (the standard post-kick call from the integrator) the kernel
   additionally accumulates the velocity-level constraint-virial contribution
   into `constraint_virial[base + i]` (additive on top of the position-level
   half already written by `settle_positions`). The per-atom velocity-level
   virial is `m_i · Δv_i · r_i^COM / dt`, where `Δv_i` is the velocity
   correction applied to atom `i`. When `dt ≤ 0` (the runner's
   initial-velocity projection at setup, which has no associated timestep)
   the virial accumulation is skipped.

4. `apply_position_projection_only` invokes `settle_positions_no_velocity`,
   used by the minimizer after each trial position update. The M-K rotation is
   an integration-time operation — it needs a pre-drift reference frame, which
   minimization does not provide, and a snap-to-rigid is not a
   descent-compatible retraction for a line search. This hook therefore
   performs the minimal-displacement projection onto the rigid manifold: a
   deterministic, bounded Gauss-Seidel sweep of the three water constraints
   with the constraint-gradient directions evaluated at the current
   (off-manifold) positions.

   ```text
   r_i ← positions[a_i]   for i in {O, H1, H2}
   Bring the hydrogens into the oxygen's lattice image (minimum image).
   for each constraint k in [(O,H1), (O,H2), (H1,H2)]:
       g_k ← r_i − r_j                                    # current gradient direction
   targets from the canonical geometry: d_OH² = rc² + (ra+rb)², d_HH² = (2·rc)²
   Gauss-Seidel sweep over the three constraints in the fixed order above
   (≤ 32 sweeps; converges in 1–3): for each k with
   |σ_k| = ||r_i − r_j|² − d_k²| above SETTLE_TOL² (= 3.57e-6 a₀²),
       λ_k ← σ_k / (2 · (r_i − r_j)·g_k · (1/m_i + 1/m_j))
       r_i ← r_i − λ_k g_k / m_i ;  r_j ← r_j + λ_k g_k / m_j
   ```

   It modifies neither velocities nor the constraint-virial buffer, and is
   bit-exact identity for a molecule that already satisfies its constraints
   (the `σ` gate skips every constraint) — which is what lets the
   steepest-descent line search make progress: the retraction of a small
   downhill step stays a small downhill step.

### Exactness and determinism <!-- rq-fa14a87f -->

The MD position reset and the velocity reset are both closed-form and
non-iterative: the position reset is the M-K rotation (a fixed sequence of
arithmetic, with `sqrt`/`asin`/`acos` arguments clamped to their valid
domains), and the velocity reset is a single direct 3×3 solve. Both restore
their respective constraints to f32 round-off in one evaluation — no
convergence loop and no tolerance parameter. The minimizer's position-only
projection is the one iterative path: a deterministic Gauss-Seidel sweep
bounded at 32 iterations that converges in 1–3 sweeps for the small per-step
displacements of steepest descent.

Every per-group computation depends only on the group's data and no
thread-scheduling decision (the velocity solve and the minimizer projection
walk the three constraints in the fixed order `(O–H1, O–H2, H1–H2)`), so two
runs on the same GPU with identical inputs produce byte-identical
`positions_*`, `velocities_*`, and `constraint_virial` after every hook (see
*Reproducibility*). The M-K reset is not equivalent bit-for-bit to a
SHAKE-style minimal projection of the same inputs — it is a different f32
arithmetic path — but it reaches the same constrained configuration to f32
round-off.

## Per-Step Kernel Sequence <!-- rq-0fe74db3 -->

For each `Constraint` hook called by the runner on a step where the slot
contains at least one group:

| Hook | Kernels launched (in order) | Notes |
|---|---|---|
| `apply_before_drift` | `settle_snapshot` | Reads current `positions_*`; writes `snapshot_*`. |
| `apply_after_drift` | `settle_positions` | Reads `snapshot_*`, current `positions_*`, current `velocities_*`. Writes reset `positions_*`, updated `velocities_*`, position-level half of `constraint_virial`. |
| `apply_after_kick` | `settle_velocities`, `settle_virial_scatter` | Reads current constrained `positions_*` and post-kick `velocities_*`. Writes velocity-reset `velocities_*`, accumulates velocity-level half of `constraint_virial`, scatters `constraint_virial` into `particle_virials` for the barostat to consume. |
| `apply_position_projection_only` | `settle_positions_no_velocity` | Reads and writes `positions_*` only; does not touch velocities or virials. |

When the slot has zero groups (a topology with no SETTLE water groups, or a
config that omits `[constraints]` entirely), every hook is a no-op and no
kernel is launched.

## Constraint Virial <!-- rq-93f8e094 -->

The per-atom constraint-virial contribution for atom `i` of group `g` is the
sum of a position-level and a velocity-level half:

```
W_i^position = (m_i / dt²) · ((r_i^constrained − r_i^unconstrained) · r_i^COM)
W_i^velocity = m_i · Δv_i · r_i^COM / dt
W_i          = W_i^position + W_i^velocity
```

where `r_i^COM = r_i^constrained − r^COM_group`, `r^COM_group` is the group's
mass-weighted centre of mass (preserved by both the position and velocity
resets), and `Δv_i` is the velocity correction applied to atom `i` by
`settle_velocities`. The position-level half is written by `settle_positions`
and the velocity-level half is accumulated by `settle_velocities`; the sum
is scattered into the global `particle_virials` array by
`settle_virial_scatter`. The barostat's scalar-virial reduction then sums
`particle_virials` across all atoms; the result is the analytic `−2 K_rot` of
a rigid rotor for each group (see `berendsen-barostat.md`,
`c-rescale-barostat.md`).

The arithmetic uses centre-of-mass-relative positions (rather than lab-frame
absolute positions) for f32 stability. At molecular-cluster scales
(`|r_i| ≈ 10⁻⁹ m` from the origin in large boxes, `|Δr_i| ≈ 10⁻¹² m`,
`1/dt² ≈ 10³⁰ s⁻²`), a direct `m · Δr · r` evaluation underflows in f32. The
`(m / dt²) · (Δr · r^COM)` regrouping keeps every intermediate well inside
f32 normal range — the same treatment SHAKE uses (see `shake.md`).

## Reproducibility <!-- rq-73497463 -->

The SETTLE reset is deterministic per group: each group's reset is a fixed
arithmetic sequence over its own data, run by one thread, on the device's
default stream with all kernel launches in a single fixed order. Two
independent runs on the same GPU with identical inputs produce byte-identical
`positions_*`, `velocities_*`, and `constraint_virial` after every hook. The
kernels use `nvcc`'s default FMA contraction and its `sqrt` / inverse-trig
implementations; the guarantee is GPU-vs-GPU on the same hardware, not
GPU-vs-CPU (see `docs/architecture.md`).

Group order in the device-side group buffers matches `groups` in the
host-side `ConstraintList`, which is sorted by each group's minimum particle
index, so every kernel processes groups in the same order across runs. The
constraint slot writes only to particle-state buffers owned by the default
stream and composes with the SPME `recip_stream` without extra
synchronisation, matching `forces/framework.md`'s default-stream convention.

## Per-Group Storage <!-- rq-49fe5f2e -->

SETTLE's per-group working set is tiny — three position triples, three
snapshot triples, the two masses `(m_O, m_H)`, and the canonical
`(ra, rb, rc)` — so each kernel runs one thread per group with the per-group
state held in registers. No shared-memory staging is used (unlike the general
SHAKE/RATTLE kernels): with only three atoms and three constraints the
per-thread footprint is far inside the register budget, which is the main
source of SETTLE's per-step speedup over the general algorithm.

## Parameters <!-- rq-eb01c35e -->

Per-constraint-type parameters are declared in the config's
`[[constraint_types]]` table with `kind = "settle"` (see
`io/config-schema.md`). One entry per distinct rigid water shape:

```toml
[[constraint_types]]
name = "SPCE"
kind = "settle"
d_OH = 1.0e-10        # apex–hydrogen bond length (metres, SI input)
d_HH = 1.633e-10      # hydrogen–hydrogen distance (metres, SI input)
```

Fields:

- `d_OH: f64` — apex–hydrogen bond length. Finite and strictly positive.
- `d_HH: f64` — hydrogen–hydrogen distance. Finite and strictly positive,
  and strictly less than `2 · d_OH` (so the apex height `h` is real and
  positive and the three atoms are not collinear).

Both distances follow the config's `units` selector (`io/unit-system.md`):
SI input is in metres and converted to Bohr (`a_0`) at config load; the
internal pipeline sees atomic units only.

A `[constraints]` row that references a SETTLE type lists exactly three
global atom indices in canonical order — the oxygen (apex) first, then the
two hydrogens — followed by the constraint-type name:

```
atom_O atom_H1 atom_H2 SPCE
```

The topology parser expands one such row into a `ConstraintGroup` with
`atom_count = 3` and `constraint_count = 3`, synthesising the canonical water
constraint pattern from the type's distances: `GroupConstraint`s
`{(0, 1, d_OH), (0, 2, d_OH), (1, 2, d_HH)}` (local-index pairs, `r0` in
Bohr). These three pairs populate the implicit intra-group exclusions
`(O, H1)`, `(O, H2)`, `(H1, H2)` and the `3 · n_settle_groups` constraint
count exactly as for any other constraint algorithm (see
`integration/constraint-framework.md` and `forces/topology.md`). The SETTLE
kernels do not consume the per-pair `r0` values directly — they use the
canonical `(ra, rb, rc)` geometry computed from `d_OH`, `d_HH`, and the
masses — but the synthesised pairs make a SETTLE group indistinguishable from
a SHAKE water group to the framework's exclusion, virial, and DOF machinery.

## Feature API <!-- rq-8e075e86 -->

### Types <!-- rq-55a683b8 -->

- `SettleParams` — typed deserialiser for the `[[constraint_types]]` entry <!-- rq-52204841 -->
  whose `kind == "settle"`. Fields:
  - `d_OH: f64` — apex–hydrogen bond length in metres.
  - `d_HH: f64` — hydrogen–hydrogen distance in metres.

- `SettleConstraintsState` — implements the `Constraint` trait. Fields <!-- rq-81b3bcaf -->
  private; public surface is the `Constraint` trait methods plus the
  constructor.

  Constructor:
  - `SettleConstraintsState::new(device: Arc<CudaDevice>, list: &ConstraintList, masses: &[f32], constraint_types: &[NamedSlotConfig]) -> Result<SettleConstraintsState, SettleError>`
    - Filters `constraint_types` to entries with `kind == "settle"`. For each,
      deserialises `SettleParams` from `entry.params` and verifies
      `d_OH` and `d_HH` are finite and strictly positive and that
      `d_HH < 2 · d_OH`.
    - Walks `list.groups`. For each group whose constraint type resolves to
      `"settle"`, copies the group's three atom indices into a flat device
      array `group_atoms`, records per-group offsets and counts, reads the
      apex mass `m_O = masses[atom 0]` and hydrogen mass
      `m_H = masses[atom 1]` (requiring `masses[atom 1] == masses[atom 2]`),
      and computes and stores the canonical geometry `(ra, rb, rc)` and the
      masses `(m_O, m_H)` per group on the device.
    - Allocates the per-atom-slot `snapshot_*` and `constraint_virial`
      buffers, sized at the total atom-slot count (`group_atoms.len()`).
    - When `list.is_empty()` or the list contains no `"settle"` groups,
      returns an empty slot whose hooks are all no-ops.
  - Returns `SettleError` on a CUDA driver failure, on a malformed
    `[[constraint_types]]` entry, or on a group whose shape is not symmetric
    three-atom water.

- `SettleError` — error type returned by `SettleConstraintsState::new`. <!-- rq-64ddcf84 -->
  Variants:
  - `Gpu(GpuError)` — a CUDA driver / kernel-launch failure during setup.
  - `Timings(TimingsError)` — a timings-system failure during setup.
  - `MalformedSettleType { name: String, reason: String }` — the named
    constraint type's `params` failed to deserialise as `SettleParams`, or
    violated a bound (`d_OH` or `d_HH` non-finite or non-positive, or
    `d_HH >= 2 · d_OH` so the geometry is degenerate / collinear).
  - `InvalidGroupShape { group_index: usize, reason: String }` — the group is
    not symmetric three-atom water: its atom count is not 3, or its two
    hydrogen masses differ (`masses[atom 1] != masses[atom 2]`). The message
    directs the user to SHAKE for general rigid clusters.

  Converts into `ConstraintError` via a `From` impl
  (`InvalidGroupShape` maps to `ConstraintError::InvalidGroupShape` with
  `kind = "settle"`).

### CUDA Kernels <!-- rq-f7521c63 -->

`kernels/settle.cu` declares five `extern "C"` kernels. Each runs one thread
per group (`g = blockIdx.x * blockDim.x + threadIdx.x`; threads past
`n_groups` return without touching any buffer), block size 256, grid size
`ceil(n_groups / 256)`, zero shared memory, on the default stream carried by
`device`. Positions are read from and written to the packed `Real4 *posq`
array (`{x, y, z, charge}` per particle, matching the rest of the engine);
velocities are the separate per-component arrays. `Real` is `float` by
default and `double` under the `f64` feature (see `kernels/precision.cuh`).

```c
extern "C" __global__ void settle_snapshot(
    const Real4 *posq,
    const unsigned int *group_atoms,
    const unsigned int *group_atom_offset,
    const unsigned int *group_atom_count,
    Real *snapshot_x,
    Real *snapshot_y,
    Real *snapshot_z,
    unsigned int n_groups);

extern "C" __global__ void settle_positions(
    Real4 *posq,
    Real *velocities_x,
    Real *velocities_y,
    Real *velocities_z,
    const Real *snapshot_x,
    const Real *snapshot_y,
    const Real *snapshot_z,
    const unsigned int *group_atoms,
    const unsigned int *group_atom_offset,
    const unsigned int *group_atom_count,
    const Real *group_ra,             // canonical geometry, per group
    const Real *group_rb,
    const Real *group_rc,
    const Real *group_m_o,            // apex mass, per group
    const Real *group_m_h,            // hydrogen mass, per group
    const Real *lattice,              // length 6: [lx, ly, lz, xy, xz, yz]
    Real dt,
    Real *constraint_virial,
    unsigned int n_groups);

extern "C" __global__ void settle_velocities(
    const Real4 *posq,
    Real *velocities_x,
    Real *velocities_y,
    Real *velocities_z,
    const unsigned int *group_atoms,
    const unsigned int *group_atom_offset,
    const unsigned int *group_atom_count,
    const Real *group_m_o,
    const Real *group_m_h,
    const Real *lattice,              // length 6: [lx, ly, lz, xy, xz, yz]
    Real dt,
    Real *constraint_virial,
    unsigned int n_groups);

extern "C" __global__ void settle_virial_scatter(
    const Real *constraint_virial,
    const unsigned int *group_atoms,
    Real *particle_virials,
    unsigned int n_atom_slots);

extern "C" __global__ void settle_positions_no_velocity(
    Real4 *posq,
    const unsigned int *group_atoms,
    const unsigned int *group_atom_offset,
    const unsigned int *group_atom_count,
    const Real *group_ra,
    const Real *group_rb,
    const Real *group_rc,
    const Real *group_m_o,
    const Real *group_m_h,
    const Real *lattice,              // length 6: [lx, ly, lz, xy, xz, yz]
    unsigned int n_groups);
```

### PTX Module Loading <!-- rq-4b4fa0d3 -->

`init_device()` loads the compiled `kernels/settle.cu` PTX as module
`"settle"` and captures all five kernel functions into a `SettleKernels`
handle held on the `Kernels` struct (see `build-pipeline.md`).

### Builder <!-- rq-0063f0ce -->

- `SettleBuilder` — implements `ConstraintBuilder` (see <!-- rq-70901353 -->
  `integration/constraint-framework.md`). Methods:
  - `kind_name() -> &'static str` returns `"settle"`.
  - `validate_params(&self, params: &toml::Value) -> Result<(), ConfigError>`
    deserialises `params` as `SettleParams` and checks the `d_OH` / `d_HH`
    bounds (finite, strictly positive, `d_HH < 2 · d_OH`). A failure surfaces
    as a `ConfigError` pointing at the offending entry.
  - `expected_atom_count(&self, _params: &toml::Value) -> usize` returns `3`.
  - `validate_group_shape(&self, group_index, atoms, constraints, params, masses) -> Result<(), ConstraintError>`
    requires the group's atom count to be 3 and the two hydrogen masses
    (`masses[atoms[1]]`, `masses[atoms[2]]`) to be equal. On failure returns
    `ConstraintError::InvalidGroupShape { group_index, kind: "settle", reason }`.
  - `supports_position_projection_only(&self, _params: &toml::Value) -> bool`
    returns `true` — SETTLE participates in minimization phases via
    `settle_positions_no_velocity`.
  - `graph_compatible(&self, _params: &toml::Value) -> bool` returns `true` —
    every hook is a pure sequence of kernel launches with no host-side state
    mutation and no host/device synchronisation between launches (see
    `cuda-graphs.md`).
  - `build(&self, gpu, particle_count, list, masses, constraint_types) -> Result<Box<dyn Constraint>, ConstraintError>`
    constructs a `SettleConstraintsState` from the sub-list it receives, wraps
    it in a `Box`, and returns it, forwarding any `SettleError` through
    `From<SettleError> for ConstraintError`.

  The topology parser obtains a SETTLE group's three canonical
  `GroupConstraint`s — `{(0, 1, d_OH), (0, 2, d_OH), (1, 2, d_HH)}` — from the
  type's `d_OH` / `d_HH` rather than from a `constraints` table (a SETTLE
  type declares none); see *Parameters* and `forces/topology.md`.

  `ConstraintRegistry::with_builtins()` registers this builder under
  `"settle"`, alongside the `"shake"` builder.

## Empty State <!-- rq-993f8c4c -->

When the slot has zero groups, every hook returns `Ok(())` without launching
any kernel and without modifying any buffer. The `constraint_virial` buffer
has length zero. The runner's barostat machinery reads `particle_virials`
regardless and sees the constraint slot's zero contribution naturally.

## Out of Scope <!-- rq-6ff885f2 -->

- Asymmetric or non-water three-atom rigid clusters (unequal hydrogen masses
  or unequal apex-bond lengths). SETTLE handles only symmetric water; SHAKE
  (`shake.md`) is the general rigid-cluster algorithm.
- Rigid clusters of more than three atoms. Handled by SHAKE within its
  per-group caps, and by future M-SHAKE above them.
- Four-site water models' virtual / massless sites (e.g. TIP4P M-site). The
  three constrained atoms are the oxygen and two hydrogens; massless
  interaction sites are placed by a separate virtual-site feature, not by
  SETTLE.
- Flexible / bonded water (treated by the bonded-force and angle kernels, not
  by `Constraint`).
- Constraint forces resolved into per-atom Cartesian components for
  visualisation; the only constraint quantity consumed downstream is the
  scalar `constraint_virial` per group.

---

## Gherkin Scenarios <!-- rq-0098a282 -->

```gherkin
Feature: SETTLE rigid-water constraints

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called

  # --- Slot construction and parameter validation ---

  @rq-a8a99082
  Scenario: SettleConstraintsState::new with a single SPC/E water group succeeds
    Given a ConstraintList with one group of 3 atoms in canonical order (O, H1, H2)
    And a [[constraint_types]] entry with name "SPCE", kind "settle",
      d_OH = 1.0e-10, d_HH = 1.633e-10
    And per-atom masses (m_O, m_H, m_H) for the group's three atoms
    When SettleConstraintsState::new is called
    Then it returns Ok(state)
    And state.group_count is 1
    And state.group_atoms on the device equals [atom_O, atom_H1, atom_H2]

  @rq-9f211910
  Scenario: SettleConstraintsState::new computes the canonical geometry from masses and distances
    Given a [[constraint_types]] entry with kind "settle", d_OH and d_HH
    And apex mass m_O and hydrogen mass m_H
    When SettleConstraintsState::new is called
    Then the stored per-group geometry satisfies rc == d_HH / 2
    And ra == (2 * m_H / (m_O + 2 * m_H)) * sqrt(d_OH^2 - (d_HH/2)^2)
    And rb == (m_O / (m_O + 2 * m_H)) * sqrt(d_OH^2 - (d_HH/2)^2)
    And m_O * ra is within f32 round-off of 2 * m_H * rb

  @rq-6bef53e7
  Scenario: SettleConstraintsState::new with an empty constraint list succeeds and is a no-op
    Given a ConstraintList with zero groups
    When SettleConstraintsState::new is called
    Then it returns Ok(state)
    And state.group_count is 0
    And every subsequent hook invocation launches no kernels and returns Ok(())

  @rq-73c1173f
  Scenario: SettleConstraintsState::new rejects a non-positive d_OH
    Given a [[constraint_types]] entry with name "bad", kind "settle",
      d_OH = 0.0, d_HH = 1.633e-10
    When SettleConstraintsState::new is called
    Then it returns Err(SettleError::MalformedSettleType { name: "bad", reason: r })
      where r mentions `d_OH must be strictly positive`

  @rq-d2976a76
  Scenario: SettleConstraintsState::new rejects a degenerate (collinear) geometry
    Given a [[constraint_types]] entry with name "flat", kind "settle",
      d_OH = 1.0e-10, d_HH = 2.0e-10
    When SettleConstraintsState::new is called
    Then it returns Err(SettleError::MalformedSettleType { name: "flat", reason: r })
      where r mentions `d_HH must be less than 2 * d_OH`

  @rq-b98dcdd2
  Scenario: SettleConstraintsState::new rejects a group with the wrong atom count
    Given a ConstraintList in which a "settle" group has 4 atoms
    When SettleConstraintsState::new is called
    Then it returns Err(SettleError::InvalidGroupShape { group_index: 0, reason: r })
      where r mentions `atom count` and references SHAKE

  @rq-56aeb344
  Scenario: SettleConstraintsState::new rejects a group with unequal hydrogen masses
    Given a ConstraintList with one "settle" group whose two hydrogens have
      different masses (masses[atom 1] != masses[atom 2])
    When SettleConstraintsState::new is called
    Then it returns Err(SettleError::InvalidGroupShape { group_index: 0, reason: r })
      where r mentions `equal hydrogen masses` and references SHAKE

  # --- Topology expansion ---

  @rq-0d91d811
  Scenario: A settle [constraints] row expands into three canonical water constraints
    Given a topology file with one row "0 1 2 SPCE" and a "settle" type
      with d_OH and d_HH
    When load_topology_file(...) is called
    Then the group has constraint_count == 3
    And group_constraints contains (0, 1, d_OH), (0, 2, d_OH), (1, 2, d_HH)
      (local pairs, r0 in Bohr)

  @rq-6de1f8d5
  Scenario: A settle group adds implicit exclusions for every intra-group pair
    Given a topology file with no [exclusions] section and one row "0 1 2 SPCE"
      referencing a "settle" type
    When load_topology_file(...) is called
    Then exclusion_list.entries contains (0, 1, 0.0, 0.0)
    And exclusion_list.entries contains (0, 2, 0.0, 0.0)
    And exclusion_list.entries contains (1, 2, 0.0, 0.0)

  @rq-1ea7342a
  Scenario: A settle [constraints] row with the wrong atom count is rejected by the parser
    Given a topology file with a row of 2 atoms referencing a "settle" type
    When load_topology_file(...) is called
    Then it returns Err pointing at the row with a reason naming the expected count 3

  # --- Position reset (SETTLE) ---

  @rq-d5b31775
  Scenario: settle_positions restores constraint distances after a small uniform translation
    Given a constructed SettleConstraintsState with one SPC/E water group at equilibrium
    And the unconstrained post-drift positions are the equilibrium positions shifted
      uniformly by 1.0e-3 nm along x
    When apply_after_drift is called with dt = 2.0e-15 s
    Then each constraint distance (O-H1, O-H2, H1-H2) equals its target to within 1.0e-13 m
    And the centre of mass of the three atoms equals the unconstrained centre of mass to
      within f32 round-off

  @rq-603bd03b
  Scenario: settle_positions restores constraint distances after a small per-atom kick
    Given a constructed SettleConstraintsState with one SPC/E water group at equilibrium
    And the unconstrained post-drift positions perturb each atom independently by ~1.0e-12 m
    When apply_after_drift is called with dt = 2.0e-15 s
    Then every constraint distance is within 1.0e-13 m of its target

  @rq-3bd5ee23
  Scenario: settle_positions updates half-step velocities consistently with the position correction
    Given a constructed SettleConstraintsState with one SPC/E water group
    And initial velocities v_i^pre and unconstrained post-drift positions r_i^unconstrained
    When apply_after_drift is called with dt = 2.0e-15 s
    Then the post-call velocity for every atom equals v_i^pre + (r_i^constrained - r_i^unconstrained) / dt
      to within f32 round-off

  @rq-95be04ff
  Scenario: settle_positions writes a non-zero position-level constraint virial
    Given a constructed SettleConstraintsState with one SPC/E water group whose
      unconstrained post-drift positions break every constraint by ~1.0e-12 m
    When apply_after_drift is called with dt = 2.0e-15 s
    Then constraint_virial on the device contains three nonzero entries for this group

  @rq-eba5c2ff
  Scenario: settle_positions handles a water group straddling a periodic boundary
    Given a constructed SettleConstraintsState with one SPC/E water group
    And a small orthorhombic simulation box (Lx = Ly = Lz = 10.0 a_0)
    And pre-drift positions placing the O atom near +Lx/2 and the two H atoms near -Lx/2,
      so that the molecule straddles the +x periodic boundary
    And unconstrained post-drift positions perturbing the O-H1 bond by ~1.0e-2 a_0
      along the O->H1 direction (computed under minimum image)
    When apply_after_drift is called with dt = 82.68 atu (~ 2 fs)
    Then every constraint distance, computed under minimum image, equals its target r0
      to within 1.0e-4 a_0 relative
    And the per-atom global positions remain in the same lattice image they occupied
      before the call (no spurious wrap of any atom)
    And the mass-weighted centre of mass, computed by bringing every atom into atom 0's
      image, equals the COM of the unconstrained post-drift positions to within 1.0e-3 a_0

  @rq-9638eee5
  Scenario: settle_positions converges at production-scale dt with thermal-amplitude displacements
    Given a constructed SettleConstraintsState with one SPC/E water group at equilibrium
    And per-atom velocities sampled from a Maxwell-Boltzmann distribution at T = 300 K
    And the unconstrained post-drift positions are r_i^pre + v_i * dt
    When apply_after_drift is called with dt = 2.0e-15 s
    Then every constraint distance is within 1.0e-13 m of its target
    And the half-step velocity update remains finite for every atom

  # --- M-K closed-form exactness ---

  @rq-76abf8d7
  Scenario: settle_positions restores exact rigidity in a single evaluation
    Given a constructed SettleConstraintsState with one SPC/E water group
    And unconstrained post-drift positions that break every constraint by ~1.0e-2 a_0
    When apply_after_drift is called once with dt = 2.0e-15 s
    Then each constraint distance equals its canonical target to within 1.0e-6 relative
      (the closed-form rotation, not an iterative tolerance)
    And the mass-weighted centre of mass equals the unconstrained centre of mass to
      within f32 round-off

  @rq-3163a55d
  Scenario: M-K reset reaches the same configuration as a converged minimal-displacement solve
    Given one SPC/E water group, a pre-drift snapshot, and unconstrained post-drift
      positions that break every constraint by ~1.0e-2 a_0
    And the constrained positions C_settle produced by apply_after_drift
    And the constrained positions C_ref produced by a fully-converged Gauss-Seidel
      minimal-displacement projection of the same unconstrained positions using the
      same snapshot bond directions
    Then C_settle and C_ref agree to within 1.0e-4 a_0 relative for every atom

  @rq-28cc7d41
  Scenario: settle_positions guards a near-degenerate instantaneous geometry
    Given one SPC/E water group whose pre-drift snapshot is rigid
    And unconstrained post-drift positions that are nearly collinear (the three atoms
      within 1.0e-3 a_0 of a line) after a large drift
    When apply_after_drift is called with dt = 2.0e-15 s
    Then every output position is finite (no NaN or infinity from a clamped
      sqrt/asin/acos argument)
    And every constraint distance equals its canonical target to within 1.0e-4 a_0 relative

  # --- Velocity reset (SETTLE velocity) ---

  @rq-9dd716cf
  Scenario: settle_velocities zeroes the constraint-distance time-derivative
    Given a constructed SettleConstraintsState with one SPC/E water group at equilibrium
    And post-kick velocities with non-trivial v_rel . d for every constraint
    When apply_after_kick is called with dt = 2.0e-15 s
    Then for each constraint (i, j) the post-call (v_i - v_j) . (r_i - r_j) is within
      1.0e-20 m^2/s of zero

  @rq-38d09177
  Scenario: settle_velocities preserves the centre-of-mass velocity
    Given a constructed SettleConstraintsState with one SPC/E water group
    And the post-kick velocities have a known mass-weighted COM velocity v_COM
    When apply_after_kick is called with dt = 2.0e-15 s
    Then the post-call mass-weighted COM velocity equals v_COM to within f32 round-off

  @rq-b317db56
  Scenario: settle_velocities accumulates a velocity-level constraint virial when dt > 0
    Given a constructed SettleConstraintsState with one SPC/E water group
    And constraint_virial entries containing the position-level half (a known
      non-zero pattern from a prior settle_positions call)
    When apply_after_kick is called with dt = 2.0e-15 s
    Then constraint_virial after the call equals the position-level half plus the
      velocity-level half, with the velocity-level half computed as
      m_i * Delta_v_i * r_i^COM / dt

  @rq-b7ed6d52
  Scenario: settle_velocities skips the velocity-level virial accumulation when dt <= 0
    Given a constructed SettleConstraintsState with one SPC/E water group
    And a stale constraint_virial pattern X on the device
    When apply_after_kick is called with dt = 0.0
    Then constraint_virial after the call is byte-identical to X
    And the post-call relative velocities along every bond are within 1.0e-20 m^2/s of zero

  # --- Virial scatter ---

  @rq-9883ee42
  Scenario: settle_virial_scatter additively writes per-atom virial into particle_virials
    Given a constructed SettleConstraintsState with one SPC/E water group
    And constraint_virial contains [w0, w1, w2] for the three atoms of the group
    And particle_virials on the device is initialised to [0; N]
    When settle_virial_scatter is launched
    Then particle_virials[atom_O] equals w0
    And particle_virials[atom_H1] equals w1
    And particle_virials[atom_H2] equals w2
    And every other particle_virials[a] is unchanged

  @rq-a8d63610
  Scenario: settle_virial_scatter handles two disjoint groups
    Given two SPC/E water groups with disjoint atom sets
    And constraint_virial contains six entries [w0_g0, w1_g0, w2_g0, w0_g1, w1_g1, w2_g1]
    When settle_virial_scatter is launched
    Then particle_virials accumulates each group's three contributions into the
      corresponding three atom slots, with no cross-group interference

  # --- Position-only projection (minimization) ---

  @rq-bab6638a
  Scenario: settle_positions_no_velocity restores constraint distances from off-manifold positions
    Given a constructed SettleConstraintsState with one SPC/E water group whose
      current positions break every constraint by ~5.0e-12 m
    When apply_position_projection_only is called
    Then every constraint distance is within 1.0e-13 m of its target
    And velocities on the device are unchanged byte-for-byte from before the call
    And constraint_virial on the device is unchanged byte-for-byte from before the call

  @rq-9e22adb3
  Scenario: settle_positions_no_velocity is a no-op for an already-rigid molecule
    Given a constructed SettleConstraintsState with one SPC/E water group whose current
      positions exactly satisfy the canonical geometry
    When apply_position_projection_only is called
    Then positions on the device are byte-identical to before the call
    (this is the property the steepest-descent line search relies on: the projection of
     an already-converged trial step does not perturb the geometry)

  @rq-261b5b46
  Scenario: A steepest-descent minimization with a SETTLE slot converges
    Given a runner configured with a [[minimization]] phase and a SETTLE constraint slot
      over a small rigid-water system started from a near-equilibrium lattice
    When the minimization phase runs
    Then it converges within its iteration budget (the line search does not collapse to a
      zero step)
    And every constraint distance is within 1.0e-4 a_0 relative of its target afterwards

  # --- Reproducibility ---

  @rq-cc3f19a3
  Scenario: Two independent runs of apply_after_drift produce byte-identical state
    Given two SettleConstraintsState instances A and B with identical inputs
    And identical unconstrained post-drift positions in both
    When apply_after_drift is called on each with dt = 2.0e-15 s
    Then run A and run B agree byte-for-byte on positions_x, positions_y, positions_z,
      velocities_x, velocities_y, velocities_z, and constraint_virial

  @rq-c90e3e46
  Scenario: Two independent runs of apply_after_kick produce byte-identical state
    Given two SettleConstraintsState instances A and B with identical inputs
    And identical post-kick velocities and identical constrained positions in both
    When apply_after_kick is called on each with dt = 2.0e-15 s
    Then run A and run B agree byte-for-byte on velocities_x, velocities_y, velocities_z,
      and constraint_virial

  # --- Composition with the integrator framework ---

  @rq-102b4b02
  Scenario: A full velocity-Verlet timestep with one rigid SPC/E group preserves all three
    constraint distances
    Given a runner with a velocity-verlet integrator (lossless=false) and a
      SettleConstraintsState slot containing one SPC/E water group at equilibrium
      with thermal velocities
    When the runner runs N = 100 timesteps at dt = 2.0e-15 s
    Then for every step n in 0..N the post-step constraint distances are within
      1.0e-13 m of their targets

  @rq-c129167f
  Scenario: A run with no Constraint slot active leaves constraint distances drifting
    Given a runner with a velocity-verlet integrator and NO constraint slot
    And a SPC/E water-shaped topology
    When the runner runs N = 100 timesteps at dt = 2.0e-15 s
    Then the constraint distances drift away from their targets

  # --- Graph and minimization compatibility ---

  @rq-63926477
  Scenario: The settle builder reports graph compatibility
    Given the registered SETTLE builder and any well-formed settle params
    Then builder.graph_compatible(&params) returns true

  @rq-c5898418
  Scenario: The settle builder supports position-only projection
    Given the registered SETTLE builder and any well-formed settle params
    Then builder.supports_position_projection_only(&params) returns true

  # --- Registry ---

  @rq-f4acb881
  Scenario: ConstraintRegistry::with_builtins registers the settle builder
    Given a ConstraintRegistry::with_builtins()
    When the registry is queried for kind "settle"
    Then it returns the SETTLE builder
    And the registry also returns the SHAKE builder for kind "shake"
```
