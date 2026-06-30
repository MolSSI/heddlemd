# Feature: Harmonic Angle Bonded Potential <!-- rq-d9adc4cb -->

The `HarmonicAngle` potential slot evaluates a harmonic angle force for each
angle in the system's angle list (see `topology.md`). Angles are triples
of atoms `(i, j, k)` whose vertex geometry at `j` is described by the
harmonic functional form `U(θ) = ½ k (θ − θ₀)²` with per-angle-type
parameters. The slot plugs into the pluggable potential framework
(`framework.md`); selection is implicit — the slot is present whenever the
config's `topology` field references a non-empty `.topology` file and at
least one `[[angle_types]]` entry has `potential = "harmonic"`.

## Algorithm <!-- rq-d12b8b49 -->

The harmonic potential at angle `θ` formed at atom `j` between rays
`r_ij = r_i − r_j` and `r_kj = r_k − r_j` is

```text
U(θ) = (1/2) · k · (θ − θ₀)²
```

where the angle-type parameters are `k` (force constant, E_h/rad²) and
`θ₀` (equilibrium angle, radians). The minimum-image displacements
`r_ij` and `r_kj` honour periodic boundary conditions; the angle is
*not* truncated by any cutoff distance — harmonic angles are intended
for bonded use where the three atoms remain close to each other.

The force derivation follows the chain rule. With

```text
r_ij = r_i − r_j,         d_ij = |r_ij|
r_kj = r_k − r_j,         d_kj = |r_kj|
cosθ = (r_ij · r_kj) / (d_ij · d_kj)
sinθ = sqrt(max(0, 1 − cosθ²))
```

and `f = −dU/dθ = −k · (θ − θ₀)`, the per-atom forces are

```text
F_i = (f / sinθ) · ((cosθ / d_ij²) · r_ij − (1 / (d_ij · d_kj)) · r_kj)
F_k = (f / sinθ) · ((cosθ / d_kj²) · r_kj − (1 / (d_ij · d_kj)) · r_ij)
F_j = −(F_i + F_k)
```

with `θ = atan2(d_ij · d_kj · sinθ, r_ij · r_kj)` to avoid the
single-branch numerical loss of `acos(cosθ)` near `cosθ = ±1`.

The kernel applies the following defensive guards in `f32` arithmetic:

- When `d_ij == 0` or `d_kj == 0`: all three force vectors and the
  per-atom energy and virial slots are zero.
- When `sinθ` evaluates to `< 1.0e-7f`: all three force vectors and the
  per-atom energy and virial slots are zero. Physically realistic
  configurations never reach this limit because `U(0) = U(π) = (k/2)·θ₀²`
  is finite but the gradient diverges; the guard keeps the kernel
  numerically safe.

For each angle `m` the kernel writes its three per-atom force triples,
its per-atom energy share `U_m / 3`, and its per-atom virial share
`W_m / 3` into consecutive slots `3·m`, `3·m + 1`, `3·m + 2` of the
per-angle scratch buffer:

| slot | atom    |
| ---- | ------- |
| `3·m`     | `atom_i` |
| `3·m + 1` | `atom_j` |
| `3·m + 2` | `atom_k` |

The energy and virial are distributed in thirds (rather than the
half-and-half convention used by bond pairs) so that summing all per-atom
shares for one angle reproduces the angle's full `U_m` and `W_m`. The
angle's scalar virial is

```text
W_m = r_ij · F_i + r_kj · F_k
```

equivalent to `Σ_a r_aj · F_a` over the three atoms with `r_jj = 0`.

## Per-Step Kernel Sequence <!-- rq-7884e3ff -->

The slot's contribution and reduction run once each per step:

| Step | Kernel | Operation | Stage label |
| --- | --- | --- | --- |
| 1 | `heddle_jit_composed_angle_<i>_{f,fev}` | compute forces per angle, write to angle-triple buffer | `JitComposedAngleForce` |
| 2 | `reduce_angle_forces` | per-atom sum of angle contributions, write to slot accumulator | `ReduceAngleForces` |

Step 1 is the JIT-composed angle module's entry point for this
slot (slot index `<i>` is the slot's zero-based position among
active angle slots in canonical slot order; the `_f` vs `_fev`
suffix is selected by the per-step `AggregateLevel`). The
JIT-composed module includes the slot's per-angle harmonic functor
source described in *Source Fragment* below. See
`jit-composed-intramolecular.md` for the composer's contract.

Step 2 runs the standalone `reduce_angle_forces` kernel compiled at
build time. The reduction is shape-universal across angle slots
(any angle potential's per-angle contributions sum into per-atom
forces the same way); it is not part of the JIT module.

The class-combine kernel runs after every slot's reduction. See
`framework.md` for the slot order.

## Force Accumulation <!-- rq-ff895387 -->

The slot owns an `AnglePairBuffer` of length `3 · A` where `A` is the
number of angles. Each slot carries five `f32` quantities: three force
components, third-energy, and third-virial. Slot `3·m + p` (where
`p ∈ {0, 1, 2}`) holds the contribution to the `p`-th atom of angle
`m`, with the atom ordering documented above.

The reduction kernel reads the precomputed `atom_angle_offsets` /
`atom_angle_indices` tables (see `topology.md`) and sums each atom's
contributions in fixed order. For atom `a`, the kernel computes five
sequential left-to-right sums:

```text
slot_force_x[a]  = sum over m in atom_angle_indices[a] of angle_triple_x[m]
slot_force_y[a]  = same with y
slot_force_z[a]  = same with z
slot_energy[a]   = sum over m in atom_angle_indices[a] of angle_triple_energy[m]
slot_virial[a]   = sum over m in atom_angle_indices[a] of angle_triple_virial[m]
```

The `atom_angle_indices` slice for each atom is sorted by underlying
angle index at file-load time, so the summation order is identical
across runs. Each thread maps to one atom; there are no atomics and no
race conditions.

## Parameters <!-- rq-b33243ff -->

Each `[[angle_types]]` entry in the config that uses
`potential = "harmonic"` contributes one row to a per-angle-type
parameter table uploaded to the device:

- `k_theta: f64` — force constant in E_h/rad². Required. Finite and
  strictly positive.
- `theta_0: f64` — equilibrium angle in radians. Required. Finite and
  in `[0, π]`.

The parameter table on the device is two `CudaSlice<f32>` arrays
(`angle_k_theta`, `angle_theta_0`), one per angle type, cast from `f64`
to `f32` at upload time. Each angle carries an `angle_type_index` (see
`topology.md`) into this table.

The only supported `potential` value for angle types is `"harmonic"`;
other values are rejected at config-load time. Future angle potentials
(cosine-harmonic, Urey-Bradley, etc.) add new `potential` values and
reuse the existing `AngleList` / `AnglePairBuffer` / reduction
infrastructure.

## Empty State <!-- rq-d940ac6c -->

When the angle list is empty (`angle_list.is_empty()`), the
`HarmonicAngleState` is not constructed by the `ForceField` and the
slot is absent from the slot list. The framework's combiner handles
slot-presence correctly (see `framework.md`).

When `particle_count == 0`, the angle list must also be empty (the
file parser rejects any angle entry with an out-of-range atom index,
and every index is out of range when `N == 0`). The slot is therefore
not constructed.

## Feature API <!-- rq-19f7ffca -->

### Types <!-- rq-db54cffa -->

- `HarmonicAngleState` — implements the `Potential` trait with <!-- rq-21a8063c -->
  `label() == "harmonic_angle"` (see `framework.md`). Fields:
  - `device: Arc<CudaDevice>`
  - `angles: CudaSlice<u32>` — flat array of `[atom_i, atom_j, atom_k,
    angle_type_index]` quadruples, length `4 · A`, sorted to match
    `AngleList::angles`.
  - `atom_angle_offsets: CudaSlice<u32>` — length `N + 1`.
  - `atom_angle_indices: CudaSlice<u32>` — length `3 · A`.
  - `angle_k_theta: CudaSlice<f32>` — length `n_angle_types`.
  - `angle_theta_0: CudaSlice<f32>` — length `n_angle_types`.
  - `angle_triple_x: CudaSlice<f32>` — length `3 · A`, per-slot force
    x contribution.
  - `angle_triple_y: CudaSlice<f32>` — length `3 · A`.
  - `angle_triple_z: CudaSlice<f32>` — length `3 · A`.
  - `angle_triple_energy: CudaSlice<f32>` — length `3 · A`, per-slot
    third-energy contribution (`U_m / 3`).
  - `angle_triple_virial: CudaSlice<f32>` — length `3 · A`, per-slot
    third-virial contribution (`W_m / 3`).
  - `angle_count: usize`
  - `particle_count: usize`

  All fields private; the slot's public surface is the per-step methods
  invoked by `ForceField::step` (see `framework.md`).

  Constructor:

  - `HarmonicAngleState::new(device: Arc<CudaDevice>, angle_list: &AngleList, angle_types: &[AngleTypeConfig]) -> Result<HarmonicAngleState, GpuError>`
    - Filters `angle_types` to entries with `potential == "harmonic"`
      and uploads their parameters.
    - Uploads `angle_list.angles`, `angle_list.atom_angle_offsets`,
      and `angle_list.atom_angle_indices` to device memory.
    - Allocates the five per-angle `angle_triple_*` buffers (force
      x/y/z, third-energy, third-virial), each of length `3 · A`.
      Per-atom output is added into the framework-supplied
      `SlotOutputView` (a view onto the slot's class accumulator; see
      `framework.md`'s *Class Output Accumulators*) during `reduce()`;
      the slot owns no per-atom accumulator buffers of its own.
    - When `angle_list.is_empty()`, this method is not called by the
      `ForceField` — see *Empty State*.

### Source Fragment <!-- rq-4c88ee0e -->

`HarmonicAngleBuilder::angle_force_fragment(cx)` returns an
`AngleForceFragment` whose functor implements the per-angle
harmonic contribution. The fragment defines a `__device__` functor
`HarmonicAngleFunctor` whose member function `evaluate(dx_ij,
dy_ij, dz_ij, dx_kj, dy_kj, dz_kj, angle_type_index, fix, fiy, fiz,
fkx, fky, fkz, u_m, w_m)` computes:

1. Computes `d_ij²`, `d_kj²`, `d_ij`, `d_kj`, `cosθ`, `sinθ`, and
   `θ` using `atan2(d_ij · d_kj · sinθ, r_ij · r_kj)`.
2. Reads `k = angle_k_theta[angle_type_index]` and
   `theta_0 = angle_theta_0[angle_type_index]` from device-buffer
   pointers held as members of the functor.
3. Computes `dθ = θ − theta_0`, `f = −k · dθ`, and `g = f / sinθ`.
4. Computes the two force vectors `F_i` and `F_k` per the formulas
   in *Algorithm* and writes them to `(fix, fiy, fiz)` and
   `(fkx, fky, fkz)`.
5. Writes the angle's full potential energy
   `u_m = 0.5 · k · dθ²` and scalar virial
   `w_m = r_ij · F_i + r_kj · F_k` (the outer-loop body
   distributes the `1/3` symmetry factor when writing to the
   scratch buffer).

When the functor's defensive guards trigger (`d_ij == 0`,
`d_kj == 0`, or `sinθ < 1e-7f`), it writes zero to every output
field. The outer-loop body then writes zeros to the corresponding
fifteen scratch-buffer entries (five quantities × three slots).

The composed kernel's outer-loop body (in the JIT-composed angle
module — see `jit-composed-intramolecular.md`) handles the
common-args reading: reads the angle list `(atom_i, atom_j, atom_k,
angle_type_index)`, computes the minimum-image displacements
`r_ij = r_i − r_j` and `r_kj = r_k − r_j`, calls the functor's
`evaluate`, then writes the three per-atom force triples (the
functor returns `F_i` and `F_k`; the outer-loop body computes
`F_j = -(F_i + F_k)`) along with `u_m / 3` and `w_m / 3` into the
slot's angle-triple scratch buffer at indices `3·m`, `3·m + 1`,
and `3·m + 2`. See `jit-composed-intramolecular.md`'s
*Composed-Module Structure* for the full outer-loop body
specification.

The fragment's `entry_point_args` declares the per-angle-type
parameter table pointers (`angle_k_theta`, `angle_theta_0`); the
`functor_init_source` assigns them to the functor's members at
the start of the entry-point body.

### Reduction Kernel <!-- rq-9d9ca545 -->

`kernels/angle.cu` declares the shape-universal reduction kernel:

```c
extern "C" __global__ void reduce_angle_forces(
    const Real *angle_triple_x, const Real *angle_triple_y, const Real *angle_triple_z,
    const Real *angle_triple_energy, const Real *angle_triple_virial,
    const unsigned int *atom_angle_offsets,
    const unsigned int *atom_angle_indices,
    Real *slot_force_x, Real *slot_force_y, Real *slot_force_z,
    Real *slot_energy, Real *slot_virial,
    unsigned int n);
```

One thread per atom `a = blockIdx.x · blockDim.x + threadIdx.x`
(block size 256, grid `ceil(n / 256)`). Thread `a`:

1. Reads `start = atom_angle_offsets[a]` and `end =
   atom_angle_offsets[a + 1]`.
2. Initialises five running sums to zero: `sum_x`, `sum_y`,
   `sum_z`, `sum_e`, `sum_w`.
3. For each `i` in `start .. end`:
   `slot = atom_angle_indices[i];
    sum_x += angle_triple_x[slot]; (similarly y, z)
    sum_e += angle_triple_energy[slot];
    sum_w += angle_triple_virial[slot];`.
4. Writes the five output slices at index `a`:
   `slot_force_x[a] = sum_x; slot_force_y[a] = sum_y;
    slot_force_z[a] = sum_z; slot_energy[a] = sum_e;
    slot_virial[a] = sum_w`.

The summation is left-to-right in `atom_angle_indices` order.
Since the indices are sorted at load time, the order is
deterministic.

The reduction kernel is universal across angle slots: any angle
potential's angle-triple scratch buffer sums into per-atom forces
the same way. It is compiled at build time via `nvcc` (not via
nvrtc) and loaded as PTX module `"angle"`.

### PTX Module Loading <!-- rq-c07d7c28 -->

`init_device()` loads the compiled `kernels/angle.cu` PTX as
module `"angle"` and captures its `reduce_angle_forces` function
into the `Kernels` handle. The angle JIT module
(`"heddle_jit_composed_angle"`) is loaded separately by
`ForceField::new` from the JIT-composed PTX; it is owned by the
`ForceField` instance, not the global `Kernels` handle. See
`build-pipeline.md` and `jit-composed-intramolecular.md`.

### Rust Launch Helpers <!-- rq-78f49cef -->

The framework's per-step dispatch (see
`jit-composed-intramolecular.md`'s *Parameter Binding and Launch*)
launches the slot's composed angle entry point and then the
universal reduction kernel. Slots do not expose standalone
launchers for the contribution kernel; participation in the
JIT-composed module is the only path to dispatch the per-angle
contribution.

The reduction is launched through the framework's
`reduce_angle_forces` helper:

- `reduce_angle_forces(state: &mut HarmonicAngleState, output_force_x: &mut CudaViewMut<'_, Real>, output_force_y: &mut CudaViewMut<'_, Real>, output_force_z: &mut CudaViewMut<'_, Real>, output_energy: &mut CudaViewMut<'_, Real>, output_virial: &mut CudaViewMut<'_, Real>) -> Result<(), GpuError>` <!-- rq-34bfe79a -->
  - Launches the `reduce_angle_forces` kernel, summing each atom's
    angle contributions into the five caller-supplied output views.
    Output views have length `state.particle_count`.
  - Block size 256; grid size `ceil(state.particle_count / 256)`.
  - Returns `Ok(())` without launching when
    `state.particle_count == 0`.

## Launch Configuration <!-- rq-e9b9f528 -->

- Composed angle contribution kernel: block size 256, grid
  `ceil(angle_count / 256)`, no shared memory. Dispatched by the
  framework from the JIT-composed angle module.
- Reduction kernel: block size 256, grid
  `ceil(particle_count / 256)`, no shared memory.
- Both run on the default stream carried by
  `particle_buffers.device`.

## Determinism <!-- rq-69de20bd -->

- Each angle's force is computed by exactly one thread; no atomics.
- Each atom's reduction is computed by exactly one thread; sums
  proceed in sorted `atom_angle_indices` order.
- Two runs with identical angles, parameters, and positions on the
  same GPU produce byte-identical `angle_triple_*` and `slot_*`
  contents.

## Out of Scope <!-- rq-38b80e11 -->

- Other angle potentials (cosine-harmonic, Urey-Bradley with an
  embedded 1-3 bond term, restricted bending potentials, etc.). Each
  lands as a new `potential` value in `[[angle_types]]` with its own
  kernel.
- Dihedral, improper, and CMAP potentials.
- Per-angle parameter overrides (every angle gets its parameters via
  its angle type).
- Constraint algorithms (rigid angles via SHAKE/RATTLE).
- Angle breaking, forming, or reordering during a simulation.

---

## Gherkin Scenarios <!-- rq-284584c3 -->

```gherkin
Feature: Harmonic angle bonded potential

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called
    And a SimulationBox with lx=ly=lz=1.0e-9 (1 nm)

  # --- Module loading ---

  @rq-f7a71238
  Scenario: init_device exposes the harmonic-angle kernels on the Kernels handle
    When init_device() is called
    Then the returned GpuContext's kernels handle exposes the harmonic_angle_force function
    And the kernels handle exposes the reduce_angle_forces function

  # --- Construction ---

  @rq-dbee9f45
  Scenario: Construct HarmonicAngleState
    Given an AngleList with 2 angles among 5 atoms and one angle type
    And [[angle_types]] with one entry "HOH" potential="harmonic"
      k_theta=5.27e-19 theta_0=1.911 (E_h/rad², radians; flexible-SPC
      Toukan-Rahman bend stiffness)
    When HarmonicAngleState::new(device, &angle_list, &angle_types) is called
    Then it returns Ok(state)
    And state.angle_count equals 2
    And state.particle_count equals 5
    And angle_k_theta and angle_theta_0 on the device equal [5.27e-19] and [1.911]

  # --- Force kernel correctness ---

  @rq-a57bcebe
  Scenario: Three atoms at equilibrium angle produce zero force on each
    Given a ParticleBuffers with positions placed so that the angle
      between (r_i − r_j) and (r_k − r_j) equals theta_0
    And an AngleList with one angle (i=0, j=1, k=2) of type "HOH"
    When harmonic_angle_force is launched
    And the angle_triple buffer is downloaded
    Then |angle_triple_x[m]|, |angle_triple_y[m]|, |angle_triple_z[m]|
      are all zero within f32 round-off for m ∈ {0, 1, 2}

  @rq-e60a2781
  Scenario: Compressed angle produces a torque that opens the angle
    Given positions placing θ < theta_0 with the bisector along +y
    And an angle (i, j, k) of type "HOH"
    When harmonic_angle_force is launched
    Then the y-component of the force on atom_i is negative (pushed away
      from the bisector)
    And the y-component of the force on atom_k is negative
    And F_i + F_j + F_k equals 0 within f32 round-off (Newton's third law)

  @rq-98fd2e40
  Scenario: Stretched angle produces a torque that closes the angle
    Given positions placing θ > theta_0 with the bisector along +y
    And an angle (i, j, k) of type "HOH"
    When harmonic_angle_force is launched
    Then the y-component of the force on atom_i is positive (pulled
      toward the bisector)
    And the y-component of the force on atom_k is positive
    And F_i + F_j + F_k equals 0 within f32 round-off

  @rq-922e1683
  Scenario: Force magnitude matches closed-form expression for an
    isolated angle in vacuum
    Given positions placed at d_ij = d_kj = 1.0e-10 with θ = 1.911 + 0.1
      and theta_0 = 1.911
    And an angle (i, j, k) with k_theta = 5.27e-19 E_h/rad²
    When harmonic_angle_force is launched
    Then sum of per-atom force magnitudes matches the analytical
      |F_i| + |F_j| + |F_k| within 5 × 10⁻³ relative error

  @rq-fbdf08ff
  Scenario: Minimum image is applied
    Given lx=1.0e-9 and positions p_i=(-0.45e-9, 0, 0), p_j=(0, 0, 0),
      p_k=(0.45e-9, 0, 0) so the wrapped i-j and j-k displacements
      use the periodic image
    And an angle (i, j, k) of type "HOH"
    When harmonic_angle_force is launched
    Then the displacements used by the kernel are
      r_ij = (0.55e-9, 0, 0) (the periodic image, not -0.45e-9)
    And the resulting force matches the unwrapped equivalent geometry

  @rq-4ffdad62
  Scenario: Degenerate geometry (atom_i overlaps atom_j) produces zero
    force, not NaN
    Given two atoms at the same position with an angle between them
    When harmonic_angle_force is launched
    Then every angle_triple_* slot for that angle is 0.0_f32

  @rq-bd367201
  Scenario: Near-collinear geometry (sin θ ≈ 0) produces zero force
    via the safety guard
    Given positions placing θ within 1e-8 radians of π
    When harmonic_angle_force is launched
    Then every angle_triple_* slot for that angle is 0.0_f32

  # --- Reduction kernel correctness ---

  @rq-27efd6a0
  Scenario: Atom appearing in one angle receives that angle's force
    directly
    Given a single angle with angle_triple_x[0]=0.5, angle_triple_x[1]=-1.0,
      angle_triple_x[2]=0.5
    And atom_angle_offsets=[0, 1, 2, 3]
    And atom_angle_indices=[0, 1, 2]
    When reduce_angle_forces is launched
    Then slot_force_x[0] equals 0.5
    And slot_force_x[1] equals -1.0
    And slot_force_x[2] equals 0.5

  @rq-ca76fc02
  Scenario: Atom appearing as the centre of one angle and a wing of
    another receives the sum
    Given two angles with atom 0 as centre in angle 0 and as wing in
      angle 1
    When reduce_angle_forces is launched
    Then slot_force_x[0] equals angle_triple_x[1] + angle_triple_x[3]
      (or equivalent index pair, sorted by angle index)

  @rq-699192b2
  Scenario: Reduction summation order is sorted angle index
    Given atom 0 with angle contributions from angles 0 and 1 in slot
      order [0, 3] within atom_angle_indices
    When reduce_angle_forces is launched
    Then slot_force_x[0] equals angle_triple_x[0] + angle_triple_x[3]
      (left-to-right)

  @rq-5fcdc437
  Scenario: Atom with no angles gets zero accumulator
    Given a 5-atom system with angles touching atoms 0..3 (atom 4 has
      no angle)
    When reduce_angle_forces is launched
    Then slot_force_x[4], slot_force_y[4], slot_force_z[4] are all 0.0

  # --- Empty states ---

  @rq-cf50db39
  Scenario: harmonic_angle_force on zero angles is a no-op
    Given a HarmonicAngleState with angle_count == 0
    When harmonic_angle_force is called
    Then it returns Ok(())

  @rq-8d5a8d9c
  Scenario: reduce_angle_forces on zero particles is a no-op
    Given a HarmonicAngleState with particle_count == 0
    When reduce_angle_forces is called
    Then it returns Ok(())

  # --- Reproducibility ---

  @rq-9120ab3c
  Scenario: Two independent calls produce byte-identical accumulators
    Given two independently-constructed HarmonicAngleStates with
      identical angle list and parameters and a ParticleBuffers built
      from identical positions
    When harmonic_angle_force then reduce_angle_forces is launched on
      each
    And both slot_* buffers are downloaded
    Then they agree byte-for-byte

  # --- End-to-end through the framework ---

  @rq-9bb3094c
  Scenario: Triatomic at equilibrium gives zero net force on all atoms
    Given a 3-atom system placed at theta_0 and at d_ij = d_kj = the
      bond's r_e (so bonds are also at equilibrium), with full LJ /
      Coulomb exclusions for the 1-2 and 1-3 pairs
    When force_field.step(...) is called
    And the buffers are downloaded
    Then forces_* on all three atoms are zero within f32 round-off

  @rq-b19189c2
  Scenario: Newton's third law holds for the framework's combined force
    Given a 3-atom angle-only system in vacuum
    When force_field.step(...) is called
    And the buffers are downloaded
    Then forces_x[0] + forces_x[1] + forces_x[2] equals 0 within f32
      round-off
    And similarly for y and z

  # --- Energy and virial outputs ---

  @rq-ee7566b4
  Scenario: A bent angle's energy matches the closed-form expression
    Given an AngleList with one angle (0, 1, 2)
    And angle type "HOH" with k_theta = 5.27e-19 E_h/rad², theta_0 = 1.911 rad
    And atoms placed at θ = 1.911 + 0.2
    When harmonic_angle_force is called
    Then angle_triple_energy[0] + angle_triple_energy[1] + angle_triple_energy[2]
      equals 0.5 * k_theta * (θ - theta_0)² within f32 round-off

  @rq-fe95ff5f
  Scenario: Angle virial equals r_ij · F_i + r_kj · F_k
    Given an AngleList with one angle (0, 1, 2)
    And atoms placed off-equilibrium
    When harmonic_angle_force is called
    Then angle_triple_virial[0] + angle_triple_virial[1] + angle_triple_virial[2]
      equals r_ij · F_i + r_kj · F_k within f32 round-off

  @rq-a587753e
  Scenario: Degenerate angle produces zero energy and virial in
    addition to zero force
    Given atoms placed so that d_ij = 0
    When harmonic_angle_force is called
    Then angle_triple_energy[0..3] and angle_triple_virial[0..3] are
      all 0.0_f32

  # --- Rejection of non-harmonic angle types ---

  @rq-225633c4
  Scenario: Config angle_type with potential != "harmonic" is rejected
    Given an [[angle_types]] entry with potential="cosine-harmonic"
    When the config is loaded
    Then it returns Err(ConfigError::InvalidValue { field:
      "angle_types[0].potential", reason: _ })

  # --- Flexible SPC water smoke test ---

  @rq-501bce66
  Scenario: Single-step force evaluation on one SPC water molecule
    matches a host-side analytical reference
    Given a 3-atom system with one O at the origin and two H placed
      at d_OH = 1.0e-10 m and an opening angle θ_HOH = 1.911 rad +
      0.05 rad (slightly off-equilibrium so the angle force is
      non-zero)
    And [[particle_types]] with O (mass 2.6566e-26 kg, charge -0.82e)
      and H (mass 1.6735e-27 kg, charge +0.41e)
    And [[pair_interactions]] for ("O","O") (σ = 3.166e-10 m,
      ε = 6.502e-22 J), ("O","H") (ε = 0), ("H","H") (ε = 0)
    And [[bond_types]] with one entry "OH" potential="morse" tuned so
      that 2·D_e·a² equals the SPC harmonic stiffness 4.515e5 J/m²
      at r_e = 1.0e-10 m
    And [[angle_types]] with one entry "HOH" potential="harmonic"
      k_theta = 5.27e-19 E_h/rad² theta_0 = 1.911 rad
    And a .topology file declaring two OH bonds and one HOH angle (so
      1-2 and 1-3 exclusions auto-derive to (0,0))
    When force_field.step(...) is called
    And the buffers are downloaded
    Then forces_x[O], forces_y[O], forces_z[O] match the host-side
      analytical sum (bond contributions from two OH bonds plus the
      angle contribution) within 5 × 10⁻³ relative error
    And forces_x[H₁] + forces_x[H₂] + forces_x[O] equals 0 within f32
      round-off, and similarly for y and z
```
