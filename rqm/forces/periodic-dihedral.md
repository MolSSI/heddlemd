# Feature: Periodic Dihedral Bonded Potential <!-- rq-4b84f452 -->

The `PeriodicDihedral` potential slot evaluates a periodic torsion force
for each dihedral in the system's dihedral list (see `topology.md`).
Dihedrals are quadruples of atoms `(i, j, k, l)` whose torsion geometry
about the central `j–k` axis is described by the periodic functional
form `U(φ) = k_phi · (1 + cos(n · φ − phi_0))` with per-dihedral-type
parameters. The slot plugs into the pluggable potential framework
(`framework.md`); selection is implicit — the slot is present whenever
the config's `topology` field references a non-empty `.topology` file
and at least one `[[dihedral_types]]` entry has `potential = "periodic"`.

The system's full set of `[[dihedral_types]]` entries may freely mix
the periodic form with future dihedral potentials (Ryckaert-Bellemans,
…). Each functional form's entries land in their own slot driven by
the same `DihedralList`; the topology file's `[dihedrals]` syntax is
shared across functional forms.

## Algorithm <!-- rq-9fd47b06 -->

The periodic potential at dihedral angle `φ` formed by atoms
`(i, j, k, l)` about the bond `j–k` is

```text
U(φ) = k_phi · (1 + cos(n · φ − phi_0))
```

with per-dihedral-type parameters `k_phi` (force constant, E_h),
`n` (integer multiplicity, in `[1, 6]`), and `phi_0` (phase offset,
radians). The same `(i, j, k, l)` quadruple may carry several Fourier
terms by appearing once in the topology's `[dihedrals]` section for
each term, with each row naming a different dihedral type that
supplies one `(k_phi, n, phi_0)` triple.

The dihedral angle `φ` is computed from the three minimum-image
displacements

```text
b1 = r_i − r_j
b2 = r_k − r_j
b3 = r_k − r_l
```

The `b3` direction (from `l` toward `k`, not from `k` toward `l`) is
the GROMACS / IUPAC convention; the cross-product formula below
yields the canonical torsion only with this sign. Using the standard
cross-product construction:

```text
m       = b1 × b2                       (normal to the i-j-k plane)
n_vec   = b2 × b3                       (normal to the j-k-l plane)
|b2|    = sqrt(b2 · b2)
|m|²    = m · m
|n_vec|²= n_vec · n_vec
cos φ   = (m · n_vec) / (|m| · |n_vec|)
sin φ   has the sign of (b1 · n_vec)
φ       = atan2(|b2| · (b1 · n_vec), m · n_vec)
```

`φ` lies in `(−π, π]`. The convention matches IUPAC and the AMBER,
CHARMM, GROMACS, and OpenMM implementations: `φ = 0` when the four
atoms are *cis*-planar (atoms `i` and `l` on the same side of the
`j–k` axis) and `φ = ±π` when *trans*-planar. The displacements
honour periodic boundary conditions through the minimum-image
convention; the dihedral itself is *not* truncated by any cutoff
distance — periodic dihedrals are intended for bonded use where the
four atoms remain close to each other.

With `f_φ = −dU/dφ = k_phi · n · sin(n · φ − phi_0)`, the per-atom
forces follow from the chain rule `F_a = f_φ · (∂φ/∂r_a)`:

```text
F_i = f_φ · ( |b2| / |m|²)        · m
F_l = f_φ · (−|b2| / |n_vec|²)    · n_vec
s   = (b1 · b2) / |b2|²
t   = (b3 · b2) / |b2|²
F_j = (s − 1) · F_i  −  t       · F_l
F_k = (−s)    · F_i  +  (t − 1) · F_l
```

By construction `F_i + F_j + F_k + F_l = 0` (Newton's third law) to
within `f32` round-off.

The dihedral's full potential energy is

```text
U_m = k_phi · (1 + cos(n · φ − phi_0))
```

and its scalar virial, derived from the per-atom forces with `j` as
the reference, is

```text
W_m = b1 · F_i  +  b2 · F_k  +  (b2 − b3) · F_l
```

equivalent to `Σ_a (r_a − r_j) · F_a` over the four atoms, with
`r_l − r_j = (r_l − r_k) + (r_k − r_j) = −b3 + b2`.

The kernel applies the following defensive guards in `f32`
arithmetic:

- When `|m|² < 1.0e−14_f32`, `|n_vec|² < 1.0e−14_f32`, or
  `|b2|² < 1.0e−14_f32`: all four force vectors and the per-atom
  energy and virial slots are written as zero.
- The functor never divides by `sin(...)`; the AMBER-style periodic
  form is regular at every `φ`, so the only singular configurations
  are the degenerate geometries trapped by the guard above.
- `n = 0` is rejected at config-load time (see *Parameters*), so the
  kernel never evaluates a constant-energy term.

For each dihedral `m` the kernel writes its four per-atom force
triples, its per-atom energy share `U_m / 4`, and its per-atom virial
share `W_m / 4` into consecutive slots `4·m`, `4·m + 1`, `4·m + 2`,
`4·m + 3` of the per-dihedral scratch buffer:

| slot      | atom    |
| --------- | ------- |
| `4·m`     | `atom_i` |
| `4·m + 1` | `atom_j` |
| `4·m + 2` | `atom_k` |
| `4·m + 3` | `atom_l` |

The energy and virial are distributed in quarters (rather than the
half-and-half convention used by bond pairs or the thirds used by
angles) so that summing all per-atom shares for one dihedral
reproduces the dihedral's full `U_m` and `W_m`.

## Per-Step Kernel Sequence <!-- rq-fb1676f8 -->

The slot's contribution and reduction run once each per step:

| Step | Kernel | Operation | Stage label |
| --- | --- | --- | --- |
| 1 | `heddle_jit_composed_dihedral_<i>_{f,fev}` | compute forces per dihedral, write to dihedral-quadruple buffer | `JitComposedDihedralForce` |
| 2 | `reduce_dihedral_forces` | per-atom sum of dihedral contributions, write to slot accumulator | `ReduceDihedralForces` |

Step 1 is the JIT-composed dihedral module's entry point for this
slot (slot index `<i>` is the slot's zero-based position among
active dihedral slots in canonical slot order; the `_f` vs `_fev`
suffix is selected by the per-step `AggregateLevel`). The
JIT-composed module includes the slot's per-dihedral periodic
functor source described in *Source Fragment* below. See
`jit-composed-intramolecular.md` for the composer's contract.

Step 2 runs the standalone `reduce_dihedral_forces` kernel compiled
at build time. The reduction is shape-universal across dihedral
slots (any dihedral potential's per-dihedral contributions sum into
per-atom forces the same way); it is not part of the JIT module.

The class-combine kernel runs after every slot's reduction. See
`framework.md` for the slot order.

## Force Accumulation <!-- rq-2ec7b7fc -->

The slot owns a `DihedralQuadrupleBuffer` of length `4 · D` where `D`
is the number of dihedrals. Each slot entry carries five `f32`
quantities: three force components, quarter-energy, and
quarter-virial. Slot `4·m + p` (where `p ∈ {0, 1, 2, 3}`) holds the
contribution to the `p`-th atom of dihedral `m`, with the atom
ordering documented above.

The reduction kernel reads the precomputed `atom_dihedral_offsets` /
`atom_dihedral_indices` tables (see `topology.md`) and sums each
atom's contributions in fixed order. For atom `a`, the kernel
computes five sequential left-to-right sums:

```text
slot_force_x[a]  = sum over m in atom_dihedral_indices[a] of dihedral_quadruple_x[m]
slot_force_y[a]  = same with y
slot_force_z[a]  = same with z
slot_energy[a]   = sum over m in atom_dihedral_indices[a] of dihedral_quadruple_energy[m]
slot_virial[a]   = sum over m in atom_dihedral_indices[a] of dihedral_quadruple_virial[m]
```

The `atom_dihedral_indices` slice for each atom is sorted by
underlying dihedral index at file-load time, so the summation order
is identical across runs. Each thread maps to one atom; there are no
atomics and no race conditions.

## Parameters <!-- rq-4eec8cf7 -->

Each `[[dihedral_types]]` entry in the config that uses
`potential = "periodic"` contributes one row to a per-dihedral-type
parameter table uploaded to the device:

- `k_phi: f64` — force constant in E_h. Required. Finite. May be
  zero or negative (negative `k_phi` is equivalent to a `phi_0`
  shift by `π` and is accepted as-is).
- `n: u32` — multiplicity. Required. Integer in `[1, 6]`. Higher
  multiplicities are rejected.
- `phi_0: f64` — phase offset in radians. Required. Finite, in
  `[−2π, 2π]`. The cosine is periodic, so any value of `phi_0`
  outside this range can be wrapped without loss; the range bound is
  defensive against typos.
- `scale_lj_14: f64` — optional. Default `0.5`. Finite, in `[0.0,
  1.0]`. The Lennard-Jones scale factor applied to the implicit 1-4
  exclusion derived from any `[dihedrals]` row that names this type
  and is the first to introduce its `(atom_i, atom_l)` pair (see
  `topology.md`'s *Effective exclusions*).
- `scale_coul_14: f64` — optional. Default `1.0 / 1.2 ≈ 0.83333`.
  Finite, in `[0.0, 1.0]`. The Coulomb scale factor applied to the
  same implicit 1-4 exclusion.

The on-device parameter tables are four `CudaSlice<f32>` /
`CudaSlice<u32>` arrays (`dihedral_k_phi`, `dihedral_phi_0`,
`dihedral_n`, all length `n_dihedral_types`), cast from `f64` or
`u32` at upload time. The 1-4 scales are not consulted by the
dihedral kernel; they are read by the topology loader and folded
into the `ExclusionList`. Each dihedral carries a `dihedral_type_index`
(see `topology.md`) into this table.

The only `potential` value handled by the periodic slot is
`"periodic"`. Other values land in different slots (e.g. a future
Ryckaert-Bellemans slot would consume `potential = "ryckaert-bellemans"`).

## Empty State <!-- rq-66bad604 -->

When the dihedral list contains no entries that resolve to a
periodic dihedral type (`dihedral_list.is_empty()` or every entry
references a non-periodic type), the `PeriodicDihedralState` is not
constructed by the `ForceField` and the slot is absent from the slot
list. The framework's combiner handles slot-presence correctly (see
`framework.md`).

When `particle_count == 0`, the dihedral list must also be empty
(the file parser rejects any dihedral entry with an out-of-range atom
index, and every index is out of range when `N == 0`). The slot is
therefore not constructed.

## Feature API <!-- rq-ccea967a -->

### Types <!-- rq-b97c4c03 -->

- `PeriodicDihedralState` — implements the `Potential` trait with <!-- rq-b7fe6425 -->
  `label() == "periodic_dihedral"` (see `framework.md`). Fields:
  - `device: Arc<CudaDevice>`
  - `dihedrals: CudaSlice<u32>` — flat array of `[atom_i, atom_j,
    atom_k, atom_l, dihedral_type_index]` quintuples, length `5 · D`,
    sorted to match the periodic-typed subset of
    `DihedralList::dihedrals`.
  - `atom_dihedral_offsets: CudaSlice<u32>` — length `N + 1`.
  - `atom_dihedral_indices: CudaSlice<u32>` — length `4 · D`.
  - `dihedral_k_phi: CudaSlice<f32>` — length `n_periodic_types`.
  - `dihedral_phi_0: CudaSlice<f32>` — length `n_periodic_types`.
  - `dihedral_n: CudaSlice<u32>` — length `n_periodic_types`.
  - `dihedral_quadruple_x: CudaSlice<f32>` — length `4 · D`, per-slot
    force x contribution.
  - `dihedral_quadruple_y: CudaSlice<f32>` — length `4 · D`.
  - `dihedral_quadruple_z: CudaSlice<f32>` — length `4 · D`.
  - `dihedral_quadruple_energy: CudaSlice<f32>` — length `4 · D`,
    per-slot quarter-energy contribution (`U_m / 4`).
  - `dihedral_quadruple_virial: CudaSlice<f32>` — length `4 · D`,
    per-slot quarter-virial contribution (`W_m / 4`).
  - `dihedral_count: usize` — `D`, the number of dihedrals that
    resolve to periodic types (which may be smaller than the
    `DihedralList`'s total length when the system mixes functional
    forms).
  - `particle_count: usize`

  All fields private; the slot's public surface is the per-step
  methods invoked by `ForceField::step` (see `framework.md`).

  Constructor:

  - `PeriodicDihedralState::new(device: Arc<CudaDevice>, dihedral_list: &DihedralList, dihedral_types: &[DihedralTypeConfig]) -> Result<PeriodicDihedralState, GpuError>`
    - Filters `dihedral_types` to entries with `potential ==
      "periodic"` and uploads their `(k_phi, n, phi_0)` parameters.
    - Filters `dihedral_list.dihedrals` to entries whose
      `dihedral_type_index` resolves to a periodic type and uploads
      the filtered list, rebuilding `atom_dihedral_offsets` /
      `atom_dihedral_indices` from the filtered set so per-atom
      indexing is internally consistent with the device-side
      `dihedrals` array.
    - Allocates the five per-dihedral `dihedral_quadruple_*` buffers
      (force x/y/z, quarter-energy, quarter-virial), each of length
      `4 · D`. Per-atom output is added into the framework-supplied
      `SlotOutputView` (a view onto the slot's class accumulator;
      see `framework.md`'s *Class Output Accumulators*) during
      `reduce()`; the slot owns no per-atom accumulator buffers of
      its own.
    - When the filtered dihedral list is empty, this method is not
      called by the `ForceField` — see *Empty State*.

### Source Fragment <!-- rq-22a1660e -->

`PeriodicDihedralBuilder::dihedral_force_fragment(cx)` returns a
`DihedralForceFragment` whose functor implements the per-dihedral
periodic contribution. The fragment defines a `__device__` functor
`PeriodicDihedralFunctor` whose member function `evaluate(dx_ij,
dy_ij, dz_ij, dx_kj, dy_kj, dz_kj, dx_lk, dy_lk, dz_lk,
dihedral_type_index, fix, fiy, fiz, fjx, fjy, fjz, fkx, fky, fkz,
flx, fly, flz, u_m, w_m)` computes:

1. Computes the geometric scalars `|b2|`, `|m|²`, `|n_vec|²`, the
   cross products `m = b1 × b2` and `n_vec = b2 × b3` (with `b1 =
   r_i − r_j`, `b2 = r_k − r_j`, `b3 = r_l − r_k` passed in via the
   `dx_*` displacements), and the dihedral angle
   `φ = atan2(|b2| · (b1 · n_vec), m · n_vec)`.
2. Reads `k = dihedral_k_phi[dihedral_type_index]`, `n =
   dihedral_n[dihedral_type_index]`, and `phi_0 =
   dihedral_phi_0[dihedral_type_index]` from device-buffer pointers
   held as members of the functor.
3. Computes `delta = n · φ − phi_0`, `u_m = k · (1 + cos(delta))`,
   and `f_phi = k · n · sin(delta)`.
4. Computes the per-atom forces per the formulas in *Algorithm* and
   writes them to `(fix, fiy, fiz)`, `(fjx, fjy, fjz)`,
   `(fkx, fky, fkz)`, and `(flx, fly, flz)`.
5. Writes the dihedral's full potential energy `u_m` and scalar
   virial `w_m = b1 · F_i + b2 · F_k + (b2 + b3) · F_l` (the
   outer-loop body distributes the `1/4` symmetry factor when
   writing to the scratch buffer).

When the functor's defensive guards trigger (`|m|²`, `|n_vec|²`, or
`|b2|²` below the threshold given in *Algorithm*), it writes zero to
every output field. The outer-loop body then writes zeros to the
corresponding twenty scratch-buffer entries (five quantities ×
four slots).

The composed kernel's outer-loop body (in the JIT-composed dihedral
module — see `jit-composed-intramolecular.md`) handles the
common-args reading: reads the dihedral list `(atom_i, atom_j,
atom_k, atom_l, dihedral_type_index)`, computes the minimum-image
displacements `b1 = r_i − r_j`, `b2 = r_k − r_j`, `b3 = r_l − r_k`,
calls the functor's `evaluate`, then writes the four per-atom force
triples along with `u_m / 4` and `w_m / 4` into the slot's
dihedral-quadruple scratch buffer at indices `4·m`, `4·m + 1`,
`4·m + 2`, and `4·m + 3`. See
`jit-composed-intramolecular.md`'s *Composed-Module Structure* for
the full outer-loop body specification.

The fragment's `entry_point_args` declares the per-dihedral-type
parameter table pointers (`dihedral_k_phi`, `dihedral_n`,
`dihedral_phi_0`); the `functor_init_source` assigns them to the
functor's members at the start of the entry-point body.

### Reduction Kernel <!-- rq-898a6225 -->

`kernels/dihedral.cu` declares the shape-universal reduction kernel:

```c
extern "C" __global__ void reduce_dihedral_forces(
    const Real *dihedral_quadruple_x,
    const Real *dihedral_quadruple_y,
    const Real *dihedral_quadruple_z,
    const Real *dihedral_quadruple_energy,
    const Real *dihedral_quadruple_virial,
    const unsigned int *atom_dihedral_offsets,
    const unsigned int *atom_dihedral_indices,
    Real *slot_force_x, Real *slot_force_y, Real *slot_force_z,
    Real *slot_energy, Real *slot_virial,
    unsigned int n);
```

One thread per atom `a = blockIdx.x · blockDim.x + threadIdx.x`
(block size 256, grid `ceil(n / 256)`). Thread `a`:

1. Reads `start = atom_dihedral_offsets[a]` and `end =
   atom_dihedral_offsets[a + 1]`.
2. Initialises five running sums to zero: `sum_x`, `sum_y`,
   `sum_z`, `sum_e`, `sum_w`.
3. For each `i` in `start .. end`:
   `slot = atom_dihedral_indices[i];
    sum_x += dihedral_quadruple_x[slot]; (similarly y, z)
    sum_e += dihedral_quadruple_energy[slot];
    sum_w += dihedral_quadruple_virial[slot];`.
4. Writes the five output slices at index `a`:
   `slot_force_x[a] = sum_x; slot_force_y[a] = sum_y;
    slot_force_z[a] = sum_z; slot_energy[a] = sum_e;
    slot_virial[a] = sum_w`.

The summation is left-to-right in `atom_dihedral_indices` order.
Since the indices are sorted at load time, the order is
deterministic.

The reduction kernel is universal across dihedral slots: any
dihedral potential's quadruple scratch buffer sums into per-atom
forces the same way. It is compiled at build time via `nvcc` (not
via nvrtc) and loaded as PTX module `"dihedral"`.

### PTX Module Loading <!-- rq-b3cbcae1 -->

`init_device()` loads the compiled `kernels/dihedral.cu` PTX as
module `"dihedral"` and captures its `reduce_dihedral_forces`
function into the `Kernels` handle. The dihedral JIT module
(`"heddle_jit_composed_dihedral"`) is loaded separately by
`ForceField::new` from the JIT-composed PTX; it is owned by the
`ForceField` instance, not the global `Kernels` handle. See
`build-pipeline.md` and `jit-composed-intramolecular.md`.

### Rust Launch Helpers <!-- rq-4dcd812b -->

The framework's per-step dispatch (see
`jit-composed-intramolecular.md`'s *Parameter Binding and Launch*)
launches the slot's composed dihedral entry point and then the
universal reduction kernel. Slots do not expose standalone
launchers for the contribution kernel; participation in the
JIT-composed module is the only path to dispatch the per-dihedral
contribution.

The reduction is launched through the framework's
`reduce_dihedral_forces` helper:

- `reduce_dihedral_forces(state: &mut PeriodicDihedralState, output_force_x: &mut CudaViewMut<'_, Real>, output_force_y: &mut CudaViewMut<'_, Real>, output_force_z: &mut CudaViewMut<'_, Real>, output_energy: &mut CudaViewMut<'_, Real>, output_virial: &mut CudaViewMut<'_, Real>) -> Result<(), GpuError>` <!-- rq-1d8fd9cf -->
  - Launches the `reduce_dihedral_forces` kernel, summing each
    atom's dihedral contributions into the five caller-supplied
    output views. Output views have length `state.particle_count`.
  - Block size 256; grid size `ceil(state.particle_count / 256)`.
  - Returns `Ok(())` without launching when
    `state.particle_count == 0`.

## Launch Configuration <!-- rq-2932ea42 -->

- Composed dihedral contribution kernel: block size 256, grid
  `ceil(dihedral_count / 256)`, no shared memory. Dispatched by the
  framework from the JIT-composed dihedral module.
- Reduction kernel: block size 256, grid
  `ceil(particle_count / 256)`, no shared memory.
- Both run on the default stream carried by
  `particle_buffers.device`.

## Determinism <!-- rq-d447cc3e -->

- Each dihedral's force is computed by exactly one thread; no atomics.
- Each atom's reduction is computed by exactly one thread; sums
  proceed in sorted `atom_dihedral_indices` order.
- Two runs with identical dihedrals, parameters, and positions on
  the same GPU produce byte-identical `dihedral_quadruple_*` and
  `slot_*` contents.

## Out of Scope <!-- rq-bb0ff6de -->

- Other dihedral potentials (Ryckaert-Bellemans polynomial,
  restricted-cosine, harmonic-in-φ, OPLS-style multi-term fused
  forms). Each lands as a new `potential` value in
  `[[dihedral_types]]` with its own functor and its own slot, both
  feeding the same shared `DihedralList` and the same shared
  `reduce_dihedral_forces` reduction kernel. The Ryckaert-Bellemans
  form is the next planned addition and the type-system shape
  (tagged `DihedralTypeConfig` enum, per-functional-form slot) is
  designed to accommodate it without disturbing the periodic slot.
- Improper-dihedral potentials (out-of-plane bending centred on one
  atom). Impropers use a distinct topology section and a distinct
  geometric convention (the angle between a bond and a plane, not
  the angle between two planes about an axis); a future
  `[impropers]` section and its own slot lands separately.
- CMAP correction terms (CHARMM's 2-D φ/ψ grid potential).
- Per-dihedral parameter overrides (every dihedral gets its
  parameters via its dihedral type).
- A device-side multi-term fused evaluator that collapses several
  Fourier terms on the same `(i, j, k, l)` quadruple into one
  thread's work. The current design assigns each Fourier term its
  own row in `[dihedrals]` and processes each row by its own thread;
  the dihedral count and the launch grid scale with the term count.
- A refactor of the per-shape JIT composer (bonded, angle, dihedral)
  into a generic n-tuple-shape composer parameterised by atoms-per-
  tuple, displacement count, and reduction stride. Each shape is
  presently a separate composer in `jit-composed-intramolecular.md`;
  the dihedral shape is added by duplicating the angle composer
  pattern and adjusting it for four atoms. Folding the three
  composers into one is a follow-up if and when the duplication
  becomes a maintenance burden.
- An explicit `[scaled_exclusions]` section in the `.topology` file
  for declaring 1-4 (or other) scaled exclusions independent of the
  dihedral list. The current effective-exclusion plumbing in
  `topology.md` handles per-pair `(scale_lj, scale_coul)` tuples
  through the existing `[exclusions]` section's four-column form;
  the implicit derivation in this feature uses the same plumbing,
  so adding a future `[scaled_exclusions]` section requires no new
  data model — only a new parser entry point.
- Dihedral breaking, forming, or reordering during a simulation.

---

## Gherkin Scenarios <!-- rq-3dc1c2d1 -->

```gherkin
Feature: Periodic dihedral bonded potential

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called
    And a SimulationBox with lx=ly=lz=1.0e-9 (1 nm)

  # --- Module loading ---

  @rq-ae1c6a11
  Scenario: init_device exposes the dihedral reduction kernel on the Kernels handle
    When init_device() is called
    Then the returned GpuContext's kernels handle exposes the reduce_dihedral_forces function

  # --- Construction ---

  @rq-d2f43e61
  Scenario: Construct PeriodicDihedralState
    Given a DihedralList with 1 dihedral among 4 atoms and one periodic dihedral type
    And [[dihedral_types]] with one entry "CT-CT-CT-CT_n3" potential="periodic"
      k_phi=2.18e-4 n=3 phi_0=0.0
    When PeriodicDihedralState::new(device, &dihedral_list, &dihedral_types) is called
    Then it returns Ok(state)
    And state.dihedral_count equals 1
    And state.particle_count equals 4
    And dihedral_k_phi on the device equals [2.18e-4]
    And dihedral_n on the device equals [3]
    And dihedral_phi_0 on the device equals [0.0]

  @rq-ae19b22f
  Scenario: Construct PeriodicDihedralState filters out non-periodic dihedral entries
    Given a DihedralList with 2 dihedrals
    And [[dihedral_types]] mixing one periodic entry (used by dihedral 0)
      and one ryckaert-bellemans entry (used by dihedral 1)
    When PeriodicDihedralState::new is called
    Then state.dihedral_count equals 1
    And state.dihedrals on the device contains only the periodic-typed dihedral

  # --- Force kernel correctness ---

  @rq-603b4597
  Scenario: Equilibrium dihedral (n*phi == phi_0) produces zero force on each atom
    Given four atoms placed such that n · φ equals phi_0
    And a dihedral (0, 1, 2, 3) of type "CT-CT-CT-CT_n3"
    When the periodic dihedral force is launched
    And the dihedral_quadruple buffer is downloaded
    Then |dihedral_quadruple_x[m]|, _y[m], _z[m] are all zero within f32 round-off
      for m in {0, 1, 2, 3}

  @rq-e0933b95
  Scenario: Newton's third law holds for the four per-atom forces
    Given four atoms placed off-equilibrium in a non-degenerate dihedral geometry
    And a dihedral (0, 1, 2, 3) of any periodic dihedral type
    When the periodic dihedral force is launched
    Then F_0 + F_1 + F_2 + F_3 equals 0 within f32 round-off (sum of dihedral_quadruple
      contributions for that dihedral)

  @rq-185a2743
  Scenario: Force matches the closed-form periodic expression for a non-degenerate dihedral
    Given a dihedral (0, 1, 2, 3) with type k_phi=2.18e-4 n=3 phi_0=0.0
    And atoms placed such that φ = 0.5 (in radians)
    When the periodic dihedral force is launched
    Then the sum of per-atom force magnitudes matches the analytical force
      derived from f_φ = k · n · sin(n·φ − phi_0) within 5 × 10⁻³ relative error

  @rq-dceedbbb
  Scenario: Force computed from analytical -dU/dφ via central differences
    Given a dihedral (0, 1, 2, 3) placed off-equilibrium
    When the periodic dihedral force is launched
    And U(φ + ε) and U(φ − ε) are computed via independent kernel launches at
      perturbed positions (ε ~ 1e-4 rad)
    Then the magnitude of the per-atom force is consistent with -(U(φ+ε) - U(φ-ε)) / (2ε)
      within 1 × 10⁻² relative error

  @rq-2835ec70
  Scenario: Multi-term dihedral (two rows on the same quadruple) sums by the framework
    Given two rows in [dihedrals] both naming (0, 1, 2, 3), one of type n=1 and
      one of type n=3, both periodic
    When the periodic dihedral force is launched
    Then the per-atom forces on atom 0 equal the sum of the analytical n=1 and n=3
      contributions (after framework reduction) within f32 round-off

  @rq-c432078d
  Scenario: Minimum image is applied to b1, b2, and b3
    Given lx=1.0e-9 and four atoms positioned so that at least one of b1, b2, b3
      wraps through the periodic boundary
    When the periodic dihedral force is launched
    Then the displacements used by the kernel are the wrapped (minimum-image) ones
    And the resulting forces match the equivalent unwrapped-geometry computation

  @rq-b600335f
  Scenario: Degenerate geometry (|m|² ~ 0) produces zero force, not NaN
    Given four atoms placed so that i, j, k are collinear (|b1 × b2|² < 1e-14)
    When the periodic dihedral force is launched
    Then every dihedral_quadruple_* slot for that dihedral is 0.0_f32

  @rq-41636703
  Scenario: Degenerate geometry (|n_vec|² ~ 0) produces zero force, not NaN
    Given four atoms placed so that j, k, l are collinear
    When the periodic dihedral force is launched
    Then every dihedral_quadruple_* slot for that dihedral is 0.0_f32

  @rq-c0951af5
  Scenario: Degenerate geometry (|b2|² ~ 0) produces zero force, not NaN
    Given atom_j and atom_k placed at the same position
    When the periodic dihedral force is launched
    Then every dihedral_quadruple_* slot for that dihedral is 0.0_f32

  @rq-7bf70ee9
  Scenario: n = 0 is rejected at config-load time
    Given a [[dihedral_types]] entry with potential="periodic" n=0
    When the config is loaded
    Then it returns Err(ConfigError::InvalidValue { field: "dihedral_types[0].n", .. })

  @rq-ce48ab4a
  Scenario: n > 6 is rejected at config-load time
    Given a [[dihedral_types]] entry with potential="periodic" n=7
    When the config is loaded
    Then it returns Err(ConfigError::InvalidValue { field: "dihedral_types[0].n", .. })

  # --- Reduction kernel correctness ---

  @rq-a73f7a4b
  Scenario: Atom appearing in one dihedral receives that dihedral's force directly
    Given a single dihedral with dihedral_quadruple_x[0..4] = [0.25, -0.50, 0.50, -0.25]
    And atom_dihedral_offsets = [0, 1, 2, 3, 4]
    And atom_dihedral_indices = [0, 1, 2, 3]
    When reduce_dihedral_forces is launched
    Then slot_force_x equals [0.25, -0.50, 0.50, -0.25]

  @rq-59b48bcd
  Scenario: Atom appearing in two dihedrals receives the sum of its slot values
    Given two dihedrals whose per-atom-0 contributions land in slots 0 and 4
    When reduce_dihedral_forces is launched
    Then slot_force_x[0] equals dihedral_quadruple_x[0] + dihedral_quadruple_x[4]

  @rq-7d412e08
  Scenario: Reduction summation order follows sorted atom_dihedral_indices
    Given an atom with two contributions whose sorted indices are [a, b] with a < b
    When reduce_dihedral_forces is launched
    Then slot_force_x is computed as dihedral_quadruple_x[a] + dihedral_quadruple_x[b]
      (left-to-right)

  @rq-91fbdf55
  Scenario: Atom with no dihedrals gets zero accumulator
    Given a 5-atom system with a dihedral touching atoms 0..3 (atom 4 has no dihedral)
    When reduce_dihedral_forces is launched
    Then slot_force_x[4], slot_force_y[4], slot_force_z[4] are all 0.0

  # --- Empty states ---

  @rq-b0e866c2
  Scenario: periodic dihedral compute on zero dihedrals is a no-op
    Given a PeriodicDihedralState with dihedral_count == 0
    When the composed dihedral force kernel would be dispatched
    Then no kernel is dispatched
    And the slot reports an empty scratch buffer

  @rq-6b38fe0b
  Scenario: reduce_dihedral_forces on zero particles is a no-op
    Given a PeriodicDihedralState with particle_count == 0
    When reduce_dihedral_forces is called
    Then it returns Ok(()) without launching

  # --- Reproducibility ---

  @rq-04a6973e
  Scenario: Two independent calls produce byte-identical accumulators
    Given two independently-constructed PeriodicDihedralStates with identical
      dihedral lists, parameters, and a ParticleBuffers built from identical positions
    When the dihedral force is computed and reduced on each
    And both slot_* buffers are downloaded
    Then they agree byte-for-byte

  # --- Energy and virial outputs ---

  @rq-d225bef2
  Scenario: Dihedral energy matches the closed-form expression
    Given a DihedralList with one dihedral (0, 1, 2, 3)
    And periodic type k_phi=2.18e-4 n=3 phi_0=0.0
    And atoms placed at φ = 0.5 rad
    When the periodic dihedral force is launched
    Then dihedral_quadruple_energy[0..4] sum equals
      k_phi · (1 + cos(n · φ − phi_0)) within f32 round-off

  @rq-3ebdb69b
  Scenario: Dihedral virial equals b1·F_i + b2·F_k + (b2−b3)·F_l
    Given a DihedralList with one dihedral (0, 1, 2, 3) placed off-equilibrium
    When the periodic dihedral force is launched
    Then dihedral_quadruple_virial[0..4] sum equals
      b1 · F_i + b2 · F_k + (b2 − b3) · F_l within f32 round-off

  @rq-f2f1372d
  Scenario: Degenerate dihedral produces zero energy and zero virial
    Given atoms placed such that |m|² < 1e-14
    When the periodic dihedral force is launched
    Then dihedral_quadruple_energy[0..4] and dihedral_quadruple_virial[0..4] are
      all 0.0_f32

  # --- Rejection of non-periodic dihedral types in this slot ---

  @rq-c0e1d856
  Scenario: Builder skips dihedral_types whose potential is not "periodic"
    Given a [[dihedral_types]] array mixing "periodic" and "ryckaert-bellemans"
      entries with at least one entry of each kind
    When PeriodicDihedralBuilder::build(cx) is called
    Then state.dihedral_count counts only dihedrals referencing periodic types
    And the device-side dihedral_k_phi table contains exactly one entry per
      periodic dihedral type

  # --- End-to-end through the framework ---

  @rq-d387c0f5
  Scenario: Two-bond, one-angle, one-dihedral n-butane configuration
    Given a 4-atom n-butane backbone with two CC bonds, one CCC angle, and one
      periodic dihedral declared in the .topology file
    And [[bond_types]], [[angle_types]], and [[dihedral_types]] supplying the
      required parameters
    When force_field.step(...) is called
    And the buffers are downloaded
    Then forces_x[0] + forces_x[1] + forces_x[2] + forces_x[3] equals 0 within
      f32 round-off (and similarly for y, z)
    And the per-atom forces match a host-side analytical sum (bond + angle +
      dihedral contributions) within 5 × 10⁻³ relative error
```
