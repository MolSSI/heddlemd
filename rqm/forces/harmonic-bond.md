# Feature: Harmonic Bonded Potential <!-- rq-c3da9ee1 -->

The `HarmonicBond` potential slot evaluates a harmonic (Hooke's-law) bond
force for each bond in the system whose bond type selects
`potential = "harmonic"` (see `topology.md`). Bonds are pairs of atoms
whose distance interaction is described by the harmonic functional form
with per-bond-type parameters. The slot plugs into the pluggable potential
framework (`framework.md`); selection is implicit — the slot is present
whenever the config's `topology` field references a `.topology` file whose
`[bonds]` section names at least one bond whose `[[bond_types]]` entry has
`potential = "harmonic"`.

The harmonic bond is the stiff-spring form used by the AMBER and CHARMM
protein force fields; it coexists with the Morse bond (`morse-bonded.md`)
in the same system, routed per bond by bond type (see *Per-Potential Bond
Selection*).

## Algorithm <!-- rq-53c2227c -->

The harmonic pair potential between atoms `i` and `j` at separation
`r = |r_i - r_j|` is

```text
U(r) = (1/2) * k * (r - r_0)^2
F(r) = -dU/dr = -k * (r - r_0)
```

where the bond-type parameters are `k` (force constant, energy per
length², in the `U = ½ k (r − r_0)²` convention) and `r_0` (equilibrium
distance). The minimum-image displacement between the two atoms is used so
the kernel honours periodic boundary conditions; the bond is *not*
truncated by any cutoff distance — the harmonic bond is intended for
short-range bonded use where atoms remain near equilibrium.

For bond `k` connecting `atom_i` and `atom_j`, the kernel computes the
minimum-image displacement `(dx, dy, dz) = (r_i - r_j)`, its length `r`,
and the per-component force factor `fmag = -k * (r - r_0) / r`. The force
on `atom_i` is `+fmag * (dx, dy, dz)`; the force on `atom_j` is
`-fmag * (dx, dy, dz)` (Newton's third law applied at the kernel level).
Because `fmag` carries the sign of `-(r - r_0)`, a stretched bond
(`r > r_0`) pulls the two atoms together (attractive) and a compressed
bond (`r < r_0`) pushes them apart (repulsive). The two contributions are
written to consecutive slots in the bond-pair buffer (see *Force
Accumulation* below).

### Prefactor collapse <!-- rq-ca10a975 -->

The config exposes `k` in the `U = ½ k (r − r_0)²` convention. The device
parameter table stores `k` itself (converted to atomic units), **not**
`k / 2`. The analytic derivative absorbs the convention's factor of one
half against the square's factor of two, so the force magnitude
`fmag = -k * (r - r_0) / r` contains no half-factor and no compensating
doubling: the per-step force path pays nothing for the half convention. The
one-half survives only in the potential-energy term
`u_k = 0.5 * k * (r - r_0)^2`, where it is a compile-time `0.5f` literal
evaluated solely on `ForcesAndScalars` (`fev`) steps. Force-only (`f`)
steps never reference it.

## Per-Step Kernel Sequence <!-- rq-d504d070 -->

The slot's contribution and reduction run once each per step:

| Step | Kernel | Operation | Stage label |
| --- | --- | --- | --- |
| 1 | `heddle_jit_composed_bonded_<i>_{f,fev}` | compute force per bond, write to bond-pair buffer | `JitComposedBondedForce` |
| 2 | `reduce_bond_forces` | per-atom sum of bond contributions, write to slot accumulator | `ReduceBondForces` |

Step 1 is the JIT-composed bonded module's entry point for this slot (slot
index `<i>` is the slot's zero-based position among active bonded slots in
canonical slot order; the `_f` vs `_fev` suffix is selected by the
per-step `AggregateLevel`). The JIT-composed module includes the slot's
per-bond harmonic functor source described in *Source Fragment* below. See
`jit-composed-intramolecular.md` for the composer's contract; it already
supports an arbitrary number of bonded slots, so no change to the composer
is required to run a harmonic slot alongside a Morse slot.

Step 2 runs the standalone `reduce_bond_forces` kernel compiled at build
time. The reduction is shape-universal across bonded slots (any bonded
potential's per-bond contributions sum into per-atom forces the same way);
it is shared with `morse-bonded.md` and is not part of the JIT module.

The class-combine kernel runs after every slot's reduction. See
`framework.md` for the slot order.

## Per-Potential Bond Selection <!-- rq-f62d94d2 -->

The parsed `BondList` (see `topology.md`) is potential-agnostic: it holds
every bond in the system, each carrying a `bond_type_index` that is a
global index into the config's `[[bond_types]]` array. A system may mix
Morse and harmonic bonds freely.

`HarmonicBondState::new` selects, from the shared `BondList`, exactly the
bonds whose `bond_type_index` names a `[[bond_types]]` entry with
`potential == "harmonic"`, preserving their `(atom_i, atom_j)` sort order.
From that selected subset it builds:

- its own device bond array (the selected `[atom_i, atom_j,
  bond_type_index]` triples), and
- its own per-atom reduction map (`atom_bond_offsets` / `atom_bond_indices`
  restricted to the selected bonds), constructed exactly as the shared
  `BondList` builds its map (see `topology.md`'s *Bond list*) but over the
  subset, so the universal `reduce_bond_forces` kernel operates unchanged.

The Morse slot performs the mirror-image selection for `potential ==
"morse"`. Every bond is therefore owned by exactly one bonded slot, and no
bond is evaluated twice. When every bond in the system is harmonic, the
selected subset is the whole list and the derived reduction map equals the
shared `BondList`'s map.

The `bond_type_index` stored in each selected triple remains the global
index; the device parameter table is sized to the full `[[bond_types]]`
array length (`n_bond_types`) and is addressed by that global index, so no
index remapping is performed. Entries for non-harmonic bond types are
present in the table but never read by this slot.

## Force Accumulation <!-- rq-1f923a98 -->

The slot owns a `BondPairBuffer` of length `2 * B_h` where `B_h` is the
number of harmonic bonds selected for this slot. Each slot carries five
`f32` quantities: three force components, half-energy, and half-virial.
Slot `2 * k` holds atom `i`'s share of the slot's `k`-th bond; slot
`2 * k + 1` holds atom `j`'s share. The half-energy and half-virial
conventions match the pair-buffer convention (see `pair-force-kernel.md`):
each slot writes `U_k / 2` and `W_k / 2` so the system total sum over slots
equals `Σ_k U_k` and `Σ_k W_k` respectively, counting each bond exactly
once.

The reduction kernel reads the slot's own `atom_bond_offsets` /
`atom_bond_indices` tables (built as described in *Per-Potential Bond
Selection*) and sums each atom's contributions in fixed order. For atom
`a`, the kernel computes five sequential left-to-right sums:

```text
slot_force_x[a]  = sum over k in atom_bond_indices[a] of bond_pair_x[k]
slot_force_y[a]  = same with y
slot_force_z[a]  = same with z
slot_energy[a]   = sum over k in atom_bond_indices[a] of bond_pair_energy[k]
slot_virial[a]   = sum over k in atom_bond_indices[a] of bond_pair_virial[k]
```

The `atom_bond_indices` slice for each atom is sorted by underlying bond
index at construction time, so the summation order is identical across
runs. Each thread maps to one atom; there are no atomics and no race
conditions.

## Parameters <!-- rq-4943810f -->

Each `[[bond_types]]` entry in the config that uses `potential =
"harmonic"` contributes one row to a per-bond-type parameter table
uploaded to the device:

- `k: f64` — force constant, Hartrees per Bohr² (`E_h / a_0²`) internally,
  in the `U = ½ k (r − r_0)²` convention. Required. Finite and strictly
  positive.
- `r0: f64` — equilibrium distance, Bohr (`a_0`). Required. Finite and
  strictly positive.

The parameter table on the device is two `CudaSlice<f32>` arrays
(`bond_k`, `bond_r0`), each of length `n_bond_types`, cast from `f64` to
`f32` at upload time and addressed by the bond's global `bond_type_index`.
Rows corresponding to non-harmonic bond types hold placeholder values that
this slot never reads.

## Empty State <!-- rq-8da40a65 -->

When no bond in the system uses a harmonic bond type (the selected subset
is empty), the `HarmonicBondState` is not constructed by the `ForceField`
and the slot is absent from the slot list. The framework's combiner
handles slot-presence correctly (see `framework.md`).

When `particle_count == 0`, the bond list must also be empty (the file
parser rejects any bond entry with an out-of-range atom index, and every
index is out of range when `N == 0`). The slot is therefore not
constructed.

## Feature API <!-- rq-de977e05 -->

### Types <!-- rq-bed893e5 -->

- `HarmonicBondState` — implements the `Potential` trait with <!-- rq-7d440d75 -->
  `label() == "harmonic_bond"` and `frequency_class() == ForceClass::Fast`
  (see `framework.md`). Fields:
  - `device: Arc<CudaDevice>`
  - `bonds: CudaSlice<u32>` — flat array of `[atom_i, atom_j,
    bond_type_index]` triples for the selected harmonic bonds, length
    `3 * B_h`, sorted by `(atom_i, atom_j)`.
  - `atom_bond_offsets: CudaSlice<u32>` — length `N + 1`.
  - `atom_bond_indices: CudaSlice<u32>` — length `2 * B_h`.
  - `bond_k: CudaSlice<f32>` — length `n_bond_types`.
  - `bond_r0: CudaSlice<f32>` — length `n_bond_types`.
  - `bond_pair_x: CudaSlice<f32>` — length `2 * B_h`, per-slot force x
    contribution.
  - `bond_pair_y: CudaSlice<f32>` — length `2 * B_h`.
  - `bond_pair_z: CudaSlice<f32>` — length `2 * B_h`.
  - `bond_pair_energy: CudaSlice<f32>` — length `2 * B_h`, per-slot
    half-energy contribution (`U_k / 2`).
  - `bond_pair_virial: CudaSlice<f32>` — length `2 * B_h`, per-slot
    half-virial contribution (`W_k / 2`).
  - `bond_count: usize` — `B_h`, the number of harmonic bonds.
  - `particle_count: usize`

  All fields private; the slot's public surface is the per-step methods
  invoked by `ForceField::step` (see `framework.md`).

  Constructor:

  - `HarmonicBondState::new(device: Arc<CudaDevice>, bond_list: &BondList, bond_types: &[BondTypeConfig]) -> Result<HarmonicBondState, GpuError>`
    - Selects the bonds of `bond_list` whose `bond_type_index` names a
      `BondTypeConfig::Harmonic` entry (see *Per-Potential Bond
      Selection*), preserving sort order.
    - Builds the slot's own `atom_bond_offsets` / `atom_bond_indices`
      reduction map over the selected subset.
    - Uploads the selected bond triples and reduction map to device
      memory.
    - Uploads the `bond_k` / `bond_r0` parameter tables (length
      `n_bond_types`), populating harmonic entries with the converted
      `k` and `r0`.
    - Allocates the five per-bond `bond_pair_*` buffers (force x/y/z,
      half-energy, half-virial), each of length `2 * B_h`. Per-atom
      output is added into the framework-supplied `SlotOutputView` (a
      view onto the slot's class accumulator; see `framework.md`'s *Class
      Output Accumulators*) during `reduce()`; the slot owns no per-atom
      accumulator buffers of its own.
    - When no harmonic bond is present, this method is not called by the
      `ForceField` — see *Empty State*.

### Source Fragment <!-- rq-9c9c1fef -->

`HarmonicBondBuilder::bonded_force_fragment(cx)` returns a
`BondedForceFragment` whose functor implements the per-bond harmonic
contribution. The fragment defines a `__device__` functor
`HarmonicPairFunctor` whose member function `evaluate(r2, r,
bond_type_index, dx, dy, dz, fmag, u_k, w_k)` computes:

1. Reads `k = bond_k[bond_type_index]` and `r0 = bond_r0[bond_type_index]`
   from device-buffer pointers held as members of the functor.
2. Computes `dr = r - r0` and the force factor
   `fmag = -k * dr / r` (the trailing `/ r` produces the per-component
   factor when multiplied by `(dx, dy, dz)` in the composed kernel's
   outer-loop body).
3. Writes `u_k = 0.5f * k * dr * dr` and `w_k = fmag * r2` (the bond's
   full potential energy and scalar virial; the outer-loop body
   distributes the `0.5` symmetry factor when writing to the scratch
   buffer).

When `r < 1.0e-7f` (degenerate overlapping atoms) the functor writes
`fmag = 0`, `u_k = 0`, `w_k = 0` rather than producing a division by a
near-zero length. This is a defensive guard; physical harmonic-bond
simulations never reach `r == 0`. Note that the true potential energy at
`r == 0` is `½ k r_0²` (finite and non-zero), but the guard drops it
because the force direction is undefined there — the same defensive
convention used by the harmonic-angle guard (`harmonic-angle.md`). The
outer-loop body then writes zeros to all ten scratch-buffer slots (five
quantities × two slots).

The composed kernel's outer-loop body (in the JIT-composed bonded module —
see `jit-composed-intramolecular.md`) handles the common-args reading:
reads the bond triple `(atom_i, atom_j, bond_type_index)`, computes the
minimum-image displacement `(dx, dy, dz) = (r_i - r_j)`, computes `r2` and
`r`, calls the functor's `evaluate`, then writes the per-atom force triples
`±fmag · (dx, dy, dz)` along with `u_k · 0.5` and `w_k · 0.5` into the
slot's bond-pair scratch buffer at indices `2·k` and `2·k + 1`. See
`jit-composed-intramolecular.md`'s *Composed-Module Structure* for the full
outer-loop body specification.

The fragment's `entry_point_args` declares the per-bond-type parameter
table pointers (`bond_k`, `bond_r0`); the `functor_init_source` assigns
them to the functor's members at the start of the entry-point body.

### Reduction Kernel <!-- rq-f184f707 -->

The slot reuses the shape-universal `reduce_bond_forces` kernel declared in
`kernels/bonded.cu` and specified in `morse-bonded.md`. No harmonic-bond
variant of the reduction kernel exists; the per-bond scratch buffer of any
bonded potential sums into per-atom forces the same way.

### PTX Module Loading <!-- rq-3bbb75a7 -->

The reduction kernel `reduce_bond_forces` is captured into the `Kernels`
handle by `init_device()` from the compiled `kernels/bonded.cu` PTX (module
`"bonded"`), as described in `morse-bonded.md`. The harmonic slot's per-bond
contribution is compiled into the bonded JIT module
(`"heddle_jit_composed_bonded"`) owned by the `ForceField` instance; see
`jit-composed-intramolecular.md`.

### Rust Launch Helpers <!-- rq-0174962e -->

The framework's per-step dispatch (see
`jit-composed-intramolecular.md`'s *Parameter Binding and Launch*) launches
the slot's composed bonded entry point and then the universal reduction
kernel. Slots do not expose standalone launchers for the contribution
kernel; participation in the JIT-composed module is the only path to
dispatch the per-bond contribution.

The reduction is launched through the framework's `reduce_bond_forces`
helper (shared with the Morse slot; see `morse-bonded.md`), summing each
atom's harmonic-bond contributions into the five caller-supplied output
views. It returns `Ok(())` without launching when `particle_count == 0`.

## Launch Configuration <!-- rq-1ca1e95a -->

- Composed bonded contribution kernel: block size 256, grid
  `ceil(bond_count / 256)`, no shared memory. Dispatched by the framework
  from the JIT-composed bonded module.
- Reduction kernel: block size 256, grid `ceil(particle_count / 256)`, no
  shared memory.
- Both run on the default stream carried by `particle_buffers.device`.

## Determinism <!-- rq-43517384 -->

- Each bond's force is computed by exactly one thread; no atomics.
- Each atom's reduction is computed by exactly one thread; sums proceed in
  sorted `atom_bond_indices` order.
- Bond selection preserves the `BondList`'s `(atom_i, atom_j)` sort order,
  so the derived reduction map is identical across runs.
- Two runs with identical bonds, parameters, and positions on the same GPU
  produce byte-identical `bond_pair_*` and accumulator contents.

## Out of Scope <!-- rq-1185d826 -->

- Other bonded potentials (Morse — see `morse-bonded.md`; FENE;
  Buckingham; class-2 quartic bonds). Each lands as a new `potential`
  value in `[[bond_types]]` with its own kernel.
- Angle, dihedral, and improper potentials.
- Per-bond parameter overrides (every bond gets its parameters via its
  bond type).
- Cutoff or switching variants of the harmonic bond. It is treated as
  full-range bonded.
- Bond breaking, forming, or reordering during a simulation.

---

## Gherkin Scenarios <!-- rq-74524d21 -->

```gherkin
Feature: Harmonic bonded potential

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called
    And a SimulationBox with lx=ly=lz=10.0

  # --- Construction and selection ---

  @rq-e2b77f30
  Scenario: Construct HarmonicBondState from an all-harmonic bond list
    Given a BondList with 3 bonds among 4 atoms
    And [[bond_types]] with entry "CC" potential="harmonic" k=2.0 r0=1.0
    And entry "CN" potential="harmonic" k=4.0 r0=1.5
    When HarmonicBondState::new(device, &bond_list, &bond_types) is called
    Then it returns Ok(state)
    And state.bond_count equals 3
    And state.particle_count equals 4
    And bond_k, bond_r0 on the device hold 2.0 and 4.0 (and 1.0, 1.5) at the "CC" and "CN" indices

  @rq-990d287d
  Scenario: Harmonic slot selects only harmonic bonds from a mixed list
    Given [[bond_types]] "CC" potential="harmonic" and "MM" potential="morse"
    And a BondList with two bonds: bond A of type "CC" and bond B of type "MM"
    When HarmonicBondState::new(device, &bond_list, &bond_types) is called
    Then state.bond_count equals 1
    And the selected bond is bond A
    And its reduction map covers only bond A's two atom shares

  @rq-c82ac64c
  Scenario: Selected reduction map matches a subset build
    Given a mixed BondList where atom 0 participates in one harmonic and one Morse bond
    When HarmonicBondState::new is called
    Then atom 0's atom_bond_indices slice references only the harmonic bond's slot

  # --- Force kernel correctness ---

  @rq-8eb932b7
  Scenario: Two atoms at equilibrium distance produce zero force
    Given positions p0=(0,0,0) and p1=(1.0, 0, 0)
    And a BondList with one bond (0, 1) of type "CC" with k=2.0, r0=1.0 (so r == r_0)
    When the harmonic bonded force is computed
    And the bond_pair buffer is downloaded
    Then bond_pair_x[0], bond_pair_y[0], bond_pair_z[0] are all 0.0_f32 within f32 round-off
    And bond_pair_x[1], bond_pair_y[1], bond_pair_z[1] are all 0.0_f32 within f32 round-off

  @rq-c3d498ad
  Scenario: Compressed bond produces a repulsive force
    Given positions p0=(0,0,0) and p1=(0.5, 0, 0)
    And bond (0, 1) of type "CC" with k=2.0, r0=1.0 (so r < r_0)
    When the harmonic bonded force is computed
    Then the force on atom 0 points in -x (away from atom 1, which lies in +x)
    And the force on atom 1 equals the negation of the force on atom 0 within f32 round-off

  @rq-86fb06cd
  Scenario: Stretched bond produces an attractive force
    Given positions p0=(0,0,0) and p1=(2.0, 0, 0)
    And bond (0, 1) of type "CC" with k=2.0, r0=1.0 (so r > r_0)
    When the harmonic bonded force is computed
    Then the force on atom 0 points in +x (toward atom 1)
    And the force on atom 1 equals the negation of the force on atom 0 within f32 round-off

  @rq-8a7c1b6a
  Scenario: Force magnitude matches the closed-form harmonic expression
    Given positions p0=(0,0,0) and p1=(1.3, 0, 0)
    And bond (0, 1) of type "CC" with k=2.0, r0=1.0
    When the harmonic bonded force is computed
    Then the magnitude of the force on atom 0 equals k * |r - r0| = 2.0 * 0.3 within f32 round-off

  @rq-2d7a2cab
  Scenario: Minimum image is applied
    Given lx=10.0 and positions p0=(-4.5, 0, 0), p1=(4.5, 0, 0)
    And bond (0, 1) of type "CC" with k=2.0, r0=1.0
    When the harmonic bonded force is computed
    Then the displacement used is dx=-1.0 (the periodic image), not dx=9.0

  @rq-eff785d2
  Scenario: r below the degenerate threshold produces zero force, not NaN
    Given two atoms at identical positions and a harmonic bond between them
    When the harmonic bonded force is computed
    Then every bond_pair_* slot is 0.0_f32

  # --- Energy and virial outputs ---

  @rq-1286e57c
  Scenario: A stretched bond's energy matches the half-convention closed form
    Given a BondList with one bond (atom 0 - atom 1) of type "CC" with k=2.0, r0=1.0
    And atoms placed at r = 1.5
    When the harmonic bonded force-and-scalars kernel is computed
    Then bond_pair_energy[0] + bond_pair_energy[1]
      equals 0.5 * k * (r - r0)^2 within f32 round-off

  @rq-1b14dbf1
  Scenario: A stretched bond's virial equals r * F_mag
    Given a BondList with one bond (atom 0 - atom 1) of type "CC" with k=2.0, r0=1.0
    And atoms placed at r = 1.5
    When the harmonic bonded force-and-scalars kernel is computed
    Then bond_pair_virial[0] + bond_pair_virial[1]
      equals r_ij · F_ij within f32 round-off, where F_ij is the force on atom 0 due to atom 1

  @rq-5c4d167d
  Scenario: Degenerate geometry produces zero energy and virial in addition to zero force
    Given two atoms placed at identical positions and a harmonic bond between them
    When the harmonic bonded force-and-scalars kernel is computed
    Then bond_pair_energy[0..2] and bond_pair_virial[0..2] are all 0.0_f32

  # --- Reduction ---

  @rq-6d5b52f0
  Scenario: Atom with two harmonic bonds receives the sum of contributions
    Given two harmonic bonds: (0,1) with force_on_0=1.5 and (0,2) with force_on_0=2.5
    When reduce_bond_forces is launched on the slot's buffers
    Then the accumulated force x on atom 0 equals 4.0 within f32 round-off

  # --- Empty states ---

  @rq-5728ee79
  Scenario: A system with only Morse bonds does not construct a HarmonicBondState
    Given a BondList whose every bond uses a "morse" bond type
    When the ForceField is built
    Then no slot with label "harmonic_bond" is present

  @rq-46ef4201
  Scenario: harmonic bonded force on zero harmonic bonds is a no-op
    Given a HarmonicBondState with bond_count == 0
    When its compute is called
    Then it returns Ok(())

  @rq-5e8cef9e
  Scenario: reduce on zero particles is a no-op
    Given a HarmonicBondState with particle_count == 0
    When its reduce is called
    Then it returns Ok(())

  # --- Reproducibility ---

  @rq-65d5cf1c
  Scenario: Two independent calls produce byte-identical accumulators
    Given two independently-constructed HarmonicBondStates with identical bond list
      and parameters and a ParticleBuffers built from identical positions
    When the force then reduction is computed on each
    And both accumulator buffers are downloaded
    Then they agree byte-for-byte

  # --- End-to-end through the framework ---

  @rq-f523bca1
  Scenario: Diatomic equilibrium gives zero net force on both atoms
    Given a 2-atom system with atoms at r_0, one harmonic bond, and no LJ interaction
      (cutoff < bond length)
    When force_field.step(...) is called
    And the buffers are downloaded
    Then forces on both atoms are zero within f32 round-off

  @rq-80eff754
  Scenario: Newton's third law holds for the framework's combined force
    Given a 2-atom harmonic-bonded system inside the LJ cutoff
    When force_field.step(...) is called
    And the buffers are downloaded
    Then forces_x[0] + forces_x[1] equals 0 within f32 round-off
    And similarly for y and z

  @rq-d757f7a1
  Scenario: A mixed Morse-and-harmonic system routes each bond to exactly one slot
    Given a system with one Morse bond and one harmonic bond sharing no atoms
    When force_field.step(...) is called
    Then the total force on each atom equals the contribution of its own bond only
    And no bond is double-counted
```
