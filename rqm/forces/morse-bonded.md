# Feature: Morse Bonded Potential <!-- rq-05d55351 -->

The `MorseBonded` potential slot evaluates a Morse bond force for each bond
in the system's bond list (see `bonds.md`). Bonds are pairs of atoms whose
distance interaction is described by the Morse functional form with
per-bond-type parameters. The slot plugs into the pluggable potential
framework (`framework.md`); selection is implicit — the slot is present
whenever the config's `bonds` field references a non-empty `.bonds` file
and at least one `[[bond_types]]` entry has `potential = "morse"`.

## Algorithm <!-- rq-cbeeea3c -->

The Morse pair potential between atoms `i` and `j` at separation
`r = |r_i - r_j|` is

```text
U(r) = D_e * (1 - exp(-a * (r - r_e)))^2
F(r) = -dU/dr = 2 * D_e * a * (1 - exp(-a * (r - r_e))) * exp(-a * (r - r_e))
```

where the bond-type parameters are `D_e` (well depth, J), `a` (width
parameter, 1/m), and `r_e` (equilibrium distance, m). The minimum-image
displacement between the two atoms is used so the kernel honours
periodic boundary conditions; the bond is *not* truncated by any cutoff
distance — Morse is intended for short-range bonded use where atoms
remain near equilibrium.

For bond `k` connecting `atom_i` and `atom_j`, the kernel computes the
direction vector `d_hat = (r_i - r_j) / r` (the minimum-image
displacement, normalised) and the scalar force magnitude `F(r)`. The
force on `atom_i` is `+F * d_hat`; the force on `atom_j` is `-F * d_hat`
(Newton's third law applied at the kernel level). The two contributions
are written to consecutive slots in the bond-pair buffer (see *Force
Accumulation* below).

## Per-Step Kernel Sequence <!-- rq-100f8b5f -->

The slot's contribution kernel and reduction kernel run once each per
step:

| Step | Kernel | Operation | Stage label |
| --- | --- | --- | --- |
| 1 | `morse_bond_force` | compute force per bond, write to bond-pair buffer | `MorseBondForce` |
| 2 | `reduce_bond_forces` | per-atom sum of bond contributions, write to slot accumulator | `ReduceBondForces` |

The combiner (`AccumulateForces`) is run by the framework after every
slot's reduction. See `framework.md` for the slot order.

## Force Accumulation <!-- rq-0c318e64 -->

The slot owns a `BondPairBuffer` of length `2 * B` where `B` is the
number of bonds. Slot `2 * k` holds the force vector applied to
`atom_i` of bond `k`; slot `2 * k + 1` holds the force on `atom_j`.

The reduction kernel reads the precomputed `atom_bond_offsets` /
`atom_bond_indices` tables (see `bonds.md`) and sums each atom's
contributions in fixed order. For atom `a`:

```text
slot_acc_x[a] = sum over k in atom_bond_indices[a] of bond_pair_x[k]
slot_acc_y[a] = same with y
slot_acc_z[a] = same with z
```

The `atom_bond_indices` slice for each atom is sorted by underlying
bond index at file-load time, so the summation order is identical across
runs. Each thread maps to one atom; there are no atomics and no race
conditions.

## Parameters <!-- rq-12872970 -->

Each `[[bond_types]]` entry in the config that uses `potential = "morse"`
contributes one row to a per-bond-type parameter table uploaded to the
device:

- `de: f64` — well depth, joules. Required. Finite and strictly positive.
- `a: f64` — width parameter, inverse metres. Required. Finite and
  strictly positive.
- `re: f64` — equilibrium distance, metres. Required. Finite and
  strictly positive.

The parameter table on the device is three `CudaSlice<f32>` arrays
(`de`, `a`, `re`), one per bond type, cast from `f64` to `f32` at
upload time. Each bond carries a `bond_type_index` (see `bonds.md`)
into this table.

In v1 the only supported `potential` value for bond types is `"morse"`;
other values are rejected at config-load time. Future bonded potentials
(harmonic, cosine, etc.) will add new `potential` values and reuse the
existing `BondList` / `BondPairBuffer` / reduction infrastructure.

## Empty State <!-- rq-21acd57c -->

When the bond list is empty (`bond_list.is_empty()`), the
`MorseBondedState` is not constructed by the `ForceField` and the slot
is absent from the slot list. The framework's combiner handles
slot-presence correctly (see `framework.md`).

When `particle_count == 0`, the bond list must also be empty (the file
parser rejects any bond entry with an out-of-range atom index, and
every index is out of range when `N == 0`). The slot is therefore not
constructed.

## Feature API <!-- rq-345d7784 -->

### Types <!-- rq-976aa4af -->

- `MorseBondedState` — variant payload owned by <!-- rq-ec18d174 -->
  `PotentialSlot::MorseBonded` (see `framework.md`). Fields:
  - `device: Arc<CudaDevice>`
  - `bonds: CudaSlice<u32>` — flat array of `[atom_i, atom_j,
    bond_type_index]` triples, length `3 * B`, sorted to match
    `BondList::bonds`.
  - `atom_bond_offsets: CudaSlice<u32>` — length `N + 1`.
  - `atom_bond_indices: CudaSlice<u32>` — length `2 * B`.
  - `bond_de: CudaSlice<f32>` — length `n_bond_types`.
  - `bond_a: CudaSlice<f32>` — length `n_bond_types`.
  - `bond_re: CudaSlice<f32>` — length `n_bond_types`.
  - `bond_pair_x: CudaSlice<f32>` — length `2 * B`, the per-bond
    contribution buffer for x components.
  - `bond_pair_y: CudaSlice<f32>` — length `2 * B`.
  - `bond_pair_z: CudaSlice<f32>` — length `2 * B`.
  - `accumulator_x: CudaSlice<f32>` — length `N`, the slot's private
    per-atom accumulator.
  - `accumulator_y: CudaSlice<f32>` — length `N`.
  - `accumulator_z: CudaSlice<f32>` — length `N`.
  - `bond_count: usize`
  - `particle_count: usize`

  All fields private; the slot's public surface is the per-step methods
  invoked by `ForceField::step` (see `framework.md`).

  Constructor:

  - `MorseBondedState::new(device: Arc<CudaDevice>, bond_list: &BondList, bond_types: &[BondTypeConfig]) -> Result<MorseBondedState, GpuError>`
    - Filters `bond_types` to entries with `potential == "morse"` and
      uploads their parameters.
    - Uploads `bond_list.bonds`, `bond_list.atom_bond_offsets`, and
      `bond_list.atom_bond_indices` to device memory.
    - Allocates the per-bond `bond_pair_*` buffers
      (`3 * 2 * B * 4 bytes`) and the per-atom `accumulator_*`
      buffers (`3 * N * 4 bytes`).
    - When `bond_list.is_empty()`, this method is not called by the
      `ForceField` — see *Empty State*.

### CUDA Kernels <!-- rq-d28ad917 -->

`kernels/morse.cu` declares two `extern "C"` kernels:

```c
extern "C" __global__ void morse_bond_force(
    const float *positions_x, const float *positions_y, const float *positions_z,
    const unsigned int *bonds,
    const float *bond_de, const float *bond_a, const float *bond_re,
    float lx, float ly, float lz,
    float *bond_pair_x, float *bond_pair_y, float *bond_pair_z,
    unsigned int n_bonds);

extern "C" __global__ void reduce_bond_forces(
    const float *bond_pair_x, const float *bond_pair_y, const float *bond_pair_z,
    const unsigned int *atom_bond_offsets,
    const unsigned int *atom_bond_indices,
    float *accumulator_x, float *accumulator_y, float *accumulator_z,
    unsigned int n);
```

#### `morse_bond_force`

One thread per bond. Thread `k`:

1. Reads `atom_i`, `atom_j`, `type_idx` from `bonds[3*k .. 3*k + 3]`.
2. Computes the minimum-image displacement
   `(dx, dy, dz) = (r_i - r_j)` wrapped against `(lx, ly, lz)` using
   the same formula as `lj_pair_force`.
3. Computes `r2 = dx² + dy² + dz²` and `r = sqrt(r2)`.
4. Reads `De = bond_de[type_idx]`, `a = bond_a[type_idx]`,
   `re = bond_re[type_idx]`.
5. Computes `e = exp(-a * (r - re))` and the force magnitude
   `Fmag = 2 * De * a * (1 - e) * e / r` (the trailing `/ r` produces
   the per-component factor when multiplied by `(dx, dy, dz)`).
6. Writes the force on `atom_i` to `bond_pair_*[2 * k]` as
   `Fmag * (dx, dy, dz)`.
7. Writes the force on `atom_j` to `bond_pair_*[2 * k + 1]` as
   `-Fmag * (dx, dy, dz)`.

When `r == 0` exactly (degenerate overlapping atoms), the kernel
writes `0` to both slots rather than producing NaN. This is a defensive
guard; physical Morse simulations never reach `r == 0` because the
exponential blows up at small `r`.

#### `reduce_bond_forces`

One thread per atom `a = blockIdx.x * blockDim.x + threadIdx.x` (block
size 256, grid `ceil(n / 256)`). Thread `a`:

1. Reads `start = atom_bond_offsets[a]` and `end =
   atom_bond_offsets[a + 1]`.
2. Initialises `sum_x = 0`, `sum_y = 0`, `sum_z = 0`.
3. For each `i` in `start .. end`:
   `slot = atom_bond_indices[i];
    sum_x += bond_pair_x[slot]; (similarly y, z)`.
4. Writes `accumulator_x[a] = sum_x`, `accumulator_y[a] = sum_y`,
   `accumulator_z[a] = sum_z`.

The summation is left-to-right in `atom_bond_indices` order. Since
the indices are sorted at load time, the order is deterministic.

### PTX Module Loading <!-- rq-aa36ee0b -->

`init_device()` loads the compiled `kernels/morse.cu` PTX with module
name `"morse"` and registers function names `"morse_bond_force"` and
`"reduce_bond_forces"`, alongside the existing modules.

### Rust Launch Helpers <!-- rq-637c4fdd -->

Two free functions in `src/gpu/kernels.rs`, re-exported from
`crate::gpu`:

- `morse_bond_force(state: &mut MorseBondedState, particle_buffers: &ParticleBuffers, sim_box: &SimulationBox) -> Result<(), GpuError>` <!-- rq-66d80d54 -->
  - Launches the `morse_bond_force` kernel.
  - Block size 256; grid size `ceil(state.bond_count / 256)`.
  - Returns `Ok(())` without launching when `state.bond_count == 0`.
  - Panics if the `"morse"` module is not loaded.

- `reduce_bond_forces(state: &mut MorseBondedState) -> Result<(), GpuError>` <!-- rq-10adebc4 -->
  - Launches the `reduce_bond_forces` kernel.
  - Block size 256; grid size `ceil(state.particle_count / 256)`.
  - Returns `Ok(())` without launching when
    `state.particle_count == 0`.
  - Panics if the `"morse"` module is not loaded.

## Launch Configuration <!-- rq-c1678fe1 -->

- Block size: 256 threads for both kernels.
- Grid size: `ceil(bond_count / 256)` for the force kernel,
  `ceil(particle_count / 256)` for the reduction.
- Shared memory: zero bytes.
- Stream: the default stream carried by `particle_buffers.device`.

## Determinism <!-- rq-e5ba2e00 -->

- Each bond's force is computed by exactly one thread; no atomics.
- Each atom's reduction is computed by exactly one thread; sums
  proceed in sorted `atom_bond_indices` order.
- Two runs with identical bonds, parameters, and positions on the same
  GPU produce byte-identical `bond_pair_*` and `accumulator_*`
  contents.

## Out of Scope <!-- rq-c79e35dc -->

- Other bonded potentials (harmonic bonds, FENE, Buckingham, etc.).
  Each lands as a new `potential` value in `[[bond_types]]` with its
  own kernel.
- Angle, dihedral, and improper potentials.
- Per-bond parameter overrides (every bond gets its parameters via its
  bond type).
- Long-range / cutoff variants of Morse. Morse is treated as
  full-range bonded.
- Pressure and virial accumulation. Only forces are computed.
- A "soft" Morse variant that smoothly switches off for large `r`.
- Bond breaking, forming, or reordering during a simulation.

---

## Gherkin Scenarios <!-- rq-bf6cc1aa -->

```gherkin
Feature: Morse bonded potential

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called
    And a SimulationBox with lx=ly=lz=10.0

  # --- Module loading ---

  @rq-679282f5
  Scenario: init_device loads the morse module
    When init_device() is called
    Then the device exposes a function named "morse_bond_force" in module "morse"
    And the device exposes a function named "reduce_bond_forces" in module "morse"

  # --- Construction ---

  @rq-9f2de58c
  Scenario: Construct MorseBondedState
    Given a BondList with 3 bonds among 4 atoms and two bond types
    And [[bond_types]] with one entry "CC" potential="morse" de=1.0 a=2.0 re=1.0
    And one entry "CN" potential="morse" de=2.0 a=3.0 re=1.5
    When MorseBondedState::new(device, &bond_list, &bond_types) is called
    Then it returns Ok(state)
    And state.bond_count equals 3
    And state.particle_count equals 4
    And bond_de, bond_a, bond_re on the device equal [1.0, 2.0], [2.0, 3.0], [1.0, 1.5]

  # --- Force kernel correctness ---

  @rq-2e4e70b4
  Scenario: Two atoms at equilibrium distance produce zero force
    Given a ParticleBuffers with positions p0=(0,0,0) and p1=(1.0, 0, 0)
    And a BondList with one bond (0, 1) of type "CC"
    And bond_types "CC" with de=1.0, a=2.0, re=1.0 (so r == r_e)
    When morse_bond_force is launched
    And the bond_pair buffer is downloaded
    Then bond_pair_x[0], bond_pair_y[0], bond_pair_z[0] are all 0.0_f32 within f32 round-off
    And bond_pair_x[1], bond_pair_y[1], bond_pair_z[1] are all 0.0_f32 within f32 round-off

  @rq-f79657d2
  Scenario: Compressed bond produces a repulsive force
    Given positions p0=(0,0,0) and p1=(0.5, 0, 0)
    And bond (0, 1) of type "CC" with de=1.0, a=2.0, re=1.0
    When morse_bond_force is launched
    Then bond_pair_x[0] is negative (force on atom 0 points in -x, away from atom 1 which is in +x ... wait, atom 0 at origin, atom 1 at +x; compressed means r < r_e so the force is repulsive, atom 0 pushed in -x)
    And bond_pair_x[1] equals -bond_pair_x[0] (Newton's third law within f32 round-off)

  @rq-2cb90e10
  Scenario: Stretched bond produces an attractive force
    Given positions p0=(0,0,0) and p1=(2.0, 0, 0)
    And bond (0, 1) of type "CC" with de=1.0, a=2.0, re=1.0 (so r > r_e)
    When morse_bond_force is launched
    Then bond_pair_x[0] is positive (atom 0 pulled toward atom 1 which is in +x)
    And bond_pair_x[1] equals -bond_pair_x[0]

  @rq-d61fa682
  Scenario: Force magnitude matches closed-form Morse expression
    Given positions p0=(0,0,0) and p1=(1.2, 0, 0)
    And bond (0, 1) of type "CC" with de=1.0, a=2.0, re=1.0
    When morse_bond_force is launched
    Then |bond_pair_x[0]| equals 2 * 1.0 * 2.0 * (1 - exp(-2*0.2)) * exp(-2*0.2) within f32 round-off

  @rq-556b7c13
  Scenario: Minimum image is applied
    Given lx=10.0 and positions p0=(-4.5, 0, 0), p1=(4.5, 0, 0)
    And bond (0, 1) of type "CC" with de=1.0, a=2.0, re=1.0
    When morse_bond_force is launched
    Then the displacement used is dx=-1.0 (the periodic image), not dx=9.0

  @rq-4811af60
  Scenario: r=0 produces zero force, not NaN
    Given two atoms at identical positions and a bond between them
    When morse_bond_force is launched
    Then every bond_pair_* slot is 0.0_f32

  # --- Reduction kernel correctness ---

  @rq-2d4efead
  Scenario: Atom with one bond receives the bond's force directly
    Given a single bond with bond_pair_x[0]=2.0, bond_pair_x[1]=-2.0
    And atom_bond_offsets=[0, 1, 2] (atom 0 receives slot 0, atom 1 receives slot 1)
    And atom_bond_indices=[0, 1]
    When reduce_bond_forces is launched
    Then accumulator_x[0] equals 2.0
    And accumulator_x[1] equals -2.0

  @rq-1ce4ce5a
  Scenario: Atom with two bonds receives sum of contributions
    Given two bonds: (0,1) with force_on_0=1.5 and (0,2) with force_on_0=2.5
    And atom 0 contributes slot 0 (from bond 0) and slot 2 (from bond 1)
    When reduce_bond_forces is launched
    Then accumulator_x[0] equals 4.0 within f32 round-off

  @rq-55f89976
  Scenario: Reduction summation order is sorted bond index
    Given atom 0 with bond contributions from bonds 0 and 1 in slot order [0, 2]
    When reduce_bond_forces is launched
    Then accumulator_x[0] equals bond_pair_x[0] + bond_pair_x[2] (left-to-right)

  @rq-1ca90a29
  Scenario: Atom with no bonds gets zero accumulator
    Given a 4-atom system with bonds only on atoms 0..3 (atom 3 has no bond)
    When reduce_bond_forces is launched
    Then accumulator_x[3], accumulator_y[3], accumulator_z[3] are all 0.0

  # --- Empty states ---

  @rq-62e2469f
  Scenario: morse_bond_force on zero bonds is a no-op
    Given a MorseBondedState with bond_count == 0
    When morse_bond_force is called
    Then it returns Ok(())

  @rq-966e43ed
  Scenario: reduce_bond_forces on zero particles is a no-op
    Given a MorseBondedState with particle_count == 0
    When reduce_bond_forces is called
    Then it returns Ok(())

  # --- Reproducibility ---

  @rq-696caf8e
  Scenario: Two independent calls produce byte-identical accumulators
    Given two independently-constructed MorseBondedStates with identical bond list
      and parameters and a ParticleBuffers built from identical positions
    When morse_bond_force then reduce_bond_forces is launched on each
    And both accumulator_* buffers are downloaded
    Then they agree byte-for-byte

  # --- End-to-end through the framework ---

  @rq-c7af1f28
  Scenario: Diatomic equilibrium gives zero net force on both atoms
    Given a 2-atom system with atoms at r_e, one bond, and no LJ interaction
      (cutoff < bond length)
    When force_field.step(...) is called
    And the buffers are downloaded
    Then forces_* on both atoms are zero within f32 round-off

  @rq-6d06e36e
  Scenario: Newton's third law holds for the framework's combined force
    Given a 2-atom Morse-bonded system inside the LJ cutoff
    When force_field.step(...) is called
    And the buffers are downloaded
    Then forces_x[0] + forces_x[1] equals 0 within f32 round-off
    And similarly for y and z

  # --- Rejection of non-Morse bond types in v1 ---

  @rq-1fc667cd
  Scenario: Config bond_type with potential != "morse" is rejected
    Given a [[bond_types]] entry with potential="harmonic"
    When the config is loaded
    Then it returns Err(ConfigError::InvalidValue { field: "bond_types[0].potential", reason: _ })
```
