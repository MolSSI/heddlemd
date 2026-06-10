# Feature: Topology File, Bond List, Angle List, Exclusion List, and Constraint List <!-- rq-9e1eee68 -->

A simulation's bonded topology is described by a `.topology` file referenced
from the TOML config (`io/config-schema.md`) and consumed by the
`MorseBonded` slot (`morse-bonded.md`), the `HarmonicAngle` slot
(`harmonic-angle.md`), the `Constraint` slot
(`integration/constraint-framework.md`, `integration/shake.md`), and the
Lennard-Jones and Coulomb slots' exclusion logic (`lj-pair-force.md`,
`coulomb-pair-force.md`). The file lists bond instances, angle instances,
per-pair non-bonded exclusions, and rigid constraint groups; bond, angle,
and constraint *types* (parameters) live in the config alongside particle
types.

The file produces four host-side structures: a `BondList` (bonds, with
precomputed per-atom indexing tables for deterministic reduction), an
`AngleList` (angles, with the same indexing pattern), an `ExclusionList`
(per-pair scaling factors for non-bonded interactions between bonded,
angle-coupled, constraint-coupled, or otherwise excluded atoms), and a
`ConstraintList` (rigid constraint groups; see
`integration/constraint-framework.md` for the SoA layout and the kernel
contract).

## File Format <!-- rq-a33c1f4f -->

The `.topology` file is UTF-8 text organised into four named sections,
`[bonds]`, `[exclusions]`, `[angles]`, and `[constraints]`. Each section
may appear at most once; each may be empty; each may be absent (an absent
`[bonds]` means no bonds, an absent `[exclusions]` means no explicit
exclusions, an absent `[angles]` means no angles, an absent
`[constraints]` means no rigid constraint groups). Section headers are
case-sensitive and must appear on their own line. Sections may appear
in any order.

```
# Comments start with '#' and run to end of line. Blank lines are <!-- rq-38285db7 -->
# ignored. <!-- rq-2e98a75d -->

[bonds]
# Column format: atom_i  atom_j  bond_type_name <!-- rq-cf456bb4 -->
# Atom indices are 0-based and refer to entries in the init file. <!-- rq-955b55f0 -->
0 1 OH
0 2 OH

[angles]
# Column format: atom_i  atom_j  atom_k  angle_type_name <!-- rq-6cac8251 -->
# atom_j is the centre (vertex) atom; atom_i and atom_k are the wings. <!-- rq-443a91da -->
# Atom indices are 0-based and refer to entries in the init file. <!-- rq-147f5ea6 -->
1 0 2 HOH

[exclusions]
# Column format: atom_i  atom_j  [scale_lj]  [scale_coul] <!-- rq-c8ea0a96 -->
# Both scales default to 0.0 (full exclusion) when omitted. <!-- rq-922c9a86 -->
# A single scale column (3-column form) sets both scale_lj and <!-- rq-10956a4e -->
# scale_coul to that value. <!-- rq-d7675fb3 -->
1 2 0.5 0.833

[constraints]
# Column format: atom_1 atom_2 ... atom_k  constraint_type_name <!-- rq-eadadcb4 -->
# Each row declares one rigid constraint group of k atoms. <!-- rq-ab21511c -->
# The atom-listing order is preserved verbatim: algorithm-specific <!-- rq-3f889ad8 -->
# conventions are encoded by position (for SHAKE-constrained rigid waters, the first <!-- rq-6dfec7f4 -->
# atom is the oxygen and the next two are the hydrogens). <!-- rq-7e2c482a -->
# atom_count per row is determined by the constraint_type's `kind`. <!-- rq-94cb21a5 -->
0 1 2 SPCE
```

Inside a section, columns are separated by ASCII whitespace (one or
more space/tab characters). The number of columns per line must match
the section's expected layout; trailing whitespace is tolerated.

### Bond entries <!-- rq-33a054f9 -->

Each non-comment line in `[bonds]` has the form `atom_i atom_j
bond_type_name` where:

- `atom_i: u32` and `atom_j: u32` are zero-based particle indices. Both
  must be `< particle_count`. `atom_i == atom_j` is rejected
  (self-bond).
- `bond_type_name: String` must match the `name` field of an entry in
  the config's `[[bond_types]]` array. Unknown names are rejected.

Bond *order* in the section is preserved for diagnostics. Internally
the parser canonicalises each `(atom_i, atom_j)` pair to
`(min, max)` and sorts the resulting list by `(min, max)` before
assigning bond indices. Two entries that normalise to the same pair
are rejected as duplicates regardless of order in the file.

### Angle entries <!-- rq-73054857 -->

Each non-comment line in `[angles]` has the form `atom_i atom_j atom_k
angle_type_name` where:

- `atom_i: u32`, `atom_j: u32`, and `atom_k: u32` are zero-based
  particle indices. All three must be `< particle_count`. `atom_j` is
  the centre (vertex) atom; `atom_i` and `atom_k` are the wings. The
  geometric angle is the one subtended at `atom_j` by the rays
  `r_i - r_j` and `r_k - r_j`.
- Repeated atoms among `(atom_i, atom_j, atom_k)` are rejected: any of
  `atom_i == atom_j`, `atom_j == atom_k`, or `atom_i == atom_k`
  produces a parser error.
- `angle_type_name: String` must match the `name` field of an entry in
  the config's `[[angle_types]]` array. Unknown names are rejected.

Angle *order* in the section is preserved for diagnostics. Internally
the parser canonicalises each `(atom_i, atom_j, atom_k)` triple by
swapping the two wings so that `atom_i < atom_k` while leaving the
centre `atom_j` in place. The canonicalised triples are then sorted by
`(atom_j, atom_i, atom_k)` before assigning angle indices. Two entries
that normalise to the same triple (same centre and same unordered wing
pair) are rejected as duplicates regardless of order in the file.

### Exclusion entries <!-- rq-4ae7794c -->

Each non-comment line in `[exclusions]` has one of three forms:

- `atom_i atom_j` — both `scale_lj` and `scale_coul` default to `0.0`
  (full exclusion of the pair from both non-bonded potentials).
- `atom_i atom_j scale` — single scale applied identically to LJ and
  Coulomb: `scale_lj = scale_coul = scale`.
- `atom_i atom_j scale_lj scale_coul` — independent per-potential
  scales.

Per-potential scaling lets a `.topology` file express force-field
conventions like AMBER's 1-4 scaling, where LJ-1-4 = 0.5 and
Coulomb-1-4 = 1/1.2 ≈ 0.833.

- `atom_i: u32` and `atom_j: u32` are zero-based particle indices. Both
  must be `< particle_count`. `atom_i == atom_j` is rejected.
- `scale_lj: f32` and `scale_coul: f32` when present are finite numbers
  in `[0.0, 1.0]`. Out-of-range values, NaN, and infinity are rejected.

Like bonds, exclusions are canonicalised to `(min, max)` and sorted by
`(min, max)`. Duplicate pairs (after canonicalisation) are rejected.

### Constraint entries <!-- rq-5f29f928 -->

Each non-comment line in `[constraints]` has the form
`atom_1 atom_2 ... atom_k constraint_type_name` where:

- Every `atom_*: u32` is a zero-based particle index. Every index must
  be `< particle_count`. No atom may repeat within a single row
  (`SelfConstraint`).
- The atom count `k` is determined by the algorithm consumed by the
  named constraint type. For `kind = "shake"`, `k` equals the
  constraint type's declared `atoms` field. Wrong atom count is
  rejected as `InvalidConstraintRow` with a reason that names the
  expected count.
- `constraint_type_name: String` must match the `name` field of an
  entry in the config's `[[constraint_types]]` array. Unknown names
  are rejected as `UnknownConstraintType`.

Each row defines one *constraint group*. v1 requires that the
constraint graph's connected components match rows one-to-one: no atom
may appear in more than one `[constraints]` row. Two rows that share
any atom are rejected as `DuplicateConstraintAtom`. (Future M-SHAKE
work will lift this restriction; the data layout already accommodates
multi-row groups via the connected-component build described in
`integration/constraint-framework.md`.)

The atom-listing order within a row is preserved verbatim through the
parsed `ConstraintList`. The named constraint type's `constraints`
table references atoms by their position in this row order; algorithms
that depend on which atom plays which role (e.g. SHAKE-constrained
rigid water, where atom 0 is the oxygen by convention) encode the
convention in the constraint type entry, not in the topology row.

A pair of atoms `(min, max)` that appears as a constraint pair (every
1-2 pair generated by the group's expansion into pairwise constraints
— see `integration/constraint-framework.md`) is forbidden from also
appearing in `[bonds]`. Conflicting rows are rejected as
`BondIsAlsoConstraint`.

### Effective exclusions <!-- rq-24f280af -->

After parsing, the consumer-facing `ExclusionList` is the *effective
exclusion list*, formed by combining the explicit `[exclusions]`
entries with implicit exclusions derived from `[bonds]`, `[angles]`,
and `[constraints]`:

- Every explicit `(i, j, scale_lj, scale_coul)` entry from the file
  becomes an effective exclusion with those two scales.
- For every bond `(i, j)` in `[bonds]` that does **not** have a
  matching explicit `(i, j, _, _)` entry, an implicit exclusion
  `(i, j, 0.0, 0.0)` is added.
- For every angle `(i, j, k)` in `[angles]`, the 1-3 pair `(i, k)` is
  considered. When `(i, k)` does **not** have a matching explicit
  `(i, k, _, _)` entry *and* is not already covered by an implicit
  bond-derived entry, an implicit exclusion `(i, k, 0.0, 0.0)` is
  added.
- For every constraint group in `[constraints]`, every pair `(p, q)`
  of distinct atoms drawn from the group's atom set is considered.
  When `(p, q)` does **not** have a matching explicit `(p, q, _, _)`
  entry *and* is not already covered by an implicit bond-derived or
  angle-derived entry, an implicit exclusion `(p, q, 0.0, 0.0)` is
  added. For a SHAKE-constrained rigid-water group `(O, H1, H2)` this
  produces three implicit exclusions: `(O, H1)`, `(O, H2)`, and
  `(H1, H2)`.

The result is the set of `(i, j, scale_lj, scale_coul)` tuples
consulted by the LJ and Coulomb pair-force kernels. The LJ kernel reads
`scale_lj`; the Coulomb kernel reads `scale_coul`. Explicit entries take
precedence over both implicit bond-derived and implicit angle-derived
entries; an explicit entry with `scale = 1.0` therefore *keeps* the
corresponding non-bonded contribution for a bonded or angle-coupled
pair, which is unusual physics but is the user's deliberate override.

Effective exclusions for pairs that are neither bonded, angle-coupled,
nor explicitly listed are absent from the list; the LJ and Coulomb
kernels treat them as `scale = 1.0` (no scaling).

The framework treats these two implicit-exclusion rules (1-2 from
bonds, 1-3 from angles) as the only default behaviour that affects
simulation results without an explicit user declaration. They are
documented here, in `lj-pair-force.md`, and in `coulomb-pair-force.md`.

### Empty file <!-- rq-1c794f95 -->

A `.topology` file containing zero bonds, zero exclusions, zero
angles, and zero constraints is valid (all sections empty or absent).
The runner produces an empty `BondList`, an empty `AngleList`, an
empty `ExclusionList`, and an empty `ConstraintList`; the
`MorseBonded`, `HarmonicAngle`, and `Constraint` slots are not
constructed.

## Data Model <!-- rq-361623d8 -->

### Bond list <!-- rq-f1210419 -->

For `B` bonds among `N` particles, the host-side `BondList` carries:

- `bonds: Vec<Bond>` — length `B`, sorted by `(atom_i, atom_j)`. Each
  `Bond` records `atom_i: u32`, `atom_j: u32` (with `atom_i < atom_j`),
  and `bond_type_index: u32` (an index into the config's
  `[[bond_types]]` array).
- `atom_bond_offsets: Vec<u32>` — length `N + 1`. For atom `a`, the
  slice
  `atom_bond_indices[atom_bond_offsets[a] .. atom_bond_offsets[a+1]]`
  lists the slot indices in the bond-pair buffer that contribute force
  to atom `a`.
- `atom_bond_indices: Vec<u32>` — length `2 * B` (each bond appears
  twice, once for `atom_i` and once for `atom_j`). Entry `k` is an
  index into the per-bond force buffer of length `2 * B`. The entries
  within a single atom's slice are sorted by the underlying bond
  index, so the reduction's summation order is deterministic across
  runs.

The atom-to-bond indexing is built at load time and uploaded to the
device once. Bonds are immutable for the lifetime of a run; no
recomputation is necessary.

### Angle list <!-- rq-7f8dba1b -->

For `A` angles among `N` particles, the host-side `AngleList` carries:

- `angles: Vec<Angle>` — length `A`, sorted by `(atom_j, atom_i,
  atom_k)`. Each `Angle` records `atom_i: u32`, `atom_j: u32`,
  `atom_k: u32` (the centre is `atom_j`; the wings satisfy
  `atom_i < atom_k`), and `angle_type_index: u32` (an index into the
  config's `[[angle_types]]` array).
- `atom_angle_offsets: Vec<u32>` — length `N + 1`. For atom `a`, the
  slice
  `atom_angle_indices[atom_angle_offsets[a] .. atom_angle_offsets[a+1]]`
  lists the slot indices in the angle-triple buffer that contribute
  force to atom `a`.
- `atom_angle_indices: Vec<u32>` — length `3 * A` (each angle appears
  three times, once for each of its three atoms). Entry `m` is an
  index into the per-angle force buffer of length `3 * A`. Within each
  atom's slice, entries are sorted by the underlying angle index so
  the reduction's summation order is deterministic across runs.

The atom-to-angle indexing is built at load time and uploaded to the
device once. Angles are immutable for the lifetime of a run; no
recomputation is necessary.

### Exclusion list <!-- rq-1e1b4e02 -->

For `E` effective exclusions, the host-side `ExclusionList` carries:

- `entries: Vec<Exclusion>` — length `E`, sorted by `(atom_i, atom_j)`
  with `atom_i < atom_j`. Each `Exclusion` records `atom_i: u32`,
  `atom_j: u32`, `scale_lj: f32`, and `scale_coul: f32`.
- `atom_excl_offsets: Vec<u32>` — length `N + 1`. For atom `a`, the
  slice
  `atom_excl_partners[atom_excl_offsets[a] .. atom_excl_offsets[a+1]]`
  lists the *partners* (`u32`) of `a` in the effective exclusion list,
  paired with their per-potential scale factors.
- `atom_excl_partners: Vec<u32>` — length `2 * E`.
- `atom_excl_lj_scales: Vec<f32>` — length `2 * E`, parallel to
  `atom_excl_partners`. Consumed by the LJ kernel.
- `atom_excl_coul_scales: Vec<f32>` — length `2 * E`, parallel to
  `atom_excl_partners`. Consumed by the Coulomb kernel.

Pair-potential CUDA kernels consult this per-atom lookup table through
the shared device helper `exclusion_scale` declared in
`kernels/exclusions.cuh` (see *Device-side Exclusion Helper* below).
The helper performs the linear scan over atom `i`'s partner range and
returns either the matching scale (from the per-potential scale array
the kernel passes in) or `1.0f` when `j` is not present.

## Device-side Exclusion Helper <!-- rq-b2f23140 -->

`kernels/exclusions.cuh` declares one inline `__device__` helper for
reading the exclusion-list buffers from a pair-potential kernel:

```c
__device__ static inline float exclusion_scale(
    unsigned int i,
    unsigned int j,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const float *atom_excl_scales);
```

The helper performs a linear scan over atom `i`'s partner range
`[atom_excl_offsets[i], atom_excl_offsets[i + 1])`. On the first
`atom_excl_partners[m] == j` match it returns `atom_excl_scales[m]`;
when no match is found (including when the range is empty) it returns
`1.0f`. Scan order is from lower index to higher with early exit on
the first match, so the typical-case cost is bounded by atom `i`'s
exclusion-partner count (≤ a few entries for typical bonded systems).

`exclusion_scale` is the canonical device-side reader of the
exclusion-list buffers described in *Exclusion list* above. Each
pair-potential kernel passes the per-potential scale array appropriate
to itself: the LJ kernel passes `atom_excl_lj_scales`, the Coulomb
kernel passes `atom_excl_coul_scales`. Pair-potential `.cu` files
`#include "exclusions.cuh"` and call `exclusion_scale(...)` at the
point they want to scale a pair's contribution by the effective
exclusion factor; nvcc inlines the body into each translation unit.
The header carries no PTX module of its own and `init_device()`
performs no `load_ptx` call for it.

When the exclusion list is empty, every atom's `atom_excl_offsets`
range is empty, the helper returns `1.0f`, and the unscaled
contribution flows through to the caller without a separate code path.

## Feature API <!-- rq-81659783 -->

### Types <!-- rq-22d0ce50 -->

- `Bond` — `Debug, Clone, Copy`. Fields: `atom_i: u32`, `atom_j: u32`, <!-- rq-0a8831b1 -->
  `bond_type_index: u32`.

- `Angle` — `Debug, Clone, Copy`. Fields: `atom_i: u32`, `atom_j: u32` <!-- rq-d278bb01 -->
  (centre), `atom_k: u32`, `angle_type_index: u32`. Wings satisfy
  `atom_i < atom_k`.

- `Exclusion` — `Debug, Clone, Copy`. Fields: `atom_i: u32`, <!-- rq-0c717392 -->
  `atom_j: u32`, `scale_lj: f32`, `scale_coul: f32`.

- `BondList` — host-side. Fields: `bonds: Vec<Bond>`, <!-- rq-ddf51309 -->
  `atom_bond_offsets: Vec<u32>`, `atom_bond_indices: Vec<u32>`,
  `particle_count: usize`.

  Method `BondList::is_empty(&self) -> bool` — `self.bonds.is_empty()`.

- `AngleList` — host-side. Fields: `angles: Vec<Angle>`, <!-- rq-07d003c4 -->
  `atom_angle_offsets: Vec<u32>`, `atom_angle_indices: Vec<u32>`,
  `particle_count: usize`.

  Method `AngleList::is_empty(&self) -> bool` —
  `self.angles.is_empty()`.

- `ExclusionList` — host-side. Fields: `entries: Vec<Exclusion>`, <!-- rq-f807cd11 -->
  `atom_excl_offsets: Vec<u32>`, `atom_excl_partners: Vec<u32>`,
  `atom_excl_lj_scales: Vec<f32>`, `atom_excl_coul_scales: Vec<f32>`,
  `particle_count: usize`.

- `ConstraintList` — host-side. The per-group SoA layout, including <!-- rq-fbd32983 -->
  fields, group ordering rules, and intra-group atom-order
  conventions, is defined in `integration/constraint-framework.md`.
  The parser populates an instance with the validated groups drawn
  from the `[constraints]` section.

  Method `ConstraintList::is_empty(&self) -> bool` —
  `self.groups.is_empty()`.

- `TopologyFileError` — error type returned by the parser. Variants: <!-- rq-bca0adbc -->
  - `Io(String)` — failed to read the file.
  - `UnknownSection { name: String, line_number: usize }` — a section
    header is not one of the accepted names (`bonds`, `exclusions`,
    `angles`).
  - `DuplicateSection { name: String, line_number: usize }` — the
    same section header appears twice.
  - `ContentOutsideSection { line_number: usize }` — non-blank,
    non-comment content appears before the first section header.
  - `InvalidBondRow { line_number: usize, reason: String }` — column
    count wrong, atom index unparseable, etc.
  - `InvalidAngleRow { line_number: usize, reason: String }`.
  - `InvalidExclusionRow { line_number: usize, reason: String }`.
  - `AtomIndexOutOfRange { line_number: usize, index: u32, max: u32 }`.
  - `SelfBond { line_number: usize, atom: u32 }`.
  - `RepeatedAtomInAngle { line_number: usize, atom: u32 }` — at
    least two of `atom_i`, `atom_j`, `atom_k` are equal.
  - `SelfExclusion { line_number: usize, atom: u32 }`.
  - `DuplicateBond { atom_i: u32, atom_j: u32 }`.
  - `DuplicateAngle { atom_i: u32, atom_j: u32, atom_k: u32 }`.
  - `DuplicateExclusion { atom_i: u32, atom_j: u32 }`.
  - `UnknownBondType { line_number: usize, name: String }`.
  - `UnknownAngleType { line_number: usize, name: String }`.
  - `ScaleOutOfRange { line_number: usize, scale: f32 }` — not in
    `[0.0, 1.0]` or non-finite.
  - `InvalidConstraintRow { line_number: usize, reason: String }` —
    column count wrong for the constraint type, atom index
    unparseable, or any other malformed-row condition.
  - `SelfConstraint { line_number: usize, atom: u32 }` — the same
    atom appears more than once in a single `[constraints]` row.
  - `UnknownConstraintType { line_number: usize, name: String }` —
    the row's `constraint_type_name` does not match any
    `[[constraint_types]]` entry.
  - `DuplicateConstraintAtom { atom: u32 }` — two `[constraints]`
    rows share at least one atom.
  - `BondIsAlsoConstraint { atom_i: u32, atom_j: u32 }` — a pair
    appears in both `[bonds]` and (after expansion) in
    `[constraints]`.

### Functions <!-- rq-e66012e0 -->

- `load_topology_file(path: &Path, particle_count: usize, bond_type_names: &[&str], angle_type_names: &[&str], constraint_types: &[NamedSlotConfig], constraint_registry: &ConstraintRegistry) -> Result<(BondList, AngleList, ExclusionList, ConstraintList), TopologyFileError>` <!-- rq-12b7dcb6 -->
  - Reads the file at `path` and parses all four sections.
  - Validates every constraint described in *File Format*.
  - Returns the canonicalised, sorted `BondList`, `AngleList`, the
    *effective* `ExclusionList` (explicit entries plus implicit
    bond-derived, angle-derived, and constraint-derived defaults),
    and the `ConstraintList` (groups sorted by minimum particle
    index per `integration/constraint-framework.md`).
  - The caller passes `particle_count` (used to bound atom indices),
    `bond_type_names` (used to resolve `bond_type_name` strings to
    indices in the config's `[[bond_types]]` array),
    `angle_type_names` (used to resolve `angle_type_name` strings to
    indices in the config's `[[angle_types]]` array),
    `constraint_types` (the full parsed `[[constraint_types]]`
    array of `NamedSlotConfig` entries, used to resolve
    `constraint_type_name` strings to indices), and
    `constraint_registry` (used to look up each constraint type's
    builder for the `expected_atom_count(&params)` query that
    size-checks `[constraints]` rows).

## Out of Scope <!-- rq-ad0edc95 -->

- Dihedrals, impropers, CMAP terms. The `.topology` file's schema
  reserves no syntax for them.
- Per-bond and per-angle parameter overrides (every bond's parameters
  come from its bond type; every angle's parameters come from its
  angle type).
- Connected-component merging of `[constraints]` rows that share
  atoms. v1 requires disjoint clusters; cross-row groups arrive with
  M-SHAKE. The on-disk format and the in-memory `ConstraintList`
  layout already accommodate the merged case.
- Forming or breaking bonds, angles, or constraints during a
  simulation. All four lists are fixed at start of run.
- A separate `[scaled_exclusions]` section or 1-4-specific syntax.
  Every exclusion is an `(i, j, scale_lj, scale_coul)` tuple; the
  user is responsible for emitting whichever 1-4 exclusions they
  need.
- Multi-`.topology`-file support; the config references at most one
  file.
- Binary topology formats.
- Automatic angle/dihedral derivation from the bond graph. Angle
  declarations are explicit.
- An exclusion "scale = NaN" sentinel to mean "remove implicit
  exclusion". An explicit entry with `scale = 1.0` already achieves
  that effect.

---

## Gherkin Scenarios <!-- rq-bf62f645 -->

```gherkin
Feature: Topology file with bonds, angles, and exclusions

  Background:
    Given a temporary directory tmp
    And a bond_type_names slice of ["CC", "CN", "OH"]
    And an angle_type_names slice of ["HOH"]
    And particle_count = 4

  # --- Happy paths ---

  @rq-8a16e6d6
  Scenario: Load a typical topology file
    Given tmp/sim.topology containing
      """
      [bonds]
      0 1 CC
      1 2 CC
      2 3 CN

      [exclusions]
      0 1 0.0
      0 2 0.5
      """
    When load_topology_file(tmp/sim.topology, 4, &["CC","CN","OH"], &["HOH"]) is called
    Then it returns Ok((bond_list, angle_list, exclusion_list))
    And bond_list.bonds equals [(0,1,0), (1,2,0), (2,3,1)]
    And angle_list.is_empty() is true
    And exclusion_list.entries equals [(0,1,0.0), (0,2,0.5), (1,2,0.0), (2,3,0.0)]
      (i.e. explicit (0,1) and (0,2) entries plus implicit (1,2,0.0) and (2,3,0.0)
       from bonds that lacked an explicit exclusion)

  @rq-fb608f06
  Scenario: Bonds are canonicalised to (min, max)
    Given tmp/sim.topology with a single bond "3 1 CC"
    When load_topology_file is called
    Then bond_list.bonds[0].atom_i equals 1
    And bond_list.bonds[0].atom_j equals 3

  @rq-d998b5c0
  Scenario: Bonds are sorted by (atom_i, atom_j)
    Given tmp/sim.topology with bonds "2 3 CC", "0 1 CC", "1 2 CC"
    When load_topology_file is called
    Then bond_list.bonds equals [(0,1,0), (1,2,0), (2,3,0)]

  @rq-9c1c58ef
  Scenario: Exclusion scales default to 0.0 when both columns omitted
    Given tmp/sim.topology with exclusion "0 1"
    When load_topology_file is called
    Then exclusion_list.entries contains (0, 1, scale_lj=0.0, scale_coul=0.0)

  @rq-1221d020
  Scenario: Single-scale form sets both LJ and Coulomb scales equally
    Given tmp/sim.topology with exclusion "0 1 0.5"
    When load_topology_file is called
    Then exclusion_list.entries contains (0, 1, scale_lj=0.5, scale_coul=0.5)

  @rq-1fde7f32
  Scenario: Four-column form sets LJ and Coulomb scales independently
    Given tmp/sim.topology with exclusion "0 1 0.5 0.833"
    When load_topology_file is called
    Then exclusion_list.entries contains (0, 1, scale_lj=0.5, scale_coul=0.833)

  @rq-dcba6fce
  Scenario: Implicit exclusion is added for a bond without explicit entry
    Given tmp/sim.topology with bond "0 1 CC" and no exclusions
    When load_topology_file is called
    Then exclusion_list.entries equals [(0, 1, scale_lj=0.0, scale_coul=0.0)]

  @rq-e9a421ef
  Scenario: Explicit exclusion overrides the bond's implicit default
    Given tmp/sim.topology with bond "0 1 CC" and exclusion "0 1 1.0"
    When load_topology_file is called
    Then exclusion_list.entries equals [(0, 1, scale_lj=1.0, scale_coul=1.0)]

  @rq-b0c18819
  Scenario: Empty topology file is valid
    Given tmp/sim.topology containing only blank lines and comments
    When load_topology_file is called
    Then it returns Ok((bond_list, angle_list, exclusion_list))
    And bond_list.is_empty() is true
    And angle_list.is_empty() is true
    And exclusion_list.entries is empty

  @rq-40f02b6a
  Scenario: Empty [bonds], [angles], and [exclusions] sections are valid
    Given tmp/sim.topology with all three section headers but no rows
    When load_topology_file is called
    Then it returns Ok with all three lists empty

  @rq-9da14aa1
  Scenario: Sections may appear in any order
    Given tmp/sim.topology with [exclusions] before [angles] before [bonds]
    When load_topology_file is called
    Then it returns Ok((bond_list, angle_list, exclusion_list))

  @rq-5b097ec0
  Scenario: Comments and blank lines are tolerated
    Given tmp/sim.topology with interleaved blank lines and # comments
    When load_topology_file is called
    Then it returns Ok((bond_list, angle_list, exclusion_list))

  # --- Per-atom indexing ---

  @rq-3eb8fe40
  Scenario: atom_bond_offsets reflects sorted bond list
    Given tmp/sim.topology with bonds "0 1 CC", "0 2 CC", "1 3 CC"
    When load_topology_file is called
    Then bond_list.atom_bond_offsets equals [0, 2, 3, 4, 5]
    And atom 0's bond indices reference slots 0 and 2 in the bond-pair buffer
    And atom 1's bond indices reference slots 1 and 4
    And atom 2's bond indices reference slot 3
    And atom 3's bond indices reference slot 5

  @rq-77f53d4b
  Scenario: atom_excl_offsets reflects sorted exclusion list
    Given an effective exclusion list of
      [(0, 1, scale_lj=0.0, scale_coul=0.0),
       (0, 2, scale_lj=0.5, scale_coul=0.5),
       (1, 3, scale_lj=0.5, scale_coul=0.833)]
    Then atom_excl_offsets equals [0, 2, 3, 4, 5]
    And atom 0's partners are [1, 2] with LJ scales [0.0, 0.5] and Coulomb scales [0.0, 0.5]
    And atom 1's partners are [0, 3] with LJ scales [0.0, 0.5] and Coulomb scales [0.0, 0.833]

  # --- IO and parser errors ---

  @rq-ef5aa4b7
  Scenario: File does not exist
    When load_topology_file("/tmp/missing.topology", 4, &["CC","CN","OH"], &["HOH"]) is called
    Then it returns Err(TopologyFileError::Io(_))

  @rq-4c245ce7
  Scenario: Unknown section header
    Given tmp/sim.topology with a section "[dihedrals]"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::UnknownSection { name: "dihedrals", line_number: _ })

  @rq-583d3df1
  Scenario: Duplicate section header
    Given tmp/sim.topology with two [bonds] headers
    When load_topology_file is called
    Then it returns Err(TopologyFileError::DuplicateSection { name: "bonds", line_number: _ })

  @rq-1ed32e10
  Scenario: Content before any section header
    Given tmp/sim.topology starting with a non-comment data line
    When load_topology_file is called
    Then it returns Err(TopologyFileError::ContentOutsideSection { line_number: _ })

  # --- Bond row validation ---

  @rq-9df1eedb
  Scenario: Bond row with wrong column count
    Given a bond row "0 1" (missing type)
    When load_topology_file is called
    Then it returns Err(TopologyFileError::InvalidBondRow { line_number: _, reason: _ })

  @rq-13b931f8
  Scenario: Bond row with non-integer index
    Given a bond row "abc 1 CC"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::InvalidBondRow { reason: _, .. })

  @rq-13e15b90
  Scenario: Bond row with atom index out of range
    Given a bond row "0 5 CC" and particle_count = 4
    When load_topology_file is called
    Then it returns Err(TopologyFileError::AtomIndexOutOfRange { index: 5, max: 3, .. })

  @rq-10d1da56
  Scenario: Self-bond rejected
    Given a bond row "2 2 CC"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::SelfBond { atom: 2, .. })

  @rq-4f78f4a2
  Scenario: Duplicate bond rejected (same canonical pair)
    Given bond rows "0 1 CC" and "1 0 CC"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::DuplicateBond { atom_i: 0, atom_j: 1 })

  @rq-e4563eec
  Scenario: Unknown bond type name rejected
    Given a bond row "0 1 XX" with bond_type_names ["CC","CN"]
    When load_topology_file is called
    Then it returns Err(TopologyFileError::UnknownBondType { name: "XX", .. })

  # --- Exclusion row validation ---

  @rq-f371677d
  Scenario: Exclusion row with too many columns
    Given an exclusion row "0 1 0.5 0.8 extra"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::InvalidExclusionRow { line_number: _, reason: _ })

  @rq-6cd92c14
  Scenario: Exclusion row with too few columns
    Given an exclusion row "0"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::InvalidExclusionRow { line_number: _, reason: _ })

  @rq-06c0e11a
  Scenario: Exclusion row with non-numeric scale
    Given an exclusion row "0 1 maybe"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::InvalidExclusionRow { reason: _, .. })

  @rq-df10e81f
  Scenario: Self-exclusion rejected
    Given an exclusion row "2 2 0.0"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::SelfExclusion { atom: 2, .. })

  @rq-17ed07e7
  Scenario: Exclusion atom out of range
    Given particle_count = 4 and exclusion row "0 9"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::AtomIndexOutOfRange { index: 9, max: 3, .. })

  @rq-eea2f5f8
  Scenario: Duplicate exclusion rejected
    Given exclusion rows "0 1 0.0" and "1 0 0.5"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::DuplicateExclusion { atom_i: 0, atom_j: 1 })

  @rq-f0b9b0f5
  Scenario: Exclusion scale less than 0 rejected
    Given an exclusion row "0 1 -0.1"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::ScaleOutOfRange { scale: -0.1, .. })

  @rq-9f658edf
  Scenario: Exclusion scale greater than 1 rejected
    Given an exclusion row "0 1 1.5"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::ScaleOutOfRange { scale: 1.5, .. })

  @rq-2b4a324a
  Scenario: Exclusion scale NaN rejected
    Given an exclusion row "0 1 nan"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::ScaleOutOfRange { scale: _, .. })

  @rq-6a9f0a18
  Scenario: Out-of-range Coulomb scale in four-column form rejected
    Given an exclusion row "0 1 0.5 1.5"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::ScaleOutOfRange { scale: 1.5, .. })

  # --- Angle row validation ---

  @rq-e768a2b1
  Scenario: Load a topology file containing an angle
    Given tmp/sim.topology containing
      """
      [bonds]
      0 1 OH
      0 2 OH

      [angles]
      1 0 2 HOH
      """
    When load_topology_file(tmp/sim.topology, 3, &["CC","CN","OH"], &["HOH"]) is called
    Then it returns Ok((bond_list, angle_list, exclusion_list))
    And angle_list.angles equals [(atom_i=1, atom_j=0, atom_k=2, angle_type_index=0)]

  @rq-f33ca120
  Scenario: Angle wings are canonicalised so atom_i < atom_k
    Given tmp/sim.topology with a single angle "2 0 1 HOH" and particle_count = 3
    When load_topology_file is called
    Then angle_list.angles[0].atom_i equals 1
    And angle_list.angles[0].atom_j equals 0
    And angle_list.angles[0].atom_k equals 2

  @rq-ba37ec6b
  Scenario: Angles are sorted by (atom_j, atom_i, atom_k)
    Given two angles "2 4 3 HOH" and "0 1 2 HOH" with particle_count = 5
    When load_topology_file is called
    Then angle_list.angles equals
      [(atom_i=0, atom_j=1, atom_k=2, type=0),
       (atom_i=2, atom_j=4, atom_k=3, type=0)
       — wait, second is canonicalised to (2,4,3) -> atom_i=2,atom_k=3]
    And the sort order is (atom_j ascending, then atom_i, then atom_k)

  @rq-021e6d82
  Scenario: Angle row with wrong column count
    Given an angle row "0 1 2" (missing type)
    When load_topology_file is called
    Then it returns Err(TopologyFileError::InvalidAngleRow { line_number: _, reason: _ })

  @rq-00bb491c
  Scenario: Angle row with non-integer index
    Given an angle row "abc 1 2 HOH"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::InvalidAngleRow { reason: _, .. })

  @rq-b05f8682
  Scenario: Angle row with atom index out of range
    Given an angle row "0 1 9 HOH" and particle_count = 4
    When load_topology_file is called
    Then it returns Err(TopologyFileError::AtomIndexOutOfRange { index: 9, max: 3, .. })

  @rq-cfc7a794
  Scenario: Repeated atom in angle rejected (i == j)
    Given an angle row "1 1 2 HOH"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::RepeatedAtomInAngle { atom: 1, .. })

  @rq-9d68f8fb
  Scenario: Repeated atom in angle rejected (j == k)
    Given an angle row "0 2 2 HOH"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::RepeatedAtomInAngle { atom: 2, .. })

  @rq-220f3f10
  Scenario: Repeated atom in angle rejected (i == k)
    Given an angle row "1 0 1 HOH"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::RepeatedAtomInAngle { atom: 1, .. })

  @rq-c7c3f66a
  Scenario: Duplicate angle rejected after canonicalisation
    Given angle rows "1 0 2 HOH" and "2 0 1 HOH" (same centre, swapped wings)
    When load_topology_file is called
    Then it returns Err(TopologyFileError::DuplicateAngle { atom_i: 1, atom_j: 0, atom_k: 2 })

  @rq-086a1bd9
  Scenario: Unknown angle type name rejected
    Given an angle row "1 0 2 XX" with angle_type_names ["HOH"]
    When load_topology_file is called
    Then it returns Err(TopologyFileError::UnknownAngleType { name: "XX", .. })

  # --- Angle-derived implicit exclusions ---

  @rq-514670c9
  Scenario: Implicit 1-3 exclusion is added for an angle without explicit (i, k) entry
    Given tmp/sim.topology with bond "0 1 OH", bond "0 2 OH", angle "1 0 2 HOH"
      and no [exclusions] section
    When load_topology_file is called
    Then exclusion_list.entries contains (0, 1, scale_lj=0.0, scale_coul=0.0)
    And exclusion_list.entries contains (0, 2, scale_lj=0.0, scale_coul=0.0)
    And exclusion_list.entries contains (1, 2, scale_lj=0.0, scale_coul=0.0)

  @rq-ea8ebebd
  Scenario: Explicit 1-3 exclusion overrides the angle's implicit default
    Given a bond "0 1 OH", a bond "0 2 OH", an angle "1 0 2 HOH",
      and an explicit exclusion "1 2 0.5 0.833"
    When load_topology_file is called
    Then exclusion_list.entries contains (1, 2, scale_lj=0.5, scale_coul=0.833)
    And exclusion_list.entries does not contain (1, 2, scale_lj=0.0, scale_coul=0.0)

  # --- Per-atom indexing for angles ---

  @rq-9a386c23
  Scenario: atom_angle_offsets reflects sorted angle list
    Given particle_count = 5 and angles "0 1 2 HOH", "0 2 3 HOH"
    When load_topology_file is called
    Then angle_list.atom_angle_offsets equals [0, 1, 2, 4, 5, 6]
    And atom 0's angle indices reference slot 0 (its slot in angle 0) and slot 3 (its slot in angle 1)
    And atom 2's angle indices reference slot 2 (its wing slot in angle 0)
      and slot 4 (its centre slot in angle 1)

  # --- Constraint rows ---

  @rq-fe3b32cf
  Scenario: Load a topology file with one rigid-water constraint
    Given tmp/sim.topology containing
      """
      [constraints]
      0 1 2 SPCE
      """
    And constraint_types contains an SPCE entry with kind = "shake"
    And constraint_registry is ConstraintRegistry::with_builtins()
    When load_topology_file(tmp/sim.topology, 3, &["CC","CN","OH"], &["HOH"], &constraint_types, &constraint_registry) is called
    Then it returns Ok((bond_list, angle_list, exclusion_list, constraint_list))
    And constraint_list.groups has length 1
    And the first group's atoms (in their declared order) are [0, 1, 2]
    And the first group's constraint_type_index resolves to constraint_types[0] (kind = "shake")
    And exclusion_list.entries contains (0, 1, 0.0, 0.0), (0, 2, 0.0, 0.0), and (1, 2, 0.0, 0.0)

  @rq-5dfc02a9
  Scenario: rigid-water constraint row with wrong atom count is rejected
    Given a [constraints] row "0 1 SPCE" (two atoms instead of three)
    When load_topology_file is called
    Then it returns Err(TopologyFileError::InvalidConstraintRow { reason: contains "3 atoms", .. })

  @rq-93506647
  Scenario: Constraint row with repeated atom is rejected
    Given a [constraints] row "0 1 1 SPCE"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::SelfConstraint { atom: 1, .. })

  @rq-44feffc6
  Scenario: Constraint row with atom index out of range is rejected
    Given particle_count = 3 and a [constraints] row "0 1 9 SPCE"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::AtomIndexOutOfRange { index: 9, max: 2, .. })

  @rq-6381db33
  Scenario: Unknown constraint type name is rejected
    Given a [constraints] row "0 1 2 UNKNOWN"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::UnknownConstraintType { name: "UNKNOWN", .. })

  @rq-15b6d3a4
  Scenario: Two constraint rows sharing an atom are rejected
    Given [constraints] rows "0 1 2 SPCE" and "2 3 4 SPCE" with particle_count = 5
    When load_topology_file is called
    Then it returns Err(TopologyFileError::DuplicateConstraintAtom { atom: 2 })

  @rq-8ea6cf9c
  Scenario: Pair appearing in both [bonds] and [constraints] is rejected
    Given a [bonds] row "0 1 OH" and a [constraints] row "0 1 2 SPCE"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::BondIsAlsoConstraint { atom_i: 0, atom_j: 1 })

  @rq-be8dfaa5
  Scenario: Constraint-derived implicit exclusion is overridden by an explicit entry
    Given a [constraints] row "0 1 2 SPCE" and an explicit exclusion "1 2 0.25 0.25"
    When load_topology_file is called
    Then exclusion_list.entries contains (1, 2, 0.25, 0.25)
    And exclusion_list.entries does not contain (1, 2, 0.0, 0.0)

  @rq-75a9815d
  Scenario: Constraint groups are sorted by minimum particle index
    Given [constraints] rows "100 101 102 SPCE", "4 5 6 SPCE", "50 51 52 SPCE"
    When load_topology_file is called
    Then constraint_list.groups[0]'s atoms start with 4
    And constraint_list.groups[1]'s atoms start with 50
    And constraint_list.groups[2]'s atoms start with 100
```
