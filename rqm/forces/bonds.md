# Feature: Bond List, Exclusion List, and `.bonds` File <!-- rq-9e1eee68 -->

A simulation's bonded topology is described by a `.bonds` file referenced from
the TOML config (`io/config-schema.md`) and consumed by the `MorseBonded`
slot (`morse-bonded.md`) and the Lennard-Jones slot's exclusion logic
(`lj-pair-force.md`). The file lists bond instances and per-pair non-bonded
exclusions; bond *types* (parameters) live in the config alongside particle
types.

The file produces two host-side structures: a `BondList` (bonds, with
precomputed per-atom indexing tables for deterministic reduction), and an
`ExclusionList` (per-pair scaling factors for non-bonded interactions
between bonded or otherwise excluded atoms).

## File Format <!-- rq-a33c1f4f -->

The `.bonds` file is UTF-8 text organised into two named sections,
`[bonds]` and `[exclusions]`. Either section may appear at most once;
either may be empty; either may be absent (an absent `[bonds]` means
no bonds; an absent `[exclusions]` means no explicit exclusions).
Section headers are case-sensitive and must appear on their own line.
Sections may appear in either order.

```
# Comments start with '#' and run to end of line. Blank lines are <!-- rq-38285db7 -->
# ignored. <!-- rq-2e98a75d -->

[bonds]
# Column format: atom_i  atom_j  bond_type_name <!-- rq-cf456bb4 -->
# Atom indices are 0-based and refer to entries in the init file. <!-- rq-955b55f0 -->
0 1 CC
1 2 CC
2 3 CN

[exclusions]
# Column format: atom_i  atom_j  [scale] <!-- rq-c8ea0a96 -->
# Scale defaults to 0.0 (full exclusion) when omitted. <!-- rq-922c9a86 -->
0 1 0.0
1 2 0.0
0 2 0.5
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

### Exclusion entries <!-- rq-4ae7794c -->

Each non-comment line in `[exclusions]` has the form
`atom_i atom_j [scale]`. The scale column is optional; when omitted
the scale defaults to `0.0` (full exclusion).

- `atom_i: u32` and `atom_j: u32` are zero-based particle indices. Both
  must be `< particle_count`. `atom_i == atom_j` is rejected.
- `scale: f32` when present is a finite number in `[0.0, 1.0]`.
  Out-of-range values, NaN, and infinity are rejected.

Like bonds, exclusions are canonicalised to `(min, max)` and sorted by
`(min, max)`. Duplicate pairs (after canonicalisation) are rejected.

### Effective exclusions <!-- rq-24f280af -->

After parsing, the consumer-facing `ExclusionList` is the *effective
exclusion list*, formed by combining the explicit `[exclusions]`
entries with implicit exclusions derived from `[bonds]`:

- Every explicit `(i, j, s)` entry from the file becomes an effective
  exclusion `(i, j, s)`.
- For every bond `(i, j)` in `[bonds]` that does **not** have a
  matching explicit `(i, j, _)` entry, an implicit exclusion
  `(i, j, 0.0)` is added.

The result is the set of `(i, j, scale)` tuples consulted by the LJ
kernel for scaling pairwise non-bonded interactions. Explicit entries
take precedence over the bond default; an explicit entry with
`scale = 1.0` therefore *keeps* the LJ contribution for a bonded pair,
which is unusual physics but is the user's deliberate override.

Effective exclusions for pairs that are neither bonded nor explicitly
listed are absent from the list; the LJ kernel treats them as
`scale = 1.0` (no scaling).

The framework treats this implicit-exclusion rule as the only
default behaviour that affects simulation results without an explicit
user declaration. It is documented here and in `lj-pair-force.md`.

### Empty file <!-- rq-1c794f95 -->

A `.bonds` file containing zero bonds and zero exclusions is valid
(both sections empty or absent). The runner produces an empty
`BondList` and an empty `ExclusionList`; the `MorseBonded` slot is not
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

### Exclusion list <!-- rq-1e1b4e02 -->

For `E` effective exclusions, the host-side `ExclusionList` carries:

- `entries: Vec<Exclusion>` — length `E`, sorted by `(atom_i, atom_j)`
  with `atom_i < atom_j`. Each `Exclusion` records `atom_i: u32`,
  `atom_j: u32`, and `scale: f32`.
- `atom_excl_offsets: Vec<u32>` — length `N + 1`. For atom `a`, the
  slice
  `atom_excl_partners[atom_excl_offsets[a] .. atom_excl_offsets[a+1]]`
  lists the *partners* (`u32`) of `a` in the effective exclusion list,
  paired with their scale factors.
- `atom_excl_partners: Vec<u32>` — length `2 * E`.
- `atom_excl_scales: Vec<f32>` — length `2 * E`, parallel to
  `atom_excl_partners`.

Pair-potential CUDA kernels consult this per-atom lookup table through
the shared device helper `exclusion_scale` declared in
`kernels/exclusions.cuh` (see *Device-side Exclusion Helper* below).
The helper performs the linear scan over atom `i`'s partner range and
returns either the matching scale or `1.0f` when `j` is not present.

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
exclusion-list buffers described in *Exclusion list* above.
Pair-potential `.cu` files `#include "exclusions.cuh"` and call
`exclusion_scale(...)` at the point they want to scale a pair's
contribution by the effective exclusion factor; nvcc inlines the body
into each translation unit. The header carries no PTX module of its
own and `init_device()` performs no `load_ptx` call for it.

When the exclusion list is empty, every atom's `atom_excl_offsets`
range is empty, the helper returns `1.0f`, and the unscaled
contribution flows through to the caller without a separate code path.

## Feature API <!-- rq-81659783 -->

### Types <!-- rq-22d0ce50 -->

- `Bond` — `Debug, Clone, Copy`. Fields: `atom_i: u32`, `atom_j: u32`, <!-- rq-0a8831b1 -->
  `bond_type_index: u32`.

- `Exclusion` — `Debug, Clone, Copy`. Fields: `atom_i: u32`, <!-- rq-0c717392 -->
  `atom_j: u32`, `scale: f32`.

- `BondList` — host-side. Fields: `bonds: Vec<Bond>`, <!-- rq-ddf51309 -->
  `atom_bond_offsets: Vec<u32>`, `atom_bond_indices: Vec<u32>`,
  `particle_count: usize`.

  Method `BondList::is_empty(&self) -> bool` — `self.bonds.is_empty()`.

- `ExclusionList` — host-side. Fields: `entries: Vec<Exclusion>`, <!-- rq-f807cd11 -->
  `atom_excl_offsets: Vec<u32>`, `atom_excl_partners: Vec<u32>`,
  `atom_excl_scales: Vec<f32>`, `particle_count: usize`.

- `BondsFileError` — error type returned by the parser. Variants: <!-- rq-bca0adbc -->
  - `Io(String)` — failed to read the file.
  - `UnknownSection { name: String, line_number: usize }` — a section
    header is not one of the accepted names.
  - `DuplicateSection { name: String, line_number: usize }` — the
    same section header appears twice.
  - `ContentOutsideSection { line_number: usize }` — non-blank,
    non-comment content appears before the first section header.
  - `InvalidBondRow { line_number: usize, reason: String }` — column
    count wrong, atom index unparseable, etc.
  - `InvalidExclusionRow { line_number: usize, reason: String }`.
  - `AtomIndexOutOfRange { line_number: usize, index: u32, max: u32 }`.
  - `SelfBond { line_number: usize, atom: u32 }`.
  - `SelfExclusion { line_number: usize, atom: u32 }`.
  - `DuplicateBond { atom_i: u32, atom_j: u32 }`.
  - `DuplicateExclusion { atom_i: u32, atom_j: u32 }`.
  - `UnknownBondType { line_number: usize, name: String }`.
  - `ScaleOutOfRange { line_number: usize, scale: f32 }` — not in
    `[0.0, 1.0]` or non-finite.

### Functions <!-- rq-e66012e0 -->

- `load_bonds_file(path: &Path, particle_count: usize, bond_type_names: &[&str]) -> Result<(BondList, ExclusionList), BondsFileError>` <!-- rq-12b7dcb6 -->
  - Reads the file at `path` and parses both sections.
  - Validates every constraint described in *File Format*.
  - Returns the canonicalised, sorted `BondList` and the *effective*
    `ExclusionList` (explicit entries plus implicit bond defaults).
  - The caller passes `particle_count` (used to bound atom indices)
    and `bond_type_names` (used to resolve `bond_type_name` strings to
    indices in the config's `[[bond_types]]` array).

## Out of Scope <!-- rq-ad0edc95 -->

- Angles, dihedrals, impropers, CMAP terms. The `.bonds` file does not
  carry angle or dihedral lists in v1; the schema does not reserve
  syntax for them.
- Per-bond parameter overrides (every bond's parameters come from its
  bond type).
- Constraint algorithms (rigid bonds via SHAKE/RATTLE).
- Forming or breaking bonds during a simulation. The bond list is
  fixed at start of run.
- Per-pair Coulomb scaling (no Coulomb force in v1).
- A separate `[scaled_exclusions]` section or 1-4-specific syntax.
  Every exclusion is a `(i, j, scale)` triple; the user is responsible
  for emitting whichever 1-2 / 1-3 / 1-4 exclusions they need.
- Multi-`.bonds`-file support; the config references at most one file.
- Binary bond formats.
- Automatic angle/dihedral derivation from the bond graph.
- An exclusion "scale = NaN" sentinel to mean "remove implicit
  exclusion". An explicit entry with `scale = 1.0` already achieves
  that effect.

---

## Gherkin Scenarios <!-- rq-bf62f645 -->

```gherkin
Feature: Bond list, exclusion list, and .bonds file

  Background:
    Given a temporary directory tmp
    And a bond_type_names slice of ["CC", "CN"]
    And particle_count = 4

  # --- Happy paths ---

  @rq-8a16e6d6
  Scenario: Load a typical .bonds file
    Given tmp/topology.bonds containing
      """
      [bonds]
      0 1 CC
      1 2 CC
      2 3 CN

      [exclusions]
      0 1 0.0
      0 2 0.5
      """
    When load_bonds_file(tmp/topology.bonds, 4, &["CC","CN"]) is called
    Then it returns Ok((bond_list, exclusion_list))
    And bond_list.bonds equals [(0,1,0), (1,2,0), (2,3,1)]
    And exclusion_list.entries equals [(0,1,0.0), (0,2,0.5), (1,2,0.0), (2,3,0.0)]
      (i.e. explicit (0,1) and (0,2) entries plus implicit (1,2,0.0) and (2,3,0.0)
       from bonds that lacked an explicit exclusion)

  @rq-fb608f06
  Scenario: Bonds are canonicalised to (min, max)
    Given tmp/topology.bonds with a single bond "3 1 CC"
    When load_bonds_file is called
    Then bond_list.bonds[0].atom_i equals 1
    And bond_list.bonds[0].atom_j equals 3

  @rq-d998b5c0
  Scenario: Bonds are sorted by (atom_i, atom_j)
    Given tmp/topology.bonds with bonds "2 3 CC", "0 1 CC", "1 2 CC"
    When load_bonds_file is called
    Then bond_list.bonds equals [(0,1,0), (1,2,0), (2,3,0)]

  @rq-9c1c58ef
  Scenario: Exclusion scale defaults to 0.0 when column omitted
    Given tmp/topology.bonds with exclusion "0 1"
    When load_bonds_file is called
    Then exclusion_list.entries contains (0, 1, 0.0)

  @rq-dcba6fce
  Scenario: Implicit exclusion is added for a bond without explicit entry
    Given tmp/topology.bonds with bond "0 1 CC" and no exclusions
    When load_bonds_file is called
    Then exclusion_list.entries equals [(0, 1, 0.0)]

  @rq-e9a421ef
  Scenario: Explicit exclusion overrides the bond's implicit default
    Given tmp/topology.bonds with bond "0 1 CC" and exclusion "0 1 1.0"
    When load_bonds_file is called
    Then exclusion_list.entries equals [(0, 1, 1.0)]

  @rq-b0c18819
  Scenario: Empty .bonds file is valid
    Given tmp/topology.bonds containing only blank lines and comments
    When load_bonds_file is called
    Then it returns Ok((bond_list, exclusion_list))
    And bond_list.is_empty() is true
    And exclusion_list.entries is empty

  @rq-40f02b6a
  Scenario: Empty [bonds] and [exclusions] sections are valid
    Given tmp/topology.bonds with both section headers but no rows
    When load_bonds_file is called
    Then it returns Ok with empty lists

  @rq-9da14aa1
  Scenario: Sections may appear in either order
    Given tmp/topology.bonds with [exclusions] before [bonds]
    When load_bonds_file is called
    Then it returns Ok((bond_list, exclusion_list))

  @rq-5b097ec0
  Scenario: Comments and blank lines are tolerated
    Given tmp/topology.bonds with interleaved blank lines and # comments
    When load_bonds_file is called
    Then it returns Ok((bond_list, exclusion_list))

  # --- Per-atom indexing ---

  @rq-3eb8fe40
  Scenario: atom_bond_offsets reflects sorted bond list
    Given tmp/topology.bonds with bonds "0 1 CC", "0 2 CC", "1 3 CC"
    When load_bonds_file is called
    Then bond_list.atom_bond_offsets equals [0, 2, 3, 4, 5]
    And atom 0's bond indices reference slots 0 and 2 in the bond-pair buffer
    And atom 1's bond indices reference slots 1 and 4
    And atom 2's bond indices reference slot 3
    And atom 3's bond indices reference slot 5

  @rq-77f53d4b
  Scenario: atom_excl_offsets reflects sorted exclusion list
    Given an effective exclusion list of [(0,1,0.0), (0,2,0.5), (1,3,0.5)]
    Then atom_excl_offsets equals [0, 2, 3, 4, 5]
    And atom 0's partners are [1, 2] with scales [0.0, 0.5]

  # --- IO and parser errors ---

  @rq-ef5aa4b7
  Scenario: File does not exist
    When load_bonds_file("/tmp/missing.bonds", 4, &["CC","CN"]) is called
    Then it returns Err(BondsFileError::Io(_))

  @rq-4c245ce7
  Scenario: Unknown section header
    Given tmp/topology.bonds with a section "[angles]"
    When load_bonds_file is called
    Then it returns Err(BondsFileError::UnknownSection { name: "angles", line_number: _ })

  @rq-583d3df1
  Scenario: Duplicate section header
    Given tmp/topology.bonds with two [bonds] headers
    When load_bonds_file is called
    Then it returns Err(BondsFileError::DuplicateSection { name: "bonds", line_number: _ })

  @rq-1ed32e10
  Scenario: Content before any section header
    Given tmp/topology.bonds starting with a non-comment data line
    When load_bonds_file is called
    Then it returns Err(BondsFileError::ContentOutsideSection { line_number: _ })

  # --- Bond row validation ---

  @rq-9df1eedb
  Scenario: Bond row with wrong column count
    Given a bond row "0 1" (missing type)
    When load_bonds_file is called
    Then it returns Err(BondsFileError::InvalidBondRow { line_number: _, reason: _ })

  @rq-13b931f8
  Scenario: Bond row with non-integer index
    Given a bond row "abc 1 CC"
    When load_bonds_file is called
    Then it returns Err(BondsFileError::InvalidBondRow { reason: _, .. })

  @rq-13e15b90
  Scenario: Bond row with atom index out of range
    Given a bond row "0 5 CC" and particle_count = 4
    When load_bonds_file is called
    Then it returns Err(BondsFileError::AtomIndexOutOfRange { index: 5, max: 3, .. })

  @rq-10d1da56
  Scenario: Self-bond rejected
    Given a bond row "2 2 CC"
    When load_bonds_file is called
    Then it returns Err(BondsFileError::SelfBond { atom: 2, .. })

  @rq-4f78f4a2
  Scenario: Duplicate bond rejected (same canonical pair)
    Given bond rows "0 1 CC" and "1 0 CC"
    When load_bonds_file is called
    Then it returns Err(BondsFileError::DuplicateBond { atom_i: 0, atom_j: 1 })

  @rq-e4563eec
  Scenario: Unknown bond type name rejected
    Given a bond row "0 1 XX" with bond_type_names ["CC","CN"]
    When load_bonds_file is called
    Then it returns Err(BondsFileError::UnknownBondType { name: "XX", .. })

  # --- Exclusion row validation ---

  @rq-f371677d
  Scenario: Exclusion row with wrong column count
    Given an exclusion row "0 1 0.5 extra"
    When load_bonds_file is called
    Then it returns Err(BondsFileError::InvalidExclusionRow { line_number: _, reason: _ })

  @rq-06c0e11a
  Scenario: Exclusion row with non-numeric scale
    Given an exclusion row "0 1 maybe"
    When load_bonds_file is called
    Then it returns Err(BondsFileError::InvalidExclusionRow { reason: _, .. })

  @rq-df10e81f
  Scenario: Self-exclusion rejected
    Given an exclusion row "2 2 0.0"
    When load_bonds_file is called
    Then it returns Err(BondsFileError::SelfExclusion { atom: 2, .. })

  @rq-17ed07e7
  Scenario: Exclusion atom out of range
    Given particle_count = 4 and exclusion row "0 9"
    When load_bonds_file is called
    Then it returns Err(BondsFileError::AtomIndexOutOfRange { index: 9, max: 3, .. })

  @rq-eea2f5f8
  Scenario: Duplicate exclusion rejected
    Given exclusion rows "0 1 0.0" and "1 0 0.5"
    When load_bonds_file is called
    Then it returns Err(BondsFileError::DuplicateExclusion { atom_i: 0, atom_j: 1 })

  @rq-f0b9b0f5
  Scenario: Exclusion scale less than 0 rejected
    Given an exclusion row "0 1 -0.1"
    When load_bonds_file is called
    Then it returns Err(BondsFileError::ScaleOutOfRange { scale: -0.1, .. })

  @rq-9f658edf
  Scenario: Exclusion scale greater than 1 rejected
    Given an exclusion row "0 1 1.5"
    When load_bonds_file is called
    Then it returns Err(BondsFileError::ScaleOutOfRange { scale: 1.5, .. })

  @rq-2b4a324a
  Scenario: Exclusion scale NaN rejected
    Given an exclusion row "0 1 nan"
    When load_bonds_file is called
    Then it returns Err(BondsFileError::ScaleOutOfRange { scale: _, .. })
```
