# Feature: Topology File, Bond List, Angle List, Dihedral List, Exclusion List, and Constraint List <!-- rq-9e1eee68 -->

A simulation's bonded topology is described by a `.topology` file referenced
from the TOML config (`io/config-schema.md`) and consumed by the
`MorseBonded` slot (`morse-bonded.md`), the `HarmonicAngle` slot
(`harmonic-angle.md`), every dihedral slot (`periodic-dihedral.md` and any
future dihedral functional forms), the `Constraint` slot
(`integration/constraint-framework.md`, `integration/shake.md`), and the
Lennard-Jones and SPME real-space slots' exclusion logic
(`lj-pair-force.md`, `spme.md`). The file lists bond instances, angle instances,
dihedral instances, per-pair non-bonded exclusions, rigid constraint
groups, and optional per-atom charges; bond, angle, dihedral, and
constraint *types* (parameters) live in the config alongside particle
types.

The file produces five host-side structures: a `BondList` (bonds, with
precomputed per-atom indexing tables for deterministic reduction), an
`AngleList` (angles, with the same indexing pattern), a `DihedralList`
(dihedrals, with the same indexing pattern extended to four atoms per
entry), an `ExclusionList` (per-pair scaling factors for non-bonded
interactions between bonded, angle-coupled, dihedral-coupled,
constraint-coupled, or otherwise excluded atoms), and a `ConstraintList`
(rigid constraint groups; see `integration/constraint-framework.md` for the
SoA layout and the kernel contract).

When the file contains a `[charges]` section it additionally produces a
`ChargeList` — a complete per-atom charge assignment (see *Charge
entries* and *Charge list*). A present `ChargeList` is the source of
every particle's charge, and the per-type `charge` field of
`[[particle_types]]` (`io/config-schema.md`) is then required to be
absent or zero; when the section is absent, the per-type `charge` field
is the fallback source. The runner assembles the per-particle charge
array and validates this precedence at setup (see
`simulation-runner.md`).

The bond and constraint connectivity additionally induces a
`MoleculeList`, a partition of the particles into connected molecular
groups derived from the combined bond + constraint graph (see *Molecule
grouping*). The Monte-Carlo barostat
(`integration/mc-barostat.md`) consumes this partition to scale molecular
centres of mass.

## File Format <!-- rq-a33c1f4f -->

The `.topology` file is UTF-8 text organised into six named sections,
`[bonds]`, `[exclusions]`, `[angles]`, `[dihedrals]`, `[constraints]`, and
`[charges]`. Each section may appear at most once; each may be empty; each
may be absent (an absent `[bonds]` means no bonds, an absent `[exclusions]`
means no explicit exclusions, an absent `[angles]` means no angles, an
absent `[dihedrals]` means no dihedrals, an absent `[constraints]` means no
rigid constraint groups, an absent `[charges]` means per-atom charges are
not supplied and the per-type `charge` fallback is used). Section headers
are case-sensitive and must appear on their own line. Sections may appear
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

[dihedrals]
# Column format: atom_i  atom_j  atom_k  atom_l  dihedral_type_name <!-- rq-68c5d30a -->
# The torsion angle is the angle between the (i, j, k) plane and the <!-- rq-d1453cff -->
# (j, k, l) plane about the j-k axis (IUPAC convention). <!-- rq-edebcf92 -->
# A given (atom_i, atom_j, atom_k, atom_l) quadruple may appear in <!-- rq-efba2cce -->
# multiple rows naming different dihedral types; each row contributes <!-- rq-d91c6352 -->
# one Fourier term to the total dihedral potential on that quadruple. <!-- rq-c8c03ed1 -->
0 1 2 3 CT-CT-CT-CT_n1
0 1 2 3 CT-CT-CT-CT_n3

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

[charges]
# Column format: atom_index  charge <!-- rq-bcc71adf -->
# One row per particle; charge is in the config's unit system <!-- rq-3376c1b0 -->
# (coulombs in SI mode, elementary charges in atomic mode). <!-- rq-2926ea10 -->
# When this section is present it must cover every atom exactly once. <!-- rq-b74e8ce4 -->
0 -0.8340
1  0.4170
2  0.4170
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

### Dihedral entries <!-- rq-056c4760 -->

Each non-comment line in `[dihedrals]` has the form `atom_i atom_j
atom_k atom_l dihedral_type_name` where:

- `atom_i: u32`, `atom_j: u32`, `atom_k: u32`, and `atom_l: u32` are
  zero-based particle indices. All four must be `< particle_count`.
  The torsion is the angle between the `(atom_i, atom_j, atom_k)`
  plane and the `(atom_j, atom_k, atom_l)` plane, measured about the
  `atom_j`–`atom_k` axis (IUPAC convention).
- The four atoms must all be distinct: any of `atom_i == atom_j`,
  `atom_i == atom_k`, `atom_i == atom_l`, `atom_j == atom_k`,
  `atom_j == atom_l`, or `atom_k == atom_l` is rejected as
  `RepeatedAtomInDihedral`.
- `dihedral_type_name: String` must match the `name` field of an
  entry in the config's `[[dihedral_types]]` array. Unknown names are
  rejected.

Dihedral *order* in the section is preserved for diagnostics.
Internally the parser canonicalises each `(atom_i, atom_j, atom_k,
atom_l)` quadruple by reversing the sequence (so `atom_i` and
`atom_l` swap and `atom_j` and `atom_k` swap) when `atom_i > atom_l`,
which leaves the dihedral angle unchanged and yields a canonical form
with `atom_i ≤ atom_l`. The canonicalised quadruples are then sorted
by `(atom_i, atom_j, atom_k, atom_l)` before assigning dihedral
indices.

A given canonical `(atom_i, atom_j, atom_k, atom_l)` quadruple may
appear in multiple `[dihedrals]` rows naming different dihedral types
— this is how multiple Fourier terms (e.g. `n = 1`, `n = 2`, `n = 3`)
on the same torsion are expressed. Two rows that share both the same
canonical quadruple *and* the same `dihedral_type_name` are rejected
as `DuplicateDihedral` regardless of file order.

The canonical quadruple `atom_i ≤ atom_l` constraint allows
`atom_i == atom_l` only when the four atoms are non-distinct, which
the `RepeatedAtomInDihedral` rule has already ruled out; in practice
the canonical form always satisfies `atom_i < atom_l`.

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
  constraint type's declared `atoms` field; for `kind = "settle"`,
  `k` is always 3. Wrong atom count is rejected as
  `InvalidConstraintRow` with a reason that names the expected count.
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
parsed `ConstraintList`. The group's pairwise constraints are derived
from the named constraint type's per-kind specification: for
`kind = "shake"` from the type's `constraints` table (which references
atoms by their position in this row order), and for `kind = "settle"`
from the type's `d_OH`/`d_HH`, synthesised as the canonical water
pattern `{(0, 1, d_OH), (0, 2, d_OH), (1, 2, d_HH)}`. Algorithms that
depend on which atom plays which role (rigid water, where atom 0 is the
oxygen by convention) encode the convention in the row's atom order.

A pair of atoms `(min, max)` that appears as a constraint pair (every
1-2 pair generated by the group's expansion into pairwise constraints
— see `integration/constraint-framework.md`) is forbidden from also
appearing in `[bonds]`. Conflicting rows are rejected as
`BondIsAlsoConstraint`.

### Charge entries <!-- rq-107a61fc -->

Each non-comment line in `[charges]` has the form `atom_index charge`
where:

- `atom_index: u32` is a zero-based particle index and must be
  `< particle_count`. An out-of-range index is rejected as
  `AtomIndexOutOfRange`.
- `charge: f64` is the particle's electric charge in the config's unit
  system (coulombs in `si` mode, elementary charges in `atomic` mode;
  see `io/unit-system.md`). It must be finite; any sign, and exactly
  zero, are accepted. A non-finite or unparseable value is rejected as
  `InvalidChargeRow`. A row whose column count is not exactly two is
  also rejected as `InvalidChargeRow`.

The `[charges]` section is **all-or-nothing**: when it is present it must
assign a charge to every particle exactly once. An atom index that
appears in more than one row is rejected as `DuplicateChargeAtom`. A
section that resolves to fewer than `particle_count` distinct atoms is
rejected as `IncompleteCharges`; a section that supplies exactly
`particle_count` distinct, in-range indices necessarily covers every
atom. Row order in the section is not significant — each charge is stored
at its declared `atom_index`.

The charge values are converted from the config's unit system to the
engine's internal atomic units (elementary charges) at load, so the
resulting `ChargeList` holds atomic-unit charges, consistent with every
other unit-bearing quantity past the I/O boundary.

A present `[charges]` section is the sole source of per-particle charge:
the per-type `charge` field of `[[particle_types]]` is then required to
be absent or zero, and a config that declares both a `[charges]` section
and a nonzero per-type `charge` is rejected at runner setup (see
`simulation-runner.md`). When `[charges]` is absent, each particle's
charge comes from its type's `charge` field, exactly as for a topology
file that omits the section. Either way the per-particle charge is stored
only in `posq.w` on the device; the `[charges]` section changes the
*source* of that value, not the device layout or any pair-force kernel.

### Effective exclusions <!-- rq-24f280af -->

After parsing, the consumer-facing `ExclusionList` is the *effective
exclusion list*, formed by combining the explicit `[exclusions]`
entries with implicit exclusions derived from `[bonds]`, `[angles]`,
`[dihedrals]`, and `[constraints]`. The four implicit sources are
layered with strict precedence, from highest to lowest:

1. **Explicit `[exclusions]` rows.** Every explicit
   `(i, j, scale_lj, scale_coul)` entry from the file becomes an
   effective exclusion with those two scales; no implicit source can
   override an explicit row.
2. **Implicit 1-2 from bonds.** For every bond `(i, j)` in `[bonds]`
   that does **not** have a matching explicit `(i, j, _, _)` entry,
   an implicit exclusion `(i, j, 0.0, 0.0)` is added.
3. **Implicit 1-3 from angles.** For every angle `(i, j, k)` in
   `[angles]`, the 1-3 pair `(i, k)` is considered. When `(i, k)`
   does **not** have a matching explicit `(i, k, _, _)` entry *and*
   is not already covered by an implicit bond-derived entry, an
   implicit exclusion `(i, k, 0.0, 0.0)` is added.
4. **Implicit 1-3 from constraints.** For every constraint group in
   `[constraints]`, every pair `(p, q)` of distinct atoms drawn from
   the group's atom set is considered. When `(p, q)` does **not**
   have a matching explicit `(p, q, _, _)` entry *and* is not already
   covered by an implicit bond-derived or angle-derived entry, an
   implicit exclusion `(p, q, 0.0, 0.0)` is added. For a SHAKE-
   constrained rigid-water group `(O, H1, H2)` this produces three
   implicit exclusions: `(O, H1)`, `(O, H2)`, and `(H1, H2)`.
5. **Implicit scaled 1-4 from dihedrals.** Each dihedral
   `(i, j, k, l)` in `[dihedrals]` considers its 1-4 pair `(i, l)`.
   When `(i, l)` is **not** already covered by an explicit entry, a
   bond-derived implicit entry, an angle-derived implicit entry, or
   a constraint-derived implicit entry, a scaled implicit exclusion
   `(i, l, scale_lj, scale_coul)` is added, where the scales are
   drawn from the dihedral's `dihedral_type` (`scale_lj_14` and
   `scale_coul_14`; see `io/config-schema.md`). When several
   dihedrals share the same canonical `(i, l)` pair, only the first
   one (in the canonical dihedral order — see *Dihedral entries*)
   introduces the implicit 1-4; subsequent dihedrals on the same
   `(i, l)` still contribute their own torque through the
   `DihedralList` but do not add a second `(i, l)` entry to the
   exclusion list. This *first-wins* policy mirrors AMBER's
   *ignore_end* convention.

Precedence is applied in the order layered above (1 → 2 → 3 → 4 → 5):
at each layer, the rule's candidate pair is added only when no
preceding layer has already produced an entry for the same canonical
`(i, j)` pair. The framework treats these implicit-exclusion rules
(1-2 from bonds, 1-3 from angles, 1-3 from constraints, scaled 1-4
from dihedrals) as the only default behaviour that affects simulation
results without an explicit user declaration. They are documented
here, in `lj-pair-force.md`, and in `spme.md`.

The result is the set of `(i, j, scale_lj, scale_coul)` tuples
consulted by the LJ and SPME real-space pair-force kernels. The LJ
kernel reads `scale_lj`; the SPME real-space kernel reads `scale_coul`.
Explicit entries take precedence over every implicit source; an
explicit entry with `scale = 1.0` therefore *keeps* the corresponding
non-bonded contribution for a bonded, angle-coupled, dihedral-coupled,
or constraint-coupled pair, which is unusual physics but is the user's
deliberate override.

Effective exclusions for pairs that are neither bonded, angle-coupled,
dihedral-coupled, constraint-coupled, nor explicitly listed are absent
from the list; the LJ and SPME real-space kernels treat them as
`scale = 1.0` (no scaling).

### Empty file <!-- rq-1c794f95 -->

A `.topology` file containing zero bonds, zero exclusions, zero
angles, zero dihedrals, zero constraints, and no `[charges]` section is
valid (all sections empty or absent). The runner produces an empty
`BondList`, an empty `AngleList`, an empty `DihedralList`, an empty
`ExclusionList`, an empty `ConstraintList`, and `None` for the charge
list; the `MorseBonded`, `HarmonicAngle`, `PeriodicDihedral` (and any
other dihedral slot), and `Constraint` slots are not constructed, and
every particle's charge comes from its type's `charge` field. An empty
`[charges]` section (the header with no rows) is valid only when
`particle_count == 0`; for a nonzero particle count an empty `[charges]`
section is rejected as `IncompleteCharges`.

## Molecule grouping <!-- rq-c200c7b8 -->

A *molecule* is a connected component of the undirected graph whose nodes
are the `N` particles and whose edges are the union of:

- every bond `(atom_i, atom_j)` in the `BondList`, and
- every intra-group pair of a `ConstraintList` group (a `k`-atom
  constraint group contributes edges among all its atoms, so the group's
  atoms land in one component regardless of which pairwise constraints the
  algorithm expands to).

Angles and exclusions do **not** induce molecule edges: a 1-3 angle pair is
already covered by the two bonds that form the angle, and exclusions carry
no bonding meaning. Every particle in no bond and no constraint is its own
singleton molecule.

The partition is derived from the already-parsed `BondList` and
`ConstraintList`; it adds no `.topology` file syntax. For a rigid SPC water
system each three-atom SETTLE (or SHAKE) constraint group is one molecule;
for a monatomic fluid every atom is a singleton molecule. The molecule
decomposition is fixed for the lifetime of a run.

Molecules are ordered by their minimum particle index, and the atoms
within each molecule are listed in ascending particle-index order. Both
orderings are deterministic functions of the parsed topology, so the
device-resident molecule tables are byte-identical across runs.

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

The `BondList` is potential-agnostic: it holds every bond regardless of
its type's `potential`, and `bond_type_index` is a global index into the
config's `[[bond_types]]` array. When more than one bond potential is
active (e.g. Morse and harmonic bonds in the same system), each bonded
slot selects the bonds whose type uses its own `potential`, preserving
the `(atom_i, atom_j)` sort order, and derives its own device bond array
plus its own `atom_bond_offsets` / `atom_bond_indices` reduction map over
that subset — built exactly as above but restricted to the selected
bonds. Every bond is owned by exactly one bonded slot; when a single bond
potential is active, its subset is the whole list and its derived map
equals the fields above. See `morse-bonded.md` and `harmonic-bond.md`.

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

### Dihedral list <!-- rq-3c978326 -->

For `D` dihedrals among `N` particles, the host-side `DihedralList`
carries:

- `dihedrals: Vec<Dihedral>` — length `D`, sorted by
  `(atom_i, atom_j, atom_k, atom_l)`. Each `Dihedral` records
  `atom_i: u32`, `atom_j: u32`, `atom_k: u32`, `atom_l: u32` (with
  `atom_i ≤ atom_l` after canonicalisation), and
  `dihedral_type_index: u32` (an index into the config's
  `[[dihedral_types]]` array).
- `atom_dihedral_offsets: Vec<u32>` — length `N + 1`. For atom `a`,
  the slice
  `atom_dihedral_indices[atom_dihedral_offsets[a] .. atom_dihedral_offsets[a+1]]`
  lists the slot indices in the dihedral-quadruple buffer that
  contribute force to atom `a`.
- `atom_dihedral_indices: Vec<u32>` — length `4 * D` (each dihedral
  appears four times, once for each of its four atoms). Entry `m`
  is an index into the per-dihedral force buffer of length `4 * D`.
  Within each atom's slice, entries are sorted by the underlying
  dihedral index so the reduction's summation order is deterministic
  across runs.

The atom-to-dihedral indexing is built at load time and uploaded to
the device once. Dihedrals are immutable for the lifetime of a run;
no recomputation is necessary.

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
  `atom_excl_partners`. Consumed by the SPME real-space kernel.

Pair-potential CUDA kernels consult this per-atom lookup table through
the shared device helper `exclusion_scale` declared in
`kernels/exclusions.cuh` (see *Device-side Exclusion Helper* below).
The helper performs the linear scan over atom `i`'s partner range and
returns either the matching scale (from the per-potential scale array
the kernel passes in) or `1.0f` when `j` is not present.

### Charge list <!-- rq-4cae9784 -->

The `ChargeList` is produced only when the `.topology` file contains a
`[charges]` section (an absent section yields no `ChargeList`). It
carries:

- `charges: Vec<Real>` — length `particle_count`. `charges[a]` is the
  charge of particle `a` in atomic units (elementary charges), stored at
  the atom's declared index. Because the section is all-or-nothing, every
  entry is populated.
- `particle_count: usize` — `N`.

The `ChargeList` is a dense per-atom assignment, not a sparse list of
overrides: its length equals the particle count and each particle's
charge is present. The runner reads it once at setup to populate the
per-particle charge array uploaded to `posq.w` (see
`simulation-runner.md`); it induces no device-side table of its own and
is immutable for the lifetime of a run.

### Molecule list <!-- rq-3246a873 -->

For `M` molecules among `N` particles, the host-side `MoleculeList`
carries:

- `mol_atom_offsets: Vec<u32>` — length `M + 1`. For molecule `m`, the
  slice `mol_atom_indices[mol_atom_offsets[m] .. mol_atom_offsets[m+1]]`
  lists the particle indices belonging to that molecule, in ascending
  order.
- `mol_atom_indices: Vec<u32>` — length `N`. Each particle index appears
  exactly once (the molecules partition the particles).
- `particle_count: usize` — `N`.
- `molecule_count: usize` — `M`.

The two index tables are uploaded to the device once at load time and read
by the Monte-Carlo barostat's centre-of-mass scale kernel
(`integration/mc-barostat.md`). They are immutable for the lifetime of a
run.

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
to itself: the LJ kernel passes `atom_excl_lj_scales`, the SPME
real-space kernel passes `atom_excl_coul_scales`. Pair-potential `.cu` files
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

- `Dihedral` — `Debug, Clone, Copy`. Fields: `atom_i: u32`, <!-- rq-a6df14c8 -->
  `atom_j: u32`, `atom_k: u32`, `atom_l: u32`,
  `dihedral_type_index: u32`. Outer atoms satisfy
  `atom_i ≤ atom_l` (in practice `<`, because the four atoms must be
  distinct).

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

- `DihedralList` — host-side. Fields: `dihedrals: Vec<Dihedral>`, <!-- rq-07be6b5e -->
  `atom_dihedral_offsets: Vec<u32>`, `atom_dihedral_indices: Vec<u32>`,
  `particle_count: usize`.

  Method `DihedralList::is_empty(&self) -> bool` —
  `self.dihedrals.is_empty()`.

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

- `MoleculeList` — host-side. Fields: `mol_atom_offsets: Vec<u32>` <!-- rq-42195e6f -->
  (length `molecule_count + 1`), `mol_atom_indices: Vec<u32>`
  (length `particle_count`), `particle_count: usize`,
  `molecule_count: usize`. Partitions the particles into the connected
  components of the combined bond + constraint graph (see *Molecule
  grouping*).

  Method `MoleculeList::molecule_count(&self) -> usize`.

- `ChargeList` — host-side. Fields: `charges: Vec<Real>` (length <!-- rq-52450608 -->
  `particle_count`, in atomic units), `particle_count: usize`. A complete
  per-atom charge assignment parsed from a `[charges]` section (see
  *Charge list*). Produced only when the section is present; `charges[a]`
  is particle `a`'s charge.

- `TopologyFileError` — error type returned by the parser. Variants: <!-- rq-bca0adbc -->
  - `Io(String)` — failed to read the file.
  - `UnknownSection { name: String, line_number: usize }` — a section
    header is not one of the accepted names (`bonds`, `exclusions`,
    `angles`, `dihedrals`, `constraints`, `charges`).
  - `DuplicateSection { name: String, line_number: usize }` — the
    same section header appears twice.
  - `ContentOutsideSection { line_number: usize }` — non-blank,
    non-comment content appears before the first section header.
  - `InvalidBondRow { line_number: usize, reason: String }` — column
    count wrong, atom index unparseable, etc.
  - `InvalidAngleRow { line_number: usize, reason: String }`.
  - `InvalidDihedralRow { line_number: usize, reason: String }`.
  - `InvalidExclusionRow { line_number: usize, reason: String }`.
  - `AtomIndexOutOfRange { line_number: usize, index: u32, max: u32 }`.
  - `SelfBond { line_number: usize, atom: u32 }`.
  - `RepeatedAtomInAngle { line_number: usize, atom: u32 }` — at
    least two of `atom_i`, `atom_j`, `atom_k` are equal.
  - `RepeatedAtomInDihedral { line_number: usize, atom: u32 }` — at
    least two of `atom_i`, `atom_j`, `atom_k`, `atom_l` are equal.
  - `SelfExclusion { line_number: usize, atom: u32 }`.
  - `DuplicateBond { atom_i: u32, atom_j: u32 }`.
  - `DuplicateAngle { atom_i: u32, atom_j: u32, atom_k: u32 }`.
  - `DuplicateDihedral { atom_i: u32, atom_j: u32, atom_k: u32, atom_l: u32, dihedral_type_name: String }` —
    a `(canonical quadruple, dihedral_type_name)` pair appears twice
    in `[dihedrals]`. Distinct dihedral types on the same quadruple
    are *not* rejected (that is the multi-Fourier-term case).
  - `DuplicateExclusion { atom_i: u32, atom_j: u32 }`.
  - `UnknownBondType { line_number: usize, name: String }`.
  - `UnknownAngleType { line_number: usize, name: String }`.
  - `UnknownDihedralType { line_number: usize, name: String }`.
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
  - `InvalidChargeRow { line_number: usize, reason: String }` — a
    `[charges]` row does not have exactly two columns, its
    `atom_index` does not parse as `u32`, or its `charge` does not
    parse as a finite `f64`.
  - `DuplicateChargeAtom { atom: u32 }` — an atom index appears in more
    than one `[charges]` row.
  - `IncompleteCharges { present: usize, particle_count: usize }` — the
    `[charges]` section is present but assigns charges to only `present`
    of the `particle_count` atoms (the section is all-or-nothing).

### Functions <!-- rq-e66012e0 -->

- `load_topology_file(path: &Path, particle_count: usize, bond_type_names: &[&str], angle_type_names: &[&str], dihedral_types: &[DihedralTypeConfig], constraint_types: &[NamedSlotConfig], constraint_registry: &ConstraintRegistry, units: UnitSystem) -> Result<(BondList, AngleList, DihedralList, ExclusionList, ConstraintList, Option<ChargeList>), TopologyFileError>` <!-- rq-12b7dcb6 -->
  - Reads the file at `path` and parses all six sections.
  - Validates every constraint described in *File Format*.
  - Returns the canonicalised, sorted `BondList`, `AngleList`,
    `DihedralList`, the *effective* `ExclusionList` (explicit
    entries plus implicit bond-derived, angle-derived,
    constraint-derived, and dihedral-derived-1-4 defaults), the
    `ConstraintList` (groups sorted by minimum particle index per
    `integration/constraint-framework.md`), and an
    `Option<ChargeList>` — `Some(charge_list)` when the file contains a
    `[charges]` section (a complete per-atom charge assignment; see
    *Charge entries* and *Charge list*), or `None` when the section is
    absent.
  - The caller passes `particle_count` (used to bound atom indices),
    `bond_type_names` (used to resolve `bond_type_name` strings to
    indices in the config's `[[bond_types]]` array),
    `angle_type_names` (used to resolve `angle_type_name` strings to
    indices in the config's `[[angle_types]]` array),
    `dihedral_types` (the full parsed `[[dihedral_types]]` array of
    `DihedralTypeConfig` entries, used to resolve
    `dihedral_type_name` strings to indices and to read each type's
    `scale_lj_14` / `scale_coul_14` fields when generating implicit
    1-4 exclusions),
    `constraint_types` (the full parsed `[[constraint_types]]`
    array of `NamedSlotConfig` entries, used to resolve
    `constraint_type_name` strings to indices),
    `constraint_registry` (used to look up each constraint type's
    builder for the `expected_atom_count(&params)` query that
    size-checks `[constraints]` rows), and
    `units` (the config's `UnitSystem`, used to convert each
    `[charges]` value from the user's unit system to atomic units so
    the returned `ChargeList` holds atomic-unit charges).
  - When a `[charges]` section is present it must assign a charge to
    every particle exactly once; a missing atom is rejected as
    `IncompleteCharges`, a repeated atom as `DuplicateChargeAtom`. The
    function does not consult the per-type `charge` fields and does not
    enforce the "`[charges]` present ⇒ per-type charge must be zero"
    rule — that cross-check, and the net-charge warning, are applied by
    the runner (see `simulation-runner.md`).

- `MoleculeList::from_topology(particle_count: usize, bonds: &BondList, constraints: &ConstraintList) -> MoleculeList` <!-- rq-b0bdc311 -->
  - Builds the connected-component partition described in *Molecule
    grouping* from the parsed bond and constraint lists.
  - Each bond contributes an edge between its two atoms; each constraint
    group contributes edges joining all its atoms into one component.
  - Every particle in no bond and no constraint becomes a singleton
    molecule.
  - Molecules are ordered by their minimum particle index; atoms within
    each molecule are listed in ascending particle-index order.
  - Pure function of its inputs: identical lists produce a byte-identical
    `MoleculeList`. With empty bond and constraint lists every particle is
    its own molecule (`molecule_count == particle_count`).

## Out of Scope <!-- rq-ad0edc95 -->

- Improper-dihedral and CMAP terms. The `.topology` file's schema
  reserves no syntax for `[impropers]` or `[cmap]`; both are
  planned as separate future sections with their own per-row layout
  and their own implicit-exclusion rules. The `[dihedrals]` section
  is reserved for proper dihedrals only.
- Per-bond, per-angle, and per-dihedral parameter overrides (every
  bond's parameters come from its bond type; every angle's
  parameters come from its angle type; every dihedral's parameters
  come from its dihedral type).
- Connected-component merging of `[constraints]` rows that share
  atoms. v1 requires disjoint clusters; cross-row groups arrive with
  M-SHAKE. The on-disk format and the in-memory `ConstraintList`
  layout already accommodate the merged case.
- Forming or breaking bonds, angles, dihedrals, or constraints
  during a simulation. All lists are fixed at start of run.
- A separate `[scaled_exclusions]` section for declaring 1-4 (or
  other) scaled exclusions independent of the dihedral list. The
  existing four-column `[exclusions]` form already lets the user
  emit any `(i, j, scale_lj, scale_coul)` row by hand, and the
  dihedral-derived implicit 1-4 rule covers the common AMBER /
  GROMACS / OPLS / CHARMM convention. A future
  `[scaled_exclusions]` section would reuse the existing
  `Exclusion` data model and the existing
  layered-precedence machinery in *Effective exclusions*; only a
  new parser entry point would be needed.
- Multi-`.topology`-file support; the config references at most one
  file.
- Binary topology formats.
- Automatic angle / dihedral derivation from the bond graph. Angle
  and dihedral declarations are explicit.
- An exclusion "scale = NaN" sentinel to mean "remove implicit
  exclusion". An explicit entry with `scale = 1.0` already achieves
  that effect.
- Partial per-atom charge coverage. A `[charges]` section that lists
  only some atoms and leaves the rest to fall back to their type
  `charge` is rejected (`IncompleteCharges`); the section is
  all-or-nothing. A system that needs distinct charges on only a few
  atoms lists every atom's charge, using the type charge value for the
  unchanged ones.
- Per-atom masses or per-atom Lennard-Jones parameters. Charge is the
  only per-atom force-field quantity carried by the `.topology` file;
  mass and LJ `sigma`/`epsilon` remain per-type (`io/config-schema.md`).
  Collapsing charge to per-atom lets a force field define one particle
  type per distinct `(mass, sigma, epsilon)` tuple rather than one per
  distinct `(mass, sigma, epsilon, charge)` tuple.

---

## Gherkin Scenarios <!-- rq-bf62f645 -->

```gherkin
Feature: Topology file with bonds, angles, and exclusions

  Background:
    Given a temporary directory tmp
    And a bond_type_names slice of ["CC", "CN", "OH"]
    And an angle_type_names slice of ["HOH"]
    And a dihedral_types slice (empty by default; scenarios that exercise
      [dihedrals] supply their own typed entries with explicit scales)
    And a units selector of "atomic" unless a scenario states otherwise
    And particle_count = 4
    # load_topology_file returns a trailing Option<ChargeList>; scenarios
    # that do not exercise [charges] receive None and omit it from the
    # returned tuple for brevity.

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
    Then atom_excl_offsets equals [0, 2, 4, 5, 6]
      (every pair is mirror-expanded so atom_j's partner list also names atom_i)
    And atom 0's partners are [1, 2] with LJ scales [0.0, 0.5] and Coulomb scales [0.0, 0.5]
    And atom 1's partners are [0, 3] with LJ scales [0.0, 0.5] and Coulomb scales [0.0, 0.833]
    And atom 2's partners are [0] with LJ scales [0.5] and Coulomb scales [0.5]
    And atom 3's partners are [1] with LJ scales [0.5] and Coulomb scales [0.833]

  # --- IO and parser errors ---

  @rq-ef5aa4b7
  Scenario: File does not exist
    When load_topology_file("/tmp/missing.topology", 4, &["CC","CN","OH"], &["HOH"]) is called
    Then it returns Err(TopologyFileError::Io(_))

  @rq-4c245ce7
  Scenario: Unknown section header
    Given tmp/sim.topology with a section "[impropers]"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::UnknownSection { name: "impropers", line_number: _ })

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

  # --- Dihedral rows ---

  @rq-2d7dd2a1
  Scenario: Load a topology file containing a dihedral
    Given tmp/sim.topology containing
      """
      [dihedrals]
      0 1 2 3 CT-CT-CT-CT_n3
      """
    And dihedral_types contains an entry "CT-CT-CT-CT_n3" potential="periodic"
    When load_topology_file(tmp/sim.topology, 4, &bond_types, &angle_types,
      &dihedral_types, &constraint_types, &constraint_registry) is called
    Then it returns Ok((bond_list, angle_list, dihedral_list, exclusion_list,
      constraint_list))
    And dihedral_list.dihedrals equals
      [(atom_i=0, atom_j=1, atom_k=2, atom_l=3, dihedral_type_index=0)]

  @rq-7b7659be
  Scenario: Dihedral is canonicalised so atom_i <= atom_l
    Given a single dihedral "3 2 1 0 CT-CT-CT-CT_n3" with particle_count = 4
    When load_topology_file is called
    Then dihedral_list.dihedrals[0].atom_i equals 0
    And dihedral_list.dihedrals[0].atom_j equals 1
    And dihedral_list.dihedrals[0].atom_k equals 2
    And dihedral_list.dihedrals[0].atom_l equals 3

  @rq-5e3b9b77
  Scenario: Dihedrals are sorted by (atom_i, atom_j, atom_k, atom_l)
    Given two dihedrals "0 1 2 4 X" and "0 1 2 3 X" with particle_count = 5
    When load_topology_file is called
    Then dihedral_list.dihedrals equals
      [(0, 1, 2, 3, type=0), (0, 1, 2, 4, type=0)]

  @rq-58221c00
  Scenario: Two rows on the same quadruple with different dihedral types are accepted
    Given two dihedral rows "0 1 2 3 CT-CT-CT-CT_n1" and "0 1 2 3 CT-CT-CT-CT_n3"
    And dihedral_types contains both "CT-CT-CT-CT_n1" and "CT-CT-CT-CT_n3"
    When load_topology_file is called
    Then dihedral_list.dihedrals has length 2
    And both entries share atom indices (0, 1, 2, 3) and differ only in
      dihedral_type_index

  @rq-0426f8c9
  Scenario: Two rows on the same quadruple naming the same dihedral type are rejected
    Given two dihedral rows "0 1 2 3 CT-CT-CT-CT_n3" and "3 2 1 0 CT-CT-CT-CT_n3"
      (the second canonicalises to (0,1,2,3))
    When load_topology_file is called
    Then it returns Err(TopologyFileError::DuplicateDihedral { atom_i: 0, atom_j: 1, atom_k: 2, atom_l: 3, dihedral_type_name: "CT-CT-CT-CT_n3" })

  @rq-10274a65
  Scenario: Dihedral row with wrong column count
    Given a dihedral row "0 1 2 3" (missing type)
    When load_topology_file is called
    Then it returns Err(TopologyFileError::InvalidDihedralRow { line_number: _, reason: _ })

  @rq-b3ffde38
  Scenario: Dihedral row with non-integer index
    Given a dihedral row "abc 1 2 3 X"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::InvalidDihedralRow { reason: _, .. })

  @rq-3c83b98a
  Scenario: Dihedral row with atom index out of range
    Given a dihedral row "0 1 2 9 X" and particle_count = 4
    When load_topology_file is called
    Then it returns Err(TopologyFileError::AtomIndexOutOfRange { index: 9, max: 3, .. })

  @rq-152fe459
  Scenario: Repeated atom in dihedral rejected (i == j)
    Given a dihedral row "1 1 2 3 X"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::RepeatedAtomInDihedral { atom: 1, .. })

  @rq-97961787
  Scenario: Repeated atom in dihedral rejected (j == k)
    Given a dihedral row "0 1 1 3 X"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::RepeatedAtomInDihedral { atom: 1, .. })

  @rq-6789ef26
  Scenario: Repeated atom in dihedral rejected (k == l)
    Given a dihedral row "0 1 2 2 X"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::RepeatedAtomInDihedral { atom: 2, .. })

  @rq-cb1a8507
  Scenario: Repeated atom in dihedral rejected (i == l)
    Given a dihedral row "0 1 2 0 X"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::RepeatedAtomInDihedral { atom: 0, .. })

  @rq-e1df90e7
  Scenario: Unknown dihedral type name rejected
    Given a dihedral row "0 1 2 3 ZZ" with dihedral_types containing only "X"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::UnknownDihedralType { name: "ZZ", .. })

  # --- Dihedral-derived implicit 1-4 exclusions ---

  @rq-99bd424a
  Scenario: Implicit 1-4 exclusion is added for a dihedral whose (i, l) pair is otherwise uncovered
    Given a single dihedral "0 1 2 3 D"
    And no bonds, no angles, no constraints, no explicit exclusions
    And dihedral_types contains "D" potential="periodic" with default
      scale_lj_14=0.5, scale_coul_14=0.8333
    When load_topology_file is called
    Then exclusion_list.entries contains (0, 3, scale_lj=0.5, scale_coul=0.8333)

  @rq-cb96dfd0
  Scenario: Implicit 1-4 uses the dihedral_type's scale_lj_14 / scale_coul_14
    Given a single dihedral "0 1 2 3 D"
    And dihedral_types contains "D" with scale_lj_14=0.25 scale_coul_14=0.75
    When load_topology_file is called
    Then exclusion_list.entries contains (0, 3, scale_lj=0.25, scale_coul=0.75)

  @rq-aa00b384
  Scenario: Bond-derived (1, 0, 0) exclusion overrides a dihedral's 1-4 entry for the same pair
    Given a 4-atom system with a [bond] "0 3 X", an angle and a dihedral whose
      1-4 pair is also (0, 3), and no explicit exclusion on (0, 3)
    When load_topology_file is called
    Then exclusion_list.entries contains (0, 3, scale_lj=0.0, scale_coul=0.0)
    And exclusion_list.entries does not contain (0, 3) with non-zero scales

  @rq-92b8e6af
  Scenario: Angle-derived (1, 0, 0) exclusion overrides a dihedral's 1-4 entry for the same pair
    Given a 4-atom system with an angle "0 _ 3 _" giving 1-3 pair (0, 3) and a
      dihedral whose 1-4 pair is also (0, 3)
    When load_topology_file is called
    Then exclusion_list.entries contains (0, 3, scale_lj=0.0, scale_coul=0.0)

  @rq-bafeed55
  Scenario: Explicit [exclusions] row overrides a dihedral-derived 1-4 entry
    Given a single dihedral "0 1 2 3 D" and an explicit exclusion row "0 3 0.4 0.6"
    When load_topology_file is called
    Then exclusion_list.entries contains (0, 3, scale_lj=0.4, scale_coul=0.6)
    And exclusion_list.entries does not contain (0, 3) with the type's
      default scales

  @rq-ce519f29
  Scenario: First-wins when two dihedrals share the same (i, l) 1-4 pair
    Given two dihedrals on the same canonical quadruple (0, 1, 2, 3) of types
      D_a (scale_lj_14=0.5, scale_coul_14=0.8333) and D_b
      (scale_lj_14=0.25, scale_coul_14=0.75)
    And the canonical dihedral order is [D_a, D_b]
    When load_topology_file is called
    Then exclusion_list.entries contains exactly one (0, 3) row with scales
      (0.5, 0.8333) — from the first dihedral
    And no (0, 3) row with scales (0.25, 0.75) is present

  @rq-720fd816
  Scenario: A dihedral with full LJ and Coul 1-4 (CHARMM convention) adds the entry
    Given a single dihedral "0 1 2 3 D" with scale_lj_14=1.0 scale_coul_14=1.0
    When load_topology_file is called
    Then exclusion_list.entries contains (0, 3, scale_lj=1.0, scale_coul=1.0)

  # --- Per-atom indexing for dihedrals ---

  @rq-2d18165a
  Scenario: atom_dihedral_offsets reflects sorted dihedral list
    Given particle_count = 5 and dihedrals "0 1 2 3 D" and "1 2 3 4 D"
    When load_topology_file is called
    Then dihedral_list.atom_dihedral_offsets equals [0, 1, 3, 5, 7, 8]
    And atom 0's dihedral indices reference slot 0 (its slot in dihedral 0)
    And atom 1's dihedral indices reference slot 1 (in dihedral 0)
      and slot 4 (its slot in dihedral 1)
    And atom 4's dihedral indices reference slot 7 (its slot in dihedral 1)

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

  # --- Molecule grouping ---

  @rq-392ac5d3
  Scenario: Each constraint group is one molecule
    Given a ConstraintList with rigid-water groups [0,1,2] and [3,4,5]
      and an empty BondList, particle_count = 6
    When MoleculeList::from_topology(6, &bonds, &constraints) is called
    Then molecule_count equals 2
    And molecule 0's atoms are [0, 1, 2]
    And molecule 1's atoms are [3, 4, 5]

  @rq-45d384b3
  Scenario: Bonds join their atoms into one molecule
    Given a BondList with bonds (0,1) and (1,2) and an empty ConstraintList,
      particle_count = 4
    When MoleculeList::from_topology(4, &bonds, &constraints) is called
    Then molecule_count equals 2
    And molecule 0's atoms are [0, 1, 2]
    And molecule 1's atoms are [3]

  @rq-763200f7
  Scenario: An atom in no bond and no constraint is its own molecule
    Given empty BondList and ConstraintList, particle_count = 3
    When MoleculeList::from_topology(3, &bonds, &constraints) is called
    Then molecule_count equals 3
    And every molecule has exactly one atom

  @rq-ae8e2b7d
  Scenario: Molecules are ordered by minimum particle index
    Given constraint groups [3,4,5] and [0,1,2], particle_count = 6
    When MoleculeList::from_topology(6, &bonds, &constraints) is called
    Then molecule 0's atoms start with 0
    And molecule 1's atoms start with 3

  @rq-ebab7fd7
  Scenario: Atoms within a molecule are listed in ascending order
    Given a BondList with bonds (2,0) and (0,1), particle_count = 3
    When MoleculeList::from_topology(3, &bonds, &constraints) is called
    Then molecule 0's atoms are [0, 1, 2]

  # --- Charge rows ---

  @rq-41691785
  Scenario: Load a topology file with a [charges] section
    Given tmp/sim.topology containing
      """
      [charges]
      0 -0.834
      1  0.417
      2  0.417
      3  0.000
      """
    And a units selector of "atomic"
    When load_topology_file is called with particle_count = 4
    Then the returned Option<ChargeList> is Some(charge_list)
    And charge_list.charges equals [-0.834, 0.417, 0.417, 0.0] (in atomic units)
    And charge_list.particle_count equals 4

  @rq-08ab6b03
  Scenario: A topology file with no [charges] section yields None
    Given tmp/sim.topology with a [bonds] section and no [charges] section
    When load_topology_file is called
    Then the returned Option<ChargeList> is None

  @rq-815534fc
  Scenario: Charge rows may appear in any atom order
    Given a [charges] section with rows "3 0.0", "1 0.417", "0 -0.834", "2 0.417"
    When load_topology_file is called with particle_count = 4
    Then charge_list.charges equals [-0.834, 0.417, 0.417, 0.0]
      (each charge stored at its declared atom_index, not row order)

  @rq-f85569fa
  Scenario: Negative and zero charges are accepted
    Given a [charges] section assigning atom 0 charge -1.0 and atom 1 charge 0.0
      (and atoms 2, 3 any finite charge)
    When load_topology_file is called with particle_count = 4
    Then it returns Ok with charge_list.charges[0] == -1.0 and charge_list.charges[1] == 0.0

  @rq-f6b8d252
  Scenario: SI charge values are converted to atomic units at load
    Given a [charges] section assigning atom 0 the value 1.602176634e-19
      (and atoms 1, 2, 3 the value 0.0)
    And a units selector of "si"
    When load_topology_file is called with particle_count = 4
    Then charge_list.charges[0] equals 1.0 within f32 round-off
      (one elementary charge, the atomic-unit charge)

  @rq-ce19b7ab
  Scenario: A [charges] section missing an atom is rejected
    Given a [charges] section with rows for atoms 0, 1, 2 only (atom 3 absent)
    When load_topology_file is called with particle_count = 4
    Then it returns Err(TopologyFileError::IncompleteCharges { present: 3, particle_count: 4 })

  @rq-194105d0
  Scenario: An empty [charges] section with a nonzero particle count is rejected
    Given a [charges] section header with no rows
    When load_topology_file is called with particle_count = 4
    Then it returns Err(TopologyFileError::IncompleteCharges { present: 0, particle_count: 4 })

  @rq-bc05f0f0
  Scenario: A duplicate atom in [charges] is rejected
    Given a [charges] section with rows for atoms 0, 1, 2, 3 and a second row for atom 1
    When load_topology_file is called with particle_count = 4
    Then it returns Err(TopologyFileError::DuplicateChargeAtom { atom: 1 })

  @rq-8f03bfba
  Scenario: A charge row with an out-of-range atom index is rejected
    Given a [charges] row "9 0.0" and particle_count = 4
    When load_topology_file is called
    Then it returns Err(TopologyFileError::AtomIndexOutOfRange { index: 9, max: 3, .. })

  @rq-c49984a5
  Scenario: A charge row with the wrong column count is rejected
    Given a [charges] row "0 0.5 extra"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::InvalidChargeRow { line_number: _, reason: _ })

  @rq-4f430b32
  Scenario: A charge row with a non-finite charge is rejected
    Given a [charges] row "0 nan"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::InvalidChargeRow { line_number: _, reason: _ })

  @rq-6f8f35a2
  Scenario: A charge row with a non-integer atom index is rejected
    Given a [charges] row "abc 0.5"
    When load_topology_file is called
    Then it returns Err(TopologyFileError::InvalidChargeRow { line_number: _, reason: _ })

  @rq-27616511
  Scenario: A duplicate [charges] section header is rejected
    Given tmp/sim.topology with two [charges] headers
    When load_topology_file is called
    Then it returns Err(TopologyFileError::DuplicateSection { name: "charges", line_number: _ })
```
