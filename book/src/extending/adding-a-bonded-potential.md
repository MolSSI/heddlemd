# Adding a bonded potential

A bonded potential describes an intramolecular interaction whose atom indices
come explicitly from the topology: two atoms for a bond, three for an angle,
or four for a dihedral. It does not search the neighbor list.

Read the [extension overview](index.md) and [Adding a pair
potential](adding-a-pair-potential.md) first. Bonded and pair potentials use
the same builder, parameter-claim, and run-time CUDA composition machinery.
This guide concentrates on the different topology and reduction contracts.

## Choose the closest example

This page uses a `"cubic"` bond as its running example. Begin with
`src/forces/harmonic_bond.rs` for the simplest complete pattern or
`src/forces/morse.rs` for another two-particle potential. Angle and dihedral
implementations follow the same workflow but use different functor signatures;
the table near the end identifies their examples.

## Understand the bonded execution path

The following pieces are the same as for a pair potential: a cloneable builder implementing `PotentialBuilder` with a
`params_claim`, `validate_params`, `convert_params`, and `build`; a `State`
implementing `Potential`; a CUDA functor emitted as a Rust source string and
compiled by `nvrtc`; registration by adding one line to the potential
`Builtins` roster.

The important differences are:

- **The claim category is `BondType`** (or `AngleType` / `DihedralType`), and
  it matches the `kind` field of `[[bond_types]]` entries.
- **`max_cutoff()` returns `None`** — bonded potentials don't consume the
  neighbor list; they iterate an explicit per-bond index table.
- **`compute()` is NOT bypassed.** For pair slots the framework skips
  `compute` (the JIT kernel does everything). For bonded slots, the composed
  kernel writes each bond's contribution into a per-bond scratch buffer, and
  then `compute()` runs the **reduction** step (`reduce_bond_forces`) that
  scatters those per-bond contributions into per-atom forces. So your
  `compute` must run exactly that reduction and nothing else — doing more
  double-applies forces.
- **No new `.cu` file is needed.** The reduction kernel `reduce_bond_forces`
  already exists and is re-exported from `crate::gpu`; the per-bond physics
  is the JIT source string.

## Map topology entries to the potential

The user declares `[[bond_types]] name = "CT-CT" kind = "cubic"` with a
`params` inline table, and references it by name from the `.topology` file's
`[bonds]` section (`atom_i atom_j <bond_type_name>`). The loader resolves the
name to a global `bond_type_index`, canonicalizes each bond `i < j`, and sorts
the bond list — that fixed order is the determinism anchor.

Your `build(cx)` filters `cx.bond_list` to the bonds whose type uses your
`kind` (`BondList::filter_by_type_index`), which rebuilds the per-atom
reduction map (`atom_bond_offsets` / `atom_bond_indices`) over just that
subset. **Parameter tables stay indexed by the global `bond_type_index`**
(rows for other kinds hold zero placeholders your slot never reads), because
the index the kernel reads out of the bond table is the global one.

Two things you get for free: **1-2 exclusions** (every bond contributes an
implicit non-bonded exclusion, derived kind-agnostically from the full bond
list), and — for dihedrals — **1-4 scaling** from the dihedral type's common
`scale_lj_14` / `scale_coul_14` fields.

## Implement the one-bond calculation

The bond functor computes one bond's contribution from the scalar distance.
The composer's generated entry point computes the minimum-image displacement
`(dx, dy, dz) = r_i − r_j`, `r2`, and `r = sqrt(r2)`, then calls your
`evaluate` with the fixed signature:

```cpp
struct CubicBondFunctor {
    const Real *bond_k; const Real *bond_r0; const Real *bond_kcub;
    __device__ inline void evaluate(
        Real r2, Real r,
        unsigned int bond_type_index,
        Real dx, Real dy, Real dz,
        Real &fmag, Real &u_k, Real &w_k) const
    {
        if (r < R(1.0e-7)) { fmag = R(0.0); u_k = R(0.0); w_k = R(0.0); return; }
        Real k = bond_k[bond_type_index], r0 = bond_r0[bond_type_index],
             kcub = bond_kcub[bond_type_index];
        Real dr = r - r0;
        // U = ½ k dr² + ⅓ k kcub dr³ ;  dU/dr = k dr (1 + kcub dr)
        fmag = -k * dr * (R(1.0) + kcub * dr) / r;   // per-component factor (÷ r)
        u_k  = R(0.5)*k*dr*dr + (R(1.0)/R(3.0))*k*kcub*dr*dr*dr;  // FULL bond energy
        w_k  = fmag * r2;                             // FULL scalar virial
    }
};
```

The generated entry point expects these conventions:

- **`fmag` is `−(dU/dr)/r`** (already divided by `r`). The entry point writes
  force `fmag·(dx,dy,dz)` onto atom `i` and its negation onto atom `j`
  (Newton's third law), into per-bond scratch slots `2k` and `2k+1`.
- **`u_k` and `w_k` are the FULL bond values** — the entry point applies the
  ½-each split across the two atoms. Don't pre-halve them.
- Use the precision shim (`Real`, `R(x)`, `Real_*`) and a unique struct/helper
  prefix, exactly as for pair fragments.

### Generate the argument interface from one schema

As with pair potentials, generate `entry_point_args` and
`functor_init_source` from one `KernelArgSchema` (built with
`KernelArgSchema::intramolecular(...)`) and route `bind_bonded_force_args`
through a `KernelArgBinder` on the same schema.

## Preserve the fixed reduction order

Each bond is processed by one thread that writes only its own two scratch
slots — no cross-thread contention. The reduction then sums each atom's bonds
in the fixed index order set by the sorted bond list, so the per-atom float
sum is bit-identical run to run. Bonded potentials therefore need no
fixed-point accumulator (unlike the pair path). Keep `evaluate` a pure
per-bond function.

## Implement the Rust types

Create `src/forces/cubic_bond.rs` and follow `harmonic_bond.rs` for the
complete implementation. The following code is an abbreviated outline rather
than a directly compilable example.

```rust
pub const CUBIC_BOND_KIND: &str = "cubic";
const LABEL: &str = "cubic_bond";

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, crate::units::Convert)]
#[serde(deny_unknown_fields)]
pub struct CubicBondParams {
    pub k: crate::units::Stiffness,        // energy / length²
    pub r0: crate::units::Length,
    pub kcub: crate::units::InverseLength,
}

#[derive(Debug)]
pub struct CubicBondState {
    /* device: bonds [i,j,type_idx]; atom_bond_offsets/indices over the filtered
       subset; bond_k/r0/kcub param tables sized to bond_types.len();
       per-bond scratch bond_pair_{x,y,z,energy,virial} of length 2*bond_count */
}

impl Potential for CubicBondState {
    fn label(&self) -> &'static str { LABEL }
    fn max_cutoff(&self) -> Option<Real> { None }          // bonded → no neighbor cutoff
    // frequency_class() defaults to Fast.
    fn compute(&mut self, /* … */ level: AggregateLevel) -> Result<(), ForceFieldError> {
        // if particle_count == 0 || bond_count == 0 { return Ok(()); }
        // reduce_bond_forces(..., write_scalars = matches!(level, ForcesAndScalars))
        Ok(())
    }
    fn jit_participant(&self) -> Option<JitParticipant<'_>> {
        Some(JitParticipant::Bonded(self))
    }
}

impl BondedPotential for CubicBondState {
    fn bonded_force_fragment(&self) -> BondedForceFragment { cubic_bonded_force_fragment() }
    fn bonded_scratch(&self) -> BondedScratchView<'_> { /* bonds + bond_pair_* + bond_count */ }
    fn bind_bonded_force_args(&self, _ctx: &ForceLaunchContext<'_>, builder: &mut ForceLaunchBuilder) {
        // KernelArgBinder::new(&cubic_arg_schema(), LABEL, builder); push bond_k/r0/kcub
    }
}

#[derive(Debug, Clone)]
pub struct CubicBondBuilder;
impl PotentialBuilder for CubicBondBuilder {
    fn build(&self, cx: &PotentialBuildContext<'_>) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        let has = cx.bond_list.bonds.iter()
            .any(|b| cx.bond_types.get(b.bond_type_index as usize)
                       .is_some_and(|t| t.kind == CUBIC_BOND_KIND));
        if !has { return Ok(None); }                       // activation
        Ok(Some(Box::new(/* CubicBondState::new(cx.gpu, cx.bond_list, cx.bond_types)? */)))
    }
    fn params_claim(&self) -> Option<PotentialParamsClaim> {
        Some(PotentialParamsClaim {
            category: PotentialParamsCategory::BondType, kind: CUBIC_BOND_KIND })
    }
    fn validate_params(&self, entry: PotentialConfigEntry<'_>) -> Result<(), ConfigError> {
        let PotentialConfigEntry::BondType(bt) = entry else { unreachable!() };
        // deserialize bt.params into CubicBondParams; k, r0 finite & positive …
        Ok(())
    }
    fn convert_params(&self, units: crate::units::UnitSystem, params: &mut toml::Value)
        -> Result<(), ConfigError> {
        crate::registry::convert_params_in_place::<CubicBondParams>(units, params)
    }
}
```

## Register and document the potential

Add `rqm/forces/cubic-bond.md`; follow `rqm/forces/harmonic-bond.md`
and cross-reference `rqm/forces/jit-composed-intramolecular.md`.

Then edit the following existing files:

- **`src/forces/mod.rs`** — three lines: `pub mod cubic_bond;`, the
  `pub use cubic_bond::{...}` re-export, and `Box::new(CubicBondBuilder)` in
  the `vec![...]` of `impl Builtins for dyn PotentialBuilder` (roster at
  ~line 386), conventionally next to the other bonded builders. Because your
  `kind` is unique within the `BondType` category, position only affects the
  slot index.

**No change needed** to `src/io/config.rs` (`BondTypeConfig` is open-shaped;
routing is claim-driven — an unclaimed kind is `UnknownKind`), `build.rs`,
`src/gpu/device.rs`, or any `.cu` file. `PotentialParamsCategory::BondType`
already exists.

## Test the potential

Add the potential to `builtin_consistency_fixtures()` in
`tests/common/consistency.rs` using `ConsistencyFixture::bond`, `angle`, or
`dihedral` as appropriate. Use the fragment label, register only the builder
under test, and choose sample coordinates that exercise nonzero forces and
the important regions of the potential. Add analytical `ReferencePoint`
values so a consistently mis-scaled energy and force cannot pass unnoticed.

`assert_fixture_coverage` compares the fixture labels with all registered
built-in fragments. Registering a built-in bonded fragment without the
corresponding fixture therefore fails the suite. [Testing Extension
Components](../developer/testing.md) explains the shape-specific geometries,
invariant checks, tolerances, and CPU negative controls.

- In-file `#[cfg(test)]` — pin the exact `entry_point_args` /
  `functor_init_source` strings from `cubic_arg_schema()`; GPU-free.
- `tests/forces_cubic_bond.rs` (model on `tests/forces_harmonic_bond.rs`) —
  a two-particle force field: assert per-atom force/energy/virial against the
  analytic `U(r)` and a finite-difference of the energy, check `F_i = −F_j`,
  and the empty-bond-list no-op.
- `tests/io_config.rs` / `tests/potential_claims.rs` — `kind = "cubic"`
  validates; an unknown kind yields `ConfigError::UnknownKind { slot:
  "bond_types" }`.

## Adapt the pattern for angles and dihedrals

Same recipe; swap the capability trait, claim category, fragment type, scratch
view, and reduction kernel:

| | Bond | Angle | Dihedral |
| --- | --- | --- | --- |
| Capability trait | `BondedPotential` | `AnglePotential` | `DihedralPotential` |
| `JitParticipant` variant | `Bonded` | `Angle` | `Dihedral` |
| Claim category | `BondType` | `AngleType` | `DihedralType` |
| Fragment type | `BondedForceFragment` | `AngleForceFragment` | `DihedralForceFragment` |
| Scratch view | `BondedScratchView` | `AngleScratchView` | `DihedralScratchView` |
| Reduction kernel | `reduce_bond_forces` | `reduce_angle_forces` | `reduce_dihedral_forces` |
| Energy/virial split | ½ per atom (2) | ⅓ per atom (3) | ¼ per atom (4) |
| Template | `harmonic_bond.rs` / `morse.rs` | `angle.rs` | `dihedral.rs` |

The angle functor returns forces on atoms `i` and `k` directly (the central
atom `j` is inferred as `−(F_i + F_k)`); the dihedral functor takes three
bond displacements and returns all four forces. Dihedral types additionally
carry the `scale_lj_14` / `scale_coul_14` common fields, consumed centrally
by the topology loader to derive scaled 1-4 exclusions — a new dihedral kind
gets that automatically.

## Completion checklist

- [ ] **`compute()` runs the reduction, and only the reduction.** It is not
  bypassed like a pair slot. Anything extra double-applies forces.
- [ ] **`max_cutoff()` returns `None`** because a non-`None` value inflates the shared
  neighbor-list cutoff for no reason.
- [ ] **`fmag` is `−(dU/dr)/r`** (pre-divided); **`u_k` / `w_k` are full values**
  (the entry point splits them). These two are the most common mistakes.
- [ ] **Parameter tables are sized to the full `bond_types.len()` and indexed by the
  global type index** — don't compact them to the filtered subset.
- [ ] **An empty bond list is a no-op:** `build` returns `Ok(None)` when no bond uses
  your kind, and `compute` early-returns on `bond_count == 0`.
- [ ] **The `kind` name is unique.** A later same-name registration never shadows a built-in.
- [ ] **A built-in fragment has the matching bond, angle, or dihedral
  consistency fixture.**
