# Adding a pair potential

A pair potential describes a short-range, non-bonded interaction evaluated
for atom pairs in the neighbor list. Buckingham dispersion-repulsion is one
example. Read the [extension overview](index.md) first for the shared Rust,
configuration, unit-conversion, and determinism terminology.

Potentials differ from integrators and thermostats because a force field can
contain several of them at once. HeddleMD therefore asks every registered
`PotentialBuilder` whether the configuration contains interactions it owns.
Each matching builder constructs one runtime component, and the force field
combines all active components.

## Choose the closest example

Use Lennard-Jones in `src/forces/lj.rs` as the primary example. A fast pair
potential is **JIT-composed**: rather than launching its own kernel, it
contributes a fragment of CUDA C++ source that the framework concatenates
with every other active pair potential's fragment into one `nvrtc`-compiled
pair-force kernel. Your job is to (a) write a builder that activates on
`[[pair_interactions]]` entries of your `kind`, and (b) write the per-pair
functor source. JIT composition means that the framework combines active CUDA
source fragments and compiles the resulting kernel at run time.

## Connect configuration to the builder

- **Activation is by presence.** Your builder's `build(cx)` scans
  `cx.pair_interactions` for entries whose `kind` matches yours; if there are
  none it returns `Ok(None)` and contributes nothing. No central `match`, no
  enum — adding a `kind` is entirely open.
- **Parameters are routed by a claim.** The builder returns a
  `PotentialParamsClaim { category: PairInteraction, kind: "buckingham" }`.
  The loader uses it to call your `convert_params` (unit conversion) and
  `validate_params` (domain checks) on each matching entry. An entry whose
  `kind` no builder claims is rejected at load with `ConfigError::UnknownKind`
  — so registering your builder is what turns `"buckingham"` from an error
  into a recognized kind.
- **Common vs typed fields.** `[[pair_interactions]]` entries carry only
  `between`, `kind`, and `cutoff` as centrally-parsed common fields;
  everything else (including `r_switch`) lives in the opaque `params` your
  typed struct deserializes.

## Implement the one-pair calculation

The composition framework lives in `src/forces/jit_composed.rs`. A potential
returns its CUDA C++ fragment as a Rust `String`; the composer inserts it into
a shared CUDA translation unit. There is no separate
`kernels/pair_compute.cuh` file to include. The example below shows the fixed
functor interface; use `src/forces/lj.rs` for the complete Rust-side wiring.

`pair_force_fragment()` returns a `PairForceFragment` with a functor struct
and three `__device__` methods whose signatures are fixed by the composer's
call sites. Mirror LJ exactly:

```cpp
struct BuckinghamPairFunctor {
    unsigned int n_types;
    const Real *type_a; const Real *type_rho; const Real *type_c6;
    const Real *type_cutoff; const Real *type_switch;
    const unsigned int *excl_offsets; const unsigned int *excl_partners;
    const Real *excl_scales;

    __device__ inline unsigned int slot(unsigned int ti, unsigned int tj) const {
        return ti * n_types + tj;
    }
    // factor = -(1/r) dU/dr   (so F_vec = factor * r_vec)
    // energy = U(r) ;  virial = factor * r2
    __device__ inline void evaluate(
        Real r2, Real inv_r, Real r, Real /*qi*/, Real /*qj*/,
        unsigned int i_type, unsigned int j_type,
        unsigned int /*i*/, unsigned int /*j*/,
        Real &factor, Real &energy, Real &virial) const
    {
        unsigned int p = slot(i_type, j_type);
        Real A = type_a[p], rho = type_rho[p], c6 = type_c6[p];
        Real inv_r6 = inv_r*inv_r * inv_r*inv_r * inv_r*inv_r;
        Real e_exp  = A * Real_exp(-r * (R(1.0) / rho));
        factor = e_exp * (R(1.0)/rho) * inv_r - R(6.0) * c6 * inv_r6 * inv_r*inv_r;
        energy = e_exp - c6 * inv_r6;
        virial = factor * r2;
    }
    // only needed for per-pair cutoffs (CutoffHandling::PerPair):
    __device__ inline Real cutoff_squared(unsigned int i_type, unsigned int j_type,
                                           unsigned int, unsigned int) const {
        Real c = type_cutoff[slot(i_type, j_type)]; return c * c;
    }
    __device__ inline Real exclusion_scale(unsigned int i, unsigned int j) const {
        return heddle_jit_exclusion_scale(i, j, excl_offsets, excl_partners, excl_scales);
    }
};
```

The generated kernel expects the following conventions:

- **`factor` is already divided by `r`** — it is `-(1/r)·dU/dr`, because the
  composer multiplies it by the raw displacement components, not the unit
  vector.
- **The composer applies the cutoff mask and the exclusion scale outside
  `evaluate`** — your `exclusion_scale` just does the table lookup; don't
  pre-multiply it into your outputs.
- **Use the precision shim** from `kernels/precision.cuh`: `Real`, `R(x)` for
  literals, and the `Real_*` intrinsics (`Real_exp`, `Real_sqrt`, …) — never
  `float`/`double`/`expf` directly.
- **Namespace everything.** All fragments share one translation unit; give
  the functor struct and any free helpers a slot-unique prefix
  (`BuckinghamPairFunctor`, `heddle_buck_*`), and reuse shared preamble
  helpers like `heddle_jit_exclusion_scale` rather than redefining them.

### Generate the argument interface from one schema

Do not hand-write `entry_point_args` or `functor_init_source`. Define one
`KernelArgSchema` (via `KernelArgSchema::pair_force(...)`) and generate both
from it — `schema.entry_point_args()` and `schema.functor_init_source()` —
and route `bind_pair_force_args` through a `KernelArgBinder` built from the
same schema. That makes the kernel parameter list, the functor-field
initializers, and the launch-time argument binding share one source of truth,
so they cannot drift. The functor struct's field names must equal the
schema's `functor_field` strings; `nvrtc` catches mismatches.

Set `consumes_type_index: true` on the fragment so the composer supplies
`i_type`/`j_type` (the per-atom `type_indices` buffer is a composer common
argument — don't bind it yourself), and `cutoff:` to `CutoffHandling::Uniform`
when every pair shares one cutoff, else `PerPair`.

## Rely on deterministic framework accumulation

You get bit-exact summation for free: the composer converts each per-pair
`(factor, energy, virial)` to integer fixed-point and `atomicAdd`s into
per-atom `u64` accumulators, and integer addition is order-independent. Your
one obligation is that `evaluate` be a **pure per-pair function** — compute
one triple from `r` and the per-type params, with no state carried across
pairs and no float accumulation of your own. All summation belongs to the
framework.

## Implement the Rust types

Create `src/forces/buckingham.rs`. It should follow `lj.rs` and contain the
kind and label constants, typed parameters, resolved state, argument schema,
potential traits, builder, and CUDA fragment. The following code is an
abbreviated outline rather than a directly compilable example.

```rust
pub const BUCKINGHAM_KIND: &str = "buckingham";
const LABEL: &str = "buckingham";

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, crate::units::Convert)]
#[serde(deny_unknown_fields)]
pub struct BuckinghamPairParams {
    pub a: crate::units::Energy,
    pub rho: crate::units::Length,
    pub c6: f64,   // energy·length⁶ has no dimension newtype — see the caveat below
    #[serde(default)] pub r_switch: Option<crate::units::Length>,
}

#[derive(Debug)]
pub struct BuckinghamState { /* device param table + Arc<DeviceExclusionList> clone + cutoffs */ }

impl Potential for BuckinghamState {
    fn label(&self) -> &'static str { LABEL }
    fn max_cutoff(&self) -> Option<Real> { Some(self.max_cutoff) }   // MUST be Some
    // frequency_class() defaults to Fast — leave it.
    fn compute(&mut self, /* … */) -> Result<(), ForceFieldError> {
        unreachable!("JIT-composed pair slot; compute() is never called")
    }
    fn jit_participant(&self) -> Option<JitParticipant<'_>> {
        Some(JitParticipant::PairForce(self))
    }
}

impl PairForcePotential for BuckinghamState {
    fn pair_force_fragment(&self) -> PairForceFragment { /* build from arg_schema() */ }
    fn bind_pair_force_args(&self, _ctx: &PairForceBindContext<'_>, builder: &mut ForceLaunchBuilder) {
        // KernelArgBinder::new(&buck_arg_schema(), LABEL, builder); push param + exclusion buffers
    }
}

#[derive(Debug, Clone)]
pub struct BuckinghamBuilder;
impl PotentialBuilder for BuckinghamBuilder {
    fn build(&self, cx: &PotentialBuildContext<'_>) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        let pairs: Vec<_> = cx.pair_interactions.iter().filter_map(resolve_buckingham_pair).collect();
        if pairs.is_empty() { return Ok(None); }               // activation
        Ok(Some(Box::new(/* BuckinghamState::from_config(...) */)))
    }
    fn params_claim(&self) -> Option<PotentialParamsClaim> {
        Some(PotentialParamsClaim {
            category: PotentialParamsCategory::PairInteraction, kind: BUCKINGHAM_KIND })
    }
    fn validate_params(&self, entry: PotentialConfigEntry<'_>) -> Result<(), ConfigError> {
        let PotentialConfigEntry::PairInteraction(p) = entry else { unreachable!() };
        // A/ρ/C₆ finite, ρ > 0, cross-field r_switch <= p.cutoff …
        Ok(())
    }
    fn convert_params(&self, units: crate::units::UnitSystem, params: &mut toml::Value)
        -> Result<(), ConfigError> {
        crate::registry::convert_params_in_place::<BuckinghamPairParams>(units, params)
    }
}
```

The per-type parameter table is SoA (one `CudaSlice<Real>` per field,
row-major `ti*n_types+tj`, uploaded with `htod_or_empty`) — copy
`LennardJonesParameterTable` in `src/gpu/kernels.rs`, or keep it local to
`buckingham.rs`.

## Register and document the potential

Add `rqm/forces/buckingham-pair-force.md`; follow
`rqm/forces/lj-pair-force.md` and cross-reference
`rqm/forces/jit-composed-pair-force.md` and
`rqm/forces/packed-neighbour-pair-force.md` rather than restating the shared
kernel contract.

Then edit the following existing files:

- **`src/forces/mod.rs`** — three lines: `pub mod buckingham;`, the
  `pub use buckingham::{...}` re-export, and `Box::new(BuckinghamBuilder)` in
  the `vec![...]` of `impl Builtins for dyn PotentialBuilder` (roster at
  ~line 386). Registration order is the slot evaluation order; since your
  `kind` is unique in its category, position only affects the slot index —
  next to `LennardJonesBuilder` is natural.

- **`src/gpu/kernels.rs`** / **`src/gpu/mod.rs`** — only if you add the param
  table there rather than locally.

**No change needed** to `src/io/config.rs` (`PairInteractionConfig` is
open-shaped and routing is claim-driven), `build.rs`, or any `.cu` file (the
functor is a runtime-compiled string; there is no new kernel).

## Test the potential

Add a `ConsistencyFixture::pair(...)` entry to
`builtin_consistency_fixtures()` in `tests/common/consistency.rs`. Use the
potential's fragment label, register only the builder under test, and provide
sample distances in the physically important and switched regions. Include
at least one analytical `ReferencePoint`: finite-difference consistency alone
cannot detect an energy and force that are both wrong by the same factor.

The shared harness checks force–energy, Newton, virial, reference-point, and
cutoff invariants through the real GPU force pipeline.
`assert_fixture_coverage` compares fixture labels with the built-in fragment
labels, so a built-in pair fragment without a fixture fails the suite. See
[Testing Extension Components](../developer/testing.md) for the fixture API,
sample selection, tolerance guidance, and negative controls.

- In-file `#[cfg(test)]` — pin the exact `entry_point_args` /
  `functor_init_source` strings the schema emits, and assert the fragment
  declares `consumes_type_index` and uses `slot(i_type, j_type)`. These run
  without a GPU.
- `tests/potential_claims.rs` — add the builder's claim to the
  claim-coverage test and a build/validate/convert round-trip.
- `tests/forces_framework.rs` — a Buckingham-only force field that steps and
  writes non-zero forces.
- `tests/io_config.rs` — a `kind = "buckingham"` parse test.

## Completion checklist

- [ ] **`compute()` is bypassed for JIT pair slots** — make it `unreachable!()`
  and put all physics in the fragment.
- [ ] **`max_cutoff()` returns `Some`** because it feeds the shared neighbor-list cutoff
  and the JIT prune constant; `None` drops your interaction or panics.
- [ ] **The shared neighbor and exclusion lists are used** — clone
  `cx.device_exclusions` (an `Arc`); never allocate your own
  `DeviceExclusionList`. Bind only your per-type param buffers and the
  exclusion buffers.
- [ ] **The units of every parameter are documented.** `A` (Energy) and `ρ` (Length) convert cleanly, but
  `C₆` carries energy·length⁶, which has no dimension newtype. Either keep it
  a plain `f64` (a `Convert` no-op) and document that it must be supplied in
  atomic units regardless of the file's `units` selector, or add a new
  `Dimension` variant and newtype in `src/units/mod.rs`.
- [ ] **Tests use the implemented fixed-point scale of `2^48`** (some spec prose says `2^32`);
  the code is authoritative, but you never touch this directly.
- [ ] **A built-in fragment has a consistency fixture with analytical
  reference data.**
