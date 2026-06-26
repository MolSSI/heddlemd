# Feature: Truncated Coulomb Pair Force <!-- rq-846bdb8b -->

Truncated Coulomb is the non-bonded electrostatic pairwise potential slot
in the pluggable potential framework (`framework.md`). The slot is present
when the config declares a `[coulomb]` table. Its parameters are a single
real-space cutoff and a single inner-switching radius; pair magnitudes are
constructed at kernel time from the per-particle charges carried by
`ParticleBuffers` (see `particle-state.md`).

The slot contributes its per-pair functional form to the JIT-composed
pair-force pipeline as a `PairForceFragment` (see
`jit-composed-pair-force.md`). The composed kernel runs the
packed-neighbour data model and force-accumulation pattern specified
in `packed-neighbour-pair-force.md`: the inner loop evaluates only
real neighbours (no per-pair cutoff masking against the neighbour
list itself), and per-particle contributions accumulate via
`atomicAdd` on the class's fixed-point force buffer. The slot does
not launch its own per-potential kernel.

This file specifies the truncated-Coulomb functional form
(`U(r) = k_e · q_i · q_j / r` with a smooth switching function
between `r_switch` and `r_cut`), the parameter inputs the slot
exposes, and the per-pair `evaluate(r², i, j) -> (factor, energy,
virial)` contract its fragment implements. The kernel topology,
launch configuration, neighbour-list contract, and reproducibility
mechanism are specified in `packed-neighbour-pair-force.md` and not
restated here.

Work is O(N · average neighbour count) when the shared list is in
cell-list mode and O(N²) when it is in trivial mode (every particle's
list contains every particle). The kernel is the same in both cases;
only the contents and size of the shared list differ.

This slot is the smallest viable charge-carrying milestone in the
project's electrostatics roadmap. It is intentionally short-range only:
real-space truncation of `1/r` is known to introduce artefacts for
charged systems larger than a few nanometres. The SPME slot pair (see
`spme.md`) supplies a smooth particle-mesh Ewald path for longer-range
electrostatics; this slot and the SPME real-space slot are mutually
exclusive.

This file specifies `CoulombParameters` (the host-side parameter
container) and the Coulomb pair functor `CoulombPairFunctor` that the
slot composes into the JIT pair-force kernel as a `PairForceFragment`
(see `jit-composed-pair-force.md`).

Each pair contribution is multiplied by a CHARMM-style C¹ switching
function `S(r²)` over the interval `[r_switch, cutoff]` so that both
energy and force go smoothly to zero at the cutoff. Below `r_switch`
the switching factor is exactly `1` and the functor evaluates the
unmodified Coulomb force and energy. `r_switch = cutoff` selects the
hard-cutoff degenerate case in which no smoothing is applied; below
the cutoff the unmodified Coulomb force and energy apply and at the
cutoff a discontinuity remains.

## Algorithm <!-- rq-bfd7004c -->

The pair potential and force between particles `i` and `j` at separation
`r = |r_i - r_j|` are

```text
U_ij(r)  = k_C · q_i · q_j / r
F_ij(r)  = -dU/dr along r_hat = k_C · q_i · q_j · r_ij / r³
```

where `r_ij = r_i - r_j`, `q_i` and `q_j` are the per-particle charges
carried in `posq.w`, and `k_C = 1 / (4 π ε₀)` is the Coulomb constant,
stored as the compile-time `f32` constant `K_COULOMB_F32` (equal to `1`
exactly in the engine's atomic units).

The Coulomb functor is composed into the JIT packed pair-force kernel
`heddle_jit_composed_pair_force_{f,fev}` (see
`jit-composed-pair-force.md`) and evaluated over the packed neighbour
list specified in `packed-neighbour-pair-force.md`. The composed
kernel's outer loop walks each particle's packed neighbour list,
applies the triclinic minimum-image convention (using the six lattice
parameters `(lx, ly, lz, xy, xz, yz)` and the fractional-coordinate
wrap algorithm defined in `simulation-box.md`; for an orthorhombic box
this reduces to three independent per-axis wraps), computes the per-pair
scalars `(r², inv_r, r, q_i, q_j)` once, and threads them into the
functor's `evaluate` for every interacting `(i, j)` pair the list
yields, where `j` is a neighbour atom distinct from `i`. Here
`inv_r = rsqrtf(r²)`, `r = r² · inv_r`, `q_i = posq[i].w`, and
`q_j = posq[j].w`.

The Coulomb-specific contribution is computed as follows.

1. Form the unswitched Coulomb factor and energy in this order:

   ```text
   inv_r2  = inv_r * inv_r
   qq      = q_i * q_j
   energy  = k_C * qq * inv_r
   factor  = k_C * qq * inv_r * inv_r2          // F = factor · r_ij
   ```

   `factor * r_ij` is the Coulomb force on particle `i` due to `j`.
2. Apply the CHARMM-style C¹ switching function `S(r²)` defined over
   `[r_switch², cutoff²]`. Let `r_s2 = r_switch * r_switch` and
   `r_c2 = cutoff * cutoff`. The polynomial is evaluated in normalised
   form so the only place the cutoff-width factor appears explicitly
   is `1/delta` where `delta = r_c2 - r_s2`.

   - If `r2 <= r_s2`, the inner plateau has `S = 1` and `dS/d(r²) = 0`,
     so `factor` and `energy` are unchanged.
   - Otherwise `r_s2 < r2` (the `r2 > r_c2` case is masked out by the
     functor's `CutoffHandling::Uniform(cutoff)`; see *Cutoff gating*
     below) and the polynomial branch runs.

     With `tau = (r² − r_s2) / delta` the polynomial reduces to
     `S = (1 − tau)² (1 + 2 tau)` and
     `dS/d(r²) = −6 tau (1 − tau) / delta`. Apply the switching factor
     as a multiplication on `energy` and on `factor`; the chain-rule
     correction adds `chain_coeff · energy` to `factor`, where
     `chain_coeff = 12 · tau · (1 − tau) / delta = −2 · dS/d(r²)`
     (matches the identical correction in `lj-pair-force.md`).

   When `r_switch = cutoff` the switching interval is degenerate and
   the polynomial branch is never entered; the functor uses the
   unchanged `factor` and `energy` for any `r2 ≤ r_c2`, producing the
   hard-cutoff case.
3. Set the scalar virial contribution from this pair as
   `virial = factor * r²` (which equals `r_ij · F_ij`).
4. The composer applies the per-pair Coulomb exclusion scale (see
   `topology.md`) returned by the functor's `exclusion_scale(i, j)`,
   which walks the per-atom exclusion slice
   `atom_excl_partners[atom_excl_offsets[i] .. atom_excl_offsets[i+1]]`
   looking for `j`; if present, the corresponding entry in
   `atom_excl_coul_scales` multiplies `factor`, `energy`, and `virial`.
   A scale of `0.0` fully excludes the pair; a scale in `(0.0, 1.0)`
   partially attenuates it. Pairs not listed in the exclusion table
   are evaluated with scale `1.0` (no attenuation).
5. The composed kernel adds `(factor · dx, factor · dy, factor · dz)` to
   particle `i`'s per-particle force accumulation. The `_fev` variant
   additionally adds `energy · 0.5` to the per-particle potential-energy
   share and `virial · 0.5` to the scalar-virial share. The `0.5`
   factor distributes each unordered pair's energy and virial across the
   two ordered contributions `(i, j)` and `(j, i)`.

### Cutoff gating <!-- inline --> <!-- rq-402e2963 -->

The functor reports `cutoff: CutoffHandling::Uniform(cutoff)` to the
composer; the composer applies a single `r² <= cutoff²` guard (or
relies on the outer max-cutoff mask described in
`jit-composed-pair-force.md`) so the functor contributes only for pairs
within the cutoff. `evaluate` itself is safe to call at any positive
`r²` because `inv_r` and `r` are well-defined for all positive `r²`.

The per-particle accumulation order, the determinism mechanism, and the
packed neighbour-list contract are specified in
`packed-neighbour-pair-force.md` and not restated here.

## Reproducibility <!-- rq-03423870 -->

The per-particle output for particle `i` is the deterministic
fixed-point accumulation of its per-pair contributions in the
packed-neighbour pair-force kernel (see
`packed-neighbour-pair-force.md`). Each contribution is a deterministic
function of `q_i`, `q_j`, the wrapped displacement, the six lattice
parameters, the cutoff and switching parameters, and the per-pair
exclusion scale. Identical runs on the same GPU with identical inputs
produce byte-identical `slot_force_*` outputs.

## Newton's third law <!-- rq-e858417a -->

The contributions to particles `i` and `j` from the pair `{i, j}` are
computed independently by the composed kernel. The displacements differ
only in sign, the wrap formula respects sign symmetry for displacements
not exactly at the primary-image boundary, and the Coulomb factor
depends only on `r²` and on the product `q_i · q_j` (commutative). The
exclusion lookup table is constructed so the `(i, j)` and `(j, i)`
scales are equal (see `topology.md`). Particle `i`'s net force from
the pair `{i, j}` and particle `j`'s net force from the same pair
therefore differ by a sign bit-for-bit for all displacements except
the measure-zero exact-boundary case documented in
`lj-pair-force.md`.

## Parameters <!-- rq-f9a9c569 -->

The slot owns a single `CoulombParameters` record (host-side) carrying
the cutoff and the inner switching radius, both in Bohr (`a_0`). The
functor evaluates the bare Coulomb pair force `F = q_i · q_j · r̂ / r²`
with the configured switching window applied; the prefactor
`1 / (4πε₀) = 1` exactly in the engine's atomic units and is therefore
absent from every expression.

- `cutoff: f32` — finite, strictly positive.
- `r_switch: f32` — finite, strictly positive, and `r_switch <= cutoff`.
  When `r_switch == cutoff` the switching interval is degenerate and no
  smoothing is applied (the functor produces the bare Coulomb force and
  energy for `r ≤ cutoff` and zero for `r > cutoff`).

Per-particle charges live on `ParticleBuffers` (see `particle-state.md`)
and are not duplicated here. The slot does not own a per-pair-type
parameter table; `q_i · q_j` is computed on the fly from the per-particle
charges.

`k_C` (the Coulomb constant) is the compile-time `f32` constant
`K_COULOMB_F32`, equal to `1` exactly in the engine's atomic units. It
is bound into the functor by the slot's `bind_pair_force_args`, never
overridden from config.

## Empty state <!-- rq-5ad9b8b9 -->

When no particle carries a non-zero charge but a `[coulomb]` table is
present, the slot is still constructed and the composed kernel evaluates
the Coulomb functor over every pair within the cutoff; the pair
contributions are all zero (since `q_i · q_j == 0`). The slot's
per-particle reduced outputs are all zero. This avoids a host-side scan
of the charge array.

When `particle_count == 0`, the composed pair-force kernel does not
launch and the slot's output buffers are length zero.

When no `[coulomb]` table is supplied in the config, `CoulombBuilder`
returns `None`, the slot is absent from the `ForceField`, and it
contributes nothing.

## Feature API <!-- rq-4968452d -->

### Types <!-- rq-a3fdbd7c -->

- `CoulombParameters` — host-side parameter container. Fields: <!-- rq-6bdfdd6d -->
  - `cutoff: f32`
  - `r_switch: f32`

  Constructed from a `&CoulombConfig` via `From`.

- `CoulombState` — implements the `Potential` trait with <!-- rq-d340b338 -->
  `label() == "coulomb"` (see `framework.md`). Fields:
  - `device: Arc<CudaDevice>`
  - `params: CoulombParameters`
  - `exclusions: DeviceExclusionList`
  - `particle_count: usize`

  All fields private. The slot launches no kernel of its own:
  `jit_participant` returns `Some(JitParticipant::PairForce(self))`, so
  the framework composes the Coulomb functor into the JIT packed
  pair-force kernel and skips this slot in the per-slot `compute` loop
  (the `Potential::compute` method is `unreachable!`). `max_cutoff()`
  returns `Some(params.cutoff)` so the shared neighbour list sizes its
  search radius accordingly.

  Constructor:

  - `CoulombState::new(gpu: &GpuContext, particle_count: usize, params: CoulombParameters, exclusion_list: &ExclusionList) -> Result<CoulombState, NeighborListError>` — uploads the host `ExclusionList` to a `DeviceExclusionList` and stores the parameters.

### Force evaluation <!-- inline --> <!-- rq-af7718be -->

The Coulomb functor is composed into the JIT-composed packed pair-force
kernel `heddle_jit_composed_pair_force_{f,fev}` (see
`jit-composed-pair-force.md`), which evaluates every active fast-class
pair-force slot in a single pass over the packed neighbour list
(`packed-neighbour-pair-force.md`). The `_f` variant writes the
per-particle net force; the `_fev` variant additionally writes the
potential-energy and scalar-virial shares. The minimum-image
displacement, cutoff test, CHARMM-style C¹ switching over
`[r_switch, cutoff]`, exclusion-list scaling, and the fixed-point
per-particle accumulation are as defined in the *Algorithm* section
above and in `packed-neighbour-pair-force.md`. The packed neighbour list
is owned by the shared `NeighborListState` on `ForceField` (see
`neighbor-list.md` and `framework.md`) and kept current by the
framework's `pre_step` before each force evaluation.

- `CoulombState::pair_force_fragment(&self) -> PairForceFragment` — <!-- rq-537cbe5e -->
  returns the Coulomb `PairForceFragment` for the composer. The fragment
  carries:
  - `functor_struct_name: "CoulombPairFunctor"` and the CUDA functor
    source. The functor's evaluation entry point is
    `evaluate(Real r2, Real inv_r, Real r, Real qi, Real qj,
    unsigned int i, unsigned int j, Real &factor, Real &energy,
    Real &virial)`; it computes `inv_r2 = inv_r · inv_r` and
    `qq = qi · qj` from the composer-supplied scalars (it does not
    recompute `1.0 / r2`).
  - `cutoff: CutoffHandling::Uniform(cutoff)` — the composer applies a
    single `r² <= cutoff²` guard (or relies on the outer max-cutoff
    mask) for this fragment.
  - `entry_point_args` and `functor_init_source` generated from
    `coulomb_arg_schema()`, the single source of truth that
    `bind_pair_force_args` is also validated against, so the functor
    fields (`k_coulomb`, `cutoff`, `r_switch`, `excl_offsets`,
    `excl_partners`, `excl_scales`) cannot drift from the binding.

- `CoulombBuilder` — `PotentialBuilder` that constructs the slot when <!-- rq-13424eab -->
  `cx.coulomb_config.is_some()` and returns `None` otherwise (see
  `framework.md`).

### Exclusion scaling <!-- inline --> <!-- rq-66bcfbe3 -->

The functor's `exclusion_scale(i, j)` calls the shared device helper
`heddle_jit_exclusion_scale(i, j, excl_offsets, excl_partners,
excl_scales)` (see `topology.md`), returning the matching Coulomb scale
when `j` appears in atom `i`'s exclusion-partner range and `1.0`
otherwise (including when the range is empty). The composer multiplies
`factor`, `energy`, and `virial` by this scale, so the unscaled Coulomb
contribution flows through unchanged for pairs that are not on the
exclusion list. The `DeviceExclusionList` (`atom_excl_offsets`,
`atom_excl_partners`, `atom_excl_coul_scales`) is built once from the
host-side `ExclusionList` at force-field construction; the Coulomb slot
consumes the `atom_excl_coul_scales` array.

## Out of Scope <!-- rq-bee8e566 -->

- Long-range Ewald or smooth particle-mesh Ewald reciprocal-space
  contribution.
- Charge-neutrality checks at config-load time. Truncated Coulomb is
  documented to misbehave on non-neutral systems; the user is
  responsible.
- Reaction-field or other dielectric-medium correction terms beyond
  the cutoff.
- Per-particle charges supplied from the init-state file; charges come
  only from the per-type config field (see `io/config-schema.md`).
- Tabulated or splined Coulomb forms.
- Polarisable models (Drude, fluctuating charges).
- Tinfoil boundary conditions or other implicit-solvent treatments.

---

## Gherkin Scenarios <!-- rq-5f7aa9ac -->

```gherkin
Feature: Truncated Coulomb pair force kernel

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called
    And an orthorhombic SimulationBox with lx=ly=lz=10.0e-9
    And the Coulomb constant k_C = 8.987551787e9 (rounded to f32)
    And a CoulombParameters with cutoff=5.0e-9 and r_switch=5.0e-9
      (hard cutoff, no smoothing) unless otherwise specified

  # --- Basic two-particle force ---

  @rq-4f23c656
  Scenario: Opposite-sign charges attract
    Given two particles with charges (+1e, -1e) at positions
      (0, 0, 0) and (3e-10, 0, 0)
    When the Coulomb pair functor is evaluated in the composed pair-force kernel
    Then slot_force_x[0] equals the closed-form Coulomb force
      F_x = k_C · (+1e) · (-1e) · (-3e-10) / (3e-10)³
        = -k_C · e² · 3e-10 / (3e-10)³
      which is negative (atom 0 is pulled toward atom 1 along +x)

  @rq-82e4f74e
  Scenario: Same-sign charges repel
    Given two particles with charges (+1e, +1e) at positions
      (0, 0, 0) and (3e-10, 0, 0)
    When the Coulomb pair functor is evaluated in the composed pair-force kernel
    Then slot_force_x[0] is positive (atom 0 is pushed away from atom 1)
    And slot_force_x[1] equals -slot_force_x[0] (Newton's third law)

  @rq-3b7da473
  Scenario: Zero charges produce zero force
    Given two neutral particles at positions (0, 0, 0) and (3e-10, 0, 0)
    When the Coulomb pair functor is evaluated in the composed pair-force kernel
    Then slot_force_x[0], slot_force_y[0], slot_force_z[0], slot_energy[0], slot_virial[0] all equal 0.0

  @rq-02a15197
  Scenario: Mixed charge magnitudes scale linearly
    Given two particles with charges (q_i, q_j) at distance r
    When the Coulomb pair functor is evaluated in the composed pair-force kernel
    Then slot_force_x[0] is proportional to q_i · q_j

  # --- Cutoff gating ---

  @rq-0896c33a
  Scenario: Pair beyond cutoff contributes zero
    Given a CoulombParameters with cutoff=2.0e-10
    And two particles with charges (+1e, +1e) at separation 5e-10 (> cutoff)
    When the Coulomb pair functor is evaluated in the composed pair-force kernel
    Then slot_force_x[0], slot_energy[0], and slot_virial[0] equal 0.0

  @rq-3df3ed61
  Scenario: Pair at exactly the cutoff contributes the smoothed (zero) value
    Given a CoulombParameters with cutoff=3.0e-10 and r_switch=2.7e-10
    And two particles with charges (+1e, +1e) at separation exactly 3.0e-10
    When the Coulomb pair functor is evaluated in the composed pair-force kernel
    Then slot_energy[0] equals 0.0 (the switching function S(r_c²) = 0)
    And slot_force_x[0], slot_force_y[0], slot_force_z[0] are all 0.0
    And slot_virial[0] equals 0.0

  # --- Switching function ---

  @rq-b07abbd4
  Scenario: Pair inside the inner plateau is unsmoothed
    Given a CoulombParameters with cutoff=5e-10 and r_switch=4e-10
    And two particles with charges (+1e, +1e) at separation 3.5e-10 (< r_switch)
    When the Coulomb pair functor is evaluated in the composed pair-force kernel
    Then slot_energy[0] + slot_energy[1] equals the unswitched Coulomb energy
      k_C · (1e)² / 3.5e-10 within f32 round-off
    And slot_force_x[0] equals the unswitched Coulomb force component
      along +x: k_C · (1e)² · 3.5e-10 / (3.5e-10)³

  @rq-d52bcc88
  Scenario: Pair inside the switching interval is smoothed
    Given a CoulombParameters with cutoff=5e-10 and r_switch=4e-10
    And two particles with charges (+1e, +1e) at separation 4.5e-10
      (between r_switch and cutoff)
    When the Coulomb pair functor is evaluated in the composed pair-force kernel
    Then slot_energy[0] is strictly between 0 and the unswitched 0.5 · k_C · (1e)² / 4.5e-10
    And slot_force_x[0] is strictly between 0 and the unswitched x-component

  @rq-67678030
  Scenario: Switching interval r_switch == cutoff selects hard-cutoff
    Given a CoulombParameters with cutoff = r_switch = 4e-10
    And two particles with charges (+1e, +1e) at separation 3.9e-10
    When the Coulomb pair functor is evaluated in the composed pair-force kernel
    Then slot_energy[0] + slot_energy[1] equals the unswitched Coulomb energy (S = 1)

  @rq-25500ae7
  Scenario: Coulomb force is C¹ continuous at r_switch
    Given a CoulombParameters with cutoff=5 a₀ and r_switch=4 a₀
    And two particles with charges (+1 e, -1 e)
    When the Coulomb pair functor is evaluated in the composed pair-force kernel at r = 4 a₀ - 1e-3 a₀ to obtain f_below
    And it is evaluated at r = 4 a₀ + 1e-3 a₀ to obtain f_above
    Then |f_below.slot_force_x[0] - f_above.slot_force_x[0]|
      is bounded by 5e-2 * |f_below.slot_force_x[0]|
      (the bound is looser than the 1% used for Lennard-Jones because the
      Coulomb chain-rule correction term `chain_coeff * energy` carries
      f32-precision noise near τ = 0 that is large relative to the small
      step in r)

  @rq-e1204b7f
  Scenario: Coulomb force decays toward zero just inside r_cut
    Given a CoulombParameters with cutoff=5 a₀ and r_switch=4 a₀
    And two particles with charges (+1 e, -1 e)
    When the Coulomb pair functor is evaluated in the composed pair-force kernel at r = 5 a₀ - 1e-5 a₀ to obtain f_inside
    Then |f_inside.slot_force_x[0]| is bounded by 1e-2 * |unswitched Coulomb force at r = r_switch|
      (the assertion uses r very close to r_cut because the
      CHARMM-style chain-rule correction `12 τ (1-τ)/Δ · U(r)` remains a
      few percent of the unswitched force at r_switch until 1 - τ ≪ 1)

  @rq-c95dd6c9
  Scenario: Exclusion scaling multiplies the switched force, energy, and virial
    Given a CoulombParameters with cutoff=5e-10 and r_switch=4e-10
    And two particles with charges (+1e, -1e) at separation 4.5e-10 (in the switching window)
    And an ExclusionList listing pair (0, 1) with scale_coul = 0.5
    When the Coulomb pair functor is evaluated in the composed pair-force kernel to obtain (f_excluded, E_excluded, W_excluded)
    And it is evaluated with the same setup but an empty exclusion list to obtain (f_unscaled, E_unscaled, W_unscaled)
    Then f_excluded.slot_force_x[0] equals 0.5 * f_unscaled.slot_force_x[0] bit-for-bit
    And f_excluded.slot_energy[0] equals 0.5 * f_unscaled.slot_energy[0] bit-for-bit
    And f_excluded.slot_virial[0] equals 0.5 * f_unscaled.slot_virial[0] bit-for-bit

  # --- PBC ---

  @rq-ef3083cd
  Scenario: Pair across the periodic boundary uses minimum-image displacement
    Given two particles with charges (+1e, -1e) at positions
      (-lx/2 + 0.1e-10, 0, 0) and (+lx/2 - 0.1e-10, 0, 0)
      (which are 0.2e-10 apart across the periodic boundary)
    And a CoulombParameters with cutoff = 5e-10
    When the Coulomb pair functor is evaluated in the composed pair-force kernel
    Then the closed-form force uses minimum-image dx = +0.2e-10, not lx - 0.2e-10
    And slot_force_x[0] is negative (attraction across the +x face)

  @rq-9af1f9dc
  Scenario: Minimum-image works for a triclinic box
    Given a SimulationBox with non-zero tilts (e.g. xy=2e-10, xz=1e-10, yz=-3e-10)
    And two particles at positions whose minimum-image separation crosses a
      tilted face of the primary parallelepiped
    When the Coulomb pair functor is evaluated in the composed pair-force kernel
    Then the computed displacement matches the triclinic minimum-image
      result of sim_box.minimum_image(r_i - r_j)

  # --- Exclusions ---

  @rq-c4d4608f
  Scenario: Pair with Coulomb exclusion scale 0.0 contributes nothing
    Given two particles with charges (+1e, -1e) at separation 3e-10 (inside cutoff)
    And an ExclusionList listing pair (0, 1) with scale_coul = 0.0
    When the Coulomb pair functor is evaluated in the composed pair-force kernel
    Then slot_force_x[0], slot_force_y[0], slot_force_z[0] are all 0.0
    And slot_energy[0] equals 0.0
    And slot_virial[0] equals 0.0

  @rq-d26e9f9c
  Scenario: Pair with Coulomb exclusion scale 0.5 contributes half
    Given two particles with charges (+1e, -1e) at separation 3e-10
    And an ExclusionList listing pair (0, 1) with scale_coul = 0.5
    When the Coulomb pair functor is evaluated in the composed pair-force kernel to obtain values_scaled
    And it is evaluated with an empty exclusion list to obtain values_unscaled
    Then values_scaled.slot_energy[0] equals 0.5 * values_unscaled.slot_energy[0] bit-for-bit
    And values_scaled.slot_force_x[0] equals 0.5 * values_unscaled.slot_force_x[0] bit-for-bit
    And values_scaled.slot_virial[0] equals 0.5 * values_unscaled.slot_virial[0] bit-for-bit

  @rq-f1dffc21
  Scenario: Pair with Coulomb exclusion scale 1.0 reproduces the un-excluded value
    Given two particles with charges (+1e, -1e) at separation 3e-10
    And an ExclusionList listing pair (0, 1) with scale_coul = 1.0
    When the Coulomb pair functor is evaluated in the composed pair-force kernel to obtain values_explicit
    And it is evaluated with an empty exclusion list to obtain values_implicit
    Then values_explicit.slot_force_x[0] equals values_implicit.slot_force_x[0] bit-for-bit
    And values_explicit.slot_energy[0] equals values_implicit.slot_energy[0] bit-for-bit
    And values_explicit.slot_virial[0] equals values_implicit.slot_virial[0] bit-for-bit

  @rq-59ea69fb
  Scenario: An exclusion entry on one pair does not attenuate other pairs
    Given a ParticleState of N=3 with positions p0=(0,0,0), p1=(2e-10,0,0), p2=(4e-10,0,0) and charges (+1e, -1e, +1e)
    And an ExclusionList listing only pair (0, 1) with scale_coul = 0.0
    When the Coulomb pair functor is evaluated in the composed pair-force kernel
    Then slot_force_x[0] equals the Coulomb force on particle 0 due to particle 2 only
      (the (0, 1) contribution is suppressed; the (0, 2) contribution is unscaled)
    And slot_force_x[2] equals the Coulomb force on particle 2 due to particles 0 and 1
      (no exclusion entry attenuates particle 2's contributions)

  @rq-8c96d3c7
  Scenario: Coulomb and LJ exclusions are independent
    Given two particles whose ExclusionList entry is scale_lj=0.5, scale_coul=0.833
    When the Coulomb pair functor is evaluated in the composed pair-force kernel
    Then the Coulomb contribution is scaled by 0.833
    When the Lennard-Jones pair functor is evaluated in the composed pair-force kernel on the same pair
    Then the LJ contribution is scaled by 0.5

  # --- Energy and virial conventions ---

  @rq-5444c7ae
  Scenario: Per-particle energy slot accumulates half the pair's potential energy
    Given a single pair (0, 1) at finite distance with no exclusion
    When the Coulomb pair functor is evaluated in the composed pair-force kernel
    Then slot_energy[0] equals 0.5 * (k_C * q_0 * q_1 / r) * S(r²)
    And slot_energy[0] + slot_energy[1] equals (k_C * q_0 * q_1 / r) * S(r²) within f32 round-off
      (half-sum convention)

  @rq-e412e54a
  Scenario: Per-particle virial slot accumulates half the pair's scalar virial
    Given a single pair (0, 1) at finite distance with no exclusion
    When the Coulomb pair functor is evaluated in the composed pair-force kernel
    Then slot_virial[0] equals 0.5 * factor * r²
      where factor is the post-switching radial scalar force prefactor
    And slot_virial[0] + slot_virial[1] equals factor * r² within f32 round-off

  # --- Reproducibility ---

  @rq-1a0f3eef
  Scenario: Identical inputs produce byte-identical slot output
    Given two evaluations of the Coulomb pair functor in the composed pair-force kernel with identical inputs on the same GPU
    When both runs complete
    Then their slot_force_x, slot_force_y, slot_force_z, slot_energy, and slot_virial slot outputs
      are byte-identical

  # --- Empty state ---

  @rq-76a6be2f
  Scenario: Zero particles is a no-op
    Given particle_count == 0
    When the composed pair-force kernel with the Coulomb functor active is evaluated
    Then the composed kernel is not launched and the slot's outputs are length zero

  @rq-ee4ebbda
  Scenario: A neutral system with [coulomb] present still launches and produces zeros
    Given every particle has charge 0.0 and a [coulomb] table is present
    When the Coulomb pair functor is evaluated in the composed pair-force kernel
    Then every slot output entry equals 0.0
    And the slot's reduced per-particle outputs equal 0.0

  # --- Newton's third law (bit-exact for non-boundary displacements) ---

  @rq-f652bf7c
  Scenario: Forces on the two members of a non-boundary pair are equal and opposite
    Given two particles whose minimum-image displacement is not at the primary-image boundary
    When the Coulomb pair functor is evaluated in the composed pair-force kernel
    Then slot_force_x[i] equals -slot_force_x[j] bit-exact (for an isolated N=2 pair)
    And similarly for y and z
```
