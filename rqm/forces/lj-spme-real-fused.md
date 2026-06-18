# Feature: Fused LJ + SPME Real-Space Pair-Force Slot <!-- rq-79282483 -->

The fused LJ + SPME real-space slot is a composite short-range
pairwise potential that walks the shared `NeighborListState` once and
evaluates both the Lennard-Jones 12-6 and the SPME real-space Coulomb
contribution per pair visit. It is the canonical short-range slot
whenever both `[[pair_interactions]]` and `[spme]` are configured.
Its existence is governed by the displacement mechanism in
`framework.md`: when both constituents would otherwise have built,
the fused slot's `displaces()` claim against `"lennard_jones"` and
`"spme_real"` suppresses their standalone slots and the fused slot
runs in their place. When only one constituent is configured, the
fused slot returns `Ok(None)` from `build` and the lone constituent's
standalone slot continues to run.

The fused kernel reuses the warp-per-particle topology, sweep loop,
warp-tree butterfly reduction, and read-modify-write per-particle
write specified in `pair-force-kernel.md`. Only the per-pair
functional form differs: at each surviving pair the lane accumulates
the sum of an LJ-12-6 contribution and an SPME real-space erfc
contribution into the same register pair, with each contribution
gated independently by its own cutoff and scaled independently by
its own per-pair exclusion factor.

## Algorithm <!-- rq-a242e878 -->

For lane `lane` of the warp handling particle `i` at sweep step `s`,
when `k = s * 32 + lane` satisfies `k < neighbor_counts[i]` and
`j = neighbor_list[i * max_neighbors + k]` is not equal to `i`:

1. Load positions for `i` and `j`, apply the triclinic minimum image
   to `(dx, dy, dz)`, compute `r2 = dx*dx + dy*dy + dz*dz`. Compute
   `r = sqrt(r2)`. Set
   - `r2_cut_lj = r_cut_lj * r_cut_lj`
   - `r2_cut_spme = r_cut_spme_real * r_cut_spme_real`

2. Initialise per-pair accumulators
   `factor = 0`, `energy = 0`, `virial = 0`.

3. If `r2 <= r2_cut_lj`, compute the LJ contribution exactly as
   specified in `lj-pair-force.md` (per-type sigma / epsilon lookup,
   CHARMM C¹ switching over `[r_switch, r_cut_lj]`, exclusion-scale
   lookup against the LJ exclusion table). Add the resulting
   `(factor_lj * scale_lj, energy_lj * scale_lj * 0.5,
   virial_lj * scale_lj * 0.5)` to `(factor, energy, virial)`.

4. If `r2 <= r2_cut_spme`, compute the SPME real-space contribution
   exactly as specified in `spme.md` (charge product `q_i * q_j`,
   `erfc(alpha * r) / r` term, derivative-of-erfc tail). The
   exclusion-scale lookup uses the Coulomb exclusion table, which is
   carried independently of the LJ exclusion table. Add the resulting
   `(factor_spme * scale_coul, energy_spme * scale_coul * 0.5,
   virial_spme * scale_coul * 0.5)` to `(factor, energy, virial)`.

5. Multiply `(factor)` by `(dx, dy, dz)` to obtain the per-pair
   `(fx, fy, fz)` and add to the lane's per-component register
   accumulators. Add `energy` and `virial` to the lane's per-scalar
   accumulators. The `0.5` symmetry factor on `energy` and `virial`
   is applied per-contribution at the lane (steps 3 and 4) so that
   summing over every particle recovers each unordered pair's
   contribution exactly once.

The warp tree, lane-0 write, and read-modify-write add to the Fast
class accumulator (`fast_total_forces_x/y/z`,
`fast_total_potential_energies`, `fast_total_virials`) follow the
shared pattern in `pair-force-kernel.md` unchanged.

When a pair falls inside `r2_cut_spme` but outside `r2_cut_lj` (the
common case when `r_cut_spme_real > r_cut_lj`), only the SPME
contribution is added — the LJ contribution stays at zero for that
lane on that sweep iteration. When a pair falls outside both
cutoffs, neither contribution is added and the lane is functionally
equivalent to an inactive slot.

## max_cutoff <!-- rq-790e0c6c -->

`LjSpmeRealFusedState::max_cutoff()` returns
`Some(max(r_cut_lj, r_cut_spme_real))`. The shared neighbour list is
therefore sized to the wider of the two cutoffs, and every neighbour
within the wider cutoff is visited even when only the SPME
contribution survives that pair's gate. This trades a slightly wider
neighbour-list walk for a single kernel pass.

## Determinism <!-- rq-5b048e60 -->

The fused kernel uses the same lane-strided sweep and 5-step
butterfly tree as every other pair-force kernel; the order of every
floating-point add depends only on `count = neighbor_counts[i]`,
not on thread scheduling. Two runs of a fused-configuration
`ForceField` on the same GPU produce byte-identical per-particle
results.

Cross-configuration equality is *not* a property: a fused-slot run
and a separately-launched LJ-then-SPME-real run agree only within
f32 round-off, because the fused kernel sums LJ + SPME-real
contributions into a single per-lane register before the warp tree,
while the standalone configuration sums per-class accumulator
entries that already contain LJ's per-particle total when SPME-real
adds in. The framework-level scope of this divergence is captured
in `framework.md`'s *Determinism Guarantees*.

## Feature API <!-- rq-29e21686 -->

### Types <!-- rq-aa7fd772 -->

- `LjSpmeRealFusedBuilder` — implements `PotentialBuilder`. <!-- rq-c2a28bda -->
  - `build(cx)` returns `Ok(Some(LjSpmeRealFusedState))` iff every
    one of the following is true; otherwise `Ok(None)`:
    - `!cx.pair_interactions.is_empty()` (LJ activation condition).
    - `cx.spme_config.is_some()` (SPME activation condition).
    The builder never returns `Err(_)` from its activation check;
    GPU allocation failures during state construction surface as
    `Err(ForceFieldError::Gpu(_))`.
  - `displaces()` returns `&["lennard_jones", "spme_real"]`.

- `LjSpmeRealFusedState` — implements `Potential` with <!-- rq-0bda299c -->
  - `label() == "lj_spme_real_fused"`.
  - `frequency_class() == ForceClass::Fast` (the trait default).
  - `max_cutoff() == Some(max(r_cut_lj, r_cut_spme_real) as Real)`.

  Owns:
  - The Lennard-Jones per-pair-type parameter table
    (`LennardJonesParameterTable`, see `lj-pair-force.md`), including
    per-type `sigma`, `epsilon`, `r_cut_lj`, `r_switch_lj`.
  - The SPME real-space parameters (`alpha`, `r_cut_spme_real`,
    per-particle `charges` device array).
  - The two independent `DeviceExclusionList`s (one for LJ, one for
    Coulomb) carried over from the constituent slots' state. They
    are constructed from the same `topology.md` exclusion entries
    using the same builders the standalone slots use; the fused slot
    does not collapse them into a single combined table.
  - A reference to the shared `NeighborListState` (held through
    `ForceFieldContext` at evaluation time; no clone).

  `compute()` launches the appropriate kernel variant
  (`lj_spme_real_fused_pair_force_f` for
  `AggregateLevel::ForcesOnly`,
  `lj_spme_real_fused_pair_force_fev` for
  `AggregateLevel::ForcesAndScalars`) on the default stream, after
  emitting a `KernelStage::LjSpmeRealFusedPairForce` start event and
  before emitting the matching stop event.

### Functions <!-- rq-9938e4d6 -->

- `gpu::lj_spme_real_fused_pair_force_f(...)` — Rust launcher around <!-- rq-343f9e75 -->
  the forces-only kernel. Parameters are the union of
  `lj_pair_force_f` and `spme_real_pair_force_f` inputs (positions,
  neighbour list + counts, both per-type tables, both exclusion
  tables, charges, alpha, lattice, slot output force slices,
  particle count, max-neighbors). Launch configuration matches the
  shared pair-force pattern: `WARPS_PER_BLOCK = 8`, block size 256,
  grid `ceil(n / 8)`.

- `gpu::lj_spme_real_fused_pair_force_fev(...)` — Rust launcher <!-- rq-6ea9e020 -->
  around the forces + energy + virial kernel. Parameters extend the
  `_f` variant with the per-particle energy and virial slot output
  slices.

Both launchers return `Result<(), GpuError>`; a zero
`particle_count` is a successful no-op.

### Kernels <!-- rq-de5f642d -->

`kernels/lj_spme_real_fused.cu` declares two `extern "C"` kernels:

```c
extern "C" __global__ void lj_spme_real_fused_pair_force_f(
    /* positions, neighbour list, per-type LJ tables,
       per-particle charges, SPME alpha,
       LJ exclusion table, Coulomb exclusion table,
       slot force slices, particle count, max_neighbors */);

extern "C" __global__ void lj_spme_real_fused_pair_force_fev(
    /* same as _f, plus slot energy and virial slices */);
```

Both implementations call the `pair_compute_{f,fev}` template helper
from `pair-force-kernel.md`, supplying a per-pair functor whose
body matches the algorithm above. The fused functor reads both
exclusion tables independently and applies each scale to its own
contribution before summing.

### Timings stage <!-- rq-bb874433 -->

`KernelStage::LJ_SPME_REAL_FUSED_PAIR_FORCE` is the timings tag for
both kernel variants. It appears in the canonical timings order
(see `framework.md`'s *Force Evaluation Pipeline*) at the position
where `LJ_PAIR_FORCE` and `SPME_REAL_PAIR_FORCE` would have appeared
when those two constituents are standalone — that is, between
`NEIGHBOR_LIST_BUILD` and `COULOMB_PAIR_FORCE`. The original
`LJ_PAIR_FORCE` and `SPME_REAL_PAIR_FORCE` stages do not appear in
the timings output of a fused-configuration run.

## Gherkin Scenarios <!-- rq-7f96174d -->

```gherkin
Feature: Fused LJ + SPME real-space pair-force slot

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called

  # --- Construction and activation ---

  @rq-dec77ff5
  Scenario: Composite builder activates when both LJ and SPME are configured
    Given a config with one [[pair_interactions]] entry for ("Ar","Ar")
    And an [spme] table with alpha = 0.3 and r_cut_real = 1.2
    And per-particle charges with at least one non-zero value
    And PotentialRegistry::with_builtins()
    When ForceField::new(...) is called
    Then it returns Ok(force_field)
    And force_field.slots contains exactly one slot with label() == "lj_spme_real_fused"
    And no slot in force_field.slots has label() in {"lennard_jones", "spme_real"}

  @rq-9dfa7b39
  Scenario: Composite builder reports max_cutoff = max(r_cut_lj, r_cut_spme_real)
    Given r_cut_lj = 1.0 and r_cut_spme_real = 1.2
    When the composite slot is built and its max_cutoff() is queried
    Then max_cutoff() == Some(1.2)

  @rq-6a16d777
  Scenario: Composite builder declines when only LJ is configured
    Given a config with one [[pair_interactions]] entry and no [spme] table
    And PotentialRegistry::with_builtins()
    When ForceField::new(...) is called
    Then force_field.slots contains exactly one slot with label() == "lennard_jones"
    And force_field.slots contains no slot with label() == "lj_spme_real_fused"

  @rq-58e9adff
  Scenario: Composite builder declines when only SPME is configured
    Given a config with no [[pair_interactions]] entries and an [spme] table
    And PotentialRegistry::with_builtins()
    When ForceField::new(...) is called
    Then force_field.slots contains exactly one slot with label() == "spme_real"
    And force_field.slots contains no slot with label() == "lj_spme_real_fused"

  @rq-9468f67a
  Scenario: Composite frequency_class is Fast
    Given a built LjSpmeRealFusedState
    Then state.frequency_class() == ForceClass::Fast

  # --- Per-pair correctness ---

  @rq-2351322d
  Scenario: Pair inside both cutoffs receives sum of LJ and SPME-real contributions
    Given two particles at separation r = 0.5 * min(r_cut_lj, r_cut_spme_real)
      with non-zero charges and an LJ pair-type defined for their species
    And no exclusions between them
    When the fused _fev kernel runs
    Then forces[0] equals f_lj + f_spme_real computed independently
      within a relative tolerance of 1e-5
    And energy share for particle 0 equals 0.5 * (U_lj + U_spme_real)
      within the same tolerance
    And virial share for particle 0 equals 0.5 * (W_lj + W_spme_real)
      within the same tolerance

  @rq-c0e835ce
  Scenario: Pair inside SPME cutoff but outside LJ cutoff contributes SPME only
    Given r_cut_lj = 1.0 and r_cut_spme_real = 1.2
    And two particles at separation r = 1.1
    When the fused _fev kernel runs
    Then the per-particle force equals the standalone SPME real-space contribution
    And the LJ contribution is zero within bit-exact equality

  @rq-bce6480f
  Scenario: Pair outside both cutoffs contributes nothing
    Given r_cut_lj = 1.0 and r_cut_spme_real = 1.2
    And two particles at separation r = 1.5
    When the fused _fev kernel runs
    Then the per-particle force, energy, and virial contributions are all zero

  @rq-28b7ed1d
  Scenario: LJ exclusion scales the LJ contribution but not the SPME contribution
    Given two particles whose LJ exclusion scale is 0.0 and Coulomb exclusion scale is 1.0
    And both inside both cutoffs
    When the fused _fev kernel runs
    Then the per-particle force equals the standalone SPME real-space contribution
      within bit-exact equality of how SPME-real would compute it solo

  @rq-b264fa75
  Scenario: Coulomb exclusion scales the SPME contribution but not the LJ contribution
    Given two particles whose LJ exclusion scale is 1.0 and Coulomb exclusion scale is 0.0
    And both inside both cutoffs
    When the fused _fev kernel runs
    Then the per-particle force equals the standalone LJ contribution
      within bit-exact equality of how LJ would compute it solo

  @rq-ecf4def9
  Scenario: Newton's third law per pair
    Given two particles inside both cutoffs with no exclusions
    When the fused _fev kernel runs
    Then forces[0].x + forces[1].x == 0.0 within bit-exact equality
    And forces[0].y + forces[1].y == 0.0
    And forces[0].z + forces[1].z == 0.0

  # --- AggregateLevel ---

  @rq-1fe9fd90
  Scenario: _f variant adds to force slices only
    Given a ForceField with the fused slot active and seeded
      fast_total_potential_energies and fast_total_virials
    When force_field.step(..., AggregateLevel::ForcesOnly) is called
    Then fast_total_forces_x/y/z are refreshed
    And fast_total_potential_energies is byte-identical to its seed
    And fast_total_virials is byte-identical to its seed

  @rq-082659a0
  Scenario: _fev variant adds to all five slices
    Given a ForceField with the fused slot active
    When force_field.step(..., AggregateLevel::ForcesAndScalars) is called
    Then fast_total_forces_x/y/z, fast_total_potential_energies, and
      fast_total_virials all reflect the fused slot's contribution

  # --- Reproducibility ---

  @rq-93609d86
  Scenario: Two runs of the fused kernel are byte-identical
    Given two independently-built fused-configuration ForceFields with
      byte-identical inputs
    When force_field.step(...) is called on each
    Then the downloaded forces_x, forces_y, forces_z, potential_energies,
      and virials agree byte-for-byte across the two runs

  @rq-4255cb6e
  Scenario: Fused kernel and standalone kernels agree within f32 round-off
    Given the same physical configuration evaluated two ways:
      (a) PotentialRegistry::with_builtins() (composite active)
      (b) a custom registry containing only LennardJonesBuilder and
          SpmeRealBuilder (no composite registered)
    When force_field.step(...) runs on each
    Then the per-particle forces agree within a relative tolerance of 1e-4
    But the per-particle forces are NOT byte-identical across (a) and (b)

  # --- Timings ---

  @rq-a12e8fa8
  Scenario: Fused configuration's timings output omits the standalone stages
    Given a fused-configuration ForceField driven through one step
    When the timings report is rendered
    Then the report contains a row for "lj_spme_real_fused_pair_force"
    And the report contains no row for "lj_pair_force"
    And the report contains no row for "spme_real_pair_force"

  @rq-2557e549
  Scenario: Standalone-LJ-only configuration emits the LJ stage as before
    Given a config with [[pair_interactions]] but no [spme]
    When the simulation is run for at least one step
    Then the timings report contains a row for "lj_pair_force"
    And the report contains no row for "lj_spme_real_fused_pair_force"
```

## Out of Scope <!-- rq-d58ba7f6 -->

- Fusion with truncated Coulomb (`coulomb.cu`). The composite covers
  LJ + SPME-real only; LJ + truncated-Coulomb continues to run as two
  standalone slots.
- A configuration-driven opt-out of the fused composite when both
  constituents are configured. Activation is automatic — the cost of
  the composite is strictly an improvement at this point and there
  is no scenario where the user would want to opt out.
- Fusion of the SPME reciprocal-space pipeline into the same kernel.
  The reciprocal stages (charge spread, R2C FFT, influence-multiply,
  C2R FFT, force gather) are bound to a different parallelism shape
  (one thread per grid cell, not one warp per particle) and stay
  Slow-class.
- A user-supplied per-pair functional form. Custom potentials
  continue to register their own `PotentialBuilder`s; the composite
  is hard-coded to the LJ-12-6 + SPME real-space pair.
