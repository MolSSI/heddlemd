# Feature: Truncated Coulomb Pair Force Kernel <!-- rq-846bdb8b -->

Truncated Coulomb is the non-bonded electrostatic pairwise potential slot
in the pluggable potential framework (`framework.md`). The slot is present
when the config declares a `[coulomb]` table. Its parameters are a single
real-space cutoff and a single inner-switching radius; pair magnitudes are
constructed at kernel time from the per-particle charges carried by
`ParticleBuffers` (see `particle-state.md`).

The slot evaluates pair forces with two fused warp-per-particle kernels
(forces only and forces + energy + virial) that read the shared
`NeighborListState` owned by `ForceField` (see `neighbor-list.md`). Each
warp handles one particle: the warp walks the particle's neighbour list,
accumulates the per-pair Coulomb contribution in register accumulators
across all 32 lanes, and adds the per-particle net force into its
class accumulator through a warp-tree butterfly reduction followed by
a lane-0 read-modify-write add (see `framework.md`'s *Class Output
Accumulators*).
The common kernel pattern is specified in `pair-force-kernel.md`; this
file specifies the truncated-Coulomb functional form, parameter inputs,
and launcher.

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
container), the two `coulomb_pair_force_*` CUDA kernels in
`kernels/coulomb.cu`, and the Rust launch helper that drives them.

Each pair contribution is multiplied by a CHARMM-style C¹ switching
function `S(r²)` over the interval `[r_switch, cutoff]` so that both
energy and force go smoothly to zero at the cutoff. Below `r_switch`
the switching factor is exactly `1` and the kernel evaluates the
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

where `r_ij = r_i - r_j`, `q_i` and `q_j` are the per-particle charges in
coulombs, and `k_C = 1 / (4 π ε₀) ≈ 8.987 551 787 × 10⁹ N·m²/C²` is the
Coulomb constant. The kernel stores `k_C` as an `f32` constant captured at
kernel launch.

The kernel topology, sweep loop, warp-tree reduction, exclusion-scale
lookup, and per-particle output write all follow the common
warp-per-particle pattern specified in `pair-force-kernel.md`. The
Coulomb-specific contribution at each `(i, k)` pair is computed as
follows.

For lane `lane` of the warp handling particle `i` at sweep step `s`,
when `k = s * 32 + lane` satisfies `k < neighbor_counts[i]` and
`j = neighbor_list[i * max_neighbors + k]` is not equal to `i`:

1. Look up `q_i = charges[i]` and `q_j = charges[j]`.
2. Compute the displacement `(dx, dy, dz) = positions[i] - positions[j]`
   and apply the triclinic minimum-image convention using the six lattice
   parameters `(lx, ly, lz, xy, xz, yz)` and the fractional-coordinate
   wrap algorithm defined in `simulation-box.md`. For an orthorhombic
   box (all tilts zero) this reduces to three independent per-axis wraps.
3. Compute `r2 = dx*dx + dy*dy + dz*dz`. If `r2 > cutoff * cutoff`, the
   pair contributes nothing; the lane skips to its next assigned
   neighbour.
4. Compute the unswitched Coulomb factor and energy in this order:

   ```text
   inv_r2  = 1.0f / r2
   inv_r   = sqrtf(inv_r2)
   qq      = q_i * q_j
   energy  = k_C * qq * inv_r
   factor  = k_C * qq * inv_r * inv_r2          // F = factor · r_ij
   ```

   `factor * r_ij` is the Coulomb force on particle `i` due to `j`.
5. Apply the CHARMM-style C¹ switching function `S(r²)` defined over
   `[r_switch², cutoff²]`. Let `r_s2 = r_switch * r_switch` and
   `r_c2 = cutoff * cutoff`. The polynomial is evaluated in normalised
   form so the only place the cutoff-width factor appears explicitly
   is `1/delta` where `delta = r_c2 - r_s2`.

   - If `r2 <= r_s2`, the inner plateau has `S = 1` and `dS/d(r²) = 0`,
     so `factor` and `energy` are unchanged.
   - Otherwise `r_s2 < r2 <= r_c2` (the `r2 > r_c2` case was gated by
     step 3) and the polynomial branch runs.

     With `tau = (r² − r_s2) / delta` the polynomial reduces to
     `S = (1 − tau)² (1 + 2 tau)` and
     `dS/d(r²) = −6 tau (1 − tau) / delta`.

     Apply the switching factor as a multiplication on `energy` and on
     `factor`. The chain-rule correction adds `2 · energy · dS_dr2` to
     `factor` (matches the identical correction in `lj-pair-force.md`).

   When `r_switch = cutoff` the switching interval is degenerate and
   the polynomial branch is never entered; the kernel uses the
   unchanged `factor` and `energy` for any `r2 ≤ r_c2`, producing the
   hard-cutoff case.
6. Apply the per-pair Coulomb exclusion scale (see `topology.md`). The
   kernel walks the per-atom exclusion slice
   `atom_excl_partners[atom_excl_offsets[i] .. atom_excl_offsets[i+1]]`
   looking for `j`; if present, the corresponding entry in
   `atom_excl_coul_scales` multiplies both `factor` and `energy`. A
   scale of `0.0` fully excludes the pair; a scale in `(0.0, 1.0)`
   partially attenuates it. Pairs not listed in the exclusion table
   are evaluated with scale `1.0` (no attenuation).
7. Compute the scalar virial contribution from this pair as
   `w = factor * r²` (which equals `r_ij · F_ij`). Apply the exclusion
   scale to `w` along with `factor` and `energy`.
8. Add `(factor * dx, factor * dy, factor * dz)` to the lane's
   `(p_x, p_y, p_z)` register accumulators. The `_fev` variant
   additionally adds `energy * 0.5f` to `p_e` and `w * 0.5f` to `p_w`.
   The `0.5` factor distributes each unordered pair's energy and virial
   across the two ordered contributions `(i, j)` and `(j, i)`.

After every lane has processed every assigned neighbour, the warp-tree
butterfly reduction collapses the 32 lane accumulators to lane 0, which
writes the particle's net force (and, in the `_fev` variant,
energy/virial) into the corresponding slot output rows. See
`pair-force-kernel.md` for the topology and reduction details.

Slots beyond `neighbor_counts[i]` are not visited: the warp's sweep
loop bounds on `count`, so unpopulated neighbour-list entries do not
contribute to the per-particle accumulators.

## Reproducibility <!-- rq-03423870 -->

The per-particle output for particle `i` is the deterministic warp-tree
sum of its per-pair contributions, accumulated in a fixed lane-strided
order (see `pair-force-kernel.md`). Each contribution is a deterministic
function of `q_i`, `q_j`, the wrapped displacement, the six lattice
parameters, the cutoff and switching parameters, and the per-pair
exclusion scale. Identical runs on the same GPU with identical inputs
produce byte-identical `slot_force_*` outputs.

## Newton's third law <!-- rq-e858417a -->

Warps `W_i` and `W_j` independently compute the per-pair Coulomb
contribution to particles `i` and `j`. The displacements differ only
in sign, the wrap formula respects sign symmetry for displacements not
exactly at the primary-image boundary, and the Coulomb factor depends
only on `r²` and on the product `q_i · q_j` (commutative). The
exclusion lookup table is constructed so the `(i, j)` and `(j, i)`
scales are equal (see `topology.md`). Particle `i`'s net force from
the pair `{i, j}` and particle `j`'s net force from the same pair
therefore differ by a sign bit-for-bit for all displacements except
the measure-zero exact-boundary case documented in
`lj-pair-force.md`.

## Parameters <!-- rq-f9a9c569 -->

The slot owns a single `CoulombParameters` record (host-side) carrying
the cutoff and the inner switching radius, both in Bohr (`a_0`). The
kernel evaluates the bare Coulomb pair force `F = q_i · q_j · r̂ / r²`
with the configured switching window applied; the prefactor
`1 / (4πε₀) = 1` exactly in the engine's atomic units and is therefore
absent from every expression.

- `cutoff: f32` — finite, strictly positive.
- `r_switch: f32` — finite, strictly positive, and `r_switch <= cutoff`.
  When `r_switch == cutoff` the switching interval is degenerate and no
  smoothing is applied (the kernel writes the bare Coulomb force and
  energy for `r ≤ cutoff` and zero for `r > cutoff`).

Per-particle charges live on `ParticleBuffers` (see `particle-state.md`)
and are not duplicated here. The slot does not own a per-pair-type
parameter table; `q_i · q_j` is computed on the fly from the per-particle
charges.

`k_C` (the Coulomb constant) is a compile-time `f32` constant equal to
the rounded f32 representation of `1 / (4 π ε₀) ≈ 8.987 551 787 × 10⁹`
N·m²/C². The kernel reads it as an argument to keep the parameter
surface explicit at the launch site; the constant is supplied by the
launch helper, never overridden from config.

## Empty state <!-- rq-5ad9b8b9 -->

When no particle carries a non-zero charge but a `[coulomb]` table is
present, the slot is still constructed and the kernel runs over every
pair within the cutoff; the pair contributions are all zero (since
`q_i · q_j == 0`). The slot's per-particle reduced outputs are
all zero. This avoids a host-side scan of the charge array.

When `particle_count == 0`, the kernel does not launch and the slot's
output buffers are length zero.

When no `[coulomb]` table is supplied in the config, the slot is absent
from the `ForceField` and contributes nothing.

## Feature API <!-- rq-4968452d -->

### Types <!-- rq-a3fdbd7c -->

- `CoulombParameters` — host-side parameter container. Fields: <!-- rq-6bdfdd6d -->
  - `cutoff: f32`
  - `r_switch: f32`

- `CoulombState` — implements the `Potential` trait with <!-- rq-d340b338 -->
  `label() == "coulomb"` (see `framework.md`). Fields:
  - `device: Arc<CudaDevice>`
  - `kernels: Arc<Kernels>`
  - `params: CoulombParameters`
  - `particle_count: usize`

  All fields private; the slot's public surface is `Potential::compute`,
  invoked by `ForceField::step` (see `framework.md`). The slot does not
  own a per-pair intermediate buffer; the fused kernels accumulate in
  registers and write per-particle output directly to the slot's
  SlotOutputView.

  Constructor:

  - `CoulombState::new(device: Arc<CudaDevice>, kernels: Arc<Kernels>, params: CoulombParameters, particle_count: usize) -> Result<CoulombState, GpuError>`

  The `Potential` implementation reports
  `max_cutoff() = Some(params.cutoff)` so the shared neighbor list
  sizes its search radius accordingly.

### Functions <!-- rq-7bae4b69 -->

- `coulomb_pair_force(particle_buffers: &ParticleBuffers, output: &mut SlotOutputView<'_>, sim_box: &SimulationBox, params: &CoulombParameters, atom_excl_offsets: &CudaSlice<u32>, atom_excl_partners: &CudaSlice<u32>, atom_excl_coul_scales: &CudaSlice<f32>, neighbor_list: &CudaSlice<u32>, neighbor_counts: &CudaSlice<u32>, max_neighbors: u32, level: AggregateLevel) -> Result<(), GpuError>` <!-- rq-38676211 -->
  - Selects the kernel variant based on `level`:
    `AggregateLevel::ForcesOnly` dispatches to `coulomb_pair_force_f`
    and writes only the three force-component slot output rows;
    `AggregateLevel::ForcesAndScalars` dispatches to
    `coulomb_pair_force_fev` and additionally writes the energy and
    virial slot output rows.
  - Launches with the warp-per-particle geometry documented in
    `pair-force-kernel.md`: `block_dim = (256, 1, 1)`,
    `grid_dim = (ceil(n / 8), 1, 1)`.
  - When `particle_buffers.particle_count() == 0`, returns `Ok(())`
    without launching.
  - Returns the underlying `GpuError` if the kernel launch fails.

### CUDA kernel <!-- rq-4a96b705 -->

`kernels/coulomb.cu` declares two `extern "C"` kernels (forces-only
and forces + energy + virial) that share the warp-per-particle pattern
documented in `pair-force-kernel.md`:

```c
extern "C" __global__ void coulomb_pair_force_f(
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    const float *charges,
    unsigned int max_neighbors,
    const float *lattice,           // length 6: [lx, ly, lz, xy, xz, yz]
    float k_coulomb,
    float cutoff,
    float r_switch,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const float *atom_excl_coul_scales,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    float *slot_force_x,
    float *slot_force_y,
    float *slot_force_z,
    unsigned int n);

extern "C" __global__ void coulomb_pair_force_fev(
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    const float *charges,
    unsigned int max_neighbors,
    const float *lattice,           // length 6: [lx, ly, lz, xy, xz, yz]
    float k_coulomb,
    float cutoff,
    float r_switch,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const float *atom_excl_coul_scales,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    float *slot_force_x,
    float *slot_force_y,
    float *slot_force_z,
    float *slot_energy,
    float *slot_virial,
    unsigned int n);
```

The launcher trusts the caller for shape consistency: in debug builds
it asserts `output.force_x.len() == particle_buffers.particle_count()`
(and similarly for the other slot output rows it writes),
`neighbor_list.len() == particle_buffers.particle_count() * max_neighbors as usize`,
`neighbor_counts.len() == particle_buffers.particle_count()`,
`atom_excl_offsets.len() == particle_buffers.particle_count() + 1`,
`atom_excl_partners.len() == atom_excl_coul_scales.len()`, and
`charges.len() == particle_buffers.particle_count()`. Release builds
skip the asserts for parity with the other kernel launchers. The
launcher does not validate the parameter values themselves and does
not check `cutoff` against `sim_box.min_perpendicular_width() / 2`
(that gating happens in the neighbor list; see `forces/neighbor-list.md`).

## Launch Configuration <!-- rq-84af056c -->

- Block size: 256 threads (8 warps × 32 lanes) for both variants.
- Grid size: `ceil(n / 8)` blocks in the x dimension.
- Shared memory: zero bytes.
- Stream: the default stream carried by `ParticleBuffers.device`.

Both numbers match the common pair-force pattern (`pair-force-kernel.md`).

This matches the LJ kernel's launch configuration (`lj-pair-force.md`).

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
    When the _f variant of coulomb_pair_force is called
    Then slot_force_x[0] equals the closed-form Coulomb force
      F_x = k_C · (+1e) · (-1e) · (-3e-10) / (3e-10)³
        = -k_C · e² · 3e-10 / (3e-10)³
      which is negative (atom 0 is pulled toward atom 1 along +x)

  @rq-82e4f74e
  Scenario: Same-sign charges repel
    Given two particles with charges (+1e, +1e) at positions
      (0, 0, 0) and (3e-10, 0, 0)
    When the _f variant of coulomb_pair_force is called
    Then slot_force_x[0] is positive (atom 0 is pushed away from atom 1)
    And slot_force_x[1] equals -slot_force_x[0] (Newton's third law)

  @rq-3b7da473
  Scenario: Zero charges produce zero force
    Given two neutral particles at positions (0, 0, 0) and (3e-10, 0, 0)
    When the _f variant of coulomb_pair_force is called
    Then slot_force_x[0], slot_force_y[0], slot_force_z[0], slot_energy[0], slot_virial[0] all equal 0.0

  @rq-02a15197
  Scenario: Mixed charge magnitudes scale linearly
    Given two particles with charges (q_i, q_j) at distance r
    When the _f variant of coulomb_pair_force is called
    Then slot_force_x[0] is proportional to q_i · q_j

  # --- Cutoff gating ---

  @rq-0896c33a
  Scenario: Pair beyond cutoff contributes zero
    Given a CoulombParameters with cutoff=2.0e-10
    And two particles with charges (+1e, +1e) at separation 5e-10 (> cutoff)
    When the _f variant of coulomb_pair_force is called
    Then slot_force_x[0], slot_energy[0], and slot_virial[0] equal 0.0

  @rq-3df3ed61
  Scenario: Pair at exactly the cutoff contributes the smoothed (zero) value
    Given a CoulombParameters with cutoff=3.0e-10 and r_switch=2.7e-10
    And two particles with charges (+1e, +1e) at separation exactly 3.0e-10
    When the _fev variant of coulomb_pair_force is called
    Then slot_energy[0] equals 0.0 (the switching function S(r_c²) = 0)
    And slot_force_x[0], slot_force_y[0], slot_force_z[0] are all 0.0
    And slot_virial[0] equals 0.0

  # --- Switching function ---

  @rq-b07abbd4
  Scenario: Pair inside the inner plateau is unsmoothed
    Given a CoulombParameters with cutoff=5e-10 and r_switch=4e-10
    And two particles with charges (+1e, +1e) at separation 3.5e-10 (< r_switch)
    When the _fev variant of coulomb_pair_force is called
    Then slot_energy[0] + slot_energy[1] equals the unswitched Coulomb energy
      k_C · (1e)² / 3.5e-10 within f32 round-off
    And slot_force_x[0] equals the unswitched Coulomb force component
      along +x: k_C · (1e)² · 3.5e-10 / (3.5e-10)³

  @rq-d52bcc88
  Scenario: Pair inside the switching interval is smoothed
    Given a CoulombParameters with cutoff=5e-10 and r_switch=4e-10
    And two particles with charges (+1e, +1e) at separation 4.5e-10
      (between r_switch and cutoff)
    When the _fev variant of coulomb_pair_force is called
    Then slot_energy[0] is strictly between 0 and the unswitched 0.5 · k_C · (1e)² / 4.5e-10
    And slot_force_x[0] is strictly between 0 and the unswitched x-component

  @rq-67678030
  Scenario: Switching interval r_switch == cutoff selects hard-cutoff
    Given a CoulombParameters with cutoff = r_switch = 4e-10
    And two particles with charges (+1e, +1e) at separation 3.9e-10
    When the _fev variant of coulomb_pair_force is called
    Then slot_energy[0] + slot_energy[1] equals the unswitched Coulomb energy (S = 1)

  @rq-25500ae7
  Scenario: Coulomb force is C¹ continuous at r_switch
    Given a CoulombParameters with cutoff=5 a₀ and r_switch=4 a₀
    And two particles with charges (+1 e, -1 e)
    When the _f variant of coulomb_pair_force is called at r = 4 a₀ - 1e-3 a₀ to obtain f_below
    And the _f variant is called at r = 4 a₀ + 1e-3 a₀ to obtain f_above
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
    When the _f variant of coulomb_pair_force is called at r = 5 a₀ - 1e-5 a₀ to obtain f_inside
    Then |f_inside.slot_force_x[0]| is bounded by 1e-2 * |unswitched Coulomb force at r = r_switch|
      (the assertion uses r very close to r_cut because the
      CHARMM-style chain-rule correction `12 τ (1-τ)/Δ · U(r)` remains a
      few percent of the unswitched force at r_switch until 1 - τ ≪ 1)

  @rq-c95dd6c9
  Scenario: Exclusion scaling multiplies the switched force, energy, and virial
    Given a CoulombParameters with cutoff=5e-10 and r_switch=4e-10
    And two particles with charges (+1e, -1e) at separation 4.5e-10 (in the switching window)
    And an ExclusionList listing pair (0, 1) with scale_coul = 0.5
    When the _fev variant of coulomb_pair_force is called to obtain (f_excluded, E_excluded, W_excluded)
    And the _fev variant is called with the same setup but an empty exclusion list to obtain (f_unscaled, E_unscaled, W_unscaled)
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
    When the _f variant of coulomb_pair_force is called
    Then the closed-form force uses minimum-image dx = +0.2e-10, not lx - 0.2e-10
    And slot_force_x[0] is negative (attraction across the +x face)

  @rq-9af1f9dc
  Scenario: Minimum-image works for a triclinic box
    Given a SimulationBox with non-zero tilts (e.g. xy=2e-10, xz=1e-10, yz=-3e-10)
    And two particles at positions whose minimum-image separation crosses a
      tilted face of the primary parallelepiped
    When the _f variant of coulomb_pair_force is called
    Then the computed displacement matches the triclinic minimum-image
      result of sim_box.minimum_image(r_i - r_j)

  # --- Self-pair handling ---

  @rq-bf7dfc6d
  Scenario: Self pair in trivial-mode neighbour list contributes nothing
    Given a 1-particle system with charge +1e in trivial mode (neighbor_list[0] = 0)
    When the _fev variant of coulomb_pair_force is called
    Then slot_force_x[0] equals 0.0 (i == j short-circuit)
    And slot_energy[0] equals 0.0
    And slot_virial[0] equals 0.0

  # --- Exclusions ---

  @rq-c4d4608f
  Scenario: Pair with Coulomb exclusion scale 0.0 contributes nothing
    Given two particles with charges (+1e, -1e) at separation 3e-10 (inside cutoff)
    And an ExclusionList listing pair (0, 1) with scale_coul = 0.0
    When the _fev variant of coulomb_pair_force is called
    Then slot_force_x[0], slot_force_y[0], slot_force_z[0] are all 0.0
    And slot_energy[0] equals 0.0
    And slot_virial[0] equals 0.0

  @rq-d26e9f9c
  Scenario: Pair with Coulomb exclusion scale 0.5 contributes half
    Given two particles with charges (+1e, -1e) at separation 3e-10
    And an ExclusionList listing pair (0, 1) with scale_coul = 0.5
    When the _fev variant of coulomb_pair_force is called to obtain values_scaled
    And the _fev variant of coulomb_pair_force is called with an empty exclusion list to obtain values_unscaled
    Then values_scaled.slot_energy[0] equals 0.5 * values_unscaled.slot_energy[0] bit-for-bit
    And values_scaled.slot_force_x[0] equals 0.5 * values_unscaled.slot_force_x[0] bit-for-bit
    And values_scaled.slot_virial[0] equals 0.5 * values_unscaled.slot_virial[0] bit-for-bit

  @rq-f1dffc21
  Scenario: Pair with Coulomb exclusion scale 1.0 reproduces the un-excluded value
    Given two particles with charges (+1e, -1e) at separation 3e-10
    And an ExclusionList listing pair (0, 1) with scale_coul = 1.0
    When the _fev variant of coulomb_pair_force is called to obtain values_explicit
    And the _fev variant of coulomb_pair_force is called with an empty exclusion list to obtain values_implicit
    Then values_explicit.slot_force_x[0] equals values_implicit.slot_force_x[0] bit-for-bit
    And values_explicit.slot_energy[0] equals values_implicit.slot_energy[0] bit-for-bit
    And values_explicit.slot_virial[0] equals values_implicit.slot_virial[0] bit-for-bit

  @rq-59ea69fb
  Scenario: An exclusion entry on one pair does not attenuate other pairs
    Given a ParticleState of N=3 with positions p0=(0,0,0), p1=(2e-10,0,0), p2=(4e-10,0,0) and charges (+1e, -1e, +1e)
    And an ExclusionList listing only pair (0, 1) with scale_coul = 0.0
    When the _fev variant of coulomb_pair_force is called
    Then slot_force_x[0] equals the Coulomb force on particle 0 due to particle 2 only
      (the (0, 1) contribution is suppressed; the (0, 2) contribution is unscaled)
    And slot_force_x[2] equals the Coulomb force on particle 2 due to particles 0 and 1
      (no exclusion entry attenuates particle 2's contributions)

  @rq-8c96d3c7
  Scenario: Coulomb and LJ exclusions are independent
    Given two particles whose ExclusionList entry is scale_lj=0.5, scale_coul=0.833
    When the _fev variant of coulomb_pair_force is called
    Then the Coulomb contribution is scaled by 0.833
    When the _fev variant of lj_pair_force is called on the same pair
    Then the LJ contribution is scaled by 0.5

  # --- Energy and virial conventions ---

  @rq-5444c7ae
  Scenario: Per-particle energy slot accumulates half the pair's potential energy
    Given a single pair (0, 1) at finite distance with no exclusion
    When the _fev variant of coulomb_pair_force is called
    Then slot_energy[0] equals 0.5 * (k_C * q_0 * q_1 / r) * S(r²)
    And slot_energy[0] + slot_energy[1] equals (k_C * q_0 * q_1 / r) * S(r²) within f32 round-off
      (half-sum convention)

  @rq-e412e54a
  Scenario: Per-particle virial slot accumulates half the pair's scalar virial
    Given a single pair (0, 1) at finite distance with no exclusion
    When the _fev variant of coulomb_pair_force is called
    Then slot_virial[0] equals 0.5 * factor * r²
      where factor is the post-switching radial scalar force prefactor
    And slot_virial[0] + slot_virial[1] equals factor * r² within f32 round-off

  # --- Reproducibility ---

  @rq-1a0f3eef
  Scenario: Identical inputs produce byte-identical slot output
    Given two launches of the same coulomb_pair_force variant with identical inputs on the same GPU
    When both kernels return
    Then their slot_force_x, slot_force_y, slot_force_z, slot_energy, and slot_virial slot outputs
      are byte-identical

  # --- Empty state ---

  @rq-76a6be2f
  Scenario: Zero particles is a no-op
    Given particle_count == 0
    When the _f variant of coulomb_pair_force is called
    Then it returns Ok(()) without launching the kernel

  @rq-ee4ebbda
  Scenario: A neutral system with [coulomb] present still launches and produces zeros
    Given every particle has charge 0.0 and a [coulomb] table is present
    When the _f variant of coulomb_pair_force is called
    Then every slot output entry equals 0.0
    And the slot's reduced per-particle outputs equal 0.0

  # --- Newton's third law (bit-exact for non-boundary displacements) ---

  @rq-f652bf7c
  Scenario: Forces on the two members of a non-boundary pair are equal and opposite
    Given two particles whose minimum-image displacement is not at the primary-image boundary
    When the _f variant of coulomb_pair_force is called
    Then slot_force_x[i] equals -slot_force_x[j] bit-exact (for an isolated N=2 pair)
    And similarly for y and z
```
