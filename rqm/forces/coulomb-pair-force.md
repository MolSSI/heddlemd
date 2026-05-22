# Feature: Truncated Coulomb Pair Force Kernel <!-- rq-846bdb8b -->

Truncated Coulomb is the non-bonded electrostatic pairwise potential slot
in the pluggable potential framework (`framework.md`). The slot is present
when the config declares a `[coulomb]` table. Its parameters are a single
real-space cutoff and a single inner-switching radius; pair magnitudes are
constructed at kernel time from the per-particle charges carried by
`ParticleBuffers` (see `particle-state.md`).

The slot evaluates pair forces with a single kernel that reads the shared
`NeighborListState` owned by `ForceField` (see `neighbor-list.md`). Each
`(i, k)` thread reads `neighbor_list[i * max_neighbors + k]` to find its
partner `j`, applies the cutoff and switching function, applies the
Coulomb-specific per-pair exclusion scale, and writes the pair force into
the `PairBuffer` at slot `i * max_neighbors + k`. Work is O(N · average
neighbour count) when the shared list is in cell-list mode and O(N²) when
it is in trivial mode (every particle's list contains every particle).
The kernel is the same in both cases; only the contents and size of the
shared list differ.

This slot is the smallest viable charge-carrying milestone in the project's
electrostatics roadmap. It is intentionally short-range only: real-space
truncation of `1/r` is known to introduce artefacts for charged systems
larger than a few nanometres. A future smooth particle-mesh Ewald slot
will replace the long-range contribution; the real-space and exclusion
machinery exercised here carries over.

This file specifies `CoulombParameters` (the host-side parameter
container), the `coulomb_pair_force` CUDA kernel in `kernels/coulomb.cu`,
and the Rust launch helper that drives it.

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

For each ordered `(i, k)` with `0 <= i < N` and `0 <= k <
neighbor_counts[i]`:

1. The pair-buffer slot is `slot = i * max_neighbors + k`.
2. Read `j = neighbor_list[slot]`. If `i == j`, write `0.0_f32` to
   `pair_forces_x[slot]`, `pair_forces_y[slot]`, `pair_forces_z[slot]`,
   `pair_energies[slot]`, and `pair_virials[slot]` and stop. (The kernel
   encounters `i == j` only when the shared neighbor list is in trivial
   mode, which lists every particle including self.)
3. Look up `q_i = charges[i]` and `q_j = charges[j]`.
4. Compute the displacement `dx = positions_x[i] - positions_x[j]` (and
   similarly `dy`, `dz`).
5. Apply the triclinic minimum-image convention to `(dx, dy, dz)` using
   the six lattice parameters `(lx, ly, lz, xy, xz, yz)` and the
   fractional-coordinate wrap algorithm defined in `simulation-box.md`.
   For an orthorhombic box (all tilts zero) this reduces to three
   independent per-axis wraps.
6. Compute `r2 = dx*dx + dy*dy + dz*dz`.
7. If `r2 > cutoff * cutoff`, write `0.0_f32` to all five slots
   (force x/y/z, energy, virial) and stop.
8. Compute the unswitched Coulomb factor and energy in this order:

   ```text
   inv_r2  = 1.0f / r2
   inv_r   = sqrtf(inv_r2)
   qq      = q_i * q_j
   energy  = k_C * qq * inv_r
   factor  = k_C * qq * inv_r * inv_r2          // F = factor · r_ij
   ```

   `factor * r_ij` is the Coulomb force on particle `i` due to `j`.
9. Apply the CHARMM-style C¹ switching function `S(r²)` defined over
   `[r_switch², cutoff²]`. Let `r_s2 = r_switch * r_switch` and
   `r_c2 = cutoff * cutoff`. The polynomial is evaluated in
   normalised form so the only place the cutoff-width factor appears
   explicitly is `1/delta` where `delta = r_c2 - r_s2`.

   - If `r2 <= r_s2`, the inner plateau has `S = 1` and `dS/d(r²) = 0`,
     so `factor` and `energy` are unchanged.
   - Otherwise `r_s2 < r2 <= r_c2` (the `r2 > r_c2` case was gated above)
     and the polynomial branch runs.

     With `tau = (r² − r_s2) / delta` the polynomial reduces to
     `S = (1 − tau)² (1 + 2 tau)` and
     `dS/d(r²) = −6 tau (1 − tau) / delta`.

     Apply the switching factor as a multiplication on `energy` and on
     `factor`. The chain-rule correction adds `2 · energy · dS_dr2` to
     `factor` (matches the identical correction in `lj-pair-force.md`).

   When `r_switch = cutoff` the switching interval is degenerate and the
   polynomial branch is never entered; the kernel writes `factor` and
   `energy` unchanged for any `r2 ≤ r_c2`, producing the hard-cutoff
   case.
10. Apply the per-pair Coulomb exclusion scale (see `topology.md`). The
    kernel walks the per-atom exclusion slice
    `atom_excl_partners[atom_excl_offsets[i] .. atom_excl_offsets[i+1]]`
    looking for `j`; if present, the corresponding entry in
    `atom_excl_coul_scales` multiplies both `factor` and `energy`. A
    scale of `0.0` fully excludes the pair; a scale in `(0.0, 1.0)`
    partially attenuates it. Pairs not listed in the exclusion table
    are evaluated with scale `1.0` (no attenuation).
11. Compute the scalar virial contribution from this pair as
    `w = factor * r²` (which equals `r_ij · F_ij`). Apply the exclusion
    scale to `w` along with `factor` and `energy`.
12. Write the final values to the pair buffer:

    ```text
    pair_forces_x[slot] = factor * dx
    pair_forces_y[slot] = factor * dy
    pair_forces_z[slot] = factor * dz
    pair_energies[slot] = 0.5 * energy
    pair_virials[slot]  = 0.5 * w
    ```

    The `0.5` factors implement the half-sum convention shared with
    `lj-pair-force.md` and `morse-bonded.md` (see `pair-reduction.md`):
    summing over every ordered pair `(i, j)` counts each unordered pair
    exactly once when totalled.

The kernel writes `0.0_f32` to every slot beyond `neighbor_counts[i]`
(the kernel grid spans all `max_neighbors` slots per particle; the
trailing slots are zeroed unconditionally).

## Reproducibility <!-- rq-03423870 -->

The pair-buffer slot for `(i, k)` is written by exactly one thread; there
are no atomics and no race conditions. Each contribution is a deterministic
function of `q_i`, `q_j`, the wrapped displacement, the six lattice
parameters, the cutoff and switching parameters, and the per-pair
exclusion scale. The reduction kernel `reduce_pair_forces` sums the
per-particle contributions in fixed slot order (see `pair-reduction.md`).
Identical runs on the same GPU with identical inputs produce
byte-identical `slot_force_*` outputs.

## Newton's third law <!-- rq-e858417a -->

Threads `(i, j)` and `(j, i)` independently compute `F_ij` and `F_ji`.
The displacements differ only in sign, the wrap formula respects sign
symmetry for displacements not exactly at the primary-image boundary,
and the Coulomb factor depends only on `r²` and on the product `q_i · q_j`
(commutative). The exclusion lookup table is constructed so the
`(i, j)` and `(j, i)` scales are equal (see `topology.md`). The two
threads' Cartesian forces therefore differ by a sign bit-for-bit for
all displacements except the measure-zero exact-boundary case
documented in `lj-pair-force.md`.

## Parameters <!-- rq-f9a9c569 -->

The slot owns a single `CoulombParameters` record (host-side) carrying
the cutoff and the inner switching radius, both in metres:

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
reduced output buffers are length zero.

When no `[coulomb]` table is supplied in the config, the slot is absent
from the `ForceField` and contributes nothing.

## Feature API <!-- rq-4968452d -->

### Types <!-- rq-a3fdbd7c -->

- `CoulombParameters` — host-side parameter container. Fields:
  - `cutoff: f32`
  - `r_switch: f32`

- `CoulombState` — implements the `Potential` trait with
  `label() == "coulomb"` (see `framework.md`). Fields:
  - `device: Arc<CudaDevice>`
  - `kernels: Arc<Kernels>`
  - `params: CoulombParameters`
  - `particle_count: usize`
  - `pair_buffer: PairBuffer` — owned by the slot.

  All fields private; the slot's public surface is the per-step methods
  invoked by `ForceField::step` (see `framework.md`).

  Constructor:

  - `CoulombState::new(device: Arc<CudaDevice>, kernels: Arc<Kernels>, params: CoulombParameters, particle_count: usize, max_neighbors: u32) -> Result<CoulombState, GpuError>`
    - Allocates the slot's `PairBuffer` with `max_neighbors` matching
      the shared neighbor list's value.

  The `Potential` implementation reports
  `max_cutoff() = Some(params.cutoff)` so the shared neighbor list
  sizes its search radius accordingly.

### Functions <!-- rq-7bae4b69 -->

- `coulomb_pair_force(particle_buffers: &ParticleBuffers, pair_buffer: &mut PairBuffer, sim_box: &SimulationBox, params: &CoulombParameters, atom_excl_offsets: &CudaSlice<u32>, atom_excl_partners: &CudaSlice<u32>, atom_excl_coul_scales: &CudaSlice<f32>, neighbor_list: &CudaSlice<u32>, neighbor_counts: &CudaSlice<u32>) -> Result<(), GpuError>`
  - Launches the `coulomb_pair_force` kernel with the layout described
    in *Launch Configuration*.
  - When `particle_buffers.particle_count() == 0`, returns `Ok(())`
    without launching.
  - Returns the underlying `GpuError` if the kernel launch fails.

### CUDA kernel <!-- rq-4a96b705 -->

`kernels/coulomb.cu` declares the following `extern "C"` kernel:

```c
extern "C" __global__ void coulomb_pair_force(
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    const float *charges,
    float *pair_forces_x,
    float *pair_forces_y,
    float *pair_forces_z,
    float *pair_energies,
    float *pair_virials,
    unsigned int max_neighbors,
    float lx, float ly, float lz, float xy, float xz, float yz,
    float k_coulomb,
    float cutoff,
    float r_switch,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const float *atom_excl_coul_scales,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    unsigned int n);
```

The launcher trusts the caller for shape consistency: in debug builds
it asserts `pair_buffer.particle_count() == particle_buffers.particle_count()`,
`neighbor_list.len() == particle_buffers.particle_count() * pair_buffer.max_neighbors() as usize`,
`neighbor_counts.len() == particle_buffers.particle_count()`,
`atom_excl_offsets.len() == particle_buffers.particle_count() + 1`,
`atom_excl_partners.len() == atom_excl_coul_scales.len()`, and
`charges.len() == particle_buffers.particle_count()`. Release builds
skip the asserts for parity with the other kernel launchers. The
launcher does not validate the parameter values themselves and does
not check `cutoff` against `sim_box.min_perpendicular_width() / 2`
(that gating happens in the neighbor list; see `forces/neighbor-list.md`).

## Launch Configuration <!-- rq-84af056c -->

- Block size: 16 × 16 × 1 = 256 threads per block.
- Grid size: `(ceil(max_neighbors / 16), ceil(n / 16), 1)` blocks.
- Shared memory: zero bytes.
- Stream: the default stream carried by `pair_buffer.device`.

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
    When coulomb_pair_force is called
    Then pair_forces_x[0*2 + 1] equals the closed-form Coulomb force
      F_x = k_C · (+1e) · (-1e) · (-3e-10) / (3e-10)³
        = -k_C · e² · 3e-10 / (3e-10)³
      which is negative (atom 0 is pulled toward atom 1 along +x)

  @rq-82e4f74e
  Scenario: Same-sign charges repel
    Given two particles with charges (+1e, +1e) at positions
      (0, 0, 0) and (3e-10, 0, 0)
    When coulomb_pair_force is called
    Then pair_forces_x[0*2 + 1] is positive (atom 0 is pushed away from atom 1)
    And pair_forces_x[1*2 + 0] equals -pair_forces_x[0*2 + 1] (Newton's third law)

  @rq-3b7da473
  Scenario: Zero charges produce zero force
    Given two neutral particles at positions (0, 0, 0) and (3e-10, 0, 0)
    When coulomb_pair_force is called
    Then all pair_forces_*, pair_energies, pair_virials slots equal 0.0

  @rq-02a15197
  Scenario: Mixed charge magnitudes scale linearly
    Given two particles with charges (q_i, q_j) at distance r
    When coulomb_pair_force is called
    Then pair_forces_x[0*2 + 1] is proportional to q_i · q_j

  # --- Cutoff gating ---

  @rq-0896c33a
  Scenario: Pair beyond cutoff contributes zero
    Given a CoulombParameters with cutoff=2.0e-10
    And two particles with charges (+1e, +1e) at separation 5e-10 (> cutoff)
    When coulomb_pair_force is called
    Then pair_forces_x[0*2 + 1], pair_energies[0*2 + 1], and pair_virials[0*2 + 1] equal 0.0

  @rq-3df3ed61
  Scenario: Pair at exactly the cutoff contributes the smoothed (zero) value
    Given a CoulombParameters with cutoff=3.0e-10 and r_switch=2.7e-10
    And two particles with charges (+1e, +1e) at separation exactly 3.0e-10
    When coulomb_pair_force is called
    Then pair_energies[0*2 + 1] equals 0.0 (the switching function S(r_c²) = 0)
    And pair_forces_*[0*2 + 1] equals 0.0

  # --- Switching function ---

  @rq-b07abbd4
  Scenario: Pair inside the inner plateau is unsmoothed
    Given a CoulombParameters with cutoff=5e-10 and r_switch=4e-10
    And two particles with charges (+1e, +1e) at separation 3.5e-10 (< r_switch)
    When coulomb_pair_force is called
    Then pair_energies[0*2 + 1] equals the unswitched Coulomb energy
      0.5 · k_C · (1e)² / 3.5e-10
    And pair_forces_x[0*2 + 1] equals the unswitched Coulomb force component
      along +x: k_C · (1e)² · 3.5e-10 / (3.5e-10)³

  @rq-d52bcc88
  Scenario: Pair inside the switching interval is smoothed
    Given a CoulombParameters with cutoff=5e-10 and r_switch=4e-10
    And two particles with charges (+1e, +1e) at separation 4.5e-10
      (between r_switch and cutoff)
    When coulomb_pair_force is called
    Then pair_energies[0*2 + 1] is strictly between 0 and the unswitched value
    And pair_forces_x[0*2 + 1] is strictly between 0 and the unswitched x-component

  @rq-67678030
  Scenario: Switching interval r_switch == cutoff selects hard-cutoff
    Given a CoulombParameters with cutoff = r_switch = 4e-10
    And two particles with charges (+1e, +1e) at separation 3.9e-10
    When coulomb_pair_force is called
    Then pair_energies[0*2 + 1] equals the unswitched Coulomb energy (S = 1)

  # --- PBC ---

  @rq-ef3083cd
  Scenario: Pair across the periodic boundary uses minimum-image displacement
    Given two particles with charges (+1e, -1e) at positions
      (-lx/2 + 0.1e-10, 0, 0) and (+lx/2 - 0.1e-10, 0, 0)
      (which are 0.2e-10 apart across the periodic boundary)
    And a CoulombParameters with cutoff = 5e-10
    When coulomb_pair_force is called
    Then the closed-form force uses minimum-image dx = +0.2e-10, not lx - 0.2e-10
    And pair_forces_x[0*2 + 1] is negative (attraction across the +x face)

  @rq-9af1f9dc
  Scenario: Minimum-image works for a triclinic box
    Given a SimulationBox with non-zero tilts (e.g. xy=2e-10, xz=1e-10, yz=-3e-10)
    And two particles at positions whose minimum-image separation crosses a
      tilted face of the primary parallelepiped
    When coulomb_pair_force is called
    Then the computed displacement matches the triclinic minimum-image
      result of sim_box.minimum_image(r_i - r_j)

  # --- Self-pair handling ---

  @rq-bf7dfc6d
  Scenario: Self slot in trivial-mode neighbor list yields zero
    Given a 1-particle system with charge +1e in trivial mode
    When coulomb_pair_force is called
    Then pair_forces_x[0*1 + 0] equals 0.0 (i == j short-circuit)
    And pair_energies[0*1 + 0] equals 0.0
    And pair_virials[0*1 + 0] equals 0.0

  # --- Exclusions ---

  @rq-c4d4608f
  Scenario: Pair with Coulomb exclusion scale 0.0 contributes nothing
    Given two particles with charges (+1e, -1e) at separation 3e-10
    And an ExclusionList listing pair (0, 1) with scale_coul = 0.0
    When coulomb_pair_force is called
    Then pair_forces_*[0*2 + 1] equals 0.0
    And pair_energies[0*2 + 1] equals 0.0
    And pair_virials[0*2 + 1] equals 0.0

  @rq-d26e9f9c
  Scenario: Pair with Coulomb exclusion scale 0.5 contributes half
    Given two particles with charges (+1e, -1e) at separation 3e-10
    And an ExclusionList listing pair (0, 1) with scale_coul = 0.5
    When coulomb_pair_force is called
    Then pair_energies[0*2 + 1] equals 0.5 times the unscaled value
    And pair_forces_x[0*2 + 1] equals 0.5 times the unscaled x-component

  @rq-8c96d3c7
  Scenario: Coulomb and LJ exclusions are independent
    Given two particles whose ExclusionList entry is scale_lj=0.5, scale_coul=0.833
    When coulomb_pair_force is called
    Then the Coulomb contribution is scaled by 0.833
    When lj_pair_force is called on the same pair
    Then the LJ contribution is scaled by 0.5

  # --- Energy and virial conventions ---

  @rq-5444c7ae
  Scenario: pair_energies carries half the pair's potential energy
    Given a single pair (i, j) at finite distance with no exclusion
    When coulomb_pair_force is called
    Then pair_energies[i*max_neighbors + slot_for_j]
        equals 0.5 * (k_C * q_i * q_j / r) * S(r²)

  @rq-e412e54a
  Scenario: pair_virials carries half the pair's scalar virial
    Given a single pair (i, j) at finite distance with no exclusion
    When coulomb_pair_force is called
    Then pair_virials[i*max_neighbors + slot_for_j] equals 0.5 * factor * r²
      where factor is the post-switching radial scalar force prefactor

  @rq-d01b6fb0
  Scenario: Slots beyond neighbor_counts[i] are explicitly zeroed
    Given particle i with neighbor_counts[i] = 3 and max_neighbors = 8
    When coulomb_pair_force is called
    Then slots i*8 + 3 through i*8 + 7 are all 0.0 in every output buffer

  # --- Reproducibility ---

  @rq-1a0f3eef
  Scenario: Identical inputs produce byte-identical pair-buffer outputs
    Given two coulomb_pair_force launches with identical inputs on the same GPU
    When both kernels return
    Then their pair_forces_x/y/z, pair_energies, and pair_virials buffers
      are byte-identical

  # --- Empty state ---

  @rq-76a6be2f
  Scenario: Zero particles is a no-op
    Given particle_count == 0
    When coulomb_pair_force is called
    Then it returns Ok(()) without launching the kernel

  @rq-ee4ebbda
  Scenario: A neutral system with [coulomb] present still launches and produces zeros
    Given every particle has charge 0.0 and a [coulomb] table is present
    When coulomb_pair_force is called
    Then every pair-buffer slot equals 0.0
    And the slot's reduced per-particle outputs equal 0.0

  # --- Newton's third law (bit-exact for non-boundary displacements) ---

  @rq-f652bf7c
  Scenario: Forces on the two members of a non-boundary pair are equal and opposite
    Given two particles whose minimum-image displacement is not at the primary-image boundary
    When coulomb_pair_force is called
    Then pair_forces_x[i*max + slot_j] == -pair_forces_x[j*max + slot_i] (bit-exact)
    And similarly for y and z
```
