# Long-Range Electrostatics (SPME)

HeddleMD computes electrostatic forces with the **smooth particle-mesh Ewald**
(SPME) method. SPME splits the `1/r` Coulomb interaction into a short-range
piece evaluated in real space (via the existing neighbor-list / pair-force
machinery) and a long-range piece evaluated on a 3D grid via FFTs. The
combined sum recovers the full Coulomb interaction within a controllable
error budget.

This document describes the architecture of the long-range electrostatics
calculation. It is a companion to `architecture.md` and assumes familiarity
with the project's reproducibility strategy, pair-buffer pattern, and
neighbor-list service.

## Ewald decomposition

The Coulomb energy between particle `i` and `j` at separation `r` is

```
U_ij(r) = q_i · q_j / r
```

with `k_C = 1 / (4πε₀) = 1` exactly in the engine's Hartree atomic
units (`q` in elementary charges, `r` in Bohr, `U` in Hartrees);
no permittivity prefactor appears in the kernel. Ewald's identity
rewrites `1/r` as the sum of a short-range and a long-range part
using the error function:

```
1/r = erfc(α · r) / r        (short-range, real space)
    + erf (α · r) / r        (long-range, smooth)
```

`α` is the Ewald splitting parameter, in inverse Bohr (`1/a_0`).
Larger `α` shifts
work into reciprocal space (short real-space cutoff, fine reciprocal grid);
smaller `α` shifts work into real space (long cutoff, coarse grid). The
total energy is partitioned into three contributions:

```
U_total = U_real + U_reciprocal − U_self
```

- **U_real** — pairwise sum of `k_C · q_i · q_j · erfc(α · r_ij) / r_ij`
  over particles within a real-space cutoff `r_cut_real`. Decays rapidly
  thanks to `erfc`, so the cutoff is short (typically a few angstroms).
- **U_reciprocal** — Fourier-space evaluation of the long-range smooth
  part using a charge density spread on a 3D grid. Decays in k-space
  thanks to a Gaussian factor `exp(−k²/4α²)`.
- **U_self** — a constant correction `k_C · α/√π · Σ q_i²` subtracted
  from `U_reciprocal` to remove each particle's self-interaction
  introduced by the charge spreading.

Forces follow from `F_i = −∇_i U_total`. The real-space piece produces
a per-pair force computed directly. The reciprocal-space piece produces
a per-particle force via the gradient of the reciprocal-space potential
evaluated at each particle's position.

## Pipeline overview

```
                                                   per-particle charges
                                                         │
                                                         v
                                              ┌──────────────────────┐
                                              │  Charge spreading    │
                                              │  (bin-and-gather)    │
                                              │  rho[g] = sum_i q_i  │
                                              │           · M(g - r_i)│
                                              └──────────┬───────────┘
                                                         v
                                                 rho[g] (real grid)
                                                         │
                                                         v
                                              ┌──────────────────────┐
 Positions ──┬─> Neighbor list                │   Forward 3D FFT     │  cuFFT
             │   (shared with                 │   R → C              │
             │    real-space slot)            └──────────┬───────────┘
             │                                            v
             │                                  rho_hat[k] (complex)
             │                                            │
             │                                            v
             │                                 ┌──────────────────────┐
             v                                 │  Influence-function  │
   ┌─────────────────┐                         │  multiply            │
   │ Real-space slot │                         │  V_hat[k] = G(k)     │
   │ U_real, F_real  │                         │            · rho_hat │
   │ erfc(α r)/r     │                         └──────────┬───────────┘
   │ (pair-buffer    │                                    v
   │  pattern)       │                            V_hat[k] (complex)
   └────────┬────────┘                                    │
            │                                             v
            │                                  ┌──────────────────────┐
            │                                  │   Inverse 3D FFT     │  cuFFT
            │                                  │   C → R              │
            │                                  └──────────┬───────────┘
            │                                             v
            │                                       V[g] (real grid)
            │                                             │
            │                                             v
            │                                  ┌──────────────────────┐
            │                                  │  Force gather        │
            │                                  │  F_i = -q_i · ∇M(g)  │
            │                                  │         · V[g]       │
            │                                  └──────────┬───────────┘
            │                                             │
            v                                             v
       F_real per                                   F_reciprocal
       particle                                     per particle
            │                                             │
            └────────────────┐         ┌──────────────────┘
                             v         v
                         ┌────────────────┐         constant
                         │ ForceField     │         self-energy
                         │ combiner       │         shift on U
                         │ F = F_real     │
                         │   + F_reciprocal
                         └────────────────┘
```

The diagram shows the per-timestep flow. The real-space and reciprocal-space
contributions run independently and combine through the standard
`ForceField` slot-output reduction (see `forces/framework.md`).

## Reciprocal-space pipeline

The reciprocal-space contribution lives in its own `Potential` slot in
the `ForceField`. The slot owns the FFT grid buffers and the cuFFT plan,
and produces a per-particle force and per-particle energy share that the
combiner sums into the final result.

### Charge spreading (bin-and-gather)

Each particle deposits charge onto a 3D real-valued grid using a B-spline
interpolation of order `p` (typically `p = 4`). A particle at fractional
position `s = (s_a, s_b, s_c)` contributes to the `p³` grid points
surrounding its position; the contribution to grid point `g = (g_a, g_b, g_c)`
is

```
rho[g] += q_i · M(s_a · n_a − g_a) · M(s_b · n_b − g_b) · M(s_c · n_c − g_c)
```

where `M` is the 1D cardinal B-spline of order `p` and `(n_a, n_b, n_c)`
is the FFT grid shape.

**Iteration direction.** The natural iteration ("for each particle, scatter
to its `p³` neighbouring grid points") requires atomic float adds when
particles overlap, which violates the project's bit-exact reproducibility
invariant. HeddleMD inverts the iteration: **one thread per grid point**,
each thread iterates over the `p³` adjacent cells of a spatial-hash bin
structure and accumulates contributions from every particle in those cells
in sorted particle-index order. Each grid point is therefore written by
exactly one thread; no atomics are needed.

The bin structure is the existing cell-list service from
`forces/neighbor-list.md`, parameterised here with a cell size equal to
one FFT grid cell (i.e. `cell_size_d = L_d / n_d` per lattice direction).
The same spatial-hash machinery that orders neighbours for the real-space
pair force also orders particles for the spread kernel, so the two share
their reproducibility story.

Memory cost of the spread pipeline:

- Spatial-hash bin offsets and sorted particle IDs: `O(N + M)` u32s where
  `M = n_a · n_b · n_c` is the total grid size.
- Charge grid: `M` f32s for `rho[g]`.

No per-particle, per-grid-point intermediate is materialised, in contrast
to the pair-buffer scatter pattern used by short-range forces. This is
the architectural choice that keeps SPME memory-tractable at large `N`:
the alternative — storing each particle's `p³` weighted contributions in
a buffer and reducing later — costs `O(N · p³)` and grows past a gigabyte
at `N = 10⁶`, `p = 6`.

### FFT and influence function

The spread kernel produces a real-valued grid `rho`. A single forward
real-to-complex 3D FFT produces `rho_hat`. cuFFT is the FFT backend:

- One plan is constructed at slot-init time for the grid dimensions
  `(n_a, n_b, n_c)`.
- The plan is reused for every timestep.
- The kernel uses cuFFT's `execR2C` (forward) and `execC2R` (inverse)
  entry points on the default stream.

Determinism: cuFFT is documented as bit-exactly deterministic for fixed
plan dimensions, fixed hardware, single-stream usage, and a single host
process — the configuration HeddleMD always uses. A smoke test at
`init_device` time validates the documented contract on the host's
specific cuFFT version.

The influence function is

```
G(k) = (4π / k²) · exp(−k² / (4α²)) · |b(k_a, p) · b(k_b, p) · b(k_c, p)|²
```

where `k = 2π · (k_a · b₁ + k_b · b₂ + k_c · b₃)` is the reciprocal-lattice
vector (rows of `H^{−T}` scaled by `2π` per the simulation box's lower-
triangular convention), and `b(k, p)` is the SPME B-spline correction
factor (a per-grid-axis precomputed array). The `k = 0` entry is set to
zero, implementing tinfoil boundary conditions.

The influence function depends only on the box (via the reciprocal lattice)
and the chosen `α` and grid; it is precomputed once at slot-init time and
re-used every step. When the box's `generation` counter changes (a future
barostat), the slot rebuilds `G(k)` from the new lattice.

The per-cell multiply `V_hat[k] = G[k] · rho_hat[k]` is one thread per
grid cell, no atomics. Followed by an inverse complex-to-real 3D FFT to
produce the smoothed reciprocal-space potential `V[g]` on the real grid.

### Force gather

The reciprocal-space force on particle `i` is the negative gradient of
the smoothed potential evaluated at the particle's position:

```
F_i_recip = −q_i · ∇ ( Σ_g V[g] · M(s_a · n_a − g_a) · M(s_b · n_b − g_b) · M(s_c · n_c − g_c) )
```

Operationally, each thread handles one particle, samples the `p³`
surrounding grid points, and accumulates the gradient via the analytic
derivative of the B-spline weights. The natural iteration direction
("for each particle, read its `p³` grid points") is already deterministic:
each particle writes to its own forces slot, no atomics, no inter-thread
data race. The gather kernel mirrors the spread kernel's iteration order
particle-side, but with no inversion needed.

Per-particle energy shares are accumulated alongside the forces: each
particle's contribution to `U_reciprocal` is `0.5 · q_i · V_interpolated_at_r_i`.

## Real-space slot

The real-space contribution is a pair-force `Potential` slot. Each thread
handles one `(i, k)` pair-buffer slot just like the truncated-Coulomb
slot it generalises, but with the screening factor `erfc(α · r) / r`
replacing `1/r`:

```
U_real_ij(r) = k_C · q_i · q_j · erfc(α · r) / r
F_real_ij    = -dU/dr · r_hat
             = k_C · q_i · q_j · (2 α/√π · exp(−α² r²) + erfc(α r)/r) · r_ij/r²
```

The slot reuses the existing pair-buffer + neighbor-list + reduction
infrastructure. The kernel reads per-particle charges from
`ParticleBuffers`, computes the screened force, applies the Coulomb
exclusion scale from the shared `DeviceExclusionList`, and writes to its
own pair buffer at deterministic offsets. Reduction is the same
`reduce_pair_forces` kernel that LJ and truncated Coulomb use.

The real-space cutoff `r_cut_real` is independent of (and typically
shorter than) the cutoffs of other short-range potentials. The neighbor
list's search radius is set to `max(LJ_cutoff, real_cutoff) + r_skin` so
both consumers share a single neighbor list.

## Self-energy

Each particle is included in its own charge-spread contribution, which
the reciprocal-space sum then over-counts as a self-interaction. The
correction is a constant scalar:

```
U_self = k_C · (α/√π) · Σ_i q_i²
```

`U_self` does not depend on positions; the SPME slot computes it once at
slot-init from the per-particle charges and subtracts it as a constant
shift on the total potential energy at every step. There is no per-
particle force contribution. The shift is bookkept as a scalar offset
the slot adds during its `reduce()` step.

## Triclinic cells

SPME works on triclinic cells with no algorithmic change beyond the
choice of reciprocal lattice. Concretely:

- **FFT grid.** The grid has `n_a × n_b × n_c` cells in fractional
  coordinates. Grid resolution per direction is chosen as a target
  spacing in Bohr divided by the corresponding perpendicular width.
- **k-vectors.** `k = 2π · (i_a · b_a + i_b · b_b + i_c · b_c)` where
  `b_a, b_b, b_c` are the rows of `H^{−T}` (the reciprocal lattice).
  The dot products in the influence function use the full Cartesian
  `k²`.
- **B-spline correction.** Computed in fractional space; identical
  structure for any cell shape.

The same SPME pipeline runs unchanged on orthorhombic and triclinic
boxes. The triclinic case adds only the per-step `H^{−T}` multiply
when computing `k` magnitudes (a constant cost per grid cell).

## Reproducibility

SPME on HeddleMD is bit-exact GPU-vs-GPU. Five components carry the
reproducibility invariant:

1. **Spatial-hash bin structure.** Inherits the existing service's
   determinism: particles are sorted by `(cell_index, particle_id)` and
   the sort is fixed-topology.
2. **Charge spreading.** One thread per grid point reads bins in fixed
   order; contributions from particles in each bin are summed in
   sorted particle-ID order. No atomics.
3. **cuFFT.** Deterministic for fixed plan dimensions and single-stream
   usage on the same hardware. Validated by a smoke test that runs at
   `init_device` and confirms two FFT passes on identical inputs produce
   byte-identical complex output.
4. **Influence-function multiply.** One thread per grid cell, no atomics.
5. **Force gather.** One thread per particle, each thread reads `p³`
   grid points in a fixed order. No atomics.

The real-space slot inherits the existing pair-buffer + reduction
reproducibility story unchanged.

## Memory model

For a system of `N` particles on an FFT grid of `M = n_a · n_b · n_c`
cells with B-spline order `p`, the SPME slot owns:

| Buffer | Size | Comment |
| --- | --- | --- |
| `rho`           | `M` f32   | Real charge density grid |
| `V`             | `M` f32   | Real smoothed-potential grid |
| `rho_hat` / `V_hat` | `(n_a · n_b · (n_c/2 + 1))` complex64 | R2C-format complex grid (cuFFT uses Hermitian symmetry) |
| `influence_G`   | `(n_a · n_b · (n_c/2 + 1))` f32 | Precomputed influence function |
| `b_factors_*`   | `n_a + n_b + n_c` f32 | Per-axis B-spline correction |

Total grid-side memory is roughly `4 · M · 4 bytes ≈ 16 · M` bytes. For
`M = 100³ = 10⁶`, that's ~16 MB. Plus the bin index (`O(N + M)` u32s,
typically a few MB). The bin-and-gather choice keeps the per-particle
overhead at `O(N)`, independent of `p³`.

The cuFFT plan itself owns workspace allocated by cuFFT at plan-init time
(typically `O(M)` complex values for an in-place 3D R2C transform).

## Configuration

The SPME slot is configured by a `[spme]` table in the simulation config
(details in `rqm/forces/spme.md`). Required parameters: real-space
cutoff, Ewald splitting parameter `α`, FFT grid dimensions, and B-spline
order. The shared `[coulomb]` table (truncated Coulomb) and the `[spme]`
table are mutually exclusive: a config may declare at most one. When
`[spme]` is present, the real-space `erfc` slot replaces the truncated
Coulomb slot in the `ForceField`'s slot order.

## Extensibility

- **Higher-order B-splines.** `p > 6` is supported by the same code path;
  the only consequence is more work per grid point in spread/gather and a
  modestly larger spatial-hash cell-list (because the gather reads `p³`
  neighbours).
- **Time-varying box.** When a barostat changes the simulation box, the
  influence function `G(k)` is rebuilt from the new lattice (the FFT
  grid dimensions stay fixed; only the per-cell coefficients change).
  The cuFFT plan does not need to be re-created.
- **PME variants.** Smooth PME with derivatives is the implemented
  variant. Particle-Particle-Particle-Mesh (P3M) differs only in the
  influence-function choice and could be added as a config option on
  the same pipeline.
- **Multi-grid PME.** Splitting the reciprocal sum across multiple grids
  of different resolutions is a known optimisation for very large
  systems. Not implemented; would add a second `(spread, FFT, multiply,
  IFFT, gather)` pipeline at a coarser resolution and sum the two
  reciprocal-space contributions.
