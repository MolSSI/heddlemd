// rq-f6d45062 rq-39b05bc9

#include "precision.cuh"

#include "exclusions.cuh"
#include "pair_frame.cuh"

// rq-9a512ed1
//
// Pair force kernel for SPME's real-space contribution. Same structure
// as `lj_pair_force` / `coulomb_pair_force` (one thread per
// (i, k) pair-buffer slot reading the shared neighbor list), but the
// pair functional form is the erfc-screened Coulomb:
//
//   U_ij(r) = k_C · q_i · q_j · erfc(α r) / r
//   F_ij    = k_C · q_i · q_j · (erfc(α r) / r + (2α/√π) exp(-α² r²))
//             · r_ij / r²
//
// 1 / sqrt(π) ≈ R(0.5641895835477563).
__device__ static const Real ONE_OVER_SQRT_PI = R(0.5641895835477563);

extern "C" __global__ void spme_real_pair_force(
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const Real *charges,
    Real *pair_forces_x,
    Real *pair_forces_y,
    Real *pair_forces_z,
    Real *pair_energies,
    Real *pair_virials,
    unsigned int max_neighbors,
    Real lx, Real ly, Real lz, Real xy, Real xz, Real yz,
    Real k_coulomb,
    Real alpha,
    Real r_cut_real,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const Real *atom_excl_coul_scales,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    unsigned int n)
{
  PairFrame f = pair_frame_setup(
      n, max_neighbors,
      positions_x, positions_y, positions_z,
      neighbor_list, neighbor_counts,
      lx, ly, lz, xy, xz, yz,
      pair_forces_x, pair_forces_y, pair_forces_z,
      pair_energies, pair_virials);
  if (!f.active) {
    return;
  }

  Real qi = charges[f.i];
  Real qj = charges[f.j];

  Real r_c2 = r_cut_real * r_cut_real;
  if (f.r2 > r_c2) {
    pair_frame_write_zero(f.slot,
        pair_forces_x, pair_forces_y, pair_forces_z,
        pair_energies, pair_virials);
    return;
  }

  Real inv_r2 = R(1.0) / f.r2;
  Real r      = Real_sqrt(f.r2);
  Real inv_r  = R(1.0) / r;
  Real qq     = qi * qj;
  Real ar     = alpha * r;
  Real erfc_ar = erfcf(ar);
  Real gauss   = Real_exp(-(ar * ar));
  Real energy  = k_coulomb * qq * erfc_ar * inv_r;
  // factor multiplies r_ij to give the force on i: F = factor · r_ij.
  //   d/dr (erfc(αr) / r) = -erfc(αr)/r² - (2α/√π) exp(-α² r²) / r
  // factor = -(1/r) · d/dr (erfc(αr)/r)
  //        = (erfc(αr) / r³ + (2α/√π) exp(-α² r²) / r²) · k_C · q_i q_j
  Real factor = k_coulomb * qq * inv_r2
               * (erfc_ar * inv_r + R(2.0) * alpha * ONE_OVER_SQRT_PI * gauss);

  // Per-pair Coulomb exclusion scale (see bonds.md). Applied via factor
  // and energy (not via pair_frame_write) so that the SPME real-space
  // arithmetic order is preserved: fx/fy/fz inherit the scale through
  // `factor` and w inherits it via `fx*dx + fy*dy + fz*dz`. The write
  // call below then passes `scale = R(1.0)`; multiplication by exactly
  // R(1.0) is bit-exact and only the R(0.5) halving runs.
  Real scale = exclusion_scale(
      f.i, f.j, atom_excl_offsets, atom_excl_partners, atom_excl_coul_scales);
  factor *= scale;
  energy *= scale;

  Real fx = factor * f.dx;
  Real fy = factor * f.dy;
  Real fz = factor * f.dz;
  Real w  = fx * f.dx + fy * f.dy + fz * f.dz;

  pair_frame_write(
      f.slot, fx, fy, fz, energy, w, R(1.0),
      pair_forces_x, pair_forces_y, pair_forces_z,
      pair_energies, pair_virials);
}
