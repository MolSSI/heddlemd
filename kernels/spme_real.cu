// rq-f6d45062 rq-39b05bc9

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
// 1 / sqrt(π) ≈ 0.5641895835477563f.
__device__ static const float ONE_OVER_SQRT_PI = 0.5641895835477563f;

extern "C" __global__ void spme_real_pair_force(
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
    float alpha,
    float r_cut_real,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const float *atom_excl_coul_scales,
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

  float qi = charges[f.i];
  float qj = charges[f.j];

  float r_c2 = r_cut_real * r_cut_real;
  if (f.r2 > r_c2) {
    pair_frame_write_zero(f.slot,
        pair_forces_x, pair_forces_y, pair_forces_z,
        pair_energies, pair_virials);
    return;
  }

  float inv_r2 = 1.0f / f.r2;
  float r      = sqrtf(f.r2);
  float inv_r  = 1.0f / r;
  float qq     = qi * qj;
  float ar     = alpha * r;
  float erfc_ar = erfcf(ar);
  float gauss   = expf(-(ar * ar));
  float energy  = k_coulomb * qq * erfc_ar * inv_r;
  // factor multiplies r_ij to give the force on i: F = factor · r_ij.
  //   d/dr (erfc(αr) / r) = -erfc(αr)/r² - (2α/√π) exp(-α² r²) / r
  // factor = -(1/r) · d/dr (erfc(αr)/r)
  //        = (erfc(αr) / r³ + (2α/√π) exp(-α² r²) / r²) · k_C · q_i q_j
  float factor = k_coulomb * qq * inv_r2
               * (erfc_ar * inv_r + 2.0f * alpha * ONE_OVER_SQRT_PI * gauss);

  // Per-pair Coulomb exclusion scale (see bonds.md). Applied via factor
  // and energy (not via pair_frame_write) so that the SPME real-space
  // arithmetic order is preserved: fx/fy/fz inherit the scale through
  // `factor` and w inherits it via `fx*dx + fy*dy + fz*dz`. The write
  // call below then passes `scale = 1.0f`; multiplication by exactly
  // 1.0f is bit-exact and only the 0.5f halving runs.
  float scale = exclusion_scale(
      f.i, f.j, atom_excl_offsets, atom_excl_partners, atom_excl_coul_scales);
  factor *= scale;
  energy *= scale;

  float fx = factor * f.dx;
  float fy = factor * f.dy;
  float fz = factor * f.dz;
  float w  = fx * f.dx + fy * f.dy + fz * f.dz;

  pair_frame_write(
      f.slot, fx, fy, fz, energy, w, 1.0f,
      pair_forces_x, pair_forces_y, pair_forces_z,
      pair_energies, pair_virials);
}
