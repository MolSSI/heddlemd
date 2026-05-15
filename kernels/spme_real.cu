// rq-f6d45062 rq-39b05bc9

#include "exclusions.cuh"
#include "pbc.cuh"

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
  unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || k >= max_neighbors) {
    return;
  }
  unsigned int slot = i * max_neighbors + k;
  if (k >= neighbor_counts[i]) {
    pair_forces_x[slot] = 0.0f;
    pair_forces_y[slot] = 0.0f;
    pair_forces_z[slot] = 0.0f;
    pair_energies[slot] = 0.0f;
    pair_virials[slot]  = 0.0f;
    return;
  }
  unsigned int j = neighbor_list[slot];

  if (i == j) {
    pair_forces_x[slot] = 0.0f;
    pair_forces_y[slot] = 0.0f;
    pair_forces_z[slot] = 0.0f;
    pair_energies[slot] = 0.0f;
    pair_virials[slot]  = 0.0f;
    return;
  }

  float qi = charges[i];
  float qj = charges[j];

  float dx = positions_x[i] - positions_x[j];
  float dy = positions_y[i] - positions_y[j];
  float dz = positions_z[i] - positions_z[j];

  triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);

  float r_c2 = r_cut_real * r_cut_real;
  float r2 = dx * dx + dy * dy + dz * dz;
  if (r2 > r_c2) {
    pair_forces_x[slot] = 0.0f;
    pair_forces_y[slot] = 0.0f;
    pair_forces_z[slot] = 0.0f;
    pair_energies[slot] = 0.0f;
    pair_virials[slot]  = 0.0f;
    return;
  }

  float inv_r2 = 1.0f / r2;
  float r      = sqrtf(r2);
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

  // Per-pair Coulomb exclusion scale (see bonds.md).
  float scale = exclusion_scale(
      i, j, atom_excl_offsets, atom_excl_partners, atom_excl_coul_scales);
  factor *= scale;
  energy *= scale;

  float fx = factor * dx;
  float fy = factor * dy;
  float fz = factor * dz;
  float w  = fx * dx + fy * dy + fz * dz;

  pair_forces_x[slot] = fx;
  pair_forces_y[slot] = fy;
  pair_forces_z[slot] = fz;
  pair_energies[slot] = energy * 0.5f;
  pair_virials[slot]  = w * 0.5f;
}
