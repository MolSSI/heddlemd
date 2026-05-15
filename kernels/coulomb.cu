// rq-846bdb8b

#include "exclusions.cuh"
#include "pbc.cuh"

// rq-bfd7004c
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

  float r_c2 = cutoff * cutoff;
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
  float inv_r  = sqrtf(inv_r2);
  float qq     = qi * qj;
  float energy = k_coulomb * qq * inv_r;
  // The Coulomb force on i due to j is F = k_C * q_i * q_j * r_ij / r^3.
  // factor = k_C * q_i * q_j / r^3, so force_x = factor * dx, etc.
  float factor = k_coulomb * qq * inv_r * inv_r2;

  // CHARMM-style C1 switching function applied over [r_switch, r_cut].
  // Identical structure to lj_pair_force; see lj-pair-force.md.
  float r_s2 = r_switch * r_switch;
  if (r2 > r_s2) {
    float delta = r_c2 - r_s2;
    float inv_delta = 1.0f / delta;
    float tau = (r2 - r_s2) * inv_delta;
    float one_minus_tau = 1.0f - tau;
    float s = one_minus_tau * one_minus_tau * (1.0f + 2.0f * tau);
    // chain_coeff = -2 * dS/d(r^2) = 12 * tau * (1 - tau) / delta
    float chain_coeff = 12.0f * tau * one_minus_tau * inv_delta;
    factor = s * factor + chain_coeff * energy;
    energy = s * energy;
  }

  float fx = factor * dx;
  float fy = factor * dy;
  float fz = factor * dz;
  float w  = fx * dx + fy * dy + fz * dz;

  float scale = exclusion_scale(
      i, j, atom_excl_offsets, atom_excl_partners, atom_excl_coul_scales);
  fx *= scale;
  fy *= scale;
  fz *= scale;
  energy *= scale;
  w *= scale;

  pair_forces_x[slot] = fx;
  pair_forces_y[slot] = fy;
  pair_forces_z[slot] = fz;
  pair_energies[slot] = energy * 0.5f;
  pair_virials[slot]  = w * 0.5f;
}
