// rq-4ddab3c7

#include "exclusions.cuh"

extern "C" __global__ void lj_pair_force(
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    const unsigned int *type_indices,
    float *pair_forces_x,
    float *pair_forces_y,
    float *pair_forces_z,
    float *pair_energies,
    float *pair_virials,
    unsigned int max_neighbors,
    float lx, float ly, float lz,
    unsigned int n_types,
    const float *type_sigma,
    const float *type_epsilon,
    const float *type_cutoff,
    const float *type_switch,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const float *atom_excl_scales,
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

  unsigned int ti = type_indices[i];
  unsigned int tj = type_indices[j];
  unsigned int p = ti * n_types + tj;
  float sigma = type_sigma[p];
  float epsilon = type_epsilon[p];
  float cutoff = type_cutoff[p];
  float r_switch = type_switch[p];

  float dx = positions_x[i] - positions_x[j];
  float dy = positions_y[i] - positions_y[j];
  float dz = positions_z[i] - positions_z[j];

  dx = dx - lx * floorf((dx + lx * 0.5f) / lx);
  dy = dy - ly * floorf((dy + ly * 0.5f) / ly);
  dz = dz - lz * floorf((dz + lz * 0.5f) / lz);

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
  float sigma2 = sigma * sigma;
  float sr2 = sigma2 * inv_r2;
  float sr6 = sr2 * sr2 * sr2;
  float sr12 = sr6 * sr6;
  float factor = 24.0f * epsilon * inv_r2 * (2.0f * sr12 - sr6);
  float energy = 4.0f * epsilon * (sr12 - sr6);

  // CHARMM-style C1 switching function applied over [r_switch, r_cut].
  // When r2 <= r_s2 the inner plateau has S = 1 and dS/d(r^2) = 0, so
  // factor and energy are unchanged. Otherwise r_s2 < r2 <= r_c2 (the
  // r2 > r_c2 case was gated above) and the polynomial branch runs.
  //
  // The polynomial is evaluated in normalised form
  //   tau = (r2 - r_s2) / delta,  delta = r_c2 - r_s2,  tau in [0, 1]
  //   S    = (1 - tau)^2 (1 + 2 tau)
  //   dS/d(r2) = -6 tau (1 - tau) / delta
  // which keeps the only place delta appears explicitly to 1/delta (not
  // 1/delta^3). At SI-scale lengths (r_c ~ 1e-9 m) the cubed form
  // underflows f32 even though 1/delta itself stays representable.
  //
  // The switch == cutoff degenerate case satisfies r_s2 == r_c2 and is
  // always handled by the first branch (the second branch is unreachable
  // because r2 > r_c2 is already gated), so no division by zero occurs.
  float r_s2 = r_switch * r_switch;
  if (r2 > r_s2) {
    float delta = r_c2 - r_s2;
    float inv_delta = 1.0f / delta;
    float tau = (r2 - r_s2) * inv_delta;
    float one_minus_tau = 1.0f - tau;
    float s = one_minus_tau * one_minus_tau * (1.0f + 2.0f * tau);
    // -2 * dS/d(r2) = 12 * tau * (1 - tau) / delta
    float chain_coeff = 12.0f * tau * one_minus_tau * inv_delta;
    factor = s * factor + chain_coeff * energy;
    energy = s * energy;
  }

  float fx = factor * dx;
  float fy = factor * dy;
  float fz = factor * dz;
  float w = fx * dx + fy * dy + fz * dz;

  // rq-dddcbf07
  float scale = exclusion_scale(
      i, j, atom_excl_offsets, atom_excl_partners, atom_excl_scales);
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
