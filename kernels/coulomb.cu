// rq-846bdb8b

#include "exclusions.cuh"
#include "pair_frame.cuh"

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

  float r_c2 = cutoff * cutoff;
  if (f.r2 > r_c2) {
    pair_frame_write_zero(f.slot,
        pair_forces_x, pair_forces_y, pair_forces_z,
        pair_energies, pair_virials);
    return;
  }

  float inv_r2 = 1.0f / f.r2;
  float inv_r  = sqrtf(inv_r2);
  float qq     = qi * qj;
  float energy = k_coulomb * qq * inv_r;
  // The Coulomb force on i due to j is F = k_C * q_i * q_j * r_ij / r^3.
  // factor = k_C * q_i * q_j / r^3, so force_x = factor * dx, etc.
  float factor = k_coulomb * qq * inv_r * inv_r2;

  // CHARMM-style C1 switching function applied over [r_switch, r_cut].
  // Identical structure to lj_pair_force; see lj-pair-force.md.
  float r_s2 = r_switch * r_switch;
  if (f.r2 > r_s2) {
    float delta = r_c2 - r_s2;
    float inv_delta = 1.0f / delta;
    float tau = (f.r2 - r_s2) * inv_delta;
    float one_minus_tau = 1.0f - tau;
    float s = one_minus_tau * one_minus_tau * (1.0f + 2.0f * tau);
    // chain_coeff = -2 * dS/d(r^2) = 12 * tau * (1 - tau) / delta
    float chain_coeff = 12.0f * tau * one_minus_tau * inv_delta;
    factor = s * factor + chain_coeff * energy;
    energy = s * energy;
  }

  float fx = factor * f.dx;
  float fy = factor * f.dy;
  float fz = factor * f.dz;
  float w  = fx * f.dx + fy * f.dy + fz * f.dz;

  float scale = exclusion_scale(
      f.i, f.j, atom_excl_offsets, atom_excl_partners, atom_excl_coul_scales);
  pair_frame_write(
      f.slot, fx, fy, fz, energy, w, scale,
      pair_forces_x, pair_forces_y, pair_forces_z,
      pair_energies, pair_virials);
}
