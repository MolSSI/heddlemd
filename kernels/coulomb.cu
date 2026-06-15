// rq-846bdb8b

#include "precision.cuh"

#include "exclusions.cuh"
#include "pair_frame.cuh"

// rq-bfd7004c
extern "C" __global__ void coulomb_pair_force(
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
    Real cutoff,
    Real r_switch,
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

  Real r_c2 = cutoff * cutoff;
  if (f.r2 > r_c2) {
    pair_frame_write_zero(f.slot,
        pair_forces_x, pair_forces_y, pair_forces_z,
        pair_energies, pair_virials);
    return;
  }

  Real inv_r2 = R(1.0) / f.r2;
  Real inv_r  = Real_sqrt(inv_r2);
  Real qq     = qi * qj;
  Real energy = k_coulomb * qq * inv_r;
  // The Coulomb force on i due to j is F = k_C * q_i * q_j * r_ij / r^3.
  // factor = k_C * q_i * q_j / r^3, so force_x = factor * dx, etc.
  Real factor = k_coulomb * qq * inv_r * inv_r2;

  // CHARMM-style C1 switching function applied over [r_switch, r_cut].
  // Identical structure to lj_pair_force; see lj-pair-force.md.
  Real r_s2 = r_switch * r_switch;
  if (f.r2 > r_s2) {
    Real delta = r_c2 - r_s2;
    Real inv_delta = R(1.0) / delta;
    Real tau = (f.r2 - r_s2) * inv_delta;
    Real one_minus_tau = R(1.0) - tau;
    Real s = one_minus_tau * one_minus_tau * (R(1.0) + R(2.0) * tau);
    // chain_coeff = -2 * dS/d(r^2) = 12 * tau * (1 - tau) / delta
    Real chain_coeff = R(12.0) * tau * one_minus_tau * inv_delta;
    factor = s * factor + chain_coeff * energy;
    energy = s * energy;
  }

  Real fx = factor * f.dx;
  Real fy = factor * f.dy;
  Real fz = factor * f.dz;
  Real w  = fx * f.dx + fy * f.dy + fz * f.dz;

  Real scale = exclusion_scale(
      f.i, f.j, atom_excl_offsets, atom_excl_partners, atom_excl_coul_scales);
  pair_frame_write(
      f.slot, fx, fy, fz, energy, w, scale,
      pair_forces_x, pair_forces_y, pair_forces_z,
      pair_energies, pair_virials);
}
