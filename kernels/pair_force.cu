// rq-4ddab3c7

#include "precision.cuh"

#include "exclusions.cuh"
#include "pair_frame.cuh"

extern "C" __global__ void lj_pair_force(
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const unsigned int *type_indices,
    Real *pair_forces_x,
    Real *pair_forces_y,
    Real *pair_forces_z,
    Real *pair_energies,
    Real *pair_virials,
    unsigned int max_neighbors,
    Real lx, Real ly, Real lz, Real xy, Real xz, Real yz,
    unsigned int n_types,
    const Real *type_sigma,
    const Real *type_epsilon,
    const Real *type_cutoff,
    const Real *type_switch,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const Real *atom_excl_lj_scales,
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

  unsigned int ti = type_indices[f.i];
  unsigned int tj = type_indices[f.j];
  unsigned int p = ti * n_types + tj;
  Real sigma = type_sigma[p];
  Real epsilon = type_epsilon[p];
  Real cutoff = type_cutoff[p];
  Real r_switch = type_switch[p];

  Real r_c2 = cutoff * cutoff;
  if (f.r2 > r_c2) {
    pair_frame_write_zero(f.slot,
        pair_forces_x, pair_forces_y, pair_forces_z,
        pair_energies, pair_virials);
    return;
  }

  Real inv_r2 = R(1.0) / f.r2;
  Real sigma2 = sigma * sigma;
  Real sr2 = sigma2 * inv_r2;
  Real sr6 = sr2 * sr2 * sr2;
  Real sr12 = sr6 * sr6;
  Real factor = R(24.0) * epsilon * inv_r2 * (R(2.0) * sr12 - sr6);
  Real energy = R(4.0) * epsilon * (sr12 - sr6);

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
  Real r_s2 = r_switch * r_switch;
  if (f.r2 > r_s2) {
    Real delta = r_c2 - r_s2;
    Real inv_delta = R(1.0) / delta;
    Real tau = (f.r2 - r_s2) * inv_delta;
    Real one_minus_tau = R(1.0) - tau;
    Real s = one_minus_tau * one_minus_tau * (R(1.0) + R(2.0) * tau);
    // -2 * dS/d(r2) = 12 * tau * (1 - tau) / delta
    Real chain_coeff = R(12.0) * tau * one_minus_tau * inv_delta;
    factor = s * factor + chain_coeff * energy;
    energy = s * energy;
  }

  Real fx = factor * f.dx;
  Real fy = factor * f.dy;
  Real fz = factor * f.dz;
  Real w = fx * f.dx + fy * f.dy + fz * f.dz;

  // rq-dddcbf07
  Real scale = exclusion_scale(
      f.i, f.j, atom_excl_offsets, atom_excl_partners, atom_excl_lj_scales);
  pair_frame_write(
      f.slot, fx, fy, fz, energy, w, scale,
      pair_forces_x, pair_forces_y, pair_forces_z,
      pair_energies, pair_virials);
}
