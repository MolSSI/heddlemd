// rq-4ddab3c7

#include "precision.cuh"
#include "pair_compute.cuh"

struct LjPairFunc {
  const unsigned int *type_indices;
  unsigned int n_types;
  const Real *type_sigma;
  const Real *type_epsilon;
  const Real *type_cutoff;
  const Real *type_switch;

  __device__ inline unsigned int slot(unsigned int i, unsigned int j) const {
    unsigned int ti = type_indices[i];
    unsigned int tj = type_indices[j];
    return ti * n_types + tj;
  }

  __device__ inline Real cutoff_squared(unsigned int i, unsigned int j) const {
    Real c = type_cutoff[slot(i, j)];
    return c * c;
  }

  __device__ inline void evaluate(
      Real r2, unsigned int i, unsigned int j,
      Real &factor, Real &energy, Real &virial) const
  {
    unsigned int p = slot(i, j);
    Real sigma = type_sigma[p];
    Real epsilon = type_epsilon[p];
    Real cutoff = type_cutoff[p];
    Real r_switch = type_switch[p];

    Real inv_r2 = R(1.0) / r2;
    Real sigma2 = sigma * sigma;
    Real sr2 = sigma2 * inv_r2;
    Real sr6 = sr2 * sr2 * sr2;
    Real sr12 = sr6 * sr6;
    factor = R(24.0) * epsilon * inv_r2 * (R(2.0) * sr12 - sr6);
    energy = R(4.0) * epsilon * (sr12 - sr6);

    // CHARMM-style C1 switching function applied over [r_switch, r_cut].
    Real r_s2 = r_switch * r_switch;
    if (r2 > r_s2) {
      Real r_c2 = cutoff * cutoff;
      Real delta = r_c2 - r_s2;
      Real inv_delta = R(1.0) / delta;
      Real tau = (r2 - r_s2) * inv_delta;
      Real one_minus_tau = R(1.0) - tau;
      Real s = one_minus_tau * one_minus_tau * (R(1.0) + R(2.0) * tau);
      Real chain_coeff = R(12.0) * tau * one_minus_tau * inv_delta;
      factor = s * factor + chain_coeff * energy;
      energy = s * energy;
    }

    virial = factor * r2;
  }
};

extern "C" __global__ void lj_pair_force_f(
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const unsigned int *type_indices,
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
    Real *slot_force_x,
    Real *slot_force_y,
    Real *slot_force_z,
    unsigned int n)
{
  LjPairFunc f { type_indices, n_types, type_sigma, type_epsilon,
                 type_cutoff, type_switch };
  pair_compute_f(
      f, n, max_neighbors,
      positions_x, positions_y, positions_z,
      neighbor_list, neighbor_counts,
      lx, ly, lz, xy, xz, yz,
      atom_excl_offsets, atom_excl_partners, atom_excl_lj_scales,
      slot_force_x, slot_force_y, slot_force_z);
}

extern "C" __global__ void lj_pair_force_fev(
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const unsigned int *type_indices,
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
    Real *slot_force_x,
    Real *slot_force_y,
    Real *slot_force_z,
    Real *slot_energy,
    Real *slot_virial,
    unsigned int n)
{
  LjPairFunc f { type_indices, n_types, type_sigma, type_epsilon,
                 type_cutoff, type_switch };
  pair_compute_fev(
      f, n, max_neighbors,
      positions_x, positions_y, positions_z,
      neighbor_list, neighbor_counts,
      lx, ly, lz, xy, xz, yz,
      atom_excl_offsets, atom_excl_partners, atom_excl_lj_scales,
      slot_force_x, slot_force_y, slot_force_z,
      slot_energy, slot_virial);
}
