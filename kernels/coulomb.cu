// rq-846bdb8b

#include "precision.cuh"
#include "pair_compute.cuh"

// rq-bfd7004c
struct CoulombPairFunc {
  Real k_coulomb;
  Real cutoff;
  Real r_switch;

  __device__ inline Real cutoff_squared(unsigned int, unsigned int) const {
    return cutoff * cutoff;
  }

  __device__ inline void evaluate(
      Real r2, Real qi, Real qj, unsigned int i, unsigned int j,
      Real &factor, Real &energy, Real &virial) const
  {
    Real qq = qi * qj;

    Real inv_r2 = R(1.0) / r2;
    Real inv_r  = Real_sqrt(inv_r2);
    energy = k_coulomb * qq * inv_r;
    factor = k_coulomb * qq * inv_r * inv_r2;

    // CHARMM-style C1 switching function over [r_switch, cutoff].
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

extern "C" __global__ void coulomb_pair_force_f(
    const Real4 *posq,
    unsigned int max_neighbors,
    const Real *lattice,
    Real k_coulomb,
    Real cutoff,
    Real r_switch,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const Real *atom_excl_coul_scales,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    Real *slot_force_x,
    Real *slot_force_y,
    Real *slot_force_z,
    unsigned int n)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  CoulombPairFunc f { k_coulomb, cutoff, r_switch };
  pair_compute_f(
      f, n, max_neighbors,
      posq,
      neighbor_list, neighbor_counts,
      lx, ly, lz, xy, xz, yz,
      atom_excl_offsets, atom_excl_partners, atom_excl_coul_scales,
      slot_force_x, slot_force_y, slot_force_z);
}

extern "C" __global__ void coulomb_pair_force_fev(
    const Real4 *posq,
    unsigned int max_neighbors,
    const Real *lattice,
    Real k_coulomb,
    Real cutoff,
    Real r_switch,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const Real *atom_excl_coul_scales,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    Real *slot_force_x,
    Real *slot_force_y,
    Real *slot_force_z,
    Real *slot_energy,
    Real *slot_virial,
    unsigned int n)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  CoulombPairFunc f { k_coulomb, cutoff, r_switch };
  pair_compute_fev(
      f, n, max_neighbors,
      posq,
      neighbor_list, neighbor_counts,
      lx, ly, lz, xy, xz, yz,
      atom_excl_offsets, atom_excl_partners, atom_excl_coul_scales,
      slot_force_x, slot_force_y, slot_force_z,
      slot_energy, slot_virial);
}
