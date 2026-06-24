// rq-f6d45062 rq-39b05bc9

#include "precision.cuh"
#include "pair_compute.cuh"

__device__ static const Real ONE_OVER_SQRT_PI = R(0.5641895835477563);

// rq-9a512ed1
struct SpmeRealPairFunc {
  Real k_coulomb;
  Real alpha;
  Real r_cut_real;

  __device__ inline Real cutoff_squared(unsigned int, unsigned int) const {
    return r_cut_real * r_cut_real;
  }

  __device__ inline void evaluate(
      Real r2, Real qi, Real qj, unsigned int i, unsigned int j,
      Real &factor, Real &energy, Real &virial) const
  {
    Real qq = qi * qj;

    Real inv_r2 = R(1.0) / r2;
    Real r = Real_sqrt(r2);
    Real inv_r = R(1.0) / r;
    Real ar = alpha * r;
    Real erfc_ar = erfcf(ar);
    Real gauss = Real_exp(-(ar * ar));
    energy = k_coulomb * qq * erfc_ar * inv_r;
    factor = k_coulomb * qq * inv_r2
            * (erfc_ar * inv_r + R(2.0) * alpha * ONE_OVER_SQRT_PI * gauss);

    virial = factor * r2;
  }
};

extern "C" __global__ void spme_real_pair_force_f(
    const Real4 *posq,
    unsigned int max_neighbors,
    const Real *lattice,
    Real k_coulomb,
    Real alpha,
    Real r_cut_real,
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
  SpmeRealPairFunc f { k_coulomb, alpha, r_cut_real };
  pair_compute_f(
      f, n, max_neighbors,
      posq,
      neighbor_list, neighbor_counts,
      lx, ly, lz, xy, xz, yz,
      atom_excl_offsets, atom_excl_partners, atom_excl_coul_scales,
      slot_force_x, slot_force_y, slot_force_z);
}

extern "C" __global__ void spme_real_pair_force_fev(
    const Real4 *posq,
    unsigned int max_neighbors,
    const Real *lattice,
    Real k_coulomb,
    Real alpha,
    Real r_cut_real,
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
  SpmeRealPairFunc f { k_coulomb, alpha, r_cut_real };
  pair_compute_fev(
      f, n, max_neighbors,
      posq,
      neighbor_list, neighbor_counts,
      lx, ly, lz, xy, xz, yz,
      atom_excl_offsets, atom_excl_partners, atom_excl_coul_scales,
      slot_force_x, slot_force_y, slot_force_z,
      slot_energy, slot_virial);
}
