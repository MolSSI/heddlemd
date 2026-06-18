// rq-79282483

#include "precision.cuh"
#include "pbc.cuh"
#include "exclusions.cuh"

#define LJ_SPME_FUSED_WARP_SIZE 32
#define LJ_SPME_FUSED_WARPS_PER_BLOCK 8
#define LJ_SPME_FUSED_BLOCK_SIZE \
  (LJ_SPME_FUSED_WARP_SIZE * LJ_SPME_FUSED_WARPS_PER_BLOCK)

__device__ static const Real LJSF_ONE_OVER_SQRT_PI = R(0.5641895835477563);

__device__ static inline Real ljsf_warp_reduce_sum(Real v) {
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v;
}

__device__ static inline void lj_pair_evaluate(
    Real r2,
    Real sigma, Real epsilon, Real cutoff, Real r_switch,
    Real &factor, Real &energy, Real &virial)
{
  Real inv_r2 = R(1.0) / r2;
  Real sigma2 = sigma * sigma;
  Real sr2 = sigma2 * inv_r2;
  Real sr6 = sr2 * sr2 * sr2;
  Real sr12 = sr6 * sr6;
  factor = R(24.0) * epsilon * inv_r2 * (R(2.0) * sr12 - sr6);
  energy = R(4.0) * epsilon * (sr12 - sr6);

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

__device__ static inline void spme_real_pair_evaluate(
    Real r2, Real qq, Real k_coulomb, Real alpha,
    Real &factor, Real &energy, Real &virial)
{
  Real inv_r2 = R(1.0) / r2;
  Real r = Real_sqrt(r2);
  Real inv_r = R(1.0) / r;
  Real ar = alpha * r;
  Real erfc_ar = erfcf(ar);
  Real gauss = Real_exp(-(ar * ar));
  energy = k_coulomb * qq * erfc_ar * inv_r;
  factor = k_coulomb * qq * inv_r2
          * (erfc_ar * inv_r + R(2.0) * alpha * LJSF_ONE_OVER_SQRT_PI * gauss);
  virial = factor * r2;
}

template <bool WriteEv>
__device__ static inline void lj_spme_real_fused_pair_compute(
    unsigned int n,
    unsigned int max_neighbors,
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const unsigned int *type_indices,
    unsigned int n_types,
    const Real *type_sigma,
    const Real *type_epsilon,
    const Real *type_cutoff,
    const Real *type_switch,
    const Real *charges,
    Real k_coulomb,
    Real alpha,
    Real r_cut_spme_real,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    Real lx, Real ly, Real lz,
    Real xy, Real xz, Real yz,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const Real *atom_excl_lj_scales,
    const Real *atom_excl_coul_scales,
    Real *slot_force_x,
    Real *slot_force_y,
    Real *slot_force_z,
    Real *slot_energy,
    Real *slot_virial)
{
  unsigned int warp_id_in_block = threadIdx.x / LJ_SPME_FUSED_WARP_SIZE;
  unsigned int lane = threadIdx.x & (LJ_SPME_FUSED_WARP_SIZE - 1);
  unsigned int i = blockIdx.x * LJ_SPME_FUSED_WARPS_PER_BLOCK + warp_id_in_block;
  if (i >= n) {
    return;
  }

  unsigned int count = neighbor_counts[i];
  unsigned int row_base = i * max_neighbors;
  unsigned int sweep_end =
      ((count + LJ_SPME_FUSED_WARP_SIZE - 1) / LJ_SPME_FUSED_WARP_SIZE)
      * LJ_SPME_FUSED_WARP_SIZE;

  Real p_x = R(0.0);
  Real p_y = R(0.0);
  Real p_z = R(0.0);
  Real p_e = R(0.0);
  Real p_w = R(0.0);

  Real pi_x = positions_x[i];
  Real pi_y = positions_y[i];
  Real pi_z = positions_z[i];

  unsigned int ti = type_indices[i];
  Real qi = charges[i];

  Real r_c2_spme = r_cut_spme_real * r_cut_spme_real;

  for (unsigned int s = 0; s < sweep_end; s += LJ_SPME_FUSED_WARP_SIZE) {
    unsigned int k = s + lane;
    if (k < count) {
      unsigned int j = neighbor_list[row_base + k];
      if (i != j) {
        Real dx = pi_x - positions_x[j];
        Real dy = pi_y - positions_y[j];
        Real dz = pi_z - positions_z[j];
        triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);
        Real r2 = dx * dx + dy * dy + dz * dz;

        Real factor = R(0.0);
        Real energy = R(0.0);
        Real virial = R(0.0);

        unsigned int tj = type_indices[j];
        unsigned int p = ti * n_types + tj;
        Real cutoff_lj = type_cutoff[p];
        Real r_c2_lj = cutoff_lj * cutoff_lj;
        if (r2 <= r_c2_lj) {
          Real lj_factor, lj_energy, lj_virial;
          lj_pair_evaluate(
              r2, type_sigma[p], type_epsilon[p], cutoff_lj, type_switch[p],
              lj_factor, lj_energy, lj_virial);
          Real lj_scale = exclusion_scale(
              i, j, atom_excl_offsets, atom_excl_partners, atom_excl_lj_scales);
          factor += lj_factor * lj_scale;
          energy += lj_energy * lj_scale;
          virial += lj_virial * lj_scale;
        }

        if (r2 <= r_c2_spme) {
          Real qq = qi * charges[j];
          Real sp_factor, sp_energy, sp_virial;
          spme_real_pair_evaluate(
              r2, qq, k_coulomb, alpha, sp_factor, sp_energy, sp_virial);
          Real coul_scale = exclusion_scale(
              i, j, atom_excl_offsets, atom_excl_partners, atom_excl_coul_scales);
          factor += sp_factor * coul_scale;
          energy += sp_energy * coul_scale;
          virial += sp_virial * coul_scale;
        }

        p_x += factor * dx;
        p_y += factor * dy;
        p_z += factor * dz;
        if (WriteEv) {
          p_e += energy * R(0.5);
          p_w += virial * R(0.5);
        }
      }
    }
  }

  p_x = ljsf_warp_reduce_sum(p_x);
  p_y = ljsf_warp_reduce_sum(p_y);
  p_z = ljsf_warp_reduce_sum(p_z);
  if (WriteEv) {
    p_e = ljsf_warp_reduce_sum(p_e);
    p_w = ljsf_warp_reduce_sum(p_w);
  }

  if (lane == 0) {
    slot_force_x[i] += p_x;
    slot_force_y[i] += p_y;
    slot_force_z[i] += p_z;
    if (WriteEv) {
      slot_energy[i] += p_e;
      slot_virial[i] += p_w;
    }
  }
}

extern "C" __global__ void lj_spme_real_fused_pair_force_f(
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const unsigned int *type_indices,
    unsigned int max_neighbors,
    const Real *lattice,
    unsigned int n_types,
    const Real *type_sigma,
    const Real *type_epsilon,
    const Real *type_cutoff,
    const Real *type_switch,
    const Real *charges,
    Real k_coulomb,
    Real alpha,
    Real r_cut_spme_real,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const Real *atom_excl_lj_scales,
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
  lj_spme_real_fused_pair_compute<false>(
      n, max_neighbors,
      positions_x, positions_y, positions_z,
      type_indices, n_types,
      type_sigma, type_epsilon, type_cutoff, type_switch,
      charges, k_coulomb, alpha, r_cut_spme_real,
      neighbor_list, neighbor_counts,
      lx, ly, lz, xy, xz, yz,
      atom_excl_offsets, atom_excl_partners,
      atom_excl_lj_scales, atom_excl_coul_scales,
      slot_force_x, slot_force_y, slot_force_z,
      nullptr, nullptr);
}

extern "C" __global__ void lj_spme_real_fused_pair_force_fev(
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const unsigned int *type_indices,
    unsigned int max_neighbors,
    const Real *lattice,
    unsigned int n_types,
    const Real *type_sigma,
    const Real *type_epsilon,
    const Real *type_cutoff,
    const Real *type_switch,
    const Real *charges,
    Real k_coulomb,
    Real alpha,
    Real r_cut_spme_real,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const Real *atom_excl_lj_scales,
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
  lj_spme_real_fused_pair_compute<true>(
      n, max_neighbors,
      positions_x, positions_y, positions_z,
      type_indices, n_types,
      type_sigma, type_epsilon, type_cutoff, type_switch,
      charges, k_coulomb, alpha, r_cut_spme_real,
      neighbor_list, neighbor_counts,
      lx, ly, lz, xy, xz, yz,
      atom_excl_offsets, atom_excl_partners,
      atom_excl_lj_scales, atom_excl_coul_scales,
      slot_force_x, slot_force_y, slot_force_z,
      slot_energy, slot_virial);
}
