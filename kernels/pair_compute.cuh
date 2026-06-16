// rq-2adca0ab
//
// Device-side helper that implements the warp-per-particle pair-force
// pattern shared by every fused pair-force kernel (lj_pair_force_*,
// coulomb_pair_force_*, spme_real_pair_force_*).
//
// Each warp handles one particle: 32 lanes sweep the particle's
// neighbour list with stride 32, evaluate the per-pair functional form
// supplied by the caller, accumulate the per-component force (and
// optionally energy / virial) in register accumulators, and reduce to
// lane 0 via a 5-step `__shfl_xor_sync` butterfly. Lane 0 writes the
// per-particle result to the slot output buffer.
//
// The per-potential `PairFunc` is a stateless `__device__` functor
// invoked once per in-cutoff pair. Its `evaluate(...)` member computes
// `(factor, energy, virial)` for the pair; `cutoff_squared(...)`
// returns the per-pair cutoff^2 used for the early-out test. The two
// pieces are split so the cutoff test happens before the per-pair
// math (which can be costly).

#pragma once

#include "precision.cuh"
#include "pbc.cuh"
#include "exclusions.cuh"

#define PAIR_FORCE_WARP_SIZE 32
#define PAIR_FORCE_WARPS_PER_BLOCK 8
#define PAIR_FORCE_BLOCK_SIZE (PAIR_FORCE_WARP_SIZE * PAIR_FORCE_WARPS_PER_BLOCK)

__device__ static inline Real pair_force_warp_reduce_sum(Real v) {
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v;
}

// Forces-only variant.
//
// `PairFunc` must expose:
//   `Real cutoff_squared(unsigned int ti, unsigned int tj)` — cutoff^2
//       for the pair-type indices (or just the global cutoff for
//       Coulomb / SPME-real).
//   `void evaluate(Real r2, unsigned int ti, unsigned int tj,
//                  Real &factor, Real &energy, Real &virial)` — fills
//       the three pair outputs. The `_f` variant ignores `energy` and
//       `virial`; the compiler dead-code-eliminates the unused stores.
//   `Real type_index(unsigned int i, const PerPotentialState &s)` — type
//       resolution if needed; passed through `state`.
//
// For potentials that do not have per-type parameters (Coulomb,
// SPME-real), the functor can ignore `ti` / `tj`.
template <typename PairFunc>
__device__ static inline void pair_compute_f(
    PairFunc per_pair,
    unsigned int n,
    unsigned int max_neighbors,
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    Real lx, Real ly, Real lz,
    Real xy, Real xz, Real yz,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const Real *atom_excl_scales,
    Real *slot_force_x,
    Real *slot_force_y,
    Real *slot_force_z)
{
  unsigned int warp_id_in_block = threadIdx.x / PAIR_FORCE_WARP_SIZE;
  unsigned int lane = threadIdx.x & (PAIR_FORCE_WARP_SIZE - 1);
  unsigned int i = blockIdx.x * PAIR_FORCE_WARPS_PER_BLOCK + warp_id_in_block;
  if (i >= n) {
    // All 32 lanes of this warp return together — `i` is uniform within
    // the warp, so `__shfl_xor_sync` below would never be reached with
    // a partial warp.
    return;
  }

  unsigned int count = neighbor_counts[i];
  unsigned int row_base = i * max_neighbors;
  unsigned int sweep_end =
      ((count + PAIR_FORCE_WARP_SIZE - 1) / PAIR_FORCE_WARP_SIZE) * PAIR_FORCE_WARP_SIZE;

  Real p_x = R(0.0);
  Real p_y = R(0.0);
  Real p_z = R(0.0);

  Real pi_x = positions_x[i];
  Real pi_y = positions_y[i];
  Real pi_z = positions_z[i];

  for (unsigned int s = 0; s < sweep_end; s += PAIR_FORCE_WARP_SIZE) {
    unsigned int k = s + lane;
    if (k < count) {
      unsigned int j = neighbor_list[row_base + k];
      if (i != j) {
        Real dx = pi_x - positions_x[j];
        Real dy = pi_y - positions_y[j];
        Real dz = pi_z - positions_z[j];
        triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);
        Real r2 = dx * dx + dy * dy + dz * dz;
        Real r_c2 = per_pair.cutoff_squared(i, j);
        if (r2 <= r_c2) {
          Real factor, energy_unused, virial_unused;
          per_pair.evaluate(r2, i, j, factor, energy_unused, virial_unused);
          Real fx = factor * dx;
          Real fy = factor * dy;
          Real fz = factor * dz;
          Real scale = exclusion_scale(
              i, j, atom_excl_offsets, atom_excl_partners, atom_excl_scales);
          p_x = p_x + fx * scale;
          p_y = p_y + fy * scale;
          p_z = p_z + fz * scale;
        }
      }
    }
  }

  p_x = pair_force_warp_reduce_sum(p_x);
  p_y = pair_force_warp_reduce_sum(p_y);
  p_z = pair_force_warp_reduce_sum(p_z);

  if (lane == 0) {
    slot_force_x[i] = p_x;
    slot_force_y[i] = p_y;
    slot_force_z[i] = p_z;
  }
}

// Forces + energy + virial variant. Accumulates the per-pair energy
// and scalar virial (halved by 0.5) alongside the force, and writes
// all five per-particle quantities through lane 0.
template <typename PairFunc>
__device__ static inline void pair_compute_fev(
    PairFunc per_pair,
    unsigned int n,
    unsigned int max_neighbors,
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    Real lx, Real ly, Real lz,
    Real xy, Real xz, Real yz,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const Real *atom_excl_scales,
    Real *slot_force_x,
    Real *slot_force_y,
    Real *slot_force_z,
    Real *slot_energy,
    Real *slot_virial)
{
  unsigned int warp_id_in_block = threadIdx.x / PAIR_FORCE_WARP_SIZE;
  unsigned int lane = threadIdx.x & (PAIR_FORCE_WARP_SIZE - 1);
  unsigned int i = blockIdx.x * PAIR_FORCE_WARPS_PER_BLOCK + warp_id_in_block;
  if (i >= n) {
    return;
  }

  unsigned int count = neighbor_counts[i];
  unsigned int row_base = i * max_neighbors;
  unsigned int sweep_end =
      ((count + PAIR_FORCE_WARP_SIZE - 1) / PAIR_FORCE_WARP_SIZE) * PAIR_FORCE_WARP_SIZE;

  Real p_x = R(0.0);
  Real p_y = R(0.0);
  Real p_z = R(0.0);
  Real p_e = R(0.0);
  Real p_w = R(0.0);

  Real pi_x = positions_x[i];
  Real pi_y = positions_y[i];
  Real pi_z = positions_z[i];

  for (unsigned int s = 0; s < sweep_end; s += PAIR_FORCE_WARP_SIZE) {
    unsigned int k = s + lane;
    if (k < count) {
      unsigned int j = neighbor_list[row_base + k];
      if (i != j) {
        Real dx = pi_x - positions_x[j];
        Real dy = pi_y - positions_y[j];
        Real dz = pi_z - positions_z[j];
        triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);
        Real r2 = dx * dx + dy * dy + dz * dz;
        Real r_c2 = per_pair.cutoff_squared(i, j);
        if (r2 <= r_c2) {
          Real factor, energy, virial;
          per_pair.evaluate(r2, i, j, factor, energy, virial);
          Real fx = factor * dx;
          Real fy = factor * dy;
          Real fz = factor * dz;
          Real scale = exclusion_scale(
              i, j, atom_excl_offsets, atom_excl_partners, atom_excl_scales);
          p_x = p_x + fx * scale;
          p_y = p_y + fy * scale;
          p_z = p_z + fz * scale;
          p_e = p_e + energy * scale * R(0.5);
          p_w = p_w + virial * scale * R(0.5);
        }
      }
    }
  }

  p_x = pair_force_warp_reduce_sum(p_x);
  p_y = pair_force_warp_reduce_sum(p_y);
  p_z = pair_force_warp_reduce_sum(p_z);
  p_e = pair_force_warp_reduce_sum(p_e);
  p_w = pair_force_warp_reduce_sum(p_w);

  if (lane == 0) {
    slot_force_x[i] = p_x;
    slot_force_y[i] = p_y;
    slot_force_z[i] = p_z;
    slot_energy[i] = p_e;
    slot_virial[i] = p_w;
  }
}
