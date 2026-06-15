// rq-73c4d574
//
// Device-side helper used by every pair-force kernel to write into the
// shared pair buffer. Centralises the universal protocol — thread→slot
// mapping, the three skip-the-pair guards, the displacement and minimum-
// image reduction, the exclusion-scale apply, and the per-pair R(0.5)
// halving — so each pair-force kernel reduces to (1) a setup call, (2)
// the per-potential cutoff test and pair functional form, and (3) a
// write call. Declares no kernels of its own; nvcc inlines every
// function into the translation unit that `#include`s it.

#include "precision.cuh"

#pragma once

#include "pbc.cuh"

struct PairFrame {
  bool active;
  unsigned int i;
  unsigned int j;
  unsigned int slot;
  Real dx;
  Real dy;
  Real dz;
  Real r2;
};

__device__ static inline void pair_frame_write_zero(
    unsigned int slot,
    Real *pair_forces_x,
    Real *pair_forces_y,
    Real *pair_forces_z,
    Real *pair_energies,
    Real *pair_virials)
{
  pair_forces_x[slot] = R(0.0);
  pair_forces_y[slot] = R(0.0);
  pair_forces_z[slot] = R(0.0);
  pair_energies[slot] = R(0.0);
  pair_virials[slot]  = R(0.0);
}

__device__ static inline PairFrame pair_frame_setup(
    unsigned int n,
    unsigned int max_neighbors,
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    Real lx, Real ly, Real lz,
    Real xy, Real xz, Real yz,
    Real *pair_forces_x,
    Real *pair_forces_y,
    Real *pair_forces_z,
    Real *pair_energies,
    Real *pair_virials)
{
  PairFrame f;
  f.active = false;
  unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || k >= max_neighbors) {
    // Out-of-grid thread: no slot to write.
    return f;
  }
  f.slot = i * max_neighbors + k;
  if (k >= neighbor_counts[i]) {
    pair_frame_write_zero(f.slot,
        pair_forces_x, pair_forces_y, pair_forces_z,
        pair_energies, pair_virials);
    return f;
  }
  unsigned int j = neighbor_list[f.slot];
  if (i == j) {
    pair_frame_write_zero(f.slot,
        pair_forces_x, pair_forces_y, pair_forces_z,
        pair_energies, pair_virials);
    return f;
  }
  Real dx = positions_x[i] - positions_x[j];
  Real dy = positions_y[i] - positions_y[j];
  Real dz = positions_z[i] - positions_z[j];
  triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);
  Real r2 = dx * dx + dy * dy + dz * dz;
  f.active = true;
  f.i = i;
  f.j = j;
  f.dx = dx;
  f.dy = dy;
  f.dz = dz;
  f.r2 = r2;
  return f;
}

// Applies `scale` to all five inputs, then halves `energy` and `virial`
// by R(0.5), and writes the results into the five pair-buffer slots at
// `slot`. The halving distributes each pair's energy and virial across
// its (i, j) and (j, i) slots so the segmented reduction counts each
// pair exactly once when summed over all particles.
__device__ static inline void pair_frame_write(
    unsigned int slot,
    Real fx, Real fy, Real fz,
    Real energy,
    Real virial,
    Real scale,
    Real *pair_forces_x,
    Real *pair_forces_y,
    Real *pair_forces_z,
    Real *pair_energies,
    Real *pair_virials)
{
  fx *= scale;
  fy *= scale;
  fz *= scale;
  energy *= scale;
  virial *= scale;
  pair_forces_x[slot] = fx;
  pair_forces_y[slot] = fy;
  pair_forces_z[slot] = fz;
  pair_energies[slot] = energy * R(0.5);
  pair_virials[slot]  = virial * R(0.5);
}
