// rq-31bd2eee

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)
#include "precision.cuh"

__device__ static inline Real warp_reduce_sum(Real v) {
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v;
}

extern "C" __global__ void reduce_pair_forces(
    const Real *pair_forces_x,
    const Real *pair_forces_y,
    const Real *pair_forces_z,
    const unsigned int *neighbor_counts,
    unsigned int max_neighbors,
    Real *net_forces_x,
    Real *net_forces_y,
    Real *net_forces_z,
    unsigned int n)
{
  unsigned int warp_id_in_block = threadIdx.x / WARP_SIZE;
  unsigned int lane = threadIdx.x & (WARP_SIZE - 1);
  unsigned int i = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
  if (i >= n) {
    // Every lane in this warp evaluates the same condition (uniform on
    // blockIdx.x, warp_id_in_block, n), so all 32 lanes return together
    // and no __shfl_xor_sync below is reached with a partial warp.
    return;
  }

  unsigned int count = neighbor_counts[i];
  unsigned int row_base = i * max_neighbors;
  unsigned int sweep_end = ((count + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  Real p_x = R(0.0);
  Real p_y = R(0.0);
  Real p_z = R(0.0);

  for (unsigned int s = 0; s < sweep_end; s += WARP_SIZE) {
    unsigned int k = s + lane;
    if (k < count) {
      unsigned int idx = row_base + k;
      p_x = p_x + pair_forces_x[idx];
      p_y = p_y + pair_forces_y[idx];
      p_z = p_z + pair_forces_z[idx];
    }
  }

  p_x = warp_reduce_sum(p_x);
  p_y = warp_reduce_sum(p_y);
  p_z = warp_reduce_sum(p_z);

  if (lane == 0) {
    net_forces_x[i] = p_x;
    net_forces_y[i] = p_y;
    net_forces_z[i] = p_z;
  }
}

extern "C" __global__ void reduce_pair_energy_virial(
    const Real *pair_energies,
    const Real *pair_virials,
    const unsigned int *neighbor_counts,
    unsigned int max_neighbors,
    Real *net_energy,
    Real *net_virial,
    unsigned int n)
{
  unsigned int warp_id_in_block = threadIdx.x / WARP_SIZE;
  unsigned int lane = threadIdx.x & (WARP_SIZE - 1);
  unsigned int i = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
  if (i >= n) {
    return;
  }

  unsigned int count = neighbor_counts[i];
  unsigned int row_base = i * max_neighbors;
  unsigned int sweep_end = ((count + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

  Real p_e = R(0.0);
  Real p_w = R(0.0);

  for (unsigned int s = 0; s < sweep_end; s += WARP_SIZE) {
    unsigned int k = s + lane;
    if (k < count) {
      unsigned int idx = row_base + k;
      p_e = p_e + pair_energies[idx];
      p_w = p_w + pair_virials[idx];
    }
  }

  p_e = warp_reduce_sum(p_e);
  p_w = warp_reduce_sum(p_w);

  if (lane == 0) {
    net_energy[i] = p_e;
    net_virial[i] = p_w;
  }
}
