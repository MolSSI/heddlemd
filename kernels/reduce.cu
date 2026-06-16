// rq-31bd2eee

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
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
  unsigned int i = blockIdx.x;
  if (i >= n) {
    return;
  }

  unsigned int count = neighbor_counts[i];
  unsigned int row_base = i * max_neighbors;
  unsigned int sweep_end = ((count + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

  Real p_x = R(0.0);
  Real p_y = R(0.0);
  Real p_z = R(0.0);

  for (unsigned int s = 0; s < sweep_end; s += BLOCK_SIZE) {
    unsigned int k = s + threadIdx.x;
    bool active = (k < count);
    if (active) {
      unsigned int idx = row_base + k;
      p_x = p_x + pair_forces_x[idx];
      p_y = p_y + pair_forces_y[idx];
      p_z = p_z + pair_forces_z[idx];
    }
  }

  p_x = warp_reduce_sum(p_x);
  p_y = warp_reduce_sum(p_y);
  p_z = warp_reduce_sum(p_z);

  __shared__ Real warp_partials[NUM_WARPS][3];

  unsigned int lane = threadIdx.x & (WARP_SIZE - 1);
  unsigned int warp_id = threadIdx.x / WARP_SIZE;

  if (lane == 0) {
    warp_partials[warp_id][0] = p_x;
    warp_partials[warp_id][1] = p_y;
    warp_partials[warp_id][2] = p_z;
  }
  __syncthreads();

  if (warp_id == 0) {
    Real q_x = (lane < NUM_WARPS) ? warp_partials[lane][0] : R(0.0);
    Real q_y = (lane < NUM_WARPS) ? warp_partials[lane][1] : R(0.0);
    Real q_z = (lane < NUM_WARPS) ? warp_partials[lane][2] : R(0.0);

    q_x = warp_reduce_sum(q_x);
    q_y = warp_reduce_sum(q_y);
    q_z = warp_reduce_sum(q_z);

    if (lane == 0) {
      net_forces_x[i] = q_x;
      net_forces_y[i] = q_y;
      net_forces_z[i] = q_z;
    }
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
  unsigned int i = blockIdx.x;
  if (i >= n) {
    return;
  }

  unsigned int count = neighbor_counts[i];
  unsigned int row_base = i * max_neighbors;
  unsigned int sweep_end = ((count + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

  Real p_e = R(0.0);
  Real p_w = R(0.0);

  for (unsigned int s = 0; s < sweep_end; s += BLOCK_SIZE) {
    unsigned int k = s + threadIdx.x;
    bool active = (k < count);
    if (active) {
      unsigned int idx = row_base + k;
      p_e = p_e + pair_energies[idx];
      p_w = p_w + pair_virials[idx];
    }
  }

  p_e = warp_reduce_sum(p_e);
  p_w = warp_reduce_sum(p_w);

  __shared__ Real warp_partials[NUM_WARPS][2];

  unsigned int lane = threadIdx.x & (WARP_SIZE - 1);
  unsigned int warp_id = threadIdx.x / WARP_SIZE;

  if (lane == 0) {
    warp_partials[warp_id][0] = p_e;
    warp_partials[warp_id][1] = p_w;
  }
  __syncthreads();

  if (warp_id == 0) {
    Real q_e = (lane < NUM_WARPS) ? warp_partials[lane][0] : R(0.0);
    Real q_w = (lane < NUM_WARPS) ? warp_partials[lane][1] : R(0.0);

    q_e = warp_reduce_sum(q_e);
    q_w = warp_reduce_sum(q_w);

    if (lane == 0) {
      net_energy[i] = q_e;
      net_virial[i] = q_w;
    }
  }
}
