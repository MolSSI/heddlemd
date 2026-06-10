// rq-31bd2eee

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

__device__ static inline float warp_reduce_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v;
}

extern "C" __global__ void reduce_pair_forces(
    const float *pair_forces_x,
    const float *pair_forces_y,
    const float *pair_forces_z,
    const float *pair_energies,
    const float *pair_virials,
    const unsigned int *neighbor_counts,
    unsigned int max_neighbors,
    float *net_forces_x,
    float *net_forces_y,
    float *net_forces_z,
    float *net_energy,
    float *net_virial,
    unsigned int n)
{
  unsigned int i = blockIdx.x;
  if (i >= n) {
    return;
  }

  unsigned int count = neighbor_counts[i];
  unsigned int row_base = i * max_neighbors;

  float p_x = 0.0f;
  float p_y = 0.0f;
  float p_z = 0.0f;
  float p_e = 0.0f;
  float p_w = 0.0f;

  for (unsigned int s = 0; s < max_neighbors; s += BLOCK_SIZE) {
    unsigned int k = s + threadIdx.x;
    if (k < max_neighbors) {
      unsigned int idx = row_base + k;
      bool active = (k < count);
      p_x = p_x + (active ? pair_forces_x[idx] : 0.0f);
      p_y = p_y + (active ? pair_forces_y[idx] : 0.0f);
      p_z = p_z + (active ? pair_forces_z[idx] : 0.0f);
      p_e = p_e + (active ? pair_energies[idx] : 0.0f);
      p_w = p_w + (active ? pair_virials[idx] : 0.0f);
    }
  }

  p_x = warp_reduce_sum(p_x);
  p_y = warp_reduce_sum(p_y);
  p_z = warp_reduce_sum(p_z);
  p_e = warp_reduce_sum(p_e);
  p_w = warp_reduce_sum(p_w);

  __shared__ float warp_partials[NUM_WARPS][5];

  unsigned int lane = threadIdx.x & (WARP_SIZE - 1);
  unsigned int warp_id = threadIdx.x / WARP_SIZE;

  if (lane == 0) {
    warp_partials[warp_id][0] = p_x;
    warp_partials[warp_id][1] = p_y;
    warp_partials[warp_id][2] = p_z;
    warp_partials[warp_id][3] = p_e;
    warp_partials[warp_id][4] = p_w;
  }
  __syncthreads();

  if (warp_id == 0) {
    float q_x = (lane < NUM_WARPS) ? warp_partials[lane][0] : 0.0f;
    float q_y = (lane < NUM_WARPS) ? warp_partials[lane][1] : 0.0f;
    float q_z = (lane < NUM_WARPS) ? warp_partials[lane][2] : 0.0f;
    float q_e = (lane < NUM_WARPS) ? warp_partials[lane][3] : 0.0f;
    float q_w = (lane < NUM_WARPS) ? warp_partials[lane][4] : 0.0f;

    q_x = warp_reduce_sum(q_x);
    q_y = warp_reduce_sum(q_y);
    q_z = warp_reduce_sum(q_z);
    q_e = warp_reduce_sum(q_e);
    q_w = warp_reduce_sum(q_w);

    if (lane == 0) {
      net_forces_x[i] = q_x;
      net_forces_y[i] = q_y;
      net_forces_z[i] = q_z;
      net_energy[i] = q_e;
      net_virial[i] = q_w;
    }
  }
}
