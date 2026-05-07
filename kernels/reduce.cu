// rq-31bd2eee

extern "C" __global__ void reduce_pair_forces(
    const float *pair_forces_x,
    const float *pair_forces_y,
    const float *pair_forces_z,
    const unsigned int *neighbor_counts,
    unsigned int max_neighbors,
    float *net_forces_x,
    float *net_forces_y,
    float *net_forces_z,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  unsigned int count = neighbor_counts[i];
  float sum_x = 0.0f;
  float sum_y = 0.0f;
  float sum_z = 0.0f;
  for (unsigned int k = 0; k < count; ++k) {
    unsigned int idx = i * max_neighbors + k;
    sum_x = sum_x + pair_forces_x[idx];
    sum_y = sum_y + pair_forces_y[idx];
    sum_z = sum_z + pair_forces_z[idx];
  }
  net_forces_x[i] = sum_x;
  net_forces_y[i] = sum_y;
  net_forces_z[i] = sum_z;
}
