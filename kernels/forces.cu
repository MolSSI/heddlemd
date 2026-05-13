// rq-c0f98145
//
// Combiner kernel for the pluggable potential framework. The framework
// owns three flat device buffers of length `num_slots * n`. Each slot k
// has its per-particle reduced force in the row
// [k * n, (k + 1) * n) of each buffer. This kernel sums every row in
// slot order and writes the per-particle totals into ParticleBuffers.forces_*.

extern "C" __global__ void accumulate_forces(
    const float *slot_forces_x,
    const float *slot_forces_y,
    const float *slot_forces_z,
    unsigned int num_slots,
    float *forces_x,
    float *forces_y,
    float *forces_z,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  float sx = 0.0f;
  float sy = 0.0f;
  float sz = 0.0f;

  for (unsigned int k = 0; k < num_slots; ++k) {
    unsigned int idx = k * n + i;
    sx += slot_forces_x[idx];
    sy += slot_forces_y[idx];
    sz += slot_forces_z[idx];
  }

  forces_x[i] = sx;
  forces_y[i] = sy;
  forces_z[i] = sz;
}
