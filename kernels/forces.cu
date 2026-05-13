// rq-c0f98145
//
// Combiner kernel for the pluggable potential framework. The framework
// owns five flat device buffers of length `num_slots * n` carrying the
// per-particle reduced contributions from every slot: three force
// components, one potential-energy share, and one scalar-virial share.
// Each slot `k` writes into row `k` of all five buffers during its
// `Potential::reduce` step. This kernel sums every row in slot order
// and writes the per-particle totals into ParticleBuffers.forces_*,
// potential_energies, and virials.

extern "C" __global__ void accumulate_forces(
    const float *slot_forces_x,
    const float *slot_forces_y,
    const float *slot_forces_z,
    const float *slot_energies,
    const float *slot_virials,
    unsigned int num_slots,
    float *forces_x,
    float *forces_y,
    float *forces_z,
    float *potential_energies,
    float *virials,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  float sx = 0.0f;
  float sy = 0.0f;
  float sz = 0.0f;
  float se = 0.0f;
  float sw = 0.0f;

  for (unsigned int k = 0; k < num_slots; ++k) {
    unsigned int idx = k * n + i;
    sx += slot_forces_x[idx];
    sy += slot_forces_y[idx];
    sz += slot_forces_z[idx];
    se += slot_energies[idx];
    sw += slot_virials[idx];
  }

  forces_x[i] = sx;
  forces_y[i] = sy;
  forces_z[i] = sz;
  potential_energies[i] = se;
  virials[i] = sw;
}
