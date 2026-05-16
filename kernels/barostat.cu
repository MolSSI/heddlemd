// rq-0d8c8688

// Single-block deterministic virial-sum reduction. Mirrors
// `kinetic_energy_reduce` in nose_hoover.cu: one block of 256 threads,
// strided per-thread accumulator, deterministic left-to-right pairwise
// tree in shared memory. Two runs with byte-identical inputs on the
// same GPU produce a byte-identical `partial_out[0]`.
extern "C" __global__ void virial_sum_reduce(
    const float *virials,
    float *partial_out,    // length 1; only thread 0 writes
    unsigned int n)
{
  __shared__ float partial[256];

  unsigned int tid = threadIdx.x;
  float sum = 0.0f;
  for (unsigned int i = tid; i < n; i += blockDim.x) {
    sum += virials[i];
  }
  partial[tid] = sum;
  __syncthreads();

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2u * stride)) == 0u && (tid + stride) < blockDim.x) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0u) {
    partial_out[0] = partial[0];
  }
}

// Uniform per-particle position rescale: x_i ← factor · x_i for every
// component of every particle. One thread per particle, no inter-thread
// interaction. Does NOT touch velocities, forces, image flags, or any
// neighbor-list reference positions; fractional coordinates are
// invariant under uniform scaling so image flags carry over unchanged,
// and the neighbor list refreshes its reference positions on the next
// `force_field.step` via the box-generation change-detection path.
extern "C" __global__ void rescale_positions(
    float *positions_x,
    float *positions_y,
    float *positions_z,
    float factor,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  positions_x[i] *= factor;
  positions_y[i] *= factor;
  positions_z[i] *= factor;
}
