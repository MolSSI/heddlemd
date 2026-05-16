// rq-f606ff6f

// Single-block deterministic kinetic-energy reduction.
//
// One block of 256 threads is launched. Each thread accumulates the
// per-particle KE contribution `½ m (vx² + vy² + vz²)` for the slice
// of particles whose indices are congruent to threadIdx.x modulo 256
// (i.e. thread `t` sees particles `t, t + 256, t + 512, …`). The
// strided per-thread sum is computed in an `f32` register without any
// inter-thread interaction.
//
// The per-thread partials are then reduced via a deterministic
// left-to-right pairwise tree in shared memory: at each stride `s`
// (1, 2, 4, …, blockDim.x/2), the lower half of every pair absorbs
// the upper half. The reduction topology and visitation order are
// completely determined by `blockDim.x` and `n`, so two runs with
// byte-identical inputs on the same GPU produce a byte-identical
// `partial_out[0]`.
extern "C" __global__ void kinetic_energy_reduce(
    const float *velocities_x,
    const float *velocities_y,
    const float *velocities_z,
    const float *masses,
    float *partial_out,    // length 1; only thread 0 writes
    unsigned int n)
{
  __shared__ float partial[256];

  unsigned int tid = threadIdx.x;
  float sum = 0.0f;
  for (unsigned int i = tid; i < n; i += blockDim.x) {
    float vx = velocities_x[i];
    float vy = velocities_y[i];
    float vz = velocities_z[i];
    float m = masses[i];
    sum += 0.5f * m * (vx * vx + vy * vy + vz * vz);
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

// Uniform per-particle velocity rescale: v_i ← factor · v_i for every
// component of every particle. One thread per particle, no inter-thread
// interaction.
extern "C" __global__ void rescale_velocities(
    float *velocities_x,
    float *velocities_y,
    float *velocities_z,
    float factor,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  velocities_x[i] *= factor;
  velocities_y[i] *= factor;
  velocities_z[i] *= factor;
}
