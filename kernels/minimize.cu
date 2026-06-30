// Steepest-descent minimizer kernels. See
// `rqm/minimization/steepest-descent.md`.

// Trial position update: x_new = x + step_size · F · inv_f_max.
// One thread per particle. `step_size` is the current adaptive step
// in metres; `inv_f_max = 1 / max_i ||F_i||` (computed by
// `sd_f_max_reduction` and divided once on the host).
#include "precision.cuh"

extern "C" __global__ void sd_compute_step(
    Real4 *posq,
    const Real *forces_x,
    const Real *forces_y,
    const Real *forces_z,
    Real step_size,
    Real inv_f_max,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  Real scale = step_size * inv_f_max;
  Real4 pq = posq[i];
  pq.x = pq.x + forces_x[i] * scale;
  pq.y = pq.y + forces_y[i] * scale;
  pq.z = pq.z + forces_z[i] * scale;
  posq[i] = pq;
}

// Snapshot positions to per-particle scratch buffers. One thread per
// particle. Used before each trial step so a rejected trial can
// restore the previous accepted positions.
extern "C" __global__ void sd_snapshot(
    const Real4 *posq,
    Real *snapshot_x,
    Real *snapshot_y,
    Real *snapshot_z,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  Real4 pq = posq[i];
  snapshot_x[i] = pq.x;
  snapshot_y[i] = pq.y;
  snapshot_z[i] = pq.z;
}

// Restore positions from the snapshot. One thread per particle.
extern "C" __global__ void sd_restore(
    Real4 *posq,
    const Real *snapshot_x,
    const Real *snapshot_y,
    const Real *snapshot_z,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  Real4 pq = posq[i];
  pq.x = snapshot_x[i];
  pq.y = snapshot_y[i];
  pq.z = snapshot_z[i];
  posq[i] = pq;
}

// Single-block deterministic max-magnitude reduction over per-atom
// force vectors. Reduces ||F_i|| = sqrt(F_xi² + F_yi² + F_zi²) into a
// single scalar in `partial_out[0]`. Block size 256, grid 1.
//
// `max` of two floats is associative and commutative (no rounding),
// so the tree-shape is irrelevant for determinism — the result is
// bit-identical regardless of thread schedule.
extern "C" __global__ void sd_f_max_reduction(
    const Real *forces_x,
    const Real *forces_y,
    const Real *forces_z,
    Real *partial_out,
    unsigned int n)
{
  __shared__ Real partial[256];

  unsigned int tid = threadIdx.x;
  Real local_max = R(0.0);
  for (unsigned int i = tid; i < n; i += blockDim.x) {
    Real fx = forces_x[i];
    Real fy = forces_y[i];
    Real fz = forces_z[i];
    Real mag2 = fx * fx + fy * fy + fz * fz;
    if (mag2 > local_max) {
      local_max = mag2;
    }
  }
  partial[tid] = local_max;
  __syncthreads();

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2u * stride)) == 0u && (tid + stride) < blockDim.x) {
      Real a = partial[tid];
      Real b = partial[tid + stride];
      partial[tid] = (a > b) ? a : b;
    }
    __syncthreads();
  }

  if (tid == 0u) {
    // Take the sqrt once on the device; the host divides by it.
    partial_out[0] = Real_sqrt(partial[0]);
  }
}
