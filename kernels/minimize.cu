// Steepest-descent minimizer kernels. See
// `rqm/minimization/steepest-descent.md`.

// Trial position update: x_new = x + step_size · F · inv_f_max, then
// wrap into the primary image and advance the image counters on any
// crossing. One thread per particle. `step_size` is the current adaptive
// step in metres (atomic units in the engine); `inv_f_max = 1 / max_i
// ||F_i||` (computed by `sd_f_max_reduction` and divided once on the
// host).
//
// The wrap keeps the primary-image invariant on `positions_x/y/z` intact
// across trials. Downstream consumers of the trial state — the constraint
// slot's `apply_position_projection_only`, the force pipeline, the
// trajectory writer if the phase writes trial frames — all assume it.
// See `rqm/minimization/steepest-descent.md` step 1.
#include "precision.cuh"

#include "pbc.cuh"

extern "C" __global__ void sd_compute_step(
    Real4 *posq,
    int *images_x,
    int *images_y,
    int *images_z,
    const Real *forces_x,
    const Real *forces_y,
    const Real *forces_z,
    const Real *lattice,
    Real step_size,
    Real inv_f_max,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  Real lx = lattice[0], ly = lattice[1], lz = lattice[2];
  Real xy = lattice[3], xz = lattice[4], yz = lattice[5];
  Real scale = step_size * inv_f_max;
  Real4 pq = posq[i];
  pq.x = pq.x + forces_x[i] * scale;
  pq.y = pq.y + forces_y[i] * scale;
  pq.z = pq.z + forces_z[i] * scale;
  int nx = images_x[i];
  int ny = images_y[i];
  int nz = images_z[i];
  wrap_and_count_triclinic(pq.x, pq.y, pq.z, nx, ny, nz,
                           lx, ly, lz, xy, xz, yz);
  posq[i] = pq;
  images_x[i] = nx;
  images_y[i] = ny;
  images_z[i] = nz;
}

// Snapshot positions and image counters to per-particle scratch buffers.
// One thread per particle. Used before each trial step so a rejected
// trial can restore the previous accepted `(positions, images)` pair —
// the two are always snapshotted and restored together so the invariant
// `unwrapped = wrapped + N · L` remains exact across every accept/reject
// cycle. See `rqm/minimization/steepest-descent.md` *Snapshot and
// restore*.
extern "C" __global__ void sd_snapshot(
    const Real4 *posq,
    const int *images_x,
    const int *images_y,
    const int *images_z,
    Real *snapshot_x,
    Real *snapshot_y,
    Real *snapshot_z,
    int *images_snapshot_x,
    int *images_snapshot_y,
    int *images_snapshot_z,
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
  images_snapshot_x[i] = images_x[i];
  images_snapshot_y[i] = images_y[i];
  images_snapshot_z[i] = images_z[i];
}

// Restore positions and image counters from the snapshot. One thread per
// particle. Called after a rejected trial; both streams are restored
// together for the reasons in `sd_snapshot` above.
extern "C" __global__ void sd_restore(
    Real4 *posq,
    int *images_x,
    int *images_y,
    int *images_z,
    const Real *snapshot_x,
    const Real *snapshot_y,
    const Real *snapshot_z,
    const int *images_snapshot_x,
    const int *images_snapshot_y,
    const int *images_snapshot_z,
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
  images_x[i] = images_snapshot_x[i];
  images_y[i] = images_snapshot_y[i];
  images_z[i] = images_snapshot_z[i];
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
