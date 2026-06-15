// rq-10cc8ddf rq-580fe6f7

#include "precision.cuh"

#include "pbc.cuh"

// Wrap a position back into the primary image of the simulation box and
// advance the three per-direction image counters by the integer triple
// returned by the triclinic wrap. Matches the host-side
// wrap_position_with_image_count formula on SimulationBox.
__device__ static inline void wrap_and_count_triclinic(
    Real &px, Real &py, Real &pz,
    int &nx, int &ny, int &nz,
    Real lx, Real ly, Real lz,
    Real xy, Real xz, Real yz)
{
  int ka, kb, kc;
  triclinic_wrap_with_image(px, py, pz, ka, kb, kc, lx, ly, lz, xy, xz, yz);
  nx += ka;
  ny += kb;
  nz += kc;
}

template <bool LOSSLESS>
__device__ inline void vv_kick_drift_body(
    unsigned int i,
    Real *positions_x, Real *positions_y, Real *positions_z,
    int *images_x, int *images_y, int *images_z,
    Real *velocities_x, Real *velocities_y, Real *velocities_z,
    double *positions_x_lo, double *positions_y_lo, double *positions_z_lo,
    double *velocities_x_lo, double *velocities_y_lo, double *velocities_z_lo,
    const Real *forces_x, const Real *forces_y, const Real *forces_z,
    const Real *masses,
    Real lx, Real ly, Real lz, Real xy, Real xz, Real yz,
    Real dt)
{
  Real m = masses[i];
  Real ax = forces_x[i] / m;
  Real ay = forces_y[i] / m;
  Real az = forces_z[i] / m;
  Real half_dt = dt * R(0.5);

  if constexpr (LOSSLESS) {
    // Compensated kick: extended-precision (v + v_lo) <- (v + v_lo) + a * half_dt
    double dvx = (double)(ax * half_dt);
    double dvy = (double)(ay * half_dt);
    double dvz = (double)(az * half_dt);

    double ext_vx = (double)velocities_x[i] + velocities_x_lo[i] + dvx;
    double ext_vy = (double)velocities_y[i] + velocities_y_lo[i] + dvy;
    double ext_vz = (double)velocities_z[i] + velocities_z_lo[i] + dvz;

    Real new_vx = (Real)ext_vx;
    Real new_vy = (Real)ext_vy;
    Real new_vz = (Real)ext_vz;
    double new_vx_lo = ext_vx - (double)new_vx;
    double new_vy_lo = ext_vy - (double)new_vy;
    double new_vz_lo = ext_vz - (double)new_vz;

    velocities_x[i] = new_vx;
    velocities_y[i] = new_vy;
    velocities_z[i] = new_vz;
    velocities_x_lo[i] = new_vx_lo;
    velocities_y_lo[i] = new_vy_lo;
    velocities_z_lo[i] = new_vz_lo;

    // Compensated drift using extended-precision velocity:
    // (x + x_lo) <- (x + x_lo) + (v + v_lo) * dt
    double dx = ((double)new_vx + new_vx_lo) * (double)dt;
    double dy = ((double)new_vy + new_vy_lo) * (double)dt;
    double dz = ((double)new_vz + new_vz_lo) * (double)dt;

    double ext_x = (double)positions_x[i] + positions_x_lo[i] + dx;
    double ext_y = (double)positions_y[i] + positions_y_lo[i] + dy;
    double ext_z = (double)positions_z[i] + positions_z_lo[i] + dz;

    Real new_x = (Real)ext_x;
    Real new_y = (Real)ext_y;
    Real new_z = (Real)ext_z;
    positions_x_lo[i] = ext_x - (double)new_x;
    positions_y_lo[i] = ext_y - (double)new_y;
    positions_z_lo[i] = ext_z - (double)new_z;

    int nx = images_x[i];
    int ny = images_y[i];
    int nz = images_z[i];
    wrap_and_count_triclinic(new_x, new_y, new_z, nx, ny, nz,
                             lx, ly, lz, xy, xz, yz);

    positions_x[i] = new_x;
    positions_y[i] = new_y;
    positions_z[i] = new_z;
    images_x[i] = nx;
    images_y[i] = ny;
    images_z[i] = nz;
  } else {
    Real vx = velocities_x[i] + ax * half_dt;
    Real vy = velocities_y[i] + ay * half_dt;
    Real vz = velocities_z[i] + az * half_dt;
    velocities_x[i] = vx;
    velocities_y[i] = vy;
    velocities_z[i] = vz;

    Real px = positions_x[i] + vx * dt;
    Real py = positions_y[i] + vy * dt;
    Real pz = positions_z[i] + vz * dt;

    int nx = images_x[i];
    int ny = images_y[i];
    int nz = images_z[i];
    wrap_and_count_triclinic(px, py, pz, nx, ny, nz,
                             lx, ly, lz, xy, xz, yz);

    positions_x[i] = px;
    positions_y[i] = py;
    positions_z[i] = pz;
    images_x[i] = nx;
    images_y[i] = ny;
    images_z[i] = nz;
  }
}

template <bool LOSSLESS>
__device__ inline void vv_kick_body(
    unsigned int i,
    Real *velocities_x, Real *velocities_y, Real *velocities_z,
    double *velocities_x_lo, double *velocities_y_lo, double *velocities_z_lo,
    const Real *forces_x, const Real *forces_y, const Real *forces_z,
    const Real *masses,
    Real dt)
{
  Real m = masses[i];
  Real ax = forces_x[i] / m;
  Real ay = forces_y[i] / m;
  Real az = forces_z[i] / m;
  Real half_dt = dt * R(0.5);

  if constexpr (LOSSLESS) {
    double dvx = (double)(ax * half_dt);
    double dvy = (double)(ay * half_dt);
    double dvz = (double)(az * half_dt);

    double ext_vx = (double)velocities_x[i] + velocities_x_lo[i] + dvx;
    double ext_vy = (double)velocities_y[i] + velocities_y_lo[i] + dvy;
    double ext_vz = (double)velocities_z[i] + velocities_z_lo[i] + dvz;

    Real new_vx = (Real)ext_vx;
    Real new_vy = (Real)ext_vy;
    Real new_vz = (Real)ext_vz;
    velocities_x_lo[i] = ext_vx - (double)new_vx;
    velocities_y_lo[i] = ext_vy - (double)new_vy;
    velocities_z_lo[i] = ext_vz - (double)new_vz;
    velocities_x[i] = new_vx;
    velocities_y[i] = new_vy;
    velocities_z[i] = new_vz;
  } else {
    velocities_x[i] += ax * half_dt;
    velocities_y[i] += ay * half_dt;
    velocities_z[i] += az * half_dt;
  }
}

extern "C" __global__ void vv_kick_drift(
    Real *positions_x, Real *positions_y, Real *positions_z,
    int *images_x, int *images_y, int *images_z,
    Real *velocities_x, Real *velocities_y, Real *velocities_z,
    const Real *forces_x, const Real *forces_y, const Real *forces_z,
    const Real *masses,
    Real lx, Real ly, Real lz, Real xy, Real xz, Real yz,
    Real dt,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  vv_kick_drift_body<false>(
      i,
      positions_x, positions_y, positions_z,
      images_x, images_y, images_z,
      velocities_x, velocities_y, velocities_z,
      nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr,
      forces_x, forces_y, forces_z,
      masses, lx, ly, lz, xy, xz, yz, dt);
}

extern "C" __global__ void vv_kick(
    Real *velocities_x, Real *velocities_y, Real *velocities_z,
    const Real *forces_x, const Real *forces_y, const Real *forces_z,
    const Real *masses,
    Real dt,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  vv_kick_body<false>(
      i,
      velocities_x, velocities_y, velocities_z,
      nullptr, nullptr, nullptr,
      forces_x, forces_y, forces_z,
      masses, dt);
}

extern "C" __global__ void vv_kick_drift_lossless(
    Real *positions_x, Real *positions_y, Real *positions_z,
    int *images_x, int *images_y, int *images_z,
    Real *velocities_x, Real *velocities_y, Real *velocities_z,
    double *positions_x_lo, double *positions_y_lo, double *positions_z_lo,
    double *velocities_x_lo, double *velocities_y_lo, double *velocities_z_lo,
    const Real *forces_x, const Real *forces_y, const Real *forces_z,
    const Real *masses,
    Real lx, Real ly, Real lz, Real xy, Real xz, Real yz,
    Real dt,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  vv_kick_drift_body<true>(
      i,
      positions_x, positions_y, positions_z,
      images_x, images_y, images_z,
      velocities_x, velocities_y, velocities_z,
      positions_x_lo, positions_y_lo, positions_z_lo,
      velocities_x_lo, velocities_y_lo, velocities_z_lo,
      forces_x, forces_y, forces_z,
      masses, lx, ly, lz, xy, xz, yz, dt);
}

extern "C" __global__ void vv_kick_lossless(
    Real *velocities_x, Real *velocities_y, Real *velocities_z,
    double *velocities_x_lo, double *velocities_y_lo, double *velocities_z_lo,
    const Real *forces_x, const Real *forces_y, const Real *forces_z,
    const Real *masses,
    Real dt,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  vv_kick_body<true>(
      i,
      velocities_x, velocities_y, velocities_z,
      velocities_x_lo, velocities_y_lo, velocities_z_lo,
      forces_x, forces_y, forces_z,
      masses, dt);
}
