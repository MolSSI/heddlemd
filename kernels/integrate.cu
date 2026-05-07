// rq-10cc8ddf rq-580fe6f7

template <bool LOSSLESS>
__device__ inline void vv_kick_drift_body(
    unsigned int i,
    float *positions_x, float *positions_y, float *positions_z,
    float *velocities_x, float *velocities_y, float *velocities_z,
    double *positions_x_lo, double *positions_y_lo, double *positions_z_lo,
    double *velocities_x_lo, double *velocities_y_lo, double *velocities_z_lo,
    const float *forces_x, const float *forces_y, const float *forces_z,
    const float *masses,
    float dt)
{
  float m = masses[i];
  float ax = forces_x[i] / m;
  float ay = forces_y[i] / m;
  float az = forces_z[i] / m;
  float half_dt = dt * 0.5f;

  if constexpr (LOSSLESS) {
    // Compensated kick: extended-precision (v + v_lo) <- (v + v_lo) + a * half_dt
    double dvx = (double)(ax * half_dt);
    double dvy = (double)(ay * half_dt);
    double dvz = (double)(az * half_dt);

    double ext_vx = (double)velocities_x[i] + velocities_x_lo[i] + dvx;
    double ext_vy = (double)velocities_y[i] + velocities_y_lo[i] + dvy;
    double ext_vz = (double)velocities_z[i] + velocities_z_lo[i] + dvz;

    float new_vx = (float)ext_vx;
    float new_vy = (float)ext_vy;
    float new_vz = (float)ext_vz;
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

    float new_x = (float)ext_x;
    float new_y = (float)ext_y;
    float new_z = (float)ext_z;
    positions_x_lo[i] = ext_x - (double)new_x;
    positions_y_lo[i] = ext_y - (double)new_y;
    positions_z_lo[i] = ext_z - (double)new_z;
    positions_x[i] = new_x;
    positions_y[i] = new_y;
    positions_z[i] = new_z;
  } else {
    float vx = velocities_x[i] + ax * half_dt;
    float vy = velocities_y[i] + ay * half_dt;
    float vz = velocities_z[i] + az * half_dt;
    velocities_x[i] = vx;
    velocities_y[i] = vy;
    velocities_z[i] = vz;
    positions_x[i] += vx * dt;
    positions_y[i] += vy * dt;
    positions_z[i] += vz * dt;
  }
}

template <bool LOSSLESS>
__device__ inline void vv_kick_body(
    unsigned int i,
    float *velocities_x, float *velocities_y, float *velocities_z,
    double *velocities_x_lo, double *velocities_y_lo, double *velocities_z_lo,
    const float *forces_x, const float *forces_y, const float *forces_z,
    const float *masses,
    float dt)
{
  float m = masses[i];
  float ax = forces_x[i] / m;
  float ay = forces_y[i] / m;
  float az = forces_z[i] / m;
  float half_dt = dt * 0.5f;

  if constexpr (LOSSLESS) {
    double dvx = (double)(ax * half_dt);
    double dvy = (double)(ay * half_dt);
    double dvz = (double)(az * half_dt);

    double ext_vx = (double)velocities_x[i] + velocities_x_lo[i] + dvx;
    double ext_vy = (double)velocities_y[i] + velocities_y_lo[i] + dvy;
    double ext_vz = (double)velocities_z[i] + velocities_z_lo[i] + dvz;

    float new_vx = (float)ext_vx;
    float new_vy = (float)ext_vy;
    float new_vz = (float)ext_vz;
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
    float *positions_x, float *positions_y, float *positions_z,
    float *velocities_x, float *velocities_y, float *velocities_z,
    const float *forces_x, const float *forces_y, const float *forces_z,
    const float *masses,
    float dt,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  vv_kick_drift_body<false>(
      i,
      positions_x, positions_y, positions_z,
      velocities_x, velocities_y, velocities_z,
      nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr,
      forces_x, forces_y, forces_z,
      masses, dt);
}

extern "C" __global__ void vv_kick(
    float *velocities_x, float *velocities_y, float *velocities_z,
    const float *forces_x, const float *forces_y, const float *forces_z,
    const float *masses,
    float dt,
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
    float *positions_x, float *positions_y, float *positions_z,
    float *velocities_x, float *velocities_y, float *velocities_z,
    double *positions_x_lo, double *positions_y_lo, double *positions_z_lo,
    double *velocities_x_lo, double *velocities_y_lo, double *velocities_z_lo,
    const float *forces_x, const float *forces_y, const float *forces_z,
    const float *masses,
    float dt,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  vv_kick_drift_body<true>(
      i,
      positions_x, positions_y, positions_z,
      velocities_x, velocities_y, velocities_z,
      positions_x_lo, positions_y_lo, positions_z_lo,
      velocities_x_lo, velocities_y_lo, velocities_z_lo,
      forces_x, forces_y, forces_z,
      masses, dt);
}

extern "C" __global__ void vv_kick_lossless(
    float *velocities_x, float *velocities_y, float *velocities_z,
    double *velocities_x_lo, double *velocities_y_lo, double *velocities_z_lo,
    const float *forces_x, const float *forces_y, const float *forces_z,
    const float *masses,
    float dt,
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
