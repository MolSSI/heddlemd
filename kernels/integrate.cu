// rq-10cc8ddf

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
  float m = masses[i];
  float ax = forces_x[i] / m;
  float ay = forces_y[i] / m;
  float az = forces_z[i] / m;
  float half_dt = dt * 0.5f;
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
  float m = masses[i];
  float half_dt = dt * 0.5f;
  velocities_x[i] += (forces_x[i] / m) * half_dt;
  velocities_y[i] += (forces_y[i] / m) * half_dt;
  velocities_z[i] += (forces_z[i] / m) * half_dt;
}
