// Langevin BAOAB integrator kernels: position half-drift and Ornstein-Uhlenbeck
// velocity update. The OU step draws per-(particle, axis) Gaussian samples from a
// counter-based Philox-4x32-10 RNG so the kernel is stateless on the device side
// and fully reproducible given (seed, step_index, particle_id, axis_id).

#include "philox.cuh"
#include "pbc.cuh"

// --- A step: x <- x + v * (dt / 2), then wrap into the primary image. -------

extern "C" __global__ void lan_drift_half(
    float *positions_x, float *positions_y, float *positions_z,
    int *images_x, int *images_y, int *images_z,
    const float *velocities_x, const float *velocities_y, const float *velocities_z,
    float lx, float ly, float lz, float xy, float xz, float yz,
    float dt,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float half_dt = dt * 0.5f;
  float px = positions_x[i] + velocities_x[i] * half_dt;
  float py = positions_y[i] + velocities_y[i] * half_dt;
  float pz = positions_z[i] + velocities_z[i] * half_dt;

  int nx = images_x[i];
  int ny = images_y[i];
  int nz = images_z[i];
  int ka, kb, kc;
  triclinic_wrap_with_image(px, py, pz, ka, kb, kc, lx, ly, lz, xy, xz, yz);
  nx += ka;
  ny += kb;
  nz += kc;

  positions_x[i] = px;
  positions_y[i] = py;
  positions_z[i] = pz;
  images_x[i] = nx;
  images_y[i] = ny;
  images_z[i] = nz;
}

// --- O step: v <- alpha * v + sigma_i * xi, sigma_i = sqrt((1-alpha^2) kT/m_i).

extern "C" __global__ void lan_ou_step(
    float *velocities_x, float *velocities_y, float *velocities_z,
    const float *masses,
    const unsigned int *particle_ids,
    unsigned int seed_lo, unsigned int seed_hi,
    unsigned int draw_counter_lo, unsigned int draw_counter_hi,
    float alpha,
    float kt,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float m = masses[i];
  float sigma_factor = sqrtf((1.0f - alpha * alpha) * kt / m);
  unsigned int pid = particle_ids[i];

  float xi_x = philox_gaussian(seed_lo, seed_hi, draw_counter_lo, draw_counter_hi, pid, 0u);
  float xi_y = philox_gaussian(seed_lo, seed_hi, draw_counter_lo, draw_counter_hi, pid, 1u);
  float xi_z = philox_gaussian(seed_lo, seed_hi, draw_counter_lo, draw_counter_hi, pid, 2u);

  velocities_x[i] = alpha * velocities_x[i] + sigma_factor * xi_x;
  velocities_y[i] = alpha * velocities_y[i] + sigma_factor * xi_y;
  velocities_z[i] = alpha * velocities_z[i] + sigma_factor * xi_z;
}
