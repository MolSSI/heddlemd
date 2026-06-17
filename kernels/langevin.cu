// Langevin BAOAB integrator kernels: position half-drift and Ornstein-Uhlenbeck
// velocity update. The OU step draws per-(particle, axis) Gaussian samples from a
// counter-based Philox-4x32-10 RNG so the kernel is stateless on the device side
// and fully reproducible given (seed, step_index, particle_id, axis_id).

#include "precision.cuh"

#include "philox.cuh"
#include "pbc.cuh"

// --- A step: x <- x + v * (dt / 2), then wrap into the primary image. -------

extern "C" __global__ void lan_drift_half(
    Real *positions_x, Real *positions_y, Real *positions_z,
    int *images_x, int *images_y, int *images_z,
    const Real *velocities_x, const Real *velocities_y, const Real *velocities_z,
    const Real *lattice,
    Real dt,
    unsigned int n)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  Real half_dt = dt * R(0.5);
  Real px = positions_x[i] + velocities_x[i] * half_dt;
  Real py = positions_y[i] + velocities_y[i] * half_dt;
  Real pz = positions_z[i] + velocities_z[i] * half_dt;

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
    Real *velocities_x, Real *velocities_y, Real *velocities_z,
    const Real *masses,
    const unsigned int *particle_ids,
    const unsigned long long *draw_counter,
    unsigned int seed_lo, unsigned int seed_hi,
    Real alpha,
    Real kt,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  unsigned long long counter = *draw_counter;
  unsigned int draw_counter_lo = (unsigned int)(counter & 0xFFFFFFFFULL);
  unsigned int draw_counter_hi = (unsigned int)(counter >> 32);

  Real m = masses[i];
  Real sigma_factor = Real_sqrt((R(1.0) - alpha * alpha) * kt / m);
  unsigned int pid = particle_ids[i];

  Real xi_x = philox_gaussian(seed_lo, seed_hi, draw_counter_lo, draw_counter_hi, pid, 0u);
  Real xi_y = philox_gaussian(seed_lo, seed_hi, draw_counter_lo, draw_counter_hi, pid, 1u);
  Real xi_z = philox_gaussian(seed_lo, seed_hi, draw_counter_lo, draw_counter_hi, pid, 2u);

  velocities_x[i] = alpha * velocities_x[i] + sigma_factor * xi_x;
  velocities_y[i] = alpha * velocities_y[i] + sigma_factor * xi_y;
  velocities_z[i] = alpha * velocities_z[i] + sigma_factor * xi_z;
}
