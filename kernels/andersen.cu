// rq-5e059f6b
//
// Andersen stochastic-collision thermostat resample kernel. One
// thread per particle: draws a uniform `U` via Philox, and if
// `U < p_collision` replaces the velocity with a Maxwell-Boltzmann
// sample at temperature T (three independent Gaussians, one per axis,
// scaled by sqrt(kt/m)). Otherwise leaves the velocity unchanged.

#include "precision.cuh"

#include "philox.cuh"

extern "C" __global__ void andersen_resample(
    Real *velocities_x, Real *velocities_y, Real *velocities_z,
    const Real *masses,
    const unsigned int *particle_ids,
    unsigned int seed_lo, unsigned int seed_hi,
    unsigned int draw_counter_lo, unsigned int draw_counter_hi,
    Real p_collision,
    Real kt,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  unsigned int pid = particle_ids[i];

  // Draw the uniform for the Bernoulli decision (draw_kind = 3).
  unsigned int o0, o1, o2, o3;
  philox4x32_10(seed_lo, seed_hi,
                draw_counter_lo, draw_counter_hi, pid, 3u,
                &o0, &o1, &o2, &o3);
  double u = u32_to_uniform_open(o0);

  if (u >= (double) p_collision) {
    return;
  }

  Real m = masses[i];
  Real sigma = Real_sqrt(kt / m);
  Real xi_x = philox_gaussian(seed_lo, seed_hi,
                               draw_counter_lo, draw_counter_hi, pid, 0u);
  Real xi_y = philox_gaussian(seed_lo, seed_hi,
                               draw_counter_lo, draw_counter_hi, pid, 1u);
  Real xi_z = philox_gaussian(seed_lo, seed_hi,
                               draw_counter_lo, draw_counter_hi, pid, 2u);

  velocities_x[i] = sigma * xi_x;
  velocities_y[i] = sigma * xi_y;
  velocities_z[i] = sigma * xi_z;
}
