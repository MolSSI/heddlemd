// rq-5e059f6b
//
// Andersen stochastic-collision thermostat resample kernels.
//
// `andersen_resample` is the per-particle form: one thread per particle draws a
// uniform `U`, and if `U < p_collision` replaces the velocity with a
// Maxwell-Boltzmann sample at temperature T (three independent Gaussians, one
// per axis, scaled by sqrt(kt/m)).
//
// `andersen_resample_grouped` is the per-rigid-group form, and is the one the
// thermostat actually launches. Its collision decision and resample are made per
// constraint group (rigid molecule): when a group's Bernoulli fires, EVERY atom
// of the group is resampled together. This is required for correctness under
// holonomic constraints. Resampling a single atom of a rigid molecule and then
// projecting the molecule onto its velocity manifold mixes the fresh velocity
// with the group's other, stale velocities and discards part of the injected
// energy into the constrained directions, so the fixed point runs cold (the
// deficit shrinks as the collision rate rises toward the massive limit, where
// every atom is resampled every step and per-atom and per-group coincide).
// Resampling a whole group at once means the group's post-projection velocity is
// a fresh full-Maxwell-Boltzmann draw projected onto its manifold, which is
// exactly a draw from the group's constrained-manifold Maxwell-Boltzmann — the
// correct equilibrium, at any collision rate. See rqm/integration/andersen.md.
//
// For a monatomic / unconstrained run every group is a singleton, and the
// grouped kernel is bit-identical to the per-particle kernel: the Bernoulli is
// keyed on the group's first atom's particle_id (= that atom's own id) and the
// Gaussians on each atom's own id, exactly as below.

#include "precision.cuh"

#include "philox.cuh"

// Resample one atom (storage index `a`, stable id `pid`) from Maxwell-Boltzmann.
__device__ static inline void andersen_draw_atom(
    Real *velocities_x, Real *velocities_y, Real *velocities_z,
    const Real *masses, unsigned int a, unsigned int pid,
    unsigned int seed_lo, unsigned int seed_hi,
    unsigned int draw_counter_lo, unsigned int draw_counter_hi,
    Real kt)
{
  Real sigma = Real_sqrt(kt / masses[a]);
  Real xi_x = philox_gaussian(seed_lo, seed_hi, draw_counter_lo, draw_counter_hi, pid, 0u);
  Real xi_y = philox_gaussian(seed_lo, seed_hi, draw_counter_lo, draw_counter_hi, pid, 1u);
  Real xi_z = philox_gaussian(seed_lo, seed_hi, draw_counter_lo, draw_counter_hi, pid, 2u);
  velocities_x[a] = sigma * xi_x;
  velocities_y[a] = sigma * xi_y;
  velocities_z[a] = sigma * xi_z;
}

extern "C" __global__ void andersen_resample(
    Real *velocities_x, Real *velocities_y, Real *velocities_z,
    const Real *masses,
    const unsigned int *particle_ids,
    const unsigned long long *draw_counter,
    unsigned int seed_lo, unsigned int seed_hi,
    Real p_collision,
    Real kt,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  unsigned long long counter = *draw_counter;
  unsigned int draw_counter_lo = (unsigned int)(counter & 0xFFFFFFFFULL);
  unsigned int draw_counter_hi = (unsigned int)(counter >> 32);

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

// Per-group resample. One thread per group. `mol_atom_offsets` (length
// n_groups + 1) and `mol_atom_indices` (length N, storage indices of the atoms
// of each group, ascending within a group) describe the connectivity-derived
// molecule partition. The Bernoulli decision keys on the group's first atom's
// particle_id, so it is a stable per-group stream and, for a singleton group,
// is exactly the per-particle kernel's decision.
extern "C" __global__ void andersen_resample_grouped(
    Real *velocities_x, Real *velocities_y, Real *velocities_z,
    const Real *masses,
    const unsigned int *particle_ids,
    const unsigned int *mol_atom_offsets,
    const unsigned int *mol_atom_indices,
    const unsigned long long *draw_counter,
    unsigned int seed_lo, unsigned int seed_hi,
    Real p_collision,
    Real kt,
    unsigned int n_groups)
{
  unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_groups) return;

  unsigned long long counter = *draw_counter;
  unsigned int draw_counter_lo = (unsigned int)(counter & 0xFFFFFFFFULL);
  unsigned int draw_counter_hi = (unsigned int)(counter >> 32);

  unsigned int lo = mol_atom_offsets[g];
  unsigned int hi = mol_atom_offsets[g + 1u];

  // One Bernoulli per group, keyed on the group's first atom (draw_kind = 3).
  unsigned int group_pid = particle_ids[mol_atom_indices[lo]];
  unsigned int o0, o1, o2, o3;
  philox4x32_10(seed_lo, seed_hi,
                draw_counter_lo, draw_counter_hi, group_pid, 3u,
                &o0, &o1, &o2, &o3);
  double u = u32_to_uniform_open(o0);
  if (u >= (double) p_collision) {
    return;
  }

  // The whole group resamples together. The step's terminal velocity projection
  // (settle_velocities / rattle_velocities) then maps this fresh full-MB group
  // velocity onto the group's constraint manifold.
  for (unsigned int k = lo; k < hi; ++k) {
    unsigned int a = mol_atom_indices[k];
    andersen_draw_atom(velocities_x, velocities_y, velocities_z, masses,
                       a, particle_ids[a],
                       seed_lo, seed_hi, draw_counter_lo, draw_counter_hi, kt);
  }
}
