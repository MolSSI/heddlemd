// rq-c0f98145
//
// Combiner kernel for the pluggable potential framework. The framework
// owns one set of five flat device buffers per ForceClass — Fast and
// Slow — each of length `num_class_slots * n`, carrying the
// per-particle reduced contributions from every slot in that class.
// Each slot writes into its row of its class's buffers during its
// `Potential::reduce` step. This kernel sums every row, ordering
// classes Fast then Slow and ordering slots within each class by
// registration, and writes the per-particle totals into
// ParticleBuffers.forces_*, potential_energies, and virials.

#include "precision.cuh"

extern "C" __global__ void accumulate_forces(
    const Real *fast_slot_forces_x,
    const Real *fast_slot_forces_y,
    const Real *fast_slot_forces_z,
    const Real *fast_slot_energies,
    const Real *fast_slot_virials,
    unsigned int num_fast_slots,
    const Real *slow_slot_forces_x,
    const Real *slow_slot_forces_y,
    const Real *slow_slot_forces_z,
    const Real *slow_slot_energies,
    const Real *slow_slot_virials,
    unsigned int num_slow_slots,
    Real *forces_x,
    Real *forces_y,
    Real *forces_z,
    Real *potential_energies,
    Real *virials,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  Real sx = R(0.0);
  Real sy = R(0.0);
  Real sz = R(0.0);
  Real se = R(0.0);
  Real sw = R(0.0);

  for (unsigned int k = 0; k < num_fast_slots; ++k) {
    unsigned int idx = k * n + i;
    sx += fast_slot_forces_x[idx];
    sy += fast_slot_forces_y[idx];
    sz += fast_slot_forces_z[idx];
    se += fast_slot_energies[idx];
    sw += fast_slot_virials[idx];
  }
  for (unsigned int k = 0; k < num_slow_slots; ++k) {
    unsigned int idx = k * n + i;
    sx += slow_slot_forces_x[idx];
    sy += slow_slot_forces_y[idx];
    sz += slow_slot_forces_z[idx];
    se += slow_slot_energies[idx];
    sw += slow_slot_virials[idx];
  }

  forces_x[i] = sx;
  forces_y[i] = sy;
  forces_z[i] = sz;
  potential_energies[i] = se;
  virials[i] = sw;
}
