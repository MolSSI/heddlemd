// rq-c0f98145
//
// Class-combine kernel for the pluggable potential framework. The
// framework owns two per-class accumulators (Fast, Slow), each holding
// five length-n device buffers: the per-particle sum of every slot's
// contribution to force-x, force-y, force-z, potential-energy share,
// and scalar-virial share within that class. This kernel writes the
// per-particle totals into ParticleBuffers.forces_*, potential_energies,
// and virials by summing the two class accumulators element-wise.

#include "precision.cuh"

extern "C" __global__ void combine_class_totals(
    const Real *fast_total_forces_x,
    const Real *fast_total_forces_y,
    const Real *fast_total_forces_z,
    const Real *fast_total_potential_energies,
    const Real *fast_total_virials,
    const Real *slow_total_forces_x,
    const Real *slow_total_forces_y,
    const Real *slow_total_forces_z,
    const Real *slow_total_potential_energies,
    const Real *slow_total_virials,
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

  forces_x[i]           = fast_total_forces_x[i]           + slow_total_forces_x[i];
  forces_y[i]           = fast_total_forces_y[i]           + slow_total_forces_y[i];
  forces_z[i]           = fast_total_forces_z[i]           + slow_total_forces_z[i];
  potential_energies[i] = fast_total_potential_energies[i] + slow_total_potential_energies[i];
  virials[i]            = fast_total_virials[i]            + slow_total_virials[i];
}
