// rq-9d9ca545
//
// Shape-universal per-atom reduction for angle slots. Every angle
// potential's angle-triple scratch buffer sums into per-atom forces
// the same way. The per-angle contribution kernel for each angle slot
// is dispatched from the framework's JIT-composed angle module (see
// `rqm/forces/jit-composed-intramolecular.md`).

#include "precision.cuh"

// Per-atom segmented reduction. One thread per atom sums every
// angle-triple-buffer slot that names this atom. Same layout as
// `reduce_bond_forces`.
extern "C" __global__ void reduce_angle_forces(
    const Real *angle_triple_x,
    const Real *angle_triple_y,
    const Real *angle_triple_z,
    const Real *angle_triple_energy,
    const Real *angle_triple_virial,
    const unsigned int *atom_angle_offsets,
    const unsigned int *atom_angle_indices,
    Real *slot_force_x,
    Real *slot_force_y,
    Real *slot_force_z,
    Real *slot_energy,
    Real *slot_virial,
    unsigned int n,
    unsigned int write_scalars)
{
  unsigned int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= n) {
    return;
  }

  unsigned int start = atom_angle_offsets[a];
  unsigned int end = atom_angle_offsets[a + 1];

  Real sum_x = R(0.0);
  Real sum_y = R(0.0);
  Real sum_z = R(0.0);
  Real sum_e = R(0.0);
  Real sum_w = R(0.0);

  for (unsigned int i = start; i < end; ++i) {
    unsigned int slot = atom_angle_indices[i];
    sum_x += angle_triple_x[slot];
    sum_y += angle_triple_y[slot];
    sum_z += angle_triple_z[slot];
    if (write_scalars) {
      sum_e += angle_triple_energy[slot];
      sum_w += angle_triple_virial[slot];
    }
  }

  slot_force_x[a] += sum_x;
  slot_force_y[a] += sum_y;
  slot_force_z[a] += sum_z;
  if (write_scalars) {
    slot_energy[a] += sum_e;
    slot_virial[a] += sum_w;
  }
}
