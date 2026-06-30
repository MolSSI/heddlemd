// rq-b2559e09
//
// Shape-universal per-atom reduction for bonded slots. Every bonded
// potential's bond-pair scratch buffer sums into per-atom forces the
// same way; this kernel implements that summation. The per-bond
// contribution kernel for each bonded slot is dispatched from the
// framework's JIT-composed bonded module (see
// `rqm/forces/jit-composed-intramolecular.md`).

#include "precision.cuh"

// Per-atom segmented reduction. One thread per atom sums every
// bond-pair-buffer slot that names this atom.
extern "C" __global__ void reduce_bond_forces(
    const Real *bond_pair_x,
    const Real *bond_pair_y,
    const Real *bond_pair_z,
    const Real *bond_pair_energy,
    const Real *bond_pair_virial,
    const unsigned int *atom_bond_offsets,
    const unsigned int *atom_bond_indices,
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

  unsigned int start = atom_bond_offsets[a];
  unsigned int end = atom_bond_offsets[a + 1];

  Real sum_x = R(0.0);
  Real sum_y = R(0.0);
  Real sum_z = R(0.0);
  Real sum_e = R(0.0);
  Real sum_w = R(0.0);

  for (unsigned int i = start; i < end; ++i) {
    unsigned int slot = atom_bond_indices[i];
    sum_x += bond_pair_x[slot];
    sum_y += bond_pair_y[slot];
    sum_z += bond_pair_z[slot];
    if (write_scalars) {
      sum_e += bond_pair_energy[slot];
      sum_w += bond_pair_virial[slot];
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
