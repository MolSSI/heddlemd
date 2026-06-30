// Shape-universal per-atom reduction for dihedral slots. Every
// dihedral potential's dihedral-quadruple scratch buffer sums into
// per-atom forces the same way. The per-dihedral contribution kernel
// for each dihedral slot is dispatched from the framework's JIT-composed
// dihedral module (see `rqm/forces/jit-composed-intramolecular.md`).

#include "precision.cuh"

// Per-atom segmented reduction. One thread per atom sums every
// dihedral-quadruple-buffer slot that names this atom. Same layout as
// `reduce_angle_forces` but indexed across 4*D slots instead of 3*A.
extern "C" __global__ void reduce_dihedral_forces(
    const Real *dihedral_quadruple_x,
    const Real *dihedral_quadruple_y,
    const Real *dihedral_quadruple_z,
    const Real *dihedral_quadruple_energy,
    const Real *dihedral_quadruple_virial,
    const unsigned int *atom_dihedral_offsets,
    const unsigned int *atom_dihedral_indices,
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

  unsigned int start = atom_dihedral_offsets[a];
  unsigned int end = atom_dihedral_offsets[a + 1];

  Real sum_x = R(0.0);
  Real sum_y = R(0.0);
  Real sum_z = R(0.0);
  Real sum_e = R(0.0);
  Real sum_w = R(0.0);

  for (unsigned int i = start; i < end; ++i) {
    unsigned int slot = atom_dihedral_indices[i];
    sum_x += dihedral_quadruple_x[slot];
    sum_y += dihedral_quadruple_y[slot];
    sum_z += dihedral_quadruple_z[slot];
    if (write_scalars) {
      sum_e += dihedral_quadruple_energy[slot];
      sum_w += dihedral_quadruple_virial[slot];
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
