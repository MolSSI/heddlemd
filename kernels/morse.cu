// rq-d28ad917

#include "precision.cuh"

#include "pbc.cuh"

extern "C" __global__ void morse_bond_force(
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const unsigned int *bonds,
    const Real *bond_de,
    const Real *bond_a,
    const Real *bond_re,
    const Real *lattice,
    Real *bond_pair_x,
    Real *bond_pair_y,
    Real *bond_pair_z,
    Real *bond_pair_energy,
    Real *bond_pair_virial,
    unsigned int n_bonds)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n_bonds) {
    return;
  }

  unsigned int atom_i = bonds[3 * k + 0];
  unsigned int atom_j = bonds[3 * k + 1];
  unsigned int type_idx = bonds[3 * k + 2];

  Real dx = positions_x[atom_i] - positions_x[atom_j];
  Real dy = positions_y[atom_i] - positions_y[atom_j];
  Real dz = positions_z[atom_i] - positions_z[atom_j];

  triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);

  Real r2 = dx * dx + dy * dy + dz * dz;
  if (r2 == R(0.0)) {
    bond_pair_x[2 * k]     = R(0.0);
    bond_pair_y[2 * k]     = R(0.0);
    bond_pair_z[2 * k]     = R(0.0);
    bond_pair_energy[2 * k] = R(0.0);
    bond_pair_virial[2 * k] = R(0.0);
    bond_pair_x[2 * k + 1]     = R(0.0);
    bond_pair_y[2 * k + 1]     = R(0.0);
    bond_pair_z[2 * k + 1]     = R(0.0);
    bond_pair_energy[2 * k + 1] = R(0.0);
    bond_pair_virial[2 * k + 1] = R(0.0);
    return;
  }
  Real r = Real_sqrt(r2);

  Real de = bond_de[type_idx];
  Real a = bond_a[type_idx];
  Real re = bond_re[type_idx];

  Real e = Real_exp(-a * (r - re));
  // F_radial = -dU/dr = -2*De*a*(1-e)*e.  fmag scales the displacement
  // vector r_i - r_j so the Cartesian force on atom_i is fmag * (dx, dy, dz);
  // dividing by r turns r_i - r_j into the unit vector r_hat.
  Real fmag = -R(2.0) * de * a * (R(1.0) - e) * e / r;

  // Per-bond potential energy U_k and scalar virial W_k = r · F_ij.
  // F_ij on atom_i = fmag * (dx, dy, dz), so r_ij · F_ij = fmag * r2.
  Real one_minus_e = R(1.0) - e;
  Real u_k = de * one_minus_e * one_minus_e;
  Real w_k = fmag * r2;

  // Force on atom_i (along +d_hat); force on atom_j is the opposite.
  bond_pair_x[2 * k]     = fmag * dx;
  bond_pair_y[2 * k]     = fmag * dy;
  bond_pair_z[2 * k]     = fmag * dz;
  bond_pair_energy[2 * k] = u_k * R(0.5);
  bond_pair_virial[2 * k] = w_k * R(0.5);
  bond_pair_x[2 * k + 1]     = -fmag * dx;
  bond_pair_y[2 * k + 1]     = -fmag * dy;
  bond_pair_z[2 * k + 1]     = -fmag * dz;
  bond_pair_energy[2 * k + 1] = u_k * R(0.5);
  bond_pair_virial[2 * k + 1] = w_k * R(0.5);
}

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
    unsigned int n)
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
    sum_e += bond_pair_energy[slot];
    sum_w += bond_pair_virial[slot];
  }

  slot_force_x[a] = sum_x;
  slot_force_y[a] = sum_y;
  slot_force_z[a] = sum_z;
  slot_energy[a] = sum_e;
  slot_virial[a] = sum_w;
}
