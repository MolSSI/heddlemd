// rq-d28ad917

#include "pbc.cuh"

extern "C" __global__ void morse_bond_force(
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    const unsigned int *bonds,
    const float *bond_de,
    const float *bond_a,
    const float *bond_re,
    float lx, float ly, float lz, float xy, float xz, float yz,
    float *bond_pair_x,
    float *bond_pair_y,
    float *bond_pair_z,
    float *bond_pair_energy,
    float *bond_pair_virial,
    unsigned int n_bonds)
{
  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n_bonds) {
    return;
  }

  unsigned int atom_i = bonds[3 * k + 0];
  unsigned int atom_j = bonds[3 * k + 1];
  unsigned int type_idx = bonds[3 * k + 2];

  float dx = positions_x[atom_i] - positions_x[atom_j];
  float dy = positions_y[atom_i] - positions_y[atom_j];
  float dz = positions_z[atom_i] - positions_z[atom_j];

  triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);

  float r2 = dx * dx + dy * dy + dz * dz;
  if (r2 == 0.0f) {
    bond_pair_x[2 * k]     = 0.0f;
    bond_pair_y[2 * k]     = 0.0f;
    bond_pair_z[2 * k]     = 0.0f;
    bond_pair_energy[2 * k] = 0.0f;
    bond_pair_virial[2 * k] = 0.0f;
    bond_pair_x[2 * k + 1]     = 0.0f;
    bond_pair_y[2 * k + 1]     = 0.0f;
    bond_pair_z[2 * k + 1]     = 0.0f;
    bond_pair_energy[2 * k + 1] = 0.0f;
    bond_pair_virial[2 * k + 1] = 0.0f;
    return;
  }
  float r = sqrtf(r2);

  float de = bond_de[type_idx];
  float a = bond_a[type_idx];
  float re = bond_re[type_idx];

  float e = expf(-a * (r - re));
  // F_radial = -dU/dr = -2*De*a*(1-e)*e.  fmag scales the displacement
  // vector r_i - r_j so the Cartesian force on atom_i is fmag * (dx, dy, dz);
  // dividing by r turns r_i - r_j into the unit vector r_hat.
  float fmag = -2.0f * de * a * (1.0f - e) * e / r;

  // Per-bond potential energy U_k and scalar virial W_k = r · F_ij.
  // F_ij on atom_i = fmag * (dx, dy, dz), so r_ij · F_ij = fmag * r2.
  float one_minus_e = 1.0f - e;
  float u_k = de * one_minus_e * one_minus_e;
  float w_k = fmag * r2;

  // Force on atom_i (along +d_hat); force on atom_j is the opposite.
  bond_pair_x[2 * k]     = fmag * dx;
  bond_pair_y[2 * k]     = fmag * dy;
  bond_pair_z[2 * k]     = fmag * dz;
  bond_pair_energy[2 * k] = u_k * 0.5f;
  bond_pair_virial[2 * k] = w_k * 0.5f;
  bond_pair_x[2 * k + 1]     = -fmag * dx;
  bond_pair_y[2 * k + 1]     = -fmag * dy;
  bond_pair_z[2 * k + 1]     = -fmag * dz;
  bond_pair_energy[2 * k + 1] = u_k * 0.5f;
  bond_pair_virial[2 * k + 1] = w_k * 0.5f;
}

extern "C" __global__ void reduce_bond_forces(
    const float *bond_pair_x,
    const float *bond_pair_y,
    const float *bond_pair_z,
    const float *bond_pair_energy,
    const float *bond_pair_virial,
    const unsigned int *atom_bond_offsets,
    const unsigned int *atom_bond_indices,
    float *slot_force_x,
    float *slot_force_y,
    float *slot_force_z,
    float *slot_energy,
    float *slot_virial,
    unsigned int n)
{
  unsigned int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= n) {
    return;
  }

  unsigned int start = atom_bond_offsets[a];
  unsigned int end = atom_bond_offsets[a + 1];

  float sum_x = 0.0f;
  float sum_y = 0.0f;
  float sum_z = 0.0f;
  float sum_e = 0.0f;
  float sum_w = 0.0f;

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
