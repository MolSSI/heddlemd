// Monte-Carlo barostat: rigid molecular-centre-of-mass volume scale.
// See `rqm/integration/mc-barostat.md`.
//
// One thread per molecule. For molecule m over its atom slice
// `mol_atom_indices[mol_atom_offsets[m] .. mol_atom_offsets[m+1]]`, the
// thread computes the mass-weighted centre of mass and translates every
// atom of the molecule rigidly by `(scale - 1) * COM`, so the molecular
// COM scales about the origin while every intramolecular displacement is
// unchanged. The per-molecule reduction sums atoms in their stored
// (ascending-index) order, so the result is bit-identical across runs on
// the same GPU.
#include "precision.cuh"

extern "C" __global__ void mc_barostat_scale_molecule_com(
    Real4 *posq,                            // positions (xyz) + charge (w)
    const unsigned int *mol_atom_offsets,   // length n_mol + 1
    const unsigned int *mol_atom_indices,   // length N, atom ids by molecule
    const Real *masses,                     // length N
    Real scale,
    unsigned int n_mol)
{
  unsigned int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= n_mol) {
    return;
  }
  unsigned int lo = mol_atom_offsets[m];
  unsigned int hi = mol_atom_offsets[m + 1u];

  Real total_mass = R(0.0);
  Real cx = R(0.0), cy = R(0.0), cz = R(0.0);
  for (unsigned int k = lo; k < hi; ++k) {
    unsigned int a = mol_atom_indices[k];
    Real mass = masses[a];
    Real4 p = posq[a];
    cx += mass * p.x;
    cy += mass * p.y;
    cz += mass * p.z;
    total_mass += mass;
  }
  Real inv = R(1.0) / total_mass;
  Real shift = scale - R(1.0);
  Real dx = shift * cx * inv;
  Real dy = shift * cy * inv;
  Real dz = shift * cz * inv;

  for (unsigned int k = lo; k < hi; ++k) {
    unsigned int a = mol_atom_indices[k];
    Real4 p = posq[a];
    p.x += dx;
    p.y += dy;
    p.z += dz;
    posq[a] = p;
  }
}
