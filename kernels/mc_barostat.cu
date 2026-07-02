// Monte-Carlo barostat: rigid molecular-centre-of-mass volume scale.
// See `rqm/integration/mc-barostat.md`.
//
// One thread per molecule. Each molecule's mass-weighted centre of mass is
// computed by reconstructing the molecule as a contiguous rigid body about
// its lowest-indexed atom: every atom's displacement from that reference
// is taken under the minimum-image convention, so a molecule whose atoms
// are wrapped across a periodic boundary (the norm, since positions are
// held in the primary image each step) yields its true centre of mass
// rather than one pulled toward the box centre. Every atom is then
// translated rigidly by `(scale - 1) * COM`, so the molecular COM scales
// about the origin while every intramolecular displacement is unchanged.
// The per-molecule reduction sums atoms in their stored (ascending-index)
// order, so the result is bit-identical across runs on the same GPU.
#include "precision.cuh"
#include "pbc.cuh"

// rq-c83742c0
extern "C" __global__ void mc_barostat_scale_molecule_com(
    Real4 *posq,                            // positions (xyz) + charge (w)
    const unsigned int *mol_atom_offsets,   // length n_mol + 1
    const unsigned int *mol_atom_indices,   // length N, atom ids by molecule
    const Real *masses,                     // length N
    const Real *lattice,                    // lx, ly, lz, xy, xz, yz
    Real scale,
    unsigned int n_mol)
{
  unsigned int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= n_mol) {
    return;
  }
  Real lx = lattice[0], ly = lattice[1], lz = lattice[2];
  Real xy = lattice[3], xz = lattice[4], yz = lattice[5];

  unsigned int lo = mol_atom_offsets[m];
  unsigned int hi = mol_atom_offsets[m + 1u];

  // Reference atom: the lowest-indexed atom of the molecule's slice.
  Real4 ref = posq[mol_atom_indices[lo]];

  Real total_mass = R(0.0);
  Real cx = R(0.0), cy = R(0.0), cz = R(0.0);
  for (unsigned int k = lo; k < hi; ++k) {
    unsigned int a = mol_atom_indices[k];
    Real mass = masses[a];
    Real4 p = posq[a];
    // Minimum-image displacement from the reference, so a molecule split
    // across a periodic boundary is reconstructed contiguously before the
    // mass-weighted average.
    Real dx = p.x - ref.x;
    Real dy = p.y - ref.y;
    Real dz = p.z - ref.z;
    triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);
    cx += mass * dx;
    cy += mass * dy;
    cz += mass * dz;
    total_mass += mass;
  }
  Real inv = R(1.0) / total_mass;
  // COM = ref + (Σ m_i d_i) / (Σ m_i); shift = (scale - 1) * COM.
  Real f = scale - R(1.0);
  Real sx = f * (ref.x + cx * inv);
  Real sy = f * (ref.y + cy * inv);
  Real sz = f * (ref.z + cz * inv);

  for (unsigned int k = lo; k < hi; ++k) {
    unsigned int a = mol_atom_indices[k];
    Real4 p = posq[a];
    p.x += sx;
    p.y += sy;
    p.z += sz;
    posq[a] = p;
  }
}
