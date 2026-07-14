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
//
// `lattice` is the PRE-scale box, and the kernel derives the post-scale box
// as `lattice * scale` itself. Both are needed, and conflating them is a
// correctness trap:
//
//   The atoms are stored wrapped into the primary image, so a molecule
//   straddling a box face has its atoms on opposite sides. Translating each
//   atom by the same vector preserves their *raw* coordinate differences —
//   but a straddling molecule's true bond vector is (raw difference ± L),
//   and every downstream consumer (SETTLE, the exclusion correction, the
//   pair kernel) recovers it by minimum image against the box it is handed.
//   Rescale the box to L' and that reconstruction yields (raw difference
//   ± L') instead: the rigid geometry is silently distorted by ΔL for every
//   molecule that straddles a face. For rigid SPC/E water those are O-H
//   pairs carrying ±0.85/±0.42 e at ~1 Å, so the spurious energy is large,
//   and it is systematic in ΔL — it always favours expansion, which drives
//   the box away without bound.
//
// So: reconstruct contiguously under the OLD box, translate, and re-wrap
// under the NEW box. Wrapping adds integer multiples of the new box vectors,
// which the consumers' minimum-image step undoes exactly, so the molecule's
// internal geometry survives the move unchanged — which is the whole premise
// of a rigid-body COM volume move.
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

  // The post-scale box the moved atoms are wrapped back into.
  Real nlx = lx * scale, nly = ly * scale, nlz = lz * scale;
  Real nxy = xy * scale, nxz = xz * scale, nyz = yz * scale;

  for (unsigned int k = lo; k < hi; ++k) {
    unsigned int a = mol_atom_indices[k];
    Real4 p = posq[a];
    // Rebuild this atom contiguously with the reference under the OLD box,
    // so `d` is the molecule's true (undistorted) internal displacement.
    Real dx = p.x - ref.x;
    Real dy = p.y - ref.y;
    Real dz = p.z - ref.z;
    triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);
    // Translate the contiguous molecule, then wrap into the NEW box. `d` is
    // carried through untouched, which is what keeps the rigid geometry exact.
    Real px = ref.x + dx + sx;
    Real py = ref.y + dy + sy;
    Real pz = ref.z + dz + sz;
    triclinic_min_image(px, py, pz, nlx, nly, nlz, nxy, nxz, nyz);
    p.x = px;
    p.y = py;
    p.z = pz;
    posq[a] = p;
  }
}
