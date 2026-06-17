// rq-19f7ffca rq-d12b8b49

#include "precision.cuh"

#include "pbc.cuh"

// Harmonic angle force kernel: one thread per angle.
//
// For angle m at atoms (i, j, k) with j the centre, this thread
// computes r_ij = r_i - r_j and r_kj = r_k - r_j (after minimum-image
// wrapping), the geometric angle θ at j, and the three per-atom force
// vectors derived from U(θ) = ½ k (θ − θ₀)². The forces are written to
// slots 3·m, 3·m + 1, 3·m + 2 of the angle-triple buffer; the per-atom
// energy share (U_m / 3) and virial share (W_m / 3) are written to all
// three slots so the per-atom reduction recovers U_m and W_m exactly.
extern "C" __global__ void harmonic_angle_force(
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const unsigned int *angles,
    const Real *angle_k_theta,
    const Real *angle_theta_0,
    const Real *lattice,
    Real *angle_triple_x,
    Real *angle_triple_y,
    Real *angle_triple_z,
    Real *angle_triple_energy,
    Real *angle_triple_virial,
    unsigned int n_angles)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  unsigned int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= n_angles) {
    return;
  }

  unsigned int atom_i = angles[4 * m + 0];
  unsigned int atom_j = angles[4 * m + 1];
  unsigned int atom_k = angles[4 * m + 2];
  unsigned int type_idx = angles[4 * m + 3];

  unsigned int slot_i = 3 * m;
  unsigned int slot_j = 3 * m + 1;
  unsigned int slot_k = 3 * m + 2;

  // r_ij = r_i - r_j, r_kj = r_k - r_j (after minimum-image wrap).
  Real dijx = positions_x[atom_i] - positions_x[atom_j];
  Real dijy = positions_y[atom_i] - positions_y[atom_j];
  Real dijz = positions_z[atom_i] - positions_z[atom_j];
  triclinic_min_image(dijx, dijy, dijz, lx, ly, lz, xy, xz, yz);

  Real dkjx = positions_x[atom_k] - positions_x[atom_j];
  Real dkjy = positions_y[atom_k] - positions_y[atom_j];
  Real dkjz = positions_z[atom_k] - positions_z[atom_j];
  triclinic_min_image(dkjx, dkjy, dkjz, lx, ly, lz, xy, xz, yz);

  Real dij2 = dijx * dijx + dijy * dijy + dijz * dijz;
  Real dkj2 = dkjx * dkjx + dkjy * dkjy + dkjz * dkjz;

  // Defensive guards: degenerate geometry → zero everything.
  if (dij2 == R(0.0) || dkj2 == R(0.0)) {
    angle_triple_x[slot_i] = R(0.0);
    angle_triple_y[slot_i] = R(0.0);
    angle_triple_z[slot_i] = R(0.0);
    angle_triple_energy[slot_i] = R(0.0);
    angle_triple_virial[slot_i] = R(0.0);
    angle_triple_x[slot_j] = R(0.0);
    angle_triple_y[slot_j] = R(0.0);
    angle_triple_z[slot_j] = R(0.0);
    angle_triple_energy[slot_j] = R(0.0);
    angle_triple_virial[slot_j] = R(0.0);
    angle_triple_x[slot_k] = R(0.0);
    angle_triple_y[slot_k] = R(0.0);
    angle_triple_z[slot_k] = R(0.0);
    angle_triple_energy[slot_k] = R(0.0);
    angle_triple_virial[slot_k] = R(0.0);
    return;
  }

  Real dij = Real_sqrt(dij2);
  Real dkj = Real_sqrt(dkj2);
  Real inv_dij_dkj = R(1.0) / (dij * dkj);

  Real dot = dijx * dkjx + dijy * dkjy + dijz * dkjz;
  Real cos_theta = dot * inv_dij_dkj;
  // Clamp to avoid sqrt of tiny negative cos_theta² overshoot.
  if (cos_theta > R(1.0)) cos_theta = R(1.0);
  if (cos_theta < -R(1.0)) cos_theta = -R(1.0);
  Real sin_sq = R(1.0) - cos_theta * cos_theta;
  Real sin_theta = Real_sqrt(sin_sq > R(0.0) ? sin_sq : R(0.0));

  // Near-collinear guard.
  if (sin_theta < R(1.0e-7)) {
    angle_triple_x[slot_i] = R(0.0);
    angle_triple_y[slot_i] = R(0.0);
    angle_triple_z[slot_i] = R(0.0);
    angle_triple_energy[slot_i] = R(0.0);
    angle_triple_virial[slot_i] = R(0.0);
    angle_triple_x[slot_j] = R(0.0);
    angle_triple_y[slot_j] = R(0.0);
    angle_triple_z[slot_j] = R(0.0);
    angle_triple_energy[slot_j] = R(0.0);
    angle_triple_virial[slot_j] = R(0.0);
    angle_triple_x[slot_k] = R(0.0);
    angle_triple_y[slot_k] = R(0.0);
    angle_triple_z[slot_k] = R(0.0);
    angle_triple_energy[slot_k] = R(0.0);
    angle_triple_virial[slot_k] = R(0.0);
    return;
  }

  // θ = atan2(|r_ij × r_kj|, r_ij · r_kj). |r_ij × r_kj| = dij·dkj·sin θ.
  Real theta = atan2f(dij * dkj * sin_theta, dot);

  Real k = angle_k_theta[type_idx];
  Real theta_0 = angle_theta_0[type_idx];
  Real dtheta = theta - theta_0;
  // f = −dU/dθ = −k · dθ. Divide by sin θ once for use in the gradient.
  Real g = -k * dtheta / sin_theta;

  // F_i = g · ((cosθ / dij²) · r_ij − (1 / (dij·dkj)) · r_kj)
  // F_k = g · ((cosθ / dkj²) · r_kj − (1 / (dij·dkj)) · r_ij)
  // F_j = −(F_i + F_k)
  Real inv_dij2 = R(1.0) / dij2;
  Real inv_dkj2 = R(1.0) / dkj2;
  Real fix = g * (cos_theta * inv_dij2 * dijx - inv_dij_dkj * dkjx);
  Real fiy = g * (cos_theta * inv_dij2 * dijy - inv_dij_dkj * dkjy);
  Real fiz = g * (cos_theta * inv_dij2 * dijz - inv_dij_dkj * dkjz);
  Real fkx = g * (cos_theta * inv_dkj2 * dkjx - inv_dij_dkj * dijx);
  Real fky = g * (cos_theta * inv_dkj2 * dkjy - inv_dij_dkj * dijy);
  Real fkz = g * (cos_theta * inv_dkj2 * dkjz - inv_dij_dkj * dijz);
  Real fjx = -(fix + fkx);
  Real fjy = -(fiy + fky);
  Real fjz = -(fiz + fkz);

  Real u_m = R(0.5) * k * dtheta * dtheta;
  // Scalar virial W_m = r_ij · F_i + r_kj · F_k.
  Real w_m = (dijx * fix + dijy * fiy + dijz * fiz)
            + (dkjx * fkx + dkjy * fky + dkjz * fkz);

  Real u_share = u_m * (R(1.0) / R(3.0));
  Real w_share = w_m * (R(1.0) / R(3.0));

  angle_triple_x[slot_i] = fix;
  angle_triple_y[slot_i] = fiy;
  angle_triple_z[slot_i] = fiz;
  angle_triple_energy[slot_i] = u_share;
  angle_triple_virial[slot_i] = w_share;

  angle_triple_x[slot_j] = fjx;
  angle_triple_y[slot_j] = fjy;
  angle_triple_z[slot_j] = fjz;
  angle_triple_energy[slot_j] = u_share;
  angle_triple_virial[slot_j] = w_share;

  angle_triple_x[slot_k] = fkx;
  angle_triple_y[slot_k] = fky;
  angle_triple_z[slot_k] = fkz;
  angle_triple_energy[slot_k] = u_share;
  angle_triple_virial[slot_k] = w_share;
}

// Per-atom segmented reduction. One thread per atom sums every
// angle-triple-buffer slot that names this atom. Identical layout to
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
    unsigned int n)
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
    sum_e += angle_triple_energy[slot];
    sum_w += angle_triple_virial[slot];
  }

  slot_force_x[a] = sum_x;
  slot_force_y[a] = sum_y;
  slot_force_z[a] = sum_z;
  slot_energy[a] = sum_e;
  slot_virial[a] = sum_w;
}
