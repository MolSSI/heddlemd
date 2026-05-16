// rq-19f7ffca rq-d12b8b49

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
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    const unsigned int *angles,
    const float *angle_k_theta,
    const float *angle_theta_0,
    float lx, float ly, float lz, float xy, float xz, float yz,
    float *angle_triple_x,
    float *angle_triple_y,
    float *angle_triple_z,
    float *angle_triple_energy,
    float *angle_triple_virial,
    unsigned int n_angles)
{
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
  float dijx = positions_x[atom_i] - positions_x[atom_j];
  float dijy = positions_y[atom_i] - positions_y[atom_j];
  float dijz = positions_z[atom_i] - positions_z[atom_j];
  triclinic_min_image(dijx, dijy, dijz, lx, ly, lz, xy, xz, yz);

  float dkjx = positions_x[atom_k] - positions_x[atom_j];
  float dkjy = positions_y[atom_k] - positions_y[atom_j];
  float dkjz = positions_z[atom_k] - positions_z[atom_j];
  triclinic_min_image(dkjx, dkjy, dkjz, lx, ly, lz, xy, xz, yz);

  float dij2 = dijx * dijx + dijy * dijy + dijz * dijz;
  float dkj2 = dkjx * dkjx + dkjy * dkjy + dkjz * dkjz;

  // Defensive guards: degenerate geometry → zero everything.
  if (dij2 == 0.0f || dkj2 == 0.0f) {
    angle_triple_x[slot_i] = 0.0f;
    angle_triple_y[slot_i] = 0.0f;
    angle_triple_z[slot_i] = 0.0f;
    angle_triple_energy[slot_i] = 0.0f;
    angle_triple_virial[slot_i] = 0.0f;
    angle_triple_x[slot_j] = 0.0f;
    angle_triple_y[slot_j] = 0.0f;
    angle_triple_z[slot_j] = 0.0f;
    angle_triple_energy[slot_j] = 0.0f;
    angle_triple_virial[slot_j] = 0.0f;
    angle_triple_x[slot_k] = 0.0f;
    angle_triple_y[slot_k] = 0.0f;
    angle_triple_z[slot_k] = 0.0f;
    angle_triple_energy[slot_k] = 0.0f;
    angle_triple_virial[slot_k] = 0.0f;
    return;
  }

  float dij = sqrtf(dij2);
  float dkj = sqrtf(dkj2);
  float inv_dij_dkj = 1.0f / (dij * dkj);

  float dot = dijx * dkjx + dijy * dkjy + dijz * dkjz;
  float cos_theta = dot * inv_dij_dkj;
  // Clamp to avoid sqrt of tiny negative cos_theta² overshoot.
  if (cos_theta > 1.0f) cos_theta = 1.0f;
  if (cos_theta < -1.0f) cos_theta = -1.0f;
  float sin_sq = 1.0f - cos_theta * cos_theta;
  float sin_theta = sqrtf(sin_sq > 0.0f ? sin_sq : 0.0f);

  // Near-collinear guard.
  if (sin_theta < 1.0e-7f) {
    angle_triple_x[slot_i] = 0.0f;
    angle_triple_y[slot_i] = 0.0f;
    angle_triple_z[slot_i] = 0.0f;
    angle_triple_energy[slot_i] = 0.0f;
    angle_triple_virial[slot_i] = 0.0f;
    angle_triple_x[slot_j] = 0.0f;
    angle_triple_y[slot_j] = 0.0f;
    angle_triple_z[slot_j] = 0.0f;
    angle_triple_energy[slot_j] = 0.0f;
    angle_triple_virial[slot_j] = 0.0f;
    angle_triple_x[slot_k] = 0.0f;
    angle_triple_y[slot_k] = 0.0f;
    angle_triple_z[slot_k] = 0.0f;
    angle_triple_energy[slot_k] = 0.0f;
    angle_triple_virial[slot_k] = 0.0f;
    return;
  }

  // θ = atan2(|r_ij × r_kj|, r_ij · r_kj). |r_ij × r_kj| = dij·dkj·sin θ.
  float theta = atan2f(dij * dkj * sin_theta, dot);

  float k = angle_k_theta[type_idx];
  float theta_0 = angle_theta_0[type_idx];
  float dtheta = theta - theta_0;
  // f = −dU/dθ = −k · dθ. Divide by sin θ once for use in the gradient.
  float g = -k * dtheta / sin_theta;

  // F_i = g · ((cosθ / dij²) · r_ij − (1 / (dij·dkj)) · r_kj)
  // F_k = g · ((cosθ / dkj²) · r_kj − (1 / (dij·dkj)) · r_ij)
  // F_j = −(F_i + F_k)
  float inv_dij2 = 1.0f / dij2;
  float inv_dkj2 = 1.0f / dkj2;
  float fix = g * (cos_theta * inv_dij2 * dijx - inv_dij_dkj * dkjx);
  float fiy = g * (cos_theta * inv_dij2 * dijy - inv_dij_dkj * dkjy);
  float fiz = g * (cos_theta * inv_dij2 * dijz - inv_dij_dkj * dkjz);
  float fkx = g * (cos_theta * inv_dkj2 * dkjx - inv_dij_dkj * dijx);
  float fky = g * (cos_theta * inv_dkj2 * dkjy - inv_dij_dkj * dijy);
  float fkz = g * (cos_theta * inv_dkj2 * dkjz - inv_dij_dkj * dijz);
  float fjx = -(fix + fkx);
  float fjy = -(fiy + fky);
  float fjz = -(fiz + fkz);

  float u_m = 0.5f * k * dtheta * dtheta;
  // Scalar virial W_m = r_ij · F_i + r_kj · F_k.
  float w_m = (dijx * fix + dijy * fiy + dijz * fiz)
            + (dkjx * fkx + dkjy * fky + dkjz * fkz);

  float u_share = u_m * (1.0f / 3.0f);
  float w_share = w_m * (1.0f / 3.0f);

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
    const float *angle_triple_x,
    const float *angle_triple_y,
    const float *angle_triple_z,
    const float *angle_triple_energy,
    const float *angle_triple_virial,
    const unsigned int *atom_angle_offsets,
    const unsigned int *atom_angle_indices,
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

  unsigned int start = atom_angle_offsets[a];
  unsigned int end = atom_angle_offsets[a + 1];

  float sum_x = 0.0f;
  float sum_y = 0.0f;
  float sum_z = 0.0f;
  float sum_e = 0.0f;
  float sum_w = 0.0f;

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
