// rq-67e62f4b — SETTLE analytic three-atom rigid-water constraint
// algorithm of Miyamoto & Kollman (J. Comput. Chem. 13, 952 (1992)).
//
// One thread per constraint group. Each group is exactly 3 atoms
// (O, H1, H2) in the order declared by the topology row. The
// `group_atoms` flat array stores atoms as
// [O_idx, H1_idx, H2_idx, O_idx, H1_idx, H2_idx, ...] in group order.
//
// The canonical body-frame positions for each constraint type are
// stored as 3-entry slices in (type_canonical_x, type_canonical_y,
// type_canonical_z): [O_body, H1_body, H2_body]. The slot constructor
// pre-computes these with mass-weighted centroid at the origin and
// with: H-H along the y axis, O on the +x axis (symmetry-plane
// convention). See the `Algorithm` section of `settle.md` for the
// closed-form derivation.

#include "pbc.cuh"

// rq-de7601cd
extern "C" __global__ void settle_snapshot(
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    const unsigned int *group_atoms,
    float *snapshot_x,
    float *snapshot_y,
    float *snapshot_z,
    unsigned int n_groups)
{
  unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_groups) {
    return;
  }
  unsigned int base = 3u * g;
  for (unsigned int a = 0; a < 3; ++a) {
    unsigned int atom_idx = group_atoms[base + a];
    snapshot_x[base + a] = positions_x[atom_idx];
    snapshot_y[base + a] = positions_y[atom_idx];
    snapshot_z[base + a] = positions_z[atom_idx];
  }
}

// Minimum-image displacement helper that always returns the image of
// `b` closest to `a`. Used to bring the three atoms of a water group
// into the same lattice image before the rigid-body solve.
__device__ static inline void min_image_to(
    float ax, float ay, float az,
    float &bx, float &by, float &bz,
    float lx, float ly, float lz,
    float xy, float xz, float yz)
{
  float dx = bx - ax;
  float dy = by - ay;
  float dz = bz - az;
  triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);
  bx = ax + dx;
  by = ay + dy;
  bz = az + dz;
}

// Mass-weighted centroid for a SETTLE-shaped triplet.
__device__ static inline void weighted_com(
    float ox, float oy, float oz,
    float h1x, float h1y, float h1z,
    float h2x, float h2y, float h2z,
    float m_o, float m_h, float total_mass,
    float &cx, float &cy, float &cz)
{
  cx = (m_o * ox + m_h * h1x + m_h * h2x) / total_mass;
  cy = (m_o * oy + m_h * h1y + m_h * h2y) / total_mass;
  cz = (m_o * oz + m_h * h1z + m_h * h2z) / total_mass;
}

// rq-de7601cd
extern "C" __global__ void settle_positions(
    float *positions_x,
    float *positions_y,
    float *positions_z,
    float *velocities_x,
    float *velocities_y,
    float *velocities_z,
    const float *snapshot_x,
    const float *snapshot_y,
    const float *snapshot_z,
    const unsigned int *group_atoms,
    const unsigned int *group_type_index,
    const float *type_canonical_x,
    const float *type_canonical_y,
    const float *type_canonical_z,
    const float *type_mass_o,
    const float *type_mass_h,
    float lx, float ly, float lz, float xy, float xz, float yz,
    float dt,
    float *constraint_virial,
    unsigned int n_groups)
{
  unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_groups) {
    return;
  }
  unsigned int base = 3u * g;
  constraint_virial[base + 0] = 0.0f;
  constraint_virial[base + 1] = 0.0f;
  constraint_virial[base + 2] = 0.0f;
  unsigned int t = group_type_index[g];

  unsigned int a_o = group_atoms[base + 0];
  unsigned int a_h1 = group_atoms[base + 1];
  unsigned int a_h2 = group_atoms[base + 2];

  // Canonical body-frame positions are used only to derive the
  // target pair distances r_OH and r_HH. They are no longer used
  // to place atoms directly: the position projection is the
  // iterative SHAKE solver below, which finds the constrained
  // positions as r_uncon + Σ_k λ_k · ∇_k C_k(r_old). That form is
  // the unique projection in the constraint-gradient subspace, and
  // the corresponding velocity correction `v ← v + Δr/dt`
  // therefore implements RATTLE-consistent leapfrog. Replacing the
  // Miyamoto-Kollman analytical rotation with SHAKE removes the
  // tangential-Δr component that breaks the energy conservation
  // of the constrained velocity-Verlet integrator.
  float ox_b = type_canonical_x[3u * t + 0];
  float h1x_b = type_canonical_x[3u * t + 1];
  float h1y_b = type_canonical_y[3u * t + 1];
  float h2y_b = type_canonical_y[3u * t + 2];
  float r_OH2 =
      (ox_b - h1x_b) * (ox_b - h1x_b) + h1y_b * h1y_b;
  // Canonical H1 and H2 have the same x and z (z = 0), so
  // r_HH² = (h1y_b - h2y_b)² = 4 · h2y_b² (since h1y_b = -h2y_b).
  float r_HH2 = 4.0f * h2y_b * h2y_b;

  float m_o = type_mass_o[t];
  float m_h = type_mass_h[t];
  float inv_m_o = 1.0f / m_o;
  float inv_m_h = 1.0f / m_h;

  // Pre-drift positions (gradients of C are evaluated here).
  float ox0 = snapshot_x[base + 0];
  float oy0 = snapshot_y[base + 0];
  float oz0 = snapshot_z[base + 0];
  float h1x0 = snapshot_x[base + 1];
  float h1y0 = snapshot_y[base + 1];
  float h1z0 = snapshot_z[base + 1];
  float h2x0 = snapshot_x[base + 2];
  float h2y0 = snapshot_y[base + 2];
  float h2z0 = snapshot_z[base + 2];
  min_image_to(ox0, oy0, oz0, h1x0, h1y0, h1z0, lx, ly, lz, xy, xz, yz);
  min_image_to(ox0, oy0, oz0, h2x0, h2y0, h2z0, lx, ly, lz, xy, xz, yz);

  // Pre-drift constraint-gradient directions (per-constraint, paired
  // atoms; sign convention: r_i_old − r_j_old for the (i, j) pair).
  float g_oh1_x = ox0 - h1x0, g_oh1_y = oy0 - h1y0, g_oh1_z = oz0 - h1z0;
  float g_oh2_x = ox0 - h2x0, g_oh2_y = oy0 - h2y0, g_oh2_z = oz0 - h2z0;
  float g_hh_x  = h1x0 - h2x0, g_hh_y  = h1y0 - h2y0, g_hh_z  = h1z0 - h2z0;

  // Unconstrained post-drift positions.
  float ox_u = positions_x[a_o];
  float oy_u = positions_y[a_o];
  float oz_u = positions_z[a_o];
  float h1x_u = positions_x[a_h1];
  float h1y_u = positions_y[a_h1];
  float h1z_u = positions_z[a_h1];
  float h2x_u = positions_x[a_h2];
  float h2y_u = positions_y[a_h2];
  float h2z_u = positions_z[a_h2];
  min_image_to(ox_u, oy_u, oz_u, h1x_u, h1y_u, h1z_u, lx, ly, lz, xy, xz, yz);
  min_image_to(ox_u, oy_u, oz_u, h2x_u, h2y_u, h2z_u, lx, ly, lz, xy, xz, yz);

  // SHAKE iterates. Start at the unconstrained positions; each
  // iteration walks the three constraints in fixed order and
  // applies the Lagrange-multiplier-style correction
  //   r_i ← r_i − λ · (r_i_old − r_j_old) / m_i
  //   r_j ← r_j + λ · (r_i_old − r_j_old) / m_j
  // where λ = σ / (2 · (r_i − r_j) · (r_i_old − r_j_old) · (1/m_i + 1/m_j))
  // and σ = |r_i − r_j|² − r_target². Convergence is quadratic for
  // small displacements and typically completes in well under 10
  // sweeps for rigid water at thermal MD step sizes.
  float ox_c = ox_u, oy_c = oy_u, oz_c = oz_u;
  float h1x_c = h1x_u, h1y_c = h1y_u, h1z_c = h1z_u;
  float h2x_c = h2x_u, h2y_c = h2y_u, h2z_c = h2z_u;
  // Absolute tolerance: ~10⁻⁶ relative on a (10⁻¹⁰ m)² distance.
  const float SHAKE_TOL2 = 1.0e-26f;
  const int SHAKE_MAX_ITER = 32;
  float inv_oh = inv_m_o + inv_m_h;
  float inv_hh = inv_m_h + inv_m_h;
  for (int iter = 0; iter < SHAKE_MAX_ITER; ++iter) {
    bool converged = true;

    // O–H1.
    float dx = ox_c - h1x_c;
    float dy = oy_c - h1y_c;
    float dz = oz_c - h1z_c;
    float dist2 = dx * dx + dy * dy + dz * dz;
    float sigma = dist2 - r_OH2;
    if (fabsf(sigma) > SHAKE_TOL2) {
      converged = false;
      float ddot = dx * g_oh1_x + dy * g_oh1_y + dz * g_oh1_z;
      float lambda = sigma / (2.0f * ddot * inv_oh);
      ox_c  -= lambda * g_oh1_x * inv_m_o;
      oy_c  -= lambda * g_oh1_y * inv_m_o;
      oz_c  -= lambda * g_oh1_z * inv_m_o;
      h1x_c += lambda * g_oh1_x * inv_m_h;
      h1y_c += lambda * g_oh1_y * inv_m_h;
      h1z_c += lambda * g_oh1_z * inv_m_h;
    }

    // O–H2.
    dx = ox_c - h2x_c;
    dy = oy_c - h2y_c;
    dz = oz_c - h2z_c;
    dist2 = dx * dx + dy * dy + dz * dz;
    sigma = dist2 - r_OH2;
    if (fabsf(sigma) > SHAKE_TOL2) {
      converged = false;
      float ddot = dx * g_oh2_x + dy * g_oh2_y + dz * g_oh2_z;
      float lambda = sigma / (2.0f * ddot * inv_oh);
      ox_c  -= lambda * g_oh2_x * inv_m_o;
      oy_c  -= lambda * g_oh2_y * inv_m_o;
      oz_c  -= lambda * g_oh2_z * inv_m_o;
      h2x_c += lambda * g_oh2_x * inv_m_h;
      h2y_c += lambda * g_oh2_y * inv_m_h;
      h2z_c += lambda * g_oh2_z * inv_m_h;
    }

    // H1–H2.
    dx = h1x_c - h2x_c;
    dy = h1y_c - h2y_c;
    dz = h1z_c - h2z_c;
    dist2 = dx * dx + dy * dy + dz * dz;
    sigma = dist2 - r_HH2;
    if (fabsf(sigma) > SHAKE_TOL2) {
      converged = false;
      float ddot = dx * g_hh_x + dy * g_hh_y + dz * g_hh_z;
      float lambda = sigma / (2.0f * ddot * inv_hh);
      h1x_c -= lambda * g_hh_x * inv_m_h;
      h1y_c -= lambda * g_hh_y * inv_m_h;
      h1z_c -= lambda * g_hh_z * inv_m_h;
      h2x_c += lambda * g_hh_x * inv_m_h;
      h2y_c += lambda * g_hh_y * inv_m_h;
      h2z_c += lambda * g_hh_z * inv_m_h;
    }

    if (converged) {
      break;
    }
  }

  // COM, for the constraint-virial computation below. By
  // construction SHAKE preserves the mass-weighted centre of mass
  // (every λ·g term contributes equal-and-opposite displacements
  // to the two atoms of its constraint, weighted by 1/m).
  float total_mass = m_o + 2.0f * m_h;
  float cx, cy, cz;
  weighted_com(ox_c, oy_c, oz_c, h1x_c, h1y_c, h1z_c, h2x_c, h2y_c, h2z_c,
               m_o, m_h, total_mass, cx, cy, cz);

  // Update half-step velocities: v ← v + (r_constrained - r_unconstrained)/dt.
  // (Use the un-image-fixed positions buffer for the post-drift values
  // since the SETTLE-corrected positions are in the same image fix-up;
  // both have the same difference vector by construction.)
  float inv_dt = (dt != 0.0f) ? (1.0f / dt) : 0.0f;
  velocities_x[a_o] += (ox_c - ox_u) * inv_dt;
  velocities_y[a_o] += (oy_c - oy_u) * inv_dt;
  velocities_z[a_o] += (oz_c - oz_u) * inv_dt;
  velocities_x[a_h1] += (h1x_c - h1x_u) * inv_dt;
  velocities_y[a_h1] += (h1y_c - h1y_u) * inv_dt;
  velocities_z[a_h1] += (h1z_c - h1z_u) * inv_dt;
  velocities_x[a_h2] += (h2x_c - h2x_u) * inv_dt;
  velocities_y[a_h2] += (h2y_c - h2y_u) * inv_dt;
  velocities_z[a_h2] += (h2z_c - h2z_u) * inv_dt;

  // Write constrained positions back. The image fix-up may have
  // shifted the post-drift hydrogens; we restore them by adding back
  // the same image offset. Since the corrected positions track the
  // unconstrained ones to within a small displacement, the original
  // image-flag bookkeeping remains valid.
  positions_x[a_o] = ox_c;
  positions_y[a_o] = oy_c;
  positions_z[a_o] = oz_c;
  positions_x[a_h1] = positions_x[a_h1] + (h1x_c - h1x_u);
  positions_y[a_h1] = positions_y[a_h1] + (h1y_c - h1y_u);
  positions_z[a_h1] = positions_z[a_h1] + (h1z_c - h1z_u);
  positions_x[a_h2] = positions_x[a_h2] + (h2x_c - h2x_u);
  positions_y[a_h2] = positions_y[a_h2] + (h2y_c - h2y_u);
  positions_z[a_h2] = positions_z[a_h2] + (h2z_c - h2z_u);

  // Constraint-virial contribution from this group's position
  // correction. The total scalar constraint virial for one group is
  //   W_g = Σ_i m_i · (Δr_i · r_constrained_i) / dt²
  // (equivalent to Σ_atoms F_constraint_i · r_i; the per-atom F·r
  // sum is translation-invariant within the group because the
  // analytic SETTLE solve preserves the mass-weighted COM, so
  // Σ_i m_i Δr_i = 0.) We compute the sum using
  // COM-relative positions to avoid f32 catastrophic cancellation:
  // when the molecule sits several nm from the origin, the absolute
  // positions are O(10⁻⁹ m) while Δr is O(10⁻¹² m), and a direct
  // m·Δr·r_lab sum loses precision long before the inter-atomic
  // cancellation reduces it to the physical O(kT) virial. The
  // group's mass-weighted COM (cx, cy, cz) is already computed
  // above, so the per-atom contribution can be written in
  // body-relative coordinates with no extra arithmetic.
  //
  // Critical ordering: the per-atom expression `m · (Δr · r) / dt²`
  // has m ≈ 10⁻²⁷ kg, (Δr · r) ≈ 10⁻²³ m², and 1/dt² ≈ 10³⁰. The
  // associative grouping `(m · (Δr · r)) / dt²` produces the
  // intermediate `m · (Δr · r) ≈ 10⁻⁵⁰`, which underflows to zero
  // in f32 (smallest denormal ≈ 1.4·10⁻⁴⁵). The associativity-
  // preserving grouping `(m / dt²) · (Δr · r)` keeps every
  // intermediate well inside f32 normal range: `m / dt² ≈ 10³`,
  // `(Δr · r) ≈ 10⁻²³`, product ≈ 10⁻²⁰ J. We therefore precompute
  // the per-mass `m · inv_dt²` scale once and multiply the dot
  // product by it.
  float inv_dt2 = (dt != 0.0f) ? (1.0f / (dt * dt)) : 0.0f;
  float scale_o = m_o * inv_dt2;
  float scale_h = m_h * inv_dt2;
  float dox = ox_c - ox_u;
  float doy = oy_c - oy_u;
  float doz = oz_c - oz_u;
  float dh1x = h1x_c - h1x_u;
  float dh1y = h1y_c - h1y_u;
  float dh1z = h1z_c - h1z_u;
  float dh2x = h2x_c - h2x_u;
  float dh2y = h2y_c - h2y_u;
  float dh2z = h2z_c - h2z_u;
  // COM-relative constrained positions: subtract the unconstrained
  // COM (cx, cy, cz) which equals the constrained COM by SETTLE's
  // COM-preservation invariant.
  float rox = ox_c - cx, roy = oy_c - cy, roz = oz_c - cz;
  float rh1x = h1x_c - cx, rh1y = h1y_c - cy, rh1z = h1z_c - cz;
  float rh2x = h2x_c - cx, rh2y = h2y_c - cy, rh2z = h2z_c - cz;
  constraint_virial[base + 0] =
      scale_o * (dox * rox + doy * roy + doz * roz);
  constraint_virial[base + 1] =
      scale_h * (dh1x * rh1x + dh1y * rh1y + dh1z * rh1z);
  constraint_virial[base + 2] =
      scale_h * (dh2x * rh2x + dh2y * rh2y + dh2z * rh2z);
}

// Scatter the slot-cached constraint virials computed by
// `settle_positions` into `particle_virials` so the barostat's
// scalar-virial reduction picks them up. One thread per atom slot
// (`n_atom_slots = 3 * n_groups`); each constraint group has a
// disjoint atom set so no atomics are needed.
extern "C" __global__ void settle_virial_scatter(
    const float *constraint_virial,
    const unsigned int *group_atoms,
    float *particle_virials,
    unsigned int n_atom_slots)
{
  unsigned int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n_atom_slots) {
    return;
  }
  unsigned int atom_index = group_atoms[s];
  particle_virials[atom_index] = particle_virials[atom_index] + constraint_virial[s];
}

// Project the post-kick velocities onto the constraint manifold by
// solving the 3x3 linear system for the three Lagrange multipliers
// associated with the three rigid-water constraints. The system is
// linear because the constraints are bilinear in the velocity
// corrections and the constraint gradients (r_i - r_j) are known.
// rq-de7601cd
extern "C" __global__ void settle_velocities(
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    float *velocities_x,
    float *velocities_y,
    float *velocities_z,
    const unsigned int *group_atoms,
    const unsigned int *group_type_index,
    const float *type_mass_o,
    const float *type_mass_h,
    float lx, float ly, float lz, float xy, float xz, float yz,
    unsigned int n_groups)
{
  unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_groups) {
    return;
  }
  unsigned int base = 3u * g;
  unsigned int t = group_type_index[g];

  unsigned int a_o = group_atoms[base + 0];
  unsigned int a_h1 = group_atoms[base + 1];
  unsigned int a_h2 = group_atoms[base + 2];

  float m_o = type_mass_o[t];
  float m_h = type_mass_h[t];
  float inv_m_o = 1.0f / m_o;
  float inv_m_h = 1.0f / m_h;

  // Constraint displacement vectors (minimum-image to keep the rigid
  // triangle coherent across periodic boundaries).
  float ox = positions_x[a_o], oy = positions_y[a_o], oz = positions_z[a_o];
  float h1x = positions_x[a_h1], h1y = positions_y[a_h1], h1z = positions_z[a_h1];
  float h2x = positions_x[a_h2], h2y = positions_y[a_h2], h2z = positions_z[a_h2];
  min_image_to(ox, oy, oz, h1x, h1y, h1z, lx, ly, lz, xy, xz, yz);
  min_image_to(ox, oy, oz, h2x, h2y, h2z, lx, ly, lz, xy, xz, yz);

  // Constraint directions: r_O - r_H1, r_O - r_H2, r_H1 - r_H2.
  float d1x = ox - h1x, d1y = oy - h1y, d1z = oz - h1z;  // O - H1
  float d2x = ox - h2x, d2y = oy - h2y, d2z = oz - h2z;  // O - H2
  float d3x = h1x - h2x, d3y = h1y - h2y, d3z = h1z - h2z;  // H1 - H2

  float v_o_x = velocities_x[a_o], v_o_y = velocities_y[a_o], v_o_z = velocities_z[a_o];
  float v_h1_x = velocities_x[a_h1], v_h1_y = velocities_y[a_h1], v_h1_z = velocities_z[a_h1];
  float v_h2_x = velocities_x[a_h2], v_h2_y = velocities_y[a_h2], v_h2_z = velocities_z[a_h2];

  // Relative velocity components along each constraint direction.
  float vrel1 = (v_o_x - v_h1_x) * d1x + (v_o_y - v_h1_y) * d1y + (v_o_z - v_h1_z) * d1z;
  float vrel2 = (v_o_x - v_h2_x) * d2x + (v_o_y - v_h2_y) * d2y + (v_o_z - v_h2_z) * d2z;
  float vrel3 = (v_h1_x - v_h2_x) * d3x + (v_h1_y - v_h2_y) * d3y + (v_h1_z - v_h2_z) * d3z;

  // RHS of M λ = vrel (we solve for λ).
  // The matrix M is 3x3 symmetric with:
  //   M[k][l] = sum over atoms of (sign of atom in constraint k) *
  //            (sign of atom in constraint l) * (d_k · d_l) / m_atom
  // For constraint 1 (O, H1): O carries +1/m_o, H1 carries -1/m_h.
  // For constraint 2 (O, H2): O carries +1/m_o, H2 carries -1/m_h.
  // For constraint 3 (H1, H2): H1 carries +1/m_h, H2 carries -1/m_h.
  float d11 = d1x * d1x + d1y * d1y + d1z * d1z;
  float d12 = d1x * d2x + d1y * d2y + d1z * d2z;
  float d13 = d1x * d3x + d1y * d3y + d1z * d3z;
  float d22 = d2x * d2x + d2y * d2y + d2z * d2z;
  float d23 = d2x * d3x + d2y * d3y + d2z * d3z;
  float d33 = d3x * d3x + d3y * d3y + d3z * d3z;

  // Mass coupling matrix M[k][l] entries.
  // Atoms shared between constraints k and l contribute (±)(±)/m.
  // constraint 1 has atoms (O, H1) with coefficients (+1, -1)/m
  // constraint 2 has atoms (O, H2) with coefficients (+1, -1)/m
  // constraint 3 has atoms (H1, H2) with coefficients (+1, -1)/m
  // (where 'm' is the corresponding atom's mass)
  //
  // M[1][1] = d11 * (1/m_o + 1/m_h)
  // M[1][2] = d12 * (1/m_o)              [shared atom: O, signs +,+]
  // M[1][3] = d13 * (-1/m_h)             [shared atom: H1, signs -,+]
  // M[2][2] = d22 * (1/m_o + 1/m_h)
  // M[2][3] = d23 * (1/m_h)              [shared atom: H2, signs -,-]
  // M[3][3] = d33 * (1/m_h + 1/m_h)
  float M00 = d11 * (inv_m_o + inv_m_h);
  float M01 = d12 * inv_m_o;
  float M02 = -d13 * inv_m_h;
  float M11 = d22 * (inv_m_o + inv_m_h);
  float M12 = d23 * inv_m_h;
  float M22 = d33 * 2.0f * inv_m_h;

  // Solve the symmetric 3x3 system M λ = vrel via Cramer's rule.
  float a = M00, b = M01, c = M02;
  float d = M01, e = M11, f = M12;
  float gg = M02, h = M12, i = M22;
  float det = a * (e * i - f * h) - b * (d * i - f * gg) + c * (d * h - e * gg);
  if (fabsf(det) < 1.0e-30f) {
    return;
  }
  float inv_det = 1.0f / det;
  float r1 = vrel1, r2 = vrel2, r3 = vrel3;
  float l1 = (r1 * (e * i - f * h) - b * (r2 * i - f * r3) + c * (r2 * h - e * r3)) * inv_det;
  float l2 = (a * (r2 * i - f * r3) - r1 * (d * i - f * gg) + c * (d * r3 - r2 * gg)) * inv_det;
  float l3 = (a * (e * r3 - r2 * h) - b * (d * r3 - r2 * gg) + r1 * (d * h - e * gg)) * inv_det;

  // Apply velocity corrections: v_atom -= sum_k λ_k * sign(atom in k) * d_k / m_atom
  velocities_x[a_o] = v_o_x - (l1 * d1x + l2 * d2x) * inv_m_o;
  velocities_y[a_o] = v_o_y - (l1 * d1y + l2 * d2y) * inv_m_o;
  velocities_z[a_o] = v_o_z - (l1 * d1z + l2 * d2z) * inv_m_o;
  velocities_x[a_h1] = v_h1_x + (l1 * d1x - l3 * d3x) * inv_m_h;
  velocities_y[a_h1] = v_h1_y + (l1 * d1y - l3 * d3y) * inv_m_h;
  velocities_z[a_h1] = v_h1_z + (l1 * d1z - l3 * d3z) * inv_m_h;
  velocities_x[a_h2] = v_h2_x + (l2 * d2x + l3 * d3x) * inv_m_h;
  velocities_y[a_h2] = v_h2_y + (l2 * d2y + l3 * d3y) * inv_m_h;
  velocities_z[a_h2] = v_h2_z + (l2 * d2z + l3 * d3z) * inv_m_h;
}

// Position-only SHAKE projection. Used by the minimization runner
// after each trial position update. Differs from `settle_positions`
// in two ways: (1) no pre-drift snapshot — the constraint-gradient
// directions are evaluated at the *current* (off-manifold) positions
// rather than at a snapshot, since minimization has no notion of a
// pre-drift configuration; (2) no velocity correction, no virial
// contribution. Velocities and virials are untouched.
extern "C" __global__ void settle_positions_no_velocity(
    float *positions_x,
    float *positions_y,
    float *positions_z,
    const unsigned int *group_atoms,
    const unsigned int *group_type_index,
    const float *type_canonical_x,
    const float *type_canonical_y,
    const float *type_canonical_z,
    const float *type_mass_o,
    const float *type_mass_h,
    float lx, float ly, float lz, float xy, float xz, float yz,
    unsigned int n_groups)
{
  unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_groups) {
    return;
  }
  unsigned int base = 3u * g;
  unsigned int t = group_type_index[g];

  unsigned int a_o = group_atoms[base + 0];
  unsigned int a_h1 = group_atoms[base + 1];
  unsigned int a_h2 = group_atoms[base + 2];

  // Derive target pair distances from the canonical body-frame
  // positions (same convention as `settle_positions`).
  float ox_b = type_canonical_x[3u * t + 0];
  float h1x_b = type_canonical_x[3u * t + 1];
  float h1y_b = type_canonical_y[3u * t + 1];
  float h2y_b = type_canonical_y[3u * t + 2];
  float r_OH2 =
      (ox_b - h1x_b) * (ox_b - h1x_b) + h1y_b * h1y_b;
  float r_HH2 = 4.0f * h2y_b * h2y_b;

  float m_o = type_mass_o[t];
  float m_h = type_mass_h[t];
  float inv_m_o = 1.0f / m_o;
  float inv_m_h = 1.0f / m_h;
  float inv_oh = inv_m_o + inv_m_h;
  float inv_hh = inv_m_h + inv_m_h;

  // Read current (off-manifold) positions and bring all three atoms
  // into the same image relative to O.
  float ox_u = positions_x[a_o];
  float oy_u = positions_y[a_o];
  float oz_u = positions_z[a_o];
  float h1x_u = positions_x[a_h1];
  float h1y_u = positions_y[a_h1];
  float h1z_u = positions_z[a_h1];
  float h2x_u = positions_x[a_h2];
  float h2y_u = positions_y[a_h2];
  float h2z_u = positions_z[a_h2];
  min_image_to(ox_u, oy_u, oz_u, h1x_u, h1y_u, h1z_u, lx, ly, lz, xy, xz, yz);
  min_image_to(ox_u, oy_u, oz_u, h2x_u, h2y_u, h2z_u, lx, ly, lz, xy, xz, yz);

  // Use the current positions themselves as the reference frame for
  // the constraint-gradient directions. This is the standard SHAKE
  // formulation when no pre-drift snapshot is available: the
  // Lagrange-multiplier solve uses ∇C evaluated at the off-manifold
  // configuration. Quadratic convergence is preserved for small
  // displacements, which is the regime minimization produces under
  // a well-chosen `step` size.
  float g_oh1_x = ox_u - h1x_u, g_oh1_y = oy_u - h1y_u, g_oh1_z = oz_u - h1z_u;
  float g_oh2_x = ox_u - h2x_u, g_oh2_y = oy_u - h2y_u, g_oh2_z = oz_u - h2z_u;
  float g_hh_x  = h1x_u - h2x_u, g_hh_y  = h1y_u - h2y_u, g_hh_z  = h1z_u - h2z_u;

  float ox_c = ox_u, oy_c = oy_u, oz_c = oz_u;
  float h1x_c = h1x_u, h1y_c = h1y_u, h1z_c = h1z_u;
  float h2x_c = h2x_u, h2y_c = h2y_u, h2z_c = h2z_u;
  const float SHAKE_TOL2 = 1.0e-26f;
  const int SHAKE_MAX_ITER = 32;
  for (int iter = 0; iter < SHAKE_MAX_ITER; ++iter) {
    bool converged = true;

    // O–H1.
    float dx = ox_c - h1x_c;
    float dy = oy_c - h1y_c;
    float dz = oz_c - h1z_c;
    float dist2 = dx * dx + dy * dy + dz * dz;
    float sigma = dist2 - r_OH2;
    if (fabsf(sigma) > SHAKE_TOL2) {
      converged = false;
      float ddot = dx * g_oh1_x + dy * g_oh1_y + dz * g_oh1_z;
      float lambda = sigma / (2.0f * ddot * inv_oh);
      ox_c  -= lambda * g_oh1_x * inv_m_o;
      oy_c  -= lambda * g_oh1_y * inv_m_o;
      oz_c  -= lambda * g_oh1_z * inv_m_o;
      h1x_c += lambda * g_oh1_x * inv_m_h;
      h1y_c += lambda * g_oh1_y * inv_m_h;
      h1z_c += lambda * g_oh1_z * inv_m_h;
    }

    // O–H2.
    dx = ox_c - h2x_c;
    dy = oy_c - h2y_c;
    dz = oz_c - h2z_c;
    dist2 = dx * dx + dy * dy + dz * dz;
    sigma = dist2 - r_OH2;
    if (fabsf(sigma) > SHAKE_TOL2) {
      converged = false;
      float ddot = dx * g_oh2_x + dy * g_oh2_y + dz * g_oh2_z;
      float lambda = sigma / (2.0f * ddot * inv_oh);
      ox_c  -= lambda * g_oh2_x * inv_m_o;
      oy_c  -= lambda * g_oh2_y * inv_m_o;
      oz_c  -= lambda * g_oh2_z * inv_m_o;
      h2x_c += lambda * g_oh2_x * inv_m_h;
      h2y_c += lambda * g_oh2_y * inv_m_h;
      h2z_c += lambda * g_oh2_z * inv_m_h;
    }

    // H1–H2.
    dx = h1x_c - h2x_c;
    dy = h1y_c - h2y_c;
    dz = h1z_c - h2z_c;
    dist2 = dx * dx + dy * dy + dz * dz;
    sigma = dist2 - r_HH2;
    if (fabsf(sigma) > SHAKE_TOL2) {
      converged = false;
      float ddot = dx * g_hh_x + dy * g_hh_y + dz * g_hh_z;
      float lambda = sigma / (2.0f * ddot * inv_hh);
      h1x_c -= lambda * g_hh_x * inv_m_h;
      h1y_c -= lambda * g_hh_y * inv_m_h;
      h1z_c -= lambda * g_hh_z * inv_m_h;
      h2x_c += lambda * g_hh_x * inv_m_h;
      h2y_c += lambda * g_hh_y * inv_m_h;
      h2z_c += lambda * g_hh_z * inv_m_h;
    }

    if (converged) {
      break;
    }
  }

  // Write constrained positions back. The hydrogens may have been
  // image-shifted relative to O for the SHAKE solve; the shift below
  // re-applies the same offset to the original position values so the
  // final positions remain in the same image as the unconstrained
  // ones.
  positions_x[a_o] = ox_c;
  positions_y[a_o] = oy_c;
  positions_z[a_o] = oz_c;
  positions_x[a_h1] = positions_x[a_h1] + (h1x_c - h1x_u);
  positions_y[a_h1] = positions_y[a_h1] + (h1y_c - h1y_u);
  positions_z[a_h1] = positions_z[a_h1] + (h1z_c - h1z_u);
  positions_x[a_h2] = positions_x[a_h2] + (h2x_c - h2x_u);
  positions_y[a_h2] = positions_y[a_h2] + (h2y_c - h2y_u);
  positions_z[a_h2] = positions_z[a_h2] + (h2z_c - h2z_u);
}
