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
  // Zero this group's slot in `constraint_virial` up front. On the
  // success path each slot is overwritten with the computed
  // per-atom virial contribution before kernel exit; on a degenerate
  // early-return path the slot remains zero, which matches the
  // no-op correction those paths perform.
  constraint_virial[base + 0] = 0.0f;
  constraint_virial[base + 1] = 0.0f;
  constraint_virial[base + 2] = 0.0f;
  unsigned int t = group_type_index[g];

  unsigned int a_o = group_atoms[base + 0];
  unsigned int a_h1 = group_atoms[base + 1];
  unsigned int a_h2 = group_atoms[base + 2];

  // Canonical body-frame positions (mass-weighted COM at origin).
  // Stored layout: indices 3t+0 = O, 3t+1 = H1, 3t+2 = H2.
  float ox_b = type_canonical_x[3u * t + 0];
  float oy_b = type_canonical_y[3u * t + 0];
  float oz_b = type_canonical_z[3u * t + 0];
  float h1x_b = type_canonical_x[3u * t + 1];
  float h1y_b = type_canonical_y[3u * t + 1];
  float h1z_b = type_canonical_z[3u * t + 1];
  float h2x_b = type_canonical_x[3u * t + 2];
  float h2y_b = type_canonical_y[3u * t + 2];
  float h2z_b = type_canonical_z[3u * t + 2];

  float m_o = type_mass_o[t];
  float m_h = type_mass_h[t];
  float total_mass = m_o + 2.0f * m_h;

  // Pre-drift positions in the same image as the snapshot.
  float ox0 = snapshot_x[base + 0];
  float oy0 = snapshot_y[base + 0];
  float oz0 = snapshot_z[base + 0];
  float h1x0 = snapshot_x[base + 1];
  float h1y0 = snapshot_y[base + 1];
  float h1z0 = snapshot_z[base + 1];
  float h2x0 = snapshot_x[base + 2];
  float h2y0 = snapshot_y[base + 2];
  float h2z0 = snapshot_z[base + 2];
  // Bring snapshot hydrogens into the oxygen's image.
  min_image_to(ox0, oy0, oz0, h1x0, h1y0, h1z0, lx, ly, lz, xy, xz, yz);
  min_image_to(ox0, oy0, oz0, h2x0, h2y0, h2z0, lx, ly, lz, xy, xz, yz);

  // Unconstrained post-drift positions; same image fix-up.
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

  // COMs of pre-drift and unconstrained post-drift.
  float cx0, cy0, cz0;
  weighted_com(ox0, oy0, oz0, h1x0, h1y0, h1z0, h2x0, h2y0, h2z0,
               m_o, m_h, total_mass, cx0, cy0, cz0);
  float cx, cy, cz;
  weighted_com(ox_u, oy_u, oz_u, h1x_u, h1y_u, h1z_u, h2x_u, h2y_u, h2z_u,
               m_o, m_h, total_mass, cx, cy, cz);

  // Pre-drift positions relative to their COM (body-frame reference).
  float a0x = ox0 - cx0, a0y = oy0 - cy0, a0z = oz0 - cz0;
  float b0x = h1x0 - cx0, b0y = h1y0 - cy0, b0z = h1z0 - cz0;
  float c0x = h2x0 - cx0, c0y = h2y0 - cy0, c0z = h2z0 - cz0;

  // Build the orthonormal frame attached to the pre-drift molecule:
  // Z0 normal to the plane defined by (a0, b0, c0); X0 along the
  // canonical symmetry axis projected into the plane; Y0 = Z0 x X0.
  float e1x = b0x - a0x, e1y = b0y - a0y, e1z = b0z - a0z;
  float e2x = c0x - a0x, e2y = c0y - a0y, e2z = c0z - a0z;
  float z0x = e1y * e2z - e1z * e2y;
  float z0y = e1z * e2x - e1x * e2z;
  float z0z = e1x * e2y - e1y * e2x;
  float z0_norm = sqrtf(z0x * z0x + z0y * z0y + z0z * z0z);
  if (z0_norm < 1.0e-30f) {
    // Degenerate (collinear) pre-drift configuration; bail out by
    // copying the unconstrained positions and treating the constraint
    // correction as a no-op. The integrator's next step will
    // re-establish a non-degenerate geometry.
    return;
  }
  z0x /= z0_norm;
  z0y /= z0_norm;
  z0z /= z0_norm;
  // X0: from H-H midpoint toward O, projected into the molecular plane.
  float mx = 0.5f * (b0x + c0x);
  float my = 0.5f * (b0y + c0y);
  float mz = 0.5f * (b0z + c0z);
  float x0x = a0x - mx;
  float x0y = a0y - my;
  float x0z = a0z - mz;
  float x0_norm = sqrtf(x0x * x0x + x0y * x0y + x0z * x0z);
  if (x0_norm < 1.0e-30f) {
    return;
  }
  x0x /= x0_norm;
  x0y /= x0_norm;
  x0z /= x0_norm;
  float y0x = z0y * x0z - z0z * x0y;
  float y0y = z0z * x0x - z0x * x0z;
  float y0z = z0x * x0y - z0y * x0x;

  // Unconstrained positions relative to new COM, projected into the
  // body frame (X0, Y0, Z0).
  float a1x = ox_u - cx, a1y = oy_u - cy, a1z = oz_u - cz;
  float b1x = h1x_u - cx, b1y = h1y_u - cy, b1z = h1z_u - cz;
  float c1x = h2x_u - cx, c1y = h2y_u - cy, c1z = h2z_u - cz;

  // Express each in body-frame coordinates.
  float a1_X = a1x * x0x + a1y * x0y + a1z * x0z;
  float a1_Y = a1x * y0x + a1y * y0y + a1z * y0z;
  float a1_Z = a1x * z0x + a1y * z0y + a1z * z0z;
  float b1_X = b1x * x0x + b1y * x0y + b1z * x0z;
  float b1_Y = b1x * y0x + b1y * y0y + b1z * y0z;
  float b1_Z = b1x * z0x + b1y * z0y + b1z * z0z;
  float c1_X = c1x * x0x + c1y * x0y + c1z * x0z;
  float c1_Y = c1x * y0x + c1y * y0y + c1z * y0z;
  float c1_Z = c1x * z0x + c1y * z0y + c1z * z0z;

  // Canonical body-frame positions for this constraint type.
  // The body-frame convention matches the host computation:
  // O on +x, H1/H2 symmetric about the x-axis with H-H along y.
  float aO_X = ox_b;  // O is on the X axis in canonical form.
  float aO_Y = oy_b;
  float bH1_X = h1x_b;
  float bH1_Y = h1y_b;
  float cH2_X = h2x_b;
  float cH2_Y = h2y_b;

  // Step 1: in-plane rotation that aligns the H-H midpoint of the
  // unconstrained configuration with the +X axis. The constrained
  // positions in body-frame coordinates have:
  //   O   = ( aO_X cos φ, aO_X sin φ, 0 )      (canonical aO_Y = 0)
  //   H1  = ( bH1_X cos φ - bH1_Y sin φ,
  //           bH1_X sin φ + bH1_Y cos φ, 0 )
  //   H2  = ( cH2_X cos φ - cH2_Y sin φ,
  //           cH2_X sin φ + cH2_Y cos φ, 0 )
  // and the SETTLE rotation is chosen so that the projection of these
  // points onto the molecular plane matches the projection of the
  // unconstrained points.
  //
  // For the rigid water with masses (m_O, m_H, m_H), the body-frame
  // canonical configuration has weighted centroid at the origin and
  // O on the +x axis. The rotation angle that best fits the
  // unconstrained projection is determined by demanding that the
  // body-frame O-vector's in-plane projection (a1_X, a1_Y) is
  // proportional to (cos φ, sin φ):
  float r0 = sqrtf(a1_X * a1_X + a1_Y * a1_Y);
  float cos_phi, sin_phi;
  if (r0 > 1.0e-30f) {
    cos_phi = a1_X / r0;
    sin_phi = a1_Y / r0;
  } else {
    cos_phi = 1.0f;
    sin_phi = 0.0f;
  }

  // Step 2: out-of-plane rotation. With the molecular plane fixed by
  // the pre-drift body frame, the unconstrained out-of-plane component
  // of O is a1_Z. After the in-plane rotation, the body-frame O lies
  // at (aO_X cos φ, aO_X sin φ, 0). To match the unconstrained
  // out-of-plane displacement we rotate about the in-plane axis
  // perpendicular to the body-frame O-vector by an angle θ such that
  // sin θ = a1_Z / |O_body|, cos θ = sqrt(1 - sin² θ). aO_X is the
  // distance from O to the COM in the canonical frame.
  float r_o_body = fabsf(aO_X);
  float sin_theta = 0.0f, cos_theta = 1.0f;
  if (r_o_body > 1.0e-30f) {
    float s = a1_Z / r_o_body;
    if (s > 1.0f) s = 1.0f;
    if (s < -1.0f) s = -1.0f;
    sin_theta = s;
    cos_theta = sqrtf(fmaxf(0.0f, 1.0f - s * s));
  }

  // Reconstruct the constrained body-frame coordinates. The combined
  // rotation R = R_oop(θ) · R_inplane(φ) maps body-frame canonical
  // positions to the projected frame. Applied to a generic body-frame
  // point (X, Y, 0):
  //   R · (X, Y, 0) = ( X cos φ - Y sin φ,
  //                      (X sin φ + Y cos φ) · cos θ ... )
  // For a rigid water all three body-frame points have Z = 0, so the
  // out-of-plane rotation introduces a Z component proportional to
  // sin θ in their in-plane displacement perpendicular to the rotation
  // axis (the body-frame x' axis after the in-plane rotation).
  //
  // To keep the implementation tractable and bit-reproducible, we
  // apply R as a composition of three rotations in the body frame:
  // (i) rotate (X, Y) by φ about Z, (ii) rotate the result by θ about
  // the new Y' axis, (iii) leave Z = 0 + lift.
  auto apply_rot = [&](float X, float Y, float &px, float &py, float &pz) {
    // In-plane rotation about Z by φ.
    float x1 = X * cos_phi - Y * sin_phi;
    float y1 = X * sin_phi + Y * cos_phi;
    // Out-of-plane rotation about Y' (the rotated +Y axis) by θ.
    float x2 = x1 * cos_theta;
    float z2 = x1 * sin_theta;
    px = x2;
    py = y1;
    pz = z2;
  };
  float oX, oY, oZ;
  apply_rot(aO_X, aO_Y, oX, oY, oZ);
  float h1X, h1Y, h1Z;
  apply_rot(bH1_X, bH1_Y, h1X, h1Y, h1Z);
  float h2X, h2Y, h2Z;
  apply_rot(cH2_X, cH2_Y, h2X, h2Y, h2Z);

  // Transform back to Cartesian via the pre-drift frame (X0, Y0, Z0).
  auto from_body = [&](float X, float Y, float Z,
                       float &cx_o, float &cy_o, float &cz_o) {
    cx_o = X * x0x + Y * y0x + Z * z0x + cx;
    cy_o = X * x0y + Y * y0y + Z * z0y + cy;
    cz_o = X * x0z + Y * y0z + Z * z0z + cz;
  };
  float ox_c, oy_c, oz_c;
  from_body(oX, oY, oZ, ox_c, oy_c, oz_c);
  float h1x_c, h1y_c, h1z_c;
  from_body(h1X, h1Y, h1Z, h1x_c, h1y_c, h1z_c);
  float h2x_c, h2y_c, h2z_c;
  from_body(h2X, h2Y, h2Z, h2x_c, h2y_c, h2z_c);

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
  float inv_dt2 = (dt != 0.0f) ? (1.0f / (dt * dt)) : 0.0f;
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
      m_o * (dox * rox + doy * roy + doz * roz) * inv_dt2;
  constraint_virial[base + 1] =
      m_h * (dh1x * rh1x + dh1y * rh1y + dh1z * rh1z) * inv_dt2;
  constraint_virial[base + 2] =
      m_h * (dh2x * rh2x + dh2y * rh2y + dh2z * rh2z) * inv_dt2;
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
