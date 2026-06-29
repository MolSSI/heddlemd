// rq-709c8eb5 — SETTLE constraint algorithm for symmetric three-atom
// rigid water. One thread per water group; the per-group working set
// (three atoms, the canonical geometry, the two masses) lives in
// registers, so no shared-memory staging is needed. See
// rqm/integration/settle.md.
//
// Atom local order is canonical: 0 = oxygen (apex), 1 = H1, 2 = H2.
//
// The position reset (`settle_positions`) is the minimal-displacement
// projection of the unconstrained positions back onto the rigid
// manifold, with the constraint-gradient directions taken from the
// pre-drift snapshot — the same energy-conserving projection SHAKE
// performs, specialised to the three water constraints and solved in
// registers. The velocity reset (`settle_velocities`) is the
// analytical SETTLE step: the relative velocity along every bond is
// driven to zero by directly solving the 3x3 linear system for the
// bond-impulse multipliers, with no iteration.

#include "precision.cuh"

#include "pbc.cuh"

// Minimum-image displacement helper that returns the image of `b`
// closest to `a`. Brings the hydrogens into the oxygen's lattice image
// before the per-group solve.
__device__ static inline void min_image_to(
    Real ax, Real ay, Real az,
    Real &bx, Real &by, Real &bz,
    Real lx, Real ly, Real lz,
    Real xy, Real xz, Real yz)
{
  Real dx = bx - ax;
  Real dy = by - ay;
  Real dz = bz - az;
  triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);
  bx = ax + dx;
  by = ay + dy;
  bz = az + dz;
}

// Snapshot the pre-drift positions of every atom of every group into the
// per-group snapshot arrays (indexed by atom slot = group_atom_offset +
// local index). `settle_positions` uses them as the constraint-gradient
// reference frame, so the projection's constraint forces act along the
// pre-drift bond directions (energy-conserving, as in SHAKE).
// rq-709c8eb5
extern "C" __global__ void settle_snapshot(
    const Real4 *posq,
    const unsigned int *group_atoms,
    const unsigned int *group_atom_offset,
    const unsigned int *group_atom_count,
    Real *snapshot_x,
    Real *snapshot_y,
    Real *snapshot_z,
    unsigned int n_groups)
{
  unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_groups) {
    return;
  }
  unsigned int off = group_atom_offset[g];
  for (unsigned int a = 0; a < 3; ++a) {
    Real4 pq = posq[group_atoms[off + a]];
    snapshot_x[off + a] = pq.x;
    snapshot_y[off + a] = pq.y;
    snapshot_z[off + a] = pq.z;
  }
}

// sqrt with the argument clamped to its valid domain, so f32 round-off
// near a planar/extreme instantaneous geometry cannot produce a NaN.
__device__ static inline Real settle_csqrt(Real x)
{
  return Real_sqrt(x > R(0.0) ? x : R(0.0));
}

// Canonical squared target distances from the per-group geometry:
//   d_OH² = rc² + (ra+rb)²   (O–H1 and O–H2)
//   d_HH² = (2·rc)²          (H1–H2)
__device__ static inline void settle_targets(
    Real ra, Real rb, Real rc, Real &d_oh2, Real &d_hh2)
{
  d_oh2 = rc * rc + (ra + rb) * (ra + rb);
  d_hh2 = (R(2.0) * rc) * (R(2.0) * rc);
}

// Minimal-displacement SHAKE projection of the three water constraints.
// `g*` are the constraint-gradient directions (from the snapshot for the
// MD reset, or from the current positions for the minimizer); `xc/yc/zc`
// start at the unconstrained positions and are projected onto the
// manifold in place. Deterministic Gauss-Seidel sweep, fixed constraint
// order (O–H1, O–H2, H1–H2).
__device__ static inline void settle_project_positions(
    Real *xc, Real *yc, Real *zc,
    const Real *gx, const Real *gy, const Real *gz,
    const Real *inv_m, Real d_oh2, Real d_hh2)
{
  const unsigned char ci[3] = {0, 0, 1};
  const unsigned char cj[3] = {1, 2, 2};
  const Real r2[3] = {d_oh2, d_oh2, d_hh2};
  Real inv_pair[3];
  for (int k = 0; k < 3; ++k) {
    inv_pair[k] = inv_m[ci[k]] + inv_m[cj[k]];
  }
  const Real SETTLE_TOL2 = R(3.57e-6);  // a_0², matching SHAKE_TOL².
  const int SETTLE_MAX_ITER = 32;
  for (int iter = 0; iter < SETTLE_MAX_ITER; ++iter) {
    bool converged = true;
    for (int k = 0; k < 3; ++k) {
      unsigned char li = ci[k], lj = cj[k];
      Real dx = xc[li] - xc[lj];
      Real dy = yc[li] - yc[lj];
      Real dz = zc[li] - zc[lj];
      Real sigma = dx * dx + dy * dy + dz * dz - r2[k];
      if (Real_fabs(sigma) > SETTLE_TOL2) {
        converged = false;
        Real ddot = dx * gx[k] + dy * gy[k] + dz * gz[k];
        Real denom = R(2.0) * ddot * inv_pair[k];
        if (denom == R(0.0)) {
          continue;
        }
        Real lambda = sigma / denom;
        Real ki = lambda * inv_m[li];
        Real kj = lambda * inv_m[lj];
        xc[li] -= ki * gx[k]; yc[li] -= ki * gy[k]; zc[li] -= ki * gz[k];
        xc[lj] += kj * gx[k]; yc[lj] += kj * gy[k]; zc[lj] += kj * gz[k];
      }
    }
    if (converged) {
      break;
    }
  }
}

// SETTLE position reset. One thread per group. Resets the unconstrained
// post-drift positions to the exact rigid configuration by the analytical
// Miyamoto-Kollman rotation (the pre-drift snapshot supplies the reference
// orientation frame), updates the half-step velocities to be consistent
// with the position correction, and writes the position-level half of the
// constraint virial. The closed form is the Miyamoto & Kollman 1992
// algorithm as in OpenMM's ReferenceSETTLEAlgorithm; every sqrt argument is
// clamped (settle_csqrt) so f32 round-off cannot produce a NaN.
// rq-709c8eb5 rq-fa14a87f rq-4617c285
extern "C" __global__ void settle_positions(
    Real4 *posq,
    Real *velocities_x,
    Real *velocities_y,
    Real *velocities_z,
    const Real *snapshot_x,
    const Real *snapshot_y,
    const Real *snapshot_z,
    const unsigned int *group_atoms,
    const unsigned int *group_atom_offset,
    const unsigned int *group_atom_count,
    const Real *group_ra,
    const Real *group_rb,
    const Real *group_rc,
    const Real *group_m_o,
    const Real *group_m_h,
    const Real *lattice,
    Real dt,
    Real *constraint_virial,
    unsigned int n_groups)
{
  unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_groups) {
    return;
  }
  Real lx = lattice[0], ly = lattice[1], lz = lattice[2];
  Real xy = lattice[3], xz = lattice[4], yz = lattice[5];

  unsigned int off = group_atom_offset[g];
  unsigned int idx[3] = {group_atoms[off + 0], group_atoms[off + 1], group_atoms[off + 2]};

  Real4 p0 = posq[idx[0]];
  Real4 p1 = posq[idx[1]];
  Real4 p2 = posq[idx[2]];
  // Raw (global) unconstrained positions, kept for the image-preserving
  // delta write-back.
  Real cur[3][3] = {{p0.x, p0.y, p0.z}, {p1.x, p1.y, p1.z}, {p2.x, p2.y, p2.z}};
  Real snp[3][3] = {
      {snapshot_x[off + 0], snapshot_y[off + 0], snapshot_z[off + 0]},
      {snapshot_x[off + 1], snapshot_y[off + 1], snapshot_z[off + 1]},
      {snapshot_x[off + 2], snapshot_y[off + 2], snapshot_z[off + 2]}};

  // Reference (snapshot) bond vectors relative to the reference oxygen,
  // minimum-imaged so a molecule straddling a periodic boundary is coherent.
  Real xb0 = snp[1][0] - snp[0][0], yb0 = snp[1][1] - snp[0][1], zb0 = snp[1][2] - snp[0][2];
  Real xc0 = snp[2][0] - snp[0][0], yc0 = snp[2][1] - snp[0][1], zc0 = snp[2][2] - snp[0][2];
  triclinic_min_image(xb0, yb0, zb0, lx, ly, lz, xy, xz, yz);
  triclinic_min_image(xc0, yc0, zc0, lx, ly, lz, xy, xz, yz);

  // Per-atom drift (current − reference), minimum-imaged. One step of
  // drift never crosses a boundary, so this is the identity in practice.
  Real xp[3][3];
  for (unsigned int a = 0; a < 3; ++a) {
    Real dx = cur[a][0] - snp[a][0];
    Real dy = cur[a][1] - snp[a][1];
    Real dz = cur[a][2] - snp[a][2];
    triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);
    xp[a][0] = dx; xp[a][1] = dy; xp[a][2] = dz;
  }

  Real m0 = group_m_o[g];
  Real m1 = group_m_h[g];
  Real m2 = group_m_h[g];
  Real invTotalMass = R(1.0) / (m0 + m1 + m2);
  Real ra = group_ra[g], rb = group_rb[g], rc = group_rc[g];
  Real d2 = R(2.0) * rc;            // target H–H distance
  Real d2sq = d2 * d2;

  // --- Miyamoto-Kollman position reset (OpenMM ReferenceSETTLEAlgorithm) ---
  // All quantities are relative to the reference oxygen snp[0].
  Real xcom = (xp[0][0] * m0 + (xb0 + xp[1][0]) * m1 + (xc0 + xp[2][0]) * m2) * invTotalMass;
  Real ycom = (xp[0][1] * m0 + (yb0 + xp[1][1]) * m1 + (yc0 + xp[2][1]) * m2) * invTotalMass;
  Real zcom = (xp[0][2] * m0 + (zb0 + xp[1][2]) * m1 + (zc0 + xp[2][2]) * m2) * invTotalMass;

  Real xa1 = xp[0][0] - xcom, ya1 = xp[0][1] - ycom, za1 = xp[0][2] - zcom;
  Real xb1 = xb0 + xp[1][0] - xcom, yb1 = yb0 + xp[1][1] - ycom, zb1 = zb0 + xp[1][2] - zcom;
  Real xc1 = xc0 + xp[2][0] - xcom, yc1 = yc0 + xp[2][1] - ycom, zc1 = zc0 + xp[2][2] - zcom;

  // Orthonormal frame: Z' = reference plane normal (b0 × c0); X' from the
  // displaced oxygen; Y' completes the right-handed triad.
  Real xaksZd = yb0 * zc0 - zb0 * yc0;
  Real yaksZd = zb0 * xc0 - xb0 * zc0;
  Real zaksZd = xb0 * yc0 - yb0 * xc0;
  Real xaksXd = ya1 * zaksZd - za1 * yaksZd;
  Real yaksXd = za1 * xaksZd - xa1 * zaksZd;
  Real zaksXd = xa1 * yaksZd - ya1 * xaksZd;
  Real xaksYd = yaksZd * zaksXd - zaksZd * yaksXd;
  Real yaksYd = zaksZd * xaksXd - xaksZd * zaksXd;
  Real zaksYd = xaksZd * yaksXd - yaksZd * xaksXd;

  Real axlng = settle_csqrt(xaksXd * xaksXd + yaksXd * yaksXd + zaksXd * zaksXd);
  Real aylng = settle_csqrt(xaksYd * xaksYd + yaksYd * yaksYd + zaksYd * zaksYd);
  Real azlng = settle_csqrt(xaksZd * xaksZd + yaksZd * yaksZd + zaksZd * zaksZd);
  Real inv_ax = (axlng > R(0.0)) ? R(1.0) / axlng : R(0.0);
  Real inv_ay = (aylng > R(0.0)) ? R(1.0) / aylng : R(0.0);
  Real inv_az = (azlng > R(0.0)) ? R(1.0) / azlng : R(0.0);
  Real trns11 = xaksXd * inv_ax, trns21 = yaksXd * inv_ax, trns31 = zaksXd * inv_ax;
  Real trns12 = xaksYd * inv_ay, trns22 = yaksYd * inv_ay, trns32 = zaksYd * inv_ay;
  Real trns13 = xaksZd * inv_az, trns23 = yaksZd * inv_az, trns33 = zaksZd * inv_az;

  Real xb0d = trns11 * xb0 + trns21 * yb0 + trns31 * zb0;
  Real yb0d = trns12 * xb0 + trns22 * yb0 + trns32 * zb0;
  Real xc0d = trns11 * xc0 + trns21 * yc0 + trns31 * zc0;
  Real yc0d = trns12 * xc0 + trns22 * yc0 + trns32 * zc0;
  Real za1d = trns13 * xa1 + trns23 * ya1 + trns33 * za1;
  Real xb1d = trns11 * xb1 + trns21 * yb1 + trns31 * zb1;
  Real yb1d = trns12 * xb1 + trns22 * yb1 + trns32 * zb1;
  Real zb1d = trns13 * xb1 + trns23 * yb1 + trns33 * zb1;
  Real xc1d = trns11 * xc1 + trns21 * yc1 + trns31 * zc1;
  Real yc1d = trns12 * xc1 + trns22 * yc1 + trns32 * zc1;
  Real zc1d = trns13 * xc1 + trns23 * yc1 + trns33 * zc1;

  // Step 2: canonical triangle tilted by (phi, psi).
  Real sinphi = za1d / ra;
  Real cosphi = settle_csqrt(R(1.0) - sinphi * sinphi);
  Real sinpsi = (zb1d - zc1d) / (R(2.0) * rc * cosphi);
  Real cospsi = settle_csqrt(R(1.0) - sinpsi * sinpsi);

  Real ya2d = ra * cosphi;
  Real xb2d = -rc * cospsi;
  Real yb2d = -rb * cosphi - rc * sinpsi * sinphi;
  Real yc2d = -rb * cosphi + rc * sinpsi * sinphi;
  Real xb2d2 = xb2d * xb2d;
  Real hh2 = R(4.0) * xb2d2 + (yb2d - yc2d) * (yb2d - yc2d) + (zb1d - zc1d) * (zb1d - zc1d);
  Real deltx = R(2.0) * xb2d + settle_csqrt(R(4.0) * xb2d2 - hh2 + d2sq);
  xb2d -= deltx * R(0.5);

  // Step 3: in-plane rotation theta.
  Real alpha = xb2d * (xb0d - xc0d) + yb0d * yb2d + yc0d * yc2d;
  Real beta = xb2d * (yc0d - yb0d) + xb0d * yb2d + xc0d * yc2d;
  Real gamma = xb0d * yb1d - xb1d * yb0d + xc0d * yc1d - xc1d * yc0d;
  Real al2be2 = alpha * alpha + beta * beta;
  Real inv_al2be2 = (al2be2 > R(0.0)) ? R(1.0) / al2be2 : R(0.0);
  Real sintheta = (alpha * gamma - beta * settle_csqrt(al2be2 - gamma * gamma)) * inv_al2be2;
  Real costheta = settle_csqrt(R(1.0) - sintheta * sintheta);

  // Step 4: final constrained positions in the primed frame.
  Real xa3d = -ya2d * sintheta;
  Real ya3d = ya2d * costheta;
  Real za3d = za1d;
  Real xb3d = xb2d * costheta - yb2d * sintheta;
  Real yb3d = xb2d * sintheta + yb2d * costheta;
  Real zb3d = zb1d;
  Real xc3d = -xb2d * costheta - yc2d * sintheta;
  Real yc3d = -xb2d * sintheta + yc2d * costheta;
  Real zc3d = zc1d;

  // Step 5: back-transform to the lab frame (positions relative to COM).
  Real rcom[3][3] = {
      {trns11 * xa3d + trns12 * ya3d + trns13 * za3d,
       trns21 * xa3d + trns22 * ya3d + trns23 * za3d,
       trns31 * xa3d + trns32 * ya3d + trns33 * za3d},
      {trns11 * xb3d + trns12 * yb3d + trns13 * zb3d,
       trns21 * xb3d + trns22 * yb3d + trns23 * zb3d,
       trns31 * xb3d + trns32 * yb3d + trns33 * zb3d},
      {trns11 * xc3d + trns12 * yc3d + trns13 * zc3d,
       trns21 * xc3d + trns22 * yc3d + trns23 * zc3d,
       trns31 * xc3d + trns32 * yc3d + trns33 * zc3d}};

  // Constrained positions relative to the reference oxygen.
  Real com[3] = {xcom, ycom, zcom};
  // Unconstrained positions relative to the reference oxygen (image-coherent):
  // O = xp0, H1 = b0 + xp1, H2 = c0 + xp2.
  Real unc[3][3] = {
      {xp[0][0], xp[0][1], xp[0][2]},
      {xb0 + xp[1][0], yb0 + xp[1][1], zb0 + xp[1][2]},
      {xc0 + xp[2][0], yc0 + xp[2][1], zc0 + xp[2][2]}};

  Real m_atom[3] = {m0, m1, m2};
  Real inv_dt = (dt != R(0.0)) ? R(1.0) / dt : R(0.0);
  Real inv_dt2 = (dt != R(0.0)) ? R(1.0) / (dt * dt) : R(0.0);

  for (unsigned int a = 0; a < 3; ++a) {
    // Image-invariant correction = constrained − unconstrained.
    Real corr0 = (com[0] + rcom[a][0]) - unc[a][0];
    Real corr1 = (com[1] + rcom[a][1]) - unc[a][1];
    Real corr2 = (com[2] + rcom[a][2]) - unc[a][2];
    velocities_x[idx[a]] += corr0 * inv_dt;
    velocities_y[idx[a]] += corr1 * inv_dt;
    velocities_z[idx[a]] += corr2 * inv_dt;
    Real4 pq = posq[idx[a]];
    pq.x = cur[a][0] + corr0;
    pq.y = cur[a][1] + corr1;
    pq.z = cur[a][2] + corr2;
    posq[idx[a]] = pq;
    // Position-level constraint virial. r_i^COM = constrained_i − COM = rcom[a].
    Real scale = m_atom[a] * inv_dt2;
    constraint_virial[off + a] =
        scale * (corr0 * rcom[a][0] + corr1 * rcom[a][1] + corr2 * rcom[a][2]);
  }
}

// Solve the 3x3 linear system M·g = rhs by Cramer's rule. Returns false
// (leaving g untouched) when the system is singular.
__device__ static inline bool solve3(
    const Real m[3][3], const Real rhs[3], Real g[3])
{
  Real det =
      m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
      m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
      m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  if (det == R(0.0)) {
    return false;
  }
  Real inv_det = R(1.0) / det;
  for (int c = 0; c < 3; ++c) {
    Real a[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        a[i][j] = (j == c) ? rhs[i] : m[i][j];
      }
    }
    Real dc =
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
        a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
        a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    g[c] = dc * inv_det;
  }
  return true;
}

// SETTLE velocity reset. One thread per group. Projects the post-kick
// velocities onto the velocity manifold of the (already constrained)
// positions by directly solving the 3x3 system for the bond-impulse
// multipliers — no iteration. When dt > 0 it accumulates the velocity-
// level half of the constraint virial.
// rq-709c8eb5 rq-4617c285
extern "C" __global__ void settle_velocities(
    const Real4 *posq,
    Real *velocities_x,
    Real *velocities_y,
    Real *velocities_z,
    const unsigned int *group_atoms,
    const unsigned int *group_atom_offset,
    const unsigned int *group_atom_count,
    const Real *group_m_o,
    const Real *group_m_h,
    const Real *lattice,
    Real dt,
    Real *constraint_virial,
    unsigned int n_groups)
{
  unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_groups) {
    return;
  }
  Real lx = lattice[0], ly = lattice[1], lz = lattice[2];
  Real xy = lattice[3], xz = lattice[4], yz = lattice[5];

  unsigned int off = group_atom_offset[g];
  unsigned int idx0 = group_atoms[off + 0];
  unsigned int idx1 = group_atoms[off + 1];
  unsigned int idx2 = group_atoms[off + 2];

  Real4 p0 = posq[idx0];
  Real4 p1 = posq[idx1];
  Real4 p2 = posq[idx2];
  Real px[3] = {p0.x, p1.x, p2.x};
  Real py[3] = {p0.y, p1.y, p2.y};
  Real pz[3] = {p0.z, p1.z, p2.z};
  for (unsigned int a = 1; a < 3; ++a) {
    min_image_to(px[0], py[0], pz[0], px[a], py[a], pz[a], lx, ly, lz, xy, xz, yz);
  }

  Real m_o = group_m_o[g];
  Real m_h = group_m_h[g];
  Real inv_o = (m_o > R(0.0)) ? R(1.0) / m_o : R(0.0);
  Real inv_h = (m_h > R(0.0)) ? R(1.0) / m_h : R(0.0);

  Real vx[3] = {velocities_x[idx0], velocities_x[idx1], velocities_x[idx2]};
  Real vy[3] = {velocities_y[idx0], velocities_y[idx1], velocities_y[idx2]};
  Real vz[3] = {velocities_z[idx0], velocities_z[idx1], velocities_z[idx2]};

  // Current bond vectors. r1 = O-H1, r2 = O-H2, r3 = H1-H2.
  Real r1x = px[0] - px[1], r1y = py[0] - py[1], r1z = pz[0] - pz[1];
  Real r2x = px[0] - px[2], r2y = py[0] - py[2], r2z = pz[0] - pz[2];
  Real r3x = px[1] - px[2], r3y = py[1] - py[2], r3z = pz[1] - pz[2];

  Real d11 = r1x * r1x + r1y * r1y + r1z * r1z;
  Real d22 = r2x * r2x + r2y * r2y + r2z * r2z;
  Real d33 = r3x * r3x + r3y * r3y + r3z * r3z;
  Real d12 = r1x * r2x + r1y * r2y + r1z * r2z;  // r1·r2
  Real d13 = r1x * r3x + r1y * r3y + r1z * r3z;  // r1·r3
  Real d23 = r2x * r3x + r2y * r3y + r2z * r3z;  // r2·r3

  // Constraint-velocity residuals b_k = (v_i - v_j) · r_k.
  Real b1 = (vx[0] - vx[1]) * r1x + (vy[0] - vy[1]) * r1y + (vz[0] - vz[1]) * r1z;
  Real b2 = (vx[0] - vx[2]) * r2x + (vy[0] - vy[2]) * r2y + (vz[0] - vz[2]) * r2z;
  Real b3 = (vx[1] - vx[2]) * r3x + (vy[1] - vy[2]) * r3y + (vz[1] - vz[2]) * r3z;

  Real inv_oh = inv_o + inv_h;
  // M·g = -b, with the corrections applying equal-and-opposite impulses
  // along the bond directions (see rqm/integration/settle.md).
  Real m[3][3];
  m[0][0] = d11 * inv_oh;  m[0][1] = d12 * inv_o;   m[0][2] = -d13 * inv_h;
  m[1][0] = d12 * inv_o;   m[1][1] = d22 * inv_oh;  m[1][2] = d23 * inv_h;
  m[2][0] = -d13 * inv_h;  m[2][1] = d23 * inv_h;   m[2][2] = R(2.0) * d33 * inv_h;
  Real rhs[3] = {-b1, -b2, -b3};

  Real gg[3] = {R(0.0), R(0.0), R(0.0)};
  solve3(m, rhs, gg);
  Real g1 = gg[0], g2 = gg[1], g3 = gg[2];

  // Velocity corrections.
  Real dvx0 = (g1 * r1x + g2 * r2x) * inv_o;
  Real dvy0 = (g1 * r1y + g2 * r2y) * inv_o;
  Real dvz0 = (g1 * r1z + g2 * r2z) * inv_o;
  Real dvx1 = (-g1 * r1x + g3 * r3x) * inv_h;
  Real dvy1 = (-g1 * r1y + g3 * r3y) * inv_h;
  Real dvz1 = (-g1 * r1z + g3 * r3z) * inv_h;
  Real dvx2 = (-g2 * r2x - g3 * r3x) * inv_h;
  Real dvy2 = (-g2 * r2y - g3 * r3y) * inv_h;
  Real dvz2 = (-g2 * r2z - g3 * r3z) * inv_h;

  velocities_x[idx0] = vx[0] + dvx0;
  velocities_y[idx0] = vy[0] + dvy0;
  velocities_z[idx0] = vz[0] + dvz0;
  velocities_x[idx1] = vx[1] + dvx1;
  velocities_y[idx1] = vy[1] + dvy1;
  velocities_z[idx1] = vz[1] + dvz1;
  velocities_x[idx2] = vx[2] + dvx2;
  velocities_y[idx2] = vy[2] + dvy2;
  velocities_z[idx2] = vz[2] + dvz2;

  if (dt > R(0.0)) {
    Real inv_dt = R(1.0) / dt;
    Real total = m_o + R(2.0) * m_h;
    Real invM = (total > R(0.0)) ? R(1.0) / total : R(0.0);
    Real cx = (m_o * px[0] + m_h * px[1] + m_h * px[2]) * invM;
    Real cy = (m_o * py[0] + m_h * py[1] + m_h * py[2]) * invM;
    Real cz = (m_o * pz[0] + m_h * pz[1] + m_h * pz[2]) * invM;
    Real dvx[3] = {dvx0, dvx1, dvx2};
    Real dvy[3] = {dvy0, dvy1, dvy2};
    Real dvz[3] = {dvz0, dvz1, dvz2};
    Real m_atom[3] = {m_o, m_h, m_h};
    for (unsigned int a = 0; a < 3; ++a) {
      Real rxc = px[a] - cx;
      Real ryc = py[a] - cy;
      Real rzc = pz[a] - cz;
      Real scale = m_atom[a] * inv_dt;
      constraint_virial[off + a] += scale * (dvx[a] * rxc + dvy[a] * ryc + dvz[a] * rzc);
    }
  }
}

// Scatter the per-atom-of-group constraint-virial values into the global
// per-particle virial array. One thread per atom slot. Groups are
// disjoint by construction, so no atomics are needed.
// rq-709c8eb5 rq-4617c285
extern "C" __global__ void settle_virial_scatter(
    const Real *constraint_virial,
    const unsigned int *group_atoms,
    Real *particle_virials,
    unsigned int n_atom_slots)
{
  unsigned int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n_atom_slots) {
    return;
  }
  unsigned int atom_index = group_atoms[s];
  particle_virials[atom_index] = particle_virials[atom_index] + constraint_virial[s];
}

// Position-only projection for the minimizer's
// apply_position_projection_only hook. Same minimal-displacement
// projection as settle_positions, but with the constraint-gradient
// directions evaluated at the current (off-manifold) positions —
// minimization has no pre-drift frame — and without touching velocities
// or the virial. Bit-exact identity for an already-rigid molecule (the
// σ gate skips every constraint), matching SHAKE. rq-709c8eb5
extern "C" __global__ void settle_positions_no_velocity(
    Real4 *posq,
    const unsigned int *group_atoms,
    const unsigned int *group_atom_offset,
    const unsigned int *group_atom_count,
    const Real *group_ra,
    const Real *group_rb,
    const Real *group_rc,
    const Real *group_m_o,
    const Real *group_m_h,
    const Real *lattice,
    unsigned int n_groups)
{
  unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_groups) {
    return;
  }
  Real lx = lattice[0], ly = lattice[1], lz = lattice[2];
  Real xy = lattice[3], xz = lattice[4], yz = lattice[5];

  unsigned int off = group_atom_offset[g];
  unsigned int idx[3] = {group_atoms[off + 0], group_atoms[off + 1], group_atoms[off + 2]};

  Real4 p0 = posq[idx[0]];
  Real4 p1 = posq[idx[1]];
  Real4 p2 = posq[idx[2]];
  Real xr[3] = {p0.x, p1.x, p2.x};
  Real yr[3] = {p0.y, p1.y, p2.y};
  Real zr[3] = {p0.z, p1.z, p2.z};
  Real xu[3] = {xr[0], xr[1], xr[2]};
  Real yu[3] = {yr[0], yr[1], yr[2]};
  Real zu[3] = {zr[0], zr[1], zr[2]};
  for (unsigned int a = 1; a < 3; ++a) {
    min_image_to(xu[0], yu[0], zu[0], xu[a], yu[a], zu[a], lx, ly, lz, xy, xz, yz);
  }

  Real m_o = group_m_o[g];
  Real m_h = group_m_h[g];
  Real inv_m[3] = {(m_o > R(0.0)) ? R(1.0) / m_o : R(0.0),
                   (m_h > R(0.0)) ? R(1.0) / m_h : R(0.0),
                   (m_h > R(0.0)) ? R(1.0) / m_h : R(0.0)};
  Real d_oh2, d_hh2;
  settle_targets(group_ra[g], group_rb[g], group_rc[g], d_oh2, d_hh2);

  const unsigned char ci[3] = {0, 0, 1};
  const unsigned char cj[3] = {1, 2, 2};
  Real gx[3], gy[3], gz[3];
  for (int k = 0; k < 3; ++k) {
    gx[k] = xu[ci[k]] - xu[cj[k]];
    gy[k] = yu[ci[k]] - yu[cj[k]];
    gz[k] = zu[ci[k]] - zu[cj[k]];
  }

  Real xc[3] = {xu[0], xu[1], xu[2]};
  Real yc[3] = {yu[0], yu[1], yu[2]};
  Real zc[3] = {zu[0], zu[1], zu[2]};
  settle_project_positions(xc, yc, zc, gx, gy, gz, inv_m, d_oh2, d_hh2);

  for (unsigned int a = 0; a < 3; ++a) {
    Real4 pq = posq[idx[a]];
    pq.x = xr[a] + (xc[a] - xu[a]);
    pq.y = yr[a] + (yc[a] - yu[a]);
    pq.z = zr[a] + (zc[a] - zu[a]);
    posq[idx[a]] = pq;
  }
}
