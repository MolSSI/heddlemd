// rq-9a80c43c — General SHAKE + RATTLE constraint algorithm.
//
// One thread per constraint group. Each group has up to MAX_GROUP_ATOMS
// atoms and up to MAX_GROUP_CONSTRAINTS pair-distance constraints.
// Per-thread state for atoms and constraints lives in registers; the
// caps keep the total state in line with contemporary GPUs' register
// budgets. See rqm/integration/shake.md.

#include "precision.cuh"

#include "pbc.cuh"

#define MAX_GROUP_ATOMS 8
#define MAX_GROUP_CONSTRAINTS 12

// SHAKE absolute tolerance on σ = |r_i - r_j|² - d_k² in m² (atomic
// units: ~3.6e-7 a_0² ≈ 1.0e-26 m² when expressed in metres, scaled
// by the SI->Bohr factor by the engine before reaching the kernel.
// The kernel sees the constant unchanged from its rqm-documented
// 1.0e-26 m² value because the engine stores distances in atomic
// units; the kernel's threshold is the atomic-unit equivalent
// (a_0² ≈ 2.8e-21 m²), which the engine's setup converts at slot
// construction time. See settle.rs comment for the same point.).
//
// The kernel reads `shake_tol2` as a launch-time scalar so the
// engine can drive the threshold without recompiling. Defaults
// described in rqm/integration/shake.md.

// Minimum-image displacement helper that always returns the image of
// `b` closest to `a`. Used to bring the atoms of a group into the
// same lattice image before the rigid-body solve.
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

// Snapshot the pre-drift positions of every atom of every group.
// One thread per group; each thread writes group_atom_count[g] entries
// into snapshot_* starting at group_atom_offset[g].
extern "C" __global__ void shake_snapshot(
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
  unsigned int cnt = group_atom_count[g];
  for (unsigned int a = 0; a < cnt; ++a) {
    unsigned int atom_idx = group_atoms[off + a];
    Real4 pq = posq[atom_idx];
    snapshot_x[off + a] = pq.x;
    snapshot_y[off + a] = pq.y;
    snapshot_z[off + a] = pq.z;
  }
}

// Mass-weighted COM of a group's atoms (current configuration). Used by
// the constraint-virial computation to express per-atom positions in
// COM-relative form (f32-stable arithmetic).
__device__ static inline void weighted_com(
    const Real *x,
    const Real *y,
    const Real *z,
    const Real *m,
    unsigned int n,
    Real &cx, Real &cy, Real &cz)
{
  Real total = R(0.0);
  Real sx = R(0.0), sy = R(0.0), sz = R(0.0);
  for (unsigned int a = 0; a < n; ++a) {
    sx += m[a] * x[a];
    sy += m[a] * y[a];
    sz += m[a] * z[a];
    total += m[a];
  }
  Real inv = (total > R(0.0)) ? R(1.0) / total : R(0.0);
  cx = sx * inv;
  cy = sy * inv;
  cz = sz * inv;
}

// SHAKE position projection. One thread per group. Iteratively solves
// the K pair-distance constraints using Gauss-Seidel sweeps with the
// constraint-gradient direction fixed at the pre-drift snapshot.
//
// Per-thread state: MAX_GROUP_ATOMS × (3 floats unconstrained + 3
// floats constrained + 3 floats snapshot + 1 Real mass) +
// MAX_GROUP_CONSTRAINTS × (2 bytes pair + 1 Real r²). With caps 8/12
// this is ~330 B per thread, comfortably in registers.
extern "C" __global__ void shake_positions(
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
    const unsigned int *group_constraint_offset,
    const unsigned int *group_constraint_count,
    const unsigned char *group_constraints_local_i,
    const unsigned char *group_constraints_local_j,
    const Real *group_constraints_r2,
    const Real *atom_mass,
    const Real *lattice,
    Real dt,
    Real *constraint_virial,
    unsigned int n_groups)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_groups) {
    return;
  }

  unsigned int aoff = group_atom_offset[g];
  unsigned int acnt = group_atom_count[g];
  unsigned int coff = group_constraint_offset[g];
  unsigned int ccnt = group_constraint_count[g];

  // Zero the per-atom constraint-virial slots for this group up front.
  for (unsigned int a = 0; a < acnt; ++a) {
    constraint_virial[aoff + a] = R(0.0);
  }

  // Load atoms and per-atom inverse masses.
  unsigned int atom_idx[MAX_GROUP_ATOMS];
  Real x_u[MAX_GROUP_ATOMS], y_u[MAX_GROUP_ATOMS], z_u[MAX_GROUP_ATOMS];
  Real x_c[MAX_GROUP_ATOMS], y_c[MAX_GROUP_ATOMS], z_c[MAX_GROUP_ATOMS];
  Real x0[MAX_GROUP_ATOMS], y0[MAX_GROUP_ATOMS], z0[MAX_GROUP_ATOMS];
  Real m_atom[MAX_GROUP_ATOMS];
  Real inv_m[MAX_GROUP_ATOMS];
  for (unsigned int a = 0; a < acnt; ++a) {
    unsigned int i = group_atoms[aoff + a];
    atom_idx[a] = i;
    Real4 pq = posq[i];
    x_u[a] = pq.x;
    y_u[a] = pq.y;
    z_u[a] = pq.z;
    x0[a] = snapshot_x[aoff + a];
    y0[a] = snapshot_y[aoff + a];
    z0[a] = snapshot_z[aoff + a];
    m_atom[a] = atom_mass[i];
    inv_m[a] = (m_atom[a] > R(0.0)) ? R(1.0) / m_atom[a] : R(0.0);
  }

  // Bring every atom of the group into the same lattice image as
  // atom 0 (both pre-drift and post-drift). The SHAKE iteration then
  // operates in a single coherent image; the lab-frame write-back at
  // the end re-applies the same image offset to the global positions
  // so trajectories don't accidentally hop a periodic boundary.
  for (unsigned int a = 1; a < acnt; ++a) {
    min_image_to(x0[0], y0[0], z0[0], x0[a], y0[a], z0[a],
                 lx, ly, lz, xy, xz, yz);
    min_image_to(x_u[0], y_u[0], z_u[0], x_u[a], y_u[a], z_u[a],
                 lx, ly, lz, xy, xz, yz);
  }

  // Initialise the constrained positions at the unconstrained
  // post-drift positions.
  for (unsigned int a = 0; a < acnt; ++a) {
    x_c[a] = x_u[a];
    y_c[a] = y_u[a];
    z_c[a] = z_u[a];
  }

  // Pre-drift constraint-gradient directions.
  unsigned char ci[MAX_GROUP_CONSTRAINTS], cj[MAX_GROUP_CONSTRAINTS];
  Real gx[MAX_GROUP_CONSTRAINTS], gy[MAX_GROUP_CONSTRAINTS], gz[MAX_GROUP_CONSTRAINTS];
  Real r2[MAX_GROUP_CONSTRAINTS];
  Real inv_pair[MAX_GROUP_CONSTRAINTS];
  for (unsigned int k = 0; k < ccnt; ++k) {
    unsigned char li = group_constraints_local_i[coff + k];
    unsigned char lj = group_constraints_local_j[coff + k];
    ci[k] = li;
    cj[k] = lj;
    gx[k] = x0[li] - x0[lj];
    gy[k] = y0[li] - y0[lj];
    gz[k] = z0[li] - z0[lj];
    r2[k] = group_constraints_r2[coff + k];
    inv_pair[k] = inv_m[li] + inv_m[lj];
  }

  // SHAKE absolute tolerance on σ. The constant value is the
  // rqm-documented 1.0e-26 m², expressed in atomic units (a_0^4).
  // 1.0e-26 m² × (1 a_0 / 5.29177e-11 m)² ≈ 3.57e-6 a_0². The
  // engine's distances are in a_0, so σ has units a_0² and the
  // threshold is 3.57e-6 a_0². We use the constant directly.
  const Real SHAKE_TOL2 = R(3.57e-6);
  const int SHAKE_MAX_ITER = 32;

  for (int iter = 0; iter < SHAKE_MAX_ITER; ++iter) {
    bool converged = true;
    for (unsigned int k = 0; k < ccnt; ++k) {
      unsigned char li = ci[k];
      unsigned char lj = cj[k];
      Real dx = x_c[li] - x_c[lj];
      Real dy = y_c[li] - y_c[lj];
      Real dz = z_c[li] - z_c[lj];
      Real dist2 = dx * dx + dy * dy + dz * dz;
      Real sigma = dist2 - r2[k];
      if (Real_fabs(sigma) > SHAKE_TOL2) {
        converged = false;
        Real ddot = dx * gx[k] + dy * gy[k] + dz * gz[k];
        Real denom = R(2.0) * ddot * inv_pair[k];
        if (denom == R(0.0)) {
          continue;
        }
        Real lambda = sigma / denom;
        Real ki = lambda * inv_m[li];
        Real kj = lambda * inv_m[lj];
        x_c[li] -= ki * gx[k];
        y_c[li] -= ki * gy[k];
        z_c[li] -= ki * gz[k];
        x_c[lj] += kj * gx[k];
        y_c[lj] += kj * gy[k];
        z_c[lj] += kj * gz[k];
      }
    }
    if (converged) {
      break;
    }
  }

  // Mass-weighted COM (preserved by SHAKE; computed from the
  // constrained configuration for the virial below).
  Real cxg, cyg, czg;
  weighted_com(x_c, y_c, z_c, m_atom, acnt, cxg, cyg, czg);

  // Update half-step velocities and write back constrained
  // positions. The image-shift fix-up at the start operated on
  // local copies; we re-apply the same delta to the original global
  // positions so the global image-flag bookkeeping (if any) stays
  // valid: x_global_new = x_global_old + (x_c_local - x_u_local).
  Real inv_dt = (dt != R(0.0)) ? R(1.0) / dt : R(0.0);
  Real inv_dt2 = (dt != R(0.0)) ? R(1.0) / (dt * dt) : R(0.0);
  for (unsigned int a = 0; a < acnt; ++a) {
    unsigned int i = atom_idx[a];
    Real dxg = x_c[a] - x_u[a];
    Real dyg = y_c[a] - y_u[a];
    Real dzg = z_c[a] - z_u[a];
    velocities_x[i] += dxg * inv_dt;
    velocities_y[i] += dyg * inv_dt;
    velocities_z[i] += dzg * inv_dt;
    Real4 pq = posq[i];
    if (a == 0) {
      // Atom 0 is the reference image; its lab-frame position
      // equals the local-frame x_c[0].
      pq.x = x_c[0];
      pq.y = y_c[0];
      pq.z = z_c[0];
    } else {
      // Atoms 1..n-1 use the delta-style update so the global
      // image bookkeeping stays valid.
      pq.x += dxg;
      pq.y += dyg;
      pq.z += dzg;
    }
    posq[i] = pq;

    // Constraint-virial position-level half: (m / dt²) · (Δr · r_COM).
    // Compute in COM-relative coordinates for f32 stability.
    Real rx = x_c[a] - cxg;
    Real ry = y_c[a] - cyg;
    Real rz = z_c[a] - czg;
    Real scale = m_atom[a] * inv_dt2;
    constraint_virial[aoff + a] = scale * (dxg * rx + dyg * ry + dzg * rz);
  }
}

// RATTLE velocity projection. One thread per group. Iteratively zeroes
// the time-derivative of each constraint distance via Gauss-Seidel
// sweeps. When dt > 0 the kernel additionally accumulates the
// velocity-level constraint-virial contribution into the buffer that
// shake_positions has already populated with the position-level half.
extern "C" __global__ void rattle_velocities(
    const Real4 *posq,
    Real *velocities_x,
    Real *velocities_y,
    Real *velocities_z,
    const unsigned int *group_atoms,
    const unsigned int *group_atom_offset,
    const unsigned int *group_atom_count,
    const unsigned int *group_constraint_offset,
    const unsigned int *group_constraint_count,
    const unsigned char *group_constraints_local_i,
    const unsigned char *group_constraints_local_j,
    const Real *atom_mass,
    const Real *lattice,
    Real dt,
    Real *constraint_virial,
    unsigned int n_groups)
{
  // Group-contiguous shared-memory staging. One thread per constraint
  // group (water molecule). Each block owns a contiguous slice of
  // groups, hence — because group_atom_offset is cumulative — a
  // contiguous slice of `group_atoms`. The block cooperatively loads
  // every atom it needs into shared memory with coalesced global reads
  // (contiguous atom ids for molecule-ordered topologies), the per-group
  // RATTLE solve runs on the shared-resident per-atom state (replacing
  // the previous local-memory arrays and the stride-3 global gather),
  // and the updated velocities are stored back coalesced. The
  // floating-point operation sequence per group is unchanged, so results
  // are bit-identical to a per-thread gather. See
  // `rqm/integration/shake.md` *RATTLE Shared-Memory Staging*.
  // rq-115e5926
  extern __shared__ Real smem[];

  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];

  unsigned int G0 = blockIdx.x * blockDim.x;        // first group of the block
  unsigned int g = G0 + threadIdx.x;
  bool active = (g < n_groups);

  // The block's contiguous atom range [atom_base, atom_base+n_block_atoms).
  unsigned int last_g = (G0 + blockDim.x <= n_groups) ? (G0 + blockDim.x - 1)
                                                      : (n_groups - 1);
  unsigned int atom_base = group_atom_offset[G0];
  unsigned int n_block_atoms =
      group_atom_offset[last_g] + group_atom_count[last_g] - atom_base;

  // Partition dynamic shared into 11 per-atom arrays (host sizes the
  // allocation to blockDim * max_group_atoms * 11 Reals; n_block_atoms
  // never exceeds that bound).
  Real *s_px  = smem;
  Real *s_py  = s_px  + n_block_atoms;
  Real *s_pz  = s_py  + n_block_atoms;
  Real *s_vx  = s_pz  + n_block_atoms;
  Real *s_vy  = s_vx  + n_block_atoms;
  Real *s_vz  = s_vy  + n_block_atoms;
  Real *s_dvx = s_vz  + n_block_atoms;
  Real *s_dvy = s_dvx + n_block_atoms;
  Real *s_dvz = s_dvy + n_block_atoms;
  Real *s_m   = s_dvz + n_block_atoms;
  Real *s_im  = s_m   + n_block_atoms;

  // Coalesced staging load.
  for (unsigned int t = threadIdx.x; t < n_block_atoms; t += blockDim.x) {
    unsigned int i = group_atoms[atom_base + t];
    Real4 pq = posq[i];
    s_px[t] = pq.x; s_py[t] = pq.y; s_pz[t] = pq.z;
    s_vx[t] = velocities_x[i];
    s_vy[t] = velocities_y[i];
    s_vz[t] = velocities_z[i];
    Real m = atom_mass[i];
    s_m[t]  = m;
    s_im[t] = (m > R(0.0)) ? R(1.0) / m : R(0.0);
    s_dvx[t] = R(0.0); s_dvy[t] = R(0.0); s_dvz[t] = R(0.0);
  }
  __syncthreads();

  if (active) {
    unsigned int aoff = group_atom_offset[g];
    unsigned int acnt = group_atom_count[g];
    unsigned int coff = group_constraint_offset[g];
    unsigned int ccnt = group_constraint_count[g];
    unsigned int lb = aoff - atom_base;   // block-local base for this group

    // Per-group shared views (each thread owns a disjoint atom range, so
    // no synchronisation is needed within the solve).
    Real *px = &s_px[lb], *py = &s_py[lb], *pz = &s_pz[lb];
    Real *vx = &s_vx[lb], *vy = &s_vy[lb], *vz = &s_vz[lb];
    Real *dvx = &s_dvx[lb], *dvy = &s_dvy[lb], *dvz = &s_dvz[lb];
    Real *m_atom = &s_m[lb], *inv_m = &s_im[lb];

    // Same-image alignment as shake_positions.
    for (unsigned int a = 1; a < acnt; ++a) {
      min_image_to(px[0], py[0], pz[0], px[a], py[a], pz[a],
                   lx, ly, lz, xy, xz, yz);
    }

    // Constraint-gradient directions at the (now constrained) positions.
    unsigned char ci[MAX_GROUP_CONSTRAINTS], cj[MAX_GROUP_CONSTRAINTS];
    Real dx[MAX_GROUP_CONSTRAINTS], dy[MAX_GROUP_CONSTRAINTS], dz[MAX_GROUP_CONSTRAINTS];
    Real d2[MAX_GROUP_CONSTRAINTS];
    Real inv_pair[MAX_GROUP_CONSTRAINTS];
    for (unsigned int k = 0; k < ccnt; ++k) {
      unsigned char li = group_constraints_local_i[coff + k];
      unsigned char lj = group_constraints_local_j[coff + k];
      ci[k] = li;
      cj[k] = lj;
      dx[k] = px[li] - px[lj];
      dy[k] = py[li] - py[lj];
      dz[k] = pz[li] - pz[lj];
      d2[k] = dx[k] * dx[k] + dy[k] * dy[k] + dz[k] * dz[k];
      inv_pair[k] = inv_m[li] + inv_m[lj];
    }

    // RATTLE_TOL on |v_rel · d_k|. rqm-documented value is 1.0e-20 m²/s.
    // Converted to atomic units (a_0² / atu): 1.0e-20 × (1/5.29177e-11)²
    // × 2.4189e-17 ≈ 8.63e-17 a_0²/atu. The constant below is the
    // engine's atomic-unit equivalent.
    const Real RATTLE_TOL = R(8.63e-17);
    const int RATTLE_MAX_ITER = 32;

    for (int iter = 0; iter < RATTLE_MAX_ITER; ++iter) {
      bool converged = true;
      for (unsigned int k = 0; k < ccnt; ++k) {
        unsigned char li = ci[k];
        unsigned char lj = cj[k];
        Real vxr = vx[li] - vx[lj];
        Real vyr = vy[li] - vy[lj];
        Real vzr = vz[li] - vz[lj];
        Real vrel = vxr * dx[k] + vyr * dy[k] + vzr * dz[k];
        if (Real_fabs(vrel) > RATTLE_TOL) {
          converged = false;
          Real denom = d2[k] * inv_pair[k];
          if (denom == R(0.0)) {
            continue;
          }
          Real mu = vrel / denom;
          Real ki = mu * inv_m[li];
          Real kj = mu * inv_m[lj];
          vx[li] -= ki * dx[k];
          vy[li] -= ki * dy[k];
          vz[li] -= ki * dz[k];
          vx[lj] += kj * dx[k];
          vy[lj] += kj * dy[k];
          vz[lj] += kj * dz[k];
          dvx[li] -= ki * dx[k];
          dvy[li] -= ki * dy[k];
          dvz[li] -= ki * dz[k];
          dvx[lj] += kj * dx[k];
          dvy[lj] += kj * dy[k];
          dvz[lj] += kj * dz[k];
        }
      }
      if (converged) {
        break;
      }
    }

    // Velocity-level constraint-virial contribution.
    if (dt > R(0.0)) {
      Real inv_dt = R(1.0) / dt;
      Real cx, cy, cz;
      weighted_com(px, py, pz, m_atom, acnt, cx, cy, cz);
      for (unsigned int a = 0; a < acnt; ++a) {
        Real rx = px[a] - cx;
        Real ry = py[a] - cy;
        Real rz = pz[a] - cz;
        Real scale = m_atom[a] * inv_dt;
        Real w = scale * (dvx[a] * rx + dvy[a] * ry + dvz[a] * rz);
        constraint_virial[aoff + a] += w;
      }
    }
  }
  __syncthreads();

  // Coalesced write-back of the updated velocities.
  for (unsigned int t = threadIdx.x; t < n_block_atoms; t += blockDim.x) {
    unsigned int i = group_atoms[atom_base + t];
    velocities_x[i] = s_vx[t];
    velocities_y[i] = s_vy[t];
    velocities_z[i] = s_vz[t];
  }
}

// Scatter the per-atom-of-group constraint-virial values into the
// global per-particle virial array. One thread per atom slot
// (n_atom_slots = total atom-of-group entries across all groups).
// Groups are disjoint by construction (v1 topology rule), so no
// atomics are needed.
extern "C" __global__ void constraint_virial_scatter(
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

// Position-only SHAKE projection (no velocity correction, no virial
// accumulation). Used by the minimizer's apply_position_projection_only
// hook. The constraint-gradient direction is evaluated at the current
// off-manifold positions rather than at a snapshot.
extern "C" __global__ void shake_positions_no_velocity(
    Real4 *posq,
    const unsigned int *group_atoms,
    const unsigned int *group_atom_offset,
    const unsigned int *group_atom_count,
    const unsigned int *group_constraint_offset,
    const unsigned int *group_constraint_count,
    const unsigned char *group_constraints_local_i,
    const unsigned char *group_constraints_local_j,
    const Real *group_constraints_r2,
    const Real *atom_mass,
    const Real *lattice,
    unsigned int n_groups)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_groups) {
    return;
  }
  unsigned int aoff = group_atom_offset[g];
  unsigned int acnt = group_atom_count[g];
  unsigned int coff = group_constraint_offset[g];
  unsigned int ccnt = group_constraint_count[g];

  unsigned int atom_idx[MAX_GROUP_ATOMS];
  Real x_u[MAX_GROUP_ATOMS], y_u[MAX_GROUP_ATOMS], z_u[MAX_GROUP_ATOMS];
  Real x_c[MAX_GROUP_ATOMS], y_c[MAX_GROUP_ATOMS], z_c[MAX_GROUP_ATOMS];
  Real inv_m[MAX_GROUP_ATOMS];
  for (unsigned int a = 0; a < acnt; ++a) {
    unsigned int i = group_atoms[aoff + a];
    atom_idx[a] = i;
    Real4 pq = posq[i];
    x_u[a] = pq.x;
    y_u[a] = pq.y;
    z_u[a] = pq.z;
    Real m = atom_mass[i];
    inv_m[a] = (m > R(0.0)) ? R(1.0) / m : R(0.0);
  }
  for (unsigned int a = 1; a < acnt; ++a) {
    min_image_to(x_u[0], y_u[0], z_u[0], x_u[a], y_u[a], z_u[a],
                 lx, ly, lz, xy, xz, yz);
  }
  for (unsigned int a = 0; a < acnt; ++a) {
    x_c[a] = x_u[a];
    y_c[a] = y_u[a];
    z_c[a] = z_u[a];
  }

  unsigned char ci[MAX_GROUP_CONSTRAINTS], cj[MAX_GROUP_CONSTRAINTS];
  Real gx[MAX_GROUP_CONSTRAINTS], gy[MAX_GROUP_CONSTRAINTS], gz[MAX_GROUP_CONSTRAINTS];
  Real r2[MAX_GROUP_CONSTRAINTS];
  Real inv_pair[MAX_GROUP_CONSTRAINTS];
  for (unsigned int k = 0; k < ccnt; ++k) {
    unsigned char li = group_constraints_local_i[coff + k];
    unsigned char lj = group_constraints_local_j[coff + k];
    ci[k] = li;
    cj[k] = lj;
    gx[k] = x_u[li] - x_u[lj];
    gy[k] = y_u[li] - y_u[lj];
    gz[k] = z_u[li] - z_u[lj];
    r2[k] = group_constraints_r2[coff + k];
    inv_pair[k] = inv_m[li] + inv_m[lj];
  }

  const Real SHAKE_TOL2 = R(3.57e-6);
  const int SHAKE_MAX_ITER = 32;
  for (int iter = 0; iter < SHAKE_MAX_ITER; ++iter) {
    bool converged = true;
    for (unsigned int k = 0; k < ccnt; ++k) {
      unsigned char li = ci[k];
      unsigned char lj = cj[k];
      Real dx = x_c[li] - x_c[lj];
      Real dy = y_c[li] - y_c[lj];
      Real dz = z_c[li] - z_c[lj];
      Real dist2 = dx * dx + dy * dy + dz * dz;
      Real sigma = dist2 - r2[k];
      if (Real_fabs(sigma) > SHAKE_TOL2) {
        converged = false;
        Real ddot = dx * gx[k] + dy * gy[k] + dz * gz[k];
        Real denom = R(2.0) * ddot * inv_pair[k];
        if (denom == R(0.0)) {
          continue;
        }
        Real lambda = sigma / denom;
        Real ki = lambda * inv_m[li];
        Real kj = lambda * inv_m[lj];
        x_c[li] -= ki * gx[k];
        y_c[li] -= ki * gy[k];
        z_c[li] -= ki * gz[k];
        x_c[lj] += kj * gx[k];
        y_c[lj] += kj * gy[k];
        z_c[lj] += kj * gz[k];
      }
    }
    if (converged) {
      break;
    }
  }

  for (unsigned int a = 0; a < acnt; ++a) {
    unsigned int i = atom_idx[a];
    Real dxg = x_c[a] - x_u[a];
    Real dyg = y_c[a] - y_u[a];
    Real dzg = z_c[a] - z_u[a];
    Real4 pq = posq[i];
    if (a == 0) {
      pq.x = x_c[0];
      pq.y = y_c[0];
      pq.z = z_c[0];
    } else {
      pq.x += dxg;
      pq.y += dyg;
      pq.z += dzg;
    }
    posq[i] = pq;
  }
}
