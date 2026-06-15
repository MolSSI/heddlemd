// rq-9ca00d25 rq-202493a5

#include "precision.cuh"

#include "pbc.cuh"

// Cardinal B-spline M_p(x) via the Cox-de Boor recursion. M_1 is the
// indicator of [0, 1); successive orders are computed from neighbouring
// M_{p-1} samples. The function works for any p in [2, 8].
__device__ static inline Real bspline_weight(int p, Real x)
{
  // M_1(x - i) for i = 0..p-1.
  Real vals[9];
  for (int i = 0; i < p; ++i) {
    Real xi = x - (Real) i;
    vals[i] = (xi >= R(0.0) && xi < R(1.0)) ? R(1.0) : R(0.0);
  }
  // Recurse up to order p.
  for (int order = 2; order <= p; ++order) {
    Real inv = R(1.0) / (Real) (order - 1);
    for (int i = 0; i < p - order + 1; ++i) {
      Real xi = x - (Real) i;
      vals[i] = xi * inv * vals[i]
                + ((Real) order - xi) * inv * vals[i + 1];
    }
  }
  return vals[0];
}

// Compute M_p(x) and M_p'(x) = M_{p-1}(x) - M_{p-1}(x-1) in one pass.
// The derivative identity follows from the Cox-de Boor recursion.
__device__ static inline void bspline_weight_and_deriv(
    int p, Real x, Real &w, Real &dw)
{
  Real vals[9];
  // M_1 indicators at x - i for i = 0..p (one extra so we can read
  // M_{p-1}(x) and M_{p-1}(x - 1) after the recursion).
  for (int i = 0; i < p + 1; ++i) {
    Real xi = x - (Real) i;
    vals[i] = (xi >= R(0.0) && xi < R(1.0)) ? R(1.0) : R(0.0);
  }
  // Recurse up to order p - 1.
  for (int order = 2; order < p; ++order) {
    Real inv = R(1.0) / (Real) (order - 1);
    for (int i = 0; i < p - order + 1; ++i) {
      Real xi = x - (Real) i;
      vals[i] = xi * inv * vals[i]
                + ((Real) order - xi) * inv * vals[i + 1];
    }
  }
  // vals[0] = M_{p-1}(x), vals[1] = M_{p-1}(x - 1).
  dw = vals[0] - vals[1];
  // Final step to compute M_p(x).
  {
    int order = p;
    Real inv = R(1.0) / (Real) (order - 1);
    Real xi = x;
    w = xi * inv * vals[0] + ((Real) order - xi) * inv * vals[1];
  }
}

// rq-9ca00d25
//
// Charge spreading kernel: one thread per real grid cell. Walks the p^3
// neighbouring bins (cells of the FFT-grid-aligned spatial hash) and
// accumulates B-spline-weighted charge contributions from every
// particle whose primary bin is offset by (d_a, d_b, d_c) ∈ [0, p)^3
// from the thread's grid point. Each grid cell is written by exactly
// one thread; no atomics.
extern "C" __global__ void spme_charge_spread(
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const Real *charges,
    const unsigned int *sorted_particle_ids,
    const unsigned int *cell_offsets,
    Real lx, Real ly, Real lz, Real xy, Real xz, Real yz,
    unsigned int n_a, unsigned int n_b, unsigned int n_c,
    unsigned int spline_order,
    Real *rho,
    unsigned int n)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int M = n_a * n_b * n_c;
  if (idx >= M) {
    return;
  }
  // Decompose idx into (g_a, g_b, g_c) under row-major ordering.
  unsigned int g_a = idx / (n_b * n_c);
  unsigned int rem = idx - g_a * (n_b * n_c);
  unsigned int g_b = rem / n_c;
  unsigned int g_c = rem - g_b * n_c;

  int p = (int) spline_order;
  Real accum = R(0.0);

  for (int da = 0; da < p; ++da) {
    int ba = ((int) g_a + da) % (int) n_a;
    for (int db = 0; db < p; ++db) {
      int bb = ((int) g_b + db) % (int) n_b;
      for (int dc = 0; dc < p; ++dc) {
        int bc = ((int) g_c + dc) % (int) n_c;
        unsigned int bin =
            ((unsigned int) ba * n_b + (unsigned int) bb) * n_c
            + (unsigned int) bc;
        unsigned int start = cell_offsets[bin];
        unsigned int end = cell_offsets[bin + 1];
        for (unsigned int s = start; s < end; ++s) {
          unsigned int i = sorted_particle_ids[s];
          Real px = positions_x[i];
          Real py = positions_y[i];
          Real pz = positions_z[i];
          // Re-wrap defensively (the integrator already wraps, but f32
          // round-off can leave fractional coords just outside [-0.5, 0.5)).
          int wrap_a, wrap_b, wrap_c;
          triclinic_wrap_with_image(px, py, pz, wrap_a, wrap_b, wrap_c,
                                    lx, ly, lz, xy, xz, yz);
          Real sa, sb, sc;
          triclinic_cart_to_frac(px, py, pz, lx, ly, lz, xy, xz, yz,
                                 sa, sb, sc);
          Real sa_p = sa + R(0.5);
          Real sb_p = sb + R(0.5);
          Real sc_p = sc + R(0.5);
          Real ua = sa_p * (Real) n_a;
          Real ub = sb_p * (Real) n_b;
          Real uc = sc_p * (Real) n_c;
          Real ta = ua - Real_floor(ua);
          Real tb = ub - Real_floor(ub);
          Real tc = uc - Real_floor(uc);
          Real wa = bspline_weight(p, (Real) da + ta);
          Real wb = bspline_weight(p, (Real) db + tb);
          Real wc = bspline_weight(p, (Real) dc + tc);
          accum += charges[i] * wa * wb * wc;
        }
      }
    }
  }
  rho[idx] = accum;
}

// rq-9ca00d25
//
// Influence-function multiply: V_hat[k] = G[k] · rho_hat[k] for every
// complex grid cell, including writing zero at k = (0, 0, 0). One
// thread per complex cell; no atomics. The complex grid is stored as
// interleaved (real, imag) Real pairs in row-major order, with the
// last (n_c/2 + 1) axis as the fastest-varying.
//
// Also writes per-cell virial contributions for the reciprocal-space
// scalar virial reduction:
//   virial_per_cell[k] = w_hermitian[k] · G[k] · |rho_hat[k]|²
//                                       · (1 − K²/(2α²))
// where w_hermitian is 2 for the modes that the R2C output omits via
// Hermitian symmetry (k_c not equal to 0 or n_c/2) and 1 otherwise.
// Summing virial_per_cell deterministically yields W_recip (per the
// definition in `rqm/forces/spme.md`).
//
// Operates in place on `rho_hat_interleaved`. The k=0 cell is
// identified by its flat index 0 (since (0, 0, 0) has row-major
// index 0).
extern "C" __global__ void spme_influence_multiply(
    const Real *influence_G,
    const Real *virial_factor,
    Real *rho_hat_interleaved,
    Real *virial_per_cell,
    unsigned int n_c,
    unsigned int n_c_complex,
    unsigned int n_complex)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_complex) {
    return;
  }
  Real g = influence_G[idx];
  Real vf = virial_factor[idx];
  unsigned int base = idx * 2u;
  Real re = rho_hat_interleaved[base];
  Real im = rho_hat_interleaved[base + 1u];
  Real rho_sq = re * re + im * im;
  unsigned int kc = idx % n_c_complex;
  // Hermitian weight: count modes paired across complex conjugation.
  // Modes at kc == 0 and (for even n_c) at kc == n_c/2 are self-paired
  // and contribute once; all other kc contribute twice.
  unsigned int hw =
      (kc == 0u || (n_c % 2u == 0u && 2u * kc == n_c)) ? 1u : 2u;
  virial_per_cell[idx] = (Real) hw * vf * rho_sq;

  rho_hat_interleaved[base]      = g * re;
  rho_hat_interleaved[base + 1u] = g * im;
}

// Compute the reciprocal-lattice rows (b_a, b_b, b_c) from the six
// lower-triangular lattice parameters. Returns them in column-major
// triples (b_*_x, b_*_y, b_*_z).
__device__ static inline void reciprocal_lattice_rows(
    Real lx, Real ly, Real lz, Real xy, Real xz, Real yz,
    Real &b_a_x, Real &b_a_y, Real &b_a_z,
    Real &b_b_x, Real &b_b_y, Real &b_b_z,
    Real &b_c_x, Real &b_c_y, Real &b_c_z)
{
  Real inv_lx = R(1.0) / lx;
  Real inv_ly = R(1.0) / ly;
  Real inv_lz = R(1.0) / lz;
  b_a_x = inv_lx;
  b_a_y = -xy * inv_lx * inv_ly;
  b_a_z = (xy * yz - xz * ly) * inv_lx * inv_ly * inv_lz;
  b_b_x = R(0.0);
  b_b_y = inv_ly;
  b_b_z = -yz * inv_ly * inv_lz;
  b_c_x = R(0.0);
  b_c_y = R(0.0);
  b_c_z = inv_lz;
}

// rq-9ca00d25
//
// Force gather kernel: one thread per particle. Each thread walks the
// p^3 grid points whose support the particle's B-spline overlaps,
// reads V[g], and accumulates per-particle force, energy, and the
// scratch quantities needed by the slot's reduce() step.
//
// Per-particle outputs:
//   slot_force_x[i] / slot_force_y[i] / slot_force_z[i] — Cartesian
//     reciprocal-space force contribution F_i_recip.
//   slot_energy[i] — per-particle reciprocal-space energy share:
//     0.5 · q_i · Σ_g V[g] · w_a · w_b · w_c. Summing over i yields
//     U_recip exactly (by the half-sum identity with `Σ_g rho V`).
//
// `u_self_per_particle[i] = k_C · (α/√π) · q_i²` is subtracted from the
// per-particle energy share so that summing `slot_energy[i]` yields
// `U_recip − U_self` exactly.
//
// The reciprocal-space scalar virial `W_recip` is reduced host-side
// from `virial_per_cell`; this kernel writes the uniform per-particle
// share `w_per_particle_virial = W_recip / N` into `slot_virial[i]`.
extern "C" __global__ void spme_force_gather(
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const Real *charges,
    const Real *V,
    const Real *u_self_per_particle,
    Real w_per_particle_virial,
    Real lx, Real ly, Real lz, Real xy, Real xz, Real yz,
    unsigned int n_a, unsigned int n_b, unsigned int n_c,
    unsigned int spline_order,
    Real *slot_force_x,
    Real *slot_force_y,
    Real *slot_force_z,
    Real *slot_energy,
    Real *slot_virial,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  Real qi = charges[i];
  Real px = positions_x[i];
  Real py = positions_y[i];
  Real pz = positions_z[i];

  // Re-wrap defensively, matching the spread kernel.
  int wrap_a, wrap_b, wrap_c;
  triclinic_wrap_with_image(px, py, pz, wrap_a, wrap_b, wrap_c,
                            lx, ly, lz, xy, xz, yz);
  Real sa, sb, sc;
  triclinic_cart_to_frac(px, py, pz, lx, ly, lz, xy, xz, yz, sa, sb, sc);
  Real sa_p = sa + R(0.5);
  Real sb_p = sb + R(0.5);
  Real sc_p = sc + R(0.5);
  Real ua = sa_p * (Real) n_a;
  Real ub = sb_p * (Real) n_b;
  Real uc = sc_p * (Real) n_c;
  int ga0 = (int) Real_floor(ua);
  int gb0 = (int) Real_floor(ub);
  int gc0 = (int) Real_floor(uc);
  Real ta = ua - (Real) ga0;
  Real tb = ub - (Real) gb0;
  Real tc = uc - (Real) gc0;

  int p = (int) spline_order;
  Real accum_phi    = R(0.0);
  Real accum_grad_a = R(0.0);  // dΦ/dt_a (in fractional-grid units)
  Real accum_grad_b = R(0.0);
  Real accum_grad_c = R(0.0);

  for (int da = 0; da < p; ++da) {
    Real wa, dwa;
    bspline_weight_and_deriv(p, (Real) da + ta, wa, dwa);
    int g_a = ga0 - da;
    g_a = ((g_a % (int) n_a) + (int) n_a) % (int) n_a;
    for (int db = 0; db < p; ++db) {
      Real wb, dwb;
      bspline_weight_and_deriv(p, (Real) db + tb, wb, dwb);
      int g_b = gb0 - db;
      g_b = ((g_b % (int) n_b) + (int) n_b) % (int) n_b;
      for (int dc = 0; dc < p; ++dc) {
        Real wc, dwc;
        bspline_weight_and_deriv(p, (Real) dc + tc, wc, dwc);
        int g_c = gc0 - dc;
        g_c = ((g_c % (int) n_c) + (int) n_c) % (int) n_c;
        unsigned int g_idx =
            ((unsigned int) g_a * n_b + (unsigned int) g_b) * n_c
            + (unsigned int) g_c;
        Real v = V[g_idx];
        accum_phi    += v * wa * wb * wc;
        // dM_p(da+t)/dt = M_p'(da+t). But we want d/d(s_a · n_a) = d/du_a.
        // Since u_a = s_a' · n_a and da+t = u_a - g_a, d(da+t)/du_a = 1.
        // So d(wa)/du_a = dwa. The chain rule into Cartesian comes
        // below via the reciprocal lattice.
        accum_grad_a += v * dwa * wb  * wc;
        accum_grad_b += v * wa  * dwb * wc;
        accum_grad_c += v * wa  * wb  * dwc;
      }
    }
  }

  // F_i_α = -q_i · (n_a · dΦ/du_a · b_a_α + n_b · dΦ/du_b · b_b_α + ...)
  // where (b_a, b_b, b_c) are rows of H^{-T}. For our lower-triangular
  // lattice, b_a / b_b / b_c are given by `reciprocal_lattice_rows`.
  Real b_a_x, b_a_y, b_a_z, b_b_x, b_b_y, b_b_z, b_c_x, b_c_y, b_c_z;
  reciprocal_lattice_rows(lx, ly, lz, xy, xz, yz,
                          b_a_x, b_a_y, b_a_z,
                          b_b_x, b_b_y, b_b_z,
                          b_c_x, b_c_y, b_c_z);
  Real ga = (Real) n_a * accum_grad_a;
  Real gb = (Real) n_b * accum_grad_b;
  Real gc = (Real) n_c * accum_grad_c;
  Real fx = -qi * (ga * b_a_x + gb * b_b_x + gc * b_c_x);
  Real fy = -qi * (ga * b_a_y + gb * b_b_y + gc * b_c_y);
  Real fz = -qi * (ga * b_a_z + gb * b_b_z + gc * b_c_z);

  slot_force_x[i] = fx;
  slot_force_y[i] = fy;
  slot_force_z[i] = fz;
  slot_energy[i]  = R(0.5) * qi * accum_phi - u_self_per_particle[i];
  slot_virial[i]  = w_per_particle_virial;
}
