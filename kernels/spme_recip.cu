// rq-9ca00d25 rq-202493a5

#include "pbc.cuh"

// Cardinal B-spline M_p(x) via the Cox-de Boor recursion. M_1 is the
// indicator of [0, 1); successive orders are computed from neighbouring
// M_{p-1} samples. The function works for any p in [2, 8].
__device__ static inline float bspline_weight(int p, float x)
{
  // M_1(x - i) for i = 0..p-1.
  float vals[9];
  for (int i = 0; i < p; ++i) {
    float xi = x - (float) i;
    vals[i] = (xi >= 0.0f && xi < 1.0f) ? 1.0f : 0.0f;
  }
  // Recurse up to order p.
  for (int order = 2; order <= p; ++order) {
    float inv = 1.0f / (float) (order - 1);
    for (int i = 0; i < p - order + 1; ++i) {
      float xi = x - (float) i;
      vals[i] = xi * inv * vals[i]
                + ((float) order - xi) * inv * vals[i + 1];
    }
  }
  return vals[0];
}

// Compute M_p(x) and M_p'(x) = M_{p-1}(x) - M_{p-1}(x-1) in one pass.
// The derivative identity follows from the Cox-de Boor recursion.
__device__ static inline void bspline_weight_and_deriv(
    int p, float x, float &w, float &dw)
{
  float vals[9];
  // M_1 indicators at x - i for i = 0..p (one extra so we can read
  // M_{p-1}(x) and M_{p-1}(x - 1) after the recursion).
  for (int i = 0; i < p + 1; ++i) {
    float xi = x - (float) i;
    vals[i] = (xi >= 0.0f && xi < 1.0f) ? 1.0f : 0.0f;
  }
  // Recurse up to order p - 1.
  for (int order = 2; order < p; ++order) {
    float inv = 1.0f / (float) (order - 1);
    for (int i = 0; i < p - order + 1; ++i) {
      float xi = x - (float) i;
      vals[i] = xi * inv * vals[i]
                + ((float) order - xi) * inv * vals[i + 1];
    }
  }
  // vals[0] = M_{p-1}(x), vals[1] = M_{p-1}(x - 1).
  dw = vals[0] - vals[1];
  // Final step to compute M_p(x).
  {
    int order = p;
    float inv = 1.0f / (float) (order - 1);
    float xi = x;
    w = xi * inv * vals[0] + ((float) order - xi) * inv * vals[1];
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
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    const float *charges,
    const unsigned int *sorted_particle_ids,
    const unsigned int *cell_offsets,
    float lx, float ly, float lz, float xy, float xz, float yz,
    unsigned int n_a, unsigned int n_b, unsigned int n_c,
    unsigned int spline_order,
    float *rho,
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
  float accum = 0.0f;

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
          float px = positions_x[i];
          float py = positions_y[i];
          float pz = positions_z[i];
          // Re-wrap defensively (the integrator already wraps, but f32
          // round-off can leave fractional coords just outside [-0.5, 0.5)).
          int wrap_a, wrap_b, wrap_c;
          triclinic_wrap_with_image(px, py, pz, wrap_a, wrap_b, wrap_c,
                                    lx, ly, lz, xy, xz, yz);
          float sa, sb, sc;
          triclinic_cart_to_frac(px, py, pz, lx, ly, lz, xy, xz, yz,
                                 sa, sb, sc);
          float sa_p = sa + 0.5f;
          float sb_p = sb + 0.5f;
          float sc_p = sc + 0.5f;
          float ua = sa_p * (float) n_a;
          float ub = sb_p * (float) n_b;
          float uc = sc_p * (float) n_c;
          float ta = ua - floorf(ua);
          float tb = ub - floorf(ub);
          float tc = uc - floorf(uc);
          float wa = bspline_weight(p, (float) da + ta);
          float wb = bspline_weight(p, (float) db + tb);
          float wc = bspline_weight(p, (float) dc + tc);
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
// interleaved (real, imag) float pairs in row-major order, with the
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
    const float *influence_G,
    const float *virial_factor,
    float *rho_hat_interleaved,
    float *virial_per_cell,
    unsigned int n_c,
    unsigned int n_c_complex,
    unsigned int n_complex)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_complex) {
    return;
  }
  float g = influence_G[idx];
  float vf = virial_factor[idx];
  unsigned int base = idx * 2u;
  float re = rho_hat_interleaved[base];
  float im = rho_hat_interleaved[base + 1u];
  float rho_sq = re * re + im * im;
  unsigned int kc = idx % n_c_complex;
  // Hermitian weight: count modes paired across complex conjugation.
  // Modes at kc == 0 and (for even n_c) at kc == n_c/2 are self-paired
  // and contribute once; all other kc contribute twice.
  unsigned int hw =
      (kc == 0u || (n_c % 2u == 0u && 2u * kc == n_c)) ? 1u : 2u;
  virial_per_cell[idx] = (float) hw * vf * rho_sq;

  rho_hat_interleaved[base]      = g * re;
  rho_hat_interleaved[base + 1u] = g * im;
}

// Compute the reciprocal-lattice rows (b_a, b_b, b_c) from the six
// lower-triangular lattice parameters. Returns them in column-major
// triples (b_*_x, b_*_y, b_*_z).
__device__ static inline void reciprocal_lattice_rows(
    float lx, float ly, float lz, float xy, float xz, float yz,
    float &b_a_x, float &b_a_y, float &b_a_z,
    float &b_b_x, float &b_b_y, float &b_b_z,
    float &b_c_x, float &b_c_y, float &b_c_z)
{
  float inv_lx = 1.0f / lx;
  float inv_ly = 1.0f / ly;
  float inv_lz = 1.0f / lz;
  b_a_x = inv_lx;
  b_a_y = -xy * inv_lx * inv_ly;
  b_a_z = (xy * yz - xz * ly) * inv_lx * inv_ly * inv_lz;
  b_b_x = 0.0f;
  b_b_y = inv_ly;
  b_b_z = -yz * inv_ly * inv_lz;
  b_c_x = 0.0f;
  b_c_y = 0.0f;
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
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    const float *charges,
    const float *V,
    const float *u_self_per_particle,
    float w_per_particle_virial,
    float lx, float ly, float lz, float xy, float xz, float yz,
    unsigned int n_a, unsigned int n_b, unsigned int n_c,
    unsigned int spline_order,
    float *slot_force_x,
    float *slot_force_y,
    float *slot_force_z,
    float *slot_energy,
    float *slot_virial,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  float qi = charges[i];
  float px = positions_x[i];
  float py = positions_y[i];
  float pz = positions_z[i];

  // Re-wrap defensively, matching the spread kernel.
  int wrap_a, wrap_b, wrap_c;
  triclinic_wrap_with_image(px, py, pz, wrap_a, wrap_b, wrap_c,
                            lx, ly, lz, xy, xz, yz);
  float sa, sb, sc;
  triclinic_cart_to_frac(px, py, pz, lx, ly, lz, xy, xz, yz, sa, sb, sc);
  float sa_p = sa + 0.5f;
  float sb_p = sb + 0.5f;
  float sc_p = sc + 0.5f;
  float ua = sa_p * (float) n_a;
  float ub = sb_p * (float) n_b;
  float uc = sc_p * (float) n_c;
  int ga0 = (int) floorf(ua);
  int gb0 = (int) floorf(ub);
  int gc0 = (int) floorf(uc);
  float ta = ua - (float) ga0;
  float tb = ub - (float) gb0;
  float tc = uc - (float) gc0;

  int p = (int) spline_order;
  float accum_phi    = 0.0f;
  float accum_grad_a = 0.0f;  // dΦ/dt_a (in fractional-grid units)
  float accum_grad_b = 0.0f;
  float accum_grad_c = 0.0f;

  for (int da = 0; da < p; ++da) {
    float wa, dwa;
    bspline_weight_and_deriv(p, (float) da + ta, wa, dwa);
    int g_a = ga0 - da;
    g_a = ((g_a % (int) n_a) + (int) n_a) % (int) n_a;
    for (int db = 0; db < p; ++db) {
      float wb, dwb;
      bspline_weight_and_deriv(p, (float) db + tb, wb, dwb);
      int g_b = gb0 - db;
      g_b = ((g_b % (int) n_b) + (int) n_b) % (int) n_b;
      for (int dc = 0; dc < p; ++dc) {
        float wc, dwc;
        bspline_weight_and_deriv(p, (float) dc + tc, wc, dwc);
        int g_c = gc0 - dc;
        g_c = ((g_c % (int) n_c) + (int) n_c) % (int) n_c;
        unsigned int g_idx =
            ((unsigned int) g_a * n_b + (unsigned int) g_b) * n_c
            + (unsigned int) g_c;
        float v = V[g_idx];
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
  float b_a_x, b_a_y, b_a_z, b_b_x, b_b_y, b_b_z, b_c_x, b_c_y, b_c_z;
  reciprocal_lattice_rows(lx, ly, lz, xy, xz, yz,
                          b_a_x, b_a_y, b_a_z,
                          b_b_x, b_b_y, b_b_z,
                          b_c_x, b_c_y, b_c_z);
  float ga = (float) n_a * accum_grad_a;
  float gb = (float) n_b * accum_grad_b;
  float gc = (float) n_c * accum_grad_c;
  float fx = -qi * (ga * b_a_x + gb * b_b_x + gc * b_c_x);
  float fy = -qi * (ga * b_a_y + gb * b_b_y + gc * b_c_y);
  float fz = -qi * (ga * b_a_z + gb * b_b_z + gc * b_c_z);

  slot_force_x[i] = fx;
  slot_force_y[i] = fy;
  slot_force_z[i] = fz;
  slot_energy[i]  = 0.5f * qi * accum_phi - u_self_per_particle[i];
  slot_virial[i]  = w_per_particle_virial;
}
