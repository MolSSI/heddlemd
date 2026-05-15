// rq-9ca00d25 rq-202493a5

#include "pbc.cuh"

// Cardinal B-spline M_p(x) via the Cox-de Boor recursion. M_1 is the
// indicator of [0, 1); successive orders are computed from neighbouring
// M_{p-1} samples. The function works for any p in [2, 8]; PR 1 uses
// p = 4 by default but the kernel accepts the order as a runtime
// argument so future PRs can lift the restriction without a kernel
// rewrite.
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
// Operates in place: `rho_hat` is both read and written. The k=0 cell
// is identified by its flat index 0 (since (0, 0, 0) has row-major
// index 0).
extern "C" __global__ void spme_influence_multiply(
    const float *influence_G,
    float *rho_hat_interleaved,
    unsigned int n_complex)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_complex) {
    return;
  }
  float g = influence_G[idx];
  unsigned int base = idx * 2u;
  float re = rho_hat_interleaved[base];
  float im = rho_hat_interleaved[base + 1u];
  rho_hat_interleaved[base]      = g * re;
  rho_hat_interleaved[base + 1u] = g * im;
}
