// rq-94bfcb7e
// Order-specialized SPME charge-spread and force-gather kernels.
//
// This translation unit is compiled at runtime by NVRTC, once per run,
// with PME_ORDER supplied as a compile-time constant (the configured
// `spline_order`). It is NOT compiled by build.rs (it is a `.cuh`, not a
// `.cu`); the host assembles the NVRTC source by prepending the contents
// of `precision.cuh` and `pbc.cuh` (with their `#include` lines stripped)
// to this file, so `Real`, `R(...)`, `Real_floor`, `Real4`, and the
// triclinic PBC helpers are in scope. See `rqm/forces/spme.md`
// *Compile-time spline-order specialization*.
//
// Because PME_ORDER is a compile-time constant, the support loops fully
// unroll and the per-axis B-spline weight / derivative arrays are
// register-resident (no local-memory stack frame). The per-contribution
// arithmetic and accumulation order are identical to the generic order-p
// form, so the results are bit-identical and the determinism guarantees
// (i64 spread associativity, fixed gather summation order) are preserved.

#ifndef PME_ORDER
// Fallback so this file is self-consistent in an editor; the runtime
// JIT always defines PME_ORDER explicitly.
#define PME_ORDER 4
#endif

// Fixed-point scale factor: maps a real value `v` to the i64 integer
// (i64) rintf(v * SPREAD_FIXED_POINT_SCALE). With charges bounded by
// O(1 e) and B-spline weights bounded by 1, a single contribution
// maps to a value of magnitude <= a few * 10^9; the worst-case
// accumulated cell sum stays well under i64::MAX ~ 9.2 * 10^18.
#define SPREAD_FIXED_POINT_SCALE 4294967296.0f  // 2^32 as f32

// Cardinal B-spline M_p(x) via the Cox-de Boor recursion at the
// compile-time order PME_ORDER. M_1 is the indicator of [0, 1);
// successive orders are computed from neighbouring M_{p-1} samples.
__device__ static inline Real bspline_weight(Real x)
{
  // M_1(x - i) for i = 0..PME_ORDER-1.
  Real vals[PME_ORDER];
  for (int i = 0; i < PME_ORDER; ++i) {
    Real xi = x - (Real) i;
    vals[i] = (xi >= R(0.0) && xi < R(1.0)) ? R(1.0) : R(0.0);
  }
  // Recurse up to order PME_ORDER.
  for (int order = 2; order <= PME_ORDER; ++order) {
    Real inv = R(1.0) / (Real) (order - 1);
    for (int i = 0; i < PME_ORDER - order + 1; ++i) {
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
    Real x, Real &w, Real &dw)
{
  Real vals[PME_ORDER + 1];
  // M_1 indicators at x - i for i = 0..PME_ORDER (one extra so we can
  // read M_{p-1}(x) and M_{p-1}(x - 1) after the recursion).
  for (int i = 0; i < PME_ORDER + 1; ++i) {
    Real xi = x - (Real) i;
    vals[i] = (xi >= R(0.0) && xi < R(1.0)) ? R(1.0) : R(0.0);
  }
  // Recurse up to order PME_ORDER - 1.
  for (int order = 2; order < PME_ORDER; ++order) {
    Real inv = R(1.0) / (Real) (order - 1);
    for (int i = 0; i < PME_ORDER - order + 1; ++i) {
      Real xi = x - (Real) i;
      vals[i] = xi * inv * vals[i]
                + ((Real) order - xi) * inv * vals[i + 1];
    }
  }
  // vals[0] = M_{p-1}(x), vals[1] = M_{p-1}(x - 1).
  dw = vals[0] - vals[1];
  // Final step to compute M_p(x).
  {
    int order = PME_ORDER;
    Real inv = R(1.0) / (Real) (order - 1);
    Real xi = x;
    w = xi * inv * vals[0] + ((Real) order - xi) * inv * vals[1];
  }
}

// Compute fractional offsets t_a, t_b, t_c and primary bin
// (g_a, g_b, g_c) for one particle.
__device__ static inline void spread_per_particle_setup(
    Real px, Real py, Real pz,
    unsigned int n_a, unsigned int n_b, unsigned int n_c,
    Real lx, Real ly, Real lz, Real xy, Real xz, Real yz,
    Real &ta, Real &tb, Real &tc,
    unsigned int &g_a, unsigned int &g_b, unsigned int &g_c)
{
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
  Real fa = Real_floor(ua);
  Real fb = Real_floor(ub);
  Real fc = Real_floor(uc);
  ta = ua - fa;
  tb = ub - fb;
  tc = uc - fc;
  long fa_l = (long) fa;
  long fb_l = (long) fb;
  long fc_l = (long) fc;
  long n_a_l = (long) n_a;
  long n_b_l = (long) n_b;
  long n_c_l = (long) n_c;
  long ga_l = ((fa_l % n_a_l) + n_a_l) % n_a_l;
  long gb_l = ((fb_l % n_b_l) + n_b_l) % n_b_l;
  long gc_l = ((fc_l % n_c_l) + n_c_l) % n_c_l;
  g_a = (unsigned int) ga_l;
  g_b = (unsigned int) gb_l;
  g_c = (unsigned int) gc_l;
}

// Rows of the reciprocal lattice H^{-T} for the lower-triangular box.
__device__ static inline void reciprocal_lattice_rows(
    const Real *lattice,
    Real &b_a_x, Real &b_a_y, Real &b_a_z,
    Real &b_b_x, Real &b_b_y, Real &b_b_z,
    Real &b_c_x, Real &b_c_y, Real &b_c_z)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
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

// Per-particle fixed-point spread. PME_ORDER threads per atom, each
// owning one z-slice (`d_c = iz`) and looping over the
// PME_ORDER * PME_ORDER (d_a, d_b) cells in that slice. The per-slice
// mapping parallelises the atomicAdd traffic — the kernel's dominant
// cost — across the warp. The compile-time PME_ORDER unrolls the
// (d_a, d_b) loop so wa / wb are register-resident.
//
// Grid: ceil(N * PME_ORDER / 256) blocks of 256 threads each. The
// `spline_order` argument is unused (the order is the compile-time
// constant PME_ORDER); it is retained so the launch argument list is
// identical to the generic kernel.
extern "C" __global__ void spme_spread_fixed_point(
    const Real4        *posq,
    const unsigned int *sorted_atom_index, // length n
    const Real         *lattice,
    unsigned int        n_a, unsigned int n_b, unsigned int n_c,
    unsigned int        spline_order,
    long long          *rho_fixed,         // length M (i64)
    unsigned int        n)
{
  (void) spline_order;
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int total = n * (unsigned int) PME_ORDER;
  if (gid >= total) return;

  unsigned int atom_slot = gid / (unsigned int) PME_ORDER;
  unsigned int iz = gid - atom_slot * (unsigned int) PME_ORDER;
  unsigned int atom = sorted_atom_index[atom_slot];

  Real4 pq = posq[atom];
  Real qi = pq.w;
  if (qi == R(0.0)) return;

  Real px = pq.x;
  Real py = pq.y;
  Real pz = pq.z;
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];

  Real ta = R(0.0), tb = R(0.0), tc = R(0.0);
  unsigned int g_a = 0u, g_b = 0u, g_c = 0u;
  spread_per_particle_setup(
      px, py, pz, n_a, n_b, n_c, lx, ly, lz, xy, xz, yz,
      ta, tb, tc, g_a, g_b, g_c);

  Real wa[PME_ORDER], wb[PME_ORDER];
  for (int d = 0; d < PME_ORDER; ++d) {
    wa[d] = bspline_weight((Real) d + ta);
    wb[d] = bspline_weight((Real) d + tb);
  }
  Real wc_iz = bspline_weight((Real) iz + tc);

  unsigned int gc = (g_c + n_c - iz) % n_c;
  Real dz = qi * wc_iz;

  for (unsigned int da = 0; da < (unsigned int) PME_ORDER; ++da) {
    unsigned int ga = (g_a + n_a - da) % n_a;
    Real dzda = dz * wa[da];
    for (unsigned int db = 0; db < (unsigned int) PME_ORDER; ++db) {
      Real v = dzda * wb[db];
      long long v_fixed = (long long) __float2ll_rn(
          (float) v * SPREAD_FIXED_POINT_SCALE);
      if (v_fixed != 0LL) {
        unsigned int gb = (g_b + n_b - db) % n_b;
        unsigned int g = (ga * n_b + gb) * n_c + gc;
        atomicAdd((unsigned long long *) &rho_fixed[g],
                  (unsigned long long) v_fixed);
      }
    }
  }
}

// Force gather: one thread per particle. Per-axis B-spline weights,
// derivatives, and wrapped grid indices are evaluated once each; with
// the compile-time PME_ORDER the p^3 sweep unrolls and those arrays are
// register-resident. The `spline_order` argument is unused (see above).
extern "C" __global__ void spme_force_gather(
    const Real4        *posq,
    const Real         *V,
    const Real         *u_self_per_particle,
    const Real         *w_per_particle_virial,   // length 1
    const unsigned int *sorted_atom_index,       // length n
    const Real         *lattice,
    unsigned int        n_a, unsigned int n_b, unsigned int n_c,
    unsigned int        spline_order,
    Real *slot_force_x,
    Real *slot_force_y,
    Real *slot_force_z,
    Real *slot_energy,
    Real *slot_virial,
    unsigned int n)
{
  (void) spline_order;
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  Real w_per_particle_virial_val = w_per_particle_virial[0];
  unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) {
    return;
  }
  unsigned int i = sorted_atom_index[t];
  Real4 pq = posq[i];
  Real qi = pq.w;
  Real px = pq.x;
  Real py = pq.y;
  Real pz = pq.z;

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

  Real wa_arr[PME_ORDER], dwa_arr[PME_ORDER];
  Real wb_arr[PME_ORDER], dwb_arr[PME_ORDER];
  Real wc_arr[PME_ORDER], dwc_arr[PME_ORDER];
  int ga_arr[PME_ORDER], gb_arr[PME_ORDER], gc_arr[PME_ORDER];
  for (int d = 0; d < PME_ORDER; ++d) {
    bspline_weight_and_deriv((Real) d + ta, wa_arr[d], dwa_arr[d]);
    bspline_weight_and_deriv((Real) d + tb, wb_arr[d], dwb_arr[d]);
    bspline_weight_and_deriv((Real) d + tc, wc_arr[d], dwc_arr[d]);
    int g_a = ga0 - d;
    ga_arr[d] = ((g_a % (int) n_a) + (int) n_a) % (int) n_a;
    int g_b = gb0 - d;
    gb_arr[d] = ((g_b % (int) n_b) + (int) n_b) % (int) n_b;
    int g_c = gc0 - d;
    gc_arr[d] = ((g_c % (int) n_c) + (int) n_c) % (int) n_c;
  }

  Real accum_phi    = R(0.0);
  Real accum_grad_a = R(0.0);
  Real accum_grad_b = R(0.0);
  Real accum_grad_c = R(0.0);
  for (int da = 0; da < PME_ORDER; ++da) {
    Real wa = wa_arr[da], dwa = dwa_arr[da];
    int g_a = ga_arr[da];
    for (int db = 0; db < PME_ORDER; ++db) {
      Real wb = wb_arr[db], dwb = dwb_arr[db];
      int g_b = gb_arr[db];
      for (int dc = 0; dc < PME_ORDER; ++dc) {
        Real wc = wc_arr[dc], dwc = dwc_arr[dc];
        int g_c = gc_arr[dc];
        unsigned int g_idx =
            ((unsigned int) g_a * n_b + (unsigned int) g_b) * n_c
            + (unsigned int) g_c;
        Real v = V[g_idx];
        accum_phi    += v * wa * wb * wc;
        accum_grad_a += v * dwa * wb  * wc;
        accum_grad_b += v * wa  * dwb * wc;
        accum_grad_c += v * wa  * wb  * dwc;
      }
    }
  }

  Real b_a_x, b_a_y, b_a_z, b_b_x, b_b_y, b_b_z, b_c_x, b_c_y, b_c_z;
  reciprocal_lattice_rows(lattice,
                          b_a_x, b_a_y, b_a_z,
                          b_b_x, b_b_y, b_b_z,
                          b_c_x, b_c_y, b_c_z);
  Real ga = (Real) n_a * accum_grad_a;
  Real gb = (Real) n_b * accum_grad_b;
  Real gc = (Real) n_c * accum_grad_c;
  Real fx = -qi * (ga * b_a_x + gb * b_b_x + gc * b_c_x);
  Real fy = -qi * (ga * b_a_y + gb * b_b_y + gc * b_c_y);
  Real fz = -qi * (ga * b_a_z + gb * b_b_z + gc * b_c_z);

  slot_force_x[i] += fx;
  slot_force_y[i] += fy;
  slot_force_z[i] += fz;
  slot_energy[i]  += R(0.5) * qi * accum_phi - u_self_per_particle[i];
  slot_virial[i]  += w_per_particle_virial_val;
}
