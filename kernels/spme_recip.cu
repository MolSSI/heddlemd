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

// SPME reciprocal-space influence-function recompute. One thread per
// complex grid cell `k = (k_a, k_b, k_c)` with `k_c < n_c / 2 + 1`.
// Each thread writes a single entry of `influence_G` and `virial_factor`
// per
//
//   m_a = (k_a <= n_a / 2) ? k_a : k_a − n_a       (similar for b, c)
//   K   = 2π · (m_a · b_a + m_b · b_b + m_c · b_c)
//   K²  = |K|²
//   G[k]            = (k_C / V_box) · (4π / K²) · exp(-K² / (4 α²))
//                     · b_factors_a[k_a] · b_factors_b[k_b]
//                     · b_factors_c[k_c]
//   virial_factor[k] = G[k] · (1 − K² / (2 α²))
//
// where (b_a, b_b, b_c) are rows of the reciprocal lattice H^(-T)
// derived from the lower-triangular lattice parameters
// (lx, ly, lz, xy, xz, yz), `V_box = lx · ly · lz`, and `k_C` is the
// Coulomb prefactor (1 in atomic units). The `k = (0, 0, 0)` cell is
// set to zero for both buffers, implementing tinfoil boundary
// conditions.
//
// Determinism: one thread per cell with no inter-thread communication
// and no atomics. All inner arithmetic is `double` precision; the
// final value is cast to `Real` at the device-store site. Two runs on
// the same GPU with byte-identical inputs produce byte-identical
// `influence_G` and `virial_factor`.
extern "C" __global__ void spme_recip_compute_influence(
    const Real *b_factors_a,           // length n_a
    const Real *b_factors_b,           // length n_b
    const Real *b_factors_c,           // length n_c
    Real *influence_G,                 // length m_complex
    Real *virial_factor,               // length m_complex
    const Real *lattice,
    unsigned int n_a, unsigned int n_b, unsigned int n_c,
    Real k_coulomb,
    Real alpha,
    unsigned int m_complex)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= m_complex) {
    return;
  }

  unsigned int n_c_complex = (n_c / 2u) + 1u;
  unsigned int kc = idx % n_c_complex;
  unsigned int kab = idx / n_c_complex;
  unsigned int kb = kab % n_b;
  unsigned int ka = kab / n_b;

  // f64 internal arithmetic.
  double lx_d = (double) lx;
  double ly_d = (double) ly;
  double lz_d = (double) lz;
  double xy_d = (double) xy;
  double xz_d = (double) xz;
  double yz_d = (double) yz;
  double alpha_d = (double) alpha;
  double v_box = lx_d * ly_d * lz_d;
  double four_alpha2 = 4.0 * alpha_d * alpha_d;
  const double pi_d = 3.141592653589793238;
  double four_pi = 4.0 * pi_d;
  double two_pi = 2.0 * pi_d;
  double prefactor = (double) k_coulomb / v_box;

  double m_a = (ka <= n_a / 2u) ? (double) ka : (double) ka - (double) n_a;
  double m_b = (kb <= n_b / 2u) ? (double) kb : (double) kb - (double) n_b;
  double m_c = (kc <= n_c / 2u) ? (double) kc : (double) kc - (double) n_c;

  // Reciprocal lattice rows of H^(-T) in f64. Matches the host-side
  // closed form for lower-triangular H.
  double inv_lx = 1.0 / lx_d;
  double inv_ly = 1.0 / ly_d;
  double inv_lz = 1.0 / lz_d;
  double recip_a_x = inv_lx;
  double recip_a_y = -xy_d * inv_lx * inv_ly;
  double recip_a_z = (xy_d * yz_d - xz_d * ly_d) * inv_lx * inv_ly * inv_lz;
  double recip_b_y = inv_ly;
  double recip_b_z = -yz_d * inv_ly * inv_lz;
  double recip_c_z = inv_lz;

  double kx = two_pi * (m_a * recip_a_x);
  double ky = two_pi * (m_a * recip_a_y + m_b * recip_b_y);
  double kz = two_pi
              * (m_a * recip_a_z + m_b * recip_b_z + m_c * recip_c_z);
  double k2 = kx * kx + ky * ky + kz * kz;

  Real g_out;
  Real vf_out;
  if (k2 == 0.0) {
    g_out = R(0.0);
    vf_out = R(0.0);
  } else {
    double b_a = (double) b_factors_a[ka];
    double b_b = (double) b_factors_b[kb];
    double b_c = (double) b_factors_c[kc];
    double g = prefactor * (four_pi / k2)
               * exp(-k2 / four_alpha2)
               * b_a * b_b * b_c;
    double virial_term = 1.0 - k2 / (2.0 * alpha_d * alpha_d);
    g_out = (Real) g;
    vf_out = (Real)(g * virial_term);
  }
  influence_G[idx] = g_out;
  virial_factor[idx] = vf_out;
}

// rq-9ca00d25 rq-382a6e66 rq-a1b761fa
//
// Fixed-point atomic-add charge-spread pipeline. Two kernels:
//   1. spme_spread_fixed_point — one warp per particle. Each lane
//                                computes its assigned (d_a, d_b, d_c)
//                                contribution and issues
//                                atomicAdd<i64>(&rho_fixed[g], v_fixed)
//                                where v_fixed = (i64) rintf(q · w_a ·
//                                w_b · w_c · 2^32). Total N · p^3 atomic
//                                adds per step.
//   2. spme_spread_finish      — one thread per real grid cell. Reads
//                                rho_fixed[c] and writes
//                                rho[c] = (f32) rho_fixed[c] · 2^-32.
//
// Determinism: i64 atomic addition is exactly associative on the same
// GPU regardless of atomic-completion order, so the accumulated
// rho_fixed grid is byte-identical across runs with byte-identical
// inputs. The f32 conversion is per-cell with no inter-thread
// communication, also deterministic.

// Compute fractional offsets t_a, t_b, t_c and primary bin
// (g_a, g_b, g_c) for one particle. Replicates the geometry of the
// original spme_charge_spread kernel so the per-particle weights are
// computed identically.
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
  // The primary bin (lowest-(d_a, d_b, d_c) corner) of the p^3 support
  // is one to the left of floor(u_d) so the offsets (d_a, d_b, d_c) in
  // {0, 1, ..., p-1} place the particle correctly inside the box of
  // p^3 grid cells. This matches the original spme_charge_spread
  // kernel's geometry: each grid cell (g) thread iterated d in [0, p)
  // and read from bin (g + d) mod n_d, so the inverse mapping (which
  // particle contributions hit grid cell g) is exactly (primary + d)
  // mod n_d with the same primary bin.
  long fa_l = (long) fa;
  long fb_l = (long) fb;
  long fc_l = (long) fc;
  // Wrap into [0, n_d).
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

// Fixed-point scale factor: maps a real value `v` to the i64 integer
// (i64) rintf(v * SPREAD_FIXED_POINT_SCALE). With charges bounded by
// O(1 e) and B-spline weights bounded by 1, a single contribution
// maps to a value of magnitude <= a few * 10^9; the worst-case
// accumulated cell sum stays well under i64::MAX ~ 9.2 * 10^18.
#define SPREAD_FIXED_POINT_SCALE 4294967296.0f  // 2^32 as f32

// Per-particle fixed-point spread. PME_ORDER (= spline_order) threads
// per atom, each thread owning one z-slice (`d_c = iz`) of the atom's
// p^3 spline-support cube and running a tight nested loop over the
// PME_ORDER * PME_ORDER = p^2 (d_a, d_b) cells in that slice. Each
// (d_a, d_b, iz) contribution issues one atomicAdd<i64> into
// rho_fixed, with a v_fixed != 0 zero-skip guard to elide the
// contribution when the fixed-point quantisation rounds to zero
// (matching OpenMM's gridSpreadCharge pattern). Grid:
// ceil(N * spline_order / 256) blocks of 256 threads each.
extern "C" __global__ void spme_spread_fixed_point(
    const Real         *positions_x,
    const Real         *positions_y,
    const Real         *positions_z,
    const Real         *charges,
    const unsigned int *sorted_atom_index, // length n
    const Real         *lattice,
    unsigned int        n_a, unsigned int n_b, unsigned int n_c,
    unsigned int        spline_order,
    long long          *rho_fixed,         // length M (i64)
    unsigned int        n)
{
  int p = (int) spline_order;
  unsigned int p_u = (unsigned int) p;
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int total = n * p_u;
  if (gid >= total) return;

  unsigned int atom_slot = gid / p_u;
  unsigned int iz = gid - atom_slot * p_u;          // z-slice for this thread
  unsigned int atom = sorted_atom_index[atom_slot];

  // Charge-zero skip — matches OpenMM. Atoms with q == 0 contribute
  // nothing to the grid.
  Real qi = charges[atom];
  if (qi == R(0.0)) return;

  Real px = positions_x[atom];
  Real py = positions_y[atom];
  Real pz = positions_z[atom];
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];

  Real ta = R(0.0), tb = R(0.0), tc = R(0.0);
  unsigned int g_a = 0u, g_b = 0u, g_c = 0u;
  spread_per_particle_setup(
      px, py, pz, n_a, n_b, n_c, lx, ly, lz, xy, xz, yz,
      ta, tb, tc, g_a, g_b, g_c);

  // Per-axis 1D B-spline weights. This thread needs the full
  // wa[0..p), wb[0..p) (for the nested (d_a, d_b) loop) plus
  // wc[iz] only (its assigned z-slice).
  Real wa[8], wb[8];
  for (int d = 0; d < p; ++d) {
    wa[d] = bspline_weight(p, (Real) d + ta);
    wb[d] = bspline_weight(p, (Real) d + tb);
  }
  Real wc_iz = bspline_weight(p, (Real) iz + tc);

  // Inverse mapping: a particle at primary bin (g_a, g_b, g_c)
  // contributes to grid cells ((g - d) mod n) for d in [0, p).
  unsigned int gc = (g_c + n_c - iz) % n_c;
  Real dz = qi * wc_iz;

  // Tight (d_a, d_b) loop: PME_ORDER * PME_ORDER = 16 cells for p = 4.
  for (unsigned int da = 0; da < p_u; ++da) {
    unsigned int ga = (g_a + n_a - da) % n_a;
    Real dzda = dz * wa[da];
    for (unsigned int db = 0; db < p_u; ++db) {
      Real v = dzda * wb[db];
      long long v_fixed = (long long) __float2ll_rn(
          (float) v * SPREAD_FIXED_POINT_SCALE);
      // Zero-skip: contributions whose round-to-nearest fixed-point
      // value is zero contribute nothing to the i64 sum and can be
      // elided without changing the grid bit-pattern.
      if (v_fixed != 0LL) {
        unsigned int gb = (g_b + n_b - db) % n_b;
        unsigned int g = (ga * n_b + gb) * n_c + gc;
        atomicAdd((unsigned long long *) &rho_fixed[g],
                  (unsigned long long) v_fixed);
      }
    }
  }
}

// rq-06f1edf2 rq-7594b1fc rq-a1b761fc
//
// Atom spatial pre-sort key computation. One thread per particle.
// Each thread runs the same `spread_per_particle_setup` geometry as
// the spread / gather kernels, takes the primary bin `(g_a, g_b, g_c)`,
// flattens it to the row-major key
// `b = g_a · n_b · n_c + g_b · n_c + g_c`, writes it to
// `atom_bin_key[i]`, and atomically increments `bin_atom_counts[b]`.
//
// The histogram atomicAdd reduces to integer addition of `+1`s and is
// independent of completion order, so the per-bin atom count is
// deterministic across runs. The caller is responsible for zeroing
// `bin_atom_counts` (length M) before this kernel launches.
extern "C" __global__ void spme_compute_bin_key(
    const Real   *positions_x,
    const Real   *positions_y,
    const Real   *positions_z,
    const Real   *lattice,
    unsigned int  n_a, unsigned int n_b, unsigned int n_c,
    unsigned int *atom_bin_key,
    unsigned int *bin_atom_counts,
    unsigned int  n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  Real px = positions_x[i];
  Real py = positions_y[i];
  Real pz = positions_z[i];
  Real ta, tb, tc;
  unsigned int g_a, g_b, g_c;
  spread_per_particle_setup(
      px, py, pz, n_a, n_b, n_c, lx, ly, lz, xy, xz, yz,
      ta, tb, tc, g_a, g_b, g_c);
  unsigned int b = (g_a * n_b + g_b) * n_c + g_c;
  atom_bin_key[i] = b;
  atomicAdd(&bin_atom_counts[b], 1u);
}

// Fixed-point -> f32 conversion. One thread per real grid cell.
extern "C" __global__ void spme_spread_finish(
    const long long *rho_fixed,         // length M (i64)
    Real            *rho,               // length M (f32 or f64)
    unsigned int     M)
{
  unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= M) {
    return;
  }
  // Reinterpret the unsigned bit pattern as signed i64. atomicAdd on
  // unsigned long long wraps modulo 2^64 the same way two's-complement
  // i64 addition wraps, so the bit pattern of every accumulated cell
  // is the correct i64 sum regardless of intermediate sign changes.
  long long fp = rho_fixed[c];
  Real inv_scale = R(1.0) / (Real) SPREAD_FIXED_POINT_SCALE;
  rho[c] = (Real) fp * inv_scale;
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
// rq-95385a9d
//
// Fused influence-multiply + per-block virial partial-sum reduction.
//
// One thread per complex grid cell:
//   - Reads rho_hat[k], influence_G[k], virial_factor[k].
//   - Writes V_hat[k] = influence_G[k] * rho_hat[k] in place into
//     rho_hat_interleaved (including a zero write at k = 0 by virtue of
//     influence_G[0] == 0, enforced by spme_recip_compute_influence).
//   - Computes the per-thread Hermitian-weighted virial contribution
//     c_k = hw[k] * virial_factor[k] * |rho_hat[k]|² and accumulates
//     it into shared memory.
//
// The block-level reduction uses the same deterministic shape as
// spme_recip_reduce_partials / barostat::virial_sum_reduce: a left-to-
// right pairwise tree in shared memory. The tree shape depends only on
// the launch block size (256 = blockDim.x), so two runs with byte-
// identical inputs on the same GPU produce byte-identical V_hat and
// byte-identical virial_partials.
//
// Lanes with idx >= n_complex contribute 0 to the reduction.
extern "C" __global__ void spme_recip_apply_influence(
    const Real *influence_G,
    const Real *virial_factor,
    Real *rho_hat_interleaved,
    Real *virial_partials,            // length = ceil(n_complex / 256)
    unsigned int n_c,
    unsigned int n_c_complex,
    unsigned int n_complex)
{
  __shared__ Real partial[256];

  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int tid = threadIdx.x;

  Real c = R(0.0);
  if (idx < n_complex) {
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
    c = (Real) hw * vf * rho_sq;

    rho_hat_interleaved[base]      = g * re;
    rho_hat_interleaved[base + 1u] = g * im;
  }
  partial[tid] = c;
  __syncthreads();

  for (unsigned int stride = 1u; stride < blockDim.x; stride *= 2u) {
    if ((tid % (2u * stride)) == 0u && (tid + stride) < blockDim.x) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0u) {
    virial_partials[blockIdx.x] = partial[0];
  }
}

// Compute the reciprocal-lattice rows (b_a, b_b, b_c) from the six
// lower-triangular lattice parameters. Returns them in column-major
// triples (b_*_x, b_*_y, b_*_z).
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
// The reciprocal-space scalar virial `W_recip` is reduced on device from
// `virial_partials` (the per-block partial sums written by
// spme_recip_apply_influence); this kernel writes the uniform per-particle
// share `w_per_particle_virial = W_recip / N` into the device-resident
// single-element scalar that `spme_force_gather` reads.
// Single-block deterministic reduction of `virial_partials` followed by
// the Ewald half-sum / per-particle scale: writes
//   w_per_particle_virial[0] = scale * Σ virial_partials[b]
// with `scale = 0.5 / n`. Same shape as `barostat::virial_sum_reduce`:
// one block of 256 threads, strided per-thread accumulator, deterministic
// left-to-right pairwise tree in shared memory. Two runs with
// byte-identical inputs on the same GPU produce a byte-identical
// `w_per_particle_virial[0]`.
extern "C" __global__ void spme_recip_reduce_partials(
    const Real *virial_partials,
    Real *w_per_particle_virial,   // length 1; only thread 0 writes
    unsigned int num_blocks,
    Real scale)
{
  __shared__ Real partial[256];

  unsigned int tid = threadIdx.x;
  Real sum = R(0.0);
  for (unsigned int i = tid; i < num_blocks; i += blockDim.x) {
    sum += virial_partials[i];
  }
  partial[tid] = sum;
  __syncthreads();

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2u * stride)) == 0u && (tid + stride) < blockDim.x) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0u) {
    w_per_particle_virial[0] = partial[0] * scale;
  }
}

extern "C" __global__ void spme_force_gather(
    const Real         *positions_x,
    const Real         *positions_y,
    const Real         *positions_z,
    const Real         *charges,
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
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  Real w_per_particle_virial_val = w_per_particle_virial[0];
  unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) {
    return;
  }
  // Sorted-slot indirection: consecutive threads address atoms with
  // nearby primary bins so the per-thread reads of V[g] cluster on
  // neighbouring cache lines. Per-particle outputs are written to the
  // canonical particle-index positions slot_*[i], preserving the
  // slot-output layout regardless of the sort permutation.
  unsigned int i = sorted_atom_index[t];
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
