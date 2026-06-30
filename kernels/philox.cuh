// rq-5e059f6b
//
// Shared device-side Philox-4×32-10 RNG primitives. Consumed by
// `langevin.cu` and `andersen.cu`; both `#include "philox.cuh"`. The
// algorithm description (key/counter convention, round structure,
// Box-Muller transform) lives in `rqm/integration/langevin-baoab.md`
// under *RNG*; this header is its single implementation site.

#ifndef DYNAMICS_PHILOX_CUH
#define DYNAMICS_PHILOX_CUH
#include "precision.cuh"

// --- Round constants (Salmon et al., SC11). --------------------------------

#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u  // Weyl increment for key word 0
#define PHILOX_W1 0xBB67AE85u  // Weyl increment for key word 1

__device__ inline unsigned int mulhi32(unsigned int a, unsigned int b)
{
  return __umulhi(a, b);
}

__device__ inline void philox4x32_10(
    unsigned int key_lo, unsigned int key_hi,
    unsigned int ctr0, unsigned int ctr1, unsigned int ctr2, unsigned int ctr3,
    unsigned int *out0, unsigned int *out1, unsigned int *out2, unsigned int *out3)
{
  unsigned int c0 = ctr0;
  unsigned int c1 = ctr1;
  unsigned int c2 = ctr2;
  unsigned int c3 = ctr3;
  unsigned int k0 = key_lo;
  unsigned int k1 = key_hi;

  for (int round = 0; round < 10; ++round) {
    unsigned int hi0 = mulhi32(c0, PHILOX_M0);
    unsigned int lo0 = c0 * PHILOX_M0;
    unsigned int hi2 = mulhi32(c2, PHILOX_M1);
    unsigned int lo2 = c2 * PHILOX_M1;

    unsigned int nc0 = hi2 ^ c1 ^ k0;
    unsigned int nc1 = lo2;
    unsigned int nc2 = hi0 ^ c3 ^ k1;
    unsigned int nc3 = lo0;

    c0 = nc0;
    c1 = nc1;
    c2 = nc2;
    c3 = nc3;

    k0 += PHILOX_W0;
    k1 += PHILOX_W1;
  }

  *out0 = c0;
  *out1 = c1;
  *out2 = c2;
  *out3 = c3;
}

// Convert one u32 to a double-precision uniform in (0, 1). The "+ 0.5" offset
// keeps the value strictly above 0 (so subsequent log(u1) is finite).
__device__ inline double u32_to_uniform_open(unsigned int x)
{
  const double scale = 1.0 / 4294967296.0; // 2^-32
  return ((double)x + 0.5) * scale;
}

// rq-eade13fb
// Precision-aware uniform converter. Returns a uniform Real in [0, 1).
// In the f32 build, consumes only `hi` (top 24 bits → 2^-24 step) and
// ignores `lo`. In the f64 build, concatenates the top 21 bits of `hi`
// with all 32 bits of `lo` to fill a 53-bit mantissa and divides by
// 2^53.
__device__ __forceinline__ Real philox_uniform_real(
    unsigned int hi, unsigned int lo)
{
#ifdef REAL_F64
  // 53-bit fill from 21 top bits of hi + 32 bits of lo.
  unsigned long long h21 = (unsigned long long)(hi >> 11);
  unsigned long long bits = (h21 << 32) | (unsigned long long)lo;
  const double scale = 1.0 / 9007199254740992.0; // 2^-53
  return (Real)((double)bits * scale);
#else
  unsigned int top24 = hi >> 8;
  const float scale = 1.0f / 16777216.0f; // 2^-24
  return (Real)((float)top24 * scale);
#endif
}

// rq-philox_normal_real_pair
// Marsaglia polar method: returns two independent unit-normal samples
// drawn from four Philox lanes. Consumes 2 (f32) or 4 (f64) lanes per
// call.
__device__ __forceinline__ void philox_normal_real_pair(
    unsigned int a, unsigned int b, unsigned int c, unsigned int d,
    Real *n0, Real *n1)
{
#ifdef REAL_F64
  Real u0 = philox_uniform_real(a, b);
  Real u1 = philox_uniform_real(c, d);
#else
  Real u0 = philox_uniform_real(a, 0u);
  Real u1 = philox_uniform_real(b, 0u);
  (void)c;
  (void)d;
#endif
  // Map u0, u1 in [0, 1) to s in (-1, 1).
  Real x = u0 * R(2.0) - R(1.0);
  Real y = u1 * R(2.0) - R(1.0);
  Real s = x * x + y * y;
  // Guard against s == 0 or s >= 1 by falling back to a deterministic
  // value (Marsaglia rejection is not feasible inside a single deterministic
  // device function; the caller chooses lanes that avoid this region).
  if (s <= R(0.0) || s >= R(1.0)) {
    s = R(0.5);
  }
  Real factor = Real_sqrt(R(-2.0) * Real_log(s) / s);
  *n0 = x * factor;
  *n1 = y * factor;
}

#ifdef REAL_F64
// rq-philox_lanes_uniform_real
__device__ __constant__ unsigned int PHILOX_LANES_PER_UNIFORM_REAL = 2;
// rq-philox_lanes_normal_real_pair
__device__ __constant__ unsigned int PHILOX_LANES_PER_NORMAL_REAL_PAIR = 4;
#else
__device__ __constant__ unsigned int PHILOX_LANES_PER_UNIFORM_REAL = 1;
__device__ __constant__ unsigned int PHILOX_LANES_PER_NORMAL_REAL_PAIR = 2;
#endif

// Generate one standard-normal sample for (particle_id, axis) at step_index.
__device__ inline Real philox_gaussian(
    unsigned int seed_lo, unsigned int seed_hi,
    unsigned int step_lo, unsigned int step_hi,
    unsigned int particle_id,
    unsigned int axis_id)
{
  unsigned int o0, o1, o2, o3;
  philox4x32_10(seed_lo, seed_hi,
                step_lo, step_hi, particle_id, axis_id,
                &o0, &o1, &o2, &o3);
  double u1 = u32_to_uniform_open(o0);
  double u2 = u32_to_uniform_open(o1);
  double r = sqrt(-2.0 * log(u1));
  double theta = 6.283185307179586 * u2; // 2 * pi
  return (Real)(r * cos(theta));
}

// Same Box-Muller layout as `philox_gaussian` but keeps the result in
// `double`. Used by the CSVR sample kernel, whose chain math (k_new from
// (k_old, k_target, c, s, r)) is sensitive to round-off and is kept in
// double precision regardless of the engine's storage precision.
__device__ inline double philox_gaussian_f64(
    unsigned int seed_lo, unsigned int seed_hi,
    unsigned int ctr0, unsigned int ctr1,
    unsigned int ctr2, unsigned int ctr3)
{
  unsigned int o0, o1, o2, o3;
  philox4x32_10(seed_lo, seed_hi, ctr0, ctr1, ctr2, ctr3,
                &o0, &o1, &o2, &o3);
  double u1 = u32_to_uniform_open(o0);
  double u2 = u32_to_uniform_open(o1);
  double r = sqrt(-2.0 * log(u1));
  double theta = 6.283185307179586 * u2;
  return r * cos(theta);
}

#endif // DYNAMICS_PHILOX_CUH
