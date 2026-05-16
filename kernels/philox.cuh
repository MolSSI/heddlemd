// rq-5e059f6b
//
// Shared device-side Philox-4×32-10 RNG primitives. Consumed by
// `langevin.cu` and `andersen.cu`; both `#include "philox.cuh"`. The
// algorithm description (key/counter convention, round structure,
// Box-Muller transform) lives in `rqm/integration/langevin-baoab.md`
// under *RNG*; this header is its single implementation site.

#ifndef DYNAMICS_PHILOX_CUH
#define DYNAMICS_PHILOX_CUH

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

// Generate one standard-normal sample for (particle_id, axis) at step_index.
__device__ inline float philox_gaussian(
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
  return (float)(r * cos(theta));
}

#endif // DYNAMICS_PHILOX_CUH
