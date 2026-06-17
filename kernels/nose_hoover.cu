// rq-f606ff6f

// Single-block deterministic kinetic-energy reduction.
//
// One block of 256 threads is launched. Each thread accumulates the
// per-particle KE contribution `½ m (vx² + vy² + vz²)` for the slice
// of particles whose indices are congruent to threadIdx.x modulo 256
// (i.e. thread `t` sees particles `t, t + 256, t + 512, …`). The
// strided per-thread sum is computed in an `f32` register without any
// inter-thread interaction.
//
// The per-thread partials are then reduced via a deterministic
// left-to-right pairwise tree in shared memory: at each stride `s`
// (1, 2, 4, …, blockDim.x/2), the lower half of every pair absorbs
// the upper half. The reduction topology and visitation order are
// completely determined by `blockDim.x` and `n`, so two runs with
// byte-identical inputs on the same GPU produce a byte-identical
// `partial_out[0]`.
#include "precision.cuh"
#include "philox.cuh"

extern "C" __global__ void kinetic_energy_reduce(
    const Real *velocities_x,
    const Real *velocities_y,
    const Real *velocities_z,
    const Real *masses,
    Real *partial_out,    // length 1; only thread 0 writes
    unsigned int n)
{
  __shared__ Real partial[256];

  unsigned int tid = threadIdx.x;
  Real sum = R(0.0);
  for (unsigned int i = tid; i < n; i += blockDim.x) {
    Real vx = velocities_x[i];
    Real vy = velocities_y[i];
    Real vz = velocities_z[i];
    Real m = masses[i];
    sum += R(0.5) * m * (vx * vx + vy * vy + vz * vz);
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
    partial_out[0] = partial[0];
  }
}

// Uniform per-particle velocity rescale: v_i ← factor · v_i for every
// component of every particle. One thread per particle, no inter-thread
// interaction.
extern "C" __global__ void rescale_velocities(
    Real *velocities_x,
    Real *velocities_y,
    Real *velocities_z,
    Real factor,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  velocities_x[i] *= factor;
  velocities_y[i] *= factor;
  velocities_z[i] *= factor;
}

// Same as `rescale_velocities` but reads the rescale factor from a
// single-element device buffer. Used when the factor is computed on
// device (e.g. by `csvr_sample_and_factor`) and the host never
// downloads it.
extern "C" __global__ void rescale_velocities_device_factor(
    Real *velocities_x,
    Real *velocities_y,
    Real *velocities_z,
    const Real *factor,        // length 1
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  Real f = factor[0];
  velocities_x[i] *= f;
  velocities_y[i] *= f;
  velocities_z[i] *= f;
}

// Single-block CSVR (Bussi-Donadio-Parrinello) sampling kernel.
//
// Computes the velocity rescale factor for one CSVR step entirely on
// device. Reads `k_old[0]` (the current kinetic energy in Hartrees,
// produced by `kinetic_energy_reduce`), draws `g_dof` standard-normal
// samples from Philox-4×32-10, evaluates the CSVR chain
//
//   k_new = c · k_old + (k_target / nf) · (1 - c) · (s + r²)
//         + 2 r · √(c · (1 - c) · k_old · k_target / nf)
//
// in double precision, and writes
//
//   factor_out[0] = √(k_new / k_old)        (or 1.0 in the
//                                            edge cases the host
//                                            code also handles)
//   cumulative_injection_delta[0] += (k_new - k_old)
//
// for the runner to consume at log-write cadence.
//
// Reduction shape: one block of 256 threads. Thread `t` accumulates
// xi² for sample indices `t + 1, t + 1 + 256, …` (the host code's
// `1..g_dof` loop, partitioned by lane), then a deterministic
// left-to-right pairwise tree in shared memory reduces the 256
// partials. Thread 0 of block 0 draws the lone `r` sample (counter
// `(seed, draw_counter, sample_index=0, 0)`, matching the host's
// `philox_normal(..., 0, 0)`), assembles k_new, and writes the
// outputs.
//
// Determinism: the parallel-reduction order is fixed by (block size,
// g_dof) and identical across runs with the same seed and
// draw_counter; two runs on the same GPU produce a byte-identical
// `factor_out[0]`. The reduction order differs from the host's
// `for sample_index in 1..g_dof` serial sum, so the new factor does
// not bit-match the legacy host-side CSVR; the trajectory remains
// statistically equivalent and the equilibrium NVT distribution is
// preserved.
extern "C" __global__ void csvr_sample_and_factor(
    const Real *k_old,                       // length 1; Real (matches kinetic_energy_reduce output)
    Real *factor_out,                        // length 1
    double *cumulative_injection_delta,      // length 1; atomic-not-required (only lane 0 writes)
    unsigned int seed_lo,
    unsigned int seed_hi,
    unsigned int draw_counter_lo,
    unsigned int draw_counter_hi,
    unsigned int g_dof,
    double c,                                // exp(-dt / tau)
    double one_minus_c,                      // 1 - c
    double k_target_over_nf)                 // k_target / nf  where  k_target = (nf/2) · kt_target
{
  __shared__ double partial[256];

  unsigned int tid = threadIdx.x;

  // Accumulate s = Σ_{i=1..g_dof - 1} xi_i² in parallel.
  double s = 0.0;
  for (unsigned int i = tid + 1u; i < g_dof; i += blockDim.x) {
    double xi = philox_gaussian_f64(
        seed_lo, seed_hi, draw_counter_lo, draw_counter_hi, i, 0u);
    s += xi * xi;
  }
  partial[tid] = s;
  __syncthreads();

  // Deterministic left-to-right pairwise tree, identical in shape to
  // `kinetic_energy_reduce` above.
  for (unsigned int stride = 1u; stride < blockDim.x; stride *= 2u) {
    if ((tid % (2u * stride)) == 0u && (tid + stride) < blockDim.x) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0u) {
    double s_total = partial[0];
    double r = philox_gaussian_f64(
        seed_lo, seed_hi, draw_counter_lo, draw_counter_hi, 0u, 0u);
    double k_old_val = (double)k_old[0];

    double cross = 0.0;
    if (k_old_val > 0.0) {
      cross = 2.0 * r * sqrt(c * one_minus_c * k_old_val * k_target_over_nf);
    }
    double k_new = c * k_old_val
                 + k_target_over_nf * one_minus_c * (s_total + r * r)
                 + cross;
    if (!isfinite(k_new) || k_new <= 0.0) {
      k_new = k_old_val;
    }

    double f = 1.0;
    if (k_old_val > 0.0) {
      double diff = k_new - k_old_val;
      if (diff < 0.0) diff = -diff;
      if (diff > 0.0) {
        f = sqrt(k_new / k_old_val);
      }
    }
    factor_out[0] = (Real)f;
    cumulative_injection_delta[0] += (k_new - k_old_val);
  }
}
