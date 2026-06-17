// rq-0d8c8688

// Single-block deterministic virial-sum reduction. Mirrors
// `kinetic_energy_reduce` in nose_hoover.cu: one block of 256 threads,
// strided per-thread accumulator, deterministic left-to-right pairwise
// tree in shared memory. Two runs with byte-identical inputs on the
// same GPU produce a byte-identical `partial_out[0]`.
#include "precision.cuh"
#include "philox.cuh"

extern "C" __global__ void virial_sum_reduce(
    const Real *virials,
    Real *partial_out,    // length 1; only thread 0 writes
    unsigned int n)
{
  __shared__ Real partial[256];

  unsigned int tid = threadIdx.x;
  Real sum = R(0.0);
  for (unsigned int i = tid; i < n; i += blockDim.x) {
    sum += virials[i];
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

// Uniform per-particle position rescale: x_i ← factor · x_i for every
// component of every particle. One thread per particle, no inter-thread
// interaction. Does NOT touch velocities, forces, image flags, or any
// neighbor-list reference positions; fractional coordinates are
// invariant under uniform scaling so image flags carry over unchanged,
// and the neighbor list refreshes its reference positions on the next
// `force_field.step` via the box-generation change-detection path.
extern "C" __global__ void rescale_positions(
    Real *positions_x,
    Real *positions_y,
    Real *positions_z,
    Real factor,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  positions_x[i] *= factor;
  positions_y[i] *= factor;
  positions_z[i] *= factor;
}

// C-rescale barostat: compute the box scale factor µ for one step
// entirely on device, plus the diagnostic pressure (for log columns)
// and the conserved-quantity injection delta.
//
// Reads `k[0]` (the kinetic energy from `kinetic_energy_reduce`) and
// `w[0]` (the total scalar virial from `virial_sum_reduce`), draws one
// standard-normal sample from Philox-4×32-10, and evaluates
//
//   P     = (2 k + w) / (3 v_pre)
//   µ³    = 1 + κ · (dt / τ) · (P - P_target) + √(2 κ kT dt / (τ v_pre)) · r
//   µ     = ∛max(µ³, µ_min)
//
// in double precision. The host downloads `mu_pressure_out` (two Real
// values: [µ, P]) as a single 2-element dtoh, calls
// `sim_box.rescale_isotropic(µ)`, and launches `rescale_positions`.
// The cumulative conserved-quantity injection
// `P_target · (v_post - v_pre) = P_target · v_pre · (µ³ - 1)` is
// accumulated into `cumulative_injection_delta[0]` on device; the host
// drains it at log-write cadence (same pattern as CSVR's
// `cumulative_injection_delta`).
//
// Determinism: single-thread; reads from device buffers; identical
// Philox counter sequence to the legacy host-side implementation. Two
// runs on the same GPU produce a byte-identical (µ, P, injection_delta)
// triple.
extern "C" __global__ void c_rescale_compute_mu(
    const Real *k,                              // length 1
    const Real *w,                              // length 1
    Real *mu_pressure_out,                      // length 2: [mu, pressure]
    double *cumulative_injection_delta,         // length 1
    unsigned int seed_lo,
    unsigned int seed_hi,
    unsigned int draw_counter_lo,
    unsigned int draw_counter_hi,
    double v_pre,
    double pressure_target,
    double tau,
    double compressibility,
    double kt,
    double dt,
    double mu_cubed_min)                        // MU_MIN³ from host
{
  if (threadIdx.x != 0u || blockIdx.x != 0u) {
    return;
  }
  double k_val = (double)k[0];
  double w_val = (double)w[0];
  double pressure = (2.0 * k_val + w_val) / (3.0 * v_pre);

  double r = philox_gaussian_f64(
      seed_lo, seed_hi, draw_counter_lo, draw_counter_hi, 0u, 0u);

  double deterministic =
      -compressibility * (dt / tau) * (pressure_target - pressure);
  double noise_amplitude = sqrt(2.0 * compressibility * kt * dt / (tau * v_pre));
  double mu_cubed = 1.0 + deterministic + noise_amplitude * r;
  double mu_cubed_clamped = mu_cubed > mu_cubed_min ? mu_cubed : mu_cubed_min;
  double mu = cbrt(mu_cubed_clamped);

  double v_post = v_pre * mu_cubed_clamped;
  double injection_delta = pressure_target * (v_post - v_pre);

  mu_pressure_out[0] = (Real)mu;
  mu_pressure_out[1] = (Real)pressure;
  cumulative_injection_delta[0] += injection_delta;
}
