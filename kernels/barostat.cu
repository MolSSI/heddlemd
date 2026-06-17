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
// Reads `factor` from a 1-element device buffer and applies a uniform
// per-particle position rescale. Same shape as `rescale_positions` but
// with no host scalar argument: used when the rescale factor is
// computed on device (e.g. by `c_rescale_compute_mu_and_rescale_lattice`
// or `berendsen_compute_mu_and_rescale_lattice`) and never copied to
// the host.
extern "C" __global__ void rescale_positions_device_factor(
    Real *positions_x,
    Real *positions_y,
    Real *positions_z,
    const Real *factor,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  Real f = factor[0];
  positions_x[i] *= f;
  positions_y[i] *= f;
  positions_z[i] *= f;
}

// Multiply every component of the six-element device-resident lattice
// buffer by a host-supplied scalar `factor`. Single-thread kernel
// (six in-place scalar multiplies); the host validates `factor`
// before launch. Bumps the box's generation counter on the host side
// (`SimulationBox::multiply_lattice_isotropic`); the kernel itself
// only mutates the lattice buffer.
extern "C" __global__ void multiply_lattice_isotropic(
    Real *lattice,
    Real factor)
{
  if (threadIdx.x != 0u || blockIdx.x != 0u) {
    return;
  }
  for (int i = 0; i < 6; ++i) {
    lattice[i] *= factor;
  }
}

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
    Real *lattice,                              // length 6, mutated in place
    Real *mu_out,                               // length 1
    double *diagnostics,                        // length 3: [P, V_post, injection_delta]
    unsigned long long *draw_counter,           // length 1; read at entry, ++ at exit
    unsigned int seed_lo,
    unsigned int seed_hi,
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
  unsigned long long counter = *draw_counter;
  unsigned int draw_counter_lo = (unsigned int)(counter & 0xFFFFFFFFULL);
  unsigned int draw_counter_hi = (unsigned int)(counter >> 32);

  double lx_d = (double)lattice[0];
  double ly_d = (double)lattice[1];
  double lz_d = (double)lattice[2];
  double v_pre = lx_d * ly_d * lz_d;

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

  mu_out[0] = (Real)mu;

  // Mutate the device lattice in place: every component scales by µ.
  Real mu_real = (Real)mu;
  for (int i = 0; i < 6; ++i) {
    lattice[i] *= mu_real;
  }

  diagnostics[0] = pressure;
  diagnostics[1] = v_post;
  diagnostics[2] += injection_delta;
  *draw_counter = counter + 1ULL;
}

// Berendsen barostat: deterministic isotropic rescale.
//
// Mirrors the c_rescale kernel shape (single thread, reads K/W and the
// device lattice, mutates lattice + writes µ + diagnostics) but drops
// the stochastic noise term. Used by the Berendsen barostat slot.
//
// `diagnostics` has length 2: [pressure, v_post]; Berendsen publishes
// no conserved-quantity column so there's no cumulative-injection
// delta slot.
extern "C" __global__ void berendsen_compute_mu(
    const Real *k,                              // length 1
    const Real *w,                              // length 1
    Real *lattice,                              // length 6, mutated in place
    Real *mu_out,                               // length 1
    double *diagnostics,                        // length 2: [P, V_post]
    double pressure_target,
    double tau,
    double compressibility,
    double dt,
    double mu_cubed_min)
{
  if (threadIdx.x != 0u || blockIdx.x != 0u) {
    return;
  }
  double lx_d = (double)lattice[0];
  double ly_d = (double)lattice[1];
  double lz_d = (double)lattice[2];
  double v_pre = lx_d * ly_d * lz_d;

  double k_val = (double)k[0];
  double w_val = (double)w[0];
  double pressure = (2.0 * k_val + w_val) / (3.0 * v_pre);

  double deterministic =
      -compressibility * (dt / tau) * (pressure_target - pressure);
  double mu_cubed = 1.0 + deterministic;
  double mu_cubed_clamped = mu_cubed > mu_cubed_min ? mu_cubed : mu_cubed_min;
  double mu = cbrt(mu_cubed_clamped);

  double v_post = v_pre * mu_cubed_clamped;

  mu_out[0] = (Real)mu;

  Real mu_real = (Real)mu;
  for (int i = 0; i < 6; ++i) {
    lattice[i] *= mu_real;
  }

  diagnostics[0] = pressure;
  diagnostics[1] = v_post;
}
