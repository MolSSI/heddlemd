// rq-3b6d5001

// Cell-coupled velocity half-kick for the MTK NPT integrator
// (isotropic). One thread per particle, no inter-thread interaction.
// The host pre-computes the two scalars in f64 then downcasts to f32:
//   exp_minus_alpha = exp(-α · dt/2)
//   phi_v_dt_half   = (dt/2) · Φ_v · exp(-α · dt/4)
// where α = (1 + 3/N_f) · (p_eps / W) and Φ_v = sinh(α·dt/4)/(α·dt/4)
// (with a host-side Taylor fallback near α ≈ 0). Implements the
// closed-form solution of dv/dt = F/m - α · v over a half-step dt/2.
#include "precision.cuh"

extern "C" __global__ void mtk_velocity_half_kick(
    Real *velocities_x,
    Real *velocities_y,
    Real *velocities_z,
    const Real *forces_x,
    const Real *forces_y,
    const Real *forces_z,
    const Real *masses,
    Real exp_minus_alpha,
    Real phi_v_dt_half,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  Real inv_m = R(1.0) / masses[i];
  velocities_x[i] = exp_minus_alpha * velocities_x[i]
                  + phi_v_dt_half * (forces_x[i] * inv_m);
  velocities_y[i] = exp_minus_alpha * velocities_y[i]
                  + phi_v_dt_half * (forces_y[i] * inv_m);
  velocities_z[i] = exp_minus_alpha * velocities_z[i]
                  + phi_v_dt_half * (forces_z[i] * inv_m);
}

// Cell-coupled position drift for the MTK NPT integrator (isotropic).
// One thread per particle, no inter-thread interaction. The host
// pre-computes the two scalars in f64 then downcasts to f32:
//   exp_b_dt  = exp(β · dt)
//   phi_x_dt  = dt · Φ_x · exp(β · dt/2)
// where β = p_eps / W and Φ_x = sinh(β·dt/2)/(β·dt/2) (with a
// host-side Taylor fallback near β ≈ 0). Implements the closed-form
// solution of dx/dt = v + β · x over a full step dt.
//
// Does NOT update image flags or wrap positions: under uniform
// isotropic scaling fractional coordinates are invariant, so image
// flags carry over unchanged. The neighbor list refreshes its
// reference positions on the next force_field.step via the
// box-generation change-detection path.
extern "C" __global__ void mtk_position_drift(
    Real *positions_x,
    Real *positions_y,
    Real *positions_z,
    const Real *velocities_x,
    const Real *velocities_y,
    const Real *velocities_z,
    Real exp_b_dt,
    Real phi_x_dt,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  positions_x[i] = exp_b_dt * positions_x[i] + phi_x_dt * velocities_x[i];
  positions_y[i] = exp_b_dt * positions_y[i] + phi_x_dt * velocities_y[i];
  positions_z[i] = exp_b_dt * positions_z[i] + phi_x_dt * velocities_z[i];
}
