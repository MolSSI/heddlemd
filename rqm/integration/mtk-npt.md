# Feature: Martyna-Tobias-Klein NPT Integrator (Isotropic) <!-- rq-3b6d5001 -->

The Martyna-Tobias-Klein NPT integrator (Martyna, Tobias, Klein,
*J. Chem. Phys.* **101**, 4177 (1994); Martyna, Tuckerman, Tobias,
Klein, *Mol. Phys.* **87**, 1117 (1996)) is a deterministic
extended-system NPT scheme that samples the canonical isothermal-
isobaric ensemble exactly. One of the pluggable integrator slots
(see `framework.md`); selected by `kind = "mtk-npt"` in the config's
`[integrator]` section.

The integrator is **fused**: it owns its own thermostat (a
Nosé-Hoover chain on the particle kinetic energy) **and** its own
barostat (an extended-system cell with mass `W` and a separate
Nosé-Hoover chain on the cell kinetic energy). It rejects
co-configured `[thermostat]` and `[barostat]` tables: the
`"mtk-npt"` builder's
`IntegratorBuilder::owns_thermostat(&params)` and
`IntegratorBuilder::owns_barostat(&params)` both return `true`.

This file documents the isotropic variant: one cell-volume degree of
freedom (the logarithmic volume `ε = (1/3) ln(V/V_0)`), one cell
momentum `p_ε` conjugate to `ε`, and one cell mass `W`. Anisotropic
(flexible-cell) MTK is out of scope; see *Out of Scope* below.

The integrator preserves centre-of-mass momentum exactly (the cell
rescale is uniform about the box origin and the particle-chain
velocity rescale is multiplicative) and carries no RNG state, so it
produces byte-identical trajectories across runs on the same GPU.

## Algorithm <!-- rq-e43528db -->

### Extended-system state <!-- rq-f5d16700 -->

In addition to the physical state (particle positions `{x_i}`,
velocities `{v_i}`, and the `SimulationBox` lattice), the integrator
carries an extended set of host-side scalar DOFs:

- `p_ε: f64` — cell momentum (single scalar; isotropic). Initialised
  to `0.0`.
- `W: f64` — cell mass. Precomputed at construction as
  `W = (N_f + 3) · k_B · T · τ_P²` (Martyna-Tobias-Klein 1994,
  Eq. 4.8) where `N_f = max(1, 3·N − n_constraints − 3)` is the
  number of thermostatted DOFs (the same constraint- and
  COM-removed convention used by CSVR — see `csvr.md` — and by
  `compute_temperature` in `io/log-output.md`) and `τ_P` is the
  user-supplied barostat coupling time. Never updated during the
  run.
- Particle thermostat chain `{ξ_j, p_ξ_j, Q_j}` for `j = 1..M` — a
  standard NHC chain identical in shape to `nose-hoover-chain.md`,
  with `Q_1 = N_f · k_B · T · τ_T²` and `Q_j = k_B · T · τ_T²` for
  `j > 1` (`τ_T` is the user-supplied thermostat coupling time). The
  chain dynamics use the same MKT splitting and the same shared
  `nhc_chain_sub_step` host-side helper from
  `nose-hoover-chain.md`.
- Cell thermostat chain `{ξ'_j, p_ξ'_j, Q'_j}` for `j = 1..M` — a
  second NHC chain thermostatting the cell kinetic energy
  `K_cell = (1/2) · p_ε² / W`. Chain mass `Q'_j = k_B · T · τ_T²`
  for all `j` (a 1-DOF chain has no `g`-prefactor on `Q'_1`). Uses
  the same shared `nhc_chain_sub_step` helper.

`T` is the user-supplied target temperature; `P_ext` is the
user-supplied target pressure; `k_B = 1.380649 × 10⁻²³ J/K`.

### Equations of motion <!-- rq-d93f05a5 -->

The continuous MTK equations of motion (isotropic) are

```text
ẋ_i  = v_i + (p_ε / W) · x_i
v̇_i  = F_i / m_i − ((1 + 3/N_f) · (p_ε / W) + p_ξ_1 / Q_1) · v_i
ε̇   = p_ε / W
ṗ_ε  = 3 · V · (P − P_ext) + (3/N_f) · 2K − (p_ξ'_1 / Q'_1) · p_ε
ξ̇_j  = p_ξ_j / Q_j                              (j = 1..M)
ṗ_ξ_1 = (2K − N_f · k_B · T) − (p_ξ_2/Q_2) · p_ξ_1
ṗ_ξ_j = (p_ξ_{j-1}²/Q_{j-1} − k_B · T) − (p_ξ_{j+1}/Q_{j+1}) · p_ξ_j
ξ̇'_j  = p_ξ'_j / Q'_j                           (j = 1..M)
ṗ_ξ'_1 = (p_ε²/W − k_B · T) − (p_ξ'_2/Q'_2) · p_ξ'_1
ṗ_ξ'_j = (p_ξ'_{j-1}²/Q'_{j-1} − k_B · T) − (p_ξ'_{j+1}/Q'_{j+1}) · p_ξ'_j
```

`K = (1/2) Σ_i m_i |v_i|²` is the instantaneous particle kinetic
energy; `V = sim_box.volume()` is the instantaneous box volume;
`P = (2K + W_virial) / (3V)` is the instantaneous pressure, where
`W_virial = Σ_i buffers.virials[i]` is the scalar virial.

### Reversible symmetric Trotter splitting <!-- rq-f7bd47f7 -->

A single timestep `dt` is propagated as the symmetric product
(Martyna-Tuckerman-Tobias-Klein 1996, Eqs. 2.11–2.15):

```text
e^(L · dt) ≈
    e^(L_chains · dt/2)
    · e^(L_baro_kick · dt/2)
    · e^(L_kick · dt/2)
    · e^(L_drift · dt)
    · e^(L_force_eval)
    · e^(L_kick · dt/2)
    · e^(L_baro_kick · dt/2)
    · e^(L_chains · dt/2)
```

The five operators:

- `L_chains`: cell-chain half-step + particle-chain half-step
  (independent chains; the particle chain reads `K` after the
  cell-coupled velocity terms in `L_kick` have already rescaled
  velocities; the cell chain reads `p_ε`).
- `L_baro_kick`: `p_ε ← p_ε + (dt/2) · (3 · V · (P − P_ext) +
  (3/N_f) · 2K)`. Uses the freshly computed `P` and `K`.
- `L_kick`: `v_i ← exp(−(1 + 3/N_f) · (p_ε / W) · (dt/2)) · v_i +
  (F_i / m_i) · (dt/2) · Φ_v` where `Φ_v = sinh(α_v) / α_v` with
  `α_v = (1 + 3/N_f) · (p_ε / W) · (dt/4)`. Closed-form solution
  to the velocity ODE under constant `p_ε`, `F`. When
  `|α_v| < 1e-6` the host substitutes the Taylor series
  `Φ_v ≈ 1 + α_v²/6` to retain `f64` precision near the
  small-cell-velocity limit.
- `L_drift`: `x_i ← exp((p_ε / W) · dt) · x_i + v_i · dt · Φ_x ·
  exp((p_ε / W) · (dt/2))` where `Φ_x = sinh(α_x) / α_x` with
  `α_x = (p_ε / W) · (dt/2)`. Same small-`α_x` Taylor expansion
  applies. Simultaneously: `ε ← ε + (p_ε / W) · dt`, i.e.
  `V_new = V_old · exp(3 · (p_ε / W) · dt)` and the box is
  rescaled by `μ_box = (V_new / V_old)^(1/3) = exp((p_ε / W) · dt)`
  via `sim_box.rescale_isotropic(μ_box)`.
- `L_force_eval`: `force_field.step(...)` recomputes `F` and the
  virials at the new positions and new box.

Each `L_chains` half-step splits internally into Yoshida-Suzuki
sub-steps, identical in structure to the NHC chain half-step
documented in `nose-hoover-chain.md`. The two chains advance
sequentially within each half-step: cell chain first, then
particle chain. Each chain's sub-step uses the shared
`nhc_chain_sub_step` host-side helper.

### Step Plan <!-- rq-971ca980 -->

`MtkNptIntegrator::plan(dt)` returns the fourteen-element MTK
symmetric Trotter sequence the runner walks:

```rust
StepPlan { steps: vec![
    SubStep::Custom   { label: "ke_reduce_pre"      },  // 1: KE reduce
    SubStep::Custom   { label: "vir_reduce_pre"     },  // 2: Virial reduce
    SubStep::Custom   { label: "cell_chain_pre"     },  // 3: Cell chain ½ (host)
    SubStep::Custom   { label: "particle_chain_pre" },  // 4: Particle chain ½
    SubStep::Custom   { label: "baro_kick_pre"      },  // 5: Baro kick ½ (host)
    SubStep::KickHalf { dt, label: "vel_kick_pre"   },  // 6: Cell-coupled vel kick ½
    SubStep::Drift    { dt, label: "drift_box"      },  // 7: Drift + box rescale
    SubStep::ForceEval,                                  // 8: Force eval
    SubStep::Custom   { label: "ke_reduce_post"     },  // 9: KE reduce
    SubStep::Custom   { label: "vir_reduce_post"    },  // 10: Virial reduce
    SubStep::KickHalf { dt, label: "vel_kick_post"  },  // 11: Cell-coupled vel kick ½
    SubStep::Custom   { label: "baro_kick_post"     },  // 12: Baro kick ½ (host)
    SubStep::Custom   { label: "particle_chain_post"},  // 13: Particle chain ½
    SubStep::Custom   { label: "cell_chain_post"    },  // 14: Cell chain ½ (host)
]}
```

`MtkNptIntegrator::execute(sub, ...)` dispatches each non-`ForceEval`
sub-step to the appropriate kernel sequence:

| Order | Sub-step variant     | Label                | Kernel / call                  | Stage label                   |
| ----- | -------------------- | -------------------- | ------------------------------ | ----------------------------- |
| 1     | `Custom`             | `ke_reduce_pre`      | `kinetic_energy_reduce`        | `KineticEnergyReduce`         |
| 2     | `Custom`             | `vir_reduce_pre`     | `virial_sum_reduce`            | `VirialSumReduce`             |
| 3     | `Custom`             | `cell_chain_pre`     | host arithmetic on `p_ε`       | —                             |
| 4     | `Custom`             | `particle_chain_pre` | `rescale_velocities` × N_sub   | `MtkNptRescaleVelocities`     |
| 5     | `Custom`             | `baro_kick_pre`      | host arithmetic on `p_ε`       | —                             |
| 6     | `KickHalf`           | `vel_kick_pre`       | `mtk_velocity_half_kick`       | `MtkNptVelocityHalfKick`      |
| 7     | `Drift`              | `drift_box`          | `mtk_position_drift` + `sim_box.rescale_isotropic` | `MtkNptPositionDrift` |
| 8     | (`ForceEval`)        |                      | force pipeline (runner)        | (force-pipeline stages)       |
| 9     | `Custom`             | `ke_reduce_post`     | `kinetic_energy_reduce`        | `KineticEnergyReduce`         |
| 10    | `Custom`             | `vir_reduce_post`    | `virial_sum_reduce`            | `VirialSumReduce`             |
| 11    | `KickHalf`           | `vel_kick_post`      | `mtk_velocity_half_kick`       | `MtkNptVelocityHalfKick`      |
| 12    | `Custom`             | `baro_kick_post`     | host arithmetic on `p_ε`       | —                             |
| 13    | `Custom`             | `particle_chain_post`| `rescale_velocities` × N_sub   | `MtkNptRescaleVelocities`     |
| 14    | `Custom`             | `cell_chain_post`    | host arithmetic on `p_ε`       | —                             |

`a = (1 + 3/N_f) · (p_ε / W)` and `b = (p_ε / W)` are scalars
recomputed on the host from the current `p_ε` inside the appropriate
`execute()` call before each kernel launch and passed as kernel
arguments. The intermediate scalars (`K`, `W_virial`, `pressure`,
`a`, `b`, `μ_box`, etc.) flow between sub-steps through the
integrator's `&mut self` state.

`N_sub = n_yoshida · n_resp` (matches `nose-hoover-chain.md`). The
particle-chain Custom sub-steps each launch `N_sub`
`rescale_velocities` kernel calls (one per Yoshida sub-step), each
labelled `MtkNptRescaleVelocities`. The cell-chain Custom sub-steps
are pure host arithmetic on `p_ε`; no kernel launches.

`mtk_velocity_half_kick` and `mtk_position_drift` are the two CUDA
kernels owned by this slot. They replace `vv_kick` / `vv_kick_drift`
for the cell-coupled `L_kick` and `L_drift` operators; the standard
VV kernels are not used by this integrator.

The `"mtk-npt"` builder's
`IntegratorBuilder::supports_constraints(&params)` returns `false`
regardless of params; the runner therefore inserts no constraint
hooks around the `Drift` or `KickHalf` sub-steps. Composing MTK NPT
with constraints is rejected at config load by
`ConfigError::IncompatibleConstraint` (see
`integration/constraint-framework.md`).

## Parameters <!-- rq-ce37404c -->

The `"mtk-npt"` builder deserialises `MtkNptParams` (with the fields
listed below) from the `[integrator]` section's `SlotConfig::params`
field:

- `temperature: f64` — bath temperature `T` in kelvin. Required.
  Finite and strictly positive. Independent of
  `simulation.temperature` (which seeds the initial Maxwell-Boltzmann
  sampler).
- `pressure: f64` — target pressure `P_ext` in pascals (Pa).
  Required. Finite. May be any sign or zero.
- `tau_t: f64` — thermostat coupling time in seconds. Required.
  Finite and strictly positive. Controls both the particle-chain
  and cell-chain masses. Typical values for liquid water are 50–100
  fs.
- `tau_p: f64` — barostat coupling time in seconds. Required. Finite
  and strictly positive. Controls the cell mass `W`. Typical values
  for liquid water are 1–5 ps (10–50× `tau_t` so the barostat
  responds slowly relative to the thermostat).
- `chain_length: u32` — number of chain elements `M` (shared by
  both the particle chain and the cell chain). Optional; defaults
  to `3`. Must be `≥ 1`.
- `yoshida_order: u32` — Suzuki-Yoshida sub-step count per chain
  half-step (shared by both chains). Optional; defaults to `3`.
  Accepted values: `1`, `3`, `5`, `7` (same set as
  `nose-hoover-chain.md`).
- `n_resp: u32` — chain RESP sub-cycle count (shared by both
  chains). Optional; defaults to `1`. Must be `≥ 1`.

No RNG seed: the MTK integrator is deterministic.

No `compressibility` parameter: in MTK the cell response timescale
is set by `τ_P` (and the derived cell mass `W`), not by an
isothermal compressibility estimate.

## MTK conserved Hamiltonian <!-- rq-2e284f4b -->

The extended Hamiltonian conserved by the continuous MTK equations
(isotropic) is

```text
H_MTK = K + U + P_ext · V
        + (1/2) · p_ε² / W
        + Σ_{j=1..M} p_ξ_j²  / (2 Q_j)
        + Σ_{j=1..M} p_ξ'_j² / (2 Q'_j)
        + N_f · k_B · T · ξ_1
        + k_B · T · Σ_{j=2..M} ξ_j
        + k_B · T · Σ_{j=1..M} ξ'_j
```

`H_MTK` is invariant under the exact MTK dynamics; under the
discretised reversible Trotter splitting documented above it drifts
by `O(dt²)` per step. Drift in `H_MTK` over a run is the canonical
correctness diagnostic for an MTK implementation. Exposed as the
per-log-row diagnostic column `mtk_npt_conserved` (see
`io/log-output.md`).

The integrator additionally exposes `pressure` (in Pa) and
`box_volume` (in m³) log columns, matching the Berendsen barostat
convention; the value reported for `pressure` is the instantaneous
`P` from step 10 of the per-step sequence (the post-step value used
by the closing chain half-step), and `box_volume` is the
post-step `V`.

## Empty State and degenerate cases <!-- rq-0c7a20a1 -->

- `buffers.particle_count() == 0`: `step()` returns `Ok(())` without
  launching any kernel, without mutating `sim_box`, and without
  advancing any chain state. `K`, `W_virial`, and `P` are not
  computed; the diagnostic columns report `0.0`, `sim_box.volume()`,
  and the chain-only contribution to `H_MTK`.
- `particle_count == 1` with `n_constraints == 0`:
  `N_f = max(1, 3 − 0 − 3) = 1`. The chain masses use `N_f = 1`.
  The integrator is mathematically well-defined but produces a
  one-thermostatted-DOF system with little physical relevance.
- Heavily-constrained systems where `3·N − n_constraints − 3 <= 0`:
  the `max(1, …)` floor keeps `N_f = 1`. Users should not pair MTK
  with such systems.
- `M == 1`: vanilla single-DOF Nosé-Hoover (no chain). The
  outermost chain-momentum kicks documented in the
  `nose-hoover-chain.md` *Chain sub-step* are skipped on both
  chains. The `nhc_chain_sub_step` shared helper handles `M == 1`
  uniformly.
- `p_ε ≈ 0` (e.g. start-of-run): the kick / drift kernels'
  `sinh(α)/α` factor is computed via the host-side Taylor expansion
  `1 + α²/6` when `|α| < 1.0e-6`, guarding against `f64`
  cancellation in the `sinh(α)/α` formula. Beyond `|α| = 1.0e-6`
  the closed-form `sinh(α)/α` is well-conditioned in `f64`.
- `V == 0`: unreachable; the existing `SimulationBox` constructor
  rejects zero-volume boxes and `rescale_isotropic` rejects
  non-positive factors.

## Feature API <!-- rq-d77527e8 -->

### Types <!-- rq-047f97da -->

- `MtkNptIntegrator` — implements the `Integrator` trait declared <!-- rq-508680c7 -->
  in `framework.md`. Registered in
  `IntegratorRegistry::with_builtins` under
  `kind_name() == "mtk-npt"`. Fields:

  - `device: Arc<CudaDevice>`
  - `temperature: f64` — `T`.
  - `pressure: f64` — `P_ext`.
  - `tau_t: f64` — particle chain coupling time.
  - `tau_p: f64` — barostat coupling time.
  - `chain_length: u32` — `M`.
  - `yoshida_order: u32`
  - `n_resp: u32`
  - `yoshida_weights: &'static [f64]` — pre-resolved Suzuki-Yoshida
    weights.
  - `g_dof: u32` — `max(1, 3 · particle_count − n_constraints − 3)`,
    computed at construction from the `n_constraints` parameter
    passed by the runner.
  - `kt: f64` — `BOLTZMANN_J_PER_K · temperature`.
  - `w_cell: f64` — `(g_dof + 3) · kt · τ_p²`.
  - `p_eps: f64` — cell momentum. Initialised to `0.0`.
  - `eps: f64` — `(1/3) · ln(V / V_0)`, tracked for the conserved
    Hamiltonian's bookkeeping. Initialised to `0.0` at construction
    (using the initial box volume as `V_0`).
  - `q_mass_part: Vec<f64>` — particle chain masses, length `M`.
    Element 0 is `g_dof · kt · τ_t²`; elements `1..M` are
    `kt · τ_t²`.
  - `xi_part: Vec<f64>`, `p_xi_part: Vec<f64>` — particle chain
    positions and momenta, length `M`. Initialised to `0.0`.
  - `q_mass_cell: Vec<f64>` — cell chain masses, length `M`. All
    elements are `kt · τ_t²` (1-DOF chain; no `g`-prefactor on
    element 0).
  - `xi_cell: Vec<f64>`, `p_xi_cell: Vec<f64>` — cell chain
    positions and momenta, length `M`. Initialised to `0.0`.
  - `ke_scratch: CudaSlice<f32>` — length-1 device buffer for
    kinetic-energy reductions; reused across calls.
  - `virial_scratch: CudaSlice<f32>` — length-1 device buffer for
    virial reductions; reused across calls.
  - `most_recent_pressure: f64`, `most_recent_volume: f64`,
    `most_recent_ke: f64` — cached for `log_column_values`.

  All fields private; the slot's public surface is the `Integrator`
  trait methods and construction via `MtkNptBuilder`. The chain
  state, `p_eps`, and `eps` fields are public for parity with other
  slots' state so a future restart-from-checkpoint flow can restore
  them explicitly.

- `MtkNptBuilder` — implements `IntegratorBuilder` with <!-- rq-0b7f7023 -->
  `kind_name() == "mtk-npt"`. `validate_params(&params)`
  deserialises a `MtkNptParams` struct from `params` and enforces
  per-field domains (positivity, allowed `yoshida_order`, etc.) plus
  default values for the optional fields. `build(gpu, particle_count,
  &params)` re-deserialises `MtkNptParams`, allocates the two
  length-1 device scratch buffers, precomputes `g_dof`, `kt`,
  `w_cell`, `q_mass_part`, `q_mass_cell`, and the Suzuki-Yoshida
  weights, initialises every chain DOF to `0.0`, and returns the
  boxed `MtkNptIntegrator`.

### `Integrator` trait overrides <!-- rq-d2d0fb5f -->

- `plan(dt)` — returns the fourteen-element `StepPlan` defined in <!-- rq-8cda2c89 -->
  *Step Plan* above. Pure; reads only the integrator's static
  configuration.
- `execute(sub, buffers, sim_box, timings)` — dispatches each <!-- rq-4c21c386 -->
  non-`ForceEval` sub-step to the kernel sequence enumerated in the
  per-step kernel table. The initial `V_0` for the `eps` bookkeeping
  is captured from `sim_box.volume()` on the very first `execute()`
  call of the `vel_kick_pre` sub-step (when `eps == 0.0` and `p_eps
  == 0.0`). Subsequent calls evolve `eps` according to the `L_drift`
  operator inside the `drift_box` sub-step.
- `log_column_names() -> &'static ["pressure", "box_volume", <!-- rq-14a7685e -->
  "mtk_npt_conserved"]`.
- `log_column_values(ke, pe) -> vec![most_recent_pressure, <!-- rq-f9ebe53f -->
  most_recent_volume, H_MTK]` where `H_MTK` is computed from the
  formula in *MTK conserved Hamiltonian* above using the runner's
  supplied `ke` and `pe`, the cached `most_recent_volume`, the
  current `p_eps` and `w_cell`, and the current chain state.

### CUDA Kernels <!-- rq-0f0412fd -->

`kernels/mtk.cu` declares two `extern "C"` kernels:

```c
extern "C" __global__ void mtk_velocity_half_kick(
    float *velocities_x, float *velocities_y, float *velocities_z,
    const float *forces_x, const float *forces_y, const float *forces_z,
    const float *masses,
    float exp_minus_alpha,  // exp(−a · dt/2)         (host, f32)
    float phi_v_dt_half,    // (dt/2) · Φ_v · exp(−a · dt/4)  (host, f32)
    unsigned int n);

extern "C" __global__ void mtk_position_drift(
    float *positions_x, float *positions_y, float *positions_z,
    const float *velocities_x, const float *velocities_y, const float *velocities_z,
    float exp_b_dt,         // exp(b · dt)            (host, f32)
    float phi_x_dt,         // dt · Φ_x · exp(b · dt/2)  (host, f32)
    unsigned int n);
```

#### `mtk_velocity_half_kick` <!-- rq-20464340 -->

One thread per particle. Implements the closed-form solution of
`dv/dt = F/m − a · v` over a half-step `dt/2`:

```c
velocities_x[i] = exp_minus_alpha * velocities_x[i]
                + phi_v_dt_half * (forces_x[i] / masses[i]);
```

and analogously for y, z. The host computes `exp_minus_alpha` and
`phi_v_dt_half` in `f64` and casts to `f32` before launch. No
inter-thread interaction; trivially deterministic. Block size 256,
grid `ceil(n / 256)`.

#### `mtk_position_drift` <!-- rq-23e2bb88 -->

One thread per particle. Implements the closed-form solution of
`dx/dt = v + b · x` over the full step `dt`:

```c
positions_x[i] = exp_b_dt * positions_x[i]
               + phi_x_dt * velocities_x[i];
```

and analogously for y, z. The host computes `exp_b_dt` and
`phi_x_dt` in `f64` and casts to `f32` before launch. No
inter-thread interaction; trivially deterministic. Block size 256,
grid `ceil(n / 256)`. Does NOT update image flags or wrap
positions: under uniform isotropic scaling fractional coordinates
are invariant, so image flags carry over unchanged. (Particle
positions evolve under both the cell-coupled rescale `exp(b · dt)`
and the velocity-driven drift `v · dt · Φ_x`; the latter can cause
particles to cross image boundaries when `v` is large enough, but
the existing rule "no PBC wrap inside integrator kernels; wrap
happens via the next force pipeline's neighbor-list refresh" is
preserved.)

### Shared `nhc_chain_sub_step` host-side helper <!-- rq-2a749645 -->

The MTK integrator and the NHC thermostat both perform identical
Yoshida-Suzuki chain sub-steps. The host-side chain math is exposed
as a pure helper function in `src/integrator/nose_hoover_chain.rs`,
which the MTK integrator imports:

```rust
pub fn nhc_chain_sub_step(
    xi: &mut [f64],
    p_xi: &mut [f64],
    q_mass: &[f64],
    dt: f64,
    k_thermalized: f64,    // 2K for an N_f-DOF chain; p_eps²/W for the cell chain
    g_dof: f64,            // N_f for the particle chain; 1.0 for the cell chain
    kt: f64,
) -> f64
```

Returns the multiplicative velocity-rescale factor that the caller
must apply to the chain's thermalized DOF (the particle chain
applies it via `rescale_velocities`; the MTK cell chain applies it
to `p_eps` host-side).

See `nose-hoover-chain.md` for the algorithmic details; this helper
is its single canonical implementation site.

### PTX Module Loading <!-- rq-f9fc04a1 -->

`init_device()` loads the compiled `kernels/mtk.cu` PTX as module
`"mtk"` and captures `mtk_velocity_half_kick` and
`mtk_position_drift` into the `Kernels` handle (see
`build-pipeline.md`).

### Rust Launch Helpers <!-- rq-7c6012c7 -->

Two free functions in `src/gpu/kernels.rs`, re-exported from
`crate::gpu`:

- `mtk_velocity_half_kick(buffers: &mut ParticleBuffers, exp_minus_alpha: f32, phi_v_dt_half: f32) -> Result<(), GpuError>` <!-- rq-cadfb824 -->
  - Launches `mtk_velocity_half_kick` with the two scalar arguments
    pre-computed on the host.
  - Block size 256, grid `ceil(n / 256)`.
  - When `buffers.particle_count() == 0` returns `Ok(())` without
    launching.
- `mtk_position_drift(buffers: &mut ParticleBuffers, exp_b_dt: f32, phi_x_dt: f32) -> Result<(), GpuError>` <!-- rq-f1c96a3f -->
  - Launches `mtk_position_drift` with the two scalar arguments
    pre-computed on the host.
  - Block size 256, grid `ceil(n / 256)`.
  - When `buffers.particle_count() == 0` returns `Ok(())` without
    launching.

## Launch Configuration <!-- rq-8f121de5 -->

Per-step launch counts (for `M = 3, n_yoshida = 3, n_resp = 1`,
which is `N_sub = 3`):

- `kinetic_energy_reduce`: 2 launches (one per KE refresh).
- `virial_sum_reduce`: 2 launches.
- `rescale_velocities`: `2 · N_sub = 6` launches (particle chain
  only; the cell chain rescales `p_eps` host-side).
- `mtk_velocity_half_kick`: 2 launches (one per half-kick).
- `mtk_position_drift`: 1 launch.
- Plus the force pipeline's launches.

All launches go through the default stream of
`ParticleBuffers::device`.

## Determinism <!-- rq-7740cfb1 -->

- All MTK chain arithmetic runs in `f64` on the host with a fixed
  arithmetic sequence (delegated to `nhc_chain_sub_step` from
  `nose-hoover-chain.md`).
- `mtk_velocity_half_kick` and `mtk_position_drift` are
  trivially deterministic (one thread per particle, no inter-thread
  interaction). The cell-coupled `exp` and `sinh/α` scalars are
  computed in `f64` and downcast to `f32` for launch.
- `SimulationBox::rescale_isotropic(μ_box)` is a pure deterministic
  multiplication; the generation counter advances monotonically.
- The MTK integrator carries no RNG; there are no stochastic draws
  to randomise.
- Two end-to-end runs with identical configs on the same GPU
  produce byte-identical trajectory and log files, including the
  `pressure`, `box_volume`, and `mtk_npt_conserved` columns.

## Out of Scope <!-- rq-574e3937 -->

- Anisotropic (flexible-cell) MTK. The 6-DOF cell with a 3×3 cell
  momentum requires per-axis virial computation throughout the
  force pipeline (currently the engine computes only the scalar
  virial trace). Would ship as a separate integrator
  (`mtk-npt-flexible`) or as an extension to this slot.
- Semi-isotropic coupling (xy-coupled, z-independent). Out of
  scope for the isotropic slot; would slot in alongside flexible
  cell.
- A lossless `(f32, f64)` compensated mode. The cell-coupled
  velocity and position updates do not have a clean compensated
  form; bit-exact time-reversibility under MTK is not a property
  the algorithm promises (the cell-chain operator is symplectic
  but the standard MKT splitting is not exactly reversible at the
  `f32` storage precision).
- Composition with `[thermostat]` or `[barostat]` slots. The MTK
  integrator owns both; `load_config` rejects co-configured
  `[thermostat]` and `[barostat]` via the existing
  `ConfigError::IncompatibleThermostat` and the new
  `ConfigError::IncompatibleBarostat` cross-validation rules.
- Constraint algorithms (SHAKE/RATTLE) and their interaction with
  the cell-coupled drift. Constraints would need to be re-projected
  after the position update; the framework does not yet ship a
  constraint slot.
- Restart-from-checkpoint of `p_eps`, `eps`, and the two chain
  states. The fields are `pub` for direct host-side assignment by
  future checkpoint code; the config layer carries no syntax for
  them.

---

## Gherkin Scenarios <!-- rq-0b04b35c -->

```gherkin
Feature: MTK NPT integrator (isotropic)

  Background:
    Given a CUDA-capable GPU available as device 0
    And a SimulationBox with lx=ly=lz=1.0e-9 unless otherwise specified
    And init_device() has been called

  # --- Construction ---

  @rq-21e17441
  Scenario: Construct MtkNptIntegrator with defaults (unconstrained system)
    Given an IntegratorKind::MtkNpt {
      temperature: 85.0, pressure: 1.0e5,
      tau_t: 1.0e-13, tau_p: 1.0e-12,
      chain_length: 3, yoshida_order: 3, n_resp: 1 }
    When registry.build(&kind, device, particle_count=4, n_constraints=0) is called
    Then it returns Ok(integrator)
    And the underlying MtkNptIntegrator has chain_length == 3
    And xi_part equals [0.0, 0.0, 0.0]
    And p_xi_part equals [0.0, 0.0, 0.0]
    And xi_cell equals [0.0, 0.0, 0.0]
    And p_xi_cell equals [0.0, 0.0, 0.0]
    And p_eps == 0.0
    And eps == 0.0
    And g_dof equals 9 (max(1, 3*4 − 0 − 3))
    And w_cell equals (g_dof + 3) · k_B · 85.0 · (1.0e-12)²
    And q_mass_part[0] equals g_dof · k_B · 85.0 · (1.0e-13)²

  @rq-0abaa85d
  Scenario: Construct MtkNptIntegrator for a SETTLE'd water system
    Given an IntegratorKind::MtkNpt { temperature: 300.0, pressure: 1.0e5,
      tau_t: 1.0e-13, tau_p: 1.0e-12,
      chain_length: 3, yoshida_order: 3, n_resp: 1 }
    When registry.build(&kind, device, particle_count=24, n_constraints=24) is called
      (8 SETTLE waters: 24 atoms, 3 constraints per molecule)
    Then it returns Ok(integrator)
    And g_dof equals 45 (max(1, 3*24 − 24 − 3))
    And w_cell equals (g_dof + 3) · k_B · 300 · (1.0e-12)²
    And q_mass_part[0] equals g_dof · k_B · 300 · (1.0e-13)²

  @rq-100e0cc8
  Scenario: Construct with chain_length = 1
    Given an IntegratorKind::MtkNpt { ..., chain_length: 1 }
    When registry.build(&kind, device, particle_count=4, n_constraints=0) is called
    Then it returns Ok(integrator)
    And xi_part has length 1
    And xi_cell has length 1

  @rq-7fcfceac
  Scenario: Construct with particle_count = 0
    Given any MtkNpt kind
    When registry.build(..., particle_count=0, n_constraints=0) is called
    Then it returns Ok(integrator)
    And g_dof equals 1 (max(1, 3·0 − 0 − 3))

  # --- Ownership flags ---

  @rq-fecc63ef
  Scenario: MtkNpt owns its own thermostat
    Given an IntegratorKind::MtkNpt { .. }
    Then kind.owns_thermostat() returns true

  @rq-2d46cad5
  Scenario: MtkNpt owns its own barostat
    Given an IntegratorKind::MtkNpt { .. }
    Then kind.owns_barostat() returns true

  # --- Config validation ---

  @rq-ee43e3d6
  Scenario: Reject yoshida_order outside {1, 3, 5, 7}
    Given a config with [integrator] kind="mtk-npt" and yoshida_order=2
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue {
      field: "integrator.yoshida_order", reason: _ })

  @rq-071e19df
  Scenario: Reject non-positive temperature
    Given a config with [integrator] kind="mtk-npt" and temperature=0.0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue {
      field: "integrator.temperature", reason: _ })

  @rq-a003ec43
  Scenario: Reject non-positive tau_t
    Given a config with [integrator] kind="mtk-npt" and tau_t=-1.0e-13
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue {
      field: "integrator.tau_t", reason: _ })

  @rq-775b0833
  Scenario: Reject non-positive tau_p
    Given a config with [integrator] kind="mtk-npt" and tau_p=0.0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue {
      field: "integrator.tau_p", reason: _ })

  @rq-880349c0
  Scenario: Accept any sign of pressure
    Given a config with [integrator] kind="mtk-npt" and pressure=-1.0e5
    When load_config is called
    Then it returns Ok(config)

  @rq-08e113ca
  Scenario: Missing tau_p rejected
    Given a config with [integrator] kind="mtk-npt" and tau_p absent
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "integrator.tau_p" })

  @rq-e08baac0
  Scenario: Reject extra fields (e.g. seed)
    Given a config with [integrator] kind="mtk-npt" and seed=42 (extra)
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, message })
    And path equals "integrator"
    And message mentions "seed"

  # --- Incompatibility ---

  @rq-6478b9c9
  Scenario: mtk-npt + [thermostat] is rejected
    Given a config with [integrator] kind="mtk-npt"
    And a [thermostat] section with any kind
    When load_config is called
    Then it returns Err(ConfigError::IncompatibleThermostat {
      integrator: "mtk-npt" })

  @rq-1b467c03
  Scenario: mtk-npt + [barostat] is rejected
    Given a config with [integrator] kind="mtk-npt"
    And a [barostat] section with any kind
    When load_config is called
    Then it returns Err(ConfigError::IncompatibleBarostat {
      integrator: "mtk-npt" })

  # --- Per-step kernel sequence ---

  @rq-89b8e85c
  Scenario: step() launches the expected kernel set
    Given an MtkNptIntegrator with chain_length=3, yoshida_order=3, n_resp=1,
      particle_count=4
    And a ForceField with one LennardJones slot
    And a warm-up force evaluation has populated forces and virials
    When the runner walks integrator.plan(dt=1e-15) once
    Then KernelStage::KINETIC_ENERGY_REDUCE has count == 2
    And KernelStage::VIRIAL_SUM_REDUCE has count == 2
    And KernelStage::MTK_NPT_RESCALE_VELOCITIES has count == 6  (3 Yoshida × 1 RESP × 2 halves)
    And KernelStage::MTK_NPT_VELOCITY_HALF_KICK has count == 2
    And KernelStage::MTK_NPT_POSITION_DRIFT has count == 1
    And KernelStage::VV_KICK has count == 0  (no plain VV kicks)
    And KernelStage::VV_KICK_DRIFT has count == 0

  @rq-07375d3f
  Scenario: Plan walk on empty state is a no-op
    Given an MtkNptIntegrator with particle_count=0
    When the runner walks integrator.plan(dt) once
    Then every execute(...) call returns Ok(())
    And sim_box.generation() is unchanged
    And p_eps, eps, and all chain DOFs are unchanged

  # --- Cell-coupled kernels ---

  @rq-0d413b82
  Scenario: mtk_velocity_half_kick with exp_minus_alpha=1.0, phi_v_dt_half=0.5 reduces to a half-VV kick
    Given a ParticleBuffers from N=4 particles with known v and F (m=1)
    When mtk_velocity_half_kick(&mut buffers, 1.0, 0.5) is called
    Then post-call velocities equal v + 0.5 · F to f32 round-off

  @rq-db6e7977
  Scenario: mtk_position_drift with exp_b_dt=1.0, phi_x_dt=dt reduces to a plain VV drift
    Given a ParticleBuffers from N=4 particles with known x and v
    When mtk_position_drift(&mut buffers, 1.0, 0.1) is called
    Then post-call positions equal x + 0.1 · v to f32 round-off

  @rq-b3b19b01
  Scenario: Both new kernels on empty state are no-ops
    Given a ParticleBuffers with particle_count() == 0
    When mtk_velocity_half_kick(&mut buffers, 1.0, 1.0) is called
    And mtk_position_drift(&mut buffers, 1.0, 1.0) is called
    Then both return Ok(())

  # --- Shared chain helper ---

  @rq-031d3307
  Scenario: nhc_chain_sub_step produces the same numerical result for the particle chain
    as the existing inline NHC implementation
    Given known xi, p_xi, q_mass, dt, k, g_dof, kt
    When nhc_chain_sub_step(...) is called
    Then it returns the same multiplicative factor (within f64 round-off)
      as the inline computation in the existing NHC slot

  @rq-e4f97cc2
  Scenario: nhc_chain_sub_step handles M = 1 without panicking
    Given a chain with M = 1 (length-1 xi, p_xi, q_mass slices)
    When nhc_chain_sub_step(...) is called
    Then it returns Ok and updates xi[0] and p_xi[0]

  # --- Fractional-coord invariance under pure box rescale ---

  @rq-4b9bb90c
  Scenario: Box rescale leg of the integrator preserves fractional coords
    Given an MtkNptIntegrator and a system whose particle velocities are all zero
    And p_eps is non-zero (e.g. 1.0e-25 kg·m²/s, set by hand)
    And a snapshot of fractional coordinates per particle
    When the runner walks integrator.plan(dt) once
    Then the post-step fractional coordinates of every particle equal the
      snapshot within f32 round-off
      (with v ≡ 0 the drift collapses to the pure box rescale x ← exp(b·dt)·x;
       the box rescales by the same factor)

  # --- Log columns ---

  @rq-aae13334
  Scenario: log_column_names returns pressure, box_volume, mtk_npt_conserved
    Given a constructed MtkNptIntegrator
    Then state.log_column_names() equals ["pressure", "box_volume", "mtk_npt_conserved"]

  @rq-a722f7ce
  Scenario: log_column_values returns the cached pressure, post-step volume,
    and H_MTK assembled from ke/pe and the chain state
    Given an MtkNptIntegrator with known most_recent_pressure, most_recent_volume,
      p_eps, w_cell, and chain state
    When state.log_column_values(ke, pe) is called
    Then it returns [most_recent_pressure, most_recent_volume, H_MTK]
      where H_MTK matches the formula in *MTK conserved Hamiltonian*
      within f64 round-off

  @rq-34943524
  Scenario: Log file header includes pressure, box_volume, mtk_npt_conserved
    when the MTK-NPT integrator is the configured integrator
    Given a config with [integrator].kind = "mtk-npt"
    And log_every > 0
    When the runner produces the log file
    Then its header line ends with "pressure,box_volume,mtk_npt_conserved"

  # --- Box-generation propagation ---

  @rq-ba4087d7
  Scenario: sim_box.generation() advances every plan walk
    Given an MtkNptIntegrator and a SimulationBox at generation g
    When the runner walks integrator.plan(dt) once
    Then sim_box.generation() ≥ g + 1
      (the integrator's `drift_box` sub-step calls sim_box.rescale_isotropic(μ_box) once per plan walk)

  # --- Determinism ---

  @rq-c5f04195
  Scenario: Two independent runs with identical configs are byte-identical
    Given two complete simulations with kind="mtk-npt", identical parameters,
      identical initial particle state, n_steps = 10
    When dynamics run is invoked on each
    Then the two trajectory files are byte-identical
    And the two log files are byte-identical, including the
      pressure, box_volume, and mtk_npt_conserved columns
    And the two final SimulationBox lattices are byte-identical

  # --- Physical correctness ---

  @rq-5e238a79
  Scenario: At equilibrium, time-averaged temperature tracks T_target
    Given an MTK-NPT run with N=256 LJ argon, T=85 K, P=1.0 bar,
      dt=1e-15, tau_t=1e-13, tau_p=1e-12, n_steps=10000,
      initial v sampled at 85 K
    When the run completes
    Then the time-averaged kinetic temperature over the last 5000 log rows
      is within 5% of 85 K

  @rq-17efb1cd
  Scenario: At equilibrium, time-averaged pressure tracks P_ext
    Given the same MTK-NPT run as above
    When the run completes
    Then the time-averaged pressure over the last 5000 log rows is within
      20% of 1.0 bar

  @rq-c5d7891a
  Scenario: H_MTK drifts only by O(dt²) per step
    Given an MTK-NPT run with N=64 LJ argon, dt=1e-15, n_steps=2000
    When the run completes
    Then |H_MTK(n_steps) − H_MTK(0)| / |H_MTK(0)| is < 5.0e-3
    And the drift is dominated by O(dt²) (a halved-dt run has drift
      ≤ 1/4 of the original)

  # --- COM-momentum preservation ---

  @rq-47129004
  Scenario: COM momentum is preserved under the MTK integrator
    Given an MtkNptIntegrator with initial COM momentum = 0
    And n_steps = 100
    When the run completes
    Then Σ_i m_i v_i evaluated on the final velocities is zero within f32 round-off
      (the cell-coupled velocity rescale and the chain rescale are both
       multiplicative; they preserve the zero-COM property exactly in real
       arithmetic, bounded by f32 ULP accumulation in practice)
```
