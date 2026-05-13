# Feature: Velocity Verlet Time Integration <!-- rq-09a2e15f -->

Velocity Verlet is one of the integrators selected by the pluggable-slot
framework (see `framework.md`). It is the deterministic, symplectic NVE
integrator and is chosen via `kind = "velocity-verlet"` in the `[integrator]`
section of the config.

The per-particle arithmetic is split across two CUDA kernels: `vv_kick_drift`
performs the first half-velocity update followed by the position update, and
`vv_kick` performs the second half-velocity update. The trait's `step()`
launches `vv_kick_drift`, then calls `force_field.step(...)` for the new
force evaluation, then launches `vv_kick`.

The integrator ships in two modes:

- **Lossy (default).** Uses ordinary IEEE-754 `f32` arithmetic. Each addition
  may discard low bits to rounding; the trajectory cannot be reversed
  bit-exactly.
- **Lossless.** Each per-particle quantity (positions and velocities) is
  augmented with an `f64` residual buffer that holds the bits the IEEE
  addition would otherwise discard. The lossless kernels carry out every
  update with `f64` intermediate precision so that the pair `(f32 high, f64
  low)` represents the running extended-precision sum to roughly `f64`
  precision. Because the algorithm is symmetric under negation, a forward
  run followed by a velocity-flip and another forward run of the same
  length restores the **observable** `f32` state — positions and velocities
  — bit-exactly. The `f64` residual buffers are internal compensation
  bookkeeping; they may drift by O(f64 ULP) per step under round-tripping
  but that drift never propagates into the observable `f32` state.

This feature covers the four CUDA kernels in `kernels/integrate.cu`, their PTX
loading via `init_device()`, the host-side `LosslessBuffers` struct, and the
Rust launch helpers that drive both modes.

## Algorithm <!-- rq-9f6a73f9 -->

Positions are kept wrapped into the primary image of the simulation
box: each component of `positions_x/y/z` is in `[-L_a / 2, +L_a / 2)`
for the corresponding edge length `L_a`. The companion image triple
`(images_x[i], images_y[i], images_z[i])` records how many full
periods particle `i` has crossed; the unwrapped position is
`positions_a[i] + images_a[i] * L_a` (see `particle-state.md`). The
drift kernels enforce this invariant: after updating each component,
they wrap the result back into the primary image and advance the
corresponding image counter by the integer number of periods crossed.

For each particle `i` with position `x_i`, velocity `v_i`, force `F_i`, and
mass `m_i`, a single timestep of size `dt` corresponds to:

1. `v_i ← v_i + (F_i / m_i) · (dt/2)`         (first half-kick)
2. `x_i ← x_i + v_i · dt`                      (drift; uses the half-step velocity)
3. force evaluation produces a fresh `F_i`
4. `v_i ← v_i + (F_i / m_i) · (dt/2)`         (second half-kick)

`vv_kick_drift` performs steps 1 and 2 in a single thread per particle. `vv_kick`
performs step 4. Step 3 is the force pipeline and is out of scope here.

Each particle's update depends only on its own state. There are no inter-thread
dependencies, no atomics, and no reductions. All four kernels are bit-wise
reproducible by construction: identical inputs yield byte-identical outputs.

### Lossless mode: compensated summation <!-- rq-580fe6f7 -->

In lossless mode, each scalar particle quantity that the integrator mutates
(positions and velocities, per axis) is represented as a `(high, low)` pair
where `high` is `f32` and `low` is `f64`. The extended-precision value
`high + low` (computed in `f64`) approximates the running sum to `f64`
precision. The lossless kernels promote every operand to `f64` for the
inner arithmetic and store the result back as a `(f32 high, f64 low)` pair
such that:

```
high + low ≈ exact_extended_precision_sum   // accurate to f64 precision
```

Concretely, an update `(buf, buf_lo) ← (buf, buf_lo) + delta` proceeds as:

```
extended = (f64) buf + buf_lo + (f64) delta   # all in f64
new_high = (f32) extended                     # IEEE round-to-nearest
new_low  = extended - (f64) new_high          # f64 remainder
```

The drift step in lossless `vv_kick_drift_lossless` uses the
extended-precision velocity `(v + v_lo)` so that the position update is
`x ← x + (v + v_lo) · dt`, decomposed into `(x_high_new, x_low_new)`.

The standard reversal protocol is:

1. Run forward integration for `N` steps to reach state `(x, x_lo, v, v_lo)`.
2. Negate every velocity component: `v ← -v`, `v_lo ← -v_lo`.
3. Run forward integration for `N` more steps with the same `dt`.
4. Negate velocity components again to restore direction.

Under this protocol, `positions_x/y/z` and `velocities_x/y/z` (the
observable `f32` arrays) return to their original byte-identical values on
the same GPU. The `f64` residual buffers may differ from their initial
contents by a small number of f64 ULPs per round-tripped step — that drift
is internal compensation that never propagates back into the observable
`f32` state, because residuals are bounded by the round-off slack of an
`f32` addition. In between forward and reverse runs, the force-evaluation
pipeline must be re-invoked at each step's intermediate position (just as
in the forward direction); forces themselves are deterministic functions of
positions so they do not require their own residuals.

## Feature API <!-- rq-74535c7a -->

### Types <!-- rq-33c0521d -->

- `LosslessBuffers` — host-side wrapper around the six device residual <!-- rq-303211d9 -->
  buffers used by the lossless integrator. All six `CudaSlice<f64>` fields
  are `pub` so the lossless launchers can pass them directly to kernel
  launches; carries an `Arc<CudaDevice>` and a private `particle_count`.

  Fields:
  - `device: Arc<CudaDevice>`
  - `positions_x_lo: CudaSlice<f64>`
  - `positions_y_lo: CudaSlice<f64>`
  - `positions_z_lo: CudaSlice<f64>`
  - `velocities_x_lo: CudaSlice<f64>`
  - `velocities_y_lo: CudaSlice<f64>`
  - `velocities_z_lo: CudaSlice<f64>`
  - private `particle_count: usize`

  Constructor:

  - `LosslessBuffers::new(device: Arc<CudaDevice>, particle_count: usize) -> Result<LosslessBuffers, GpuError>`
    - Allocates six `CudaSlice<f64>` of length `particle_count` via
      `CudaDevice::alloc_zeros`. Every residual starts at `0.0_f64`,
      meaning the initial extended-precision state is exactly the value
      held by the corresponding `ParticleBuffers` field.
    - A `particle_count` of zero is permitted and yields six length-zero
      device allocations.

  Accessor:

  - `LosslessBuffers::particle_count(&self) -> usize` — returns the value
    supplied at construction.

### CUDA Kernels <!-- rq-10cc8ddf -->

`kernels/integrate.cu` declares four `extern "C"` kernels:

```c
extern "C" __global__ void vv_kick_drift(
    float *positions_x, float *positions_y, float *positions_z,
    int *images_x, int *images_y, int *images_z,
    float *velocities_x, float *velocities_y, float *velocities_z,
    const float *forces_x, const float *forces_y, const float *forces_z,
    const float *masses,
    float lx, float ly, float lz,
    float dt,
    unsigned int n);

extern "C" __global__ void vv_kick(
    float *velocities_x, float *velocities_y, float *velocities_z,
    const float *forces_x, const float *forces_y, const float *forces_z,
    const float *masses,
    float dt,
    unsigned int n);

extern "C" __global__ void vv_kick_drift_lossless(
    float *positions_x, float *positions_y, float *positions_z,
    int *images_x, int *images_y, int *images_z,
    float *velocities_x, float *velocities_y, float *velocities_z,
    double *positions_x_lo, double *positions_y_lo, double *positions_z_lo,
    double *velocities_x_lo, double *velocities_y_lo, double *velocities_z_lo,
    const float *forces_x, const float *forces_y, const float *forces_z,
    const float *masses,
    float lx, float ly, float lz,
    float dt,
    unsigned int n);

extern "C" __global__ void vv_kick_lossless(
    float *velocities_x, float *velocities_y, float *velocities_z,
    double *velocities_x_lo, double *velocities_y_lo, double *velocities_z_lo,
    const float *forces_x, const float *forces_y, const float *forces_z,
    const float *masses,
    float dt,
    unsigned int n);
```

`vv_kick` and `vv_kick_lossless` update only velocities and so do not
touch `images_*` or take box edges. `vv_kick_drift` and
`vv_kick_drift_lossless` are the drift-bearing kernels that wrap
positions and advance image counts.

Each thread computes its global index as `blockIdx.x * blockDim.x + threadIdx.x`.
If the index is `>= n` the thread returns without touching any buffer.

The four kernel bodies share their per-particle math through a
`__device__` helper function templated on a `bool LOSSLESS` parameter.
`if constexpr (LOSSLESS)` selects between plain IEEE addition (lossy mode)
and the compensated `(high, low)` decomposition described in the algorithm
section. Each `extern "C"` wrapper instantiates the template with the
appropriate value:

```c
template <bool LOSSLESS>
__device__ inline void vv_kick_drift_body(
    unsigned int i, /* full argument list including residual pointers */);

extern "C" __global__ void vv_kick_drift(...)            { ... vv_kick_drift_body<false>(...); }
extern "C" __global__ void vv_kick_drift_lossless(...)   { ... vv_kick_drift_body<true>(...);  }
```

The lossy wrappers do not pass residual pointers and the lossless code
path is dead-code-eliminated by `if constexpr`, so the lossy kernels
incur no FLOP, register, or memory-traffic cost from the lossless
machinery.

Within `vv_kick_drift` (lossy), each thread performs in this order, on `f32`:

```
ax = forces_x[i] / masses[i]
ay = forces_y[i] / masses[i]
az = forces_z[i] / masses[i]
vx = velocities_x[i] + ax * (dt * 0.5f)
vy = velocities_y[i] + ay * (dt * 0.5f)
vz = velocities_z[i] + az * (dt * 0.5f)
velocities_x[i] = vx
velocities_y[i] = vy
velocities_z[i] = vz
px = positions_x[i] + vx * dt
py = positions_y[i] + vy * dt
pz = positions_z[i] + vz * dt
(positions_x[i], images_x[i]) = wrap_and_count(px, lx, images_x[i])
(positions_y[i], images_y[i]) = wrap_and_count(py, ly, images_y[i])
(positions_z[i], images_z[i]) = wrap_and_count(pz, lz, images_z[i])
```

where `wrap_and_count(p, L, n)` returns `(p - k*L, n + k)` with
`k = floor((p + L*0.5f) / L)` evaluated in `f32`. This is the same
wrap formula `SimulationBox::wrap_position` uses on the host. A single
update can cross any integer number of periods (the formula is
branch-free in `k`); the integer `k` is always added to the existing
image count, so the unwrapped position
`positions_a[i] + images_a[i] * L_a` is invariant under wrapping.

Within `vv_kick` (lossy), each thread performs:

```
ax = forces_x[i] / masses[i]
ay = forces_y[i] / masses[i]
az = forces_z[i] / masses[i]
velocities_x[i] += ax * (dt * 0.5f)
velocities_y[i] += ay * (dt * 0.5f)
velocities_z[i] += az * (dt * 0.5f)
```

Within `vv_kick_drift_lossless`, each thread performs the algebraically
equivalent computation but with every velocity and position update done
via the compensated `(high, low)` decomposition described in the algorithm
section. The drift step uses `(v + v_lo)` (computed in the chosen
intermediate precision) when forming the position increment. After
rounding the post-drift `(high, low)` pair back into the canonical
form, the high half is wrapped via `wrap_and_count` against the
corresponding edge length and the image counter is advanced; the low
half is left unchanged by the wrap (the rounding has already
re-zeroed it within ULP).

Within `vv_kick_lossless`, each thread performs the velocity update via
the same compensated decomposition.

None of the four kernels writes to `forces_*` or to `masses`. The lossy
kernels do not read or write the residual buffers; the lossless kernels
read and write both the high and the low halves of the quantities they
mutate.

### PTX Module Loading <!-- rq-e20b2f39 -->

`init_device()` loads the compiled `kernels/integrate.cu` PTX with module name
`"integrate"` and registers the function names `"vv_kick_drift"`, `"vv_kick"`,
`"vv_kick_drift_lossless"`, and `"vv_kick_lossless"`. The `fill` smoke-test
module continues to be loaded alongside it.

### Rust Launch Helpers <!-- rq-581ec3ff -->

Four free functions live in `src/gpu/kernels.rs` and are re-exported from
`crate::gpu`:

- `vv_kick_drift(buffers: &mut ParticleBuffers, sim_box: &SimulationBox, dt: f32) -> Result<(), GpuError>` <!-- rq-f1ba909b -->
  - Launches the `vv_kick_drift` kernel using the device buffers carried
    by `buffers` and the edge lengths read from `sim_box`.
  - Block size is 256; grid size is `ceil(buffers.particle_count() / 256)`.
  - When `buffers.particle_count() == 0`, returns `Ok(())` without launching a
    kernel.
  - Returns the underlying `GpuError` if the kernel launch fails.
  - Panics if the `"integrate"` module is not loaded on the device, since this
    indicates a programming error in `init_device()`.

- `vv_kick(buffers: &mut ParticleBuffers, dt: f32) -> Result<(), GpuError>` <!-- rq-f2e3fa58 -->
  - Same launch configuration and error/empty-state handling as `vv_kick_drift`.
  - Launches the `vv_kick` kernel. Does not consult `sim_box` because the
    velocity-only update never wraps positions.

- `vv_kick_drift_lossless(buffers: &mut ParticleBuffers, lossless: &mut LosslessBuffers, sim_box: &SimulationBox, dt: f32) -> Result<(), GpuError>` <!-- rq-7d5e87ee -->
  - Launches the `vv_kick_drift_lossless` kernel using both the
    `ParticleBuffers` (for `(high)` halves and image flags) and the
    `LosslessBuffers` (for `(low)` halves), with edge lengths read from
    `sim_box`.
  - Same block/grid configuration and empty-state handling as the lossy
    variant; debug-asserts that `lossless.particle_count()` equals
    `buffers.particle_count()`.
  - Panics if the `"integrate"` module is not loaded.

- `vv_kick_lossless(buffers: &mut ParticleBuffers, lossless: &mut LosslessBuffers, dt: f32) -> Result<(), GpuError>` <!-- rq-4ea8bbb2 -->
  - Same configuration and empty-state handling as `vv_kick_drift_lossless`.
  - Launches the `vv_kick_lossless` kernel. Does not consult `sim_box`.

None of the four helpers inspects mass values or constrains `dt`. NaN,
infinite, zero, or negative values flow through the kernel arithmetic and
produce corresponding NaN/Inf outputs.

The two velocity-only launchers (`vv_kick`, `vv_kick_lossless`) take no
`SimulationBox`; the two drift-bearing launchers (`vv_kick_drift`,
`vv_kick_drift_lossless`) require it for the wrap step. A run with
`lossless = false` never constructs a `LosslessBuffers` and pays no
memory or kernel-launch cost from the lossless code path.

## Slot Integration <!-- rq-39ab439e -->

`VelocityVerletState` implements the `Integrator` trait declared in
`framework.md`. The `velocity-verlet` builder registered in
`IntegratorRegistry::with_builtins()` takes the `lossless` flag from
the parsed `IntegratorKind::VelocityVerlet { lossless }` config variant
and, when `lossless == true`, allocates a `LosslessBuffers` of the
runner's particle count on the same `Arc<CudaDevice>`.

The implementation's `step(buffers, sim_box, force_field, dt,
timings)` performs the following sequence:

1. Launch `vv_kick_drift` (or `vv_kick_drift_lossless`), bracketed by
   `timings.kernel_start(KernelStage::VV_KICK_DRIFT)` /
   `timings.kernel_stop(KernelStage::VV_KICK_DRIFT)` (or the
   `*_LOSSLESS` stage names — see `performance-analysis.md`). This
   applies the first half-kick using the cached `F(t)` and drifts
   positions to `x(t+dt)`.
2. Call `force_field.step(buffers, sim_box, timings)`, which writes
   `F(t+dt)` into `buffers.forces_*` and the per-particle energy /
   virial buffers.
3. Launch `vv_kick` (or `vv_kick_lossless`) with matching timing
   bracketing. This applies the second half-kick using `F(t+dt)`.

`sim_box` is borrowed mutably but velocity-Verlet does not modify it.
Velocity Verlet is deterministic and stateless across `step()` calls
(beyond the particle buffers themselves); it carries no per-call
counter.

## Launch Configuration <!-- rq-0540b862 -->

- Block size: 256 threads.
- Grid size: `ceil(n / 256)` blocks in the x dimension.
- Shared memory: zero bytes.
- Stream: the default stream carried by `ParticleBuffers::device`.

## Out of Scope <!-- rq-92f5e5c4 -->

- Force evaluation (pair force kernel, segmented reduction, neighbor lists).
- Higher-level orchestration of the timestep loop, including the
  input-file flag that selects between lossy and lossless mode at runtime
  (the simulation-loop feature owns the dispatch; this feature only
  provides the kernels and buffers).
- Other integrators (leapfrog, RK4, predictor-corrector).
- Thermostats, barostats, and constraint algorithms (SHAKE/RATTLE).
- Energy diagnostics and trajectory output.
- Numerical validation of inputs (zero/non-finite masses, dt sign).
- The `f64` precision feature flag for the *primary* particle state
  (the lossless kernels use `f64` only as an internal intermediate to
  compute the compensated decomposition; the public state remains `f32`
  pairs).
- Multi-stream or multi-GPU launches.
- Bit-reversibility under variable forces verified through the full
  force-evaluation pipeline. The lossless integrator kernels are
  reversibility-correct in isolation; whole-pipeline reversibility is the
  responsibility of a future end-to-end reversibility test that drives
  `lj_pair_force` and `reduce_pair_forces` between integrator calls.

---

## Gherkin Scenarios <!-- rq-8c220501 -->

```gherkin
Feature: Velocity Verlet time integration

  Background:
    Given a SimulationBox with lx=ly=lz=1.0e6 (large enough that no
      particle in the scenarios below wraps in a single drift call, so
      image flags stay at zero and positions match the closed-form
      free-streaming arithmetic)

  # --- Module loading ---

  @rq-bc8375ce
  Scenario: init_device loads the integrate module with both kernels
    Given a CUDA-capable GPU is available as device 0
    When init_device() is called
    Then the device exposes a function named "vv_kick_drift" in module "integrate"
    And the device exposes a function named "vv_kick" in module "integrate"

  # --- vv_kick_drift on a free particle ---

  @rq-4f7dc024
  Scenario: vv_kick_drift advances position and leaves velocity unchanged when force is zero
    Given a ParticleBuffers built from a single particle at x=(1.0, 2.0, 3.0), v=(0.5, -0.25, 0.125), F=(0, 0, 0), m=1.0
    When vv_kick_drift(&mut buffers, dt=0.1) is called
    And the buffers are downloaded into a host ParticleState
    Then positions equal (1.0 + 0.5*0.1, 2.0 + -0.25*0.1, 3.0 + 0.125*0.1)
    And velocities equal (0.5, -0.25, 0.125)

  # --- vv_kick_drift under constant force ---

  @rq-d25000c5
  Scenario: vv_kick_drift produces the exact half-step kinematics under a constant force
    Given a ParticleBuffers built from a single particle at x=0, v=0, F=(2.0, -4.0, 1.0), m=1.0
    When vv_kick_drift(&mut buffers, dt=0.1) is called
    And the buffers are downloaded into a host ParticleState
    Then velocities equal (2.0*0.05, -4.0*0.05, 1.0*0.05)
    And positions equal (2.0*0.05*0.1, -4.0*0.05*0.1, 1.0*0.05*0.1)

  # --- vv_kick on a free particle ---

  @rq-2f52c25e
  Scenario: vv_kick leaves velocity unchanged when force is zero
    Given a ParticleBuffers built from a single particle at x=(1, 2, 3), v=(0.5, -0.25, 0.125), F=(0, 0, 0), m=1.0
    When vv_kick(&mut buffers, dt=0.1) is called
    And the buffers are downloaded into a host ParticleState
    Then velocities equal (0.5, -0.25, 0.125)
    And positions equal (1, 2, 3)

  # --- Full velocity-Verlet step under a constant force ---

  @rq-29718dcf
  Scenario: kick_drift followed by kick reproduces closed-form constant-acceleration kinematics
    Given a ParticleBuffers built from a single particle at x=(0, 0, 0), v=(0, 0, 0), F=(2.0, 0, 0), m=1.0
    When vv_kick_drift(&mut buffers, dt=0.1) is called
    And vv_kick(&mut buffers, dt=0.1) is called
    And the buffers are downloaded into a host ParticleState
    Then positions_x[0] equals 0 + 0*0.1 + 0.5*(2.0/1.0)*0.1*0.1
    And velocities_x[0] equals 0 + (2.0/1.0)*0.1
    And positions_y[0], positions_z[0], velocities_y[0], velocities_z[0] are all 0

  # --- Mass scaling ---

  @rq-bd149f52
  Scenario: Acceleration scales inversely with mass
    Given a ParticleBuffers built from two particles both at x=0, v=0
    And both particles have F=(1.0, 0, 0)
    And particle 0 has m=1.0 and particle 1 has m=4.0
    When vv_kick_drift(&mut buffers, dt=0.2) is called
    And the buffers are downloaded into a host ParticleState
    Then velocities_x[0] equals (1.0/1.0) * 0.1
    And velocities_x[1] equals (1.0/4.0) * 0.1

  # --- Multi-particle independence ---

  @rq-b13eba96
  Scenario: Particles evolve independently with their own forces
    Given a ParticleBuffers built from three particles with distinct forces and the same mass m=1.0
    When vv_kick_drift(&mut buffers, dt=0.1) is called
    And vv_kick(&mut buffers, dt=0.1) is called
    And the buffers are downloaded into a host ParticleState
    Then each particle's position and velocity match the closed-form constant-force update for its own F

  # --- Zero timestep ---

  @rq-e8c21a03
  Scenario: dt=0 leaves state unchanged for vv_kick_drift
    Given a ParticleBuffers built from N=8 particles with arbitrary nonzero positions, velocities, forces, and masses
    And a snapshot of the host state before launch
    When vv_kick_drift(&mut buffers, dt=0.0) is called
    And the buffers are downloaded into a host ParticleState
    Then every field of the downloaded state is byte-identical to the snapshot

  @rq-d28737dd
  Scenario: dt=0 leaves state unchanged for vv_kick
    Given a ParticleBuffers built from N=8 particles with arbitrary nonzero positions, velocities, forces, and masses
    And a snapshot of the host state before launch
    When vv_kick(&mut buffers, dt=0.0) is called
    And the buffers are downloaded into a host ParticleState
    Then every field of the downloaded state is byte-identical to the snapshot

  # --- Empty state ---

  @rq-1e2d749d
  Scenario: vv_kick_drift on an empty state is a no-op
    Given a ParticleBuffers with particle_count() == 0
    When vv_kick_drift(&mut buffers, dt=0.1) is called
    Then it returns Ok(())
    And the buffers remain at particle_count() == 0

  @rq-386cfae3
  Scenario: vv_kick on an empty state is a no-op
    Given a ParticleBuffers with particle_count() == 0
    When vv_kick(&mut buffers, dt=0.1) is called
    Then it returns Ok(())
    And the buffers remain at particle_count() == 0

  # --- Bounds handling ---

  @rq-a93a5b14
  Scenario: Block-non-aligned particle counts are handled by the bounds check
    Given a ParticleBuffers built from N=1000 particles with F=0 and v=(1, 0, 0) for every particle
    And a snapshot of all device buffers before launch
    When vv_kick_drift(&mut buffers, dt=0.1) is called
    And the buffers are downloaded into a host ParticleState
    Then positions_x[i] == initial_positions_x[i] + 0.1 for every i in 0..1000
    And velocities, forces, masses, and particle_ids are byte-identical to the snapshot

  # --- Buffer side effects ---

  @rq-7dfa14cf
  Scenario: vv_kick_drift does not modify forces or masses
    Given a ParticleBuffers built from N=4 particles with arbitrary nonzero values
    And a snapshot of forces_x, forces_y, forces_z, and masses before launch
    When vv_kick_drift(&mut buffers, dt=0.1) is called
    And the buffers are downloaded into a host ParticleState
    Then forces_x, forces_y, forces_z, and masses are byte-identical to the snapshot

  @rq-f721b7a1
  Scenario: vv_kick does not modify forces, masses, or positions
    Given a ParticleBuffers built from N=4 particles with arbitrary nonzero values
    And a snapshot of forces, masses, and positions before launch
    When vv_kick(&mut buffers, dt=0.1) is called
    And the buffers are downloaded into a host ParticleState
    Then forces_x, forces_y, forces_z, masses, positions_x, positions_y, and positions_z are byte-identical to the snapshot

  # --- Reproducibility ---

  @rq-37e6f318
  Scenario: Two independent runs produce byte-identical outputs
    Given two ParticleBuffers built from identical ParticleState inputs of N=128 particles
    When vv_kick_drift then vv_kick is launched on each, both with dt=0.01
    And both buffers are downloaded into host ParticleStates
    Then every f32 and u32 array of run A is byte-identical to run B

  # --- NaN propagation ---

  @rq-8e47334c
  Scenario: NaN forces propagate to NaN velocities and positions
    Given a ParticleBuffers built from a single particle with F_x=f32::NAN and finite m, v, x
    When vv_kick_drift(&mut buffers, dt=0.1) is called
    And the buffers are downloaded into a host ParticleState
    Then velocities_x[0] is NaN
    And positions_x[0] is NaN
    And velocities_y[0], velocities_z[0], positions_y[0], positions_z[0] are unchanged

  # --- Time-reversibility ---

  @rq-b2d67b57
  Scenario: Forward then negated step returns to the original state for a free particle
    Given a ParticleBuffers built from N=4 free particles (F=0) with arbitrary nonzero positions and velocities
    And a snapshot of the host state before launch
    When vv_kick_drift(&mut buffers, dt=0.1) is called
    And vv_kick_drift(&mut buffers, dt=-0.1) is called
    And the buffers are downloaded into a host ParticleState
    Then positions and velocities equal their snapshot values within an absolute tolerance of 1e-6

  # --- Lossless mode: module loading and construction ---

  @rq-70fe268a
  Scenario: init_device exposes the lossless integrator kernels
    Given a CUDA-capable GPU is available as device 0
    When init_device() is called
    Then the device exposes a function named "vv_kick_drift_lossless" in module "integrate"
    And the device exposes a function named "vv_kick_lossless" in module "integrate"

  @rq-5bfa5b37
  Scenario: LosslessBuffers::new allocates six zero-initialised residual buffers
    Given a CudaDevice obtained from init_device()
    When LosslessBuffers::new(device, particle_count=4) is called
    Then it returns Ok(buffers)
    And buffers.particle_count() is 4
    And each of positions_{x,y,z}_lo and velocities_{x,y,z}_lo has length 4
    And every element of every residual buffer equals 0.0_f32 when downloaded

  @rq-b96ce51d
  Scenario: LosslessBuffers::new with particle_count = 0
    Given a CudaDevice obtained from init_device()
    When LosslessBuffers::new(device, particle_count=0) is called
    Then it returns Ok(buffers)
    And every residual device buffer has length 0

  # --- Lossless mode: empty-state and bounds handling ---

  @rq-58cac735
  Scenario: vv_kick_drift_lossless on an empty state is a no-op
    Given a ParticleBuffers and LosslessBuffers both with particle_count() == 0
    When vv_kick_drift_lossless(&mut buffers, &mut lossless, dt=0.1) is called
    Then it returns Ok(())

  @rq-5626dfc6
  Scenario: vv_kick_lossless on an empty state is a no-op
    Given a ParticleBuffers and LosslessBuffers both with particle_count() == 0
    When vv_kick_lossless(&mut buffers, &mut lossless, dt=0.1) is called
    Then it returns Ok(())

  @rq-5a6d5e9e
  Scenario: vv_kick_drift_lossless handles a block-non-aligned particle count
    Given a ParticleBuffers and LosslessBuffers both with particle_count() == 1000
    And every particle has F=0, v=(1, 0, 0), x_lo=0, v_lo=0
    When vv_kick_drift_lossless(&mut buffers, &mut lossless, dt=0.1) is called
    Then no out-of-bounds device-memory write occurs
    And the high parts of positions_x evolve consistently with the lossless drift

  # --- Lossless mode: side effects ---

  @rq-bb075030
  Scenario: vv_kick_drift_lossless does not modify forces, masses, or particle_ids
    Given a ParticleBuffers built from N=4 particles with arbitrary nonzero values
    And a fresh LosslessBuffers
    And a snapshot of forces_*, masses, and particle_ids before launch
    When vv_kick_drift_lossless(&mut buffers, &mut lossless, dt=0.1) is called
    And particle_buffers is downloaded
    Then forces_x, forces_y, forces_z, masses, and particle_ids are byte-identical to the snapshot

  @rq-acafdfe4
  Scenario: vv_kick_lossless does not modify positions, forces, masses, or particle_ids
    Given a ParticleBuffers built from N=4 particles with arbitrary nonzero values
    And a fresh LosslessBuffers
    And a snapshot of positions_*, forces_*, masses, and particle_ids before launch
    When vv_kick_lossless(&mut buffers, &mut lossless, dt=0.1) is called
    And particle_buffers is downloaded
    Then positions_x, positions_y, positions_z, forces_x, forces_y, forces_z, masses, and particle_ids are byte-identical to the snapshot

  # --- Lossless mode: bit-reversibility ---

  @rq-1a504311
  Scenario: Single-step round-trip restores the observable state bit-exactly under zero force
    Given a ParticleBuffers built from N=8 particles with F=0 and arbitrary nonzero positions and velocities
    And a fresh LosslessBuffers (all residuals zero)
    And a snapshot of the high and low halves of every position and velocity component
    When vv_kick_drift_lossless(&mut buffers, &mut lossless, dt=0.1) is called
    And every velocity component (high and low) is negated on the device
    And vv_kick_drift_lossless(&mut buffers, &mut lossless, dt=0.1) is called
    And every velocity component (high and low) is negated on the device
    And buffers and lossless are downloaded
    Then positions_x, positions_y, positions_z agree with the snapshot byte-for-byte
    And velocities_x, velocities_y, velocities_z agree with the snapshot byte-for-byte
    And every residual buffer agrees with the snapshot to within 1e-12

  @rq-b73316ed
  Scenario: Multi-step round-trip restores the observable state bit-exactly under constant force
    Given a ParticleBuffers built from N=16 particles with arbitrary nonzero positions, velocities, masses, and forces
    And a fresh LosslessBuffers (all residuals zero)
    And a snapshot of all eleven host arrays and all six residual arrays before the loop
    When the lossless kernels (vv_kick_drift_lossless then vv_kick_lossless) are launched 50 times with dt=0.001
    And every velocity component (high and low) is negated on the device
    And the same lossless kernels are launched 50 more times with dt=0.001
    And every velocity component (high and low) is negated on the device
    And buffers and lossless are downloaded
    Then every f32 and u32 observable array agrees with the snapshot byte-for-byte
    And every residual buffer agrees with the snapshot to within 1e-10

  @rq-2a0e97f5
  Scenario: Two independent lossless runs produce byte-identical results
    Given two pairs of ParticleBuffers and LosslessBuffers built from identical inputs at N=64
    When vv_kick_drift_lossless then vv_kick_lossless is launched 10 times on each pair with dt=0.001
    And both pairs are downloaded
    Then every f32 array of run A agrees byte-for-byte with run B
    And every residual array of run A agrees byte-for-byte with run B

  # --- Image-flag wrap (drift only) ---

  @rq-e6b6f2d8
  Scenario: vv_kick_drift wraps positions back into the primary image
    Given a SimulationBox with lx=ly=lz=10.0
    And a ParticleBuffers built from a single particle at x=(4.9, 0.0, 0.0),
      v=(2.0, 0.0, 0.0), F=(0, 0, 0), m=1.0 and zero image flags
    When vv_kick_drift(&mut buffers, &sim_box, dt=0.1) is called
    And the buffers are downloaded
    Then positions_x[0] equals -4.9
      (raw position 4.9 + 2.0 * 0.1 = 5.1; wrap subtracts lx, giving -4.9)
    And images_x[0] equals 1
    And images_y[0] and images_z[0] are 0

  @rq-aaf3d06f
  Scenario: vv_kick_drift wraps in the negative-x direction
    Given a SimulationBox with lx=ly=lz=10.0
    And a ParticleBuffers built from a single particle at x=(-4.9, 0.0, 0.0),
      v=(-2.0, 0.0, 0.0), F=(0, 0, 0), m=1.0 and zero image flags
    When vv_kick_drift(&mut buffers, &sim_box, dt=0.1) is called
    And the buffers are downloaded
    Then positions_x[0] equals 4.9
    And images_x[0] equals -1

  @rq-dae60da6
  Scenario: vv_kick_drift increments image counts on multi-period crossings
    Given a SimulationBox with lx=10.0
    And a single particle at x=0.0 with v=(250.0, 0, 0), F=(0, 0, 0), m=1.0,
      images_x[0] = 7
    When vv_kick_drift(&mut buffers, &sim_box, dt=0.1) is called
    And the buffers are downloaded
    Then positions_x[0] is in [-5.0, +5.0)
    And images_x[0] equals 7 + 2
      (raw position 0 + 25.0 = 25.0 crosses two full periods of lx=10)
    And positions_x[0] + images_x[0] * lx equals 25.0 to f32 precision

  @rq-9cd01384
  Scenario: vv_kick does not modify image flags
    Given a ParticleBuffers built from N=4 particles with arbitrary nonzero positions,
      velocities, forces, masses, and image flags
    And a snapshot of images_x, images_y, images_z before launch
    When vv_kick(&mut buffers, dt=0.1) is called
    And the buffers are downloaded
    Then images_x, images_y, images_z are byte-identical to the snapshot

  @rq-b8fde05b
  Scenario: Unwrapped displacement is preserved under wrap
    Given a SimulationBox with lx=ly=lz=10.0
    And a ParticleBuffers built from N=2 particles with arbitrary positions
      inside [-5.0, +5.0), zero image flags, velocities chosen so each
      particle crosses one boundary in the next step, F=0
    And a recorded unwrapped position p0 = positions + images * L per particle
    When vv_kick_drift(&mut buffers, &sim_box, dt=0.1) is called
    And the buffers are downloaded
    Then for every particle, positions_a + images_a * L_a equals
      p0 + v_a * 0.1 (in f32) to within one ULP per component
```
