# Feature: Velocity Verlet Time Integration <!-- rq-09a2e15f -->

The simulation advances particles in time using velocity Verlet, a second-order
symplectic integrator. The per-particle arithmetic is split across two CUDA
kernels: `vv_kick_drift` performs the first half-velocity update followed by the
position update, and `vv_kick` performs the second half-velocity update. The
host sequences these two kernels around a force-evaluation pipeline (which is
the responsibility of a separate feature).

This feature covers the two CUDA kernels in `kernels/integrate.cu`, their PTX
loading via `init_device()`, and Rust-side launch helpers that drive them.

## Algorithm <!-- rq-9f6a73f9 -->

For each particle `i` with position `x_i`, velocity `v_i`, force `F_i`, and
mass `m_i`, a single timestep of size `dt` corresponds to:

1. `v_i ← v_i + (F_i / m_i) · (dt/2)`         (first half-kick)
2. `x_i ← x_i + v_i · dt`                      (drift; uses the half-step velocity)
3. force evaluation produces a fresh `F_i`
4. `v_i ← v_i + (F_i / m_i) · (dt/2)`         (second half-kick)

`vv_kick_drift` performs steps 1 and 2 in a single thread per particle. `vv_kick`
performs step 4. Step 3 is the force pipeline and is out of scope here.

Each particle's update depends only on its own state. There are no inter-thread
dependencies, no atomics, and no reductions. Both kernels are bit-wise
reproducible by construction: identical inputs yield byte-identical outputs.

## Feature API <!-- rq-74535c7a -->

### CUDA Kernels <!-- rq-10cc8ddf -->

`kernels/integrate.cu` declares two `extern "C"` kernels with these signatures:

```c
extern "C" __global__ void vv_kick_drift(
    float *positions_x, float *positions_y, float *positions_z,
    float *velocities_x, float *velocities_y, float *velocities_z,
    const float *forces_x, const float *forces_y, const float *forces_z,
    const float *masses,
    float dt,
    unsigned int n);

extern "C" __global__ void vv_kick(
    float *velocities_x, float *velocities_y, float *velocities_z,
    const float *forces_x, const float *forces_y, const float *forces_z,
    const float *masses,
    float dt,
    unsigned int n);
```

Each thread computes its global index as `blockIdx.x * blockDim.x + threadIdx.x`.
If the index is `>= n` the thread returns without touching any buffer.

Within `vv_kick_drift`, each thread performs (in this order, on `f32`):

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
positions_x[i] += vx * dt
positions_y[i] += vy * dt
positions_z[i] += vz * dt
```

Within `vv_kick`, each thread performs:

```
ax = forces_x[i] / masses[i]
ay = forces_y[i] / masses[i]
az = forces_z[i] / masses[i]
velocities_x[i] += ax * (dt * 0.5f)
velocities_y[i] += ay * (dt * 0.5f)
velocities_z[i] += az * (dt * 0.5f)
```

Neither kernel writes to `forces_*` or to `masses`.

### PTX Module Loading <!-- rq-e20b2f39 -->

`init_device()` loads the compiled `kernels/integrate.cu` PTX with module name
`"integrate"` and registers the function names `"vv_kick_drift"` and
`"vv_kick"`. The `fill` smoke-test module continues to be loaded alongside it.

### Rust Launch Helpers <!-- rq-581ec3ff -->

Two free functions live in `src/gpu/kernels.rs` and are re-exported from
`crate::gpu`:

- `vv_kick_drift(buffers: &mut ParticleBuffers, dt: f32) -> Result<(), GpuError>` <!-- rq-f1ba909b -->
  - Launches the `vv_kick_drift` kernel using the device buffers carried by
    `buffers`.
  - Block size is 256; grid size is `ceil(buffers.particle_count() / 256)`.
  - When `buffers.particle_count() == 0`, returns `Ok(())` without launching a
    kernel.
  - Returns the underlying `GpuError` if the kernel launch fails.
  - Panics if the `"integrate"` module is not loaded on the device, since this
    indicates a programming error in `init_device()`.

- `vv_kick(buffers: &mut ParticleBuffers, dt: f32) -> Result<(), GpuError>` <!-- rq-f2e3fa58 -->
  - Same launch configuration and error/empty-state handling as `vv_kick_drift`.
  - Launches the `vv_kick` kernel.

Neither helper inspects mass values or constrains `dt`. NaN, infinite, zero,
or negative values flow through the kernel arithmetic and produce corresponding
NaN/Inf outputs.

## Launch Configuration <!-- rq-0540b862 -->

- Block size: 256 threads.
- Grid size: `ceil(n / 256)` blocks in the x dimension.
- Shared memory: zero bytes.
- Stream: the default stream carried by `ParticleBuffers::device`.

## Out of Scope <!-- rq-92f5e5c4 -->

- Force evaluation (pair force kernel, segmented reduction, neighbor lists).
- Higher-level orchestration of the timestep loop.
- Other integrators (leapfrog, RK4, predictor-corrector).
- Thermostats, barostats, and constraint algorithms (SHAKE/RATTLE).
- Energy diagnostics and trajectory output.
- Numerical validation of inputs (zero/non-finite masses, dt sign).
- The `f64` precision feature flag.
- Multi-stream or multi-GPU launches.

---

## Gherkin Scenarios <!-- rq-8c220501 -->

```gherkin
Feature: Velocity Verlet time integration

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
```
