# Feature: CUDA Graph Per-Step Loop <!-- rq-cf7e5025 -->

HeddleMD's per-step MD kernel sequence runs as a CUDA graph that is
captured once per phase and replayed in batches. Graph replay collapses
the ~15–20 `cuLaunchKernel` calls per step into a single `cuGraphLaunch`,
eliminating per-step driver overhead. The captured graph contains every
kernel that runs during a step's thermostat / integrator / barostat /
constraint sequence; host-visible bookkeeping (displacement-check,
trajectory writes, log rows) runs at batch boundaries.

This file specifies the activation policy, the capture lifecycle, the
batched replay loop, the per-slot eligibility hooks, the device-resident
RNG counter contract that graph replay depends on, the `Timings`
interaction, and the configuration knobs.

The runner's overall phase shape is described in
`simulation-runner.md`; the displacement-check semantics are described
in `forces/neighbor-list.md`; the deterministic-reduction policy that
graph replay must preserve is described in
`pipeline-reproducibility.md`.

## Activation Policy <!-- rq-3c78ea7d -->

Graph mode is active for an MD phase when **all** of the following
hold:

- The phase is an MD phase (not a minimization phase).
- `[simulation].cuda_graphs_disable` is `false` (the default).
- The neighbor-list mode is `CellList` or `Trivial`.
- Every potential slot configured in the phase's force field reports
  `Potential::graph_compatible() == true`.
- Every active slot for the phase (integrator, optional thermostat,
  optional barostat, optional constraint) reports
  `graph_compatible(&params) == true`.
- Every active integrator / thermostat / barostat slot returns
  `Some(_)` from `post_force_per_particle_fragment()`. The
  JIT-composed post-force per-particle kernel is part of the
  captured sequence and the per-step loop alike; a slot that does
  not expose a fragment cannot participate. Every in-tree slot
  satisfies this. User-registered slots that return `None` raise
  `StepError::MissingPostForcePerParticleFragment` at phase setup
  before either graph capture or per-step launch is attempted.

When any condition fails the phase runs the per-step launch loop
described in `simulation-runner.md` step 17, with full per-kernel
`Timings`.

Graph mode is the default for eligible phases. There is no per-phase
opt-in: eligibility is the activation criterion.

### Slot eligibility <!-- rq-26c9b8cb -->

Each in-tree slot's builder reports its `graph_compatible` value;
every in-tree slot exposes a post-force per-particle fragment:

| Kind        | Slot                    | `graph_compatible` | post-force fragment |
|-------------|-------------------------|--------------------|---------------------|
| Integrator  | `velocity-verlet`       | `true`             | yes (kick)          |
| Integrator  | `langevin-baoab`        | `true`             | yes (B kick)        |
| Integrator  | `mtk-npt`               | `false`            | yes (cell-coupled kick) |
| Thermostat  | `csvr`                  | `true`             | yes (factor rescale) |
| Thermostat  | `andersen`              | `true`             | yes (per-particle Bernoulli + MB) |
| Thermostat  | `berendsen`             | `true`             | yes (factor rescale) |
| Thermostat  | `nose-hoover-chain`     | `false`            | yes (cumulative factor rescale) |
| Barostat    | `c-rescale`             | `true`             | yes (mu position rescale) |
| Barostat    | `berendsen`             | `true`             | yes (mu position rescale) |
| Constraint  | `shake`                 | `true`             | n/a (constraints have their own hooks) |
| Constraint  | `settle`                | `true`             | n/a                 |

`mtk-npt` and `nose-hoover-chain` carry host-side scalar arithmetic
inside their per-step plan executors that runs **outside** the
post-force composed kernel — MTK's chain variable `eps` is updated
host-side during `apply_pre`-equivalent SubSteps; NHC's Yoshida
chain integrates `xi` / `p_xi` host-side during `apply_pre` and
the host portion of `apply_post`. A captured graph cannot reproduce
those host steps. Both slots expose post-force fragments that
contribute to the composed kernel and use it on the per-step
launch path. They remain on the per-step launch path until their
host arithmetic is ported to device kernels.

All in-tree potentials (`lennard_jones`, `coulomb`, `spme_real`,
`spme_reciprocal`, `morse_bond`, `harmonic_angle`) report
`Potential::graph_compatible() == true`. Every kernel and cuFFT call
they dispatch runs on the device's default stream, so the captured
graph naturally records all per-step force evaluation work.
Out-of-tree potentials override `Potential::graph_compatible` to
`false` if their `compute` introduces a secondary stream or performs
host-side work between launches.

External slots that do not override the trait method default to
`graph_compatible = true`. A slot that runs any of the following
operations inside its per-step entry points must override the default
to `false`:

- host-to-device or device-to-host copies (`htod_sync_copy`,
  `dtoh_sync_copy`, and similar)
- host arithmetic that consumes a value read back from the device
- mutation of a struct field used by the next kernel's scalar argument

## RNG Counter Contract <!-- rq-b8a61f12 -->

Every slot that draws random numbers per step (`csvr`, `andersen`,
`langevin-baoab`'s OU sub-step, `c-rescale`) holds a one-element
device buffer `draw_counter_device: CudaSlice<u64>` instead of a
host-side scalar field. Its kernel reads the counter, computes its
Philox sequence from `(seed, counter)`, and writes `counter + 1`
back. The increment is performed by a single thread / lane within
the kernel; no atomic is required.

This contract applies whether or not graph mode is active for the
phase. The per-step launch path and the graph-replay path share the
same device-resident counter, so both produce the same Philox draw
sequence for a given `(seed, initial_counter)` pair.

A host-side `draw_counter: u64` cache field on the slot is refreshed
by `flush_pending_injection` (alongside the conserved-quantity log
columns it already drains). The host field is used only for
diagnostic log columns; it is never an input to a kernel arg.

## JIT-Composed Post-Force Kernel in the Captured Sequence <!-- rq-1db7cf2a -->

The captured per-step sequence includes the JIT-composed post-force
per-particle kernel
(`heddle_jit_composed_post_force_per_particle`) as its final per-
particle node. The composition mechanism and source-fragment
contract are specified in
`rqm/integration/jit-composed-post-force.md`. The captured graph
folds the trailing per-particle work (integrator's final kick,
thermostat's velocity rescale, barostat's position rescale) into
one launch in place of the per-slot rescale kernels.

Every built-in integrator, thermostat, and barostat exposes a
post-force per-particle fragment. The runner gathers these
fragments before `begin_stream_capture`, JIT-compiles the composed
kernel, and binds the resulting `CudaFunction` to the phase. The
captured iteration's `cuLaunchKernel` for the composed kernel
becomes a node in the recorded graph; replays of the graph
re-issue the composed kernel exactly once per physical step.

The non-graph per-step launch loop (cuda_graphs_disable = true or
graph-ineligible phase) uses the same composition mechanism: it
JIT-compiles the composed kernel and launches it directly per
step. Both paths produce byte-identical results.

The standalone per-particle post-force kernels
(`vv_kick`, `vv_kick_lossless`, `csvr_rescale_velocities`,
`rescale_velocities_device_factor`, `andersen_resample`,
`berendsen_rescale_velocities`,
`rescale_positions_device_factor`,
`berendsen_barostat_rescale_positions`) are not declared. The
only entry point that performs per-particle post-force work is
the composed kernel.

`rescale_velocities` and the corresponding Rust launcher remain
declared. The pre-force half-step rescales (the NHC `apply_pre`
cumulative rescale and the MTK particle-chain pre-force rescale)
still use it; there is no pre-force composed kernel.
`mtk_velocity_half_kick` likewise remains declared for the MTK
pre-force `vel_kick_pre` SubStep; only its post-force
`vel_kick_post` invocation is folded into the composed kernel.

### Slot apply-method contract <!-- rq-781966e6 -->

Each slot's `apply_pre`, `apply_post`, and `apply` method runs
only its **scalar-prep** work — the kinetic-energy reduction,
virial reduction, sample-and-factor or compute-mu device kernel,
box mutation, and per-step accumulator bookkeeping. None of them
launches a per-particle kernel. The per-particle update is the
JIT-composed kernel's sole responsibility.

Per-slot scalar-prep details:

- **CSVR thermostat**: `apply_post` runs `compute_kinetic_energy_on_device`
  (writes `ke_scratch`) and `csvr_sample_and_factor` (writes
  `factor_device`). The fragment's per-thread body reads
  `factor_device[0]` and scales velocities.

- **Berendsen thermostat**: `apply_post` runs
  `compute_kinetic_energy_on_device` and a device kernel
  `berendsen_compute_factor` that reads `ke_scratch` plus the
  configured `(tau, kT_target, g_dof)` and writes `factor_device`.
  No host-side λ computation. The fragment reads `factor_device[0]`
  and scales velocities.

- **Nose–Hoover-chain thermostat**: `apply_post` performs the
  Yoshida × `n_resp` chain integration host-side, accumulating
  the per-iteration rescale factors into a single
  cumulative factor. The host writes the cumulative factor to
  `factor_device` via one `htod_sync_copy_into` call at the end
  of `apply_post`. No per-iteration `rescale_velocities` launch
  occurs. The fragment reads `factor_device[0]` and scales
  velocities. The mathematical equivalence is that
  `v_final = v_initial · ∏_i f_i = v_initial · (∏_i f_i)`, and
  the per-iteration `k *= f_i²` host update reads the running
  product, not the device buffer.

- **Andersen thermostat**: `apply_post` runs no scalar-prep
  kernel; the per-particle Bernoulli draw + Maxwell-Boltzmann
  resample happens inside the composed kernel. The fragment
  declares the Andersen functor's `helper_source` containing a
  `__device__` Philox draw routine (the standalone `andersen.cu`
  helpers are referenced verbatim from the fragment string), and
  the fragment's per-thread body draws the Bernoulli per-particle
  sample, branches on `p < p_collision`, and writes new velocity
  components. The bind method pushes `draw_counter_device`,
  `seed`, `p_collision`, and `kT` onto the launch builder.

- **c-rescale barostat**: `apply` runs `compute_kinetic_energy_on_device`,
  `compute_total_virial_on_device`, `c_rescale_compute_mu`
  (writes `mu_device`, mutates the device lattice, and writes
  diagnostics), and increments the device draw counter. The
  fragment reads `mu_device[0]` and scales positions.

- **Berendsen barostat**: `apply` runs `compute_total_virial_on_device`,
  a device kernel that reads `virial_scratch` + the configured
  `(tau, P_target, compressibility, dt)` and writes `mu_device`,
  and mutates the device lattice. The fragment reads `mu_device[0]`
  and scales positions.

The runner's `Integrator::set_jit_composed_post_force_active`,
`Thermostat::set_jit_composed_post_force_active`, and
`Barostat::set_jit_composed_post_force_active` trait methods do
not exist. Slot behaviour is single-mode: `apply_pre` /
`apply_post` / `apply` always do scalar prep only, regardless of
whether the runner is in graph mode or per-step mode.

## Capture Lifecycle <!-- rq-766c88fb -->

Each eligible phase captures one graph after `simulation-runner.md`
step 15 (warm-up force evaluation) and step 16 (step-0 outputs), and
before step 17 (timestep loop). The capture happens in this
sequence:

1. The runner gathers the active integrator's, thermostat's (if
   any), and barostat's (if any) post-force per-particle fragments,
   JIT-compiles the composed kernel via
   `JitComposedPostForcePerParticle::compile_and_load`, and binds
   the resulting `CudaFunction` handle to phase-local state. A
   built-in slot returning `None` from
   `post_force_per_particle_fragment()` raises
   `StepError::MissingPostForcePerParticleFragment` and fails
   phase setup; this is a programmer error rather than a runtime
   fallback.
2. The runner calls `nl.pre_step(sim_box, buffers, timings)` once.
   `pre_step` downloads the single-word
   `disp_rebuild_flag` and rebuilds the neighbor list when the flag
   is non-zero (see `forces/neighbor-list.md` *Displacement Check*).
   The call happens outside any graph capture and is not recorded.
3. The runner calls
   `device.begin_stream_capture(CaptureMode::ThreadLocal)` on the
   default stream.
4. The runner executes the kernel sequence for one physical step,
   using `force_field.step_no_neighbor_check(...)` in place of the
   ordinary `force_field.step(...)`. The sequence is:
   - `thermostat.apply_pre(buffers, dt, timings)` if a thermostat is
     active
   - `run_step_with_skipped_substep(integrator, buffers, sim_box,
     force_field, constraint, ..., dt, timings,
     integrator.post_force_substep_index(dt).unwrap())`, where every
     internal `force_field.step` call is replaced by
     `force_field.step_no_neighbor_check`. The integrator's
     post-force SubStep (the trailing `KickHalf` / `KickDrift` per
     `Integrator::post_force_substep_index`) is skipped — the
     composed kernel handles it.
   - `thermostat.apply_post(buffers, dt, timings)` if a thermostat
     is active (scalar prep only — no per-particle rescale)
   - `barostat.apply(buffers, sim_box, dt, timings)` if a barostat
     is active (scalar prep + box mutation — no per-particle
     rescale)
   - The composed JIT post-force per-particle kernel launch: the
     runner pre-populates a `ForceLaunchBuilder` with the common
     args (positions, images, velocities, forces, masses, device
     lattice), then calls each active slot's
     `bind_post_force_per_particle_args(...)` in canonical order
     (integrator → thermostat → barostat), then pushes the
     trailing `n` arg, then issues one `cuLaunchKernel`.
   - The displacement-check kernel
     `neighbor_displacement_check_flag` launched by
     `force_field.step_no_neighbor_check` after the post-force
     per-particle kernel. The kernel reads the now-updated positions
     against `reference_positions_*` and sets
     `cl.disp_rebuild_flag` to `1u` via `atomicOr` if any atom's
     minimum-image displacement exceeds `r_skin / 2`. The flag is
     sticky across replays of the captured graph until cleared by
     the host between batches.
5. The runner calls `device.end_stream_capture()` to obtain a
   `CudaGraph`.
6. The runner instantiates the executable graph via
   `CudaGraph::instantiate()` to obtain a `CudaGraphExec`. The
   instantiated graph is stored as the phase's `GraphLoop`.

The captured iteration counts as physical step 1; the timestep loop
replays from step 2 onward.

If any of begin-capture / end-capture / instantiate returns a CUDA
driver error, the runner logs a single line to stderr of the form
`warning: cuda graph capture failed for phase `<name>`: <reason>;
falling back to per-step launches` and runs the entire phase via the
per-step launch loop with full `Timings`. The fallback path completes
the run normally; the warning is informational.

## Batched Replay Loop <!-- rq-76db55bb -->

For an eligible phase with `[simulation].graph_batch_size = K`, the
per-phase loop has the shape:

```text
step = 1                                  // captured iteration is step 1
remaining = P.n_steps - 1
while remaining > 0:
    next_log    = if log_every  > 0 then log_every  - (step % log_every)  else remaining
    next_traj   = if traj_every > 0 then traj_every - (step % traj_every) else remaining
    batch = min(K, remaining, next_log, next_traj)

    for _ in 0..batch:
        graph_loop.launch(stream)

    step      += batch
    remaining -= batch

    nl.pre_step(sim_box, buffers, timings)              // 4-byte dtoh of disp_rebuild_flag; rebuild iff non-zero
    if step % traj_every == 0:
        sim_box.flush_from_device()
        download positions (and velocities when configured)
        write trajectory frame
    if step % log_every == 0:
        sim_box.flush_from_device()
        thermostat.flush_pending_injection()
        barostat.flush_pending_injection()
        download velocities; compute KE / T; compute PE if needed
        write log row
```

`graph_batch_size` is a phase-independent host parameter. Output
cadences (`log_every`, `trajectory_every`) shrink the effective batch
when they are not multiples of `graph_batch_size`. The captured graph
itself is always one physical step.

Every per-batch `nl.pre_step` synchronises against the device only
via a single 4-byte `dtoh_sync_copy` of `disp_rebuild_flag`. When the
flag is zero (the common case at typical liquid-MD displacement
rates and the default `K = 50` cadence) no further host work happens.
When the flag is non-zero, the rebuild pipeline runs synchronously,
the reference positions are refreshed, and `disp_rebuild_flag` is
zeroed via a single `memset_zeros` before the next batch's first
graph launch.

### Skin-distance contract under batched replay <!-- rq-b57700e0 -->

The neighbor-list rebuild trigger fires when any particle's
displacement from its last-reference position exceeds `r_skin / 2`
*at any captured step inside the batch*. The displacement-check
kernel runs every step inside the captured graph and writes
`disp_rebuild_flag = 1u` via `atomicOr` the first time a step's
particle exceeds the threshold; the flag is sticky until the host
clears it. With `graph_batch_size = K` the host consults the flag
once per K steps; in the worst case a particle covers
`K * max_step_displacement` before the host acts on the trigger and
runs a rebuild on the next batch boundary. The skin-distance
contract therefore holds when

```
K * max_step_displacement < r_skin / 2.
```

At the typical setting `r_skin = 0.3 * r_cut` and `K = 50`, the
per-step displacement bound is `0.003 * r_cut`. Liquid MD at room
temperature with `r_cut ≈ 9 Å` and `dt = 1 fs` rarely exceeds
`0.001 * r_cut` per step, leaving a 3× safety margin at the default
batch size.

If users tune `K` or `r_skin` outside the safe regime the skin
contract degrades silently: particles may drift past `r_skin / 2`
between checks and contribute to neighbor-list misses. The
configuration accepts any positive `K`; this is a tuning
responsibility, not a runtime guard.

## Neighbor-List Pre-Step Decomposition <!-- rq-011b7cea -->

`ForceField::step_no_neighbor_check` performs the same per-slot
compute as `ForceField::step` but skips the internal
`NeighborListState::pre_step` invocation. The runner is responsible
for calling `nl.pre_step` at every batch boundary instead.

`ForceField::step` (the un-prefixed variant) continues to call
`nl.pre_step` internally; minimization phases, per-step-launch-loop
phases (graph-ineligible), and the warm-up force evaluation continue
to use the un-prefixed variant.

A rebuild triggered by `nl.pre_step` updates the cell-list buffers in
place. The captured graph references those buffers by device pointer
only; their contents change but the pointers do not. The captured
graph remains valid across rebuilds without re-capture.

If a buffer that the graph references is reallocated (currently this
happens only when `max_neighbors` overflows, which halts the
simulation), the existing `CudaGraphExec` is dropped and the phase
falls back to per-step launches for its remaining steps. The
per-step path runs the same kernel sequence with the same
determinism guarantee; the only loss is the driver-overhead
elimination for the rest of that phase.

## `Timings` Interaction <!-- rq-9ec19227 -->

When graph mode is active for a phase:

- The captured graph contains kernel-launch nodes only. The
  `Timings::kernel_start` / `Timings::kernel_stop` calls inside the
  dry capture iteration emit one event-record pair per kernel stage;
  every subsequent replay step contributes zero further event pairs.
  Per-kernel rows in the phase's `.timings` file therefore show one
  sample regardless of `n_steps`.
- Aggregate per-phase total wall time and per-phase host stages are
  recorded normally. The per-batch host calls (`nl.pre_step`,
  `flush_pending_injection`, output writes) go through the existing
  host `Timings` stages.
- The `total_runtime` per-phase sample reflects end-to-end phase
  wall-clock and is comparable between graph-mode and non-graph-mode
  runs.

A user who needs a full per-kernel profile sets
`[simulation].cuda_graphs_disable = true` and re-runs. The per-step
launch loop produces per-step samples for every `KernelStage`.

## Configuration <!-- rq-006bc38c -->

`[simulation]` schema fields:

- `graph_batch_size: u32` (optional, default `50`) — number of step
  replays between displacement-flag downloads and output-cadence
  re-evaluations. Must be `>= 1`. The displacement-check *kernel*
  runs every step inside the captured graph regardless of this
  value; raising the batch size lowers the per-batch flag-download
  rate without changing the per-step displacement bookkeeping.
  Setting `graph_batch_size = 1` adds one `cuGraphLaunch` per step on
  top of the existing kernel sequence; it is slower than non-graph
  mode and is intended for diagnostic use only.
- `cuda_graphs_disable: bool` (optional, default `false`) — when
  `true`, every MD phase runs the per-step launch loop with full
  per-kernel `Timings`. Provided as a diagnostic escape hatch for
  graph-related issues.

Both fields are validated at config load. `graph_batch_size = 0` is
rejected as `ConfigError::InvalidValue { field:
"simulation.graph_batch_size", reason: "value must be >= 1, got 0" }`.

## Feature API <!-- rq-391a7d23 -->

### Types <!-- rq-38ce8ffa -->

- `CudaGraph` — RAII wrapper around `cudarc::driver::sys::CUgraph`. <!-- rq-2c1b569c -->
  Drop calls `cuGraphDestroy`. Carries:
  - `instantiate(&self) -> Result<CudaGraphExec, GraphError>` —
    invokes `cuGraphInstantiateWithFlags` with
    `CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 0`.

- `CudaGraphExec` — RAII wrapper around <!-- rq-9298b4b5 -->
  `cudarc::driver::sys::CUgraphExec`. Drop calls `cuGraphExecDestroy`.
  Carries:
  - `launch(&self, stream: &CudaStream) -> Result<(), GraphError>` —
    invokes `cuGraphLaunch`.

- `GraphLoop` — phase-owned executable graph + replay state. Carries: <!-- rq-6887c76d -->
  - `exec: CudaGraphExec` — the instantiated graph for one physical
    step.
  - `batch_size: u32` — the phase's `graph_batch_size`.
  - `launch(&self, stream: &CudaStream) -> Result<(), GraphError>` —
    forwards to `exec.launch(stream)`.

- `GraphError` — error type for graph capture / instantiate / launch. <!-- rq-5026f499 -->
  Variants:
  - `BeginCaptureFailed(DriverError)` — `cuStreamBeginCapture_v2`
    returned an error.
  - `EndCaptureFailed(DriverError)` — `cuStreamEndCapture` returned
    an error or returned an empty graph.
  - `InstantiateFailed(DriverError)` — `cuGraphInstantiateWithFlags`
    returned an error.
  - `LaunchFailed(DriverError)` — `cuGraphLaunch` returned an error.
  - `DestroyFailed(DriverError)` — `cuGraphDestroy` or
    `cuGraphExecDestroy` returned an error.

### Functions <!-- rq-126eba00 -->

- `CudaDevice::begin_stream_capture(mode: StreamCaptureMode) -> <!-- rq-a1d555ec -->
  Result<(), GraphError>` — wraps `cuStreamBeginCapture_v2` on the
  device's default stream. `StreamCaptureMode` is the safe analogue
  of `CUstreamCaptureMode` and exposes `Global`, `ThreadLocal`, and
  `Relaxed`.
- `CudaDevice::end_stream_capture() -> Result<CudaGraph, GraphError>` <!-- rq-46e415c0 -->
  — wraps `cuStreamEndCapture` on the device's default stream.
- `capture_phase_graph(setup: &mut SimulationSetup, phase: &Phase, <!-- rq-e35fa835 -->
  integrator: &mut dyn Integrator, thermostat: Option<&mut dyn
  Thermostat>, barostat: Option<&mut dyn Barostat>, constraint:
  Option<&mut dyn Constraint>, timings: &mut Timings) ->
  Result<Option<GraphLoop>, GraphError>` — runs the five-step capture
  procedure described under *Capture Lifecycle*. Returns `Ok(None)`
  when any of the supplied slots reports `graph_compatible = false`
  or when `[simulation].cuda_graphs_disable = true`. Returns
  `Err(...)` only when a CUDA driver call fails during capture or
  instantiate.

### Slot Eligibility Hooks <!-- rq-b2e5e90c -->

- `IntegratorBuilder::graph_compatible(&self, params: &toml::Value) <!-- rq-f84229ac -->
  -> bool` — default `true`. Implementations override to `false`
  when the slot's per-step plan executor reads device state into
  host scalars or mutates host fields between sub-steps.
- `ThermostatBuilder::graph_compatible(&self, params: &toml::Value) <!-- rq-1aa94cd6 -->
  -> bool` — default `true`. Same opt-out criteria.
- `BarostatBuilder::graph_compatible(&self, params: &toml::Value) -> <!-- rq-cf4a2e05 -->
  bool` — default `true`. Same opt-out criteria.
- `ConstraintBuilder::graph_compatible(&self, params: &toml::Value) <!-- rq-6bbf6545 -->
  -> bool` — default `true`. Same opt-out criteria.

### `ForceField` <!-- rq-6e82b441 -->

- `ForceField::step_no_neighbor_check(buffers: &mut ParticleBuffers, <!-- rq-2e53772f -->
  sim_box: &SimulationBox, timings: &mut Timings, level:
  AggregateLevel) -> Result<(), ForceFieldError>` — same per-slot
  compute path as `ForceField::step`, but skips the internal
  `NeighborListState::pre_step` call. Used inside graph capture and
  inside the batched replay loop.

### Per-slot device counters <!-- rq-753cce64 -->

Each RNG-using slot grows a one-element device buffer field:

- `CsvrThermostat::draw_counter_device: CudaSlice<u64>` <!-- rq-6c5b63e6 -->
- `AndersenThermostat::draw_counter_device: CudaSlice<u64>` <!-- rq-2c6de27d -->
- `LangevinBaoabIntegrator::draw_counter_device: CudaSlice<u64>` <!-- rq-47b7bed9 -->
- `CRescaleBarostat::draw_counter_device: CudaSlice<u64>` <!-- rq-53620d2c -->

Each slot's corresponding kernel signature carries a `unsigned long
long *draw_counter` pointer argument in place of the current scalar.
The host-side `draw_counter` field on each slot becomes a cached
value drained by `flush_pending_injection`.

## Gherkin Scenarios <!-- rq-9320c9d4 -->

```gherkin
Feature: CUDA graph capture and replay

  Background:
    Given a CUDA driver that supports stream capture
    And [simulation].cuda_graphs_disable = false
    And [simulation].graph_batch_size = 50

  @rq-acc595b8
  Scenario: Eligible phase captures a graph at phase start
    Given an MD phase with integrator "velocity-verlet" and thermostat "csvr"
    When the runner enters the phase
    Then nl.pre_step is called once before begin_stream_capture
    And begin_stream_capture is called on the default stream with CaptureMode::Global
    And one physical step's worth of kernels is launched on the default stream
    And end_stream_capture returns a CudaGraph
    And CudaGraph::instantiate returns a CudaGraphExec
    And the GraphLoop is stored on the phase

  @rq-1a85bb52
  Scenario: Captured iteration counts as physical step 1
    Given a phase with n_steps = 100 enters graph mode
    When the timestep loop runs to completion
    Then cuGraphLaunch is invoked 99 times in total across batches

  @rq-accf1a4b
  Scenario: Per-kernel timings empty in graph mode
    Given an eligible MD phase runs to completion in graph mode
    When the phase's .timings file is written
    Then every per-kernel KernelStage row reports 1 sample
    And the per-phase total_runtime row is populated normally

  @rq-882db733
  Scenario: Full per-kernel timings on cuda_graphs_disable
    Given cuda_graphs_disable = true and an otherwise-eligible MD phase
    When the phase runs n_steps = 100
    Then every per-kernel KernelStage row reports 100 samples

  @rq-6ae261b5
  Scenario: mtk-npt phase falls back to per-step launches
    Given an MD phase with integrator "mtk-npt"
    When the runner enters the phase
    Then no graph is captured
    And the per-step launch loop runs with full Timings
    And no warning is logged

  @rq-6f09d7e3
  Scenario: nose-hoover-chain phase falls back to per-step launches
    Given an MD phase with thermostat "nose-hoover-chain"
    When the runner enters the phase
    Then no graph is captured
    And the per-step launch loop runs with full Timings

  @rq-dadec448
  Scenario: Capture-time CUDA driver error falls back gracefully
    Given an eligible phase whose dry iteration triggers a non-captureable operation
    When end_stream_capture returns CUDA_ERROR_STREAM_CAPTURE_INVALIDATED
    Then the runner logs "warning: cuda graph capture failed for phase `<name>`: <reason>; falling back to per-step launches"
    And the phase runs the per-step launch loop
    And the run completes with exit code 0

  @rq-4c0ddae3
  Scenario: Two graph-mode runs are byte-identical
    Given two runs of the same config with cuda_graphs_disable = false
    When both runs complete
    Then both phase log files compare byte-identical
    And both phase trajectory files compare byte-identical

  @rq-e954f09e
  Scenario: Graph-mode and non-graph-mode runs are byte-identical (GPU)
    Given a config with seed S
    When run A sets cuda_graphs_disable = false
    And run B sets cuda_graphs_disable = true
    Then run A and run B produce byte-identical phase log files
    And run A and run B produce byte-identical phase trajectory files

  @rq-b4f36b2a
  Scenario: Log cadence shrinks the effective batch
    Given graph_batch_size = 5 and log_every = 3 and traj_every = 0
    When the timestep loop runs 10 steps
    Then the runner issues batches of sizes 2, 3, 3, 2 (total 10)
    And log rows are written at steps 3, 6, 9

  @rq-794e4d2e
  Scenario: Trajectory cadence shrinks the effective batch
    Given graph_batch_size = 5 and log_every = 0 and traj_every = 4
    When the timestep loop runs 10 steps
    Then the runner issues batches of sizes 3, 4, 3 (total 10)
    And trajectory frames are written at steps 4, 8

  @rq-1c8a6d37
  Scenario: nl.pre_step is called once per batch boundary
    Given graph_batch_size = 5
    When the timestep loop runs 25 steps
    Then nl.pre_step is called 5 times outside the captured graph
    And nl.pre_step is never called inside the captured graph

  @rq-813b7e0f
  Scenario: Rebuild without buffer reallocation does not invalidate the graph
    Given an eligible phase running in graph mode
    When nl.pre_step rebuilds the neighbor list without changing max_neighbors
    Then the existing CudaGraphExec is reused without re-capture
    And subsequent step replays produce the same kernel sequence in the same order
    And subsequent step replays produce bit-identical results to a non-rebuild reference

  @rq-3c62b49b
  Scenario: graph_batch_size = 1 is valid and runs every step under graph mode
    Given graph_batch_size = 1
    When the runner enters an eligible phase
    Then every physical step incurs exactly one cuGraphLaunch
    And nl.pre_step is called every physical step

  @rq-bac7d92d
  Scenario: graph_batch_size = 0 rejected at config load
    When a config sets graph_batch_size = 0
    Then config load returns ConfigError::InvalidValue with field "simulation.graph_batch_size" and reason "value must be >= 1, got 0"

  # --- Device-side displacement check ---

  @rq-59bbfa07
  Scenario: Captured graph includes the displacement-check kernel
    Given an eligible MD phase enters graph capture
    When the captured kernel sequence is enumerated
    Then neighbor_displacement_check_flag appears exactly once per captured step
    And its launch is recorded after the post-force per-particle kernel

  @rq-faf1dd2e
  Scenario: Per-batch host work is a single 4-byte download
    Given an eligible phase running in graph mode with graph_batch_size = 50
    And no log_every or traj_every output is due at this batch boundary
    When the batch completes its 50 graph launches
    Then nl.pre_step issues exactly one dtoh_sync_copy of length 1 (u32) against disp_rebuild_flag
    And no host-device particle transfer is performed at this batch boundary

  @rq-c4cc1d99
  Scenario: Quiescent batch incurs no rebuild
    Given an eligible phase in which no particle exceeds r_skin / 2 across any of the 50 captured replays
    When the batch completes
    Then disp_rebuild_flag downloaded by nl.pre_step is 0u
    And nl.pre_step performs no cell-list rebuild
    And reference_positions_{x,y,z} are unchanged

  @rq-f4069c16
  Scenario: Triggered batch rebuilds exactly once
    Given an eligible phase in which at least one particle exceeds r_skin / 2 on some captured replay inside the batch
    When the batch completes
    Then disp_rebuild_flag downloaded by nl.pre_step is 1u
    And nl.pre_step performs exactly one cell-list rebuild
    And disp_rebuild_flag is zeroed via memset_zeros before the next batch's first graph launch

  @rq-151a7e82
  Scenario: Default graph_batch_size is 50
    Given a config without [simulation].graph_batch_size
    When config load completes
    Then simulation.graph_batch_size resolves to 50

  @rq-6caca2f6
  Scenario: Skin contract holds for default K and typical liquid MD displacement rates
    Given graph_batch_size = 50
    And r_skin = 0.3 * r_cut
    And max_step_displacement <= 0.001 * r_cut
    When the timestep loop runs
    Then K * max_step_displacement <= 0.05 * r_cut < r_skin / 2 = 0.15 * r_cut

  @rq-2333f6af
  Scenario: cuda_graphs_disable overrides slot eligibility
    Given an eligible MD phase
    And cuda_graphs_disable = true
    When the runner enters the phase
    Then no graph is captured
    And no graph-capture warning is logged

  @rq-60c3085f
  Scenario: RNG draw counter is device-resident
    Given a CSVR thermostat slot
    When the slot is constructed
    Then draw_counter_device is allocated as a 1-element CudaSlice<u64>
    And the kernel reads-and-increments draw_counter_device in place

  @rq-879395e8
  Scenario: Replays advance the device counter once per step
    Given a CSVR thermostat slot in graph mode at the start of a phase
    When the captured graph is launched 10 times
    Then draw_counter_device holds the value 10
    And each replay produced a distinct Philox draw sequence

  @rq-871ebfef
  Scenario: Per-slot RNG matches between graph and non-graph modes
    Given a phase with a CSVR thermostat and a c-rescale barostat and seed S
    When run A executes the phase in graph mode
    And run B executes the same phase with cuda_graphs_disable = true
    Then every Philox sample drawn by run A equals the corresponding sample in run B
    And both runs produce byte-identical phase log files

  @rq-2c941abf
  Scenario: Conserved-quantity log columns match across modes
    Given a phase with a c-rescale barostat
    When the same phase runs once in graph mode and once with cuda_graphs_disable = true
    Then the cumulative_barostat_injection log column matches at every log row

  @rq-68bdda7c
  Scenario: Per-step launch loop unaffected by Phase 3 plumbing
    Given a config with cuda_graphs_disable = true and an integrator + thermostat + barostat all reporting graph_compatible = true
    When the phase runs to completion
    Then ForceField::step is called every physical step
    And NeighborListState::pre_step is invoked from inside ForceField::step every physical step
    And every per-kernel KernelStage row reports n_steps samples

  @rq-53db82b9
  Scenario: Custom external slot defaults to graph_compatible = true
    Given an out-of-tree integrator that does not override graph_compatible
    When the runner inspects the slot's builder
    Then graph_compatible returns true

  @rq-5f4fc894
  Scenario: Custom external slot that does host arithmetic disables itself
    Given an out-of-tree integrator whose execute() reads dtoh into a host field between sub-steps
    When the builder overrides graph_compatible to return false
    Then phases using the slot run the per-step launch loop with full Timings

  # --- JIT-composed post-force kernel in capture ---

  @rq-0b8e5852
  Scenario: Composed post-force kernel is compiled before capture begins
    Given an eligible MD phase with VelocityVerlet + CSVR + c-rescale
    When the runner enters the phase
    Then JitComposedPostForcePerParticle::compile_and_load is invoked
      before begin_stream_capture
    And the composed kernel's CudaFunction handle is bound to phase-local state

  @rq-8b964ce3
  Scenario: Captured graph contains exactly one composed post-force launch per step
    Given an eligible MD phase with VelocityVerlet + CSVR + c-rescale
    When the captured graph is replayed N times
    Then the device has issued exactly N cuLaunchKernel calls for
      heddle_jit_composed_post_force_per_particle
    And the device has issued zero cuLaunchKernel calls for
      vv_kick, csvr_rescale_velocities, c_rescale_barostat_rescale_positions

  @rq-f917104b
  Scenario: Standalone post-force per-particle kernels are not declared
    Given the project's kernel source tree
    When the kernel symbols are enumerated
    Then no extern "C" kernel named vv_kick exists
    And no extern "C" kernel named vv_kick_lossless exists
    And no extern "C" kernel named csvr_rescale_velocities exists
    And no extern "C" kernel named rescale_velocities_device_factor exists
    And no extern "C" kernel named rescale_positions_device_factor exists
    And no extern "C" kernel named berendsen_rescale_velocities exists
    And no extern "C" kernel named berendsen_barostat_rescale_positions exists
    And no extern "C" kernel named andersen_resample exists
    But extern "C" kernel named rescale_velocities continues to exist
      (used by NHC apply_pre and MTK particle-chain pre-force rescale)
    And extern "C" kernel named mtk_velocity_half_kick continues to exist
      (used by MTK pre-force vel_kick_pre)

  @rq-3d84a5b8
  Scenario: Built-in slot returning None is rejected at phase setup
    Given an MD phase configured with a user-registered integrator
      whose post_force_per_particle_fragment returns None
    When the runner enters the phase
    Then phase setup returns Err(StepError::MissingPostForcePerParticleFragment
      { kind: "integrator", label: <slot's label> })
    And no graph capture is attempted
    And no per-step launch loop is entered

  @rq-0bc3a66e
  Scenario: Graph mode and per-step mode use the same composed kernel
    Given a phase with VelocityVerlet + CSVR + c-rescale and seed S
    When run A executes the phase in graph mode
    And run B executes the phase with cuda_graphs_disable = true
    Then both runs invoke heddle_jit_composed_post_force_per_particle
      with the same launch config and the same argument list per step
    And both runs produce byte-identical phase log files
    And both runs produce byte-identical phase trajectory files

  @rq-b154f270
  Scenario: NHC apply_post writes a single cumulative factor to factor_device
    Given a phase with NHC thermostat configured with n_yoshida=3 and n_resp=2
    When apply_post runs once
    Then the NHC chain integrates host-side over 3*2 = 6 Yoshida × n_resp iterations
    And exactly one htod_sync_copy_into writes to factor_device
    And no rescale_velocities launch is issued from inside apply_post
    And factor_device[0] equals the host-computed product of the 6 per-iteration factors

  @rq-6bd00f49
  Scenario: Berendsen thermostat apply_post writes factor_device via on-device compute
    Given a phase with Berendsen thermostat
    When apply_post runs once
    Then compute_kinetic_energy_on_device writes ke_scratch
    And berendsen_compute_factor reads ke_scratch and writes factor_device
    And no host-side dtoh of kinetic energy occurs
    And no rescale_velocities launch is issued from inside apply_post

  @rq-6dc9fc6d
  Scenario: Andersen fragment is self-contained for Philox draw
    Given a phase with Andersen thermostat
    When the composed kernel source is generated
    Then the Andersen fragment's helper_source declares a __device__
      Philox draw routine
    And the per-thread body performs a Bernoulli draw against p_collision
    And the per-thread body branches into a Maxwell-Boltzmann resample
      when the Bernoulli draw selects the particle
    And the bind method pushes draw_counter_device, seed, p_collision, kT
      onto the launch builder

  @rq-a50d8a1f
  Scenario: Berendsen barostat apply writes mu_device via on-device compute
    Given a phase with Berendsen barostat
    When apply runs once
    Then compute_total_virial_on_device writes virial_scratch
    And a device kernel berendsen_barostat_compute_mu reads virial_scratch
      and writes mu_device
    And the lattice is mutated in-place on the device
    And no rescale_positions launch is issued from inside apply

  @rq-91c02dd8
  Scenario: NHC phase falls back to per-step launches but still uses the composed kernel
    Given an MD phase with NHC thermostat
    When the runner enters the phase
    Then no graph is captured (graph_compatible = false)
    And the per-step launch loop runs with full Timings
    And the composed JIT post-force kernel is launched once per physical step
    And no standalone per-particle post-force kernel is launched

  @rq-c0548f4c
  Scenario: MTK-NPT phase falls back to per-step launches but still uses the composed kernel
    Given an MD phase with MTK-NPT integrator
    When the runner enters the phase
    Then no graph is captured (graph_compatible = false)
    And the per-step launch loop runs with full Timings
    And the composed JIT post-force kernel is launched once per physical step
    And no standalone per-particle post-force kernel is launched

  @rq-8a66232e
  Scenario: Slot apply methods perform scalar prep only
    Given an MD phase with CSVR thermostat and c-rescale barostat
    When apply_post and apply each run once (in graph capture or per-step mode)
    Then CSVR's apply_post launches compute_kinetic_energy_on_device
      and csvr_sample_and_factor only
    And c-rescale's apply launches compute_kinetic_energy_on_device,
      compute_total_virial_on_device, and c_rescale_compute_mu only
    And neither method launches any per-particle rescale kernel

  @rq-d638d799
  Scenario: No set_jit_composed_post_force_active trait method exists
    Given the Thermostat / Barostat / Integrator trait surfaces
    When the runtime is inspected
    Then no trait method named set_jit_composed_post_force_active is declared
    And slot behaviour is single-mode: apply methods always do scalar prep only
```
