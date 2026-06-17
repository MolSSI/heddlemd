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

When any condition fails the phase runs the per-step launch loop
described in `simulation-runner.md` step 17, with full per-kernel
`Timings`.

Graph mode is the default for eligible phases. There is no per-phase
opt-in: eligibility is the activation criterion.

### Slot eligibility <!-- rq-26c9b8cb -->

Each in-tree slot's builder reports its `graph_compatible` value:

| Kind        | Slot                    | `graph_compatible` |
|-------------|-------------------------|--------------------|
| Integrator  | `velocity-verlet`       | `true`             |
| Integrator  | `langevin-baoab`        | `true`             |
| Integrator  | `mtk-npt`               | `false`            |
| Thermostat  | `csvr`                  | `true`             |
| Thermostat  | `andersen`              | `true`             |
| Thermostat  | `berendsen`             | `true`             |
| Thermostat  | `nose-hoover-chain`     | `false`            |
| Barostat    | `c-rescale`             | `true`             |
| Barostat    | `berendsen`             | `true`             |
| Constraint  | `shake`                 | `true`             |
| Constraint  | `settle`                | `true`             |

`mtk-npt` and `nose-hoover-chain` carry host-side scalar arithmetic
inside their per-step plan executors (the MTK chain variable `eps`,
the NHC Yoshida chain integration over `xi` / `p_xi`); a captured
graph cannot reproduce those host steps. They remain on the per-step
launch path until their host arithmetic is ported to device kernels.

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

## Capture Lifecycle <!-- rq-766c88fb -->

Each eligible phase captures one graph after `simulation-runner.md`
step 15 (warm-up force evaluation) and step 16 (step-0 outputs), and
before step 17 (timestep loop). The capture happens in this
sequence:

1. The runner calls `nl.pre_step(sim_box, buffers, timings)` once.
   This runs the displacement check (a host dtoh of `disp_sq` followed
   by a host max + comparison) and rebuilds the neighbor list if
   triggered. The call happens outside any graph capture and is not
   recorded.
2. The runner calls
   `device.begin_stream_capture(CaptureMode::Global)` on the default
   stream.
3. The runner executes the kernel sequence for one physical step,
   using `force_field.step_no_neighbor_check(...)` in place of the
   ordinary `force_field.step(...)`. The sequence is:
   - `thermostat.apply_pre(buffers, dt, timings)` if a thermostat is
     active
   - `run_step(integrator, buffers, sim_box, force_field,
     constraint, ..., dt, timings)`, where every internal
     `force_field.step` call is replaced by
     `force_field.step_no_neighbor_check`
   - `thermostat.apply_post(buffers, dt, timings)` if a thermostat
     is active
   - `barostat.apply(buffers, sim_box, dt, timings)` if a barostat
     is active
4. The runner calls `device.end_stream_capture()` to obtain a
   `CudaGraph`.
5. The runner instantiates the executable graph via
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

    nl.pre_step(sim_box, buffers, timings)              // displacement check
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

### Skin-distance contract under batched replay <!-- rq-b57700e0 -->

The neighbor-list rebuild trigger fires when any particle's
displacement from its last-reference position exceeds `r_skin / 2`.
With `graph_batch_size = K` the displacement check is invoked once
per K steps; in the worst case a particle covers `K *
max_step_displacement` before the next check. The skin-distance
contract holds when

```
K * max_step_displacement < r_skin / 2.
```

At the typical setting `r_skin = 0.3 * r_cut` and `K = 5`, the
per-step displacement bound is `0.03 * r_cut`. Liquid MD at room
temperature with `r_cut ≈ 9 Å` and `dt = 1 fs` rarely exceeds
`0.01 * r_cut` per step, leaving a 3× safety margin.

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

- `graph_batch_size: u32` (optional, default `5`) — number of step
  replays between displacement checks and output-cadence
  re-evaluations. Must be `>= 1`. Setting `graph_batch_size = 1`
  retains the per-step displacement-check cadence and adds one
  `cuGraphLaunch` per step on top of the existing kernel sequence;
  it is slower than non-graph mode and is intended for diagnostic
  use only.
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
    And [simulation].graph_batch_size = 5

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
```
