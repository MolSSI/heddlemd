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
- No thermostat is configured. A thermostat couples on a cadence and
  runs host-side arithmetic (chain integration, sampling) on coupling
  steps that stream capture cannot record, so a thermostatted phase runs
  on the per-step launch path (see *Batched Replay Loop*). Every
  post-force per-particle operation — the integrator's trailing kick and
  a per-step barostat's rescale — is a standalone kernel launch captured
  into the graph like any other launch; there is no composed post-force
  kernel and no per-slot fragment requirement.

- The integrator's plan has no `ThermostatHalf` sub-step and no
  *interleaved* (non-terminal) `BarostatPoint`. Both dispatch host-side
  slot arithmetic mid-plan, which stream capture cannot record, so such
  plans run on the eager path. A *terminal* `BarostatPoint` (the
  canonical placement the built-in integrators emit) keeps the phase
  eligible: the barostat's `apply` — including its standalone
  per-particle rescale — runs in the post-force tail and is captured
  like any other launch.

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
| Barostat    | `c-rescale`             | `true`             |
| Barostat    | `berendsen`             | `true`             |
| Barostat    | `monte-carlo`           | `true`             |
| Constraint  | `shake`                 | `true`             |
| Constraint  | `settle`                | `true`             |

Thermostats are absent from the table: a configured thermostat makes
the phase graph-ineligible regardless of kind (its coupling-step
arithmetic runs on the per-step path). `mtk-npt` reports
`graph_compatible == false` because it carries host-side chain
arithmetic (`eps`, `xi` / `p_xi`) inside its per-step plan executor
that a captured graph cannot reproduce; it runs on the per-step launch
path until that arithmetic is ported to device kernels.

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

## Post-Force Per-Particle Launches in the Captured Sequence <!-- rq-1db7cf2a -->

An eligible phase has no thermostat (a thermostatted phase runs on the
per-step path). Its post-force per-particle work is the integrator's
trailing kick (`vv_kick` / `vv_kick_lossless`, or a class kick) and,
when a per-step barostat is configured, the barostat's standalone
position rescale (`rescale_positions_device_factor`). Each is an
ordinary `cuLaunchKernel` recorded as its own node in the captured
graph; replays re-issue them exactly once per physical step. There is
no composed post-force kernel.

The non-graph per-step launch loop (`cuda_graphs_disable = true` or a
graph-ineligible phase) issues the identical launches directly per
step; both paths produce byte-identical results.

### Slot apply-method contract <!-- rq-781966e6 -->

Each slot's `apply_pre`, `apply_post`, and `apply` runs the slot's full
per-step work as standalone launches — reductions, device scalar
kernels, box mutation, per-step accumulator bookkeeping, and the
per-particle rescale — with no composed kernel. A per-step barostat's
`apply` therefore performs its own position rescale
(`rescale_positions_device_factor`) as its final launch. A thermostat's
`apply_post` performs its own velocity rescale, but only on the per-step
path — a thermostatted phase is graph-ineligible, so no thermostat
launch is ever captured.

## Capture Lifecycle <!-- rq-766c88fb -->

Each eligible phase captures one graph after `simulation-runner.md`
step 15 (warm-up force evaluation) and step 16 (step-0 outputs), and
before step 17 (timestep loop). The capture happens in this
sequence:

1. The runner calls `nl.pre_step(sim_box, buffers, timings)` once.
   `pre_step` downloads the single-word
   `neighbor_status` and rebuilds the neighbor list when the flag
   is non-zero (see `forces/neighbor-list.md` *Displacement Check*).
   Because `needs_rebuild` starts `true`, this first call always
   rebuilds; that probe rebuild sizes the packed-neighbour buffers to
   the system's true interaction count before any device pointer is
   captured (see `forces/packed-neighbour-pair-force.md` *Capacity*).
   The call happens outside any graph capture and is not recorded.
2. The runner captures the phase's executable graphs. Each graph is
   captured by calling
   `device.begin_stream_capture(CaptureMode::ThreadLocal)` on the
   default stream, recording one physical step's kernel sequence
   (step 3), calling `device.end_stream_capture()` to obtain a
   `CudaGraph`, and instantiating it with `CudaGraph::instantiate()`
   to obtain a `CudaGraphExec`. Stream capture records the kernel
   sequence without executing it, so capturing a graph advances no
   simulation state. Up to two graphs are captured from the same
   post-probe device state:
   - **forces+scalars graph** — the recorded sequence with the force
     evaluation at `AggregateLevel::ForcesAndScalars`: the `_fev`
     composed pair-force kernel plus the per-step potential-energy and
     virial reductions. Always captured.
   - **forces-only graph** — the same sequence with the force
     evaluation at `AggregateLevel::ForcesOnly`: the `_f` composed
     pair-force kernel, with the scalar reductions absent. Captured
     unless a *per-step* barostat is active for the phase.

   A per-step barostat consumes the per-step scalar virial inside the
   captured sequence (`barostat.apply` in step 3), so a per-step-barostat
   phase evaluates scalars on every step and captures only the
   forces+scalars graph. A periodic (Monte-Carlo) barostat puts no node
   in the captured sequence and consumes no per-step virial, so its phase
   captures both graphs exactly as an NVT phase does; its volume move runs
   on the host between batches (see *Periodic-barostat moves*). When no
   per-step barostat is active the per-particle force, position, and
   velocity results of the two graphs are bit-identical — the scalar
   reductions are diagnostic outputs that do not feed back into the
   dynamics — so replaying either graph for a given step produces the
   same trajectory.
3. The recorded one-step kernel sequence, run under capture once per
   graph with `force_field.step_no_neighbor_check(...)` in place of the
   ordinary `force_field.step(...)`, is (an eligible phase has no
   thermostat, so no `apply_pre` / `apply_post` appears):
   - `run_step(integrator, buffers, sim_box, force_field, constraint,
     None, barostat, dt, timings, RunStepOptions {
     run_neighbor_pre_step: false, runner_needs_scalars,
     coupling_dt: None })`, where `runner_needs_scalars` is `true` while
     recording the forces+scalars graph and `false` while recording the
     forces-only graph. `coupling_dt` is `None` because an eligible phase
     has no thermostat. `run_step` walks the **entire** plan — routing
     every `force_field.step` to `force_field.step_no_neighbor_check`,
     dispatching interleaved `ConstraintPoint` markers to the constraint
     slot (a no-op when the slot is `None`), the integrator's trailing
     kick (via `execute`), and then the post-force tail. See
     `integration/framework.md` for `RunStepOptions`.
   - The post-force tail, dispatched by the same `run_step` call, records
     each op as a standalone captured launch: `barostat.apply(buffers,
     sim_box, dt, timings)` when the plan carries a terminal
     `BarostatPoint` (its reductions, box mutation, and per-particle
     position rescale); then `constraint.apply_after_kick(buffers,
     sim_box, dt, timings)` when the plan ends with a velocity projection
     — the last per-particle velocity operation. The constraint kernels
     are pure launches (the SETTLE / SHAKE builders report
     `graph_compatible == true`), so a constrained phase is
     graph-eligible.
   - The displacement-check kernel
     `neighbor_displacement_check_flag` launched by
     `force_field.step_no_neighbor_check` after the post-force
     per-particle kernel. The kernel reads the now-updated positions
     against `reference_positions_*` and sets
     `cl.neighbor_status` to `1u` via `atomicOr` if any atom's
     minimum-image displacement exceeds `r_skin / 2`. The flag is
     sticky across replays of the captured graph until cleared by
     the host between batches.
4. Both instantiated graphs are stored in the phase's `GraphLoop`
   (the forces-only graph as an `Option`, absent for per-step-barostat
   phases).

Capture records but does not execute, so capturing the graphs
advances no simulation state; every physical step of the phase is
produced by a graph replay in the batched replay loop (apart from the
short instrumented per-step calibration run described under
*`Timings` Interaction*).

Force kernels whose result depends on the simulation box must read the
box from the persistent device lattice buffer (`sim_box.lattice_device()`)
and must launch **unconditionally** during the force evaluation, never
gated on a host-side state check. The barostat mutates that device buffer
in place (step 4 above), so a box-dependent kernel that is recorded into
the graph automatically tracks the per-replay box: on each replay it
reads the lattice the previous step's captured barostat kernel wrote. The
SPME reciprocal influence-function recompute (`spme_recip_compute_influence`,
see `forces/spme.md`) is the load-bearing case — it runs every force
evaluation and is captured into the graph. A box-dependent kernel that
instead gates its launch on a host-side counter (for example a box
generation) would be skipped at capture time, because the force
evaluation precedes the step's barostat update, so the host counter has
not yet advanced; the kernel would never be recorded and the captured
batch would run on a stale, box-independent result.

If any of begin-capture / end-capture / instantiate returns a CUDA
driver error, the runner logs a single line to stderr of the form
`warning: cuda graph capture failed for phase `<name>`: <reason>;
falling back to per-step launches` and runs the entire phase via the
per-step launch loop with full `Timings`. The fallback path completes
the run normally; the warning is informational.

## Batched Replay Loop <!-- rq-76db55bb -->

A thermostat couples on a cadence, and on coupling steps runs host-side
work (e.g. NHC chain integration) that cannot be captured; its full-step
kinetic-energy reduction is a fusion barrier (`framework.md`,
`docs/architecture.md`). A **thermostatted phase therefore runs on the
per-step launch path**, not the batched graph loop — it is
graph-ineligible. The coupling interval's benefit is still realised on
that path: on a non-coupling step (`step % coupling_interval != 0`) the
thermostat is inert (no reduction, no rescale) and only the integrator
kick and any barostat rescale run, while a coupling step runs the full
post-force region. Only a phase with no thermostat is eligible for the
batched graph loop.

For an eligible (thermostat-free) phase with
`[simulation].graph_batch_size = K`, the per-phase loop has the shape:

```text
needs_scalars(s) = (log_every > 0 and s % log_every == 0) or per_step_barostat_active
while remaining > 0:
    next_log    = if log_every  > 0 then log_every  - (step % log_every)  else remaining
    next_traj   = if traj_every > 0 then traj_every - (step % traj_every) else remaining
    next_move   = if periodic_barostat then frequency - (step % frequency) else remaining
    batch = min(K, remaining, next_log, next_traj, next_move)

    for i in 1..=batch:
        s = step + i
        graph_loop.launch(stream, scalars = needs_scalars(s))   // forces+scalars graph iff needs_scalars(s), else forces-only

    step      += batch
    remaining -= batch

    if periodic_barostat and step % frequency == 0:
        barostat.apply_move(force_field, buffers, sim_box, constraint, dt, timings)  // host-orchestrated MC volume move

    outcome = nl.pre_step(sim_box, buffers, timings)    // 4-byte dtoh of neighbor_status; rebuild iff non-zero
    if outcome.reallocated and remaining > 0:           // a rebuild grew (reallocated) a packed-neighbour buffer
        graph_loop = capture_phase_graph(...)           // re-capture against new pointers / grid dims; records, does not execute
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
cadences (`log_every`, `trajectory_every`) and a periodic barostat's
move cadence (`frequency`) shrink the effective batch when they are not
multiples of `graph_batch_size`. The captured graph itself is always one
physical step.

A step requires force-kernel scalars — the total potential energy and
virial — only when it produces a log row (the log row reports the
potential energy) or when a *per-step* barostat is active (it consumes
the per-step virial). This is the `needs_scalars(s)` predicate above. A
periodic (Monte-Carlo) barostat does not raise the predicate: it consumes
no per-step virial, and its host-side move evaluates its own energy
outside the captured graphs.
A trajectory frame carries positions and velocities, not force-kernel
scalars, and the thermostat's kinetic energy is reduced from
velocities independently of the force evaluation, so neither forces a
scalars step. The same predicate selects `AggregateLevel` in the
per-step launch loop (see `simulation-runner.md`), so the two loops
compute scalars on exactly the same steps.

Each replay therefore launches the forces-only graph unless
`needs_scalars(s)` holds, in which case it launches the
forces+scalars graph. Because a batch is bounded so it ends on the
next log or trajectory cadence boundary, the forces+scalars graph
runs at most once per batch — on the boundary step that feeds a log
row — and every other step in the batch runs forces-only. When a
*per-step* barostat is active every step requires scalars, the
forces-only graph is not captured, and every replay launches the
forces+scalars graph. A periodic (Monte-Carlo) barostat does not change
this scalar cadence; it only adds the `next_move` batch bound and the
host-side `apply_move` call at each move boundary.

Every per-batch `nl.pre_step` synchronises against the device only
via a single 4-byte `dtoh_sync_copy` of `neighbor_status`. When the
flag is zero (the common case at typical liquid-MD displacement
rates and the default `K = 50` cadence) no further host work happens.
When the flag is non-zero, the rebuild pipeline runs synchronously,
the reference positions are refreshed, and `neighbor_status` is
zeroed via a single `memset_zeros` before the next batch's first
graph launch.

When that synchronous rebuild grows a packed-neighbour buffer,
`nl.pre_step` returns `reallocated = true` and the runner re-captures
the phase graph before the next batch (see *Neighbor-List Pre-Step
Decomposition*). Stream capture records the kernel sequence without
executing it, so the re-capture consumes no physical step: the step
counter is unchanged and the phase still runs exactly `n_steps` steps.

### Periodic-barostat moves <!-- new --> <!-- rq-6e822476 -->

A periodic barostat (`Barostat::periodicity() == EveryNSteps(frequency)`,
the Monte-Carlo barostat) runs its move on the host at batch boundaries,
never inside a captured graph. The runner adds `next_move = frequency -
(step % frequency)` to the batch-bound minimum so a batch always ends on a
move boundary, and calls `barostat.apply_move(force_field, buffers,
sim_box, constraint, dt, timings)` after that batch's last replay and
before `nl.pre_step`. The move evaluates the force field at the pre-move
and trial configurations, performs a host-side Metropolis accept/reject,
and mutates (or restores) the device lattice and positions; its full
contract is in `integration/mc-barostat.md`.

Because the move mutates the device lattice through
`SimulationBox::set_lattice` / `multiply_lattice_isotropic`, the box
generation advances, and the next batch's first force evaluation rebuilds
the neighbor list and refreshes the SPME influence function against the
new box through the existing generation-change path — the same mechanism
the per-batch `nl.pre_step` rebuild already uses. The captured graphs hold
buffer pointers, not box values, so a move never invalidates them; no
re-capture is triggered by a move (only a packed-neighbour reallocation
re-captures, exactly as without a barostat).

A move performs host dtoh reads (the trial energy) and host branching, so
it is run only between batches on the per-step launch path's host side,
never under stream capture. The phase otherwise replays the forces-only
graph for every non-log step, which is the production benefit of the
periodic barostat over a per-step barostat.

### Skin-distance contract under batched replay <!-- rq-b57700e0 -->

The neighbor-list rebuild trigger fires when any particle's
displacement from its last-reference position exceeds `r_skin / 2`
*at any captured step inside the batch*. The displacement-check
kernel runs every step inside the captured graph and writes
`neighbor_status = 1u` via `atomicOr` the first time a step's
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

A rebuild triggered by `nl.pre_step` that does not change any buffer
length updates the cell-list and packed-neighbour buffers in place.
The captured graph references those buffers by device pointer only;
their contents change but the pointers do not, so the captured graph
remains valid across such rebuilds without re-capture.

A rebuild grows a packed-neighbour buffer (`interacting_tiles`,
`interacting_atoms`, or `single_pair_atoms`) when the combined
`neighbor_status` word the runner reads at the batch boundary carries
a `*_high_water` bit — the previous build came within
`tile_pair_fill_threshold` of capacity while dropping nothing (see
`forces/packed-neighbour-pair-force.md` *Capacity*). Growth reallocates
the buffer, invalidating both the device pointers the captured nodes
hold and the single-pair launch's captured grid dimensions.
`nl.pre_step` reports this through its outcome's `reallocated` flag
(see `forces/neighbor-list.md` *Rebuild Policy*). A `*_overflow` bit in
the same word means a build actually dropped within-`r_search` entries;
`nl.pre_step` then returns `PackedNeighborOverflow` and the run halts
rather than re-capturing. When the `reallocated` flag is set, the
runner re-captures the phase graph at that batch boundary:
it drops the current `GraphLoop`, runs `capture_phase_graph` again to
record a fresh one-step graph against the new pointers and grid
dimensions, and continues replaying in graph mode. Stream capture
records the kernel sequence without executing it, so the re-capture
consumes no physical step. Because growth is monotonic and each step
multiplies capacity by `tile_pair_growth_factor`, a phase incurs at
most a handful of re-captures, so the amortised cost is negligible.

This decouples the captured graph's buffer pointers and launch
dimensions — which need to be stable only for the lifetime of one
graph instance — from the phase lifetime. Buffers are therefore sized
to the actual `O(N)` interaction count rather than to the
`O(n_blocks²)` all-pairs bound that whole-phase pointer stability
would otherwise force.

If the re-capture itself fails with a CUDA driver error, the runner
logs the same `warning: cuda graph capture failed for phase
`<name>`: <reason>; falling back to per-step launches` line described
under *Capture Lifecycle* and finishes the phase on the per-step
launch loop with full `Timings`. The per-step path runs the same
kernel sequence with the same determinism guarantee; the only loss is
driver-overhead elimination for the rest of that phase.

## `Timings` Interaction <!-- rq-9ec19227 -->

CUDA forbids `cuEventElapsedTime` on an event that has been recorded
into a graph (it returns `CUDA_ERROR_INVALID_VALUE`), so the per-kernel
`start`/`stop` events captured into the phase graph cannot be timed by
replaying them. Graph-mode per-kernel timings are instead produced by a
short **calibration** before the replay loop.

When graph mode is active for a phase:

- **Calibration.** The runner executes the first
  `min(GRAPH_TIMING_CALIBRATION_STEPS, n_steps)` physical steps on the
  instrumented per-step launch path (live CUDA-event timing per
  `KernelStage`), then calls `Timings::snapshot_graph_representatives`,
  which records each stage's mean calibrated duration as its
  representative per-replay value. `GRAPH_TIMING_CALIBRATION_STEPS` is a
  small constant (8). The captured graph then covers the remaining
  steps. Because the per-step path is bit-identical to the graph replay
  (forces accumulate in order-invariant fixed point), running these
  steps per-step does not perturb the trajectory.
- **Replay accounting.** For each batch the runner records, per graph
  variant launched, the number of forces-only replays and the number of
  forces+scalars replays. `Timings::record_graph_replays` folds the
  representative duration into every stage's accumulator scaled by that
  graph's `captured_stops_per_replay × (its replay count)`. A stage that
  appears only in the forces+scalars graph — the potential-energy and
  virial reductions, and the `_fev` pair-force kernel — therefore
  accrues samples only from forces+scalars replays, so its `.timings`
  sample count tracks the number of scalar steps rather than `n_steps`.
  Stages present in both graphs carry a sample count equal to the total
  step count.
- **Representativeness.** The calibration measures the *first* few
  steps. For a phase already near steady state (e.g. NPT production) the
  per-kernel values match a full graphs-disabled run to within a few
  percent. For a phase with a strong early transient (e.g. NVT
  equilibration from a lattice, whose first steps carry more pair
  interactions) the absolute values can run ~10–20% above the phase
  mean; the *relative* per-kernel breakdown and ranking remain
  representative. A stage that never runs during calibration keeps a
  zero representative.
- Aggregate per-phase total wall time and per-phase host stages are
  recorded normally. The per-batch host calls (`nl.pre_step`,
  `flush_pending_injection`, output writes) go through the existing
  host `Timings` stages.
- The `total_runtime` per-phase sample reflects end-to-end phase
  wall-clock and is comparable between graph-mode and non-graph-mode
  runs. The handful of calibration steps add negligible cost.

A user who needs an exact per-step per-kernel profile sets
`[simulation].cuda_graphs_disable = true` and re-runs; the per-step
launch loop produces a real sample for every step of every
`KernelStage`.

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

- `GraphLoop` — phase-owned executable graphs + replay state. Carries: <!-- rq-6887c76d -->
  - `forces_and_scalars: CudaGraphExec` — the instantiated
    forces+scalars graph for one physical step. Always present.
  - `forces_only: Option<CudaGraphExec>` — the instantiated
    forces-only graph for one physical step. `Some` when no per-step
    barostat is active for the phase (including periodic Monte-Carlo
    barostat phases), `None` otherwise.
  - `batch_size: u32` — the phase's `graph_batch_size`.
  - `launch(&self, stream: &CudaStream, scalars: bool) -> Result<(),
    GraphError>` — forwards to `forces_and_scalars.launch(stream)`
    when `scalars` is `true` or when `forces_only` is `None`,
    otherwise to `forces_only`'s `launch(stream)`.

  The runner holds the phase's `GraphLoop` in a mutable binding so it
  can replace it mid-phase: when `nl.pre_step` reports a
  packed-neighbour reallocation, the runner runs `capture_phase_graph`
  again and stores the new `GraphLoop`, dropping the old one (whose
  `CudaGraphExec` releases the stale graph on `Drop`).

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
  Result<Option<GraphLoop>, GraphError>` — runs the capture procedure
  described under *Capture Lifecycle*, capturing the forces+scalars
  graph and, when no per-step barostat is active, the forces-only graph,
  and returning them in one `GraphLoop`. Returns `Ok(None)`
  when any of the supplied slots reports `graph_compatible = false`
  or when `[simulation].cuda_graphs_disable = true`. Returns
  `Err(...)` only when a CUDA driver call fails during capture or
  instantiate. The runner invokes it both at phase start and, in the
  batched replay loop, whenever a packed-neighbour buffer is
  reallocated mid-phase (see *Neighbor-List Pre-Step Decomposition*).
  Stream capture records the kernel sequence without executing it, so a
  re-capture consumes no physical step.

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
  Scenario: Graph replays cover every step after the calibration prefix
    Given a phase with n_steps = 100 enters graph mode
    And GRAPH_TIMING_CALIBRATION_STEPS = 8
    When the timestep loop runs to completion
    Then the first 8 steps run on the per-step launch path
    And cuGraphLaunch is invoked 92 times in total across batches, one launch per remaining step

  @rq-accf1a4b
  Scenario: Per-kernel timings are populated in graph mode
    Given an eligible MD phase runs to completion in graph mode
    When the phase's .timings file is written
    Then every per-kernel KernelStage row reports a non-zero representative duration
    And a stage present in both graphs reports a sample count equal to the step count
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

  @rq-dce6f4cf
  Scenario: A thermostatted phase is graph-ineligible and runs on the per-step path
    Given graph_batch_size = 5, log_every = 0, traj_every = 0
    And an active thermostat with coupling_interval = 4
    When the timestep loop runs 8 steps
    Then the phase runs on the per-step launch path (no phase graph is replayed)
    And on the six non-coupling steps the thermostat is inert (no reduction,
      no rescale) and the integrator's trailing kick runs standalone
    And the two coupling steps (4, 8) run the trailing kick standalone and
      the thermostat's reduction and rescale

  @rq-49f6bbfb
  Scenario: A thermostat coupling every step also runs on the per-step path
    Given graph_batch_size = 50 and an active thermostat with coupling_interval = 1
    When the phase runs
    Then every step runs on the per-step launch path (no phase graph is replayed)
    And every step is a coupling step; the thermostat's reduction and rescale
      run standalone each step

  @rq-1c8a6d37
  Scenario: nl.pre_step is called once per batch boundary
    Given graph_batch_size = 5
    When the timestep loop runs 25 steps
    Then nl.pre_step is called 5 times outside the captured graph
    And nl.pre_step is never called inside the captured graph

  @rq-813b7e0f
  Scenario: Rebuild without buffer reallocation does not invalidate the graph
    Given an eligible phase running in graph mode
    When nl.pre_step rebuilds the neighbor list without growing any
      packed-neighbour buffer
    Then nl.pre_step returns reallocated = false
    And the existing CudaGraphExec is reused without re-capture
    And subsequent step replays produce the same kernel sequence in the same order
    And subsequent step replays produce bit-identical results to a non-rebuild reference

  @rq-d5e451eb
  Scenario: Rebuild that grows a packed-neighbour buffer triggers re-capture
    Given an eligible phase running in graph mode
    When nl.pre_step rebuilds and grows interacting_tiles at a batch boundary
    Then nl.pre_step returns reallocated = true
    And the runner drops the current GraphLoop and calls capture_phase_graph again
    And the new GraphLoop's captured nodes reference the reallocated buffers
    And the phase continues replaying in graph mode

  @rq-c5116f45
  Scenario: Re-capture consumes no physical step
    Given an eligible phase of n_steps = 100 running in graph mode
    And exactly one batch boundary triggers a packed-neighbour reallocation
    When the timestep loop runs to completion
    Then the total number of physical steps executed is exactly 100
    And the re-capture replaces the GraphLoop without advancing the step counter

  @rq-f986ba18
  Scenario: Graph-mode run with a mid-phase re-capture is byte-identical to per-step
    Given a config with seed S whose phase grows a packed-neighbour buffer mid-phase
    When run A executes the phase in graph mode (with the re-capture)
    And run B executes the same phase with cuda_graphs_disable = true
    Then run A and run B produce byte-identical phase log files
    And run A and run B produce byte-identical phase trajectory files

  @rq-65c0327d
  Scenario: Failed re-capture falls back to per-step launches
    Given an eligible phase running in graph mode
    And a mid-phase reallocation whose capture_phase_graph call returns a CUDA driver error
    When the runner attempts the re-capture
    Then the runner logs "warning: cuda graph capture failed for phase `<name>`: <reason>; falling back to per-step launches"
    And the remaining steps of the phase run on the per-step launch loop with full Timings
    And the run completes with exit code 0

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
    Then nl.pre_step issues exactly one dtoh_sync_copy of length 1 (u32) against neighbor_status
    And no host-device particle transfer is performed at this batch boundary

  @rq-c4cc1d99
  Scenario: Quiescent batch incurs no rebuild
    Given an eligible phase in which no particle exceeds r_skin / 2 across any of the 50 captured replays
    When the batch completes
    Then neighbor_status downloaded by nl.pre_step is 0u
    And nl.pre_step performs no cell-list rebuild
    And reference_positions_{x,y,z} are unchanged

  @rq-f4069c16
  Scenario: Triggered batch rebuilds exactly once
    Given an eligible phase in which at least one particle exceeds r_skin / 2 on some captured replay inside the batch
    When the batch completes
    Then neighbor_status downloaded by nl.pre_step is 1u
    And nl.pre_step performs exactly one cell-list rebuild
    And neighbor_status is zeroed via memset_zeros before the next batch's first graph launch

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

  # --- Post-force per-particle launches in capture ---

  @rq-8b964ce3
  Scenario: Captured graph contains the standalone post-force launches per step
    Given an eligible MD phase with VelocityVerlet + c-rescale (no thermostat)
    When the captured graph is replayed N times
    Then the device has issued exactly N cuLaunchKernel calls for vv_kick
    And exactly N for c_rescale_barostat_rescale_positions
    And zero calls for any composed post-force kernel (none exists)

  @rq-f917104b
  Scenario: Post-force per-particle work is ordinary standalone kernels
    Given the project's kernel source tree
    When the kernel symbols are enumerated
    Then extern "C" kernels vv_kick, vv_kick_lossless, rescale_velocities,
      rescale_positions_device_factor, and andersen_resample all exist
    And no composed post-force per-particle entry point exists

  @rq-3d84a5b8
  Scenario: A user integrator that is graph-compatible keeps the phase eligible
    Given an MD phase with a user-registered integrator whose builder
      reports graph_compatible = true, and no thermostat
    When the runner enters the phase
    Then phase setup succeeds and the phase is eligible for graph capture
    And the integrator's trailing kick is captured as a standalone launch

  @rq-0bc3a66e
  Scenario: Graph mode and per-step mode issue the same launches
    Given a phase with VelocityVerlet + c-rescale (no thermostat) and seed S
    When run A executes the phase in graph mode
    And run B executes the phase with cuda_graphs_disable = true
    Then both runs issue the same standalone post-force launches per step
    And both runs produce byte-identical phase log and trajectory files

  @rq-91c02dd8
  Scenario: A thermostatted phase is graph-ineligible and runs per-step
    Given an MD phase with any thermostat
    When the runner enters the phase
    Then no graph is captured
    And the per-step launch loop runs with full Timings
    And the thermostat's reduction and rescale run as standalone launches
      on coupling steps

  @rq-c0548f4c
  Scenario: MTK-NPT phase runs per-step with standalone launches
    Given an MD phase with MTK-NPT integrator
    When the runner enters the phase
    Then no graph is captured (graph_compatible = false)
    And the per-step launch loop runs with full Timings
    And every post-force per-particle operation runs as a standalone launch

  @rq-d638d799
  Scenario: No composed post-force kernel or activation toggle exists
    Given the Thermostat / Barostat / Integrator trait surfaces
    When the runtime is inspected
    Then there is no post_force_per_particle accessor and no composed
      post-force kernel; every post-force pointwise op is a standalone launch

  # --- Forces-only / forces+scalars graph selection ---

  @rq-26dce0f6
  Scenario: A phase without a barostat captures both graphs
    Given an MD phase with a thermostat and no barostat
    When the runner captures the phase graphs
    Then the GraphLoop's forces_and_scalars graph is present
    And the GraphLoop's forces_only graph is Some

  @rq-c6c56cdc
  Scenario: A phase with a barostat captures only the forces+scalars graph
    Given an MD phase with a c-rescale barostat
    When the runner captures the phase graphs
    Then the GraphLoop's forces_and_scalars graph is present
    And the GraphLoop's forces_only graph is None

  @rq-867630af
  Scenario: GraphLoop.launch routes by the scalars flag
    Given a GraphLoop whose forces_only graph is Some
    When launch(stream, scalars = false) is called
    Then the forces_only graph is launched
    When launch(stream, scalars = true) is called
    Then the forces_and_scalars graph is launched

  @rq-8c24f057
  Scenario: GraphLoop.launch ignores the scalars flag when forces_only is None
    Given a GraphLoop whose forces_only graph is None
    When launch(stream, scalars = false) is called
    Then the forces_and_scalars graph is launched

  @rq-009eed1b
  Scenario: NVT graph phase evaluates force-kernel scalars only on log steps
    Given an MD phase with a thermostat, no barostat, n_steps = 100, log_every = 25, trajectory_every = 0
    When the phase runs to completion in graph mode
    Then the potential_energy_reduce KernelStage sample count equals the number of log steps, not n_steps

  @rq-d34ff8f7
  Scenario: Trajectory-only steps do not trigger force-kernel scalars in graph mode
    Given an NVT graph phase with trajectory_every = 10, log_every = 0, n_steps = 100
    When the phase runs to completion
    Then no replay launches the forces+scalars graph
    And the potential_energy_reduce KernelStage records zero samples from the replay loop

  @rq-1b40a671
  Scenario: NPT graph phase evaluates force-kernel scalars on every step
    Given an MD phase with a c-rescale barostat, n_steps = 100, log_every = 25
    When the phase runs to completion in graph mode
    Then every replayed step launches the forces+scalars graph

  @rq-dae13654
  Scenario: Forces-only replay does not change the trajectory
    Given an NVT phase with a thermostat, no barostat, and a disordered liquid configuration
    When the phase is run to completion in graph mode twice on the same GPU
    Then the two runs produce byte-identical per-particle positions and velocities

  # --- Per-step launch loop scalar cadence ---

  @rq-ed183041
  Scenario: Per-step loop computes ForcesOnly on a non-log non-barostat step
    Given the per-step launch loop on an NVT phase with log_every = 25 and no barostat
    When a step that is neither a log nor trajectory boundary is executed
    Then runner_needs_scalars is false and the force evaluation runs at AggregateLevel::ForcesOnly

  @rq-2af44cf4
  Scenario: Per-step loop computes ForcesAndScalars on a log step
    Given the per-step launch loop on an NVT phase with log_every = 25
    When a step on the log cadence boundary is executed
    Then runner_needs_scalars is true and the force evaluation runs at AggregateLevel::ForcesAndScalars

  @rq-091a4341
  Scenario: Per-step loop does not compute scalars on a trajectory-only step
    Given the per-step launch loop on an NVT phase with trajectory_every = 10, log_every = 0, no barostat
    When a step on the trajectory cadence boundary is executed
    Then runner_needs_scalars is false and the force evaluation runs at AggregateLevel::ForcesOnly
```
