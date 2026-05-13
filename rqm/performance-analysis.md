# Feature: Performance Analysis and Timings Output <!-- rq-bbb62e9c -->

The simulation runner collects per-stage timing data throughout every run and
writes a summary table to a `.timings` file alongside the trajectory and log.
Kernel launches are timed with CUDA events (submitted to the default stream so
they do not disturb GPU/host pipelining beyond the inherent kernel cost); host
operations are timed with `std::time::Instant` wall-clock measurements.

Timings are always collected — there is no opt-out in schema v1. The cost is
~1 µs of overhead per CUDA event, which is negligible compared to typical
kernel runtimes.

Timing data never enters the trajectory or log files; those remain bit-wise
reproducible across runs on the same GPU. The `.timings` file contents are
expected to vary run-to-run because wall-clock measurements are inherently
non-deterministic.

## Instrumented Stages <!-- rq-1bd297f4 -->

The runner records samples against a fixed set of stage names. Stages with
zero samples in a given run are omitted from the output file.

### Kernel stages (CUDA-event timed) <!-- rq-312e5f8e -->

One CUDA event pair per stage. The runner (or the integrator slot acting
through the runner) records the start event immediately before each kernel
launch and the stop event immediately after. Stages:

Lennard-Jones potential slot (always present in the `ForceField`):

- `lj_pair_force` — all-pairs kernel; recorded only when
  `NeighborListConfig::AllPairs` is selected.
- `lj_pair_force_neighbor` — neighbor-list-driven kernel; recorded
  only when `NeighborListConfig::CellList` is selected.
- `reduce_pair_forces`

Neighbor list (present when `NeighborListConfig::CellList` is selected;
see `forces/neighbor-list.md`):

- `neighbor_displacement_squared` — per-atom displacement² versus the
  reference positions. Launched once per timestep.
- `neighbor_list_build` — full cell-list neighbor search. Launched
  once per rebuild.
- `copy_positions_into_reference` — refresh of the reference positions
  array. Launched once per rebuild.

Morse-bonded potential slot (present when the config supplies a `bonds`
file and at least one bond exists):

- `morse_bond_force` — per-bond force kernel.
- `reduce_bond_forces` — per-atom reduction of bond contributions.

Force-field combiner (always run; sums every slot's private accumulator
into `particle_buffers.forces_*`):

- `accumulate_forces`

Velocity-Verlet integrator stages (selected by `kind = "velocity-verlet"`):

- `vv_kick_drift`
- `vv_kick`
- `vv_kick_drift_lossless`
- `vv_kick_lossless`

Langevin BAOAB integrator stages (selected by `kind = "langevin-baoab"`,
see `integration/langevin-baoab.md`):

- `langevin_kick_half` — the B step (reuses the `vv_kick` kernel
  internally; the distinct stage label distinguishes Langevin calls).
- `langevin_drift_half` — the A step (`lan_drift_half` kernel).
- `langevin_ou_step` — the O step (`lan_ou_step` kernel).

Only the stages corresponding to the chosen integrator have nonzero
counts; the others are absent from the output file. The velocity-Verlet
slot also leaves exactly one of `vv_kick_drift` / `vv_kick_drift_lossless`
and one of `vv_kick` / `vv_kick_lossless` empty, determined by the
`lossless` flag.

A kernel-launch helper that early-returns (e.g. when `particle_count == 0`)
does not record any sample. The CUDA event pair for that stage may still
exist in the runner's `Timings` instance with zero accumulated samples.

### Host stages (Instant-timed) <!-- rq-36105526 -->

One host timer per stage. Each is recorded as a single elapsed `Duration`
per occurrence. Stages:

- `config_load` — duration of `load_config`.
- `init_load` — duration of `load_init_state`.
- `gpu_init` — duration of `init_device`.
- `velocity_generation` — duration of Maxwell-Boltzmann sampling and
  momentum subtraction (zero samples when the init file supplies
  velocities).
- `host_to_device_upload` — duration of `ParticleBuffers::new` (which
  performs the initial bulk upload). One sample per run.
- `device_to_host_download` — duration of `ParticleState::download_from`
  inside the loop's snapshot/log download steps. One sample per call.
- `trajectory_write` — duration of one `TrajectoryWriter::write_frame`
  call. One sample per written frame.
- `log_write` — duration of one `LogWriter::write_row` call. One sample
  per written row.
- `neighbor_list_rebuild` — wall-clock duration of one full neighbor
  list rebuild, covering the device-to-host download of cell indices,
  the host-side stable sort, the upload of the sorted ordering, and
  the kernel launches that run inside the rebuild (sample is recorded
  by the runner around the entire rebuild block). Present only when
  `NeighborListConfig::CellList` is selected. One sample per rebuild.

### Synthetic stages <!-- rq-7eb22aad -->

- `total_runtime` — single-sample wall-clock measurement from the start of
  `run_simulation` to the moment just before the `.timings` file is
  serialised. Count is exactly `1` on a successful run.

## Timing Methodology <!-- rq-fdbb902d -->

### CUDA events <!-- rq-91cec484 -->

The runner holds one `(start, stop)` event pair per kernel stage. Both events
are created with default flags on the device returned by `init_device()`.

Launch sequencing per kernel stage:

```
record start event on default stream
call kernel-launch helper
record stop event on default stream
```

The start event for the first occurrence of a stage is recorded immediately
before the helper call. The elapsed time is computed lazily:

- On the **second and subsequent** occurrences of a stage, the previous
  pair's elapsed time is queried before being reused. The query
  (`cuEventElapsedTime`) blocks until both events have completed; the
  duration is accumulated into the stage's running statistics.
- At end of run, the final outstanding pair for each stage is drained by
  querying its elapsed time and accumulating the result.

Event objects are reused across timesteps. Memory cost is constant in the
number of timesteps.

When a kernel-launch helper early-returns (empty state), the runner skips
the record/start/stop sequence entirely. The stage's event pair remains
unrecorded for that iteration; the outstanding-pair state machine treats
it as if no launch ever happened.

### Host wall-clock <!-- rq-5d1f8e8e -->

Each host stage is timed with `std::time::Instant`:

```
let started = std::time::Instant::now();
operation();
timings.record_host(stage, started.elapsed());
```

Elapsed durations are stored internally as `u128` nanoseconds and converted
to floating-point microseconds at output time.

### Sample bookkeeping <!-- rq-e16e72bd -->

For every stage, the runner maintains:

- `count: u64` — total recorded samples.
- `total_ns: u128` — sum of all sample durations in nanoseconds.
- `min_ns: u64` — minimum sample duration (max-initialised before first
  sample; remains undefined when count is 0).
- `max_ns: u64` — maximum sample duration (zero-initialised; remains
  undefined when count is 0).

CUDA event elapsed times are reported by cudarc as `f32` milliseconds.
The runner converts to nanoseconds via
`(elapsed_ms * 1_000_000.0).round() as u128` and clamps a negative-valued
result (which can occasionally appear due to NVIDIA driver noise) to `0`.

Mean is derived at output time: `mean_us = total_ns as f64 / 1000.0 / count as f64`.

## Output File Format <!-- rq-56364532 -->

The timings file is a UTF-8 text file with one header line followed by one
row per non-empty stage. Lines end with `\n`.

Column layout, fixed widths, ASCII:

| Column | Width | Alignment | Content |
| ------ | ----- | --------- | ------- |
| `stage` | 28 | left | stage name (snake_case identifier) |
| `count` | 10 | right | sample count, base-10 integer |
| `total_ms` | 14 | right | total time across all samples in milliseconds, three decimals |
| `mean_us` | 13 | right | mean time per sample in microseconds, one decimal |
| `min_us` | 11 | right | minimum sample in microseconds, one decimal |
| `max_us` | 11 | right | maximum sample in microseconds, one decimal |

Columns are separated by single spaces (one column's right edge sits one
character before the next column's left edge), so each row is exactly
`28 + 1 + 10 + 1 + 14 + 1 + 13 + 1 + 11 + 1 + 11 = 92` characters wide
followed by `\n`.

A value that does not fit in its nominal column width is written in full
(no truncation, no scientific notation) — the row simply becomes longer
than 92 characters.

### Row ordering <!-- rq-1b34278b -->

Rows appear in this fixed order, with absent stages skipped:

1. `vv_kick_drift` or `vv_kick_drift_lossless` (whichever is present)
2. `langevin_kick_half` (B step, pre and post)
3. `langevin_drift_half` (A step)
4. `langevin_ou_step` (O step)
5. `neighbor_displacement_squared`
6. `copy_positions_into_reference`
7. `neighbor_list_build`
8. `lj_pair_force` or `lj_pair_force_neighbor` (whichever is present)
9. `reduce_pair_forces`
10. `morse_bond_force`
11. `reduce_bond_forces`
12. `accumulate_forces`
13. `vv_kick` or `vv_kick_lossless` (whichever is present)
14. `host_to_device_upload`
15. `device_to_host_download`
16. `neighbor_list_rebuild`
17. `trajectory_write`
18. `log_write`
19. `velocity_generation`
20. `config_load`
21. `init_load`
22. `gpu_init`
23. `total_runtime`

### Example <!-- rq-6289f344 -->

```
stage                             count       total_ms       mean_us      min_us      max_us
vv_kick_drift                       100         2.345          23.5        20.1        28.7
lj_pair_force                       101        12.467         123.4        98.7       145.2
reduce_pair_forces                  101         4.612          45.7        38.1        52.4
vv_kick                             100         2.111          21.1        18.5        25.9
host_to_device_upload                 1         1.890        1890.0      1890.0      1890.0
device_to_host_download              20         8.910         445.5       420.3       482.1
trajectory_write                     10         1.234         123.4       100.5       180.7
log_write                            10         0.234          23.4        20.1        30.5
velocity_generation                   1         0.456         456.0       456.0       456.0
config_load                           1         0.234         234.0       234.0       234.0
init_load                             1         0.567         567.0       567.0       567.0
gpu_init                              1        42.123       42123.0     42123.0     42123.0
total_runtime                         1        78.235       78235.0     78235.0     78235.0
```

### Overwrite policy <!-- rq-4cf03f96 -->

The runner refuses to start when a non-empty file already exists at the
resolved `output.timings_path`. The pre-flight check happens alongside the
trajectory and log existence checks, before the init file is read. The
actual file creation uses `OpenOptions::create_new` so the resolution is
race-free.

### Failed runs <!-- rq-b7317e2d -->

The timings file is written only when `run_simulation` succeeds. A run
that fails before the loop completes leaves no timings file at all (any
partial timings are discarded).

## Feature API <!-- rq-410afcd3 -->

### Types <!-- rq-4f5643f1 -->

- `KernelStage` — opaque newtype wrapping a `&'static str` stage name. <!-- rq-dc8a0ff7 -->
  `Copy`, `Eq`, `Hash`, `Debug`. Method
  `KernelStage::name(self) -> &'static str` returns the wrapped name.
  Stage instances are referenced via associated consts, one per timed
  kernel:
  - `KernelStage::VV_KICK_DRIFT` → `"vv_kick_drift"`
  - `KernelStage::VV_KICK` → `"vv_kick"`
  - `KernelStage::VV_KICK_DRIFT_LOSSLESS` → `"vv_kick_drift_lossless"`
  - `KernelStage::VV_KICK_LOSSLESS` → `"vv_kick_lossless"`
  - `KernelStage::LJ_PAIR_FORCE` → `"lj_pair_force"`
  - `KernelStage::REDUCE_PAIR_FORCES` → `"reduce_pair_forces"`
  - `KernelStage::LANGEVIN_KICK_HALF` → `"langevin_kick_half"`
  - `KernelStage::LANGEVIN_DRIFT_HALF` → `"langevin_drift_half"`
  - `KernelStage::LANGEVIN_OU_STEP` → `"langevin_ou_step"`
  - `KernelStage::MORSE_BOND_FORCE` → `"morse_bond_force"`
  - `KernelStage::REDUCE_BOND_FORCES` → `"reduce_bond_forces"`
  - `KernelStage::ACCUMULATE_FORCES` → `"accumulate_forces"`
  - `KernelStage::NEIGHBOR_DISPLACEMENT_SQUARED` → `"neighbor_displacement_squared"`
  - `KernelStage::NEIGHBOR_LIST_BUILD` → `"neighbor_list_build"`
  - `KernelStage::COPY_POSITIONS_INTO_REFERENCE` → `"copy_positions_into_reference"`

  The associated const `KernelStage::ORDER: &'static [KernelStage]`
  is the canonical registry of known kernel stages and fixes the row
  order used by `finalize` and the output file. Adding a stage means
  declaring a new associated const and appending it to `ORDER`.

- `HostStage` — opaque newtype wrapping a `&'static str` stage name. <!-- rq-d29f2811 -->
  `Copy`, `Eq`, `Hash`, `Debug`. Method
  `HostStage::name(self) -> &'static str` returns the wrapped name.
  Stage instances are referenced via associated consts:
  - `HostStage::CONFIG_LOAD` → `"config_load"`
  - `HostStage::INIT_LOAD` → `"init_load"`
  - `HostStage::GPU_INIT` → `"gpu_init"`
  - `HostStage::VELOCITY_GENERATION` → `"velocity_generation"`
  - `HostStage::HOST_TO_DEVICE_UPLOAD` → `"host_to_device_upload"`
  - `HostStage::DEVICE_TO_HOST_DOWNLOAD` → `"device_to_host_download"`
  - `HostStage::TRAJECTORY_WRITE` → `"trajectory_write"`
  - `HostStage::LOG_WRITE` → `"log_write"`
  - `HostStage::NEIGHBOR_LIST_REBUILD` → `"neighbor_list_rebuild"`
  - `HostStage::TOTAL_RUNTIME` → `"total_runtime"`

  The associated const `HostStage::ORDER: &'static [HostStage]` is the
  canonical registry of known host stages and fixes the row order used
  by `finalize` and the output file. Adding a stage means declaring a
  new associated const and appending it to `ORDER`.

- `Timings` — host-side state for collecting samples. Carries: <!-- rq-baf03449 -->
  - one `(CudaEvent, CudaEvent)` pair for every stage in
    `KernelStage::ORDER`
  - an outstanding-stop tracker per kernel stage (set when the
    previous occurrence's stop event has not yet been queried)
  - per-stage accumulators (`count`, `total_ns`, `min_ns`, `max_ns`)
    for every stage in `KernelStage::ORDER` and `HostStage::ORDER`
  - the `Arc<CudaDevice>` used to construct the events

  The set of valid stages is fixed at `Timings::new` time from
  `KernelStage::ORDER` and `HostStage::ORDER`; passing any other
  stage value to `kernel_start`, `kernel_stop`, or `record_host`
  panics.

- `StageStats` — public, `Clone`, `Debug` snapshot for one stage. <!-- rq-0dab90aa -->
  Fields:
  - `name: String` — stage name as it appears in the file
  - `count: u64`
  - `total_ns: u128`
  - `min_ns: u64`
  - `max_ns: u64`

  Helper methods:
  - `mean_us(&self) -> f64` — returns `total_ns as f64 / 1000.0 / count as f64`; panics in debug builds when `count == 0` (callers must skip empty stages first).

- `TimingsReport` — public, `Clone`, `Debug`. Carries an ordered <!-- rq-7453115b -->
  `Vec<StageStats>` in the row order specified above; stages with
  `count == 0` are excluded.

- `TimingsError` — error type. Variants: <!-- rq-779092ca -->
  - `Gpu(GpuError)` — CUDA driver error during event creation, recording,
    or elapsed-time query.

- `TimingsWriterError` — error type. Variants: <!-- rq-ec06c8e1 -->
  - `OutputExists { path: PathBuf }` — refusing to overwrite.
  - `Io(String)` — underlying filesystem error.

### Functions and methods <!-- rq-dae150d9 -->

- `Timings::new(device: Arc<CudaDevice>) -> Result<Timings, TimingsError>` <!-- rq-8a9c44f8 -->
  - Allocates one `(start, stop)` CUDA event pair on the given device
    (default flags) for every stage in `KernelStage::ORDER`.
  - Initialises an accumulator at zero samples for every stage in
    `KernelStage::ORDER` and `HostStage::ORDER`.
  - Returns `Gpu(_)` if event creation fails.

- `Timings::kernel_start(&mut self, stage: KernelStage) -> Result<(), TimingsError>` <!-- rq-58981e16 -->
  - Panics if `stage` is not in `KernelStage::ORDER` (the registry is
    fixed at construction time).
  - If a previous launch of the same stage has an outstanding stop event
    that has not yet been drained, this call first queries that elapsed
    time (blocking until completion), accumulates it into the stage's
    statistics, and clears the outstanding flag.
  - Records a new start event on the device's default stream.

- `Timings::kernel_stop(&mut self, stage: KernelStage) -> Result<(), TimingsError>` <!-- rq-b17e6de6 -->
  - Panics if `stage` is not in `KernelStage::ORDER`.
  - Records the stop event for the most recent `kernel_start` of the
    same stage on the default stream.
  - Sets the outstanding-stop flag so the next `kernel_start` (or
    `finalize`) will drain it.

  Calls to `kernel_start` and `kernel_stop` always come in pairs in
  source order. A bare `kernel_stop` with no preceding `kernel_start`
  for that stage debug-asserts.

- `Timings::record_host(&mut self, stage: HostStage, duration: std::time::Duration)` <!-- rq-037a9326 -->
  - Panics if `stage` is not in `HostStage::ORDER`.
  - Accumulates `duration.as_nanos()` into the stage's `total_ns`,
    increments `count`, and updates `min_ns` / `max_ns`.

- `Timings::finalize(&mut self) -> Result<TimingsReport, TimingsError>` <!-- rq-c4845f90 -->
  - For every `KernelStage` with an outstanding stop event, queries its
    elapsed time, accumulates it, and clears the flag.
  - Constructs and returns the `TimingsReport` in row order, excluding
    stages with `count == 0`.

- `write_timings_file(path: &Path, report: &TimingsReport) -> Result<(), TimingsWriterError>` <!-- rq-9b85fa6c -->
  - Creates the file at `path` using
    `OpenOptions::new().write(true).create_new(true).open(path)`.
  - Refuses to overwrite an existing file (`OutputExists`).
  - Writes one header line followed by one row per stage in the order
    encoded in `report.stages`, using the fixed column widths described
    above.
  - Flushes and closes the file before returning.

## Integration with the Runner <!-- rq-dcfdb7c9 -->

`run_simulation` constructs a `Timings` at the start of the run (after
`init_device` returns) and a `started: Instant` immediately on entry. The
sequence of recorded stages within `run_simulation` is:

1. `host_to_device_upload` — single `Instant` measurement around
   `ParticleBuffers::new(...)`.
2. `velocity_generation` — `Instant` measurement around the
   `generate_velocities` call. Skipped when explicit velocities are read
   from the init file.
3. Warm-up: `kernel_start(KernelStage::LJ_PAIR_FORCE)` / launch /
   `kernel_stop(KernelStage::LJ_PAIR_FORCE)` then
   `kernel_start(KernelStage::REDUCE_PAIR_FORCES)` / launch /
   `kernel_stop(KernelStage::REDUCE_PAIR_FORCES)`.
4. For each timestep:
   - `kernel_start` / launch / `kernel_stop` for the active kick-drift
     variant.
   - `kernel_start` / launch / `kernel_stop` for `LJ_PAIR_FORCE`.
   - `kernel_start` / launch / `kernel_stop` for `REDUCE_PAIR_FORCES`.
   - `kernel_start` / launch / `kernel_stop` for the active kick variant.
   - If a snapshot/log download is happening this step:
     - `Instant` measurement around `frame.download_from(&buffers)`.
   - If a trajectory frame is being written:
     - `Instant` measurement around `write_traj_frame(...)`.
   - If a log row is being written:
     - `Instant` measurement around `writer.write_row(...)`.
5. Flush trajectory and log writers (untimed).
6. `record_host(TotalRuntime, started.elapsed())`.
7. `let report = timings.finalize()?;`
8. `write_timings_file(&config.output.timings_path, &report)?;`

`config_load`, `init_load`, and `gpu_init` are timed by the runner's
outer wrapper (i.e. the same `run_simulation_with_phase` function records
them via `Instant` around each call). The `Timings` instance does not
yet exist when `gpu_init` is being timed, so those three durations are
buffered on the stack and recorded into the `Timings` immediately after
`Timings::new` returns.

The runner does **not** time the `.timings` file write itself; the
`total_runtime` row is captured before serialisation, and the cost of
writing the timings file is excluded from every row.

## Out of Scope <!-- rq-1434abd2 -->

- Per-timestep time-series output. The output file aggregates over the
  whole run.
- A `[profiling]` configuration section. Timings collection is always on
  and has no per-feature toggles in schema v1.
- Standard deviation, percentiles, or histograms.
- GPU memory usage / occupancy / SM utilisation. Only kernel wall time is
  measured.
- Per-block or per-thread profiling. Per-kernel-launch is the smallest
  unit.
- Comparison against a previous run (regression detection). The file is a
  point-in-time snapshot.
- Cross-host or cross-GPU timing comparisons. Timings vary by hardware.
- Reproducibility of timing data; only the *trajectory* and *log* outputs
  are bit-reproducible across runs.
- A public API for external-crate stage registration. The
  `KernelStage::ORDER` and `HostStage::ORDER` arrays are the canonical
  registry; the set of valid stages is fixed at compile time, and adding
  a stage requires editing those arrays in this crate.
- Streaming the timings file to stdout. Output is to a file only.
- Compressed (gzip, etc.) timings files.
- Profile data formats for external tools (Tracy, Perfetto, Chrome
  trace). The output is plain text only.
- Concurrent / multi-stream profiling. The default stream is the only
  one in use.

---

## Gherkin Scenarios <!-- rq-7acbda72 -->

```gherkin
Feature: Performance analysis and timings output

  Background:
    Given a CUDA-capable GPU available as device 0
    And a temporary directory tmp

  # --- File presence and structure ---

  @rq-f5e25186
  Scenario: Successful run writes a timings file
    Given a valid config with n_steps=5, trajectory_every=5, log_every=5
    And init.xyz contains N=2 particles with explicit velocities
    When dynamics run sim.toml is invoked
    Then it exits with code 0
    And the file tmp/sim.timings exists
    And the first line of tmp/sim.timings begins with "stage" and contains "count", "total_ms", "mean_us", "min_us", "max_us"

  @rq-86423766
  Scenario: Timings file is absent when run fails before the loop
    Given a config with a malformed init file
    When dynamics run sim.toml is invoked
    Then it exits with code 1
    And the file tmp/sim.timings does not exist

  @rq-afb80e25
  Scenario: Timings file uses the default path derived from config stem
    Given a valid config at tmp/sim.toml with no output.timings_path field
    When dynamics run sim.toml is invoked
    Then the timings file is written to tmp/sim.timings

  @rq-a2ebdaaf
  Scenario: Timings file path can be overridden in [output]
    Given a valid config with output.timings_path = "custom.timings"
    When dynamics run sim.toml is invoked
    Then the timings file is written to tmp/custom.timings

  @rq-11132169
  Scenario: Pre-flight refuses to overwrite an existing timings file
    Given a valid config
    And tmp/sim.timings already exists with arbitrary content
    When dynamics run sim.toml is invoked
    Then it exits with code 1
    And stderr contains "OutputExists" and "sim.timings"
    And tmp/sim.timings is unchanged

  # --- Row presence ---

  @rq-a7fdf81f
  Scenario: Lossy run includes lossy kick rows and excludes lossless rows
    Given a valid lossy config with n_steps=10
    When dynamics run sim.toml is invoked
    Then tmp/sim.timings has a row whose stage column equals "vv_kick_drift"
    And tmp/sim.timings has a row whose stage column equals "vv_kick"
    And tmp/sim.timings has no row whose stage column equals "vv_kick_drift_lossless"
    And tmp/sim.timings has no row whose stage column equals "vv_kick_lossless"

  @rq-b2fa4a1f
  Scenario: Lossless run includes lossless kick rows and excludes lossy rows
    Given a valid config with integrator.kind="velocity-verlet" and lossless=true and n_steps=10
    When dynamics run sim.toml is invoked
    Then tmp/sim.timings has a row whose stage column equals "vv_kick_drift_lossless"
    And tmp/sim.timings has a row whose stage column equals "vv_kick_lossless"
    And tmp/sim.timings has no row whose stage column equals "vv_kick_drift"
    And tmp/sim.timings has no row whose stage column equals "vv_kick"

  @rq-c1c9fe3a
  Scenario: Langevin run includes Langevin rows and excludes velocity-Verlet rows
    Given a valid config with integrator.kind="langevin-baoab",
      friction=1.0e12, temperature=300.0, seed=42, n_steps=10
    When dynamics run sim.toml is invoked
    Then tmp/sim.timings has a row whose stage column equals "langevin_kick_half"
    And tmp/sim.timings has a row whose stage column equals "langevin_drift_half"
    And tmp/sim.timings has a row whose stage column equals "langevin_ou_step"
    And tmp/sim.timings has no row whose stage column begins with "vv_"

  @rq-0c2265eb
  Scenario: Langevin kick_half count is 2 N_steps; drift_half is 2 N_steps; ou_step is N_steps
    Given a valid Langevin config with n_steps=10
    When dynamics run sim.toml is invoked
    Then the row for langevin_kick_half has count = 20
    And the row for langevin_drift_half has count = 20
    And the row for langevin_ou_step has count = 10

  @rq-14b8e042
  Scenario: Morse-bonded run records morse_bond_force, reduce_bond_forces, accumulate_forces
    Given a valid config with `bonds = "topology.bonds"` and a non-empty bond list
    And n_steps = 10
    When dynamics run sim.toml is invoked
    Then tmp/sim.timings has rows whose stage columns equal
      "morse_bond_force", "reduce_bond_forces", and "accumulate_forces"
    And each row's count equals 11 (one warm-up plus ten loop iterations)

  @rq-c7df5714
  Scenario: Bond-free run omits morse_bond_force and reduce_bond_forces
    Given a valid config without a `bonds` field
    When dynamics run sim.toml is invoked
    Then tmp/sim.timings has no row whose stage column equals "morse_bond_force"
    And tmp/sim.timings has no row whose stage column equals "reduce_bond_forces"
    And tmp/sim.timings has a row whose stage column equals "accumulate_forces"
      (the combiner runs whenever any slot is present)

  @rq-bde625cf
  Scenario: Empty (N=0) run omits all kernel rows but retains host rows
    Given a valid config with kind="velocity-verlet", N=0 particles, n_steps=5
    When dynamics run sim.toml is invoked
    Then tmp/sim.timings has no rows whose stage column begins with "vv_"
      or "langevin_" or "morse_" or "reduce_" or equals "lj_pair_force"
      or "accumulate_forces"
    And tmp/sim.timings has a row whose stage column equals "gpu_init"
    And tmp/sim.timings has a row whose stage column equals "total_runtime"

  @rq-62300a18
  Scenario: All-pairs neighbor mode records lj_pair_force and omits neighbor-list rows
    Given a valid config with [neighbor_list] mode="all-pairs" and n_steps=5
    When dynamics run sim.toml is invoked
    Then tmp/sim.timings has a row whose stage column equals "lj_pair_force"
    And tmp/sim.timings has no row whose stage column equals "lj_pair_force_neighbor"
    And tmp/sim.timings has no row whose stage column equals "neighbor_displacement_squared"
    And tmp/sim.timings has no row whose stage column equals "neighbor_list_build"
    And tmp/sim.timings has no row whose stage column equals "copy_positions_into_reference"
    And tmp/sim.timings has no row whose stage column equals "neighbor_list_rebuild"

  @rq-ef918dc6
  Scenario: Cell-list neighbor mode records the neighbor stages and omits lj_pair_force
    Given a valid config with [neighbor_list] mode="cell-list" and n_steps=10
    When dynamics run sim.toml is invoked
    Then tmp/sim.timings has a row whose stage column equals "lj_pair_force_neighbor"
    And tmp/sim.timings has no row whose stage column equals "lj_pair_force"
    And tmp/sim.timings has a row whose stage column equals "neighbor_displacement_squared"
    And tmp/sim.timings has a row whose stage column equals "neighbor_list_build"
    And tmp/sim.timings has a row whose stage column equals "copy_positions_into_reference"
    And tmp/sim.timings has a row whose stage column equals "neighbor_list_rebuild"

  @rq-75746f64
  Scenario: neighbor_displacement_squared count equals one warm-up plus per-step launches
    Given a valid cell-list config with n_steps=10
    When dynamics run sim.toml is invoked
    Then the row for neighbor_displacement_squared has count = 10

  @rq-7f2310ac
  Scenario: neighbor_list_build and copy_positions_into_reference counts match rebuilds
    Given a valid cell-list config with n_steps=10 and r_skin chosen so that exactly
      one rebuild fires after the initial build (i.e. two builds total: warm-up + one)
    When dynamics run sim.toml is invoked
    Then the row for neighbor_list_build has count = 2
    And the row for copy_positions_into_reference has count = 2
    And the row for neighbor_list_rebuild has count = 2

  @rq-3bd5336c
  Scenario: Velocity generation row is absent when init supplies velocities
    Given a valid config with N=2 and an init file that provides explicit velocities
    When dynamics run sim.toml is invoked
    Then tmp/sim.timings has no row whose stage column equals "velocity_generation"

  @rq-a555a750
  Scenario: Velocity generation row is present when init lacks velocities
    Given a valid config with N=4 and temperature=300 and an init file without velocities
    When dynamics run sim.toml is invoked
    Then tmp/sim.timings has a row whose stage column equals "velocity_generation"
    And that row's count column equals 1

  # --- Sample counts ---

  @rq-a9b511ea
  Scenario: Kernel-stage counts match the runner's launch counts
    Given a valid lossy config with n_steps=10
    When dynamics run sim.toml is invoked
    Then the row for lj_pair_force has count = 11 (one warm-up + ten loop)
    And the row for reduce_pair_forces has count = 11
    And the row for vv_kick_drift has count = 10
    And the row for vv_kick has count = 10

  @rq-46c317ef
  Scenario: trajectory_write count equals frames written
    Given a valid config with n_steps=20, trajectory_every=5, log_every=0
    When dynamics run sim.toml is invoked
    Then the row for trajectory_write has count = 5
    And tmp/sim.timings has no row whose stage column equals "log_write"

  @rq-34bfc634
  Scenario: log_write count equals log rows written
    Given a valid config with n_steps=20, trajectory_every=0, log_every=10
    When dynamics run sim.toml is invoked
    Then the row for log_write has count = 3
    And tmp/sim.timings has no row whose stage column equals "trajectory_write"

  @rq-6fe2b058
  Scenario: device_to_host_download count equals snapshot points reached
    Given a valid config with n_steps=20, trajectory_every=5, log_every=10
    When dynamics run sim.toml is invoked
    Then the row for device_to_host_download has count equal to the number of distinct steps at which a snapshot or log row was requested (including step 0)

  @rq-44e5c930
  Scenario: Each single-occurrence host stage has count = 1
    Given any successful run
    Then the rows for config_load, init_load, gpu_init, host_to_device_upload, and total_runtime each have count = 1

  # --- Statistic invariants ---

  @rq-105afb1d
  Scenario: min <= mean <= max for every row
    Given any successful run with at least one stage that has count >= 2
    Then for that stage, min_us <= mean_us <= max_us

  @rq-9818f429
  Scenario: total_ms equals mean_us * count / 1000 within reporting precision
    Given any successful run
    Then for every row, total_ms is within 0.001 of (mean_us * count / 1000)

  @rq-7e79501b
  Scenario: total_runtime is at least as large as the maximum single-stage total
    Given any successful run
    Then total_runtime.total_ms >= max(other_row.total_ms for other_row in the file)

  # --- Format and ordering ---

  @rq-f1850393
  Scenario: Header line has the documented column names in the documented order
    Given any successful run
    Then the first line of tmp/sim.timings, when split on whitespace, equals
      ["stage", "count", "total_ms", "mean_us", "min_us", "max_us"]

  @rq-ef7c2bb9
  Scenario: Rows appear in the documented order
    Given a valid lossy config with n_steps=10, trajectory_every=5, log_every=5
    When dynamics run sim.toml is invoked
    Then the stage column of consecutive data rows of tmp/sim.timings reads
      ["vv_kick_drift", "lj_pair_force", "reduce_pair_forces", "vv_kick",
       "host_to_device_upload", "device_to_host_download", "trajectory_write",
       "log_write", "config_load", "init_load", "gpu_init", "total_runtime"]
      (omitting any absent stages; "velocity_generation" is absent because
      explicit velocities are provided)

  @rq-44dfc3da
  Scenario: Numeric columns have the documented precision
    Given any successful run
    Then every row's total_ms field has exactly three digits after the decimal point
    And every row's mean_us, min_us, and max_us fields have exactly one digit after the decimal point

  @rq-f4780029
  Scenario: Stages with count = 0 are absent
    Given any successful run with at least one stage having zero samples
    Then no row whose stage column corresponds to a count-zero stage appears in the file

  # --- Concurrency / overhead ---

  @rq-ad403fb6
  Scenario: Trajectory and log files remain bit-identical between two identical runs
    Given two identical valid configs with explicit init velocities
    When dynamics run is invoked on each twice with the same files
    Then the two trajectory files are byte-identical
    And the two log files are byte-identical

  @rq-1b8fd2a0
  Scenario: Two identical runs produce timings files with the same stage rows and the same counts
    Given two identical valid configs
    When dynamics run is invoked on each
    Then each row of timings file A has a matching row in timings file B with the same stage and the same count
    And numeric columns may differ between A and B

  # --- Path collision validation (delegated to config-schema) ---

  @rq-f84e4fa1
  Scenario: Timings path equal to trajectory path is rejected
    Given a config with output.trajectory_path = "out.dat" and output.timings_path = "out.dat"
    When dynamics run sim.toml is invoked
    Then it exits with code 1
    And stderr contains "PathCollision"
    And no timings file is written

  @rq-ec5b6cf1
  Scenario: Timings path equal to log path is rejected
    Given a config with output.log_path = "out.log" and output.timings_path = "out.log"
    When dynamics run sim.toml is invoked
    Then it exits with code 1
    And stderr contains "PathCollision"

  @rq-2ebf81fd
  Scenario: Timings path equal to init path is rejected
    Given a config with init = "particles.dat" and output.timings_path = "particles.dat"
    When dynamics run sim.toml is invoked
    Then it exits with code 1
    And stderr contains "PathCollision"

  # --- write_timings_file behaviour ---

  @rq-de5ca07a
  Scenario: write_timings_file refuses to overwrite an existing file
    Given a path tmp/run.timings that already exists with arbitrary content
    And a TimingsReport with at least one stage
    When write_timings_file(tmp/run.timings, &report) is called
    Then it returns Err(TimingsWriterError::OutputExists { path: tmp/run.timings })
    And tmp/run.timings is unchanged

  @rq-e93e7ad1
  Scenario: write_timings_file fails when the parent directory does not exist
    Given tmp/missing/ does not exist
    When write_timings_file(tmp/missing/run.timings, &report) is called
    Then it returns Err(TimingsWriterError::Io(_))

  @rq-5c716d48
  Scenario: write_timings_file with an empty report writes only the header
    Given a TimingsReport with zero stages
    When write_timings_file(tmp/run.timings, &report) is called
    Then it returns Ok(())
    And the file contains exactly the header line followed by a newline

  # --- Timings struct behaviour ---

  @rq-e946870f
  Scenario: Timings::new allocates the documented event pairs
    Given a CudaDevice obtained from init_device()
    When Timings::new(device) is called
    Then it returns Ok(timings)
    And every per-stage accumulator has count = 0

  @rq-79291197
  Scenario: kernel_start followed by kernel_stop and finalize records one sample
    Given a Timings constructed on a fresh device
    When timings.kernel_start(KernelStage::LJ_PAIR_FORCE) is called
    And a small kernel is launched
    And timings.kernel_stop(KernelStage::LJ_PAIR_FORCE) is called
    And timings.finalize() is called
    Then the resulting report contains exactly one StageStats entry for lj_pair_force
    And that entry's count equals 1

  @rq-56043142
  Scenario: Repeated kernel_start/kernel_stop pairs accumulate
    Given a Timings constructed on a fresh device
    When ten matched kernel_start/kernel_stop pairs for KernelStage::LJ_PAIR_FORCE are issued around real launches
    And timings.finalize() is called
    Then the resulting report's lj_pair_force entry has count = 10
    And the total_ns is non-zero

  @rq-2cbe0828
  Scenario: record_host updates count, total, min, and max
    Given a fresh Timings
    When record_host(HostStage::CONFIG_LOAD, Duration::from_micros(100)) is called
    And record_host(HostStage::CONFIG_LOAD, Duration::from_micros(50)) is called
    And record_host(HostStage::CONFIG_LOAD, Duration::from_micros(200)) is called
    And timings.finalize() is called
    Then the config_load entry has count = 3
    And total_ns = 350_000
    And min_ns = 50_000
    And max_ns = 200_000

  @rq-f232d41b
  Scenario: kernel_start with a KernelStage not in ORDER panics
    Given a fresh Timings
    And a KernelStage value `unknown = KernelStage::new("not_a_stage")` whose
      name does not appear in KernelStage::ORDER
    When timings.kernel_start(unknown) is called
    Then it panics

  @rq-264d2234
  Scenario: record_host with a HostStage not in ORDER panics
    Given a fresh Timings
    And a HostStage value `unknown = HostStage::new("not_a_host_stage")` whose
      name does not appear in HostStage::ORDER
    When timings.record_host(unknown, Duration::from_micros(10)) is called
    Then it panics
```
