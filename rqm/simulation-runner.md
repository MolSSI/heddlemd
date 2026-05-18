# Feature: `dynamics run` Simulation Runner <!-- rq-357909e4 -->

The simulation runner is the command-line entry point that turns a TOML
configuration file into a complete simulation. It reads the config and the
referenced initial-state file, allocates the GPU pipeline described by
`build-pipeline.md`, `particle-state.md`, `pair-reduction.md`,
`lj-pair-force.md`, and the integrator slots in `integration/`, drives the
timestep loop for `simulation.n_steps` iterations, and writes snapshots
and diagnostics at the declared cadences using `trajectory-output.md` and
`log-output.md`.

The runner is the only piece in the project that has visibility of every
subsystem; it is the integration point.

## CLI <!-- rq-82d0c34a -->

The `dynamics` binary carries two subcommands:

```
dynamics run  <config-path>
dynamics lint <config-path> [--with-gpu]
```

`<config-path>` is the path to a TOML simulation config (see
`config-schema.md`). Relative paths are resolved against the current
working directory. No environment variables or configuration sources
beyond the config file are accepted in schema v1; every parameter
affecting the trajectory lives in the file the path points at.

Errors are reported as a single line on stderr beginning with
`error: ` followed by a human-readable description that includes the
responsible file path and field name where applicable.

### `run` subcommand <!-- rq-cfb6aadb -->

Executes the simulation described by `<config-path>` to completion.
The full setup-and-loop pipeline is described under *Runner flow*.

- Exit codes:
  - `0` — simulation completed successfully (every requested step ran
    and every requested output flushed).
  - `1` — any error before the loop starts: malformed CLI args, config
    load failure, init-state load failure, output-file overwrite check
    failure, GPU initialisation failure.
  - `2` — error during the loop: a kernel launch failed, a write to
    the trajectory or log failed, or a download from the device
    failed.

### `lint` subcommand <!-- rq-c1d5b25d -->

Validates the config and its referenced inputs against every error
the runner can detect without executing the integration loop, then
exits. The lint pipeline is described under *Lint flow*. Designed
for HPC contexts where a long submission queue makes trial-and-error
iteration expensive: `dynamics lint` runs cheaply on a login node and
reports every issue that would cause a `run` to fail at setup time.

- `--with-gpu` (optional flag) extends the lint to include device
  initialisation and full GPU-side setup (see *Lint flow*'s
  *`--with-gpu` stages*). Omit the flag to keep the lint CPU-only
  (the default, suitable for login nodes without a CUDA device).
- Exit codes:
  - `0` — every check passed.
  - `1` — at least one check failed.
- Lint writes no files. Pre-existing output files are detected with
  `Path::exists()`; the filesystem is otherwise unchanged.
- Stops at the first failed check (short-circuit). Subsequent stages
  are reported as **skipped (earlier check failed)** in the
  per-stage report (see *Lint flow*'s *Output format*).

### Usage error messages <!-- rq-7e5cb9f8 -->

`dynamics` with no arguments, an unrecognised subcommand, a
recognised subcommand without its required `<config-path>` argument,
or `lint` with an unrecognised flag prints the following usage block
to stderr and exits with code `1`:

```
usage: dynamics run  <config-path>
       dynamics lint <config-path> [--with-gpu]
```

## Runner flow <!-- rq-ef902cf6 -->

A single invocation proceeds through these stages in order. Any stage that
fails terminates the process with the appropriate exit code and stderr
message.

1. **Parse CLI.** Confirm the form `run <config-path>`. Capture `<config-path>`.
2. **Load config.** Call `load_config(&config_path)`
   (`config-schema.md`). Failure → exit 1.
3. **Pre-flight output checks.** Verify each enabled output path does not
   already exist. Trajectory and log are gated by their respective
   `_every > 0` predicates; the timings file (see
   `performance-analysis.md`) is always written and always checked.
   Failure → exit 1 with `OutputExists` reporting the offending path.
   This check is performed before the init file is read so the runner
   refuses long, expensive runs early.
4. **Build type-name slice.** Construct `type_names: Vec<&str>` from
   `config.particle_types[i].name`, indexed left-to-right.
5. **Load init state.** Call
   `load_init_state(&config.init, &type_names)`
   (`init-state-file.md`). Failure → exit 1.
6. **Build SimulationBox.** From `init_state.box`. This is the box the
   simulation uses; the config does not specify a box. Immediately
   after the box is known, verify the cell-list compatibility check:
   when `config.neighbor_list` is `NeighborListConfig::CellList { r_skin, .. }`,
   the runner computes `cutoff_max` as the largest cutoff across
   `config.pair_interactions`, `config.coulomb.cutoff`, and
   `config.spme.r_cut_real` (whichever are present), forms
   `required = 3 * (cutoff_max + r_skin)`, and delegates the
   per-direction check to
   `sim_box.check_min_perpendicular_width(required)` (see
   `simulation-box.md`). On `Err(SimulationBoxError::PerpendicularWidthTooSmall
   { direction, width, required })` the runner translates the payload
   verbatim into `RunnerError::CellListBoxTooSmall { direction, width,
   required }` and exits with code `1`.
   `NeighborListConfig::AllPairs` skips this check.
6a. **Load topology file (if supplied).** When
    `config.topology.is_some()`, build the slice of bond type names
    from `config.bond_types` and the slice of angle type names from
    `config.angle_types`, call
    `load_topology_file(path, particle_count, &bond_type_names,
    &angle_type_names, &config.constraint_types)`
    (`forces/topology.md`), and capture the resulting `(BondList,
    AngleList, ExclusionList, ConstraintList)`. Failure → exit 1.
    When `config.topology.is_none()`, use an empty `BondList`, an
    empty `AngleList`, an empty `ExclusionList`, and an empty
    `ConstraintList`.
7. **Initialise CUDA.** Call `init_device()` (`build-pipeline.md`).
   Failure → exit 1. When `config.spme.is_some()`, `init_device` runs
   the cuFFT determinism smoke test described in `forces/spme.md`. A
   smoke-test failure surfaces as
   `RunnerError::CuFftNonDeterministic { differences: usize }` and
   exits with code `1`.
8. **Generate velocities (if absent).** When
   `init_state.velocities.is_none()`, sample velocities via the
   Maxwell-Boltzmann procedure described in *Velocity generation*. When
   `Some(_)`, copy directly from the init state.
9. **Construct host particle state.** Build a `ParticleState`
   (`particle-state.md`) with:
   - `positions_*` from `init_state.positions_*`,
   - `velocities_*` from either the init state or the generated values,
   - `masses` populated from
     `config.particle_types[init_state.type_indices[i]].mass` cast to `f32`,
   - `charges` populated from
     `config.particle_types[init_state.type_indices[i]].charge` cast to
     `f32`,
   - `ids = None` (default `0..N`),
   - `forces_*` zero-initialised by the constructor.
10. **Allocate `ParticleBuffers`.** Construct `ParticleBuffers` from
    the host state.
11. **Construct the integrator, thermostat, barostat, and constraint.**
    Build the four slot handles by dispatching through the
    caller-supplied `Registries` bundle (see
    `integration/framework.md` and
    `integration/constraint-framework.md`):
    - `registries.integrators.build(&config.integrator,
      device.clone(), N)` → `Box<dyn Integrator>`.
    - `registries.thermostats.build_optional(config.thermostat.as_ref(),
      device.clone(), N)` → `Option<Box<dyn Thermostat>>`.
    - `registries.barostats.build_optional(config.barostat.as_ref(),
      device.clone(), N)` → `Option<Box<dyn Barostat>>`.
    - `registries.constraint_types.build_optional(&constraint_list,
      device.clone(), N)` → `Option<Box<dyn Constraint>>`. Returns
      `None` when the topology file's `[constraints]` section is
      empty or absent.

    Each slot owns any per-run state it needs (e.g. `LosslessBuffers`
    for `velocity-verlet` when `lossless == true`; the chain state and
    `ke_scratch` buffer for `nose-hoover-chain`; the per-group
    snapshot, atom-index, and per-type parameter buffers for
    `settle`). The integrator-owns-its-own-thermostat and
    integrator-supports-constraints compatibility checks (builder
    predicates `IntegratorBuilder::owns_thermostat(&params)`,
    `owns_barostat(&params)`, and `supports_constraints(&params)`
    queried via `Config::validate_against(&registries)` and
    `Config::validate_constraint_compatibility(&registries, has_constraints)`)
    have already been enforced at this point; no runtime guard is
    required here.

    The constraint slot is threaded through the runner's plan walk
    (via the `run_step` helper in `integration/framework.md`); see
    that file for the dispatch sequence.
12. **Construct the force field.** Call
    `ForceField::new(&registries.potentials, device.clone(),
    N, &sim_box, &config.pair_interactions, &config.bond_types,
    &bond_list, &exclusion_list, &config.neighbor_list)` (see
    `forces/framework.md`). The slot list is the one produced by
    iterating `registries.potentials.builders` and keeping every
    `Some(slot)` they return. For the built-in `PotentialRegistry`
    on a typical config that yields the LJ slot (always) and the
    `MorseBonded` slot (when the bond list is non-empty);
    user-registered builders extend the slot list per the registry's
    iteration order. When `config.neighbor_list` selects `CellList`, the
    LJ slot's sub-state owns the cell-list and neighbor-list buffers
    described in `forces/neighbor-list.md`. The runner no longer
    manipulates the `PairBuffer`, neighbor-counts, or Lennard-Jones
    parameter struct directly.
13. **Open output writers.** Open `TrajectoryWriter` and/or `LogWriter`
    depending on the `_every` settings. The `LogWriter` is opened with
    the concatenation of every slot's extra-column names, in dispatch
    order:
    `integrator.log_column_names() ++ thermostat.map(|t|
    t.log_column_names()).unwrap_or_default() ++ barostat.map(|b|
    b.log_column_names()).unwrap_or_default()`. The runner caches the
    concatenated slice for the duration of the run. When that cached
    slice is non-empty, the runner additionally allocates a length-1
    `CudaSlice<f32>` named `pe_scratch` (the per-log-row scratch buffer
    passed to `compute_total_potential_energy` in steps 15 and 16.f).
    Failure → exit 1.
14. **Warm up forces.** Call `force_field.step(&mut buffers, &sim_box,
    &mut timings)` once to populate `forces_*` with `F(x_0)`. This is
    the same warm-up pattern used in `pipeline-reproducibility.md`.
15. **Write step-0 outputs.** When trajectory output is enabled, download
    the relevant buffers and call `write_frame(step=0, ...)`. When log
    output is enabled, download `velocities_*` and `masses` (the
    `masses` download is cached for the remainder of the run), compute
    KE and T via `compute_kinetic_energy` and `compute_temperature`
    (`log-output.md`); if the cached extra-column slice is non-empty,
    additionally call
    `compute_total_potential_energy(&buffers, &mut pe_scratch)` (see
    `integration/nose-hoover-chain.md`) to obtain the total PE in
    joules via a single-block deterministic GPU reduction of
    `buffers.potential_energies` followed by a one-scalar download.
    Assemble the extras vector as the concatenation of each slot's
    `log_column_values(ke, pe)` in the same dispatch order used at
    log-open time (integrator, thermostat, barostat). Call
    `write_row(0, 0.0, ke, t, &extras)`.
16. **Timestep loop.** For each step `s` in `1 ..= n_steps`:
    a. If `thermostat.is_some()`,
       `thermostat.apply_pre(&mut buffers, dt, &mut timings)` — runs
       the thermostat's pre-step modification (a Trotter half-step for
       NHC; a no-op for thermostats whose `apply_pre` is left at the
       trait default).
    b. `integrator.step(&mut buffers, &mut sim_box, &mut force_field, dt, &mut timings)`
       — runs the integrator's time-stepping body and calls
       `force_field.step(...)` internally at the integrator's chosen
       point(s). On return, `particle_buffers.forces_*`,
       `potential_energies`, and `virials` hold the values for the
       post-step positions.
    c. If `thermostat.is_some()`,
       `thermostat.apply_post(&mut buffers, dt, &mut timings)` — runs
       the thermostat's post-step modification (the right half-step
       for NHC; the full single rescale for CSVR / Berendsen; the
       Bernoulli resample for Andersen).
    d. If `barostat.is_some()`,
       `barostat.apply(&mut buffers, &mut sim_box, dt, &mut timings)`
       — runs the barostat's box and position rescale. The default
       registry has no implementations; this branch is dead code at
       runtime until a concrete barostat lands.

    The loop variable `s` is local to the runner and gates trajectory
    and log writes below; it is not passed to any slot.

    e. If trajectory output is enabled and `s % trajectory_every == 0`,
       download positions (and velocities when configured) and call
       `write_frame(step=s, ...)`.
    f. If log output is enabled and `s % log_every == 0`, download
       velocities, compute KE and T; when the cached extra-column
       slice is non-empty, additionally call
       `compute_total_potential_energy(&buffers, &mut pe_scratch)` to
       obtain the total PE via a single-block deterministic GPU
       reduction (one f32 scalar downloaded; no per-particle download),
       and assemble the extras vector as the concatenation of each
       slot's `log_column_values(ke, pe)` in dispatch order. Call
       `write_row(s, s as f64 * dt, ke, t, &extras)`.
    g. Possibly emit a progress line (see *Progress reporting*).
17. **Flush and close.** Call `flush()` on each open writer. The writers'
    `Drop` impls are best-effort but the runner calls `flush` explicitly
    so flush errors propagate.
18. **Write timings file.** Capture the total-runtime measurement, drain
    outstanding CUDA event pairs via `Timings::finalize`, and serialise
    the resulting report to `config.output.timings_path` via
    `write_timings_file`. See `performance-analysis.md` for the
    instrumentation contract and file format.
19. **Final summary.** Emit one summary line to stdout (see
    *Final summary*). Exit 0.

The `dt` value passed to integrator launches is `config.simulation.dt as
f32`. KE/temperature computation uses `f64` arithmetic on `f32`-downloaded
values.

## Lint flow <!-- rq-c02c6b45 -->

`dynamics lint <config-path> [--with-gpu]` exercises every setup-phase
check `run` performs, but stops after the last setup check, writes no
output files, and never enters the integration loop. The pipeline
reuses the same loader functions, the same validators, and the same
error types as *Runner flow*; the only differences are the absence of
side effects (no writer opens, no trajectory frames, no `.timings`
file) and the optional skipping of every GPU-touching stage.

### Stage order <!-- rq-23f05652 -->

Stages run in the order below. Outcomes are recorded into a
`LintReport` (see *Feature API*); a failed stage marks every
subsequent stage as **skipped (earlier check failed)**.

#### Always-on stages (CPU only) <!-- rq-65a63eec -->

1. **`config`** — `load_config(&config_path)` (when `with_gpu = false`)
   or `load_config_raw(&config_path)` followed by
   `config.validate_against(&registries)` and
   `config.validate_constraint_compatibility(&registries,
   has_constraints)` (when `with_gpu = true`, to match `run`'s
   registry dispatch). Covers the filename-convention check, TOML
   parse, every per-field domain check, path-collision check, and
   per-kind registry dispatch.
2. **`output paths`** — for each enabled output path
   (`trajectory_path` when `trajectory_every > 0`, `log_path` when
   `log_every > 0`, `timings_path` always), test with
   `Path::exists()`. A pre-existing file is a FAIL carrying the same
   payload (`RunnerError::OutputExists { path }`) that `run`'s
   pre-flight surfaces. No file is created or removed; multiple
   pre-existing paths are detected but only the first is reported as
   the stage failure (consistent with the short-circuit semantics).
3. **`init`** — `load_init_state(&config.init, &type_names)`. On
   success the stage description carries the particle count and the
   box dimensions extracted from `init_state.box`.
4. **`box/cutoff`** — when `config.neighbor_list` is `CellList`,
   compute `cutoff_max` as in *Runner flow* step 6 and call
   `sim_box.check_min_perpendicular_width(3 * (cutoff_max + r_skin))`.
   A failure surfaces as `RunnerError::CellListBoxTooSmall { .. }`.
   When `config.neighbor_list` is `AllPairs`, the stage is recorded
   as **not applicable (mode = all-pairs)** and is not a failure.
5. **`topology`** — when `config.topology.is_some()`, call
   `load_topology_file(...)` as in *Runner flow* step 6a and record
   the bond, angle, and constraint-group counts on success. When
   `config.topology.is_none()`, the stage is recorded as **not
   supplied** and is not a failure.

#### `--with-gpu` stages <!-- rq-688fb553 -->

6. **`gpu`** — when `with_gpu = true`, runs every remaining
   setup-phase stage of *Runner flow* in order: `init_device()` (step
   7, including the cuFFT determinism smoke test when SPME is
   configured), velocity generation (step 8, when the init file
   omits velocities), `ParticleState::new` (step 9), the
   host-to-device upload via `ParticleBuffers::new` (step 10),
   construction of the integrator / thermostat / barostat /
   constraint slots (step 11), and `ForceField::new` (step 12). Stops
   before step 13 (output writers). Any error from these stages —
   `RunnerError::Gpu`, `RunnerError::CuFftNonDeterministic`,
   `RunnerError::Integrator`, `RunnerError::Thermostat`,
   `RunnerError::Barostat`, `RunnerError::Constraint`,
   `RunnerError::ForceField`, `RunnerError::ParticleState` — surfaces
   as the stage failure.

   When `with_gpu = false`, the stage is recorded as **not checked
   (re-run with `--with-gpu`)** unconditionally; no GPU device is
   opened and no allocation is attempted.

### Output format <!-- rq-25185894 -->

On exit, the runner emits the lint report on stdout, followed by a
single-line error summary on stderr when at least one stage failed.
Each per-stage line is formatted as a left-aligned 12-column label, a
single space, and a description.

Successful CPU-only example:

```
[dynamics lint] OK
  config       /tmp/sim/argon.in.toml
  output paths none pre-exist
  init         resolved, 10000 particles, box 8.0e-9 × 8.0e-9 × 1.0e-8 m
  box/cutoff   min perp width 8.0e-9 m ≥ required 3.30e-9 m
  topology     not supplied
  gpu          not checked (re-run with --with-gpu)
```

Successful `--with-gpu` example:

```
[dynamics lint] OK
  config       /tmp/sim/argon.in.toml
  output paths none pre-exist
  init         resolved, 10000 particles, box 8.0e-9 × 8.0e-9 × 1.0e-8 m
  box/cutoff   min perp width 8.0e-9 m ≥ required 3.30e-9 m
  topology     not supplied
  gpu          init_device OK; ParticleBuffers, slots, ForceField allocated
```

Failure at `box/cutoff` (CPU-only):

```
[dynamics lint] FAIL
  config       /tmp/sim/argon.in.toml
  output paths none pre-exist
  init         resolved, 10000 particles, box 2.0e-9 × 5.0e-9 × 5.0e-9 m
  box/cutoff   FAIL — min perp width 2.0e-9 m along `a` < required 3.30e-9 m
  topology     skipped (earlier check failed)
  gpu          not checked (re-run with --with-gpu)
error: simulation box perpendicular width along lattice direction `a` is 2e-9, below the required 3.3e-9
```

Failure at `config` (filename-convention violation):

```
[dynamics lint] FAIL
  config       FAIL — /tmp/sim/argon.toml does not end in `.in.toml`
  output paths skipped (earlier check failed)
  init         skipped (earlier check failed)
  box/cutoff   skipped (earlier check failed)
  topology     skipped (earlier check failed)
  gpu          skipped (earlier check failed)
error: config filename `/tmp/sim/argon.toml` does not end in `.in.toml` (or its derived root is empty)
```

Description text is descriptive prose, not machine-parsed. The
canonical, parseable surface for callers is the `LintReport` returned
by `lint_simulation` (see *Feature API*); the stdout block is for
human consumption.

## Velocity generation <!-- rq-2be8ef35 -->

When the init file does not supply velocities, the runner samples them from
a Maxwell-Boltzmann distribution at `config.simulation.temperature` using a
deterministic RNG seeded by `config.simulation.seed`. Sampling is followed
by a centre-of-mass momentum subtraction and a single-scalar rescale, so the
generated array's flat-3N temperature equals the configured target (within
f32 storage round-off) for any system with thermal degrees of freedom
(`N >= 2`). The procedure is fully specified so that two runs with identical
config and identical init files produce byte-identical velocity arrays.

### RNG <!-- rq-1b7680ad -->

The runner constructs `rand_chacha::ChaCha8Rng::seed_from_u64(seed)`. This
yields a deterministic sequence across `rand_chacha 0.3` patch releases.
The `rand_chacha` and `rand` crates are added as runtime dependencies of
the `dynamics` crate.

### Sampling order <!-- rq-2249f685 -->

Velocities are generated in nested loop order: particle index outer, axis
inner (x, y, z). For each `(particle, axis)`, the runner consumes two `f64`
uniforms `(u1, u2)` from the RNG and computes one normal sample via
Box-Muller; the second Box-Muller sample is discarded to keep the
specification trivial.

```
for i in 0..N:
    for axis in (x, y, z):
        u1 = sample_uniform_open(rng)   # f64 in (0.0, 1.0]
        u2 = sample_uniform_unit(rng)   # f64 in [0.0, 1.0)
        z  = sqrt(-2.0 * ln(u1)) * cos(2.0 * pi * u2)
        sigma = sqrt(k_B * T / (masses[i] as f64))
        v_high_precision = z * sigma
        v[axis][i] = v_high_precision as f32
```

- `sample_uniform_open(rng)`: draws an `f64` in `(0.0, 1.0]`. Use
  `1.0 - rng.gen::<f64>()` where `rng.gen::<f64>()` returns `[0.0, 1.0)`.
- `sample_uniform_unit(rng)`: draws an `f64` in `[0.0, 1.0)` directly via
  `rng.gen::<f64>()`.
- `k_B = 1.380649e-23` (CODATA 2019). Same constant used by
  `compute_temperature`.

When `temperature == 0.0`, every velocity is `0.0_f32`. The runner takes
the explicit `T == 0.0` shortcut to avoid generating samples that all
scale by zero (and to skip the momentum-subtraction step).

### Momentum subtraction <!-- rq-8e239d36 -->

After all velocities are sampled (but before they are uploaded to the
device), the runner subtracts the per-axis centre-of-mass velocity so that
total momentum is zero.

```
total_mass = sum(masses[i] as f64 for i in 0..N)
if total_mass > 0.0 and N > 0:
    for axis in (x, y, z):
        p = sum(masses[i] as f64 * v[axis][i] as f64 for i in 0..N)
        v_com = p / total_mass
        for i in 0..N:
            v[axis][i] = ((v[axis][i] as f64) - v_com) as f32
```

When `N == 0` the subtraction is skipped (no particles to centre). When
`temperature == 0.0` the subtraction is skipped (every velocity is already
zero).

### Temperature rescaling <!-- inline --> <!-- rq-be568071 -->

After the momentum subtraction, when the centre-of-mass-removed velocity
field has thermal degrees of freedom (`N >= 2`), the runner rescales every
velocity by a single scalar so the realised kinetic energy matches the
flat-3N target `(3 * N / 2) * k_B * T`. This is the degrees-of-freedom
convention used by `compute_temperature` (see `io/log-output.md`), so the
generated array's reported temperature equals
`config.simulation.temperature`.

```
if N < 2:
    # no thermal degrees of freedom after centring
    for i in 0..N:
        for axis in (x, y, z):
            v[axis][i] = 0.0
else:
    ke = sum(0.5 * (masses[i] as f64) *
             ((v_x[i] as f64)^2 + (v_y[i] as f64)^2 + (v_z[i] as f64)^2)
             for i in 0..N)
    if ke > 0.0:
        target_ke = 1.5 * (N as f64) * k_B * T
        scale = sqrt(target_ke / ke)
        for i in 0..N:
            for axis in (x, y, z):
                v[axis][i] = ((v[axis][i] as f64) * scale) as f32
```

The kinetic-energy sum and the scale factor are computed in `f64`; each
rescaled component is stored back as `f32`. Reading the result back through
`compute_temperature` therefore recovers `T` to within f32
velocity-storage round-off.

The rescale targets the thermal degrees of freedom of the
centre-of-mass-removed velocity field, which exist only for `N >= 2`. The
edge cases:

- `temperature == 0.0` — velocity generation returns zero velocities
  before sampling, the momentum subtraction, or the rescale ever run.
- `N == 0` — there are no particles; the velocity arrays are empty.
- `N == 1` — a single centred particle is its own centre of mass and has
  no internal motion, so its velocity components are set to exactly zero.
  The `N == 1` momentum subtraction cancels the sampled velocity only up
  to a floating-point rounding residual; the rescale would otherwise
  amplify that residual into a full thermal velocity, so the velocities
  are zeroed explicitly rather than scaled.
- `N >= 2` with a positive post-subtraction kinetic energy — every
  component is multiplied by `scale`. The `ke > 0.0` guard also covers the
  measure-zero degenerate case in which every sampled velocity is
  identical, in which case the velocities are left as the momentum
  subtraction produced them.

The rescale is a single deterministic scalar multiply, so it preserves both
the determinism of velocity generation and the zero-total-momentum property
established by the momentum subtraction.

### Empty init velocities <!-- rq-e6552df6 -->

When `init_state.velocities = Some(_)`, the velocities are used directly:
the RNG is not consulted, and neither the momentum subtraction nor the
rescale is applied. The caller is presumed to have set velocities
deliberately; `compute_temperature` reports their flat-3N temperature as-is
(see `io/log-output.md`).

## Progress reporting <!-- rq-73fbb111 -->

The runner emits a progress line to stdout at the start of the loop, at
approximately each 1% completion (i.e. every `max(1, n_steps / 100)` steps),
and at completion. Each line has the form:

```
[dynamics] step 1000/10000 (10.0%) — 3.2e4 steps/sec
```

Step counts past `n_steps / 100` rounding always include the final step.
Progress lines are not emitted when stdout is not a TTY; in that case the
runner emits only the final-summary line. The TTY check is made once at
startup. (Implementations may use `std::io::IsTerminal` from std.)

## Final summary <!-- rq-d29872e4 -->

After flushing all writers, the runner emits a single line to stdout:

```
[dynamics] complete: 10000 steps in 312 ms (frames: 101, log rows: 101)
```

Where:

- "frames" is the number of trajectory frames written (zero when
  `trajectory_every == 0`).
- "log rows" is the number of CSV rows written, excluding the header
  (zero when `log_every == 0`).
- The elapsed time is the wall-clock interval between the start of the
  warm-up step and the end of the final `flush`, formatted in `ms` with no
  fractional digits when `>= 10 ms` and in `µs` when shorter.

## Feature API <!-- rq-02edd314 -->

The runner exposes two entry points: `run_simulation` (executes the
full pipeline) and `lint_simulation` (validates the setup phase and
returns a structured report). The CLI in `src/main.rs` is a thin
wrapper that dispatches between the two on the first CLI argument.

### Types <!-- rq-77c1d5d9 -->

- `RunnerError` — error type returned from `run_simulation`. Variants: <!-- rq-8ee27e27 -->
  - `Config(ConfigError)` — from `load_config`.
  - `InitState(InitStateError)` — from `load_init_state`.
  - `ParticleState(ParticleStateError)` — from `ParticleState::new` or
    buffer transfer.
  - `Gpu(GpuError)` — from `init_device`, buffer allocation, or kernel
    launch.
  - `Integrator(IntegratorError)` — from `Integrator::new` or the per-step
    methods on the chosen integrator variant (see
    `integration/framework.md`).
  - `TopologyFile(TopologyFileError)` — from `load_topology_file`
    (see `forces/topology.md`).
  - `ForceField(ForceFieldError)` — from `ForceField::new` or
    `ForceField::step` (see `forces/framework.md`).
  - `Trajectory(TrajectoryWriterError)` — from trajectory writer
    construction or `write_frame`/`flush`.
  - `Log(LogWriterError)` — from log writer construction or `write_row`/
    `flush`.
  - `Timings(TimingsError)` — from `Timings::new`, kernel-event
    recording, or `Timings::finalize`.
  - `TimingsWriter(TimingsWriterError)` — from `write_timings_file`.
  - `MissingArgs` — CLI invoked without the required `<config-path>` arg.
  - `OutputExists { path: PathBuf }` — pre-flight check before the init
    file is read; surfaces the same condition the writers detect at
    open time, but earlier.
  - `CellListBoxTooSmall { direction: &'static str, width: f32, required: f32 }`
    — when `config.neighbor_list` is `CellList`, the box read from the
    init file has a perpendicular width along some lattice direction
    that is shorter than `3 * (cutoff_max + r_skin)`, where `cutoff_max`
    is the largest cutoff across `config.pair_interactions`,
    `config.coulomb.cutoff`, and `config.spme.r_cut_real` (whichever
    are present). `direction` is one of `"a"`, `"b"`, `"c"`. The
    payload is filled by translating
    `SimulationBoxError::PerpendicularWidthTooSmall` returned by
    `sim_box.check_min_perpendicular_width(required)` (see
    `simulation-box.md`); the runner aggregates `cutoff_max`,
    computes `required = 3 * (cutoff_max + r_skin)`, and invokes the
    method.
  - `CuFftNonDeterministic { differences: usize }` — `init_device`'s
    cuFFT determinism smoke test detected `differences > 0` bytes
    differing between two consecutive R2C transforms of the same
    input. Raised only when `config.spme.is_some()`. Indicates the
    host's cuFFT installation does not meet SPME's reproducibility
    contract.

- `LintReport` — structured result of `lint_simulation`. All fields <!-- rq-a831fb00 -->
  are `pub`. Carries one `LintStage` per stage of the lint pipeline
  in execution order:
  - `stages: Vec<LintStage>` — the per-stage outcomes. Length is
    exactly `6` (one entry per stage label listed in *Lint flow*'s
    *Stage order*) so callers can index by position; the canonical
    label for each entry is also carried by the `LintStage` itself.
  - `overall: LintOverall` — `Ok` iff every stage's `status` is
    `Ok`, `Skipped`, or `NotChecked`; `Fail` iff at least one stage's
    status is `Fail`.

  Methods:
  - `ok(&self) -> bool` — `self.overall == LintOverall::Ok`.
  - `first_failure(&self) -> Option<&RunnerError>` — returns the
    `RunnerError` carried by the first `LintStage` whose status is
    `Fail`, or `None` when no stage failed.
  - `write_to(&self, w: &mut dyn std::io::Write) -> std::io::Result<()>`
    — emits the human-readable per-stage block documented in
    *Lint flow*'s *Output format*: a `[dynamics lint] OK` or
    `[dynamics lint] FAIL` header followed by one indented line per
    stage. Does **not** emit the trailing `error: ...` line; that
    line is written to stderr by the CLI wrapper using
    `first_failure().map(|e| format!("error: {e}"))`.

- `LintStage` — one row of the report. Fields: <!-- rq-334f5685 -->
  - `label: &'static str` — one of `"config"`, `"output paths"`,
    `"init"`, `"box/cutoff"`, `"topology"`, `"gpu"` (the
    column-1 text rendered by `write_to`).
  - `status: LintStatus` — see below.

- `LintStatus` — per-stage outcome. Variants: <!-- rq-ff560c3b -->
  - `Ok { detail: String }` — the stage ran to completion. `detail`
    is the human-readable description rendered after the label
    (e.g. `"resolved, 10000 particles, box 8.0e-9 × 8.0e-9 × 1.0e-8 m"`).
  - `Fail { detail: String, error: RunnerError }` — the stage ran
    and reported an error. `detail` is a short summary suitable for
    the per-stage line (rendered as `"FAIL — {detail}"` by
    `write_to`); `error` is the canonical structured error and is
    the value returned by `LintReport::first_failure` (for the first
    such stage).
  - `Skipped { reason: String }` — an earlier stage failed (or this
    stage was not applicable). Examples: `"earlier check failed"`,
    `"not supplied"`, `"not applicable (mode = all-pairs)"`.
  - `NotChecked { reason: String }` — the stage was deliberately not
    attempted by the current lint mode. The `gpu` stage carries this
    status when `with_gpu = false`, with `reason = "re-run with
    --with-gpu"`.

- `LintOverall` — `enum { Ok, Fail }`. `LintReport::overall` is `Ok` <!-- rq-30c21c70 -->
  iff no stage has a `Fail` status.

### Functions <!-- rq-e5e4b048 -->

- `run_simulation(config_path: &Path) -> Result<RunSummary, RunnerError>` <!-- rq-1fc57c00 -->
  - Convenience wrapper. Equivalent to
    `run_simulation_with_registries(config_path, &Registries::with_builtins())`.
  - Used by `main.rs` and by every caller that wants the default
    built-in registries.

- `run_simulation_with_registries(config_path: &Path, registries: &Registries) -> Result<RunSummary, RunnerError>` <!-- rq-a71cef31 -->
  - Executes the entire runner flow described above, dispatching
    every `[integrator]`, `[thermostat]`, `[barostat]`,
    `[[constraint_types]]`, and force-slot `kind` through
    `registries`. Used by callers that have registered custom
    builders.
  - Reads and parses the config file via `load_config_raw` (the
    structural-validation-only entry point), then runs
    `Config::validate_against(&registries)` and
    `Config::validate_constraint_compatibility(&registries, has_constraints)`
    with the caller-supplied registries. Any `ConfigError` returned
    by either method is wrapped in `RunnerError::Config`.
  - Uses `registries.potentials` instead of constructing
    `PotentialRegistry::with_builtins()` internally.
  - Returns a `RunSummary` carrying the step count, frame count, log
    row count, and elapsed time on success.

- `RunSummary` fields: <!-- rq-5c1cfc93 -->
  - `n_steps: u64`
  - `frames_written: u64`
  - `log_rows_written: u64`
  - `elapsed_micros: u128`

- `lint_simulation(config_path: &Path, with_gpu: bool) -> LintReport` <!-- rq-4ff84310 -->
  - Convenience wrapper. Equivalent to
    `lint_simulation_with_registries(config_path,
    &Registries::with_builtins(), with_gpu)`.
  - Returns a fully populated `LintReport` rather than a `Result`:
    every error mode of the underlying loaders is captured as a
    `LintStatus::Fail` carrying the structured `RunnerError`. Callers
    inspect `report.ok()` or `report.first_failure()`.
  - The function never panics on user input. CPU-only mode performs
    no GPU access at all and is safe to call on a host without a
    CUDA device.

- `lint_simulation_with_registries(config_path: &Path, registries: &Registries, with_gpu: bool) -> LintReport` <!-- rq-9ed993de -->
  - Runs every stage of *Lint flow* against `registries`. The
    config-load stage uses `load_config_raw` followed by
    `config.validate_against(registries)` and
    `config.validate_constraint_compatibility(registries,
    has_constraints)` so per-kind validation runs against the
    caller-supplied registries (matching what
    `run_simulation_with_registries` does).
  - Short-circuits on the first stage that returns `Fail`: that
    stage's status is set to `Fail { detail, error }` and every
    subsequent stage's status is set to
    `Skipped { reason: "earlier check failed" }`.
  - When `with_gpu` is `false` the `gpu` stage is recorded as
    `NotChecked { reason: "re-run with --with-gpu" }`
    unconditionally. When `with_gpu` is `true` and every preceding
    stage passed, the function performs the steps listed under
    *Lint flow*'s *`--with-gpu` stages* and records the outcome.

- `Registries` — bundled handle to every open builder registry the <!-- rq-74bb02cc -->
  runner consults. Lives at `dynamics::Registries` (the crate root,
  so it does not appear to belong to any single subsystem). Fields:
  - `integrators: IntegratorRegistry`
  - `thermostats: ThermostatRegistry`
  - `barostats: BarostatRegistry`
  - `constraint_types: ConstraintRegistry`
  - `potentials: PotentialRegistry`

  Constructors:
  - `Registries::with_builtins() -> Registries` — every inner
    registry is `XRegistry::with_builtins()`. Used by
    `run_simulation` (the default-built-ins convenience wrapper).
  - `Registries::new() -> Registries` — every inner registry is
    `XRegistry::new()` (empty). Callers that want full control
    over registration order start from `new()` and register every
    builder they need explicitly.

  Convenience registration methods, each forwarding to the matching
  inner registry's `register`:
  - `Registries::register_integrator(&mut self, builder: Box<dyn IntegratorBuilder>)`
  - `Registries::register_thermostat(&mut self, builder: Box<dyn ThermostatBuilder>)`
  - `Registries::register_barostat(&mut self, builder: Box<dyn BarostatBuilder>)`
  - `Registries::register_constraint_type(&mut self, builder: Box<dyn ConstraintBuilder>)`
  - `Registries::register_potential(&mut self, builder: Box<dyn PotentialBuilder>)`

  Custom builders compose with the built-ins via the standard pattern:

  ```rust
  let mut registries = Registries::with_builtins();
  registries.register_integrator(Box::new(MyCustomIntegratorBuilder));
  registries.register_potential(Box::new(MyCustomPotentialBuilder));
  let summary = run_simulation_with_registries(&config_path, &registries)?;
  ```

  Or from an empty bundle when the caller wants no built-ins:

  ```rust
  let mut registries = Registries::new();
  registries.register_integrator(Box::new(VelocityVerletBuilder));
  registries.register_potential(Box::new(LennardJonesBuilder));
  // ... etc.
  ```

  The inner registry types remain accessible by their own paths
  (`dynamics::integrator::IntegratorRegistry`,
  `dynamics::forces::PotentialRegistry`, etc.) for callers that want
  to construct or compose a single registry without going through
  the bundle.

- `main(args: Vec<String>) -> ExitCode` (in `src/main.rs`) <!-- rq-f7e279ee -->
  - Parses the CLI and dispatches on the first argument:
    - `run <config-path>` calls `run_simulation` (the built-ins
      convenience wrapper), prints any error to stderr and the
      final-summary line to stdout, and returns the exit code
      described in *CLI*'s *`run` subcommand*.
    - `lint <config-path> [--with-gpu]` calls
      `lint_simulation(config_path, with_gpu)`, writes the report
      to stdout via `LintReport::write_to(&mut io::stdout())`,
      writes `error: {first_failure}` to stderr when
      `!report.ok()`, and returns the exit code described in
      *CLI*'s *`lint` subcommand* (`0` on `ok()`, `1` otherwise).
  - Any other invocation prints the usage block from
    *Usage error messages* and exits `1`.
  - The bundled CLI does not expose any mechanism for registering
    custom builders; a binary that wants custom builders is a Rust
    program depending on `dynamics` that constructs its own
    `Registries` and calls `run_simulation_with_registries` or
    `lint_simulation_with_registries` directly.

## Determinism guarantees <!-- rq-0485e79f -->

The runner preserves the project's bit-wise reproducibility invariant:

- Velocity generation is fully deterministic in `(seed, temperature,
  masses, N)` (the masses and N derive from the config and init file).
- Every device kernel launch is on the default stream of the `Arc<CudaDevice>`
  carried by the `GpuContext` from `init_device()`; no other streams are
  introduced.
- `compute_kinetic_energy` sums in particle order, so the log values are
  byte-identical across runs on the same GPU.
- The Maxwell-Boltzmann RNG is `ChaCha8Rng::seed_from_u64(seed)` and is
  consumed in the order specified in *Sampling order*.

Two invocations of `dynamics run sim.toml` with identical inputs on the
same hardware produce trajectory files and log files that are byte-identical.

## Out of Scope <!-- rq-1bf226c9 -->

- Restart files and `dynamics resume` (separate planned feature).
- Multi-GPU and multi-host execution.
- Per-step force-field switching, time-varying parameters.
- Thermostat / barostat composition logic. The runner chains the
  three slot handles in a fixed order; per-slot algorithms live in
  the respective requirements files under `integration/`.
- Multi-type simulations. The runner rejects configs with more than one
  `[[particle_types]]` (`MultiTypeUnsupported`), pending a future
  multi-type LJ kernel.
- Energy minimisation / pre-equilibration steps.
- Per-particle mass overrides in the init file.
- Subsampling the trajectory in space (region selectors, type filters).
- CLI flags beyond the positional `<config-path>`. No `--seed`, no
  `--steps`, no `--config`. Every parameter is in the file.
- Live console output beyond the simple progress line (e.g. interactive
  energy plots, TUI dashboards).
- Validation that `pair_interactions[0].cutoff <= min(box edge) / 2`.
  The current LJ kernel and PBC code do not enforce this; the runner
  inherits whatever behaviour those layers produce when the cutoff
  exceeds the half-box.
- Removing per-axis angular momentum during velocity generation.
  Removing linear momentum is sufficient for the current scale; angular
  momentum drifts on the order of the temperature.

---

## Gherkin Scenarios <!-- rq-459d8e74 -->

```gherkin
Feature: dynamics run simulation runner

  Background:
    Given a CUDA-capable GPU available as device 0
    And a temporary directory tmp

  # --- CLI ---

  @rq-1ae622bb
  Scenario: Run a valid minimal config to completion
    Given tmp/sim.in.toml is a valid one-type config with n_steps=10, dt=1.0e-15,
      seed=42, temperature=0.0, trajectory_every=5, log_every=5
    And tmp/sim.in.xyz is a valid init file with N=2 particles inside the box, no velocities
    When dynamics is invoked with arguments ["run", "tmp/sim.in.toml"]
    Then it exits with code 0
    And tmp/sim.out.xyz exists and contains 3 frames (steps 0, 5, 10)
    And tmp/sim.out.log exists and contains a header plus 3 rows (steps 0, 5, 10)
    And the final-summary line on stdout matches "[dynamics] complete: 10 steps in .* (frames: 3, log rows: 3)"

  @rq-2a36b95f
  Scenario: Missing CLI argument prints usage and exits 1
    When dynamics is invoked with arguments []
    Then it exits with code 1
    And stderr contains "usage: dynamics run <config-path>"

  @rq-2214f0a1
  Scenario: Unrecognised subcommand prints usage and exits 1
    When dynamics is invoked with arguments ["benchmark"]
    Then it exits with code 1
    And stderr contains "usage: dynamics run <config-path>"

  # --- Config and init failures ---

  @rq-b746e796
  Scenario: Config does not exist
    When dynamics is invoked with arguments ["run", "tmp/no-such.in.toml"]
    Then it exits with code 1
    And stderr contains "error: " and "no-such.in.toml"

  @rq-6606584b
  Scenario: Config rejected by load_config
    Given tmp/sim.in.toml has schema_version=2
    When dynamics is invoked with arguments ["run", "tmp/sim.in.toml"]
    Then it exits with code 1
    And stderr contains "UnsupportedSchemaVersion"

  @rq-91f5f34e
  Scenario: Config rejected by the filename convention
    Given tmp/sim.toml is otherwise valid (but lacks the `.in.toml` suffix)
    When dynamics is invoked with arguments ["run", "tmp/sim.toml"]
    Then it exits with code 1
    And stderr contains "InvalidConfigFilename" and "sim.toml"
    And the file at tmp/sim.toml was not opened

  @rq-f6927716
  Scenario: Init file rejected by load_init_state
    Given tmp/sim.in.toml references init="bad.xyz"
    And tmp/bad.xyz has a position outside the primary cell
    When dynamics is invoked with arguments ["run", "tmp/sim.in.toml"]
    Then it exits with code 1
    And stderr contains "PositionOutsideBox"

  # --- Output overwrite ---

  @rq-d9a98e51
  Scenario: Pre-flight refuses to overwrite existing trajectory
    Given tmp/sim.in.toml is valid with trajectory_every=5
    And tmp/sim.out.xyz already exists
    When dynamics is invoked with arguments ["run", "tmp/sim.in.toml"]
    Then it exits with code 1
    And stderr contains "OutputExists" and "sim.out.xyz"
    And the init file is not read (verified by check that load_init_state was not entered)

  @rq-52c483c0
  Scenario: Pre-flight refuses to overwrite existing log
    Given tmp/sim.in.toml is valid with log_every=5
    And tmp/sim.out.log already exists
    When dynamics is invoked with arguments ["run", "tmp/sim.in.toml"]
    Then it exits with code 1
    And stderr contains "OutputExists" and "sim.out.log"

  @rq-acbbd59a
  Scenario: Disabled trajectory and log outputs are not checked, but timings is
    Given tmp/sim.in.toml has trajectory_every=0 and log_every=0
    And tmp/sim.out.xyz and tmp/sim.out.log both already exist with arbitrary content
    And tmp/sim.out.timings does not exist
    When dynamics is invoked with arguments ["run", "tmp/sim.in.toml"]
    Then it exits with code 0
    And tmp/sim.out.xyz is unchanged
    And tmp/sim.out.log is unchanged
    And tmp/sim.out.timings exists

  @rq-fc523f30
  Scenario: Pre-flight refuses to overwrite existing timings file
    Given tmp/sim.in.toml is valid
    And tmp/sim.out.timings already exists with arbitrary content
    When dynamics is invoked with arguments ["run", "tmp/sim.in.toml"]
    Then it exits with code 1
    And stderr contains "OutputExists" and "sim.out.timings"

  # --- Velocity generation ---

  @rq-621ce7b6
  Scenario: Sampled velocities round-trip to the configured temperature
    Given tmp/sim.in.toml has seed=1, temperature=300.0, n_steps=0
    And tmp/sim.in.xyz has 100 particles with positions but no velocities
    When dynamics is invoked
    Then it exits with code 0
    And the step-0 log row's kinetic_energy is greater than 0
    And the step-0 log row's temperature equals 300.0 within a relative tolerance of 1e-4
      (the rescale makes this an exact round-trip up to f32 velocity-storage round-off,
       not a statistical estimate)

  @rq-04fda32f
  Scenario: Explicit init velocities override sampled velocities
    Given tmp/sim.in.xyz declares velocities of (1.0, 0.0, 0.0) m/s for every particle
    And tmp/sim.in.toml has temperature=300.0 (would normally sample)
    When dynamics is invoked
    Then the step-0 log row's kinetic_energy equals 0.5 * sum(m_i) * 1.0^2 exactly (no RNG consumed)

  @rq-f8df9364
  Scenario: Velocity generation is deterministic in the seed
    Given two identical configs and init files, both with no velocities in the init
    When dynamics is invoked on each
    Then the two resulting log files are byte-identical
    And the two resulting trajectory files are byte-identical

  @rq-81b241e7
  Scenario: Different seeds produce different velocities
    Given two configs identical except seed=1 vs seed=2
    When dynamics is invoked on each
    Then the two resulting log files differ on the step-0 row's kinetic_energy

  @rq-3c17477d
  Scenario: Total momentum after generation is zero (within f32 round-off)
    Given a config with seed=42, temperature=300.0, N=64 particles, equal masses
    When dynamics is invoked with n_steps=0
    And the per-axis momenta are computed from the step-0 frame velocities
    Then |p_x|, |p_y|, |p_z| are each less than 1e-3 times the typical thermal momentum

  @rq-f7e2d0f1
  Scenario: temperature=0 yields exactly zero velocities and skips momentum subtraction and rescaling
    Given a config with temperature=0.0 and N=4
    When dynamics is invoked with n_steps=0
    Then every velocity component written to the step-0 frame is exactly 0.0_f32

  @rq-d82ce4aa
  Scenario: Single-particle generated velocities are zeroed for lack of thermal degrees of freedom
    Given a config with seed=1, temperature=300.0, and N=1, with no init velocities
    When dynamics is invoked with n_steps=0
    Then the rescale step sets the single particle's velocity components to exactly zero
      (a centred one-particle system has no thermal degrees of freedom)
    And the step-0 log row's kinetic_energy is exactly 0.0 and temperature is exactly 0.0

  # --- Timestep loop ---

  @rq-985230a5
  Scenario: Loop executes exactly n_steps integration steps
    Given a config with n_steps=7 and trajectory_every=1 and log_every=1
    When dynamics is invoked
    Then the trajectory contains 8 frames (steps 0..=7)
    And the log contains 8 rows (steps 0..=7)

  @rq-18f7fce9
  Scenario: trajectory_every > n_steps writes only the step-0 frame
    Given a config with n_steps=5 and trajectory_every=100
    When dynamics is invoked
    Then the trajectory contains exactly 1 frame at step=0

  @rq-56ad97f1
  Scenario: log_every > n_steps writes only the step-0 row
    Given a config with n_steps=5 and log_every=100
    When dynamics is invoked
    Then the log contains the header plus exactly 1 row at step=0

  @rq-ff707382
  Scenario: n_steps = 0 writes only step-0 outputs
    Given a config with n_steps=0, trajectory_every=10, log_every=10
    When dynamics is invoked
    Then the trajectory contains exactly 1 frame at step=0
    And the log contains the header plus exactly 1 row at step=0

  @rq-fe1eaade
  Scenario: Reproducibility byte-for-byte across two identical runs
    Given a config with n_steps=200 and explicit init velocities
    When dynamics is invoked twice on the same files
    Then the two resulting trajectory files are byte-identical
    And the two resulting log files are byte-identical

  # --- Integrator selection ---

  @rq-9eb167f0
  Scenario: Lossless mode selects the lossless integrator kernels
    Given a config with integrator.kind="velocity-verlet" and lossless=true
    When dynamics is invoked
    Then it exits with code 0
    And the simulation completes without GPU error

  @rq-a97789e6
  Scenario: Lossy mode is the default for velocity-Verlet
    Given a config with [integrator] kind="velocity-verlet" and no lossless field
    When the config is loaded
    Then config.integrator.kind equals "velocity-verlet"
    And config.integrator.params.get("lossless") equals Some(toml::Value::Boolean(false))

  @rq-00cbbf51
  Scenario: Langevin BAOAB runs end-to-end through the runner
    Given a valid config with [integrator] kind="langevin-baoab",
      friction=1.0e12, temperature=300.0, seed=42, n_steps=5
    When dynamics is invoked
    Then it exits with code 0
    And the trajectory and log files exist
    And the timings file contains rows for KernelStage::LangevinKickHalf,
      KernelStage::LangevinDriftHalf, and KernelStage::LangevinOuStep

  @rq-88e3ac79
  Scenario: Switching integrator.kind changes the trajectory
    Given two configs identical except [integrator] kind="velocity-verlet" vs
      [integrator] kind="langevin-baoab" (with friction=1.0e12, temperature=300.0, seed=1)
    When dynamics is invoked on each
    Then the two trajectory files differ

  # --- Multi-type restriction ---

  @rq-34db7b7b
  Scenario: Refuse to run with multiple types
    Given tmp/sim.in.toml declares particle_types ["Ar", "Kr"] and all three pair interactions
    When dynamics is invoked
    Then it exits with code 1
    And stderr contains "MultiTypeUnsupported"

  # --- Empty system ---

  @rq-d065447f
  Scenario: Run an empty (N=0) simulation
    Given tmp/sim.in.xyz declares N=0 with a valid Lattice
    And tmp/sim.in.toml is otherwise valid with n_steps=5, trajectory_every=1, log_every=1
    When dynamics is invoked
    Then it exits with code 0
    And the trajectory contains 6 frames each with N=0 data rows
    And every log row has kinetic_energy=0 and temperature=0

  # --- Neighbor list / box-size compatibility ---

  @rq-4b4f85c7
  Scenario: Box too small for cell-list rejected before forces are constructed
    Given tmp/sim.in.toml has [neighbor_list] mode="cell-list" r_skin=1.0e-10
      and one [[pair_interactions]] with cutoff=1.0e-9
    And tmp/sim.in.xyz has an orthorhombic box with lx=2.0e-9, ly=lz=5.0e-9
    When dynamics is invoked with arguments ["run", "tmp/sim.in.toml"]
    Then it exits with code 1
    And stderr contains "CellListBoxTooSmall" and "a" and "2"

  @rq-0cb544f4
  Scenario: Coulomb cutoff participates in box-too-small check
    Given tmp/sim.in.toml has [neighbor_list] mode="cell-list" r_skin=1.0e-10
      and one [[pair_interactions]] with cutoff=5.0e-10
      and a [coulomb] table with cutoff=2.0e-9 (the larger of the two)
    And tmp/sim.in.xyz has an orthorhombic box with lx=ly=lz=5.0e-9
      (so 3*(2.0e-9 + 1.0e-10) = 6.3e-9 > 5.0e-9)
    When dynamics is invoked with arguments ["run", "tmp/sim.in.toml"]
    Then it exits with code 1
    And stderr contains "CellListBoxTooSmall"

  @rq-21d27f06
  Scenario: All-pairs mode skips the box-too-small check
    Given tmp/sim.in.toml has [neighbor_list] mode="all-pairs"
      and one [[pair_interactions]] with cutoff=1.0e-9
    And tmp/sim.in.xyz has box edges (2.0e-9, 2.0e-9, 2.0e-9)
    When dynamics is invoked
    Then it exits with code 0

  # --- GPU initialisation failure ---

  @rq-57c1b6a3
  Scenario: No GPU available
    Given no CUDA-capable GPU is present
    When dynamics is invoked with a valid config
    Then it exits with code 1
    And stderr contains "Gpu" or "CUDA"

  # --- Run summary ---

  @rq-f4a85dda
  Scenario: RunSummary reflects what was actually written
    Given a config with n_steps=100, trajectory_every=25, log_every=10
    When run_simulation is called from a library client
    Then the returned RunSummary has n_steps=100
    And frames_written equals 5 (steps 0, 25, 50, 75, 100)
    And log_rows_written equals 11 (steps 0, 10, 20, ..., 100)
    And elapsed_micros is greater than 0

  # --- Output of an interrupted run ---

  @rq-889076d5
  Scenario: Kernel failure mid-loop returns exit code 2
    Given a config with parameters that would cause a kernel launch failure on step 5
    When dynamics is invoked
    Then it exits with code 2
    And the partial trajectory and log files exist with frames/rows up to the last successful write

  # --- User-registered builders ---

  @rq-b5e263e1
  Scenario: run_simulation defers to run_simulation_with_registries with built-ins
    Given a valid config that uses only built-in slot kinds
    When run_simulation(&config_path) is called
    Then it returns the same RunSummary as
      run_simulation_with_registries(&config_path, &Registries::with_builtins())

  @rq-923fc84f
  Scenario: run_simulation_with_registries dispatches an integrator from a user-registered builder
    Given a valid config whose [integrator] section sets kind = "my-stub"
    And a Registries bundle constructed via Registries::with_builtins()
      then `register_integrator(Box::new(MyStubBuilder))` where
      MyStubBuilder::kind_name() returns "my-stub"
    When run_simulation_with_registries(&config_path, &registries) is called
    Then it returns Ok(summary)
    And the dispatched integrator is the one constructed by MyStubBuilder
      (verified e.g. by a side-channel counter the builder closes over)

  @rq-fb917fd5
  Scenario: run_simulation_with_registries dispatches a potential from a user-registered builder
    Given a valid config that activates the conditions for a user-defined Potential
    And a Registries bundle whose `register_potential(Box::new(MyPotentialBuilder))`
      has been called after with_builtins()
    When run_simulation_with_registries(&config_path, &registries) is called
    Then it returns Ok(summary)
    And the constructed ForceField's slot list contains the MyPotentialBuilder slot
      at the position determined by the registry's iteration order

  @rq-0069339b
  Scenario: A custom-kind integrator with an empty Registries fails with UnknownKind
    Given a valid config whose [integrator] section sets kind = "my-stub"
    And Registries::new() with no integrator builders registered
    When run_simulation_with_registries(&config_path, &registries) is called
    Then it returns Err(RunnerError::Config(ConfigError::UnknownKind {
      slot: "integrator", kind: "my-stub" }))

  @rq-eb9e43e7
  Scenario: run_simulation rejects a custom-kind config because its built-in registries do not match
    Given a valid config whose [integrator] section sets kind = "my-stub"
    When run_simulation(&config_path) is called (which uses Registries::with_builtins())
    Then it returns Err(RunnerError::Config(ConfigError::UnknownKind {
      slot: "integrator", kind: "my-stub" }))
    (the caller's recourse is to use run_simulation_with_registries with a bundle
     that has the custom builder registered)

  @rq-d9726854
  Scenario: Registries::new() starts every inner registry empty
    Given let registries = Registries::new()
    Then registries.integrators.builders is empty
    And registries.thermostats.builders is empty
    And registries.barostats.builders is empty
    And registries.constraint_types.builders is empty
    And registries.potentials.builders is empty

  @rq-5f8f7d00
  Scenario: Registries::with_builtins() pre-populates every inner registry
    Given let registries = Registries::with_builtins()
    Then registries.integrators.builders is non-empty
    And registries.thermostats.builders is non-empty
    And registries.barostats.builders is non-empty
    And registries.constraint_types.builders is non-empty
    And registries.potentials.builders is non-empty

  @rq-bbb25583
  Scenario: register_potential appends to the bundle's PotentialRegistry
    Given let mut registries = Registries::with_builtins()
    And the initial length of registries.potentials.builders is N
    When registries.register_potential(Box::new(MyPotentialBuilder)) is called
    Then registries.potentials.builders has length N + 1
    And registries.potentials.builders[N] is the registered MyPotentialBuilder

  # --- Lint subcommand ---

  @rq-c52b8ece
  Scenario: Lint of a valid CPU-only config succeeds with the gpu stage not checked
    Given tmp/sim.in.toml is a valid one-type config (no `--with-gpu`)
    And tmp/sim.in.xyz is a valid init file matching the config
    And none of tmp/sim.out.xyz, tmp/sim.out.log, tmp/sim.out.timings exist
    When dynamics is invoked with arguments ["lint", "tmp/sim.in.toml"]
    Then it exits with code 0
    And stdout begins with "[dynamics lint] OK"
    And stdout contains a line whose label is "config" and whose description names tmp/sim.in.toml
    And stdout contains a line whose label is "output paths" and whose description is "none pre-exist"
    And stdout contains a line whose label is "init" and whose description names the particle count and box dimensions
    And stdout contains a line whose label is "topology" and whose description is "not supplied"
    And stdout contains a line whose label is "gpu" and whose description is "not checked (re-run with --with-gpu)"
    And no file was written (tmp/sim.out.xyz, tmp/sim.out.log, tmp/sim.out.timings still do not exist)

  @rq-b54d8111
  Scenario: Lint reports filename-convention violations under the config stage
    Given a valid config body is written to tmp/sim.toml (no `.in.toml` suffix)
    When dynamics is invoked with arguments ["lint", "tmp/sim.toml"]
    Then it exits with code 1
    And stdout begins with "[dynamics lint] FAIL"
    And stdout contains a line whose label is "config" and whose description begins with "FAIL —"
    And stdout contains a line whose label is "init" and whose description is "skipped (earlier check failed)"
    And stderr contains "error: " and "InvalidConfigFilename" or "does not end in `.in.toml`"

  @rq-37bfb0fc
  Scenario: Lint reports a pre-existing output file under the output paths stage
    Given tmp/sim.in.toml is valid with trajectory_every=5
    And tmp/sim.in.xyz is a valid init file
    And tmp/sim.out.xyz already exists with arbitrary content
    When dynamics is invoked with arguments ["lint", "tmp/sim.in.toml"]
    Then it exits with code 1
    And stdout contains a line whose label is "output paths" and whose description begins with "FAIL —" and names tmp/sim.out.xyz
    And stdout contains a line whose label is "init" and whose description is "skipped (earlier check failed)"
    And stderr contains "error: " and "OutputExists" and "sim.out.xyz"
    And tmp/sim.out.xyz is unchanged

  @rq-a479680a
  Scenario: Lint reports a box-too-small failure under the box/cutoff stage
    Given tmp/sim.in.toml has [neighbor_list] mode="cell-list" r_skin=1.0e-10
      and one [[pair_interactions]] with cutoff=1.0e-9
    And tmp/sim.in.xyz has an orthorhombic box with lx=2.0e-9, ly=lz=5.0e-9
    When dynamics is invoked with arguments ["lint", "tmp/sim.in.toml"]
    Then it exits with code 1
    And stdout contains a line whose label is "init" and whose description names the box dimensions
    And stdout contains a line whose label is "box/cutoff" and whose description begins with "FAIL —" and names direction `a`
    And stderr contains "error: " and "CellListBoxTooSmall"

  @rq-cdf1cd7c
  Scenario: Lint marks box/cutoff as not applicable in all-pairs mode
    Given tmp/sim.in.toml has [neighbor_list] mode="all-pairs"
    And tmp/sim.in.xyz is a valid init file with an arbitrary box
    When dynamics is invoked with arguments ["lint", "tmp/sim.in.toml"]
    Then it exits with code 0
    And stdout contains a line whose label is "box/cutoff" and whose description contains "not applicable" and "all-pairs"

  @rq-60433fcd
  Scenario: Lint reports a topology-load failure under the topology stage
    Given tmp/sim.in.toml sets topology = "tmp/bad.in.topology"
    And tmp/bad.in.topology declares a bond with an atom index out of range
    When dynamics is invoked with arguments ["lint", "tmp/sim.in.toml"]
    Then it exits with code 1
    And stdout contains a line whose label is "topology" and whose description begins with "FAIL —"
    And stderr contains "error: " and a description of the topology error

  @rq-2b614db6
  Scenario: Lint with --with-gpu runs init_device and surfaces a cuFFT smoke-test failure
    Given a host whose cuFFT implementation does not satisfy SPME's determinism contract
    And tmp/sim.in.toml configures [spme] (so the smoke test runs)
    When dynamics is invoked with arguments ["lint", "tmp/sim.in.toml", "--with-gpu"]
    Then it exits with code 1
    And stdout contains a line whose label is "gpu" and whose description begins with "FAIL —"
    And stderr contains "error: " and "CuFftNonDeterministic"

  @rq-dba8d096
  Scenario: Lint with --with-gpu reports successful GPU setup
    Given tmp/sim.in.toml is a valid one-type config
    And tmp/sim.in.xyz is a valid init file
    And a CUDA-capable GPU is available
    When dynamics is invoked with arguments ["lint", "tmp/sim.in.toml", "--with-gpu"]
    Then it exits with code 0
    And stdout begins with "[dynamics lint] OK"
    And stdout contains a line whose label is "gpu" and whose description contains "init_device OK" and "ForceField"

  @rq-a4fbc3a4
  Scenario: Lint never creates output files
    Given tmp/sim.in.toml is valid and exists
    And tmp/sim.in.xyz is a valid init file
    And none of tmp/sim.out.xyz, tmp/sim.out.log, tmp/sim.out.timings exist
    When dynamics is invoked with arguments ["lint", "tmp/sim.in.toml"]
    Then it exits with code 0
    And tmp/sim.out.xyz still does not exist
    And tmp/sim.out.log still does not exist
    And tmp/sim.out.timings still does not exist

  @rq-69bf814f
  Scenario: Lint short-circuits on the first failure
    Given tmp/sim.in.toml is valid
    And tmp/sim.in.xyz does NOT exist (init load will fail)
    And tmp/sim.out.timings does NOT exist (the output-paths stage would pass)
    When dynamics is invoked with arguments ["lint", "tmp/sim.in.toml"]
    Then it exits with code 1
    And stdout contains a line whose label is "config" and whose description does NOT begin with "FAIL"
    And stdout contains a line whose label is "init" and whose description begins with "FAIL —"
    And stdout contains a line whose label is "box/cutoff" and whose description is "skipped (earlier check failed)"
    And stdout contains a line whose label is "topology" and whose description is "skipped (earlier check failed)"
    And stdout contains a line whose label is "gpu" and whose description is "not checked (re-run with --with-gpu)"

  @rq-d87f15bd
  Scenario: Lint without --with-gpu does not open a GPU device
    Given a host with no CUDA-capable GPU
    And tmp/sim.in.toml and tmp/sim.in.xyz are otherwise valid
    When dynamics is invoked with arguments ["lint", "tmp/sim.in.toml"]
    Then it exits with code 0
    And no CUDA driver call was made (verified by spy on the GpuContext constructor, or by lack of CUDA dynamic-library load)

  @rq-8044a6f5
  Scenario: LintReport API short-circuits and carries the structured error
    Given a config whose init file declares a position outside the primary cell
    When lint_simulation(&path, with_gpu=false) is called from a library client
    Then the returned LintReport has overall == LintOverall::Fail
    And report.first_failure() returns Some(RunnerError::InitState(_))
    And the stage labelled "init" has status LintStatus::Fail
    And the stage labelled "box/cutoff" has status LintStatus::Skipped { reason: "earlier check failed" }
    And the stage labelled "gpu" has status LintStatus::NotChecked { reason: "re-run with --with-gpu" }
```
