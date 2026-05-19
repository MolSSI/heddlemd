# Command-Line Interface

The `dynamics` binary has three subcommands: `run` (executes the
simulation), `lint` (validates inputs without running), and `analyze`
(post-processes a trajectory written by `run`).

## Usage

```
dynamics run     <config-path>
dynamics lint    <config-path> [--with-gpu]
dynamics analyze <analysis-path>
```

Invoking `dynamics` with no arguments, an unknown subcommand, or a
subcommand without its required path argument prints

```
usage: dynamics run     <config-path>
       dynamics lint    <config-path> [--with-gpu]
       dynamics analyze <analysis-path>
```

to stderr and exits with code `1`.

## `dynamics run <config-path>`

Loads the TOML config at `<config-path>` and runs the simulation it
describes to completion.

- `<config-path>` is the path to a [TOML config file](../guide/configuration.md).
  Relative paths are resolved against the current working directory.
- All output paths in the config (`init`, `output.trajectory_path`,
  `output.log_path`, `output.timings_path`, optional `topology`) are
  resolved relative to the *config file's* directory, not the current
  working directory. Absolute paths are honored as-is.

### What it does, in order

1. Parses the CLI arguments.
2. Loads and validates the TOML config.
3. Checks that none of the enabled output files already exist. Failing
   this check up front means the runner never starts a long run that
   would be unable to write its results.
4. Loads the init file and, when supplied, the topology file.
5. Initialises CUDA and uploads the particle state to the GPU.
6. Generates initial velocities (only when the init file omits them).
7. For each `[[phase]]` in order: opens the phase's trajectory and
   log files, runs the timestep loop for `phase.n_steps` iterations
   writing frames and rows at the configured cadences, flushes the
   files, then writes the phase's `.timings` file before moving on.

### On success

Prints one line per phase plus a final aggregate on stdout:

```
[dynamics] phase `<name>`: <N> steps in <T> ms (frames: <F>, log rows: <R>)
...
[dynamics] complete: <total_N> steps in <total_T> ms
```

and exits with code `0`. For very short runs (`< 10 ms`) the elapsed
time is shown in microseconds instead (`<T> µs`).

### On failure

Prints a single line to stderr beginning with `error: `, followed by a
human-readable description that names the offending file or field where
applicable, and exits with the appropriate non-zero code.

### Exit codes

| Code | Meaning |
|------|---------|
| `0`  | Simulation completed; every requested step ran and every output flushed. |
| `1`  | Any error *before* the integration loop started: malformed CLI args, config-load failure, init-state load failure, output-file overwrite check failure, GPU initialisation failure, box-vs-cutoff compatibility failure, cuFFT determinism smoke-test failure. |
| `2`  | Any error *during* the integration loop: a kernel launch failed, a write to the trajectory or log failed, or a device-to-host download failed. |

The split between exit code `1` and `2` makes it cheap for a wrapper
script to distinguish input mistakes (re-edit the config) from
mid-flight failures (likely transient: re-run, check GPU health).

## `dynamics lint <config-path> [--with-gpu]`

Validates the config and its referenced inputs against every error the
runner can detect without executing the integration loop, then exits
without writing any output files. Designed for HPC contexts where a
long submission queue makes trial-and-error iteration expensive: run
`dynamics lint` on a login node before queueing a job, and fix any
reported issues up front instead of after the queue eventually grants
GPU time.

The subcommand dispatches on the file extension:

- `<path>.in.toml` runs the *simulation lint pipeline* (described
  in *What it checks* below).
- `<path>.in.analysis` runs the *analyze lint pipeline* (described
  under [Analysis](../guide/analysis.md) and
  `rqm/analysis/framework.md`).

- `--with-gpu` (optional) is accepted only for the simulation lint
  pipeline; it extends the lint to include device initialisation,
  the cuFFT determinism smoke test (when SPME is configured), the
  host-to-device upload, slot construction, and the force-field
  allocation. Passing `--with-gpu` together with a
  `.in.analysis` path is rejected (analysis is CPU-only in v1).
- Lint writes no files at any time. Pre-existing output files are
  detected with `Path::exists()`; the filesystem is otherwise
  unchanged.
- Stops at the first failed check (short-circuit). Subsequent stages
  appear as `skipped (earlier check failed)` in the per-stage report.

### What it checks

Stages run in the order below. Each stage reports `Ok`, `FAIL`,
`Skipped`, or `not checked (re-run with --with-gpu)` in the per-stage
report.

1. **`config`** — TOML parse, the `.in.toml` filename convention,
   every per-field domain check, path-collision check, and per-kind
   registry dispatch.
2. **`output paths`** — checks each enabled output path with
   `Path::exists()`. A pre-existing file is reported as a FAIL with
   the same `OutputExists` payload `run` would surface.
3. **`init`** — loads the extended-XYZ init file, reports the
   particle count and box dimensions.
4. **`box/cutoff`** — for cell-list mode, verifies
   `min_perpendicular_width ≥ 3 · (cutoff_max + r_skin)`. For
   all-pairs mode, reported as `not applicable`.
5. **`topology`** — loads the topology file (when supplied) and
   reports the bond/angle/constraint-group counts. When the config
   omits `topology`, reported as `not supplied`.
6. **`gpu`** — only attempted with `--with-gpu`. Runs `init_device`
   (including the cuFFT smoke test for SPME configs), allocates the
   particle buffers, constructs every slot, and constructs the force
   field. Without `--with-gpu`, reported as `not checked`.

### On success

Prints the per-stage report on stdout (header `[dynamics lint] OK`)
and exits with code `0`:

```
[dynamics lint] OK
  config       /path/to/argon.in.toml
  output paths none pre-exist
  init         resolved, 10000 particles, box 8.0e-9 × 8.0e-9 × 1.0e-8 m
  box/cutoff   min perp width 8.00e-9 m ≥ required 3.32e-9 m
  topology     not supplied
  gpu          not checked (re-run with --with-gpu)
```

### On failure

Prints the per-stage report on stdout (header `[dynamics lint] FAIL`),
followed by a single `error: <message>` line on stderr that matches
what `run` would print for the same condition, and exits with code
`1`:

```
[dynamics lint] FAIL
  config       /path/to/argon.in.toml
  output paths none pre-exist
  init         resolved, 10000 particles, box 2.0e-9 × 5.0e-9 × 5.0e-9 m
  box/cutoff   FAIL — min perp width 2.00e-9 m along `a` < required 3.30e-9 m
  topology     skipped (earlier check failed)
  gpu          not checked (re-run with --with-gpu)
error: simulation box perpendicular width along lattice direction `a` is 2e-9, below the required 3.3e-9
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0`  | Every check passed. |
| `1`  | At least one check failed, or the CLI was invoked with bad arguments. |

## `dynamics analyze <analysis-path>`

Runs every analysis declared in `<analysis-path>` (a
`<root>.in.analysis` file) against the trajectory it points at, and
writes one CSV per analysis. See [Analysis](../guide/analysis.md) for
the input-file schema, the implicit pairing rule, the trajectory
selection knobs, and the built-in `rdf` kind.

- `<analysis-path>` is the path to a `<root>.in.analysis` file. The
  filename must end in `.in.analysis`. Relative paths are resolved
  against the current working directory.
- Implicit pairing: when the analysis file does not set `simulation`
  or `trajectory` explicitly, the runner pairs with the sibling
  `<root>.in.toml` (the same `<root>` as the analysis filename) and
  reads the trajectory from that config's resolved
  `output.trajectory_path`.
- v1 is CPU-only; no `--with-gpu` flag.
- Output files default to `<root>.out.<name>.csv` (where `<name>`
  comes from each `[[analyses]]` entry's `name` field). Each
  per-analysis `output_path` field overrides.
- Pre-existing output files cause a hard error (`OutputExists`) at
  pre-flight, before any frame is read.

### On success

Prints one line on stdout:

```
[dynamics] analyze complete: <K> analyses over <F> frames in <T> ms
```

where `<K>` is the number of analyses and `<F>` is the number of
frames consumed after `first_frame`, `last_frame`, and `stride` are
applied.

### Exit codes

| Code | Meaning |
|------|---------|
| `0`  | Every analysis ran to completion and every CSV flushed. |
| `1`  | Error before the trajectory pass: filename-convention violation, parse error, sibling-config load failure, trajectory open failure, output-path collision, pre-existing output. |
| `2`  | Error during the trajectory pass or output write. |

## What is not provided

Schema v1 deliberately keeps the CLI minimal. There are no
environment variables or alternative config sources — every parameter
affecting the trajectory lives in the TOML config so that two runs of
the same file produce identical bits.

The following do **not** exist:

- A `--seed` flag (set `simulation.seed` and any thermostat/barostat
  `seed` field in the config).
- A `--steps` or `--dt` flag (set `phase.n_steps` and `phase.dt` in
  the relevant `[[phase]]`).
- A `--output-dir` flag (set the paths under `[phase.output]`).
- A `--help` flag. The only help is this book and the usage line above.
