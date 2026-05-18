# Command-Line Interface

The `dynamics` binary has one subcommand in schema v1.

## Usage

```
dynamics run <config-path>
```

Invoking `dynamics` with no arguments, an unknown subcommand, or `run`
without a path prints

```
usage: dynamics run <config-path>
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
7. Runs the timestep loop for `simulation.n_steps` iterations, writing
   trajectory frames and log rows at the configured cadences.
8. Writes the `.timings` file.

### On success

Prints one line on stdout:

```
[dynamics] complete: <N> steps in <T> ms (frames: <F>, log rows: <R>)
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

## What is not provided

Schema v1 deliberately keeps the CLI minimal. There are no flags, no
environment variables, and no alternative config sources — every
parameter affecting the trajectory lives in the TOML config so that two
runs of the same file produce identical bits.

The following do **not** exist:

- A `--seed` flag (set `simulation.seed` and any thermostat/barostat
  `seed` field in the config).
- A `--steps` or `--dt` flag (set `simulation.n_steps` and
  `simulation.dt`).
- A `--output-dir` flag (set the paths under `[output]`).
- A dry-run / validate-only flag (load the config in a Rust test
  harness if you need this).
- A `--help` flag. The only help is this book and the usage line above.
