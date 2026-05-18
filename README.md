# Dynamics

A GPU-accelerated molecular dynamics engine in Rust + CUDA, designed for
**bit-wise reproducibility**: identical inputs produce byte-identical
trajectory and log files across runs on the same GPU.

## What it does

- Lennard-Jones pair forces (O(N²) kernel) with the minimum-image convention
  for periodic boundary conditions.
- Velocity Verlet integration in either an ordinary `f32` mode (lossy) or a
  compensated `(f32, f64)` mode (lossless) that supports bit-exact time
  reversal.
- Single-stream CUDA execution and a deterministic segmented reduction so
  that floating-point sums are performed in the same order on every run.
- Extended-XYZ trajectory output, CSV diagnostic log (step, time, KE, T),
  and a per-stage performance summary measured with CUDA events plus
  host wall-clocks.

See [`docs/architecture.md`](docs/architecture.md) for the data flow,
reproducibility strategy, and per-kernel design. Every behaviour the engine
ships with is canonically described under [`rqm/`](rqm/); the source tree
references those entities by stable IDs (`rq-XXXXXXXX`).

## Prerequisites

- **An NVIDIA GPU** with a recent driver.
- **CUDA Toolkit 11.8 or newer** on `PATH` so `nvcc` can compile the
  device kernels at build time.
- **Rust** (the project uses Cargo edition 2024; install via
  [rustup](https://rustup.rs/)).

## Build

```
cargo build --release
```

The build script invokes `nvcc` for each `.cu` file under `kernels/`,
embeds the resulting PTX, and produces the `dynamics` binary at
`target/release/dynamics`.

## Run the example

A complete 10,000-atom Lennard-Jones argon example lives at
[`examples/lj-10000-argon/`](examples/lj-10000-argon/). It runs 100
integration timesteps in roughly a second on a recent NVIDIA GPU.

From the project root:

```
./target/release/dynamics run examples/lj-10000-argon/argon.in.toml
```

(Or `cargo run --release -- run examples/lj-10000-argon/argon.in.toml`.)

A run produces three files alongside the config:

- **`argon.out.xyz`** — 11 trajectory frames (steps 0, 10, …, 100) in
  extended-XYZ format. Each frame is self-describing (lattice vectors,
  column layout, simulation time). The trajectory frames can be re-loaded
  as an init file.
- **`argon.out.log`** — CSV with `step,time,kinetic_energy,temperature`;
  one header line plus 21 data rows.
- **`argon.out.timings`** — a fixed-width text table with one row per
  instrumented stage: per-kernel timings (CUDA events) and host stages
  (`config_load`, `init_load`, `gpu_init`, `host_to_device_upload`,
  `device_to_host_download`, `trajectory_write`, `log_write`,
  `velocity_generation`, `total_runtime`). Columns: `count`,
  `total_ms`, `mean_us`, `min_us`, `max_us`.

By convention, config filenames end in `.in.toml` and the loader
derives the default output paths from the filename root (`argon.in.toml`
→ `argon.out.{xyz,log,timings}`). The runner rejects a config path
that does not match the suffix. The example's
[`README.md`](examples/lj-10000-argon/README.md) describes the lattice
layout and how to regenerate `argon.in.xyz`.

## Writing your own simulation

A simulation is fully specified by two files:

- A **TOML config** that pins everything affecting the trajectory:
  RNG seed, `n_steps`, `dt`, target temperature, integrator mode
  (lossy or lossless), particle-type masses, and per-pair Lennard-Jones
  coefficients. SI units throughout (metres, kilograms, seconds, joules,
  kelvin). Output paths and cadences live in the optional `[output]`
  section; see [`rqm/io/config-schema.md`](rqm/io/config-schema.md) for
  the full field reference.
- An **extended-XYZ init file** carrying the particle count, simulation
  box (orthorhombic `Lattice="lx 0 0 0 ly 0 0 0 lz"`), per-particle
  type names, positions, and optionally velocities. Positions must lie
  inside the primary cell `[-L/2, L/2)` per axis. Velocities are
  optional; absent velocities are sampled from a Maxwell-Boltzmann
  distribution at the configured temperature using a deterministic
  ChaCha8 RNG seeded by the config seed, with the centre-of-mass drift
  removed. See [`rqm/io/init-state-file.md`](rqm/io/init-state-file.md).

The runner currently accepts one particle type per simulation; the
schema is forward-compatible with multi-type runs once the kernel
supports them.

## Reproducibility

The `<root>.out.xyz` trajectory and the `<root>.out.log` log are
byte-identical across two runs of the same config on the same GPU.
The `<root>.out.timings` file is intentionally **not** reproducible:
wall-clock measurements vary run-to-run and would corrupt the
comparison if mixed with the deterministic outputs.
Cross-hardware reproducibility is not a goal; CUDA permits FMA
contraction differences between GPUs.

## Project structure

```
src/                Rust host code: I/O, runner, GPU buffer wrappers
kernels/            CUDA C source for the device kernels (compiled to PTX)
docs/architecture.md  System design and data flow
rqm/                Canonical requirements, by feature
examples/           Ready-to-run input bundles
tests/              Integration tests (one per requirements file)
```

## Development workflow

This repository follows a **requirements-driven** workflow: every
feature has a canonical description under `rqm/` with Gherkin scenarios,
and every type, function, and test in `src/` and `tests/` carries the
stable `rq-XXXXXXXX` ID of the requirement it implements. The traceability
registry at `rqm/registry.json` is rebuilt by
`./.claude/skills/plan-feature/rqm.sh index`.

Two skills assist this loop:

- **`/plan-feature`** drafts or extends a requirements file, asks
  clarifying questions, and stamps stable IDs on every heading, API
  item, and scenario.
- **`/implement`** writes the code and tests for an existing
  requirements file. One test per Gherkin scenario, annotated with the
  scenario's `rq-` ID.

When iterating on a feature, edit the requirements file first, then ask
the assistant to update the implementation. This keeps `rqm/` as the
source of truth: if `src/` were deleted, the requirements files would
be enough to reproduce the engine.

## Safety notes for AI-assisted development

LLMs are susceptible to prompt injection and data poisoning. When using
this repo with an agentic assistant:

- Run the assistant inside a sandboxed container (the included Podman
  setup blocks the assistant from running outside one).
- Never expose private SSH keys, credentials, or write access to remote
  repositories.
- Review every generated change before pushing.
