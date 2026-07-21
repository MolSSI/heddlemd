# HeddleMD

A GPU-accelerated molecular dynamics engine for condensed-phase systems —
liquids, solutions, and solvated biomolecules — on a single NVIDIA GPU.
Written in Rust with CUDA compute kernels, and designed around one
distinguishing property: **bit-wise reproducibility**. Rerun the same
input on the same GPU and you get a byte-identical trajectory and log,
every time.

You describe a system with a force field, a starting configuration, and a
thermodynamic ensemble in a human-readable TOML file; the engine
propagates it in time and writes a trajectory you can analyse.

## What you can do

- **Ensembles.** NVE, NVT, and NPT dynamics. Compose a `velocity-verlet`
  integrator with a thermostat and/or barostat, or pick a self-contained
  integrator (`langevin-baoab` for NVT, `mtk-npt` for NPT).
  - **Thermostats:** Nosé–Hoover chains, CSVR (Bussi), Andersen, and
    Berendsen (equilibration only).
  - **Barostats:** stochastic cell-rescale (C-rescale), Monte-Carlo, and
    Berendsen (equilibration only).
- **Force fields.** Lennard-Jones van der Waals with a smooth switching
  function, smooth particle-mesh Ewald (SPME) electrostatics, and bonded
  terms — harmonic and Morse bonds, harmonic angles, and periodic
  torsions — with automatic 1-2/1-3 exclusions and scaled 1-4
  interactions.
- **Rigid constraints.** SETTLE (analytic, for three-site rigid water such
  as SPC/E and TIP3P) and SHAKE/RATTLE (iterative, arbitrary rigid
  clusters), so you can take the longer timesteps rigid water models
  assume.
- **Energy minimization.** Adaptive steepest descent, constraint-aware,
  as a phase that chains freely with MD phases (minimize → equilibrate →
  produce in a single run).
- **Analysis.** `heddlemd analyze` post-processes a trajectory in place;
  the radial distribution function (RDF) between any two particle types
  ships built-in.
- **Boundary conditions.** Orthorhombic and triclinic periodic boxes with
  the minimum-image convention; wrapped trajectory output with optional
  image flags for exact unwrapping.

Every quantity that affects the physics lives in the TOML config, in
**SI units by default** (set `units = "atomic"` for Hartree atomic units).
Inputs can be checked on a login node with `heddlemd lint` before a job
reaches the queue.

**Documentation.** The full user guide is published at
<https://molssi.github.io/heddlemd/> — install, first simulation,
configuration reference, init and output file formats, analysis, and the
precise scope of the reproducibility guarantee. Its source is the mdBook
under [`book/`](book/); render it locally with `mdbook serve book`. For the
internal design — the GPU compute pipeline
and the deterministic-reduction strategy — see
[`docs/architecture.md`](docs/architecture.md). Every behaviour the engine
ships with is canonically described under [`rqm/`](rqm/); the source tree
references those requirements by stable IDs (`rq-XXXXXXXX`).

## Prerequisites

- **An NVIDIA GPU** with a recent driver. The engine is GPU-only; there is
  no CPU or non-NVIDIA execution path.
- **CUDA Toolkit 11.8 or newer** on `PATH` so `nvcc` can compile the
  device kernels at build time (`nvcc --version` to check).
- **Rust** (Cargo edition 2024; install via [rustup](https://rustup.rs/)).

A `Containerfile` is included for Podman/Docker if you prefer a
known-good, pinned toolchain over installing CUDA on the host.

## Build

```
cargo build --release
```

The build script invokes `nvcc` for each `.cu` file under `kernels/`,
embeds the resulting PTX, and produces the `heddlemd` binary at
`target/release/heddlemd`.

## Run the example

A complete 10,000-atom Lennard-Jones argon system lives at
[`examples/lj-10000-argon/`](examples/lj-10000-argon/). It runs 100
NVE timesteps in roughly a second on a recent NVIDIA GPU. From the
project root:

```
./target/release/heddlemd run examples/lj-10000-argon/argon.in.toml
```

(Or `cargo run --release -- run examples/lj-10000-argon/argon.in.toml`.)
On success it prints one `[heddlemd] complete: ...` line and exits `0`.

Output filenames are derived from the config root and each phase's `name`,
so the config's `.in.` becomes `.out.<phase>.`. For this example (phase
`name = "run"`) a run produces three files alongside the config:

- **`argon.out.run.xyz`** — extended-XYZ trajectory frames. Each frame is
  self-describing (lattice vectors, column layout, step, simulation time)
  and is itself a valid init file, so any frame can be lifted out to
  restart a run.
- **`argon.out.run.log`** — CSV of `step,time,kinetic_energy,temperature`
  (plus any conserved-quantity columns the chosen thermostat or barostat
  contributes). Loads directly into pandas.
- **`argon.out.run.timings`** — a per-stage wall-clock performance summary.
  This file is intentionally **not** reproducible; the trajectory and log
  are.

The runner refuses to overwrite existing outputs, so delete or move these
three files before re-running. See [`examples/`](examples/) for more input
bundles (SPC/E water, ethane, chignolin, an RDF analysis).

## Writing your own simulation

A simulation is specified by two files (plus an optional topology file for
bonded or constrained systems):

- A **TOML config** (`*.in.toml`) pinning everything that affects the
  trajectory: RNG seed, particle-type masses and charges, force-field
  parameters, neighbor scheme, and one or more `[[phase]]` /
  `[[minimization]]` blocks — each with its own steps, timestep,
  integrator, and optional thermostat/barostat/output. See the
  [Configuration Reference](https://molssi.github.io/heddlemd/guide/configuration.html).
- An **extended-XYZ init file** (`*.in.xyz`) carrying the particle count,
  simulation box, per-particle type names, positions, and optionally
  velocities. Absent velocities are sampled from a Maxwell-Boltzmann
  distribution at the configured temperature using a deterministic RNG,
  with centre-of-mass drift removed. See the
  [Init Files guide](https://molssi.github.io/heddlemd/guide/init-files.html).

## Validating without running

`heddlemd lint <config>` runs every input-validation check the runner
would perform — TOML parse, init-file load, topology load, output-path
collisions, box-vs-cutoff geometry — without touching the GPU or writing
any files. Designed for HPC contexts where a long queue makes
trial-and-error iteration expensive: lint on a login node and fix the
report up front. Add `--with-gpu` to extend the lint through device
initialisation and force-field allocation when a GPU is available. See the
[CLI Reference](https://molssi.github.io/heddlemd/reference/cli.html) for the full specification.

## Reproducibility

The trajectory (`*.out.<phase>.xyz`) and log (`*.out.<phase>.log`) are
byte-identical across two runs of the same config on the same GPU. This
covers the trajectory, the log, initial-velocity generation, and any
stochastic thermostat or barostat (each seeded explicitly in the config).
The `*.out.<phase>.timings` file is intentionally **not** reproducible:
wall-clock measurements vary run-to-run and would corrupt the comparison
if mixed with the deterministic outputs. Cross-hardware reproducibility is
not a goal; CUDA permits FMA-contraction differences between GPUs. The
full scope and its limits are documented in the
[Reproducibility guide](https://molssi.github.io/heddlemd/guide/reproducibility.html).

## Precision

Positions, velocities, and forces use `f32` end to end — well suited to
short-range condensed-phase workloads and maximising GPU throughput. A
double-precision build path is reserved for the future. Independently, the
velocity-Verlet integrator has an opt-in `lossless` mode (compensated,
Kahan-style summation of the integrator bookkeeping) that makes a run
exactly invertible for bit-exact time reversal; this is not a
double-precision simulation and does not change the `f32` physics.

## Project structure

```
src/                  Rust host code: I/O, runner, GPU buffer wrappers
kernels/              CUDA C source for the device kernels (compiled to PTX)
book/                 User guide (mdBook) — start here
docs/architecture.md  System design and data flow
rqm/                  Canonical requirements, by feature
examples/             Ready-to-run input bundles
tests/                Integration tests (one per requirements file)
```
