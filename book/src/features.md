# Features

A quick scan of what the engine supports. Each entry links the place in
this book or in `rqm/` where the feature is documented in full.

## Core guarantees

- **Bit-wise reproducibility on the same GPU.** Two runs of the same
  config produce byte-identical trajectory and log files. The
  guarantee is load-bearing and explicit: see
  [Reproducibility](guide/reproducibility.md).
- **Deterministic reduction ordering.** Pair forces are written to
  pre-indexed buffer slots; per-particle sums use a fixed-topology
  segmented reduction with no `atomicAdd` on floats.
- **Single-stream CUDA execution.** Every kernel launch goes through
  one `CudaStream`, so submission order is the execution order.
- **Lossless time-reversible integration (opt-in).** A compensated
  `(f32, f64)` mode for velocity-Verlet whose `x += v · dt` and
  `v += a · dt/2` updates are exactly invertible; runs in this mode
  can be stepped backward to the bit-exact starting state.
- **Deterministic Maxwell-Boltzmann velocity sampling.** ChaCha8 RNG,
  centre-of-mass momentum subtracted, rescaled so the realised
  flat-3N temperature equals the configured target.

## Hardware

- **NVIDIA GPU only.** No CPU fallback, no non-NVIDIA backend.
- **CUDA Toolkit 11.8 or newer.** Kernels are compiled to PTX at
  build time and embedded in the binary.
- **f32 storage and compute throughout.** An `f64` build path is
  reserved as a future compile-time feature flag.

## Integrators

- `velocity-verlet` — symplectic NVE core. Compose with
  `[phase.thermostat]` and/or `[phase.barostat]` for NVT / NpT.
  - `lossless = true` — opt-in compensated-summation variant; enables
    bit-exact time reversal. Incompatible with constraint groups.
- `langevin-baoab` — stochastic NVT via the Leimkuhler-Matthews BAOAB
  splitting. Owns its own thermostat.
- `mtk-npt` — deterministic extended-system NpT (Martyna-Tobias-Klein,
  isotropic). Owns both its thermostat (Nosé-Hoover chains on
  particles and cell) and its barostat (extended-system cell).

## Thermostats (compose with `velocity-verlet`)

- `nose-hoover-chain` — deterministic NVT (Martyna-Klein-Tuckerman
  1992); configurable chain length, Suzuki-Yoshida sub-stepping, RESP
  sub-cycling. Adds a `nhc_conserved` column to the log.
- `csvr` — stochastic canonical sampling velocity rescaling
  (Bussi-Donadio-Parrinello 2007).
- `andersen` — per-particle stochastic Maxwell-Boltzmann resampling
  (Andersen 1980); per-step collision probability
  `clamp(ν · dt, 0, 1)`.
- `berendsen` — deterministic weak-coupling. **Equilibration only** —
  does not sample the canonical ensemble.

## Barostats (compose with `velocity-verlet`)

- `c-rescale` — stochastic isotropic cell-rescaling barostat
  (Bernetti-Bussi 2020). Samples the canonical NpT distribution exactly.
- `berendsen` — deterministic isotropic weak-coupling barostat.
  **Equilibration only.**

## Energy minimization

- `steepest-descent` — adaptive-step steepest descent (GROMACS-style):
  per-iteration trial `x_new = x + step · F / F_max`, then accept and
  grow `step` on energy decrease or reject and shrink `step` on
  increase. Convergence criteria are configurable (max-force-per-atom,
  relative energy-change tolerance, max-iterations); non-convergence
  is a hard error (exit code `2`). Configured via a `[[minimization]]`
  phase, freely interleavable with `[[phase]]` blocks. Particle
  velocities and the simulation box pass through the phase
  untouched. Writes a per-iteration `.minlog` CSV with `iter, energy,
  max_force, step, accepted` columns.
- **Constraint-aware.** When the topology declares constraint groups,
  the minimizer projects every trial position back onto the constraint
  manifold before evaluating the trial energy (SETTLE rigid water in
  v1).

## Pair potentials (short-range, non-bonded)

- **Lennard-Jones** — per type-pair `(σ, ε)`, cutoff with CHARMM-style
  C¹ switching function over `[r_switch, cutoff]` (or hard cutoff when
  `r_switch = cutoff`).
- **Truncated Coulomb (`[coulomb]`)** — per-particle charges from
  `[[particle_types]]`; same C¹ switching function over
  `[r_switch, cutoff]`.

## Electrostatics

- **Smooth Particle-Mesh Ewald (`[spme]`)** — real-space `erfc`-screened
  pair force plus reciprocal-space spread / FFT / multiply / IFFT /
  gather pipeline. Configurable splitting parameter `α`, real-space
  cutoff, FFT grid, and B-spline order (`4, 5, 6, 7, 8`). Mutually
  exclusive with `[coulomb]`.
- **cuFFT determinism smoke test** at startup when SPME is configured,
  surfaces a `CuFftNonDeterministic` error before any integration step
  if the FFT library would break bit-exactness on the current GPU.

## Bonded interactions

- **Bonds: Morse** — `(D_e, a, r_e)` per `[[bond_types]]` entry.
- **Angles: Harmonic** — `(k_θ, θ_0)` per `[[angle_types]]` entry.
- **Per-bond / per-angle exclusions** auto-derived (1-2 and 1-3) from
  the topology file, with optional explicit overrides.
- **Long-range / 1-4 LJ + Coulomb scaling** in the four-column
  exclusion form.

## Rigid constraints

- `settle-water` — SETTLE algorithm for rigid three-site water
  (Miyamoto-Kollman 1992). Configurable `r_OH` and `r_HH`.
- Compatible with the lossy `velocity-verlet` integrator and with the
  `steepest-descent` minimizer (position-only projection in the
  latter). Langevin BAOAB, MTK NpT, and the lossless velocity-Verlet
  variant reject topologies that declare any constraint group.

## Neighbor lists

- `cell-list` (default) — spatial hashing with skin distance, periodic
  rebuild triggered by a per-step max-displacement check. Configurable
  `r_skin` and `max_neighbors`.
- `all-pairs` — O(N²) kernel, no neighbor list. Useful for small
  systems and as a reference.
- **Pre-flight box-vs-cutoff sanity check.** The runner verifies
  `min_perpendicular_width ≥ 3 · (cutoff_max + r_skin)` before any
  integration step.

## Boundary conditions

- **Orthorhombic periodic boxes.**
- **Triclinic periodic boxes** (lower-triangular lattice matrix; six
  free parameters `lx, ly, lz, xy, xz, yz`).
- **Minimum-image convention** for pair distance evaluation.
- **Wrapped positions in trajectory output**, plus an optional
  per-particle integer image triple `(image_a, image_b, image_c)` so
  consumers can reconstruct unwrapped trajectories exactly.

## I/O

- **Inputs**: TOML config (`*.in.toml`), extended-XYZ init file
  (`*.in.xyz`), optional `.in.topology` file. Config-filename
  convention enforced at load time; defaults for output paths derive
  from the config root (see
  [Configuration Reference](guide/configuration.md#config-filename-convention)).
- **Outputs** (one set per `[[phase]]`):
  - `*.out.<phase>.xyz` — extended-XYZ trajectory, self-describing
    per frame, re-loadable as an init file.
  - `*.out.<phase>.log` — CSV with
    `step,time,kinetic_energy,temperature`, plus any
    integrator-supplied extras (e.g. `nhc_conserved`).
  - `*.out.<phase>.timings` — fixed-width per-stage performance
    summary (kernel + host).
- **Pre-flight overwrite refusal.** The runner refuses to start when
  any output file (across all phases) already exists.
- **Restart from any trajectory frame.** Trajectory frames are valid
  init files; lift one out and point a fresh config at it.

## Configuration and ergonomics

- **TOML config**, SI units throughout, strict per-field validation.
- **Tagged-enum selection** for integrator / thermostat / barostat /
  constraint kinds; per-kind parameter validation via builder traits.
- **Open registries.** Custom integrator, thermostat, barostat,
  constraint, potential, and **analysis** builders can be registered
  alongside the built-ins via `Registries::register_*`.
- **Three CLI subcommands**: `dynamics run <config>` to execute,
  `dynamics lint <config> [--with-gpu]` to validate inputs without
  running, and `dynamics analyze <analysis-path>` to post-process a
  trajectory. See [CLI Reference](reference/cli.md). No environment
  variables — every parameter affecting the trajectory lives in the
  config.
- **Input linter** for HPC contexts. `dynamics lint` runs the
  setup-phase checks (TOML parse, filename convention, init-file
  load, topology load, pre-existing-output detection, box-vs-cutoff
  geometry) without touching the GPU; an optional `--with-gpu` flag
  extends the lint through `init_device`, slot construction, and
  force-field allocation. Catches input errors on a login node before
  a long submission queue runs the job. Dispatches on file extension:
  `.in.toml` runs the simulation lint, `.in.analysis` runs the
  analyze lint.
- **In-tree post-processing**: `dynamics analyze` reads a
  `<root>.in.analysis` file, walks the trajectory frame-by-frame
  with `first_frame`/`last_frame`/`stride` selection, and writes one
  CSV per declared analysis. CPU-only in v1; outputs are byte-
  identical across runs. Ships the radial distribution function
  (`rdf`) as the first built-in kind; see
  [Analysis](guide/analysis.md).

## Diagnostics

- **Always-on per-stage timings.** Kernel launches timed with CUDA
  events on the default stream; host stages timed with
  `std::time::Instant`. ~1 µs overhead per CUDA event.
- **Structured error reporting** via `thiserror`. Every error type has
  a `Display` rendering, a `Debug` rendering, and a walkable cause
  chain via `source()`.

## Out of scope (today)

The following are **not** part of the engine in its current form:

- f64 storage / f64 force kernels (reserved for a future feature flag).
- Buckingham, FENE, Coulomb-Wolf, or other potentials beyond those
  listed above.
- Anisotropic / flexible-cell NpT.
- Cross-hardware bit-wise reproducibility (CUDA permits FMA
  contraction differences between GPUs; only same-GPU runs match
  byte-for-byte).
- Binary trajectory formats (NetCDF, HDF5), gzip/xz compression.
- CPU or non-NVIDIA execution.
