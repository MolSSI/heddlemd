# Features

This chapter is a capability map: the ensembles, force-field terms, and
methods HeddleMD supports today, and — at the end — what it does not yet
do. Every method below is selected and parameterised in the TOML config;
see the [Configuration Reference](guide/configuration.md) for the fields.

## Ensembles and integrators

Choose an ensemble by composing an integrator with an optional
thermostat and barostat in the phase's config.

- `velocity-verlet` — symplectic **NVE** time-stepping. Add a
  `[phase.thermostat]` for **NVT** and/or a `[phase.barostat]` for
  **NPT**.
  - `lossless = true` — an opt-in variant whose position and velocity
    updates are exactly invertible, so a run can be stepped backward to
    its exact starting state. Incompatible with constraint groups. See
    [Reproducibility](guide/reproducibility.md).
- `langevin-baoab` — stochastic **NVT** via the Leimkuhler-Matthews
  BAOAB splitting. Thermostats through the friction term; needs no
  separate `[phase.thermostat]`.
- `mtk-npt` — deterministic extended-system **NPT** (Martyna-Tobias-Klein,
  isotropic). Integrates its own Nosé-Hoover chains and cell degree of
  freedom, so it needs neither a separate thermostat nor barostat.

Initial velocities, when the starting-structure file omits them, are
drawn from a **Maxwell-Boltzmann distribution** at the configured
temperature, with centre-of-mass momentum removed and the ensemble
rescaled to hit the target temperature exactly.

## Thermostats

Compose any of these with `velocity-verlet` to sample the canonical
ensemble (or to equilibrate).

- `nose-hoover-chain` — deterministic canonical sampling
  (Martyna-Klein-Tuckerman 1992). Configurable chain length,
  Suzuki-Yoshida sub-stepping, and RESPA sub-cycling; reports a
  conserved-quantity column (`nhc_conserved`) you can watch for drift.
- `csvr` — stochastic canonical-sampling velocity rescaling
  (Bussi-Donadio-Parrinello 2007).
- `andersen` — stochastic per-particle Maxwell-Boltzmann resampling
  (Andersen 1980), with a configurable collision rate.
- `berendsen` — weak-coupling temperature control. **Equilibration
  only** — it relaxes to the target temperature quickly but does not
  sample the canonical ensemble.

## Barostats

Compose with `velocity-verlet` (and a thermostat) for NPT.

- `c-rescale` — stochastic isotropic cell-rescaling (Bernetti-Bussi
  2020). Samples the NPT distribution exactly.
- `monte-carlo` — Metropolis Monte-Carlo barostat: periodic trial volume
  moves that rigidly scale molecular centres of mass and accept/reject
  against the NPT weight. Needs no instantaneous virial, and because it
  scales whole molecules it composes cleanly with rigid-water
  constraints.
- `berendsen` — weak-coupling pressure control. **Equilibration only.**

## Energy minimization

- `steepest-descent` — adaptive-step steepest descent:
  each iteration takes a trial step down the force, accepting and
  growing the step on an energy decrease or rejecting and shrinking it
  on an increase. Convergence criteria (max force per atom, relative
  energy change, iteration cap) are all configurable. Declared as a
  `[[minimization]]` phase that interleaves freely with MD `[[phase]]`
  blocks; velocities and the box pass through untouched, so a subsequent
  MD phase starts from a relaxed configuration. Writes a per-iteration
  log (energy, max force, step size, accept/reject).
- **Constraint-aware.** When the topology declares rigid groups, every
  trial configuration is projected back onto the constraint manifold
  before its energy is evaluated.

## Non-bonded interactions

### Van der Waals

- **Lennard-Jones** — per type-pair `(σ, ε)`, with a switching function
  that ramps the interaction smoothly to zero over `[r_switch, cutoff]`
  (continuous in value and first derivative, so energy and forces have no
  discontinuity at the cutoff), or a hard cutoff when
  `r_switch = cutoff`.

### Electrostatics

- **Smooth particle-mesh Ewald (SPME)** — the full long-range treatment:
  a real-space screened-Coulomb term plus a reciprocal-space mesh
  contribution. Per-particle charges come from
  `[[particle_types]].charge`. You control the Ewald splitting parameter
  `α`, the real-space cutoff, the FFT grid, and the B-spline
  interpolation order (`4`–`8`). Requires the cell-list neighbor mode.

  A short determinism self-test runs at startup when SPME is configured,
  so a GPU whose FFT library would break reproducibility is caught
  before any dynamics run rather than after.

## Bonded interactions

Bonded terms are declared in an optional topology file and parameterised
by named type entries in the config.

- **Bonds** — harmonic (`U = ½ k (r − r₀)²`, the usual stiff-spring
  bond) and Morse (`U = D_e (1 − e^{−a(r−r_e)})²`). A system may mix
  both; each bond is routed to its potential by type.
- **Angles** — harmonic (`U = ½ k_θ (θ − θ₀)²`).
- **Dihedrals** — periodic torsions (`U = k_φ (1 + cos(n·φ − φ₀))`,
  multiplicity `n ∈ [1, 6]`). Express a multi-term torsion by naming the
  same quadruple once per Fourier term.
- **Exclusions** — 1-2 and 1-3 neighbours are excluded automatically
  from the non-bonded interaction, with explicit per-pair overrides
  available. Periodic dihedrals additionally introduce scaled **1-4 LJ
  and Coulomb** interactions with per-type scale factors.

## Rigid constraints

Constraint groups let you hold selected internal coordinates rigid — the
standard way to run rigid-water models and to take the longer timesteps
that come with removing the fastest bond vibrations.

- `settle` — analytic (non-iterative) constraint for a symmetric
  three-atom rigid water molecule (Miyamoto & Kollman 1992). The fast
  path for the common rigid three-site water models (such as SPC/E and
  TIP3P): it holds the three intramolecular distances rigid with a
  closed-form solve, no iteration, no tolerance.
- `shake` — iterative SHAKE (position) paired with RATTLE (velocity) for
  arbitrary rigid clusters up to 8 atoms / 12 constraint pairs per
  group, with per-pair target distances on the constraint type.

Both work with the standard `velocity-verlet` integrator and the
`steepest-descent` minimizer. The Langevin, MTK-NPT, and lossless
velocity-Verlet integrators do not currently accept constraint groups.

## Boundary conditions

- **Orthorhombic and triclinic** periodic boxes (the triclinic box takes
  the six lower-triangular lattice parameters `lx, ly, lz, xy, xz, yz`).
- **Minimum-image convention** for pair distances.
- **Wrapped positions in the trajectory**, with an optional per-particle
  integer image triple so you can reconstruct unwrapped coordinates
  exactly for diffusion and other unwrapped analyses.

## Finding neighbours

Two schemes enumerate the short-range non-bonded pairs:

- `cell-list` (default) — spatial cell list with a skin distance, rebuilt
  only when an atom has drifted far enough to need it. The `r_skin`
  margin trades rebuild frequency against pairs-per-step and is
  configurable. Buffers size themselves to the actual pair count, so
  there is no per-particle neighbour cap to tune.
- `all-pairs` — the O(N²) direct sum, with no neighbor list. Convenient
  for small systems and as a reference.

Before any dynamics run, the engine checks that the box is large enough
for the chosen cutoff (minimum perpendicular width ≥ 3 · (cutoff +
`r_skin`)), so a too-small box is reported up front.

## Input and output

- **Inputs**: a TOML config (`*.in.toml`), an extended-XYZ
  starting-structure file (`*.in.xyz`), and an optional topology file.
  Output filenames default from the config's name, so a directory
  listing cleanly separates inputs (`*.in.*`) from generated outputs
  (`*.out.*`).
- **Trajectory** (`*.out.<phase>.xyz`) — self-describing extended-XYZ
  frames, each a valid starting-structure file, so any frame can be
  lifted out to **restart** a run.
- **Log** (`*.out.<phase>.log`) — a CSV of step, time, kinetic energy,
  and temperature, plus any conserved-quantity columns the chosen
  thermostat or barostat contributes. Loads directly into pandas.
- **Timings** (`*.out.<phase>.timings`) — a per-stage wall-clock
  performance summary.
- **Overwrite protection** — the runner refuses to start if any output
  file already exists, so a run never silently clobbers previous
  results.
- **Units** — SI by default; set `units = "atomic"` to work in Hartree
  atomic units. The choice applies uniformly to the config, the
  starting structure, and every output file.

## Trajectory analysis

- `heddlemd analyze` post-processes a trajectory in place, driven by a
  small `.in.analysis` file. It is CPU-only, cheap enough to run on a
  login node, and its output is byte-identical across runs.
- The **radial distribution function** (`rdf`) between any two particle
  types ships built-in; see [Analysis](guide/analysis.md). Custom
  builds can register additional analyses alongside it.

## Reproducibility

- **Bit-wise reproducibility on the same GPU.** Two runs of the same
  config produce byte-identical trajectory and log files. This covers
  the trajectory, the log, initial-velocity generation, and any
  stochastic thermostat or barostat (each seeded explicitly in the
  config). The full scope and its limits — cross-hardware differences,
  the intentionally non-deterministic timings file — are in
  [Reproducibility](guide/reproducibility.md).

## Requirements

- **NVIDIA GPU**, with the CUDA Toolkit 11.8 or newer available at build
  time. The engine is GPU-only: there is no CPU or non-NVIDIA execution
  path.
- **Single precision (`f32`)** for positions, velocities, and forces
  throughout. This is well suited to short-range condensed-phase
  workloads and maximises throughput; a double-precision build path is
  reserved for the future.
- **Built from source.** See [Installation](getting-started/installation.md).

## Not yet supported

The following are outside the engine's current scope:

- Double-precision storage and force kernels (reserved for a future
  build option).
- Pair and bond potentials beyond those above (e.g. Buckingham, FENE,
  Coulomb-Wolf).
- Ryckaert-Bellemans and improper dihedrals (only the periodic torsion
  form exists today).
- Anisotropic or fully flexible-cell NPT.
- Bit-wise reproducibility *across different hardware* — the guarantee is
  same-GPU only (see [Reproducibility](guide/reproducibility.md)).
- Binary trajectory formats (NetCDF, HDF5) and compressed output.
- CPU or non-NVIDIA execution.
