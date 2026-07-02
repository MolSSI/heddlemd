# Writing a Simulation

A simulation is fully specified by two files:

- a **TOML config** that pins everything affecting the trajectory —
  timestep, step count, integrator, particle types, pair potentials,
  output cadences;
- an **extended-XYZ init file** that carries the simulation box, the
  particles, and their starting positions (and optionally velocities).

Run them with `heddlemd run <config>`. The runner resolves the init
path relative to the config's directory.

This chapter walks through writing both files from scratch for a small
custom system. For exhaustive field reference, see
[Configuration Reference](configuration.md) and
[Init Files](init-files.md).

## Worked example: 8 argon atoms on a 2×2×2 lattice

Goal: a 2 nm³ box containing 8 argon atoms on a simple-cubic lattice,
run for 500 fs at 100 K with the standard Lennard-Jones argon
parameters. Everything stays in SI units.

We will keep the two files side by side in a directory. The file-naming
[convention](configuration.md#config-filename-convention) requires the
config to end in `.in.toml`; we'll use the matching `.in.xyz` form for
the init file too so a future `ls` of the directory keeps inputs and
outputs visually grouped:

```
my-run/
├── argon.in.toml
└── argon.in.xyz
```

### Step 1: design choices to make before writing anything

| Decision                | Value here                         | Where it lives    |
|-------------------------|------------------------------------|-------------------|
| Box geometry            | 2 nm cube, orthorhombic            | `argon.in.xyz`    |
| Particle layout         | 2×2×2 lattice, 1.0 nm spacing      | `argon.in.xyz`    |
| Number of types         | 1 (Ar)                             | `argon.in.toml`   |
| Mass and charge per type| 6.6335e-26 kg, no charge           | `argon.in.toml`   |
| Pair potential          | Lennard-Jones (σ=3.4 Å, ε≈120 k_B) | `argon.in.toml`   |
| Cutoff                  | 8.5 Å (2.5σ)                       | `argon.in.toml`   |
| Initial velocities      | none — let the runner sample       | both              |
| Integrator              | velocity-Verlet, NVE               | `argon.in.toml`   |
| Timestep                | 1 fs                               | `argon.in.toml`   |
| Number of steps         | 500                                | `argon.in.toml`   |

### Step 2: write `argon.in.xyz`

A 2 nm³ box centred at the origin means each position lies in
`[-1.0e-9, 1.0e-9)` per axis. The four lattice corners at fractional
coords `(±0.25, ±0.25, ±0.25)` give a 1 nm spacing:

```
8
Lattice="2.0e-9 0 0 0 2.0e-9 0 0 0 2.0e-9" Properties=species:S:1:pos:R:3
Ar -5.0e-10 -5.0e-10 -5.0e-10
Ar  5.0e-10 -5.0e-10 -5.0e-10
Ar -5.0e-10  5.0e-10 -5.0e-10
Ar  5.0e-10  5.0e-10 -5.0e-10
Ar -5.0e-10 -5.0e-10  5.0e-10
Ar  5.0e-10 -5.0e-10  5.0e-10
Ar -5.0e-10  5.0e-10  5.0e-10
Ar  5.0e-10  5.0e-10  5.0e-10
```

Things to remember:

- `Lattice` is a 9-component row-major list. For an orthorhombic box,
  the three off-diagonal pairs are all `0`.
- `Properties` is a fixed enum — see [Init Files](init-files.md) for
  the four accepted forms. Here we omit `velo:R:3` so the runner will
  sample velocities for us.
- Positions are in metres and must satisfy `pos ∈ [-L/2, L/2)` per
  axis. A particle at exactly `+L/2` is **rejected** (upper bound is
  exclusive). The eight lattice points above sit safely inside the box.
- Particle IDs are implicit (row order). No ID column is supported.

### Step 3: write `argon.in.toml`

```toml
schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 1            # RNG seed for the initial Maxwell-Boltzmann sampling
temperature = 100.0 # K — used because argon.in.xyz has no `velo:R:3`

[[phase]]
name = "run"
n_steps = 500
dt = 1.0e-15        # 1 fs

[phase.integrator]
kind = "velocity-verlet"
lossless = false

[phase.output]
trajectory_every = 50
log_every = 10
include_velocities = true

[[particle_types]]
name = "Ar"
mass = 6.6335e-26   # kg

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10    # m
epsilon = 1.65e-21  # J  (≈ 120 K · k_B)
cutoff = 8.5e-10    # m  (2.5 σ)
```

Why these particular values:

- `seed = 1` is sufficient — any `u64` is fine, but the same value gives
  the same byte-identical trajectory on the same GPU.
- `temperature = 100.0` is *only* used because `argon.in.xyz` lacks
  velocities. If you supply them, the field is still required and
  validated but ignored at runtime.
- `cutoff = 8.5e-10` is well below `L/2 = 1.0e-9`, so the
  minimum-image convention is safe. For very small boxes like this,
  always sanity-check `cutoff + r_skin < L/2` along the *shortest* box
  vector. The default cell-list configuration adds a `0.3·cutoff`
  skin, which keeps you comfortably under the limit here.
- We let `phase.output.trajectory_path`, `phase.output.log_path`, and
  `phase.output.timings_path` default to `argon.out.run.xyz`,
  `argon.out.run.log`, `argon.out.run.timings` — derived from the
  config's root and the phase's `name` by the
  [filename convention](configuration.md#config-filename-convention).

### Step 4: run it

```
cd my-run
heddlemd run argon.in.toml
```

On success the runner prints one line per phase plus a final
aggregate:

```
[heddlemd] phase `run`: 500 steps in <T> ms (frames: 11, log rows: 51)
[heddlemd] complete: 500 steps in <T> ms
```

and writes `argon.out.run.xyz`, `argon.out.run.log`, and
`argon.out.run.timings` next to the config. Format details for each
file live in [Output Files](output.md).

### Step 5: re-run reproducibility check

Move the outputs aside, re-run, and `diff` — `argon.out.run.xyz` and
`argon.out.run.log` should match byte-for-byte.
`argon.out.run.timings` will differ; that file is intentionally
non-deterministic. See [Reproducibility](reproducibility.md).

## Common variations

### Supply explicit velocities

Replace `Properties=species:S:1:pos:R:3` with
`Properties=species:S:1:pos:R:3:velo:R:3` and append three velocity
columns (m/s) to every data row. `simulation.temperature` is still
required and validated, but the runner will not generate or rescale
velocities.

### Switch to NVT (thermostat composition)

Add a `[phase.thermostat]` section to the phase. For a deterministic
canonical ensemble:

```toml
[phase.thermostat]
kind = "nose-hoover-chain"
temperature = 100.0
tau = 1.0e-13       # 100 fs coupling time
```

For a stochastic alternative:

```toml
[phase.thermostat]
kind = "csvr"
temperature = 100.0
tau = 1.0e-13
seed = 7            # independent of simulation.seed
```

For an integrator that owns its own thermostat use
`kind = "langevin-baoab"` under `[phase.integrator]` instead, and
**omit** `[phase.thermostat]` — combining the two is rejected at load
time.

### Reach NPT (barostat composition)

Add a `[phase.barostat]` section, or switch the integrator to
`kind = "mtk-npt"` (which owns both thermostat and barostat). See
[Configuration Reference](configuration.md#phasebarostat-optional)
for the field reference.

### Minimize before sampling

Add a `[[minimization]]` phase before the `[[phase]]` block. The
minimizer relaxes positions along the negative energy gradient until
the maximum per-atom force or the relative energy change drops below
a tolerance; velocities and the box pass through untouched, so any
subsequent MD phase starts from a clean low-strain configuration.

```toml
[[minimization]]
name = "min"

[minimization.algorithm]
kind = "steepest-descent"
# Optional — all of these have sensible defaults:
#   initial_step = 1.0e-12     # m
#   max_step = 1.0e-10         # m
#   step_increase = 1.2        # multiplier on accept
#   step_decrease = 0.2        # multiplier on reject
#   force_tolerance = 1.0e-10  # N
#   energy_tolerance = 1.0e-7  # relative
#   max_iterations = 1000

[[phase]]
name = "run"
n_steps = 500
dt = 1.0e-15
...
```

Phases run in source-document order, so the minimization above
executes before the MD `[[phase]]`. The minimization writes
`<root>.out.min.minlog` (per-iteration energy, max-force, step size,
accept/reject flag); see [Output Files](output.md#minlog-file-outphaseminlog).

Non-convergence at `max_iterations` is a hard error (exit code `2`);
subsequent phases do not run. If you are tuning a new system,
inspect the `.minlog` to see whether energy is still decreasing at
the cap and either raise `max_iterations` or relax one of the
tolerances.

### Equilibrate, then sample (multi-phase)

Append a second `[[phase]]` to the config. Particle state (positions,
velocities, box) carries across the boundary; the integrator,
thermostat, and barostat slots reset. Common pattern: a short NVT
equilibration that suppresses trajectory output, followed by a longer
NPT production phase that writes a frame per picosecond. See
`examples/spc-water-8192/water.in.toml` for a fully worked version.

### Bonded forces, exclusions, rigid constraints

Add a `topology` field at the top level pointing to a `.topology` file,
then declare the matching parameter arrays:

- `[[bond_types]]` — harmonic (`potential = "harmonic"`, `k`/`r0`) or
  Morse (`potential = "morse"`) two-atom bonds; a system may mix both.
- `[[angle_types]]` — harmonic three-atom angles (`k_theta`/`theta_0`).
- `[[dihedral_types]]` — periodic torsions (`potential = "periodic"`,
  `k_phi`/`n`/`phi_0`); repeat a `[dihedrals]` row per Fourier term for
  a multi-term torsion.
- `[[constraint_types]]` — rigid groups. Use `kind = "settle"`
  (`d_OH`/`d_HH`) for rigid three-atom water (the fast analytic path),
  or `kind = "shake"` (`atoms` + a `constraints` list) for a general
  rigid cluster. Constraints require the plain `velocity-verlet`
  integrator (not `lossless`, Langevin BAOAB, or MTK NPT).

See the [Configuration Reference](configuration.md#bond_types-angle_types-dihedral_types-constraint_types)
for the full field tables. Worked bundles:

- `examples/spc-water-256/` — flexible SPC water (harmonic + Morse
  bonds, harmonic angles).
- `examples/ethane-216/` — liquid ethane exercising the periodic
  dihedral force.
- `examples/chignolin/` — a solvated all-atom protein combining bonds,
  angles, dihedrals, and SETTLE-constrained rigid water.

### Restart from a previous frame

Trajectory frames are themselves valid init files. Extract a single
frame into its own file, point a fresh config's `init` field at it,
and run. To keep the velocity field exactly, write the original
trajectory with `include_velocities = true` and ensure the new
config's `simulation.temperature` is set to *some* value (it will be
ignored because the frame supplies velocities).

## Pre-flight checklist

Before launching a long run:

- [ ] Every `[[particle_types]]` name appears in the init file's
      `species` column, and vice versa.
- [ ] You supplied a `[[pair_interactions]]` entry for **every**
      unordered pair, including same-type self-pairs. For `N` types
      that is `N · (N+1) / 2` entries.
- [ ] All positions lie in `[-L/2, L/2)` per axis (the upper bound is
      exclusive).
- [ ] `cutoff + r_skin` is at most `1/3` of the minimum perpendicular
      width of the box (the runner enforces this — easier to fix in
      your editor than to debug at startup).
- [ ] None of the resolved output files already exist; the runner
      refuses to overwrite.
