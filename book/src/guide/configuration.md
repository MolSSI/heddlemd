# Configuration Reference

A simulation is specified by a TOML config file. The config pins every
parameter that affects the trajectory; everything else (positions,
velocities, box) lives in the [init file](init-files.md). Run a config
with:

```
dynamics run path/to/argon.in.toml
```

This chapter is the user-facing rendering of the field reference. The
canonical, exhaustive schema lives at `rqm/io/config-schema.md` in the
repository — consult it for the deserialiser/validator error catalogue
and any field this chapter elides.

## Config filename convention

The config-file path passed to `dynamics run` must end in `.in.toml`.
The loader rejects any other name (`InvalidConfigFilename`) before it
opens the file. The convention has two purposes:

- An `ls` of a simulation directory groups inputs (`*.in.*`) and
  outputs (`*.out.*`) into two visible blocks, so it is obvious which
  files were generated and can safely be deleted before a re-run.
- The loader derives every default output path from the **`<root>`**
  of the config filename without an extra config field: strip the
  `.toml` extension, then strip one trailing `.in`. The result is the
  config's root.

Examples:

| Config filename       | `<root>` | Default output paths                                                |
| --------------------- | -------- | ------------------------------------------------------------------- |
| `argon.in.toml`       | `argon`  | `argon.out.xyz`, `argon.out.log`, `argon.out.timings`               |
| `spc.in.toml`         | `spc`    | `spc.out.xyz`, `spc.out.log`, `spc.out.timings`                     |
| `run-01.in.toml`      | `run-01` | `run-01.out.xyz`, `run-01.out.log`, `run-01.out.timings`            |
| `foo.in.in.toml`      | `foo.in` | `foo.in.out.xyz`, `foo.in.out.log`, `foo.in.out.timings`            |

The suffix match is case-sensitive on the whole `.in.toml` string.
`argon.IN.toml` is rejected. Filenames whose `<root>` derivation would
yield the empty string (e.g. `.in.toml`) are likewise rejected.

The init-file path, the topology-file path, and any explicit
`output.*_path` field are not subject to the convention — they are
arbitrary user-supplied paths.

## Units

All quantities are SI throughout:

- length — metres
- mass — kilograms
- time — seconds
- energy — joules
- charge — coulombs
- temperature — kelvin

No alternative unit systems and no unit suffixes are supported in
schema v1.

## Path resolution

All file paths in the config (`init`, `topology`, `phase.<name>.output.*`)
are interpreted relative to the **directory containing the config file**,
not the current working directory. Absolute paths are honored as-is.
After resolution, every supplied path must be pairwise distinct.

## Phases

A simulation is composed of one or more **phases**. Phases come in
two kinds:

- **MD phases** declared as `[[phase]]` array elements. Each MD phase
  carries its own `n_steps`, `dt`, and slot blocks
  (`[phase.integrator]`, optional `[phase.thermostat]`, optional
  `[phase.barostat]`, optional `[phase.output]`).
- **Minimization phases** declared as `[[minimization]]` array
  elements. Each minimization phase carries an algorithm selector
  (`steepest-descent` in v1), algorithm parameters, convergence
  criteria, and an optional `[minimization.output]` block.

Particle state (positions, velocities, box) carries from one phase
into the next regardless of phase kind; per-phase slot state resets
at every boundary. Velocities and the simulation box are **unchanged**
by a minimization phase — only positions move.

The two arrays may be freely interleaved: phases execute in
**source-document order** across `[[phase]]` and `[[minimization]]`
combined. Typical pattern is one minimization phase followed by one
or more MD phases:

```toml
[[minimization]]
name = "min"

[minimization.algorithm]
kind = "steepest-descent"

[[phase]]
name = "equil"
n_steps = 5000
dt = 1.0e-15
...

[[phase]]
name = "prod"
n_steps = 10000
dt = 1.0e-15
...
```

Each phase produces its own output files derived from `<root>` (the
config file stem with one trailing `.in` stripped) and the phase's
`name`:

- MD phases write `<root>.out.<phase>.xyz`, `<root>.out.<phase>.log`,
  `<root>.out.<phase>.timings`.
- Minimization phases write `<root>.out.<phase>.minlog` and
  `<root>.out.<phase>.timings` (always), plus
  `<root>.out.<phase>.xyz` when `trajectory_every > 0`.

Per-phase paths can be overridden via the phase's output block
(`[phase.output]` for MD, `[minimization.output]` for minimization).

Phases must have distinct names **across the union of both arrays**.
Stochastic MD slots of the same kind across phases (e.g. two CSVR
thermostats) must use distinct seeds.

## Worked example

A complete single-phase config for 10,000 argon atoms in NVE:

Saved as `argon.in.toml`:

```toml
schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 1
temperature = 100.0 # K

[[phase]]
name = "run"
n_steps = 100
dt = 1.0e-15        # 1 fs

[phase.integrator]
kind = "velocity-verlet"
lossless = false

[phase.output]
trajectory_every = 10
include_velocities = true
log_every = 5

[[particle_types]]
name = "Ar"
mass = 6.6335e-26   # kg

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10    # m
epsilon = 1.65e-21  # J  (~120 K · k_B)
cutoff = 8.5e-10    # m  (2.5σ)
```

This is the bundled `examples/lj-10000-argon/argon.in.toml` file. The
[Configuration sections](#configuration-sections) below walk through
every field.

A multi-phase NPT example (equilibration then production) is shown in
the bundled `examples/spc-water-8192/water.in.toml`.

## Configuration sections

### Top level

| Field            | Type   | Required | Default | Notes |
|------------------|--------|----------|---------|-------|
| `schema_version` | u64    | yes      | —       | must be `1` |
| `init`           | string | yes      | —       | path to the init file |
| `topology`       | string | no       | none    | path to a `.topology` file (bonded forces, exclusions, constraints) |

When `topology` is omitted no bonded forces are evaluated and the
LJ/Coulomb kernels see no exclusion list.

### `[simulation]`

| Field         | Type | Required | Default | Notes |
|---------------|------|----------|---------|-------|
| `seed`        | u64  | yes      | —       | RNG seed for Maxwell-Boltzmann velocity generation. Required even when the init file supplies velocities. |
| `temperature` | f64  | yes      | —       | Initial-velocity temperature in K. Finite, `>= 0`. Used only when the init file omits velocities. The thermostat's bath temperature is a separate field. |

### `[[phase]]`

At least one `[[phase]]` array element is required. Each phase carries:

| Field         | Type | Required | Default | Notes |
|---------------|------|----------|---------|-------|
| `name`        | string | yes    | —       | Identifier used in output filenames and error messages. Must be unique across phases and use only ASCII letters, digits, `-`, or `_`. |
| `n_steps`     | u64  | yes      | —       | Number of integration steps in this phase. `0` is permitted (the runner writes the initial state and moves on). |
| `dt`          | f64  | yes      | —       | Integration timestep in seconds. Finite, strictly positive. |

A phase is then specified by its `[phase.integrator]` (required),
`[phase.thermostat]` (optional), `[phase.barostat]` (optional), and
`[phase.output]` (optional) sub-tables. The schema of each of those
slot blocks is described in the sections that follow.

### `[phase.integrator]`

A required `kind` field selects the integrator; all other fields are
kind-specific. Configs that pair an integrator that owns its own
thermostat (`langevin-baoab`, `mtk-npt`) with a `[phase.thermostat]`
section are rejected at load time.

#### `kind = "velocity-verlet"`

Symplectic NVE time-stepping. Compose with `[phase.thermostat]` for NVT.

| Field      | Type | Required | Default | Notes |
|------------|------|----------|---------|-------|
| `lossless` | bool | no       | `false` | When `true`, runs the compensated-summation variant that enables bit-exact time reversal. See [Reproducibility](reproducibility.md). Incompatible with topology-file `[constraints]`. |

#### `kind = "langevin-baoab"`

Stochastic NVT via the BAOAB splitting (Leimkuhler–Matthews). Owns its
own thermostat; incompatible with `[phase.thermostat]`.

| Field         | Type | Required | Notes |
|---------------|------|----------|-------|
| `friction`    | f64  | yes      | Damping `γ` in s⁻¹. Strictly positive. |
| `temperature` | f64  | yes      | Bath temperature in K. Independent of `simulation.temperature`. |
| `seed`        | u64  | yes      | RNG seed for the OU noise. Independent of `simulation.seed`. |

#### `kind = "mtk-npt"`

Deterministic extended-system NPT (Martyna-Tobias-Klein, isotropic).
Owns both its thermostat and its barostat; incompatible with both
`[phase.thermostat]` and `[phase.barostat]`.

| Field           | Type | Required | Default | Notes |
|-----------------|------|----------|---------|-------|
| `temperature`   | f64  | yes      | —       | Bath temperature in K. |
| `pressure`      | f64  | yes      | —       | Target pressure in Pa. Any sign. |
| `tau_t`         | f64  | yes      | —       | Thermostat coupling time (s). |
| `tau_p`         | f64  | yes      | —       | Barostat coupling time (s). |
| `chain_length`  | u32  | no       | `3`     | Length of both particle and cell thermostat chains. |
| `yoshida_order` | u32  | no       | `3`     | Suzuki-Yoshida sub-steps; one of `1, 3, 5, 7`. |
| `n_resp`        | u32  | no       | `1`     | Chain RESP sub-cycle count. |

### `[phase.thermostat]` (optional)

Omit for NVE composition with the integrator. A required `kind` field
selects the thermostat; per-kind fields are listed below. Configs that
combine `[phase.thermostat]` with an integrator that owns its own
thermostat are rejected.

#### `kind = "nose-hoover-chain"`

Deterministic NVT. Adds `nhc_conserved` to the log.

| Field           | Type | Required | Default | Notes |
|-----------------|------|----------|---------|-------|
| `temperature`   | f64  | yes      | —       | Bath temperature in K. |
| `tau`           | f64  | yes      | —       | Coupling time (s). 50–100 fs typical for water. |
| `chain_length`  | u32  | no       | `3`     | `M = 1` reduces to vanilla Nosé-Hoover. |
| `yoshida_order` | u32  | no       | `3`     | One of `1, 3, 5, 7`. |
| `n_resp`        | u32  | no       | `1`     | Chain RESP sub-cycle count. |

#### `kind = "csvr"`

Stochastic canonical sampling velocity rescaling (Bussi-Donadio-Parrinello).

| Field         | Type | Required | Notes |
|---------------|------|----------|-------|
| `temperature` | f64  | yes      | Bath temperature in K. |
| `tau`         | f64  | yes      | Coupling time (s). Larger `τ` → closer to NVE. |
| `seed`        | u64  | yes      | RNG seed. Independent of other slots. |

#### `kind = "andersen"`

Per-particle stochastic Maxwell-Boltzmann resampling.

| Field            | Type | Required | Notes |
|------------------|------|----------|-------|
| `temperature`    | f64  | yes      | Bath temperature in K. |
| `collision_rate` | f64  | yes      | `ν` in s⁻¹. `>= 0`; `0` degenerates to NVE. |
| `seed`           | u64  | yes      | RNG seed. |

#### `kind = "berendsen"`

Deterministic weak-coupling thermostat. **Equilibration only** — does
not sample the canonical ensemble.

| Field         | Type | Required | Notes |
|---------------|------|----------|-------|
| `temperature` | f64  | yes      | Bath temperature in K. |
| `tau`         | f64  | yes      | Coupling time (s). |

### `[phase.barostat]` (optional)

Omit for constant-volume composition. A required `kind` selects the
barostat. Configs that combine `[phase.barostat]` with an integrator
that owns its own barostat (currently `mtk-npt`) are rejected.

#### `kind = "berendsen"`

Deterministic isotropic weak-coupling barostat. **Equilibration only.**

| Field             | Type | Required | Notes |
|-------------------|------|----------|-------|
| `pressure`        | f64  | yes      | Target pressure (Pa). Any sign. |
| `tau`             | f64  | yes      | Coupling time (s). Should be longer than the thermostat's `τ`. |
| `compressibility` | f64  | yes      | Isothermal compressibility `β` (1/Pa). |

#### `kind = "c-rescale"`

Stochastic isotropic cell-rescaling barostat (Bernetti-Bussi). Samples
the canonical NPT distribution exactly.

| Field             | Type | Required | Notes |
|-------------------|------|----------|-------|
| `pressure`        | f64  | yes      | Target pressure (Pa). |
| `temperature`     | f64  | yes      | Used in the noise term. Keep consistent with the thermostat for canonical sampling. |
| `tau`             | f64  | yes      | Coupling time (s). |
| `compressibility` | f64  | yes      | Isothermal compressibility (1/Pa). |
| `seed`            | u64  | yes      | RNG seed. |

### `[[minimization]]`

A minimization phase advances positions along the negative gradient
of the potential energy. Each entry carries:

| Field   | Type   | Required | Default | Notes |
|---------|--------|----------|---------|-------|
| `name`  | string | yes      | —       | Identifier used in output filenames. Same character set and uniqueness rules as `[[phase]]`'s `name` (the uniqueness check spans **both** arrays). |

A minimization entry has a required `[minimization.algorithm]` block
and an optional `[minimization.output]` block. Unlike MD phases,
minimization rejects `n_steps`, `dt`, `[minimization.integrator]`,
`[minimization.thermostat]`, and `[minimization.barostat]` — those
concepts do not apply.

#### `[minimization.algorithm]`

A required `kind` field selects the algorithm; all other fields are
optional with documented defaults.

##### `kind = "steepest-descent"`

Adaptive-step steepest descent (GROMACS-style). Each iteration
proposes a trial position `x_trial = x + step · F / F_max`, where
`F_max = max_i ||F_i||`; if the trial energy is lower, the step is
accepted and `step` is multiplied by `step_increase` (capped at
`max_step`); otherwise positions are restored and `step` is
multiplied by `step_decrease`.

| Field              | Type | Required | Default   | Notes |
|--------------------|------|----------|-----------|-------|
| `initial_step`     | f64  | no       | `1.0e-12` | Initial scalar step in m. Finite, strictly positive. |
| `max_step`         | f64  | no       | `1.0e-10` | Upper bound on `step` in m. Must be `>= initial_step`. |
| `step_increase`    | f64  | no       | `1.2`     | Multiplier applied to `step` on accept. Must be `>= 1.0`. |
| `step_decrease`    | f64  | no       | `0.2`     | Multiplier applied to `step` on reject. Must be in `(0.0, 1.0)`. |
| `force_tolerance`  | f64  | no       | `1.0e-10` | Convergence threshold on `F_max` in N. `0.0` disables this criterion. |
| `energy_tolerance` | f64  | no       | `1.0e-7`  | Relative convergence threshold on `|ΔE| / max(|E_prev|, |E_curr|, ε)` between two consecutive accepted iterations. `0.0` disables. |
| `max_iterations`   | u64  | no       | `1000`    | Iteration cap. Reaching it without satisfying any physical criterion is a hard error and exits with code `2`. |

The phase terminates with **success** when either `F_max <= force_tolerance`
or the energy-tolerance criterion fires on an accepted iteration.

#### `[minimization.output]` (optional; all fields have defaults)

| Field              | Type   | Default                          | Notes |
|--------------------|--------|----------------------------------|-------|
| `minlog_path`      | string | `<root>.out.<phase>.minlog`      | Path to the per-iteration CSV. |
| `minlog_every`     | u64    | `1`                              | Iterations between `.minlog` rows. `0` disables the file. The final convergence iteration is always logged. |
| `trajectory_path`  | string | `<root>.out.<phase>.xyz`         | Path to the optional `.xyz` trajectory. |
| `trajectory_every` | u64    | `0`                              | Iterations between trajectory frames. `0` (the default) disables intermediate frames; only the step-0 and convergence frames are written when this is `> 0`. |
| `include_images`   | bool   | `true`                           | Include `image:I:3` columns in trajectory frames. |
| `timings_path`     | string | `<root>.out.<phase>.timings`     | Path to the per-stage performance summary. Always written. |

Velocities never appear in minimization trajectory frames; setting
`include_velocities` under `[minimization.output]` is rejected as an
unknown field. See [Output Files](output.md#minlog-file-outphaseminlog)
for the `.minlog` format.

### `[[particle_types]]` (array, ≥ 1 entry)

One entry per species. Names must be unique.

| Field    | Type   | Required | Default | Notes |
|----------|--------|----------|---------|-------|
| `name`   | string | yes      | —       | Identifier referenced from the init file and from `pair_interactions.between`. Case-sensitive. Non-empty. |
| `mass`   | f64    | yes      | —       | Particle mass in kg. Finite, strictly positive. |
| `charge` | f64    | no       | `0.0`   | Per-particle charge in C. Any sign. Required by the Coulomb / SPME slots. |

### `[[pair_interactions]]` (array)

One entry per **unordered pair** of declared types. For `N` types you
must supply exactly `N · (N + 1) / 2` entries (including same-type
self-pairs). `["A", "B"]` and `["B", "A"]` refer to the same pair.

Common fields:

| Field       | Type        | Required | Default        | Notes |
|-------------|-------------|----------|----------------|-------|
| `between`   | [string; 2] | yes      | —              | Unordered pair of type names. |
| `potential` | string      | yes      | —              | Currently only `"lennard-jones"`. |
| `cutoff`    | f64         | yes      | —              | Cutoff distance in m. |
| `r_switch`  | f64         | no       | `0.9 · cutoff` | Inner radius of the CHARMM-style C¹ switching function applied over `[r_switch, cutoff]`. `r_switch = cutoff` selects the hard-cutoff degenerate case. |

Fields for `potential = "lennard-jones"`:

| Field     | Type | Required | Notes |
|-----------|------|----------|-------|
| `sigma`   | f64  | yes      | LJ zero-crossing distance (m). |
| `epsilon` | f64  | yes      | LJ well depth (J). |

### `[coulomb]` (optional)

Activates the truncated short-range Coulomb pair-force slot. The
per-particle charges come from `[[particle_types]].charge`; this table
carries only the cutoff parameters. Mutually exclusive with `[spme]`.

| Field      | Type | Required | Default        | Notes |
|------------|------|----------|----------------|-------|
| `cutoff`   | f64  | yes      | —              | Cutoff distance (m). |
| `r_switch` | f64  | no       | `0.9 · cutoff` | Inner switching radius. |

### `[spme]` (optional)

Activates smooth particle-mesh Ewald. Mutually exclusive with
`[coulomb]`.

| Field          | Type     | Required | Default | Notes |
|----------------|----------|----------|---------|-------|
| `alpha`        | f64      | yes      | —       | Ewald splitting parameter (1/m). |
| `r_cut_real`   | f64      | yes      | —       | Real-space cutoff (m). |
| `grid`         | [u32; 3] | yes      | —       | FFT grid `[n_a, n_b, n_c]`. Each `n_d >= 2 · spline_order`. |
| `spline_order` | u32      | no       | `4`     | One of `4, 5, 6, 7, 8`. |

### `[[bond_types]]`, `[[angle_types]]`, `[[constraint_types]]`

Declare parameter sets referenced by name from the optional `.topology`
file. All three arrays are optional. The schema details (Morse bonds,
harmonic angles, SETTLE water constraints) live in
`rqm/io/config-schema.md` and the matching pages under `rqm/forces/`
and `rqm/integration/`.

### `[neighbor_list]` (optional)

Selects the algorithm short-range pair-force slots use to enumerate
non-bonded pairs. Defaults to the cell-list pipeline.

| Field           | Type   | Required | Default        | Notes |
|-----------------|--------|----------|----------------|-------|
| `mode`          | string | no       | `"cell-list"`  | `"cell-list"` or `"all-pairs"`. |
| `max_neighbors` | u64    | no       | `256`          | Cell-list only. Per-particle neighbor capacity. A rebuild that exceeds this halts the run. |
| `r_skin`        | f64    | no       | `0.3 · cutoff` | Cell-list only. Skin distance in m. |

For `mode = "all-pairs"`, `max_neighbors` and `r_skin` are rejected as
unknown fields. The simulation box's minimum perpendicular width must
satisfy `>= 3 · (cutoff_max + r_skin)`; the runner validates this once
the box is known.

### `[phase.output]` (optional; all fields have defaults)

| Field                | Type   | Default                       | Notes |
|----------------------|--------|-------------------------------|-------|
| `trajectory_path`    | string | `<root>.out.<phase>.xyz`      | Path to the trajectory file. Relative to the config's directory. |
| `trajectory_every`   | u64    | `100`                         | Steps between frames. `0` disables trajectory output. |
| `include_velocities` | bool   | `true`                        | Include `velo:R:3` columns. |
| `include_images`     | bool   | `true`                        | Include `image:I:3` columns. |
| `log_path`           | string | `<root>.out.<phase>.log`      | Path to the CSV log. |
| `log_every`          | u64    | `100`                         | Steps between log rows. `0` disables the log. |
| `timings_path`       | string | `<root>.out.<phase>.timings`  | Path to the per-stage performance summary. Always written; there is no `timings_every`. |

`<root>` is the config root derived from the filename per the
[Config filename convention](#config-filename-convention), and
`<phase>` is the phase's `name`. See [Output Files](output.md) for the
full format spec.

## Validation

The loader rejects configs that:

- live at a path whose final filename component does not end in
  `.in.toml`, or whose `<root>` derivation is empty
  (`InvalidConfigFilename`; see the
  [Config filename convention](#config-filename-convention));
- carry an unrecognised top-level or section field;
- choose an unknown `kind` for the integrator, thermostat, barostat, or
  any constraint type;
- omit a required field, or supply a non-finite or out-of-domain value
  (negative `dt`, empty type name, `r_switch > cutoff`, etc.);
- declare a `[[pair_interactions]]` entry referencing an undeclared
  type, omit a required pair, or duplicate a pair;
- resolve two supplied file paths to the same location;
- combine `[coulomb]` with `[spme]`;
- combine `[phase.thermostat]`/`[phase.barostat]` with an integrator
  that owns its own;
- declare zero phases (the union of `[[phase]]` and
  `[[minimization]]` must be non-empty), duplicate a phase name
  across either array, or share a seed between two stochastic slots
  of the same kind across phases;
- choose an unknown minimization `kind`, or supply out-of-domain
  values for the SD parameters (e.g. `step_decrease >= 1.0`,
  `max_step < initial_step`, `max_iterations = 0`);
- pair the constraint framework with an integrator that does not
  support constraints (Langevin BAOAB, MTK NPT, and lossless
  velocity-Verlet currently do not).

Errors are reported on stderr as `error: <message>` and exit with code
`1` (no integration steps run). The runner only proceeds when the
config validates cleanly.
