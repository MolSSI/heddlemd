# Feature: TOML Simulation Config Schema <!-- rq-6432ab1f -->

A simulation is specified by a TOML configuration file. The config pins every
parameter required to predict the trajectory bit-exactly. It carries no
positions or velocities; those live in a separate extended-XYZ initial-state
file referenced from the config.

The runner consumes the config via `dynamics run <path>` (see
`simulation-runner.md`). Trajectory and log outputs are described in
`trajectory-output.md` and `log-output.md` respectively.

## Schema <!-- rq-1c7a9cfd -->

The top-level table carries one mandatory field, `schema_version`. The schema
described here is version `1`. Loading a config whose `schema_version` is any
other value is an error.

Sections:

| Section | Required | Purpose |
| ------- | -------- | ------- |
| top-level `schema_version` | yes | format version |
| top-level `init` | yes | path to initial-state file |
| top-level `topology` | no | path to .topology file |
| `[simulation]` | yes | timestep, step count, RNG seed, initial-velocity temperature |
| `[integrator]` | yes | integrator slot + per-kind parameters |
| `[thermostat]` | no | thermostat slot + per-kind parameters |
| `[barostat]` | no | barostat slot + per-kind parameters |
| `[[particle_types]]` | yes (>= 1) | per-type properties |
| `[[pair_interactions]]` | yes (covers every pair) | per-pair potential + parameters |
| `[[bond_types]]` | no | per-bond-type parameters |
| `[[angle_types]]` | no | per-angle-type parameters |
| `[[constraint_types]]` | no | per-constraint-type parameters |
| `[neighbor_list]` | no | non-bonded pair-evaluation algorithm |
| `[output]` | no | trajectory & log paths and cadences |

### Example <!-- rq-ecc664ff -->

```toml
schema_version = 1
init = "argon.xyz"

[simulation]
seed = 12345
n_steps = 10000
dt = 1.0e-15        # s
temperature = 300.0 # K (used only if init file lacks a `velo` column)

[integrator]
kind = "velocity-verlet"
lossless = false

# Optional. Omit to run NVE. Cannot be combined with an integrator
# that owns its own thermostat (e.g. langevin-baoab).
#
# [thermostat]
# kind = "csvr"
# temperature = 300.0
# tau = 1.0e-13
# seed = 7

# Optional. Omit to run constant-volume.
#
# [barostat]
# kind = "berendsen"
# pressure = 1.0e5
# tau = 1.0e-12
# compressibility = 4.5e-10

[[particle_types]]
name = "Ar"
mass = 6.6335e-26   # kg

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10    # m
epsilon = 1.65e-21  # J
cutoff = 1.0e-9     # m
r_switch = 9.0e-10  # m  (defaults to 0.9 * cutoff when omitted)

# Optional: path to a .topology file declaring bonds, angles, and
# explicit non-bonded exclusions. When omitted, no bonded forces are
# computed and the LJ / Coulomb kernels see no exclusions.
topology = "argon.topology"

[[bond_types]]
name = "ArAr"
potential = "morse"
de = 1.65e-21       # J  (well depth)
a = 1.9e10          # 1/m (width)
re = 3.40e-10       # m  (equilibrium distance)

[[angle_types]]
name = "HOH"
potential = "harmonic"
k_theta = 5.27e-19  # J/rad²  (75.9 kcal/mol/rad², flexible SPC)
theta_0 = 1.911     # rad (~109.47°)

# Constraint types are referenced from the `.topology` file's
# [constraints] section. Each entry's `kind` selects the algorithm
# that processes any group declared with this type.
[[constraint_types]]
name = "SPCE"
kind = "settle-water"
r_oh = 1.0e-10        # m (O-H constraint distance)
r_hh = 1.633e-10      # m (H-H constraint distance)

[neighbor_list]
mode = "cell-list"
max_neighbors = 256
r_skin = 1.0e-10    # m  (defaults to 0.3 * cutoff when omitted)

[output]
trajectory_path = "argon-traj.xyz"
trajectory_every = 100
include_velocities = true
include_images = true
log_path = "argon.log"
log_every = 100
timings_path = "argon.timings"
```

### Units <!-- rq-ed997636 -->

All physical quantities are SI: lengths in metres, mass in kilograms, time in
seconds, energy in joules, temperature in kelvin. No alternative unit systems
or unit suffixes are supported in schema v1.

### Field reference <!-- rq-e367855a -->

#### Top level <!-- rq-4c42a952 -->

- `schema_version: u64` — must equal `1`. See *Schema version handling* below.
- `init: String` — path to the extended-XYZ initial-state file. Resolved
  relative to the config file's directory; absolute paths are honored as-is.
- `topology: String` — optional path to a `.topology` file (see
  `forces/topology.md`). Resolved relative to the config file's
  directory; absolute paths are honored as-is. When omitted, no
  bonded forces are computed and the LJ and Coulomb kernels see an
  empty exclusion list. When supplied, the file is loaded after the
  init file (so atom-index bounds checking has access to the
  particle count).

#### `[simulation]` <!-- rq-a84e1c76 -->

- `seed: u64` — RNG seed used for Maxwell-Boltzmann velocity generation.
  Required even when the init file supplies explicit velocities. No default.
- `n_steps: u64` — number of integration steps to execute. `0` is permitted
  (the runner writes the initial state and exits).
- `dt: f64` — integration timestep in seconds. Must be finite and strictly
  positive.
- `temperature: f64` — initial-velocity temperature in kelvin. Required.
  Used to initialise velocities at the Maxwell-Boltzmann distribution
  when the init file's `Properties` lacks a `velo:R:3` field; ignored
  (but still required and validated) when the init file supplies
  velocities. Must be finite and `>= 0.0`. The thermostat's bath
  temperature is a separate field under `[thermostat]`.

#### `[integrator]` <!-- rq-27f9fae8 -->

The integrator section is a tagged variant. A required `kind` field
selects one of the pluggable integrator slots (see
`integration/framework.md` for the slot interface). Every other field
in this table is kind-specific: extra fields not recognised by the
chosen `kind` are rejected, and missing required fields are rejected.

- `kind: String` — required. One of:
  - `"velocity-verlet"` — symplectic NVE time-stepping core. Does not
    own a thermostat; compose with `[thermostat]` for NVT. See
    `integration/velocity-verlet.md`.
  - `"langevin-baoab"` — stochastic NVT via the Leimkuhler-Matthews
    BAOAB splitting. Owns its own thermostat (the OU step);
    incompatible with `[thermostat]`. See
    `integration/langevin-baoab.md`.
  - `"mtk-npt"` — deterministic extended-system NPT via the
    Martyna-Tobias-Klein integrator (isotropic). Owns both its
    thermostat (Nosé-Hoover chains on particles and cell) and its
    barostat (extended-system cell); incompatible with both
    `[thermostat]` and `[barostat]`. See `integration/mtk-npt.md`.

Fields accepted for `kind = "velocity-verlet"`:

- `lossless: bool` — selects the lossless compensated-summation variant.
  Optional; defaults to `false`. When `true`, the runner allocates
  `LosslessBuffers` and launches the `*_lossless` kernels. Composing
  `lossless = true` with a non-empty `[constraints]` topology section
  is rejected at config load (see
  `integration/constraint-framework.md`); the lossy variant
  (`lossless = false`) is the only integrator in the default registry
  that supports constraints.

Fields accepted for `kind = "langevin-baoab"`:

- `friction: f64` — damping coefficient `γ` in inverse seconds. Required.
  Finite and strictly positive; `0.0` is rejected.
- `temperature: f64` — bath temperature in kelvin. Required. Finite and
  strictly positive. Independent of `simulation.temperature`.
- `seed: u64` — counter-based RNG seed. Required, independent of
  `simulation.seed`.

Fields accepted for `kind = "mtk-npt"`:

- `temperature: f64` — bath temperature `T` in kelvin. Required.
  Finite and strictly positive. Independent of
  `simulation.temperature`.
- `pressure: f64` — target pressure `P_ext` in pascals (Pa).
  Required. Finite. May be any sign or zero.
- `tau_t: f64` — thermostat coupling time in seconds. Required.
  Finite and strictly positive. Controls both the particle-chain
  and cell-chain masses.
- `tau_p: f64` — barostat coupling time in seconds. Required.
  Finite and strictly positive. Controls the cell mass `W`.
- `chain_length: u32` — length `M` of both the particle and cell
  thermostat chains. Optional; defaults to `3`. Must be `≥ 1`.
- `yoshida_order: u32` — Suzuki-Yoshida sub-step count per chain
  half-step (shared by both chains). Optional; defaults to `3`.
  Accepted values: `1`, `3`, `5`, `7`.
- `n_resp: u32` — chain RESP sub-cycle count (shared by both
  chains). Optional; defaults to `1`. Must be `≥ 1`.

#### `[thermostat]` (optional) <!-- rq-1c2d8eba -->

The thermostat section is an optional tagged variant. When omitted,
the runner runs no thermostat (NVE composition with the integrator,
or whatever fused thermostat the integrator owns). When present, a
required `kind` field selects one of the pluggable thermostat slots
(see `integration/framework.md`). Every other field in this table is
kind-specific: extra fields not recognised by the chosen `kind` are
rejected, and missing required fields are rejected.

Configuring `[thermostat]` alongside an integrator that owns its own
thermostat (currently only `langevin-baoab`) is rejected at
config-load time with
`ConfigError::IncompatibleThermostat { integrator: <kind name> }`.

- `kind: String` — required when the section is present. One of:
  - `"nose-hoover-chain"` — deterministic NVT via the Nosé-Hoover
    chain (Martyna-Klein-Tuckerman, 1992). See
    `integration/nose-hoover-chain.md`.
  - `"csvr"` — stochastic NVT via canonical sampling velocity
    rescaling (Bussi-Donadio-Parrinello, 2007). See
    `integration/csvr.md`.
  - `"andersen"` — stochastic NVT via per-particle Maxwell-Boltzmann
    resampling (Andersen, 1980). See `integration/andersen.md`.
  - `"berendsen"` — deterministic weak-coupling thermostat (Berendsen
    et al., 1984). Suitable for **equilibration only**; does not
    sample the canonical ensemble. See `integration/berendsen.md`
    for the full caveat and parameter description.

Fields accepted for `kind = "nose-hoover-chain"`:

- `temperature: f64` — bath temperature in kelvin. Required. Finite
  and strictly positive. Independent of `simulation.temperature`.
- `tau: f64` — thermostat coupling time in seconds. Required. Finite
  and strictly positive. Typical values for liquid water are 50–100
  fs.
- `chain_length: u32` — number of chain elements `M`. Optional;
  defaults to `3`. Must be `≥ 1`. `M = 1` reduces to vanilla
  Nosé-Hoover.
- `yoshida_order: u32` — Suzuki-Yoshida sub-step count per chain
  half-step. Optional; defaults to `3`. Accepted values: `1`, `3`,
  `5`, `7`.
- `n_resp: u32` — chain RESP sub-cycle count. Optional; defaults to
  `1`. Must be `≥ 1`.

Fields accepted for `kind = "csvr"`:

- `temperature: f64` — bath temperature in kelvin. Required. Finite
  and strictly positive. Independent of `simulation.temperature`.
- `tau: f64` — thermostat coupling time in seconds. Required. Finite
  and strictly positive. Typical values for liquid water are 100 fs
  to 1 ps; larger `τ` leaves the dynamics closer to NVE.
- `seed: u64` — counter-based RNG seed for the chi-squared and
  standard-normal draws consumed by the rescale formula. Required,
  independent of `simulation.seed` and any other slot's seed.

Fields accepted for `kind = "andersen"`:

- `temperature: f64` — bath temperature in kelvin. Required. Finite
  and strictly positive. Independent of `simulation.temperature`.
- `collision_rate: f64` — per-particle stochastic collision frequency
  `ν` in inverse seconds. Required. Finite and `≥ 0` (`0`
  degenerates to NVE — no resampling — and is permitted as a
  diagnostic mode). Typical values for liquid water are
  `10¹¹–10¹² s⁻¹`. The per-step collision probability `p` is
  computed as `clamp(collision_rate · dt, 0.0, 1.0)`.
- `seed: u64` — counter-based RNG seed for the Bernoulli decisions
  and Maxwell-Boltzmann draws consumed by the resample kernel.
  Required, independent of `simulation.seed` and any other slot's
  seed.

Fields accepted for `kind = "berendsen"`:

- `temperature: f64` — bath temperature in kelvin. Required. Finite
  and strictly positive. Independent of `simulation.temperature`.
- `tau: f64` — thermostat coupling time in seconds. Required. Finite
  and strictly positive. Typical values for liquid water are 100 fs
  to 1 ps; larger `τ` leaves the dynamics closer to NVE.

#### `[barostat]` (optional) <!-- rq-d28e9105 -->

The barostat section is an optional tagged variant. When omitted, the
runner runs no barostat (constant-volume composition). When present,
a required `kind` field selects one of the pluggable barostat slots
(see `integration/framework.md`). Every other field in this table is
kind-specific: extra fields not recognised by the chosen `kind` are
rejected, and missing required fields are rejected.

- `kind: String` — required when the section is present. One of:
  - `"berendsen"` — deterministic isotropic weak-coupling barostat
    (Berendsen et al., 1984). Suitable for **equilibration only**;
    does not sample the isobaric ensemble. See
    `integration/berendsen-barostat.md` for the full caveat and
    parameter description.
  - `"c-rescale"` — stochastic isotropic cell-rescaling barostat
    (Bernetti-Bussi, 2020). Samples the canonical NPT distribution
    exactly. See `integration/c-rescale-barostat.md`.

Fields accepted for `kind = "berendsen"`:

- `pressure: f64` — target pressure `P_target` in pascals (Pa).
  Required. Finite. May be any sign or zero.
- `tau: f64` — pressure-coupling time constant in seconds. Required.
  Finite and strictly positive. Typical values for liquid water are
  1–5 ps; should be longer than the thermostat's `τ` to keep the
  two relaxation processes decoupled.
- `compressibility: f64` — isothermal compressibility `β` in 1/Pa
  (= m³/J). Required. Finite and strictly positive. Typical values
  are around `4.5e-10 1/Pa` for water; an inaccurate value changes
  the effective relaxation rate but not the long-time mean pressure.

Fields accepted for `kind = "c-rescale"`:

- `pressure: f64` — target pressure `P_target` in pascals (Pa).
  Required. Finite. May be any sign or zero.
- `temperature: f64` — target temperature `T` in kelvin used in the
  noise term. Required. Finite and strictly positive. Independent
  of `simulation.temperature` and of any `[thermostat].temperature`;
  the framework performs no cross-slot validation. For canonical
  NPT sampling the user must keep this value consistent with the
  thermostat (or, for `langevin-baoab`, with the integrator's bath
  temperature).
- `tau: f64` — pressure-coupling time constant in seconds. Required.
  Finite and strictly positive. Typical values for liquid water are
  1–5 ps.
- `compressibility: f64` — isothermal compressibility `β` in 1/Pa
  (= m³/J). Required. Finite and strictly positive. An inaccurate
  value changes the effective relaxation rate but not the long-time
  NPT distribution.
- `seed: u64` — counter-based RNG seed for the per-step normal
  draw. Required, independent of `simulation.seed` and any other
  slot's seed. Two runs with identical configs on the same GPU
  produce byte-identical trajectories.

#### `[[particle_types]]` (array of tables) <!-- rq-78487f38 -->

One entry per particle species. At least one entry required.

- `name: String` — unique identifier, used in the init file and in
  `pair_interactions.between`. Case-sensitive. Empty strings are rejected.
- `mass: f64` — particle mass in kilograms. Must be finite and strictly
  positive.
- `charge: f64` — optional. Particle charge in coulombs. Must be finite
  when supplied; any sign is accepted. Defaults to `0.0` when omitted.
  The charge applies to every particle of this type uniformly. The runner
  uploads the per-type charges into a per-particle `charges` buffer at
  init time (see `particle-state.md`).

Names must be unique within the array.

#### `[[pair_interactions]]` (array of tables) <!-- rq-9244aae4 -->

One entry per unordered pair of declared types. The collection contains
exactly one entry for every unordered pair, including same-type self pairs.
For `N` declared types the array contains exactly `N * (N + 1) / 2` entries.

Each entry is a tagged variant: a required `potential` field selects the
pair potential, and every other field is either a common field shared by
all potentials or specific to the chosen `potential`.

Common fields:

- `between: [String; 2]` — unordered pair of declared type names. Order is
  not significant: `["A", "B"]` and `["B", "A"]` refer to the same pair.
- `potential: String` — selects the pair potential. The only supported
  value in `[[pair_interactions]]` is `"lennard-jones"`. Electrostatic
  pair interactions are configured globally through the top-level
  `[coulomb]` table (see below) rather than per type pair, since the
  Coulomb pair magnitude is constructed from per-particle charges and
  carries no per-pair parameters. Future values
  (`"buckingham"`, ...) for `[[pair_interactions]]` are reserved.
- `cutoff: f64` — pair distance in metres beyond which the force is treated
  as zero. Finite, strictly positive.
- `r_switch: f64` — optional. Inner radius of the CHARMM-style C¹
  switching function applied over `[r_switch, cutoff]` (see
  `forces/lj-pair-force.md`). Finite, strictly positive, and
  `r_switch <= cutoff`. Defaults to `0.9 * cutoff` when omitted.
  Setting `r_switch = cutoff` selects the hard-cutoff degenerate case
  in which no smoothing is applied.

Fields accepted for `potential = "lennard-jones"` (see
`forces/lj-pair-force.md`):

- `sigma: f64` — LJ zero-crossing distance in metres. Finite, strictly
  positive.
- `epsilon: f64` — LJ well depth in joules. Finite, strictly positive.

Same-type pairs are required even when only one type is declared:
`between = ["Ar", "Ar"]` must appear. Unknown fields for the chosen
`potential` are rejected.

#### `[[bond_types]]` (optional array of tables) <!-- rq-e4420955 -->

Declares the parameter sets for bonded potentials referenced by name from
the `.topology` file. The array is optional and may be empty. When
supplied, every bond type's `name` field appears as the third column of
one or more rows in the `.topology` file's `[bonds]` section. Bond
types whose `name` is never used in the file are permitted
(declared-but-unused).

Common fields:

- `name: String` — unique identifier within the `[[bond_types]]` array.
  Empty strings are rejected. Case-sensitive.
- `potential: String` — selects the bonded potential. In schema v1 the
  only supported value is `"morse"`. Future values (`"harmonic"`,
  `"fene"`, ...) are reserved.

Fields accepted for `potential = "morse"` (see `forces/morse-bonded.md`):

- `de: f64` — Morse well depth in joules. Required. Finite, strictly
  positive.
- `a: f64` — Morse width parameter in inverse metres. Required. Finite,
  strictly positive.
- `re: f64` — Morse equilibrium distance in metres. Required. Finite,
  strictly positive.

Names must be unique within the array. Unknown fields for the chosen
`potential` are rejected.

#### `[[angle_types]]` (optional array of tables) <!-- rq-f2946c4a -->

Declares the parameter sets for angle potentials referenced by name from
the `.topology` file. The array is optional and may be empty. When
supplied, every angle type's `name` field appears as the fourth column
of one or more rows in the `.topology` file's `[angles]` section. Angle
types whose `name` is never used in the file are permitted
(declared-but-unused).

Common fields:

- `name: String` — unique identifier within the `[[angle_types]]`
  array. Empty strings are rejected. Case-sensitive.
- `potential: String` — selects the angle potential. The only
  supported value is `"harmonic"`. Future values
  (`"cosine-harmonic"`, `"urey-bradley"`, ...) are reserved.

Fields accepted for `potential = "harmonic"` (see
`forces/harmonic-angle.md`):

- `k_theta: f64` — angle force constant in joules per radian². Required.
  Finite, strictly positive.
- `theta_0: f64` — equilibrium angle in radians. Required. Finite, in
  `[0, π]`.

Names must be unique within the array. Unknown fields for the chosen
`potential` are rejected.

#### `[[constraint_types]]` (optional array of tables) <!-- rq-7e9cb164 -->

Declares the parameter sets for rigid constraint groups referenced by
name from the `.topology` file's `[constraints]` section. The array is
optional and may be empty. When supplied, every constraint type's
`name` field appears as the final column of one or more rows in the
`.topology` file's `[constraints]` section. Constraint types whose
`name` is never used in the file are permitted (declared-but-unused).

The presence of at least one constraint group in the topology file
triggers construction of the `Constraint` slot (see
`integration/constraint-framework.md`) and is incompatible with any
integrator whose `IntegratorKind::supports_constraints()` returns
`false`. Cross-validation handles this incompatibility and returns
`ConfigError::IncompatibleConstraint { integrator: <kind name> }`.

Common fields:

- `name: String` — unique identifier within the `[[constraint_types]]`
  array. Empty strings are rejected. Case-sensitive.
- `kind: String` — selects the algorithm that processes any group
  declared with this type. The only supported value is
  `"settle-water"`. Future values (`"m-shake"`, `"lincs"`, ...) are
  reserved.

Fields accepted for `kind = "settle-water"` (see
`integration/settle.md`):

- `r_oh: f64` — O–H constraint distance in metres. Required. Finite,
  strictly positive.
- `r_hh: f64` — H–H constraint distance in metres. Required. Finite,
  strictly positive. Must satisfy `r_hh < 2 · r_oh`; otherwise the
  loader returns `ConfigError::SettleGeometryInfeasible { name,
  r_oh, r_hh }`.

Names must be unique within the array. Unknown fields for the chosen
`kind` are rejected.

#### `[coulomb]` (optional table) <!-- rq-28d519b5 -->

Activates the truncated Coulomb pair-force slot (see
`forces/coulomb-pair-force.md`). The slot is present iff this table is
present. The kernel computes pair contributions from the per-particle
charges carried by `ParticleBuffers` (sourced from each particle type's
`charge` field in `[[particle_types]]`); the table carries only the
real-space cutoff parameters that apply uniformly to every pair.

- `cutoff: f64` — pair distance in metres beyond which the Coulomb
  force is treated as zero. Required when the table is present. Finite,
  strictly positive.
- `r_switch: f64` — optional. Inner radius of the CHARMM-style C¹
  switching function applied over `[r_switch, cutoff]` (see
  `forces/coulomb-pair-force.md`). Finite, strictly positive, and
  `r_switch <= cutoff`. Defaults to `0.9 * cutoff` when omitted.
  Setting `r_switch = cutoff` selects the hard-cutoff degenerate case
  in which no smoothing is applied.

The `[coulomb]` and `[spme]` tables are mutually exclusive: a config
declaring both is rejected with `ConfigError::ConflictingElectrostatics`.

Cross-validation alongside the other pair potentials feeds into the
neighbor list's box-compatibility check: the shared neighbor list's
search radius is `max(pair_interactions.cutoff_max, coulomb.cutoff,
spme.r_cut_real) + r_skin`, and the simulation box's minimum
perpendicular width must be at least `3 *` that value.

#### `[spme]` (optional table) <!-- rq-08131b48 -->

Activates smooth particle-mesh Ewald (see `forces/spme.md`). The two
SPME slots — the real-space `erfc`-screened pair-force slot and the
reciprocal-space spread / FFT / multiply / IFFT / gather pipeline —
are present iff this table is present. Per-particle charges come from
each particle type's `charge` field in `[[particle_types]]`; the table
carries the SPME parameters that apply uniformly to every pair and
every grid cell.

- `alpha: f64` — Ewald splitting parameter in inverse metres.
  Required. Finite, strictly positive. Larger values shift work into
  reciprocal space (shorter real-space cutoff feasible, finer grid
  needed). A common starting point for typical accuracy targets is
  `alpha ≈ 3.5 / r_cut_real`.
- `r_cut_real: f64` — real-space cutoff in metres beyond which the
  screened Coulomb force is treated as zero. Required. Finite,
  strictly positive.
- `grid: [u32; 3]` — FFT grid dimensions in the lattice-direction order
  `[n_a, n_b, n_c]`. Required. Each component must satisfy
  `n_d >= 2 · spline_order`. A typical starting point is one grid
  point per `~1 Å` of perpendicular width.
- `spline_order: u32` — B-spline interpolation order. Optional;
  defaults to `4` when omitted. Accepted values are `4`, `5`, `6`,
  `7`, `8`.

The `[spme]` and `[coulomb]` tables are mutually exclusive (see
above).

#### `[neighbor_list]` (optional table) <!-- rq-adddaf1a -->

Selects the algorithm used by the Lennard-Jones slot to enumerate
non-bonded pairs. The table is optional; when omitted the runner uses
the cell-list defaults documented below. See `forces/neighbor-list.md`
for the algorithm.

- `mode: String` — `"cell-list"` (default) or `"all-pairs"`. The
  `cell-list` value selects the cell-list / neighbor-list pipeline; the
  `all-pairs` value forces the O(N²) kernel.

Fields accepted for `mode = "cell-list"`:

- `max_neighbors: u64` — optional; defaults to `256`. Strictly
  positive. Upper bound on the number of neighbours retained per
  particle; a rebuild that finds more than this many neighbours for
  some atom halts the simulation via
  `RunnerError::ForceField(NeighborList(NeighborListOverflow))`. The
  default suits typical liquid-density short-range simulations
  (≤ ~200 neighbours per atom at cubic-LJ density and 2.5σ cutoff);
  dense or long-cutoff systems must raise it.
- `r_skin: f64` — skin distance in metres. Optional; defaults to
  `0.3 * cutoff` where `cutoff` is the maximum cutoff among
  `[[pair_interactions]]` entries. Strictly positive and finite.

Fields accepted for `mode = "all-pairs"`: none. `max_neighbors` and
`r_skin` are rejected as unknown fields in this mode.

Cross-validation:

- `r_skin > 0` and finite.
- `max_neighbors > 0`.
- The simulation box's minimum perpendicular width satisfies
  `min_perpendicular_width >= 3 * (cutoff_max + r_skin)` where
  `cutoff_max` is the largest cutoff among `[[pair_interactions]]`
  and the `[coulomb]` table's `cutoff` (when present). The box is
  read from the init file, so this check is performed by the runner,
  not by `load_config`. See `simulation-runner.md` for the
  runner-side validation and the corresponding `RunnerError` variant.

#### `[output]` (optional table; all fields have defaults) <!-- rq-6340fae2 -->

- `trajectory_path: String` — output trajectory path. Default:
  `<config-stem>-traj.xyz` in the same directory as the config file
  (e.g. `sim.toml` → `sim-traj.xyz`). Resolved relative to the config
  file's directory; absolute paths are honored as-is.
- `trajectory_every: u64` — write one trajectory frame every this many
  integration steps. Default `100`. `0` disables trajectory output entirely
  (not even the step-0 frame is written). TOML parses `u64` fields, so
  negative integers fail at TOML parse time.
- `include_velocities: bool` — include `velo:R:3` columns in every
  trajectory frame. Default `true`.
- `include_images: bool` — include `image:I:3` columns in every
  trajectory frame. Default `true`. When `true`, each frame's data rows
  carry three integer columns after the position (and velocity, if
  present) columns; consumers reconstruct unwrapped positions as
  `pos + images_x · a + images_y · b + images_z · c`, which reduces to
  `pos + image · (lx, ly, lz)` for an orthorhombic box. When `false`,
  image columns are omitted from the file; positions in the trajectory
  are still wrapped into the primary image.
- `log_path: String` — output log path. Default: `<config-stem>.log` in the
  same directory as the config file. Resolved like `trajectory_path`.
- `log_every: u64` — write one log row every this many integration steps.
  Default `100`. `0` disables the log entirely.
- `timings_path: String` — output path for the per-stage performance summary
  file. Default: `<config-stem>.timings` in the same directory as the config
  file (e.g. `sim.toml` → `sim.timings`). Resolved like `trajectory_path`.
  See `performance-analysis.md` for the file format. There is no
  `timings_every` field; the file is written once at end of run.

### Path resolution and overwrite policy <!-- rq-6d99f9c8 -->

- All file paths (`init`, `topology`, `output.trajectory_path`,
  `output.log_path`, `output.timings_path`) are interpreted relative
  to the **config file's containing directory** when not absolute.
  The loader resolves them before returning. The `topology` field is
  optional; when absent the resolved topology path is `None` and no
  topology file is loaded.
- After resolution, every supplied path must be pairwise distinct
  from every other supplied path. When `topology` is supplied, that
  path is included in the distinctness check.
- The loader does not check whether the resolved output files already
  exist; that check lives in the runner (`simulation-runner.md`) so that
  configs can be loaded for validation without filesystem side effects.

### Schema version handling <!-- rq-fc58e2c5 -->

- `schema_version` must appear as a top-level field. A missing field
  produces `MissingField { field: "schema_version" }`.
- The only accepted value is `1`. Any other value produces
  `UnsupportedSchemaVersion { actual, supported: 1 }`. The check runs
  before any other validation so that future formats fail loudly before
  any other field is read.

### Cross-validation <!-- rq-bd228ef7 -->

The checks below run inside `Config::validate(&self)`, after the
typed deserialiser populates `Config`. `load_config` invokes
`Config::validate` automatically; callers that build a `Config` in
memory can invoke it directly to obtain the same guarantees.

1. Every name in `[[pair_interactions]]`'s `between` refers to a declared
   `[[particle_types]]`. Unknown names produce `UnknownTypeInPair { name,
   pair_index }` where `pair_index` is the zero-based index in the
   `pair_interactions` array.
2. Every unordered pair of declared types appears in
   `[[pair_interactions]]` exactly once. A missing pair produces
   `MissingPairInteraction { types: (String, String) }`. A duplicate
   produces `DuplicatePairInteraction { types: (String, String) }`. The
   reported tuple is normalised so the lexicographically smaller name
   comes first.
3. After path resolution, every supplied path is pairwise distinct from
   every other supplied path (`PathCollision { kind_a, kind_b, path }`).
   The set of paths under check is `init`, `output.trajectory_path`,
   `output.log_path`, `output.timings_path`, and (when supplied) `topology`.
4. If `[thermostat]` is present and `config.integrator.owns_thermostat()`
   returns `true` (`langevin-baoab` and `mtk-npt`), validation returns
   `IncompatibleThermostat { integrator: <integrator kind name> }`.
5. If `[barostat]` is present and `config.integrator.owns_barostat()`
   returns `true` (currently only `mtk-npt`), validation returns
   `IncompatibleBarostat { integrator: <integrator kind name> }`.
6. If the topology file's `[constraints]` section is non-empty (which
   the loader detects after `load_topology_file` has run) and
   `config.integrator.supports_constraints()` returns `false`,
   validation returns
   `IncompatibleConstraint { integrator: <integrator kind name> }`. In
   the default registry this rejects every combination of constraints
   with `langevin-baoab`, `mtk-npt`, or `velocity-verlet { lossless =
   true }`.
7. Every `[[constraint_types]]` entry's algorithm-specific geometry is
   feasible. For `kind = "settle-water"`, the loader requires `r_hh <
   2 · r_oh` and returns `SettleGeometryInfeasible { name, r_oh,
   r_hh }` otherwise.
8. Every constraint type name referenced from the topology file's
   `[constraints]` section appears in `[[constraint_types]]`.
   Unknown names surface through `load_topology_file` as
   `TopologyFileError::UnknownConstraintType { .. }`.

## Feature API <!-- rq-110285ae -->

### Types <!-- rq-b719c42c -->

- `Config` — parsed configuration. All fields are `pub`; field names match <!-- rq-2a6a51c8 -->
  TOML keys directly (snake_case).

  Fields:
  - `schema_version: u64`
  - `init: PathBuf` — resolved against the config file's directory.
  - `topology: Option<PathBuf>` — `Some(_)` when the optional top-level
    `topology` field is present; resolved against the config file's
    directory.
  - `simulation: SimulationConfig`
  - `integrator: IntegratorKind`
  - `thermostat: Option<ThermostatKind>` — `Some` when the
    `[thermostat]` table is present in the config, `None` otherwise.
  - `barostat: Option<BarostatKind>` — `Some` when the `[barostat]`
    table is present in the config, `None` otherwise.
  - `particle_types: Vec<ParticleTypeConfig>`
  - `pair_interactions: Vec<PairInteractionConfig>`
  - `bond_types: Vec<BondTypeConfig>` — empty when the `[[bond_types]]`
    array is absent.
  - `angle_types: Vec<AngleTypeConfig>` — empty when the
    `[[angle_types]]` array is absent.
  - `constraint_types: Vec<ConstraintTypeConfig>` — empty when the
    `[[constraint_types]]` array is absent.
  - `coulomb: Option<CoulombConfig>` — `Some` when the `[coulomb]` table
    is present in the config, `None` otherwise. Mutually exclusive with
    `spme`.
  - `spme: Option<SpmeConfig>` — `Some` when the `[spme]` table is
    present in the config, `None` otherwise. Mutually exclusive with
    `coulomb`.
  - `neighbor_list: NeighborListConfig` — defaults to
    `NeighborListConfig::CellList { max_neighbors: 256, r_skin: 0.3 *
    max_cutoff }` when the `[neighbor_list]` table is omitted from the
    config, where `max_cutoff` is the largest cutoff across
    `[[pair_interactions]]`, the `[coulomb]` table, and the `[spme]`
    table's `r_cut_real` (whichever are present).
  - `output: OutputConfig`
  - `config_path: PathBuf` — the absolute path of the source config file,
    retained for error messages and default output-path derivation.

- `SimulationConfig` <!-- rq-53055a5b -->
  - `seed: u64`
  - `n_steps: u64`
  - `dt: f64`
  - `temperature: f64`

- `IntegratorKind` — tagged enum carrying the chosen integrator slot <!-- rq-661bf664 -->
  and its parameters. Variants:
  - `VelocityVerlet { lossless: bool }` — selected by
    `kind = "velocity-verlet"`. Does not own its own thermostat or
    its own barostat.
  - `LangevinBaoab { friction: f64, temperature: f64, seed: u64 }` —
    selected by `kind = "langevin-baoab"`. Owns its own thermostat
    (the OU step); does not own its own barostat.
  - `MtkNpt { temperature: f64, pressure: f64, tau_t: f64, tau_p: f64,
    chain_length: u32, yoshida_order: u32, n_resp: u32 }` — selected
    by `kind = "mtk-npt"`. Owns both its own thermostat (the
    Nosé-Hoover chains on particles and cell) and its own barostat
    (the extended-system cell). Optional fields are populated from
    the corresponding TOML defaults (`chain_length = 3`,
    `yoshida_order = 3`, `n_resp = 1`) when omitted.

  Variant-bearing parameters reflect the per-kind fields listed under
  the `[integrator]` section above.

  Methods:

  - `IntegratorKind::owns_thermostat(&self) -> bool` — returns `true`
    for variants that bundle their own thermostat (`LangevinBaoab`
    and `MtkNpt`); `false` otherwise. Consulted by cross-validation
    rule 4.
  - `IntegratorKind::owns_barostat(&self) -> bool` — returns `true`
    for variants that bundle their own barostat (currently only
    `MtkNpt`); `false` otherwise. Consulted by cross-validation
    rule 5.

- `ThermostatKind` — tagged enum carrying the chosen thermostat slot <!-- rq-3fdb7e01 -->
  and its parameters. Used in `Config::thermostat: Option<ThermostatKind>`.
  Variants:
  - `NoseHooverChain { temperature: f64, tau: f64, chain_length: u32,
    yoshida_order: u32, n_resp: u32 }` — selected by
    `kind = "nose-hoover-chain"`. Optional fields are populated from
    the corresponding TOML defaults (`chain_length = 3`,
    `yoshida_order = 3`, `n_resp = 1`) when omitted.
  - `Csvr { temperature: f64, tau: f64, seed: u64 }` — selected by
    `kind = "csvr"`. All three fields are required.
  - `Andersen { temperature: f64, collision_rate: f64, seed: u64 }`
    — selected by `kind = "andersen"`. All three fields are
    required. `collision_rate` is the per-particle stochastic
    collision frequency in inverse seconds and may be zero.
  - `Berendsen { temperature: f64, tau: f64 }` — selected by
    `kind = "berendsen"`. Both fields are required. Deterministic;
    no RNG seed.

  Variant-bearing parameters reflect the per-kind fields listed under
  the `[thermostat]` section above.

- `BarostatKind` — tagged enum carrying the chosen barostat slot and <!-- rq-aa6ce5c0 -->
  its parameters. Used in `Config::barostat: Option<BarostatKind>`.
  Variants:
  - `Berendsen { pressure: f64, tau: f64, compressibility: f64 }` —
    selected by `kind = "berendsen"`. All three fields are required.
    Deterministic; no RNG seed.
  - `CRescale { pressure: f64, temperature: f64, tau: f64, compressibility: f64, seed: u64 }`
    — selected by `kind = "c-rescale"`. All five fields are
    required. Stochastic; uses the per-step RNG seed.

  Variant-bearing parameters reflect the per-kind fields listed under
  the `[barostat]` section above.

- `ParticleTypeConfig` <!-- rq-a5ccc1de -->
  - `name: String`
  - `mass: f64`
  - `charge: f64` — defaults to `0.0` when the TOML field is omitted.

- `PairInteractionConfig` <!-- rq-f001eaf8 -->
  - `between: (String, String)` — stored normalised so the lexicographically
    smaller string comes first, regardless of source order.
  - `cutoff: f64` — pair distance in metres beyond which the force is
    treated as zero.
  - `r_switch: f64` — inner switching radius. Populated from the
    user-supplied `r_switch` when present, otherwise from the default
    `0.9 * cutoff`. Always satisfies `0 < r_switch <= cutoff`.
  - `potential: PairPotentialParams` — tagged enum carrying the chosen
    pair potential's functional-form parameters.

- `PairPotentialParams` — tagged enum carrying the functional-form <!-- rq-70442e07 -->
  parameters of a pair potential. Variants:
  - `LennardJones { sigma: f64, epsilon: f64 }` — selected by
    `potential = "lennard-jones"`. `sigma` is the LJ zero-crossing
    distance in metres; `epsilon` is the LJ well depth in joules.

- `BondTypeConfig` — tagged enum carrying the chosen bonded-potential <!-- rq-2f230ccb -->
  parameters. Variants:
  - `Morse { name: String, de: f64, a: f64, re: f64 }` — selected by
    `potential = "morse"`.

  The `name` field is the lookup key referenced from the `.topology`
  file's `[bonds]` section.

- `AngleTypeConfig` — tagged enum carrying the chosen angle-potential <!-- rq-a47beb76 -->
  parameters. Variants:
  - `Harmonic { name: String, k_theta: f64, theta_0: f64 }` — selected
    by `potential = "harmonic"`.

  The `name` field is the lookup key referenced from the `.topology`
  file's `[angles]` section.

- `ConstraintTypeConfig` — tagged enum carrying the chosen <!-- rq-ac8fc96a -->
  constraint-algorithm parameters. Variants:
  - `SettleWater { name: String, r_oh: f64, r_hh: f64 }` — selected by
    `kind = "settle-water"`.

  The `name` field is the lookup key referenced from the `.topology`
  file's `[constraints]` section. Each variant additionally exposes a
  `expected_atom_count(&self) -> usize` method used by the topology
  parser to validate row column counts (`SettleWater` returns `3`).

- `CoulombConfig` <!-- rq-793a7cbb -->
  - `cutoff: f64` — real-space cutoff in metres.
  - `r_switch: f64` — populated from the optional `r_switch` field with
    the documented default `0.9 * cutoff`.

- `SpmeConfig` <!-- rq-a03de3d5 -->
  - `alpha: f64` — Ewald splitting parameter (1/m).
  - `r_cut_real: f64` — real-space cutoff in metres.
  - `grid: [u32; 3]` — FFT grid dimensions `[n_a, n_b, n_c]`.
  - `spline_order: u32` — populated from the optional `spline_order`
    field with the documented default `4`.

- `NeighborListConfig` — tagged enum selecting the algorithm used by <!-- rq-a8320030 -->
  every short-range pair-force slot (Lennard-Jones, truncated Coulomb,
  SPME real-space) to enumerate non-bonded pairs. Variants:
  - `AllPairs` — selected by `mode = "all-pairs"`. Carries no
    parameters; pair-force slots use the O(N²) kernel.
  - `CellList { max_neighbors: u32, r_skin: f64 }` — selected by
    `mode = "cell-list"` (and used as the default when the
    `[neighbor_list]` table is omitted). `max_neighbors` is the
    per-particle neighbor capacity; `r_skin` is the skin distance in
    metres.

  See `forces/neighbor-list.md` for the runtime semantics of each
  variant.

- `OutputConfig` <!-- rq-1254cd3a -->
  - `trajectory_path: PathBuf` — resolved.
  - `trajectory_every: u64`
  - `include_velocities: bool`
  - `include_images: bool`
  - `log_path: PathBuf` — resolved.
  - `log_every: u64`
  - `timings_path: PathBuf` — resolved.

- `PathRole` — `enum { Init, Trajectory, Log, Timings, Topology }`. Used in `PathCollision`. <!-- rq-f0084057 -->

- `ConfigError` — error type returned by `load_config` and by <!-- rq-3108381e -->
  `Config::validate`. Variants:
  - `Io(String)` — failed to read the config file (with the OS error
    message). Only produced by `load_config`.
  - `Parse { path: String, message: String }` — the TOML deserialiser
    rejected the document for a structural reason: a syntax error, a
    type mismatch, a field that is unknown for its enclosing table,
    or an enum tag (`kind`, `potential`, `mode`) whose string does not
    match any registered variant. `path` is a dotted, JSON-pointer-like
    location within the document; `message` is the underlying
    parser/deserialiser message.
    - For an **unknown enum tag** (e.g. `[integrator] kind =
      "custom"`), `path` is the path of the tag field itself
      (`"integrator.kind"`, `"pair_interactions[0].potential"`,
      `"bond_types[0].potential"`, `"angle_types[0].potential"`,
      `"neighbor_list.mode"`, etc.).
    - For an **unknown field** (e.g. `[integrator] kind =
      "velocity-verlet" friction = 1.0e12`), `path` is the path of
      the enclosing table (`"integrator"`, `"thermostat"`,
      `"barostat"`, `"pair_interactions[0]"`, `"bond_types[0]"`,
      `"angle_types[0]"`, `"neighbor_list"`). The offending field
      name appears in `message`.
    - For a **type mismatch or syntax error**, `path` is the path
      reached before the failure (empty string for top-level syntax
      errors).
  - `UnsupportedSchemaVersion { actual: u64, supported: u64 }` —
    `schema_version` was structurally a u64 but its value is not the
    supported version (`1`).
  - `MissingField { field: String }` — a required field is absent at
    deserialisation time. `field` uses the same dotted notation as
    `Parse.path` (e.g. `"simulation.dt"`, `"integrator.kind"`,
    `"pair_interactions[0].sigma"`).
  - `InvalidValue { field: String, reason: String }` — a field
    deserialised to the right type but failed a range, finiteness, or
    domain check (positive, non-NaN, in `[0, π]`, non-empty string,
    `r_switch <= cutoff`, etc.). Produced exclusively by the post-parse
    `Config::validate` pass; the deserialiser itself never emits this
    variant.
  - `DuplicateTypeName { name: String }` — two `[[particle_types]]`
    entries share a `name`.
  - `UnknownTypeInPair { name: String, pair_index: usize }` — a
    `[[pair_interactions]]` entry's `between` field names a type not
    declared in `[[particle_types]]`.
  - `MissingPairInteraction { types: (String, String) }` — the
    declared types omit an unordered pair from `[[pair_interactions]]`.
    Tuple is normalised so the lexicographically smaller name comes
    first.
  - `DuplicatePairInteraction { types: (String, String) }` — two
    `[[pair_interactions]]` entries cover the same unordered pair.
    Tuple normalised as above.
  - `PathCollision { kind_a: PathRole, kind_b: PathRole, path: PathBuf }`
    — two supplied file paths resolve to the same location.
  - `ConflictingElectrostatics` — the config declares both `[coulomb]`
    and `[spme]`. Only one electrostatics method may be active per run.
  - `IncompatibleThermostat { integrator: String }` — the config
    pairs an integrator that owns its own thermostat
    (`langevin-baoab` or `mtk-npt`) with a `[thermostat]` table.
  - `IncompatibleBarostat { integrator: String }` — the config
    pairs an integrator that owns its own barostat (currently only
    `mtk-npt`) with a `[barostat]` table.
  - `DuplicateBondTypeName { name: String }` — two `[[bond_types]]`
    entries share a `name`.
  - `DuplicateAngleTypeName { name: String }` — two `[[angle_types]]`
    entries share a `name`.
  - `DuplicateConstraintTypeName { name: String }` — two
    `[[constraint_types]]` entries share a `name`.
  - `IncompatibleConstraint { integrator: String }` — the topology
    file's `[constraints]` section is non-empty and
    `IntegratorKind::supports_constraints()` returns `false` for the
    chosen integrator (in the default registry: `langevin-baoab`,
    `mtk-npt`, or `velocity-verlet { lossless = true }`).
  - `SettleGeometryInfeasible { name: String, r_oh: f64, r_hh: f64 }`
    — a `[[constraint_types]]` entry with `kind = "settle-water"`
    declares `r_hh ≥ 2 · r_oh`.

  `Parse` covers every shape error the typed deserialiser flags
  structurally: unknown fields under a tagged-enum table
  (e.g. `lossless` under `[integrator] kind = "langevin-baoab"`,
  `seed` under `[barostat] kind = "berendsen"`, `max_neighbors` under
  `[neighbor_list] mode = "all-pairs"`); unknown enum tags
  (e.g. `kind = "not-a-real-integrator"`, `potential = "harmonic"` in
  `[[pair_interactions]]`, `mode = "kd-tree"` in `[neighbor_list]`);
  wrong TOML types (e.g. a string where an integer is required); and
  raw TOML syntax errors. `path` carries the document location so
  callers can present a useful diagnostic without inspecting
  `message`.

### Functions <!-- rq-39891001 -->

- `load_config(path: &Path) -> Result<Config, ConfigError>` <!-- rq-45bb8194 -->
  - Reads the file at `path`, runs the typed TOML deserialiser, fills
    in field-derived defaults (`r_switch = 0.9 * cutoff` for
    `[[pair_interactions]]` and `[coulomb]`, `r_skin = 0.3 * max_cutoff`
    for the `cell-list` `[neighbor_list]` mode, `[output]` defaults
    derived from the config-file stem), resolves every supplied path
    against `path.parent()` (or `"."` if `path` has no parent), calls
    `Config::validate(&config)` on the resulting `Config`, and returns
    it.
  - File-read failure yields `Io(String)`. Deserialiser failures yield
    either `MissingField`, `UnsupportedSchemaVersion`, or `Parse`
    depending on the failure kind (see the `ConfigError` description
    above). Range, finiteness, domain, and cross-validation failures
    yield the variant emitted by `Config::validate`.
  - On any validation failure, returns the first error encountered in
    declaration order: deserialiser errors first (top-level fields,
    then `[simulation]`, then `[integrator]`, then
    `[[particle_types]]`, then `[[pair_interactions]]`, then
    `[bond_types]` / `[angle_types]` / `[coulomb]` / `[spme]` /
    `[neighbor_list]` / `[output]`); then `Config::validate`'s
    field-domain checks in the same declaration order; then
    cross-validation rules 1–5 in the order listed under
    *Cross-validation*.

- `Config::validate(&self) -> Result<(), ConfigError>` <!-- rq-a54cc657 -->
  - Pure host-side function. Takes a `Config` (typically obtained from
    `load_config` after deserialisation, but also constructable in
    memory by callers) and applies every check documented under
    *Field reference* that is not already enforceable by the typed
    deserialiser: range/finiteness/domain constraints on numeric
    fields (positivity, NaN-rejection, `theta_0 in [0, π]`,
    `r_switch <= cutoff`, non-empty string identifiers), plus the
    five rules listed under *Cross-validation*.
  - On the first failure, returns the structured error variant
    documented for that check (`InvalidValue` for per-field domain
    failures; `DuplicateTypeName`, `UnknownTypeInPair`,
    `MissingPairInteraction`, `DuplicatePairInteraction`,
    `PathCollision`, `ConflictingElectrostatics`,
    `IncompatibleThermostat`, `IncompatibleBarostat`,
    `DuplicateBondTypeName`, or `DuplicateAngleTypeName` for the
    cross-validation rules).
  - Order of checks: per-field domain checks in the section order
    above, then cross-validation rules 1, 2, 3, 4, 5 in that order.
  - Calling `Config::validate` on a `Config` returned by
    `load_config` is a no-op (returns `Ok(())`): `load_config` already
    invoked it.

## Out of Scope <!-- rq-35722a66 -->

- Non-SI units (LJ-reduced, nm/ps, ...).
- Non-orthorhombic boxes (the box lives in the init file; see
  `init-state-file.md`).
- Potentials other than Lennard-Jones; bonded terms; long-range
  electrostatics.
- Mixing rules (Lorentz-Berthelot, geometric); every pair is enumerated
  explicitly.
- Restart files and resume semantics (a separate planned feature).
- CLI flag overrides of config fields. Every parameter lives in the file.
- Thermostats and barostats; the integrator is microcanonical.
- Per-particle or per-type initial temperatures (one global field).
- Compile-time `f64` precision feature flag.
- Filesystem existence checks for resolved paths; that is the runner's
  responsibility.

---

## Gherkin Scenarios <!-- rq-6aeb039a -->

```gherkin
Feature: TOML simulation config schema

  Background:
    Given a valid minimal config containing schema_version = 1, init = "argon.xyz",
      one [simulation] section with seed=12345, n_steps=10, dt=1.0e-15, temperature=300.0,
      one [integrator] section with kind="velocity-verlet" and lossless=false,
      one [[particle_types]] entry with name="Ar" and mass=6.6335e-26,
      one [[pair_interactions]] entry between=["Ar","Ar"], potential="lennard-jones",
        sigma=3.40e-10, epsilon=1.65e-21, cutoff=1.0e-9

  # --- Happy path ---

  @rq-7df1515f
  Scenario: Load a valid minimal config
    Given a config file written to "/tmp/sim/sim.toml" containing the Background
    When load_config("/tmp/sim/sim.toml") is called
    Then it returns Ok(config)
    And config.schema_version equals 1
    And config.simulation.seed equals 12345
    And config.simulation.n_steps equals 10
    And config.simulation.dt equals 1.0e-15
    And config.simulation.temperature equals 300.0
    And config.integrator matches IntegratorKind::VelocityVerlet { lossless: false }
    And config.particle_types has length 1
    And config.particle_types[0].name equals "Ar"
    And config.particle_types[0].mass equals 6.6335e-26
    And config.pair_interactions has length 1
    And config.pair_interactions[0].between equals ("Ar", "Ar")
    And config.pair_interactions[0].cutoff equals 1.0e-9
    And config.pair_interactions[0].potential matches PairPotentialParams::LennardJones { sigma: 3.40e-10, epsilon: 1.65e-21 }
    And config.init equals "/tmp/sim/argon.xyz"
    And config.config_path equals "/tmp/sim/sim.toml"

  @rq-894c16c4
  Scenario: Defaults populate the output section when [output] is omitted
    Given the Background config with no [output] section, written to "/tmp/sim/sim.toml"
    When load_config("/tmp/sim/sim.toml") is called
    Then config.output.trajectory_path equals "/tmp/sim/sim-traj.xyz"
    And config.output.trajectory_every equals 100
    And config.output.include_velocities equals true
    And config.output.include_images equals true
    And config.output.log_path equals "/tmp/sim/sim.log"
    And config.output.log_every equals 100

  @rq-d148149f
  Scenario: Explicit [output] values override defaults
    Given the Background config plus [output] with trajectory_every=50, log_every=25,
      trajectory_path="custom-traj.xyz", log_path="custom.log", include_velocities=false,
      written to "/tmp/sim/sim.toml"
    When load_config("/tmp/sim/sim.toml") is called
    Then config.output.trajectory_path equals "/tmp/sim/custom-traj.xyz"
    And config.output.trajectory_every equals 50
    And config.output.include_velocities equals false
    And config.output.log_path equals "/tmp/sim/custom.log"
    And config.output.log_every equals 25

  @rq-5ded1806
  Scenario: Absolute paths are honored unchanged
    Given the Background config with init="/data/argon.xyz",
      [output].trajectory_path="/data/out/traj.xyz",
      [output].log_path="/data/out/run.log"
    When load_config is called
    Then config.init equals "/data/argon.xyz"
    And config.output.trajectory_path equals "/data/out/traj.xyz"
    And config.output.log_path equals "/data/out/run.log"

  @rq-d5085350
  Scenario: Pair `between` is normalised to lexicographic order
    Given the Background config with between=["Kr","Ar"] (the only declared type is "Ar")
    When load_config is called
    Then it returns Err(ConfigError::UnknownTypeInPair { name: "Kr", pair_index: 0 })

  @rq-d3d4b6b3
  Scenario: Pair `between` normalisation when both types are declared
    Given a config with two declared types "Ar" and "Kr" and three pair_interactions
      between=["Kr","Ar"], ["Ar","Ar"], ["Kr","Kr"]
    When load_config is called
    Then it returns Ok(config)
    And every config.pair_interactions[i].between has lexicographically ordered names

  @rq-106dcabd
  Scenario: n_steps = 0 is accepted
    Given the Background config with n_steps=0
    When load_config is called
    Then it returns Ok(config)
    And config.simulation.n_steps equals 0

  # --- Schema version ---

  @rq-69a31102
  Scenario: Reject missing schema_version
    Given a TOML file with no schema_version field
    When load_config(path) is called
    Then it returns Err(ConfigError::MissingField { field: "schema_version" })

  @rq-0cb3c41c
  Scenario: Reject unknown schema_version
    Given the Background config with schema_version=2
    When load_config is called
    Then it returns Err(ConfigError::UnsupportedSchemaVersion { actual: 2, supported: 1 })

  @rq-4169d3af
  Scenario: Reject schema_version = 0
    Given the Background config with schema_version=0
    When load_config is called
    Then it returns Err(ConfigError::UnsupportedSchemaVersion { actual: 0, supported: 1 })

  # --- TOML and IO failures ---

  @rq-ae7f8045
  Scenario: File does not exist
    When load_config("/tmp/does-not-exist.toml") is called
    Then it returns Err(ConfigError::Io(_))

  @rq-57f8de41
  Scenario: Malformed TOML
    Given a file at "/tmp/sim.toml" containing the bytes "schema_version = ["
    When load_config("/tmp/sim.toml") is called
    Then it returns Err(ConfigError::Parse { .. })

  @rq-761f26c6
  Scenario: Unknown top-level key is permitted
    Given the Background config plus an extra top-level field unknown_key="x"
    When load_config is called
    Then it returns Ok(config)

  # --- Required fields ---

  @rq-f0e3b004
  Scenario: Missing init field
    Given the Background config with the `init` field removed
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "init" })

  @rq-9bfc2c1d
  Scenario: Missing [simulation].seed
    Given the Background config with the simulation.seed field removed
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "simulation.seed" })

  @rq-221b1bb4
  Scenario: Missing [simulation].dt
    Given the Background config with the simulation.dt field removed
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "simulation.dt" })

  @rq-52c9b17a
  Scenario: Missing [simulation].temperature
    Given the Background config with the simulation.temperature field removed
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "simulation.temperature" })

  @rq-66bf31c6
  Scenario: Missing [integrator] section is rejected
    Given the Background config with the [integrator] section removed
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "integrator" })

  @rq-100115a0
  Scenario: Missing integrator.kind is rejected
    Given the Background config with [integrator] containing only lossless=false (no kind field)
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "integrator.kind" })

  @rq-657bbbd6
  Scenario: Unknown integrator kind is rejected
    Given the Background config with [integrator] kind="custom"
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, .. })
    And path equals "integrator.kind"

  @rq-f067338a
  Scenario: [integrator] rejects an unknown field for the chosen kind
    Given the Background config with [integrator] kind="velocity-verlet"
      plus the unknown field friction=1.0e12
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, message })
    And path equals "integrator"
    And message mentions "friction"
    # The deserialiser reports the enclosing table's path; the offending
    # field name appears in the message. The same rejection applies to
    # every (kind, unknown-field) pair the parameterised unit test
    # enumerates. At minimum the test exercises
    #   (velocity-verlet, friction), (langevin-baoab, lossless),
    #   (mtk-npt, seed). Adding an integrator kind extends the matrix.

  @rq-86aa2be7
  Scenario: Langevin BAOAB kind with valid parameters is accepted
    Given the Background config with [integrator] kind="langevin-baoab",
      friction=1.0e12, temperature=300.0, seed=42
    When load_config is called
    Then it returns Ok(config)
    And config.integrator matches IntegratorKind::LangevinBaoab { friction: 1.0e12, temperature: 300.0, seed: 42 }

  @rq-40ed9975
  Scenario: Langevin BAOAB missing friction is rejected
    Given the Background config with [integrator] kind="langevin-baoab",
      temperature=300.0, seed=42 (no friction field)
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "integrator.friction" })

  @rq-f2431cc4
  Scenario: Langevin BAOAB missing temperature is rejected
    Given the Background config with [integrator] kind="langevin-baoab",
      friction=1.0e12, seed=42 (no temperature field)
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "integrator.temperature" })

  @rq-92f643cb
  Scenario: Langevin BAOAB missing seed is rejected
    Given the Background config with [integrator] kind="langevin-baoab",
      friction=1.0e12, temperature=300.0 (no seed field)
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "integrator.seed" })

  @rq-385408d0
  Scenario: Langevin BAOAB rejects friction=0
    Given the Background config with [integrator] kind="langevin-baoab",
      friction=0.0, temperature=300.0, seed=42
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "integrator.friction", reason: _ })

  @rq-583201cb
  Scenario: Langevin BAOAB rejects negative friction
    Given the Background config with [integrator] kind="langevin-baoab",
      friction=-1.0, temperature=300.0, seed=42
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "integrator.friction", reason: _ })

  @rq-789b7a33
  Scenario: Langevin BAOAB rejects non-positive temperature
    Given the Background config with [integrator] kind="langevin-baoab",
      friction=1.0e12, temperature=0.0, seed=42
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "integrator.temperature", reason: _ })

  @rq-66ec7ee4
  Scenario: Velocity-Verlet lossless defaults to false when omitted
    Given the Background config with [integrator] containing only kind="velocity-verlet"
    When load_config is called
    Then it returns Ok(config)
    And config.integrator matches IntegratorKind::VelocityVerlet { lossless: false }

  # --- Per-thermostat-kind parameter validation ---
  # Scenarios validating temperature / tau / seed / extra-field handling
  # for individual `[thermostat]` kinds live alongside each thermostat's
  # algorithmic scenarios:
  #   - nose-hoover-chain.md
  #   - csvr.md
  #   - andersen.md
  #   - berendsen.md
  # They reference `ConfigError::MissingField { field: "thermostat.*" }`,
  # `ConfigError::InvalidValue { field: "thermostat.*", .. }`, and
  # `ConfigError::Parse { path: "thermostat.*", .. }` to exercise the
  # same loader path documented in this section.

  @rq-1e1c5f3b
  Scenario: Missing [[particle_types]] is rejected
    Given the Background config with no [[particle_types]] entries
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "particle_types" })

  @rq-a94d2c13
  Scenario: Missing [[pair_interactions]] is rejected
    Given the Background config with no [[pair_interactions]] entries
    When load_config is called
    Then it returns Err(ConfigError::MissingPairInteraction { types: ("Ar", "Ar") })

  # --- Per-field validation ---

  @rq-025b2c3b
  Scenario: Reject non-positive dt
    Given the Background config with simulation.dt=0.0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "simulation.dt", reason: _ })

  @rq-0051b248
  Scenario: Reject negative dt
    Given the Background config with simulation.dt=-1.0e-15
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "simulation.dt", reason: _ })

  @rq-dffdd81c
  Scenario: Reject NaN dt
    Given the Background config with simulation.dt=nan
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "simulation.dt", reason: _ })

  @rq-f009e02b
  Scenario: Reject negative temperature
    Given the Background config with simulation.temperature=-1.0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "simulation.temperature", reason: _ })

  @rq-cc12f2d8
  Scenario: Zero temperature is accepted
    Given the Background config with simulation.temperature=0.0
    When load_config is called
    Then it returns Ok(config)
    And config.simulation.temperature equals 0.0

  @rq-47697f4a
  Scenario: Reject non-positive mass
    Given the Background config with particle_types[0].mass=0.0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "particle_types[0].mass", reason: _ })

  @rq-aa19f894
  Scenario: Reject non-positive sigma
    Given the Background config with pair_interactions[0].sigma=0.0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "pair_interactions[0].sigma", reason: _ })

  @rq-017b6769
  Scenario: Reject non-positive epsilon
    Given the Background config with pair_interactions[0].epsilon=-1.0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "pair_interactions[0].epsilon", reason: _ })

  @rq-ae65c293
  Scenario: Reject non-positive cutoff
    Given the Background config with pair_interactions[0].cutoff=0.0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "pair_interactions[0].cutoff", reason: _ })

  @rq-d1d84e31
  Scenario: Accept user-supplied r_switch in pair_interactions
    Given the Background config with pair_interactions[0].cutoff=1.0e-9
      and pair_interactions[0].r_switch=9.0e-10
    When load_config is called
    Then it returns Ok
    And config.pair_interactions[0].r_switch equals 9.0e-10

  @rq-6f4f5ece
  Scenario: Default r_switch to 0.9 * cutoff when omitted
    Given the Background config with pair_interactions[0].cutoff=1.0e-9
      and no r_switch field on pair_interactions[0]
    When load_config is called
    Then it returns Ok
    And config.pair_interactions[0].r_switch equals 9.0e-10 within f64 round-off

  @rq-1d8b8efe
  Scenario: Accept r_switch equal to cutoff (hard-cutoff degenerate case)
    Given the Background config with pair_interactions[0].cutoff=1.0e-9
      and pair_interactions[0].r_switch=1.0e-9
    When load_config is called
    Then it returns Ok
    And config.pair_interactions[0].r_switch equals 1.0e-9

  @rq-7cd9471a
  Scenario: Reject r_switch greater than cutoff
    Given the Background config with pair_interactions[0].cutoff=1.0e-9
      and pair_interactions[0].r_switch=1.1e-9
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "pair_interactions[0].r_switch", reason: _ })

  @rq-b4d2f559
  Scenario: Reject non-positive r_switch
    Given the Background config with pair_interactions[0].r_switch=0.0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "pair_interactions[0].r_switch", reason: _ })

  @rq-871f0292
  Scenario: Reject non-finite r_switch
    Given the Background config with pair_interactions[0].r_switch=NaN
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "pair_interactions[0].r_switch", reason: _ })

  @rq-e38aac7b
  Scenario: Reject unknown pair potential
    Given the Background config with pair_interactions[0].potential="morse"
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, .. })
    And path equals "pair_interactions[0].potential"

  @rq-45a14d49
  Scenario: Lennard-Jones pair interaction missing sigma is rejected
    Given the Background config with pair_interactions[0] having potential="lennard-jones",
      epsilon=1.65e-21, cutoff=1.0e-9, and no sigma field
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "pair_interactions[0].sigma" })

  @rq-053613b6
  Scenario: Pair interaction rejects an extra field for the chosen potential
    Given the Background config with pair_interactions[0] having potential="lennard-jones"
      and an unknown field stiffness=1.0
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, message })
    And path equals "pair_interactions[0]"
    And message mentions "stiffness"

  @rq-a30ac09f
  Scenario: Reject empty type name
    Given the Background config with particle_types[0].name=""
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "particle_types[0].name", reason: _ })

  # --- Cross-validation ---

  @rq-560dffb8
  Scenario: Reject duplicate type names
    Given a config with two [[particle_types]] both named "Ar"
    When load_config is called
    Then it returns Err(ConfigError::DuplicateTypeName { name: "Ar" })

  @rq-c9fa5cda
  Scenario: Reject pair referencing an unknown type
    Given the Background config plus an extra [[pair_interactions]] between=["Ar","Xe"]
    When load_config is called
    Then it returns Err(ConfigError::UnknownTypeInPair { name: "Xe", pair_index: 1 })

  @rq-ae6d5db8
  Scenario: Reject missing pair interaction
    Given a config with two declared types "Ar" and "Kr"
      and pair_interactions containing only between=["Ar","Ar"] and between=["Kr","Kr"]
    When load_config is called
    Then it returns Err(ConfigError::MissingPairInteraction { types: ("Ar", "Kr") })

  @rq-f11e9d4c
  Scenario: Reject duplicate pair interaction
    Given a config with two pair_interactions both between=["Ar","Ar"]
    When load_config is called
    Then it returns Err(ConfigError::DuplicatePairInteraction { types: ("Ar", "Ar") })

  @rq-9e4d8944
  Scenario: Reject duplicate pair under different orderings
    Given a config with pair_interactions between=["Ar","Kr"] and between=["Kr","Ar"]
      with two declared types "Ar" and "Kr" and a same-type pair for each
    When load_config is called
    Then it returns Err(ConfigError::DuplicatePairInteraction { types: ("Ar", "Kr") })

  # --- Path collision ---

  @rq-e553c05b
  Scenario: Reject init = trajectory_path
    Given the Background config at "/tmp/sim/sim.toml" with init="out.xyz" and
      [output].trajectory_path="out.xyz"
    When load_config is called
    Then it returns Err(ConfigError::PathCollision { kind_a: PathRole::Init, kind_b: PathRole::Trajectory, path: "/tmp/sim/out.xyz" })

  @rq-765c96c5
  Scenario: Reject trajectory_path = log_path
    Given the Background config with [output].trajectory_path="run.dat" and
      [output].log_path="run.dat"
    When load_config is called
    Then it returns Err(ConfigError::PathCollision { kind_a: PathRole::Trajectory, kind_b: PathRole::Log, path: _ })

  @rq-330d6b42
  Scenario: Reject init = log_path
    Given the Background config with init="run.log" and [output].log_path="run.log"
    When load_config is called
    Then it returns Err(ConfigError::PathCollision { kind_a: PathRole::Init, kind_b: PathRole::Log, path: _ })

  @rq-a5c86770
  Scenario: timings_path defaults to <stem>.timings
    Given the Background config at "/tmp/sim/sim.toml" with no [output].timings_path
    When load_config is called
    Then config.output.timings_path equals "/tmp/sim/sim.timings"

  @rq-fa24a8d1
  Scenario: timings_path can be overridden in [output]
    Given the Background config at "/tmp/sim/sim.toml" with
      [output].timings_path = "custom.timings"
    When load_config is called
    Then config.output.timings_path equals "/tmp/sim/custom.timings"

  @rq-7d5915bb
  Scenario: Reject init = timings_path
    Given the Background config with init="argon.dat" and
      [output].timings_path="argon.dat"
    When load_config is called
    Then it returns Err(ConfigError::PathCollision { kind_a: PathRole::Init, kind_b: PathRole::Timings, path: _ })

  @rq-ec8d715d
  Scenario: Reject trajectory_path = timings_path
    Given the Background config with [output].trajectory_path="run.dat"
      and [output].timings_path="run.dat"
    When load_config is called
    Then it returns Err(ConfigError::PathCollision { kind_a: PathRole::Trajectory, kind_b: PathRole::Timings, path: _ })

  @rq-8f665dd0
  Scenario: Reject log_path = timings_path
    Given the Background config with [output].log_path="run.dat"
      and [output].timings_path="run.dat"
    When load_config is called
    Then it returns Err(ConfigError::PathCollision { kind_a: PathRole::Log, kind_b: PathRole::Timings, path: _ })

  # --- Bonds field ---

  @rq-6cb9ab62
  Scenario: topology field is optional and defaults to None
    Given the Background config without a top-level `topology` field
    When load_config is called
    Then it returns Ok(config)
    And config.topology equals None

  @rq-027153d9
  Scenario: topology field is resolved relative to the config directory
    Given the Background config at "/tmp/sim/sim.toml" with topology="argon.topology"
    When load_config is called
    Then config.topology equals Some("/tmp/sim/argon.topology")

  @rq-576561a2
  Scenario: topology absolute path is preserved
    Given the Background config with topology="/data/argon.topology"
    When load_config is called
    Then config.topology equals Some("/data/argon.topology")

  @rq-4186d4f4
  Scenario: Reject topology = init
    Given the Background config with init="argon.xyz" and topology="argon.xyz"
    When load_config is called
    Then it returns Err(ConfigError::PathCollision { kind_a: PathRole::Init, kind_b: PathRole::Topology, path: _ })

  @rq-98180119
  Scenario: Reject topology = trajectory_path
    Given the Background config with [output].trajectory_path="run.dat" and topology="run.dat"
    When load_config is called
    Then it returns Err(ConfigError::PathCollision { kind_a: PathRole::Trajectory, kind_b: PathRole::Topology, path: _ })

  # --- Bond types ---

  @rq-6ad9a0f8
  Scenario: bond_types is optional and defaults to empty
    Given the Background config without a [[bond_types]] array
    When load_config is called
    Then it returns Ok(config)
    And config.bond_types is empty

  @rq-f704561b
  Scenario: Valid Morse bond_type is accepted
    Given the Background config plus
      [[bond_types]] name="ArAr" potential="morse" de=1.65e-21 a=1.9e10 re=3.4e-10
    When load_config is called
    Then it returns Ok(config)
    And config.bond_types has length 1
    And config.bond_types[0] matches BondTypeConfig::Morse { name: "ArAr", de: 1.65e-21, a: 1.9e10, re: 3.4e-10 }

  @rq-c79a1408
  Scenario: bond_type missing potential field
    Given the Background config plus a [[bond_types]] entry with name="X" but no potential
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "bond_types[0].potential" })

  @rq-3f01c746
  Scenario: bond_type unknown potential is rejected
    Given a [[bond_types]] entry with potential="harmonic"
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, .. })
    And path equals "bond_types[0].potential"

  @rq-3b0e8140
  Scenario: Morse bond_type missing de is rejected
    Given a [[bond_types]] entry with potential="morse", a=1.9e10, re=3.4e-10 (no de)
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "bond_types[0].de" })

  @rq-ecc8f632
  Scenario: Morse bond_type rejects non-positive de
    Given a [[bond_types]] entry with potential="morse", de=0.0, a=1.9e10, re=3.4e-10
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "bond_types[0].de", reason: _ })

  @rq-ae85bf7b
  Scenario: Morse bond_type rejects non-positive a
    Given a [[bond_types]] entry with potential="morse", de=1.0, a=-1.0, re=1.0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "bond_types[0].a", reason: _ })

  @rq-3533e8a9
  Scenario: Morse bond_type rejects non-positive re
    Given a [[bond_types]] entry with potential="morse", de=1.0, a=1.0, re=0.0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "bond_types[0].re", reason: _ })

  @rq-a208c9ba
  Scenario: Morse bond_type rejects extra fields
    Given a [[bond_types]] entry with potential="morse" and an unknown field stiffness=1.0
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, message })
    And path equals "bond_types[0]"
    And message mentions "stiffness"

  @rq-ed1d6c71
  Scenario: Reject duplicate bond_type names
    Given two [[bond_types]] entries with the same name "ArAr"
    When load_config is called
    Then it returns Err(ConfigError::DuplicateBondTypeName { name: "ArAr" })

  @rq-50521f04
  Scenario: Empty bond_type name rejected
    Given a [[bond_types]] entry with name=""
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "bond_types[0].name", reason: _ })

  # --- Angle types ---

  @rq-24dc9578
  Scenario: angle_types is optional and defaults to empty
    Given the Background config without an [[angle_types]] array
    When load_config is called
    Then it returns Ok(config)
    And config.angle_types is empty

  @rq-91bf10ec
  Scenario: Valid harmonic angle_type is accepted
    Given the Background config plus
      [[angle_types]] name="HOH" potential="harmonic" k_theta=5.27e-19 theta_0=1.911
    When load_config is called
    Then it returns Ok(config)
    And config.angle_types has length 1
    And config.angle_types[0] matches AngleTypeConfig::Harmonic { name: "HOH", k_theta: 5.27e-19, theta_0: 1.911 }

  @rq-57518e01
  Scenario: angle_type missing potential field
    Given the Background config plus an [[angle_types]] entry with name="X" but no potential
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "angle_types[0].potential" })

  @rq-ffa771bd
  Scenario: angle_type unknown potential is rejected
    Given an [[angle_types]] entry with potential="cosine-harmonic"
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, .. })
    And path equals "angle_types[0].potential"

  @rq-dc94d9e3
  Scenario: Harmonic angle_type missing k_theta is rejected
    Given an [[angle_types]] entry with potential="harmonic", theta_0=1.911 (no k_theta)
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "angle_types[0].k_theta" })

  @rq-aad6ca63
  Scenario: Harmonic angle_type rejects non-positive k_theta
    Given an [[angle_types]] entry with potential="harmonic", k_theta=0.0, theta_0=1.911
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "angle_types[0].k_theta", reason: _ })

  @rq-e399422c
  Scenario: Harmonic angle_type rejects theta_0 outside [0, π]
    Given an [[angle_types]] entry with potential="harmonic", k_theta=5.27e-19, theta_0=4.0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "angle_types[0].theta_0", reason: _ })

  @rq-c5fa34f5
  Scenario: Harmonic angle_type rejects extra fields
    Given an [[angle_types]] entry with potential="harmonic" and an unknown field stiffness=1.0
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, message })
    And path equals "angle_types[0]"
    And message mentions "stiffness"

  @rq-9255c192
  Scenario: Reject duplicate angle_type names
    Given two [[angle_types]] entries with the same name "HOH"
    When load_config is called
    Then it returns Err(ConfigError::DuplicateAngleTypeName { name: "HOH" })

  @rq-dc35ae30
  Scenario: Empty angle_type name rejected
    Given an [[angle_types]] entry with name=""
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "angle_types[0].name", reason: _ })

  # --- Neighbor list ---

  @rq-e2f32af0
  Scenario: neighbor_list defaults to cell-list with derived r_skin when [neighbor_list] is omitted
    Given the Background config with no [neighbor_list] section
    When load_config is called
    Then it returns Ok(config)
    And config.neighbor_list matches NeighborListConfig::CellList { max_neighbors: 256, r_skin: 3.0e-10 }

  @rq-b1f33ea4
  Scenario: neighbor_list cell-list with explicit parameters is accepted
    Given the Background config plus
      [neighbor_list] mode="cell-list" max_neighbors=128 r_skin=2.0e-10
    When load_config is called
    Then it returns Ok(config)
    And config.neighbor_list matches NeighborListConfig::CellList { max_neighbors: 128, r_skin: 2.0e-10 }

  @rq-e643b070
  Scenario: neighbor_list cell-list mode omitting max_neighbors uses default 256
    Given the Background config plus
      [neighbor_list] mode="cell-list" r_skin=2.0e-10
    When load_config is called
    Then it returns Ok(config)
    And config.neighbor_list matches NeighborListConfig::CellList { max_neighbors: 256, r_skin: 2.0e-10 }

  @rq-cde6e114
  Scenario: neighbor_list cell-list mode omitting r_skin uses 0.3 * max cutoff
    Given the Background config plus
      [neighbor_list] mode="cell-list" max_neighbors=128
    When load_config is called
    Then it returns Ok(config)
    And config.neighbor_list matches NeighborListConfig::CellList { max_neighbors: 128, r_skin: 3.0e-10 }

  @rq-931f1ab8
  Scenario: neighbor_list all-pairs mode is accepted
    Given the Background config plus [neighbor_list] mode="all-pairs"
    When load_config is called
    Then it returns Ok(config)
    And config.neighbor_list matches NeighborListConfig::AllPairs

  @rq-0a92d90b
  Scenario: Unknown neighbor_list mode is rejected
    Given the Background config plus [neighbor_list] mode="kd-tree"
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, .. })
    And path equals "neighbor_list.mode"

  @rq-13ca0415
  Scenario: [neighbor_list] rejects unknown fields for the chosen mode
    Given the Background config plus [neighbor_list] mode="all-pairs"
      and the unknown field max_neighbors=128
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, message })
    And path equals "neighbor_list"
    And message mentions "max_neighbors"
    # The same rejection applies to every (mode, unknown-field) pair the
    # parameterised unit test enumerates. At minimum the test exercises
    #   (all-pairs, max_neighbors), (all-pairs, r_skin),
    #   (cell-list, stencil="huge").

  @rq-fedef74d
  Scenario: neighbor_list rejects zero max_neighbors
    Given the Background config plus [neighbor_list] mode="cell-list" max_neighbors=0 r_skin=1.0e-10
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "neighbor_list.max_neighbors", reason: _ })

  @rq-f7856bcc
  Scenario: neighbor_list rejects non-positive r_skin
    Given the Background config plus [neighbor_list] mode="cell-list" max_neighbors=128 r_skin=0.0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "neighbor_list.r_skin", reason: _ })

  @rq-ec28cbfb
  Scenario: neighbor_list rejects non-finite r_skin
    Given the Background config plus [neighbor_list] mode="cell-list" max_neighbors=128 r_skin=inf
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "neighbor_list.r_skin", reason: _ })

  # --- Multi-type configs ---

  @rq-f114c560
  Scenario: Accept a two-type config with a complete pair_interactions table
    Given a config with two declared types "Ar" and "Kr"
      and pair_interactions for all three unordered pairs ("Ar","Ar"), ("Ar","Kr"), ("Kr","Kr")
    When load_config is called
    Then it returns Ok(config)
    And config.particle_types has length 2
    And config.pair_interactions has length 3

  @rq-66dfc50f
  Scenario: Reject a two-type config that omits a pair
    Given a config with two declared types "Ar" and "Kr"
      and pair_interactions for ("Ar","Ar") and ("Kr","Kr") only
    When load_config is called
    Then it returns Err(ConfigError::MissingPairInteraction { types: ("Ar", "Kr") })

  # --- Output cadence semantics ---

  @rq-97e525d8
  Scenario: trajectory_every = 0 is accepted (disables trajectory output)
    Given the Background config with [output].trajectory_every=0
    When load_config is called
    Then it returns Ok(config)
    And config.output.trajectory_every equals 0

  @rq-318cd47d
  Scenario: log_every = 0 is accepted (disables log output)
    Given the Background config with [output].log_every=0
    When load_config is called
    Then it returns Ok(config)
    And config.output.log_every equals 0

  # --- [thermostat] presence and absence ---

  @rq-ca356c08
  Scenario: [thermostat] section absent yields config.thermostat = None
    Given the Background config with no [thermostat] section
    When load_config is called
    Then it returns Ok(config)
    And config.thermostat is None

  @rq-b7cd6d16
  Scenario: [thermostat] kind = "berendsen" is accepted
    Given the Background config with [integrator] kind="velocity-verlet"
    And a [thermostat] section with kind="berendsen", temperature=300.0, tau=1.0e-13
    When load_config is called
    Then it returns Ok(config)
    And config.thermostat matches Some(ThermostatKind::Berendsen { temperature: 300.0, tau: 1.0e-13 })

  @rq-5c28eee0
  Scenario: [thermostat] kind = "csvr" requires a seed
    Given the Background config with [integrator] kind="velocity-verlet"
    And a [thermostat] section with kind="csvr", temperature=300.0, tau=1.0e-13
      (no seed)
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "thermostat.seed" })

  @rq-19ccc047
  Scenario: [thermostat] unknown kind is rejected
    Given the Background config with [integrator] kind="velocity-verlet"
    And a [thermostat] section with kind="not-a-real-thermostat"
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, .. })
    And path equals "thermostat.kind"

  @rq-c4a74903
  Scenario: [thermostat] missing kind is rejected
    Given the Background config with [integrator] kind="velocity-verlet"
    And a [thermostat] section containing only temperature=300.0
      (no kind field)
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "thermostat.kind" })

  @rq-f4eeb849
  Scenario: [thermostat] rejects an unknown field for the chosen kind
    Given the Background config with [integrator] kind="velocity-verlet"
    And a [thermostat] section with kind="berendsen", temperature=300.0,
      tau=1.0e-13, plus the unknown field seed=42
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, message })
    And path equals "thermostat"
    And message mentions "seed"
    # The same rejection applies to every (kind, unknown-field) pair the
    # parameterised unit test enumerates. At minimum the test exercises
    #   (berendsen, seed), (csvr, chain_length), (andersen, tau),
    #   (nose-hoover-chain, collision_rate). Adding a kind extends the
    #   matrix.

  # --- Integrator-owns-thermostat compatibility ---

  @rq-bdd03f85
  Scenario: langevin-baoab + [thermostat] is rejected
    Given the Background config with [integrator] kind="langevin-baoab",
      friction=1.0e12, temperature=300.0, seed=1
    And a [thermostat] section with kind="csvr", temperature=300.0,
      tau=1.0e-13, seed=2
    When load_config is called
    Then it returns Err(ConfigError::IncompatibleThermostat {
      integrator: "langevin-baoab" })

  @rq-c4ae19e0
  Scenario: velocity-verlet + [thermostat] is accepted
    Given the Background config with [integrator] kind="velocity-verlet"
    And a [thermostat] section with kind="csvr", temperature=300.0,
      tau=1.0e-13, seed=1
    When load_config is called
    Then it returns Ok(config)
    And config.integrator matches IntegratorKind::VelocityVerlet { .. }
    And config.thermostat matches Some(ThermostatKind::Csvr { .. })

  @rq-4cd2ec5b
  Scenario: IntegratorKind::owns_thermostat agrees with the validation rule
    Given an IntegratorKind::VelocityVerlet { lossless: false }
    Then kind.owns_thermostat() returns false
    Given an IntegratorKind::LangevinBaoab { friction: 1.0, temperature: 300.0, seed: 0 }
    Then kind.owns_thermostat() returns true

  @rq-23806456
  Scenario: MtkNpt integrator with all required fields is accepted
    Given the Background config with [integrator] kind="mtk-npt",
      temperature=85.0, pressure=1.0e5, tau_t=1.0e-13, tau_p=1.0e-12
    When load_config is called
    Then it returns Ok(config)
    And config.integrator matches IntegratorKind::MtkNpt {
      temperature: 85.0, pressure: 1.0e5, tau_t: 1.0e-13, tau_p: 1.0e-12,
      chain_length: 3, yoshida_order: 3, n_resp: 1 }

  @rq-d572a90a
  Scenario: MtkNpt missing tau_p is rejected
    Given the Background config with [integrator] kind="mtk-npt",
      temperature=85.0, pressure=1.0e5, tau_t=1.0e-13 (no tau_p)
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "integrator.tau_p" })

  @rq-129edb76
  Scenario: MtkNpt + [thermostat] is rejected (owns its thermostat)
    Given the Background config with [integrator] kind="mtk-npt",
      temperature=85.0, pressure=1.0e5, tau_t=1.0e-13, tau_p=1.0e-12
    And a [thermostat] section with kind="csvr",
      temperature=85.0, tau=1.0e-13, seed=1
    When load_config is called
    Then it returns Err(ConfigError::IncompatibleThermostat {
      integrator: "mtk-npt" })

  @rq-fbb836fb
  Scenario: MtkNpt + [barostat] is rejected (owns its barostat)
    Given the Background config with [integrator] kind="mtk-npt",
      temperature=85.0, pressure=1.0e5, tau_t=1.0e-13, tau_p=1.0e-12
    And a [barostat] section with kind="berendsen", pressure=1.0e5,
      tau=1.0e-12, compressibility=4.5e-10
    When load_config is called
    Then it returns Err(ConfigError::IncompatibleBarostat {
      integrator: "mtk-npt" })

  @rq-014a82ef
  Scenario: IntegratorKind::owns_barostat matrix
    Given an IntegratorKind::VelocityVerlet { lossless: false }
    Then kind.owns_barostat() returns false
    Given an IntegratorKind::LangevinBaoab { friction: 1.0, temperature: 300.0, seed: 0 }
    Then kind.owns_barostat() returns false
    Given an IntegratorKind::MtkNpt { temperature: 85.0, pressure: 1.0e5,
      tau_t: 1.0e-13, tau_p: 1.0e-12,
      chain_length: 3, yoshida_order: 3, n_resp: 1 }
    Then kind.owns_barostat() returns true

  # --- [barostat] section ---

  @rq-4bbbada4
  Scenario: [barostat] section absent yields config.barostat = None
    Given the Background config with no [barostat] section
    When load_config is called
    Then it returns Ok(config)
    And config.barostat is None

  @rq-f03e2af2
  Scenario: Unknown [barostat] kind is rejected
    Given the Background config with [integrator] kind="velocity-verlet"
    And a [barostat] section with kind="not-a-real-barostat"
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, .. })
    And path equals "barostat.kind"

  @rq-5d91f07d
  Scenario: [barostat] rejects an unknown field for the chosen kind
    Given the Background config with [integrator] kind="velocity-verlet"
    And a [barostat] section with kind="berendsen", pressure=1.0e5,
      tau=1.0e-12, compressibility=4.5e-10, plus the unknown field seed=42
    When load_config is called
    Then it returns Err(ConfigError::Parse { path, message })
    And path equals "barostat"
    And message mentions "seed"
    # The same rejection applies to every (kind, unknown-field) pair the
    # parameterised unit test enumerates. At minimum the test exercises
    #   (berendsen, seed), (c-rescale, friction). Adding a kind extends
    #   the matrix.

  @rq-bda9c0a2
  Scenario: [barostat] missing kind is rejected
    Given the Background config with [integrator] kind="velocity-verlet"
    And a [barostat] section containing only an arbitrary field
      (no kind field)
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "barostat.kind" })

  @rq-0fcb5a1e
  Scenario: [barostat] kind = "berendsen" with all required fields accepted
    Given the Background config with [integrator] kind="velocity-verlet"
    And a [barostat] section with kind="berendsen", pressure=1.0e5,
      tau=1.0e-12, compressibility=4.5e-10
    When load_config is called
    Then it returns Ok(config)
    And config.barostat matches
      Some(BarostatKind::Berendsen { pressure: 1.0e5, tau: 1.0e-12, compressibility: 4.5e-10 })

  @rq-c1d79d33
  Scenario: [barostat] kind = "c-rescale" with all required fields accepted
    Given the Background config with [integrator] kind="velocity-verlet"
    And a [barostat] section with kind="c-rescale", pressure=1.0e5,
      temperature=85.0, tau=1.0e-12, compressibility=4.5e-10, seed=42
    When load_config is called
    Then it returns Ok(config)
    And config.barostat matches Some(BarostatKind::CRescale {
      pressure: 1.0e5, temperature: 85.0, tau: 1.0e-12,
      compressibility: 4.5e-10, seed: 42 })

  @rq-d623f23d
  Scenario: C-rescale barostat missing temperature is rejected
    Given the Background config with [integrator] kind="velocity-verlet"
    And a [barostat] section with kind="c-rescale", pressure=1.0e5,
      tau=1.0e-12, compressibility=4.5e-10, seed=1
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "barostat.temperature" })

  @rq-b33d1ff0
  Scenario: C-rescale barostat missing seed is rejected
    Given the Background config with [integrator] kind="velocity-verlet"
    And a [barostat] section with kind="c-rescale", pressure=1.0e5,
      temperature=85.0, tau=1.0e-12, compressibility=4.5e-10
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "barostat.seed" })

  # Per-field parameter-validation scenarios for the Berendsen and
  # C-rescale barostats (missing pressure / tau / compressibility / seed,
  # non-positive values, negative pressure accepted, etc.) live alongside
  # the algorithm scenarios in `integration/berendsen-barostat.md` and
  # `integration/c-rescale-barostat.md`. They reference
  # `ConfigError::MissingField { field: "barostat.*" }` and
  # `ConfigError::InvalidValue { field: "barostat.*", .. }` to exercise
  # the same loader path documented in this section.
```
