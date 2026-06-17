# Feature: TOML Simulation Config Schema <!-- rq-6432ab1f -->

A simulation is specified by a TOML configuration file. The config pins every
parameter required to predict the trajectory bit-exactly. It carries no
positions or velocities; those live in a separate extended-XYZ initial-state
file referenced from the config.

The runner consumes the config via `heddlemd run <path>` (see
`simulation-runner.md`). Trajectory and log outputs are described in
`trajectory-output.md` and `log-output.md` respectively.

## Schema <!-- rq-1c7a9cfd -->

The top-level table carries one mandatory field, `schema_version`. The schema
described here is version `1`. Loading a config whose `schema_version` is any
other value is an error.

A simulation is a sequence of one or more **phases**. Phases come in
two kinds:

- **MD phases** declared as `[[phase]]` array entries. Each carries
  its own timestep, step count, integrator, optional thermostat,
  optional barostat, and optional output cadences.
- **Minimization phases** declared as `[[minimization]]` array
  entries. Each carries an algorithm selector
  (`steepest-descent` in v1), algorithm-specific parameters,
  convergence criteria, and output cadences. See
  `rqm/minimization/steepest-descent.md` for the algorithm.

A config carries at least one phase across the two arrays combined
(neither array is individually required, but the union must be
non-empty). Phases execute in **source-document order** across both
arrays: the deserialiser captures the byte span of each `[[phase]]`
and `[[minimization]]` entry via `toml::Spanned<T>` and the loader
merges them into a single `Vec<PhaseKind>` sorted by span start.
MD phases and minimization phases may be freely interleaved.

Particle state (positions, velocities, simulation box, particle
types, charges) carries over between phases regardless of kind; slot
state (integrator buffers, thermostat chains, barostat state, the
minimizer's adaptive step size) is reset at every phase boundary. See
`simulation-runner.md` for the runtime model.

Sections:

| Section | Required | Purpose |
| ------- | -------- | ------- |
| top-level `schema_version` | yes | format version |
| top-level `units` | no | unit system the file is written in (`"si"` default, or `"atomic"`) |
| top-level `init` | yes | path to initial-state file |
| top-level `topology` | no | path to .topology file |
| `[simulation]` | yes | RNG seed and target temperature for initial-velocity sampling |
| `[[phase]]` | conditional | per-phase integrator / thermostat / barostat / outputs (at least one of `[[phase]]` or `[[minimization]]` must be present) |
| `[[minimization]]` | conditional | per-phase minimization algorithm / parameters / outputs |
| `[[particle_types]]` | yes (>= 1) | per-type properties |
| `[[pair_interactions]]` | yes (covers every pair) | per-pair potential + parameters |
| `[[bond_types]]` | no | per-bond-type parameters |
| `[[angle_types]]` | no | per-angle-type parameters |
| `[[constraint_types]]` | no | per-constraint-type parameters |
| `[coulomb]` | no | truncated short-range Coulomb |
| `[spme]` | no | smooth particle-mesh Ewald |
| `[neighbor_list]` | no | non-bonded pair-evaluation algorithm |

### Example <!-- rq-ecc664ff -->

Saved as `argon.in.toml`. A two-phase protocol: an NVT equilibration
phase followed by an NPT production phase. Each phase writes its own
trajectory/log/timings file (see *Per-phase output paths* below).

```toml
schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 12345
temperature = 300.0 # K (used only if init file lacks a `velo` column;
                    #     applied at phase-0 entry)

# --- Phase 1: NVT equilibration ---

[[phase]]
name = "equil"
n_steps = 5000
dt = 1.0e-15        # s

[phase.integrator]
kind = "velocity-verlet"
lossless = false

[phase.thermostat]
kind = "csvr"
temperature = 300.0
tau = 1.0e-13
seed = 11           # any per-phase stochastic-slot seed is required

[phase.output]
# trajectory_every = 0 skips trajectory output during equilibration.
trajectory_every = 0
log_every = 100

# --- Phase 2: NPT production ---

[[phase]]
name = "production"
n_steps = 10000
dt = 1.0e-15

[phase.integrator]
kind = "velocity-verlet"
lossless = false

[phase.thermostat]
kind = "csvr"
temperature = 300.0
tau = 1.0e-13
seed = 12           # distinct from the equilibration phase's seed

[phase.barostat]
kind = "c-rescale"
pressure = 1.0e5
temperature = 300.0
tau = 1.0e-12
compressibility = 4.5e-10
seed = 13

[phase.output]
trajectory_every = 100
include_velocities = true
include_images = true
log_every = 100

# --- Global fields below apply to every phase ---

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
topology = "argon.in.topology"

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
kind = "shake"
atoms = 3
constraints = [
    { i = 0, j = 1, d = 1.0e-10 },     # O-H1
    { i = 0, j = 2, d = 1.0e-10 },     # O-H2
    { i = 1, j = 2, d = 1.633e-10 },   # H1-H2
]

[neighbor_list]
mode = "cell-list"
max_neighbors = 256
r_skin = 1.0e-10    # m  (defaults to 0.3 * cutoff when omitted)
```

### Per-phase output paths <!-- rq-f45b8166 -->

Each `[[phase]]` produces three output files in the directory
containing the config:

- `<root>.out.<phase-name>.xyz` — trajectory frames written during
  that phase (omitted entirely when the phase's
  `output.trajectory_every == 0`).
- `<root>.out.<phase-name>.log` — CSV log written during that phase
  (omitted when `output.log_every == 0`).
- `<root>.out.<phase-name>.timings` — per-stage performance summary
  for that phase (always written when the phase runs to completion).

`<root>` is the config-file root (the `<root>.in.toml` filename with
`.in.toml` stripped; see *Config filename convention* below).
`<phase-name>` is the per-phase `name` field. Phase names must be
unique within the config; the loader rejects duplicates with
`ConfigError::DuplicatePhaseName { name }`.

A phase's `[phase.output]` block accepts explicit
`trajectory_path`, `log_path`, and `timings_path` overrides; when
omitted, the three paths default to the form above. The path-
collision rules in *Validation* enforce that no two paths anywhere in
the config resolve to the same location.

### Units <!-- rq-ed997636 -->

The top-level `units` field selects the unit system the user's TOML
file, the referenced `.in.xyz` initial-state file, and every output
file the run produces are written in. Two values are accepted: `"si"`
(the default, equivalent to omitting the field) and `"atomic"`
(Hartree atomic units). Any other value is rejected with
`ConfigError::UnknownUnits { got: <value> }`.

The engine itself stores and computes in **Hartree atomic units**. The
loader converts every unit-bearing scalar — both typed fields and the
unit-bearing fields of open-shaped slot `params` (e.g.
`[phase.thermostat]` `temperature` and `tau`) — from the user's chosen
system into atomic units before populating the `Config` struct. All
downstream consumers therefore see atomic-unit values: length in Bohr
radii, mass in electron masses, time in atomic time units (`hbar /
E_h`), energy in Hartrees, temperature in `E_h / k_B` (i.e. each
"temperature" value is `k_B · T` in atomic units), pressure in
`E_h / a_0^3`, charge in elementary charges, velocity in
`a_0 / (hbar / E_h)`. The chosen unit system is preserved on the
returned `Config` as the `units: UnitSystem` field; the runner threads
it to every output writer so the user's view stays consistent
end-to-end.

Validation runs on the post-conversion (atomic-unit) values; range and
ordering checks are preserved by the strictly-positive conversion
factors, so the same checks fire on the same physical inputs
regardless of the chosen unit system.

Defaults the loader computes from other fields (e.g.
`r_switch = 0.9 * cutoff`, `r_skin = 0.3 * max_cutoff`,
`[output]` paths) inherit their dimension from the field they default
off, so the defaults end up atomic-unit-valued without any extra
conversion logic.

The full mechanism — accepted values, conversion factors, slot-kind
field-to-dimension table, output-writer conversion — is documented in
`unit-system.md`. The `.in.xyz` side is documented in
`init-state-file.md`; the trajectory- and CSV-log output sides in
`trajectory-output.md` and `log-output.md` respectively.

### Field reference <!-- rq-e367855a -->

Unit-bearing scalars below name their SI form for clarity (e.g. "in
metres", "in joules"). The `units` selector at the top of the file
controls whether the user writes a given field in SI or in Hartree
atomic units; the loader converts every value to atomic units before
storing it on the returned `Config`. The corresponding atomic-unit
dimension for each named SI unit is fixed (length → Bohr,
mass → electron mass, time → atomic time, energy → Hartree,
temperature → `E_h / k_B`, pressure → `E_h / a_0^3`,
charge → elementary charge, velocity → `a_0 / (hbar / E_h)`); the
mapping is enumerated in `unit-system.md`. Fields are validated on
their post-conversion (atomic-unit) values, but the same positivity
and ordering invariants hold regardless of the chosen unit system.

#### Top level <!-- rq-4c42a952 -->

- `schema_version: u64` — must equal `1`. See *Schema version handling* below.
- `units: String` — optional. Selects the unit system the file is
  written in. Accepted values are `"si"` (the default; equivalent to
  omitting the field) and `"atomic"` (Hartree atomic units). Any other
  string is rejected with `ConfigError::UnknownUnits { got }`.
  Comparison is case-sensitive. See *Units* above and
  `unit-system.md` for the full mechanism.
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

The `[simulation]` table carries the settings that apply to the
**initial-velocity sampling** performed once at phase-0 entry plus the
run-wide CUDA graph execution knobs. Per-step settings (timestep, step
count, integrator/thermostat/barostat composition, output cadences)
live in the `[[phase]]` array.

- `seed: u64` — RNG seed used for Maxwell-Boltzmann velocity
  generation. Required even when the init file supplies explicit
  velocities (still validated, even though unused). No default.
- `temperature: f64` — initial-velocity temperature, expressed in the
  unit system selected by the top-level `units` field (kelvin in `si`
  mode, `E_h / k_B` in `atomic` mode). Required. The loader converts
  the value to atomic units (`k_B · T` in Hartrees) before storing it
  on the returned `Config`. Used to initialise velocities at the
  Maxwell-Boltzmann distribution when the init file's `Properties`
  lacks a `velo:R:3` field; ignored (but still required and
  validated) when the init file supplies velocities. Must be finite
  and `>= 0.0`. The thermostat's bath temperature is a separate
  per-phase field under `[phase.thermostat]`.
- `graph_batch_size: u32` — optional. Number of step replays between
  displacement checks and output-cadence re-evaluations when an MD
  phase runs under CUDA graph mode. Default `5`. Must be `>= 1`; the
  loader rejects `0` with
  `ConfigError::InvalidValue { field: "simulation.graph_batch_size",
  reason: "value must be >= 1, got 0" }`. See `cuda-graphs.md` for
  the activation policy and the skin-distance tuning constraint.
- `cuda_graphs_disable: bool` — optional. When `true`, every MD
  phase runs the per-step launch loop with full per-kernel
  `Timings`. Default `false`. Provided as a diagnostic escape hatch;
  see `cuda-graphs.md`.

#### `[[phase]]` (array of tables, >= 1 entry) <!-- rq-18441e33 -->

A simulation is a sequence of one or more phases, declared as a TOML
array of tables. Particle state (positions, velocities, simulation
box, particle types, charges) carries over between phases; slot
state (integrator buffers, thermostat chain variables, barostat
piston, c-rescale conserved quantity) is reset at every phase
boundary. See `simulation-runner.md` for the runtime model.

Each `[[phase]]` entry carries:

- `name: String` — required. Identifier used to derive output
  filenames (`<root>.out.<name>.{xyz,log,timings}`). Non-empty,
  case-sensitive, must contain only ASCII letters, digits, `-`, and
  `_`. Phase names must be unique within the config; duplicates
  surface as `ConfigError::DuplicatePhaseName { name }`.
- `n_steps: u64` — required. Number of integration steps the phase
  executes. `0` is permitted (the phase writes its initial-state
  snapshot to its trajectory/log, then immediately advances to the
  next phase or exits).
- `dt: f64` — required. Integration timestep for this phase, expressed
  in the unit system selected by the top-level `units` field (seconds
  in `si` mode, atomic time units `hbar / E_h` in `atomic` mode). The
  loader converts the value to atomic time units before storing it on
  the returned `Config`. Must be finite and strictly positive.
- `[phase.integrator]` — required. Selects the per-phase integrator
  slot and its kind-specific parameters; see *`[phase.integrator]`*
  below for the field reference (the schema is identical to the
  former top-level `[integrator]`).
- `[phase.thermostat]` — optional. Per-phase thermostat slot; see
  *`[phase.thermostat]`* below.
- `[phase.barostat]` — optional. Per-phase barostat slot; see
  *`[phase.barostat]`* below.
- `[phase.output]` — optional. Per-phase output cadences and paths;
  see *`[phase.output]`* below. When omitted, the phase uses the
  default cadences (`trajectory_every = 100`, `log_every = 100`,
  `include_velocities = true`, `include_images = true`) and the
  default file paths (`<root>.out.<name>.{xyz,log,timings}`).

Seed uniqueness: every stochastic slot (`csvr`, `andersen`,
`c-rescale`, `langevin-baoab` integrator, etc.) carries its own
required `seed: u64`. Across the whole config, no two slots of the
**same `kind`** may declare the same `seed`; duplicates surface as
`ConfigError::DuplicatePhaseSeed { kind, seed }`. The check is
across all phases, since a duplicate seed in two phases of the same
kind would produce correlated noise sequences at the same dynamical
times. Two slots of *different* kinds may safely share a numerical
seed value (their RNG streams are independent).

The integrator–thermostat compatibility, integrator–barostat
compatibility, and integrator–constraint compatibility rules
documented under *Validation* are checked **per phase**: every phase
must individually satisfy the rules. A single
`[[constraint_types]]`/topology block applies to every phase (the
constraint algorithm cannot be enabled or disabled mid-protocol);
each phase's integrator must therefore be able to drive the
configured constraints.

#### `[phase.integrator]` <!-- rq-27f9fae8 -->

The integrator section carries a required `kind` field that selects
one of the registered integrator slots (see `integration/framework.md`
for the slot interface) plus any number of kind-specific parameter
fields at the same level. The Rust-side deserialiser captures `kind`
into a `SlotConfig` (see *Feature API* below) and flattens the rest of
the section into a `toml::Value` for the builder to consume. Each
registered builder validates its own parameter schema:
fields not recognised by the chosen `kind`'s builder are rejected at
config load via the builder's `validate_params(&toml::Value)` method,
and missing required fields are rejected the same way.

- `kind: String` — required. One of:
  - `"velocity-verlet"` — symplectic NVE time-stepping core. Does not
    own a thermostat; compose with `[phase.thermostat]` for NVT. See
    `integration/velocity-verlet.md`.
  - `"langevin-baoab"` — stochastic NVT via the Leimkuhler-Matthews
    BAOAB splitting. Owns its own thermostat (the OU step);
    incompatible with `[phase.thermostat]`. See
    `integration/langevin-baoab.md`.
  - `"mtk-npt"` — deterministic extended-system NPT via the
    Martyna-Tobias-Klein integrator (isotropic). Owns both its
    thermostat (Nosé-Hoover chains on particles and cell) and its
    barostat (extended-system cell); incompatible with both
    `[phase.thermostat]` and `[phase.barostat]`. See
    `integration/mtk-npt.md`.

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

#### `[phase.thermostat]` (optional) <!-- rq-ee10237d -->

The per-phase thermostat section is optional. When omitted, the phase
runs no thermostat (NVE composition with the integrator, or whatever
fused thermostat the integrator owns). When present, a required `kind`
field selects one of the registered thermostat slots (see
`integration/framework.md`) and the kind-specific parameter fields
sit at the same level. The Rust-side deserialiser captures `kind`
into a `SlotConfig` and flattens the rest of the section into a
`toml::Value`; the chosen builder's `validate_params(&toml::Value)`
enforces required fields, domains, and rejects unknown fields.

Configuring `[phase.thermostat]` alongside an integrator whose
builder's `owns_thermostat(&params)` returns `true` (currently the
`langevin-baoab` and `mtk-npt` builders) is rejected at config-load
time with
`ConfigError::IncompatibleThermostat { integrator: <kind name>, phase: <phase name> }`.

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

#### `[phase.barostat]` (optional) <!-- rq-cc557e8c -->

The per-phase barostat section is optional. When omitted, the phase
runs no barostat (constant-volume composition). When present, a
required `kind` field selects one of the registered barostat slots
(see `integration/framework.md`) and the kind-specific parameter
fields sit at the same level. The Rust-side deserialiser captures
`kind` into a `SlotConfig` and flattens the rest of the section into
a `toml::Value`; the chosen builder's `validate_params(&toml::Value)`
enforces required fields, domains, and rejects unknown fields.

Configuring `[phase.barostat]` alongside an integrator whose builder's
`owns_barostat(&params)` returns `true` (currently the `mtk-npt`
builder) is rejected at config-load time with
`ConfigError::IncompatibleBarostat { integrator: <kind name>, phase: <phase name> }`.

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
  of `simulation.temperature` and of any
  `[phase.thermostat].temperature`; the framework performs no
  cross-slot validation. For canonical NPT sampling the user must
  keep this value consistent with the thermostat (or, for
  `langevin-baoab`, with the integrator's bath temperature).
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

#### `[[minimization]]` (array of tables) <!-- rq-0132d7a5 -->

Each `[[minimization]]` table declares a minimization phase. The full
schema, parameter defaults, convergence criteria, and the
`[minimization.output]` sub-table are documented in
`rqm/minimization/steepest-descent.md`'s *Schema* section. Summary:

- `name: String` — required. Identifier used to derive output
  filenames (`<root>.out.<name>.{minlog,xyz,timings}`). Same
  character set and uniqueness rules as `[[phase]]`'s `name`.
  Uniqueness is enforced across the union of `[[phase]]` and
  `[[minimization]]` names; collisions surface as
  `ConfigError::DuplicatePhaseName { name }`.
- `[minimization.algorithm]` — required. Carries a `kind` field
  (`"steepest-descent"` in v1) plus algorithm-specific parameters.
  Dispatched through the `MinimizerRegistry`. Unknown kinds surface
  as `ConfigError::UnknownKind { slot: "minimization", kind }`.
- `[minimization.output]` — optional. Controls `.minlog`, optional
  `.xyz`, and `.timings` cadences and paths. Defaults: `minlog_every
  = 1`, `trajectory_every = 0`, `include_images = true`.

A `[[minimization]]` entry rejects any of the following keys at
deserialisation time: `n_steps`, `dt`, `[minimization.integrator]`,
`[minimization.thermostat]`, `[minimization.barostat]`. Those
concepts belong to MD phases only.

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

Each array entry deserialises into a `NamedSlotConfig` (see *Feature
API* below). Two fields are common to every entry; everything else is
captured into a `toml::Value` `params` field for the chosen
algorithm's builder to consume:

- `name: String` — unique identifier within the `[[constraint_types]]`
  array. Empty strings are rejected. Case-sensitive.
- `kind: String` — selects the algorithm that processes any group
  declared with this type. The currently registered value is
  `"shake"` (see `integration/shake.md`). Future values
  (`"settle"`, `"m-shake"`, `"lincs"`, ...) are added by registering
  additional `ConstraintBuilder`s.

Per-kind parameter fields are validated by the matching
`ConstraintBuilder`'s `validate_params(&toml::Value)` method (see
`integration/constraint-framework.md`). For `kind = "shake"`
(documented in `integration/shake.md`):

- `atoms: u32` — number of atoms in every group of this type. Required.
  Strictly positive; at most `MAX_GROUP_ATOMS = 8`.
- `constraints: Vec<{ i: u32, j: u32, d: f64 }>` — one entry per
  pair-distance constraint inside the group. Required. The list must
  be non-empty and at most `MAX_GROUP_CONSTRAINTS = 12` entries long.
  - `i` and `j` are local atom indices in `0..atoms`; the pair
    `(min(i, j), max(i, j))` must be unique across constraints; `i`
    and `j` must differ.
  - `d` is the target distance in metres. Finite and strictly positive.

Names must be unique within the array. Unknown parameter fields for
the chosen `kind` are rejected by the matching builder's
`validate_params`.

The presence of at least one constraint group in the topology file
triggers construction of the `Constraint` slot (see
`integration/constraint-framework.md`) and is incompatible with any
integrator (in any phase) whose builder's
`IntegratorBuilder::supports_constraints(&params)` returns `false`.
The runner enforces this **per phase** (see *Validation*) and surfaces
failures as
`ConfigError::IncompatibleConstraint { integrator: <kind name>, phase: <phase name> }`.

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

#### `[phase.output]` (optional table; all fields have defaults) <!-- rq-3ea7fedd -->

Every `[[phase]]` entry accepts an optional `[phase.output]` sub-table
controlling that phase's trajectory, log, and timings outputs. Each
field defaults independently when omitted; the table itself is
optional. Output paths default to
`<config-root>.out.<phase-name>.{xyz,log,timings}` (one file per
phase per output kind).

- `trajectory_path: String` — output trajectory path for this phase.
  Default: `<config-root>.out.<phase-name>.xyz` in the same directory
  as the config file (e.g. config root `argon`, phase name `equil` →
  `argon.out.equil.xyz`). `<config-root>` is the config-filename
  derivation defined under *Config filename convention*. Resolved
  relative to the config file's directory; absolute paths are
  honored as-is.
- `trajectory_every: u64` — write one trajectory frame every this many
  integration steps **within this phase**. Default `100`. `0` disables
  trajectory output for this phase entirely (not even the phase's
  step-0 frame is written; no `<phase-name>.xyz` file is created).
  TOML parses `u64` fields, so negative integers fail at TOML parse
  time.
- `include_velocities: bool` — include `velo:R:3` columns in this
  phase's trajectory frames. Default `true`.
- `include_images: bool` — include `image:I:3` columns in this
  phase's trajectory frames. Default `true`. When `true`, each
  frame's data rows carry three integer columns after the position
  (and velocity, if present) columns; consumers reconstruct unwrapped
  positions as `pos + images_x · a + images_y · b + images_z · c`,
  which reduces to `pos + image · (lx, ly, lz)` for an orthorhombic
  box. When `false`, image columns are omitted from the file;
  positions in the trajectory are still wrapped into the primary
  image.
- `log_path: String` — output log path for this phase. Default:
  `<config-root>.out.<phase-name>.log` in the same directory as the
  config file. Resolved like `trajectory_path`.
- `log_every: u64` — write one log row every this many integration
  steps within this phase. Default `100`. `0` disables the log
  entirely for this phase (no header is written, no file is created).
- `timings_path: String` — output path for this phase's per-stage
  performance summary file. Default:
  `<config-root>.out.<phase-name>.timings` in the same directory as
  the config file. Resolved like `trajectory_path`. See
  `performance-analysis.md` for the file format. There is no
  `timings_every` field; one timings file is written per phase at
  end of phase.

Phase-local step numbering: within a single phase's output files,
the trajectory `Step=` attribute and the log `step` column start at
`0` (the phase's initial-state snapshot) and run through `n_steps`.
Step numbering does **not** accumulate across phases — each phase's
output file is structurally identical to a single-phase trajectory
or log of the same `n_steps`. Phase-local `time` is computed as
`step * dt_phase` using that phase's `dt`.

### Config filename convention <!-- rq-5a0f5c00 -->

The config-file path passed to `load_config` (and `load_config_raw`)
must end in `.in.toml` (the suffix match is case-sensitive on the
whole `.in.toml` string). The runner inspects the path's final
filename component and rejects any name that does not have this
suffix with `ConfigError::InvalidConfigFilename { path }`. The rule is
a load-time check on the path string itself; the file is not opened
when it fails. This makes the in/out file-naming pairing visible at
every `ls` of a simulation directory and lets the loader derive every
output default from the config filename without an extra config field.

The `<config-root>` referenced by the `[output]` defaults is derived
as:

```
1. Take the path's final filename component.
2. Strip the trailing `.toml` suffix.
3. Strip one trailing `.in` suffix (and only one).
```

Examples:

| Config filename       | `<config-root>` | Default trajectory path |
| --------------------- | --------------- | ----------------------- |
| `argon.in.toml`       | `argon`         | `argon.out.xyz`         |
| `spc.in.toml`         | `spc`           | `spc.out.xyz`           |
| `run-01.in.toml`      | `run-01`        | `run-01.out.xyz`        |
| `foo.in.in.toml`      | `foo.in`        | `foo.in.out.xyz`        |

The strip is case-sensitive on the `.in` segment: a filename ending in
`.IN.toml` does not satisfy the suffix rule and is rejected at the
filename-convention check. Filenames whose `<config-root>` derivation
would yield the empty string (e.g. `.in.toml` itself) are likewise
rejected with `ConfigError::InvalidConfigFilename { path }`.

The init-file path, the topology-file path, and any explicit
`output.*_path` field are not subject to the filename convention; they
are arbitrary user-supplied paths.

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

### Validation <!-- rq-bd228ef7 -->

Config validation is split between two methods. `Config::validate(&self)`
runs the structural and topology-independent checks; the open registries
do not need to be available for it. `Config::validate_against(&self,
registries: &Registries)` runs the per-kind and cross-cutting checks
that consult the registered builders. `Registries` lives at
`heddle_md::Registries` (see `simulation-runner.md`) and bundles the
five open registries the runner consults (integrators, thermostats,
barostats, constraint_types, potentials).

`load_config` invokes both methods in order, against
`Registries::with_builtins()`, so the common path surfaces every
per-kind validation error at parse time without callers having to
remember the second step. `load_config_raw` is the parse-only entry
point: it runs `Config::validate` and stops. Callers that compose
custom builders use `load_config_raw` followed by
`config.validate_against(&registries)` with their own bundle.

`load_config_raw` performs one pre-deserialiser check on the input
path:

0. The path's final filename component, lower-cased, ends in
   `.in.toml`, and the `<config-root>` derivation (see *Config
   filename convention*) yields a non-empty string. A failure of
   either part returns
   `ConfigError::InvalidConfigFilename { path: PathBuf }` and the
   file is not opened. This check runs before
   `schema_version` validation.

`Config::validate(&self)` checks:

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
3. The merged phase sequence (`[[phase]]` ∪ `[[minimization]]`) is
   non-empty (`EmptyPhases` otherwise).
4. Every phase (MD or minimization) has a non-empty ASCII-only `name`
   (letters/digits/`-`/`_`); names are unique across the merged
   sequence (`DuplicatePhaseName { name }` otherwise).
5. Every MD phase's `n_steps` is a `u64` (TOML-enforced) and every
   MD phase's `dt` is finite and strictly positive
   (`InvalidValue { field: "phase[<i>].dt", reason }` otherwise).
6. After path resolution, every supplied path is pairwise distinct from
   every other supplied path (`PathCollision { kind_a, kind_b, path }`).
   The set of paths under check is `init`, every MD phase's
   `output.trajectory_path` / `output.log_path` /
   `output.timings_path`, every minimization phase's
   `output.minlog_path` / `output.trajectory_path` /
   `output.timings_path`, and (when supplied) `topology`.
7. Across every stochastic slot in every phase, no two slots of the
   same `kind` declare the same numerical `seed`
   (`DuplicatePhaseSeed { kind: String, seed: u64 }` otherwise). Two
   slots of different `kind`s may share a numerical seed.
8. Every `[[constraint_types]]` entry has a unique `name`; duplicates
   surface as `DuplicateConstraintTypeName { name }`.
9. The electrostatics tables are mutually exclusive: declaring both
   `[coulomb]` and `[spme]` surfaces as `ConflictingElectrostatics`.

`Config::validate_against(&self, registries: &Registries)` checks,
per phase, in declaration order:

10. The `[phase.integrator]` section's `kind` is a registered key of
    `registries.integrators`. Unknown kinds surface as
    `ConfigError::UnknownKind { slot: "integrator", kind }`.
11. The chosen integrator builder's
    `validate_params(&phase.integrator.params)` succeeds; otherwise
    the builder-produced `ConfigError` propagates.
12. Same as (10) and (11) for `[phase.thermostat]` (when present)
    against `registries.thermostats`, and for `[phase.barostat]`
    (when present) against `registries.barostats`.
12a. For every `PhaseKind::Minimization` entry: the
    `[minimization.algorithm]` section's `kind` is a registered key
    of `registries.minimizers`. Unknown kinds surface as
    `ConfigError::UnknownKind { slot: "minimization", kind }`. The
    chosen builder's
    `validate_params(&minimization.algorithm.params)` runs and any
    failure propagates.
13. Same as (10) and (11) for every `[[constraint_types]]` entry
    against `registries.constraint_types` (checked once, not per
    phase — constraint types are global).
14. If `[phase.thermostat]` is present and
    `registries.integrators.lookup(&phase.integrator.kind).unwrap()
    .owns_thermostat(&phase.integrator.params)` returns `true`,
    validation returns `IncompatibleThermostat { integrator: <kind name>,
    phase: <phase name> }`.
15. If `[phase.barostat]` is present and
    `registries.integrators.lookup(&phase.integrator.kind).unwrap()
    .owns_barostat(&phase.integrator.params)` returns `true`,
    validation returns `IncompatibleBarostat { integrator: <kind name>,
    phase: <phase name> }`.
16. Every constraint type name referenced from the topology file's
    `[constraints]` section appears in `[[constraint_types]]`.
    Unknown names surface through `load_topology_file` as
    `TopologyFileError::UnknownConstraintType { .. }`. This check is
    performed by `load_topology_file`, not by `Config::validate_against`,
    because the topology file is loaded separately.

The runner additionally calls
`Config::validate_constraint_compatibility(&self, registries: &Registries, has_constraints: bool)`
after `load_topology_file` returns. This method runs the
`IncompatibleConstraint` check **per phase**:

- Every MD phase's chosen integrator builder must satisfy
  `IntegratorBuilder::supports_constraints(&params)` when the
  topology declares any constraint group. In the default registry
  this rejects every combination of a non-empty `[constraints]`
  section with `langevin-baoab`, `mtk-npt`, or `velocity-verlet`'s
  lossless variant.
- Every minimization phase requires that every registered
  `[[constraint_types]]` entry's builder satisfies
  `ConstraintBuilder::supports_position_projection_only(&params)`.
  In the default registry the `shake` builder returns `true`, so the
  current registry never rejects on this path; future constraint
  algorithms that cannot project positions in isolation flip the
  predicate to `false` and surface here.

Failures of either form surface as
`IncompatibleConstraint { integrator: <kind-or-algorithm name>,
phase: <phase name> }`. The `integrator` field carries the
integrator's `kind` for MD-phase failures and the minimization
algorithm's `kind` (e.g. `"steepest-descent"`) for minimization-
phase failures.

## Feature API <!-- rq-110285ae -->

### Types <!-- rq-b719c42c -->

- `Config` — parsed configuration. All fields are `pub`; field names match <!-- rq-2a6a51c8 -->
  TOML keys directly (snake_case).

  Fields:
  - `schema_version: u64`
  - `units: UnitSystem` — the unit system the source TOML, the
    referenced `.in.xyz` file, and every output file the run produces
    are written in. The loader has already converted every
    unit-bearing value to Hartree atomic units in the returned
    `Config`; this field is the selector the runner threads to every
    output writer so trajectory, log, and minlog files come back in
    the same system the user authored. See `unit-system.md`.
  - `init: PathBuf` — resolved against the config file's directory.
  - `topology: Option<PathBuf>` — `Some(_)` when the optional top-level
    `topology` field is present; resolved against the config file's
    directory.
  - `simulation: SimulationConfig`
  - `phases: Vec<PhaseKind>` — one entry per `[[phase]]` or
    `[[minimization]]` table in **source-document order** across
    both arrays (merged by byte-span comparison; see *Schema*
    above). Non-empty (enforced by `Config::validate`).
    `PhaseKind::Md(PhaseConfig)` for `[[phase]]` entries;
    `PhaseKind::Minimization(MinimizationConfig)` for
    `[[minimization]]` entries. The `MinimizationConfig` type is
    documented in `rqm/minimization/steepest-descent.md`.
  - `particle_types: Vec<ParticleTypeConfig>`
  - `pair_interactions: Vec<PairInteractionConfig>`
  - `bond_types: Vec<BondTypeConfig>` — empty when the `[[bond_types]]`
    array is absent.
  - `angle_types: Vec<AngleTypeConfig>` — empty when the
    `[[angle_types]]` array is absent.
  - `constraint_types: Vec<NamedSlotConfig>` — empty when the
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
  - `config_path: PathBuf` — the absolute path of the source config file,
    retained for error messages and default output-path derivation.

- `SimulationConfig` <!-- rq-53055a5b -->
  - `seed: u64`
  - `temperature: f64`

- `PhaseConfig` — parsed `[[phase]]` (MD) entry. <!-- rq-f1c04d3b -->
  - `name: String`
  - `n_steps: u64`
  - `dt: f64`
  - `integrator: SlotConfig` — the chosen integrator kind plus its
    raw `toml::Value` parameters.
  - `thermostat: Option<SlotConfig>` — `Some` when the phase declares
    `[phase.thermostat]`, `None` otherwise.
  - `barostat: Option<SlotConfig>` — `Some` when the phase declares
    `[phase.barostat]`, `None` otherwise.
  - `output: OutputConfig` — resolved per-phase trajectory/log/timings
    paths and cadences, with defaults derived from
    `<config-root>.out.<phase-name>.{xyz,log,timings}` when omitted.

- `PhaseKind` — discriminated union over the unified phase sequence. <!-- rq-19226daf -->
  ```rust
  pub enum PhaseKind {
      Md(PhaseConfig),
      Minimization(MinimizationConfig),
  }
  ```
  `MinimizationConfig` is documented in
  `rqm/minimization/steepest-descent.md`. `Config.phases` is a
  `Vec<PhaseKind>` in source-document order across the two arrays.

- `SlotConfig` — open-shaped parsed selection for a per-phase <!-- rq-661bf664 -->
  `[phase.integrator]`, `[phase.thermostat]`, or `[phase.barostat]`
  section.

  ```rust
  pub struct SlotConfig {
      pub kind: String,
      pub params: toml::Value,
  }
  ```

  - `kind` carries the user-supplied `kind = "..."` field verbatim
    (case-sensitive). It is the lookup key the runner uses against
    the corresponding registry (`integration/framework.md`).
  - `params` carries every other field of the section flattened into
    a `toml::Value` (via `#[serde(flatten)]`). The framework never
    inspects `params`; each registered builder deserialises its own
    typed parameter struct from it via `validate_params(&toml::Value)`
    and `build(...)`.

- `NamedSlotConfig` — open-shaped parsed entry for the <!-- rq-3fdb7e01 -->
  `[[constraint_types]]` array (and any future similarly-shaped array
  of named, kind-tagged entries).

  ```rust
  pub struct NamedSlotConfig {
      pub name: String,
      pub kind: String,
      pub params: toml::Value,
  }
  ```

  - `name` is the user-facing identifier referenced from elsewhere in
    the config (for `[[constraint_types]]`, from the topology file's
    `[constraints]` section).
  - `kind` and `params` work the same as `SlotConfig`'s fields; the
    matching builder (`ConstraintBuilder` for constraint types)
    deserialises and validates `params`.

  See `integration/framework.md` for the integrator / thermostat /
  barostat builder traits and `integration/constraint-framework.md`
  for the `ConstraintBuilder` trait; each defines the
  `validate_params`, `build`, and (where relevant) compatibility
  predicate methods that consume `SlotConfig`'s and
  `NamedSlotConfig`'s `params`.

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

- `[[constraint_types]]` entries deserialise into `NamedSlotConfig` <!-- rq-ac8fc96a -->
  values (see `NamedSlotConfig` above). Each entry's `kind` selects
  the algorithm; each registered `ConstraintBuilder` (see
  `integration/constraint-framework.md`) deserialises its own typed
  parameter struct from `params` via `validate_params(&toml::Value)`
  and answers `expected_atom_count(&toml::Value)` for the topology
  parser's row-column-count check. For
  `kind = "shake"`, the builder's typed params are
  `ShakeParams { atoms: u32, constraints: Vec<{ i: u32, j: u32, d: f64 }> }`
  (see `integration/shake.md`).

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

- `PathRole` — `enum`. Used in `PathCollision`. Variants: <!-- rq-f0084057 -->
  - `Init`
  - `Topology`
  - `PhaseTrajectory { phase: String }`
  - `PhaseLog { phase: String }`
  - `PhaseTimings { phase: String }`
  - `MinimizationMinlog { phase: String }`
  - `MinimizationTrajectory { phase: String }`
  - `MinimizationTimings { phase: String }`

- `ConfigError` — error type returned by `load_config` and by <!-- rq-3108381e -->
  `Config::validate`. Variants:
  - `InvalidConfigFilename { path: PathBuf }` — the path passed to
    `load_config` / `load_config_raw` does not satisfy the
    config-filename convention: either the final filename component
    does not end in `.in.toml` (case-sensitive `.in` segment), or the
    derived `<config-root>` is empty (e.g. the filename is `.in.toml`).
    Produced before the file is opened.
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
  - `UnknownUnits { got: String }` — the top-level `units` field is
    present but is not one of the accepted lowercase strings (`"si"`
    or `"atomic"`). Comparison is case-sensitive.
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
    — two supplied file paths resolve to the same location. `PathRole`
    carries variants `Init`, `Topology`, and
    `PhaseTrajectory { phase: String }` /
    `PhaseLog { phase: String }` /
    `PhaseTimings { phase: String }` for per-phase outputs.
  - `ConflictingElectrostatics` — the config declares both `[coulomb]`
    and `[spme]`. Only one electrostatics method may be active per run.
  - `EmptyPhases` — the merged phase sequence
    (`[[phase]]` ∪ `[[minimization]]`) is empty. A simulation
    requires at least one phase of either kind.
  - `DuplicatePhaseName { name: String }` — two phases across the
    merged `[[phase]]` ∪ `[[minimization]]` sequence share a
    `name`. Phase names must be unique because they derive per-phase
    output filenames.
  - `DuplicatePhaseSeed { kind: String, seed: u64 }` — two stochastic
    slots of the same `kind` across the `[[phase]]` array share the
    same numerical `seed`. The check applies across every phase's
    `[phase.thermostat]`, `[phase.barostat]`, and any seeded
    `[phase.integrator]` (e.g. `langevin-baoab`).
  - `IncompatibleThermostat { integrator: String, phase: String }` —
    a phase pairs an integrator that owns its own thermostat
    (`langevin-baoab` or `mtk-npt`) with a `[phase.thermostat]`
    table. `phase` is the offending phase's `name`.
  - `IncompatibleBarostat { integrator: String, phase: String }` —
    a phase pairs an integrator that owns its own barostat
    (currently only `mtk-npt`) with a `[phase.barostat]` table.
  - `DuplicateBondTypeName { name: String }` — two `[[bond_types]]`
    entries share a `name`.
  - `DuplicateAngleTypeName { name: String }` — two `[[angle_types]]`
    entries share a `name`.
  - `DuplicateConstraintTypeName { name: String }` — two
    `[[constraint_types]]` entries share a `name`.
  - `IncompatibleConstraint { integrator: String, phase: String }` —
    the topology file's `[constraints]` section is non-empty and the
    chosen integrator builder's
    `IntegratorBuilder::supports_constraints(&params)` returns
    `false` for the named phase (in the default registry:
    `langevin-baoab`, `mtk-npt`, or `velocity-verlet` with
    `lossless = true`).
  - `UnknownKind { slot: &'static str, kind: String }` — a
    `[integrator]`, `[thermostat]`, `[barostat]`,
    `[[constraint_types]]`, or `[minimization.algorithm]` entry's
    `kind` field does not match any registered builder in the
    corresponding registry. `slot` carries `"integrator"`,
    `"thermostat"`, `"barostat"`, `"constraint_types"`, or
    `"minimization"`.
  - `ShakeParamsMalformed { name: String, reason: String }`
    — a `[[constraint_types]]` entry with `kind = "shake"` violates
    one of the `ShakeParams` constraints documented in
    `integration/shake.md` (`atoms` zero or above the per-group cap,
    `constraints` empty or above the per-group cap, an out-of-range
    `i` or `j`, `i == j`, a duplicated pair, or a non-positive `d`).

  `Parse` covers every shape error the typed deserialiser flags
  structurally that does not require registry knowledge: unknown
  fields under a closed-enum table (e.g. `mode` outside the accepted
  set in `[neighbor_list]`, an unknown `potential` in
  `[[pair_interactions]]`); wrong TOML types (e.g. a string where an
  integer is required); and raw TOML syntax errors. `path` carries the
  document location so callers can present a useful diagnostic without
  inspecting `message`.

  Per-kind parameter shape errors for the open-builder slots
  (`[integrator]`, `[thermostat]`, `[barostat]`,
  `[[constraint_types]]`) — including unknown fields under the chosen
  `kind` and out-of-domain numeric values — surface from the matching
  builder's `validate_params(&toml::Value)` during
  `Config::validate_against(&registries)`. Builders return
  `ConfigError::InvalidValue`, `ConfigError::ShakeParamsMalformed`,
  or `ConfigError::Parse` as appropriate for their parameter shape.

### Functions <!-- rq-39891001 -->

- `load_config(path: &Path) -> Result<Config, ConfigError>` <!-- rq-45bb8194 -->
  - The default loader for callers that use only the built-in slot
    kinds. Equivalent to:
    ```rust
    let config = load_config_raw(path)?;
    config.validate_against(&Registries::with_builtins())?;
    Ok(config)
    ```
  - Any error from `load_config_raw` or from `validate_against`
    propagates unchanged.
  - Callers that register custom builders use `load_config_raw`
    plus `Config::validate_against(&their_registries)` directly,
    so the registry-dispatched validation runs against the right
    builder set.

- `load_config_raw(path: &Path) -> Result<Config, ConfigError>` <!-- rq-deaf8b59 -->
  - Validates the config-filename convention (see *Config filename
    convention*) on `path` before opening the file; failures return
    `ConfigError::InvalidConfigFilename { path }` without any I/O.
  - Reads the file at `path`, runs the typed TOML deserialiser, parses
    the top-level `units` selector (defaulting to `UnitSystem::Si`;
    rejecting any other value with `ConfigError::UnknownUnits`),
    rescales every unit-bearing scalar from the user's chosen system
    into Hartree atomic units as it builds the `Config` (both typed
    fields and the unit-bearing fields of open-shaped slot `params`
    — see `unit-system.md`), fills in field-derived defaults
    (`r_switch = 0.9 * cutoff` for `[[pair_interactions]]` and
    `[coulomb]`, `r_skin = 0.3 * max_cutoff` for the `cell-list`
    `[neighbor_list]` mode, `[output]` defaults derived from
    `<config-root>` per *Config filename convention*), resolves every
    supplied path against `path.parent()` (or `"."` if `path` has no
    parent), calls `Config::validate(&config)` on the resulting
    `Config`, and returns it. Does not run `Config::validate_against`
    — that is the caller's responsibility.
  - File-read failure yields `Io(String)`. Filename-convention failure
    yields `InvalidConfigFilename { path }`. Deserialiser failures
    yield either `MissingField`, `UnsupportedSchemaVersion`, or
    `Parse` depending on the failure kind (see the `ConfigError`
    description above). Range, finiteness, domain, and structural
    cross-validation failures yield the variant emitted by
    `Config::validate`.
  - On any validation failure, returns the first error encountered in
    declaration order: deserialiser errors first (top-level fields,
    then `[simulation]`, then `[integrator]`, then
    `[[particle_types]]`, then `[[pair_interactions]]`, then
    `[bond_types]` / `[angle_types]` / `[coulomb]` / `[spme]` /
    `[neighbor_list]` / `[output]`); then `Config::validate`'s
    field-domain checks in the same declaration order; then
    `Config::validate`'s structural cross-validation rules in the
    order listed under *Validation*.

- `Config::validate(&self) -> Result<(), ConfigError>` <!-- rq-a54cc657 -->
  - Pure host-side function. Takes a `Config` (typically obtained from
    `load_config` after deserialisation, but also constructable in
    memory by callers) and applies every structural check documented
    under *Field reference* that is not already enforceable by the
    typed deserialiser and that does not require registry knowledge:
    range/finiteness/domain constraints on numeric fields (positivity,
    NaN-rejection, `theta_0 in [0, π]`, `r_switch <= cutoff`, non-empty
    string identifiers), plus the structural cross-validation rules
    listed under *Validation*.
  - Per-kind parameter validation for the open-builder slots
    (`[integrator]`, `[thermostat]`, `[barostat]`,
    `[[constraint_types]]`) lives in `Config::validate_against`
    because it requires registry access.
  - On the first failure, returns the structured error variant
    documented for that check (`InvalidValue` for per-field domain
    failures; `DuplicateTypeName`, `UnknownTypeInPair`,
    `MissingPairInteraction`, `DuplicatePairInteraction`,
    `PathCollision`, `ConflictingElectrostatics`,
    `IncompatibleThermostat`, `IncompatibleBarostat`,
    `DuplicateBondTypeName`, or `DuplicateAngleTypeName` for the
    cross-validation rules).
  - Order of checks: per-field domain checks in the section order
    above, then the structural cross-validation rules listed under
    *Validation* in that order.
  - Calling `Config::validate` on a `Config` returned by
    `load_config` is a no-op (returns `Ok(())`): `load_config` already
    invoked it.

- `Config::validate_against(&self, registries: &Registries) -> Result<(), ConfigError>` <!-- rq-6082cd2d -->
  - Runs the registry-dispatched per-kind validation: looks up each
    open-builder slot's `kind` in the corresponding registry, calls
    the builder's `validate_params(&toml::Value)`, and then queries
    `owns_thermostat(&params)` / `owns_barostat(&params)` to enforce
    the integrator-thermostat and integrator-barostat compatibility
    rules.
  - Returns `ConfigError::UnknownKind { slot, kind }` for any
    `kind` that does not match a registered builder.
  - Surfaces every builder-produced `ConfigError` (typically
    `InvalidValue`, `ShakeParamsMalformed`, or `Parse`).
  - Runs the `[thermostat]` integrator-ownership check and the
    `[barostat]` integrator-ownership check as documented in
    *Validation*; these produce `IncompatibleThermostat` and
    `IncompatibleBarostat` respectively.
  - The runner invokes this method after `load_config` returns and
    after constructing its registries; the constraint compatibility
    check (which also requires the topology file's
    `[constraints]` count) lives in
    `Config::validate_constraint_compatibility`.

- `Config::validate_constraint_compatibility(&self, registries: &Registries, has_constraints: bool) -> Result<(), ConfigError>` <!-- rq-723d202b -->
  - When `has_constraints` is `true` and the chosen integrator
    builder's `supports_constraints(&integrator.params)` returns
    `false`, returns
    `ConfigError::IncompatibleConstraint { integrator: <kind name> }`.
  - When `has_constraints` is `false`, always returns `Ok(())`.
  - The runner calls this method after `load_topology_file` returns
    so the topology's `[constraints]` count is known.

- `Registries` — host-side bundle of the four open registries the <!-- rq-32308250 -->
  config-validation API consults.

  ```rust
  pub struct Registries {
      pub integrators: IntegratorRegistry,
      pub thermostats: ThermostatRegistry,
      pub barostats: BarostatRegistry,
      pub constraint_types: ConstraintRegistry,
  }
  ```

  Each field is the open registry documented in its own requirements
  file (`integration/framework.md` for the first three;
  `integration/constraint-framework.md` for the constraint registry).
  `Registries::with_builtins()` constructs a `Registries` whose four
  sub-registries are each `*Registry::with_builtins()` — the default
  set the runner uses.

## Out of Scope <!-- rq-35722a66 -->

- Per-quantity unit suffixes (e.g. `cutoff = "1.0 nm"`). Numeric values
  are bare floats interpreted under the file-level `units` selector
  (see *Units* above and `unit-system.md`).
- Reduced-unit systems other than the two named in `unit-system.md`
  (e.g. LJ-reduced, GROMACS nm/ps, LAMMPS real/metal).
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
    Given a valid minimal config containing schema_version = 1, init = "argon.in.xyz",
      one [simulation] section with seed=12345, temperature=300.0,
      one [[phase]] entry with name="run", n_steps=10, dt=1.0e-15,
        [phase.integrator] kind="velocity-verlet" and lossless=false,
      one [[particle_types]] entry with name="Ar" and mass=6.6335e-26,
      one [[pair_interactions]] entry between=["Ar","Ar"], potential="lennard-jones",
        sigma=3.40e-10, epsilon=1.65e-21, cutoff=1.0e-9

  # --- Happy path ---

  @rq-7df1515f
  Scenario: Load a valid minimal config
    Given a config file written to "/tmp/sim/argon.in.toml" containing the Background
    When load_config("/tmp/sim/argon.in.toml") is called
    Then it returns Ok(config)
    And config.schema_version equals 1
    And config.simulation.seed equals 12345
    And config.phases[0].n_steps equals 10
    And config.phases[0].dt equals 1.0e-15
    And config.simulation.temperature equals 300.0
    And config.phases[0].integrator matches IntegratorKind::VelocityVerlet { lossless: false }
    And config.particle_types has length 1
    And config.particle_types[0].name equals "Ar"
    And config.particle_types[0].mass equals 6.6335e-26
    And config.pair_interactions has length 1
    And config.pair_interactions[0].between equals ("Ar", "Ar")
    And config.pair_interactions[0].cutoff equals 1.0e-9
    And config.pair_interactions[0].potential matches PairPotentialParams::LennardJones { sigma: 3.40e-10, epsilon: 1.65e-21 }
    And config.init equals "/tmp/sim/argon.in.xyz"
    And config.config_path equals "/tmp/sim/argon.in.toml"

  @rq-894c16c4
  Scenario: Defaults populate the output section when [output] is omitted
    Given the Background config with no [output] section, written to "/tmp/sim/argon.in.toml"
    When load_config("/tmp/sim/argon.in.toml") is called
    Then config.phases[0].output.trajectory_path equals "/tmp/sim/argon.out.xyz"
    And config.phases[0].output.trajectory_every equals 100
    And config.phases[0].output.include_velocities equals true
    And config.phases[0].output.include_images equals true
    And config.phases[0].output.log_path equals "/tmp/sim/argon.out.log"
    And config.phases[0].output.log_every equals 100
    And config.phases[0].output.timings_path equals "/tmp/sim/argon.out.timings"

  @rq-0622d4b0
  Scenario: Default output paths drop a single trailing `.in` from the config-file root
    Given the Background config with no [output] section,
      written to "/tmp/sim/foo.in.in.toml"
    When load_config("/tmp/sim/foo.in.in.toml") is called
    Then config.phases[0].output.trajectory_path equals "/tmp/sim/foo.in.out.xyz"
    And config.phases[0].output.log_path equals "/tmp/sim/foo.in.out.log"
    And config.phases[0].output.timings_path equals "/tmp/sim/foo.in.out.timings"

  @rq-d148149f
  Scenario: Explicit [output] values override defaults
    Given the Background config plus [output] with trajectory_every=50, log_every=25,
      trajectory_path="custom-traj.xyz", log_path="custom.log", include_velocities=false,
      written to "/tmp/sim/argon.in.toml"
    When load_config("/tmp/sim/argon.in.toml") is called
    Then config.phases[0].output.trajectory_path equals "/tmp/sim/custom-traj.xyz"
    And config.phases[0].output.trajectory_every equals 50
    And config.phases[0].output.include_velocities equals false
    And config.phases[0].output.log_path equals "/tmp/sim/custom.log"
    And config.phases[0].output.log_every equals 25

  @rq-5ded1806
  Scenario: Absolute paths are honored unchanged
    Given the Background config with init="/data/argon.in.xyz",
      [output].trajectory_path="/data/out/traj.xyz",
      [output].log_path="/data/out/run.log"
    When load_config is called
    Then config.init equals "/data/argon.in.xyz"
    And config.phases[0].output.trajectory_path equals "/data/out/traj.xyz"
    And config.phases[0].output.log_path equals "/data/out/run.log"

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
    And config.phases[0].n_steps equals 0

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
    When load_config("/tmp/does-not-exist.in.toml") is called
    Then it returns Err(ConfigError::Io(_))

  @rq-57f8de41
  Scenario: Malformed TOML
    Given a file at "/tmp/sim.in.toml" containing the bytes "schema_version = ["
    When load_config("/tmp/sim.in.toml") is called
    Then it returns Err(ConfigError::Parse { .. })

  # --- Filename convention ---

  @rq-43819abc
  Scenario: Reject a config whose filename does not end in `.in.toml`
    Given a valid Background config written to "/tmp/sim/sim.toml"
    When load_config("/tmp/sim/sim.toml") is called
    Then it returns Err(ConfigError::InvalidConfigFilename { path: "/tmp/sim/sim.toml" })
    And the file was not opened (verified by spy on the IO layer)

  @rq-1514bec6
  Scenario: Reject a config with an upper-case `.IN.toml` suffix
    Given a valid Background config written to "/tmp/sim/argon.IN.toml"
    When load_config("/tmp/sim/argon.IN.toml") is called
    Then it returns Err(ConfigError::InvalidConfigFilename { path: "/tmp/sim/argon.IN.toml" })

  @rq-032b4b79
  Scenario: Reject a config whose derived root is empty
    Given a valid Background config written to "/tmp/sim/.in.toml"
    When load_config("/tmp/sim/.in.toml") is called
    Then it returns Err(ConfigError::InvalidConfigFilename { path: "/tmp/sim/.in.toml" })

  @rq-9ecf5a3a
  Scenario: Filename-convention check runs before schema_version check
    Given a file at "/tmp/sim/sim.toml" containing only "schema_version = 2"
    When load_config("/tmp/sim/sim.toml") is called
    Then it returns Err(ConfigError::InvalidConfigFilename { path: "/tmp/sim/sim.toml" })
      (the schema_version mismatch is never reached)

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
    And config.phases[0].integrator matches IntegratorKind::LangevinBaoab { friction: 1.0e12, temperature: 300.0, seed: 42 }

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
    And config.phases[0].integrator matches IntegratorKind::VelocityVerlet { lossless: false }

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
    Given the Background config at "/tmp/sim/argon.in.toml" with init="out.xyz" and
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
  Scenario: timings_path defaults to <config-root>.out.timings
    Given the Background config at "/tmp/sim/argon.in.toml" with no [output].timings_path
    When load_config is called
    Then config.phases[0].output.timings_path equals "/tmp/sim/argon.out.timings"

  @rq-fa24a8d1
  Scenario: timings_path can be overridden in [output]
    Given the Background config at "/tmp/sim/argon.in.toml" with
      [output].timings_path = "custom.timings"
    When load_config is called
    Then config.phases[0].output.timings_path equals "/tmp/sim/custom.timings"

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
    Given the Background config at "/tmp/sim/argon.in.toml" with topology="argon.in.topology"
    When load_config is called
    Then config.topology equals Some("/tmp/sim/argon.in.topology")

  @rq-576561a2
  Scenario: topology absolute path is preserved
    Given the Background config with topology="/data/argon.in.topology"
    When load_config is called
    Then config.topology equals Some("/data/argon.in.topology")

  @rq-4186d4f4
  Scenario: Reject topology = init
    Given the Background config with init="argon.in.xyz" and topology="argon.in.xyz"
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
    And config.phases[0].output.trajectory_every equals 0

  @rq-318cd47d
  Scenario: log_every = 0 is accepted (disables log output)
    Given the Background config with [output].log_every=0
    When load_config is called
    Then it returns Ok(config)
    And config.phases[0].output.log_every equals 0

  # --- [thermostat] presence and absence ---

  @rq-ca356c08
  Scenario: [thermostat] section absent yields config.phases[0].thermostat = None
    Given the Background config with no [thermostat] section
    When load_config is called
    Then it returns Ok(config)
    And config.phases[0].thermostat is None

  @rq-b7cd6d16
  Scenario: [thermostat] kind = "berendsen" is accepted
    Given the Background config with [integrator] kind="velocity-verlet"
    And a [thermostat] section with kind="berendsen", temperature=300.0, tau=1.0e-13
    When load_config is called
    Then it returns Ok(config)
    And config.phases[0].thermostat matches Some(ThermostatKind::Berendsen { temperature: 300.0, tau: 1.0e-13 })

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
    And config.phases[0].integrator matches IntegratorKind::VelocityVerlet { .. }
    And config.phases[0].thermostat matches Some(ThermostatKind::Csvr { .. })

  @rq-4cd2ec5b
  Scenario: IntegratorBuilder::owns_thermostat agrees with the validation rule
    Given the "velocity-verlet" builder from IntegratorRegistry::with_builtins()
    Then builder.owns_thermostat(&{ lossless: false }) returns false
    Given the "langevin-baoab" builder from IntegratorRegistry::with_builtins()
    Then builder.owns_thermostat(&{ friction: 1.0, temperature: 300.0, seed: 0 }) returns true

  @rq-23806456
  Scenario: MtkNpt integrator with all required fields is accepted
    Given the Background config with [integrator] kind="mtk-npt",
      temperature=85.0, pressure=1.0e5, tau_t=1.0e-13, tau_p=1.0e-12
    When load_config is called
    Then it returns Ok(config)
    And config.phases[0].integrator matches IntegratorKind::MtkNpt {
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
  Scenario: IntegratorBuilder::owns_barostat matrix
    Given the "velocity-verlet" builder from IntegratorRegistry::with_builtins()
    Then builder.owns_barostat(&{ lossless: false }) returns false
    Given the "langevin-baoab" builder
    Then builder.owns_barostat(&{ friction: 1.0, temperature: 300.0, seed: 0 }) returns false
    Given the "mtk-npt" builder
    Then builder.owns_barostat(&{ temperature: 85.0, pressure: 1.0e5,
      tau_t: 1.0e-13, tau_p: 1.0e-12,
      chain_length: 3, yoshida_order: 3, n_resp: 1 }) returns true

  # --- [barostat] section ---

  @rq-4bbbada4
  Scenario: [barostat] section absent yields config.phases[0].barostat = None
    Given the Background config with no [barostat] section
    When load_config is called
    Then it returns Ok(config)
    And config.phases[0].barostat is None

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
    And config.phases[0].barostat matches
      Some(BarostatKind::Berendsen { pressure: 1.0e5, tau: 1.0e-12, compressibility: 4.5e-10 })

  @rq-c1d79d33
  Scenario: [barostat] kind = "c-rescale" with all required fields accepted
    Given the Background config with [integrator] kind="velocity-verlet"
    And a [barostat] section with kind="c-rescale", pressure=1.0e5,
      temperature=85.0, tau=1.0e-12, compressibility=4.5e-10, seed=42
    When load_config is called
    Then it returns Ok(config)
    And config.phases[0].barostat matches Some(BarostatKind::CRescale {
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

  # --- SlotConfig shape and registry-dispatched validation ---

  @rq-993ec182
  Scenario: [integrator] section parses into a SlotConfig
    Given the Background config with [integrator] kind="velocity-verlet" and lossless=false
    When load_config is called
    Then config.integrator.kind equals "velocity-verlet"
    And config.integrator.params.get("lossless") equals Some(toml::Value::Boolean(false))

  @rq-9ce1bd52
  Scenario: validate_against accepts a well-formed config
    Given a Config returned by load_config from the Background config
    And the default Registries::with_builtins()
    When config.validate_against(&registries) is called
    Then it returns Ok(())

  @rq-aa1492a7
  Scenario: validate_against rejects an unknown integrator kind
    Given a Config whose integrator.kind equals "no-such-integrator"
    And the default Registries::with_builtins()
    When config.validate_against(&registries) is called
    Then it returns Err(ConfigError::UnknownKind {
      slot: "integrator", kind: "no-such-integrator" })

  @rq-2afb76f2
  Scenario: validate_against surfaces per-builder validate_params errors
    Given a Config whose integrator is the "langevin-baoab" kind with friction = -1.0
    And the default Registries::with_builtins()
    When config.validate_against(&registries) is called
    Then it returns Err(ConfigError::InvalidValue { field: "integrator.friction", .. })

  @rq-0040baca
  Scenario: validate_against enforces IncompatibleThermostat from the builder predicate
    Given a Config whose integrator is "langevin-baoab" and whose thermostat is "csvr"
    And the default Registries::with_builtins()
    When config.validate_against(&registries) is called
    Then it returns Err(ConfigError::IncompatibleThermostat {
      integrator: "langevin-baoab" })

  @rq-d370907d
  Scenario: validate_constraint_compatibility rejects lossless velocity-Verlet with constraints
    Given a Config whose integrator is "velocity-verlet" with lossless = true
    And the default Registries::with_builtins()
    When config.validate_constraint_compatibility(&registries, true) is called
    Then it returns Err(ConfigError::IncompatibleConstraint {
      integrator: "velocity-verlet" })

  @rq-8a3c0426
  Scenario: validate_constraint_compatibility accepts lossy velocity-Verlet with constraints
    Given a Config whose integrator is "velocity-verlet" with lossless = false
    And the default Registries::with_builtins()
    When config.validate_constraint_compatibility(&registries, true) is called
    Then it returns Ok(())

  # --- [[phase]] array ---

  @rq-5e69125b
  Scenario: Reject an empty [[phase]] array
    Given the Background config with the [[phase]] array removed
    When load_config is called
    Then it returns Err(ConfigError::EmptyPhases)

  @rq-f6107e43
  Scenario: Reject duplicate phase names
    Given the Background config plus a second [[phase]] entry also named "run"
    When load_config is called
    Then it returns Err(ConfigError::DuplicatePhaseName { name: "run" })

  @rq-0dc50ce0
  Scenario: Reject a phase name containing non-ASCII characters
    Given the Background config with [[phase]] name="αβ"
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "phase[0].name", .. })

  @rq-96d9c9df
  Scenario: Reject phase dt <= 0
    Given the Background config with [[phase]] dt = 0.0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "phase[0].dt", .. })

  @rq-90e307b2
  Scenario: Two-phase config loads with distinct per-phase parameters
    Given a config with two [[phase]] entries: phase 0 name="equil"
      n_steps=5000 dt=1.0e-15 with [phase.integrator] kind="velocity-verlet";
      phase 1 name="prod" n_steps=10000 dt=2.0e-15 with [phase.integrator]
      kind="velocity-verlet"
    When load_config is called
    Then config.phases has length 2
    And config.phases[0].name equals "equil"
    And config.phases[0].n_steps equals 5000
    And config.phases[0].dt equals 1.0e-15
    And config.phases[1].name equals "prod"
    And config.phases[1].n_steps equals 10000
    And config.phases[1].dt equals 2.0e-15

  @rq-707b5520
  Scenario: Per-phase [phase.output] defaults derive from <root> and phase name
    Given a two-phase config at "/tmp/sim/argon.in.toml" with phase names "equil"
      and "prod" and no [phase.output] blocks
    When load_config is called
    Then config.phases[0].output.trajectory_path equals "/tmp/sim/argon.out.equil.xyz"
    And config.phases[0].output.log_path equals "/tmp/sim/argon.out.equil.log"
    And config.phases[0].output.timings_path equals "/tmp/sim/argon.out.equil.timings"
    And config.phases[1].output.trajectory_path equals "/tmp/sim/argon.out.prod.xyz"
    And config.phases[1].output.log_path equals "/tmp/sim/argon.out.prod.log"
    And config.phases[1].output.timings_path equals "/tmp/sim/argon.out.prod.timings"

  @rq-bdfb11e3
  Scenario: Per-phase [phase.output] overrides default paths
    Given a [[phase]] entry with [phase.output].trajectory_path = "custom.xyz"
    When load_config is called
    Then config.phases[0].output.trajectory_path equals "/tmp/sim/custom.xyz"

  # --- Seed uniqueness across phases ---

  @rq-46b1a697
  Scenario: Reject two phases of the same stochastic kind sharing a seed
    Given a two-phase config where both [phase.thermostat] entries declare
      kind="csvr" and seed=7
    When load_config is called
    Then it returns Err(ConfigError::DuplicatePhaseSeed { kind: "csvr", seed: 7 })

  @rq-60b19852
  Scenario: Two phases of different kinds may share a numerical seed
    Given a two-phase config where phase 0's [phase.thermostat] is csvr
      with seed=7 and phase 1's [phase.barostat] is c-rescale with seed=7
    When load_config is called
    Then it returns Ok(config)

  # --- Per-phase compatibility checks ---

  @rq-982ddb8d
  Scenario: Reject [phase.thermostat] alongside an integrator that owns its own
    Given a [[phase]] named "x" with [phase.integrator] kind="langevin-baoab"
      and [phase.thermostat] kind="csvr"
    When load_config is called
    Then it returns Err(ConfigError::IncompatibleThermostat {
      integrator: "langevin-baoab", phase: "x" })

  @rq-d617bf4a
  Scenario: Reject [phase.barostat] alongside mtk-npt in any single phase
    Given a [[phase]] named "prod" with [phase.integrator] kind="mtk-npt"
      and [phase.barostat] kind="c-rescale"
    When load_config is called
    Then it returns Err(ConfigError::IncompatibleBarostat {
      integrator: "mtk-npt", phase: "prod" })

  @rq-d70c3517
  Scenario: A topology with constraint groups paired with mtk-npt in a phase is rejected
    Given a topology declaring at least one [constraints] entry of type "SPCE"
    And a [[phase]] named "p" using [phase.integrator] kind="mtk-npt"
    When config.validate_constraint_compatibility(&registries, true) is called
    Then it returns Err(ConfigError::IncompatibleConstraint {
      integrator: "mtk-npt", phase: "p" })

  # --- Per-phase output collisions ---

  @rq-57bcb870
  Scenario: Reject two phases whose default trajectory paths collide
    Given two [[phase]] entries with name="x" each (duplicate names also trigger
      DuplicatePhaseName, but the collision check applies when explicit
      output_path settings collide too)
    And phase 0 sets [phase.output].trajectory_path = "shared.xyz"
    And phase 1 sets [phase.output].trajectory_path = "shared.xyz"
    When load_config is called
    Then it returns Err(ConfigError::PathCollision {
      kind_a: PathRole::PhaseTrajectory { phase: _ },
      kind_b: PathRole::PhaseTrajectory { phase: _ },
      path: _ })

  # --- Top-level units field ---

  @rq-062438e7
  Scenario: Top-level units field defaults to SI when absent
    Given a TOML file with no top-level units field
    When load_config is called
    Then config.units equals UnitSystem::Si

  @rq-b68761f1
  Scenario: Top-level units = "atomic" is accepted and recorded on Config
    Given a TOML file with `units = "atomic"` at the top level
    When load_config is called
    Then config.units equals UnitSystem::Atomic

  @rq-e5f889f3
  Scenario: Top-level units with an unknown string is rejected
    Given a TOML file with `units = "imperial"` at the top level
    When load_config is called
    Then it returns Err(ConfigError::UnknownUnits { got: "imperial" })

  @rq-971663ba
  Scenario: SI-mode physical scalars are atomic-unit on the returned Config
    Given a TOML file with `units = "si"` and a pair_interaction
      sigma written in metres
    When load_config is called
    Then config.pair_interactions[0].potential's sigma equals the input
      sigma divided by the bohr -> meter conversion factor (i.e. expressed
      in Bohr)

  @rq-6c7779ef
  Scenario: Atomic-mode physical scalars pass through unchanged
    Given a TOML file with `units = "atomic"` and a pair_interaction
      sigma written in Bohr
    When load_config is called
    Then config.pair_interactions[0].potential's sigma equals the input
      sigma exactly

  @rq-cedbd670
  Scenario: Validation of SI-mode input runs on post-conversion atomic values
    Given a TOML file with `units = "si"` and a [phase.thermostat]
      with `tau = -1.0` (a negative time in seconds)
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { .. }) referring to
      the post-conversion (negative atomic-unit) value
```
