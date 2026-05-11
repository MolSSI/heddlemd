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
| `[simulation]` | yes | timestep, step count, RNG seed, temperature |
| `[integrator]` | yes | lossy vs lossless mode |
| `[[particle_types]]` | yes (>= 1) | per-type properties |
| `[[pair_interactions]]` | yes (covers every pair) | per-pair LJ coefficients |
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
lossless = false

[[particle_types]]
name = "Ar"
mass = 6.6335e-26   # kg

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10    # m
epsilon = 1.65e-21  # J
cutoff = 1.0e-9     # m

[output]
trajectory_path = "argon-traj.xyz"
trajectory_every = 100
include_velocities = true
log_path = "argon.log"
log_every = 100
```

### Units <!-- rq-ed997636 -->

All physical quantities are SI: lengths in metres, mass in kilograms, time in
seconds, energy in joules, temperature in kelvin. No alternative unit systems
or unit suffixes are supported in schema v1.

### Field reference <!-- rq-e367855a -->

#### Top level

- `schema_version: u64` — must equal `1`. See *Schema version handling* below.
- `init: String` — path to the extended-XYZ initial-state file. Resolved
  relative to the config file's directory; absolute paths are honored as-is.

#### `[simulation]`

- `seed: u64` — RNG seed used for Maxwell-Boltzmann velocity generation.
  Required even when the init file supplies explicit velocities. No default.
- `n_steps: u64` — number of integration steps to execute. `0` is permitted
  (the runner writes the initial state and exits).
- `dt: f64` — integration timestep in seconds. Must be finite and strictly
  positive.
- `temperature: f64` — target temperature in kelvin. Required. Used to
  initialise velocities when the init file's `Properties` lacks a `velo:R:3`
  field; ignored (but still required and validated) when the init file
  supplies velocities. Must be finite and `>= 0.0`.

#### `[integrator]`

- `lossless: bool` — selects lossless mode (default `false`). When `true`,
  the runner allocates `LosslessBuffers` and launches `vv_kick_drift_lossless`
  / `vv_kick_lossless`. See `integration.md`.

#### `[[particle_types]]` (array of tables)

One entry per particle species. At least one entry required.

- `name: String` — unique identifier, used in the init file and in
  `pair_interactions.between`. Case-sensitive. Empty strings are rejected.
- `mass: f64` — particle mass in kilograms. Must be finite and strictly
  positive.

Names must be unique within the array.

#### `[[pair_interactions]]` (array of tables)

One entry per unordered pair of declared types. The collection contains
exactly one entry for every unordered pair, including same-type self pairs.
For `N` declared types the array contains exactly `N * (N + 1) / 2` entries.

- `between: [String; 2]` — unordered pair of declared type names. Order is
  not significant: `["A", "B"]` and `["B", "A"]` refer to the same pair.
- `potential: String` — currently must equal `"lennard-jones"`. Future values
  (`"morse"`, `"buckingham"`, ...) are reserved.
- `sigma: f64` — LJ zero-crossing distance in metres. Finite, strictly
  positive.
- `epsilon: f64` — LJ well depth in joules. Finite, strictly positive.
- `cutoff: f64` — pair distance in metres beyond which the force is treated
  as zero. Finite, strictly positive.

Same-type pairs are required even when only one type is declared:
`between = ["Ar", "Ar"]` must appear.

#### `[output]` (optional table; all fields have defaults)

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
- `log_path: String` — output log path. Default: `<config-stem>.log` in the
  same directory as the config file. Resolved like `trajectory_path`.
- `log_every: u64` — write one log row every this many integration steps.
  Default `100`. `0` disables the log entirely.

### Path resolution and overwrite policy <!-- rq-6d99f9c8 -->

- All file paths (`init`, `output.trajectory_path`, `output.log_path`) are
  interpreted relative to the **config file's containing directory** when
  not absolute. The loader resolves them before returning.
- After resolution, the three paths (init, trajectory, log) must be
  pairwise distinct.
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

Beyond per-field validation, the loader checks:

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
3. After path resolution, the three paths (init, trajectory, log) are
   pairwise distinct (`PathCollision { kind_a, kind_b, path }`).
4. **Runner restriction (this feature only):** the number of declared
   `[[particle_types]]` equals `1`. More than one type raises
   `MultiTypeUnsupported { count }`. The schema permits multiple types so
   the format is forward-compatible; the runtime restriction is enforced
   because the present LJ kernel does not consume per-pair parameter
   tables (see `lj-pair-force.md`). A future feature lifts the
   restriction without a schema change.

## Feature API <!-- rq-110285ae -->

### Types <!-- rq-b719c42c -->

- `Config` — parsed configuration. All fields are `pub`; field names match <!-- rq-2a6a51c8 -->
  TOML keys directly (snake_case).

  Fields:
  - `schema_version: u64`
  - `init: PathBuf` — resolved against the config file's directory.
  - `simulation: SimulationConfig`
  - `integrator: IntegratorConfig`
  - `particle_types: Vec<ParticleTypeConfig>`
  - `pair_interactions: Vec<PairInteractionConfig>`
  - `output: OutputConfig`
  - `config_path: PathBuf` — the absolute path of the source config file,
    retained for error messages and default output-path derivation.

- `SimulationConfig` <!-- rq-53055a5b -->
  - `seed: u64`
  - `n_steps: u64`
  - `dt: f64`
  - `temperature: f64`

- `IntegratorConfig` <!-- rq-661bf664 -->
  - `lossless: bool`

- `ParticleTypeConfig` <!-- rq-a5ccc1de -->
  - `name: String`
  - `mass: f64`

- `PairInteractionConfig` <!-- rq-f001eaf8 -->
  - `between: (String, String)` — stored normalised so the lexicographically
    smaller string comes first, regardless of source order.
  - `potential: String`
  - `sigma: f64`
  - `epsilon: f64`
  - `cutoff: f64`

- `OutputConfig` <!-- rq-1254cd3a -->
  - `trajectory_path: PathBuf` — resolved.
  - `trajectory_every: u64`
  - `include_velocities: bool`
  - `log_path: PathBuf` — resolved.
  - `log_every: u64`

- `PathRole` — `enum { Init, Trajectory, Log }`. Used in `PathCollision`. <!-- rq-f0084057 -->

- `ConfigError` — error type returned by `load_config`. Variants: <!-- rq-0b9372e8 -->
  - `Io(String)` — failed to read the config file (with the OS error
    message).
  - `Parse(String)` — TOML parser rejected the file (with location info
    from the underlying parser).
  - `MissingField { field: &'static str }` — required field absent. `field`
    uses dot notation (e.g. `"simulation.dt"`, `"schema_version"`,
    `"particle_types[0].name"`).
  - `UnsupportedSchemaVersion { actual: u64, supported: u64 }`.
  - `InvalidValue { field: &'static str, reason: String }` — finite or
    positivity constraint violated, or `potential` is not
    `"lennard-jones"`, or a type/pair name is empty.
  - `DuplicateTypeName { name: String }`.
  - `UnknownTypeInPair { name: String, pair_index: usize }`.
  - `MissingPairInteraction { types: (String, String) }`.
  - `DuplicatePairInteraction { types: (String, String) }`.
  - `PathCollision { kind_a: PathRole, kind_b: PathRole, path: PathBuf }`.
  - `MultiTypeUnsupported { count: usize }`.

### Functions <!-- rq-39881bb0 -->

- `load_config(path: &Path) -> Result<Config, ConfigError>` <!-- rq-e8259ee5 -->
  - Reads the file at `path`, parses TOML, performs every validation in
    *Field reference* and *Cross-validation*, resolves the three paths
    against `path.parent()` (or `"."` if `path` has no parent), and
    returns the populated `Config`.
  - On any validation failure, returns the first error encountered in
    declaration order: top-level fields first, then `[simulation]`, then
    `[integrator]`, then `[[particle_types]]`, then `[[pair_interactions]]`,
    then `[output]`, then cross-validation steps 1–4 in that order.

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
      one [integrator] section with lossless=false,
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
    And config.integrator.lossless equals false
    And config.particle_types has length 1
    And config.particle_types[0].name equals "Ar"
    And config.particle_types[0].mass equals 6.6335e-26
    And config.pair_interactions has length 1
    And config.pair_interactions[0].between equals ("Ar", "Ar")
    And config.init equals "/tmp/sim/argon.xyz"
    And config.config_path equals "/tmp/sim/sim.toml"

  @rq-894c16c4
  Scenario: Defaults populate the output section when [output] is omitted
    Given the Background config with no [output] section, written to "/tmp/sim/sim.toml"
    When load_config("/tmp/sim/sim.toml") is called
    Then config.output.trajectory_path equals "/tmp/sim/sim-traj.xyz"
    And config.output.trajectory_every equals 100
    And config.output.include_velocities equals true
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
    Then it returns Err(ConfigError::Parse(_))

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

  @rq-a3a5905d
  Scenario: Reject unknown potential
    Given the Background config with pair_interactions[0].potential="morse"
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "pair_interactions[0].potential", reason: _ })

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

  # --- Multi-type restriction ---

  @rq-f114c560
  Scenario: Reject multi-type configs in this feature version
    Given a config with two declared types "Ar" and "Kr"
      and pair_interactions for all three unordered pairs ("Ar","Ar"), ("Ar","Kr"), ("Kr","Kr")
    When load_config is called
    Then it returns Err(ConfigError::MultiTypeUnsupported { count: 2 })

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
```
