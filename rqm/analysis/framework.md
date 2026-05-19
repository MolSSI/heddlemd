# Feature: Analysis Framework and `dynamics analyze` <!-- rq-fd8bb824 -->

The `dynamics analyze` subcommand performs post-processing analyses
on a trajectory written by an earlier `dynamics run`. The work is
driven by a TOML input file — `<root>.in.analysis` — that declares
one or more analyses to compute. Outputs are written next to the
input as `<root>.out.<name>.csv` (one CSV per declared analysis).

The framework defines the CLI surface, the input-file schema, the
trajectory iteration pass, the per-analysis dispatch contract, and
the open-registry pattern used to register additional analysis
kinds. Concrete analysis kinds are documented in their own files
under `rqm/analysis/` (currently: `rdf.md`).

## CLI <!-- rq-f2f2519f -->

```
dynamics analyze <root>.in.analysis
```

- `<root>.in.analysis` is the path to a TOML analysis file (schema
  below). Relative paths resolve against the current working
  directory. The filename must end in `.in.analysis`
  (case-sensitive on the whole suffix) and the derived `<root>` must
  be non-empty; violations are rejected at load time with
  `AnalyzeError::InvalidAnalysisFilename { path: PathBuf }` without
  opening the file.
- No flags. v1 is CPU-only; no `--with-gpu` mode is offered.
- Exit codes:
  - `0` — every declared analysis completed and every output file
    flushed.
  - `1` — any error before the first frame is processed: malformed
    CLI args, filename-convention violation, analysis-file parse
    failure, sibling-config load failure, trajectory open failure,
    output-collision failure.
  - `2` — error during the trajectory pass or output write: a
    malformed frame, a per-analysis runtime error, or a write
    failure on a `*.csv` output.
- Errors are reported as a single `error: ...` line on stderr, in
  the same shape as `dynamics run`.
- A successful run prints one final-summary line on stdout:
  ```
  [dynamics] analyze complete: <K> analyses over <F> frames (...)
  ```
  where `<K>` is the number of declared analyses and `<F>` is the
  number of frames consumed after `first_frame`, `last_frame`, and
  `stride` are applied (see *Frame selection* below).

`dynamics lint` (see `simulation-runner.md`) dispatches on the
config-path filename extension:
- A path ending in `.in.toml` runs the simulation lint pipeline.
- A path ending in `.in.analysis` runs the analysis lint pipeline
  documented under *Analyze lint flow* below.
- Any other extension is rejected with the existing
  `InvalidConfigFilename`-style filename-convention error.

## `<root>.in.analysis` file <!-- rq-76d8bba2 -->

### Schema <!-- rq-22755b52 -->

```toml
schema_version = 1

# Optional. Defaults to "<root>.in.toml" in the same directory.
# simulation = "argon.in.toml"

# Optional. Selects which simulation phase's trajectory the
# analyses consume. Must match the `name` of an entry in the loaded
# simulation config's [[phase]] array. Defaults to the *last* phase
# in the config (typically the production phase).
# phase = "production"

# Optional. Defaults to the resolved `output.trajectory_path` from
# the selected phase of the loaded simulation config (which itself
# defaults to "<root>.out.<phase-name>.xyz").
# trajectory = "argon.out.production.xyz"

# Optional frame-selection bounds, applied in order before any
# analysis is dispatched. Defaults shown.
first_frame = 0     # u64, 0-based position of the first frame to use
# last_frame = ...   # u64, 0-based position of the last frame to use (inclusive). Defaults to the last frame in the file.
stride = 1          # u64, must be >= 1

[[analyses]]
name = "ar-ar"
kind = "rdf"
# ...kind-specific parameters (see rqm/analysis/rdf.md)
```

### Top-level fields <!-- rq-759d2445 -->

- `schema_version: u64` — must equal `1`. A missing field produces
  `AnalyzeError::MissingField { field: "schema_version" }`; any
  other value produces
  `AnalyzeError::UnsupportedSchemaVersion { actual, supported: 1 }`.
- `simulation: String` — optional. Path to the `.in.toml` config
  that produced the trajectory. Resolved relative to the
  `.in.analysis` file's directory. Defaults to
  `<root>.in.toml` in the same directory.
- `phase: String` — optional. Selects which phase of the loaded
  simulation config to analyze. Must match the `name` field of an
  entry in the simulation config's `[[phase]]` array. Defaults to
  the **last** phase in the config (which is conventionally the
  production phase in equilibration-then-production protocols).
  Unknown phase names produce
  `AnalyzeError::UnknownPhase { phase: String, available: Vec<String> }`
  at load time, with `available` listing the phase names declared
  by the simulation config.
- `trajectory: String` — optional. Path to the trajectory file
  (`.out.<phase-name>.xyz` produced by `dynamics run`). Resolved
  relative to the `.in.analysis` file's directory. Defaults to the
  resolved `output.trajectory_path` of the **selected phase** in
  the loaded simulation config (which itself defaults to
  `<root>.out.<phase-name>.xyz` per `rqm/io/config-schema.md`).
- `first_frame: u64` — optional, default `0`. Position of the first
  trajectory frame to use, counted from the start of the file
  (`first_frame = 0` means "use the step-0 frame").
- `last_frame: u64` — optional, default `<file frame count> - 1`.
  Position of the last frame to use, inclusive.
  `last_frame < first_frame` is rejected with `InvalidValue` at
  load time; `last_frame >= <file frame count>` is rejected with
  `FrameOutOfRange` at trajectory-open time.
- `stride: u64` — optional, default `1`. The trajectory pass
  consumes frames at positions `first_frame, first_frame + stride,
  first_frame + 2 * stride, ...` up to and including `last_frame`.
  `stride = 0` is rejected with `InvalidValue` at load time.

### `[[analyses]]` array <!-- rq-65ad92d0 -->

Required and non-empty. Each entry is a TOML table with:

- `name: String` — required. Identifier used to derive the output
  filename (`<root>.out.<name>.csv` by default). Non-empty,
  case-sensitive, must contain only ASCII letters, digits, `-`, and
  `_`. Names must be unique within the file; duplicates produce
  `DuplicateAnalysisName { name }`.
- `kind: String` — required. Selects the registered analysis
  builder. v1's built-in registry contains `"rdf"`.
- `output_path: String` — optional. Overrides the default output
  path. Resolved relative to the `.in.analysis` file's directory;
  absolute paths are honored as-is.
- Any further fields are kind-specific and validated by the
  matching builder's `validate_params(&toml::Value)` (see
  *AnalysisBuilder* under *Feature API*). Unknown fields for the
  chosen `kind` are rejected as `AnalyzeError::Parse`.

Output-path collision rules:
- The set of resolved `output_path` values must be pairwise
  distinct across all `[[analyses]]` entries.
- No resolved `output_path` may collide with the resolved
  `simulation`, `trajectory`, `.in.analysis` itself, or any output
  path declared by the loaded simulation config
  (`output.trajectory_path`, `output.log_path`,
  `output.timings_path`). Violations produce
  `AnalyzePathCollision { kind_a: PathRole, kind_b: PathRole, path: PathBuf }`.

### Validation order <!-- rq-5bad4341 -->

`load_analysis_config(path: &Path)` performs the checks below in
this order. The first failure short-circuits the rest:

1. Filename-convention check (must end in `.in.analysis`, derived
   root non-empty). Fail → `InvalidAnalysisFilename`.
2. File read. Fail → `Io(String)`.
3. `schema_version` deserialisation. Fail → `MissingField` /
   `UnsupportedSchemaVersion`.
4. TOML deserialisation of the top-level fields and the
   `[[analyses]]` array (without per-kind parameter validation).
   Fail → `Parse { path, message }` or `MissingField`.
5. Per-field domain checks: `first_frame`, `last_frame`, `stride`,
   `[[analyses]]` non-empty, `name` non-empty and ASCII-only, name
   uniqueness. Fail → `InvalidValue` or `DuplicateAnalysisName`.
6. Path resolution: resolve `simulation` (or `<root>.in.toml`
   default), `trajectory` (or simulation-config default), each
   `output_path` (or `<root>.out.<name>.csv` default).
7. Path-collision check across the resolved set. Fail →
   `AnalyzePathCollision`.

Per-kind parameter validation runs separately via
`validate_against(&registries)` (see *Feature API*); the loader
returns the parsed `AnalysisConfig` with each `[[analyses]]` entry
carrying its parameters as a `toml::Value` for the chosen builder
to inspect.

## Analyze flow <!-- rq-618864ce -->

`dynamics analyze` proceeds through these stages in order. Any stage
that fails terminates the process with the appropriate exit code
and stderr message.

1. **Parse CLI.** Confirm the form `analyze <path>`.
2. **Load analysis config.** Call `load_analysis_config(&path)`.
   Failure → exit 1.
3. **Load simulation config.** Call `load_config(&analysis.simulation)`
   (see `rqm/io/config-schema.md`). Failure → exit 1.
4. **Validate per-kind params.** For each `[[analyses]]` entry,
   look up the kind in `registries.analyses` and call
   `builder.validate_params(&entry.params)`. Failure → exit 1.
5. **Open trajectory.** Call
   `TrajectoryReader::open(&analysis.trajectory, &type_names)`
   (see `rqm/io/trajectory-output.md`). The reader validates the
   first-frame header against the loaded simulation config's
   particle types and box convention. Failure → exit 1.
6. **Construct analysis slots.** For each `[[analyses]]` entry,
   call `builder.build(&entry.params, &reader.first_frame_header,
   &simulation_config)` → `Box<dyn Analysis>`. Each builder is
   free to allocate its per-run state (histogram buffers, etc.)
   sized against the first-frame header (particle count, box
   dimensions, particle-type counts). Failure → exit 1.
7. **Pre-flight output checks.** For each resolved
   `output_path`, verify the file does not already exist.
   Failure → exit 1 with `OutputExists { path }`.
8. **Trajectory pass.** Iterate `reader.frames()` with the
   `first_frame`, `last_frame`, and `stride` selection applied. For
   each selected frame, call
   `analysis.consume_frame(&frame, &sim_box)` on every analysis in
   declaration order. A `consume_frame` error → exit 2.
9. **Finalise and write.** For each analysis, call
   `analysis.finalize_and_write(&output_path, &simulation_config)`,
   which produces the CSV file. A write failure → exit 2.
10. **Final summary.** Print one line on stdout (see *CLI*) and
    exit 0.

Lint stops at stage 6 (no trajectory pass, no writes); see
*Analyze lint flow* below.

## Analyze lint flow <!-- rq-5a6f582d -->

`dynamics lint <path>.in.analysis` performs the setup-phase checks
the analyze pipeline would perform, then exits. The pipeline
reuses the same loaders as *Analyze flow*; no output files are
created, and the trajectory is opened only to read its first frame
header (not iterated). Stages, in order, recorded as a `LintReport`
following the conventions in `rqm/simulation-runner.md`:

1. **`config`** — `load_analysis_config(&path)` (filename
   convention, TOML parse, per-field domain checks, name
   uniqueness, path-collision check). The `.in.toml` referenced
   by `simulation` (or the implicit default) is also loaded and
   validated via `load_config`.
2. **`output paths`** — each resolved `output_path` is tested with
   `Path::exists()`. A pre-existing file is a FAIL with
   `OutputExists { path }`.
3. **`trajectory`** — `TrajectoryReader::open(&path, &type_names)`
   succeeds and the first frame's particle count and box are
   recorded in the stage description.
4. **`analyses`** — each `[[analyses]]` entry's
   `builder.validate_params(&entry.params)` succeeds, and
   `builder.build(...)` against the first-frame header succeeds
   (covering builder-detected geometric errors such as RDF
   `r_max > min_perpendicular_width / 2`).

Skipped / not-checked semantics match the simulation-lint flow
(short-circuit on first failure; subsequent stages are recorded as
`skipped (earlier check failed)`).

## Output convention <!-- rq-f32a2004 -->

- Each `[[analyses]]` entry writes one CSV file to its resolved
  `output_path`.
- Default `output_path` is `<root>.out.<name>.csv` in the same
  directory as the `.in.analysis` file, where `<root>` is the
  derived root of the analysis filename and `<name>` is the entry's
  `name` field. Example: `argon.in.analysis` with
  `name = "ar-ar"` → `argon.out.ar-ar.csv`.
- The CSV header and column meaning are defined per-kind (see e.g.
  `rqm/analysis/rdf.md`). The framework guarantees only that the
  file is UTF-8 with `\n` line endings, ASCII-compatible.
- The framework refuses to overwrite an existing file (the
  pre-flight `OutputExists` check plus the per-writer
  `create_new` open).

## Reproducibility contract <!-- rq-175e5b45 -->

Two `dynamics analyze` runs over the same `.in.analysis`, the same
sibling `.in.toml`, the same trajectory file, and the same
analysis-builder registry produce byte-identical `*.out.*.csv`
output files. The guarantee is unconditional on hardware (the v1
pipeline is CPU-only and performs no GPU calls), but conditional on:

- Each registered builder enforcing its own determinism (canonical
  enumeration order over pairs / triples / atoms, fixed reduction
  order, fixed numeric formatting).
- The trajectory file itself being identical (which is already
  guaranteed by `dynamics run`'s reproducibility contract on the
  same GPU).
- The same set of registered builders being present.

Every built-in analysis kind shipped with the engine satisfies the
per-builder requirements. Custom user-registered builders must do
the same to inherit the guarantee.

## Frame selection <!-- rq-0ebfcbbc -->

The trajectory pass consumes frames at positions
`first_frame, first_frame + stride, first_frame + 2 * stride, ...`
up to and including `last_frame`. Positions are 0-based and refer
to the frame's order in the file, not the `Step=` value on the
comment line. The frame count `F` reported in the final-summary
line is the number of selected frames (after `first_frame`,
`last_frame`, and `stride` are applied).

Degenerate cases:
- `last_frame < first_frame`: rejected at load time with
  `InvalidValue`.
- `stride = 0`: rejected at load time with `InvalidValue`.
- The trajectory has fewer frames than `last_frame + 1`: rejected
  at trajectory-open time with
  `FrameOutOfRange { requested: u64, available: u64 }`.
- No frame falls within the selection (e.g. `first_frame >
  available - 1` while `last_frame = first_frame` itself):
  rejected at trajectory-open time with `FrameOutOfRange`.

## Feature API <!-- rq-362c9774 -->

### Types <!-- rq-6c5a7246 -->

- `AnalysisConfig` — parsed `<root>.in.analysis`. All fields are <!-- rq-aa91623d -->
  `pub`.
  - `schema_version: u64`
  - `simulation: PathBuf` — resolved.
  - `phase: String` — the selected phase's name. Populated from the
    optional `phase` field; defaults to the last phase in the loaded
    simulation config.
  - `trajectory: PathBuf` — resolved. Defaults to the selected
    phase's `output.trajectory_path` from the loaded simulation
    config.
  - `first_frame: u64`
  - `last_frame: Option<u64>` — `None` means "the last frame in
    the file at trajectory-open time".
  - `stride: u64`
  - `analyses: Vec<AnalysisEntry>`
  - `config_path: PathBuf` — the absolute path of the source
    `.in.analysis` file.

- `AnalysisEntry` — one row of the `[[analyses]]` array. <!-- rq-ca3ec865 -->
  - `name: String`
  - `kind: String`
  - `output_path: PathBuf` — resolved.
  - `params: toml::Value` — the kind-specific parameters captured
    via `#[serde(flatten)]`, consumed by the matching builder.

- `AnalyzeError` — error type returned by `load_analysis_config` <!-- rq-cd7d7ee5 -->
  and `run_analyses`. Variants:
  - `InvalidAnalysisFilename { path: PathBuf }` — the filename
    convention check failed.
  - `Io(String)` — failed to read the analysis file.
  - `Parse { path: String, message: String }` — TOML structural
    error or per-kind unknown-field rejection.
  - `MissingField { field: String }`
  - `UnsupportedSchemaVersion { actual: u64, supported: u64 }`
  - `InvalidValue { field: String, reason: String }`
  - `DuplicateAnalysisName { name: String }`
  - `EmptyAnalyses` — the `[[analyses]]` array is empty or absent.
  - `UnknownKind { kind: String }` — the chosen `kind` is not
    registered.
  - `UnknownPhase { phase: String, available: Vec<String> }` — the
    `phase` field names a phase that does not appear in the loaded
    simulation config's `[[phase]]` array. `available` lists the
    actual phase names declared by the config.
  - `AnalyzePathCollision { kind_a: PathRole, kind_b: PathRole,
    path: PathBuf }` — two supplied / defaulted paths resolve to
    the same location. `PathRole` is the same enum used by
    `ConfigError::PathCollision`, extended with the variants
    `AnalysisInput`, `AnalysisOutput { name: String }`.
  - `Config(ConfigError)` — propagated from the loaded simulation
    config.
  - `Trajectory(TrajectoryReaderError)` — propagated from the
    trajectory reader (see `rqm/io/trajectory-output.md`).
  - `Analysis { name: String, error: AnalysisRuntimeError }` — a
    per-analysis builder, `consume_frame`, or `finalize_and_write`
    error. `name` is the entry's `name` field.
  - `OutputExists { path: PathBuf }` — pre-flight output-path
    collision; surfaces the same condition that the CSV writer's
    `create_new` would detect later.
  - `FrameOutOfRange { requested: u64, available: u64 }` —
    `last_frame >= available` at trajectory-open time.

- `AnalysisRuntimeError` — per-analysis error type returned by the <!-- rq-3825d7c4 -->
  builder, `consume_frame`, and `finalize_and_write`. Variants:
  - `InvalidValue { field: String, reason: String }` — builder
    parameter validation failed (e.g. RDF
    `r_max > min_perpendicular_width / 2`).
  - `Io(String)` — CSV write failure.
  - `Other(String)` — escape hatch for kind-specific errors that
    do not fit the above.

- `AnalysisBuilder` — open-shaped builder trait. Mirrors <!-- rq-86f01d20 -->
  `IntegratorBuilder` etc.:
  ```rust
  pub trait AnalysisBuilder: Send + Sync {
      fn kind_name(&self) -> &'static str;
      fn validate_params(&self, params: &toml::Value) -> Result<(), AnalyzeError>;
      fn build(
          &self,
          params: &toml::Value,
          header: &TrajectoryFrameHeader,
          sim_config: &Config,
      ) -> Result<Box<dyn Analysis>, AnalysisRuntimeError>;
  }
  ```

- `Analysis` — per-run handle returned by an `AnalysisBuilder`. <!-- rq-8464775b -->
  Mirrors the per-step interface used by integrators:
  ```rust
  pub trait Analysis: Send {
      fn consume_frame(
          &mut self,
          frame: &TrajectoryFrame,
          sim_box: &SimulationBox,
      ) -> Result<(), AnalysisRuntimeError>;

      fn finalize_and_write(
          &mut self,
          output_path: &Path,
          sim_config: &Config,
      ) -> Result<(), AnalysisRuntimeError>;
  }
  ```
  `consume_frame` is invoked once per selected trajectory frame in
  declaration order across the `[[analyses]]` array.
  `finalize_and_write` is invoked exactly once per analysis at the
  end of the trajectory pass.

- `AnalysisRegistry` — open-builder registry, mirrors <!-- rq-e3ba8c3b -->
  `IntegratorRegistry`. Fields and methods:
  - `pub fn new() -> Self` — empty registry.
  - `pub fn with_builtins() -> Self` — registry pre-populated with
    every built-in builder. v1 ships exactly one: the RDF builder
    documented in `rqm/analysis/rdf.md`.
  - `pub fn register(&mut self, builder: Box<dyn AnalysisBuilder>)`
  - `pub fn lookup(&self, kind: &str) -> Option<&dyn AnalysisBuilder>`

- `Registries` (the bundle defined in `rqm/simulation-runner.md`) <!-- rq-a7211dfd -->
  carries an additional field `analyses: AnalysisRegistry`. The
  bundle's `with_builtins()` constructor populates it with
  `AnalysisRegistry::with_builtins()`; `new()` leaves it empty.
  The bundle exposes a `register_analysis(&mut self, builder:
  Box<dyn AnalysisBuilder>)` convenience method that forwards to
  `analyses.register`.

- `AnalyzeSummary` — public struct returned by `run_analyses` on <!-- rq-8914e9ff -->
  success.
  - `frames_consumed: u64` — number of selected frames passed to
    every analysis's `consume_frame`.
  - `analyses_written: u64` — number of `*.out.*.csv` files
    written (equal to `analysis_config.analyses.len()` on success).
  - `elapsed_micros: u128` — wall-clock duration of the trajectory
    pass plus the final writes.

### Functions <!-- rq-2be9d8e3 -->

- `load_analysis_config(path: &Path) -> Result<AnalysisConfig, AnalyzeError>` <!-- rq-9fa942b1 -->
  - Performs the validation order documented under *Validation
    order*. Returns the populated `AnalysisConfig` on success.
  - Does not open the trajectory and does not validate per-kind
    parameters (that is the caller's responsibility via
    `validate_against`).

- `AnalysisConfig::validate_against(&self, registries: &Registries) -> Result<(), AnalyzeError>` <!-- rq-d79986d0 -->
  - For each `analyses` entry, looks up `kind` in
    `registries.analyses` (failure → `UnknownKind { kind }`) and
    calls `builder.validate_params(&entry.params)` (failure
    propagates).

- `run_analyses(config_path: &Path) -> Result<AnalyzeSummary, AnalyzeError>` <!-- rq-8c1de56e -->
  - Convenience wrapper: equivalent to
    `run_analyses_with_registries(config_path,
    &Registries::with_builtins())`.

- `run_analyses_with_registries(config_path: &Path, registries: &Registries) -> Result<AnalyzeSummary, AnalyzeError>` <!-- rq-c9a3109a -->
  - Executes every stage of *Analyze flow* against `registries`.
    Dispatches every `[[analyses]]` entry's `kind` through
    `registries.analyses`.

- `lint_analyses(config_path: &Path) -> LintReport` <!-- rq-bcf7e0eb -->
  - Convenience wrapper: equivalent to
    `lint_analyses_with_registries(config_path,
    &Registries::with_builtins())`.

- `lint_analyses_with_registries(config_path: &Path, registries: &Registries) -> LintReport` <!-- rq-6eb18608 -->
  - Runs every stage of *Analyze lint flow* against `registries`.
    Returns a `LintReport` whose `stages` field has length `4`
    (`config`, `output paths`, `trajectory`, `analyses`).
  - Called by the `dynamics lint <path>.in.analysis` CLI dispatch.

The CLI wrapper in `src/main.rs` selects between
`run_simulation`, `lint_simulation`, `run_analyses`, and
`lint_analyses` based on the subcommand and (for `lint`) the
file extension of the supplied path. See
`rqm/simulation-runner.md` for the full dispatch contract.

## Out of Scope <!-- rq-3933f0c3 -->

- GPU-accelerated analysis kernels (v1 is CPU-only;
  `--with-gpu` is reserved for a future feature flag).
- Streaming output (analyses buffer histograms / accumulators in
  memory and write once at the end of the trajectory pass).
- Variable-box trajectories. The framework assumes the
  simulation box is constant across all consumed frames; the
  first frame's lattice is the canonical box. NPT-trajectory
  support is reserved for when a barostat that actually rescales
  the cell is integrated end-to-end.
- Cross-trajectory analysis (e.g. comparing two runs). A single
  `.in.analysis` consumes exactly one trajectory.
- Restart / append mode. Each `dynamics analyze` invocation
  produces fresh output files; appending to an existing CSV is
  not supported.
- Selecting frames by `Step=` value rather than by file position.
  `first_frame`/`last_frame`/`stride` always refer to position
  in the file (0-based).
- Filtering by particle subset beyond the type-based selection
  each kind provides.

## Gherkin Scenarios <!-- rq-4a6d949d -->

```gherkin
Feature: Analysis framework and `dynamics analyze`

  Background:
    Given a temporary directory tmp
    And tmp/argon.in.toml is a valid simulation config writing the trajectory to tmp/argon.out.xyz
    And tmp/argon.out.xyz is a valid trajectory with 11 frames

  # --- Filename convention ---

  @rq-a1735ae4
  Scenario: Reject an analysis filename that does not end in `.in.analysis`
    Given a valid analysis body is written to tmp/argon.analysis (wrong suffix)
    When dynamics is invoked with arguments ["analyze", "tmp/argon.analysis"]
    Then it exits with code 1
    And stderr contains "InvalidAnalysisFilename" and "argon.analysis"
    And the file was not opened

  @rq-bf98584a
  Scenario: Reject an analysis filename whose derived root is empty
    Given a valid analysis body is written to tmp/.in.analysis
    When dynamics is invoked with arguments ["analyze", "tmp/.in.analysis"]
    Then it exits with code 1
    And stderr contains "InvalidAnalysisFilename"

  # --- Loader ---

  @rq-f5166314
  Scenario: Load a valid minimal analysis file with implicit pairing
    Given tmp/argon.in.analysis declares schema_version=1 and one [[analyses]] entry
      with name="ar-ar" kind="rdf" between=["Ar","Ar"] r_max=8e-10 n_bins=64
    When load_analysis_config("tmp/argon.in.analysis") is called
    Then it returns Ok(config)
    And config.simulation equals canonical "tmp/argon.in.toml"
    And config.trajectory equals canonical "tmp/argon.out.xyz"
    And config.first_frame equals 0
    And config.last_frame is None
    And config.stride equals 1
    And config.analyses has length 1
    And config.analyses[0].name equals "ar-ar"
    And config.analyses[0].kind equals "rdf"
    And config.analyses[0].output_path equals canonical "tmp/argon.out.ar-ar.csv"

  @rq-ee7c4af4
  Scenario: Explicit `simulation` and `trajectory` override the implicit defaults
    Given tmp/argon.in.analysis sets simulation="other.in.toml" and trajectory="other.out.xyz"
    When load_analysis_config is called
    Then config.simulation equals canonical "tmp/other.in.toml"
    And config.trajectory equals canonical "tmp/other.out.xyz"

  @rq-2d107b4d
  Scenario: Reject an empty `[[analyses]]` array
    Given tmp/argon.in.analysis declares no [[analyses]] entries
    When load_analysis_config is called
    Then it returns Err(AnalyzeError::EmptyAnalyses)

  @rq-f067cfe1
  Scenario: Reject duplicate analysis names
    Given tmp/argon.in.analysis declares two [[analyses]] entries both with name="x"
    When load_analysis_config is called
    Then it returns Err(AnalyzeError::DuplicateAnalysisName { name: "x" })

  @rq-28fd5e34
  Scenario: Reject a name containing non-ASCII characters
    Given tmp/argon.in.analysis declares one [[analyses]] entry with name="αβ"
    When load_analysis_config is called
    Then it returns Err(AnalyzeError::InvalidValue { field: "analyses[0].name", .. })

  @rq-ede9a68f
  Scenario: Reject stride = 0
    Given tmp/argon.in.analysis sets stride=0
    When load_analysis_config is called
    Then it returns Err(AnalyzeError::InvalidValue { field: "stride", .. })

  @rq-f0c97cb7
  Scenario: Reject last_frame less than first_frame
    Given tmp/argon.in.analysis sets first_frame=5 and last_frame=2
    When load_analysis_config is called
    Then it returns Err(AnalyzeError::InvalidValue { field: "last_frame", .. })

  # --- Path collisions ---

  @rq-d2b16164
  Scenario: Reject output_path equal to the trajectory
    Given tmp/argon.in.analysis sets analyses[0].output_path="argon.out.xyz"
    When load_analysis_config is called
    Then it returns Err(AnalyzeError::AnalyzePathCollision { kind_a, kind_b, .. })
    And the collision pair includes PathRole::AnalysisOutput { name: "ar-ar" } and PathRole::Trajectory

  @rq-bb96fc2f
  Scenario: Reject two analyses sharing the same explicit output_path
    Given tmp/argon.in.analysis has two analyses both with output_path="shared.csv"
    When load_analysis_config is called
    Then it returns Err(AnalyzeError::AnalyzePathCollision { .. })

  @rq-bedbd6d9
  Scenario: Reject an output_path equal to the simulation's log file
    Given tmp/argon.in.analysis sets analyses[0].output_path="argon.out.log"
    When load_analysis_config followed by run_analyses is called
    Then it returns Err(AnalyzeError::AnalyzePathCollision { .. })

  # --- Run-time end-to-end ---

  @rq-f388ae12
  Scenario: Successful run over the full trajectory
    Given tmp/argon.in.analysis declares one RDF analysis with valid parameters
    And tmp/argon.out.ar-ar.csv does not exist
    When dynamics is invoked with arguments ["analyze", "tmp/argon.in.analysis"]
    Then it exits with code 0
    And tmp/argon.out.ar-ar.csv exists
    And the file's last line is the maximum-r bin
    And stdout matches "\[dynamics\] analyze complete: 1 analyses over 11 frames .*"

  @rq-270637dd
  Scenario: Pre-flight refuses to overwrite an existing output file
    Given tmp/argon.out.ar-ar.csv already exists with arbitrary content
    When dynamics is invoked with arguments ["analyze", "tmp/argon.in.analysis"]
    Then it exits with code 1
    And stderr contains "OutputExists" and "argon.out.ar-ar.csv"
    And tmp/argon.out.ar-ar.csv is unchanged

  @rq-36ec88d9
  Scenario: Missing trajectory file is reported before any analysis builds
    Given tmp/argon.out.xyz is deleted between writing the .in.analysis and invoking analyze
    When dynamics is invoked with arguments ["analyze", "tmp/argon.in.analysis"]
    Then it exits with code 1
    And stderr contains "argon.out.xyz"
    And no analysis builder's `build` method was called

  @rq-f76d0576
  Scenario: Missing sibling .in.toml under implicit pairing
    Given tmp/argon.in.analysis exists but tmp/argon.in.toml does not
    When dynamics is invoked with arguments ["analyze", "tmp/argon.in.analysis"]
    Then it exits with code 1
    And stderr contains "argon.in.toml"

  @rq-5fe9c9e2
  Scenario: Frame selection: stride > 1 reduces frames consumed
    Given tmp/argon.out.xyz has 11 frames (positions 0..10)
    And tmp/argon.in.analysis sets first_frame=0 last_frame=10 stride=2
    When dynamics analyze runs
    Then each analysis's consume_frame is called exactly 6 times (positions 0, 2, 4, 6, 8, 10)
    And the final-summary line reports "6 frames"

  @rq-8aae6b06
  Scenario: Frame selection: last_frame past end is rejected at trajectory-open time
    Given tmp/argon.out.xyz has 11 frames
    And tmp/argon.in.analysis sets last_frame=20
    When dynamics analyze runs
    Then it exits with code 1
    And stderr contains "FrameOutOfRange" and "requested: 20" and "available: 11"

  @rq-5de7798b
  Scenario: Reproducibility across two analyze runs
    Given a valid .in.analysis, .in.toml, and trajectory
    When dynamics analyze is invoked twice on the same files
    Then the two resulting output CSVs are byte-identical for every analysis

  # --- Open registry ---

  @rq-4e095363
  Scenario: Unknown analysis kind is reported with UnknownKind
    Given tmp/argon.in.analysis declares an analysis with kind="msd"
      and the default Registries::with_builtins() is used
    When dynamics analyze runs
    Then it exits with code 1
    And stderr contains "UnknownKind" and "msd"

  @rq-ca13c67e
  Scenario: A custom analysis builder composes with the built-ins
    Given let mut registries = Registries::with_builtins()
    And a custom builder MyAnalysisBuilder with kind_name() = "my-analysis"
    When registries.register_analysis(Box::new(MyAnalysisBuilder)) is called
    Then registries.analyses.lookup("my-analysis") is Some(_)
    And registries.analyses.lookup("rdf") remains Some(_)

  # --- Lint dispatch ---

  @rq-fa2fc3a9
  Scenario: `dynamics lint` on an .in.analysis path runs the analyze lint
    Given a valid tmp/argon.in.analysis
    When dynamics is invoked with arguments ["lint", "tmp/argon.in.analysis"]
    Then it exits with code 0
    And stdout begins with "[dynamics lint] OK"
    And the report has exactly the stages "config", "output paths", "trajectory", "analyses" in that order

  @rq-b306b357
  Scenario: Lint reports a trajectory open failure under the trajectory stage
    Given tmp/argon.out.xyz is missing
    When dynamics is invoked with arguments ["lint", "tmp/argon.in.analysis"]
    Then it exits with code 1
    And stdout has a "trajectory" stage line beginning with "FAIL —"
    And stderr contains "argon.out.xyz"

  @rq-c67ad79e
  Scenario: Lint reports a builder geometric failure under the analyses stage
    Given tmp/argon.in.analysis declares an RDF analysis with r_max greater than half the box width
    When dynamics is invoked with arguments ["lint", "tmp/argon.in.analysis"]
    Then it exits with code 1
    And stdout has an "analyses" stage line beginning with "FAIL —"
    And stderr contains "r_max"

  # --- Phase selection ---

  @rq-963604a4
  Scenario: Implicit pairing defaults to the last phase of the simulation config
    Given tmp/argon.in.toml declares two phases named "equil" and "prod"
    And tmp/argon.in.analysis omits the `phase` field
    When load_analysis_config followed by run_analyses is called
    Then config.phase equals "prod"
    And config.trajectory equals canonical "tmp/argon.out.prod.xyz"

  @rq-38117e33
  Scenario: Explicit `phase` selects the matching phase's trajectory
    Given tmp/argon.in.toml declares two phases named "equil" and "prod"
    And tmp/argon.in.analysis sets phase = "equil"
    When load_analysis_config is called
    Then config.phase equals "equil"
    And config.trajectory equals canonical "tmp/argon.out.equil.xyz"

  @rq-b6d22242
  Scenario: Unknown phase name is rejected at load time
    Given tmp/argon.in.toml declares phases named "equil" and "prod"
    And tmp/argon.in.analysis sets phase = "warmup"
    When load_analysis_config is called
    Then it returns Err(AnalyzeError::UnknownPhase {
      phase: "warmup", available: ["equil", "prod"] })

  @rq-2674f18a
  Scenario: Explicit `trajectory` overrides phase-derived default
    Given tmp/argon.in.analysis sets trajectory = "alt.xyz"
    When load_analysis_config is called
    Then config.trajectory equals canonical "tmp/alt.xyz"
    And the `phase` field is still resolved (defaults to last phase if omitted)
      so that builders that read phase-dependent metadata see the right value
```
