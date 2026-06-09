# Feature: Radial Distribution Function Analysis (`rdf`) <!-- rq-4d1082c4 -->

The `rdf` analysis kind computes the radial distribution function
`g_AB(r)` for an ordered or unordered pair of particle types `A`, `B`
declared in the simulation config. The histogram is accumulated over
every trajectory frame the framework dispatches to it (see
`rqm/analysis/framework.md` for frame selection and the
`Analysis` trait), then normalised against an ideal-gas reference
and written as a CSV file at end of pass.

This file describes the `rdf`-kind `[[analyses]]` entry in
`<root>.in.analysis`, the pair-enumeration and binning algorithm,
the normalisation convention, and the CSV output format.

## Schema <!-- rq-fb62c422 -->

A `[[analyses]]` entry of `kind = "rdf"` accepts the fields below
in addition to the framework-level `name`, `kind`, and
optional `output_path`.

```toml
[[analyses]]
name = "ar-ar"
kind = "rdf"
between = ["Ar", "Ar"]    # required, [String; 2]
r_max = 8.5e-10           # required, f64 (m)
n_bins = 100              # required, u64
```

### Field reference <!-- rq-4a6ec476 -->

- `between: [String; 2]` — required. Pair of particle-type names
  drawn from the loaded simulation config's `[[particle_types]]`.
  Both names must be declared types; unknown names produce
  `AnalysisRuntimeError::InvalidValue { field: "between", reason }`
  at builder-`build` time. The pair is treated as **unordered**:
  `["A", "B"]` and `["B", "A"]` produce identical output. Same-type
  pairs (`["A", "A"]`) are accepted and use the unordered-pair
  counting convention documented in *Algorithm* below.
- `r_max: f64` — required. Maximum pair distance considered, in
  Bohr (`a_0`). Finite and strictly positive. Must satisfy
  `r_max <= sim_box.min_perpendicular_width() / 2` so the minimum-
  image convention assigns at most one image per pair; violations
  produce `AnalysisRuntimeError::InvalidValue { field: "r_max",
  reason }` at builder-`build` time using the first-frame box.
- `n_bins: u64` — required. Number of uniform-width radial bins in
  `[0, r_max]`. Must satisfy `1 <= n_bins <= 2^30` (the upper
  bound prevents accidentally allocating a multi-GB histogram).
  The bin width is `Δr = r_max / n_bins`; bin `i` (zero-based)
  covers `[i · Δr, (i+1) · Δr)` and is reported at the centre
  `(i + 0.5) · Δr`.

Unknown fields for `kind = "rdf"` are rejected at
`validate_params` time as `AnalyzeError::Parse`.

## Algorithm <!-- rq-c9b62a3c -->

### Pair enumeration <!-- rq-cf285ed5 -->

For each consumed trajectory frame:

1. Read `type_indices`, `positions_x`, `positions_y`, `positions_z`
   from the frame.
2. Resolve `between` to a pair of type indices
   `(t_a, t_b)` looked up in the simulation config's
   `[[particle_types]]` array (left-to-right declaration order, so
   the index of `"Ar"` matches the index `dynamics run` assigns).
3. Build two index lists: `idx_a` containing every particle index
   `i` with `type_indices[i] == t_a`, and `idx_b` containing every
   index with `type_indices[i] == t_b`. Both lists are in
   ascending particle-index order (the natural iteration order
   over `0..particle_count`).
4. Enumerate unordered pairs:
   - **Same-type (`t_a == t_b`)**: iterate `for i in idx_a` and
     `for j in idx_a where j > i`. This is the canonical
     `i < j` upper-triangular enumeration.
   - **Cross-type (`t_a != t_b`)**: iterate `for i in idx_a` and
     `for j in idx_b`. Each `(i, j)` pair is visited exactly once
     because the two index lists are disjoint.
5. For each pair, compute the minimum-image displacement
   `Δ = sim_box.minimum_image(pos_j - pos_i)` and its squared
   distance `d2 = Δ.x² + Δ.y² + Δ.z²` in `f64` (positions are
   cast from `f32` to `f64` before subtraction).
6. If `d2 < r_max²`, compute `d = sqrt(d2)` in `f64`, find the bin
   index `bin = floor(d / Δr)`, clamp to `n_bins - 1` for the
   rare `d == r_max` case (a numeric edge that arises when
   `r_max² = d2` exactly under `f64`), and increment
   `histogram[bin]` by `1`.
7. Increment `frames_consumed` by `1` after the pair loop.

Pair enumeration is fully deterministic: the index lists are in
ascending particle-index order, the pair loop is the canonical
nested form, every numeric step uses standard IEEE `f64`
operations with no fused-multiply-add reliance, and the
histogram counts are integer increments.

### Normalisation <!-- rq-0806c514 -->

After the trajectory pass, the builder converts the integer
histogram into `g_AB(r)` using the simulation box's volume `V` from
the first frame (the framework assumes a constant box, see
`rqm/analysis/framework.md`).

For each bin `i` with centre `r_i = (i + 0.5) · Δr`:

- Compute the exact shell volume
  `V_shell[i] = (4π / 3) · ((r_{i+1})³ - (r_i')³)`
  where `r_{i+1} = (i + 1) · Δr` is the bin's outer radius and
  `r_i' = i · Δr` is the bin's inner radius. The exact form
  (rather than the small-Δr `4π r² Δr` approximation) is used so
  the first bin's normalisation is well defined.
- Compute the ideal-pair count for the bin:
  - **Same-type**: `ideal[i] = frames_consumed · (N_A · (N_A - 1) / 2) · V_shell[i] / V`.
  - **Cross-type**: `ideal[i] = frames_consumed · (N_A · N_B) · V_shell[i] / V`.
- `g_AB(r_i) = histogram[i] as f64 / ideal[i]` (`0.0` when
  `histogram[i] == 0`; the divisor is non-zero for every bin
  because `n_bins >= 1` and `V > 0`).

`N_A` and `N_B` are taken from the first frame and assumed
constant (the framework rejects analyses on trajectories whose
particle counts vary; see `rqm/analysis/framework.md`'s *Out of
Scope*).

### Reproducibility <!-- rq-7479022f -->

Two `dynamics analyze` runs on the same `.in.analysis`,
`.in.toml`, and trajectory produce byte-identical RDF CSVs. The
guarantee follows from:

- Deterministic pair enumeration order (particle-index ascending,
  upper-triangular).
- Pure integer histogram accumulation.
- `f64` `sqrt`, multiplications, and division (no FMA contraction
  on the CPU side; the implementation must avoid `fma` /
  `f64::mul_add` in the histogram-to-`g_r` reduction).
- Fixed numeric formatting (see *Output file format* below).
- Constant box convention (the first frame's `lengths()` are
  cached at builder-`build` time and used throughout).

The guarantee is unconditional on hardware in v1 because the
pipeline is CPU-only.

## Output file format <!-- rq-8a063042 -->

One CSV file per RDF analysis, written to the resolved
`output_path` (see `rqm/analysis/framework.md`'s *Output
convention*). UTF-8, `\n` line endings, ASCII-compatible.

### Header and rows <!-- rq-e591b51f -->

```
r,g_r,count
0.000000000e0,0.000000000e0,0
8.500000000e-12,1.230000000e-1,4
...
8.415000000e-10,9.876543210e-1,42
```

- First line: the literal header `r,g_r,count` (no leading
  whitespace, no quoting, no trailing comma).
- Subsequent lines: one row per bin in ascending-`i` order, with
  exactly `n_bins` data rows.

### Columns <!-- rq-964b59fd -->

- `r: f64` — bin centre `(i + 0.5) · Δr`, expressed in the unit system
  the writer was opened with (metres in `UnitSystem::Si`, Bohr in
  `UnitSystem::Atomic`); the engine computes `Δr` and the bin centres
  in Bohr and the writer applies the output-direction length
  conversion before formatting. Formatted with
  Rust's `{:.9e}` (nine fractional digits, lower-case `e` exponent).
- `g_r: f64` — `g_AB(r_i)`, dimensionless. Formatted with `{:.9e}`.
- `count: u64` — raw histogram count for the bin. Base-10
  integer, no padding.

Row count is exactly `n_bins`; header is exactly one line.

## Empty / degenerate cases <!-- rq-d6b9b019 -->

- `frames_consumed == 0` — the trajectory pass produced zero
  frames (selection emptied the file). The RDF writes its rows
  with `g_r = 0.0e0` for every bin and `count = 0` for every bin.
  The header row is still written.
- `N_A == 0` (or, for cross-type, `N_B == 0`) — the ideal-pair
  count is zero for every bin. The builder rejects this at
  `build` time with
  `AnalysisRuntimeError::InvalidValue { field: "between", reason:
  "type `X` has zero particles in the trajectory" }`; it does not
  silently produce an all-NaN CSV.
- `N_A == 1` for a same-type RDF — `N_A · (N_A - 1) / 2 == 0`, no
  pairs to enumerate. Rejected the same way.

## Feature API <!-- rq-2caa4efb -->

### Types <!-- rq-06a8b986 -->

- `RdfBuilder` — implements `AnalysisBuilder` (see <!-- rq-2dc76b67 -->
  `rqm/analysis/framework.md`).
  - `kind_name(&self) -> "rdf"`.
  - `validate_params(&self, params: &toml::Value) -> Result<(),
    AnalyzeError>` — checks that `between`, `r_max`, and `n_bins`
    are present, of the right type, and individually in range;
    rejects unknown fields. Does not consult the simulation
    config (so it can be called during `load_analysis_config`).
  - `build(&self, params: &toml::Value, header:
    &TrajectoryFrameHeader, sim_config: &Config) -> Result<Box<dyn
    Analysis>, AnalysisRuntimeError>` — resolves `between` against
    `sim_config.particle_types`, counts `N_A` and `N_B` against
    `header.type_indices`, computes the bin width, caches the
    box's volume, and verifies `r_max <=
    header.sim_box.min_perpendicular_width() / 2`. On success,
    returns a `RdfAnalysis` boxed as `dyn Analysis`.

- `RdfAnalysis` — per-run handle. Fields are private; the type <!-- rq-e0b5377f -->
  owns:
  - the two type indices `(t_a, t_b)` and a flag for the
    same-type case,
  - the cached `(N_A, N_B)` counts and box volume `V`,
  - the histogram (length `n_bins`, `Vec<u64>`),
  - the bin width `Δr` and `r_max` for re-use in
    `finalize_and_write`,
  - the running `frames_consumed: u64`.

  Methods:
  - `consume_frame(&mut self, frame: &TrajectoryFrame, sim_box:
    &SimulationBox) -> Result<(), AnalysisRuntimeError>` — runs
    the pair-enumeration loop documented under *Pair enumeration*.
    Returns `Other(_)` if a frame's `type_indices` does not
    contain the cached `(N_A, N_B)` counts (i.e. the trajectory's
    composition changed mid-run, which v1 does not support).
  - `finalize_and_write(&mut self, output_path: &Path,
    _sim_config: &Config) -> Result<(), AnalysisRuntimeError>` —
    computes the per-bin `g_AB`, opens the output path with
    `OpenOptions::new().write(true).create_new(true)`, writes
    the header + `n_bins` rows, and flushes the file. Returns
    `Io(String)` on filesystem failure.

### Registration <!-- rq-7da27d04 -->

`AnalysisRegistry::with_builtins()` registers exactly one builder
in v1: `RdfBuilder`. Custom builders compose via
`Registries::register_analysis` as described in
`rqm/analysis/framework.md`'s *Feature API*.

## Out of Scope <!-- rq-60c1e792 -->

- Three-particle correlation functions (`g_3`, angular RDFs).
- Time-dependent RDFs (van Hove function).
- Density-weighted or charge-weighted RDFs.
- Cumulative coordination number columns. Users compute these
  from the CSV.
- Box rescaling within a single trajectory. The first frame's
  box is the canonical box.
- `r_min > 0` lower cutoffs. The histogram always starts at `r =
  0`; users mask low-`r` bins in post-processing if desired.
- Logarithmic or otherwise non-uniform bin spacings. Bins are
  uniform on `[0, r_max]`.

## Gherkin Scenarios <!-- rq-9dd82f60 -->

```gherkin
Feature: Radial distribution function analysis

  Background:
    Given a temporary directory tmp
    And tmp/argon.in.toml is a valid one-type config declaring [[particle_types]] with name="Ar"
    And tmp/argon.out.xyz is a valid trajectory written by `dynamics run` of that config

  # --- Parameter validation ---

  @rq-cfd1d536
  Scenario: Reject an entry missing `between`
    Given an RDF [[analyses]] entry omitting `between`
    When validate_params is called
    Then it returns Err(AnalyzeError::MissingField { field: "between" })

  @rq-d4c17bd3
  Scenario: Reject `r_max` larger than half the box's minimum perpendicular width
    Given a 4 nm cubic box and an RDF entry with r_max = 3.0e-9
    When RdfBuilder::build is called against that header
    Then it returns Err(AnalysisRuntimeError::InvalidValue { field: "r_max", .. })

  @rq-5f1d5034
  Scenario: Reject `n_bins = 0`
    Given an RDF entry with n_bins = 0
    When validate_params is called
    Then it returns Err(AnalyzeError::InvalidValue { field: "n_bins", .. })

  @rq-ba2f07bd
  Scenario: Reject `between` referencing an undeclared type
    Given a one-type config declaring only "Ar" and an RDF with between=["Ar","Kr"]
    When RdfBuilder::build is called
    Then it returns Err(AnalysisRuntimeError::InvalidValue { field: "between", .. })

  @rq-40dd09ae
  Scenario: Reject `between` referencing a type with zero particles
    Given a config declaring "Ar" and "Kr" but a trajectory containing only Ar atoms
    And an RDF entry with between=["Kr","Kr"]
    When RdfBuilder::build is called
    Then it returns Err(AnalysisRuntimeError::InvalidValue { field: "between", .. })

  @rq-0b5e6b6c
  Scenario: `between` order does not matter
    Given two RDF entries with between=["Ar","Kr"] and between=["Kr","Ar"] respectively
    When both analyses run against the same trajectory
    Then the two output CSVs are byte-identical (modulo the differing `name`-derived filename)

  # --- Algorithm and output ---

  @rq-66f2679e
  Scenario: Output CSV has exactly `n_bins` data rows
    Given an RDF with n_bins=64
    When dynamics analyze runs to completion
    Then the output CSV has 65 lines total (header + 64 data rows)

  @rq-43567b30
  Scenario: First bin's `r` column equals 0.5 · Δr
    Given an RDF with r_max=1.0e-9 and n_bins=10
    When dynamics analyze runs to completion
    Then the first data row's `r` column equals 5.0e-11 within f64 round-off

  @rq-60c534f2
  Scenario: Last bin's `r` column equals (n_bins - 0.5) · Δr
    Given an RDF with r_max=1.0e-9 and n_bins=10
    When dynamics analyze runs to completion
    Then the last data row's `r` column equals 9.5e-10 within f64 round-off

  @rq-c505f34b
  Scenario: Same-type RDF on a one-particle frame yields zero counts
    Given a one-frame trajectory with a single Ar particle
    And an RDF with between=["Ar","Ar"]
    When RdfBuilder::build is called
    Then it returns Err(AnalysisRuntimeError::InvalidValue { field: "between", .. })

  @rq-9aacb1f9
  Scenario: Same-type RDF on a two-particle frame at exactly r_max - epsilon
    Given a one-frame trajectory with two Ar particles separated by 5.0e-10 m along x
    And an RDF with r_max=1.0e-9 and n_bins=10
    When dynamics analyze runs to completion
    Then the bin containing 5.0e-10 has count = 1
    And every other bin has count = 0

  @rq-8cec425d
  Scenario: Cross-type RDF on a two-particle frame
    Given a one-frame trajectory with one Ar and one Kr separated by 3.0e-10 m
    And an RDF with between=["Ar","Kr"], r_max=1.0e-9, n_bins=10
    When dynamics analyze runs to completion
    Then exactly one bin has count = 1 and every other bin has count = 0

  @rq-56306e1f
  Scenario: Frames at distances >= r_max do not contribute
    Given a one-frame trajectory with two Ar particles separated by 1.5e-9 m
    And an RDF with r_max=1.0e-9
    When dynamics analyze runs to completion
    Then every bin has count = 0

  @rq-17fd53d9
  Scenario: Histogram accumulates across multiple frames in order
    Given a three-frame trajectory with the same two Ar particles per frame
      and per-frame separations 3.0e-10, 4.0e-10, 5.0e-10
    And an RDF with r_max=1.0e-9 and n_bins=10
    When dynamics analyze runs to completion
    Then exactly three bins have count = 1 each (one per distance)
    And every other bin has count = 0

  # --- Normalisation ---

  @rq-c70f6309
  Scenario: Empty-bin g_r is exactly 0.0
    Given an RDF whose histogram has count = 0 in some bin
    When dynamics analyze runs to completion
    Then the corresponding `g_r` column value is exactly 0.0e0

  @rq-36665dda
  Scenario: g_r normalisation matches the ideal-gas reference
    Given a two-particle Ar/Ar trajectory whose distances sum to ideal-gas expectations
    When dynamics analyze runs to completion
    Then the resulting g_r values agree with the analytical normalisation
      g_r = (V * count) / (frames * N_A * (N_A - 1) / 2 * shell_volume)
      to within f64 round-off

  # --- Reproducibility ---

  @rq-8b41bc4d
  Scenario: Two `dynamics analyze` runs on the same inputs produce byte-identical CSVs
    Given a valid .in.analysis and trajectory
    When dynamics analyze is invoked twice
    Then the two output CSVs are byte-identical for every analysis

  # --- Output overwrite ---

  @rq-23678707
  Scenario: Refuse to overwrite an existing CSV at the resolved output path
    Given the resolved output_path already exists
    When dynamics analyze runs
    Then it exits with code 1
    And stderr contains "OutputExists"
    And the existing file is unchanged

  # --- Lint coverage ---

  @rq-e2cbe4fd
  Scenario: Lint reports `r_max` greater than half-box under the analyses stage
    Given an RDF entry with r_max greater than half the trajectory box's min perpendicular width
    When dynamics is invoked with arguments ["lint", "tmp/argon.in.analysis"]
    Then it exits with code 1
    And stdout has an "analyses" stage line beginning with "FAIL —"
    And stderr contains "r_max"
```
