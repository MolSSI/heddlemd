# Analysis (`heddlemd analyze`)

`heddlemd analyze` runs post-processing analyses on a trajectory
written by an earlier `heddlemd run`. The work is declared in a
`<root>.in.analysis` TOML file alongside the simulation inputs; each
analysis writes a CSV next to the input. v1 ships one built-in
analysis kind — the radial distribution function (`rdf`) — and an
open registry that lets custom builds add more without touching the
framework.

The subcommand is CPU-only, runs cheaply on a login node, and its
outputs are byte-identical across runs on identical inputs.

## Quick start

In a directory containing the simulation outputs from a previous
`heddlemd run` (`argon.in.toml`, `argon.in.xyz`,
`argon.out.run.xyz`), write a minimal `argon.in.analysis`:

```toml
schema_version = 1

[[analyses]]
name = "ar-ar"
kind = "rdf"
between = ["Ar", "Ar"]
r_max = 3.5e-9    # m; must satisfy r_max <= min_perp_width / 2
n_bins = 200
```

Then run:

```
heddlemd analyze argon.in.analysis
```

A successful invocation prints

```
[heddlemd] analyze complete: 1 analyses over 11 frames in <T> ms
```

and writes `argon.out.ar-ar.csv` next to the analysis file.

## The `<root>.in.analysis` file

### Filename convention

The path passed to `heddlemd analyze` must end in `.in.analysis`. The
derived root (filename with `.in.analysis` stripped) is used to
default the sibling simulation config and every analysis's output
path:

| Analysis filename       | `<root>` | Default sibling config | Default output for `name = "x"` |
|-------------------------|----------|------------------------|---------------------------------|
| `argon.in.analysis`     | `argon`  | `argon.in.toml`        | `argon.out.x.csv`               |
| `spc.in.analysis`       | `spc`    | `spc.in.toml`          | `spc.out.x.csv`                 |
| `run-01.in.analysis`    | `run-01` | `run-01.in.toml`       | `run-01.out.x.csv`              |

A filename whose `<root>` would be empty (e.g. `.in.analysis` alone)
is rejected.

### Implicit pairing

When the analysis file does not set `simulation` or `trajectory`
explicitly, the runner pairs with the sibling `<root>.in.toml` (must
exist) and selects a phase from that config. The `phase` field on the
analysis file picks the phase by name; when omitted, the **last**
phase in the config is used. The chosen phase's resolved
`output.trajectory_path` (which itself defaults to
`<root>.out.<phase>.xyz` per the
[Configuration Reference](configuration.md)) is then used as the
trajectory.

You can override either default explicitly:

```toml
schema_version = 1

# Analyse the equilibration phase rather than the default last phase.
phase = "equil"

# Or, analyse a trajectory from a different simulation directory.
simulation = "../other-run/other.in.toml"
trajectory = "../other-run/other.out.prod.xyz"

[[analyses]]
name = "ar-ar"
kind = "rdf"
between = ["Ar", "Ar"]
r_max = 3.0e-9
n_bins = 100
```

### Frame selection

Three optional top-level fields select which trajectory frames each
analysis consumes:

| Field         | Default        | Notes |
|---------------|----------------|-------|
| `first_frame` | `0`            | 0-based position of the first frame (skip equilibration). |
| `last_frame`  | last in file   | 0-based inclusive position of the last frame. |
| `stride`      | `1`            | Use every `stride`-th frame starting from `first_frame`. Must be `>= 1`. |

Frame positions count frames in the file, **not** the `Step=` value
on the comment line. `last_frame >= file_frame_count` is rejected at
trajectory-open time with `FrameOutOfRange`; `last_frame <
first_frame` is rejected at load time.

### `[[analyses]]` array

Each entry is a TOML table. Required common fields:

| Field         | Type        | Notes |
|---------------|-------------|-------|
| `name`        | string      | Identifier used in the default output filename. Non-empty, ASCII letters/digits/`-`/`_` only. Unique within the file. |
| `kind`        | string      | Registered analysis kind. v1 ships `"rdf"`. |
| `output_path` | string      | Optional. Overrides the default `<root>.out.<name>.csv`. |

Kind-specific fields follow on the same entry; see *RDF parameters*
below.

### Output naming

Default output is `<root>.out.<name>.csv` next to the analysis file.
The framework refuses to overwrite a pre-existing file at the
resolved output path — delete or move the old CSV between runs.

## RDF parameters

The `rdf` kind computes the radial distribution function
`g_AB(r)` between two particle types.

| Field     | Type        | Notes |
|-----------|-------------|-------|
| `between` | [string; 2] | Type-name pair from `[[particle_types]]`. Treated as unordered: `["A","B"]` and `["B","A"]` are equivalent. Same-type (`["A","A"]`) accepted. |
| `r_max`   | f64 (m)     | Maximum pair distance. Must satisfy `r_max <= sim_box.min_perpendicular_width() / 2` so the minimum-image convention assigns at most one image per pair. |
| `n_bins`  | u64         | Number of uniform bins in `[0, r_max]`. Bin `i` covers `[i·Δr, (i+1)·Δr)` with `Δr = r_max / n_bins`; reported at the centre `(i + 0.5)·Δr`. |

### Algorithm

Per consumed frame, the analysis enumerates unordered type-pair
distances in particle-index order (`i < j` for same-type; the
Cartesian product `A × B` for cross-type), applies the
minimum-image convention via the trajectory's `Lattice` attribute,
and increments the histogram bin containing each distance under
`r_max`. After the trajectory pass it converts the integer histogram
into `g_AB(r)` against an ideal-gas reference using exact shell
volumes and the constant box volume from the first frame.

### Output CSV

```
r,g_r,count
<r_0>,<g_0>,<count_0>
<r_1>,<g_1>,<count_1>
...
```

| Column   | Type | Units | Notes |
|----------|------|-------|-------|
| `r`      | f64  | m     | Bin centre `(i + 0.5)·Δr`. Formatted `{:.9e}`. |
| `g_r`    | f64  | —     | Normalised RDF value. Formatted `{:.9e}`. |
| `count`  | u64  | —     | Raw histogram count for the bin. |

Exactly `n_bins` data rows after the one-line header.

## Linting an `.in.analysis` file

`heddlemd lint` dispatches on file extension: pointing it at a
`.in.analysis` runs the *analyze lint pipeline*, which performs the
same four setup-phase stages `heddlemd analyze` would run but stops
before the trajectory pass and writes no files:

```
$ heddlemd lint argon.in.analysis
[heddlemd lint] OK
  config       argon.in.analysis
  output paths none pre-exist
  trajectory   resolved, 10000 particles, box 8.0e-9 × 8.0e-9 × 1.0e-8 m
  analyses     1 analysis builders validated
```

A failure surfaces the standard `FAIL — <reason>` line on the
offending stage plus an `error: <message>` line on stderr — same
shape as the simulation lint. Useful for catching geometric mistakes
(`r_max` too large) or output-path collisions on a login node before
queueing the analysis job.

## Reproducibility

Two `heddlemd analyze` runs on the same `.in.analysis`,
`.in.toml`, and trajectory produce byte-identical output CSVs. The
guarantee is unconditional on hardware in v1 (CPU-only). It rests on
deterministic pair enumeration order, integer histogram
accumulation, and fixed numeric formatting; see
`rqm/analysis/rdf.md` for the full argument.

## Out of scope

- GPU-accelerated analyses.
- Streaming output (each analysis writes its CSV once at end of pass).
- Variable-box trajectories (the first frame's lattice is taken as
  canonical).
- Cross-trajectory analysis or restart/append mode.
- Selecting frames by `Step=` value rather than file position.
