# Output Files

Every successful run writes three files alongside the config:
`<root>.out.xyz`, `<root>.out.log`, and `<root>.out.timings`, where
`<root>` is the config's root derived per the
[Config filename convention](configuration.md#config-filename-convention)
(e.g. `argon.in.toml` → `argon.out.xyz`, `argon.out.log`,
`argon.out.timings`). Paths and cadences are controlled by the
[`[output]` section](configuration.md) of the TOML config.

The runner refuses to start when any of these files already exists at the
resolved path. Delete or move them before re-running. The check is done
up front, before the init file is read, so the runner fails fast.

## Trajectory file (`*.out.xyz`)

Extended-XYZ frames concatenated into a single file. Each frame is fully
self-describing — particle count, box, column layout, step, and time —
and matches the format the [init-file parser](init-files.md) accepts, so
any single frame can be lifted out and used to restart.

### Frame layout

```
N
Lattice="lx 0 0 xy ly 0 xz yz lz" Properties=species:S:1:pos:R:3[:velo:R:3][:image:I:3] Step=<u64> Time=<f64>
<row 1>
...
<row N>
```

- `Lattice` repeats verbatim in every frame even though the box does not
  change (it makes single-frame extraction easy).
- `Properties` is fixed at writer construction by
  `output.include_velocities` and `output.include_images`, and never
  varies within a file.
- `Step` is the integration-step index (`0` for the initial frame).
- `Time` is `step * dt` in seconds.

Positions are always written **wrapped** into the primary image. When
`include_images = true`, the per-particle integer image triple
`(images_x, images_y, images_z)` is appended to each row; the unwrapped
position is

```
pos + images_x · a + images_y · b + images_z · c
```

which reduces to `pos + image · (lx, ly, lz)` for an orthorhombic box.

### Cadence

A frame is written for the initial state (`Step=0`) plus one every
`trajectory_every` steps up to and including `n_steps`. Total frame
count when `trajectory_every > 0` is
`floor(n_steps / trajectory_every) + 1`. Setting `trajectory_every = 0`
disables trajectory output entirely (not even the step-0 frame is
written, and no file is created).

### Number formatting

Floats use Rust's `{:.9e}` formatter — nine fractional digits and a
lower-case `e` exponent, which round-trips every `f32` value exactly.

## Log file (`*.out.log`)

A plain CSV with a fixed four-column header followed by one row per log
interval. No comment characters, no quoting, no trailing summary — easy
to load with pandas or grep.

```
step,time,kinetic_energy,temperature
0,0.000000000e0,2.070973475e-17,9.999999878e1
5,5.000000000e-15,2.070582761e-17,9.998113260e1
...
```

| Column           | Type | Units | Notes |
|------------------|------|-------|-------|
| `step`           | u64  | —     | integration-step index; base-10 integer |
| `time`           | f64  | s     | `step * dt`, formatted `{:.9e}` |
| `kinetic_energy` | f64  | J     | `0.5 · Σ m_i (vx²+vy²+vz²)`, summed in particle-ID order |
| `temperature`    | f64  | K     | `2 · KE / (3 · N · k_B)`, `k_B = 1.380649e-23` |

### Extra columns

Some integrators append diagnostic columns. The Nosé-Hoover chain
thermostat, for example, adds `nhc_conserved`, giving a header of

```
step,time,kinetic_energy,temperature,nhc_conserved
```

When the chosen integrator declares no extras (the default for
`velocity-verlet` and `langevin-baoab`), the header is exactly the four
columns above.

### Temperature convention

The temperature column uses a **flat-3N** degrees-of-freedom convention.
This is exact for Langevin-thermostatted runs and for the initial
sampled velocities (which the runner rescales to this convention). For
an NVE run with centre-of-mass momentum removed, the per-thermal-DOF
equipartition temperature is `N/(N-1)` times this value — a difference
that vanishes for non-trivial system sizes.

### Cadence

A row is written for `Step=0` and then every `log_every` steps. Total
rows when `log_every > 0` is `floor(n_steps / log_every) + 1` plus the
one header line. Setting `log_every = 0` disables the log entirely (no
header, no file).

## Timings file (`*.out.timings`)

A fixed-width text table with one row per instrumented stage that
collected at least one sample. The runner times every kernel launch
with a pair of CUDA events on the default stream and every host stage
with `std::time::Instant`. There is no opt-out.

```
stage                             count       total_ms       mean_us      min_us      max_us
vv_kick_drift                       100          0.996          10.0         6.1       111.8
neighbor_displacement_squared        100          0.518           5.2         4.1        13.3
...
total_runtime                         1        460.234      460233.7    460233.7    460233.7
```

| Column     | Width | Meaning |
|------------|-------|---------|
| `stage`    | 28    | snake_case stage name, left-aligned |
| `count`    | 10    | sample count, right-aligned base-10 integer |
| `total_ms` | 14    | sum of all samples, in milliseconds (3 decimals) |
| `mean_us`  | 13    | mean per sample, in microseconds (1 decimal) |
| `min_us`   | 11    | minimum sample, in microseconds (1 decimal) |
| `max_us`   | 11    | maximum sample, in microseconds (1 decimal) |

Each row is exactly 92 columns wide followed by `\n`. A value that does
not fit its nominal width is written in full rather than truncated.

Stages that recorded zero samples are omitted from the file, so the
exact set of rows depends on which integrator, force slots, and
neighbor-list mode were active. The fixed row ordering is documented in
`rqm/performance-analysis.md`.

The `.timings` file is written **only on successful exit**, once, just
before the runner returns; the path is reserved at startup but no
partial data is left behind on failure.

### Why this file is not reproducible

Wall-clock measurements vary run-to-run for reasons that have nothing
to do with the simulation: GPU clocks, OS scheduling, driver state.
Mixing them into the deterministic outputs would silently break
reproducibility checks. They live in their own file precisely so a
`diff` of `*.out.xyz` and `*.out.log` against a reference run is a clean
yes/no answer. See [Reproducibility](reproducibility.md) for the full
guarantee.

## stdout

On success the runner prints one line:

```
[dynamics] complete: <N> steps in <T> ms (frames: <F>, log rows: <R>)
```

On failure it prints `error: <message>` on stderr and exits non-zero;
see [the CLI reference](../reference/cli.md) for exit codes.
