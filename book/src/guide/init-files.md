# Init Files (Extended XYZ)

The init file carries everything the runner needs about the *state* of the
system at `t = 0`: the particle count, the simulation box, per-particle
type names, positions, and (optionally) velocities and integer image
flags. Per-type properties such as mass and charge live in the
[TOML config](configuration.md), not the init file.

The format is a restricted subset of the extended-XYZ convention used by
ASE, OVITO, and VMD — restricted because the runner is strict about what
it will accept.

## File structure

```
N
key1=value1 key2="value with spaces" ...
<row 1>
<row 2>
...
<row N>
```

- **Line 1** — a non-negative integer `N`, the particle count.
- **Line 2** — the *comment line*: space-separated `key=value` attributes.
  Values may be double-quoted to embed spaces. Unknown keys are ignored.
- **Lines 3..N+2** — one data row per particle.

The file is UTF-8. Lines end in `\n` or `\r\n`. Blank lines after the
last data row are tolerated; non-blank trailing content is an error.

## Required attributes

### `Lattice`

A nine-element space-separated `f64` list in double quotes — the three
Cartesian box vectors in row-major order:

```
Lattice="lx 0 0 xy ly 0 xz yz lz"
```

The parser accepts only **lower-triangular** matrices: the three
upper-triangular slots (positions 2, 3, 6 — `a_y`, `a_z`, `b_z`) must be
exactly `0.0`. The orthorhombic case (`xy = xz = yz = 0`) is the common
form:

```
Lattice="8.0e-9 0 0 0 8.0e-9 0 0 0 1.0e-8"
```

The three diagonal entries (`lx`, `ly`, `lz`) must be finite and strictly
positive; the three tilt entries (`xy`, `xz`, `yz`) may be any finite
value. All values are in metres.

### `Properties`

A colon-separated column-layout spec. Exactly one of these four forms is
accepted, and the column order is fixed:

```
Properties=species:S:1:pos:R:3
Properties=species:S:1:pos:R:3:image:I:3
Properties=species:S:1:pos:R:3:velo:R:3
Properties=species:S:1:pos:R:3:velo:R:3:image:I:3
```

The `velo` and `image` blocks are independent — either may be present or
absent — but when present each one must supply all three components in
every row.

## Data rows

| Column   | Type | Units  | Notes |
|----------|------|--------|-------|
| `species`| str  | —      | must equal a `[[particle_types]].name` from the config |
| `pos`    | f64  | metres | three values; must lie in the primary image |
| `velo`   | f64  | m/s    | three values; optional |
| `image`  | i32  | —      | three values; optional |

Whitespace separates columns. Non-finite (NaN, ±Inf) values in any real
column are rejected. Integer columns that fail to parse as `i32` are
rejected.

### Positions must lie in the primary image

Every position must satisfy `s_a, s_b, s_c ∈ [-1/2, 1/2)` in fractional
coordinates. For an orthorhombic box this is
`pos_x ∈ [-lx/2, lx/2)` (and similarly for `y`, `z`) — the lower bound
is inclusive, the upper bound exclusive. A particle sitting exactly on
`+L/2` is rejected. The runner does not silently wrap out-of-cell
positions; if your generator can produce them, wrap or wrap-and-record
the image triple before writing the file.

### Particle IDs

Particle IDs are implicit: row 1 is particle `0`, row 2 is particle `1`,
through `N-1`. There is no explicit ID column.

### Image flags

When `image:I:3` is omitted, every particle's image triple defaults to
`(0, 0, 0)`. When present, the unwrapped position is
`pos + n_a · a + n_b · b + n_c · c`, which for an orthorhombic box
reduces to `pos + image · (lx, ly, lz)`.

## Velocities: file vs. config

- **Velocities present in the file** — the runner uses them verbatim,
  cast from `f64` to `f32`. `simulation.temperature` in the config is
  still required and validated, but it is ignored for velocity setup.
- **Velocities absent from the file** — the runner samples a
  Maxwell-Boltzmann distribution at `simulation.temperature` using a
  ChaCha8 RNG seeded by `simulation.seed`, subtracts the centre-of-mass
  momentum so the total momentum is zero, then rescales by a single
  scalar so the realised flat-3N temperature equals the configured
  target. The procedure is fully deterministic in
  `(seed, temperature, masses)`, so two runs with the same config and
  init file produce byte-identical starting velocities.

## Round-tripping with the trajectory

Trajectory frames written by the engine are valid init files. The frame
carries an extra `Step=` and `Time=` on the comment line (which the
init parser ignores as unknown attributes), but the lattice, properties,
and data rows match the init format exactly. To restart from frame `k`
of a previous run, extract that frame into its own file and point a
fresh config's `init` field at it.

## A minimal example

Two argon atoms, no velocities, in a 1 nm cubic box:

```
2
Lattice="1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9" Properties=species:S:1:pos:R:3
Ar  0.0      0.0  0.0
Ar  3.4e-10  0.0  0.0
```

A larger example lives at
`examples/lj-10000-argon/init.xyz`; the Python script next to it
regenerates it deterministically.

## What's *not* supported

- The crystallographic `a, b, c, α, β, γ` lattice syntax. Convert to
  the 9-component matrix yourself.
- Column types other than `species:S:1`, `pos:R:3`, `velo:R:3`,
  `image:I:3` — no masses, charges, forces, or arbitrary user columns
  in schema v1.
- Multi-frame init files. Only the first frame is read; trailing
  non-blank content after the last data row is an error.
- Gzip, bzip2, xz, or binary-format variants.
