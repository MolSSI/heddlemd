# Feature: Extended-XYZ Trajectory Output <!-- rq-2cca54cc -->

The runner appends one trajectory frame per snapshot interval (and one frame
for the initial state) to a single extended-XYZ file at the output path
declared in the config. Each frame is fully self-describing: it carries the
particle count, the simulation box, the column layout, the step index, the
simulation time, the per-particle type names, wrapped positions, the per-
particle integer image triple (when enabled), and per-particle velocities
(when enabled).

Frame format matches the file format read by `init-state-file.md`, so a
trajectory frame can be hand-edited and reused as an init file — including
restart from a recorded `(positions, images)` pair.

## File Format <!-- rq-1658f77d -->

A trajectory file is a concatenation of frames. Each frame has the structure:

```
N
Lattice="lx 0 0 xy ly 0 xz yz lz" Properties=species:S:1:pos:R:3[:velo:R:3] Step=<u64> Time=<f64>
<row 1>
<row 2>
...
<row N>
```

- Line 1: the particle count `N`. The runner writes the same value in every
  frame; it does not change during a simulation.
- Line 2: the comment line containing `Lattice`, `Properties`, `Step`, and
  `Time` attributes. Attribute order is fixed in this order to make the
  files easy to diff and grep.
- Data rows: one per particle, columns determined by `Properties`.

Lines end in `\n` (Unix line endings). The file is UTF-8.

### `Lattice` <!-- rq-c5518458 -->

`Lattice="lx 0 0 xy ly 0 xz yz lz"` where the nine values are the row-major
entries of the lower-triangular lattice matrix (row 1 = `a` vector, row 2
= `b` vector, row 3 = `c` vector) in metres. The three upper-triangular
slots (`a_y`, `a_z`, `b_z`) are always written as `0` to make the file
consumable by tools that expect a 9-component lattice. The orthorhombic
case (`xy = xz = yz = 0`) prints the three middle slots as zeros and is
indistinguishable from the v0 format. The runner writes a constant
lattice throughout a simulation; the `Lattice` attribute is repeated in
every frame for self-description.

Box components are written using Rust's `{:.9e}` formatter (9 fractional
digits, lower-case `e` exponent), which round-trips `f32` values exactly.

### `Properties` <!-- rq-e06bcfb0 -->

The Properties string is determined at writer construction from
`output.include_velocities` and `output.include_images` (see
`config-schema.md`). The four possible values are, in the same order
`init-state-file.md` accepts them:

- `Properties=species:S:1:pos:R:3` — neither velocities nor images.
- `Properties=species:S:1:pos:R:3:image:I:3` — images only.
- `Properties=species:S:1:pos:R:3:velo:R:3` — velocities only.
- `Properties=species:S:1:pos:R:3:velo:R:3:image:I:3` — both.

The value chosen at the start of a run is constant for every frame in the
file.

### `Step` <!-- rq-df244549 -->

A non-negative integer, the integration-step index at which the frame was
captured. The initial-conditions frame has `Step=0`. Subsequent frames
carry `Step=trajectory_every`, `Step=2*trajectory_every`, ..., up to the
last multiple of `trajectory_every` that is `<= n_steps`.

### `Time` <!-- rq-6ec75323 -->

A real number, the simulation time in seconds: `Time = step * dt`. Written
using `{:.9e}`. The initial frame carries `Time=0.0e0`. Identical to
`Step * dt` evaluated in `f64`.

### Data rows <!-- rq-00c68095 -->

The first column is the type **name** (the string declared in the config's
`[[particle_types]]`), not an integer index. Names are written verbatim
without quoting. Subsequent columns are real numbers written with `{:.9e}`,
separated by single spaces.

Positions are written in their current (live) wrapped value: each
particle lies inside the primary image of the simulation box (its
fractional coordinates are in `[-1/2, 1/2)³`; see `simulation-box.md`).
The integrator's drift kernels enforce this invariant on the device
state, so positions read out for the trajectory are always
already-wrapped. For an orthorhombic box this reduces to
`pos_x ∈ [-lx/2, lx/2)` etc.

Image columns (when `Properties` declares `image:I:3`) are the per-
particle integer image triple `(images_x[i], images_y[i],
images_z[i])` carried by `ParticleBuffers` (see `particle-state.md`),
counting lattice-vector crossings along the `a`, `b`, `c` directions.
The unwrapped position used by external analyses is
`pos + images_x · a + images_y · b + images_z · c`, which reduces to
`pos + image · (lx, ly, lz)` for an orthorhombic box.

Velocities, when included, are written in m/s.

## Cadence <!-- rq-74b5c137 -->

The runner writes one frame for the initial state (`Step=0`) plus one
frame at every step `s` such that `s % trajectory_every == 0` and
`1 <= s <= n_steps`. When `trajectory_every == 0`, no frames are written
(not even the step-0 frame). When `trajectory_every > n_steps`, only the
step-0 frame is written.

Total frame count when `trajectory_every > 0`:
`floor(n_steps / trajectory_every) + 1`.

## Feature API <!-- rq-22e4e198 -->

### Types <!-- rq-2196fc45 -->

- `TrajectoryWriter` — handle to an open trajectory file. Fields are <!-- rq-40a34caa -->
  private; the type encapsulates the buffered writer and the metadata
  needed to format each frame.

- `TrajectoryWriterError` — error type. Variants: <!-- rq-1fcaf334 -->
  - `OutputExists { path: PathBuf }` — `TrajectoryWriter::open` was called
    on a path that already exists.
  - `Io(String)` — underlying filesystem error during open or write.

### Functions and methods <!-- rq-3adef71d -->

- `TrajectoryWriter::open(path: &Path, include_velocities: bool, include_images: bool, type_names: Vec<String>) -> Result<TrajectoryWriter, TrajectoryWriterError>` <!-- rq-28659fbe -->
  - Creates the output file at `path`. If the file already exists, returns
    `OutputExists { path }`. The check and create are performed atomically
    via `OpenOptions::new().write(true).create_new(true)`.
  - Wraps the file in a buffered writer.
  - Stores `include_velocities`, `include_images`, and `type_names` so
    future `write_frame` calls produce the correct columns.
    `type_names[i]` is the string used to render particles whose
    `type_index` equals `i`.
  - Returns the constructed writer on success.

- `TrajectoryWriter::write_frame(&mut self, step: u64, dt: f64, box: &SimulationBox, type_indices: &[u32], positions_x: &[f32], positions_y: &[f32], positions_z: &[f32], velocities: Option<(&[f32], &[f32], &[f32])>, images: Option<(&[i32], &[i32], &[i32])>) -> Result<(), TrajectoryWriterError>` <!-- rq-be899bef -->
  - Writes one frame to the underlying file in the format described above.
  - Asserts in debug builds that all slice lengths agree (`type_indices`,
    `positions_*`, each `velocities` slice, and each `images` slice).
  - When `self.include_velocities == true`, the caller must supply
    `velocities = Some(_)`; when `false`, the caller must supply
    `velocities = None`. Mismatch is a debug assertion (programming
    error from the runner).
  - When `self.include_images == true`, the caller must supply
    `images = Some(_)`; when `false`, the caller must supply
    `images = None`. Mismatch is a debug assertion.
  - Looks up each particle's type name via
    `self.type_names[type_indices[i] as usize]`. Out-of-range indices are
    a debug assertion.
  - Returns `Io(_)` on filesystem write failure.

- `TrajectoryWriter::flush(&mut self) -> Result<(), TrajectoryWriterError>` <!-- rq-2ad32a7b -->
  - Flushes the internal buffer to disk. Called by the runner at the end
    of the simulation and after the last frame. May be called more
    frequently for crash-resilience without affecting correctness.

`TrajectoryWriter` implements `Drop` which best-effort flushes on drop
without panicking; programs that need crash-safe trajectories call
`flush` explicitly and check the result.

### Number formatting <!-- rq-88ec92fc -->

- Real numbers use Rust's `{:.9e}` (e.g. `3.400000095e-10` for the
  `f32` nearest neighbor of `3.4e-10`). This is the minimum precision
  that round-trips every `f32` value via the same parser. The trailing
  zero in the exponent is not suppressed.
- Integers are written in base 10 without padding.
- Strings (type names) are written verbatim with no escaping. Type names
  are guaranteed by the config loader to contain no whitespace.

## Out of Scope <!-- rq-15b050fb -->

- Binary trajectory formats (NetCDF, HDF5, AMBER `.nc`, etc.).
- Compression on the fly (gzip, xz).
- Multi-stream output (one file per particle type, etc.).
- Emitting unwrapped positions in the `pos:R:3` columns. Positions are
  always wrapped into the primary image; consumers that need unwrapped
  coordinates request `output.include_images = true` and compute
  `pos + images_x · a + images_y · b + images_z · c` themselves, or do
  the equivalent at parse time.
- Per-frame box changes (the box is constant for the run; NPT and box
  rescaling are not part of this feature).
- Velocity components on the data line when
  `include_velocities == false`. Either every frame carries velocities
  or none do; the choice is fixed at writer construction. The same
  all-or-nothing rule applies to image columns.
- Embedded run metadata (engine version, config hash, seed) in the
  trajectory header. Captured by `simulation-runner.md` only in stdout
  output for now; a future feature may add a separate `.meta` file.
- A separate energy or temperature attribute on the comment line; those
  go to the log file.

---

## Gherkin Scenarios <!-- rq-0ec0f65d -->

```gherkin
Feature: Extended-XYZ trajectory output

  Background:
    Given a temporary directory tmp
    And a SimulationBox lx=ly=lz=1.0e-9
    And type_names = ["Ar"]

  # --- Open and overwrite policy ---

  @rq-a403f778
  Scenario: Open creates a new trajectory file
    Given tmp/traj.xyz does not exist
    When TrajectoryWriter::open(tmp/traj.xyz, include_velocities=true, include_images=true, type_names) is called
    Then it returns Ok(writer)
    And tmp/traj.xyz exists and has length 0

  @rq-8f31cb78
  Scenario: Open refuses to overwrite an existing file
    Given tmp/traj.xyz exists with any contents
    When TrajectoryWriter::open(tmp/traj.xyz, include_velocities=true, include_images=true, type_names) is called
    Then it returns Err(TrajectoryWriterError::OutputExists { path: tmp/traj.xyz })
    And tmp/traj.xyz is unchanged

  @rq-17666e4f
  Scenario: Open fails when the parent directory does not exist
    Given tmp/missing/ does not exist
    When TrajectoryWriter::open(tmp/missing/traj.xyz, ...) is called
    Then it returns Err(TrajectoryWriterError::Io(_))

  # --- Frame format: positions only ---

  @rq-9021ec4b
  Scenario: Write a single frame without velocities and without images
    Given a writer opened with include_velocities=false and include_images=false
    And type_indices=[0, 0], positions_x=[0.0, 3.4e-10], positions_y=[0.0, 0.0], positions_z=[0.0, 0.0]
    When writer.write_frame(step=0, dt=1.0e-15, &box, type_indices, positions_x, positions_y, positions_z, None, None) is called
    And writer.flush() is called
    Then the file contains exactly:
      """
      2
      Lattice="1.000000000e-9 0.000000000e0 0.000000000e0 0.000000000e0 1.000000000e-9 0.000000000e0 0.000000000e0 0.000000000e0 1.000000000e-9" Properties=species:S:1:pos:R:3 Step=0 Time=0.000000000e0
      Ar 0.000000000e0 0.000000000e0 0.000000000e0
      Ar 3.400000095e-10 0.000000000e0 0.000000000e0
      """

  # --- Frame format: with velocities ---

  @rq-c5e00a28
  Scenario: Write a single frame with velocities and without images
    Given a writer opened with include_velocities=true and include_images=false
    And type_indices=[0], positions=(0,0,0), velocities=(100.0, 0.0, 0.0)
    When writer.write_frame(step=10, dt=1.0e-15, &box, ..., Some((vx,vy,vz)), None) is called
    And writer.flush() is called
    Then the file contains exactly one frame with N=1
    And the comment line contains 'Properties=species:S:1:pos:R:3:velo:R:3'
    And the comment line contains 'Step=10'
    And the comment line contains 'Time=1.000000000e-14'
    And the data row equals "Ar 0.000000000e0 0.000000000e0 0.000000000e0 1.000000000e2 0.000000000e0 0.000000000e0"

  # --- Multi-frame ---

  @rq-fd593357
  Scenario: Append frames in order
    Given a writer opened with include_velocities=false
    When writer.write_frame is called three times with step=0, step=10, step=20
    And writer.flush() is called
    Then the file contains three frames in that order
    And the first frame's comment line contains 'Step=0'
    And the second frame's comment line contains 'Step=10'
    And the third frame's comment line contains 'Step=20'

  # --- Empty particle state ---

  @rq-f5e94e6b
  Scenario: Write a frame for an empty state
    Given a writer opened with include_velocities=false and empty type_indices
    When writer.write_frame(step=0, ...) is called with empty slices
    And writer.flush() is called
    Then the file contains exactly:
      """
      0
      Lattice="1.000000000e-9 0.000000000e0 0.000000000e0 0.000000000e0 1.000000000e-9 0.000000000e0 0.000000000e0 0.000000000e0 1.000000000e-9" Properties=species:S:1:pos:R:3 Step=0 Time=0.000000000e0
      """

  # --- Type-name resolution ---

  @rq-f76b6cde
  Scenario: Render multiple type names from indices
    Given type_names = ["Ar", "Kr"]
    And type_indices = [0, 1, 1, 0]
    When writer.write_frame is called
    Then the data rows have first columns "Ar", "Kr", "Kr", "Ar" in that order

  # --- Round-trip via the init parser ---

  @rq-70e9fd38
  Scenario: A trajectory frame can be re-parsed by load_init_state
    Given a writer opened with include_velocities=true
    And a frame written for N=4 particles with non-trivial positions and velocities
    When writer.flush() is called
    And load_init_state(file_path, &type_names) is called on the same file
    Then it returns Ok(state)
    And state.particle_count equals 4
    And state.positions_x, state.positions_y, state.positions_z agree byte-for-byte with the values passed to write_frame
    And state.velocities.unwrap() agrees byte-for-byte with the values passed to write_frame

  # --- Numeric precision ---

  @rq-ddec3d72
  Scenario: f32 positions round-trip through the writer and parser
    Given an arbitrary f32 position p
    When writer.write_frame is called with positions_x=[p]
    And load_init_state is called on the resulting file
    Then the parsed positions_x[0] equals p byte-for-byte after casting back to f32

  # --- Flush semantics ---

  @rq-e7ddefaf
  Scenario: Flush is idempotent
    Given a writer that has written one frame
    When writer.flush() is called twice
    Then it returns Ok(()) both times

  @rq-03ff6434
  Scenario: Drop best-effort flushes
    Given a writer that has written one frame
    When the writer is dropped without calling flush
    Then the file contains the written frame after the drop completes

  # --- Image columns ---

  @rq-b8463a3b
  Scenario: Frame Properties carries image:I:3 when include_images=true and include_velocities=false
    Given a writer opened with include_velocities=false and include_images=true
    When writer.write_frame is called with images=Some(([1, -2], [0, 3], [-4, 0]))
    And writer.flush() is called
    Then the comment line contains 'Properties=species:S:1:pos:R:3:image:I:3'

  @rq-3df3a993
  Scenario: Frame Properties carries velo:R:3:image:I:3 when both flags are true
    Given a writer opened with include_velocities=true and include_images=true
    When writer.write_frame is called with both velocities=Some(_) and images=Some(_)
    And writer.flush() is called
    Then the comment line contains 'Properties=species:S:1:pos:R:3:velo:R:3:image:I:3'

  @rq-5395785a
  Scenario: Image columns appear after pos (and after velo when present) in each data row
    Given a writer opened with include_velocities=true and include_images=true
    And one particle with pos=(0.1, 0.2, 0.3), velo=(1.0, 2.0, 3.0), image=(4, -5, 6)
    When writer.write_frame is called and the file is flushed
    Then the data row equals
      "Ar 1.000000015e-1 2.000000030e-1 3.000000119e-1 1.000000000e0 2.000000000e0 3.000000000e0 4 -5 6"
      (with positions cast to f32 and integers rendered in base 10)

  @rq-7e6a503c
  Scenario: Round-trip through load_init_state preserves image flags
    Given a writer opened with include_velocities=true and include_images=true
    And a frame written for N=4 with non-trivial positions, velocities, and image flags
    When writer.flush() is called
    And load_init_state(file_path, &type_names) is called on the same file
    Then it returns Ok(state)
    And state.images.unwrap() agrees with the values passed to write_frame component-by-component

  @rq-48d14580
  Scenario: Positions written by the writer are inside the primary cell
    Given a writer opened with include_velocities=false and include_images=true
    And a ParticleBuffers whose positions are inside the primary image of
      the simulation box (the invariant maintained by the integrator
      drift kernels)
    When writer.write_frame is called
    Then every row's fractional coordinates lie in [-0.5, 0.5)³
    And for an orthorhombic box every pos_a value lies in [-L_a/2, +L_a/2)

  @rq-ef891f9c
  Scenario: Triclinic Lattice attribute carries non-zero tilts
    Given a writer opened on a frame produced by a SimulationBox with
      lx=1.0e-9, ly=1.0e-9, lz=1.0e-9, xy=0.2e-9, xz=0.1e-9, yz=-0.3e-9
    When writer.write_frame is called and the file is flushed
    Then the comment-line Lattice attribute equals
      'Lattice="1.000000000e-9 0.000000000e0 0.000000000e0 2.000000030e-10 1.000000000e-9 0.000000000e0 1.000000015e-10 -3.000000119e-10 1.000000000e-9"'
      (the three upper-triangular slots are exactly 0.0e0; the three
       tilts are the f32 representations of the construction-time
       values)

  @rq-ce15e04e
  Scenario: Round-trip through load_init_state preserves a triclinic lattice
    Given a writer opened on a frame produced by a triclinic SimulationBox
    When writer.write_frame is called and the file is flushed
    And load_init_state is called on the same file
    Then it returns Ok(state)
    And state.box.lattice() agrees with the write-side lattice byte-for-byte
      after casting back to f32
```
