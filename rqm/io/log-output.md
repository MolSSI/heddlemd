# Feature: CSV Diagnostic Log <!-- rq-965c504d -->

The runner writes a CSV log alongside the trajectory containing per-snapshot
diagnostic quantities. The log captures step index, simulation time, total
kinetic energy, and instantaneous temperature, sampled at the cadence
declared by `output.log_every` in the config (`config-schema.md`).

The log is intended to be greppable, pandas-friendly, and human-readable —
no header decoration, no comment characters, no trailing summary.

## File Format <!-- rq-1ddf84f1 -->

A standard RFC-4180-style CSV file:

```
step,time,kinetic_energy,temperature
0,0.000000000e0,0.000000000e0,3.000000000e2
100,1.000000000e-13,4.123456789e-21,2.987654321e2
200,2.000000000e-13,4.234567890e-21,2.945678901e2
...
```

- First line: header row with the exact field names
  `step,time,kinetic_energy,temperature`. No spaces, no quoting.
- Subsequent lines: one row per snapshot. Fields separated by single
  commas, no quoting, no spaces.
- Line endings: `\n` (Unix).
- File encoding: UTF-8 (ASCII-compatible).

### Field semantics <!-- rq-f4750851 -->

- `step: u64` — integration-step index at which the diagnostic was
  captured. The initial state has `step=0`; subsequent rows carry
  `log_every`, `2*log_every`, ..., up to the last multiple of `log_every`
  that is `<= n_steps`.
- `time: f64` — simulation time in seconds, computed as `step * dt`.
- `kinetic_energy: f64` — total kinetic energy in joules, computed as
  `0.5 * sum_i(m_i * (v_xi^2 + v_yi^2 + v_zi^2))`. The sum runs over all
  particles in source-array order (i.e. by particle ID). Masses are taken
  from the device-side `masses` buffer (downloaded once into host memory
  by the runner; subsequent log writes reuse the cached host copy).
- `temperature: f64` — instantaneous temperature in kelvin, computed as
  `T = 2 * kinetic_energy / (3 * N * k_B)` using the CODATA-2019 value
  `k_B = 1.380649e-23 J/K`. When `N == 0`, temperature is written as
  `0.0e0` (zero by definition rather than NaN).

### Number formatting <!-- rq-4a6969aa -->

- `step` is written in base 10 without padding.
- `time`, `kinetic_energy`, `temperature` are written using Rust's
  `{:.9e}` formatter (e.g. `1.234567890e-13`). The trailing zero in the
  exponent is not suppressed.

### Cadence <!-- rq-606197d5 -->

The runner writes one row for the initial state (`step=0`) plus one row
at every step `s` such that `s % log_every == 0` and
`1 <= s <= n_steps`. When `log_every == 0`, no rows are written (not even
the step-0 row, and the header row is not written either; the file is not
created — see *Disabled-output behaviour* in `simulation-runner.md`).

Total row count when `log_every > 0`:
`floor(n_steps / log_every) + 1`, plus the one header line.

### Empty simulation <!-- rq-72c57874 -->

When the runner exits without ever stepping (`n_steps == 0`), the log
contains the header line plus the one step-0 row, provided `log_every > 0`.

## Feature API <!-- rq-7a26eeae -->

### Types <!-- rq-c0aa3b5c -->

- `LogWriter` — handle to an open log file. Fields are private; the type <!-- rq-2344fcec -->
  encapsulates the buffered writer.

- `LogWriterError` — error type. Variants: <!-- rq-45eb243b -->
  - `OutputExists { path: PathBuf }` — `LogWriter::open` was called on a
    path that already exists.
  - `Io(String)` — underlying filesystem error.

### Functions and methods <!-- rq-8b4243e0 -->

- `LogWriter::open(path: &Path) -> Result<LogWriter, LogWriterError>` <!-- rq-e0ef1221 -->
  - Creates the file at `path`. If the file already exists, returns
    `OutputExists { path }`. The check and create are performed atomically
    via `OpenOptions::new().write(true).create_new(true)`.
  - Writes the header line `step,time,kinetic_energy,temperature\n`
    immediately.
  - Returns the writer on success.

- `LogWriter::write_row(&mut self, step: u64, time: f64, kinetic_energy: f64, temperature: f64) -> Result<(), LogWriterError>` <!-- rq-e409ce75 -->
  - Writes one CSV row in the format described above, terminated by `\n`.
  - Does not flush; the caller flushes via `flush` or relies on `Drop`.
  - Returns `Io(_)` on filesystem write failure.

- `LogWriter::flush(&mut self) -> Result<(), LogWriterError>` <!-- rq-925e5583 -->
  - Flushes the internal buffer.

- `compute_kinetic_energy(masses: &[f32], vx: &[f32], vy: &[f32], vz: &[f32]) -> f64` <!-- rq-6e51f09c -->
  - Free helper. Returns
    `0.5 * sum_i(masses[i] as f64 * (vx[i] as f64 * vx[i] as f64 + vy[i] as f64 * vy[i] as f64 + vz[i] as f64 * vz[i] as f64))`.
  - The summation order is fixed left-to-right by particle index; this
    is what makes the log bit-wise reproducible across runs on the same
    GPU even though IEEE addition is not associative.
  - Debug-asserts that all four slices have the same length.
  - Returns `0.0_f64` when the slices are empty.

- `compute_temperature(kinetic_energy: f64, particle_count: usize) -> f64` <!-- rq-46a39249 -->
  - Free helper. Returns `0.0_f64` when `particle_count == 0`. Otherwise
    returns `2.0 * kinetic_energy / (3.0 * particle_count as f64 * 1.380649e-23_f64)`.

`LogWriter` implements `Drop` which best-effort flushes on drop without
panicking.

## Out of Scope <!-- rq-9f080e19 -->

- Per-particle quantities (per-atom velocities, displacements, energies).
- Potential energy and total energy. Computing PE requires a deterministic
  pair-energy evaluator, which is a separate planned feature. With only
  KE, `total_energy` would equal `kinetic_energy` and adds no signal, so
  it is omitted.
- Pressure, virial tensor, stress.
- Conservation-drift columns (e.g. `dE/E_initial`).
- Per-type breakdowns of any quantity.
- Reduced-precision output for size; the log is small enough that f64
  precision is the right default.
- Binary (Parquet, Arrow) variants.
- Header comments with simulation metadata; the log is a pure data file.

---

## Gherkin Scenarios <!-- rq-1e7ef382 -->

```gherkin
Feature: CSV diagnostic log

  Background:
    Given a temporary directory tmp

  # --- Open and overwrite policy ---

  @rq-6d087460
  Scenario: Open creates a new log file with the header line
    Given tmp/run.log does not exist
    When LogWriter::open(tmp/run.log) is called
    And the writer is flushed
    Then it returns Ok(writer)
    And tmp/run.log exists
    And tmp/run.log contains exactly the bytes "step,time,kinetic_energy,temperature\n"

  @rq-f20e017d
  Scenario: Open refuses to overwrite an existing file
    Given tmp/run.log exists with any contents
    When LogWriter::open(tmp/run.log) is called
    Then it returns Err(LogWriterError::OutputExists { path: tmp/run.log })
    And tmp/run.log is unchanged

  @rq-9baf16d1
  Scenario: Open fails when the parent directory does not exist
    Given tmp/missing/ does not exist
    When LogWriter::open(tmp/missing/run.log) is called
    Then it returns Err(LogWriterError::Io(_))

  # --- Row format ---

  @rq-90517bb6
  Scenario: Write a single row at step 0
    Given a freshly opened writer
    When writer.write_row(0, 0.0, 0.0, 300.0) is called
    And writer.flush() is called
    Then the file contains:
      """
      step,time,kinetic_energy,temperature
      0,0.000000000e0,0.000000000e0,3.000000000e2
      """

  @rq-9198cc8e
  Scenario: Write a row at step 100 with non-trivial values
    Given a freshly opened writer
    When writer.write_row(100, 1.0e-13, 4.123456789e-21, 298.7654321) is called
    And writer.flush() is called
    Then the last line of the file equals "100,1.000000000e-13,4.123456789e-21,2.987654321e2"

  @rq-3ef10542
  Scenario: Append rows in source order
    Given a freshly opened writer
    When writer.write_row is called three times with step=0, step=100, step=200
    And writer.flush() is called
    Then the file has 4 lines (header + 3 rows)
    And the rows appear in step order 0, 100, 200

  # --- Kinetic energy helper ---

  @rq-107a7187
  Scenario: KE of a single particle at rest is zero
    Given masses=[1.0_f32], vx=[0.0], vy=[0.0], vz=[0.0]
    When compute_kinetic_energy is called
    Then it returns 0.0

  @rq-7c23d271
  Scenario: KE of a single particle with v=(1, 0, 0)
    Given masses=[2.0_f32], vx=[1.0_f32], vy=[0.0_f32], vz=[0.0_f32]
    When compute_kinetic_energy is called
    Then it returns 1.0_f64

  @rq-553f28a3
  Scenario: KE of three particles is summed in particle order
    Given masses=[1.0, 2.0, 4.0]
    And vx=[1.0, 1.0, 1.0], vy=[0.0, 0.0, 0.0], vz=[0.0, 0.0, 0.0]
    When compute_kinetic_energy is called
    Then it returns (0.5 + 1.0) + 2.0 as f64 (left-to-right)

  @rq-1feec66c
  Scenario: KE is bit-identical across two host runs with identical f32 inputs
    Given identical masses, vx, vy, vz slices
    When compute_kinetic_energy is called twice
    Then both results agree byte-for-byte

  @rq-fa6f7414
  Scenario: KE of empty input is zero
    Given empty masses, vx, vy, vz slices
    When compute_kinetic_energy is called
    Then it returns 0.0_f64

  # --- Temperature helper ---

  @rq-2a9acb69
  Scenario: Temperature of an empty system is zero
    When compute_temperature(0.0, 0) is called
    Then it returns 0.0

  @rq-4518fa47
  Scenario: Temperature uses k_B = 1.380649e-23
    Given KE = 1.5 * N * k_B * T_target for T_target=300 K and N=10
    When compute_temperature(KE, 10) is called
    Then it returns 300.0 within an absolute tolerance of 1.0e-12

  # --- Empty-simulation edge case ---

  @rq-1b97afd8
  Scenario: Log contains header plus step-0 row when n_steps = 0
    Given a runner has called LogWriter::open and write_row exactly once with step=0
    When the writer is flushed
    Then the file has exactly 2 lines (header + step-0 row)

  # --- Flush semantics ---

  @rq-9d0ea87b
  Scenario: Flush is idempotent
    Given a writer that has written one row
    When writer.flush() is called twice
    Then it returns Ok(()) both times

  @rq-02bde767
  Scenario: Drop best-effort flushes
    Given a writer that has written one row
    When the writer is dropped without calling flush
    Then the file contains the written row after the drop completes
```
