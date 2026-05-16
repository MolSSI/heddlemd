# Error Handling Convention <!-- rq-91ac50d0 -->

Every fallible operation in the engine returns `Result<T, E>`, where `E` is a
module-specific error type. All of these error types follow one convention so
that failures surface uniformly regardless of where they originate — config
parsing, a CUDA kernel launch, neighbor-list construction, or trajectory I/O.

The convention has four parts: every error type derives `thiserror::Error`;
its `Display` output is human-readable prose; error-to-error conversions are
declared with `#[from]`; and a variant that wraps another error exposes that
error through `std::error::Error::source`.

## Governed Error Types <!-- rq-44ce75ec -->

The following error types are returned across the engine's public API and all
conform to this convention. Each is canonically defined — with its full
variant list — in the requirements file for the feature that owns it.

- `GpuError` (`src/gpu/device.rs`) — a newtype wrapping
  `cudarc::driver::DriverError`; returned by every GPU operation.
- `ConfigError` (`src/io/config.rs`) — TOML config loading and validation.
- `InitStateError` (`src/io/init_state.rs`) — extended-XYZ init-file parsing.
- `TopologyFileError` (`src/forces/topology.rs`) — `.topology`-file parsing
  (bonds, angles, and exclusions).
- `ParticleStateError` (`src/state.rs`) — particle-state construction and
  host/device transfer.
- `SimulationBoxError` (`src/pbc.rs`) — simulation-box construction.
- `NeighborListError` (`src/forces/neighbor_list.rs`) — neighbor-list
  construction and rebuild.
- `ForceFieldError` (`src/forces/mod.rs`) — force-field assembly and the
  per-step force pipeline.
- `IntegratorError` (`src/integrator.rs`) — integrator construction and
  stepping.
- `TimingsError` (`src/timings.rs`) — CUDA-event timing instrumentation.
- `TimingsWriterError` (`src/timings.rs`) — timings-file output.
- `TrajectoryWriterError` (`src/io/trajectory.rs`) — trajectory-file output.
- `LogWriterError` (`src/io/log_output.rs`) — diagnostic-log output.
- `RunnerError` (`src/runner.rs`) — the top-level error of `run_simulation`;
  wraps every error type above.

## Trait Derivation <!-- rq-e1ceb5c0 -->

Every error type derives `thiserror::Error` together with `Debug`:

```rust
#[derive(Debug, thiserror::Error)]
pub enum ConfigError { /* variants */ }
```

The derive supplies the `std::fmt::Display` and `std::error::Error` impls. No
error type carries a hand-written `Display` or `Error` impl. The variant
names, fields, and payloads of each error enum are exactly those documented
in the owning feature's requirements file — the convention governs how the
traits are implemented, not the shape of the data.

## Display Messages <!-- rq-1bbcf3b7 -->

`Display` renders human-readable prose, never a transcription of the variant's
Rust structure. Every message follows one style:

- it begins with a lower-case letter and carries no trailing period;
- identifiers, file paths, and literal values are wrapped in backticks;
- dynamic values are interpolated inline, not appended as a debug-formatted
  tuple or struct.

Representative messages:

```
ConfigError::InvalidValue { field: "simulation.dt", reason: "must be finite and positive" }
  ->  invalid value for `simulation.dt`: must be finite and positive

NeighborListError::TooManyCells { n_cells_total: 4298942376, max_supported: 4294967295 }
  ->  cell grid has 4298942376 cells, exceeding the device limit of 4294967295

RunnerError::OutputExists { path: "/tmp/sim/sim-traj.xyz" }
  ->  output file already exists: `/tmp/sim/sim-traj.xyz`
```

## Conversions and Source Chains <!-- rq-6cf916af -->

Error-to-error conversions are declared with thiserror's `#[from]` attribute
on the wrapped field. A single annotation generates both the `From` impl and
the `source()` link:

- `GpuError` converts from `cudarc::driver::DriverError`.
- `ParticleStateError`, `NeighborListError`, `ForceFieldError`, and
  `IntegratorError` convert from `GpuError`.
- `ForceFieldError` additionally converts from `TimingsError` and
  `NeighborListError`.
- `IntegratorError` additionally converts from `TimingsError` and
  `ForceFieldError`.
- `TimingsError` converts from `GpuError`.

One conversion cannot be expressed as a single `#[from]` hop: turning a
`cudarc::driver::DriverError` into a `TimingsError` routes through `GpuError`.
That conversion is a hand-written `From` impl that coexists with the derived
ones.

`RunnerError` carries no `From` impls. The runner tags each error it lifts
with the execution phase it occurred in (see `simulation-runner.md`), so every
conversion into `RunnerError` is an explicit `.map_err` at the call site
rather than an implicit `?`. `RunnerError`'s wrapping variants still expose
their inner error through `source()`.

Every variant that wraps another error exposes it through
`std::error::Error::source`, so a caller can walk the whole cause chain — for
instance `RunnerError` → `ForceFieldError` → `NeighborListError`. A wrapping
variant that adds no information of its own also delegates its `Display` to
the inner error, so the inner message surfaces verbatim with no `Variant(...)`
decoration.

## Dependencies <!-- rq-6e136da6 -->

`Cargo.toml` declares `thiserror`. It is a compile-time-only proc-macro crate:
the `Display`, `Error`, and `From` impls are expanded during compilation, and
it links no runtime code or transitive runtime dependency. It is the one
dependency the engine takes purely for ergonomics; it is justified because
error reporting is cross-cutting — every module defines an error type — and
the convention would otherwise require repeating the same trait-impl
boilerplate in each.

## Out of Scope <!-- rq-745fda7c -->

- Redesigning error variants. Variant names, fields, and payloads are
  unchanged. Variants that hold a pre-formatted `String` (such as
  `ConfigError::Io(String)`, which carries an already-rendered I/O error
  message rather than a wrapped `std::io::Error`) keep that shape.
- The runner's mapping of errors to process exit codes — setup-phase failures
  to exit code `1`, loop-phase failures to exit code `2`. The convention
  governs trait impls and message text only; the exit-code mapping is
  specified in `simulation-runner.md`.
- A dynamic error type (`anyhow`-style) or a single crate-wide error enum.
  Each module keeps its own concrete, exhaustively-matchable error type.

---

## Gherkin Scenarios <!-- rq-fc0de81b -->

```gherkin
Feature: Error handling convention

  @rq-fdf7a255
  Scenario: thiserror is a declared dependency
    When Cargo.toml is parsed
    Then "thiserror" appears under [dependencies]

  @rq-494626a0
  Scenario: Every governed error type implements std::error::Error
    Given a generic function fn assert_error<E: std::error::Error + 'static>()
    When it is instantiated once for each of the 14 governed error types
    Then the crate compiles

  @rq-3298bdc5
  Scenario: ConfigError::InvalidValue renders as prose
    Given ConfigError::InvalidValue { field: "simulation.dt", reason: "must be finite and positive" }
    When it is formatted with Display
    Then the output equals "invalid value for `simulation.dt`: must be finite and positive"

  @rq-af191d10
  Scenario: NeighborListError::TooManyCells renders as prose
    Given NeighborListError::TooManyCells { n_cells_total: 4298942376, max_supported: 4294967295 }
    When it is formatted with Display
    Then the output equals "cell grid has 4298942376 cells, exceeding the device limit of 4294967295"

  @rq-77c04470
  Scenario: RunnerError::OutputExists renders as prose
    Given RunnerError::OutputExists { path: "/tmp/sim/sim-traj.xyz" }
    When it is formatted with Display
    Then the output equals "output file already exists: `/tmp/sim/sim-traj.xyz`"

  @rq-5d9d7f83
  Scenario: Display output is distinct from the Debug rendering
    Given the values ConfigError::InvalidValue { .. }, NeighborListError::TooManyCells { .. },
      and RunnerError::OutputExists { .. }
    When each is formatted with Display and with Debug
    Then the Display output differs from the Debug output for every value

  @rq-5d6085ba
  Scenario: A direct error-to-error conversion is generated by #[from]
    Given a NeighborListError::TooManyCells value
    When it is converted with Into into a ForceFieldError
    Then the result is ForceFieldError::NeighborList wrapping the original NeighborListError

  @rq-8abcd634
  Scenario: The DriverError-to-TimingsError conversion is retained
    Given a generic function fn assert_from<T: From<cudarc::driver::DriverError>>()
    When it is instantiated for TimingsError
    Then the crate compiles

  @rq-4f8e37af
  Scenario: A wrapped error is reachable through source()
    Given ForceFieldError::NeighborList wrapping a NeighborListError::TooManyCells
    When source() is called on the ForceFieldError
    Then it returns the wrapped NeighborListError

  @rq-7dd509c8
  Scenario: source() walks the full cause chain
    Given RunnerError::ForceField wrapping ForceFieldError::NeighborList wrapping a
      NeighborListError::TooManyCells
    When source() is walked from the RunnerError
    Then it yields the ForceFieldError, then the NeighborListError, then None

  @rq-244fceb1
  Scenario: A wrapping variant's Display delegates to the inner error
    Given RunnerError::ForceField wrapping ForceFieldError::NeighborList wrapping
      NeighborListError::TooManyCells { n_cells_total: 4298942376, max_supported: 4294967295 }
    When the RunnerError is formatted with Display
    Then the output equals "cell grid has 4298942376 cells, exceeding the device limit of 4294967295"
```
