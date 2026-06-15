# Feature: f64 Precision Compile-Time Feature Flag <!-- rq-6e828da5 -->

HeddleMD's canonical floating-point storage and compute type is `Real`. `Real`
is selected at compile time by a Cargo feature: when the `f64` feature is off
(the default), `Real` resolves to `f32` and CUDA `Real` to `float`; when the
`f64` feature is on, `Real` resolves to `f64` and CUDA `Real` to `double`. A
single build always uses one precision throughout the engine; the feature
flag never produces runtime branches and the default build pays no
abstraction cost relative to a hand-written `f32` engine.

The feature widens positions, velocities, forces, masses, charges, energies,
virials, pair-buffer slots, neighbor displacement scratch buffers, integrator
scratch buffers, kinetic-energy reductions, Nose–Hoover / MTK / barostat
state, Langevin / Andersen / CSVR / c-rescale random-number outputs, SPME
spread / influence / gather grids, SPME `cufft` plans, SHAKE Lagrange-
multiplier scratch buffers, every kernel parameter that is currently `float`,
and every constant or numeric literal consumed by those kernels. It does not
widen integer types: image flags stay `i32`, particle IDs stay `u32`, type
indices stay `u32`, step counters stay `u64`, neighbor counts and cell
indices stay `u32`, and the host-side `dt` / `time` carried by trajectory and
log writers stays `f64` regardless of the feature (these are I/O-boundary
scalars, not engine storage).

The engine's `f32` round-off behaviour, IEEE-754 FMA contraction settings,
deterministic neighbor-list ordering, deterministic reduction topology, and
the bit-wise GPU-vs-GPU reproducibility guarantee are unchanged. A single
build remains bit-exact run-to-run on the same GPU. Cross-build comparison
(f32 build vs f64 build) is not a reproducibility property the engine
promises.

## Reading Existing Requirements <!-- rq-08887654 -->

Throughout the requirements directory, type references of the form `f32`,
`Vec<f32>`, `&[f32]`, `CudaSlice<f32>`, and `float` (in CUDA fragments) on
quantities listed in the feature description above are governed by this
file: they are `Real`, which resolves to `f32` in the default build and
`f64` under the `f64` feature. References to `f32` on the I/O-boundary
scalars enumerated above (`dt`, `time`, lattice-print precision in test
expectations) are precision-policy-independent and continue to mean `f32`
or `f64` exactly as written.

Existing Gherkin scenarios that assert byte-for-byte equality on `Vec<f32>`
fields likewise read as byte-for-byte equality on `Vec<Real>` fields; the
assertion holds independently in each build.

## Feature API <!-- rq-45ebbc76 -->

### Cargo feature <!-- rq-3d83d0a4 -->

`Cargo.toml` declares:

```toml
[features]
default = []
f64 = []
```

The `f64` feature is a marker feature: it carries no transitive
dependencies. Builds are selected with `cargo build` (default, `Real = f32`)
or `cargo build --features f64` (`Real = f64`). The two are mutually
exclusive within a single build artefact; the feature flag has no
runtime form.

### Rust precision module <!-- rq-89c517aa -->

`src/precision.rs` is the canonical home of the Rust-side `Real` alias and
the host-side math shim layer.

- `pub type Real` <!-- rq-9edae17f -->
  - `f32` when the `f64` feature is off.
  - `f64` when the `f64` feature is on.
- `pub const REAL_BYTES: usize = std::mem::size_of::<Real>();` <!-- rq-182dd348 -->
  - `4` in the default build, `8` under the `f64` feature.
- `pub const REAL_IS_F64: bool;` <!-- rq-507c40d1 -->
  - `false` in the default build, `true` under the `f64` feature.
  - Used by code that needs a compile-time check (`if REAL_IS_F64 { ... }`
    compiles to dead-code elimination on either branch).
- `pub const REAL_NAME: &str;` <!-- rq-d4759a9a -->
  - `"f32"` in the default build, `"f64"` under the `f64` feature.
  - Used by stdout banner output, error messages, and trajectory-writer
    precision selection.
- `pub const REAL_FMT_DIGITS: usize;` <!-- rq-f969ffbb -->
  - `9` in the default build, `17` under the `f64` feature.
  - The minimum number of fractional decimal digits required for the
    `{:.Ne}` exponent formatter to round-trip every `Real` value via
    `<Real as FromStr>::from_str`. Consumed by every text-format writer
    (trajectory, log, minimization log) and asserted by the writer's
    formatter helper.

### CUDA precision header <!-- rq-9922bfd1 -->

`kernels/precision.cuh` is the canonical home of the CUDA-side `Real`
typedef, the math intrinsic shim layer, and the literal-cast helper.
Every `.cu` and `.cuh` file under `kernels/` includes it directly or
transitively.

- `typedef float Real;` (default) or `typedef double Real;` (when nvcc <!-- rq-0c7e2ed1 -->
  is invoked with `-DREAL_F64`).
- `typedef float2 Real2;` / `typedef double2 Real2;` — used by SPME <!-- rq-928da8e4 -->
  reciprocal kernels for complex grid values.
- `__device__ __forceinline__ Real Real_sqrt(Real x);` <!-- rq-c3d3dc8e -->
- `__device__ __forceinline__ Real Real_rsqrt(Real x);` <!-- rq-24048902 -->
- `__device__ __forceinline__ Real Real_exp(Real x);` <!-- rq-48e7115e -->
- `__device__ __forceinline__ Real Real_log(Real x);` <!-- rq-c6364175 -->
- `__device__ __forceinline__ Real Real_sin(Real x);` <!-- rq-043796fa -->
- `__device__ __forceinline__ Real Real_cos(Real x);` <!-- rq-23398f89 -->
- `__device__ __forceinline__ Real Real_pow(Real x, Real y);` <!-- rq-781ce8fc -->
- `__device__ __forceinline__ Real Real_fabs(Real x);` <!-- rq-13070c27 -->
- `__device__ __forceinline__ Real Real_floor(Real x);` <!-- rq-3f77dca9 -->
- `__device__ __forceinline__ Real Real_rint(Real x);` <!-- rq-688a5b6c -->
- `__device__ __forceinline__ Real Real_fma(Real x, Real y, Real z);` <!-- rq-02f5d756 -->
- `__device__ __forceinline__ void Real_sincos(Real x, Real *s, Real *c);` <!-- rq-7d9d35f7 -->

Each of these resolves at compile time to the precision-matching CUDA math
intrinsic: `sqrtf`/`sqrt`, `rsqrtf`/`rsqrt`, `expf`/`exp`, `logf`/`log`,
`__sincosf`/`sincos`, etc. Every kernel that currently calls a precision-
suffixed CUDA intrinsic (`sqrtf`, `expf`, `rsqrtf`, `__sinf`, `__cosf`,
`__expf`, `rintf`, etc.) calls the shim instead. The shim layer is the
only place CUDA precision-suffixed names appear.

- `R(x)` — a function-style macro that casts a numeric literal to `Real`: <!-- rq-4dd1b9e3 -->
  ```c
  #define R(x) (Real(x))
  ```
  Used at every literal site where the bare literal would otherwise widen
  to `double` and trigger a narrowing-conversion warning under `Real = float`.
  Example: `Real half = R(0.5);` instead of `Real half = 0.5f;` or
  `Real half = 0.5;`. The macro compiles to the literal itself in both
  builds and produces no runtime cost.

### nvcc precision flag <!-- rq-a740c278 -->

`build.rs` passes `-DREAL_F64` to every `nvcc --ptx` invocation when the
`f64` Cargo feature is on, and omits the flag when the feature is off. The
flag is the only mechanism by which CUDA source code observes the
precision selector; `kernels/precision.cuh` keys off it directly. No other
build-time inputs change between the two builds.

`build.rs` is sensitive to the feature via Cargo's standard
`CARGO_FEATURE_F64` environment variable. It emits
`cargo:rerun-if-env-changed=CARGO_FEATURE_F64` so toggling the feature
forces a recompile of every kernel.

### Philox uniform-`Real` conversion <!-- rq-da488e3a -->

`kernels/philox.cuh` exposes a precision-aware uniform converter:

- `__device__ __forceinline__ Real philox_uniform_real(uint32 hi, uint32 lo);` <!-- rq-eade13fb -->
  - Returns a uniform `Real` in the half-open interval `[0, 1)`.
  - In the `f32` build, consumes only `hi`, extracts the top 24 bits, and
    divides by `2^24`. This produces every f32 representable value in
    `[0, 1)` at uniform density. The `lo` argument is ignored.
  - In the `f64` build, concatenates the top 21 bits of `hi` with all 32
    bits of `lo` into a 53-bit mantissa fill, and divides by `2^53`. This
    produces every f64 representable value in `[0, 1)` at uniform
    density at the precision the type permits.
  - Each call consumes exactly one (`f32`) or two (`f64`) Philox lanes;
    callers are responsible for advancing the counter accordingly.

- `__device__ __forceinline__ void philox_normal_real_pair(uint32 a, uint32 b, uint32 c, uint32 d, Real *n0, Real *n1);` <!-- rq-0d14734d -->
  - Returns two independent unit-normal samples via the Marsaglia polar
    method, computed in `Real` precision. Internally calls
    `philox_uniform_real` twice; consumes two (`f32`) or four (`f64`)
    Philox lanes.

Every Langevin, Andersen, CSVR, and c-rescale kernel routes its uniform
or normal sampling through these two functions and accounts for the
build-dependent lane consumption via a `Real` lane-count constant
declared in the same header (`PHILOX_LANES_PER_UNIFORM_REAL`,
`PHILOX_LANES_PER_NORMAL_REAL_PAIR`).

### cuFFT precision selection <!-- rq-5a89475f -->

`src/gpu/cufft.rs` exposes precision-aware 3D plan wrappers:

- `pub struct Plan3dR2C` — 3D real-to-complex forward FFT plan. <!-- rq-2f27efa4 -->
  - Backed by `cufftType` `CUFFT_R2C` in the f32 build, `CUFFT_D2Z` in
    the f64 build.
  - Method `execute(&self, idata: &CudaSlice<Real>, odata: &mut CudaSlice<RealComplex>) -> Result<(), GpuError>`
    invokes `cufftExecR2C` in the f32 build and `cufftExecD2Z` in the
    f64 build.
- `pub struct Plan3dC2R` — 3D complex-to-real inverse FFT plan. <!-- rq-584542bb -->
  - Backed by `CUFFT_C2R` in the f32 build, `CUFFT_Z2D` in the f64
    build. Method `execute` dispatches to `cufftExecC2R` or
    `cufftExecZ2D` respectively.
- `pub type RealComplex` — alias for `cufftComplex` in the f32 build and <!-- rq-ca8f0ae5 -->
  `cufftDoubleComplex` in the f64 build. Layout: two contiguous `Real`
  values (real then imaginary).

The plan wrappers' public API and Hermitian-symmetry expected sizes
(`n_a * n_b * (n_c / 2 + 1)` `RealComplex` values for `Plan3dR2C` output)
are unchanged across builds; the only difference is the underlying
`cufftType` value and the executor function name.

The cuFFT extern declarations in `src/gpu/cufft.rs` carry both the
single-precision and the double-precision function symbols
(`cufftExecR2C`, `cufftExecC2R`, `cufftExecD2Z`, `cufftExecZ2D`) in every
build. The Rust-level dispatch is `#[cfg(feature = "f64")]`; the
unselected symbol is still linked against (cuFFT is a single library
exporting both) but never called.

### Energy and virial widening <!-- rq-6c3cddc1 -->

The per-particle `potential_energies` and `virials` fields of
`ParticleState` (see `particle-state.md`) are `Vec<Real>`; the
matching device buffers in `ParticleBuffers` are `CudaSlice<Real>`. The
kinetic-energy reduction kernel (`kinetic_energy_reduce` in
`kernels.nose_hoover`), the virial-sum reduction
(`virial_sum_reduce` in `kernels.barostat`), and every accumulator
downstream of those reductions (Nose–Hoover chain state, MTK barostat
state, Berendsen / c-rescale rescaling factors, Langevin BAOAB
`kT_target`, CSVR draws, SHAKE per-constraint multipliers) follow
`Real`. Host-side scalars derived from a single device-side `Real`
reduction (e.g. instantaneous temperature, instantaneous pressure)
also follow `Real`.

### Velocity-Verlet `lossless` mode interaction <!-- rq-f5b9695a -->

The velocity-Verlet integrator's `lossless` mode (see
`rqm/pipeline-reversibility.md` and `rqm/integration/velocity-verlet.md`)
is only available in the default (`f32`) build. The Kahan-compensated
`f64` low-part it carries is a residual of an `f32` accumulation; it has
no meaning when storage is already `f64`. When the `f64` Cargo feature
is on:

- `Config::integrator` rejects `lossless = true` at config-load time with <!-- rq-65d18162 -->
  a precision-aware error variant
  `ConfigError::LosslessUnsupportedInF64Build`. The error message names
  the option and instructs the user to rebuild without `--features f64`
  if they require reversibility, or to set `lossless = false` if they
  want the f64 storage precision instead.
- The `vv_kick_drift_lossless` and `vv_kick_lossless` kernel handles are
  not loaded by `IntegrateKernels::load` in the f64 build; the
  `LosslessLowPart` device buffers and host scratch are not allocated.
  The default-precision-only kernels remain compiled and loaded
  unchanged in the f32 build.

The default (`f32`) build's behaviour, scenarios, and API surface for
`lossless` mode are unaffected.

### File format precision selection <!-- rq-03c6490c -->

The trajectory writer, init-state writer, log output, and minimization log
output format `Real` columns with a precision width determined at
construction by `precision::REAL_FMT_DIGITS`:

- `f32` build (default): `{:.9e}` — matches the existing format described <!-- rq-2c81a766 -->
  by `rqm/io/trajectory-output.md` and `rqm/io/log-output.md`. The
  9-fractional-digit exponent representation round-trips every `f32`
  value via the trajectory reader.
- `f64` build: `{:.17e}` — 17-fractional-digit exponent representation, <!-- rq-6f99094a -->
  the minimum width that round-trips every `f64` value.

The same applies to the `Lattice` and per-row real columns in the
trajectory file, to per-row real columns in the init-state file when
written, to the log file's per-step `kT`, total energy, kinetic energy,
potential energy, pressure, temperature, and box columns, and to the
minimization log's per-iteration energy and max-force columns. The
`Time` attribute on the trajectory comment line is always `f64` (host-
side seconds-or-atomic-time scalar) and uses `{:.9e}` regardless of the
precision feature.

The trajectory reader, init-state reader, and log reader parse with
`<Real as FromStr>::from_str`. A file written by an `f64` build is
parseable by the same `f64` build's reader. A file written by an `f64`
build and parsed by an `f32` build yields a `MalformedRow` error variant
`ValueExceedsPrecision { line_number, column, value }` only when a
parsed double is outside the representable range of `f32` (i.e. is
`±inf` after the cast); narrowing that merely truncates mantissa bits is
silent and accepted (the reader's contract is value parsing, not
provenance preservation). A file written by an `f32` build and parsed by
an `f64` build is parseable with zero-padded mantissa.

### Reproducibility under the feature flag <!-- rq-bdbf6752 -->

The bit-wise reproducibility guarantee continues to hold within a single
build: two runs of the `f32` build on the same GPU produce byte-identical
output; two runs of the `f64` build on the same GPU produce byte-identical
output. The guarantee does not extend across builds: an `f32` build and
an `f64` build of the same source code produce different bytes from the
same input, and that divergence is expected. Existing reproducibility
tests (`pipeline_reproducibility.rs`, per-kernel reproducibility tests
under `tests/`) pass under both feature combinations independently.

Existing CPU-reference tolerance tests (those that compare a kernel result
against a CPU-computed expected value) parameterise their tolerance on
the build: `1e-5 * |expected|` in the `f32` build and
`1e-13 * |expected|` in the `f64` build. The tolerance value is exposed
as `precision::CPU_REFERENCE_TOLERANCE` so test code carries a single
named constant. Tolerance values are upper bounds, not predictions of
the actual numerical error.

## Test Targets <!-- rq-c15b187c -->

The integration test `tests/precision_feature.rs` validates the build-time
behaviour of the feature flag directly:

- Asserts `std::mem::size_of::<Real>()` matches `REAL_BYTES`.
- Asserts `REAL_NAME == "f32"` in the default build and `"f64"` under
  `--features f64`.
- Asserts `REAL_FMT_DIGITS == 9` / `17` to match.
- Constructs a `ParticleState` whose `positions_x` is a single
  representative `Real` value (`std::f32::consts::PI` cast to `Real` in
  the f32 build, `std::f64::consts::PI` in the f64 build), uploads it to
  a `ParticleBuffers`, downloads it, and asserts byte-for-byte equality.

`tests/precision_lossless_reject.rs` validates the `lossless` config
rejection in the f64 build only (`#[cfg(feature = "f64")]`): given a
config with `[integrator] lossless = true`, asserts that
`Config::load` returns `Err(ConfigError::LosslessUnsupportedInF64Build)`
and does not initialise any GPU state.

`tests/precision_trajectory_format.rs` validates the
`REAL_FMT_DIGITS`-driven format:

- In the default build, asserts the trajectory writer emits exactly
  `{:.9e}` formatted columns and that a round-trip through the reader
  preserves every value byte-for-byte after cast to `Real`.
- Under `--features f64`, asserts the writer emits exactly `{:.17e}`
  formatted columns and that round-trip preserves every value.

The existing `pipeline_reproducibility.rs` test is run under both feature
sets (CI is responsible for the matrix; see `docs/architecture.md` build
section). The test itself is precision-agnostic: it compares two
independent pipelines in the same build for byte-identical output.

## Out of Scope <!-- rq-8c83aff4 -->

- Mixed-precision kernels (e.g. f32 forces with an f64 reduction). The
  feature is uniform within a build.
- Runtime precision selection. The `f64` feature is purely compile-time;
  the binary contains a single precision pipeline.
- Cross-build trajectory interchange as a first-class guarantee. The
  precision-narrowing parser path is convenience, not contract.
- `f128`, quad-double, or arbitrary-precision storage. Not in this
  feature or any planned extension.
- A double-double (f64 low-part of f64) extension of velocity-Verlet
  `lossless` mode. `lossless` is f32-only; a future feature may revisit.
- f64-tuned kernel grid dimensions or block sizes. Existing block sizes
  remain; the f64 build accepts whatever throughput drop the hardware
  produces on f64 ops.
- Detection or warning of poor-f64-throughput hardware (consumer GPUs).
  An operational concern, not a requirements concern.
- Switching to `-fmad=false` under f64. FMA contraction is left enabled
  in both builds; the cross-hardware match is not promised in either.

---

## Gherkin Scenarios <!-- rq-483fc240 -->

```gherkin
Feature: f64 precision compile-time feature flag

  # --- Type alias ---

  @rq-5fa49fa5
  Scenario: Default build resolves Real to f32
    Given the crate is built with default features
    Then std::mem::size_of::<Real>() equals 4
    And precision::REAL_BYTES equals 4
    And precision::REAL_IS_F64 is false
    And precision::REAL_NAME equals "f32"
    And precision::REAL_FMT_DIGITS equals 9

  @rq-5b13a93d
  Scenario: f64 build resolves Real to f64
    Given the crate is built with --features f64
    Then std::mem::size_of::<Real>() equals 8
    And precision::REAL_BYTES equals 8
    And precision::REAL_IS_F64 is true
    And precision::REAL_NAME equals "f64"
    And precision::REAL_FMT_DIGITS equals 17

  # --- Build script behaviour ---

  @rq-67279128
  Scenario: Default build does not pass -DREAL_F64 to nvcc
    Given the crate is built with default features
    When build.rs invokes nvcc on any kernels/*.cu source
    Then the nvcc command line does not contain "-DREAL_F64"

  @rq-831fee6e
  Scenario: f64 build passes -DREAL_F64 to nvcc
    Given the crate is built with --features f64
    When build.rs invokes nvcc on any kernels/*.cu source
    Then the nvcc command line contains "-DREAL_F64"

  @rq-cd3943e1
  Scenario: Toggling the f64 feature forces a kernel recompile
    Given the crate has been built with default features
    And the build artefacts (target/) are intact
    When the crate is rebuilt with --features f64
    Then every kernel .ptx file in OUT_DIR is regenerated

  # --- CUDA precision header ---

  @rq-88a10028
  Scenario: CUDA Real typedef matches the build precision
    Given the crate is built with default features
    When a kernel that declares "Real x = R(1.5);" is compiled
    Then x is a 4-byte float in the resulting PTX

  @rq-8b40b72e
  Scenario: CUDA Real typedef under f64 build
    Given the crate is built with --features f64
    When a kernel that declares "Real x = R(1.5);" is compiled
    Then x is an 8-byte double in the resulting PTX

  @rq-2a30cf81
  Scenario: Math shim dispatches to single-precision intrinsics in default build
    Given the crate is built with default features
    When a kernel calls Real_sqrt(x) on a Real argument
    Then the resulting PTX call resolves to sqrt.approx.f32 (or sqrt.rn.f32)
    And does not contain a sqrt.f64 call

  @rq-2d46ce11
  Scenario: Math shim dispatches to double-precision intrinsics in f64 build
    Given the crate is built with --features f64
    When a kernel calls Real_sqrt(x) on a Real argument
    Then the resulting PTX call resolves to sqrt.rn.f64
    And does not contain a sqrt.approx.f32 call

  # --- ParticleState / ParticleBuffers ---

  @rq-dd417155
  Scenario: ParticleState fields are Vec<Real>
    Given the crate is built with default features
    Then ParticleState::positions_x has type Vec<Real>
    And the in-memory layout of positions_x is 4 bytes per element

  @rq-dc5041ea
  Scenario: ParticleState fields are Vec<Real> under f64
    Given the crate is built with --features f64
    Then ParticleState::positions_x has type Vec<Real>
    And the in-memory layout of positions_x is 8 bytes per element

  @rq-67e6c4fb
  Scenario: ParticleBuffers device allocations widen with the feature
    Given the crate is built with --features f64
    And a ParticleState with particle_count == 1024
    When ParticleBuffers::new(device, &state) is called
    Then the byte size of buffers.positions_x equals 8 * 1024

  @rq-eb4b5952
  Scenario: Image flags do not widen under f64
    Given the crate is built with --features f64
    And a ParticleState with particle_count == 1024
    Then state.images_x has type Vec<i32>
    And the byte size of buffers.images_x equals 4 * 1024

  @rq-f4a44029
  Scenario: Particle IDs do not widen under f64
    Given the crate is built with --features f64
    And a ParticleState with particle_count == 1024
    Then state.particle_ids has type Vec<u32>
    And the byte size of buffers.particle_ids equals 4 * 1024

  # --- Round-trip through the device ---

  @rq-2a9e2342
  Scenario: Round-trip preserves a representative Real value in the default build
    Given the crate is built with default features
    And a ParticleState A with particle_count == 1
    And A.positions_x[0] has been set to std::f32::consts::PI as Real
    And a ParticleBuffers built from A
    And A.positions_x[0] has been zeroed on the host
    When A.download_from(&buffers) is called
    Then A.positions_x[0] equals std::f32::consts::PI as Real byte-for-byte

  @rq-ed8dcaee
  Scenario: Round-trip preserves a representative Real value under f64
    Given the crate is built with --features f64
    And a ParticleState A with particle_count == 1
    And A.positions_x[0] has been set to std::f64::consts::PI as Real
    And a ParticleBuffers built from A
    And A.positions_x[0] has been zeroed on the host
    When A.download_from(&buffers) is called
    Then A.positions_x[0] equals std::f64::consts::PI as Real byte-for-byte

  # --- Pipeline reproducibility within a build ---

  @rq-1148ddc5
  Scenario: Bit-exact pipeline reproducibility in the default build
    Given the crate is built with default features
    And the pipeline-reproducibility fixture from rqm/pipeline-reproducibility.md
    When two independent pipelines run 100 steps each
    Then every Vec<Real> field of state_A equals the corresponding field of state_B byte-for-byte

  @rq-b11e4631
  Scenario: Bit-exact pipeline reproducibility under f64
    Given the crate is built with --features f64
    And the pipeline-reproducibility fixture from rqm/pipeline-reproducibility.md
    When two independent pipelines run 100 steps each
    Then every Vec<Real> field of state_A equals the corresponding field of state_B byte-for-byte

  @rq-3ce546db
  Scenario: Cross-build outputs are not bit-exact (informational)
    Given identical input states are run for 100 steps in the default build and the f64 build
    When the two final ParticleStates are compared
    Then their Vec<Real> fields are not required to agree byte-for-byte
    And the divergence is a non-failure (no test asserts equality)

  # --- Philox uniform converter ---

  @rq-fc912e57
  Scenario: Philox uniform fills 24 bits in the default build
    Given the crate is built with default features
    When philox_uniform_real(hi=0xFFFFFFFF, lo=0xFFFFFFFF) is called on the device
    Then the result equals the f32 value nextafter(1.0, 0.0)
    And the result lies in the half-open interval [0, 1)

  @rq-0709c5ec
  Scenario: Philox uniform fills 53 bits in the f64 build
    Given the crate is built with --features f64
    When philox_uniform_real(hi=0xFFFFFFFF, lo=0xFFFFFFFF) is called on the device
    Then the result equals the f64 value nextafter(1.0, 0.0)
    And the result lies in the half-open interval [0, 1)

  @rq-a6bc92a1
  Scenario: Philox uniform consumes one lane per call in the default build
    Given the crate is built with default features
    Then PHILOX_LANES_PER_UNIFORM_REAL equals 1

  @rq-86c31917
  Scenario: Philox uniform consumes two lanes per call in the f64 build
    Given the crate is built with --features f64
    Then PHILOX_LANES_PER_UNIFORM_REAL equals 2

  # --- cuFFT plan dispatch ---

  @rq-ba281906
  Scenario: SPME reciprocal stream uses single-precision cuFFT plans in the default build
    Given the crate is built with default features
    And an SpmeReciprocalState constructed for an (nx, ny, nz) grid
    Then its Plan3dR2C is backed by a CUFFT_R2C plan
    And its Plan3dC2R is backed by a CUFFT_C2R plan
    And RealComplex equals cufftComplex (two contiguous f32 values)

  @rq-6dc17c20
  Scenario: SPME reciprocal stream uses double-precision cuFFT plans under f64
    Given the crate is built with --features f64
    And an SpmeReciprocalState constructed for an (nx, ny, nz) grid
    Then its Plan3dR2C is backed by a CUFFT_D2Z plan
    And its Plan3dC2R is backed by a CUFFT_Z2D plan
    And RealComplex equals cufftDoubleComplex (two contiguous f64 values)

  @rq-0c285656
  Scenario: cuFFT execute dispatches to the precision-matched executor in the default build
    Given the crate is built with default features
    And a Plan3dR2C plan with valid input and output buffers
    When plan.execute(&input, &mut output) is called
    Then the underlying cuFFT call is cufftExecR2C
    And cufftExecD2Z is not invoked

  @rq-0873d567
  Scenario: cuFFT execute dispatches to the precision-matched executor under f64
    Given the crate is built with --features f64
    And a Plan3dR2C plan with valid input and output buffers
    When plan.execute(&input, &mut output) is called
    Then the underlying cuFFT call is cufftExecD2Z
    And cufftExecR2C is not invoked

  # --- Lossless integrator rejection under f64 ---

  @rq-aba94795
  Scenario: Default build accepts lossless = true at config load
    Given the crate is built with default features
    And a TOML config with [integrator] lossless = true
    When Config::load is called on the config string
    Then it returns Ok(config)

  @rq-4d8894d3
  Scenario: f64 build rejects lossless = true at config load
    Given the crate is built with --features f64
    And a TOML config with [integrator] lossless = true
    When Config::load is called on the config string
    Then it returns Err(ConfigError::LosslessUnsupportedInF64Build)
    And no GPU state is allocated

  @rq-b376229c
  Scenario: f64 build accepts lossless = false at config load
    Given the crate is built with --features f64
    And a TOML config with [integrator] lossless = false
    When Config::load is called on the config string
    Then it returns Ok(config)

  @rq-9bc38ffd
  Scenario: f64 build does not load lossless kernel handles
    Given the crate is built with --features f64
    When init_device() is called
    Then GpuContext::kernels::integrate does not carry a vv_kick_drift_lossless field
    And GpuContext::kernels::integrate does not carry a vv_kick_lossless field
    And init_device returns Ok

  # --- File-format precision selection ---

  @rq-64627cdb
  Scenario: Default build trajectory writer emits 9-digit columns
    Given the crate is built with default features
    And a TrajectoryWriter opened with units = UnitSystem::Atomic
    And positions_x = [3.4e-10 as Real]
    When write_frame is called
    Then the first position column is formatted as "3.400000095e-10"
    And the column matches the regex /^-?[0-9]\.[0-9]{9}e-?[0-9]+$/

  @rq-28967c97
  Scenario: f64 build trajectory writer emits 17-digit columns
    Given the crate is built with --features f64
    And a TrajectoryWriter opened with units = UnitSystem::Atomic
    And positions_x = [3.4e-10 as Real]
    When write_frame is called
    Then the first position column is formatted as "3.40000000000000023e-10"
    And the column matches the regex /^-?[0-9]\.[0-9]{17}e-?[0-9]+$/

  @rq-8893982c
  Scenario: Trajectory reader round-trips Real values byte-for-byte within a build
    Given the crate is built with the feature flag set to X (X in {off, on})
    And a TrajectoryWriter has written a frame with positions_x containing arbitrary Real values
    When TrajectoryReader::open is called on the same file in the same build
    And next_frame() is called
    Then frame.positions_x equals the write-side values byte-for-byte after cast through Real

  @rq-f7853c49
  Scenario: Trajectory reader in the f32 build accepts an f64-built file with silent narrowing
    Given the crate is built with default features
    And a trajectory file written by an --features f64 build with 17-digit columns
    When TrajectoryReader::open is called on the file
    And next_frame() is called
    Then it returns Ok(frame)
    And frame.positions_x[i] equals the parsed f64 value cast to f32 for each i

  @rq-fbe7ea38
  Scenario: Trajectory reader in the f32 build rejects an out-of-f32-range value
    Given the crate is built with default features
    And a trajectory file containing a position column with the textual value "1.0e40"
    When TrajectoryReader::open is called on the file
    And next_frame() is called
    Then it returns Err(TrajectoryReaderError::MalformedRow { reason: contains "ValueExceedsPrecision", .. })

  @rq-26c02ccc
  Scenario: Time attribute is always 9-digit f64 regardless of build
    Given the crate is built with the feature flag set to X (X in {off, on})
    And a TrajectoryWriter opened with units = UnitSystem::Atomic
    When write_frame is called with step = 1 and dt = 1.0
    Then the Time attribute in the written frame is "1.000000000e0"

  # --- Energy and virial widening ---

  @rq-f0d29c33
  Scenario: potential_energies and virials follow Real in both builds
    Given the crate is built with the feature flag set to X (X in {off, on})
    Then ParticleState::potential_energies has type Vec<Real>
    And ParticleState::virials has type Vec<Real>
    And the byte size of buffers.potential_energies equals REAL_BYTES * particle_count
    And the byte size of buffers.virials equals REAL_BYTES * particle_count

  @rq-e4c2141e
  Scenario: Kinetic energy reduction produces a Real in both builds
    Given the crate is built with the feature flag set to X (X in {off, on})
    And a ParticleBuffers with non-trivial velocities and masses
    When kinetic_energy_reduce is launched and the host-side reduction completes
    Then the host receives a Real value
    And REAL_BYTES bytes are transferred from device to host

  # --- CPU-reference tolerance ---

  @rq-abab7081
  Scenario: CPU-reference tolerance constant tightens in the f64 build
    Given the crate is built with default features
    Then precision::CPU_REFERENCE_TOLERANCE equals 1e-5
    Given the crate is built with --features f64
    Then precision::CPU_REFERENCE_TOLERANCE equals 1e-13
```
