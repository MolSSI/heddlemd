# Feature: CUDA Build Pipeline and Smoke Test Kernel <!-- rq-cc1b997d -->

The project compiles CUDA C kernels to PTX at build time and loads them at
runtime via cudarc. A trivial "fill" kernel serves as a permanent smoke test
that validates the entire toolchain end-to-end: `nvcc` compilation, PTX
embedding, device initialization, kernel launch, and host readback.

## Build Script <!-- rq-438b895b -->

A `build.rs` build script compiles every `.cu` file under `kernels/` to PTX
using `nvcc`.

### nvcc invocation <!-- rq-f379e47c -->

For each `.cu` file, `build.rs` invokes:

```
nvcc --ptx -std=c++17 -o <out_dir>/<stem>.ptx <source_path>
```

where `<out_dir>` is the value of the `OUT_DIR` environment variable set by
Cargo and `<stem>` is the file name without the `.cu` extension. The
`-std=c++17` flag is required so kernels can use `if constexpr` and
function templates.

`build.rs` emits `cargo:rerun-if-changed=kernels/` so that Cargo reruns the
build script whenever any file under `kernels/` changes.

### Missing nvcc <!-- rq-4a3feaf7 -->

If `nvcc` is not found on `PATH` or exits with a non-zero status, `build.rs`
panics with a message that includes:

- The failing command and its exit code (or "not found" if the binary is
  absent).
- A note that the CUDA Toolkit is required and a pointer to the NVIDIA
  installation documentation.

### PTX embedding <!-- rq-c1cfec9a -->

Each compiled PTX file is made available to Rust source code via
`include_str!`. `build.rs` writes a generated Rust file to `OUT_DIR` that
contains one `pub const` per kernel module:

```rust
pub const FILL: &str = include_str!("<out_dir>/fill.ptx");
```

The generated file is included in the Rust source tree with
`include!(concat!(env!("OUT_DIR"), "/kernels.rs"))`.

## GPU Device Module <!-- rq-0c3a23bb -->

`src/gpu/device.rs` exposes a function for obtaining an initialized CUDA
device with all compiled PTX modules loaded, together with a typed,
per-subsystem nested handle to every kernel function.

### Functions <!-- rq-c38c8f3b -->

- `init_device() -> Result<GpuContext, GpuError>`
  - Calls `CudaDevice::new(0)` to initialize the default CUDA device.
  - Calls `Kernels::load(&device)`, which in turn invokes every
    subsystem's `XKernels::load(device)` associated function. Each
    `load` is responsible for loading its own PTX module onto the
    device with `CudaDevice::load_ptx` and pulling its function
    handles with `CudaDevice::get_func`.
  - Returns a `GpuContext` bundling the `Arc<CudaDevice>` and the
    composed `Arc<Kernels>` handle.
  - Returns `Err(GpuError)` if device creation, any subsystem's PTX
    loading, or any subsystem's kernel lookup fails. The first
    failing subsystem short-circuits the rest.

### Types <!-- rq-2093594f -->

- `GpuError` — error type for GPU operations. Wraps the underlying
  `cudarc::driver::DriverError`.

- `GpuContext` — the value returned by `init_device`. A plain struct with
  two public fields and no `Deref`:
  - `device: Arc<CudaDevice>` — the initialized cudarc device, used for
    memory allocation and host/device transfers.
  - `kernels: Arc<Kernels>` — the kernel-function handle.

  Both fields are cheap to clone, so each GPU-resource struct stores its
  own clones rather than borrowing the `GpuContext`.

- `Kernels` — composed handle holding, for every CUDA kernel compiled
  from `kernels/*.cu`, its launchable function. The composition is
  per-subsystem-nested: `Kernels` carries one field per subsystem,
  each holding that subsystem's typed sub-struct. Kernel launch sites
  obtain their function through a two-level field chain
  (`particle_buffers.kernels.<subsystem>.<kernel_name>`) rather than a
  string-keyed lookup. A reference to a non-existent kernel is a
  compile error, and a kernel missing from its PTX module is caught
  once, by `init_device`, rather than on first launch.

  `Kernels`, `Kernels::load`, and the `KernelStage::ORDER` registry
  (see `performance-analysis.md`) are expanded from one central
  manifest by the `define_kernels!` macro, so the three cannot drift
  apart:

  ```rust
  define_kernels! {
      fill:        FillKernels,
      integrate:   IntegrateKernels,
      spme_recip:  SpmeRecipKernels,
      langevin:    LangevinKernels,
      morse:       MorseKernels,
      angle:       AngleKernels,
      nose_hoover: NoseHooverKernels,
      andersen:    AndersenKernels,
      barostat:    BarostatKernels,
      mtk:         MtkKernels,
      shake:       ShakeKernels,
      settle:      SettleKernels,
      forces:      ForcesKernels,
      neighbor:    NeighborKernels,
      minimize:    MinimizeKernels,
  }
  ```

  The macro emits the `Kernels` struct (one field per entry, of the
  named sub-struct type), `Kernels::load` (each sub-struct's `load` in
  manifest order, short-circuiting on the first failure), and
  `KernelStage::ORDER` (the manifest-order concatenation of every
  sub-struct's `STAGES`). Adding a kernel to an existing subsystem is a
  one-line edit to that subsystem's `gpu_kernels!` invocation; adding a
  whole subsystem is one manifest line plus the subsystem's own file.
  `device.rs` names no individual kernel and carries no per-subsystem
  stage list.

  The subsystem field set is:

  | Field         | Sub-struct type      | Sub-struct home                                | PTX module      | Kernels                                                                                           |
  | ---           | ---                  | ---                                            | ---             | ---                                                                                              |
  | `fill`        | `FillKernels`        | `src/gpu/fill.rs`                              | `fill`          | `fill`                                                                                            |
  | `integrate`   | `IntegrateKernels`   | `src/integrator/velocity_verlet.rs`            | `integrate`     | `vv_kick_drift`, `vv_kick`, `vv_kick_drift_lossless`, `vv_kick_lossless`                          |
  | `spme_recip`  | `SpmeRecipKernels`   | `src/forces/spme.rs`                           | `spme_recip`    | `spme_charge_spread`, `spme_influence_multiply`, `spme_force_gather`                              |
  | `langevin`    | `LangevinKernels`    | `src/integrator/langevin_baoab.rs`             | `langevin`      | `lan_drift_half`, `lan_ou_step`                                                                   |
  | `morse`       | `MorseKernels`       | `src/forces/morse.rs`                          | `morse`         | `morse_bond_force`, `reduce_bond_forces`                                                          |
  | `angle`       | `AngleKernels`       | `src/forces/angle.rs`                          | `angle`         | `harmonic_angle_force`, `reduce_angle_forces`                                                     |
  | `nose_hoover` | `NoseHooverKernels`  | `src/integrator/nose_hoover_chain.rs`          | `nose_hoover`   | `kinetic_energy_reduce`, `rescale_velocities`                                                     |
  | `andersen`    | `AndersenKernels`    | `src/integrator/andersen.rs`                   | `andersen`      | `andersen_resample`                                                                               |
  | `barostat`    | `BarostatKernels`    | `src/gpu/barostat_kernels.rs`                  | `barostat`      | `virial_sum_reduce`, `rescale_positions`                                                          |
  | `mtk`         | `MtkKernels`         | `src/integrator/mtk_npt.rs`                    | `mtk`           | `mtk_velocity_half_kick`, `mtk_position_drift`                                                    |
  | `shake`       | `ShakeKernels`       | `src/integrator/shake.rs`                      | `shake`         | `shake_snapshot`, `shake_positions`, `rattle_velocities`, `constraint_virial_scatter`, `shake_positions_no_velocity`                                                                                                                                              |
  | `settle`      | `SettleKernels`      | `src/integrator/settle.rs`                     | `settle`        | `settle_snapshot`, `settle_positions`, `settle_velocities`, `settle_virial_scatter`, `settle_positions_no_velocity`                                                                                                                                              |
  | `forces`      | `ForcesKernels`      | `src/forces/mod.rs`                            | `forces`        | `accumulate_forces`                                                                               |
  | `neighbor`    | `NeighborKernels`    | `src/forces/neighbor_list.rs`                  | `neighbor`      | `neighbor_displacement_squared`, `copy_positions_into_reference`, `compute_cell_indices_and_histogram`, `prefix_scan_local_blocks`, `prefix_scan_apply_block_totals`, `prefix_scan_finalize_offsets`, `scatter_atoms_into_cells`, `sort_cells_by_particle_id` |
  | `minimize`    | `MinimizeKernels`    | `src/minimizer/mod.rs`                         | `minimize`      | `sd_compute_step`, `sd_snapshot`, `sd_restore`, `sd_f_max_reduction`                              |

  Each subsystem's sub-struct is expanded by a `gpu_kernels!`
  invocation in the sub-struct's home file, from one list of the
  subsystem's kernel function names and one list of the stages it
  times:

  ```rust
  gpu_kernels! {
      module: "settle",
      struct: SettleKernels,
      kernels: [
          settle_snapshot, settle_positions, settle_velocities,
          settle_virial_scatter, settle_positions_no_velocity,
      ],
      stages: {
          SETTLE_SNAPSHOT              = "settle_snapshot",
          SETTLE_POSITIONS             = "settle_positions",
          SETTLE_VELOCITIES            = "settle_velocities",
          SETTLE_VIRIAL_SCATTER        = "settle_virial_scatter",
          SETTLE_POSITIONS_NO_VELOCITY = "settle_positions_no_velocity",
      },
  }
  ```

  From that single declaration the macro emits:
  - the `pub struct` with one `pub <name>: CudaFunction` field per
    `kernels` entry, deriving `Debug + Clone` (a `CudaFunction` is
    itself cheap to clone);
  - `pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError>`,
    which loads the named PTX `module` once with `CudaDevice::load_ptx`
    and pulls each field with `get_func` — the kernel-name list is
    written exactly once and feeds both the `load_ptx` name array and
    the per-field `get_func` calls;
  - the `KernelStage` consts named in `stages` (inherent consts on
    `KernelStage`; see `performance-analysis.md`) and a `STAGES` slice
    holding exactly those consts, in launch order;
  - an `impl SubsystemKernels` exposing `MODULE`, `STAGES`, and `load`.

  Because a subsystem's stage consts and its `STAGES` slice come from
  the one `stages` list, a stage const can never be absent from the
  subsystem's registry contribution, and the registry can never name a
  stage no subsystem declares. A subsystem that times no stages (a
  thermostat or barostat reusing another subsystem's kernels) gives an
  empty `stages` list and contributes an empty `STAGES`.

  Cross-subsystem reads follow the home rule: a kernel lives in the
  sub-struct named after its `.cu` file, and consumers in other
  subsystems reach into that sub-struct directly. For example, the
  `kinetic_energy_reduce` and `rescale_velocities` kernels live in
  `kernels.nose_hoover.*` and are consumed by NHC, CSVR, and the MTK
  barostat substep. No kernel handle is duplicated across sub-structs.

- `SubsystemKernels` — trait implemented by every per-subsystem
  sub-struct (via `gpu_kernels!`). Carries `const MODULE: &'static str`
  (the `.cu` stem / PTX module name), `const STAGES: &'static
  [KernelStage]` (the subsystem's timed stages in launch order; empty
  when it records none), and `fn load(device: &Arc<CudaDevice>) ->
  Result<Self, GpuError>`. `define_kernels!` is generic over this
  trait: it builds `Kernels::load` from each field's `load` and
  `KernelStage::ORDER` from each field's `STAGES`.

- `gpu_kernels!` — declarative `macro_rules!` macro (no external
  dependency) invoked once per subsystem in its home file. Expands one
  kernel-name list and one stage list into the sub-struct, its `load`,
  the `KernelStage` consts, the `STAGES` slice, and the
  `SubsystemKernels` impl described above.

- `define_kernels!` — declarative `macro_rules!` macro invoked once, in
  `device.rs`, over the central subsystem manifest. Expands it into the
  `Kernels` struct, `Kernels::load`, and `KernelStage::ORDER`.

## Fill Kernel <!-- rq-599b2eb4 -->

`kernels/fill.cu` contains a CUDA kernel that writes a constant value to
every element of a floating-point output buffer.

### Kernel signature <!-- rq-25afada1 -->

```c
extern "C" __global__ void fill(float *out, float value, unsigned int n)
```

- `out` — pointer to the output buffer on the device.
- `value` — the constant value to write.
- `n` — number of elements in the buffer.

Each thread computes its global index as `blockIdx.x * blockDim.x + threadIdx.x`.
If the index is less than `n`, it writes `value` to `out[index]`.

## Smoke Test <!-- rq-c669b757 -->

An integration test in `tests/smoke_gpu.rs` validates the full pipeline.

### Test procedure <!-- rq-f7fe7f1b -->

1. Call `init_device()` to obtain a `GpuContext`.
2. Allocate a device buffer of 1024 `f32` elements using `CudaDevice::alloc_zeros`.
3. Launch the `fill` kernel — obtained from the `GpuContext`'s `kernels`
   handle — with `value = 1.0` and `n = 1024`.
   - Block size: 256 threads.
   - Grid size: `ceil(n / 256)` blocks.
4. Copy the buffer back to the host using `CudaDevice::dtoh_sync_copy`.
5. Assert that every element in the host buffer is exactly `1.0_f32`.

### Edge case: non-block-aligned buffer size <!-- rq-89d64ca3 -->

A second test case uses a buffer size that is not a multiple of the block size
(e.g., 1000) to verify that the kernel's bounds check works correctly and no
out-of-bounds writes occur.

## Project Structure <!-- rq-b735d0b0 -->

```
build.rs                  # compiles kernels/*.cu to PTX, generates kernels.rs
kernels/
  fill.cu                 # trivial fill kernel (permanent smoke test)
src/
  gpu/
    mod.rs                # re-exports gpu submodules
    device.rs             # init_device(), GpuContext, Kernels, GpuError
    fill.rs               # FillKernels (smoke-test kernel handle)
    barostat_kernels.rs   # BarostatKernels (shared by Berendsen + c-rescale)
tests/
  smoke_gpu.rs            # integration test for fill kernel
Cargo.toml                # depends on cudarc
```

Subsystem `*Kernels` sub-structs live next to the code that owns each
PTX module (see the *Types* table above for the full home list), each
expanded by a `gpu_kernels!` invocation in that file. `src/gpu/device.rs`
carries `init_device`, `GpuContext`, `GpuError`, and the single
`define_kernels!` manifest from which the `Kernels` struct,
`Kernels::load`, and `KernelStage::ORDER` are generated — it names no
individual kernel.

## Dependencies <!-- rq-93367a8f -->

`Cargo.toml` declares:

- `cudarc` — CUDA device management, memory allocation, and kernel launch.

No other new dependencies are introduced.

## Out of Scope <!-- rq-e9a67206 -->

- Real simulation kernels (pair force, reduction, integration).
- SoA particle state buffers.
- The `f64` precision feature flag (the fill kernel uses `float`/`f32` only).
- Multi-GPU support (the smoke test uses device 0).

---

## Gherkin Scenarios <!-- rq-61606908 -->

```gherkin
Feature: CUDA build pipeline and smoke test kernel

  # --- Build script ---

  @rq-de92101c
  Scenario: build.rs compiles a .cu file to PTX
    Given a file "kernels/fill.cu" containing a valid CUDA kernel
    And nvcc is available on PATH
    When cargo build is run
    Then a file "fill.ptx" is produced in OUT_DIR
    And the build succeeds

  @rq-c17ef35f
  Scenario: PTX is embedded in the Rust binary
    Given the build has succeeded
    When the generated kernels.rs is inspected
    Then it contains a pub const FILL whose value is the contents of fill.ptx

  # --- Device initialization ---

  @rq-05691d2f
  Scenario: init_device succeeds on a machine with a CUDA GPU
    Given a CUDA-capable GPU is available as device 0
    When init_device() is called
    Then it returns Ok containing a GpuContext
    And the GpuContext's kernels handle exposes the fill function as
      `gpu_context.kernels.fill.fill`

  @rq-299c69c9
  Scenario: init_device fails when no GPU is available
    Given no CUDA-capable GPU is available
    When init_device() is called
    Then it returns Err(GpuError)

  # --- Per-subsystem kernel composition ---

  @rq-6211a82f
  Scenario: Kernels is composed of per-subsystem sub-structs
    Given a GpuContext obtained from init_device()
    Then gpu_context.kernels has one field per entry in the
      define_kernels! manifest: fill, integrate, spme_recip, langevin,
      morse, angle, nose_hoover, andersen, barostat, mtk, shake, settle,
      forces, neighbor, minimize
    And each field is the matching subsystem's typed kernel struct

  @rq-6745e7c5
  Scenario: Each subsystem's XKernels::load returns its kernel handle
    Given a CUDA-capable GPU initialized via CudaDevice::new(0)
    When SpmeRecipKernels::load(&device) is called
    Then it returns Ok(SpmeRecipKernels) whose `spme_charge_spread` field
      is a launchable CudaFunction
    And the PTX module `spme_recip` is loaded on the device

  # --- Macro-generated declarations ---

  @rq-73a85df1
  Scenario: A subsystem's STAGES holds exactly the stages it declares
    Given the gpu_kernels! invocation for SettleKernels whose `stages`
      list names SETTLE_SNAPSHOT, SETTLE_POSITIONS, SETTLE_VELOCITIES,
      SETTLE_VIRIAL_SCATTER, SETTLE_POSITIONS_NO_VELOCITY
    Then <SettleKernels as SubsystemKernels>::STAGES equals exactly that
      list, in that order
    And SettleKernels::MODULE equals "settle"

  @rq-0919ff0a
  Scenario: A subsystem that times no stages contributes an empty STAGES
    Given a gpu_kernels! invocation whose `stages` list is empty
    Then the generated sub-struct's SubsystemKernels::STAGES is empty
    And that sub-struct contributes no rows to KernelStage::ORDER

  @rq-cfc89131
  Scenario: A subsystem load that names a missing kernel surfaces as GpuError at init_device
    Given a CUDA-capable GPU is available
    And a subsystem's load(...) names a kernel that does not appear in
      its PTX module (e.g. via a typo)
    When init_device() is called
    Then it returns Err(GpuError) at that subsystem's load step
    And no subsequent subsystem's load is invoked

  @rq-7b651edb
  Scenario: Cross-subsystem reads pull from the kernel's home sub-struct
    Given a GpuContext obtained from init_device()
    Then the Nose-Hoover launch wrapper reads
      `particle_buffers.kernels.nose_hoover.kinetic_energy_reduce` for
      the shared kinetic-energy reduction kernel
    And the same `kinetic_energy_reduce` handle is used by the CSVR
      thermostat launch wrapper

  # --- Fill kernel correctness ---

  @rq-9cf657ed
  Scenario: fill kernel writes 1.0 to every element of a block-aligned buffer
    Given a GpuContext obtained from init_device()
    And a device buffer of 1024 f32 elements initialized to zero
    When the fill kernel is launched with value=1.0 and n=1024
    And the buffer is copied back to the host
    Then all 1024 host elements are exactly 1.0_f32

  @rq-26d8c08c
  Scenario: fill kernel handles a non-block-aligned buffer size
    Given a GpuContext obtained from init_device()
    And a device buffer of 1000 f32 elements initialized to zero
    When the fill kernel is launched with value=1.0 and n=1000
    And the buffer is copied back to the host
    Then all 1000 host elements are exactly 1.0_f32

  @rq-d920e446
  Scenario: fill kernel does not write beyond the buffer
    Given a GpuContext obtained from init_device()
    And a device buffer of 1024 f32 elements initialized to zero
    When the fill kernel is launched with value=1.0 and n=1000
    And the buffer is copied back to the host
    Then the first 1000 elements are exactly 1.0_f32
    And the remaining 24 elements are exactly 0.0_f32
```
