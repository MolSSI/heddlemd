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
  once, by `init_device`, rather than on first launch. Adding a new
  kernel to an existing subsystem is a one-line edit to the
  subsystem's own kernel struct, and adding a new subsystem is a
  one-line edit to `Kernels` (plus the new subsystem file). `device.rs`
  does not name any individual kernel.

  The subsystem field set is:

  | Field         | Sub-struct type      | Sub-struct home                                | PTX module      | Kernels                                                                                           |
  | ---           | ---                  | ---                                            | ---             | ---                                                                                              |
  | `fill`        | `FillKernels`        | `src/gpu/fill.rs`                              | `fill`          | `fill`                                                                                            |
  | `integrate`   | `IntegrateKernels`   | `src/integrator/velocity_verlet.rs`            | `integrate`     | `vv_kick_drift`, `vv_kick`, `vv_kick_drift_lossless`, `vv_kick_lossless`                          |
  | `reduce`      | `ReduceKernels`      | `src/gpu/pair_buffer.rs`                       | `reduce`        | `reduce_pair_forces`                                                                              |
  | `lj`          | `LjKernels`          | `src/forces/lj.rs`                             | `pair_force`    | `lj_pair_force`                                                                                   |
  | `coulomb`     | `CoulombKernels`     | `src/forces/coulomb.rs`                        | `coulomb`       | `coulomb_pair_force`                                                                              |
  | `spme_real`   | `SpmeRealKernels`    | `src/forces/spme.rs`                           | `spme_real`     | `spme_real_pair_force`                                                                            |
  | `spme_recip`  | `SpmeRecipKernels`   | `src/forces/spme.rs`                           | `spme_recip`    | `spme_charge_spread`, `spme_influence_multiply`, `spme_force_gather`                              |
  | `langevin`    | `LangevinKernels`    | `src/integrator/langevin_baoab.rs`             | `langevin`      | `lan_drift_half`, `lan_ou_step`                                                                   |
  | `morse`       | `MorseKernels`       | `src/forces/morse.rs`                          | `morse`         | `morse_bond_force`, `reduce_bond_forces`                                                          |
  | `angle`       | `AngleKernels`       | `src/forces/angle.rs`                          | `angle`         | `harmonic_angle_force`, `reduce_angle_forces`                                                     |
  | `nose_hoover` | `NoseHooverKernels`  | `src/integrator/nose_hoover_chain.rs`          | `nose_hoover`   | `kinetic_energy_reduce`, `rescale_velocities`                                                     |
  | `andersen`    | `AndersenKernels`    | `src/integrator/andersen.rs`                   | `andersen`      | `andersen_resample`                                                                               |
  | `barostat`    | `BarostatKernels`    | `src/gpu/barostat_kernels.rs`                  | `barostat`      | `virial_sum_reduce`, `rescale_positions`                                                          |
  | `mtk`         | `MtkKernels`         | `src/integrator/mtk_npt.rs`                    | `mtk`           | `mtk_velocity_half_kick`, `mtk_position_drift`                                                    |
  | `shake`       | `ShakeKernels`       | `src/integrator/shake.rs`                      | `shake`         | `shake_snapshot`, `shake_positions`, `rattle_velocities`, `constraint_virial_scatter`, `shake_positions_no_velocity`                                                                                                                                              |
  | `forces`      | `ForcesKernels`      | `src/forces/mod.rs`                            | `forces`        | `accumulate_forces`                                                                               |
  | `neighbor`    | `NeighborKernels`    | `src/forces/neighbor_list.rs`                  | `neighbor`      | `neighbor_displacement_squared`, `neighbor_list_build`, `copy_positions_into_reference`, `compute_cell_indices_and_histogram`, `prefix_scan_local_blocks`, `prefix_scan_apply_block_totals`, `prefix_scan_finalize_offsets`, `scatter_atoms_into_cells`, `sort_cells_by_particle_id` |

  Each subsystem's sub-struct carries one `pub <name>: CudaFunction`
  field per kernel listed in its row, and provides an associated
  function `pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError>`
  that loads the PTX module and pulls the function handles. The struct
  derives `Debug + Clone` (a `CudaFunction` is itself cheap to clone).

  Cross-subsystem reads follow the home rule: a kernel lives in the
  sub-struct named after its `.cu` file, and consumers in other
  subsystems reach into that sub-struct directly. For example, the
  pair-reduction kernel lives in `kernels.reduce.reduce_pair_forces`
  and is consumed by `lj_pair_force`, `coulomb_pair_force`, and
  `spme_real_pair_force` launch wrappers; the
  `kinetic_energy_reduce` and `rescale_velocities` kernels live in
  `kernels.nose_hoover.*` and are consumed by NHC, CSVR, and the MTK
  barostat substep. No kernel handle is duplicated across sub-structs.

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
PTX module (see the *Types* table above for the full home list).
`src/gpu/device.rs` carries `init_device`, `GpuContext`, `GpuError`,
and the central `Kernels` struct whose fields are typed sub-structs —
it does not name any individual kernel.

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

  @rq-123f560e
  Scenario: build.rs fails with a clear message when nvcc is missing
    Given nvcc is not available on PATH
    When cargo build is run
    Then the build fails
    And the error output contains "nvcc"
    And the error output contains "CUDA Toolkit"

  @rq-a199e0aa
  Scenario: build.rs fails with a clear message when nvcc rejects the source
    Given a file "kernels/fill.cu" containing invalid CUDA syntax
    And nvcc is available on PATH
    When cargo build is run
    Then the build fails
    And the error output contains the nvcc error message

  @rq-85baad9d
  Scenario: build.rs reruns when a kernel source file changes
    Given "kernels/fill.cu" has been modified since the last build
    When cargo build is run
    Then build.rs executes again
    And fill.ptx is recompiled

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
    Then gpu_context.kernels has fields named after each subsystem
      in the Types table: fill, integrate, reduce, lj, coulomb,
      spme_real, spme_recip, langevin, morse, angle, nose_hoover,
      andersen, barostat, mtk, settle, forces, neighbor
    And each field is the matching subsystem's typed kernel struct

  @rq-6745e7c5
  Scenario: Each subsystem's XKernels::load returns its kernel handle
    Given a CUDA-capable GPU initialized via CudaDevice::new(0)
    When LjKernels::load(&device) is called
    Then it returns Ok(LjKernels) whose `pair_force` field is a
      launchable CudaFunction
    And the PTX module `pair_force` is now loaded on the device

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
    Then the LJ pair-force launch wrapper reads
      `particle_buffers.kernels.lj.pair_force` for its own kernel
    And reads `particle_buffers.kernels.reduce.reduce_pair_forces` for
      the shared pair-reduction kernel
    And the same `reduce_pair_forces` handle is used by the Coulomb
      and SPME-real launch wrappers

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
