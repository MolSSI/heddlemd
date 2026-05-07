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
nvcc --ptx -o <out_dir>/<stem>.ptx <source_path>
```

where `<out_dir>` is the value of the `OUT_DIR` environment variable set by
Cargo and `<stem>` is the file name without the `.cu` extension.

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
device with all compiled PTX modules loaded.

### Functions <!-- rq-c38c8f3b -->

- `init_device() -> Result<Arc<CudaDevice>, GpuError>`
  - Calls `CudaDevice::new(0)` to initialize the default CUDA device.
  - Loads each embedded PTX constant into the device using
    `CudaDevice::load_ptx`.
  - Returns the device wrapped in an `Arc`.
  - Returns `Err(GpuError)` if device creation or PTX loading fails.

### Types <!-- rq-2093594f -->

- `GpuError` — error type for GPU operations. Wraps the underlying
  `cudarc::driver::DriverError`.

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

1. Call `init_device()` to obtain a `CudaDevice`.
2. Allocate a device buffer of 1024 `f32` elements using `CudaDevice::alloc_zeros`.
3. Launch the `fill` kernel with `value = 1.0` and `n = 1024`.
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
    device.rs             # init_device(), GpuError
tests/
  smoke_gpu.rs            # integration test for fill kernel
Cargo.toml                # depends on cudarc
```

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
    Then it returns Ok containing a CudaDevice
    And the fill PTX module is loaded on the device

  @rq-299c69c9
  Scenario: init_device fails when no GPU is available
    Given no CUDA-capable GPU is available
    When init_device() is called
    Then it returns Err(GpuError)

  # --- Fill kernel correctness ---

  @rq-9cf657ed
  Scenario: fill kernel writes 1.0 to every element of a block-aligned buffer
    Given a CudaDevice with the fill module loaded
    And a device buffer of 1024 f32 elements initialized to zero
    When the fill kernel is launched with value=1.0 and n=1024
    And the buffer is copied back to the host
    Then all 1024 host elements are exactly 1.0_f32

  @rq-26d8c08c
  Scenario: fill kernel handles a non-block-aligned buffer size
    Given a CudaDevice with the fill module loaded
    And a device buffer of 1000 f32 elements initialized to zero
    When the fill kernel is launched with value=1.0 and n=1000
    And the buffer is copied back to the host
    Then all 1000 host elements are exactly 1.0_f32

  @rq-d920e446
  Scenario: fill kernel does not write beyond the buffer
    Given a CudaDevice with the fill module loaded
    And a device buffer of 1024 f32 elements initialized to zero
    When the fill kernel is launched with value=1.0 and n=1000
    And the buffer is copied back to the host
    Then the first 1000 elements are exactly 1.0_f32
    And the remaining 24 elements are exactly 0.0_f32
```
