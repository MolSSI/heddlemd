# Installation

Dynamics is built from source. There are no pre-built binaries.

## Prerequisites

- **NVIDIA GPU** with a recent driver. The engine is CUDA-only; there is no
  CPU or non-NVIDIA fallback.
- **CUDA Toolkit 11.8 or newer.** `nvcc` must be on `PATH` so the build
  script can compile the kernel sources under `kernels/` to PTX. Verify
  with:
  ```
  nvcc --version
  ```
- **Rust** (Cargo edition 2024). Install via [rustup](https://rustup.rs/).

The build script invokes `nvcc` once per `.cu` file at `cargo build` time
and embeds the resulting PTX into the binary; nothing extra needs to be
installed at runtime.

## Build

From the repository root:

```
cargo build --release
```

This produces the binary at `target/release/dynamics`. A debug build
(`cargo build`, no flag) lives at `target/debug/dynamics` and is suitable
for development but several times slower per timestep.

## Verify the install

Run the bundled example (described in detail in
[Your First Simulation](first-simulation.md)):

```
./target/release/dynamics run examples/lj-10000-argon/argon.in.toml
```

A successful run finishes in roughly a second on a recent GPU and prints
a single `[dynamics] complete: ...` line on stdout.

## Container build

The repository ships a `Containerfile` for Podman/Docker that pins a
known-good toolchain. Use it when you do not want to install CUDA on the
host or when running the project under an AI-assistant sandbox.
