# Introduction

HeddleMD is a GPU-accelerated molecular dynamics engine written in Rust with
CUDA compute kernels. Its primary design goal is **bit-wise reproducibility**:
identical inputs produce byte-identical trajectory and log files across runs
on the same GPU.

This book is the user-facing guide. If you want the internal design — data
flow, kernel-by-kernel breakdown, the deterministic-reduction strategy — read
`docs/architecture.md` in the repository instead.

## What's in this book

- **Getting Started** walks through installation and running the bundled
  10,000-atom Lennard-Jones argon example.
- **User Guide** covers how to write your own simulation: TOML config files,
  extended-XYZ init files, output formats, and what reproducibility does and
  does not guarantee.
- **Reference** documents the CLI.
