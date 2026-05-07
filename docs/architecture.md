# Architecture

Dynamics is a GPU-accelerated molecular dynamics engine written in Rust with
CUDA compute kernels. Its primary design goal is **bit-wise reproducibility**:
given identical inputs, the simulation must produce identical outputs regardless
of GPU thread scheduling.

## Reproducibility strategy

Floating-point addition is not associative. When a GPU reduces forces from
many neighbor interactions into a per-particle sum, the thread execution order
is non-deterministic, so naive atomic accumulation produces different bits on
each run.

Dynamics solves this with **deterministic reduction ordering**: every
floating-point operation is performed on the same inputs in the same order on
every run. No precision is sacrificed (there is no fixed-point quantization),
and the approach requires no hardware features beyond standard 32/64-bit
floats.

The three invariants that make this work:

1. **Deterministic neighbor lists.** Neighbor lists are sorted by particle
   index so that every particle always sees its neighbors in the same order.
2. **No atomic float accumulation.** Force contributions are written to a
   pre-allocated pair buffer at deterministic offsets, not accumulated with
   `atomicAdd`.
3. **Fixed-topology reduction.** Per-particle force sums are computed with a
   segmented reduction kernel that processes contributions in index order.

### Scope of bit-wise reproducibility

The reproducibility guarantee covers **multiple runs on the same GPU**. It
does not extend across different hardware: a CPU reference implementation
and a GPU kernel will not in general agree bit-for-bit, because IEEE-754
permits implementations to fuse multiply-add operations (FMA) and other
contractions that the two backends choose differently. CUDA kernels in this
project are compiled with `nvcc`'s default FMA contraction enabled; the
performance benefit is real and the cross-hardware match is not a property
the architecture promises.

Tests that compare a kernel result against a CPU-computed expected value
should use a small relative tolerance. Tests that compare two GPU runs to
each other use exact equality — that is the load-bearing invariant.

## High-level data flow

```
 Positions ──> Spatial hash ──> Neighbor lists (sorted)
                                      │
                                      v
                              Pair force kernel
                              (one thread per pair,
                               writes to pair buffer)
                                      │
                                      v
                              Pair buffer [N x max_neighbors x 3]
                                      │
                                      v
                              Segmented reduction kernel
                              (deterministic per-particle sum)
                                      │
                                      v
                              Net forces per particle
                                      │
                                      v
                              Integration kernel
                              (velocity Verlet)
                                      │
                                      v
                              Updated positions & velocities
```

Each stage reads from and writes to well-defined buffers with no data races.
The pipeline repeats every timestep.

## Data layout

Following the project convention of structures of arrays (SoA), particle state
is stored as separate contiguous arrays rather than an array of particle
structs:

```
positions_x:  [f32; N]
positions_y:  [f32; N]
positions_z:  [f32; N]
velocities_x: [f32; N]
velocities_y: [f32; N]
velocities_z: [f32; N]
forces_x:     [f32; N]
forces_y:     [f32; N]
forces_z:     [f32; N]
masses:       [f32; N]
particle_ids: [u32; N]   // stable identifiers, never reordered
```

SoA layout maximizes GPU memory coalescing: threads accessing consecutive
particles read consecutive memory addresses.

## Pair buffer

The pair buffer is the key data structure for reproducibility. It is a 2D
array of shape `[N, max_neighbors]` for each force component (x, y, z):

```
pair_forces_x: [f32; N * max_neighbors]
pair_forces_y: [f32; N * max_neighbors]
pair_forces_z: [f32; N * max_neighbors]
```

The pair force kernel writes the force on particle `i` due to neighbor `j` at
index `i * max_neighbors + k`, where `k` is the position of `j` in `i`'s
sorted neighbor list. Each slot is written by exactly one thread, so no
synchronization is needed.

`max_neighbors` is a simulation parameter. If a particle exceeds this count,
the simulation halts with an error rather than silently dropping interactions.

### Memory cost

For `N = 100,000` particles and `max_neighbors = 128`:

```
100,000 * 128 * 3 components * 4 bytes = ~147 MB
```

This fits comfortably on modern GPUs.

## Neighbor list construction

Neighbor lists are rebuilt periodically using a **skin distance** scheme to
amortize the cost of reconstruction over many timesteps.

### Skin distance

The neighbor search uses an expanded cutoff of `r_cut + r_skin`, where
`r_skin` is the skin distance. This means the neighbor list includes particles
that are not yet within interaction range but may drift into range before the
next rebuild.

A rebuild is triggered when any particle has moved more than `r_skin / 2` from
its position at the time the list was last built. Since two particles
approaching each other could each move up to `r_skin / 2`, the worst-case
relative displacement is `r_skin`, which is exactly the extra margin in the
neighbor search radius.

Each particle's position at the last rebuild is stored in a reference position
array. A lightweight check kernel runs every timestep to compute the maximum
displacement from these reference positions. If the maximum exceeds
`r_skin / 2`, the host triggers a full neighbor list rebuild and updates the
reference positions.

### Build procedure

1. **Spatial hash.** Assign each particle to a cell based on position.
   Cell size equals `r_cut + r_skin`.
2. **Sort particles by cell.** A radix sort by cell index produces a
   deterministic ordering within each cell (ties broken by particle index).
3. **Neighbor search.** For each particle, iterate over the 27 adjacent cells
   (3D). Collect all particles within the expanded cutoff `r_cut + r_skin`.
   Sort the resulting neighbor list by particle index.

Sorting by particle index in step 3 is what guarantees the pair force kernel
always processes neighbors in the same order.

### Choosing `r_skin`

Larger skin distances mean fewer rebuilds but more pair force evaluations per
timestep (since the neighbor list includes more particles). The optimal value
depends on the system dynamics and density. A typical starting point is
`r_skin = 0.3 * r_cut`.

## CUDA kernels

### Pair force kernel

- **Grid:** one thread per (particle, neighbor) pair.
- **Input:** positions, neighbor lists (sorted), interaction parameters.
- **Output:** pair buffer entries at deterministic offsets.
- Each thread computes the pairwise force between particle `i` and its `k`-th
  neighbor, then writes the result to `pair_buffer[i * max_neighbors + k]`.
- No atomics, no race conditions.

### Segmented reduction kernel

- **Grid:** one thread (or warp) per particle.
- **Input:** pair buffer, neighbor counts.
- **Output:** net force arrays.
- Sums the `neighbor_count[i]` entries in the pair buffer for particle `i`
  in sequential index order (k = 0, 1, 2, ...).
- The sequential order is what produces identical floating-point results.
  For particles with many neighbors, a fixed tree reduction with a
  deterministic topology (e.g. left-to-right pairwise) may be used instead,
  as long as the tree shape depends only on the neighbor count, not on
  thread scheduling.

### Integration kernel

- **Grid:** one thread per particle.
- Velocity Verlet integration:
  1. `v(t + dt/2) = v(t) + (F(t) / m) * (dt / 2)`
  2. `x(t + dt) = x(t) + v(t + dt/2) * dt`
  3. Compute new forces `F(t + dt)` (requires a full force evaluation)
  4. `v(t + dt) = v(t + dt/2) + (F(t + dt) / m) * (dt / 2)`

  Steps 1-2 and 4 are trivially parallel (one thread per particle). Step 3
  is the full force pipeline described above.

## Rust / CUDA boundary

The host code is written in Rust. CUDA kernels are written in CUDA C and
compiled to PTX at build time. The Rust side uses
[cudarc](https://crates.io/crates/cudarc) to manage devices, streams, GPU
memory, and kernel launches.

```
src/
  main.rs              # entry point, CLI, I/O
  simulation.rs        # timestep loop, orchestration
  state.rs             # SoA particle state, host-side buffers
  gpu/
    device.rs          # CudaDevice setup, PTX module loading
    buffers.rs         # typed wrappers around CudaSlice allocations
    kernels.rs         # kernel launch helpers (grid dims, argument packing)
  neighbor/
    spatial_hash.rs    # cell assignment and sorting
    list.rs            # neighbor list construction and rebuild check
  io/
    trajectory.rs      # output trajectory frames
    config.rs          # simulation parameter parsing

kernels/
  pair_force.cu        # pairwise force computation
  reduce.cu            # segmented force reduction
  integrate.cu         # velocity Verlet integration
  neighbor.cu          # spatial hashing, neighbor search, displacement check
```

cudarc provides safe Rust types for GPU buffers (`CudaSlice<T>`), device
management (`CudaDevice`), and kernel launches (`LaunchAsync`). The wrapper
layer in `gpu/kernels.rs` keeps kernel-specific details (function names, grid
dimensions, parameter ordering) out of the simulation loop.

All kernel launches go through a single `CudaStream` to maintain deterministic
execution order.

## Build

CUDA kernels are compiled to PTX during `cargo build` via a `build.rs` build
script that invokes `nvcc`. The resulting PTX is loaded at runtime using
`CudaDevice::load_ptx`.

## Precision policy

All positions, velocities, and forces use `f32` by default. This is sufficient
for most short-range MD workloads and maximizes GPU throughput.

If higher precision is needed (e.g., for long simulations where energy drift
matters), the data layout and kernels can be widened to `f64`. This should be
a compile-time feature flag, not a runtime branch, to avoid paying any
abstraction cost.

## Boundary conditions

Periodic boundary conditions (PBC) are applied during the neighbor search and
pair force computation. The minimum image convention is used: when computing
the displacement between two particles, the shortest vector accounting for
box periodicity is selected.

## Extensibility

New force models (e.g., bonded interactions, long-range electrostatics via
Ewald/PME) follow the same pattern:

1. Compute pairwise (or per-particle) contributions in a dedicated kernel.
2. Write results to a deterministic buffer.
3. Reduce with a fixed-order summation.

As long as every new kernel writes to pre-indexed buffer slots and avoids
atomic float accumulation, reproducibility is preserved.
