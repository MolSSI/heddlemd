# Architecture

HeddleMD is a GPU-accelerated molecular dynamics engine written in Rust with
CUDA compute kernels. Its primary design goal is **bit-wise reproducibility**:
given identical inputs, the simulation must produce identical outputs regardless
of GPU thread scheduling.

## Reproducibility strategy

Floating-point addition is not associative. When a GPU reduces forces from
many neighbor interactions into a per-particle sum, the thread execution order
is non-deterministic, so naive atomic accumulation produces different bits on
each run.

HeddleMD solves this with **deterministic reduction ordering**: every
floating-point operation is performed on the same inputs in the same order on
every run. No precision is sacrificed (there is no fixed-point quantization),
and the approach requires no hardware features beyond standard 32/64-bit
floats.

The three invariants that make this work:

1. **Deterministic neighbor lists.** Each particle's neighbor list is
   produced in a fixed, deterministic order — the cell-sweep order
   documented in `rqm/forces/neighbor-list.md` — so that every particle
   always sees its neighbors in the same order across runs on the same
   GPU.
2. **No atomic float accumulation.** Each particle's force is accumulated
   inside one CUDA warp's registers; no force value is shared between
   warps and no `atomicAdd` is used.
3. **Fixed-topology warp reduction.** Per-particle force sums are
   computed by a deterministic lane-strided sweep over the neighbour
   list followed by a fixed-shape warp-tree butterfly reduction. The
   sweep order and tree shape depend only on the neighbour count, not
   on thread scheduling.

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
                              Fused pair-force kernel
                              (one warp per particle;
                               sweeps the neighbour list,
                               accumulates in registers,
                               warp-tree reduces to
                               per-particle net force)
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

## Pair force accumulation

Forces stay in per-warp registers from the moment they are computed
until the per-particle total is written. No per-pair intermediate is
materialised on the device. Each warp's 32 lanes accumulate their
assigned per-pair contributions in register-resident scalars and
combine them through a fixed-shape butterfly tree; lane 0 of the warp
writes the per-particle net force, potential-energy share, and
scalar-virial share to the slot output buffer described in
`rqm/forces/framework.md`.

The `(i, k)` contribution — the force on particle `i` due to its
`k`-th neighbour — is computed by lane `k mod 32` of the warp
assigned to particle `i`. Each lane is the sole writer of its own
accumulator across all sweep iterations; no synchronisation between
lanes is needed until the final warp-tree butterfly.

`max_neighbors` is a simulation parameter that sizes the neighbour
list. If a particle exceeds this count, the simulation halts with an
error rather than silently dropping interactions.

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

### Fused pair-force kernel

- **Grid:** one warp per particle, 8 warps per block, `ceil(N / 8)` blocks.
- **Input:** positions, neighbor lists (sorted), interaction parameters,
  per-particle exclusion table.
- **Output:** per-particle net force (and, on `ForcesAndScalars` calls,
  per-particle potential-energy share and scalar-virial share) written
  directly to the slot output buffer.
- The warp sweeps the particle's neighbour list with lane stride 32.
  Each lane reads the per-pair inputs for its assigned neighbours,
  computes the pair functional form, applies the per-pair exclusion
  scale, and adds the contribution to its register accumulators.
- After the sweep, a fixed 5-step warp-pairwise butterfly via
  `__shfl_xor_sync` reduces the 32 lane accumulators to lane 0, which
  writes the per-particle result.
- No atomics, no shared memory, no inter-warp synchronisation.
- The tree shape depends only on the neighbour count, not on thread
  scheduling — that is what makes the per-particle result bit-exact
  across runs on the same GPU.

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
  pair_compute.cuh     # shared warp-per-particle device helper
  pair_force.cu        # fused LJ pair-force kernels (_f, _fev)
  coulomb.cu           # fused truncated-Coulomb pair-force kernels
  spme_real.cu         # fused SPME real-space pair-force kernels
  integrate.cu         # velocity Verlet integration
  neighbor.cu          # spatial hashing, neighbor search, displacement check
```

cudarc provides safe Rust types for GPU buffers (`CudaSlice<T>`), device
management (`CudaDevice`), and kernel launches (`LaunchAsync`). The wrapper
layer in `gpu/kernels.rs` keeps kernel-specific details (function names, grid
dimensions, parameter ordering) out of the simulation loop.

Most kernel launches go through the device's default `CudaStream`. The SPME
reciprocal pipeline (charge spread, R2C FFT, influence-function multiply,
C2R FFT) runs on a dedicated stream owned by `SpmeReciprocalState` so it can
overlap with default-stream work in the same timestep; the two streams write
to disjoint device buffers and synchronise at deterministic event-based
boundaries at the entry and exit of the slot's `contribute` / `reduce`
calls. See `rqm/forces/spme.md` for the topology. Adding further streams
beyond this requires the same disjoint-buffer + event-boundary contract.

## Build

CUDA kernels are compiled to PTX during `cargo build` via a `build.rs` build
script that invokes `nvcc`. The resulting PTX is loaded at runtime using
`CudaDevice::load_ptx`.

## Units

The engine stores and computes in **Hartree atomic units** throughout:
lengths in Bohr radii (`a_0`), masses in electron rest masses (`m_e`),
times in atomic time units (`hbar / E_h` ≈ 24.2 attoseconds), energies
in Hartrees (`E_h`), charges in elementary charges (`e`), and
temperatures as `k_B · T` in Hartrees (so `k_B = 1` exactly inside the
engine, and the Coulomb prefactor `1 / (4πε₀) = 1` exactly inside every
electrostatic kernel).

The TOML configuration file, the extended-XYZ initial-state file, and
all output files (trajectory, CSV log, minimization log) optionally
accept SI input/output through a top-level `units = "si" | "atomic"`
selector — see `rqm/io/unit-system.md`. The conversion to and from
atomic units happens at the I/O boundary; the internal pipeline below
sees atomic units only.

## Precision policy

All positions, velocities, and forces use `f32`. This is sufficient for most
short-range MD workloads and maximizes GPU throughput.

Two precision concerns are easy to conflate; they are independent.

### Storage precision (f32 vs f64)

Widening the particle data layout and the compute kernels to `f64` — for long
simulations where `f32` round-off in the force evaluation itself becomes the
dominant error — is **not yet implemented**. When it is added it should be a
compile-time feature flag, not a runtime branch, so the `f32` build pays no
abstraction cost. Until then the engine is `f32` end to end: the pair-force
kernels, the segmented reduction, and every integrator increment (`a = F/m`,
`v · dt`, and `dt` itself) are single precision.

### Lossless reversible integration

Independently of storage precision, the velocity-Verlet integrator has a
`lossless` mode, selected per run by `[integrator].lossless` in the config.
This is **not** a double-precision simulation. It is compensated (Kahan-style)
summation applied only to the integrator's accumulation: each particle carries
an `f64` low-part for position and velocity that holds the rounding residual
of the running sums, making the `x += v · dt` and `v += a · dt/2` updates
exactly invertible. That exact invertibility is what enables bit-exact time
reversal.

The physics driving a `lossless` run is still `f32`: forces, `a = F/m`, and
`dt` are single precision whether `lossless` is on or off. `lossless` does not
improve physical accuracy the way an `f64` force evaluation would — it makes
the integrator's bookkeeping exact, nothing more. A run with `lossless = true`
and one with `lossless = false` follow different `f32` trajectories; only the
former can be stepped backward to its exact starting state.

## Boundary conditions

Periodic boundary conditions (PBC) are applied during the neighbor search and
pair force computation. The minimum image convention is used: when computing
the displacement between two particles, the shortest vector accounting for
box periodicity is selected.

## Extensibility

New pair force models (e.g., Buckingham, tabulated) follow the fused
warp-per-particle pattern specified in `rqm/forces/pair-force-kernel.md`:

1. Compute the per-pair functional form `(factor, energy, virial)` from
   `(r²)` and per-pair parameters in a `__device__` functor.
2. Plug the functor into the shared `pair_compute_{f,fev}` helper to
   obtain the two `extern "C"` kernel variants for the new potential.
3. Register a `PotentialBuilder` for the slot in
   `PotentialRegistry::with_builtins()`.

New bonded force models (e.g., harmonic angles) follow a similar
pattern but with their own per-bond / per-angle index tables; see
`rqm/forces/framework.md` and the per-potential files. Reproducibility
is preserved as long as each warp's accumulation order depends only on
the neighbour / bond count and on no thread-scheduling decision.

Long-range electrostatics (SPME) uses an inverted iteration pattern — one
thread per grid cell rather than per-particle — to preserve determinism
without an `O(N · p³)` intermediate buffer. See
`long-range-electrostatics.md` for the architectural overview.
