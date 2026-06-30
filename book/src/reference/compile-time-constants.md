# Compile-Time Tuning Constants

A handful of constants that govern GPU kernel launch configuration and
neighbour-list sizing are fixed in the source at build time. They are **not**
exposed through the TOML configuration file: changing them requires editing the
constant and recompiling. They are collected here for reference.

## Pair-force kernel launch shape

The fused real-space pair-force kernel (`jit_composed_pair_force`) is launched
one block per atom-block ("i-block"), and its launch shape is set by two
constants in `src/forces/jit_composed.rs`:

| Constant | Value | Meaning |
|----------|------:|---------|
| `WARPS_PER_BLOCK` | 8 | Warps per CUDA block. |
| `BLOCK_SIZE` | 256 | Threads per block (`WARPS_PER_BLOCK × 32`). |

## Pair-force occupancy (`PACKED_MIN_BLOCKS_PER_SM`)

| Constant | Value | Meaning |
|----------|------:|---------|
| `PACKED_MIN_BLOCKS_PER_SM` | 4 | Minimum resident blocks per SM requested via `__launch_bounds__`. |

The packed-neighbour pass entry points are declared
`__launch_bounds__(BLOCK_SIZE, PACKED_MIN_BLOCKS_PER_SM)`. This asks the compiler
to cap the kernel's per-thread register count so that at least
`PACKED_MIN_BLOCKS_PER_SM` blocks of `BLOCK_SIZE` threads can be co-resident on
one streaming multiprocessor. On hardware with a 64 K-register file the bound
implies a ceiling of roughly `65536 / (BLOCK_SIZE × PACKED_MIN_BLOCKS_PER_SM)`
registers per thread.

A higher value forces a tighter register cap, which raises the number of
resident warps; too high a value forces the compiler to spill registers to local
memory, which slows the kernel down. `PACKED_MIN_BLOCKS_PER_SM` is the single
knob for this balance.

The value 4 is the spill-free occupancy ceiling measured on SM 8.6: the
forces-and-scalars kernel fits in 63 registers with no spilling (67% theoretical
occupancy), while a value of 5 or more spills and runs slower. The
packed-neighbour kernel is not occupancy-limited on the GPUs measured — raising
occupancy beyond ~50% produced no measurable speedup — so this bound is
throughput-neutral and mainly guards against future register-count growth
silently reducing occupancy.

The single-pair and exclusion-correction passes carry no launch bound; they run
one thread per pair over short lists and are not occupancy-limited.

Adjusting `PACKED_MIN_BLOCKS_PER_SM` never changes results. It constrains only
register allocation and resident-block count, not the per-pair arithmetic or the
order of the deterministic fixed-point force accumulation, so same-GPU
run-to-run reproducibility is unaffected.

## Neighbour-list packed-tile sizing

The packed neighbour list grows its tile and single-pair buffers geometrically.
Two constants in `src/forces/neighbor_list.rs` control this:

| Constant | Value | Meaning |
|----------|------:|---------|
| `DEFAULT_TILE_PAIR_FILL_THRESHOLD` | 0.8 | Occupancy fraction of a buffer's capacity above which it is grown on the next rebuild. |
| `DEFAULT_TILE_PAIR_GROWTH_FACTOR` | 1.5 | Multiplier applied to a buffer's capacity when it is grown. |

A larger growth factor reduces how often buffers are reallocated (each
reallocation re-captures the CUDA graph) at the cost of more GPU memory; a higher
fill threshold packs buffers more tightly before growing them.
