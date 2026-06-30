# 65,536-molecule rigid SPC/E water, NPT, with RDF post-processing

A larger variant of the `spc-water-8192` example, scaled up by 8× in
system size to stress the engine at a system size where pair-force
kernel throughput, neighbour-list rebuild cost, and SPME reciprocal-
space FFT cost all become first-order concerns. 65,536 SPC/E water
molecules (196,608 atoms) relaxed by a steepest-descent minimization,
then integrated for 500 fs of NVT equilibration followed by 250 fs
of NPT production at 298.15 K, 1 atm, then analyzed for the O-O,
O-H, and H-H radial distribution functions of the production
trajectory.

The step counts are scaled down ~8–10× from the 8192-molecule example
to keep total wall-clock time within the same order of magnitude;
this means the run is **even less converged** than the 8192-molecule
bundle — it is primarily a performance and scaling benchmark, not a
production-quality calculation.

## Layout

- `water.in.toml` — simulation config (rigid SPC/E + SHAKE + SPME,
  three phases: `min`, `equil`, and `prod`).
- `water.in.xyz` — initial state: 32 × 32 × 64 simple-cubic lattice
  of water molecules at sub-liquid density (~0.47 g/cm³), each with a
  randomized SO(3) orientation seeded for determinism.
- `water.in.topology` — 65,536 SHAKE constraint groups, one per
  molecule.
- `water.in.analysis` — declares three RDF analyses (O-O, O-H, H-H,
  each 200 bins over `[0, 7 Å]`) against the `prod` phase's
  trajectory.
- `generate_init.py` — regenerates `water.in.xyz` deterministically
  (Shoemake's uniform-on-SO(3) rotation seed = 42).
- `generate_topology.py` — regenerates `water.in.topology`
  deterministically.

## Run

From this directory:

```
cargo run --release -- run water.in.toml
```

The combined run is roughly:

| Phase   | Steps / iters | Wall-clock cost (rough)  |
|---------|---------------|--------------------------|
| `min`   | up to 500     | ~30 s — converges in ~80 |
| `equil` | 250           | ~5 min — 500 fs NVT      |
| `prod`  | 125           | ~3 min — 250 fs NPT      |

Total runtime is a few minutes on a recent NVIDIA GPU. Compared with
the 8192-molecule example, every per-step cost goes up roughly 8×
(pair-force kernel, neighbour-list rebuild, SPME reciprocal pipeline,
all scale linearly with N); the step counts have been reduced by
8–10× to keep total wall time comparable.

The run produces several output files in this directory — one
`.timings` per phase, plus a `.log` for the MD phases and a `.minlog`
for the minimization phase. No phase writes a trajectory in the
default config (the example focuses on the `prod` log + analysis;
turn on `[phase.output].trajectory_every` to write frames):

- `water.out.min.minlog` — per-iteration `iter, energy, max_force,
  step, accepted` rows from the SD minimization phase.
- `water.out.min.timings` — minimization performance summary.
- `water.out.equil.log` — `step, time, kinetic_energy, temperature,
  csvr_conserved` (NVT diagnostics), one row per 50 steps.
- `water.out.equil.timings` — equilibration performance summary.
- `water.out.prod.log` — `step, time, kinetic_energy, temperature,
  csvr_conserved, pressure, box_volume, c_rescale_conserved` (NPT
  diagnostics), one row per 25 steps.
- `water.out.prod.timings` — production performance summary.

## Analyze

After `heddlemd run` has produced `water.out.prod.xyz`, post-process
with:

```
cargo run --release -- analyze water.in.analysis
```

This writes three CSV files next to the analysis input:

- `water.out.o-o.csv` — `r, g_r, count` for the O-O RDF.
- `water.out.o-h.csv` — O-H RDF.
- `water.out.h-h.csv` — H-H RDF.

The analyze step is CPU-only and walks every selected frame
enumerating every pair within `r_max = 7 Å`. With 65,536 waters
and a 7 Å cutoff, expect several minutes per frame.

## What scales with N relative to the 8192-molecule example

| Subsystem | Scaling | 8192 → 65,536 (×8) |
|---|---|---|
| Pair-force kernel (per step) | O(N) at fixed cutoff | ~8× |
| Neighbour-list rebuild (per rebuild) | O(N) cell-list + O(N · neighbours) sweep | ~8× |
| SPME real-space pair pass | O(N) | ~8× |
| SPME reciprocal (spread, gather) | O(N) | ~8× |
| SPME reciprocal (FFTs) | O(M log M), M = grid cells | ~9–10× (grid 8×, log 1.1×) |
| Integrator / SHAKE / RATTLE | O(N) | ~8× |
| Neighbour list buffers (memory) | O(n_blocks²) initial alloc | ~64× pre-grow, ~8× after first rebuild |

The neighbour-list buffer initial allocation is sized to the all-
pairs upper bound `n_blocks²`, which grows quadratically in N. For
65,536 molecules the initial allocation is ~1 GB of GPU memory; after
the first rebuild the buffers are grown to fit the actual interaction
count which is O(N) and much smaller. On GPUs with limited memory
this initial sizing may need to be reduced via the (optional) config
fields documented in `rqm/forces/packed-neighbour-pair-force.md`.

## Caveats

This bundle is a **scaling benchmark**, not a production-quality
calculation. In addition to the caveats from the 8192-molecule
example (NPT-RDF normalisation, slow equilibration from a lattice
initial state) the step counts here are reduced 8–10× so:

- The equilibration phase (500 fs NVT) is too short to reach a
  steady-state thermostat-coupled temperature distribution.
- The production phase (250 fs NPT) produces only ~5 logged
  frames and no trajectory by default. RDF peak positions remain
  meaningful (they reflect the actual minimum-image distances under
  each frame's per-frame box) but the running statistics are noisy.

For converged-RDF runs, scale the step counts up to match the
8192-molecule example's resolution per molecule (i.e., 1000 NPT steps
gives equivalent sampling per-molecule at this size and is therefore
8× more total wall time).

## Parameters

Same SPC/E parameters as the 8192-molecule example. Only the FFT
grid changes: `(96, 96, 192)` (vs `(48, 48, 96)`), preserving the
~1.33 Å grid spacing at the doubled box size.
