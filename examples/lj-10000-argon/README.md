# 10,000-atom Lennard-Jones argon

A small example that exercises the full simulation pipeline at 10⁴ particles
for 100 timesteps.

## Layout

- `argon.in.toml` — simulation config (SI units; 1 fs timestep; 100 steps at
  100 K)
- `argon.in.xyz` — 10,000 argon atoms on a 20 × 20 × 25 simple-cubic lattice
  with 4.0 Å spacing, centred at the origin in an 8 × 8 × 10 nm box.
- `argon.in.analysis` — example post-processing input declaring an Ar-Ar
  radial distribution function (200 bins, `r_max = 3.5 nm`).
- `generate_init.py` — regenerates `argon.in.xyz` deterministically.

## Run

From this directory:

```
cargo run --release -- run argon.in.toml
```

Or with the debug binary already built:

```
../../target/debug/dynamics run argon.in.toml
```

A run produces three files in this directory:

- `argon.out.xyz` — 11 trajectory frames (steps 0, 10, …, 100), extended-XYZ
- `argon.out.log` — 21 CSV rows of step, time, kinetic energy, temperature
- `argon.out.timings` — per-stage timing summary (kernels and host I/O)

## Analyze

After `dynamics run` has written `argon.out.xyz`, post-process the
trajectory with:

```
cargo run --release -- analyze argon.in.analysis
```

The analyze run writes one CSV per declared analysis. For the bundled
`argon.in.analysis` that is `argon.out.ar-ar.csv` (200 rows of
`r, g_r, count`). See the [Analysis chapter](../../book/src/guide/analysis.md)
for the full file-format reference.

## Notes

- The `[[pair_interactions]]` table uses standard LJ-argon parameters
  (σ = 3.4 Å, ε ≈ 120 k_B). Initial velocities are generated from a
  Maxwell-Boltzmann distribution, the centre-of-mass drift is removed, and
  the result is rescaled so the realised temperature is exactly 100 K
  (RNG seed = 1; deterministic across runs on the same GPU).
- The runner uses the O(N²) pair-force kernel; the pair buffer for this
  example occupies ~1.2 GB of GPU memory.
