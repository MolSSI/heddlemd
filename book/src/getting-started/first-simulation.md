# Your First Simulation

The repository ships a complete 10,000-atom Lennard-Jones argon system at
`examples/lj-10000-argon/`. It is the fastest way to confirm a working
install and the easiest reference when you start building your own input
decks.

## What's in the bundle

```
examples/lj-10000-argon/
├── sim.toml            # simulation config
├── init.xyz            # initial particle state (10,000 atoms)
└── generate_init.py    # regenerates init.xyz deterministically
```

`sim.toml` runs 100 integration steps at `dt = 1 fs` and a target
temperature of 100 K. Atoms sit on a 20 × 20 × 25 simple-cubic lattice
with 4.0 Å spacing, centred at the origin in an 8 × 8 × 10 nm box. The
pair potential uses standard LJ-argon parameters (`σ = 3.4 Å`,
`ε ≈ 120 k_B`) and the O(N²) all-pairs kernel.

## Run it

From the repository root, with a release build available
(`cargo build --release`):

```
./target/release/dynamics run examples/lj-10000-argon/sim.toml
```

`cargo run --release -- run examples/lj-10000-argon/sim.toml` works the
same way. On a recent NVIDIA GPU the run finishes in roughly a second
and prints one line on stdout:

```
[dynamics] complete: 100 steps in <N> ms (frames: 11, log rows: 21)
```

An exit code of `0` means every requested step ran and every requested
output flushed. Non-zero exit codes are documented in
[the CLI reference](../reference/cli.md).

## What you should see on disk

Three files appear in `examples/lj-10000-argon/` alongside the config:

- **`sim-traj.xyz`** — 11 extended-XYZ frames: the initial state plus
  one frame every 10 steps. Each frame is self-describing (lattice
  vectors, column layout, step index, simulation time) and is itself a
  valid init file. Format details in
  [Output Files](../guide/output.md) and
  [Init Files](../guide/init-files.md).
- **`sim.log`** — 22 lines of CSV: one header plus 21 rows logged every
  5 steps. Columns are `step,time,kinetic_energy,temperature`:
  ```
  step,time,kinetic_energy,temperature
  0,0.000000000e0,2.070973475e-17,9.999999878e1
  5,5.000000000e-15,2.070582761e-17,9.998113260e1
  ...
  ```
  The realised step-0 temperature is exactly 100 K (within `f32`
  storage round-off) because the initial velocities were sampled from a
  Maxwell-Boltzmann distribution and rescaled to the configured target.
- **`sim.timings`** — fixed-width text table with one row per
  instrumented kernel and host stage. Wall-clock measurements vary
  every run by design; the trajectory and log do not.

## Did it work?

Two cheap sanity checks:

1. `sim.log`'s `kinetic_energy` column should hold steady to about
   four significant figures across all 21 rows (energy drift is
   small over 100 steps of NVE).
2. Re-run the command after deleting the three output files. The
   regenerated `sim-traj.xyz` and `sim.log` must be byte-identical to
   the previous run:
   ```
   diff sim-traj.xyz <prior copy>
   diff sim.log     <prior copy>
   ```
   That is the load-bearing
   [reproducibility guarantee](../guide/reproducibility.md). `sim.timings`
   will differ — that file is intentionally non-deterministic.

## Re-running

The runner refuses to overwrite existing outputs. Delete (or move)
`sim-traj.xyz`, `sim.log`, and `sim.timings` between runs, or change the
output paths in `sim.toml`.

## Next steps

- [Writing a Simulation](../guide/writing-simulations.md) — build a config
  and init file for your own system.
- [Configuration Reference](../guide/configuration.md) — every TOML field.
