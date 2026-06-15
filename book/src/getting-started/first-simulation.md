# Your First Simulation

The repository ships a complete 10,000-atom Lennard-Jones argon system at
`examples/lj-10000-argon/`. It is the fastest way to confirm a working
install and the easiest reference when you start building your own input
decks.

## What's in the bundle

```
examples/lj-10000-argon/
├── argon.in.toml       # simulation config
├── argon.in.xyz        # initial particle state (10,000 atoms)
└── generate_init.py    # regenerates argon.in.xyz deterministically
```

The `.in.{toml,xyz}` suffixes follow the file-naming
[convention](../guide/configuration.md#config-filename-convention) the
loader uses to distinguish inputs from outputs: every input file ends
in `.in.<ext>`, and the runner derives the output filenames from the
config's root by replacing `.in.` with `.out.`.

`argon.in.toml` runs 100 integration steps at `dt = 1 fs` and a target
temperature of 100 K. Atoms sit on a 20 × 20 × 25 simple-cubic lattice
with 4.0 Å spacing, centred at the origin in an 8 × 8 × 10 nm box. The
pair potential uses standard LJ-argon parameters (`σ = 3.4 Å`,
`ε ≈ 120 k_B`) and the O(N²) all-pairs kernel.

## Run it

From the repository root, with a release build available
(`cargo build --release`):

```
./target/release/heddlemd run examples/lj-10000-argon/argon.in.toml
```

`cargo run --release -- run examples/lj-10000-argon/argon.in.toml`
works the same way. On a recent NVIDIA GPU the run finishes in roughly
a second and prints one line on stdout:

```
[heddlemd] complete: 100 steps in <N> ms (frames: 11, log rows: 21)
```

An exit code of `0` means every requested step ran and every requested
output flushed. Non-zero exit codes are documented in
[the CLI reference](../reference/cli.md).

## What you should see on disk

Three files appear in `examples/lj-10000-argon/` alongside the config:

- **`argon.out.run.xyz`** — 11 extended-XYZ frames: the initial state plus
  one frame every 10 steps. Each frame is self-describing (lattice
  vectors, column layout, step index, simulation time) and is itself a
  valid init file. Format details in
  [Output Files](../guide/output.md) and
  [Init Files](../guide/init-files.md).
- **`argon.out.run.log`** — 22 lines of CSV: one header plus 21 rows logged
  every 5 steps. Columns are `step,time,kinetic_energy,temperature`:
  ```
  step,time,kinetic_energy,temperature
  0,0.000000000e0,2.070973475e-17,9.999999878e1
  5,5.000000000e-15,2.070582761e-17,9.998113260e1
  ...
  ```
  The realised step-0 temperature is exactly 100 K (within `f32`
  storage round-off) because the initial velocities were sampled from a
  Maxwell-Boltzmann distribution and rescaled to the configured target.
- **`argon.out.run.timings`** — fixed-width text table with one row per
  instrumented kernel and host stage. Wall-clock measurements vary
  every run by design; the trajectory and log do not.

## Did it work?

Two cheap sanity checks:

1. `argon.out.run.log`'s `kinetic_energy` column should hold steady to about
   four significant figures across all 21 rows (energy drift is small
   over 100 steps of NVE).
2. Re-run the command after deleting the three output files. The
   regenerated `argon.out.run.xyz` and `argon.out.run.log` must be
   byte-identical to the previous run:
   ```
   diff argon.out.run.xyz <prior copy>
   diff argon.out.run.log <prior copy>
   ```
   That is the load-bearing
   [reproducibility guarantee](../guide/reproducibility.md).
   `argon.out.run.timings` will differ — that file is intentionally
   non-deterministic.

## Re-running

The runner refuses to overwrite existing outputs. Delete (or move)
`argon.out.run.xyz`, `argon.out.run.log`, and `argon.out.run.timings` between runs,
or set explicit `output.*_path` fields in `argon.in.toml`.

## Validating without running (`heddlemd lint`)

Before queueing a long job on shared GPU hardware, run

```
heddlemd lint examples/lj-10000-argon/argon.in.toml
```

to check the config, init file, output-path collisions, and box-vs-cutoff
geometry on a login node without touching the GPU. Add `--with-gpu` to
extend the lint through GPU initialisation and force-field allocation
when you do have a GPU available. The full reference lives in the
[CLI Reference](../reference/cli.md).

## Next steps

- [Writing a Simulation](../guide/writing-simulations.md) — build a config
  and init file for your own system.
- [Configuration Reference](../guide/configuration.md) — every TOML field.
