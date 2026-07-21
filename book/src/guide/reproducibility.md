# Reproducibility

Bit-wise reproducibility is the engine's primary design goal. This chapter
spells out exactly what that means, what it does *not* mean, and how to
configure the `lossless` integrator mode that takes the guarantee one
step further (bit-exact time reversal).

## The guarantee

> Two runs of the same config and init file, on the same GPU, produce
> byte-identical `*.out.<phase>.xyz` and `*.out.<phase>.log` files.

"Byte-identical" is meant literally — `diff` and `sha256sum` both say
zero. The guarantee covers:

- Every floating-point value written to the trajectory and log,
- The Maxwell-Boltzmann velocity-generation step (when the init file
  omits velocities), and
- Any thermostat or barostat that draws stochastic noise: each one is
  seeded by an explicit `seed` field in its TOML section, and the RNGs
  are counter-based or deterministic.

## What is *not* covered

### `*.timings`

The timings file is intentionally non-deterministic. Wall-clock
measurements vary every run for reasons that have nothing to do with the
simulation (GPU clocks, OS scheduling, driver state). They live in their
own file so that comparing trajectories against a reference is a clean
yes/no operation. See [Output Files](output.md).

### Cross-hardware bit-exactness

The guarantee is **same GPU only**. A run on an A100 will not produce
the same bytes as a run on an H100, and a CPU reference implementation
will not match either. The reason is fused multiply-add (FMA): IEEE-754
permits implementations to fuse `a*b + c` into a single rounded
operation, and CUDA's `nvcc` is free to make that decision differently
on different hardware (and at different optimisation levels). The
engine ships with `nvcc`'s default FMA contraction enabled because the
performance benefit is real; cross-hardware bit-match is not promised.

For tests, the rule of thumb is:

- GPU vs. same GPU → exact equality.
- GPU vs. CPU reference, or GPU vs. different GPU → small relative
  tolerance (`f32` round-off scale).

### Driver and CUDA-toolkit changes

Reproducibility holds for a single binary against a single GPU. Changing
the CUDA toolkit, the NVIDIA driver, or the Rust compiler can perturb
the generated PTX or the kernel's compiled SASS and break the bit-exact
match. A reference trajectory is only meaningful when checked back
against the binary that produced it.

## Why it works, in one paragraph

Floating-point addition is not associative: `(a + b) + c` and
`a + (b + c)` can differ in the last bits. On a GPU, many threads
contribute to each atom's force in an order that varies run to run, so
simply summing them as they arrive would give a slightly different result
each time. HeddleMD avoids this by fixing the summation order — every run
adds the same contributions in the same order, with no order-dependent
atomic operations — and no precision is sacrificed to do it (there is no
fixed-point quantization). The same discipline applies to the sums that
produce log quantities such as kinetic energy. The mechanics of how this
is arranged on the GPU are an implementation matter that lives in
`docs/architecture.md`; the determinism rules a new kernel must preserve
are summarized in the [Extending HeddleMD](../extending/index.md) section
of this book.

### Fast math and the guarantee

`[simulation].fast_math` (default `true`) builds the JIT kernels with
`nvcc`'s `--use_fast_math`. It does **not** weaken the same-GPU
guarantee: the deterministic force accumulation above does
not depend on the per-pair division/transcendental precision, so a
`fast_math` run is still byte-identical to itself run-to-run. What
changes is *which* f32 trajectory you get — a `fast_math = true` run and
a `fast_math = false` run follow different (each independently
reproducible) trajectories, so compare a reference trajectory only
against a binary built with the same setting.

## Lossless integrator mode

The velocity-Verlet integrator has an opt-in `lossless` mode, enabled
per run via:

```toml
[phase.integrator]
kind = "velocity-verlet"
lossless = true
```

What it does, and what it does *not* do:

### What it does

`lossless` adds a compensated-summation (Kahan-style) low-part to every
particle's position and velocity. Each accumulator becomes an `(f32,
f64)` pair: the `f32` carries the running value, the `f64` carries the
rounding residual. The integrator's `x += v · dt` and `v += a · dt/2`
updates become *exactly invertible* — running the simulation backward
from any timestep recovers the bits of the starting state.

### What it does *not* do

`lossless` is **not** a double-precision simulation. Forces, `a = F/m`,
and `dt` itself are still single-precision regardless of the flag. A
run with `lossless = true` and a run with `lossless = false` follow
different `f32` trajectories — both are reproducible against
themselves; only the former is reversible.

If you want better physical accuracy, you want `f64` storage and `f64`
force kernels. That is not yet implemented and will be a compile-time
feature flag (the `f32` build should pay no abstraction cost).

### Cost

Per-particle memory roughly doubles for the integrator buffers.
Per-step compute is somewhat higher because each update reads and
writes the low-part as well. The exact margin depends on the workload;
for most systems the integrator stage is not the bottleneck, so the
end-to-end slowdown is modest.

### Compatibility

`lossless = true` is currently incompatible with constraint groups in
the topology file. Configs that combine the two are rejected at load
time. The constraint-supporting integrator is the standard
`lossless = false` variant.

## Verifying reproducibility yourself

Run the bundled example, save the outputs, delete and re-run:

```
./target/release/heddlemd run examples/lj-10000-argon/argon.in.toml
mv examples/lj-10000-argon/argon.out.run.xyz     /tmp/traj.ref
mv examples/lj-10000-argon/argon.out.run.log     /tmp/log.ref
mv examples/lj-10000-argon/argon.out.run.timings /tmp/timings.ref
./target/release/heddlemd run examples/lj-10000-argon/argon.in.toml
diff examples/lj-10000-argon/argon.out.run.xyz /tmp/traj.ref
diff examples/lj-10000-argon/argon.out.run.log /tmp/log.ref
```

Both `diff` invocations should print nothing. A `diff` on the timings
file will, by design, show differences.
