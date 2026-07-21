# Adding a thermostat

A thermostat couples the system to a temperature bath. It is a
**named-selection** registry slot: the user picks it with
`[phase.thermostat] kind = "..."`, composed with the `velocity-verlet`
integrator. Read the [overview](index.md) first for the registry, config,
CUDA, and determinism machinery this page builds on.

Templates in the tree, easiest first:

- `src/integrator/berendsen.rs` — deterministic, post-only, reuses shared
  kernels (**no new `.cu` file**). The best starting point.
- `src/integrator/csvr.rs` — stochastic, reuses shared kernels, keeps a
  device-side conserved-quantity accumulator.
- `src/integrator/andersen.rs` — stochastic **with its own kernel**; copy
  this if you need bespoke device work.
- `src/integrator/nose_hoover_chain.rs` — the complex case: a leading
  `apply_pre` half, host-side chain arithmetic, `graph_compatible = false`.

## What a thermostat has to do

A thermostat implements the `Thermostat` trait (`src/integrator/mod.rs`):

- `apply_pre(&mut self, buffers, dt, timings)` — the leading (pre-walk)
  coupling half. Defaults to a no-op; only symmetric-Trotter thermostats
  (Nosé-Hoover) override it.
- `apply_post(&mut self, buffers, dt, timings)` — the trailing coupling.
  This is the one method you must implement. It reduces the **full-step**
  kinetic energy of the post-trailing-kick velocities and rescales from
  it.
- `log_column_names` / `log_column_values` — optional CSV diagnostic
  columns (conventionally a `*_conserved` energy column).

Two things about the runtime contract that shape the code:

- **You are called only on coupling steps, and `dt` is already the
  effective coupling timestep** `coupling_interval · base_dt`. Do not
  re-scale by the interval. The runner owns the step counter and the
  wrapping; the `coupling_interval` field is peeled from the config before
  your builder ever sees the params, so it is not a field in your params
  struct.
- **The kinetic-energy reduction is a fusion barrier.** A global reduction
  can't live inside a per-particle kernel without the order-dependent
  atomics the reproducibility strategy forbids, so your KE reduce and your
  rescale each run as their own standalone launches. This is structural —
  see `docs/architecture.md`.

Only `apply_post` is mandatory. Both hooks must early-return `Ok(())` when
`buffers.particle_count() == 0`.

## Reusing the shared kernels (the common case)

A temperature-rescaling thermostat needs no new CUDA. The `nose_hoover`
PTX module already exports the two kernels every rescale thermostat reuses,
wrapped in `src/gpu/kernels.rs`:

- `compute_kinetic_energy(buffers, &mut ke_scratch)` — the deterministic
  full-step KE reduction (`ke_scratch` is a length-1 device buffer you
  allocate once in your constructor).
- `rescale_velocities(buffers, lambda)` (uniform host factor) and
  `rescale_velocities_device_factor(...)` (factor left on device).

So `apply_post` is: reduce KE → compute the rescale factor `λ` on the host
→ `rescale_velocities`. Berendsen is exactly this shape.

If you keep the factor on device and never copy a scalar back per step,
your thermostat stays `graph_compatible = true` (the default). If you read
device scalars into host fields between launches — as Nosé-Hoover does for
its chain — override `graph_compatible` to `false`.

## Determinism and RNG

For a stochastic thermostat, draw noise from the counter-based Philox
generator, never from wall-clock or thread state. The reproducible pattern
(see CSVR and Andersen) is a **device-resident per-slot draw counter** that
the kernel reads and increments, so every launch — and every CUDA-graph
replay — draws a fresh, deterministic stream. Store the `seed` as a plain
`u64` field in your params struct (it is dimensionless, so the `Convert`
derive leaves it alone) and split it into `seed_lo`/`seed_hi` for the
kernel. The host-side `philox_4x32_10` in `src/integrator/philox.rs` is
bit-identical to the device helper in `kernels/philox.cuh`.

## Manifest

### New files

**`src/integrator/my_thermostat.rs`** — the whole slot: the typed params
struct, the state object, the `Thermostat` impl, and the builder. Skeleton
(Berendsen-shaped, no new kernel):

```rust
use serde::Deserialize;
use crate::gpu::{GpuContext, GpuError, ParticleBuffers,
                 compute_kinetic_energy, rescale_velocities};
use crate::io::config::ConfigError;
use crate::registry::{KindedBuilder, convert_params_in_place};
use crate::timings::{KernelStage, Timings};
use crate::precision::Real;
use super::{Thermostat, ThermostatBuilder, ThermostatError};

#[derive(Debug, Clone, Deserialize, serde::Serialize, crate::units::Convert)]
#[serde(deny_unknown_fields)]
pub struct MyThermostatParams {
    pub temperature: crate::units::Temperature, // dimensioned → converted
    pub tau: crate::units::Time,                // dimensioned → converted
    // pub seed: u64,                            // stochastic only; Convert no-op
}

#[derive(Debug)]
pub struct MyThermostat {
    temperature: f64,
    tau: f64,
    g_dof: u32,                         // thermal DOF: (3N − n_constraints − 3).max(1)
    ke_scratch: cudarc::driver::CudaSlice<Real>, // length-1, allocated once
}

impl Thermostat for MyThermostat {
    // apply_pre defaults to no-op — override only for a leading half.
    fn apply_post(&mut self, buffers: &mut ParticleBuffers, dt: Real,
                  timings: &mut Timings) -> Result<(), ThermostatError> {
        if buffers.particle_count() == 0 { return Ok(()); }        // mandatory
        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let lambda = /* relaxation factor from k, dt, tau, g_dof */ 1.0;
        timings.kernel_start(KernelStage::MY_THERMOSTAT_RESCALE)?;
        rescale_velocities(buffers, lambda as Real)?;
        timings.kernel_stop(KernelStage::MY_THERMOSTAT_RESCALE)?;
        Ok(())
    }

    fn log_column_names(&self) -> &'static [(&'static str, crate::units::Dimension)] {
        &[("mythermostat_conserved", crate::units::Dimension::Energy)]
    }
    fn log_column_values(&self, ke: f64, pe: f64) -> Vec<f64> { vec![ke + pe] }
}

#[derive(Debug, Clone)]  // Clone is required (registry_builder_clone! needs it)
pub struct MyThermostatBuilder;

impl KindedBuilder for MyThermostatBuilder {
    fn kind_name(&self) -> &'static str { "my-thermostat" }  // the TOML `kind`
    fn convert_params(&self, units: crate::units::UnitSystem, params: &mut toml::Value)
        -> Result<(), ConfigError> {
        convert_params_in_place::<MyThermostatParams>(units, params)
    }
}

impl ThermostatBuilder for MyThermostatBuilder {
    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> {
        let p: MyThermostatParams = params.clone().try_into()
            .map_err(/* → ConfigError */)?;
        // domain checks: temperature/tau finite & strictly positive …
        Ok(())
    }
    // graph_compatible defaults to true; override to false if apply_* does
    // host arithmetic between launches or a per-step dtoh copy.
    fn build(&self, gpu: &GpuContext, particle_count: usize, n_constraints: usize,
             params: &toml::Value) -> Result<Box<dyn Thermostat>, ThermostatError> {
        let p: MyThermostatParams = params.clone().try_into().map_err(/* … */)?;
        // g_dof = ((3*particle_count) − n_constraints − 3).max(1);
        // ke_scratch = gpu.device.alloc_zeros::<Real>(1)?;
        Ok(Box::new(/* MyThermostat { … } */))
    }
}
```

**`rqm/integration/my-thermostat.md`** — the requirements spec. Mirror
`rqm/integration/csvr.md`'s section order (Algorithm, Per-Step Kernel
Sequence, Parameters, RNG if stochastic, conserved quantity, Empty State,
Feature API, Determinism, Gherkin scenarios).

### Existing files to edit

- **`src/integrator/mod.rs`** — three lines:
  - `pub mod my_thermostat;` (with the other `pub mod` lines, ~15–30).
  - `pub use my_thermostat::{MyThermostat, MyThermostatBuilder};` (with the
    other re-exports, ~32–50).
  - Add `Box::new(MyThermostatBuilder)` to the `vec![...]` in
    `impl Builtins for dyn ThermostatBuilder` (the roster at ~line 1075).
    This alone makes `kind = "my-thermostat"` resolvable.

- **`src/integrator/nose_hoover_chain.rs`** — only if you added a distinct
  rescale timing stage: add one line (e.g.
  `MY_THERMOSTAT_RESCALE = "my_thermostat_rescale",`) to the `stages { … }`
  block of that file's `gpu_kernels!` invocation. (Reusing the existing
  `KINETIC_ENERGY_REDUCE` stage avoids even this.)

**No change needed** to `src/io/config.rs` (kind dispatch is
registry-driven; `validate_params` and `convert_params` are called by
lookup), `src/registries.rs`, `build.rs`, or `src/gpu/device.rs`.

### If you need a bespoke device kernel

Follow the Andersen pattern (see the [overview](index.md) CUDA section):
add `kernels/my_thermostat.cu` (auto-compiled to `crate::kernels::MY_THERMOSTAT`),
a `gpu_kernels!` block in `my_thermostat.rs`, one field in the
`define_kernels!` manifest in `src/gpu/device.rs`, and a `gpu_launch!`
wrapper in `src/gpu/kernels.rs` re-exported from `src/gpu/mod.rs`.

### Tests

- `tests/integrators_my_thermostat.rs` — model on `tests/integrators_csvr.rs`:
  drive `apply_post` over `ParticleBuffers`, assert the post-rescale kinetic
  energy relaxes toward the target, and assert the `particle_count == 0`
  no-op.
- `tests/integrator_framework.rs` — add a lint assertion that
  `ThermostatRegistry::with_builtins()` contains a builder with
  `kind_name() == "my-thermostat"`.
- Config round-trip / unit-conversion cases in `tests/io_config.rs` and
  `tests/unit_conversion.rs`.

## Gotchas

- **A later same-kind registration never shadows a built-in.** `lookup`
  returns the first match and built-ins are seeded first, so registering
  another `"my-thermostat"` after the built-ins is dead code — the built-in
  wins. Choose a fresh `kind`, or build the registry from `Registry::new()`
  with explicit `register_*` calls to replace one.
- **`dt` is already `coupling_interval · base_dt`** on the coupling steps
  you're called — don't re-scale.
- **Allocate all device scratch in the constructor**, never per step;
  `ke_scratch` must be length 1.
- **`graph_compatible` must be truthful** — return `false` if you read a
  device scalar to the host between launches.
