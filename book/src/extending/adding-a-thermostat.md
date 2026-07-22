# Adding a thermostat

A thermostat couples particle motion to a target temperature. This guide
explains how to add a thermostat that occupies the optional
`[phase.thermostat]` configuration slot.

Use this extension point when the thermostat is separate from the integrator.
It can be combined with `velocity-verlet` or `respa`. Do not use it for an
integrator such as `langevin-baoab` or `mtk-npt`, which owns its thermostat and
rejects a separately configured thermostat.

Read the [extension overview](index.md) first for shared configuration,
registration, unit-conversion, and determinism conventions.

## Choose the closest example

Start from the simplest existing thermostat that resembles your algorithm:

- `src/integrator/berendsen.rs` is the best starting point for a deterministic
  velocity-rescaling method. Its per-step calculation remains on the GPU.
- `src/integrator/csvr.rs` demonstrates a stochastic rescaling method, a
  device-resident Philox counter, and conserved-quantity bookkeeping.
- `src/integrator/andersen.rs` demonstrates a stochastic method that requires
  its own CUDA kernel.
- `src/integrator/nose_hoover_chain.rs` is an advanced example with coupling
  both before and after integration and host-side chain arithmetic. It cannot
  be captured in a CUDA graph.

Copying a nearby implementation is preferable to beginning with the full
skeleton below: the existing file includes complete error handling, timing,
and tests for its particular execution path.

## Understand when a thermostat runs

A runtime thermostat implements the `Thermostat` trait in
`src/integrator/mod.rs`. Two methods can couple it to a timestep:

- `apply_pre` runs before the integrator walk. Its default implementation does
  nothing. Override it only when the mathematical splitting requires an
  initial coupling operation, as in a symmetric Nosé–Hoover scheme.
- `apply_post` runs after the trailing velocity update. Every separately
  configured thermostat must implement this method.

The runner calls these methods only on coupling steps. The `dt` argument is
already the effective coupling time,
`coupling_interval * simulation_timestep`. Do not multiply it by the coupling
interval again. The runner removes `coupling_interval` before passing the
remaining parameter table to your builder.

Both methods must return `Ok(())` without launching a kernel when
`buffers.particle_count() == 0`. Empty particle sets are used by framework
tests and must remain valid.

Thermostats may also provide `log_column_names` and `log_column_values` for
diagnostic or conserved-quantity columns. Follow the naming and unit
conventions of the closest existing thermostat.

## Implement a velocity-rescaling thermostat

Most rescaling thermostats perform three operations:

1. Calculate the kinetic energy after the integrator's final velocity update.
2. Calculate a velocity scale factor, conventionally called `lambda`, from
   the kinetic energy, target temperature, coupling time, and method-specific
   parameters.
3. Multiply every particle velocity by that factor.

The kinetic-energy calculation is a global reduction and the velocity update
is a per-particle operation. They must remain separate kernel launches; trying
to combine the reduction with the velocity update would require the
order-dependent synchronization that HeddleMD avoids.

### Preferred path: keep intermediate values on the GPU

Use this path when the scale-factor calculation can be expressed as a CUDA
kernel:

1. Allocate a one-element `CudaSlice<Real>` for the kinetic energy and another
   for the scale factor when constructing the thermostat.
2. Call `compute_kinetic_energy_on_device`. It writes the result into the
   kinetic-energy buffer without copying it to Rust.
3. Launch a small method-specific kernel that reads the kinetic energy and
   writes the scale factor.
4. Call `rescale_velocities_device_factor` with the scale-factor buffer.

Because all per-step operations remain on the device, this form can retain the
default `graph_compatible() == true`. Berendsen and CSVR follow this pattern.

### Alternative path: calculate the factor on the host

If the algorithm requires substantial CPU-side state or arithmetic, call
`compute_kinetic_energy`, which performs the same deterministic reduction and
then copies the scalar result to the host. Calculate `lambda` in Rust and pass
it to `rescale_velocities`.

This device-to-host copy and intervening Rust calculation cannot be recorded
as an uninterrupted CUDA graph. The builder must therefore override
`graph_compatible` and return `false`. Nosé–Hoover chain is the relevant
example.

Do not mix the two paths accidentally. In particular, calling the
host-returning `compute_kinetic_energy` while leaving graph compatibility at
its default is incorrect.

## Add stochastic sampling safely

Stochastic thermostats use the counter-based Philox generator. Store the
configured seed as a `u64`; it is dimensionless and is not changed by unit
conversion.

For per-step CUDA work, allocate a one-element draw counter on the device. The
sampling kernel reads and advances that counter. A captured CUDA graph then
advances the random stream on every replay. A counter stored only in Rust
would be captured with a fixed value and would repeat random draws.

Use `src/integrator/csvr.rs` or `src/integrator/andersen.rs` as the complete
pattern. The host implementation in `src/integrator/philox.rs` and the device
helper in `kernels/philox.cuh` produce matching draws for the same keys.

## Define the Rust types

The implementation file normally contains four pieces:

1. A typed parameter struct with physical dimension types.
2. A runtime thermostat holding converted parameters and device buffers.
3. An implementation of `Thermostat` for that runtime type.
4. A builder implementing `KindedBuilder` and `ThermostatBuilder`.

The following is an abbreviated outline, not compilable code:

```rust
#[derive(
    Debug,
    Clone,
    serde::Deserialize,
    serde::Serialize,
    crate::units::Convert,
)]
#[serde(deny_unknown_fields)]
struct MyThermostatParams {
    temperature: crate::units::Temperature,
    tau: crate::units::Time,
}

struct MyThermostat {
    temperature: f64,
    tau: f64,
    kinetic_energy: cudarc::driver::CudaSlice<Real>,
    scale_factor: cudarc::driver::CudaSlice<Real>,
}

impl Thermostat for MyThermostat {
    fn apply_post(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: Real,
        timings: &mut Timings,
    ) -> Result<(), ThermostatError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }

        // Time and launch the deterministic kinetic-energy reduction.
        compute_kinetic_energy_on_device(buffers, &mut self.kinetic_energy)?;

        // Launch a method-specific kernel that computes self.scale_factor.

        // Time and apply the device-resident scale factor.
        rescale_velocities_device_factor(buffers, &self.scale_factor)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct MyThermostatBuilder;

impl KindedBuilder for MyThermostatBuilder {
    fn kind_name(&self) -> &'static str {
        "my-thermostat"
    }

    fn convert_params(
        &self,
        units: crate::units::UnitSystem,
        params: &mut toml::Value,
    ) -> Result<(), ConfigError> {
        convert_params_in_place::<MyThermostatParams>(units, params)
    }
}
```

In `validate_params`, deserialize `MyThermostatParams` and reject non-finite
or non-positive temperatures and timescales. In `build`, deserialize the
already converted values, compute the thermal degrees of freedom consistently
with the existing thermostats, allocate device buffers once, and return
`Box<dyn Thermostat>`.

## Register the thermostat

In `src/integrator/mod.rs`:

1. Declare the module with `pub mod my_thermostat;`.
2. Re-export the runtime type and builder with the other thermostat exports.
3. Add `Box::new(MyThermostatBuilder)` to the built-in thermostat roster.

The roster entry makes `kind = "my-thermostat"` resolvable. No central
configuration enum or dispatch statement needs to change.

If you add a CUDA file, also follow the kernel wiring described in the
[extension overview](index.md). A thermostat that reuses existing kernels
does not need changes to `build.rs` or the aggregate device-kernel manifest.

## Write the specification and tests

Add `rqm/integration/my-thermostat.md`. Use
`rqm/integration/csvr.md` as a structural example and document:

- the physical equations and coupling location;
- parameters, units, and domain restrictions;
- the per-step kernel sequence;
- random-number keys and counter advancement, if stochastic;
- any conserved or diagnostic quantity;
- empty-system behavior and CUDA-graph compatibility;
- the determinism guarantee.

Add an integration test modeled on the closest existing thermostat. At a
minimum, test:

- builder discovery and successful construction;
- valid, invalid, unknown, and unit-converted parameters;
- no work for an empty particle set;
- relaxation or sampling toward the expected temperature;
- the analytical scale factor for a controlled state;
- repeated-run identity for stochastic methods;
- the declared CUDA-graph compatibility.

## Completion checklist

- [ ] The method belongs in a separate thermostat slot rather than an
      integrator-owned thermostat.
- [ ] `dt` is used as the effective coupling time without additional scaling.
- [ ] Device scratch is allocated during construction, not during a step.
- [ ] `apply_post` handles an empty particle set without launching work.
- [ ] Host/device transfers agree with the `graph_compatible` declaration.
- [ ] Stochastic device work uses a device-resident Philox counter.
- [ ] Parameter units and physical domains are validated.
- [ ] The builder is declared, re-exported, and added to the built-in roster.
- [ ] Physics, configuration, empty-state, and determinism tests are present.
