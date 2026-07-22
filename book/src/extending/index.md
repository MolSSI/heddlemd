# Extending HeddleMD

This section explains how to add new physical models to HeddleMD. It is aimed
at scientists who are comfortable describing a molecular-dynamics algorithm
but may be new to Rust and CUDA. Before continuing, read the [Developer Guide
overview](../developer/index.md), especially its explanations of host and
device code, Rust traits, device buffers, and deterministic reductions.

Choose the guide that matches the physical role of your new method:

- [Adding a thermostat](adding-a-thermostat.md)
- [Adding an integrator](adding-an-integrator.md)
- [Adding a barostat](adding-a-barostat.md)
- [Adding a pair potential](adding-a-pair-potential.md)
- [Adding a bonded potential](adding-a-bonded-potential.md)

The guides use the same overall workflow: choose a nearby built-in, describe
and validate the configuration, implement the runtime behavior, register the
builder, and test both the physics and the framework contract.

## Shared terminology

| Term | Meaning in HeddleMD |
| --- | --- |
| Registry | A collection of builders for one category of component |
| Builder | A small object that validates configuration and constructs a runtime component |
| Slot | A constructed component used in a simulation, such as one thermostat |
| Params | The configuration values owned by a builder |
| Roster | The list of builders included with HeddleMD |
| Host | Rust code and memory on the CPU |
| Device | CUDA code and memory on the GPU |
| Scratch buffer | Reusable temporary device memory allocated during construction |
| Reduction | An operation that combines many values into one, such as total kinetic energy |

## Decide how the component is selected

HeddleMD has seven registries, collected in `Registries` in
`src/registries.rs`:

| Component | Builder trait | How configuration selects it | Built-in roster |
| --- | --- | --- | --- |
| Integrator | `IntegratorBuilder` | By name | `src/integrator/mod.rs` |
| Thermostat | `ThermostatBuilder` | By name | `src/integrator/mod.rs` |
| Barostat | `BarostatBuilder` | By name | `src/integrator/mod.rs` |
| Constraint type | `ConstraintBuilder` | By name | `src/integrator/constraint.rs` |
| Minimizer | `MinimizerBuilder` | By name | `src/minimizer/mod.rs` |
| Analysis | `AnalysisBuilder` | By name | `src/analysis/mod.rs` |
| Potential | `PotentialBuilder` | By the configured interactions | `src/forces/mod.rs` |

The first six use **named selection**. For example, the configuration
`[phase.thermostat] kind = "csvr"` asks the thermostat registry for the
builder whose `kind_name()` is `"csvr"`. Only one thermostat and one
integrator can occupy their respective slots in a phase.

Potentials use **compositional activation** because a force field normally
contains several contributions. Every potential builder examines the
configuration. It returns a component when its interaction type is present
and returns no component otherwise. The force field is the sum of all active
potential components.

## Understand the builder and runtime component

Most extensions contain two conceptually different Rust types:

1. The **builder** is small and normally contains no fields. It owns the
   configuration schema, validates values, and constructs the component.
2. The **runtime component** holds parameters, device buffers, and any state
   that changes during a simulation. It implements `Thermostat`, `Barostat`,
   `Integrator`, or `Potential`.

For named components, the builder also implements `KindedBuilder`, which
supplies the configuration name and unit conversion. A typical declaration is
`struct MyThermostatBuilder;`: the trailing semicolon defines a zero-field
struct. `#[derive(Clone)]` is sufficient for the registry's generated cloning
support.

## Define and validate configuration parameters

Named slots are initially parsed into a `kind` string and an untyped
`toml::Value` containing the remaining parameters. The selected builder owns
the interpretation of that value. Its `validate_params` method should:

1. Deserialize the TOML value into a typed parameter struct.
2. Reject unknown fields with `#[serde(deny_unknown_fields)]`.
3. Check physical and numerical restrictions, such as finite positive
   temperatures, timescales, and cutoffs.
4. Return a user-facing `ConfigError` that identifies the invalid field.

Potentials route parameter tables through a **parameter claim**. A claim is a
category and kind pair, such as `(PairInteraction, "buckingham")`. It tells
the loader which builder validates and converts each interaction entry.

### Preserve physical units

Configuration values may be expressed in SI or Hartree atomic units. Values
must be converted to atomic units before the runtime component is built. Use
the dimension types in `crate::units` rather than representing every value as
a plain `f64`:

```rust
#[derive(
    Debug,
    Clone,
    serde::Deserialize,
    serde::Serialize,
    crate::units::Convert,
)]
#[serde(deny_unknown_fields)]
struct MyParams {
    temperature: crate::units::Temperature,
    tau: crate::units::Time,
    seed: u64,
}
```

`Temperature` and `Time` are converted according to the user's unit system.
The dimensionless integer seed is left unchanged. For a named builder,
`convert_params` can normally delegate to
`convert_params_in_place::<MyParams>(units, params)`.

Do not manually apply conversion factors in `build`. That duplicates policy
and makes it easy for SI and atomic-unit inputs to behave differently.

## Add GPU work only when necessary

First look for an existing launch wrapper in `src/gpu/kernels.rs`. Many new
thermostats and integrators can reuse the kinetic-energy, rescaling, kick, and
drift kernels already provided by HeddleMD.

If the physical method requires a new CUDA kernel, the wiring is:

1. Add a `.cu` file under `kernels/`. `build.rs` discovers it and compiles it
   to embedded PTX automatically.
2. Declare its kernel handles with `gpu_kernels!` in the appropriate Rust
   subsystem file.
3. Add that handle set to the `define_kernels!` manifest in
   `src/gpu/device.rs`.
4. Add a launch wrapper in `src/gpu/kernels.rs` and re-export it from
   `src/gpu/mod.rs`.

Use the types and mathematical functions from `kernels/precision.cuh`, such
as `Real`, `R(x)`, and `Real_sqrt`, instead of spelling the precision as
`float` or `double`. Launch wrappers should use the project's `gpu_launch!`
conventions and should not contain their own timing calls; the caller brackets
the launch with the appropriate timing stage.

Pair and bonded potentials are a special case: their physical calculation is
usually supplied as a CUDA source fragment that the framework combines and
compiles at run time. Their individual guides explain that mechanism.

## Preserve reproducibility

Every new component must preserve HeddleMD's fixed-GPU reproducibility
guarantee. In particular:

- Do not accumulate floating-point values in an order determined by thread
  arrival. Avoid multi-writer floating-point `atomicAdd`.
- Use existing deterministic reductions for kinetic energy, virial, and
  force contributions.
- Make per-particle or per-interaction CUDA functions pure: their outputs
  should depend only on their inputs, not on mutable shared state.
- For stochastic per-step device work, use Philox with an explicit seed and a
  device-resident counter. A host counter does not advance correctly when a
  captured CUDA graph is replayed.
- Allocate device scratch during construction and reuse it.

Run the same simulation twice and compare outputs in an end-to-end
reproducibility test when the new algorithm changes per-step numerical work.

## Register the builder

To include a component with HeddleMD, add its builder to the appropriate
`Builtins::builtins()` roster shown in the table above. Module declarations
and public re-exports are separate Rust steps; the feature-specific guide
lists each required edit.

Libraries embedding HeddleMD can also create `Registries` and call methods
such as `register_thermostat` or `register_potential` before invoking
`run_simulation_with_registries`. Registration appends to the registry. A
named registry uses the first builder matching a `kind`, so a later builder
with the same name does not replace an earlier built-in. Potential registries
are different: every builder is consulted during construction, while the
first matching parameter claim owns validation and conversion. A duplicate
potential claim can therefore construct an additional component without
owning its configuration processing. Use distinct names and parameter claims
unless constructing the corresponding registry from empty components.

## Document and test the extension

Requirements specifications live under `rqm/`. For a substantial physical
method, add a neighboring specification that records its equations,
parameters, execution sequence, limitations, determinism requirements, and
behavior for an empty particle set.

Tests normally cover three layers:

1. **Configuration:** valid values, invalid domains, unknown fields, and unit
   conversion.
2. **Framework integration:** registry discovery, successful construction,
   empty-system behavior, and compatibility checks.
3. **Physics and numerics:** a known analytical case, conservation or target
   behavior where appropriate, and deterministic repeated runs.

Each feature-specific guide ends with a concise file checklist. Use it after
understanding the implementation path, rather than treating it as a substitute
for the preceding explanation.
