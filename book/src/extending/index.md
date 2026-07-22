# Extending HeddleMD

This sub-section of the [Developer Guide](../developer/index.md) covers
adding new physics to the engine: a new thermostat, integrator, barostat,
pair potential, bonded potential, and so on. It is written for someone
modifying the engine source, not for someone running simulations — the
rest of this book covers running simulations.

Almost every capability in HeddleMD is a **plugin behind an open
registry**. Adding one means implementing a small trait and registering
a builder for it; you never edit a central `match` over kinds, and you
never touch unrelated potentials or integrators. This page explains the
machinery all of the "how to add X" pages rely on. Read it once, then
jump to the page for the thing you are adding:

- [Adding a thermostat](adding-a-thermostat.md)
- [Adding an integrator](adding-an-integrator.md)
- [Adding a barostat](adding-a-barostat.md)
- [Adding a pair potential](adding-a-pair-potential.md)
- [Adding a bonded potential](adding-a-bonded-potential.md)

## The seven registries

The engine holds seven registries, bundled in `Registries`
(`src/registries.rs`). Each stores boxed *builder* trait objects and is
extended by registering more builders before a run starts.

| Registry | Builder trait | Selection model | Built-in roster (`Builtins::builtins`) |
| --- | --- | --- | --- |
| Integrators | `IntegratorBuilder` | named | `src/integrator/mod.rs` |
| Thermostats | `ThermostatBuilder` | named | `src/integrator/mod.rs` |
| Barostats | `BarostatBuilder` | named | `src/integrator/mod.rs` |
| Constraint types | `ConstraintBuilder` | named | `src/integrator/constraint.rs` |
| Minimizers | `MinimizerBuilder` | named | `src/minimizer/mod.rs` |
| Analyses | `AnalysisBuilder` | named | `src/analysis/mod.rs` |
| Potentials | `PotentialBuilder` | compositional | `src/forces/mod.rs` |

All seven share one generic container, `Registry<B>`
(`src/registry.rs`); the framework is specified in
`rqm/registry-framework.md`.

### Two selection models

- **Named selection** (six registries). The config names a `kind`
  string; the registry looks up the single builder whose `kind_name()`
  matches and builds from it. This is the model for choices that are
  mutually exclusive — one integrator per phase, one thermostat, one
  barostat. Named-selection builder traits carry the `KindedBuilder`
  supertrait (which supplies `kind_name()` and unit conversion).

- **Compositional activation** (the potential registry only). No `kind`
  selects a single builder; instead *every* registered builder is
  consulted and each returns `Some(slot)` when the configuration data it
  consumes is present. A force field is the sum of all the potentials
  that activated. `PotentialBuilder` deliberately does **not** carry
  `KindedBuilder`, so `PotentialRegistry` has no `lookup` — the type
  system enforces the distinction between the two models.

## Anatomy of a builder

Every builder is a small, stateless, cloneable factory. The pattern is
the same across all seven registries:

1. **A unit (zero-field) struct** — e.g. `struct CsvrBuilder;` — deriving
   `Clone` and `Debug`. Boxed-trait-object cloning is generated for you
   by the `registry_builder_clone!` macro (invoked once per builder trait
   in that subsystem's `mod.rs`), so you never write a `box_clone`
   method — `#[derive(Clone)]` is enough.

2. **The `KindedBuilder` impl** (named-selection registries only) —
   supplies `kind_name()` (the TOML `kind` lookup key) and
   `convert_params()` (unit conversion; see below).

3. **The builder-trait impl** — `validate_params()` (config-load-time
   domain checks), `build()` (construct the boxed slot on the GPU), and
   any capability predicates the subsystem defines (for integrators:
   `owns_thermostat`, `owns_barostat`, `supports_constraints`,
   `graph_compatible`).

4. **The slot type** — the actual `Thermostat` / `Integrator` /
   `Barostat` / `Potential` object the builder constructs, holding the
   device buffers and per-step kernel launches.

### Registering the builder

Two ways, and you usually want the first:

- **A built-in** — add `Box::new(YourBuilder)` to the relevant
  `Builtins::builtins()` roster (the table above says which file). For a
  potential, the roster's **order is the slot evaluation order**, so
  insert it at the right position. This is the only place the built-in
  set is defined; there is no parallel enum to update.

- **An out-of-tree extension** — an embedder building on the library
  constructs `Registries` and calls `register_integrator`,
  `register_thermostat`, `register_potential`, … before
  `run_simulation_with_registries`. Registration appends, and a later
  registration never shadows an earlier builder of the same kind, so a
  custom builder cannot override a built-in of the same `kind` — pick a
  fresh name.

Both paths flow through the same construction dispatch, so a registered
custom builder behaves exactly like a built-in.

## Configuration plumbing

The config layer is open-shaped and registry-driven — this is why adding
a kind needs no central config edit.

- A slot section (`[phase.integrator]`, `[phase.thermostat]`,
  `[phase.barostat]`) parses to a `SlotConfig { kind: String, params:
  toml::Value }` (`src/io/config.rs`). `params` holds every field of the
  section except `kind`, untyped. The engine core never inspects
  `params`; the builder owns its schema.

- At load time the loader looks the builder up by `kind` and calls its
  `validate_params(&params)`. An unknown `kind` surfaces as
  `IntegratorError::UnknownKind` / `ConfigError::UnknownKind` — you get
  that for free the moment your builder is (not) in the roster.

- Each builder deserialises its own **typed params struct** from
  `params` inside `validate_params` and again in `build`. Validation
  guarantees the deserialise in `build` succeeds, so a failure there is
  an internal error, not a user error.

Potentials are activated compositionally rather than selected, so their
per-entry parameters (in `[[pair_interactions]]`, `[[bond_types]]`, …)
are routed by a **params claim** — a `(category, kind)` pair the builder
returns from `params_claim()` — instead of a registry lookup. The
mechanics are the same (`validate_params`, `convert_params` per claimed
entry); see the potential pages and `rqm/forces/framework.md`.

### Unit conversion

Config values arrive in the user's unit system (SI or Hartree atomic)
and must be converted to atomic units before `build`. You do **not**
write conversion code by hand. Make each unit-bearing params field a
**dimension newtype** from `crate::units` (`Temperature`, `Time`,
`Energy`, `Length`, `Pressure`, …) and derive `Convert`:

```rust
#[derive(Debug, Clone, Deserialize, serde::Serialize, crate::units::Convert)]
pub struct CsvrParams {
    pub temperature: crate::units::Temperature, // converted
    pub tau: crate::units::Time,                // converted
    pub seed: u64,                              // dimensionless, untouched
}
```

The `Convert` derive (in the `heddle-md-derive` crate) recurses through
each field, calling `Convert::from_user` on the dimension newtypes and
leaving plain scalars alone. Your `convert_params` implementation is then
the one-liner `convert_params_in_place::<YourParams>(units, params)`
(`src/registry.rs`). See `rqm/io/unit-system.md`.

## CUDA kernel wiring

Most new features need one or more device kernels. The build is fully
declarative — there is no per-kernel entry in a makefile.

1. **Write the `.cu` file** in `kernels/`. `build.rs` discovers every
   `kernels/*.cu` automatically, compiles it to PTX with `nvcc`, and
   generates a `&str` PTX constant named after the file stem in
   upper-case (e.g. `kernels/andersen.cu` → `crate::kernels::ANDERSEN`).
   Use the precision shim in `kernels/precision.cuh` (`Real`, etc.) so
   the same source compiles for both the `f32` and future `f64` builds.

2. **Declare a subsystem kernel handle set** with the `gpu_kernels!`
   macro in your subsystem's Rust file. It lists the kernel entry-point
   names and their timing stages:

   ```rust
   crate::gpu_kernels! {
       module: "andersen",
       ptx: crate::kernels::ANDERSEN,
       struct: AndersenKernels,
       kernels: [andersen_resample],
       stages: { ANDERSEN_RESAMPLE = "andersen_resample" },
   }
   ```

3. **Register the handle set** by adding a field to the `define_kernels!`
   manifest in `src/gpu/device.rs`. That manifest is the single source of
   truth for both the aggregate `Kernels` struct and the timings-stage
   order, so the two can never drift.

4. **Write a launch wrapper** in `src/gpu/kernels.rs` using the
   `gpu_launch!` macro, which supplies the empty-`N` guard, launch config,
   and error mapping. Kernel timing is bracketed **externally** (by the
   runner / CUDA-graph capture) — a launch wrapper must contain no timing
   calls (there is a test that enforces this).

A purely host-side feature (rare) can skip all of this. If your feature
reuses kernels that already exist (e.g. an integrator that only needs the
existing kick/drift kernels), you skip steps 1–3 and just launch them.

## Determinism: the rule you cannot break

Bit-wise reproducibility on a fixed GPU is the engine's load-bearing
guarantee (`docs/architecture.md`). Any new kernel must preserve it:

- **No order-dependent float accumulation.** Never `atomicAdd` into a
  float that more than one thread contributes to. Per-particle force sums
  use either a fixed-topology reduction or the fixed-point (integer)
  accumulators, whose integer adds are associative and therefore
  order-independent (see `rqm/forces/packed-neighbour-pair-force.md`).
- **Fixed summation order.** Any reduction a slot performs (a per-bond
  sum, a kinetic-energy reduction) must sum in an order fixed by data
  (particle/bond index), not by thread arrival.
- **Seeded, counter-based RNG.** Stochastic features draw noise from the
  counter-based `philox` generator (`src/integrator/philox.rs`) keyed by
  an explicit config `seed` and a per-slot step counter, never from
  wall-clock or thread-id state.

Two runs of the same config on the same GPU must produce byte-identical
trajectory and log files. When in doubt, add a reproducibility test (run
twice, `diff` the outputs).

## Specs and tests

- **Requirements specs** live under `rqm/`, one file per feature
  (`rqm/integration/csvr.md`, `rqm/forces/lj-pair-force.md`, …). Each is
  tagged with `rq-` identifiers and carries Gherkin scenarios. Add a spec
  file for a substantial new feature and follow the neighbouring one's
  shape; the per-feature pages point at the closest template.
- **Tests** live in `tests/` (integration / end-to-end) and in
  `#[cfg(test)]` modules next to the code (unit). The registries have
  **lint tests** that assert invariants across every built-in (for
  example, that every integrator accepting a per-step barostat emits a
  `BarostatPoint`); if you add a built-in, make sure it satisfies them.

Each per-feature page ends with the concrete manifest — the files to
create and the exact existing lines to edit — for that extension point.
