# Adding an integrator

An integrator defines the ordered velocity updates, position updates, and
force evaluations that make up one timestep. The user selects it by name with
`[phase.integrator] kind = "..."`.

Read the [extension overview](index.md) first. This guide assumes you
understand its descriptions of builders, runtime components, host and device
work, and CUDA-graph compatibility.

Unlike a conventional integrator routine, a HeddleMD integrator does not run
the whole timestep itself. It describes the timestep by returning a
`StepPlan`, which is an ordered list of typed `SubStep` values. The shared
runner executes that list. This allows the runner to insert configured
thermostat, barostat, and constraint operations, validate data dependencies,
and capture compatible device work in a CUDA graph.

## Choose the closest example

- `src/integrator/velocity_verlet.rs` — the canonical symplectic template.
  Start here for an integrator with ordinary kick and drift operations.
- `src/integrator/langevin_baoab.rs` — owns its own thermostat; a `Custom`
  sub-step (the OU step) with a device-resident RNG counter.
- `src/integrator/mtk_npt.rs` — owns thermostat and barostat; reads virial
  every step (`ForcesAndScalars`); `graph_compatible = false`; log columns.
- `src/integrator/respa.rs` — multiple force evaluations per step and
  `KickSource::Class` impulse splitting.

Before writing Rust, express the algorithm as an ordered list of operations
and identify which forces each velocity update consumes. This makes it much
easier to construct a valid `StepPlan`.

## Describe the timestep with a StepPlan

`plan(dt) -> StepPlan` must be a **pure function** of `dt` and the
integrator's fixed configuration. In this context, pure means that it does not
modify the integrator, launch kernels, allocate buffers, or depend on a
per-step counter. It must return the same sequence of operation types every
time. The runner relies on that stable sequence when validating the phase and
recording CUDA graphs.

The canonical symplectic (velocity-Verlet family) shape reads `F(t)`
before the force evaluation and `F(t+dt)` after it:

```
ConstraintPoint { BeforeDrift }      // snapshot; no-op without a constraint slot
KickDrift { Total }                  // v += (F(t)/m)·dt/2 ; x += v·dt   (fused)
ConstraintPoint { AfterDrift }       // project positions
ForceEval { class: None, level }     // runner recomputes F(t+dt)
KickHalf  { Total }                  // v += (F(t+dt)/m)·dt/2
BarostatPoint                        // terminal; no-op without a per-step barostat
ConstraintPoint { AfterKick }        // project velocities — RATTLE-last
```

The following rules explain how that sequence is executed:

- **`execute()` never sees `ForceEval`.** Only the runner holds the
  `ForceField`, so it dispatches force evaluation itself. Your `execute`
  handles the kicks, drifts, and any integrator-private `Custom` steps, and
  returns `IntegratorError::UnexpectedSubStep` for anything it shouldn't
  receive.
- **Forces at the start of a step are already available.** At step entry the runner treats the cached forces as
  valid `F(t)` (left by the previous step's trailing `ForceEval`, or the
  runner's one warm-up evaluation before the loop). So the leading
  `KickDrift` legitimately reads `F(t)` — **do not prepend a `ForceEval` to
  "prime" forces**; that would double-evaluate, and the schedule validator
  already assumes carried-in forces are valid.
- **Placement markers are harmless when no matching component is configured.** Emit `ConstraintPoint`,
  `BarostatPoint`, and (if you own thermostat placement) `ThermostatHalf`
  markers unconditionally, so one plan drives constrained and unconstrained,
  NVE and NPT runs alike. Order matters: a terminal `BarostatPoint` goes
  **before** the final `AfterKick` velocity projection (RATTLE-last).
- **`AggregateLevel` controls which force-derived quantities are calculated.** On the `ForceEval`, it selects the cheap `ForcesOnly` path
  unless the integrator itself needs energy/virial every step (NPT
  integrators emit `ForcesAndScalars`). Use `None` to defer to the runner,
  which upgrades to `ForcesAndScalars` on steps that log or write a frame.
- **`KickSource` identifies the force buffer used by a kick.** `Total` reads the combined force
  buffer and is executed by the integrator. `Class(Fast|Slow)` reads one
  class accumulator and is dispatched by the runner — use it only for
  multiple-timestep (RESPA) impulse splitting.

The schedule is validated at phase setup: `StepPlan::validate()` rejects a
plan that reads force-derived state a preceding position/box mutation
invalidated without an intervening `ForceEval`. Each built-in `SubStep` has
a fixed resource footprint; a `Custom` step declares its own `reads`/`writes`
(see `src/integrator/op_model.rs` and `rqm/integration/op-model.md`). Model a
`Custom` step's label-and-footprint as a single enum, like MTK's `MtkStep`,
so `execute` re-derives the same footprint it declared.

## Declare compatibility with other components

The builder provides several methods that tell the configuration loader which
combinations are valid. These methods receive the parsed parameters, so an
answer may depend on the selected mode:

| Predicate | Default | If it returns the non-default, the loader/runner… |
| --- | --- | --- |
| `owns_thermostat` | `false` | rejects a co-configured `[thermostat]` (`IncompatibleThermostat`) |
| `owns_barostat` | `false` | rejects a co-configured `[barostat]` (`IncompatibleBarostat`) |
| `supports_barostat` | `true` | (if `false`) rejects any `[barostat]` — distinct from owning one |
| `supports_constraints` | `false` | (if `false`) rejects a non-empty topology `[constraints]` |
| `graph_compatible` | `true` | (if `false`) forces the phase onto the eager per-step path |

`velocity-verlet` computes `supports_constraints` as `!lossless` (the
lossless mode can't project). Set `graph_compatible = false` if `execute`
does host-side scalar arithmetic or a device→host copy between launches
(MTK does). A separate runtime guard, `BarostatPlacementMissing`, fires if a
per-step barostat is configured with a plan that carries no `BarostatPoint`
— so a barostat-hosting integrator must emit one.

If the integrator supports constraints, also implement
`ConstraintCapableIntegrator`. The builder predicate says that the
configuration is supported; this runtime trait lets the constraint framework
exercise the constructed integrator and reject a state-dependent case.

## Reuse device kernels and preserve determinism

Reuse the existing `integrate.cu` kernels where you can — `vv_kick`,
`vv_kick_drift`, and the class-kick kernels are re-exported from
`crate::gpu`, so a standard symplectic integrator needs no new CUDA. If you
add one, follow the [overview](index.md) CUDA section. Keep RNG state in a
**device-resident counter** (like Langevin's `draw_counter_device`) so it
survives CUDA-graph replay deterministically; a host counter would freeze
under capture and break reproducibility.

## Implement the Rust types

The main implementation belongs in a new file such as
`src/integrator/my_integrator.rs`. It contains the parameter type, runtime
state, `Integrator` implementation, and builder. The following skeleton is an
abbreviated outline: comments replace method-specific calculations and error
mapping, so it is not directly compilable.

```rust
use serde::Deserialize;
use crate::gpu::{GpuContext, ParticleBuffers, vv_kick, vv_kick_drift};
use crate::io::config::ConfigError;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};
use crate::precision::Real;
use super::{Integrator, IntegratorBuilder, IntegratorError,
            KickSource, StepPlan, SubStep, ConstraintPhase};

#[derive(Debug, Clone, Deserialize)]         // add serde::Serialize + Convert
#[serde(deny_unknown_fields)]                // if any field is unit-bearing
pub struct MyIntegratorParams {
    #[serde(default)] pub lossless: bool,
    // pub tau: crate::units::Time,          // unit-bearing → dimension newtype
}

#[derive(Debug)]
pub struct MyIntegratorState { /* device scratch, RNG counters, … */ }

impl Integrator for MyIntegratorState {
    fn plan(&self, dt: Real) -> StepPlan {           // MUST be pure
        StepPlan { steps: vec![
            SubStep::ConstraintPoint { phase: ConstraintPhase::BeforeDrift, dt },
            SubStep::KickDrift { dt, label: "my_kick_drift", source: KickSource::Total },
            SubStep::ConstraintPoint { phase: ConstraintPhase::AfterDrift, dt },
            SubStep::ForceEval { class: None,
                                 level: Some(crate::forces::AggregateLevel::ForcesOnly) },
            SubStep::KickHalf { dt, label: "my_kick", source: KickSource::Total },
            SubStep::BarostatPoint { dt },           // terminal, before AfterKick
            SubStep::ConstraintPoint { phase: ConstraintPhase::AfterKick, dt },
        ]}
    }
    fn execute(&mut self, substep: &SubStep, buffers: &mut ParticleBuffers,
               sim_box: &mut SimulationBox, timings: &mut Timings)
        -> Result<(), IntegratorError> {
        if buffers.particle_count() == 0 { return Ok(()); }        // mandatory
        match substep {
            SubStep::KickDrift { dt, .. } => { /* time + vv_kick_drift(buffers, sim_box, *dt) */ Ok(()) }
            SubStep::KickHalf  { dt, .. } => { /* time + vv_kick(buffers, *dt) */ Ok(()) }
            other => Err(IntegratorError::UnexpectedSubStep { variant: other.variant_name() }),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MyIntegratorBuilder;

impl crate::registry::KindedBuilder for MyIntegratorBuilder {
    fn kind_name(&self) -> &'static str { "my-integrator" }
    // convert_params via convert_params_in_place::<MyIntegratorParams> — unit-bearing only
}

impl IntegratorBuilder for MyIntegratorBuilder {
    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> { /* … */ Ok(()) }
    // override owns_thermostat / owns_barostat / supports_barostat /
    // supports_constraints / graph_compatible where they differ from defaults
    fn build(&self, gpu: &GpuContext, particle_count: usize, n_constraints: usize,
             params: &toml::Value) -> Result<Box<dyn Integrator>, IntegratorError> {
        Ok(Box::new(MyIntegratorState { /* … */ }))
    }
}
```

## Register and document the integrator

Add `rqm/integration/my-integrator.md` for the requirements specification; follow
`rqm/integration/velocity-verlet.md` and the op-model conventions in
`rqm/integration/op-model.md`.

Then edit the following existing files:

- **`src/integrator/mod.rs`** — declare `pub mod my_integrator;`, add the
  `pub use my_integrator::{MyIntegratorBuilder, MyIntegratorState};`
  re-export, and `Box::new(MyIntegratorBuilder)` in the `vec![...]` of
  `impl Builtins for dyn IntegratorBuilder` (roster at ~line 958).

- **`src/gpu/device.rs`** — only if you added a `.cu`: one field in the
  `define_kernels!` manifest.

**No change needed** to `src/io/config.rs` (compatibility checks call your
builder's predicates; kind dispatch is registry-driven) or `build.rs`.

## Test the integrator

There is currently no registry-wide physical conformance harness for
integrators analogous to the potential-consistency or thermostat/barostat
slot-conformance harnesses. Integrators instead share schedule validation and
framework tests. Add pure, GPU-free plan assertions to
`tests/integrator_framework.rs` when the assertion is an invariant that every
built-in integrator should satisfy; keep method-specific plan and numerical
tests in the new integrator's test file.

The [testing chapter](../developer/testing.md) explains how shared-contract
tests complement focused analytical and end-to-end tests. For an integrator,
the end-to-end portion is particularly important because it exercises carried
forces and composition with thermostat, barostat, and constraint markers.

`tests/integrators_my_integrator.rs` (model on `tests/integrator_framework.rs`
and `tests/integrators_respa.rs`):

- Registry roster / construction succeeds for the new kind, including via
  `Registries::with_builtins()`.
- Build the plan and assert `StepPlan::validate()` is `Ok` (pure, no GPU);
  assert `has_barostat_points()` if applicable.
- The predicates return their intended values, and co-configuring an
  incompatible slot yields the matching `ConfigError`.
- Empty-state (`particle_count == 0`) construction and step are no-ops.
- If NVE-like, an energy-drift / determinism end-to-end test.

## Completion checklist

- [ ] **`plan()` is pure** — no mutation of `self`, no branching on a
  per-step counter. The runner relies on a stable shape for graph capture.
- [ ] **The warm-up force evaluation remains the runner's job**, before the loop —
  don't prime forces in your plan.
- [ ] **`execute` handles `particle_count == 0`**; construction with zero
  particles must still succeed.
- [ ] **`graph_compatible` reflects the execution path** — any host-side scalar read
  between launches means `false`, or graph replay freezes stale values.
- [ ] **The `kind` name is unique** — a later same-name registration never shadows a built-in, so pick a
  fresh name.
- [ ] **Physical parameters use dimension types**, derive `Convert`, and
  implement `convert_params`; otherwise SI inputs may be interpreted as
  atomic units.
