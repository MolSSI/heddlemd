# Adding an integrator

An integrator is the core time-stepping algorithm — the velocity kicks,
position drifts, and in-step force evaluation that make one timestep. It
is a **named-selection** registry slot (`[phase.integrator] kind = "..."`).
Read the [overview](index.md) first.

The distinctive thing about an integrator is that it does **not** run a
timestep imperatively. It returns a `StepPlan` — an ordered list of typed
`SubStep`s — and the runner's single `run_step` executor walks it. This
lets the runner insert thermostat, barostat, and constraint hooks at the
points the plan declares, validate the schedule's data dependencies, and
capture it into a CUDA graph.

Templates:

- `src/integrator/velocity_verlet.rs` — the canonical symplectic template.
  Start here.
- `src/integrator/langevin_baoab.rs` — owns its own thermostat; a `Custom`
  sub-step (the OU step) with a device-resident RNG counter.
- `src/integrator/mtk_npt.rs` — owns thermostat and barostat; reads virial
  every step (`ForcesAndScalars`); `graph_compatible = false`; log columns.
- `src/integrator/respa.rs` — multiple force evaluations per step and
  `KickSource::Class` impulse splitting.

## The StepPlan contract

`plan(dt) -> StepPlan` must be a **pure function** of `dt` and the
integrator's static configuration — the same shape every call. The runner
probes it once per phase (to pick the thermostat topology and CUDA-graph
eligibility) and calls it once per step to walk. It launches no kernels and
allocates no buffers.

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

Load-bearing details:

- **`execute()` never sees `ForceEval`.** Only the runner holds the
  `ForceField`, so it dispatches force evaluation itself. Your `execute`
  handles the kicks, drifts, and any integrator-private `Custom` steps, and
  returns `IntegratorError::UnexpectedSubStep` for anything it shouldn't
  receive.
- **The F(t) cache.** At step entry the runner treats the cached forces as
  valid `F(t)` (left by the previous step's trailing `ForceEval`, or the
  runner's one warm-up evaluation before the loop). So the leading
  `KickDrift` legitimately reads `F(t)` — **do not prepend a `ForceEval` to
  "prime" forces**; that would double-evaluate, and the schedule validator
  already assumes carried-in forces are valid.
- **Markers are no-ops without their slot.** Emit `ConstraintPoint`,
  `BarostatPoint`, and (if you own thermostat placement) `ThermostatHalf`
  markers unconditionally, so one plan drives constrained and unconstrained,
  NVE and NPT runs alike. Order matters: a terminal `BarostatPoint` goes
  **before** the final `AfterKick` velocity projection (RATTLE-last).
- **`AggregateLevel`** on the `ForceEval` picks the cheap `ForcesOnly` path
  unless the integrator itself needs energy/virial every step (NPT
  integrators emit `ForcesAndScalars`). Use `None` to defer to the runner,
  which upgrades to `ForcesAndScalars` on steps that log or write a frame.
- **`KickSource::Total` vs `Class`.** `Total` reads the combined force
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

## Compatibility predicates

The builder answers predicates the config loader and runner consult, each
taking the parsed params so the answer can depend on configuration:

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

To let tests drive your integrator with a constraint slot, also implement
the `ConstraintCapableIntegrator` trait (orthogonal to the
`supports_constraints` predicate); its `check_accepts_constraints_now` is a
runtime veto.

## Determinism and CUDA

Reuse the existing `integrate.cu` kernels where you can — `vv_kick`,
`vv_kick_drift`, and the class-kick kernels are re-exported from
`crate::gpu`, so a standard symplectic integrator needs no new CUDA. If you
add one, follow the [overview](index.md) CUDA section. Keep RNG state in a
**device-resident counter** (like Langevin's `draw_counter_device`) so it
survives CUDA-graph replay deterministically; a host counter would freeze
under capture and break reproducibility.

## Manifest

### New files

**`src/integrator/my_integrator.rs`** — the whole integrator: params
struct, state + `Integrator` impl (`plan`/`execute`, optional log columns),
and the builder. Skeleton (canonical symplectic, reusing `integrate.cu`):

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

**`rqm/integration/my-integrator.md`** — the spec; follow
`rqm/integration/velocity-verlet.md` and the op-model conventions in
`rqm/integration/op-model.md`.

### Existing files to edit

- **`src/integrator/mod.rs`** — three lines: `pub mod my_integrator;`, the
  `pub use my_integrator::{MyIntegratorBuilder, MyIntegratorState};`
  re-export, and `Box::new(MyIntegratorBuilder)` in the `vec![...]` of
  `impl Builtins for dyn IntegratorBuilder` (roster at ~line 958).

- **`src/gpu/device.rs`** — only if you added a `.cu`: one field in the
  `define_kernels!` manifest.

**No change needed** to `src/io/config.rs` (compatibility checks call your
builder's predicates; kind dispatch is registry-driven) or `build.rs`.

### Tests

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

## Gotchas

- **`plan()` must be pure** — no mutation of `self`, no branching on a
  per-step counter. The runner relies on a stable shape for graph capture.
- **The warm-up force evaluation is the runner's job**, before the loop —
  don't prime forces in your plan.
- **Guard `execute` on `particle_count == 0`**; construction with zero
  particles must still succeed.
- **`graph_compatible` must be truthful** — any host-side scalar read
  between launches means `false`, or graph replay freezes stale values.
- **A later same-`kind` registration never shadows a built-in** — pick a
  fresh name.
- **If params carry physical units**, use `crate::units` dimension newtypes
  + derive `Convert` + implement `convert_params`, or SI inputs are silently
  read as atomic units.
