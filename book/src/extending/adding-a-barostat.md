# Adding a barostat

A barostat couples a system to a pressure bath by changing the simulation box
and the particle or molecular positions within it. The user selects a
separate barostat with `[phase.barostat] kind = "..."`; it can be combined
with velocity Verlet and a thermostat to perform NPT dynamics.

Read the [extension overview](index.md) first for the shared builder,
configuration, unit-conversion, and GPU terminology.

## Choose the execution model

HeddleMD supports two kinds of barostat operation. Decide which model matches
the physical algorithm before implementing the trait:

- **Per-step** (`periodicity() == EveryStep`, the default) — override
  `apply`, which runs every step at a `BarostatPoint` in the plan. It reads
  the current virial and kinetic energy, computes a scale factor, mutates
  the box, and rescales positions. Use `berendsen_barostat.rs` as the simpler
  equilibration-only example or `c_rescale_barostat.rs` as a canonical,
  stochastic example.
- **Periodic** (`periodicity() == EveryNSteps(f)`) — override `apply_move`,
  a host-orchestrated Monte-Carlo-style move run every `f` steps at a batch
  boundary. It receives mutable access to the force field because it must
  evaluate the energy of a trial box. Use `mc_barostat.rs` as the example.

Implement `apply` for a per-step barostat or `apply_move` for a periodic
barostat. Do not put both algorithms into one implementation.

## Per-step barostats

The runner calls `apply(&mut self, buffers, sim_box, dt, timings)` at the
barostat point in every timestep. A typical implementation performs these
operations:

1. Reduce the kinetic energy and the total virial into your own length-1
   device scratch buffers (`compute_kinetic_energy_on_device`,
   `compute_total_virial_on_device` — both re-exported from `crate::gpu`).
   You do not read `buffers.virials` directly; the virial reduction sums the
   per-particle scalar-virial shares the force pipeline wrote.
2. Compute the scale factor `µ` and mutate the box lattice — in the existing
   barostats this is one scalar kernel that also writes diagnostics.
3. Rescale positions with the device-resident factor
   (`rescale_positions_device_factor`).

### How the barostat receives a current virial

The per-particle virial shares are
only current if the step's `ForceEval` ran at `AggregateLevel::ForcesAndScalars`.
You get this without the integrator having to know: when a per-step barostat
is active, the runner sets `runner_needs_scalars`, which upgrades every
`ForceEval` to `ForcesAndScalars` and disables the forces-only CUDA graph
for the phase. So c-rescale/Berendsen work even with plain velocity-Verlet
(which otherwise emits `ForcesOnly`).

### Where the barostat appears in a timestep

`apply` runs only at a `SubStep::BarostatPoint` marker.
`velocity-verlet` emits a terminal one, placed *before* the final
`AfterKick` velocity projection so a RATTLE projection stays last. If a
per-step barostat is configured with an integrator whose plan carries no
`BarostatPoint`, phase setup fails with `BarostatPlacementMissing`.

### Notify dependent calculations when the box changes

Mutating the lattice through `sim_box.lattice_device_mut()`
bumps the box generation counter. The neighbor-list pipeline observes that
counter and rebuilds on the next step. SPME then observes the neighbor-list
rebuild generation and re-sorts its cached particle order. You get that for
free.

## Periodic barostats

The runner calls
`apply_move(&mut self, force_field, buffers, sim_box, constraint, dt, timings)`
at the requested host interval. A Monte Carlo volume move normally follows
this sequence:

1. Read the current potential energy `U_old` (the runner guarantees the
   pre-move step evaluated `ForcesAndScalars`).
2. Snapshot positions, forces, and the lattice (device-to-device copies).
3. Draw the trial from Philox (host counter is fine here — the move is not
   captured in a CUDA graph), propose a volume change, guard the new box
   width against the cutoff, scale the box and the molecular centres of mass
   (`mc_barostat_scale_molecule_com`), and re-evaluate the energy with
   `force_field.step(..., ForcesAndScalars)` to get `U_new`.
4. Metropolis accept/reject; on reject, restore the snapshots.

Override `init_run` to upload the per-molecule tables once the molecule list
is known. Because the move scales whole molecules' centres of mass rather
than individual atoms, it preserves rigid geometry and composes with
constrained water. A periodic barostat needs **no** `BarostatPoint` marker
and does no per-step work, so it stays `graph_compatible` (its move runs
between graph replays).

> Current limitation: the runner currently passes `None` for the `constraint`
> argument to `apply_move`. If your move needs to reproject constraints after
> scaling, that wiring does not exist yet — treat it as a limitation.

## Preserve determinism in stochastic methods

For a stochastic barostat, draw from Philox keyed by a config `seed`. A
per-step barostat must keep its counter **on device** (the c-rescale pattern:
the `µ` kernel reads and increments a length-1 `draw_counter_device` buffer)
so it survives CUDA-graph replay. A periodic barostat may use a host counter
since its move isn't captured. Same `(seed, counter)` → same draw → identical
box trajectory.

## Implement the Rust types

Create `src/integrator/my_barostat.rs` containing the typed parameters,
runtime state, `Barostat` implementation, and builder. The following per-step
skeleton is abbreviated and is not directly compilable; the comments stand in
for the method-specific equations, launches, and error mapping.

```rust
use serde::Deserialize;
use crate::gpu::{GpuContext, GpuError, ParticleBuffers, c_rescale_compute_mu,
                 compute_kinetic_energy_on_device, compute_total_virial_on_device,
                 rescale_positions_device_factor};
use crate::io::config::ConfigError;
use crate::pbc::SimulationBox;
use crate::registry::{KindedBuilder, convert_params_in_place};
use crate::timings::{KernelStage, Timings};
use crate::precision::Real;
use super::{Barostat, BarostatBuilder, BarostatError};

#[derive(Debug, Clone, Deserialize, serde::Serialize, crate::units::Convert)]
#[serde(deny_unknown_fields)]
pub struct MyBarostatParams {
    pub pressure: crate::units::Pressure,
    pub tau: crate::units::Time,
    pub compressibility: crate::units::InversePressure,
    pub temperature: crate::units::Temperature, // stochastic only
    pub seed: u64,                              // stochastic only
}

#[derive(Debug)]
pub struct MyBarostat { /* params + length-1 device scratch (ke, virial, mu,
                           diagnostics, draw_counter) allocated in new() */ }

impl Barostat for MyBarostat {
    // periodicity() defaults to EveryStep — do NOT override for per-step.
    fn apply(&mut self, buffers: &mut ParticleBuffers, sim_box: &mut SimulationBox,
             dt: Real, timings: &mut Timings) -> Result<(), BarostatError> {
        if buffers.particle_count() == 0 { return Ok(()); }        // mandatory
        // reduce KE and virial → compute µ + mutate lattice + diagnostics
        //  → rescale_positions_device_factor(buffers, &self.mu_device)
        Ok(())
    }
    fn flush_pending_injection(&mut self, device: &std::sync::Arc<cudarc::driver::CudaDevice>)
        -> Result<(), BarostatError> { /* drain the injection accumulator for the log */ Ok(()) }
    fn log_column_names(&self) -> &'static [(&'static str, crate::units::Dimension)] {
        use crate::units::Dimension;
        &[("pressure", Dimension::Pressure), ("box_volume", Dimension::Dimensionless)]
    }
    fn log_column_values(&self, _ke: f64, _pe: f64) -> Vec<f64> { vec![/* pressure, volume */] }
}

#[derive(Debug, Clone)]
pub struct MyBarostatBuilder;
impl KindedBuilder for MyBarostatBuilder {
    fn kind_name(&self) -> &'static str { "my-barostat" }
    fn convert_params(&self, units: crate::units::UnitSystem, params: &mut toml::Value)
        -> Result<(), ConfigError> { convert_params_in_place::<MyBarostatParams>(units, params) }
}
impl BarostatBuilder for MyBarostatBuilder {
    fn validate_params(&self, params: &toml::Value) -> Result<(), ConfigError> { /* … */ Ok(()) }
    // graph_compatible defaults to true — keep it for a pure-launch per-step barostat.
    fn build(&self, gpu: &GpuContext, particle_count: usize, _n_constraints: usize,
             params: &toml::Value) -> Result<Box<dyn Barostat>, BarostatError> {
        Ok(Box::new(/* MyBarostat::new(...)? */))
    }
}
```

For a **periodic** barostat, instead override `periodicity()` →
`BarostatPeriodicity::EveryNSteps(freq)`, `init_run`, and `apply_move`
(taking `&mut ForceField`); leave `apply` at the no-op default. See
`mc_barostat.rs`.

## Register and document the barostat

Add `rqm/integration/my-barostat.md`; follow
`rqm/integration/c-rescale-barostat.md` (per-step) or
`rqm/integration/mc-barostat.md` (periodic), and add a row to the barostat
table in `rqm/integration/framework.md`.

Then edit the following existing files:

- **`src/integrator/mod.rs`** — declare `pub mod my_barostat;`, add the
  `pub use my_barostat::{MyBarostat, MyBarostatBuilder};` re-export, and
  `Box::new(MyBarostatBuilder)` in the `vec![...]` of
  `impl Builtins for dyn BarostatBuilder` (roster at ~line 1230). This is the
  only registration point — dispatch is fully generic over `dyn Barostat`.

- **`src/gpu/device.rs`** — only if you add a new `.cu`: one field in the
  `define_kernels!` manifest.

**No change needed** to `src/io/config.rs` (validate/convert are called by
registry lookup) or `src/runner.rs` (barostat dispatch is generic).

## Test the barostat

Use both focused component tests and the shared slot-conformance harness. For
a built-in barostat, add a `SlotCase` to `builtin_barostat_cases()` in
`tests/common/slot_conformance.rs`. The case names the registered `kind`, uses
`Expect::HoldsDensity` with scientifically justified bounds, and records why
those bounds are appropriate for the finite constrained-water system.

Add a named test in `tests/slot_conformance.rs` that passes the case to
`run_barostat_case`, and update the barostat sweep count in that file. Registry
coverage fails if a built-in barostat has no case, while the sweep-count test
fails if the case is not driven by a named test. The [testing
chapter](../developer/testing.md) explains the dense SPC/E system, negative
controls, and the distinction between conformance bounds and bulk experimental
accuracy.

`tests/barostats_my.rs` (model on `tests/barostats_c_rescale.rs` /
`tests/barostats_mc.rs`):

- Registry exposes the kind (`BarostatRegistry::with_builtins().builders()`
  scan for `kind_name()`).
- `validate_params` accepts/rejects the expected params (`deny_unknown_fields`
  rejects extras).
- `apply` launches exactly its expected kernel set and none of the
  integrator kernels; `µ` matches the analytical formula for a known
  `(K, W)`.
- Empty-state (`particle_count == 0`) `apply` is a no-op and does **not** bump
  the box generation.
- Determinism: same seed → identical box trajectory across builds.
- For periodic: the draw counter increments per attempted move, velocities
  are untouched by a move, and rigid-molecule geometry is invariant under a
  scale.

## Completion checklist

- [ ] **The execution model and method agree** — `apply` (per-step)
  *or* `apply_move` (periodic), consistent with `periodicity()`.
- [ ] **A per-step barostat has a `BarostatPoint` in the integrator plan**, or
  setup fails with `BarostatPlacementMissing`. `velocity-verlet` provides one.
- [ ] **`apply` or `apply_move` handles `particle_count == 0`** and does not mutate the
  box in that case.
- [ ] **The specification identifies the ensemble behavior.** A canonical
  barostat should carry conserved-quantity bookkeeping in its log column;
  Berendsen is explicitly equilibration-only.
- [ ] **A per-step stochastic barostat keeps its RNG counter on the device** so graph replay
  stays deterministic; a host counter is fine for a periodic move.
- [ ] **The `kind` name is unique.** A later same-name registration never shadows a built-in.
- [ ] **A built-in barostat has a `SlotCase` and named conformance test.**
