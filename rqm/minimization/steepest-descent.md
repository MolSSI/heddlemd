# Feature: Steepest-Descent Energy Minimizer <!-- rq-08aba7ee -->

Energy minimization is a phase-kind of the simulation runner alongside
time-integration MD. A minimization phase advances particle positions
along the negative gradient of the potential energy until either a
force-per-atom threshold or an energy-change threshold is met, or a
maximum iteration count is reached. Velocities and the simulation box
pass through the phase unchanged; only positions move.

The runner exposes minimization through a `Minimizer` slot trait that
parallels the `Integrator` slot trait. The `MinimizerRegistry` carries
one registered algorithm in v1, `kind = "steepest-descent"`, and is
structured so that future conjugate-gradient or L-BFGS variants register
alongside without changing the runner's dispatch surface.

The default registry exposes one minimizer:

| `kind` value         | Implementation                                                | File                       |
| -------------------- | ------------------------------------------------------------- | -------------------------- |
| `steepest-descent`   | adaptive-step steepest descent (GROMACS-style)                | this file                  |

## Phase-Kind Dispatch <!-- rq-54810de5 -->

A simulation is a sequence of one or more phases declared as a TOML
array of tables. Phases come in two **kinds**:

- **MD phases** declared as `[[phase]]` tables. Each MD phase carries a
  timestep, integrator, optional thermostat, optional barostat, and
  output cadences (see `io/config-schema.md`'s `[[phase]]` section).
- **Minimization phases** declared as `[[minimization]]` tables. Each
  minimization phase carries an algorithm selector, algorithm-specific
  parameters, convergence criteria, and output cadences (see *Schema*
  below).

The runner unifies the two arrays into a single `Vec<PhaseSlot>` in
the order they appear in the source TOML document. The
deserialiser captures the source byte offset of each `[[phase]]` and
`[[minimization]]` table via `toml::Spanned<T>`, then sorts the merged
sequence by offset to recover document order. MD phases and
minimization phases may be freely interleaved.

Particle state (positions, velocities, simulation box, charges) and
the global `ForceField` carry over between phases regardless of kind.
Slot handles (integrator, thermostat, barostat, constraint, minimizer)
are built fresh at every phase boundary and dropped at every phase end.
Per-phase output writers and the per-phase `Timings` instance are
likewise per-phase. Phase names are unique across the unified
sequence; a name collision between an MD phase and a minimization
phase surfaces as `ConfigError::DuplicatePhaseName { name }`.

## Algorithm <!-- rq-d4c2b6cd -->

Steepest descent advances positions along the negative gradient of the
potential energy `U(x)`, with a single global scalar step size that is
adapted between iterations based on whether the previous trial reduced
the energy. The step formula is GROMACS-style:

```
x_trial = x + step · (F / F_max)
```

where `F = -∇U(x)` is the per-atom force at the current accepted
positions, `F_max = max_i ||F_i||` is the maximum per-atom force
magnitude, and `step` is the current scalar step size (metres). The
atom with the largest force magnitude moves by exactly `step`; every
other atom moves proportionally less. The division by `F_max` keeps
the maximum per-atom displacement bounded regardless of force scale.

### Per-iteration sequence <!-- rq-3f080cf2 -->

For iteration `k` with current accepted positions `x_k`, accepted
energy `E_k`, and current step size `step_k`:

1. **Build trial.** Compute `F_max = max_i ||F_i(x_k)||`. If
   `F_max == 0.0`, declare convergence with reason
   `MinimizerConvergence::ForceZero` and stop. Otherwise compute
   `x_trial = x_k + step_k · (F(x_k) / F_max)`.
2. **Project constraints (if enabled).** When the run carries a
   `Constraint` slot, call
   `constraint.apply_position_projection_only(buffers, sim_box,
   timings)` on `x_trial` (see `integration/constraint-framework.md`).
   The projection mutates `x_trial` in place; velocities are not
   touched.
3. **Evaluate trial energy.** Run
   `force_field.step(buffers, sim_box, timings)` to populate
   `buffers.forces_*` at `x_trial` and to compute the trial potential
   energy `E_trial` via `compute_total_potential_energy`.
4. **Accept or reject.**
   - If `E_trial < E_k`: **accept**. Set `x_{k+1} = x_trial`,
     `E_{k+1} = E_trial`, `step_{k+1} = min(step_increase · step_k,
     max_step)`, increment the accepted-step counter.
   - If `E_trial >= E_k`: **reject**. Discard `x_trial` (restore the
     accepted positions from the iteration's pre-trial snapshot), set
     `step_{k+1} = step_decrease · step_k`. Forces are re-evaluated at
     the restored positions before the next iteration's trial.
5. **Check convergence.** A successful step ends the phase when either
   physical criterion is met:
   - `F_max(x_{k+1}) ≤ force_tolerance` — `MinimizerConvergence::ForceTolerance`.
   - `|E_k - E_{k+1}| / max(|E_k|, |E_{k+1}|, ε_floor) ≤ energy_tolerance`
     and at least one prior step was accepted —
     `MinimizerConvergence::EnergyTolerance`, with
     `ε_floor = 1.0e-30` to avoid divide-by-zero when both energies
     are zero.
   Rejected steps never trigger convergence; the energy-tolerance check
   compares the last two distinct accepted energies.
6. **Check iteration cap.** When the iteration counter `k` reaches
   `max_iterations`, the phase ends with a **non-convergence failure**.
   The runner surfaces this as
   `RunnerError::MinimizerNonConvergence { phase: <phase name>,
   iterations: max_iterations, final_force: F_max(x_k),
   final_step: step_k }` and exits with code `2`. Subsequent phases
   do not run.

The iteration counter advances on every loop body iteration (accepted
**or** rejected), so `max_iterations` caps total force evaluations.
A rejected step counts as one iteration.

### Initial iteration <!-- rq-39ab27d9 -->

Before the loop, the runner performs the same warm-up
`force_field.step` call as for an MD phase (step 15 of *Runner flow*
in `simulation-runner.md`) so that `buffers.forces_*` and the initial
potential energy `E_0` are populated. The first iteration's trial uses
`step_0 = initial_step`.

If `F_max(x_0) ≤ force_tolerance`, the phase converges immediately
without any trial step; the runner writes a step-0 `.minlog` row and
the convergence-reason summary, then proceeds to the next phase.

### Empty state <!-- rq-cc9f4623 -->

When `particle_count == 0`, the phase emits a single step-0 `.minlog`
row with `energy = 0.0`, `max_force = 0.0`, `step = initial_step`,
and `accepted = true`, declares convergence with
`MinimizerConvergence::ForceZero`, and proceeds to the next phase
without launching any kernel beyond the warm-up `force_field.step`
(which itself is a no-op on empty buffers).

## Constraint Handling <!-- rq-4241d94b -->

Constraints participate in minimization through a new trait method on
the `Constraint` slot, `apply_position_projection_only`, that performs
the SHAKE-style position projection without the per-step velocity
correction that the integration hooks apply. The trait method, its
contract, and the registered SETTLE implementation are documented in
`integration/constraint-framework.md` and `integration/settle.md`.

A minimization phase consults the same global `ConstraintList` as the
MD phases. The runner constructs the constraint slot at runner
startup (once, before any phase begins) and reuses it across both
phase kinds. The `Minimizer` does not own the constraint slot; the
runner holds it and threads it into the per-iteration step.

The SD minimizer is compatible with every registered constraint
algorithm. The compatibility predicate
`MinimizerBuilder::supports_constraints(&params)` returns `true` for
the `steepest-descent` builder regardless of its params, parallel to
the `velocity-verlet { lossless = false }` integrator's behaviour.

When the run has no constraints, the projection step is skipped (the
runner holds `None` for the constraint slot and the loop body does
not call the projection hook).

## Output <!-- rq-71fa84f4 -->

A minimization phase writes two files (in addition to the per-phase
`.timings` file that every phase writes):

- **`.minlog`** — per-iteration diagnostics CSV. Always written when
  `output.minlog_every > 0`. Always opened with a header line; rows
  appear at the cadence below.
- **`.xyz`** — periodic positions trajectory, sharing the same writer
  and format as MD trajectory output (`io/trajectory-output.md`).
  Optional; controlled by `output.trajectory_every`. Velocities are
  not written even when `output.include_velocities = true` (the
  velocity columns carry whatever values were in the buffer at phase
  entry; they do not change during minimization, so writing them would
  duplicate the init-state velocities).

### `.minlog` format <!-- rq-119cbe46 -->

The `.minlog` is RFC-4180 CSV with header line and one row per logged
iteration:

```
iter,energy,max_force,step,accepted
0,4.123456789e-18,2.345678901e-10,1.000000000e-12,1
1,4.012345678e-18,2.234567890e-10,1.200000000e-12,1
2,4.012345678e-18,2.234567890e-10,2.400000000e-13,0
...
```

Columns:

- `iter: u64` — phase-local iteration counter. The pre-loop step-0
  row carries `iter = 0`. Rejected iterations advance the counter
  alongside accepted ones.
- `energy: f64` — total potential energy at the iteration's accepted
  positions, expressed in the unit system the minlog writer was opened
  with (joules in `UnitSystem::Si`, Hartrees in `UnitSystem::Atomic`).
  The engine computes the value in Hartrees via
  `compute_total_potential_energy`. For a rejected iteration the
  reported `energy` is the accepted energy *before* the rejected
  trial (i.e., the same energy that was reported on the previous
  accepted row), since the rejected trial's positions and energy are
  discarded.
- `max_force: f64` — `F_max` at the iteration's accepted positions,
  expressed in the unit system the minlog writer was opened with
  (newtons in `UnitSystem::Si`, `E_h / a_0` in `UnitSystem::Atomic`).
  The engine computes `F_max` in atomic-unit force and the writer
  applies the output-direction conversion before formatting.
- `step: f64` — the step size *used for the trial that this row
  represents*. The step-0 row carries `initial_step`.
- `accepted: u32` — `1` if the trial was accepted, `0` if rejected.
  The step-0 row carries `1` (the initial state is "accepted" by
  fiat).

Number formatting: same as the MD log (`{:.9e}` for floats, base-10
unpadded for integers). Line endings `\n`; encoding UTF-8.

Cadence: when `minlog_every == 0`, no `.minlog` file is created. When
`minlog_every > 0`, the writer emits the step-0 row at phase entry
plus one row at every iteration `k` such that
`k % minlog_every == 0` and `1 ≤ k ≤ max_iterations`, plus one final
row at the iteration that triggered convergence (if it would not
otherwise have been written). The final row always appears so the
user always sees the converged state's energy and `F_max`.

### `.xyz` trajectory <!-- rq-2032660c -->

When `output.trajectory_every > 0`, the runner writes one trajectory
frame at phase entry (step-0 positions), one frame at every
iteration `k` such that `k % trajectory_every == 0` (only accepted
iterations advance the position state — a rejected iteration's
restored positions are byte-identical to the previous accepted state,
so writing a frame for a rejected iteration is permitted but
redundant), and one final frame at the convergence iteration.

The trajectory file's `Step=` attribute and `Time=` attribute carry
the iteration counter as `Step=k` and `Time=0.0` (minimization has
no physical-time semantics). Consumers parse this the same way they
parse MD trajectory frames; the missing `Time` progression is the
only difference.

### Final summary <!-- rq-6eb845c5 -->

After the phase completes (successfully or with non-convergence), the
runner emits one line of phase summary to stdout, parallel to the
per-phase line MD phases emit:

```
[dynamics] phase `min`: 87 iters in 412 ms (converged: force_tolerance, frames: 0, log rows: 88)
```

Carries: phase name, accepted-iteration count (or `max_iterations`
on non-convergence), wall-clock elapsed time, convergence reason
(`force_tolerance`, `energy_tolerance`, `force_zero`, or
`max_iterations`), trajectory frame count, and minlog row count.

## Schema <!-- rq-38d5e7a8 -->

A minimization phase is declared as a `[[minimization]]` table:

```toml
[[minimization]]
name = "min"

[minimization.algorithm]
kind = "steepest-descent"

# Optional adaptive-step parameters; defaults shown:
initial_step = 1.0e-12         # m
max_step = 1.0e-10             # m
step_increase = 1.2            # multiplicative factor on accept
step_decrease = 0.2            # multiplicative factor on reject

# Optional convergence parameters; defaults shown:
force_tolerance = 1.0e-10      # N
energy_tolerance = 1.0e-7      # relative
max_iterations = 1000

[minimization.output]
minlog_every = 1               # 0 disables the .minlog
trajectory_every = 0           # 0 disables intermediate frames
include_images = true
```

### Top-level fields <!-- rq-50f2cc42 -->

- `name: String` — required. Identifier used to derive output filenames
  (`<root>.out.<name>.{minlog,xyz,timings}`). Same character set and
  uniqueness rules as `[[phase]]`'s `name` (non-empty, ASCII
  letters/digits/`-`/`_`, unique across both `[[phase]]` and
  `[[minimization]]` arrays).

### `[minimization.algorithm]` <!-- rq-0a2ca9ac -->

The `[minimization.algorithm]` table is required and carries a `kind`
field plus algorithm-specific parameters. The Rust-side deserialiser
captures `kind` into a `SlotConfig` and flattens the rest of the
section into `params: toml::Value`. The chosen builder's
`validate_params(&toml::Value)` enforces required fields, domains, and
rejects unknown fields.

- `kind: String` — required. One of:
  - `"steepest-descent"` — adaptive-step steepest descent (see *Algorithm*
    above).

Fields accepted for `kind = "steepest-descent"` (all optional with
defaults):

- `initial_step: f64` — initial scalar step `step_0` in Bohr (`a_0`).
  Default `1.0e-12` metres ≈ `1.89e-2` Bohr (the loader converts
  user-supplied SI defaults). Finite and strictly positive.
- `max_step: f64` — upper bound on `step` in Bohr (`a_0`). Default
  `1.0e-10` metres = 1 Å ≈ 1.89 Bohr. Finite, strictly positive, and
  `≥ initial_step`.
- `step_increase: f64` — multiplicative factor applied to `step` on
  an accepted iteration. Default `1.2`. Finite and `≥ 1.0`. A value
  of `1.0` disables step growth (a fixed-step variant).
- `step_decrease: f64` — multiplicative factor applied to `step` on
  a rejected iteration. Default `0.2`. Finite and in `(0.0, 1.0)`.
- `force_tolerance: f64` — convergence threshold on `F_max` in
  `E_h / a_0` (the engine's atomic force unit). Default `1.0e-10`
  newtons ≈ `1.2e-3` in atomic units. Finite and `≥ 0.0`. A value of
  `0.0` disables this criterion (only `energy_tolerance` and
  `max_iterations` end the loop).
- `energy_tolerance: f64` — relative convergence threshold on
  `|ΔE| / max(|E_prev|, |E_curr|, 1.0e-30)` between consecutive
  accepted iterations. Default `1.0e-7`. Finite and `≥ 0.0`. A value
  of `0.0` disables this criterion.
- `max_iterations: u64` — iteration cap. Default `1000`. Strictly
  positive. Non-convergence after `max_iterations` is a hard error
  (see *Algorithm*'s step 6).

Unknown fields under `[minimization.algorithm]` for the chosen `kind`
surface as `ConfigError::Parse { path:
"minimization[<i>].algorithm", message }`.

### `[minimization.output]` <!-- rq-2443920b -->

The `[minimization.output]` table is optional. When omitted, the
phase uses the default cadences and the default file paths.

- `minlog_path: String` — output `.minlog` path for this phase.
  Default: `<config-root>.out.<phase-name>.minlog` in the same
  directory as the config file. Resolved relative to the config
  file's directory; absolute paths are honoured as-is.
- `minlog_every: u64` — write one `.minlog` row every this many
  iterations within this phase. Default `1`. `0` disables the
  `.minlog` entirely (no header is written, no file is created).
- `trajectory_path: String` — output `.xyz` path for this phase.
  Default: `<config-root>.out.<phase-name>.xyz`. Same resolution
  rules as `[phase.output].trajectory_path`.
- `trajectory_every: u64` — write one trajectory frame every this
  many iterations. Default `0` (disabled). When `> 0`, the frame at
  iteration `0` is always written, plus one at every multiple of
  `trajectory_every`, plus one final frame at the convergence
  iteration.
- `include_images: bool` — include `image:I:3` columns in
  trajectory frames. Default `true`. Same semantics as the MD-phase
  field.
- `timings_path: String` — output `.timings` path for this phase.
  Default: `<config-root>.out.<phase-name>.timings`.

The minimization phase does not emit a `.log` (MD CSV log); the
`.minlog` replaces it. The `include_velocities` field is not
accepted (velocities do not change during minimization; including
them in trajectory frames would duplicate the phase-entry velocities
without adding information). Setting `include_velocities` under
`[minimization.output]` surfaces as `ConfigError::Parse`.

### Cross-validation <!-- rq-e2bb500b -->

- Phase names are unique across the unified `[[phase]]` and
  `[[minimization]]` arrays. A collision surfaces as
  `ConfigError::DuplicatePhaseName { name }`.
- Output paths are pairwise distinct across the whole config
  (extending the existing `PathRole` enum with
  `MinimizationMinlog { phase: String }`,
  `MinimizationTrajectory { phase: String }`, and
  `MinimizationTimings { phase: String }` variants).
- A minimization phase's algorithm kind is dispatched against
  `registries.minimizers` (a new field on the `Registries` bundle).
  An unknown kind surfaces as
  `ConfigError::UnknownKind { slot: "minimization", kind }`.
- A minimization phase is **incompatible** with any constraint
  algorithm whose builder's
  `ConstraintBuilder::supports_position_projection_only(&params)`
  returns `false` (a new predicate; see
  `integration/constraint-framework.md`). The check runs per phase via
  `Config::validate_constraint_compatibility`. The default-registry
  `settle` builder returns `true`, so this rejection path is
  reachable only by future constraint algorithms.
- A `[[minimization]]` phase carries no integrator, thermostat, or
  barostat section. The TOML deserialiser rejects those keys under
  `[[minimization]]` as unknown fields.

## Construction and Lifetime <!-- rq-61b646d2 -->

The runner constructs each minimization phase's `Minimizer` slot
between *Runner flow* steps 11 (force-field construction) and 13
(output writer opening) of `simulation-runner.md`, in the same place
the integrator / thermostat / barostat slots are constructed for an
MD phase. The unified per-phase loop dispatches on the phase kind:

```text
for phase in phases:
    match phase.kind:
        PhaseKind::Md(P) =>
            run_md_phase(P, &mut buffers, &mut sim_box, &mut force_field,
                         constraint.as_mut(), &registries, ...)
        PhaseKind::Minimization(P) =>
            run_minimization_phase(P, &mut buffers, &mut sim_box, &mut force_field,
                                   constraint.as_mut(), &registries, ...)
```

`run_minimization_phase` builds the `Minimizer` slot via
`registries.minimizers.build(&P.algorithm, &gpu, particle_count,
n_constraints)`, opens the per-phase `.minlog` and optional `.xyz`
writers, runs the SD loop documented under *Algorithm*, flushes and
closes the writers, drains `Timings`, drops the slot.

The minimizer slot's per-phase allocations (a length-1
`CudaSlice<f32>` for `F_max` reduction, a length-1 `CudaSlice<f32>`
for the total potential energy reduction, and per-particle scratch
buffers for the position snapshot on rejection rollback) persist for
the lifetime of the phase and are dropped at end of phase.

## Determinism <!-- rq-6112a43b -->

Steepest descent preserves the project's bit-wise reproducibility
invariant on the same GPU under identical inputs:

- The `F_max` reduction is a deterministic single-block reduction in
  fixed left-to-right index order, parallel to
  `compute_total_potential_energy`'s structure.
- The position update `x_trial = x + step · (F / F_max)` is a
  per-particle `f32` arithmetic sequence with no atomics and no
  cross-particle dependencies.
- The accept/reject decision compares two `f64` energy values
  computed by the same deterministic reduction; the comparison is
  bit-exact.
- The position-snapshot rollback on rejection copies device buffers
  through `cudaMemcpyAsync` on the default stream; the source and
  destination are both per-particle SoA arrays of fixed length.
- The minimizer slot holds no random-number state.

Two runs on the same GPU with identical configs and init files
produce byte-identical post-minimization positions and byte-identical
`.minlog` rows.

## Feature API <!-- rq-f1befa71 -->

### Types <!-- rq-187a5252 -->

- `Minimizer` — object-safe trait implemented by every concrete <!-- rq-f69350df -->
  minimizer slot.

  ```rust
  pub trait Minimizer: std::fmt::Debug + Send {
      /// Execute one outer iteration. Reads `buffers.forces_*` at
      /// the current accepted positions (populated by the runner via
      /// `force_field.step` before the call), proposes a trial
      /// position, optionally projects constraints, evaluates the
      /// trial energy via the supplied `force_field` callback,
      /// accepts or rejects, and reports the outcome.
      fn step(
          &mut self,
          buffers: &mut ParticleBuffers,
          sim_box: &SimulationBox,
          force_field: &mut dyn ForceFieldCallback,
          constraint: Option<&mut dyn Constraint>,
          timings: &mut Timings,
      ) -> Result<MinimizerStepReport, MinimizerError>;

      /// Convergence-check helper used by the runner between calls
      /// to `step`. Returns `Some(reason)` when the most recent
      /// accepted state satisfies any of the configured criteria,
      /// or `None` to continue iterating.
      fn check_convergence(
          &self,
          report: &MinimizerStepReport,
      ) -> Option<MinimizerConvergence>;
  }
  ```

  - `step` is called once per outer iteration by the runner. It must
    not call `force_field.step` more than once per call (one trial
    evaluation per iteration). It must not advance any output
    counter; the runner owns the iteration counter.
  - `ForceFieldCallback` is a thin trait re-exposing
    `force_field.step(buffers, sim_box, timings)` for the minimizer
    to invoke; the runner's `ForceField` implements it. The
    indirection keeps the `Minimizer` trait object-safe and avoids
    threading the full `ForceField` type through the trait surface.
  - The runner exposes the constraint slot to `step` as
    `Option<&mut dyn Constraint>` so the minimizer can invoke
    `apply_position_projection_only` after the position update.
    Passing `None` is permitted (no constraints).
  - `check_convergence` is pure (no kernel launches, no buffer
    mutations).

- `MinimizerStepReport` — outcome of one `step` call. <!-- rq-208aaa72 -->

  ```rust
  pub struct MinimizerStepReport {
      pub accepted: bool,
      pub energy: f64,       // Hartrees, at the post-step accepted positions
      pub max_force: f64,    // E_h / a_0, at the post-step accepted positions
      pub step_size: f64,    // Bohr (a_0), the step used for this iteration's trial
      pub prev_energy: f64,  // Hartrees, accepted energy before this iteration
  }
  ```

  - `energy` is the accepted-state energy. For an accepted iteration
    it is the new trial's energy; for a rejected iteration it is the
    pre-trial accepted energy (unchanged from the previous row).
  - `max_force` is the accepted-state `F_max`. Same accepted-vs-
    rejected semantics as `energy`.
  - `prev_energy` is the accepted energy *entering* the iteration
    (i.e., before any trial). Used by the runner's energy-tolerance
    check.

- `MinimizerConvergence` — enum reporting why the loop ended. <!-- rq-77f64e46 -->

  ```rust
  pub enum MinimizerConvergence {
      ForceTolerance,
      EnergyTolerance,
      ForceZero,
      MaxIterations,
  }
  ```

  - `ForceTolerance` — `F_max ≤ force_tolerance` after the iteration.
  - `EnergyTolerance` — `|ΔE| / max(|E_prev|, |E_curr|, 1e-30) ≤
    energy_tolerance` between consecutive accepted iterations.
  - `ForceZero` — `F_max == 0.0` at iteration `k`'s accepted state
    (force evaluation produced an exact-zero gradient). Used for the
    `particle_count == 0` case and for already-converged inputs.
  - `MaxIterations` — the iteration counter reached
    `max_iterations` without any physical criterion firing. Drives
    `RunnerError::MinimizerNonConvergence`.

- `MinimizerError` — error type returned by every trait method. <!-- rq-78041a25 -->
  Variants:
  - `Gpu(GpuError)` — CUDA driver / kernel-launch failure.
  - `Timings(TimingsError)` — CUDA event recording failure.
  - `ForceField(ForceFieldError)` — the embedded
    `force_field.step` call failed.
  - `Constraint(ConstraintError)` — the embedded
    `apply_position_projection_only` call failed.
  - `UnknownKind(String)` — registry has no builder for the requested
    minimizer kind.

  The runner's `RunnerError` wraps this via
  `RunnerError::Minimizer(MinimizerError)`.

- `MinimizerRegistry` — host-side registry of minimizer builders. <!-- rq-d5b07d2a -->
  Holds `builders: Vec<Box<dyn MinimizerBuilder>>`.

  Methods:

  - `MinimizerRegistry::with_builtins() -> MinimizerRegistry` —
    constructs a registry pre-populated with the `steepest-descent`
    builder.
  - `MinimizerRegistry::register(&mut self, builder: Box<dyn MinimizerBuilder>)`
    — appends a builder. Duplicate `kind_name()` is not detected at
    registration; the lookup returns the first match.
  - `MinimizerRegistry::lookup(&self, kind: &str) -> Option<&dyn MinimizerBuilder>`
    — returns the first registered builder whose `kind_name()`
    equals `kind`.
  - `MinimizerRegistry::build(&self, slot: &SlotConfig, gpu: &GpuContext, particle_count: usize, n_constraints: usize) -> Result<Box<dyn Minimizer>, MinimizerError>`
    — looks up the builder whose `kind_name()` equals `slot.kind` and
    delegates `build(gpu, particle_count, n_constraints,
    &slot.params)`. Returns
    `MinimizerError::UnknownKind(slot.kind.clone())` when no builder
    matches.

  `Registries` (the runner-level bundle in `simulation-runner.md`)
  carries a new field `minimizers: MinimizerRegistry`, populated by
  `MinimizerRegistry::with_builtins()` in
  `Registries::with_builtins()`.

- `MinimizerBuilder` — trait describing a registered minimizer <!-- rq-dddb8e7a -->
  implementation. Implementations are stateless and self-register at
  construction time.

  ```rust
  pub trait MinimizerBuilder: std::fmt::Debug + Send + Sync {
      fn kind_name(&self) -> &'static str;

      fn validate_params(&self, params: &toml::Value)
          -> Result<(), ConfigError>;

      /// `true` iff this minimizer can drive a constraint slot whose
      /// `apply_position_projection_only` predicate is `true`. The
      /// default returns `true`; future minimizers that cannot
      /// participate in constrained minimization may override.
      fn supports_constraints(&self, _params: &toml::Value) -> bool { true }

      fn build(
          &self,
          gpu: &GpuContext,
          particle_count: usize,
          n_constraints: usize,
          params: &toml::Value,
      ) -> Result<Box<dyn Minimizer>, MinimizerError>;
  }
  ```

  - `validate_params` is a pure function of the supplied parameters
    and is called by `Config::validate_against` before any GPU work.
  - The `steepest-descent` builder's `validate_params` deserialises
    every optional parameter, fills in defaults for absent fields,
    enforces domains (positivity, ranges, `max_step ≥ initial_step`,
    `step_increase ≥ 1.0`, `0.0 < step_decrease < 1.0`), and rejects
    unknown fields.
  - `build` constructs the device-side slot. Allocations include a
    length-`particle_count` snapshot buffer for rollback and two
    length-1 reduction scratches (`f_max_scratch`, `pe_scratch`).
    `particle_count == 0` is permitted; reduction scratches still
    allocate at length 1.

- `SteepestDescentMinimizer` — concrete implementor of `Minimizer` <!-- rq-dc3b1bb5 -->
  for `kind = "steepest-descent"`. Fields:
  - `device: Arc<CudaDevice>`
  - `particle_count: usize`
  - `n_constraints: usize`
  - `initial_step: f32`
  - `max_step: f32`
  - `step_increase: f32`
  - `step_decrease: f32`
  - `force_tolerance: f64`
  - `energy_tolerance: f64`
  - `max_iterations: u64`
  - `current_step: f32` — the accumulated adaptive step size; reset
    to `initial_step` at slot construction.
  - `iteration: u64` — phase-local iteration counter owned by the
    minimizer for diagnostic purposes (the runner also tracks
    iterations independently for the `.minlog`).
  - `snapshot_x: CudaSlice<f32>`, `snapshot_y: CudaSlice<f32>`,
    `snapshot_z: CudaSlice<f32>` — per-particle position snapshot
    captured before each trial and used to roll back on rejection.
  - `f_max_scratch: CudaSlice<f32>` — length-1 device buffer for the
    deterministic `F_max` reduction.
  - `pe_scratch: CudaSlice<f32>` — length-1 device buffer for the
    deterministic total-potential-energy reduction.

- `MinimizationConfig` — parsed `[[minimization]]` entry. Lives in <!-- rq-ed61cf26 -->
  `crate::io::config` alongside `PhaseConfig`.

  ```rust
  pub struct MinimizationConfig {
      pub name: String,
      pub algorithm: SlotConfig,
      pub output: MinimizationOutputConfig,
  }
  ```

- `MinimizationOutputConfig` — parsed `[minimization.output]` table <!-- rq-758b03ef -->
  with resolved paths and populated defaults.

  ```rust
  pub struct MinimizationOutputConfig {
      pub minlog_path: PathBuf,
      pub minlog_every: u64,
      pub trajectory_path: PathBuf,
      pub trajectory_every: u64,
      pub include_images: bool,
      pub timings_path: PathBuf,
  }
  ```

- `PhaseKind` — runtime-level discriminated union over the unified <!-- rq-4a0c5f2e -->
  phase sequence. Lives in `crate::runner`.

  ```rust
  pub enum PhaseKind {
      Md(PhaseConfig),
      Minimization(MinimizationConfig),
  }
  ```

  The `Config` type's `phases` field is replaced by
  `phases: Vec<PhaseKind>` (see `io/config-schema.md` for the
  config-schema-level statement of this).

- `MinlogWriter` — handle to an open `.minlog` file. Fields are <!-- rq-dc140510 -->
  private; the type encapsulates the buffered writer. The header
  emitted at open is exactly
  `iter,energy,max_force,step,accepted\n`.

  Methods:

  - `MinlogWriter::open(path: &Path) -> Result<MinlogWriter, MinlogWriterError>`
    — creates the file with `OpenOptions::new().write(true).create_new(true)`;
    returns `OutputExists { path }` when the file already exists; writes
    the header line immediately.
  - `MinlogWriter::write_row(&mut self, iter: u64, energy: f64, max_force: f64, step: f64, accepted: bool) -> Result<(), MinlogWriterError>`
    — writes one CSV row in the format described under *Output /
    `.minlog` format*. The `accepted` bool serialises as the integer
    `1` or `0`.
  - `MinlogWriter::flush(&mut self) -> Result<(), MinlogWriterError>`
    — flushes the internal buffer.

- `MinlogWriterError` — error type. Variants `OutputExists { path: <!-- rq-82621837 -->
  PathBuf }` and `Io(String)`, parallel to `LogWriterError`.

### Functions and methods <!-- rq-e38eed0c -->

- `run_minimization_phase(phase: &MinimizationConfig, buffers: &mut ParticleBuffers, sim_box: &mut SimulationBox, force_field: &mut ForceField, constraint: Option<&mut dyn Constraint>, registries: &Registries, gpu: &GpuContext) -> Result<MinimizationPhaseSummary, RunnerError>` <!-- rq-393a57e4 -->
  - The runner's per-phase entry point for a minimization phase.
    Constructs the minimizer slot via
    `registries.minimizers.build(...)`, opens the `.minlog` and
    optional `.xyz` writers, runs the loop documented under
    *Algorithm*, flushes and closes the writers, drains the per-phase
    `Timings`, and returns a summary suitable for the runner's final
    summary line.
  - Non-convergence after `max_iterations` returns
    `Err(RunnerError::MinimizerNonConvergence { phase:
    phase.name.clone(), iterations, final_force, final_step })`. The
    runner propagates the error and exits with code `2`. No further
    phases run.

- `Minimizer::step(&mut self, buffers, sim_box, force_field, constraint, timings)` <!-- rq-53d48fb8 -->
  - As described under *Types*.

- `Minimizer::check_convergence(&self, report)` <!-- rq-1440b6e6 -->
  - As described under *Types*.

- `MinimizerRegistry::with_builtins() -> MinimizerRegistry` <!-- rq-237b5543 -->
  - Returns a registry with one registered builder
    (`SteepestDescentBuilder`).

- `MinlogWriter::open`, `MinlogWriter::write_row`, <!-- rq-e963f27a -->
  `MinlogWriter::flush` — as described under *Types*.

## CUDA Kernels <!-- rq-47a5fe0e -->

The minimizer's per-step work runs on the existing
`compute_total_potential_energy` reduction plus three small new
kernels in `kernels/minimize.cu`:

```c
extern "C" __global__ void sd_compute_step(
    float *positions_x, float *positions_y, float *positions_z,
    const float *forces_x, const float *forces_y, const float *forces_z,
    float step_size,
    float inv_f_max,
    unsigned int n);

extern "C" __global__ void sd_snapshot(
    const float *positions_x, const float *positions_y, const float *positions_z,
    float *snapshot_x, float *snapshot_y, float *snapshot_z,
    unsigned int n);

extern "C" __global__ void sd_restore(
    float *positions_x, float *positions_y, float *positions_z,
    const float *snapshot_x, const float *snapshot_y, const float *snapshot_z,
    unsigned int n);
```

Plus the existing `compute_f_max_reduction` kernel (or, if not yet
exposed by the existing reduction utilities, a deterministic
single-block reduction kernel with the same fixed left-to-right
index order as `compute_total_potential_energy`).

- `sd_compute_step` reads `(positions_*, forces_*)`, computes
  `positions_d ← positions_d + step_size · forces_d · inv_f_max` per
  axis, and writes back to `positions_*`. One thread per particle.
- `sd_snapshot` copies `positions_*` into the slot's snapshot
  buffers; called before each trial.
- `sd_restore` copies snapshot buffers back into `positions_*`;
  called after a rejected trial.

All three kernels run with block size 256, grid size
`ceil(n / 256)`, zero shared memory, on the default stream of the
`ParticleBuffers`'s `Arc<CudaDevice>`.

### Per-iteration kernel sequence (one accepted iteration) <!-- rq-063beff1 -->

| Order | Kernel                                | Notes                                                                                  |
| ----- | ------------------------------------- | -------------------------------------------------------------------------------------- |
| 1     | `compute_f_max_reduction`             | reduces `forces_*` to a single `f32` `F_max` on the device                             |
| 2     | `sd_snapshot`                         | captures current accepted positions                                                    |
| 3     | `sd_compute_step`                     | writes trial positions                                                                 |
| 4     | (constraint slot, when present)       | `apply_position_projection_only` — projects trial positions onto the manifold          |
| 5     | `force_field.step` (full pipeline)    | populates `forces_*` and `potential_energies_*` at the trial positions                 |
| 6     | `compute_total_potential_energy`      | reduces `potential_energies_*` to a single `f32` `E_trial` on the device                |

Rejected iteration: as above, but after step 6 the runner calls
`sd_restore` to roll positions back to the snapshot, then calls
`force_field.step` and `compute_total_potential_energy` one more time
to re-populate forces and the accepted energy at the restored
positions. (The next iteration's `sd_compute_step` reads the
re-evaluated forces.) The cost of a rejection is one extra
`force_field.step` per rejected iteration.

## Out of Scope <!-- rq-7573ac91 -->

- Conjugate-gradient (`kind = "conjugate-gradient"`) and L-BFGS
  (`kind = "l-bfgs"`) minimizers. Reserved for future features;
  registered alongside `steepest-descent` in the same
  `MinimizerRegistry`.
- Per-atom step size adaptation. SD uses a single global scalar
  `step` adapted across the whole system; per-atom step controllers
  (preconditioned SD, FIRE) are out of scope.
- Minimization of cell parameters / barostatic minimization (variable
  cell). The simulation box is read-only during minimization.
- Restart from a `.minlog`. Restart input/output is out of scope for
  the runner generally (see `simulation-runner.md`).
- Multi-stream or multi-GPU launches.
- Reporting per-particle gradient magnitudes in the `.minlog`. The
  per-iteration row reports the scalar `F_max` only; per-particle
  inspection is left to trajectory frames.
- Composing a minimization phase with a thermostat or barostat. The
  `[[minimization]]` table rejects `[minimization.thermostat]` /
  `[minimization.barostat]` keys at TOML parse time.

---

## Gherkin Scenarios <!-- rq-89e8f3d7 -->

```gherkin
Feature: Steepest-descent energy minimizer

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called
    And a MinimizerRegistry::with_builtins() with one builder of kind "steepest-descent"

  # --- Schema parsing ---

  @rq-57a0f297
  Scenario: Parse a [[minimization]] section with defaults
    Given a config file containing exactly one [[minimization]] table with name="min"
      and [minimization.algorithm] kind="steepest-descent" and no other fields
    When load_config(path) is called
    Then it returns Ok(config)
    And config.phases has length 1
    And config.phases[0] is PhaseKind::Minimization(M)
    And M.name == "min"
    And M.algorithm.kind == "steepest-descent"
    And the builder-resolved params carry initial_step = 1.0e-12, max_step = 1.0e-10,
      step_increase = 1.2, step_decrease = 0.2, force_tolerance = 1.0e-10,
      energy_tolerance = 1.0e-7, max_iterations = 1000

  @rq-bcc16e10
  Scenario: Reject an unknown field under [minimization.algorithm]
    Given a config file with [minimization.algorithm] kind="steepest-descent" and a junk field
    When load_config(path) is called
    Then it returns Err(ConfigError::Parse { path: "minimization[0].algorithm", .. })

  @rq-5cbcbabe
  Scenario: Reject a thermostat under [[minimization]]
    Given a config file with [[minimization]] name="min" and [minimization.thermostat] kind="csvr"
    When load_config(path) is called
    Then it returns Err(ConfigError::Parse { path: "minimization[0]", .. })

  @rq-30d3f1fa
  Scenario: Reject step_decrease >= 1.0
    Given a config with [minimization.algorithm] kind="steepest-descent" and step_decrease=1.0
    When load_config(path) is called
    Then it returns Err(ConfigError::InvalidValue { field: "minimization[0].algorithm.step_decrease", .. })

  @rq-a82a275a
  Scenario: Reject max_step < initial_step
    Given a config with initial_step=1.0e-10 and max_step=1.0e-12
    When load_config(path) is called
    Then it returns Err(ConfigError::InvalidValue { field: "minimization[0].algorithm.max_step", .. })

  @rq-69109415
  Scenario: Reject an unknown minimization algorithm kind
    Given a config with [minimization.algorithm] kind="quasi-newton"
    When load_config(path) is called
    Then it returns Err(ConfigError::UnknownKind { slot: "minimization", kind: "quasi-newton" })

  @rq-33b1f2d2
  Scenario: Phase-name collision between [[phase]] and [[minimization]]
    Given a config with a [[phase]] named "step1" and a [[minimization]] also named "step1"
    When load_config(path) is called
    Then it returns Err(ConfigError::DuplicatePhaseName { name: "step1" })

  # --- Phase-kind dispatch and ordering ---

  @rq-09ed630b
  Scenario: Interleaved [[phase]] and [[minimization]] preserve document order
    Given a config whose TOML body declares, in source order,
      [[phase]] name="equil", then [[minimization]] name="min", then [[phase]] name="prod"
    When load_config(path) is called
    Then config.phases[0] is PhaseKind::Md with name="equil"
    And config.phases[1] is PhaseKind::Minimization with name="min"
    And config.phases[2] is PhaseKind::Md with name="prod"

  @rq-f5a4623c
  Scenario: A run with one [[minimization]] phase executes the SD loop and proceeds to the next phase
    Given a config with one [[minimization]] phase followed by one [[phase]] of n_steps=10
    And a 4-particle initial state with non-zero forces
    When dynamics run <config> is invoked
    Then the .minlog for the minimization phase exists and has at least the step-0 row
    And the subsequent MD phase's log exists and has the expected number of rows
    And the exit code is 0

  # --- Algorithm semantics ---

  @rq-420d2eb3
  Scenario: SD reduces energy of a 1D harmonic well in one step
    Given a 2-particle system with a single Morse bond at +5% stretch from r_e
    And [minimization.algorithm] kind="steepest-descent" with initial_step=1.0e-13
    When one minimizer.step is invoked
    Then the reported energy decreases by more than 1.0e-25 J
    And the reported accepted flag is true

  @rq-48e36163
  Scenario: SD step formula moves the largest-force atom by exactly `step`
    Given a 3-particle system with forces (F_max, F_max/2, 0) along x
    And initial_step = 1.0e-12 m
    When one minimizer.step is invoked
    Then particle 0's x-position changes by 1.0e-12 m within absolute tolerance 1.0e-18
    And particle 1's x-position changes by 5.0e-13 m within absolute tolerance 1.0e-18
    And particle 2's x-position is unchanged within absolute tolerance 1.0e-18

  @rq-265de297
  Scenario: SD doubles the step on accepted iteration
    Given a SD slot with initial_step=1.0e-12 and step_increase=2.0
    And the first trial reduces the energy
    When one minimizer.step is invoked
    Then the next iteration's step size is 2.0e-12 m within absolute tolerance 1.0e-20

  @rq-ba6d3eaa
  Scenario: SD caps step at max_step
    Given a SD slot with current step=8.0e-11 m, step_increase=2.0, max_step=1.0e-10
    And the trial reduces the energy
    When one minimizer.step is invoked
    Then the next iteration's step size is exactly max_step (1.0e-10 m)

  @rq-b10ed5ec
  Scenario: SD halves the step on rejected iteration and restores positions
    Given a SD slot with initial_step=1.0e-12, step_decrease=0.5
    And a snapshot of positions before the call
    And a trial that increases the energy (e.g. via an artificially large initial_step over a Morse well)
    When one minimizer.step is invoked
    Then the report has accepted == false
    And the next iteration's step size is 5.0e-13 m within absolute tolerance 1.0e-20
    And positions_x, positions_y, positions_z are byte-identical to the snapshot

  @rq-020ff80e
  Scenario: SD does not modify velocities
    Given a SD slot and a snapshot of buffers.velocities_* before the phase
    When the entire minimization phase runs to convergence
    Then velocities_x, velocities_y, velocities_z are byte-identical to the snapshot

  @rq-cd12ce51
  Scenario: SD does not modify simulation box
    Given a SD slot and a snapshot of sim_box before the phase
    When the entire minimization phase runs to convergence
    Then sim_box equals the snapshot byte-for-byte

  # --- Convergence ---

  @rq-77d98b9c
  Scenario: Convergence on force tolerance ends the phase
    Given a SD slot with force_tolerance = 1.0e-10
    And a system whose F_max already equals 5.0e-11 N at the initial positions
    When the minimization phase is invoked
    Then the phase terminates after iteration 0 with reason ForceTolerance
    And the .minlog contains exactly the step-0 row plus the final convergence row

  @rq-45e6b20a
  Scenario: Convergence on energy tolerance ends the phase
    Given a SD slot with energy_tolerance = 1.0e-7 and force_tolerance = 0.0
    And consecutive accepted energies E_k, E_{k+1} satisfying |E_k - E_{k+1}| / max(|E_k|, |E_{k+1}|, 1e-30) ≤ 1.0e-7
    When the phase is invoked
    Then the phase terminates with reason EnergyTolerance

  @rq-2e27a5fc
  Scenario: Energy tolerance does not fire on a rejected step
    Given a SD slot whose first trial is rejected
    When two iterations have run
    Then check_convergence does not return EnergyTolerance based on the rejected iteration alone

  @rq-90a67daf
  Scenario: Non-convergence at max_iterations is a hard error
    Given a SD slot with max_iterations = 5 and a system whose forces never fall below force_tolerance
    When dynamics run <config> is invoked
    Then the process exits with code 2
    And stderr contains "MinimizerNonConvergence" with iterations = 5

  @rq-901c006f
  Scenario: Zero force at initial positions triggers ForceZero convergence
    Given a system at the exact minimum of the potential (F_max == 0 at x_0)
    When the minimization phase is invoked
    Then the phase terminates after iteration 0 with reason ForceZero
    And no SD step is taken

  # --- Output ---

  @rq-b977777a
  Scenario: .minlog header line
    Given a [[minimization]] phase with output.minlog_every = 1
    When the phase runs to convergence
    Then the .minlog file's first line is exactly "iter,energy,max_force,step,accepted\n"

  @rq-85e9b9da
  Scenario: .minlog rows include the final convergence iteration
    Given output.minlog_every = 10 and the phase converges at iteration 47
    When the phase runs
    Then the .minlog contains rows at iter ∈ {0, 10, 20, 30, 40, 47}

  @rq-c4aea411
  Scenario: .minlog reports accepted=0 for rejected iterations
    Given a phase in which iteration 1 is rejected and iteration 2 is accepted
    When the .minlog is read
    Then row iter=1 has accepted=0 and energy equal to row iter=0's energy
    And row iter=2 has accepted=1

  @rq-9d165c7f
  Scenario: minlog_every=0 disables the file
    Given a [[minimization]] phase with output.minlog_every = 0
    When the phase runs
    Then no .minlog file is created

  @rq-d87be0ca
  Scenario: trajectory frames are written at the configured cadence and at convergence
    Given a [[minimization]] phase with output.trajectory_every = 5 and the phase converges at iteration 12
    When the phase runs
    Then the trajectory file contains frames at Step ∈ {0, 5, 10, 12}

  @rq-49e26627
  Scenario: trajectory frames do not include velocity columns even when velocities are present
    Given a [[minimization]] phase with output.trajectory_every = 1
    And buffers carrying non-zero velocities at phase entry
    When the phase writes any trajectory frame
    Then the frame's Properties string does not contain "velo:R:3"

  # --- Constraints ---

  @rq-13e42f65
  Scenario: SD with SETTLE projects every trial onto the rigid-water manifold
    Given a system with one SPC/E water (atoms [0,1,2]) and a SETTLE constraint group
    And [minimization.algorithm] kind="steepest-descent"
    When one minimizer.step is invoked
    Then |positions[1] - positions[0]| equals r_oh within relative tolerance 1.0e-6
    And |positions[2] - positions[0]| equals r_oh within relative tolerance 1.0e-6
    And |positions[2] - positions[1]| equals r_hh within relative tolerance 1.0e-6

  @rq-881bb997
  Scenario: SD with a constraint slot but no projection support is rejected at config load
    Given a hypothetical constraint algorithm "fictional-cluster" whose
      ConstraintBuilder::supports_position_projection_only returns false
    And a config combining a [[minimization]] phase with a [[constraint_types]] entry of that kind
    When load_config(path) is called
    Then it returns Err(ConfigError::IncompatibleConstraint { integrator: "steepest-descent", phase: "min" })
    (Reuses the IncompatibleConstraint variant from the integrator-constraint compatibility check;
     phase carries the minimization phase's name.)

  @rq-5e075019
  Scenario: SD with no constraint slot does not call any projection hook
    Given a config with no [[constraint_types]] / no topology [constraints] section
    And a recording wrapper that timestamps every Constraint hook call
    When the minimization phase runs
    Then no constraint hooks are recorded

  # --- Empty state ---

  @rq-0165303c
  Scenario: SD on empty buffers converges immediately
    Given a ParticleBuffers with particle_count == 0
    When the minimization phase runs
    Then the phase terminates after iteration 0 with reason ForceZero
    And the .minlog contains exactly the step-0 row
    And no SD kernel launches are recorded

  # --- Determinism ---

  @rq-e92c18c6
  Scenario: Two SD runs on the same GPU with identical inputs are byte-identical
    Given two minimization phases constructed from byte-identical configs and init states (16 SPC/E waters)
    When each runs to convergence
    Then every f32 array of run A is byte-identical to run B
    And the .minlog files are byte-identical

  # --- Registry ---

  @rq-0b7acd88
  Scenario: MinimizerRegistry::with_builtins exposes the registered minimizer
    Given a MinimizerRegistry::with_builtins()
    Then the registry contains a builder whose kind_name() is "steepest-descent"

  @rq-3e0a3040
  Scenario: Unknown minimizer kind in a populated registry reports UnknownKind
    Given a MinimizerRegistry::with_builtins()
    And a SlotConfig { kind: "fancy-gradient", params: { } }
    When registry.build(&slot, &gpu, particle_count=4, n_constraints=0) is called
    Then it returns Err(MinimizerError::UnknownKind("fancy-gradient"))

  @rq-09c8a503
  Scenario: Custom minimizer builder is selectable
    Given a MinimizerRegistry::with_builtins()
    And a custom MinimizerBuilder whose kind_name() is "test-stub"
    When registry.register(custom_builder) is called
    Then registry.build(...) routes "test-stub" kind requests to the custom builder
```
