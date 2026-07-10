# Feature: RESPA Multiple-Timestep Integrator <!-- rq-564d914c -->

The RESPA integrator (reversible reference-system propagator algorithm,
Tuckerman-Berne-Martyna, *J. Chem. Phys.* **97**, 1990 (1992)) is a
two-level impulse-splitting multiple-timestep integrator. One of the
pluggable integrator slots (see `framework.md`); selected by
`kind = "respa"` in the config's `[integrator]` section.

RESPA advances the system by an outer timestep `Δt` composed of
`ν = n_inner` velocity-Verlet inner steps of `δt = Δt / ν`. Forces are
split by the force-class machinery in `rqm/forces/framework.md`:
fast-class slots (`frequency_class() == ForceClass::Fast`: short-range
pair forces, bonded terms, SPME real-space) are re-evaluated every
inner step; slow-class slots (`ForceClass::Slow`: SPME reciprocal
space) are re-evaluated once per outer step and applied as half-impulse
kicks at the outer boundaries. The integrator carries no RNG state and
produces byte-identical trajectories across runs on the same GPU.

## Algorithm <!-- rq-4755aa0a -->

The impulse (Trotter) splitting for one outer step:

```text
v ← v + (F_slow / m) · Δt/2          # outer half-kick, slow forces
repeat ν times:
    v ← v + (F_fast / m) · δt/2      # inner half-kick, fast forces
    x ← x + v · δt                   # inner drift
    F_fast ← evaluate fast-class slots
    v ← v + (F_fast / m) · δt/2      # inner half-kick, fast forces
F_slow ← evaluate slow-class slots
v ← v + (F_slow / m) · Δt/2          # outer half-kick, slow forces
```

The kicks consume per-class forces, never the combined total: the
inner kicks read the fast-class accumulator and the outer kicks the
slow-class accumulator (`KickSource::Class`, dispatched by the runner
— see `framework.md`, *SubStep*). Every kick and drift is a one-thread-
per-particle kernel with no reductions, preserving the engine's
determinism guarantee.

The first sub-step of an outer step consumes the slow forces cached by
the previous outer step's slow evaluation (or by the runner's warm-up
force evaluation on the first step), following the framework's
symplectic-with-cached-F contract. Both force classes are evaluated at
the same (final) positions within each outer step, so combined
potential energy and virial read from a log step are position-
consistent.

## Plan shape <!-- rq-0af322d7 -->

`plan(Δt)` returns the inner loop unrolled — a flat `StepPlan` of
`3ν + 3` sub-steps whose shape depends only on `Δt` and the
construction-time `n_inner`:

```text
KickHalf  { dt: Δt, source: Class(Slow), label: "respa_outer_kick" }
ν × [
  KickDrift { dt: δt, source: Class(Fast), label: "respa_inner_kick_drift" }
  ForceEval { class: Some(Fast), level: None }
  KickHalf  { dt: δt, source: Class(Fast), label: "respa_inner_kick" }
]
ForceEval { class: Some(Slow), level: None }
KickHalf  { dt: Δt, source: Class(Slow), label: "respa_outer_kick" }
```

(`KickHalf { dt }` kicks with `dt/2` and `KickDrift { dt }` kicks with
`dt/2` and drifts with `dt`, per the framework's sub-step
conventions.)

The plan contains no `ThermostatHalf` sub-steps: a configured
thermostat is applied by the runner's default wrapping topology, once
per outer step at the outer boundaries (`apply_pre` before the plan
walk with `dt = Δt`, `apply_post` after). This is the XO-RESPA
coupling; inner-loop thermostatting (XI-RESPA) is a plan-shape change
an implementation may adopt later via `ThermostatHalf` markers without
framework changes.

Because the plan is flat and its shape is static, RESPA phases are
CUDA-graph eligible (see `cuda-graphs.md`); one captured graph spans
one outer step.

## Parameters <!-- rq-9f8480ae -->

The `[integrator]` table with `kind = "respa"` accepts:

- `n_inner: u32` — the number of inner steps `ν` per outer step.
  Required. Must satisfy `n_inner >= 1`; `n_inner == 0` is rejected at
  config-load time with `ConfigError::InvalidValue`. `n_inner == 1` is
  valid and degenerates to velocity Verlet with class-split kicks.

The outer timestep `Δt` is the simulation-wide `[simulation].dt`; the
inner timestep is derived as `δt = Δt / n_inner` and is not
independently configurable.

## Compatibility <!-- rq-cb480c95 -->

- **Thermostats:** any thermostat slot may be paired with RESPA; it
  couples at the outer boundaries (see *Plan shape*).
- **Constraints:** rejected. `RespaBuilder::supports_constraints`
  returns `false`, so a config carrying both `kind = "respa"` and a
  `[constraints]` section fails validation with the framework's
  standard integrator/constraint incompatibility error. Correct
  constrained RESPA requires RATTLE after every inner velocity update,
  which the constraint hook contract does not express.
- **Barostats:** rejected. A config carrying both `kind = "respa"`
  and a `[barostat]` section fails config validation with
  `ConfigError::InvalidValue` naming the incompatibility. RESPA-NPT
  splittings are a separate design.
- **Lossless mode:** rejected. `[integrator].lossless = true` with
  `kind = "respa"` fails config validation with
  `ConfigError::InvalidValue`; compensated inner-kick accumulation is
  not implemented.

## Neighbor-list interaction <!-- rq-9b71abcb -->

Each inner `ForceEval { class: Some(Fast) }` goes through
`ForceField::step_class`, which on the eager path runs the standard
neighbor-list displacement check and conditional rebuild before the
force launch (`rqm/forces/neighbor-list.md`). On the CUDA-graph path
the displacement check runs once per outer step, at the captured-graph
boundary, so particles drift for up to one full outer step between
checks — the same exposure as a velocity-Verlet run with timestep
`Δt`. The standard `r_skin` sizing guidance therefore applies with the
outer timestep as the reference interval.

## Post-force composed kernel <!-- rq-d46f59c6 -->

RESPA participates in the JIT-composed post-force per-particle kernel
(`jit-composed-post-force.md`). Its fragment performs the trailing
outer half-kick, `v ← v + (F_slow / m) · Δt/2`, reading the slow-class
accumulator buffers bound by `bind_post_force_per_particle_args`. The
default `post_force_substep_index` resolution (last `KickHalf` /
`KickDrift` in the plan) selects exactly this sub-step, so the runner
skips it in the plan walk when the composed kernel is active.

## Empty State and degenerate cases <!-- rq-ca983738 -->

- `particle_count == 0`: every sub-step is a no-op; one outer step
  completes with `Ok(())`.
- No slow-class slots configured: `ForceEval { class: Some(Slow) }`
  is the forces framework's no-op path and the slow accumulator holds
  exactly `0.0` for every particle, so the outer kicks add `0.0` to
  every velocity component. The outer step reduces to `ν` inner
  velocity-Verlet steps of `δt` over the fast-class forces.
- No fast-class slots configured: the inner loop reduces to pure
  drift; the slow impulses at the outer boundaries are the only
  velocity updates.
- `n_inner == 1`: one inner step per outer step; the splitting is
  algebraically velocity Verlet with the kick separated into fast and
  slow halves.

## Determinism <!-- rq-3c582d1d -->

All sub-steps are one-thread-per-particle kernels or force-pipeline
evaluations covered by the engine's reproducibility invariants; the
plan shape is a pure function of `Δt` and `n_inner`. Two runs with
identical inputs on the same GPU produce byte-identical trajectories.

## Feature API <!-- rq-9ebb8531 -->

### Types <!-- rq-dd36c79b -->

- `RespaIntegrator` — implements the `Integrator` trait declared in <!-- rq-6922b0be -->
  `framework.md`. Registered in `IntegratorRegistry::with_builtins`
  under `kind_name() == "respa"`. Fields:

  - `n_inner: u32` — copied from the config, `>= 1`.
  - The type holds no per-step mutable state beyond what
    `PostForcePerParticle` binding requires; `plan` and
    `post_force_substep_index` are pure.

  `plan(dt)` returns the shape in *Plan shape*. `execute` receives no
  sub-steps in normal operation — every RESPA sub-step is either a
  `ForceEval`, a `Class`-sourced kick, or skipped in favour of the
  composed post-force kernel, all of which the runner dispatches — and
  returns `IntegratorError::UnexpectedSubStep` for anything it is
  handed.

- `RespaBuilder` — implements `IntegratorBuilder` with <!-- rq-1c2c92b7 -->
  `kind_name() == "respa"`.
  - `build(gpu, particle_count, n_constraints, params)` deserialises
    `RespaParams { n_inner: u32 }` from `params` and constructs the
    integrator.
  - `supports_constraints(&params)` returns `false`.
  - Parameter validation rejects `n_inner == 0`.

### Config validation <!-- rq-abb4cdfe -->

- `kind = "respa"` together with a `[barostat]` section is rejected at <!-- rq-441de6c1 -->
  config-load time with `ConfigError::InvalidValue` on field
  `barostat.kind`, reason naming the RESPA incompatibility.
- `kind = "respa"` together with `[integrator].lossless = true` is <!-- rq-64a7b057 -->
  rejected at config-load time with `ConfigError::InvalidValue` on
  field `integrator.lossless`.
- `kind = "respa"` together with a `[constraints]` section is rejected <!-- rq-406da2fc -->
  by the framework's integrator/constraint compatibility check
  (`framework.md`, *Compatibility Rules*).

## Gherkin Scenarios <!-- rq-d7e882ba -->

```gherkin
Feature: RESPA multiple-timestep integrator

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called

  # --- Construction and plan shape ---

  @rq-b141648c
  Scenario: Registry builds a RESPA integrator
    Given an integrator kind "respa" with n_inner = 4
    When registry.build(kind, device, particle_count, n_constraints = 0) is called
    Then it returns Ok(integrator)

  @rq-204ec51c
  Scenario: Plan is the unrolled inner loop
    Given a RESPA integrator with n_inner = 4
    When plan(dt) is called
    Then the plan has 15 sub-steps
    And sub-step 0 is KickHalf { dt, source: Class(Slow) }
    And sub-steps 1..=12 are 4 repetitions of
      [KickDrift { dt/4, source: Class(Fast) },
       ForceEval { class: Some(Fast) },
       KickHalf { dt/4, source: Class(Fast) }]
    And sub-step 13 is ForceEval { class: Some(Slow) }
    And sub-step 14 is KickHalf { dt, source: Class(Slow) }

  @rq-5c63aa07
  Scenario: Plan shape is a pure function of dt
    Given a RESPA integrator with n_inner = 2
    When plan(dt) is called twice with the same dt
    Then both plans have identical sub-step sequences

  @rq-b0a520ac
  Scenario: Plan contains no ThermostatHalf markers
    Given a RESPA integrator with n_inner = 2
    Then plan(dt).has_thermostat_points() is false

  @rq-482921eb
  Scenario: post_force_substep_index selects the trailing outer kick
    Given a RESPA integrator with n_inner = 4
    Then post_force_substep_index(dt) is Some(14)

  # --- Config validation ---

  @rq-d62c351e
  Scenario: Reject n_inner = 0
    Given a config with [integrator] kind = "respa" and n_inner = 0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "integrator.n_inner", reason: _ })

  @rq-d9a51e75
  Scenario: Accept n_inner = 1
    Given a config with [integrator] kind = "respa" and n_inner = 1
    When load_config is called
    Then it returns Ok(config)

  @rq-7e9e52c4
  Scenario: Reject RESPA with a constraints section
    Given a config with [integrator] kind = "respa" and a [constraints] section
    When runner validation runs
    Then it fails with the integrator/constraint incompatibility error for kind "respa"

  @rq-4bf89376
  Scenario: Reject RESPA with a barostat
    Given a config with [integrator] kind = "respa" and [barostat] kind = "mc"
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "barostat.kind", reason: _ })

  @rq-ecd52ccb
  Scenario: Reject RESPA with lossless mode
    Given a config with [integrator] kind = "respa" and lossless = true
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "integrator.lossless", reason: _ })

  # --- Dynamics ---

  @rq-19504e09
  Scenario: Fast-only system matches velocity Verlet at the inner timestep
    Given an LJ-only system (no slow-class slots) with N = 8
    And a RESPA run with n_inner = 2 and dt = 2·δt for 5 outer steps
    And a velocity-Verlet run of the same initial state with dt = δt for 10 steps
    When both runs complete
    Then every position agrees between the two runs within relative tolerance 1e-5
    And every velocity agrees within the same tolerance

  @rq-bcec000b
  Scenario: Slow-only system applies pure outer impulses
    Given a system whose only potential is a registered test slot with frequency_class Slow producing constant force F on particle 0
    And a RESPA integrator with n_inner = 2, outer timestep dt, particle 0 at rest
    When one outer step runs
    Then particle 0's velocity is (F/m) · dt within f32 round-off
    And its position advanced by (F/m) · (dt/2) · dt within f32 round-off

  @rq-860c8ff9
  Scenario: NVE energy is conserved to the same order as velocity Verlet
    Given an LJ + SPME system with N = 64 at equilibrium density
    And a RESPA run with n_inner = 4 over 200 outer steps in NVE
    When the run completes
    Then the relative drift of total energy (KE + PE) is below 1e-3

  @rq-c8835814
  Scenario: Two identical RESPA runs are byte-identical
    Given a config with kind = "respa", n_inner = 4, fixed seed inputs
    When the same run executes twice on the same GPU
    Then the trajectory files are byte-identical

  @rq-c3746bb2
  Scenario: Thermostat couples once per outer step
    Given a recording thermostat and a RESPA integrator with n_inner = 4
    When the runner executes one outer step
    Then the thermostat records exactly one apply_pre and one apply_post
    And both receive dt equal to the outer timestep

  # --- Degenerate cases ---

  @rq-f746a29f
  Scenario: RESPA with no slow-class slots leaves outer kicks as exact no-ops
    Given an LJ-only ForceField and a RESPA integrator with n_inner = 2
    And a particle with velocity v before the outer kick sub-step
    When the outer kick executes against the zeroed slow accumulator
    Then the particle's velocity is bit-identical to v

  @rq-64893fbb
  Scenario: RESPA with N = 0 completes a step
    Given a config with kind = "respa" and an init file with N = 0
    When heddlemd run is invoked
    Then it exits with code 0
```
