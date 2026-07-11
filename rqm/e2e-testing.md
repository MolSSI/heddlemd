# Feature: End-to-End Test Harness and Coverage <!-- rq-99dd6bf8 -->

Several of the engine's correctness properties appear only when a complete
simulation runs many timesteps through the production runner: bit-wise
reproducibility, energy conservation, thermostat and barostat control, and the
correct composition of the integrator, thermostat, barostat, and constraint
slots in the post-force tail. Per-kernel and per-slot unit tests cannot observe
these properties, because they exercise components in isolation against
synthetic force fields and never run the runner's ordering, CUDA-graph capture,
global reductions, or stream sequencing.

The end-to-end (e2e) test layer drives full simulations through
`run_simulation` (see `simulation-runner.md`) from generated input files and
asserts these properties. It is built on a shared harness that constructs
input systems, runs them, and reads back physics, so that each e2e test is a
short, declarative statement of a physical or compositional invariant rather
than a re-implementation of file I/O and configuration boilerplate.

The e2e tests live in `tests/e2e.rs`; the harness lives in the shared test
module `tests/common/`.

## Test Harness <!-- rq-8d45ae10 -->

The harness has three parts: a working-directory factory, a system builder that
writes a runnable input set, and a set of assertions over the results of a run.

### Types <!-- rq-a5e72a70 -->

- `Case` — an isolated working directory for one e2e test. `Case::new(name)`
  creates a unique directory under the system temp directory whose name
  incorporates `name`, the process id, and a monotonic nonce, so that
  concurrently-running tests never collide. The directory is created empty; the
  `Case` removes it when dropped.
  - `Case::dir(&self) -> &Path` — the working directory.
  - `Case::config_path(&self) -> PathBuf` — the path `dir()/sim.in.toml`
    (whether or not it has been written yet).

- `SystemBuilder` — a builder that writes a matched extended-XYZ initial-state
  file and TOML configuration into a `Case`. It is constructed from a physical
  preset and then configured with ensemble and run parameters.
  - Presets (associated constructors):
    - `SystemBuilder::argon_lattice(n_per_side)` — a simple-cubic
      Lennard-Jones argon lattice with a small deterministic symmetry-breaking
      perturbation.
    - `SystemBuilder::disordered_lj_liquid(n_per_side, spacing)` — a
      Lennard-Jones fluid on a lattice with a deterministic pseudo-random
      displacement per particle, breaking all lattice symmetry.
    - `SystemBuilder::spce_water(n_molecules)` — SPC/E water molecules, each
      carrying the intramolecular topology and rigid-geometry constraint
      metadata needed to drive SETTLE or SHAKE.
    - `SystemBuilder::ionic_lattice(n_per_side)` — a simple-cubic lattice of
      alternating cations (`+1 e`) and anions (`−1 e`), two particle types with
      equal and opposite charge and a Lennard-Jones core to prevent collapse.
      The total charge is zero (the preset requires an even total particle
      count so the two species have equal population). The lattice spacing and
      side are sized so the periodic box exceeds the cell-list minimum width,
      which SPME requires. This is the charged preset for SPME runs.
  - Ensemble and run configuration (chainable, each returning `Self`):
    - `.integrator(kind)` — selects the integrator (default velocity-Verlet).
    - `.thermostat(kind, target_temperature)` — installs an external
      thermostat slot at the target temperature.
    - `.barostat(kind, target_pressure)` — installs a barostat slot at the
      target pressure.
    - `.constraints(kind)` — installs a constraint slot (`settle`, `shake`,
      `rattle`, or `geometric`) over the preset's constraint metadata.
    - `.with_spme()` — enables smooth particle-mesh Ewald electrostatics by
      emitting an `[spme]` table (see `io/config-schema.md`). It selects the
      Ewald splitting parameter, real-space cutoff, FFT grid, and spline order
      from the preset's box and cutoff (a grid of roughly one point per
      ångström of box width, `alpha ≈ 3.5 / r_cut_real`, spline order 4). SPME
      requires the cell-list neighbour pipeline, so a builder with SPME enabled
      leaves the neighbour mode at its cell-list default. Charges come from the
      particle types, so `.with_spme()` is meaningful only on a charged preset
      (`ionic_lattice`).
    - `.units(system)` — selects `atomic` (default) or `si` for the written
      input and output files.
    - `.dt(dt)`, `.n_steps(n)`, `.log_every(k)`, `.trajectory_every(k)`,
      `.seed(s)` — timestep, step count, output cadences, and the run's RNG
      seed.
  - `.write(&mut self, case: &Case) -> PathBuf` — writes `sim.in.xyz` and
    `sim.in.toml` into `case.dir()` and returns the config path. A builder whose
    configured slot combination the runner would reject (see the compatibility
    guards in `integration/framework.md`) is a programming error in the test,
    not a runtime path this harness is expected to exercise.

### Functions <!-- rq-e9ed7da2 -->

- `run_case(config_path: &Path) -> RunSummary` — a thin wrapper over
  `run_simulation` that unwraps the result and attaches the config path to any
  error message. Returns the `RunSummary`, whose per-phase physics series
  (`simulation-runner.md`) the assertions below consume.

- `assert_runs_reproducible(builder: &SystemBuilder, n_runs: usize)` — writes
  and runs the same system `n_runs` times, each into its own fresh `Case`, and
  asserts that the trajectory and log output files are byte-for-byte identical
  across all runs. This is the direct expression of the reproducibility
  invariant (`pipeline-reproducibility.md`): identical inputs produce identical
  output bytes on the same GPU.

- `assert_energy_drift_bounded(phase: &PhaseSummary, max_abs_slope: f64)` —
  fits a line to the phase's `total_energy` physics samples against time and
  asserts the absolute slope is at most `max_abs_slope` (in Hartree per atomic
  time unit). Used for NVE energy-conservation checks.

- `assert_mean_temperature_near(phase: &PhaseSummary, target: f64, rel_tol: f64)`
  — asserts the mean of the phase's `temperature` samples, taken over the
  samples after an initial equilibration fraction, is within `rel_tol` of
  `target`.

- `assert_mean_pressure_near(phase: &PhaseSummary, target: f64, rel_tol: f64)` —
  the pressure analogue of the temperature check, over the phase's `pressure`
  samples.

- `read_last_frame(trajectory_path: &Path) -> ParticleState` — parses the final
  frame of a written trajectory file into a `ParticleState`.

- `assert_on_constraint_manifold(state: &ParticleState, topology: &Topology, rel_tol: f64)`
  — asserts every constrained bond length in `state` equals its reference length
  within `rel_tol`, so that a constrained trajectory is checked to remain on the
  position manifold at its end (not merely after a single projection).

## Physics Readout <!-- rq-40e46b58 -->

The assertions above read physics from the `RunSummary` returned by
`run_simulation`. Each `PhaseSummary` carries a `physics: Vec<PhysicsSample>`
series captured at the phase's log cadence; the fields of `PhysicsSample` and
`PhaseSummary` are defined in `simulation-runner.md`. The samples are computed
by the same per-log-row evaluation that writes the CSV log, so the series is
populated exactly when the phase's `log_every` is nonzero and adds no force
evaluations beyond those logging already performs.

## Coverage <!-- rq-86d56532 -->

The e2e layer exercises the following, each through `run_simulation` over a
multi-step run with real slots and a real force field.

### Slot composition (post-force tail) <!-- rq-9213dcb5 -->

The integrator, thermostat, barostat, and constraint slots interleave in the
post-force tail, and their ordering is load-bearing (see
`integration/constraint-framework.md`, RATTLE-last). The e2e layer covers the
combinations where a per-particle position or velocity update from one slot
feeds another:

- velocity-Verlet + a rigid-water constraint + a per-step barostat;
- velocity-Verlet + a thermostat + a per-step barostat + a constraint (all
  four slots active at once).

### Reproducibility <!-- rq-ff8b6a9a -->

Bit-wise reproducibility is covered for the configurations whose full-pipeline
determinism is not guaranteed by per-kernel tests alone — in particular the
RNG-driven and FFT-driven paths, where warp-level randomness, multi-block
reduction order, and stream sequencing are determinism hazards:

- a stochastic velocity-rescale thermostat (CSVR);
- a stochastic velocity-randomizing thermostat (Andersen);
- a stochastic barostat (C-rescale);
- SPME long-range electrostatics;
- a constrained system (SETTLE) driven through the full runner.

### Energy conservation <!-- rq-f29e3f73 -->

- a microcanonical (NVE) velocity-Verlet run conserves total energy: the
  energy-drift slope over the trajectory is bounded;
- a microcanonical (NVE) run of a charged ionic lattice with SPME
  electrostatics active conserves total energy over the trajectory, giving the
  full SPME pipeline (real-space screening plus the reciprocal spread / FFT /
  influence / IFFT / gather path) an end-to-end physics check rather than only
  the component-level comparison against an Ewald reference.

### Pressure control <!-- rq-bb76b520 -->

- a constant-pressure (NPT) run with a per-step barostat drives the mean
  pressure to the configured target, and the box volume responds (differs from
  its initial value).
- a constant-pressure (NPT) run with SPME electrostatics active runs stably and
  the box volume responds to the barostat, exercising the reciprocal pipeline
  under a changing box: the influence function is recomputed every step from the
  live lattice and the reciprocal-space virial feeds the barostat's pressure, so
  the long-range electrostatics track the box the barostat mutates.

## Out of Scope <!-- rq-77a0c4d3 -->

- **Per-kernel and per-slot unit tests.** Kinematics of individual kernels,
  single-application slot behaviour, parser acceptance/rejection, and
  neighbour-list construction are covered by their own suites, not the e2e
  layer.
- **Cross-hardware reproducibility.** The reproducibility guarantee is limited
  to runs on the same GPU (`docs/architecture.md`).
- **Restart / continue-from-state.** The engine has no restart feature; there is
  no state to round-trip beyond trajectory-format write/read-back, which its own
  I/O suite covers.
- **The following combinations and invariants are not part of this e2e layer:**
  the Nose-Hoover-chain, Andersen, and Berendsen thermostats driven through the
  runner beyond the reproducibility checks above; Langevin combined with a
  per-step barostat; end-to-end time-reversibility with per-step force
  recomputation (`pipeline-reversibility.md` covers the lossless round trip); a
  SHAKE-specific constrained-trajectory run; and an SI-versus-atomic unit-system
  equivalence run.
- **Combinations the runner forbids.** The compatibility guards
  (`integration/framework.md`) reject, and the config suite already covers the
  rejection of: Langevin with an external thermostat or constraints; RESPA with
  a barostat or constraints; MTK-NPT with an external thermostat, barostat, or
  constraints; and lossless velocity-Verlet with constraints.

---

## Gherkin Scenarios <!-- rq-ef58789c -->

```gherkin
Feature: End-to-end test harness

  @rq-0a87253a
  Scenario: A Case creates a unique, empty working directory
    Given two Cases created with the same name
    Then their directories are distinct paths
    And each directory exists and is empty

  @rq-1fdcfc84
  Scenario: SystemBuilder writes a runnable input set
    Given a SystemBuilder from the argon_lattice preset with dt and n_steps set
    And a fresh Case
    When the builder is written to the Case
    Then the files sim.in.xyz and sim.in.toml exist in the Case directory
    And run_simulation on the returned config path succeeds

Feature: End-to-end slot composition

  @rq-0baf3195
  Scenario: A constrained water system with a per-step barostat stays on the manifold
    Given a SystemBuilder from the spce_water preset with SETTLE constraints and a per-step c-rescale barostat
    And a run of 200 steps with trajectory output enabled
    When the simulation is run
    Then the last trajectory frame lies on the constraint manifold within relative tolerance 1e-4

  @rq-75dc5b88
  Scenario: A four-slot run (thermostat, barostat, constraint) completes and stays on the manifold
    Given a SystemBuilder from the spce_water preset with a CSVR thermostat, a per-step c-rescale barostat, and SETTLE constraints
    And a run of 200 steps with trajectory output enabled
    When the simulation is run
    Then the run completes without error
    And the last trajectory frame lies on the constraint manifold within relative tolerance 1e-4

Feature: End-to-end reproducibility of stochastic and long-range paths

  @rq-dd4240b5
  Scenario: A CSVR thermostat run is byte-identical across runs
    Given a SystemBuilder from the disordered_lj_liquid preset with a CSVR thermostat and a fixed seed
    When the same system is run 3 times into separate Cases
    Then the trajectory and log output files are byte-for-byte identical across all runs

  @rq-a9e9f039
  Scenario: An Andersen thermostat run is byte-identical across runs
    Given a SystemBuilder from the disordered_lj_liquid preset with an Andersen thermostat and a fixed seed
    When the same system is run 3 times into separate Cases
    Then the trajectory and log output files are byte-for-byte identical across all runs

  @rq-9ef78757
  Scenario: A C-rescale barostat run is byte-identical across runs
    Given a SystemBuilder from the disordered_lj_liquid preset with a per-step c-rescale barostat and a fixed seed
    When the same system is run 3 times into separate Cases
    Then the trajectory and log output files are byte-for-byte identical across all runs

  @rq-86bf63c3
  Scenario: An SPME electrostatics run is byte-identical across runs
    Given a SystemBuilder from the ionic_lattice preset with SPME enabled and a fixed seed
    When the same system is run 3 times into separate Cases
    Then the trajectory and log output files are byte-for-byte identical across all runs

  @rq-87307e49
  Scenario: A SETTLE-constrained run is byte-identical across runs
    Given a SystemBuilder from the spce_water preset with SETTLE constraints and a fixed seed
    When the same system is run 3 times into separate Cases
    Then the trajectory and log output files are byte-for-byte identical across all runs

Feature: End-to-end energy conservation

  @rq-d2cd8351
  Scenario: An NVE velocity-Verlet run conserves total energy
    Given a SystemBuilder from the argon_lattice preset with no thermostat and no barostat
    And a run long enough to accumulate many log samples
    When the simulation is run
    Then the total-energy drift slope over the phase's physics samples is bounded within the NVE tolerance

  @rq-4fcc97ab
  Scenario: An NVE run with SPME electrostatics conserves total energy
    Given a SystemBuilder from the ionic_lattice preset with SPME enabled and no thermostat and no barostat
    And a run long enough to accumulate many log samples
    When the simulation is run
    Then the total-energy drift slope over the phase's physics samples is bounded within the NVE tolerance

Feature: End-to-end pressure control

  @rq-b02a9070
  Scenario: An NPT run drives mean pressure to the target
    Given a SystemBuilder from the disordered_lj_liquid preset with a per-step barostat at a target pressure
    And a run long enough to accumulate many log samples
    When the simulation is run
    Then the mean pressure over the post-equilibration physics samples is within relative tolerance of the target

  @rq-5214e3b3
  Scenario: An NPT run's box volume responds to the barostat
    Given a SystemBuilder from the disordered_lj_liquid preset with a per-step barostat at a target pressure away from the system's initial pressure
    When the simulation is run
    Then the final physics sample's volume differs from the initial volume

  @rq-5a00037b
  Scenario: An NPT run with SPME electrostatics runs stably and the box responds
    Given a SystemBuilder from the ionic_lattice preset with SPME enabled and a per-step barostat at a target pressure away from the system's initial pressure
    When the simulation is run
    Then the run completes without error
    And the final physics sample's volume differs from the initial volume
    And every energy and pressure sample is finite
```
