# Feature: End-to-End Pipeline Reproducibility Test <!-- rq-72157184 -->

The project's marquee invariant is bit-wise reproducibility: identical
inputs produce byte-identical outputs across runs on the same GPU.
Per-kernel reproducibility tests cover the integration, reduction, and
Lennard-Jones pair-force kernels in isolation. This test confirms
reproducibility through composition of the whole production pipeline: the
same input system is run twice through `run_simulation` (see
`simulation-runner.md`), and the two runs' output files agree byte-for-byte.

The test drives the production runner from generated input files using the
shared end-to-end harness (`e2e-testing.md`), so it exercises the real
neighbour-list rebuilds, force pipeline, integrator schedule, CUDA-graph
capture, and trajectory/log writers rather than any hand-assembled loop. It
introduces no new public engine types or launchers.

Reproducibility of the RNG-driven and FFT-driven configurations — stochastic
thermostats and barostats, SPME electrostatics, and constrained systems — is
covered by the reproducibility scenarios in `e2e-testing.md`. This file
covers the base velocity-Verlet Lennard-Jones case.

## Test Fixture <!-- rq-8dfac0eb -->

The test runs a disordered Lennard-Jones fluid produced by the harness
`disordered_lj_liquid` preset: a simple-cubic lattice with a deterministic
per-particle displacement that breaks all lattice symmetry, so the system
evolves visibly over the run. The preset fixes the particle count, box,
Lennard-Jones parameters, timestep, and RNG seed, and writes a matched
extended-XYZ initial-state file and TOML configuration. Velocity-Verlet is
the integrator; no thermostat, barostat, or constraint slot is installed.
The neighbour list is the packed list built by the production pipeline during
the run.

## Run Procedure <!-- rq-6b6180af -->

Each run is one full `run_simulation` invocation on the fixture's config path,
writing its trajectory and CSV log into its own working directory (a distinct
harness `Case`). A run performs its own warm-up force evaluation, neighbour
rebuilds, and per-step schedule internally; the test does not step the
pipeline by hand.

## Comparison Procedure <!-- rq-24a2b5ef -->

After both runs complete, the test compares their written trajectory files
and CSV log files byte-for-byte. Byte-identical output files are the direct
expression of the invariant. The final frame is additionally checked to be
finite; NaN or Inf would indicate a regression.

## Out of Scope <!-- rq-9c94f23b -->

- New public engine types or launchers; this is a test only.
- Energy conservation, temperature, or other physical-correctness
  diagnostics. This test concerns bit-wise reproducibility, not physics
  validation; physics invariants live in `e2e-testing.md`.
- Reproducibility of the stochastic, long-range, and constrained
  configurations, which `e2e-testing.md` covers.
- Cross-hardware reproducibility. The architecture limits the guarantee to
  runs on the same GPU.

---

## Gherkin Scenarios <!-- rq-5ece2ef9 -->

```gherkin
Feature: End-to-end pipeline reproducibility

  Background:
    Given a SystemBuilder from the disordered_lj_liquid preset with velocity-Verlet, a fixed seed, and trajectory and log output enabled

  @rq-b2314952
  Scenario: Bit-exact output after a single-step run
    Given the system configured for a one-step run
    When the same system is run twice into separate Cases
    Then the two runs' trajectory files are byte-for-byte identical
    And the two runs' CSV log files are byte-for-byte identical

  @rq-2846ee8b
  Scenario: Bit-exact output after a 100-step run
    Given the system configured for a 100-step run
    When the same system is run twice into separate Cases
    Then the two runs' trajectory files are byte-for-byte identical
    And the two runs' CSV log files are byte-for-byte identical

  @rq-d0a54b3c
  Scenario: Positions visibly evolve over the 100-step run
    Given the system configured for a 100-step run with a trajectory frame at the first and last step
    When the system is run
    Then for at least one particle the displacement between the first and last trajectory frame exceeds 0.001

  @rq-3f46fb2e
  Scenario: All output values are finite after the 100-step run
    Given the system configured for a 100-step run
    When the system is run
    Then every position, velocity, and force component in the final trajectory frame is finite
```
