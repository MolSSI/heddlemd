# Feature: End-to-End Pipeline Reproducibility Test <!-- rq-72157184 -->

The project's marquee invariant is bit-wise reproducibility: identical
inputs produce byte-identical outputs across runs on the same GPU.
Per-kernel reproducibility tests already cover the integration, reduction,
and Lennard-Jones pair-force kernels in isolation. This feature adds a
single integration test that exercises the full velocity-Verlet pipeline
end-to-end and confirms reproducibility through composition: two
independently-constructed pipelines run side-by-side over many timesteps,
and every field of the resulting `ParticleState` agrees byte-for-byte
across the two runs.

The test lives in `tests/pipeline_reproducibility.rs`. It introduces no
new public types or functions; the velocity-Verlet loop is inlined in the
test using the existing `jit_composed_pair_force`,
`vv_kick_drift`, and `vv_kick` launchers.

## Test Fixture <!-- rq-8dfac0eb -->

The test runs a 4×4×4 simple-cubic Lennard-Jones fluid with a small
deterministic perturbation that breaks symmetry so the system actually
evolves.

- Particle count: `N = 64`.
- Simulation box: orthorhombic `lx = ly = lz = 8.0`.
- Lattice positions: particle `(ix, iy, iz)` with `i = ix*16 + iy*4 + iz`
  is placed at `(ix * 2.0 - 3.0, iy * 2.0 - 3.0, iz * 2.0 - 3.0)`.
- Position perturbation: each component is offset by
  `0.2 * sin(i * 0.7 + axis_phase)` with a distinct phase per axis
  (e.g. `0.0` for x, `1.1` for y, `2.3` for z) so no axis is symmetric
  with another. The amplitude is large enough to push the system well
  out of the lattice ground state so that visible motion develops within
  the 100-step run.
- Initial velocities: all zero.
- Initial forces: zero (the constructor allocates them; the first
  force-evaluation pass overwrites them before the loop starts).
- Particle masses: all `1.0`.
- Particle IDs: `0..N` (default).
- Lennard-Jones parameters: `sigma = 1.0`, `epsilon = 1.0`,
  `cutoff = 2.5`.
- Timestep: `dt = 0.001`.
- Neighbour list: the packed neighbour list (`interacting_tiles`,
  `interacting_atoms`, `interaction_count`) built by `NeighborListState`
  during the pipeline. It is sized automatically by the rebuild; the test
  drives the production pipeline rather than hand-building any list.

## Velocity-Verlet Loop <!-- rq-6b6180af -->

A single timestep follows the standard kick-drift / force / kick pattern:

```
vv_kick_drift(particle_buffers, dt)
jit_composed_pair_force(particle_buffers, output, sim_box, params, ...)
vv_kick(particle_buffers, dt)
```

Before the first timestep, `jit_composed_pair_force` are
invoked once to populate `forces_*` with `F(0)`. This warm-up is required
because `vv_kick_drift` consumes the current force state.

## Comparison Procedure <!-- rq-24a2b5ef -->

After the loop completes on a given run, the test downloads
`ParticleState` from the device via `state.download_from(&buffers)` and
captures every field. The two runs' downloaded states are compared using
exact `assert_eq!` on each `Vec<f32>` and `Vec<u32>`. NaN / Inf are not
present in the fixture (zero initial velocities, finite forces); their
appearance would indicate a regression and is checked separately.

## Out of Scope <!-- rq-9c94f23b -->

- New public types or launchers; this feature is a test only.
- A `simulation_step` orchestration function (a future feature in
  `src/simulation.rs`).
- Energy conservation, temperature, or other physical-correctness
  diagnostics. The test is about bit-wise reproducibility, not physics
  validation.
- Cross-hardware reproducibility. The architecture explicitly limits the
  guarantee to runs on the same GPU.
- Trajectory I/O.

---

## Gherkin Scenarios <!-- rq-5ece2ef9 -->

```gherkin
Feature: End-to-end pipeline reproducibility

  Background:
    Given the test fixture defined above (N=64 LJ fluid, 4×4×4 perturbed lattice, dt=0.001)
    And two independent pipeline instances A and B, each consisting of fresh ParticleBuffers and a fresh SlotOutputView constructed from byte-identical ParticleState inputs, each owning the packed neighbour list built by its NeighborListState

  @rq-b2314952
  Scenario: Bit-exact equality after a single full velocity-Verlet step
    Given each pipeline has been warmed up with one jit_composed_pair_force pass
    When one full step (vv_kick_drift, jit_composed_pair_force, vv_kick) is executed on each pipeline
    And each pipeline's ParticleBuffers is downloaded into a host ParticleState
    Then state_A.positions_x equals state_B.positions_x byte-for-byte
    And state_A.positions_y equals state_B.positions_y byte-for-byte
    And state_A.positions_z equals state_B.positions_z byte-for-byte
    And state_A.velocities_x equals state_B.velocities_x byte-for-byte
    And state_A.velocities_y equals state_B.velocities_y byte-for-byte
    And state_A.velocities_z equals state_B.velocities_z byte-for-byte
    And state_A.forces_x equals state_B.forces_x byte-for-byte
    And state_A.forces_y equals state_B.forces_y byte-for-byte
    And state_A.forces_z equals state_B.forces_z byte-for-byte
    And state_A.masses equals state_B.masses byte-for-byte
    And state_A.particle_ids equals state_B.particle_ids byte-for-byte

  @rq-2846ee8b
  Scenario: Bit-exact equality after a 100-step run
    Given each pipeline has been warmed up with one jit_composed_pair_force pass
    When 100 full velocity-Verlet steps are executed on each pipeline
    And each pipeline's ParticleBuffers is downloaded into a host ParticleState
    Then state_A.positions_x equals state_B.positions_x byte-for-byte
    And state_A.positions_y equals state_B.positions_y byte-for-byte
    And state_A.positions_z equals state_B.positions_z byte-for-byte
    And state_A.velocities_x equals state_B.velocities_x byte-for-byte
    And state_A.velocities_y equals state_B.velocities_y byte-for-byte
    And state_A.velocities_z equals state_B.velocities_z byte-for-byte
    And state_A.forces_x equals state_B.forces_x byte-for-byte
    And state_A.forces_y equals state_B.forces_y byte-for-byte
    And state_A.forces_z equals state_B.forces_z byte-for-byte
    And state_A.masses equals state_B.masses byte-for-byte
    And state_A.particle_ids equals state_B.particle_ids byte-for-byte

  @rq-d0a54b3c
  Scenario: Positions visibly evolve over the 100-step run
    Given a snapshot of the initial host positions before the loop
    When 100 full velocity-Verlet steps are executed on pipeline A
    And pipeline A's ParticleBuffers is downloaded into a host ParticleState
    Then for at least one particle index i, the displacement
         sqrt((final.positions_x[i] - initial.positions_x[i])^2 +
              (final.positions_y[i] - initial.positions_y[i])^2 +
              (final.positions_z[i] - initial.positions_z[i])^2)
         is greater than 0.001

  @rq-3f46fb2e
  Scenario: All output values are finite after the 100-step run
    When 100 full velocity-Verlet steps are executed on pipeline A
    And pipeline A's ParticleBuffers is downloaded into a host ParticleState
    Then every element of positions_x, positions_y, positions_z is finite
    And every element of velocities_x, velocities_y, velocities_z is finite
    And every element of forces_x, forces_y, forces_z is finite
```
