# Feature: End-to-End Pipeline Reversibility Test <!-- rq-d836d884 -->

The lossless integrator carries a stronger guarantee than ordinary
reproducibility: a forward run followed by a velocity flip and another
forward run of the same length restores the observable `f32` state
(positions and velocities) bit-exactly on the same GPU. The per-kernel
reversibility scenarios in `integration.md` exercise this property under
zero or constant forces. This feature exercises it under a *position-
dependent* force field driven by the full Lennard-Jones force pipeline,
confirming that round-trip reversibility holds end-to-end through the
pair-force kernel and the segmented reduction.

The test lives in `tests/pipeline_reversibility.rs`. It introduces no new
public types or functions; it composes the existing `lj_pair_force`,
`reduce_pair_forces`, `vv_kick_drift_lossless`, and `vv_kick_lossless`
launchers around the existing `LosslessBuffers`.

A negative companion test in the same file confirms that the lossy
integrator does **not** preserve bit-exact reversibility under the same
protocol — establishing that the lossless residual machinery is what
buys the guarantee.

## Test Fixture <!-- rq-7600786b -->

The reversibility test reuses the perturbed Lennard-Jones lattice from
the pipeline-reproducibility test so that the system actually evolves
under nontrivial forces during the forward run.

- Particle count: `N = 64`.
- Simulation box: orthorhombic `lx = ly = lz = 8.0`.
- Lattice positions: particle `(ix, iy, iz)` with `i = ix*16 + iy*4 + iz`
  is placed at `(ix * 2.0 - 3.0, iy * 2.0 - 3.0, iz * 2.0 - 3.0)`.
- Position perturbation: each component is offset by
  `0.2 * sin(i * 0.7 + axis_phase)` with axis phases `0.0` (x), `1.1`
  (y), `2.3` (z).
- Initial velocities: all zero.
- Initial forces: zero (overwritten by the warm-up force evaluation).
- Particle masses: all `1.0`.
- Particle IDs: `0..N` (default).
- Lennard-Jones parameters: `sigma = 1.0`, `epsilon = 1.0`,
  `cutoff = 2.5`.
- Timestep: `dt = 0.001`.
- `PairBuffer` shape: `particle_count = 64`, `max_neighbors = 64`.
- `neighbor_counts`: every entry equals `64`.
- `LosslessBuffers`: freshly allocated, all six residual buffers
  zero-initialised at the start of every run.
- Step count: `N_steps = 100` for the round-trip scenarios.

## Lossless Velocity-Verlet Loop <!-- rq-7b5eef8c -->

A single timestep follows the lossless kick-drift / force / kick
pattern:

```
vv_kick_drift_lossless(particle_buffers, lossless, dt)
lj_pair_force(particle_buffers, pair_buffer, sim_box, params)
reduce_pair_forces(pair_buffer, neighbor_counts, particle_buffers)
vv_kick_lossless(particle_buffers, lossless, dt)
```

Before the first timestep, `lj_pair_force` and `reduce_pair_forces` are
invoked once to populate `forces_*` with `F(x_0)`. This warm-up is
required because `vv_kick_drift_lossless` consumes the current force
state.

## Reversal Protocol <!-- rq-1d618f18 -->

The standard reversal protocol from `integration.md` applies in full:

1. Capture a snapshot of all eleven `ParticleBuffers` arrays plus all
   six `LosslessBuffers` residual arrays *after* the warm-up but
   *before* the first integrator step. This snapshot is the "initial
   state" the round-trip must restore.
2. Run the lossless velocity-Verlet loop forward for `N_steps`
   iterations. After the loop the device holds positions `x_N`,
   velocities `v_N`, and a forces buffer holding `F(x_N)` from the
   final iteration's `lj_pair_force` / `reduce_pair_forces` pass.
3. Negate every velocity component, both the `f32` high half and the
   `f64` low half: `v ← -v`, `v_lo ← -v_lo`. Forces are not negated;
   they are deterministic functions of positions and remain valid for
   the first reverse step.
4. Run the lossless velocity-Verlet loop forward for `N_steps`
   iterations. No second warm-up is required: the forces buffer
   already holds `F(x_N)`, which is what the next `vv_kick_drift_lossless`
   consumes. Each step recomputes the force via `lj_pair_force` and
   `reduce_pair_forces` at the new intermediate position, exactly as
   in the forward direction.
5. Negate every velocity component again to restore direction.

The single-step round-trip uses the same protocol with `N_steps = 1`.

The lossy negative test follows the same five-step protocol but
substitutes `vv_kick_drift` and `vv_kick` for the lossless launchers
and operates only on `ParticleBuffers` (no `LosslessBuffers`).

## Comparison Procedure <!-- rq-090b70bb -->

After the round trip completes, the test downloads `ParticleBuffers`
into a host `ParticleState` and downloads each `LosslessBuffers`
field into a host `Vec<f64>`. Comparison against the initial snapshot
splits into three classes:

- **Bit-exact `f32` and `u32` arrays.** `positions_x/y/z`,
  `velocities_x/y/z`, `forces_x/y/z`, `masses`, and `particle_ids` are
  compared with `assert_eq!`. Any drift fails the test.
- **Tolerated `f64` residual arrays.** The six `*_lo` buffers are
  compared element-wise with an absolute tolerance of `1e-10`, matching
  the tolerance used by `rq-b73316ed` in `integration.md`. Residuals
  are internal compensation bookkeeping and are permitted to drift by
  a few `f64` ULPs under round-tripping; the architecture's invariant
  is that this drift never propagates into the observable `f32` state.
- **Lossy negative comparison.** The lossy round-trip's downloaded
  `ParticleState` is compared against the snapshot using `assert_ne!`
  on the union of `positions_x/y/z` and `velocities_x/y/z`. The
  assertion succeeds when *any* element of *any* of those six arrays
  differs from its snapshot value — proving the lossy mode does not
  preserve bit-exact reversibility.

## Out of Scope <!-- rq-c2bc2592 -->

- New public types or launchers; this feature is a test only.
- A `simulation_step` or `simulation_step_lossless` orchestration
  function (a future feature in `src/simulation.rs`).
- Energy conservation or other physical-correctness diagnostics.
- Cross-hardware reversibility. The architecture's bit-exactness
  guarantee is GPU-vs-GPU on the same device.
- Reversibility of the residual buffers themselves. The architecture
  permits O(`f64` ULP) per-step residual drift; only the observable
  `f32` state must round-trip bit-exactly.
- Quantitative bounds on how badly the lossy mode fails. The negative
  test only confirms that *some* element differs; it does not measure
  the magnitude of the failure.
- Trajectory I/O.

---

## Gherkin Scenarios <!-- rq-e4f403e9 -->

```gherkin
Feature: End-to-end pipeline reversibility

  Background:
    Given the test fixture defined above (N=64 LJ fluid, 4×4×4 perturbed lattice, dt=0.001)
    And a fresh ParticleBuffers, fresh PairBuffer, fresh LosslessBuffers, and fresh neighbor_counts CudaSlice constructed from the fixture
    And the warm-up pass (lj_pair_force followed by reduce_pair_forces) has populated forces with F(x_0)
    And a snapshot of all eleven ParticleBuffers arrays and all six LosslessBuffers residual arrays has been captured immediately after warm-up

  @rq-0099ef65
  Scenario: Single-step lossless round trip restores observables bit-exactly
    When one full lossless step (vv_kick_drift_lossless, lj_pair_force, reduce_pair_forces, vv_kick_lossless) is executed with dt=0.001
    And every velocity component (high and low) is negated on the device
    And one full lossless step is executed with dt=0.001
    And every velocity component (high and low) is negated on the device
    And ParticleBuffers and LosslessBuffers are downloaded
    Then positions_x, positions_y, positions_z agree with the snapshot byte-for-byte
    And velocities_x, velocities_y, velocities_z agree with the snapshot byte-for-byte
    And forces_x, forces_y, forces_z agree with the snapshot byte-for-byte
    And masses and particle_ids agree with the snapshot byte-for-byte
    And every residual buffer agrees with the snapshot to within an absolute tolerance of 1e-10

  @rq-7822b88b
  Scenario: 100-step lossless round trip restores observables bit-exactly
    When 100 full lossless steps are executed with dt=0.001
    And every velocity component (high and low) is negated on the device
    And 100 full lossless steps are executed with dt=0.001
    And every velocity component (high and low) is negated on the device
    And ParticleBuffers and LosslessBuffers are downloaded
    Then positions_x, positions_y, positions_z agree with the snapshot byte-for-byte
    And velocities_x, velocities_y, velocities_z agree with the snapshot byte-for-byte
    And forces_x, forces_y, forces_z agree with the snapshot byte-for-byte
    And masses and particle_ids agree with the snapshot byte-for-byte
    And every residual buffer agrees with the snapshot to within an absolute tolerance of 1e-10

  @rq-b87fd5e8
  Scenario: Positions visibly evolve over the 100-step lossless forward run
    Given a snapshot of the host positions_x, positions_y, positions_z immediately after warm-up
    When 100 full lossless steps are executed with dt=0.001
    And ParticleBuffers is downloaded into a host ParticleState
    Then for at least one particle index i, the displacement
         sqrt((final.positions_x[i] - initial.positions_x[i])^2 +
              (final.positions_y[i] - initial.positions_y[i])^2 +
              (final.positions_z[i] - initial.positions_z[i])^2)
         is greater than 0.001

  @rq-ed048159
  Scenario: All observables are finite after the 100-step lossless forward run
    When 100 full lossless steps are executed with dt=0.001
    And ParticleBuffers is downloaded into a host ParticleState
    Then every element of positions_x, positions_y, positions_z is finite
    And every element of velocities_x, velocities_y, velocities_z is finite
    And every element of forces_x, forces_y, forces_z is finite

  @rq-1b44b5da
  Scenario: Lossy 100-step round trip does NOT restore observables
    Given a fresh ParticleBuffers, fresh PairBuffer, and fresh neighbor_counts built from the fixture
    And the warm-up pass has populated forces with F(x_0)
    And a snapshot of positions_x, positions_y, positions_z, velocities_x, velocities_y, velocities_z immediately after warm-up
    When 100 full lossy steps (vv_kick_drift, lj_pair_force, reduce_pair_forces, vv_kick) are executed with dt=0.001
    And every velocity component is negated on the device
    And 100 full lossy steps are executed with dt=0.001
    And every velocity component is negated on the device
    And ParticleBuffers is downloaded into a host ParticleState
    Then at least one element across positions_x, positions_y, positions_z, velocities_x, velocities_y, velocities_z differs from its snapshot value
```
