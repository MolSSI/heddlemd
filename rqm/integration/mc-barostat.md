# Feature: Monte-Carlo Barostat <!-- rq-b13a5224 -->

The Monte-Carlo barostat is an isotropic pressure-coupling slot that
samples the NPT distribution by periodic Metropolis volume moves rather
than a continuous per-step rescale. One of the pluggable barostat slots
(see `framework.md`); selected by `kind = "monte-carlo"` in the config's
`[barostat]` section.

Every `frequency` steps the barostat proposes a trial change in box
volume, rigidly scales every molecule's centre of mass by the
corresponding linear factor (internal molecular geometry is unchanged),
re-evaluates the total potential energy at the trial configuration, and
accepts or rejects the move with the Metropolis criterion for the NPT
ensemble. Between moves the system propagates under the integrator and
optional thermostat alone, with no pressure coupling and **no per-step
virial**: the move's accept/reject decision uses only the configurational
potential energy, never the instantaneous virial.

Because the move scales molecular centres of mass and not individual
atoms, it is a valid trial move for molecular systems with rigid
constraints (e.g. SETTLE/SHAKE water): a uniform per-atom scale would
stretch every rigid molecule and drive the acceptance rate to zero,
whereas centre-of-mass scaling displaces whole molecules and leaves their
internal coordinates — and therefore the intramolecular and constraint
energies — invariant. The molecules the barostat scales are the
connectivity-derived `MoleculeList` described in `forces/topology.md`.

The barostat preserves velocities exactly (a volume move is a
configurational Monte-Carlo move; it never touches `velocities_*`) and
uses a counter-based Philox-4×32-10 RNG with two uniform draws per
attempted move, so it produces byte-identical trajectories across runs on
the same GPU with the same seed.

Unlike the Berendsen barostat, the Monte-Carlo barostat is suitable for
canonical-ensemble production runs: the Metropolis accept/reject step
samples the correct NPT volume distribution. Unlike the C-rescale
barostat, it needs no per-pair virial accumulation and no per-step scalar
reduction during production, which lets the dynamics between moves run the
cheaper forces-only force evaluation (see *Graph Interaction*).

## Periodicity and the Barostat Slot Contract <!-- rq-2acc094a -->

A barostat declares whether it couples to the dynamics on every step or
only periodically through `Barostat::periodicity()` (see *Feature API*):

- A **per-step** barostat (`BarostatPeriodicity::EveryStep`; the
  Berendsen and C-rescale barostats) computes a rescale factor from the
  current kinetic energy and virial inside the captured per-step
  sequence. Its `apply` runs scalar-prep every step and its per-particle
  rescale is folded into the JIT-composed post-force kernel.
- A **periodic** barostat (`BarostatPeriodicity::EveryNSteps(frequency)`;
  the Monte-Carlo barostat) performs no per-step work at all. Its `apply`
  hook is a no-op. The whole move is host-orchestrated and runs through
  the separate `apply_move` hook the runner invokes at a batch boundary
  once every `frequency` steps (see *Per-Move Sequence*).

The Monte-Carlo barostat returns `BarostatPeriodicity::EveryNSteps(self.frequency)`
and leaves `Barostat::apply` at the no-op default; it implements
`apply_move`. It contributes no JIT-composed post-force fragment
(`post_force_per_particle()` returns `None`): its per-particle work is the
centre-of-mass scale kernel launched directly inside `apply_move`, not a
per-step fragment.

## Graph Interaction <!-- rq-0d729ecb -->

The Monte-Carlo barostat reports `graph_compatible = true`. The captured
per-step graph contains only the integrator (and optional thermostat)
sequence — no barostat node — exactly as an NVT phase does. The move runs
on the host between graph batches, so it never appears inside a captured
graph and never needs to be replayed.

A periodic barostat does **not** force every step to evaluate scalars.
The runner's `needs_scalars(s)` predicate is driven by the output cadence
(`log_every`) only; a `BarostatPeriodicity::EveryNSteps` barostat does not
set it true. The phase therefore captures **both** the forces-only and the
forces+scalars graphs (see `cuda-graphs.md`), and every non-log dynamics
step replays the cheap forces-only graph. The forces+scalars evaluations
the move itself performs are issued by `apply_move` on the host, outside
the captured graphs.

The runner bounds each replay batch so it ends on the next move boundary,
in the same way it already bounds batches on the next log / trajectory
boundary:

```text
batch = min(K, remaining, next_log, next_traj, next_move)
```

where `next_move = frequency - (step % frequency)`. At each move boundary
the runner calls `barostat.apply_move(...)` after the batch's last replay
and before the next batch's first replay. A move mutates the device
lattice and particle positions; the next batch's first force evaluation
observes the new box through the existing `SimulationBox::generation()`
change-detection path and rebuilds the neighbor list (and refreshes the
SPME influence function) accordingly. When a move is rejected the box and
positions are restored to their pre-move values, which bumps the
generation again, so the next batch still rebuilds against the restored
box.

`apply_move` performs host-side dtoh reads (the trial potential energy and
the acceptance decision) and host-side branching, so it is never run under
stream capture. It runs on the same default stream as every other kernel,
between graph replays, so the dynamics and the move observe each other's
results through ordinary per-stream ordering.

## Algorithm <!-- rq-8114b8c4 -->

The move is invoked through `apply_move(force_field, buffers, sim_box,
constraint, dt, timings)` at a batch boundary on steps where
`step % frequency == 0`. It reads `buffers.positions_*`, the molecule
table, and (re)evaluates the potential energy through `force_field`. It
writes `buffers.positions_*` and `sim_box`. It never reads or writes
`buffers.velocities_*`.

For each invocation:

1. **Current energy.** Evaluate the force field at the current
   configuration with `force_field.step(buffers, sim_box, timings,
   AggregateLevel::ForcesAndScalars)` and reduce the total potential
   energy `U_old` with `compute_total_potential_energy`. This refreshes
   `buffers.forces_*` and `buffers.potential_energies` from the current
   (pre-move) box, which the preceding forces-only dynamics steps may have
   left stale. The reduction is a single host dtoh of one `Real`.

2. **Snapshot.** Record the pre-move state needed for a possible revert:
   - copy `positions_x/y/z` device-to-device into the slot-owned
     `pos_snapshot_{x,y,z}` buffers,
   - copy `forces_x/y/z` device-to-device into the slot-owned
     `force_snapshot_{x,y,z}` buffers (so a reject restores the correct
     `F(t)` for the next dynamics step without an extra force evaluation),
   - read the current lattice into the host six-tuple via
     `sim_box.flush_from_device()` and cache it, and compute
     `V_old = lx · ly · lz`.

3. **Pre-increment** the host-side `draw_counter` by `+1`.

4. **Propose a volume change.** Draw the first Philox uniform
   `u1 ∈ [0, 1)` from `(seed, draw_counter)` (see *RNG*) and form

   ```text
   ΔV    = max_volume_step · (2 · u1 − 1)
   V_new = V_old + ΔV
   scale = (V_new / V_old)^(1/3)
   ```

   When `V_new ≤ 0` the move is rejected immediately at step 8 with
   `accepted = false` (no trial energy is evaluated); it still counts as
   an attempted move for the adaptive-step bookkeeping.

5. **Box-width guard.** When the proposed (contracting) box would make the
   minimum perpendicular width too small for the force field's interaction
   radius — `sim_box`-with-`scale`-applied fails
   `check_min_perpendicular_width(r_search)`, where `r_search = r_cut +
   r_skin` is the neighbor-search radius — the move is rejected at step 8
   with `accepted = false` and no trial energy is evaluated. It counts as
   an attempted move. (An expanding move always passes this guard.)

6. **Apply the trial.** Multiply all six lattice parameters by `scale`
   (`sim_box.multiply_lattice_isotropic(scale)`; bumps the generation),
   and scale every molecule's centre of mass by `scale` with the
   `mc_barostat_scale_molecule_com` kernel: each molecule's atoms are
   translated rigidly by `(scale − 1) · COM_molecule` so the molecular
   centre of mass scales about the origin while every intramolecular
   displacement is unchanged. Fractional coordinates of each molecule's
   centre of mass are invariant under the scale, so no PBC wrap is
   required and image flags carry over unchanged.

7. **Trial energy.** Evaluate the force field at the trial configuration
   with `force_field.step(buffers, sim_box, timings,
   AggregateLevel::ForcesAndScalars)` (the generation bump from step 6
   makes this rebuild the neighbor list against the trial box) and reduce
   the total potential energy `U_new` with
   `compute_total_potential_energy`.

8. **Metropolis test.** With `k_B = 1` in atomic units, let `kT =
   temperature`, `N_mol` the number of molecules in the `MoleculeList`,
   and

   ```text
   w = (U_new − U_old) + P_target · ΔV − N_mol · kT · ln(V_new / V_old)
   ```

   Draw the second Philox uniform `u2 ∈ [0, 1)` from the same
   `(seed, draw_counter)` invocation (a second output lane; see *RNG*).
   The move is accepted when

   ```text
   w ≤ 0   or   u2 < exp(−w / kT).
   ```

   The `−N_mol · kT · ln(V_new / V_old)` term is the Jacobian of the
   centre-of-mass volume move for `N_mol` independent molecular groups;
   `N_mol` is the molecule count, **not** the atom count. (Equivalently,
   the acceptance weight is `exp(−β·[ΔU + P_target·ΔV] + N_mol ·
   ln(V_new/V_old))` with `β = 1/kT`.)

9. **Commit or revert.**
   - **Accepted:** keep the trial box and positions.
     `buffers.forces_*` and `buffers.potential_energies` already hold the
     trial-configuration values from step 7, which are the correct `F(t)`
     and energy for the new box, so the next dynamics step consumes them
     directly.
   - **Rejected:** restore `positions_x/y/z` from `pos_snapshot_*`, restore
     `forces_x/y/z` from `force_snapshot_*`, and restore the lattice with
     `sim_box.set_lattice(...)` from the cached pre-move six-tuple (bumps
     the generation). The restored `forces_*` are the correct `F(t)` for
     the restored box, so the next dynamics step needs no extra force
     evaluation. When the early reject of step 4 or step 5 fired, no trial
     was applied, so the revert is the identity and only the
     attempted-move counters advance.

10. **Adaptive step bookkeeping.** Increment `n_attempted` by 1 and, when
    the move was accepted, `n_accepted` by 1. When `n_attempted` reaches
    `ADAPT_INTERVAL` (10): if `n_accepted < 0.25 · n_attempted`, set
    `max_volume_step ← max_volume_step / 1.1`; if
    `n_accepted > 0.75 · n_attempted`, set `max_volume_step ←
    min(max_volume_step · 1.1, 0.3 · V_new_or_current)`; otherwise leave it
    unchanged; then reset `n_attempted` and `n_accepted` to 0. The
    adjustment is a pure deterministic function of the accept history, so
    it preserves byte-for-byte reproducibility.

11. **Diagnostics.** Update the host-side diagnostic fields used by the log
    columns: `most_recent_volume` (the post-move volume — `V_new` on
    accept, `V_old` on reject), the running `accepted_moves` /
    `attempted_moves` totals, and `cumulative_barostat_injection += P_target
    · (V_post − V_old)` (zero on a rejected move).

When `buffers.particle_count() == 0`, the entire hook is a no-op: no force
evaluation, no kernel launch, no RNG draw, no box or position mutation, and
no diagnostic update.

The user is responsible for keeping the barostat's `temperature` parameter
consistent with any configured thermostat's target temperature. The
framework performs no cross-slot validation; the barostat reads only its
own `temperature` field.

## Per-Move Sequence <!-- rq-88ef7e60 -->

Per move (once every `frequency` steps, at a batch boundary), `apply_move`
issues:

| Order | Step                | Call                                   | Operation                                                                                   | Stage label                     |
| ----- | ------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------- |
| 1     | Current force eval  | `force_field.step(.., ForcesAndScalars)` | refresh forces / energy at the pre-move box                                                | (force-field stages)            |
| 2     | Current PE reduce   | `compute_total_potential_energy`       | dtoh of `U_old`                                                                              | `POTENTIAL_ENERGY_REDUCE`       |
| 3     | Position snapshot   | device-to-device copy ×3               | save `positions_{x,y,z}`                                                                     | `MC_BAROSTAT_SNAPSHOT`          |
| 4     | Force snapshot      | device-to-device copy ×3               | save `forces_{x,y,z}`                                                                        | `MC_BAROSTAT_SNAPSHOT`          |
| 5     | COM scale           | `mc_barostat_scale_molecule_com`       | scale each molecule's COM by `scale`; mutate lattice via `multiply_lattice_isotropic`       | `MC_BAROSTAT_SCALE_COM`         |
| 6     | Trial force eval    | `force_field.step(.., ForcesAndScalars)` | rebuild neighbor list + forces / energy at the trial box                                   | (force-field stages)            |
| 7     | Trial PE reduce     | `compute_total_potential_energy`       | dtoh of `U_new`                                                                              | `POTENTIAL_ENERGY_REDUCE`       |
| 8     | Revert (reject only)| device-to-device copy ×6 + `set_lattice` | restore positions, forces, and lattice                                                     | `MC_BAROSTAT_REVERT`            |

Steps 3–4 and 6–8 are skipped on an early reject (step 4 / step 5 of the
*Algorithm*). The two force evaluations dominate the move's cost; at the
default `frequency = 25` they add roughly `2/25 ≈ 8%` extra force
evaluations on top of the dynamics, which the forces-only dynamics graph
between moves more than offsets relative to a per-step virial barostat.

## Molecule Grouping <!-- rq-3e1fba8b -->

The barostat scales the molecules of the connectivity-derived
`MoleculeList` defined in `forces/topology.md`: every connected component
of the combined bond + constraint graph is one molecule, and every atom in
no bond and no constraint is its own singleton molecule. For a rigid SPC
water system each three-atom SETTLE group is one molecule, so `N_mol`
equals the water-molecule count. For a monatomic fluid with no bonds or
constraints every atom is a singleton molecule and the move reduces to
per-atom scaling with `N_mol = N`.

The barostat reads the device-resident `mol_atom_offsets` /
`mol_atom_indices` tables and `masses` to compute each molecule's centre of
mass inside `mc_barostat_scale_molecule_com`. The molecule table is built
once at load time and never changes during a run.

## Parameters <!-- rq-f8dc8ee0 -->

The matching builder deserialises a typed `McBarostatParams` from the
`[barostat]` section's `SlotConfig::params` (see `framework.md`); the
per-field reference below documents that parameter struct:

- `pressure: f64` — target pressure `P_target` in `E_h / a_0^3` (the
  engine's atomic pressure unit). Required. Finite. May be any sign or
  zero.
- `temperature: f64` — target temperature `T` as `k_B · T` in Hartrees
  (the engine's internal temperature representation; `k_B = 1`) used in the
  Metropolis weight. Required. Finite and strictly positive. Independent of
  `simulation.temperature` and of any `[thermostat].temperature`; the
  framework performs no cross-slot validation.
- `frequency: u32` — number of timesteps between attempted volume moves.
  Optional; default `25`. Must be `>= 1`.
- `volume_step: f64` — initial maximum volume displacement
  `max_volume_step` in `a_0^3`, the half-width of the uniform volume
  proposal. Optional; default `0.01 · V_0`, one percent of the initial box
  volume `V_0` at slot construction. Must be finite and strictly positive
  when supplied. Retuned during the run by the adaptive-step rule (step 10
  of the *Algorithm*).
- `seed: u64` — counter-based RNG seed for the per-move uniform draws.
  Required, independent of `simulation.seed` and any other slot's seed.

## CUDA Kernels and launch helpers <!-- rq-c83742c0 -->

`kernels/mc_barostat.cu` declares the molecule-centre-of-mass scale
kernel:

```c
extern "C" __global__ void mc_barostat_scale_molecule_com(
    float *positions_x,
    float *positions_y,
    float *positions_z,
    const unsigned int *mol_atom_offsets,   // length n_mol + 1
    const unsigned int *mol_atom_indices,   // length N, atom ids grouped by molecule
    const float *masses,                    // length N
    float scale,
    unsigned int n_mol);
```

One warp (or thread) per molecule. For molecule `m` over its atom slice
`mol_atom_indices[mol_atom_offsets[m] .. mol_atom_offsets[m+1]]`, the
kernel computes `COM = (Σ m_i · x_i) / (Σ m_i)` componentwise, then writes
`x_i ← x_i + (scale − 1) · COM` for every atom `i` in the molecule. The
per-molecule reduction sums atoms in their stored (ascending-index) order,
so the result is bit-identical across runs on the same GPU. A singleton
molecule's COM is its single atom's position, so the atom is simply scaled
about the origin.

The Rust launcher `mc_barostat_scale_molecule_com` is exposed under
`crate::gpu`. The barostat reuses `compute_total_potential_energy`
(`forces/framework.md`) for both energy reductions and
`SimulationBox::multiply_lattice_isotropic` / `set_lattice`
(`simulation-box.md`) for the lattice mutation and revert. Position and
force snapshot / restore are device-to-device copies of the SoA buffers; no
dedicated kernel is required.

## RNG <!-- rq-32f5770f -->

The Monte-Carlo barostat draws two standard-uniform samples per attempted
move from a single counter-based Philox-4×32-10 invocation
(Salmon et al., SC11) on the **host**, using the same algorithm as every
other host-side stochastic slot in the engine
(`src/integrator/philox.rs`). The two uniforms are two output lanes of the
one Philox block; `u1` drives the volume proposal (step 4) and `u2` drives
the acceptance test (step 8).

### Counter packing <!-- rq-174505f0 -->

Each per-move draw uses one Philox invocation with:

- **Key (2 × u32)**: `(seed_lo, seed_hi)` — low and high halves of the
  barostat's `seed`.
- **Counter (4 × u32)**:
  - `counter[0] = draw_counter_lo` — low 32 bits of the barostat's
    `draw_counter`.
  - `counter[1] = draw_counter_hi` — high 32 bits.
  - `counter[2] = 0` — reserved.
  - `counter[3] = 0` — reserved.

The `draw_counter` lives on `McBarostat`. It starts at `0` at construction
and is pre-incremented on every `apply_move` call before the draws; the
first move in a run uses `draw_counter == 1`. Early-rejected moves (steps 4
and 5) still pre-increment and consume the draw, so the counter advances by
exactly `+1` per attempted move whether or not a trial energy was
evaluated.

### Reproducibility <!-- rq-1dc985cc -->

Two runs with identical `(seed, pressure, temperature, frequency,
volume_step)`, identical initial particle state, and identical molecule
tables on the same GPU produce byte-identical trajectory and log files. The
Philox stream is stateless; each move's two uniforms are a pure function of
`(seed, draw_counter)`. The accept/reject branch, the adaptive-step update,
and the cumulative-injection accumulation are deterministic functions of
the reduced `f32` energies and the deterministic parameters.

## Diagnostic log columns <!-- rq-d0506951 -->

The barostat exposes four per-log-row diagnostic columns when it is
configured (see `io/log-output.md`):

- `box_volume` — simulation-box volume `V` in `a_0^3`, the post-move volume
  as of the most recent move, matching the lattice the trajectory frame
  writes at the same step.
- `mc_acceptance` — the running acceptance ratio
  `accepted_moves / attempted_moves` since the start of the run
  (dimensionless). Reported as `0.0` before the first attempted move.
- `mc_volume_step` — the current adaptive `max_volume_step` in `a_0^3`.
- `mc_conserved` — `ke + pe + P_target · V − cumulative_barostat_injection`,
  where the cumulative term accumulates `P_target · (V_post − V_pre)` over
  every accepted move. Mirrors the C-rescale barostat's
  `c_rescale_conserved` diagnostic. The runner supplies the freshly-computed
  total kinetic and potential energies for the row; the barostat combines
  them with its own cached state.

Between log rows these host fields are current as of the most recent move
(the move runs on the host, so no device flush is required to read them).

## Empty State and degenerate cases <!-- rq-29b5ed1e -->

- `buffers.particle_count() == 0`: `apply_move` returns `Ok(())` without
  evaluating the force field, launching any kernel, drawing from the RNG,
  or mutating `sim_box`. The `box_volume` column reports
  `sim_box.volume()`; `mc_acceptance` and `mc_volume_step` report their
  initial values; `mc_conserved` reports `ke + pe + P_target · V` (the
  cumulative term is zero).
- `V_new ≤ 0` (proposed non-positive volume): the move is rejected without
  a trial energy evaluation and counts as an attempted move.
- A contracting proposal that violates the minimum-perpendicular-width
  guard (step 5): rejected without a trial energy evaluation and counts as
  an attempted move.
- `frequency` larger than the phase's `n_steps`: no move boundary is ever
  reached and the barostat performs no move; the phase runs as constant
  volume. This is accepted at config-load time (the barostat does not know
  the phase length).

## Feature API <!-- rq-d651a188 -->

### Types <!-- rq-4fead095 -->

- `McBarostat` — implements the `Barostat` trait declared in <!-- rq-09ac44ea -->
  `framework.md`. Registered in `BarostatRegistry::with_builtins` under
  `kind_name() == "monte-carlo"`. Fields:

  - `device: Arc<CudaDevice>`
  - `pressure: f64` — `P_target`.
  - `temperature: f64` — `kT`.
  - `frequency: u32` — steps between attempted moves.
  - `max_volume_step: f64` — current adaptive proposal half-width.
  - `seed: u64`
  - `draw_counter: u64` — Philox counter advance. Initialised to `0`;
    pre-incremented on every `apply_move`.
  - `n_attempted: u32`, `n_accepted: u32` — adaptive-window counters,
    reset every `ADAPT_INTERVAL` attempts.
  - `attempted_moves: u64`, `accepted_moves: u64` — run-total counters for
    the `mc_acceptance` column.
  - `cumulative_barostat_injection: f64` — running sum of
    `P_target · (V_post − V_pre)` across accepted moves.
  - `most_recent_volume: f64` — post-move volume from the latest move.
  - `n_molecules: usize` — molecule count `N_mol`.
  - `mol_atom_offsets: CudaSlice<u32>` — device molecule offset table
    (length `n_mol + 1`), uploaded once at construction from the
    `MoleculeList`.
  - `mol_atom_indices: CudaSlice<u32>` — device atom-index table
    (length `N`).
  - `pos_snapshot_x/y/z: CudaSlice<Real>` — pre-move position snapshot
    buffers (length `N`).
  - `force_snapshot_x/y/z: CudaSlice<Real>` — pre-move force snapshot
    buffers (length `N`).
  - `pe_scratch: CudaSlice<Real>` — length-1 device buffer for the
    potential-energy reductions.

  All fields private; the slot's public surface is the `Barostat` trait
  methods and construction via `McBarostatBuilder`.

- `McBarostatParams` — `#[serde(deny_unknown_fields)]` parameter struct: <!-- rq-c6ee2fb9 -->
  `pressure: Pressure`, `temperature: Temperature`,
  `frequency: u32` (default `25`), `volume_step: Option<f64>`,
  `seed: u64`. Mirrors the unit-converting parameter pattern of
  `CRescaleBarostatParams`.

- `McBarostatBuilder` — implements `BarostatBuilder` with <!-- rq-6e1916c0 -->
  `kind_name() == "monte-carlo"`. `build(gpu, particle_count,
  n_constraints, params)` deserialises `McBarostatParams`, derives the
  `MoleculeList` from the run's `BondList` + `ConstraintList`, uploads the
  molecule tables, allocates the snapshot and scratch buffers, resolves the
  initial `max_volume_step` (the supplied `volume_step` or `0.01 · V_0`),
  and returns the boxed `McBarostat`. The builder reports
  `graph_compatible(&params) == true`.

- `BarostatPeriodicity` — closed enum describing how often a barostat <!-- rq-bfd5cc3a -->
  couples to the dynamics:

  ```rust
  pub enum BarostatPeriodicity {
      EveryStep,
      EveryNSteps(u32),
  }
  ```

### `Barostat` trait surface <!-- rq-106fbbdf -->

The `Barostat` trait (`framework.md`) carries the periodicity declaration
and the periodic-move hook in addition to the per-step `apply`:

- `periodicity(&self) -> BarostatPeriodicity` — default <!-- rq-0ba1a24a -->
  `BarostatPeriodicity::EveryStep`. `McBarostat` returns
  `EveryNSteps(self.frequency)`.
- `apply(&mut self, ...) -> Result<(), BarostatError>` — `McBarostat` <!-- rq-3597f2a0 -->
  leaves this at the no-op default (it does no per-step work).
- `apply_move(&mut self, force_field: &mut ForceField, buffers: &mut <!-- rq-03a5a290 -->
  ParticleBuffers, sim_box: &mut SimulationBox, constraint: Option<&mut dyn
  Constraint>, dt: Real, timings: &mut Timings) -> Result<(),
  BarostatError>` — default no-op; `McBarostat` implements the
  *Per-Move Sequence*. Receives `&mut ForceField` (the only barostat hook
  that does) because the Metropolis test requires trial energy
  evaluations. The `constraint` argument lets a future move re-project
  constraints after the scale; for the centre-of-mass move it is unused
  because rigid-molecule geometry is invariant under the scale.
- `post_force_per_particle(&self) -> Option<&dyn PostForcePerParticle>` — <!-- rq-282850ad -->
  `McBarostat` returns `None`.
- `log_column_names(&self) -> &'static [(&'static str, Dimension)]` — <!-- rq-cf8948a8 -->
  returns `[("box_volume", Dimensionless), ("mc_acceptance",
  Dimensionless), ("mc_volume_step", Dimensionless), ("mc_conserved",
  Energy)]`.
- `log_column_values(&self, ke, pe) -> Vec<f64>` — returns <!-- rq-59f42062 -->
  `[most_recent_volume, acceptance_ratio, max_volume_step, ke + pe +
  pressure · most_recent_volume − cumulative_barostat_injection]`.

### Functions <!-- rq-fa2fd108 -->

- `mc_barostat_scale_molecule_com(buffers, mol_atom_offsets, <!-- rq-1f5391f7 -->
  mol_atom_indices, scale) -> Result<(), GpuError>` — launch helper for the
  centre-of-mass scale kernel (one warp per molecule). Exposed under
  `crate::gpu`.

## Out of Scope <!-- rq-56545622 -->

- Semi-isotropic and anisotropic (per-axis) volume moves. The move applies
  a single scalar `scale` to all six lattice parameters; per-axis or
  membrane (xy-coupled, z-independent) moves are a separate feature.
- Per-atom (non-molecular) scaling as a selectable mode. The move always
  scales molecular centres of mass; a monatomic system reduces to per-atom
  scaling because every atom is its own molecule.
- A force-free `AggregateLevel::EnergyOnly` evaluation. The trial energy is
  obtained from a `ForcesAndScalars` force evaluation, which also computes
  forces that are reused as `F(t)` on an accepted move. An energy-only
  evaluation level that skips the force write is a possible later
  optimization.
- Freezing the adaptive step for rigorous fixed-proposal sampling. The
  `max_volume_step` retunes throughout the run; a configurable freeze is a
  later addition.
- Restart-from-checkpoint restoration of the adaptive `max_volume_step` and
  the move counters. They reset to their construction values on a fresh
  run.
- Constraint re-projection after the scale. The centre-of-mass move leaves
  rigid-molecule geometry invariant, so no re-projection is needed; a move
  that scaled atoms independently would require it.
- Multiple simultaneous barostats. The runner holds at most one
  `Box<dyn Barostat>` (`framework.md`).

---

## Gherkin Scenarios <!-- rq-db9cf120 -->

```gherkin
Feature: Monte-Carlo barostat

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called
    And a SimulationBox with lx=ly=lz=20.0 (a_0) unless otherwise specified

  # --- Construction ---

  @rq-220808ef
  Scenario: Construct McBarostat via the registry
    Given a BarostatKind::MonteCarlo {
      pressure: 3.4e-9, temperature: 9.4e-4,
      frequency: 25, volume_step: 80.0, seed: 42 }
    When registry.build_optional(Some(&kind), device, particle_count=12, n_constraints=9) is called
    Then it returns Ok(Some(barostat))
    And the underlying McBarostat has draw_counter == 0
    And the underlying McBarostat has frequency == 25
    And the underlying McBarostat has max_volume_step == 80.0
    And the underlying McBarostat reports periodicity() == EveryNSteps(25)

  @rq-9d8f0e2a
  Scenario: frequency defaults to 25 when omitted
    Given a [barostat] kind="monte-carlo" with pressure, temperature, seed and no frequency
    When load_config is called
    Then it returns Ok(config)
    And the barostat slot's resolved frequency is 25

  @rq-21ad8af9
  Scenario: volume_step defaults to one percent of the initial box volume
    Given a [barostat] kind="monte-carlo" with no volume_step and an initial box of volume V0
    When the barostat is built
    Then max_volume_step equals 0.01 * V0

  @rq-f609ea67
  Scenario: BarostatRegistry::with_builtins() exposes monte-carlo alongside berendsen and c-rescale
    Given a BarostatRegistry::with_builtins()
    Then the registry contains a builder whose kind_name() is "monte-carlo"

  # --- Config validation ---

  @rq-26d969aa
  Scenario: Accept negative target pressure
    Given a [barostat] kind="monte-carlo" with pressure=-3.4e-9, temperature=9.4e-4, seed=1
    When load_config is called
    Then it returns Ok(config)

  @rq-cb831438
  Scenario: Reject non-positive temperature
    Given a [barostat] kind="monte-carlo" with pressure=3.4e-9, temperature=0.0, seed=1
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "barostat.temperature", reason: _ })

  @rq-69ba800a
  Scenario: Reject zero frequency
    Given a [barostat] kind="monte-carlo" with pressure, temperature, seed, frequency=0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "barostat.frequency", reason: _ })

  @rq-e8f97894
  Scenario: Reject non-positive volume_step
    Given a [barostat] kind="monte-carlo" with pressure, temperature, seed, volume_step=0.0
    When load_config is called
    Then it returns Err(ConfigError::InvalidValue { field: "barostat.volume_step", reason: _ })

  @rq-1dadb8e9
  Scenario: Missing pressure rejected
    Given a [barostat] kind="monte-carlo" with temperature=9.4e-4, seed=1
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "barostat.pressure" })

  @rq-4207d7ef
  Scenario: Missing seed rejected
    Given a [barostat] kind="monte-carlo" with pressure=3.4e-9, temperature=9.4e-4
    When load_config is called
    Then it returns Err(ConfigError::MissingField { field: "barostat.seed" })

  # --- Periodicity / scalar cadence ---

  @rq-98c80507
  Scenario: apply is a no-op every step
    Given a monte-carlo barostat and a ParticleBuffers with N>0
    When barostat.apply(&mut buffers, &mut sim_box, dt, &mut timings) is called
    Then it returns Ok(())
    And sim_box.generation() is unchanged
    And no kernel is launched

  @rq-627fc517
  Scenario: a periodic barostat does not force per-step scalars
    Given an MD phase with a monte-carlo barostat, log_every=25, n_steps=100
    When the runner computes needs_scalars(s) for a non-log step s
    Then needs_scalars(s) is false
    And the phase captures both the forces-only and forces+scalars graphs

  @rq-1bcfd19c
  Scenario: a per-step barostat still forces per-step scalars
    Given an MD phase with a c-rescale barostat
    Then needs_scalars(s) is true for every step
    And the phase captures only the forces+scalars graph

  # --- Move cadence ---

  @rq-95944e8b
  Scenario: a move fires once per frequency steps at a batch boundary
    Given a monte-carlo barostat with frequency=25 and n_steps=100
    When the phase runs to completion
    Then apply_move is called exactly 4 times
    And each call occurs at a step where step % 25 == 0
    And apply_move is never called inside a captured graph

  @rq-6a983abc
  Scenario: draw_counter starts at 0 and increments by 1 per attempted move
    Given a freshly built McBarostat
    Then draw_counter == 0
    When apply_move is called once
    Then draw_counter == 1
    When apply_move is called again
    Then draw_counter == 2

  # --- Centre-of-mass scaling ---

  @rq-2aad1786
  Scenario: rigid-water internal geometry is invariant under a move
    Given an N=12 system of 4 SPC water molecules (each a 3-atom SETTLE group)
    And a snapshot of every intramolecular O-H and H-H displacement
    When a move with scale s is applied (forced accept)
    Then every intramolecular displacement equals its snapshot within f32 round-off
    And each molecule's centre of mass equals s times its pre-move centre of mass

  @rq-92eee9d8
  Scenario: a singleton-molecule system scales each atom about the origin
    Given an N=8 monatomic system with no bonds and no constraints
    And n_molecules equals 8
    When a move with scale s is applied (forced accept)
    Then every atom position equals s times its pre-move position within f32 round-off

  @rq-24c7ed51
  Scenario: N_mol is the molecule count, not the atom count
    Given an N=12 system of 4 three-atom water molecules
    Then the Metropolis weight uses N_mol = 4

  # --- Metropolis correctness ---

  @rq-c4cd5622
  Scenario: a move that lowers the weight is always accepted
    Given a configuration and a proposed volume change with w <= 0
    When apply_move evaluates the Metropolis test
    Then the move is accepted regardless of the acceptance uniform u2

  @rq-94c5a0ec
  Scenario: a move with w > 0 is accepted iff u2 < exp(-w/kT)
    Given a configuration and a proposed volume change with a known w > 0
    And a barostat seeded so the acceptance uniform u2 is known
    When apply_move evaluates the Metropolis test
    Then the move is accepted exactly when u2 < exp(-w / kT)

  @rq-8b4f642c
  Scenario: rejected move restores positions, forces, and box exactly
    Given a configuration with positions P, forces F, and lattice L
    And a proposed move that is rejected
    When apply_move returns
    Then positions equal P bit-for-bit
    And forces equal F bit-for-bit
    And sim_box.lattice() equals L bit-for-bit

  @rq-b4f1c3a8
  Scenario: accepted move keeps the trial box and trial forces
    Given a proposed move that is accepted with trial lattice L' and trial forces F'
    When apply_move returns
    Then sim_box.lattice() equals L'
    And buffers.forces_* equal F' (the trial-configuration forces) bit-for-bit

  @rq-cdd66a5a
  Scenario: velocities are never modified by a move
    Given a snapshot of every velocity component
    When apply_move is called (accept or reject)
    Then every velocity component equals its snapshot bit-for-bit

  # --- Degenerate proposals ---

  @rq-f632c56c
  Scenario: a non-positive proposed volume is rejected without a trial energy
    Given a proposal that yields V_new <= 0
    When apply_move is called
    Then no trial force evaluation is performed
    And the move is counted as attempted but not accepted
    And the box and positions are unchanged

  @rq-37e9c873
  Scenario: a contracting move that violates the width guard is rejected
    Given a box whose contraction by scale s would make min_perpendicular_width < r_cut + r_skin
    When apply_move proposes that contraction
    Then no trial force evaluation is performed
    And the move is rejected and counted as attempted

  @rq-e9b8f0a6
  Scenario: apply_move on empty state is a no-op
    Given a monte-carlo barostat with particle_count = 0
    When apply_move is called
    Then it returns Ok(())
    And sim_box.generation() is unchanged
    And draw_counter is unchanged

  # --- Adaptive step ---

  @rq-c479f47e
  Scenario: max_volume_step shrinks when acceptance is low
    Given a barostat whose last 10 attempted moves accepted fewer than 3
    When the 10th move completes
    Then max_volume_step has been multiplied by 1/1.1
    And the adaptive-window counters reset to zero

  @rq-a516b529
  Scenario: max_volume_step grows when acceptance is high
    Given a barostat whose last 10 attempted moves accepted more than 7
    When the 10th move completes
    Then max_volume_step has been multiplied by 1.1, capped at 0.3 * V
    And the adaptive-window counters reset to zero

  @rq-5977755f
  Scenario: max_volume_step is unchanged for mid-range acceptance
    Given a barostat whose last 10 attempted moves accepted between 3 and 7 inclusive
    When the 10th move completes
    Then max_volume_step is unchanged
    And the adaptive-window counters reset to zero

  # --- Log columns ---

  @rq-ebe75818
  Scenario: log_column_names returns the four MC barostat columns
    Given a constructed McBarostat
    Then log_column_names() equals
      ["box_volume", "mc_acceptance", "mc_volume_step", "mc_conserved"]

  @rq-a41f3e8b
  Scenario: log_column_values returns volume, acceptance, step, and conserved quantity
    Given a McBarostat with most_recent_volume = V, accepted_moves = a,
      attempted_moves = n, max_volume_step = d, cumulative_barostat_injection = c,
      pressure = P
    When log_column_values(ke, pe) is called
    Then it returns [V, a / n, d, ke + pe + P * V - c]

  @rq-656ce38f
  Scenario: Log header includes the MC barostat columns when configured
    Given a config with [barostat].kind = "monte-carlo" and log_every > 0
    When the runner produces the log file
    Then its header line ends with "box_volume,mc_acceptance,mc_volume_step,mc_conserved"

  # --- Composition ---

  @rq-98ac24d9
  Scenario: monte-carlo composes with velocity-Verlet + CSVR for NPT water
    Given a composed runner of velocity-Verlet + CSVR + monte-carlo with
      4096 SPC water molecules at T_target = 298.15 K (on both CSVR and the
      barostat), P_target = 1 bar, frequency = 25, n_steps = 20000
    When the run completes
    Then the run finishes without error
    And the time-averaged temperature over the last half is within 5% of 298.15 K
    And the time-averaged pressure over the last half is within 20% of 1 bar
    And the acceptance ratio settles into [0.25, 0.75]

  @rq-e8ceba10
  Scenario: monte-carlo composes with langevin-baoab
    Given a composed runner of langevin-baoab (no [thermostat]) + monte-carlo
      with N=128 LJ argon and matched temperatures
    When load_config and run are invoked
    Then the run finishes without error

  # --- Determinism ---

  @rq-5af90224
  Scenario: Two graph-mode runs with identical configs and seeds are byte-identical
    Given two complete simulations composing velocity-Verlet + monte-carlo with
      identical parameters (including identical barostat.seed) and identical
      initial state, n_steps = 200, cuda_graphs_disable = false
    When heddlemd run is invoked on each
    Then the trajectory files are byte-identical
    And the log files are byte-identical, including box_volume, mc_acceptance,
      mc_volume_step, and mc_conserved
    And the final SimulationBox lattices are byte-identical

  @rq-b33106df
  Scenario: Graph-mode and non-graph-mode runs are byte-identical
    Given a config with a monte-carlo barostat and seed S
    When run A sets cuda_graphs_disable = false
    And run B sets cuda_graphs_disable = true
    Then run A and run B produce byte-identical trajectory and log files

  @rq-9fbe4022
  Scenario: Different seeds produce different trajectories
    Given two simulations identical except barostat.seed = 1 and = 2
    When heddlemd run is invoked on each
    Then the trajectory files differ
```
