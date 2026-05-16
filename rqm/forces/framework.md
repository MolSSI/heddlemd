# Feature: Pluggable Potential Slot Framework <!-- rq-c1ea073b -->

The runner evaluates inter-particle forces through an ordered collection of
*potential slots* assembled into a `ForceField`. Each slot implements the
`Potential` trait, which exposes a contribution kernel, a reduction step,
and a per-axis output destination of length `particle_count`. The
`ForceField`'s `step()` method runs every slot's kernels in slot order and
combines their reduced outputs into the particle state's `forces_*` arrays
via a deterministic combiner kernel. The runner's force-evaluation step
calls `force_field.step(...)` once; it has no visibility into which
potentials participated.

A new potential is added by writing a type that implements `Potential` and
extending `ForceField::new` to construct it from the parsed config. The
combiner kernel and the slot collection are generic in the number of slots;
neither needs editing when a potential is added.

## Slots <!-- rq-cc73f184 -->

The framework constructs the following `Potential` implementations from
config:

| `kind` | Implementation | File |
| --- | --- | --- |
| `lennard-jones` | non-bonded pairwise via `[[pair_interactions]]` | `lj-pair-force.md` |
| `coulomb` | non-bonded truncated electrostatics via `[coulomb]` | `coulomb-pair-force.md` |
| `spme_real` | non-bonded `erfc`-screened pair force via `[spme]` | `spme.md` |
| `spme_reciprocal` | reciprocal-space FFT pipeline via `[spme]` | `spme.md` |
| `morse-bonded` | bonded pairwise via `[[bond_types]]` + `.topology` file | `morse-bonded.md` |
| `harmonic-angle` | three-body angle force via `[[angle_types]]` + `.topology` file | `harmonic-angle.md` |

Slots are present in the `ForceField` according to the config:

- The `LennardJones` slot is present when `[[pair_interactions]]` contains
  at least one entry.
- The `Coulomb` slot is present when the config supplies a `[coulomb]`
  table.
- The `SpmeRealSpace` and `SpmeReciprocal` slots are both present when
  the config supplies a `[spme]` table.
- The `MorseBonded` slot is present when the config supplies a
  `topology = "..."` path, the topology file's `[bonds]` section is
  non-empty, and at least one `[[bond_types]]` entry uses
  `potential = "morse"`.
- The `HarmonicAngle` slot is present when the config supplies a
  `topology = "..."` path, the topology file's `[angles]` section is
  non-empty, and at least one `[[angle_types]]` entry uses
  `potential = "harmonic"`.

The `[coulomb]` and `[spme]` tables are mutually exclusive at config
load (see `io/config-schema.md`); a `ForceField` therefore contains at
most one electrostatics path.

A `ForceField` with zero slots is a valid configuration. `step()` writes
zeros into `particle_buffers.forces_*` and returns without launching any
contribution or reduction kernels.

When multiple slots are present, they appear in the `ForceField`'s slot
list in a fixed order determined at construction:

1. `LennardJones`
2. `Coulomb`
3. `SpmeRealSpace`
4. `SpmeReciprocal`
5. `MorseBonded`
6. `HarmonicAngle`

Future potentials are inserted into this order at fixed positions defined
in `ForceField::new`.

## Force Evaluation Pipeline <!-- rq-7bab5c1e -->

The runner's force evaluation does the following per step:

1. **Shared neighbor-list update.** If `ForceField::neighbor_list` is
   `Some`, call its `pre_step` method (see `neighbor-list.md`). In
   cell-list mode this runs the displacement-check kernel and rebuilds
   the neighbor list when an atom's reference displacement exceeds
   `r_skin / 2`. In trivial mode and when `neighbor_list` is `None`,
   this step launches no kernels.
2. **Contribution kernels.** For each slot, in slot order, invoke
   `Potential::contribute`, passing a `ForceFieldContext` that carries
   a reference to the shared `NeighborListState` (when present) and any
   other shared services. The slot fills its private intermediate
   buffers (e.g. a `PairBuffer` or a `BondPairBuffer`) with per-pair
   force, energy, and virial contributions; the framework does not
   inspect them.
3. **Per-slot reductions.** For each slot, in slot order, invoke
   `Potential::reduce`, passing a `SlotOutputView` that points to the
   slot's assigned row of the framework's flat slot-output buffers.
   The reduction overwrites that row with the slot's per-particle
   reduced contributions: three force components, one potential-energy
   share, and one scalar-virial share (five quantities total).
4. **Combiner.** Run `accumulate_forces` once. The combiner reads the
   flat slot-output buffers and writes, for each per-particle quantity
   `Q` in `{force_x, force_y, force_z, potential_energy, virial}`:
   `particle_buffers.Q[i] = sum over k in 0..num_slots of slot_Q[k * n + i]`.
   The summation is left-to-right in `k`; each thread handles one `i`.

Identical runs on the same GPU with the same config produce byte-identical
`particle_buffers.forces_*`, `potential_energies`, and `virials`, and
therefore byte-identical trajectories.

## Slot Output Buffers <!-- rq-cd28340e -->

`ForceField` owns five contiguous device buffers, one per per-particle
output quantity, named `slot_forces_x`, `slot_forces_y`, `slot_forces_z`,
`slot_energies`, and `slot_virials`. Each has length
`num_slots * particle_count`. The row for slot `k` is the half-open range
`[k * particle_count, (k + 1) * particle_count)` within each buffer.

After slot `k`'s `reduce()` returns, row `k` contains that slot's
per-particle reduced contribution along that axis or scalar quantity:
three force components, one potential-energy share, and one scalar-virial
share. The combiner reads every row and sums in slot order, producing the
five per-particle aggregates on `ParticleBuffers`.

Memory cost: `5 * num_slots * particle_count * 4 bytes`. For
`particle_count = 10⁴` and four slots, this is ~800 KB — negligible.
For `particle_count = 10⁵` and four slots, ~8 MB.

When `num_slots == 0` the five buffers have length zero. When
`particle_count == 0` the five buffers have length zero regardless of
`num_slots`.

## Empty State <!-- rq-aa52268c -->

When `particle_count == 0`, every slot's `contribute` and `reduce` methods
early-return without launching, and the combiner returns without
launching. `ForceField::step` returns `Ok(())` having done no GPU work.

When `num_slots == 0`, the combiner kernel still launches (with
`num_slots = 0`) and writes zeros to every entry of
`particle_buffers.forces_*`, `potential_energies`, and `virials`. The
contribution and reduction phases launch no kernels. `ForceField::step`
returns `Ok(())`.

When a slot's *input list* is empty (e.g. a `MorseBonded` slot constructed
with `bonds.is_empty()`), the slot's `contribute` does not launch any
contribution kernel, the slot's `reduce` writes zeros into its assigned
rows of all five slot-output buffers, and the rest of the pipeline runs
normally. (The combiner reads every row unconditionally; empty-input
slots must carry valid zeros in their rows.)

## Feature API <!-- rq-0da87ca1 -->

### Types <!-- rq-e4960f89 -->

- `Potential` — object-safe trait implemented by every slot. <!-- rq-67ebf3b1 -->

  ```rust
  pub trait Potential: std::fmt::Debug + Send {
      fn label(&self) -> &'static str;

      fn max_cutoff(&self) -> Option<f32>;

      fn contribute(
          &mut self,
          buffers: &ParticleBuffers,
          sim_box: &SimulationBox,
          cx: &ForceFieldContext<'_>,
          timings: &mut Timings,
      ) -> Result<(), ForceFieldError>;

      fn reduce(
          &mut self,
          output: SlotOutputView<'_>,
          cx: &ForceFieldContext<'_>,
          timings: &mut Timings,
      ) -> Result<(), ForceFieldError>;
  }
  ```

  - `label` returns a short, stable, lower-snake-case identifier (for
    example `"lennard_jones"`, `"morse_bonded"`) used in error messages
    and diagnostic output. Two slots in the same `ForceField` must have
    distinct labels.
  - `max_cutoff` returns the maximum short-range interaction cutoff the
    potential needs the shared neighbor list to cover, in the same units
    as `SimulationBox`. Returns `None` if the potential does not consume
    the neighbor list (bonded potentials, intramolecular potentials
    keyed by index, etc.). `ForceField::new` aggregates these values to
    size the shared `NeighborListState`.
  - `contribute` runs the slot's contribution kernel(s) against the
    current `ParticleBuffers` and `SimulationBox`. The slot reads any
    shared resources it needs from `cx`. Implementations may read from
    `buffers` but must not write to it. Implementations that report
    `max_cutoff() == Some(_)` may assume `cx.neighbor_list` is
    `Some(_)`; implementations that report `None` are free to ignore
    `cx.neighbor_list` (it may be `None` or `Some` depending on whether
    any other slot needs the list).
  - `reduce` writes exactly `buffers.particle_count()` floats per slice
    into the five slices referenced by `output`: per-particle force x,
    force y, force z, potential-energy share, and scalar-virial share.

  Implementations are responsible for emitting their own
  `KernelStage` start/stop events through `timings`.

- `ForceFieldContext<'a>` — bundle of shared services that the framework <!-- inline --> <!-- rq-559783fe -->
  exposes to every `contribute` call. Constructed by `ForceField::step`
  for the duration of one step's contribution phase. Fields:

  ```rust
  pub struct ForceFieldContext<'a> {
      pub neighbor_list: Option<&'a NeighborListState>,
  }
  ```

  - `neighbor_list` is `Some(_)` when at least one slot reports
    `max_cutoff() == Some(_)` at construction; otherwise `None`. New
    shared services land here as additional fields without changing the
    `Potential::contribute` signature.

- `SlotOutputView<'a>` — five exclusive references to per-particle output <!-- rq-304b191b -->
  slices, each of length `particle_count`. Constructed by `ForceField`
  and passed into `Potential::reduce`. Implementations must treat the
  slices as write-only output buffers.

  ```rust
  pub struct SlotOutputView<'a> {
      pub force_x: CudaViewMut<'a, f32>,
      pub force_y: CudaViewMut<'a, f32>,
      pub force_z: CudaViewMut<'a, f32>,
      pub energy: CudaViewMut<'a, f32>,
      pub virial: CudaViewMut<'a, f32>,
  }
  ```

  Each field is a `CudaViewMut` of length `particle_count` onto the
  corresponding row of the framework's flat slot-output buffer. The
  view borrows the framework's storage for the duration of the
  `reduce` call.

- `LennardJonesState` — implements `Potential` with `label() == "lennard_jones"`. <!-- rq-af2d1628 -->
  Owns the slot's `PairBuffer`, `LennardJonesParameters`, and the
  `DeviceExclusionList` (see `topology.md`). Its internal state is one of
  two variants determined at construction time by the parsed
  `NeighborListConfig`:
  - `AllPairs` — additionally carries a `neighbor_counts` device slice
    with every entry equal to `N`. `max_neighbors == N`.
  - `CellList` — additionally carries a `NeighborListState` (see
    `neighbor-list.md`) that owns the cell list, neighbor list,
    reference positions, and overflow flag. `max_neighbors` comes from
    the config; `neighbor_counts` is populated per-rebuild by the
    neighbor-list build kernel.

  In either variant, the slot's reduction reads `neighbor_counts` (so
  the existing segmented `reduce_pair_forces` kernel works without
  branching) and writes its per-particle output into the
  `SlotOutputView` it receives.

- `MorseBondedState` — implements `Potential` with `label() == "morse_bonded"`. <!-- rq-2361f2b8 -->
  Owns the slot's `BondPairBuffer`, the bond index/offset tables, and
  the per-bond-type parameter table. Construction requires a non-empty
  bond list; see `morse-bonded.md`. Its reduction writes per-particle
  output into the `SlotOutputView` it receives.

- `HarmonicAngleState` — implements `Potential` with `label() == "harmonic_angle"`. <!-- rq-454ad2cf -->
  Owns the slot's `AnglePairBuffer`, the angle index/offset tables,
  and the per-angle-type parameter table. Construction requires a
  non-empty angle list; see `harmonic-angle.md`. Its reduction writes
  per-particle output into the `SlotOutputView` it receives.

- `ForceField` — handle owning the slot collection, the flat slot-output buffers, and the shared neighbor list. <!-- rq-684a29f1 -->

  Fields:
  - `device: Arc<CudaDevice>`
  - `slots: Vec<Box<dyn Potential>>` — in fixed evaluation order. May
    be empty.
  - `slot_forces_x: CudaSlice<f32>` — length `slots.len() * N`.
  - `slot_forces_y: CudaSlice<f32>` — length `slots.len() * N`.
  - `slot_forces_z: CudaSlice<f32>` — length `slots.len() * N`.
  - `slot_energies: CudaSlice<f32>` — length `slots.len() * N`.
  - `slot_virials: CudaSlice<f32>` — length `slots.len() * N`.
  - `neighbor_list: Option<NeighborListState>` — `Some(_)` when at
    least one slot returns `max_cutoff() == Some(_)`, `None`
    otherwise (bonded-only and zero-slot configurations).

- `ForceFieldError` — error type. Variants: <!-- rq-a2e20b02 -->
  - `Gpu(GpuError)` — CUDA driver / kernel-launch failure from any
    slot's kernel or the combiner.
  - `Timings(TimingsError)` — CUDA event recording failure.
  - `NeighborList(NeighborListError)` — surfaces failures from the
    cell-list pipeline (see `neighbor-list.md`), including the
    `NeighborListOverflow` and `BoxTooSmallForCells` cases.
  - `DuplicateLabel(&'static str)` — two slots constructed with the
    same `label()`. Reported from `ForceField::new`.

### Functions and methods <!-- rq-17abcb76 -->

- `ForceField::new(device: Arc<CudaDevice>, particle_count: usize, sim_box: &SimulationBox, particle_types: &[ParticleTypeConfig], pair_interactions: &[PairInteractionConfig], bond_types: &[BondTypeConfig], angle_types: &[AngleTypeConfig], coulomb_config: Option<&CoulombConfig>, spme_config: Option<&SpmeConfig>, charges: &[f32], bond_list: &BondList, angle_list: &AngleList, exclusion_list: &ExclusionList, neighbor_list_config: &NeighborListConfig) -> Result<ForceField, ForceFieldError>` <!-- rq-79938dbf -->
  - Constructs the slot collection in fixed evaluation order:
    - Appends a `LennardJonesState` slot when `pair_interactions` is
      non-empty.
    - Appends a `CoulombState` slot when `coulomb_config.is_some()`.
    - Appends `SpmeRealSpaceState` and `SpmeReciprocalState` slots
      when `spme_config.is_some()`.
    - Appends a `MorseBondedState` slot when `bond_list.is_empty()`
      is false.
    - Appends a `HarmonicAngleState` slot when
      `angle_list.is_empty()` is false.
  - When every slot-bearing input is absent, returns a `ForceField`
    with `slots.len() == 0`.
  - Allocates the five flat slot-output buffers (`slot_forces_x/y/z`,
    `slot_energies`, `slot_virials`) of length
    `slots.len() * particle_count` on `device`. When either factor is
    zero, the allocations are length-zero.
  - Builds the shared `NeighborListState`:
    - Computes `r_cut = max(slot.max_cutoff() for slot in slots if
      slot.max_cutoff().is_some())`. If no slot reports a cutoff,
      `neighbor_list` is set to `None` and the framework launches no
      neighbor-list kernels for the lifetime of the run.
    - Otherwise consults `neighbor_list_config`:
      - `CellList { max_neighbors, r_skin }`: calls
        `NeighborListState::new_cell_list(device, sim_box,
        particle_count, r_cut, max_neighbors, r_skin as f32)`. May
        return `ForceFieldError::NeighborList(_)` (e.g.
        `BoxTooSmallForCells`).
      - `AllPairs`: calls `NeighborListState::new_trivial(device,
        sim_box, particle_count)`.
  - Returns `ForceFieldError::DuplicateLabel(_)` if two slots end up
    with the same `label()`.

- `ForceField::step(&mut self, buffers: &mut ParticleBuffers, sim_box: &SimulationBox, timings: &mut Timings) -> Result<(), ForceFieldError>` <!-- rq-3579df3b -->
  - When `self.neighbor_list` is `Some(nl)`, calls `nl.pre_step(sim_box,
    buffers, timings)` to run the generation-cache check, the
    displacement check, and the rebuild as needed. When `None`, this
    step is skipped.
  - Constructs a `ForceFieldContext { neighbor_list:
    self.neighbor_list.as_ref() }` valid for the duration of the
    contribution phase.
  - For each slot in `self.slots`, in order, calls
    `slot.contribute(buffers, sim_box, &cx, timings)`.
  - For each slot in `self.slots`, in order, calls `slot.reduce(view,
    &cx, timings)` where `view` is a `SlotOutputView` whose five fields
    point into the row `slot_index * particle_count ..
    (slot_index + 1) * particle_count` of the five flat slot-output
    buffers.
  - Launches `accumulate_forces` once (with
    `KernelStage::AccumulateForces`). When `slots.is_empty()`,
    `accumulate_forces` still launches and writes zeros to
    `buffers.forces_*`, `buffers.potential_energies`, and
    `buffers.virials`.
  - Returns `Ok(())` on success.
  - Empty-state contract per *Empty State* above.

### Combiner Kernel <!-- rq-c0f98145 -->

`kernels/forces.cu` declares one `extern "C"` kernel:

```c
extern "C" __global__ void accumulate_forces(
    const float *slot_forces_x,    // shape [num_slots * n]
    const float *slot_forces_y,    // shape [num_slots * n]
    const float *slot_forces_z,    // shape [num_slots * n]
    const float *slot_energies,    // shape [num_slots * n]
    const float *slot_virials,     // shape [num_slots * n]
    unsigned int num_slots,
    float *forces_x,
    float *forces_y,
    float *forces_z,
    float *potential_energies,
    float *virials,
    unsigned int n);
```

Each thread maps to one particle index
`i = blockIdx.x * blockDim.x + threadIdx.x` (block size 256, grid
`ceil(n / 256)`, no shared memory, default stream of
`particle_buffers.device`). The thread computes, for each output quantity
`Q` in `{forces_x, forces_y, forces_z, potential_energies, virials}`:

```text
sum_Q = 0
for k in 0..num_slots:
    sum_Q += slot_Q[k * n + i]
Q[i] = sum_Q
```

The sums are performed left-to-right in slot order, so identical inputs
yield byte-identical outputs across runs.

When `num_slots == 0`, the inner loop does not execute and the five
output slices are set to zero at index `i`.

The kernel does not branch on slot identity and does not read pointers
beyond `slot_*[num_slots * n - 1]`. Adding a new slot does not change
the kernel's signature.

## Determinism Guarantees <!-- rq-76cb9922 -->

- All slot kernels and the combiner launch on the default stream of the
  same `Arc<CudaDevice>` carried by `ParticleBuffers`.
- The slot order produced by `ForceField::new` is deterministic and
  identical across runs with the same config.
- Each slot writes into its assigned row of the five flat slot-output
  buffers; rows are disjoint and written by exactly one slot.
- The combiner's left-to-right summation across slots is fixed by the
  slot order, so per-atom force, potential-energy, and virial values are
  byte-reproducible.

## Out of Scope <!-- rq-e448909a -->

- A user-supplied DSL for custom potentials. Implementing `Potential`
  is a Rust source-code change; potentials are not loaded from
  configuration or shared libraries.
- Multi-time-step (RESPA) force splitting; every slot runs every step.
- Long-range Ewald / PME electrostatics; the framework today only
  evaluates short-range potentials.
- Per-slot streams or async overlap of contribution and reduction
  kernels.
- Mid-run reconfiguration of slot membership. The slot list is fixed
  at `ForceField::new` and never modified.
- Slot ordering being user-configurable. The order is fixed in
  `ForceField::new`.
- Wiring the per-particle `potential_energies` and `virials` aggregates
  into log output, trajectory output, or pressure-coupling barostats.
  The framework produces the per-particle aggregates each step; the
  consumers of those aggregates (log writer, pressure logger, NPT
  barostat) are documented in their own files.
- Full virial-tensor accumulation. The framework's per-pair virial is
  the scalar trace `r_ij · F_ij`; per-component virial accumulation
  (xx, yy, zz, xy, xz, yz) is not in scope.

---

## Gherkin Scenarios <!-- rq-37ccfc1f -->

```gherkin
Feature: Pluggable potential slot framework

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called

  # --- Construction ---

  @rq-56c8a238
  Scenario: Construct a ForceField with LennardJones only
    Given a particle_count of 4
    And one [[pair_interactions]] entry for ("Ar","Ar")
    And no bond list and no bond types
    When ForceField::new(device, 4, sim_box, &pair_interactions, &[], &empty_bonds, &empty_excl, &nl_config) is called
    Then it returns Ok(force_field)
    And force_field.slots has length 1
    And force_field.slots[0].label() == "lennard_jones"

  @rq-3de16ce0
  Scenario: Construct a ForceField with LennardJones and MorseBonded
    Given a particle_count of 4
    And one [[pair_interactions]] entry for ("Ar","Ar")
    And one [[bond_types]] entry "CC" with potential="morse" and valid Morse parameters
    And a BondList with at least one bond of type "CC"
    And an ExclusionList consistent with the bonds
    When ForceField::new(...) is called
    Then it returns Ok(force_field)
    And force_field.slots has length 2
    And force_field.slots[0].label() == "lennard_jones"
    And force_field.slots[1].label() == "morse_bonded"

  @rq-0f34d11b
  Scenario: Construct a ForceField with bond_types declared but no bonds
    Given a particle_count of 4
    And one [[pair_interactions]] entry
    And one [[bond_types]] entry
    And an empty BondList
    When ForceField::new(...) is called
    Then it returns Ok(force_field)
    And force_field.slots has length 1
    And force_field.slots[0].label() == "lennard_jones"

  @rq-60f445b2
  Scenario: Construct a ForceField with zero slots
    Given a particle_count of 4
    And no [[pair_interactions]] entries
    And no bonds
    When ForceField::new(device, 4, sim_box, &[], &[], &empty_bonds, &empty_excl, &nl_config) is called
    Then it returns Ok(force_field)
    And force_field.slots is empty
    And force_field.slot_forces_x, slot_forces_y, slot_forces_z have length 0

  @rq-455db9c2
  Scenario: Slot accumulator buffers are sized num_slots * particle_count
    Given a particle_count of 8
    And a config producing 2 slots
    When ForceField::new(...) is called
    Then force_field.slot_forces_x has length 16
    And force_field.slot_forces_y has length 16
    And force_field.slot_forces_z has length 16

  @rq-c525ee79
  Scenario: Construct an empty (N=0) ForceField with potentials configured
    Given a particle_count of 0
    And a config producing 2 slots
    When ForceField::new(...) is called
    Then it returns Ok(force_field)
    And force_field.slot_forces_x, slot_forces_y, slot_forces_z have length 0

  @rq-c170c0b7
  Scenario: Reject construction when two slots share the same label
    Given a constructed ForceField scenario in which two slots would return
      the same label() value
    When ForceField::new(...) is called
    Then it returns Err(ForceFieldError::DuplicateLabel(_))

  # --- Force evaluation pipeline ---

  @rq-32e981cc
  Scenario: step() on a ForceField with only LennardJones writes LJ forces to forces_*
    Given a constructed ForceField with the LennardJones slot only
    And a ParticleBuffers with particle_count() == 2 placed so the LJ force is non-zero
    When force_field.step(&mut buffers, &sim_box, &mut timings) is called
    And particle_buffers is downloaded
    Then forces_x is non-zero in the expected pattern
    And forces_y, forces_z are consistent with the closed-form LJ result
    And timings reports counts for KernelStage::LjPairForce, ReducePairForces,
      and AccumulateForces

  @rq-df3a50f6
  Scenario: step() on a ForceField with both slots sums LJ and Morse
    Given a constructed ForceField with both slots
    And a ParticleBuffers and bond configuration where the LJ force and the Morse
      force on atom 0 are known a priori
    When force_field.step(&mut buffers, &sim_box, &mut timings) is called
    And particle_buffers is downloaded
    Then forces_x[0] equals lj_force_x_on_0 + morse_force_x_on_0 within f32 round-off

  @rq-fc7b1565
  Scenario: step() on a ForceField with zero slots writes zeros to forces_*
    Given a constructed ForceField with force_field.slots.is_empty()
    And a ParticleBuffers with particle_count() == 4 and arbitrary prior contents in forces_*
    When force_field.step(&mut buffers, &sim_box, &mut timings) is called
    And particle_buffers is downloaded
    Then forces_x, forces_y, forces_z are all zero
    And timings reports count==1 for KernelStage::AccumulateForces
    And timings reports count==0 for every slot-specific KernelStage

  @rq-de47c1ac
  Scenario: step() with N=0 launches no kernels
    Given a ForceField constructed with particle_count == 0
    When force_field.step(...) is called
    Then it returns Ok(())
    And timings.finalize() reports zero samples for every KernelStage

  @rq-7d8485b3
  Scenario: Each slot writes into its own row of the slot-output buffers
    Given a constructed ForceField with two slots and particle_count == 3
    When force_field.step(...) is called
    And slot_forces_x is downloaded
    Then entries [0, 1, 2] equal slot 0's per-particle force_x output
    And entries [3, 4, 5] equal slot 1's per-particle force_x output

  # --- Trait dispatch ---

  @rq-a9642241
  Scenario: Adding a new Potential implementation requires no edits to ForceField or accumulate_forces
    Given a hypothetical third Potential implementation `Buckingham` with label() == "buckingham"
    And ForceField::new is extended to construct it after MorseBonded when the config provides Buckingham parameters
    When ForceField::new(...) is called with a config that activates all three slots
    Then force_field.slots has length 3
    And the accumulate_forces kernel binary is unchanged
    And the SlotOutputView passed to Buckingham's reduce points at row 2 of the slot-output buffers

  # --- Reproducibility ---

  @rq-c8e5b14e
  Scenario: Two independent runs with identical inputs are byte-identical
    Given two independently-constructed ForceFields with identical parameters
    And two ParticleBuffers built from byte-identical ParticleStates of N=64
    When force_field.step(...) is called on each
    And the two ParticleBuffers are downloaded
    Then run A's forces_x, forces_y, forces_z agree byte-for-byte with run B's

  # --- Combiner correctness ---

  @rq-a5aa743e
  Scenario: Combiner sums slot rows in slot order
    Given a ForceField with two slots whose slot_forces_x rows are
      row 0 = [1.0, 2.0] and row 1 = [10.0, 20.0]
    When the combiner runs with num_slots = 2 and n = 2
    Then forces_x equals [11.0, 22.0]

  @rq-3e9217e2
  Scenario: Combiner with num_slots == 0 writes zeros
    Given a ForceField with zero slots and particle_count == 4
    When the combiner runs with num_slots = 0 and n = 4
    Then forces_x, forces_y, forces_z are all zero

  @rq-82acb52f
  Scenario: Combiner is a single-threaded write per output element
    Given a ForceField with two slots whose slot_forces_* rows are known
    When force_field.step(...) is called twice on identical inputs
    Then the resulting forces_* agree byte-for-byte across the two calls

  # --- Shared neighbor list ---

  @rq-b33cf896
  Scenario: ForceField with a short-range potential owns a shared neighbor list
    Given a ForceField with one LennardJones slot in CellList mode
    When ForceField::new completes
    Then ForceField::neighbor_list is Some(_)
    And the shared NeighborListState's max_neighbors equals the config value

  @rq-433c972f
  Scenario: ForceField with only a bonded potential owns no neighbor list
    Given a ForceField with one MorseBonded slot (and no pair_interactions)
    When ForceField::new completes
    Then ForceField::neighbor_list is None

  @rq-81e84c73
  Scenario: ForceFieldContext exposes the shared neighbor list to contribute
    Given a ForceField with a LennardJones slot in any mode
    And a stub Potential whose contribute() records the value of cx.neighbor_list
    When ForceField::step is called
    Then the stub records `Some(_)` (the same NeighborListState reference the LJ slot uses)

  @rq-e39d0ed8
  Scenario: max_cutoff aggregation determines the neighbor-list radius
    Given two short-range Potential implementations reporting max_cutoff() = Some(2.0) and Some(5.0)
    And NeighborListConfig::CellList { max_neighbors, r_skin }
    When ForceField::new constructs the shared neighbor list
    Then the neighbor list's r_search equals 5.0 + r_skin

  @rq-47540d14
  Scenario: A bonded-only ForceField step launches no neighbor-list kernels
    Given a ForceField whose only slot returns max_cutoff() = None
    When ForceField::step is called
    Then timings reports zero samples for KernelStage::NeighborDisplacementSquared
    And timings reports zero samples for KernelStage::NeighborListBuild

  # --- Per-particle energy and virial outputs ---

  @rq-531faea9
  Scenario: ForceField with N=4 LJ-only step populates potential_energies and virials
    Given a constructed ForceField with one LennardJones slot and N=4
    And a ParticleState placed so the LJ contributions are non-zero
    When ForceField::step is called
    And ParticleBuffers is downloaded
    Then potential_energies is finite and non-zero in the expected pattern
    And virials is finite and non-zero in the expected pattern

  @rq-a85e8216
  Scenario: Slot-output buffers have five flat arrays sized num_slots * N
    Given a ForceField with two slots and particle_count = 8
    Then ForceField::slot_forces_x, slot_forces_y, slot_forces_z,
      slot_energies, and slot_virials each have length 16

  @rq-3d38868e
  Scenario: Combiner sums slot energies and virials in slot order
    Given a ForceField with two slots whose slot_energies rows are
      row 0 = [1.0, 2.0] and row 1 = [10.0, 20.0]
    And whose slot_virials rows are row 0 = [0.5, 1.0] and row 1 = [5.0, 10.0]
    When the combiner runs with num_slots = 2 and n = 2
    Then particle_buffers.potential_energies equals [11.0, 22.0]
    And particle_buffers.virials equals [5.5, 11.0]

  @rq-c0f2daca
  Scenario: Zero-slot step writes zeros to potential_energies and virials
    Given a constructed ForceField with force_field.slots.is_empty()
    And a ParticleBuffers with particle_count() == 4 and arbitrary prior contents in
      potential_energies and virials
    When ForceField::step is called
    And ParticleBuffers is downloaded
    Then potential_energies and virials are each [0.0, 0.0, 0.0, 0.0]

  @rq-db3b3d5e
  Scenario: System total potential energy equals sum of particle shares
    Given a constructed ForceField with one LennardJones slot and N atoms
    When ForceField::step is called
    And the per-particle potential_energies are downloaded
    Then their sum equals the expected total LJ energy of the configuration within f32 round-off

  @rq-7fe57a77
  Scenario: System total scalar virial equals sum of particle shares
    Given a constructed ForceField with one LennardJones slot and N atoms
    When ForceField::step is called
    And the per-particle virials are downloaded
    Then their sum equals Σ_{i<j within cutoff} r_ij · F_ij within f32 round-off
```
