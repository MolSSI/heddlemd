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
| `morse-bonded` | bonded pairwise via `[[bond_types]]` + `.bonds` file | `morse-bonded.md` |

Slots are present in the `ForceField` according to the config:

- The `LennardJones` slot is present when `[[pair_interactions]]` contains
  at least one entry.
- The `MorseBonded` slot is present when the config supplies a
  `bonds = "..."` path *and* at least one `[[bond_types]]` entry uses
  `potential = "morse"`.

A `ForceField` with zero slots is a valid configuration. `step()` writes
zeros into `particle_buffers.forces_*` and returns without launching any
contribution or reduction kernels.

When multiple slots are present, they appear in the `ForceField`'s slot
list in a fixed order determined at construction:

1. `LennardJones`
2. `MorseBonded`

Future potentials are inserted into this order at fixed positions defined
in `ForceField::new`.

## Force Evaluation Pipeline <!-- rq-7bab5c1e -->

The runner's force evaluation does the following per step:

1. **Contribution kernels.** For each slot, in slot order, invoke
   `Potential::contribute`. The slot fills its private intermediate
   buffers (e.g. a `PairBuffer` or a `BondPairBuffer`); the framework does
   not inspect them.
2. **Per-slot reductions.** For each slot, in slot order, invoke
   `Potential::reduce`, passing a `SlotForceView` that points to the
   slot's assigned row of the framework's flat slot-accumulator buffers.
   The reduction overwrites that row with the slot's per-particle
   reduced force contribution.
3. **Combiner.** Run `accumulate_forces` once. The combiner reads the
   flat slot-accumulator buffers and writes
   `forces_*[i] = sum over k in 0..num_slots of slot_forces_*[k * n + i]`.
   The summation is left-to-right in `k`; each thread handles one `i`.

Identical runs on the same GPU with the same config produce byte-identical
`forces_*` and therefore byte-identical trajectories.

## Slot Accumulator Buffers <!-- rq-cd28340e -->

`ForceField` owns three contiguous device buffers, one per axis, named
`slot_forces_x`, `slot_forces_y`, `slot_forces_z`. Each has length
`num_slots * particle_count`. The row for slot `k` is the half-open range
`[k * particle_count, (k + 1) * particle_count)` within each buffer.

After slot `k`'s `reduce()` returns, row `k` contains that slot's
per-particle reduced force along that axis. The combiner reads every row
and sums in slot order.

Memory cost: `3 * num_slots * particle_count * 4 bytes`. For
`particle_count = 10⁴` and four slots, this is ~480 KB — negligible.
For `particle_count = 10⁵` and four slots, ~4.8 MB.

When `num_slots == 0` the three buffers have length zero. When
`particle_count == 0` the three buffers have length zero regardless of
`num_slots`.

## Empty State <!-- rq-aa52268c -->

When `particle_count == 0`, every slot's `contribute` and `reduce` methods
early-return without launching, and the combiner returns without
launching. `ForceField::step` returns `Ok(())` having done no GPU work.

When `num_slots == 0`, the combiner kernel still launches (with
`num_slots = 0`) and writes zeros to every entry of
`particle_buffers.forces_*`. The contribution and reduction phases launch
no kernels. `ForceField::step` returns `Ok(())`.

When a slot's *input list* is empty (e.g. a `MorseBonded` slot constructed
with `bonds.is_empty()`), the slot's `contribute` does not launch any
contribution kernel, the slot's `reduce` writes zeros into its assigned
row of the slot-accumulator buffers, and the rest of the pipeline runs
normally. (The combiner reads every row unconditionally; empty-input
slots must carry valid zeros in their row.)

## Feature API <!-- rq-0da87ca1 -->

### Types <!-- rq-e4960f89 -->

- `Potential` — object-safe trait implemented by every slot. <!-- rq-67ebf3b1 -->

  ```rust
  pub trait Potential: std::fmt::Debug + Send {
      fn label(&self) -> &'static str;

      fn contribute(
          &mut self,
          buffers: &ParticleBuffers,
          sim_box: &SimulationBox,
          timings: &mut Timings,
      ) -> Result<(), ForceFieldError>;

      fn reduce(
          &mut self,
          output: SlotForceView<'_>,
          timings: &mut Timings,
      ) -> Result<(), ForceFieldError>;
  }
  ```

  - `label` returns a short, stable, lower-snake-case identifier (for
    example `"lennard_jones"`, `"morse_bonded"`) used in error messages
    and diagnostic output. Two slots in the same `ForceField` must have
    distinct labels.
  - `contribute` runs the slot's contribution kernel(s) against the
    current `ParticleBuffers` and `SimulationBox`. Implementations may
    read from `buffers` but must not write to it.
  - `reduce` writes exactly `buffers.particle_count()` floats per axis
    into the slices referenced by `output`. After `reduce` returns,
    those slices hold the slot's per-particle force contribution.

  Implementations are responsible for emitting their own
  `KernelStage` start/stop events through `timings`.

- `SlotForceView<'a>` — three exclusive references to per-axis output <!-- rq-304b191b -->
  slices, each of length `particle_count`. Constructed by `ForceField`
  and passed into `Potential::reduce`. Implementations must treat the
  slices as write-only output buffers.

  ```rust
  pub struct SlotForceView<'a> {
      pub x: CudaViewMut<'a, f32>,
      pub y: CudaViewMut<'a, f32>,
      pub z: CudaViewMut<'a, f32>,
  }
  ```

  Each field is a `CudaViewMut` of length `particle_count` onto the
  corresponding row of the framework's flat slot-accumulator buffer.
  The view borrows the framework's storage for the duration of the
  `reduce` call.

- `LennardJonesState` — implements `Potential` with `label() == "lennard_jones"`. <!-- rq-af2d1628 -->
  Owns the slot's `PairBuffer`, `LennardJonesParameters`, and the
  `DeviceExclusionList` (see `bonds.md`). Its internal state is one of
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
  `SlotForceView` it receives.

- `MorseBondedState` — implements `Potential` with `label() == "morse_bonded"`. <!-- rq-2361f2b8 -->
  Owns the slot's `BondPairBuffer`, the bond index/offset tables, and
  the per-bond-type parameter table. Construction requires a non-empty
  bond list; see `morse-bonded.md`. Its reduction writes per-particle
  output into the `SlotForceView` it receives.

- `ForceField` — handle owning the slot collection and the flat slot-accumulator buffers. <!-- rq-684a29f1 -->

  Fields:
  - `device: Arc<CudaDevice>`
  - `slots: Vec<Box<dyn Potential>>` — in fixed evaluation order. May
    be empty.
  - `slot_forces_x: CudaSlice<f32>` — length `slots.len() * N`.
  - `slot_forces_y: CudaSlice<f32>` — length `slots.len() * N`.
  - `slot_forces_z: CudaSlice<f32>` — length `slots.len() * N`.

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

- `ForceField::new(device: Arc<CudaDevice>, particle_count: usize, sim_box: &SimulationBox, pair_interactions: &[PairInteractionConfig], bond_types: &[BondTypeConfig], bond_list: &BondList, exclusion_list: &ExclusionList, neighbor_list_config: &NeighborListConfig) -> Result<ForceField, ForceFieldError>` <!-- rq-79938dbf -->
  - Constructs the slot collection in fixed evaluation order:
    - Appends a `LennardJonesState` slot when `pair_interactions` is
      non-empty.
    - Appends a `MorseBondedState` slot when `bond_list.is_empty()` is
      false.
  - When both inputs are absent, returns a `ForceField` with
    `slots.len() == 0`.
  - Allocates the three flat slot-accumulator buffers of length
    `slots.len() * particle_count` on `device`. When either factor is
    zero, the allocations are length-zero.
  - The `LennardJones` slot is built in `AllPairs` or `CellList` mode
    according to `neighbor_list_config`. In `CellList` mode, the
    construction may return `ForceFieldError::NeighborList(_)` when the
    box is too small for the requested cell layout.
  - Returns `ForceFieldError::DuplicateLabel(_)` if two slots end up
    with the same `label()`.

- `ForceField::step(&mut self, buffers: &mut ParticleBuffers, sim_box: &SimulationBox, timings: &mut Timings) -> Result<(), ForceFieldError>` <!-- rq-3579df3b -->
  - For each slot in `self.slots`, in order, calls
    `slot.contribute(buffers, sim_box, timings)`.
  - For each slot in `self.slots`, in order, calls `slot.reduce(view,
    timings)` where `view` is a `SlotForceView` pointing into the row
    `slot_index * particle_count .. (slot_index + 1) * particle_count`
    of the flat slot-accumulator buffers.
  - Launches `accumulate_forces` once (with
    `KernelStage::AccumulateForces`). When `slots.is_empty()`,
    `accumulate_forces` still launches and writes zeros to
    `buffers.forces_*`.
  - Returns `Ok(())` on success.
  - Empty-state contract per *Empty State* above.

### Combiner Kernel <!-- rq-c0f98145 -->

`kernels/forces.cu` declares one `extern "C"` kernel:

```c
extern "C" __global__ void accumulate_forces(
    const float *slot_forces_x,    // shape [num_slots * n]
    const float *slot_forces_y,    // shape [num_slots * n]
    const float *slot_forces_z,    // shape [num_slots * n]
    unsigned int num_slots,
    float *forces_x,
    float *forces_y,
    float *forces_z,
    unsigned int n);
```

Each thread maps to one particle index
`i = blockIdx.x * blockDim.x + threadIdx.x` (block size 256, grid
`ceil(n / 256)`, no shared memory, default stream of
`particle_buffers.device`). The thread computes

```text
sum_x = 0
for k in 0..num_slots:
    sum_x += slot_forces_x[k * n + i]
forces_x[i] = sum_x
```

and analogously for `y` and `z`. The sum is performed left-to-right in
slot order, so identical inputs yield byte-identical outputs across runs.

When `num_slots == 0`, the inner loop does not execute and `forces_*[i]`
is set to zero.

The kernel does not branch on slot identity and does not read pointers
beyond `slot_forces_*[num_slots * n - 1]`. Adding a new slot does not
change the kernel's signature.

## Determinism Guarantees <!-- rq-76cb9922 -->

- All slot kernels and the combiner launch on the default stream of the
  same `Arc<CudaDevice>` carried by `ParticleBuffers`.
- The slot order produced by `ForceField::new` is deterministic and
  identical across runs with the same config.
- Each slot writes into its assigned row of the flat slot-accumulator
  buffers; rows are disjoint and written by exactly one slot.
- The combiner's left-to-right summation across slots is fixed by the
  slot order, so per-atom force values are byte-reproducible.

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
- Per-pair (force, energy, virial) output. The `Potential` trait
  reduces forces only; energy and virial accumulation are out of scope
  for this framework.

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
  Scenario: Each slot writes into its own row of the slot-accumulator buffers
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
    And the SlotForceView passed to Buckingham's reduce points at row 2 of the slot-accumulator buffers

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
```
