# Feature: Pluggable Potential Slot Framework <!-- rq-c1ea073b -->

The runner evaluates inter-particle forces through a fixed, named set of
*potential slots* assembled into a `ForceField`. Each slot owns its own
contribution kernel, private accumulator buffers, and reduction kernel; the
`ForceField`'s `step()` method runs every slot's kernels in a fixed order
and combines their accumulators into the particle state's `forces_*` arrays
via a deterministic combiner. The runner's force-evaluation step calls
`force_field.step(...)` once; it has no visibility into which potentials
participated.

The framework mirrors the integrator-slot pattern (see
`integration/framework.md`): adding a new potential means adding a new
variant to the `PotentialSlot` enum, a matching parameter set in the config,
and a per-variant kernel pair.

## Slots <!-- rq-cc73f184 -->

| `kind` | Variant | File |
| --- | --- | --- |
| `lennard-jones` | non-bonded pairwise via `[[pair_interactions]]` | `lj-pair-force.md` |
| `morse-bonded` | bonded pairwise via `[[bond_types]]` + `.bonds` file | `morse-bonded.md` |

Slots are present in the `ForceField` according to the config:

- `LennardJones` is always present (the existing `[[pair_interactions]]`
  declaration is required).
- `MorseBonded` is present when the config supplies a `bonds = "..."` path
  *and* at least one `[[bond_types]]` entry uses `potential = "morse"`.

Future non-bonded potentials are added as new `PotentialSlot` variants
alongside `LennardJones`; future bonded potentials reuse the
`MorseBonded` machinery through extra `[[bond_types]]` entries whose
`potential` field selects a different variant.

## Force Evaluation Pipeline <!-- rq-7bab5c1e -->

The runner's force evaluation does the following per step:

1. **Contribution kernels.** For each slot, in fixed order, run the
   contribution kernel that fills the slot's private buffer:
   - `LennardJones` → `lj_pair_force` fills the slot's `PairBuffer`.
   - `MorseBonded` → `morse_bond_force` fills the slot's `BondPairBuffer`.
2. **Per-slot reductions.** For each slot, in fixed order, reduce the
   private buffer into the slot's *private accumulator* (three
   `CudaSlice<f32>` of length `N`). The accumulator is overwritten on
   each reduction.
3. **Combiner.** Run `accumulate_forces`, which writes
   `forces_*[i] = sum over slots of slot.accumulator_*[i]` in a fixed
   slot order. Single-threaded per `i`; no atomics.

The slot order is `LennardJones` first, then `MorseBonded`. Identical
runs on the same GPU produce byte-identical `forces_*` and therefore
byte-identical trajectories.

## Per-Slot Private Accumulators <!-- rq-cd28340e -->

Each slot owns three `CudaSlice<f32>` of length `N` named
`accumulator_x`, `accumulator_y`, `accumulator_z`. The slot's reduction
kernel writes the per-atom force contribution into these buffers
(overwriting). The combiner kernel reads from every slot's accumulator
and writes to `particle_buffers.forces_*`.

Memory cost: `3 * N * 4 bytes * num_slots`. For `N = 10⁴` and the two
slots in v1, this is ~240 KB — negligible.

## Empty State <!-- rq-aa52268c -->

When `particle_count == 0`, every slot's contribution and reduction
kernels early-return without launching, and the combiner returns
without launching. `ForceField::step` returns `Ok(())` having done no
GPU work.

When a slot's *input list* is empty (e.g. `bonds.is_empty()` for
`MorseBonded`), the slot's contribution kernel does not launch, the
slot's accumulator is zeroed by a single-kernel zero-fill before the
combiner reads it, and the rest of the pipeline runs normally. (The
combiner reads slot accumulators unconditionally; empty-input slots
must carry valid zeros there.)

## Feature API <!-- rq-0da87ca1 -->

### Types <!-- rq-e4960f89 -->

- `PotentialSlot` — closed `enum`. Variants: <!-- rq-097c7d9c -->

  ```rust
  pub enum PotentialSlot {
      LennardJones(LennardJonesState),
      MorseBonded(MorseBondedState),
  }
  ```

  Variant payloads are public `Debug` types with private fields; their
  per-step methods are dispatched through `PotentialSlot`'s inherent
  impl.

- `LennardJonesState` — owns the slot's `PairBuffer`, `neighbor_counts` <!-- rq-af2d1628 -->
  device slice, `LennardJonesParameters`, the `ExclusionList` device
  representation (see `bonds.md`), and the three accumulator slices.

- `MorseBondedState` — owns the slot's `BondPairBuffer`, the bond <!-- rq-2361f2b8 -->
  index/offset tables, per-bond-type parameter table, and the three
  accumulator slices. Construction requires a non-empty bond list; see
  `morse-bonded.md`.

- `ForceField` — handle owning the ordered slot collection plus the <!-- rq-684a29f1 -->
  combiner kernel module.

  Fields:
  - `device: Arc<CudaDevice>`
  - `slots: Vec<PotentialSlot>` — in fixed evaluation order:
    `LennardJones` then `MorseBonded` when present.

- `ForceFieldError` — error type. Variants: <!-- rq-a2e20b02 -->
  - `Gpu(GpuError)` — CUDA driver / kernel-launch failure from any
    slot's kernel or the combiner.
  - `Timings(TimingsError)` — CUDA event recording failure.

### Functions and methods <!-- rq-17abcb76 -->

- `ForceField::new(device: Arc<CudaDevice>, particle_count: usize, sim_box: &SimulationBox, pair_interactions: &[PairInteractionConfig], bond_types: &[BondTypeConfig], bond_list: &BondList, exclusion_list: &ExclusionList) -> Result<ForceField, ForceFieldError>` <!-- rq-79938dbf -->
  - Constructs every slot in evaluation order. The `LennardJones` slot
    is always added; the `MorseBonded` slot is added when `bond_list`
    is non-empty.
  - Allocates per-slot private accumulators of length `particle_count`
    on `device`.
  - When `particle_count == 0`, allocations have length zero; the slot
    construction still succeeds.

- `ForceField::step(&mut self, buffers: &mut ParticleBuffers, sim_box: &SimulationBox, timings: &mut Timings) -> Result<(), ForceFieldError>` <!-- rq-3579df3b -->
  - Runs every slot's contribution kernel (with the slot's labelled
    `KernelStage` start/stop events).
  - Runs every slot's reduction kernel (with the slot's labelled
    `KernelStage`).
  - Runs the combiner kernel (with `KernelStage::AccumulateForces`).
  - Returns `Ok(())` on success.
  - Empty-state contract per *Empty State* above.

### Combiner Kernel <!-- rq-c0f98145 -->

`kernels/forces.cu` declares one `extern "C"` kernel:

```c
extern "C" __global__ void accumulate_forces(
    const float *slot0_x, const float *slot0_y, const float *slot0_z,
    const float *slot1_x, const float *slot1_y, const float *slot1_z,
    unsigned int n_slots,
    unsigned int present_slots_bitmask,
    float *forces_x, float *forces_y, float *forces_z,
    unsigned int n);
```

The runner only ever passes the slot count and presence bitmask for the
slots currently in the `ForceField`. For v1 the maximum slot count is
two; the kernel reads pointers only for the slots indicated by the
bitmask. Adding a new slot will widen the signature once needed; until
then the kernel hard-codes two slot inputs.

Each thread maps to one particle index `i = blockIdx.x * blockDim.x +
threadIdx.x` (block size 256, grid `ceil(n / 256)`, no shared memory,
default stream of `particle_buffers.device`). The thread computes
`forces_x[i] = sum over present slots of slot_k.accumulator_x[i]` and
similarly for `y` and `z`. The sum is performed left-to-right in slot
order, so identical inputs yield byte-identical outputs across runs.

## Determinism Guarantees <!-- rq-76cb9922 -->

- All slot kernels and the combiner launch on the default stream of the
  same `Arc<CudaDevice>` carried by `ParticleBuffers`.
- The slot order at construction is deterministic and identical across
  runs with the same config.
- The combiner's left-to-right slot summation is fixed by the slot
  order, so per-atom force values are byte-reproducible.

## Out of Scope <!-- rq-e448909a -->

- A user-supplied DSL for custom potentials. Slots are a closed set;
  new potentials land as new enum variants.
- Multi-time-step (RESPA) force splitting; every slot runs every step.
- Long-range Ewald / PME electrostatics; the framework today only
  evaluates short-range potentials.
- Per-slot streams or async overlap of contribution and reduction
  kernels.
- Mid-run reconfiguration of slot membership.
- Slot ordering being user-configurable. The order is fixed in code.

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
    When ForceField::new(device, 4, sim_box, &pair_interactions, &[], &empty_bonds, &empty_excl) is called
    Then it returns Ok(force_field)
    And force_field.slots has length 1
    And force_field.slots[0] matches PotentialSlot::LennardJones(_)

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
    And force_field.slots[0] matches PotentialSlot::LennardJones(_)
    And force_field.slots[1] matches PotentialSlot::MorseBonded(_)

  @rq-0f34d11b
  Scenario: Construct a ForceField with bond_types declared but no bonds
    Given a particle_count of 4
    And one [[bond_types]] entry
    And an empty BondList
    When ForceField::new(...) is called
    Then it returns Ok(force_field)
    And force_field.slots has length 1
    And force_field.slots[0] matches PotentialSlot::LennardJones(_)

  @rq-c525ee79
  Scenario: Construct an empty (N=0) ForceField
    Given a particle_count of 0
    When ForceField::new(device, 0, ..., empty_bonds, empty_excl) is called
    Then it returns Ok(force_field)
    And every slot's accumulator buffers have length 0

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

  @rq-de47c1ac
  Scenario: step() with N=0 launches no kernels
    Given a ForceField constructed with particle_count == 0
    When force_field.step(...) is called
    Then it returns Ok(())
    And timings.finalize() reports zero samples for every KernelStage

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
  Scenario: Combiner sums slot accumulators in slot order
    Given a ForceField with LennardJones (accumulator_x_lj = [1, 2]) and
      MorseBonded (accumulator_x_morse = [10, 20])
    When the combiner runs
    Then forces_x equals [11, 22]

  @rq-82acb52f
  Scenario: Combiner is a single-threaded write per output element
    Given a ForceField with two slots whose accumulators are known
    When force_field.step(...) is called twice on identical inputs
    Then the resulting forces_* agree byte-for-byte across the two calls
```
