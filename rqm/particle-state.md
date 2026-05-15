# Feature: SoA Particle State and GPU Buffers <!-- rq-f04e473c -->

The simulation's particle data is stored in two coordinated forms: a host-side
structure of arrays (`ParticleState`) backed by `Vec<f32>` and `Vec<u32>`
fields, and a device-side mirror (`ParticleBuffers`) backed by `cudarc`
`CudaSlice` allocations. Each per-particle quantity is split into one array
per Cartesian component so that GPU threads accessing consecutive particles
read consecutive memory addresses (coalesced loads).

`ParticleState` carries the canonical initial conditions and any data the host
needs for I/O. `ParticleBuffers` carries the canonical live state during a
simulation: kernels read from and write to the device buffers. Movement
between the two is explicit — the host calls `ParticleBuffers::upload` to push
host data to the device and `ParticleState::download_from` to pull device data
back into host arrays.

## Data Layout <!-- rq-add95876 -->

`ParticleState` holds these fields, each a contiguous SoA array of length
`particle_count()`:

```
positions_x:        Vec<f32>
positions_y:        Vec<f32>
positions_z:        Vec<f32>
images_x:           Vec<i32>
images_y:           Vec<i32>
images_z:           Vec<i32>
velocities_x:       Vec<f32>
velocities_y:       Vec<f32>
velocities_z:       Vec<f32>
forces_x:           Vec<f32>
forces_y:           Vec<f32>
forces_z:           Vec<f32>
potential_energies: Vec<f32>
virials:            Vec<f32>
masses:             Vec<f32>
type_indices:       Vec<u32>
particle_ids:       Vec<u32>
```

`positions_x[i]`, `positions_y[i]`, and `positions_z[i]` carry the
*wrapped* position of particle `i`: each component is in
`[-L_a / 2, +L_a / 2)` for the corresponding box edge `L_a`. The
companion image triple `(images_x[i], images_y[i], images_z[i])`
records how many full periods the particle has crossed since the start
of the run. The *unwrapped* position of particle `i` along axis `a`
is `positions_a[i] + images_a[i] * L_a`. Image flags start at zero (or
at the values supplied via the init file) and are advanced by the
integrator's drift kernels whenever the wrapped position crosses a
`±L/2` boundary.

`potential_energies[i]` holds particle `i`'s share of the system's total
potential energy after a force-evaluation step (the sum of `U_ij / 2` over
its neighbours plus the sum of `U_k / 2` over the bonds it participates
in, where `U_ij`/`U_k` are pair/bond potential energies). Summing
`potential_energies` over all particles yields the system's total
potential energy with each pair/bond counted exactly once. Initial
allocation is zero-filled; the buffer is overwritten each step by the
force pipeline.

`virials[i]` holds particle `i`'s share of the system's total scalar
virial, `Σ_{ij neighbours} r_ij · F_ij / 2`. Summing `virials` over all
particles yields the system's total scalar virial. Initial allocation is
zero-filled; the buffer is overwritten each step by the force pipeline.

`type_indices[i]` is the index of particle `i`'s entry in
`Config::particle_types` (see `io/config-schema.md`); the index is in the
range `0..config.particle_types.len()`. ParticleState does not carry the
particle-type table itself; the host enforces that every `type_indices[i]`
falls within the declared range at the point where the state is built
from the parsed init file.

`ParticleBuffers` holds the device-side mirror: one `CudaSlice<f32>` per
`f32` host array, one `CudaSlice<i32>` per image-flag axis, one
`CudaSlice<u32>` for `particle_ids`, and one `CudaSlice<u32>` for
`type_indices`. The two structures have identical particle counts;
allocation sizes match exactly (no extra capacity).

All arrays of the same particle state must have the same length. This
invariant is checked at construction time and again at every upload/download
call.

## Feature API <!-- rq-c9016748 -->

### Types <!-- rq-08066bdf -->

- `ParticleState` — host-side SoA state. All seventeen per-particle arrays <!-- rq-3766be01 -->
  are declared as `pub` fields so callers may iterate, index, and mutate
  them directly. Length consistency between fields is the caller's
  responsibility while the state is held on the host; it is re-validated
  at every upload or download.

- `ParticleBuffers` — device-side mirror. Holds a `CudaSlice<f32>` for each <!-- rq-4a8de06c -->
  `f32` host array, a `CudaSlice<i32>` for each of `images_x`, `images_y`,
  and `images_z`, and a `CudaSlice<u32>` for each of `particle_ids` and
  `type_indices`. Each buffer is exposed as a `pub` field so kernel launch
  sites can pass `&CudaSlice` references directly. Also carries an
  `Arc<CudaDevice>` for upload/download bookkeeping.

- `ParticleStateError` — error type returned by construction and host↔device <!-- rq-bec7b519 -->
  transfer. Variants:
  - `LengthMismatch { array: &'static str, expected: usize, actual: usize }`
    — an input array (during construction), a host array (during upload), or
    a host array (during download) has a length other than the established
    particle count. `array` names the offending array (e.g. `"positions_y"`,
    `"images_z"`, `"particle_ids"`, `"type_indices"`,
    `"potential_energies"`, `"virials"`).
  - `DuplicateParticleId(u32)` — caller-supplied IDs contain at least one
    duplicate; the variant reports one offending value.
  - `Gpu(GpuError)` — a CUDA driver operation failed during upload or
    download. Wraps the existing `crate::gpu::GpuError`.

### Functions and methods <!-- rq-7206ab76 -->

- `ParticleState::new(positions_x: Vec<f32>, positions_y: Vec<f32>, positions_z: Vec<f32>, velocities_x: Vec<f32>, velocities_y: Vec<f32>, velocities_z: Vec<f32>, masses: Vec<f32>, type_indices: Vec<u32>, ids: Option<Vec<u32>>, images: Option<(Vec<i32>, Vec<i32>, Vec<i32>)>) -> Result<ParticleState, ParticleStateError>` <!-- rq-5e0598cb -->
  - The particle count is taken from `positions_x.len()`.
  - Validates that `positions_y`, `positions_z`, `velocities_x`,
    `velocities_y`, `velocities_z`, `masses`, and `type_indices` all have
    the same length as `positions_x`. Returns `LengthMismatch` on the first
    offending array (checked in declaration order).
  - If `ids` is `Some(v)`, validates that `v.len()` matches the particle
    count and that `v` contains no duplicates; returns `LengthMismatch` or
    `DuplicateParticleId` accordingly.
  - If `ids` is `None`, the constructor populates `particle_ids` with
    `0..particle_count` cast to `u32`.
  - If `images` is `Some((ix, iy, iz))`, validates that each of `ix`, `iy`,
    and `iz` has length `particle_count`; returns `LengthMismatch` on the
    first offending array (checked in declaration order:
    `"images_x"`, `"images_y"`, `"images_z"`). The constructor stores the
    arrays as `images_x`, `images_y`, `images_z`.
  - If `images` is `None`, the constructor populates `images_x`,
    `images_y`, and `images_z` with `Vec<i32>` of length `particle_count`,
    zero-initialised.
  - Allocates `forces_x`, `forces_y`, `forces_z`, `potential_energies`,
    and `virials` as `Vec<f32>` of length `particle_count`,
    zero-initialised.
  - A particle count of zero is permitted: the constructor returns a state
    whose every field is an empty `Vec`.
  - Does not validate numerical content (NaN, infinity, sign, magnitude are
    accepted as-is), `type_indices` values (range checks happen in the
    runner against the parsed `Config::particle_types` length), or the
    spatial consistency of position / image pairs (the constructor does
    not enforce `positions_a[i] ∈ [-L_a/2, +L_a/2)`; that invariant is
    re-established by the integrator's drift kernels on the first step).

- `ParticleState::particle_count(&self) -> usize` <!-- rq-ac035b90 -->
  - Returns `positions_x.len()`. Callers are expected to keep the other
    arrays at the same length.

- `ParticleBuffers::new(device: Arc<CudaDevice>, state: &ParticleState) -> Result<ParticleBuffers, ParticleStateError>` <!-- rq-b09032cb -->
  - Allocates one device buffer per host array, sized to
    `state.particle_count()`.
  - Validates that every host array has length `state.particle_count()`;
    returns `LengthMismatch` otherwise.
  - Copies all seventeen host arrays into the corresponding device buffers.
  - Returns the populated `ParticleBuffers` on success.

- `ParticleBuffers::particle_count(&self) -> usize` <!-- rq-18411920 -->
  - Returns the per-buffer length established at construction.

- `ParticleBuffers::upload(&mut self, state: &ParticleState) -> Result<(), ParticleStateError>` <!-- rq-179ed985 -->
  - Validates that `state.particle_count()` equals `self.particle_count()`
    and that every host array has that length; returns `LengthMismatch`
    otherwise.
  - Copies all seventeen host arrays into the existing device buffers
    in-place.

- `ParticleState::download_from(&mut self, buffers: &ParticleBuffers) -> Result<(), ParticleStateError>` <!-- rq-9a19bfa3 -->
  - Validates that `self.particle_count()` equals `buffers.particle_count()`
    and that every host array has that length; returns `LengthMismatch`
    otherwise.
  - Copies every device buffer into the corresponding host `Vec` in place,
    overwriting prior contents. No reallocation occurs.

## Construction Details <!-- rq-ef7b719d -->

- `ParticleState::new` consumes its input `Vec`s; no copies are made of the
  host arrays at construction.
- Length validation is performed in declaration order
  (`positions_y`, `positions_z`, `velocities_x`, `velocities_y`,
  `velocities_z`, `masses`, `type_indices`, then `ids` when `Some`,
  then `images_x`, `images_y`, `images_z` when `images` is `Some`).
- Duplicate-ID detection is performed using a hash set; expected complexity
  is O(N) for N particles.

## Host ↔ Device Transfer Semantics <!-- rq-9026181d -->

- `ParticleBuffers::new` performs both allocation and the initial copy in a
  single call.
- `ParticleBuffers::upload` and `ParticleState::download_from` operate on
  pre-allocated buffers and pre-allocated host `Vec`s; they never reallocate.
- All transfers are synchronous; on return, the destination side reflects
  the source side at the moment of the call.
- Per-array uploads/downloads are not exposed in this feature. Callers that
  need finer-grained transfers reach for `cudarc` directly via the `pub`
  `CudaSlice` fields.

## Out of Scope <!-- rq-0087f355 -->

- Periodic boundary conditions and the simulation box.
- The pair buffer used by force kernels (`pair_forces_*`).
- The reference position array used by the skin-distance neighbor-list
  rebuild check.
- Spatial-hash cell indices and neighbor lists.
- The `f64` precision feature flag (everything in this feature uses `f32`
  for scalar quantities).
- Numerical validation (NaN, infinity, negative masses).
- Trajectory I/O.
- Force computation, integration, and the simulation loop.

---

## Gherkin Scenarios <!-- rq-9b0aad2c -->

```gherkin
Feature: SoA particle state and GPU buffers

  # --- Construction ---

  @rq-81f4ec9d
  Scenario: Construct with matching arrays and default IDs
    Given seven Vec<f32> of length 4 for positions_x/y/z, velocities_x/y/z, and masses
    And a Vec<u32> type_indices of length 4 with values [0, 0, 0, 0]
    When ParticleState::new(...) is called with ids=None
    Then it returns Ok(state)
    And state.particle_count() is 4
    And state.particle_ids equals [0, 1, 2, 3]
    And state.type_indices equals [0, 0, 0, 0]
    And state.forces_x, state.forces_y, and state.forces_z are each Vec<f32> of length 4 with every element 0.0

  @rq-2bbc4121
  Scenario: Construct with matching arrays and explicit unique IDs
    Given seven Vec<f32> of length 3 for positions_x/y/z, velocities_x/y/z, and masses
    And a Vec<u32> type_indices with values [0, 1, 0]
    And a Vec<u32> with values [10, 20, 30]
    When ParticleState::new(...) is called with ids=Some([10, 20, 30])
    Then it returns Ok(state)
    And state.particle_ids equals [10, 20, 30]
    And state.type_indices equals [0, 1, 0]

  @rq-c22483b4
  Scenario: Construct an empty state
    Given every input Vec is empty
    When ParticleState::new(...) is called with ids=None
    Then it returns Ok(state)
    And state.particle_count() is 0
    And every Vec field of state has length 0

  @rq-91aa1f1c
  Scenario: Construct an empty state with explicit empty IDs
    Given every input Vec is empty
    When ParticleState::new(...) is called with ids=Some(vec![])
    Then it returns Ok(state)
    And state.particle_count() is 0

  @rq-1e8b3c79
  Scenario: Reject when positions_y has the wrong length
    Given positions_x has length 4
    And positions_y has length 3
    And every other input array has length 4
    When ParticleState::new(...) is called
    Then it returns Err(ParticleStateError::LengthMismatch { array: "positions_y", expected: 4, actual: 3 })

  @rq-ce89d4a4
  Scenario: Reject when masses has the wrong length
    Given positions_x, positions_y, positions_z, velocities_x, velocities_y, velocities_z each have length 4
    And masses has length 5
    And type_indices has length 4
    When ParticleState::new(...) is called with ids=None
    Then it returns Err(ParticleStateError::LengthMismatch { array: "masses", expected: 4, actual: 5 })

  @rq-790c1f86
  Scenario: Reject when type_indices has the wrong length
    Given positions_x, positions_y, positions_z, velocities_x, velocities_y, velocities_z, masses each have length 4
    And type_indices has length 3
    When ParticleState::new(...) is called with ids=None
    Then it returns Err(ParticleStateError::LengthMismatch { array: "type_indices", expected: 4, actual: 3 })

  @rq-391cb266
  Scenario: Reject when explicit IDs have the wrong length
    Given every f32 input array has length 4
    And ids=Some([0, 1])
    When ParticleState::new(...) is called
    Then it returns Err(ParticleStateError::LengthMismatch { array: "particle_ids", expected: 4, actual: 2 })

  @rq-4b38148c
  Scenario: Reject duplicate explicit IDs
    Given every f32 input array has length 4
    And ids=Some([7, 1, 7, 3])
    When ParticleState::new(...) is called
    Then it returns Err(ParticleStateError::DuplicateParticleId(7))

  @rq-9447dfcf
  Scenario: NaN values are accepted at construction
    Given positions_x contains f32::NAN at index 0
    And every other input array is well-formed and length-consistent
    When ParticleState::new(...) is called with ids=None
    Then it returns Ok(state)
    And state.positions_x[0] is NaN

  # --- ParticleBuffers allocation and upload ---

  @rq-0ffccee0
  Scenario: Allocate device buffers and perform initial upload
    Given a GpuContext obtained from init_device()
    And a ParticleState with particle_count() == 4 and known field values
    When ParticleBuffers::new(device, &state) is called
    Then it returns Ok(buffers)
    And buffers.particle_count() is 4
    And every device buffer has length 4
    And the bytes copied from each device buffer back to the host equal the corresponding state field
    And the device-side type_indices buffer holds the same values as state.type_indices

  @rq-0519e35c
  Scenario: New state has zero-initialised potential_energies and virials
    Given seven Vec<f32> of length 4 for positions_x/y/z, velocities_x/y/z, and masses
    And a Vec<u32> type_indices of length 4 with values [0, 0, 0, 0]
    When ParticleState::new(...) is called with ids=None
    Then state.potential_energies equals [0.0, 0.0, 0.0, 0.0]
    And state.virials equals [0.0, 0.0, 0.0, 0.0]

  @rq-9504346c
  Scenario: potential_energies and virials round-trip through ParticleBuffers
    Given a ParticleState A with particle_count() == 4
    And A.potential_energies has been overwritten with [1.0, -2.0, 3.5, 0.25]
    And A.virials has been overwritten with [10.0, 20.0, -30.0, 40.0]
    And a ParticleBuffers built from A
    And A.potential_energies and A.virials have been zeroed on the host
    When A.download_from(&buffers) is called
    Then A.potential_energies equals [1.0, -2.0, 3.5, 0.25]
    And A.virials equals [10.0, 20.0, -30.0, 40.0]

  @rq-c8aa7417
  Scenario: Allocate device buffers from an empty state
    Given a GpuContext obtained from init_device()
    And a ParticleState with particle_count() == 0
    When ParticleBuffers::new(device, &state) is called
    Then it returns Ok(buffers)
    And buffers.particle_count() is 0
    And every device buffer has length 0

  @rq-780d68ea
  Scenario: Reject ParticleBuffers::new when a host array length is inconsistent
    Given a GpuContext obtained from init_device()
    And a ParticleState whose positions_x.len() is 4 but velocities_z.len() is 3
    When ParticleBuffers::new(device, &state) is called
    Then it returns Err(ParticleStateError::LengthMismatch { array: "velocities_z", expected: 4, actual: 3 })
    And no device buffers are allocated

  @rq-4d226dff
  Scenario: Re-upload after host-side mutation
    Given a ParticleBuffers built from a ParticleState with particle_count() == 4
    And the host state's positions_x has been overwritten with new values
    When buffers.upload(&state) is called
    Then it returns Ok(())
    And the device positions_x buffer reflects the new host values
    And device buffers for unchanged host arrays still match their host counterparts

  @rq-9ff4fd10
  Scenario: Reject upload when host particle count differs from buffers
    Given a ParticleBuffers with particle_count() == 4
    And a ParticleState with particle_count() == 5
    When buffers.upload(&state) is called
    Then it returns Err(ParticleStateError::LengthMismatch { array: "positions_x", expected: 4, actual: 5 })

  @rq-b4bb7096
  Scenario: Reject upload when one host array drifts in length
    Given a ParticleBuffers with particle_count() == 4
    And a ParticleState whose positions_x.len() is 4 but forces_y.len() is 3
    When buffers.upload(&state) is called
    Then it returns Err(ParticleStateError::LengthMismatch { array: "forces_y", expected: 4, actual: 3 })

  # --- Download ---

  @rq-39a260b5
  Scenario: Download into the source state preserves values
    Given a ParticleState A with particle_count() == 4
    And a ParticleBuffers built from A
    When A.download_from(&buffers) is called
    Then it returns Ok(())
    And every field of A equals its value before the download

  @rq-9594de53
  Scenario: Download overwrites in-place after host mutation
    Given a ParticleState A with particle_count() == 4
    And a ParticleBuffers built from A
    And A's host arrays have subsequently been overwritten with garbage values
    When A.download_from(&buffers) is called
    Then it returns Ok(())
    And every field of A equals the values held by the buffers

  @rq-f4ebf12a
  Scenario: Download reflects device-side changes pushed by an interim re-upload
    Given a ParticleState A with particle_count() == 4
    And a ParticleBuffers built from A
    And a ParticleState B with the same particle count but different field values
    And buffers.upload(&B) has been called
    And A's host arrays have subsequently been overwritten with garbage values
    When A.download_from(&buffers) is called
    Then it returns Ok(())
    And every field of A equals the corresponding field of B

  @rq-7ab80063
  Scenario: Reject download when host particle count differs from buffers
    Given a ParticleBuffers with particle_count() == 4
    And a ParticleState with particle_count() == 5
    When state.download_from(&buffers) is called
    Then it returns Err(ParticleStateError::LengthMismatch { array: "positions_x", expected: 4, actual: 5 })

  @rq-1bdbd71e
  Scenario: Reject download when one host array drifts in length
    Given a ParticleBuffers with particle_count() == 4
    And a ParticleState whose positions_x.len() is 4 but masses.len() is 6
    When state.download_from(&buffers) is called
    Then it returns Err(ParticleStateError::LengthMismatch { array: "masses", expected: 4, actual: 6 })

  @rq-58254790
  Scenario: Download from an empty buffers into an empty state
    Given a ParticleBuffers with particle_count() == 0
    And a ParticleState with particle_count() == 0
    When state.download_from(&buffers) is called
    Then it returns Ok(())

  # --- Image flags ---

  @rq-6f897168
  Scenario: images defaults to zero-initialised arrays when None is passed
    Given seven Vec<f32> of length 4 and one type_indices Vec<u32> of length 4
    When ParticleState::new(..., ids=None, images=None) is called
    Then it returns Ok(state)
    And state.images_x, state.images_y, and state.images_z are each Vec<i32> of length 4 with every element 0

  @rq-2315b501
  Scenario: Explicit non-zero images are stored as-supplied
    Given seven Vec<f32> of length 3 and one type_indices Vec<u32> of length 3
    And images=Some((vec![1, -2, 0], vec![0, 3, -1], vec![-4, 0, 5]))
    When ParticleState::new(...) is called
    Then it returns Ok(state)
    And state.images_x equals [1, -2, 0]
    And state.images_y equals [0, 3, -1]
    And state.images_z equals [-4, 0, 5]

  @rq-0705e380
  Scenario: Reject explicit images_y of wrong length
    Given seven Vec<f32> of length 4 and one type_indices Vec<u32> of length 4
    And images=Some((vec![0; 4], vec![0; 3], vec![0; 4]))
    When ParticleState::new(...) is called
    Then it returns Err(ParticleStateError::LengthMismatch { array: "images_y", expected: 4, actual: 3 })

  @rq-2c31aa6e
  Scenario: ParticleBuffers carries the image-flag device buffers
    Given a ParticleState A with particle_count() == 4 and non-zero images
    When ParticleBuffers::new(device, &A) is called
    Then it returns Ok(buffers)
    And buffers.images_x.len(), buffers.images_y.len(), and buffers.images_z.len() each equal 4
    And downloading the three image buffers into Vec<i32>s reproduces A.images_x, A.images_y, A.images_z byte-for-byte

  @rq-cef2288f
  Scenario: Reject upload when images_x has the wrong length
    Given a ParticleBuffers with particle_count() == 4
    And a ParticleState whose images_x.len() is 3 (all other arrays length 4)
    When buffers.upload(&state) is called
    Then it returns Err(ParticleStateError::LengthMismatch { array: "images_x", expected: 4, actual: 3 })
```
