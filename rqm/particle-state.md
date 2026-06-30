# Feature: Particle State and GPU Buffers <!-- rq-f04e473c -->

The simulation's particle data is stored in two coordinated forms with
different layouts tuned to each side's access pattern:

- **Host-side** `ParticleState` is a structure of arrays backed by
  `Vec<Real>` and `Vec<u32>` / `Vec<i32>` fields. Per-component
  indexing (`state.positions_x[i]`) is the natural pattern for CPU
  iteration, file I/O, and tests.
- **Device-side** `ParticleBuffers` carries the positions and
  per-particle charge interleaved as `posq: CudaSlice<Real4>`
  (`.x`, `.y`, `.z` are the wrapped position; `.w` is the charge).
  Other per-particle quantities (velocities, forces, masses, type
  indices, image flags, etc.) remain as separate
  `CudaSlice<Real>` / `CudaSlice<u32>` / `CudaSlice<i32>` buffers
  on the device.

The packed `posq` form maximises the GPU's memory-coalescing for
the dominant kernels: a warp loading 32 consecutive atoms' posq
issues one 512-byte transaction instead of four separate 128-byte
transactions, and the pair-force kernels (which always need both
position and charge per j-atom) read `posq.w` from the same cache
line as the position. The SoA host form keeps per-component CPU
iteration ergonomic.

`ParticleState` carries the canonical initial conditions and any
data the host needs for I/O. `ParticleBuffers` carries the
canonical live state during a simulation: kernels read from and
write to the device buffers. Movement between the two is explicit
— `ParticleBuffers::upload` interleaves the host's
`positions_x/y/z` and `charges` arrays into the device `posq`
buffer (and copies every other per-particle array straight
through), and `ParticleState::download_from` splits the device
`posq` buffer back into the host's per-component
`positions_x/y/z` + `charges` arrays (and copies every other
buffer straight through).

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
charges:            Vec<f32>
type_indices:       Vec<u32>
particle_ids:       Vec<u32>
```

`positions_x[i]`, `positions_y[i]`, and `positions_z[i]` carry the
*wrapped* position of particle `i`: the position lies inside the primary
image of the simulation box (its fractional coordinates lie in
`[-1/2, 1/2)³`; see `simulation-box.md`). The companion image triple
`(images_x[i], images_y[i], images_z[i])` records how many lattice
vectors the particle has crossed since the start of the run, counted
per lattice direction `(a, b, c)`. The *unwrapped* position of
particle `i` is:

```
unwrapped = wrapped + images_x[i] * a + images_y[i] * b + images_z[i] * c
```

where `a`, `b`, `c` are the lattice vectors carried by `SimulationBox`.
For an orthorhombic box this reduces to the per-axis form
`unwrapped_a = wrapped_a + image_a * L_a`. Image flags start at zero (or
at the values supplied via the init file) and are advanced by the
integrator's drift kernels whenever the wrapped position crosses a
boundary of the primary image.

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

`ParticleBuffers` holds the device-side mirror with the
following layout:

```
posq:               CudaSlice<Real4>  // (.x = positions_x, .y = positions_y,
                                       //  .z = positions_z, .w = charges)
images_x:           CudaSlice<i32>
images_y:           CudaSlice<i32>
images_z:           CudaSlice<i32>
velocities_x:       CudaSlice<Real>
velocities_y:       CudaSlice<Real>
velocities_z:       CudaSlice<Real>
forces_x:           CudaSlice<Real>
forces_y:           CudaSlice<Real>
forces_z:           CudaSlice<Real>
potential_energies: CudaSlice<Real>
virials:            CudaSlice<Real>
masses:             CudaSlice<Real>
type_indices:       CudaSlice<u32>
particle_ids:       CudaSlice<u32>
```

Every device buffer has length equal to the particle count
established at `ParticleBuffers::new` time. `posq` is `length` ×
16 bytes under the default `f32` precision and `length` × 32
bytes under `--features f64`; the rest of the layout matches the
host SoA arrays one-for-one.

All arrays of the same particle state must have the same length. This
invariant is checked at construction time and again at every upload/download
call.

## Feature API <!-- rq-c9016748 -->

### Types <!-- rq-08066bdf -->

- `ParticleState` — host-side SoA state. All eighteen per-particle arrays <!-- rq-3766be01 -->
  are declared as `pub` fields so callers may iterate, index, and mutate
  them directly. Length consistency between fields is the caller's
  responsibility while the state is held on the host; it is re-validated
  at every upload or download.

- `ParticleBuffers` — device-side mirror. Holds `posq: <!-- rq-4a8de06c -->
  CudaSlice<Real4>` (interleaved position + charge), separate
  `CudaSlice<Real>` buffers for `velocities_x`, `velocities_y`,
  `velocities_z`, `forces_x`, `forces_y`, `forces_z`,
  `potential_energies`, `virials`, and `masses`, separate
  `CudaSlice<i32>` buffers for `images_x`, `images_y`, `images_z`,
  and `CudaSlice<u32>` buffers for `particle_ids` and
  `type_indices`. Each buffer is exposed as a `pub` field so kernel
  launch sites can pass `&CudaSlice` references directly. Also
  carries an `Arc<CudaDevice>` for upload/download bookkeeping.

- `Real4` — packed 4-tuple of `Real` values used for `posq`. <!-- rq-real4-type --> <!-- rq-eaaddf05 -->
  Definition follows the precision shim:

  ```rust
  #[cfg(not(feature = "f64"))]
  pub type Real4 = cudarc::driver::sys::float4;

  #[cfg(feature = "f64")]
  pub type Real4 = cudarc::driver::sys::double4;
  ```

  The host-side construction is `Real4 { x, y, z, w }`. The device
  side uses cudarc's `DeviceRepr` impls for these CUDA vector
  types so a `CudaSlice<Real4>` participates in `htod_sync_copy`,
  `dtoh_sync_copy`, and kernel-argument passing without per-call
  conversion. Inside CUDA kernels the type is `float4` or
  `double4` (via the `Real` typedef shim in `precision.cuh`); a
  posq load is one 16- or 32-byte coalesced transaction per warp
  for 32 consecutive atoms.

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

- `ParticleState::new(positions_x: Vec<f32>, positions_y: Vec<f32>, positions_z: Vec<f32>, velocities_x: Vec<f32>, velocities_y: Vec<f32>, velocities_z: Vec<f32>, masses: Vec<f32>, charges: Vec<f32>, type_indices: Vec<u32>, ids: Option<Vec<u32>>, images: Option<(Vec<i32>, Vec<i32>, Vec<i32>)>) -> Result<ParticleState, ParticleStateError>` <!-- rq-5e0598cb -->
  - The particle count is taken from `positions_x.len()`.
  - Validates that `positions_y`, `positions_z`, `velocities_x`,
    `velocities_y`, `velocities_z`, `masses`, `charges`, and
    `type_indices` all have the same length as `positions_x`. Returns
    `LengthMismatch` on the first offending array (checked in
    declaration order).
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
    not enforce that fractional coordinates lie in `[-1/2, 1/2)³`; that
    invariant is re-established by the integrator's drift kernels on the
    first step).

- `ParticleState::particle_count(&self) -> usize` <!-- rq-ac035b90 -->
  - Returns `positions_x.len()`. Callers are expected to keep the other
    arrays at the same length.

- `ParticleBuffers::new(device: Arc<CudaDevice>, state: &ParticleState) -> Result<ParticleBuffers, ParticleStateError>` <!-- rq-b09032cb -->
  - Allocates the device buffers listed in *Data Layout*, sized to
    `state.particle_count()`.
  - Validates that every host array has length
    `state.particle_count()`; returns `LengthMismatch` otherwise.
  - Builds a host-side `Vec<Real4>` of length
    `state.particle_count()` by interleaving
    `state.positions_x/y/z` and `state.charges`
    (`Real4 { x: positions_x[i], y: positions_y[i],
    z: positions_z[i], w: charges[i] }` for each `i`), then copies
    it into the device `posq` buffer.
  - Copies every other host array (velocities, forces, masses,
    image flags, type_indices, particle_ids, potential_energies,
    virials) into the corresponding device buffers one-for-one.
  - Returns the populated `ParticleBuffers` on success.

- `ParticleBuffers::particle_count(&self) -> usize` <!-- rq-18411920 -->
  - Returns the per-buffer length established at construction.

- `ParticleBuffers::upload(&mut self, state: &ParticleState) -> Result<(), ParticleStateError>` <!-- rq-179ed985 -->
  - Validates that `state.particle_count()` equals
    `self.particle_count()` and that every host array has that
    length; returns `LengthMismatch` otherwise.
  - Rebuilds a host-side `Vec<Real4>` from the current
    `state.positions_x/y/z` + `state.charges`, then writes it into
    the existing device `posq` buffer in-place.
  - Copies every other host array into the existing device buffers
    in-place.

- `ParticleState::download_from(&mut self, buffers: &ParticleBuffers) -> Result<(), ParticleStateError>` <!-- rq-9a19bfa3 -->
  - Validates that `self.particle_count()` equals
    `buffers.particle_count()` and that every host array has that
    length; returns `LengthMismatch` otherwise.
  - Reads the device `posq` buffer back into a host-side
    `Vec<Real4>`, then splits each entry into
    `state.positions_x[i] = posq[i].x`,
    `state.positions_y[i] = posq[i].y`,
    `state.positions_z[i] = posq[i].z`,
    `state.charges[i] = posq[i].w`. Both the position split and the
    charge split are stored in place, overwriting prior contents.
  - Copies every other device buffer into the corresponding host
    `Vec` in place, overwriting prior contents.
  - No reallocation occurs.

## Construction Details <!-- rq-ef7b719d -->

- `ParticleState::new` consumes its input `Vec`s; no copies are made of the
  host arrays at construction.
- Length validation is performed in declaration order
  (`positions_y`, `positions_z`, `velocities_x`, `velocities_y`,
  `velocities_z`, `masses`, `charges`, `type_indices`, then `ids` when
  `Some`, then `images_x`, `images_y`, `images_z` when `images` is
  `Some`).
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
- The `f64` precision feature flag changes the storage type of
  `Real` (and hence `Real4` and every `CudaSlice<Real>` buffer
  documented above) from 32- to 64-bit IEEE-754. Every numerical
  invariant in this file holds under both precisions; the round-
  trip scenarios below are byte-for-byte exact within the active
  precision.
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
    Given positions_x, positions_y, positions_z, velocities_x, velocities_y, velocities_z, masses, charges each have length 4
    And type_indices has length 3
    When ParticleState::new(...) is called with ids=None
    Then it returns Err(ParticleStateError::LengthMismatch { array: "type_indices", expected: 4, actual: 3 })

  @rq-7fd19f00
  Scenario: Reject when charges has the wrong length
    Given every other input Vec has length 4
    And charges has length 5
    When ParticleState::new(...) is called with ids=None
    Then it returns Err(ParticleStateError::LengthMismatch { array: "charges", expected: 4, actual: 5 })

  @rq-fdc02bdb
  Scenario: charges round-trip through ParticleBuffers via posq.w
    Given a ParticleState A with particle_count() == 4
    And A.charges has been overwritten with [+1.602e-19, -1.602e-19, 0.0, 3.2e-19]
    And a ParticleBuffers built from A
    And A.charges has been zeroed on the host
    When A.download_from(&buffers) is called
    Then A.charges equals [+1.602e-19, -1.602e-19, 0.0, 3.2e-19] byte-for-byte
    And A.charges[i] equals the .w component of every Real4 entry in buffers.posq

  @rq-dabd2130
  Scenario: positions round-trip through ParticleBuffers via posq.xyz
    Given a ParticleState A with particle_count() == 4
    And A.positions_x has been overwritten with [0.1, 0.2, 0.3, 0.4]
    And A.positions_y has been overwritten with [1.1, 1.2, 1.3, 1.4]
    And A.positions_z has been overwritten with [2.1, 2.2, 2.3, 2.4]
    And a ParticleBuffers built from A
    And A.positions_x, A.positions_y, A.positions_z have been zeroed on the host
    When A.download_from(&buffers) is called
    Then A.positions_x equals [0.1, 0.2, 0.3, 0.4] byte-for-byte
    And A.positions_y equals [1.1, 1.2, 1.3, 1.4] byte-for-byte
    And A.positions_z equals [2.1, 2.2, 2.3, 2.4] byte-for-byte
    And A.positions_x[i], A.positions_y[i], A.positions_z[i] equal the .x, .y, .z components of buffers.posq[i]

  @rq-01d1bb68
  Scenario: posq interleave preserves per-atom (x, y, z, charge) grouping
    Given a ParticleState A with particle_count() == 3 whose
      positions_x = [10.0, 20.0, 30.0],
      positions_y = [11.0, 21.0, 31.0],
      positions_z = [12.0, 22.0, 32.0],
      charges     = [+1.0, -1.0, +2.0]
    When ParticleBuffers::new(device, &A) is called
    And the resulting device posq buffer is downloaded as Vec<Real4>
    Then the downloaded Vec equals [Real4{10.0, 11.0, 12.0, +1.0},
                                     Real4{20.0, 21.0, 22.0, -1.0},
                                     Real4{30.0, 31.0, 32.0, +2.0}]
      byte-for-byte

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
