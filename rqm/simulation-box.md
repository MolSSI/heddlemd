# Feature: Simulation Box and Periodic Boundary Conditions <!-- rq-03830444 -->

The simulation runs in a periodic, axis-aligned, rectangular cell of edge
lengths `(lx, ly, lz)`. The cell's primary image is centered at the origin and
spans `[-lx/2, lx/2) × [-ly/2, ly/2) × [-lz/2, lz/2)`.

`SimulationBox` is an immutable host-side type carrying the three edge lengths.
It exposes two pure operations used by neighbor search and pair-force
computation:

- `minimum_image` — given a displacement vector between two particles, return
  the shortest equivalent displacement under periodicity (the "minimum image").
- `wrap_position` — given an absolute position, return the equivalent position
  inside the primary image of the cell.

Both operations are pure functions of the box and the input vector. They are
defined on the host in Rust; future kernel features inline equivalent math in
CUDA, taking `(lx, ly, lz)` as kernel arguments.

## Coordinate Conventions <!-- rq-c1308495 -->

- The primary cell is `[-lx/2, lx/2) × [-ly/2, ly/2) × [-lz/2, lz/2)`. The
  lower bound is included; the upper bound is excluded.
- The minimum image of a displacement falls in the same half-open interval
  per axis: `[-lx/2, lx/2) × [-ly/2, ly/2) × [-lz/2, lz/2)`.
- All quantities are `f32`.

## Wrap Formula <!-- rq-4ca9b179 -->

Wrapping a coordinate `x` along an axis of length `L` into `[-L/2, L/2)`:

```
x_wrapped = x - L * floor((x + L/2) / L)
```

This formula handles arbitrary multiples of `L` and treats both boundaries
deterministically: `x = +L/2` maps to `-L/2`, `x = -L/2` maps to `-L/2`.

`minimum_image` and `wrap_position` are the same wrap operation applied to a
displacement and an absolute position respectively.

## Feature API <!-- rq-63f3e0b9 -->

### Types <!-- rq-fdf2db79 -->

- `SimulationBox` — host-side, immutable, `Copy`. Carries three `f32` edge <!-- rq-b75afb31 -->
  lengths. All accessors are total; the constructor enforces the invariants.

- `SimulationBoxError` — error type returned by the constructor: <!-- rq-aef9888b -->
  - `NonFiniteLength { axis: &'static str, value: f32 }` — at least one edge
    length is NaN or infinite. `axis` is one of `"lx"`, `"ly"`, `"lz"`.
  - `NonPositiveLength { axis: &'static str, value: f32 }` — at least one edge
    length is finite but `<= 0.0`. `axis` is one of `"lx"`, `"ly"`, `"lz"`.

### Constructor <!-- rq-b8070abb -->

- `SimulationBox::new_orthorhombic(lx: f32, ly: f32, lz: f32) -> Result<SimulationBox, SimulationBoxError>` <!-- rq-f0da71ea -->
  - Validates the three lengths in declaration order (`lx`, `ly`, `lz`).
  - For each length, checks finiteness first (returns `NonFiniteLength` on NaN
    or infinity), then positivity (returns `NonPositiveLength` if the finite
    value is `<= 0.0`).
  - On success, stores the three lengths and returns the constructed box.

### Accessors <!-- rq-b015ef15 -->

- `SimulationBox::lengths(&self) -> [f32; 3]` <!-- rq-e8be1a1c -->
  - Returns `[lx, ly, lz]` in that order.

- `SimulationBox::lx(&self) -> f32`, `SimulationBox::ly(&self) -> f32`, <!-- rq-f73a0f99 -->
  `SimulationBox::lz(&self) -> f32`
  - Per-axis getters; equivalent to indexing `lengths()`.

- `SimulationBox::volume(&self) -> f32` <!-- rq-3b9ed390 -->
  - Returns `lx * ly * lz` (multiplication left-to-right in `f32`).

### Periodic-boundary operations <!-- rq-fb632dfc -->

- `SimulationBox::minimum_image(&self, displacement: [f32; 3]) -> [f32; 3]` <!-- rq-d49c9093 -->
  - Applies the wrap formula independently per axis with the corresponding
    edge length. Returns the minimum-image displacement.

- `SimulationBox::wrap_position(&self, position: [f32; 3]) -> [f32; 3]` <!-- rq-9b1c84c3 -->
  - Applies the wrap formula independently per axis with the corresponding
    edge length. Returns the position inside the primary image.

The two methods produce identical output for identical input; they exist as
separate names so call sites communicate intent (displacement vs absolute
position).

## Numerical Behaviour <!-- rq-70ff0369 -->

- Non-finite inputs to `minimum_image` or `wrap_position` propagate to
  non-finite outputs (no validation; matches the trust-the-caller posture
  used elsewhere in the project for kernel inputs).
- The wrap formula uses `f32::floor`, which is IEEE-754 deterministic.
- Repeated application of `wrap_position` is idempotent: for any finite
  input `p`, `wrap_position(wrap_position(p)) == wrap_position(p)`.

## Out of Scope <!-- rq-987dc616 -->

- Triclinic / non-orthorhombic cells.
- Box rescaling, NPT ensembles, deformable cells.
- Non-periodic boundaries (open or reflecting).
- Device-side (CUDA) PBC helpers; consuming kernels inline the math.
- Per-particle bulk wrap helpers operating on `Vec<f32>` SoA arrays
  (callers loop over `wrap_position` until a bulk helper is needed).
- The `f64` precision feature flag.

---

## Gherkin Scenarios <!-- rq-1012fb8a -->

```gherkin
Feature: Simulation box and periodic boundary conditions

  Background:
    Given a SimulationBox constructed with lx=10.0, ly=8.0, lz=6.0

  # --- Construction ---

  @rq-27ffd3f4
  Scenario: Construct with positive finite edge lengths
    When SimulationBox::new_orthorhombic(10.0, 8.0, 6.0) is called
    Then it returns Ok(box)
    And box.lengths() equals [10.0, 8.0, 6.0]
    And box.lx() equals 10.0
    And box.ly() equals 8.0
    And box.lz() equals 6.0

  @rq-e1b51bd9
  Scenario: volume returns the product of the edge lengths
    Given a SimulationBox constructed with lx=2.0, ly=3.0, lz=5.0
    Then box.volume() equals 30.0

  @rq-8259c9ca
  Scenario: Reject zero lx
    When SimulationBox::new_orthorhombic(0.0, 8.0, 6.0) is called
    Then it returns Err(SimulationBoxError::NonPositiveLength { axis: "lx", value: 0.0 })

  @rq-05eb9fbb
  Scenario: Reject zero ly
    When SimulationBox::new_orthorhombic(10.0, 0.0, 6.0) is called
    Then it returns Err(SimulationBoxError::NonPositiveLength { axis: "ly", value: 0.0 })

  @rq-74aa3a99
  Scenario: Reject zero lz
    When SimulationBox::new_orthorhombic(10.0, 8.0, 0.0) is called
    Then it returns Err(SimulationBoxError::NonPositiveLength { axis: "lz", value: 0.0 })

  @rq-9b1f8a7c
  Scenario: Reject negative lx
    When SimulationBox::new_orthorhombic(-1.0, 8.0, 6.0) is called
    Then it returns Err(SimulationBoxError::NonPositiveLength { axis: "lx", value: -1.0 })

  @rq-19fe4806
  Scenario: Reject NaN lx
    When SimulationBox::new_orthorhombic(f32::NAN, 8.0, 6.0) is called
    Then it returns Err(SimulationBoxError::NonFiniteLength { axis: "lx", value: v }) where v is NaN

  @rq-7f867e37
  Scenario: Reject infinite ly
    When SimulationBox::new_orthorhombic(10.0, f32::INFINITY, 6.0) is called
    Then it returns Err(SimulationBoxError::NonFiniteLength { axis: "ly", value: f32::INFINITY })

  @rq-7541fd8a
  Scenario: Validation order is lx then ly then lz
    When SimulationBox::new_orthorhombic(0.0, -1.0, f32::NAN) is called
    Then it returns Err(SimulationBoxError::NonPositiveLength { axis: "lx", value: 0.0 })

  @rq-b9a4e3de
  Scenario: Non-finite check precedes non-positive check on the same axis
    When SimulationBox::new_orthorhombic(f32::NAN, 8.0, 6.0) is called
    Then it returns Err(SimulationBoxError::NonFiniteLength { axis: "lx", value: v }) where v is NaN

  # --- minimum_image ---

  @rq-8c045718
  Scenario: minimum_image of the zero displacement is zero
    When box.minimum_image([0.0, 0.0, 0.0]) is called
    Then the result equals [0.0, 0.0, 0.0]

  @rq-bfb3b9d8
  Scenario: minimum_image leaves a displacement strictly inside the primary image unchanged
    Given displacement = [4.0, 3.0, 2.0]
    When box.minimum_image(displacement) is called
    Then the result equals [4.0, 3.0, 2.0]

  @rq-9a9523d9
  Scenario: minimum_image at the +L/2 boundary maps to -L/2
    When box.minimum_image([5.0, 0.0, 0.0]) is called
    Then the result equals [-5.0, 0.0, 0.0]

  @rq-d19fc020
  Scenario: minimum_image at the -L/2 boundary stays at -L/2
    When box.minimum_image([-5.0, 0.0, 0.0]) is called
    Then the result equals [-5.0, 0.0, 0.0]

  @rq-f7b922df
  Scenario: minimum_image just past +L/2 wraps by one period
    When box.minimum_image([6.0, 0.0, 0.0]) is called
    Then the result equals [6.0 - 10.0, 0.0, 0.0]

  @rq-a8df30ac
  Scenario: minimum_image just past -L/2 wraps by one period
    When box.minimum_image([-6.0, 0.0, 0.0]) is called
    Then the result equals [-6.0 + 10.0, 0.0, 0.0]

  @rq-0ae304bc
  Scenario: minimum_image handles many-period displacements
    When box.minimum_image([24.0, 0.0, 0.0]) is called
    Then the result_x lies in [-5.0, 5.0)
    And result_x equals 24.0 - 10.0 * round_to_nearest_period(24.0)

  @rq-c9618bdd
  Scenario: minimum_image is per-axis independent
    Given displacement = [6.0, -5.0, 4.0]
    When box.minimum_image(displacement) is called
    Then the x-component is wrapped against lx=10.0
    And the y-component is wrapped against ly=8.0
    And the z-component is wrapped against lz=6.0

  # --- wrap_position ---

  @rq-3e8324c2
  Scenario: wrap_position leaves a position inside the primary image unchanged
    Given position = [4.0, 3.0, 2.0]
    When box.wrap_position(position) is called
    Then the result equals [4.0, 3.0, 2.0]

  @rq-4b9d059e
  Scenario: wrap_position wraps a position outside the primary image
    Given position = [12.0, -5.0, 7.0]
    When box.wrap_position(position) is called
    Then result_x lies in [-5.0, 5.0)
    And result_y lies in [-4.0, 4.0)
    And result_z lies in [-3.0, 3.0)
    And the result equals box.minimum_image(position)

  @rq-941c4000
  Scenario: wrap_position is idempotent
    Given position = [123.45, -67.89, 42.0]
    When wrapped_once = box.wrap_position(position)
    And wrapped_twice = box.wrap_position(wrapped_once)
    Then wrapped_twice equals wrapped_once

  @rq-a1fc0841
  Scenario: wrap_position and minimum_image agree on the same input
    Given v = [17.0, -13.0, 9.5]
    When mi = box.minimum_image(v)
    And wp = box.wrap_position(v)
    Then mi equals wp

  # --- Numerical edge cases ---

  @rq-4b63564b
  Scenario: NaN displacement propagates to NaN output
    When box.minimum_image([f32::NAN, 0.0, 0.0]) is called
    Then result_x is NaN
    And result_y equals 0.0
    And result_z equals 0.0
```
