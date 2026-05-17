# Feature: Simulation Box and Periodic Boundary Conditions <!-- rq-03830444 -->

The simulation runs in a periodic, parallelepiped cell described in
lower-triangular form. The primary image of the cell is the parallelepiped
centered at the origin spanned by three lattice vectors:

```
a = (lx,  0,   0)
b = (xy,  ly,  0)
c = (xz,  yz,  lz)
```

`(lx, ly, lz)` are strictly positive diagonal edge lengths; `(xy, xz, yz)`
are tilt parameters that place `b` in the xy-plane and `c` in general
position. An orthorhombic cell corresponds to `xy = xz = yz = 0`.

`SimulationBox` is a host-side `Copy` type carrying the six lattice
parameters and a monotonic `generation` counter. It exposes pure operations
used by neighbor search and pair-force computation:

- `minimum_image` — given a displacement vector between two particles, returns
  the shortest equivalent displacement under periodicity (the "minimum image").
- `wrap_position` — given an absolute position, returns the equivalent position
  inside the primary image of the cell.
- `wrap_position_with_image_count` — same as `wrap_position` but also returns
  the integer image triple `(k_a, k_b, k_c)` recording how many lattice
  vectors were subtracted along each fractional direction. Integrator drift
  kernels use this to update the per-particle image flags.
- `min_perpendicular_width` — the minimum perpendicular distance between
  opposite faces of the parallelepiped. Consumers gate cutoff-vs-box
  validity against this value rather than `min(lx, ly, lz)`.
- `fractional_coords` and `cartesian_coords` — map between Cartesian
  coordinates and fractional coordinates along the lattice vectors. Used
  by the spatial-hash kernel (which indexes cells in fractional space) and
  by the init-state parser's position-bounds check.

All operations are pure functions of the box and the input vector. They are
defined on the host in Rust; consuming kernels inline equivalent math in
CUDA, taking the six lattice parameters as kernel arguments.

The six lattice parameters are mutable in place through `set_lattice`. Every
successful mutation increments the box's `generation` counter by one.
Downstream consumers that cache values derived from the box's lattice (the
neighbor list's cached cell layout; future barostat bookkeeping) record the
generation alongside their cache and refresh whenever the live box's
generation differs from the recorded one.

## Lattice Convention <!-- rq-c1308495 -->

The three lattice vectors are arranged so that `a` lies along the x-axis,
`b` lies in the xy-plane, and `c` is general. In matrix form, the lattice
matrix `H` with rows = lattice vectors is lower-triangular:

```
H = [lx   0    0 ]
    [xy   ly   0 ]
    [xz   yz   lz]
```

The matrix with columns = lattice vectors is the transpose `H^T` and is
upper-triangular. The two forms describe the same physical cell; this file
uses the rows-are-vectors convention throughout.

For a position or displacement `v = (v_x, v_y, v_z)` treated as a column
vector, the Cartesian-to-fractional transform is `s = H^(-T) · v` and the
fractional-to-Cartesian transform is `v = H^T · s`. Both have closed-form
back-substitution implementations:

```
s_c = v_z / lz
s_b = (v_y - s_c · yz) / ly
s_a = (v_x - s_b · xy - s_c · xz) / lx
```

```
v_x = s_a · lx + s_b · xy + s_c · xz
v_y = s_b · ly + s_c · yz
v_z = s_c · lz
```

The primary cell is the set `{ H^T · s : s ∈ [-1/2, 1/2)³ }`. A position is
inside the primary cell iff its fractional coordinates satisfy
`s_a, s_b, s_c ∈ [-1/2, 1/2)`. The interval is half-open: the lower bound
is included, the upper bound is excluded, matching the convention used by
the integrator's drift kernels.

For an orthorhombic box (`xy = xz = yz = 0`), every formula above reduces
to the per-axis equivalents `s_a = v_x / lx`, `v_x = s_a · lx`, and the
primary cell becomes `[-lx/2, lx/2) × [-ly/2, ly/2) × [-lz/2, lz/2)`.

## Wrap Algorithm <!-- rq-4ca9b179 -->

Wrapping a displacement or position `(v_x, v_y, v_z)` into the primary
image is a fractional-coordinate wrap. The algorithm computes the
fractional coordinates of the input via back-substitution
(z-then-y-then-x), picks the integer image triple that brings each
fractional component into `[-1/2, 1/2)`, and applies the image-vector
correction directly in Cartesian coordinates:

```
s_c = v_z / lz
s_b = (v_y - s_c · yz) / ly
s_a = (v_x - s_b · xy - s_c · xz) / lx

k_a = floor(s_a + 0.5)
k_b = floor(s_b + 0.5)
k_c = floor(s_c + 0.5)

v_x := v_x - k_a · lx - k_b · xy - k_c · xz
v_y := v_y - k_b · ly - k_c · yz
v_z := v_z - k_c · lz
```

After these three steps the result `(v_x, v_y, v_z)` has fractional
coordinates in `[-1/2, 1/2)³` and therefore lies inside the primary
parallelepiped. The integer triple `(k_a, k_b, k_c)` records the number
of image shifts along the `a`, `b`, `c` lattice directions;
`minimum_image` and `wrap_position` discard it while
`wrap_position_with_image_count` returns it so integrator drift kernels
can advance per-particle image flags.

The algorithm uses `f32::floor` and a fixed sequence of `f32`
adds/subtracts/multiplies/divides in the order shown. The bit pattern of
the result is identical across runs given identical inputs on the same
hardware. For an orthorhombic box (`xy = xz = yz = 0`), `s_d` reduces to
`v_d / L_d`, the image counts reduce to `floor(v_d / L_d + 0.5)` which
equals `floor((v_d + L_d · 0.5) / L_d)`, and the Cartesian correction
reduces to three independent per-axis subtractions `v_d -= k_d · L_d` —
the v0 orthorhombic implementation, bit-for-bit.

`minimum_image` and `wrap_position` apply this algorithm to a displacement
and an absolute position respectively. They return identical output for
identical input; the two methods exist as separate names so call sites
communicate intent.

## Perpendicular Widths <!-- rq-9d8d96f1 -->

The perpendicular width along lattice direction `a` is the perpendicular
distance between the two faces of the parallelepiped that are parallel to
the plane spanned by `b` and `c`. In closed form for the lower-triangular
lattice:

```
w_a = (lx · ly · lz) / sqrt((ly · lz)² + (xy · lz)² + (xy · yz - ly · xz)²)
w_b = (ly · lz) / sqrt(lz² + yz²)
w_c = lz
```

`min_perpendicular_width()` returns `min(w_a, w_b, w_c)`. All intermediate
operations are `f32`. For an orthorhombic box the formula reduces to
`min(lx, ly, lz)`.

Consumers (the neighbor list, a future barostat) gate cutoff-vs-box
validity against `min_perpendicular_width / 2` rather than
`min(lx, ly, lz) / 2`: the minimum image is well-defined precisely when
every interaction radius is less than half the shortest perpendicular
width.

## Feature API <!-- rq-63f3e0b9 -->

### Types <!-- rq-fdf2db79 -->

- `SimulationBox` — host-side, `Copy`. Carries six `f32` lattice <!-- rq-b75afb31 -->
  parameters (`lx`, `ly`, `lz`, `xy`, `xz`, `yz`) and a `u64` `generation`
  counter. The constructor enforces invariants on the lattice parameters
  and starts the generation at `0`; `set_lattice` enforces the same
  invariants on each subsequent mutation and increments the generation
  on success. All accessors are total.

- `SimulationBoxError` — error type returned by the constructor, the <!-- rq-aef9888b -->
  mutator, and `check_min_perpendicular_width`:
  - `NonFiniteLatticeValue { name: &'static str, value: f32 }` — at least
    one lattice parameter is NaN or infinite. `name` is one of `"lx"`,
    `"ly"`, `"lz"`, `"xy"`, `"xz"`, `"yz"`.
  - `NonPositiveDiagonal { name: &'static str, value: f32 }` — at least
    one diagonal is finite but `<= 0.0`. `name` is one of `"lx"`, `"ly"`,
    `"lz"`. Tilts (`xy`, `xz`, `yz`) are not subject to this check; any
    finite sign or magnitude is accepted.
  - `PerpendicularWidthTooSmall { direction: &'static str, width: f32, required: f32 }`
    — at least one of the box's perpendicular widths is strictly less
    than the supplied `required` value. `direction` is one of `"a"`,
    `"b"`, `"c"` and identifies the first lattice direction (scanning
    `a → b → c`) whose perpendicular width fails the threshold. `width`
    is the failing direction's `f32` perpendicular width; `required` is
    the supplied threshold. Only produced by
    `check_min_perpendicular_width`; the constructor and mutator never
    surface this variant.

### Constructor <!-- rq-b8070abb -->

- `SimulationBox::new(lx: f32, ly: f32, lz: f32, xy: f32, xz: f32, yz: f32) -> Result<SimulationBox, SimulationBoxError>` <!-- rq-f0da71ea -->
  - Validates each parameter in declaration order (`lx`, `ly`, `lz`, `xy`,
    `xz`, `yz`).
  - For each parameter, checks finiteness first (returns
    `NonFiniteLatticeValue` on NaN or infinity).
  - For each diagonal (`lx`, `ly`, `lz`), additionally checks positivity
    after finiteness (returns `NonPositiveDiagonal` if the finite value
    is `<= 0.0`).
  - Tilts (`xy`, `xz`, `yz`) need only be finite; any sign and magnitude
    are accepted.
  - On success, stores the six parameters and returns the constructed
    box with `generation = 0`.

### Accessors <!-- rq-b015ef15 -->

- `SimulationBox::lattice(&self) -> [f32; 6]` <!-- rq-e8be1a1c -->
  - Returns `[lx, ly, lz, xy, xz, yz]` in that order.

- `SimulationBox::lx(&self) -> f32`, `SimulationBox::ly(&self) -> f32`, <!-- rq-f73a0f99 -->
  `SimulationBox::lz(&self) -> f32`, `SimulationBox::xy(&self) -> f32`,
  `SimulationBox::xz(&self) -> f32`, `SimulationBox::yz(&self) -> f32`
  - Per-parameter getters.

- `SimulationBox::volume(&self) -> f32` <!-- rq-3b9ed390 -->
  - Returns `lx * ly * lz` (multiplication left-to-right in `f32`). The
    determinant of a lower-triangular matrix is the product of its
    diagonal entries, so tilts do not enter.

- `SimulationBox::min_perpendicular_width(&self) -> f32` <!-- rq-5fe22acb -->
  - Returns `min(w_a, w_b, w_c)` computed via the closed-form expressions
    in *Perpendicular Widths*. All intermediate operations are `f32` in
    the order shown.

- `SimulationBox::check_min_perpendicular_width(&self, required: f32) -> Result<(), SimulationBoxError>` <!-- rq-1a7bd47a -->
  - Computes the three perpendicular widths via the closed-form
    expressions in *Perpendicular Widths* (no allocation), then scans
    them in lattice-direction order `a → b → c` and returns
    `Err(SimulationBoxError::PerpendicularWidthTooSmall { direction,
    width, required })` on the first direction whose width is strictly
    less than `required`. `direction` is `"a"`, `"b"`, or `"c"`
    matching the failing direction; `width` is that direction's
    `f32` perpendicular width. Returns `Ok(())` when every direction
    has `width >= required`.
  - The scan order is fixed at `a → b → c`. When more than one
    direction fails, the variant reports the lowest-indexed failing
    direction; remaining directions are not inspected.
  - `required` is taken verbatim — no transformation or sign check is
    applied. A `required <= 0` always returns `Ok(())` because every
    `f32` perpendicular width produced by a valid box is strictly
    positive. A non-finite `required` (NaN or infinity) yields
    `Err(...)` for direction `"a"` because every `f32 < NaN` and every
    finite width is `< +inf`.
  - The widths' computation is pure `f32` in the order shown by
    `perpendicular_widths`; two calls with identical `required` on the
    same `SimulationBox` produce byte-identical outcomes.

- `SimulationBox::generation(&self) -> u64` <!-- rq-dc17132d -->
  - Returns the box's generation counter. The counter is `0` immediately
    after construction and increments by `1` on every successful
    `set_lattice` call. A consumer that caches a value derived from the
    box's lattice records the generation alongside the cache and
    re-derives the cached value whenever the observed generation differs
    from the recorded one.

### Mutators <!-- rq-b033ac1d -->

- `SimulationBox::set_lattice(&mut self, lx: f32, ly: f32, lz: f32, xy: f32, xz: f32, yz: f32) -> Result<(), SimulationBoxError>` <!-- rq-71fbbafb -->
  - Validates the six parameters in declaration order (`lx`, `ly`, `lz`,
    `xy`, `xz`, `yz`) using the same finiteness-then-positivity rules as
    the constructor.
  - On validation failure: returns the matching `SimulationBoxError`;
    the box's stored lattice and `generation` counter are left
    unchanged.
  - On success: replaces all six parameters with the new values and
    increments `generation` by `1` (wrapping at `u64::MAX`; collisions
    take `2^64` mutations and are not considered).

### Periodic-boundary operations <!-- rq-fb632dfc -->

- `SimulationBox::minimum_image(&self, displacement: [f32; 3]) -> [f32; 3]` <!-- rq-d49c9093 -->
  - Applies the *Wrap Algorithm* and returns the minimum-image
    displacement. The image triple `(k_a, k_b, k_c)` is discarded.

- `SimulationBox::wrap_position(&self, position: [f32; 3]) -> [f32; 3]` <!-- rq-9b1c84c3 -->
  - Applies the *Wrap Algorithm* and returns the position inside the
    primary image. The image triple is discarded.

- `SimulationBox::wrap_position_with_image_count(&self, position: [f32; 3]) -> ([f32; 3], [i32; 3])` <!-- rq-a4d5e711 -->
  - Applies the *Wrap Algorithm* and returns both the wrapped position
    and the integer image triple `(k_a, k_b, k_c)`. Integrator drift
    kernels use the integer triple to update the per-particle image
    flags so that the unwrapped position
    `wrapped + k_a · a + k_b · b + k_c · c` is invariant under the
    wrap.

- `SimulationBox::fractional_coords(&self, position: [f32; 3]) -> [f32; 3]` <!-- rq-1a3ec0c8 -->
  - Returns `s = H^(-T) · position` computed via back-substitution in the
    order `s_c → s_b → s_a` described in *Lattice Convention*. All
    operations are `f32` in the order shown.

- `SimulationBox::cartesian_coords(&self, fractional: [f32; 3]) -> [f32; 3]` <!-- rq-be7b9fe6 -->
  - Returns `v = H^T · fractional` computed via forward substitution in
    the order `v_z → v_y → v_x` described in *Lattice Convention*. All
    operations are `f32` in the order shown.

`minimum_image` and `wrap_position` produce identical output for identical
input; they exist as separate names so call sites communicate intent
(displacement vs absolute position).

## Numerical Behaviour <!-- rq-70ff0369 -->

- Non-finite inputs to `minimum_image`, `wrap_position`,
  `wrap_position_with_image_count`, `fractional_coords`, or
  `cartesian_coords` propagate to non-finite outputs (no validation;
  matches the trust-the-caller posture used elsewhere in the project for
  kernel inputs).
- The wrap algorithm uses `f32::floor`, which is IEEE-754 deterministic.
- Repeated application of `wrap_position` is idempotent: for any finite
  input `p`, `wrap_position(wrap_position(p)) == wrap_position(p)`.
- All intermediate `f32` operations in the wrap algorithm and the
  Cartesian↔fractional helpers run in the fixed order shown in the
  algorithm pseudocode. The bit pattern of the result is identical across
  runs given identical inputs on the same hardware.

## Out of Scope <!-- rq-987dc616 -->

- NPT ensembles, barostats, and deformable cell tensors. `set_lattice` is
  the underlying primitive that future ensemble code drives;
  ensemble-level orchestration (when to mutate, how the box couples to a
  piston, etc.) is its own feature.
- Anisotropic or strain-tensor mutators (e.g. `scale_isotropic`,
  `scale_per_axis`, shear). Callers compose them out of `set_lattice`
  until a barostat-specific API is needed.
- Reduced-tilt enforcement (LAMMPS-style `|xy| <= lx/2`, etc.) and the
  associated "box flip" remapping. Tilts are unbounded in magnitude; the
  only operational constraint is `min_perpendicular_width / 2 > max
  interaction cutoff`, enforced by the neighbor list rather than the box.
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
    Given an orthorhombic SimulationBox constructed via
      SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0)

  # --- Construction: orthorhombic ---

  @rq-27ffd3f4
  Scenario: Construct an orthorhombic box (all tilts zero)
    When SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0) is called
    Then it returns Ok(box)
    And box.lattice() equals [10.0, 8.0, 6.0, 0.0, 0.0, 0.0]
    And box.lx() equals 10.0
    And box.ly() equals 8.0
    And box.lz() equals 6.0
    And box.xy() equals 0.0
    And box.xz() equals 0.0
    And box.yz() equals 0.0
    And box.generation() equals 0

  @rq-e1b51bd9
  Scenario: volume returns lx * ly * lz regardless of tilts
    Given a SimulationBox constructed via SimulationBox::new(2.0, 3.0, 5.0, 7.0, -9.0, 11.0)
    Then box.volume() equals 30.0

  # --- Construction: triclinic ---

  @rq-7a1c24be
  Scenario: Construct a triclinic box with non-zero tilts
    When SimulationBox::new(10.0, 8.0, 6.0, 1.5, -2.0, 0.5) is called
    Then it returns Ok(box)
    And box.lattice() equals [10.0, 8.0, 6.0, 1.5, -2.0, 0.5]

  @rq-67c5a863
  Scenario: Tilts may be negative
    When SimulationBox::new(10.0, 8.0, 6.0, -3.0, -5.0, -1.5) is called
    Then it returns Ok(box)
    And box.xy() equals -3.0
    And box.xz() equals -5.0
    And box.yz() equals -1.5

  @rq-650875cc
  Scenario: Tilts may exceed the corresponding diagonals
    When SimulationBox::new(2.0, 3.0, 4.0, 50.0, 50.0, 50.0) is called
    Then it returns Ok(box)
    (no reduced-tilt enforcement; geometric infeasibility is caught by
     the neighbor list via min_perpendicular_width)

  # --- Construction: rejection ---

  @rq-8259c9ca
  Scenario: Reject zero lx
    When SimulationBox::new(0.0, 8.0, 6.0, 0.0, 0.0, 0.0) is called
    Then it returns Err(SimulationBoxError::NonPositiveDiagonal { name: "lx", value: 0.0 })

  @rq-05eb9fbb
  Scenario: Reject zero ly
    When SimulationBox::new(10.0, 0.0, 6.0, 0.0, 0.0, 0.0) is called
    Then it returns Err(SimulationBoxError::NonPositiveDiagonal { name: "ly", value: 0.0 })

  @rq-74aa3a99
  Scenario: Reject zero lz
    When SimulationBox::new(10.0, 8.0, 0.0, 0.0, 0.0, 0.0) is called
    Then it returns Err(SimulationBoxError::NonPositiveDiagonal { name: "lz", value: 0.0 })

  @rq-9b1f8a7c
  Scenario: Reject negative diagonal
    When SimulationBox::new(-1.0, 8.0, 6.0, 0.0, 0.0, 0.0) is called
    Then it returns Err(SimulationBoxError::NonPositiveDiagonal { name: "lx", value: -1.0 })

  @rq-19fe4806
  Scenario: Reject NaN diagonal
    When SimulationBox::new(f32::NAN, 8.0, 6.0, 0.0, 0.0, 0.0) is called
    Then it returns Err(SimulationBoxError::NonFiniteLatticeValue { name: "lx", value: v }) where v is NaN

  @rq-7f867e37
  Scenario: Reject infinite diagonal
    When SimulationBox::new(10.0, f32::INFINITY, 6.0, 0.0, 0.0, 0.0) is called
    Then it returns Err(SimulationBoxError::NonFiniteLatticeValue { name: "ly", value: f32::INFINITY })

  @rq-0c9dc32b
  Scenario: Reject NaN tilt
    When SimulationBox::new(10.0, 8.0, 6.0, f32::NAN, 0.0, 0.0) is called
    Then it returns Err(SimulationBoxError::NonFiniteLatticeValue { name: "xy", value: v }) where v is NaN

  @rq-5318db55
  Scenario: Reject infinite tilt
    When SimulationBox::new(10.0, 8.0, 6.0, 0.0, f32::INFINITY, 0.0) is called
    Then it returns Err(SimulationBoxError::NonFiniteLatticeValue { name: "xz", value: f32::INFINITY })

  @rq-7541fd8a
  Scenario: Validation order is lx, ly, lz, xy, xz, yz
    When SimulationBox::new(0.0, -1.0, 0.0, f32::NAN, 0.0, 0.0) is called
    Then it returns Err(SimulationBoxError::NonPositiveDiagonal { name: "lx", value: 0.0 })

  @rq-b9a4e3de
  Scenario: Non-finite check precedes non-positive check on a diagonal
    When SimulationBox::new(f32::NAN, 8.0, 6.0, 0.0, 0.0, 0.0) is called
    Then it returns Err(SimulationBoxError::NonFiniteLatticeValue { name: "lx", value: v }) where v is NaN

  # --- minimum_image: orthorhombic special case ---

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
    Then the result equals [-4.0, 0.0, 0.0]

  @rq-a8df30ac
  Scenario: minimum_image just past -L/2 wraps by one period
    When box.minimum_image([-6.0, 0.0, 0.0]) is called
    Then the result equals [4.0, 0.0, 0.0]

  @rq-0ae304bc
  Scenario: minimum_image handles many-period displacements (orthorhombic)
    When box.minimum_image([24.0, 0.0, 0.0]) is called
    Then result_x lies in [-5.0, 5.0)
    And result_x equals 24.0 - 10.0 * floor((24.0 + 5.0) / 10.0)

  @rq-c9618bdd
  Scenario: minimum_image is per-axis independent for an orthorhombic box
    Given displacement = [6.0, -5.0, 4.0]
    When box.minimum_image(displacement) is called
    Then the x-component is wrapped against lx=10.0
    And the y-component is wrapped against ly=8.0
    And the z-component is wrapped against lz=6.0

  # --- minimum_image: triclinic ---

  @rq-b4e4bdc7
  Scenario: minimum_image of a c-aligned displacement subtracts the c lattice vector
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 0.0, 2.0, 3.0)
    When box.minimum_image([2.0, 3.0, 4.0]) is called
    Then the result equals [0.0, 0.0, -2.0]
      (fractional coords are (s_a, s_b, s_c) ≈ (0.067, 0.125, 0.667);
       k = (0, 0, 1); v − c = (2 − xz, 3 − yz, 4 − lz) = (0, 0, −2))

  @rq-261fde88
  Scenario: minimum_image of a displacement that requires b-tilt cancellation
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 1.0, 0.0, 0.0)
    When box.minimum_image([0.0, 5.0, 0.0]) is called
    Then the result equals [-1.0, -3.0, 0.0]
      (fractional coords are (s_a, s_b, s_c) ≈ (−0.0625, 0.625, 0.0);
       k = (0, 1, 0); v − b = (0 − xy, 5 − ly, 0) = (−1, −3, 0))

  @rq-a9ab33a8
  Scenario: A wrap result lies inside the primary parallelepiped
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 4.0, -3.0, 1.5)
    And an arbitrary displacement v
    When result = box.minimum_image(v) is called
    Then box.fractional_coords(result) has every component in [-0.5, 0.5)

  # --- wrap_position ---

  @rq-3e8324c2
  Scenario: wrap_position leaves a position inside the primary cell unchanged
    Given position = [4.0, 3.0, 2.0]
    When box.wrap_position(position) is called
    Then the result equals [4.0, 3.0, 2.0]

  @rq-4b9d059e
  Scenario: wrap_position wraps a position outside the primary cell (orthorhombic)
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

  @rq-5269221c
  Scenario: wrap_position is idempotent for a triclinic box
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 1.5, -2.0, 0.5)
    And position = [200.0, -150.0, 75.5]
    When wrapped_once = box.wrap_position(position)
    And wrapped_twice = box.wrap_position(wrapped_once)
    Then wrapped_twice equals wrapped_once

  @rq-a1fc0841
  Scenario: wrap_position and minimum_image agree on the same input
    Given v = [17.0, -13.0, 9.5]
    When mi = box.minimum_image(v)
    And wp = box.wrap_position(v)
    Then mi equals wp

  # --- wrap_position_with_image_count ---

  @rq-870ed681
  Scenario: wrap_position_with_image_count returns the image triple together with the wrapped position
    Given position = [12.0, 0.0, 0.0] and an orthorhombic SimulationBox with lx=10.0
    When (wrapped, image) = box.wrap_position_with_image_count(position)
    Then wrapped equals [2.0, 0.0, 0.0]
    And image equals [1, 0, 0]

  @rq-5355f3f0
  Scenario: wrap_position_with_image_count tracks per-direction image counts for a triclinic box
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 1.0, 2.0, 3.0)
    And position = [0.0, 0.0, 20.0] (k_c = 3 to wrap into fractional [-0.5, 0.5))
    When (wrapped, image) = box.wrap_position_with_image_count(position)
    Then image[2] equals 3
    And box.fractional_coords(wrapped) has every component in [-0.5, 0.5)
    And wrapped[0] equals 0.0 - 3 * 2.0 (the xz tilt subtraction propagates)
    And wrapped[1] equals 0.0 - 3 * 3.0 - k_b * 8.0 (the yz tilt subtraction then a possible y wrap)

  @rq-6c52e57d
  Scenario: Unwrap invariant holds across wrap_position_with_image_count
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 1.0, 2.0, 3.0)
    And an arbitrary position p
    When (wrapped, image) = box.wrap_position_with_image_count(p)
    Then wrapped + image[0]*(lx, 0, 0) + image[1]*(xy, ly, 0) + image[2]*(xz, yz, lz) equals p
      (within f32 round-off; the wrap is exact for displacements representable in f32)

  # --- Fractional coordinates ---

  @rq-545c961a
  Scenario: fractional_coords inverts cartesian_coords
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 1.0, 2.0, 3.0)
    And an arbitrary fractional triple s = [0.1, -0.2, 0.3]
    When v = box.cartesian_coords(s)
    And s_round = box.fractional_coords(v)
    Then s_round agrees with s within f32 round-off

  @rq-7f018040
  Scenario: cartesian_coords of unit fractional triples yields the lattice vectors
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 1.0, 2.0, 3.0)
    When box.cartesian_coords([1.0, 0.0, 0.0]) is called
    Then the result equals [10.0, 0.0, 0.0] (the a vector)
    When box.cartesian_coords([0.0, 1.0, 0.0]) is called
    Then the result equals [1.0, 8.0, 0.0] (the b vector)
    When box.cartesian_coords([0.0, 0.0, 1.0]) is called
    Then the result equals [2.0, 3.0, 6.0] (the c vector)

  # --- min_perpendicular_width ---

  @rq-ef6ae25a
  Scenario: min_perpendicular_width equals min(lx, ly, lz) for an orthorhombic box
    Given an orthorhombic SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0)
    Then box.min_perpendicular_width() equals 6.0

  @rq-47e800e0
  Scenario: min_perpendicular_width of a c-tilted box reflects the yz contribution
    Given a SimulationBox::new(10.0, 10.0, 10.0, 0.0, 0.0, 10.0)
    Then box.min_perpendicular_width() equals (ly * lz) / sqrt(lz² + yz²)
      = 100.0 / sqrt(200.0)
    And this value is less than 10.0

  @rq-9c3ecf3f
  Scenario: min_perpendicular_width of an xy-tilted box reflects the xy contribution
    Given a SimulationBox::new(10.0, 10.0, 10.0, 5.0, 0.0, 0.0)
    Then box.min_perpendicular_width() equals (lx*ly*lz) / sqrt((ly*lz)² + (xy*lz)² + 0²)
      = 1000.0 / sqrt(12500.0)
    And this value is less than 10.0

  # --- Numerical edge cases ---

  @rq-4b63564b
  Scenario: NaN displacement propagates to NaN output
    When box.minimum_image([f32::NAN, 0.0, 0.0]) is called
    Then result_x is NaN
    And result_y equals 0.0
    And result_z equals 0.0

  @rq-74f48855
  Scenario: NaN z-displacement propagates through tilt coupling for a triclinic box
    Given a SimulationBox::new(10.0, 8.0, 6.0, 0.0, 2.0, 3.0)
    When box.minimum_image([0.0, 0.0, f32::NAN]) is called
    Then every result component is NaN
      (k_c is NaN, propagated into the y channel via yz and the x channel via xz)

  # --- Generation counter ---

  @rq-2cb82d44
  Scenario: Newly-constructed box reports generation 0
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 1.0, 2.0, 3.0)
    Then box.generation() equals 0

  @rq-a3563587
  Scenario: Successful set_lattice increments generation by 1
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0)
    When box.set_lattice(12.0, 9.0, 7.0, 1.0, 2.0, 3.0) is called
    Then it returns Ok(())
    And box.lattice() equals [12.0, 9.0, 7.0, 1.0, 2.0, 3.0]
    And box.generation() equals 1

  @rq-9e09673b
  Scenario: Successive successful set_lattice calls increment generation monotonically
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0)
    When box.set_lattice(11.0, 8.0, 6.0, 0.0, 0.0, 0.0),
      then box.set_lattice(11.0, 9.0, 6.0, 0.0, 0.0, 0.0),
      then box.set_lattice(11.0, 9.0, 7.0, 1.0, 2.0, 3.0) are called in sequence
    Then every call returns Ok(())
    And box.lattice() equals [11.0, 9.0, 7.0, 1.0, 2.0, 3.0] after the third call
    And box.generation() equals 3 after the third call

  @rq-89c71321
  Scenario: set_lattice rejects a non-positive diagonal without mutating the box
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0)
    When box.set_lattice(0.0, 9.0, 7.0, 0.0, 0.0, 0.0) is called
    Then it returns Err(SimulationBoxError::NonPositiveDiagonal { name: "lx", value: 0.0 })
    And box.lattice() equals [10.0, 8.0, 6.0, 0.0, 0.0, 0.0]
    And box.generation() equals 0

  @rq-d28774dc
  Scenario: set_lattice rejects a non-finite diagonal without mutating the box
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0)
    When box.set_lattice(10.0, f32::NAN, 7.0, 0.0, 0.0, 0.0) is called
    Then it returns Err(SimulationBoxError::NonFiniteLatticeValue { name: "ly", value: v }) where v is NaN
    And box.lattice() equals [10.0, 8.0, 6.0, 0.0, 0.0, 0.0]
    And box.generation() equals 0

  @rq-50fa922c
  Scenario: set_lattice rejects a non-finite tilt without mutating the box
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0)
    When box.set_lattice(10.0, 8.0, 6.0, 1.0, f32::NAN, 0.0) is called
    Then it returns Err(SimulationBoxError::NonFiniteLatticeValue { name: "xz", value: v }) where v is NaN
    And box.lattice() equals [10.0, 8.0, 6.0, 0.0, 0.0, 0.0]
    And box.generation() equals 0

  @rq-153dd875
  Scenario: set_lattice validation order is lx, ly, lz, xy, xz, yz
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0)
    When box.set_lattice(0.0, -1.0, 0.0, f32::NAN, 0.0, 0.0) is called
    Then it returns Err(SimulationBoxError::NonPositiveDiagonal { name: "lx", value: 0.0 })
    And box.generation() equals 0

  @rq-7edab504
  Scenario: set_lattice non-finite check precedes non-positive check on a diagonal
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0)
    When box.set_lattice(f32::NAN, 9.0, 7.0, 0.0, 0.0, 0.0) is called
    Then it returns Err(SimulationBoxError::NonFiniteLatticeValue { name: "lx", value: v }) where v is NaN
    And box.generation() equals 0

  @rq-d6e10419
  Scenario: minimum_image after set_lattice reflects the new lattice parameters
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0)
    When box.set_lattice(20.0, 8.0, 6.0, 0.0, 0.0, 0.0) is called
    And box.minimum_image([12.0, 0.0, 0.0]) is called
    Then result_x equals -8.0
    (12.0 - 20.0; the wrap uses the post-mutation lx = 20.0, not 10.0)

  @rq-491235c1
  Scenario: minimum_image after set_lattice reflects new tilts
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0)
    When box.set_lattice(10.0, 8.0, 6.0, 0.0, 4.0, 0.0) is called
    And box.minimum_image([0.0, 0.0, 4.0]) is called
    Then k_c = floor((4.0 + 3.0) / 6.0) = 1
    And the result equals [-4.0, 0.0, -2.0]
    (the new xz=4.0 propagates into the x channel even though the displacement has v_x=0)

  @rq-fa98ca13
  Scenario: Copy of a SimulationBox carries the original's generation
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0)
    And box.set_lattice(11.0, 8.0, 6.0, 1.0, 0.0, 0.0) has been called once
    When let copy = box (a value copy via the Copy derive)
    Then copy.generation() equals 1
    And copy.lattice() equals [11.0, 8.0, 6.0, 1.0, 0.0, 0.0]

  @rq-22fb3b0e
  Scenario: Mutating a copy does not affect the original
    Given a SimulationBox constructed via SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0)
    And let copy = box (a value copy)
    When copy.set_lattice(20.0, 8.0, 6.0, 0.0, 0.0, 0.0) is called
    Then copy.lattice() equals [20.0, 8.0, 6.0, 0.0, 0.0, 0.0]
    And copy.generation() equals 1
    And box.lattice() equals [10.0, 8.0, 6.0, 0.0, 0.0, 0.0]
    And box.generation() equals 0

  # --- check_min_perpendicular_width ---

  @rq-0fa3b49f
  Scenario: check_min_perpendicular_width returns Ok when every width meets the threshold
    Given an orthorhombic SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0)
    When box.check_min_perpendicular_width(5.0) is called
    Then it returns Ok(())

  @rq-0061906c
  Scenario: check_min_perpendicular_width returns Ok at exact equality
    Given an orthorhombic SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0)
    When box.check_min_perpendicular_width(6.0) is called
    Then it returns Ok(()) (smallest width is 6.0, threshold is 6.0)

  @rq-394a4bb1
  Scenario: check_min_perpendicular_width flags direction "a" when w_a fails
    Given an orthorhombic SimulationBox::new(4.0, 10.0, 10.0, 0.0, 0.0, 0.0)
    When box.check_min_perpendicular_width(5.0) is called
    Then it returns Err(SimulationBoxError::PerpendicularWidthTooSmall {
      direction: "a", width: 4.0, required: 5.0 })

  @rq-7600d28c
  Scenario: check_min_perpendicular_width flags direction "b" when only w_b fails
    Given an orthorhombic SimulationBox::new(10.0, 4.0, 10.0, 0.0, 0.0, 0.0)
    When box.check_min_perpendicular_width(5.0) is called
    Then it returns Err(SimulationBoxError::PerpendicularWidthTooSmall {
      direction: "b", width: 4.0, required: 5.0 })

  @rq-5ffa0551
  Scenario: check_min_perpendicular_width flags direction "c" when only w_c fails
    Given an orthorhombic SimulationBox::new(10.0, 10.0, 4.0, 0.0, 0.0, 0.0)
    When box.check_min_perpendicular_width(5.0) is called
    Then it returns Err(SimulationBoxError::PerpendicularWidthTooSmall {
      direction: "c", width: 4.0, required: 5.0 })

  @rq-743ae35c
  Scenario: check_min_perpendicular_width reports the first failing direction when multiple fail
    Given an orthorhombic SimulationBox::new(4.0, 4.0, 4.0, 0.0, 0.0, 0.0)
    When box.check_min_perpendicular_width(5.0) is called
    Then it returns Err(SimulationBoxError::PerpendicularWidthTooSmall {
      direction: "a", width: 4.0, required: 5.0 })
    And the "b" and "c" directions are not reported

  @rq-8ac1a52f
  Scenario: check_min_perpendicular_width on a triclinic box uses perpendicular widths, not edge lengths
    Given a SimulationBox::new(10.0, 10.0, 10.0, 0.0, 0.0, 10.0)
    And w_b equals 100.0 / sqrt(200.0) ≈ 7.071
    When box.check_min_perpendicular_width(8.0) is called
    Then it returns Err(SimulationBoxError::PerpendicularWidthTooSmall {
      direction: "b", width: ≈ 7.071, required: 8.0 })

  @rq-98ac1915
  Scenario: check_min_perpendicular_width with non-positive threshold always returns Ok
    Given an orthorhombic SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0)
    When box.check_min_perpendicular_width(-1.0) is called
    Then it returns Ok(())
    When box.check_min_perpendicular_width(0.0) is called
    Then it returns Ok(())

  @rq-3eaf65b6
  Scenario: check_min_perpendicular_width is deterministic
    Given two SimulationBox instances constructed from identical six-parameter tuples
    When check_min_perpendicular_width(required) is called on each with the same required value
    Then both calls return byte-identical outcomes
```
