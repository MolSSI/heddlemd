# Feature: JIT-Composed Post-Force Per-Particle Kernel <!-- rq-8ac9773d -->

Every integrator, thermostat, and barostat slot whose post-force
work includes a one-thread-per-particle update exposes that work as
a CUDA source fragment. At runner-construction time the runner
collects the active fragments, JIT-compiles a single composed
per-particle kernel via `cudarc::nvrtc::compile_ptx_with_opts`, and
loads it on the device. Per-step, after the force evaluation
finishes and after each slot's scalar-prep work has written its
device-resident factor scalars, the runner launches the composed
kernel once. The composed kernel runs each fragment's per-thread
body in canonical order (integrator → thermostat → barostat),
collapsing what would otherwise be one launch per slot per step
into one launch covering every active post-force per-particle slot.

This file specifies the mechanism, the source-fragment contract,
the composed-kernel structure, the runner's dispatch contract, and
the determinism guarantees. The pair-force composer described in
`rqm/forces/jit-composed-pair-force.md` and the bonded / angle
composer described in `rqm/forces/jit-composed-intramolecular.md`
follow the same shape; this file describes the analogous mechanism
for the integration framework.

## Slot Participation <!-- rq-b85a38d6 -->

Three trait families participate:

- **`Integrator`** — the post-force fragment describes the
  integrator's final per-particle SubStep (the `KickHalf` or
  `KickDrift` immediately following `ForceEval` in the integrator's
  plan). Velocity-Verlet's post-force phase is one half-kick;
  Langevin-BAOAB's post-force phase contains a half-kick (and
  possibly an OU update). MTK-NPT's post-force phase contains the
  chain rescale and the half-kick. Every integrator in
  `IntegratorRegistry::with_builtins()` exposes a non-empty
  post-force fragment.

- **`Thermostat`** — the post-force fragment describes the
  thermostat's per-particle velocity rescale (or per-particle
  velocity resample). CSVR rescales by a device-resident factor
  scalar; Berendsen thermostat rescales by a similar scalar; NHC
  rescales by its chain-output scalar; Andersen draws a fresh
  Maxwell-Boltzmann velocity per particle whose Philox state is
  device-resident. Every thermostat in
  `ThermostatRegistry::with_builtins()` exposes a non-empty
  post-force fragment.

- **`Barostat`** — the post-force fragment describes the
  barostat's per-particle position and/or velocity rescale.
  C-rescale rescales both velocities and positions by separate
  device-resident factor scalars; Berendsen barostat rescales
  positions by its scalar. Every barostat in
  `BarostatRegistry::with_builtins()` exposes a non-empty
  post-force fragment.

When at least one of the three slot families is active and
configured, the composed kernel is built and the runner dispatches
it once per step. When all three families are absent (e.g. a
minimisation phase with no integrator step plan), the composer is
not invoked and no composed-kernel launch fires.

A slot that returns `None` from its post-force fragment method
when its corresponding configuration is present is rejected at
runner construction with
`StepError::MissingPostForcePerParticleFragment { label }`. Custom
user-registered slots that participate in the post-force phase
must expose a fragment or be replaced. Slots that genuinely do no
post-force per-particle work (a hypothetical pre-force-only
thermostat) return `None`, and the framework accepts that — the
composed kernel simply omits the corresponding section.

## Source-Fragment Contract <!-- rq-ce72fe43 -->

A `PerParticleFragment` carries CUDA C++ source plus identifying
metadata. The framework concatenates fragments' contributions into
one composed kernel; each fragment's per-thread body executes in
canonical slot order inside the per-particle thread.

```rust
pub struct PerParticleFragment {
    pub label: &'static str,
    pub helper_source: String,
    pub entry_point_args: String,
    pub per_thread_body: String,
}
```

Each field's role:

- `label` — the slot's stable identifier; matches the slot's
  human-facing name (e.g. `"velocity_verlet"`, `"csvr"`,
  `"c_rescale_barostat"`). Used to namespace the fragment's
  emitted helper symbols and to surface the slot in error messages.

- `helper_source` — CUDA C++ source declaring any helper
  `__device__` functions, structs, or `__shared__`-free constants
  the fragment's per-thread body depends on. Concatenated verbatim
  into the composed source above the entry point. Empty for
  fragments that need no helpers. Every helper symbol the fragment
  emits must be prefixed with the slot's label (or use a slot-
  scoped struct) so two fragments cannot collide.

- `entry_point_args` — CUDA C++ source declaring the fragment's
  contribution to the composed entry point's argument list. Each
  line declares one `extern "C"` kernel parameter, comma-
  terminated. The owning slot's `bind_post_force_per_particle_args`
  pushes one argument per declared parameter onto the launch
  builder in the same order.

- `per_thread_body` — CUDA C++ source for the fragment's
  per-thread work. The composer inlines this body into the
  composed kernel inside a fixed scope where the following
  variables are pre-declared and in scope:
  - `unsigned int i` — particle index (0-based, `i < n`).
  - `Real lx, ly, lz, xy, xz, yz` — the simulation box's six
    lattice parameters, read once at the top of the entry point.

  The body reads / writes particle state through pointer
  parameters declared in `entry_point_args` (positions,
  velocities, forces, masses, images, _lo buffers as needed). It
  uses only the precision shims (`Real`, `R(x)`, `Real_sqrt`,
  `Real_exp`, etc.) and the inlined PBC helpers
  (`heddle_jit_triclinic_wrap_with_image`, etc.) from the
  composer's preamble. It must not allocate shared memory, must
  not use atomics, and must not call `__syncthreads()`. It must
  be a pure function of its inputs (no static state, no global
  reads beyond the declared parameters).

The fragment carries no static state. The same composed kernel
runs every step for the `ForceField`'s lifetime; per-step values
(`dt`, factor scalars, Philox counter pointers) are passed as
kernel arguments and bound fresh per launch through the slot's
`bind_post_force_per_particle_args`.

## Composed-Kernel Structure <!-- rq-215c2fd9 -->

The composed kernel has the following shape:

```c
extern "C" __global__ void heddle_jit_composed_post_force_per_particle(
    /* common args */
    Real *positions_x, Real *positions_y, Real *positions_z,
    int *images_x, int *images_y, int *images_z,
    Real *velocities_x, Real *velocities_y, Real *velocities_z,
    const Real *forces_x, const Real *forces_y, const Real *forces_z,
    const Real *masses,
    const Real *lattice,
    /* per-fragment args, in canonical slot order */,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
    Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];

    /* Fragment per-thread bodies inlined in canonical slot order:
       1. Integrator's post-force body (e.g. vv_kick body).
       2. Thermostat's post-force body (e.g. CSVR rescale-velocities body).
       3. Barostat's post-force body (e.g. c-rescale velocity + position rescale). */
}
```

Canonical evaluation order: integrator → thermostat → barostat.
The integrator writes the final velocity; the thermostat reads &
writes velocity; the barostat reads & writes velocity and
positions. Each fragment's per-thread body sees the prior
fragments' writes through the per-thread variables they wrote.

Pre-step work (kinetic-energy reductions, scalar samples, mu
computations) runs as separate kernel launches before the composed
kernel. The runner orchestrates the sequence:

1. Force evaluation (J1 + J2 composed kernels + SPME-recip).
2. Thermostat's scalar prep — the existing `apply_post` is
   responsible for producing every scalar the thermostat's
   fragment reads (e.g. CSVR's factor scalar, NHC's chain output)
   and writing it to a device-resident buffer.
3. Barostat's scalar prep — same, for `apply`.
4. Composed-kernel launch (this kernel).
5. Post-rescale accumulators — barostat / thermostat injection
   deltas (no per-particle work; only scalar bookkeeping).

The composed kernel reads the prepared scalars as `Real *factor`
pointers passed through `entry_point_args` and dereferences
them in the per-thread body. The slot's `bind` method pushes the
factor pointer onto the launch builder.

When the integrator's post-force SubStep is a `KickDrift`
(unusual: drift after force eval), the integrator's per-thread
body includes the drift, which means the body writes positions and
applies the minimum-image wrap (via the inlined helper). The body
must also update `images_x/y/z` for the wrap. Velocity-Verlet's
post-force SubStep is `KickHalf` so its body writes only
velocities; positions and images are untouched until the next
step's pre-force `KickDrift`.

When zero post-force per-particle slots are active, the composer
is skipped and no composed-kernel launch fires.

## Compilation <!-- rq-532a9e35 -->

`SimulationSetup::new` (or whatever construction path the runner
uses to assemble integrator + thermostat + barostat slots)
performs the following at construction:

1. Collect the active integrator's, thermostat's (if any), and
   barostat's (if any) post-force per-particle fragments by calling
   each slot's `post_force_per_particle()` accessor and, for each
   `Some(p)`, reading `p.post_force_per_particle_fragment()`. Reject
   construction with
   `StepError::MissingPostForcePerParticleFragment { label }` if any
   built-in slot returns `None`.

2. Build the composed-kernel source by concatenating:
   1. The integration-shape preamble — the precision shims, PBC
      minimum-image helpers, and the per-particle block constants.
      Held verbatim as one `&'static str`.
   2. Each fragment's `helper_source`, in canonical slot order,
      with the slot's label prefixed onto every emitted helper
      symbol.
   3. The generated entry point: common args + per-fragment args
      (in canonical order) + `n` arg, with the entry-point body
      pre-reading the lattice and dispatching to each fragment's
      `per_thread_body` in order.

3. JIT-compile via `cudarc::nvrtc::compile_ptx_with_opts` with
   `--std=c++17` and the device's detected compute capability.
   Compile failure surfaces as
   `StepError::PostForceFragmentCompileFailed { log }`. Load
   failure surfaces as `StepError::PostForceFragmentLoadFailed`.

4. Load the compiled PTX under module name
   `"heddle_jit_composed_post_force_per_particle"` and resolve the
   single entry point `heddle_jit_composed_post_force_per_particle`
   into a `CudaFunction`. Hold the handle on the runner's setup
   state for the simulation's lifetime.

The composed-kernel source is not cached to disk. Every runner
construction recompiles; the ~100 ms cost is paid once at
simulation startup.

## Parameter Binding and Launch <!-- rq-ece2a1ec -->

Each slot's `bind_post_force_per_particle_args(builder, ctx)`
method pushes the slot's parameter buffers and scalars onto a
`ForceLaunchBuilder` in the order its fragment's
`entry_point_args` declares them. The runner constructs the
builder once per launch, pre-populates it with the common args
(positions, velocities, forces, masses, images, lattice in that
order — the same order the entry-point template uses), and then
calls each active slot's bind method in canonical order. The
trailing `n` arg is pushed last.

Slot-side fields (e.g. CSVR's `factor_device` scalar buffer) are
already populated by the slot's `apply_post` / `apply` work that
ran in the immediately-preceding pre-rescale phase. The bind
method just pushes the buffer's device pointer onto the launch
arg list.

The launch configuration is:

- block size: 256 threads (matches `vv_kick`'s standalone launch
  config).
- grid: `ceil(n / 256)` blocks.
- shared memory: zero bytes.
- stream: the default stream carried by `particle_buffers.device`.

The composed-kernel launch is recorded in `timings` under
`KernelStage::JitComposedPostForce`. The per-slot
`KernelStage::VV_KICK`, `KernelStage::CSVR_RESCALE_VELOCITIES`,
etc. stages do not appear in runs that go through the JIT path —
those standalone kernels do not exist.

## Determinism <!-- rq-6b628814 -->

The composed kernel preserves same-GPU bit-exact reproducibility:

1. *Composition order is deterministic.* Fragments are
   concatenated in canonical slot order (integrator → thermostat
   → barostat). Two runner constructions from byte-identical
   configurations produce byte-identical composed source. nvrtc
   with fixed flags produces byte-identical PTX from byte-
   identical input.

2. *Per-thread evaluation is independent.* Each thread handles
   one particle's worth of state. No atomics, no shared memory,
   no inter-thread reads. The slot order inside the per-thread
   body fixes the in-register arithmetic order.

3. *Read-of-prepared-scalars is deterministic.* The scalar
   factors that each fragment reads are produced by the
   immediately-preceding `apply_post` / `apply` kernel sequence,
   which runs on the same default stream. CUDA's implicit
   per-stream ordering guarantees the scalar is committed before
   the composed kernel reads it.

Cross-configuration equality with the legacy per-kernel path
(separate `vv_kick` + `csvr_rescale_velocities` + `c_rescale_rescale_velocities` launches) is not a property: the
composed kernel's in-register state-update order differs from the
back-to-back-launch path's, so the two produce results that agree
only within `f32` round-off. Same-configuration run-to-run byte
equality is preserved.

## Slot Behaviour Contract <!-- rq-c71470d0 -->

When a slot exposes a post-force fragment, its existing trait
method behaves as follows:

- `Integrator::execute(SubStep::KickHalf | SubStep::KickDrift,
  ...)` for the post-force SubStep — **no-op at the
  kernel-launch level**. The body the standalone kernel would
  have launched is now part of the composed kernel. The
  integrator may still update internal counters or bookkeeping;
  it must not launch a per-particle kernel.

- `Thermostat::apply_post(...)` — runs every part of the
  thermostat's post-force work EXCEPT the per-particle rescale.
  For CSVR: kinetic-energy reduction + sample-and-factor
  (producing the factor scalar). For NHC: chain integration
  producing the rescale scalar. For Andersen: any Philox-counter
  bookkeeping. The per-particle rescale / resample is dispatched
  via the composed kernel, not by `apply_post`.

- `Barostat::apply(...)` — runs every part of the barostat's
  work EXCEPT the per-particle rescale (velocities + positions).
  For c-rescale: virial reduction + mu compute + injection
  accumulator bookkeeping. For Berendsen barostat: mu compute +
  box mutation + injection accumulator bookkeeping (the box
  mutation must happen before the composed kernel runs so the
  composed kernel reads the new lattice). The per-particle
  rescale is dispatched via the composed kernel.

These contracts are part of each slot's `Potential`-equivalent
trait surface. The slot's documentation file (e.g.
`velocity-verlet.md`, `csvr.md`, `c-rescale-barostat.md`)
specifies how its `execute` / `apply_post` / `apply` is
restructured.

## Runner Integration <!-- rq-c0384b03 -->

The runner's per-step loop (see `framework.md`'s *Per-Step
Interface*) reads as follows when the composed-kernel path is
active:

```text
loop step in 1..=n_steps:
    if let Some(t) = thermostat { t.apply_pre(buffers, dt, timings) }
    let plan = integrator.plan(dt)
    for sub in &plan.steps:
        match sub:
            SubStep::ForceEval{..} => force_field.step(...)
            SubStep::KickHalf{..} | SubStep::KickDrift{..}
                if this is the post-force SubStep:
                    /* skipped — composed kernel handles it */
            other => integrator.execute(other, ...)
    /* Post-force composed-kernel dispatch */
    if let Some(t) = thermostat { t.apply_post(...) }  /* scalar prep */
    if let Some(b) = barostat   { b.apply(...) }       /* scalar prep + box mutation */
    launch_post_force_composed_kernel(buffers, integrator, thermostat, barostat)
    /* output cadence */
```

The runner recognises the "post-force" SubStep as the final
per-particle SubStep that follows the last `ForceEval` in the
plan. For velocity-Verlet that is the last `KickHalf`; for
Langevin-BAOAB the post-force phase may contain more substeps
(the `O` step plus the trailing `A`), in which case those
substeps' bodies are folded into the integrator's fragment's
`per_thread_body` as a single composed action. The integrator's
plan still names the substeps for documentation / log purposes
but the runner consults the fragment to know which substeps were
folded.

For phases without a post-force composed kernel (no integrator
plan, e.g. SD minimisation), the composed-kernel launch is
skipped entirely; the runner's loop reverts to its non-JIT shape.

## Feature API <!-- rq-edf9864f -->

### Types <!-- rq-9e9db6d9 -->

- `PerParticleFragment` — see *Source-Fragment Contract* above. <!-- rq-d2cacf91 -->

- `PostForceBindContext<'a>` — context passed to every active <!-- rq-5c607daa -->
  slot's `bind_post_force_per_particle_args(...)` call. Exposes
  references to per-step inputs (positions, velocities, etc. via
  `ParticleBuffers`; lattice via `SimulationBox`; `dt`).

  ```rust
  pub struct PostForceBindContext<'a> {
      pub buffers: &'a ParticleBuffers,
      pub sim_box: &'a SimulationBox,
      pub dt: Real,
  }
  ```

- `ForceLaunchBuilder` — reused from <!-- rq-7a000f0e -->
  `rqm/forces/jit-composed-pair-force.md`. The launch builder
  is shape-agnostic; the same type carries arguments for every
  composer.

- `JitComposedPostForcePerParticle` — module + entry-point handle <!-- rq-56a36cba -->
  owned by the runner's setup state. Fields:

  ```rust
  pub struct JitComposedPostForcePerParticle {
      pub fragment_labels: Vec<&'static str>,
      pub entry_point: CudaFunction,
  }
  ```

### Error variants <!-- rq-b929e7e0 -->

`StepError` carries variants for the J3 mechanism:

- `MissingPostForcePerParticleFragment { label: &'static str }` — <!-- rq-88b188d3 -->
  a built-in slot in the runner's active configuration returned
  `None` from `post_force_per_particle()`. Reported from runner
  construction.

- `PostForceFragmentCompileFailed { log: String }` — nvrtc <!-- rq-9ebdaea4 -->
  rejected the composed source.

- `PostForceFragmentLoadFailed(GpuError)` — `load_ptx` rejected <!-- rq-96749659 -->
  the compiled PTX.

### Trait methods <!-- rq-ba5e545b -->

`Integrator`, `Thermostat`, and `Barostat` each carry one accessor
that declares post-force participation:

```rust
fn post_force_per_particle(&self) -> Option<&dyn PostForcePerParticle> {
    None
}
```

The default returns `None` (the slot does not participate). A slot
that participates implements the `PostForcePerParticle` capability
trait (defined in `integration/framework.md`'s *Feature API*) and
returns `Some(self)`:

```rust
pub trait PostForcePerParticle {
    fn post_force_per_particle_fragment(&self) -> PerParticleFragment;
    fn bind_post_force_per_particle_args(
        &self,
        ctx: &PostForceBindContext<'_>,
        builder: &mut ForceLaunchBuilder,
    );
}
```

Neither capability method has a default. Because the fragment and the
binding live on one trait, a slot that participates supplies both —
it cannot expose a fragment without a binding, and a non-participating
slot implements neither. The framework reads `post_force_per_particle()`
once at runner construction to collect each participant's
`post_force_per_particle_fragment()`, and calls
`bind_post_force_per_particle_args` at each launch.

### Composed-module name and entry point <!-- rq-dd480027 -->

The CUDA module loaded into the device carries the name
`"heddle_jit_composed_post_force_per_particle"`. It exposes one
`extern "C"` kernel:

- `heddle_jit_composed_post_force_per_particle` — the composed <!-- rq-02d0e4f9 -->
  post-force per-particle kernel. Block size 256, grid
  `ceil(n / 256)`, no shared memory.

There is no separate `_f` vs `_fev` variant — the post-force
per-particle work depends on velocities / positions / forces but
not on the `AggregateLevel` of the prior force evaluation.

## Out of Scope <!-- rq-a77e722d -->

- **Pre-force phase composition.** The pre-force `KickDrift` /
  `Drift` SubSteps (and any pre-force thermostat / barostat work)
  stay as separate launches via the existing
  `integrator.execute(sub, ...)` / `thermostat.apply_pre(...)`
  path. Pre-force composition is a separate feature that follows
  the same pattern; it would JIT-compose a parallel
  `heddle_jit_composed_pre_force_per_particle` kernel. K
  (multi-step persistent loop) will likely re-open this question.

- **Composition of scalar-reduction work** (kinetic-energy reduce,
  virial reduce, scalar sample / mu compute). These remain
  standalone kernels because they are shape-universal across
  slots and their per-step launch count is already small (one per
  thermostat or barostat per step). The composed kernel reads
  their device-resident output scalars but does not include their
  computation.

- **Composition of the SHAKE / RATTLE constraint hooks.**
  Constraint slots (`constraint-framework.md`) install hooks
  before / after Drift substeps; those hooks iterate to
  convergence and do not fit the single-pass per-particle
  fragment model. They stay outside the composed kernel.

- **Composition of pre-force or mid-plan stochastic substeps.**
  Langevin-BAOAB's mid-plan OU step runs between drift substeps
  and is not part of the post-force phase. It stays as a separate
  launch via `integrator.execute(SubStep::Custom { label: "ou",
  ... }, ...)`.

- **Per-particle fragments from `Constraint` slots.** Constraint
  slots have their own framework (`constraint-framework.md`); J3
  does not extend the fragment mechanism to them.

- **On-disk PTX caching of the composed module.** Same policy as
  the pair-force composer.

- **Hot-reload of the composed module mid-run.** The slot list is
  fixed at runner construction; the module is loaded once.

- **Multiple composed kernels per step phase.** J3 produces
  exactly one composed kernel for the post-force phase. If
  multiple per-particle phases are introduced later (pre-force,
  mid-plan), each phase has its own composed kernel.

## Gherkin Scenarios <!-- rq-1c911e56 -->

```gherkin
Feature: JIT-composed post-force per-particle kernel

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called

  # --- Construction ---

  @rq-b3c1def1
  Scenario: Composed kernel is compiled when an integrator is active
    Given a SimulationSetup with VelocityVerlet integrator only
      (no thermostat, no barostat)
    When the runner is constructed
    Then it exposes a CudaFunction handle for
      "heddle_jit_composed_post_force_per_particle"
    And the loaded module name equals "heddle_jit_composed_post_force_per_particle"

  @rq-9a8a7dfa
  Scenario: Composed kernel is compiled with thermostat + barostat fragments
    Given a SimulationSetup with VelocityVerlet + CSVR + c-rescale
    When the runner is constructed
    Then the composed source contains every active slot's fragment in
      canonical order [velocity_verlet, csvr, c_rescale_barostat]
    And the composed kernel is loaded successfully

  @rq-dcd0d421
  Scenario: Construction is rejected when a built-in slot does not participate
    Given a SimulationSetup with a custom integrator whose
      post_force_per_particle() returns None
    When the runner is constructed
    Then it returns Err(StepError::MissingPostForcePerParticleFragment
      { label: <slot's label> })

  @rq-8a7ef593
  Scenario: Composed kernel is not compiled when no integrator is active
    Given a SimulationSetup with no integrator plan (e.g. minimisation only)
    When the runner is constructed
    Then no composed post-force kernel module is loaded
    And no nvrtc compile is invoked

  @rq-b4788314
  Scenario: FragmentCompileFailed surfaces every active fragment's label
    Given an active slot whose fragment's per_thread_body contains a
      deliberate syntax error
    When the runner is constructed
    Then it returns Err(StepError::PostForceFragmentCompileFailed { log })
    And log contains every active slot's label

  # --- Per-step dispatch ---

  @rq-8bfffd42
  Scenario: One step launches the composed kernel exactly once
    Given a runner with VelocityVerlet + CSVR + c-rescale active
    When the runner runs one timestep
    Then timings records exactly one sample for KernelStage::JitComposedPostForce
    And timings records zero samples for KernelStage::VV_KICK
    And timings records zero samples for KernelStage::CSVR_RESCALE_VELOCITIES
    And timings records zero samples for KernelStage::C_RESCALE_RESCALE_VELOCITIES

  @rq-e12c2668
  Scenario: Slot bind methods are invoked once each in canonical order
    Given a runner with three active slots [A=integrator, B=thermostat, C=barostat],
      each with instrumented bind methods that record their invocation order
    When the runner runs one timestep
    Then A's bind_post_force_per_particle_args is invoked before B's
    And B's bind_post_force_per_particle_args is invoked before C's
    And each bind method is invoked exactly once per step

  @rq-86dea9a1
  Scenario: Thermostat's apply_post runs the scalar prep but not the rescale
    Given a runner with CSVR thermostat active
    And CSVR's apply_post is instrumented to record kernel-launch counts
    When the runner runs one timestep
    Then CSVR's apply_post launched compute_kinetic_energy and csvr_sample_and_factor
    And CSVR's apply_post did NOT launch any per-particle rescale kernel

  @rq-56044cc3
  Scenario: Barostat's apply runs scalar prep but not per-particle rescale
    Given a runner with c-rescale barostat active
    When the runner runs one timestep
    Then c-rescale's apply launched virial_sum_reduce and c_rescale_compute_mu
    And c-rescale's apply did NOT launch rescale_velocities_device_factor
    And c-rescale's apply did NOT launch rescale_positions_device_factor

  # --- Correctness ---

  @rq-d6d4f598
  Scenario: Composed-kernel output matches the legacy launch sequence within f32 round-off
    Given the same physical state run two ways:
      (a) the J3 composed-kernel path
      (b) the legacy per-slot kernel sequence (vv_kick → csvr_rescale → c_rescale_rescale)
    When one timestep is run on each
    Then per-particle positions, velocities agree within 1e-5 relative tolerance
    But the per-particle outputs are NOT byte-identical across (a) and (b)

  @rq-f3373134
  Scenario: Two independent runs of the composed-kernel path are byte-identical
    Given two independently-constructed runners with byte-identical configurations
    And two ParticleBuffers built from byte-identical ParticleStates
    When each runs the same number of timesteps
    Then per-particle positions, velocities, images, _lo buffers agree
      byte-for-byte across the two runs

  # --- Per-fragment evaluation order ---

  @rq-9c5226e5
  Scenario: Integrator kick runs before thermostat rescale (canonical order)
    Given a runner with VelocityVerlet + CSVR
    And the CSVR factor scalar is set artificially to 0.5
    When one timestep is run
    Then the post-step velocity equals 0.5 * (pre-step velocity + a · dt/2)
      within f32 round-off
    (The integrator's kick updates v first; then the thermostat's rescale
     reads the updated v and scales it.)

  @rq-ae74d89b
  Scenario: Barostat rescale runs after integrator + thermostat (canonical order)
    Given a runner with VelocityVerlet + CSVR + c-rescale
    And the c-rescale velocity scalar is set artificially to 1.1
    When one timestep is run
    Then the post-step velocity equals 1.1 * (CSVR-rescaled velocity)
      within f32 round-off

  # --- Empty state ---

  @rq-bf2e99c3
  Scenario: A runner with no active integrator / thermostat / barostat skips the composed kernel
    Given a SimulationSetup whose phase has no step plan (minimisation only)
    When the phase runs
    Then no composed post-force kernel launch is recorded

  # --- Standalone-kernel retirement ---

  @rq-e274d0d2
  Scenario: kernels/integrate.cu does not declare a vv_kick entry point
    Given the project's kernel source tree
    When the integrate-shape standalone kernel symbols are enumerated
    Then no extern "C" kernel named vv_kick exists
    And no extern "C" kernel named vv_kick_lossless exists
    And vv_kick_drift and vv_kick_drift_lossless are declared
      (the pre-force phase is out of scope for J3)

  @rq-a57fd4d5
  Scenario: csvr_rescale_velocities standalone kernel does not exist
    Given the project's kernel source tree
    When the thermostat-shape standalone kernel symbols are enumerated
    Then no extern "C" kernel named csvr_rescale_velocities exists
    And kinetic_energy_reduce and csvr_sample_and_factor still exist
      (the scalar-prep kernels stay standalone)

  @rq-33fa8597
  Scenario: c-rescale rescale_velocities and rescale_positions standalone kernels do not exist
    Given the project's kernel source tree
    When the barostat-shape standalone kernel symbols are enumerated
    Then no extern "C" kernel named rescale_velocities_device_factor exists
    And no extern "C" kernel named rescale_positions_device_factor exists
    And virial_sum_reduce and c_rescale_compute_mu still exist
```
