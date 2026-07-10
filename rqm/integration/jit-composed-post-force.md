# Feature: JIT-Composed Post-Force Per-Particle Kernel <!-- rq-8ac9773d -->

Every integrator and barostat slot whose post-force work includes a
one-thread-per-particle update exposes that work as a CUDA source
fragment. At runner-construction time the runner collects the active
fragments, JIT-compiles a single composed per-particle kernel via
`cudarc::nvrtc::compile_ptx_with_opts`, and loads it on the device.
Per-step, after the force evaluation finishes and after each slot's
scalar-prep work has written its device-resident factor scalars, the
runner launches the composed kernel once. The composed kernel runs each
fragment's per-thread body in canonical order (integrator → barostat),
collapsing what would otherwise be one launch per slot per step into one
launch covering every active post-force per-particle slot.

Thermostats do not participate in this kernel. A thermostat's coupling
reduces the full-step kinetic energy — the kinetic energy of the
velocities after the integrator's trailing kick — so the reduction is a
fusion barrier between the trailing kick and the rescale
(`docs/architecture.md`, *Step orchestration and kernel fusion*): the
two cannot share a launch. The thermostat runs its reduction and rescale
as their own launches on coupling steps, and on those steps the composed
kernel is not used at all (see *Composed-Kernel Use and Coupling Steps*).

This file specifies the mechanism, the source-fragment contract,
the composed-kernel structure, the runner's dispatch contract, and
the determinism guarantees. The pair-force composer described in
`rqm/forces/jit-composed-pair-force.md` and the bonded / angle
composer described in `rqm/forces/jit-composed-intramolecular.md`
follow the same shape; this file describes the analogous mechanism
for the integration framework.

## Slot Participation <!-- rq-b85a38d6 -->

Two trait families participate:

- **`Integrator`** — the post-force fragment describes the
  integrator's final per-particle SubStep (the `KickHalf` or
  `KickDrift` immediately following `ForceEval` in the integrator's
  plan). Velocity-Verlet's post-force phase is one half-kick;
  Langevin-BAOAB's post-force phase contains a half-kick (and
  possibly an OU update). MTK-NPT's post-force phase contains the
  chain rescale and the half-kick. Every integrator in
  `IntegratorRegistry::with_builtins()` exposes a non-empty
  post-force fragment.

- **`Barostat`** — the post-force fragment describes the
  barostat's per-particle position and/or velocity rescale.
  C-rescale rescales both velocities and positions by separate
  device-resident factor scalars; Berendsen barostat rescales
  positions by its scalar. Every barostat in
  `BarostatRegistry::with_builtins()` exposes a non-empty
  post-force fragment.

A **`Thermostat`** never participates. Its post-force velocity rescale
(or resample) follows a full-step kinetic-energy reduction that reads the
post-trailing-kick velocities; that reduction is a fusion barrier
(`docs/architecture.md`), so the thermostat runs its reduction and
rescale as standalone launches in `apply_post` (`framework.md`,
`csvr.md`, `nose-hoover-chain.md`, `berendsen.md`, `andersen.md`).

Participation is optional per slot: a slot that returns `None`
from `post_force_per_particle()` runs its post-force work through
its natural eager path instead (the integrator's trailing kick
executes in the plan walk; a per-step barostat performs its own
rescale inside `apply`). Every integrator and barostat kind in the
built-in registries participates — that guarantee is enforced by a
registry lint test, not by a runtime rejection.

### Suffix-closed subset selection <!-- rq-09306735 -->

The eager execution order of post-force work is

```text
integrator trailing kick (plan walk)
    → per-step barostat (apply)
    → composed-kernel launch
```

Because the composed launch is last, a slot excluded from the
composed kernel runs *before* every composed fragment. The composed
set must therefore be a **suffix** of the configured order
`[integrator, barostat]` (unconfigured slots and periodic barostats
are absent from the order):

- both participate → full fusion (`{I, B}`);
- the integrator does not participate → its kick runs in the plan
  walk and `{B}` composes after it;
- no slot participates, or the maximal suffix is empty → no
  composed kernel is built and the plan walk executes every
  sub-step (the same shape as a phase with no post-force work).

The per-step barostat participates in the composed set only when the
plan carries a **terminal** `BarostatPoint` (its canonical placement,
`framework.md`). A barostat placed at an interleaved (mid-plan)
`BarostatPoint` is absent from the composed order — its `apply` runs
its full standalone rescale during the plan walk — so it neither
appears in the suffix nor forces the integrator out of it.

`skip_substep_index` (see `framework.md`, `RunStepOptions`) is set
to the plan's trailing-kick index **iff the integrator is in the
composed set**; a non-participating integrator's kick is never
skipped. This derivation makes a silently lost kick
unrepresentable.

Because a thermostat contributes no fragment and its rescale always
runs eagerly, every configured combination has a valid execution
order: an excluded integrator falls back to its plan-walk kick, and an
excluded barostat falls back to its standalone `apply` rescale. There
is no unsatisfiable topology and the runner performs no phase-setup
rejection on fusion grounds.

When a configured slot is excluded from the composed set, the
runner prints one explanatory line to stderr at phase setup naming
the slot and the resulting composed coverage, so an unexpected
loss of fusion is diagnosable.

### Composed-Kernel Use and Coupling Steps <!-- rq-73e738ee -->

The composed kernel is used only on steps where no thermostat
couples. On a **coupling step** — a step where a configured thermostat
acts, i.e. `step % coupling_interval == 0` (`framework.md`,
`io/config-schema.md`) — the full-step kinetic-energy reduction sits
between the trailing kick and the thermostat rescale, so the runner
does not launch the composed kernel: the trailing kick, the reduction,
the thermostat rescale, and any barostat rescale all run eagerly, in
canonical order `integrator → thermostat → barostat`. On a
non-coupling step (and on every step of an un-thermostatted run) the
composed kernel fuses the trailing kick with any barostat rescale as
described above. A run whose thermostat couples every step
(`coupling_interval == 1`) therefore never uses the composed kernel;
larger intervals restore fusion on the intervening steps.

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
       2. Barostat's post-force body (e.g. c-rescale velocity + position rescale). */
}
```

Canonical evaluation order: integrator → barostat. The integrator
writes the final velocity; the barostat reads & writes velocity and
positions. Each fragment's per-thread body sees the prior fragments'
writes through the per-thread variables they wrote.

Pre-step work (virial reductions, scalar samples, mu computations)
runs as separate kernel launches before the composed kernel. The
runner orchestrates the sequence:

1. Force evaluation (J1 + J2 composed kernels + SPME-recip).
2. Barostat's scalar prep — the existing `apply` produces every
   scalar the barostat's fragment reads (e.g. c-rescale's mu scalar)
   and writes it to a device-resident buffer.
3. Composed-kernel launch (this kernel).
4. Post-rescale accumulators — barostat injection deltas (no
   per-particle work; only scalar bookkeeping).

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
uses to assemble integrator + barostat slots) performs the following
at construction:

1. Collect the active integrator's and per-step barostat's (if any)
   post-force per-particle fragments by calling each slot's
   `post_force_per_particle()` accessor and, for each `Some(p)`,
   reading `p.post_force_per_particle_fragment()`. Select the maximal
   participating suffix of `[integrator, barostat]` (see *Slot
   Participation*); fragments outside the suffix are discarded and
   their slots run eagerly. An empty suffix skips steps 2–4 entirely
   (no composed kernel for the phase).

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
`KernelStage::VV_KICK` and the barostat rescale stages do not appear
for slots covered by the composed kernel — those standalone kernels do
not launch. A thermostat's standalone reduction and rescale stages
(e.g. `KernelStage::CSVR_RESCALE_VELOCITIES`) do appear, on coupling
steps, because a thermostat is never covered by the composed kernel.

## Determinism <!-- rq-6b628814 -->

The composed kernel preserves same-GPU bit-exact reproducibility:

1. *Composition order is deterministic.* Fragments are
   concatenated in canonical slot order (integrator → barostat).
   Two runner constructions from byte-identical configurations
   produce byte-identical composed source. nvrtc with fixed flags
   produces byte-identical PTX from byte-identical input.

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

Cross-configuration equality with the separate per-kernel path
(back-to-back `vv_kick` + `c_rescale_rescale_velocities` launches) is
not a property: the composed kernel's in-register state-update order
differs from the back-to-back-launch path's, so the two produce
results that agree only within `f32` round-off. Same-configuration
run-to-run byte equality is preserved.

## Slot Behaviour Contract <!-- rq-c71470d0 -->

When a slot exposes a post-force fragment, its existing trait
method behaves as follows:

- `Integrator::execute(SubStep::KickHalf | SubStep::KickDrift,
  ...)` for the post-force SubStep — **no-op at the
  kernel-launch level**. The body the standalone kernel would
  have launched is now part of the composed kernel. The
  integrator may still update internal counters or bookkeeping;
  it must not launch a per-particle kernel.

- `Thermostat::apply_post(...)` — a thermostat exposes no post-force
  fragment, so `apply_post` runs **every** part of the thermostat's
  post-force work, including the per-particle rescale, as its own
  kernel launches. For CSVR: full-step kinetic-energy reduction +
  sample-and-factor + the `rescale_velocities` launch. For NHC: chain
  integration + the rescale launch. For Andersen: the per-particle
  resample launch. It runs only on coupling steps (`framework.md`),
  after the integrator's trailing kick, so the reduction reads the
  full-step velocities.

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

The runner's per-step loop is specified in full by `framework.md`'s
*Per-Step Interface*; that pseudocode is the single authoritative
description and is not duplicated here. The points specific to the
composed-kernel path are:

- The composed kernel is launched only on a step that uses it — a
  non-coupling step (`step % coupling_interval != 0`, or an
  un-thermostatted run) with a participating integrator. On that step
  the runner sets `skip_substep_index` to the integrator's trailing
  kick, dispatches a deferred terminal `BarostatPoint`'s `apply` in the
  post-walk tail, launches the composed kernel, then fires any deferred
  terminal velocity projection.
- On a coupling step the composed kernel is not launched. The trailing
  kick runs in the plan walk, `thermostat.apply_post` performs the
  full-step reduction and rescale, a terminal `BarostatPoint`'s `apply`
  runs its full standalone rescale, and a trailing `AfterKick`
  projection dispatches during the walk like any other
  `ConstraintPoint`.

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

### Interaction with the constraint slot <!-- rq-d9290792 -->

The composed post-force kernel and the `Constraint` slot
(`constraint-framework.md`) compose without changing the suffix
selection above. A constraint slot never contributes a per-particle
fragment — its position and velocity projections iterate to
convergence and do not fit the single-pass per-particle model — so the
integrator's participation in the composed set is decided exactly as
for an unconstrained run.

The constraint slot's hooks run as their own kernel launches at the
plan-declared `ConstraintPoint` markers (`framework.md`). Their
placement relative to the composed launch is:

- `BeforeDrift` (snapshot) and `AfterDrift` (position projection)
  precede the last `ForceEval` and therefore run in the plan walk,
  ahead of the composed launch, in both the composed and non-composed
  paths.
- A trailing `AfterKick` (velocity projection) must follow the
  integrator's final kick. When the composed kernel absorbs that kick,
  the kick executes after the plan walk, so the runner defers the
  trailing `AfterKick` marker (`defer_terminal_velocity_projection`,
  `framework.md`) and fires `apply_after_kick` after
  `launch_post_force_composed_kernel`. The velocity projection is thus
  the last per-particle velocity operation of the step, after the
  composed kernel's fused kick and any barostat velocity rescale. On a
  coupling step the composed kernel is not used, so the trailing kick
  and the thermostat rescale run eagerly in the walk / `apply_post` and
  the trailing `AfterKick` follows them there (see
  `constraint-framework.md`, *Ordering of the terminal velocity
  projection*).

The integrator therefore stays in the composed set on constrained
runs, and kick / barostat fusion is preserved on non-coupling steps;
only the constraint projections (and, on coupling steps, the thermostat
reduction and rescale) run as extra launches, exactly as they do
without the composed kernel.

## Feature API <!-- rq-edf9864f -->

### Types <!-- rq-9e9db6d9 -->

- `PerParticleFragment` — see *Source-Fragment Contract* above. <!-- rq-d2cacf91 -->

- `PostForceBindContext<'a>` — context passed to every active <!-- rq-5c607daa -->
  slot's `bind_post_force_per_particle_args(...)` call. Exposes
  references to per-step inputs (positions, velocities, etc. via
  `ParticleBuffers`; lattice via `SimulationBox`; `dt`; the
  `ForceField`, through which a fragment binds class accumulator
  buffers — the RESPA fragment binds
  `force_field.class_forces(ForceClass::Slow)` for its trailing
  outer kick, see `respa.md`).

  ```rust
  pub struct PostForceBindContext<'a> {
      pub buffers: &'a ParticleBuffers,
      pub sim_box: &'a SimulationBox,
      pub force_field: &'a ForceField,
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

- `PostForceFragmentCompileFailed { log: String }` — nvrtc <!-- rq-9ebdaea4 -->
  rejected the composed source.

- `PostForceFragmentLoadFailed(GpuError)` — `load_ptx` rejected <!-- rq-96749659 -->
  the compiled PTX.

### Trait methods <!-- rq-ba5e545b -->

`Integrator` and `Barostat` each carry one accessor that declares
post-force participation:

```rust
fn post_force_per_particle(&self) -> Option<&dyn PostForcePerParticle> {
    None
}
```

The default returns `None` (the slot does not participate). A slot
that participates implements the `PostForcePerParticle` capability
trait (defined in `integration/framework.md`'s *Feature API*) and
returns `Some(self)`. `Thermostat` carries no such accessor: a
thermostat's rescale always follows a full-step kinetic-energy
reduction (a fusion barrier) and so runs as its own launch, never as a
composed fragment.

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
  Constraint slots (`constraint-framework.md`) run their hooks at
  plan-declared `ConstraintPoint` markers; those hooks iterate to
  convergence and do not fit the single-pass per-particle fragment
  model. They stay outside the composed kernel as their own launches;
  see *Interaction with the constraint slot* for how the trailing
  velocity projection is ordered relative to the composed launch.

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
  Scenario: Composed kernel is compiled with the barostat fragment
    Given a SimulationSetup with VelocityVerlet + c-rescale
    When the runner is constructed
    Then the composed source contains every active fragment in
      canonical order [velocity_verlet, c_rescale_barostat]
    And the composed source contains no thermostat fragment
    And the composed kernel is loaded successfully

  @rq-dcd0d421
  Scenario: A non-participating integrator's trailing kick executes in the plan walk
    Given a custom integrator whose post_force_per_particle() returns None
    And whose plan ends in a KickHalf
    And a configured built-in thermostat (coupling every step)
    When the runner runs the phase
    Then the run completes with Ok
    And the trailing KickHalf is dispatched to execute() every step (not skipped)
    And no composed kernel is launched (the integrator has no fragment and
      the thermostat never participates)

  @rq-79b8a246
  Scenario: A thermostat never participates in the composed set
    Given velocity-Verlet and any built-in thermostat with coupling_interval = 1
    When the runner is constructed
    Then the composed source contains no thermostat fragment
    And the thermostat exposes no post_force_per_particle fragment

  @rq-7bd422a5
  Scenario: A non-coupling step keeps kick fusion; a coupling step does not
    Given VelocityVerlet + CSVR active with coupling_interval = 4
    When the runner runs step 1 (non-coupling)
    Then the composed kernel is launched and covers the integrator
    And the trailing KickHalf is skipped from the plan walk
    When the runner runs step 4 (coupling)
    Then no composed kernel is launched
    And the trailing KickHalf is dispatched in the plan walk
    And CSVR's apply_post performs the reduction and rescale after it

  @rq-609dc377
  Scenario: Every built-in fusible slot kind exposes a post-force fragment
    Given the built-in integrator and barostat registries
    When each kind is built with default-valid parameters
    Then every built integrator returns Some from post_force_per_particle()
    And every built per-step barostat returns Some
    And every built thermostat returns None from post_force_per_particle()

  @rq-5e904c5d
  Scenario: A fragment-less per-step barostat with a thermostat has a valid execution order
    Given a built-in thermostat and a custom per-step barostat whose
      post_force_per_particle() returns None
    When the runner enters the phase
    Then phase setup succeeds (there is no PostForceTopologyUnsatisfiable rejection)
    And on a coupling step the barostat's apply runs its own standalone rescale

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
  Scenario: A non-coupling step launches the composed kernel exactly once
    Given a runner with VelocityVerlet + CSVR (coupling_interval = 4) + c-rescale active
    When the runner runs step 1 (non-coupling)
    Then timings records exactly one sample for KernelStage::JitComposedPostForce
    And timings records zero samples for KernelStage::VV_KICK
    And timings records zero samples for KernelStage::C_RESCALE_RESCALE_VELOCITIES
    And timings records zero samples for KernelStage::CSVR_RESCALE_VELOCITIES
      (the thermostat is inert on a non-coupling step)

  @rq-e12c2668
  Scenario: Slot bind methods are invoked once each in canonical order
    Given a runner with two active fusible slots [A=integrator, B=barostat],
      each with instrumented bind methods that record their invocation order,
      on a non-coupling step
    When the runner runs one timestep
    Then A's bind_post_force_per_particle_args is invoked before B's
    And each bind method is invoked exactly once per step

  @rq-86dea9a1
  Scenario: Thermostat's apply_post runs the reduction and the rescale
    Given a runner with CSVR thermostat active (coupling_interval = 1)
    And CSVR's apply_post is instrumented to record kernel-launch counts
    When the runner runs one timestep
    Then CSVR's apply_post launched compute_kinetic_energy and csvr_sample_and_factor
    And CSVR's apply_post also launched its per-particle rescale_velocities kernel
    And no composed post-force kernel was launched that step

  @rq-56044cc3
  Scenario: Barostat's apply runs scalar prep but not per-particle rescale
    Given a runner with c-rescale barostat active
    When the runner runs one timestep
    Then c-rescale's apply launched virial_sum_reduce and c_rescale_compute_mu
    And c-rescale's apply did NOT launch rescale_velocities_device_factor
    And c-rescale's apply did NOT launch rescale_positions_device_factor

  # --- Correctness ---

  @rq-d6d4f598
  Scenario: Composed-kernel output matches the separate launch sequence within f32 round-off
    Given the same physical state (VelocityVerlet + c-rescale, non-coupling step) run two ways:
      (a) the composed-kernel path
      (b) the separate per-slot kernel sequence (vv_kick → c_rescale_rescale)
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
  Scenario: Integrator kick runs before thermostat rescale (canonical order, eager coupling step)
    Given a runner with VelocityVerlet + CSVR (coupling_interval = 1)
    And the CSVR factor scalar is set artificially to 0.5
    When one timestep is run
    Then the post-step velocity equals 0.5 * (pre-step velocity + a · dt/2)
      within f32 round-off
    (The integrator's trailing kick updates v first, in the plan walk; then
     the thermostat's apply_post reduces the full-step KE and rescales it.)

  @rq-ae74d89b
  Scenario: Barostat rescale runs after integrator and thermostat (canonical order)
    Given a runner with VelocityVerlet + CSVR (coupling_interval = 1) + c-rescale
    And the c-rescale velocity scalar is set artificially to 1.1
    When one timestep is run (a coupling step: integrator kick → thermostat
      rescale → barostat rescale all run eagerly)
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
  Scenario: A thermostat's rescale runs as a standalone kernel
    Given the project's kernel source tree
    When the thermostat-shape standalone kernel symbols are enumerated
    Then the per-particle rescale kernel a thermostat uses (e.g. rescale_velocities)
      exists as an extern "C" kernel
    And kinetic_energy_reduce and csvr_sample_and_factor also exist
      (a thermostat launches its reduction, factor, and rescale itself)

  @rq-33fa8597
  Scenario: c-rescale rescale_velocities and rescale_positions standalone kernels do not exist
    Given the project's kernel source tree
    When the barostat-shape standalone kernel symbols are enumerated
    Then no extern "C" kernel named rescale_velocities_device_factor exists
    And no extern "C" kernel named rescale_positions_device_factor exists
    And virial_sum_reduce and c_rescale_compute_mu still exist
```
