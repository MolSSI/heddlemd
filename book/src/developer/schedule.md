# Schedules and CUDA Graphs

The [overview](index.md) sketched a timestep as an ordered list of
operations and noted that it runs on the GPU. This page fills in both ideas:
**what** a timestep is — the *schedule* — and **how** that schedule is
actually launched on the hardware — either a per-step launch loop or a
replayed *CUDA graph*. The two are separate concerns. The schedule defines
the physics; the launch mechanism only decides how efficiently that same
physics reaches the GPU.

## The schedule: a timestep as a list of operations

A textbook velocity-Verlet step is "half-kick, drift, recompute forces,
half-kick." A real step usually does more: it may rescale velocities for a
thermostat, rescale the simulation box for a barostat, and project
velocities and positions back onto rigid-bond constraints — and each of
those slots into the kick/drift sequence at a method-specific point.

Rather than write a bespoke, hand-ordered routine for every combination,
HeddleMD builds each timestep as a single ordered list of typed operations
called a **`StepPlan`**: a half-kick, a drift, a force evaluation, a
constraint projection, a barostat point, and so on. Every operation
declares the state it *reads* and the state it *writes* — its footprint.
One executor, `run_step`, walks the list in order, and before the run
begins the list is checked against those footprints for dependency
mistakes. An operation may not, for example, read a force that a preceding
position update has already invalidated without a force re-evaluation in
between; such a plan is rejected rather than silently producing wrong
dynamics.

This list *is* the definition of the step. An integrator does not run a
timestep by calling kernels itself; it hands back a `StepPlan`, and the
runner executes it. The payoff is that thermostats, barostats, and
constraints compose with any integrator without special-casing, and the
ordering rules live in exactly one place.

One principle protects that: any optimization that merges adjacent
operations into a single GPU launch (**fusion**) must preserve the
schedule's order and the state each operation observes. Fusion may make the
schedule cheaper to run; it may never become a second, competing definition
of the step.

## Two ways to run the same schedule

Once the schedule for a step is fixed, the runner has two ways to get its
kernels onto the GPU.

**The per-step launch loop.** The straightforward way: every timestep the
host walks the schedule and issues each kernel launch itself — on the order
of 15–20 individual launches per step (the pair-force kernel, the integrator
kicks and drift, the neighbour displacement check, and any
thermostat/barostat/constraint kernels). This is simple, and every kernel is
timed individually, but each launch carries a small fixed cost paid by the
CPU-side driver. For a modestly sized system the per-atom compute of a step
can be fast enough that these launch costs become a real fraction of the
wall clock — and an MD run issues them millions of times.

**Replayed CUDA graphs.** A **CUDA graph** is a feature of the CUDA driver
itself (no extra library) that lets you *record* a sequence of GPU
operations once and then replay the whole sequence with a single launch.
HeddleMD records one physical step's kernel sequence into a graph at the
start of a phase, then replays that graph once per step. The ~15–20 per-step
launches collapse into a single `cuGraphLaunch`, and the per-launch driver
overhead essentially disappears. Picture it as recording a macro of one
step's GPU work and replaying the macro, instead of re-issuing every command
by hand each step.

Graph mode is the default for any phase eligible for it; there is no
opt-in. Two knobs under `[simulation]` control it: `graph_batch_size` (how
many steps replay back-to-back between host check-ins, default 50) and
`cuda_graphs_disable` (a diagnostic escape hatch that forces the per-step
loop).

### Batched replay

The host still has periodic work that cannot live inside the graph:
checking whether atoms have drifted far enough to require rebuilding the
neighbour list, and writing trajectory frames and log rows. Graph mode
handles this in **batches**. The graph replays `graph_batch_size` steps in
a row with almost no host involvement; only at the batch boundary does the
host download the small neighbour-rebuild flag and write any output that is
due. Between boundaries the GPU runs nearly unsupervised — which is where
the speed-up comes from. Output cadences shrink a batch when needed so it
always ends exactly on a step that produces a log row or trajectory frame.

## When graph mode cannot be used

Recording a step into a graph works only if the *entire* step is device
work — GPU kernels one after another, with nothing the CPU must compute or
read back in the middle. Most of the engine is deliberately built this way.
A few methods are not: the Nosé–Hoover-chain thermostat integrates its
extended-system variables with arithmetic on the host and reads the kinetic
energy back to the CPU every step, and the MTK-NPT integrator does similar
host-side chain arithmetic inside its step. A step containing that kind of
host work cannot be captured, so a phase using one of those slots runs on
the per-step launch loop instead.

This fallback is automatic and silent — you do not configure it — and it
changes nothing about the physics or the results; it only forgoes the
launch-overhead saving for that phase. Each slot advertises whether it is
graph-compatible, and the runner captures a graph only when every slot in
the phase is.

## The graph is a recording, not a second timestep

The key conceptual point: a captured graph is not a different definition of
the timestep. It is a *recording of the same schedule's kernel sequence* —
the identical kernels, in the identical order, that the per-step loop would
have launched. This is the schedule-is-authoritative principle again: the
graph is an execution optimization over the one schedule, never a competing
execution model.

That has a precise consequence for reproducibility, worth stating carefully:

- **Run-to-run within one path is bit-exact.** Two runs of the same config
  on the same GPU — both in graph mode, or both with `cuda_graphs_disable =
  true` — produce byte-identical trajectories and logs. This is the
  load-bearing invariant, and it is what surfaces a scheduling-dependent
  bug: such a bug would break byte-identity the moment thread scheduling
  varied between runs.
- **The two paths agree only to floating-point rounding.** The per-step loop
  and the graph replay are two different *summation orders*, not two
  spellings of one order. They run the same kernels and match to f32
  rounding — and are bit-identical for regularly-packed systems — but on a
  disordered system they can differ by a single-precision quantum (order
  `1e-6`). That difference is a benign consequence of summation order, not a
  race; it does mean you should not treat one path as a bit-exact reference
  for the other.

Stochastic methods keep their random-number counter in a one-element device
buffer that both paths share, so a thermostat or barostat draws the same
random sequence whichever way the step is launched — the RNG is never a
source of divergence between the two paths.

The [Reproducibility](../guide/reproducibility.md) chapter states the
user-facing guarantee and its limits; the exhaustive treatment of schedule
orchestration and graph capture and replay lives in `docs/architecture.md`
in the repository.
