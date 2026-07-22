# Developer Guide

This part of the book is for people who want to work on the engine itself.
It explains the mechanics of how each timestep is computed, and how to add
a new thermostat or potential. It assumes you know molecular
dynamics as a field (force fields, integrators, thermostats, neighbour
lists) and are comfortable with programming in general, but it does **not**
assume you have written Rust or CUDA before. This overview orients you and
explains the handful of language and GPU ideas the rest of the guide leans
on; the later pages, and the exhaustive `docs/architecture.md` in the
repository, go deeper.

## What the engine does, in MD terms

At heart HeddleMD does what every MD code does, in a loop:

1. Given the current atomic positions, work out which atoms are close
   enough to interact, and store them in the **neighbor list**.
2. Evaluate the **force field** to get the net force on each atom.
3. **Integrate** Newton's equations one timestep to get new positions and
   velocities, optionally coupled to a thermostat or barostat.
4. Repeat.

It is designed to executes on a single NVIDIA GPU, and is built so
that running the same input twice on the same GPU produces a
**byte-for-byte identical** trajectory. Many of the design decisions
described below are in service to that guarantee.

The per-step data flow is:

```
positions ─▶ neighbour lists ─▶ pair-force kernel ─▶ net forces ─▶ integrator ─▶ new positions
              (who is near        (sum each atom's      (one force    (move atoms; apply
               whom, sorted)       interactions)         per atom)     thermostat / barostat)
```

## Two languages, two roles

HeddleMD is written in two languages, and it helps to know which does what:

- **Rust** (`src/`) is the *host* code. It reads the config, allocates GPU
  memory, decides what runs in what order, and writes output. It is the
  orchestrator; it does not itself crunch the per-atom arithmetic.
- **CUDA C** (`kernels/`) is the *device* code — the routines that actually
  run on the GPU, one instance per atom (or per interaction). These are
  called **kernels**.

At build time each CUDA kernel is compiled to **PTX** (a portable
GPU-assembly format) and embedded in the binary; at run time the Rust side
loads that PTX and launches the kernels. If you have written CUDA or used
OpenMP offload before, this is the familiar host/device split.

## A few Rust ideas you will meet

Rust is not currently a commonly used code for scientific programming,
so it is worthwhile to consider a few words and phrases that are common
in Rust programming.

- **Struct** — a bundle of named data fields, like a C++ `struct`. For
  instance `CsvrParams { temperature, tau, seed }` is just those three
  values held together.
- **Trait** — a named set of methods that a type promises to provide,
  *without* saying how. If you know C++, a trait is close to a pure-virtual
  abstract base class (an interface), or a `concept`. Rust composes
  behavior by having types *implement traits* rather than by class
  inheritance. For example, every thermostat comes with a
  `ThermostatBuilder`: the trait declares "you can build a thermostat from
  these config parameters," and each specific thermostat fills in the how.
- **Trait object** (`Box<dyn ThermostatBuilder>`) — a value handled through
  its trait rather than its concrete type, much like a
  `std::unique_ptr<ThermostatBuilder>` pointing at some concrete subclass.
  It lets the engine hold a list of "things that can build thermostats"
  without knowing in advance which specific ones.
- **The registry pattern** — instead of one large `switch` over every
  thermostat kind, the engine keeps a **registry**: a lookup table from a
  name (the config's `kind = "csvr"`) to a small factory — a *builder* —
  that knows how to construct that one thing. Adding a new thermostat means
  writing a new builder and dropping it into the registry; you never edit a
  central list of `if kind == …`. HeddleMD has seven such registries —
  integrators, thermostats, barostats, constraints, minimizers, analyses,
  and potentials.
- **Structure of arrays (SoA)** — rather than one array of `Particle`
  structs (each holding a position, velocity, and mass together), HeddleMD
  stores each quantity in its own array: `positions_x`, `positions_y`,
  `positions_z`, `velocities_x`, and so on. This is a performance choice,
  explained under the GPU ideas below.

## A few GPU ideas you will meet

The GPU is what makes the engine fast, and also what makes reproducibility
non-trivial. The essentials:

- **A kernel is a parallel loop body.** When the host launches a kernel, the
  GPU runs the same short function across thousands of **threads** at once,
  each thread typically responsible for one atom. Writing GPU code is
  largely a matter of expressing "the work for one atom" and letting the
  hardware run it for all of them simultaneously.
- **Threads come in warps.** Threads are grouped into blocks, and within a
  block they run in lockstep bundles of 32 called a **warp** (each of the 32
  is a **lane**). This matters here because the pair-force kernel assigns
  *one warp to one atom*: the 32 lanes divide up that atom's neighbours,
  then cooperate to total their partial forces.
- **Coalesced memory reads reward SoA.** When the 32 lanes of a warp read
  from consecutive memory addresses in the same instruction, the hardware
  services them as one wide transaction. Storing positions as
  `positions_x[0], positions_x[1], …` (SoA) puts consecutive atoms at
  consecutive addresses, so this coalescing happens; an array-of-structs
  layout would scatter the reads across memory. That is why the engine uses
  SoA everywhere.
- **Thread order is not deterministic — and that is the whole problem.**
  The GPU makes no promise about the order in which threads or warps finish.
  If many threads added their contributions into one atom's force with an
  atomic add, those additions would happen in a different order on every
  run. Floating-point addition is not associative — `(a + b) + c` can
  differ in the last bit from `a + (b + c)` — so the force would come out
  slightly different run to run. HeddleMD forbids that pattern, which is the
  subject of the next section.

## How reproducibility is achieved

The guarantee — identical bits across two runs on the same GPU — rests on
fixing the *order* of every floating-point sum so it never depends on thread
scheduling:

- **Deterministic neighbour lists.** Each atom's neighbours are collected
  and then **sorted by atom index**, so a warp always sweeps a given atom's
  neighbours in the same order on every run.
- **No atomic float accumulation.** No force is summed with `atomicAdd`.
  Instead each atom's force stays inside one warp: the 32 lanes accumulate
  their partial contributions in registers, then combine them through a
  fixed **butterfly reduction** — a five-step pairwise exchange of register
  values within the warp. The shape of that reduction depends only on how
  many neighbours the atom has, never on which thread happened to arrive
  first.

The user-facing [Reproducibility](../guide/reproducibility.md) chapter
states the guarantee and its limits — for instance, it holds within one GPU,
not across different GPU models — while the device-level mechanics live in
`docs/architecture.md`.

## The timestep as a schedule

A real timestep is more than "kick, drift, kick": it may also apply a
thermostat, rescale the box for a barostat, and project constraints, and
those pieces interleave with the integrator at method-specific points.
Rather than hard-code each integrator's sequence imperatively, HeddleMD
represents a timestep as **one ordered list of typed operations** — a
`StepPlan`. Each operation declares which state it reads and writes, a
single executor (`run_step`) walks the list, and the list is checked for
dependency mistakes before the run begins (you cannot, for example, read a
force that a position update has just invalidated without recomputing it).

One principle governs any optimization of that list: merging adjacent
operations into a single GPU launch (**fusion**) may never change the order
in which things happen or which state an operation observes. The schedule is
the authoritative description of the physics; fusion is only permitted to
make it cheaper, not to redefine it.

How that schedule is then launched on the GPU — step by step, or recorded
once into a replayable CUDA graph — and what that choice means for
reproducibility is the subject of [Schedules and CUDA Graphs](schedule.md).

## Adding new physics, and where to go next

Because every capability is entered into one of the registries, adding
new physics — a thermostat, integrator, barostat, pair potential, or bonded
potential — means writing a small builder and registering it, not editing
the core. The [Extending HeddleMD](../extending/index.md) sub-section is the
step-by-step guide: read its overview once for the registry, configuration,
GPU-kernel, and determinism conventions, then follow the page for the
specific registry item you are adding.

For the complete internal design — the full data flow, the pair-force
accumulation scheme, the schedule and its fusion rules, the Rust/CUDA
boundary, the unit system, and the precision policy — see
`docs/architecture.md` in the repository.
