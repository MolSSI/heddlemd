# Validating Timestep Dependencies

The [Schedules and CUDA Graphs](schedule.md) page introduced a timestep as a
single ordered list of operations — a `StepPlan` — and mentioned in passing
that, before a run begins, HeddleMD checks that list for "dependency
mistakes." This page explains that check in full: what it protects against,
how it works, and — through a complete worked example — how it catches a
realistic multiple-timestep (RESPA) bug before the simulation ever runs.

The check is worth understanding for anyone adding or modifying an
integrator, because an integrator's whole job is to produce a correct
`StepPlan`. The check is your safety net: it turns an ordering mistake that
would otherwise produce silently wrong dynamics into a clear error at
startup, naming the exact operation at fault.

## Two kinds of state: held and computed

A running simulation keeps several pieces of per-atom and global state. For
the purpose of this check they fall into two groups.

**Held state** is stored directly and is always available to read:

- the atom **positions**,
- the atom **velocities**,
- the periodic **image** flags (which periodic copy of the box each atom is
  in),
- the simulation **box** (the lattice).

**Computed state** is not stored independently — it is *calculated from* the
positions and the box by a force evaluation:

- the **forces** on the atoms,
- and, when the force field is split for multiple-timestep integration, the
  separate **fast-force** and **slow-force** accumulators (in the code these
  are the `Fast` and `Slow` force classes; more on this below).

The distinction is the whole point. The moment you move the atoms — or change
the box — the stored forces no longer describe the arrangement the atoms are
now in. They have gone **stale**. They stay stale until a force evaluation
recomputes them at the new positions. Reading stale forces is the single
mistake this check exists to catch: a kick that pushes the atoms using forces
that belong to where they *used* to be is wrong, and wrong in a way that a
short test run may not obviously reveal.

## Every operation declares what it touches

Each operation in a `StepPlan` declares two things: the state it **reads** and
the state it **writes**. Together these are the operation's **footprint**. The
footprints are fixed by the kind of operation, so the engine knows them
without running anything:

| Operation | Reads | Writes |
| --- | --- | --- |
| Half-kick (from total force) | velocities, forces | velocities |
| Half-kick (from fast or slow force) | velocities, that class's forces | velocities |
| Drift | velocities | positions, images |
| Kick-and-drift (fused) | velocities, forces | velocities, positions, images |
| Force evaluation (all forces) | positions, box | forces, fast forces, slow forces |
| Force evaluation (one class) | positions, box | forces, that class's forces |
| Thermostat half-step | velocities | velocities |
| Constraint projection | positions, velocities, box | velocities (and positions, depending on placement) |
| Barostat point | velocities, box | positions, velocities, box |

Two rows carry the key idea. A **force evaluation** is the only operation that
*produces* forces: it reads the current positions and box and writes fresh
forces. A **half-kick** *consumes* forces: it reads them to update the
velocities. Every dependency the check reasons about is some arrangement of
producers and consumers of that computed force state.

One subtlety that matters for the worked example: a force evaluation for a
single class — say, only the fast forces — refreshes *only that class*. It
does not revive the slow forces. This is exactly what multiple-timestep
integration needs, and it is exactly where the bug below hides.

## The check: walking the plan in order

The check walks the operations from first to last, keeping track of which
computed force state is currently valid. Think of it as reading the recipe top
to bottom and, at each step, asking "does this step use an ingredient that is
still fresh?"

It works like this:

1. **Start of the step.** The walk begins assuming the forces are valid —
   all of them, including the fast and slow accumulators. This reflects
   reality: a step inherits fresh forces from the previous step's final force
   evaluation (or, for the very first step, from the warm-up evaluation before
   the loop). So an integrator is entitled to read forces at the very top of a
   step without recomputing them.

2. **For each operation, in order:**
   - **Check its reads.** Every piece of state the operation reads must
     currently be valid. If it reads a force that has gone stale, the check
     fails here and reports the offending operation.
   - **Mark forces stale if the atoms moved.** If the operation writes the
     positions or the box, every kind of computed force is now stale — mark
     them all invalid.
   - **Record what it produces.** If the operation is a force evaluation,
     mark the forces it computed as valid again.

The reads are checked *before* the same operation marks anything stale, so a
fused kick-and-drift — which reads the forces and *then* moves the atoms —
correctly reads the still-valid forces first and only afterward invalidates
them.

Each step is checked on its own, starting from the "forces are valid" state.
The check does not try to carry staleness across the boundary between one
timestep and the next; a barostat that changes the box at the very end of a
step is fine, because the next step begins by recomputing forces as usual.
What the check is looking for is a stale read *within* a single step — a force
consumer placed after the atoms moved, with no force evaluation in between.

## A worked example: a RESPA multiple-timestep bug

Here is the check earning its keep on a real class of mistake.

### Background: splitting the forces

Multiple-timestep integration (r-RESPA) speeds up a simulation by treating
different parts of the force differently. The **fast** part — bonded terms and
short-range interactions — changes quickly and is cheap, so it is evaluated
several times per step on a small inner timestep. The **slow** part — chiefly
long-range electrostatics — changes gradually and is expensive, so it is
evaluated only *once* per outer step. That is the entire economy of RESPA:
pay for the expensive forces rarely, and keep accuracy by refreshing the cheap
forces often.

A correct RESPA outer step, with two inner substeps, is this list of
operations:

```text
1.  half-kick from slow force
2.  kick-and-drift from fast force     (inner substep 1)
3.  evaluate fast force
4.  half-kick from fast force
5.  kick-and-drift from fast force     (inner substep 2)
6.  evaluate fast force
7.  half-kick from fast force
8.  evaluate slow force
9.  half-kick from slow force
```

The slow force brackets the step (operations 1 and 9), and the inner loop
(operations 2–7) advances the atoms while refreshing only the fast force.
Operation 8 — re-evaluating the slow force once, at the end, before the final
slow kick — is easy to overlook. It is a single expensive line in the middle
of cheap inner-loop bookkeeping, and the developer already computed a slow
force at the top of the step. Forgetting it is a natural mistake.

### The bug

Suppose an integrator author omits operation 8. The plan they hand back is:

```text
1.  half-kick from slow force
2.  kick-and-drift from fast force
3.  evaluate fast force
4.  half-kick from fast force
5.  kick-and-drift from fast force
6.  evaluate fast force
7.  half-kick from fast force
8.  half-kick from slow force          <-- was preceded by a slow-force evaluation
```

Physically, the final slow half-kick now pushes the atoms with a slow force
that was computed back at the *start* of the step — before the two inner
drifts moved every atom. The dynamics are wrong. A brief run might look
plausible; the energy drift it introduces can take many steps to become
obvious, and by then it is hard to trace back to a missing force evaluation.

### Walking the buggy plan

The check tracks which forces are valid as it reads the plan top to bottom.
The last column shows the state *after* the operation runs.

| # | Operation | Reads | Fresh? | Valid forces afterward |
| --- | --- | --- | --- | --- |
| — | *(start of step)* | — | — | total, fast, slow |
| 1 | half-kick from slow force | velocities, **slow force** | yes | total, fast, slow |
| 2 | kick-and-drift from fast force | velocities, **fast force** | yes | *(moves atoms → all forces stale)* — none |
| 3 | evaluate fast force | positions, box | yes | fast |
| 4 | half-kick from fast force | velocities, **fast force** | yes | fast |
| 5 | kick-and-drift from fast force | velocities, **fast force** | yes | *(moves atoms → all forces stale)* — none |
| 6 | evaluate fast force | positions, box | yes | fast |
| 7 | half-kick from fast force | velocities, **fast force** | yes | fast |
| 8 | half-kick from slow force | velocities, **slow force** | **NO** | — |

At operation 2 the kick-and-drift moves the atoms, so *every* force — total,
fast, and slow — is marked stale. Operations 3 and 6 bring the **fast** force
back, exactly as RESPA intends, but nothing ever brings the **slow** force
back. When operation 8 tries to read the slow force, it is no longer valid.
The check stops and reports:

> The operation at index 7 (`KickHalf`) reads a stale resource: the slow-force
> accumulator.

(The operations are numbered from zero internally, so the eighth operation is
reported as index 7.) The message names the operation kind and the exact piece
of state that was stale, which points straight at the cause: a slow-force
consumer with no slow-force evaluation between the drifts and itself.

### The fix

Restore the missing slow-force evaluation before the final slow kick:

```text
7.  half-kick from fast force
8.  evaluate slow force                <-- restored
9.  half-kick from slow force
```

Re-walking the plan, operation 8 reads the positions and box (always
available) and writes a fresh slow force, so the slow force is valid again.
Operation 9 then reads a valid slow force, and the walk reaches the end with
no error. The plan is accepted and the simulation runs.

The lesson generalizes beyond RESPA: any time an integrator moves the atoms
and later consumes a force, there must be a force evaluation in between for
the force class it consumes. The check enforces exactly that, and the separate
tracking of fast and slow forces is what lets it distinguish a correct
multiple-timestep schedule from one that kicks on a stale accumulator.

## When the check runs, and why it is cheap

The check runs **once per simulation phase, at setup** — after the integrator
produces its plan and before the timestep loop starts. If the plan fails, the
phase is aborted with an error carrying the offending operation and resource,
and the loop never runs. There is no per-step cost.

It is inexpensive and safe to run at startup because it reads **only the plan
itself**: the list of operations and their declared footprints. It launches no
GPU work and touches no positions, velocities, or forces. For the same reason
it is easy to test directly — the test suite feeds it hand-written plans,
correct and incorrect, and checks that it accepts or rejects each one, all
without a GPU.

## What the check does and does not cover

A few boundaries are worth knowing when you extend the engine.

- **Integrator-private operations must declare their own footprint.** Most
  operations have footprints fixed by their kind, but an integrator may emit a
  custom operation for a step that does not fit the standard shapes. When it
  does, it states that operation's reads and writes explicitly, and the check
  holds it to the same rule as any built-in: a custom operation that reads a
  force after a drift, with no evaluation in between, is rejected. Declaring a
  custom operation's footprint honestly is the author's responsibility.

- **Kinetic energy, virial, and potential energy are not tracked here.** The
  check reasons about positions, velocities, box, and forces. The scalar
  quantities that a thermostat or barostat reduces — the kinetic energy, the
  virial — are computed inside those operations and are not part of this
  dependency check. The ordering rule that a constraint must publish its
  virial before a barostat reads it, for instance, is enforced elsewhere in
  the runner, not by this walk.

- **This is a correctness check, not an optimization.** The check never
  reorders, merges, or removes operations. Every operation still runs as its
  own GPU launch, in the order the plan lists them. The check only confirms,
  before the run, that this order is free of stale-force mistakes.

## Why this supports reproducibility

HeddleMD's central guarantee is that two runs of the same input on the same
GPU produce byte-for-byte identical output. That guarantee is only meaningful
if the dynamics being reproduced are the *correct* dynamics. A stale-force bug
would reproduce perfectly — it would give the same wrong answer every time —
and byte-identity alone would never flag it. By rejecting an ill-ordered plan
at startup, this check closes that gap: it ensures the schedule the engine
faithfully and reproducibly executes is one whose force dependencies are
sound. Together with the reproducible reductions described in the
[overview](index.md), it is part of what lets an integrator author compose
kicks, drifts, force evaluations, thermostats, and barostats and trust that a
mistake in their ordering becomes a clear error rather than a subtle,
repeatable physical wrongness.
