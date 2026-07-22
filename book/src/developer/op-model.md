# Validating Timestep Dependencies

The [Schedules and CUDA Graphs](schedule.md) page introduced a timestep as a
single ordered list of operations — a `StepPlan` — and mentioned in passing
that, before a run begins, HeddleMD checks that list for "dependency
mistakes." This page explains that check in full: what it protects against,
how it works, and — through two worked examples — how it catches both a
*within-step* multiple-timestep (RESPA) bug and an *across-step* barostat bug
before the simulation ever runs.

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

## The check: two passes over the plan

The heart of the check is a single **walk** over the operations from first to
last, keeping track of which computed force state is currently valid. Think of
it as reading a recipe top to bottom and, at each step, asking "does this step
use an ingredient that is still fresh?"

The walk starts from some set of forces assumed valid, and then, for each
operation in order:

- **Check its reads.** Every piece of state the operation reads must currently
  be valid. If it reads a force that has gone stale, the walk stops and reports
  the offending operation.
- **Mark forces stale if the atoms moved.** If the operation writes the
  positions or the box, every kind of computed force is now stale — mark them
  all invalid.
- **Record what it produces.** If the operation is a force evaluation, mark the
  forces it computed as valid again.

The reads are checked *before* the same operation marks anything stale, so a
fused kick-and-drift — which reads the forces and *then* moves the atoms —
correctly reads the still-valid forces first and only afterward invalidates
them.

A timestep is not run once: the same plan is replayed on every step. So the
walk is run **twice**, from two different starting points, to catch two
different kinds of mistake.

- **The within-step pass** starts by assuming *all* forces are valid — which is
  true on the very first step, whose forces come from a warm-up evaluation
  before the loop begins. This pass catches a force consumer placed after a
  move *inside the same step* with no force evaluation between them. Its failure
  is a *stale resource* error.
- **The across-step pass** recognizes that from the second step onward a step
  does *not* start with all forces valid — it starts with only the forces the
  *previous* run of the plan actually left valid at its end. So this pass starts
  the walk from "the forces this plan leaves valid at its end" and checks it
  again. It catches a plan whose *opening* operations read forces that the
  plan's own *closing* operations invalidated. Its failure is a *stale cached
  forces* error.

A plan must pass both. The two worked examples below show one bug each: the
first fails the within-step pass, the second fails the across-step pass.

## Worked example 1: a within-step RESPA bug

Here is the within-step pass earning its keep on a real class of mistake.

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
the force class it consumes. The within-step pass enforces exactly that, and
the separate tracking of fast and slow forces is what lets it distinguish a
correct multiple-timestep schedule from one that kicks on a stale accumulator.

## Worked example 2: an across-step barostat bug

The within-step pass sees each step as a self-contained recipe. But a step is
run over and over, and a mistake can hide in the *seam* between one step and the
next. That is what the across-step pass is for.

### The plan

A constant-pressure (NPT) step ends by letting a **barostat** rescale the
simulation box — and every atom's position with it — to nudge the system
toward its target pressure. A minimal velocity-Verlet NPT step looks like:

```text
1.  kick-and-drift from total force   (first half-kick, then move the atoms)
2.  evaluate forces
3.  half-kick from total force        (second half-kick)
4.  barostat point                    (rescale the box and every position)
```

Read on its own, this step is fine. Operation 1 reads the leftover forces from
the previous step, operation 2 refreshes them for the new positions, operation
3 uses those fresh forces, and operation 4 rescales at the very end. The
within-step pass is satisfied — nothing after operation 4 reads a force.

(The barostat point only rescales when a barostat is actually configured. On a
constant-volume NVE or NVT run the same operation is present but does nothing —
it is a placeholder — so it moves no atoms and none of what follows applies.)

### The bug across the boundary

But the step repeats. When it runs again, operation 1 reads forces — and those
forces are the ones operation 2 computed *last* time, **before** operation 4
rescaled the box and shifted every atom. They describe the configuration the
system was in before the last rescale, not the one it is in now. The leading
half-kick pushes the atoms with slightly wrong forces, and it does so on every
step.

The across-step pass catches this. It first asks: *what forces does this plan
leave valid at its end?* Walking the plan, operation 2 makes the forces valid,
but operation 4 moves the atoms afterward — so by the end of the plan, **no**
forces are valid. It then re-walks the plan starting from that leftover set (no
valid forces) and immediately hits operation 1 reading the total force, which
is not valid. The check stops and reports, in spirit:

> The operation at index 0 (`KickDrift`) reads stale cached forces (the total
> force): the schedule's own terminal operations invalidated it, with no
> trailing force evaluation.

The two ways to fix it are named in the message itself.

### Two ways to make it valid

1. **Add a force evaluation after the barostat.** Append "evaluate forces" as a
   fifth operation. Now the plan leaves fresh forces at its end, the leading
   half-kick of the next step reads valid forces, and the across-step pass is
   satisfied. This is exactly correct — but it pays for a *second* full force
   evaluation every step, a steep price for what is usually a tiny correction.

2. **Declare the barostat weak-coupling tolerant.** The built-in pressure-
   coupling barostats (`berendsen`, `c-rescale`) rescale by a very small factor
   each step, so letting the next step's leading half-kick consume the
   slightly-stale forces is a bounded, standard approximation — the same one
   long-established MD codes make. Such a barostat reports that it *tolerates
   stale cached forces*, and the across-step pass then accepts the leftover
   forces and lets the plan through. The exception is now written down in the
   barostat's own code, rather than left as an unstated assumption that a reader
   has to know about.

An integrator that instead makes a *large* configuration change at the end of a
step and does **not** declare tolerance is exactly the case the across-step pass
exists to reject. And a barostat handled the measure-preserving way — where the
box update happens in the *middle* of the step, before the force evaluation,
rather than after it (as the `mtk-npt` integrator does) — leaves its forces
valid at the plan's end and needs no tolerance at all.

## When the check runs, and why it is cheap

The check runs **once per simulation phase, at setup** — after the integrator
produces its plan and before the timestep loop starts. (A *phase* is one stage
of a run, with one fixed choice of integrator, thermostat, and barostat; see
[Phases: the stages of a run](index.md#phases-the-stages-of-a-run).) If the plan fails, the
phase is aborted with an error carrying the offending operation and resource,
and the loop never runs. There is no per-step cost.

Along with the plan, the check is handed a small amount of phase context: **is
a per-step barostat configured**, and if so **does it tolerate stale cached
forces**. The first tells the across-step pass whether the barostat point
actually rescales (so an NVE/NVT plan is not wrongly flagged); the second tells
it whether to accept the leftover forces a weak-coupling rescale leaves behind.

It is inexpensive and safe to run at startup because it reads **only the plan
and that context** — the list of operations, their declared footprints, and two
true/false facts about the barostat. It launches no GPU work and touches no
positions, velocities, or forces. For the same reason it is easy to test
directly — the test suite feeds it hand-written plans, correct and incorrect,
and checks that it accepts or rejects each one, all without a GPU.

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

- **Weak-coupling tolerance covers only forces.** When a barostat declares that
  it tolerates stale cached forces, that applies to the *forces* alone. Keeping
  the neighbour list valid across a box change is a separate concern, handled by
  the rebuild trigger described in the neighbour-list documentation, not by this
  check.

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
