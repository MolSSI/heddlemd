# Registries and Pluggable Components

A HeddleMD run is assembled at startup from interchangeable pieces: one or more
**potentials** (Lennard-Jones, electrostatics, bonds, angles, …), an
**integrator**, and optionally a **thermostat**, a **barostat**, a
**constraint** method, a **minimizer**, and **analyses**. Which pieces are used
is decided entirely by the configuration file, and each kind of piece can be
extended without editing the engine's core.

This page explains the machinery that makes that possible — the **registry
framework** — and why it is built the way it is. It is a conceptual companion
to [Extending HeddleMD](../extending/index.md): that section is the
step-by-step recipe for adding a component, while this page explains the
mechanism those recipes plug into. If you have written a plugin system in
Python or an abstract factory in C++, the shape will be familiar.

## The problem it solves: no central switch

The naïve way to let a configuration select a method is one big conditional
somewhere in the engine:

```text
if kind == "lennard-jones": ...
else if kind == "buckingham": ...
else if kind == "coulomb": ...
```

Every new method means editing that conditional, and the list becomes a
bottleneck that every contributor touches and every merge conflicts over. It
also couples the engine's core to the full catalogue of methods, which is the
opposite of what an extensible research code wants.

HeddleMD avoids this. There is **no central list of `if`s**. Instead each
method is a small self-describing unit that announces what configuration it
handles, and the engine simply asks a collection of those units which one (or
ones) apply. Adding a method means adding a unit; it does not mean editing a
decision that lives somewhere else.

## Three roles: registry, builder, component

The framework has three kinds of object. The separation matters, so it is worth
naming each precisely.

| Role | What it is | Closest analogy |
| --- | --- | --- |
| **Registry** | A collection of *builders* for one category (all the thermostats, all the potentials, …) | A plugin registry, or a dictionary of factories |
| **Builder** | A small object — usually holding no data — that knows one method's configuration schema, validates it, and constructs the method | A factory class, or a `from_config` class-method |
| **Component** | The constructed, running object that does the physics on every timestep | An instance of a class implementing an interface |

A component is also called a **slot** once it is installed in a running
simulation — "the phase's thermostat slot," for example.

The reason a builder and a component are *separate* objects is that they have
different lifetimes and different jobs. The builder exists before the
simulation and its only responsibilities are "do you understand this
configuration, and if so, build the thing." The component exists during the
simulation and holds the parameters, the GPU buffers, and any state that
evolves as the run proceeds. In C++ terms the builder is an abstract factory
and the component is the product; in Python terms the builder is the class
object with a `from_config` constructor and the component is an instance.

Each component implements a **trait** — Rust's word for an interface or an
abstract base class. A `Thermostat` provides the operations every thermostat
must supply; a `Potential` provides those every potential must supply; and so
on. The rest of the engine only ever talks to a component through its trait, so
the timestep loop can hold "whichever thermostat the user chose" without
knowing or caring which concrete one it is — exactly the polymorphism a C++
virtual base class or a Python duck-typed object gives you.

## Two ways a configuration selects a component

There are two selection styles, because two different questions are being
asked.

### Named selection — "which one?"

An integrator, thermostat, barostat, constraint method, minimizer, or analysis
is a single choice: a phase has *one* integrator and at most one thermostat. So
the configuration names it directly:

```toml
[phase.thermostat]
kind = "csvr"
```

The thermostat registry is asked for the builder whose name is `"csvr"`, that
builder validates the rest of the section and constructs the thermostat, and
the result fills the phase's thermostat slot. If several builders share a name,
the first one registered wins — a detail that matters only when a library adds
its own, discussed below.

### Compositional activation — "which ones?"

Potentials are different, and the difference is the interesting part of the
framework. A force field is not *one* potential; it is a **sum** of
contributions — Lennard-Jones *and* electrostatics *and* bonds *and* angles,
all at once. There is no single `kind` to name.

So instead of naming a potential, the configuration lists the interactions it
contains, and **every potential builder is shown the whole configuration and
decides for itself whether it applies.** A builder returns a component when its
interaction is present and returns nothing when it is absent. The force field is
the sum of all the components that opted in.

```text
For each potential builder in the registry:
    look at the configuration
    if it mentions an interaction I own  ->  build my component
    otherwise                           ->  contribute nothing
```

This is the same pattern as a chain of plugins each inspecting an input and
volunteering if it is relevant, and it is why a potential is added without
touching any other potential: a new builder simply becomes one more voice that
is consulted, and it stays silent unless its interaction appears.

## How parameters reach the right builder

Configuration parameters arrive loosely typed — a `kind` string and a bag of
not-yet-interpreted values — and the framework routes each bag to the builder
that owns it.

- For **named** components, the `kind` string is the routing key: the matching
  builder deserializes the bag into a typed parameter struct, rejects unknown
  or misspelled fields, and checks physical ranges (a positive temperature, a
  finite timescale, a sensible cutoff).
- For **potentials**, routing uses a **parameter claim** — a
  `(category, kind)` pair such as `(PairInteraction, "buckingham")`. The claim
  tells the loader which builder validates and unit-converts each interaction
  entry, even though every builder is consulted at construction time.

Either way, values given in SI units are converted to the engine's internal
atomic units at this boundary, using the dimension types described in the
[unit-conversion](../extending/index.md#preserve-physical-units) discussion, so
the rest of the engine only ever sees atomic units.

## Potentials compose into a single kernel

For potentials the pluggability goes one level deeper than swapping objects, and
this is worth understanding because it is unusual.

The active pair potentials do **not** each launch their own GPU kernel. Each
one instead contributes a small fragment of CUDA source — just its per-pair
formula, "given a distance, return a force and an energy" — and the framework
stitches the active fragments together and compiles them into a **single fused
kernel** at startup. That one kernel walks each particle's neighbour list once
and adds up every active potential's contribution as it goes.

So when you add a pair potential you are not adding a kernel; you are adding a
few lines of formula that get folded into the shared one. This is what lets the
framework carry many potentials at no per-potential launch cost, and it is
designed so that a new fragment cannot disturb the others or break the engine's
reproducibility guarantee. The mechanism is described in full in [Adding a pair
potential](../extending/adding-a-pair-potential.md); the point here is that the
"plug" for a potential is a piece of source code, not just an object.

## Why it is built this way

Three payoffs justify the machinery.

- **Open to extension without central edits.** Adding a method is a *new* file
  plus one line appended to a roster; no existing decision logic changes. This
  is the classic "open for extension, closed for modification" goal, and it is
  what keeps a growing method catalogue from turning the engine's core into a
  merge battleground.

- **Safe by construction.** A component does not re-implement force
  accumulation, reductions, or neighbour search — it plugs into the shared,
  deterministic machinery the engine already provides. That is what lets a
  contributor add physics without accidentally breaking the fixed-GPU
  [reproducibility guarantee](../guide/reproducibility.md); the safe path is
  the default path. (The [op-model](op-model.md) applies the same philosophy to
  the timestep schedule.)

- **Embeddable as a library.** Because the built-in methods are just the
  default contents of the registries, a program that embeds HeddleMD can create
  the registries, register *its own* builders alongside the built-ins, and run.
  HeddleMD is therefore a library you can extend from the outside, not only a
  program with a fixed menu. (Named registries use the first builder that
  matches a name, and potential registries consult every builder, so an
  externally registered method composes with the built-ins rather than
  silently replacing one.)

## Where the pieces live, and how to add one

The seven registries are collected in one `Registries` value, and each ships a
built-in **roster** — the list of builders included with HeddleMD:

| Component | Built-in roster |
| --- | --- |
| Integrator, thermostat, barostat | `src/integrator/` |
| Constraint method | `src/integrator/constraint.rs` |
| Potential | `src/forces/mod.rs` |
| Minimizer | `src/minimizer/mod.rs` |
| Analysis | `src/analysis/mod.rs` |

Adding a built-in means implementing the builder and the component for your
method and appending the builder to the appropriate roster; adding one from an
embedding program means registering it on the `Registries` value before the run
starts. The [Extending HeddleMD](../extending/index.md) guides walk through each
kind step by step, and [Testing Extension Components](testing.md) explains the
coverage checks that make sure every built-in method carries the shared physical
tests its category requires.

This framework is, ultimately, the reason the rest of the developer guide can
speak of "a potential" or "a thermostat" in the abstract: the registries are
what turn a name or an interaction in a configuration file into a concrete,
running piece of physics, without any part of the engine needing to know the
whole catalogue in advance.
