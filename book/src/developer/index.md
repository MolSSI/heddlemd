# Developer Guide

This part of the book is for people who want to change the HeddleMD engine.
It assumes that you know the basic ideas of molecular dynamics, including
force fields, integrators, thermostats, and neighbor lists. It does not assume
that you have programmed in Rust or CUDA.

This page introduces the vocabulary used throughout the developer guide. You
do not need to learn Rust or GPU programming in full before you begin, but you
should understand the host/device split and be able to recognize the Rust
constructs described below. The extension guides explain the project-specific
interfaces in more detail:

- [Schedules and CUDA Graphs](schedule.md) explains how HeddleMD represents
  and runs a timestep.
- [Testing Extension Components](testing.md) explains the shared physical
  test harnesses and how they complement focused component tests.
- [Extending HeddleMD](../extending/index.md) explains how to add a thermostat,
  integrator, barostat, or potential.
- `docs/architecture.md` in the repository is the detailed design reference.
  It is useful when changing the framework itself, but it is not a prerequisite
  for following the extension guides.

## The simulation loop

At a high level, HeddleMD performs the same loop as other molecular-dynamics
programs:

1. Use the current positions to build a **neighbor list** of nearby atoms.
2. Evaluate the force field and sum the force on each atom.
3. Advance the positions and velocities by one timestep.
4. Apply any thermostat, barostat, or constraint operations required by the
   selected method.
5. Repeat.

The principal data flow is:

```text
positions -> neighbor lists -> force evaluation -> net forces -> integration
     ^                                                               |
     +---------------------- next timestep ---------------------------+
```

HeddleMD runs its numerical work on a single NVIDIA GPU. On a fixed GPU, two
runs of the same input are designed to produce byte-for-byte identical output.
That reproducibility requirement affects how reductions, random numbers, and
timestep operations are implemented.

## Phases: the stages of a run

A single HeddleMD run is divided into one or more **phases**, each a
self-contained stage that runs the simulation loop above for a fixed number of
steps. A phase is the unit the user configures: every `[[phase]]` (and every
`[[minimization]]`) block in the input file is one phase, and phases execute in
the order they appear. Particle state — positions, velocities, and the box —
carries over from one phase to the next, so the phases of a run form a
continuous trajectory.

Each phase fixes its own method for its whole duration: one integrator, at most
one thermostat, at most one barostat, its own timestep, and its own output
cadences. A typical run uses several phases to do different jobs with different
settings — for example, an energy **minimization** to relax the starting
geometry, then an **equilibration** phase under a thermostat, then a
**production** phase (perhaps adding a barostat for constant pressure) that
writes the trajectory you analyze. The [Configuration
Reference](../guide/configuration.md#phase) describes the `[[phase]]` schema in
full.

The developer guide refers to "a phase" constantly, and the reason it is the
natural unit is exactly that its method is fixed for its whole span. Anything
the engine only needs to prepare once — validating the integrator's schedule
(see [Validating Timestep Dependencies](op-model.md)), recording a CUDA graph
(see [Schedules and CUDA Graphs](schedule.md)) — is set up once at the start of
each phase and reused for every step of that phase, because the operations that
setup depends on cannot change until the next phase begins.

## Host code and device code

HeddleMD uses two languages with different responsibilities:

- **Rust host code** lives under `src/` and runs on the CPU. It reads the
  configuration, allocates GPU memory, arranges operations in the correct
  order, launches GPU work, handles errors, and writes output.
- **CUDA C device code** lives under `kernels/` and runs on the GPU. A CUDA
  function launched across many GPU threads is called a **kernel**.

The word **host** therefore means the CPU and its Rust code. The word
**device** means the GPU and its memory. A device buffer is memory allocated
on the GPU; the CPU cannot use its contents directly. Moving a value from the
device to the host requires an explicit copy and usually forces the CPU to wait
for earlier GPU work to finish.

At build time, CUDA sources are compiled to PTX, an intermediate GPU
instruction format, and embedded in the HeddleMD executable. At run time, the
Rust code loads the PTX and launches its kernels.

### A useful mental model for a kernel

A kernel is similar to the body of a parallel loop. HeddleMD launches many
instances of that body at once, usually assigning one particle or interaction
to each GPU thread. Threads are grouped into blocks, and groups of 32 threads
within a block execute together as a **warp**. An individual thread within a
warp is sometimes called a **lane**.

GPU operations are normally **asynchronous**: launching a kernel asks the GPU
to do work but does not necessarily wait for it to finish. A device-to-host
copy of a result often introduces a synchronization point. This distinction is
important for CUDA graphs and for performance.

### Device buffers and scratch space

The extension guides frequently refer to a **scratch buffer**. This is
temporary device memory used to hold an intermediate result, such as the total
kinetic energy or a velocity scale factor. Allocate scratch buffers when the
component is constructed and reuse them on every timestep. Allocating device
memory inside a per-step method is both expensive and incompatible with some
execution optimizations.

A “length-1 device buffer” is simply a GPU allocation containing one scalar.
It is used when one GPU kernel produces a value that a later kernel consumes.
Keeping the scalar on the device avoids copying it to the CPU between kernels.

## Reading the Rust examples

The examples in this guide use a small set of recurring Rust constructs. This
section is a notation guide, not a general Rust tutorial.

### Structs and methods

A **struct** groups named fields, much like a C struct or a simple Python data
class:

```rust
struct ThermostatState {
    target_temperature: f64,
    kinetic_energy: CudaSlice<Real>,
}
```

An `impl` block defines methods for a type. `Self` means the type named after
`impl`, and `self` is the current value:

```rust
impl ThermostatState {
    fn target(&self) -> f64 {
        self.target_temperature
    }
}
```

### References and mutable references

Rust passes borrowed access with references:

- `&value` borrows read-only access.
- `&mut value` borrows mutable access, allowing the called function to change
  the value.
- `&self` is a method that reads its object.
- `&mut self` is a method that may update its object.

For example, `compute_kinetic_energy_on_device(buffers, &mut scratch)` may
write into the existing `scratch` buffer. It does not allocate a new buffer or
transfer ownership of the old one.

### Traits and trait implementations

A **trait** defines behavior that a type must provide. It is similar to a C++
interface or an abstract base class. The `Thermostat` trait, for example,
defines the operations that every thermostat implementation may perform.

```rust
impl Thermostat for MyThermostat {
    fn apply_post(&mut self, /* arguments */) -> Result<(), ThermostatError> {
        // implementation
        Ok(())
    }
}
```

The first line means “implement the `Thermostat` interface for
`MyThermostat`.” Traits can provide default methods, so an implementation only
needs to override behavior that differs from the default.

A type such as `Box<dyn Thermostat>` is a heap-owned value used through the
`Thermostat` trait. This lets the simulation hold whichever concrete
thermostat the user selected without encoding every thermostat type in the
runner.

### Results, options, and the question-mark operator

Rust uses `Result<T, E>` for operations that can fail. `Ok(value)` represents
success and `Err(error)` represents failure. A function returning `Result`
can use `?` after another fallible call:

```rust
launch_kernel()?;
```

If the launch succeeds, execution continues. If it fails, the error is
returned to the caller. This replaces the unchecked error codes common in C.

`Option<T>` represents a value that may be absent. `Some(value)` contains a
value and `None` means that no value is present. Potential builders use this
distinction: `Ok(Some(slot))` activates a potential, while `Ok(None)` means
that the configuration did not request it.

### Derive attributes and configuration types

An attribute such as `#[derive(Debug, Clone, Deserialize)]` asks the compiler
to generate standard behavior for a struct. In this guide:

- `Debug` makes a value printable for diagnostics.
- `Clone` allows the registry framework to copy a builder.
- `Deserialize` lets Serde construct the value from TOML.
- `Serialize` and `Convert` support HeddleMD's unit-conversion path.

`#[serde(deny_unknown_fields)]` rejects misspelled or unsupported configuration
keys instead of silently ignoring them.

### Modules and public re-exports

Rust source files belong to modules. Adding a file is not enough by itself:
the parent module must declare it with `pub mod my_thermostat;`. A statement
such as `pub use my_thermostat::MyThermostatBuilder;` makes the named type
available through the parent module. The per-feature guides identify the
required declarations and registration entry.

## Data layout on the GPU

HeddleMD stores particle data as a **structure of arrays** (SoA). Instead of
an array in which every element contains one particle's position, velocity,
and mass, the engine keeps separate arrays such as `positions_x`,
`positions_y`, and `positions_z`.

This layout allows adjacent GPU threads to read adjacent memory locations.
Such reads are called **coalesced** and are substantially more efficient than
scattered reads. New device code should use the existing `ParticleBuffers`
layout rather than constructing an array of particle structs.

## Deterministic parallel calculations

GPU threads do not finish in a predictable order. This matters because
floating-point addition is not associative: `(a + b) + c` may differ in its
least significant bits from `a + (b + c)`.

HeddleMD therefore avoids floating-point sums whose order depends on thread
scheduling. Its principal strategies are:

- Sort neighbor lists so that interactions are visited in a fixed order.
- Use fixed-topology reductions when summing floating-point values.
- Use integer fixed-point accumulators where many threads must contribute to
  the same result. Integer addition is independent of arrival order.
- Avoid floating-point `atomicAdd` when multiple threads contribute to one
  value.
- Use seeded, counter-based random-number generation rather than wall-clock
  or thread-local state.

An **atomic operation** safely updates shared memory when threads run
concurrently. It prevents lost updates, but a floating-point atomic addition
does not fix the order of those updates. Thread safety alone is therefore not
enough for bitwise reproducibility.

## Timesteps, schedules, and CUDA graphs

An integrator does not directly run an entire timestep. It returns a
`StepPlan`, an ordered list of operations such as kicks, drifts, force
evaluations, and thermostat or barostat coupling points. The runner validates
this schedule and executes it.

This separation makes the schedule the authoritative definition of the
physics. It also permits two execution paths:

- The **per-step launch path** has the CPU launch each kernel separately.
- A **CUDA graph** records a compatible sequence of device operations and
  replays it with less CPU launch overhead.

A component is graph-compatible only when all of its per-step work can be
recorded as device operations. Reading a device scalar back to Rust, doing
host-side arithmetic with it, and then launching another kernel creates host
work in the middle of the sequence and normally makes that component
incompatible with graph capture. See [Schedules and CUDA
Graphs](schedule.md) for the complete execution model.

## Registries, builders, and slots

Most extensible HeddleMD components use three related concepts:

- A **registry** stores the available implementations.
- A **builder** validates configuration parameters and constructs one
  implementation.
- A **slot** is the constructed runtime component used by a simulation phase,
  such as its thermostat or barostat.

This design avoids a central conditional containing every supported method.
Adding a built-in generally means implementing the appropriate traits and
adding its builder to one registry roster. The [Extending
HeddleMD](../extending/index.md) overview explains this process and introduces
the configuration and testing conventions shared by all extension points.
