# Init Files (Extended XYZ)

> **Stub.** Cover the init-file format the runner expects:
>
> - Particle count.
> - Orthorhombic simulation box: `Lattice="lx 0 0 0 ly 0 0 0 lz"`.
> - Per-particle type names and positions; positions must lie inside the
>   primary cell `[-L/2, L/2)` per axis.
> - Optional velocities. When absent, velocities are sampled from a
>   Maxwell-Boltzmann distribution at the configured temperature using a
>   deterministic ChaCha8 RNG seeded by the config seed, with the
>   centre-of-mass drift removed.
>
> Trajectory frames written by the engine are themselves valid init files,
> so a run can be restarted from any frame.
>
> Canonical reference: `rqm/io/init-state-file.md`.
