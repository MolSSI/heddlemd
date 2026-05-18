# Configuration Reference

> **Stub.** Document each TOML field. The canonical schema lives at
> `rqm/io/config-schema.md` in the repository — this chapter should be the
> user-facing rendering of it (worked examples, units, defaults, validation
> rules).
>
> Top-level sections to cover:
>
> - `[simulation]` — `seed`, `n_steps`, `dt`, `temperature`
> - `[integrator]` — `lossless` (compensated `(f32, f64)` mode vs ordinary
>   `f32`)
> - `[particles]` — per-type masses
> - `[interactions]` — per-pair Lennard-Jones coefficients
> - `[output]` — paths and cadences for the trajectory, log, and timings files
