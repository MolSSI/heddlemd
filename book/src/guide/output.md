# Output Files

> **Stub.** Each run produces three files alongside the config:
>
> - **`*-traj.xyz`** — extended-XYZ trajectory frames, self-describing
>   (lattice vectors, column layout, simulation time). Re-loadable as an
>   init file.
> - **`*.log`** — CSV with `step,time,kinetic_energy,temperature`.
> - **`*.timings`** — fixed-width table with one row per instrumented stage
>   (per-kernel CUDA-event timings plus host stages). Columns: `count`,
>   `total_ms`, `mean_us`, `min_us`, `max_us`. Intentionally **not**
>   reproducible across runs.
>
> Worked examples of reading each format belong here.
