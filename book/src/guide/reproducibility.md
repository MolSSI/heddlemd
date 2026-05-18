# Reproducibility

> **Stub.** What the engine guarantees, and what it does not.
>
> **Guaranteed:** `*-traj.xyz` and `*.log` are byte-identical across two runs
> of the same config on the same GPU.
>
> **Not guaranteed:**
>
> - `*.timings` — wall-clock measurements vary run-to-run by design.
> - Cross-hardware bit-exactness — CUDA permits FMA contraction differences
>   between GPUs, and the engine ships with `nvcc`'s default FMA contraction
>   enabled for performance. A CPU reference and a GPU kernel will not in
>   general agree bit-for-bit either.
>
> This chapter should also explain `lossless` integrator mode (compensated
> `(f32, f64)` accumulation that enables bit-exact time reversal), and the
> important caveat that `lossless` is **not** a double-precision simulation —
> forces, `a = F/m`, and `dt` remain `f32` regardless.
