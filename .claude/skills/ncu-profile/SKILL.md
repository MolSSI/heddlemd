---
name: ncu-profile
description: Profile HeddleMD's CUDA kernels with NVIDIA Nsight Compute (ncu). Use when the user wants to profile a GPU kernel, investigate the pair-force (or any) kernel's Speed-of-Light / occupancy / register / warp-stall / memory-coalescing behaviour, or diagnose where a kernel's time goes. Profiling itself must run on the host (bare metal); this skill coordinates that with the user and then analyses the resulting report.
allowed-tools: Read, Bash, AskUserQuestion
---

Profiling with `ncu` cannot run inside this project's rootless podman
container: the NVIDIA driver gates GPU performance counters on
init-namespace `CAP_SYS_ADMIN` (`RmProfilingAdminOnly=1`), which a
rootless user namespace cannot provide even with `--cap-add=SYS_ADMIN`.
So this skill splits the work: **the user runs the profiler on the
host; you analyse the report inside the container** (reading a saved
`.ncu-rep` needs no counters and works anywhere).

`scripts/ncu_timings.sh` does the host-side collection. See
`pairforces2.md` for a worked example of the analysis and
interpretation, and `pairforces.md` for the optimization-option context.

## Step 1 — Ask the user to run the profiler on the host

Tell the user to run the script **on the host (bare metal)**, from the
repository's top-level directory, and then report back. Give them the
exact command for what they want to profile. Defaults profile the packed
pair-force kernel at the 8192-molecule system:

```sh
sudo -E ./scripts/ncu_timings.sh            # pair-force kernel, spc-water-8192
sudo -E ./scripts/ncu_timings.sh 65536      # larger system (more memory-bound)
sudo -E ./scripts/ncu_timings.sh 8192 "shake|rattle"        # other kernels
sudo -E ./scripts/ncu_timings.sh 8192 heddle_jit_composed_pair_force_f full  # full metric set
```

Arguments: `[size] [kernel-regex] [ncu-set]`. Valid sets are
`default` / `detailed` / `full` / `roofline` / `source` — **there is no
`basic` set** (ncu silently collects nothing if given an invalid set).
The script defaults to `detailed`, which collects SpeedOfLight,
Occupancy, LaunchStats, WarpStateStats, and MemoryWorkloadAnalysis. Use
`full` (adds SourceCounters) when you need per-source-line attribution.

**Critically, remind the user of the shared-library rebuild:** before
profiling on the host they will likely need to rebuild from scratch, or
the `target/release/heddlemd` binary (built inside the container) will
link against the container's CUDA shared libraries and fail or misbehave
on the host:

```sh
rm -rf target && cargo build --release      # on the host, before profiling
```

`sudo` is required on the host because real root carries the
`CAP_SYS_ADMIN` the driver demands. If `sudo` cannot find `ncu`, pass it:
`sudo NCU=/usr/local/cuda/bin/ncu ./scripts/ncu_timings.sh …`. (The
alternative to per-run `sudo` is a one-time host change:
`NVreg_RestrictProfilingToAdminUsers=0` + reboot.)

Then wait for the user to confirm the run finished.

## Step 2 — Analyse the report (inside the container)

The reports land in `profile_out/ncu/`, which is mounted at
`/work/profile_out/ncu/`. Find the newest report and import it — `ncu
--import` needs no counters, so this works in the container:

```sh
cd /work/profile_out/ncu && ls -t *.ncu-rep | head
```

Pull the headline metrics (per profiled kernel instance):

```sh
REP=<newest>.ncu-rep
ncu --import "$REP" --csv --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,\
gpu__time_duration.sum
```

Pull ncu's own rule findings (the most actionable signal — bottleneck,
uncoalesced access, non-fused FP, busiest pipe):

```sh
ncu --import "$REP" --page details \
  | grep -E "utilizing greater than|uncoalesced (global|shared)|non-fused FP32|highest-utilized pipeline|limited by the number of required registers" \
  | sort -u
```

Pull the warp-stall breakdown (needs `detailed`/`full`):

```sh
ncu --import "$REP" --csv --metrics \
  smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio
```

### Interpretation

- **Clock caveat:** ncu locks clocks to base by default, so its absolute
  `Duration` (~2× the engine's boost-clock `.timings`) is not comparable
  to wall-clock numbers. Use the **% SOL and ratios**, which are
  clock-independent.
- **Compute (SM) % vs Memory %:** whichever is higher is the ceiling.
  Both high (~85%+) ⇒ roofline-bound on both axes; relieving one just
  shifts the wall to the other.
- **DRAM % vs L2 % vs L1/TEX %:** high L1/TEX with low DRAM ⇒ *not*
  bandwidth-bound; uncoalesced accesses wasting L1 sectors (check the
  uncoalesced-access rule).
- **Achieved vs theoretical occupancy + Block Limit (Registers):** if
  achieved tracks theoretical and theoretical is register-capped, there
  is no occupancy headroom (so `__launch_bounds__`-style changes won't
  help).
- **Warp stalls:** `long_scoreboard` ⇒ global-memory latency;
  `short_scoreboard` ⇒ shared-memory latency; `mio_throttle` ⇒
  L1/shared/special-function-unit throughput; `lg_throttle` ⇒
  local/global throughput; `math_pipe_throttle` ⇒ math-pipe throughput.
- **Non-fused FP32 rule:** a large non-fused count flags missed FMA
  contraction — a compute-side codegen lever.

Report the findings as: which SOL axis is the ceiling, what ncu's rules
flag, and which optimization that points to (cross-reference
`pairforces.md` options where relevant).

## Step 3 — Remind the user to rebuild in the container afterwards

Once analysis is done and you (or the user) resume normal work **inside
the container**, the `target/` directory now holds the host-built binary
linked against the host's shared libraries, which is wrong for the
container. Remind the user — and do this yourself before running anything
that builds or executes the engine in the container:

```sh
rm -rf target && cargo build --release      # in the container, after profiling
```

## Notes

- `class_accumulator_memset` and other `cudaMemsetAsync` operations are
  **not** named kernels — `-k regex` will not match them ("No kernels
  were profiled"). Their cost is per-step launch overhead that graph
  mode collapses; it is not a real production cost.
- To pinpoint exact source lines (uncoalesced sectors, non-fused FP),
  use the `full` set and open the report's Source page in `ncu-ui` on the
  host: `ncu-ui profile_out/ncu/<report>.ncu-rep`.
