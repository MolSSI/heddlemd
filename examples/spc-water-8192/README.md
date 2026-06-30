# 8192-molecule rigid SPC/E water, NPT, with RDF post-processing

A larger, more realistic example that exercises rigid water (SHAKE),
SPME electrostatics, multi-phase scheduling, energy minimization,
NVT-then-NPT dynamics (CSVR thermostat + c-rescale barostat), and
the `heddlemd analyze` post-processing pipeline. 8192 SPC/E water
molecules (24,576 atoms) relaxed by a steepest-descent minimization,
then integrated for 5 ps of NVT equilibration followed by 2 ps of
NPT production at 298.15 K, 1 atm, then analyzed for the O-O, O-H,
and H-H radial distribution functions of the production trajectory.

## Layout

- `water.in.toml` — simulation config (rigid SPC/E + SHAKE + SPME,
  three phases: `min`, `equil`, and `prod`).
- `water.in.xyz` — initial state: 16 × 16 × 32 simple-cubic lattice
  of water molecules at sub-liquid density (~0.47 g/cm³), each with a
  randomized SO(3) orientation seeded for determinism.
- `water.in.topology` — 8192 SHAKE constraint groups, one per
  molecule.
- `water.in.analysis` — declares three RDF analyses (O-O, O-H, H-H,
  each 200 bins over `[0, 7 Å]`) against the `prod` phase's
  trajectory.
- `generate_init.py` — regenerates `water.in.xyz` deterministically
  (Shoemake's uniform-on-SO(3) rotation seed = 42).
- `generate_topology.py` — regenerates `water.in.topology`
  deterministically.

## Run

From this directory:

```
cargo run --release -- run water.in.toml
```

The combined run is roughly:

| Phase   | Steps / iters | Wall-clock cost          |
|---------|---------------|--------------------------|
| `min`   | up to 500     | ~3 s — converges in ~80  |
| `equil` | 2,500         | ~2 min — 5 ps at dt=2 fs |
| `prod`  | 1,000         | ~1 min — 2 ps NPT        |

Total runtime is a few minutes on a recent NVIDIA GPU; the dominant
cost during NPT is the cell-list rebuild as the box volume changes
under the barostat.

The run produces several output files in this directory — one
`.timings` per phase, plus a `.log` for the MD phases and a `.minlog`
for the minimization phase. No phase writes a trajectory in the
default config (the example focuses on the `prod` log + analysis;
turn on `[phase.output].trajectory_every` to write frames):

- `water.out.min.minlog` — per-iteration `iter, energy, max_force,
  step, accepted` rows from the SD minimization phase. ~10 rows
  given the configured `minlog_every = 10`.
- `water.out.min.timings` — minimization performance summary.
- `water.out.equil.log` — `step, time, kinetic_energy, temperature,
  csvr_conserved` (NVT diagnostics), one row per 250 steps.
- `water.out.equil.timings` — equilibration performance summary.
- `water.out.prod.log` — `step, time, kinetic_energy, temperature,
  csvr_conserved, pressure, box_volume, c_rescale_conserved` (NPT
  diagnostics), one row per 50 steps.
- `water.out.prod.timings` — production performance summary.

## Analyze

After `heddlemd run` has produced `water.out.prod.xyz`, post-process
with:

```
cargo run --release -- analyze water.in.analysis
```

This writes three CSV files next to the analysis input:

- `water.out.o-o.csv` — `r, g_r, count` for the O-O RDF.
- `water.out.o-h.csv` — O-H RDF.
- `water.out.h-h.csv` — H-H RDF.

The analyze step is CPU-only and takes a few minutes for this
trajectory (it walks every selected frame, enumerating every pair
within `r_max = 7 Å`).

The analysis file also lints cleanly:

```
cargo run --release -- lint water.in.analysis
```

reports `[heddlemd lint] OK` (after the trajectory exists).

## Minimization phase

The lattice-on-a-grid initial state has occasional H–H close contacts
(two waters whose hydrogens happen to point at each other can sit
~1 Å apart, well inside the repulsive Coulomb core). Starting NVT
directly from this state requires a tiny timestep — even a 2 fs
leapfrog kick generates supersonic velocities for the closest
contacts.

The `min` phase relaxes positions along the negative energy gradient
until either the per-atom force or the relative energy change drops
below a loose tolerance. SHAKE projects every trial position back
onto the rigid SPC/E manifold before the trial energy is evaluated,
so the minimization respects the constraint geometry exactly.
Velocities (Maxwell-Boltzmann sampled at setup, see
`[simulation].temperature`) and the simulation box pass through this
phase unchanged; only positions move.

Tolerances are chosen to reach a *non-pathological* starting
geometry, not a true energy minimum:

- `force_tolerance = 1.0e-9 N` — a few × thermal force per atom; the
  long tail of the gradient is left for the thermostat to handle.
- `energy_tolerance = 1.0e-5` (relative) — stops once consecutive
  accepted iterations move the total potential energy by less than
  `~10⁻²⁰ J`.
- `max_iterations = 500` — non-convergence is a hard error (exit code
  `2`). Lattice → relaxed typically completes in `~80` iterations.

The `.minlog` records every iteration's `(energy, F_max, step,
accepted)` so the descent is easy to inspect post hoc.

## Parameters

- SPC/E geometry: `r_OH = 1.0 Å`, `θ_HOH = 109.47°`, enforced rigidly
  by SHAKE (three pair constraints per molecule: O–H₁, O–H₂, H₁–H₂).
- SPC/E charges: `q_O = -0.8476 e`, `q_H = +0.4238 e`.
- SPC/E Lennard-Jones (O–O only): `σ = 3.166 Å`, `ε = 78.2 k_B`
  (`1.080 × 10⁻²¹ J`). The H–H and O–H pair entries carry a negligible
  `ε = 1 × 10⁻³⁰ J` only to satisfy the loader's "every unordered pair
  declared" rule.
- Electrostatics: SPME with `α = 3.5 nm⁻¹`, real-space cutoff
  `r_cut = 1 nm` (matching the LJ cutoff so one neighbour list serves
  both), FFT grid `(48, 48, 96)` (~1 Å spacing, small-prime-factored
  for cuFFT), spline order 4.
- Integrator: `velocity-verlet`, `dt = 2 fs` (allowed because SHAKE
  removes the high-frequency OH stretching mode).
- Thermostat: CSVR (Bussi-Donadio-Parrinello), `T = 298.15 K`,
  `τ = 100 fs`.
- Barostat: c-rescale (Bernetti-Bussi), `P = 1.013 × 10⁵ Pa`,
  `τ = 1 ps`, `β = 4.5 × 10⁻¹⁰ Pa⁻¹`.
- Cell list: `r_skin = 3 Å` (= 0.3 · r_cut). The packed-neighbour
  pair-force pipeline (see `rqm/forces/packed-neighbour-pair-force.md`)
  sizes its entry list to the actual interaction count, so no
  user-supplied per-atom cap is required.

## What you should see in the RDFs

Peak positions are dead-on for SPC/E water — peak heights are
inflated by ~30 % (see *Caveats* below).

| RDF       | Bundle peak                                        | SPC/E textbook position |
|-----------|----------------------------------------------------|-------------------------|
| O-O       | first peak at **2.78 Å**, g ≈ 3.5                  | 2.75 Å, g ≈ 2.9         |
| O-O       | second peak / first minimum at **3.5 Å**           | 3.5 Å                   |
| O-O       | second neighbour shell **~4.5–5 Å**, g ≈ 1.34      | 4.5 Å                   |
| O-H       | intramolecular bond at **0.998 Å** (SHAKE pin)     | 1.000 Å                 |
| O-H       | **hydrogen-bond peak at 1.77 Å**, g ≈ 1.51         | 1.85 Å                  |
| O-H       | second peak at **3.17 Å**, g ≈ 1.89                | 3.30 Å                  |
| H-H       | intramolecular pair at **1.628 Å** (SHAKE pin)     | 1.633 Å                 |
| H-H       | first intermolecular at **2.50 Å**, g ≈ 1.67       | 2.40 Å                  |

The intramolecular SHAKE-constrained peaks (O-H at 0.998 Å, H-H at
1.628 Å) appear as nearly-delta-function spikes, confirming the
constraint algorithm is enforcing the rigid SPC/E geometry exactly.
The hydrogen-bond signature in the O-H RDF at 1.77 Å is the
characteristic structural marker of liquid water and is well
reproduced here.

## Caveats

This bundle's primary value is demonstrating the engine end-to-end at
realistic system size: rigid water + SHAKE + SPME + NPT + RDF
post-processing all run cleanly, and the resulting RDFs show the
correct liquid-water peak structure. It is **not** a fully converged
production-quality calculation — two limitations are worth knowing
about before treating the `g(r)` heights as quantitative:

- **Equilibration from a lattice initial state is hard.** Even after
  the SD minimization knocks down the worst close contacts, a
  rigid-body lattice of charges carries a large residual virial
  pressure and the c-rescale barostat in the production phase
  responds by expanding the box. The 5 ps NVT + 2 ps NPT schedule
  in this bundle does not converge to the SPC/E equilibrium liquid
  density — the system instead relaxes into a metastable two-phase
  configuration (liquid cluster + low-density surroundings). The
  asymptotic value of every `g(r)` in this bundle plateaus around
  1.30–1.35 rather than the canonical 1.0; the offset is the ratio
  of the actual in-cluster density to the box-averaged density. For
  a strictly converged liquid-water density users should either:
  1. Start from a pre-equilibrated structure provided by another
     code (the init-file parser accepts any extended-XYZ file
     satisfying the engine's lower-triangular `Lattice` constraint),
     or
  2. Lengthen the equilibration phase by an order of magnitude.
- **RDF analysis assumes a constant simulation box.** The framework
  documented in `rqm/analysis/framework.md` caches the first frame's
  box volume for the ideal-gas normalisation; under NPT the box
  changes per frame, so the reported `g(r)` values are normalised
  against the initial volume rather than the per-frame average.
  Peak *positions* in the resulting CSVs are still meaningful (they
  reflect the actual minimum-image distances under each frame's
  per-frame box) and visibly match SPC/E theory, but peak *heights*
  are biased by the volume mismatch. Variable-box-aware
  normalisation is listed under *Out of Scope* in
  `rqm/analysis/framework.md`.

The bundle is therefore a useful pipeline demonstration, a working
template for a real-world workflow, and a sanity check that the
engine produces water-like RDFs from a rigid SPC/E + SHAKE + SPME +
NPT setup; for production-quality `g(r)` heights, treat it as
scaffolding to refine (longer run, better initial equilibration)
rather than as a finished calculation.
