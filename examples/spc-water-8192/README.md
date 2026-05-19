# 8192-molecule rigid SPC/E water, NPT, with RDF post-processing

A larger, more realistic example that exercises rigid water (SETTLE),
SPME electrostatics, multi-phase NVT-then-NPT scheduling (CSVR
thermostat + c-rescale barostat), and the `dynamics analyze`
post-processing pipeline. 8192 SPC/E water molecules (24,576 atoms)
integrated for 10 ps of NVT equilibration followed by 30 ps of NPT
production at 298.15 K, 1 atm, then analyzed for the O-O, O-H, and
H-H radial distribution functions of the production trajectory.

## Layout

- `water.in.toml` — simulation config (rigid SPC/E + SETTLE + SPME,
  two phases: `equil` and `prod`).
- `water.in.xyz` — initial state: 16 × 16 × 32 simple-cubic lattice
  of water molecules at sub-liquid density (~0.47 g/cm³), each with a
  randomized SO(3) orientation seeded for determinism.
- `water.in.topology` — 8192 SETTLE constraint groups, one per
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

This is **not** a quick example. On a recent NVIDIA GPU the combined
20,000-step run (10 ps NVT + 30 ps NPT at `dt = 2 fs`) takes roughly
20 minutes; the dominant cost is the cell-list rebuild as the box
volume changes under the barostat.

The run produces six output files in this directory — three per
phase. The equilibration phase suppresses trajectory output, so its
`.xyz` file is not created:

- `water.out.equil.log` — 51 rows of `step, time, kinetic_energy,
  temperature, csvr_conserved` (NVT diagnostics).
- `water.out.equil.timings` — equilibration performance summary.
- `water.out.prod.xyz` — 31 production frames (1 per ps), ~38 MB.
- `water.out.prod.log` — 151 rows of `step, time, kinetic_energy,
  temperature, csvr_conserved, pressure, box_volume,
  c_rescale_conserved` (NPT diagnostics).
- `water.out.prod.timings` — production performance summary.

## Analyze

After `dynamics run` has produced `water.out.prod.xyz`, post-process
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

reports `[dynamics lint] OK` (after the trajectory exists).

## Parameters

- SPC/E geometry: `r_OH = 1.0 Å`, `θ_HOH = 109.47°`, enforced rigidly
  by SETTLE.
- SPC/E charges: `q_O = -0.8476 e`, `q_H = +0.4238 e`.
- SPC/E Lennard-Jones (O–O only): `σ = 3.166 Å`, `ε = 78.2 k_B`
  (`1.080 × 10⁻²¹ J`). The H–H and O–H pair entries carry a negligible
  `ε = 1 × 10⁻³⁰ J` only to satisfy the loader's "every unordered pair
  declared" rule.
- Electrostatics: SPME with `α = 3.5 nm⁻¹`, real-space cutoff
  `r_cut = 1 nm` (matching the LJ cutoff so one neighbour list serves
  both), FFT grid `(48, 48, 96)` (~1 Å spacing, small-prime-factored
  for cuFFT), spline order 4.
- Integrator: `velocity-verlet`, `dt = 2 fs` (allowed because SETTLE
  removes the high-frequency OH stretching mode).
- Thermostat: CSVR (Bussi-Donadio-Parrinello), `T = 298.15 K`,
  `τ = 100 fs`.
- Barostat: c-rescale (Bernetti-Bussi), `P = 1.013 × 10⁵ Pa`,
  `τ = 1 ps`, `β = 4.5 × 10⁻¹⁰ Pa⁻¹`.
- Cell list: `max_neighbors = 1024` (liquid-density water with a
  1.1 nm shell needs ~560 neighbours per atom; 1024 leaves headroom),
  `r_skin = 1 Å`.

## What you should see in the RDFs

Peak positions are dead-on for SPC/E water — peak heights are
inflated by ~30 % (see *Caveats* below).

| RDF       | Bundle peak                                        | SPC/E textbook position |
|-----------|----------------------------------------------------|-------------------------|
| O-O       | first peak at **2.78 Å**, g ≈ 3.5                  | 2.75 Å, g ≈ 2.9         |
| O-O       | second peak / first minimum at **3.5 Å**           | 3.5 Å                   |
| O-O       | second neighbour shell **~4.5–5 Å**, g ≈ 1.34      | 4.5 Å                   |
| O-H       | intramolecular bond at **0.998 Å** (SETTLE pin)    | 1.000 Å                 |
| O-H       | **hydrogen-bond peak at 1.77 Å**, g ≈ 1.51         | 1.85 Å                  |
| O-H       | second peak at **3.17 Å**, g ≈ 1.89                | 3.30 Å                  |
| H-H       | intramolecular pair at **1.628 Å** (SETTLE pin)    | 1.633 Å                 |
| H-H       | first intermolecular at **2.50 Å**, g ≈ 1.67       | 2.40 Å                  |

The intramolecular SETTLE-constrained peaks (O-H at 0.998 Å, H-H at
1.628 Å) appear as nearly-delta-function spikes, confirming the
constraint algorithm is enforcing the rigid SPC/E geometry exactly.
The hydrogen-bond signature in the O-H RDF at 1.77 Å is the
characteristic structural marker of liquid water and is well
reproduced here.

## Caveats

This bundle's primary value is demonstrating the engine end-to-end at
realistic system size: rigid water + SETTLE + SPME + NPT + RDF
post-processing all run cleanly, and the resulting RDFs show the
correct liquid-water peak structure. It is **not** a fully converged
production-quality calculation — two limitations are worth knowing
about before treating the `g(r)` heights as quantitative:

- **Equilibration from a lattice initial state is hard.** A
  rigid-body lattice of charges has a large virial pressure even when
  the molecule orientations are randomized, and the c-rescale
  barostat in the production phase responds by expanding the box. In
  this 30 ps NPT phase (following 10 ps of NVT equilibration) the box
  does not converge to the SPC/E equilibrium liquid density — the
  system instead settles into a metastable two-phase configuration
  (liquid cluster + low-density surroundings). The asymptotic value
  of every `g(r)` in this bundle plateaus around 1.30–1.35 rather
  than the canonical 1.0; the offset is the ratio of the actual
  in-cluster density to the box-averaged density. For a strictly
  converged liquid-water density users should either:
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
engine produces water-like RDFs from a rigid SPC/E + SETTLE + SPME +
NPT setup; for production-quality `g(r)` heights, treat it as
scaffolding to refine (longer run, better initial equilibration)
rather than as a finished calculation.
