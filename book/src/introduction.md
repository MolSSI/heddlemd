# Introduction

HeddleMD is a molecular dynamics engine for simulating condensed-phase
systems — liquids, solutions, and solvated biomolecules — on a single
NVIDIA GPU. You describe a system with a force field, a starting
configuration, and a thermodynamic ensemble, and the engine propagates
it in time and writes a trajectory you can analyse.

Out of the box you can:

- Run **NVE, NVT, and NPT** dynamics, with a choice of thermostats
  (Nosé-Hoover chains, CSVR, Andersen, Berendsen) and barostats
  (stochastic cell-rescale, Monte-Carlo, Berendsen).
- Build a force field from **Lennard-Jones** van der Waals terms,
  **smooth particle-mesh Ewald** (SPME) electrostatics, and **bonded**
  interactions (harmonic and Morse bonds, harmonic angles, periodic
  torsions) with automatic 1-2/1-3 exclusions and 1-4 scaling.
- Keep water and other rigid groups rigid with **SETTLE** or
  **SHAKE/RATTLE** constraints, so you can take the longer timesteps
  those force fields assume.
- **Minimize** a starting structure before sampling, then chain
  **multiple phases** (equilibration followed by production) in a single
  run.
- Post-process trajectories in place with `heddlemd analyze` — the
  **radial distribution function** ships built-in.

Every quantity that affects the physics is set in a human-readable TOML
file, in SI units by default, and inputs can be checked on a login node
with `heddlemd lint` before a job reaches the queue.

## Reproducibility

A distinguishing feature of HeddleMD is **bit-wise reproducibility**:
rerun the same input on the same GPU and you get a byte-identical
trajectory and log, every time. This makes debugging, regression
testing, and sharing an exact result with a collaborator
straightforward. The scope and limits of the guarantee are spelled out
in [Reproducibility](guide/reproducibility.md).

## What's in this book

- **Getting Started** walks through installation and running the bundled
  10,000-atom Lennard-Jones argon example.
- **User Guide** covers building your own simulation: the TOML config,
  the extended-XYZ starting-structure file, output formats, trajectory
  analysis, and what reproducibility does and does not guarantee.
- **Reference** documents the command-line interface.

This book is the user-facing guide. A companion developer guide covering
the engine's internal design — the GPU compute pipeline and the
deterministic-reduction strategy — is planned separately; until then the
internals live in `docs/architecture.md` in the repository.
