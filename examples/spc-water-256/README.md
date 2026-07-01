# 256-molecule flexible-SPC water

A small example that exercises the full simulation pipeline on a system
with intramolecular bonds, angles, and SPME electrostatics. 256 water
molecules (768 atoms) in a (2.0 × 2.0 × 8.0) nm box.

## Layout

- `spc.in.toml` — simulation config (SI units; 0.5 fs timestep; 100 steps
  at 300 K)
- `spc.in.xyz` — 256 water molecules on a 4 × 4 × 16 lattice
  (5.0 Å spacing). Each molecule is placed with O at the lattice site,
  H₁ at `(+r_OH, 0, 0)`, and H₂ at the SPC equilibrium HOH angle
  (109.47°) from H₁ in the xy-plane.
- `spc.in.topology` — bonds (512 O–H bonds), angles (256 H–O–H angles),
  and the implicit 1-2 + 1-3 exclusions auto-derived from those.
- `generate_init.py` — regenerates `spc.in.xyz` deterministically.
- `generate_topology.py` — regenerates `spc.in.topology` deterministically.

## Run

From this directory:

```
cargo run --release -- run spc.in.toml
```

Or with the debug binary already built:

```
../../target/debug/heddlemd run spc.in.toml
```

A run produces three files in this directory:

- `spc.out.run.xyz` — 11 trajectory frames (steps 0, 10, …, 100),
  extended-XYZ
- `spc.out.run.log` — 21 CSV rows of step, time, kinetic energy, temperature
- `spc.out.run.timings` — per-stage timing summary (kernels and host I/O)

## Parameters

- SPC geometry: r_OH = 1.0 Å, θ_HOH = 109.47°.
- SPC Lennard-Jones: σ_O = 3.166 Å, ε_O = 6.502 × 10⁻²² J. Off-diagonal
  pairs (O–H, H–H) have ε = 0.
- SPC charges: q_O = −0.82 e, q_H = +0.41 e (each water is electrically
  neutral).
- Bond forces use the `morse` variant of `[[bond_types]]` with
  parameters tuned so the curvature `2·D_e·a²` at the equilibrium bond
  length r_e = 1.0 Å matches the flexible-SPC harmonic stiffness
  k_b = 4.515 × 10⁵ J/m². With D_e = 2.0 × 10⁻¹⁹ J this gives
  a = √(k_b / (2 D_e)) ≈ 1.063 × 10¹² m⁻¹.
- Angle forces use the `harmonic` `[[angle_types]]` variant with the
  flexible-SPC bend stiffness k_θ = 5.27 × 10⁻¹⁹ J/rad² (Toukan-Rahman,
  75.9 kcal/mol/rad²) at θ₀ = 1.911 rad.
- Coulomb electrostatics use SPME: real-space Ewald cutoff = 6 Å
  (matching the LJ cutoff so both share the neighbour list),
  α = 5.83 × 10⁹ m⁻¹ (= 3.5 / r_cut_real), a 20 × 20 × 80 FFT grid
  (~1 Å spacing, small-prime-factored for cuFFT), and spline order
  four. The pairwise `[coulomb]` slot was retired; SPME is the only
  electrostatics option.

## Notes

- The 256-water box at this geometry has a number density of
  ≈ 8 molecules per nm³ — well below typical liquid water
  (~33 molecules per nm³). The intent is to keep the box small enough
  that the cell-list constraint `3 · (cutoff + r_skin) ≤
  perpendicular_width` is comfortably satisfied (2.0 nm minimum width
  vs. 1.95 nm required at the configured cutoff).
- The 0.5 fs timestep is sized for the OH stretching mode (period ≈
  9 fs at the SPC bond stiffness), and is shorter than the 1 fs used
  by the LJ-only argon example.
