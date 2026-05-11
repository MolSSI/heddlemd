#!/usr/bin/env python3
# Generates `init.xyz` for the lj-10000-argon example.
#
# Layout: 20 x 20 x 25 simple-cubic lattice = 10,000 argon atoms,
# spacing a = 4.0e-10 m, box (Lx, Ly, Lz) = (8.0e-9, 8.0e-9, 1.0e-8) m.
# Lattice is centred on the origin so every position lies inside the
# primary cell [-L/2, L/2) per axis.

nx, ny, nz = 20, 20, 25
a = 4.0e-10                      # lattice spacing (m)
lx, ly, lz = nx * a, ny * a, nz * a

with open("init.xyz", "w") as f:
    n = nx * ny * nz
    f.write(f"{n}\n")
    f.write(
        f'Lattice="{lx:.9e} 0 0 0 {ly:.9e} 0 0 0 {lz:.9e}" '
        f"Properties=species:S:1:pos:R:3\n"
    )
    cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
    for i in range(nx):
        x = (i - cx) * a
        for j in range(ny):
            y = (j - cy) * a
            for k in range(nz):
                z = (k - cz) * a
                f.write(f"Ar {x:.9e} {y:.9e} {z:.9e}\n")
