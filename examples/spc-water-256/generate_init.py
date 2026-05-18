#!/usr/bin/env python3
# Generates `spc.in.xyz` for the spc-water-256 example.
#
# Layout: 4 x 4 x 16 simple-cubic lattice = 256 water molecules
# (768 atoms total) on a (2.0 x 2.0 x 8.0) nm box. Lattice spacing
# a = 5.0e-10 m so each water sits at the centre of its unit cell.
# Each water is placed with its oxygen at the lattice site, the first
# hydrogen at (+r_OH, 0, 0), and the second hydrogen at the SPC bond
# angle 109.47° from the first, both lying in the xy-plane.

import math

NX, NY, NZ = 4, 4, 16          # 4*4*16 = 256 waters → 768 atoms
A = 5.0e-10                    # lattice spacing (m)
LX, LY, LZ = NX * A, NY * A, NZ * A
R_OH = 1.0e-10                 # SPC equilibrium O–H bond length (m)
THETA_HOH = 1.910611931        # SPC equilibrium H–O–H angle (rad, 109.47°)

def main():
    n_atoms = 3 * NX * NY * NZ
    with open("spc.in.xyz", "w") as f:
        f.write(f"{n_atoms}\n")
        f.write(
            f'Lattice="{LX:.9e} 0 0 0 {LY:.9e} 0 0 0 {LZ:.9e}" '
            f"Properties=species:S:1:pos:R:3\n"
        )
        cx0, cy0, cz0 = (NX - 1) / 2.0, (NY - 1) / 2.0, (NZ - 1) / 2.0
        h1x = R_OH
        h1y = 0.0
        h2x = R_OH * math.cos(THETA_HOH)
        h2y = R_OH * math.sin(THETA_HOH)
        for i in range(NX):
            for j in range(NY):
                for k in range(NZ):
                    cx = (i - cx0) * A
                    cy = (j - cy0) * A
                    cz = (k - cz0) * A
                    f.write(f"O  {cx:.9e} {cy:.9e} {cz:.9e}\n")
                    f.write(f"H  {cx + h1x:.9e} {cy + h1y:.9e} {cz:.9e}\n")
                    f.write(f"H  {cx + h2x:.9e} {cy + h2y:.9e} {cz:.9e}\n")

if __name__ == "__main__":
    main()
