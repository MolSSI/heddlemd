#!/usr/bin/env python3
# Generates `water.in.xyz` for the spc-water-8192 example.
#
# Layout: 16 x 16 x 32 simple-cubic lattice = 8192 water molecules
# (24,576 atoms total) at sub-liquid initial density. Lattice spacing
# a = 4.0e-10 m gives a 6.4 x 6.4 x 12.8 nm box, centred at the origin
# (initial density ~0.47 of SPC/E liquid). The c-rescale barostat
# contracts the box toward equilibrium during the simulation; the
# rationale for starting under-dense rather than at exact liquid
# density is documented in README.md.
#
# Each molecule is placed with its oxygen at the lattice site and a
# random rigid-body orientation (uniform on SO(3) via Shoemake's
# quaternion construction, seeded for determinism). H1 sits at distance
# r_OH from O along the molecule's local +x axis; H2 sits at the SPC/E
# H-O-H angle (109.47 deg) from H1 in the local xy-plane. The random
# orientations remove the artificial ferroelectric dipole order of an
# aligned initial state, which would otherwise produce a large initial
# pressure that the barostat must violently relax.

import math
import random

NX, NY, NZ = 16, 16, 32        # 16*16*32 = 8192 waters -> 24,576 atoms
A = 4.0e-10                    # lattice spacing (m); ~0.47 of liquid density
LX, LY, LZ = NX * A, NY * A, NZ * A
R_OH = 1.0e-10                 # SPC/E O-H bond length (m)
THETA_HOH = 1.910611931        # SPC/E H-O-H angle (rad; 109.47 deg)
RNG_SEED = 42

def random_rotation_matrix(rng):
    # Shoemake (1992) uniform-on-SO(3) construction.
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    q0 = math.sqrt(1.0 - u1) * math.sin(2 * math.pi * u2)
    q1 = math.sqrt(1.0 - u1) * math.cos(2 * math.pi * u2)
    q2 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q3 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    return [
        [1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q3 * q0), 2 * (q1 * q3 + q2 * q0)],
        [2 * (q1 * q2 + q3 * q0), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q1 * q0)],
        [2 * (q1 * q3 - q2 * q0), 2 * (q2 * q3 + q1 * q0), 1 - 2 * (q1 * q1 + q2 * q2)],
    ]

def apply(R, v):
    return (
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    )

def main():
    rng = random.Random(RNG_SEED)
    n_atoms = 3 * NX * NY * NZ
    h1_local = (R_OH, 0.0, 0.0)
    h2_local = (R_OH * math.cos(THETA_HOH), R_OH * math.sin(THETA_HOH), 0.0)
    cx0 = (NX - 1) / 2.0
    cy0 = (NY - 1) / 2.0
    cz0 = (NZ - 1) / 2.0
    with open("water.in.xyz", "w") as f:
        f.write(f"{n_atoms}\n")
        f.write(
            f'Lattice="{LX:.9e} 0 0 0 {LY:.9e} 0 0 0 {LZ:.9e}" '
            f"Properties=species:S:1:pos:R:3\n"
        )
        for i in range(NX):
            for j in range(NY):
                for k in range(NZ):
                    cx = (i - cx0) * A
                    cy = (j - cy0) * A
                    cz = (k - cz0) * A
                    R = random_rotation_matrix(rng)
                    h1 = apply(R, h1_local)
                    h2 = apply(R, h2_local)
                    f.write(f"O  {cx:.9e} {cy:.9e} {cz:.9e}\n")
                    f.write(
                        f"H  {cx + h1[0]:.9e} {cy + h1[1]:.9e} {cz + h1[2]:.9e}\n"
                    )
                    f.write(
                        f"H  {cx + h2[0]:.9e} {cy + h2[1]:.9e} {cz + h2[2]:.9e}\n"
                    )

if __name__ == "__main__":
    main()
