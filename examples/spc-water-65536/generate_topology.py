#!/usr/bin/env python3
# Generates `water.in.topology` for the spc-water-65536 example.
#
# 65,536 waters -> 65,536 SETTLE constraint groups, one per molecule.
# Atom indices per molecule m: O = 3m, H1 = 3m+1, H2 = 3m+2 (matches
# the row order produced by `generate_init.py`).

N_WATERS = 32 * 32 * 64  # must match generate_init.py

def main():
    with open("water.in.topology", "w") as f:
        f.write("# Topology for 65,536 SPC/E water molecules\n")
        f.write("# O at 3m, H1 at 3m+1, H2 at 3m+2; each triple rigidly\n")
        f.write("# constrained via SETTLE under the `SPCE` constraint type.\n\n")
        f.write("[constraints]\n")
        for m in range(N_WATERS):
            o, h1, h2 = 3 * m, 3 * m + 1, 3 * m + 2
            f.write(f"{o} {h1} {h2} SPCE\n")

if __name__ == "__main__":
    main()
