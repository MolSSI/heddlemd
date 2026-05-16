#!/usr/bin/env python3
# Generates `spc.topology` for the spc-water-256 example.
#
# 256 waters → 256 angles (one HOH per molecule) and 512 bonds (two OH per
# molecule). The 1-2 and 1-3 exclusions auto-derive from the bonds and
# angles. Atom indices per molecule m: O = 3m, H1 = 3m+1, H2 = 3m+2.

N_WATERS = 4 * 4 * 16  # must match generate_init.py

def main():
    with open("spc.topology", "w") as f:
        f.write("# Topology for 256 SPC water molecules\n")
        f.write("# O at 3m, H1 at 3m+1, H2 at 3m+2\n\n")
        f.write("[bonds]\n")
        for m in range(N_WATERS):
            o, h1, h2 = 3 * m, 3 * m + 1, 3 * m + 2
            f.write(f"{o} {h1} OH\n")
            f.write(f"{o} {h2} OH\n")
        f.write("\n[angles]\n")
        for m in range(N_WATERS):
            o, h1, h2 = 3 * m, 3 * m + 1, 3 * m + 2
            # atom_j = O (centre), wings are H1 and H2.
            f.write(f"{h1} {o} {h2} HOH\n")

if __name__ == "__main__":
    main()
