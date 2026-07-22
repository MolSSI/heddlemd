# Local Extensions to the rr-implement Skill

Instructions added below extend the rr-implement skill. Where they conflict with the
skill's built-in instructions, this file wins.

- Before implementing a new registry item — a thermostat, integrator, barostat, pair potential, or
  bonded potential — read the **Extending HeddleMD** developer guide in `book/src/extending/`.
  Start with the overview (`book/src/extending/index.md`), then read the specific page for the
  kind of registry item you are adding (`adding-a-thermostat.md`, `adding-an-integrator.md`,
  `adding-a-barostat.md`, `adding-a-pair-potential.md`, or `adding-a-bonded-potential.md`).
  Follow the registry, configuration, CUDA-kernel-wiring, and determinism conventions those
  pages describe.
