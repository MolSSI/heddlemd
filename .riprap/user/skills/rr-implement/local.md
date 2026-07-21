# Local Extensions to the rr-implement Skill

This agent-neutral file belongs to your project. The Riprap template creates it once and never
updates it, so anything you write here survives `copier update`.

Instructions added below extend the rr-implement skill. Where they conflict with the
skill's built-in instructions, this file wins.

Examples of instructions you might add:

<!--
- Run the linter and fix all warnings before reporting the implementation complete.
- Never add a new dependency without first confirming with the user.
- Integration tests live in `tests/integration/` and follow the naming pattern there.
-->

- Before implementing a new plugin — a thermostat, integrator, barostat, pair potential, or
  bonded potential — read the **Extending HeddleMD** developer guide in `book/src/extending/`.
  Start with the overview (`book/src/extending/index.md`), then read the specific page for the
  kind of plugin you are adding (`adding-a-thermostat.md`, `adding-an-integrator.md`,
  `adding-a-barostat.md`, `adding-a-pair-potential.md`, or `adding-a-bonded-potential.md`).
  Follow the registry, configuration, CUDA-kernel-wiring, and determinism conventions those
  pages describe.
