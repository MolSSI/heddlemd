# Workflows

This chapter walks through end-to-end uses of `rqm` for the scenarios
that come up most often.

## Initial migration of an existing project

Goal: bring an existing project that already uses `rq-XXXXXXXX` stamps
into `.rqm/` management.

```
$ rqm init
initialized .rqm
$ rqm migrate rqm/feature.md
migrated markdown rqm/feature.md
$ rqm migrate src/feature.rs
migrated source src/feature.rs
$ rqm migrate tests/feature.rs
migrated source tests/feature.rs
$ rqm check
ok (3 paths)
```

Markdown files must be migrated before source files that reference
them. When a source migration fails because of an unresolved stamp,
the error names which req file is missing — migrate that and retry.

For a heavily-shared source file (e.g. one that several features
contribute to), all of its owning requirements' markdown files must be
migrated first. This is the *migration neighborhood* — a cluster of
requirements that come in together. Plan migrations one neighborhood
at a time.

## Editing a requirement's prose

```
$ rqm edit rqm/io/config-schema.md:120
# (editor opens with the text-blob containing line 120)
# save and quit
new blob: 8a3b...
  updated meta: rq-XXXXXXXX
  rewrote file-tree: rqm/io/config-schema.md
  wrote rqm/io/config-schema.md
$ rqm check
ok (...)
```

The working tree is regenerated automatically. Commit the change to
`.rqm/` and the materialized file together; both are tracked.

## Editing source code through `.rqm/`

For a source-blob edit, the file:line cursor works the same way:

```
$ rqm edit src/forces/spme.rs:200
# editor opens with the blob containing line 200
```

This is the recommended way to edit source code once a file is under
`rqm` management. Hand-editing the source file directly will trigger
`rqm check` to fail.

## Inspecting before editing

When you don't remember a requirement's stable_id, or want to see what
it owns before making changes:

```
$ rqm log src/integrator/csvr.rs:50
requirement: rq-7a124d43  (behavior)
  ...
ancestry:
  rq-891232bf  # Feature: CSVR ...
children (...): ...
source_blobs (2):
  ...  src/integrator/csvr.rs:42
  ...  tests/integrators_csvr.rs:90
```

The output shows everything you need to plan an edit: the canonical
stable_id, where the requirement lives in the DAG, and every source
location that implements it.

## Reorganizing source files

When refactoring source layout — for example, splitting a heavily-
shared file along feature lines — `rqm mv` does the work:

```
# Move the SpmeConfig type from config.rs to a feature-local file.
$ rqm mv src/io/config.rs:419 src/forces/spme.rs:end
moved blob: 7c4f...
  rewrote file-tree: src/io/config.rs
  rewrote file-tree: src/forces/spme.rs
  wrote src/io/config.rs
  wrote src/forces/spme.rs
$ rqm check
ok (...)
```

The owning requirement is unchanged — only the file the blob is in
has changed. Tests should pass without further code changes (the
blob's stamp travels with it).

## CI lockdown

The contract: every change to managed files must go through `rqm`.
The check that enforces this is a single test that runs on every CI
build:

```rust
// crates/rqm/tests/lockdown.rs
let store = Store::open(workspace_root.join(".rqm"))?;
let report = check(&store, &workspace_root)?;
assert!(report.is_clean());
```

If someone hand-edits a managed file and forgets to push it through
`rqm`, this test fails with the specific path and reason. The fix is
either `rqm migrate <path>` (accept the hand-edit) or `rqm build`
(revert it).

## Recovering from drift

```
$ rqm check
diff: src/forces/spme.rs
```

Pick one of two recovery paths:

- The hand-edit is the version you want to keep:
  ```
  $ rqm migrate src/forces/spme.rs   # re-import into .rqm/
  $ rqm check                         # confirms clean
  ```
- The hand-edit was unintended:
  ```
  $ rqm build                         # overwrites with .rqm/ version
  $ rqm check                         # confirms clean
  ```

The right choice depends on which side is currently correct.
