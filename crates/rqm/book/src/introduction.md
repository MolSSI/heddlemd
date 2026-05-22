# Introduction

`rqm` is a requirements-traceability tool that maintains a structural
link between project requirements (markdown files under `rqm/`) and
the source code that implements them (`src/` and `tests/`). It is
designed to make drift between the two structurally impossible rather
than relying on author discipline.

## The problem rqm solves

Most projects encode requirements as documents and then trust authors
to keep code in sync. Three failure modes show up repeatedly:

1. Source code is written without any record of *which* requirement it
   satisfies.
2. Source code references requirement IDs that no longer exist.
3. Requirements are edited without revisiting the code that implements
   them.

A linter or convention catches some of these; none of them catches all
three. `rqm` addresses all three at once by treating the
requirement-to-source mapping as a first-class data structure stored in
a content-addressed object store at `.rqm/`. The on-disk
`rqm/`/`src/`/`tests/` files are *generated* from that store, and a CI
check (`rqm check`) fails when they diverge.

## Mental model

The working tree you see — markdown requirements files, source files,
test files — is a *materialization* of `.rqm/`. You edit `.rqm/` (via
`rqm edit` and friends), and the working tree is regenerated. Direct
edits to the working tree are detectable as drift and rejected in CI.

The store itself is git-flavored: every piece of content (a blob of
text, a requirement's metadata, an ordered list of blobs that
constitutes a file) is identified by the SHA-256 hash of its bytes.
Identity is content. There is no way to forge an ID.

## What you can do today

This documentation covers the Phase 1.5 surface, which is what is
implemented and tested. In rough order of how a project adopts the
tool:

- `rqm init` — create an empty `.rqm/`.
- `rqm migrate` — bootstrap an existing markdown or source file into
  the store.
- `rqm build` — materialize the working tree from `.rqm/`.
- `rqm check` — verify the working tree matches `.rqm/`.
- `rqm log` — show a requirement's current state, ancestry, children,
  and source-blob locations.
- `rqm edit` — open a blob's bytes in `$EDITOR`; on save, `.rqm/` is
  updated and the working tree is rebuilt.
- `rqm mv` — relocate a blob to a different position within its file
  or in a different file.

What's not yet implemented (and where to find the design notes):
version history, requirement renaming, GC of unreachable objects, and a
few other things called out under [Limitations](./limitations.md).

## Where to read next

If you want to use `rqm` in a project, start with
[Workflows](./workflows.md) for end-to-end walkthroughs. If you want to
understand what's happening under the hood, start with
[Data model](./model.md) and [Addressing](./addressing.md).
