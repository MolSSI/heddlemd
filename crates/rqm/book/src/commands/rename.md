# `rqm rename`

Change a requirement's stable_id everywhere.

```
rqm rename <old> <new>
```

- `<old>` — `rq-XXXXXXXX` or `<path>:<line>`. Must resolve to a
  canonical requirement (aliases are rejected).
- `<new>` — a fresh `rq-XXXXXXXX` (8 lowercase hex chars) not already
  registered as a canonical or an alias.

## What it touches

A rename is comprehensive — it changes the stable_id in every place
the old id appears:

1. **Ref name** — `.rqm/refs/<old>` is deleted; `.rqm/refs/<new>` is
   created.
2. **Meta's `stable_id` field** — the renamed requirement's meta gets
   its primary identifier updated.
3. **Children's `parents` lists** — every meta that lists the old id
   in its `parents` is rewritten to use the new id.
4. **File-tree entries** — every `entry.stable_id` matching the old
   id is updated to the new id (across all managed paths).
5. **Aliases pointing at it** — every alias whose canonical was the
   old id is redirected to the new id.
6. **Blob contents** — every reachable blob whose bytes contain the
   old id as a word-bounded `rq-XXXXXXXX` stamp is rewritten. Since
   both ids are 11 bytes, the substitution preserves blob size and
   line numbers. The materialized source and markdown files no longer
   contain any reference to the old id.

After all updates, `rqm build` runs to push the rewritten blobs out
to the working tree.

## Validation

`rqm rename` refuses:

- **Unknown target.** `unknown stable_id: rq-...`.
- **Alias target.** `... is an alias for ...; rename the canonical
  id instead.` — aliases are migration-era constructs that don't
  have their own metas.
- **Invalid new id format.** `invalid new stable_id "..."; expected
  rq-XXXXXXXX (8 lowercase hex digits)`.
- **New id already a canonical.** `... already exists as a
  canonical requirement`.
- **New id already an alias.** `... already exists as an alias`.

Renaming a requirement to itself (`old == new`) is a no-op and
succeeds silently.

## Example

```
$ rqm rename rq-4d1082c4 rq-1234abcd
renamed rq-4d1082c4 -> rq-1234abcd
  blobs rewritten: 3
  updated meta: rq-1234abcd
  updated meta: rq-fb62c422
  ...
  rewrote file-tree: rqm/analysis/rdf.md
  rewrote file-tree: src/analysis/rdf.rs
  rewrote file-tree: tests/rdf.rs
  redirected alias: rq-2dc76b67
  wrote rqm/analysis/rdf.md
  wrote src/analysis/rdf.rs
  wrote tests/rdf.rs
```

## Why blob-content rewriting is the right scope

A metadata-only rename (Option 1 in the original design choice) would
have left stamps like `// rq-4d1082c4` inside source files pointing
at an id that no longer exists. Those stale tokens are invisible to
the integrity check (the check function does not parse blob contents
for stamps), so they would have silently misled anyone reading the
source.

The comprehensive rewrite produces a clean end state at the cost of
~30 extra lines of byte-substitution code. Once stamps are removed
from source/markdown entirely (a planned post-migration cleanup),
the blob-content rewrite becomes a no-op — but until then it's
load-bearing for "rename leaves no trace of the old id."

## Limitations

- Renames affect only the named requirement. To rename multiple
  requirements, invoke `rqm rename` multiple times in sequence.
- A rename does not introduce DAG changes — the requirement's
  position, children, and parents are unchanged (modulo updating
  parents lists in children that pointed at the old name).
- The old meta object and old blob objects remain in
  `.rqm/objects/` as immutable history until a future `rqm gc`
  prunes unreachable objects.
