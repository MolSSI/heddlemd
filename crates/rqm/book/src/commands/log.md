# `rqm log`

Show the current state of a requirement.

```
rqm log <target>
```

`<target>` is the standard form (see [Addressing](../addressing.md)).
Unlike `rqm edit`, `log` follows aliases — passing an alias id resolves
to the canonical and shows the canonical's information.

## What is shown

```
$ rqm log rq-4d1082c4
requirement: rq-4d1082c4  (behavior)
  meta: e54fff41cac95c1335013fa533191c6afa2d012fc73f1db918a02678bbb658be

ancestry:
  (DAG root)

children (7):
  rq-2caa4efb  ## Feature API <!-- rq-2caa4efb -->
  rq-60c1e792  ## Out of Scope <!-- rq-60c1e792 -->
  ...

text_blob: 5c2d1a644bb79e81a2e4c4fcafa9d77e22be146ac459ca41af187f701b97a671
  # Feature: Radial Distribution Function Analysis (`rdf`) ...

source_blobs (2):
  c672bd35...  tests/rdf.rs:1
  d6866fc0...  src/analysis/rdf.rs:1
```

Sections:

- **header** — canonical stable_id, `kind`, current meta hash.
- **ancestry** — parent → grandparent → ... up to the DAG root, with a
  first-line preview of each.
- **children** — direct children with their first-line previews.
- **text_blob** — the blob hash and a preview of its first line.
- **source_blobs** — for each blob, the hash and a `path:start-line`
  for every managed path that contains it.
- **aliases** — if any aliases point at this canonical, they are
  listed.

## When to use

- To understand what a requirement is and where it's implemented
  before editing it.
- To explore the requirement DAG from any starting point.
- To find the canonical id for a bullet alias (`rqm log rq-alias_id`
  shows `requirement: rq-canonical_id` in the header).
- To find the source-file location of a requirement's implementation
  (the `source_blobs` section lists every occurrence).

## Limitations

Version history is not yet tracked. `rqm log` shows the *current*
state; there is no way to see previous versions of a requirement's
text or earlier sets of source blobs. Adding history would require a
`predecessor: Option<Hash>` field on requirements, updated on every
meta rewrite; this is a planned extension.
