# Data model

This chapter describes the on-disk structure of `.rqm/` and the
relationships between its objects.

## Directory layout

```
.rqm/
  objects/<aa>/<rest>     content-addressed objects (SHA-256, 2-char prefix)
  refs/<rq-XXXXXXXX>      stable_id -> current meta hash
  aliases/<rq-XXXXXXXX>   alias_id -> canonical stable_id (migration-era)
  trees/<path>            managed path -> file-tree hash (mirrors path layout)
  managed_paths           plain-text list of paths under rqm's control
```

The store is rooted at `.rqm/` in the workspace root by default;
`--rqm-dir` overrides.

## Object types

Three kinds of objects live in `.rqm/objects/`. The file's name is the
64-character hex SHA-256 hash of its bytes; the first two hex
characters are the subdirectory.

### Blob

Raw bytes. No structure. Used both for the prose of a requirement and
for chunks of source code. A blob's identity *is* its content.

### Requirement meta (JSON)

```json
{
  "type": "requirement",
  "stable_id": "rq-4d1082c4",
  "kind": "behavior",
  "text_blob": "<64-hex>",
  "parent": "rq-XXXXXXXX",
  "source_blobs": ["<64-hex>", "<64-hex>", "..."]
}
```

A requirement carries:

- **`stable_id`** — a persistent identifier (`rq-` followed by 8 hex
  characters). Does not change across edits.
- **`kind`** — `behavior` (must have at least one source blob),
  `design` (documentation-only — out-of-scope material, glossary,
  design intent), or `pending` (aspirational, source forthcoming).
- **`text_blob`** — hash of the blob holding this requirement's
  markdown prose (its heading plus body, up to but not including
  child headings).
- **`parent`** — stable_id of the parent requirement in the DAG, or
  omitted at DAG roots. Parents are referenced by stable_id (not
  meta hash) so that updating a parent's content does not cascade to
  rewrite all its descendants.
- **`source_blobs`** — hashes of blobs the requirement owns in source
  files. Sorted canonically so the meta hash is stable across
  reorderings.

### File tree (JSON)

```json
{
  "type": "file_tree",
  "path": "src/analysis/rdf.rs",
  "entries": [
    { "stable_id": "rq-4d1082c4", "blob": "<64-hex>" },
    { "stable_id": "rq-cfd1d536", "blob": "<64-hex>" }
  ]
}
```

A file tree describes how a managed file is assembled. `entries` is
ordered; the materialized file is the byte-concatenation of
`entries[i].blob` for `i` in order. `stable_id` on each entry is the
canonical owner of that blob; the entry's `blob` is one of that
requirement's `source_blobs` (or its `text_blob` for markdown files).

## Refs, aliases, and trees

Three flat index directories make lookup O(1):

- **`refs/<rq-XXXXXXXX>`** — text file containing the hex hash of the
  current meta for that stable_id. Updated whenever a meta is
  rewritten (e.g. by `rqm edit`).
- **`aliases/<rq-XXXXXXXX>`** — text file containing a canonical
  stable_id. Created at migration time when a markdown file contains
  bullet-level rq stamps that are not promoted to first-class
  requirements. See [Addressing](./addressing.md) for the resolution
  rule.
- **`trees/<path>`** — text file containing the file-tree hash for
  the managed path. The directory mirrors the workspace layout.

## Stable IDs vs content hashes

`rqm` uses two kinds of identifier:

| | Stable ID | Content hash |
|---|---|---|
| Format | `rq-XXXXXXXX` (8 hex chars) | 64 hex chars (SHA-256) |
| Stability | persists across edits | changes whenever content changes |
| Used in | `refs/`, `parent`, file-tree entries, source stamps | `text_blob`, `source_blobs`, tree refs |

The separation is the central design decision that prevents cascading
rewrites: when you edit one requirement, only that meta's hash
changes; nothing further up or down the DAG needs to be touched.

## Round-trip invariant

The contract `rqm` enforces is:

> For every path listed in `.rqm/managed_paths`, the bytes on disk
> equal what `rqm build` would write from `.rqm/`.

`rqm check` is the executable form of this invariant. CI runs it on
every change.
