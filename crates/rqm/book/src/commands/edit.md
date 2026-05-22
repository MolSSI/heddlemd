# `rqm edit`

Replace a blob's bytes. Interactive by default (opens `$EDITOR`);
non-interactive via `--from-file` or `--from-stdin`. On save / accept,
`.rqm/` is updated and the working tree is rebuilt.

```
rqm edit <target> [--from-file <path> | --from-stdin]
```

- `<target>` — `rq-XXXXXXXX` (text-blob of that requirement) or
  `<path>:<line>` (whichever blob covers that line in either a
  markdown or source file).
- `--from-file <path>` — read replacement bytes from a file.
- `--from-stdin` — read replacement bytes from stdin.

When neither flag is given, `$EDITOR` opens with the blob's current
contents; save and exit to apply.

`<target>` follows the standard convention — see
[Addressing](../addressing.md):

- `rq-XXXXXXXX` — edit the requirement's text-blob (markdown prose).
- `<path>:<line>` — edit whichever blob covers that line.
  - In a markdown file, this is the text-blob of the enclosing
    requirement.
  - In a source file, this is the source-blob containing the line.

## What happens

1. The tool resolves the target to a specific blob hash.
2. Writes the blob's current bytes to a temp file.
3. Spawns `$EDITOR` on the temp file.
4. On editor exit:
   - If the editor exits non-zero, the operation is canceled.
   - If the bytes are unchanged, no-op (`no changes`).
   - Otherwise the new bytes are written as a new blob, and every
     meta and file-tree that referenced the old blob is rewritten to
     reference the new blob.
5. `rqm build` runs to update the working tree.

## Joint ownership

When the edited blob is owned by multiple requirements (a source line
stamped with multiple stable_ids), *all* their metas are updated in one
operation. The output lists each updated meta.

## Aliases are rejected

`rqm edit rq-XXXXXXXX` refuses to operate on an alias, because the
edit would silently mutate the canonical's text-blob — likely more
than the user intended:

```
$ rqm edit rq-cccccccc
rqm: error: rq-cccccccc is an alias for rq-bbbbbbbb; edit the canonical
            id directly. ...
```

Use the canonical id (or a file:line cursor) instead.

## Output

```
$ rqm edit rq-4d1082c4
new blob: 32e347c03c29f79e7591beba56fb6fc4ce6169fdb0309b1ca5cfd5a4da6f26b9
  updated meta: rq-4d1082c4
  rewrote file-tree: rqm/analysis/rdf.md
  wrote rqm/analysis/rdf.md
```

## Limitations

- The text-blob's heading line carries the requirement's
  `<!-- rq-XXXXXXXX -->` annotation. Changing the annotation during an
  edit is a footgun — the meta still uses the original stable_id, so
  the annotation in source becomes stale. There is no `rqm rename`
  yet; don't edit the annotation in place.
- The edit operates on whatever bytes the blob currently has. If the
  working tree has hand-edits not yet captured by `rqm migrate`, the
  starting point of the edit is the `.rqm/` version (not what you see
  on disk). Run `rqm check` first if you're unsure.
