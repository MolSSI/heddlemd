# `rqm mv`

Relocate a blob to a different position, either within its current
file or in another managed file.

```
rqm mv <src> <dst> [--before | --split]
```

- `<src>` — `<path>:<line>` of any line inside the blob to move.
- `<dst>` — `<path>:<line>`, `<path>:start`, or `<path>:end`.
- `--before` — insert before the destination blob (default is after).
- `--split` — split the destination blob at the requested line, insert
  the moved blob at the split point.

`--before` and `--split` are mutually exclusive. `:start` and `:end`
take only the default modifier (no `--before` or `--split`).

## What happens

`mv` is a file-tree-only operation by default. The moved blob's bytes,
its content hash, and the owning requirement's `source_blobs` are
unchanged; only the entry's position in some file-tree changes.

With `--split`, additionally:

1. The destination's anchor blob is divided into `left` and `right`
   halves at the requested line.
2. The two halves are written as new blobs.
3. Every meta that owned the anchor blob has it replaced by the two
   halves in `source_blobs`. Joint ownership is handled — if the
   anchor was owned by N requirements, all N are updated in one
   operation.
4. The destination file-tree's entry list goes from
   `[..., anchor, ...]` to `[..., left, moved, right, ...]`.

After any successful move, `rqm build` runs to update the working
tree.

## Examples

**Move a function across files, append at the destination:**

```
rqm mv src/forces/lj.rs:42 src/forces/morse.rs:end
```

**Insert before the first blob in a file:**

```
rqm mv src/old.rs:10 src/new.rs:start
```

**Reorder within a single file (move the third blob to before the
first):**

```
rqm mv src/x.rs:100 src/x.rs:1 --before
```

**Split the destination blob and insert in the middle:**

```
rqm mv src/donor.rs:40 src/target.rs:120 --split
```

The blob containing line 120 of `target.rs` is divided at line 120;
the moved blob is inserted at the cut.

## Same-file moves

When `<src>` and `<dst>` are in the same file, the tool handles the
index-shift bookkeeping internally. Moving a blob to immediately
adjacent to itself (the no-op move) is rejected with a clear error:

```
rqm: error: source and destination resolve to the same position;
            nothing to move
```

## Consequence of `--split`

After `--split`, the destination file has a blob boundary that does
not correspond to a `// rq-XXXXXXXX` stamp in the source. The
materialized file is byte-perfect and `rqm check` is clean — the file
just has fewer stamps than blob boundaries. This is expected and is
discussed under [Limitations](../limitations.md).

## Limitations

- The moved blob's *owner* is preserved verbatim. To move a blob *and*
  re-attribute it to a different requirement, separate operations are
  needed; an `rqm reassign` command is not yet implemented.
- Move does not currently work on alias IDs (it operates on file:line
  positions only).
- There is no dry-run mode. Inspect with `rqm log <src>` first if
  uncertain.
