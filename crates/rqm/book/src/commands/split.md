# `rqm split`

Divide a blob at a line, optionally re-attributing either half to
different owners. The primary mechanism for taking a coarse-grained
blob (e.g. a whole source file attributed to a file-root requirement)
and slicing it into smaller blobs attributed to more specific child
requirements.

```
rqm split <file>:<line> [--left <owner>...] [--right <owner>...]
```

- `<file>:<line>` — the split point. The blob covering that line is
  divided at the line boundary.
- `--left <owner>` — re-attribute the left half (lines before the
  split point) to these owners. Repeatable for joint ownership.
- `--right <owner>` — re-attribute the right half (the split line and
  everything after) to these owners. Repeatable.

Each `<owner>` is `rq-XXXXXXXX` or `<path>:<line>`. Aliases are
rejected. Omitting a side flag preserves the original blob's owners
on that half.

## What happens

1. Find the blob covering `<line>` in `<file>`'s file-tree.
2. Find the byte offset of the line's start within the blob.
3. Split into `left` (bytes before the offset) and `right` (bytes
   from the offset onward).
4. Write both as new blobs.
5. For each meta currently owning the original blob:
   - Remove the original from its `source_blobs`.
   - Add the left half if this meta is in the left's final owner set.
   - Add the right half if this meta is in the right's final owner
     set.
6. For owners listed in `--left` or `--right` but not currently
   owning the original blob: add the appropriate half to their
   `source_blobs`.
7. Replace the original file-tree entry with two new entries (one
   per half). Each entry's `stable_id` is the first listed new owner
   for that side, or the original entry's `stable_id` if that side
   was not re-attributed.

After all updates, `rqm build` runs to refresh the working tree.

## Owner semantics

The left/right "final owner sets" depend on the flags passed:

| `--left` given? | `--right` given? | Left owners | Right owners |
|---|---|---|---|
| No | No | original | original |
| Yes | No | the given set | original |
| No | Yes | original | the given set |
| Yes | Yes | the given left set | the given right set |

"Original" means whichever requirements had the blob in their
`source_blobs` before the split. Joint ownership is preserved.

## Iterative subdivision

To slice a single big blob into several pieces, run `rqm split`
multiple times. After each split, the file-tree has one more entry;
`rqm build` updates the working tree so subsequent line numbers
reflect the current state. Example carving a 6-line blob into three
sections:

```
# Initial: src/x.rs is one blob owned by rq-root, lines 1-6.

$ rqm split src/x.rs:3 --right rq-second
# Now: [lines 1-2 (rq-root), lines 3-6 (rq-second)]

$ rqm split src/x.rs:5 --right rq-third
# Splits the rq-second blob (lines 3-6) at line 5.
# Now: [lines 1-2 (rq-root), lines 3-4 (rq-second), lines 5-6 (rq-third)]
```

## Edge cases

`rqm split` refuses:

- **Split at the start of the blob.** The line is the first line of
  the blob, so the left half would be empty. Use no-op or re-attribute
  the blob via other operations instead.
- **Split at or past the end of the blob.** Symmetric to the above.
- **Unknown owner.** Any owner that doesn't resolve to an existing
  ref. The error names the bad id.
- **Alias owner.** Pass the canonical id (or a file:line cursor that
  resolves to one).
- **Unmanaged path.** The path must already be in `.rqm/managed_paths`.

## Comparison with `rqm mv --split`

`rqm mv --split <src> <dst> --split` *moves* a blob from `<src>` into
the middle of a destination blob, dividing the destination as a side
effect. `rqm split` does only the dividing — no moving, no insertion.
Use:

- `rqm split` to subdivide an existing blob in place, optionally
  re-attributing each half.
- `rqm mv ... --split` to relocate a blob into the middle of another
  blob.

## Limitations

- No multi-cut form (yet). To carve N pieces, run N-1 splits.
- No way to "merge" adjacent file-tree entries back into one blob. If
  you split and then want to undo, you'd need to manually edit or
  re-migrate.
- The left/right split-point semantics are line-aligned. Sub-line
  precision is not supported.
