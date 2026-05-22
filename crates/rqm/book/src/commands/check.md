# `rqm check`

Verify that the working tree matches what `rqm build` would produce
from `.rqm/`.

```
rqm check
```

Returns exit 0 if every managed path is byte-identical to its
materialization. Returns non-zero otherwise, with one line per
discrepancy.

## What is checked

For every path in `.rqm/managed_paths`:

- The path exists on disk → otherwise reported as `missing`.
- The bytes match the materialization → otherwise reported as `diff`.

Plus the cross-reference integrity rules:

- Every alias points at a stable_id that has a real ref.
- Every ref's name matches the meta's `stable_id`.
- Every blob referenced by a meta (as `text_blob` or in
  `source_blobs`) exists in `.rqm/objects/`.
- Every blob referenced by a file-tree exists.
- Every parent stable_id referenced by a meta has a ref.
- Every managed path has a tree ref; every tree ref's file-tree
  has a `path` matching the managed path.

Violations are listed under `integrity:` in the output.

## Output

```
$ rqm check
ok (18 paths)
```

```
$ rqm check
diff: rqm/analysis/rdf.md
$ echo $?
1
```

## When to use

- In CI on every change — this is the lockdown check. A passing
  `rqm check` is the contract that the working tree and `.rqm/` agree.
- Before committing, to catch unintended hand-edits.
- After running a tool that might have mutated `.rqm/` directly, to
  confirm the working tree is in sync (then run `rqm build` to
  reconcile).

## Recovering from a failed check

If `rqm check` reports `diff` on a path, you have two options:

1. **The working-tree change was intended.** Run `rqm migrate <path>`
   to re-import the file into `.rqm/`. The new content becomes the
   canonical version.
2. **The working-tree change was accidental.** Run `rqm build` to
   overwrite the working tree with `.rqm/`'s contents.

The right choice depends on which side is currently correct.
