# `rqm rm`

Remove a blob or a requirement, depending on the target form.

```
rqm rm <target> [--force]
```

- `<target>` — either `<path>:<line>` (remove a blob) or
  `rq-XXXXXXXX` (remove a requirement). See the two sections below.
- `--force` — applies only to the blob form; allows removing a blob
  that is a requirement's `text_blob`.

## Blob form: `rqm rm <path>:<line>`

1. Resolves `<target>` to the file-tree entry covering that line.
2. Removes the entry from the file-tree.
3. If the blob is no longer referenced by *any* file-tree entry under
   any managed path, removes it from every meta's `source_blobs`. This
   preserves the integrity rule that `source_blobs` entries must be
   materialized somewhere.
4. Runs `rqm build` to update the working tree.

## Joint ownership

When the removed blob was owned by multiple requirements (joint
ownership via a multi-stamp source line), *all* their metas are
updated in one operation. The output lists each updated meta.

## text_blob protection

Each requirement's `text_blob` field holds the hash of the blob
carrying that requirement's prose. Removing such a blob from its
file-tree means the requirement still exists in `.rqm/objects/` but
its content is no longer materialized anywhere — the requirement is
"ghosted". This is almost certainly not what the user wants, so by
default `rqm rm` refuses:

```
$ rqm rm rqm/foo.md:1
rqm: error: rqm/foo.md:1 is the text_blob of rq-aaaaaaaa; removing it
            would leave the requirement without materialized prose.
            Pass --force to override...
```

When `--force` is used, the operation succeeds and the requirement
becomes ghosted. The output flags this:

```
$ rqm rm rqm/foo.md:1 --force
removed blob: 3666b3be...
  warning: rq-aaaaaaaa is now ghosted (text_blob no longer materialized)
  wrote rqm/foo.md
```

To fully delete a ghosted requirement, edit its meta directly (no
`rqm delete-requirement` command exists yet — see
[Limitations](../limitations.md)).

## Duplicate-reference handling

If the same blob hash is referenced from multiple file-tree entries
(e.g. identical content stored in two files), `rqm rm` removes only
the targeted entry. The blob remains in `source_blobs` because it is
still materialized through the remaining entries:

```
$ rqm rm src/a.rs:1        # src/b.rs still contains the same blob
removed blob: 1ce5cf94...
  wrote src/a.rs            # source_blobs untouched
```

## Requirement form: `rqm rm rq-XXXXXXXX`

Removes the requirement entirely. The operation is safe-by-default —
it refuses on any condition that would create dangling references,
and forces the user to clear those dependencies explicitly:

```
$ rqm rm rq-4d1082c4
rqm: error: rq-4d1082c4 has 2 source blob(s) attached; remove them
            first (e.g. via `rqm rm <path>:<line>`).
```

When all conditions are met, the tool:

1. Removes every file-tree entry whose `stable_id` matches the
   requirement (typically just the requirement's text_blob entry in
   its markdown file).
2. Auto-deletes every alias pointing at this canonical (bullet
   aliases become meaningless without their canonical).
3. Deletes the ref.
4. Runs `rqm build`. The materialized markdown file shrinks by the
   removed entry.

**Refused conditions** (each must be cleared first):

- `... has N source blob(s) attached` — use the blob form of `rqm
  rm` to remove each source blob first.
- `... has N child requirement(s)` — remove or reassign each child
  first. (There is no `rqm reassign` yet; for now, edit the child's
  meta directly or delete the child first.)
- `... is an alias for ...` — aliases can't be removed directly;
  they are auto-deleted when their canonical is removed.

`--force` does **not** override these checks — the safety is
intentional. To force-delete, manually clear the dependencies and
then re-run `rqm rm <id>`.

The requirement's meta and text_blob remain in `.rqm/objects/` as
immutable history. A future `rqm gc` will prune unreachable
objects.
