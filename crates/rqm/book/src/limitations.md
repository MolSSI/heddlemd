# Limitations

This is the Phase 1.5 surface. The following are deliberately
out-of-scope for the current iteration; each is noted with what would
be required to address it.

## No version history

`rqm log` shows the *current* state of a requirement, not its
predecessors. Old metas remain in `.rqm/objects/` as immutable history
but are not reachable from refs.

To enable history, add a `predecessor: Option<ObjectHash>` field to
the `Requirement` schema and populate it on every meta rewrite (in
`edit::edit_blob_with` and `migrate::migrate_source`). Then `rqm log`
can walk the predecessor chain. The data is there; the linkage is not.

## No `rqm rename`

Changing a requirement's stable_id is not supported. If a requirement
needs a new id, the workaround is to manually:

1. Choose a new id.
2. Rewrite every source stamp that uses the old id.
3. Update the markdown heading annotation.
4. Re-migrate.

A proper `rqm rename <old-id> <new-id>` would do this atomically and
rewrite the alias entries that point at the old id. Not yet
implemented.

## No `rqm reassign`

`rqm mv` preserves the moved blob's owner. To move a blob *and* change
its owning requirement is two separate concepts that should be
distinct operations. The reassign operation is not yet implemented.

## No object GC

Old metas, blobs, and file-trees accumulate in `.rqm/objects/` forever
after edits. A future `rqm gc` would walk reachable objects (from
refs, aliases, trees) and delete the rest. Until then, the store grows
monotonically.

## Source-blob editing requires a materialized working tree

`rqm edit src/foo.rs:N` resolves `N` against the materialized file on
disk. If the working tree is missing or stale (e.g. you've just cloned
and not yet built), the lookup fails. Run `rqm build` first.

This does not apply to stable_id-form edits, which work regardless of
working-tree state.

## Bullet aliases are transitional

Bullet-level rq stamps in markdown become aliases pointing at their
enclosing heading. This was a deliberate choice to avoid
over-decomposed requirements during migration. The intended endpoint
is to remove bullet stamps from both markdown and source as part of a
post-migration cleanup pass, after which `.rqm/aliases/` becomes empty
and can be removed.

Until then, aliases are second-class — they can't be passed to
`rqm edit` and they appear only in the resolution layer.

## `--split` decouples blob structure from stamps

After `rqm mv ... --split`, the destination file has a blob boundary
that does not correspond to a `// rq-XXXXXXXX` stamp in the source.
The materialized file is byte-perfect and round-trips through
`rqm check` cleanly, but a stamp-based re-parser (e.g. `rqm migrate`
on the materialized file) would not reconstruct the same blob layout
— it would merge the split halves into one blob again.

This is consistent with the model (file-trees are authoritative;
stamps are a Phase-1-era source-side annotation), but it's worth
knowing if you ever blow away `.rqm/` and re-migrate from the working
tree.

## Joint ownership and the "lowest encompassing requirement" rule

Source lines stamped with multiple stable_ids (`// rq-a rq-b`)
materialize as a single blob jointly owned by both requirements. This
is mechanically clean but tensions with the design principle that
each blob should belong to the lowest requirement that fully
encompasses it — when two requirements are co-owners, neither alone
is the lowest.

The current behavior is permissive (joint ownership is accepted and
both owners' `source_blobs` carry the blob hash). A future post-
migration review may either introduce synthetic shared parents or
require refactoring to single ownership; the data needed to make that
decision is collected as the migration proceeds.

## No `rqm fsck` or repair

`rqm check` validates the cross-reference integrity rules during
ordinary check operations, but there is no standalone deep-verify
command. Corrupted stores currently have to be diagnosed manually by
inspecting `.rqm/objects/`. A dedicated `rqm fsck` is straightforward
to add and a reasonable next step once the surface stabilizes.

## File-tree path representation

The `path` field in a file-tree object stores a workspace-relative
path verbatim. There is no normalization (e.g. resolving `.` or `..`
components). Don't pass paths that need normalization; the migration
and CLI layers reject absolute paths and 0-line numbers but otherwise
trust the input.

## CLI rough edges

- No `--dry-run` for `rqm mv` or `rqm edit`. Use `rqm log` to inspect
  before acting.
- No batch migration command. To migrate many files, script `rqm
  migrate` calls in dependency order.
- `rqm log` output is plain text only; no JSON or machine-readable
  format.

None of these block typical use; they are noted so adopters know what
to expect.
