# `rqm insert`

Add a new blob at a specified position in a managed file. Works for
both source files and markdown requirements files. Auto-creates the
file if it is not yet under management.

```
rqm insert <target>  (--owner <owner>... | --parent <p> | --no-parent)
                     [--kind <K>] [--before]
                     [--from-file <path> | --from-stdin]
```

`rqm insert` operates in one of two modes — exactly one of `--owner`,
`--parent`, or `--no-parent` must be selected.

## Attribute mode

```
rqm insert <target> --owner <owner>... [--before]
```

Attaches the new blob to one or more **existing** requirements. The
blob hash is appended to each owner's `source_blobs`; the new
file-tree entry's `stable_id` is the first owner listed.

- `--owner` — `rq-XXXXXXXX` or `<path>:<line>`. Repeatable for joint
  ownership. Aliases are rejected.

## Create-new mode

```
rqm insert <target> (--parent <id> | --no-parent) [--kind <K>]
```

Generates a fresh `rq-XXXXXXXX` stable_id and creates a new
requirement that owns the new blob as its `text_blob`. The new
requirement's stable_id is printed.

- `--parent <p>` — parent in the DAG. May be `rq-XXXXXXXX` or
  `<path>:<line>`. Aliases are rejected.
- `--no-parent` — create as a DAG root (no parent). Mutually exclusive
  with `--parent`.
- `--kind` — `behavior` (default), `design`, or `pending`.

The generated stable_id is **not** injected into the inserted
content. If you want the canonical `<!-- rq-XXXXXXXX -->` annotation
on a heading line, include it yourself or add it later with
`rqm edit`.

## Common options

- `<target>` — `<path>:<line>`, `<path>:start`, or `<path>:end`.
- `--before` — anchor before the destination blob instead of after.
  Not compatible with `:start` or `:end`.
- Content source (defaults to `$EDITOR` on an empty buffer):
  - `--from-file <path>` — read content from a file.
  - `--from-stdin` — read content from stdin.

After the insertion, `rqm build` runs automatically.

## Creating a new managed file

If `<path>` is not yet in `managed_paths`, `rqm insert` creates an
empty file-tree and adds the path. Only `:start` and `:end` make sense
for the first insertion into a new file (there is no existing `:line`
to anchor against):

```
$ echo "// rq-3d7c8e53 — new helper\nfn helper() {}\n" | \
    rqm insert src/forces/extra.rs:start --owner rq-3d7c8e53 --from-stdin
created managed path: src/forces/extra.rs
new blob: b697d2ac...
  updated meta: rq-3d7c8e53
  wrote src/forces/extra.rs
```

After creation, subsequent inserts into the same file accept `:line`
anchors as usual.

## Inserting into existing files

```
$ rqm insert src/integrator/philox.rs:end --owner rq-3d7c8e53
# editor opens with an empty buffer; user writes content; on save:
new blob: 1ce5cf94...
  updated meta: rq-3d7c8e53
  wrote src/integrator/philox.rs

$ rqm insert src/x.rs:42 --owner rq-aaaaaaaa --before --from-file /tmp/snippet.rs
# inserts before the blob containing line 42
```

## `--owner` via file:line

The owner can be specified by the file:line of any blob owned by the
target requirement. This is useful when you are reading the source and
want to attribute new content to "whatever the surrounding section
belongs to":

```
$ rqm insert src/x.rs:end --owner rqm/forces/spme.md:200
# resolves rqm/forces/spme.md:200 to its canonical owning requirement,
# then attributes the new blob to it
```

Joint ownership: repeat `--owner` for multiple requirements. The new
blob's hash appears in every owner's `source_blobs`.

## Inserting into markdown

`rqm insert` works on markdown files too, in both modes:

**Attribute mode** extends an existing requirement's section with
additional content (the blob is added to the requirement's
`source_blobs`):

```
$ rqm insert rqm/foo.md:end --owner rq-aaaaaaaa --from-file /tmp/scenario.gherkin
```

**Create-new mode** introduces a new requirement at the specified
position with a generated stable_id:

```
$ rqm insert rqm/new_feature.md:start --no-parent --kind behavior < draft.md
created managed path: rqm/new_feature.md
new blob: ab12c4f0...
  created requirement: rq-bf3a7c91
  wrote rqm/new_feature.md
```

For a child requirement inside an existing file:

```
$ rqm insert rqm/feature.md:80 --parent rq-aaaaaaaa --kind behavior < section.md
new blob: ...
  created requirement: rq-XXXXXXXX
```

## Validation

`rqm insert` refuses:

- Empty content (`refusing to insert empty content`).
- No mode selected (`must specify one of --owner, --parent, or --no-parent`).
- Combinations of `--owner` / `--parent` / `--no-parent` (clap rejects
  these at parse time).
- Zero `--owner` arguments while in attribute mode.
- An owner or parent that doesn't resolve (`unknown stable_id: ...`).
- An owner or parent that is a bullet alias (`... is an alias for
  ...; aliases are not accepted here.`).
- An unknown `--kind` (`expected behavior, design, or pending`).
- `--before` with `:start` or `:end`.
- `:line` anchor against a not-yet-managed file (`use :start or :end
  to create the file`).

## Limitations

- No way to insert *between* characters within a line. Granularity is
  whole-line at minimum (`:line` is interpreted at line boundaries).
- The inserted blob carries whatever bytes the user provides — there
  is no automatic `// rq-XXXXXXXX` or `<!-- rq-XXXXXXXX -->` stamp
  injection. The generated stable_id in create-new mode is reported
  but not embedded in the content.
- Create-new mode generates a brand-new top-level or child
  requirement. To split an existing requirement into two, edit the
  original (via `rqm edit`) and then create a new sibling — the tool
  does not yet have a single-command "split a requirement" operation.
