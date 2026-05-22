# Addressing

Every command that operates on a requirement or blob takes a *target*
argument. Two forms are accepted, and the same convention applies
across all such commands.

## The two forms

### Stable ID

A literal `rq-XXXXXXXX` (the prefix `rq-` followed by 8 lowercase hex
characters):

```
rqm edit rq-4d1082c4
rqm log rq-4d1082c4
```

This is the canonical identifier for a requirement. It does not change
across edits, so it is stable in scripts and durable references.

### File:line cursor

A `<path>:<line>` pair, where `path` is a managed path and `line` is a
1-based line number:

```
rqm edit rqm/analysis/rdf.md:42
rqm log src/io/config.rs:412
rqm mv src/io/config.rs:412 src/forces/spme.rs:200
```

The tool resolves this to whichever blob covers that line in the
materialized file. For markdown files, that blob is a requirement's
text-blob; for source files, it is a source-blob. Either way, the
target is unambiguous.

Use file:line when you are reading the file and have found the line
you want to act on. Use the stable ID when you already know it (e.g.
from a previous `rqm log`) or when scripting.

## Resolution semantics

How a target is resolved depends on the command's intent:

| Command | Aliases | Why |
|---|---|---|
| `rqm edit` | rejected | editing through an alias would silently mutate a different requirement's text-blob; the tool requires the canonical id |
| `rqm log` | followed | inspection is read-only; resolving the alias and showing the canonical's information is what the user wants |
| `rqm mv` | n/a | operates on file-tree positions, addressed by file:line only |

When resolution fails, the tool surfaces the reason concretely:

```
rqm: error: rq-cccccccc is an alias for rq-bbbbbbbb; edit the canonical
            id directly. (aliases are migration-only and don't have
            their own metas)
```

## Aliases

Aliases exist because the migration step (`rqm migrate`) encountered
markdown files where bullet items carried their own `<!-- rq-... -->`
stamps. Promoting every such bullet to a first-class requirement would
produce many tiny requirements and source blobs — a metric (3)
violation under the data model's health metrics. Instead, the
migration records each bullet-level stamp in `.rqm/aliases/` pointing
at its enclosing heading's stable_id. The heading is the *canonical*
requirement; the bullet stamp is an alias.

Source files that referenced a bullet stamp resolve through the alias
to attribute their blob to the heading. This preserves the existing
source-side stamping without inflating the requirement count.

Aliases are a transitional construct. A post-migration cleanup pass
will remove bullet stamps from both markdown and source, after which
`.rqm/aliases/` becomes empty and can be removed.

## File:line in markdown vs source

The same `file:line` form works for both, but means different things:

- In a markdown file, the resolved blob is a requirement's
  **text-blob**. Editing it changes the requirement's text. (Useful
  for ergonomic edits without knowing the stable_id.)
- In a source file, the resolved blob is a **source-blob**. Editing
  it changes the source code of that section.

`rqm log` accepts both forms and produces the same output regardless
— it shows the canonical owner's full state.
