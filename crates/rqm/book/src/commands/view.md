# `rqm view`

Dump a requirement and its full associated content: the requirement's
prose, every source blob attached to it, and every direct child
requirement's prose. Designed as the primary mechanism for reading
the requirements-to-source mapping for a given requirement, suitable
for both human inspection and LLM consumption.

```
rqm view <target>
```

`<target>` follows the standard convention (see
[Addressing](../addressing.md)):

- `rq-XXXXXXXX` — the canonical id of the requirement to view.
- `<path>:<line>` in a markdown file — resolves to the requirement
  whose text_blob covers that line.
- `<path>:<line>` in a source file — resolves to the **owning
  requirement** of the blob at that line.
- A bullet alias — follows through to its canonical.

## Output structure

Requirements information first (the requirement itself, then its
direct children), then the source blobs attributed to it:

```
=== Requirement rq-XXXXXXXX ===
kind:       behavior
file:       rqm/foo.md:42
text_blob:  <64-hex hash>
meta:       <64-hex hash>
parents:    rq-AAAAAAAA, rq-BBBBBBBB    (or "(DAG root)")
aliases:    rq-CCCCCCCC                  (only when aliases exist)

<full text_blob bytes>


=== Children (N) ===

--- rq-YYYYYYYY (behavior) ---
file:       rqm/foo.md:80
text_blob:  <hash>

<full child text_blob bytes>

--- rq-ZZZZZZZZ (design) ---
file:       rqm/foo.md:120
text_blob:  <hash>

<full child text_blob bytes>


=== Source blobs (N) ===

--- src/foo.rs:1 ---
blob: <64-hex hash>

<full blob bytes>

--- tests/foo.rs:5 ---
blob: <64-hex hash>

<full blob bytes>
```

Sections are present even when empty (e.g. `=== Source blobs (0) ===`
followed by `(none)`), so the output shape is predictable.

## Source-file targets

A common pattern when reading code: you're looking at a function and
want to know which requirement it implements and what the requirement
*says*. Point `rqm view` at any line of that function:

```
$ rqm view src/integrator/csvr.rs:42
=== Requirement rq-7a124d43 ===
kind:       behavior
file:       rqm/integration/csvr.md:5
...
```

The output shows the full requirement context — what the requirement
says, every source blob attributed to it (including the one you
pointed at, plus any others in other files), and the requirement's
direct children.

## Children, not descendants

`rqm view` shows *direct* children only — one level down. Multi-parent
DAGs and large requirement trees mean recursive expansion would
explode quickly. To drill into a child, run `rqm view <child-id>`
separately.

## Aliases in the header

When bullet aliases point at the requirement (`<!-- rq-XXXXXXXX -->`
on a bullet item whose canonical is the heading above), they are
listed under `aliases:`. The view output describes the canonical
itself — the alias is just a label the user might have used to find
this requirement.

## Empty / unmaterialized requirements

A "ghosted" requirement (one whose text_blob is no longer in any
file-tree, e.g. after `rqm rm --force`) renders as `file: (not
materialized)`. The text_blob bytes are still shown — the requirement
exists in `.rqm/objects/`, just not in any managed file.

## What this is not

- Not interactive — it's a one-shot dump.
- Not paginated — pipe to `less` if the output is long.
- Not formatted as JSON or anything machine-readable. The output is
  designed to be informative when displayed verbatim (e.g. in an LLM
  context window). A machine-readable variant (`--json`) is a future
  extension if needed.
