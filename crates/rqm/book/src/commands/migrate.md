# `rqm migrate`

Bootstrap an existing file into `.rqm/`. After migration the file
becomes a *managed path*: its bytes are owned by `.rqm/`, and any drift
from the materialized output is rejected by `rqm check`.

```
rqm migrate <path>
```

`<path>` is a workspace-relative path. Whether the file is treated as a
requirements file or a source file is determined by the extension:

- `.md` — parsed as a markdown requirements file. Headings with
  `<!-- rq-XXXXXXXX -->` annotations become first-class requirements;
  Gherkin scenarios with `@rq-XXXXXXXX` tags become leaf
  requirements; bullet-level rq stamps are recorded as aliases for
  their enclosing heading.
- anything else — parsed as a source file. Lines containing
  `rq-XXXXXXXX` tokens mark blob boundaries; the blob from one stamp
  to the next becomes a source blob owned by that stamp's
  requirement.

## Order matters

Source files can only be migrated *after* the requirements files they
reference. The tool fails fast with a clear message when a stamp does
not resolve:

```
$ rqm migrate src/integrator/csvr.rs
rqm: error: src/integrator/csvr.rs: stamp rq-1f87880c has no corresponding
            requirement in .rqm/refs/ or .rqm/aliases/ (migrate the
            requirements file first)
```

When a source file references stamps from multiple requirements files,
all of those must be migrated first. This is the **migration
neighborhood** concept — heavily-shared source files form clusters of
requirements that have to migrate together.

## Pilot-1-era constraints

- The markdown file must start with a heading on line 1. Prelude
  content before the first heading is not supported.
- Bullet-level stamps inside markdown bullets are recorded as aliases
  rather than promoted to first-class requirements. See
  [Addressing](../addressing.md).
- Multi-stamp source lines (`// rq-a rq-b`) define a single blob with
  joint ownership; the blob hash appears in *both* metas' `source_blobs`.

## Example

```
$ rqm migrate rqm/analysis/rdf.md
migrated markdown rqm/analysis/rdf.md
$ rqm migrate src/analysis/rdf.rs
migrated source src/analysis/rdf.rs
$ rqm migrate tests/rdf.rs
migrated source tests/rdf.rs
$ rqm check
ok (3 paths)
```
