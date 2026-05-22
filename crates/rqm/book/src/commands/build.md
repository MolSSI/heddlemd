# `rqm build`

Materialize all managed paths from `.rqm/`. Overwrites existing files
in place.

```
rqm build
```

For each entry in `.rqm/managed_paths`, `build`:

1. Looks up the file-tree hash in `.rqm/trees/<path>`.
2. Reads the file-tree from `.rqm/objects/`.
3. Concatenates the blobs referenced by `entries[]` in order.
4. Writes the result to `<root>/<path>`, creating parent directories
   if needed.

Unchanged files are left alone (the write only happens when the bytes
differ from what's on disk).

## Output

```
$ rqm build
wrote rqm/analysis/rdf.md
wrote src/analysis/rdf.rs
...
```

When a path is already up to date, it appears as part of the
"unchanged" count rather than being individually listed.

## When to use

- After `rqm edit` or `rqm mv` — these commands auto-run `build` for
  you, but a manual rebuild is occasionally useful (e.g., after
  manually editing JSON in `.rqm/objects/`).
- After cloning a repository, if you only have `.rqm/` and need to
  regenerate the working tree.
- When recovering from a partial state — `rqm build` is idempotent
  and produces the canonical materialization.

## Caveats

`build` overwrites the working tree without preserving local
modifications. If you have hand-edited a managed file and run
`build`, the hand edits are lost. Use `rqm check` first to detect
drift, then `rqm edit` or `rqm migrate` to bring changes into `.rqm/`
properly.
