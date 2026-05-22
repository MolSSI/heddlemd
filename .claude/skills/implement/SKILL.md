---
name: implement
description: >-
  Helps the user implement a feature.
  TRIGGER when: the user asks to add, build, create, make, or implement something (new feature, new
  system, new command, new API, new behaviour); the user asks to implement functionality described
  in a requirements file.
  SKIP: the user is asking to fix a bug in existing code; the user is asking to refactor or rename
  existing code without changing behaviour; the user is explicitly asking only to plan or design
  (those go to plan-feature instead).
allowed-tools: Read, Grep, Glob, Bash, AskUserQuestion, Write
---

Check for a requirements file under `rqm/` that corresponds to the requested
feature. If no such file is found, instead execute the /plan-feature skill.

Ensure that the implemented code satisfies all Gherkin scenarios in the
requirements file. Create at least one test corresponding to every Gherkin
scenario.

## The `rqm` tool

Source and test files implementing managed requirements are themselves managed
by `.rqm/` — direct edits to those files are rejected by `rqm check`. All
changes to managed source files go through the `rqm` CLI at
`./target/release/rqm` (build with `cargo build -p rqm --release`).

Full CLI reference: `crates/rqm/book/src/`. Commands you'll use in this skill:

| Command | Purpose |
|---|---|
| `rqm check` | List managed paths and verify the working tree matches `.rqm/`. Run before and after changes. |
| `rqm view <target>` | Print a requirement's full text, its children's text, and every source-blob already attached to it. **Read this before writing code that implements the requirement.** |
| `rqm log <target>` | Compact summary (kind, ancestry, children, source-blob locations). Use when you encounter an `rq-XXXXXXXX` stamp in source and want to look up what it refers to. |
| `rqm insert <path>:<anchor> --owner <id> --from-file <path>` | Add a new source blob to a managed file, attributed to one or more existing requirements. Creates the file if not yet managed. |
| `rqm edit <target> --from-file <path>` | Replace an existing source blob's bytes with the contents of a file. |
| `rqm rm <target>` | Remove a source blob (or a whole requirement, but that's for the planning skill). |

Every `<target>` is `rq-XXXXXXXX` or `<path>:<line>`.

**Stable IDs are never invented.** When you reference a requirement, use the
id printed by `rqm view`/`rqm log` or pass a `<path>:<line>` cursor. If you
need to find the canonical id for a bullet alias, `rqm log <alias-id>`
resolves and prints the canonical.

## Workflow

### 1. Read the requirement(s)

Before writing any code, read the relevant requirement in full:

```
./target/release/rqm view <root-id-or-file:line>
```

This prints:
- The requirement's prose (heading + body).
- Every direct child requirement, including any Gherkin scenarios.
- Every source blob already attributed to the requirement, with file
  locations and full content.

For implementing a feature, this is the primary read mechanism. Pipe to
`less` if the output is long.

### 2. Identify the scenarios that need tests

The Gherkin scenarios appear as children of (or attached to) the requirement.
Each scenario has its own stable_id (the `@rq-XXXXXXXX` tag).

`rqm log <scenario-id>` shows whether the scenario already has a test
attached (its `source_blobs` would point at a test function). Scenarios with
empty `source_blobs` need new tests.

### 3. Write the implementation

For each piece of code that implements a requirement:

**Adding a new function to a managed source file:**

```
# Stage the function body
cat > /tmp/new_function.rs <<'EOF'
// rq-XXXXXXXX
pub fn the_function(...) -> ... {
    ...
}
EOF

# Insert into the managed source file
./target/release/rqm insert src/path/to/file.rs:end \
    --owner <requirement-id> \
    --from-file /tmp/new_function.rs
```

- `<anchor>` choices: `:end` to append, `:start` to prepend,
  `:<line>` (default = "after that line"), `:<line> --before`.
- `--owner` may repeat for joint ownership (one piece of code implementing
  multiple requirements).
- The `// rq-XXXXXXXX` comment in the body is human-readable annotation; the
  actual attribution is the file-tree entry's stable_id set by `--owner`.

**Modifying an existing source blob:**

```
# Read what's there
./target/release/rqm view src/path/to/file.rs:<line>

# Stage the replacement
cat > /tmp/replacement.rs <<'EOF'
// rq-XXXXXXXX
pub fn updated_function(...) -> ... {
    ...new body...
}
EOF

# Apply
./target/release/rqm edit src/path/to/file.rs:<line> --from-file /tmp/replacement.rs
```

`rqm edit` replaces the entire blob; stage the full new content.

**Creating a brand-new managed source file:**

```
./target/release/rqm insert src/new_module.rs:start \
    --owner <requirement-id> \
    --from-file /tmp/file_contents.rs
```

The new file is created, added to `managed_paths`, and seeded with the given
content. Subsequent edits go through `rqm`.

### 4. Write tests

Each Gherkin scenario becomes a test, attributed to the scenario's
stable_id. Tests typically live in `tests/<area>.rs`.

```
cat > /tmp/test_function.rs <<'EOF'
// rq-<scenario-id>
#[test]
fn scenario_name() {
    // ... test body driven by the Gherkin scenario ...
}
EOF

./target/release/rqm insert tests/<area>.rs:end \
    --owner <scenario-id> \
    --from-file /tmp/test_function.rs
```

If the test file is not yet managed, `rqm insert` will create it and add it
to `managed_paths`.

### 5. Joint ownership

When a single piece of code implements multiple requirements, pass multiple
`--owner` flags:

```
./target/release/rqm insert src/shared.rs:end \
    --owner rq-aaaaaaaa --owner rq-bbbbbbbb \
    --from-file /tmp/shared.rs
```

The blob's hash appears in every owner's `source_blobs`. Aliases are
rejected; pass canonical ids.

### 6. Verify

After the implementation is in place:

```
./target/release/rqm check         # round-trip integrity
cargo test                          # actual test suite passes
```

For a comprehensive view of how a requirement is now implemented:

```
./target/release/rqm view <requirement-id>
```

The output's "Source blobs" section now lists each piece of code attributed
to the requirement.

## Looking up an unfamiliar stamp

If you encounter `// rq-XXXXXXXX` in source and need to know what it
references:

```
./target/release/rqm log rq-XXXXXXXX
```

Prints the requirement's id, kind, ancestry chain, and source-blob locations.
For the full prose, follow with:

```
./target/release/rqm view rq-XXXXXXXX
```

Read the referenced requirement in full before proposing any change to code
attributed to it. If the proposed change would alter behaviour covered by
that requirement, the requirements file itself may need to be updated first
— use the `/plan-feature` skill for that.

## Don't fabricate stable IDs

Every `rq-XXXXXXXX` token you reference must exist in `.rqm/refs/` (or
`.rqm/aliases/`). The CLI rejects unknown ids with a clear error
(`unknown stable_id: ...`). If you find yourself wanting to type an id that
hasn't been generated, you're in the wrong skill — go to `/plan-feature` to
create the requirement first.

## Files not yet under `.rqm/` management

If `rqm check` does not list a source or test file as a managed path, follow
the legacy workflow for that file:

1. Edit the file directly with your editor.
2. Annotate functions/tests with `// rq-XXXXXXXX` comments referencing the
   relevant requirement IDs.
3. Run `.claude/skills/plan-feature/rqm.sh index` to refresh the legacy
   registry.

When the file is later migrated to `.rqm/` (via `rqm migrate <path>`), all
future edits go through the `rqm` CLI.

## Comment-style fallbacks for legacy / unmanaged code

For files still on the legacy workflow, the comment style is:

- Rust / C / C++ / JavaScript / TypeScript: `// rq-XXXXXXXX`
- Python / Shell: `# rq-XXXXXXXX`
- SQL: `-- rq-XXXXXXXX`

A single source line may carry multiple IDs if it implements multiple
requirement entities:

```rust
// rq-9b4d2f1a rq-3a7f1c2e
pub fn fetch_basis(...) { ... }
```

Under `.rqm/`-managed files, the stamp text in the blob is informational —
the file-tree entry's stable_id (set by `--owner`) is the authoritative
attribution. The eventual plan is to remove stamps from managed source
entirely; for now they remain for human readability.
