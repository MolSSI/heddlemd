---
name: plan-feature
description: Helps the user plan a feature. Use when the user asks for help designing or planning a feature, or when the user asks for assistance writing, modifying, fleshing out, completing, expanding, or detailing a requirements file.
allowed-tools: Read, Grep, Glob, Bash, AskUserQuestion, Write
---

Do not start implementation. Focus only on planning and documentation. Do not write any code; only requirements under `rqm/` (via the `rqm` CLI).

## The `rqm` tool

The project's requirements live in a content-addressed store at `.rqm/`. The
markdown files under `rqm/` are *materialized output* of that store —
direct edits are rejected as drift by `rqm check`. All changes to managed
requirements go through the `rqm` CLI.

The binary is at `./target/release/rqm` (run `cargo build -p rqm --release`
to produce it). Full CLI reference is in `crates/rqm/book/src/` (run
`mdbook serve crates/rqm/book` to read it as HTML, or read the source markdown
directly).

Commands you'll use most in this skill:

| Command | Purpose |
|---|---|
| `rqm check` | List managed paths and verify the working tree matches `.rqm/`. |
| `rqm view <target>` | Print a requirement's full text, its direct children, and its source-blob locations. **Read this before changing a requirement.** |
| `rqm log <target>` | Compact summary of a requirement (id, kind, ancestry, children, source-blob locations). |
| `rqm edit <target> --from-file <path>` | Replace a text-blob's content with the contents of `<path>`. (Interactive `$EDITOR` mode is also available; this skill uses `--from-file`.) |
| `rqm insert <path>:<anchor> --owner <id> --from-file <path>` | Attribute a new blob to an existing requirement (extends its content). |
| `rqm insert <path>:<anchor> (--parent <id> \| --no-parent) --kind <K> --from-file <path>` | Create a new requirement with a freshly generated stable_id. |
| `rqm rm <target>` | Remove a blob or an entire requirement. |
| `rqm reassign <target> --parent <id>...` | Change a requirement's parents in the DAG. |
| `rqm rename <old> <new>` | Change a requirement's stable_id everywhere. |

Every `<target>` is either a stable_id (`rq-XXXXXXXX`) or a file:line cursor
(`rqm/foo.md:42` resolves to the requirement whose text covers line 42).

**Stable IDs are generated automatically** by `rqm insert` in create mode. Do
not write `rq-XXXXXXXX` annotations into draft content yourself; the tool
prints the new id and you can add the annotation later if convention demands
it.

## Right-size the documentation to the change

The default action of this skill is to **modify an existing requirements file**, not
to create a new one. New requirements files exist to describe features that are large
or clearly distinct from anything already documented. Most invocations of this skill
are tweaks, extensions, or behaviour changes to systems that are already described
somewhere under `rqm/` — those go into the existing canonical file.

Decision tree:

1. **The change tweaks or extends behaviour described by an existing requirements
   file.** Modify that file in place via `rqm edit` / `rqm insert`. Update only
   the sections affected by the change. Match the size of the edit to the size
   of the change.
2. **The change spans multiple existing requirements files.** Edit each affected
   file in place. A new file is appropriate only when the cross-cutting concern
   itself warrants its own canonical reference (rare).
3. **The change introduces a feature that is large and clearly distinct in scope
   from anything described under `rqm/`.** Create a new requirements file via
   `rqm insert <new-path>:start --no-parent ...`.

When in doubt between editing in place and creating a new file, prefer editing
in place. A trivial change should produce a trivial diff.

## Examine the codebase

Read `CLAUDE.md` and any architecture documents it references.

Identify whether the requested change is already covered by, partially covered
by, or adjacent to an existing requirements file. The most common outcome is
that an existing file describes the system the change touches even when no
existing file is named after the change itself; finding that file is part of
this step.

**To read an existing requirement comprehensively**, use:

```
./target/release/rqm view <rq-id-or-file:line>
```

This is the primary mechanism for reading the requirements-to-source mapping.
It shows the requirement's prose, its children's prose, and every source-blob
attributed to it (with `<path>:<line>` locations). Pipe to `less` if the
output is long.

For a quicker overview without the full text content, use `rqm log <target>`.

Once you know the right requirement(s), apply the right-sizing decision tree:

- An existing file already fully describes the requested behaviour → report
  this and stop execution of this skill.
- An existing file partially describes the behaviour → the change goes in that
  file. Edit it in place.
- The change is large and clearly distinct from anything under `rqm/` → draft a
  new requirements file.

## Current-state framing

Requirements files describe the **current desired state** of the code, not
deltas relative to a prior state. The text should be Markovian: a reader who
has never seen prior versions of the code or this document should be able to
read it and understand exactly what the system should look like, without
context about what existed before.

**Avoid** delta language and historical framing:

- "This feature adds…", "This feature delivers…", "This feature ships…"
- "A new field X is added to Y", "Two new variants are appended", "We extend Z with…"
- "The existing W is replaced by…", "We modify…", "We rewrite…"
- Comparisons to prior code versions ("the legacy X", "previously…", "today's behaviour")
- Cross-references that frame other requirements files as superseded or being superseded
- Section titles like "Schema Changes" (just call it "Schema")

**Prefer** flat, declarative descriptions of what the code looks like:

- "X carries field Y", "Type X has variants A, B, C"
- "The system has N components: …"
- "Y is populated from Z at creation time"
- "Field F controls W"
- "Templates use this effect to install …"

This applies to every section: feature description, schema, API, and any
cross-references to other requirements files.

**Migration content** belongs in a requirements file only when implementation
of the feature is expected to deliberately leave certain things unmigrated.
Aim for hard cutovers that leave the codebase in a consistent state, not
phased migrations that span multiple feature increments. If a Migration Notes
section is genuinely warranted, it must explicitly describe the residual
unmigrated state and justify why that state is intentional.

When **modifying an existing requirements file**, rewrite affected sections so
the file continues to describe the current desired state in flat, Markovian
terms — do not append a "the X section is now updated to read…" or "previously
X, now Y" delta. The result should look like a from-scratch description of the
current intent.

## Example requirements file

A complete example requirements file is at
`.claude/skills/plan-feature/bse.md`. Read this file before drafting new
requirements; it demonstrates the section structure (file-level description,
Feature API, Gherkin Scenarios) and the current-state framing.

## Feature scope

Features should be as small and self-contained as reasonably possible. Consider
whether the user's feature idea can be cleanly subdivided into smaller
components. If so, use the AskUserQuestion tool to ask the user if it is
acceptable to subdivide the feature into multiple smaller requirements files
corresponding to each of these natural subdivisions.

## Ask clarifying questions

Use the AskUserQuestion tool to ask the user clarifying questions regarding the
planned feature. Anticipate edge cases, and ask the user how they should be
handled. Batch related questions into a single call. Continue requesting
clarification from the user until every identified edge case has an assigned
handling strategy and the API surface is fully specified.

## Mechanics: modifying an existing managed requirement

This is the most common operation. Identify the target by either its stable_id
(if you know it from `rqm view`) or a `rqm/foo.md:<line>` cursor.

### Revising existing prose

Stage the new text in a temp file, then call `rqm edit`:

```
# 1. Read what's there first
./target/release/rqm view <target>

# 2. Stage the revised text (whole text-blob, not just a diff)
cat > /tmp/revised.md <<'EOF'
## Section Title <!-- rq-XXXXXXXX -->

Revised prose. The full text-blob content goes here, including the heading
line and the section's body.
EOF

# 3. Apply
./target/release/rqm edit <target> --from-file /tmp/revised.md
```

`rqm edit` replaces the entire text-blob; you must stage the full new content,
not a partial diff. The heading line itself (with the `<!-- rq-... -->`
annotation if one is already present) is part of the blob.

### Adding a new sub-requirement (new heading) inside an existing file

```
# Stage the new section's full content
cat > /tmp/new_section.md <<'EOF'
### New Subsection

Body of the new subsection.
EOF

# Insert it after a specific line, as a child of the enclosing heading
./target/release/rqm insert rqm/feature.md:80 \
    --parent <enclosing-heading-id-or-file:line> \
    --kind behavior \
    --from-file /tmp/new_section.md
```

The tool generates and prints the new requirement's stable_id. Note it for
follow-up operations.

### Adding extra content to an existing section (no new requirement)

Use `--owner` instead of `--parent`. The new blob is attributed to the existing
requirement (added to its `source_blobs`); the file-tree gets a new entry at
the chosen position.

```
./target/release/rqm insert rqm/feature.md:120 \
    --owner <existing-requirement-id> \
    --from-file /tmp/extra_paragraph.md
```

This is the right mechanism for adding additional Gherkin scenarios under an
existing scenarios heading, or extending a section's prose without introducing
a new requirement.

### Removing material

```
./target/release/rqm rm rqm/feature.md:<line>     # remove the blob at that line
./target/release/rqm rm <rq-id>                    # remove the whole requirement
```

`rqm rm` on a stable_id refuses if the requirement has source blobs attached
or has children — clear those first. See `crates/rqm/book/src/commands/rm.md`
for details.

### Restructuring the DAG

```
./target/release/rqm reassign <target> --parent <new-parent-id> [...]
./target/release/rqm reassign <target> --no-parent
./target/release/rqm rename <old-id> <new-id>
```

These are rare in routine planning; usually only needed during refactoring
passes.

## Mechanics: creating a new requirements file

When the right-sizing decision tree leads here, the workflow is:

1. Decide the path (see *Markdown file location* below).
2. Stage the full file content in a temp file. The file should begin with a
   `# Feature: ...` heading and include any sections appropriate to the
   feature size (Feature API, Gherkin Scenarios, etc.).
3. Insert as a DAG root:
   ```
   ./target/release/rqm insert rqm/new_feature.md:start \
       --no-parent --kind behavior --from-file /tmp/new_feature.md
   ```
4. The tool creates the managed path, generates a stable_id for the root
   requirement, and prints it.
5. Verify with `./target/release/rqm view <root-id>` and
   `./target/release/rqm check`.

If the feature has substructure (subsections that are themselves first-class
requirements), add each as a child of the root with additional
`rqm insert ... --parent <root-id> ...` calls.

## Markdown file location

This section applies when *Right-size the documentation to the change*
selected the new-file path. For in-place edits to existing files, the file's
existing location is the answer.

Place the new requirements markdown file in the `rqm` directory. The file name
should be brief and descriptive of the feature. The file should begin with a
clear description of the feature.

Features that have been subdivided into smaller components may be organized
into appropriate subdirectories of `rqm`.

## Feature API section

A new requirements file describing functions, classes, or types that are
expected to be accessible to other portions of the code must include a Feature
API section indicating the interface and expected behaviour of those items.

In-place edits to existing files extend whatever API section is already
present (if any). Do not introduce a Feature API section solely to document a
small behaviour tweak; the section structure of an in-place edit should match
the size and shape of the change.

For example, a feature that implements a function in Rust might include:

```
## Feature API

### Functions

- `fetch_basis(element: &str, basis_name: &str) -> Result<PathBuf, BseError>`
  - Validates the element symbol against the known periodic table (elements 1–118).
  - Normalizes `basis_name` to lowercase and `element` to title case before use in file paths and
    API requests.
  - Checks whether a valid cached file already exists at `data/basis/{basis_name}/{element}.json`.
  - If the cache is missing or corrupt, downloads the basis set data for the given element from the
    BSE REST API in QCSchema (JSON) format, creating any missing directories, and overwrites the
    cache file with the fresh response.
  - Returns the `PathBuf` to the cached file on success.

### Types

- `BseError` — error type returned by `fetch_basis`. Must include at minimum:
  - `InvalidElement(String)` — the element symbol does not correspond to a known element (Z = 1–118).
  - `InvalidBasisSetName(String)` — the basis set name is empty or otherwise malformed before any
    API request is made.
  - `ElementNotInBasisSet { element: String, basis_name: String }` — the basis set exists but does
    not include data for this element.
  - `UnknownBasisSet(String)` — the BSE does not recognise the basis set name (HTTP 404).
  - `NetworkError(String)` — a network or HTTP-level failure (unreachable host, timeout, or
    non-200/404 status code).
  - `IoError(String)` — a filesystem operation failed (directory creation, file write, or file read).
  - `InvalidResponse(String)` — the BSE returned a response that could not be parsed as valid JSON.
```

## Gherkin Scenarios section

A new requirements file must include a section for Gherkin Scenarios. These
scenarios should clarify the requirements as well as the proper handling for
any edge cases. Be complete and thorough.

In-place edits to existing requirements files follow the existing file's
structure. If the existing file uses Gherkin scenarios, extend that section
to cover the changed behaviour. If the existing file does not currently use
Gherkin scenarios, do not add one solely because of a small in-place edit.

When the feature is later implemented, these scenarios will be used to
construct unit tests, and they should therefore be designed to be suitable
for this purpose. It should ideally be straightforward and reasonable to
construct a single unit test corresponding to each scenario.

Example scenario block (subset of a real feature):

```gherkin
Feature: Fetch basis set from Basis Set Exchange

  Background:
    Given the BSE base URL is "https://www.basissetexchange.org"

  Scenario: Download a basis set that is not cached
    Given the file "data/basis/sto-3g/H.json" does not exist
    And the BSE API will return a valid QCSchema JSON response for element "H" and basis "sto-3g"
    When fetch_basis("H", "sto-3g") is called
    Then the file "data/basis/sto-3g/H.json" is created
    And the file contains the JSON response returned by the BSE API
    And fetch_basis returns Ok with the path "data/basis/sto-3g/H.json"

  Scenario: Reject an unrecognised element symbol
    When fetch_basis("Xx", "sto-3g") is called
    Then no HTTP request is made to the BSE API
    And fetch_basis returns Err(BseError::InvalidElement("Xx"))
```

Each scenario's `@rq-XXXXXXXX` tag is added automatically when the file is
migrated (or when scenarios are inserted via `rqm insert` with appropriate
target).

## Other sections

Add any other sections that are useful for specifying the feature requirements.
Examples: Data Model, Performance Constraints, Security Considerations,
External API Details, Out of Scope (deliberate non-goals).

Do **not** include sections that describe transient implementation activity
rather than the current desired state — for example: "Source-Code Rename
Targets," "Documentation Changes" (listing other rqm files that need editing),
or "Migration Notes" listing mechanical call-site updates. That information
belongs in the implementation PR's description, not the requirements file.

## Verifying the change

After any modification:

```
./target/release/rqm check
```

should report `ok`. If it reports `diff` or `integrity` violations, the change
introduced inconsistency — investigate before proceeding.

For a sanity-check of the new state, `./target/release/rqm view <root-id>`
prints the full result.

## Files not yet under `.rqm/` management

If `rqm check` does not list your target file as a managed path, the file is
still on the legacy workflow. To bring it under management:

```
./target/release/rqm migrate <path>
```

The file must already have `<!-- rq-XXXXXXXX -->` heading stamps for migration
to succeed. If it does not, run the legacy stamper first:

```
.claude/skills/plan-feature/rqm.sh stamp <path>
```

After migration, use the `rqm` CLI for all further edits to that file.

Some requirements files may still be unmanaged during the transitional phase.
For those files, use the legacy workflow: edit the markdown directly, then
run `rqm.sh stamp` and `rqm.sh index` as documented in
`.claude/skills/plan-feature/ids.md`.
