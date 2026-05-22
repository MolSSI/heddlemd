# `rqm reassign`

Change a requirement's parents in the DAG.

```
rqm reassign <target> (--parent <p>... | --no-parent)
```

- `<target>` — `rq-XXXXXXXX` or `<path>:<line>`. Must resolve to a
  canonical requirement (aliases are rejected).
- `--parent` — new parent(s). May repeat for multi-parent DAGs. Each
  is `rq-XXXXXXXX` or `<path>:<line>`. Aliases are rejected.
- `--no-parent` — set the requirement as a DAG root (empty parents
  list). Mutually exclusive with `--parent`.

The operation replaces the requirement's `parents` list entirely.
There is no "add parent" or "remove parent" mode (yet) — pass the
full desired list.

## Multi-parent DAG semantics

`rqm` supports requirements that have multiple parents. A diamond
structure (B and C both descend from A; D descends from both B and C)
is well-formed:

```
$ rqm reassign rq-d --parent rq-b --parent rq-c
reassigned rq-d
  old parents: (DAG root)
  new parent: rq-b
  new parent: rq-c
```

Multi-parent does *not* affect materialization. Each requirement
appears in exactly one position in one markdown file (its file-tree
entry); parents are DAG metadata used for ancestry queries
(`rqm log`) and integrity checks, not for blob placement.

## Validation

`rqm reassign` refuses on any condition that would corrupt the DAG:

- **Unknown target or parent.** `unknown stable_id: rq-...`.
- **Alias.** `... is an alias for ...; pass the canonical id.` —
  aliases can't be the target or a parent.
- **Self-loop.** `a requirement cannot be its own parent (rq-...)`.
- **Cycle.** `setting parent rq-P on rq-T would create a cycle
  (rq-T is an ancestor of rq-P)`. The check walks the transitive
  ancestry of every proposed parent and refuses if the target appears
  anywhere.
- **No mode specified.** `must specify --parent <id>... or
  --no-parent`.

## Examples

**Move a requirement under a different parent:**

```
$ rqm reassign rq-cccccccc --parent rq-bbbbbbbb
reassigned rq-cccccccc
  old parent: rq-aaaaaaaa
  new parent: rq-bbbbbbbb
```

**Promote a requirement to a DAG root:**

```
$ rqm reassign rqm/foo.md:42 --no-parent
reassigned rq-XXXXXXXX
  old parent: rq-aaaaaaaa
  new parents: (DAG root)
```

**Give a requirement multiple parents (cross-cutting concern):**

```
$ rqm reassign rq-shared --parent rq-feature-a --parent rq-feature-b
```

This is useful for material that legitimately belongs under more than
one section — e.g. a shared utility documented at the intersection of
two features.

## Limitations

- No incremental `--add-parent` / `--remove-parent` modes. Pass the
  full desired list each time.
- Reassigning a requirement updates only its own meta. Its
  *descendants* are unaffected (their parent references still point
  at the right place).
- Reassigning across an alias boundary is rejected — the user must
  always pass canonical ids. Aliases are a migration-era construct
  and shouldn't participate in DAG restructuring.
