# Commands

`rqm` is invoked as a single binary with subcommands. Each subcommand
has its own page; this index gives the one-line summary and the global
options.

## Global options

```
--rqm-dir <PATH>     Path to the .rqm/ directory. Default: ./.rqm
--root <PATH>        Workspace root for resolving relative paths.
                     Default: current working directory.
```

Both options can be passed before any subcommand.

## Subcommand index

| Command | Purpose |
|---|---|
| [`init`](./commands/init.md) | Create an empty `.rqm/` directory |
| [`migrate`](./commands/migrate.md) | Bootstrap an existing file into `.rqm/` |
| [`build`](./commands/build.md) | Materialize all managed paths from `.rqm/` |
| [`check`](./commands/check.md) | Verify the working tree matches `.rqm/` |
| [`edit`](./commands/edit.md) | Open a blob in `$EDITOR`; apply on save |
| [`log`](./commands/log.md) | Show a requirement's current state |
| [`mv`](./commands/mv.md) | Relocate a blob within or across files |
| [`insert`](./commands/insert.md) | Add a new blob at a specified position |
| [`rm`](./commands/rm.md) | Remove a blob or a requirement |
| [`reassign`](./commands/reassign.md) | Change a requirement's parents in the DAG |

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success |
| non-zero | error (the message is printed to stderr) |

For `rqm check`, exit `0` means the working tree matches `.rqm/`; any
diff, missing file, or integrity violation produces a non-zero exit
with a per-line description. CI is expected to treat non-zero from
`rqm check` as a failure.
