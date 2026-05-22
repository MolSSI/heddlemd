# `rqm init`

Create an empty `.rqm/` directory.

```
rqm init
```

Creates the four subdirectories (`objects/`, `refs/`, `aliases/`,
`trees/`) and an empty `managed_paths` file. If `.rqm/` already exists
and has the expected structure, init is idempotent.

## When to use

Once per repository, before any `rqm migrate` calls. Typically run
during initial setup.

## Example

```
$ rqm init
initialized .rqm
$ ls .rqm
aliases  managed_paths  objects  refs  trees
```

After `init`, `rqm check` will pass trivially (nothing is managed
yet). Add files to `.rqm/` with `rqm migrate`.
