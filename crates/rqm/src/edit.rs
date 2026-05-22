//! Authoring helpers: edit a blob's bytes through `$EDITOR`.
//!
//! A blob can be either a requirement's text-blob (the prose in a
//! markdown file) or one of a requirement's source-blobs (a chunk of a
//! source file). The core operation is the same in both cases — read
//! bytes, mutate, write a new blob, update every meta and file-tree
//! that referenced the old blob hash — so the implementation is unified
//! around blob hashes; the higher-level helpers just resolve the user's
//! target to a hash first.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::object::{Blob, ObjectHash, StableId};
use crate::store::Store;

/// What to edit. Users can address a blob either by the owning
/// requirement's canonical stable_id (text-blob) or by file:line (any
/// blob, either a text-blob in a markdown file or a source-blob in a
/// source file).
#[derive(Debug, Clone)]
pub enum EditTarget {
    Id(StableId),
    FileLine { path: PathBuf, line: usize },
}

impl EditTarget {
    /// Parse a CLI argument. `rq-XXXXXXXX` is a stable_id;
    /// `<path>:<line>` is a file:line.
    pub fn parse(s: &str) -> Result<Self> {
        if s.starts_with("rq-")
            && s.len() == 11
            && s[3..]
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        {
            return Ok(EditTarget::Id(StableId::new(s)));
        }
        if let Some((path, line)) = s.rsplit_once(':') {
            let line: usize = line.parse().with_context(|| {
                format!("expected file:line or rq-XXXXXXXX, got {s:?}")
            })?;
            if line == 0 {
                bail!("line numbers are 1-based");
            }
            if path.is_empty() {
                bail!("empty path in {s:?}");
            }
            return Ok(EditTarget::FileLine {
                path: PathBuf::from(path),
                line,
            });
        }
        bail!("expected rq-XXXXXXXX or file:line, got {s:?}")
    }
}

#[derive(Debug)]
pub enum EditOutcome {
    Unchanged,
    Canceled,
    Changed {
        new_blob: ObjectHash,
        /// Every meta whose text_blob or source_blobs was updated, by
        /// stable_id (canonical). Joint-ownership source blobs cause
        /// multiple entries.
        metas_updated: Vec<StableId>,
        /// Every managed path whose file-tree was rewritten.
        paths_updated: Vec<PathBuf>,
    },
}

/// User-facing edit step: returns either bytes to apply, or a cancel
/// signal. The CLI provides an editor-based implementation; tests
/// provide a programmatic one.
pub enum EditFn {
    Apply(Vec<u8>),
    Cancel,
}

/// Resolve an [`EditTarget`] to the blob hash it identifies. Stable_id
/// targets resolve to the requirement's text_blob; file:line targets
/// resolve to whichever blob in that file's file-tree covers the line.
///
/// Aliases are rejected (would silently edit more than the user
/// intended). For permissive resolution (alias-following), see
/// [`target_to_canonical_id`].
pub fn target_to_blob(store: &Store, target: &EditTarget) -> Result<ObjectHash> {
    match target {
        EditTarget::Id(id) => {
            let canonical = store
                .resolve(id)?
                .ok_or_else(|| anyhow::anyhow!("unknown stable_id: {id}"))?;
            if &canonical != id {
                bail!(
                    "{id} is an alias for {canonical}; edit the canonical id directly. \
                     (aliases are migration-only and don't have their own metas)"
                );
            }
            let meta_hash = store
                .ref_get(&canonical)?
                .ok_or_else(|| anyhow::anyhow!("ref missing for {canonical}"))?;
            let meta = store.read_requirement(&meta_hash)?;
            Ok(meta.text_blob)
        }
        EditTarget::FileLine { path, line } => Ok(find_entry_at(store, path, *line)?.1),
    }
}

/// Resolve an [`EditTarget`] to the canonical stable_id of the
/// requirement it identifies. Aliases follow through to their canonical
/// (unlike [`target_to_blob`]).
pub fn target_to_canonical_id(store: &Store, target: &EditTarget) -> Result<StableId> {
    match target {
        EditTarget::Id(id) => store
            .resolve(id)?
            .ok_or_else(|| anyhow::anyhow!("unknown stable_id: {id}")),
        EditTarget::FileLine { path, line } => {
            Ok(find_entry_at(store, path, *line)?.0)
        }
    }
}

/// Like [`target_to_canonical_id`], but refuses alias `Id` targets.
/// `FileLine` targets always resolve through a file-tree entry, which
/// already carries canonical stable_ids, so they never need alias
/// rejection.
pub fn target_to_canonical_id_strict(
    store: &Store,
    target: &EditTarget,
) -> Result<StableId> {
    match target {
        EditTarget::Id(id) => {
            let canonical = store
                .resolve(id)?
                .ok_or_else(|| anyhow::anyhow!("unknown stable_id: {id}"))?;
            if &canonical != id {
                bail!(
                    "{id} is an alias for {canonical}; aliases are not accepted here. \
                     Use the canonical id or a file:line cursor."
                );
            }
            Ok(canonical)
        }
        EditTarget::FileLine { path, line } => {
            Ok(find_entry_at(store, path, *line)?.0)
        }
    }
}

/// Find the file-tree entry whose blob covers `line` in `path`.
/// Returns `(entry.stable_id, entry.blob)`.
pub fn find_entry_at(
    store: &Store,
    path: &Path,
    line: usize,
) -> Result<(StableId, ObjectHash)> {
    let tree_hash = store.tree_get(path)?.ok_or_else(|| {
        anyhow::anyhow!(
            "{}: not a managed path in .rqm/ (use `rqm migrate` first)",
            path.display()
        )
    })?;
    let tree = store.read_file_tree(&tree_hash)?;
    if tree.entries.is_empty() {
        bail!("{} has no blobs", path.display());
    }

    // Concatenate blob bytes so we can find a stable byte offset for the
    // requested line, then locate which blob covers that offset.
    let mut concat = Vec::new();
    let mut blob_starts = Vec::with_capacity(tree.entries.len() + 1);
    for entry in &tree.entries {
        blob_starts.push(concat.len());
        let blob = store.read_blob(&entry.blob)?;
        concat.extend_from_slice(&blob.0);
    }
    blob_starts.push(concat.len());

    let target_offset = if line == 1 {
        0
    } else {
        let mut newlines_seen = 0usize;
        let mut found = None;
        for (i, &b) in concat.iter().enumerate() {
            if b == b'\n' {
                newlines_seen += 1;
                if newlines_seen == line - 1 {
                    found = Some(i + 1);
                    break;
                }
            }
        }
        found.ok_or_else(|| {
            anyhow::anyhow!(
                "{}: line {} out of range (file has {} line(s))",
                path.display(),
                line,
                newlines_seen + 1
            )
        })?
    };

    for (i, entry) in tree.entries.iter().enumerate() {
        if target_offset >= blob_starts[i] && target_offset < blob_starts[i + 1] {
            return Ok((entry.stable_id.clone(), entry.blob));
        }
    }
    // target_offset == EOF — attribute to the last blob.
    let last = tree.entries.last().unwrap();
    Ok((last.stable_id.clone(), last.blob))
}

/// Apply `new_bytes` as the replacement contents for `blob_hash`. The
/// non-interactive counterpart to [`edit_blob_interactive`]. Refuses
/// empty content (use `rqm rm` instead).
pub fn edit_blob_from_bytes(
    store: &Store,
    blob_hash: &ObjectHash,
    new_bytes: Vec<u8>,
) -> Result<EditOutcome> {
    if new_bytes.is_empty() {
        bail!("refusing to replace blob with empty content; use `rqm rm` instead");
    }
    edit_blob_with(store, blob_hash, |_old| Ok(EditFn::Apply(new_bytes)))
}

/// Open `blob_hash` in `$EDITOR` (or `vi` if unset), then apply the
/// result through [`edit_blob_with`].
pub fn edit_blob_interactive(store: &Store, blob_hash: &ObjectHash) -> Result<EditOutcome> {
    edit_blob_with(store, blob_hash, |old| {
        let tmp = tempfile::Builder::new()
            .prefix("rqm-edit-")
            .suffix(".txt")
            .tempfile()
            .context("create scratch file")?;
        fs::write(tmp.path(), old).context("write scratch file")?;

        let editor = env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
        let status = Command::new(&editor)
            .arg(tmp.path())
            .status()
            .with_context(|| format!("spawn editor {editor}"))?;
        if !status.success() {
            return Ok(EditFn::Cancel);
        }
        let new = fs::read(tmp.path()).context("read scratch file after edit")?;
        Ok(EditFn::Apply(new))
    })
}

/// Replace `old_blob_hash` everywhere it appears in `.rqm/` — in any
/// requirement's `text_blob` or `source_blobs`, and in any file-tree's
/// entries — with a new blob produced by `edit_fn`.
///
/// Updated metas have their refs repointed; updated file-trees have
/// their tree refs repointed. Old objects remain in `.rqm/objects/`
/// (immutable history) until a future GC pass.
pub fn edit_blob_with<F>(
    store: &Store,
    old_blob_hash: &ObjectHash,
    edit_fn: F,
) -> Result<EditOutcome>
where
    F: FnOnce(&[u8]) -> Result<EditFn>,
{
    let old_blob = store.read_blob(old_blob_hash)?;

    let new_bytes = match edit_fn(&old_blob.0)? {
        EditFn::Cancel => return Ok(EditOutcome::Canceled),
        EditFn::Apply(b) => b,
    };
    if new_bytes == old_blob.0 {
        return Ok(EditOutcome::Unchanged);
    }
    let new_blob_hash = store.write_blob(&Blob(new_bytes))?;

    // Walk every ref. If its meta references the old blob (as text_blob
    // or in source_blobs), produce an updated meta and repoint the ref.
    let mut metas_updated = Vec::new();
    for (stable_id, meta_hash) in store.ref_list()? {
        let mut meta = store.read_requirement(&meta_hash)?;
        // Skip metas reached via alias names — their canonical ref will
        // be visited separately, so visiting them here would double-write.
        if meta.stable_id != stable_id {
            continue;
        }
        let mut changed = false;
        if meta.text_blob == *old_blob_hash {
            meta.text_blob = new_blob_hash;
            changed = true;
        }
        let mut new_source = Vec::with_capacity(meta.source_blobs.len());
        for h in &meta.source_blobs {
            if *h == *old_blob_hash {
                new_source.push(new_blob_hash);
                changed = true;
            } else {
                new_source.push(*h);
            }
        }
        if changed {
            // Re-sort to preserve canonical ordering of source_blobs.
            new_source.sort();
            meta.source_blobs = new_source;
            let new_meta_hash = store.write_requirement(&meta)?;
            store.ref_set(&stable_id, &new_meta_hash)?;
            metas_updated.push(stable_id);
        }
    }

    // Walk every managed path's file-tree. Replace old-blob entries.
    let mut paths_updated = Vec::new();
    for path in store.managed_paths()? {
        let Some(tree_hash) = store.tree_get(&path)? else { continue };
        let mut tree = store.read_file_tree(&tree_hash)?;
        let mut changed = false;
        for entry in tree.entries.iter_mut() {
            if entry.blob == *old_blob_hash {
                entry.blob = new_blob_hash;
                changed = true;
            }
        }
        if changed {
            let new_tree_hash = store.write_file_tree(&tree)?;
            store.tree_set(&path, &new_tree_hash)?;
            paths_updated.push(path);
        }
    }

    Ok(EditOutcome::Changed {
        new_blob: new_blob_hash,
        metas_updated,
        paths_updated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materialize;
    use crate::migrate;
    use crate::object::{Kind, Requirement};

    fn setup() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::init(dir.path().join(".rqm")).unwrap();
        (dir, store)
    }

    fn write_at(root: &Path, rel: &str, content: &str) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(&p, content).unwrap();
    }

    // ── EditTarget parsing ────────────────────────────────────────────

    #[test]
    fn parse_stable_id_target() {
        match EditTarget::parse("rq-4d1082c4").unwrap() {
            EditTarget::Id(id) => assert_eq!(id.to_string(), "rq-4d1082c4"),
            other => panic!("expected Id, got {other:?}"),
        }
    }

    #[test]
    fn parse_file_line_target() {
        match EditTarget::parse("rqm/foo.md:42").unwrap() {
            EditTarget::FileLine { path, line } => {
                assert_eq!(path, PathBuf::from("rqm/foo.md"));
                assert_eq!(line, 42);
            }
            other => panic!("expected FileLine, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_invalid() {
        assert!(EditTarget::parse("not-an-id").is_err());
        assert!(EditTarget::parse("foo.md:notanumber").is_err());
        assert!(EditTarget::parse("rq-tooshort").is_err());
        assert!(EditTarget::parse("rq-XXXXXXXX").is_err()); // uppercase
        assert!(EditTarget::parse("foo.md:0").is_err());
    }

    // ── target_to_blob: text-blob via stable_id ───────────────────────

    fn setup_md() -> (tempfile::TempDir, Store) {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "\
# Top <!-- rq-aaaaaaaa -->

Top body line 3.
Top body line 4.

## First <!-- rq-bbbbbbbb -->

First body.

## Second <!-- rq-cccccccc -->

Second body.
";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        (dir, store)
    }

    #[test]
    fn resolve_stable_id_to_text_blob() {
        let (_dir, store) = setup_md();
        let target = EditTarget::Id(StableId::new("rq-aaaaaaaa"));
        let blob = target_to_blob(&store, &target).unwrap();
        let bytes = store.read_blob(&blob).unwrap();
        assert!(std::str::from_utf8(&bytes.0).unwrap().contains("Top body"));
    }

    #[test]
    fn resolve_rejects_alias_id() {
        let (dir, store) = setup();
        let root = dir.path();
        write_at(
            root,
            "rqm/doc.md",
            "# Top <!-- rq-aaaaaaaa -->\n\n## Sub <!-- rq-bbbbbbbb -->\n\n- bullet <!-- rq-cccccccc -->\n",
        );
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        let target = EditTarget::Id(StableId::new("rq-cccccccc"));
        let err = target_to_blob(&store, &target).unwrap_err();
        assert!(
            err.to_string().contains("alias for rq-bbbbbbbb"),
            "{err}"
        );
    }

    #[test]
    fn resolve_rejects_unknown_id() {
        let (_dir, store) = setup_md();
        let target = EditTarget::Id(StableId::new("rq-12345678"));
        let err = target_to_blob(&store, &target).unwrap_err();
        assert!(err.to_string().contains("unknown stable_id"), "{err}");
    }

    // ── target_to_blob: file:line on markdown ─────────────────────────

    #[test]
    fn resolve_file_line_md_top_blob() {
        let (_dir, store) = setup_md();
        let target = EditTarget::FileLine {
            path: PathBuf::from("rqm/doc.md"),
            line: 1,
        };
        let blob = target_to_blob(&store, &target).unwrap();
        let bytes = store.read_blob(&blob).unwrap();
        let s = std::str::from_utf8(&bytes.0).unwrap();
        assert!(s.starts_with("# Top"));
    }

    #[test]
    fn resolve_file_line_md_second_blob() {
        let (_dir, store) = setup_md();
        // Line 7 is "## First <!-- rq-bbbbbbbb -->"
        let target = EditTarget::FileLine {
            path: PathBuf::from("rqm/doc.md"),
            line: 7,
        };
        let blob = target_to_blob(&store, &target).unwrap();
        let bytes = store.read_blob(&blob).unwrap();
        let s = std::str::from_utf8(&bytes.0).unwrap();
        assert!(s.starts_with("## First"));
    }

    #[test]
    fn resolve_file_line_out_of_range() {
        let (_dir, store) = setup_md();
        let target = EditTarget::FileLine {
            path: PathBuf::from("rqm/doc.md"),
            line: 9999,
        };
        let err = target_to_blob(&store, &target).unwrap_err();
        assert!(err.to_string().contains("out of range"), "{err}");
    }

    #[test]
    fn resolve_file_line_unmanaged_path() {
        let (_dir, store) = setup_md();
        let target = EditTarget::FileLine {
            path: PathBuf::from("rqm/missing.md"),
            line: 1,
        };
        let err = target_to_blob(&store, &target).unwrap_err();
        assert!(err.to_string().contains("not a managed path"), "{err}");
    }

    // ── target_to_blob: file:line on source ───────────────────────────

    fn make_req(store: &Store, id: &str) -> ObjectHash {
        let text = store.write_blob(&Blob(b"prose".to_vec())).unwrap();
        let req = Requirement {
            stable_id: StableId::new(id),
            kind: Kind::Behavior,
            text_blob: text,
            parents: vec![],
            source_blobs: vec![],
        };
        let h = store.write_requirement(&req).unwrap();
        store.ref_set(&StableId::new(id), &h).unwrap();
        h
    }

    fn setup_with_source() -> (tempfile::TempDir, Store) {
        let (dir, store) = setup();
        let root = dir.path();
        // Two-blob source file: prelude+rq-aaaaaaaa for first half, rq-bbbbbbbb second half.
        make_req(&store, "rq-aaaaaaaa");
        make_req(&store, "rq-bbbbbbbb");
        let src = "\
// rq-aaaaaaaa
fn alpha() {
    println!(\"alpha\");
}

// rq-bbbbbbbb
fn beta() {
    println!(\"beta\");
}
";
        write_at(root, "src/example.rs", src);
        migrate::migrate_source(&store, root, Path::new("src/example.rs")).unwrap();
        (dir, store)
    }

    #[test]
    fn resolve_file_line_source_first_blob() {
        let (_dir, store) = setup_with_source();
        let target = EditTarget::FileLine {
            path: PathBuf::from("src/example.rs"),
            line: 3, // body of alpha()
        };
        let blob = target_to_blob(&store, &target).unwrap();
        let bytes = store.read_blob(&blob).unwrap();
        let s = std::str::from_utf8(&bytes.0).unwrap();
        assert!(s.contains("alpha"));
        assert!(!s.contains("beta"));
    }

    #[test]
    fn resolve_file_line_source_second_blob() {
        let (_dir, store) = setup_with_source();
        let target = EditTarget::FileLine {
            path: PathBuf::from("src/example.rs"),
            line: 8, // body of beta()
        };
        let blob = target_to_blob(&store, &target).unwrap();
        let bytes = store.read_blob(&blob).unwrap();
        let s = std::str::from_utf8(&bytes.0).unwrap();
        assert!(s.contains("beta"));
        assert!(!s.contains("alpha"));
    }

    // ── edit_blob_with: text-blob path ────────────────────────────────

    #[test]
    fn text_blob_edit_unchanged() {
        let (_dir, store) = setup_md();
        let target = EditTarget::Id(StableId::new("rq-aaaaaaaa"));
        let blob = target_to_blob(&store, &target).unwrap();
        let outcome =
            edit_blob_with(&store, &blob, |old| Ok(EditFn::Apply(old.to_vec()))).unwrap();
        assert!(matches!(outcome, EditOutcome::Unchanged));
    }

    #[test]
    fn text_blob_edit_canceled() {
        let (_dir, store) = setup_md();
        let target = EditTarget::Id(StableId::new("rq-aaaaaaaa"));
        let blob = target_to_blob(&store, &target).unwrap();
        let outcome = edit_blob_with(&store, &blob, |_| Ok(EditFn::Cancel)).unwrap();
        assert!(matches!(outcome, EditOutcome::Canceled));
    }

    #[test]
    fn text_blob_edit_changes_propagate() {
        let (dir, store) = setup_md();
        let root = dir.path();
        let target = EditTarget::Id(StableId::new("rq-aaaaaaaa"));
        let blob = target_to_blob(&store, &target).unwrap();
        let outcome = edit_blob_with(&store, &blob, |old| {
            let s = std::str::from_utf8(old).unwrap();
            Ok(EditFn::Apply(s.replace("Top body", "REPLACED").into_bytes()))
        })
        .unwrap();
        match outcome {
            EditOutcome::Changed {
                metas_updated,
                paths_updated,
                ..
            } => {
                assert_eq!(metas_updated, vec![StableId::new("rq-aaaaaaaa")]);
                assert_eq!(paths_updated, vec![PathBuf::from("rqm/doc.md")]);
            }
            other => panic!("expected Changed, got {other:?}"),
        }
        materialize::build(&store, root).unwrap();
        let on_disk = fs::read_to_string(root.join("rqm/doc.md")).unwrap();
        assert!(on_disk.contains("REPLACED"));
        assert!(!on_disk.contains("Top body"));
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    // ── edit_blob_with: source-blob path ──────────────────────────────

    #[test]
    fn source_blob_edit_round_trips() {
        let (dir, store) = setup_with_source();
        let root = dir.path();
        let target = EditTarget::FileLine {
            path: PathBuf::from("src/example.rs"),
            line: 3,
        };
        let blob = target_to_blob(&store, &target).unwrap();
        let outcome = edit_blob_with(&store, &blob, |old| {
            let s = std::str::from_utf8(old).unwrap();
            Ok(EditFn::Apply(
                s.replace("println!(\"alpha\")", "println!(\"ALPHA-EDITED\")")
                    .into_bytes(),
            ))
        })
        .unwrap();
        match outcome {
            EditOutcome::Changed {
                metas_updated,
                paths_updated,
                ..
            } => {
                assert_eq!(metas_updated, vec![StableId::new("rq-aaaaaaaa")]);
                assert_eq!(paths_updated, vec![PathBuf::from("src/example.rs")]);
            }
            other => panic!("expected Changed, got {other:?}"),
        }
        materialize::build(&store, root).unwrap();
        let on_disk = fs::read_to_string(root.join("src/example.rs")).unwrap();
        assert!(on_disk.contains("ALPHA-EDITED"));
        assert!(!on_disk.contains("println!(\"alpha\")"));
        // beta() is untouched.
        assert!(on_disk.contains("println!(\"beta\")"));
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn source_blob_joint_ownership_updates_all_metas() {
        let (dir, store) = setup();
        let root = dir.path();
        make_req(&store, "rq-aaaaaaaa");
        make_req(&store, "rq-bbbbbbbb");
        make_req(&store, "rq-cccccccc");
        // The "shared" blob is owned by both rq-bbbbbbbb and rq-cccccccc.
        let src = "\
// rq-aaaaaaaa
fn one() {}

// rq-bbbbbbbb rq-cccccccc
fn shared() {
    let x = 1;
}
";
        write_at(root, "src/example.rs", src);
        migrate::migrate_source(&store, root, Path::new("src/example.rs")).unwrap();

        // Edit the shared blob.
        let target = EditTarget::FileLine {
            path: PathBuf::from("src/example.rs"),
            line: 5, // inside shared()
        };
        let blob = target_to_blob(&store, &target).unwrap();
        let outcome = edit_blob_with(&store, &blob, |old| {
            let s = std::str::from_utf8(old).unwrap();
            Ok(EditFn::Apply(s.replace("let x = 1", "let x = 99").into_bytes()))
        })
        .unwrap();

        // Both jointly-owning metas were updated.
        match outcome {
            EditOutcome::Changed { metas_updated, .. } => {
                let mut ids: Vec<String> =
                    metas_updated.iter().map(|i| i.to_string()).collect();
                ids.sort();
                assert_eq!(ids, vec!["rq-bbbbbbbb", "rq-cccccccc"]);
            }
            other => panic!("expected Changed, got {other:?}"),
        }

        // After build, the materialized source reflects the edit.
        materialize::build(&store, root).unwrap();
        let on_disk = fs::read_to_string(root.join("src/example.rs")).unwrap();
        assert!(on_disk.contains("let x = 99"));
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }
}
