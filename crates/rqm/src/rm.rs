//! `rqm rm` — remove a blob from a managed file.
//!
//! Removes the file-tree entry covering the specified line. If the
//! blob is no longer referenced by *any* file-tree entry afterward,
//! it is also removed from every meta's `source_blobs` (preserving
//! the round-trip integrity rule that source_blobs entries must be
//! materialized somewhere).
//!
//! Refuses to remove a blob that is some requirement's `text_blob`
//! unless `--force` is given — doing so leaves the requirement
//! "ghosted" (its prose lives in `.rqm/objects/` but is not
//! materialized in any file).

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::mv::find_entry_at_line_with_offset;
use crate::object::{ObjectHash, Requirement, StableId};
use crate::store::Store;

#[derive(Debug, Clone)]
pub struct RemoveSpec {
    pub path: PathBuf,
    pub line: usize,
}

impl RemoveSpec {
    pub fn parse(s: &str) -> Result<Self> {
        let (path, line) = s.rsplit_once(':').ok_or_else(|| {
            anyhow::anyhow!("target must be <path>:<line>, got {s:?}")
        })?;
        if path.is_empty() {
            bail!("empty path in {s:?}");
        }
        let line: usize = line
            .parse()
            .with_context(|| format!("invalid line number in {s:?}"))?;
        if line == 0 {
            bail!("line numbers are 1-based");
        }
        Ok(RemoveSpec {
            path: PathBuf::from(path),
            line,
        })
    }
}

#[derive(Debug)]
pub struct RemoveOutcome {
    pub removed_blob: ObjectHash,
    /// Metas whose `source_blobs` was modified.
    pub metas_updated: Vec<StableId>,
    /// `Some(id)` when the removed blob was `id`'s text_blob — i.e.
    /// the requirement is now ghosted. Empty when the removed blob
    /// wasn't anyone's text_blob.
    pub ghosted_requirement: Option<StableId>,
    pub path: PathBuf,
}

pub fn do_remove(store: &Store, spec: RemoveSpec, force: bool) -> Result<RemoveOutcome> {
    let tree_hash = store.tree_get(&spec.path)?.ok_or_else(|| {
        anyhow::anyhow!(
            "{}: not a managed path in .rqm/",
            spec.path.display()
        )
    })?;
    let mut tree = store.read_file_tree(&tree_hash)?;

    let (idx, _) = find_entry_at_line_with_offset(store, &tree, spec.line)?;
    let removed_blob = tree.entries[idx].blob;

    // Is the blob any requirement's text_blob? If so, refuse without --force.
    let ghosted = text_blob_owner(store, &removed_blob)?;
    if let Some(id) = &ghosted {
        if !force {
            bail!(
                "{}:{} is the text_blob of {id}; removing it would leave the \
                 requirement without materialized prose. Pass --force to override \
                 (the requirement will continue to exist but won't render in any \
                 managed file).",
                spec.path.display(),
                spec.line
            );
        }
    }

    // Remove the entry from the file-tree and rewrite.
    tree.entries.remove(idx);
    let new_tree_hash = store.write_file_tree(&tree)?;
    store.tree_set(&spec.path, &new_tree_hash)?;

    // If the blob isn't referenced by any file-tree entry anywhere now,
    // remove it from every meta's source_blobs to preserve integrity rule
    // 3 (every blob in source_blobs appears in some file-tree).
    let mut metas_updated = Vec::new();
    if !blob_still_referenced(store, &removed_blob)? {
        for (sid, mhash) in store.ref_list()? {
            let mut meta: Requirement = store.read_requirement(&mhash)?;
            if meta.stable_id != sid {
                // Alias ref — the canonical will be visited separately.
                continue;
            }
            if meta.source_blobs.contains(&removed_blob) {
                meta.source_blobs.retain(|h| *h != removed_blob);
                let new_mhash = store.write_requirement(&meta)?;
                store.ref_set(&sid, &new_mhash)?;
                metas_updated.push(sid);
            }
        }
    }

    Ok(RemoveOutcome {
        removed_blob,
        metas_updated,
        ghosted_requirement: ghosted,
        path: spec.path,
    })
}

/// Returns the canonical stable_id whose `text_blob` is `blob_hash`,
/// or `None` if no requirement claims it.
fn text_blob_owner(store: &Store, blob_hash: &ObjectHash) -> Result<Option<StableId>> {
    for (sid, mhash) in store.ref_list()? {
        let meta: Requirement = store.read_requirement(&mhash)?;
        if meta.stable_id != sid {
            continue;
        }
        if meta.text_blob == *blob_hash {
            return Ok(Some(sid));
        }
    }
    Ok(None)
}

/// Walks every managed file-tree looking for any entry that references
/// `blob_hash`. Returns true on first hit.
fn blob_still_referenced(store: &Store, blob_hash: &ObjectHash) -> Result<bool> {
    for path in store.managed_paths()? {
        let Some(tree_hash) = store.tree_get(&path)? else { continue };
        let tree = store.read_file_tree(&tree_hash)?;
        for entry in &tree.entries {
            if entry.blob == *blob_hash {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

// ── Requirement removal ───────────────────────────────────────────────

#[derive(Debug)]
pub struct RemoveRequirementOutcome {
    pub removed: StableId,
    /// Managed paths whose file-trees had entries removed.
    pub paths_updated: Vec<PathBuf>,
    /// Alias IDs that pointed at this canonical and were auto-deleted.
    pub aliases_removed: Vec<StableId>,
}

pub fn do_remove_requirement(
    store: &Store,
    id: &StableId,
) -> Result<RemoveRequirementOutcome> {
    // Refuse if the id is unknown or is itself an alias. (Aliases are
    // auto-cleaned when their canonical is removed; trying to "remove"
    // an alias directly is not meaningful.)
    let canonical = store.resolve(id)?.ok_or_else(|| {
        anyhow::anyhow!("unknown stable_id: {id}")
    })?;
    if &canonical != id {
        bail!(
            "{id} is an alias for {canonical}; aliases are auto-removed when \
             their canonical requirement is deleted. Pass the canonical id."
        );
    }

    let meta_hash = store
        .ref_get(id)?
        .expect("resolve() above confirmed the ref exists");
    let meta: Requirement = store.read_requirement(&meta_hash)?;

    // Refuse if any source blobs remain attached.
    if !meta.source_blobs.is_empty() {
        bail!(
            "{id} has {} source blob(s) attached; remove them first \
             (e.g. via `rqm rm <path>:<line>`).",
            meta.source_blobs.len()
        );
    }

    // Refuse if any other requirement names this one as parent.
    let mut children = Vec::new();
    for (sid, mhash) in store.ref_list()? {
        let cmeta: Requirement = store.read_requirement(&mhash)?;
        if cmeta.stable_id != sid {
            // Alias ref — skip; the canonical will be visited separately.
            continue;
        }
        if cmeta.parent.as_ref() == Some(id) {
            children.push(sid);
        }
    }
    if !children.is_empty() {
        let names = children
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        bail!(
            "{id} has {} child requirement(s): {names}. Remove or reassign \
             them first.",
            children.len()
        );
    }

    // Remove every file-tree entry whose stable_id is this requirement.
    let mut paths_updated = Vec::new();
    for path in store.managed_paths()? {
        let Some(tree_hash) = store.tree_get(&path)? else { continue };
        let mut tree = store.read_file_tree(&tree_hash)?;
        let before = tree.entries.len();
        tree.entries.retain(|e| e.stable_id != *id);
        if tree.entries.len() != before {
            let new_hash = store.write_file_tree(&tree)?;
            store.tree_set(&path, &new_hash)?;
            paths_updated.push(path);
        }
    }

    // Auto-delete any aliases pointing at this canonical.
    let mut aliases_removed = Vec::new();
    for (alias, canon) in store.alias_list()? {
        if &canon == id {
            store.alias_delete(&alias)?;
            aliases_removed.push(alias);
        }
    }

    // Delete the ref. (The meta and text_blob remain in
    // .rqm/objects/ — they are immutable history. A future `rqm gc`
    // will prune unreachable objects.)
    store.ref_delete(id)?;

    Ok(RemoveRequirementOutcome {
        removed: id.clone(),
        paths_updated,
        aliases_removed,
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::*;
    use crate::insert::{self, InsertMode, InsertSpec};
    use crate::materialize;
    use crate::migrate;
    use crate::object::{Blob, Kind};

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

    fn make_req(store: &Store, id: &str) {
        let text = store.write_blob(&Blob(b"prose".to_vec())).unwrap();
        let req = Requirement {
            stable_id: StableId::new(id),
            kind: Kind::Behavior,
            text_blob: text,
            parent: None,
            source_blobs: vec![],
        };
        let h = store.write_requirement(&req).unwrap();
        store.ref_set(&StableId::new(id), &h).unwrap();
    }

    fn setup_with_source() -> (tempfile::TempDir, Store) {
        let (dir, store) = setup();
        let root = dir.path();
        make_req(&store, "rq-aaaaaaaa");
        make_req(&store, "rq-bbbbbbbb");
        let src = "// rq-aaaaaaaa\nfn alpha() {}\n\n// rq-bbbbbbbb\nfn beta() {}\n";
        write_at(root, "src/x.rs", src);
        migrate::migrate_source(&store, root, Path::new("src/x.rs")).unwrap();
        (dir, store)
    }

    // ── Parsing ───────────────────────────────────────────────────────

    #[test]
    fn parse_target() {
        let s = RemoveSpec::parse("src/x.rs:42").unwrap();
        assert_eq!(s.path, PathBuf::from("src/x.rs"));
        assert_eq!(s.line, 42);
        assert!(RemoveSpec::parse("no-colon").is_err());
        assert!(RemoveSpec::parse(":42").is_err());
        assert!(RemoveSpec::parse("src/x.rs:0").is_err());
    }

    // ── Source blob removal ───────────────────────────────────────────

    #[test]
    fn rm_source_blob_drops_entry_and_source_blob() {
        let (dir, store) = setup_with_source();
        let root = dir.path();
        // Remove the beta() blob (line 5 in our test fixture).
        let outcome = do_remove(
            &store,
            RemoveSpec::parse("src/x.rs:5").unwrap(),
            false,
        )
        .unwrap();
        assert_eq!(outcome.metas_updated, vec![StableId::new("rq-bbbbbbbb")]);
        assert!(outcome.ghosted_requirement.is_none());

        materialize::build(&store, root).unwrap();
        let on_disk = fs::read_to_string(root.join("src/x.rs")).unwrap();
        assert!(on_disk.contains("alpha"));
        assert!(!on_disk.contains("beta"));

        // rq-bbbbbbbb's source_blobs should be empty now.
        let h = store.ref_get(&StableId::new("rq-bbbbbbbb")).unwrap().unwrap();
        let req = store.read_requirement(&h).unwrap();
        assert!(req.source_blobs.is_empty());

        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn rm_blob_owned_by_multiple_metas_updates_all() {
        let (dir, store) = setup();
        let root = dir.path();
        make_req(&store, "rq-aaaaaaaa");
        make_req(&store, "rq-bbbbbbbb");
        let src = "// rq-aaaaaaaa rq-bbbbbbbb\nfn shared() {\n    let x = 1;\n}\n";
        write_at(root, "src/x.rs", src);
        migrate::migrate_source(&store, root, Path::new("src/x.rs")).unwrap();

        // Both rq-aaaaaaaa and rq-bbbbbbbb own the (single) blob in src/x.rs.
        let outcome = do_remove(
            &store,
            RemoveSpec::parse("src/x.rs:2").unwrap(),
            false,
        )
        .unwrap();
        let mut ids: Vec<String> = outcome.metas_updated.iter().map(|i| i.to_string()).collect();
        ids.sort();
        assert_eq!(ids, vec!["rq-aaaaaaaa", "rq-bbbbbbbb"]);
        materialize::build(&store, root).unwrap();
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    // ── text_blob protection ──────────────────────────────────────────

    #[test]
    fn rm_text_blob_refused_without_force() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n\nBody.\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        let err = do_remove(
            &store,
            RemoveSpec::parse("rqm/doc.md:1").unwrap(),
            false,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("text_blob of rq-aaaaaaaa"),
            "{err}"
        );
        // The file-tree should be unchanged.
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn rm_text_blob_with_force_ghosts_the_requirement() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n\nBody.\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        let outcome = do_remove(
            &store,
            RemoveSpec::parse("rqm/doc.md:1").unwrap(),
            true,
        )
        .unwrap();
        assert_eq!(
            outcome.ghosted_requirement.as_ref().map(|s| s.to_string()),
            Some("rq-aaaaaaaa".to_string())
        );
        materialize::build(&store, root).unwrap();
        let on_disk = fs::read_to_string(root.join("rqm/doc.md")).unwrap();
        // The file is now empty (or contains nothing of the prose).
        assert_eq!(on_disk, "");
        // The requirement still exists but is ghosted.
        let h = store.ref_get(&StableId::new("rq-aaaaaaaa")).unwrap().unwrap();
        let _req = store.read_requirement(&h).unwrap();
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    // ── Validation ────────────────────────────────────────────────────

    #[test]
    fn rm_rejects_unmanaged_path() {
        let (_dir, store) = setup_with_source();
        let err = do_remove(
            &store,
            RemoveSpec::parse("src/no_such.rs:1").unwrap(),
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("not a managed path"), "{err}");
    }

    #[test]
    fn rm_rejects_out_of_range_line() {
        let (_dir, store) = setup_with_source();
        let err = do_remove(
            &store,
            RemoveSpec::parse("src/x.rs:9999").unwrap(),
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("out of range"), "{err}");
    }

    // ── Duplicate-reference handling ──────────────────────────────────

    // ── Requirement removal ───────────────────────────────────────────

    #[test]
    fn rm_requirement_with_source_blobs_refused() {
        let (dir, store) = setup_with_source();
        let _root = dir.path();
        let err = do_remove_requirement(&store, &StableId::new("rq-aaaaaaaa"))
            .unwrap_err();
        assert!(
            err.to_string().contains("source blob"),
            "{err}"
        );
    }

    #[test]
    fn rm_requirement_with_children_refused() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n\n## Child <!-- rq-bbbbbbbb -->\n\nBody.\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        let err = do_remove_requirement(&store, &StableId::new("rq-aaaaaaaa"))
            .unwrap_err();
        assert!(
            err.to_string().contains("child requirement"),
            "{err}"
        );
    }

    #[test]
    fn rm_requirement_alias_refused() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n\n- bullet <!-- rq-bbbbbbbb -->\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        // rq-bbbbbbbb is an alias for rq-aaaaaaaa.
        let err = do_remove_requirement(&store, &StableId::new("rq-bbbbbbbb"))
            .unwrap_err();
        assert!(
            err.to_string().contains("is an alias for"),
            "{err}"
        );
    }

    #[test]
    fn rm_requirement_unknown_refused() {
        let (_dir, store) = setup();
        let err = do_remove_requirement(&store, &StableId::new("rq-99999999"))
            .unwrap_err();
        assert!(err.to_string().contains("unknown stable_id"), "{err}");
    }

    #[test]
    fn rm_requirement_succeeds_for_leaf_with_no_source_blobs() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n\n## Leaf <!-- rq-bbbbbbbb -->\n\nBody.\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        let outcome = do_remove_requirement(&store, &StableId::new("rq-bbbbbbbb"))
            .unwrap();
        assert_eq!(outcome.removed.to_string(), "rq-bbbbbbbb");
        assert!(outcome.aliases_removed.is_empty());
        assert_eq!(outcome.paths_updated, vec![PathBuf::from("rqm/doc.md")]);

        // The ref is gone.
        assert!(store.ref_get(&StableId::new("rq-bbbbbbbb")).unwrap().is_none());
        // The materialized markdown no longer contains the heading.
        materialize::build(&store, root).unwrap();
        let on_disk = fs::read_to_string(root.join("rqm/doc.md")).unwrap();
        assert!(!on_disk.contains("Leaf"));
        assert!(on_disk.contains("Top"));
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn rm_requirement_auto_deletes_aliases() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "\
# Top <!-- rq-aaaaaaaa -->

## Section <!-- rq-bbbbbbbb -->

- item one <!-- rq-cccccccc -->
- item two <!-- rq-dddddddd -->
";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();

        // rq-cccccccc and rq-dddddddd are aliases for rq-bbbbbbbb.
        // Removing rq-bbbbbbbb should auto-delete them.
        let outcome = do_remove_requirement(&store, &StableId::new("rq-bbbbbbbb"))
            .unwrap();
        let mut aliases: Vec<String> =
            outcome.aliases_removed.iter().map(|i| i.to_string()).collect();
        aliases.sort();
        assert_eq!(aliases, vec!["rq-cccccccc", "rq-dddddddd"]);

        // Aliases really are gone from .rqm/aliases/.
        assert!(store.alias_get(&StableId::new("rq-cccccccc")).unwrap().is_none());
        assert!(store.alias_get(&StableId::new("rq-dddddddd")).unwrap().is_none());
    }

    // ── Duplicate-reference handling (blob form) ──────────────────────

    #[test]
    fn rm_keeps_blob_in_source_blobs_when_still_referenced() {
        // Use insert to create a situation where the same blob hash
        // appears in two file-trees. We do this by inserting identical
        // content in two places — content-addressing dedups to one blob.
        let (dir, store) = setup();
        let root = dir.path();
        make_req(&store, "rq-aaaaaaaa");
        write_at(root, "src/a.rs", "// rq-aaaaaaaa\nfn a() {}\n");
        migrate::migrate_source(&store, root, Path::new("src/a.rs")).unwrap();

        // Insert the same blob content into a second file.
        insert::do_insert(
            &store,
            InsertSpec::parse("src/b.rs:end", false).unwrap(),
            InsertMode::AttributeTo {
                owners: vec![StableId::new("rq-aaaaaaaa")],
            },
            b"// rq-aaaaaaaa\nfn a() {}\n".to_vec(),
        )
        .unwrap();

        // Now remove the entry in src/a.rs. The blob is still referenced
        // from src/b.rs, so source_blobs should NOT be cleaned.
        let outcome = do_remove(
            &store,
            RemoveSpec::parse("src/a.rs:1").unwrap(),
            false,
        )
        .unwrap();
        assert!(
            outcome.metas_updated.is_empty(),
            "blob still in use elsewhere; source_blobs shouldn't change: {:?}",
            outcome.metas_updated
        );
        // The requirement still has the blob.
        let h = store.ref_get(&StableId::new("rq-aaaaaaaa")).unwrap().unwrap();
        let req = store.read_requirement(&h).unwrap();
        assert_eq!(req.source_blobs.len(), 1);

        materialize::build(&store, root).unwrap();
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }
}
