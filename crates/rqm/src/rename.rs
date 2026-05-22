//! `rqm rename` — change a requirement's stable_id everywhere.
//!
//! Touches: the ref name, the meta's `stable_id` field, every child's
//! `parents` list, every file-tree entry that names this stable_id,
//! every alias whose canonical points at it, and every reachable blob
//! whose bytes contain the old id as a word-bounded `rq-XXXXXXXX`
//! stamp.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::object::{Blob, FileTree, ObjectHash, Requirement, StableId};
use crate::store::Store;

#[derive(Debug)]
pub struct RenameOutcome {
    pub old: StableId,
    pub new: StableId,
    /// Canonical stable_ids whose metas were rewritten (the renamed
    /// requirement itself, plus any child whose `parents` list was
    /// updated, plus any whose owned blobs were rewritten).
    pub metas_updated: Vec<StableId>,
    /// Managed paths whose file-trees were rewritten.
    pub paths_updated: Vec<std::path::PathBuf>,
    /// Alias IDs whose canonical pointer was redirected.
    pub aliases_updated: Vec<StableId>,
    /// Number of blobs that were rewritten to substitute the new id
    /// for the old in stamp tokens.
    pub blobs_rewritten: usize,
}

/// Rename canonical `old` to `new`. The new id must not already be
/// registered as a canonical or an alias.
pub fn do_rename(
    store: &Store,
    old: &StableId,
    new: &StableId,
) -> Result<RenameOutcome> {
    // Validate target is canonical and exists.
    let canonical = store
        .resolve(old)?
        .ok_or_else(|| anyhow::anyhow!("unknown stable_id: {old}"))?;
    if &canonical != old {
        bail!(
            "{old} is an alias for {canonical}; rename the canonical id instead."
        );
    }

    // Validate new id format.
    if !is_valid_stable_id(new) {
        bail!(
            "invalid new stable_id {new:?}; expected rq-XXXXXXXX \
             (8 lowercase hex digits)"
        );
    }

    // No-op.
    if old == new {
        return Ok(RenameOutcome {
            old: old.clone(),
            new: new.clone(),
            metas_updated: Vec::new(),
            paths_updated: Vec::new(),
            aliases_updated: Vec::new(),
            blobs_rewritten: 0,
        });
    }

    // New id must be unused.
    if store.ref_get(new)?.is_some() {
        bail!("{new} already exists as a canonical requirement");
    }
    if store.alias_get(new)?.is_some() {
        bail!("{new} already exists as an alias");
    }

    // Step 1: enumerate reachable blobs and rewrite any that contain
    // the old id as a word-bounded stamp token. Build a remap from
    // old blob hash to new blob hash.
    let reachable = enumerate_reachable_blobs(store)?;
    let mut blob_remap: HashMap<ObjectHash, ObjectHash> = HashMap::new();
    for blob_hash in &reachable {
        let blob = store.read_blob(blob_hash)?;
        if let Some(rewritten) = rewrite_stamps(&blob.0, old, new) {
            let new_hash = store.write_blob(&Blob(rewritten))?;
            if new_hash != *blob_hash {
                blob_remap.insert(*blob_hash, new_hash);
            }
        }
    }

    // Step 2: rewrite every meta that needs updating. A meta needs
    // rewriting if it's the renamed target, has the target in its
    // parents, or owns a remapped blob.
    let mut metas_updated = Vec::new();
    for (sid, mhash) in store.ref_list()? {
        let mut meta: Requirement = store.read_requirement(&mhash)?;
        // Skip alias refs (the canonical will be visited separately).
        if meta.stable_id != sid {
            continue;
        }
        let mut changed = false;

        if meta.stable_id == *old {
            meta.stable_id = new.clone();
            changed = true;
        }
        if meta.parents.iter().any(|p| p == old) {
            for p in meta.parents.iter_mut() {
                if p == old {
                    *p = new.clone();
                }
            }
            meta.parents.sort();
            meta.parents.dedup();
            changed = true;
        }
        if let Some(new_b) = blob_remap.get(&meta.text_blob) {
            meta.text_blob = *new_b;
            changed = true;
        }
        let mut new_source = Vec::with_capacity(meta.source_blobs.len());
        for h in &meta.source_blobs {
            if let Some(new_b) = blob_remap.get(h) {
                new_source.push(*new_b);
                changed = true;
            } else {
                new_source.push(*h);
            }
        }
        if changed {
            new_source.sort();
            new_source.dedup();
            meta.source_blobs = new_source;
            let new_mhash = store.write_requirement(&meta)?;
            // If this is the renamed target, the ref name itself changes.
            if meta.stable_id == *new && sid == *old {
                store.ref_delete(old)?;
                store.ref_set(new, &new_mhash)?;
            } else {
                store.ref_set(&sid, &new_mhash)?;
            }
            metas_updated.push(meta.stable_id.clone());
        }
    }

    // Step 3: rewrite file-trees. For each entry, update the stable_id
    // if it's the renamed target, and remap the blob hash if applicable.
    let mut paths_updated = Vec::new();
    for path in store.managed_paths()? {
        let Some(tree_hash) = store.tree_get(&path)? else { continue };
        let mut tree: FileTree = store.read_file_tree(&tree_hash)?;
        let mut changed = false;
        for entry in tree.entries.iter_mut() {
            if entry.stable_id == *old {
                entry.stable_id = new.clone();
                changed = true;
            }
            if let Some(new_b) = blob_remap.get(&entry.blob) {
                entry.blob = *new_b;
                changed = true;
            }
        }
        if changed {
            let new_tree_hash = store.write_file_tree(&tree)?;
            store.tree_set(&path, &new_tree_hash)?;
            paths_updated.push(path);
        }
    }

    // Step 4: redirect aliases whose canonical is the renamed target.
    let mut aliases_updated = Vec::new();
    for (alias, canon) in store.alias_list()? {
        if &canon == old {
            store.alias_set(&alias, new)?;
            aliases_updated.push(alias);
        }
    }

    Ok(RenameOutcome {
        old: old.clone(),
        new: new.clone(),
        metas_updated,
        paths_updated,
        aliases_updated,
        blobs_rewritten: blob_remap.len(),
    })
}

/// Enumerate every blob hash reachable from refs (meta text_blob and
/// source_blobs) and file-trees (entry blobs).
fn enumerate_reachable_blobs(store: &Store) -> Result<HashSet<ObjectHash>> {
    let mut out: HashSet<ObjectHash> = HashSet::new();
    for (_sid, mhash) in store.ref_list()? {
        let meta: Requirement = store.read_requirement(&mhash)?;
        out.insert(meta.text_blob);
        for h in meta.source_blobs {
            out.insert(h);
        }
    }
    for path in store.managed_paths()? {
        let Some(tree_hash) = store.tree_get(&path)? else { continue };
        let tree = store.read_file_tree(&tree_hash)?;
        for entry in tree.entries {
            out.insert(entry.blob);
        }
    }
    Ok(out)
}

/// If `bytes` contains the old id as a word-bounded `rq-XXXXXXXX`
/// stamp anywhere, return a new Vec with each occurrence replaced by
/// the new id. Returns None if no substitutions were made.
///
/// `old` and `new` must be the same length (both are
/// `rq-` + 8 hex chars = 11 bytes). The result is the same length as
/// the input.
fn rewrite_stamps(bytes: &[u8], old: &StableId, new: &StableId) -> Option<Vec<u8>> {
    let old_bytes = old.as_str().as_bytes();
    let new_bytes = new.as_str().as_bytes();
    debug_assert_eq!(old_bytes.len(), new_bytes.len());

    let mut result: Option<Vec<u8>> = None;
    let mut i = 0;
    while i + old_bytes.len() <= bytes.len() {
        if &bytes[i..i + old_bytes.len()] == old_bytes {
            let prior_ok = i == 0 || !is_word_byte(bytes[i - 1]);
            let trail_ok = bytes
                .get(i + old_bytes.len())
                .map_or(true, |c| !is_word_byte(*c));
            if prior_ok && trail_ok {
                let buf = result.get_or_insert_with(|| bytes.to_vec());
                buf[i..i + old_bytes.len()].copy_from_slice(new_bytes);
                i += old_bytes.len();
                continue;
            }
        }
        i += 1;
    }
    result
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn is_valid_stable_id(id: &StableId) -> bool {
    let s = id.as_str();
    s.len() == 11
        && s.starts_with("rq-")
        && s[3..]
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::materialize;
    use crate::migrate;
    use crate::object::Kind;

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

    // ── Stamp rewriting unit ──────────────────────────────────────────

    #[test]
    fn rewrite_stamps_basic_replacement() {
        let old = StableId::new("rq-aaaaaaaa");
        let new = StableId::new("rq-bbbbbbbb");
        let bytes = b"// rq-aaaaaaaa\nfn foo() {}\n";
        let out = rewrite_stamps(bytes, &old, &new).unwrap();
        assert_eq!(out, b"// rq-bbbbbbbb\nfn foo() {}\n");
    }

    #[test]
    fn rewrite_stamps_multiple_occurrences() {
        let old = StableId::new("rq-aaaaaaaa");
        let new = StableId::new("rq-bbbbbbbb");
        let bytes = b"// rq-aaaaaaaa other rq-aaaaaaaa end";
        let out = rewrite_stamps(bytes, &old, &new).unwrap();
        assert_eq!(out, b"// rq-bbbbbbbb other rq-bbbbbbbb end");
    }

    #[test]
    fn rewrite_stamps_word_boundary_respect() {
        let old = StableId::new("rq-aaaaaaaa");
        let new = StableId::new("rq-bbbbbbbb");
        // Surrounded by word characters — should NOT match.
        let bytes = b"xrq-aaaaaaaay";
        assert!(rewrite_stamps(bytes, &old, &new).is_none());
    }

    #[test]
    fn rewrite_stamps_no_match_returns_none() {
        let old = StableId::new("rq-aaaaaaaa");
        let new = StableId::new("rq-bbbbbbbb");
        let bytes = b"// rq-cccccccc\nfn foo() {}\n";
        assert!(rewrite_stamps(bytes, &old, &new).is_none());
    }

    // ── End-to-end rename ────────────────────────────────────────────

    #[test]
    fn rename_updates_meta_stable_id_and_ref() {
        let (_dir, store) = setup();
        make_req(&store, "rq-aaaaaaaa");
        let outcome = do_rename(
            &store,
            &StableId::new("rq-aaaaaaaa"),
            &StableId::new("rq-99999999"),
        )
        .unwrap();
        assert_eq!(outcome.new.to_string(), "rq-99999999");

        // Old ref gone; new ref points at a meta with the new stable_id.
        assert!(store.ref_get(&StableId::new("rq-aaaaaaaa")).unwrap().is_none());
        let h = store.ref_get(&StableId::new("rq-99999999")).unwrap().unwrap();
        let m = store.read_requirement(&h).unwrap();
        assert_eq!(m.stable_id, StableId::new("rq-99999999"));
    }

    #[test]
    fn rename_updates_child_parents() {
        let (_dir, store) = setup();
        // Parent rq-aaaaaaaa, child rq-bbbbbbbb with parents=[rq-aaaaaaaa].
        make_req(&store, "rq-aaaaaaaa");
        let text = store.write_blob(&Blob(b"prose".to_vec())).unwrap();
        let child = Requirement {
            stable_id: StableId::new("rq-bbbbbbbb"),
            kind: Kind::Behavior,
            text_blob: text,
            parents: vec![StableId::new("rq-aaaaaaaa")],
            source_blobs: vec![],
        };
        let h = store.write_requirement(&child).unwrap();
        store.ref_set(&StableId::new("rq-bbbbbbbb"), &h).unwrap();

        do_rename(
            &store,
            &StableId::new("rq-aaaaaaaa"),
            &StableId::new("rq-99999999"),
        )
        .unwrap();

        // The child's parents list now references the new id.
        let h = store.ref_get(&StableId::new("rq-bbbbbbbb")).unwrap().unwrap();
        let m = store.read_requirement(&h).unwrap();
        assert_eq!(m.parents, vec![StableId::new("rq-99999999")]);
    }

    #[test]
    fn rename_rewrites_stamps_in_text_and_source_blobs() {
        let (dir, store) = setup();
        let root = dir.path();
        // Markdown migration creates a text_blob whose first line carries
        // the rq stamp; source migration creates a source_blob with a
        // `// rq-...` comment. Both should be rewritten.
        let md = "# Feature <!-- rq-aaaaaaaa -->\n\nBody.\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        let src = "// rq-aaaaaaaa\nfn foo() {}\n";
        write_at(root, "src/foo.rs", src);
        migrate::migrate_source(&store, root, Path::new("src/foo.rs")).unwrap();

        let outcome = do_rename(
            &store,
            &StableId::new("rq-aaaaaaaa"),
            &StableId::new("rq-99999999"),
        )
        .unwrap();
        assert!(outcome.blobs_rewritten >= 2);

        // After build, the on-disk files reference the new id.
        materialize::build(&store, root).unwrap();
        let md_on_disk = fs::read_to_string(root.join("rqm/doc.md")).unwrap();
        let src_on_disk = fs::read_to_string(root.join("src/foo.rs")).unwrap();
        assert!(md_on_disk.contains("rq-99999999"));
        assert!(!md_on_disk.contains("rq-aaaaaaaa"));
        assert!(src_on_disk.contains("rq-99999999"));
        assert!(!src_on_disk.contains("rq-aaaaaaaa"));
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn rename_updates_file_tree_entry_stable_id() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        do_rename(
            &store,
            &StableId::new("rq-aaaaaaaa"),
            &StableId::new("rq-99999999"),
        )
        .unwrap();
        // The file-tree's entry now uses the new id.
        let th = store.tree_get(Path::new("rqm/doc.md")).unwrap().unwrap();
        let t = store.read_file_tree(&th).unwrap();
        assert_eq!(t.entries[0].stable_id, StableId::new("rq-99999999"));
    }

    #[test]
    fn rename_redirects_aliases() {
        let (dir, store) = setup();
        let root = dir.path();
        // Bullet alias rq-bbbbbbbb points at the heading rq-aaaaaaaa.
        let md = "# Top <!-- rq-aaaaaaaa -->\n\n- bullet <!-- rq-bbbbbbbb -->\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();

        let outcome = do_rename(
            &store,
            &StableId::new("rq-aaaaaaaa"),
            &StableId::new("rq-99999999"),
        )
        .unwrap();
        assert!(
            outcome
                .aliases_updated
                .iter()
                .any(|a| a == &StableId::new("rq-bbbbbbbb"))
        );
        // The alias now points at the new canonical.
        let canon = store.alias_get(&StableId::new("rq-bbbbbbbb")).unwrap().unwrap();
        assert_eq!(canon, StableId::new("rq-99999999"));
    }

    // ── Validation ────────────────────────────────────────────────────

    #[test]
    fn rename_rejects_unknown_target() {
        let (_dir, store) = setup();
        let err = do_rename(
            &store,
            &StableId::new("rq-99999999"),
            &StableId::new("rq-aaaaaaaa"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown stable_id"), "{err}");
    }

    #[test]
    fn rename_rejects_alias_target() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n\n- bullet <!-- rq-bbbbbbbb -->\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        let err = do_rename(
            &store,
            &StableId::new("rq-bbbbbbbb"),
            &StableId::new("rq-99999999"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("alias for"), "{err}");
    }

    #[test]
    fn rename_rejects_invalid_new_id() {
        let (_dir, store) = setup();
        make_req(&store, "rq-aaaaaaaa");
        for bad in ["rq-TOO_SHORT", "rq-XXXXXXXX", "not-an-id", "rq-tooooolong"] {
            let err = do_rename(
                &store,
                &StableId::new("rq-aaaaaaaa"),
                &StableId::new(bad),
            )
            .unwrap_err();
            assert!(err.to_string().contains("invalid new stable_id"), "for {bad}: {err}");
        }
    }

    #[test]
    fn rename_rejects_existing_canonical() {
        let (_dir, store) = setup();
        make_req(&store, "rq-aaaaaaaa");
        make_req(&store, "rq-bbbbbbbb");
        let err = do_rename(
            &store,
            &StableId::new("rq-aaaaaaaa"),
            &StableId::new("rq-bbbbbbbb"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("already exists as a canonical"), "{err}");
    }

    #[test]
    fn rename_rejects_existing_alias() {
        let (dir, store) = setup();
        let root = dir.path();
        // rq-bbbbbbbb is registered as an alias.
        let md = "# Top <!-- rq-aaaaaaaa -->\n\n- bullet <!-- rq-bbbbbbbb -->\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        // Create a separate canonical to try renaming TO rq-bbbbbbbb.
        make_req(&store, "rq-cccccccc");
        let err = do_rename(
            &store,
            &StableId::new("rq-cccccccc"),
            &StableId::new("rq-bbbbbbbb"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("already exists as an alias"), "{err}");
    }

    #[test]
    fn rename_noop_when_old_equals_new() {
        let (_dir, store) = setup();
        make_req(&store, "rq-aaaaaaaa");
        let outcome = do_rename(
            &store,
            &StableId::new("rq-aaaaaaaa"),
            &StableId::new("rq-aaaaaaaa"),
        )
        .unwrap();
        assert!(outcome.metas_updated.is_empty());
        assert_eq!(outcome.blobs_rewritten, 0);
    }

    #[test]
    fn rename_preserves_paths_updated_list() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        let outcome = do_rename(
            &store,
            &StableId::new("rq-aaaaaaaa"),
            &StableId::new("rq-99999999"),
        )
        .unwrap();
        assert_eq!(outcome.paths_updated, vec![PathBuf::from("rqm/doc.md")]);
    }
}
