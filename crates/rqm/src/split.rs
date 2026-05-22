//! `rqm split` — divide a blob at a line, optionally re-attributing
//! either half to different owners.
//!
//! `rqm split <file>:<line>` finds the blob containing the line and
//! divides it at the line boundary. The left half is everything before
//! that line; the right half is the line itself and everything after.
//! Each half can independently be re-attributed via `--left <owner>...`
//! and `--right <owner>...`. Omitting a side flag preserves the
//! original blob's owners on that half.
//!
//! This is the primary mechanism for taking a coarse-grained source
//! blob (e.g. a whole file attributed to the file-root requirement)
//! and slicing it into smaller blobs attributed to specific child
//! requirements.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::mv::find_entry_at_line_with_offset;
use crate::object::{Blob, FileTreeEntry, ObjectHash, Requirement, StableId};
use crate::store::Store;

#[derive(Debug, Clone)]
pub struct SplitSpec {
    pub path: PathBuf,
    pub line: usize,
}

impl SplitSpec {
    pub fn parse(s: &str) -> Result<Self> {
        let (path, line) = s.rsplit_once(':').ok_or_else(|| {
            anyhow::anyhow!("target must be <path>:<line>, got {s:?}")
        })?;
        if path.is_empty() {
            bail!("empty path in {s:?}");
        }
        let line: usize = line
            .parse()
            .with_context(|| format!("invalid line in {s:?}"))?;
        if line == 0 {
            bail!("line numbers are 1-based");
        }
        Ok(SplitSpec {
            path: PathBuf::from(path),
            line,
        })
    }
}

#[derive(Debug)]
pub struct SplitOutcome {
    pub original_blob: ObjectHash,
    pub left_blob: ObjectHash,
    pub right_blob: ObjectHash,
    pub left_owners: Vec<StableId>,
    pub right_owners: Vec<StableId>,
    /// Metas whose source_blobs was rewritten.
    pub metas_updated: Vec<StableId>,
    pub path: PathBuf,
}

/// Split the blob at the spec'd file:line. `left_owners` /
/// `right_owners` are optional re-attributions; pass `None` to keep the
/// original blob's owners on that side. Owners must be canonical
/// stable_ids that already exist (the CLI layer is expected to resolve
/// any file:line cursors and reject aliases first).
pub fn do_split(
    store: &Store,
    spec: SplitSpec,
    left_owners: Option<Vec<StableId>>,
    right_owners: Option<Vec<StableId>>,
) -> Result<SplitOutcome> {
    let tree_hash = store.tree_get(&spec.path)?.ok_or_else(|| {
        anyhow::anyhow!("{}: not a managed path in .rqm/", spec.path.display())
    })?;
    let mut tree = store.read_file_tree(&tree_hash)?;
    let (idx, byte_offset) = find_entry_at_line_with_offset(store, &tree, spec.line)?;
    let original_entry = tree.entries[idx].clone();
    let original_blob_hash = original_entry.blob;
    let original_blob = store.read_blob(&original_blob_hash)?;

    if byte_offset == 0 {
        bail!(
            "{}:{} is at the start of the blob (nothing to split off the left)",
            spec.path.display(),
            spec.line
        );
    }
    if byte_offset >= original_blob.0.len() {
        bail!(
            "{}:{} is at the end of the blob (nothing to split off the right)",
            spec.path.display(),
            spec.line
        );
    }

    // Find every meta currently owning the blob.
    let mut original_owners: HashSet<StableId> = HashSet::new();
    for (sid, mhash) in store.ref_list()? {
        let meta: Requirement = store.read_requirement(&mhash)?;
        if meta.stable_id != sid {
            continue;
        }
        if meta.source_blobs.contains(&original_blob_hash) {
            original_owners.insert(sid);
        }
    }

    // Validate any new owners are real canonical refs.
    for side in [left_owners.as_ref(), right_owners.as_ref()] {
        if let Some(owners) = side {
            for o in owners {
                if store.ref_get(o)?.is_none() {
                    bail!("unknown owner: {o}");
                }
            }
        }
    }

    let left_set: HashSet<StableId> = match &left_owners {
        Some(o) => o.iter().cloned().collect(),
        None => original_owners.clone(),
    };
    let right_set: HashSet<StableId> = match &right_owners {
        Some(o) => o.iter().cloned().collect(),
        None => original_owners.clone(),
    };

    // Write the new blobs.
    let left_bytes = original_blob.0[..byte_offset].to_vec();
    let right_bytes = original_blob.0[byte_offset..].to_vec();
    let left_hash = store.write_blob(&Blob(left_bytes))?;
    let right_hash = store.write_blob(&Blob(right_bytes))?;

    // Update every affected meta: the original owners need
    // `original_blob` removed; the left_set / right_set members need
    // the appropriate new hash.
    let mut affected: HashSet<StableId> = HashSet::new();
    affected.extend(original_owners.iter().cloned());
    affected.extend(left_set.iter().cloned());
    affected.extend(right_set.iter().cloned());

    let mut metas_updated = Vec::new();
    let mut sorted_affected: Vec<&StableId> = affected.iter().collect();
    sorted_affected.sort();
    for sid in sorted_affected {
        let h = store.ref_get(sid)?.expect("validated above");
        let mut meta: Requirement = store.read_requirement(&h)?;
        let original = meta.source_blobs.clone();
        let mut new_source: Vec<ObjectHash> = meta
            .source_blobs
            .iter()
            .copied()
            .filter(|h| *h != original_blob_hash)
            .collect();
        if left_set.contains(sid) && !new_source.contains(&left_hash) {
            new_source.push(left_hash);
        }
        if right_set.contains(sid) && !new_source.contains(&right_hash) {
            new_source.push(right_hash);
        }
        new_source.sort();
        new_source.dedup();
        if new_source != original {
            meta.source_blobs = new_source;
            let new_h = store.write_requirement(&meta)?;
            store.ref_set(sid, &new_h)?;
            metas_updated.push(sid.clone());
        }
    }

    // Decide each entry's stable_id. When a side was re-attributed,
    // use the first listed new owner. When unchanged, preserve the
    // original entry's stable_id.
    let left_entry_stable_id = match &left_owners {
        Some(o) => o.first().cloned().unwrap_or_else(|| original_entry.stable_id.clone()),
        None => original_entry.stable_id.clone(),
    };
    let right_entry_stable_id = match &right_owners {
        Some(o) => o.first().cloned().unwrap_or_else(|| original_entry.stable_id.clone()),
        None => original_entry.stable_id.clone(),
    };

    // Replace the original entry with two new entries.
    tree.entries[idx] = FileTreeEntry {
        stable_id: left_entry_stable_id,
        blob: left_hash,
    };
    tree.entries.insert(
        idx + 1,
        FileTreeEntry {
            stable_id: right_entry_stable_id,
            blob: right_hash,
        },
    );
    let new_tree_hash = store.write_file_tree(&tree)?;
    store.tree_set(&spec.path, &new_tree_hash)?;

    let mut left_owners_out: Vec<StableId> = left_set.into_iter().collect();
    left_owners_out.sort();
    let mut right_owners_out: Vec<StableId> = right_set.into_iter().collect();
    right_owners_out.sort();

    Ok(SplitOutcome {
        original_blob: original_blob_hash,
        left_blob: left_hash,
        right_blob: right_hash,
        left_owners: left_owners_out,
        right_owners: right_owners_out,
        metas_updated,
        path: spec.path,
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::*;
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
            parents: vec![],
            source_blobs: vec![],
        };
        let h = store.write_requirement(&req).unwrap();
        store.ref_set(&StableId::new(id), &h).unwrap();
    }

    fn setup_single_blob() -> (tempfile::TempDir, Store) {
        let (dir, store) = setup();
        let root = dir.path();
        make_req(&store, "rq-aaaaaaaa");
        // One blob, owned by rq-aaaaaaaa, spans 6 lines.
        let src = "// rq-aaaaaaaa\nfn one() {}\n\nfn two() {}\n\nfn three() {}\n";
        write_at(root, "src/x.rs", src);
        migrate::migrate_source(&store, root, Path::new("src/x.rs")).unwrap();
        (dir, store)
    }

    fn entries(store: &Store, path: &str) -> Vec<(String, ObjectHash)> {
        let h = store.tree_get(Path::new(path)).unwrap().unwrap();
        let t = store.read_file_tree(&h).unwrap();
        t.entries
            .into_iter()
            .map(|e| (e.stable_id.to_string(), e.blob))
            .collect()
    }

    // ── Parsing ───────────────────────────────────────────────────────

    #[test]
    fn parse_basic() {
        let s = SplitSpec::parse("src/x.rs:42").unwrap();
        assert_eq!(s.line, 42);
        assert!(SplitSpec::parse("no-colon").is_err());
        assert!(SplitSpec::parse("foo.rs:0").is_err());
        assert!(SplitSpec::parse(":42").is_err());
    }

    // ── Splits ────────────────────────────────────────────────────────

    #[test]
    fn split_preserves_ownership_when_no_owner_flags() {
        let (dir, store) = setup_single_blob();
        let root = dir.path();
        let outcome = do_split(
            &store,
            SplitSpec::parse("src/x.rs:3").unwrap(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(outcome.left_owners, vec![StableId::new("rq-aaaaaaaa")]);
        assert_eq!(outcome.right_owners, vec![StableId::new("rq-aaaaaaaa")]);

        // Two entries now; both belong to rq-aaaaaaaa.
        let es = entries(&store, "src/x.rs");
        assert_eq!(es.len(), 2);
        assert_eq!(es[0].0, "rq-aaaaaaaa");
        assert_eq!(es[1].0, "rq-aaaaaaaa");

        // Round-trip materializes identically.
        materialize::build(&store, root).unwrap();
        let on_disk = fs::read_to_string(root.join("src/x.rs")).unwrap();
        assert!(on_disk.contains("fn one"));
        assert!(on_disk.contains("fn two"));
        assert!(on_disk.contains("fn three"));
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn split_right_reattributes_right_half() {
        let (dir, store) = setup_single_blob();
        let root = dir.path();
        make_req(&store, "rq-99999999");
        let outcome = do_split(
            &store,
            SplitSpec::parse("src/x.rs:3").unwrap(),
            None,
            Some(vec![StableId::new("rq-99999999")]),
        )
        .unwrap();
        assert_eq!(outcome.right_owners, vec![StableId::new("rq-99999999")]);

        let es = entries(&store, "src/x.rs");
        assert_eq!(es.len(), 2);
        assert_eq!(es[0].0, "rq-aaaaaaaa");
        assert_eq!(es[1].0, "rq-99999999");

        // rq-aaaaaaaa should own only left now; rq-99999999 only right.
        let h = store.ref_get(&StableId::new("rq-aaaaaaaa")).unwrap().unwrap();
        let m = store.read_requirement(&h).unwrap();
        assert_eq!(m.source_blobs, vec![outcome.left_blob]);
        let h = store.ref_get(&StableId::new("rq-99999999")).unwrap().unwrap();
        let m = store.read_requirement(&h).unwrap();
        assert_eq!(m.source_blobs, vec![outcome.right_blob]);

        materialize::build(&store, root).unwrap();
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn split_left_reattributes_left_half() {
        let (dir, store) = setup_single_blob();
        let root = dir.path();
        make_req(&store, "rq-99999999");
        do_split(
            &store,
            SplitSpec::parse("src/x.rs:3").unwrap(),
            Some(vec![StableId::new("rq-99999999")]),
            None,
        )
        .unwrap();
        let es = entries(&store, "src/x.rs");
        assert_eq!(es.len(), 2);
        assert_eq!(es[0].0, "rq-99999999");
        assert_eq!(es[1].0, "rq-aaaaaaaa");
    }

    #[test]
    fn split_both_sides_reattributed() {
        let (dir, store) = setup_single_blob();
        let root = dir.path();
        make_req(&store, "rq-99999999");
        make_req(&store, "rq-88888888");
        do_split(
            &store,
            SplitSpec::parse("src/x.rs:3").unwrap(),
            Some(vec![StableId::new("rq-99999999")]),
            Some(vec![StableId::new("rq-88888888")]),
        )
        .unwrap();
        // rq-aaaaaaaa no longer owns anything in src/x.rs.
        let h = store.ref_get(&StableId::new("rq-aaaaaaaa")).unwrap().unwrap();
        let m = store.read_requirement(&h).unwrap();
        assert!(m.source_blobs.is_empty());

        let es = entries(&store, "src/x.rs");
        assert_eq!(es[0].0, "rq-99999999");
        assert_eq!(es[1].0, "rq-88888888");
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn split_right_with_joint_ownership() {
        let (dir, store) = setup();
        let root = dir.path();
        make_req(&store, "rq-aaaaaaaa");
        make_req(&store, "rq-bbbbbbbb");
        let src = "// rq-aaaaaaaa rq-bbbbbbbb\nfn one() {}\n\nfn two() {}\n";
        write_at(root, "src/x.rs", src);
        migrate::migrate_source(&store, root, Path::new("src/x.rs")).unwrap();
        make_req(&store, "rq-99999999");

        let outcome = do_split(
            &store,
            SplitSpec::parse("src/x.rs:3").unwrap(),
            None,
            Some(vec![StableId::new("rq-99999999")]),
        )
        .unwrap();

        // rq-aaaaaaaa and rq-bbbbbbbb own only left.
        let h_a = store.ref_get(&StableId::new("rq-aaaaaaaa")).unwrap().unwrap();
        let h_b = store.ref_get(&StableId::new("rq-bbbbbbbb")).unwrap().unwrap();
        assert_eq!(
            store.read_requirement(&h_a).unwrap().source_blobs,
            vec![outcome.left_blob]
        );
        assert_eq!(
            store.read_requirement(&h_b).unwrap().source_blobs,
            vec![outcome.left_blob]
        );
        // rq-99999999 owns only right.
        let h_n = store.ref_get(&StableId::new("rq-99999999")).unwrap().unwrap();
        assert_eq!(
            store.read_requirement(&h_n).unwrap().source_blobs,
            vec![outcome.right_blob]
        );
        materialize::build(&store, root).unwrap();
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn split_at_blob_start_rejected() {
        let (_dir, store) = setup_single_blob();
        let err = do_split(
            &store,
            SplitSpec::parse("src/x.rs:1").unwrap(),
            None,
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("start of the blob"), "{err}");
    }

    #[test]
    fn split_unknown_owner_rejected() {
        let (_dir, store) = setup_single_blob();
        let err = do_split(
            &store,
            SplitSpec::parse("src/x.rs:3").unwrap(),
            None,
            Some(vec![StableId::new("rq-99999999")]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown owner"), "{err}");
    }

    #[test]
    fn split_then_split_again_subdivides_further() {
        let (dir, store) = setup();
        let root = dir.path();
        make_req(&store, "rq-00000001");
        make_req(&store, "rq-00000002");
        make_req(&store, "rq-00000003");
        let src = "// rq-00000001\nfn one() {}\n\nfn two() {}\n\nfn three() {}\n";
        write_at(root, "src/x.rs", src);
        migrate::migrate_source(&store, root, Path::new("src/x.rs")).unwrap();

        // First split: lines 1-2 (rq-00000001), 3+ (rq-00000002).
        do_split(
            &store,
            SplitSpec::parse("src/x.rs:3").unwrap(),
            None,
            Some(vec![StableId::new("rq-00000002")]),
        )
        .unwrap();

        // After first split, materialize so line numbers reflect new state.
        materialize::build(&store, root).unwrap();

        // Second split inside the rq-00000002 blob: lines 3-4 (rq-00000002), 5+ (rq-00000003).
        do_split(
            &store,
            SplitSpec::parse("src/x.rs:5").unwrap(),
            None,
            Some(vec![StableId::new("rq-00000003")]),
        )
        .unwrap();

        let es = entries(&store, "src/x.rs");
        assert_eq!(es.len(), 3);
        assert_eq!(es[0].0, "rq-00000001");
        assert_eq!(es[1].0, "rq-00000002");
        assert_eq!(es[2].0, "rq-00000003");
        materialize::build(&store, root).unwrap();
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }
}
