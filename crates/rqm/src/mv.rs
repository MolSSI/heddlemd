//! `rqm move` — relocate a blob to a different position, either within
//! its current file or in another managed file.
//!
//! A move is a file-tree-only operation. The blob's bytes, its content
//! hash, and the owning requirement's `source_blobs` are unchanged; only
//! the entry's *position* in some file-tree changes.
//!
//! The `--split` modifier additionally subdivides the destination's
//! anchor blob at the requested line and inserts the moved blob between
//! the two halves. Splitting touches the meta(s) that own the anchor:
//! the anchor's blob hash in `source_blobs` is replaced with two new
//! hashes (one per half).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::object::{Blob, FileTree, FileTreeEntry, ObjectHash, Requirement, StableId};
use crate::store::Store;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modifier {
    /// Insert after the blob containing the destination line. Default.
    After,
    /// Insert before the blob containing the destination line.
    Before,
    /// Split the destination blob at the requested line; insert the
    /// moved blob between the two halves.
    Split,
}

#[derive(Debug, Clone)]
pub struct SourceSpec {
    pub path: PathBuf,
    pub line: usize,
}

impl SourceSpec {
    pub fn parse(s: &str) -> Result<Self> {
        let (path, line) = s.rsplit_once(':').ok_or_else(|| {
            anyhow::anyhow!("source must be <path>:<line>, got {s:?}")
        })?;
        let line: usize = line
            .parse()
            .with_context(|| format!("invalid line number in source {s:?}"))?;
        if line == 0 {
            bail!("line numbers are 1-based");
        }
        if path.is_empty() {
            bail!("empty path in {s:?}");
        }
        Ok(SourceSpec {
            path: PathBuf::from(path),
            line,
        })
    }
}

#[derive(Debug, Clone)]
pub enum DestSpec {
    Line {
        path: PathBuf,
        line: usize,
        modifier: Modifier,
    },
    Start {
        path: PathBuf,
    },
    End {
        path: PathBuf,
    },
}

impl DestSpec {
    pub fn parse(s: &str, modifier: Modifier) -> Result<Self> {
        let (path, anchor) = s.rsplit_once(':').ok_or_else(|| {
            anyhow::anyhow!(
                "dest must be <path>:<line|start|end>, got {s:?}"
            )
        })?;
        if path.is_empty() {
            bail!("empty path in {s:?}");
        }
        if anchor == "start" {
            if modifier != Modifier::After {
                bail!("--before/--split are not compatible with :start");
            }
            return Ok(DestSpec::Start {
                path: PathBuf::from(path),
            });
        }
        if anchor == "end" {
            if modifier != Modifier::After {
                bail!("--before/--split are not compatible with :end");
            }
            return Ok(DestSpec::End {
                path: PathBuf::from(path),
            });
        }
        let line: usize = anchor
            .parse()
            .with_context(|| format!("invalid line/anchor in dest {s:?}"))?;
        if line == 0 {
            bail!("line numbers are 1-based");
        }
        Ok(DestSpec::Line {
            path: PathBuf::from(path),
            line,
            modifier,
        })
    }

    pub fn path(&self) -> &Path {
        match self {
            DestSpec::Line { path, .. } => path,
            DestSpec::Start { path } => path,
            DestSpec::End { path } => path,
        }
    }
}

#[derive(Debug)]
pub struct MoveOutcome {
    pub moved_blob: ObjectHash,
    pub paths_updated: Vec<PathBuf>,
    pub split: Option<SplitInfo>,
}

#[derive(Debug)]
pub struct SplitInfo {
    pub original: ObjectHash,
    pub left: ObjectHash,
    pub right: ObjectHash,
    pub metas_updated: Vec<StableId>,
}

pub fn do_move(store: &Store, src: SourceSpec, dst: DestSpec) -> Result<MoveOutcome> {
    let src_path = src.path.clone();
    let dst_path = dst.path().to_path_buf();

    // Both paths must be managed.
    let src_tree_hash = store.tree_get(&src_path)?.ok_or_else(|| {
        anyhow::anyhow!(
            "source path {} is not managed in .rqm/",
            src_path.display()
        )
    })?;
    let dst_tree_hash = store.tree_get(&dst_path)?.ok_or_else(|| {
        anyhow::anyhow!(
            "dest path {} is not managed in .rqm/",
            dst_path.display()
        )
    })?;
    let src_tree = store.read_file_tree(&src_tree_hash)?;
    let dst_tree = if src_path == dst_path {
        src_tree.clone()
    } else {
        store.read_file_tree(&dst_tree_hash)?
    };

    // Resolve source: which entry covers src.line.
    let (src_index, _) = find_entry_at_line(store, &src_tree, src.line).with_context(|| {
        format!("resolving source {}:{}", src_path.display(), src.line)
    })?;
    let moved_entry = src_tree.entries[src_index].clone();

    // Resolve destination.
    let resolved_dst = resolve_dst(store, &dst_tree, &dst).with_context(|| {
        format!("resolving destination in {}", dst_path.display())
    })?;

    // Reject move-onto-self for non-split cases.
    if src_path == dst_path {
        if let ResolvedDst::AtIndex(idx) = resolved_dst {
            // After removal, the moved entry would land at the same spot
            // it was already at iff idx == src_index || idx == src_index + 1.
            if idx == src_index || idx == src_index + 1 {
                bail!("source and destination resolve to the same position; nothing to move");
            }
        }
        if let ResolvedDst::Split { anchor_index, .. } = &resolved_dst {
            if *anchor_index == src_index {
                bail!("cannot split the blob being moved");
            }
        }
    }

    if src_path == dst_path {
        apply_same_file(store, &src_path, src_tree, src_index, moved_entry, resolved_dst)
    } else {
        apply_cross_file(
            store,
            &src_path,
            src_tree,
            src_index,
            &dst_path,
            dst_tree,
            moved_entry,
            resolved_dst,
        )
    }
}

#[derive(Debug, Clone)]
enum ResolvedDst {
    /// Insert at this index in the destination file-tree (entries length
    /// inclusive = append at end).
    AtIndex(usize),
    /// Split the entry at `anchor_index` into two blobs, with the byte
    /// boundary at `split_offset` bytes into the anchor blob; insert the
    /// moved blob at the split.
    Split {
        anchor_index: usize,
        split_offset: usize,
    },
}

fn resolve_dst(store: &Store, tree: &FileTree, dst: &DestSpec) -> Result<ResolvedDst> {
    match dst {
        DestSpec::Start { .. } => Ok(ResolvedDst::AtIndex(0)),
        DestSpec::End { .. } => Ok(ResolvedDst::AtIndex(tree.entries.len())),
        DestSpec::Line {
            line, modifier, ..
        } => {
            let (idx, byte_offset_within) =
                find_entry_at_line_with_offset(store, tree, *line)?;
            match modifier {
                Modifier::After => Ok(ResolvedDst::AtIndex(idx + 1)),
                Modifier::Before => Ok(ResolvedDst::AtIndex(idx)),
                Modifier::Split => Ok(ResolvedDst::Split {
                    anchor_index: idx,
                    split_offset: byte_offset_within,
                }),
            }
        }
    }
}

fn apply_cross_file(
    store: &Store,
    src_path: &Path,
    src_tree: FileTree,
    src_index: usize,
    dst_path: &Path,
    dst_tree: FileTree,
    moved_entry: FileTreeEntry,
    resolved_dst: ResolvedDst,
) -> Result<MoveOutcome> {
    // Build new source tree: remove src.
    let mut new_src_entries = src_tree.entries;
    new_src_entries.remove(src_index);

    // Build new dest tree: insert (or insert+split).
    let mut new_dst_entries = dst_tree.entries;
    let mut split_info = None;
    match resolved_dst {
        ResolvedDst::AtIndex(idx) => {
            new_dst_entries.insert(idx, moved_entry.clone());
        }
        ResolvedDst::Split {
            anchor_index,
            split_offset,
        } => {
            let anchor = new_dst_entries[anchor_index].clone();
            let (left, right, info) =
                perform_split(store, &anchor, split_offset)?;
            new_dst_entries[anchor_index] = left;
            new_dst_entries.insert(anchor_index + 1, moved_entry.clone());
            new_dst_entries.insert(anchor_index + 2, right);
            split_info = Some(info);
        }
    }

    // Write both trees.
    let new_src_tree = FileTree {
        path: src_path.to_path_buf(),
        entries: new_src_entries,
    };
    let new_dst_tree = FileTree {
        path: dst_path.to_path_buf(),
        entries: new_dst_entries,
    };
    let src_hash = store.write_file_tree(&new_src_tree)?;
    let dst_hash = store.write_file_tree(&new_dst_tree)?;
    store.tree_set(src_path, &src_hash)?;
    store.tree_set(dst_path, &dst_hash)?;

    Ok(MoveOutcome {
        moved_blob: moved_entry.blob,
        paths_updated: vec![src_path.to_path_buf(), dst_path.to_path_buf()],
        split: split_info,
    })
}

fn apply_same_file(
    store: &Store,
    path: &Path,
    tree: FileTree,
    src_index: usize,
    moved_entry: FileTreeEntry,
    resolved_dst: ResolvedDst,
) -> Result<MoveOutcome> {
    let mut entries = tree.entries;
    let mut split_info = None;

    match resolved_dst {
        ResolvedDst::AtIndex(dst_idx) => {
            // Remove first; then insert (adjusting the dest index if it
            // pointed past the removal).
            entries.remove(src_index);
            let insert_at = if dst_idx > src_index { dst_idx - 1 } else { dst_idx };
            entries.insert(insert_at, moved_entry.clone());
        }
        ResolvedDst::Split {
            anchor_index,
            split_offset,
        } => {
            // 1. Split the anchor blob.
            let anchor = entries[anchor_index].clone();
            let (left, right, info) =
                perform_split(store, &anchor, split_offset)?;
            split_info = Some(info);
            // 2. Replace anchor with [left, right]. This shifts later
            //    indices by +1.
            entries[anchor_index] = left;
            entries.insert(anchor_index + 1, right);
            let src_after_split =
                if src_index > anchor_index { src_index + 1 } else { src_index };
            // 3. Remove the moved entry at its (possibly shifted) index.
            entries.remove(src_after_split);
            // 4. Insert moved entry between left and right. Left is still
            //    at anchor_index (unchanged by the removal unless src was
            //    before anchor; in that case left shifted to anchor_index-1).
            let left_index_after_removal = if src_index < anchor_index {
                anchor_index - 1
            } else {
                anchor_index
            };
            entries.insert(left_index_after_removal + 1, moved_entry.clone());
        }
    }

    let new_tree = FileTree {
        path: path.to_path_buf(),
        entries,
    };
    let tree_hash = store.write_file_tree(&new_tree)?;
    store.tree_set(path, &tree_hash)?;

    Ok(MoveOutcome {
        moved_blob: moved_entry.blob,
        paths_updated: vec![path.to_path_buf()],
        split: split_info,
    })
}

/// Split `anchor`'s blob at `split_offset` bytes, write the two halves,
/// and update every meta that owned the original blob to reference the
/// two halves instead.
fn perform_split(
    store: &Store,
    anchor: &FileTreeEntry,
    split_offset: usize,
) -> Result<(FileTreeEntry, FileTreeEntry, SplitInfo)> {
    let original_hash = anchor.blob;
    let blob = store.read_blob(&original_hash)?;
    if split_offset > blob.0.len() {
        bail!(
            "split offset {} out of range (anchor blob is {} bytes)",
            split_offset,
            blob.0.len()
        );
    }
    let left_bytes = blob.0[..split_offset].to_vec();
    let right_bytes = blob.0[split_offset..].to_vec();
    let left_hash = store.write_blob(&Blob(left_bytes))?;
    let right_hash = store.write_blob(&Blob(right_bytes))?;

    // Update every meta that has `original_hash` in source_blobs (or
    // text_blob, defensively) to reference left_hash and right_hash
    // instead. This handles joint ownership.
    let mut metas_updated = Vec::new();
    for (sid, mhash) in store.ref_list()? {
        let mut meta: Requirement = store.read_requirement(&mhash)?;
        // Skip alias refs — the canonical will be visited separately.
        if meta.stable_id != sid {
            continue;
        }
        let mut changed = false;
        if meta.text_blob == original_hash {
            // Defensive: a text-blob being split would be unusual but
            // mechanically valid. Replace with left; the right half
            // becomes an additional source-blob.
            meta.text_blob = left_hash;
            meta.source_blobs.push(right_hash);
            changed = true;
        }
        let mut new_source = Vec::with_capacity(meta.source_blobs.len() + 1);
        for h in &meta.source_blobs {
            if *h == original_hash {
                new_source.push(left_hash);
                new_source.push(right_hash);
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
            store.ref_set(&sid, &new_mhash)?;
            metas_updated.push(sid);
        }
    }

    let left_entry = FileTreeEntry {
        stable_id: anchor.stable_id.clone(),
        blob: left_hash,
    };
    let right_entry = FileTreeEntry {
        stable_id: anchor.stable_id.clone(),
        blob: right_hash,
    };
    Ok((
        left_entry,
        right_entry,
        SplitInfo {
            original: original_hash,
            left: left_hash,
            right: right_hash,
            metas_updated,
        },
    ))
}

/// Find the file-tree entry index whose blob covers `line`. Returns the
/// index and a copy of the entry.
fn find_entry_at_line(
    store: &Store,
    tree: &FileTree,
    line: usize,
) -> Result<(usize, FileTreeEntry)> {
    let (idx, _) = find_entry_at_line_with_offset(store, tree, line)?;
    Ok((idx, tree.entries[idx].clone()))
}

/// Find the file-tree entry index covering `line`, and the byte offset
/// of the start of `line` within that entry's blob.
fn find_entry_at_line_with_offset(
    store: &Store,
    tree: &FileTree,
    line: usize,
) -> Result<(usize, usize)> {
    if tree.entries.is_empty() {
        bail!("{} has no blobs", tree.path.display());
    }
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
                "line {} out of range (file has {} line(s))",
                line,
                newlines_seen + 1
            )
        })?
    };

    for (i, _) in tree.entries.iter().enumerate() {
        if target_offset >= blob_starts[i] && target_offset < blob_starts[i + 1] {
            return Ok((i, target_offset - blob_starts[i]));
        }
    }
    // EOF case: attribute to the last blob, at its end.
    let last = tree.entries.len() - 1;
    Ok((last, blob_starts[last + 1] - blob_starts[last]))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::*;
    use crate::materialize;
    use crate::migrate;
    use crate::object::{Blob, Kind, Requirement, StableId};

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

    fn setup_two_source_files() -> (tempfile::TempDir, Store) {
        let (dir, store) = setup();
        let root = dir.path();
        for id in ["rq-aaaaaaaa", "rq-bbbbbbbb", "rq-cccccccc"] {
            make_req(&store, id);
        }
        write_at(
            root,
            "src/source.rs",
            "// rq-aaaaaaaa\nfn alpha() {}\n\n// rq-bbbbbbbb\nfn beta() {}\n",
        );
        write_at(root, "src/dest.rs", "// rq-cccccccc\nfn gamma() {}\n");
        migrate::migrate_source(&store, root, Path::new("src/source.rs")).unwrap();
        migrate::migrate_source(&store, root, Path::new("src/dest.rs")).unwrap();
        (dir, store)
    }

    // ── Parsing ───────────────────────────────────────────────────────

    #[test]
    fn source_spec_parses() {
        let s = SourceSpec::parse("src/foo.rs:42").unwrap();
        assert_eq!(s.path, PathBuf::from("src/foo.rs"));
        assert_eq!(s.line, 42);
        assert!(SourceSpec::parse("no-colon").is_err());
        assert!(SourceSpec::parse("foo.rs:0").is_err());
    }

    #[test]
    fn dest_spec_parses() {
        match DestSpec::parse("src/foo.rs:end", Modifier::After).unwrap() {
            DestSpec::End { path } => assert_eq!(path, PathBuf::from("src/foo.rs")),
            other => panic!("{other:?}"),
        }
        match DestSpec::parse("src/foo.rs:start", Modifier::After).unwrap() {
            DestSpec::Start { path } => assert_eq!(path, PathBuf::from("src/foo.rs")),
            other => panic!("{other:?}"),
        }
        match DestSpec::parse("src/foo.rs:42", Modifier::Split).unwrap() {
            DestSpec::Line { line, modifier, .. } => {
                assert_eq!(line, 42);
                assert_eq!(modifier, Modifier::Split);
            }
            other => panic!("{other:?}"),
        }
        // start/end disallow non-After modifiers
        assert!(DestSpec::parse("src/foo.rs:start", Modifier::Split).is_err());
        assert!(DestSpec::parse("src/foo.rs:end", Modifier::Before).is_err());
    }

    // ── Cross-file move ───────────────────────────────────────────────

    #[test]
    fn cross_file_move_after() {
        let (dir, store) = setup_two_source_files();
        let root = dir.path();
        // Move beta() from source.rs to after gamma() in dest.rs.
        let src = SourceSpec::parse("src/source.rs:5").unwrap();
        let dst = DestSpec::parse("src/dest.rs:1", Modifier::After).unwrap();
        let outcome = do_move(&store, src, dst).unwrap();
        assert_eq!(outcome.paths_updated.len(), 2);

        materialize::build(&store, root).unwrap();
        let source = fs::read_to_string(root.join("src/source.rs")).unwrap();
        let dest = fs::read_to_string(root.join("src/dest.rs")).unwrap();
        assert!(source.contains("rq-aaaaaaaa"));
        assert!(!source.contains("rq-bbbbbbbb"));
        assert!(dest.contains("rq-cccccccc"));
        assert!(dest.contains("rq-bbbbbbbb"));
        let gamma = dest.find("rq-cccccccc").unwrap();
        let beta = dest.find("rq-bbbbbbbb").unwrap();
        assert!(beta > gamma, "beta should come after gamma:\n{dest}");

        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn cross_file_move_to_start() {
        let (dir, store) = setup_two_source_files();
        let root = dir.path();
        let src = SourceSpec::parse("src/source.rs:5").unwrap();
        let dst = DestSpec::parse("src/dest.rs:start", Modifier::After).unwrap();
        do_move(&store, src, dst).unwrap();
        materialize::build(&store, root).unwrap();
        let dest = fs::read_to_string(root.join("src/dest.rs")).unwrap();
        let beta_at = dest.find("rq-bbbbbbbb").unwrap();
        let gamma_at = dest.find("rq-cccccccc").unwrap();
        assert!(beta_at < gamma_at, "beta should come before gamma:\n{dest}");
    }

    // ── Same-file move ────────────────────────────────────────────────

    #[test]
    fn same_file_reorder() {
        let (dir, store) = setup();
        let root = dir.path();
        for id in ["rq-aaaaaaaa", "rq-bbbbbbbb", "rq-cccccccc"] {
            make_req(&store, id);
        }
        let src = "\
// rq-aaaaaaaa
fn alpha() {}

// rq-bbbbbbbb
fn beta() {}

// rq-cccccccc
fn gamma() {}
";
        write_at(root, "src/x.rs", src);
        migrate::migrate_source(&store, root, Path::new("src/x.rs")).unwrap();

        // Move gamma to before alpha.
        let outcome = do_move(
            &store,
            SourceSpec::parse("src/x.rs:8").unwrap(),
            DestSpec::parse("src/x.rs:1", Modifier::Before).unwrap(),
        )
        .unwrap();
        assert_eq!(outcome.paths_updated, vec![PathBuf::from("src/x.rs")]);
        materialize::build(&store, root).unwrap();
        let result = fs::read_to_string(root.join("src/x.rs")).unwrap();
        let g = result.find("gamma").unwrap();
        let a = result.find("alpha").unwrap();
        let b = result.find("beta").unwrap();
        assert!(g < a && a < b, "expected gamma < alpha < beta:\n{result}");
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn same_file_noop_rejected() {
        let (dir, store) = setup();
        let root = dir.path();
        make_req(&store, "rq-aaaaaaaa");
        make_req(&store, "rq-bbbbbbbb");
        write_at(
            root,
            "src/x.rs",
            "// rq-aaaaaaaa\nfn a() {}\n\n// rq-bbbbbbbb\nfn b() {}\n",
        );
        migrate::migrate_source(&store, root, Path::new("src/x.rs")).unwrap();
        // "Move" the second blob to after itself.
        let err = do_move(
            &store,
            SourceSpec::parse("src/x.rs:4").unwrap(),
            DestSpec::parse("src/x.rs:4", Modifier::After).unwrap(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("same position"), "{err}");
    }

    // ── --split ───────────────────────────────────────────────────────

    #[test]
    fn cross_file_split_inserts_between_halves() {
        let (dir, store) = setup_two_source_files();
        let root = dir.path();
        // Split dest.rs at line 2 (the body of gamma); insert beta there.
        let src = SourceSpec::parse("src/source.rs:5").unwrap();
        let dst = DestSpec::parse("src/dest.rs:2", Modifier::Split).unwrap();
        let outcome = do_move(&store, src, dst).unwrap();
        assert!(outcome.split.is_some(), "expected split info");
        let split = outcome.split.as_ref().unwrap();
        // rq-cccccccc owned the original blob; its source_blobs should now
        // contain the two halves.
        assert!(
            split.metas_updated.contains(&StableId::new("rq-cccccccc")),
            "{:?}",
            split.metas_updated
        );

        materialize::build(&store, root).unwrap();
        let dest = fs::read_to_string(root.join("src/dest.rs")).unwrap();
        // beta should appear between the stamp and body of gamma.
        let gamma_stamp = dest.find("rq-cccccccc").unwrap();
        let beta = dest.find("rq-bbbbbbbb").unwrap();
        let gamma_body = dest.find("fn gamma()").unwrap();
        assert!(
            gamma_stamp < beta && beta < gamma_body,
            "expected stamp < moved < body:\n{dest}"
        );
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn split_updates_joint_ownership_metas() {
        let (dir, store) = setup();
        let root = dir.path();
        for id in ["rq-aaaaaaaa", "rq-bbbbbbbb", "rq-cccccccc", "rq-dddddddd"] {
            make_req(&store, id);
        }
        // Jointly-owned anchor blob (rq-bbbbbbbb + rq-cccccccc).
        let src = "\
// rq-aaaaaaaa
fn a() {}

// rq-bbbbbbbb rq-cccccccc
fn shared() {
    let x = 1;
}

// rq-dddddddd
fn d() {}
";
        write_at(root, "src/x.rs", src);
        migrate::migrate_source(&store, root, Path::new("src/x.rs")).unwrap();

        // Move d() so it splits the shared blob at its body line.
        let outcome = do_move(
            &store,
            SourceSpec::parse("src/x.rs:9").unwrap(),
            DestSpec::parse("src/x.rs:6", Modifier::Split).unwrap(),
        )
        .unwrap();

        let split = outcome.split.expect("expected split");
        let mut ids: Vec<String> =
            split.metas_updated.iter().map(|s| s.to_string()).collect();
        ids.sort();
        assert_eq!(ids, vec!["rq-bbbbbbbb", "rq-cccccccc"]);
        materialize::build(&store, root).unwrap();
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn cannot_split_blob_being_moved() {
        let (dir, store) = setup();
        let root = dir.path();
        make_req(&store, "rq-aaaaaaaa");
        write_at(
            root,
            "src/x.rs",
            "// rq-aaaaaaaa\nfn a() {\n    let x = 1;\n}\n",
        );
        migrate::migrate_source(&store, root, Path::new("src/x.rs")).unwrap();
        // Try to "move" the only blob to split itself.
        let err = do_move(
            &store,
            SourceSpec::parse("src/x.rs:1").unwrap(),
            DestSpec::parse("src/x.rs:3", Modifier::Split).unwrap(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("split the blob being moved"),
            "{err}"
        );
    }
}
