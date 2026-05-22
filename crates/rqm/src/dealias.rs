//! `rqm dealias` — post-migration cleanup that removes bullet-level
//! alias stable_ids.
//!
//! For each alias `A` pointing at canonical `C`:
//!  * In markdown content (blobs that live in a `.md` file-tree), the
//!    ` <!-- rq-A -->` annotation is stripped from the bullet line.
//!    The parent heading already carries the canonical's annotation,
//!    so the bullet annotation is redundant.
//!  * In source content (blobs in non-`.md` file-trees), `rq-A` tokens
//!    are substituted with `rq-C`. This preserves source-side
//!    traceability comments while pointing them at the canonical.
//!  * The alias file at `.rqm/aliases/<A>` is deleted.
//!
//! All affected blobs are rewritten, metas pointing at them get new
//! hashes, and file-trees are updated to reference the new blobs.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use anyhow::Result;

use crate::object::{Blob, ObjectHash, Requirement, StableId};
use crate::store::Store;

#[derive(Debug)]
pub struct DealiasOutcome {
    pub aliases_removed: Vec<StableId>,
    pub blobs_rewritten: usize,
    pub paths_updated: Vec<PathBuf>,
    pub metas_updated: Vec<StableId>,
}

/// Dealias the given aliases (or every alias in the store if `subset`
/// is `None`).
pub fn do_dealias(
    store: &Store,
    subset: Option<Vec<StableId>>,
) -> Result<DealiasOutcome> {
    // Resolve the working set of (alias, canonical) pairs.
    let pairs: Vec<(StableId, StableId)> = match subset {
        Some(ids) => {
            let mut out = Vec::with_capacity(ids.len());
            for alias in ids {
                let canonical = store.alias_get(&alias)?.ok_or_else(|| {
                    anyhow::anyhow!("{alias} is not a registered alias")
                })?;
                out.push((alias, canonical));
            }
            out
        }
        None => store.alias_list()?,
    };

    if pairs.is_empty() {
        return Ok(DealiasOutcome {
            aliases_removed: Vec::new(),
            blobs_rewritten: 0,
            paths_updated: Vec::new(),
            metas_updated: Vec::new(),
        });
    }

    let alias_to_canonical: HashMap<StableId, StableId> = pairs.iter().cloned().collect();

    // Classify each reachable blob as markdown (lives in any `.md`
    // file-tree) or source (otherwise).
    let blob_kinds = classify_reachable_blobs(store)?;

    // Rewrite blobs that need it. Build a remap.
    let mut blob_remap: HashMap<ObjectHash, ObjectHash> = HashMap::new();
    for (blob_hash, kind) in &blob_kinds {
        let blob = store.read_blob(blob_hash)?;
        let rewritten = match kind {
            BlobKind::Markdown => strip_markdown_aliases(&blob.0, &alias_to_canonical),
            BlobKind::Source => substitute_source_aliases(&blob.0, &alias_to_canonical),
        };
        if let Some(new_bytes) = rewritten {
            let new_hash = store.write_blob(&Blob(new_bytes))?;
            if new_hash != *blob_hash {
                blob_remap.insert(*blob_hash, new_hash);
            }
        }
    }

    // Update metas referencing remapped blobs.
    let mut metas_updated = Vec::new();
    for (sid, mhash) in store.ref_list()? {
        let mut meta: Requirement = store.read_requirement(&mhash)?;
        if meta.stable_id != sid {
            continue;
        }
        let mut changed = false;
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
            store.ref_set(&sid, &new_mhash)?;
            metas_updated.push(sid);
        }
    }

    // Update file-trees referencing remapped blobs.
    let mut paths_updated = Vec::new();
    for path in store.managed_paths()? {
        let Some(tree_hash) = store.tree_get(&path)? else { continue };
        let mut tree = store.read_file_tree(&tree_hash)?;
        let mut changed = false;
        for entry in tree.entries.iter_mut() {
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

    // Delete alias files.
    let mut aliases_removed = Vec::new();
    for (alias, _) in &pairs {
        store.alias_delete(alias)?;
        aliases_removed.push(alias.clone());
    }

    Ok(DealiasOutcome {
        aliases_removed,
        blobs_rewritten: blob_remap.len(),
        paths_updated,
        metas_updated,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlobKind {
    Markdown,
    Source,
}

/// Walk every managed file-tree and classify each reachable blob. A
/// blob is markdown if it appears in any `.md` file-tree's entries;
/// otherwise source.
fn classify_reachable_blobs(store: &Store) -> Result<HashMap<ObjectHash, BlobKind>> {
    let mut out: HashMap<ObjectHash, BlobKind> = HashMap::new();
    // Also pick up text_blobs and source_blobs from metas, in case a
    // blob is referenced by a meta but somehow not by a file-tree.
    let mut seen: HashSet<ObjectHash> = HashSet::new();
    for (_sid, mhash) in store.ref_list()? {
        let meta: Requirement = store.read_requirement(&mhash)?;
        seen.insert(meta.text_blob);
        for h in &meta.source_blobs {
            seen.insert(*h);
        }
    }
    for path in store.managed_paths()? {
        let Some(tree_hash) = store.tree_get(&path)? else { continue };
        let tree = store.read_file_tree(&tree_hash)?;
        let is_md = path
            .extension()
            .and_then(|s| s.to_str())
            == Some("md");
        for entry in &tree.entries {
            seen.insert(entry.blob);
            let kind = if is_md { BlobKind::Markdown } else { BlobKind::Source };
            // If a blob appears in both markdown and source file-trees,
            // prefer markdown (very unlikely in practice).
            out.entry(entry.blob).or_insert(kind);
            if is_md {
                out.insert(entry.blob, BlobKind::Markdown);
            }
        }
    }
    // Any blob seen only via metas (not file-trees) defaults to source.
    for h in seen {
        out.entry(h).or_insert(BlobKind::Source);
    }
    Ok(out)
}

/// Strip ` <!-- rq-A -->` annotations from `bytes` for each alias `A`
/// in the remap. The leading space is included in the strip so the
/// line ends naturally after removal. Returns None if no strips were
/// made.
fn strip_markdown_aliases(
    bytes: &[u8],
    aliases: &HashMap<StableId, StableId>,
) -> Option<Vec<u8>> {
    let mut current: Vec<u8> = bytes.to_vec();
    let mut any_changed = false;
    for alias in aliases.keys() {
        let pattern = format!(" <!-- {} -->", alias.as_str());
        let pat_bytes = pattern.as_bytes();
        loop {
            let Some(pos) = find_bytes(&current, pat_bytes) else { break };
            current.splice(pos..pos + pat_bytes.len(), std::iter::empty());
            any_changed = true;
        }
    }
    if any_changed { Some(current) } else { None }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Substitute `rq-A` with `rq-C` in `bytes` for each `(A, C)` in the
/// remap. Tokens are word-bounded (surrounding characters must not be
/// ASCII alphanumeric or underscore). Returns None if nothing changed.
fn substitute_source_aliases(
    bytes: &[u8],
    aliases: &HashMap<StableId, StableId>,
) -> Option<Vec<u8>> {
    let mut result: Option<Vec<u8>> = None;
    let mut i = 0;
    let mut buf = bytes.to_vec();
    while i + 11 <= buf.len() {
        if &buf[i..i + 3] == b"rq-" {
            let prior_ok = i == 0 || !is_word_byte(buf[i - 1]);
            let id_bytes = &buf[i + 3..i + 11];
            let hex_ok = id_bytes
                .iter()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase());
            let trail_ok = buf
                .get(i + 11)
                .map_or(true, |c| !is_word_byte(*c));
            if prior_ok && hex_ok && trail_ok {
                // bytes[i..i+11] is valid ASCII by construction.
                let id_str = std::str::from_utf8(&buf[i..i + 11]).unwrap();
                let id = StableId::new(id_str);
                if let Some(canonical) = aliases.get(&id) {
                    let canonical_bytes = canonical.as_str().as_bytes();
                    buf[i..i + 11].copy_from_slice(canonical_bytes);
                    result.get_or_insert_with(|| Vec::new());
                }
                i += 11;
                continue;
            }
        }
        i += 1;
    }
    if result.is_some() { Some(buf) } else { None }
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
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

    // ── Helper unit tests ─────────────────────────────────────────────

    #[test]
    fn strip_markdown_basic() {
        let mut aliases = HashMap::new();
        aliases.insert(
            StableId::new("rq-aaaaaaaa"),
            StableId::new("rq-bbbbbbbb"),
        );
        let bytes = b"- foo bar <!-- rq-aaaaaaaa -->\n";
        let out = strip_markdown_aliases(bytes, &aliases).unwrap();
        assert_eq!(out, b"- foo bar\n");
    }

    #[test]
    fn strip_markdown_multiple_aliases() {
        let mut aliases = HashMap::new();
        aliases.insert(
            StableId::new("rq-aaaaaaaa"),
            StableId::new("rq-canonical"),
        );
        aliases.insert(
            StableId::new("rq-bbbbbbbb"),
            StableId::new("rq-canonical"),
        );
        let bytes = b"- one <!-- rq-aaaaaaaa -->\n- two <!-- rq-bbbbbbbb -->\n";
        let out = strip_markdown_aliases(bytes, &aliases).unwrap();
        assert_eq!(out, b"- one\n- two\n");
    }

    #[test]
    fn strip_markdown_leaves_non_matching_annotations_alone() {
        let mut aliases = HashMap::new();
        aliases.insert(
            StableId::new("rq-aaaaaaaa"),
            StableId::new("rq-bbbbbbbb"),
        );
        let bytes = b"## Heading <!-- rq-99999999 -->\n- bullet <!-- rq-aaaaaaaa -->\n";
        let out = strip_markdown_aliases(bytes, &aliases).unwrap();
        // Heading's annotation (rq-99999999, not in aliases) survives;
        // bullet's annotation is stripped.
        assert_eq!(out, b"## Heading <!-- rq-99999999 -->\n- bullet\n");
    }

    #[test]
    fn substitute_source_basic() {
        let mut aliases = HashMap::new();
        aliases.insert(
            StableId::new("rq-aaaaaaaa"),
            StableId::new("rq-bbbbbbbb"),
        );
        let bytes = b"// rq-aaaaaaaa\nfn foo() {}\n";
        let out = substitute_source_aliases(bytes, &aliases).unwrap();
        assert_eq!(out, b"// rq-bbbbbbbb\nfn foo() {}\n");
    }

    #[test]
    fn substitute_source_respects_word_boundary() {
        let mut aliases = HashMap::new();
        aliases.insert(
            StableId::new("rq-aaaaaaaa"),
            StableId::new("rq-bbbbbbbb"),
        );
        let bytes = b"xrq-aaaaaaaay";
        assert!(substitute_source_aliases(bytes, &aliases).is_none());
    }

    #[test]
    fn substitute_source_leaves_canonical_alone() {
        let mut aliases = HashMap::new();
        aliases.insert(
            StableId::new("rq-aaaaaaaa"),
            StableId::new("rq-bbbbbbbb"),
        );
        let bytes = b"// rq-bbbbbbbb\nfn foo() {}\n";
        assert!(substitute_source_aliases(bytes, &aliases).is_none());
    }

    // ── End-to-end ────────────────────────────────────────────────────

    #[test]
    fn dealias_strips_bullet_annotation_from_markdown() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "\
# Top <!-- rq-aaaaaaaa -->

## Section <!-- rq-bbbbbbbb -->

- bullet item <!-- rq-cccccccc -->
- second item <!-- rq-dddddddd -->
";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();

        // rq-cccccccc and rq-dddddddd are aliases for rq-bbbbbbbb.
        let outcome = do_dealias(&store, None).unwrap();
        assert_eq!(outcome.aliases_removed.len(), 2);
        materialize::build(&store, root).unwrap();
        let on_disk = fs::read_to_string(root.join("rqm/doc.md")).unwrap();
        // Bullet annotations gone, headings preserved.
        assert!(on_disk.contains("# Top <!-- rq-aaaaaaaa -->"));
        assert!(on_disk.contains("## Section <!-- rq-bbbbbbbb -->"));
        assert!(on_disk.contains("- bullet item\n"));
        assert!(on_disk.contains("- second item\n"));
        assert!(!on_disk.contains("rq-cccccccc"));
        assert!(!on_disk.contains("rq-dddddddd"));

        // Aliases dir is empty.
        assert!(store.alias_list().unwrap().is_empty());
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn dealias_substitutes_source_stamps() {
        let (dir, store) = setup();
        let root = dir.path();
        // Markdown with a bullet alias.
        let md = "# Top <!-- rq-aaaaaaaa -->\n\n- bullet <!-- rq-bbbbbbbb -->\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        // Source that references the alias.
        let src = "// rq-bbbbbbbb\nfn foo() {}\n";
        write_at(root, "src/foo.rs", src);
        migrate::migrate_source(&store, root, Path::new("src/foo.rs")).unwrap();

        let outcome = do_dealias(&store, None).unwrap();
        assert!(outcome.blobs_rewritten >= 2); // text + source

        materialize::build(&store, root).unwrap();
        let on_disk = fs::read_to_string(root.join("src/foo.rs")).unwrap();
        // Source stamp now references the canonical.
        assert!(on_disk.contains("// rq-aaaaaaaa"));
        assert!(!on_disk.contains("rq-bbbbbbbb"));
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn dealias_subset_only_touches_listed_aliases() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "\
# Top <!-- rq-aaaaaaaa -->

- one <!-- rq-bbbbbbbb -->
- two <!-- rq-cccccccc -->
";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        // Dealias only rq-bbbbbbbb.
        let outcome = do_dealias(
            &store,
            Some(vec![StableId::new("rq-bbbbbbbb")]),
        )
        .unwrap();
        assert_eq!(outcome.aliases_removed, vec![StableId::new("rq-bbbbbbbb")]);
        // The other alias is still registered.
        assert!(store.alias_get(&StableId::new("rq-cccccccc")).unwrap().is_some());
        materialize::build(&store, root).unwrap();
        let on_disk = fs::read_to_string(root.join("rqm/doc.md")).unwrap();
        assert!(on_disk.contains("- one\n"));
        assert!(on_disk.contains("- two <!-- rq-cccccccc -->\n"));
    }

    #[test]
    fn dealias_rejects_unknown_alias() {
        let (_dir, store) = setup();
        make_req(&store, "rq-aaaaaaaa");
        let err = do_dealias(
            &store,
            Some(vec![StableId::new("rq-aaaaaaaa")]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("not a registered alias"), "{err}");
    }

    #[test]
    fn dealias_no_aliases_is_noop() {
        let (_dir, store) = setup();
        make_req(&store, "rq-aaaaaaaa");
        let outcome = do_dealias(&store, None).unwrap();
        assert!(outcome.aliases_removed.is_empty());
        assert_eq!(outcome.blobs_rewritten, 0);
    }
}
