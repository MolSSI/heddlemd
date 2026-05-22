//! `rqm log` — display the current state of a requirement: its kind,
//! ancestry, children, text-blob preview, source-blob locations, and
//! any aliases that point at it.
//!
//! Version history (e.g. predecessor meta versions over time) is not
//! yet tracked; it would require adding a `predecessor: Option<Hash>`
//! field on `Requirement` and updating it on every meta rewrite. Noted
//! as a future extension; the present command focuses on state.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::edit::{EditTarget, target_to_canonical_id};
use crate::object::{Kind, ObjectHash, Requirement, StableId};
use crate::store::Store;

/// Render the log output to a writer. Separated from the printing entry
/// point so tests can assert on the string content.
pub fn render(store: &Store, target: &EditTarget, out: &mut dyn std::fmt::Write) -> Result<()> {
    let canonical = target_to_canonical_id(store, target)?;
    let meta_hash = store.ref_get(&canonical)?.ok_or_else(|| {
        anyhow::anyhow!("ref missing for {canonical}")
    })?;
    let meta = store.read_requirement(&meta_hash)?;

    writeln!(out, "requirement: {canonical}  ({})", kind_label(meta.kind))?;
    writeln!(out, "  meta: {meta_hash}")?;

    // Direct parents. Under a multi-parent DAG, listing direct parents
    // rather than walking a single chain keeps the output bounded; the
    // user can `rqm log` on each parent to walk further.
    writeln!(out, "\nparents ({}):", meta.parents.len())?;
    if meta.parents.is_empty() {
        writeln!(out, "  (DAG root)")?;
    } else {
        for pid in &meta.parents {
            let parent_hash = store.ref_get(pid)?.with_context(|| {
                format!("parent {pid} has no ref")
            })?;
            let parent_meta = store.read_requirement(&parent_hash)?;
            let preview = blob_preview(store, &parent_meta.text_blob)?;
            writeln!(out, "  {pid}  {preview}")?;
        }
    }

    // Children (scan all refs)
    let children = direct_children(store, &canonical)?;
    writeln!(out, "\nchildren ({}):", children.len())?;
    if children.is_empty() {
        writeln!(out, "  (none)")?;
    } else {
        for cid in &children {
            let child_hash = store.ref_get(cid)?.unwrap();
            let child_meta = store.read_requirement(&child_hash)?;
            let preview = blob_preview(store, &child_meta.text_blob)?;
            writeln!(out, "  {cid}  {preview}")?;
        }
    }

    // Text blob
    let text_preview = blob_preview(store, &meta.text_blob)?;
    writeln!(out, "\ntext_blob: {}", meta.text_blob)?;
    writeln!(out, "  {text_preview}")?;

    // Source blobs (with file:line locations)
    writeln!(out, "\nsource_blobs ({}):", meta.source_blobs.len())?;
    if meta.source_blobs.is_empty() {
        writeln!(out, "  (none)")?;
    } else {
        let locations = blob_locations(store)?;
        for blob_hash in &meta.source_blobs {
            let here = locations.get(blob_hash).cloned().unwrap_or_default();
            if here.is_empty() {
                writeln!(out, "  {blob_hash}  (not in any file-tree)")?;
            } else {
                for (path, start_line) in &here {
                    writeln!(out, "  {blob_hash}  {}:{}", path.display(), start_line)?;
                }
            }
        }
    }

    // Aliases pointing at this canonical
    let aliases = aliases_for(store, &canonical)?;
    if !aliases.is_empty() {
        writeln!(out, "\naliases pointing here:")?;
        for alias in &aliases {
            writeln!(out, "  {alias}")?;
        }
    }

    Ok(())
}

/// Entry point used by the CLI: render to stdout via String buffer.
pub fn run(store: &Store, target: &EditTarget) -> Result<()> {
    let mut s = String::new();
    render(store, target, &mut s)?;
    print!("{s}");
    Ok(())
}

fn kind_label(k: Kind) -> &'static str {
    match k {
        Kind::Behavior => "behavior",
        Kind::Design => "design",
        Kind::Pending => "pending",
    }
}

/// First line of a blob, truncated to a reasonable preview length.
fn blob_preview(store: &Store, hash: &ObjectHash) -> Result<String> {
    let blob = store.read_blob(hash)?;
    let s = String::from_utf8_lossy(&blob.0);
    let first = s.lines().next().unwrap_or("").trim_end();
    const MAX: usize = 80;
    if first.chars().count() <= MAX {
        Ok(first.to_string())
    } else {
        // truncate by chars to avoid splitting a multi-byte sequence.
        let truncated: String = first.chars().take(MAX - 1).collect();
        Ok(format!("{truncated}…"))
    }
}

/// Direct children of `parent_id` — every ref whose meta lists
/// `parent_id` as its parent. Returns canonical stable_ids only
/// (aliases don't have metas of their own).
fn direct_children(store: &Store, parent_id: &StableId) -> Result<Vec<StableId>> {
    let mut out = Vec::new();
    for (sid, mhash) in store.ref_list()? {
        let meta: Requirement = store.read_requirement(&mhash)?;
        if meta.stable_id != sid {
            // Alias ref — skip; the canonical will be visited separately.
            continue;
        }
        if meta.parents.contains(parent_id) {
            out.push(sid);
        }
    }
    out.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    Ok(out)
}

/// For every blob hash that appears in any managed file-tree, return
/// the list of (path, start-line-1-based) locations where it occurs.
fn blob_locations(store: &Store) -> Result<BTreeMap<ObjectHash, Vec<(PathBuf, usize)>>> {
    let mut out: BTreeMap<ObjectHash, Vec<(PathBuf, usize)>> = BTreeMap::new();
    for path in store.managed_paths()? {
        let Some(tree_hash) = store.tree_get(&path)? else { continue };
        let tree = store.read_file_tree(&tree_hash)?;
        let mut line = 1usize;
        for entry in &tree.entries {
            let blob = store.read_blob(&entry.blob)?;
            out.entry(entry.blob).or_default().push((path.clone(), line));
            // Advance the running line counter by the blob's newline count.
            // (Migration-created blobs always end with a newline, so this is
            // exact; the byte-offset walk in `find_entry_at` covers
            // arbitrary cases for the lookup direction.)
            line += blob.0.iter().filter(|&&b| b == b'\n').count();
        }
    }
    Ok(out)
}

fn aliases_for(store: &Store, canonical: &StableId) -> Result<Vec<StableId>> {
    let mut out = Vec::new();
    for (alias, canon) in store.alias_list()? {
        if &canon == canonical {
            out.push(alias);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::*;
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
            parents: vec![],
            source_blobs: vec![],
        };
        let h = store.write_requirement(&req).unwrap();
        store.ref_set(&StableId::new(id), &h).unwrap();
    }

    fn setup_doc() -> (tempfile::TempDir, Store) {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "\
# Top <!-- rq-aaaaaaaa -->

Top body.

## First <!-- rq-bbbbbbbb -->

First body.

- bullet item <!-- rq-cccccccc -->

## Second <!-- rq-dddddddd -->

Second body.
";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        (dir, store)
    }

    #[test]
    fn log_root_shows_no_parents_and_lists_children() {
        let (_dir, store) = setup_doc();
        let mut out = String::new();
        let target = EditTarget::Id(StableId::new("rq-aaaaaaaa"));
        render(&store, &target, &mut out).unwrap();
        assert!(out.contains("requirement: rq-aaaaaaaa  (behavior)"));
        assert!(out.contains("parents (0):\n  (DAG root)"));
        assert!(out.contains("children (2):"));
        assert!(out.contains("rq-bbbbbbbb"));
        assert!(out.contains("rq-dddddddd"));
    }

    #[test]
    fn log_child_shows_direct_parents() {
        let (_dir, store) = setup_doc();
        let mut out = String::new();
        let target = EditTarget::Id(StableId::new("rq-bbbbbbbb"));
        render(&store, &target, &mut out).unwrap();
        let parents_idx = out.find("parents").unwrap();
        let children_idx = out.find("children").unwrap();
        let parents_section = &out[parents_idx..children_idx];
        assert!(parents_section.contains("rq-aaaaaaaa"));
    }

    #[test]
    fn log_lists_aliases_pointing_at_requirement() {
        let (_dir, store) = setup_doc();
        let mut out = String::new();
        // rq-bbbbbbbb is the canonical for the bullet alias rq-cccccccc.
        let target = EditTarget::Id(StableId::new("rq-bbbbbbbb"));
        render(&store, &target, &mut out).unwrap();
        assert!(
            out.contains("aliases pointing here:"),
            "expected aliases section in: {out}"
        );
        assert!(out.contains("rq-cccccccc"));
    }

    #[test]
    fn log_accepts_alias_as_target() {
        let (_dir, store) = setup_doc();
        let mut out = String::new();
        // Passing the alias should resolve to its canonical and show the
        // same info as if we'd passed the canonical directly.
        let target = EditTarget::Id(StableId::new("rq-cccccccc"));
        render(&store, &target, &mut out).unwrap();
        // The output describes rq-bbbbbbbb, not rq-cccccccc.
        assert!(out.contains("requirement: rq-bbbbbbbb"));
    }

    #[test]
    fn log_accepts_file_line_target() {
        let (_dir, store) = setup_doc();
        let mut out = String::new();
        // Line 1 of rqm/doc.md is the rq-aaaaaaaa heading.
        let target = EditTarget::FileLine {
            path: PathBuf::from("rqm/doc.md"),
            line: 1,
        };
        render(&store, &target, &mut out).unwrap();
        assert!(out.contains("requirement: rq-aaaaaaaa"));
    }

    #[test]
    fn log_shows_source_blob_locations() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n\nBody.\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();

        let src = "// rq-aaaaaaaa\nfn foo() {}\n";
        write_at(root, "src/example.rs", src);
        migrate::migrate_source(&store, root, Path::new("src/example.rs")).unwrap();

        let mut out = String::new();
        let target = EditTarget::Id(StableId::new("rq-aaaaaaaa"));
        render(&store, &target, &mut out).unwrap();
        assert!(out.contains("source_blobs (1):"));
        assert!(
            out.contains("src/example.rs:1"),
            "expected source-blob location in: {out}"
        );
    }

    #[test]
    fn log_accepts_source_file_line_target() {
        let (dir, store) = setup();
        let root = dir.path();
        write_at(root, "rqm/doc.md", "# Top <!-- rq-aaaaaaaa -->\n");
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        let src = "// rq-aaaaaaaa\nfn foo() {}\n";
        write_at(root, "src/example.rs", src);
        migrate::migrate_source(&store, root, Path::new("src/example.rs")).unwrap();

        let mut out = String::new();
        let target = EditTarget::FileLine {
            path: PathBuf::from("src/example.rs"),
            line: 2,
        };
        render(&store, &target, &mut out).unwrap();
        assert!(out.contains("requirement: rq-aaaaaaaa"));
    }

    #[test]
    fn log_rejects_unknown_id() {
        let (_dir, store) = setup_doc();
        let mut out = String::new();
        let target = EditTarget::Id(StableId::new("rq-12345678"));
        let err = render(&store, &target, &mut out).unwrap_err();
        assert!(err.to_string().contains("unknown stable_id"), "{err}");
    }

    #[test]
    fn direct_children_with_isolated_root() {
        let (_dir, store) = setup();
        make_req(&store, "rq-aaaaaaaa");
        let children = direct_children(&store, &StableId::new("rq-aaaaaaaa")).unwrap();
        assert!(children.is_empty());
    }
}
