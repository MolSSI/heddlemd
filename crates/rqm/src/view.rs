//! `rqm view` — read-oriented dump of a requirement and its associated
//! blobs. Designed as the primary mechanism by which a user or LLM
//! reads the requirements-to-source mapping for a given requirement.
//!
//! Output structure (requirements information first, then source):
//!   1. Header — stable_id, kind, materialized file location, text_blob
//!      hash, meta hash, parents, aliases pointing here.
//!   2. The text_blob's full bytes.
//!   3. Each direct child's header + text_blob bytes.
//!   4. Each source_blob's location, hash, and full bytes.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::Result;

use crate::edit::{EditTarget, target_to_canonical_id};
use crate::object::{Kind, ObjectHash, Requirement, StableId};
use crate::store::Store;

/// Render the view to an arbitrary writer.
pub fn render(store: &Store, target: &EditTarget, out: &mut String) -> Result<()> {
    let canonical = target_to_canonical_id(store, target)?;
    render_requirement(store, &canonical, out, /* is_child */ false)?;

    // Direct children (requirements info first)
    let children = direct_children(store, &canonical)?;
    writeln!(out, "\n=== Children ({}) ===", children.len())?;
    if children.is_empty() {
        writeln!(out, "\n(none)")?;
    } else {
        for cid in &children {
            writeln!(out)?;
            render_requirement(store, cid, out, /* is_child */ true)?;
        }
    }

    // Source blobs (after the requirement-side material)
    let meta_hash = store
        .ref_get(&canonical)?
        .ok_or_else(|| anyhow::anyhow!("ref missing for {canonical}"))?;
    let meta = store.read_requirement(&meta_hash)?;
    let locations = blob_locations(store)?;

    writeln!(out, "\n=== Source blobs ({}) ===", meta.source_blobs.len())?;
    if meta.source_blobs.is_empty() {
        writeln!(out, "\n(none)")?;
    } else {
        for blob_hash in &meta.source_blobs {
            writeln!(out)?;
            let locs = locations.get(blob_hash).cloned().unwrap_or_default();
            if locs.is_empty() {
                writeln!(out, "--- {blob_hash} (not in any file-tree) ---")?;
            } else {
                for (path, line) in &locs {
                    writeln!(out, "--- {}:{line} ---", path.display())?;
                }
                writeln!(out, "blob: {blob_hash}")?;
            }
            writeln!(out)?;
            let blob = store.read_blob(blob_hash)?;
            out.push_str(&String::from_utf8_lossy(&blob.0));
            if !blob.0.ends_with(b"\n") {
                out.push('\n');
            }
        }
    }

    Ok(())
}

/// Entry point for the CLI: render to stdout.
pub fn run(store: &Store, target: &EditTarget) -> Result<()> {
    let mut s = String::new();
    render(store, target, &mut s)?;
    print!("{s}");
    Ok(())
}

/// Render a single requirement's header + text_blob bytes. When
/// `is_child` is true, uses a slightly tighter header style.
fn render_requirement(
    store: &Store,
    id: &StableId,
    out: &mut String,
    is_child: bool,
) -> Result<()> {
    let meta_hash = store
        .ref_get(id)?
        .ok_or_else(|| anyhow::anyhow!("ref missing for {id}"))?;
    let meta = store.read_requirement(&meta_hash)?;

    if is_child {
        writeln!(out, "--- {id} ({}) ---", kind_label(meta.kind))?;
    } else {
        writeln!(out, "=== Requirement {id} ===")?;
        writeln!(out, "kind:       {}", kind_label(meta.kind))?;
    }

    let location = locate_text_blob(store, id, &meta)?;
    if let Some((path, line)) = location {
        writeln!(out, "file:       {}:{line}", path.display())?;
    } else {
        writeln!(out, "file:       (not materialized)")?;
    }
    writeln!(out, "text_blob:  {}", meta.text_blob)?;
    if !is_child {
        writeln!(out, "meta:       {meta_hash}")?;
        if meta.parents.is_empty() {
            writeln!(out, "parents:    (DAG root)")?;
        } else {
            let s = meta
                .parents
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(out, "parents:    {s}")?;
        }
        let aliases = aliases_for(store, id)?;
        if !aliases.is_empty() {
            let s = aliases
                .iter()
                .map(|a| a.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(out, "aliases:    {s}")?;
        }
    }
    writeln!(out)?;

    let blob = store.read_blob(&meta.text_blob)?;
    out.push_str(&String::from_utf8_lossy(&blob.0));
    if !blob.0.ends_with(b"\n") {
        out.push('\n');
    }
    Ok(())
}

fn kind_label(k: Kind) -> &'static str {
    match k {
        Kind::Behavior => "behavior",
        Kind::Design => "design",
        Kind::Pending => "pending",
    }
}

/// Find the (path, start_line) location of a requirement's text_blob
/// by scanning managed file-trees for a matching entry.
fn locate_text_blob(
    store: &Store,
    id: &StableId,
    meta: &Requirement,
) -> Result<Option<(PathBuf, usize)>> {
    for path in store.managed_paths()? {
        let Some(th) = store.tree_get(&path)? else { continue };
        let tree = store.read_file_tree(&th)?;
        let mut line = 1usize;
        for entry in &tree.entries {
            if &entry.stable_id == id && entry.blob == meta.text_blob {
                return Ok(Some((path.clone(), line)));
            }
            let blob = store.read_blob(&entry.blob)?;
            line += blob.0.iter().filter(|&&b| b == b'\n').count();
        }
    }
    Ok(None)
}

/// For every blob hash that appears in any managed file-tree, return
/// the list of (path, start-line) locations where it occurs.
fn blob_locations(
    store: &Store,
) -> Result<BTreeMap<ObjectHash, Vec<(PathBuf, usize)>>> {
    let mut out: BTreeMap<ObjectHash, Vec<(PathBuf, usize)>> = BTreeMap::new();
    for path in store.managed_paths()? {
        let Some(tree_hash) = store.tree_get(&path)? else { continue };
        let tree = store.read_file_tree(&tree_hash)?;
        let mut line = 1usize;
        for entry in &tree.entries {
            let blob = store.read_blob(&entry.blob)?;
            out.entry(entry.blob).or_default().push((path.clone(), line));
            line += blob.0.iter().filter(|&&b| b == b'\n').count();
        }
    }
    Ok(out)
}

/// Direct children — canonical stable_ids whose `parents` list
/// contains `id`. Sorted alphabetically for stable output.
fn direct_children(store: &Store, parent_id: &StableId) -> Result<Vec<StableId>> {
    let mut out = Vec::new();
    for (sid, mhash) in store.ref_list()? {
        let meta: Requirement = store.read_requirement(&mhash)?;
        if meta.stable_id != sid {
            // Alias ref — skip; canonical will be visited separately.
            continue;
        }
        if meta.parents.contains(parent_id) {
            out.push(sid);
        }
    }
    out.sort_by(|a, b| a.as_str().cmp(b.as_str()));
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
    use crate::insert::{self, InsertMode, InsertSpec};
    use crate::migrate;
    use crate::object::{Blob, Kind, Requirement};

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

Top body line 3.
Top body line 4.

## Child A <!-- rq-bbbbbbbb -->

Child A body.

## Child B <!-- rq-cccccccc -->

Child B body.
";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        (dir, store)
    }

    #[test]
    fn view_root_shows_text_metadata_and_children() {
        let (_dir, store) = setup_doc();
        let mut out = String::new();
        let target = EditTarget::Id(StableId::new("rq-aaaaaaaa"));
        render(&store, &target, &mut out).unwrap();
        // Header
        assert!(out.contains("=== Requirement rq-aaaaaaaa ==="));
        assert!(out.contains("kind:       behavior"));
        assert!(out.contains("file:       rqm/doc.md:1"));
        assert!(out.contains("parents:    (DAG root)"));
        // Text content
        assert!(out.contains("# Top"));
        assert!(out.contains("Top body line 4."));
        // No source blobs
        assert!(out.contains("=== Source blobs (0) ==="));
        // Children listed with their text
        assert!(out.contains("=== Children (2) ==="));
        assert!(out.contains("--- rq-bbbbbbbb (behavior) ---"));
        assert!(out.contains("--- rq-cccccccc (behavior) ---"));
        assert!(out.contains("Child A body."));
        assert!(out.contains("Child B body."));
    }

    #[test]
    fn view_accepts_file_line_target() {
        let (_dir, store) = setup_doc();
        let mut out = String::new();
        // line 4 is in the rq-aaaaaaaa text_blob region
        let target = EditTarget::FileLine {
            path: PathBuf::from("rqm/doc.md"),
            line: 4,
        };
        render(&store, &target, &mut out).unwrap();
        assert!(out.contains("=== Requirement rq-aaaaaaaa ==="));
    }

    #[test]
    fn view_resolves_source_file_line_to_owning_requirement() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n\nBody.\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        let src = "// rq-aaaaaaaa\nfn foo() {\n    let x = 1;\n}\n";
        write_at(root, "src/foo.rs", src);
        migrate::migrate_source(&store, root, Path::new("src/foo.rs")).unwrap();

        let mut out = String::new();
        let target = EditTarget::FileLine {
            path: PathBuf::from("src/foo.rs"),
            line: 3, // body of foo()
        };
        render(&store, &target, &mut out).unwrap();
        // Should show the OWNING requirement (rq-aaaaaaaa), not just the blob.
        assert!(out.contains("=== Requirement rq-aaaaaaaa ==="));
        // Should include the source blob's content under "Source blobs"
        assert!(out.contains("=== Source blobs (1) ==="));
        assert!(out.contains("--- src/foo.rs:1 ---"));
        assert!(out.contains("fn foo() {"));
    }

    #[test]
    fn view_shows_source_blob_locations_and_content() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        let src1 = "// rq-aaaaaaaa\nfn one() {}\n";
        write_at(root, "src/a.rs", src1);
        migrate::migrate_source(&store, root, Path::new("src/a.rs")).unwrap();
        let src2 = "// rq-aaaaaaaa\nfn two() {}\n";
        write_at(root, "src/b.rs", src2);
        migrate::migrate_source(&store, root, Path::new("src/b.rs")).unwrap();

        let mut out = String::new();
        let target = EditTarget::Id(StableId::new("rq-aaaaaaaa"));
        render(&store, &target, &mut out).unwrap();
        assert!(out.contains("=== Source blobs (2) ==="));
        // Both file locations appear, and both contents.
        assert!(out.contains("src/a.rs:1"));
        assert!(out.contains("src/b.rs:1"));
        assert!(out.contains("fn one()"));
        assert!(out.contains("fn two()"));
    }

    #[test]
    fn view_follows_alias_to_canonical() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n\n- bullet <!-- rq-bbbbbbbb -->\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();

        let mut out = String::new();
        // rq-bbbbbbbb is an alias for rq-aaaaaaaa.
        let target = EditTarget::Id(StableId::new("rq-bbbbbbbb"));
        render(&store, &target, &mut out).unwrap();
        // Output should describe the canonical, not the alias.
        assert!(out.contains("=== Requirement rq-aaaaaaaa ==="));
        // And should list the alias under aliases.
        assert!(out.contains("aliases:    rq-bbbbbbbb"));
    }

    #[test]
    fn view_shows_multi_parent_in_header() {
        let (_dir, store) = setup();
        make_req(&store, "rq-aaaaaaaa");
        make_req(&store, "rq-bbbbbbbb");
        // Multi-parent child.
        let text = store.write_blob(&Blob(b"prose".to_vec())).unwrap();
        let req = Requirement {
            stable_id: StableId::new("rq-cccccccc"),
            kind: Kind::Behavior,
            text_blob: text,
            parents: vec![
                StableId::new("rq-aaaaaaaa"),
                StableId::new("rq-bbbbbbbb"),
            ],
            source_blobs: vec![],
        };
        let h = store.write_requirement(&req).unwrap();
        store.ref_set(&StableId::new("rq-cccccccc"), &h).unwrap();

        let mut out = String::new();
        render(
            &store,
            &EditTarget::Id(StableId::new("rq-cccccccc")),
            &mut out,
        )
        .unwrap();
        assert!(out.contains("parents:    rq-aaaaaaaa, rq-bbbbbbbb"));
    }

    #[test]
    fn view_rejects_unknown_target() {
        let (_dir, store) = setup_doc();
        let mut out = String::new();
        let err = render(
            &store,
            &EditTarget::Id(StableId::new("rq-99999999")),
            &mut out,
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown stable_id"), "{err}");
    }

    #[test]
    fn view_handles_ghosted_requirement() {
        // A requirement whose text_blob isn't in any file-tree
        // (e.g. after `rqm rm --force` on the text_blob).
        let (_dir, store) = setup();
        make_req(&store, "rq-aaaaaaaa");
        let mut out = String::new();
        render(
            &store,
            &EditTarget::Id(StableId::new("rq-aaaaaaaa")),
            &mut out,
        )
        .unwrap();
        assert!(out.contains("file:       (not materialized)"));
    }

    #[test]
    fn view_shows_attributed_blob_in_source_section_when_attributed() {
        // Use insert to add a non-text-blob source attribution.
        let (dir, store) = setup();
        let root = dir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        insert::do_insert(
            &store,
            InsertSpec::parse("src/foo.rs:start", false).unwrap(),
            InsertMode::AttributeTo {
                owners: vec![StableId::new("rq-aaaaaaaa")],
            },
            b"// rq-aaaaaaaa\nfn added() {}\n".to_vec(),
        )
        .unwrap();
        let mut out = String::new();
        render(
            &store,
            &EditTarget::Id(StableId::new("rq-aaaaaaaa")),
            &mut out,
        )
        .unwrap();
        assert!(out.contains("=== Source blobs (1) ==="));
        assert!(out.contains("src/foo.rs:1"));
        assert!(out.contains("fn added()"));
    }
}
