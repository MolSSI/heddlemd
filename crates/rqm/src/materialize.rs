//! Materialize the managed working tree from `.rqm/` (the `build` and
//! `check` subcommands).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::object::FileTree;
use crate::store::Store;

/// Materialize every managed path into a `path -> bytes` map. The paths
/// are exactly as recorded in the file-trees (resolved against `root`
/// only if relative).
pub fn materialize(store: &Store) -> Result<BTreeMap<PathBuf, Vec<u8>>> {
    let paths = store.managed_paths()?;
    let mut out = BTreeMap::new();
    for path in paths {
        let bytes = materialize_one(store, &path)
            .with_context(|| format!("materialize {}", path.display()))?;
        out.insert(path, bytes);
    }
    Ok(out)
}

fn materialize_one(store: &Store, path: &Path) -> Result<Vec<u8>> {
    let tree_hash = store
        .tree_get(path)?
        .ok_or_else(|| anyhow::anyhow!("no tree ref for managed path {}", path.display()))?;
    let tree: FileTree = store.read_file_tree(&tree_hash)?;
    if tree.path != path {
        bail!(
            "tree ref for {} points to a file-tree whose path is {}",
            path.display(),
            tree.path.display()
        );
    }
    let mut buf = Vec::new();
    for entry in &tree.entries {
        let blob = store.read_blob(&entry.blob)?;
        buf.extend_from_slice(&blob.0);
    }
    Ok(buf)
}

/// Write every managed path's materialized contents to disk under `root`.
/// Relative paths in the file-trees are joined with `root`; absolute
/// paths are written as-is.
pub fn build(store: &Store, root: &Path) -> Result<BuildReport> {
    let materials = materialize(store)?;
    let mut written = Vec::new();
    let mut unchanged = Vec::new();
    for (rel, bytes) in &materials {
        let target = resolve(root, rel);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        let needs_write = match fs::read(&target) {
            Ok(existing) => existing != *bytes,
            Err(_) => true,
        };
        if needs_write {
            fs::write(&target, bytes)
                .with_context(|| format!("write {}", target.display()))?;
            written.push(rel.clone());
        } else {
            unchanged.push(rel.clone());
        }
    }
    Ok(BuildReport { written, unchanged })
}

#[derive(Debug, Default)]
pub struct BuildReport {
    pub written: Vec<PathBuf>,
    pub unchanged: Vec<PathBuf>,
}

/// Compare the materialized output against on-disk files. Returns Ok with
/// a report; the caller decides whether mismatches are a failure.
///
/// Integrity checks run first; if they pass, materialization is attempted
/// for each managed path independently so that a single corrupted path
/// doesn't suppress the report on the others.
pub fn check(store: &Store, root: &Path) -> Result<CheckReport> {
    let mut report = CheckReport::default();
    // Cross-reference integrity (subset of rules in rqm-spec.md). The
    // bidirectional-symmetry check is deferred until joint ownership is
    // revisited after Pilot 3.
    integrity_checks(store, &mut report)?;

    let paths = store.managed_paths()?;
    for rel in &paths {
        match materialize_one(store, rel) {
            Ok(expected) => {
                let target = resolve(root, rel);
                match fs::read(&target) {
                    Ok(actual) if actual == expected => report.matches.push(rel.clone()),
                    Ok(_) => report.diffs.push(rel.clone()),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        report.missing.push(rel.clone())
                    }
                    Err(e) => {
                        return Err(anyhow::Error::from(e)
                            .context(format!("read {}", target.display())));
                    }
                }
            }
            Err(e) => {
                report
                    .integrity
                    .push(format!("materialize {}: {e:#}", rel.display()));
            }
        }
    }

    Ok(report)
}

#[derive(Debug, Default)]
pub struct CheckReport {
    /// Paths whose on-disk contents match the materialized output.
    pub matches: Vec<PathBuf>,
    /// Paths whose on-disk contents differ from the materialized output.
    pub diffs: Vec<PathBuf>,
    /// Managed paths with no file on disk.
    pub missing: Vec<PathBuf>,
    /// Integrity-rule violations (descriptive strings).
    pub integrity: Vec<String>,
}

impl CheckReport {
    pub fn is_clean(&self) -> bool {
        self.diffs.is_empty() && self.missing.is_empty() && self.integrity.is_empty()
    }
}

fn integrity_checks(store: &Store, report: &mut CheckReport) -> Result<()> {
    let managed: Vec<PathBuf> = store.managed_paths()?;

    // Every managed path has a tree ref, and the file-tree's path matches.
    for p in &managed {
        match store.tree_get(p)? {
            None => report
                .integrity
                .push(format!("managed path {} has no tree ref", p.display())),
            Some(h) => match store.read_file_tree(&h) {
                Ok(t) if t.path == *p => {}
                Ok(t) => report.integrity.push(format!(
                    "tree ref {} -> file-tree whose path is {}",
                    p.display(),
                    t.path.display()
                )),
                Err(e) => report
                    .integrity
                    .push(format!("read file-tree for {}: {e:#}", p.display())),
            },
        }
    }

    // Every alias points at a canonical stable_id that has a real ref.
    for (alias, canonical) in store.alias_list()? {
        if store.ref_get(&canonical)?.is_none() {
            report.integrity.push(format!(
                "alias {alias} -> {canonical} (canonical has no ref)"
            ));
        }
    }

    // Every requirement-ref resolves and every blob referenced by a meta
    // or file-tree exists in the object store.
    for (id, h) in store.ref_list()? {
        let req = match store.read_requirement(&h) {
            Ok(r) => r,
            Err(e) => {
                report
                    .integrity
                    .push(format!("ref {id} -> unreadable meta {h}: {e:#}"));
                continue;
            }
        };
        if req.stable_id != id {
            report.integrity.push(format!(
                "ref {id} points to a meta with stable_id {}",
                req.stable_id
            ));
        }
        if !store.has_object(&req.text_blob) {
            report
                .integrity
                .push(format!("meta {id}: text_blob {} missing", req.text_blob));
        }
        for b in &req.source_blobs {
            if !store.has_object(b) {
                report
                    .integrity
                    .push(format!("meta {id}: source_blob {b} missing"));
            }
        }
        if let Some(parent) = &req.parent {
            if store.ref_get(parent)?.is_none() {
                report
                    .integrity
                    .push(format!("meta {id}: parent {parent} has no ref"));
            }
        }
    }

    for p in &managed {
        let Some(h) = store.tree_get(p)? else { continue };
        let tree = match store.read_file_tree(&h) {
            Ok(t) => t,
            Err(_) => continue, // already reported above
        };
        for entry in &tree.entries {
            if !store.has_object(&entry.blob) {
                report.integrity.push(format!(
                    "file-tree {}: entry blob {} missing",
                    p.display(),
                    entry.blob
                ));
            }
            if store.ref_get(&entry.stable_id)?.is_none() {
                report.integrity.push(format!(
                    "file-tree {}: entry stable_id {} has no ref",
                    p.display(),
                    entry.stable_id
                ));
            }
        }
    }

    Ok(())
}

fn resolve(root: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        root.join(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::migrate;

    fn setup() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::init(dir.path().join(".rqm")).unwrap();
        (dir, store)
    }

    fn write_at(root: &Path, rel: &str, content: &str) -> PathBuf {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(&p, content).unwrap();
        p
    }

    #[test]
    fn build_and_check_round_trip() {
        let (workdir, store) = setup();
        let root = workdir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n\nBody.\n\n## Child <!-- rq-bbbbbbbb -->\n\nChild body.\n";
        let src = "// rq-aaaaaaaa\nfn root_func() {}\n\n// rq-bbbbbbbb\nfn child_func() {}\n";
        write_at(root, "rqm/test.md", md);
        write_at(root, "src/test.rs", src);

        migrate::migrate_markdown(&store, root, Path::new("rqm/test.md")).unwrap();
        migrate::migrate_source(&store, root, Path::new("src/test.rs")).unwrap();

        // Check passes after migration.
        let r = check(&store, root).unwrap();
        assert!(r.is_clean(), "post-migrate check should be clean: {r:?}");
        assert_eq!(r.matches.len(), 2);
    }

    #[test]
    fn build_writes_managed_paths() {
        let (workdir, store) = setup();
        let root = workdir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n\nHello.\n";
        let rel = Path::new("rqm/doc.md");
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, rel).unwrap();

        // Delete the original; build should restore it.
        fs::remove_file(root.join(rel)).unwrap();
        let report = build(&store, root).unwrap();
        assert!(report.written.iter().any(|p| p == rel));
        assert_eq!(fs::read_to_string(root.join(rel)).unwrap(), md);
    }

    #[test]
    fn check_detects_drift() {
        let (workdir, store) = setup();
        let root = workdir.path();
        let rel = Path::new("rqm/doc.md");
        write_at(root, "rqm/doc.md", "# Top <!-- rq-aaaaaaaa -->\n\nHello.\n");
        migrate::migrate_markdown(&store, root, rel).unwrap();

        let r = check(&store, root).unwrap();
        assert!(r.is_clean(), "expected clean, got {r:?}");

        fs::write(root.join(rel), "# Top <!-- rq-aaaaaaaa -->\n\nGoodbye.\n").unwrap();
        let r = check(&store, root).unwrap();
        assert!(!r.is_clean());
        assert_eq!(r.diffs.len(), 1);
    }

    #[test]
    fn check_detects_missing_object() {
        let (workdir, store) = setup();
        let root = workdir.path();
        let rel = Path::new("rqm/doc.md");
        write_at(root, "rqm/doc.md", "# Top <!-- rq-aaaaaaaa -->\n");
        migrate::migrate_markdown(&store, root, rel).unwrap();

        // Find any blob object and delete it.
        let objects = store.root().join("objects");
        let mut deleted = false;
        'outer: for sub in fs::read_dir(&objects).unwrap() {
            let sub = sub.unwrap();
            for obj in fs::read_dir(sub.path()).unwrap() {
                let obj = obj.unwrap();
                let bytes = fs::read(obj.path()).unwrap();
                if !bytes.starts_with(b"{") {
                    fs::remove_file(obj.path()).unwrap();
                    deleted = true;
                    break 'outer;
                }
            }
        }
        assert!(deleted, "should have found a blob to delete");

        let r = check(&store, root).unwrap();
        assert!(!r.integrity.is_empty(), "expected integrity violation");
    }
}
