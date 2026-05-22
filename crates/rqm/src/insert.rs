//! `rqm insert` — add a new blob at a specified position in a managed
//! file. Works for both source files and markdown requirements files;
//! the file is auto-created (added to `managed_paths`) if it is not yet
//! under management.
//!
//! The new blob is attributed to one or more existing requirements
//! (canonical stable_ids only — aliases are rejected). Each owner's
//! `source_blobs` is extended with the new blob's hash. The file-tree
//! entry carries the first owner's stable_id.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rand::RngCore;

use crate::mv::find_entry_at_line_with_offset;
use crate::object::{Blob, FileTree, FileTreeEntry, Kind, ObjectHash, Requirement, StableId};
use crate::store::Store;

#[derive(Debug, Clone)]
pub struct InsertSpec {
    pub path: PathBuf,
    pub anchor: InsertAnchor,
}

#[derive(Debug, Clone, Copy)]
pub enum InsertAnchor {
    Start,
    End,
    Line { line: usize, before: bool },
}

impl InsertSpec {
    /// Parse a `<path>:<anchor>` argument. `anchor` is `start`, `end`,
    /// or a 1-based line number. `before` adjusts a line anchor.
    pub fn parse(s: &str, before: bool) -> Result<Self> {
        let (path, anchor) = s.rsplit_once(':').ok_or_else(|| {
            anyhow::anyhow!("target must be <path>:<line|start|end>, got {s:?}")
        })?;
        if path.is_empty() {
            bail!("empty path in {s:?}");
        }
        let anchor = match anchor {
            "start" => {
                if before {
                    bail!("--before is incompatible with :start");
                }
                InsertAnchor::Start
            }
            "end" => {
                if before {
                    bail!("--before is incompatible with :end");
                }
                InsertAnchor::End
            }
            _ => {
                let line: usize = anchor
                    .parse()
                    .with_context(|| format!("invalid line/anchor in {s:?}"))?;
                if line == 0 {
                    bail!("line numbers are 1-based");
                }
                InsertAnchor::Line { line, before }
            }
        };
        Ok(InsertSpec {
            path: PathBuf::from(path),
            anchor,
        })
    }
}

/// What this insert is doing — extend an existing requirement's
/// content, or create a new requirement that owns the new blob.
pub enum InsertMode {
    /// Attribute the new blob to one or more existing requirements.
    /// Each owner gets the blob hash appended to its `source_blobs`.
    AttributeTo {
        /// Canonical stable_ids only. Order is preserved so the first
        /// listed becomes the file-tree entry's `stable_id`.
        owners: Vec<StableId>,
    },
    /// Create a new requirement that owns the new blob (as its
    /// `text_blob`). A fresh stable_id is generated.
    CreateNew {
        /// Parent stable_id in the DAG, or `None` for a DAG root.
        parent: Option<StableId>,
        kind: Kind,
    },
}

#[derive(Debug)]
pub struct InsertOutcome {
    pub new_blob: ObjectHash,
    pub created_file: bool,
    pub path: PathBuf,
    pub change: InsertChange,
}

#[derive(Debug)]
pub enum InsertChange {
    /// New blob attributed to existing requirements. Lists every owner
    /// whose meta was actually rewritten (a no-op happens if the blob
    /// was already in the source_blobs).
    Attributed(Vec<StableId>),
    /// New requirement created with this stable_id.
    Created(StableId),
}

/// Perform the insert. `content` must be non-empty.
pub fn do_insert(
    store: &Store,
    spec: InsertSpec,
    mode: InsertMode,
    content: Vec<u8>,
) -> Result<InsertOutcome> {
    if content.is_empty() {
        bail!("refusing to insert empty content");
    }

    // Validate inputs specific to each mode.
    match &mode {
        InsertMode::AttributeTo { owners } => {
            if owners.is_empty() {
                bail!("at least one --owner is required");
            }
            for o in owners {
                if store.ref_get(o)?.is_none() {
                    bail!("unknown stable_id: {o}");
                }
            }
        }
        InsertMode::CreateNew { parent, .. } => {
            if let Some(p) = parent {
                if store.ref_get(p)?.is_none() {
                    bail!("unknown parent stable_id: {p}");
                }
            }
        }
    }

    // Locate (or create) the destination file-tree.
    let (mut tree, created) = match store.tree_get(&spec.path)? {
        Some(h) => (store.read_file_tree(&h)?, false),
        None => {
            if matches!(spec.anchor, InsertAnchor::Line { .. }) {
                bail!(
                    "{} is not a managed file; cannot insert at :line. \
                     Use :start or :end to create the file.",
                    spec.path.display()
                );
            }
            (
                FileTree {
                    path: spec.path.clone(),
                    entries: Vec::new(),
                },
                true,
            )
        }
    };

    // Resolve the insertion index.
    let insert_at = match spec.anchor {
        InsertAnchor::Start => 0,
        InsertAnchor::End => tree.entries.len(),
        InsertAnchor::Line { line, before } => {
            let (idx, _) = find_entry_at_line_with_offset(store, &tree, line)?;
            if before { idx } else { idx + 1 }
        }
    };

    let new_blob = store.write_blob(&Blob(content))?;

    let (change, entry_stable_id) = match mode {
        InsertMode::AttributeTo { owners } => {
            // Dedupe while preserving order.
            let mut deduped: Vec<StableId> = Vec::new();
            for o in &owners {
                if !deduped.contains(o) {
                    deduped.push(o.clone());
                }
            }
            let mut updated = Vec::new();
            for owner in &deduped {
                let cur = store.ref_get(owner)?.expect("validated above");
                let mut req: Requirement = store.read_requirement(&cur)?;
                if !req.source_blobs.contains(&new_blob) {
                    req.source_blobs.push(new_blob);
                    req.source_blobs.sort();
                    let new_meta_hash = store.write_requirement(&req)?;
                    store.ref_set(owner, &new_meta_hash)?;
                    updated.push(owner.clone());
                }
            }
            let entry_id = deduped[0].clone();
            (InsertChange::Attributed(updated), entry_id)
        }
        InsertMode::CreateNew { parent, kind } => {
            let new_id = generate_stable_id(store)?;
            let req = Requirement {
                stable_id: new_id.clone(),
                kind,
                text_blob: new_blob,
                parent,
                source_blobs: Vec::new(),
            };
            let meta_hash = store.write_requirement(&req)?;
            store.ref_set(&new_id, &meta_hash)?;
            (InsertChange::Created(new_id.clone()), new_id)
        }
    };

    tree.entries.insert(
        insert_at,
        FileTreeEntry {
            stable_id: entry_stable_id,
            blob: new_blob,
        },
    );
    let tree_hash = store.write_file_tree(&tree)?;
    store.tree_set(&spec.path, &tree_hash)?;

    if created {
        let mut paths = store.managed_paths()?;
        paths.push(spec.path.clone());
        paths.sort();
        store.set_managed_paths(&paths)?;
    }

    Ok(InsertOutcome {
        new_blob,
        created_file: created,
        path: spec.path,
        change,
    })
}

/// Generate a fresh `rq-XXXXXXXX` stable_id that does not collide with
/// any existing ref or alias. 4 bytes of randomness → ~4 billion
/// possibilities; collisions are vanishingly unlikely but checked.
fn generate_stable_id(store: &Store) -> Result<StableId> {
    let mut rng = rand::thread_rng();
    let mut bytes = [0u8; 4];
    for _ in 0..100 {
        rng.fill_bytes(&mut bytes);
        let id = StableId::new(format!("rq-{}", hex::encode(bytes)));
        if store.ref_get(&id)?.is_none() && store.alias_get(&id)?.is_none() {
            return Ok(id);
        }
    }
    bail!("could not generate a unique stable_id after 100 attempts")
}

/// Convenience: read content from one of the supported sources.
pub enum ContentSource<'a> {
    Bytes(Vec<u8>),
    File(&'a Path),
    Stdin,
}

pub fn read_content(source: ContentSource<'_>) -> Result<Vec<u8>> {
    match source {
        ContentSource::Bytes(b) => Ok(b),
        ContentSource::File(p) => {
            std::fs::read(p).with_context(|| format!("read {}", p.display()))
        }
        ContentSource::Stdin => {
            use std::io::Read;
            let mut buf = Vec::new();
            std::io::stdin()
                .read_to_end(&mut buf)
                .context("read stdin")?;
            Ok(buf)
        }
    }
}

/// Open `$EDITOR` (or `vi`) on a temporary file and return its contents
/// on save. Empty content is preserved (callers decide how to handle).
pub fn read_via_editor() -> Result<Vec<u8>> {
    let tmp = tempfile::Builder::new()
        .prefix("rqm-insert-")
        .suffix(".txt")
        .tempfile()
        .context("create scratch file")?;
    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
    let status = std::process::Command::new(&editor)
        .arg(tmp.path())
        .status()
        .with_context(|| format!("spawn editor {editor}"))?;
    if !status.success() {
        bail!("editor exited non-zero; aborting insert");
    }
    std::fs::read(tmp.path()).context("read scratch file after edit")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::*;
    use crate::materialize;
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
            parent: None,
            source_blobs: vec![],
        };
        let h = store.write_requirement(&req).unwrap();
        store.ref_set(&StableId::new(id), &h).unwrap();
    }

    // ── Parsing ───────────────────────────────────────────────────────

    #[test]
    fn parse_line_anchor() {
        let s = InsertSpec::parse("src/x.rs:42", false).unwrap();
        assert!(matches!(s.anchor, InsertAnchor::Line { line: 42, before: false }));
    }

    #[test]
    fn parse_start_end() {
        let s = InsertSpec::parse("src/x.rs:start", false).unwrap();
        assert!(matches!(s.anchor, InsertAnchor::Start));
        let s = InsertSpec::parse("src/x.rs:end", false).unwrap();
        assert!(matches!(s.anchor, InsertAnchor::End));
    }

    #[test]
    fn parse_rejects_before_with_start_end() {
        assert!(InsertSpec::parse("x:start", true).is_err());
        assert!(InsertSpec::parse("x:end", true).is_err());
    }

    #[test]
    fn parse_rejects_empty_path_and_zero_line() {
        assert!(InsertSpec::parse(":42", false).is_err());
        assert!(InsertSpec::parse("x:0", false).is_err());
    }

    // ── Existing-file inserts ─────────────────────────────────────────

    fn setup_with_source() -> (tempfile::TempDir, Store) {
        let (dir, store) = setup();
        let root = dir.path();
        make_req(&store, "rq-aaaaaaaa");
        make_req(&store, "rq-bbbbbbbb");
        write_at(
            root,
            "src/x.rs",
            "// rq-aaaaaaaa\nfn alpha() {}\n\n// rq-bbbbbbbb\nfn beta() {}\n",
        );
        migrate::migrate_source(&store, root, Path::new("src/x.rs")).unwrap();
        (dir, store)
    }

    #[test]
    fn insert_at_end_of_existing_source() {
        let (dir, store) = setup_with_source();
        let root = dir.path();
        let spec = InsertSpec::parse("src/x.rs:end", false).unwrap();
        let outcome = do_insert(
            &store,
            spec,
            InsertMode::AttributeTo {
                owners: vec![StableId::new("rq-aaaaaaaa")],
            },
            b"// rq-aaaaaaaa\nfn alpha_extra() {}\n".to_vec(),
        )
        .unwrap();
        assert!(!outcome.created_file);
        match &outcome.change {
            InsertChange::Attributed(ids) => {
                assert_eq!(ids, &vec![StableId::new("rq-aaaaaaaa")])
            }
            other => panic!("expected Attributed, got {other:?}"),
        }

        materialize::build(&store, root).unwrap();
        let on_disk = fs::read_to_string(root.join("src/x.rs")).unwrap();
        assert!(on_disk.contains("fn alpha_extra"));
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn insert_before_line_in_existing_source() {
        let (dir, store) = setup_with_source();
        let root = dir.path();
        let spec = InsertSpec::parse("src/x.rs:4", true).unwrap();
        do_insert(
            &store,
            spec,
            InsertMode::AttributeTo {
                owners: vec![StableId::new("rq-aaaaaaaa")],
            },
            b"// rq-aaaaaaaa\nfn middle() {}\n\n".to_vec(),
        )
        .unwrap();
        materialize::build(&store, root).unwrap();
        let on_disk = fs::read_to_string(root.join("src/x.rs")).unwrap();
        let alpha = on_disk.find("fn alpha").unwrap();
        let middle = on_disk.find("fn middle").unwrap();
        let beta = on_disk.find("fn beta").unwrap();
        assert!(alpha < middle && middle < beta, "ordering wrong:\n{on_disk}");
    }

    #[test]
    fn insert_joint_ownership_appends_to_all_owners() {
        let (dir, store) = setup_with_source();
        let root = dir.path();
        let spec = InsertSpec::parse("src/x.rs:end", false).unwrap();
        do_insert(
            &store,
            spec,
            InsertMode::AttributeTo {
                owners: vec![
                    StableId::new("rq-aaaaaaaa"),
                    StableId::new("rq-bbbbbbbb"),
                ],
            },
            b"// rq-aaaaaaaa rq-bbbbbbbb\nfn shared() {}\n".to_vec(),
        )
        .unwrap();

        // Both owners should have the new blob in source_blobs.
        let ha = store.ref_get(&StableId::new("rq-aaaaaaaa")).unwrap().unwrap();
        let hb = store.ref_get(&StableId::new("rq-bbbbbbbb")).unwrap().unwrap();
        let ra = store.read_requirement(&ha).unwrap();
        let rb = store.read_requirement(&hb).unwrap();
        let shared = ra
            .source_blobs
            .iter()
            .find(|h| rb.source_blobs.contains(h))
            .expect("expected a shared blob hash");
        assert!(ra.source_blobs.contains(shared));
        assert!(rb.source_blobs.contains(shared));

        materialize::build(&store, root).unwrap();
        let on_disk = fs::read_to_string(root.join("src/x.rs")).unwrap();
        assert!(on_disk.contains("fn shared"));
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    // ── File creation ─────────────────────────────────────────────────

    #[test]
    fn insert_creates_new_managed_file() {
        let (dir, store) = setup();
        let root = dir.path();
        make_req(&store, "rq-aaaaaaaa");

        let spec = InsertSpec::parse("src/new.rs:end", false).unwrap();
        let outcome = do_insert(
            &store,
            spec,
            InsertMode::AttributeTo {
                owners: vec![StableId::new("rq-aaaaaaaa")],
            },
            b"// rq-aaaaaaaa\nfn fresh() {}\n".to_vec(),
        )
        .unwrap();
        assert!(outcome.created_file);

        materialize::build(&store, root).unwrap();
        let on_disk = fs::read_to_string(root.join("src/new.rs")).unwrap();
        assert!(on_disk.contains("fn fresh"));

        // managed_paths includes the new file.
        let paths = store.managed_paths().unwrap();
        assert!(paths.iter().any(|p| p == Path::new("src/new.rs")));
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn insert_into_new_file_rejects_line_anchor() {
        let (_dir, store) = setup();
        make_req(&store, "rq-aaaaaaaa");
        let spec = InsertSpec::parse("src/new.rs:5", false).unwrap();
        let err = do_insert(
            &store,
            spec,
            InsertMode::AttributeTo {
                owners: vec![StableId::new("rq-aaaaaaaa")],
            },
            b"// rq-aaaaaaaa\nfn x() {}\n".to_vec(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("not a managed file"),
            "{err}"
        );
    }

    // ── Markdown inserts ──────────────────────────────────────────────

    #[test]
    fn insert_into_existing_markdown() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n\nBody.\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();

        // Append an additional text-content blob owned by the same
        // requirement (a Phase 1.5 use case: adding a Gherkin scenario
        // or an extra paragraph after migration).
        let spec = InsertSpec::parse("rqm/doc.md:end", false).unwrap();
        do_insert(
            &store,
            spec,
            InsertMode::AttributeTo {
                owners: vec![StableId::new("rq-aaaaaaaa")],
            },
            b"\nAn extra paragraph.\n".to_vec(),
        )
        .unwrap();

        materialize::build(&store, root).unwrap();
        let on_disk = fs::read_to_string(root.join("rqm/doc.md")).unwrap();
        assert!(on_disk.contains("Body."));
        assert!(on_disk.contains("An extra paragraph."));
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    // ── Validation ────────────────────────────────────────────────────

    #[test]
    fn rejects_empty_content() {
        let (_dir, store) = setup_with_source();
        let spec = InsertSpec::parse("src/x.rs:end", false).unwrap();
        let err = do_insert(
            &store,
            spec,
            InsertMode::AttributeTo {
                owners: vec![StableId::new("rq-aaaaaaaa")],
            },
            Vec::new(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("empty content"), "{err}");
    }

    #[test]
    fn rejects_no_owners() {
        let (_dir, store) = setup_with_source();
        let spec = InsertSpec::parse("src/x.rs:end", false).unwrap();
        let err = do_insert(
            &store,
            spec,
            InsertMode::AttributeTo { owners: Vec::new() },
            b"// hi\n".to_vec(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("at least one --owner"), "{err}");
    }

    // ── Create mode ───────────────────────────────────────────────────

    #[test]
    fn create_new_root_in_new_file() {
        let (dir, store) = setup();
        let root = dir.path();
        let spec = InsertSpec::parse("rqm/new_feature.md:start", false).unwrap();
        let outcome = do_insert(
            &store,
            spec,
            InsertMode::CreateNew {
                parent: None,
                kind: Kind::Behavior,
            },
            b"# Feature: New thing\n\nBody.\n".to_vec(),
        )
        .unwrap();
        assert!(outcome.created_file);
        let new_id = match &outcome.change {
            InsertChange::Created(id) => id.clone(),
            other => panic!("expected Created, got {other:?}"),
        };

        // The new requirement exists and is a DAG root.
        let h = store.ref_get(&new_id).unwrap().unwrap();
        let req = store.read_requirement(&h).unwrap();
        assert_eq!(req.parent, None);
        assert_eq!(req.kind, Kind::Behavior);

        materialize::build(&store, root).unwrap();
        let on_disk = fs::read_to_string(root.join("rqm/new_feature.md")).unwrap();
        assert_eq!(on_disk, "# Feature: New thing\n\nBody.\n");
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn create_new_child_in_existing_file() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n\nBody.\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();

        let spec = InsertSpec::parse("rqm/doc.md:end", false).unwrap();
        let outcome = do_insert(
            &store,
            spec,
            InsertMode::CreateNew {
                parent: Some(StableId::new("rq-aaaaaaaa")),
                kind: Kind::Behavior,
            },
            b"\n## New section\n\nDetails.\n".to_vec(),
        )
        .unwrap();
        let new_id = match &outcome.change {
            InsertChange::Created(id) => id.clone(),
            other => panic!("expected Created, got {other:?}"),
        };

        let h = store.ref_get(&new_id).unwrap().unwrap();
        let req = store.read_requirement(&h).unwrap();
        assert_eq!(req.parent, Some(StableId::new("rq-aaaaaaaa")));

        materialize::build(&store, root).unwrap();
        let on_disk = fs::read_to_string(root.join("rqm/doc.md")).unwrap();
        assert!(on_disk.contains("# Top"));
        assert!(on_disk.contains("## New section"));
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }

    #[test]
    fn create_kind_design() {
        let (dir, store) = setup();
        let root = dir.path();
        let spec = InsertSpec::parse("rqm/doc.md:start", false).unwrap();
        let outcome = do_insert(
            &store,
            spec,
            InsertMode::CreateNew {
                parent: None,
                kind: Kind::Design,
            },
            b"# Glossary\n".to_vec(),
        )
        .unwrap();
        let new_id = match outcome.change {
            InsertChange::Created(id) => id,
            other => panic!("expected Created, got {other:?}"),
        };
        let h = store.ref_get(&new_id).unwrap().unwrap();
        let req = store.read_requirement(&h).unwrap();
        assert_eq!(req.kind, Kind::Design);
        let _ = root;
    }

    #[test]
    fn create_rejects_unknown_parent() {
        let (_dir, store) = setup();
        let spec = InsertSpec::parse("rqm/doc.md:start", false).unwrap();
        let err = do_insert(
            &store,
            spec,
            InsertMode::CreateNew {
                parent: Some(StableId::new("rq-99999999")),
                kind: Kind::Behavior,
            },
            b"# X\n".to_vec(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("unknown parent stable_id"),
            "{err}"
        );
    }

    #[test]
    fn generated_ids_are_unique_across_calls() {
        let (dir, store) = setup();
        let root = dir.path();
        let spec1 = InsertSpec::parse("rqm/a.md:start", false).unwrap();
        let spec2 = InsertSpec::parse("rqm/b.md:start", false).unwrap();
        let out1 = do_insert(
            &store,
            spec1,
            InsertMode::CreateNew {
                parent: None,
                kind: Kind::Behavior,
            },
            b"# A\n".to_vec(),
        )
        .unwrap();
        let out2 = do_insert(
            &store,
            spec2,
            InsertMode::CreateNew {
                parent: None,
                kind: Kind::Behavior,
            },
            b"# B\n".to_vec(),
        )
        .unwrap();
        let id1 = match out1.change {
            InsertChange::Created(id) => id,
            _ => unreachable!(),
        };
        let id2 = match out2.change {
            InsertChange::Created(id) => id,
            _ => unreachable!(),
        };
        assert_ne!(id1, id2);
        let _ = root;
    }

    #[test]
    fn content_is_stored_verbatim_no_stamp_injection() {
        // The user-provided content goes into the text_blob byte-for-byte.
        // The generated stable_id is NOT injected into the heading
        // annotation — that's intentional.
        let (dir, store) = setup();
        let root = dir.path();
        let spec = InsertSpec::parse("rqm/foo.md:start", false).unwrap();
        let content = b"# Title without annotation\n\nBody.\n";
        let outcome = do_insert(
            &store,
            spec,
            InsertMode::CreateNew {
                parent: None,
                kind: Kind::Behavior,
            },
            content.to_vec(),
        )
        .unwrap();
        let new_id = match outcome.change {
            InsertChange::Created(id) => id,
            _ => unreachable!(),
        };
        let h = store.ref_get(&new_id).unwrap().unwrap();
        let req = store.read_requirement(&h).unwrap();
        let blob = store.read_blob(&req.text_blob).unwrap();
        assert_eq!(blob.0, content);
        // The blob bytes don't contain the generated id.
        let s = std::str::from_utf8(&blob.0).unwrap();
        assert!(
            !s.contains(new_id.as_str()),
            "generated id should not have been injected: {s}"
        );
        let _ = root;
    }

    #[test]
    fn rejects_unknown_owner() {
        let (_dir, store) = setup_with_source();
        let spec = InsertSpec::parse("src/x.rs:end", false).unwrap();
        let err = do_insert(
            &store,
            spec,
            InsertMode::AttributeTo {
                owners: vec![StableId::new("rq-99999999")],
            },
            b"// hi\n".to_vec(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown stable_id"), "{err}");
    }
}
