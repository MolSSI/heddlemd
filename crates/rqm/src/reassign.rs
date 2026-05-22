//! `rqm reassign` — change a requirement's parents in the DAG.
//!
//! Replaces the target requirement's `parents` list with the given set
//! (zero for a DAG root, or one or more canonical stable_ids). Performs
//! validation to prevent cycles, self-loops, and unknown parent
//! references.

use std::collections::HashSet;

use anyhow::{Result, bail};

use crate::object::{Requirement, StableId};
use crate::store::Store;

#[derive(Debug)]
pub struct ReassignOutcome {
    pub target: StableId,
    pub old_parents: Vec<StableId>,
    pub new_parents: Vec<StableId>,
}

/// Replace `target`'s parents with `new_parents`. Aliases are rejected
/// for both the target and any parent. `new_parents` is deduplicated
/// internally; pass an empty Vec to make the requirement a DAG root.
pub fn do_reassign(
    store: &Store,
    target: &StableId,
    new_parents: Vec<StableId>,
) -> Result<ReassignOutcome> {
    // Target must be canonical and exist.
    let target_canonical = store
        .resolve(target)?
        .ok_or_else(|| anyhow::anyhow!("unknown stable_id: {target}"))?;
    if &target_canonical != target {
        bail!(
            "{target} is an alias for {target_canonical}; pass the canonical id."
        );
    }

    // Each parent must be canonical and exist, and cannot be the target
    // itself.
    let mut deduped: Vec<StableId> = Vec::new();
    for p in &new_parents {
        if p == target {
            bail!("a requirement cannot be its own parent ({target})");
        }
        let canonical = store
            .resolve(p)?
            .ok_or_else(|| anyhow::anyhow!("unknown parent stable_id: {p}"))?;
        if &canonical != p {
            bail!(
                "{p} is an alias for {canonical}; pass the canonical id for the parent."
            );
        }
        if !deduped.contains(p) {
            deduped.push(p.clone());
        }
    }

    // Cycle detection: target must not appear in the transitive ancestry
    // of any proposed parent.
    for p in &deduped {
        if would_create_cycle(store, target, p)? {
            bail!(
                "setting parent {p} on {target} would create a cycle ({target} \
                 is an ancestor of {p})"
            );
        }
    }

    // Apply the update.
    let meta_hash = store.ref_get(target)?.expect("validated above");
    let mut meta: Requirement = store.read_requirement(&meta_hash)?;
    let old_parents = meta.parents.clone();
    let mut sorted = deduped.clone();
    sorted.sort();
    meta.parents = sorted;
    let new_meta_hash = store.write_requirement(&meta)?;
    store.ref_set(target, &new_meta_hash)?;

    Ok(ReassignOutcome {
        target: target.clone(),
        old_parents,
        new_parents: meta.parents,
    })
}

/// Walk transitively up from `start` through `parents` edges; return
/// true if `target` is encountered.
fn would_create_cycle(
    store: &Store,
    target: &StableId,
    start: &StableId,
) -> Result<bool> {
    let mut to_visit: Vec<StableId> = vec![start.clone()];
    let mut visited: HashSet<StableId> = HashSet::new();
    while let Some(id) = to_visit.pop() {
        if &id == target {
            return Ok(true);
        }
        if !visited.insert(id.clone()) {
            continue;
        }
        let Some(h) = store.ref_get(&id)? else { continue };
        let meta: Requirement = store.read_requirement(&h)?;
        for parent in meta.parents {
            to_visit.push(parent);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use super::*;
    use crate::materialize;
    use crate::migrate;
    use crate::object::{Blob, Kind, ObjectHash};

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

    fn make_req_with_parents(store: &Store, id: &str, parents: Vec<StableId>) -> ObjectHash {
        let text = store.write_blob(&Blob(b"prose".to_vec())).unwrap();
        let req = Requirement {
            stable_id: StableId::new(id),
            kind: Kind::Behavior,
            text_blob: text,
            parents,
            source_blobs: vec![],
        };
        let h = store.write_requirement(&req).unwrap();
        store.ref_set(&StableId::new(id), &h).unwrap();
        h
    }

    fn make_req(store: &Store, id: &str) -> ObjectHash {
        make_req_with_parents(store, id, vec![])
    }

    fn parents_of(store: &Store, id: &str) -> Vec<StableId> {
        let h = store.ref_get(&StableId::new(id)).unwrap().unwrap();
        store.read_requirement(&h).unwrap().parents
    }

    // ── Basic reassign ────────────────────────────────────────────────

    #[test]
    fn reassign_changes_single_parent() {
        let (_dir, store) = setup();
        make_req(&store, "rq-aaaaaaaa");
        make_req(&store, "rq-bbbbbbbb");
        make_req_with_parents(
            &store,
            "rq-cccccccc",
            vec![StableId::new("rq-aaaaaaaa")],
        );
        let outcome = do_reassign(
            &store,
            &StableId::new("rq-cccccccc"),
            vec![StableId::new("rq-bbbbbbbb")],
        )
        .unwrap();
        assert_eq!(outcome.old_parents, vec![StableId::new("rq-aaaaaaaa")]);
        assert_eq!(outcome.new_parents, vec![StableId::new("rq-bbbbbbbb")]);
        assert_eq!(parents_of(&store, "rq-cccccccc"), vec![StableId::new("rq-bbbbbbbb")]);
    }

    #[test]
    fn reassign_sets_multiple_parents() {
        let (_dir, store) = setup();
        make_req(&store, "rq-aaaaaaaa");
        make_req(&store, "rq-bbbbbbbb");
        make_req(&store, "rq-cccccccc");
        do_reassign(
            &store,
            &StableId::new("rq-cccccccc"),
            vec![StableId::new("rq-aaaaaaaa"), StableId::new("rq-bbbbbbbb")],
        )
        .unwrap();
        let parents = parents_of(&store, "rq-cccccccc");
        assert_eq!(
            parents,
            vec![StableId::new("rq-aaaaaaaa"), StableId::new("rq-bbbbbbbb")]
        );
    }

    #[test]
    fn reassign_to_dag_root() {
        let (_dir, store) = setup();
        make_req(&store, "rq-aaaaaaaa");
        make_req_with_parents(
            &store,
            "rq-bbbbbbbb",
            vec![StableId::new("rq-aaaaaaaa")],
        );
        do_reassign(&store, &StableId::new("rq-bbbbbbbb"), vec![]).unwrap();
        assert!(parents_of(&store, "rq-bbbbbbbb").is_empty());
    }

    #[test]
    fn reassign_dedupes_input() {
        let (_dir, store) = setup();
        make_req(&store, "rq-aaaaaaaa");
        make_req(&store, "rq-bbbbbbbb");
        do_reassign(
            &store,
            &StableId::new("rq-bbbbbbbb"),
            vec![
                StableId::new("rq-aaaaaaaa"),
                StableId::new("rq-aaaaaaaa"),
            ],
        )
        .unwrap();
        assert_eq!(parents_of(&store, "rq-bbbbbbbb"), vec![StableId::new("rq-aaaaaaaa")]);
    }

    // ── Validation ────────────────────────────────────────────────────

    #[test]
    fn reassign_rejects_unknown_target() {
        let (_dir, store) = setup();
        let err = do_reassign(&store, &StableId::new("rq-99999999"), vec![])
            .unwrap_err();
        assert!(err.to_string().contains("unknown stable_id"), "{err}");
    }

    #[test]
    fn reassign_rejects_unknown_parent() {
        let (_dir, store) = setup();
        make_req(&store, "rq-aaaaaaaa");
        let err = do_reassign(
            &store,
            &StableId::new("rq-aaaaaaaa"),
            vec![StableId::new("rq-99999999")],
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown parent stable_id"), "{err}");
    }

    #[test]
    fn reassign_rejects_self_loop() {
        let (_dir, store) = setup();
        make_req(&store, "rq-aaaaaaaa");
        let err = do_reassign(
            &store,
            &StableId::new("rq-aaaaaaaa"),
            vec![StableId::new("rq-aaaaaaaa")],
        )
        .unwrap_err();
        assert!(err.to_string().contains("its own parent"), "{err}");
    }

    #[test]
    fn reassign_rejects_alias_target() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n\n- bullet <!-- rq-bbbbbbbb -->\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        // rq-bbbbbbbb is an alias for rq-aaaaaaaa.
        let err = do_reassign(&store, &StableId::new("rq-bbbbbbbb"), vec![])
            .unwrap_err();
        assert!(err.to_string().contains("is an alias"), "{err}");
    }

    #[test]
    fn reassign_rejects_alias_parent() {
        let (dir, store) = setup();
        let root = dir.path();
        let md = "# Top <!-- rq-aaaaaaaa -->\n\n- bullet <!-- rq-bbbbbbbb -->\n";
        write_at(root, "rqm/doc.md", md);
        migrate::migrate_markdown(&store, root, Path::new("rqm/doc.md")).unwrap();
        // Create a separate canonical to reassign.
        make_req(&store, "rq-cccccccc");
        let err = do_reassign(
            &store,
            &StableId::new("rq-cccccccc"),
            vec![StableId::new("rq-bbbbbbbb")],
        )
        .unwrap_err();
        assert!(err.to_string().contains("is an alias"), "{err}");
    }

    // ── Cycle detection ──────────────────────────────────────────────

    #[test]
    fn reassign_rejects_direct_cycle() {
        // A → B; now try to set B's parent to A (already that),
        // then try to set A's parent to B — would create a cycle.
        let (_dir, store) = setup();
        make_req(&store, "rq-aaaaaaaa");
        make_req_with_parents(
            &store,
            "rq-bbbbbbbb",
            vec![StableId::new("rq-aaaaaaaa")],
        );
        let err = do_reassign(
            &store,
            &StableId::new("rq-aaaaaaaa"),
            vec![StableId::new("rq-bbbbbbbb")],
        )
        .unwrap_err();
        assert!(err.to_string().contains("cycle"), "{err}");
    }

    #[test]
    fn reassign_rejects_indirect_cycle() {
        // A → B → C; try to set A's parent to C — would create cycle A→C→B→A.
        let (_dir, store) = setup();
        make_req(&store, "rq-aaaaaaaa");
        make_req_with_parents(
            &store,
            "rq-bbbbbbbb",
            vec![StableId::new("rq-aaaaaaaa")],
        );
        make_req_with_parents(
            &store,
            "rq-cccccccc",
            vec![StableId::new("rq-bbbbbbbb")],
        );
        let err = do_reassign(
            &store,
            &StableId::new("rq-aaaaaaaa"),
            vec![StableId::new("rq-cccccccc")],
        )
        .unwrap_err();
        assert!(err.to_string().contains("cycle"), "{err}");
    }

    #[test]
    fn reassign_allows_diamond_dag() {
        // A → B, A → C, then D wants both B and C as parents (diamond).
        let (dir, store) = setup();
        let root = dir.path();
        make_req(&store, "rq-aaaaaaaa");
        make_req_with_parents(
            &store,
            "rq-bbbbbbbb",
            vec![StableId::new("rq-aaaaaaaa")],
        );
        make_req_with_parents(
            &store,
            "rq-cccccccc",
            vec![StableId::new("rq-aaaaaaaa")],
        );
        make_req(&store, "rq-dddddddd");
        do_reassign(
            &store,
            &StableId::new("rq-dddddddd"),
            vec![StableId::new("rq-bbbbbbbb"), StableId::new("rq-cccccccc")],
        )
        .unwrap();
        assert_eq!(
            parents_of(&store, "rq-dddddddd"),
            vec![StableId::new("rq-bbbbbbbb"), StableId::new("rq-cccccccc")]
        );
        let r = materialize::check(&store, root).unwrap();
        assert!(r.is_clean(), "{r:?}");
    }
}
