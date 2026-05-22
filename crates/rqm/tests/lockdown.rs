//! Scoped lockdown check: every path listed in `.rqm/managed_paths` must
//! match the materializer's output. This is the executable form of the
//! round-trip invariant in `rqm-spec.md`; it runs on every `cargo test`
//! and fails CI when the working tree diverges from `.rqm/`.

use std::path::PathBuf;

use rqm::materialize;
use rqm::store::Store;

#[test]
fn managed_paths_match_rqm_store() {
    // The rqm crate lives at `<workspace-root>/crates/rqm/`, so the
    // workspace root is two levels up from CARGO_MANIFEST_DIR.
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("canonicalize workspace root");
    let rqm_dir = workspace_root.join(".rqm");

    if !rqm_dir.is_dir() {
        // No `.rqm/` yet — pre-migration. The test exists but has nothing
        // to enforce. This branch goes away once the migration is
        // expected on every clone.
        eprintln!(
            "no .rqm/ found at {}; lockdown check is a no-op until migration",
            rqm_dir.display()
        );
        return;
    }

    let store = Store::open(&rqm_dir).expect("open .rqm/");
    let report = materialize::check(&store, &workspace_root).expect("rqm check");

    if !report.is_clean() {
        let mut msg = String::from("\n.rqm/ is out of sync with the working tree:\n");
        for p in &report.diffs {
            msg.push_str(&format!("  diff: {}\n", p.display()));
        }
        for p in &report.missing {
            msg.push_str(&format!("  missing: {}\n", p.display()));
        }
        for v in &report.integrity {
            msg.push_str(&format!("  integrity: {v}\n"));
        }
        msg.push_str(
            "\nTo fix: either re-run `rqm migrate <path>` for the edited file,\n\
             or run `rqm build` to overwrite the working tree with .rqm/ contents.\n",
        );
        panic!("{msg}");
    }
}
