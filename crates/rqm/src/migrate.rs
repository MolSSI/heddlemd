//! Migration of existing markdown and source files into `.rqm/` objects.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::object::{Blob, FileTree, FileTreeEntry, Kind, ObjectHash, Requirement, StableId};
use crate::store::Store;

/// Migrate a markdown requirements file into the store. The file must
/// start with a level-1 (or higher) heading on line 1 — prelude content
/// before the first heading is not supported in Pilot 1.
///
/// `rel_path` is the path stored in the file-tree (typically the
/// repo-relative path). The file is read from `root.join(rel_path)`.
pub fn migrate_markdown(store: &Store, root: &Path, rel_path: &Path) -> Result<()> {
    if rel_path.is_absolute() {
        bail!("migrate path must be relative: {}", rel_path.display());
    }
    let physical = root.join(rel_path);
    let text = fs::read_to_string(&physical)
        .with_context(|| format!("read {}", physical.display()))?;

    let parsed = parse_boundaries(&text)
        .with_context(|| format!("parse {}", physical.display()))?;
    let ParseResult {
        boundaries,
        aliases,
    } = parsed;

    if boundaries.is_empty() {
        bail!("no rq-tagged headings found in {}", rel_path.display());
    }
    if boundaries[0].line != 0 {
        bail!(
            "{}: first heading is not on line 1; Pilot 1 does not support prelude content",
            rel_path.display()
        );
    }

    // split_inclusive keeps trailing newlines on each line, so concatenating
    // any contiguous slice of `lines` reproduces the corresponding span of
    // the file byte-for-byte.
    let lines: Vec<&str> = text.split_inclusive('\n').collect();

    // Slice the file into one blob per boundary. Each blob spans from the
    // boundary's line index to (but not including) the next boundary's
    // line index, or to EOF for the last boundary.
    let mut blob_hashes = Vec::with_capacity(boundaries.len());
    for (i, bn) in boundaries.iter().enumerate() {
        let start = bn.line;
        let end = boundaries.get(i + 1).map(|b| b.line).unwrap_or(lines.len());
        let mut bytes = Vec::new();
        for line in &lines[start..end] {
            bytes.extend_from_slice(line.as_bytes());
        }
        let h = store.write_blob(&Blob(bytes))?;
        blob_hashes.push(h);
    }

    // Write metas. Order doesn't matter — `parent` is a stable_id, not a
    // hash, so there's no write-ordering dependency between parent and child.
    for (bn, blob) in boundaries.iter().zip(blob_hashes.iter()) {
        let req = Requirement {
            stable_id: bn.stable_id.clone(),
            kind: bn.kind,
            text_blob: *blob,
            parent: bn.parent_id.clone(),
            source_blobs: vec![],
        };
        let meta_hash = store.write_requirement(&req)?;
        store.ref_set(&bn.stable_id, &meta_hash)?;
    }

    // Write the file-tree.
    let entries = boundaries
        .iter()
        .zip(blob_hashes.iter())
        .map(|(bn, blob)| FileTreeEntry {
            stable_id: bn.stable_id.clone(),
            blob: *blob,
        })
        .collect();
    let tree = FileTree {
        path: rel_path.to_path_buf(),
        entries,
    };
    let tree_hash = store.write_file_tree(&tree)?;
    store.tree_set(rel_path, &tree_hash)?;

    // Record bullet-level stamps as aliases for their enclosing heading.
    for (alias, canonical) in &aliases {
        store.alias_set(alias, canonical)?;
    }

    // Ensure the path is in managed_paths.
    add_managed_path(store, rel_path)?;

    Ok(())
}

fn add_managed_path(store: &Store, path: &Path) -> Result<()> {
    let mut paths = store.managed_paths()?;
    if !paths.iter().any(|p| p == path) {
        paths.push(path.to_path_buf());
        paths.sort();
        store.set_managed_paths(&paths)?;
    }
    Ok(())
}

/// A requirement boundary identified by parsing — the line where a
/// requirement's text-blob begins.
#[derive(Debug, Clone)]
struct Boundary {
    line: usize,
    stable_id: StableId,
    parent_id: Option<StableId>,
    kind: Kind,
    #[allow(dead_code)]
    source: BoundarySource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundarySource {
    Heading,
    GherkinScenario,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FenceState {
    None,
    Code,
    Gherkin,
}

#[derive(Debug, Default)]
struct ParseResult {
    boundaries: Vec<Boundary>,
    /// (alias_id, canonical_stable_id) — see "Bullet-level stamps and
    /// migration aliases" in rqm-spec.md.
    aliases: Vec<(StableId, StableId)>,
}

fn parse_boundaries(text: &str) -> Result<ParseResult> {
    let mut out = ParseResult::default();
    let mut fence = FenceState::None;
    // Stack of (level, stable_id) for currently-open headings. When a new
    // heading at level N is seen, all entries at level ≥ N are popped.
    let mut heading_stack: Vec<(u8, StableId)> = Vec::new();
    // Some(level) means we are currently inside an Out-of-Scope subtree
    // rooted at the heading at that level. Cleared when a heading at that
    // level (or shallower) appears.
    let mut oos_level: Option<u8> = None;

    for (idx, raw_line) in text.split('\n').enumerate() {
        let line = raw_line.trim_end_matches('\r');

        if let Some(lang) = detect_fence(line) {
            fence = match fence {
                FenceState::None => match lang {
                    "gherkin" => FenceState::Gherkin,
                    _ => FenceState::Code,
                },
                _ => FenceState::None,
            };
            continue;
        }

        match fence {
            FenceState::Code => {}
            FenceState::Gherkin => {
                if let Some(id) = parse_scenario_tag(line) {
                    let parent_id = heading_stack.last().map(|(_, id)| id.clone());
                    let kind = if oos_level.is_some() { Kind::Design } else { Kind::Behavior };
                    out.boundaries.push(Boundary {
                        line: idx,
                        stable_id: id,
                        parent_id,
                        kind,
                        source: BoundarySource::GherkinScenario,
                    });
                }
            }
            FenceState::None => {
                if let Some((level, heading_text, id)) = parse_heading(line) {
                    while heading_stack.last().map_or(false, |(l, _)| *l >= level) {
                        heading_stack.pop();
                    }
                    if let Some(l) = oos_level {
                        if level <= l {
                            oos_level = None;
                        }
                    }

                    let parent_id = heading_stack.last().map(|(_, id)| id.clone());
                    let entering_oos = heading_text.trim().eq_ignore_ascii_case("Out of Scope");
                    let kind = if oos_level.is_some() || entering_oos {
                        Kind::Design
                    } else {
                        Kind::Behavior
                    };
                    if entering_oos {
                        oos_level = Some(level);
                    }

                    out.boundaries.push(Boundary {
                        line: idx,
                        stable_id: id.clone(),
                        parent_id,
                        kind,
                        source: BoundarySource::Heading,
                    });
                    heading_stack.push((level, id));
                } else if let Some(alias_id) = parse_bullet_stamp(line) {
                    // A bullet with an rq stamp inside a heading region.
                    // Record as an alias for the enclosing heading rather
                    // than promoting to a first-class requirement (see
                    // rqm-spec.md, "Bullet-level stamps and migration
                    // aliases").
                    if let Some((_, canonical)) = heading_stack.last() {
                        out.aliases.push((alias_id, canonical.clone()));
                    }
                    // Bullets outside any heading are ignored — pathological
                    // input that the markdown structure makes unlikely.
                }
            }
        }
    }

    Ok(out)
}

/// Parses a bullet line carrying a trailing `<!-- rq-XXXXXXXX -->` stamp.
/// Returns the stable_id, or None if the line is not such a bullet.
fn parse_bullet_stamp(line: &str) -> Option<StableId> {
    let trimmed = line.trim_start();
    // Standard markdown bullet markers; we accept `- ` and `* ` though our
    // data only uses `- `.
    if !(trimmed.starts_with("- ") || trimmed.starts_with("* ")) {
        return None;
    }
    let end_trimmed = trimmed.trim_end();
    if !end_trimmed.ends_with("-->") {
        return None;
    }
    let comment_start = end_trimmed.rfind("<!--")?;
    let inner = end_trimmed[comment_start + 4..end_trimmed.len() - 3].trim();
    parse_rq_id(inner)
}

/// Returns the language tag if `line` is a code fence, otherwise None.
/// The language tag may be empty for a fence with no language.
fn detect_fence(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with("```") {
        return None;
    }
    let after = trimmed.trim_start_matches('`');
    // The number of backticks must be at least 3.
    let backtick_count = trimmed.len() - after.len();
    if backtick_count < 3 {
        return None;
    }
    Some(after.trim())
}

/// Parses a heading line. Returns (level, heading_text, stable_id).
fn parse_heading(line: &str) -> Option<(u8, &str, StableId)> {
    let bytes = line.as_bytes();
    let mut level: usize = 0;
    while level < bytes.len() && level < 6 && bytes[level] == b'#' {
        level += 1;
    }
    if level == 0 || level > 6 {
        return None;
    }
    if bytes.get(level) != Some(&b' ') {
        return None;
    }

    // Heading line must end with "-->" preceded by a valid rq-XXXXXXXX comment.
    let trimmed = line.trim_end();
    if !trimmed.ends_with("-->") {
        return None;
    }
    let comment_start = trimmed.rfind("<!--")?;
    let inner = trimmed[comment_start + 4..trimmed.len() - 3].trim();
    let id = parse_rq_id(inner)?;
    let heading_text = trimmed[level + 1..comment_start].trim();
    Some((level as u8, heading_text, id))
}

/// Parses a Gherkin scenario tag line like `  @rq-cfd1d536`. Returns the
/// stable ID, or None if the line is not a scenario tag.
fn parse_scenario_tag(line: &str) -> Option<StableId> {
    let trimmed = line.trim();
    let body = trimmed.strip_prefix('@')?;
    parse_rq_id(body)
}

/// Parses `rq-XXXXXXXX` (exactly 8 lowercase hex chars). Returns None on
/// any deviation. Trailing whitespace in the input is acceptable; leading
/// whitespace is not (the caller is expected to have trimmed).
fn parse_rq_id(s: &str) -> Option<StableId> {
    let s = s.trim_end();
    if s.len() != 11 {
        return None;
    }
    if !s.starts_with("rq-") {
        return None;
    }
    if !s[3..].chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()) {
        return None;
    }
    Some(StableId::new(s))
}

// ── Source-file migration ─────────────────────────────────────────────────

/// Migrate a source file (or test file, or any file with `rq-XXXXXXXX`
/// stamps) into the store. Requires every stamp's stable_id to already
/// have a meta in the store — i.e., the corresponding requirements file
/// must already be migrated.
///
/// `rel_path` is the path stored in the file-tree; the file is read
/// from `root.join(rel_path)`.
pub fn migrate_source(store: &Store, root: &Path, rel_path: &Path) -> Result<()> {
    if rel_path.is_absolute() {
        bail!("migrate path must be relative: {}", rel_path.display());
    }
    let physical = root.join(rel_path);
    let text = fs::read_to_string(&physical)
        .with_context(|| format!("read {}", physical.display()))?;
    let slices = slice_source(&text);
    if slices.is_empty() {
        bail!("no rq stamps found in {}", rel_path.display());
    }

    // Resolve every owner stamp to its canonical stable_id (handling
    // aliases for bullet-level stamps). Fails if any stamp is unknown.
    let resolved: Vec<Vec<StableId>> = slices
        .iter()
        .map(|slice| -> Result<Vec<StableId>> {
            slice
                .owners
                .iter()
                .map(|owner| {
                    store.resolve(owner)?.ok_or_else(|| {
                        anyhow::anyhow!(
                            "{}: stamp {} has no corresponding requirement in .rqm/refs/ \
                             or .rqm/aliases/ (migrate the requirements file first)",
                            rel_path.display(),
                            owner
                        )
                    })
                })
                .collect()
        })
        .collect::<Result<_>>()?;

    // Write each slice as a blob; group blob hashes by canonical owner.
    let mut blob_hashes = Vec::with_capacity(slices.len());
    let mut blobs_by_owner: HashMap<StableId, Vec<ObjectHash>> = HashMap::new();
    for (slice, canonical_owners) in slices.iter().zip(resolved.iter()) {
        let h = store.write_blob(&Blob(slice.bytes.clone()))?;
        blob_hashes.push(h);
        for owner in canonical_owners {
            blobs_by_owner.entry(owner.clone()).or_default().push(h);
        }
    }

    // For each canonical owner that received new blobs, load its current
    // meta, extend `source_blobs`, rewrite, and update the ref.
    let mut owners: Vec<_> = blobs_by_owner.iter().collect();
    owners.sort_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
    for (owner, new_blobs) in owners {
        let cur = store
            .ref_get(owner)?
            .expect("canonical owner resolved above");
        let mut req = store.read_requirement(&cur)?;
        for h in new_blobs {
            if !req.source_blobs.contains(h) {
                req.source_blobs.push(*h);
            }
        }
        req.source_blobs.sort();
        let new_hash = store.write_requirement(&req)?;
        store.ref_set(owner, &new_hash)?;
    }

    // Build the file-tree. The entry's stable_id is the *canonical* form
    // of the first owner on the stamp line; alias IDs do not appear in
    // file-trees (only in the underlying blob bytes via the stamp text).
    let entries: Vec<FileTreeEntry> = resolved
        .iter()
        .zip(blob_hashes.iter())
        .map(|(canonical_owners, blob)| FileTreeEntry {
            stable_id: canonical_owners[0].clone(),
            blob: *blob,
        })
        .collect();
    let tree = FileTree {
        path: rel_path.to_path_buf(),
        entries,
    };
    let tree_hash = store.write_file_tree(&tree)?;
    store.tree_set(rel_path, &tree_hash)?;

    add_managed_path(store, rel_path)?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceSlice {
    owners: Vec<StableId>,
    bytes: Vec<u8>,
}

/// Slice a source file into (owners, bytes) tuples at every rq-stamp
/// boundary. The first slice begins at line 1 (including any prelude
/// before the first stamp); subsequent slices begin at their stamp line.
/// Returns an empty vec if no stamps are found.
fn slice_source(text: &str) -> Vec<SourceSlice> {
    let lines: Vec<&str> = text.split_inclusive('\n').collect();

    let mut stamps: Vec<(usize, Vec<StableId>)> = Vec::new();
    for (idx, raw_line) in lines.iter().enumerate() {
        let ids = find_stamps(raw_line);
        if !ids.is_empty() {
            stamps.push((idx, ids));
        }
    }
    if stamps.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(stamps.len());
    for (i, (stamp_line, ids)) in stamps.iter().enumerate() {
        let start = if i == 0 { 0 } else { *stamp_line };
        let end = stamps.get(i + 1).map(|(l, _)| *l).unwrap_or(lines.len());
        let mut bytes = Vec::new();
        for line in &lines[start..end] {
            bytes.extend_from_slice(line.as_bytes());
        }
        out.push(SourceSlice {
            owners: ids.clone(),
            bytes,
        });
    }
    out
}

/// Find every `rq-XXXXXXXX` token on a line, in order. Tokens must be
/// word-bounded: the surrounding characters (if any) must not be ASCII
/// alphanumeric or underscore.
fn find_stamps(line: &str) -> Vec<StableId> {
    let bytes = line.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 11 <= bytes.len() {
        if &bytes[i..i + 3] == b"rq-" {
            let prior_ok = i == 0 || !is_word_byte(bytes[i - 1]);
            let id_chars = &bytes[i + 3..i + 11];
            let hex_ok = id_chars
                .iter()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase());
            let trail_ok = bytes
                .get(i + 11)
                .map_or(true, |c| !is_word_byte(*c));
            if prior_ok && hex_ok && trail_ok {
                // bytes[i..i+11] is valid ASCII by construction.
                let s = std::str::from_utf8(&bytes[i..i + 11]).unwrap();
                out.push(StableId::new(s));
                i += 11;
                continue;
            }
        }
        i += 1;
    }
    out
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(boundaries: &[Boundary]) -> Vec<String> {
        boundaries.iter().map(|b| b.stable_id.to_string()).collect()
    }

    fn parse(text: &str) -> ParseResult {
        parse_boundaries(text).unwrap()
    }

    #[test]
    fn parses_simple_heading() {
        let h = parse_heading("# Top <!-- rq-abcdef01 -->").unwrap();
        assert_eq!(h.0, 1);
        assert_eq!(h.1, "Top");
        assert_eq!(h.2.to_string(), "rq-abcdef01");
    }

    #[test]
    fn rejects_uppercase_hex() {
        assert!(parse_heading("# X <!-- rq-ABCDEF01 -->").is_none());
    }

    #[test]
    fn rejects_wrong_length_id() {
        assert!(parse_heading("# X <!-- rq-abc -->").is_none());
        assert!(parse_heading("# X <!-- rq-abcdef012 -->").is_none());
    }

    #[test]
    fn parses_scenario_tag() {
        let id = parse_scenario_tag("  @rq-cfd1d536").unwrap();
        assert_eq!(id.to_string(), "rq-cfd1d536");
    }

    #[test]
    fn ignores_code_fence_contents() {
        let md = "\
# Top <!-- rq-aaaaaaaa -->

```rust
// rq-bbbbbbbb — should be ignored, not a markdown rq
```

## Sub <!-- rq-cccccccc -->
";
        let p = parse(md);
        assert_eq!(ids(&p.boundaries), vec!["rq-aaaaaaaa", "rq-cccccccc"]);
    }

    #[test]
    fn captures_gherkin_scenarios() {
        let md = "\
# Top <!-- rq-aaaaaaaa -->

## Scenarios <!-- rq-bbbbbbbb -->

```gherkin
Feature: Foo

  @rq-cccccccc
  Scenario: A

  @rq-dddddddd
  Scenario: B
```
";
        let p = parse(md);
        assert_eq!(
            ids(&p.boundaries),
            vec!["rq-aaaaaaaa", "rq-bbbbbbbb", "rq-cccccccc", "rq-dddddddd"]
        );
        assert_eq!(p.boundaries[2].parent_id.as_ref().unwrap().to_string(), "rq-bbbbbbbb");
        assert_eq!(p.boundaries[3].parent_id.as_ref().unwrap().to_string(), "rq-bbbbbbbb");
    }

    #[test]
    fn assigns_parents_by_heading_level() {
        let md = "\
# A <!-- rq-aaaaaaaa -->
## B <!-- rq-bbbbbbbb -->
### C <!-- rq-cccccccc -->
## D <!-- rq-dddddddd -->
";
        let p = parse(md);
        assert_eq!(p.boundaries[0].parent_id, None);
        assert_eq!(p.boundaries[1].parent_id.as_ref().unwrap().to_string(), "rq-aaaaaaaa");
        assert_eq!(p.boundaries[2].parent_id.as_ref().unwrap().to_string(), "rq-bbbbbbbb");
        assert_eq!(p.boundaries[3].parent_id.as_ref().unwrap().to_string(), "rq-aaaaaaaa");
    }

    #[test]
    fn out_of_scope_subtree_gets_design_kind() {
        let md = "\
# A <!-- rq-aaaaaaaa -->
## Out of Scope <!-- rq-bbbbbbbb -->
### Sub <!-- rq-cccccccc -->
## After <!-- rq-dddddddd -->
";
        let p = parse(md);
        assert_eq!(p.boundaries[0].kind, Kind::Behavior);
        assert_eq!(p.boundaries[1].kind, Kind::Design);
        assert_eq!(p.boundaries[2].kind, Kind::Design);
        assert_eq!(p.boundaries[3].kind, Kind::Behavior);
    }

    #[test]
    fn bullets_become_aliases_for_enclosing_heading() {
        let md = "\
# Top <!-- rq-aaaaaaaa -->

## Types <!-- rq-bbbbbbbb -->

- `Foo` — first type. <!-- rq-cccccccc -->
- `Bar` — second type. <!-- rq-dddddddd -->

## Other <!-- rq-eeeeeeee -->

- `Baz` <!-- rq-ffffffff -->
";
        let p = parse(md);
        // No bullets in boundaries — only headings.
        assert_eq!(
            ids(&p.boundaries),
            vec!["rq-aaaaaaaa", "rq-bbbbbbbb", "rq-eeeeeeee"]
        );
        // Three aliases, each pointing at its enclosing heading.
        let aliases: Vec<(String, String)> = p
            .aliases
            .iter()
            .map(|(a, c)| (a.to_string(), c.to_string()))
            .collect();
        assert_eq!(
            aliases,
            vec![
                ("rq-cccccccc".to_string(), "rq-bbbbbbbb".to_string()),
                ("rq-dddddddd".to_string(), "rq-bbbbbbbb".to_string()),
                ("rq-ffffffff".to_string(), "rq-eeeeeeee".to_string()),
            ]
        );
    }

    fn fresh_store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::init(dir.path()).unwrap();
        (dir, store)
    }

    #[test]
    fn migrate_round_trips_small_file() {
        let (_tmp, store) = fresh_store();
        let md = "\
# Top <!-- rq-aaaaaaaa -->

Body of top.

## Child <!-- rq-bbbbbbbb -->

Body of child.
";
        let work = tempfile::tempdir().unwrap();
        let rel = Path::new("doc.md");
        fs::write(work.path().join(rel), md).unwrap();

        migrate_markdown(&store, work.path(), rel).unwrap();

        let h_top = store.ref_get(&StableId::new("rq-aaaaaaaa")).unwrap().unwrap();
        let h_child = store.ref_get(&StableId::new("rq-bbbbbbbb")).unwrap().unwrap();
        let top = store.read_requirement(&h_top).unwrap();
        let child = store.read_requirement(&h_child).unwrap();
        assert_eq!(top.parent, None);
        assert_eq!(child.parent, Some(StableId::new("rq-aaaaaaaa")));

        let tree_hash = store.tree_get(rel).unwrap().expect("tree ref");
        let tree = store.read_file_tree(&tree_hash).unwrap();
        let mut materialized = Vec::new();
        for e in &tree.entries {
            let b = store.read_blob(&e.blob).unwrap();
            materialized.extend_from_slice(&b.0);
        }
        assert_eq!(String::from_utf8(materialized).unwrap(), md);
    }

    #[test]
    fn migrate_writes_managed_path() {
        let (_tmp, store) = fresh_store();
        let work = tempfile::tempdir().unwrap();
        let rel = Path::new("doc.md");
        fs::write(work.path().join(rel), "# T <!-- rq-aaaaaaaa -->\n").unwrap();

        migrate_markdown(&store, work.path(), rel).unwrap();
        let paths = store.managed_paths().unwrap();
        assert!(paths.iter().any(|p| p == rel));
    }

    // ── Source migration ──────────────────────────────────────────────

    #[test]
    fn find_stamps_single() {
        let line = "// rq-4d1082c4 — Radial distribution function";
        let ids = find_stamps(line);
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0].to_string(), "rq-4d1082c4");
    }

    #[test]
    fn find_stamps_multi() {
        let line = "// rq-43567b30 rq-60c534f2";
        let ids = find_stamps(line);
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0].to_string(), "rq-43567b30");
        assert_eq!(ids[1].to_string(), "rq-60c534f2");
    }

    #[test]
    fn find_stamps_inside_html_comment() {
        let line = "// Feature: Radial Distribution Function (`rdf`) <!-- rq-4d1082c4 -->";
        let ids = find_stamps(line);
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0].to_string(), "rq-4d1082c4");
    }

    #[test]
    fn find_stamps_rejects_word_boundary_violations() {
        // Leading word char: `xrq-12345678` is part of a larger identifier.
        assert!(find_stamps("xrq-12345678").is_empty());
        // Trailing word char: 9-char tail.
        assert!(find_stamps("rq-123456789").is_empty());
        // Underscore as boundary char.
        assert!(find_stamps("_rq-12345678").is_empty());
        assert!(find_stamps("rq-12345678_").is_empty());
        // Uppercase hex is invalid (we generate lowercase only).
        assert!(find_stamps("rq-ABCDEF01").is_empty());
    }

    #[test]
    fn slice_source_includes_prelude_in_first_slice() {
        let text = "\
// Tests for the RDF analysis feature

// Feature: RDF <!-- rq-4d1082c4 -->
mod tests {
    // rq-cfd1d536
    fn test_one() {}

    // rq-d4c17bd3
    fn test_two() {}
}
";
        let slices = slice_source(text);
        assert_eq!(slices.len(), 3);

        // First slice owns the file from line 0; first stamp is on line 2.
        assert_eq!(slices[0].owners[0].to_string(), "rq-4d1082c4");
        assert!(
            std::str::from_utf8(&slices[0].bytes)
                .unwrap()
                .starts_with("// Tests for the RDF analysis feature\n\n// Feature: RDF"),
            "first slice should include prelude: {:?}",
            std::str::from_utf8(&slices[0].bytes)
        );

        // Concatenation reproduces the original.
        let cat: Vec<u8> = slices.iter().flat_map(|s| s.bytes.iter().copied()).collect();
        assert_eq!(cat, text.as_bytes());
    }

    #[test]
    fn slice_source_joint_ownership() {
        let text = "\
// rq-aaaaaaaa
fn one() {}

// rq-bbbbbbbb rq-cccccccc
fn shared() {}

// rq-dddddddd
fn last() {}
";
        let slices = slice_source(text);
        assert_eq!(slices.len(), 3);
        assert_eq!(slices[1].owners.len(), 2);
        assert_eq!(slices[1].owners[0].to_string(), "rq-bbbbbbbb");
        assert_eq!(slices[1].owners[1].to_string(), "rq-cccccccc");
    }

    fn make_req(store: &Store, id: &str) -> ObjectHash {
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
        h
    }

    #[test]
    fn migrate_source_round_trip() {
        let (_tmp, store) = fresh_store();
        make_req(&store, "rq-aaaaaaaa");
        make_req(&store, "rq-bbbbbbbb");

        let work = tempfile::tempdir().unwrap();
        let rel = Path::new("src.rs");
        let text = "\
// prelude line

// rq-aaaaaaaa
fn alpha() {}

// rq-bbbbbbbb
fn beta() {}
";
        fs::write(work.path().join(rel), text).unwrap();

        migrate_source(&store, work.path(), rel).unwrap();

        let tree_hash = store.tree_get(rel).unwrap().unwrap();
        let tree = store.read_file_tree(&tree_hash).unwrap();
        let mut materialized = Vec::new();
        for e in &tree.entries {
            let b = store.read_blob(&e.blob).unwrap();
            materialized.extend_from_slice(&b.0);
        }
        assert_eq!(std::str::from_utf8(&materialized).unwrap(), text);
    }

    #[test]
    fn migrate_source_joint_ownership_appears_in_both_metas() {
        let (_tmp, store) = fresh_store();
        make_req(&store, "rq-aaaaaaaa");
        make_req(&store, "rq-bbbbbbbb");
        make_req(&store, "rq-cccccccc");

        let work = tempfile::tempdir().unwrap();
        let rel = Path::new("src.rs");
        fs::write(
            work.path().join(rel),
            "\
// rq-aaaaaaaa
fn one() {}

// rq-bbbbbbbb rq-cccccccc
fn shared() {}
",
        )
        .unwrap();

        migrate_source(&store, work.path(), rel).unwrap();

        let h_b = store.ref_get(&StableId::new("rq-bbbbbbbb")).unwrap().unwrap();
        let h_c = store.ref_get(&StableId::new("rq-cccccccc")).unwrap().unwrap();
        let req_b = store.read_requirement(&h_b).unwrap();
        let req_c = store.read_requirement(&h_c).unwrap();
        assert_eq!(req_b.source_blobs.len(), 1);
        assert_eq!(req_c.source_blobs.len(), 1);
        assert_eq!(req_b.source_blobs[0], req_c.source_blobs[0]);

        let tree_hash = store.tree_get(rel).unwrap().unwrap();
        let tree = store.read_file_tree(&tree_hash).unwrap();
        assert_eq!(tree.entries[1].stable_id.to_string(), "rq-bbbbbbbb");
    }

    #[test]
    fn migrate_source_fails_when_meta_missing() {
        let (_tmp, store) = fresh_store();
        make_req(&store, "rq-bbbbbbbb");

        let work = tempfile::tempdir().unwrap();
        let rel = Path::new("src.rs");
        fs::write(
            work.path().join(rel),
            "// rq-aaaaaaaa\nfn x() {}\n// rq-bbbbbbbb\nfn y() {}\n",
        )
        .unwrap();

        let err = migrate_source(&store, work.path(), rel).unwrap_err();
        assert!(
            err.to_string().contains("rq-aaaaaaaa"),
            "error should mention missing id: {err}"
        );
    }
}
