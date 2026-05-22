//! Canonical JSON serialization and SHA-256 hashing for objects.

use anyhow::{Context, Result, bail};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::object::{Blob, FileTree, ObjectHash, Requirement};

const TAG_REQUIREMENT: &str = "requirement";
const TAG_FILE_TREE: &str = "file_tree";

pub fn hash_bytes(bytes: &[u8]) -> ObjectHash {
    let mut h = Sha256::new();
    h.update(bytes);
    ObjectHash(h.finalize().into())
}

pub fn hash_blob(blob: &Blob) -> ObjectHash {
    hash_bytes(&blob.0)
}

pub fn write_requirement(req: &Requirement) -> Result<Vec<u8>> {
    let mut v = serde_json::to_value(req).context("requirement to_value")?;
    inject_type(&mut v, TAG_REQUIREMENT)?;
    Ok(canonical_bytes(&v))
}

pub fn write_file_tree(tree: &FileTree) -> Result<Vec<u8>> {
    let mut v = serde_json::to_value(tree).context("file_tree to_value")?;
    inject_type(&mut v, TAG_FILE_TREE)?;
    Ok(canonical_bytes(&v))
}

pub fn read_requirement(bytes: &[u8]) -> Result<Requirement> {
    let mut v: Value = serde_json::from_slice(bytes).context("requirement json parse")?;
    take_type(&mut v, TAG_REQUIREMENT)?;
    serde_json::from_value(v).context("requirement decode")
}

pub fn read_file_tree(bytes: &[u8]) -> Result<FileTree> {
    let mut v: Value = serde_json::from_slice(bytes).context("file_tree json parse")?;
    take_type(&mut v, TAG_FILE_TREE)?;
    serde_json::from_value(v).context("file_tree decode")
}

fn canonical_bytes(value: &Value) -> Vec<u8> {
    // serde_json's default Map is BTreeMap (preserve_order feature OFF),
    // so to_vec emits keys in lexicographic order with no extra whitespace.
    // This is canonical for our purposes.
    serde_json::to_vec(value).expect("infallible Value serialization")
}

fn inject_type(v: &mut Value, tag: &str) -> Result<()> {
    let Value::Object(map) = v else {
        bail!("expected JSON object");
    };
    map.insert("type".to_string(), Value::String(tag.to_string()));
    Ok(())
}

fn take_type(v: &mut Value, expected: &str) -> Result<()> {
    let Value::Object(map) = v else {
        bail!("expected JSON object");
    };
    let tag = map
        .remove("type")
        .ok_or_else(|| anyhow::anyhow!("missing type field"))?;
    let Value::String(s) = tag else {
        bail!("type field must be a string");
    };
    if s != expected {
        bail!("expected type=\"{expected}\", got type=\"{s}\"");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::object::{FileTreeEntry, Kind, StableId};

    fn h(byte: u8) -> ObjectHash {
        ObjectHash([byte; 32])
    }

    #[test]
    fn requirement_round_trip() {
        let req = Requirement {
            stable_id: StableId::new("rq-4d1082c4"),
            kind: Kind::Behavior,
            text_blob: h(1),
            parent: None,
            source_blobs: vec![h(2), h(3)],
        };
        let bytes = write_requirement(&req).unwrap();
        let back = read_requirement(&bytes).unwrap();
        assert_eq!(req, back);
    }

    #[test]
    fn requirement_canonical_form_is_sorted() {
        let req = Requirement {
            stable_id: StableId::new("rq-aaaaaaaa"),
            kind: Kind::Design,
            text_blob: h(1),
            parent: Some(StableId::new("rq-bbbbbbbb")),
            source_blobs: vec![],
        };
        let bytes = write_requirement(&req).unwrap();
        let s = std::str::from_utf8(&bytes).unwrap();
        // Keys must appear in lexicographic order.
        let key_positions: Vec<_> = ["kind", "parent", "source_blobs", "stable_id", "text_blob", "type"]
            .iter()
            .map(|k| s.find(&format!("\"{k}\"")).unwrap_or(usize::MAX))
            .collect();
        let mut sorted = key_positions.clone();
        sorted.sort();
        assert_eq!(key_positions, sorted, "canonical form must have sorted keys: {s}");
    }

    #[test]
    fn requirement_omits_parent_when_none() {
        let req = Requirement {
            stable_id: StableId::new("rq-deadbeef"),
            kind: Kind::Behavior,
            text_blob: h(7),
            parent: None,
            source_blobs: vec![h(8)],
        };
        let bytes = write_requirement(&req).unwrap();
        let s = std::str::from_utf8(&bytes).unwrap();
        assert!(!s.contains("\"parent\""), "parent should be omitted: {s}");
    }

    #[test]
    fn file_tree_round_trip() {
        let tree = FileTree {
            path: PathBuf::from("src/analysis/rdf.rs"),
            entries: vec![FileTreeEntry {
                stable_id: StableId::new("rq-4d1082c4"),
                blob: h(42),
            }],
        };
        let bytes = write_file_tree(&tree).unwrap();
        let back = read_file_tree(&bytes).unwrap();
        assert_eq!(tree, back);
    }

    #[test]
    fn rejects_wrong_type_tag() {
        let req = Requirement {
            stable_id: StableId::new("rq-12345678"),
            kind: Kind::Behavior,
            text_blob: h(1),
            parent: None,
            source_blobs: vec![],
        };
        let bytes = write_requirement(&req).unwrap();
        let err = read_file_tree(&bytes).unwrap_err();
        assert!(err.to_string().contains("expected type=\"file_tree\""), "{err}");
    }
}
