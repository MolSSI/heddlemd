use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObjectHash(pub [u8; 32]);

impl ObjectHash {
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    pub fn from_hex(s: &str) -> Result<Self, FromHexError> {
        if s.len() != 64 {
            return Err(FromHexError::WrongLength(s.len()));
        }
        let mut arr = [0u8; 32];
        hex::decode_to_slice(s, &mut arr).map_err(FromHexError::Hex)?;
        Ok(ObjectHash(arr))
    }
}

impl std::fmt::Display for ObjectHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl std::fmt::Debug for ObjectHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let hex = self.to_hex();
        write!(f, "ObjectHash({}...)", &hex[..12])
    }
}

impl Serialize for ObjectHash {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for ObjectHash {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        ObjectHash::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug)]
pub enum FromHexError {
    WrongLength(usize),
    Hex(hex::FromHexError),
}

impl std::fmt::Display for FromHexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FromHexError::WrongLength(n) => write!(f, "expected 64 hex chars, got {n}"),
            FromHexError::Hex(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for FromHexError {}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StableId(String);

impl StableId {
    pub fn new(s: impl Into<String>) -> Self {
        StableId(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for StableId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Behavior,
    Design,
    Pending,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Blob(pub Vec<u8>);

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(from = "RequirementRaw")]
pub struct Requirement {
    pub stable_id: StableId,
    pub kind: Kind,
    pub text_blob: ObjectHash,
    /// Parent requirements in the DAG. Empty for a DAG root.
    /// Multi-parent semantics: a requirement may have any number of
    /// parents. Canonical-sorted for hash stability.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parents: Vec<StableId>,
    pub source_blobs: Vec<ObjectHash>,
}

/// On-disk form used for backward-compatible deserialization. Accepts
/// the legacy single-parent `parent` field as well as the new `parents`
/// list. Writing always uses the new `parents` form (via the derived
/// Serialize impl on `Requirement`).
#[derive(Deserialize)]
struct RequirementRaw {
    stable_id: StableId,
    kind: Kind,
    text_blob: ObjectHash,
    #[serde(default)]
    parent: Option<StableId>,
    #[serde(default)]
    parents: Vec<StableId>,
    #[serde(default)]
    source_blobs: Vec<ObjectHash>,
}

impl From<RequirementRaw> for Requirement {
    fn from(raw: RequirementRaw) -> Self {
        let mut parents = raw.parents;
        // If only the legacy single-parent field is present, migrate
        // it into the new list.
        if parents.is_empty() {
            if let Some(p) = raw.parent {
                parents.push(p);
            }
        }
        parents.sort();
        parents.dedup();
        Requirement {
            stable_id: raw.stable_id,
            kind: raw.kind,
            text_blob: raw.text_blob,
            parents,
            source_blobs: raw.source_blobs,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct FileTree {
    pub path: PathBuf,
    pub entries: Vec<FileTreeEntry>,
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct FileTreeEntry {
    pub stable_id: StableId,
    pub blob: ObjectHash,
}
