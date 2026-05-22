//! On-disk object store rooted at `.rqm/`.

use std::fs;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::codec;
use crate::object::{Blob, FileTree, ObjectHash, Requirement, StableId};

const OBJECTS: &str = "objects";
const REFS: &str = "refs";
const ALIASES: &str = "aliases";
const TREES: &str = "trees";
const MANAGED_PATHS: &str = "managed_paths";

pub struct Store {
    root: PathBuf,
}

impl Store {
    /// Open an existing `.rqm/` directory. The directory must exist; use
    /// [`Store::init`] to create one.
    pub fn open(rqm_dir: impl AsRef<Path>) -> Result<Self> {
        let root = rqm_dir.as_ref().to_path_buf();
        if !root.is_dir() {
            bail!("not a directory: {}", root.display());
        }
        if !root.join(OBJECTS).is_dir() {
            bail!("missing {} subdir in {}", OBJECTS, root.display());
        }
        Ok(Store { root })
    }

    /// Create a fresh `.rqm/` directory with empty objects/, refs/,
    /// aliases/, trees/, and an empty managed_paths file.
    pub fn init(rqm_dir: impl AsRef<Path>) -> Result<Self> {
        let root = rqm_dir.as_ref().to_path_buf();
        for sub in [OBJECTS, REFS, ALIASES, TREES] {
            fs::create_dir_all(root.join(sub))
                .with_context(|| format!("create {}", root.join(sub).display()))?;
        }
        let mp = root.join(MANAGED_PATHS);
        if !mp.exists() {
            fs::write(&mp, b"").with_context(|| format!("create {}", mp.display()))?;
        }
        Ok(Store { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    // ── Object writes ─────────────────────────────────────────────────────

    pub fn write_blob(&self, blob: &Blob) -> Result<ObjectHash> {
        let hash = codec::hash_blob(blob);
        self.put_object_bytes(&hash, &blob.0)?;
        Ok(hash)
    }

    pub fn write_requirement(&self, req: &Requirement) -> Result<ObjectHash> {
        let bytes = codec::write_requirement(req)?;
        let hash = codec::hash_bytes(&bytes);
        self.put_object_bytes(&hash, &bytes)?;
        Ok(hash)
    }

    pub fn write_file_tree(&self, tree: &FileTree) -> Result<ObjectHash> {
        let bytes = codec::write_file_tree(tree)?;
        let hash = codec::hash_bytes(&bytes);
        self.put_object_bytes(&hash, &bytes)?;
        Ok(hash)
    }

    // ── Object reads ──────────────────────────────────────────────────────

    pub fn read_blob(&self, hash: &ObjectHash) -> Result<Blob> {
        let bytes = self.read_object_bytes(hash)?;
        Ok(Blob(bytes))
    }

    pub fn read_requirement(&self, hash: &ObjectHash) -> Result<Requirement> {
        let bytes = self.read_object_bytes(hash)?;
        codec::read_requirement(&bytes)
            .with_context(|| format!("decode requirement {hash}"))
    }

    pub fn read_file_tree(&self, hash: &ObjectHash) -> Result<FileTree> {
        let bytes = self.read_object_bytes(hash)?;
        codec::read_file_tree(&bytes)
            .with_context(|| format!("decode file_tree {hash}"))
    }

    pub fn has_object(&self, hash: &ObjectHash) -> bool {
        self.object_path(hash).is_file()
    }

    // ── Refs ──────────────────────────────────────────────────────────────

    pub fn ref_get(&self, id: &StableId) -> Result<Option<ObjectHash>> {
        let path = self.ref_path(id);
        match fs::read_to_string(&path) {
            Ok(s) => {
                let s = s.trim();
                Ok(Some(ObjectHash::from_hex(s).with_context(|| {
                    format!("ref {} contains invalid hash {:?}", id, s)
                })?))
            }
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(anyhow::Error::from(e).context(format!("read ref {id}"))),
        }
    }

    pub fn ref_set(&self, id: &StableId, hash: &ObjectHash) -> Result<()> {
        let path = self.ref_path(id);
        fs::write(&path, format!("{}\n", hash.to_hex()))
            .with_context(|| format!("write ref {id}"))?;
        Ok(())
    }

    /// Delete a ref. Idempotent — no error if the ref does not exist.
    pub fn ref_delete(&self, id: &StableId) -> Result<()> {
        let path = self.ref_path(id);
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
            Err(e) => Err(anyhow::Error::from(e).context(format!("delete ref {id}"))),
        }
    }

    pub fn ref_list(&self) -> Result<Vec<(StableId, ObjectHash)>> {
        let dir = self.root.join(REFS);
        let mut out = Vec::new();
        for entry in fs::read_dir(&dir).with_context(|| format!("read {}", dir.display()))? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_str().ok_or_else(|| {
                anyhow::anyhow!("non-utf8 ref filename: {:?}", entry.file_name())
            })?;
            let id = StableId::new(name);
            let hash = self.ref_get(&id)?.ok_or_else(|| {
                anyhow::anyhow!("ref {} disappeared during read", id)
            })?;
            out.push((id, hash));
        }
        out.sort_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
        Ok(out)
    }

    // ── Aliases (migration-only; stable_id → canonical stable_id) ─────────

    pub fn alias_get(&self, id: &StableId) -> Result<Option<StableId>> {
        let p = self.root.join(ALIASES).join(id.as_str());
        match fs::read_to_string(&p) {
            Ok(s) => Ok(Some(StableId::new(s.trim()))),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(anyhow::Error::from(e).context(format!("read alias {id}"))),
        }
    }

    pub fn alias_set(&self, alias: &StableId, canonical: &StableId) -> Result<()> {
        let p = self.root.join(ALIASES).join(alias.as_str());
        fs::write(&p, format!("{}\n", canonical.as_str()))
            .with_context(|| format!("write alias {alias}"))?;
        Ok(())
    }

    /// Delete an alias. Idempotent — no error if the alias does not exist.
    pub fn alias_delete(&self, alias: &StableId) -> Result<()> {
        let p = self.root.join(ALIASES).join(alias.as_str());
        match fs::remove_file(&p) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
            Err(e) => Err(anyhow::Error::from(e).context(format!("delete alias {alias}"))),
        }
    }

    pub fn alias_list(&self) -> Result<Vec<(StableId, StableId)>> {
        let dir = self.root.join(ALIASES);
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for entry in fs::read_dir(&dir).with_context(|| format!("read {}", dir.display()))? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_str().ok_or_else(|| {
                anyhow::anyhow!("non-utf8 alias filename: {:?}", entry.file_name())
            })?;
            let alias = StableId::new(name);
            let canonical = self.alias_get(&alias)?.ok_or_else(|| {
                anyhow::anyhow!("alias {} disappeared during read", alias)
            })?;
            out.push((alias, canonical));
        }
        out.sort_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
        Ok(out)
    }

    /// Resolve a stable_id to its canonical form by following one level
    /// of alias indirection. Returns `Some(id)` unchanged when `id` is
    /// already canonical (has a direct ref), `Some(canonical)` when
    /// `id` is an alias, and `None` when `id` is unknown.
    pub fn resolve(&self, id: &StableId) -> Result<Option<StableId>> {
        if self.ref_get(id)?.is_some() {
            return Ok(Some(id.clone()));
        }
        Ok(self.alias_get(id)?)
    }

    // ── Tree refs (path → file-tree hash) ─────────────────────────────────

    pub fn tree_get(&self, path: &Path) -> Result<Option<ObjectHash>> {
        let p = self.tree_ref_path(path);
        match fs::read_to_string(&p) {
            Ok(s) => {
                let s = s.trim();
                Ok(Some(ObjectHash::from_hex(s).with_context(|| {
                    format!("tree ref {} contains invalid hash {:?}", path.display(), s)
                })?))
            }
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(anyhow::Error::from(e).context(format!("read tree ref {}", path.display()))),
        }
    }

    pub fn tree_set(&self, path: &Path, hash: &ObjectHash) -> Result<()> {
        let p = self.tree_ref_path(path);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        fs::write(&p, format!("{}\n", hash.to_hex()))
            .with_context(|| format!("write tree ref {}", path.display()))?;
        Ok(())
    }

    // ── managed_paths ─────────────────────────────────────────────────────

    pub fn managed_paths(&self) -> Result<Vec<PathBuf>> {
        let path = self.root.join(MANAGED_PATHS);
        let text = fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        Ok(text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(PathBuf::from)
            .collect())
    }

    pub fn set_managed_paths(&self, paths: &[PathBuf]) -> Result<()> {
        let path = self.root.join(MANAGED_PATHS);
        let mut buf = String::new();
        for p in paths {
            buf.push_str(&p.to_string_lossy());
            buf.push('\n');
        }
        fs::write(&path, buf).with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    // ── Internals ─────────────────────────────────────────────────────────

    fn object_path(&self, hash: &ObjectHash) -> PathBuf {
        let hex = hash.to_hex();
        self.root.join(OBJECTS).join(&hex[..2]).join(&hex[2..])
    }

    fn ref_path(&self, id: &StableId) -> PathBuf {
        self.root.join(REFS).join(id.as_str())
    }

    fn tree_ref_path(&self, path: &Path) -> PathBuf {
        self.root.join(TREES).join(path)
    }

    fn put_object_bytes(&self, hash: &ObjectHash, bytes: &[u8]) -> Result<()> {
        let path = self.object_path(hash);
        // Idempotent: if the object already exists, skip the write. Content
        // addressing guarantees identical bytes for identical hashes.
        if path.is_file() {
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        // Write atomically via tempfile + rename so partial writes don't
        // leave invalid object files on disk.
        let tmp = path.with_extension("tmp");
        {
            let mut f = fs::File::create(&tmp)
                .with_context(|| format!("create {}", tmp.display()))?;
            f.write_all(bytes)
                .with_context(|| format!("write {}", tmp.display()))?;
        }
        fs::rename(&tmp, &path)
            .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
        Ok(())
    }

    fn read_object_bytes(&self, hash: &ObjectHash) -> Result<Vec<u8>> {
        let path = self.object_path(hash);
        fs::read(&path).with_context(|| format!("read object {}", hash))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::object::{FileTreeEntry, Kind};

    fn fresh_store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::init(dir.path()).unwrap();
        (dir, store)
    }

    #[test]
    fn blob_round_trip() {
        let (_dir, store) = fresh_store();
        let blob = Blob(b"hello world\n".to_vec());
        let hash = store.write_blob(&blob).unwrap();
        let back = store.read_blob(&hash).unwrap();
        assert_eq!(blob, back);
    }

    #[test]
    fn requirement_round_trip_via_store() {
        let (_dir, store) = fresh_store();
        let text_hash = store.write_blob(&Blob(b"prose".to_vec())).unwrap();
        let req = Requirement {
            stable_id: StableId::new("rq-4d1082c4"),
            kind: Kind::Behavior,
            text_blob: text_hash,
            parents: vec![],
            source_blobs: vec![],
        };
        let h = store.write_requirement(&req).unwrap();
        let back = store.read_requirement(&h).unwrap();
        assert_eq!(req, back);
    }

    #[test]
    fn refs_round_trip() {
        let (_dir, store) = fresh_store();
        let id = StableId::new("rq-deadbeef");
        assert!(store.ref_get(&id).unwrap().is_none());
        let h = ObjectHash([7; 32]);
        store.ref_set(&id, &h).unwrap();
        assert_eq!(store.ref_get(&id).unwrap(), Some(h));
        let listed = store.ref_list().unwrap();
        assert_eq!(listed, vec![(id, h)]);
    }

    #[test]
    fn managed_paths_round_trip() {
        let (_dir, store) = fresh_store();
        assert!(store.managed_paths().unwrap().is_empty());
        let paths = vec![
            PathBuf::from("rqm/analysis/rdf.md"),
            PathBuf::from("src/analysis/rdf.rs"),
            PathBuf::from("tests/rdf.rs"),
        ];
        store.set_managed_paths(&paths).unwrap();
        assert_eq!(store.managed_paths().unwrap(), paths);
    }

    #[test]
    fn write_is_idempotent() {
        let (_dir, store) = fresh_store();
        let blob = Blob(b"same".to_vec());
        let h1 = store.write_blob(&blob).unwrap();
        let h2 = store.write_blob(&blob).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn file_tree_round_trip() {
        let (_dir, store) = fresh_store();
        let blob_h = store.write_blob(&Blob(b"// rq-4d1082c4\n".to_vec())).unwrap();
        let tree = FileTree {
            path: PathBuf::from("src/analysis/rdf.rs"),
            entries: vec![FileTreeEntry {
                stable_id: StableId::new("rq-4d1082c4"),
                blob: blob_h,
            }],
        };
        let h = store.write_file_tree(&tree).unwrap();
        let back = store.read_file_tree(&h).unwrap();
        assert_eq!(tree, back);
    }
}
