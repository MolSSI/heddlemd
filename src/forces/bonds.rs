// rq-c2dbaa72 (bonds module — defined in forces/bonds.md)
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};

use crate::gpu::GpuError;

#[derive(Debug, Clone, Copy)]
pub struct Bond {
    pub atom_i: u32,
    pub atom_j: u32,
    pub bond_type_index: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct Exclusion {
    pub atom_i: u32,
    pub atom_j: u32,
    pub scale: f32,
}

#[derive(Debug, Clone)]
pub struct BondList {
    pub bonds: Vec<Bond>,
    pub atom_bond_offsets: Vec<u32>,
    pub atom_bond_indices: Vec<u32>,
    pub particle_count: usize,
}

impl BondList {
    pub fn empty(particle_count: usize) -> Self {
        BondList {
            bonds: Vec::new(),
            atom_bond_offsets: vec![0; particle_count + 1],
            atom_bond_indices: Vec::new(),
            particle_count,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.bonds.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct ExclusionList {
    pub entries: Vec<Exclusion>,
    pub atom_excl_offsets: Vec<u32>,
    pub atom_excl_partners: Vec<u32>,
    pub atom_excl_scales: Vec<f32>,
    pub particle_count: usize,
}

impl ExclusionList {
    pub fn empty(particle_count: usize) -> Self {
        ExclusionList {
            entries: Vec::new(),
            atom_excl_offsets: vec![0; particle_count + 1],
            atom_excl_partners: Vec::new(),
            atom_excl_scales: Vec::new(),
            particle_count,
        }
    }
}

/// Host-side handle around the exclusion list's three device buffers.
#[derive(Debug)]
pub struct DeviceExclusionList {
    pub atom_excl_offsets: CudaSlice<u32>,
    pub atom_excl_partners: CudaSlice<u32>,
    pub atom_excl_scales: CudaSlice<f32>,
    pub particle_count: usize,
}

impl DeviceExclusionList {
    pub fn from_host(
        device: &Arc<CudaDevice>,
        list: &ExclusionList,
    ) -> Result<Self, GpuError> {
        let atom_excl_offsets = device
            .htod_sync_copy(&list.atom_excl_offsets)
            .map_err(GpuError::from)?;
        let atom_excl_partners = if list.atom_excl_partners.is_empty() {
            device.alloc_zeros::<u32>(0).map_err(GpuError::from)?
        } else {
            device
                .htod_sync_copy(&list.atom_excl_partners)
                .map_err(GpuError::from)?
        };
        let atom_excl_scales = if list.atom_excl_scales.is_empty() {
            device.alloc_zeros::<f32>(0).map_err(GpuError::from)?
        } else {
            device
                .htod_sync_copy(&list.atom_excl_scales)
                .map_err(GpuError::from)?
        };
        Ok(DeviceExclusionList {
            atom_excl_offsets,
            atom_excl_partners,
            atom_excl_scales,
            particle_count: list.particle_count,
        })
    }
}

// rq-573b650b-style errors for bond file parsing.
#[derive(Debug)]
pub enum BondsFileError {
    Io(String),
    UnknownSection {
        name: String,
        line_number: usize,
    },
    DuplicateSection {
        name: String,
        line_number: usize,
    },
    ContentOutsideSection {
        line_number: usize,
    },
    InvalidBondRow {
        line_number: usize,
        reason: String,
    },
    InvalidExclusionRow {
        line_number: usize,
        reason: String,
    },
    AtomIndexOutOfRange {
        line_number: usize,
        index: u32,
        max: u32,
    },
    SelfBond {
        line_number: usize,
        atom: u32,
    },
    SelfExclusion {
        line_number: usize,
        atom: u32,
    },
    DuplicateBond {
        atom_i: u32,
        atom_j: u32,
    },
    DuplicateExclusion {
        atom_i: u32,
        atom_j: u32,
    },
    UnknownBondType {
        line_number: usize,
        name: String,
    },
    ScaleOutOfRange {
        line_number: usize,
        scale: f32,
    },
}

impl std::fmt::Display for BondsFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BondsFileError::Io(s) => write!(f, "Io({s})"),
            BondsFileError::UnknownSection { name, line_number } => write!(
                f,
                "UnknownSection {{ name: {name:?}, line_number: {line_number} }}"
            ),
            BondsFileError::DuplicateSection { name, line_number } => write!(
                f,
                "DuplicateSection {{ name: {name:?}, line_number: {line_number} }}"
            ),
            BondsFileError::ContentOutsideSection { line_number } => write!(
                f,
                "ContentOutsideSection {{ line_number: {line_number} }}"
            ),
            BondsFileError::InvalidBondRow {
                line_number,
                reason,
            } => write!(
                f,
                "InvalidBondRow {{ line_number: {line_number}, reason: {reason:?} }}"
            ),
            BondsFileError::InvalidExclusionRow {
                line_number,
                reason,
            } => write!(
                f,
                "InvalidExclusionRow {{ line_number: {line_number}, reason: {reason:?} }}"
            ),
            BondsFileError::AtomIndexOutOfRange {
                line_number,
                index,
                max,
            } => write!(
                f,
                "AtomIndexOutOfRange {{ line_number: {line_number}, index: {index}, max: {max} }}"
            ),
            BondsFileError::SelfBond { line_number, atom } => write!(
                f,
                "SelfBond {{ line_number: {line_number}, atom: {atom} }}"
            ),
            BondsFileError::SelfExclusion { line_number, atom } => write!(
                f,
                "SelfExclusion {{ line_number: {line_number}, atom: {atom} }}"
            ),
            BondsFileError::DuplicateBond { atom_i, atom_j } => write!(
                f,
                "DuplicateBond {{ atom_i: {atom_i}, atom_j: {atom_j} }}"
            ),
            BondsFileError::DuplicateExclusion { atom_i, atom_j } => write!(
                f,
                "DuplicateExclusion {{ atom_i: {atom_i}, atom_j: {atom_j} }}"
            ),
            BondsFileError::UnknownBondType { line_number, name } => write!(
                f,
                "UnknownBondType {{ line_number: {line_number}, name: {name:?} }}"
            ),
            BondsFileError::ScaleOutOfRange { line_number, scale } => write!(
                f,
                "ScaleOutOfRange {{ line_number: {line_number}, scale: {scale} }}"
            ),
        }
    }
}

impl std::error::Error for BondsFileError {}

pub fn load_bonds_file(
    path: &Path,
    particle_count: usize,
    bond_type_names: &[&str],
) -> Result<(BondList, ExclusionList), BondsFileError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| BondsFileError::Io(format!("{}: {}", path.display(), e)))?;
    parse_bonds_file(&raw, particle_count, bond_type_names)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    None,
    Bonds,
    Exclusions,
}

pub(crate) fn parse_bonds_file(
    raw: &str,
    particle_count: usize,
    bond_type_names: &[&str],
) -> Result<(BondList, ExclusionList), BondsFileError> {
    let max_index_for_check: i64 = particle_count as i64 - 1;

    let mut current: Section = Section::None;
    let mut bonds_seen = false;
    let mut exclusions_seen = false;
    let mut raw_bonds: Vec<(usize, u32, u32, u32)> = Vec::new(); // (line, i, j, type_idx)
    let mut raw_excl: Vec<(usize, u32, u32, f32)> = Vec::new();

    for (idx, line) in raw.lines().enumerate() {
        let line_number = idx + 1;
        let trimmed = strip_comment(line).trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(header) = trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            let header = header.trim();
            match header {
                "bonds" => {
                    if bonds_seen {
                        return Err(BondsFileError::DuplicateSection {
                            name: "bonds".to_string(),
                            line_number,
                        });
                    }
                    bonds_seen = true;
                    current = Section::Bonds;
                }
                "exclusions" => {
                    if exclusions_seen {
                        return Err(BondsFileError::DuplicateSection {
                            name: "exclusions".to_string(),
                            line_number,
                        });
                    }
                    exclusions_seen = true;
                    current = Section::Exclusions;
                }
                other => {
                    return Err(BondsFileError::UnknownSection {
                        name: other.to_string(),
                        line_number,
                    });
                }
            }
            continue;
        }

        match current {
            Section::None => {
                return Err(BondsFileError::ContentOutsideSection { line_number });
            }
            Section::Bonds => {
                let cols: Vec<&str> = trimmed.split_ascii_whitespace().collect();
                if cols.len() != 3 {
                    return Err(BondsFileError::InvalidBondRow {
                        line_number,
                        reason: format!("expected 3 columns, got {}", cols.len()),
                    });
                }
                let atom_i = cols[0].parse::<u32>().map_err(|_| {
                    BondsFileError::InvalidBondRow {
                        line_number,
                        reason: format!("atom_i {:?} is not a u32", cols[0]),
                    }
                })?;
                let atom_j = cols[1].parse::<u32>().map_err(|_| {
                    BondsFileError::InvalidBondRow {
                        line_number,
                        reason: format!("atom_j {:?} is not a u32", cols[1]),
                    }
                })?;
                if (atom_i as i64) > max_index_for_check {
                    return Err(BondsFileError::AtomIndexOutOfRange {
                        line_number,
                        index: atom_i,
                        max: max_index_for_check.max(0) as u32,
                    });
                }
                if (atom_j as i64) > max_index_for_check {
                    return Err(BondsFileError::AtomIndexOutOfRange {
                        line_number,
                        index: atom_j,
                        max: max_index_for_check.max(0) as u32,
                    });
                }
                if atom_i == atom_j {
                    return Err(BondsFileError::SelfBond {
                        line_number,
                        atom: atom_i,
                    });
                }
                let type_idx = bond_type_names
                    .iter()
                    .position(|n| *n == cols[2])
                    .ok_or_else(|| BondsFileError::UnknownBondType {
                        line_number,
                        name: cols[2].to_string(),
                    })? as u32;
                raw_bonds.push((line_number, atom_i, atom_j, type_idx));
            }
            Section::Exclusions => {
                let cols: Vec<&str> = trimmed.split_ascii_whitespace().collect();
                if cols.len() < 2 || cols.len() > 3 {
                    return Err(BondsFileError::InvalidExclusionRow {
                        line_number,
                        reason: format!("expected 2 or 3 columns, got {}", cols.len()),
                    });
                }
                let atom_i = cols[0].parse::<u32>().map_err(|_| {
                    BondsFileError::InvalidExclusionRow {
                        line_number,
                        reason: format!("atom_i {:?} is not a u32", cols[0]),
                    }
                })?;
                let atom_j = cols[1].parse::<u32>().map_err(|_| {
                    BondsFileError::InvalidExclusionRow {
                        line_number,
                        reason: format!("atom_j {:?} is not a u32", cols[1]),
                    }
                })?;
                if (atom_i as i64) > max_index_for_check {
                    return Err(BondsFileError::AtomIndexOutOfRange {
                        line_number,
                        index: atom_i,
                        max: max_index_for_check.max(0) as u32,
                    });
                }
                if (atom_j as i64) > max_index_for_check {
                    return Err(BondsFileError::AtomIndexOutOfRange {
                        line_number,
                        index: atom_j,
                        max: max_index_for_check.max(0) as u32,
                    });
                }
                if atom_i == atom_j {
                    return Err(BondsFileError::SelfExclusion {
                        line_number,
                        atom: atom_i,
                    });
                }
                let scale = if cols.len() == 3 {
                    cols[2].parse::<f32>().map_err(|_| {
                        BondsFileError::InvalidExclusionRow {
                            line_number,
                            reason: format!("scale {:?} is not an f32", cols[2]),
                        }
                    })?
                } else {
                    0.0
                };
                if !scale.is_finite() || !(0.0..=1.0).contains(&scale) {
                    return Err(BondsFileError::ScaleOutOfRange { line_number, scale });
                }
                raw_excl.push((line_number, atom_i, atom_j, scale));
            }
        }
    }

    // Canonicalise + sort bonds; reject duplicates after canonicalisation.
    let mut bonds: Vec<Bond> = raw_bonds
        .iter()
        .map(|&(_, i, j, t)| {
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            Bond {
                atom_i: a,
                atom_j: b,
                bond_type_index: t,
            }
        })
        .collect();
    bonds.sort_by_key(|b| (b.atom_i, b.atom_j));
    for w in bonds.windows(2) {
        if w[0].atom_i == w[1].atom_i && w[0].atom_j == w[1].atom_j {
            return Err(BondsFileError::DuplicateBond {
                atom_i: w[0].atom_i,
                atom_j: w[0].atom_j,
            });
        }
    }

    // Canonicalise + sort explicit exclusions; reject duplicates.
    let mut explicit: Vec<Exclusion> = raw_excl
        .iter()
        .map(|&(_, i, j, s)| {
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            Exclusion {
                atom_i: a,
                atom_j: b,
                scale: s,
            }
        })
        .collect();
    explicit.sort_by_key(|e| (e.atom_i, e.atom_j));
    for w in explicit.windows(2) {
        if w[0].atom_i == w[1].atom_i && w[0].atom_j == w[1].atom_j {
            return Err(BondsFileError::DuplicateExclusion {
                atom_i: w[0].atom_i,
                atom_j: w[0].atom_j,
            });
        }
    }

    // Build the effective exclusion list: explicit entries plus implicit
    // (0.0) entries for any bonded pair that does not already appear.
    let mut effective = explicit.clone();
    for b in &bonds {
        let already = effective
            .binary_search_by_key(&(b.atom_i, b.atom_j), |e| (e.atom_i, e.atom_j))
            .is_ok();
        if !already {
            effective.push(Exclusion {
                atom_i: b.atom_i,
                atom_j: b.atom_j,
                scale: 0.0,
            });
            effective.sort_by_key(|e| (e.atom_i, e.atom_j));
        }
    }

    // Build the atom-to-bond indexing.
    let mut atom_bond_offsets = vec![0u32; particle_count + 1];
    for b in &bonds {
        atom_bond_offsets[b.atom_i as usize + 1] += 1;
        atom_bond_offsets[b.atom_j as usize + 1] += 1;
    }
    for i in 1..=particle_count {
        atom_bond_offsets[i] += atom_bond_offsets[i - 1];
    }
    let mut atom_bond_indices = vec![0u32; bonds.len() * 2];
    let mut cursor: Vec<u32> = atom_bond_offsets[..particle_count].to_vec();
    for (k, b) in bonds.iter().enumerate() {
        let slot_i = (2 * k) as u32;
        let slot_j = (2 * k + 1) as u32;
        let pi = b.atom_i as usize;
        let pj = b.atom_j as usize;
        atom_bond_indices[cursor[pi] as usize] = slot_i;
        cursor[pi] += 1;
        atom_bond_indices[cursor[pj] as usize] = slot_j;
        cursor[pj] += 1;
    }

    // Build the atom-to-exclusion indexing.
    let mut atom_excl_offsets = vec![0u32; particle_count + 1];
    for e in &effective {
        atom_excl_offsets[e.atom_i as usize + 1] += 1;
        atom_excl_offsets[e.atom_j as usize + 1] += 1;
    }
    for i in 1..=particle_count {
        atom_excl_offsets[i] += atom_excl_offsets[i - 1];
    }
    let total_partner_entries = atom_excl_offsets[particle_count] as usize;
    let mut atom_excl_partners = vec![0u32; total_partner_entries];
    let mut atom_excl_scales = vec![0f32; total_partner_entries];
    let mut cursor_e: Vec<u32> = atom_excl_offsets[..particle_count].to_vec();
    for e in &effective {
        let pi = e.atom_i as usize;
        let pj = e.atom_j as usize;
        atom_excl_partners[cursor_e[pi] as usize] = e.atom_j;
        atom_excl_scales[cursor_e[pi] as usize] = e.scale;
        cursor_e[pi] += 1;
        atom_excl_partners[cursor_e[pj] as usize] = e.atom_i;
        atom_excl_scales[cursor_e[pj] as usize] = e.scale;
        cursor_e[pj] += 1;
    }

    let bond_list = BondList {
        bonds,
        atom_bond_offsets,
        atom_bond_indices,
        particle_count,
    };
    let exclusion_list = ExclusionList {
        entries: effective,
        atom_excl_offsets,
        atom_excl_partners,
        atom_excl_scales,
        particle_count,
    };
    Ok((bond_list, exclusion_list))
}

fn strip_comment(line: &str) -> &str {
    match line.find('#') {
        Some(i) => &line[..i],
        None => line,
    }
}
