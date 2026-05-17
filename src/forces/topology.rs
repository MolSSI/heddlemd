// rq-c2dbaa72 (topology module — defined in forces/topology.md)
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};

use crate::gpu::GpuError;
use crate::io::config::ConstraintTypeConfig;

#[derive(Debug, Clone, Copy)]
pub struct Bond {
    pub atom_i: u32,
    pub atom_j: u32,
    pub bond_type_index: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct Angle {
    pub atom_i: u32,
    pub atom_j: u32,
    pub atom_k: u32,
    pub angle_type_index: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct Exclusion {
    pub atom_i: u32,
    pub atom_j: u32,
    pub scale_lj: f32,
    pub scale_coul: f32,
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
pub struct AngleList {
    pub angles: Vec<Angle>,
    pub atom_angle_offsets: Vec<u32>,
    pub atom_angle_indices: Vec<u32>,
    pub particle_count: usize,
}

impl AngleList {
    pub fn empty(particle_count: usize) -> Self {
        AngleList {
            angles: Vec::new(),
            atom_angle_offsets: vec![0; particle_count + 1],
            atom_angle_indices: Vec::new(),
            particle_count,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.angles.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct ExclusionList {
    pub entries: Vec<Exclusion>,
    pub atom_excl_offsets: Vec<u32>,
    pub atom_excl_partners: Vec<u32>,
    pub atom_excl_lj_scales: Vec<f32>,
    pub atom_excl_coul_scales: Vec<f32>,
    pub particle_count: usize,
}

impl ExclusionList {
    pub fn empty(particle_count: usize) -> Self {
        ExclusionList {
            entries: Vec::new(),
            atom_excl_offsets: vec![0; particle_count + 1],
            atom_excl_partners: Vec::new(),
            atom_excl_lj_scales: Vec::new(),
            atom_excl_coul_scales: Vec::new(),
            particle_count,
        }
    }
}

// rq-3d5f2e98 — constraint slot framework data layout. See
// `integration/constraint-framework.md` for the SoA contract.

/// Algorithm-tag for a `[[constraint_types]]` entry. Drives the
/// constraint slot's per-group kernel dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintTypeKind {
    SettleWater,
}

/// One pairwise distance constraint inside a `ConstraintGroup`. The
/// local indices `(local_i, local_j)` refer to slots in the group's
/// own atom slice (`0..group.atom_count`) — not into the global
/// `ParticleBuffers`.
#[derive(Debug, Clone, Copy)]
pub struct GroupConstraint {
    pub local_i: u8,
    pub local_j: u8,
    pub r0: f32,
}

/// One connected component of the constraint graph: a set of atoms
/// rigidified by a set of pairwise distance constraints. Algorithms
/// (SETTLE in v1; M-SHAKE in a future feature) dispatch one thread per
/// group.
#[derive(Debug, Clone, Copy)]
pub struct ConstraintGroup {
    pub atom_offset: u32,
    pub atom_count: u32,
    pub constraint_offset: u32,
    pub constraint_count: u32,
    pub constraint_type_index: u32,
}

/// Host-side parsed-and-validated view of every constraint declared by
/// the topology file. See `integration/constraint-framework.md`.
#[derive(Debug, Clone)]
pub struct ConstraintList {
    pub groups: Vec<ConstraintGroup>,
    pub group_atoms: Vec<u32>,
    pub group_constraints: Vec<GroupConstraint>,
    pub constraint_type_kind: Vec<ConstraintTypeKind>,
    pub particle_count: usize,
}

impl ConstraintList {
    pub fn empty(particle_count: usize) -> Self {
        ConstraintList {
            groups: Vec::new(),
            group_atoms: Vec::new(),
            group_constraints: Vec::new(),
            constraint_type_kind: Vec::new(),
            particle_count,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

/// Host-side handle around the exclusion list's four device buffers.
/// Shared between the LJ and Coulomb pair-force slots; each consumes
/// the scale array appropriate to itself.
#[derive(Debug)]
pub struct DeviceExclusionList {
    pub atom_excl_offsets: CudaSlice<u32>,
    pub atom_excl_partners: CudaSlice<u32>,
    pub atom_excl_lj_scales: CudaSlice<f32>,
    pub atom_excl_coul_scales: CudaSlice<f32>,
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
        let atom_excl_lj_scales = if list.atom_excl_lj_scales.is_empty() {
            device.alloc_zeros::<f32>(0).map_err(GpuError::from)?
        } else {
            device
                .htod_sync_copy(&list.atom_excl_lj_scales)
                .map_err(GpuError::from)?
        };
        let atom_excl_coul_scales = if list.atom_excl_coul_scales.is_empty() {
            device.alloc_zeros::<f32>(0).map_err(GpuError::from)?
        } else {
            device
                .htod_sync_copy(&list.atom_excl_coul_scales)
                .map_err(GpuError::from)?
        };
        Ok(DeviceExclusionList {
            atom_excl_offsets,
            atom_excl_partners,
            atom_excl_lj_scales,
            atom_excl_coul_scales,
            particle_count: list.particle_count,
        })
    }
}

// rq-bca0adbc — errors for topology file parsing.
#[derive(Debug, thiserror::Error)]
pub enum TopologyFileError {
    #[error("failed to read topology file: {0}")]
    Io(String),
    #[error("line {line_number}: unknown section `{name}`")]
    UnknownSection { name: String, line_number: usize },
    #[error("line {line_number}: duplicate section `{name}`")]
    DuplicateSection { name: String, line_number: usize },
    #[error("line {line_number}: content appears outside any section")]
    ContentOutsideSection { line_number: usize },
    #[error("line {line_number}: invalid bond row: {reason}")]
    InvalidBondRow { line_number: usize, reason: String },
    #[error("line {line_number}: invalid angle row: {reason}")]
    InvalidAngleRow { line_number: usize, reason: String },
    #[error("line {line_number}: invalid exclusion row: {reason}")]
    InvalidExclusionRow { line_number: usize, reason: String },
    #[error("line {line_number}: atom index {index} is out of range (max {max})")]
    AtomIndexOutOfRange {
        line_number: usize,
        index: u32,
        max: u32,
    },
    #[error("line {line_number}: atom {atom} is bonded to itself")]
    SelfBond { line_number: usize, atom: u32 },
    #[error("line {line_number}: atom {atom} appears more than once in this angle")]
    RepeatedAtomInAngle { line_number: usize, atom: u32 },
    #[error("line {line_number}: atom {atom} is excluded from itself")]
    SelfExclusion { line_number: usize, atom: u32 },
    #[error("duplicate bond between atoms {atom_i} and {atom_j}")]
    DuplicateBond { atom_i: u32, atom_j: u32 },
    #[error("duplicate angle between atoms ({atom_i}, {atom_j}, {atom_k})")]
    DuplicateAngle {
        atom_i: u32,
        atom_j: u32,
        atom_k: u32,
    },
    #[error("duplicate exclusion between atoms {atom_i} and {atom_j}")]
    DuplicateExclusion { atom_i: u32, atom_j: u32 },
    #[error("line {line_number}: unknown bond type `{name}`")]
    UnknownBondType { line_number: usize, name: String },
    #[error("line {line_number}: unknown angle type `{name}`")]
    UnknownAngleType { line_number: usize, name: String },
    #[error("line {line_number}: exclusion scale {scale} is out of the range [0, 1]")]
    ScaleOutOfRange { line_number: usize, scale: f32 },
    #[error("line {line_number}: invalid constraint row: {reason}")]
    InvalidConstraintRow { line_number: usize, reason: String },
    #[error("line {line_number}: atom {atom} appears more than once in this constraint row")]
    SelfConstraint { line_number: usize, atom: u32 },
    #[error("line {line_number}: unknown constraint type `{name}`")]
    UnknownConstraintType { line_number: usize, name: String },
    #[error("atom {atom} appears in more than one [constraints] row")]
    DuplicateConstraintAtom { atom: u32 },
    #[error("pair (atoms {atom_i}, {atom_j}) appears in both [bonds] and [constraints]")]
    BondIsAlsoConstraint { atom_i: u32, atom_j: u32 },
}

// rq-12b7dcb6
pub fn load_topology_file(
    path: &Path,
    particle_count: usize,
    bond_type_names: &[&str],
    angle_type_names: &[&str],
    constraint_types: &[ConstraintTypeConfig],
) -> Result<(BondList, AngleList, ExclusionList, ConstraintList), TopologyFileError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| TopologyFileError::Io(format!("{}: {}", path.display(), e)))?;
    parse_topology_file(
        &raw,
        particle_count,
        bond_type_names,
        angle_type_names,
        constraint_types,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    None,
    Bonds,
    Exclusions,
    Angles,
    Constraints,
}

pub(crate) fn parse_topology_file(
    raw: &str,
    particle_count: usize,
    bond_type_names: &[&str],
    angle_type_names: &[&str],
    constraint_types: &[ConstraintTypeConfig],
) -> Result<(BondList, AngleList, ExclusionList, ConstraintList), TopologyFileError> {
    let max_index_for_check: i64 = particle_count as i64 - 1;

    let mut current: Section = Section::None;
    let mut bonds_seen = false;
    let mut exclusions_seen = false;
    let mut angles_seen = false;
    let mut constraints_seen = false;
    let mut raw_bonds: Vec<(usize, u32, u32, u32)> = Vec::new();
    let mut raw_excl: Vec<(usize, u32, u32, f32, f32)> = Vec::new();
    let mut raw_angles: Vec<(usize, u32, u32, u32, u32)> = Vec::new();
    // (line_number, atom_indices_in_declared_order, constraint_type_index)
    let mut raw_constraint_rows: Vec<(usize, Vec<u32>, u32)> = Vec::new();

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
                        return Err(TopologyFileError::DuplicateSection {
                            name: "bonds".to_string(),
                            line_number,
                        });
                    }
                    bonds_seen = true;
                    current = Section::Bonds;
                }
                "exclusions" => {
                    if exclusions_seen {
                        return Err(TopologyFileError::DuplicateSection {
                            name: "exclusions".to_string(),
                            line_number,
                        });
                    }
                    exclusions_seen = true;
                    current = Section::Exclusions;
                }
                "angles" => {
                    if angles_seen {
                        return Err(TopologyFileError::DuplicateSection {
                            name: "angles".to_string(),
                            line_number,
                        });
                    }
                    angles_seen = true;
                    current = Section::Angles;
                }
                "constraints" => {
                    if constraints_seen {
                        return Err(TopologyFileError::DuplicateSection {
                            name: "constraints".to_string(),
                            line_number,
                        });
                    }
                    constraints_seen = true;
                    current = Section::Constraints;
                }
                other => {
                    return Err(TopologyFileError::UnknownSection {
                        name: other.to_string(),
                        line_number,
                    });
                }
            }
            continue;
        }

        match current {
            Section::None => {
                return Err(TopologyFileError::ContentOutsideSection { line_number });
            }
            Section::Bonds => {
                let cols: Vec<&str> = trimmed.split_ascii_whitespace().collect();
                if cols.len() != 3 {
                    return Err(TopologyFileError::InvalidBondRow {
                        line_number,
                        reason: format!("expected 3 columns, got {}", cols.len()),
                    });
                }
                let atom_i = cols[0].parse::<u32>().map_err(|_| {
                    TopologyFileError::InvalidBondRow {
                        line_number,
                        reason: format!("atom_i {:?} is not a u32", cols[0]),
                    }
                })?;
                let atom_j = cols[1].parse::<u32>().map_err(|_| {
                    TopologyFileError::InvalidBondRow {
                        line_number,
                        reason: format!("atom_j {:?} is not a u32", cols[1]),
                    }
                })?;
                if (atom_i as i64) > max_index_for_check {
                    return Err(TopologyFileError::AtomIndexOutOfRange {
                        line_number,
                        index: atom_i,
                        max: max_index_for_check.max(0) as u32,
                    });
                }
                if (atom_j as i64) > max_index_for_check {
                    return Err(TopologyFileError::AtomIndexOutOfRange {
                        line_number,
                        index: atom_j,
                        max: max_index_for_check.max(0) as u32,
                    });
                }
                if atom_i == atom_j {
                    return Err(TopologyFileError::SelfBond {
                        line_number,
                        atom: atom_i,
                    });
                }
                let type_idx = bond_type_names
                    .iter()
                    .position(|n| *n == cols[2])
                    .ok_or_else(|| TopologyFileError::UnknownBondType {
                        line_number,
                        name: cols[2].to_string(),
                    })? as u32;
                raw_bonds.push((line_number, atom_i, atom_j, type_idx));
            }
            Section::Angles => {
                let cols: Vec<&str> = trimmed.split_ascii_whitespace().collect();
                if cols.len() != 4 {
                    return Err(TopologyFileError::InvalidAngleRow {
                        line_number,
                        reason: format!("expected 4 columns, got {}", cols.len()),
                    });
                }
                let atom_i = cols[0].parse::<u32>().map_err(|_| {
                    TopologyFileError::InvalidAngleRow {
                        line_number,
                        reason: format!("atom_i {:?} is not a u32", cols[0]),
                    }
                })?;
                let atom_j = cols[1].parse::<u32>().map_err(|_| {
                    TopologyFileError::InvalidAngleRow {
                        line_number,
                        reason: format!("atom_j {:?} is not a u32", cols[1]),
                    }
                })?;
                let atom_k = cols[2].parse::<u32>().map_err(|_| {
                    TopologyFileError::InvalidAngleRow {
                        line_number,
                        reason: format!("atom_k {:?} is not a u32", cols[2]),
                    }
                })?;
                for &a in &[atom_i, atom_j, atom_k] {
                    if (a as i64) > max_index_for_check {
                        return Err(TopologyFileError::AtomIndexOutOfRange {
                            line_number,
                            index: a,
                            max: max_index_for_check.max(0) as u32,
                        });
                    }
                }
                if atom_i == atom_j {
                    return Err(TopologyFileError::RepeatedAtomInAngle {
                        line_number,
                        atom: atom_i,
                    });
                }
                if atom_j == atom_k {
                    return Err(TopologyFileError::RepeatedAtomInAngle {
                        line_number,
                        atom: atom_j,
                    });
                }
                if atom_i == atom_k {
                    return Err(TopologyFileError::RepeatedAtomInAngle {
                        line_number,
                        atom: atom_i,
                    });
                }
                let type_idx = angle_type_names
                    .iter()
                    .position(|n| *n == cols[3])
                    .ok_or_else(|| TopologyFileError::UnknownAngleType {
                        line_number,
                        name: cols[3].to_string(),
                    })? as u32;
                raw_angles.push((line_number, atom_i, atom_j, atom_k, type_idx));
            }
            Section::Exclusions => {
                let cols: Vec<&str> = trimmed.split_ascii_whitespace().collect();
                if !(2..=4).contains(&cols.len()) {
                    return Err(TopologyFileError::InvalidExclusionRow {
                        line_number,
                        reason: format!("expected 2, 3, or 4 columns, got {}", cols.len()),
                    });
                }
                let atom_i = cols[0].parse::<u32>().map_err(|_| {
                    TopologyFileError::InvalidExclusionRow {
                        line_number,
                        reason: format!("atom_i {:?} is not a u32", cols[0]),
                    }
                })?;
                let atom_j = cols[1].parse::<u32>().map_err(|_| {
                    TopologyFileError::InvalidExclusionRow {
                        line_number,
                        reason: format!("atom_j {:?} is not a u32", cols[1]),
                    }
                })?;
                if (atom_i as i64) > max_index_for_check {
                    return Err(TopologyFileError::AtomIndexOutOfRange {
                        line_number,
                        index: atom_i,
                        max: max_index_for_check.max(0) as u32,
                    });
                }
                if (atom_j as i64) > max_index_for_check {
                    return Err(TopologyFileError::AtomIndexOutOfRange {
                        line_number,
                        index: atom_j,
                        max: max_index_for_check.max(0) as u32,
                    });
                }
                if atom_i == atom_j {
                    return Err(TopologyFileError::SelfExclusion {
                        line_number,
                        atom: atom_i,
                    });
                }
                let (scale_lj, scale_coul) = match cols.len() {
                    2 => (0.0_f32, 0.0_f32),
                    3 => {
                        let s = parse_scale(line_number, "scale", cols[2])?;
                        (s, s)
                    }
                    4 => {
                        let lj = parse_scale(line_number, "scale_lj", cols[2])?;
                        let coul = parse_scale(line_number, "scale_coul", cols[3])?;
                        (lj, coul)
                    }
                    _ => unreachable!(),
                };
                raw_excl.push((line_number, atom_i, atom_j, scale_lj, scale_coul));
            }
            Section::Constraints => {
                let cols: Vec<&str> = trimmed.split_ascii_whitespace().collect();
                if cols.len() < 2 {
                    return Err(TopologyFileError::InvalidConstraintRow {
                        line_number,
                        reason: format!(
                            "expected at least one atom index and one constraint_type_name, got {} columns",
                            cols.len()
                        ),
                    });
                }
                let type_name = cols[cols.len() - 1];
                let type_idx = constraint_types
                    .iter()
                    .position(|t| t.name() == type_name)
                    .ok_or_else(|| TopologyFileError::UnknownConstraintType {
                        line_number,
                        name: type_name.to_string(),
                    })? as u32;
                let expected_atoms =
                    constraint_types[type_idx as usize].expected_atom_count();
                let atom_cols = &cols[..cols.len() - 1];
                if atom_cols.len() != expected_atoms {
                    return Err(TopologyFileError::InvalidConstraintRow {
                        line_number,
                        reason: format!(
                            "constraint type `{type_name}` requires {expected_atoms} atoms, got {}",
                            atom_cols.len()
                        ),
                    });
                }
                let mut atoms: Vec<u32> = Vec::with_capacity(expected_atoms);
                for (idx, col) in atom_cols.iter().enumerate() {
                    let a = col.parse::<u32>().map_err(|_| {
                        TopologyFileError::InvalidConstraintRow {
                            line_number,
                            reason: format!("atom[{idx}] {:?} is not a u32", col),
                        }
                    })?;
                    if (a as i64) > max_index_for_check {
                        return Err(TopologyFileError::AtomIndexOutOfRange {
                            line_number,
                            index: a,
                            max: max_index_for_check.max(0) as u32,
                        });
                    }
                    atoms.push(a);
                }
                // Reject duplicate atoms within a single row.
                for i in 0..atoms.len() {
                    for j in (i + 1)..atoms.len() {
                        if atoms[i] == atoms[j] {
                            return Err(TopologyFileError::SelfConstraint {
                                line_number,
                                atom: atoms[i],
                            });
                        }
                    }
                }
                raw_constraint_rows.push((line_number, atoms, type_idx));
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
            return Err(TopologyFileError::DuplicateBond {
                atom_i: w[0].atom_i,
                atom_j: w[0].atom_j,
            });
        }
    }

    // Canonicalise + sort angles. Wings swap so atom_i < atom_k;
    // sorting is by (atom_j, atom_i, atom_k); duplicates after
    // canonicalisation are rejected.
    let mut angles: Vec<Angle> = raw_angles
        .iter()
        .map(|&(_, i, j, k, t)| {
            let (a, c) = if i < k { (i, k) } else { (k, i) };
            Angle {
                atom_i: a,
                atom_j: j,
                atom_k: c,
                angle_type_index: t,
            }
        })
        .collect();
    angles.sort_by_key(|a| (a.atom_j, a.atom_i, a.atom_k));
    for w in angles.windows(2) {
        if w[0].atom_i == w[1].atom_i
            && w[0].atom_j == w[1].atom_j
            && w[0].atom_k == w[1].atom_k
        {
            return Err(TopologyFileError::DuplicateAngle {
                atom_i: w[0].atom_i,
                atom_j: w[0].atom_j,
                atom_k: w[0].atom_k,
            });
        }
    }

    // Canonicalise + sort explicit exclusions; reject duplicates.
    let mut explicit: Vec<Exclusion> = raw_excl
        .iter()
        .map(|&(_, i, j, sl, sc)| {
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            Exclusion {
                atom_i: a,
                atom_j: b,
                scale_lj: sl,
                scale_coul: sc,
            }
        })
        .collect();
    explicit.sort_by_key(|e| (e.atom_i, e.atom_j));
    for w in explicit.windows(2) {
        if w[0].atom_i == w[1].atom_i && w[0].atom_j == w[1].atom_j {
            return Err(TopologyFileError::DuplicateExclusion {
                atom_i: w[0].atom_i,
                atom_j: w[0].atom_j,
            });
        }
    }

    // Build the per-group ConstraintList. Each [constraints] row is
    // its own group in v1; verify no atom appears in more than one row.
    let mut constraint_groups: Vec<ConstraintGroup> = Vec::with_capacity(raw_constraint_rows.len());
    let mut group_atoms: Vec<u32> = Vec::new();
    let mut group_constraints: Vec<GroupConstraint> = Vec::new();
    let mut atom_to_row: std::collections::HashMap<u32, usize> =
        std::collections::HashMap::new();
    for (row_idx, (_line, atoms, type_idx)) in raw_constraint_rows.iter().enumerate() {
        for &a in atoms {
            if atom_to_row.insert(a, row_idx).is_some() {
                return Err(TopologyFileError::DuplicateConstraintAtom { atom: a });
            }
        }
        let atom_offset = group_atoms.len() as u32;
        let atom_count = atoms.len() as u32;
        let constraint_offset = group_constraints.len() as u32;
        // Expand every group into its pairwise constraints. The
        // per-algorithm shape check (SETTLE wants exactly 3 atoms and
        // the (0,1)/(0,2)/(1,2) pattern) is enforced by the slot's
        // builder; the topology parser produces the complete pairwise
        // expansion so the data layout already matches the framework
        // contract.
        for &a in atoms {
            group_atoms.push(a);
        }
        for i in 0..atoms.len() {
            for j in (i + 1)..atoms.len() {
                group_constraints.push(GroupConstraint {
                    local_i: i as u8,
                    local_j: j as u8,
                    r0: 0.0,
                });
            }
        }
        let constraint_count = group_constraints.len() as u32 - constraint_offset;
        constraint_groups.push(ConstraintGroup {
            atom_offset,
            atom_count,
            constraint_offset,
            constraint_count,
            constraint_type_index: *type_idx,
        });
    }
    // Sort groups by minimum particle index for reproducibility.
    let mut order: Vec<usize> = (0..constraint_groups.len()).collect();
    order.sort_by_key(|&i| {
        let g = constraint_groups[i];
        let slice = &group_atoms[g.atom_offset as usize
            ..(g.atom_offset + g.atom_count) as usize];
        slice.iter().copied().min().unwrap_or(u32::MAX)
    });
    let mut sorted_groups: Vec<ConstraintGroup> = Vec::with_capacity(constraint_groups.len());
    let mut sorted_atoms: Vec<u32> = Vec::with_capacity(group_atoms.len());
    let mut sorted_constraints: Vec<GroupConstraint> =
        Vec::with_capacity(group_constraints.len());
    for &orig_idx in &order {
        let g = constraint_groups[orig_idx];
        let new_atom_offset = sorted_atoms.len() as u32;
        let new_constraint_offset = sorted_constraints.len() as u32;
        sorted_atoms.extend_from_slice(
            &group_atoms[g.atom_offset as usize
                ..(g.atom_offset + g.atom_count) as usize],
        );
        sorted_constraints.extend_from_slice(
            &group_constraints[g.constraint_offset as usize
                ..(g.constraint_offset + g.constraint_count) as usize],
        );
        sorted_groups.push(ConstraintGroup {
            atom_offset: new_atom_offset,
            atom_count: g.atom_count,
            constraint_offset: new_constraint_offset,
            constraint_count: g.constraint_count,
            constraint_type_index: g.constraint_type_index,
        });
    }
    let constraint_groups = sorted_groups;
    let group_atoms = sorted_atoms;
    let group_constraints = sorted_constraints;

    // Reject any (atom_i, atom_j) pair that appears in both [bonds]
    // and (after expansion) [constraints].
    {
        let bond_set: std::collections::HashSet<(u32, u32)> =
            bonds.iter().map(|b| (b.atom_i, b.atom_j)).collect();
        for g in &constraint_groups {
            let atom_slice = &group_atoms[g.atom_offset as usize
                ..(g.atom_offset + g.atom_count) as usize];
            let cstr_slice = &group_constraints[g.constraint_offset as usize
                ..(g.constraint_offset + g.constraint_count) as usize];
            for c in cstr_slice {
                let a = atom_slice[c.local_i as usize];
                let b = atom_slice[c.local_j as usize];
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                if bond_set.contains(&(lo, hi)) {
                    return Err(TopologyFileError::BondIsAlsoConstraint {
                        atom_i: lo,
                        atom_j: hi,
                    });
                }
            }
        }
    }

    // Algorithm-tag table: one entry per [[constraint_types]] entry.
    let constraint_type_kind: Vec<ConstraintTypeKind> = constraint_types
        .iter()
        .map(|t| match t {
            ConstraintTypeConfig::SettleWater { .. } => ConstraintTypeKind::SettleWater,
        })
        .collect();

    // Build the effective exclusion list:
    //   1. Every explicit entry kept as-is.
    //   2. For every bond (i, j) lacking an explicit (i, j) entry,
    //      add implicit (i, j, 0.0, 0.0).
    //   3. For every angle (i, j, k), consider the 1-3 pair (i, k).
    //      If neither an explicit entry nor an already-added implicit
    //      bond entry covers (i, k), add implicit (i, k, 0.0, 0.0).
    //   4. For every constraint group, add implicit (i, j, 0.0, 0.0)
    //      for every distinct intra-group pair (1-2 and 1-3) not
    //      already covered by an explicit or earlier-implicit entry.
    let mut effective = explicit.clone();
    for b in &bonds {
        let already = effective
            .binary_search_by_key(&(b.atom_i, b.atom_j), |e| (e.atom_i, e.atom_j))
            .is_ok();
        if !already {
            effective.push(Exclusion {
                atom_i: b.atom_i,
                atom_j: b.atom_j,
                scale_lj: 0.0,
                scale_coul: 0.0,
            });
            effective.sort_by_key(|e| (e.atom_i, e.atom_j));
        }
    }
    for a in &angles {
        // 1-3 pair: angle's two wings, already sorted so atom_i < atom_k.
        let (lo, hi) = (a.atom_i, a.atom_k);
        let already = effective
            .binary_search_by_key(&(lo, hi), |e| (e.atom_i, e.atom_j))
            .is_ok();
        if !already {
            effective.push(Exclusion {
                atom_i: lo,
                atom_j: hi,
                scale_lj: 0.0,
                scale_coul: 0.0,
            });
            effective.sort_by_key(|e| (e.atom_i, e.atom_j));
        }
    }
    for g in &constraint_groups {
        let atom_slice = &group_atoms[g.atom_offset as usize
            ..(g.atom_offset + g.atom_count) as usize];
        for i in 0..atom_slice.len() {
            for j in (i + 1)..atom_slice.len() {
                let a = atom_slice[i];
                let b = atom_slice[j];
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                let already = effective
                    .binary_search_by_key(&(lo, hi), |e| (e.atom_i, e.atom_j))
                    .is_ok();
                if !already {
                    effective.push(Exclusion {
                        atom_i: lo,
                        atom_j: hi,
                        scale_lj: 0.0,
                        scale_coul: 0.0,
                    });
                    effective.sort_by_key(|e| (e.atom_i, e.atom_j));
                }
            }
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

    // Build the atom-to-angle indexing. Each angle contributes three
    // slots (3·k for atom_i, 3·k+1 for atom_j, 3·k+2 for atom_k).
    // Entries within each atom's slice are sorted by underlying angle
    // index since we iterate the sorted `angles` vec in order.
    let mut atom_angle_offsets = vec![0u32; particle_count + 1];
    for a in &angles {
        atom_angle_offsets[a.atom_i as usize + 1] += 1;
        atom_angle_offsets[a.atom_j as usize + 1] += 1;
        atom_angle_offsets[a.atom_k as usize + 1] += 1;
    }
    for i in 1..=particle_count {
        atom_angle_offsets[i] += atom_angle_offsets[i - 1];
    }
    let mut atom_angle_indices = vec![0u32; angles.len() * 3];
    let mut cursor_a: Vec<u32> = atom_angle_offsets[..particle_count].to_vec();
    for (k, a) in angles.iter().enumerate() {
        let slot_i = (3 * k) as u32;
        let slot_j = (3 * k + 1) as u32;
        let slot_k = (3 * k + 2) as u32;
        let pi = a.atom_i as usize;
        let pj = a.atom_j as usize;
        let pk = a.atom_k as usize;
        atom_angle_indices[cursor_a[pi] as usize] = slot_i;
        cursor_a[pi] += 1;
        atom_angle_indices[cursor_a[pj] as usize] = slot_j;
        cursor_a[pj] += 1;
        atom_angle_indices[cursor_a[pk] as usize] = slot_k;
        cursor_a[pk] += 1;
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
    let mut atom_excl_lj_scales = vec![0f32; total_partner_entries];
    let mut atom_excl_coul_scales = vec![0f32; total_partner_entries];
    let mut cursor_e: Vec<u32> = atom_excl_offsets[..particle_count].to_vec();
    for e in &effective {
        let pi = e.atom_i as usize;
        let pj = e.atom_j as usize;
        atom_excl_partners[cursor_e[pi] as usize] = e.atom_j;
        atom_excl_lj_scales[cursor_e[pi] as usize] = e.scale_lj;
        atom_excl_coul_scales[cursor_e[pi] as usize] = e.scale_coul;
        cursor_e[pi] += 1;
        atom_excl_partners[cursor_e[pj] as usize] = e.atom_i;
        atom_excl_lj_scales[cursor_e[pj] as usize] = e.scale_lj;
        atom_excl_coul_scales[cursor_e[pj] as usize] = e.scale_coul;
        cursor_e[pj] += 1;
    }

    let bond_list = BondList {
        bonds,
        atom_bond_offsets,
        atom_bond_indices,
        particle_count,
    };
    let angle_list = AngleList {
        angles,
        atom_angle_offsets,
        atom_angle_indices,
        particle_count,
    };
    let exclusion_list = ExclusionList {
        entries: effective,
        atom_excl_offsets,
        atom_excl_partners,
        atom_excl_lj_scales,
        atom_excl_coul_scales,
        particle_count,
    };
    let constraint_list = ConstraintList {
        groups: constraint_groups,
        group_atoms,
        group_constraints,
        constraint_type_kind,
        particle_count,
    };
    Ok((bond_list, angle_list, exclusion_list, constraint_list))
}

fn parse_scale(
    line_number: usize,
    column: &'static str,
    raw: &str,
) -> Result<f32, TopologyFileError> {
    let scale = raw.parse::<f32>().map_err(|_| {
        TopologyFileError::InvalidExclusionRow {
            line_number,
            reason: format!("{column} {:?} is not an f32", raw),
        }
    })?;
    if !scale.is_finite() || !(0.0..=1.0).contains(&scale) {
        return Err(TopologyFileError::ScaleOutOfRange { line_number, scale });
    }
    Ok(scale)
}

fn strip_comment(line: &str) -> &str {
    match line.find('#') {
        Some(i) => &line[..i],
        None => line,
    }
}
