// rq-ff8e283d rq-321fe0d0 rq-a953fc1d
use std::path::Path;

use crate::pbc::SimulationBox;

// rq-8df7fb0c
#[derive(Debug, Clone)]
pub struct InitState {
    pub sim_box: SimulationBox,
    pub particle_count: usize,
    pub type_indices: Vec<u32>,
    pub positions_x: Vec<f32>,
    pub positions_y: Vec<f32>,
    pub positions_z: Vec<f32>,
    pub velocities: Option<InitVelocities>,
}

// rq-abd761d4
#[derive(Debug, Clone)]
pub struct InitVelocities {
    pub velocities_x: Vec<f32>,
    pub velocities_y: Vec<f32>,
    pub velocities_z: Vec<f32>,
}

// rq-573b650b
#[derive(Debug)]
pub enum InitStateError {
    Io(String),
    Empty,
    InvalidParticleCount {
        line_number: usize,
        raw: String,
    },
    MissingCommentLine,
    MissingAttribute {
        name: &'static str,
    },
    InvalidLattice(String),
    InvalidProperties(String),
    RowCountMismatch {
        expected: usize,
        actual: usize,
    },
    RowColumnCountMismatch {
        line_number: usize,
        expected: usize,
        actual: usize,
    },
    UnknownType {
        line_number: usize,
        name: String,
    },
    InvalidNumber {
        line_number: usize,
        column: &'static str,
        raw: String,
    },
    NonFiniteValue {
        line_number: usize,
        column: &'static str,
    },
    PositionOutsideBox {
        line_number: usize,
        axis: &'static str,
        value: f64,
        half_length: f64,
    },
    TrailingContent {
        line_number: usize,
    },
}

impl std::fmt::Display for InitStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitStateError::Io(s) => write!(f, "Io({s})"),
            InitStateError::Empty => write!(f, "Empty"),
            InitStateError::InvalidParticleCount { line_number, raw } => write!(
                f,
                "InvalidParticleCount {{ line_number: {line_number}, raw: {raw:?} }}"
            ),
            InitStateError::MissingCommentLine => write!(f, "MissingCommentLine"),
            InitStateError::MissingAttribute { name } => {
                write!(f, "MissingAttribute {{ name: {name:?} }}")
            }
            InitStateError::InvalidLattice(s) => write!(f, "InvalidLattice({s})"),
            InitStateError::InvalidProperties(s) => write!(f, "InvalidProperties({s})"),
            InitStateError::RowCountMismatch { expected, actual } => write!(
                f,
                "RowCountMismatch {{ expected: {expected}, actual: {actual} }}"
            ),
            InitStateError::RowColumnCountMismatch {
                line_number,
                expected,
                actual,
            } => write!(
                f,
                "RowColumnCountMismatch {{ line_number: {line_number}, expected: {expected}, actual: {actual} }}"
            ),
            InitStateError::UnknownType { line_number, name } => write!(
                f,
                "UnknownType {{ line_number: {line_number}, name: {name:?} }}"
            ),
            InitStateError::InvalidNumber {
                line_number,
                column,
                raw,
            } => write!(
                f,
                "InvalidNumber {{ line_number: {line_number}, column: {column:?}, raw: {raw:?} }}"
            ),
            InitStateError::NonFiniteValue {
                line_number,
                column,
            } => write!(
                f,
                "NonFiniteValue {{ line_number: {line_number}, column: {column:?} }}"
            ),
            InitStateError::PositionOutsideBox {
                line_number,
                axis,
                value,
                half_length,
            } => write!(
                f,
                "PositionOutsideBox {{ line_number: {line_number}, axis: {axis:?}, value: {value}, half_length: {half_length} }}"
            ),
            InitStateError::TrailingContent { line_number } => {
                write!(f, "TrailingContent {{ line_number: {line_number} }}")
            }
        }
    }
}

impl std::error::Error for InitStateError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PropertiesShape {
    SpeciesPos,
    SpeciesPosVelo,
}

impl PropertiesShape {
    fn columns(self) -> usize {
        match self {
            PropertiesShape::SpeciesPos => 4,
            PropertiesShape::SpeciesPosVelo => 7,
        }
    }

    fn has_velocities(self) -> bool {
        matches!(self, PropertiesShape::SpeciesPosVelo)
    }
}

// rq-20e7cab6
fn parse_attribute_line(line: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let key_start = i;
        while i < bytes.len() && bytes[i] != b'=' && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'=' {
            // No '=', skip the rest of this token.
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            continue;
        }
        let key = std::str::from_utf8(&bytes[key_start..i]).unwrap().to_string();
        i += 1; // skip '='
        if i >= bytes.len() {
            out.push((key, String::new()));
            break;
        }
        let value = if bytes[i] == b'"' {
            i += 1;
            let v_start = i;
            while i < bytes.len() && bytes[i] != b'"' {
                i += 1;
            }
            let v = std::str::from_utf8(&bytes[v_start..i]).unwrap().to_string();
            if i < bytes.len() {
                i += 1;
            }
            v
        } else {
            let v_start = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            std::str::from_utf8(&bytes[v_start..i]).unwrap().to_string()
        };
        out.push((key, value));
    }
    out
}

fn parse_lattice(raw: &str) -> Result<(SimulationBox, [f64; 3]), InitStateError> {
    let parts: Vec<&str> = raw.split_ascii_whitespace().collect();
    if parts.len() != 9 {
        return Err(InitStateError::InvalidLattice(format!(
            "expected 9 components, got {}",
            parts.len()
        )));
    }
    let mut vals = [0.0_f64; 9];
    for (i, p) in parts.iter().enumerate() {
        vals[i] = p.parse::<f64>().map_err(|_| {
            InitStateError::InvalidLattice(format!("component {i} is not a number: {p:?}"))
        })?;
    }
    for (i, v) in vals.iter().enumerate() {
        if !v.is_finite() {
            return Err(InitStateError::InvalidLattice(format!(
                "component {i} is not finite: {v}"
            )));
        }
    }
    let off_diag_indices = [1, 2, 3, 5, 6, 7];
    for &i in &off_diag_indices {
        if vals[i] != 0.0 {
            return Err(InitStateError::InvalidLattice(format!(
                "off-diagonal component {i} must be 0.0, got {}",
                vals[i]
            )));
        }
    }
    let lx = vals[0];
    let ly = vals[4];
    let lz = vals[8];
    if lx <= 0.0 || ly <= 0.0 || lz <= 0.0 {
        return Err(InitStateError::InvalidLattice(format!(
            "diagonal must be strictly positive; got ({lx}, {ly}, {lz})"
        )));
    }
    let sim_box = SimulationBox::new_orthorhombic(lx as f32, ly as f32, lz as f32)
        .map_err(|e| InitStateError::InvalidLattice(format!("{e}")))?;
    Ok((sim_box, [lx, ly, lz]))
}

fn parse_properties(raw: &str) -> Result<PropertiesShape, InitStateError> {
    match raw {
        "species:S:1:pos:R:3" => Ok(PropertiesShape::SpeciesPos),
        "species:S:1:pos:R:3:velo:R:3" => Ok(PropertiesShape::SpeciesPosVelo),
        _ => Err(InitStateError::InvalidProperties(format!(
            "unsupported Properties value {raw:?}"
        ))),
    }
}

// rq-5711e6b2 rq-dad38fdd
pub fn load_init_state(
    path: &Path,
    type_names: &[&str],
) -> Result<InitState, InitStateError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| InitStateError::Io(format!("{}: {}", path.display(), e)))?;
    parse_init_state(&raw, type_names)
}

pub(crate) fn parse_init_state(
    raw: &str,
    type_names: &[&str],
) -> Result<InitState, InitStateError> {
    let mut lines = raw.lines();
    let first = lines.next().ok_or(InitStateError::Empty)?;
    let count_str = first.trim();
    let particle_count: usize = match count_str.parse::<i64>() {
        Ok(n) if n >= 0 => n as usize,
        _ => {
            return Err(InitStateError::InvalidParticleCount {
                line_number: 1,
                raw: count_str.to_string(),
            });
        }
    };

    let comment = lines.next().ok_or(InitStateError::MissingCommentLine)?;
    let attrs = parse_attribute_line(comment);

    let lattice_value = attrs
        .iter()
        .find(|(k, _)| k == "Lattice")
        .map(|(_, v)| v.as_str())
        .ok_or(InitStateError::MissingAttribute { name: "Lattice" })?;
    let (sim_box, lengths_f64) = parse_lattice(lattice_value)?;

    let properties_value = attrs
        .iter()
        .find(|(k, _)| k == "Properties")
        .map(|(_, v)| v.as_str())
        .ok_or(InitStateError::MissingAttribute { name: "Properties" })?;
    let shape = parse_properties(properties_value)?;
    let expected_cols = shape.columns();
    let has_velo = shape.has_velocities();

    let half = [lengths_f64[0] / 2.0, lengths_f64[1] / 2.0, lengths_f64[2] / 2.0];

    let mut type_indices: Vec<u32> = Vec::with_capacity(particle_count);
    let mut positions_x: Vec<f32> = Vec::with_capacity(particle_count);
    let mut positions_y: Vec<f32> = Vec::with_capacity(particle_count);
    let mut positions_z: Vec<f32> = Vec::with_capacity(particle_count);
    let mut velocities_x: Vec<f32> = Vec::with_capacity(particle_count);
    let mut velocities_y: Vec<f32> = Vec::with_capacity(particle_count);
    let mut velocities_z: Vec<f32> = Vec::with_capacity(particle_count);

    let mut row_idx: usize = 0;
    let mut current_line_number: usize = 2;

    // rq-d8a08c7a rq-bc442f5b
    for line in &mut lines {
        current_line_number += 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            // Blank lines after the last data row are tolerated; blank lines
            // *inside* the data block produce a row-count mismatch below.
            continue;
        }
        if row_idx >= particle_count {
            return Err(InitStateError::TrailingContent {
                line_number: current_line_number,
            });
        }
        let cols: Vec<&str> = trimmed.split_ascii_whitespace().collect();
        if cols.len() != expected_cols {
            return Err(InitStateError::RowColumnCountMismatch {
                line_number: current_line_number,
                expected: expected_cols,
                actual: cols.len(),
            });
        }
        let species = cols[0];
        let type_index = match type_names.iter().position(|t| *t == species) {
            Some(idx) => idx as u32,
            None => {
                return Err(InitStateError::UnknownType {
                    line_number: current_line_number,
                    name: species.to_string(),
                });
            }
        };
        let px = parse_number(cols[1], current_line_number, "pos_x")?;
        let py = parse_number(cols[2], current_line_number, "pos_y")?;
        let pz = parse_number(cols[3], current_line_number, "pos_z")?;
        check_finite(px, current_line_number, "pos_x")?;
        check_finite(py, current_line_number, "pos_y")?;
        check_finite(pz, current_line_number, "pos_z")?;
        check_in_box(px, current_line_number, "x", half[0])?;
        check_in_box(py, current_line_number, "y", half[1])?;
        check_in_box(pz, current_line_number, "z", half[2])?;

        let (vx, vy, vz) = if has_velo {
            let vx = parse_number(cols[4], current_line_number, "velo_x")?;
            let vy = parse_number(cols[5], current_line_number, "velo_y")?;
            let vz = parse_number(cols[6], current_line_number, "velo_z")?;
            check_finite(vx, current_line_number, "velo_x")?;
            check_finite(vy, current_line_number, "velo_y")?;
            check_finite(vz, current_line_number, "velo_z")?;
            (vx, vy, vz)
        } else {
            (0.0, 0.0, 0.0)
        };

        type_indices.push(type_index);
        positions_x.push(px as f32);
        positions_y.push(py as f32);
        positions_z.push(pz as f32);
        if has_velo {
            velocities_x.push(vx as f32);
            velocities_y.push(vy as f32);
            velocities_z.push(vz as f32);
        }
        row_idx += 1;
    }

    if row_idx != particle_count {
        return Err(InitStateError::RowCountMismatch {
            expected: particle_count,
            actual: row_idx,
        });
    }

    let velocities = if has_velo {
        Some(InitVelocities {
            velocities_x,
            velocities_y,
            velocities_z,
        })
    } else {
        None
    };

    Ok(InitState {
        sim_box,
        particle_count,
        type_indices,
        positions_x,
        positions_y,
        positions_z,
        velocities,
    })
}

fn parse_number(
    raw: &str,
    line_number: usize,
    column: &'static str,
) -> Result<f64, InitStateError> {
    raw.parse::<f64>().map_err(|_| InitStateError::InvalidNumber {
        line_number,
        column,
        raw: raw.to_string(),
    })
}

fn check_finite(value: f64, line_number: usize, column: &'static str) -> Result<(), InitStateError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(InitStateError::NonFiniteValue {
            line_number,
            column,
        })
    }
}

fn check_in_box(
    value: f64,
    line_number: usize,
    axis: &'static str,
    half: f64,
) -> Result<(), InitStateError> {
    if value >= -half && value < half {
        Ok(())
    } else {
        Err(InitStateError::PositionOutsideBox {
            line_number,
            axis,
            value,
            half_length: half,
        })
    }
}
