// rq-2cca54cc rq-22e4e198 rq-2196fc45
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::pbc::SimulationBox;
use crate::units::{Dimension, UnitSystem};
use crate::precision::{Real, REAL_FMT_DIGITS};

// rq-40a34caa
pub struct TrajectoryWriter {
    writer: BufWriter<File>,
    units: UnitSystem,
    include_velocities: bool,
    include_images: bool,
    type_names: Vec<String>,
}

impl std::fmt::Debug for TrajectoryWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrajectoryWriter")
            .field("include_velocities", &self.include_velocities)
            .field("include_images", &self.include_images)
            .field("type_names", &self.type_names)
            .finish_non_exhaustive()
    }
}

// rq-1fcaf334 rq-e1ceb5c0
#[derive(Debug, thiserror::Error)]
pub enum TrajectoryWriterError {
    #[error("output file already exists: `{}`", .path.display())]
    OutputExists { path: PathBuf },
    #[error("failed to write trajectory file: {0}")]
    Io(String),
}

impl TrajectoryWriter {
    // rq-28659fbe
    pub fn open(
        path: &Path,
        units: UnitSystem,
        include_velocities: bool,
        include_images: bool,
        type_names: Vec<String>,
    ) -> Result<Self, TrajectoryWriterError> {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(file) => Ok(TrajectoryWriter {
                writer: BufWriter::new(file),
                units,
                include_velocities,
                include_images,
                type_names,
            }),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Err(TrajectoryWriterError::OutputExists {
                    path: path.to_path_buf(),
                })
            }
            Err(e) => Err(TrajectoryWriterError::Io(format!("{}: {}", path.display(), e))),
        }
    }

    // rq-be899bef
    #[allow(clippy::too_many_arguments)]
    pub fn write_frame(
        &mut self,
        step: u64,
        dt: f64,
        sim_box: &SimulationBox,
        type_indices: &[u32],
        positions_x: &[Real],
        positions_y: &[Real],
        positions_z: &[Real],
        velocities: Option<(&[Real], &[Real], &[Real])>,
        images: Option<(&[i32], &[i32], &[i32])>,
    ) -> Result<(), TrajectoryWriterError> {
        let n = type_indices.len();
        debug_assert_eq!(positions_x.len(), n);
        debug_assert_eq!(positions_y.len(), n);
        debug_assert_eq!(positions_z.len(), n);
        if self.include_velocities {
            let (vx, vy, vz) = velocities.expect("include_velocities=true requires velocities");
            debug_assert_eq!(vx.len(), n);
            debug_assert_eq!(vy.len(), n);
            debug_assert_eq!(vz.len(), n);
        } else {
            debug_assert!(velocities.is_none());
        }
        if self.include_images {
            let (ix, iy, iz) = images.expect("include_images=true requires images");
            debug_assert_eq!(ix.len(), n);
            debug_assert_eq!(iy.len(), n);
            debug_assert_eq!(iz.len(), n);
        } else {
            debug_assert!(images.is_none());
        }

        let lat = sim_box.lattice();
        let time = (step as f64) * dt;
        // Output-direction conversion factors. For UnitSystem::Atomic
        // these all reduce to 1.0 and the engine's internal scalars
        // are emitted directly.
        let len_f = self.units.factor(Dimension::Length) as Real;
        let vel_f = self.units.factor(Dimension::Velocity) as Real;
        let time_out = self.units.to_user(Dimension::Time, time);

        // rq-1658f77d rq-c5518458 rq-e06bcfb0 rq-df244549 rq-6ec75323 rq-88ec92fc
        //
        // Emit the 9-component lattice in row-major order with the three
        // upper-triangular slots fixed to 0.0 (lower-triangular form).
        // Orthorhombic boxes (xy = xz = yz = 0) print three middle zeros
        // and are byte-identical to the previous orthorhombic-only format.
        writeln!(self.writer, "{n}").map_err(io_err)?;
        let zero = 0.0;
        let props = match (self.include_velocities, self.include_images) {
            (false, false) => "species:S:1:pos:R:3",
            (false, true) => "species:S:1:pos:R:3:image:I:3",
            (true, false) => "species:S:1:pos:R:3:velo:R:3",
            (true, true) => "species:S:1:pos:R:3:velo:R:3:image:I:3",
        };
        let p = REAL_FMT_DIGITS;
        writeln!(
            self.writer,
            "Lattice=\"{lx:.p$e} {z:.p$e} {z:.p$e} {xy:.p$e} {ly:.p$e} {z:.p$e} {xz:.p$e} {yz:.p$e} {lz:.p$e}\" Properties={props} Step={step} Time={time:.9e}",
            lx = lat[0] * len_f,
            ly = lat[1] * len_f,
            lz = lat[2] * len_f,
            xy = lat[3] * len_f,
            xz = lat[4] * len_f,
            yz = lat[5] * len_f,
            z = zero,
            props = props,
            step = step,
            time = time_out,
            p = p,
        )
        .map_err(io_err)?;

        // rq-00c68095
        for i in 0..n {
            let name = self
                .type_names
                .get(type_indices[i] as usize)
                .map(|s| s.as_str())
                .unwrap_or("?");
            write!(
                self.writer,
                "{name} {:.p$e} {:.p$e} {:.p$e}",
                positions_x[i] * len_f,
                positions_y[i] * len_f,
                positions_z[i] * len_f,
                p = p,
            )
            .map_err(io_err)?;
            if self.include_velocities {
                let (vx, vy, vz) = velocities.unwrap();
                write!(
                    self.writer,
                    " {:.p$e} {:.p$e} {:.p$e}",
                    vx[i] * vel_f,
                    vy[i] * vel_f,
                    vz[i] * vel_f,
                    p = p,
                )
                .map_err(io_err)?;
            }
            if self.include_images {
                let (ix, iy, iz) = images.unwrap();
                write!(self.writer, " {} {} {}", ix[i], iy[i], iz[i]).map_err(io_err)?;
            }
            writeln!(self.writer).map_err(io_err)?;
        }
        Ok(())
    }

    // rq-2ad32a7b
    pub fn flush(&mut self) -> Result<(), TrajectoryWriterError> {
        self.writer.flush().map_err(io_err)
    }

    pub fn include_velocities(&self) -> bool {
        self.include_velocities
    }

    pub fn include_images(&self) -> bool {
        self.include_images
    }
}

impl Drop for TrajectoryWriter {
    fn drop(&mut self) {
        let _ = self.writer.flush();
    }
}

fn io_err(e: std::io::Error) -> TrajectoryWriterError {
    TrajectoryWriterError::Io(format!("{e}"))
}

// =====================================================================
// TrajectoryReader — companion to TrajectoryWriter, used by
// `heddlemd analyze` to walk every frame in declaration order.
// =====================================================================

// rq-972e319b
#[derive(Debug, Clone)]
pub struct TrajectoryFrameHeader {
    pub particle_count: usize,
    pub sim_box: SimulationBox,
    pub type_indices: Vec<u32>,
    pub include_velocities: bool,
    pub include_images: bool,
}

// rq-f3e82236
#[derive(Debug)]
pub struct TrajectoryFrame {
    pub step: u64,
    pub time: f64,
    pub sim_box: SimulationBox,
    pub type_indices: Vec<u32>,
    pub positions_x: Vec<Real>,
    pub positions_y: Vec<Real>,
    pub positions_z: Vec<Real>,
    pub velocities: Option<(Vec<Real>, Vec<Real>, Vec<Real>)>,
    pub images: Option<(Vec<i32>, Vec<i32>, Vec<i32>)>,
}

// rq-9adb9a5b
#[derive(Debug, thiserror::Error)]
pub enum TrajectoryReaderError {
    #[error("failed to read trajectory file: {0}")]
    Io(String),
    #[error("trajectory file is empty")]
    Empty,
    #[error("malformed trajectory header on line {line_number}: {reason}")]
    MalformedHeader { line_number: usize, reason: String },
    #[error("unknown particle type `{name}` on line {line_number}")]
    UnknownType { line_number: usize, name: String },
    #[error("frame {frame_index} disagrees with first-frame header on line {line_number}: {reason}")]
    FrameMismatch {
        frame_index: u64,
        line_number: usize,
        reason: String,
    },
    #[error("malformed trajectory row on line {line_number}: {reason}")]
    MalformedRow { line_number: usize, reason: String },
    #[error("trajectory file ended mid-frame (frame {frame_index}: expected {expected} rows, got {got})")]
    Truncated {
        expected: usize,
        got: usize,
        frame_index: u64,
    },
}

// rq-d1814271
pub struct TrajectoryReader {
    reader: BufReader<File>,
    // rq-2456a906
    pub first_frame_header: TrajectoryFrameHeader,
    /// 1-based current line number in the file (updated as lines are read).
    line_number: usize,
    /// 0-based index of the next frame to return from `next_frame`.
    next_frame_index: u64,
    /// Filename, cached for error messages.
    path: PathBuf,
    /// Owned type-name list, kept alive for the lifetime of the reader so
    /// it can be re-used by `next_frame` to map species columns to indices
    /// without the caller having to thread the slice through every call.
    type_names: Vec<String>,
    /// Buffer reused for reading lines.
    line_buf: String,
    /// Unit system the file is written in. The reader applies
    /// `from_user` to every numeric column on read so the
    /// `TrajectoryFrame` slices and `SimulationBox` it exposes are in
    /// the engine's atomic units.
    units: UnitSystem,
    /// `Some` when `open` peeked the first frame's data rows during
    /// header construction (they are buffered here so `next_frame` can
    /// return them on the first call without re-reading).
    first_frame: Option<TrajectoryFrame>,
}

impl std::fmt::Debug for TrajectoryReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrajectoryReader")
            .field("path", &self.path)
            .field("first_frame_header", &self.first_frame_header)
            .field("next_frame_index", &self.next_frame_index)
            .finish_non_exhaustive()
    }
}

impl TrajectoryReader {
    // rq-af1c88ae
    pub fn open(
        path: &Path,
        units: UnitSystem,
        type_names: &[&str],
    ) -> Result<TrajectoryReader, TrajectoryReaderError> {
        let file = File::open(path)
            .map_err(|e| TrajectoryReaderError::Io(format!("{}: {}", path.display(), e)))?;
        let mut reader = BufReader::new(file);
        let owned_type_names: Vec<String> =
            type_names.iter().map(|s| (*s).to_string()).collect();

        // Read the first frame in full so we can populate
        // `first_frame_header` and stash the frame for the first
        // `next_frame` call.
        let mut line_number: usize = 0;
        let mut line_buf = String::new();
        let first_frame = read_one_frame(
            &mut reader,
            &owned_type_names,
            &mut line_number,
            &mut line_buf,
            0,
            None,
            units,
        )?;
        let first_frame = match first_frame {
            Some(f) => f,
            None => return Err(TrajectoryReaderError::Empty),
        };

        let header = TrajectoryFrameHeader {
            particle_count: first_frame.type_indices.len(),
            sim_box: first_frame.sim_box.clone(),
            type_indices: first_frame.type_indices.clone(),
            include_velocities: first_frame.velocities.is_some(),
            include_images: first_frame.images.is_some(),
        };

        Ok(TrajectoryReader {
            reader,
            first_frame_header: header,
            line_number,
            next_frame_index: 0,
            path: path.to_path_buf(),
            type_names: owned_type_names,
            line_buf,
            units,
            first_frame: Some(first_frame),
        })
    }

    // rq-e2e25bba
    pub fn next_frame(&mut self) -> Result<Option<TrajectoryFrame>, TrajectoryReaderError> {
        if let Some(frame) = self.first_frame.take() {
            self.next_frame_index = 1;
            return Ok(Some(frame));
        }
        let frame = read_one_frame(
            &mut self.reader,
            &self.type_names,
            &mut self.line_number,
            &mut self.line_buf,
            self.next_frame_index,
            Some(&self.first_frame_header),
            self.units,
        )?;
        if frame.is_some() {
            self.next_frame_index += 1;
        }
        Ok(frame)
    }

    // rq-d2b595a3
    pub fn frames(&mut self) -> TrajectoryFrameIter<'_> {
        TrajectoryFrameIter { reader: self }
    }
}

pub struct TrajectoryFrameIter<'a> {
    reader: &'a mut TrajectoryReader,
}

impl<'a> Iterator for TrajectoryFrameIter<'a> {
    type Item = Result<TrajectoryFrame, TrajectoryReaderError>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.reader.next_frame() {
            Ok(Some(frame)) => Some(Ok(frame)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Read one frame from `reader`. Returns `Ok(None)` on clean EOF (no
/// further frames). The caller passes `expected_header = None` when
/// reading the first frame and `Some(&header)` afterward so the reader
/// can detect mid-trajectory header changes.
fn read_one_frame<R: BufRead>(
    reader: &mut R,
    type_names: &[String],
    line_number: &mut usize,
    line_buf: &mut String,
    frame_index: u64,
    expected_header: Option<&TrajectoryFrameHeader>,
    units: UnitSystem,
) -> Result<Option<TrajectoryFrame>, TrajectoryReaderError> {
    let len_f = units.factor(Dimension::Length) as Real;
    let vel_f = units.factor(Dimension::Velocity) as Real;
    // Particle-count line.
    let count_line = match read_nonblank_line(reader, line_number, line_buf)? {
        Some(s) => s,
        None => return Ok(None),
    };
    let particle_count: usize = count_line.trim().parse::<i64>().ok().and_then(|n| {
        if n >= 0 { Some(n as usize) } else { None }
    }).ok_or_else(|| TrajectoryReaderError::MalformedHeader {
        line_number: *line_number,
        reason: format!("particle-count line is not a non-negative integer: {count_line:?}"),
    })?;

    // Comment line.
    line_buf.clear();
    let bytes = reader
        .read_line(line_buf)
        .map_err(|e| TrajectoryReaderError::Io(format!("{e}")))?;
    if bytes == 0 {
        return Err(TrajectoryReaderError::MalformedHeader {
            line_number: *line_number,
            reason: "expected comment line after particle-count line, got EOF".to_string(),
        });
    }
    *line_number += 1;
    let comment_line_number = *line_number;
    let comment = line_buf.trim_end_matches('\n').trim_end_matches('\r').to_string();
    let attrs = parse_attribute_line(&comment);

    let lattice_value = attrs
        .iter()
        .find(|(k, _)| k == "Lattice")
        .map(|(_, v)| v.as_str())
        .ok_or_else(|| TrajectoryReaderError::MalformedHeader {
            line_number: comment_line_number,
            reason: "missing `Lattice` attribute".to_string(),
        })?;
    let sim_box = parse_lattice(lattice_value, len_f).map_err(|e| {
        TrajectoryReaderError::MalformedHeader {
            line_number: comment_line_number,
            reason: e,
        }
    })?;

    let properties_value = attrs
        .iter()
        .find(|(k, _)| k == "Properties")
        .map(|(_, v)| v.as_str())
        .ok_or_else(|| TrajectoryReaderError::MalformedHeader {
            line_number: comment_line_number,
            reason: "missing `Properties` attribute".to_string(),
        })?;
    let (include_velocities, include_images) =
        parse_properties(properties_value).map_err(|e| {
            TrajectoryReaderError::MalformedHeader {
                line_number: comment_line_number,
                reason: e,
            }
        })?;
    let step_value: u64 = attrs
        .iter()
        .find(|(k, _)| k == "Step")
        .and_then(|(_, v)| v.parse::<u64>().ok())
        .unwrap_or(0);
    let time_value: f64 = attrs
        .iter()
        .find(|(k, _)| k == "Time")
        .and_then(|(_, v)| v.parse::<f64>().ok())
        .map(|t| units.from_user(Dimension::Time, t))
        .unwrap_or(0.0);

    // Cross-check against expected_header (frames 1..).
    if let Some(expected) = expected_header {
        if particle_count != expected.particle_count {
            return Err(TrajectoryReaderError::FrameMismatch {
                frame_index,
                line_number: comment_line_number,
                reason: format!(
                    "particle_count={particle_count} differs from first-frame {}",
                    expected.particle_count
                ),
            });
        }
        if include_velocities != expected.include_velocities {
            return Err(TrajectoryReaderError::FrameMismatch {
                frame_index,
                line_number: comment_line_number,
                reason: "`Properties` velocity column presence differs from first-frame header"
                    .to_string(),
            });
        }
        if include_images != expected.include_images {
            return Err(TrajectoryReaderError::FrameMismatch {
                frame_index,
                line_number: comment_line_number,
                reason: "`Properties` image column presence differs from first-frame header"
                    .to_string(),
            });
        }
    }

    let expected_cols: usize = 4
        + if include_velocities { 3 } else { 0 }
        + if include_images { 3 } else { 0 };
    let image_offset = if include_velocities { 7 } else { 4 };

    let mut type_indices: Vec<u32> = Vec::with_capacity(particle_count);
    let mut positions_x: Vec<Real> = Vec::with_capacity(particle_count);
    let mut positions_y: Vec<Real> = Vec::with_capacity(particle_count);
    let mut positions_z: Vec<Real> = Vec::with_capacity(particle_count);
    let mut velocities_x: Vec<Real> = Vec::with_capacity(particle_count);
    let mut velocities_y: Vec<Real> = Vec::with_capacity(particle_count);
    let mut velocities_z: Vec<Real> = Vec::with_capacity(particle_count);
    let mut images_x: Vec<i32> = Vec::with_capacity(particle_count);
    let mut images_y: Vec<i32> = Vec::with_capacity(particle_count);
    let mut images_z: Vec<i32> = Vec::with_capacity(particle_count);

    for row_idx in 0..particle_count {
        let row = match read_nonblank_line(reader, line_number, line_buf)? {
            Some(s) => s,
            None => {
                return Err(TrajectoryReaderError::Truncated {
                    expected: particle_count,
                    got: row_idx,
                    frame_index,
                });
            }
        };
        let cols: Vec<&str> = row.split_ascii_whitespace().collect();
        if cols.len() != expected_cols {
            return Err(TrajectoryReaderError::MalformedRow {
                line_number: *line_number,
                reason: format!(
                    "expected {expected_cols} columns, got {}",
                    cols.len()
                ),
            });
        }
        let species = cols[0];
        let type_index = match type_names.iter().position(|t| t == species) {
            Some(idx) => idx as u32,
            None => {
                return Err(TrajectoryReaderError::UnknownType {
                    line_number: *line_number,
                    name: species.to_string(),
                });
            }
        };
        if let Some(expected) = expected_header {
            if expected.type_indices[row_idx] != type_index {
                return Err(TrajectoryReaderError::FrameMismatch {
                    frame_index,
                    line_number: *line_number,
                    reason: format!(
                        "row {row_idx} species `{species}` (index {type_index}) differs from first-frame index {}",
                        expected.type_indices[row_idx]
                    ),
                });
            }
        }
        let px = parse_f64_col(cols[1], *line_number, "pos_x")? as Real;
        let py = parse_f64_col(cols[2], *line_number, "pos_y")? as Real;
        let pz = parse_f64_col(cols[3], *line_number, "pos_z")? as Real;
        type_indices.push(type_index);
        positions_x.push(px / len_f);
        positions_y.push(py / len_f);
        positions_z.push(pz / len_f);
        if include_velocities {
            let vx = parse_f64_col(cols[4], *line_number, "velo_x")? as Real;
            let vy = parse_f64_col(cols[5], *line_number, "velo_y")? as Real;
            let vz = parse_f64_col(cols[6], *line_number, "velo_z")? as Real;
            velocities_x.push(vx / vel_f);
            velocities_y.push(vy / vel_f);
            velocities_z.push(vz / vel_f);
        }
        if include_images {
            let ix = parse_i32_col(cols[image_offset], *line_number, "image_x")?;
            let iy = parse_i32_col(cols[image_offset + 1], *line_number, "image_y")?;
            let iz = parse_i32_col(cols[image_offset + 2], *line_number, "image_z")?;
            images_x.push(ix);
            images_y.push(iy);
            images_z.push(iz);
        }
    }

    Ok(Some(TrajectoryFrame {
        step: step_value,
        time: time_value,
        sim_box,
        type_indices,
        positions_x,
        positions_y,
        positions_z,
        velocities: if include_velocities {
            Some((velocities_x, velocities_y, velocities_z))
        } else {
            None
        },
        images: if include_images {
            Some((images_x, images_y, images_z))
        } else {
            None
        },
    }))
}

fn read_nonblank_line<R: BufRead>(
    reader: &mut R,
    line_number: &mut usize,
    buf: &mut String,
) -> Result<Option<String>, TrajectoryReaderError> {
    loop {
        buf.clear();
        let bytes = reader
            .read_line(buf)
            .map_err(|e| TrajectoryReaderError::Io(format!("{e}")))?;
        if bytes == 0 {
            return Ok(None);
        }
        *line_number += 1;
        let trimmed = buf.trim_end_matches('\n').trim_end_matches('\r');
        if trimmed.trim().is_empty() {
            continue;
        }
        return Ok(Some(trimmed.to_string()));
    }
}

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
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            continue;
        }
        let key = std::str::from_utf8(&bytes[key_start..i]).unwrap().to_string();
        i += 1;
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

fn parse_lattice(raw: &str, len_f: Real) -> Result<SimulationBox, String> {
    let parts: Vec<&str> = raw.split_ascii_whitespace().collect();
    if parts.len() != 9 {
        return Err(format!(
            "`Lattice` expected 9 components, got {}",
            parts.len()
        ));
    }
    let mut vals = [0.0_f64; 9];
    for (i, p) in parts.iter().enumerate() {
        vals[i] = p
            .parse::<f64>()
            .map_err(|_| format!("`Lattice` component {i} is not a number: {p:?}"))?;
    }
    for (i, v) in vals.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("`Lattice` component {i} is not finite: {v}"));
        }
    }
    // Lower-triangular form: positions 1, 2, 5 (a_y, a_z, b_z) must be 0.
    for (i, name) in [(1, "a_y"), (2, "a_z"), (5, "b_z")].iter() {
        if vals[*i] != 0.0 {
            return Err(format!(
                "`Lattice` has non-zero upper-triangular component `{name}` = {}",
                vals[*i]
            ));
        }
    }
    // Convert from the file's units to engine-side atomic units by
    // dividing by `len_f` (the user-system value of one atomic length
    // unit). No-op when the file is already in atomic units.
    let lx = vals[0] as Real / len_f;
    let xy = vals[3] as Real / len_f;
    let ly = vals[4] as Real / len_f;
    let xz = vals[6] as Real / len_f;
    let yz = vals[7] as Real / len_f;
    let lz = vals[8] as Real / len_f;
    SimulationBox::new(lx, ly, lz, xy, xz, yz)
        .map_err(|e| format!("`Lattice` produced an invalid SimulationBox: {e}"))
}

fn parse_properties(raw: &str) -> Result<(bool, bool), String> {
    match raw {
        "species:S:1:pos:R:3" => Ok((false, false)),
        "species:S:1:pos:R:3:image:I:3" => Ok((false, true)),
        "species:S:1:pos:R:3:velo:R:3" => Ok((true, false)),
        "species:S:1:pos:R:3:velo:R:3:image:I:3" => Ok((true, true)),
        other => Err(format!(
            "`Properties` value `{other}` is not one of the four accepted forms"
        )),
    }
}

fn parse_f64_col(
    raw: &str,
    line_number: usize,
    column: &'static str,
) -> Result<f64, TrajectoryReaderError> {
    let v: f64 = raw
        .parse()
        .map_err(|_| TrajectoryReaderError::MalformedRow {
            line_number,
            reason: format!("column `{column}` is not a number: {raw:?}"),
        })?;
    if !v.is_finite() {
        return Err(TrajectoryReaderError::MalformedRow {
            line_number,
            reason: format!("column `{column}` is not finite: {v}"),
        });
    }
    Ok(v)
}

fn parse_i32_col(
    raw: &str,
    line_number: usize,
    column: &'static str,
) -> Result<i32, TrajectoryReaderError> {
    raw.parse::<i32>().map_err(|_| TrajectoryReaderError::MalformedRow {
        line_number,
        reason: format!("column `{column}` is not an i32: {raw:?}"),
    })
}
