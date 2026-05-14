// rq-2cca54cc rq-22e4e198 rq-2196fc45
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::pbc::SimulationBox;

// rq-40a34caa
pub struct TrajectoryWriter {
    writer: BufWriter<File>,
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
        include_velocities: bool,
        include_images: bool,
        type_names: Vec<String>,
    ) -> Result<Self, TrajectoryWriterError> {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(file) => Ok(TrajectoryWriter {
                writer: BufWriter::new(file),
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
        positions_x: &[f32],
        positions_y: &[f32],
        positions_z: &[f32],
        velocities: Option<(&[f32], &[f32], &[f32])>,
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

        let lengths = sim_box.lengths();
        let time = (step as f64) * dt;

        // rq-1658f77d rq-c5518458 rq-e06bcfb0 rq-df244549 rq-6ec75323 rq-88ec92fc
        writeln!(self.writer, "{n}").map_err(io_err)?;
        let zero = 0.0_f32;
        let props = match (self.include_velocities, self.include_images) {
            (false, false) => "species:S:1:pos:R:3",
            (false, true) => "species:S:1:pos:R:3:image:I:3",
            (true, false) => "species:S:1:pos:R:3:velo:R:3",
            (true, true) => "species:S:1:pos:R:3:velo:R:3:image:I:3",
        };
        writeln!(
            self.writer,
            "Lattice=\"{lx:.9e} {z:.9e} {z:.9e} {z:.9e} {ly:.9e} {z:.9e} {z:.9e} {z:.9e} {lz:.9e}\" Properties={props} Step={step} Time={time:.9e}",
            lx = lengths[0],
            ly = lengths[1],
            lz = lengths[2],
            z = zero,
            props = props,
            step = step,
            time = time,
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
                "{name} {:.9e} {:.9e} {:.9e}",
                positions_x[i], positions_y[i], positions_z[i]
            )
            .map_err(io_err)?;
            if self.include_velocities {
                let (vx, vy, vz) = velocities.unwrap();
                write!(
                    self.writer,
                    " {:.9e} {:.9e} {:.9e}",
                    vx[i], vy[i], vz[i]
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
