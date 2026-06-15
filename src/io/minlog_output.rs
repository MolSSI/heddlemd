// rq-119cbe46 — Per-iteration diagnostic CSV log for energy minimization phases.
// See `rqm/minimization/steepest-descent.md`.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::precision::REAL_FMT_DIGITS;
use crate::units::{Dimension, UnitSystem};

// rq-dc140510
pub struct MinlogWriter {
    writer: BufWriter<File>,
    units: UnitSystem,
}

impl std::fmt::Debug for MinlogWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MinlogWriter").finish_non_exhaustive()
    }
}

// rq-82621837
#[derive(Debug, thiserror::Error)]
pub enum MinlogWriterError {
    #[error("output file already exists: `{}`", .path.display())]
    OutputExists { path: PathBuf },
    #[error("failed to write minlog file: {0}")]
    Io(String),
}

impl MinlogWriter {
    // rq-e963f27a
    pub fn open(path: &Path, units: UnitSystem) -> Result<Self, MinlogWriterError> {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(file) => {
                let mut writer = BufWriter::new(file);
                writer
                    .write_all(b"iter,energy,max_force,step,accepted\n")
                    .map_err(io_err)?;
                Ok(MinlogWriter { writer, units })
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Err(MinlogWriterError::OutputExists {
                    path: path.to_path_buf(),
                })
            }
            Err(e) => Err(MinlogWriterError::Io(format!("{}: {}", path.display(), e))),
        }
    }

    pub fn write_row(
        &mut self,
        iter: u64,
        energy: f64,
        max_force: f64,
        step: f64,
        accepted: bool,
    ) -> Result<(), MinlogWriterError> {
        let acc = if accepted { 1 } else { 0 };
        let energy_out = self.units.to_user(Dimension::Energy, energy);
        let force_out = self.units.to_user(Dimension::Force, max_force);
        let step_out = self.units.to_user(Dimension::Length, step);
        let p = REAL_FMT_DIGITS;
        writeln!(
            self.writer,
            "{iter},{energy_out:.p$e},{force_out:.p$e},{step_out:.p$e},{acc}",
            p = p,
        )
        .map_err(io_err)
    }

    pub fn flush(&mut self) -> Result<(), MinlogWriterError> {
        self.writer.flush().map_err(io_err)
    }
}

impl Drop for MinlogWriter {
    fn drop(&mut self) {
        let _ = self.writer.flush();
    }
}

fn io_err(e: std::io::Error) -> MinlogWriterError {
    MinlogWriterError::Io(format!("{e}"))
}
