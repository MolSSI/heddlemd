// Per-iteration diagnostic CSV log for energy minimization phases.
// See `rqm/minimization/steepest-descent.md`.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub struct MinlogWriter {
    writer: BufWriter<File>,
}

impl std::fmt::Debug for MinlogWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MinlogWriter").finish_non_exhaustive()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MinlogWriterError {
    #[error("output file already exists: `{}`", .path.display())]
    OutputExists { path: PathBuf },
    #[error("failed to write minlog file: {0}")]
    Io(String),
}

impl MinlogWriter {
    pub fn open(path: &Path) -> Result<Self, MinlogWriterError> {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(file) => {
                let mut writer = BufWriter::new(file);
                writer
                    .write_all(b"iter,energy,max_force,step,accepted\n")
                    .map_err(io_err)?;
                Ok(MinlogWriter { writer })
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
        writeln!(
            self.writer,
            "{iter},{energy:.9e},{max_force:.9e},{step:.9e},{acc}"
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
