// rq-965c504d rq-7a26eeae rq-c0aa3b5c
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

// CODATA 2019 value, exact.
pub const BOLTZMANN_J_PER_K: f64 = 1.380649e-23;

// rq-2344fcec
pub struct LogWriter {
    writer: BufWriter<File>,
    extras_count: usize,
}

impl std::fmt::Debug for LogWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogWriter").finish_non_exhaustive()
    }
}

// rq-45eb243b rq-e1ceb5c0
#[derive(Debug, thiserror::Error)]
pub enum LogWriterError {
    #[error("output file already exists: `{}`", .path.display())]
    OutputExists { path: PathBuf },
    #[error("failed to write log file: {0}")]
    Io(String),
}

impl LogWriter {
    // rq-e0ef1221 rq-8b4243e0
    pub fn open(path: &Path, extra_columns: &[&str]) -> Result<Self, LogWriterError> {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(file) => {
                let mut writer = BufWriter::new(file);
                writer
                    .write_all(b"step,time,kinetic_energy,temperature")
                    .map_err(io_err)?;
                for col in extra_columns {
                    writer.write_all(b",").map_err(io_err)?;
                    writer.write_all(col.as_bytes()).map_err(io_err)?;
                }
                writer.write_all(b"\n").map_err(io_err)?;
                Ok(LogWriter {
                    writer,
                    extras_count: extra_columns.len(),
                })
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Err(LogWriterError::OutputExists {
                    path: path.to_path_buf(),
                })
            }
            Err(e) => Err(LogWriterError::Io(format!("{}: {}", path.display(), e))),
        }
    }

    // rq-e409ce75 rq-4a6969aa
    pub fn write_row(
        &mut self,
        step: u64,
        time: f64,
        kinetic_energy: f64,
        temperature: f64,
        extras: &[f64],
    ) -> Result<(), LogWriterError> {
        debug_assert_eq!(
            extras.len(),
            self.extras_count,
            "extras length does not match the count declared at open()",
        );
        write!(
            self.writer,
            "{step},{time:.9e},{kinetic_energy:.9e},{temperature:.9e}"
        )
        .map_err(io_err)?;
        for v in extras {
            write!(self.writer, ",{v:.9e}").map_err(io_err)?;
        }
        writeln!(self.writer).map_err(io_err)
    }

    // rq-925e5583
    pub fn flush(&mut self) -> Result<(), LogWriterError> {
        self.writer.flush().map_err(io_err)
    }
}

impl Drop for LogWriter {
    fn drop(&mut self) {
        let _ = self.writer.flush();
    }
}

fn io_err(e: std::io::Error) -> LogWriterError {
    LogWriterError::Io(format!("{e}"))
}

// rq-6e51f09c
pub fn compute_kinetic_energy(
    masses: &[f32],
    vx: &[f32],
    vy: &[f32],
    vz: &[f32],
) -> f64 {
    debug_assert_eq!(masses.len(), vx.len());
    debug_assert_eq!(masses.len(), vy.len());
    debug_assert_eq!(masses.len(), vz.len());
    let mut sum = 0.0_f64;
    for i in 0..masses.len() {
        let m = masses[i] as f64;
        let a = vx[i] as f64;
        let b = vy[i] as f64;
        let c = vz[i] as f64;
        sum += m * (a * a + b * b + c * c);
    }
    0.5 * sum
}

// rq-46a39249
//
// Instantaneous thermodynamic temperature `T = 2K / (N_thermal_dof · k_B)`.
// `n_thermal_dof` is supplied by the runner as
// `max(0, 3 * particle_count - n_constraints - 3)`: the constraint- and
// COM-removed thermal degrees of freedom. With this convention an
// equilibrated thermostat at setpoint `T_set` produces a long-run mean of
// `temperature` equal to `T_set` to within sampling fluctuations.
pub fn compute_temperature(kinetic_energy: f64, n_thermal_dof: u32) -> f64 {
    if n_thermal_dof == 0 {
        0.0
    } else {
        2.0 * kinetic_energy / (n_thermal_dof as f64 * BOLTZMANN_J_PER_K)
    }
}
