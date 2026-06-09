// rq-965c504d rq-7a26eeae rq-c0aa3b5c
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::units::{Dimension, UnitSystem};

// k_B = 1 exactly inside the engine: temperatures are stored as
// `k_B · T` in Hartrees, so the kinetic-energy → temperature
// relation `T = 2 K / N_thermal_dof` carries no explicit Boltzmann
// factor.

// rq-2344fcec
pub struct LogWriter {
    writer: BufWriter<File>,
    units: UnitSystem,
    extra_dims: Vec<Dimension>,
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
    pub fn open(
        path: &Path,
        units: UnitSystem,
        extra_columns: &[(&str, Dimension)],
    ) -> Result<Self, LogWriterError> {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(file) => {
                let mut writer = BufWriter::new(file);
                writer
                    .write_all(b"step,time,kinetic_energy,temperature")
                    .map_err(io_err)?;
                for (col, _) in extra_columns {
                    writer.write_all(b",").map_err(io_err)?;
                    writer.write_all(col.as_bytes()).map_err(io_err)?;
                }
                writer.write_all(b"\n").map_err(io_err)?;
                Ok(LogWriter {
                    writer,
                    units,
                    extra_dims: extra_columns.iter().map(|(_, d)| *d).collect(),
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
    //
    // Each input scalar is an engine-side atomic-unit value; the writer
    // applies the output-direction conversion to the user's chosen
    // unit system before formatting.
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
            self.extra_dims.len(),
            "extras length does not match the count declared at open()",
        );
        let time_out = self.units.to_user(Dimension::Time, time);
        let ke_out = self.units.to_user(Dimension::Energy, kinetic_energy);
        let temp_out = self.units.to_user(Dimension::Temperature, temperature);
        write!(
            self.writer,
            "{step},{time_out:.9e},{ke_out:.9e},{temp_out:.9e}"
        )
        .map_err(io_err)?;
        for (v, dim) in extras.iter().zip(self.extra_dims.iter()) {
            let v_out = self.units.to_user(*dim, *v);
            write!(self.writer, ",{v_out:.9e}").map_err(io_err)?;
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
        2.0 * kinetic_energy / n_thermal_dof as f64
    }
}
