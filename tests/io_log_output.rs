use std::path::{Path, PathBuf};

use dynamics::io::{
    LogWriter, LogWriterError, compute_kinetic_energy, compute_temperature,
};
use dynamics::units::{Dimension, UnitSystem};

// k_B = 1 in the engine's atomic units. Tests still need an SI value
// to set up SI-mode kinetic energies whose `compute_temperature` output
// is verified in either system.
#[allow(dead_code)]
const BOLTZMANN_J_PER_K: f64 = 1.380649e-23;

fn tmp_path(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!(
        "dynamics-log-{}-{}-{}",
        std::process::id(),
        name,
        nanos
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap()
}

// rq-6d087460
#[test]
fn open_creates_log_with_header() {
    let dir = tmp_path("open_header");
    let path = dir.join("run.log");
    let mut writer = LogWriter::open(&path, UnitSystem::Atomic, &[]).unwrap();
    writer.flush().unwrap();
    drop(writer);
    let body = read(&path);
    assert_eq!(body, "step,time,kinetic_energy,temperature\n");
}

// rq-f20e017d
#[test]
fn open_refuses_existing_log() {
    let dir = tmp_path("refuse_existing");
    let path = dir.join("run.log");
    std::fs::write(&path, "preexisting").unwrap();
    match LogWriter::open(&path, UnitSystem::Atomic, &[]).unwrap_err() {
        LogWriterError::OutputExists { path: p } => assert_eq!(p, path),
        other => panic!("unexpected: {other:?}"),
    }
    assert_eq!(read(&path), "preexisting");
}

// rq-9baf16d1
#[test]
fn open_fails_missing_parent() {
    let dir = tmp_path("missing_parent");
    let path = dir.join("missing").join("run.log");
    match LogWriter::open(&path, UnitSystem::Atomic, &[]).unwrap_err() {
        LogWriterError::Io(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-90517bb6
#[test]
fn write_single_row_step_zero() {
    let dir = tmp_path("row_step_zero");
    let path = dir.join("run.log");
    let mut writer = LogWriter::open(&path, UnitSystem::Atomic, &[]).unwrap();
    writer.write_row(0, 0.0, 0.0, 300.0, &[]).unwrap();
    writer.flush().unwrap();
    let body = read(&path);
    let expected = format!(
        "step,time,kinetic_energy,temperature\n0,{},{},{}\n",
        format_args!("{:.9e}", 0.0_f64),
        format_args!("{:.9e}", 0.0_f64),
        format_args!("{:.9e}", 300.0_f64),
    );
    assert_eq!(body, expected);
}

// rq-9198cc8e
// rq-4381eec2 rq-a9cb9e03 rq-9c883334
#[test]
fn si_mode_log_row_multiplies_time_kinetic_energy_and_temperature_by_factors() {
    // Open the log writer with UnitSystem::Si. The engine state is in
    // atomic units; the writer must multiply time by the atu→second
    // factor, kinetic_energy by the Hartree→Joule factor, and
    // temperature by the (E_h/k_B)→Kelvin factor before formatting.
    let dir = tmp_path("si_log_factors");
    let path = dir.join("run.log");
    let mut writer = LogWriter::open(&path, UnitSystem::Si, &[]).unwrap();
    let t_au: f64 = 41.34;     // ~1 fs in atomic time units
    let ke_au: f64 = 0.0128;   // arbitrary Hartrees
    let temp_au: f64 = 9.5e-4; // ~300 K in Hartree/k_B
    writer.write_row(100, t_au, ke_au, temp_au, &[]).unwrap();
    writer.flush().unwrap();

    let time_factor = UnitSystem::Si.factor(Dimension::Time);
    let energy_factor = UnitSystem::Si.factor(Dimension::Energy);
    let temp_factor = UnitSystem::Si.factor(Dimension::Temperature);

    let body = std::fs::read_to_string(&path).unwrap();
    let last = body.lines().last().unwrap();
    let cols: Vec<&str> = last.split(',').collect();
    assert_eq!(cols.len(), 4, "header + step,time,ke,temperature row, got {cols:?}");
    assert_eq!(cols[0], "100");
    let t_si: f64 = cols[1].parse().unwrap();
    let ke_si: f64 = cols[2].parse().unwrap();
    let temp_si: f64 = cols[3].parse().unwrap();
    let rel = 1e-9;
    let approx = |a: f64, b: f64| (a - b).abs() <= rel * a.abs().max(b.abs()).max(1e-300);
    assert!(approx(t_si, t_au * time_factor),
            "time SI {} != t_au * time_factor {}", t_si, t_au * time_factor);
    assert!(approx(ke_si, ke_au * energy_factor),
            "kinetic_energy SI {} != ke_au * energy_factor {}", ke_si, ke_au * energy_factor);
    assert!(approx(temp_si, temp_au * temp_factor),
            "temperature SI {} != temp_au * temp_factor {}", temp_si, temp_au * temp_factor);
}

// rq-5b934ecd rq-7986038d
#[test]
fn write_row_non_trivial_values() {
    let dir = tmp_path("row_nontrivial");
    let path = dir.join("run.log");
    let mut writer = LogWriter::open(&path, UnitSystem::Atomic, &[]).unwrap();
    writer
        .write_row(100, 1.0e-13, 4.123456789e-21, 298.7654321, &[]).unwrap();
    writer.flush().unwrap();
    let body = read(&path);
    let last = body.lines().last().unwrap();
    let expected = format!(
        "100,{},{},{}",
        format_args!("{:.9e}", 1.0e-13_f64),
        format_args!("{:.9e}", 4.123456789e-21_f64),
        format_args!("{:.9e}", 298.7654321_f64),
    );
    assert_eq!(last, expected);
}

// rq-3ef10542
#[test]
fn append_rows_in_order() {
    let dir = tmp_path("append_rows");
    let path = dir.join("run.log");
    let mut writer = LogWriter::open(&path, UnitSystem::Atomic, &[]).unwrap();
    writer.write_row(0, 0.0, 0.0, 0.0, &[]).unwrap();
    writer.write_row(100, 1.0e-13, 1.0, 100.0, &[]).unwrap();
    writer.write_row(200, 2.0e-13, 2.0, 200.0, &[]).unwrap();
    writer.flush().unwrap();
    let body = read(&path);
    let lines: Vec<&str> = body.lines().collect();
    assert_eq!(lines.len(), 4);
    assert!(lines[1].starts_with("0,"));
    assert!(lines[2].starts_with("100,"));
    assert!(lines[3].starts_with("200,"));
}

// rq-107a7187
#[test]
fn ke_of_single_at_rest() {
    let ke = compute_kinetic_energy(&[1.0_f32], &[0.0_f32], &[0.0_f32], &[0.0_f32]);
    assert_eq!(ke, 0.0);
}

// rq-7c23d271
#[test]
fn ke_of_single_v_along_x() {
    let ke = compute_kinetic_energy(&[2.0_f32], &[1.0_f32], &[0.0_f32], &[0.0_f32]);
    assert_eq!(ke, 1.0_f64);
}

// rq-553f28a3
#[test]
fn ke_three_particles_in_order() {
    let masses = [1.0_f32, 2.0_f32, 4.0_f32];
    let vx = [1.0_f32; 3];
    let vy = [0.0_f32; 3];
    let vz = [0.0_f32; 3];
    let ke = compute_kinetic_energy(&masses, &vx, &vy, &vz);
    // Each contribution: m_i * 1.0; sum = 1+2+4=7; KE = 0.5 * 7 = 3.5
    assert_eq!(ke, 3.5_f64);
}

// rq-1feec66c
#[test]
fn ke_bit_identical_across_invocations() {
    let masses = [1.0_f32, 2.5_f32, 7.5_f32];
    let vx = [0.1_f32, -0.3_f32, 0.7_f32];
    let vy = [0.4_f32, 0.5_f32, -0.6_f32];
    let vz = [-0.2_f32, 0.0_f32, 0.8_f32];
    let a = compute_kinetic_energy(&masses, &vx, &vy, &vz);
    let b = compute_kinetic_energy(&masses, &vx, &vy, &vz);
    assert_eq!(a.to_bits(), b.to_bits());
}

// rq-fa6f7414
#[test]
fn ke_of_empty_is_zero() {
    let ke = compute_kinetic_energy(&[], &[], &[], &[]);
    assert_eq!(ke, 0.0);
}

// rq-8f554438
#[test]
fn temperature_of_empty_is_zero() {
    assert_eq!(compute_temperature(0.0, 0), 0.0);
}

// rq-7d831804 rq-fee5b8e2 rq-9d8f0c97
#[test]
fn temperature_uses_kb_unity() {
    // k_B = 1 inside the engine: temperature is `k_B · T` in Hartrees.
    // 10 unconstrained particles, COM-removed: N_thermal_dof = 3*10 - 0 - 3 = 27.
    let n_thermal_dof: u32 = 27;
    let t_target_au = 9.5e-4_f64; // ~300 K in atomic units
    let ke = 0.5 * (n_thermal_dof as f64) * t_target_au;
    let t = compute_temperature(ke, n_thermal_dof);
    assert!((t - t_target_au).abs() < 1.0e-12);
}

// rq-1b97afd8
#[test]
fn log_header_plus_step_zero_only() {
    let dir = tmp_path("step_zero_only");
    let path = dir.join("run.log");
    let mut writer = LogWriter::open(&path, UnitSystem::Atomic, &[]).unwrap();
    writer.write_row(0, 0.0, 0.0, 0.0, &[]).unwrap();
    writer.flush().unwrap();
    let body = read(&path);
    assert_eq!(body.lines().count(), 2);
}

// rq-9d0ea87b
#[test]
fn log_flush_idempotent() {
    let dir = tmp_path("flush_idem");
    let path = dir.join("run.log");
    let mut writer = LogWriter::open(&path, UnitSystem::Atomic, &[]).unwrap();
    writer.flush().unwrap();
    writer.flush().unwrap();
}

// rq-02bde767
#[test]
fn drop_flushes_log() {
    let dir = tmp_path("drop_flush");
    let path = dir.join("run.log");
    {
        let mut writer = LogWriter::open(&path, UnitSystem::Atomic, &[]).unwrap();
        writer.write_row(0, 0.0, 0.0, 0.0, &[]).unwrap();
    }
    let body = read(&path);
    assert!(body.contains("step,time,"));
    assert_eq!(body.lines().count(), 2);
}

// rq-3ba42667
#[test]
fn open_with_extra_columns_appends_to_header() {
    let dir = tmp_path("open_extras_header");
    let path = dir.join("run.log");
    let mut writer = LogWriter::open(&path, UnitSystem::Atomic, &[
        ("nhc_conserved", Dimension::Energy),
    ]).unwrap();
    writer.flush().unwrap();
    drop(writer);
    let body = read(&path);
    assert_eq!(body, "step,time,kinetic_energy,temperature,nhc_conserved\n");
}

// rq-e49460ac
#[test]
fn write_row_with_extra_columns() {
    let dir = tmp_path("write_row_extras");
    let path = dir.join("run.log");
    let mut writer = LogWriter::open(&path, UnitSystem::Atomic, &[
        ("nhc_conserved", Dimension::Energy),
    ]).unwrap();
    writer.write_row(100, 1.0e-13, 4.0e-21, 298.0, &[5.5e-20]).unwrap();
    writer.flush().unwrap();
    let body = read(&path);
    let last = body.lines().last().unwrap();
    let expected = format!(
        "100,{},{},{},{}",
        format_args!("{:.9e}", 1.0e-13_f64),
        format_args!("{:.9e}", 4.0e-21_f64),
        format_args!("{:.9e}", 298.0_f64),
        format_args!("{:.9e}", 5.5e-20_f64),
    );
    assert_eq!(last, expected);
}

// rq-bcbe7cf6
#[test]
#[should_panic]
fn write_row_panics_on_mismatched_extras_length() {
    let dir = tmp_path("write_row_mismatched_extras");
    let path = dir.join("run.log");
    let mut writer = LogWriter::open(&path, UnitSystem::Atomic, &[
        ("nhc_conserved", Dimension::Energy),
    ]).unwrap();
    // One extra column declared at open, zero extras supplied to write_row:
    // the debug-build assertion is expected to fail.
    writer.write_row(0, 0.0, 0.0, 300.0, &[]).unwrap();
}

// rq-5939f04a
#[test]
fn temperature_for_3atom_settled_water() {
    // N = 3, n_constraints = 3 (one rigid SETTLE'd water molecule);
    // N_thermal_dof = 3 * 3 - 3 - 3 = 3. KE = (3/2) * 9.5e-4
    let n_thermal_dof: u32 = 3;
    let t_target_au = 9.5e-4_f64;
    let ke = 0.5 * (n_thermal_dof as f64) * t_target_au;
    let t = compute_temperature(ke, n_thermal_dof);
    assert!((t - t_target_au).abs() < 1.0e-12);
}
