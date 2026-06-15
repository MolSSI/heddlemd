// Verifies that the trajectory writer formats Real columns with the
// build-time precision width: 9 fractional digits in the default
// (f32) build and 17 in the f64 build.
//
// rq-default_build_trajectory_writer_emits_9_digit_columns
// rq-f64_build_trajectory_writer_emits_17_digit_columns

use std::fs;
use std::io::Read;

use heddle_md::io::trajectory::TrajectoryWriter;
use heddle_md::pbc::SimulationBox;
use heddle_md::precision::{REAL_FMT_DIGITS, Real};
use heddle_md::units::UnitSystem;

fn make_tempdir() -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("precision_traj_{pid}_{nanos}"));
    fs::create_dir_all(&p).expect("create tempdir");
    p
}

fn write_one_frame(path: &std::path::Path, positions: Vec<Real>) -> String {
    let sim_box = SimulationBox::new(10.0, 10.0, 10.0, 0.0, 0.0, 0.0).expect("box");
    let type_names = vec!["Ar".to_string()];
    let mut writer = TrajectoryWriter::open(
        path,
        UnitSystem::Atomic,
        false,
        false,
        type_names,
    )
    .expect("open");
    let n = positions.len();
    let zeros: Vec<Real> = vec![0.0; n];
    let type_indices: Vec<u32> = vec![0u32; n];
    writer
        .write_frame(
            0,
            1.0e-15,
            &sim_box,
            &type_indices,
            &positions,
            &zeros,
            &zeros,
            None,
            None,
        )
        .expect("write_frame");
    writer.flush().expect("flush");
    drop(writer);
    let mut text = String::new();
    fs::File::open(path)
        .unwrap()
        .read_to_string(&mut text)
        .unwrap();
    text
}

#[test]
fn trajectory_real_columns_use_real_fmt_digits() {
    let dir = make_tempdir();
    let path = dir.join("traj.xyz");
    let positions: Vec<Real> = vec![3.4e-10 as Real];
    let text = write_one_frame(&path, positions);

    // Find the first data row (line 3: line 0 = N, line 1 = comment, lines >=2 are data).
    let line = text.lines().nth(2).expect("data row");
    // Format pattern check: "Ar <num> 0...e0 0...e0"
    let cols: Vec<&str> = line.split_whitespace().collect();
    assert!(cols.len() >= 4, "data row has at least 4 columns: {line}");

    // The first number column carries the position. Verify the exponent
    // format is "<digit>.<REAL_FMT_DIGITS digits>e<exponent>".
    let num = cols[1];
    let mantissa = num.split('e').next().unwrap();
    let frac = mantissa.split('.').nth(1).expect("decimal fraction");
    assert_eq!(
        frac.len(),
        REAL_FMT_DIGITS,
        "fractional-digit width in trajectory column: got {} digits in {num:?}, expected {} (REAL_FMT_DIGITS)",
        frac.len(),
        REAL_FMT_DIGITS,
    );

    // Cleanup
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn trajectory_lattice_uses_real_fmt_digits() {
    let dir = make_tempdir();
    let path = dir.join("traj.xyz");
    let positions: Vec<Real> = vec![0.0 as Real];
    let text = write_one_frame(&path, positions);
    // The comment line carries Lattice="...".
    let comment = text.lines().nth(1).expect("comment line");
    let start = comment.find("Lattice=\"").expect("Lattice attr") + "Lattice=\"".len();
    let rest = &comment[start..];
    let end = rest.find('"').expect("close-quote");
    let lattice = &rest[..end];
    // The first lattice number should follow REAL_FMT_DIGITS too.
    let first = lattice.split_whitespace().next().expect("at least one number");
    let mantissa = first.split('e').next().unwrap();
    let frac = mantissa.split('.').nth(1).expect("decimal fraction");
    assert_eq!(frac.len(), REAL_FMT_DIGITS);

    let _ = fs::remove_dir_all(&dir);
}
