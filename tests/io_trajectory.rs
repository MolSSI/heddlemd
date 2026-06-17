use std::path::{Path, PathBuf};

use heddle_md::gpu::init_device;
use heddle_md::io::{TrajectoryWriter, TrajectoryWriterError, load_init_state};
use heddle_md::pbc::SimulationBox;
use heddle_md::units::UnitSystem;
use heddle_md::precision::Real;

fn dev() -> std::sync::Arc<cudarc::driver::CudaDevice> {
    init_device().expect("init_device").device
}

fn tmp_path(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!(
        "heddlemd-traj-{}-{}-{}",
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

fn sim_box() -> SimulationBox {
    SimulationBox::new(&dev(), 1.0, 1.0, 1.0, 0.0, 0.0, 0.0).unwrap()
}

// rq-a403f778
#[test]
fn open_creates_new_file() {
    let dir = tmp_path("open_new");
    let path = dir.join("traj.xyz");
    let _writer = TrajectoryWriter::open(&path, UnitSystem::Atomic, true, false, vec!["Ar".to_string()]).unwrap();
    assert!(path.exists());
}

// rq-8f31cb78
#[test]
fn open_refuses_overwrite() {
    let dir = tmp_path("refuse_overwrite");
    let path = dir.join("traj.xyz");
    std::fs::write(&path, "existing").unwrap();
    match TrajectoryWriter::open(&path, UnitSystem::Atomic, true, false, vec!["Ar".to_string()]).unwrap_err() {
        TrajectoryWriterError::OutputExists { path: p } => assert_eq!(p, path),
        other => panic!("unexpected: {other:?}"),
    }
    assert_eq!(read(&path), "existing");
}

// rq-17666e4f
#[test]
fn open_fails_when_parent_missing() {
    let dir = tmp_path("missing_parent");
    let path = dir.join("missing-dir").join("traj.xyz");
    match TrajectoryWriter::open(&path, UnitSystem::Atomic, true, false, vec!["Ar".to_string()]).unwrap_err() {
        TrajectoryWriterError::Io(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-9021ec4b
#[test]
fn write_single_frame_no_velocities() {
    let dir = tmp_path("single_no_velo");
    let path = dir.join("traj.xyz");
    let mut writer = TrajectoryWriter::open(&path, UnitSystem::Atomic, false, false, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(
            0,
            1.0e-15,
            &sim_box(),
            &[0, 0],
            &[0.0, 3.4e-10],
            &[0.0, 0.0],
            &[0.0, 0.0],
            None,
            None,
        )
        .unwrap();
    writer.flush().unwrap();
    let body = read(&path);
    let lines: Vec<&str> = body.lines().collect();
    assert_eq!(lines.len(), 4);
    assert_eq!(lines[0], "2");
    assert!(lines[1].contains("Properties=species:S:1:pos:R:3"));
    assert!(lines[1].contains("Step=0"));
    assert!(lines[1].contains("Time=0.000000000e0"));
    assert!(lines[1].starts_with("Lattice=\""));
    assert!(lines[2].starts_with("Ar "));
    assert!(lines[3].starts_with("Ar "));
}

// rq-c5e00a28
#[test]
fn write_single_frame_with_velocities() {
    let dir = tmp_path("single_with_velo");
    let path = dir.join("traj.xyz");
    let mut writer = TrajectoryWriter::open(&path, UnitSystem::Atomic, true, false, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(
            10,
            1.0e-15,
            &sim_box(),
            &[0],
            &[0.0],
            &[0.0],
            &[0.0],
            Some((&[100.0], &[0.0], &[0.0])),
            None,
        )
        .unwrap();
    writer.flush().unwrap();
    let body = read(&path);
    let lines: Vec<&str> = body.lines().collect();
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0], "1");
    assert!(lines[1].contains("Properties=species:S:1:pos:R:3:velo:R:3"));
    assert!(lines[1].contains("Step=10"));
    let expected_time = format!("{:.9e}", 10.0 * 1.0e-15_f64);
    assert!(lines[1].contains(&format!("Time={expected_time}")));
    // Data row: "Ar  px py pz vx vy vz" — 7 columns
    let cols: Vec<&str> = lines[2].split_ascii_whitespace().collect();
    assert_eq!(cols.len(), 7);
    assert_eq!(cols[0], "Ar");
}

// rq-fd593357
#[test]
fn append_frames_in_order() {
    let dir = tmp_path("append_frames");
    let path = dir.join("traj.xyz");
    let mut writer = TrajectoryWriter::open(&path, UnitSystem::Atomic, false, false, vec!["Ar".to_string()]).unwrap();
    for step in [0_u64, 10, 20] {
        writer
            .write_frame(
                step,
                1.0e-15,
                &sim_box(),
                &[0],
                &[0.0],
                &[0.0],
                &[0.0],
                None,
                None,
            )
            .unwrap();
    }
    writer.flush().unwrap();
    let body = read(&path);
    let lines: Vec<&str> = body.lines().collect();
    assert_eq!(lines.len(), 9);
    assert!(lines[1].contains("Step=0"));
    assert!(lines[4].contains("Step=10"));
    assert!(lines[7].contains("Step=20"));
}

// rq-f5e94e6b
#[test]
fn write_empty_frame() {
    let dir = tmp_path("empty_frame");
    let path = dir.join("traj.xyz");
    let mut writer = TrajectoryWriter::open(&path, UnitSystem::Atomic, false, false, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(0, 1.0e-15, &sim_box(), &[], &[], &[], &[], None, None)
        .unwrap();
    writer.flush().unwrap();
    let body = read(&path);
    let lines: Vec<&str> = body.lines().collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0], "0");
}

// rq-f76b6cde
#[test]
fn render_multiple_type_names() {
    let dir = tmp_path("multi_names");
    let path = dir.join("traj.xyz");
    let mut writer = TrajectoryWriter::open(&path, UnitSystem::Atomic, false, false, vec!["Ar".to_string(), "Kr".to_string()], )
    .unwrap();
    writer
        .write_frame(
            0,
            1.0e-15,
            &sim_box(),
            &[0, 1, 1, 0],
            &[0.0; 4],
            &[0.0; 4],
            &[0.0; 4],
            None,
            None,
        )
        .unwrap();
    writer.flush().unwrap();
    let body = read(&path);
    let data_rows: Vec<&str> = body.lines().skip(2).collect();
    assert_eq!(data_rows.len(), 4);
    let species: Vec<&str> = data_rows
        .iter()
        .map(|r| r.split_whitespace().next().unwrap())
        .collect();
    assert_eq!(species, vec!["Ar", "Kr", "Kr", "Ar"]);
}

// rq-70e9fd38
// rq-f9714477 rq-94f9228c rq-ee50492a
// rq-afde087b rq-f61f5d24 rq-6d9f6db6
#[test]
fn si_mode_writer_multiplies_positions_lattice_and_velocities_by_factors() {
    // Open the trajectory writer with UnitSystem::Si. The engine state
    // is in atomic units; the writer must multiply positions and
    // lattice components by the bohr→meter factor and velocities by
    // the (Bohr/atu)→(m/s) factor before formatting.
    let dir = tmp_path("si_writer_factors");
    let path = dir.join("traj.xyz");

    // Engine-side (atomic) values.
    let l_au: Real = 10.0;
    let pos_x_au: Real = 3.7;
    let pos_y_au: Real = -2.1;
    let pos_z_au: Real = 0.5;
    let vx_au: Real = 0.04;
    let vy_au: Real = -0.02;
    let vz_au: Real = 0.01;

    let length_factor = UnitSystem::Si.factor(heddle_md::units::Dimension::Length) as Real;
    let velocity_factor = UnitSystem::Si.factor(heddle_md::units::Dimension::Velocity) as Real;

    let sim_box = SimulationBox::new(&dev(), l_au, l_au, l_au, 0.0, 0.0, 0.0).unwrap();
    let mut writer = TrajectoryWriter::open(
        &path,
        UnitSystem::Si,
        true,  // include_velocities
        false, // include_images
        vec!["Ar".to_string()],
    )
    .unwrap();
    writer
        .write_frame(
            0,
            1.0,
            &sim_box,
            &[0],
            &[pos_x_au],
            &[pos_y_au],
            &[pos_z_au],
            Some((&[vx_au], &[vy_au], &[vz_au])),
            None,
        )
        .unwrap();
    writer.flush().unwrap();

    let body = read(&path);
    let lines: Vec<&str> = body.lines().collect();
    let header = lines[1];

    // --- Lattice: header has Lattice="lx 0 0 0 ly 0 0 0 lz" with
    //              each component in metres. ---
    let lat_start = header.find("Lattice=\"").unwrap() + "Lattice=\"".len();
    let lat_end = lat_start + header[lat_start..].find('"').unwrap();
    let lat_values: Vec<Real> = header[lat_start..lat_end]
        .split_ascii_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(lat_values.len(), 9);
    let rel = 1e-5;
    let approx = |a: Real, b: Real| {
        (a - b).abs() <= rel * a.abs().max(b.abs()).max(Real::MIN_POSITIVE)
    };
    assert!(
        approx(lat_values[0], l_au * length_factor),
        "Lattice[0,0] {} != l_au * length_factor {}",
        lat_values[0],
        l_au * length_factor
    );
    assert!(approx(lat_values[4], l_au * length_factor));
    assert!(approx(lat_values[8], l_au * length_factor));

    // --- Particle row: species pos_x pos_y pos_z velo_x velo_y velo_z ---
    let cols: Vec<&str> = lines[2].split_ascii_whitespace().collect();
    assert!(cols.len() >= 7, "expected 7+ columns, got {}: {:?}", cols.len(), cols);
    assert_eq!(cols[0], "Ar");
    let px: Real = cols[1].parse().unwrap();
    let py: Real = cols[2].parse().unwrap();
    let pz: Real = cols[3].parse().unwrap();
    let vx: Real = cols[4].parse().unwrap();
    let vy: Real = cols[5].parse().unwrap();
    let vz: Real = cols[6].parse().unwrap();
    assert!(approx(px, pos_x_au * length_factor), "px {} != pos_x_au * length_factor {}", px, pos_x_au * length_factor);
    assert!(approx(py, pos_y_au * length_factor));
    assert!(approx(pz, pos_z_au * length_factor));
    assert!(approx(vx, vx_au * velocity_factor), "vx {} != vx_au * velocity_factor {}", vx, vx_au * velocity_factor);
    assert!(approx(vy, vy_au * velocity_factor));
    assert!(approx(vz, vz_au * velocity_factor));
}

// rq-7edb9c67 rq-64e90d53
#[test]
fn round_trip_via_init_parser() {
    let dir = tmp_path("round_trip");
    let path = dir.join("traj.xyz");
    let positions_x = [0.1, -0.2, 0.3, -0.4];
    let positions_y = [0.05, -0.05, 0.15, -0.15];
    let positions_z = [0.0; 4];
    let velocities_x = [1.0, -1.0, 2.0, -2.0];
    let velocities_y = [0.0; 4];
    let velocities_z = [0.0; 4];
    let mut writer = TrajectoryWriter::open(&path, UnitSystem::Atomic, true, false, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(
            0,
            1.0e-15,
            &sim_box(),
            &[0; 4],
            &positions_x,
            &positions_y,
            &positions_z,
            Some((&velocities_x, &velocities_y, &velocities_z)),
            None,
        )
        .unwrap();
    writer.flush().unwrap();
    // Reader uses the same UnitSystem the writer was opened with so the
    // numerical round-trip is the identity.
    let state = load_init_state(&dev(), &path, &["Ar"], UnitSystem::Atomic).unwrap();
    assert_eq!(state.particle_count, 4);
    assert_eq!(state.positions_x.as_slice(), &positions_x);
    assert_eq!(state.positions_y.as_slice(), &positions_y);
    assert_eq!(state.positions_z.as_slice(), &positions_z);
    let v = state.velocities.unwrap();
    assert_eq!(v.velocities_x.as_slice(), &velocities_x);
    assert_eq!(v.velocities_y.as_slice(), &velocities_y);
    assert_eq!(v.velocities_z.as_slice(), &velocities_z);
}

// rq-ddec3d72
#[test]
fn f32_position_round_trip() {
    let dir = tmp_path("f32_round_trip");
    let path = dir.join("traj.xyz");
    // Arbitrary f32 value that's not exactly representable in decimal.
    let p: Real = 0.123456;
    let mut writer = TrajectoryWriter::open(&path, UnitSystem::Atomic, false, false, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(
            0,
            1.0e-15,
            &sim_box(),
            &[0],
            &[p],
            &[0.0],
            &[0.0],
            None,
            None,
        )
        .unwrap();
    writer.flush().unwrap();
    let state = load_init_state(&dev(), &path, &["Ar"], UnitSystem::Atomic).unwrap();
    assert_eq!(state.positions_x[0].to_bits(), p.to_bits());
}

// rq-e7ddefaf
#[test]
fn flush_is_idempotent() {
    let dir = tmp_path("flush_idempotent");
    let path = dir.join("traj.xyz");
    let mut writer = TrajectoryWriter::open(&path, UnitSystem::Atomic, false, false, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(
            0,
            1.0e-15,
            &sim_box(),
            &[0],
            &[0.0],
            &[0.0],
            &[0.0],
            None,
            None,
        )
        .unwrap();
    writer.flush().unwrap();
    writer.flush().unwrap();
}

// rq-03ff6434
#[test]
fn drop_flushes_best_effort() {
    let dir = tmp_path("drop_flush");
    let path = dir.join("traj.xyz");
    {
        let mut writer = TrajectoryWriter::open(&path, UnitSystem::Atomic, false, false, vec!["Ar".to_string()]).unwrap();
        writer
            .write_frame(
                0,
                1.0e-15,
                &sim_box(),
                &[0],
                &[0.0],
                &[0.0],
                &[0.0],
                None,
                None,
            )
            .unwrap();
        // Drop without explicit flush.
    }
    let body = read(&path);
    let lines: Vec<&str> = body.lines().collect();
    assert!(lines.len() >= 3);
    assert_eq!(lines[0], "1");
}

// --- Image columns ---

// rq-b8463a3b
#[test]
fn frame_with_images_only_carries_image_property() {
    let dir = tmp_path("images_only");
    let path = dir.join("traj.xyz");
    let mut writer = TrajectoryWriter::open(&path, UnitSystem::Atomic, false, true, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(
            0,
            1.0e-15,
            &sim_box(),
            &[0, 0],
            &[0.0, 0.1],
            &[0.0, 0.0],
            &[0.0, 0.0],
            None,
            Some((&[1_i32, -2], &[0_i32, 3], &[-4_i32, 0])),
        )
        .unwrap();
    writer.flush().unwrap();
    let body = read(&path);
    let lines: Vec<&str> = body.lines().collect();
    assert!(lines[1].contains("Properties=species:S:1:pos:R:3:image:I:3"));
}

// rq-3df3a993 rq-5395785a
#[test]
fn frame_with_velocities_and_images_carries_both_properties() {
    let dir = tmp_path("velo_images");
    let path = dir.join("traj.xyz");
    let mut writer = TrajectoryWriter::open(&path, UnitSystem::Atomic, true, true, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(
            0,
            1.0e-15,
            &sim_box(),
            &[0],
            &[0.1],
            &[0.2],
            &[0.3],
            Some((&[1.0], &[2.0], &[3.0])),
            Some((&[4_i32], &[-5_i32], &[6_i32])),
        )
        .unwrap();
    writer.flush().unwrap();
    let body = read(&path);
    let lines: Vec<&str> = body.lines().collect();
    assert!(lines[1].contains("Properties=species:S:1:pos:R:3:velo:R:3:image:I:3"));
    // Data row: "Ar  px py pz vx vy vz  nx ny nz" → 10 columns.
    let cols: Vec<&str> = lines[2].split_ascii_whitespace().collect();
    assert_eq!(cols.len(), 10);
    assert_eq!(cols[7], "4");
    assert_eq!(cols[8], "-5");
    assert_eq!(cols[9], "6");
}

// rq-7e6a503c
#[test]
fn image_round_trip_via_init_parser() {
    let dir = tmp_path("images_round_trip");
    let path = dir.join("traj.xyz");
    let images_x = vec![1_i32, -2, 3, 0];
    let images_y = vec![0_i32, 4, -5, 6];
    let images_z = vec![-7_i32, 0, 8, -9];
    let mut writer = TrajectoryWriter::open(&path, UnitSystem::Atomic, true, true, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(
            0,
            1.0e-15,
            &sim_box(),
            &[0; 4],
            &[0.0; 4],
            &[0.0; 4],
            &[0.0; 4],
            Some((&[1.0; 4], &[0.0; 4], &[0.0; 4])),
            Some((&images_x, &images_y, &images_z)),
        )
        .unwrap();
    writer.flush().unwrap();
    let state = load_init_state(&dev(), &path, &["Ar"], UnitSystem::Si).unwrap();
    assert_eq!(state.particle_count, 4);
    let imgs = state.images.unwrap();
    assert_eq!(imgs.images_x, images_x);
    assert_eq!(imgs.images_y, images_y);
    assert_eq!(imgs.images_z, images_z);
}

#[test] // rq-ce15e04e rq-ef891f9c
fn round_trip_preserves_triclinic_lattice() {
    let dir = tmp_path("triclinic_roundtrip");
    let path = dir.join("traj.xyz");
    let tri = SimulationBox::new(&dev(), 1.0, 1.0, 1.0, 0.2, 0.1, -0.3).unwrap();
    let mut writer =
        TrajectoryWriter::open(&path, UnitSystem::Atomic, false, false, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(
            0,
            1.0e-15,
            &tri,
            &[0; 0],
            &[0.0; 0],
            &[0.0; 0],
            &[0.0; 0],
            None,
            None,
        )
        .unwrap();
    writer.flush().unwrap();
    let state = load_init_state(&dev(), &path, &["Ar"], UnitSystem::Atomic).unwrap();
    let parsed = state.sim_box.lattice();
    let original = tri.lattice();
    for d in 0..6 {
        assert!(
            (parsed[d] - original[d]).abs() < 1.0e-9,
            "lattice[{d}] parsed {} vs original {}",
            parsed[d],
            original[d]
        );
    }
}

// rq-a1d80d47
#[test]
fn si_mode_writer_multiplies_time_by_atomic_time_factor() {
    let dir = tmp_path("si_writer_time_factor");
    let path = dir.join("traj.xyz");
    let dt_au: f64 = 41.3;
    let mut writer =
        TrajectoryWriter::open(&path, UnitSystem::Si, false, false, vec!["Ar".to_string()])
            .unwrap();
    writer
        .write_frame(
            1,
            dt_au,
            &sim_box(),
            &[0],
            &[0.0],
            &[0.0],
            &[0.0],
            None,
            None,
        )
        .unwrap();
    writer.flush().unwrap();
    let body = read(&path);
    let header = body.lines().nth(1).unwrap();
    let needle = "Time=";
    let tstart = header.find(needle).unwrap() + needle.len();
    let trest = &header[tstart..];
    let tend = trest.find(' ').unwrap_or(trest.len());
    let time_written: f64 = trest[..tend].parse().unwrap();
    let factor = UnitSystem::Si.factor(heddle_md::units::Dimension::Time);
    let expected = dt_au * factor;
    // The header formats Time with `{:.9e}`, so the read-back value
    // matches the unrounded product only to ~1e-9 relative precision.
    let rel = 1e-8_f64;
    assert!(
        (time_written - expected).abs() <= rel * expected.abs(),
        "Time written {} != expected {} (factor {})",
        time_written,
        expected,
        factor
    );
}

// rq-c19d0ce4
#[test]
fn si_mode_reader_divides_lattice_and_positions_by_length_factor() {
    use heddle_md::io::TrajectoryReader;
    let dir = tmp_path("si_reader_divides");
    let path = dir.join("traj.xyz");
    let l_au: Real = 10.0;
    let pos_au: [Real; 3] = [3.7, -2.1, 0.5];
    let sim_box = SimulationBox::new(&dev(), l_au, l_au, l_au, 0.0, 0.0, 0.0).unwrap();
    let mut writer =
        TrajectoryWriter::open(&path, UnitSystem::Si, false, false, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(
            0,
            1.0,
            &sim_box,
            &[0],
            &[pos_au[0]],
            &[pos_au[1]],
            &[pos_au[2]],
            None,
            None,
        )
        .unwrap();
    writer.flush().unwrap();

    // Read the same file back with the matching unit system.
    let mut reader = TrajectoryReader::open(&dev(), &path, UnitSystem::Si, &["Ar"]).unwrap();
    let frame = reader.next_frame().unwrap().expect("at least one frame");
    let rel = 1e-5;
    let approx = |a: Real, b: Real| {
        (a - b).abs() <= rel * a.abs().max(b.abs()).max(Real::MIN_POSITIVE)
    };
    assert!(approx(frame.sim_box.lx(), l_au));
    assert!(approx(frame.sim_box.ly(), l_au));
    assert!(approx(frame.sim_box.lz(), l_au));
    assert!(approx(frame.positions_x[0], pos_au[0]));
    assert!(approx(frame.positions_y[0], pos_au[1]));
    assert!(approx(frame.positions_z[0], pos_au[2]));
}

// rq-48d14580
#[test]
fn writer_emits_positions_inside_primary_cell() {
    let dir = tmp_path("positions_in_primary_cell");
    let path = dir.join("traj.xyz");
    let l_au: Real = 4.0;
    let sim_box = SimulationBox::new(&dev(), l_au, l_au, l_au, 0.0, 0.0, 0.0).unwrap();
    // Caller's invariant: every position lies in [-L/2, L/2). The writer
    // is expected to emit exactly these values; the reader (atomic
    // round-trip) must read back values that still satisfy the bound.
    let positions_x = [-1.999, 0.0, 1.0];
    let positions_y = [-1.0, 0.5, 1.999];
    let positions_z = [0.0, -1.5, 1.5];
    let mut writer = TrajectoryWriter::open(
        &path,
        UnitSystem::Atomic,
        false,
        true,
        vec!["Ar".to_string()],
    )
    .unwrap();
    writer
        .write_frame(
            0,
            1.0e-15,
            &sim_box,
            &[0, 0, 0],
            &positions_x,
            &positions_y,
            &positions_z,
            None,
            Some((&[0_i32, 0, 0], &[0_i32, 0, 0], &[0_i32, 0, 0])),
        )
        .unwrap();
    writer.flush().unwrap();
    let state = load_init_state(&dev(), &path, &["Ar"], UnitSystem::Atomic).unwrap();
    let half = l_au / 2.0;
    for (i, (&x, (&y, &z))) in state
        .positions_x
        .iter()
        .zip(state.positions_y.iter().zip(state.positions_z.iter()))
        .enumerate()
    {
        assert!(x >= -half && x < half, "x[{i}] = {x} not in [-{half}, {half})");
        assert!(y >= -half && y < half, "y[{i}] = {y} not in [-{half}, {half})");
        assert!(z >= -half && z < half, "z[{i}] = {z} not in [-{half}, {half})");
    }
}
