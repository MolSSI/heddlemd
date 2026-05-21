use std::path::{Path, PathBuf};

use dynamics::io::{TrajectoryWriter, TrajectoryWriterError, load_init_state};
use dynamics::pbc::SimulationBox;

fn tmp_path(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!(
        "dynamics-traj-{}-{}-{}",
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
    SimulationBox::new(1.0, 1.0, 1.0, 0.0, 0.0, 0.0).unwrap()
}

// rq-a403f778
#[test]
fn open_creates_new_file() {
    let dir = tmp_path("open_new");
    let path = dir.join("traj.xyz");
    let _writer = TrajectoryWriter::open(&path, true, false, vec!["Ar".to_string()]).unwrap();
    assert!(path.exists());
}

// rq-8f31cb78
#[test]
fn open_refuses_overwrite() {
    let dir = tmp_path("refuse_overwrite");
    let path = dir.join("traj.xyz");
    std::fs::write(&path, "existing").unwrap();
    match TrajectoryWriter::open(&path, true, false, vec!["Ar".to_string()]).unwrap_err() {
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
    match TrajectoryWriter::open(&path, true, false, vec!["Ar".to_string()]).unwrap_err() {
        TrajectoryWriterError::Io(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-9021ec4b
#[test]
fn write_single_frame_no_velocities() {
    let dir = tmp_path("single_no_velo");
    let path = dir.join("traj.xyz");
    let mut writer = TrajectoryWriter::open(&path, false, false, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(
            0,
            1.0e-15,
            &sim_box(),
            &[0, 0],
            &[0.0_f32, 3.4e-10_f32],
            &[0.0_f32, 0.0_f32],
            &[0.0_f32, 0.0_f32],
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
    let mut writer = TrajectoryWriter::open(&path, true, false, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(
            10,
            1.0e-15,
            &sim_box(),
            &[0],
            &[0.0_f32],
            &[0.0_f32],
            &[0.0_f32],
            Some((&[100.0_f32], &[0.0_f32], &[0.0_f32])),
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
    let mut writer = TrajectoryWriter::open(&path, false, false, vec!["Ar".to_string()]).unwrap();
    for step in [0_u64, 10, 20] {
        writer
            .write_frame(
                step,
                1.0e-15,
                &sim_box(),
                &[0],
                &[0.0_f32],
                &[0.0_f32],
                &[0.0_f32],
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
    let mut writer = TrajectoryWriter::open(&path, false, false, vec!["Ar".to_string()]).unwrap();
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
    let mut writer = TrajectoryWriter::open(
        &path,
        false,
        false,
        vec!["Ar".to_string(), "Kr".to_string()],
    )
    .unwrap();
    writer
        .write_frame(
            0,
            1.0e-15,
            &sim_box(),
            &[0, 1, 1, 0],
            &[0.0_f32; 4],
            &[0.0_f32; 4],
            &[0.0_f32; 4],
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
#[test]
fn round_trip_via_init_parser() {
    let dir = tmp_path("round_trip");
    let path = dir.join("traj.xyz");
    let positions_x = [0.1_f32, -0.2_f32, 0.3_f32, -0.4_f32];
    let positions_y = [0.05_f32, -0.05_f32, 0.15_f32, -0.15_f32];
    let positions_z = [0.0_f32; 4];
    let velocities_x = [1.0_f32, -1.0_f32, 2.0_f32, -2.0_f32];
    let velocities_y = [0.0_f32; 4];
    let velocities_z = [0.0_f32; 4];
    let mut writer = TrajectoryWriter::open(&path, true, false, vec!["Ar".to_string()]).unwrap();
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
    let state = load_init_state(&path, &["Ar"]).unwrap();
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
    let p: f32 = 0.123456_f32;
    let mut writer = TrajectoryWriter::open(&path, false, false, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(
            0,
            1.0e-15,
            &sim_box(),
            &[0],
            &[p],
            &[0.0_f32],
            &[0.0_f32],
            None,
            None,
        )
        .unwrap();
    writer.flush().unwrap();
    let state = load_init_state(&path, &["Ar"]).unwrap();
    assert_eq!(state.positions_x[0].to_bits(), p.to_bits());
}

// rq-e7ddefaf
#[test]
fn flush_is_idempotent() {
    let dir = tmp_path("flush_idempotent");
    let path = dir.join("traj.xyz");
    let mut writer = TrajectoryWriter::open(&path, false, false, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(
            0,
            1.0e-15,
            &sim_box(),
            &[0],
            &[0.0_f32],
            &[0.0_f32],
            &[0.0_f32],
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
        let mut writer = TrajectoryWriter::open(&path, false, false, vec!["Ar".to_string()]).unwrap();
        writer
            .write_frame(
                0,
                1.0e-15,
                &sim_box(),
                &[0],
                &[0.0_f32],
                &[0.0_f32],
                &[0.0_f32],
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
    let mut writer = TrajectoryWriter::open(&path, false, true, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(
            0,
            1.0e-15,
            &sim_box(),
            &[0, 0],
            &[0.0_f32, 0.1_f32],
            &[0.0_f32, 0.0_f32],
            &[0.0_f32, 0.0_f32],
            None,
            Some((&[1_i32, -2], &[0_i32, 3], &[-4_i32, 0])),
        )
        .unwrap();
    writer.flush().unwrap();
    let body = read(&path);
    let lines: Vec<&str> = body.lines().collect();
    assert!(lines[1].contains("Properties=species:S:1:pos:R:3:image:I:3"));
}

// rq-3df3a993
#[test]
fn frame_with_velocities_and_images_carries_both_properties() {
    let dir = tmp_path("velo_images");
    let path = dir.join("traj.xyz");
    let mut writer = TrajectoryWriter::open(&path, true, true, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(
            0,
            1.0e-15,
            &sim_box(),
            &[0],
            &[0.1_f32],
            &[0.2_f32],
            &[0.3_f32],
            Some((&[1.0_f32], &[2.0_f32], &[3.0_f32])),
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
    let mut writer = TrajectoryWriter::open(&path, true, true, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(
            0,
            1.0e-15,
            &sim_box(),
            &[0; 4],
            &[0.0_f32; 4],
            &[0.0_f32; 4],
            &[0.0_f32; 4],
            Some((&[1.0_f32; 4], &[0.0_f32; 4], &[0.0_f32; 4])),
            Some((&images_x, &images_y, &images_z)),
        )
        .unwrap();
    writer.flush().unwrap();
    let state = load_init_state(&path, &["Ar"]).unwrap();
    assert_eq!(state.particle_count, 4);
    let imgs = state.images.unwrap();
    assert_eq!(imgs.images_x, images_x);
    assert_eq!(imgs.images_y, images_y);
    assert_eq!(imgs.images_z, images_z);
}

#[test] // rq-ce15e04e
fn round_trip_preserves_triclinic_lattice() {
    let dir = tmp_path("triclinic_roundtrip");
    let path = dir.join("traj.xyz");
    let tri = SimulationBox::new(1.0, 1.0, 1.0, 0.2, 0.1, -0.3).unwrap();
    let mut writer =
        TrajectoryWriter::open(&path, false, false, vec!["Ar".to_string()]).unwrap();
    writer
        .write_frame(
            0,
            1.0e-15,
            &tri,
            &[0; 0],
            &[0.0_f32; 0],
            &[0.0_f32; 0],
            &[0.0_f32; 0],
            None,
            None,
        )
        .unwrap();
    writer.flush().unwrap();
    let state = load_init_state(&path, &["Ar"]).unwrap();
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
