use std::path::{Path, PathBuf};

use dynamics::io::{InitStateError, load_init_state};
use dynamics::units::UnitSystem;

fn tmp_path(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!(
        "dynamics-init-{}-{}-{}",
        std::process::id(),
        name,
        nanos
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn write_init(dir: &Path, contents: &str) -> PathBuf {
    let path = dir.join("init.xyz");
    std::fs::write(&path, contents).unwrap();
    path
}

// rq-38ebe278
#[test]
fn load_two_particles_no_velocities() {
    let dir = tmp_path("two_no_velo");
    let body = "2\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\nAr 0.0 0.0 0.0\nAr 3.4e-10 0.0 0.0\n";
    let path = write_init(&dir, body);
    let state = load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap();
    assert_eq!(state.particle_count, 2);
    assert!((state.sim_box.lx() - 1.0e-9_f32).abs() < 1.0e-18);
    assert!((state.sim_box.ly() - 1.0e-9_f32).abs() < 1.0e-18);
    assert!((state.sim_box.lz() - 1.0e-9_f32).abs() < 1.0e-18);
    assert_eq!(state.type_indices, vec![0, 0]);
    assert!((state.positions_x[0] - 0.0_f32).abs() < 1e-30);
    assert!((state.positions_x[1] - 3.4e-10_f32).abs() < 1e-20);
    assert!(state.velocities.is_none());
}

// rq-bb807252
#[test]
fn load_with_velocities() {
    let dir = tmp_path("with_velo");
    let body = "2\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3:velo:R:3\nAr 0.0 0.0 0.0 100.0 0.0 0.0\nAr 3.4e-10 0.0 0.0 -100.0 0.0 0.0\n";
    let path = write_init(&dir, body);
    let state = load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap();
    let v = state.velocities.unwrap();
    assert!((v.velocities_x[0] - 100.0_f32).abs() < 1e-3);
    assert!((v.velocities_x[1] - (-100.0_f32)).abs() < 1e-3);
}

// rq-32fda118
#[test]
fn load_empty_file() {
    let dir = tmp_path("empty");
    let body = "0\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\n";
    let path = write_init(&dir, body);
    let state = load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap();
    assert_eq!(state.particle_count, 0);
    assert!(state.positions_x.is_empty());
    assert!(state.positions_y.is_empty());
    assert!(state.positions_z.is_empty());
    assert!(state.type_indices.is_empty());
    assert!(state.velocities.is_none());
}

// rq-9e5d6525
#[test]
fn type_indices_reflect_ordering() {
    let dir = tmp_path("type_indices_ordering");
    let body = "2\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\nKr 0.0 0.0 0.0\nAr 0.0 0.0 0.0\n";
    let path = write_init(&dir, body);
    let state = load_init_state(&path, &["Ar", "Kr"], UnitSystem::Si).unwrap();
    assert_eq!(state.type_indices, vec![1, 0]);
}

// rq-93be622c
#[test]
fn unknown_attributes_ignored() {
    let dir = tmp_path("unknown_attrs");
    let body = "1\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Time=0.0 Comment=\"hello world\" Properties=species:S:1:pos:R:3\nAr 0.0 0.0 0.0\n";
    let path = write_init(&dir, body);
    load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap();
}

// rq-66233215
#[test]
fn quoted_attributes_with_spaces() {
    let dir = tmp_path("quoted_attrs");
    let body = "1\nOrigin=\"0.0 0.0 0.0\" Lattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\nAr 0.0 0.0 0.0\n";
    let path = write_init(&dir, body);
    load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap();
}

// rq-dad92a8c
#[test]
fn reject_empty_file() {
    let dir = tmp_path("reject_empty");
    let path = write_init(&dir, "");
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::Empty => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-6575e627
#[test]
fn reject_non_integer_count() {
    let dir = tmp_path("non_integer_count");
    let path = write_init(&dir, "abc\nLattice=\"1.0 0 0 0 1.0 0 0 0 1.0\" Properties=species:S:1:pos:R:3\n");
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::InvalidParticleCount { line_number, raw } => {
            assert_eq!(line_number, 1);
            assert_eq!(raw, "abc");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-7306fc6f
#[test]
fn reject_negative_count() {
    let dir = tmp_path("neg_count");
    let path = write_init(&dir, "-3\nLattice=\"1.0 0 0 0 1.0 0 0 0 1.0\" Properties=species:S:1:pos:R:3\n");
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::InvalidParticleCount { line_number, raw } => {
            assert_eq!(line_number, 1);
            assert_eq!(raw, "-3");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-4994ba90
#[test]
fn reject_missing_comment_line() {
    let dir = tmp_path("missing_comment");
    let path = write_init(&dir, "2");
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::MissingCommentLine => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-fc6700b5
#[test]
fn reject_missing_lattice() {
    let dir = tmp_path("missing_lattice");
    let path = write_init(&dir, "0\nProperties=species:S:1:pos:R:3\n");
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::MissingAttribute { name } => assert_eq!(name, "Lattice"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-9f596df3
#[test]
fn reject_missing_properties() {
    let dir = tmp_path("missing_properties");
    let path = write_init(
        &dir,
        "0\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\"\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::MissingAttribute { name } => assert_eq!(name, "Properties"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-2ed137d9
// rq-2ed137d9
#[test]
fn reject_lattice_with_non_zero_upper_triangular_entry() {
    // a_y (slot 1 in row-major) must be exactly 0 for lower-triangular form.
    let dir = tmp_path("non_lower_tri");
    let path = write_init(
        &dir,
        "0\nLattice=\"1.0e-9 0.1e-9 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::InvalidLattice(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-57173f96
#[test]
fn reject_nonpositive_lattice_diagonal() {
    let dir = tmp_path("nonpositive_diag");
    let path = write_init(
        &dir,
        "0\nLattice=\"1.0e-9 0 0 0 0.0 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::InvalidLattice(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-e8d29f08
#[test]
fn reject_nonfinite_lattice() {
    let dir = tmp_path("nonfinite_lattice");
    let path = write_init(
        &dir,
        "0\nLattice=\"1.0e-9 0 0 0 nan 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::InvalidLattice(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-00963c5d
#[test]
fn reject_lattice_wrong_components() {
    let dir = tmp_path("bad_lattice_count");
    let path = write_init(
        &dir,
        "0\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0\" Properties=species:S:1:pos:R:3\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::InvalidLattice(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-8373db52
#[test]
fn reject_unsupported_properties() {
    let dir = tmp_path("unsupported_props");
    let path = write_init(
        &dir,
        "0\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3:mass:R:1\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::InvalidProperties(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-680ab854
#[test]
fn reject_reordered_properties() {
    let dir = tmp_path("reordered_props");
    let path = write_init(
        &dir,
        "0\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=pos:R:3:species:S:1\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::InvalidProperties(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-44ccc1cb
#[test]
fn reject_too_few_rows() {
    let dir = tmp_path("too_few_rows");
    let path = write_init(
        &dir,
        "3\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\nAr 0 0 0\nAr 1e-10 0 0\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::RowCountMismatch { expected, actual } => {
            assert_eq!(expected, 3);
            assert_eq!(actual, 2);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-c88dde28
#[test]
fn reject_too_many_rows() {
    let dir = tmp_path("too_many_rows");
    let path = write_init(
        &dir,
        "2\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\nAr 0 0 0\nAr 1e-10 0 0\nAr 2e-10 0 0\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::TrailingContent { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-bafc5900
#[test]
fn reject_missing_velocity_column() {
    let dir = tmp_path("missing_velo_col");
    let path = write_init(
        &dir,
        "1\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3:velo:R:3\nAr 0 0 0 0 0\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::RowColumnCountMismatch {
            line_number,
            expected,
            actual,
        } => {
            assert_eq!(line_number, 3);
            assert_eq!(expected, 7);
            assert_eq!(actual, 6);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-f357fe5a
#[test]
fn reject_extra_column() {
    let dir = tmp_path("extra_col");
    let path = write_init(
        &dir,
        "1\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\nAr 0 0 0 99\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::RowColumnCountMismatch {
            line_number,
            expected,
            actual,
        } => {
            assert_eq!(line_number, 3);
            assert_eq!(expected, 4);
            assert_eq!(actual, 5);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-15647873
#[test]
fn reject_unknown_species() {
    let dir = tmp_path("unknown_species");
    let path = write_init(
        &dir,
        "1\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\nXe 0 0 0\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::UnknownType { line_number, name } => {
            assert_eq!(line_number, 3);
            assert_eq!(name, "Xe");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-ca6f69f2
#[test]
fn reject_unparseable_position() {
    let dir = tmp_path("bad_pos");
    let path = write_init(
        &dir,
        "1\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\nAr abc 0 0\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::InvalidNumber {
            line_number,
            column,
            raw,
        } => {
            assert_eq!(line_number, 3);
            assert_eq!(column, "pos_x");
            assert_eq!(raw, "abc");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-7af67a7b
#[test]
fn reject_nan_position() {
    let dir = tmp_path("nan_pos");
    let path = write_init(
        &dir,
        "1\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\nAr nan 0 0\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::NonFiniteValue {
            line_number,
            column,
        } => {
            assert_eq!(line_number, 3);
            assert_eq!(column, "pos_x");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-8057cb7e
#[test]
fn reject_infinite_velocity() {
    let dir = tmp_path("inf_velo");
    let path = write_init(
        &dir,
        "1\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3:velo:R:3\nAr 0 0 0 inf 0 0\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::NonFiniteValue {
            line_number,
            column,
        } => {
            assert_eq!(line_number, 3);
            assert_eq!(column, "velo_x");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-685006bb
#[test]
fn accept_strictly_inside() {
    let dir = tmp_path("inside_box");
    let path = write_init(
        &dir,
        "1\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\nAr 4.999e-10 0 0\n",
    );
    load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap();
}

// rq-a35ca20b
#[test]
fn accept_lower_boundary() {
    let dir = tmp_path("lower_boundary");
    let path = write_init(
        &dir,
        "1\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\nAr -5.0e-10 0 0\n",
    );
    load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap();
}

// rq-7a4ed012
#[test]
fn reject_upper_boundary() {
    let dir = tmp_path("upper_boundary");
    let path = write_init(
        &dir,
        "1\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\nAr 5.0e-10 0 0\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::PositionOutsideBox {
            line_number,
            direction,
            ..
        } => {
            assert_eq!(line_number, 3);
            assert_eq!(direction, "a");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-ea3bef6c
#[test]
fn reject_past_upper() {
    let dir = tmp_path("past_upper");
    let path = write_init(
        &dir,
        "1\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\nAr 6.0e-10 0 0\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::PositionOutsideBox { direction, .. } => {
            assert_eq!(direction, "a")
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-11dbe0a5
#[test]
fn reject_past_lower_y() {
    let dir = tmp_path("past_lower_y");
    let path = write_init(
        &dir,
        "1\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\nAr 0 -6.0e-10 0\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::PositionOutsideBox { direction, .. } => {
            assert_eq!(direction, "b")
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-ab40fe6d
#[test]
fn reject_nonblank_trailing() {
    let dir = tmp_path("trailing_garbage");
    let path = write_init(
        &dir,
        "2\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\nAr 0 0 0\nAr 1e-10 0 0\ngarbage\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::TrailingContent { line_number } => {
            assert_eq!(line_number, 5);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-c9e83e73
#[test]
fn tolerate_blank_trailing() {
    let dir = tmp_path("trailing_blank");
    let path = write_init(
        &dir,
        "2\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\nAr 0 0 0\nAr 1e-10 0 0\n\n\n",
    );
    load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap();
}

// rq-965c3b59
#[test]
fn implicit_ids_in_row_order() {
    let dir = tmp_path("implicit_ids");
    let path = write_init(
        &dir,
        "3\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\nAr 0 0 0\nAr 1e-10 0 0\nAr 2e-10 0 0\n",
    );
    let state = load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap();
    // Positions correspond to row order.
    assert!(state.positions_x[0].abs() < 1e-30);
    assert!((state.positions_x[1] - 1.0e-10_f32).abs() < 1e-20);
    assert!((state.positions_x[2] - 2.0e-10_f32).abs() < 1e-20);
}

// rq-ba380d56
#[test]
fn file_does_not_exist() {
    let dir = tmp_path("init_missing");
    let path = dir.join("nope.xyz");
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::Io(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// --- Image flags ---

// rq-8eb0050f
#[test]
fn file_without_image_property_has_images_none() {
    let dir = tmp_path("img_none");
    let path = dir.join("init.xyz");
    std::fs::write(
        &path,
        "1\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3\nAr 0.0 0.0 0.0\n",
    )
    .unwrap();
    let state = load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap();
    assert!(state.images.is_none());
}

// rq-36f771d7
#[test]
fn file_with_image_property_parses_three_integer_columns() {
    let dir = tmp_path("img_only");
    let path = dir.join("init.xyz");
    std::fs::write(
        &path,
        "3\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3:image:I:3\n\
         Ar 0.0 0.0 0.0 2 -1 0\nAr 0.0 0.0 0.0 0 0 0\nAr 0.0 0.0 0.0 -3 4 -7\n",
    )
    .unwrap();
    let state = load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap();
    let imgs = state.images.unwrap();
    assert_eq!(imgs.images_x, vec![2, 0, -3]);
    assert_eq!(imgs.images_y, vec![-1, 0, 4]);
    assert_eq!(imgs.images_z, vec![0, 0, -7]);
}

// rq-e794b794
#[test]
fn file_with_velo_and_image_parses_both_blocks() {
    let dir = tmp_path("velo_img");
    let path = dir.join("init.xyz");
    std::fs::write(
        &path,
        "2\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3:velo:R:3:image:I:3\n\
         Ar 0.0 0.0 0.0 1.0 2.0 3.0 4 -5 6\nAr 0.0 0.0 0.0 -1.0 -2.0 -3.0 -4 5 -6\n",
    )
    .unwrap();
    let state = load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap();
    assert!(state.velocities.is_some());
    let imgs = state.images.unwrap();
    assert_eq!(imgs.images_x, vec![4, -4]);
}

// rq-d65bcc90
#[test]
fn reject_image_column_with_non_integer_value() {
    let dir = tmp_path("img_bad");
    let path = dir.join("init.xyz");
    std::fs::write(
        &path,
        "1\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3:image:I:3\nAr 0.0 0.0 0.0 1.5 0 0\n",
    )
    .unwrap();
    let err = load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err();
    match err {
        InitStateError::InvalidNumber { column, .. } => assert_eq!(column, "image_x"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-febdd0b3
#[test]
fn reject_image_property_with_wrong_count() {
    let dir = tmp_path("img_wrong");
    let path = dir.join("init.xyz");
    std::fs::write(
        &path,
        "0\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3:image:I:2\n",
    )
    .unwrap();
    let err = load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err();
    match err {
        InitStateError::InvalidProperties(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-01607062
#[test]
fn reject_image_before_velo_in_properties() {
    let dir = tmp_path("img_order");
    let path = dir.join("init.xyz");
    std::fs::write(
        &path,
        "0\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3:image:I:3:velo:R:3\n",
    )
    .unwrap();
    let err = load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err();
    match err {
        InitStateError::InvalidProperties(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// --- Triclinic Lattice scenarios ---

#[test] // rq-b7573bd5
fn accept_lower_triangular_triclinic_lattice() {
    let dir = tmp_path("triclinic_lattice");
    let path = write_init(
        &dir,
        "0\nLattice=\"1.0e-9 0 0 0.2e-9 1.0e-9 0 0.1e-9 -0.3e-9 1.0e-9\" Properties=species:S:1:pos:R:3\n",
    );
    let state = load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap();
    let lat = state.sim_box.lattice();
    let eps = 1.0e-18_f32;
    assert!((lat[0] - 1.0e-9_f32).abs() < eps);
    assert!((lat[1] - 1.0e-9_f32).abs() < eps);
    assert!((lat[2] - 1.0e-9_f32).abs() < eps);
    assert!((lat[3] - 0.2e-9_f32).abs() < eps);
    assert!((lat[4] - 0.1e-9_f32).abs() < eps);
    assert!((lat[5] - (-0.3e-9_f32)).abs() < eps);
}

#[test] // rq-cc4e9821
fn accept_lattice_with_negative_tilts() {
    let dir = tmp_path("negative_tilts");
    let path = write_init(
        &dir,
        "0\nLattice=\"1.0e-9 0 0 -0.5e-9 1.0e-9 0 -0.1e-9 -0.2e-9 1.0e-9\" Properties=species:S:1:pos:R:3\n",
    );
    let state = load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap();
    let lat = state.sim_box.lattice();
    assert!(lat[3] < 0.0);
    assert!(lat[4] < 0.0);
    assert!(lat[5] < 0.0);
}

#[test] // rq-1aeb2da1
fn accept_position_inside_primary_parallelepiped_of_triclinic_box() {
    // xy = 0.2e-9: a particle at (0.4e-9, 0.3e-9, 0) has fractional
    //   s_c = 0, s_b = 0.3, s_a = (0.4 - 0.3*0.2) / 1.0 = 0.34 — inside.
    let dir = tmp_path("triclinic_inside");
    let path = write_init(
        &dir,
        "1\nLattice=\"1.0e-9 0 0 0.2e-9 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\nAr 0.4e-9 0.3e-9 0\n",
    );
    load_init_state(&path, &["Ar"], UnitSystem::Si).expect("position should be accepted");
}

#[test]
fn atomic_units_rescale_lattice_and_positions() {
    // Same physical system written in atomic and SI units must produce
    // identical SI-side InitState values (to f32 round-off).
    let dir = tmp_path("atomic_units_xyz");

    // SI version: 1 nm cubic box, Ar at (0.2, 0.3, 0.4) nm.
    let xyz_si = "1\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" \
                  Properties=species:S:1:pos:R:3\n\
                  Ar 2.0e-10 3.0e-10 4.0e-10\n";

    // Atomic version: same box / position rescaled to Bohr.
    let bohr_to_m = 5.29177210903e-11;
    let l_au = 1.0e-9 / bohr_to_m;
    let px_au = 2.0e-10 / bohr_to_m;
    let py_au = 3.0e-10 / bohr_to_m;
    let pz_au = 4.0e-10 / bohr_to_m;
    let xyz_au = format!(
        "1\nLattice=\"{l_au:.16e} 0 0 0 {l_au:.16e} 0 0 0 {l_au:.16e}\" \
         Properties=species:S:1:pos:R:3\nAr {px_au:.16e} {py_au:.16e} {pz_au:.16e}\n"
    );

    let path_si = write_init(&dir, xyz_si);
    let path_au = dir.join("atomic.in.xyz");
    std::fs::write(&path_au, xyz_au).unwrap();

    let state_si = load_init_state(&path_si, &["Ar"], UnitSystem::Si).unwrap();
    let state_au = load_init_state(&path_au, &["Ar"], UnitSystem::Atomic).unwrap();

    // f32 round-trip tolerance: position factor LOSES significant
    // digits when narrowing 5.29e-11 * num → f32. Allow ~1e-6 relative.
    let rel = 1e-5;
    let approx = |a: f32, b: f32| {
        (a - b).abs() <= rel * a.abs().max(b.abs()).max(f32::MIN_POSITIVE)
    };
    assert!(approx(state_au.positions_x[0], state_si.positions_x[0]));
    assert!(approx(state_au.positions_y[0], state_si.positions_y[0]));
    assert!(approx(state_au.positions_z[0], state_si.positions_z[0]));
    // SimulationBox edge lengths should agree on the meter scale.
    assert!(approx(state_au.sim_box.lx(), state_si.sim_box.lx()));
    assert!(approx(state_au.sim_box.ly(), state_si.sim_box.ly()));
    assert!(approx(state_au.sim_box.lz(), state_si.sim_box.lz()));
}

#[test]
fn atomic_units_rescale_velocities() {
    let dir = tmp_path("atomic_units_velocities");
    // Velocities in m/s vs atomic-unit velocities.
    let v_si = 500.0_f64; // 500 m/s
    let v_au_per_si = 1.0 / 2187691.2636411153;
    let v_au = v_si * v_au_per_si;

    let xyz_si = format!(
        "1\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" \
         Properties=species:S:1:pos:R:3:velo:R:3\n\
         Ar 0 0 0 {v_si:.16e} 0 0\n"
    );
    let l_au = 1.0e-9 / 5.29177210903e-11;
    let xyz_au = format!(
        "1\nLattice=\"{l_au:.16e} 0 0 0 {l_au:.16e} 0 0 0 {l_au:.16e}\" \
         Properties=species:S:1:pos:R:3:velo:R:3\n\
         Ar 0 0 0 {v_au:.16e} 0 0\n"
    );

    let path_si = write_init(&dir, &xyz_si);
    let path_au = dir.join("atomic.in.xyz");
    std::fs::write(&path_au, xyz_au).unwrap();

    let state_si = load_init_state(&path_si, &["Ar"], UnitSystem::Si).unwrap();
    let state_au = load_init_state(&path_au, &["Ar"], UnitSystem::Atomic).unwrap();

    let vx_si = state_si.velocities.as_ref().unwrap().velocities_x[0];
    let vx_au = state_au.velocities.as_ref().unwrap().velocities_x[0];
    let rel = 1e-5;
    assert!((vx_au - vx_si).abs() <= rel * vx_si.abs().max(f32::MIN_POSITIVE));
}

#[test] // rq-8a3858fd
fn reject_position_with_fractional_coord_outside_primary_image() {
    // Lattice with non-zero yz=0.5e-9; a position with s_b >= 0.5
    // along b should be rejected.
    let dir = tmp_path("triclinic_outside");
    let path = write_init(
        &dir,
        // s_b = 0.5e-9 / 1.0e-9 = 0.5 — exactly the upper bound, rejected.
        "1\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0.5e-9 1.0e-9\" Properties=species:S:1:pos:R:3\nAr 0 5.0e-10 0\n",
    );
    match load_init_state(&path, &["Ar"], UnitSystem::Si).unwrap_err() {
        InitStateError::PositionOutsideBox {
            direction,
            fractional,
            ..
        } => {
            assert_eq!(direction, "b");
            assert!(fractional >= 0.5);
        }
        other => panic!("unexpected: {other:?}"),
    }
}
