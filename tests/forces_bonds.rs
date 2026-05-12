use std::path::{Path, PathBuf};

use dynamics::forces::{BondsFileError, load_bonds_file};

fn tmp_path(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!(
        "dynamics-bonds-{}-{}-{}",
        std::process::id(),
        name,
        nanos
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn write(dir: &Path, body: &str) -> PathBuf {
    let p = dir.join("topology.bonds");
    std::fs::write(&p, body).unwrap();
    p
}

// rq-8a16e6d6
#[test]
fn load_typical_bonds_file() {
    let dir = tmp_path("typical");
    let body = r#"[bonds]
0 1 CC
1 2 CC
2 3 CN

[exclusions]
0 1 0.0
0 2 0.5
"#;
    let path = write(&dir, body);
    let (bl, el) = load_bonds_file(&path, 4, &["CC", "CN"]).unwrap();
    assert_eq!(bl.bonds.len(), 3);
    assert_eq!((bl.bonds[0].atom_i, bl.bonds[0].atom_j, bl.bonds[0].bond_type_index), (0, 1, 0));
    assert_eq!((bl.bonds[1].atom_i, bl.bonds[1].atom_j, bl.bonds[1].bond_type_index), (1, 2, 0));
    assert_eq!((bl.bonds[2].atom_i, bl.bonds[2].atom_j, bl.bonds[2].bond_type_index), (2, 3, 1));
    // Effective exclusions: explicit (0,1,0.0), (0,2,0.5), implicit (1,2,0.0), (2,3,0.0)
    assert_eq!(el.entries.len(), 4);
}

// rq-fb608f06
#[test]
fn bonds_canonicalised_min_max() {
    let dir = tmp_path("canonical");
    let body = "[bonds]\n3 1 CC\n";
    let path = write(&dir, body);
    let (bl, _) = load_bonds_file(&path, 4, &["CC"]).unwrap();
    assert_eq!(bl.bonds[0].atom_i, 1);
    assert_eq!(bl.bonds[0].atom_j, 3);
}

// rq-d998b5c0
#[test]
fn bonds_sorted() {
    let dir = tmp_path("sorted");
    let body = "[bonds]\n2 3 CC\n0 1 CC\n1 2 CC\n";
    let path = write(&dir, body);
    let (bl, _) = load_bonds_file(&path, 4, &["CC"]).unwrap();
    let pairs: Vec<(u32, u32)> = bl.bonds.iter().map(|b| (b.atom_i, b.atom_j)).collect();
    assert_eq!(pairs, vec![(0, 1), (1, 2), (2, 3)]);
}

// rq-9c1c58ef
#[test]
fn exclusion_scale_defaults_to_zero() {
    let dir = tmp_path("default_scale");
    let body = "[exclusions]\n0 1\n";
    let path = write(&dir, body);
    let (_, el) = load_bonds_file(&path, 2, &[]).unwrap();
    assert_eq!(el.entries[0].scale, 0.0);
}

// rq-dcba6fce
#[test]
fn implicit_exclusion_added_for_bond() {
    let dir = tmp_path("implicit");
    let body = "[bonds]\n0 1 CC\n";
    let path = write(&dir, body);
    let (_, el) = load_bonds_file(&path, 2, &["CC"]).unwrap();
    assert_eq!(el.entries.len(), 1);
    assert_eq!(el.entries[0].atom_i, 0);
    assert_eq!(el.entries[0].atom_j, 1);
    assert_eq!(el.entries[0].scale, 0.0);
}

// rq-e9a421ef
#[test]
fn explicit_exclusion_overrides_bond_default() {
    let dir = tmp_path("override");
    let body = "[bonds]\n0 1 CC\n\n[exclusions]\n0 1 1.0\n";
    let path = write(&dir, body);
    let (_, el) = load_bonds_file(&path, 2, &["CC"]).unwrap();
    assert_eq!(el.entries.len(), 1);
    assert_eq!(el.entries[0].scale, 1.0);
}

// rq-b0c18819
#[test]
fn empty_file_is_valid() {
    let dir = tmp_path("empty_file");
    let body = "# nothing here\n\n";
    let path = write(&dir, body);
    let (bl, el) = load_bonds_file(&path, 4, &[]).unwrap();
    assert!(bl.is_empty());
    assert!(el.entries.is_empty());
}

// rq-40f02b6a
#[test]
fn empty_sections_valid() {
    let dir = tmp_path("empty_sections");
    let body = "[bonds]\n[exclusions]\n";
    let path = write(&dir, body);
    let (bl, el) = load_bonds_file(&path, 4, &[]).unwrap();
    assert!(bl.is_empty());
    assert!(el.entries.is_empty());
}

// rq-9da14aa1
#[test]
fn sections_either_order() {
    let dir = tmp_path("section_order");
    let body = "[exclusions]\n0 2 0.5\n\n[bonds]\n0 1 CC\n";
    let path = write(&dir, body);
    let (bl, el) = load_bonds_file(&path, 4, &["CC"]).unwrap();
    assert_eq!(bl.bonds.len(), 1);
    assert_eq!(el.entries.len(), 2);
}

// rq-5b097ec0
#[test]
fn comments_and_blanks_tolerated() {
    let dir = tmp_path("comments");
    let body = "# header\n\n[bonds]\n# inside\n0 1 CC   # trailing\n\n[exclusions]\n0 1 0.0\n";
    let path = write(&dir, body);
    let (bl, _) = load_bonds_file(&path, 2, &["CC"]).unwrap();
    assert_eq!(bl.bonds.len(), 1);
}

// rq-cab7b9f3 (atom_bond_offsets matches sorted bond list)
#[test]
fn atom_bond_offsets_reflect_sorted_list() {
    let dir = tmp_path("offsets");
    let body = "[bonds]\n0 1 CC\n0 2 CC\n1 3 CC\n";
    let path = write(&dir, body);
    let (bl, _) = load_bonds_file(&path, 4, &["CC"]).unwrap();
    // 3 bonds → 6 slot entries (each bond contributes 2 to atom_bond_indices).
    // Atom 0 in bonds 0,1: offsets[0..2]
    // Atom 1 in bonds 0,2: offsets[2..4]
    // Atom 2 in bond 1:    offsets[4..5]
    // Atom 3 in bond 2:    offsets[5..6]
    assert_eq!(bl.atom_bond_offsets, vec![0, 2, 4, 5, 6]);
    assert_eq!(bl.atom_bond_indices.len(), 6);
}

#[test]
fn file_does_not_exist() {
    let dir = tmp_path("missing");
    let path = dir.join("no-such.bonds");
    match load_bonds_file(&path, 4, &["CC"]).unwrap_err() {
        BondsFileError::Io(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn unknown_section_header() {
    let dir = tmp_path("unknown_section");
    let body = "[angles]\n0 1 2 X\n";
    let path = write(&dir, body);
    match load_bonds_file(&path, 4, &["CC"]).unwrap_err() {
        BondsFileError::UnknownSection { name, .. } => assert_eq!(name, "angles"),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn duplicate_section_header() {
    let dir = tmp_path("duplicate_section");
    let body = "[bonds]\n0 1 CC\n[bonds]\n";
    let path = write(&dir, body);
    match load_bonds_file(&path, 4, &["CC"]).unwrap_err() {
        BondsFileError::DuplicateSection { name, .. } => assert_eq!(name, "bonds"),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn content_before_section_rejected() {
    let dir = tmp_path("orphan_content");
    let body = "0 1 CC\n";
    let path = write(&dir, body);
    match load_bonds_file(&path, 4, &["CC"]).unwrap_err() {
        BondsFileError::ContentOutsideSection { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn bond_row_wrong_column_count() {
    let dir = tmp_path("bond_cols");
    let body = "[bonds]\n0 1\n";
    let path = write(&dir, body);
    match load_bonds_file(&path, 4, &["CC"]).unwrap_err() {
        BondsFileError::InvalidBondRow { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn bond_row_atom_out_of_range() {
    let dir = tmp_path("bond_oob");
    let body = "[bonds]\n0 5 CC\n";
    let path = write(&dir, body);
    match load_bonds_file(&path, 4, &["CC"]).unwrap_err() {
        BondsFileError::AtomIndexOutOfRange { index, max, .. } => {
            assert_eq!(index, 5);
            assert_eq!(max, 3);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn self_bond_rejected() {
    let dir = tmp_path("self_bond");
    let body = "[bonds]\n2 2 CC\n";
    let path = write(&dir, body);
    match load_bonds_file(&path, 4, &["CC"]).unwrap_err() {
        BondsFileError::SelfBond { atom, .. } => assert_eq!(atom, 2),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn duplicate_bond_rejected() {
    let dir = tmp_path("dup_bond");
    let body = "[bonds]\n0 1 CC\n1 0 CC\n";
    let path = write(&dir, body);
    match load_bonds_file(&path, 4, &["CC"]).unwrap_err() {
        BondsFileError::DuplicateBond { atom_i, atom_j } => {
            assert_eq!((atom_i, atom_j), (0, 1));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn unknown_bond_type() {
    let dir = tmp_path("unknown_type");
    let body = "[bonds]\n0 1 XX\n";
    let path = write(&dir, body);
    match load_bonds_file(&path, 4, &["CC"]).unwrap_err() {
        BondsFileError::UnknownBondType { name, .. } => assert_eq!(name, "XX"),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn exclusion_row_wrong_cols() {
    let dir = tmp_path("excl_cols");
    let body = "[exclusions]\n0 1 0.5 extra\n";
    let path = write(&dir, body);
    match load_bonds_file(&path, 4, &["CC"]).unwrap_err() {
        BondsFileError::InvalidExclusionRow { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn self_exclusion_rejected() {
    let dir = tmp_path("self_excl");
    let body = "[exclusions]\n2 2 0.0\n";
    let path = write(&dir, body);
    match load_bonds_file(&path, 4, &["CC"]).unwrap_err() {
        BondsFileError::SelfExclusion { atom, .. } => assert_eq!(atom, 2),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn duplicate_exclusion_rejected() {
    let dir = tmp_path("dup_excl");
    let body = "[exclusions]\n0 1 0.0\n1 0 0.5\n";
    let path = write(&dir, body);
    match load_bonds_file(&path, 4, &["CC"]).unwrap_err() {
        BondsFileError::DuplicateExclusion { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn exclusion_scale_out_of_range_negative() {
    let dir = tmp_path("scale_neg");
    let body = "[exclusions]\n0 1 -0.1\n";
    let path = write(&dir, body);
    match load_bonds_file(&path, 2, &[]).unwrap_err() {
        BondsFileError::ScaleOutOfRange { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn exclusion_scale_out_of_range_above_one() {
    let dir = tmp_path("scale_high");
    let body = "[exclusions]\n0 1 1.5\n";
    let path = write(&dir, body);
    match load_bonds_file(&path, 2, &[]).unwrap_err() {
        BondsFileError::ScaleOutOfRange { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn exclusion_scale_nan_rejected() {
    let dir = tmp_path("scale_nan");
    let body = "[exclusions]\n0 1 nan\n";
    let path = write(&dir, body);
    match load_bonds_file(&path, 2, &[]).unwrap_err() {
        BondsFileError::ScaleOutOfRange { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}
