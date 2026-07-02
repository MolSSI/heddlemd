// rq-9e1eee68
use std::path::{Path, PathBuf};

use heddle_md::forces::{TopologyFileError, load_topology_file};
use heddle_md::integrator::ConstraintRegistry;
use heddle_md::io::config::{DihedralTypeConfig, NamedSlotConfig};
use heddle_md::units::UnitSystem;

fn dihedral_type(name: &str, scale_lj: f64, scale_coul: f64) -> DihedralTypeConfig {
    DihedralTypeConfig::Periodic {
        name: name.to_string(),
        k_phi: 1.0,
        n: 1,
        phi_0: 0.0,
        scale_lj_14: scale_lj,
        scale_coul_14: scale_coul,
    }
}

fn registry() -> ConstraintRegistry {
    ConstraintRegistry::with_builtins()
}

fn tmp_path(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!(
        "heddlemd-bonds-{}-{}-{}",
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
    let (bl, _al, _dl, el, _cl, _ql) = load_topology_file(&path, 4, &["CC", "CN"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
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
    let (bl, _al, _dl, _, _cl, _ql) = load_topology_file(&path, 4, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    assert_eq!(bl.bonds[0].atom_i, 1);
    assert_eq!(bl.bonds[0].atom_j, 3);
}

// rq-d998b5c0
#[test]
fn bonds_sorted() {
    let dir = tmp_path("sorted");
    let body = "[bonds]\n2 3 CC\n0 1 CC\n1 2 CC\n";
    let path = write(&dir, body);
    let (bl, _al, _dl, _, _cl, _ql) = load_topology_file(&path, 4, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    let pairs: Vec<(u32, u32)> = bl.bonds.iter().map(|b| (b.atom_i, b.atom_j)).collect();
    assert_eq!(pairs, vec![(0, 1), (1, 2), (2, 3)]);
}

// rq-9c1c58ef
#[test]
fn exclusion_scale_defaults_to_zero() {
    let dir = tmp_path("default_scale");
    let body = "[exclusions]\n0 1\n";
    let path = write(&dir, body);
    let (_, _al, _dl, el, _cl, _ql) = load_topology_file(&path, 2, &[], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    assert_eq!(el.entries[0].scale_lj, 0.0);
}

// rq-dcba6fce
#[test]
fn implicit_exclusion_added_for_bond() {
    let dir = tmp_path("implicit");
    let body = "[bonds]\n0 1 CC\n";
    let path = write(&dir, body);
    let (_, _al, _dl, el, _cl, _ql) = load_topology_file(&path, 2, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    assert_eq!(el.entries.len(), 1);
    assert_eq!(el.entries[0].atom_i, 0);
    assert_eq!(el.entries[0].atom_j, 1);
    assert_eq!(el.entries[0].scale_lj, 0.0);
}

// rq-e9a421ef
#[test]
fn explicit_exclusion_overrides_bond_default() {
    let dir = tmp_path("override");
    let body = "[bonds]\n0 1 CC\n\n[exclusions]\n0 1 1.0\n";
    let path = write(&dir, body);
    let (_, _al, _dl, el, _cl, _ql) = load_topology_file(&path, 2, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    assert_eq!(el.entries.len(), 1);
    assert_eq!(el.entries[0].scale_lj, 1.0);
}

// rq-b0c18819
#[test]
fn empty_file_is_valid() {
    let dir = tmp_path("empty_file");
    let body = "# nothing here\n\n";
    let path = write(&dir, body);
    let (bl, _al, _dl, el, _cl, _ql) = load_topology_file(&path, 4, &[], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    assert!(bl.is_empty());
    assert!(el.entries.is_empty());
}

// rq-40f02b6a
#[test]
fn empty_sections_valid() {
    let dir = tmp_path("empty_sections");
    let body = "[bonds]\n[exclusions]\n";
    let path = write(&dir, body);
    let (bl, _al, _dl, el, _cl, _ql) = load_topology_file(&path, 4, &[], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    assert!(bl.is_empty());
    assert!(el.entries.is_empty());
}

// rq-9da14aa1
#[test]
fn sections_either_order() {
    let dir = tmp_path("section_order");
    let body = "[exclusions]\n0 2 0.5\n\n[bonds]\n0 1 CC\n";
    let path = write(&dir, body);
    let (bl, _al, _dl, el, _cl, _ql) = load_topology_file(&path, 4, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    assert_eq!(bl.bonds.len(), 1);
    assert_eq!(el.entries.len(), 2);
}

// rq-5b097ec0
#[test]
fn comments_and_blanks_tolerated() {
    let dir = tmp_path("comments");
    let body = "# header\n\n[bonds]\n# inside\n0 1 CC   # trailing\n\n[exclusions]\n0 1 0.0\n";
    let path = write(&dir, body);
    let (bl, _al, _dl, _, _cl, _ql) = load_topology_file(&path, 2, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    assert_eq!(bl.bonds.len(), 1);
}

// rq-3eb8fe40 (atom_bond_offsets matches sorted bond list)
#[test]
fn atom_bond_offsets_reflect_sorted_list() {
    let dir = tmp_path("offsets");
    let body = "[bonds]\n0 1 CC\n0 2 CC\n1 3 CC\n";
    let path = write(&dir, body);
    let (bl, _al, _dl, _, _cl, _ql) = load_topology_file(&path, 4, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    // 3 bonds → 6 slot entries (each bond contributes 2 to atom_bond_indices).
    // Atom 0 in bonds 0,1: offsets[0..2]
    // Atom 1 in bonds 0,2: offsets[2..4]
    // Atom 2 in bond 1:    offsets[4..5]
    // Atom 3 in bond 2:    offsets[5..6]
    assert_eq!(bl.atom_bond_offsets, vec![0, 2, 4, 5, 6]);
    assert_eq!(bl.atom_bond_indices.len(), 6);
}

// rq-ef5aa4b7
#[test]
fn file_does_not_exist() {
    let dir = tmp_path("missing");
    let path = dir.join("no-such.bonds");
    match load_topology_file(&path, 4, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::Io(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-4c245ce7
#[test]
fn unknown_section_header() {
    let dir = tmp_path("unknown_section");
    // `[dihedrals]` is now an accepted section; an unknown one like
    // `[impropers]` exercises the UnknownSection path.
    let body = "[impropers]\n0 1 2 3 X\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::UnknownSection { name, .. } => assert_eq!(name, "impropers"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-583d3df1
#[test]
fn duplicate_section_header() {
    let dir = tmp_path("duplicate_section");
    let body = "[bonds]\n0 1 CC\n[bonds]\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::DuplicateSection { name, .. } => assert_eq!(name, "bonds"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-1ed32e10
#[test]
fn content_before_section_rejected() {
    let dir = tmp_path("orphan_content");
    let body = "0 1 CC\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::ContentOutsideSection { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-9df1eedb
#[test]
fn bond_row_wrong_column_count() {
    let dir = tmp_path("bond_cols");
    let body = "[bonds]\n0 1\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::InvalidBondRow { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-13e15b90
#[test]
fn bond_row_atom_out_of_range() {
    let dir = tmp_path("bond_oob");
    let body = "[bonds]\n0 5 CC\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::AtomIndexOutOfRange { index, max, .. } => {
            assert_eq!(index, 5);
            assert_eq!(max, 3);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-10d1da56
#[test]
fn self_bond_rejected() {
    let dir = tmp_path("self_bond");
    let body = "[bonds]\n2 2 CC\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::SelfBond { atom, .. } => assert_eq!(atom, 2),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-4f78f4a2
#[test]
fn duplicate_bond_rejected() {
    let dir = tmp_path("dup_bond");
    let body = "[bonds]\n0 1 CC\n1 0 CC\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::DuplicateBond { atom_i, atom_j } => {
            assert_eq!((atom_i, atom_j), (0, 1));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-e4563eec
#[test]
fn unknown_bond_type() {
    let dir = tmp_path("unknown_type");
    let body = "[bonds]\n0 1 XX\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::UnknownBondType { name, .. } => assert_eq!(name, "XX"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-f371677d
#[test]
fn exclusion_row_wrong_cols() {
    let dir = tmp_path("excl_cols");
    let body = "[exclusions]\n0 1 0.5 extra\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::InvalidExclusionRow { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-df10e81f
#[test]
fn self_exclusion_rejected() {
    let dir = tmp_path("self_excl");
    let body = "[exclusions]\n2 2 0.0\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::SelfExclusion { atom, .. } => assert_eq!(atom, 2),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-eea2f5f8
#[test]
fn duplicate_exclusion_rejected() {
    let dir = tmp_path("dup_excl");
    let body = "[exclusions]\n0 1 0.0\n1 0 0.5\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::DuplicateExclusion { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-f0b9b0f5
#[test]
fn exclusion_scale_out_of_range_negative() {
    let dir = tmp_path("scale_neg");
    let body = "[exclusions]\n0 1 -0.1\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 2, &[], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::ScaleOutOfRange { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-9f658edf
#[test]
fn exclusion_scale_out_of_range_above_one() {
    let dir = tmp_path("scale_high");
    let body = "[exclusions]\n0 1 1.5\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 2, &[], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::ScaleOutOfRange { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-2b4a324a rq-9f658edf rq-f0b9b0f5
#[test]
fn exclusion_scale_nan_rejected() {
    let dir = tmp_path("scale_nan");
    let body = "[exclusions]\n0 1 nan\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 2, &[], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::ScaleOutOfRange { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-1221d020
#[test]
fn single_scale_form_sets_both_lj_and_coul_scales_equally() {
    let dir = tmp_path("single_scale");
    let body = "[exclusions]\n0 1 0.5\n";
    let path = write(&dir, body);
    let (_b, _al, _dl, el, _cl, _ql) = load_topology_file(&path, 2, &[], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    assert_eq!(el.entries.len(), 1);
    assert_eq!(el.entries[0].scale_lj, 0.5);
    assert_eq!(el.entries[0].scale_coul, 0.5);
}

// rq-1fde7
#[test]
fn four_column_form_sets_lj_and_coul_scales_independently() {
    let dir = tmp_path("four_col");
    let body = "[exclusions]\n0 1 0.5 0.833\n";
    let path = write(&dir, body);
    let (_b, _al, _dl, el, _cl, _ql) = load_topology_file(&path, 2, &[], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    assert_eq!(el.entries.len(), 1);
    assert_eq!(el.entries[0].scale_lj, 0.5);
    assert_eq!(el.entries[0].scale_coul, 0.833);
}

// rq-6a9f0a18
#[test]
fn out_of_range_coul_scale_in_four_column_form_rejected() {
    let dir = tmp_path("coul_oor");
    let body = "[exclusions]\n0 1 0.5 1.5\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 2, &[], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::ScaleOutOfRange { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-6cd92c14
#[test]
fn too_few_exclusion_columns_rejected() {
    let dir = tmp_path("too_few");
    let body = "[exclusions]\n0\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 2, &[], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::InvalidExclusionRow { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-f371677d
#[test]
fn too_many_exclusion_columns_rejected() {
    let dir = tmp_path("too_many");
    let body = "[exclusions]\n0 1 0.5 0.8 extra\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 2, &[], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::InvalidExclusionRow { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// --- Angle parsing tests ---

// rq-e768a2b1 rq-514670c9
#[test]
fn load_topology_with_an_angle() {
    let dir = tmp_path("angle_basic");
    let body = "[bonds]\n0 1 OH\n0 2 OH\n\n[angles]\n1 0 2 HOH\n";
    let path = write(&dir, body);
    let (bl, al, _dl, el, _cl, _ql) = load_topology_file(&path, 3, &["OH"], &["HOH"], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    assert_eq!(bl.bonds.len(), 2);
    assert_eq!(al.angles.len(), 1);
    let a = &al.angles[0];
    assert_eq!(a.atom_i, 1);
    assert_eq!(a.atom_j, 0);
    assert_eq!(a.atom_k, 2);
    assert_eq!(a.angle_type_index, 0);
    // 1-2 exclusions: (0,1), (0,2); 1-3 exclusion: (1,2). All auto-derived.
    assert_eq!(el.entries.len(), 3);
}

// rq-f33ca120
#[test]
fn angle_wings_canonicalised_so_atom_i_lt_atom_k() {
    let dir = tmp_path("angle_canon");
    let body = "[angles]\n2 0 1 HOH\n";
    let path = write(&dir, body);
    let (_bl, al, _dl, _el, _cl, _ql) = load_topology_file(&path, 3, &[], &["HOH"], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    let a = &al.angles[0];
    assert_eq!(a.atom_i, 1);
    assert_eq!(a.atom_j, 0);
    assert_eq!(a.atom_k, 2);
}

// rq-ba37ec6b
#[test]
fn angles_sorted_by_centre_then_wings() {
    let dir = tmp_path("angle_sort");
    let body = "[angles]\n3 2 1 HOH\n2 0 1 HOH\n";
    let path = write(&dir, body);
    let (_bl, al, _dl, _el, _cl, _ql) = load_topology_file(&path, 4, &[], &["HOH"], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    assert_eq!(al.angles.len(), 2);
    // After canonicalisation (atom_i<atom_k) and sort by (j, i, k):
    //   first  centre j=0 → (1, 0, 2)
    //   second centre j=2 → (1, 2, 3)
    assert_eq!((al.angles[0].atom_i, al.angles[0].atom_j, al.angles[0].atom_k), (1, 0, 2));
    assert_eq!((al.angles[1].atom_i, al.angles[1].atom_j, al.angles[1].atom_k), (1, 2, 3));
}

// rq-021e6d82
#[test]
fn angle_row_wrong_column_count_rejected() {
    let dir = tmp_path("angle_short");
    let body = "[angles]\n0 1 2\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 3, &[], &["HOH"], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::InvalidAngleRow { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-b05f8682
#[test]
fn angle_row_atom_out_of_range() {
    let dir = tmp_path("angle_oor");
    let body = "[angles]\n0 1 9 HOH\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &[], &["HOH"], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::AtomIndexOutOfRange { index, max, .. } => {
            assert_eq!(index, 9);
            assert_eq!(max, 3);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-cfc7a794
#[test]
fn angle_repeated_atom_i_eq_j_rejected() {
    let dir = tmp_path("angle_ij");
    let body = "[angles]\n1 1 2 HOH\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 3, &[], &["HOH"], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::RepeatedAtomInAngle { atom, .. } => assert_eq!(atom, 1),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-9d68f8fb
#[test]
fn angle_repeated_atom_j_eq_k_rejected() {
    let dir = tmp_path("angle_jk");
    let body = "[angles]\n0 2 2 HOH\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 3, &[], &["HOH"], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::RepeatedAtomInAngle { atom, .. } => assert_eq!(atom, 2),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-220f3f10
#[test]
fn angle_repeated_atom_i_eq_k_rejected() {
    let dir = tmp_path("angle_ik");
    let body = "[angles]\n1 0 1 HOH\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 3, &[], &["HOH"], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::RepeatedAtomInAngle { atom, .. } => assert_eq!(atom, 1),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-c7c3f66a
#[test]
fn duplicate_angle_after_canonicalisation_rejected() {
    let dir = tmp_path("angle_dup");
    let body = "[angles]\n1 0 2 HOH\n2 0 1 HOH\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 3, &[], &["HOH"], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::DuplicateAngle {
            atom_i,
            atom_j,
            atom_k,
        } => {
            assert_eq!((atom_i, atom_j, atom_k), (1, 0, 2));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-086a1bd9
#[test]
fn unknown_angle_type_rejected() {
    let dir = tmp_path("angle_unk");
    let body = "[angles]\n1 0 2 XX\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 3, &[], &["HOH"], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::UnknownAngleType { name, .. } => assert_eq!(name, "XX"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-ea8ebebd
#[test]
fn explicit_exclusion_overrides_angle_implicit_default() {
    let dir = tmp_path("angle_excl_override");
    let body = "[bonds]\n0 1 OH\n0 2 OH\n[angles]\n1 0 2 HOH\n[exclusions]\n1 2 0.5 0.833\n";
    let path = write(&dir, body);
    let (_bl, _al, _dl, el, _cl, _ql) = load_topology_file(&path, 3, &["OH"], &["HOH"], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    let entry_1_2 = el.entries.iter().find(|e| e.atom_i == 1 && e.atom_j == 2).unwrap();
    assert_eq!(entry_1_2.scale_lj, 0.5);
    assert_eq!(entry_1_2.scale_coul, 0.833);
}

// rq-9a386c23
#[test]
fn atom_angle_offsets_reflect_sorted_list() {
    let dir = tmp_path("angle_offsets");
    let body = "[angles]\n0 1 2 HOH\n0 2 3 HOH\n";
    let path = write(&dir, body);
    let (_bl, al, _dl, _el, _cl, _ql) = load_topology_file(&path, 5, &[], &["HOH"], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    // Two angles after canonicalisation+sort:
    //   angle 0: (0, 1, 2)  → slots 0, 1, 2 belong to atoms 0, 1, 2
    //   angle 1: (0, 2, 3)  → slots 3, 4, 5 belong to atoms 0, 2, 3
    // Per-atom counts: atom 0 → 2, atom 1 → 1, atom 2 → 2, atom 3 → 1, atom 4 → 0
    assert_eq!(al.atom_angle_offsets, vec![0, 2, 3, 5, 6, 6]);
    assert_eq!(al.atom_angle_indices.len(), 6);
}

// --- constraint-section parsing tests --------------------------------

fn spce() -> NamedSlotConfig {
    NamedSlotConfig::from_params_str(
        "SPCE",
        "shake",
        "atoms = 3\nconstraints = [\n  { i = 0, j = 1, d = 1.0e-10 },\n  { i = 0, j = 2, d = 1.0e-10 },\n  { i = 1, j = 2, d = 1.633e-10 },\n]\n",
    )
}

// rq-fe3b32cf rq-18f5ef7a
#[test]
fn load_topology_with_a_shake_constraint() {
    let dir = tmp_path("constraint_shake");
    let body = "[constraints]\n0 1 2 SPCE\n";
    let path = write(&dir, body);
    let cts = vec![spce()];
    let (_bl, _al, _dl, el, cl, _ql) =
        load_topology_file(&path, 3, &[], &[], &[], &cts, &registry(), UnitSystem::Atomic).unwrap();
    assert_eq!(cl.groups.len(), 1);
    assert_eq!(cl.groups[0].atom_count, 3);
    assert_eq!(cl.groups[0].constraint_count, 3);
    // The algorithm is resolved at consume time via
    // `constraint_types[group.constraint_type_index].kind`.
    let resolved_kind = &cts[cl.groups[0].constraint_type_index as usize].kind;
    assert_eq!(resolved_kind, "shake");
    let atoms = &cl.group_atoms[cl.groups[0].atom_offset as usize
        ..(cl.groups[0].atom_offset + cl.groups[0].atom_count) as usize];
    assert_eq!(atoms, &[0, 1, 2]);
    // implicit exclusions added for (0,1), (0,2), (1,2)
    assert!(el.entries.iter().any(|e| e.atom_i == 0 && e.atom_j == 1));
    assert!(el.entries.iter().any(|e| e.atom_i == 0 && e.atom_j == 2));
    assert!(el.entries.iter().any(|e| e.atom_i == 1 && e.atom_j == 2));
}

// rq-5dfc02a9 rq-2fbcb56c
#[test]
fn constraint_row_wrong_atom_count_rejected() {
    let dir = tmp_path("constraint_wrong_count");
    let body = "[constraints]\n0 1 SPCE\n";
    let path = write(&dir, body);
    let cts = vec![spce()];
    match load_topology_file(&path, 3, &[], &[], &[], &cts, &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::InvalidConstraintRow { line_number, reason } => {
            assert_eq!(line_number, 2);
            assert!(reason.contains("3 atoms"));
        }
        other => panic!("expected InvalidConstraintRow, got {other:?}"),
    }
}

// rq-93506647
#[test]
fn constraint_row_repeated_atom_rejected() {
    let dir = tmp_path("constraint_repeated");
    let body = "[constraints]\n0 1 1 SPCE\n";
    let path = write(&dir, body);
    let cts = vec![spce()];
    match load_topology_file(&path, 3, &[], &[], &[], &cts, &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::SelfConstraint { atom, .. } => assert_eq!(atom, 1),
        other => panic!("expected SelfConstraint, got {other:?}"),
    }
}

// rq-44feffc6
#[test]
fn constraint_row_atom_out_of_range_rejected() {
    let dir = tmp_path("constraint_oob");
    let body = "[constraints]\n0 1 9 SPCE\n";
    let path = write(&dir, body);
    let cts = vec![spce()];
    match load_topology_file(&path, 3, &[], &[], &[], &cts, &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::AtomIndexOutOfRange { index, max, .. } => {
            assert_eq!(index, 9);
            assert_eq!(max, 2);
        }
        other => panic!("expected AtomIndexOutOfRange, got {other:?}"),
    }
}

// rq-6381db33
#[test]
fn unknown_constraint_type_rejected() {
    let dir = tmp_path("constraint_unknown_type");
    let body = "[constraints]\n0 1 2 UNKNOWN\n";
    let path = write(&dir, body);
    let cts = vec![spce()];
    match load_topology_file(&path, 3, &[], &[], &[], &cts, &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::UnknownConstraintType { name, .. } => {
            assert_eq!(name, "UNKNOWN");
        }
        other => panic!("expected UnknownConstraintType, got {other:?}"),
    }
}

// rq-15b6d3a4 rq-7f5b3a74
#[test]
fn duplicate_constraint_atom_across_rows_rejected() {
    let dir = tmp_path("constraint_dup_atom");
    let body = "[constraints]\n0 1 2 SPCE\n2 3 4 SPCE\n";
    let path = write(&dir, body);
    let cts = vec![spce()];
    match load_topology_file(&path, 5, &[], &[], &[], &cts, &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::DuplicateConstraintAtom { atom } => assert_eq!(atom, 2),
        other => panic!("expected DuplicateConstraintAtom, got {other:?}"),
    }
}

// rq-8ea6cf9c rq-d05a8f16
#[test]
fn bond_and_constraint_pair_overlap_rejected() {
    let dir = tmp_path("constraint_bond_overlap");
    let body = "[bonds]\n0 1 OH\n\n[constraints]\n0 1 2 SPCE\n";
    let path = write(&dir, body);
    let cts = vec![spce()];
    match load_topology_file(&path, 3, &["OH"], &[], &[], &cts, &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::BondIsAlsoConstraint { atom_i, atom_j } => {
            assert_eq!(atom_i, 0);
            assert_eq!(atom_j, 1);
        }
        other => panic!("expected BondIsAlsoConstraint, got {other:?}"),
    }
}

// rq-be8dfaa5 rq-413d9c2b
#[test]
fn explicit_exclusion_overrides_constraint_derived_default() {
    let dir = tmp_path("constraint_excl_override");
    let body = "[exclusions]\n1 2 0.25 0.25\n\n[constraints]\n0 1 2 SPCE\n";
    let path = write(&dir, body);
    let cts = vec![spce()];
    let (_bl, _al, _dl, el, _cl, _ql) =
        load_topology_file(&path, 3, &[], &[], &[], &cts, &registry(), UnitSystem::Atomic).unwrap();
    let entry = el.entries.iter().find(|e| e.atom_i == 1 && e.atom_j == 2).unwrap();
    assert_eq!(entry.scale_lj, 0.25);
    assert_eq!(entry.scale_coul, 0.25);
}

// rq-75a9815d rq-930121d6
#[test]
fn constraint_groups_sorted_by_minimum_atom_index() {
    let dir = tmp_path("constraint_sort");
    let body = "[constraints]\n100 101 102 SPCE\n4 5 6 SPCE\n50 51 52 SPCE\n";
    let path = write(&dir, body);
    let cts = vec![spce()];
    let (_bl, _al, _dl, _el, cl, _ql) =
        load_topology_file(&path, 103, &[], &[], &[], &cts, &registry(), UnitSystem::Atomic).unwrap();
    assert_eq!(cl.groups.len(), 3);
    let first = cl.group_atoms[cl.groups[0].atom_offset as usize];
    let second = cl.group_atoms[cl.groups[1].atom_offset as usize];
    let third = cl.group_atoms[cl.groups[2].atom_offset as usize];
    assert_eq!(first, 4);
    assert_eq!(second, 50);
    assert_eq!(third, 100);
}

#[test]
fn empty_constraints_section_is_valid() {
    let dir = tmp_path("constraint_empty");
    let body = "[constraints]\n";
    let path = write(&dir, body);
    let cts = vec![spce()];
    let (_bl, _al, _dl, _el, cl, _ql) =
        load_topology_file(&path, 3, &[], &[], &[], &cts, &registry(), UnitSystem::Atomic).unwrap();
    assert!(cl.is_empty());
}

// --- Non-integer / non-numeric parser errors -----------------------------

// rq-13b931f8
#[test]
fn bond_row_with_non_integer_index_rejected() {
    let dir = tmp_path("bond_non_int");
    let body = "[bonds]\nabc 1 CC\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::InvalidBondRow { .. } => {}
        other => panic!("expected InvalidBondRow, got {other:?}"),
    }
}

// rq-00bb491c
#[test]
fn angle_row_with_non_integer_index_rejected() {
    let dir = tmp_path("angle_non_int");
    let body = "[angles]\nabc 1 2 HOH\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &[], &["HOH"], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::InvalidAngleRow { .. } => {}
        other => panic!("expected InvalidAngleRow, got {other:?}"),
    }
}

// rq-06c0e11a
#[test]
fn exclusion_row_with_non_numeric_scale_rejected() {
    let dir = tmp_path("excl_non_numeric_scale");
    let body = "[exclusions]\n0 1 maybe\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &[], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::InvalidExclusionRow { .. } => {}
        other => panic!("expected InvalidExclusionRow, got {other:?}"),
    }
}

// rq-17ed07e7
#[test]
fn exclusion_atom_out_of_range_rejected() {
    let dir = tmp_path("excl_oor");
    let body = "[exclusions]\n0 9\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &[], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::AtomIndexOutOfRange { index, max, .. } => {
            assert_eq!(index, 9);
            assert_eq!(max, 3);
        }
        other => panic!("expected AtomIndexOutOfRange, got {other:?}"),
    }
}

// --- ExclusionList CSR shape --------------------------------------------

// rq-77f53d4b
#[test]
fn atom_excl_offsets_reflects_sorted_exclusion_list() {
    // Build a 4-particle topology with the scenario's exclusion set:
    //   (0, 1, lj=0.0, coul=0.0)
    //   (0, 2, lj=0.5, coul=0.5)
    //   (1, 3, lj=0.5, coul=0.833)
    // Every pair is mirror-expanded — both atom_i's and atom_j's lists
    // name the other. CSR offsets are [0, 2, 4, 5, 6]:
    //   atom 0 → 2 partners (1, 2)
    //   atom 1 → 2 partners (0, 3)
    //   atom 2 → 1 partner  (0)
    //   atom 3 → 1 partner  (1)
    let dir = tmp_path("excl_csr_shape");
    let body = "\
[exclusions]
0 1 0.0 0.0
0 2 0.5 0.5
1 3 0.5 0.833
";
    let path = write(&dir, body);
    let (_bl, _al, _dl, el, _cl, _ql) =
        load_topology_file(&path, 4, &[], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    assert_eq!(el.atom_excl_offsets, vec![0u32, 2, 4, 5, 6]);

    let slice = |a: usize| -> (usize, usize) {
        (el.atom_excl_offsets[a] as usize, el.atom_excl_offsets[a + 1] as usize)
    };
    let (s0, e0) = slice(0);
    assert_eq!(&el.atom_excl_partners[s0..e0], &[1u32, 2]);
    assert_eq!(&el.atom_excl_lj_scales[s0..e0], &[0.0, 0.5]);
    assert_eq!(&el.atom_excl_coul_scales[s0..e0], &[0.0, 0.5]);

    let (s1, e1) = slice(1);
    assert_eq!(&el.atom_excl_partners[s1..e1], &[0u32, 3]);
    assert_eq!(&el.atom_excl_lj_scales[s1..e1], &[0.0, 0.5]);
    assert!((el.atom_excl_coul_scales[s1] - 0.0).abs() < 1.0e-6);
    assert!((el.atom_excl_coul_scales[e1 - 1] - 0.833).abs() < 1.0e-6);

    let (s2, e2) = slice(2);
    assert_eq!(&el.atom_excl_partners[s2..e2], &[0u32]);
    assert!((el.atom_excl_lj_scales[s2] - 0.5).abs() < 1.0e-6);
    assert!((el.atom_excl_coul_scales[s2] - 0.5).abs() < 1.0e-6);

    let (s3, e3) = slice(3);
    assert_eq!(&el.atom_excl_partners[s3..e3], &[1u32]);
    assert!((el.atom_excl_lj_scales[s3] - 0.5).abs() < 1.0e-6);
    assert!((el.atom_excl_coul_scales[s3] - 0.833).abs() < 1.0e-6);
}

// =====================================================================
// Dihedral tests — see `rqm/forces/periodic-dihedral.md` and
// `rqm/forces/topology.md`'s [dihedrals] section.
// =====================================================================

#[test]
fn load_dihedral_row() {
    let dir = tmp_path("dih_one");
    let body = "[dihedrals]\n0 1 2 3 D\n";
    let path = write(&dir, body);
    let dts = vec![dihedral_type("D", 0.5, 0.8333)];
    let (_, _, dl, _, _, _ql) =
        load_topology_file(&path, 4, &[], &[], &dts, &[], &registry(), UnitSystem::Atomic).unwrap();
    assert_eq!(dl.dihedrals.len(), 1);
    let d = &dl.dihedrals[0];
    assert_eq!(
        (d.atom_i, d.atom_j, d.atom_k, d.atom_l, d.dihedral_type_index),
        (0, 1, 2, 3, 0)
    );
}

#[test]
fn dihedral_canonicalised_so_i_leq_l() {
    let dir = tmp_path("dih_canon");
    let body = "[dihedrals]\n3 2 1 0 D\n";
    let path = write(&dir, body);
    let dts = vec![dihedral_type("D", 0.5, 0.8333)];
    let (_, _, dl, _, _, _ql) =
        load_topology_file(&path, 4, &[], &[], &dts, &[], &registry(), UnitSystem::Atomic).unwrap();
    let d = &dl.dihedrals[0];
    assert_eq!((d.atom_i, d.atom_j, d.atom_k, d.atom_l), (0, 1, 2, 3));
}

#[test]
fn dihedrals_sorted_by_quadruple() {
    let dir = tmp_path("dih_sort");
    let body = "[dihedrals]\n0 1 2 4 D\n0 1 2 3 D\n";
    let path = write(&dir, body);
    let dts = vec![dihedral_type("D", 0.5, 0.8333)];
    let (_, _, dl, _, _, _ql) =
        load_topology_file(&path, 5, &[], &[], &dts, &[], &registry(), UnitSystem::Atomic).unwrap();
    assert_eq!(dl.dihedrals.len(), 2);
    assert_eq!(dl.dihedrals[0].atom_l, 3);
    assert_eq!(dl.dihedrals[1].atom_l, 4);
}

#[test]
fn two_dihedrals_same_quad_different_types_accepted() {
    let dir = tmp_path("dih_multi_type");
    let body = "[dihedrals]\n0 1 2 3 A\n0 1 2 3 B\n";
    let path = write(&dir, body);
    let dts = vec![
        dihedral_type("A", 0.5, 0.8333),
        dihedral_type("B", 0.5, 0.8333),
    ];
    let (_, _, dl, _, _, _ql) =
        load_topology_file(&path, 4, &[], &[], &dts, &[], &registry(), UnitSystem::Atomic).unwrap();
    assert_eq!(dl.dihedrals.len(), 2);
    assert!(dl.dihedrals[0].dihedral_type_index != dl.dihedrals[1].dihedral_type_index);
}

#[test]
fn duplicate_dihedral_rejected() {
    let dir = tmp_path("dih_dup");
    // Same canonical quadruple AND same type -> DuplicateDihedral.
    let body = "[dihedrals]\n0 1 2 3 D\n3 2 1 0 D\n";
    let path = write(&dir, body);
    let dts = vec![dihedral_type("D", 0.5, 0.8333)];
    match load_topology_file(&path, 4, &[], &[], &dts, &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::DuplicateDihedral {
            atom_i,
            atom_j,
            atom_k,
            atom_l,
            dihedral_type_name,
        } => {
            assert_eq!((atom_i, atom_j, atom_k, atom_l), (0, 1, 2, 3));
            assert_eq!(dihedral_type_name, "D");
        }
        e => panic!("unexpected: {e:?}"),
    }
}

#[test]
fn dihedral_wrong_column_count_rejected() {
    let dir = tmp_path("dih_cols");
    let body = "[dihedrals]\n0 1 2 3\n";
    let path = write(&dir, body);
    let dts = vec![dihedral_type("D", 0.5, 0.8333)];
    match load_topology_file(&path, 4, &[], &[], &dts, &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::InvalidDihedralRow { .. } => {}
        e => panic!("unexpected: {e:?}"),
    }
}

#[test]
fn dihedral_atom_out_of_range() {
    let dir = tmp_path("dih_oob");
    let body = "[dihedrals]\n0 1 2 9 D\n";
    let path = write(&dir, body);
    let dts = vec![dihedral_type("D", 0.5, 0.8333)];
    match load_topology_file(&path, 4, &[], &[], &dts, &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::AtomIndexOutOfRange { index, max, .. } => {
            assert_eq!(index, 9);
            assert_eq!(max, 3);
        }
        e => panic!("unexpected: {e:?}"),
    }
}

#[test]
fn repeated_atom_in_dihedral_rejected() {
    let dir = tmp_path("dih_repeat");
    for body in &[
        "[dihedrals]\n1 1 2 3 D\n",
        "[dihedrals]\n0 1 1 3 D\n",
        "[dihedrals]\n0 1 2 2 D\n",
        "[dihedrals]\n0 1 2 0 D\n",
    ] {
        let path = write(&dir, body);
        let dts = vec![dihedral_type("D", 0.5, 0.8333)];
        match load_topology_file(&path, 4, &[], &[], &dts, &[], &registry(), UnitSystem::Atomic).unwrap_err() {
            TopologyFileError::RepeatedAtomInDihedral { .. } => {}
            e => panic!("unexpected for body {body:?}: {e:?}"),
        }
    }
}

#[test]
fn unknown_dihedral_type_rejected() {
    let dir = tmp_path("dih_unknown");
    let body = "[dihedrals]\n0 1 2 3 ZZ\n";
    let path = write(&dir, body);
    let dts = vec![dihedral_type("D", 0.5, 0.8333)];
    match load_topology_file(&path, 4, &[], &[], &dts, &[], &registry(), UnitSystem::Atomic).unwrap_err() {
        TopologyFileError::UnknownDihedralType { name, .. } => {
            assert_eq!(name, "ZZ");
        }
        e => panic!("unexpected: {e:?}"),
    }
}

#[test]
fn dihedral_implicit_14_exclusion_uses_type_scales() {
    let dir = tmp_path("dih_14_implicit");
    let body = "[dihedrals]\n0 1 2 3 D\n";
    let path = write(&dir, body);
    let dts = vec![dihedral_type("D", 0.25, 0.75)];
    let (_, _, _dl, el, _, _ql) =
        load_topology_file(&path, 4, &[], &[], &dts, &[], &registry(), UnitSystem::Atomic).unwrap();
    let entry = el
        .entries
        .iter()
        .find(|e| (e.atom_i, e.atom_j) == (0, 3))
        .expect("expected (0,3) 1-4 entry");
    assert!((entry.scale_lj - 0.25).abs() < 1.0e-6);
    assert!((entry.scale_coul - 0.75).abs() < 1.0e-6);
}

#[test]
fn explicit_14_overrides_dihedral_implicit() {
    let dir = tmp_path("dih_14_override");
    let body = "[exclusions]\n0 3 0.4 0.6\n[dihedrals]\n0 1 2 3 D\n";
    let path = write(&dir, body);
    let dts = vec![dihedral_type("D", 0.25, 0.75)];
    let (_, _, _dl, el, _, _ql) =
        load_topology_file(&path, 4, &[], &[], &dts, &[], &registry(), UnitSystem::Atomic).unwrap();
    let entry = el
        .entries
        .iter()
        .find(|e| (e.atom_i, e.atom_j) == (0, 3))
        .expect("expected (0,3) entry");
    assert!((entry.scale_lj - 0.4).abs() < 1.0e-6);
    assert!((entry.scale_coul - 0.6).abs() < 1.0e-6);
}

#[test]
fn first_dihedral_wins_for_14_exclusion() {
    let dir = tmp_path("dih_14_first_wins");
    let body = "[dihedrals]\n0 1 2 3 A\n0 1 2 3 B\n";
    let path = write(&dir, body);
    let dts = vec![
        dihedral_type("A", 0.5, 0.8333),
        dihedral_type("B", 0.25, 0.75),
    ];
    let (_, _, _dl, el, _, _ql) =
        load_topology_file(&path, 4, &[], &[], &dts, &[], &registry(), UnitSystem::Atomic).unwrap();
    let entries_for_pair: Vec<_> = el
        .entries
        .iter()
        .filter(|e| (e.atom_i, e.atom_j) == (0, 3))
        .collect();
    assert_eq!(entries_for_pair.len(), 1, "exactly one (0,3) row");
    assert!((entries_for_pair[0].scale_lj - 0.5).abs() < 1.0e-6);
    assert!((entries_for_pair[0].scale_coul - 0.8333).abs() < 1.0e-3);
}

#[test]
fn bond_derived_exclusion_overrides_dihedral_14() {
    let dir = tmp_path("dih_14_vs_bond");
    let body = "[bonds]\n0 3 CC\n[dihedrals]\n0 1 2 3 D\n";
    let path = write(&dir, body);
    let dts = vec![dihedral_type("D", 0.5, 0.8333)];
    let (_, _, _dl, el, _, _ql) =
        load_topology_file(&path, 4, &["CC"], &[], &dts, &[], &registry(), UnitSystem::Atomic).unwrap();
    let entry = el
        .entries
        .iter()
        .find(|e| (e.atom_i, e.atom_j) == (0, 3))
        .expect("expected (0,3) entry");
    assert_eq!(entry.scale_lj, 0.0);
    assert_eq!(entry.scale_coul, 0.0);
}

#[test]
fn atom_dihedral_offsets_correct() {
    let dir = tmp_path("dih_atom_offsets");
    let body = "[dihedrals]\n0 1 2 3 D\n1 2 3 4 D\n";
    let path = write(&dir, body);
    let dts = vec![dihedral_type("D", 0.5, 0.8333)];
    let (_, _, dl, _, _, _ql) =
        load_topology_file(&path, 5, &[], &[], &dts, &[], &registry(), UnitSystem::Atomic).unwrap();
    assert_eq!(dl.dihedrals.len(), 2);
    // Counts: atom 0 in 1 dihedral, atoms 1,2,3 in 2, atom 4 in 1.
    assert_eq!(dl.atom_dihedral_offsets, vec![0, 1, 3, 5, 7, 8]);
    assert_eq!(dl.atom_dihedral_indices.len(), 8);
}

// =================================================================
// Per-potential bond selection (BondList::filter_by_type_index)
// =================================================================

// rq-febe169b rq-f62d94d2
#[test]
fn filter_by_type_index_selects_subset_and_rebuilds_map() {
    use heddle_md::forces::{Bond, BondList};
    // 4 atoms; bonds of two types. type 0 kept, type 1 dropped.
    let bonds = vec![
        Bond { atom_i: 0, atom_j: 1, bond_type_index: 0 },
        Bond { atom_i: 1, atom_j: 2, bond_type_index: 1 },
        Bond { atom_i: 2, atom_j: 3, bond_type_index: 0 },
    ];
    // A deliberately-empty CSR in the input is irrelevant: the filter
    // rebuilds its own map over the selected subset.
    let full = BondList {
        bonds,
        atom_bond_offsets: vec![0; 5],
        atom_bond_indices: vec![],
        particle_count: 4,
    };
    let kept = full.filter_by_type_index(|ti| ti == 0);
    // Two bonds survive, in original order: (0,1) and (2,3).
    assert_eq!(kept.bonds.len(), 2);
    assert_eq!((kept.bonds[0].atom_i, kept.bonds[0].atom_j), (0, 1));
    assert_eq!((kept.bonds[1].atom_i, kept.bonds[1].atom_j), (2, 3));
    // Reduction map is rebuilt over the subset: atoms 0,1,2,3 each in one
    // kept bond; slot indices are 0/1 for bond 0 and 2/3 for bond 1.
    assert_eq!(kept.atom_bond_offsets, vec![0, 1, 2, 3, 4]);
    assert_eq!(kept.atom_bond_indices, vec![0, 1, 2, 3]);
    assert_eq!(kept.particle_count, 4);
}

// rq-febe169b rq-f62d94d2
#[test]
fn filter_by_type_index_keep_all_equals_original_bonds() {
    use heddle_md::forces::{Bond, BondList};
    let bonds = vec![
        Bond { atom_i: 0, atom_j: 1, bond_type_index: 0 },
        Bond { atom_i: 1, atom_j: 2, bond_type_index: 1 },
    ];
    let full = BondList {
        bonds: bonds.clone(),
        atom_bond_offsets: vec![0, 1, 3, 4],
        atom_bond_indices: vec![0, 1, 2, 3],
        particle_count: 3,
    };
    let kept = full.filter_by_type_index(|_| true);
    assert_eq!(kept.bonds.len(), 2);
    assert_eq!(kept.atom_bond_offsets, full.atom_bond_offsets);
    assert_eq!(kept.atom_bond_indices, full.atom_bond_indices);
}

// ---------------------------------------------------------------------
// [charges] section — per-atom charges (rq-107a61fc / rq-4cae9784)
// ---------------------------------------------------------------------

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1.0e-6 * (1.0 + b.abs())
}

// rq-41691785
#[test]
fn load_topology_file_with_charges_section() {
    let dir = tmp_path("charges-load");
    let body = "[charges]\n0 -0.834\n1 0.417\n2 0.417\n3 0.000\n";
    let path = write(&dir, body);
    let (_bl, _al, _dl, _el, _cl, ql) =
        load_topology_file(&path, 4, &[], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    let cl = ql.expect("charge list present");
    assert_eq!(cl.particle_count, 4);
    let expect = [-0.834_f32, 0.417, 0.417, 0.0];
    for (got, want) in cl.charges.iter().zip(expect.iter()) {
        assert!(approx(*got, *want), "got {got}, want {want}");
    }
}

// rq-08ab6b03
#[test]
fn no_charges_section_yields_none() {
    let dir = tmp_path("charges-none");
    let body = "[bonds]\n0 1 CC\n";
    let path = write(&dir, body);
    let (_bl, _al, _dl, _el, _cl, ql) =
        load_topology_file(&path, 4, &["CC"], &[], &[], &[], &registry(), UnitSystem::Atomic)
            .unwrap();
    assert!(ql.is_none());
}

// rq-815534fc
#[test]
fn charge_rows_may_appear_in_any_order() {
    let dir = tmp_path("charges-order");
    let body = "[charges]\n3 0.0\n1 0.417\n0 -0.834\n2 0.417\n";
    let path = write(&dir, body);
    let (_bl, _al, _dl, _el, _cl, ql) =
        load_topology_file(&path, 4, &[], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    let cl = ql.unwrap();
    let expect = [-0.834_f32, 0.417, 0.417, 0.0];
    for (got, want) in cl.charges.iter().zip(expect.iter()) {
        assert!(approx(*got, *want), "got {got}, want {want}");
    }
}

// rq-f85569fa
#[test]
fn negative_and_zero_charges_accepted() {
    let dir = tmp_path("charges-signs");
    let body = "[charges]\n0 -1.0\n1 0.0\n2 0.5\n3 0.5\n";
    let path = write(&dir, body);
    let (_bl, _al, _dl, _el, _cl, ql) =
        load_topology_file(&path, 4, &[], &[], &[], &[], &registry(), UnitSystem::Atomic).unwrap();
    let cl = ql.unwrap();
    assert!(approx(cl.charges[0], -1.0));
    assert!(approx(cl.charges[1], 0.0));
}

// rq-f6b8d252
#[test]
fn si_charge_values_converted_to_atomic_at_load() {
    let dir = tmp_path("charges-si");
    // 1 e = 1.602176634e-19 C; in SI mode this converts to 1.0 atomic (e).
    let body = "[charges]\n0 1.602176634e-19\n1 0.0\n2 0.0\n3 0.0\n";
    let path = write(&dir, body);
    let (_bl, _al, _dl, _el, _cl, ql) =
        load_topology_file(&path, 4, &[], &[], &[], &[], &registry(), UnitSystem::Si).unwrap();
    let cl = ql.unwrap();
    assert!(approx(cl.charges[0], 1.0), "got {}", cl.charges[0]);
}

// rq-ce19b7ab
#[test]
fn charges_section_missing_an_atom_rejected() {
    let dir = tmp_path("charges-missing");
    let body = "[charges]\n0 -0.834\n1 0.417\n2 0.417\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &[], &[], &[], &[], &registry(), UnitSystem::Atomic)
        .unwrap_err()
    {
        TopologyFileError::IncompleteCharges { present, particle_count } => {
            assert_eq!(present, 3);
            assert_eq!(particle_count, 4);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

// rq-194105d0
#[test]
fn empty_charges_section_with_nonzero_particle_count_rejected() {
    let dir = tmp_path("charges-empty");
    let body = "[charges]\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &[], &[], &[], &[], &registry(), UnitSystem::Atomic)
        .unwrap_err()
    {
        TopologyFileError::IncompleteCharges { present, particle_count } => {
            assert_eq!(present, 0);
            assert_eq!(particle_count, 4);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

// rq-bc05f0f0
#[test]
fn duplicate_atom_in_charges_rejected() {
    let dir = tmp_path("charges-dup");
    let body = "[charges]\n0 -0.834\n1 0.417\n2 0.417\n3 0.0\n1 0.1\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &[], &[], &[], &[], &registry(), UnitSystem::Atomic)
        .unwrap_err()
    {
        TopologyFileError::DuplicateChargeAtom { atom } => assert_eq!(atom, 1),
        other => panic!("unexpected error: {other:?}"),
    }
}

// rq-8f03bfba
#[test]
fn charge_row_atom_index_out_of_range_rejected() {
    let dir = tmp_path("charges-oor");
    let body = "[charges]\n9 0.0\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &[], &[], &[], &[], &registry(), UnitSystem::Atomic)
        .unwrap_err()
    {
        TopologyFileError::AtomIndexOutOfRange { index, max, .. } => {
            assert_eq!(index, 9);
            assert_eq!(max, 3);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

// rq-c49984a5
#[test]
fn charge_row_wrong_column_count_rejected() {
    let dir = tmp_path("charges-cols");
    let body = "[charges]\n0 0.5 extra\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &[], &[], &[], &[], &registry(), UnitSystem::Atomic)
        .unwrap_err()
    {
        TopologyFileError::InvalidChargeRow { .. } => {}
        other => panic!("unexpected error: {other:?}"),
    }
}

// rq-4f430b32
#[test]
fn charge_row_non_finite_rejected() {
    let dir = tmp_path("charges-nan");
    let body = "[charges]\n0 nan\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &[], &[], &[], &[], &registry(), UnitSystem::Atomic)
        .unwrap_err()
    {
        TopologyFileError::InvalidChargeRow { .. } => {}
        other => panic!("unexpected error: {other:?}"),
    }
}

// rq-6f8f35a2
#[test]
fn charge_row_non_integer_atom_index_rejected() {
    let dir = tmp_path("charges-badidx");
    let body = "[charges]\nabc 0.5\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &[], &[], &[], &[], &registry(), UnitSystem::Atomic)
        .unwrap_err()
    {
        TopologyFileError::InvalidChargeRow { .. } => {}
        other => panic!("unexpected error: {other:?}"),
    }
}

// rq-27616511
#[test]
fn duplicate_charges_section_header_rejected() {
    let dir = tmp_path("charges-dupsec");
    let body = "[charges]\n0 0.0\n\n[charges]\n1 0.0\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &[], &[], &[], &[], &registry(), UnitSystem::Atomic)
        .unwrap_err()
    {
        TopologyFileError::DuplicateSection { name, .. } => assert_eq!(name, "charges"),
        other => panic!("unexpected error: {other:?}"),
    }
}
