// rq-9e1eee68
use std::path::{Path, PathBuf};

use dynamics::forces::{TopologyFileError, load_topology_file};
use dynamics::integrator::ConstraintRegistry;
use dynamics::io::config::NamedSlotConfig;

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
    let (bl, _al, el, _cl) = load_topology_file(&path, 4, &["CC", "CN"], &[], &[], &registry()).unwrap();
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
    let (bl, _al, _, _cl) = load_topology_file(&path, 4, &["CC"], &[], &[], &registry()).unwrap();
    assert_eq!(bl.bonds[0].atom_i, 1);
    assert_eq!(bl.bonds[0].atom_j, 3);
}

// rq-d998b5c0
#[test]
fn bonds_sorted() {
    let dir = tmp_path("sorted");
    let body = "[bonds]\n2 3 CC\n0 1 CC\n1 2 CC\n";
    let path = write(&dir, body);
    let (bl, _al, _, _cl) = load_topology_file(&path, 4, &["CC"], &[], &[], &registry()).unwrap();
    let pairs: Vec<(u32, u32)> = bl.bonds.iter().map(|b| (b.atom_i, b.atom_j)).collect();
    assert_eq!(pairs, vec![(0, 1), (1, 2), (2, 3)]);
}

// rq-9c1c58ef
#[test]
fn exclusion_scale_defaults_to_zero() {
    let dir = tmp_path("default_scale");
    let body = "[exclusions]\n0 1\n";
    let path = write(&dir, body);
    let (_, _al, el, _cl) = load_topology_file(&path, 2, &[], &[], &[], &registry()).unwrap();
    assert_eq!(el.entries[0].scale_lj, 0.0);
}

// rq-dcba6fce
#[test]
fn implicit_exclusion_added_for_bond() {
    let dir = tmp_path("implicit");
    let body = "[bonds]\n0 1 CC\n";
    let path = write(&dir, body);
    let (_, _al, el, _cl) = load_topology_file(&path, 2, &["CC"], &[], &[], &registry()).unwrap();
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
    let (_, _al, el, _cl) = load_topology_file(&path, 2, &["CC"], &[], &[], &registry()).unwrap();
    assert_eq!(el.entries.len(), 1);
    assert_eq!(el.entries[0].scale_lj, 1.0);
}

// rq-b0c18819
#[test]
fn empty_file_is_valid() {
    let dir = tmp_path("empty_file");
    let body = "# nothing here\n\n";
    let path = write(&dir, body);
    let (bl, _al, el, _cl) = load_topology_file(&path, 4, &[], &[], &[], &registry()).unwrap();
    assert!(bl.is_empty());
    assert!(el.entries.is_empty());
}

// rq-40f02b6a
#[test]
fn empty_sections_valid() {
    let dir = tmp_path("empty_sections");
    let body = "[bonds]\n[exclusions]\n";
    let path = write(&dir, body);
    let (bl, _al, el, _cl) = load_topology_file(&path, 4, &[], &[], &[], &registry()).unwrap();
    assert!(bl.is_empty());
    assert!(el.entries.is_empty());
}

// rq-9da14aa1
#[test]
fn sections_either_order() {
    let dir = tmp_path("section_order");
    let body = "[exclusions]\n0 2 0.5\n\n[bonds]\n0 1 CC\n";
    let path = write(&dir, body);
    let (bl, _al, el, _cl) = load_topology_file(&path, 4, &["CC"], &[], &[], &registry()).unwrap();
    assert_eq!(bl.bonds.len(), 1);
    assert_eq!(el.entries.len(), 2);
}

// rq-5b097ec0
#[test]
fn comments_and_blanks_tolerated() {
    let dir = tmp_path("comments");
    let body = "# header\n\n[bonds]\n# inside\n0 1 CC   # trailing\n\n[exclusions]\n0 1 0.0\n";
    let path = write(&dir, body);
    let (bl, _al, _, _cl) = load_topology_file(&path, 2, &["CC"], &[], &[], &registry()).unwrap();
    assert_eq!(bl.bonds.len(), 1);
}

// rq-3eb8fe40 (atom_bond_offsets matches sorted bond list)
#[test]
fn atom_bond_offsets_reflect_sorted_list() {
    let dir = tmp_path("offsets");
    let body = "[bonds]\n0 1 CC\n0 2 CC\n1 3 CC\n";
    let path = write(&dir, body);
    let (bl, _al, _, _cl) = load_topology_file(&path, 4, &["CC"], &[], &[], &registry()).unwrap();
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
    match load_topology_file(&path, 4, &["CC"], &[], &[], &registry()).unwrap_err() {
        TopologyFileError::Io(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-4c245ce7
#[test]
fn unknown_section_header() {
    let dir = tmp_path("unknown_section");
    let body = "[dihedrals]\n0 1 2 3 X\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &["CC"], &[], &[], &registry()).unwrap_err() {
        TopologyFileError::UnknownSection { name, .. } => assert_eq!(name, "dihedrals"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-583d3df1
#[test]
fn duplicate_section_header() {
    let dir = tmp_path("duplicate_section");
    let body = "[bonds]\n0 1 CC\n[bonds]\n";
    let path = write(&dir, body);
    match load_topology_file(&path, 4, &["CC"], &[], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 4, &["CC"], &[], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 4, &["CC"], &[], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 4, &["CC"], &[], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 4, &["CC"], &[], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 4, &["CC"], &[], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 4, &["CC"], &[], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 4, &["CC"], &[], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 4, &["CC"], &[], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 4, &["CC"], &[], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 2, &[], &[], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 2, &[], &[], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 2, &[], &[], &[], &registry()).unwrap_err() {
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
    let (_b, _al, el, _cl) = load_topology_file(&path, 2, &[], &[], &[], &registry()).unwrap();
    assert_eq!(el.entries.len(), 1);
    assert_eq!(el.entries[0].scale_lj, 0.5);
    assert_eq!(el.entries[0].scale_coul, 0.5);
}

// rq-1fde7f32
#[test]
fn four_column_form_sets_lj_and_coul_scales_independently() {
    let dir = tmp_path("four_col");
    let body = "[exclusions]\n0 1 0.5 0.833\n";
    let path = write(&dir, body);
    let (_b, _al, el, _cl) = load_topology_file(&path, 2, &[], &[], &[], &registry()).unwrap();
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
    match load_topology_file(&path, 2, &[], &[], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 2, &[], &[], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 2, &[], &[], &[], &registry()).unwrap_err() {
        TopologyFileError::InvalidExclusionRow { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// --- Angle parsing tests ---

// rq-e768a2b1
#[test]
fn load_topology_with_an_angle() {
    let dir = tmp_path("angle_basic");
    let body = "[bonds]\n0 1 OH\n0 2 OH\n\n[angles]\n1 0 2 HOH\n";
    let path = write(&dir, body);
    let (bl, al, el, _cl) = load_topology_file(&path, 3, &["OH"], &["HOH"], &[], &registry()).unwrap();
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
    let (_bl, al, _el, _cl) = load_topology_file(&path, 3, &[], &["HOH"], &[], &registry()).unwrap();
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
    let (_bl, al, _el, _cl) = load_topology_file(&path, 4, &[], &["HOH"], &[], &registry()).unwrap();
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
    match load_topology_file(&path, 3, &[], &["HOH"], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 4, &[], &["HOH"], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 3, &[], &["HOH"], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 3, &[], &["HOH"], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 3, &[], &["HOH"], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 3, &[], &["HOH"], &[], &registry()).unwrap_err() {
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
    match load_topology_file(&path, 3, &[], &["HOH"], &[], &registry()).unwrap_err() {
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
    let (_bl, _al, el, _cl) = load_topology_file(&path, 3, &["OH"], &["HOH"], &[], &registry()).unwrap();
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
    let (_bl, al, _el, _cl) = load_topology_file(&path, 5, &[], &["HOH"], &[], &registry()).unwrap();
    // Two angles after canonicalisation+sort:
    //   angle 0: (0, 1, 2)  → slots 0, 1, 2 belong to atoms 0, 1, 2
    //   angle 1: (0, 2, 3)  → slots 3, 4, 5 belong to atoms 0, 2, 3
    // Per-atom counts: atom 0 → 2, atom 1 → 1, atom 2 → 2, atom 3 → 1, atom 4 → 0
    assert_eq!(al.atom_angle_offsets, vec![0, 2, 3, 5, 6, 6]);
    assert_eq!(al.atom_angle_indices.len(), 6);
}

// rq-67e62f4b — constraint-section parsing tests.

fn spce() -> NamedSlotConfig {
    NamedSlotConfig::from_params_str(
        "SPCE",
        "shake",
        "atoms = 3\nconstraints = [\n  { i = 0, j = 1, d = 1.0e-10 },\n  { i = 0, j = 2, d = 1.0e-10 },\n  { i = 1, j = 2, d = 1.633e-10 },\n]\n",
    )
}

// rq-fe3b32cf
#[test]
fn load_topology_with_a_shake_constraint() {
    let dir = tmp_path("constraint_shake");
    let body = "[constraints]\n0 1 2 SPCE\n";
    let path = write(&dir, body);
    let cts = vec![spce()];
    let (_bl, _al, el, cl) =
        load_topology_file(&path, 3, &[], &[], &cts, &registry()).unwrap();
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
    match load_topology_file(&path, 3, &[], &[], &cts, &registry()).unwrap_err() {
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
    match load_topology_file(&path, 3, &[], &[], &cts, &registry()).unwrap_err() {
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
    match load_topology_file(&path, 3, &[], &[], &cts, &registry()).unwrap_err() {
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
    match load_topology_file(&path, 3, &[], &[], &cts, &registry()).unwrap_err() {
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
    match load_topology_file(&path, 5, &[], &[], &cts, &registry()).unwrap_err() {
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
    match load_topology_file(&path, 3, &["OH"], &[], &cts, &registry()).unwrap_err() {
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
    let (_bl, _al, el, _cl) =
        load_topology_file(&path, 3, &[], &[], &cts, &registry()).unwrap();
    let entry = el.entries.iter().find(|e| e.atom_i == 1 && e.atom_j == 2).unwrap();
    assert_eq!(entry.scale_lj, 0.25);
    assert_eq!(entry.scale_coul, 0.25);
}

// rq-75a9815d
#[test]
fn constraint_groups_sorted_by_minimum_atom_index() {
    let dir = tmp_path("constraint_sort");
    let body = "[constraints]\n100 101 102 SPCE\n4 5 6 SPCE\n50 51 52 SPCE\n";
    let path = write(&dir, body);
    let cts = vec![spce()];
    let (_bl, _al, _el, cl) =
        load_topology_file(&path, 103, &[], &[], &cts, &registry()).unwrap();
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
    let (_bl, _al, _el, cl) =
        load_topology_file(&path, 3, &[], &[], &cts, &registry()).unwrap();
    assert!(cl.is_empty());
}
