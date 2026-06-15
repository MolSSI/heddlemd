// Integration tests for the RDF analysis kind.
//
// Feature: Radial Distribution Function Analysis (`rdf`) <!-- rq-4d1082c4 -->
//
// These tests construct trajectories by hand and run `heddlemd analyze`
// (or the library API directly), then inspect the resulting CSV to
// verify pair-enumeration correctness, bin placement, normalisation,
// reproducibility, and the empty/degenerate-case behaviour spelled out
// in rqm/analysis/rdf.md.

use std::path::{Path, PathBuf};

use heddle_md::analysis::{AnalyzeError, run_analyses};

fn tmp_path(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!(
        "heddlemd-rdf-{}-{}-{}",
        std::process::id(),
        name,
        nanos
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn sim_toml_with_types(types: &[&str], box_edge_m: f64) -> String {
    let mut s = String::from(
        r#"schema_version = 1
units = "atomic"
init = "sim.in.xyz"

[simulation]
seed = 1
temperature = 0.0

[[phase]]
name = "run"
n_steps = 0
dt = 1.0e-15


[phase.integrator]
kind = "velocity-verlet"
lossless = false

"#,
    );
    for t in types {
        s.push_str(&format!(
            "[[particle_types]]\nname = \"{t}\"\nmass = 6.6335e-26\n\n"
        ));
    }
    // Cover every unordered type pair with a tiny LJ entry.
    for i in 0..types.len() {
        for j in i..types.len() {
            s.push_str(&format!(
                "[[pair_interactions]]\nbetween = [\"{}\", \"{}\"]\npotential = \"lennard-jones\"\nsigma = 1.0e-10\nepsilon = 1.0e-25\ncutoff = 1.0e-10\nr_switch = 1.0e-10\n\n",
                types[i], types[j],
            ));
        }
    }
    s.push_str(&format!(
        "[neighbor_list]\nmode = \"all-pairs\"\n\n[output]\ntrajectory_every = 1\nlog_every = 0\n"
    ));
    // We don't actually run the simulation, so the box-vs-cutoff check
    // doesn't matter here; the box is determined by the trajectory.
    let _ = box_edge_m;
    s
}

fn write_init(dir: &Path, types_per_atom: &[&str]) {
    let mut body = format!(
        "{n}\nLattice=\"4.0e-9 0 0 0 4.0e-9 0 0 0 4.0e-9\" Properties=species:S:1:pos:R:3\n",
        n = types_per_atom.len()
    );
    for t in types_per_atom {
        body.push_str(&format!("{t} 0.0 0.0 0.0\n"));
    }
    std::fs::write(dir.join("sim.in.xyz"), body).unwrap();
}

/// Write a trajectory with one or more frames. Each frame lists the
/// particles in declaration order; `positions` is per-frame
/// `(species, x, y, z)`.
fn write_trajectory(
    dir: &Path,
    frames: &[Vec<(&str, f64, f64, f64)>],
) {
    let mut body = String::new();
    for (i, frame) in frames.iter().enumerate() {
        body.push_str(&format!(
            "{n}\nLattice=\"4.0e-9 0 0 0 4.0e-9 0 0 0 4.0e-9\" Properties=species:S:1:pos:R:3 Step={i} Time={t:.9e}\n",
            n = frame.len(),
            t = i as f64,
        ));
        for (sp, x, y, z) in frame {
            body.push_str(&format!("{sp} {x:.9e} {y:.9e} {z:.9e}\n"));
        }
    }
    std::fs::write(dir.join("sim.out.run.xyz"), body).unwrap();
}

fn write_analysis(dir: &Path, body: &str) -> PathBuf {
    let p = dir.join("sim.in.analysis");
    std::fs::write(&p, body).unwrap();
    p
}

fn read_csv(path: &Path) -> Vec<(f64, f64, u64)> {
    let s = std::fs::read_to_string(path).unwrap();
    let mut out = Vec::new();
    for (i, line) in s.lines().enumerate() {
        if i == 0 {
            assert_eq!(line, "r,g_r,count", "csv header malformed");
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        assert_eq!(parts.len(), 3, "row {i} has {} cols", parts.len());
        out.push((
            parts[0].parse().unwrap(),
            parts[1].parse().unwrap(),
            parts[2].parse().unwrap(),
        ));
    }
    out
}

fn rdf_body(name: &str, between: [&str; 2], r_max: f64, n_bins: u64) -> String {
    format!(
        r#"schema_version = 1

[[analyses]]
name = "{name}"
kind = "rdf"
between = ["{a}", "{b}"]
r_max = {r_max:e}
n_bins = {n_bins}
"#,
        a = between[0],
        b = between[1],
    )
}

// --- Parameter validation ---

// rq-cfd1d536
#[test]
fn reject_missing_between() {
    let dir = tmp_path("missing_between");
    std::fs::write(dir.join("sim.in.toml"), sim_toml_with_types(&["Ar"], 4.0e-9)).unwrap();
    write_init(&dir, &["Ar", "Ar"]);
    write_trajectory(
        &dir,
        &[vec![("Ar", -2.5e-10, 0.0, 0.0), ("Ar", 2.5e-10, 0.0, 0.0)]],
    );
    let path = write_analysis(
        &dir,
        r#"schema_version = 1

[[analyses]]
name = "x"
kind = "rdf"
r_max = 1.0e-9
n_bins = 8
"#,
    );
    match run_analyses(&path).unwrap_err() {
        AnalyzeError::MissingField { field } => assert!(field.ends_with("between")),
        other => panic!("expected MissingField, got {other:?}"),
    }
}

// rq-d4c17bd3
#[test]
fn reject_r_max_above_half_box() {
    let dir = tmp_path("r_max_too_large");
    std::fs::write(dir.join("sim.in.toml"), sim_toml_with_types(&["Ar"], 4.0e-9)).unwrap();
    write_init(&dir, &["Ar", "Ar"]);
    write_trajectory(
        &dir,
        &[vec![("Ar", -2.5e-10, 0.0, 0.0), ("Ar", 2.5e-10, 0.0, 0.0)]],
    );
    // 4 nm box, half = 2 nm. r_max = 3 nm fails.
    let path = write_analysis(&dir, &rdf_body("x", ["Ar", "Ar"], 3.0e-9, 8));
    match run_analyses(&path).unwrap_err() {
        AnalyzeError::Analysis { error: e, .. } => {
            let s = format!("{e}");
            assert!(s.contains("r_max"), "got {s}");
        }
        other => panic!("expected Analysis, got {other:?}"),
    }
}

// rq-5f1d5034
#[test]
fn reject_n_bins_zero() {
    let dir = tmp_path("n_bins_zero");
    std::fs::write(dir.join("sim.in.toml"), sim_toml_with_types(&["Ar"], 4.0e-9)).unwrap();
    write_init(&dir, &["Ar", "Ar"]);
    write_trajectory(
        &dir,
        &[vec![("Ar", -2.5e-10, 0.0, 0.0), ("Ar", 2.5e-10, 0.0, 0.0)]],
    );
    let path = write_analysis(&dir, &rdf_body("x", ["Ar", "Ar"], 1.0e-9, 0));
    match run_analyses(&path).unwrap_err() {
        AnalyzeError::InvalidValue { field, .. } => assert!(field.ends_with("n_bins")),
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

// rq-ba2f07bd
#[test]
fn reject_between_undeclared_type() {
    let dir = tmp_path("undeclared_type");
    std::fs::write(dir.join("sim.in.toml"), sim_toml_with_types(&["Ar"], 4.0e-9)).unwrap();
    write_init(&dir, &["Ar", "Ar"]);
    write_trajectory(
        &dir,
        &[vec![("Ar", -2.5e-10, 0.0, 0.0), ("Ar", 2.5e-10, 0.0, 0.0)]],
    );
    let path = write_analysis(&dir, &rdf_body("x", ["Ar", "Kr"], 1.0e-9, 8));
    match run_analyses(&path).unwrap_err() {
        AnalyzeError::Analysis { error, .. } => {
            let s = format!("{error}");
            assert!(s.contains("Kr"));
        }
        other => panic!("expected Analysis error, got {other:?}"),
    }
}

// rq-c505f34b
#[test]
fn reject_same_type_single_particle() {
    let dir = tmp_path("single_particle");
    std::fs::write(dir.join("sim.in.toml"), sim_toml_with_types(&["Ar"], 4.0e-9)).unwrap();
    write_init(&dir, &["Ar"]);
    write_trajectory(&dir, &[vec![("Ar", 0.0, 0.0, 0.0)]]);
    let path = write_analysis(&dir, &rdf_body("x", ["Ar", "Ar"], 1.0e-9, 8));
    match run_analyses(&path).unwrap_err() {
        AnalyzeError::Analysis { error, .. } => {
            let s = format!("{error}");
            assert!(s.contains("N_A"), "got {s}");
        }
        other => panic!("expected Analysis error, got {other:?}"),
    }
}

// --- Algorithm and output ---

// rq-66f2679e
#[test]
fn output_has_expected_row_count() {
    let dir = tmp_path("row_count");
    std::fs::write(dir.join("sim.in.toml"), sim_toml_with_types(&["Ar"], 4.0e-9)).unwrap();
    write_init(&dir, &["Ar", "Ar"]);
    write_trajectory(
        &dir,
        &[vec![("Ar", -2.5e-10, 0.0, 0.0), ("Ar", 2.5e-10, 0.0, 0.0)]],
    );
    let path = write_analysis(&dir, &rdf_body("x", ["Ar", "Ar"], 1.0e-9, 64));
    run_analyses(&path).unwrap();
    let rows = read_csv(&dir.join("sim.out.x.csv"));
    assert_eq!(rows.len(), 64);
}

// rq-43567b30 rq-60c534f2
#[test]
fn bin_centers_match_formula() {
    let dir = tmp_path("bin_centres");
    std::fs::write(dir.join("sim.in.toml"), sim_toml_with_types(&["Ar"], 4.0e-9)).unwrap();
    write_init(&dir, &["Ar", "Ar"]);
    write_trajectory(
        &dir,
        &[vec![("Ar", -2.5e-10, 0.0, 0.0), ("Ar", 2.5e-10, 0.0, 0.0)]],
    );
    // r_max = 1e-9, n_bins = 10 → Δr = 1e-10, first centre = 5e-11,
    // last centre = 9.5e-10.
    let path = write_analysis(&dir, &rdf_body("x", ["Ar", "Ar"], 1.0e-9, 10));
    run_analyses(&path).unwrap();
    let rows = read_csv(&dir.join("sim.out.x.csv"));
    assert!((rows[0].0 - 5.0e-11).abs() < 1e-20);
    assert!((rows[9].0 - 9.5e-10).abs() < 1e-20);
}

// rq-9aacb1f9
#[test]
fn same_type_two_particles_at_known_distance() {
    let dir = tmp_path("two_ar");
    std::fs::write(dir.join("sim.in.toml"), sim_toml_with_types(&["Ar"], 4.0e-9)).unwrap();
    write_init(&dir, &["Ar", "Ar"]);
    // Separation 5e-10 m along x.
    write_trajectory(
        &dir,
        &[vec![("Ar", -2.5e-10, 0.0, 0.0), ("Ar", 2.5e-10, 0.0, 0.0)]],
    );
    // r_max = 1e-9, n_bins = 10 → 5e-10 lands in bin 4 (covers [4e-10, 5e-10)).
    // Actually 5e-10 is on the boundary; floor(5e-10 / 1e-10) = 5, so bin 5.
    let path = write_analysis(&dir, &rdf_body("x", ["Ar", "Ar"], 1.0e-9, 10));
    run_analyses(&path).unwrap();
    let rows = read_csv(&dir.join("sim.out.x.csv"));
    let total: u64 = rows.iter().map(|(_, _, c)| c).sum();
    assert_eq!(total, 1, "expected exactly one pair counted, got {total}");
}

// rq-8cec425d
#[test]
fn cross_type_two_particles() {
    let dir = tmp_path("ar_kr");
    std::fs::write(
        dir.join("sim.in.toml"),
        sim_toml_with_types(&["Ar", "Kr"], 4.0e-9),
    )
    .unwrap();
    write_init(&dir, &["Ar", "Kr"]);
    write_trajectory(
        &dir,
        &[vec![("Ar", -1.5e-10, 0.0, 0.0), ("Kr", 1.5e-10, 0.0, 0.0)]],
    );
    let path = write_analysis(&dir, &rdf_body("x", ["Ar", "Kr"], 1.0e-9, 10));
    run_analyses(&path).unwrap();
    let rows = read_csv(&dir.join("sim.out.x.csv"));
    let nonzero: Vec<_> = rows.iter().enumerate().filter(|(_, (_, _, c))| *c > 0).collect();
    assert_eq!(nonzero.len(), 1, "expected exactly one non-zero bin");
    assert_eq!(nonzero[0].1.2, 1);
}

// rq-56306e1f
#[test]
fn distances_beyond_r_max_contribute_nothing() {
    let dir = tmp_path("beyond_rmax");
    std::fs::write(dir.join("sim.in.toml"), sim_toml_with_types(&["Ar"], 4.0e-9)).unwrap();
    write_init(&dir, &["Ar", "Ar"]);
    // Separation 1.5 nm along x — outside r_max = 1 nm.
    write_trajectory(
        &dir,
        &[vec![("Ar", -7.5e-10, 0.0, 0.0), ("Ar", 7.5e-10, 0.0, 0.0)]],
    );
    let path = write_analysis(&dir, &rdf_body("x", ["Ar", "Ar"], 1.0e-9, 10));
    run_analyses(&path).unwrap();
    let rows = read_csv(&dir.join("sim.out.x.csv"));
    let total: u64 = rows.iter().map(|(_, _, c)| c).sum();
    assert_eq!(total, 0);
}

// rq-17fd53d9
#[test]
fn histogram_accumulates_across_frames() {
    let dir = tmp_path("multi_frame");
    std::fs::write(dir.join("sim.in.toml"), sim_toml_with_types(&["Ar"], 4.0e-9)).unwrap();
    write_init(&dir, &["Ar", "Ar"]);
    // Separations 3.5e-10, 5.5e-10, 7.5e-10 chosen so each lands
    // cleanly in the middle of its bin (no boundary collision under
    // f32 round-off).
    write_trajectory(
        &dir,
        &[
            vec![("Ar", -1.75e-10, 0.0, 0.0), ("Ar", 1.75e-10, 0.0, 0.0)],
            vec![("Ar", -2.75e-10, 0.0, 0.0), ("Ar", 2.75e-10, 0.0, 0.0)],
            vec![("Ar", -3.75e-10, 0.0, 0.0), ("Ar", 3.75e-10, 0.0, 0.0)],
        ],
    );
    let path = write_analysis(&dir, &rdf_body("x", ["Ar", "Ar"], 1.0e-9, 10));
    run_analyses(&path).unwrap();
    let rows = read_csv(&dir.join("sim.out.x.csv"));
    let total: u64 = rows.iter().map(|(_, _, c)| c).sum();
    assert_eq!(total, 3, "expected 3 total pair counts, got {total}");
    let nonzero_bins = rows.iter().filter(|(_, _, c)| *c > 0).count();
    assert_eq!(nonzero_bins, 3);
}

// --- Normalisation ---

// rq-c70f6309
#[test]
fn empty_bin_g_r_is_zero() {
    let dir = tmp_path("empty_bin");
    std::fs::write(dir.join("sim.in.toml"), sim_toml_with_types(&["Ar"], 4.0e-9)).unwrap();
    write_init(&dir, &["Ar", "Ar"]);
    write_trajectory(
        &dir,
        &[vec![("Ar", -2.5e-10, 0.0, 0.0), ("Ar", 2.5e-10, 0.0, 0.0)]],
    );
    let path = write_analysis(&dir, &rdf_body("x", ["Ar", "Ar"], 1.0e-9, 10));
    run_analyses(&path).unwrap();
    let rows = read_csv(&dir.join("sim.out.x.csv"));
    for (_, g, c) in &rows {
        if *c == 0 {
            assert_eq!(*g, 0.0);
        }
    }
}

// rq-36665dda
#[test]
fn normalisation_formula_matches_analytical_value() {
    let dir = tmp_path("normalisation");
    std::fs::write(dir.join("sim.in.toml"), sim_toml_with_types(&["Ar"], 4.0e-9)).unwrap();
    write_init(&dir, &["Ar", "Ar"]);
    // Two-particle system separated by 5e-10 m along x. Box 4 nm cubic.
    write_trajectory(
        &dir,
        &[vec![("Ar", -2.5e-10, 0.0, 0.0), ("Ar", 2.5e-10, 0.0, 0.0)]],
    );
    let r_max = 1.0e-9;
    let n_bins = 10u64;
    let path = write_analysis(&dir, &rdf_body("x", ["Ar", "Ar"], r_max, n_bins));
    run_analyses(&path).unwrap();
    let rows = read_csv(&dir.join("sim.out.x.csv"));
    // Locate the bin with count == 1.
    let (i, _) = rows.iter().enumerate().find(|(_, (_, _, c))| *c == 1).unwrap();
    let dr = r_max / (n_bins as f64);
    let r_inner = (i as f64) * dr;
    let r_outer = ((i + 1) as f64) * dr;
    let shell_volume = 4.0 * std::f64::consts::PI / 3.0
        * (r_outer.powi(3) - r_inner.powi(3));
    // 4 nm cubic box (f32 -> f64 round-off in `volume()` shows up; use
    // the same `f64` path the implementation does by constructing the
    // box via SimulationBox::volume() ourselves).
    let box_edge = 4.0e-9_f32 as f64;
    let v = box_edge * box_edge * box_edge;
    let n_a: f64 = 2.0;
    let n_pairs = n_a * (n_a - 1.0) / 2.0;
    let expected = v / (1.0 * n_pairs * shell_volume);
    let got = rows[i].1;
    assert!(
        (got / expected - 1.0).abs() < 1e-6,
        "expected g_r ≈ {expected:e}, got {got:e}"
    );
}

// --- Reproducibility ---

// rq-8b41bc4d
#[test]
fn two_analyze_runs_produce_byte_identical_csv() {
    let dir = tmp_path("repro");
    std::fs::write(dir.join("sim.in.toml"), sim_toml_with_types(&["Ar"], 4.0e-9)).unwrap();
    write_init(&dir, &["Ar", "Ar", "Ar", "Ar"]);
    write_trajectory(
        &dir,
        &[vec![
            ("Ar", -1.5e-10, 0.0, 0.0),
            ("Ar", 0.0, 0.0, 0.0),
            ("Ar", 1.5e-10, 0.0, 0.0),
            ("Ar", 0.0, 1.5e-10, 0.0),
        ]],
    );
    let path = write_analysis(&dir, &rdf_body("x", ["Ar", "Ar"], 1.0e-9, 32));
    run_analyses(&path).unwrap();
    let a = std::fs::read(dir.join("sim.out.x.csv")).unwrap();
    std::fs::remove_file(dir.join("sim.out.x.csv")).unwrap();
    run_analyses(&path).unwrap();
    let b = std::fs::read(dir.join("sim.out.x.csv")).unwrap();
    assert_eq!(a, b);
}

// --- `between` selection ---------------------------------------------------

// rq-40dd09ae
#[test]
fn reject_between_referencing_a_type_with_zero_particles() {
    // Config declares both Ar and Kr (so `between = ["Kr", "Kr"]` passes
    // the *undeclared-type* check), but the trajectory contains only Ar
    // atoms. The RDF builder must reject the entry at build time because
    // N_A = 0 for the Kr selector.
    let dir = tmp_path("zero_particle_type");
    std::fs::write(
        dir.join("sim.in.toml"),
        sim_toml_with_types(&["Ar", "Kr"], 4.0e-9),
    )
    .unwrap();
    write_init(&dir, &["Ar", "Ar"]);
    write_trajectory(
        &dir,
        &[vec![("Ar", -2.5e-10, 0.0, 0.0), ("Ar", 2.5e-10, 0.0, 0.0)]],
    );
    let path = write_analysis(&dir, &rdf_body("x", ["Kr", "Kr"], 1.0e-9, 8));
    match run_analyses(&path).unwrap_err() {
        AnalyzeError::Analysis { error, .. } => {
            let s = format!("{error}");
            assert!(
                s.contains("between") || s.contains("Kr") || s.contains("N_A"),
                "expected an error mentioning the empty type selector; got {s}"
            );
        }
        other => panic!("expected Analysis error, got {other:?}"),
    }
}

// rq-0b5e6b6c
#[test]
fn between_order_does_not_matter() {
    // Two RDF entries, one with between = ["Ar","Kr"] and the other with
    // between = ["Kr","Ar"]. Their output CSVs (modulo the
    // `name`-derived filename) must be byte-identical: the cross-type
    // pair count is symmetric in the type selectors.
    let dir = tmp_path("between_order");
    std::fs::write(
        dir.join("sim.in.toml"),
        sim_toml_with_types(&["Ar", "Kr"], 4.0e-9),
    )
    .unwrap();
    write_init(&dir, &["Ar", "Kr", "Ar", "Kr"]);
    write_trajectory(
        &dir,
        &[vec![
            ("Ar", -1.5e-10, 0.0, 0.0),
            ("Kr", -5.0e-11, 0.0, 0.0),
            ("Ar", 5.0e-11, 0.0, 0.0),
            ("Kr", 1.5e-10, 0.0, 0.0),
        ]],
    );
    let body = r#"schema_version = 1

[[analyses]]
name = "forward"
kind = "rdf"
between = ["Ar", "Kr"]
r_max = 1.0e-9
n_bins = 10

[[analyses]]
name = "reverse"
kind = "rdf"
between = ["Kr", "Ar"]
r_max = 1.0e-9
n_bins = 10
"#;
    let path = write_analysis(&dir, body);
    run_analyses(&path).unwrap();
    let forward = std::fs::read(dir.join("sim.out.forward.csv")).unwrap();
    let reverse = std::fs::read(dir.join("sim.out.reverse.csv")).unwrap();
    assert_eq!(
        forward, reverse,
        "between = [A, B] and [B, A] must produce byte-identical CSV output"
    );
}
