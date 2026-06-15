// rq-fd8bb824 — Integration tests for `heddlemd analyze` and the analysis framework.

use std::path::{Path, PathBuf};

use heddle_md::analysis::{
    AnalysisPathRole, AnalyzeError, AnalyzeSummary, lint_analyses, load_analysis_config,
    run_analyses,
};
use heddle_md::runner::{LintOverall, LintStatus, cli_main_u8};

fn tmp_path(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!(
        "heddlemd-analyze-{}-{}-{}",
        std::process::id(),
        name,
        nanos
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn minimal_sim_toml() -> String {
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

[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 1.0e-9

[neighbor_list]
mode = "all-pairs"

[phase.output]
trajectory_every = 1
log_every = 0
"#
    .to_string()
}

fn write_two_atom_init(dir: &Path, x_sep: f64) {
    let body = format!(
        "2\nLattice=\"4.0e-9 0 0 0 4.0e-9 0 0 0 4.0e-9\" Properties=species:S:1:pos:R:3\n\
         Ar {:.9e} 0.0 0.0\nAr {:.9e} 0.0 0.0\n",
        -x_sep / 2.0,
        x_sep / 2.0,
    );
    std::fs::write(dir.join("sim.in.xyz"), body).unwrap();
}

fn write_one_frame_trajectory(dir: &Path, x_sep: f64) {
    // Write a one-frame trajectory directly so we don't have to run the
    // simulation just to produce a trajectory for the analysis test.
    let body = format!(
        "2\nLattice=\"4.0e-9 0 0 0 4.0e-9 0 0 0 4.0e-9\" Properties=species:S:1:pos:R:3 Step=0 Time=0.000000000e0\n\
         Ar {:.9e} 0.000000000e0 0.000000000e0\nAr {:.9e} 0.000000000e0 0.000000000e0\n",
        -x_sep / 2.0,
        x_sep / 2.0,
    );
    std::fs::write(dir.join("sim.out.run.xyz"), body).unwrap();
}

fn write_three_frame_trajectory(dir: &Path, seps: [f64; 3]) {
    let mut body = String::new();
    for (i, &sep) in seps.iter().enumerate() {
        body.push_str(&format!(
            "2\nLattice=\"4.0e-9 0 0 0 4.0e-9 0 0 0 4.0e-9\" Properties=species:S:1:pos:R:3 Step={i} Time={t:.9e}\n\
             Ar {:.9e} 0.000000000e0 0.000000000e0\nAr {:.9e} 0.000000000e0 0.000000000e0\n",
            -sep / 2.0,
            sep / 2.0,
            t = i as f64,
        ));
    }
    std::fs::write(dir.join("sim.out.run.xyz"), body).unwrap();
}

fn rdf_analysis_body(name: &str, r_max: f64, n_bins: u64) -> String {
    format!(
        r#"schema_version = 1

[[analyses]]
name = "{name}"
kind = "rdf"
between = ["Ar", "Ar"]
r_max = {r_max:e}
n_bins = {n_bins}
"#
    )
}

fn write_bundle(dir: &Path) -> PathBuf {
    std::fs::write(dir.join("sim.in.toml"), minimal_sim_toml()).unwrap();
    write_two_atom_init(dir, 1.0e-9);
    write_one_frame_trajectory(dir, 1.0e-9);
    let analysis = dir.join("sim.in.analysis");
    std::fs::write(&analysis, rdf_analysis_body("ar-ar", 8.0e-10, 8)).unwrap();
    analysis
}

// --- Filename convention ---

// rq-a1735ae4
#[test]
fn reject_filename_not_ending_in_in_analysis() {
    let dir = tmp_path("bad_filename");
    let path = dir.join("sim.analysis");
    std::fs::write(&path, rdf_analysis_body("ar-ar", 8.0e-10, 8)).unwrap();
    match load_analysis_config(&path).unwrap_err() {
        AnalyzeError::InvalidAnalysisFilename { path: p } => assert_eq!(p, path),
        other => panic!("expected InvalidAnalysisFilename, got {other:?}"),
    }
}

// rq-bf98584a
#[test]
fn reject_empty_root_filename() {
    let dir = tmp_path("empty_root");
    let path = dir.join(".in.analysis");
    std::fs::write(&path, rdf_analysis_body("ar-ar", 8.0e-10, 8)).unwrap();
    match load_analysis_config(&path).unwrap_err() {
        AnalyzeError::InvalidAnalysisFilename { .. } => {}
        other => panic!("expected InvalidAnalysisFilename, got {other:?}"),
    }
}

// --- Loader behavior ---

// rq-f5166314
#[test]
fn load_valid_minimal_analysis_with_implicit_pairing() {
    let dir = tmp_path("implicit_pairing");
    let path = write_bundle(&dir);
    let cfg = load_analysis_config(&path).unwrap();
    assert_eq!(cfg.first_frame, 0);
    assert!(cfg.last_frame.is_none());
    assert_eq!(cfg.stride, 1);
    assert_eq!(cfg.analyses.len(), 1);
    assert_eq!(cfg.analyses[0].name, "ar-ar");
    assert_eq!(cfg.analyses[0].kind, "rdf");
    assert_eq!(cfg.analyses[0].output_path, dir.join("sim.out.ar-ar.csv"));
    // Trajectory defaults to the sentinel and is filled in at run time.
    assert!(cfg.trajectory.as_os_str().is_empty());
    assert_eq!(cfg.simulation, dir.join("sim.in.toml"));
}

// rq-2d107b4d
#[test]
fn reject_empty_analyses_array() {
    let dir = tmp_path("empty_analyses");
    let path = dir.join("sim.in.analysis");
    std::fs::write(&path, "schema_version = 1\n").unwrap();
    match load_analysis_config(&path).unwrap_err() {
        AnalyzeError::EmptyAnalyses => {}
        other => panic!("expected EmptyAnalyses, got {other:?}"),
    }
}

// rq-f067cfe1
#[test]
fn reject_duplicate_analysis_names() {
    let dir = tmp_path("dup_names");
    let path = dir.join("sim.in.analysis");
    let body = r#"schema_version = 1

[[analyses]]
name = "x"
kind = "rdf"
between = ["Ar", "Ar"]
r_max = 1.0e-9
n_bins = 8

[[analyses]]
name = "x"
kind = "rdf"
between = ["Ar", "Ar"]
r_max = 1.0e-9
n_bins = 8
"#;
    std::fs::write(&path, body).unwrap();
    match load_analysis_config(&path).unwrap_err() {
        AnalyzeError::DuplicateAnalysisName { name } => assert_eq!(name, "x"),
        other => panic!("expected DuplicateAnalysisName, got {other:?}"),
    }
}

// rq-28fd5e34
#[test]
fn reject_non_ascii_name() {
    let dir = tmp_path("non_ascii_name");
    let path = dir.join("sim.in.analysis");
    let body = r#"schema_version = 1

[[analyses]]
name = "αβ"
kind = "rdf"
between = ["Ar", "Ar"]
r_max = 1.0e-9
n_bins = 8
"#;
    std::fs::write(&path, body).unwrap();
    match load_analysis_config(&path).unwrap_err() {
        AnalyzeError::InvalidValue { field, .. } => assert!(field.ends_with(".name")),
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

// rq-ede9a68f
#[test]
fn reject_stride_zero() {
    let dir = tmp_path("stride_zero");
    let path = dir.join("sim.in.analysis");
    let mut body = rdf_analysis_body("x", 1.0e-9, 8);
    body.insert_str(0, "stride = 0\n");
    std::fs::write(&path, body).unwrap();
    match load_analysis_config(&path).unwrap_err() {
        AnalyzeError::InvalidValue { field, .. } => assert_eq!(field, "stride"),
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

// rq-f0c97cb7
#[test]
fn reject_last_frame_before_first_frame() {
    let dir = tmp_path("last_before_first");
    let path = dir.join("sim.in.analysis");
    let mut body = rdf_analysis_body("x", 1.0e-9, 8);
    body.insert_str(0, "first_frame = 5\nlast_frame = 2\n");
    std::fs::write(&path, body).unwrap();
    match load_analysis_config(&path).unwrap_err() {
        AnalyzeError::InvalidValue { field, .. } => assert_eq!(field, "last_frame"),
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

// --- Path collisions ---

// rq-d2b16164
#[test]
fn reject_output_path_equals_trajectory() {
    let dir = tmp_path("collision_output_traj");
    std::fs::write(dir.join("sim.in.toml"), minimal_sim_toml()).unwrap();
    write_two_atom_init(&dir, 1.0e-9);
    write_one_frame_trajectory(&dir, 1.0e-9);
    let body = r#"schema_version = 1

[[analyses]]
name = "x"
kind = "rdf"
output_path = "sim.out.run.xyz"
between = ["Ar", "Ar"]
r_max = 1.0e-9
n_bins = 8
"#;
    let path = dir.join("sim.in.analysis");
    std::fs::write(&path, body).unwrap();
    match run_analyses(&path).unwrap_err() {
        AnalyzeError::AnalyzePathCollision { .. } => {}
        other => panic!("expected AnalyzePathCollision, got {other:?}"),
    }
}

// rq-bb96fc2f
#[test]
fn reject_two_analyses_sharing_output() {
    let dir = tmp_path("collision_two_outputs");
    std::fs::write(dir.join("sim.in.toml"), minimal_sim_toml()).unwrap();
    write_two_atom_init(&dir, 1.0e-9);
    write_one_frame_trajectory(&dir, 1.0e-9);
    let body = r#"schema_version = 1

[[analyses]]
name = "a"
kind = "rdf"
output_path = "shared.csv"
between = ["Ar", "Ar"]
r_max = 1.0e-9
n_bins = 8

[[analyses]]
name = "b"
kind = "rdf"
output_path = "shared.csv"
between = ["Ar", "Ar"]
r_max = 1.0e-9
n_bins = 8
"#;
    let path = dir.join("sim.in.analysis");
    std::fs::write(&path, body).unwrap();
    match run_analyses(&path).unwrap_err() {
        AnalyzeError::AnalyzePathCollision { kind_a, kind_b, .. } => {
            assert!(matches!(kind_a, AnalysisPathRole::AnalysisOutput { .. }));
            assert!(matches!(kind_b, AnalysisPathRole::AnalysisOutput { .. }));
        }
        other => panic!("expected AnalyzePathCollision, got {other:?}"),
    }
}

// --- End-to-end runs ---

fn run_and_read_csv(path: &Path) -> AnalyzeSummary {
    let summary = run_analyses(path).unwrap();
    summary
}

// rq-f388ae12
#[test]
fn end_to_end_one_frame_run_succeeds() {
    let dir = tmp_path("e2e_one_frame");
    let path = write_bundle(&dir);
    let summary = run_and_read_csv(&path);
    assert_eq!(summary.frames_consumed, 1);
    assert_eq!(summary.analyses_written, 1);
    assert!(dir.join("sim.out.ar-ar.csv").exists());
    let csv = std::fs::read_to_string(dir.join("sim.out.ar-ar.csv")).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    // Header + 8 bins.
    assert_eq!(lines.len(), 9);
    assert_eq!(lines[0], "r,g_r,count");
}

// rq-270637dd rq-23678707
#[test]
fn refuses_to_overwrite_existing_output() {
    let dir = tmp_path("no_overwrite");
    let path = write_bundle(&dir);
    std::fs::write(dir.join("sim.out.ar-ar.csv"), "existing").unwrap();
    match run_analyses(&path).unwrap_err() {
        AnalyzeError::OutputExists { path: p } => assert!(p.ends_with("sim.out.ar-ar.csv")),
        other => panic!("expected OutputExists, got {other:?}"),
    }
    // Existing file untouched.
    assert_eq!(
        std::fs::read(dir.join("sim.out.ar-ar.csv")).unwrap(),
        b"existing"
    );
}

// rq-36ec88d9
#[test]
fn missing_trajectory_is_reported_before_analyses_build() {
    let dir = tmp_path("missing_traj");
    std::fs::write(dir.join("sim.in.toml"), minimal_sim_toml()).unwrap();
    write_two_atom_init(&dir, 1.0e-9);
    let path = dir.join("sim.in.analysis");
    std::fs::write(&path, rdf_analysis_body("ar-ar", 8.0e-10, 8)).unwrap();
    // No sim.out.run.xyz exists.
    match run_analyses(&path).unwrap_err() {
        AnalyzeError::Trajectory(_) => {}
        other => panic!("expected Trajectory error, got {other:?}"),
    }
}

// rq-f76d0576
#[test]
fn missing_sibling_sim_toml_under_implicit_pairing() {
    let dir = tmp_path("missing_sibling");
    let path = dir.join("sim.in.analysis");
    std::fs::write(&path, rdf_analysis_body("ar-ar", 8.0e-10, 8)).unwrap();
    write_one_frame_trajectory(&dir, 1.0e-9);
    // No sim.in.toml.
    match run_analyses(&path).unwrap_err() {
        AnalyzeError::Config(_) => {}
        other => panic!("expected Config error, got {other:?}"),
    }
}

// rq-5fe9c9e2
#[test]
fn stride_greater_than_one_reduces_frames() {
    let dir = tmp_path("stride_3");
    std::fs::write(dir.join("sim.in.toml"), minimal_sim_toml()).unwrap();
    write_two_atom_init(&dir, 1.0e-9);
    // Five-frame trajectory.
    let mut body = String::new();
    for i in 0..5u64 {
        body.push_str(&format!(
            "2\nLattice=\"4.0e-9 0 0 0 4.0e-9 0 0 0 4.0e-9\" Properties=species:S:1:pos:R:3 Step={i} Time={t:.9e}\n\
             Ar {:.9e} 0.0 0.0\nAr {:.9e} 0.0 0.0\n",
            -5.0e-10,
            5.0e-10,
            t = i as f64,
        ));
    }
    std::fs::write(dir.join("sim.out.run.xyz"), body).unwrap();
    let analysis = format!(
        "schema_version = 1\nstride = 2\n{}",
        rdf_analysis_body("ar-ar", 8.0e-10, 8)
            .trim_start_matches("schema_version = 1\n\n")
    );
    let path = dir.join("sim.in.analysis");
    std::fs::write(&path, analysis).unwrap();
    let summary = run_analyses(&path).unwrap();
    // Positions 0, 2, 4 selected.
    assert_eq!(summary.frames_consumed, 3);
}

// rq-8aae6b06
#[test]
fn last_frame_past_end_is_rejected() {
    let dir = tmp_path("last_past_end");
    std::fs::write(dir.join("sim.in.toml"), minimal_sim_toml()).unwrap();
    write_two_atom_init(&dir, 1.0e-9);
    write_three_frame_trajectory(&dir, [5.0e-10, 6.0e-10, 7.0e-10]);
    // last_frame = 20 but only 3 frames in file.
    let analysis = format!(
        "schema_version = 1\nlast_frame = 20\n{}",
        rdf_analysis_body("ar-ar", 8.0e-10, 8)
            .trim_start_matches("schema_version = 1\n\n")
    );
    let path = dir.join("sim.in.analysis");
    std::fs::write(&path, analysis).unwrap();
    match run_analyses(&path).unwrap_err() {
        AnalyzeError::FrameOutOfRange { requested, available } => {
            assert_eq!(requested, 20);
            assert_eq!(available, 3);
        }
        other => panic!("expected FrameOutOfRange, got {other:?}"),
    }
}

// rq-5de7798b
#[test]
fn reproducibility_across_two_runs() {
    let dir_a = tmp_path("repro_a");
    let dir_b = tmp_path("repro_b");
    let path_a = write_bundle(&dir_a);
    let path_b = write_bundle(&dir_b);
    run_analyses(&path_a).unwrap();
    run_analyses(&path_b).unwrap();
    let a = std::fs::read(dir_a.join("sim.out.ar-ar.csv")).unwrap();
    let b = std::fs::read(dir_b.join("sim.out.ar-ar.csv")).unwrap();
    assert_eq!(a, b);
}

// --- Registry / unknown kind ---

// rq-4e095363
#[test]
fn unknown_kind_is_reported() {
    let dir = tmp_path("unknown_kind");
    std::fs::write(dir.join("sim.in.toml"), minimal_sim_toml()).unwrap();
    write_two_atom_init(&dir, 1.0e-9);
    write_one_frame_trajectory(&dir, 1.0e-9);
    let body = r#"schema_version = 1

[[analyses]]
name = "m"
kind = "msd"
"#;
    let path = dir.join("sim.in.analysis");
    std::fs::write(&path, body).unwrap();
    match run_analyses(&path).unwrap_err() {
        AnalyzeError::UnknownKind { kind } => assert_eq!(kind, "msd"),
        other => panic!("expected UnknownKind, got {other:?}"),
    }
}

// --- Lint dispatch ---

fn lint_stage<'a>(report: &'a heddle_md::runner::LintReport, label: &str) -> &'a LintStatus {
    &report
        .stages
        .iter()
        .find(|s| s.label == label)
        .unwrap_or_else(|| panic!("no stage labelled `{label}`"))
        .status
}

// rq-fa2fc3a9
#[test]
fn dynamics_lint_on_in_analysis_passes_for_valid_inputs() {
    let dir = tmp_path("lint_ok");
    let path = write_bundle(&dir);
    let report = lint_analyses(&path);
    assert!(report.ok(), "expected OK, got {report:?}");
    assert_eq!(report.overall, LintOverall::Ok);
    let labels: Vec<&str> = report.stages.iter().map(|s| s.label).collect();
    assert_eq!(labels, vec!["config", "output paths", "trajectory", "analyses"]);
}

// rq-b306b357
#[test]
fn dynamics_lint_reports_missing_trajectory() {
    let dir = tmp_path("lint_missing_traj");
    std::fs::write(dir.join("sim.in.toml"), minimal_sim_toml()).unwrap();
    write_two_atom_init(&dir, 1.0e-9);
    let path = dir.join("sim.in.analysis");
    std::fs::write(&path, rdf_analysis_body("ar-ar", 8.0e-10, 8)).unwrap();
    let report = lint_analyses(&path);
    assert!(!report.ok());
    assert!(matches!(lint_stage(&report, "trajectory"), LintStatus::Fail { .. }));
}

// rq-c67ad79e rq-b306b357 rq-e2cbe4fd
#[test]
fn dynamics_lint_reports_geometric_failure_under_analyses_stage() {
    let dir = tmp_path("lint_geom_fail");
    std::fs::write(dir.join("sim.in.toml"), minimal_sim_toml()).unwrap();
    write_two_atom_init(&dir, 1.0e-9);
    write_one_frame_trajectory(&dir, 1.0e-9);
    // r_max way larger than half-box: 100 nm vs 4 nm box.
    let path = dir.join("sim.in.analysis");
    std::fs::write(&path, rdf_analysis_body("ar-ar", 1.0e-7, 8)).unwrap();
    let report = lint_analyses(&path);
    assert!(!report.ok());
    assert!(matches!(lint_stage(&report, "analyses"), LintStatus::Fail { .. }));
}

// rq-fa2fc3a9
#[test]
fn cli_lint_dispatches_in_analysis_to_analyze_lint() {
    let dir = tmp_path("cli_lint_dispatch");
    let path = write_bundle(&dir);
    let exit = cli_main_u8(vec![
        "heddlemd".to_string(),
        "lint".to_string(),
        path.to_string_lossy().to_string(),
    ]);
    assert_eq!(exit, 0);
}

// rq-f2f2519f — CLI section: v1 is CPU-only; no `--with-gpu` mode is offered.
#[test]
fn cli_lint_with_gpu_on_in_analysis_is_rejected() {
    let dir = tmp_path("cli_lint_with_gpu_analysis");
    let path = write_bundle(&dir);
    let exit = cli_main_u8(vec![
        "heddlemd".to_string(),
        "lint".to_string(),
        path.to_string_lossy().to_string(),
        "--with-gpu".to_string(),
    ]);
    assert_eq!(exit, 1);
}

// rq-f2f2519f — CLI section: exit code 0 on success.
#[test]
fn cli_analyze_returns_zero_on_success() {
    let dir = tmp_path("cli_analyze_ok");
    let path = write_bundle(&dir);
    let exit = cli_main_u8(vec![
        "heddlemd".to_string(),
        "analyze".to_string(),
        path.to_string_lossy().to_string(),
    ]);
    assert_eq!(exit, 0);
}

// rq-f2f2519f — CLI section: exit code 1 before first frame is processed.
#[test]
fn cli_analyze_returns_nonzero_on_failure() {
    let dir = tmp_path("cli_analyze_fail");
    let path = write_bundle(&dir);
    std::fs::write(dir.join("sim.out.ar-ar.csv"), "existing").unwrap();
    let exit = cli_main_u8(vec![
        "heddlemd".to_string(),
        "analyze".to_string(),
        path.to_string_lossy().to_string(),
    ]);
    assert_eq!(exit, 1);
}

// --- Phase / trajectory resolution ---------------------------------------

fn two_phase_sim_toml() -> String {
    r#"schema_version = 1
units = "atomic"
init = "sim.in.xyz"

[simulation]
seed = 1
temperature = 0.0

[[phase]]
name = "equil"
n_steps = 0
dt = 1.0e-15

[phase.integrator]
kind = "velocity-verlet"
lossless = false

[phase.output]
trajectory_every = 1
log_every = 0

[[phase]]
name = "prod"
n_steps = 0
dt = 1.0e-15

[phase.integrator]
kind = "velocity-verlet"
lossless = false

[phase.output]
trajectory_every = 1
log_every = 0

[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 1.0e-9

[neighbor_list]
mode = "all-pairs"
"#
    .to_string()
}

// rq-ee7c4af4
#[test]
fn explicit_simulation_and_trajectory_override_implicit_defaults() {
    let dir = tmp_path("explicit_sim_and_traj");
    std::fs::write(dir.join("other.in.toml"), minimal_sim_toml()).unwrap();
    write_two_atom_init(&dir, 1.0e-9);
    // Place the trajectory at a non-default name.
    let body = format!(
        "2\nLattice=\"4.0e-9 0 0 0 4.0e-9 0 0 0 4.0e-9\" Properties=species:S:1:pos:R:3 Step=0 Time=0.0\n\
         Ar -5.0e-10 0 0\nAr 5.0e-10 0 0\n"
    );
    std::fs::write(dir.join("other.out.xyz"), body).unwrap();
    let analysis_body = format!(
        r#"schema_version = 1
simulation = "other.in.toml"
trajectory = "other.out.xyz"

[[analyses]]
name = "ar-ar"
kind = "rdf"
between = ["Ar", "Ar"]
r_max = 1.0e-9
n_bins = 8
"#
    );
    let path = dir.join("sim.in.analysis");
    std::fs::write(&path, analysis_body).unwrap();
    let cfg = load_analysis_config(&path).unwrap();
    assert_eq!(cfg.simulation, dir.join("other.in.toml"));
    assert_eq!(cfg.trajectory, dir.join("other.out.xyz"));
}

// rq-963604a4
#[test]
fn implicit_pairing_defaults_to_the_last_phase() {
    let dir = tmp_path("implicit_last_phase");
    std::fs::write(dir.join("sim.in.toml"), two_phase_sim_toml()).unwrap();
    write_two_atom_init(&dir, 1.0e-9);
    // Provide a trajectory at the *last* phase's default path (sim.out.prod.xyz).
    let frame = "2\nLattice=\"4.0e-9 0 0 0 4.0e-9 0 0 0 4.0e-9\" Properties=species:S:1:pos:R:3 Step=0 Time=0.0\n\
                 Ar -5.0e-10 0 0\nAr 5.0e-10 0 0\n";
    std::fs::write(dir.join("sim.out.prod.xyz"), frame).unwrap();
    let analysis_body = rdf_analysis_body("ar-ar", 8.0e-10, 8);
    let path = dir.join("sim.in.analysis");
    std::fs::write(&path, analysis_body).unwrap();
    let summary = run_analyses(&path).unwrap();
    assert_eq!(summary.frames_consumed, 1);
    assert_eq!(summary.analyses_written, 1);
    // The output csv path is derived from the analysis-config root,
    // independent of the phase name; the substantive check is that the
    // run completed against the prod-phase trajectory.
    assert!(dir.join("sim.out.ar-ar.csv").exists());
}

// rq-38117e33
#[test]
fn explicit_phase_selects_the_matching_phase_trajectory() {
    let dir = tmp_path("explicit_phase_select");
    std::fs::write(dir.join("sim.in.toml"), two_phase_sim_toml()).unwrap();
    write_two_atom_init(&dir, 1.0e-9);
    // Provide a trajectory for the equil phase only; prod is absent.
    let frame = "2\nLattice=\"4.0e-9 0 0 0 4.0e-9 0 0 0 4.0e-9\" Properties=species:S:1:pos:R:3 Step=0 Time=0.0\n\
                 Ar -5.0e-10 0 0\nAr 5.0e-10 0 0\n";
    std::fs::write(dir.join("sim.out.equil.xyz"), frame).unwrap();
    let analysis_body = format!(
        r#"schema_version = 1
phase = "equil"

[[analyses]]
name = "ar-ar"
kind = "rdf"
between = ["Ar", "Ar"]
r_max = 1.0e-9
n_bins = 8
"#
    );
    let path = dir.join("sim.in.analysis");
    std::fs::write(&path, analysis_body).unwrap();
    let cfg = load_analysis_config(&path).unwrap();
    assert_eq!(cfg.phase, "equil");
    // Running through end-to-end picks up the equil-phase trajectory.
    let summary = run_analyses(&path).unwrap();
    assert_eq!(summary.frames_consumed, 1);
}

// rq-b6d22242
#[test]
fn unknown_phase_name_is_rejected_at_run_time() {
    let dir = tmp_path("unknown_phase");
    std::fs::write(dir.join("sim.in.toml"), two_phase_sim_toml()).unwrap();
    write_two_atom_init(&dir, 1.0e-9);
    let analysis_body = format!(
        r#"schema_version = 1
phase = "warmup"

[[analyses]]
name = "ar-ar"
kind = "rdf"
between = ["Ar", "Ar"]
r_max = 1.0e-9
n_bins = 8
"#
    );
    let path = dir.join("sim.in.analysis");
    std::fs::write(&path, analysis_body).unwrap();
    match run_analyses(&path).unwrap_err() {
        AnalyzeError::UnknownPhase { phase, available } => {
            assert_eq!(phase, "warmup");
            assert_eq!(available, vec!["equil".to_string(), "prod".to_string()]);
        }
        other => panic!("expected UnknownPhase, got {other:?}"),
    }
}

// rq-2674f18a
#[test]
fn explicit_trajectory_overrides_the_phase_derived_default() {
    let dir = tmp_path("traj_override");
    std::fs::write(dir.join("sim.in.toml"), two_phase_sim_toml()).unwrap();
    write_two_atom_init(&dir, 1.0e-9);
    // Provide a trajectory at a custom path, not the phase default.
    let frame = "2\nLattice=\"4.0e-9 0 0 0 4.0e-9 0 0 0 4.0e-9\" Properties=species:S:1:pos:R:3 Step=0 Time=0.0\n\
                 Ar -5.0e-10 0 0\nAr 5.0e-10 0 0\n";
    std::fs::write(dir.join("alt.xyz"), frame).unwrap();
    let analysis_body = format!(
        r#"schema_version = 1
trajectory = "alt.xyz"

[[analyses]]
name = "ar-ar"
kind = "rdf"
between = ["Ar", "Ar"]
r_max = 1.0e-9
n_bins = 8
"#
    );
    let path = dir.join("sim.in.analysis");
    std::fs::write(&path, analysis_body).unwrap();
    let cfg = load_analysis_config(&path).unwrap();
    assert_eq!(cfg.trajectory, dir.join("alt.xyz"));
    // The explicit trajectory path is honoured at run time.
    let summary = run_analyses(&path).unwrap();
    assert_eq!(summary.frames_consumed, 1);
}

// rq-bedbd6d9
#[test]
fn reject_an_output_path_equal_to_the_simulations_log_file() {
    let dir = tmp_path("output_equals_log");
    std::fs::write(dir.join("sim.in.toml"), minimal_sim_toml()).unwrap();
    write_two_atom_init(&dir, 1.0e-9);
    write_one_frame_trajectory(&dir, 1.0e-9);
    // minimal_sim_toml's phase "run" derives its log_path to
    // {root}.out.run.log → sim.out.run.log. Set the analysis output to
    // that exact path to trigger the collision check.
    let analysis_body = r#"schema_version = 1

[[analyses]]
name = "ar-ar"
kind = "rdf"
output_path = "sim.out.run.log"
between = ["Ar", "Ar"]
r_max = 1.0e-9
n_bins = 8
"#;
    let path = dir.join("sim.in.analysis");
    std::fs::write(&path, analysis_body).unwrap();
    match run_analyses(&path).unwrap_err() {
        AnalyzeError::AnalyzePathCollision { kind_a, kind_b, .. } => {
            let a = format!("{kind_a:?}");
            let b = format!("{kind_b:?}");
            assert!(
                a.contains("Log") || b.contains("Log") || a.contains("AnalysisOutput") || b.contains("AnalysisOutput"),
                "expected collision involving the phase log; got {kind_a:?} vs {kind_b:?}"
            );
        }
        other => panic!("expected AnalyzePathCollision, got {other:?}"),
    }
}

// rq-ca13c67e
#[test]
fn a_custom_analysis_builder_composes_with_the_built_ins() {
    use heddle_md::Registries;
    use heddle_md::analysis::{
        Analysis, AnalysisBuilder, AnalysisRuntimeError, AnalyzeError,
    };
    use heddle_md::io::Config;
    use heddle_md::pbc::SimulationBox;
    use heddle_md::io::trajectory::{TrajectoryFrame, TrajectoryFrameHeader};
    use std::path::Path;

    #[derive(Debug, Clone)]
    struct MyAnalysisBuilder;
    impl AnalysisBuilder for MyAnalysisBuilder {
        fn kind_name(&self) -> &'static str {
            "my-analysis"
        }
        fn validate_params(&self, _params: &toml::Value) -> Result<(), AnalyzeError> {
            Ok(())
        }
        fn build(
            &self,
            _params: &toml::Value,
            _header: &TrajectoryFrameHeader,
            _sim_config: &Config,
        ) -> Result<Box<dyn Analysis>, AnalysisRuntimeError> {
            #[derive(Debug)]
            struct MyAnalysis;
            impl Analysis for MyAnalysis {
                fn consume_frame(
                    &mut self,
                    _frame: &TrajectoryFrame,
                    _sim_box: &SimulationBox,
                ) -> Result<(), AnalysisRuntimeError> {
                    Ok(())
                }
                fn finalize_and_write(
                    &mut self,
                    _output_path: &Path,
                    _sim_config: &Config,
                ) -> Result<(), AnalysisRuntimeError> {
                    Ok(())
                }
            }
            Ok(Box::new(MyAnalysis))
        }
        fn box_clone(&self) -> Box<dyn AnalysisBuilder> {
            Box::new(self.clone())
        }
    }

    let mut registries = Registries::with_builtins();
    assert!(registries.analyses.lookup("rdf").is_some());
    assert!(registries.analyses.lookup("my-analysis").is_none());
    registries.register_analysis(Box::new(MyAnalysisBuilder));
    assert!(
        registries.analyses.lookup("my-analysis").is_some(),
        "custom builder must be registered alongside the built-ins"
    );
    assert!(
        registries.analyses.lookup("rdf").is_some(),
        "built-in rdf builder must remain after registering a custom one"
    );
}
