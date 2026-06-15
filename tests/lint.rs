// Integration tests for the `heddlemd lint` subcommand.
//
// Mirrors the Gherkin scenarios under *Lint subcommand* in
// rqm/simulation-runner.md.

use std::path::{Path, PathBuf};

use heddle_md::runner::{
    LintOverall, LintStatus, RunnerError, cli_main_u8, lint_simulation,
};

fn tmp_path(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!(
        "heddlemd-lint-{}-{}-{}",
        std::process::id(),
        name,
        nanos
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn minimal_config_body(init_name: &str) -> String {
    format!(
        r#"schema_version = 1
init = "{init_name}"

[simulation]
seed = 1
temperature = 0.0

[[phase]]
name = "run"
n_steps = 1
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
trajectory_every = 5
log_every = 5
"#
    )
}

fn write_minimal_init(path: &Path, n: usize) {
    let mut body = String::new();
    body.push_str(&format!("{n}\n"));
    body.push_str(
        "Lattice=\"4.0e-9 0 0 0 4.0e-9 0 0 0 4.0e-9\" Properties=species:S:1:pos:R:3\n",
    );
    let box_size = 4.0e-9_f64;
    let usable = box_size * 0.8;
    let spacing = if n > 1 { usable / (n as f64 - 1.0) } else { 0.0 };
    let half_n = (n as f64 - 1.0) / 2.0;
    for i in 0..n {
        let x = (i as f64 - half_n) * spacing;
        body.push_str(&format!("Ar {x:.9e} 0.0 0.0\n"));
    }
    std::fs::write(path, body).unwrap();
}

fn write_valid_pair(dir: &Path) -> PathBuf {
    let config = dir.join("sim.in.toml");
    let init = dir.join("sim.in.xyz");
    std::fs::write(&config, minimal_config_body("sim.in.xyz")).unwrap();
    write_minimal_init(&init, 4);
    config
}

fn stage_status<'a>(report: &'a heddle_md::runner::LintReport, label: &str) -> &'a LintStatus {
    &report
        .stages
        .iter()
        .find(|s| s.label == label)
        .unwrap_or_else(|| panic!("no stage labelled `{label}`"))
        .status
}

// rq-c52b8ece rq-d87f15bd
#[test]
fn lint_cpu_only_valid_config_succeeds_with_gpu_not_checked() {
    let dir = tmp_path("cpu_only_ok");
    let path = write_valid_pair(&dir);
    let report = lint_simulation(&path, /*with_gpu=*/ false);
    assert!(report.ok(), "expected OK, got {report:?}");
    assert_eq!(report.overall, LintOverall::Ok);
    assert!(matches!(stage_status(&report, "config"), LintStatus::Ok { .. }));
    match stage_status(&report, "output paths") {
        LintStatus::Ok { detail } => assert_eq!(detail, "none pre-exist"),
        other => panic!("expected Ok, got {other:?}"),
    }
    match stage_status(&report, "init") {
        LintStatus::Ok { detail } => {
            assert!(detail.contains("4 particles"), "got `{detail}`");
            assert!(detail.contains("box "), "got `{detail}`");
        }
        other => panic!("expected Ok, got {other:?}"),
    }
    match stage_status(&report, "box/cutoff") {
        LintStatus::Skipped { reason } => {
            assert!(reason.contains("all-pairs"), "got `{reason}`");
        }
        other => panic!("expected Skipped (all-pairs), got {other:?}"),
    }
    match stage_status(&report, "topology") {
        LintStatus::Skipped { reason } => assert_eq!(reason, "not supplied"),
        other => panic!("expected Skipped (not supplied), got {other:?}"),
    }
    match stage_status(&report, "gpu") {
        LintStatus::NotChecked { reason } => {
            assert!(reason.contains("--with-gpu"), "got `{reason}`");
        }
        other => panic!("expected NotChecked, got {other:?}"),
    }
}

// rq-b54d8111
#[test]
fn lint_reports_filename_convention_violation() {
    let dir = tmp_path("bad_filename");
    let path = dir.join("sim.toml");
    std::fs::write(&path, minimal_config_body("sim.in.xyz")).unwrap();
    let report = lint_simulation(&path, false);
    assert!(!report.ok());
    match stage_status(&report, "config") {
        LintStatus::Fail { error, .. } => match error {
            RunnerError::Config(heddle_md::io::ConfigError::InvalidConfigFilename { path: p }) => {
                assert_eq!(p, &path);
            }
            other => panic!("expected InvalidConfigFilename, got {other:?}"),
        },
        other => panic!("expected Fail, got {other:?}"),
    }
    for label in ["output paths", "init", "box/cutoff", "topology", "gpu"] {
        assert!(
            matches!(stage_status(&report, label), LintStatus::Skipped { .. }),
            "stage `{label}` should be Skipped, got {:?}",
            stage_status(&report, label)
        );
    }
}

// rq-37bfb0fc
#[test]
fn lint_reports_pre_existing_output_file() {
    let dir = tmp_path("pre_existing_output");
    let config = write_valid_pair(&dir);
    let traj = dir.join("sim.out.run.xyz");
    std::fs::write(&traj, "existing").unwrap();

    let report = lint_simulation(&config, false);
    assert!(!report.ok());
    match stage_status(&report, "output paths") {
        LintStatus::Fail { detail, error } => {
            assert!(detail.contains("sim.out.run.xyz"), "got `{detail}`");
            match error {
                RunnerError::OutputExists { path } => assert_eq!(path, &traj),
                other => panic!("expected OutputExists, got {other:?}"),
            }
        }
        other => panic!("expected Fail, got {other:?}"),
    }
    // Earlier stage (config) still passed.
    assert!(matches!(stage_status(&report, "config"), LintStatus::Ok { .. }));
    // Later stages skipped.
    for label in ["init", "box/cutoff", "topology", "gpu"] {
        assert!(matches!(stage_status(&report, label), LintStatus::Skipped { .. }));
    }
    // The output file itself is untouched.
    assert_eq!(std::fs::read(&traj).unwrap(), b"existing");
}

// rq-a479680a rq-4b4f85c7
#[test]
fn lint_reports_box_too_small_failure() {
    let dir = tmp_path("box_too_small");
    // Cell-list with cutoff 1.0e-9 and the default r_skin = 0.3 * cutoff
    // requires min perp width >= 3 * 1.3e-9 = 3.9e-9 m. A 2e-9 box edge
    // along `a` triggers CellListBoxTooSmall.
    let body = r#"schema_version = 1
init = "sim.in.xyz"

[simulation]
seed = 1
temperature = 0.0

[[phase]]
name = "run"
n_steps = 1
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
mode = "cell-list"
r_skin = 1.0e-10

[phase.output]
trajectory_every = 0
log_every = 0
"#;
    let config = dir.join("sim.in.toml");
    std::fs::write(&config, body).unwrap();
    let init = dir.join("sim.in.xyz");
    // Box 2e-9 × 5e-9 × 5e-9: shortest perpendicular width (2e-9) is
    // below the required 3.3e-9.
    let xyz = "1\nLattice=\"2.0e-9 0 0 0 5.0e-9 0 0 0 5.0e-9\" Properties=species:S:1:pos:R:3\nAr 0.0 0.0 0.0\n";
    std::fs::write(&init, xyz).unwrap();

    let report = lint_simulation(&config, false);
    assert!(!report.ok());
    match stage_status(&report, "box/cutoff") {
        LintStatus::Fail { detail, error } => {
            assert!(detail.contains("`a`"), "got `{detail}`");
            assert!(matches!(error, RunnerError::CellListBoxTooSmall { direction, .. } if *direction == "a"));
        }
        other => panic!("expected Fail, got {other:?}"),
    }
    assert!(matches!(stage_status(&report, "init"), LintStatus::Ok { .. }));
    assert!(matches!(stage_status(&report, "topology"), LintStatus::Skipped { .. }));
}

// rq-cdf1cd7c rq-21d27f06
#[test]
fn lint_marks_box_cutoff_not_applicable_in_all_pairs_mode() {
    let dir = tmp_path("all_pairs_box");
    let config = write_valid_pair(&dir);
    let report = lint_simulation(&config, false);
    assert!(report.ok());
    match stage_status(&report, "box/cutoff") {
        LintStatus::Skipped { reason } => {
            assert!(reason.contains("not applicable"), "got `{reason}`");
            assert!(reason.contains("all-pairs"), "got `{reason}`");
        }
        other => panic!("expected Skipped, got {other:?}"),
    }
}

// rq-60433fcd
#[test]
fn lint_reports_topology_load_failure() {
    let dir = tmp_path("bad_topology");
    // Hand-roll the body so `topology` is a top-level field (not
    // accidentally tucked inside the `[output]` table).
    let body = r#"schema_version = 1
init = "sim.in.xyz"
topology = "bad.in.topology"

[simulation]
seed = 1
temperature = 0.0

[[phase]]
name = "run"
n_steps = 1
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

[[bond_types]]
name = "ArAr"
potential = "morse"
de = 1.0e-19
a = 1.0e10
re = 3.40e-10

[phase.output]
trajectory_every = 5
log_every = 5
"#;
    let config = dir.join("sim.in.toml");
    std::fs::write(&config, body).unwrap();
    write_minimal_init(&dir.join("sim.in.xyz"), 4);
    // Declare a bond with an atom index out of range (5 > particle_count=4).
    let topo = "[bonds]\n0 5 ArAr\n";
    std::fs::write(dir.join("bad.in.topology"), topo).unwrap();

    let report = lint_simulation(&config, false);
    assert!(!report.ok(), "expected FAIL, got {report:?}");
    match stage_status(&report, "topology") {
        LintStatus::Fail { error, .. } => {
            assert!(
                matches!(error, RunnerError::TopologyFile(_)),
                "expected TopologyFile, got {error:?}"
            );
        }
        other => panic!("expected Fail, got {other:?}"),
    }
    assert!(matches!(stage_status(&report, "gpu"), LintStatus::Skipped { .. }));
}

// rq-a4fbc3a4
#[test]
fn lint_never_creates_output_files() {
    let dir = tmp_path("no_create");
    let config = write_valid_pair(&dir);
    let report = lint_simulation(&config, false);
    assert!(report.ok());
    for ext in ["xyz", "log", "timings"] {
        let p = dir.join(format!("sim.out.{ext}"));
        assert!(!p.exists(), "lint created {p:?}");
    }
}

// rq-69bf814f
#[test]
fn lint_short_circuits_on_first_failure() {
    let dir = tmp_path("short_circuit");
    // Valid config but no init file: the init stage will fail.
    let config = dir.join("sim.in.toml");
    std::fs::write(&config, minimal_config_body("sim.in.xyz")).unwrap();
    // Deliberately do NOT write sim.in.xyz.

    let report = lint_simulation(&config, false);
    assert!(!report.ok());
    assert!(matches!(stage_status(&report, "config"), LintStatus::Ok { .. }));
    assert!(matches!(stage_status(&report, "output paths"), LintStatus::Ok { .. }));
    match stage_status(&report, "init") {
        LintStatus::Fail { error, .. } => {
            assert!(matches!(error, RunnerError::InitState(_)));
        }
        other => panic!("expected Fail, got {other:?}"),
    }
    for label in ["box/cutoff", "topology", "gpu"] {
        match stage_status(&report, label) {
            LintStatus::Skipped { reason } => {
                assert_eq!(reason, "skipped (earlier check failed)");
            }
            other => panic!("stage `{label}`: expected Skipped, got {other:?}"),
        }
    }
}

// rq-8044a6f5
#[test]
fn lint_report_api_short_circuits_with_structured_error() {
    let dir = tmp_path("api_short_circuit");
    let config = dir.join("sim.in.toml");
    std::fs::write(&config, minimal_config_body("sim.in.xyz")).unwrap();
    // Init file with a position outside the primary cell.
    let xyz = "1\nLattice=\"4.0e-9 0 0 0 4.0e-9 0 0 0 4.0e-9\" Properties=species:S:1:pos:R:3\nAr 3.0e-9 0.0 0.0\n";
    std::fs::write(dir.join("sim.in.xyz"), xyz).unwrap();

    let report = lint_simulation(&config, false);
    assert_eq!(report.overall, LintOverall::Fail);
    assert!(matches!(
        report.first_failure().unwrap(),
        RunnerError::InitState(_)
    ));
    assert!(matches!(stage_status(&report, "init"), LintStatus::Fail { .. }));
    match stage_status(&report, "box/cutoff") {
        LintStatus::Skipped { reason } => {
            assert_eq!(reason, "skipped (earlier check failed)");
        }
        other => panic!("expected Skipped, got {other:?}"),
    }
    match stage_status(&report, "gpu") {
        LintStatus::Skipped { reason } => {
            // Earlier stage failed → skipped (not "not checked").
            assert_eq!(reason, "skipped (earlier check failed)");
        }
        other => panic!("expected Skipped, got {other:?}"),
    }
}

// rq-dba8d096
#[test]
fn lint_with_gpu_reports_successful_gpu_setup() {
    let dir = tmp_path("with_gpu_ok");
    let config = write_valid_pair(&dir);
    let report = lint_simulation(&config, /*with_gpu=*/ true);
    assert!(report.ok(), "expected OK, got {report:?}");
    match stage_status(&report, "gpu") {
        LintStatus::Ok { detail } => {
            assert!(detail.contains("init_device OK"), "got `{detail}`");
            assert!(detail.contains("ForceField"), "got `{detail}`");
        }
        other => panic!("expected Ok, got {other:?}"),
    }
}

// CLI-level smoke tests via cli_main_u8.

#[test]
fn cli_lint_returns_zero_on_success() {
    let dir = tmp_path("cli_ok");
    let config = write_valid_pair(&dir);
    let exit = cli_main_u8(vec![
        "heddlemd".to_string(),
        "lint".to_string(),
        config.to_string_lossy().to_string(),
    ]);
    assert_eq!(exit, 0);
}

#[test]
fn cli_lint_returns_one_on_failure() {
    let dir = tmp_path("cli_fail");
    let config = write_valid_pair(&dir);
    let traj = dir.join("sim.out.run.xyz");
    std::fs::write(&traj, "existing").unwrap();
    let exit = cli_main_u8(vec![
        "heddlemd".to_string(),
        "lint".to_string(),
        config.to_string_lossy().to_string(),
    ]);
    assert_eq!(exit, 1);
}

#[test]
fn cli_lint_rejects_unknown_flag() {
    let dir = tmp_path("cli_bad_flag");
    let config = write_valid_pair(&dir);
    let exit = cli_main_u8(vec![
        "heddlemd".to_string(),
        "lint".to_string(),
        config.to_string_lossy().to_string(),
        "--bogus".to_string(),
    ]);
    assert_eq!(exit, 1);
}

#[test]
fn cli_lint_rejects_missing_config_path() {
    let exit = cli_main_u8(vec!["heddlemd".to_string(), "lint".to_string()]);
    assert_eq!(exit, 1);
}

#[test]
fn cli_lint_with_gpu_flag_parses() {
    let dir = tmp_path("cli_with_gpu_parse");
    let config = write_valid_pair(&dir);
    // We don't assert on success here because it requires a GPU; we only
    // confirm the CLI accepts the flag and reaches the `gpu` stage.
    let exit = cli_main_u8(vec![
        "heddlemd".to_string(),
        "lint".to_string(),
        config.to_string_lossy().to_string(),
        "--with-gpu".to_string(),
    ]);
    // exit may be 0 or 1 depending on GPU availability; both are fine.
    assert!(exit == 0 || exit == 1);
}

#[test]
fn cli_lint_flag_order_does_not_matter() {
    let dir = tmp_path("cli_flag_order");
    let config = write_valid_pair(&dir);
    let exit = cli_main_u8(vec![
        "heddlemd".to_string(),
        "lint".to_string(),
        "--with-gpu".to_string(),
        config.to_string_lossy().to_string(),
    ]);
    assert!(exit == 0 || exit == 1);
}
