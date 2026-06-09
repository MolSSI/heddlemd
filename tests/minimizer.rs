// rq-08aba7ee — End-to-end tests for the steepest-descent energy minimizer.
//
// These tests exercise the full runner pipeline against a small
// argon system with an off-equilibrium initial geometry. Each test
// writes a temporary config, runs `run_simulation`, and checks the
// resulting `.minlog` against the expected energy descent.

use std::fs;
use std::path::PathBuf;

use dynamics::io::PhaseKind;
use dynamics::runner::{run_simulation, RunnerError};

fn tmp_dir(suffix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "dynamics-min-{}-{}",
        suffix,
        std::process::id()
    ));
    if dir.exists() {
        let _ = fs::remove_dir_all(&dir);
    }
    fs::create_dir_all(&dir).unwrap();
    dir
}

// Two argon atoms 5% stretched from the LJ minimum: F_max is large
// and points along the bond. SD should decrease the energy and the
// max force at every accepted iteration.
fn two_argon_offset_init() -> &'static str {
    // LJ minimum for Ar-Ar is at 2^(1/6) · σ ≈ 3.816e-10 m. Place
    // atoms at ±1.5e-10 m along x: 3.0e-10 m apart, well inside the
    // repulsive core, so F_max is large (well above the default
    // force_tolerance of 1e-10 N) and SD will reduce the energy over
    // multiple iterations.
    "2\n\
     Lattice=\"5.0e-9 0 0 0 5.0e-9 0 0 0 5.0e-9\" Properties=species:S:1:pos:R:3\n\
     Ar  -1.500000000e-10 0.000000000e0 0.000000000e0\n\
     Ar   1.500000000e-10 0.000000000e0 0.000000000e0\n"
}

fn argon_min_config() -> String {
    r#"schema_version = 1
units = "atomic"
init = "argon.in.xyz"

[simulation]
seed = 1
temperature = 0.0

[[minimization]]
name = "min"

[minimization.algorithm]
kind = "steepest-descent"
initial_step = 1.0e-13
max_step = 1.0e-11
max_iterations = 200

[minimization.output]
minlog_every = 1
trajectory_every = 0

[[particle_types]]
name = "Ar"
mass = 6.6335e-26
charge = 0.0

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 1.5e-9
r_switch = 1.5e-9

[neighbor_list]
mode = "all-pairs"
"#
    .to_string()
}

// rq-f5a4623c rq-b977777a
#[test]
fn steepest_descent_reduces_lj_energy() {
    let dir = tmp_dir("descent_reduces");
    fs::write(dir.join("argon.in.xyz"), two_argon_offset_init()).unwrap();
    let cfg_path = dir.join("argon.in.toml");
    fs::write(&cfg_path, argon_min_config()).unwrap();

    let summary = match run_simulation(&cfg_path) {
        Ok(s) => s,
        Err(e) => panic!("run_simulation failed: {e:?}"),
    };

    assert_eq!(summary.phases.len(), 1);
    let ps = &summary.phases[0];
    assert_eq!(ps.kind, "minimization");
    assert!(
        ps.n_steps > 0 && ps.n_steps <= 200,
        "iterations should be > 0 and <= max_iterations, got {}",
        ps.n_steps
    );

    let minlog_path = dir.join("argon.out.min.minlog");
    let content = fs::read_to_string(&minlog_path).unwrap();
    let mut lines = content.lines();
    assert_eq!(
        lines.next().unwrap(),
        "iter,energy,max_force,step,accepted"
    );
    let rows: Vec<&str> = lines.collect();
    assert!(rows.len() >= 2, "expected step-0 row + at least one more");

    // Energy decreases between the first and last rows.
    let parse_energy = |row: &str| -> f64 {
        row.split(',').nth(1).unwrap().parse::<f64>().unwrap()
    };
    let first_energy = parse_energy(rows.first().unwrap());
    let last_energy = parse_energy(rows.last().unwrap());
    assert!(
        last_energy < first_energy,
        "energy should decrease: first={first_energy} last={last_energy}"
    );
}

// rq-57a0f297
#[test]
fn config_loader_parses_minimization_block() {
    let dir = tmp_dir("min_config_load");
    fs::write(dir.join("argon.in.xyz"), two_argon_offset_init()).unwrap();
    let cfg_path = dir.join("argon.in.toml");
    fs::write(&cfg_path, argon_min_config()).unwrap();

    let cfg = dynamics::io::load_config(&cfg_path).unwrap();
    assert_eq!(cfg.phases.len(), 1);
    let min = match &cfg.phases[0] {
        PhaseKind::Minimization(m) => m,
        _ => panic!("expected a minimization phase"),
    };
    assert_eq!(min.name, "min");
    assert_eq!(min.algorithm.kind, "steepest-descent");
    assert_eq!(min.output.minlog_every, 1);
    assert_eq!(min.output.trajectory_every, 0);
}

// rq-09ed630b
#[test]
fn interleaved_phase_and_minimization_preserve_document_order() {
    let dir = tmp_dir("interleaved_order");
    fs::write(dir.join("argon.in.xyz"), two_argon_offset_init()).unwrap();
    let body = r#"schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 1
temperature = 100.0

[[phase]]
name = "equil"
n_steps = 1
dt = 1.0e-15

[phase.integrator]
kind = "velocity-verlet"
lossless = false

[[minimization]]
name = "min"

[minimization.algorithm]
kind = "steepest-descent"

[[phase]]
name = "prod"
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
cutoff = 1.5e-9

[neighbor_list]
mode = "all-pairs"
"#;
    let cfg_path = dir.join("argon.in.toml");
    fs::write(&cfg_path, body).unwrap();

    let cfg = dynamics::io::load_config(&cfg_path).unwrap();
    assert_eq!(cfg.phases.len(), 3);
    assert!(matches!(&cfg.phases[0], PhaseKind::Md(p) if p.name == "equil"));
    assert!(matches!(&cfg.phases[1], PhaseKind::Minimization(m) if m.name == "min"));
    assert!(matches!(&cfg.phases[2], PhaseKind::Md(p) if p.name == "prod"));
}

// rq-33b1f2d2
#[test]
fn duplicate_phase_name_across_arrays_is_rejected() {
    let dir = tmp_dir("dup_phase_name");
    fs::write(dir.join("argon.in.xyz"), two_argon_offset_init()).unwrap();
    let body = r#"schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 1
temperature = 100.0

[[phase]]
name = "step1"
n_steps = 1
dt = 1.0e-15

[phase.integrator]
kind = "velocity-verlet"

[[minimization]]
name = "step1"

[minimization.algorithm]
kind = "steepest-descent"

[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 1.5e-9

[neighbor_list]
mode = "all-pairs"
"#;
    let cfg_path = dir.join("argon.in.toml");
    fs::write(&cfg_path, body).unwrap();

    let err = dynamics::io::load_config(&cfg_path).err().unwrap();
    assert!(
        matches!(
            err,
            dynamics::io::ConfigError::DuplicatePhaseName { ref name }
                if name == "step1"
        ),
        "got: {err:?}"
    );
}

// rq-69109415
#[test]
fn unknown_minimization_kind_rejected() {
    let dir = tmp_dir("unknown_min_kind");
    fs::write(dir.join("argon.in.xyz"), two_argon_offset_init()).unwrap();
    let body = r#"schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 1
temperature = 0.0

[[minimization]]
name = "min"

[minimization.algorithm]
kind = "quasi-newton"

[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 1.5e-9

[neighbor_list]
mode = "all-pairs"
"#;
    let cfg_path = dir.join("argon.in.toml");
    fs::write(&cfg_path, body).unwrap();

    let err = dynamics::io::load_config(&cfg_path).err().unwrap();
    assert!(
        matches!(
            err,
            dynamics::io::ConfigError::UnknownKind { slot: "minimization", ref kind }
                if kind == "quasi-newton"
        ),
        "got: {err:?}"
    );
}

// rq-30d3f1fa
#[test]
fn invalid_step_decrease_rejected_at_load() {
    let dir = tmp_dir("invalid_step_decrease");
    fs::write(dir.join("argon.in.xyz"), two_argon_offset_init()).unwrap();
    let body = r#"schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 1
temperature = 0.0

[[minimization]]
name = "min"

[minimization.algorithm]
kind = "steepest-descent"
step_decrease = 1.0

[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 1.5e-9

[neighbor_list]
mode = "all-pairs"
"#;
    let cfg_path = dir.join("argon.in.toml");
    fs::write(&cfg_path, body).unwrap();

    let err = dynamics::io::load_config(&cfg_path).err().unwrap();
    assert!(
        matches!(
            err,
            dynamics::io::ConfigError::InvalidValue { ref field, .. }
                if field == "minimization.algorithm.step_decrease"
        ),
        "got: {err:?}"
    );
}

// rq-90a67daf
#[test]
fn non_convergence_returns_hard_error() {
    let dir = tmp_dir("non_convergence");
    fs::write(dir.join("argon.in.xyz"), two_argon_offset_init()).unwrap();
    let body = r#"schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 1
temperature = 0.0

[[minimization]]
name = "min"

[minimization.algorithm]
kind = "steepest-descent"
# Tiny step and tight force tolerance so 3 iterations can't possibly
# satisfy the criterion.
initial_step = 1.0e-20
max_step = 1.0e-20
step_increase = 1.0
step_decrease = 0.5
force_tolerance = 1.0e-30
energy_tolerance = 0.0
max_iterations = 3

[minimization.output]
minlog_every = 1

[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 1.5e-9

[neighbor_list]
mode = "all-pairs"
"#;
    let cfg_path = dir.join("argon.in.toml");
    fs::write(&cfg_path, body).unwrap();

    let err = run_simulation(&cfg_path).err().unwrap();
    match err {
        RunnerError::MinimizerNonConvergence { phase, iterations, .. } => {
            assert_eq!(phase, "min");
            assert_eq!(iterations, 3);
        }
        other => panic!("expected MinimizerNonConvergence, got: {other:?}"),
    }
}
