// rq-08aba7ee — End-to-end tests for the steepest-descent energy minimizer.
//
// These tests exercise the full runner pipeline against a small
// argon system with an off-equilibrium initial geometry. Each test
// writes a temporary config, runs `run_simulation`, and checks the
// resulting `.minlog` against the expected energy descent.

use std::fs;
use std::path::{Path, PathBuf};

use heddle_md::io::PhaseKind;
use heddle_md::runner::{run_simulation, RunnerError};
use heddle_md::precision::Real;

fn tmp_dir(suffix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "heddlemd-min-{}-{}",
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

    let cfg = heddle_md::io::load_config(&cfg_path).unwrap();
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

    let cfg = heddle_md::io::load_config(&cfg_path).unwrap();
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

    let err = heddle_md::io::load_config(&cfg_path).err().unwrap();
    assert!(
        matches!(
            err,
            heddle_md::io::ConfigError::DuplicatePhaseName { ref name }
                if name == "step1"
        ),
        "got: {err:?}"
    );
}

// rq-69109415 rq-3e0a3040
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

    let err = heddle_md::io::load_config(&cfg_path).err().unwrap();
    assert!(
        matches!(
            err,
            heddle_md::io::ConfigError::UnknownKind { slot: "minimization", ref kind }
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

    let err = heddle_md::io::load_config(&cfg_path).err().unwrap();
    assert!(
        matches!(
            err,
            heddle_md::io::ConfigError::InvalidValue { ref field, .. }
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

// rq-bcc16e10
#[test]
fn reject_unknown_field_under_minimization_algorithm() {
    let dir = tmp_dir("unknown_field_min_algo");
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
junk_field = 17

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
    let err = heddle_md::io::load_config(&cfg_path).err().unwrap();
    let s = format!("{err}");
    assert!(
        s.contains("junk_field") || s.contains("unknown field"),
        "expected an unknown-field error mentioning junk_field, got: {s}"
    );
}

// rq-a82a275a
#[test]
fn reject_max_step_less_than_initial_step() {
    let dir = tmp_dir("max_step_lt_initial");
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
initial_step = 1.0e-10
max_step = 1.0e-12

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
    let err = heddle_md::io::load_config(&cfg_path).err().unwrap();
    assert!(
        matches!(
            err,
            heddle_md::io::ConfigError::InvalidValue { ref field, .. }
                if field.contains("max_step") || field.contains("initial_step")
        ),
        "got: {err:?}"
    );
}

// rq-5cbcbabe
#[test]
fn reject_thermostat_under_minimization_phase() {
    let dir = tmp_dir("thermostat_under_min");
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

[minimization.thermostat]
kind = "csvr"
temperature = 300.0
tau = 1.0e-13
seed = 11

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
    let err = heddle_md::io::load_config(&cfg_path).err().unwrap();
    let s = format!("{err}");
    assert!(
        s.contains("thermostat") || s.contains("unknown field"),
        "expected an error mentioning thermostat, got: {s}"
    );
}

// rq-881bb997
#[test]
fn sd_with_no_projection_constraint_slot_rejected_at_config_load() {
    use std::sync::Arc;
    use cudarc::driver::CudaDevice;
    use heddle_md::Registries;
    use heddle_md::forces::{ConstraintGroup as _ConstraintGroup, ConstraintList, GroupConstraint};
    use heddle_md::gpu::GpuContext;
    use heddle_md::integrator::{
        Constraint, ConstraintBuilder, ConstraintError, ConstraintRegistry,
    };
    use heddle_md::io::config::{ConfigError, NamedSlotConfig};
    use heddle_md::io::load_config_raw;

    // Custom constraint builder whose supports_position_projection_only
    // returns false. The runner's minimization-compatibility check must
    // surface this as IncompatibleConstraint at config-load time.
    #[derive(Debug, Clone)]
    struct NoProjectionBuilder;
    impl ConstraintBuilder for NoProjectionBuilder {
        fn kind_name(&self) -> &'static str { "fictional-cluster" }
        fn validate_params(&self, _p: &toml::Value) -> Result<(), ConfigError> { Ok(()) }
        fn supports_position_projection_only(&self, _p: &toml::Value) -> bool { false }
        fn expected_atom_count(&self, _p: &toml::Value) -> usize { 1 }
        fn validate_group_shape(
            &self,
            _gi: usize,
            _a: &[u32],
            _c: &[GroupConstraint],
            _p: &toml::Value,
            _m: &[Real],
        ) -> Result<(), ConstraintError> { Ok(()) }
        fn build(
            &self,
            _device: Arc<CudaDevice>,
            _gpu: &GpuContext,
            _particle_count: usize,
            _list: &ConstraintList,
            _masses: &[Real],
            _constraint_types: &[NamedSlotConfig],
        ) -> Result<Box<dyn Constraint>, ConstraintError> {
            unreachable!("rejected before construction")
        }
        fn box_clone(&self) -> Box<dyn ConstraintBuilder> {
            Box::new(self.clone())
        }
    }

    let dir = tmp_dir("sd_no_projection_constraint");
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

[[constraint_types]]
name = "Cluster"
kind = "fictional-cluster"

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
    let mut registries = Registries::with_builtins();
    registries.constraint_types = ConstraintRegistry::new();
    registries.constraint_types.register(Box::new(NoProjectionBuilder));
    let cfg = load_config_raw(&cfg_path).unwrap();
    // validate_against alone doesn't trigger the projection-support
    // check; the runner reaches it via validate_constraint_compatibility
    // after the topology file is parsed (which is what surfaces the
    // declared constraints to the cross-check). Drive that path
    // directly with has_constraints=true.
    cfg.validate_against(&registries).unwrap();
    let err = cfg
        .validate_constraint_compatibility(&registries, true)
        .err()
        .unwrap();
    match err {
        ConfigError::IncompatibleConstraint { integrator, phase } => {
            assert_eq!(integrator, "steepest-descent");
            assert_eq!(phase, "min");
        }
        other => panic!("expected IncompatibleConstraint, got {other:?}"),
    }
}

// rq-09c8a503
#[test]
fn custom_minimizer_builder_is_selectable() {
    use heddle_md::gpu::{GpuContext, ParticleBuffers};
    use heddle_md::minimizer::{
        Minimizer, MinimizerBuilder, MinimizerConvergence, MinimizerError,
        MinimizerRegistry, MinimizerStepReport,
    };
    use heddle_md::pbc::SimulationBox;
    use heddle_md::timings::Timings;
    use heddle_md::io::SlotConfig;
    use heddle_md::forces::ForceField;

    use heddle_md::integrator::Constraint;
    #[derive(Debug)]
    struct StubMinimizer;
    impl Minimizer for StubMinimizer {
        fn step(
            &mut self,
            _buffers: &mut ParticleBuffers,
            _sim_box: &SimulationBox,
            _force_field: &mut ForceField,
            _constraint: Option<&mut dyn Constraint>,
            _timings: &mut Timings,
        ) -> Result<MinimizerStepReport, MinimizerError> {
            unreachable!("not run")
        }
        fn initial_state(
            &mut self,
            _buffers: &mut ParticleBuffers,
            _timings: &mut Timings,
        ) -> Result<(f64, f64), MinimizerError> {
            unreachable!("not run")
        }
        fn check_convergence(
            &self,
            _report: &MinimizerStepReport,
        ) -> Option<MinimizerConvergence> {
            None
        }
        fn max_iterations(&self) -> u64 { 0 }
    }
    #[derive(Debug, Clone)]
    struct StubBuilder;
    impl MinimizerBuilder for StubBuilder {
        fn kind_name(&self) -> &'static str { "test-stub" }
        fn validate_params(&self, _p: &toml::Value) -> Result<(), heddle_md::io::config::ConfigError> {
            Ok(())
        }
        fn build(
            &self,
            _gpu: &GpuContext,
            _particle_count: usize,
            _n_constraints: usize,
            _params: &toml::Value,
        ) -> Result<Box<dyn Minimizer>, MinimizerError> {
            Ok(Box::new(StubMinimizer))
        }
        fn box_clone(&self) -> Box<dyn MinimizerBuilder> {
            Box::new(self.clone())
        }
    }
    let mut registry = MinimizerRegistry::with_builtins();
    registry.register(Box::new(StubBuilder));
    let builder = registry.lookup("test-stub").unwrap();
    assert_eq!(builder.kind_name(), "test-stub");
    // Confirm built-ins still present.
    assert!(registry.lookup("steepest-descent").is_some());
    // Unknown kind reports UnknownKind via lookup.
    assert!(registry.lookup("not-a-real-kind").is_none());
    // Build the stub via the registry's dispatch.
    let gpu = heddle_md::gpu::init_device().unwrap();
    let slot = SlotConfig::from_params_str("test-stub", "");
    let _ = registry.build(&slot, &gpu, 0, 0).unwrap();
}

// Shared helper: write a SD config templated by per-test overrides.
// Same as argon_min_config_with but with configurable minlog_every and
// optional trajectory_every for output-shape tests.
fn argon_min_config_full(
    initial_step: f64,
    max_step: f64,
    force_tolerance: f64,
    energy_tolerance: f64,
    max_iterations: u64,
    minlog_every: u64,
    trajectory_every: u64,
) -> String {
    format!(
        r#"schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 1
temperature = 0.0

[[minimization]]
name = "min"

[minimization.algorithm]
kind = "steepest-descent"
initial_step = {initial_step:.16e}
max_step = {max_step:.16e}
force_tolerance = {force_tolerance:.16e}
energy_tolerance = {energy_tolerance:.16e}
max_iterations = {max_iterations}

[minimization.output]
minlog_every = {minlog_every}
trajectory_every = {trajectory_every}

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
"#
    )
}

fn argon_min_config_with(
    initial_step: f64,
    max_step: f64,
    force_tolerance: f64,
    energy_tolerance: f64,
    max_iterations: u64,
    init_separation_m: f64,
) -> String {
    format!(
        r#"schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 1
temperature = 0.0

[[minimization]]
name = "min"

[minimization.algorithm]
kind = "steepest-descent"
initial_step = {initial_step:.16e}
max_step = {max_step:.16e}
force_tolerance = {force_tolerance:.16e}
energy_tolerance = {energy_tolerance:.16e}
max_iterations = {max_iterations}

[minimization.output]
minlog_every = 1
trajectory_every = 0

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
"#,
        initial_step = initial_step,
        max_step = max_step,
        force_tolerance = force_tolerance,
        energy_tolerance = energy_tolerance,
        max_iterations = max_iterations,
    )
    .replace("init = \"argon.in.xyz\"", &format!("init = \"argon.in.xyz\"\n# r0 = {init_separation_m:.3e}"))
}

fn write_argon_pair_init(dir: &Path, separation_m: f64) {
    let body = format!(
        "2\nLattice=\"1.0e-8 0 0 0 1.0e-8 0 0 0 1.0e-8\" Properties=species:S:1:pos:R:3\nAr 0 0 0\nAr {separation_m:.16e} 0 0\n"
    );
    fs::write(dir.join("argon.in.xyz"), body).unwrap();
}

// rq-77d98b9c
#[test]
fn sd_force_tolerance_terminates_phase() {
    // Initial state at LJ minimum (r ≈ 2^(1/6) · sigma ≈ 3.82e-10 m).
    // Forces are tiny; with a huge tolerance the phase terminates at
    // iter 0 with reason ForceTolerance.
    let dir = tmp_dir("sd_force_tol");
    let r_min = 3.82e-10_f64;
    write_argon_pair_init(&dir, r_min);
    let cfg = argon_min_config_with(1.0e-13, 1.0e-11, 1.0e10, 0.0, 10, r_min);
    let cfg_path = dir.join("argon.in.toml");
    fs::write(&cfg_path, cfg).unwrap();
    let summary = run_simulation(&cfg_path).unwrap();
    let ps = &summary.phases[0];
    assert_eq!(ps.kind, "minimization");
    assert_eq!(ps.convergence, Some("force_tolerance"));
    assert_eq!(ps.n_steps, 0);
}

// rq-45e6b20a
#[test]
fn sd_energy_tolerance_terminates_phase() {
    // With force_tolerance = 0.0 (disabled) and a loose energy_tolerance,
    // a near-equilibrium starting point produces consecutive accepted
    // iterations whose energy difference satisfies the tolerance.
    let dir = tmp_dir("sd_energy_tol");
    write_argon_pair_init(&dir, 3.82e-10);
    // initial_step small, energy_tolerance huge.
    let cfg = argon_min_config_with(1.0e-14, 1.0e-12, 0.0, 1.0e10, 10, 3.82e-10);
    let cfg_path = dir.join("argon.in.toml");
    fs::write(&cfg_path, cfg).unwrap();
    let summary = run_simulation(&cfg_path).unwrap();
    let ps = &summary.phases[0];
    assert_eq!(ps.convergence, Some("energy_tolerance"));
}

// rq-901c006f
#[test]
fn sd_zero_initial_force_triggers_force_zero() {
    // A lone particle has zero force exactly under any pair potential
    // (no neighbors). The SD phase terminates at iter 0 with ForceZero.
    let dir = tmp_dir("sd_force_zero");
    let body = "1\nLattice=\"1.0e-8 0 0 0 1.0e-8 0 0 0 1.0e-8\" \
                Properties=species:S:1:pos:R:3\nAr 0 0 0\n";
    fs::write(dir.join("argon.in.xyz"), body).unwrap();
    let cfg = argon_min_config_with(1.0e-13, 1.0e-11, 1.0e-30, 0.0, 10, 0.0);
    let cfg_path = dir.join("argon.in.toml");
    fs::write(&cfg_path, cfg).unwrap();
    let summary = run_simulation(&cfg_path).unwrap();
    let ps = &summary.phases[0];
    assert_eq!(ps.convergence, Some("force_zero"));
    assert_eq!(ps.n_steps, 0);
}

// rq-0165303c
#[test]
fn sd_on_empty_buffers_converges_immediately_with_force_zero() {
    // Zero-particle init.xyz; SD must accept the empty state and
    // converge at iter 0 with reason ForceZero (F_max == 0 trivially).
    let dir = tmp_dir("sd_empty_buffers");
    let body = "0\nLattice=\"1.0e-8 0 0 0 1.0e-8 0 0 0 1.0e-8\" Properties=species:S:1:pos:R:3\n";
    fs::write(dir.join("argon.in.xyz"), body).unwrap();
    let cfg = argon_min_config_with(1.0e-13, 1.0e-11, 1.0e-30, 0.0, 10, 0.0);
    let cfg_path = dir.join("argon.in.toml");
    fs::write(&cfg_path, cfg).unwrap();
    let summary = run_simulation(&cfg_path).unwrap();
    let ps = &summary.phases[0];
    assert_eq!(ps.convergence, Some("force_zero"));
    assert_eq!(ps.n_steps, 0);
}

// rq-2e27a5fc
#[test]
fn sd_energy_tolerance_does_not_fire_on_rejected_step() {
    // Force a rejection on iteration 1 by setting initial_step large
    // enough to overshoot the LJ well. The runner records the rejected
    // row in the .minlog with energy unchanged from iter 0; if
    // check_convergence were called on that row pair, the relative
    // energy difference would be 0 and would trivially satisfy a
    // permissive energy_tolerance. We assert that does NOT happen:
    // the phase keeps iterating past the rejected step.
    let dir = tmp_dir("sd_energy_tol_rejected");
    write_argon_pair_init(&dir, 3.5e-10); // off equilibrium
    let cfg = argon_min_config_with(1.0e-9, 1.0e-9, 0.0, 1.0e-2, 20, 3.5e-10);
    let cfg_path = dir.join("argon.in.toml");
    fs::write(&cfg_path, cfg).unwrap();
    let summary = run_simulation(&cfg_path).unwrap();
    let ps = &summary.phases[0];
    // Read the minlog and confirm at least one rejected row (accepted=0)
    // exists and that the phase did NOT terminate immediately after it.
    let minlog = fs::read_to_string(dir.join("argon.out.min.minlog")).unwrap();
    let rows: Vec<&str> = minlog.lines().skip(1).collect();
    let rejected_iters: Vec<usize> = rows.iter().enumerate()
        .filter_map(|(i, row)| {
            let acc: u8 = row.split(',').last().unwrap().parse().unwrap();
            if acc == 0 { Some(i) } else { None }
        })
        .collect();
    assert!(!rejected_iters.is_empty(), "test setup must produce at least one rejected iteration");
    // For the rejected step not to falsely trigger EnergyTolerance, the
    // phase must continue past it (more rows recorded after).
    let last_rejected = *rejected_iters.last().unwrap();
    assert!(
        rows.len() > last_rejected + 1,
        "phase terminated immediately after a rejected step — EnergyTolerance fired falsely; total_rows={} last_rejected={}",
        rows.len(),
        last_rejected
    );
    // Sanity: phase didn't error out.
    assert_eq!(ps.kind, "minimization");
}

// Unit-level SD harness for Wave 5 step-formula tests. The engine uses
// atomic units internally; this helper takes SI inputs, converts to
// atomic units up front, builds the SD minimizer + an LJ-only
// ForceField against a 2-argon system, and exposes step() directly.
fn build_sd_argon_pair(
    initial_step_m: f64,
    max_step_m: f64,
    step_increase: f64,
    step_decrease: f64,
    force_tolerance_n: f64,
    energy_tolerance_rel: f64,
    pair_separation_m: f64,
) -> (
    heddle_md::gpu::GpuContext,
    Box<dyn heddle_md::minimizer::Minimizer>,
    heddle_md::forces::ForceField,
    heddle_md::gpu::ParticleBuffers,
    heddle_md::pbc::SimulationBox,
    heddle_md::timings::Timings,
) {
    use heddle_md::forces::{ForceField, PotentialRegistry, BondList, AngleList, ExclusionList};
    use heddle_md::gpu::{ParticleBuffers, init_device};
    use heddle_md::io::config::{NeighborListConfig, PairInteractionConfig, PairPotentialParams, ParticleTypeConfig};
    use heddle_md::minimizer::MinimizerRegistry;
    use heddle_md::pbc::SimulationBox;
    use heddle_md::state::ParticleState;
    use heddle_md::timings::Timings;
    use heddle_md::units::{Dimension, UnitSystem};

    let len_f = UnitSystem::Si.factor(Dimension::Length);
    let mass_f = UnitSystem::Si.factor(Dimension::Mass);
    let energy_f = UnitSystem::Si.factor(Dimension::Energy);
    let force_f = UnitSystem::Si.factor(Dimension::Force);

    // SI argon LJ parameters → atomic units.
    let sigma_au = 3.40e-10_f64 / len_f;
    let epsilon_au = 1.65e-21_f64 / energy_f;
    let cutoff_au = 1.5e-9_f64 / len_f;
    let mass_au = 6.6335e-26_f64 / mass_f;
    let r_au = pair_separation_m / len_f;

    let initial_step_au = initial_step_m / len_f;
    let max_step_au = max_step_m / len_f;
    let force_tolerance_au = force_tolerance_n / force_f;

    let gpu = init_device().unwrap();
    let box_au = 1.0e-7 / len_f;
    let sim_box = SimulationBox::new(box_au as Real, box_au as Real, box_au as Real, 0.0, 0.0, 0.0).unwrap();
    let particle_state = ParticleState::new(
        vec![0.0, r_au as Real],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![mass_au as Real, mass_au as Real],
        vec![0.0, 0.0],
        vec![0u32, 0u32],
        None,
        None,
    )
    .unwrap();
    let buffers = ParticleBuffers::new(&gpu, &particle_state).unwrap();
    let ff = ForceField::new(
        &PotentialRegistry::with_builtins(),
        &gpu,
        2,
        &sim_box,
        &[ParticleTypeConfig {
            name: "Ar".to_string(),
            mass: mass_au,
            charge: 0.0,
        }],
        &[PairInteractionConfig {
            between: ("Ar".to_string(), "Ar".to_string()),
            cutoff: cutoff_au,
            r_switch: cutoff_au,
            potential: PairPotentialParams::LennardJones {
                sigma: sigma_au,
                epsilon: epsilon_au,
            },
        }],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(2),
        &AngleList::empty(0),
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let timings = Timings::new(&gpu).unwrap();
    let registry = MinimizerRegistry::with_builtins();
    let slot = heddle_md::io::SlotConfig::from_params_str(
        "steepest-descent",
        &format!(
            "initial_step = {initial_step_au:.16e}\n\
             max_step = {max_step_au:.16e}\n\
             step_increase = {step_increase}\n\
             step_decrease = {step_decrease}\n\
             force_tolerance = {force_tolerance_au:.16e}\n\
             energy_tolerance = {energy_tolerance_rel:.16e}\n\
             max_iterations = 100\n"
        ),
    );
    let sd = registry.build(&slot, &gpu, 2, 0).unwrap();
    (gpu, sd, ff, buffers, sim_box, timings)
}

// Populate forces_* / potential_energies on the buffers (the runner's
// pre-loop warm-up). SD's initial_state and the per-iteration accept
// check both consume these.
fn warm_up_forces(
    ff: &mut heddle_md::forces::ForceField,
    buffers: &mut heddle_md::gpu::ParticleBuffers,
    sim_box: &heddle_md::pbc::SimulationBox,
    timings: &mut heddle_md::timings::Timings,
) {
    ff.step(
        buffers,
        sim_box,
        timings,
        heddle_md::forces::AggregateLevel::ForcesAndScalars,
    )
    .unwrap();
}

// rq-420d2eb3
#[test]
fn sd_reduces_lj_energy_in_one_accepted_step() {
    // Two argons at 3.5 Å (slightly compressed below sigma = 3.40 Å);
    // forces are strongly repulsive. One SD iteration with a small
    // initial_step is guaranteed accepted and reduces the energy.
    let (gpu, mut sd, mut ff, mut buffers, sim_box, mut timings) =
        build_sd_argon_pair(1.0e-13, 1.0e-11, 1.2, 0.2, 0.0, 0.0, 3.5e-10);
    let _ = gpu;
    warm_up_forces(&mut ff, &mut buffers, &sim_box, &mut timings);
    let (e0, _) = sd.initial_state(&mut buffers, &mut timings).unwrap();
    let report = sd
        .step(&mut buffers, &sim_box, &mut ff, None, &mut timings)
        .unwrap();
    assert!(report.accepted, "first SD step on compressed pair should be accepted");
    assert!(
        report.energy < e0,
        "step should reduce energy; e0 = {e0}, e1 = {}",
        report.energy
    );
}

// rq-48e36163
#[test]
fn sd_step_formula_moves_largest_force_atom_by_exactly_step() {
    // Symmetric 2-atom argon pair: forces on the two atoms are equal in
    // magnitude and opposite in sign. By the SD step formula
    //   Δx_i = step · F_i / F_max
    // each atom's |Δx| equals step. Verify the post-iteration positions
    // by comparing pre/post via the buffers.
    let (gpu, mut sd, mut ff, mut buffers, sim_box, mut timings) =
        build_sd_argon_pair(1.0e-13, 1.0e-11, 1.2, 0.2, 0.0, 0.0, 3.5e-10);
    let pre_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    warm_up_forces(&mut ff, &mut buffers, &sim_box, &mut timings);
    let _ = sd.initial_state(&mut buffers, &mut timings).unwrap();
    let report = sd
        .step(&mut buffers, &sim_box, &mut ff, None, &mut timings)
        .unwrap();
    assert!(report.accepted);
    let post_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let dx0 = (post_x[0] - pre_x[0]) as f64;
    let dx1 = (post_x[1] - pre_x[1]) as f64;
    let step = report.step_size;
    let rel = 1e-3;
    // |dx_i| should equal step (within f32 round-off + SD's two-pass
    // computation tolerance).
    assert!(
        (dx0.abs() - step).abs() <= rel * step,
        "atom 0 |Δx| = {} vs step {}",
        dx0.abs(),
        step
    );
    assert!(
        (dx1.abs() - step).abs() <= rel * step,
        "atom 1 |Δx| = {} vs step {}",
        dx1.abs(),
        step
    );
    // The atoms have opposite-sign forces, so the displacements have
    // opposite sign.
    assert!(dx0 * dx1 < 0.0, "atoms should move in opposite directions");
}

// rq-265de297
#[test]
fn sd_step_doubles_on_accepted_iteration() {
    use heddle_md::units::{Dimension, UnitSystem};
    let init = 1.0e-13_f64;
    let max = 1.0e-9_f64;
    let len_f = UnitSystem::Si.factor(Dimension::Length);
    let init_au = init / len_f;
    let (_, mut sd, mut ff, mut buffers, sim_box, mut timings) =
        build_sd_argon_pair(init, max, 2.0, 0.5, 0.0, 0.0, 3.5e-10);
    warm_up_forces(&mut ff, &mut buffers, &sim_box, &mut timings);
    let _ = sd.initial_state(&mut buffers, &mut timings).unwrap();
    let r1 = sd
        .step(&mut buffers, &sim_box, &mut ff, None, &mut timings)
        .unwrap();
    assert!(r1.accepted);
    assert!((r1.step_size - init_au).abs() <= 1e-6 * init_au);
    let r2 = sd
        .step(&mut buffers, &sim_box, &mut ff, None, &mut timings)
        .unwrap();
    assert!(r2.accepted);
    let expected_au = 2.0 * init_au;
    assert!(
        (r2.step_size - expected_au).abs() <= 1e-6 * expected_au,
        "second iteration step {} should be 2 × initial_step {} (atomic units)",
        r2.step_size,
        init_au
    );
}

// rq-ba6d3eaa
#[test]
fn sd_step_caps_at_max_step() {
    use heddle_md::units::{Dimension, UnitSystem};
    let init = 8.0e-12_f64;
    let max = 1.0e-11_f64;
    let len_f = UnitSystem::Si.factor(Dimension::Length);
    let max_au = max / len_f;
    let (_, mut sd, mut ff, mut buffers, sim_box, mut timings) =
        build_sd_argon_pair(init, max, 10.0, 0.5, 0.0, 0.0, 3.5e-10);
    warm_up_forces(&mut ff, &mut buffers, &sim_box, &mut timings);
    let _ = sd.initial_state(&mut buffers, &mut timings).unwrap();
    let r1 = sd
        .step(&mut buffers, &sim_box, &mut ff, None, &mut timings)
        .unwrap();
    assert!(r1.accepted);
    let r2 = sd
        .step(&mut buffers, &sim_box, &mut ff, None, &mut timings)
        .unwrap();
    assert!(r2.accepted);
    assert!(
        (r2.step_size - max_au).abs() <= 1e-6 * max_au,
        "expected step capped at max_step = {max_au} (atomic units), got {}",
        r2.step_size
    );
}

// rq-b10ed5ec
#[test]
fn sd_step_halves_and_restores_positions_on_rejection() {
    use heddle_md::units::{Dimension, UnitSystem};
    let init = 1.0e-9_f64;
    let max = 1.0e-9_f64;
    let step_decrease = 0.5_f64;
    let len_f = UnitSystem::Si.factor(Dimension::Length);
    let init_au = init / len_f;
    // At r = 3.5e-10 (compressed), the initial step massively overshoots.
    let (gpu, mut sd, mut ff, mut buffers, sim_box, mut timings) =
        build_sd_argon_pair(init, max, 1.0, step_decrease, 0.0, 0.0, 3.5e-10);
    let pre_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    warm_up_forces(&mut ff, &mut buffers, &sim_box, &mut timings);
    let _ = sd.initial_state(&mut buffers, &mut timings).unwrap();
    let r1 = sd
        .step(&mut buffers, &sim_box, &mut ff, None, &mut timings)
        .unwrap();
    assert!(!r1.accepted, "first step should be rejected (initial_step too large)");
    let post_x: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    assert_eq!(post_x, pre_x, "positions should be restored byte-for-byte on rejection");
    // The next iteration's step has the decrease applied.
    let r2 = sd
        .step(&mut buffers, &sim_box, &mut ff, None, &mut timings)
        .unwrap();
    let expected_au = init_au * step_decrease;
    assert!(
        (r2.step_size - expected_au).abs() <= 1e-6 * expected_au,
        "next-iteration step should be initial_step × step_decrease = {expected_au} (atomic units), got {}",
        r2.step_size
    );
}

// rq-9d165c7f
#[test]
fn sd_minlog_every_zero_disables_the_file() {
    let dir = tmp_dir("sd_minlog_every_zero");
    write_argon_pair_init(&dir, 3.82e-10);
    let cfg = argon_min_config_full(1.0e-13, 1.0e-11, 1.0e10, 0.0, 5, 0, 0);
    let cfg_path = dir.join("argon.in.toml");
    fs::write(&cfg_path, cfg).unwrap();
    let _ = run_simulation(&cfg_path).unwrap();
    let minlog_path = dir.join("argon.out.min.minlog");
    assert!(!minlog_path.exists(), "expected no .minlog file with minlog_every=0");
}

// rq-85e9b9da
#[test]
fn sd_minlog_includes_the_final_convergence_iteration_even_off_cadence() {
    // minlog_every = 3, but the phase terminates at an iter that is not
    // a multiple of 3. Verify the final convergence iteration row is
    // written regardless.
    let dir = tmp_dir("sd_minlog_off_cadence");
    write_argon_pair_init(&dir, 3.6e-10);
    // Loose force tolerance so the phase converges within a few
    // iterations on the SD-driven argon pair.
    let cfg = argon_min_config_full(1.0e-13, 1.0e-11, 1.0e-12, 0.0, 50, 3, 0);
    let cfg_path = dir.join("argon.in.toml");
    fs::write(&cfg_path, cfg).unwrap();
    let summary = run_simulation(&cfg_path).unwrap();
    let n_steps = summary.phases[0].n_steps;
    let minlog = fs::read_to_string(dir.join("argon.out.min.minlog")).unwrap();
    let rows: Vec<&str> = minlog.lines().skip(1).collect();
    // Rows include iter 0 plus every multiple of 3 up through n_steps;
    // the final row must report the convergence iteration (n_steps),
    // regardless of cadence.
    let final_iter: u64 = rows
        .last()
        .unwrap()
        .split(',')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(
        final_iter, n_steps,
        ".minlog's final row should report iter == n_steps ({n_steps}) regardless of minlog_every; got {final_iter}"
    );
}

// rq-c4aea411
#[test]
fn sd_minlog_reports_accepted_zero_for_rejected_iterations() {
    // Force at least one rejection by making initial_step too large for
    // the LJ well. Verify the rejected row has accepted=0 and the same
    // energy as the prior accepted row.
    let dir = tmp_dir("sd_minlog_accepted_zero");
    write_argon_pair_init(&dir, 3.5e-10);
    // Same parameters as the Wave 2 rejected-step test, which is known
    // to produce at least one rejected iteration.
    let cfg = argon_min_config_with(1.0e-9, 1.0e-9, 0.0, 1.0e-2, 20, 3.5e-10);
    let cfg_path = dir.join("argon.in.toml");
    fs::write(&cfg_path, cfg).unwrap();
    let _ = run_simulation(&cfg_path).unwrap();
    let minlog = fs::read_to_string(dir.join("argon.out.min.minlog")).unwrap();
    let rows: Vec<&str> = minlog.lines().skip(1).collect();
    let mut found_rejected_with_unchanged_energy = false;
    for w in rows.windows(2) {
        let prev_energy: f64 = w[0].split(',').nth(1).unwrap().parse().unwrap();
        let row_cols: Vec<&str> = w[1].split(',').collect();
        let row_energy: f64 = row_cols[1].parse().unwrap();
        let accepted: u8 = row_cols.last().unwrap().parse().unwrap();
        if accepted == 0 {
            assert_eq!(
                row_energy, prev_energy,
                "rejected row's energy should equal the preceding accepted energy; \
                 got prev={prev_energy}, rejected={row_energy}"
            );
            found_rejected_with_unchanged_energy = true;
        }
    }
    assert!(
        found_rejected_with_unchanged_energy,
        ".minlog must contain at least one rejected row for this test setup"
    );
}

// rq-d87be0ca
#[test]
fn sd_trajectory_frames_written_at_cadence_and_at_convergence() {
    let dir = tmp_dir("sd_traj_cadence");
    write_argon_pair_init(&dir, 3.6e-10);
    // trajectory_every = 2; let SD converge in a small number of steps.
    let cfg = argon_min_config_full(1.0e-13, 1.0e-11, 1.0e-12, 0.0, 50, 1, 2);
    let cfg_path = dir.join("argon.in.toml");
    fs::write(&cfg_path, cfg).unwrap();
    let summary = run_simulation(&cfg_path).unwrap();
    let n_steps = summary.phases[0].n_steps;
    let traj = fs::read_to_string(dir.join("argon.out.min.xyz")).unwrap();
    // Count frames by scanning header lines containing "Step=".
    let frame_steps: Vec<u64> = traj.lines()
        .filter_map(|l| l.find("Step=").map(|i| {
            let s = &l[i + "Step=".len()..];
            let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
            s[..end].parse::<u64>().unwrap()
        }))
        .collect();
    assert!(!frame_steps.is_empty(), "expected at least one trajectory frame");
    // Every cadence-multiple iter ≤ n_steps must appear, and the final
    // convergence iter must appear.
    for k in 0..=n_steps {
        let on_cadence = k % 2 == 0;
        if on_cadence || k == n_steps {
            assert!(
                frame_steps.contains(&k),
                "expected a trajectory frame at Step={k} (cadence={on_cadence}, final={}); got {:?}",
                k == n_steps,
                frame_steps
            );
        }
    }
}

// Helper: write init.xyz with known velocities so subsequent phases can
// observe them.
fn write_argon_pair_init_with_velocities(
    dir: &Path,
    r: f64,
    v0: (f64, f64, f64),
    v1: (f64, f64, f64),
) {
    let body = format!(
        "2\nLattice=\"1.0e-8 0 0 0 1.0e-8 0 0 0 1.0e-8\" \
         Properties=species:S:1:pos:R:3:velo:R:3\n\
         Ar 0 0 0 {vx0:.16e} {vy0:.16e} {vz0:.16e}\n\
         Ar {r:.16e} 0 0 {vx1:.16e} {vy1:.16e} {vz1:.16e}\n",
        vx0 = v0.0, vy0 = v0.1, vz0 = v0.2,
        vx1 = v1.0, vy1 = v1.1, vz1 = v1.2,
    );
    fs::write(dir.join("argon.in.xyz"), body).unwrap();
}

// rq-020ff80e
#[test]
fn sd_does_not_modify_velocities_across_the_phase() {
    // Use a multi-phase config: minimization followed by a 1-step MD
    // phase. The runner writes the MD phase's step-0 frame BEFORE the
    // first MD step, so its velocity columns reflect the state at MD
    // entry — which is the post-SD state. If SD preserves velocities,
    // the MD trajectory's first frame matches the init.xyz velocities.
    let dir = tmp_dir("sd_preserves_velocities");
    let v0 = (1.7_f64, -0.3, 0.0);
    let v1 = (-0.5_f64, 0.8, 0.0);
    write_argon_pair_init_with_velocities(&dir, 3.6e-10, v0, v1);
    let body = r#"schema_version = 1
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
force_tolerance = 1.0e-12
energy_tolerance = 0.0
max_iterations = 50

[minimization.output]
minlog_every = 1
trajectory_every = 0

[[phase]]
name = "post"
n_steps = 1
dt = 1.0e-15

[phase.integrator]
kind = "velocity-verlet"
lossless = false

[phase.output]
trajectory_every = 1
include_velocities = true
log_every = 0

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
    let _ = run_simulation(&cfg_path).unwrap();
    let traj = fs::read_to_string(dir.join("argon.out.post.xyz")).unwrap();
    let lines: Vec<&str> = traj.lines().collect();
    // First frame: line 0 = "2", line 1 = header, lines 2..4 = particles.
    let cols0: Vec<&str> = lines[2].split_ascii_whitespace().collect();
    let cols1: Vec<&str> = lines[3].split_ascii_whitespace().collect();
    // Columns are: species pos_x pos_y pos_z velo_x velo_y velo_z.
    let vx0_out: f64 = cols0[4].parse().unwrap();
    let vy0_out: f64 = cols0[5].parse().unwrap();
    let vz0_out: f64 = cols0[6].parse().unwrap();
    let vx1_out: f64 = cols1[4].parse().unwrap();
    let vy1_out: f64 = cols1[5].parse().unwrap();
    let vz1_out: f64 = cols1[6].parse().unwrap();
    let approx = |a: f64, b: f64, what: &str| {
        // f32 round-trip through XYZ format; allow eps ~1e-6 relative.
        let scale = a.abs().max(b.abs()).max(1e-300);
        assert!(
            (a - b).abs() < 1e-5 * scale,
            "{what}: {a} != {b}"
        );
    };
    approx(vx0_out, v0.0, "vx0");
    approx(vy0_out, v0.1, "vy0");
    approx(vz0_out, v0.2, "vz0");
    approx(vx1_out, v1.0, "vx1");
    approx(vy1_out, v1.1, "vy1");
    approx(vz1_out, v1.2, "vz1");
}

// rq-cd12ce51
#[test]
fn sd_does_not_modify_simulation_box() {
    // SD's job is to project positions; the simulation box is invariant
    // across the phase. Verify by reading the Lattice header from a
    // post-SD MD phase's first trajectory frame and comparing to the
    // init.xyz Lattice diagonal.
    let dir = tmp_dir("sd_preserves_sim_box");
    write_argon_pair_init(&dir, 3.6e-10);
    let body = r#"schema_version = 1
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
force_tolerance = 1.0e-12
energy_tolerance = 0.0
max_iterations = 50

[minimization.output]
minlog_every = 1
trajectory_every = 0

[[phase]]
name = "post"
n_steps = 1
dt = 1.0e-15

[phase.integrator]
kind = "velocity-verlet"
lossless = false

[phase.output]
trajectory_every = 1
include_velocities = false
log_every = 0

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
    let _ = run_simulation(&cfg_path).unwrap();
    let traj = fs::read_to_string(dir.join("argon.out.post.xyz")).unwrap();
    let header = traj.lines().nth(1).unwrap();
    let lat_start = header.find("Lattice=\"").unwrap() + "Lattice=\"".len();
    let lat_end = lat_start + header[lat_start..].find('"').unwrap();
    let lat_values: Vec<f64> = header[lat_start..lat_end]
        .split_ascii_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(lat_values.len(), 9);
    // init.xyz had Lattice="1.0e-8 0 0 0 1.0e-8 0 0 0 1.0e-8".
    let l: f64 = 1.0e-8;
    let rel = 1e-6;
    let approx = |a: f64, b: f64| (a - b).abs() <= rel * a.abs().max(b.abs()).max(1e-300);
    assert!(approx(lat_values[0], l));
    assert!(approx(lat_values[4], l));
    assert!(approx(lat_values[8], l));
    // Off-diagonal entries are zero.
    for &idx in &[1usize, 2, 3, 5, 6, 7] {
        assert!(lat_values[idx].abs() < 1e-20);
    }
}

// rq-5e075019
#[test]
fn sd_with_no_constraint_slot_does_not_fire_any_constraint_kernel() {
    // No [[constraint_types]] section → no constraint slot → no
    // constraint kernels can fire. Verify by inspecting the
    // minimization phase's per-stage timings: stages whose name starts
    // with "shake_" or "rattle_" or equals "constraint_virial_scatter"
    // must have zero invocations.
    let dir = tmp_dir("sd_no_constraint_hooks");
    write_argon_pair_init(&dir, 3.6e-10);
    let cfg = argon_min_config_full(1.0e-13, 1.0e-11, 1.0e-12, 0.0, 50, 1, 0);
    let cfg_path = dir.join("argon.in.toml");
    fs::write(&cfg_path, cfg).unwrap();
    let _ = run_simulation(&cfg_path).unwrap();
    let timings = fs::read_to_string(dir.join("argon.out.min.timings")).unwrap();
    for line in timings.lines() {
        for prefix in ["shake_", "rattle_", "constraint_virial_scatter"] {
            if line.contains(prefix) {
                // Stage rows look like "<name>  <count>  <total>  ...".
                let cols: Vec<&str> = line.split_ascii_whitespace().collect();
                let count: u64 = cols.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
                assert_eq!(
                    count, 0,
                    "stage `{prefix}` has count {count} > 0 with no constraints declared; line: {line}"
                );
            }
        }
    }
}

// rq-49e26627
#[test]
fn sd_trajectory_frames_omit_velocity_columns() {
    // Even when buffers carry non-zero velocities at phase entry, the
    // minimization-phase trajectory writer must not emit velocity
    // columns (`velo:R:3`).
    let dir = tmp_dir("sd_traj_no_velocities");
    // Write an init.xyz that DOES include velocity columns. Loader will
    // populate buffers.velocities_*, but the minimization phase's
    // trajectory writer should still omit them.
    let body = "2\nLattice=\"1.0e-8 0 0 0 1.0e-8 0 0 0 1.0e-8\" \
                Properties=species:S:1:pos:R:3:velo:R:3\n\
                Ar 0 0 0 1.0 0 0\n\
                Ar 3.6e-10 0 0 -1.0 0 0\n";
    fs::write(dir.join("argon.in.xyz"), body).unwrap();
    let cfg = argon_min_config_full(1.0e-13, 1.0e-11, 1.0e-12, 0.0, 50, 1, 1);
    let cfg_path = dir.join("argon.in.toml");
    fs::write(&cfg_path, cfg).unwrap();
    let _ = run_simulation(&cfg_path).unwrap();
    let traj = fs::read_to_string(dir.join("argon.out.min.xyz")).unwrap();
    for line in traj.lines() {
        assert!(
            !line.contains("velo:R:3"),
            "trajectory header line should not declare velo:R:3 even when buffers carry velocities; got: {line}"
        );
    }
}

// rq-13e42f65
#[test]
fn sd_with_shake_projects_every_trial_onto_rigid_water_manifold() {
    // Two SPC/E waters: the inter-molecule LJ + Coulomb forces drive
    // SD to iterate, the constraint slot's
    // apply_position_projection_only hook fires after each trial, and
    // the final positions must satisfy the rigid-water distances.
    use heddle_md::units::{Dimension, UnitSystem};
    let dir = tmp_dir("sd_shake_water");
    let len_f = UnitSystem::Si.factor(Dimension::Length);
    let r_oh = 1.0e-10_f64;
    let r_hh = 1.633e-10_f64;
    // Two waters; molecule A near origin, molecule B at +3.5 Å along x.
    // Molecule A's H positions are slightly off-manifold (perturbed by
    // ~10%) so SHAKE has something to project.
    let body = format!(
        "6\nLattice=\"1.0e-8 0 0 0 1.0e-8 0 0 0 1.0e-8\" \
         Properties=species:S:1:pos:R:3\n\
         O 0.0 0.0 0.0\n\
         H 1.20e-10 0.0 0.0\n\
         H -3.0e-11 1.5e-10 0.0\n\
         O 3.5e-10 0.0 0.0\n\
         H 4.4e-10 5.0e-11 0.0\n\
         H 4.4e-10 -5.0e-11 0.0\n"
    );
    fs::write(dir.join("water.in.xyz"), body).unwrap();
    // Topology file declaring two SHAKE groups (one per water).
    let topology = "[constraints]\n0 1 2 SPCE\n3 4 5 SPCE\n";
    fs::write(dir.join("water.in.topology"), topology).unwrap();
    let cfg = format!(
        r#"schema_version = 1
init = "water.in.xyz"
topology = "water.in.topology"

[simulation]
seed = 1
temperature = 0.0

[[minimization]]
name = "min"

[minimization.algorithm]
kind = "steepest-descent"
initial_step = 1.0e-13
max_step = 1.0e-12
force_tolerance = 1.0e-30
energy_tolerance = 1.0e-2
max_iterations = 50

[minimization.output]
minlog_every = 1
trajectory_every = 1

[[particle_types]]
name = "O"
mass = 2.6566e-26
charge = -8.476e-20

[[particle_types]]
name = "H"
mass = 1.6735e-27
charge = 4.238e-20

[[pair_interactions]]
between = ["O", "O"]
potential = "lennard-jones"
sigma = 3.166e-10
epsilon = 1.080e-21
cutoff = 1.0e-9

[[pair_interactions]]
between = ["O", "H"]
potential = "lennard-jones"
sigma = 1.0e-10
epsilon = 1.0e-30
cutoff = 1.0e-9

[[pair_interactions]]
between = ["H", "H"]
potential = "lennard-jones"
sigma = 1.0e-10
epsilon = 1.0e-30
cutoff = 1.0e-9

[[constraint_types]]
name = "SPCE"
kind = "shake"
atoms = 3
constraints = [
    {{ i = 0, j = 1, d = {r_oh:.16e} }},
    {{ i = 0, j = 2, d = {r_oh:.16e} }},
    {{ i = 1, j = 2, d = {r_hh:.16e} }},
]

[neighbor_list]
mode = "all-pairs"
"#,
        r_oh = r_oh,
        r_hh = r_hh,
    );
    let cfg_path = dir.join("water.in.toml");
    fs::write(&cfg_path, cfg).unwrap();
    let _ = run_simulation(&cfg_path).unwrap();
    // Read the final trajectory frame and verify constraint distances.
    let traj = fs::read_to_string(dir.join("water.out.min.xyz")).unwrap();
    let lines: Vec<&str> = traj.lines().collect();
    // Parse the LAST frame's particle rows (first 3 atoms = molecule A).
    let n: usize = lines[0].parse().unwrap();
    let total_lines = lines.len();
    let last_frame_start = total_lines - (n + 2);
    let mut pos = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        let cols: Vec<&str> = lines[last_frame_start + 2 + i].split_ascii_whitespace().collect();
        pos[i][0] = cols[1].parse().unwrap();
        pos[i][1] = cols[2].parse().unwrap();
        pos[i][2] = cols[3].parse().unwrap();
    }
    // Trajectory is in SI units (the config default). Compare against
    // r_oh / r_hh in metres.
    let _ = len_f;
    let dist = |a: [f64; 3], b: [f64; 3]| -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let d_oh1 = dist(pos[0], pos[1]);
    let d_oh2 = dist(pos[0], pos[2]);
    let d_hh = dist(pos[1], pos[2]);
    let rel = 5.0e-3;
    assert!(
        ((d_oh1 - r_oh) / r_oh).abs() < rel,
        "O-H1 distance {d_oh1} m differs from r_oh {r_oh} m"
    );
    assert!(
        ((d_oh2 - r_oh) / r_oh).abs() < rel,
        "O-H2 distance {d_oh2} m differs from r_oh {r_oh} m"
    );
    assert!(
        ((d_hh - r_hh) / r_hh).abs() < rel,
        "H-H distance {d_hh} m differs from r_hh {r_hh} m"
    );
}

// rq-e92c18c6
#[test]
fn two_sd_runs_on_the_same_gpu_produce_byte_identical_minlogs() {
    // Reproducibility invariant: two independent SD runs with
    // byte-identical configs and init.xyz produce byte-identical
    // .minlog files on the same GPU.
    let mk = |name: &str| -> PathBuf {
        let dir = tmp_dir(name);
        write_argon_pair_init(&dir, 3.5e-10);
        let cfg = argon_min_config_full(1.0e-9, 1.0e-9, 0.0, 1.0e-2, 50, 1, 0);
        let cfg_path = dir.join("argon.in.toml");
        fs::write(&cfg_path, cfg).unwrap();
        run_simulation(&cfg_path).unwrap();
        dir.join("argon.out.min.minlog")
    };
    let log_a = fs::read_to_string(mk("sd_repro_a")).unwrap();
    let log_b = fs::read_to_string(mk("sd_repro_b")).unwrap();
    assert_eq!(log_a, log_b, "two SD runs with identical inputs must produce byte-identical .minlog");
}
