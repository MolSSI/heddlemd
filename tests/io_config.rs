use std::path::{Path, PathBuf};

use dynamics::io::{ConfigError, NeighborListConfig, PathRole, load_config};

fn tmp_path(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!(
        "dynamics-cfg-{}-{}-{}",
        std::process::id(),
        name,
        nanos
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn minimal_config() -> String {
    r#"schema_version = 1
init = "argon.xyz"

[simulation]
seed = 12345
n_steps = 10
dt = 1.0e-15
temperature = 300.0

[integrator]
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
"#
    .to_string()
}

fn write_config(dir: &Path, contents: &str) -> PathBuf {
    let path = dir.join("sim.toml");
    std::fs::write(&path, contents).unwrap();
    path
}

// rq-7df1515f
#[test]
fn load_valid_minimal_config() {
    let dir = tmp_path("load_valid_minimal_config");
    let path = write_config(&dir, &minimal_config());
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.schema_version, 1);
    assert_eq!(cfg.simulation.seed, 12345);
    assert_eq!(cfg.simulation.n_steps, 10);
    assert_eq!(cfg.simulation.dt, 1.0e-15);
    assert_eq!(cfg.simulation.temperature, 300.0);
    assert!(matches!(
        cfg.integrator,
        dynamics::io::IntegratorKind::VelocityVerlet { lossless: false }
    ));
    assert_eq!(cfg.particle_types.len(), 1);
    assert_eq!(cfg.particle_types[0].name, "Ar");
    assert_eq!(cfg.particle_types[0].mass, 6.6335e-26);
    assert_eq!(cfg.pair_interactions.len(), 1);
    assert_eq!(cfg.pair_interactions[0].between, ("Ar".to_string(), "Ar".to_string()));
    let canonical_dir = std::fs::canonicalize(&dir).unwrap();
    assert_eq!(cfg.init, canonical_dir.join("argon.xyz"));
    assert_eq!(cfg.config_path, canonical_dir.join("sim.toml"));
}

// rq-894c16c4
#[test]
fn defaults_populate_output_section() {
    let dir = tmp_path("defaults_populate_output");
    let path = write_config(&dir, &minimal_config());
    let cfg = load_config(&path).unwrap();
    let canonical_dir = std::fs::canonicalize(&dir).unwrap();
    assert_eq!(cfg.output.trajectory_path, canonical_dir.join("sim-traj.xyz"));
    assert_eq!(cfg.output.trajectory_every, 100);
    assert!(cfg.output.include_velocities);
    assert_eq!(cfg.output.log_path, canonical_dir.join("sim.log"));
    assert_eq!(cfg.output.log_every, 100);
}

// rq-d148149f
#[test]
fn explicit_output_overrides_defaults() {
    let dir = tmp_path("explicit_output");
    let body = format!(
        "{}\n[output]\ntrajectory_path = \"custom-traj.xyz\"\ntrajectory_every = 50\nlog_path = \"custom.log\"\nlog_every = 25\ninclude_velocities = false\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    let canonical_dir = std::fs::canonicalize(&dir).unwrap();
    assert_eq!(cfg.output.trajectory_path, canonical_dir.join("custom-traj.xyz"));
    assert_eq!(cfg.output.trajectory_every, 50);
    assert!(!cfg.output.include_velocities);
    assert_eq!(cfg.output.log_path, canonical_dir.join("custom.log"));
    assert_eq!(cfg.output.log_every, 25);
}

// rq-5ded1806
#[test]
fn absolute_paths_honored() {
    let dir = tmp_path("absolute_paths");
    let abs_init = dir.join("abs-init.xyz");
    let abs_traj = dir.join("abs-traj.xyz");
    let abs_log = dir.join("abs.log");
    let body = format!(
        "schema_version = 1\ninit = \"{}\"\n[simulation]\nseed=1\nn_steps=1\ndt=1.0e-15\ntemperature=0.0\n[integrator]\nkind=\"velocity-verlet\"\nlossless=false\n[[particle_types]]\nname=\"Ar\"\nmass=1.0\n[[pair_interactions]]\nbetween=[\"Ar\",\"Ar\"]\npotential=\"lennard-jones\"\nsigma=1.0\nepsilon=1.0\ncutoff=1.0\n[output]\ntrajectory_path=\"{}\"\nlog_path=\"{}\"\n",
        abs_init.display(),
        abs_traj.display(),
        abs_log.display(),
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.init, abs_init);
    assert_eq!(cfg.output.trajectory_path, abs_traj);
    assert_eq!(cfg.output.log_path, abs_log);
}

// rq-d5085350
#[test]
fn pair_unknown_type_under_normalisation() {
    let dir = tmp_path("pair_unknown_normalised");
    let body = minimal_config().replace(
        "between = [\"Ar\", \"Ar\"]",
        "between = [\"Kr\", \"Ar\"]",
    );
    let path = write_config(&dir, &body);
    let err = load_config(&path).unwrap_err();
    match err {
        ConfigError::UnknownTypeInPair { name, pair_index } => {
            assert_eq!(name, "Kr");
            assert_eq!(pair_index, 0);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

// rq-d3d4b6b3
#[test]
fn pair_between_normalisation_with_both_declared() {
    let dir = tmp_path("pair_normalisation_full");
    // Two types and all three pairs — multi-component configs are accepted;
    // verify the loaded shape rather than the legacy rejection.
    let body = r#"schema_version = 1
init = "init.xyz"

[simulation]
seed = 1
n_steps = 1
dt = 1.0e-15
temperature = 0.0

[integrator]
kind = "velocity-verlet"
lossless = false

[[particle_types]]
name = "Ar"
mass = 1.0

[[particle_types]]
name = "Kr"
mass = 2.0

[[pair_interactions]]
between = ["Kr", "Ar"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0

[[pair_interactions]]
between = ["Kr", "Kr"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0
"#;
    let path = write_config(&dir, body);
    let config = load_config(&path).expect("multi-type config loads");
    assert_eq!(config.particle_types.len(), 2);
    assert_eq!(config.pair_interactions.len(), 3);
    // Pair entry whose source order was reversed gets normalised.
    let normalised: Vec<(String, String)> = config
        .pair_interactions
        .iter()
        .map(|p| p.between.clone())
        .collect();
    assert!(normalised.contains(&("Ar".to_string(), "Kr".to_string())));
}

// rq-106dcabd
#[test]
fn n_steps_zero_is_accepted() {
    let dir = tmp_path("n_steps_zero");
    let body = minimal_config().replace("n_steps = 10", "n_steps = 0");
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.simulation.n_steps, 0);
}

// rq-69a31102
#[test]
fn missing_schema_version() {
    let dir = tmp_path("missing_schema_version");
    let body = minimal_config().replace("schema_version = 1\n", "");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "schema_version"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-0cb3c41c
#[test]
fn unknown_schema_version() {
    let dir = tmp_path("unknown_schema_version");
    let body = minimal_config().replace("schema_version = 1", "schema_version = 2");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::UnsupportedSchemaVersion { actual, supported } => {
            assert_eq!(actual, 2);
            assert_eq!(supported, 1);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-4169d3af
#[test]
fn reject_schema_version_zero() {
    let dir = tmp_path("schema_version_zero");
    let body = minimal_config().replace("schema_version = 1", "schema_version = 0");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::UnsupportedSchemaVersion { actual, supported } => {
            assert_eq!(actual, 0);
            assert_eq!(supported, 1);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-ae7f8045
#[test]
fn file_does_not_exist() {
    let dir = tmp_path("missing_file");
    let path = dir.join("nope.toml");
    match load_config(&path).unwrap_err() {
        ConfigError::Io(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-57f8de41
#[test]
fn malformed_toml() {
    let dir = tmp_path("malformed_toml");
    let path = write_config(&dir, "schema_version = [");
    match load_config(&path).unwrap_err() {
        ConfigError::Parse(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-761f26c6
#[test]
fn unknown_top_level_key_permitted() {
    let dir = tmp_path("unknown_top_level");
    let body = format!("{}unknown_key = \"x\"\n", minimal_config());
    let path = write_config(&dir, &body);
    load_config(&path).unwrap();
}

// rq-f0e3b004
#[test]
fn missing_init_field() {
    let dir = tmp_path("missing_init");
    let body = minimal_config().replace("init = \"argon.xyz\"\n", "");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "init"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-9bfc2c1d
#[test]
fn missing_seed() {
    let dir = tmp_path("missing_seed");
    let body = minimal_config().replace("seed = 12345\n", "");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "simulation.seed"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-221b1bb4
#[test]
fn missing_dt() {
    let dir = tmp_path("missing_dt");
    let body = minimal_config().replace("dt = 1.0e-15\n", "");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "simulation.dt"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-52c9b17a
#[test]
fn missing_temperature() {
    let dir = tmp_path("missing_temperature");
    let body = minimal_config().replace("temperature = 300.0\n", "");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "simulation.temperature"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-66bf31c6
#[test]
fn missing_integrator_section() {
    let dir = tmp_path("missing_integrator");
    let body = minimal_config()
        .replace("[integrator]\nkind = \"velocity-verlet\"\nlossless = false\n", "");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "integrator"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-1e1c5f3b
#[test]
fn missing_particle_types() {
    let dir = tmp_path("missing_particle_types");
    let body = r#"schema_version = 1
init = "init.xyz"

[simulation]
seed = 1
n_steps = 1
dt = 1.0e-15
temperature = 0.0

[integrator]
kind = "velocity-verlet"
lossless = false

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0
"#;
    let path = write_config(&dir, body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "particle_types"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-a94d2c13
#[test]
fn missing_pair_interactions() {
    let dir = tmp_path("missing_pair");
    let body = r#"schema_version = 1
init = "init.xyz"

[simulation]
seed = 1
n_steps = 1
dt = 1.0e-15
temperature = 0.0

[integrator]
kind = "velocity-verlet"
lossless = false

[[particle_types]]
name = "Ar"
mass = 1.0
"#;
    let path = write_config(&dir, body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingPairInteraction { types } => {
            assert_eq!(types, ("Ar".to_string(), "Ar".to_string()));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-025b2c3b
#[test]
fn reject_zero_dt() {
    let dir = tmp_path("zero_dt");
    let body = minimal_config().replace("dt = 1.0e-15", "dt = 0.0");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "simulation.dt"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-0051b248
#[test]
fn reject_negative_dt() {
    let dir = tmp_path("neg_dt");
    let body = minimal_config().replace("dt = 1.0e-15", "dt = -1.0e-15");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "simulation.dt"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-dffdd81c
#[test]
fn reject_nan_dt() {
    let dir = tmp_path("nan_dt");
    let body = minimal_config().replace("dt = 1.0e-15", "dt = nan");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "simulation.dt"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-f009e02b
#[test]
fn reject_negative_temperature() {
    let dir = tmp_path("neg_temp");
    let body = minimal_config().replace("temperature = 300.0", "temperature = -1.0");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "simulation.temperature"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-cc12f2d8
#[test]
fn zero_temperature_accepted() {
    let dir = tmp_path("zero_temp");
    let body = minimal_config().replace("temperature = 300.0", "temperature = 0.0");
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.simulation.temperature, 0.0);
}

// rq-47697f4a
#[test]
fn reject_nonpositive_mass() {
    let dir = tmp_path("zero_mass");
    let body = minimal_config().replace("mass = 6.6335e-26", "mass = 0.0");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "particle_types[0].mass"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-aa19f894
#[test]
fn reject_nonpositive_sigma() {
    let dir = tmp_path("zero_sigma");
    let body = minimal_config().replace("sigma = 3.40e-10", "sigma = 0.0");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "pair_interactions[0].sigma"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-017b6769
#[test]
fn reject_nonpositive_epsilon() {
    let dir = tmp_path("neg_epsilon");
    let body = minimal_config().replace("epsilon = 1.65e-21", "epsilon = -1.0");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => {
            assert_eq!(field, "pair_interactions[0].epsilon");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-ae65c293
#[test]
fn reject_nonpositive_cutoff() {
    let dir = tmp_path("zero_cutoff");
    let body = minimal_config().replace("cutoff = 1.0e-9", "cutoff = 0.0");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "pair_interactions[0].cutoff"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-a3a5905d
#[test]
fn reject_unknown_potential() {
    let dir = tmp_path("unknown_potential");
    let body = minimal_config().replace("potential = \"lennard-jones\"", "potential = \"morse\"");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => {
            assert_eq!(field, "pair_interactions[0].potential");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-a30ac09f
#[test]
fn reject_empty_type_name() {
    let dir = tmp_path("empty_type_name");
    let body = minimal_config().replace("name = \"Ar\"", "name = \"\"");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "particle_types[0].name"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-560dffb8
#[test]
fn reject_duplicate_type_names() {
    let dir = tmp_path("duplicate_type_names");
    let body = r#"schema_version = 1
init = "init.xyz"

[simulation]
seed = 1
n_steps = 1
dt = 1.0e-15
temperature = 0.0

[integrator]
kind = "velocity-verlet"
lossless = false

[[particle_types]]
name = "Ar"
mass = 1.0

[[particle_types]]
name = "Ar"
mass = 2.0

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0
"#;
    let path = write_config(&dir, body);
    match load_config(&path).unwrap_err() {
        ConfigError::DuplicateTypeName { name } => assert_eq!(name, "Ar"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-c9fa5cda
#[test]
fn reject_pair_unknown_type() {
    let dir = tmp_path("pair_unknown_type");
    let body = format!(
        "{}\n[[pair_interactions]]\nbetween = [\"Ar\", \"Xe\"]\npotential = \"lennard-jones\"\nsigma = 1.0\nepsilon = 1.0\ncutoff = 1.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::UnknownTypeInPair { name, pair_index } => {
            assert_eq!(name, "Xe");
            assert_eq!(pair_index, 1);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-ae6d5db8
#[test]
fn reject_missing_pair() {
    let dir = tmp_path("missing_pair_interaction");
    let body = r#"schema_version = 1
init = "init.xyz"

[simulation]
seed = 1
n_steps = 1
dt = 1.0e-15
temperature = 0.0

[integrator]
kind = "velocity-verlet"
lossless = false

[[particle_types]]
name = "Ar"
mass = 1.0

[[particle_types]]
name = "Kr"
mass = 2.0

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0

[[pair_interactions]]
between = ["Kr", "Kr"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0
"#;
    let path = write_config(&dir, body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingPairInteraction { types } => {
            assert_eq!(types, ("Ar".to_string(), "Kr".to_string()));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-f11e9d4c
#[test]
fn reject_duplicate_pair() {
    let dir = tmp_path("dup_pair");
    let body = format!(
        "{}\n[[pair_interactions]]\nbetween = [\"Ar\", \"Ar\"]\npotential = \"lennard-jones\"\nsigma = 1.0\nepsilon = 1.0\ncutoff = 1.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::DuplicatePairInteraction { types } => {
            assert_eq!(types, ("Ar".to_string(), "Ar".to_string()));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-9e4d8944
#[test]
fn reject_duplicate_pair_reversed() {
    let dir = tmp_path("dup_pair_reversed");
    let body = r#"schema_version = 1
init = "init.xyz"

[simulation]
seed = 1
n_steps = 1
dt = 1.0e-15
temperature = 0.0

[integrator]
kind = "velocity-verlet"
lossless = false

[[particle_types]]
name = "Ar"
mass = 1.0

[[particle_types]]
name = "Kr"
mass = 2.0

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0

[[pair_interactions]]
between = ["Kr", "Kr"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0

[[pair_interactions]]
between = ["Ar", "Kr"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0

[[pair_interactions]]
between = ["Kr", "Ar"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0
"#;
    let path = write_config(&dir, body);
    match load_config(&path).unwrap_err() {
        ConfigError::DuplicatePairInteraction { types } => {
            assert_eq!(types, ("Ar".to_string(), "Kr".to_string()));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-e553c05b
#[test]
fn reject_init_equals_trajectory() {
    let dir = tmp_path("init_eq_traj");
    let body = format!(
        "{}\n[output]\ntrajectory_path = \"argon.xyz\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::PathCollision { kind_a, kind_b, .. } => {
            assert_eq!(kind_a, PathRole::Init);
            assert_eq!(kind_b, PathRole::Trajectory);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-765c96c5
#[test]
fn reject_trajectory_equals_log() {
    let dir = tmp_path("traj_eq_log");
    let body = format!(
        "{}\n[output]\ntrajectory_path = \"run.dat\"\nlog_path = \"run.dat\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::PathCollision { kind_a, kind_b, .. } => {
            assert_eq!(kind_a, PathRole::Trajectory);
            assert_eq!(kind_b, PathRole::Log);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-330d6b42
#[test]
fn reject_init_equals_log() {
    let dir = tmp_path("init_eq_log");
    let body = format!(
        "{}\n[output]\nlog_path = \"argon.xyz\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::PathCollision { kind_a, kind_b, .. } => {
            assert_eq!(kind_a, PathRole::Init);
            assert_eq!(kind_b, PathRole::Log);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-f114c560
#[test]
fn accept_multi_type_with_complete_pair_table() {
    let dir = tmp_path("multi_type");
    let body = r#"schema_version = 1
init = "init.xyz"

[simulation]
seed = 1
n_steps = 1
dt = 1.0e-15
temperature = 0.0

[integrator]
kind = "velocity-verlet"
lossless = false

[[particle_types]]
name = "Ar"
mass = 1.0

[[particle_types]]
name = "Kr"
mass = 2.0

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0

[[pair_interactions]]
between = ["Ar", "Kr"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0

[[pair_interactions]]
between = ["Kr", "Kr"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0
"#;
    let path = write_config(&dir, body);
    let config = load_config(&path).expect("two-type config loads");
    assert_eq!(config.particle_types.len(), 2);
    assert_eq!(config.pair_interactions.len(), 3);
}

#[test]
fn reject_multi_type_with_missing_pair() {
    let dir = tmp_path("multi_type_missing_pair");
    let body = r#"schema_version = 1
init = "init.xyz"

[simulation]
seed = 1
n_steps = 1
dt = 1.0e-15
temperature = 0.0

[integrator]
kind = "velocity-verlet"
lossless = false

[[particle_types]]
name = "Ar"
mass = 1.0

[[particle_types]]
name = "Kr"
mass = 2.0

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0

[[pair_interactions]]
between = ["Kr", "Kr"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0
"#;
    let path = write_config(&dir, body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingPairInteraction { types } => {
            assert_eq!(types, ("Ar".to_string(), "Kr".to_string()));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-a5c86770
#[test]
fn timings_path_defaults_to_stem_timings() {
    let dir = tmp_path("timings_default");
    let path = write_config(&dir, &minimal_config());
    let cfg = load_config(&path).unwrap();
    let canonical_dir = std::fs::canonicalize(&dir).unwrap();
    assert_eq!(cfg.output.timings_path, canonical_dir.join("sim.timings"));
}

// rq-fa24a8d1
#[test]
fn timings_path_override() {
    let dir = tmp_path("timings_override");
    let body = format!(
        "{}\n[output]\ntimings_path = \"custom.timings\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    let canonical_dir = std::fs::canonicalize(&dir).unwrap();
    assert_eq!(cfg.output.timings_path, canonical_dir.join("custom.timings"));
}

// rq-7d5915bb
#[test]
fn reject_init_equals_timings() {
    let dir = tmp_path("init_eq_timings");
    let body = format!(
        "{}\n[output]\ntimings_path = \"argon.xyz\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::PathCollision { kind_a, kind_b, .. } => {
            assert_eq!(kind_a, PathRole::Init);
            assert_eq!(kind_b, PathRole::Timings);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-ec8d715d
#[test]
fn reject_trajectory_equals_timings() {
    let dir = tmp_path("traj_eq_timings");
    let body = format!(
        "{}\n[output]\ntrajectory_path = \"run.dat\"\ntimings_path = \"run.dat\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::PathCollision { kind_a, kind_b, .. } => {
            assert_eq!(kind_a, PathRole::Trajectory);
            assert_eq!(kind_b, PathRole::Timings);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-8f665dd0
#[test]
fn reject_log_equals_timings() {
    let dir = tmp_path("log_eq_timings");
    let body = format!(
        "{}\n[output]\nlog_path = \"run.dat\"\ntimings_path = \"run.dat\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::PathCollision { kind_a, kind_b, .. } => {
            assert_eq!(kind_a, PathRole::Log);
            assert_eq!(kind_b, PathRole::Timings);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-100115a0
#[test]
fn missing_integrator_kind() {
    let dir = tmp_path("missing_integrator_kind");
    let body = minimal_config().replace(
        "[integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[integrator]\nlossless = false\n",
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "integrator.kind"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-9d882742
#[test]
fn unknown_integrator_kind() {
    let dir = tmp_path("unknown_integrator_kind");
    let body = minimal_config().replace("kind = \"velocity-verlet\"", "kind = \"custom\"");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::UnknownIntegratorKind { actual } => assert_eq!(actual, "custom"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-86aa2be7
#[test]
fn langevin_with_valid_parameters_accepted() {
    let dir = tmp_path("langevin_valid");
    let body = minimal_config().replace(
        "[integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[integrator]\nkind = \"langevin-baoab\"\nfriction = 1.0e12\ntemperature = 300.0\nseed = 42\n",
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    match cfg.integrator {
        dynamics::io::IntegratorKind::LangevinBaoab {
            friction,
            temperature,
            seed,
        } => {
            assert_eq!(friction, 1.0e12);
            assert_eq!(temperature, 300.0);
            assert_eq!(seed, 42);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-40ed9975
#[test]
fn langevin_missing_friction() {
    let dir = tmp_path("langevin_missing_friction");
    let body = minimal_config().replace(
        "[integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[integrator]\nkind = \"langevin-baoab\"\ntemperature = 300.0\nseed = 42\n",
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "integrator.friction"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-f2431cc4
#[test]
fn langevin_missing_temperature() {
    let dir = tmp_path("langevin_missing_temperature");
    let body = minimal_config().replace(
        "[integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[integrator]\nkind = \"langevin-baoab\"\nfriction = 1.0e12\nseed = 42\n",
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "integrator.temperature"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-92f643cb
#[test]
fn langevin_missing_seed() {
    let dir = tmp_path("langevin_missing_seed");
    let body = minimal_config().replace(
        "[integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[integrator]\nkind = \"langevin-baoab\"\nfriction = 1.0e12\ntemperature = 300.0\n",
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "integrator.seed"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-385408d0
#[test]
fn langevin_friction_zero_rejected() {
    let dir = tmp_path("langevin_friction_zero");
    let body = minimal_config().replace(
        "[integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[integrator]\nkind = \"langevin-baoab\"\nfriction = 0.0\ntemperature = 300.0\nseed = 42\n",
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "integrator.friction"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-583201cb
#[test]
fn langevin_friction_negative_rejected() {
    let dir = tmp_path("langevin_friction_negative");
    let body = minimal_config().replace(
        "[integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[integrator]\nkind = \"langevin-baoab\"\nfriction = -1.0\ntemperature = 300.0\nseed = 42\n",
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "integrator.friction"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-789b7a33
#[test]
fn langevin_temperature_zero_rejected() {
    let dir = tmp_path("langevin_temperature_zero");
    let body = minimal_config().replace(
        "[integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[integrator]\nkind = \"langevin-baoab\"\nfriction = 1.0e12\ntemperature = 0.0\nseed = 42\n",
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "integrator.temperature"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-30270f03
#[test]
fn vv_rejects_langevin_fields() {
    let dir = tmp_path("vv_rejects_langevin");
    let body = minimal_config().replace(
        "[integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[integrator]\nkind = \"velocity-verlet\"\nfriction = 1.0e12\n",
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::UnknownIntegratorField { kind, field } => {
            assert_eq!(kind, "velocity-verlet");
            assert_eq!(field, "friction");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-e7c05140
#[test]
fn langevin_rejects_vv_fields() {
    let dir = tmp_path("langevin_rejects_vv");
    let body = minimal_config().replace(
        "[integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[integrator]\nkind = \"langevin-baoab\"\nfriction = 1.0e12\ntemperature = 300.0\nseed = 42\nlossless = false\n",
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::UnknownIntegratorField { kind, field } => {
            assert_eq!(kind, "langevin-baoab");
            assert_eq!(field, "lossless");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-66ec7ee4
#[test]
fn vv_lossless_defaults_to_false() {
    let dir = tmp_path("vv_lossless_default");
    let body = minimal_config().replace(
        "[integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[integrator]\nkind = \"velocity-verlet\"\n",
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert!(matches!(
        cfg.integrator,
        dynamics::io::IntegratorKind::VelocityVerlet { lossless: false }
    ));
}

// rq-6cb9ab62
#[test]
fn bonds_field_optional_defaults_none() {
    let dir = tmp_path("bonds_optional");
    let path = write_config(&dir, &minimal_config());
    let cfg = load_config(&path).unwrap();
    assert!(cfg.bonds.is_none());
}

// rq-027153d9
#[test]
fn bonds_field_resolved_relative() {
    let dir = tmp_path("bonds_relative");
    let body = minimal_config().replace(
        "init = \"argon.xyz\"\n",
        "init = \"argon.xyz\"\nbonds = \"topology.bonds\"\n",
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    let canonical_dir = std::fs::canonicalize(&dir).unwrap();
    assert_eq!(cfg.bonds, Some(canonical_dir.join("topology.bonds")));
}

// rq-576561a2
#[test]
fn bonds_absolute_preserved() {
    let dir = tmp_path("bonds_absolute");
    let body = minimal_config().replace(
        "init = \"argon.xyz\"\n",
        "init = \"argon.xyz\"\nbonds = \"/data/topology.bonds\"\n",
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.bonds, Some(std::path::PathBuf::from("/data/topology.bonds")));
}

// rq-4186d4f4
#[test]
fn reject_bonds_eq_init() {
    let dir = tmp_path("bonds_eq_init");
    let body = minimal_config().replace(
        "init = \"argon.xyz\"\n",
        "init = \"argon.xyz\"\nbonds = \"argon.xyz\"\n",
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::PathCollision { kind_a, kind_b, .. } => {
            assert_eq!(kind_a, PathRole::Init);
            assert_eq!(kind_b, PathRole::Bonds);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-98180119
#[test]
fn reject_bonds_eq_trajectory() {
    let dir = tmp_path("bonds_eq_traj");
    let body = format!(
        "{}\n[output]\ntrajectory_path = \"run.dat\"\n",
        minimal_config().replace(
            "init = \"argon.xyz\"\n",
            "init = \"argon.xyz\"\nbonds = \"run.dat\"\n",
        )
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::PathCollision { kind_a, kind_b, .. } => {
            assert!(matches!(kind_a, PathRole::Trajectory | PathRole::Bonds));
            assert!(matches!(kind_b, PathRole::Trajectory | PathRole::Bonds));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-6ad9a0f8
#[test]
fn bond_types_optional_empty() {
    let dir = tmp_path("bond_types_empty");
    let path = write_config(&dir, &minimal_config());
    let cfg = load_config(&path).unwrap();
    assert!(cfg.bond_types.is_empty());
}

// rq-f704561b
#[test]
fn valid_morse_bond_type_accepted() {
    let dir = tmp_path("morse_valid");
    let body = format!(
        "{}\n[[bond_types]]\nname = \"ArAr\"\npotential = \"morse\"\nde = 1.65e-21\na = 1.9e10\nre = 3.4e-10\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.bond_types.len(), 1);
    match &cfg.bond_types[0] {
        dynamics::io::BondTypeConfig::Morse { name, de, a, re } => {
            assert_eq!(name, "ArAr");
            assert_eq!(*de, 1.65e-21);
            assert_eq!(*a, 1.9e10);
            assert_eq!(*re, 3.4e-10);
        }
    }
}

// rq-c79a1408
#[test]
fn bond_type_missing_potential() {
    let dir = tmp_path("bond_type_missing_potential");
    let body = format!(
        "{}\n[[bond_types]]\nname = \"ArAr\"\nde = 1.0\na = 1.0\nre = 1.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "bond_types[0].potential"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-e34d764e
#[test]
fn bond_type_unknown_potential() {
    let dir = tmp_path("bond_type_unknown");
    let body = format!(
        "{}\n[[bond_types]]\nname = \"ArAr\"\npotential = \"harmonic\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::UnknownBondPotential {
            actual,
            bond_type_index,
        } => {
            assert_eq!(actual, "harmonic");
            assert_eq!(bond_type_index, 0);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-3b0e8140
#[test]
fn morse_bond_type_missing_de() {
    let dir = tmp_path("morse_missing_de");
    let body = format!(
        "{}\n[[bond_types]]\nname = \"X\"\npotential = \"morse\"\na = 1.0\nre = 1.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "bond_types[0].de"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-ecc8f632
#[test]
fn morse_bond_type_rejects_zero_de() {
    let dir = tmp_path("morse_zero_de");
    let body = format!(
        "{}\n[[bond_types]]\nname = \"X\"\npotential = \"morse\"\nde = 0.0\na = 1.0\nre = 1.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "bond_types[0].de"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-ae85bf7b
#[test]
fn morse_bond_type_rejects_negative_a() {
    let dir = tmp_path("morse_neg_a");
    let body = format!(
        "{}\n[[bond_types]]\nname = \"X\"\npotential = \"morse\"\nde = 1.0\na = -1.0\nre = 1.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "bond_types[0].a"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-3533e8a9
#[test]
fn morse_bond_type_rejects_zero_re() {
    let dir = tmp_path("morse_zero_re");
    let body = format!(
        "{}\n[[bond_types]]\nname = \"X\"\npotential = \"morse\"\nde = 1.0\na = 1.0\nre = 0.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "bond_types[0].re"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-e40d2722
#[test]
fn morse_bond_type_rejects_extra_field() {
    let dir = tmp_path("morse_extra_field");
    let body = format!(
        "{}\n[[bond_types]]\nname = \"X\"\npotential = \"morse\"\nde = 1.0\na = 1.0\nre = 1.0\nstiffness = 2.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::UnknownBondTypeField {
            potential,
            field,
            bond_type_index,
        } => {
            assert_eq!(potential, "morse");
            assert_eq!(field, "stiffness");
            assert_eq!(bond_type_index, 0);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-ed1d6c71
#[test]
fn reject_duplicate_bond_type_name() {
    let dir = tmp_path("dup_bond_type");
    let body = format!(
        "{}\n[[bond_types]]\nname = \"X\"\npotential = \"morse\"\nde = 1.0\na = 1.0\nre = 1.0\n[[bond_types]]\nname = \"X\"\npotential = \"morse\"\nde = 2.0\na = 2.0\nre = 2.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::DuplicateBondTypeName { name } => assert_eq!(name, "X"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-50521f04
#[test]
fn empty_bond_type_name_rejected() {
    let dir = tmp_path("empty_bond_name");
    let body = format!(
        "{}\n[[bond_types]]\nname = \"\"\npotential = \"morse\"\nde = 1.0\na = 1.0\nre = 1.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "bond_types[0].name"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-97e525d8
#[test]
fn trajectory_every_zero_accepted() {
    let dir = tmp_path("traj_every_zero");
    let body = format!(
        "{}\n[output]\ntrajectory_every = 0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.output.trajectory_every, 0);
}

// rq-318cd47d
#[test]
fn log_every_zero_accepted() {
    let dir = tmp_path("log_every_zero");
    let body = format!("{}\n[output]\nlog_every = 0\n", minimal_config());
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.output.log_every, 0);
}

// --- Neighbor list ---

#[test]
fn neighbor_list_defaults_to_cell_list_when_section_omitted() {
    let dir = tmp_path("nl_default");
    let path = write_config(&dir, &minimal_config());
    let cfg = load_config(&path).unwrap();
    match cfg.neighbor_list {
        NeighborListConfig::CellList { max_neighbors, r_skin } => {
            assert_eq!(max_neighbors, 256);
            // cutoff = 1.0e-9, default r_skin = 0.3 * cutoff = 3.0e-10
            assert!((r_skin - 3.0e-10).abs() < 1.0e-20);
        }
        other => panic!("expected CellList, got {other:?}"),
    }
}

#[test]
fn neighbor_list_cell_list_explicit_parameters() {
    let dir = tmp_path("nl_explicit");
    let body = format!(
        "{}\n[neighbor_list]\nmode = \"cell-list\"\nmax_neighbors = 128\nr_skin = 2.0e-10\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(
        cfg.neighbor_list,
        NeighborListConfig::CellList { max_neighbors: 128, r_skin: 2.0e-10 }
    );
}

#[test]
fn neighbor_list_cell_list_default_max_neighbors() {
    let dir = tmp_path("nl_default_max");
    let body = format!(
        "{}\n[neighbor_list]\nmode = \"cell-list\"\nr_skin = 2.0e-10\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(
        cfg.neighbor_list,
        NeighborListConfig::CellList { max_neighbors: 256, r_skin: 2.0e-10 }
    );
}

#[test]
fn neighbor_list_cell_list_default_r_skin() {
    let dir = tmp_path("nl_default_rskin");
    let body = format!(
        "{}\n[neighbor_list]\nmode = \"cell-list\"\nmax_neighbors = 128\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    match cfg.neighbor_list {
        NeighborListConfig::CellList { max_neighbors, r_skin } => {
            assert_eq!(max_neighbors, 128);
            assert!((r_skin - 3.0e-10).abs() < 1.0e-20);
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn neighbor_list_all_pairs_mode() {
    let dir = tmp_path("nl_all_pairs");
    let body = format!(
        "{}\n[neighbor_list]\nmode = \"all-pairs\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.neighbor_list, NeighborListConfig::AllPairs);
}

#[test]
fn neighbor_list_unknown_mode_rejected() {
    let dir = tmp_path("nl_unknown_mode");
    let body = format!(
        "{}\n[neighbor_list]\nmode = \"kd-tree\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let err = load_config(&path).unwrap_err();
    match err {
        ConfigError::UnknownNeighborListMode { actual } => assert_eq!(actual, "kd-tree"),
        other => panic!("got {other:?}"),
    }
}

#[test]
fn neighbor_list_all_pairs_rejects_max_neighbors() {
    let dir = tmp_path("nl_all_pairs_rejects_max");
    let body = format!(
        "{}\n[neighbor_list]\nmode = \"all-pairs\"\nmax_neighbors = 64\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let err = load_config(&path).unwrap_err();
    match err {
        ConfigError::UnknownNeighborListField { mode, field } => {
            assert_eq!(mode, "all-pairs");
            assert_eq!(field, "max_neighbors");
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn neighbor_list_cell_list_rejects_unknown_field() {
    let dir = tmp_path("nl_cell_unknown_field");
    let body = format!(
        "{}\n[neighbor_list]\nmode = \"cell-list\"\nstencil = \"huge\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let err = load_config(&path).unwrap_err();
    match err {
        ConfigError::UnknownNeighborListField { mode, field } => {
            assert_eq!(mode, "cell-list");
            assert_eq!(field, "stencil");
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn neighbor_list_rejects_zero_max_neighbors() {
    let dir = tmp_path("nl_zero_max");
    let body = format!(
        "{}\n[neighbor_list]\nmode = \"cell-list\"\nmax_neighbors = 0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let err = load_config(&path).unwrap_err();
    assert!(matches!(
        err,
        ConfigError::InvalidValue { ref field, .. } if field == "neighbor_list.max_neighbors"
    ), "got {err:?}");
}

#[test]
fn neighbor_list_rejects_non_positive_r_skin() {
    let dir = tmp_path("nl_zero_rskin");
    let body = format!(
        "{}\n[neighbor_list]\nmode = \"cell-list\"\nr_skin = 0.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let err = load_config(&path).unwrap_err();
    assert!(matches!(
        err,
        ConfigError::InvalidValue { ref field, .. } if field == "neighbor_list.r_skin"
    ), "got {err:?}");
}
