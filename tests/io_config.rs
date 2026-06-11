use std::path::{Path, PathBuf};

use dynamics::Registries;
use dynamics::integrator::IntegratorRegistry;
use dynamics::io::{ConfigError, NeighborListConfig, PathRole, SlotConfig, load_config};

fn param_f64(slot: &SlotConfig, key: &str) -> f64 {
    slot.params.get(key).and_then(|v| v.as_float()).unwrap()
}

fn param_u64(slot: &SlotConfig, key: &str) -> u64 {
    slot.params
        .get(key)
        .and_then(|v| v.as_integer())
        .map(|i| i as u64)
        .unwrap()
}

fn param_bool(slot: &SlotConfig, key: &str) -> bool {
    slot.params.get(key).and_then(|v| v.as_bool()).unwrap()
}

/// Same as `param_bool` but returns `default` when the field is absent.
/// Useful for testing that optional kind-specific fields like
/// `lossless` are not required to round-trip through `Config`.
fn param_bool_or(slot: &SlotConfig, key: &str, default: bool) -> bool {
    slot.params
        .get(key)
        .and_then(|v| v.as_bool())
        .unwrap_or(default)
}

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
    // Author the helper in `units = "atomic"` so the numeric values
    // round-trip through the loader unchanged — every test that
    // checks `cfg.X == <value>` exercises parsing structure rather
    // than the unit conversion. Conversion-specific tests use
    // `units = "si"` explicitly and verify the post-conversion
    // (atomic) value.
    r#"schema_version = 1
units = "atomic"
init = "argon.in.xyz"

[simulation]
seed = 12345
temperature = 300.0

[[phase]]
name = "run"
n_steps = 10
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
"#
    .to_string()
}

fn write_config(dir: &Path, contents: &str) -> PathBuf {
    write_config_named(dir, "sim.in.toml", contents)
}

fn write_config_named(dir: &Path, filename: &str, contents: &str) -> PathBuf {
    let path = dir.join(filename);
    std::fs::write(&path, contents).unwrap();
    path
}

fn assert_parse(err: &ConfigError, expected_path: &str) {
    match err {
        ConfigError::Parse { path, message } => {
            assert_eq!(
                path, expected_path,
                "expected Parse path `{expected_path}`, got `{path}` (message: {message})"
            );
        }
        other => panic!("expected ConfigError::Parse, got {other:?}"),
    }
}

/// For "unknown field" errors the deserialiser reports the *enclosing*
/// table's path (e.g. `thermostat` or `pair_interactions[0]`) and names
/// the offending field inside the message string. This helper checks
/// both.
fn assert_parse_path_and_field(err: &ConfigError, expected_path: &str, unknown_field: &str) {
    match err {
        ConfigError::Parse { path, message } => {
            assert_eq!(
                path, expected_path,
                "expected Parse path `{expected_path}`, got `{path}` (message: {message})"
            );
            assert!(
                message.contains(unknown_field),
                "expected message to mention `{unknown_field}`, got `{message}`"
            );
        }
        other => panic!("expected ConfigError::Parse, got {other:?}"),
    }
}

// rq-7df1515f
#[test]
fn load_valid_minimal_config() {
    let dir = tmp_path("load_valid_minimal_config");
    let path = write_config(&dir, &minimal_config());
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.schema_version, 1);
    assert_eq!(cfg.simulation.seed, 12345);
    assert_eq!(cfg.phases[0].as_md().unwrap().n_steps, 10);
    assert_eq!(cfg.phases[0].as_md().unwrap().dt, 1.0e-15);
    assert_eq!(cfg.simulation.temperature, 300.0);
    assert_eq!(cfg.phases[0].as_md().unwrap().integrator.kind, "velocity-verlet");
    assert!(!param_bool(&cfg.phases[0].as_md().unwrap().integrator, "lossless"));
    assert_eq!(cfg.particle_types.len(), 1);
    assert_eq!(cfg.particle_types[0].name, "Ar");
    assert_eq!(cfg.particle_types[0].mass, 6.6335e-26);
    assert_eq!(cfg.pair_interactions.len(), 1);
    assert_eq!(cfg.pair_interactions[0].between, ("Ar".to_string(), "Ar".to_string()));
    assert_eq!(cfg.pair_interactions[0].cutoff, 1.0e-9);
    assert!(matches!(
        cfg.pair_interactions[0].potential,
        dynamics::io::PairPotentialParams::LennardJones { sigma, epsilon }
            if sigma == 3.40e-10 && epsilon == 1.65e-21
    ));
    let canonical_dir = std::fs::canonicalize(&dir).unwrap();
    assert_eq!(cfg.init, canonical_dir.join("argon.in.xyz"));
    assert_eq!(cfg.config_path, canonical_dir.join("sim.in.toml"));
}

// rq-894c16c4
#[test]
fn defaults_populate_output_section() {
    let dir = tmp_path("defaults_populate_output");
    let path = write_config(&dir, &minimal_config());
    let cfg = load_config(&path).unwrap();
    let canonical_dir = std::fs::canonicalize(&dir).unwrap();
    assert_eq!(cfg.phases[0].as_md().unwrap().output.trajectory_path, canonical_dir.join("sim.out.run.xyz"));
    assert_eq!(cfg.phases[0].as_md().unwrap().output.trajectory_every, 100);
    assert!(cfg.phases[0].as_md().unwrap().output.include_velocities);
    assert_eq!(cfg.phases[0].as_md().unwrap().output.log_path, canonical_dir.join("sim.out.run.log"));
    assert_eq!(cfg.phases[0].as_md().unwrap().output.log_every, 100);
    assert_eq!(cfg.phases[0].as_md().unwrap().output.timings_path, canonical_dir.join("sim.out.run.timings"));
}

// rq-0622d4b0 — default output paths strip exactly one trailing `.in`.
#[test]
fn defaults_strip_only_one_trailing_in() {
    let dir = tmp_path("defaults_strip_one_in");
    let path = write_config_named(&dir, "foo.in.in.toml", &minimal_config());
    let cfg = load_config(&path).unwrap();
    let canonical_dir = std::fs::canonicalize(&dir).unwrap();
    assert_eq!(cfg.phases[0].as_md().unwrap().output.trajectory_path, canonical_dir.join("foo.in.out.run.xyz"));
    assert_eq!(cfg.phases[0].as_md().unwrap().output.log_path, canonical_dir.join("foo.in.out.run.log"));
    assert_eq!(cfg.phases[0].as_md().unwrap().output.timings_path, canonical_dir.join("foo.in.out.run.timings"));
}

// rq-d148149f
#[test]
fn explicit_output_overrides_defaults() {
    let dir = tmp_path("explicit_output");
    let body = format!(
        "{}\n[phase.output]\ntrajectory_path = \"custom-traj.xyz\"\ntrajectory_every = 50\nlog_path = \"custom.log\"\nlog_every = 25\ninclude_velocities = false\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    let canonical_dir = std::fs::canonicalize(&dir).unwrap();
    assert_eq!(cfg.phases[0].as_md().unwrap().output.trajectory_path, canonical_dir.join("custom-traj.xyz"));
    assert_eq!(cfg.phases[0].as_md().unwrap().output.trajectory_every, 50);
    assert!(!cfg.phases[0].as_md().unwrap().output.include_velocities);
    assert_eq!(cfg.phases[0].as_md().unwrap().output.log_path, canonical_dir.join("custom.log"));
    assert_eq!(cfg.phases[0].as_md().unwrap().output.log_every, 25);
}

// rq-5ded1806
#[test]
fn absolute_paths_honored() {
    let dir = tmp_path("absolute_paths");
    let abs_init = dir.join("abs-init.xyz");
    let abs_traj = dir.join("abs-traj.xyz");
    let abs_log = dir.join("abs.log");
    let body = format!(
        "schema_version = 1\ninit = \"{}\"\n[simulation]\nseed=1\ntemperature=0.0\n[[phase]]\nname=\"run\"\nn_steps=1\ndt=1.0e-15\n[phase.integrator]\nkind=\"velocity-verlet\"\nlossless=false\n[phase.output]\ntrajectory_path=\"{}\"\nlog_path=\"{}\"\n[[particle_types]]\nname=\"Ar\"\nmass=1.0\n[[pair_interactions]]\nbetween=[\"Ar\",\"Ar\"]\npotential=\"lennard-jones\"\nsigma=1.0\nepsilon=1.0\ncutoff=1.0\n",
        abs_init.display(),
        abs_traj.display(),
        abs_log.display(),
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.init, abs_init);
    assert_eq!(cfg.phases[0].as_md().unwrap().output.trajectory_path, abs_traj);
    assert_eq!(cfg.phases[0].as_md().unwrap().output.log_path, abs_log);
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
init = "init.in.xyz"

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
    assert_eq!(cfg.phases[0].as_md().unwrap().n_steps, 0);
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
    // Use the convention-satisfying suffix so the filename-convention
    // check passes and the loader proceeds to opening the file.
    let path = dir.join("nope.in.toml");
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
        ConfigError::Parse { .. } => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-761f26c6
#[test]
fn unknown_top_level_key_permitted() {
    let dir = tmp_path("unknown_top_level");
    // Insert the unknown key at the genuine top level, before the first
    // section header, so it is not absorbed into a table.
    let body = minimal_config().replace(
        "init = \"argon.in.xyz\"\n",
        "init = \"argon.in.xyz\"\nunknown_key = \"x\"\n",
    );
    let path = write_config(&dir, &body);
    load_config(&path).unwrap();
}

// rq-f0e3b004
#[test]
fn missing_init_field() {
    let dir = tmp_path("missing_init");
    let body = minimal_config().replace("init = \"argon.in.xyz\"\n", "");
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
        ConfigError::MissingField { field } => assert_eq!(field, "dt"),
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
        .replace("[phase.integrator]\nkind = \"velocity-verlet\"\nlossless = false\n", "");
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
init = "init.in.xyz"

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
init = "init.in.xyz"

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
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "phase[0].dt"),
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
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "phase[0].dt"),
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
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "phase[0].dt"),
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

// rq-d1d84e31
#[test]
fn accept_user_supplied_r_switch() {
    let dir = tmp_path("accept_r_switch");
    let body = minimal_config().replace(
        "cutoff = 1.0e-9\n",
        "cutoff = 1.0e-9\nr_switch = 9.0e-10\n",
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).expect("load_config");
    assert_eq!(cfg.pair_interactions[0].r_switch, 9.0e-10);
}

// rq-6f4f5ece rq-c195ddf0
#[test]
fn default_r_switch_to_0_9_times_cutoff_when_omitted() {
    let dir = tmp_path("default_r_switch");
    let path = write_config(&dir, &minimal_config());
    let cfg = load_config(&path).expect("load_config");
    // 0.9 * 1.0e-9 = 9.0e-10 exactly in f64 since 0.9 is not exact but
    // the relative tolerance below tolerates the round-off.
    let expected = 0.9_f64 * 1.0e-9;
    assert!((cfg.pair_interactions[0].r_switch - expected).abs() < 1.0e-25);
}

// rq-1d8b8efe
#[test]
fn accept_r_switch_equal_to_cutoff() {
    let dir = tmp_path("r_switch_eq_cutoff");
    let body = minimal_config().replace(
        "cutoff = 1.0e-9\n",
        "cutoff = 1.0e-9\nr_switch = 1.0e-9\n",
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).expect("load_config");
    assert_eq!(cfg.pair_interactions[0].r_switch, 1.0e-9);
}

// rq-7cd9471a
#[test]
fn reject_r_switch_greater_than_cutoff() {
    let dir = tmp_path("r_switch_gt_cutoff");
    let body = minimal_config().replace(
        "cutoff = 1.0e-9\n",
        "cutoff = 1.0e-9\nr_switch = 1.1e-9\n",
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => {
            assert_eq!(field, "pair_interactions[0].r_switch")
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-b4d2f559
#[test]
fn reject_nonpositive_r_switch() {
    let dir = tmp_path("r_switch_zero");
    let body = minimal_config().replace(
        "cutoff = 1.0e-9\n",
        "cutoff = 1.0e-9\nr_switch = 0.0\n",
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => {
            assert_eq!(field, "pair_interactions[0].r_switch")
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-871f0292
#[test]
fn reject_nonfinite_r_switch() {
    let dir = tmp_path("r_switch_nan");
    let body = minimal_config().replace(
        "cutoff = 1.0e-9\n",
        "cutoff = 1.0e-9\nr_switch = nan\n",
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => {
            assert_eq!(field, "pair_interactions[0].r_switch")
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-e38aac7b
#[test]
fn reject_unknown_potential() {
    let dir = tmp_path("unknown_potential");
    let body = minimal_config().replace("potential = \"lennard-jones\"", "potential = \"morse\"");
    let path = write_config(&dir, &body);
    assert_parse(
        &load_config(&path).unwrap_err(),
        "pair_interactions[0].potential",
    );
}

// rq-45a14d49
#[test]
fn reject_lennard_jones_missing_sigma() {
    let dir = tmp_path("lj_missing_sigma");
    let body = minimal_config().replace("sigma = 3.40e-10\n", "");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => {
            assert_eq!(field, "pair_interactions[0].sigma");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-053613b6
#[test]
fn reject_unknown_pair_interaction_field() {
    let dir = tmp_path("pair_unknown_field");
    let body = minimal_config().replace(
        "cutoff = 1.0e-9\n",
        "cutoff = 1.0e-9\nstiffness = 1.0\n",
    );
    let path = write_config(&dir, &body);
    assert_parse_path_and_field(
        &load_config(&path).unwrap_err(),
        "pair_interactions[0]",
        "stiffness",
    );
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

// rq-560dffb8 rq-f6107e43
#[test]
fn reject_duplicate_type_names() {
    let dir = tmp_path("duplicate_type_names");
    let body = r#"schema_version = 1
init = "init.in.xyz"

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
init = "init.in.xyz"

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
init = "init.in.xyz"

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
        "{}\n[phase.output]\ntrajectory_path = \"argon.in.xyz\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::PathCollision { kind_a, kind_b, .. } => {
            assert_eq!(kind_a, PathRole::Init);
            assert_eq!(kind_b, PathRole::PhaseTrajectory { phase: "run".into() });
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-765c96c5
#[test]
fn reject_trajectory_equals_log() {
    let dir = tmp_path("traj_eq_log");
    let body = format!(
        "{}\n[phase.output]\ntrajectory_path = \"run.dat\"\nlog_path = \"run.dat\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::PathCollision { kind_a, kind_b, .. } => {
            assert_eq!(kind_a, PathRole::PhaseTrajectory { phase: "run".into() });
            assert_eq!(kind_b, PathRole::PhaseLog { phase: "run".into() });
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-330d6b42
#[test]
fn reject_init_equals_log() {
    let dir = tmp_path("init_eq_log");
    let body = format!(
        "{}\n[phase.output]\nlog_path = \"argon.in.xyz\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::PathCollision { kind_a, kind_b, .. } => {
            assert_eq!(kind_a, PathRole::Init);
            assert_eq!(kind_b, PathRole::PhaseLog { phase: "run".into() });
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-f114c560
#[test]
fn accept_multi_type_with_complete_pair_table() {
    let dir = tmp_path("multi_type");
    let body = r#"schema_version = 1
init = "init.in.xyz"

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

// rq-66dfc50f
#[test]
fn reject_multi_type_with_missing_pair() {
    let dir = tmp_path("multi_type_missing_pair");
    let body = r#"schema_version = 1
init = "init.in.xyz"

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
    assert_eq!(cfg.phases[0].as_md().unwrap().output.timings_path, canonical_dir.join("sim.out.run.timings"));
}

// rq-fa24a8d1
#[test]
fn timings_path_override() {
    let dir = tmp_path("timings_override");
    let body = format!(
        "{}\n[phase.output]\ntimings_path = \"custom.timings\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    let canonical_dir = std::fs::canonicalize(&dir).unwrap();
    assert_eq!(cfg.phases[0].as_md().unwrap().output.timings_path, canonical_dir.join("custom.timings"));
}

// rq-7d5915bb
#[test]
fn reject_init_equals_timings() {
    let dir = tmp_path("init_eq_timings");
    let body = format!(
        "{}\n[phase.output]\ntimings_path = \"argon.in.xyz\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::PathCollision { kind_a, kind_b, .. } => {
            assert_eq!(kind_a, PathRole::Init);
            assert_eq!(kind_b, PathRole::PhaseTimings { phase: "run".into() });
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-ec8d715d
#[test]
fn reject_trajectory_equals_timings() {
    let dir = tmp_path("traj_eq_timings");
    let body = format!(
        "{}\n[phase.output]\ntrajectory_path = \"run.dat\"\ntimings_path = \"run.dat\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::PathCollision { kind_a, kind_b, .. } => {
            assert_eq!(kind_a, PathRole::PhaseTrajectory { phase: "run".into() });
            assert_eq!(kind_b, PathRole::PhaseTimings { phase: "run".into() });
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-8f665dd0
#[test]
fn reject_log_equals_timings() {
    let dir = tmp_path("log_eq_timings");
    let body = format!(
        "{}\n[phase.output]\nlog_path = \"run.dat\"\ntimings_path = \"run.dat\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::PathCollision { kind_a, kind_b, .. } => {
            assert_eq!(kind_a, PathRole::PhaseLog { phase: "run".into() });
            assert_eq!(kind_b, PathRole::PhaseTimings { phase: "run".into() });
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-100115a0
#[test]
fn missing_integrator_kind() {
    let dir = tmp_path("missing_integrator_kind");
    let body = minimal_config().replace(
        "[phase.integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[phase.integrator]\nlossless = false\n",
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "integrator.kind"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-657bbbd6
#[test]
fn unknown_integrator_kind() {
    let dir = tmp_path("unknown_integrator_kind");
    let body = minimal_config().replace("kind = \"velocity-verlet\"", "kind = \"custom\"");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::UnknownKind { slot, kind } => {
            assert_eq!(slot, "integrator");
            assert_eq!(kind, "custom");
        }
        other => panic!("expected UnknownKind, got {other:?}"),
    }
}

// rq-86aa2be7
#[test]
fn langevin_with_valid_parameters_accepted() {
    let dir = tmp_path("langevin_valid");
    let body = minimal_config().replace(
        "[phase.integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[phase.integrator]\nkind = \"langevin-baoab\"\nfriction = 1.0e12\ntemperature = 300.0\nseed = 42\n",
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.phases[0].as_md().unwrap().integrator.kind, "langevin-baoab");
    assert_eq!(param_f64(&cfg.phases[0].as_md().unwrap().integrator, "friction"), 1.0e12);
    assert_eq!(param_f64(&cfg.phases[0].as_md().unwrap().integrator, "temperature"), 300.0);
    assert_eq!(param_u64(&cfg.phases[0].as_md().unwrap().integrator, "seed"), 42);
}

// rq-40ed9975
#[test]
fn langevin_missing_friction() {
    let dir = tmp_path("langevin_missing_friction");
    let body = minimal_config().replace(
        "[phase.integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[phase.integrator]\nkind = \"langevin-baoab\"\ntemperature = 300.0\nseed = 42\n",
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
        "[phase.integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[phase.integrator]\nkind = \"langevin-baoab\"\nfriction = 1.0e12\nseed = 42\n",
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
        "[phase.integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[phase.integrator]\nkind = \"langevin-baoab\"\nfriction = 1.0e12\ntemperature = 300.0\n",
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
        "[phase.integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[phase.integrator]\nkind = \"langevin-baoab\"\nfriction = 0.0\ntemperature = 300.0\nseed = 42\n",
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
        "[phase.integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[phase.integrator]\nkind = \"langevin-baoab\"\nfriction = -1.0\ntemperature = 300.0\nseed = 42\n",
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
        "[phase.integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[phase.integrator]\nkind = \"langevin-baoab\"\nfriction = 1.0e12\ntemperature = 0.0\nseed = 42\n",
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "integrator.temperature"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-f067338a rq-aa1492a7
#[test]
fn integrator_rejects_unknown_field_for_chosen_kind() {
    // Consolidates the per-kind "unknown integrator field" check across
    // every variant in IntegratorKind. Each row is one (kind, body,
    // expected-path) tuple.
    // Each row: (label, integrator-block, unknown-field-name). The
    // deserialiser reports the parent table's path on the Parse error;
    // the message itself names the unknown field. Each case checks both:
    // the path equals "integrator" and the message mentions the offending
    // field name.
    let cases: &[(&str, &str, &str)] = &[
        (
            "velocity-verlet",
            "[phase.integrator]\nkind = \"velocity-verlet\"\nfriction = 1.0e12\n",
            "friction",
        ),
        (
            "langevin-baoab",
            "[phase.integrator]\nkind = \"langevin-baoab\"\nfriction = 1.0e12\ntemperature = 300.0\nseed = 42\nlossless = false\n",
            "lossless",
        ),
        (
            "mtk-npt",
            "[phase.integrator]\nkind = \"mtk-npt\"\ntemperature = 85.0\npressure = 1.0e5\ntau_t = 1.0e-13\ntau_p = 1.0e-12\nseed = 42\n",
            "seed",
        ),
    ];
    for (label, integrator_block, unknown_field) in cases {
        let dir = tmp_path(&format!("integrator_unknown_field_{label}"));
        let body = minimal_config().replace(
            "[phase.integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
            integrator_block,
        );
        let path = write_config(&dir, &body);
        let err = load_config(&path).unwrap_err();
        assert_parse_path_and_field(&err, "integrator", unknown_field);
    }
}

// rq-66ec7ee4
#[test]
fn vv_lossless_defaults_to_false() {
    let dir = tmp_path("vv_lossless_default");
    let body = minimal_config().replace(
        "[phase.integrator]\nkind = \"velocity-verlet\"\nlossless = false\n",
        "[phase.integrator]\nkind = \"velocity-verlet\"\n",
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.phases[0].as_md().unwrap().integrator.kind, "velocity-verlet");
    assert!(!param_bool_or(&cfg.phases[0].as_md().unwrap().integrator, "lossless", false));
}

// rq-6cb9ab62
#[test]
fn bonds_field_optional_defaults_none() {
    let dir = tmp_path("bonds_optional");
    let path = write_config(&dir, &minimal_config());
    let cfg = load_config(&path).unwrap();
    assert!(cfg.topology.is_none());
}

// rq-027153d9
#[test]
fn bonds_field_resolved_relative() {
    let dir = tmp_path("bonds_relative");
    let body = minimal_config().replace(
        "init = \"argon.in.xyz\"\n",
        "init = \"argon.in.xyz\"\ntopology = \"argon.in.topology\"\n",
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    let canonical_dir = std::fs::canonicalize(&dir).unwrap();
    assert_eq!(cfg.topology, Some(canonical_dir.join("argon.in.topology")));
}

// rq-576561a2
#[test]
fn bonds_absolute_preserved() {
    let dir = tmp_path("bonds_absolute");
    let body = minimal_config().replace(
        "init = \"argon.in.xyz\"\n",
        "init = \"argon.in.xyz\"\ntopology = \"/data/argon.in.topology\"\n",
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.topology, Some(std::path::PathBuf::from("/data/argon.in.topology")));
}

// rq-4186d4f4
#[test]
fn reject_bonds_eq_init() {
    let dir = tmp_path("bonds_eq_init");
    let body = minimal_config().replace(
        "init = \"argon.in.xyz\"\n",
        "init = \"argon.in.xyz\"\ntopology = \"argon.in.xyz\"\n",
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::PathCollision { kind_a, kind_b, .. } => {
            assert_eq!(kind_a, PathRole::Init);
            assert_eq!(kind_b, PathRole::Topology);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-98180119
#[test]
fn reject_bonds_eq_trajectory() {
    let dir = tmp_path("bonds_eq_traj");
    let body = format!(
        "{}\n[phase.output]\ntrajectory_path = \"run.dat\"\n",
        minimal_config().replace(
            "init = \"argon.in.xyz\"\n",
            "init = \"argon.in.xyz\"\ntopology = \"run.dat\"\n",
        )
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::PathCollision { kind_a, kind_b, .. } => {
            assert!(matches!(kind_a, PathRole::PhaseTrajectory { .. } | PathRole::Topology));
            assert!(matches!(kind_b, PathRole::PhaseTrajectory { .. } | PathRole::Topology));
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

// rq-3f01c746
#[test]
fn bond_type_unknown_potential() {
    let dir = tmp_path("bond_type_unknown");
    let body = format!(
        "{}\n[[bond_types]]\nname = \"ArAr\"\npotential = \"harmonic\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    assert_parse(
        &load_config(&path).unwrap_err(),
        "bond_types[0].potential",
    );
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

// rq-a208c9ba
#[test]
fn morse_bond_type_rejects_extra_field() {
    let dir = tmp_path("morse_extra_field");
    let body = format!(
        "{}\n[[bond_types]]\nname = \"X\"\npotential = \"morse\"\nde = 1.0\na = 1.0\nre = 1.0\nstiffness = 2.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    assert_parse_path_and_field(
        &load_config(&path).unwrap_err(),
        "bond_types[0]",
        "stiffness",
    );
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
        "{}\n[phase.output]\ntrajectory_every = 0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.phases[0].as_md().unwrap().output.trajectory_every, 0);
}

// rq-318cd47d
#[test]
fn log_every_zero_accepted() {
    let dir = tmp_path("log_every_zero");
    let body = format!("{}\n[phase.output]\nlog_every = 0\n", minimal_config());
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.phases[0].as_md().unwrap().output.log_every, 0);
}

// --- Neighbor list ---

// rq-e2f32af0
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

// rq-b1f33ea4
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

// rq-e643b070
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

// rq-cde6e114
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

// rq-931f1ab8
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

// rq-0a92d90b
#[test]
fn neighbor_list_unknown_mode_rejected() {
    let dir = tmp_path("nl_unknown_mode");
    let body = format!(
        "{}\n[neighbor_list]\nmode = \"kd-tree\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    assert_parse(&load_config(&path).unwrap_err(), "neighbor_list.mode");
}

// rq-13ca0415
#[test]
fn neighbor_list_rejects_unknown_field_for_chosen_mode() {
    // Consolidates the per-mode "unknown neighbor_list field" check.
    // Each row is one (label, body, expected-path) tuple.
    // Each row: (label, neighbor_list-block, unknown-field-name).
    let cases: &[(&str, &str, &str)] = &[
        (
            "all_pairs_rejects_max_neighbors",
            "[neighbor_list]\nmode = \"all-pairs\"\nmax_neighbors = 64\n",
            "max_neighbors",
        ),
        (
            "all_pairs_rejects_r_skin",
            "[neighbor_list]\nmode = \"all-pairs\"\nr_skin = 1.0e-10\n",
            "r_skin",
        ),
        (
            "cell_list_rejects_stencil",
            "[neighbor_list]\nmode = \"cell-list\"\nstencil = \"huge\"\n",
            "stencil",
        ),
    ];
    for (label, nl_block, unknown_field) in cases {
        let dir = tmp_path(label);
        let body = format!("{}\n{}", minimal_config(), nl_block);
        let path = write_config(&dir, &body);
        let err = load_config(&path).unwrap_err();
        assert_parse_path_and_field(&err, "neighbor_list", unknown_field);
    }
}

// rq-fedef74d
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

// rq-f7856bcc rq-ec28cbfb
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

// --- Angle types ---

// rq-24dc9578
#[test]
fn angle_types_optional_empty() {
    let dir = tmp_path("angle_types_optional");
    let path = write_config(&dir, &minimal_config());
    let cfg = load_config(&path).unwrap();
    assert!(cfg.angle_types.is_empty());
}

// rq-91bf10ec
#[test]
fn valid_harmonic_angle_type_accepted() {
    let dir = tmp_path("angle_types_harmonic");
    let body = format!(
        "{}\n[[angle_types]]\nname = \"HOH\"\npotential = \"harmonic\"\nk_theta = 5.27e-19\ntheta_0 = 1.911\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.angle_types.len(), 1);
    match &cfg.angle_types[0] {
        dynamics::io::config::AngleTypeConfig::Harmonic { name, k_theta, theta_0 } => {
            assert_eq!(name, "HOH");
            assert!((k_theta - 5.27e-19).abs() < 1.0e-28);
            assert!((theta_0 - 1.911).abs() < 1.0e-9);
        }
    }
}

// rq-57518e01 rq-dc94d9e3
#[test]
fn angle_type_missing_potential_rejected() {
    let dir = tmp_path("angle_no_pot");
    let body = format!(
        "{}\n[[angle_types]]\nname = \"X\"\nk_theta = 1.0\ntheta_0 = 1.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => {
            assert_eq!(field, "angle_types[0].potential");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-ffa771bd
#[test]
fn angle_type_unknown_potential_rejected() {
    let dir = tmp_path("angle_unk_pot");
    let body = format!(
        "{}\n[[angle_types]]\nname = \"X\"\npotential = \"cosine-harmonic\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    assert_parse(
        &load_config(&path).unwrap_err(),
        "angle_types[0].potential",
    );
}

// rq-aad6ca63
#[test]
fn harmonic_angle_rejects_non_positive_k_theta() {
    let dir = tmp_path("angle_k_neg");
    let body = format!(
        "{}\n[[angle_types]]\nname = \"X\"\npotential = \"harmonic\"\nk_theta = 0.0\ntheta_0 = 1.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => {
            assert_eq!(field, "angle_types[0].k_theta");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-e399422c
#[test]
fn harmonic_angle_rejects_theta_0_outside_zero_pi() {
    let dir = tmp_path("angle_t0_oor");
    let body = format!(
        "{}\n[[angle_types]]\nname = \"X\"\npotential = \"harmonic\"\nk_theta = 1.0\ntheta_0 = 4.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => {
            assert_eq!(field, "angle_types[0].theta_0");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-c5fa34f5
#[test]
fn harmonic_angle_rejects_extra_fields() {
    let dir = tmp_path("angle_extra");
    let body = format!(
        "{}\n[[angle_types]]\nname = \"X\"\npotential = \"harmonic\"\nk_theta = 1.0\ntheta_0 = 1.0\nstiffness = 2.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    assert_parse_path_and_field(
        &load_config(&path).unwrap_err(),
        "angle_types[0]",
        "stiffness",
    );
}

// rq-9255c192
#[test]
fn reject_duplicate_angle_type_name() {
    let dir = tmp_path("angle_dup_name");
    let body = format!(
        "{}\n[[angle_types]]\nname = \"X\"\npotential = \"harmonic\"\nk_theta = 1.0\ntheta_0 = 1.0\n\n[[angle_types]]\nname = \"X\"\npotential = \"harmonic\"\nk_theta = 1.0\ntheta_0 = 1.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::DuplicateAngleTypeName { name } => assert_eq!(name, "X"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-dc35ae30
#[test]
fn empty_angle_type_name_rejected() {
    let dir = tmp_path("angle_empty_name");
    let body = format!(
        "{}\n[[angle_types]]\nname = \"\"\npotential = \"harmonic\"\nk_theta = 1.0\ntheta_0 = 1.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => {
            assert_eq!(field, "angle_types[0].name");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// =====================================================================
// [thermostat] section
// =====================================================================

// Build a config body with `[integrator] kind="velocity-verlet"` and
// an injected `[thermostat]` block. The `[thermostat]` body is
// supplied by callers as a complete TOML fragment beginning with the
// `[thermostat]` header so they can omit individual fields to test
// MissingField paths.
fn config_with_thermostat(thermostat_block: &str) -> String {
    let phased = thermostat_block
        .replace("[thermostat]", "[phase.thermostat]")
        .replace("[integrator]", "[phase.integrator]");
    format!(
        r#"schema_version = 1
units = "atomic"
init = "argon.in.xyz"

[simulation]
seed = 12345
temperature = 300.0

[[phase]]
name = "run"
n_steps = 10
dt = 1.0e-15


[phase.integrator]
kind = "velocity-verlet"
lossless = false

{phased}

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
    )
}

// --- [thermostat] presence / absence ---

// rq-ca356c08
#[test]
fn thermostat_section_absent_yields_none() {
    let dir = tmp_path("therm_absent");
    let path = write_config(&dir, &minimal_config());
    let cfg = load_config(&path).unwrap();
    assert!(cfg.phases[0].as_md().unwrap().thermostat.is_none());
}

// rq-19ccc047
#[test]
fn thermostat_unknown_kind_rejected() {
    let dir = tmp_path("therm_unknown");
    let body = config_with_thermostat(
        r#"[thermostat]
kind = "not-a-real-thermostat""#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::UnknownKind { slot, kind } => {
            assert_eq!(slot, "thermostat");
            assert_eq!(kind, "not-a-real-thermostat");
        }
        other => panic!("expected UnknownKind, got {other:?}"),
    }
}

// rq-c4a74903
#[test]
fn thermostat_missing_kind_rejected() {
    let dir = tmp_path("therm_no_kind");
    let body = config_with_thermostat(
        r#"[thermostat]
temperature = 300.0"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "thermostat.kind"),
        other => panic!("unexpected: {other:?}"),
    }
}

// --- Nosé-Hoover chain ---

#[test]
fn thermostat_nhc_defaults_accepted() {
    let dir = tmp_path("nhc_defaults");
    let body = config_with_thermostat(
        r#"[thermostat]
kind = "nose-hoover-chain"
temperature = 300.0
tau = 1.0e-13"#,
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    let t = cfg.phases[0].as_md().unwrap().thermostat.as_ref().unwrap();
    assert_eq!(t.kind, "nose-hoover-chain");
    assert_eq!(param_f64(t, "temperature"), 300.0);
    assert_eq!(param_f64(t, "tau"), 1.0e-13);
    // Defaults for the optional fields are applied by the builder's
    // validate_params / build at consume time; not present in the
    // parsed params (they were not in the TOML).
    assert!(t.params.get("chain_length").is_none());
    assert!(t.params.get("yoshida_order").is_none());
    assert!(t.params.get("n_resp").is_none());
}

#[test]
fn thermostat_nhc_explicit_chain_params_accepted() {
    let dir = tmp_path("nhc_explicit");
    let body = config_with_thermostat(
        r#"[thermostat]
kind = "nose-hoover-chain"
temperature = 300.0
tau = 1.0e-13
chain_length = 5
yoshida_order = 5
n_resp = 2"#,
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    let t = cfg.phases[0].as_md().unwrap().thermostat.as_ref().unwrap();
    assert_eq!(t.kind, "nose-hoover-chain");
    assert_eq!(param_u64(t, "chain_length"), 5);
    assert_eq!(param_u64(t, "yoshida_order"), 5);
    assert_eq!(param_u64(t, "n_resp"), 2);
}

#[test]
fn thermostat_nhc_missing_temperature_rejected() {
    let dir = tmp_path("nhc_no_t");
    let body = config_with_thermostat(
        r#"[thermostat]
kind = "nose-hoover-chain"
tau = 1.0e-13"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "thermostat.temperature"),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn thermostat_nhc_missing_tau_rejected() {
    let dir = tmp_path("nhc_no_tau");
    let body = config_with_thermostat(
        r#"[thermostat]
kind = "nose-hoover-chain"
temperature = 300.0"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "thermostat.tau"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-e5b63a73
#[test]
fn thermostat_nhc_rejects_non_positive_temperature() {
    let dir = tmp_path("nhc_T_zero");
    let body = config_with_thermostat(
        r#"[thermostat]
kind = "nose-hoover-chain"
temperature = 0.0
tau = 1.0e-13"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "thermostat.temperature"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-d532de58
#[test]
fn thermostat_nhc_rejects_non_positive_tau() {
    let dir = tmp_path("nhc_tau_neg");
    let body = config_with_thermostat(
        r#"[thermostat]
kind = "nose-hoover-chain"
temperature = 300.0
tau = -1.0e-13"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "thermostat.tau"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-811c598f
#[test]
fn thermostat_nhc_rejects_chain_length_zero() {
    let dir = tmp_path("nhc_chain0");
    let body = config_with_thermostat(
        r#"[thermostat]
kind = "nose-hoover-chain"
temperature = 300.0
tau = 1.0e-13
chain_length = 0"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "thermostat.chain_length"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-6dc8454d
#[test]
fn thermostat_nhc_rejects_yoshida_order_outside_allowed_set() {
    let dir = tmp_path("nhc_yoshida2");
    let body = config_with_thermostat(
        r#"[thermostat]
kind = "nose-hoover-chain"
temperature = 300.0
tau = 1.0e-13
yoshida_order = 2"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "thermostat.yoshida_order"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-dd6fe266
#[test]
fn thermostat_nhc_rejects_n_resp_zero() {
    let dir = tmp_path("nhc_nresp0");
    let body = config_with_thermostat(
        r#"[thermostat]
kind = "nose-hoover-chain"
temperature = 300.0
tau = 1.0e-13
n_resp = 0"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "thermostat.n_resp"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-f4eeb849
#[test]
fn thermostat_rejects_unknown_field_for_chosen_kind() {
    // Consolidates the per-kind "unknown thermostat field" check across
    // every variant in ThermostatKind. Each row is one (label,
    // thermostat-block, expected-path) tuple.
    // Each row: (label, thermostat-block, unknown-field-name). The
    // deserialiser reports `path = "thermostat"`; the message names the
    // offending field.
    let cases: &[(&str, &str, &str)] = &[
        (
            "nhc_extra_friction",
            r#"[thermostat]
kind = "nose-hoover-chain"
temperature = 300.0
tau = 1.0e-13
friction = 1.0e12"#,
            "friction",
        ),
        (
            "csvr_extra_chain_length",
            r#"[thermostat]
kind = "csvr"
temperature = 300.0
tau = 1.0e-13
seed = 1
chain_length = 3"#,
            "chain_length",
        ),
        (
            "andersen_extra_tau",
            r#"[thermostat]
kind = "andersen"
temperature = 300.0
collision_rate = 1.0e12
seed = 1
tau = 1.0e-13"#,
            "tau",
        ),
        (
            "berendsen_extra_seed",
            r#"[thermostat]
kind = "berendsen"
temperature = 300.0
tau = 1.0e-13
seed = 42"#,
            "seed",
        ),
    ];
    for (label, thermostat_block, unknown_field) in cases {
        let dir = tmp_path(label);
        let body = config_with_thermostat(thermostat_block);
        let path = write_config(&dir, &body);
        let err = load_config(&path).unwrap_err();
        assert_parse_path_and_field(&err, "thermostat", unknown_field);
    }
}

// --- CSVR ---

#[test]
fn thermostat_csvr_accepted() {
    let dir = tmp_path("csvr_accepted");
    let body = config_with_thermostat(
        r#"[thermostat]
kind = "csvr"
temperature = 300.0
tau = 1.0e-13
seed = 42"#,
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    let t = cfg.phases[0].as_md().unwrap().thermostat.as_ref().unwrap();
    assert_eq!(t.kind, "csvr");
    assert_eq!(param_f64(t, "temperature"), 300.0);
    assert_eq!(param_f64(t, "tau"), 1.0e-13);
    assert_eq!(param_u64(t, "seed"), 42);
}

// rq-5c28eee0 rq-84927a79
#[test]
fn thermostat_csvr_missing_seed_rejected() {
    let dir = tmp_path("csvr_no_seed");
    let body = config_with_thermostat(
        r#"[thermostat]
kind = "csvr"
temperature = 300.0
tau = 1.0e-13"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "thermostat.seed"),
        other => panic!("unexpected: {other:?}"),
    }
}

// CSVR extra-fields coverage lives in the parameterised
// `thermostat_rejects_unknown_field_for_chosen_kind` test above.

// --- Andersen ---

#[test]
fn thermostat_andersen_accepted() {
    let dir = tmp_path("andersen_accepted");
    let body = config_with_thermostat(
        r#"[thermostat]
kind = "andersen"
temperature = 300.0
collision_rate = 1.0e12
seed = 42"#,
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    let t = cfg.phases[0].as_md().unwrap().thermostat.as_ref().unwrap();
    assert_eq!(t.kind, "andersen");
    assert_eq!(param_f64(t, "temperature"), 300.0);
    assert_eq!(param_f64(t, "collision_rate"), 1.0e12);
    assert_eq!(param_u64(t, "seed"), 42);
}

// rq-c4581536
#[test]
fn thermostat_andersen_accepts_collision_rate_zero() {
    let dir = tmp_path("andersen_rate_zero");
    let body = config_with_thermostat(
        r#"[thermostat]
kind = "andersen"
temperature = 300.0
collision_rate = 0.0
seed = 42"#,
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    let t = cfg.phases[0].as_md().unwrap().thermostat.as_ref().unwrap();
    assert_eq!(t.kind, "andersen");
    assert_eq!(param_f64(t, "collision_rate"), 0.0);
}

// rq-0f3b352b
#[test]
fn thermostat_andersen_rejects_negative_collision_rate() {
    let dir = tmp_path("andersen_rate_neg");
    let body = config_with_thermostat(
        r#"[thermostat]
kind = "andersen"
temperature = 300.0
collision_rate = -1.0
seed = 42"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "thermostat.collision_rate"),
        other => panic!("unexpected: {other:?}"),
    }
}

// Andersen extra-fields coverage lives in the parameterised
// `thermostat_rejects_unknown_field_for_chosen_kind` test above.

// --- Berendsen ---

// rq-b7cd6d16
#[test]
fn thermostat_berendsen_accepted() {
    let dir = tmp_path("berendsen_accepted");
    let body = config_with_thermostat(
        r#"[thermostat]
kind = "berendsen"
temperature = 300.0
tau = 1.0e-13"#,
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    let t = cfg.phases[0].as_md().unwrap().thermostat.as_ref().unwrap();
    assert_eq!(t.kind, "berendsen");
    assert_eq!(param_f64(t, "temperature"), 300.0);
    assert_eq!(param_f64(t, "tau"), 1.0e-13);
}

// Berendsen extra-fields coverage lives in the parameterised
// `thermostat_rejects_unknown_field_for_chosen_kind` test above.

// --- Integrator-owns-thermostat compatibility ---

// rq-bdd03f85
#[test]
fn langevin_with_thermostat_is_rejected() {
    let dir = tmp_path("incompat_langevin_therm");
    let body = format!(
        r#"schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 12345
temperature = 300.0

[[phase]]
name = "run"
n_steps = 10
dt = 1.0e-15


[phase.integrator]
kind = "langevin-baoab"
friction = 1.0e12
temperature = 300.0
seed = 1

[phase.thermostat]
kind = "csvr"
temperature = 300.0
tau = 1.0e-13
seed = 2

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
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::IncompatibleThermostat { integrator, phase: _ } => {
            assert_eq!(integrator, "langevin-baoab");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-c4ae19e0
#[test]
fn velocity_verlet_with_thermostat_is_accepted() {
    let dir = tmp_path("vv_plus_csvr");
    let body = config_with_thermostat(
        r#"[thermostat]
kind = "csvr"
temperature = 300.0
tau = 1.0e-13
seed = 1"#,
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.phases[0].as_md().unwrap().integrator.kind, "velocity-verlet");
    assert_eq!(cfg.phases[0].as_md().unwrap().thermostat.as_ref().unwrap().kind, "csvr");
}

// rq-4cd2ec5b
#[test]
fn integrator_kind_owns_thermostat_matrix_config_layer() {
    let registry = IntegratorRegistry::with_builtins();
    let vv = SlotConfig::from_params_str("velocity-verlet", "lossless = false\n");
    let lan = SlotConfig::from_params_str(
        "langevin-baoab",
        "friction = 1.0e12\ntemperature = 300.0\nseed = 0\n",
    );
    let mtk = SlotConfig::from_params_str(
        "mtk-npt",
        "temperature = 85.0\npressure = 1.0e5\ntau_t = 1.0e-13\ntau_p = 1.0e-12\nchain_length = 3\nyoshida_order = 3\nn_resp = 1\n",
    );
    let owns = |slot: &SlotConfig| {
        registry
            .lookup(&slot.kind)
            .unwrap()
            .owns_thermostat(&slot.params)
    };
    assert!(!owns(&vv));
    assert!(owns(&lan));
    assert!(owns(&mtk));
}

// rq-014a82ef
#[test]
fn integrator_kind_owns_barostat_matrix_config_layer() {
    let registry = IntegratorRegistry::with_builtins();
    let vv = SlotConfig::from_params_str("velocity-verlet", "lossless = false\n");
    let lan = SlotConfig::from_params_str(
        "langevin-baoab",
        "friction = 1.0e12\ntemperature = 300.0\nseed = 0\n",
    );
    let mtk = SlotConfig::from_params_str(
        "mtk-npt",
        "temperature = 85.0\npressure = 1.0e5\ntau_t = 1.0e-13\ntau_p = 1.0e-12\nchain_length = 3\nyoshida_order = 3\nn_resp = 1\n",
    );
    let owns = |slot: &SlotConfig| {
        registry
            .lookup(&slot.kind)
            .unwrap()
            .owns_barostat(&slot.params)
    };
    assert!(!owns(&vv));
    assert!(!owns(&lan));
    assert!(owns(&mtk));
}

// --- mtk-npt parsing + incompatibility ---

fn mtk_minimal_body(extras: &str) -> String {
    format!(
        r#"schema_version = 1
units = "atomic"
init = "argon.in.xyz"

[simulation]
seed = 12345
temperature = 85.0

[[phase]]
name = "run"
n_steps = 10
dt = 1.0e-15


[phase.integrator]
kind = "mtk-npt"
temperature = 85.0
pressure = 1.0e5
tau_t = 1.0e-13
tau_p = 1.0e-12
{extras}

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
    )
}

// rq-23806456
#[test]
fn mtk_npt_with_defaults_accepted() {
    let dir = tmp_path("mtk_defaults");
    let body = mtk_minimal_body("");
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    let i = &cfg.phases[0].as_md().unwrap().integrator;
    assert_eq!(i.kind, "mtk-npt");
    assert_eq!(param_f64(i, "temperature"), 85.0);
    assert_eq!(param_f64(i, "pressure"), 1.0e5);
    assert_eq!(param_f64(i, "tau_t"), 1.0e-13);
    assert_eq!(param_f64(i, "tau_p"), 1.0e-12);
    // chain_length, yoshida_order, n_resp are not in the TOML; the
    // builder applies its serde defaults during build, but the raw
    // params on the SlotConfig only carry the user-provided fields.
    assert!(i.params.get("chain_length").is_none());
    assert!(i.params.get("yoshida_order").is_none());
    assert!(i.params.get("n_resp").is_none());
}

// rq-d572a90a rq-08e113ca
#[test]
fn mtk_npt_missing_tau_p_rejected() {
    let dir = tmp_path("mtk_no_tau_p");
    let body = mtk_minimal_body("").replace("tau_p = 1.0e-12\n", "");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "integrator.tau_p"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-775b0833
#[test]
fn mtk_npt_rejects_non_positive_tau_p() {
    let dir = tmp_path("mtk_tau_p_zero");
    let body = mtk_minimal_body("").replace("tau_p = 1.0e-12", "tau_p = 0.0");
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "integrator.tau_p"),
        other => panic!("unexpected: {other:?}"),
    }
}

// `mtk-npt` extra-fields coverage lives in the parameterised
// `integrator_rejects_unknown_field_for_chosen_kind` test above.

// rq-129edb76 rq-6478b9c9
#[test]
fn mtk_npt_with_thermostat_is_rejected() {
    let dir = tmp_path("mtk_plus_therm");
    let body = format!(
        r#"schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 12345
temperature = 85.0

[[phase]]
name = "run"
n_steps = 10
dt = 1.0e-15


[phase.integrator]
kind = "mtk-npt"
temperature = 85.0
pressure = 1.0e5
tau_t = 1.0e-13
tau_p = 1.0e-12

[phase.thermostat]
kind = "csvr"
temperature = 85.0
tau = 1.0e-13
seed = 1

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
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::IncompatibleThermostat { integrator, phase: _ } => {
            assert_eq!(integrator, "mtk-npt");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-fbb836fb rq-1b467c03
#[test]
fn mtk_npt_with_barostat_is_rejected() {
    let dir = tmp_path("mtk_plus_baro");
    let body = format!(
        r#"schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 12345
temperature = 85.0

[[phase]]
name = "run"
n_steps = 10
dt = 1.0e-15


[phase.integrator]
kind = "mtk-npt"
temperature = 85.0
pressure = 1.0e5
tau_t = 1.0e-13
tau_p = 1.0e-12

[phase.barostat]
kind = "berendsen"
pressure = 1.0e5
tau = 1.0e-12
compressibility = 4.5e-10

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
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::IncompatibleBarostat { integrator, phase: _ } => {
            assert_eq!(integrator, "mtk-npt");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// --- [barostat] section (always rejected with the empty registry) ---

// rq-4bbbada4
#[test]
fn barostat_section_absent_yields_none() {
    let dir = tmp_path("baro_absent");
    let path = write_config(&dir, &minimal_config());
    let cfg = load_config(&path).unwrap();
    assert!(cfg.phases[0].as_md().unwrap().barostat.is_none());
}

// rq-f03e2af2
#[test]
fn barostat_unknown_kind_rejected() {
    let dir = tmp_path("baro_unknown");
    let body = format!(
        r#"schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 12345
temperature = 300.0

[[phase]]
name = "run"
n_steps = 10
dt = 1.0e-15


[phase.integrator]
kind = "velocity-verlet"
lossless = false

[phase.barostat]
kind = "not-a-real-barostat"

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
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::UnknownKind { slot, kind } => {
            assert_eq!(slot, "barostat");
            assert_eq!(kind, "not-a-real-barostat");
        }
        other => panic!("expected UnknownKind, got {other:?}"),
    }
}

// Helper for the [barostat] kind="berendsen" scenarios below: the
// returned config body has `[integrator] kind="velocity-verlet"` plus
// the supplied `[barostat]` fragment.
fn config_with_barostat(barostat_block: &str) -> String {
    let phased = barostat_block
        .replace("[barostat]", "[phase.barostat]")
        .replace("[thermostat]", "[phase.thermostat]")
        .replace("[integrator]", "[phase.integrator]");
    format!(
        r#"schema_version = 1
units = "atomic"
init = "argon.in.xyz"

[simulation]
seed = 12345
temperature = 300.0

[[phase]]
name = "run"
n_steps = 10
dt = 1.0e-15


[phase.integrator]
kind = "velocity-verlet"
lossless = false

{phased}

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
    )
}

// rq-0fcb5a1e
#[test]
fn barostat_berendsen_accepted() {
    let dir = tmp_path("baro_berendsen");
    let body = config_with_barostat(
        r#"[barostat]
kind = "berendsen"
pressure = 1.0e5
tau = 1.0e-12
compressibility = 4.5e-10"#,
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    let b = cfg.phases[0].as_md().unwrap().barostat.as_ref().unwrap();
    assert_eq!(b.kind, "berendsen");
    assert_eq!(param_f64(b, "pressure"), 1.0e5);
    assert_eq!(param_f64(b, "tau"), 1.0e-12);
    assert_eq!(param_f64(b, "compressibility"), 4.5e-10);
}

// rq-2adf8b58
#[test]
fn barostat_berendsen_accepts_negative_pressure() {
    let dir = tmp_path("baro_berendsen_neg");
    let body = config_with_barostat(
        r#"[barostat]
kind = "berendsen"
pressure = -1.0e5
tau = 1.0e-12
compressibility = 4.5e-10"#,
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert!(cfg.phases[0].as_md().unwrap().barostat.is_some());
}

// rq-7ac01c02
#[test]
fn barostat_berendsen_missing_pressure_rejected() {
    let dir = tmp_path("baro_berendsen_no_p");
    let body = config_with_barostat(
        r#"[barostat]
kind = "berendsen"
tau = 1.0e-12
compressibility = 4.5e-10"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "barostat.pressure"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-5dce727f
#[test]
fn barostat_berendsen_missing_tau_rejected() {
    let dir = tmp_path("baro_berendsen_no_tau");
    let body = config_with_barostat(
        r#"[barostat]
kind = "berendsen"
pressure = 1.0e5
compressibility = 4.5e-10"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "barostat.tau"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-414ed5ef
#[test]
fn barostat_berendsen_missing_compressibility_rejected() {
    let dir = tmp_path("baro_berendsen_no_beta");
    let body = config_with_barostat(
        r#"[barostat]
kind = "berendsen"
pressure = 1.0e5
tau = 1.0e-12"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => {
            assert_eq!(field, "barostat.compressibility")
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-125677a3
#[test]
fn barostat_berendsen_rejects_non_positive_tau() {
    let dir = tmp_path("baro_berendsen_tau_neg");
    let body = config_with_barostat(
        r#"[barostat]
kind = "berendsen"
pressure = 1.0e5
tau = -1.0e-12
compressibility = 4.5e-10"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "barostat.tau"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-06772617
#[test]
fn barostat_berendsen_rejects_non_positive_compressibility() {
    let dir = tmp_path("baro_berendsen_beta_zero");
    let body = config_with_barostat(
        r#"[barostat]
kind = "berendsen"
pressure = 1.0e5
tau = 1.0e-12
compressibility = 0.0"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => {
            assert_eq!(field, "barostat.compressibility")
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-5d91f07d
#[test]
fn barostat_rejects_unknown_field_for_chosen_kind() {
    // Consolidates the per-kind "unknown barostat field" check across
    // every variant in BarostatKind.
    // Each row: (label, barostat-block, unknown-field-name).
    let cases: &[(&str, &str, &str)] = &[
        (
            "berendsen_extra_seed",
            r#"[barostat]
kind = "berendsen"
pressure = 1.0e5
tau = 1.0e-12
compressibility = 4.5e-10
seed = 42"#,
            "seed",
        ),
        (
            "c_rescale_extra_friction",
            r#"[barostat]
kind = "c-rescale"
pressure = 1.0e5
temperature = 85.0
tau = 1.0e-12
compressibility = 4.5e-10
seed = 42
friction = 1.0e12"#,
            "friction",
        ),
    ];
    for (label, barostat_block, unknown_field) in cases {
        let dir = tmp_path(label);
        let body = config_with_barostat(barostat_block);
        let path = write_config(&dir, &body);
        let err = load_config(&path).unwrap_err();
        assert_parse_path_and_field(&err, "barostat", unknown_field);
    }
}

// rq-c1d79d33
#[test]
fn barostat_c_rescale_accepted() {
    let dir = tmp_path("baro_c_rescale_ok");
    let body = config_with_barostat(
        r#"[barostat]
kind = "c-rescale"
pressure = 1.0e5
temperature = 85.0
tau = 1.0e-12
compressibility = 4.5e-10
seed = 42"#,
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    let b = cfg.phases[0].as_md().unwrap().barostat.as_ref().unwrap();
    assert_eq!(b.kind, "c-rescale");
    assert_eq!(param_f64(b, "pressure"), 1.0e5);
    assert_eq!(param_f64(b, "temperature"), 85.0);
    assert_eq!(param_f64(b, "tau"), 1.0e-12);
    assert_eq!(param_f64(b, "compressibility"), 4.5e-10);
    assert_eq!(param_u64(b, "seed"), 42);
}

// rq-8904d7cb
#[test]
fn barostat_c_rescale_accepts_negative_pressure() {
    let dir = tmp_path("baro_c_rescale_neg_p");
    let body = config_with_barostat(
        r#"[barostat]
kind = "c-rescale"
pressure = -1.0e5
temperature = 85.0
tau = 1.0e-12
compressibility = 4.5e-10
seed = 1"#,
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert!(cfg.phases[0].as_md().unwrap().barostat.is_some());
}

// rq-d406cdc2
#[test]
fn barostat_c_rescale_rejects_non_positive_temperature() {
    let dir = tmp_path("baro_c_rescale_T_zero");
    let body = config_with_barostat(
        r#"[barostat]
kind = "c-rescale"
pressure = 1.0e5
temperature = 0.0
tau = 1.0e-12
compressibility = 4.5e-10
seed = 1"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => {
            assert_eq!(field, "barostat.temperature")
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-a3b6838f
#[test]
fn barostat_c_rescale_rejects_non_positive_tau() {
    let dir = tmp_path("baro_c_rescale_tau_neg");
    let body = config_with_barostat(
        r#"[barostat]
kind = "c-rescale"
pressure = 1.0e5
temperature = 85.0
tau = -1.0e-12
compressibility = 4.5e-10
seed = 1"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "barostat.tau"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-de8b9cd5
#[test]
fn barostat_c_rescale_rejects_non_positive_compressibility() {
    let dir = tmp_path("baro_c_rescale_beta_zero");
    let body = config_with_barostat(
        r#"[barostat]
kind = "c-rescale"
pressure = 1.0e5
temperature = 85.0
tau = 1.0e-12
compressibility = 0.0
seed = 1"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => {
            assert_eq!(field, "barostat.compressibility")
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-d623f23d rq-0a18c7c2
#[test]
fn barostat_c_rescale_missing_temperature_rejected() {
    let dir = tmp_path("baro_c_rescale_no_T");
    let body = config_with_barostat(
        r#"[barostat]
kind = "c-rescale"
pressure = 1.0e5
tau = 1.0e-12
compressibility = 4.5e-10
seed = 1"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => {
            assert_eq!(field, "barostat.temperature")
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-b33d1ff0 rq-0b5f0881
#[test]
fn barostat_c_rescale_missing_seed_rejected() {
    let dir = tmp_path("baro_c_rescale_no_seed");
    let body = config_with_barostat(
        r#"[barostat]
kind = "c-rescale"
pressure = 1.0e5
temperature = 85.0
tau = 1.0e-12
compressibility = 4.5e-10"#,
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "barostat.seed"),
        other => panic!("unexpected: {other:?}"),
    }
}

// c-rescale extra-fields coverage lives in the parameterised
// `barostat_rejects_unknown_field_for_chosen_kind` test above.

// rq-bda9c0a2
#[test]
fn barostat_missing_kind_rejected() {
    let dir = tmp_path("baro_no_kind");
    let body = format!(
        r#"schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 12345
temperature = 300.0

[[phase]]
name = "run"
n_steps = 10
dt = 1.0e-15


[phase.integrator]
kind = "velocity-verlet"
lossless = false

[phase.barostat]
placeholder = true

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
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::MissingField { field } => assert_eq!(field, "barostat.kind"),
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-9a80c43c — [[constraint_types]] schema + IntegratorKind::supports_constraints tests.

#[test]
fn load_constraint_types_shake() {
    let dir = tmp_path("constraint_types_shake");
    let body = format!(
        "{}\n[[constraint_types]]\nname = \"SPCE\"\nkind = \"shake\"\natoms = 3\nconstraints = [\n  {{ i = 0, j = 1, d = 1.0e-10 }},\n  {{ i = 0, j = 2, d = 1.0e-10 }},\n  {{ i = 1, j = 2, d = 1.633e-10 }},\n]\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.constraint_types.len(), 1);
    let ct = &cfg.constraint_types[0];
    assert_eq!(ct.name, "SPCE");
    assert_eq!(ct.kind, "shake");
    assert_eq!(ct.params.get("atoms").unwrap().as_integer().unwrap(), 3);
    let constraints = ct.params.get("constraints").unwrap().as_array().unwrap();
    assert_eq!(constraints.len(), 3);
    // The shake builder reports the per-row expected atom count from
    // its `atoms` parameter.
    let registries = Registries::with_builtins();
    let builder = registries.constraint_types.lookup("shake").unwrap();
    assert_eq!(builder.expected_atom_count(&ct.params), 3);
}

// rq-9a80c43c
#[test]
fn reject_shake_params_malformed() {
    let dir = tmp_path("constraint_infeasible");
    // Self-loop constraint (i == j): rejected by validate_shake_params.
    let body = format!(
        "{}\n[[constraint_types]]\nname = \"BAD\"\nkind = \"shake\"\natoms = 3\nconstraints = [\n  {{ i = 0, j = 0, d = 1.0e-10 }},\n]\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::ShakeParamsMalformed { name, .. } => {
            assert_eq!(name, "BAD");
        }
        other => panic!("expected ShakeParamsMalformed, got {other:?}"),
    }
}

#[test]
fn reject_duplicate_constraint_type_name() {
    let dir = tmp_path("constraint_dup_name");
    let body = format!(
        "{}\n[[constraint_types]]\nname = \"X\"\nkind = \"shake\"\natoms = 2\nconstraints = [\n  {{ i = 0, j = 1, d = 1.0e-10 }},\n]\n\n[[constraint_types]]\nname = \"X\"\nkind = \"shake\"\natoms = 2\nconstraints = [\n  {{ i = 0, j = 1, d = 1.0e-10 }},\n]\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match load_config(&path).unwrap_err() {
        ConfigError::DuplicateConstraintTypeName { name } => assert_eq!(name, "X"),
        other => panic!("expected DuplicateConstraintTypeName, got {other:?}"),
    }
}

fn supports_constraints_for(kind: &SlotConfig) -> bool {
    let registry = IntegratorRegistry::with_builtins();
    registry
        .lookup(&kind.kind)
        .unwrap()
        .supports_constraints(&kind.params)
}

// rq-fd07b4dc
#[test]
fn supports_constraints_velocity_verlet_lossy_true() {
    let k = SlotConfig::from_params_str("velocity-verlet", "lossless = false\n");
    assert!(supports_constraints_for(&k));
}

// rq-53237ec4 rq-66ec7ee4
#[test]
fn supports_constraints_velocity_verlet_lossless_false() {
    let k = SlotConfig::from_params_str("velocity-verlet", "lossless = true\n");
    assert!(!supports_constraints_for(&k));
}

// rq-047c1f4d
#[test]
fn supports_constraints_langevin_baoab_false() {
    let k = SlotConfig::from_params_str(
        "langevin-baoab",
        "friction = 1.0e12\ntemperature = 300.0\nseed = 0\n",
    );
    assert!(!supports_constraints_for(&k));
}

// rq-09a19014
#[test]
fn supports_constraints_mtk_npt_false() {
    let k = SlotConfig::from_params_str(
        "mtk-npt",
        "temperature = 85.0\npressure = 1.0e5\ntau_t = 1.0e-13\ntau_p = 1.0e-12\nchain_length = 3\nyoshida_order = 3\nn_resp = 1\n",
    );
    assert!(!supports_constraints_for(&k));
}

// rq-064c9df1
#[test]
fn validate_constraint_compatibility_rejects_langevin_with_constraints() {
    let dir = tmp_path("compat_langevin");
    let body = format!(
        r#"schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 12345
temperature = 300.0

[[phase]]
name = "run"
n_steps = 10
dt = 1.0e-15


[phase.integrator]
kind = "langevin-baoab"
friction = 1.0e12
temperature = 300.0
seed = 1

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
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    let registries = Registries::with_builtins();
    match cfg.validate_constraint_compatibility(&registries, true).unwrap_err() {
        ConfigError::IncompatibleConstraint { integrator, phase: _ } => {
            assert_eq!(integrator, "langevin-baoab");
        }
        other => panic!("expected IncompatibleConstraint, got {other:?}"),
    }
}

// rq-8a3c0426
#[test]
fn validate_constraint_compatibility_accepts_velocity_verlet_lossy() {
    let dir = tmp_path("compat_vv_lossy");
    let body = minimal_config();
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    let registries = Registries::with_builtins();
    assert!(cfg.validate_constraint_compatibility(&registries, true).is_ok());
}

// rq-d370907d rq-58476106
#[test]
fn validate_constraint_compatibility_rejects_lossless_with_constraints() {
    let dir = tmp_path("compat_vv_lossless");
    let body = format!(
        r#"schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 12345
temperature = 300.0

[[phase]]
name = "run"
n_steps = 10
dt = 1.0e-15


[phase.integrator]
kind = "velocity-verlet"
lossless = true

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
    );
    let path = write_config(&dir, &body);
    let cfg = load_config(&path).unwrap();
    let registries = Registries::with_builtins();
    match cfg.validate_constraint_compatibility(&registries, true).unwrap_err() {
        ConfigError::IncompatibleConstraint { integrator, phase: _ } => {
            assert_eq!(integrator, "velocity-verlet");
        }
        other => panic!("expected IncompatibleConstraint, got {other:?}"),
    }
}

// rq-43819abc rq-1514bec6 — config filename must end in `.in.toml`.
#[test]
fn reject_config_without_in_toml_suffix() {
    let dir = tmp_path("reject_no_in_suffix");
    let path = write_config_named(&dir, "sim.toml", &minimal_config());
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidConfigFilename { path: p } => {
            assert_eq!(p, path);
        }
        other => panic!("expected InvalidConfigFilename, got {other:?}"),
    }
}

// rq-1514bec6 — the `.in.toml` suffix check is case-sensitive.
#[test]
fn reject_uppercase_in_toml_suffix() {
    let dir = tmp_path("reject_uppercase_in");
    let path = write_config_named(&dir, "sim.IN.toml", &minimal_config());
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidConfigFilename { path: p } => {
            assert_eq!(p, path);
        }
        other => panic!("expected InvalidConfigFilename, got {other:?}"),
    }
}

// rq-032b4b79 — a filename whose `<config-root>` derivation is empty is rejected.
#[test]
fn reject_empty_root_filename() {
    let dir = tmp_path("reject_empty_root");
    let path = write_config_named(&dir, ".in.toml", &minimal_config());
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidConfigFilename { path: p } => {
            assert_eq!(p, path);
        }
        other => panic!("expected InvalidConfigFilename, got {other:?}"),
    }
}

// rq-9ecf5a3a — filename-convention check fires before the file is opened
// (so a non-existent path that fails the convention still surfaces
// `InvalidConfigFilename`, not `Io`).
#[test]
fn filename_check_runs_before_io() {
    let dir = tmp_path("filename_pre_io");
    // No file is written to this path at all.
    let path = dir.join("does-not-exist.toml");
    match load_config(&path).unwrap_err() {
        ConfigError::InvalidConfigFilename { path: p } => {
            assert_eq!(p, path);
        }
        other => panic!("expected InvalidConfigFilename, got {other:?}"),
    }
}

// =====================================================================
// Unit-system selector (`units = "si" | "atomic"`)
// =====================================================================

use dynamics::io::PairPotentialParams;
use dynamics::units::{Dimension, UnitSystem};

// Build a minimal SI-mode config (the test helper `minimal_config()`
// is atomic-mode for round-trip ergonomics; these tests need SI to
// verify the conversion direction).
fn minimal_si_config() -> String {
    minimal_config().replace("units = \"atomic\"\n", "")
}

#[test]
fn units_default_is_si() {
    let dir = tmp_path("units_default_si");
    let path = write_config(&dir, &minimal_si_config());
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.units, UnitSystem::Si);
    // Loader converts SI input to atomic units on Config.
    let expected_temp = 300.0 / UnitSystem::Si.factor(Dimension::Temperature);
    let expected_dt = 1.0e-15 / UnitSystem::Si.factor(Dimension::Time);
    let expected_mass = 6.6335e-26 / UnitSystem::Si.factor(Dimension::Mass);
    assert!((cfg.simulation.temperature - expected_temp).abs() < 1e-12 * expected_temp);
    assert!((cfg.phases[0].as_md().unwrap().dt - expected_dt).abs() < 1e-12 * expected_dt);
    assert!((cfg.particle_types[0].mass - expected_mass).abs() < 1e-12 * expected_mass);
}

#[test]
fn units_explicit_si_accepted() {
    let dir = tmp_path("units_explicit_si");
    let cfg_text = format!("units = \"si\"\n{}", minimal_si_config());
    let path = write_config(&dir, &cfg_text);
    let cfg = load_config(&path).unwrap();
    assert_eq!(cfg.units, UnitSystem::Si);
}

// rq-4b760f1c
#[test]
fn units_unknown_value_rejected() {
    let dir = tmp_path("units_unknown");
    let cfg_text = format!("units = \"imperial\"\n{}", minimal_si_config());
    let path = write_config(&dir, &cfg_text);
    match load_config(&path).unwrap_err() {
        ConfigError::UnknownUnits { got } => assert_eq!(got, "imperial"),
        other => panic!("expected UnknownUnits, got {other:?}"),
    }
}

#[test]
fn atomic_units_yield_same_si_config_as_native_si() {
    // Two TOML files describing the same physical system: one in SI,
    // one in atomic units. After `load_config` the two `Config`
    // structs should be numerically equal (to f64 round-off) on every
    // unit-bearing field. Conversion factors come from the units
    // module, so this test pins the SI side and lets the atomic side
    // be derived through the conversion path.
    // `factor()` returns the user-per-atomic factor: in SI mode it is
    // the SI value of one atomic unit, in atomic mode it is 1.0. Use
    // the SI mode factors to convert physical SI values to their
    // atomic-unit counterparts for the atomic-side TOML file.
    let len_f = UnitSystem::Si.factor(Dimension::Length);
    let mass_f = UnitSystem::Si.factor(Dimension::Mass);
    let energy_f = UnitSystem::Si.factor(Dimension::Energy);
    let time_f = UnitSystem::Si.factor(Dimension::Time);
    let temp_f = UnitSystem::Si.factor(Dimension::Temperature);

    // Argon Lennard-Jones; values picked so the SI-side numbers are
    // physical and the atomic-side numbers stay finite and well-scaled.
    let sigma_si = 3.40e-10;
    let epsilon_si = 1.65e-21;
    let cutoff_si = 1.0e-9;
    let mass_si = 6.6335e-26;
    let dt_si = 1.0e-15;
    let temperature_si = 300.0;

    let sigma_au = sigma_si / len_f;
    let epsilon_au = epsilon_si / energy_f;
    let cutoff_au = cutoff_si / len_f;
    let mass_au = mass_si / mass_f;
    let dt_au = dt_si / time_f;
    let temperature_au = temperature_si / temp_f;

    let cfg_si = format!(
        r#"schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 12345
temperature = {temperature_si}

[[phase]]
name = "run"
n_steps = 10
dt = {dt_si:.16e}

[phase.integrator]
kind = "velocity-verlet"

[[particle_types]]
name = "Ar"
mass = {mass_si:.16e}

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = {sigma_si:.16e}
epsilon = {epsilon_si:.16e}
cutoff = {cutoff_si:.16e}
"#
    );

    let cfg_au = format!(
        r#"schema_version = 1
units = "atomic"
init = "argon.in.xyz"

[simulation]
seed = 12345
temperature = {temperature_au:.16e}

[[phase]]
name = "run"
n_steps = 10
dt = {dt_au:.16e}

[phase.integrator]
kind = "velocity-verlet"

[[particle_types]]
name = "Ar"
mass = {mass_au:.16e}

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = {sigma_au:.16e}
epsilon = {epsilon_au:.16e}
cutoff = {cutoff_au:.16e}
"#
    );

    let dir_si = tmp_path("atomic_vs_si_si_side");
    let dir_au = tmp_path("atomic_vs_si_au_side");
    let path_si = write_config(&dir_si, &cfg_si);
    let path_au = write_config(&dir_au, &cfg_au);

    let loaded_si = load_config(&path_si).unwrap();
    let loaded_au = load_config(&path_au).unwrap();

    assert_eq!(loaded_si.units, UnitSystem::Si);
    assert_eq!(loaded_au.units, UnitSystem::Atomic);

    // After conversion, both configs hold atomic-unit values.
    fn approx_eq(a: f64, b: f64, rel: f64) -> bool {
        if a == b {
            return true;
        }
        let scale = a.abs().max(b.abs());
        (a - b).abs() <= rel * scale
    }

    let rel = 1e-12;
    assert!(approx_eq(
        loaded_au.simulation.temperature,
        loaded_si.simulation.temperature,
        rel
    ));
    let md_si = loaded_si.phases[0].as_md().unwrap();
    let md_au = loaded_au.phases[0].as_md().unwrap();
    assert!(approx_eq(md_au.dt, md_si.dt, rel));
    assert!(approx_eq(
        loaded_au.particle_types[0].mass,
        loaded_si.particle_types[0].mass,
        rel
    ));

    let pi_si = &loaded_si.pair_interactions[0];
    let pi_au = &loaded_au.pair_interactions[0];
    assert!(approx_eq(pi_au.cutoff, pi_si.cutoff, rel));
    assert!(approx_eq(pi_au.r_switch, pi_si.r_switch, rel));
    match (&pi_si.potential, &pi_au.potential) {
        (
            PairPotentialParams::LennardJones {
                sigma: s_si,
                epsilon: e_si,
            },
            PairPotentialParams::LennardJones {
                sigma: s_au,
                epsilon: e_au,
            },
        ) => {
            assert!(approx_eq(*s_au, *s_si, rel));
            assert!(approx_eq(*e_au, *e_si, rel));
        }
    }
}

// rq-a17033aa
#[test]
fn atomic_units_rescale_csvr_thermostat_params() {
    // Verify that slot params (open-shaped toml::Value) get rescaled by
    // walking a CSVR thermostat block.
    let dir = tmp_path("atomic_csvr_params");
    let temp_au = 9.5e-4; // ~300 K in Hartree/k_B
    let tau_au = 41.34;   // ~1 fs in atomic time units
    let cfg = format!(
        r#"schema_version = 1
units = "atomic"
init = "argon.in.xyz"

[simulation]
seed = 12345
temperature = {temp_au}

[[phase]]
name = "run"
n_steps = 10
dt = 41.34

[phase.integrator]
kind = "velocity-verlet"

[phase.thermostat]
kind = "csvr"
temperature = {temp_au}
tau = {tau_au}
seed = 7

[[particle_types]]
name = "Ar"
mass = 72820.0

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 6.43
epsilon = 3.78e-4
cutoff = 18.9
"#
    );
    let path = write_config(&dir, &cfg);
    let loaded = load_config(&path).unwrap();

    let thermostat = loaded.phases[0]
        .as_md()
        .unwrap()
        .thermostat
        .as_ref()
        .unwrap();
    assert_eq!(thermostat.kind, "csvr");
    let t = param_f64(thermostat, "temperature");
    let tau = param_f64(thermostat, "tau");
    let seed = param_u64(thermostat, "seed");
    // Atomic-mode input passes through; the Config stores the same
    // numerical values supplied in the TOML.
    assert_eq!(t, temp_au);
    assert_eq!(tau, tau_au);
    assert_eq!(seed, 7);
}
