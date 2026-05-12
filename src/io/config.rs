// rq-6432ab1f rq-110285ae rq-b719c42c
use std::path::{Path, PathBuf};

// rq-f0084057
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathRole {
    Init,
    Trajectory,
    Log,
    Timings,
    Bonds,
}

impl std::fmt::Display for PathRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PathRole::Init => write!(f, "init"),
            PathRole::Trajectory => write!(f, "trajectory"),
            PathRole::Log => write!(f, "log"),
            PathRole::Timings => write!(f, "timings"),
            PathRole::Bonds => write!(f, "bonds"),
        }
    }
}

// rq-0b9372e8
#[derive(Debug)]
pub enum ConfigError {
    Io(String),
    Parse(String),
    MissingField {
        field: String,
    },
    UnsupportedSchemaVersion {
        actual: u64,
        supported: u64,
    },
    InvalidValue {
        field: String,
        reason: String,
    },
    DuplicateTypeName {
        name: String,
    },
    UnknownTypeInPair {
        name: String,
        pair_index: usize,
    },
    MissingPairInteraction {
        types: (String, String),
    },
    DuplicatePairInteraction {
        types: (String, String),
    },
    PathCollision {
        kind_a: PathRole,
        kind_b: PathRole,
        path: PathBuf,
    },
    MultiTypeUnsupported {
        count: usize,
    },
    UnknownIntegratorKind {
        actual: String,
    },
    UnknownIntegratorField {
        kind: String,
        field: String,
    },
    UnknownBondPotential {
        actual: String,
        bond_type_index: usize,
    },
    UnknownBondTypeField {
        potential: String,
        field: String,
        bond_type_index: usize,
    },
    DuplicateBondTypeName {
        name: String,
    },
    UnknownNeighborListMode {
        actual: String,
    },
    UnknownNeighborListField {
        mode: String,
        field: String,
    },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io(s) => write!(f, "Io({s})"),
            ConfigError::Parse(s) => write!(f, "Parse({s})"),
            ConfigError::MissingField { field } => {
                write!(f, "MissingField {{ field: {field:?} }}")
            }
            ConfigError::UnsupportedSchemaVersion { actual, supported } => write!(
                f,
                "UnsupportedSchemaVersion {{ actual: {actual}, supported: {supported} }}"
            ),
            ConfigError::InvalidValue { field, reason } => {
                write!(f, "InvalidValue {{ field: {field:?}, reason: {reason:?} }}")
            }
            ConfigError::DuplicateTypeName { name } => {
                write!(f, "DuplicateTypeName {{ name: {name:?} }}")
            }
            ConfigError::UnknownTypeInPair { name, pair_index } => write!(
                f,
                "UnknownTypeInPair {{ name: {name:?}, pair_index: {pair_index} }}"
            ),
            ConfigError::MissingPairInteraction { types } => write!(
                f,
                "MissingPairInteraction {{ types: ({:?}, {:?}) }}",
                types.0, types.1
            ),
            ConfigError::DuplicatePairInteraction { types } => write!(
                f,
                "DuplicatePairInteraction {{ types: ({:?}, {:?}) }}",
                types.0, types.1
            ),
            ConfigError::PathCollision {
                kind_a,
                kind_b,
                path,
            } => write!(
                f,
                "PathCollision {{ kind_a: {kind_a}, kind_b: {kind_b}, path: {} }}",
                path.display()
            ),
            ConfigError::MultiTypeUnsupported { count } => {
                write!(f, "MultiTypeUnsupported {{ count: {count} }}")
            }
            ConfigError::UnknownIntegratorKind { actual } => {
                write!(f, "UnknownIntegratorKind {{ actual: {actual:?} }}")
            }
            ConfigError::UnknownIntegratorField { kind, field } => write!(
                f,
                "UnknownIntegratorField {{ kind: {kind:?}, field: {field:?} }}"
            ),
            ConfigError::UnknownBondPotential { actual, bond_type_index } => write!(
                f,
                "UnknownBondPotential {{ actual: {actual:?}, bond_type_index: {bond_type_index} }}"
            ),
            ConfigError::UnknownBondTypeField {
                potential,
                field,
                bond_type_index,
            } => write!(
                f,
                "UnknownBondTypeField {{ potential: {potential:?}, field: {field:?}, bond_type_index: {bond_type_index} }}"
            ),
            ConfigError::DuplicateBondTypeName { name } => {
                write!(f, "DuplicateBondTypeName {{ name: {name:?} }}")
            }
            ConfigError::UnknownNeighborListMode { actual } => {
                write!(f, "UnknownNeighborListMode {{ actual: {actual:?} }}")
            }
            ConfigError::UnknownNeighborListField { mode, field } => write!(
                f,
                "UnknownNeighborListField {{ mode: {mode:?}, field: {field:?} }}"
            ),
        }
    }
}

impl std::error::Error for ConfigError {}

// rq-53055a5b
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub seed: u64,
    pub n_steps: u64,
    pub dt: f64,
    pub temperature: f64,
}

// rq-661bf664 rq-686b0d37
#[derive(Debug, Clone)]
pub enum IntegratorKind {
    VelocityVerlet {
        lossless: bool,
    },
    LangevinBaoab {
        friction: f64,
        temperature: f64,
        seed: u64,
    },
}

// rq-a5ccc1de
#[derive(Debug, Clone)]
pub struct ParticleTypeConfig {
    pub name: String,
    pub mass: f64,
}

// rq-f001eaf8
#[derive(Debug, Clone)]
pub struct PairInteractionConfig {
    pub between: (String, String),
    pub potential: String,
    pub sigma: f64,
    pub epsilon: f64,
    pub cutoff: f64,
}

#[derive(Debug, Clone)]
pub enum BondTypeConfig {
    Morse {
        name: String,
        de: f64,
        a: f64,
        re: f64,
    },
}

impl BondTypeConfig {
    pub fn name(&self) -> &str {
        match self {
            BondTypeConfig::Morse { name, .. } => name,
        }
    }
}

// rq-060b1fab
#[derive(Debug, Clone, PartialEq)]
pub enum NeighborListConfig {
    AllPairs,
    CellList { max_neighbors: u32, r_skin: f64 },
}

// rq-1254cd3a
#[derive(Debug, Clone)]
pub struct OutputConfig {
    pub trajectory_path: PathBuf,
    pub trajectory_every: u64,
    pub include_velocities: bool,
    pub log_path: PathBuf,
    pub log_every: u64,
    pub timings_path: PathBuf,
}

// rq-2a6a51c8
#[derive(Debug, Clone)]
pub struct Config {
    pub schema_version: u64,
    pub init: PathBuf,
    pub bonds: Option<PathBuf>,
    pub simulation: SimulationConfig,
    pub integrator: IntegratorKind,
    pub particle_types: Vec<ParticleTypeConfig>,
    pub pair_interactions: Vec<PairInteractionConfig>,
    pub bond_types: Vec<BondTypeConfig>,
    pub neighbor_list: NeighborListConfig,
    pub output: OutputConfig,
    pub config_path: PathBuf,
}

const SUPPORTED_SCHEMA_VERSION: u64 = 1;

fn missing(field: &str) -> ConfigError {
    ConfigError::MissingField {
        field: field.to_string(),
    }
}

fn invalid(field: &str, reason: impl Into<String>) -> ConfigError {
    ConfigError::InvalidValue {
        field: field.to_string(),
        reason: reason.into(),
    }
}

fn get_table<'a>(
    parent: &'a toml::value::Table,
    field: &str,
) -> Result<&'a toml::value::Table, ConfigError> {
    match parent.get(field) {
        Some(toml::Value::Table(t)) => Ok(t),
        Some(_) => Err(invalid(field, "expected a TOML table")),
        None => Err(missing(field)),
    }
}

fn get_u64(parent: &toml::value::Table, field: &str) -> Result<u64, ConfigError> {
    match parent.get(field) {
        Some(toml::Value::Integer(i)) if *i >= 0 => Ok(*i as u64),
        Some(toml::Value::Integer(_)) => Err(invalid(field, "expected a non-negative integer")),
        Some(_) => Err(invalid(field, "expected an integer")),
        None => Err(missing(field)),
    }
}

fn get_f64(parent: &toml::value::Table, field: &str) -> Result<f64, ConfigError> {
    match parent.get(field) {
        Some(toml::Value::Float(v)) => Ok(*v),
        Some(toml::Value::Integer(i)) => Ok(*i as f64),
        Some(_) => Err(invalid(field, "expected a float")),
        None => Err(missing(field)),
    }
}

fn get_str<'a>(parent: &'a toml::value::Table, field: &str) -> Result<&'a str, ConfigError> {
    match parent.get(field) {
        Some(toml::Value::String(s)) => Ok(s),
        Some(_) => Err(invalid(field, "expected a string")),
        None => Err(missing(field)),
    }
}

fn require_finite_positive(field: &str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() {
        return Err(invalid(field, format!("expected a finite number, got {value}")));
    }
    if value <= 0.0 {
        return Err(invalid(field, format!("expected a strictly positive value, got {value}")));
    }
    Ok(())
}

fn require_finite_non_negative(field: &str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() {
        return Err(invalid(field, format!("expected a finite number, got {value}")));
    }
    if value < 0.0 {
        return Err(invalid(field, format!("expected a non-negative value, got {value}")));
    }
    Ok(())
}

fn resolve_path(base_dir: &Path, raw: &str) -> PathBuf {
    let p = Path::new(raw);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base_dir.join(p)
    }
}

fn normalise_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

// rq-e8259ee5 rq-39881bb0
pub fn load_config(path: &Path) -> Result<Config, ConfigError> {
    let raw = std::fs::read_to_string(path).map_err(|e| ConfigError::Io(format!("{}: {}", path.display(), e)))?;
    let value: toml::Value = raw.parse().map_err(|e: toml::de::Error| ConfigError::Parse(e.to_string()))?;

    let root = value
        .as_table()
        .ok_or_else(|| ConfigError::Parse("top-level TOML must be a table".to_string()))?;

    // rq-fc58e2c5
    let schema_version = get_u64(root, "schema_version")?;
    if schema_version != SUPPORTED_SCHEMA_VERSION {
        return Err(ConfigError::UnsupportedSchemaVersion {
            actual: schema_version,
            supported: SUPPORTED_SCHEMA_VERSION,
        });
    }

    let init_raw = get_str(root, "init")?.to_string();
    let bonds_raw: Option<String> = match root.get("bonds") {
        Some(toml::Value::String(s)) => Some(s.clone()),
        Some(_) => return Err(invalid("bonds", "expected a string")),
        None => None,
    };

    let sim_tbl = get_table(root, "simulation")?;
    let seed = get_u64(sim_tbl, "seed").map_err(rename_field("simulation.seed".into()))?;
    let n_steps = get_u64(sim_tbl, "n_steps").map_err(rename_field("simulation.n_steps".into()))?;
    let dt_raw = sim_tbl
        .get("dt")
        .ok_or_else(|| missing("simulation.dt"))?;
    let dt = match dt_raw {
        toml::Value::Float(v) => *v,
        toml::Value::Integer(i) => *i as f64,
        _ => return Err(invalid("simulation.dt", "expected a float")),
    };
    require_finite_positive("simulation.dt", dt)?;
    let temperature_raw = sim_tbl
        .get("temperature")
        .ok_or_else(|| missing("simulation.temperature"))?;
    let temperature = match temperature_raw {
        toml::Value::Float(v) => *v,
        toml::Value::Integer(i) => *i as f64,
        _ => return Err(invalid("simulation.temperature", "expected a float")),
    };
    require_finite_non_negative("simulation.temperature", temperature)?;
    let simulation = SimulationConfig {
        seed,
        n_steps,
        dt,
        temperature,
    };

    let integ_tbl = get_table(root, "integrator")?;
    let kind_str = get_str(integ_tbl, "kind")
        .map_err(rename_field("integrator.kind".into()))?
        .to_string();
    let integrator = match kind_str.as_str() {
        "velocity-verlet" => {
            for key in integ_tbl.keys() {
                if !matches!(key.as_str(), "kind" | "lossless") {
                    return Err(ConfigError::UnknownIntegratorField {
                        kind: "velocity-verlet".to_string(),
                        field: key.clone(),
                    });
                }
            }
            let lossless = match integ_tbl.get("lossless") {
                Some(toml::Value::Boolean(b)) => *b,
                Some(_) => return Err(invalid("integrator.lossless", "expected a boolean")),
                None => false,
            };
            IntegratorKind::VelocityVerlet { lossless }
        }
        "langevin-baoab" => {
            for key in integ_tbl.keys() {
                if !matches!(key.as_str(), "kind" | "friction" | "temperature" | "seed") {
                    return Err(ConfigError::UnknownIntegratorField {
                        kind: "langevin-baoab".to_string(),
                        field: key.clone(),
                    });
                }
            }
            let friction = get_f64(integ_tbl, "friction")
                .map_err(rename_field("integrator.friction".into()))?;
            require_finite_positive("integrator.friction", friction)?;
            let temperature = get_f64(integ_tbl, "temperature")
                .map_err(rename_field("integrator.temperature".into()))?;
            require_finite_positive("integrator.temperature", temperature)?;
            let seed = get_u64(integ_tbl, "seed")
                .map_err(rename_field("integrator.seed".into()))?;
            IntegratorKind::LangevinBaoab {
                friction,
                temperature,
                seed,
            }
        }
        other => {
            return Err(ConfigError::UnknownIntegratorKind {
                actual: other.to_string(),
            });
        }
    };

    let pt_array = root
        .get("particle_types")
        .ok_or_else(|| missing("particle_types"))?
        .as_array()
        .ok_or_else(|| invalid("particle_types", "expected an array of tables"))?;
    if pt_array.is_empty() {
        return Err(missing("particle_types"));
    }
    let mut particle_types: Vec<ParticleTypeConfig> = Vec::with_capacity(pt_array.len());
    let mut seen_names: Vec<String> = Vec::new();
    for (i, entry) in pt_array.iter().enumerate() {
        let tbl = entry
            .as_table()
            .ok_or_else(|| invalid(&format!("particle_types[{i}]"), "expected a table"))?;
        let name = get_str(tbl, "name")
            .map_err(rename_field(format!("particle_types[{i}].name")))?
            .to_string();
        if name.is_empty() {
            return Err(invalid(
                &format!("particle_types[{i}].name"),
                "name must not be empty",
            ));
        }
        let mass = get_f64(tbl, "mass")
            .map_err(rename_field(format!("particle_types[{i}].mass")))?;
        require_finite_positive(&format!("particle_types[{i}].mass"), mass)?;
        if seen_names.contains(&name) {
            return Err(ConfigError::DuplicateTypeName { name });
        }
        seen_names.push(name.clone());
        particle_types.push(ParticleTypeConfig { name, mass });
    }

    let pi_array = root
        .get("pair_interactions")
        .ok_or_else(|| ConfigError::MissingPairInteraction {
            types: (
                particle_types[0].name.clone(),
                particle_types[0].name.clone(),
            ),
        })?
        .as_array()
        .ok_or_else(|| invalid("pair_interactions", "expected an array of tables"))?;

    let mut pair_interactions: Vec<PairInteractionConfig> = Vec::with_capacity(pi_array.len());
    for (i, entry) in pi_array.iter().enumerate() {
        let tbl = entry
            .as_table()
            .ok_or_else(|| invalid(&format!("pair_interactions[{i}]"), "expected a table"))?;
        let between_raw = tbl
            .get("between")
            .ok_or_else(|| missing(&format!("pair_interactions[{i}].between")))?
            .as_array()
            .ok_or_else(|| {
                invalid(
                    &format!("pair_interactions[{i}].between"),
                    "expected a two-element string array",
                )
            })?;
        if between_raw.len() != 2 {
            return Err(invalid(
                &format!("pair_interactions[{i}].between"),
                "expected exactly two type names",
            ));
        }
        let a = between_raw[0].as_str().ok_or_else(|| {
            invalid(
                &format!("pair_interactions[{i}].between"),
                "expected string entries",
            )
        })?;
        let b = between_raw[1].as_str().ok_or_else(|| {
            invalid(
                &format!("pair_interactions[{i}].between"),
                "expected string entries",
            )
        })?;

        for name in [a, b] {
            if !particle_types.iter().any(|t| t.name == name) {
                return Err(ConfigError::UnknownTypeInPair {
                    name: name.to_string(),
                    pair_index: i,
                });
            }
        }
        let between = normalise_pair(a, b);

        let potential = get_str(tbl, "potential")
            .map_err(rename_field(format!("pair_interactions[{i}].potential")))?
            .to_string();
        if potential != "lennard-jones" {
            return Err(invalid(
                &format!("pair_interactions[{i}].potential"),
                format!("unsupported potential {potential:?}"),
            ));
        }
        let sigma = get_f64(tbl, "sigma")
            .map_err(rename_field(format!("pair_interactions[{i}].sigma")))?;
        require_finite_positive(&format!("pair_interactions[{i}].sigma"), sigma)?;
        let epsilon = get_f64(tbl, "epsilon")
            .map_err(rename_field(format!("pair_interactions[{i}].epsilon")))?;
        require_finite_positive(&format!("pair_interactions[{i}].epsilon"), epsilon)?;
        let cutoff = get_f64(tbl, "cutoff")
            .map_err(rename_field(format!("pair_interactions[{i}].cutoff")))?;
        require_finite_positive(&format!("pair_interactions[{i}].cutoff"), cutoff)?;

        pair_interactions.push(PairInteractionConfig {
            between,
            potential,
            sigma,
            epsilon,
            cutoff,
        });
    }

    // rq-bd228ef7
    // Duplicate-pair check
    for i in 0..pair_interactions.len() {
        for j in 0..i {
            if pair_interactions[i].between == pair_interactions[j].between {
                return Err(ConfigError::DuplicatePairInteraction {
                    types: pair_interactions[i].between.clone(),
                });
            }
        }
    }

    // Pair-coverage check: every unordered pair of declared types must appear.
    for i in 0..particle_types.len() {
        for j in i..particle_types.len() {
            let key = normalise_pair(&particle_types[i].name, &particle_types[j].name);
            if !pair_interactions.iter().any(|p| p.between == key) {
                return Err(ConfigError::MissingPairInteraction { types: key });
            }
        }
    }

    // Bond types (optional)
    let mut bond_types: Vec<BondTypeConfig> = Vec::new();
    if let Some(bt_value) = root.get("bond_types") {
        let bt_array = bt_value
            .as_array()
            .ok_or_else(|| invalid("bond_types", "expected an array of tables"))?;
        let mut seen_names: Vec<String> = Vec::new();
        for (i, entry) in bt_array.iter().enumerate() {
            let tbl = entry
                .as_table()
                .ok_or_else(|| invalid(&format!("bond_types[{i}]"), "expected a table"))?;
            let name = get_str(tbl, "name")
                .map_err(rename_field(format!("bond_types[{i}].name")))?
                .to_string();
            if name.is_empty() {
                return Err(invalid(
                    &format!("bond_types[{i}].name"),
                    "name must not be empty",
                ));
            }
            if seen_names.contains(&name) {
                return Err(ConfigError::DuplicateBondTypeName { name });
            }
            let potential = get_str(tbl, "potential")
                .map_err(rename_field(format!("bond_types[{i}].potential")))?
                .to_string();
            match potential.as_str() {
                "morse" => {
                    for key in tbl.keys() {
                        if !matches!(key.as_str(), "name" | "potential" | "de" | "a" | "re") {
                            return Err(ConfigError::UnknownBondTypeField {
                                potential: "morse".to_string(),
                                field: key.clone(),
                                bond_type_index: i,
                            });
                        }
                    }
                    let de = get_f64(tbl, "de")
                        .map_err(rename_field(format!("bond_types[{i}].de")))?;
                    require_finite_positive(&format!("bond_types[{i}].de"), de)?;
                    let a = get_f64(tbl, "a")
                        .map_err(rename_field(format!("bond_types[{i}].a")))?;
                    require_finite_positive(&format!("bond_types[{i}].a"), a)?;
                    let re = get_f64(tbl, "re")
                        .map_err(rename_field(format!("bond_types[{i}].re")))?;
                    require_finite_positive(&format!("bond_types[{i}].re"), re)?;
                    seen_names.push(name.clone());
                    bond_types.push(BondTypeConfig::Morse { name, de, a, re });
                }
                other => {
                    return Err(ConfigError::UnknownBondPotential {
                        actual: other.to_string(),
                        bond_type_index: i,
                    });
                }
            }
        }
    }

    // Neighbor list (optional table)
    // rq-060b1fab
    let max_cutoff: f64 = pair_interactions
        .iter()
        .map(|p| p.cutoff)
        .fold(0.0_f64, f64::max);
    let neighbor_list = match root.get("neighbor_list") {
        None => NeighborListConfig::CellList {
            max_neighbors: 256,
            r_skin: 0.3 * max_cutoff,
        },
        Some(toml::Value::Table(nl_tbl)) => {
            let mode = match nl_tbl.get("mode") {
                Some(toml::Value::String(s)) => s.clone(),
                Some(_) => return Err(invalid("neighbor_list.mode", "expected a string")),
                None => "cell-list".to_string(),
            };
            match mode.as_str() {
                "all-pairs" => {
                    for key in nl_tbl.keys() {
                        if key != "mode" {
                            return Err(ConfigError::UnknownNeighborListField {
                                mode: "all-pairs".to_string(),
                                field: key.clone(),
                            });
                        }
                    }
                    NeighborListConfig::AllPairs
                }
                "cell-list" => {
                    for key in nl_tbl.keys() {
                        if !matches!(key.as_str(), "mode" | "max_neighbors" | "r_skin") {
                            return Err(ConfigError::UnknownNeighborListField {
                                mode: "cell-list".to_string(),
                                field: key.clone(),
                            });
                        }
                    }
                    let max_neighbors = match nl_tbl.get("max_neighbors") {
                        Some(toml::Value::Integer(i)) if *i > 0 => *i as u32,
                        Some(toml::Value::Integer(_)) => {
                            return Err(invalid(
                                "neighbor_list.max_neighbors",
                                "expected a strictly positive integer",
                            ));
                        }
                        Some(_) => {
                            return Err(invalid(
                                "neighbor_list.max_neighbors",
                                "expected an integer",
                            ));
                        }
                        None => 256u32,
                    };
                    let r_skin = match nl_tbl.get("r_skin") {
                        Some(toml::Value::Float(v)) => *v,
                        Some(toml::Value::Integer(i)) => *i as f64,
                        Some(_) => {
                            return Err(invalid("neighbor_list.r_skin", "expected a float"));
                        }
                        None => 0.3 * max_cutoff,
                    };
                    require_finite_positive("neighbor_list.r_skin", r_skin)?;
                    NeighborListConfig::CellList {
                        max_neighbors,
                        r_skin,
                    }
                }
                other => {
                    return Err(ConfigError::UnknownNeighborListMode {
                        actual: other.to_string(),
                    });
                }
            }
        }
        Some(_) => return Err(invalid("neighbor_list", "expected a table")),
    };

    // Output section with defaults
    let config_path_canonical = std::fs::canonicalize(path)
        .map_err(|e| ConfigError::Io(format!("canonicalize {}: {}", path.display(), e)))?;
    let base_dir = config_path_canonical
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    let stem = config_path_canonical
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "sim".to_string());
    let default_traj = format!("{stem}-traj.xyz");
    let default_log = format!("{stem}.log");
    let default_timings = format!("{stem}.timings");

    let (
        trajectory_path,
        trajectory_every,
        include_velocities,
        log_path,
        log_every,
        timings_path,
    ) = match root.get("output") {
        Some(toml::Value::Table(out_tbl)) => {
            let tpath = match out_tbl.get("trajectory_path") {
                Some(toml::Value::String(s)) => resolve_path(&base_dir, s),
                Some(_) => return Err(invalid("output.trajectory_path", "expected a string")),
                None => resolve_path(&base_dir, &default_traj),
            };
            let tevery = match out_tbl.get("trajectory_every") {
                Some(toml::Value::Integer(i)) if *i >= 0 => *i as u64,
                Some(toml::Value::Integer(_)) => {
                    return Err(invalid(
                        "output.trajectory_every",
                        "expected a non-negative integer",
                    ));
                }
                Some(_) => return Err(invalid("output.trajectory_every", "expected an integer")),
                None => 100,
            };
            let incv = match out_tbl.get("include_velocities") {
                Some(toml::Value::Boolean(b)) => *b,
                Some(_) => return Err(invalid("output.include_velocities", "expected a boolean")),
                None => true,
            };
            let lpath = match out_tbl.get("log_path") {
                Some(toml::Value::String(s)) => resolve_path(&base_dir, s),
                Some(_) => return Err(invalid("output.log_path", "expected a string")),
                None => resolve_path(&base_dir, &default_log),
            };
            let levery = match out_tbl.get("log_every") {
                Some(toml::Value::Integer(i)) if *i >= 0 => *i as u64,
                Some(toml::Value::Integer(_)) => {
                    return Err(invalid("output.log_every", "expected a non-negative integer"));
                }
                Some(_) => return Err(invalid("output.log_every", "expected an integer")),
                None => 100,
            };
            let timings = match out_tbl.get("timings_path") {
                Some(toml::Value::String(s)) => resolve_path(&base_dir, s),
                Some(_) => return Err(invalid("output.timings_path", "expected a string")),
                None => resolve_path(&base_dir, &default_timings),
            };
            (tpath, tevery, incv, lpath, levery, timings)
        }
        Some(_) => return Err(invalid("output", "expected a table")),
        None => (
            resolve_path(&base_dir, &default_traj),
            100,
            true,
            resolve_path(&base_dir, &default_log),
            100,
            resolve_path(&base_dir, &default_timings),
        ),
    };

    // rq-6d99f9c8
    let init_path = resolve_path(&base_dir, &init_raw);
    let bonds_path: Option<PathBuf> = bonds_raw.as_deref().map(|s| resolve_path(&base_dir, s));

    // Path collision checks (init/traj/log/timings/bonds pairwise distinct)
    let check_collision = |kind_a: PathRole, path_a: &PathBuf, kind_b: PathRole, path_b: &PathBuf| {
        if path_a == path_b {
            Some(ConfigError::PathCollision {
                kind_a,
                kind_b,
                path: path_a.clone(),
            })
        } else {
            None
        }
    };
    if let Some(e) = check_collision(PathRole::Init, &init_path, PathRole::Trajectory, &trajectory_path) {
        return Err(e);
    }
    if let Some(e) = check_collision(PathRole::Trajectory, &trajectory_path, PathRole::Log, &log_path) {
        return Err(e);
    }
    if let Some(e) = check_collision(PathRole::Init, &init_path, PathRole::Log, &log_path) {
        return Err(e);
    }
    if let Some(e) = check_collision(PathRole::Init, &init_path, PathRole::Timings, &timings_path) {
        return Err(e);
    }
    if let Some(e) = check_collision(PathRole::Trajectory, &trajectory_path, PathRole::Timings, &timings_path) {
        return Err(e);
    }
    if let Some(e) = check_collision(PathRole::Log, &log_path, PathRole::Timings, &timings_path) {
        return Err(e);
    }
    if let Some(b) = bonds_path.as_ref() {
        if let Some(e) = check_collision(PathRole::Init, &init_path, PathRole::Bonds, b) {
            return Err(e);
        }
        if let Some(e) = check_collision(PathRole::Trajectory, &trajectory_path, PathRole::Bonds, b) {
            return Err(e);
        }
        if let Some(e) = check_collision(PathRole::Log, &log_path, PathRole::Bonds, b) {
            return Err(e);
        }
        if let Some(e) = check_collision(PathRole::Timings, &timings_path, PathRole::Bonds, b) {
            return Err(e);
        }
    }

    // Multi-type restriction
    if particle_types.len() != 1 {
        return Err(ConfigError::MultiTypeUnsupported {
            count: particle_types.len(),
        });
    }

    Ok(Config {
        schema_version,
        init: init_path,
        bonds: bonds_path,
        simulation,
        integrator,
        particle_types,
        pair_interactions,
        bond_types,
        neighbor_list,
        output: OutputConfig {
            trajectory_path,
            trajectory_every,
            include_velocities,
            log_path,
            log_every,
            timings_path,
        },
        config_path: config_path_canonical,
    })
}

// Helper: rewrite a nested `MissingField`/`InvalidValue` to use a fully-qualified field name.
fn rename_field(full: String) -> impl FnOnce(ConfigError) -> ConfigError {
    move |e| match e {
        ConfigError::MissingField { .. } => ConfigError::MissingField { field: full },
        ConfigError::InvalidValue { reason, .. } => ConfigError::InvalidValue {
            field: full,
            reason,
        },
        other => other,
    }
}
