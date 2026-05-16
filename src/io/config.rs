// rq-6432ab1f rq-110285ae rq-b719c42c
use std::path::{Path, PathBuf};

// rq-f0084057
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathRole {
    Init,
    Trajectory,
    Log,
    Timings,
    Topology,
}

impl std::fmt::Display for PathRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PathRole::Init => write!(f, "init"),
            PathRole::Trajectory => write!(f, "trajectory"),
            PathRole::Log => write!(f, "log"),
            PathRole::Timings => write!(f, "timings"),
            PathRole::Topology => write!(f, "topology"),
        }
    }
}

// rq-0b9372e8 rq-e1ceb5c0 rq-1bbcf3b7
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file: {0}")]
    Io(String),
    #[error("failed to parse config TOML: {0}")]
    Parse(String),
    #[error("missing required field `{field}`")]
    MissingField { field: String },
    #[error("unsupported schema version {actual}: only version {supported} is supported")]
    UnsupportedSchemaVersion { actual: u64, supported: u64 },
    #[error("invalid value for `{field}`: {reason}")]
    InvalidValue { field: String, reason: String },
    #[error("duplicate particle type name `{name}`")]
    DuplicateTypeName { name: String },
    #[error("pair_interactions[{pair_index}] references unknown particle type `{name}`")]
    UnknownTypeInPair { name: String, pair_index: usize },
    #[error("missing pair interaction for type pair (`{}`, `{}`)", types.0, types.1)]
    MissingPairInteraction { types: (String, String) },
    #[error("duplicate pair interaction for type pair (`{}`, `{}`)", types.0, types.1)]
    DuplicatePairInteraction { types: (String, String) },
    #[error("pair_interactions[{pair_index}] has unknown potential `{actual}`")]
    UnknownPairPotential { actual: String, pair_index: usize },
    #[error("pair_interactions[{pair_index}] has unknown field `{field}` for potential `{potential}`")]
    UnknownPairInteractionField {
        potential: String,
        field: String,
        pair_index: usize,
    },
    #[error("output paths collide: `{kind_a}` and `{kind_b}` both resolve to `{}`", path.display())]
    PathCollision {
        kind_a: PathRole,
        kind_b: PathRole,
        path: PathBuf,
    },
    #[error("config declares both [coulomb] and [spme]; only one electrostatics method may be active per run")]
    ConflictingElectrostatics,
    #[error("unknown integrator kind `{actual}`")]
    UnknownIntegratorKind { actual: String },
    #[error("unknown field `{field}` for integrator kind `{kind}`")]
    UnknownIntegratorField { kind: String, field: String },
    // rq-1c2d8eba rq-3fdb7e01
    #[error("unknown thermostat kind `{actual}`")]
    UnknownThermostatKind { actual: String },
    #[error("unknown field `{field}` for thermostat kind `{kind}`")]
    UnknownThermostatField { kind: String, field: String },
    // rq-d28e9105 rq-aa6ce5c0
    #[error("unknown barostat kind `{actual}`")]
    UnknownBarostatKind { actual: String },
    #[error("unknown field `{field}` for barostat kind `{kind}`")]
    UnknownBarostatField { kind: String, field: String },
    // Cross-validation rule 4: integrator owns its own thermostat and
    // therefore cannot be paired with a `[thermostat]` table.
    #[error("integrator `{integrator}` owns its own thermostat and is incompatible with `[thermostat]`")]
    IncompatibleThermostat { integrator: String },
    #[error("bond_types[{bond_type_index}] has unknown potential `{actual}`")]
    UnknownBondPotential {
        actual: String,
        bond_type_index: usize,
    },
    #[error("bond_types[{bond_type_index}] has unknown field `{field}` for potential `{potential}`")]
    UnknownBondTypeField {
        potential: String,
        field: String,
        bond_type_index: usize,
    },
    #[error("duplicate bond type name `{name}`")]
    DuplicateBondTypeName { name: String },
    #[error("angle_types[{angle_type_index}] has unknown potential `{actual}`")]
    UnknownAnglePotential {
        actual: String,
        angle_type_index: usize,
    },
    #[error("angle_types[{angle_type_index}] has unknown field `{field}` for potential `{potential}`")]
    UnknownAngleTypeField {
        potential: String,
        field: String,
        angle_type_index: usize,
    },
    #[error("duplicate angle type name `{name}`")]
    DuplicateAngleTypeName { name: String },
    #[error("unknown neighbor_list mode `{actual}`")]
    UnknownNeighborListMode { actual: String },
    #[error("unknown field `{field}` for neighbor_list mode `{mode}`")]
    UnknownNeighborListField { mode: String, field: String },
}

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

impl IntegratorKind {
    /// Lookup key used by `IntegratorRegistry` to dispatch this kind to its
    /// builder. Matches the `kind` field accepted by the config parser.
    pub fn name(&self) -> &'static str {
        match self {
            IntegratorKind::VelocityVerlet { .. } => "velocity-verlet",
            IntegratorKind::LangevinBaoab { .. } => "langevin-baoab",
        }
    }

    // Returns `true` for variants that bundle their own thermostat (the
    // OU step inside Langevin BAOAB); `false` otherwise. Consulted by
    // `load_config`'s cross-validation rule 4 to reject co-configured
    // `[thermostat]` tables. See `integration/framework.md`.
    pub fn owns_thermostat(&self) -> bool {
        matches!(self, IntegratorKind::LangevinBaoab { .. })
    }
}

// rq-3fdb7e01
#[derive(Debug, Clone)]
pub enum ThermostatKind {
    NoseHooverChain {
        temperature: f64,
        tau: f64,
        chain_length: u32,
        yoshida_order: u32,
        n_resp: u32,
    },
    Csvr {
        temperature: f64,
        tau: f64,
        seed: u64,
    },
    Andersen {
        temperature: f64,
        collision_rate: f64,
        seed: u64,
    },
    Berendsen {
        temperature: f64,
        tau: f64,
    },
}

impl ThermostatKind {
    /// Lookup key used by `ThermostatRegistry` to dispatch this kind.
    pub fn name(&self) -> &'static str {
        match self {
            ThermostatKind::NoseHooverChain { .. } => "nose-hoover-chain",
            ThermostatKind::Csvr { .. } => "csvr",
            ThermostatKind::Andersen { .. } => "andersen",
            ThermostatKind::Berendsen { .. } => "berendsen",
        }
    }
}

// rq-aa6ce5c0
#[derive(Debug, Clone)]
pub enum BarostatKind {
    // rq-3db027c2
    Berendsen {
        pressure: f64,
        tau: f64,
        compressibility: f64,
    },
    // rq-c2211c85
    CRescale {
        pressure: f64,
        temperature: f64,
        tau: f64,
        compressibility: f64,
        seed: u64,
    },
}

impl BarostatKind {
    /// Lookup key used by `BarostatRegistry` to dispatch this kind.
    pub fn name(&self) -> &'static str {
        match self {
            BarostatKind::Berendsen { .. } => "berendsen",
            BarostatKind::CRescale { .. } => "c-rescale",
        }
    }
}

// rq-a5ccc1de
#[derive(Debug, Clone)]
pub struct ParticleTypeConfig {
    pub name: String,
    pub mass: f64,
    pub charge: f64,
}

// rq-f001eaf8
#[derive(Debug, Clone)]
pub struct PairInteractionConfig {
    pub between: (String, String),
    pub cutoff: f64,
    pub r_switch: f64,
    pub potential: PairPotentialParams,
}

// rq-70442e07
#[derive(Debug, Clone)]
pub enum PairPotentialParams {
    LennardJones { sigma: f64, epsilon: f64 },
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

// AngleTypeConfig — variants for the [[angle_types]] array.
#[derive(Debug, Clone)]
pub enum AngleTypeConfig {
    Harmonic {
        name: String,
        k_theta: f64,
        theta_0: f64,
    },
}

impl AngleTypeConfig {
    pub fn name(&self) -> &str {
        match self {
            AngleTypeConfig::Harmonic { name, .. } => name,
        }
    }
}

// rq-060b1fab
#[derive(Debug, Clone, PartialEq)]
pub enum NeighborListConfig {
    AllPairs,
    CellList { max_neighbors: u32, r_skin: f64 },
}

// CoulombConfig — parsed `[coulomb]` table; rq-846bdb8b
#[derive(Debug, Clone, PartialEq)]
pub struct CoulombConfig {
    pub cutoff: f64,
    pub r_switch: f64,
}

// SpmeConfig — parsed `[spme]` table; rq-7bd2d9ca rq-202493a5
#[derive(Debug, Clone, PartialEq)]
pub struct SpmeConfig {
    pub alpha: f64,
    pub r_cut_real: f64,
    pub grid: [u32; 3],
    pub spline_order: u32,
}

// rq-1254cd3a
#[derive(Debug, Clone)]
pub struct OutputConfig {
    pub trajectory_path: PathBuf,
    pub trajectory_every: u64,
    pub include_velocities: bool,
    pub include_images: bool,
    pub log_path: PathBuf,
    pub log_every: u64,
    pub timings_path: PathBuf,
}

// rq-2a6a51c8
#[derive(Debug, Clone)]
pub struct Config {
    pub schema_version: u64,
    pub init: PathBuf,
    pub topology: Option<PathBuf>,
    pub simulation: SimulationConfig,
    pub integrator: IntegratorKind,
    pub thermostat: Option<ThermostatKind>,
    pub barostat: Option<BarostatKind>,
    pub particle_types: Vec<ParticleTypeConfig>,
    pub pair_interactions: Vec<PairInteractionConfig>,
    pub bond_types: Vec<BondTypeConfig>,
    pub angle_types: Vec<AngleTypeConfig>,
    pub coulomb: Option<CoulombConfig>,
    pub spme: Option<SpmeConfig>,
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
    let topology_raw: Option<String> = match root.get("topology") {
        Some(toml::Value::String(s)) => Some(s.clone()),
        Some(_) => return Err(invalid("topology", "expected a string")),
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

    // rq-1c2d8eba: optional [thermostat] section.
    let thermostat: Option<ThermostatKind> = match root.get("thermostat") {
        None => None,
        Some(toml::Value::Table(therm_tbl)) => Some(parse_thermostat(therm_tbl)?),
        Some(_) => return Err(invalid("thermostat", "expected a table")),
    };

    // rq-d28e9105: optional [barostat] section.
    let barostat: Option<BarostatKind> = match root.get("barostat") {
        None => None,
        Some(toml::Value::Table(baro_tbl)) => Some(parse_barostat(baro_tbl)?),
        Some(_) => return Err(invalid("barostat", "expected a table")),
    };

    // Cross-validation rule 4: integrator-owns-thermostat compatibility.
    if thermostat.is_some() && integrator.owns_thermostat() {
        return Err(ConfigError::IncompatibleThermostat {
            integrator: integrator.name().to_string(),
        });
    }

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
        // rq-78487f38: `charge` is optional and defaults to 0.0; when
        // supplied it must be finite (any sign is accepted).
        let charge = match tbl.get("charge") {
            Some(v) => {
                let f = match v {
                    toml::Value::Float(f) => *f,
                    toml::Value::Integer(i) => *i as f64,
                    _ => {
                        return Err(invalid(
                            &format!("particle_types[{i}].charge"),
                            "expected a float",
                        ));
                    }
                };
                if !f.is_finite() {
                    return Err(invalid(
                        &format!("particle_types[{i}].charge"),
                        format!("expected a finite number, got {f}"),
                    ));
                }
                f
            }
            None => 0.0,
        };
        if seen_names.contains(&name) {
            return Err(ConfigError::DuplicateTypeName { name });
        }
        seen_names.push(name.clone());
        particle_types.push(ParticleTypeConfig { name, mass, charge });
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

        // The recognised key set depends on the chosen potential: the
        // common keys are always allowed, and each potential adds its
        // own. Any other key is rejected, mirroring [[bond_types]].
        let recognised: &[&str] = match potential.as_str() {
            "lennard-jones" => {
                &["between", "potential", "cutoff", "r_switch", "sigma", "epsilon"]
            }
            other => {
                return Err(ConfigError::UnknownPairPotential {
                    actual: other.to_string(),
                    pair_index: i,
                });
            }
        };
        for key in tbl.keys() {
            if !recognised.contains(&key.as_str()) {
                return Err(ConfigError::UnknownPairInteractionField {
                    potential: potential.clone(),
                    field: key.clone(),
                    pair_index: i,
                });
            }
        }

        let cutoff = get_f64(tbl, "cutoff")
            .map_err(rename_field(format!("pair_interactions[{i}].cutoff")))?;
        require_finite_positive(&format!("pair_interactions[{i}].cutoff"), cutoff)?;

        // r_switch: optional, defaults to 0.9 * cutoff; must satisfy
        // 0 < r_switch <= cutoff. Setting r_switch == cutoff selects the
        // hard-cutoff degenerate case in the LJ kernel.
        let r_switch = match tbl.get("r_switch") {
            Some(toml::Value::Float(v)) => *v,
            Some(toml::Value::Integer(n)) => *n as f64,
            Some(_) => {
                return Err(invalid(
                    &format!("pair_interactions[{i}].r_switch"),
                    "expected a float",
                ));
            }
            None => 0.9 * cutoff,
        };
        require_finite_positive(&format!("pair_interactions[{i}].r_switch"), r_switch)?;
        if r_switch > cutoff {
            return Err(invalid(
                &format!("pair_interactions[{i}].r_switch"),
                format!("r_switch ({r_switch}) exceeds cutoff ({cutoff})"),
            ));
        }

        let potential_params = match potential.as_str() {
            "lennard-jones" => {
                let sigma = get_f64(tbl, "sigma")
                    .map_err(rename_field(format!("pair_interactions[{i}].sigma")))?;
                require_finite_positive(&format!("pair_interactions[{i}].sigma"), sigma)?;
                let epsilon = get_f64(tbl, "epsilon")
                    .map_err(rename_field(format!("pair_interactions[{i}].epsilon")))?;
                require_finite_positive(
                    &format!("pair_interactions[{i}].epsilon"),
                    epsilon,
                )?;
                PairPotentialParams::LennardJones { sigma, epsilon }
            }
            // Unreachable: the recognised-key match above already returned
            // UnknownPairPotential for any other value.
            _ => unreachable!("potential tag validated above"),
        };

        pair_interactions.push(PairInteractionConfig {
            between,
            cutoff,
            r_switch,
            potential: potential_params,
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

    // Angle types (optional)
    let mut angle_types: Vec<AngleTypeConfig> = Vec::new();
    if let Some(at_value) = root.get("angle_types") {
        let at_array = at_value
            .as_array()
            .ok_or_else(|| invalid("angle_types", "expected an array of tables"))?;
        let mut seen_names: Vec<String> = Vec::new();
        for (i, entry) in at_array.iter().enumerate() {
            let tbl = entry
                .as_table()
                .ok_or_else(|| invalid(&format!("angle_types[{i}]"), "expected a table"))?;
            let name = get_str(tbl, "name")
                .map_err(rename_field(format!("angle_types[{i}].name")))?
                .to_string();
            if name.is_empty() {
                return Err(invalid(
                    &format!("angle_types[{i}].name"),
                    "name must not be empty",
                ));
            }
            if seen_names.contains(&name) {
                return Err(ConfigError::DuplicateAngleTypeName { name });
            }
            let potential = get_str(tbl, "potential")
                .map_err(rename_field(format!("angle_types[{i}].potential")))?
                .to_string();
            match potential.as_str() {
                "harmonic" => {
                    for key in tbl.keys() {
                        if !matches!(
                            key.as_str(),
                            "name" | "potential" | "k_theta" | "theta_0"
                        ) {
                            return Err(ConfigError::UnknownAngleTypeField {
                                potential: "harmonic".to_string(),
                                field: key.clone(),
                                angle_type_index: i,
                            });
                        }
                    }
                    let k_theta = get_f64(tbl, "k_theta")
                        .map_err(rename_field(format!("angle_types[{i}].k_theta")))?;
                    require_finite_positive(
                        &format!("angle_types[{i}].k_theta"),
                        k_theta,
                    )?;
                    let theta_0 = get_f64(tbl, "theta_0")
                        .map_err(rename_field(format!("angle_types[{i}].theta_0")))?;
                    if !theta_0.is_finite()
                        || !(0.0..=std::f64::consts::PI).contains(&theta_0)
                    {
                        return Err(invalid(
                            &format!("angle_types[{i}].theta_0"),
                            "theta_0 must be finite and in [0, π]",
                        ));
                    }
                    seen_names.push(name.clone());
                    angle_types.push(AngleTypeConfig::Harmonic {
                        name,
                        k_theta,
                        theta_0,
                    });
                }
                other => {
                    return Err(ConfigError::UnknownAnglePotential {
                        actual: other.to_string(),
                        angle_type_index: i,
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

    // rq-846bdb8b: parse the optional [coulomb] table. The slot is
    // present iff this table is present in the config.
    let coulomb: Option<CoulombConfig> = match root.get("coulomb") {
        None => None,
        Some(toml::Value::Table(coul_tbl)) => {
            for key in coul_tbl.keys() {
                if !matches!(key.as_str(), "cutoff" | "r_switch") {
                    return Err(invalid(
                        &format!("coulomb.{key}"),
                        "unknown field for [coulomb]",
                    ));
                }
            }
            let cutoff = get_f64(coul_tbl, "cutoff")
                .map_err(rename_field("coulomb.cutoff".to_string()))?;
            require_finite_positive("coulomb.cutoff", cutoff)?;
            let r_switch = match coul_tbl.get("r_switch") {
                Some(toml::Value::Float(v)) => *v,
                Some(toml::Value::Integer(i)) => *i as f64,
                Some(_) => {
                    return Err(invalid("coulomb.r_switch", "expected a float"));
                }
                None => 0.9 * cutoff,
            };
            require_finite_positive("coulomb.r_switch", r_switch)?;
            if r_switch > cutoff {
                return Err(invalid(
                    "coulomb.r_switch",
                    format!(
                        "expected r_switch <= cutoff; got r_switch={r_switch}, cutoff={cutoff}"
                    ),
                ));
            }
            Some(CoulombConfig { cutoff, r_switch })
        }
        Some(_) => return Err(invalid("coulomb", "expected a table")),
    };

    // rq-7bd2d9ca rq-202493a5: parse the optional [spme] table. The
    // SPME slots are present iff this table is present in the config.
    let spme: Option<SpmeConfig> = match root.get("spme") {
        None => None,
        Some(toml::Value::Table(spme_tbl)) => {
            for key in spme_tbl.keys() {
                if !matches!(
                    key.as_str(),
                    "alpha" | "r_cut_real" | "grid" | "spline_order"
                ) {
                    return Err(invalid(
                        &format!("spme.{key}"),
                        "unknown field for [spme]",
                    ));
                }
            }
            let alpha = get_f64(spme_tbl, "alpha")
                .map_err(rename_field("spme.alpha".to_string()))?;
            require_finite_positive("spme.alpha", alpha)?;
            let r_cut_real = get_f64(spme_tbl, "r_cut_real")
                .map_err(rename_field("spme.r_cut_real".to_string()))?;
            require_finite_positive("spme.r_cut_real", r_cut_real)?;
            let spline_order: u32 = match spme_tbl.get("spline_order") {
                Some(toml::Value::Integer(i)) if (4..=8).contains(i) => *i as u32,
                Some(toml::Value::Integer(i)) => {
                    return Err(invalid(
                        "spme.spline_order",
                        format!("expected an integer in [4, 8]; got {i}"),
                    ));
                }
                Some(_) => return Err(invalid("spme.spline_order", "expected an integer")),
                None => 4,
            };
            let grid_value = spme_tbl
                .get("grid")
                .ok_or_else(|| missing("spme.grid"))?;
            let grid_array = grid_value
                .as_array()
                .ok_or_else(|| invalid("spme.grid", "expected an array of three integers"))?;
            if grid_array.len() != 3 {
                return Err(invalid(
                    "spme.grid",
                    format!("expected 3 integers, got {}", grid_array.len()),
                ));
            }
            let mut grid = [0u32; 3];
            let axis_names = ["a", "b", "c"];
            for (i, item) in grid_array.iter().enumerate() {
                let v = match item {
                    toml::Value::Integer(n) if *n > 0 => *n as u32,
                    toml::Value::Integer(n) => {
                        return Err(invalid(
                            &format!("spme.grid[{}]", axis_names[i]),
                            format!("expected a strictly positive integer, got {n}"),
                        ));
                    }
                    _ => {
                        return Err(invalid(
                            &format!("spme.grid[{}]", axis_names[i]),
                            "expected an integer",
                        ));
                    }
                };
                if v < 2 * spline_order {
                    return Err(invalid(
                        &format!("spme.grid[{}]", axis_names[i]),
                        format!(
                            "must be >= 2 * spline_order = {}; got {}",
                            2 * spline_order,
                            v
                        ),
                    ));
                }
                grid[i] = v;
            }
            Some(SpmeConfig {
                alpha,
                r_cut_real,
                grid,
                spline_order,
            })
        }
        Some(_) => return Err(invalid("spme", "expected a table")),
    };

    if coulomb.is_some() && spme.is_some() {
        return Err(ConfigError::ConflictingElectrostatics);
    }

    // Effective max cutoff includes pair_interactions, coulomb, and spme.
    let max_cutoff = max_cutoff
        .max(coulomb.as_ref().map(|c| c.cutoff).unwrap_or(0.0))
        .max(spme.as_ref().map(|s| s.r_cut_real).unwrap_or(0.0));

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
        include_images,
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
            let inci = match out_tbl.get("include_images") {
                Some(toml::Value::Boolean(b)) => *b,
                Some(_) => return Err(invalid("output.include_images", "expected a boolean")),
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
            (tpath, tevery, incv, inci, lpath, levery, timings)
        }
        Some(_) => return Err(invalid("output", "expected a table")),
        None => (
            resolve_path(&base_dir, &default_traj),
            100,
            true,
            true,
            resolve_path(&base_dir, &default_log),
            100,
            resolve_path(&base_dir, &default_timings),
        ),
    };

    // rq-6d99f9c8
    let init_path = resolve_path(&base_dir, &init_raw);
    let topology_path: Option<PathBuf> = topology_raw.as_deref().map(|s| resolve_path(&base_dir, s));

    // Path collision checks (init/traj/log/timings/topology pairwise distinct)
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
    if let Some(b) = topology_path.as_ref() {
        if let Some(e) = check_collision(PathRole::Init, &init_path, PathRole::Topology, b) {
            return Err(e);
        }
        if let Some(e) = check_collision(PathRole::Trajectory, &trajectory_path, PathRole::Topology, b) {
            return Err(e);
        }
        if let Some(e) = check_collision(PathRole::Log, &log_path, PathRole::Topology, b) {
            return Err(e);
        }
        if let Some(e) = check_collision(PathRole::Timings, &timings_path, PathRole::Topology, b) {
            return Err(e);
        }
    }

    Ok(Config {
        schema_version,
        init: init_path,
        topology: topology_path,
        simulation,
        integrator,
        thermostat,
        barostat,
        particle_types,
        pair_interactions,
        bond_types,
        angle_types,
        coulomb,
        spme,
        neighbor_list,
        output: OutputConfig {
            trajectory_path,
            trajectory_every,
            include_velocities,
            include_images,
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

// rq-1c2d8eba rq-3fdb7e01
fn parse_thermostat(therm_tbl: &toml::value::Table) -> Result<ThermostatKind, ConfigError> {
    let kind_str = get_str(therm_tbl, "kind")
        .map_err(rename_field("thermostat.kind".into()))?
        .to_string();
    match kind_str.as_str() {
        "nose-hoover-chain" => {
            for key in therm_tbl.keys() {
                if !matches!(
                    key.as_str(),
                    "kind"
                        | "temperature"
                        | "tau"
                        | "chain_length"
                        | "yoshida_order"
                        | "n_resp"
                ) {
                    return Err(ConfigError::UnknownThermostatField {
                        kind: "nose-hoover-chain".to_string(),
                        field: key.clone(),
                    });
                }
            }
            let temperature = get_f64(therm_tbl, "temperature")
                .map_err(rename_field("thermostat.temperature".into()))?;
            require_finite_positive("thermostat.temperature", temperature)?;
            let tau = get_f64(therm_tbl, "tau")
                .map_err(rename_field("thermostat.tau".into()))?;
            require_finite_positive("thermostat.tau", tau)?;
            let chain_length = match therm_tbl.get("chain_length") {
                Some(toml::Value::Integer(v)) if *v >= 1 => *v as u32,
                Some(toml::Value::Integer(_)) => {
                    return Err(invalid(
                        "thermostat.chain_length",
                        "chain_length must be a positive integer",
                    ));
                }
                Some(_) => {
                    return Err(invalid(
                        "thermostat.chain_length",
                        "expected an integer",
                    ));
                }
                None => 3,
            };
            let yoshida_order = match therm_tbl.get("yoshida_order") {
                Some(toml::Value::Integer(v)) if matches!(*v, 1 | 3 | 5 | 7) => *v as u32,
                Some(toml::Value::Integer(_)) => {
                    return Err(invalid(
                        "thermostat.yoshida_order",
                        "yoshida_order must be one of 1, 3, 5, 7",
                    ));
                }
                Some(_) => {
                    return Err(invalid(
                        "thermostat.yoshida_order",
                        "expected an integer",
                    ));
                }
                None => 3,
            };
            let n_resp = match therm_tbl.get("n_resp") {
                Some(toml::Value::Integer(v)) if *v >= 1 => *v as u32,
                Some(toml::Value::Integer(_)) => {
                    return Err(invalid(
                        "thermostat.n_resp",
                        "n_resp must be a positive integer",
                    ));
                }
                Some(_) => {
                    return Err(invalid("thermostat.n_resp", "expected an integer"));
                }
                None => 1,
            };
            Ok(ThermostatKind::NoseHooverChain {
                temperature,
                tau,
                chain_length,
                yoshida_order,
                n_resp,
            })
        }
        "csvr" => {
            for key in therm_tbl.keys() {
                if !matches!(key.as_str(), "kind" | "temperature" | "tau" | "seed") {
                    return Err(ConfigError::UnknownThermostatField {
                        kind: "csvr".to_string(),
                        field: key.clone(),
                    });
                }
            }
            let temperature = get_f64(therm_tbl, "temperature")
                .map_err(rename_field("thermostat.temperature".into()))?;
            require_finite_positive("thermostat.temperature", temperature)?;
            let tau = get_f64(therm_tbl, "tau")
                .map_err(rename_field("thermostat.tau".into()))?;
            require_finite_positive("thermostat.tau", tau)?;
            let seed = get_u64(therm_tbl, "seed")
                .map_err(rename_field("thermostat.seed".into()))?;
            Ok(ThermostatKind::Csvr {
                temperature,
                tau,
                seed,
            })
        }
        "andersen" => {
            for key in therm_tbl.keys() {
                if !matches!(
                    key.as_str(),
                    "kind" | "temperature" | "collision_rate" | "seed"
                ) {
                    return Err(ConfigError::UnknownThermostatField {
                        kind: "andersen".to_string(),
                        field: key.clone(),
                    });
                }
            }
            let temperature = get_f64(therm_tbl, "temperature")
                .map_err(rename_field("thermostat.temperature".into()))?;
            require_finite_positive("thermostat.temperature", temperature)?;
            let collision_rate = get_f64(therm_tbl, "collision_rate")
                .map_err(rename_field("thermostat.collision_rate".into()))?;
            require_finite_non_negative("thermostat.collision_rate", collision_rate)?;
            let seed = get_u64(therm_tbl, "seed")
                .map_err(rename_field("thermostat.seed".into()))?;
            Ok(ThermostatKind::Andersen {
                temperature,
                collision_rate,
                seed,
            })
        }
        "berendsen" => {
            for key in therm_tbl.keys() {
                if !matches!(key.as_str(), "kind" | "temperature" | "tau") {
                    return Err(ConfigError::UnknownThermostatField {
                        kind: "berendsen".to_string(),
                        field: key.clone(),
                    });
                }
            }
            let temperature = get_f64(therm_tbl, "temperature")
                .map_err(rename_field("thermostat.temperature".into()))?;
            require_finite_positive("thermostat.temperature", temperature)?;
            let tau = get_f64(therm_tbl, "tau")
                .map_err(rename_field("thermostat.tau".into()))?;
            require_finite_positive("thermostat.tau", tau)?;
            Ok(ThermostatKind::Berendsen { temperature, tau })
        }
        other => Err(ConfigError::UnknownThermostatKind {
            actual: other.to_string(),
        }),
    }
}

// rq-d28e9105 rq-aa6ce5c0 rq-3db027c2
fn parse_barostat(baro_tbl: &toml::value::Table) -> Result<BarostatKind, ConfigError> {
    let kind_str = get_str(baro_tbl, "kind")
        .map_err(rename_field("barostat.kind".into()))?
        .to_string();
    match kind_str.as_str() {
        "berendsen" => {
            for key in baro_tbl.keys() {
                if !matches!(
                    key.as_str(),
                    "kind" | "pressure" | "tau" | "compressibility"
                ) {
                    return Err(ConfigError::UnknownBarostatField {
                        kind: "berendsen".to_string(),
                        field: key.clone(),
                    });
                }
            }
            let pressure = get_f64(baro_tbl, "pressure")
                .map_err(rename_field("barostat.pressure".into()))?;
            if !pressure.is_finite() {
                return Err(invalid(
                    "barostat.pressure",
                    format!("expected a finite number, got {pressure}"),
                ));
            }
            let tau = get_f64(baro_tbl, "tau")
                .map_err(rename_field("barostat.tau".into()))?;
            require_finite_positive("barostat.tau", tau)?;
            let compressibility = get_f64(baro_tbl, "compressibility")
                .map_err(rename_field("barostat.compressibility".into()))?;
            require_finite_positive("barostat.compressibility", compressibility)?;
            Ok(BarostatKind::Berendsen {
                pressure,
                tau,
                compressibility,
            })
        }
        "c-rescale" => {
            for key in baro_tbl.keys() {
                if !matches!(
                    key.as_str(),
                    "kind"
                        | "pressure"
                        | "temperature"
                        | "tau"
                        | "compressibility"
                        | "seed"
                ) {
                    return Err(ConfigError::UnknownBarostatField {
                        kind: "c-rescale".to_string(),
                        field: key.clone(),
                    });
                }
            }
            let pressure = get_f64(baro_tbl, "pressure")
                .map_err(rename_field("barostat.pressure".into()))?;
            if !pressure.is_finite() {
                return Err(invalid(
                    "barostat.pressure",
                    format!("expected a finite number, got {pressure}"),
                ));
            }
            let temperature = get_f64(baro_tbl, "temperature")
                .map_err(rename_field("barostat.temperature".into()))?;
            require_finite_positive("barostat.temperature", temperature)?;
            let tau = get_f64(baro_tbl, "tau")
                .map_err(rename_field("barostat.tau".into()))?;
            require_finite_positive("barostat.tau", tau)?;
            let compressibility = get_f64(baro_tbl, "compressibility")
                .map_err(rename_field("barostat.compressibility".into()))?;
            require_finite_positive("barostat.compressibility", compressibility)?;
            let seed = get_u64(baro_tbl, "seed")
                .map_err(rename_field("barostat.seed".into()))?;
            Ok(BarostatKind::CRescale {
                pressure,
                temperature,
                tau,
                compressibility,
                seed,
            })
        }
        other => Err(ConfigError::UnknownBarostatKind {
            actual: other.to_string(),
        }),
    }
}
