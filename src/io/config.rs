// rq-6432ab1f rq-110285ae rq-b719c42c
use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer};

// rq-f0084057
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathRole {
    Init,
    Topology,
    PhaseTrajectory { phase: String },
    PhaseLog { phase: String },
    PhaseTimings { phase: String },
    MinimizationMinlog { phase: String },
    MinimizationTrajectory { phase: String },
    MinimizationTimings { phase: String },
}

impl std::fmt::Display for PathRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PathRole::Init => write!(f, "init"),
            PathRole::Topology => write!(f, "topology"),
            PathRole::PhaseTrajectory { phase } => write!(f, "phase `{phase}` trajectory"),
            PathRole::PhaseLog { phase } => write!(f, "phase `{phase}` log"),
            PathRole::PhaseTimings { phase } => write!(f, "phase `{phase}` timings"),
            PathRole::MinimizationMinlog { phase } => {
                write!(f, "minimization `{phase}` minlog")
            }
            PathRole::MinimizationTrajectory { phase } => {
                write!(f, "minimization `{phase}` trajectory")
            }
            PathRole::MinimizationTimings { phase } => {
                write!(f, "minimization `{phase}` timings")
            }
        }
    }
}

// rq-3108381e rq-e1ceb5c0 rq-1bbcf3b7
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    // rq-5a0f5c00
    #[error(
        "config filename `{}` does not end in `.in.toml` (or its derived root is empty)",
        path.display()
    )]
    InvalidConfigFilename { path: PathBuf },
    #[error("failed to read config file: {0}")]
    Io(String),
    // Structural error from the typed deserialiser: TOML syntax error,
    // type mismatch, unknown field for the enclosing table, or unknown
    // tagged-enum variant. `path` is a dotted JSON-pointer-like location
    // within the document; `message` is the underlying parser/deserialiser
    // message.
    #[error("config parse error at `{path}`: {message}")]
    Parse { path: String, message: String },
    #[error("unsupported schema version {actual}: only version {supported} is supported")]
    UnsupportedSchemaVersion { actual: u64, supported: u64 },
    #[error("missing required field `{field}`")]
    MissingField { field: String },
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
    #[error("output paths collide: `{kind_a}` and `{kind_b}` both resolve to `{}`", path.display())]
    PathCollision {
        kind_a: PathRole,
        kind_b: PathRole,
        path: PathBuf,
    },
    #[error("config declares both [coulomb] and [spme]; only one electrostatics method may be active per run")]
    ConflictingElectrostatics,
    #[error("config declares no [[phase]] entries; a simulation requires at least one phase")]
    EmptyPhases,
    #[error("duplicate phase name `{name}`")]
    DuplicatePhaseName { name: String },
    #[error("two stochastic slots of kind `{kind}` across the [[phase]] array declare the same seed = {seed}; pick distinct seeds to avoid correlated noise")]
    DuplicatePhaseSeed { kind: String, seed: u64 },
    #[error("integrator `{integrator}` in phase `{phase}` owns its own thermostat and is incompatible with `[phase.thermostat]`")]
    IncompatibleThermostat { integrator: String, phase: String },
    #[error("integrator `{integrator}` in phase `{phase}` owns its own barostat and is incompatible with `[phase.barostat]`")]
    IncompatibleBarostat { integrator: String, phase: String },
    #[error("duplicate bond type name `{name}`")]
    DuplicateBondTypeName { name: String },
    #[error("duplicate angle type name `{name}`")]
    DuplicateAngleTypeName { name: String },
    #[error("duplicate constraint type name `{name}`")]
    DuplicateConstraintTypeName { name: String },
    #[error("integrator `{integrator}` in phase `{phase}` does not support holonomic constraints; remove the topology file's [constraints] section or choose a different integrator")]
    IncompatibleConstraint { integrator: String, phase: String },
    #[error("constraint type `{name}` has infeasible SETTLE geometry: r_hh = {r_hh} must be < 2 * r_oh ({r_oh})")]
    SettleGeometryInfeasible { name: String, r_oh: f64, r_hh: f64 },
    #[error("[{slot}] section's `kind = \"{kind}\"` does not match any registered builder")]
    UnknownKind { slot: &'static str, kind: String },
}

// =====================================================================
// Public config types
// =====================================================================

// rq-53055a5b — `[simulation]` carries only the inputs for the
// initial Maxwell-Boltzmann velocity sampling (fired once at phase-0
// entry). Per-step settings (`dt`, `n_steps`) live on each
// `[[phase]]` entry.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationConfig {
    pub seed: u64,
    pub temperature: f64,
}

// rq-18441e33 — parsed `[[phase]]` entry. The runner walks
// `Config::phases` in declaration order; particle state carries
// across phase boundaries while slot state is rebuilt at each one.
#[derive(Debug, Clone)]
pub struct PhaseConfig {
    pub name: String,
    pub n_steps: u64,
    pub dt: f64,
    pub integrator: SlotConfig,
    pub thermostat: Option<SlotConfig>,
    pub barostat: Option<SlotConfig>,
    pub output: OutputConfig,
}

/// Parsed `[[minimization]]` entry. Energy-minimization phases run
/// the SD outer loop documented in
/// `rqm/minimization/steepest-descent.md`.
// rq-ed61cf26
#[derive(Debug, Clone)]
pub struct MinimizationConfig {
    pub name: String,
    pub algorithm: SlotConfig,
    pub output: MinimizationOutputConfig,
}

// rq-758b03ef
/// Resolved per-phase outputs for a `[[minimization]]` entry.
#[derive(Debug, Clone)]
pub struct MinimizationOutputConfig {
    pub minlog_path: PathBuf,
    pub minlog_every: u64,
    pub trajectory_path: PathBuf,
    pub trajectory_every: u64,
    pub include_images: bool,
    pub timings_path: PathBuf,
}

// rq-19226daf
/// Discriminated union over the unified phase sequence. The runner
/// walks `Config::phases: Vec<PhaseKind>` in source-document order
/// (see `Phase kinds` in `rqm/io/config-schema.md`).
#[derive(Debug, Clone)]
pub enum PhaseKind {
    Md(PhaseConfig),
    Minimization(MinimizationConfig),
}

impl PhaseKind {
    pub fn name(&self) -> &str {
        match self {
            PhaseKind::Md(p) => &p.name,
            PhaseKind::Minimization(m) => &m.name,
        }
    }

    pub fn timings_path(&self) -> &Path {
        match self {
            PhaseKind::Md(p) => &p.output.timings_path,
            PhaseKind::Minimization(m) => &m.output.timings_path,
        }
    }

    pub fn as_md(&self) -> Option<&PhaseConfig> {
        match self {
            PhaseKind::Md(p) => Some(p),
            _ => None,
        }
    }

    pub fn as_minimization(&self) -> Option<&MinimizationConfig> {
        match self {
            PhaseKind::Minimization(m) => Some(m),
            _ => None,
        }
    }
}

// rq-661bf664
/// Open-shaped parsed selection for a singleton `[integrator]`,
/// `[thermostat]`, or `[barostat]` config section. The Rust-side
/// deserialiser captures the user's `kind = "..."` field into `kind`
/// and flattens every other field of the section into a `toml::Value`
/// (a `toml::Table`) that the chosen builder consumes via
/// `validate_params(&toml::Value)` and `build(...)`.
#[derive(Debug, Clone)]
pub struct SlotConfig {
    pub kind: String,
    pub params: toml::Value,
}

impl SlotConfig {
    pub fn new(kind: impl Into<String>, params: toml::Value) -> Self {
        SlotConfig {
            kind: kind.into(),
            params,
        }
    }

    /// Convenience for tests: parse a TOML fragment into the
    /// `params` field. Panics on malformed input.
    pub fn from_params_str(kind: &str, params_toml: &str) -> Self {
        let value: toml::Value = toml::from_str(params_toml)
            .unwrap_or_else(|e| panic!("malformed params TOML: {e}"));
        SlotConfig {
            kind: kind.to_string(),
            params: value,
        }
    }
}

impl<'de> Deserialize<'de> for SlotConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let mut table = <toml::Table as Deserialize>::deserialize(d)?;
        let kind = table
            .remove("kind")
            .ok_or_else(|| serde::de::Error::missing_field("kind"))?;
        let kind = match kind {
            toml::Value::String(s) => s,
            _ => {
                return Err(serde::de::Error::custom(
                    "field `kind` must be a string",
                ));
            }
        };
        Ok(SlotConfig {
            kind,
            params: toml::Value::Table(table),
        })
    }
}

// rq-3fdb7e01
/// Open-shaped parsed entry for an array-of-named-slots config
/// section (currently only `[[constraint_types]]`). Adds a `name`
/// field that other parts of the config reference by string.
#[derive(Debug, Clone)]
pub struct NamedSlotConfig {
    pub name: String,
    pub kind: String,
    pub params: toml::Value,
}

impl NamedSlotConfig {
    pub fn new(
        name: impl Into<String>,
        kind: impl Into<String>,
        params: toml::Value,
    ) -> Self {
        NamedSlotConfig {
            name: name.into(),
            kind: kind.into(),
            params,
        }
    }

    /// Convenience for tests: parse a TOML fragment into the
    /// `params` field. Panics on malformed input.
    pub fn from_params_str(name: &str, kind: &str, params_toml: &str) -> Self {
        let value: toml::Value = toml::from_str(params_toml)
            .unwrap_or_else(|e| panic!("malformed params TOML: {e}"));
        NamedSlotConfig {
            name: name.to_string(),
            kind: kind.to_string(),
            params: value,
        }
    }
}

/// Translate a `toml::de::Error` (typically from
/// `toml::Value::try_into::<Params>()`) into a `ConfigError` for use
/// from per-builder `validate_params` impls. Routes the
/// "missing field `x`" case to `MissingField { field: "<slot>.x" }`
/// to preserve the user-visible error shape, and otherwise wraps the
/// message in `Parse { path: <slot>, message }`.
pub fn translate_params_error(slot: &str, e: toml::de::Error) -> ConfigError {
    let msg = e.to_string();
    // serde's missing-field message starts with "missing field `<name>`".
    if let Some(rest) = msg.strip_prefix("missing field `") {
        if let Some(end) = rest.find('`') {
            let field = &rest[..end];
            return ConfigError::MissingField {
                field: format!("{slot}.{field}"),
            };
        }
    }
    ConfigError::Parse {
        path: slot.to_string(),
        message: msg,
    }
}

impl<'de> Deserialize<'de> for NamedSlotConfig {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let mut table = <toml::Table as Deserialize>::deserialize(d)?;
        let name = table
            .remove("name")
            .ok_or_else(|| serde::de::Error::missing_field("name"))?;
        let name = match name {
            toml::Value::String(s) => s,
            _ => {
                return Err(serde::de::Error::custom(
                    "field `name` must be a string",
                ));
            }
        };
        let kind = table
            .remove("kind")
            .ok_or_else(|| serde::de::Error::missing_field("kind"))?;
        let kind = match kind {
            toml::Value::String(s) => s,
            _ => {
                return Err(serde::de::Error::custom(
                    "field `kind` must be a string",
                ));
            }
        };
        Ok(NamedSlotConfig {
            name,
            kind,
            params: toml::Value::Table(table),
        })
    }
}

// rq-a5ccc1de
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParticleTypeConfig {
    pub name: String,
    pub mass: f64,
    #[serde(default)]
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

// rq-2f230ccb
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

// rq-a47beb76
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
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpmeConfig {
    pub alpha: f64,
    pub r_cut_real: f64,
    pub grid: [u32; 3],
    #[serde(default = "default_spline_order")]
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
    /// Unified phase sequence in source-document order: each entry is
    /// either a `PhaseKind::Md(PhaseConfig)` from a `[[phase]]` table
    /// or a `PhaseKind::Minimization(MinimizationConfig)` from a
    /// `[[minimization]]` table.
    pub phases: Vec<PhaseKind>,
    pub particle_types: Vec<ParticleTypeConfig>,
    pub pair_interactions: Vec<PairInteractionConfig>,
    pub bond_types: Vec<BondTypeConfig>,
    pub angle_types: Vec<AngleTypeConfig>,
    pub constraint_types: Vec<NamedSlotConfig>,
    pub coulomb: Option<CoulombConfig>,
    pub spme: Option<SpmeConfig>,
    pub neighbor_list: NeighborListConfig,
    pub config_path: PathBuf,
}

// =====================================================================
// Default-value helpers used by `#[serde(default = "...")]`
// =====================================================================

fn default_spline_order() -> u32 {
    4
}
fn default_max_neighbors() -> u32 {
    256
}
fn default_trajectory_every() -> u64 {
    100
}
fn default_log_every() -> u64 {
    100
}
fn default_true() -> bool {
    true
}

// =====================================================================
// Raw types: deserialise-side mirrors for entries with field-derived
// defaults or post-parse normalisation. Convert into the public type via
// `From` (when the conversion is context-free) or via the helpers in
// `build_config` (when it needs e.g. the max cutoff).
// =====================================================================

const SUPPORTED_SCHEMA_VERSION: u64 = 1;

#[derive(Debug, Deserialize)]
struct RawConfig {
    schema_version: u64,
    init: String,
    #[serde(default)]
    topology: Option<String>,
    simulation: SimulationConfig,
    #[serde(default, rename = "phase")]
    phases: Vec<toml::Spanned<RawPhaseConfig>>,
    #[serde(default, rename = "minimization")]
    minimizations: Vec<toml::Spanned<RawMinimizationConfig>>,
    particle_types: Vec<ParticleTypeConfig>,
    #[serde(default)]
    pair_interactions: Vec<RawPairInteraction>,
    #[serde(default)]
    bond_types: Vec<RawBondType>,
    #[serde(default)]
    angle_types: Vec<RawAngleType>,
    #[serde(default)]
    constraint_types: Vec<NamedSlotConfig>,
    #[serde(default)]
    coulomb: Option<RawCoulombConfig>,
    #[serde(default)]
    spme: Option<SpmeConfig>,
    #[serde(default)]
    neighbor_list: Option<RawNeighborList>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMinimizationConfig {
    name: String,
    algorithm: SlotConfig,
    #[serde(default)]
    output: Option<RawMinimizationOutputConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMinimizationOutputConfig {
    #[serde(default)]
    minlog_path: Option<String>,
    #[serde(default = "default_minlog_every")]
    minlog_every: u64,
    #[serde(default)]
    trajectory_path: Option<String>,
    #[serde(default)]
    trajectory_every: u64,
    #[serde(default = "default_true")]
    include_images: bool,
    #[serde(default)]
    timings_path: Option<String>,
}

fn default_minlog_every() -> u64 {
    1
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPhaseConfig {
    name: String,
    n_steps: u64,
    dt: f64,
    integrator: SlotConfig,
    #[serde(default)]
    thermostat: Option<SlotConfig>,
    #[serde(default)]
    barostat: Option<SlotConfig>,
    #[serde(default)]
    output: Option<RawOutputConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "potential", rename_all = "kebab-case", deny_unknown_fields)]
enum RawPairInteraction {
    LennardJones {
        between: [String; 2],
        cutoff: f64,
        #[serde(default)]
        r_switch: Option<f64>,
        sigma: f64,
        epsilon: f64,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "potential", rename_all = "kebab-case", deny_unknown_fields)]
enum RawBondType {
    Morse {
        name: String,
        de: f64,
        a: f64,
        re: f64,
    },
}

impl From<RawBondType> for BondTypeConfig {
    fn from(r: RawBondType) -> Self {
        match r {
            RawBondType::Morse { name, de, a, re } => BondTypeConfig::Morse { name, de, a, re },
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "potential", rename_all = "kebab-case", deny_unknown_fields)]
enum RawAngleType {
    Harmonic {
        name: String,
        k_theta: f64,
        theta_0: f64,
    },
}

impl From<RawAngleType> for AngleTypeConfig {
    fn from(r: RawAngleType) -> Self {
        match r {
            RawAngleType::Harmonic {
                name,
                k_theta,
                theta_0,
            } => AngleTypeConfig::Harmonic {
                name,
                k_theta,
                theta_0,
            },
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCoulombConfig {
    cutoff: f64,
    #[serde(default)]
    r_switch: Option<f64>,
}

impl From<RawCoulombConfig> for CoulombConfig {
    fn from(r: RawCoulombConfig) -> Self {
        let r_switch = r.r_switch.unwrap_or(0.9 * r.cutoff);
        CoulombConfig {
            cutoff: r.cutoff,
            r_switch,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "mode", rename_all = "kebab-case", deny_unknown_fields)]
enum RawNeighborList {
    // Empty-struct form (`AllPairs {}`) so `deny_unknown_fields` rejects
    // sibling fields like `max_neighbors` / `r_skin` under
    // `mode = "all-pairs"`. Unit variants in internally-tagged enums
    // skip the deny check.
    AllPairs {},
    CellList {
        #[serde(default = "default_max_neighbors")]
        max_neighbors: u32,
        #[serde(default)]
        r_skin: Option<f64>,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOutputConfig {
    #[serde(default)]
    trajectory_path: Option<String>,
    #[serde(default = "default_trajectory_every")]
    trajectory_every: u64,
    #[serde(default = "default_true")]
    include_velocities: bool,
    #[serde(default = "default_true")]
    include_images: bool,
    #[serde(default)]
    log_path: Option<String>,
    #[serde(default = "default_log_every")]
    log_every: u64,
    #[serde(default)]
    timings_path: Option<String>,
}

// =====================================================================
// load_config / load_config_raw
// =====================================================================

// rq-45bb8194 — default loader for callers that use only the built-in
// slot kinds. Custom-builder callers use `load_config_raw` plus
// `validate_against(&their_registries)` instead.
pub fn load_config(path: &Path) -> Result<Config, ConfigError> {
    let config = load_config_raw(path)?;
    config.validate_against(&crate::Registries::with_builtins())?;
    Ok(config)
}

// rq-deaf8b59 — parse-only entry point: read the file, run the typed
// TOML deserialiser, fill defaults, resolve paths, run `Config::validate`,
// and return. Skips `Config::validate_against` so callers can register
// custom builders and supply their own registries.
pub fn load_config_raw(path: &Path) -> Result<Config, ConfigError> {
    // rq-5a0f5c00 — filename-convention check runs before any I/O.
    let _root = derive_config_root(path)?;

    let raw_text = std::fs::read_to_string(path)
        .map_err(|e| ConfigError::Io(format!("{}: {}", path.display(), e)))?;

    let de = toml::Deserializer::new(&raw_text);
    let raw_config: RawConfig =
        serde_path_to_error::deserialize(de).map_err(serde_error_to_config_error)?;

    if raw_config.schema_version != SUPPORTED_SCHEMA_VERSION {
        return Err(ConfigError::UnsupportedSchemaVersion {
            actual: raw_config.schema_version,
            supported: SUPPORTED_SCHEMA_VERSION,
        });
    }

    let base_dir = path.parent().unwrap_or(Path::new("."));
    let config = build_config(raw_config, path, base_dir);
    config.validate()?;
    Ok(config)
}

// Translate a `serde_path_to_error::Error<toml::de::Error>` into the
// `ConfigError` shape: detect "missing field `X`" patterns and route
// those to `MissingField`; everything else becomes `Parse`.
fn serde_error_to_config_error(
    err: serde_path_to_error::Error<toml::de::Error>,
) -> ConfigError {
    let raw_path = err.path().to_string();
    // serde_path_to_error renders the empty path as "." (the root
    // marker). Strip it so callers see "init" rather than ".init".
    let trimmed = raw_path.trim_matches('.');
    let path = normalise_path(trimmed);
    // Strip the `phase[N].` prefix so per-slot error paths look the same
    // whether the error came from the raw deserialisation step or from a
    // builder's `validate_params` call.
    let path = strip_phase_prefix(&path);
    let inner = err.into_inner();
    let message = inner.to_string();

    if let Some(field) = extract_missing_field(&message) {
        let full = if path.is_empty() {
            field
        } else {
            format!("{path}.{field}")
        };
        ConfigError::MissingField { field: full }
    } else {
        ConfigError::Parse { path, message }
    }
}

fn strip_phase_prefix(path: &str) -> String {
    // `[[phase]]` and `[[minimization]]` deserialise through
    // `toml::Spanned<T>`, which inserts an internal
    // `$__serde_spanned_private_value` segment in the serde_path_to_error
    // path. Strip it so error paths look the same whether the entry was
    // wrapped or not.
    let path = strip_spanned_prefix(path, "phase");
    let path = strip_spanned_prefix(&path, "minimization");
    // Match `phase[N]` or `phase[N].`; strip both the bracket section
    // and any trailing `.`.
    if let Some(rest) = path.strip_prefix("phase[") {
        if let Some(end) = rest.find(']') {
            let after = &rest[end + 1..];
            return after.strip_prefix('.').unwrap_or(after).to_string();
        }
    }
    if let Some(rest) = path.strip_prefix("minimization[") {
        if let Some(end) = rest.find(']') {
            let after = &rest[end + 1..];
            return after.strip_prefix('.').unwrap_or(after).to_string();
        }
    }
    path.to_string()
}

// Collapse a `toml::Spanned<T>` path segment by removing the internal
// `$__serde_spanned_private_value` marker. After the Spanned wrap the
// path of a `[[phase]]` (or `[[minimization]]`) entry's field looks
// like `phase[N].$__serde_spanned_private_value.integrator` (or
// `minimization[N].$__serde_spanned_private_value.algorithm`); we
// strip the marker so the remaining path looks the same as before
// wrapping.
fn strip_spanned_prefix(path: &str, field: &str) -> String {
    let needle_with_idx = format!("{field}[");
    if let Some(idx) = path.find(&needle_with_idx) {
        // Skip past `field[N]`.
        let after_bracket = &path[idx + needle_with_idx.len()..];
        if let Some(close) = after_bracket.find(']') {
            let prefix = &path[..idx + needle_with_idx.len() + close + 1];
            let rest = &after_bracket[close + 1..];
            let stripped = rest
                .strip_prefix(".$__serde_spanned_private_value")
                .unwrap_or(rest);
            return format!("{prefix}{stripped}");
        }
    }
    path.to_string()
}

// serde_path_to_error renders array indices as `.0`, `.1`, ...; convert
// them to the `[i]` form used in error-message contracts.
fn normalise_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    let mut chars = path.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '.' && chars.peek().map_or(false, |n| n.is_ascii_digit()) {
            // ".N" -> "[N]"
            out.push('[');
            while let Some(&n) = chars.peek() {
                if n.is_ascii_digit() {
                    out.push(n);
                    chars.next();
                } else {
                    break;
                }
            }
            out.push(']');
        } else {
            out.push(c);
        }
    }
    out
}

// rq-5a0f5c00 — derive `<config-root>` from a config path:
//   1. Take the final filename component.
//   2. Require it to end in `.in.toml` (case-sensitive, exact suffix).
//   3. Strip the `.toml` and one trailing `.in`.
//   4. Reject an empty result (e.g. the filename is `.in.toml`).
// The check is purely syntactic; the file is not opened.
fn derive_config_root(path: &Path) -> Result<String, ConfigError> {
    let invalid = || ConfigError::InvalidConfigFilename {
        path: path.to_path_buf(),
    };
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(invalid)?;
    let without_toml = filename.strip_suffix(".in.toml").ok_or_else(invalid)?;
    if without_toml.is_empty() {
        return Err(invalid());
    }
    Ok(without_toml.to_string())
}

// Extract `dt` from messages like ``missing field `dt` `` or
// ``missing field "dt"``. Returns None for anything else.
fn extract_missing_field(msg: &str) -> Option<String> {
    let needle = "missing field";
    let idx = msg.find(needle)?;
    let rest = &msg[idx + needle.len()..];
    // Skip whitespace, then expect a quote (backtick or double-quote).
    let rest = rest.trim_start();
    let open = rest.chars().next()?;
    let close = match open {
        '`' => '`',
        '"' => '"',
        _ => return None,
    };
    let after_open = &rest[open.len_utf8()..];
    let end = after_open.find(close)?;
    Some(after_open[..end].to_string())
}

// Populate `Config` from `RawConfig` by resolving paths, filling
// derived defaults, and converting the Raw sub-types. Does not validate
// (that's `Config::validate`).
fn build_config(raw: RawConfig, config_path: &Path, base_dir: &Path) -> Config {
    let init = resolve_path(base_dir, &raw.init);
    let topology = raw.topology.as_deref().map(|s| resolve_path(base_dir, s));

    // Translate the pair_interactions raw form into the public form,
    // normalising the type-name pair and filling r_switch defaults.
    let pair_interactions: Vec<PairInteractionConfig> = raw
        .pair_interactions
        .into_iter()
        .map(|r| match r {
            RawPairInteraction::LennardJones {
                between,
                cutoff,
                r_switch,
                sigma,
                epsilon,
            } => PairInteractionConfig {
                between: normalise_pair(&between[0], &between[1]),
                cutoff,
                r_switch: r_switch.unwrap_or(0.9 * cutoff),
                potential: PairPotentialParams::LennardJones { sigma, epsilon },
            },
        })
        .collect();

    let bond_types: Vec<BondTypeConfig> = raw.bond_types.into_iter().map(Into::into).collect();
    let angle_types: Vec<AngleTypeConfig> = raw.angle_types.into_iter().map(Into::into).collect();
    let constraint_types: Vec<NamedSlotConfig> = raw.constraint_types;
    let coulomb: Option<CoulombConfig> = raw.coulomb.map(Into::into);
    let spme = raw.spme;

    // Compute the maximum cutoff across pair_interactions, coulomb,
    // and spme.r_cut_real; used to derive r_skin's default when
    // [neighbor_list] is absent or its r_skin field is omitted.
    let max_cutoff = {
        let mut m: f64 = 0.0;
        for p in &pair_interactions {
            if p.cutoff > m {
                m = p.cutoff;
            }
        }
        if let Some(c) = coulomb.as_ref() {
            if c.cutoff > m {
                m = c.cutoff;
            }
        }
        if let Some(s) = spme.as_ref() {
            if (s.r_cut_real as f64) > m {
                m = s.r_cut_real;
            }
        }
        m
    };

    let neighbor_list = match raw.neighbor_list {
        None => NeighborListConfig::CellList {
            max_neighbors: default_max_neighbors(),
            r_skin: 0.3 * max_cutoff,
        },
        Some(RawNeighborList::AllPairs {}) => NeighborListConfig::AllPairs,
        Some(RawNeighborList::CellList {
            max_neighbors,
            r_skin,
        }) => NeighborListConfig::CellList {
            max_neighbors,
            r_skin: r_skin.unwrap_or(0.3 * max_cutoff),
        },
    };

    // rq-5a0f5c00 — `<config-root>` is the filename with `.toml` stripped
    // and one trailing `.in` stripped. `derive_config_root` is the single
    // source of truth; `load_config_raw` has already validated the suffix
    // before reaching this point, but `build_config` is also reachable from
    // `Config::from_raw_for_tests`-style paths that bypass the loader, so
    // fall back to the bare file stem if derivation fails rather than
    // panicking — `Config::validate` will not catch this, but only the
    // loader entry point should ever be calling `build_config` in practice.
    let config_root = derive_config_root(config_path)
        .unwrap_or_else(|_| {
            config_path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "sim".to_string())
        });

    // Per-phase output paths default to
    // `<root>.out.<phase-name>.{xyz,log,timings}` when the per-phase
    // `[phase.output]` block is absent or its individual fields are
    // omitted. The merged sequence preserves source-document order
    // by sorting both `[[phase]]` and `[[minimization]]` entries by
    // their byte-span start (via `toml::Spanned<T>`).
    enum SpannedEntry {
        Md(toml::Spanned<RawPhaseConfig>),
        Min(toml::Spanned<RawMinimizationConfig>),
    }
    impl SpannedEntry {
        fn span_start(&self) -> usize {
            match self {
                SpannedEntry::Md(s) => s.span().start,
                SpannedEntry::Min(s) => s.span().start,
            }
        }
    }
    let mut entries: Vec<SpannedEntry> = Vec::with_capacity(
        raw.phases.len() + raw.minimizations.len(),
    );
    for p in raw.phases {
        entries.push(SpannedEntry::Md(p));
    }
    for m in raw.minimizations {
        entries.push(SpannedEntry::Min(m));
    }
    entries.sort_by_key(|e| e.span_start());

    let phases: Vec<PhaseKind> = entries
        .into_iter()
        .map(|entry| match entry {
            SpannedEntry::Md(spanned) => {
                let p = spanned.into_inner();
                let name = p.name;
                let output = match p.output {
                    None => OutputConfig {
                        trajectory_path: base_dir
                            .join(format!("{config_root}.out.{name}.xyz")),
                        trajectory_every: default_trajectory_every(),
                        include_velocities: true,
                        include_images: true,
                        log_path: base_dir.join(format!("{config_root}.out.{name}.log")),
                        log_every: default_log_every(),
                        timings_path: base_dir
                            .join(format!("{config_root}.out.{name}.timings")),
                    },
                    Some(o) => OutputConfig {
                        trajectory_path: o
                            .trajectory_path
                            .as_deref()
                            .map(|s| resolve_path(base_dir, s))
                            .unwrap_or_else(|| {
                                base_dir.join(format!("{config_root}.out.{name}.xyz"))
                            }),
                        trajectory_every: o.trajectory_every,
                        include_velocities: o.include_velocities,
                        include_images: o.include_images,
                        log_path: o
                            .log_path
                            .as_deref()
                            .map(|s| resolve_path(base_dir, s))
                            .unwrap_or_else(|| {
                                base_dir.join(format!("{config_root}.out.{name}.log"))
                            }),
                        log_every: o.log_every,
                        timings_path: o
                            .timings_path
                            .as_deref()
                            .map(|s| resolve_path(base_dir, s))
                            .unwrap_or_else(|| {
                                base_dir.join(format!("{config_root}.out.{name}.timings"))
                            }),
                    },
                };
                PhaseKind::Md(PhaseConfig {
                    name,
                    n_steps: p.n_steps,
                    dt: p.dt,
                    integrator: p.integrator,
                    thermostat: p.thermostat,
                    barostat: p.barostat,
                    output,
                })
            }
            SpannedEntry::Min(spanned) => {
                let m = spanned.into_inner();
                let name = m.name;
                let output = match m.output {
                    None => MinimizationOutputConfig {
                        minlog_path: base_dir
                            .join(format!("{config_root}.out.{name}.minlog")),
                        minlog_every: default_minlog_every(),
                        trajectory_path: base_dir
                            .join(format!("{config_root}.out.{name}.xyz")),
                        trajectory_every: 0,
                        include_images: true,
                        timings_path: base_dir
                            .join(format!("{config_root}.out.{name}.timings")),
                    },
                    Some(o) => MinimizationOutputConfig {
                        minlog_path: o
                            .minlog_path
                            .as_deref()
                            .map(|s| resolve_path(base_dir, s))
                            .unwrap_or_else(|| {
                                base_dir.join(format!("{config_root}.out.{name}.minlog"))
                            }),
                        minlog_every: o.minlog_every,
                        trajectory_path: o
                            .trajectory_path
                            .as_deref()
                            .map(|s| resolve_path(base_dir, s))
                            .unwrap_or_else(|| {
                                base_dir.join(format!("{config_root}.out.{name}.xyz"))
                            }),
                        trajectory_every: o.trajectory_every,
                        include_images: o.include_images,
                        timings_path: o
                            .timings_path
                            .as_deref()
                            .map(|s| resolve_path(base_dir, s))
                            .unwrap_or_else(|| {
                                base_dir.join(format!("{config_root}.out.{name}.timings"))
                            }),
                    },
                };
                PhaseKind::Minimization(MinimizationConfig {
                    name,
                    algorithm: m.algorithm,
                    output,
                })
            }
        })
        .collect();

    Config {
        schema_version: raw.schema_version,
        init,
        topology,
        simulation: raw.simulation,
        phases,
        particle_types: raw.particle_types,
        pair_interactions,
        bond_types,
        angle_types,
        constraint_types,
        coulomb,
        spme,
        neighbor_list,
        config_path: config_path.to_path_buf(),
    }
}

// =====================================================================
// Config::validate
// =====================================================================

impl Config {
    // rq-a54cc657
    /// Structural validation that does not require registry access.
    /// Per-field domain checks for the per-slot `params` and the
    /// integrator-thermostat / integrator-barostat / lossless-with-
    /// constraints compatibility checks live in
    /// [`Config::validate_against`] because they consult the open
    /// builder registries.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Per-field domain checks in declaration order.
        validate_simulation(&self.simulation)?;
        validate_phases(&self.phases)?;
        validate_particle_types(&self.particle_types)?;
        validate_pair_interactions(&self.pair_interactions, &self.particle_types)?;
        validate_bond_types(&self.bond_types)?;
        validate_angle_types(&self.angle_types)?;
        validate_constraint_type_names(&self.constraint_types)?;
        if let Some(c) = &self.coulomb {
            validate_coulomb(c)?;
        }
        if let Some(s) = &self.spme {
            validate_spme(s)?;
        }
        validate_neighbor_list(&self.neighbor_list)?;

        // Structural cross-validation: pair coverage, path collisions,
        // and electrostatics exclusivity. The integrator/thermostat/
        // barostat compatibility rules require builder predicates, so
        // they live in `validate_against`.
        check_pair_coverage(&self.particle_types, &self.pair_interactions)?;
        check_path_collisions(self)?;
        if self.coulomb.is_some() && self.spme.is_some() {
            return Err(ConfigError::ConflictingElectrostatics);
        }
        Ok(())
    }

    /// Registry-dispatched validation: looks up each slot's `kind` in
    /// the corresponding registry, calls
    /// `builder.validate_params(&params)`, and enforces the
    // rq-6082cd2d
    /// integrator-thermostat and integrator-barostat compatibility
    /// rules using the integrator builder's `owns_thermostat` /
    /// `owns_barostat` predicates.
    pub fn validate_against(
        &self,
        registries: &crate::Registries,
    ) -> Result<(), ConfigError> {
        // Constraint types are global (one declaration across the
        // whole run); validate them once.
        for ct in &self.constraint_types {
            let cb = registries
                .constraint_types
                .lookup(&ct.kind)
                .ok_or_else(|| ConfigError::UnknownKind {
                    slot: "constraint_types",
                    kind: ct.kind.clone(),
                })?;
            cb.validate_params(&ct.params).map_err(|e| match e {
                // Promote the entry's `name` into name-bearing errors
                // that the builder couldn't fill in itself (it only
                // sees the params, not the entry's name).
                ConfigError::SettleGeometryInfeasible {
                    name: _,
                    r_oh,
                    r_hh,
                } => ConfigError::SettleGeometryInfeasible {
                    name: ct.name.clone(),
                    r_oh,
                    r_hh,
                },
                other => other,
            })?;
        }

        // Per-phase validation. MD and minimization phases follow
        // distinct dispatch paths.
        for phase in &self.phases {
            match phase {
                PhaseKind::Md(md) => {
                    let integ_builder = registries
                        .integrators
                        .lookup(&md.integrator.kind)
                        .ok_or_else(|| ConfigError::UnknownKind {
                            slot: "integrator",
                            kind: md.integrator.kind.clone(),
                        })?;
                    integ_builder.validate_params(&md.integrator.params)?;

                    if let Some(t) = &md.thermostat {
                        let b = registries.thermostats.lookup(&t.kind).ok_or_else(
                            || ConfigError::UnknownKind {
                                slot: "thermostat",
                                kind: t.kind.clone(),
                            },
                        )?;
                        b.validate_params(&t.params)?;
                    }

                    if let Some(b) = &md.barostat {
                        let bb = registries.barostats.lookup(&b.kind).ok_or_else(
                            || ConfigError::UnknownKind {
                                slot: "barostat",
                                kind: b.kind.clone(),
                            },
                        )?;
                        bb.validate_params(&b.params)?;
                    }

                    // Integrator-owns-thermostat / integrator-owns-
                    // barostat compatibility, per phase.
                    if md.thermostat.is_some()
                        && integ_builder.owns_thermostat(&md.integrator.params)
                    {
                        return Err(ConfigError::IncompatibleThermostat {
                            integrator: md.integrator.kind.clone(),
                            phase: md.name.clone(),
                        });
                    }
                    if md.barostat.is_some()
                        && integ_builder.owns_barostat(&md.integrator.params)
                    {
                        return Err(ConfigError::IncompatibleBarostat {
                            integrator: md.integrator.kind.clone(),
                            phase: md.name.clone(),
                        });
                    }
                }
                PhaseKind::Minimization(min) => {
                    let mb = registries.minimizers.lookup(&min.algorithm.kind).ok_or_else(
                        || ConfigError::UnknownKind {
                            slot: "minimization",
                            kind: min.algorithm.kind.clone(),
                        },
                    )?;
                    mb.validate_params(&min.algorithm.params)?;
                }
            }
        }
        Ok(())
    }

    /// Topology-coupled cross-validation. For every MD phase: rejects
    // rq-723d202b
    /// a non-empty constraint list when the chosen integrator's
    /// builder `IntegratorBuilder::supports_constraints(&params)`
    /// returns `false`. For every minimization phase: rejects a
    /// non-empty constraint list when any registered constraint-type
    /// builder reports
    /// `ConstraintBuilder::supports_position_projection_only(&params) == false`.
    pub fn validate_constraint_compatibility(
        &self,
        registries: &crate::Registries,
        has_constraints: bool,
    ) -> Result<(), ConfigError> {
        if !has_constraints {
            return Ok(());
        }
        for phase in &self.phases {
            match phase {
                PhaseKind::Md(md) => {
                    let integ_builder = registries
                        .integrators
                        .lookup(&md.integrator.kind)
                        .ok_or_else(|| ConfigError::UnknownKind {
                            slot: "integrator",
                            kind: md.integrator.kind.clone(),
                        })?;
                    if !integ_builder.supports_constraints(&md.integrator.params) {
                        return Err(ConfigError::IncompatibleConstraint {
                            integrator: md.integrator.kind.clone(),
                            phase: md.name.clone(),
                        });
                    }
                }
                PhaseKind::Minimization(min) => {
                    // Cross-check every registered constraint type: if
                    // any reports `supports_position_projection_only =
                    // false`, the combination with this minimization
                    // phase is rejected. In the default registry
                    // SETTLE returns `true`, so this branch is
                    // reachable only with custom builders.
                    for ct in &self.constraint_types {
                        let cb = registries
                            .constraint_types
                            .lookup(&ct.kind)
                            .ok_or_else(|| ConfigError::UnknownKind {
                                slot: "constraint_types",
                                kind: ct.kind.clone(),
                            })?;
                        if !cb.supports_position_projection_only(&ct.params) {
                            return Err(ConfigError::IncompatibleConstraint {
                                integrator: min.algorithm.kind.clone(),
                                phase: min.name.clone(),
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// =====================================================================
// Per-field validation helpers
// =====================================================================

fn invalid(field: impl Into<String>, reason: impl Into<String>) -> ConfigError {
    ConfigError::InvalidValue {
        field: field.into(),
        reason: reason.into(),
    }
}

fn require_finite_positive(field: &str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() {
        return Err(invalid(field, format!("expected a finite number, got {value}")));
    }
    if value <= 0.0 {
        return Err(invalid(
            field,
            format!("expected a strictly positive value, got {value}"),
        ));
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

fn require_finite(field: &str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() {
        return Err(invalid(field, format!("expected a finite number, got {value}")));
    }
    Ok(())
}

fn validate_simulation(s: &SimulationConfig) -> Result<(), ConfigError> {
    require_finite_non_negative("simulation.temperature", s.temperature)?;
    Ok(())
}

// rq-18441e33 — per-phase structural validation: non-empty merged
// phase sequence, non-empty/ASCII-only/unique names, finite positive
// dt (MD only), plus the cross-phase seed-uniqueness rule (no two
// stochastic slots of the same kind across all phases may share a
// seed).
fn validate_phases(phases: &[PhaseKind]) -> Result<(), ConfigError> {
    if phases.is_empty() {
        return Err(ConfigError::EmptyPhases);
    }
    let mut seen: std::collections::HashSet<&str> =
        std::collections::HashSet::with_capacity(phases.len());
    for (i, p) in phases.iter().enumerate() {
        let (name, is_min) = match p {
            PhaseKind::Md(md) => (md.name.as_str(), false),
            PhaseKind::Minimization(min) => (min.name.as_str(), true),
        };
        let label = if is_min { "minimization" } else { "phase" };
        if name.is_empty() {
            return Err(invalid(
                format!("{label}[{i}].name"),
                "must be non-empty",
            ));
        }
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(invalid(
                format!("{label}[{i}].name"),
                "must contain only ASCII letters, digits, `-`, and `_`",
            ));
        }
        if !seen.insert(name) {
            return Err(ConfigError::DuplicatePhaseName {
                name: name.to_string(),
            });
        }
        if let PhaseKind::Md(md) = p {
            require_finite_positive(&format!("phase[{i}].dt"), md.dt)?;
        }
    }

    // Cross-phase seed uniqueness: collect (kind, seed) for every
    // stochastic slot across every MD phase and reject duplicates.
    let mut seed_seen: std::collections::HashMap<(String, u64), ()> =
        std::collections::HashMap::new();
    for p in phases {
        let md = match p {
            PhaseKind::Md(md) => md,
            _ => continue,
        };
        if let Some(seed) = extract_slot_seed(&md.integrator) {
            let key = (md.integrator.kind.clone(), seed);
            if seed_seen.insert(key.clone(), ()).is_some() {
                return Err(ConfigError::DuplicatePhaseSeed {
                    kind: key.0,
                    seed: key.1,
                });
            }
        }
        if let Some(t) = &md.thermostat {
            if let Some(seed) = extract_slot_seed(t) {
                let key = (t.kind.clone(), seed);
                if seed_seen.insert(key.clone(), ()).is_some() {
                    return Err(ConfigError::DuplicatePhaseSeed {
                        kind: key.0,
                        seed: key.1,
                    });
                }
            }
        }
        if let Some(b) = &md.barostat {
            if let Some(seed) = extract_slot_seed(b) {
                let key = (b.kind.clone(), seed);
                if seed_seen.insert(key.clone(), ()).is_some() {
                    return Err(ConfigError::DuplicatePhaseSeed {
                        kind: key.0,
                        seed: key.1,
                    });
                }
            }
        }
    }
    Ok(())
}

// Pull the optional `seed` field out of a SlotConfig's `params`
// table. Returns `None` for slots that don't carry one (NVE
// integrators, deterministic thermostats like NHC, deterministic
// barostats like Berendsen).
fn extract_slot_seed(slot: &SlotConfig) -> Option<u64> {
    slot.params.as_table()?.get("seed")?.as_integer().map(|n| n as u64)
}

// rq-1f87880c — per-kind validation lives in each builder's
// `validate_params(&toml::Value)` method (see `integration/framework.md`).
// `Config::validate_against` looks up the right builder and dispatches.

// Used by Config::validate to enforce just the structural constraints
// of `[[constraint_types]]` that do not require registry knowledge.
fn validate_constraint_type_names(cts: &[NamedSlotConfig]) -> Result<(), ConfigError> {
    let mut seen: Vec<&str> = Vec::with_capacity(cts.len());
    for (i, ct) in cts.iter().enumerate() {
        if ct.name.is_empty() {
            return Err(invalid(
                format!("constraint_types[{i}].name"),
                "name must not be empty",
            ));
        }
        if seen.iter().any(|n| *n == ct.name.as_str()) {
            return Err(ConfigError::DuplicateConstraintTypeName {
                name: ct.name.clone(),
            });
        }
        seen.push(ct.name.as_str());
    }
    Ok(())
}

fn validate_particle_types(pts: &[ParticleTypeConfig]) -> Result<(), ConfigError> {
    if pts.is_empty() {
        return Err(ConfigError::MissingField {
            field: "particle_types".to_string(),
        });
    }
    let mut seen: Vec<&str> = Vec::with_capacity(pts.len());
    for (i, pt) in pts.iter().enumerate() {
        if pt.name.is_empty() {
            return Err(invalid(
                format!("particle_types[{i}].name"),
                "name must not be empty",
            ));
        }
        require_finite_positive(&format!("particle_types[{i}].mass"), pt.mass)?;
        require_finite(&format!("particle_types[{i}].charge"), pt.charge)?;
        if seen.iter().any(|n| *n == pt.name) {
            return Err(ConfigError::DuplicateTypeName {
                name: pt.name.clone(),
            });
        }
        seen.push(&pt.name);
    }
    Ok(())
}

fn validate_pair_interactions(
    pis: &[PairInteractionConfig],
    pts: &[ParticleTypeConfig],
) -> Result<(), ConfigError> {
    if pis.is_empty() {
        // No pair interactions at all — surfaces as a missing pair for the
        // first declared type pair. (Pair-coverage check below covers the
        // case where the array is non-empty but incomplete.)
        return Err(ConfigError::MissingPairInteraction {
            types: (pts[0].name.clone(), pts[0].name.clone()),
        });
    }
    for (i, p) in pis.iter().enumerate() {
        require_finite_positive(&format!("pair_interactions[{i}].cutoff"), p.cutoff)?;
        require_finite_positive(&format!("pair_interactions[{i}].r_switch"), p.r_switch)?;
        if p.r_switch > p.cutoff {
            return Err(invalid(
                format!("pair_interactions[{i}].r_switch"),
                format!(
                    "r_switch ({}) exceeds cutoff ({})",
                    p.r_switch, p.cutoff
                ),
            ));
        }
        // 1: every name in `between` refers to a declared type.
        for name in [&p.between.0, &p.between.1] {
            if !pts.iter().any(|t| t.name == *name) {
                return Err(ConfigError::UnknownTypeInPair {
                    name: name.clone(),
                    pair_index: i,
                });
            }
        }
        match &p.potential {
            PairPotentialParams::LennardJones { sigma, epsilon } => {
                require_finite_positive(&format!("pair_interactions[{i}].sigma"), *sigma)?;
                require_finite_positive(&format!("pair_interactions[{i}].epsilon"), *epsilon)?;
            }
        }
    }
    // Duplicate-pair check.
    for i in 0..pis.len() {
        for j in 0..i {
            if pis[i].between == pis[j].between {
                return Err(ConfigError::DuplicatePairInteraction {
                    types: pis[i].between.clone(),
                });
            }
        }
    }
    Ok(())
}

fn validate_bond_types(bts: &[BondTypeConfig]) -> Result<(), ConfigError> {
    let mut seen: Vec<&str> = Vec::with_capacity(bts.len());
    for (i, bt) in bts.iter().enumerate() {
        match bt {
            BondTypeConfig::Morse { name, de, a, re } => {
                if name.is_empty() {
                    return Err(invalid(
                        format!("bond_types[{i}].name"),
                        "name must not be empty",
                    ));
                }
                if seen.iter().any(|n| *n == name) {
                    return Err(ConfigError::DuplicateBondTypeName { name: name.clone() });
                }
                seen.push(name);
                require_finite_positive(&format!("bond_types[{i}].de"), *de)?;
                require_finite_positive(&format!("bond_types[{i}].a"), *a)?;
                require_finite_positive(&format!("bond_types[{i}].re"), *re)?;
            }
        }
    }
    Ok(())
}

fn validate_angle_types(ats: &[AngleTypeConfig]) -> Result<(), ConfigError> {
    let mut seen: Vec<&str> = Vec::with_capacity(ats.len());
    for (i, at) in ats.iter().enumerate() {
        match at {
            AngleTypeConfig::Harmonic {
                name,
                k_theta,
                theta_0,
            } => {
                if name.is_empty() {
                    return Err(invalid(
                        format!("angle_types[{i}].name"),
                        "name must not be empty",
                    ));
                }
                if seen.iter().any(|n| *n == name) {
                    return Err(ConfigError::DuplicateAngleTypeName { name: name.clone() });
                }
                seen.push(name);
                require_finite_positive(&format!("angle_types[{i}].k_theta"), *k_theta)?;
                if !theta_0.is_finite()
                    || !(0.0..=std::f64::consts::PI).contains(theta_0)
                {
                    return Err(invalid(
                        format!("angle_types[{i}].theta_0"),
                        "theta_0 must be finite and in [0, π]",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_coulomb(c: &CoulombConfig) -> Result<(), ConfigError> {
    require_finite_positive("coulomb.cutoff", c.cutoff)?;
    require_finite_positive("coulomb.r_switch", c.r_switch)?;
    if c.r_switch > c.cutoff {
        return Err(invalid(
            "coulomb.r_switch",
            format!(
                "r_switch ({}) exceeds cutoff ({})",
                c.r_switch, c.cutoff
            ),
        ));
    }
    Ok(())
}

fn validate_spme(s: &SpmeConfig) -> Result<(), ConfigError> {
    require_finite_positive("spme.alpha", s.alpha)?;
    require_finite_positive("spme.r_cut_real", s.r_cut_real)?;
    let required = 2 * s.spline_order;
    let axes = ["a", "b", "c"];
    for (d, n) in s.grid.iter().enumerate() {
        if *n < required {
            return Err(invalid(
                format!("spme.grid[{d}]"),
                format!("grid[{}] = {n} must be >= 2 * spline_order = {required}", axes[d]),
            ));
        }
    }
    if !matches!(s.spline_order, 4 | 5 | 6 | 7 | 8) {
        return Err(invalid(
            "spme.spline_order",
            "spline_order must be one of 4, 5, 6, 7, 8",
        ));
    }
    Ok(())
}

fn validate_neighbor_list(n: &NeighborListConfig) -> Result<(), ConfigError> {
    match n {
        NeighborListConfig::AllPairs => Ok(()),
        NeighborListConfig::CellList {
            max_neighbors,
            r_skin,
        } => {
            if *max_neighbors == 0 {
                return Err(invalid(
                    "neighbor_list.max_neighbors",
                    "max_neighbors must be strictly positive",
                ));
            }
            require_finite_positive("neighbor_list.r_skin", *r_skin)?;
            Ok(())
        }
    }
}

fn check_pair_coverage(
    pts: &[ParticleTypeConfig],
    pis: &[PairInteractionConfig],
) -> Result<(), ConfigError> {
    for i in 0..pts.len() {
        for j in i..pts.len() {
            let key = normalise_pair(&pts[i].name, &pts[j].name);
            if !pis.iter().any(|p| p.between == key) {
                return Err(ConfigError::MissingPairInteraction { types: key });
            }
        }
    }
    Ok(())
}

fn check_path_collisions(config: &Config) -> Result<(), ConfigError> {
    let mut entries: Vec<(PathRole, PathBuf)> =
        Vec::with_capacity(2 + 3 * config.phases.len());
    entries.push((PathRole::Init, config.init.clone()));
    if let Some(p) = config.topology.as_deref() {
        entries.push((PathRole::Topology, p.to_path_buf()));
    }
    for phase in &config.phases {
        match phase {
            PhaseKind::Md(p) => {
                entries.push((
                    PathRole::PhaseTrajectory {
                        phase: p.name.clone(),
                    },
                    p.output.trajectory_path.clone(),
                ));
                entries.push((
                    PathRole::PhaseLog {
                        phase: p.name.clone(),
                    },
                    p.output.log_path.clone(),
                ));
                entries.push((
                    PathRole::PhaseTimings {
                        phase: p.name.clone(),
                    },
                    p.output.timings_path.clone(),
                ));
            }
            PhaseKind::Minimization(m) => {
                entries.push((
                    PathRole::MinimizationMinlog {
                        phase: m.name.clone(),
                    },
                    m.output.minlog_path.clone(),
                ));
                entries.push((
                    PathRole::MinimizationTrajectory {
                        phase: m.name.clone(),
                    },
                    m.output.trajectory_path.clone(),
                ));
                entries.push((
                    PathRole::MinimizationTimings {
                        phase: m.name.clone(),
                    },
                    m.output.timings_path.clone(),
                ));
            }
        }
    }

    for i in 0..entries.len() {
        for j in (i + 1)..entries.len() {
            if entries[i].1 == entries[j].1 {
                return Err(ConfigError::PathCollision {
                    kind_a: entries[i].0.clone(),
                    kind_b: entries[j].0.clone(),
                    path: entries[i].1.clone(),
                });
            }
        }
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
