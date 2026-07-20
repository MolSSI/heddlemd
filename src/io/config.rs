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

use crate::units::UnitSystem;

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
    // rq-9244aae4 — a pair with no override that cannot be combined.
    #[error(
        "type pair (`{}`, `{}`) has no [[pair_interactions]] override and cannot be combined: \
         supply an override, or add a [lennard_jones] table and give both types sigma/epsilon",
        types.0, types.1
    )]
    UnresolvedPairInteraction { types: (String, String) },
    #[error("duplicate pair interaction for type pair (`{}`, `{}`)", types.0, types.1)]
    DuplicatePairInteraction { types: (String, String) },
    // rq-be18633a
    #[error("unknown `[lennard_jones]` combining_rule `{got}`: expected `lorentz-berthelot`")]
    UnknownCombiningRule { got: String },
    #[error("output paths collide: `{kind_a}` and `{kind_b}` both resolve to `{}`", path.display())]
    PathCollision {
        kind_a: PathRole,
        kind_b: PathRole,
        path: PathBuf,
    },
    // rq-coulomb-retired
    #[error("the pairwise `[coulomb]` slot has been retired; use `[spme]` for electrostatics")]
    CoulombRetired,
    // rq-adddaf1a
    #[error(
        "[neighbor_list] mode = \"all-pairs\" is incompatible with a [spme] table: SPME runs \
         through the cell-list neighbour pipeline only. Use mode = \"cell-list\"."
    )]
    AllPairsWithSpme,
    // rq-lossless_unsupported_in_f64
    #[error(
        "[integrator] lossless = true is not available in the f64 build (the velocity-Verlet \
         compensated f64 low-part has no meaning when storage is already double precision); \
         rebuild without --features f64 to use lossless mode, or set lossless = false"
    )]
    LosslessUnsupportedInF64Build,
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
    #[error("duplicate dihedral type name `{name}`")]
    DuplicateDihedralTypeName { name: String },
    #[error("duplicate constraint type name `{name}`")]
    DuplicateConstraintTypeName { name: String },
    #[error("integrator `{integrator}` in phase `{phase}` does not support holonomic constraints; remove the topology file's [constraints] section or choose a different integrator")]
    IncompatibleConstraint { integrator: String, phase: String },
    #[error("constraint type `{name}` is malformed: {reason}")]
    ShakeParamsMalformed { name: String, reason: String },
    #[error("settle constraint type `{name}` is malformed: {reason}")]
    SettleParamsMalformed { name: String, reason: String },
    #[error("[{slot}] section's `kind = \"{kind}\"` does not match any registered builder")]
    UnknownKind { slot: &'static str, kind: String },
    #[error("unknown `units` value `{got}`: expected one of `si`, `atomic`")]
    UnknownUnits { got: String },
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
    /// Number of step replays between displacement-flag downloads and
    /// output-cadence re-evaluations when an MD phase runs under CUDA
    /// graph mode. Default 50. Must be `>= 1`. The displacement-check
    /// kernel runs every step inside the captured graph regardless of
    /// this value; raising the batch size lowers the per-batch
    /// flag-download rate without changing the per-step displacement
    /// bookkeeping. See `cuda-graphs.md`.
    #[serde(default = "default_graph_batch_size")]
    pub graph_batch_size: u32,
    /// When `true`, every MD phase runs the per-step launch loop with
    /// full per-kernel `Timings`. Default `false`: graph mode is the
    /// default for every eligible phase. Provided as a diagnostic
    /// escape hatch for graph-related issues. See `cuda-graphs.md`.
    #[serde(default)]
    pub cuda_graphs_disable: bool,
    /// When `true`, JIT-compiled CUDA kernels are built with
    /// `--use_fast_math`. Default `true`. Fast-math is bit-reproducible
    /// run-to-run on a fixed GPU (the load-bearing reproducibility
    /// invariant still holds); it trades a few ULP of
    /// transcendental/division precision — within the engine's f32 error
    /// class — for faster pair-force evaluation. Set `false` to compile
    /// the precise-IEEE kernels instead. See `precision.md`.
    #[serde(default = "default_fast_math")]
    pub fast_math: bool,
}

// rq-a84e1c76
fn default_fast_math() -> bool {
    true
}

fn default_graph_batch_size() -> u32 {
    50
}

// rq-18441e33 — parsed `[[phase]]` entry. The runner walks
// `Config::phases` in declaration order; particle state carries
// across phase boundaries while slot state is rebuilt at each one.
// rq-f1c04d3b
#[derive(Debug, Clone)]
pub struct PhaseConfig {
    pub name: String,
    pub n_steps: u64,
    pub dt: f64,
    pub integrator: SlotConfig,
    pub thermostat: Option<SlotConfig>,
    pub barostat: Option<SlotConfig>,
    pub output: OutputConfig,
    // rq-ee10237d — kind-agnostic thermostat coupling cadence: the
    // thermostat couples every `coupling_interval` steps (on steps where
    // `step % coupling_interval == 0`), with the effective coupling
    // timestep `coupling_interval * dt`. Peeled from `[phase.thermostat]`
    // before the kind's builder validates its params;
    // `DEFAULT_COUPLING_INTERVAL` when the section omits it or has no
    // thermostat.
    pub coupling_interval: u32,
}

// rq-732daa1b — the kind-agnostic coupling cadence applied when a
// thermostat section omits `coupling_interval`. A stochastic global
// thermostat (CSVR) samples the canonical ensemble regardless of
// cadence, so coupling every step is unnecessary; coupling every 25
// steps amortizes the full-step kinetic-energy reduction and rescale
// over the intervening steps and keeps those steps on the forces-only
// graph variant.
pub const DEFAULT_COUPLING_INTERVAL: u32 = 25;

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

// rq-19226daf rq-4a0c5f2e
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

impl crate::units::Convert for SlotConfig {
    fn from_user(&mut self, _u: crate::units::UnitSystem) {}
    fn to_user(&mut self, _u: crate::units::UnitSystem) {}
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

impl crate::units::Convert for NamedSlotConfig {
    fn from_user(&mut self, _u: crate::units::UnitSystem) {}
    fn to_user(&mut self, _u: crate::units::UnitSystem) {}
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
    /// Per-type LJ zero-crossing distance (metres); `Some` iff `epsilon`
    /// is `Some`. Used by the combining rule for pairs without an override.
    #[serde(default)]
    pub sigma: Option<f64>,
    /// Per-type LJ well depth (joules); `Some` iff `sigma` is `Some`.
    #[serde(default)]
    pub epsilon: Option<f64>,
    #[serde(default)]
    pub charge: f64,
}

// rq-be18633a — enum selecting the Lennard-Jones combining rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombiningRule {
    // rq-f5b943c1 — sigma_ij = (sigma_i + sigma_j) / 2,
    // epsilon_ij = sqrt(epsilon_i * epsilon_j).
    LorentzBerthelot,
}

// rq-be18633a — parsed `[lennard_jones]` table: the combining rule and
// the cutoff / switching radius applied to every combined pair.
#[derive(Debug, Clone)]
pub struct LennardJonesConfig {
    pub combining_rule: CombiningRule,
    pub cutoff: f64,
    pub r_switch: f64,
}

// rq-f001eaf8
/// Open-shaped parsed entry for the `[[pair_interactions]]` array.
/// `between`, `kind`, and `cutoff` are the centrally-parsed common
/// fields; every other field is captured into `params` and owned by
/// the potential builder claiming `(PairInteraction, kind)` (see
/// `rqm/forces/framework.md`, *Potential Params Claims*). `cutoff` is
/// common because the shared neighbour list's search radius and the
/// box-compatibility check read it without consulting any builder.
#[derive(Debug, Clone)]
pub struct PairInteractionConfig {
    /// Normalised so the lexicographically smaller name comes first.
    pub between: (String, String),
    pub kind: String,
    pub cutoff: f64,
    pub params: toml::Value,
}

impl PairInteractionConfig {
    /// Convenience constructor for a `kind = "lennard-jones"` entry
    /// whose params are already in atomic units (tests, embedders).
    /// `r_switch = None` leaves the build-time `0.9 * cutoff` default
    /// in effect.
    pub fn lennard_jones(
        between: (impl Into<String>, impl Into<String>),
        sigma: f64,
        epsilon: f64,
        cutoff: f64,
        r_switch: Option<f64>,
    ) -> Self {
        let mut table = toml::Table::new();
        table.insert("sigma".into(), toml::Value::Float(sigma));
        table.insert("epsilon".into(), toml::Value::Float(epsilon));
        if let Some(rs) = r_switch {
            table.insert("r_switch".into(), toml::Value::Float(rs));
        }
        let (a, b) = (between.0.into(), between.1.into());
        PairInteractionConfig {
            between: normalise_pair(&a, &b),
            kind: "lennard-jones".to_string(),
            cutoff,
            params: toml::Value::Table(table),
        }
    }
}

// rq-2f230ccb
/// Open-shaped parsed entry for the `[[bond_types]]` array. `name` is
/// the lookup key referenced from the `.topology` file's `[bonds]`
/// section; `params` is owned by the potential builder claiming
/// `(BondType, kind)`.
#[derive(Debug, Clone)]
pub struct BondTypeConfig {
    pub name: String,
    pub kind: String,
    pub params: toml::Value,
}

impl BondTypeConfig {
    /// Convenience constructor for a `kind = "morse"` entry whose
    /// params are already in atomic units (tests, embedders).
    pub fn morse(name: impl Into<String>, de: f64, a: f64, re: f64) -> Self {
        let mut table = toml::Table::new();
        table.insert("de".into(), toml::Value::Float(de));
        table.insert("a".into(), toml::Value::Float(a));
        table.insert("re".into(), toml::Value::Float(re));
        BondTypeConfig {
            name: name.into(),
            kind: "morse".to_string(),
            params: toml::Value::Table(table),
        }
    }

    /// Convenience constructor for a `kind = "harmonic"` entry whose
    /// params are already in atomic units (tests, embedders).
    pub fn harmonic(name: impl Into<String>, k: f64, r0: f64) -> Self {
        let mut table = toml::Table::new();
        table.insert("k".into(), toml::Value::Float(k));
        table.insert("r0".into(), toml::Value::Float(r0));
        BondTypeConfig {
            name: name.into(),
            kind: "harmonic".to_string(),
            params: toml::Value::Table(table),
        }
    }
}

// rq-a47beb76
/// Open-shaped parsed entry for the `[[angle_types]]` array. Same
/// shape as `BondTypeConfig`; `name` is referenced from the
/// `.topology` file's `[angles]` section.
#[derive(Debug, Clone)]
pub struct AngleTypeConfig {
    pub name: String,
    pub kind: String,
    pub params: toml::Value,
}

impl AngleTypeConfig {
    /// Convenience constructor for a `kind = "harmonic"` entry whose
    /// params are already in atomic units (tests, embedders).
    pub fn harmonic(name: impl Into<String>, k_theta: f64, theta_0: f64) -> Self {
        let mut table = toml::Table::new();
        table.insert("k_theta".into(), toml::Value::Float(k_theta));
        table.insert("theta_0".into(), toml::Value::Float(theta_0));
        AngleTypeConfig {
            name: name.into(),
            kind: "harmonic".to_string(),
            params: toml::Value::Table(table),
        }
    }
}

// rq-edfc2b75
/// Open-shaped parsed entry for the `[[dihedral_types]]` array. The
/// 1-4 exclusion scale factors are common fields (not per-kind
/// `params`) because the implicit 1-4 exclusion derivation in
/// `topology.rs` is a property of the dihedral type independent of the
/// functional form.
#[derive(Debug, Clone)]
pub struct DihedralTypeConfig {
    pub name: String,
    pub kind: String,
    /// Lennard-Jones scale applied to the implicit 1-4 exclusion
    /// derived from any `[dihedrals]` row of this type. Default
    /// `0.5` (AMBER convention).
    pub scale_lj_14: f64,
    /// Coulomb scale applied to the same implicit 1-4 exclusion.
    /// Default `1.0 / 1.2 ≈ 0.83333` (AMBER convention).
    pub scale_coul_14: f64,
    pub params: toml::Value,
}

impl DihedralTypeConfig {
    /// Convenience constructor for a `kind = "periodic"` entry whose
    /// params are already in atomic units (tests, embedders), with the
    /// AMBER-default 1-4 scale factors.
    pub fn periodic(name: impl Into<String>, k_phi: f64, n: u32, phi_0: f64) -> Self {
        Self::periodic_with_scales(
            name,
            k_phi,
            n,
            phi_0,
            default_scale_lj_14(),
            default_scale_coul_14(),
        )
    }

    /// `periodic` with explicit 1-4 scale factors.
    pub fn periodic_with_scales(
        name: impl Into<String>,
        k_phi: f64,
        n: u32,
        phi_0: f64,
        scale_lj_14: f64,
        scale_coul_14: f64,
    ) -> Self {
        let mut table = toml::Table::new();
        table.insert("k_phi".into(), toml::Value::Float(k_phi));
        table.insert("n".into(), toml::Value::Integer(n as i64));
        table.insert("phi_0".into(), toml::Value::Float(phi_0));
        DihedralTypeConfig {
            name: name.into(),
            kind: "periodic".to_string(),
            scale_lj_14,
            scale_coul_14,
            params: toml::Value::Table(table),
        }
    }
}

// rq-060b1fab rq-a8320030
#[derive(Debug, Clone, PartialEq)]
pub enum NeighborListConfig {
    AllPairs,
    CellList { r_skin: f64 },
}

// SpmeConfig — parsed `[spme]` table; rq-7bd2d9ca rq-202493a5
// rq-61889ff1 rq-a03de3d5
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
    /// Unit system the source TOML and the referenced `.in.xyz` file
    /// are written in. The loader converts every unit-bearing value to
    /// SI before populating this struct, so all downstream code can
    /// continue to assume SI.
    pub units: UnitSystem,
    pub init: PathBuf,
    pub topology: Option<PathBuf>,
    pub simulation: SimulationConfig,
    /// Unified phase sequence in source-document order: each entry is
    /// either a `PhaseKind::Md(PhaseConfig)` from a `[[phase]]` table
    /// or a `PhaseKind::Minimization(MinimizationConfig)` from a
    /// `[[minimization]]` table.
    pub phases: Vec<PhaseKind>,
    pub particle_types: Vec<ParticleTypeConfig>,
    /// Explicit per-type-pair LJ overrides. May be empty when every pair
    /// is resolved by combining (see `lennard_jones`).
    pub pair_interactions: Vec<PairInteractionConfig>,
    /// LJ combining rule and cutoff for pairs without an explicit
    /// override. `Some` when the `[lennard_jones]` table is present.
    pub lennard_jones: Option<LennardJonesConfig>,
    pub bond_types: Vec<BondTypeConfig>,
    pub angle_types: Vec<AngleTypeConfig>,
    pub dihedral_types: Vec<DihedralTypeConfig>,
    pub constraint_types: Vec<NamedSlotConfig>,
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

// Raw (deserialise-time) forms of the structs shared with the public
// Config: their unit-bearing fields are dimensioned newtypes converted by
// the `from_user` pass; build_config unwraps them into the f64 public form.
#[derive(Debug, Deserialize, crate::units::Convert)]
#[serde(deny_unknown_fields)]
struct RawSimulationConfig {
    seed: u64,
    temperature: crate::units::Temperature,
    #[serde(default = "default_graph_batch_size")]
    graph_batch_size: u32,
    #[serde(default)]
    cuda_graphs_disable: bool,
    #[serde(default = "default_fast_math")]
    fast_math: bool,
}

#[derive(Debug, Deserialize, crate::units::Convert)]
#[serde(deny_unknown_fields)]
struct RawParticleTypeConfig {
    name: String,
    mass: crate::units::Mass,
    #[serde(default)]
    sigma: Option<crate::units::Length>,
    #[serde(default)]
    epsilon: Option<crate::units::Energy>,
    #[serde(default)]
    charge: crate::units::Charge,
}

// rq-be18633a — raw `[lennard_jones]` table. `combining_rule` is a raw
// string parsed to `CombiningRule` in `load_config_raw`; the Length
// fields are converted to atomic units by the `from_user` pass.
#[derive(Debug, Deserialize, crate::units::Convert)]
#[serde(deny_unknown_fields)]
struct RawLennardJonesConfig {
    combining_rule: String,
    cutoff: crate::units::Length,
    #[serde(default)]
    r_switch: Option<crate::units::Length>,
}

#[derive(Debug, Deserialize, crate::units::Convert)]
#[serde(deny_unknown_fields)]
struct RawSpmeConfig {
    alpha: crate::units::InverseLength,
    r_cut_real: crate::units::Length,
    grid: [u32; 3],
    #[serde(default = "default_spline_order")]
    spline_order: u32,
}

#[derive(Debug, Deserialize, crate::units::Convert)]
struct RawConfig {
    schema_version: u64,
    /// Optional `units` selector. Accepts the strings `"si"` (default)
    /// or `"atomic"`. Translated to `UnitSystem` in `build_config`.
    #[serde(default)]
    units: Option<String>,
    init: String,
    #[serde(default)]
    topology: Option<String>,
    simulation: RawSimulationConfig,
    #[serde(default, rename = "phase")]
    phases: Vec<toml::Spanned<RawPhaseConfig>>,
    #[serde(default, rename = "minimization")]
    minimizations: Vec<toml::Spanned<RawMinimizationConfig>>,
    particle_types: Vec<RawParticleTypeConfig>,
    #[serde(default)]
    pair_interactions: Vec<RawPairInteraction>,
    #[serde(default)]
    lennard_jones: Option<RawLennardJonesConfig>,
    #[serde(default)]
    bond_types: Vec<RawBondType>,
    #[serde(default)]
    angle_types: Vec<RawAngleType>,
    #[serde(default)]
    dihedral_types: Vec<RawDihedralType>,
    #[serde(default)]
    constraint_types: Vec<NamedSlotConfig>,
    #[serde(default)]
    spme: Option<RawSpmeConfig>,
    #[serde(default)]
    neighbor_list: Option<RawNeighborList>,
}

#[derive(Debug, Deserialize, crate::units::Convert)]
#[serde(deny_unknown_fields)]
struct RawMinimizationConfig {
    name: String,
    algorithm: SlotConfig,
    #[serde(default)]
    output: Option<RawMinimizationOutputConfig>,
}

#[derive(Debug, Deserialize, crate::units::Convert)]
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

#[derive(Debug, Deserialize, crate::units::Convert)]
#[serde(deny_unknown_fields)]
struct RawPhaseConfig {
    name: String,
    n_steps: u64,
    dt: crate::units::Time,
    integrator: SlotConfig,
    #[serde(default)]
    thermostat: Option<SlotConfig>,
    #[serde(default)]
    barostat: Option<SlotConfig>,
    #[serde(default)]
    output: Option<RawOutputConfig>,
}

// rq-9244aae4 rq-f001eaf8
// Open-shaped raw `[[pair_interactions]]` entry: the common fields
// (`between`, `kind`, `cutoff`) are extracted here; every other field
// stays in `params` for the claiming potential builder. Only `cutoff`
// is unit-bearing among the common fields, so `Convert` is implemented
// by hand rather than derived (the derive cannot see through the
// custom `Deserialize`).
#[derive(Debug)]
struct RawPairInteraction {
    between: [String; 2],
    kind: String,
    cutoff: crate::units::Length,
    params: toml::Value,
}

impl crate::units::Convert for RawPairInteraction {
    fn from_user(&mut self, u: UnitSystem) {
        self.cutoff.from_user(u);
        // `params` is converted later by the claiming builder's
        // `convert_params` (see `convert_all_slot_params`).
    }
    fn to_user(&mut self, u: UnitSystem) {
        self.cutoff.to_user(u);
    }
}

impl<'de> Deserialize<'de> for RawPairInteraction {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let mut table = <toml::Table as Deserialize>::deserialize(d)?;
        let between: [String; 2] = take_field(&mut table, "between")?;
        let kind: String = take_field(&mut table, "kind")?;
        let cutoff: crate::units::Length = take_field(&mut table, "cutoff")?;
        Ok(RawPairInteraction {
            between,
            kind,
            cutoff,
            params: toml::Value::Table(table),
        })
    }
}

// rq-e4420955 rq-2f230ccb rq-f2946c4a rq-a47beb76
// Open-shaped raw `[[bond_types]]` / `[[angle_types]]` entry: `name`
// and `kind` are common; everything else flows to the claiming
// builder. The shape is exactly `NamedSlotConfig`'s, so both tables
// deserialise through it.
type RawBondType = NamedSlotConfig;
type RawAngleType = NamedSlotConfig;

impl From<NamedSlotConfig> for BondTypeConfig {
    fn from(r: NamedSlotConfig) -> Self {
        BondTypeConfig {
            name: r.name,
            kind: r.kind,
            params: r.params,
        }
    }
}

impl From<NamedSlotConfig> for AngleTypeConfig {
    fn from(r: NamedSlotConfig) -> Self {
        AngleTypeConfig {
            name: r.name,
            kind: r.kind,
            params: r.params,
        }
    }
}

// AMBER convention: scale_lj_14 = 0.5, scale_coul_14 = 1.0 / 1.2.
fn default_scale_lj_14() -> f64 {
    0.5
}
fn default_scale_coul_14() -> f64 {
    1.0 / 1.2
}

// rq-6d4a49ac rq-edfc2b75
// Open-shaped raw `[[dihedral_types]]` entry: `name`, `kind`, and the
// dimensionless 1-4 scale factors are common (the topology loader
// consumes the scales centrally); everything else flows to the
// claiming builder. No unit-bearing common field, so `Convert` is a
// no-op.
#[derive(Debug)]
struct RawDihedralType {
    name: String,
    kind: String,
    scale_lj_14: f64,
    scale_coul_14: f64,
    params: toml::Value,
}

impl crate::units::Convert for RawDihedralType {
    fn from_user(&mut self, _u: UnitSystem) {}
    fn to_user(&mut self, _u: UnitSystem) {}
}

impl<'de> Deserialize<'de> for RawDihedralType {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let mut table = <toml::Table as Deserialize>::deserialize(d)?;
        let name: String = take_field(&mut table, "name")?;
        let kind: String = take_field(&mut table, "kind")?;
        let scale_lj_14: f64 = take_field_or(&mut table, "scale_lj_14", default_scale_lj_14)?;
        let scale_coul_14: f64 =
            take_field_or(&mut table, "scale_coul_14", default_scale_coul_14)?;
        Ok(RawDihedralType {
            name,
            kind,
            scale_lj_14,
            scale_coul_14,
            params: toml::Value::Table(table),
        })
    }
}

impl From<RawDihedralType> for DihedralTypeConfig {
    fn from(r: RawDihedralType) -> Self {
        DihedralTypeConfig {
            name: r.name,
            kind: r.kind,
            scale_lj_14: r.scale_lj_14,
            scale_coul_14: r.scale_coul_14,
            params: r.params,
        }
    }
}

/// Extract and deserialise a required common field from an open-shaped
/// entry's table, surfacing serde's `missing field` shape so
/// `serde_error_to_config_error` routes it to `MissingField`.
fn take_field<T: serde::de::DeserializeOwned, E: serde::de::Error>(
    table: &mut toml::Table,
    key: &'static str,
) -> Result<T, E> {
    let value = table
        .remove(key)
        .ok_or_else(|| E::missing_field(key))?;
    value
        .try_into()
        .map_err(|e| E::custom(format!("field `{key}`: {e}")))
}

/// Extract and deserialise an optional common field, filling
/// `default` when the key is absent.
fn take_field_or<T: serde::de::DeserializeOwned, E: serde::de::Error>(
    table: &mut toml::Table,
    key: &'static str,
    default: fn() -> T,
) -> Result<T, E> {
    match table.remove(key) {
        None => Ok(default()),
        Some(value) => value
            .try_into()
            .map_err(|e| E::custom(format!("field `{key}`: {e}"))),
    }
}

#[derive(Debug, Deserialize, crate::units::Convert)]
#[serde(tag = "mode", rename_all = "kebab-case", deny_unknown_fields)]
enum RawNeighborList {
    // Empty-struct form (`AllPairs {}`) so `deny_unknown_fields` rejects
    // sibling fields like `max_neighbors` / `r_skin` under
    // `mode = "all-pairs"`. Unit variants in internally-tagged enums
    // skip the deny check.
    AllPairs {},
    CellList {
        #[serde(default)]
        r_skin: Option<crate::units::Length>,
    },
}

#[derive(Debug, Deserialize, crate::units::Convert)]
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

    // The pairwise `[coulomb]` slot was retired after producing
    // non-conservative forces at short intramolecular distances. Detect
    // it up front so users see a friendly retirement notice rather than
    // a silent field drop or a misleading downstream error.
    if raw_text
        .lines()
        .any(|ln| ln.trim_start().starts_with("[coulomb]"))
    {
        return Err(ConfigError::CoulombRetired);
    }

    let de = toml::Deserializer::new(&raw_text);
    let mut raw_config: RawConfig =
        serde_path_to_error::deserialize(de).map_err(serde_error_to_config_error)?;

    if raw_config.schema_version != SUPPORTED_SCHEMA_VERSION {
        return Err(ConfigError::UnsupportedSchemaVersion {
            actual: raw_config.schema_version,
            supported: SUPPORTED_SCHEMA_VERSION,
        });
    }

    // Resolve the optional `units` selector. Default to SI; reject
    // anything else with `UnknownUnits` so the user gets a precise
    // pointer at the offending value.
    let units = match raw_config.units.as_deref() {
        None | Some("si") => UnitSystem::Si,
        Some("atomic") => UnitSystem::Atomic,
        Some(other) => {
            return Err(ConfigError::UnknownUnits {
                got: other.to_string(),
            });
        }
    };

    let base_dir = path.parent().unwrap_or(Path::new("."));
    // rq-bf5df23e — one recursive Convert pass rescales every typed
    // unit-bearing field to atomic units before structural assembly.
    {
        use crate::units::Convert;
        raw_config.from_user(units);
    }
    // rq-be18633a — parse the `[lennard_jones]` table (its Length fields
    // are now atomic). The `combining_rule` string is the one fallible
    // string→enum parse, mirroring `units`; an unknown value is rejected
    // here so `build_config` can stay infallible.
    let lennard_jones = match raw_config.lennard_jones.take() {
        None => None,
        Some(raw_lj) => {
            let combining_rule = match raw_lj.combining_rule.as_str() {
                "lorentz-berthelot" => CombiningRule::LorentzBerthelot,
                other => {
                    return Err(ConfigError::UnknownCombiningRule {
                        got: other.to_string(),
                    });
                }
            };
            let cutoff = raw_lj.cutoff.0;
            let r_switch = raw_lj.r_switch.map(|x| x.0).unwrap_or(0.9 * cutoff);
            Some(LennardJonesConfig {
                combining_rule,
                cutoff,
                r_switch,
            })
        }
    };
    let mut config = build_config(raw_config, path, base_dir, units, lennard_jones);
    // rq-ee10237d — peel the kind-agnostic `coupling_interval` off each
    // `[phase.thermostat]` before the kind's builder (which uses
    // `deny_unknown_fields`) ever sees the params, and before
    // `convert_all_slot_params` runs its per-kind unit conversion.
    extract_coupling_intervals(&mut config)?;
    // Open-shaped slot params are converted to atomic units by the owning
    // builder's `convert_params`. The built-in registries supply those
    // builders; an unknown kind is left untouched and rejected later by
    // `validate_against`. rq-0f6b7b7a
    convert_all_slot_params(&mut config)?;
    config.validate()?;
    Ok(config)
}

// rq-0f6b7b7a — drive each open-shaped slot's unit conversion through the
// rq-ee10237d — remove the kind-agnostic `coupling_interval` from each MD
// phase's `[phase.thermostat]` params table and record it on the phase, so
// the thermostat's kind-specific builder never sees the field (its params
// struct is `deny_unknown_fields`) and so the value is validated once, in
// one place. A phase with no thermostat, or a thermostat section that omits
// the field, keeps the default of `1` (couple every step).
fn extract_coupling_intervals(config: &mut Config) -> Result<(), ConfigError> {
    for phase in &mut config.phases {
        if let PhaseKind::Md(md) = phase {
            if let Some(t) = md.thermostat.as_mut() {
                md.coupling_interval = take_coupling_interval(&mut t.params)?;
            }
        }
    }
    Ok(())
}

// rq-ee10237d rq-10b0c99c — extract and validate `coupling_interval` from a
// thermostat slot's params table, removing it. Must be a positive integer;
// `0`, negatives, non-integers, and values exceeding `u32` are rejected.
fn take_coupling_interval(params: &mut toml::Value) -> Result<u32, ConfigError> {
    let Some(table) = params.as_table_mut() else {
        return Ok(DEFAULT_COUPLING_INTERVAL);
    };
    let Some(value) = table.remove("coupling_interval") else {
        return Ok(DEFAULT_COUPLING_INTERVAL);
    };
    let n = value.as_integer().ok_or_else(|| {
        invalid(
            "thermostat.coupling_interval",
            "value must be an integer >= 1",
        )
    })?;
    if n < 1 || n > i64::from(u32::MAX) {
        return Err(invalid(
            "thermostat.coupling_interval",
            format!("value must be >= 1, got {n}"),
        ));
    }
    Ok(n as u32)
}

// builder that owns its kind's schema (see `KindedBuilder::convert_params`).
fn convert_all_slot_params(config: &mut Config) -> Result<(), ConfigError> {
    use crate::registry::{KindedBuilder, Registry};
    let units = config.units;
    let registries = crate::Registries::with_builtins();

    fn conv<B: KindedBuilder + ?Sized>(
        reg: &Registry<B>,
        units: UnitSystem,
        kind: &str,
        params: &mut toml::Value,
    ) -> Result<(), ConfigError> {
        match reg.lookup(kind) {
            Some(b) => b.convert_params(units, params),
            None => Ok(()),
        }
    }

    for phase in &mut config.phases {
        match phase {
            PhaseKind::Md(p) => {
                conv(&registries.integrators, units, &p.integrator.kind, &mut p.integrator.params)?;
                if let Some(t) = &mut p.thermostat {
                    conv(&registries.thermostats, units, &t.kind, &mut t.params)?;
                }
                if let Some(b) = &mut p.barostat {
                    conv(&registries.barostats, units, &b.kind, &mut b.params)?;
                }
            }
            PhaseKind::Minimization(m) => {
                conv(&registries.minimizers, units, &m.algorithm.kind, &mut m.algorithm.params)?;
            }
        }
    }
    for ct in &mut config.constraint_types {
        conv(&registries.constraint_types, units, &ct.kind, &mut ct.params)?;
    }

    // rq-73801d98 — potential-table entry params are converted by the
    // claiming potential builder. An unclaimed kind is left untouched
    // and rejected later by `validate_against`.
    use crate::forces::PotentialParamsCategory as Cat;
    fn conv_potential(
        reg: &crate::forces::PotentialRegistry,
        units: UnitSystem,
        category: Cat,
        kind: &str,
        params: &mut toml::Value,
        prefix: &str,
    ) -> Result<(), ConfigError> {
        match reg.lookup_claim(category, kind) {
            Some(b) => b
                .convert_params(units, params)
                .map_err(|e| prefix_params_error(prefix, e)),
            None => Ok(()),
        }
    }
    for (i, p) in config.pair_interactions.iter_mut().enumerate() {
        conv_potential(
            &registries.potentials,
            units,
            Cat::PairInteraction,
            &p.kind,
            &mut p.params,
            &format!("pair_interactions[{i}]"),
        )?;
    }
    for (i, bt) in config.bond_types.iter_mut().enumerate() {
        conv_potential(
            &registries.potentials,
            units,
            Cat::BondType,
            &bt.kind,
            &mut bt.params,
            &format!("bond_types[{i}]"),
        )?;
    }
    for (i, at) in config.angle_types.iter_mut().enumerate() {
        conv_potential(
            &registries.potentials,
            units,
            Cat::AngleType,
            &at.kind,
            &mut at.params,
            &format!("angle_types[{i}]"),
        )?;
    }
    for (i, dt) in config.dihedral_types.iter_mut().enumerate() {
        conv_potential(
            &registries.potentials,
            units,
            Cat::DihedralType,
            &dt.kind,
            &mut dt.params,
            &format!("dihedral_types[{i}]"),
        )?;
    }
    Ok(())
}

/// Prefix a table-and-index document path onto the relative `field` /
/// `path` carried by a builder-produced params error, so potential
/// builders can report errors relative to their entry (`"de"`) and the
/// user still sees the full location (`"bond_types[0].de"`).
// rq-73801d98
pub fn prefix_params_error(prefix: &str, e: ConfigError) -> ConfigError {
    fn join(prefix: &str, rel: &str) -> String {
        if rel.is_empty() {
            prefix.to_string()
        } else {
            format!("{prefix}.{rel}")
        }
    }
    match e {
        ConfigError::MissingField { field } => ConfigError::MissingField {
            field: join(prefix, &field),
        },
        ConfigError::InvalidValue { field, reason } => ConfigError::InvalidValue {
            field: join(prefix, &field),
            reason,
        },
        ConfigError::Parse { path, message } => ConfigError::Parse {
            path: join(prefix, &path),
            message,
        },
        other => other,
    }
}

/// Translate a `toml::de::Error` from a potential builder's typed
/// params deserialisation into a `ConfigError` with entry-relative
/// paths: `missing field \`x\`` becomes `MissingField { field: "x" }`,
/// everything else `Parse { path: "", message }`. The loader prefixes
/// the entry's document path via [`prefix_params_error`].
// rq-73801d98
pub fn translate_params_error_local(e: toml::de::Error) -> ConfigError {
    let msg = e.to_string();
    if let Some(rest) = msg.strip_prefix("missing field `") {
        if let Some(end) = rest.find('`') {
            return ConfigError::MissingField {
                field: rest[..end].to_string(),
            };
        }
    }
    ConfigError::Parse {
        path: String::new(),
        message: msg,
    }
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
//
// When `units != Si`, every unit-bearing scalar from the source is
// rescaled to SI as it flows in. Defaults that we compute here
// (`r_switch = 0.9 * cutoff`, `r_skin = 0.3 * max_cutoff`, …) inherit
// their dimension from the field they default off of, so the
// arithmetic is unit-agnostic and happens after the inputs themselves
// have already been converted.
fn build_config(
    raw: RawConfig,
    config_path: &Path,
    base_dir: &Path,
    units: UnitSystem,
    lennard_jones: Option<LennardJonesConfig>,
) -> Config {
    // Unit conversion already happened in the single `raw.from_user(units)`
    // pass before this function; every Raw unit-bearing field is in atomic
    // units, so build_config only unwraps the dimensioned newtypes (`.0`)
    // and assembles the public structs.
    let init = resolve_path(base_dir, &raw.init);
    let topology = raw.topology.as_deref().map(|s| resolve_path(base_dir, s));

    // Translate the pair_interactions raw form into the public form,
    // normalising the type-name pair. Per-kind params (including any
    // r_switch default) are owned by the claiming builder.
    let pair_interactions: Vec<PairInteractionConfig> = raw
        .pair_interactions
        .into_iter()
        .map(|r| PairInteractionConfig {
            between: normalise_pair(&r.between[0], &r.between[1]),
            kind: r.kind,
            cutoff: r.cutoff.0,
            params: r.params,
        })
        .collect();

    // The `From<Raw…>` impls unwrap the dimensioned newtypes; the values
    // are already atomic from the `from_user` pass.
    let bond_types: Vec<BondTypeConfig> =
        raw.bond_types.into_iter().map(Into::into).collect();
    let angle_types: Vec<AngleTypeConfig> =
        raw.angle_types.into_iter().map(Into::into).collect();
    let dihedral_types: Vec<DihedralTypeConfig> =
        raw.dihedral_types.into_iter().map(Into::into).collect();

    // Open-shaped slot params (constraint types here; integrator /
    // thermostat / barostat / minimizer below) are carried through in the
    // user's unit system and converted to atomic units by the owning
    // builder's `convert_params` in the post-build pass
    // (`convert_all_slot_params`), so no conversion happens here.
    let constraint_types: Vec<NamedSlotConfig> = raw.constraint_types;

    let spme = raw.spme.map(|s| SpmeConfig {
        alpha: s.alpha.0,
        r_cut_real: s.r_cut_real.0,
        grid: s.grid,
        spline_order: s.spline_order,
    });

    // Compute the maximum cutoff across pair_interaction overrides, the
    // [lennard_jones] combined cutoff, and spme.r_cut_real; used to derive
    // r_skin's default when [neighbor_list] is absent or its r_skin field
    // is omitted.
    let max_cutoff = {
        let mut m: f64 = 0.0;
        for p in &pair_interactions {
            if p.cutoff > m {
                m = p.cutoff;
            }
        }
        // rq-be18633a — combined pairs use the [lennard_jones] cutoff.
        if let Some(lj) = lennard_jones.as_ref() {
            if lj.cutoff > m {
                m = lj.cutoff;
            }
        }
        if let Some(s) = spme.as_ref() {
            if s.r_cut_real > m {
                m = s.r_cut_real;
            }
        }
        m
    };

    let neighbor_list = match raw.neighbor_list {
        None => NeighborListConfig::CellList {
            r_skin: 0.3 * max_cutoff,
        },
        Some(RawNeighborList::AllPairs {}) => NeighborListConfig::AllPairs,
        Some(RawNeighborList::CellList { r_skin }) => NeighborListConfig::CellList {
            r_skin: r_skin.map(|x| x.0).unwrap_or(0.3 * max_cutoff),
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
                let integrator = p.integrator;
                let thermostat = p.thermostat;
                let barostat = p.barostat;
                PhaseKind::Md(PhaseConfig {
                    name,
                    n_steps: p.n_steps,
                    dt: p.dt.0,
                    integrator,
                    thermostat,
                    barostat,
                    output,
                    // rq-ee10237d — default; `extract_coupling_intervals`
                    // overwrites it from the peeled `[phase.thermostat]`
                    // field before the builders see the params.
                    coupling_interval: DEFAULT_COUPLING_INTERVAL,
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
                let algorithm = m.algorithm;
                PhaseKind::Minimization(MinimizationConfig {
                    name,
                    algorithm,
                    output,
                })
            }
        })
        .collect();

    let simulation = SimulationConfig {
        seed: raw.simulation.seed,
        temperature: raw.simulation.temperature.0,
        graph_batch_size: raw.simulation.graph_batch_size,
        cuda_graphs_disable: raw.simulation.cuda_graphs_disable,
        fast_math: raw.simulation.fast_math,
    };
    let particle_types: Vec<ParticleTypeConfig> = raw
        .particle_types
        .into_iter()
        .map(|p| ParticleTypeConfig {
            name: p.name,
            mass: p.mass.0,
            sigma: p.sigma.map(|x| x.0),
            epsilon: p.epsilon.map(|x| x.0),
            charge: p.charge.0,
        })
        .collect();

    Config {
        schema_version: raw.schema_version,
        units,
        init,
        topology,
        simulation,
        phases,
        particle_types,
        pair_interactions,
        lennard_jones,
        bond_types,
        angle_types,
        dihedral_types,
        constraint_types,
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
        if let Some(lj) = &self.lennard_jones {
            validate_lennard_jones(lj)?;
        }
        // Resolvability is a pair-interaction concern; check it here, before
        // neighbor-list validation, so a config with no LJ configuration at
        // all surfaces the informative `UnresolvedPairInteraction` rather
        // than the downstream `r_skin = 0.3 * max_cutoff = 0` error.
        check_pair_coverage(
            &self.particle_types,
            &self.pair_interactions,
            self.lennard_jones.as_ref(),
        )?;
        validate_bond_types(&self.bond_types)?;
        validate_angle_types(&self.angle_types)?;
        validate_dihedral_types(&self.dihedral_types)?;
        validate_constraint_type_names(&self.constraint_types)?;
        if let Some(s) = &self.spme {
            validate_spme(s)?;
        }
        validate_neighbor_list(&self.neighbor_list)?;
        // rq-adddaf1a — SPME is evaluated only through the cell-list
        // neighbour pipeline; the all-pairs kernel does not drive the
        // SPME real-space sum, so reject the combination at load time.
        if self.spme.is_some()
            && matches!(self.neighbor_list, NeighborListConfig::AllPairs)
        {
            return Err(ConfigError::AllPairsWithSpme);
        }

        // Structural cross-validation: pair coverage and path
        // collisions. The integrator/thermostat/barostat compatibility
        // rules require builder predicates, so they live in
        // `validate_against`.
        check_path_collisions(self)?;
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
                ConfigError::ShakeParamsMalformed { name: _, reason } => {
                    ConfigError::ShakeParamsMalformed {
                        name: ct.name.clone(),
                        reason,
                    }
                }
                ConfigError::SettleParamsMalformed { name: _, reason } => {
                    ConfigError::SettleParamsMalformed {
                        name: ct.name.clone(),
                        reason,
                    }
                }
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
                    // rq-cb480c95 — integrators that cannot be
                    // combined with any external barostat (RESPA).
                    if md.barostat.is_some()
                        && !integ_builder.supports_barostat(&md.integrator.params)
                    {
                        return Err(ConfigError::InvalidValue {
                            field: "barostat.kind".to_string(),
                            reason: format!(
                                "integrator kind `{}` cannot be combined with a barostat",
                                md.integrator.kind
                            ),
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
        // rq-73801d98 — potential-table entries are global; each
        // entry's `kind` must be claimed by a registered potential
        // builder for the entry's category, and the claiming builder's
        // `validate_params` runs on the entry. Builder errors carry
        // entry-relative paths; prefix the document location here.
        {
            use crate::forces::{PotentialConfigEntry, PotentialParamsCategory as Cat};
            fn check_entry(
                registries: &crate::Registries,
                category: Cat,
                slot: &'static str,
                kind: &str,
                prefix: &str,
                entry: PotentialConfigEntry<'_>,
            ) -> Result<(), ConfigError> {
                let b = registries
                    .potentials
                    .lookup_claim(category, kind)
                    .ok_or_else(|| ConfigError::UnknownKind {
                        slot,
                        kind: kind.to_string(),
                    })?;
                b.validate_params(entry)
                    .map_err(|e| prefix_params_error(prefix, e))
            }
            for (i, p) in self.pair_interactions.iter().enumerate() {
                check_entry(
                    registries,
                    Cat::PairInteraction,
                    "pair_interactions",
                    &p.kind,
                    &format!("pair_interactions[{i}]"),
                    PotentialConfigEntry::PairInteraction(p),
                )?;
            }
            for (i, bt) in self.bond_types.iter().enumerate() {
                check_entry(
                    registries,
                    Cat::BondType,
                    "bond_types",
                    &bt.kind,
                    &format!("bond_types[{i}]"),
                    PotentialConfigEntry::BondType(bt),
                )?;
            }
            for (i, at) in self.angle_types.iter().enumerate() {
                check_entry(
                    registries,
                    Cat::AngleType,
                    "angle_types",
                    &at.kind,
                    &format!("angle_types[{i}]"),
                    PotentialConfigEntry::AngleType(at),
                )?;
            }
            for (i, dt) in self.dihedral_types.iter().enumerate() {
                check_entry(
                    registries,
                    Cat::DihedralType,
                    "dihedral_types",
                    &dt.kind,
                    &format!("dihedral_types[{i}]"),
                    PotentialConfigEntry::DihedralType(dt),
                )?;
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
    if s.graph_batch_size < 1 {
        return Err(invalid(
            "simulation.graph_batch_size",
            format!("value must be >= 1, got {}", s.graph_batch_size),
        ));
    }
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
        // rq-be18633a — per-type LJ sigma/epsilon are declared together
        // and each is finite and >= 0.
        match (pt.sigma, pt.epsilon) {
            (Some(sigma), Some(epsilon)) => {
                require_finite_non_negative(&format!("particle_types[{i}].sigma"), sigma)?;
                require_finite_non_negative(&format!("particle_types[{i}].epsilon"), epsilon)?;
            }
            (Some(_), None) => {
                return Err(invalid(
                    format!("particle_types[{i}].epsilon"),
                    "sigma is set but epsilon is missing; declare both or neither",
                ));
            }
            (None, Some(_)) => {
                return Err(invalid(
                    format!("particle_types[{i}].sigma"),
                    "epsilon is set but sigma is missing; declare both or neither",
                ));
            }
            (None, None) => {}
        }
        if seen.iter().any(|n| *n == pt.name) {
            return Err(ConfigError::DuplicateTypeName {
                name: pt.name.clone(),
            });
        }
        seen.push(&pt.name);
    }
    Ok(())
}

// rq-9244aae4 — common-field checks only: `cutoff` domain, `between`
// referring to declared types, and the same-pair-same-kind duplicate
// rule. Per-kind `params` contents are validated by the claiming
// builder during `validate_against`.
fn validate_pair_interactions(
    pis: &[PairInteractionConfig],
    pts: &[ParticleTypeConfig],
) -> Result<(), ConfigError> {
    // An empty `[[pair_interactions]]` array is valid: every pair is then
    // resolved by combining (see `check_pair_coverage`).
    for (i, p) in pis.iter().enumerate() {
        require_finite_positive(&format!("pair_interactions[{i}].cutoff"), p.cutoff)?;
        // 1: every name in `between` refers to a declared type.
        for name in [&p.between.0, &p.between.1] {
            if !pts.iter().any(|t| t.name == *name) {
                return Err(ConfigError::UnknownTypeInPair {
                    name: name.clone(),
                    pair_index: i,
                });
            }
        }
    }
    // Duplicate-pair check: at most one entry per (unordered pair, kind);
    // entries of different kinds on the same pair are additive and
    // permitted. rq-f11e9d4c
    for i in 0..pis.len() {
        for j in 0..i {
            if pis[i].between == pis[j].between && pis[i].kind == pis[j].kind {
                return Err(ConfigError::DuplicatePairInteraction {
                    types: pis[i].between.clone(),
                });
            }
        }
    }
    Ok(())
}

// rq-be18633a — the `[lennard_jones]` table's cutoff is finite and
// strictly positive, and its r_switch is finite, strictly positive, and
// `<= cutoff`. (The combining_rule string was validated at parse time.)
fn validate_lennard_jones(lj: &LennardJonesConfig) -> Result<(), ConfigError> {
    require_finite_positive("lennard_jones.cutoff", lj.cutoff)?;
    require_finite_positive("lennard_jones.r_switch", lj.r_switch)?;
    if lj.r_switch > lj.cutoff {
        return Err(invalid(
            "lennard_jones.r_switch",
            format!(
                "r_switch ({}) exceeds cutoff ({})",
                lj.r_switch, lj.cutoff
            ),
        ));
    }
    Ok(())
}

// rq-e4420955 — common-field checks only (`name` non-empty and unique);
// per-kind params are validated by the claiming builder during
// `validate_against`.
fn validate_bond_types(bts: &[BondTypeConfig]) -> Result<(), ConfigError> {
    let mut seen: Vec<&str> = Vec::with_capacity(bts.len());
    for (i, bt) in bts.iter().enumerate() {
        if bt.name.is_empty() {
            return Err(invalid(
                format!("bond_types[{i}].name"),
                "name must not be empty",
            ));
        }
        if seen.iter().any(|n| *n == bt.name) {
            return Err(ConfigError::DuplicateBondTypeName {
                name: bt.name.clone(),
            });
        }
        seen.push(&bt.name);
    }
    Ok(())
}

// rq-f2946c4a — common-field checks only; see `validate_bond_types`.
fn validate_angle_types(ats: &[AngleTypeConfig]) -> Result<(), ConfigError> {
    let mut seen: Vec<&str> = Vec::with_capacity(ats.len());
    for (i, at) in ats.iter().enumerate() {
        if at.name.is_empty() {
            return Err(invalid(
                format!("angle_types[{i}].name"),
                "name must not be empty",
            ));
        }
        if seen.iter().any(|n| *n == at.name) {
            return Err(ConfigError::DuplicateAngleTypeName {
                name: at.name.clone(),
            });
        }
        seen.push(&at.name);
    }
    Ok(())
}

// rq-6d4a49ac — common-field checks: `name` non-empty and unique, and
// the 1-4 scale factors (consumed centrally by the topology loader)
// finite and in [0, 1]. Per-kind params are validated by the claiming
// builder during `validate_against`.
fn validate_dihedral_types(dts: &[DihedralTypeConfig]) -> Result<(), ConfigError> {
    let mut seen: Vec<&str> = Vec::with_capacity(dts.len());
    for (i, dt) in dts.iter().enumerate() {
        if dt.name.is_empty() {
            return Err(invalid(
                format!("dihedral_types[{i}].name"),
                "name must not be empty",
            ));
        }
        if seen.iter().any(|x| *x == dt.name) {
            return Err(ConfigError::DuplicateDihedralTypeName {
                name: dt.name.clone(),
            });
        }
        seen.push(&dt.name);
        if !dt.scale_lj_14.is_finite() || !(0.0..=1.0).contains(&dt.scale_lj_14) {
            return Err(invalid(
                format!("dihedral_types[{i}].scale_lj_14"),
                "scale_lj_14 must be finite and in [0.0, 1.0]",
            ));
        }
        if !dt.scale_coul_14.is_finite() || !(0.0..=1.0).contains(&dt.scale_coul_14) {
            return Err(invalid(
                format!("dihedral_types[{i}].scale_coul_14"),
                "scale_coul_14 must be finite and in [0.0, 1.0]",
            ));
        }
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
        NeighborListConfig::CellList { r_skin } => {
            require_finite_positive("neighbor_list.r_skin", *r_skin)?;
            Ok(())
        }
    }
}

// rq-9244aae4 rq-be18633a — whenever the config declares any
// Lennard-Jones input (a `[lennard_jones]` table, per-type
// sigma/epsilon, or a `kind = "lennard-jones"` entry), every unordered
// type pair must resolve exactly one way: by an explicit
// `kind = "lennard-jones"` entry, or by combining the two types'
// per-type sigma/epsilon via the `[lennard_jones]` table. A pair that
// is neither overridden nor combinable is rejected. A config with no
// Lennard-Jones input at all carries no LJ slot and imposes no
// coverage requirement.
fn check_pair_coverage(
    pts: &[ParticleTypeConfig],
    pis: &[PairInteractionConfig],
    lennard_jones: Option<&LennardJonesConfig>,
) -> Result<(), ConfigError> {
    let has_lj_input = lennard_jones.is_some()
        || pts.iter().any(|t| t.sigma.is_some() || t.epsilon.is_some())
        || pis.iter().any(|p| p.kind == crate::forces::lj::LJ_KIND);
    if !has_lj_input {
        return Ok(());
    }
    let can_combine = lennard_jones.is_some();
    for i in 0..pts.len() {
        for j in i..pts.len() {
            let key = normalise_pair(&pts[i].name, &pts[j].name);
            if pis
                .iter()
                .any(|p| p.kind == crate::forces::lj::LJ_KIND && p.between == key)
            {
                continue; // resolved by an explicit LJ entry
            }
            // No LJ entry: resolvable only by combining, which requires a
            // [lennard_jones] table and sigma/epsilon on both types.
            let combinable =
                can_combine && pts[i].sigma.is_some() && pts[j].sigma.is_some();
            if !combinable {
                return Err(ConfigError::UnresolvedPairInteraction { types: key });
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
