// rq-fd8bb824
//
// The analysis framework — types, traits, registry, loader, runner,
// and lint entry points for `dynamics analyze`. Concrete analysis
// kinds live in their own submodules (currently: `rdf`).

pub mod rdf;

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Deserialize;

use crate::io::config::{Config, ConfigError, NeighborListConfig, load_config};
use crate::io::{
    TrajectoryFrame, TrajectoryFrameHeader, TrajectoryReader, TrajectoryReaderError,
};
use crate::pbc::SimulationBox;

pub use rdf::RdfBuilder;

// =====================================================================
// Public types: AnalysisConfig, AnalysisEntry, AnalyzeError, etc.
// =====================================================================

// rq-fd8bb824 rq-aa91623d
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    pub schema_version: u64,
    pub simulation: PathBuf,
    /// Name of the simulation phase whose trajectory the analyses
    /// consume. Populated from the optional `phase` field; defaults
    /// to the *last* phase in the loaded simulation config (the
    /// production phase in equilibration-then-production protocols).
    /// Empty string until `run_analyses` resolves the simulation
    /// config; see `validate_phase_against`.
    pub phase: String,
    pub trajectory: PathBuf,
    pub first_frame: u64,
    pub last_frame: Option<u64>,
    pub stride: u64,
    pub analyses: Vec<AnalysisEntry>,
    pub config_path: PathBuf,
}

// rq-ca3ec865
#[derive(Debug, Clone)]
pub struct AnalysisEntry {
    pub name: String,
    pub kind: String,
    pub output_path: PathBuf,
    pub params: toml::Value,
}

// rq-fd8bb824
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AnalysisPathRole {
    Analysis,
    Simulation,
    Trajectory,
    SimulationPhaseTrajectory { phase: String },
    SimulationPhaseLog { phase: String },
    SimulationPhaseTimings { phase: String },
    AnalysisOutput { name: String },
}

// rq-cd7d7ee5
#[derive(Debug, thiserror::Error)]
pub enum AnalyzeError {
    #[error(
        "analysis filename `{}` does not end in `.in.analysis` (or its derived root is empty)",
        path.display()
    )]
    InvalidAnalysisFilename { path: PathBuf },
    #[error("failed to read analysis file: {0}")]
    Io(String),
    #[error("analysis parse error at `{path}`: {message}")]
    Parse { path: String, message: String },
    #[error("missing required field `{field}`")]
    MissingField { field: String },
    #[error("unsupported analysis schema_version {actual}: only version {supported} is supported")]
    UnsupportedSchemaVersion { actual: u64, supported: u64 },
    #[error("invalid value for `{field}`: {reason}")]
    InvalidValue { field: String, reason: String },
    #[error("duplicate analysis name `{name}`")]
    DuplicateAnalysisName { name: String },
    #[error("analysis file declares no `[[analyses]]` entries")]
    EmptyAnalyses,
    #[error("[[analyses]] kind `{kind}` does not match any registered builder")]
    UnknownKind { kind: String },
    #[error(
        "analysis paths collide: `{kind_a:?}` and `{kind_b:?}` both resolve to `{}`",
        path.display()
    )]
    AnalyzePathCollision {
        kind_a: AnalysisPathRole,
        kind_b: AnalysisPathRole,
        path: PathBuf,
    },
    #[error("{0}")]
    Config(#[source] ConfigError),
    #[error("{0}")]
    Trajectory(#[source] TrajectoryReaderError),
    #[error("analysis `{name}`: {error}")]
    Analysis {
        name: String,
        #[source]
        error: AnalysisRuntimeError,
    },
    #[error("output file already exists: `{}`", path.display())]
    OutputExists { path: PathBuf },
    #[error("requested last_frame={requested} exceeds available frame count {available}")]
    FrameOutOfRange { requested: u64, available: u64 },
    #[error("phase `{phase}` is not declared in the loaded simulation config; available phases: {available:?}")]
    UnknownPhase { phase: String, available: Vec<String> },
}

// rq-3825d7c4
#[derive(Debug, thiserror::Error)]
pub enum AnalysisRuntimeError {
    #[error("invalid value for `{field}`: {reason}")]
    InvalidValue { field: String, reason: String },
    #[error("io error: {0}")]
    Io(String),
    #[error("{0}")]
    Other(String),
}

// rq-8914e9ff
#[derive(Debug, Clone, Copy)]
pub struct AnalyzeSummary {
    pub frames_consumed: u64,
    pub analyses_written: u64,
    pub elapsed_micros: u128,
}

// =====================================================================
// AnalysisBuilder / Analysis traits + AnalysisRegistry
// =====================================================================

// rq-86f01d20
pub trait AnalysisBuilder: std::fmt::Debug + Send + Sync {
    fn kind_name(&self) -> &'static str;

    fn validate_params(&self, params: &toml::Value) -> Result<(), AnalyzeError>;

    fn build(
        &self,
        params: &toml::Value,
        header: &TrajectoryFrameHeader,
        sim_config: &Config,
    ) -> Result<Box<dyn Analysis>, AnalysisRuntimeError>;

    fn box_clone(&self) -> Box<dyn AnalysisBuilder>;
}

// rq-8464775b
pub trait Analysis: Send {
    fn consume_frame(
        &mut self,
        frame: &TrajectoryFrame,
        sim_box: &SimulationBox,
    ) -> Result<(), AnalysisRuntimeError>;

    fn finalize_and_write(
        &mut self,
        output_path: &Path,
        sim_config: &Config,
    ) -> Result<(), AnalysisRuntimeError>;
}

// rq-e3ba8c3b
#[derive(Debug)]
pub struct AnalysisRegistry {
    pub builders: Vec<Box<dyn AnalysisBuilder>>,
}

impl Clone for AnalysisRegistry {
    fn clone(&self) -> Self {
        AnalysisRegistry {
            builders: self.builders.iter().map(|b| b.box_clone()).collect(),
        }
    }
}

impl AnalysisRegistry {
    pub fn new() -> Self {
        AnalysisRegistry { builders: Vec::new() }
    }

    pub fn with_builtins() -> Self {
        AnalysisRegistry {
            builders: vec![Box::new(RdfBuilder)],
        }
    }

    pub fn register(&mut self, builder: Box<dyn AnalysisBuilder>) {
        self.builders.push(builder);
    }

    pub fn lookup(&self, kind: &str) -> Option<&dyn AnalysisBuilder> {
        self.builders
            .iter()
            .find(|b| b.kind_name() == kind)
            .map(|b| b.as_ref())
    }
}

impl Default for AnalysisRegistry {
    fn default() -> Self {
        AnalysisRegistry::with_builtins()
    }
}

// =====================================================================
// Raw deserialisation types.
// =====================================================================

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAnalysisConfig {
    schema_version: u64,
    #[serde(default)]
    simulation: Option<String>,
    #[serde(default)]
    phase: Option<String>,
    #[serde(default)]
    trajectory: Option<String>,
    #[serde(default)]
    first_frame: Option<u64>,
    #[serde(default)]
    last_frame: Option<u64>,
    #[serde(default)]
    stride: Option<u64>,
    #[serde(default)]
    analyses: Vec<RawAnalysisEntry>,
}

#[derive(Debug, Deserialize)]
struct RawAnalysisEntry {
    name: String,
    kind: String,
    #[serde(default)]
    output_path: Option<String>,
    #[serde(flatten)]
    params: toml::Value,
}

const SUPPORTED_SCHEMA_VERSION: u64 = 1;

// =====================================================================
// derive_analysis_root + filename convention
// =====================================================================

// rq-fd8bb824
fn derive_analysis_root(path: &Path) -> Result<String, AnalyzeError> {
    let invalid = || AnalyzeError::InvalidAnalysisFilename {
        path: path.to_path_buf(),
    };
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(invalid)?;
    let without_ext = filename.strip_suffix(".in.analysis").ok_or_else(invalid)?;
    if without_ext.is_empty() {
        return Err(invalid());
    }
    Ok(without_ext.to_string())
}

fn resolve_path(base: &Path, raw: &str) -> PathBuf {
    let p = Path::new(raw);
    if p.is_absolute() { p.to_path_buf() } else { base.join(p) }
}

// =====================================================================
// load_analysis_config
// =====================================================================

// rq-9fa942b1
pub fn load_analysis_config(path: &Path) -> Result<AnalysisConfig, AnalyzeError> {
    let root = derive_analysis_root(path)?;
    let base_dir = path.parent().unwrap_or(Path::new("."));

    let raw_text = fs::read_to_string(path)
        .map_err(|e| AnalyzeError::Io(format!("{}: {}", path.display(), e)))?;

    let de = toml::Deserializer::new(&raw_text);
    let raw: RawAnalysisConfig =
        serde_path_to_error::deserialize(de).map_err(|e| {
            let dotted = e.path().to_string();
            let trimmed = dotted.trim_matches('.').to_string();
            let inner_msg = e.into_inner().to_string();
            if let Some(field) = extract_missing_field(&inner_msg) {
                let full = if trimmed.is_empty() {
                    field
                } else {
                    format!("{trimmed}.{field}")
                };
                AnalyzeError::MissingField { field: full }
            } else {
                AnalyzeError::Parse {
                    path: trimmed,
                    message: inner_msg,
                }
            }
        })?;

    if raw.schema_version != SUPPORTED_SCHEMA_VERSION {
        return Err(AnalyzeError::UnsupportedSchemaVersion {
            actual: raw.schema_version,
            supported: SUPPORTED_SCHEMA_VERSION,
        });
    }

    let first_frame = raw.first_frame.unwrap_or(0);
    let last_frame = raw.last_frame;
    let stride = raw.stride.unwrap_or(1);
    if stride == 0 {
        return Err(AnalyzeError::InvalidValue {
            field: "stride".to_string(),
            reason: "must be >= 1".to_string(),
        });
    }
    if let Some(lf) = last_frame {
        if lf < first_frame {
            return Err(AnalyzeError::InvalidValue {
                field: "last_frame".to_string(),
                reason: format!("last_frame ({lf}) must be >= first_frame ({first_frame})"),
            });
        }
    }

    if raw.analyses.is_empty() {
        return Err(AnalyzeError::EmptyAnalyses);
    }

    // Validate per-entry shape + name uniqueness.
    let mut seen_names: HashSet<String> = HashSet::new();
    let mut entries: Vec<AnalysisEntry> = Vec::with_capacity(raw.analyses.len());
    for (i, raw_entry) in raw.analyses.into_iter().enumerate() {
        if raw_entry.name.is_empty() {
            return Err(AnalyzeError::InvalidValue {
                field: format!("analyses[{i}].name"),
                reason: "must be non-empty".to_string(),
            });
        }
        if !raw_entry
            .name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(AnalyzeError::InvalidValue {
                field: format!("analyses[{i}].name"),
                reason: "must contain only ASCII letters, digits, `-`, and `_`".to_string(),
            });
        }
        if !seen_names.insert(raw_entry.name.clone()) {
            return Err(AnalyzeError::DuplicateAnalysisName {
                name: raw_entry.name,
            });
        }
        let output_path = match &raw_entry.output_path {
            Some(p) => resolve_path(base_dir, p),
            None => base_dir.join(format!("{root}.out.{name}.csv", name = raw_entry.name)),
        };
        entries.push(AnalysisEntry {
            name: raw_entry.name,
            kind: raw_entry.kind,
            output_path,
            params: raw_entry.params,
        });
    }

    let simulation_path: PathBuf = match raw.simulation.as_deref() {
        Some(s) => resolve_path(base_dir, s),
        None => base_dir.join(format!("{root}.in.toml")),
    };

    // Trajectory defaults to the loaded simulation config's resolved
    // output.trajectory_path. We do not load the simulation config here
    // (load_analysis_config is the pure-syntactic loader); the deferred
    // load happens in `validate_and_resolve_trajectory`.
    let trajectory_path: PathBuf = match raw.trajectory.as_deref() {
        Some(t) => resolve_path(base_dir, t),
        None => PathBuf::new(), // sentinel filled by load_simulation_for_analysis later
    };

    let config_path = std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf());

    // `phase` is resolved against the loaded simulation config in
    // `run_analyses` / `lint_analyses` (we don't load the sim config
    // here). The explicit value is captured verbatim; the default
    // (last phase) is filled in when the sim config is available.
    let phase = raw.phase.unwrap_or_default();

    Ok(AnalysisConfig {
        schema_version: raw.schema_version,
        simulation: simulation_path,
        phase,
        trajectory: trajectory_path,
        first_frame,
        last_frame,
        stride,
        analyses: entries,
        config_path,
    })
}

fn extract_missing_field(msg: &str) -> Option<String> {
    let needle = "missing field";
    let idx = msg.find(needle)?;
    let rest = &msg[idx + needle.len()..].trim_start();
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

// =====================================================================
// validate_against + path-collision check
// =====================================================================

impl AnalysisConfig {
    // rq-d79986d0
    pub fn validate_against(
        &self,
        registries: &crate::Registries,
    ) -> Result<(), AnalyzeError> {
        for (i, entry) in self.analyses.iter().enumerate() {
            let builder = registries.analyses.lookup(&entry.kind).ok_or_else(|| {
                AnalyzeError::UnknownKind {
                    kind: entry.kind.clone(),
                }
            })?;
            builder.validate_params(&entry.params).map_err(|e| match e {
                AnalyzeError::InvalidValue { field, reason } => AnalyzeError::InvalidValue {
                    field: format!("analyses[{i}].{field}"),
                    reason,
                },
                AnalyzeError::MissingField { field } => AnalyzeError::MissingField {
                    field: format!("analyses[{i}].{field}"),
                },
                other => other,
            })?;
        }
        Ok(())
    }
}

// Path-collision check against the loaded simulation config's resolved
// output paths plus self/trajectory/analysis-file. Run after the
// simulation config has been loaded so its OutputConfig is available.
/// Resolve `AnalysisConfig::phase` and `AnalysisConfig::trajectory`
/// against the loaded simulation config. Called once the sim config
/// is available; mutates `analysis` to fill in the resolved values.
///
/// - When `analysis.phase` is empty, default to the **last** phase
///   in the sim config.
/// - When non-empty, verify the name matches a declared phase;
///   otherwise return `UnknownPhase`.
/// - When `analysis.trajectory` is empty (the sentinel set by
///   `load_analysis_config` when no explicit `trajectory` was given),
///   fill it from the selected phase's
///   `output.trajectory_path`.
fn resolve_phase_and_trajectory(
    analysis: &mut AnalysisConfig,
    sim_config: &Config,
) -> Result<(), AnalyzeError> {
    let chosen_index: usize = if analysis.phase.is_empty() {
        // Default to the last phase.
        sim_config.phases.len().saturating_sub(1)
    } else {
        match sim_config
            .phases
            .iter()
            .position(|p| p.name() == analysis.phase)
        {
            Some(i) => i,
            None => {
                let available = sim_config
                    .phases
                    .iter()
                    .map(|p| p.name().to_string())
                    .collect();
                return Err(AnalyzeError::UnknownPhase {
                    phase: analysis.phase.clone(),
                    available,
                });
            }
        }
    };
    let chosen_phase = &sim_config.phases[chosen_index];
    analysis.phase = chosen_phase.name().to_string();
    if analysis.trajectory.as_os_str().is_empty() {
        match chosen_phase {
            crate::io::PhaseKind::Md(p) => {
                analysis.trajectory = p.output.trajectory_path.clone();
            }
            crate::io::PhaseKind::Minimization(m) => {
                analysis.trajectory = m.output.trajectory_path.clone();
            }
        }
    }
    Ok(())
}

fn check_path_collisions(
    analysis: &AnalysisConfig,
    sim_config: &Config,
) -> Result<(), AnalyzeError> {
    let mut entries: Vec<(AnalysisPathRole, PathBuf)> = vec![
        (AnalysisPathRole::Analysis, analysis.config_path.clone()),
        (AnalysisPathRole::Simulation, analysis.simulation.clone()),
        (AnalysisPathRole::Trajectory, analysis.trajectory.clone()),
    ];
    for phase in &sim_config.phases {
        match phase {
            crate::io::PhaseKind::Md(p) => {
                entries.push((
                    AnalysisPathRole::SimulationPhaseTrajectory {
                        phase: p.name.clone(),
                    },
                    p.output.trajectory_path.clone(),
                ));
                entries.push((
                    AnalysisPathRole::SimulationPhaseLog {
                        phase: p.name.clone(),
                    },
                    p.output.log_path.clone(),
                ));
                entries.push((
                    AnalysisPathRole::SimulationPhaseTimings {
                        phase: p.name.clone(),
                    },
                    p.output.timings_path.clone(),
                ));
            }
            crate::io::PhaseKind::Minimization(m) => {
                entries.push((
                    AnalysisPathRole::SimulationPhaseTrajectory {
                        phase: m.name.clone(),
                    },
                    m.output.trajectory_path.clone(),
                ));
                entries.push((
                    AnalysisPathRole::SimulationPhaseTimings {
                        phase: m.name.clone(),
                    },
                    m.output.timings_path.clone(),
                ));
            }
        }
    }
    for entry in &analysis.analyses {
        entries.push((
            AnalysisPathRole::AnalysisOutput {
                name: entry.name.clone(),
            },
            entry.output_path.clone(),
        ));
    }
    for i in 0..entries.len() {
        for j in (i + 1)..entries.len() {
            // The Trajectory entry pointing at the selected phase's
            // own trajectory is the expected case under implicit
            // pairing — skip that specific collision.
            let (a_role, a_path) = &entries[i];
            let (b_role, b_path) = &entries[j];
            let analysis_outputs = matches!(
                a_role,
                AnalysisPathRole::AnalysisOutput { .. }
            ) || matches!(
                b_role,
                AnalysisPathRole::AnalysisOutput { .. }
            );
            let benign_trajectory_pair = (matches!(a_role, AnalysisPathRole::Trajectory)
                && matches!(b_role, AnalysisPathRole::SimulationPhaseTrajectory { .. }))
                || (matches!(a_role, AnalysisPathRole::SimulationPhaseTrajectory { .. })
                    && matches!(b_role, AnalysisPathRole::Trajectory));
            if a_path == b_path {
                if benign_trajectory_pair {
                    continue;
                }
                // Two simulation-side paths colliding is a sim-config
                // problem, but it would have been caught at simulation
                // load time. Still flag any analysis output colliding
                // with anything else.
                if !analysis_outputs
                    && !matches!(a_role, AnalysisPathRole::Analysis)
                    && !matches!(b_role, AnalysisPathRole::Analysis)
                {
                    continue;
                }
                return Err(AnalyzeError::AnalyzePathCollision {
                    kind_a: a_role.clone(),
                    kind_b: b_role.clone(),
                    path: a_path.clone(),
                });
            }
        }
    }
    Ok(())
}

// =====================================================================
// run_analyses entry points
// =====================================================================

// rq-8c1de56e
pub fn run_analyses(config_path: &Path) -> Result<AnalyzeSummary, AnalyzeError> {
    let registries = crate::Registries::with_builtins();
    run_analyses_with_registries(config_path, &registries)
}

// rq-c9a3109a
pub fn run_analyses_with_registries(
    config_path: &Path,
    registries: &crate::Registries,
) -> Result<AnalyzeSummary, AnalyzeError> {
    let started = Instant::now();

    // Stage 1: analysis-config load.
    let mut analysis = load_analysis_config(config_path)?;

    // Stage 1b: simulation-config load.
    let sim_config = load_config(&analysis.simulation).map_err(AnalyzeError::Config)?;
    resolve_phase_and_trajectory(&mut analysis, &sim_config)?;

    // Stage 1c: per-kind parameter validation.
    analysis.validate_against(registries)?;

    // Path-collision check now that the simulation config is loaded.
    check_path_collisions(&analysis, &sim_config)?;

    // Stage 2: pre-flight output checks (no file is created).
    for entry in &analysis.analyses {
        if entry.output_path.exists() {
            return Err(AnalyzeError::OutputExists {
                path: entry.output_path.clone(),
            });
        }
    }

    // Stage 3: open trajectory.
    let type_names_owned: Vec<String> = sim_config
        .particle_types
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let type_name_refs: Vec<&str> = type_names_owned.iter().map(|s| s.as_str()).collect();
    let mut reader =
        TrajectoryReader::open(&analysis.trajectory, sim_config.units, &type_name_refs)
            .map_err(AnalyzeError::Trajectory)?;

    // Stage 4: construct analysis slots from the first-frame header.
    let mut slots: Vec<Box<dyn Analysis>> = Vec::with_capacity(analysis.analyses.len());
    for entry in &analysis.analyses {
        let builder = registries
            .analyses
            .lookup(&entry.kind)
            .ok_or_else(|| AnalyzeError::UnknownKind {
                kind: entry.kind.clone(),
            })?;
        let slot = builder
            .build(&entry.params, &reader.first_frame_header, &sim_config)
            .map_err(|e| AnalyzeError::Analysis {
                name: entry.name.clone(),
                error: e,
            })?;
        slots.push(slot);
    }

    // Stage 5: trajectory pass with frame selection.
    let mut frames_consumed: u64 = 0;
    let mut position_in_file: u64 = 0;
    loop {
        let maybe_frame = reader.next_frame().map_err(AnalyzeError::Trajectory)?;
        let frame = match maybe_frame {
            Some(f) => f,
            None => break,
        };
        let pos = position_in_file;
        position_in_file += 1;
        if pos < analysis.first_frame {
            continue;
        }
        if let Some(lf) = analysis.last_frame {
            if pos > lf {
                continue;
            }
        }
        let stride_offset = pos - analysis.first_frame;
        if stride_offset % analysis.stride != 0 {
            continue;
        }
        let sim_box = frame.sim_box.clone();
        for (slot, entry) in slots.iter_mut().zip(analysis.analyses.iter()) {
            slot.consume_frame(&frame, &sim_box).map_err(|e| {
                AnalyzeError::Analysis {
                    name: entry.name.clone(),
                    error: e,
                }
            })?;
        }
        frames_consumed += 1;
    }

    // If `last_frame` was set and exceeds what the file actually contained,
    // report `FrameOutOfRange`. (Frames past `last_frame` are not an error
    // because the loop above ignores them; missing frames are.)
    if let Some(lf) = analysis.last_frame {
        if lf >= position_in_file {
            return Err(AnalyzeError::FrameOutOfRange {
                requested: lf,
                available: position_in_file,
            });
        }
    }

    // Stage 6: finalise + write outputs.
    let mut written: u64 = 0;
    for (slot, entry) in slots.iter_mut().zip(analysis.analyses.iter()) {
        slot.finalize_and_write(&entry.output_path, &sim_config)
            .map_err(|e| AnalyzeError::Analysis {
                name: entry.name.clone(),
                error: e,
            })?;
        written += 1;
    }

    Ok(AnalyzeSummary {
        frames_consumed,
        analyses_written: written,
        elapsed_micros: started.elapsed().as_micros(),
    })
}

// =====================================================================
// Lint entry points (analyze lint pipeline).
// =====================================================================

use crate::runner::{LintOverall, LintReport, LintStage, LintStatus, RunnerError};

// rq-bcf7e0eb
pub fn lint_analyses(config_path: &Path) -> LintReport {
    let registries = crate::Registries::with_builtins();
    lint_analyses_with_registries(config_path, &registries)
}

// rq-6eb18608
pub fn lint_analyses_with_registries(
    config_path: &Path,
    registries: &crate::Registries,
) -> LintReport {
    let mut stages: Vec<LintStage> = Vec::with_capacity(4);

    // Stage 1: config.
    let mut analysis = match load_analysis_config(config_path) {
        Ok(a) => {
            stages.push(LintStage {
                label: "config",
                status: LintStatus::Ok {
                    detail: config_path.display().to_string(),
                },
            });
            a
        }
        Err(e) => {
            return push_lint_fail(stages, "config", &format!("{e}"), wrap_analyze(e), &["output paths", "trajectory", "analyses"]);
        }
    };
    let sim_config = match load_config(&analysis.simulation) {
        Ok(c) => c,
        Err(e) => {
            // The simulation config failed to load — fold it into the
            // config stage.
            let display = format!("{e}");
            let err = wrap_analyze(AnalyzeError::Config(e));
            // Reframe the prior `config` Ok into a Fail by replacing the
            // last stage.
            stages.pop();
            stages.push(LintStage {
                label: "config",
                status: LintStatus::Fail {
                    detail: display,
                    error: err,
                },
            });
            return finalise_lint_skips(stages, &["output paths", "trajectory", "analyses"]);
        }
    };
    if let Err(e) = resolve_phase_and_trajectory(&mut analysis, &sim_config) {
        let display = format!("{e}");
        stages.pop();
        stages.push(LintStage {
            label: "config",
            status: LintStatus::Fail {
                detail: display,
                error: wrap_analyze(e),
            },
        });
        return finalise_lint_skips(stages, &["output paths", "trajectory", "analyses"]);
    }
    if let Err(e) = check_path_collisions(&analysis, &sim_config) {
        let display = format!("{e}");
        stages.pop();
        stages.push(LintStage {
            label: "config",
            status: LintStatus::Fail {
                detail: display,
                error: wrap_analyze(e),
            },
        });
        return finalise_lint_skips(stages, &["output paths", "trajectory", "analyses"]);
    }

    // Stage 2: output paths.
    let mut output_collision: Option<PathBuf> = None;
    for entry in &analysis.analyses {
        if entry.output_path.exists() {
            output_collision = Some(entry.output_path.clone());
            break;
        }
    }
    if let Some(path) = output_collision {
        let display = format!("`{}` already exists", path.display());
        stages.push(LintStage {
            label: "output paths",
            status: LintStatus::Fail {
                detail: display,
                error: wrap_analyze(AnalyzeError::OutputExists { path }),
            },
        });
        return finalise_lint_skips(stages, &["trajectory", "analyses"]);
    }
    stages.push(LintStage {
        label: "output paths",
        status: LintStatus::Ok {
            detail: "none pre-exist".to_string(),
        },
    });

    // Stage 3: trajectory.
    let type_names_owned: Vec<String> = sim_config
        .particle_types
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let type_name_refs: Vec<&str> = type_names_owned.iter().map(|s| s.as_str()).collect();
    let reader = match TrajectoryReader::open(&analysis.trajectory, sim_config.units, &type_name_refs) {
        Ok(r) => {
            let h = &r.first_frame_header;
            stages.push(LintStage {
                label: "trajectory",
                status: LintStatus::Ok {
                    detail: format!(
                        "resolved, {} particles, box {:.1e} × {:.1e} × {:.1e} m",
                        h.particle_count,
                        h.sim_box.lx(),
                        h.sim_box.ly(),
                        h.sim_box.lz(),
                    ),
                },
            });
            r
        }
        Err(e) => {
            let display = format!("{e}");
            stages.push(LintStage {
                label: "trajectory",
                status: LintStatus::Fail {
                    detail: display,
                    error: wrap_analyze(AnalyzeError::Trajectory(e)),
                },
            });
            return finalise_lint_skips(stages, &["analyses"]);
        }
    };

    // Stage 4: analyses (validate_params + build against first frame).
    if let Err(e) = analysis.validate_against(registries) {
        let display = format!("{e}");
        stages.push(LintStage {
            label: "analyses",
            status: LintStatus::Fail {
                detail: display,
                error: wrap_analyze(e),
            },
        });
        return LintReport {
            stages,
            overall: LintOverall::Fail,
        };
    }
    for entry in &analysis.analyses {
        let builder = registries
            .analyses
            .lookup(&entry.kind)
            .expect("validate_against confirmed kind is registered");
        if let Err(e) =
            builder.build(&entry.params, &reader.first_frame_header, &sim_config)
        {
            let display = format!("{e}");
            stages.push(LintStage {
                label: "analyses",
                status: LintStatus::Fail {
                    detail: display,
                    error: wrap_analyze(AnalyzeError::Analysis {
                        name: entry.name.clone(),
                        error: e,
                    }),
                },
            });
            return LintReport {
                stages,
                overall: LintOverall::Fail,
            };
        }
    }
    stages.push(LintStage {
        label: "analyses",
        status: LintStatus::Ok {
            detail: format!("{} analysis builders validated", analysis.analyses.len()),
        },
    });

    // The neighbor-list check is irrelevant to analyses; we silence the
    // unused-warning by referencing the variant. (Pure no-op.)
    let _ = NeighborListConfig::AllPairs;

    LintReport {
        stages,
        overall: LintOverall::Ok,
    }
}

// Wrap any AnalyzeError as a RunnerError so the lint plumbing in
// runner.rs (which carries `RunnerError` payloads in `LintStatus::Fail`)
// accepts it. AnalyzeError flows through `RunnerError::Config` via the
// existing Display delegation: we synthesize a one-line wrapper here.
fn wrap_analyze(err: AnalyzeError) -> RunnerError {
    // Re-use the analyze error's Display via the shared "Other"-style
    // ConfigError variant. We pack the analyze error into a
    // ConfigError::InvalidValue carrying the rendered message; tests
    // that compare against the analyze error type should inspect the
    // Display rendering rather than the wrapped variant.
    RunnerError::Analyze(err)
}

fn push_lint_fail(
    mut stages: Vec<LintStage>,
    label: &'static str,
    detail: &str,
    error: RunnerError,
    skipped: &[&'static str],
) -> LintReport {
    stages.push(LintStage {
        label,
        status: LintStatus::Fail {
            detail: detail.to_string(),
            error,
        },
    });
    finalise_lint_skips(stages, skipped)
}

fn finalise_lint_skips(mut stages: Vec<LintStage>, skipped: &[&'static str]) -> LintReport {
    for s in skipped {
        stages.push(LintStage {
            label: s,
            status: LintStatus::Skipped {
                reason: "skipped (earlier check failed)".to_string(),
            },
        });
    }
    LintReport {
        stages,
        overall: LintOverall::Fail,
    }
}
