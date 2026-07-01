// rq-357909e4 rq-02edd314 rq-77c1d5d9
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::forces::{
    AngleList, BondList, ConstraintList, DihedralList, ExclusionList, ForceField, ForceFieldError,
    TopologyFileError,
    load_topology_file,
};
use crate::gpu::{ParticleBuffers, compute_total_potential_energy, init_device};
use crate::integrator::{
    BarostatError, ConstraintError, IntegratorError, ThermostatError,
};
use crate::io::{MinlogWriter, MinlogWriterError};
use crate::minimizer::{MinimizerConvergence, MinimizerError};
use crate::io::config::NeighborListConfig;
use crate::io::{
    ConfigError, InitState, InitStateError, InitVelocities, LogWriter, LogWriterError,
    TrajectoryWriter, TrajectoryWriterError, load_config_raw, load_init_state,
};
use crate::io::log_output::{compute_kinetic_energy, compute_temperature};
use crate::state::{ParticleState, ParticleStateError};
use crate::timings::{
    GraphVariant, HostStage, KernelStage, Timings, TimingsError, TimingsWriterError,
    write_timings_file,
};
use crate::precision::Real;

// rq-8ee27e27 rq-e1ceb5c0 rq-6cf916af
//
// `RunnerError` carries no `From` impls: the runner attaches an
// `ExitPhase` tag at each `.map_err` call site. The wrapping variants
// delegate `Display` to the inner error via `#[error("{0}")]` and expose
// it through `source()` via `#[source]`, but deliberately omit `#[from]`,
// so no implicit conversion exists.
#[derive(Debug, thiserror::Error)]
pub enum RunnerError {
    #[error("{0}")]
    Config(#[source] ConfigError),
    #[error("{0}")]
    InitState(#[source] InitStateError),
    #[error("{0}")]
    ParticleState(#[source] ParticleStateError),
    #[error("{0}")]
    Gpu(#[source] crate::gpu::GpuError),
    #[error("{0}")]
    Integrator(#[source] IntegratorError),
    #[error("{0}")]
    Thermostat(#[source] ThermostatError),
    #[error("{0}")]
    Barostat(#[source] BarostatError),
    #[error("{0}")]
    SimulationBox(#[source] crate::pbc::SimulationBoxError),
    #[error("{0}")]
    Constraint(#[source] ConstraintError),
    #[error("{0}")]
    TopologyFile(#[source] TopologyFileError),
    #[error("{0}")]
    ForceField(#[source] ForceFieldError),
    #[error("{0}")]
    Trajectory(#[source] TrajectoryWriterError),
    #[error("{0}")]
    Log(#[source] LogWriterError),
    #[error("{0}")]
    Timings(#[source] TimingsError),
    #[error("{0}")]
    TimingsWriter(#[source] TimingsWriterError),
    #[error("missing command-line arguments")]
    MissingArgs,
    #[error("output file already exists: `{}`", .path.display())]
    OutputExists { path: PathBuf },
    #[error("simulation box perpendicular width along lattice direction `{direction}` is {width}, below the required {required}")]
    CellListBoxTooSmall {
        direction: &'static str,
        width: Real,
        required: Real,
    },
    // rq-8ee27e27 rq-02f4d342
    #[error(
        "cuFFT returned non-deterministic R2C output between two identical runs ({differences} differing floats); SPME requires bit-exact reciprocal-space behaviour"
    )]
    CuFftNonDeterministic { differences: usize },
    // rq-fd8bb824 — wraps an analyze-pipeline error for surfacing in
    // the lint / CLI paths.
    #[error("{0}")]
    Analyze(#[source] crate::analysis::AnalyzeError),
    #[error("{0}")]
    Minimizer(#[source] MinimizerError),
    #[error("{0}")]
    Minlog(#[source] MinlogWriterError),
    #[error(
        "minimization phase `{phase}` failed to converge after {iterations} iterations (max_force = {final_force:.3e} N, step = {final_step:.3e} m)"
    )]
    MinimizerNonConvergence {
        phase: String,
        iterations: u64,
        final_force: f64,
        final_step: f64,
    },
    #[error(
        "built-in {kind} slot `{label}` did not expose a post-force per-particle source fragment"
    )]
    MissingPostForcePerParticleFragment {
        kind: &'static str,
        label: &'static str,
    },
    #[error("JIT-composed post-force per-particle kernel failed to compile: {log}")]
    PostForceFragmentCompileFailed { log: String },
    #[error("JIT-composed post-force per-particle kernel failed to load: {0}")]
    PostForceFragmentLoadFailed(crate::gpu::GpuError),
}

// rq-5c1cfc93 rq-b00170c6
#[derive(Debug, Clone)]
pub struct PhaseSummary {
    pub name: String,
    pub n_steps: u64,
    pub frames_written: u64,
    pub log_rows_written: u64,
    pub elapsed_micros: u128,
    /// Phase kind: "md" for `[[phase]]`, "minimization" for
    /// `[[minimization]]`. Used by the CLI summary formatter.
    pub kind: &'static str,
    /// For minimization phases: the convergence reason as a short
    /// token (`"force_tolerance"`, `"energy_tolerance"`,
    /// `"force_zero"`, `"max_iterations"`). `None` for MD phases.
    pub convergence: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub struct RunSummary {
    pub phases: Vec<PhaseSummary>,
    pub total_n_steps: u64,
    pub total_elapsed_micros: u128,
}

// rq-b1a2d006 — host-stage durations captured during one-time setup.
// Replayed into phase-0 `Timings` as static one-shot samples by
// `run_md_phase` / `run_minimization_phase`.
#[derive(Debug, Clone, Default)]
pub struct PrePhaseDurations {
    pub config_load: Duration,
    pub init_load: Duration,
    pub gpu_init: Duration,
    pub velocity_generation: Duration,
    pub upload: Duration,
}

// rq-b1a2d006 — cross-phase state owned for the duration of a run.
//
// Constructed once via `SimulationSetup::new`, then mutated by the
// per-phase functions `run_md_phase` and `run_minimization_phase`.
// Every field is `pub` so external scenario-driving binaries can
// inspect or replace pieces between calls.
#[derive(Debug)]
pub struct SimulationSetup {
    pub config: crate::io::Config,
    pub registries: crate::Registries,
    pub gpu: crate::gpu::GpuContext,
    pub buffers: ParticleBuffers,
    pub sim_box: crate::pbc::SimulationBox,
    pub force_field: ForceField,
    pub constraint_list: ConstraintList,
    pub bond_list: BondList,
    pub angle_list: AngleList,
    pub dihedral_list: DihedralList,
    pub exclusion_list: ExclusionList,
    pub masses: Vec<Real>,
    pub charges: Vec<Real>,
    pub type_indices: Vec<u32>,
    pub n_constraints: u32,
    pub n_thermal_dof: u32,
    pub pre_phase_durations: PrePhaseDurations,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExitPhase {
    Setup,
    Loop,
}

const USAGE_LINE: &str = "\
usage: heddlemd run     <config-path>
       heddlemd lint    <config-path> [--with-gpu]
       heddlemd analyze <analysis-path>";

// rq-30c21c70
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintOverall {
    Ok,
    Fail,
}

// rq-ff560c3b
#[derive(Debug)]
pub enum LintStatus {
    Ok { detail: String },
    Fail { detail: String, error: RunnerError },
    Skipped { reason: String },
    NotChecked { reason: String },
}

// rq-334f5685
#[derive(Debug)]
pub struct LintStage {
    pub label: &'static str,
    pub status: LintStatus,
}

// rq-a831fb00
#[derive(Debug)]
pub struct LintReport {
    pub stages: Vec<LintStage>,
    pub overall: LintOverall,
}

impl LintReport {
    pub fn ok(&self) -> bool {
        matches!(self.overall, LintOverall::Ok)
    }

    pub fn first_failure(&self) -> Option<&RunnerError> {
        self.stages.iter().find_map(|s| match &s.status {
            LintStatus::Fail { error, .. } => Some(error),
            _ => None,
        })
    }

    pub fn write_to(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        let header = match self.overall {
            LintOverall::Ok => "[heddlemd lint] OK",
            LintOverall::Fail => "[heddlemd lint] FAIL",
        };
        writeln!(w, "{header}")?;
        for stage in &self.stages {
            let desc = match &stage.status {
                LintStatus::Ok { detail } => detail.clone(),
                LintStatus::Fail { detail, .. } => format!("FAIL — {detail}"),
                LintStatus::Skipped { reason } => reason.clone(),
                LintStatus::NotChecked { reason } => reason.clone(),
            };
            writeln!(w, "  {label:<12} {desc}", label = stage.label, desc = desc)?;
        }
        Ok(())
    }
}

// rq-1fc57c00 rq-e5e4b048 — convenience wrapper for the built-in case.
// Equivalent to `run_simulation_with_registries(config_path,
// &Registries::with_builtins())`. Used by `main.rs` and by every
// caller that does not register custom builders.
pub fn run_simulation(config_path: &Path) -> Result<RunSummary, RunnerError> {
    let registries = crate::Registries::with_builtins();
    run_simulation_with_registries(config_path, &registries)
}

// rq-a71cef31 — entry point for callers that supply their own
// `Registries`. Dispatches every integrator / thermostat / barostat /
// constraint / potential builder lookup through `registries`.
pub fn run_simulation_with_registries(
    config_path: &Path,
    registries: &crate::Registries,
) -> Result<RunSummary, RunnerError> {
    run_simulation_with_phase(config_path, registries).map_err(|(e, _)| e)
}

// rq-4ff84310 — built-ins convenience wrapper for the lint entry point.
pub fn lint_simulation(config_path: &Path, with_gpu: bool) -> LintReport {
    let registries = crate::Registries::with_builtins();
    lint_simulation_with_registries(config_path, &registries, with_gpu)
}

// rq-9ed993de — runs every stage of the lint flow against `registries`.
// Short-circuits on the first stage that fails: that stage carries the
// structured `RunnerError`, every subsequent stage is `Skipped`.
pub fn lint_simulation_with_registries(
    config_path: &Path,
    registries: &crate::Registries,
    with_gpu: bool,
) -> LintReport {
    let mut stages: Vec<LintStage> = Vec::with_capacity(6);

    // Stage 1: config.
    let config = match load_config_raw(config_path)
        .and_then(|c| c.validate_against(registries).map(|_| c))
    {
        Ok(c) => {
            stages.push(LintStage {
                label: "config",
                status: LintStatus::Ok {
                    detail: config_path.display().to_string(),
                },
            });
            c
        }
        Err(e) => {
            stages.push(LintStage {
                label: "config",
                status: LintStatus::Fail {
                    detail: format!("{e}"),
                    error: RunnerError::Config(e),
                },
            });
            return finalize_with_skips(stages, &["output paths", "init", "box/cutoff", "topology", "gpu"]);
        }
    };

    // Stage 2: output paths. Check every enabled path across every
    // phase; report the first pre-existing one (phases in declaration
    // order).
    let mut output_collision: Option<PathBuf> = None;
    'outer: for phase in &config.phases {
        match phase {
            crate::io::PhaseKind::Md(p) => {
                if p.output.trajectory_every > 0 && p.output.trajectory_path.exists() {
                    output_collision = Some(p.output.trajectory_path.clone());
                    break 'outer;
                }
                if p.output.log_every > 0 && p.output.log_path.exists() {
                    output_collision = Some(p.output.log_path.clone());
                    break 'outer;
                }
                if p.output.timings_path.exists() {
                    output_collision = Some(p.output.timings_path.clone());
                    break 'outer;
                }
            }
            crate::io::PhaseKind::Minimization(m) => {
                if m.output.minlog_every > 0 && m.output.minlog_path.exists() {
                    output_collision = Some(m.output.minlog_path.clone());
                    break 'outer;
                }
                if m.output.trajectory_every > 0 && m.output.trajectory_path.exists() {
                    output_collision = Some(m.output.trajectory_path.clone());
                    break 'outer;
                }
                if m.output.timings_path.exists() {
                    output_collision = Some(m.output.timings_path.clone());
                    break 'outer;
                }
            }
        }
    }
    if let Some(path) = output_collision {
        let detail = format!("`{}` already exists", path.display());
        stages.push(LintStage {
            label: "output paths",
            status: LintStatus::Fail {
                detail,
                error: RunnerError::OutputExists { path },
            },
        });
        return finalize_with_skips(stages, &["init", "box/cutoff", "topology", "gpu"]);
    } else {
        stages.push(LintStage {
            label: "output paths",
            status: LintStatus::Ok {
                detail: "none pre-exist".to_string(),
            },
        });
    }

    // Stage 3: init. Lint always initialises the GPU here because
    // `SimulationBox` is device-resident; the `--with-gpu` flag now
    // only gates the heavier GPU stages (cuFFT smoke test, ForceField
    // allocation).
    let type_name_strings: Vec<String> = config
        .particle_types
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let type_name_refs: Vec<&str> = type_name_strings.iter().map(|s| s.as_str()).collect();
    let lint_gpu = match init_device() {
        Ok(g) => g,
        Err(e) => {
            stages.push(LintStage {
                label: "init",
                status: LintStatus::Fail {
                    detail: format!("init_device failed: {e}"),
                    error: RunnerError::Gpu(e),
                },
            });
            return finalize_with_skips(stages, &["box/cutoff", "topology", "gpu"]);
        }
    };
    let init = match load_init_state(&lint_gpu.device, &config.init, &type_name_refs, config.units) {
        Ok(i) => {
            stages.push(LintStage {
                label: "init",
                status: LintStatus::Ok {
                    detail: format!(
                        "resolved, {} particles, box {:.1e} × {:.1e} × {:.1e} m",
                        i.particle_count,
                        i.sim_box.lx(),
                        i.sim_box.ly(),
                        i.sim_box.lz(),
                    ),
                },
            });
            i
        }
        Err(e) => {
            stages.push(LintStage {
                label: "init",
                status: LintStatus::Fail {
                    detail: format!("{e}"),
                    error: RunnerError::InitState(e),
                },
            });
            return finalize_with_skips(stages, &["box/cutoff", "topology", "gpu"]);
        }
    };

    let sim_box = init.sim_box.clone();
    let n = init.particle_count;

    // Stage 4: box/cutoff.
    match &config.neighbor_list {
        NeighborListConfig::AllPairs => {
            stages.push(LintStage {
                label: "box/cutoff",
                status: LintStatus::Skipped {
                    reason: "not applicable (mode = all-pairs)".to_string(),
                },
            });
        }
        NeighborListConfig::CellList { r_skin, .. } => {
            let cutoff_max = compute_cutoff_max(&config);
            let required = (3.0 * (cutoff_max + r_skin)) as Real;
            match sim_box.check_min_perpendicular_width(required) {
                Ok(()) => {
                    stages.push(LintStage {
                        label: "box/cutoff",
                        status: LintStatus::Ok {
                            detail: format!(
                                "min perp width {:.2e} m ≥ required {:.2e} m",
                                sim_box.min_perpendicular_width(),
                                required,
                            ),
                        },
                    });
                }
                Err(crate::pbc::SimulationBoxError::PerpendicularWidthTooSmall {
                    direction,
                    width,
                    required,
                }) => {
                    stages.push(LintStage {
                        label: "box/cutoff",
                        status: LintStatus::Fail {
                            detail: format!(
                                "min perp width {width:.2e} m along `{direction}` < required {required:.2e} m"
                            ),
                            error: RunnerError::CellListBoxTooSmall {
                                direction,
                                width,
                                required,
                            },
                        },
                    });
                    return finalize_with_skips(stages, &["topology", "gpu"]);
                }
                Err(_) => unreachable!(
                    "check_min_perpendicular_width only produces PerpendicularWidthTooSmall"
                ),
            }
        }
    }

    // Stage 5: topology.
    let bond_type_names: Vec<&str> = config.bond_types.iter().map(|bt| bt.name()).collect();
    let angle_type_names: Vec<&str> = config.angle_types.iter().map(|at| at.name()).collect();
    let topology = match config.topology.as_ref() {
        Some(path) => match load_topology_file(
            path,
            n,
            &bond_type_names,
            &angle_type_names,
            &config.dihedral_types,
            &config.constraint_types,
            &registries.constraint_types,
        ) {
            Ok((bond_list, angle_list, dihedral_list, exclusion_list, constraint_list)) => {
                // Cross-check integrator/constraint compatibility now that the
                // constraint list is known.
                if let Err(e) = config
                    .validate_constraint_compatibility(registries, !constraint_list.is_empty())
                {
                    stages.push(LintStage {
                        label: "topology",
                        status: LintStatus::Fail {
                            detail: format!("{e}"),
                            error: RunnerError::Config(e),
                        },
                    });
                    return finalize_with_skips(stages, &["gpu"]);
                }
                stages.push(LintStage {
                    label: "topology",
                    status: LintStatus::Ok {
                        detail: format!(
                            "{}: {} bonds, {} angles, {} dihedrals, {} constraint groups",
                            path.display(),
                            bond_list.bonds.len(),
                            angle_list.angles.len(),
                            dihedral_list.dihedrals.len(),
                            constraint_list.groups.len(),
                        ),
                    },
                });
                Some((bond_list, angle_list, dihedral_list, exclusion_list, constraint_list))
            }
            Err(e) => {
                stages.push(LintStage {
                    label: "topology",
                    status: LintStatus::Fail {
                        detail: format!("{e}"),
                        error: RunnerError::TopologyFile(e),
                    },
                });
                return finalize_with_skips(stages, &["gpu"]);
            }
        },
        None => {
            stages.push(LintStage {
                label: "topology",
                status: LintStatus::Skipped {
                    reason: "not supplied".to_string(),
                },
            });
            None
        }
    };

    // Stage 6: gpu.
    if !with_gpu {
        stages.push(LintStage {
            label: "gpu",
            status: LintStatus::NotChecked {
                reason: "not checked (re-run with --with-gpu)".to_string(),
            },
        });
        return LintReport {
            stages,
            overall: LintOverall::Ok,
        };
    }

    let _ = n;
    match lint_gpu_full_setup(config, init, sim_box, topology, registries, lint_gpu) {
        Ok(()) => {
            stages.push(LintStage {
                label: "gpu",
                status: LintStatus::Ok {
                    detail: "init_device OK; ParticleBuffers, slots, ForceField allocated"
                        .to_string(),
                },
            });
            LintReport {
                stages,
                overall: LintOverall::Ok,
            }
        }
        Err((detail, error)) => {
            stages.push(LintStage {
                label: "gpu",
                status: LintStatus::Fail { detail, error },
            });
            LintReport {
                stages,
                overall: LintOverall::Fail,
            }
        }
    }
}

fn finalize_with_skips(mut stages: Vec<LintStage>, remaining: &[&'static str]) -> LintReport {
    for label in remaining {
        let reason = if *label == "gpu" {
            // Without --with-gpu the gpu stage is "not checked" regardless of
            // whether an earlier stage failed; with --with-gpu we'd still
            // never get here on an earlier failure, so report it as skipped.
            // The runtime call site sets the reason; we use a simple
            // heuristic: prior-stage failure short-circuits to a skipped gpu
            // entry, since this helper is only invoked on a failure.
            "skipped (earlier check failed)".to_string()
        } else {
            "skipped (earlier check failed)".to_string()
        };
        stages.push(LintStage {
            label,
            status: LintStatus::Skipped { reason },
        });
    }
    LintReport {
        stages,
        overall: LintOverall::Fail,
    }
}

fn compute_cutoff_max(config: &crate::io::Config) -> f64 {
    let mut cutoff_max: f64 = config
        .pair_interactions
        .iter()
        .map(|p| p.cutoff)
        .fold(0.0, f64::max);
    if let Some(s) = config.spme.as_ref() {
        cutoff_max = cutoff_max.max(s.r_cut_real);
    }
    cutoff_max
}

// Runs the GPU-touching half of the setup phase (init_device, cuFFT
// smoke test, velocity generation, particle state, buffers, slots,
// force field). Used by `lint_simulation_with_registries` when
// `with_gpu = true`. Returns `(detail, error)` on failure, suitable
// for embedding in a `LintStatus::Fail` on the `gpu` stage.
//
// The body delegates the steps 7-11 + 10a work to
// `simulation_setup_finish_gpu`, the same helper `SimulationSetup::new`
// uses, then walks `config.phases` to dry-run the per-phase slot
// builders. Any change to the shared helper is observed by both code
// paths by construction.
fn lint_gpu_full_setup(
    config: crate::io::Config,
    init: InitState,
    sim_box: crate::pbc::SimulationBox,
    topology: Option<(BondList, AngleList, DihedralList, ExclusionList, ConstraintList)>,
    registries: &crate::Registries,
    gpu: crate::gpu::GpuContext,
) -> Result<(), (String, RunnerError)> {
    let n = init.particle_count;
    let topology = topology.unwrap_or_else(|| {
        (
            BondList::empty(n),
            AngleList::empty(n),
            DihedralList::empty(n),
            ExclusionList::empty(n),
            ConstraintList::empty(n),
        )
    });
    let setup = simulation_setup_finish_gpu(
        config,
        registries.clone(),
        init,
        sim_box,
        topology,
        Duration::ZERO,
        Duration::ZERO,
        gpu,
        Duration::ZERO,
    )
    .map_err(|e| (format!("{e}"), e))?;

    let n_constraints = setup.constraint_list.total_constraint_count();
    for phase in &setup.config.phases {
        match phase {
            crate::io::PhaseKind::Md(md) => {
                let _integrator = setup
                    .registries
                    .integrators
                    .build(&md.integrator, &setup.gpu, n, n_constraints)
                    .map_err(|e| (format!("{e}"), RunnerError::Integrator(e)))?;
                let _thermostat = setup
                    .registries
                    .thermostats
                    .build_optional(md.thermostat.as_ref(), &setup.gpu, n, n_constraints)
                    .map_err(|e| (format!("{e}"), RunnerError::Thermostat(e)))?;
                let _barostat = setup
                    .registries
                    .barostats
                    .build_optional(md.barostat.as_ref(), &setup.gpu, n, n_constraints)
                    .map_err(|e| (format!("{e}"), RunnerError::Barostat(e)))?;
                let _constraint = setup
                    .registries
                    .constraint_types
                    .build_optional(
                        &setup.constraint_list,
                        &setup.gpu,
                        n,
                        &setup.masses,
                        &setup.config.constraint_types,
                    )
                    .map_err(|e| (format!("{e}"), RunnerError::Constraint(e)))?;
            }
            crate::io::PhaseKind::Minimization(min) => {
                let _minimizer = setup
                    .registries
                    .minimizers
                    .build(&min.algorithm, &setup.gpu, n, n_constraints)
                    .map_err(|e| (format!("{e}"), RunnerError::Minimizer(e)))?;
                let _constraint = setup
                    .registries
                    .constraint_types
                    .build_optional(
                        &setup.constraint_list,
                        &setup.gpu,
                        n,
                        &setup.masses,
                        &setup.config.constraint_types,
                    )
                    .map_err(|e| (format!("{e}"), RunnerError::Constraint(e)))?;
            }
        }
    }

    Ok(())
}

fn timed<T>(target: &mut Duration, f: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let value = f();
    *target = started.elapsed();
    value
}

// rq-ef902cf6 rq-dcfdb7c9
fn run_simulation_with_phase(
    config_path: &Path,
    registries: &crate::Registries,
) -> Result<RunSummary, (RunnerError, ExitPhase)> {
    let mut setup = SimulationSetup::new(config_path, registries.clone())
        .map_err(|e| (e, ExitPhase::Setup))?;
    setup.run_all_phases_with_exit_phase()
}

// Implementation detail: the public `SimulationSetup::new` body. Kept as
// a free function so the GPU-half can be shared with the lint flow via
// `simulation_setup_finish_gpu`.
fn simulation_setup_new_impl(
    config_path: &Path,
    registries: crate::Registries,
) -> Result<SimulationSetup, RunnerError> {
    // Time config_load before any other instrumentation exists.
    // Parse the config without running the registry-dispatched
    // validation, then run that validation against the
    // caller-supplied `registries`. This is what lets a custom-kind
    // config validate cleanly when a matching custom builder is
    // registered.
    let mut config_load_duration = Duration::ZERO;
    let config = timed(&mut config_load_duration, || load_config_raw(config_path))
        .map_err(RunnerError::Config)?;
    config
        .validate_against(&registries)
        .map_err(RunnerError::Config)?;

    // Pre-flight output existence checks across every phase.
    // Trajectory and log are gated by their per-phase `_every > 0`
    // predicates; the timings file is always written for every phase.
    for phase in &config.phases {
        match phase {
            crate::io::PhaseKind::Md(p) => {
                if p.output.trajectory_every > 0 && p.output.trajectory_path.exists() {
                    return Err(RunnerError::OutputExists {
                        path: p.output.trajectory_path.clone(),
                    });
                }
                if p.output.log_every > 0 && p.output.log_path.exists() {
                    return Err(RunnerError::OutputExists {
                        path: p.output.log_path.clone(),
                    });
                }
                if p.output.timings_path.exists() {
                    return Err(RunnerError::OutputExists {
                        path: p.output.timings_path.clone(),
                    });
                }
            }
            crate::io::PhaseKind::Minimization(m) => {
                if m.output.minlog_every > 0 && m.output.minlog_path.exists() {
                    return Err(RunnerError::OutputExists {
                        path: m.output.minlog_path.clone(),
                    });
                }
                if m.output.trajectory_every > 0 && m.output.trajectory_path.exists() {
                    return Err(RunnerError::OutputExists {
                        path: m.output.trajectory_path.clone(),
                    });
                }
                if m.output.timings_path.exists() {
                    return Err(RunnerError::OutputExists {
                        path: m.output.timings_path.clone(),
                    });
                }
            }
        }
    }

    let type_name_strings: Vec<String> = config
        .particle_types
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let type_name_refs: Vec<&str> = type_name_strings.iter().map(|s| s.as_str()).collect();

    // `SimulationBox` is device-resident, so we need a CudaDevice to
    // load the init state. `init_device` initialises the singleton
    // device + module-cache; downstream stages reuse the same handle.
    let mut gpu_init_duration = Duration::ZERO;
    let gpu = timed(&mut gpu_init_duration, init_device).map_err(RunnerError::Gpu)?;

    let mut init_load_duration = Duration::ZERO;
    let init = timed(&mut init_load_duration, || {
        load_init_state(&gpu.device, &config.init, &type_name_refs, config.units)
    })
    .map_err(RunnerError::InitState)?;

    let sim_box = init.sim_box.clone();
    let n = init.particle_count;

    // Cell-list box-compatibility check (uses the init file's box).
    // Cutoff aggregation stays here because it walks the config; the
    // per-direction width check is delegated to `SimulationBox`.
    if let NeighborListConfig::CellList { r_skin, .. } = &config.neighbor_list {
        let mut cutoff_max: f64 = config
            .pair_interactions
            .iter()
            .map(|p| p.cutoff)
            .fold(0.0, f64::max);
        if let Some(s) = config.spme.as_ref() {
            cutoff_max = cutoff_max.max(s.r_cut_real);
        }
        let required = (3.0 * (cutoff_max + r_skin)) as Real;
        if let Err(e) = sim_box.check_min_perpendicular_width(required) {
            return Err(match e {
                crate::pbc::SimulationBoxError::PerpendicularWidthTooSmall {
                    direction,
                    width,
                    required,
                } => RunnerError::CellListBoxTooSmall {
                    direction,
                    width,
                    required,
                },
                _ => unreachable!(
                    "check_min_perpendicular_width only produces PerpendicularWidthTooSmall"
                ),
            });
        }
    }

    // Load the .topology file when supplied, otherwise build empty bond /
    // angle / exclusion lists keyed to `n`.
    let bond_type_names: Vec<&str> =
        config.bond_types.iter().map(|bt| bt.name()).collect();
    let angle_type_names: Vec<&str> =
        config.angle_types.iter().map(|at| at.name()).collect();
    let topology: (BondList, AngleList, DihedralList, ExclusionList, ConstraintList) =
        match config.topology.as_ref() {
            Some(path) => load_topology_file(
                path,
                n,
                &bond_type_names,
                &angle_type_names,
                &config.dihedral_types,
                &config.constraint_types,
                &registries.constraint_types,
            )
            .map_err(RunnerError::TopologyFile)?,
            None => (
                BondList::empty(n),
                AngleList::empty(n),
                DihedralList::empty(n),
                ExclusionList::empty(n),
                ConstraintList::empty(n),
            ),
        };

    simulation_setup_finish_gpu(
        config,
        registries,
        init,
        sim_box,
        topology,
        config_load_duration,
        init_load_duration,
        gpu,
        gpu_init_duration,
    )
}

// Runs steps 7-11 + 10a of `SimulationSetup::new`. Shared between
// `SimulationSetup::new` (via `simulation_setup_new_impl`) and the
// GPU-touching half of the lint pipeline. Takes ownership of the
// inputs because the resulting `SimulationSetup` owns them.
#[allow(clippy::too_many_arguments)]
fn simulation_setup_finish_gpu(
    config: crate::io::Config,
    registries: crate::Registries,
    init: InitState,
    sim_box: crate::pbc::SimulationBox,
    topology: (BondList, AngleList, DihedralList, ExclusionList, ConstraintList),
    config_load_duration: Duration,
    init_load_duration: Duration,
    gpu: crate::gpu::GpuContext,
    gpu_init_duration: Duration,
) -> Result<SimulationSetup, RunnerError> {
    let n = init.particle_count;
    let (bond_list, angle_list, dihedral_list, exclusion_list, constraint_list) = topology;

    // rq-637cd1a5 rq-02f4d342 rq-ea4205ec
    if config.spme.is_some() {
        let differences = crate::gpu::cufft::cufft_determinism_smoke_test(&gpu.device)
            .map_err(|_| {
                RunnerError::Gpu(crate::gpu::GpuError(cudarc::driver::DriverError(
                    cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
                )))
            })?;
        if differences != 0 {
            return Err(RunnerError::CuFftNonDeterministic { differences });
        }
    }

    // Build masses and charges arrays from per-particle type_index lookup.
    let mut masses_f64: Vec<f64> = Vec::with_capacity(n);
    let mut masses: Vec<Real> = Vec::with_capacity(n);
    let mut charges: Vec<Real> = Vec::with_capacity(n);
    for &ti in &init.type_indices {
        let pt = &config.particle_types[ti as usize];
        masses_f64.push(pt.mass);
        masses.push(pt.mass as Real);
        charges.push(pt.charge as Real);
    }

    let n_constraints = constraint_list.total_constraint_count();
    // Thermal degrees of freedom used by `compute_temperature` and by
    // the initial-velocity equipartition rescale: constraint- and
    // COM-removed.
    let n_thermal_dof: u32 = ((3 * n as i64) - n_constraints as i64 - 3).max(0) as u32;

    // Build velocities: either from the init state or sampled.
    // Velocity generation runs once at phase-0 entry; the duration is
    // recorded into phase 0's Timings inside the per-phase loop.
    let mut velocity_generation_duration = Duration::ZERO;
    let (velocities_x, velocities_y, velocities_z) = match init.velocities {
        Some(InitVelocities {
            velocities_x,
            velocities_y,
            velocities_z,
        }) => (velocities_x, velocities_y, velocities_z),
        None => timed(&mut velocity_generation_duration, || {
            generate_velocities(
                n,
                n_constraints,
                config.simulation.temperature,
                config.simulation.seed,
                &masses_f64,
            )
        }),
    };

    let images_arg = init
        .images
        .as_ref()
        .map(|im| (im.images_x.clone(), im.images_y.clone(), im.images_z.clone()));
    let charges_for_force_field = charges.clone();
    let state = ParticleState::new(
        init.positions_x.clone(),
        init.positions_y.clone(),
        init.positions_z.clone(),
        velocities_x,
        velocities_y,
        velocities_z,
        masses.clone(),
        charges.clone(),
        init.type_indices.clone(),
        None,
        images_arg,
    )
    .map_err(RunnerError::ParticleState)?;

    // ParticleBuffers persist across phases; allocation happens once.
    let mut upload = Duration::ZERO;
    let mut buffers = timed(&mut upload, || ParticleBuffers::new(&gpu, &state)).map_err(|e| {
        match e {
            ParticleStateError::Gpu(g) => RunnerError::Gpu(g),
            other => RunnerError::ParticleState(other),
        }
    })?;
    // rq-acfda5d4 — runner-side enforcement of the integrator/constraint
    // compatibility rule, applied to every phase. Cannot run during
    // `Config::validate_against` because the topology file is loaded
    // separately.
    config
        .validate_constraint_compatibility(&registries, !constraint_list.is_empty())
        .map_err(RunnerError::Config)?;

    // Project the freshly-sampled initial velocities onto the
    // constraint velocity manifold and re-scale to match the target
    // thermal kinetic energy.
    if !constraint_list.is_empty() && config.simulation.temperature > 0.0 && n >= 2 {
        let mut init_constraint = registries
            .constraint_types
            .build_optional(
                &constraint_list,
                &gpu,
                n,
                &masses,
                &config.constraint_types,
            )
            .map_err(RunnerError::Constraint)?;
        if let Some(c) = init_constraint.as_mut() {
            let mut init_timings = Timings::new(&gpu).map_err(RunnerError::Timings)?;
            c.apply_initial_velocity_projection(&mut buffers, &sim_box, &mut init_timings)
                .map_err(RunnerError::Constraint)?;
            let mut ke_scratch = gpu
                .device
                .alloc_zeros::<Real>(1)
                .map_err(|e| RunnerError::Gpu(crate::gpu::GpuError::from(e)))?;
            let ke_after = crate::gpu::compute_kinetic_energy(&mut buffers, &mut ke_scratch)
                .map_err(RunnerError::Gpu)? as f64;
            let n_thermal_dof_f64 = n_thermal_dof as f64;
            // k_B = 1 in atomic units; simulation.temperature is k_B · T in Hartrees.
            let target_ke = 0.5 * n_thermal_dof_f64 * config.simulation.temperature;
            if ke_after > 0.0 && target_ke > 0.0 {
                let factor = (target_ke / ke_after).sqrt() as Real;
                crate::gpu::rescale_velocities(&mut buffers, factor).map_err(RunnerError::Gpu)?;
            }
        }
    }

    // Select the JIT fast-math compile mode before any kernel is built
    // (ForceField::new compiles the composed pair/bonded/angle and SPME
    // kernels; the post-force composer is built further below). rq-a84e1c76
    crate::forces::set_jit_fast_math(config.simulation.fast_math);

    // ForceField persists across phases.
    let force_field = ForceField::new(
        &registries.potentials,
        &gpu,
        n,
        &sim_box,
        &config.particle_types,
        &config.pair_interactions,
        &config.bond_types,
        &config.angle_types,
        &config.dihedral_types,
        config.spme.as_ref(),
        &charges_for_force_field,
        &bond_list,
        &angle_list,
        &dihedral_list,
        &exclusion_list,
        &config.neighbor_list,
    )
    .map_err(RunnerError::ForceField)?;

    Ok(SimulationSetup {
        config,
        registries,
        gpu,
        buffers,
        sim_box,
        force_field,
        constraint_list,
        bond_list,
        angle_list,
        dihedral_list,
        exclusion_list,
        masses,
        charges,
        type_indices: init.type_indices,
        n_constraints: n_constraints as u32,
        n_thermal_dof,
        pre_phase_durations: PrePhaseDurations {
            config_load: config_load_duration,
            init_load: init_load_duration,
            gpu_init: gpu_init_duration,
            velocity_generation: velocity_generation_duration,
            upload,
        },
    })
}

impl SimulationSetup {
    // rq-b1a2d006 — constructor: runs steps 2-11 of *Once-only setup*.
    pub fn new(
        config_path: &Path,
        registries: crate::Registries,
    ) -> Result<SimulationSetup, RunnerError> {
        simulation_setup_new_impl(config_path, registries)
    }

    // rq-b1a2d006 — orchestrator: iterates `self.config.phases`,
    // dispatches to `run_md_phase` or `run_minimization_phase`,
    // aggregates a `RunSummary`.
    pub fn run_all_phases(&mut self) -> Result<RunSummary, RunnerError> {
        self.run_all_phases_with_exit_phase().map_err(|(e, _)| e)
    }

    // Internal variant used by `run_simulation_with_phase` so the CLI
    // can preserve the setup-vs-loop exit-code distinction.
    pub(crate) fn run_all_phases_with_exit_phase(
        &mut self,
    ) -> Result<RunSummary, (RunnerError, ExitPhase)> {
        let total_started = Instant::now();
        let n_phases = self.config.phases.len();
        let mut phase_summaries: Vec<PhaseSummary> = Vec::with_capacity(n_phases);
        let mut total_n_steps: u64 = 0;

        for phase_index in 0..n_phases {
            // Dispatch by re-borrowing each iteration so we don't clash
            // with the &mut self required by the per-phase functions.
            let kind_tag = match &self.config.phases[phase_index] {
                crate::io::PhaseKind::Md(_) => 0u8,
                crate::io::PhaseKind::Minimization(_) => 1u8,
            };
            let summary = if kind_tag == 0 {
                // Clone the phase config so we can release the borrow on
                // `self.config` before calling `run_md_phase`, which takes
                // `&mut self`.
                let phase = match &self.config.phases[phase_index] {
                    crate::io::PhaseKind::Md(p) => p.clone(),
                    _ => unreachable!(),
                };
                run_md_phase_inner(self, &phase, phase_index)?
            } else {
                let phase = match &self.config.phases[phase_index] {
                    crate::io::PhaseKind::Minimization(p) => p.clone(),
                    _ => unreachable!(),
                };
                run_minimization_phase_inner(self, &phase, phase_index)?
            };
            total_n_steps += summary.n_steps;
            phase_summaries.push(summary);
        }

        Ok(RunSummary {
            phases: phase_summaries,
            total_n_steps,
            total_elapsed_micros: total_started.elapsed().as_micros(),
        })
    }
}

// rq-63d694e9 — per-MD-phase function. Public, takes `&mut SimulationSetup`.
pub fn run_md_phase(
    setup: &mut SimulationSetup,
    phase: &crate::io::PhaseConfig,
    phase_index: usize,
) -> Result<PhaseSummary, RunnerError> {
    run_md_phase_inner(setup, phase, phase_index).map_err(|(e, _)| e)
}

// rq-10903c8d — per-minimization-phase function. Public, takes
// `&mut SimulationSetup`.
pub fn run_minimization_phase(
    setup: &mut SimulationSetup,
    phase: &crate::io::MinimizationConfig,
    phase_index: usize,
) -> Result<PhaseSummary, RunnerError> {
    run_minimization_phase_inner(setup, phase, phase_index).map_err(|(e, _)| e)
}

// rq-63d694e9 — per-MD-phase impl. Companion to `run_md_phase` that
// preserves the `ExitPhase` tag so the CLI can distinguish setup-time
// errors (exit 1) from loop-time errors (exit 2).
pub(crate) fn run_md_phase_inner(
    setup: &mut SimulationSetup,
    phase: &crate::io::PhaseConfig,
    phase_index: usize,
) -> Result<PhaseSummary, (RunnerError, ExitPhase)> {
    let progress_to_stdout = std::io::IsTerminal::is_terminal(&std::io::stdout());
    let n = setup.buffers.particle_count();
    let n_constraints = setup.n_constraints as usize;
    let n_thermal_dof = setup.n_thermal_dof;
    let units = setup.config.units;

    // Per-phase Timings instance (fresh kernel-event pairs).
    let mut timings = Timings::new(&setup.gpu)
        .map_err(|e| (RunnerError::Timings(e), ExitPhase::Setup))?;
    // Phase 0 replays the pre-instrumented host stages.
    if phase_index == 0 {
        let pre = &setup.pre_phase_durations;
        timings.record_host(HostStage::CONFIG_LOAD, pre.config_load);
        timings.record_host(HostStage::INIT_LOAD, pre.init_load);
        timings.record_host(HostStage::GPU_INIT, pre.gpu_init);
        if pre.velocity_generation > Duration::ZERO {
            timings.record_host(HostStage::VELOCITY_GENERATION, pre.velocity_generation);
        }
        timings.record_host(HostStage::HOST_TO_DEVICE_UPLOAD, pre.upload);
    }

    let type_name_strings: Vec<String> = setup
        .config
        .particle_types
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let type_indices: Vec<u32> = setup.type_indices.clone();

    // Build the four slot handles for this phase.
    let mut integrator = setup
        .registries
        .integrators
        .build(&phase.integrator, &setup.gpu, n, n_constraints)
        .map_err(|e| (RunnerError::Integrator(e), ExitPhase::Setup))?;
    let mut thermostat = setup
        .registries
        .thermostats
        .build_optional(phase.thermostat.as_ref(), &setup.gpu, n, n_constraints)
        .map_err(|e| (RunnerError::Thermostat(e), ExitPhase::Setup))?;
    let mut barostat = setup
        .registries
        .barostats
        .build_optional(phase.barostat.as_ref(), &setup.gpu, n, n_constraints)
        .map_err(|e| (RunnerError::Barostat(e), ExitPhase::Setup))?;
    // rq-3e1fba8b — hand the barostat the connectivity-derived molecule
    // partition and the initial box (the Monte-Carlo barostat uploads
    // its molecule tables and resolves its default volume step here).
    if let Some(b) = barostat.as_mut() {
        let molecules = crate::forces::MoleculeList::from_topology(
            n,
            &setup.bond_list,
            &setup.constraint_list,
        );
        b.init_run(&setup.sim_box, &molecules)
            .map_err(|e| (RunnerError::Barostat(e), ExitPhase::Setup))?;
    }
    let mut constraint = setup
        .registries
        .constraint_types
        .build_optional(
            &setup.constraint_list,
            &setup.gpu,
            n,
            &setup.masses,
            &setup.config.constraint_types,
        )
        .map_err(|e| (RunnerError::Constraint(e), ExitPhase::Setup))?;

    // rq-8ac9773d — JIT-composed post-force per-particle kernel setup
    // runs once per phase, before the CUDA-graph eligibility check.
    // Both the graph-capture path and the per-step launch path
    // dispatch the same composed kernel. Built-in slots always
    // expose a post-force fragment; a slot returning `None` raises
    // `MissingPostForcePerParticleFragment` before either path runs.
    let mut composed_post_force: Option<crate::forces::JitComposedPostForcePerParticle> =
        None;
    let mut post_force_substep_index: Option<usize> = None;
    {
        let int_frag = integrator
            .post_force_per_particle()
            .map(|p| p.post_force_per_particle_fragment());
        let therm_active = thermostat.is_some();
        // Only a per-step barostat contributes a post-force per-particle
        // fragment; a periodic (Monte-Carlo) barostat does no per-step
        // work and is exempt from the fragment requirement.
        let baro_active = barostat_couples_per_step(&barostat);
        let therm_frag = thermostat
            .as_ref()
            .and_then(|t| t.post_force_per_particle())
            .map(|p| p.post_force_per_particle_fragment());
        let baro_frag = barostat
            .as_ref()
            .and_then(|b| b.post_force_per_particle())
            .map(|p| p.post_force_per_particle_fragment());
        if int_frag.is_none() {
            return Err((
                RunnerError::MissingPostForcePerParticleFragment {
                    kind: "integrator",
                    label: "<integrator>",
                },
                ExitPhase::Setup,
            ));
        }
        if therm_active && therm_frag.is_none() {
            return Err((
                RunnerError::MissingPostForcePerParticleFragment {
                    kind: "thermostat",
                    label: "<thermostat>",
                },
                ExitPhase::Setup,
            ));
        }
        if baro_active && baro_frag.is_none() {
            return Err((
                RunnerError::MissingPostForcePerParticleFragment {
                    kind: "barostat",
                    label: "<barostat>",
                },
                ExitPhase::Setup,
            ));
        }
        let mut fragments: Vec<crate::forces::PerParticleFragment> = Vec::new();
        fragments.push(int_frag.unwrap());
        if let Some(f) = therm_frag {
            fragments.push(f);
        }
        if let Some(f) = baro_frag {
            fragments.push(f);
        }
        match crate::forces::JitComposedPostForcePerParticle::compile_and_load(
            &setup.gpu.device,
            &fragments,
        ) {
            Ok(k) => {
                composed_post_force = Some(k);
                post_force_substep_index =
                    integrator.post_force_substep_index(phase.dt as Real);
            }
            Err(e) => {
                return Err((
                    match e {
                        crate::forces::ForceFieldError::FragmentCompileFailed { log } => {
                            RunnerError::PostForceFragmentCompileFailed { log }
                        }
                        crate::forces::ForceFieldError::FragmentLoadFailed(g) => {
                            RunnerError::PostForceFragmentLoadFailed(g)
                        }
                        other => RunnerError::ForceField(other),
                    },
                    ExitPhase::Setup,
                ));
            }
        }
    }

    // Open per-phase output writers.
    let mut traj_writer: Option<TrajectoryWriter> = if phase.output.trajectory_every > 0 {
        Some(
            TrajectoryWriter::open(
                &phase.output.trajectory_path,
                units,
                phase.output.include_velocities,
                phase.output.include_images,
                type_name_strings.clone(),
            )
            .map_err(|e| (RunnerError::Trajectory(e), ExitPhase::Setup))?,
        )
    } else {
        None
    };
    let mut log_extra_columns: Vec<(&'static str, crate::units::Dimension)> =
        integrator.log_column_names().to_vec();
    if let Some(t) = thermostat.as_ref() {
        log_extra_columns.extend_from_slice(t.log_column_names());
    }
    if let Some(b) = barostat.as_ref() {
        log_extra_columns.extend_from_slice(b.log_column_names());
    }
    let mut log_writer: Option<LogWriter> = if phase.output.log_every > 0 {
        Some(
            LogWriter::open(&phase.output.log_path, units, &log_extra_columns)
                .map_err(|e| (RunnerError::Log(e), ExitPhase::Setup))?,
        )
    } else {
        None
    };
    let mut pe_scratch: Option<cudarc::driver::CudaSlice<Real>> = if !log_extra_columns.is_empty() {
        Some(setup.gpu.device.alloc_zeros::<Real>(1).map_err(|e| {
            (RunnerError::Gpu(crate::gpu::GpuError(e)), ExitPhase::Setup)
        })?)
    } else {
        None
    };

    // Host-side frame buffer for download_from. Allocated fresh per
    // phase; the buffer is overwritten by each `download_from` call.
    let mut frame = new_frame_buffer(n, &setup.masses, &setup.charges, &type_indices)
        .map_err(|e| (RunnerError::ParticleState(e), ExitPhase::Setup))?;

    let phase_started = Instant::now();

    // Warm-up: refresh forces to match current positions.
    setup
        .force_field
        .step(
            &mut setup.buffers,
            &setup.sim_box,
            &mut timings,
            crate::forces::AggregateLevel::ForcesAndScalars,
        )
        .map_err(|e| (RunnerError::ForceField(e), ExitPhase::Setup))?;

    let mut frames_written: u64 = 0;
    let mut log_rows_written: u64 = 0;

    // Phase step-0 outputs.
    if traj_writer.is_some() || log_writer.is_some() {
        let mut dl = Duration::ZERO;
        timed(&mut dl, || frame.download_from(&setup.buffers)).map_err(|e| match e {
            ParticleStateError::Gpu(g) => (RunnerError::Gpu(g), ExitPhase::Setup),
            other => (RunnerError::ParticleState(other), ExitPhase::Setup),
        })?;
        timings.record_host(HostStage::DEVICE_TO_HOST_DOWNLOAD, dl);
    }
    if let Some(writer) = traj_writer.as_mut() {
        let mut tw = Duration::ZERO;
        timed(&mut tw, || {
            write_traj_frame(writer, 0, phase.dt, &setup.sim_box, &type_indices, &frame)
        })
        .map_err(|e| (RunnerError::Trajectory(e), ExitPhase::Setup))?;
        timings.record_host(HostStage::TRAJECTORY_WRITE, tw);
        frames_written += 1;
    }
    if let Some(writer) = log_writer.as_mut() {
        if let Some(t) = thermostat.as_mut() {
            t.flush_pending_injection(&setup.gpu.device)
                .map_err(|e| (RunnerError::Thermostat(e), ExitPhase::Setup))?;
        }
        if let Some(b) = barostat.as_mut() {
            b.flush_pending_injection(&setup.gpu.device)
                .map_err(|e| (RunnerError::Barostat(e), ExitPhase::Setup))?;
        }
        let ke = compute_kinetic_energy(
            &frame.masses,
            &frame.velocities_x,
            &frame.velocities_y,
            &frame.velocities_z,
        );
        let t = compute_temperature(ke, n_thermal_dof);
        let extras = if log_extra_columns.is_empty() {
            Vec::new()
        } else {
            let scratch = pe_scratch
                .as_mut()
                .expect("pe_scratch allocated when log_extra_columns non-empty");
            timings
                .kernel_start(KernelStage::POTENTIAL_ENERGY_REDUCE)
                .map_err(|e| (RunnerError::Timings(e), ExitPhase::Setup))?;
            let pe = compute_total_potential_energy(&mut setup.buffers, scratch)
                .map_err(|g| (RunnerError::Gpu(g), ExitPhase::Setup))?;
            timings
                .kernel_stop(KernelStage::POTENTIAL_ENERGY_REDUCE)
                .map_err(|e| (RunnerError::Timings(e), ExitPhase::Setup))?;
            collect_log_extras(
                integrator.as_ref(),
                thermostat.as_deref(),
                barostat.as_deref(),
                ke,
                pe as f64,
            )
        };
        let mut lw = Duration::ZERO;
        timed(&mut lw, || writer.write_row(0, 0.0, ke, t, &extras))
            .map_err(|e| (RunnerError::Log(e), ExitPhase::Setup))?;
        timings.record_host(HostStage::LOG_WRITE, lw);
        log_rows_written += 1;
    }

    let n_steps = phase.n_steps;
    let progress_every = (n_steps / 100).max(1);
    let dt = phase.dt as Real;
    let phase_name = phase.name.as_str();

    // Pre-fetch the supports_constraints bit before the per-step loop;
    // the registry borrow does not need to be held during the loop body.
    let supports_constraints = setup
        .registries
        .integrators
        .lookup(&phase.integrator.kind)
        .map(|b| b.supports_constraints(&phase.integrator.params))
        .unwrap_or(false);

    // Decide whether this phase is eligible for CUDA graph capture.
    // See `rqm/cuda-graphs.md` for the activation policy. Every
    // active slot must report `graph_compatible == true`, the
    // run-wide override flag must be off, and capture must succeed
    // at runtime; otherwise the phase runs the per-step launch loop
    // with full per-kernel `Timings`.
    let graph_eligible = !setup.config.simulation.cuda_graphs_disable
        && phase_slots_graph_compatible(setup, phase);

    // Graph-timing calibration: CUDA forbids `cuEventElapsedTime` on the
    // per-kernel events captured into a graph, so graph mode cannot time
    // its own replays. Instead run a few instrumented per-step iterations
    // up front (real CUDA-event timing) and snapshot a representative
    // per-kernel duration; the replay loop folds it in for every step.
    // The per-step path is bit-identical to the graph replay (fixed-point
    // forces are summation-order invariant), so this does not perturb the
    // trajectory; the cost is a handful of steps out of `n_steps`.
    const GRAPH_TIMING_CALIBRATION_STEPS: u64 = 8;
    let calib: u64 = if graph_eligible && n_steps >= 1 {
        GRAPH_TIMING_CALIBRATION_STEPS.min(n_steps)
    } else {
        0
    };
    if calib > 0 {
        run_per_step_range(
            1,
            calib,
            setup,
            phase,
            &mut integrator,
            &mut thermostat,
            &mut barostat,
            &mut constraint,
            supports_constraints,
            dt,
            composed_post_force.as_ref(),
            post_force_substep_index,
            &mut timings,
            &mut frame,
            &mut traj_writer,
            &mut log_writer,
            &mut pe_scratch,
            &type_indices,
            n_thermal_dof,
            &log_extra_columns,
            phase_started,
            phase_name,
            progress_to_stdout,
            progress_every,
            &mut frames_written,
            &mut log_rows_written,
        )?;
        timings.snapshot_graph_representatives();
    }

    let graph_loop: Option<crate::gpu::GraphLoop> = if graph_eligible && calib < n_steps {
        match capture_phase_graph(
            &mut setup.buffers,
            &mut setup.sim_box,
            &mut setup.force_field,
            integrator.as_mut(),
            &mut thermostat,
            &mut barostat,
            &mut constraint,
            supports_constraints,
            dt,
            &mut timings,
            &setup.gpu.device,
            composed_post_force.as_ref(),
            post_force_substep_index,
        ) {
            Ok(exec) => Some(exec),
            Err(e) => {
                eprintln!(
                    "warning: cuda graph capture failed for phase `{phase_name}`: {e}; falling back to per-step launches"
                );
                None
            }
        }
    } else {
        None
    };

    if let Some(mut graph_loop) = graph_loop {
        // rq-67a09135 rq-1217c816
        // The batched loop re-captures the phase graph in place when a
        // batch-boundary rebuild reallocates a packed-neighbour buffer.
        // It returns `Some(resume_step)` only when such a re-capture
        // failed, in which case the remaining steps run on the per-step
        // launch loop.
        let fallback_from = run_batched_graph_loop(
            setup,
            phase,
            phase_index,
            calib,
            &mut graph_loop,
            integrator.as_mut(),
            &mut thermostat,
            &mut barostat,
            &mut constraint,
            supports_constraints,
            dt,
            composed_post_force.as_ref(),
            post_force_substep_index,
            &mut timings,
            &mut frame,
            &mut traj_writer,
            &mut log_writer,
            &mut pe_scratch,
            &type_indices,
            n_thermal_dof,
            &log_extra_columns,
            phase_started,
            phase_name,
            progress_to_stdout,
            progress_every,
            &mut frames_written,
            &mut log_rows_written,
        )?;
        if let Some(resume) = fallback_from {
            run_per_step_range(
                resume,
                n_steps,
                setup,
                phase,
                &mut integrator,
                &mut thermostat,
                &mut barostat,
                &mut constraint,
                supports_constraints,
                dt,
                composed_post_force.as_ref(),
                post_force_substep_index,
                &mut timings,
                &mut frame,
                &mut traj_writer,
                &mut log_writer,
                &mut pe_scratch,
                &type_indices,
                n_thermal_dof,
                &log_extra_columns,
                phase_started,
                phase_name,
                progress_to_stdout,
                progress_every,
                &mut frames_written,
                &mut log_rows_written,
            )?;
        }
    } else {
        // Steps `1..=calib` already ran on the per-step path as graph
        // calibration; resume after them. When `calib == 0` (graphs
        // disabled / ineligible) this is the whole phase; when
        // `calib == n_steps` (tiny phase fully covered by calibration)
        // the range is empty and this is a no-op.
        run_per_step_range(
            calib + 1,
            n_steps,
            setup,
            phase,
            &mut integrator,
            &mut thermostat,
            &mut barostat,
            &mut constraint,
            supports_constraints,
            dt,
            composed_post_force.as_ref(),
            post_force_substep_index,
            &mut timings,
            &mut frame,
            &mut traj_writer,
            &mut log_writer,
            &mut pe_scratch,
            &type_indices,
            n_thermal_dof,
            &log_extra_columns,
            phase_started,
            phase_name,
            progress_to_stdout,
            progress_every,
            &mut frames_written,
            &mut log_rows_written,
        )?;
    }

    if let Some(writer) = traj_writer.as_mut() {
        writer
            .flush()
            .map_err(|e| (RunnerError::Trajectory(e), ExitPhase::Loop))?;
    }
    if let Some(writer) = log_writer.as_mut() {
        writer
            .flush()
            .map_err(|e| (RunnerError::Log(e), ExitPhase::Loop))?;
    }

    let phase_elapsed = phase_started.elapsed();
    timings.record_host(HostStage::TOTAL_RUNTIME, phase_elapsed);
    let report = timings
        .finalize()
        .map_err(|e| (RunnerError::Timings(e), ExitPhase::Loop))?;
    write_timings_file(&phase.output.timings_path, &report)
        .map_err(|e| (RunnerError::TimingsWriter(e), ExitPhase::Loop))?;

    Ok(PhaseSummary {
        name: phase.name.clone(),
        n_steps,
        frames_written,
        log_rows_written,
        elapsed_micros: phase_elapsed.as_micros(),
        kind: "md",
        convergence: None,
    })
}

// Allocate a fresh host-side `ParticleState` buffer for the per-phase
// download path. The arrays are zero-initialised; `download_from`
// overwrites them.
fn new_frame_buffer(
    n: usize,
    masses: &[Real],
    charges: &[Real],
    type_indices: &[u32],
) -> Result<ParticleState, ParticleStateError> {
    ParticleState::new(
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        masses.to_vec(),
        charges.to_vec(),
        type_indices.to_vec(),
        None,
        None,
    )
}


// rq-10903c8d — per-minimization-phase impl. Companion to
// `run_minimization_phase` that preserves the `ExitPhase` tag so the
// CLI can distinguish setup-time errors (exit 1) from loop-time errors
// (exit 2).
// rq-393a57e4
pub(crate) fn run_minimization_phase_inner(
    setup: &mut SimulationSetup,
    min: &crate::io::MinimizationConfig,
    phase_index: usize,
) -> Result<PhaseSummary, (RunnerError, ExitPhase)> {
    let progress_to_stdout = std::io::IsTerminal::is_terminal(&std::io::stdout());
    let n = setup.buffers.particle_count();
    let n_constraints = setup.n_constraints as usize;
    let units = setup.config.units;

    let mut timings = Timings::new(&setup.gpu)
        .map_err(|e| (RunnerError::Timings(e), ExitPhase::Setup))?;
    // Phase 0 replays the pre-instrumented host stages, just like the
    // MD branch.
    if phase_index == 0 {
        let pre = &setup.pre_phase_durations;
        timings.record_host(HostStage::CONFIG_LOAD, pre.config_load);
        timings.record_host(HostStage::INIT_LOAD, pre.init_load);
        timings.record_host(HostStage::GPU_INIT, pre.gpu_init);
        if pre.velocity_generation > Duration::ZERO {
            timings.record_host(HostStage::VELOCITY_GENERATION, pre.velocity_generation);
        }
        timings.record_host(HostStage::HOST_TO_DEVICE_UPLOAD, pre.upload);
    }

    let type_name_strings: Vec<String> = setup
        .config
        .particle_types
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let type_indices: Vec<u32> = setup.type_indices.clone();
    let mut frame = new_frame_buffer(n, &setup.masses, &setup.charges, &type_indices)
        .map_err(|e| (RunnerError::ParticleState(e), ExitPhase::Setup))?;
    let timings = &mut timings;
    let buffers = &mut setup.buffers;
    let sim_box = &mut setup.sim_box;
    let force_field = &mut setup.force_field;
    let gpu = &setup.gpu;

    let mut minimizer = setup
        .registries
        .minimizers
        .build(&min.algorithm, gpu, n, n_constraints)
        .map_err(|e| (RunnerError::Minimizer(e), ExitPhase::Setup))?;
    let mut constraint = setup
        .registries
        .constraint_types
        .build_optional(
            &setup.constraint_list,
            gpu,
            n,
            &setup.masses,
            &setup.config.constraint_types,
        )
        .map_err(|e| (RunnerError::Constraint(e), ExitPhase::Setup))?;

    let mut minlog_writer: Option<MinlogWriter> = if min.output.minlog_every > 0 {
        Some(
            MinlogWriter::open(&min.output.minlog_path, units)
                .map_err(|e| (RunnerError::Minlog(e), ExitPhase::Setup))?,
        )
    } else {
        None
    };
    let mut traj_writer: Option<TrajectoryWriter> = if min.output.trajectory_every > 0 {
        Some(
            TrajectoryWriter::open(
                &min.output.trajectory_path,
                units,
                false, // never include velocities for minimization frames
                min.output.include_images,
                type_name_strings.clone(),
            )
            .map_err(|e| (RunnerError::Trajectory(e), ExitPhase::Setup))?,
        )
    } else {
        None
    };

    let phase_started = Instant::now();

    // Warm up forces and potential energy at the current positions.
    force_field
        .step(
            buffers,
            sim_box,
            timings,
            crate::forces::AggregateLevel::ForcesAndScalars,
        )
        .map_err(|e| (RunnerError::ForceField(e), ExitPhase::Setup))?;

    // Compute initial accepted state via the minimizer.
    let (energy0, fmax0) = minimizer
        .initial_state(buffers, timings)
        .map_err(|e| (RunnerError::Minimizer(e), ExitPhase::Setup))?;
    let initial_step = {
        // The minimizer's `current_step` is private; we report
        // whatever step it would use on iteration 0 via the helper
        // baked into the first report below. Use 0.0 here for the
        // step-0 row's `step` column — the convention noted in the
        // requirements doc is to use `initial_step`, but that value
        // is not exposed through the trait; reporting 0.0 keeps the
        // contract simple and the step-0 row trivially identifiable.
        0.0_f64
    };

    let mut frames_written: u64 = 0;
    let mut log_rows_written: u64 = 0;

    // Phase step-0 row + frame.
    let mut last_logged_iter: Option<u64> = None;
    if let Some(writer) = minlog_writer.as_mut() {
        let mut lw = Duration::ZERO;
        timed(&mut lw, || {
            writer.write_row(0, energy0, fmax0, initial_step, true)
        })
        .map_err(|e| (RunnerError::Minlog(e), ExitPhase::Loop))?;
        timings.record_host(HostStage::LOG_WRITE, lw);
        log_rows_written += 1;
        last_logged_iter = Some(0);
    }
    if traj_writer.is_some() {
        // Download positions for the step-0 frame.
        let mut dl = Duration::ZERO;
        timed(&mut dl, || frame.download_from(&*buffers)).map_err(|e| match e {
            ParticleStateError::Gpu(g) => (RunnerError::Gpu(g), ExitPhase::Loop),
            other => (RunnerError::ParticleState(other), ExitPhase::Loop),
        })?;
        timings.record_host(HostStage::DEVICE_TO_HOST_DOWNLOAD, dl);
        let writer = traj_writer.as_mut().expect("traj writer is some");
        let mut tw = Duration::ZERO;
        timed(&mut tw, || {
            write_traj_frame(writer, 0, 0.0, sim_box, &type_indices, &frame)
        })
        .map_err(|e| (RunnerError::Trajectory(e), ExitPhase::Loop))?;
        timings.record_host(HostStage::TRAJECTORY_WRITE, tw);
        frames_written += 1;
    }

    // Pre-loop convergence check on the initial state. Only
    // force-based criteria can fire here — the energy-tolerance check
    // compares two distinct accepted energies and there is only one
    // before the first iteration.
    let initial_report = crate::minimizer::MinimizerStepReport {
        accepted: true,
        energy: energy0,
        max_force: fmax0,
        step_size: initial_step,
        prev_energy: energy0,
    };
    let mut convergence_reason: Option<MinimizerConvergence> = if fmax0 == 0.0 {
        Some(MinimizerConvergence::ForceZero)
    } else {
        // Use a synthetic "rejected" report so the energy-tolerance
        // branch is suppressed; only force criteria can fire.
        let force_only_report = crate::minimizer::MinimizerStepReport {
            accepted: false,
            ..initial_report
        };
        minimizer.check_convergence(&force_only_report)
    };

    let max_iter = minimizer.max_iterations();
    let progress_every = (max_iter / 100).max(1);
    let phase_name = min.name.as_str();
    let mut final_report = initial_report;
    let mut iter_taken: u64 = 0;

    if convergence_reason.is_none() {
        for iter in 1..=max_iter {
            let report = {
                let constraint_arg: Option<&mut dyn crate::integrator::Constraint> =
                    match constraint.as_mut() {
                        Some(b) => Some(b.as_mut()),
                        None => None,
                    };
                minimizer
                    .step(buffers, sim_box, force_field, constraint_arg, timings)
                    .map_err(|e| match e {
                        MinimizerError::ForceField(ff) => {
                            (RunnerError::ForceField(ff), ExitPhase::Loop)
                        }
                        MinimizerError::Constraint(c) => {
                            (RunnerError::Constraint(c), ExitPhase::Loop)
                        }
                        other => (RunnerError::Minimizer(other), ExitPhase::Loop),
                    })?
            };
            iter_taken = iter;
            final_report = report;

            // Per-iteration minlog row at the configured cadence.
            if let Some(writer) = minlog_writer.as_mut() {
                if iter % min.output.minlog_every == 0 {
                    let mut lw = Duration::ZERO;
                    timed(&mut lw, || {
                        writer.write_row(
                            iter,
                            report.energy,
                            report.max_force,
                            report.step_size,
                            report.accepted,
                        )
                    })
                    .map_err(|e| (RunnerError::Minlog(e), ExitPhase::Loop))?;
                    timings.record_host(HostStage::LOG_WRITE, lw);
                    log_rows_written += 1;
                    last_logged_iter = Some(iter);
                }
            }
            // Periodic trajectory frame (accepted iterations only).
            if let Some(writer) = traj_writer.as_mut() {
                if report.accepted
                    && min.output.trajectory_every > 0
                    && iter % min.output.trajectory_every == 0
                {
                    let mut dl = Duration::ZERO;
                    timed(&mut dl, || frame.download_from(&*buffers)).map_err(|e| match e {
                        ParticleStateError::Gpu(g) => {
                            (RunnerError::Gpu(g), ExitPhase::Loop)
                        }
                        other => (RunnerError::ParticleState(other), ExitPhase::Loop),
                    })?;
                    timings.record_host(HostStage::DEVICE_TO_HOST_DOWNLOAD, dl);
                    let mut tw = Duration::ZERO;
                    timed(&mut tw, || {
                        write_traj_frame(writer, iter, 0.0, sim_box, &type_indices, &frame)
                    })
                    .map_err(|e| (RunnerError::Trajectory(e), ExitPhase::Loop))?;
                    timings.record_host(HostStage::TRAJECTORY_WRITE, tw);
                    frames_written += 1;
                }
            }
            // Convergence check (only after the iteration completed).
            convergence_reason = minimizer.check_convergence(&report);
            if convergence_reason.is_some() {
                break;
            }

            if progress_to_stdout && (iter % progress_every == 0 || iter == max_iter) {
                let rate = iter as f64 / phase_started.elapsed().as_secs_f64().max(1e-9);
                println!(
                    "[heddlemd] minimization `{phase_name}` iter {iter}/{max_iter} \
                     (E={:.6e} J, F_max={:.3e} N) — {rate:.1e} iters/sec",
                    report.energy, report.max_force,
                );
            }
        }
    }

    // Non-convergence is a hard error.
    let reason = match convergence_reason {
        Some(r) => r,
        None => {
            return Err((
                RunnerError::MinimizerNonConvergence {
                    phase: min.name.clone(),
                    iterations: iter_taken,
                    final_force: final_report.max_force,
                    final_step: final_report.step_size,
                },
                ExitPhase::Loop,
            ));
        }
    };

    // If the convergence iteration isn't already logged, emit a final row.
    if let Some(writer) = minlog_writer.as_mut() {
        if last_logged_iter != Some(iter_taken) {
            let mut lw = Duration::ZERO;
            timed(&mut lw, || {
                writer.write_row(
                    iter_taken,
                    final_report.energy,
                    final_report.max_force,
                    final_report.step_size,
                    final_report.accepted,
                )
            })
            .map_err(|e| (RunnerError::Minlog(e), ExitPhase::Loop))?;
            timings.record_host(HostStage::LOG_WRITE, lw);
            log_rows_written += 1;
        }
    }
    // Final convergence frame.
    if let Some(writer) = traj_writer.as_mut() {
        if iter_taken > 0 && iter_taken % min.output.trajectory_every.max(1) != 0 {
            let mut dl = Duration::ZERO;
            timed(&mut dl, || frame.download_from(&*buffers)).map_err(|e| match e {
                ParticleStateError::Gpu(g) => (RunnerError::Gpu(g), ExitPhase::Loop),
                other => (RunnerError::ParticleState(other), ExitPhase::Loop),
            })?;
            timings.record_host(HostStage::DEVICE_TO_HOST_DOWNLOAD, dl);
            let mut tw = Duration::ZERO;
            timed(&mut tw, || {
                write_traj_frame(writer, iter_taken, 0.0, sim_box, &type_indices, &frame)
            })
            .map_err(|e| (RunnerError::Trajectory(e), ExitPhase::Loop))?;
            timings.record_host(HostStage::TRAJECTORY_WRITE, tw);
            frames_written += 1;
        }
    }

    if let Some(writer) = minlog_writer.as_mut() {
        writer
            .flush()
            .map_err(|e| (RunnerError::Minlog(e), ExitPhase::Loop))?;
    }
    if let Some(writer) = traj_writer.as_mut() {
        writer
            .flush()
            .map_err(|e| (RunnerError::Trajectory(e), ExitPhase::Loop))?;
    }

    let phase_elapsed = phase_started.elapsed();
    timings.record_host(HostStage::TOTAL_RUNTIME, phase_elapsed);
    // Drain Timings and write the per-phase .timings file. To do this we
    // need to take ownership of `timings`; the outer loop owns it, so
    // finalize via a swap-in fresh instance.
    let report = std::mem::replace(timings, Timings::new(gpu)
        .map_err(|e| (RunnerError::Timings(e), ExitPhase::Setup))?)
        .finalize()
        .map_err(|e| (RunnerError::Timings(e), ExitPhase::Loop))?;
    write_timings_file(&min.output.timings_path, &report)
        .map_err(|e| (RunnerError::TimingsWriter(e), ExitPhase::Loop))?;

    Ok(PhaseSummary {
        name: min.name.clone(),
        n_steps: iter_taken,
        frames_written,
        log_rows_written,
        elapsed_micros: phase_elapsed.as_micros(),
        kind: "minimization",
        convergence: Some(reason.token()),
    })
}

/// Returns `true` iff every active slot for the phase reports
/// `graph_compatible = true`. Used by the runner to decide whether a
/// phase is eligible for CUDA graph capture. See `cuda-graphs.md`.
fn phase_slots_graph_compatible(
    setup: &SimulationSetup,
    phase: &crate::io::PhaseConfig,
) -> bool {
    // ForceField-level check: any potential whose `compute` uses a
    // secondary stream (e.g. SPME reciprocal) makes the phase
    // ineligible. Work on uncaptured streams runs immediately and is
    // not part of the resulting graph.
    if !setup.force_field.graph_compatible() {
        return false;
    }
    let int_ok = setup
        .registries
        .integrators
        .lookup(&phase.integrator.kind)
        .map(|b| b.graph_compatible(&phase.integrator.params))
        .unwrap_or(false);
    if !int_ok {
        return false;
    }
    if let Some(t) = phase.thermostat.as_ref() {
        let ok = setup
            .registries
            .thermostats
            .lookup(&t.kind)
            .map(|builder| builder.graph_compatible(&t.params))
            .unwrap_or(false);
        if !ok {
            return false;
        }
    }
    if let Some(b) = phase.barostat.as_ref() {
        let ok = setup
            .registries
            .barostats
            .lookup(&b.kind)
            .map(|builder| builder.graph_compatible(&b.params))
            .unwrap_or(false);
        if !ok {
            return false;
        }
    }
    // Constraints are looked up per `[[constraint_types]]` entry; if any
    // type's builder reports `graph_compatible = false`, the phase is
    // ineligible.
    for ct in &setup.config.constraint_types {
        let ok = setup
            .registries
            .constraint_types
            .lookup(&ct.kind)
            .map(|builder| builder.graph_compatible(&ct.params))
            .unwrap_or(false);
        if !ok {
            return false;
        }
    }
    true
}

/// Captures the per-step kernel sequence for a phase into an
/// executable CUDA graph. The sequence is:
///   1. `thermostat.apply_pre` (if any)
///   2. `run_step` with `RunStepOptions { run_neighbor_pre_step: false, .. }`
///   3. `thermostat.apply_post` (if any)
///   4. `barostat.apply` (if any)
///
/// The captured iteration counts as physical step 1: the device state
/// is left advanced by exactly one step after this call returns.
/// Launches the JIT-composed post-force per-particle kernel for one
/// physical step. Used both inside `capture_phase_graph` (where the
/// launch becomes a node in the captured graph) and inside the
/// per-step launch loop. Pre-populates the launch builder with the
/// common args, invokes each active slot's `bind_post_force_per_particle_args`
/// in canonical order, pushes the trailing `n` arg, then issues
/// `cuLaunchKernel`. Returns any timing or launch error to the
/// caller.
fn launch_composed_post_force(
    composed: &crate::forces::JitComposedPostForcePerParticle,
    buffers: &crate::gpu::ParticleBuffers,
    sim_box: &crate::pbc::SimulationBox,
    integrator: &dyn crate::integrator::Integrator,
    thermostat: &Option<Box<dyn crate::integrator::Thermostat>>,
    barostat: &Option<Box<dyn crate::integrator::Barostat>>,
    dt: Real,
    timings: &mut Timings,
) -> Result<(), RunnerError> {
    if buffers.particle_count() == 0 {
        return Ok(());
    }
    let mut builder = crate::forces::ForceLaunchBuilder::new();
    builder.push_device_buffer(&buffers.posq);
    builder.push_device_buffer(&buffers.images_x);
    builder.push_device_buffer(&buffers.images_y);
    builder.push_device_buffer(&buffers.images_z);
    builder.push_device_buffer(&buffers.velocities_x);
    builder.push_device_buffer(&buffers.velocities_y);
    builder.push_device_buffer(&buffers.velocities_z);
    builder.push_device_buffer(&buffers.forces_x);
    builder.push_device_buffer(&buffers.forces_y);
    builder.push_device_buffer(&buffers.forces_z);
    builder.push_device_buffer(&buffers.masses);
    builder.push_device_buffer(sim_box.lattice_device());
    let bind_ctx = crate::forces::PostForceBindContext {
        buffers,
        sim_box,
        dt,
    };
    integrator
        .post_force_per_particle()
        .expect("composed post-force kernel implies the integrator participates")
        .bind_post_force_per_particle_args(&bind_ctx, &mut builder);
    if let Some(t) = thermostat.as_ref().and_then(|t| t.post_force_per_particle()) {
        t.bind_post_force_per_particle_args(&bind_ctx, &mut builder);
    }
    if let Some(b) = barostat.as_ref().and_then(|b| b.post_force_per_particle()) {
        b.bind_post_force_per_particle_args(&bind_ctx, &mut builder);
    }
    let n_particles = buffers.particle_count() as u32;
    builder.push_scalar::<u32>(n_particles);
    timings
        .kernel_start(crate::timings::KernelStage::JIT_COMPOSED_POST_FORCE)
        .map_err(RunnerError::Timings)?;
    unsafe {
        composed
            .launch(n_particles, builder)
            .map_err(RunnerError::Gpu)?;
    }
    timings
        .kernel_stop(crate::timings::KernelStage::JIT_COMPOSED_POST_FORCE)
        .map_err(RunnerError::Timings)?;
    Ok(())
}

/// See `cuda-graphs.md` for the capture lifecycle.
// rq-76db55bb
/// Whether a physical step's force evaluation must compute the total
/// potential energy and virial. True only when the step produces a log
/// row or a barostat consumes the per-step virial; a trajectory frame
/// and the thermostat's kinetic-energy reduction need no force-kernel
/// scalars. The per-step launch loop and the graph replay loop both
/// select `AggregateLevel` / the captured graph through this predicate.
fn step_needs_force_scalars(log_due: bool, barostat_active: bool) -> bool {
    log_due || barostat_active
}

// rq-26dce0f6 rq-c6c56cdc
/// Whether a phase captures the forces-only graph alongside the
/// always-captured forces+scalars graph. A barostat consumes the virial
/// on every step, so a barostat phase evaluates scalars every step and
/// captures only the forces+scalars graph.
fn captures_forces_only_graph(barostat_active: bool) -> bool {
    !barostat_active
}

// rq-0d729ecb rq-2acc094a — periodic (Monte-Carlo) barostat helpers. A
// periodic barostat does no per-step work: it consumes no virial (so it
// does not force per-step scalars or suppress the forces-only graph) and
// runs a host-orchestrated move every `frequency` steps at a batch
// boundary.
fn barostat_move_frequency(
    barostat: &Option<Box<dyn crate::integrator::Barostat>>,
) -> Option<u32> {
    match barostat.as_ref().map(|b| b.periodicity()) {
        Some(crate::integrator::BarostatPeriodicity::EveryNSteps(n)) => Some(n),
        _ => None,
    }
}

/// Whether a barostat couples to the dynamics on every step (and so
/// consumes the per-step virial). False for an absent or periodic
/// barostat.
fn barostat_couples_per_step(
    barostat: &Option<Box<dyn crate::integrator::Barostat>>,
) -> bool {
    matches!(
        barostat.as_ref().map(|b| b.periodicity()),
        Some(crate::integrator::BarostatPeriodicity::EveryStep)
    )
}

/// Packed-neighbour buffer capacities, used to detect whether a barostat
/// move's trial force evaluation grew (and so reallocated) a buffer —
/// which invalidates the captured graph's device pointers. A move that
/// leaves the capacities unchanged needs no graph re-capture.
fn packed_capacities(force_field: &ForceField) -> Option<(u32, u32)> {
    force_field
        .neighbor_list
        .as_ref()
        .and_then(|nl| nl.packed.as_ref())
        .map(|p| (p.interacting_tiles_capacity, p.single_pairs_capacity))
}

// rq-766c88fb rq-e35fa835
/// Captures the phase's executable graphs: the forces+scalars graph
/// always, and the forces-only graph unless a barostat is active. Both
/// are recorded from the same pre-capture device state; stream capture
/// records without executing, so neither advances the simulation. See
/// `cuda-graphs.md` *Capture Lifecycle*.
#[allow(clippy::too_many_arguments)]
fn capture_phase_graph(
    buffers: &mut crate::gpu::ParticleBuffers,
    sim_box: &mut crate::pbc::SimulationBox,
    force_field: &mut crate::forces::ForceField,
    integrator: &mut dyn crate::integrator::Integrator,
    thermostat: &mut Option<Box<dyn crate::integrator::Thermostat>>,
    barostat: &mut Option<Box<dyn crate::integrator::Barostat>>,
    constraint: &mut Option<Box<dyn crate::integrator::Constraint>>,
    supports_constraints: bool,
    dt: Real,
    timings: &mut Timings,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    composed_post_force: Option<&crate::forces::JitComposedPostForcePerParticle>,
    post_force_substep_index: Option<usize>,
) -> Result<crate::gpu::GraphLoop, crate::gpu::GraphError> {
    let forces_and_scalars = capture_one_graph(
        buffers,
        sim_box,
        force_field,
        integrator,
        thermostat,
        barostat,
        constraint,
        supports_constraints,
        dt,
        timings,
        device,
        composed_post_force,
        post_force_substep_index,
        true,
        GraphVariant::ForcesAndScalars,
    )?;
    // A barostat consumes the per-step virial inside the captured
    // sequence, so a barostat phase evaluates scalars on every step and
    // does not capture a forces-only graph.
    let forces_only = if captures_forces_only_graph(barostat_couples_per_step(barostat)) {
        Some(capture_one_graph(
            buffers,
            sim_box,
            force_field,
            integrator,
            thermostat,
            barostat,
            constraint,
            supports_constraints,
            dt,
            timings,
            device,
            composed_post_force,
            post_force_substep_index,
            false,
            GraphVariant::ForcesOnly,
        )?)
    } else {
        None
    };
    Ok(crate::gpu::GraphLoop {
        forces_and_scalars,
        forces_only,
    })
}

/// Captures and instantiates a single one-step graph with the force
/// evaluation at the `AggregateLevel` selected by `needs_scalars`, and
/// commits its per-stage `kernel_stop` counts to `variant`.
// rq-766c88fb
#[allow(clippy::too_many_arguments)]
fn capture_one_graph(
    buffers: &mut crate::gpu::ParticleBuffers,
    sim_box: &mut crate::pbc::SimulationBox,
    force_field: &mut crate::forces::ForceField,
    integrator: &mut dyn crate::integrator::Integrator,
    thermostat: &mut Option<Box<dyn crate::integrator::Thermostat>>,
    barostat: &mut Option<Box<dyn crate::integrator::Barostat>>,
    constraint: &mut Option<Box<dyn crate::integrator::Constraint>>,
    supports_constraints: bool,
    dt: Real,
    timings: &mut Timings,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
    composed_post_force: Option<&crate::forces::JitComposedPostForcePerParticle>,
    post_force_substep_index: Option<usize>,
    needs_scalars: bool,
    variant: GraphVariant,
) -> Result<crate::gpu::CudaGraphExec, crate::gpu::GraphError> {
    use crate::gpu::{CaptureMode, begin_stream_capture, end_stream_capture};
    // Settle any outstanding `Timings` event pairs from the warm-up
    // step before capture begins; `event::synchronize` calls inside
    // `kernel_start` would otherwise invalidate the capture region.
    if let Err(e) = timings.drain_outstanding() {
        eprintln!("warning: timings drain before graph capture failed: {e}; falling back");
        return Err(crate::gpu::GraphError::BeginCaptureFailed(
            cudarc::driver::DriverError(
                cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY,
            ),
        ));
    }
    timings.begin_capture();
    // `ThreadLocal` restricts capture-mode side effects to the
    // calling thread; without this, every other thread sharing the
    // CUDA primary context fails routine ops like `cuMemAllocAsync`
    // with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` for the duration
    // of the capture. The runner's per-phase loop is single-threaded,
    // so the broader restrictions of `Global` are not needed.
    begin_stream_capture(device, CaptureMode::ThreadLocal)?;
    let mut inner_failure: Option<String> = None;
    if let Some(t) = thermostat.as_mut() {
        if let Err(e) = t.apply_pre(buffers, dt, timings) {
            inner_failure = Some(format!("thermostat.apply_pre: {e}"));
        }
    }
    if inner_failure.is_none() {
        let constraint_arg: Option<&mut dyn crate::integrator::Constraint> = match constraint
            .as_mut()
        {
            Some(c) => Some(c.as_mut()),
            None => None,
        };
        // `needs_scalars` selects the force evaluation's `AggregateLevel`:
        // `true` records the forces+scalars (`_fev`) graph, `false` the
        // forces-only (`_f`) graph. The replay loop launches the
        // forces-only graph on steps that need no scalars (see
        // `cuda-graphs.md` *Batched Replay Loop*).
        let result = if let (Some(_), Some(skip_idx)) =
            (composed_post_force, post_force_substep_index)
        {
            crate::integrator::run_step(
                integrator,
                buffers,
                sim_box,
                force_field,
                constraint_arg,
                dt,
                timings,
                crate::integrator::RunStepOptions {
                    run_neighbor_pre_step: false,
                    skip_substep_index: Some(skip_idx),
                    install_constraint_hooks: supports_constraints,
                    runner_needs_scalars: needs_scalars,
                },
            )
        } else {
            crate::integrator::run_step(
                integrator,
                buffers,
                sim_box,
                force_field,
                constraint_arg,
                dt,
                timings,
                crate::integrator::RunStepOptions {
                    run_neighbor_pre_step: false,
                    install_constraint_hooks: supports_constraints,
                    runner_needs_scalars: needs_scalars,
                    ..Default::default()
                },
            )
        };
        if let Err(e) = result {
            inner_failure = Some(format!("run_step: {e:?}"));
        }
    }
    if inner_failure.is_none() {
        if let Some(t) = thermostat.as_mut() {
            if let Err(e) = t.apply_post(buffers, dt, timings) {
                inner_failure = Some(format!("thermostat.apply_post: {e}"));
            }
        }
    }
    if inner_failure.is_none() {
        if let Some(b) = barostat.as_mut() {
            if let Err(e) = b.apply(buffers, sim_box, dt, timings) {
                inner_failure = Some(format!("barostat.apply: {e}"));
            }
        }
    }
    if inner_failure.is_none() {
        if let Some(composed) = composed_post_force {
            if let Err(e) = launch_composed_post_force(
                composed,
                buffers,
                sim_box,
                integrator,
                thermostat,
                barostat,
                dt,
                timings,
            ) {
                inner_failure = Some(format!("composed post-force launch: {e:?}"));
            }
        }
    }
    // Always end capture, even on inner failure — a captured stream
    // must be closed to avoid leaving the device in capture mode.
    let graph = end_stream_capture(device)?;
    timings.end_capture(variant);
    if let Some(reason) = inner_failure {
        eprintln!("warning: cuda graph capture inner sequence failed ({reason}); falling back");
        return Err(crate::gpu::GraphError::EndCaptureFailed(
            cudarc::driver::DriverError(
                cudarc::driver::sys::CUresult::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED,
            ),
        ));
    }
    graph.instantiate()
}

/// Per-step launch loop over physical steps `start_step..=n_steps`.
/// Used both for graph-ineligible phases (`start_step = 1`) and as the
/// fallback when a mid-phase graph re-capture fails (`start_step` is the
/// first un-run step). See `cuda-graphs.md` *Neighbor-List Pre-Step
/// Decomposition*.
#[allow(clippy::too_many_arguments)]
fn run_per_step_range(
    start_step: u64,
    n_steps: u64,
    setup: &mut SimulationSetup,
    phase: &crate::io::PhaseConfig,
    integrator: &mut Box<dyn crate::integrator::Integrator>,
    thermostat: &mut Option<Box<dyn crate::integrator::Thermostat>>,
    barostat: &mut Option<Box<dyn crate::integrator::Barostat>>,
    constraint: &mut Option<Box<dyn crate::integrator::Constraint>>,
    supports_constraints: bool,
    dt: Real,
    composed_post_force: Option<&crate::forces::JitComposedPostForcePerParticle>,
    post_force_substep_index: Option<usize>,
    timings: &mut Timings,
    frame: &mut ParticleState,
    traj_writer: &mut Option<TrajectoryWriter>,
    log_writer: &mut Option<LogWriter>,
    pe_scratch: &mut Option<cudarc::driver::CudaSlice<Real>>,
    type_indices: &[u32],
    n_thermal_dof: u32,
    log_extra_columns: &[(&'static str, crate::units::Dimension)],
    phase_started: Instant,
    phase_name: &str,
    progress_to_stdout: bool,
    progress_every: u64,
    frames_written: &mut u64,
    log_rows_written: &mut u64,
) -> Result<(), (RunnerError, ExitPhase)> {
    for step in start_step..=n_steps {
        if let Some(t) = thermostat.as_mut() {
            t.apply_pre(&mut setup.buffers, dt, &mut *timings)
                .map_err(|e| (RunnerError::Thermostat(e), ExitPhase::Loop))?;
        }
        {
            let constraint_arg: Option<&mut dyn crate::integrator::Constraint> =
                match constraint.as_mut() {
                    Some(b) => Some(b.as_mut()),
                    None => None,
                };
            // rq-76db55bb — the force evaluation computes total PE and
            // virial only when the step produces a log row or a barostat
            // consumes the virial. A trajectory frame carries positions
            // and velocities (no force-kernel scalars), and the
            // thermostat reduces kinetic energy independently, so neither
            // forces a scalars step. The graph replay loop selects its
            // forces-only / forces+scalars graphs on the same condition.
            let log_due = phase.output.log_every > 0 && step % phase.output.log_every == 0;
            // rq-03a5a290 — evaluate scalars on the step preceding a
            // periodic-barostat move so the move reads the current-config
            // energy without a redundant force evaluation.
            let is_move_boundary = barostat_move_frequency(barostat)
                .is_some_and(|f| f > 0 && step % f as u64 == 0);
            let runner_needs_scalars =
                step_needs_force_scalars(log_due, barostat_couples_per_step(barostat))
                    || is_move_boundary;
            let result = if let Some(skip_idx) = post_force_substep_index {
                crate::integrator::run_step(
                    integrator.as_mut(),
                    &mut setup.buffers,
                    &mut setup.sim_box,
                    &mut setup.force_field,
                    constraint_arg,
                    dt,
                    &mut *timings,
                    crate::integrator::RunStepOptions {
                        skip_substep_index: Some(skip_idx),
                        install_constraint_hooks: supports_constraints,
                        runner_needs_scalars,
                        ..Default::default()
                    },
                )
            } else {
                crate::integrator::run_step(
                    integrator.as_mut(),
                    &mut setup.buffers,
                    &mut setup.sim_box,
                    &mut setup.force_field,
                    constraint_arg,
                    dt,
                    &mut *timings,
                    crate::integrator::RunStepOptions {
                        install_constraint_hooks: supports_constraints,
                        runner_needs_scalars,
                        ..Default::default()
                    },
                )
            };
            result.map_err(|e| {
                let runner_err = match e {
                    crate::integrator::StepError::Integrator(e) => RunnerError::Integrator(e),
                    crate::integrator::StepError::ForceField(e) => RunnerError::ForceField(e),
                    crate::integrator::StepError::Constraint(e) => RunnerError::Constraint(e),
                    crate::integrator::StepError::IntegratorRejectsConstraint { reason } => {
                        unreachable!("run_step returned IntegratorRejectsConstraint ({reason})")
                    }
                    crate::integrator::StepError::MissingPostForcePerParticleFragment {
                        kind,
                        label,
                    } => RunnerError::MissingPostForcePerParticleFragment { kind, label },
                    crate::integrator::StepError::PostForceFragmentCompileFailed { log } => {
                        RunnerError::PostForceFragmentCompileFailed { log }
                    }
                    crate::integrator::StepError::PostForceFragmentLoadFailed(e) => {
                        RunnerError::PostForceFragmentLoadFailed(e)
                    }
                };
                (runner_err, ExitPhase::Loop)
            })?;
        }
        if let Some(t) = thermostat.as_mut() {
            t.apply_post(&mut setup.buffers, dt, &mut *timings)
                .map_err(|e| (RunnerError::Thermostat(e), ExitPhase::Loop))?;
        }
        if let Some(b) = barostat.as_mut() {
            b.apply(&mut setup.buffers, &mut setup.sim_box, dt, &mut *timings)
                .map_err(|e| (RunnerError::Barostat(e), ExitPhase::Loop))?;
        }
        if let Some(composed) = composed_post_force {
            launch_composed_post_force(
                composed,
                &setup.buffers,
                &setup.sim_box,
                integrator.as_ref(),
                thermostat,
                barostat,
                dt,
                &mut *timings,
            )
            .map_err(|e| (e, ExitPhase::Loop))?;
        }

        // rq-03a5a290 — a periodic (Monte-Carlo) barostat runs its
        // host-orchestrated move every `frequency` steps, after the
        // dynamics step and before this step's output. The neighbour-
        // checking force evaluations inside the move (and the next step's
        // full `step` call) rebuild the neighbour list against the moved
        // box.
        if let Some(freq) = barostat_move_frequency(barostat) {
            if freq > 0 && step % freq as u64 == 0 {
                if let Some(b) = barostat.as_mut() {
                    b.apply_move(
                        &mut setup.force_field,
                        &mut setup.buffers,
                        &mut setup.sim_box,
                        None,
                        dt,
                        &mut *timings,
                    )
                    .map_err(|e| (RunnerError::Barostat(e), ExitPhase::Loop))?;
                }
            }
        }

        let want_traj =
            phase.output.trajectory_every > 0 && step % phase.output.trajectory_every == 0;
        let want_log = phase.output.log_every > 0 && step % phase.output.log_every == 0;
        if want_traj || want_log {
            let mut dl = Duration::ZERO;
            timed(&mut dl, || frame.download_from(&setup.buffers)).map_err(|e| match e {
                ParticleStateError::Gpu(g) => (RunnerError::Gpu(g), ExitPhase::Loop),
                other => (RunnerError::ParticleState(other), ExitPhase::Loop),
            })?;
            timings.record_host(HostStage::DEVICE_TO_HOST_DOWNLOAD, dl);
            if barostat.is_some() {
                setup
                    .sim_box
                    .flush_from_device()
                    .map_err(|e| (RunnerError::SimulationBox(e), ExitPhase::Loop))?;
            }
        }
        if want_traj {
            let writer = traj_writer.as_mut().expect("traj_writer enabled");
            let mut tw = Duration::ZERO;
            timed(&mut tw, || {
                write_traj_frame(writer, step, phase.dt, &setup.sim_box, type_indices, frame)
            })
            .map_err(|e| (RunnerError::Trajectory(e), ExitPhase::Loop))?;
            timings.record_host(HostStage::TRAJECTORY_WRITE, tw);
            *frames_written += 1;
        }
        if want_log {
            let writer = log_writer.as_mut().expect("log_writer enabled");
            if let Some(t) = thermostat.as_mut() {
                t.flush_pending_injection(&setup.gpu.device)
                    .map_err(|e| (RunnerError::Thermostat(e), ExitPhase::Loop))?;
            }
            if let Some(b) = barostat.as_mut() {
                b.flush_pending_injection(&setup.gpu.device)
                    .map_err(|e| (RunnerError::Barostat(e), ExitPhase::Loop))?;
            }
            let ke = compute_kinetic_energy(
                &frame.masses,
                &frame.velocities_x,
                &frame.velocities_y,
                &frame.velocities_z,
            );
            let t = compute_temperature(ke, n_thermal_dof);
            let time = (step as f64) * phase.dt;
            let extras = if log_extra_columns.is_empty() {
                Vec::new()
            } else {
                let scratch = pe_scratch
                    .as_mut()
                    .expect("pe_scratch allocated when log_extra_columns non-empty");
                timings
                    .kernel_start(KernelStage::POTENTIAL_ENERGY_REDUCE)
                    .map_err(|e| (RunnerError::Timings(e), ExitPhase::Loop))?;
                let pe = compute_total_potential_energy(&mut setup.buffers, scratch)
                    .map_err(|g| (RunnerError::Gpu(g), ExitPhase::Loop))?;
                timings
                    .kernel_stop(KernelStage::POTENTIAL_ENERGY_REDUCE)
                    .map_err(|e| (RunnerError::Timings(e), ExitPhase::Loop))?;
                collect_log_extras(
                    integrator.as_ref(),
                    thermostat.as_deref(),
                    barostat.as_deref(),
                    ke,
                    pe as f64,
                )
            };
            let mut lw = Duration::ZERO;
            timed(&mut lw, || writer.write_row(step, time, ke, t, &extras))
                .map_err(|e| (RunnerError::Log(e), ExitPhase::Loop))?;
            timings.record_host(HostStage::LOG_WRITE, lw);
            *log_rows_written += 1;
        }

        // rq-73fbb111
        if progress_to_stdout && (step % progress_every == 0 || step == n_steps) {
            let pct = 100.0 * step as f64 / n_steps.max(1) as f64;
            let rate = step as f64 / phase_started.elapsed().as_secs_f64().max(1e-9);
            println!(
                "[heddlemd] phase `{phase_name}` step {step}/{n_steps} ({pct:.1}%) — {rate:.1e} steps/sec"
            );
        }
    }
    Ok(())
}

/// Runs the batched graph-replay loop, replaying the captured graph in
/// batches up to `phase.n_steps`. When a batch-boundary rebuild
/// reallocates a packed-neighbour buffer the phase graph is re-captured
/// in place (see `cuda-graphs.md` *Neighbor-List Pre-Step
/// Decomposition*). Returns `Some(resume_step)` when a re-capture failed
/// and the caller must finish the phase on the per-step launch loop from
/// `resume_step`; `None` on normal completion.
#[allow(clippy::too_many_arguments)]
fn run_batched_graph_loop(
    setup: &mut SimulationSetup,
    phase: &crate::io::PhaseConfig,
    _phase_index: usize,
    start_step: u64,
    graph_loop: &mut crate::gpu::GraphLoop,
    integrator: &mut dyn crate::integrator::Integrator,
    thermostat: &mut Option<Box<dyn crate::integrator::Thermostat>>,
    barostat: &mut Option<Box<dyn crate::integrator::Barostat>>,
    constraint: &mut Option<Box<dyn crate::integrator::Constraint>>,
    supports_constraints: bool,
    dt: Real,
    composed_post_force: Option<&crate::forces::JitComposedPostForcePerParticle>,
    post_force_substep_index: Option<usize>,
    timings: &mut Timings,
    frame: &mut ParticleState,
    traj_writer: &mut Option<TrajectoryWriter>,
    log_writer: &mut Option<LogWriter>,
    pe_scratch: &mut Option<cudarc::driver::CudaSlice<Real>>,
    type_indices: &[u32],
    n_thermal_dof: u32,
    log_extra_columns: &[(&'static str, crate::units::Dimension)],
    phase_started: Instant,
    phase_name: &str,
    progress_to_stdout: bool,
    progress_every: u64,
    frames_written: &mut u64,
    log_rows_written: &mut u64,
) -> Result<Option<u64>, (RunnerError, ExitPhase)> {
    let n_steps = phase.n_steps;
    let log_every = phase.output.log_every;
    let traj_every = phase.output.trajectory_every;
    let batch_size = setup.config.simulation.graph_batch_size as u64;
    // A clone of the default-stream device handle, used to re-capture the
    // phase graph without aliasing the `&mut setup` borrow.
    let device = setup.gpu.device.clone();

    // The captured graph records a one-step kernel sequence with no
    // work executed during capture (`CU_STREAM_CAPTURE_MODE_GLOBAL`
    // semantics). The batched loop launches it for every physical step
    // from `start_step + 1` to `n_steps`; the first `start_step` steps
    // ran on the instrumented per-step path (graph-timing calibration).
    let mut step: u64 = start_step;

    // rq-0d729ecb — a periodic (Monte-Carlo) barostat runs a host move
    // every `move_frequency` steps; batches are bounded so each move lands
    // exactly on a batch boundary.
    let move_frequency = barostat_move_frequency(barostat).map(|f| f as u64);

    while step < n_steps {
        let remaining = n_steps - step;
        let next_log = if log_every > 0 {
            log_every - (step % log_every)
        } else {
            remaining
        };
        let next_traj = if traj_every > 0 {
            traj_every - (step % traj_every)
        } else {
            remaining
        };
        let next_move = match move_frequency {
            Some(f) if f > 0 => f - (step % f),
            _ => remaining,
        };
        let batch = batch_size
            .min(remaining)
            .min(next_log)
            .min(next_traj)
            .min(next_move);
        let barostat_active = barostat_couples_per_step(barostat);
        let mut forces_only_launches: u32 = 0;
        let mut forces_and_scalars_launches: u32 = 0;
        for i in 1..=batch {
            let s = step + i;
            // rq-76db55bb — a step needs force-kernel scalars (total PE +
            // virial) only when it produces a log row or a barostat
            // consumes the per-step virial. Other steps replay the
            // forces-only graph.
            let needs_scalars =
                step_needs_force_scalars(log_every > 0 && s % log_every == 0, barostat_active)
                    || move_frequency.is_some_and(|f| f > 0 && s % f == 0);
            graph_loop.launch(needs_scalars).map_err(|e| {
                (
                    RunnerError::Gpu(crate::gpu::GpuError(match e {
                        crate::gpu::GraphError::LaunchFailed(d) => d,
                        crate::gpu::GraphError::BeginCaptureFailed(d) => d,
                        crate::gpu::GraphError::EndCaptureFailed(d) => d,
                        crate::gpu::GraphError::InstantiateFailed(d) => d,
                        crate::gpu::GraphError::DestroyFailed(d) => d,
                    })),
                    ExitPhase::Loop,
                )
            })?;
            if needs_scalars {
                forces_and_scalars_launches += 1;
            } else {
                forces_only_launches += 1;
            }
        }
        // rq-9ec19227 — advance per-stage sample counts per graph variant,
        // so a stage present only in the forces+scalars graph (the scalar
        // reductions, the `_fev` pair-force kernel) accrues samples only
        // from the scalar steps. Durations are the calibrated
        // representatives (see `cuda-graphs.md`).
        if forces_only_launches > 0 {
            timings.record_graph_replays(GraphVariant::ForcesOnly, forces_only_launches);
        }
        if forces_and_scalars_launches > 0 {
            timings
                .record_graph_replays(GraphVariant::ForcesAndScalars, forces_and_scalars_launches);
        }
        step += batch;

        // rq-03a5a290 — a periodic (Monte-Carlo) barostat runs its
        // host-orchestrated move at the move boundary, before the
        // neighbour pre-step so that the moved box is rebuilt against. The
        // captured graph reads the box from the persistent lattice device
        // buffer (pointer stable), so a move alone needs no re-capture;
        // only a move whose trial force evaluation grew a packed-neighbour
        // buffer invalidates the captured device pointers and forces one.
        let mut move_reallocated = false;
        if let Some(f) = move_frequency {
            if f > 0 && step % f == 0 && step <= n_steps {
                if let Some(b) = barostat.as_mut() {
                    let caps_before = packed_capacities(&setup.force_field);
                    b.apply_move(
                        &mut setup.force_field,
                        &mut setup.buffers,
                        &mut setup.sim_box,
                        None,
                        dt,
                        timings,
                    )
                    .map_err(|e| (RunnerError::Barostat(e), ExitPhase::Loop))?;
                    move_reallocated = caps_before != packed_capacities(&setup.force_field);
                }
            }
        }

        // Displacement check + neighbor-list rebuild (if triggered)
        // run at every batch boundary, outside the captured graph.
        let reallocated = setup
            .force_field
            .run_neighbor_pre_step(&mut setup.buffers, &setup.sim_box, timings)
            .map_err(|e| (RunnerError::ForceField(e), ExitPhase::Loop))?;

        // A rebuild that grew a packed-neighbour buffer reallocated it,
        // invalidating the device pointers and single-pair grid
        // dimensions baked into the captured graph. Re-capture the phase
        // graph against the new buffers before the next batch. On a
        // re-capture driver error, finish the phase on the per-step
        // launch loop from the next un-run step. rq-67a09135 rq-1217c816
        if (reallocated || move_reallocated) && step < n_steps {
            match capture_phase_graph(
                &mut setup.buffers,
                &mut setup.sim_box,
                &mut setup.force_field,
                integrator,
                thermostat,
                barostat,
                constraint,
                supports_constraints,
                dt,
                timings,
                &device,
                composed_post_force,
                post_force_substep_index,
            ) {
                Ok(new_loop) => {
                    *graph_loop = new_loop;
                }
                Err(e) => {
                    eprintln!(
                        "warning: cuda graph capture failed for phase `{phase_name}`: {e}; falling back to per-step launches"
                    );
                    return Ok(Some(step + 1));
                }
            }
        }

        handle_step_output(
            setup,
            phase,
            step,
            thermostat,
            barostat,
            timings,
            frame,
            traj_writer,
            log_writer,
            pe_scratch,
            type_indices,
            n_thermal_dof,
            log_extra_columns,
            frames_written,
            log_rows_written,
        )?;

        if progress_to_stdout && (step % progress_every == 0 || step == n_steps) {
            let pct = 100.0 * step as f64 / n_steps.max(1) as f64;
            let rate = step as f64 / phase_started.elapsed().as_secs_f64().max(1e-9);
            println!(
                "[heddlemd] phase `{phase_name}` step {step}/{n_steps} ({pct:.1}%) — {rate:.1e} steps/sec"
            );
        }
    }
    Ok(None)
}

/// Per-step output handler shared by the per-step launch loop and the
/// batched graph-replay loop. Downloads the host frame, flushes the
/// simulation box, drains slot diagnostic accumulators, and writes
/// log / trajectory rows if `step` aligns with the configured cadence.
#[allow(clippy::too_many_arguments)]
fn handle_step_output(
    setup: &mut SimulationSetup,
    phase: &crate::io::PhaseConfig,
    step: u64,
    thermostat: &mut Option<Box<dyn crate::integrator::Thermostat>>,
    barostat: &mut Option<Box<dyn crate::integrator::Barostat>>,
    timings: &mut Timings,
    frame: &mut ParticleState,
    traj_writer: &mut Option<TrajectoryWriter>,
    log_writer: &mut Option<LogWriter>,
    pe_scratch: &mut Option<cudarc::driver::CudaSlice<Real>>,
    type_indices: &[u32],
    n_thermal_dof: u32,
    log_extra_columns: &[(&'static str, crate::units::Dimension)],
    frames_written: &mut u64,
    log_rows_written: &mut u64,
) -> Result<(), (RunnerError, ExitPhase)> {
    let want_traj =
        phase.output.trajectory_every > 0 && step % phase.output.trajectory_every == 0;
    let want_log = phase.output.log_every > 0 && step % phase.output.log_every == 0;
    if !(want_traj || want_log) {
        return Ok(());
    }
    let mut dl = Duration::ZERO;
    timed(&mut dl, || frame.download_from(&setup.buffers)).map_err(|e| match e {
        ParticleStateError::Gpu(g) => (RunnerError::Gpu(g), ExitPhase::Loop),
        other => (RunnerError::ParticleState(other), ExitPhase::Loop),
    })?;
    timings.record_host(HostStage::DEVICE_TO_HOST_DOWNLOAD, dl);
    if barostat.is_some() {
        setup
            .sim_box
            .flush_from_device()
            .map_err(|e| (RunnerError::SimulationBox(e), ExitPhase::Loop))?;
    }
    if want_traj {
        let writer = traj_writer.as_mut().expect("traj_writer enabled");
        let mut tw = Duration::ZERO;
        timed(&mut tw, || {
            write_traj_frame(writer, step, phase.dt, &setup.sim_box, type_indices, frame)
        })
        .map_err(|e| (RunnerError::Trajectory(e), ExitPhase::Loop))?;
        timings.record_host(HostStage::TRAJECTORY_WRITE, tw);
        *frames_written += 1;
    }
    if want_log {
        let writer = log_writer.as_mut().expect("log_writer enabled");
        if let Some(t) = thermostat.as_mut() {
            t.flush_pending_injection(&setup.gpu.device)
                .map_err(|e| (RunnerError::Thermostat(e), ExitPhase::Loop))?;
        }
        if let Some(b) = barostat.as_mut() {
            b.flush_pending_injection(&setup.gpu.device)
                .map_err(|e| (RunnerError::Barostat(e), ExitPhase::Loop))?;
        }
        let ke = compute_kinetic_energy(
            &frame.masses,
            &frame.velocities_x,
            &frame.velocities_y,
            &frame.velocities_z,
        );
        let t = compute_temperature(ke, n_thermal_dof);
        let time = (step as f64) * phase.dt;
        let extras = if log_extra_columns.is_empty() {
            Vec::new()
        } else {
            let scratch = pe_scratch
                .as_mut()
                .expect("pe_scratch allocated when log_extra_columns non-empty");
            timings
                .kernel_start(KernelStage::POTENTIAL_ENERGY_REDUCE)
                .map_err(|e| (RunnerError::Timings(e), ExitPhase::Loop))?;
            let pe = compute_total_potential_energy(&mut setup.buffers, scratch)
                .map_err(|g| (RunnerError::Gpu(g), ExitPhase::Loop))?;
            timings
                .kernel_stop(KernelStage::POTENTIAL_ENERGY_REDUCE)
                .map_err(|e| (RunnerError::Timings(e), ExitPhase::Loop))?;
            collect_log_extras_from_slots(
                thermostat.as_deref(),
                barostat.as_deref(),
                ke,
                pe as f64,
            )
        };
        let mut lw = Duration::ZERO;
        timed(&mut lw, || writer.write_row(step, time, ke, t, &extras))
            .map_err(|e| (RunnerError::Log(e), ExitPhase::Loop))?;
        timings.record_host(HostStage::LOG_WRITE, lw);
        *log_rows_written += 1;
    }
    Ok(())
}

/// Variant of `collect_log_extras` that omits the integrator's extras
/// — used by the batched-replay output handler, which does not have a
/// borrow on the integrator. Integrator log columns are absent for the
/// graph-mode case in this implementation; integrators that publish
/// log columns require their values to be drained device-side at
/// batch boundaries.
fn collect_log_extras_from_slots(
    thermostat: Option<&dyn crate::integrator::Thermostat>,
    barostat: Option<&dyn crate::integrator::Barostat>,
    ke: f64,
    pe: f64,
) -> Vec<f64> {
    let mut extras: Vec<f64> = Vec::new();
    if let Some(t) = thermostat {
        extras.extend(t.log_column_values(ke, pe));
    }
    if let Some(b) = barostat {
        extras.extend(b.log_column_values(ke, pe));
    }
    extras
}

// Concatenate the diagnostic-column values from every configured slot in
// dispatch order (integrator, thermostat, barostat). Mirrors the
// header-construction order in `LogWriter::open(...)` above.
fn collect_log_extras(
    integrator: &dyn crate::integrator::Integrator,
    thermostat: Option<&dyn crate::integrator::Thermostat>,
    barostat: Option<&dyn crate::integrator::Barostat>,
    ke: f64,
    pe: f64,
) -> Vec<f64> {
    let mut extras = integrator.log_column_values(ke, pe);
    if let Some(t) = thermostat {
        extras.extend(t.log_column_values(ke, pe));
    }
    if let Some(b) = barostat {
        extras.extend(b.log_column_values(ke, pe));
    }
    extras
}

fn write_traj_frame(
    writer: &mut TrajectoryWriter,
    step: u64,
    dt: f64,
    sim_box: &crate::pbc::SimulationBox,
    type_indices: &[u32],
    frame: &ParticleState,
) -> Result<(), TrajectoryWriterError> {
    let n = frame.particle_count();
    let traj_velocities = if writer.include_velocities() {
        Some((
            &frame.velocities_x[..n],
            &frame.velocities_y[..n],
            &frame.velocities_z[..n],
        ))
    } else {
        None
    };
    let traj_images = if writer.include_images() {
        Some((
            &frame.images_x[..n],
            &frame.images_y[..n],
            &frame.images_z[..n],
        ))
    } else {
        None
    };
    writer.write_frame(
        step,
        dt,
        sim_box,
        &type_indices[..n],
        &frame.positions_x[..n],
        &frame.positions_y[..n],
        &frame.positions_z[..n],
        traj_velocities,
        traj_images,
    )
}

// rq-2be8ef35 rq-1b7680ad rq-2249f685 rq-8e239d36 rq-e6552df6
fn generate_velocities(
    n: usize,
    n_constraints: usize,
    temperature: f64,
    seed: u64,
    masses: &[f64],
) -> (Vec<Real>, Vec<Real>, Vec<Real>) {
    let mut vx = vec![0.0; n];
    let mut vy = vec![0.0; n];
    let mut vz = vec![0.0; n];
    if temperature == 0.0 || n == 0 {
        return (vx, vy, vz);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for i in 0..n {
        // k_B = 1 in atomic units; temperature is k_B · T in Hartrees.
        let sigma = (temperature / masses[i]).sqrt();
        for axis in 0..3 {
            let u1 = 1.0 - rng.r#gen::<f64>(); // (0, 1]
            let u2 = rng.r#gen::<f64>();        // [0, 1)
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let v = (z * sigma) as Real;
            match axis {
                0 => vx[i] = v,
                1 => vy[i] = v,
                _ => vz[i] = v,
            }
        }
    }

    // Momentum subtraction.
    let total_mass: f64 = masses.iter().copied().sum();
    if total_mass > 0.0 {
        for axis in 0..3 {
            let slice: &mut [Real] = match axis {
                0 => &mut vx,
                1 => &mut vy,
                _ => &mut vz,
            };
            let p: f64 = masses
                .iter()
                .zip(slice.iter())
                .map(|(m, v)| (*m) * (*v as f64))
                .sum();
            let v_com = p / total_mass;
            for v in slice.iter_mut() {
                *v = ((*v as f64) - v_com) as Real;
            }
        }
    }

    // Rescale every velocity by a single scalar so the realised kinetic
    // energy matches the equipartition target
    // `(N_thermal_dof / 2) * k_B * T`, where
    // `N_thermal_dof = max(0, 3N − n_constraints − 3)` is the
    // constraint- and COM-removed thermal degrees of freedom — the
    // same convention used by `compute_temperature` and by the
    // momentum-conserving thermostats. When the system has no
    // thermal DOFs remaining (N == 0 or 1, or pathologically
    // over-constrained), every velocity is set to exactly zero.
    let n_thermal_dof: i64 = (3 * n as i64) - n_constraints as i64 - 3;
    if n_thermal_dof <= 0 {
        for i in 0..n {
            vx[i] = 0.0;
            vy[i] = 0.0;
            vz[i] = 0.0;
        }
    } else {
        let ke: f64 = (0..n)
            .map(|i| {
                0.5 * masses[i]
                    * ((vx[i] as f64).powi(2)
                        + (vy[i] as f64).powi(2)
                        + (vz[i] as f64).powi(2))
            })
            .sum();
        if ke > 0.0 {
            // k_B = 1 in atomic units; temperature is k_B · T in Hartrees.
            let target_ke =
                0.5 * (n_thermal_dof as f64) * temperature;
            let scale = (target_ke / ke).sqrt();
            for i in 0..n {
                vx[i] = ((vx[i] as f64) * scale) as Real;
                vy[i] = ((vy[i] as f64) * scale) as Real;
                vz[i] = ((vz[i] as f64) * scale) as Real;
            }
        }
    }

    (vx, vy, vz)
}

// rq-f7e279ee
pub fn cli_main(args: Vec<String>) -> ExitCode {
    ExitCode::from(cli_main_u8(args))
}

// Testable variant returning the raw exit code.
pub fn cli_main_u8(args: Vec<String>) -> u8 {
    // rq-82d0c34a rq-7e5cb9f8
    let mut iter = args.into_iter();
    let _exe = iter.next();
    let sub = match iter.next() {
        Some(s) => s,
        None => {
            eprintln!("{USAGE_LINE}");
            return 1;
        }
    };
    match sub.as_str() {
        "run" => cli_main_run(iter.collect()),
        "lint" => cli_main_lint(iter.collect()),
        "analyze" => cli_main_analyze(iter.collect()),
        _ => {
            eprintln!("{USAGE_LINE}");
            1
        }
    }
}

fn cli_main_run(rest: Vec<String>) -> u8 {
    let mut iter = rest.into_iter();
    let config_path = match iter.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("{USAGE_LINE}");
            return 1;
        }
    };
    if iter.next().is_some() {
        eprintln!("{USAGE_LINE}");
        return 1;
    }

    let registries = crate::Registries::with_builtins();
    match run_simulation_with_phase(&config_path, &registries) {
        Ok(summary) => {
            // rq-d29872e4 — one line per phase + one aggregate line.
            for ps in &summary.phases {
                let disp = if ps.elapsed_micros >= 10_000 {
                    format!("{} ms", ps.elapsed_micros / 1000)
                } else {
                    format!("{} \u{00b5}s", ps.elapsed_micros)
                };
                if ps.kind == "minimization" {
                    let conv = ps.convergence.unwrap_or("unknown");
                    println!(
                        "[heddlemd] phase `{}`: {} iters in {} (converged: {}, frames: {}, log rows: {})",
                        ps.name, ps.n_steps, disp, conv, ps.frames_written, ps.log_rows_written
                    );
                } else {
                    println!(
                        "[heddlemd] phase `{}`: {} steps in {} (frames: {}, log rows: {})",
                        ps.name, ps.n_steps, disp, ps.frames_written, ps.log_rows_written
                    );
                }
            }
            let total_disp = if summary.total_elapsed_micros >= 10_000 {
                format!("{} ms", summary.total_elapsed_micros / 1000)
            } else {
                format!("{} \u{00b5}s", summary.total_elapsed_micros)
            };
            println!(
                "[heddlemd] complete: {} phases, {} steps in {}",
                summary.phases.len(),
                summary.total_n_steps,
                total_disp,
            );
            0
        }
        Err((err, phase)) => {
            eprintln!("error: {err}");
            match phase {
                ExitPhase::Setup => 1,
                ExitPhase::Loop => 2,
            }
        }
    }
}

fn cli_main_lint(rest: Vec<String>) -> u8 {
    let mut config_path: Option<PathBuf> = None;
    let mut with_gpu = false;
    for arg in rest {
        match arg.as_str() {
            "--with-gpu" => with_gpu = true,
            a if a.starts_with("--") => {
                eprintln!("{USAGE_LINE}");
                return 1;
            }
            _ => {
                if config_path.is_some() {
                    eprintln!("{USAGE_LINE}");
                    return 1;
                }
                config_path = Some(PathBuf::from(arg));
            }
        }
    }
    let config_path = match config_path {
        Some(p) => p,
        None => {
            eprintln!("{USAGE_LINE}");
            return 1;
        }
    };

    // Dispatch on extension: `.in.analysis` runs the analyze lint
    // pipeline; everything else falls through to the simulation lint
    // pipeline (whose filename-convention check rejects non-`.in.toml`
    // paths internally with `InvalidConfigFilename`).
    let is_analysis = config_path
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.ends_with(".in.analysis"))
        .unwrap_or(false);

    if is_analysis {
        if with_gpu {
            eprintln!("{USAGE_LINE}");
            return 1;
        }
        let report = crate::analysis::lint_analyses(&config_path);
        let _ = report.write_to(&mut std::io::stdout());
        if let Some(err) = report.first_failure() {
            eprintln!("error: {err}");
        }
        return if report.ok() { 0 } else { 1 };
    }

    let report = lint_simulation(&config_path, with_gpu);
    let _ = report.write_to(&mut std::io::stdout());
    if let Some(err) = report.first_failure() {
        eprintln!("error: {err}");
    }
    if report.ok() { 0 } else { 1 }
}

fn cli_main_analyze(rest: Vec<String>) -> u8 {
    let mut iter = rest.into_iter();
    let config_path = match iter.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("{USAGE_LINE}");
            return 1;
        }
    };
    if iter.next().is_some() {
        eprintln!("{USAGE_LINE}");
        return 1;
    }
    match crate::analysis::run_analyses(&config_path) {
        Ok(summary) => {
            let elapsed = summary.elapsed_micros;
            let elapsed_disp = if elapsed >= 10_000 {
                format!("{} ms", elapsed / 1000)
            } else {
                format!("{elapsed} \u{00b5}s")
            };
            println!(
                "[heddlemd] analyze complete: {} analyses over {} frames in {}",
                summary.analyses_written, summary.frames_consumed, elapsed_disp
            );
            0
        }
        Err(e) => {
            eprintln!("error: {e}");
            // Distinguish setup vs loop-time errors. The trajectory
            // pass and per-analysis writes correspond to the loop
            // phase; everything else is setup.
            match &e {
                crate::analysis::AnalyzeError::Trajectory(_)
                | crate::analysis::AnalyzeError::Analysis { .. } => 2,
                _ => 1,
            }
        }
    }
}

#[cfg(test)]
mod scalar_predicate_tests {
    use super::{captures_forces_only_graph, step_needs_force_scalars};

    // rq-2af44cf4
    #[test]
    fn log_step_needs_force_scalars() {
        // A log step (no barostat) evaluates forces+scalars.
        assert!(step_needs_force_scalars(true, false));
    }

    // rq-ed183041
    #[test]
    fn plain_step_is_forces_only() {
        // Neither a log step nor a barostat step: forces only.
        assert!(!step_needs_force_scalars(false, false));
    }

    // rq-091a4341
    #[test]
    fn trajectory_only_step_is_forces_only() {
        // A trajectory-only step is not a log step and has no barostat,
        // so it does not require force-kernel scalars.
        assert!(!step_needs_force_scalars(false, false));
    }

    #[test]
    fn barostat_step_needs_force_scalars() {
        // A barostat consumes the per-step virial regardless of logging.
        assert!(step_needs_force_scalars(false, true));
        assert!(step_needs_force_scalars(true, true));
    }

    // rq-26dce0f6
    #[test]
    fn no_barostat_captures_forces_only_graph() {
        assert!(captures_forces_only_graph(false));
    }

    // rq-c6c56cdc
    #[test]
    fn barostat_skips_forces_only_graph() {
        assert!(!captures_forces_only_graph(true));
    }
}
