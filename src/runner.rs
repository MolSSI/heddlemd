// rq-357909e4 rq-02edd314 rq-77c1d5d9
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::forces::{
    AngleList, BondList, ConstraintList, ExclusionList, ForceField, ForceFieldError,
    TopologyFileError,
    load_topology_file,
};
use crate::gpu::{ParticleBuffers, compute_total_potential_energy, init_device};
use crate::integrator::{
    BarostatError, ConstraintError, IntegratorError, ThermostatError,
};
use crate::io::config::NeighborListConfig;
use crate::io::{
    ConfigError, InitState, InitStateError, InitVelocities, LogWriter, LogWriterError,
    TrajectoryWriter, TrajectoryWriterError, load_config_raw, load_init_state,
};
use crate::io::log_output::{BOLTZMANN_J_PER_K, compute_kinetic_energy, compute_temperature};
use crate::state::{ParticleState, ParticleStateError};
use crate::timings::{
    HostStage, KernelStage, Timings, TimingsError, TimingsWriterError, write_timings_file,
};

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
        width: f32,
        required: f32,
    },
    // rq-8ee27e27 rq-02f4d342
    #[error(
        "cuFFT returned non-deterministic R2C output between two identical runs ({differences} differing floats); SPME requires bit-exact reciprocal-space behaviour"
    )]
    CuFftNonDeterministic { differences: usize },
}

// rq-5c1cfc93
#[derive(Debug, Clone, Copy)]
pub struct RunSummary {
    pub n_steps: u64,
    pub frames_written: u64,
    pub log_rows_written: u64,
    pub elapsed_micros: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExitPhase {
    Setup,
    Loop,
}

const USAGE_LINE: &str = "\
usage: dynamics run  <config-path>
       dynamics lint <config-path> [--with-gpu]";

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
            LintOverall::Ok => "[dynamics lint] OK",
            LintOverall::Fail => "[dynamics lint] FAIL",
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

    // Stage 2: output paths.
    let output_collision: Option<PathBuf> =
        if config.output.trajectory_every > 0 && config.output.trajectory_path.exists() {
            Some(config.output.trajectory_path.clone())
        } else if config.output.log_every > 0 && config.output.log_path.exists() {
            Some(config.output.log_path.clone())
        } else if config.output.timings_path.exists() {
            Some(config.output.timings_path.clone())
        } else {
            None
        };
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

    // Stage 3: init.
    let type_name_strings: Vec<String> = config
        .particle_types
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let type_name_refs: Vec<&str> = type_name_strings.iter().map(|s| s.as_str()).collect();
    let init = match load_init_state(&config.init, &type_name_refs) {
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
            let required = (3.0 * (cutoff_max + r_skin)) as f32;
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
            &config.constraint_types,
            &registries.constraint_types,
        ) {
            Ok((bond_list, angle_list, exclusion_list, constraint_list)) => {
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
                            "{}: {} bonds, {} angles, {} constraint groups",
                            path.display(),
                            bond_list.bonds.len(),
                            angle_list.angles.len(),
                            constraint_list.groups.len(),
                        ),
                    },
                });
                Some((bond_list, angle_list, exclusion_list, constraint_list))
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

    match lint_gpu_full_setup(&config, &init, &sim_box, n, topology, registries) {
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
    if let Some(c) = config.coulomb.as_ref() {
        cutoff_max = cutoff_max.max(c.cutoff);
    }
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
fn lint_gpu_full_setup(
    config: &crate::io::Config,
    init: &InitState,
    sim_box: &crate::pbc::SimulationBox,
    n: usize,
    topology: Option<(BondList, AngleList, ExclusionList, ConstraintList)>,
    registries: &crate::Registries,
) -> Result<(), (String, RunnerError)> {
    let gpu = init_device().map_err(|e| (format!("{e}"), RunnerError::Gpu(e)))?;

    if config.spme.is_some() {
        let differences = crate::gpu::cufft::cufft_determinism_smoke_test(&gpu.device)
            .map_err(|e| {
                let synthetic = crate::gpu::GpuError(cudarc::driver::DriverError(
                    cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
                ));
                (format!("cuFFT smoke test errored: {e}"), RunnerError::Gpu(synthetic))
            })?;
        if differences != 0 {
            return Err((
                format!("cuFFT produced {differences} differing bytes between two R2C transforms of the same input"),
                RunnerError::CuFftNonDeterministic { differences },
            ));
        }
    }

    let mut masses_f64: Vec<f64> = Vec::with_capacity(n);
    let mut masses_f32: Vec<f32> = Vec::with_capacity(n);
    let mut charges_f32: Vec<f32> = Vec::with_capacity(n);
    for &ti in &init.type_indices {
        let pt = &config.particle_types[ti as usize];
        masses_f64.push(pt.mass);
        masses_f32.push(pt.mass as f32);
        charges_f32.push(pt.charge as f32);
    }

    let (vx, vy, vz) = match &init.velocities {
        Some(v) => (
            v.velocities_x.clone(),
            v.velocities_y.clone(),
            v.velocities_z.clone(),
        ),
        None => generate_velocities(
            n,
            config.simulation.temperature,
            config.simulation.seed,
            &masses_f64,
        ),
    };

    let images_arg = init
        .images
        .as_ref()
        .map(|im| (im.images_x.clone(), im.images_y.clone(), im.images_z.clone()));
    let charges_for_ff = charges_f32.clone();
    let state = ParticleState::new(
        init.positions_x.clone(),
        init.positions_y.clone(),
        init.positions_z.clone(),
        vx,
        vy,
        vz,
        masses_f32.clone(),
        charges_f32,
        init.type_indices.clone(),
        None,
        images_arg,
    )
    .map_err(|e| (format!("{e}"), RunnerError::ParticleState(e)))?;

    let _buffers = ParticleBuffers::new(&gpu, &state).map_err(|e| match e {
        ParticleStateError::Gpu(g) => (format!("{g}"), RunnerError::Gpu(g)),
        other => (format!("{other}"), RunnerError::ParticleState(other)),
    })?;

    let _integrator = registries
        .integrators
        .build(&config.integrator, &gpu, n)
        .map_err(|e| (format!("{e}"), RunnerError::Integrator(e)))?;
    let _thermostat = registries
        .thermostats
        .build_optional(config.thermostat.as_ref(), &gpu, n)
        .map_err(|e| (format!("{e}"), RunnerError::Thermostat(e)))?;
    let _barostat = registries
        .barostats
        .build_optional(config.barostat.as_ref(), &gpu, n)
        .map_err(|e| (format!("{e}"), RunnerError::Barostat(e)))?;

    let (bond_list, angle_list, exclusion_list, constraint_list) = topology.unwrap_or_else(|| {
        (
            BondList::empty(n),
            AngleList::empty(n),
            ExclusionList::empty(n),
            ConstraintList::empty(n),
        )
    });

    let _constraint = registries
        .constraint_types
        .build_optional(
            &constraint_list,
            &gpu,
            n,
            &masses_f32,
            &config.constraint_types,
        )
        .map_err(|e| (format!("{e}"), RunnerError::Constraint(e)))?;

    let _force_field = ForceField::new(
        &registries.potentials,
        &gpu,
        n,
        sim_box,
        &config.particle_types,
        &config.pair_interactions,
        &config.bond_types,
        &config.angle_types,
        config.coulomb.as_ref(),
        config.spme.as_ref(),
        &charges_for_ff,
        &bond_list,
        &angle_list,
        &exclusion_list,
        &config.neighbor_list,
    )
    .map_err(|e| (format!("{e}"), RunnerError::ForceField(e)))?;

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
    let total_started = Instant::now();

    // Time config_load before any other instrumentation exists.
    // Parse the config without running the registry-dispatched
    // validation, then run that validation against the
    // caller-supplied `registries`. This is what lets a custom-kind
    // config validate cleanly when a matching custom builder is
    // registered.
    let mut config_load_duration = Duration::ZERO;
    let config = timed(&mut config_load_duration, || load_config_raw(config_path))
        .map_err(|e| (RunnerError::Config(e), ExitPhase::Setup))?;
    config
        .validate_against(registries)
        .map_err(|e| (RunnerError::Config(e), ExitPhase::Setup))?;

    // Pre-flight output existence checks. Trajectory and log are gated by
    // their `_every > 0` predicates; the timings file is always written.
    if config.output.trajectory_every > 0 && config.output.trajectory_path.exists() {
        return Err((
            RunnerError::OutputExists {
                path: config.output.trajectory_path.clone(),
            },
            ExitPhase::Setup,
        ));
    }
    if config.output.log_every > 0 && config.output.log_path.exists() {
        return Err((
            RunnerError::OutputExists {
                path: config.output.log_path.clone(),
            },
            ExitPhase::Setup,
        ));
    }
    if config.output.timings_path.exists() {
        return Err((
            RunnerError::OutputExists {
                path: config.output.timings_path.clone(),
            },
            ExitPhase::Setup,
        ));
    }

    let type_name_strings: Vec<String> = config
        .particle_types
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let type_name_refs: Vec<&str> = type_name_strings.iter().map(|s| s.as_str()).collect();

    let mut init_load_duration = Duration::ZERO;
    let init = timed(&mut init_load_duration, || {
        load_init_state(&config.init, &type_name_refs)
    })
    .map_err(|e| (RunnerError::InitState(e), ExitPhase::Setup))?;

    let mut sim_box = init.sim_box;
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
        if let Some(c) = config.coulomb.as_ref() {
            cutoff_max = cutoff_max.max(c.cutoff);
        }
        if let Some(s) = config.spme.as_ref() {
            cutoff_max = cutoff_max.max(s.r_cut_real);
        }
        let required = (3.0 * (cutoff_max + r_skin)) as f32;
        if let Err(e) = sim_box.check_min_perpendicular_width(required) {
            return Err(match e {
                crate::pbc::SimulationBoxError::PerpendicularWidthTooSmall {
                    direction,
                    width,
                    required,
                } => (
                    RunnerError::CellListBoxTooSmall {
                        direction,
                        width,
                        required,
                    },
                    ExitPhase::Setup,
                ),
                // The constructor / mutator variants are unreachable here:
                // `check_min_perpendicular_width` only produces
                // `PerpendicularWidthTooSmall`.
                _ => unreachable!(
                    "check_min_perpendicular_width only produces PerpendicularWidthTooSmall"
                ),
            });
        }
    }

    let mut gpu_init_duration = Duration::ZERO;
    let gpu = timed(&mut gpu_init_duration, init_device)
        .map_err(|e| (RunnerError::Gpu(e), ExitPhase::Setup))?;

    // rq-637cd1a5 rq-02f4d342 rq-ea4205ec
    //
    // SPME runs depend on cuFFT producing bit-identical output on
    // repeated calls with identical input. cuFFT's plan selector is
    // deterministic on a given GPU, but we verify this up front so a
    // misconfigured environment fails loudly at setup rather than
    // producing silently non-reproducible simulations.
    if config.spme.is_some() {
        let differences = crate::gpu::cufft::cufft_determinism_smoke_test(&gpu.device)
            .map_err(|_| (RunnerError::Gpu(crate::gpu::GpuError(
                cudarc::driver::DriverError(
                    cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
                ),
            )), ExitPhase::Setup))?;
        if differences != 0 {
            return Err((
                RunnerError::CuFftNonDeterministic { differences },
                ExitPhase::Setup,
            ));
        }
    }

    // Construct the Timings instance now and replay the three pre-instrumented
    // host stages.
    let mut timings = Timings::new(&gpu)
        .map_err(|e| (RunnerError::Timings(e), ExitPhase::Setup))?;
    timings.record_host(HostStage::CONFIG_LOAD, config_load_duration);
    timings.record_host(HostStage::INIT_LOAD, init_load_duration);
    timings.record_host(HostStage::GPU_INIT, gpu_init_duration);

    // Build masses and charges arrays from per-particle type_index lookup.
    let mut masses_f64: Vec<f64> = Vec::with_capacity(n);
    let mut masses_f32: Vec<f32> = Vec::with_capacity(n);
    let mut charges_f32: Vec<f32> = Vec::with_capacity(n);
    for &ti in &init.type_indices {
        let pt = &config.particle_types[ti as usize];
        masses_f64.push(pt.mass);
        masses_f32.push(pt.mass as f32);
        charges_f32.push(pt.charge as f32);
    }

    // Build velocities: either from the init state or sampled.
    let (velocities_x, velocities_y, velocities_z) = match init.velocities {
        Some(InitVelocities {
            velocities_x,
            velocities_y,
            velocities_z,
        }) => (velocities_x, velocities_y, velocities_z),
        None => {
            let mut velgen = Duration::ZERO;
            let vs = timed(&mut velgen, || {
                generate_velocities(
                    n,
                    config.simulation.temperature,
                    config.simulation.seed,
                    &masses_f64,
                )
            });
            timings.record_host(HostStage::VELOCITY_GENERATION, velgen);
            vs
        }
    };

    let images_arg = init.images.as_ref().map(|im| {
        (im.images_x.clone(), im.images_y.clone(), im.images_z.clone())
    });
    let charges_for_force_field = charges_f32.clone();
    let state = ParticleState::new(
        init.positions_x.clone(),
        init.positions_y.clone(),
        init.positions_z.clone(),
        velocities_x,
        velocities_y,
        velocities_z,
        masses_f32.clone(),
        charges_f32,
        init.type_indices.clone(),
        None,
        images_arg,
    )
    .map_err(|e| (RunnerError::ParticleState(e), ExitPhase::Setup))?;

    let mut upload = Duration::ZERO;
    let mut buffers = timed(&mut upload, || ParticleBuffers::new(&gpu, &state))
        .map_err(|e| match e {
            ParticleStateError::Gpu(g) => (RunnerError::Gpu(g), ExitPhase::Setup),
            other => (RunnerError::ParticleState(other), ExitPhase::Setup),
        })?;
    timings.record_host(HostStage::HOST_TO_DEVICE_UPLOAD, upload);

    // Build the three slot handles in the framework's dispatch order
    // through the caller-supplied `Registries` bundle.
    let mut integrator = registries
        .integrators
        .build(&config.integrator, &gpu, n)
        .map_err(|e| (RunnerError::Integrator(e), ExitPhase::Setup))?;
    let mut thermostat = registries
        .thermostats
        .build_optional(config.thermostat.as_ref(), &gpu, n)
        .map_err(|e| (RunnerError::Thermostat(e), ExitPhase::Setup))?;
    let mut barostat = registries
        .barostats
        .build_optional(config.barostat.as_ref(), &gpu, n)
        .map_err(|e| (RunnerError::Barostat(e), ExitPhase::Setup))?;

    // Load the .topology file when supplied, otherwise build empty bond /
    // angle / exclusion lists keyed to `n`.
    let bond_type_names: Vec<&str> =
        config.bond_types.iter().map(|bt| bt.name()).collect();
    let angle_type_names: Vec<&str> =
        config.angle_types.iter().map(|at| at.name()).collect();
    let (bond_list, angle_list, exclusion_list, constraint_list): (
        BondList,
        AngleList,
        ExclusionList,
        ConstraintList,
    ) = match config.topology.as_ref() {
        Some(path) => load_topology_file(
            path,
            n,
            &bond_type_names,
            &angle_type_names,
            &config.constraint_types,
            &registries.constraint_types,
        )
        .map_err(|e| (RunnerError::TopologyFile(e), ExitPhase::Setup))?,
        None => (
            BondList::empty(n),
            AngleList::empty(n),
            ExclusionList::empty(n),
            ConstraintList::empty(n),
        ),
    };
    // rq-acfda5d4 — runner-side enforcement of the integrator/constraint
    // compatibility rule; consults the builder predicate via the
    // registry. Cannot run during `Config::validate_against` because
    // the topology file is loaded separately.
    config
        .validate_constraint_compatibility(registries, !constraint_list.is_empty())
        .map_err(|e| (RunnerError::Config(e), ExitPhase::Setup))?;
    // rq-3d5f2e98 — construct the constraint slot. `build_optional`
    // returns `None` for an empty list.
    let mut constraint = registries
        .constraint_types
        .build_optional(
            &constraint_list,
            &gpu,
            n,
            &masses_f32,
            &config.constraint_types,
        )
        .map_err(|e| (RunnerError::Constraint(e), ExitPhase::Setup))?;

    let mut force_field = ForceField::new(
        &registries.potentials,
        &gpu,
        n,
        &sim_box,
        &config.particle_types,
        &config.pair_interactions,
        &config.bond_types,
        &config.angle_types,
        config.coulomb.as_ref(),
        config.spme.as_ref(),
        &charges_for_force_field,
        &bond_list,
        &angle_list,
        &exclusion_list,
        &config.neighbor_list,
    )
    .map_err(|e| (RunnerError::ForceField(e), ExitPhase::Setup))?;

    // Open output writers (only the enabled ones).
    let mut traj_writer: Option<TrajectoryWriter> = if config.output.trajectory_every > 0 {
        Some(
            TrajectoryWriter::open(
                &config.output.trajectory_path,
                config.output.include_velocities,
                config.output.include_images,
                type_name_strings.clone(),
            )
            .map_err(|e| (RunnerError::Trajectory(e), ExitPhase::Setup))?,
        )
    } else {
        None
    };
    // Concatenate per-slot log columns in dispatch order: integrator,
    // thermostat, barostat. The runner caches the joined slice for the
    // run's duration.
    let mut log_extra_columns: Vec<&'static str> =
        integrator.log_column_names().to_vec();
    if let Some(t) = thermostat.as_ref() {
        log_extra_columns.extend_from_slice(t.log_column_names());
    }
    if let Some(b) = barostat.as_ref() {
        log_extra_columns.extend_from_slice(b.log_column_names());
    }
    let mut log_writer: Option<LogWriter> = if config.output.log_every > 0 {
        Some(
            LogWriter::open(&config.output.log_path, &log_extra_columns)
                .map_err(|e| (RunnerError::Log(e), ExitPhase::Setup))?,
        )
    } else {
        None
    };

    // rq-fc6859df: length-1 scratch for the runner's per-log-row PE
    // reduction. Only constructed when at least one slot declares a
    // PE-using log column; otherwise the device allocation is skipped.
    let mut pe_scratch: Option<cudarc::driver::CudaSlice<f32>> =
        if !log_extra_columns.is_empty() {
            Some(
                gpu.device
                    .alloc_zeros::<f32>(1)
                    .map_err(|e| (RunnerError::Gpu(crate::gpu::GpuError(e)), ExitPhase::Setup))?,
            )
        } else {
            None
        };

    let started = Instant::now();

    // Warm-up: populate forces with F(x_0) via the force-field pipeline.
    force_field
        .step(&mut buffers, &sim_box, &mut timings)
        .map_err(|e| (RunnerError::ForceField(e), ExitPhase::Setup))?;

    // Host scratch state for downloads.
    let mut frame: ParticleState = state.clone();
    let mut frames_written: u64 = 0;
    let mut log_rows_written: u64 = 0;

    // Step-0 outputs.
    if traj_writer.is_some() || log_writer.is_some() {
        let mut dl = Duration::ZERO;
        timed(&mut dl, || frame.download_from(&buffers)).map_err(|e| match e {
            ParticleStateError::Gpu(g) => (RunnerError::Gpu(g), ExitPhase::Setup),
            other => (RunnerError::ParticleState(other), ExitPhase::Setup),
        })?;
        timings.record_host(HostStage::DEVICE_TO_HOST_DOWNLOAD, dl);
    }
    if let Some(writer) = traj_writer.as_mut() {
        let mut tw = Duration::ZERO;
        timed(&mut tw, || {
            write_traj_frame(writer, 0, config.simulation.dt, &sim_box, &init.type_indices, &frame)
        })
        .map_err(|e| (RunnerError::Trajectory(e), ExitPhase::Setup))?;
        timings.record_host(HostStage::TRAJECTORY_WRITE, tw);
        frames_written += 1;
    }
    if let Some(writer) = log_writer.as_mut() {
        let ke = compute_kinetic_energy(
            &frame.masses,
            &frame.velocities_x,
            &frame.velocities_y,
            &frame.velocities_z,
        );
        let t = compute_temperature(ke, n);
        let extras = if log_extra_columns.is_empty() {
            Vec::new()
        } else {
            let scratch = pe_scratch
                .as_mut()
                .expect("pe_scratch allocated when log_extra_columns non-empty");
            timings
                .kernel_start(KernelStage::POTENTIAL_ENERGY_REDUCE)
                .map_err(|e| (RunnerError::Timings(e), ExitPhase::Setup))?;
            let pe = compute_total_potential_energy(&buffers, scratch).map_err(|g| {
                (RunnerError::Gpu(g), ExitPhase::Setup)
            })?;
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

    let progress_to_stdout = std::io::IsTerminal::is_terminal(&std::io::stdout());
    let n_steps = config.simulation.n_steps;
    let progress_every = (n_steps / 100).max(1);
    let dt_f32 = config.simulation.dt as f32;

    // Main loop. Dispatch order per timestep:
    //   thermostat.apply_pre → integrator.step → thermostat.apply_post
    //   → barostat.apply
    for step in 1..=n_steps {
        if let Some(t) = thermostat.as_mut() {
            t.apply_pre(&mut buffers, dt_f32, &mut timings)
                .map_err(|e| (RunnerError::Thermostat(e), ExitPhase::Loop))?;
        }
        // Walk the integrator's plan. The constraint slot's hooks are
        // inserted by run_step around any Drift/KickDrift sub-step and
        // after the final velocity update, gated by the kind-level
        // supports_constraints() predicate (`framework.md`).
        {
            let constraint_arg: Option<&mut dyn crate::integrator::Constraint> =
                match constraint.as_mut() {
                    Some(b) => Some(b.as_mut()),
                    None => None,
                };
            let supports_constraints = registries
                .integrators
                .lookup(&config.integrator.kind)
                .map(|b| b.supports_constraints(&config.integrator.params))
                .unwrap_or(false);
            crate::integrator::run_step(
                integrator.as_mut(),
                &mut buffers,
                &mut sim_box,
                &mut force_field,
                constraint_arg,
                supports_constraints,
                dt_f32,
                &mut timings,
            )
            .map_err(|e| {
                let runner_err = match e {
                    crate::integrator::StepError::Integrator(e) => RunnerError::Integrator(e),
                    crate::integrator::StepError::ForceField(e) => RunnerError::ForceField(e),
                    crate::integrator::StepError::Constraint(e) => RunnerError::Constraint(e),
                };
                (runner_err, ExitPhase::Loop)
            })?;
        }
        if let Some(t) = thermostat.as_mut() {
            t.apply_post(&mut buffers, dt_f32, &mut timings)
                .map_err(|e| (RunnerError::Thermostat(e), ExitPhase::Loop))?;
        }
        if let Some(b) = barostat.as_mut() {
            b.apply(&mut buffers, &mut sim_box, dt_f32, &mut timings)
                .map_err(|e| (RunnerError::Barostat(e), ExitPhase::Loop))?;
        }

        let want_traj =
            config.output.trajectory_every > 0 && step % config.output.trajectory_every == 0;
        let want_log = config.output.log_every > 0 && step % config.output.log_every == 0;
        if want_traj || want_log {
            let mut dl = Duration::ZERO;
            timed(&mut dl, || frame.download_from(&buffers)).map_err(|e| match e {
                ParticleStateError::Gpu(g) => (RunnerError::Gpu(g), ExitPhase::Loop),
                other => (RunnerError::ParticleState(other), ExitPhase::Loop),
            })?;
            timings.record_host(HostStage::DEVICE_TO_HOST_DOWNLOAD, dl);
        }
        if want_traj {
            let writer = traj_writer.as_mut().expect("traj_writer enabled");
            let mut tw = Duration::ZERO;
            timed(&mut tw, || {
                write_traj_frame(writer, step, config.simulation.dt, &sim_box, &init.type_indices, &frame)
            })
            .map_err(|e| (RunnerError::Trajectory(e), ExitPhase::Loop))?;
            timings.record_host(HostStage::TRAJECTORY_WRITE, tw);
            frames_written += 1;
        }
        if want_log {
            let writer = log_writer.as_mut().expect("log_writer enabled");
            let ke = compute_kinetic_energy(
                &frame.masses,
                &frame.velocities_x,
                &frame.velocities_y,
                &frame.velocities_z,
            );
            let t = compute_temperature(ke, n);
            let time = (step as f64) * config.simulation.dt;
            let extras = if log_extra_columns.is_empty() {
                Vec::new()
            } else {
                let scratch = pe_scratch
                    .as_mut()
                    .expect("pe_scratch allocated when log_extra_columns non-empty");
                timings
                    .kernel_start(KernelStage::POTENTIAL_ENERGY_REDUCE)
                    .map_err(|e| (RunnerError::Timings(e), ExitPhase::Loop))?;
                let pe = compute_total_potential_energy(&buffers, scratch)
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
            log_rows_written += 1;
        }

        // rq-73fbb111
        if progress_to_stdout && (step % progress_every == 0 || step == n_steps) {
            let pct = 100.0 * step as f64 / n_steps.max(1) as f64;
            let rate = step as f64 / started.elapsed().as_secs_f64().max(1e-9);
            println!(
                "[dynamics] step {step}/{n_steps} ({pct:.1}%) — {rate:.1e} steps/sec"
            );
        }
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

    let elapsed_micros = started.elapsed().as_micros();

    // Capture total runtime *before* finalising and writing the timings file
    // so the value reported in the file reflects everything except the file
    // write itself.
    timings.record_host(HostStage::TOTAL_RUNTIME, total_started.elapsed());

    let report = timings
        .finalize()
        .map_err(|e| (RunnerError::Timings(e), ExitPhase::Loop))?;
    write_timings_file(&config.output.timings_path, &report)
        .map_err(|e| (RunnerError::TimingsWriter(e), ExitPhase::Loop))?;

    Ok(RunSummary {
        n_steps,
        frames_written,
        log_rows_written,
        elapsed_micros,
    })
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
    temperature: f64,
    seed: u64,
    masses: &[f64],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut vx = vec![0.0_f32; n];
    let mut vy = vec![0.0_f32; n];
    let mut vz = vec![0.0_f32; n];
    if temperature == 0.0 || n == 0 {
        return (vx, vy, vz);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for i in 0..n {
        let sigma = (BOLTZMANN_J_PER_K * temperature / masses[i]).sqrt();
        for axis in 0..3 {
            let u1 = 1.0 - rng.r#gen::<f64>(); // (0, 1]
            let u2 = rng.r#gen::<f64>();        // [0, 1)
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let v = (z * sigma) as f32;
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
            let slice: &mut [f32] = match axis {
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
                *v = ((*v as f64) - v_com) as f32;
            }
        }
    }

    // rq-be568071
    // Rescale every velocity by a single scalar so the realised kinetic
    // energy matches the flat-3N target `(3N/2) * k_B * T`, the
    // degrees-of-freedom convention used by `compute_temperature`.
    //
    // The rescale targets the thermal degrees of freedom of the
    // centre-of-mass-removed velocity field, which exist only for N >= 2.
    // A single centred particle is its own centre of mass and has no
    // internal motion, so its velocity is set to exactly zero: the N == 1
    // momentum subtraction cancels the sampled velocity only up to a
    // floating-point rounding residual, and the rescale would otherwise
    // amplify that residual into a full thermal velocity. (N == 0 has
    // already returned above.)
    //
    // For N >= 2 a single deterministic scalar multiply preserves both
    // determinism and the zero-total-momentum property established above.
    if n < 2 {
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
            let target_ke = 1.5 * (n as f64) * BOLTZMANN_J_PER_K * temperature;
            let scale = (target_ke / ke).sqrt();
            for i in 0..n {
                vx[i] = ((vx[i] as f64) * scale) as f32;
                vy[i] = ((vy[i] as f64) * scale) as f32;
                vz[i] = ((vz[i] as f64) * scale) as f32;
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
            // rq-d29872e4
            let elapsed_micros = summary.elapsed_micros;
            let elapsed_disp = if elapsed_micros >= 10_000 {
                format!("{} ms", elapsed_micros / 1000)
            } else {
                format!("{elapsed_micros} \u{00b5}s")
            };
            println!(
                "[dynamics] complete: {} steps in {} (frames: {}, log rows: {})",
                summary.n_steps, elapsed_disp, summary.frames_written, summary.log_rows_written
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

    let report = lint_simulation(&config_path, with_gpu);
    // Write the per-stage report to stdout. Errors from stdout writes
    // are ignored — the report is informational and an unwritable
    // stdout is not a lint failure.
    let _ = report.write_to(&mut std::io::stdout());
    if let Some(err) = report.first_failure() {
        eprintln!("error: {err}");
    }
    if report.ok() { 0 } else { 1 }
}
