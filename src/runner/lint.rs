use super::*;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::forces::{
    AngleList, BondList, ChargeList, ConstraintList, DihedralList, ExclusionList,
    load_topology_file,
};
use crate::gpu::init_device;
use crate::io::config::NeighborListConfig;
use crate::io::{
    InitState, load_config_raw, load_init_state,
};
use crate::precision::Real;

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
                    detail: {
                        let (lf, lu) = length_display(config.units);
                        format!(
                            "resolved, {} particles, box {:.1e} × {:.1e} × {:.1e} {lu}",
                            i.particle_count,
                            i.sim_box.lx() as f64 * lf,
                            i.sim_box.ly() as f64 * lf,
                            i.sim_box.lz() as f64 * lf,
                        )
                    },
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
                    let (lf, lu) = length_display(config.units);
                    stages.push(LintStage {
                        label: "box/cutoff",
                        status: LintStatus::Ok {
                            detail: format!(
                                "min perp width {:.2e} {lu} ≥ required {:.2e} {lu}",
                                sim_box.min_perpendicular_width() as f64 * lf,
                                required as f64 * lf,
                            ),
                        },
                    });
                }
                Err(crate::pbc::SimulationBoxError::PerpendicularWidthTooSmall {
                    direction,
                    width,
                    required,
                }) => {
                    let (lf, lu) = length_display(config.units);
                    let width_u = width as f64 * lf;
                    let required_u = required as f64 * lf;
                    stages.push(LintStage {
                        label: "box/cutoff",
                        status: LintStatus::Fail {
                            detail: format!(
                                "min perp width {width_u:.2e} {lu} along `{direction}` < required {required_u:.2e} {lu}"
                            ),
                            error: RunnerError::CellListBoxTooSmall {
                                direction,
                                width: width_u as Real,
                                required: required_u as Real,
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
    let bond_type_names: Vec<&str> = config.bond_types.iter().map(|bt| bt.name.as_str()).collect();
    let angle_type_names: Vec<&str> = config.angle_types.iter().map(|at| at.name.as_str()).collect();
    let topology = match config.topology.as_ref() {
        Some(path) => match load_topology_file(
            path,
            n,
            &bond_type_names,
            &angle_type_names,
            &config.dihedral_types,
            &config.constraint_types,
            &registries.constraint_types,
            config.units,
        ) {
            Ok((
                bond_list,
                angle_list,
                dihedral_list,
                exclusion_list,
                constraint_list,
                charge_list,
            )) => {
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
                Some((
                    bond_list,
                    angle_list,
                    dihedral_list,
                    exclusion_list,
                    constraint_list,
                    charge_list,
                ))
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
    topology: Option<(
        BondList,
        AngleList,
        DihedralList,
        ExclusionList,
        ConstraintList,
        Option<ChargeList>,
    )>,
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
            None,
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
