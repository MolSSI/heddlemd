use super::*;
use std::path::Path;
use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::forces::{
    AngleList, BondList, ChargeList, ConstraintList, DihedralList, ExclusionList, ForceField,
    load_topology_file,
};
use crate::gpu::{ParticleBuffers, init_device};
use crate::io::config::NeighborListConfig;
use crate::io::{
    InitState, InitVelocities, load_config_raw, load_init_state,
};
use crate::state::{ParticleState, ParticleStateError};
use crate::timings::Timings;
use crate::precision::Real;

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

// rq-ef902cf6 rq-dcfdb7c9
pub(crate) fn run_simulation_with_phase(
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
        let cutoff_max = compute_cutoff_max(&config);
        let required = (3.0 * (cutoff_max + r_skin)) as Real;
        if let Err(e) = sim_box.check_min_perpendicular_width(required) {
            return Err(match e {
                crate::pbc::SimulationBoxError::PerpendicularWidthTooSmall {
                    direction,
                    width,
                    required,
                } => {
                    // Report the payload in the config's unit system.
                    let (lf, _lu) = length_display(config.units);
                    RunnerError::CellListBoxTooSmall {
                        direction,
                        width: (width as f64 * lf) as Real,
                        required: (required as f64 * lf) as Real,
                    }
                }
                _ => unreachable!(
                    "check_min_perpendicular_width only produces PerpendicularWidthTooSmall"
                ),
            });
        }
    }

    // Load the .topology file when supplied, otherwise build empty bond /
    // angle / exclusion lists keyed to `n`.
    let bond_type_names: Vec<&str> =
        config.bond_types.iter().map(|bt| bt.name.as_str()).collect();
    let angle_type_names: Vec<&str> =
        config.angle_types.iter().map(|at| at.name.as_str()).collect();
    let topology: (
        BondList,
        AngleList,
        DihedralList,
        ExclusionList,
        ConstraintList,
        Option<ChargeList>,
    ) = match config.topology.as_ref() {
        Some(path) => load_topology_file(
            path,
            n,
            &bond_type_names,
            &angle_type_names,
            &config.dihedral_types,
            &config.constraint_types,
            &registries.constraint_types,
            config.units,
        )
        .map_err(RunnerError::TopologyFile)?,
        None => (
            BondList::empty(n),
            AngleList::empty(n),
            DihedralList::empty(n),
            ExclusionList::empty(n),
            ConstraintList::empty(n),
            None,
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
pub(crate) fn simulation_setup_finish_gpu(
    config: crate::io::Config,
    registries: crate::Registries,
    init: InitState,
    sim_box: crate::pbc::SimulationBox,
    topology: (
        BondList,
        AngleList,
        DihedralList,
        ExclusionList,
        ConstraintList,
        Option<ChargeList>,
    ),
    config_load_duration: Duration,
    init_load_duration: Duration,
    gpu: crate::gpu::GpuContext,
    gpu_init_duration: Duration,
) -> Result<SimulationSetup, RunnerError> {
    let n = init.particle_count;
    let (bond_list, angle_list, dihedral_list, exclusion_list, constraint_list, per_atom_charges) =
        topology;

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

    // rq-d734328e — a topology [charges] section is the sole charge source
    // and is mutually exclusive with any nonzero per-type charge.
    if per_atom_charges.is_some() {
        if let Some(pt) = config.particle_types.iter().find(|pt| pt.charge != 0.0) {
            return Err(RunnerError::TypeChargeWithPerAtomCharges {
                type_name: pt.name.clone(),
                charge: pt.charge,
            });
        }
    }

    // Build masses and charges arrays from per-particle type_index lookup.
    // Charge comes from the topology [charges] section (a ChargeList) when
    // present, otherwise from the per-type `charge` fallback.
    let mut masses_f64: Vec<f64> = Vec::with_capacity(n);
    let mut masses: Vec<Real> = Vec::with_capacity(n);
    let mut charges: Vec<Real> = Vec::with_capacity(n);
    for (i, &ti) in init.type_indices.iter().enumerate() {
        let pt = &config.particle_types[ti as usize];
        masses_f64.push(pt.mass);
        masses.push(pt.mass as Real);
        let q = match &per_atom_charges {
            Some(cl) => cl.charges[i],
            None => pt.charge as Real,
        };
        charges.push(q);
    }

    // rq-d734328e — non-blocking net-charge warning. SPME tolerates a
    // nonzero net charge via a uniform neutralising background, so this is
    // a likely-modelling-error hint, not a fatal condition.
    let q_net: f64 = charges.iter().map(|&q| q as f64).sum();
    if q_net.abs() > 1.0e-4 {
        eprintln!(
            "warning: system net charge = {q_net:.6e} e (exceeds 1e-4 e); \
             SPME applies a uniform neutralizing background"
        );
    }

    let n_constraints = constraint_list.total_constraint_count();
    // Thermal degrees of freedom used by `compute_temperature` and by
    // the initial-velocity equipartition rescale: `3N` Cartesian
    // components, minus one per holonomic constraint, minus 3 for the
    // COM momentum (subtracted at velocity generation and preserved by
    // the momentum-conserving dynamics). Clamped to 0, not 1: zero
    // thermal DOF is a legitimate state — `compute_temperature`
    // reports 0.0 for it and velocity generation zeroes all velocities
    // — unlike the thermostats' `max(1)` floors, which exist to keep
    // their internal divisions by `N_f` well-defined.
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
            // Equipartition target over the thermal DOF only: the 3 COM
            // modes were just subtracted and constrained modes carry no
            // kinetic energy, so `K_target = (N_thermal_dof / 2) · k_B·T`.
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
    let force_field = ForceField::new_with_combining(
        &registries.potentials,
        &gpu,
        n,
        &sim_box,
        &config.particle_types,
        &config.pair_interactions,
        config.lennard_jones.as_ref(),
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
