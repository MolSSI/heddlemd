use super::*;
use std::time::{Duration, Instant};

use crate::gpu::compute_total_potential_energy;
use crate::io::MinlogWriter;
use crate::minimizer::{MinimizerConvergence, MinimizerError};
use crate::io::{
    LogWriter,
    TrajectoryWriter,
};
use crate::io::log_output::{compute_kinetic_energy, compute_temperature};
use crate::state::{ParticleState, ParticleStateError};
use crate::timings::{
    HostStage, KernelStage, Timings,
    write_timings_file,
};
use crate::precision::Real;

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
    // rq-3e1fba8b — hand the slots that need it the connectivity-derived
    // molecule partition. The Monte-Carlo barostat uploads its molecule tables
    // and resolves its default volume step here; the Andersen thermostat uploads
    // the same tables so its stochastic collisions resample whole rigid
    // molecules rather than individual atoms (required for correctness under
    // holonomic constraints — see `rqm/integration/andersen.md`).
    if thermostat.is_some() || barostat.is_some() {
        let molecules = crate::forces::MoleculeList::from_topology(
            n,
            &setup.bond_list,
            &setup.constraint_list,
        );
        if let Some(t) = thermostat.as_mut() {
            t.init_run(&molecules)
                .map_err(|e| (RunnerError::Thermostat(e), ExitPhase::Setup))?;
        }
        if let Some(b) = barostat.as_mut() {
            b.init_run(&setup.sim_box, &molecules)
                .map_err(|e| (RunnerError::Barostat(e), ExitPhase::Setup))?;
        }
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

    // Each slot runs its own work as standalone launches: the
    // integrator's trailing kick in the plan walk, and the thermostat's
    // / barostat's launches in apply_post / apply. The CUDA graph
    // captures these like any other launch.
    //
    // The plan shape is a pure function of `dt` and the integrator's
    // static configuration, so one probe serves every setup-time shape
    // query (thermostat/barostat/constraint marker topology, graph
    // eligibility). Computing it once avoids redundant `plan()`
    // allocations during phase setup.
    let probe_plan = integrator.plan(phase.dt as Real);

    // rq-77f1e6ef — validate the schedule's data dependencies before the
    // timestep loop: reject a plan whose operation reads force-derived
    // state that a preceding position/box mutation invalidated, with no
    // intervening force evaluation (see `op-model.md`). The plan shape is
    // static, so one check per phase covers every step.
    //
    // rq-b83f8ae6 — the validation context reflects the phase's pressure
    // coupling: a per-step barostat makes the plan's `BarostatPoint` marker
    // active (mutating), and a weak-coupling barostat tolerates the cached
    // forces its terminal rescale leaves stale for the next step. A
    // periodic (Monte-Carlo) barostat is inert per step, and an integrator
    // that owns its coupling (mtk-npt) has no barostat slot, so both leave
    // the marker inert.
    let validation_ctx = match barostat.as_ref() {
        Some(b)
            if b.periodicity() == crate::integrator::BarostatPeriodicity::EveryStep =>
        {
            crate::integrator::StepValidationContext::per_step_barostat(
                b.tolerates_stale_cached_forces(),
            )
        }
        _ => crate::integrator::StepValidationContext::no_barostat(),
    };
    probe_plan.validate(&validation_ctx).map_err(|source| {
        (
            RunnerError::InvalidSchedule {
                integrator: phase.integrator.kind.clone(),
                source,
            },
            ExitPhase::Setup,
        )
    })?;

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
    // rq-0286c77d — physics series, one entry per emitted log row.
    let mut physics: Vec<PhysicsSample> = Vec::new();

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
        // rq-0286c77d
        physics.push(
            capture_physics_sample(0, 0.0, ke, n_thermal_dof, &mut setup.buffers, &setup.sim_box)
                .map_err(|g| (RunnerError::Gpu(g), ExitPhase::Setup))?,
        );
    }

    let n_steps = phase.n_steps;
    let progress_every = (n_steps / 100).max(1);
    let dt = phase.dt as Real;
    let phase_name = phase.name.as_str();
    // rq-ee10237d — the thermostat couples every `coupling_interval`
    // steps; on those steps the per-step path runs the whole post-force
    // region eagerly (the full-step KE reduction is a fusion barrier).
    let coupling_interval = phase.coupling_interval as u64;

    // rq-dbbffa7d — a per-step barostat requires the integrator's plan
    // to carry a BarostatPoint, or its `apply` would never fire.
    // `run_step` dispatches a terminal BarostatPoint's `apply` (which
    // runs its rescale as a standalone launch) in the post-force tail of
    // its whole-plan walk, and an interleaved BarostatPoint during the
    // walk; either way the plan must declare the marker.
    if barostat_couples_per_step(&barostat) && !probe_plan.has_barostat_points() {
        return Err((
            RunnerError::BarostatPlacementMissing {
                integrator: phase.integrator.kind.clone(),
            },
            ExitPhase::Setup,
        ));
    }

    // Decide whether this phase is eligible for CUDA graph capture.
    // See `rqm/cuda-graphs.md` for the activation policy. Every
    // active slot must report `graph_compatible == true`, the
    // run-wide override flag must be off, and capture must succeed
    // at runtime; otherwise the phase runs the per-step launch loop
    // with full per-kernel `Timings`.
    // rq-9fbba3be — a plan containing ThermostatHalf sub-steps
    // dispatches host-side thermostat arithmetic mid-plan, which
    // cannot be captured; marker-bearing plans run on the eager path.
    let plan_has_thermostat_points = probe_plan.has_thermostat_points();
    // rq-dbbffa7d — an interleaved (non-terminal) BarostatPoint runs
    // its `apply` mid-walk, whose host-side barostat arithmetic cannot
    // be captured; such plans run eager. A terminal BarostatPoint keeps
    // the phase eligible (its `apply` runs in the captured post-force
    // tail as standalone launches).
    let plan_has_interleaved_barostat = probe_plan.has_interleaved_barostat_point();
    // rq-3c78ea7d — a graph-eligible thermostat's coupling is a device-side
    // kernel sequence recorded into the coupling-variant graph; the replay
    // loop selects that variant on coupling steps and a non-coupling variant
    // otherwise. This holds for any `coupling_interval >= 1`, so the
    // thermostat does not gate eligibility beyond its `graph_compatible`
    // hook (checked by `phase_slots_graph_compatible`, which rejects a
    // thermostat whose coupling is not a pure device sequence).
    let graph_eligible = !setup.config.simulation.cuda_graphs_disable
        && !plan_has_thermostat_points
        && !plan_has_interleaved_barostat
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
            &mut constraint,            plan_has_thermostat_points,
            coupling_interval,
            dt,
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
            &mut physics,
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
            coupling_interval,
            dt,
            &mut timings,
            &setup.gpu.device,
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
            &mut integrator,
            &mut thermostat,
            &mut barostat,
            &mut constraint,
            plan_has_thermostat_points,
            coupling_interval,
            dt,
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
            &mut physics,
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
                &mut constraint,                plan_has_thermostat_points,
                coupling_interval,
                dt,
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
                &mut physics,
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
            &mut constraint,            plan_has_thermostat_points,
            coupling_interval,
            dt,
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
            &mut physics,
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
        min_final_max_force: None,
        // rq-0286c77d
        physics,
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
        min_final_max_force: Some(final_report.max_force),
        // rq-0286c77d — minimization phases capture no physics series.
        physics: Vec::new(),
    })
}
