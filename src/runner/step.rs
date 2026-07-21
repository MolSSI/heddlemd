use super::*;
use std::time::{Duration, Instant};

use crate::forces::ForceField;
use crate::gpu::compute_total_potential_energy;
use crate::io::{
    LogWriter,
    TrajectoryWriter,
};
use crate::io::log_output::{compute_kinetic_energy, compute_temperature};
use crate::state::{ParticleState, ParticleStateError};
use crate::timings::{
    GraphVariant, HostStage, KernelStage, Timings,
};
use crate::precision::Real;

// rq-3c78ea7d — a physical step couples (replays the coupling-variant
// graph) when it is a multiple of `coupling_interval`. At
// `coupling_interval == 1` every step couples; the batched replay loop uses
// this to pick the coupling variant per step.
fn step_couples(step: u64, coupling_interval: u64) -> bool {
    step % coupling_interval == 0
}

/// Returns `true` iff every active slot for the phase reports
/// `graph_compatible = true`. Used by the runner to decide whether a
/// phase is eligible for CUDA graph capture. See `cuda-graphs.md`.
pub(crate) fn phase_slots_graph_compatible(
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
    // rq-26c9b8cb — a thermostat gates eligibility only when it is not
    // cadence-inert. csvr / berendsen / andersen do their per-step work
    // solely on coupling steps (through the runner's apply_pre / apply_post),
    // so they report graph_compatible == true and the captured non-coupling
    // steps are pure device sequences; nose-hoover-chain reports false (its
    // Yoshida chain and per-step KE dtoh cannot be captured). The
    // `coupling_interval == 1` condition is applied by the caller.
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
pub(crate) fn barostat_couples_per_step(
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
/// Captures the phase's executable graphs: the non-coupling forces+scalars
/// graph and (unless a per-step barostat is active) the non-coupling
/// forces-only graph when the phase has non-coupling steps, plus the
/// coupling-variant graph when a graph-eligible thermostat is present. All
/// are recorded from the same pre-capture device state; stream capture
/// records without executing, so none advances the simulation. See
/// `cuda-graphs.md` *Capture Lifecycle*.
#[allow(clippy::too_many_arguments)]
pub(crate) fn capture_phase_graph(
    buffers: &mut crate::gpu::ParticleBuffers,
    sim_box: &mut crate::pbc::SimulationBox,
    force_field: &mut crate::forces::ForceField,
    integrator: &mut dyn crate::integrator::Integrator,
    thermostat: &mut Option<Box<dyn crate::integrator::Thermostat>>,
    barostat: &mut Option<Box<dyn crate::integrator::Barostat>>,
    constraint: &mut Option<Box<dyn crate::integrator::Constraint>>,
    coupling_interval: u64,
    dt: Real,
    timings: &mut Timings,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
) -> Result<crate::gpu::GraphLoop, crate::gpu::GraphError> {
    // rq-6887c76d — the cells of the coupling × scalars matrix that this
    // phase produces. The non-coupling row exists iff the phase has
    // non-coupling steps (no thermostat, or coupling_interval > 1); the
    // coupling row iff a thermostat is present; the forces-only column iff no
    // per-step barostat consumes the per-step virial (every step then needs
    // scalars).
    let has_non_coupling = thermostat.is_none() || coupling_interval > 1;
    let has_coupling = thermostat.is_some();
    let has_forces_only = captures_forces_only_graph(barostat_couples_per_step(barostat));
    let coupling_dt = coupling_interval as Real * dt;

    // graphs[coupling as usize][scalars as usize]
    let mut graphs: [[Option<crate::gpu::CudaGraphExec>; 2]; 2] = [[None, None], [None, None]];

    // Non-coupling row (thermostat inert; coupling_dt = None).
    if has_non_coupling {
        graphs[0][1] = Some(capture_one_graph(
            buffers, sim_box, force_field, integrator, None, barostat, constraint, None, dt,
            timings, device, true, GraphVariant::ForcesAndScalars,
        )?);
        if has_forces_only {
            graphs[0][0] = Some(capture_one_graph(
                buffers, sim_box, force_field, integrator, None, barostat, constraint, None, dt,
                timings, device, false, GraphVariant::ForcesOnly,
            )?);
        }
    }

    // rq-c4ba5005 — coupling row: the thermostat's device-side coupling
    // recorded at `coupling_dt = coupling_interval * dt`, at both
    // force-evaluation levels so a non-log coupling step replays the cheap
    // forces-only cell.
    if has_coupling {
        let therm_arg: Option<&mut dyn crate::integrator::Thermostat> = match thermostat.as_mut() {
            Some(t) => Some(t.as_mut()),
            None => None,
        };
        graphs[1][1] = Some(capture_one_graph(
            buffers, sim_box, force_field, integrator, therm_arg, barostat, constraint,
            Some(coupling_dt), dt, timings, device, true, GraphVariant::CouplingForcesAndScalars,
        )?);
        if has_forces_only {
            let therm_arg: Option<&mut dyn crate::integrator::Thermostat> =
                match thermostat.as_mut() {
                    Some(t) => Some(t.as_mut()),
                    None => None,
                };
            graphs[1][0] = Some(capture_one_graph(
                buffers, sim_box, force_field, integrator, therm_arg, barostat, constraint,
                Some(coupling_dt), dt, timings, device, false, GraphVariant::CouplingForcesOnly,
            )?);
        }
    }

    Ok(crate::gpu::GraphLoop { graphs })
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
    // rq-3c78ea7d — the coupling variant is captured with the thermostat
    // and `coupling_dt = Some(coupling_interval * dt)` so `run_step` records
    // the thermostat's device-side coupling; non-coupling variants pass
    // `thermostat = None` and `coupling_dt = None`.
    thermostat: Option<&mut dyn crate::integrator::Thermostat>,
    barostat: &mut Option<Box<dyn crate::integrator::Barostat>>,
    constraint: &mut Option<Box<dyn crate::integrator::Constraint>>,
    coupling_dt: Option<Real>,
    dt: Real,
    timings: &mut Timings,
    device: &std::sync::Arc<cudarc::driver::CudaDevice>,
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
    {
        let constraint_arg: Option<&mut dyn crate::integrator::Constraint> = match constraint
            .as_mut()
        {
            Some(c) => Some(c.as_mut()),
            None => None,
        };
        let barostat_arg: Option<&mut dyn crate::integrator::Barostat> = match barostat.as_mut() {
            Some(b) => Some(b.as_mut()),
            None => None,
        };
        // `needs_scalars` selects the force evaluation's `AggregateLevel`:
        // `true` records the forces+scalars (`_fev`) graph, `false` the
        // forces-only (`_f`) graph. The replay loop launches the
        // forces-only graph on steps that need no scalars (see
        // `cuda-graphs.md` *Batched Replay Loop*). For the coupling variant
        // `thermostat` is `Some` and `coupling_dt = Some(coupling_interval *
        // dt)`, so `run_step` records the thermostat's `apply_pre` /
        // `apply_post` device-side coupling; non-coupling variants pass
        // `None` and record the thermostat as inert. `run_step` walks the
        // whole plan — the trailing kick, a terminal BarostatPoint's
        // `apply`, and the terminal velocity projection are all captured by
        // this one call. Marker-bearing (ThermostatHalf) and
        // interleaved-barostat plans never reach capture.
        let result = crate::integrator::run_step(
            integrator,
            buffers,
            sim_box,
            force_field,
            constraint_arg,
            thermostat,
            barostat_arg,
            dt,
            timings,
            crate::integrator::RunStepOptions {
                run_neighbor_pre_step: false,
                runner_needs_scalars: needs_scalars,
                coupling_dt,
            },
        );
        if let Err(e) = result {
            inner_failure = Some(format!("run_step: {e:?}"));
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
pub(crate) fn run_per_step_range(
    start_step: u64,
    n_steps: u64,
    setup: &mut SimulationSetup,
    phase: &crate::io::PhaseConfig,
    integrator: &mut Box<dyn crate::integrator::Integrator>,
    thermostat: &mut Option<Box<dyn crate::integrator::Thermostat>>,
    barostat: &mut Option<Box<dyn crate::integrator::Barostat>>,
    constraint: &mut Option<Box<dyn crate::integrator::Constraint>>,
    plan_owns_thermostat: bool,
    coupling_interval: u64,
    dt: Real,
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
    // rq-0286c77d — physics series appended to on each emitted log row.
    physics: &mut Vec<PhysicsSample>,
) -> Result<(), (RunnerError, ExitPhase)> {
    for step in start_step..=n_steps {
        // rq-ee10237d — a coupling step: the runner-wrapped thermostat
        // couples this step (every `coupling_interval` steps). The runner
        // owns the step counter and computes the cadence; `run_step` fires
        // apply_pre / apply_post at their canonical positions when passed
        // `coupling_dt = Some(dt_couple)`. `dt_couple = coupling_interval
        // · dt` scales the coupling to the elapsed interval.
        let couples = !plan_owns_thermostat
            && thermostat.is_some()
            && step % coupling_interval == 0;
        let coupling_dt = couples.then(|| coupling_interval as Real * dt);
        {
            let constraint_arg: Option<&mut dyn crate::integrator::Constraint> =
                match constraint.as_mut() {
                    Some(b) => Some(b.as_mut()),
                    None => None,
                };
            let thermostat_arg: Option<&mut dyn crate::integrator::Thermostat> =
                match thermostat.as_mut() {
                    Some(t) => Some(t.as_mut()),
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
            // rq-dbbffa7d — the barostat is passed so run_step can
            // dispatch an interleaved BarostatPoint; a terminal one is
            // deferred (skipped here, fired in the post-walk tail below).
            // Created after the shared-borrow helpers above.
            let barostat_arg: Option<&mut dyn crate::integrator::Barostat> =
                match barostat.as_mut() {
                    Some(b) => Some(b.as_mut()),
                    None => None,
                };
            let result = crate::integrator::run_step(
                integrator.as_mut(),
                &mut setup.buffers,
                &mut setup.sim_box,
                &mut setup.force_field,
                constraint_arg,
                thermostat_arg,
                barostat_arg,
                dt,
                &mut *timings,
                crate::integrator::RunStepOptions {
                    runner_needs_scalars,
                    coupling_dt,
                    ..Default::default()
                },
            );
            result.map_err(|e| {
                let runner_err = match e {
                    crate::integrator::StepError::Integrator(e) => RunnerError::Integrator(e),
                    crate::integrator::StepError::ForceField(e) => RunnerError::ForceField(e),
                    crate::integrator::StepError::Constraint(e) => RunnerError::Constraint(e),
                    crate::integrator::StepError::IntegratorRejectsConstraint { reason } => {
                        unreachable!("run_step returned IntegratorRejectsConstraint ({reason})")
                    }
                    crate::integrator::StepError::Gpu(e) => RunnerError::Gpu(e),
                    crate::integrator::StepError::Thermostat(e) => RunnerError::Thermostat(e),
                    crate::integrator::StepError::Barostat(e) => RunnerError::Barostat(e),
                    crate::integrator::StepError::Timings(e) => RunnerError::Timings(e),
                };
                (runner_err, ExitPhase::Loop)
            })?;
        }
        // rq-277dbeb2 — `run_step` walked the whole plan and dispatched
        // the entire post-force tail (the trailing kick, the wrapped
        // thermostat's apply_post on a coupling step, a terminal
        // BarostatPoint's apply, and the terminal velocity projection) in
        // canonical order. The runner adds no post-force tail of its own.

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
            // rq-0286c77d
            physics.push(
                capture_physics_sample(
                    step,
                    time,
                    ke,
                    n_thermal_dof,
                    &mut setup.buffers,
                    &setup.sim_box,
                )
                .map_err(|g| (RunnerError::Gpu(g), ExitPhase::Loop))?,
            );
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
pub(crate) fn run_batched_graph_loop(
    setup: &mut SimulationSetup,
    phase: &crate::io::PhaseConfig,
    _phase_index: usize,
    start_step: u64,
    graph_loop: &mut crate::gpu::GraphLoop,
    integrator: &mut Box<dyn crate::integrator::Integrator>,
    thermostat: &mut Option<Box<dyn crate::integrator::Thermostat>>,
    barostat: &mut Option<Box<dyn crate::integrator::Barostat>>,
    constraint: &mut Option<Box<dyn crate::integrator::Constraint>>,
    // rq-3c78ea7d — a plan-owned ThermostatHalf suppresses runner-wrapped
    // coupling; such phases are graph-ineligible, so `plan_owns_thermostat`
    // is always false here, but it is threaded for the single-coupling-step
    // per-step call, which shares `run_per_step_range`.
    plan_owns_thermostat: bool,
    // rq-3c78ea7d — the thermostat coupling cadence. Non-coupling steps
    // replay the graph; every `coupling_interval`-th step runs per-step.
    coupling_interval: u64,
    dt: Real,
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
    // rq-0286c77d — physics series appended to on each emitted log row.
    physics: &mut Vec<PhysicsSample>,
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

    // rq-3c78ea7d — a graph-eligible thermostat's coupling is recorded into
    // the coupling-variant graph; the loop selects that variant on coupling
    // steps (`step % coupling_interval == 0`) and a non-coupling variant
    // otherwise. Coupling steps are ordinary graph replays, not batch
    // boundaries.
    let thermostatted = thermostat.is_some() && !plan_owns_thermostat;

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
        // rq-6887c76d — one replay counter per cell of the coupling × scalars
        // matrix: [coupling][scalars].
        let mut cell_launches: [[u32; 2]; 2] = [[0, 0], [0, 0]];
        for i in 1..=batch {
            let s = step + i;
            // rq-c4ba5005 — a coupling step replays a coupling cell (which
            // carries the thermostat's device-side coupling), and within the
            // row the forces-only cell unless the step needs scalars.
            let couples = thermostatted && step_couples(s, coupling_interval);
            // rq-76db55bb — a step needs force-kernel scalars (total PE +
            // virial) only when it produces a log row or a barostat consumes
            // the per-step virial. Other steps replay a forces-only cell.
            let needs_scalars =
                step_needs_force_scalars(log_every > 0 && s % log_every == 0, barostat_active)
                    || move_frequency.is_some_and(|f| f > 0 && s % f == 0);
            graph_loop.launch(couples, needs_scalars).map_err(|e| {
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
            cell_launches[couples as usize][needs_scalars as usize] += 1;
        }
        // rq-9ec19227 — advance per-stage sample counts per matrix cell, so a
        // stage present only in some cells (the scalar reductions and `_fev`
        // kernel in the forces+scalars cells; the thermostat kernels in the
        // coupling cells) accrues samples only from the steps that replay a
        // cell containing it. Durations are the calibrated representatives
        // (see `cuda-graphs.md`). A step selecting a forces-only cell that
        // was not captured (per-step barostat) launches its row's
        // forces+scalars cell via the `launch` fallback, so the fallback is
        // impossible here — the forces-only column is captured exactly when
        // `needs_scalars` can be false.
        let variants = [
            [GraphVariant::ForcesOnly, GraphVariant::ForcesAndScalars],
            [
                GraphVariant::CouplingForcesOnly,
                GraphVariant::CouplingForcesAndScalars,
            ],
        ];
        for c in 0..2 {
            for sc in 0..2 {
                if cell_launches[c][sc] > 0 {
                    timings.record_graph_replays(variants[c][sc], cell_launches[c][sc]);
                }
            }
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
                integrator.as_mut(),
                thermostat,
                barostat,
                constraint,
                coupling_interval,
                dt,
                timings,
                &device,
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
            physics,
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

#[cfg(test)]
mod coupling_variant_tests {
    use super::step_couples;

    // rq-5b2a1cde — with coupling_interval = 25 over steps 1..=100, the
    // coupling steps (which replay the coupling variant) are exactly 25, 50,
    // 75, 100; every other step replays a non-coupling variant.
    #[test]
    fn coupling_steps_are_the_interval_multiples() {
        let ci = 25;
        let coupling: Vec<u64> = (1..=100).filter(|&s| step_couples(s, ci)).collect();
        assert_eq!(coupling, vec![25, 50, 75, 100]);
        // 96 non-coupling steps replay a non-coupling variant.
        let non_coupling = (1..=100u64).filter(|&s| !step_couples(s, ci)).count();
        assert_eq!(non_coupling, 96);
    }

    // rq-dce6f4cf — coupling_interval = 4 over 8 steps: coupling steps are 4
    // and 8; the six others (1,2,3,5,6,7) replay a non-coupling variant.
    #[test]
    fn coupling_variant_selected_on_interval_boundaries() {
        let ci = 4;
        let coupling: Vec<u64> = (1..=8).filter(|&s| step_couples(s, ci)).collect();
        assert_eq!(coupling, vec![4, 8]);
    }

    // rq-49f6bbfb — at coupling_interval == 1 every step couples, so every
    // step replays the coupling variant.
    #[test]
    fn interval_one_couples_every_step() {
        for s in 1..=10u64 {
            assert!(step_couples(s, 1));
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
