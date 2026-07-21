use super::*;
use std::time::Duration;

use crate::gpu::{ParticleBuffers, compute_total_potential_energy};
use crate::io::{
    LogWriter,
    TrajectoryWriter, TrajectoryWriterError,
};
use crate::io::log_output::{compute_kinetic_energy, compute_temperature};
use crate::state::{ParticleState, ParticleStateError};
use crate::timings::{
    HostStage, KernelStage, Timings,
};
use crate::precision::Real;

// rq-0286c77d — capture one `PhysicsSample` for a log row. The forces
// buffers hold valid scalars here because a log-due step evaluates
// forces-and-scalars (see `step_needs_force_scalars`), so the potential
// energy and virial reductions read current-configuration data. Uses a
// throwaway length-1 reduction scratch; runs only at the log cadence.
pub(crate) fn capture_physics_sample(
    step: u64,
    time: f64,
    ke: f64,
    n_thermal_dof: u32,
    buffers: &mut ParticleBuffers,
    sim_box: &crate::pbc::SimulationBox,
) -> Result<PhysicsSample, crate::gpu::GpuError> {
    let mut scratch = buffers
        .device
        .alloc_zeros::<Real>(1)
        .map_err(crate::gpu::GpuError::from)?;
    let pe = compute_total_potential_energy(buffers, &mut scratch)? as f64;
    let virial = crate::gpu::compute_total_virial(buffers, &mut scratch)? as f64;
    let volume = sim_box.volume() as f64;
    let pressure = if volume > 0.0 {
        (2.0 * ke + virial) / (3.0 * volume)
    } else {
        0.0
    };
    Ok(PhysicsSample {
        step,
        time,
        kinetic_energy: ke,
        potential_energy: pe,
        total_energy: ke + pe,
        temperature: compute_temperature(ke, n_thermal_dof),
        pressure,
        volume,
    })
}

/// Per-step output handler shared by the per-step launch loop and the
/// batched graph-replay loop. Downloads the host frame, flushes the
/// simulation box, drains slot diagnostic accumulators, and writes
/// log / trajectory rows if `step` aligns with the configured cadence.
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_step_output(
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
    // rq-0286c77d — physics series appended to on each emitted log row.
    physics: &mut Vec<PhysicsSample>,
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
        // rq-0286c77d
        physics.push(
            capture_physics_sample(step, time, ke, n_thermal_dof, &mut setup.buffers, &setup.sim_box)
                .map_err(|g| (RunnerError::Gpu(g), ExitPhase::Loop))?,
        );
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
pub(crate) fn collect_log_extras(
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

pub(crate) fn write_traj_frame(
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
