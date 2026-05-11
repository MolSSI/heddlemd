// rq-357909e4 rq-02edd314 rq-77c1d5d9
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::gpu::{
    LennardJonesParameters, LosslessBuffers, PairBuffer, ParticleBuffers, init_device, lj_pair_force,
    reduce_pair_forces, vv_kick, vv_kick_drift, vv_kick_drift_lossless, vv_kick_lossless,
};
use crate::io::{
    ConfigError, InitStateError, InitVelocities, LogWriter, LogWriterError, TrajectoryWriter,
    TrajectoryWriterError, load_config, load_init_state,
};
use crate::io::log_output::{BOLTZMANN_J_PER_K, compute_kinetic_energy, compute_temperature};
use crate::state::{ParticleState, ParticleStateError};

// rq-8ee27e27
#[derive(Debug)]
pub enum RunnerError {
    Config(ConfigError),
    InitState(InitStateError),
    ParticleState(ParticleStateError),
    Gpu(crate::gpu::GpuError),
    Trajectory(TrajectoryWriterError),
    Log(LogWriterError),
    MissingArgs,
    OutputExists { path: PathBuf },
}

impl std::fmt::Display for RunnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunnerError::Config(e) => write!(f, "Config({e})"),
            RunnerError::InitState(e) => write!(f, "InitState({e})"),
            RunnerError::ParticleState(e) => write!(f, "ParticleState({e})"),
            RunnerError::Gpu(e) => write!(f, "Gpu({e})"),
            RunnerError::Trajectory(e) => write!(f, "Trajectory({e})"),
            RunnerError::Log(e) => write!(f, "Log({e})"),
            RunnerError::MissingArgs => write!(f, "MissingArgs"),
            RunnerError::OutputExists { path } => {
                write!(f, "OutputExists {{ path: {} }}", path.display())
            }
        }
    }
}

impl std::error::Error for RunnerError {}

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

const USAGE_LINE: &str = "usage: dynamics run <config-path>";

// rq-1fc57c00 rq-e5e4b048
pub fn run_simulation(config_path: &Path) -> Result<RunSummary, RunnerError> {
    run_simulation_with_phase(config_path).map_err(|(e, _)| e)
}

// rq-ef902cf6
fn run_simulation_with_phase(
    config_path: &Path,
) -> Result<RunSummary, (RunnerError, ExitPhase)> {
    let config = load_config(config_path).map_err(|e| (RunnerError::Config(e), ExitPhase::Setup))?;

    // Pre-flight output existence checks (only for enabled outputs).
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

    let type_name_strings: Vec<String> = config
        .particle_types
        .iter()
        .map(|t| t.name.clone())
        .collect();
    let type_name_refs: Vec<&str> = type_name_strings.iter().map(|s| s.as_str()).collect();
    let init = load_init_state(&config.init, &type_name_refs)
        .map_err(|e| (RunnerError::InitState(e), ExitPhase::Setup))?;

    let sim_box = init.sim_box;
    let n = init.particle_count;

    let device = init_device().map_err(|e| (RunnerError::Gpu(e), ExitPhase::Setup))?;

    // Build masses array from per-particle type_index lookup.
    let mut masses_f64: Vec<f64> = Vec::with_capacity(n);
    let mut masses_f32: Vec<f32> = Vec::with_capacity(n);
    for &ti in &init.type_indices {
        let m = config.particle_types[ti as usize].mass;
        masses_f64.push(m);
        masses_f32.push(m as f32);
    }

    // Build velocities: either from the init state or sampled.
    let (velocities_x, velocities_y, velocities_z) = match init.velocities {
        Some(InitVelocities {
            velocities_x,
            velocities_y,
            velocities_z,
        }) => (velocities_x, velocities_y, velocities_z),
        None => generate_velocities(
            n,
            config.simulation.temperature,
            config.simulation.seed,
            &masses_f64,
        ),
    };

    let state = ParticleState::new(
        init.positions_x.clone(),
        init.positions_y.clone(),
        init.positions_z.clone(),
        velocities_x,
        velocities_y,
        velocities_z,
        masses_f32.clone(),
        None,
    )
    .map_err(|e| (RunnerError::ParticleState(e), ExitPhase::Setup))?;

    let mut buffers = ParticleBuffers::new(device.clone(), &state)
        .map_err(|e| match e {
            ParticleStateError::Gpu(g) => (RunnerError::Gpu(g), ExitPhase::Setup),
            other => (RunnerError::ParticleState(other), ExitPhase::Setup),
        })?;

    let max_neighbors = if n == 0 { 0u32 } else { n as u32 };
    let mut pair_buffer = PairBuffer::new(device.clone(), n, max_neighbors)
        .map_err(|e| (RunnerError::Gpu(e), ExitPhase::Setup))?;

    let mut lossless: Option<LosslessBuffers> = if config.integrator.lossless {
        Some(
            LosslessBuffers::new(device.clone(), n)
                .map_err(|e| (RunnerError::Gpu(e), ExitPhase::Setup))?,
        )
    } else {
        None
    };

    let neighbor_counts = device
        .htod_sync_copy(&vec![n as u32; n])
        .map_err(|e| (RunnerError::Gpu(crate::gpu::GpuError::from(e)), ExitPhase::Setup))?;

    let pair = &config.pair_interactions[0];
    let lj_params = LennardJonesParameters {
        sigma: pair.sigma as f32,
        epsilon: pair.epsilon as f32,
        cutoff: pair.cutoff as f32,
    };

    // Open output writers (only the enabled ones).
    let mut traj_writer: Option<TrajectoryWriter> = if config.output.trajectory_every > 0 {
        Some(
            TrajectoryWriter::open(
                &config.output.trajectory_path,
                config.output.include_velocities,
                type_name_strings.clone(),
            )
            .map_err(|e| (RunnerError::Trajectory(e), ExitPhase::Setup))?,
        )
    } else {
        None
    };
    let mut log_writer: Option<LogWriter> = if config.output.log_every > 0 {
        Some(
            LogWriter::open(&config.output.log_path)
                .map_err(|e| (RunnerError::Log(e), ExitPhase::Setup))?,
        )
    } else {
        None
    };

    let started = Instant::now();

    // Warm-up: populate forces with F(x_0).
    lj_pair_force(&buffers, &mut pair_buffer, &sim_box, &lj_params)
        .map_err(|e| (RunnerError::Gpu(e), ExitPhase::Setup))?;
    reduce_pair_forces(&pair_buffer, &neighbor_counts, &mut buffers)
        .map_err(|e| (RunnerError::Gpu(e), ExitPhase::Setup))?;

    // Host scratch state for downloads.
    let mut frame: ParticleState = state.clone();
    let mut frames_written: u64 = 0;
    let mut log_rows_written: u64 = 0;

    // Step-0 outputs.
    if traj_writer.is_some() || log_writer.is_some() {
        frame
            .download_from(&buffers)
            .map_err(|e| match e {
                ParticleStateError::Gpu(g) => (RunnerError::Gpu(g), ExitPhase::Setup),
                other => (RunnerError::ParticleState(other), ExitPhase::Setup),
            })?;
    }
    if let Some(writer) = traj_writer.as_mut() {
        write_traj_frame(writer, 0, config.simulation.dt, &sim_box, &init.type_indices, &frame)
            .map_err(|e| (RunnerError::Trajectory(e), ExitPhase::Setup))?;
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
        writer
            .write_row(0, 0.0, ke, t)
            .map_err(|e| (RunnerError::Log(e), ExitPhase::Setup))?;
        log_rows_written += 1;
    }

    let progress_to_stdout = std::io::IsTerminal::is_terminal(&std::io::stdout());
    let n_steps = config.simulation.n_steps;
    let progress_every = (n_steps / 100).max(1);
    let dt_f32 = config.simulation.dt as f32;

    // Main loop.
    for step in 1..=n_steps {
        if let Some(ll) = lossless.as_mut() {
            vv_kick_drift_lossless(&mut buffers, ll, dt_f32)
                .map_err(|e| (RunnerError::Gpu(e), ExitPhase::Loop))?;
        } else {
            vv_kick_drift(&mut buffers, dt_f32)
                .map_err(|e| (RunnerError::Gpu(e), ExitPhase::Loop))?;
        }
        lj_pair_force(&buffers, &mut pair_buffer, &sim_box, &lj_params)
            .map_err(|e| (RunnerError::Gpu(e), ExitPhase::Loop))?;
        reduce_pair_forces(&pair_buffer, &neighbor_counts, &mut buffers)
            .map_err(|e| (RunnerError::Gpu(e), ExitPhase::Loop))?;
        if let Some(ll) = lossless.as_mut() {
            vv_kick_lossless(&mut buffers, ll, dt_f32)
                .map_err(|e| (RunnerError::Gpu(e), ExitPhase::Loop))?;
        } else {
            vv_kick(&mut buffers, dt_f32)
                .map_err(|e| (RunnerError::Gpu(e), ExitPhase::Loop))?;
        }

        let want_traj =
            config.output.trajectory_every > 0 && step % config.output.trajectory_every == 0;
        let want_log = config.output.log_every > 0 && step % config.output.log_every == 0;
        if want_traj || want_log {
            frame
                .download_from(&buffers)
                .map_err(|e| match e {
                    ParticleStateError::Gpu(g) => (RunnerError::Gpu(g), ExitPhase::Loop),
                    other => (RunnerError::ParticleState(other), ExitPhase::Loop),
                })?;
        }
        if want_traj {
            let writer = traj_writer.as_mut().expect("traj_writer enabled");
            write_traj_frame(writer, step, config.simulation.dt, &sim_box, &init.type_indices, &frame)
                .map_err(|e| (RunnerError::Trajectory(e), ExitPhase::Loop))?;
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
            writer
                .write_row(step, time, ke, t)
                .map_err(|e| (RunnerError::Log(e), ExitPhase::Loop))?;
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
    Ok(RunSummary {
        n_steps,
        frames_written,
        log_rows_written,
        elapsed_micros,
    })
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
    writer.write_frame(
        step,
        dt,
        sim_box,
        &type_indices[..n],
        &frame.positions_x[..n],
        &frame.positions_y[..n],
        &frame.positions_z[..n],
        traj_velocities,
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
    if sub != "run" {
        eprintln!("{USAGE_LINE}");
        return 1;
    }
    let config_path = match iter.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("{USAGE_LINE}");
            return 1;
        }
    };

    match run_simulation_with_phase(&config_path) {
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
