// rq-357909e4 rq-02edd314 rq-77c1d5d9
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::forces::{
    BondList, BondsFileError, ExclusionList, ForceField, ForceFieldError, load_bonds_file,
};
use crate::gpu::{ParticleBuffers, init_device};
use crate::integrator::{IntegratorError, IntegratorRegistry};
use crate::io::config::NeighborListConfig;
use crate::io::{
    ConfigError, InitStateError, InitVelocities, LogWriter, LogWriterError, TrajectoryWriter,
    TrajectoryWriterError, load_config, load_init_state,
};
use crate::io::log_output::{BOLTZMANN_J_PER_K, compute_kinetic_energy, compute_temperature};
use crate::state::{ParticleState, ParticleStateError};
use crate::timings::{
    HostStage, Timings, TimingsError, TimingsWriterError, write_timings_file,
};

// rq-8ee27e27
#[derive(Debug)]
pub enum RunnerError {
    Config(ConfigError),
    InitState(InitStateError),
    ParticleState(ParticleStateError),
    Gpu(crate::gpu::GpuError),
    Integrator(IntegratorError),
    BondsFile(BondsFileError),
    ForceField(ForceFieldError),
    Trajectory(TrajectoryWriterError),
    Log(LogWriterError),
    Timings(TimingsError),
    TimingsWriter(TimingsWriterError),
    MissingArgs,
    OutputExists { path: PathBuf },
    CellListBoxTooSmall {
        axis: &'static str,
        length: f32,
        required: f32,
    },
}

impl std::fmt::Display for RunnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunnerError::Config(e) => write!(f, "Config({e})"),
            RunnerError::InitState(e) => write!(f, "InitState({e})"),
            RunnerError::ParticleState(e) => write!(f, "ParticleState({e})"),
            RunnerError::Gpu(e) => write!(f, "Gpu({e})"),
            RunnerError::Integrator(e) => write!(f, "Integrator({e})"),
            RunnerError::BondsFile(e) => write!(f, "BondsFile({e})"),
            RunnerError::ForceField(e) => write!(f, "ForceField({e})"),
            RunnerError::Trajectory(e) => write!(f, "Trajectory({e})"),
            RunnerError::Log(e) => write!(f, "Log({e})"),
            RunnerError::Timings(e) => write!(f, "Timings({e})"),
            RunnerError::TimingsWriter(e) => write!(f, "TimingsWriter({e})"),
            RunnerError::MissingArgs => write!(f, "MissingArgs"),
            RunnerError::OutputExists { path } => {
                write!(f, "OutputExists {{ path: {} }}", path.display())
            }
            RunnerError::CellListBoxTooSmall {
                axis,
                length,
                required,
            } => write!(
                f,
                "CellListBoxTooSmall {{ axis: {axis:?}, length: {length}, required: {required} }}"
            ),
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

fn timed<T>(target: &mut Duration, f: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let value = f();
    *target = started.elapsed();
    value
}

// rq-ef902cf6 rq-dcfdb7c9
fn run_simulation_with_phase(
    config_path: &Path,
) -> Result<RunSummary, (RunnerError, ExitPhase)> {
    let total_started = Instant::now();

    // Time config_load before any other instrumentation exists.
    let mut config_load_duration = Duration::ZERO;
    let config = timed(&mut config_load_duration, || load_config(config_path))
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
    if let NeighborListConfig::CellList { r_skin, .. } = &config.neighbor_list {
        let cutoff_max: f64 = config
            .pair_interactions
            .iter()
            .map(|p| p.cutoff)
            .fold(0.0, f64::max);
        let required = (3.0 * (cutoff_max + r_skin)) as f32;
        let lengths = sim_box.lengths();
        let axis_names: [&'static str; 3] = ["x", "y", "z"];
        for a in 0..3 {
            if lengths[a] < required {
                return Err((
                    RunnerError::CellListBoxTooSmall {
                        axis: axis_names[a],
                        length: lengths[a],
                        required,
                    },
                    ExitPhase::Setup,
                ));
            }
        }
    }

    let mut gpu_init_duration = Duration::ZERO;
    let device = timed(&mut gpu_init_duration, init_device)
        .map_err(|e| (RunnerError::Gpu(e), ExitPhase::Setup))?;

    // Construct the Timings instance now and replay the three pre-instrumented
    // host stages.
    let mut timings = Timings::new(device.clone())
        .map_err(|e| (RunnerError::Timings(e), ExitPhase::Setup))?;
    timings.record_host(HostStage::CONFIG_LOAD, config_load_duration);
    timings.record_host(HostStage::INIT_LOAD, init_load_duration);
    timings.record_host(HostStage::GPU_INIT, gpu_init_duration);

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

    let state = ParticleState::new(
        init.positions_x.clone(),
        init.positions_y.clone(),
        init.positions_z.clone(),
        velocities_x,
        velocities_y,
        velocities_z,
        masses_f32.clone(),
        init.type_indices.clone(),
        None,
    )
    .map_err(|e| (RunnerError::ParticleState(e), ExitPhase::Setup))?;

    let mut upload = Duration::ZERO;
    let mut buffers = timed(&mut upload, || ParticleBuffers::new(device.clone(), &state))
        .map_err(|e| match e {
            ParticleStateError::Gpu(g) => (RunnerError::Gpu(g), ExitPhase::Setup),
            other => (RunnerError::ParticleState(other), ExitPhase::Setup),
        })?;
    timings.record_host(HostStage::HOST_TO_DEVICE_UPLOAD, upload);

    let registry = IntegratorRegistry::with_builtins();
    let mut integrator = registry
        .build(&config.integrator, device.clone(), n)
        .map_err(|e| (RunnerError::Integrator(e), ExitPhase::Setup))?;

    // Load the .bonds file when supplied, otherwise build empty bond / exclusion
    // lists keyed to `n`.
    let bond_type_names: Vec<&str> =
        config.bond_types.iter().map(|bt| bt.name()).collect();
    let (bond_list, exclusion_list): (BondList, ExclusionList) = match config.bonds.as_ref() {
        Some(path) => load_bonds_file(path, n, &bond_type_names)
            .map_err(|e| (RunnerError::BondsFile(e), ExitPhase::Setup))?,
        None => (BondList::empty(n), ExclusionList::empty(n)),
    };

    let mut force_field = ForceField::new(
        device.clone(),
        n,
        &sim_box,
        &config.particle_types,
        &config.pair_interactions,
        &config.bond_types,
        &bond_list,
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
        let mut lw = Duration::ZERO;
        timed(&mut lw, || writer.write_row(0, 0.0, ke, t))
            .map_err(|e| (RunnerError::Log(e), ExitPhase::Setup))?;
        timings.record_host(HostStage::LOG_WRITE, lw);
        log_rows_written += 1;
    }

    let progress_to_stdout = std::io::IsTerminal::is_terminal(&std::io::stdout());
    let n_steps = config.simulation.n_steps;
    let progress_every = (n_steps / 100).max(1);
    let dt_f32 = config.simulation.dt as f32;

    // Main loop.
    for step in 1..=n_steps {
        integrator
            .step(
                &mut buffers,
                &mut sim_box,
                &mut force_field,
                dt_f32,
                step,
                &mut timings,
            )
            .map_err(|e| (RunnerError::Integrator(e), ExitPhase::Loop))?;

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
            let mut lw = Duration::ZERO;
            timed(&mut lw, || writer.write_row(step, time, ke, t))
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
