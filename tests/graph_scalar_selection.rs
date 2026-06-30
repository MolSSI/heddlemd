//! Forces-only / forces+scalars graph selection (option 2). A
//! graph-eligible phase captures a forces-only and a forces+scalars
//! graph and replays the cheaper forces-only graph except on steps that
//! need the total potential energy or virial. See `rqm/cuda-graphs.md`.

use std::path::{Path, PathBuf};

use heddle_md::gpu::{
    CaptureMode, GraphLoop, begin_stream_capture, end_stream_capture, init_device,
};
use heddle_md::runner::run_simulation;

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("heddle_graph_scalar_{name}"));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// A small disordered LJ liquid (deterministic jitter, so the input is
/// byte-identical every call) spanning many atom-blocks.
fn write_lj_liquid(dir: &Path) {
    let side = 10usize;
    let spacing = 4.0e-10;
    let n = side * side * side;
    let l = side as f64 * spacing;
    let c = (side as f64 - 1.0) / 2.0;
    let mut lcg: u64 = 0x1234_5678;
    let mut jitter = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((lcg >> 33) as f64 / (1u64 << 31) as f64) - 0.5) * 0.6 * spacing
    };
    let mut body = format!("{n}\n");
    body.push_str(&format!(
        "Lattice=\"{l:.6e} 0 0 0 {l:.6e} 0 0 0 {l:.6e}\" Properties=species:S:1:pos:R:3\n"
    ));
    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                let px = (i as f64 - c) * spacing + jitter();
                let py = (j as f64 - c) * spacing + jitter();
                let pz = (k as f64 - c) * spacing + jitter();
                body.push_str(&format!("Ar {px:.9e} {py:.9e} {pz:.9e}\n"));
            }
        }
    }
    std::fs::write(dir.join("sim.in.xyz"), body).unwrap();
}

/// Assemble an MD config in graph mode (the default). `thermostat` /
/// `barostat` are inserted verbatim; `output` sets the cadences.
fn write_config(dir: &Path, thermostat: &str, barostat: &str, n_steps: u64, output: &str) {
    write_lj_liquid(dir);
    let cfg = format!(
        r#"schema_version = 1
init = "sim.in.xyz"

[simulation]
seed = 1
temperature = 100.0

[[phase]]
name = "run"
n_steps = {n_steps}
dt = 1.0e-15

[phase.integrator]
kind = "velocity-verlet"
lossless = false

{thermostat}
{barostat}

[phase.output]
{output}

[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 9.0e-10

[neighbor_list]
mode = "cell-list"
r_skin = 3.0e-10
"#
    );
    std::fs::write(dir.join("sim.in.toml"), cfg).unwrap();
}

const CSVR: &str = r#"[phase.thermostat]
kind = "csvr"
temperature = 100.0
tau = 1.0e-13
seed = 7"#;

const C_RESCALE: &str = r#"[phase.barostat]
kind = "c-rescale"
pressure = 1.0e5
temperature = 100.0
tau = 1.0e-12
compressibility = 4.5e-10
seed = 9"#;

/// Sample count recorded for `stage` in a phase's `.timings` file, or
/// `None` when the stage has no row (it never ran).
fn timings_count(dir: &Path, stage: &str) -> Option<u64> {
    let text = std::fs::read_to_string(dir.join("sim.out.run.timings")).unwrap();
    for line in text.lines() {
        let mut cols = line.split_whitespace();
        if cols.next() == Some(stage) {
            return cols.next().and_then(|c| c.parse::<u64>().ok());
        }
    }
    None
}

/// Number of data rows (excluding the header) in a phase's `.log` file.
fn log_row_count(dir: &Path) -> u64 {
    let text = std::fs::read_to_string(dir.join("sim.out.run.log")).unwrap();
    text.lines().skip(1).filter(|l| !l.trim().is_empty()).count() as u64
}

// rq-867630af rq-8c24f057
// Build a GraphLoop from two captured single-node graphs that zero
// distinct device buffers, then observe which buffer a launch zeroes to
// confirm the `scalars` flag selects the graph.
#[test]
fn graph_loop_launch_routes_by_scalars_flag() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();

    let mut buf_fo = device.htod_copy(vec![7u32]).unwrap();
    let mut buf_fev = device.htod_copy(vec![7u32]).unwrap();

    begin_stream_capture(&device, CaptureMode::ThreadLocal).unwrap();
    device.memset_zeros(&mut buf_fo).unwrap();
    let g_fo = end_stream_capture(&device).unwrap().instantiate().unwrap();

    begin_stream_capture(&device, CaptureMode::ThreadLocal).unwrap();
    device.memset_zeros(&mut buf_fev).unwrap();
    let g_fev = end_stream_capture(&device).unwrap().instantiate().unwrap();

    let gl = GraphLoop {
        forces_and_scalars: g_fev,
        forces_only: Some(g_fo),
    };

    // scalars = false -> forces-only graph (zeros buf_fo, leaves buf_fev).
    gl.launch(false).unwrap();
    device.synchronize().unwrap();
    assert_eq!(device.dtoh_sync_copy(&buf_fo).unwrap()[0], 0);
    assert_eq!(device.dtoh_sync_copy(&buf_fev).unwrap()[0], 7);

    // scalars = true -> forces+scalars graph (zeros buf_fev).
    device.htod_copy_into(vec![7u32], &mut buf_fo).unwrap();
    gl.launch(true).unwrap();
    device.synchronize().unwrap();
    assert_eq!(device.dtoh_sync_copy(&buf_fev).unwrap()[0], 0);
    assert_eq!(device.dtoh_sync_copy(&buf_fo).unwrap()[0], 7);

    // forces_only = None -> scalars = false still routes to forces+scalars.
    let mut buf_none = device.htod_copy(vec![7u32]).unwrap();
    begin_stream_capture(&device, CaptureMode::ThreadLocal).unwrap();
    device.memset_zeros(&mut buf_none).unwrap();
    let g_none = end_stream_capture(&device).unwrap().instantiate().unwrap();
    let gl_none = GraphLoop {
        forces_and_scalars: g_none,
        forces_only: None,
    };
    gl_none.launch(false).unwrap();
    device.synchronize().unwrap();
    assert_eq!(device.dtoh_sync_copy(&buf_none).unwrap()[0], 0);
}

// rq-009eed1b rq-26dce0f6
// NVT (thermostat, no barostat): the total potential energy is reduced
// only on log steps, so `potential_energy_reduce` runs once per log row
// rather than once per physical step.
#[test]
fn nvt_graph_phase_reduces_scalars_only_on_log_steps() {
    let dir = tmp("nvt_log");
    write_config(
        &dir,
        CSVR,
        "",
        40,
        "log_every = 10\ntrajectory_every = 0",
    );
    run_simulation(&dir.join("sim.in.toml")).unwrap();
    let pe = timings_count(&dir, "potential_energy_reduce").unwrap_or(0);
    let rows = log_row_count(&dir);
    assert!(rows > 0 && rows < 40, "log rows = {rows}");
    assert_eq!(
        pe, rows,
        "potential_energy_reduce ran {pe} times; expected one per log row ({rows}), not per step"
    );
}

// rq-d34ff8f7
// A phase that writes trajectory frames but no log rows and has no
// barostat never needs force-kernel scalars: `potential_energy_reduce`
// never runs.
#[test]
fn trajectory_only_graph_phase_never_reduces_scalars() {
    let dir = tmp("traj_only");
    write_config(
        &dir,
        CSVR,
        "",
        40,
        "log_every = 0\ntrajectory_every = 10\ninclude_velocities = false",
    );
    run_simulation(&dir.join("sim.in.toml")).unwrap();
    assert_eq!(
        timings_count(&dir, "potential_energy_reduce").unwrap_or(0),
        0,
        "no log rows and no barostat: scalars should never be reduced"
    );
}

// rq-1b40a671 rq-c6c56cdc
// NPT (c-rescale barostat): the barostat consumes the per-step virial,
// so every step evaluates scalars and `virial_sum_reduce` runs once per
// physical step.
#[test]
fn npt_graph_phase_reduces_virial_every_step() {
    let dir = tmp("npt");
    let n_steps = 40u64;
    write_config(
        &dir,
        CSVR,
        C_RESCALE,
        n_steps,
        "log_every = 10\ntrajectory_every = 0",
    );
    run_simulation(&dir.join("sim.in.toml")).unwrap();
    assert_eq!(
        timings_count(&dir, "virial_sum_reduce").unwrap_or(0),
        n_steps,
        "a barostat phase must compute the virial on every step"
    );
}

// rq-dae13654
// Replaying the forces-only graph for non-log steps does not perturb the
// trajectory: two graph-mode NVT runs on the same GPU are byte-identical.
#[test]
fn forces_only_replay_is_byte_identical_run_to_run() {
    let out = |name: &str| {
        let dir = tmp(name);
        write_config(
            &dir,
            CSVR,
            "",
            30,
            "log_every = 10\ntrajectory_every = 1",
        );
        run_simulation(&dir.join("sim.in.toml")).unwrap();
        std::fs::read(dir.join("sim.out.run.xyz")).unwrap()
    };
    assert_eq!(
        out("det_a"),
        out("det_b"),
        "graph-mode NVT trajectory is not run-to-run deterministic"
    );
}
