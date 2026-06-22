use std::path::{Path, PathBuf};
use std::time::Duration;

mod common;
use common::*;

use heddle_md::gpu::{ParticleBuffers, init_device};
use heddle_md::io::{TrajectoryWriterError, load_init_state};
use heddle_md::pbc::SimulationBox;
use heddle_md::runner::{RunnerError, run_simulation};
use heddle_md::state::ParticleState;
use heddle_md::timings::{
    HostStage, KernelStage, StageStats, Timings, TimingsReport, TimingsWriterError,
    write_timings_file,
};

fn tmp_path(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!(
        "heddlemd-timings-{}-{}-{}",
        std::process::id(),
        name,
        nanos
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn write_pair(
    dir: &Path,
    n_steps: u64,
    traj_every: u64,
    log_every: u64,
    temperature: f64,
    init_velocities: bool,
    lossless: bool,
    seed: u64,
    n_particles: usize,
) -> PathBuf {
    let lossless_str = if lossless { "true" } else { "false" };
    let config = format!(
        r#"schema_version = 1
init = "sim.in.xyz"

[simulation]
cuda_graphs_disable = true
seed = {seed}
temperature = {temperature}

[[phase]]
name = "run"
n_steps = {n_steps}
dt = 1.0e-15


[phase.integrator]
kind = "velocity-verlet"
lossless = {lossless_str}

[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 1.0e-9

[neighbor_list]
mode = "all-pairs"

[phase.output]
trajectory_every = {traj_every}
log_every = {log_every}
"#
    );
    let config_path = dir.join("sim.in.toml");
    std::fs::write(&config_path, config).unwrap();
    write_init(dir, n_particles, init_velocities);
    config_path
}

fn write_init(dir: &Path, n: usize, include_velocities: bool) {
    let mut body = String::new();
    body.push_str(&format!("{n}\n"));
    let props = if include_velocities {
        "species:S:1:pos:R:3:velo:R:3"
    } else {
        "species:S:1:pos:R:3"
    };
    body.push_str(&format!(
        "Lattice=\"4.0e-9 0 0 0 4.0e-9 0 0 0 4.0e-9\" Properties={props}\n"
    ));
    let box_size = 4.0e-9_f64;
    let usable = box_size * 0.8;
    let spacing = if n > 1 { usable / (n as f64 - 1.0) } else { 0.0 };
    let half_n = (n as f64 - 1.0) / 2.0;
    for i in 0..n {
        let x = (i as f64 - half_n) * spacing;
        if include_velocities {
            body.push_str(&format!("Ar {x:.9e} 0.0 0.0 0.0 0.0 0.0\n"));
        } else {
            body.push_str(&format!("Ar {x:.9e} 0.0 0.0\n"));
        }
    }
    std::fs::write(dir.join("sim.in.xyz"), body).unwrap();
}

fn read_timings(dir: &Path) -> String {
    let canon = std::fs::canonicalize(dir).unwrap();
    std::fs::read_to_string(canon.join("sim.out.run.timings")).unwrap()
}

fn stage_row<'a>(body: &'a str, name: &str) -> Option<&'a str> {
    for line in body.lines().skip(1) {
        let first = line.split_whitespace().next()?;
        if first == name {
            return Some(line);
        }
    }
    None
}

fn stage_count(body: &str, name: &str) -> Option<u64> {
    let line = stage_row(body, name)?;
    let cols: Vec<&str> = line.split_whitespace().collect();
    Some(cols[1].parse::<u64>().unwrap())
}

// rq-f5e25186
#[test]
fn successful_run_writes_timings_file() {
    let dir = tmp_path("write_timings");
    let path = write_pair(&dir, 5, 5, 5, 0.0, true, false, 1, 2);
    run_simulation(&path).unwrap();
    let canon = std::fs::canonicalize(&dir).unwrap();
    let timings = canon.join("sim.out.run.timings");
    assert!(timings.exists());
    let body = std::fs::read_to_string(&timings).unwrap();
    let header = body.lines().next().unwrap();
    let cols: Vec<&str> = header.split_whitespace().collect();
    assert_eq!(
        cols,
        vec!["stage", "count", "total_ms", "mean_us", "min_us", "max_us"]
    );
}

// rq-86423766
#[test]
fn timings_absent_on_setup_failure() {
    let dir = tmp_path("setup_fail");
    let path = write_pair(&dir, 1, 0, 0, 0.0, false, false, 1, 1);
    // Make the init file malformed so init_load fails after config_load.
    std::fs::write(
        dir.join("sim.in.xyz"),
        "1\nLattice=\"4.0e-9 0 0 0 4.0e-9 0 0 0 4.0e-9\" Properties=species:S:1:pos:R:3\nAr 3.0e-9 0.0 0.0\n",
    )
    .unwrap();
    let _ = run_simulation(&path);
    let canon = std::fs::canonicalize(&dir).unwrap();
    assert!(!canon.join("sim.out.run.timings").exists());
}

// rq-afb80e25
#[test]
fn timings_file_uses_default_path() {
    let dir = tmp_path("timings_default_path");
    let path = write_pair(&dir, 1, 0, 0, 0.0, true, false, 1, 1);
    run_simulation(&path).unwrap();
    let canon = std::fs::canonicalize(&dir).unwrap();
    assert!(canon.join("sim.out.run.timings").exists());
}

// rq-a2ebdaaf
#[test]
fn timings_file_can_be_overridden() {
    let dir = tmp_path("timings_override");
    let path = write_pair(&dir, 1, 0, 0, 0.0, true, false, 1, 1);
    // Rewrite config with a timings_path override.
    let body = std::fs::read_to_string(&path).unwrap();
    let body = body.replace(
        "[phase.output]",
        "[phase.output]\ntimings_path = \"custom.timings\"",
    );
    std::fs::write(&path, body).unwrap();
    run_simulation(&path).unwrap();
    let canon = std::fs::canonicalize(&dir).unwrap();
    assert!(canon.join("custom.timings").exists());
    assert!(!canon.join("sim.out.run.timings").exists());
}

// rq-11132169
#[test]
fn timings_pre_flight_refuses_overwrite() {
    let dir = tmp_path("timings_overwrite");
    let path = write_pair(&dir, 1, 0, 0, 0.0, true, false, 1, 1);
    let canon = std::fs::canonicalize(&dir).unwrap();
    let timings = canon.join("sim.out.run.timings");
    std::fs::write(&timings, "existing").unwrap();
    let err = run_simulation(&path).unwrap_err();
    match err {
        RunnerError::OutputExists { path: p } => assert_eq!(p, timings),
        other => panic!("unexpected: {other:?}"),
    }
    assert_eq!(std::fs::read_to_string(&timings).unwrap(), "existing");
}

// rq-a7fdf81f
#[test]
fn lossy_includes_lossy_kick_rows() {
    let dir = tmp_path("lossy_kick");
    let path = write_pair(&dir, 5, 0, 0, 0.0, true, false, 1, 2);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    assert!(stage_row(&body, "vv_kick_drift").is_some());
    assert!(stage_row(&body, "vv_kick").is_some());
    assert!(stage_row(&body, "vv_kick_drift_lossless").is_none());
    assert!(stage_row(&body, "vv_kick_lossless").is_none());
}

// rq-b2fa4a1f
// Lossless mode is only available in the default (f32) build.
#[cfg(not(feature = "f64"))]
#[test]
fn lossless_includes_lossless_kick_rows() {
    let dir = tmp_path("lossless_kick");
    let path = write_pair(&dir, 5, 0, 0, 0.0, true, true, 1, 2);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    assert!(stage_row(&body, "vv_kick_drift_lossless").is_some());
    assert!(stage_row(&body, "vv_kick_lossless").is_some());
    assert!(stage_row(&body, "vv_kick_drift").is_none());
    assert!(stage_row(&body, "vv_kick").is_none());
}

fn write_langevin_config(dir: &Path, n_steps: u64) -> PathBuf {
    let cfg = format!(
        r#"schema_version = 1
init = "sim.in.xyz"

[simulation]
cuda_graphs_disable = true
seed = 1
temperature = 300.0

[[phase]]
name = "run"
n_steps = {n_steps}
dt = 1.0e-15


[phase.integrator]
kind = "langevin-baoab"
friction = 1.0e12
temperature = 300.0
seed = 42

[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 1.0e-9

[phase.output]
trajectory_every = 0
log_every = 0
"#
    );
    let path = dir.join("sim.in.toml");
    std::fs::write(&path, cfg).unwrap();
    std::fs::write(
        dir.join("sim.in.xyz"),
        "2\nLattice=\"4.0e-9 0 0 0 4.0e-9 0 0 0 4.0e-9\" Properties=species:S:1:pos:R:3\n\
         Ar -5.0e-10 0 0\nAr 5.0e-10 0 0\n",
    )
    .unwrap();
    path
}

// rq-c1c9fe3a
#[test]
fn langevin_includes_langevin_rows_excludes_vv() {
    let dir = tmp_path("langevin_rows");
    let path = write_langevin_config(&dir, 10);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    assert!(stage_row(&body, "langevin_kick_half").is_some());
    assert!(stage_row(&body, "langevin_drift_half").is_some());
    assert!(stage_row(&body, "langevin_ou_step").is_some());
    for line in body.lines().skip(1) {
        let stage = line.split_whitespace().next().unwrap();
        assert!(
            !stage.starts_with("vv_"),
            "expected no vv_* stage, got {stage}"
        );
    }
}

// rq-0c2265eb
#[test]
fn langevin_stage_counts_match() {
    let dir = tmp_path("langevin_counts");
    let path = write_langevin_config(&dir, 10);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    assert_eq!(stage_count(&body, "langevin_kick_half"), Some(20));
    assert_eq!(stage_count(&body, "langevin_drift_half"), Some(20));
    assert_eq!(stage_count(&body, "langevin_ou_step"), Some(10));
}

// rq-bde625cf
#[test]
fn empty_n_zero_omits_kernel_rows() {
    let dir = tmp_path("empty_n_zero");
    let path = write_pair(&dir, 3, 0, 0, 0.0, false, false, 1, 0);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    assert!(stage_row(&body, "jit_composed_pair_force").is_none());
    assert!(stage_row(&body, "vv_kick_drift").is_none());
    assert!(stage_row(&body, "vv_kick").is_none());
    assert!(stage_row(&body, "gpu_init").is_some());
    assert!(stage_row(&body, "total_runtime").is_some());
}

// rq-3bd5336c
#[test]
fn velocity_generation_absent_when_init_supplies_velocities() {
    let dir = tmp_path("velgen_absent");
    let path = write_pair(&dir, 1, 0, 0, 0.0, true, false, 1, 2);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    assert!(stage_row(&body, "velocity_generation").is_none());
}

// rq-a555a750
#[test]
fn velocity_generation_present_when_init_lacks_velocities() {
    let dir = tmp_path("velgen_present");
    let path = write_pair(&dir, 1, 0, 0, 300.0, false, false, 1, 4);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    let row = stage_row(&body, "velocity_generation").unwrap();
    let count: u64 = row.split_whitespace().nth(1).unwrap().parse().unwrap();
    assert_eq!(count, 1);
}

// rq-a9b511ea rq-62300a18 rq-c7df5714
#[test]
fn kernel_counts_match_runner_launches() {
    let dir = tmp_path("kernel_counts");
    // write_pair emits an all-pairs config with no topology.
    let path = write_pair(&dir, 10, 0, 0, 0.0, true, false, 1, 2);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    assert_eq!(stage_count(&body, "jit_composed_pair_force"), Some(11));
    assert_eq!(stage_count(&body, "vv_kick_drift"), Some(10));
    assert_eq!(stage_count(&body, "vv_kick"), Some(10));
    // rq-62300a18: all-pairs records jit_composed_pair_force and
    // omits every neighbor-list-related row.
    assert!(stage_row(&body, "lj_pair_force_neighbor").is_none());
    assert!(stage_row(&body, "neighbor_displacement_squared").is_none());
    assert!(stage_row(&body, "neighbor_list_build").is_none());
    assert!(stage_row(&body, "copy_positions_into_reference").is_none());
    assert!(stage_row(&body, "neighbor_list_rebuild").is_none());
    // rq-c7df5714: no topology → no jit_composed_bonded_force or
    // reduce_bond_forces; combine_class_totals still runs because at
    // least one pair-force slot is present.
    assert!(stage_row(&body, "jit_composed_bonded_force").is_none());
    assert!(stage_row(&body, "reduce_bond_forces").is_none());
    assert!(stage_row(&body, "combine_class_totals").is_some());
}

// rq-46c317ef
#[test]
fn trajectory_write_count_matches_frames() {
    let dir = tmp_path("traj_count");
    let path = write_pair(&dir, 20, 5, 0, 0.0, true, false, 1, 2);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    assert_eq!(stage_count(&body, "trajectory_write"), Some(5));
    assert!(stage_row(&body, "log_write").is_none());
}

// rq-34bfc634
#[test]
fn log_write_count_matches_rows() {
    let dir = tmp_path("log_count");
    let path = write_pair(&dir, 20, 0, 10, 0.0, true, false, 1, 2);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    assert_eq!(stage_count(&body, "log_write"), Some(3));
    assert!(stage_row(&body, "trajectory_write").is_none());
}

// rq-6fe2b058
#[test]
fn download_count_matches_snapshot_points() {
    let dir = tmp_path("download_count");
    // n_steps=20, trajectory_every=5 hits 5,10,15,20; log_every=10 hits 10,20.
    // Union of steps with a download: {0, 5, 10, 15, 20} = 5 download events
    // (step 0 plus four loop steps because steps 10 and 20 share a single
    // download when traj+log coincide).
    let path = write_pair(&dir, 20, 5, 10, 0.0, true, false, 1, 2);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    assert_eq!(stage_count(&body, "device_to_host_download"), Some(5));
}

// rq-44e5c930
#[test]
fn single_occurrence_host_stages_have_count_one() {
    let dir = tmp_path("single_count");
    let path = write_pair(&dir, 1, 0, 0, 0.0, true, false, 1, 2);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    for stage in &[
        "config_load",
        "init_load",
        "gpu_init",
        "host_to_device_upload",
        "total_runtime",
    ] {
        assert_eq!(stage_count(&body, stage), Some(1), "stage {stage} count");
    }
}

// rq-105afb1d
#[test]
fn min_leq_mean_leq_max() {
    let dir = tmp_path("min_mean_max");
    let path = write_pair(&dir, 20, 0, 0, 0.0, true, false, 1, 2);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    for line in body.lines().skip(1) {
        let cols: Vec<&str> = line.split_whitespace().collect();
        let count: u64 = cols[1].parse().unwrap();
        if count < 2 {
            continue;
        }
        let mean: f64 = cols[3].parse().unwrap();
        let min: f64 = cols[4].parse().unwrap();
        let max: f64 = cols[5].parse().unwrap();
        assert!(min <= mean, "row {line:?}: min {min} > mean {mean}");
        assert!(mean <= max, "row {line:?}: mean {mean} > max {max}");
    }
}

// rq-9818f429
#[test]
fn total_ms_matches_mean_us_times_count() {
    let dir = tmp_path("total_eq_mean_count");
    let path = write_pair(&dir, 20, 5, 5, 0.0, true, false, 1, 2);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    for line in body.lines().skip(1) {
        let cols: Vec<&str> = line.split_whitespace().collect();
        let count: f64 = cols[1].parse::<u64>().unwrap() as f64;
        let total_ms: f64 = cols[2].parse().unwrap();
        let mean_us: f64 = cols[3].parse().unwrap();
        let expected = mean_us * count / 1000.0;
        // Allow precision slack from one-decimal mean and three-decimal total.
        assert!(
            (total_ms - expected).abs() < 0.01 * (1.0 + total_ms.abs()),
            "row {line:?}: total_ms={total_ms} expected≈{expected}"
        );
    }
}

// rq-7e79501b
#[test]
fn total_runtime_dominates_other_rows() {
    let dir = tmp_path("total_runtime_dominates");
    let path = write_pair(&dir, 20, 5, 5, 0.0, true, false, 1, 2);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    let total_runtime = stage_row(&body, "total_runtime").unwrap();
    let total_runtime_ms: f64 = total_runtime
        .split_whitespace()
        .nth(2)
        .unwrap()
        .parse()
        .unwrap();
    // Pre-phase-0 setup stages (`config_load`, `init_load`, `gpu_init`,
    // `velocity_generation`, `host_to_device_upload`) are replayed once
    // into phase 0's Timings but happen *before* the phase's elapsed
    // clock starts, so they legitimately can exceed `total_runtime` on
    // short configs. Skip them here.
    const PRE_PHASE_SETUP_STAGES: &[&str] = &[
        "config_load",
        "init_load",
        "gpu_init",
        "velocity_generation",
        "host_to_device_upload",
    ];
    for line in body.lines().skip(1) {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols[0] == "total_runtime" {
            continue;
        }
        if PRE_PHASE_SETUP_STAGES.contains(&cols[0]) {
            continue;
        }
        let row_total_ms: f64 = cols[2].parse().unwrap();
        assert!(
            total_runtime_ms + 1.0 >= row_total_ms,
            "stage {} reports {} ms > total_runtime {} ms",
            cols[0],
            row_total_ms,
            total_runtime_ms
        );
    }
}

// rq-f1850393
#[test]
fn header_line_uses_documented_columns() {
    let dir = tmp_path("header_cols");
    let path = write_pair(&dir, 1, 0, 0, 0.0, true, false, 1, 2);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    let header = body.lines().next().unwrap();
    let cols: Vec<&str> = header.split_whitespace().collect();
    assert_eq!(
        cols,
        vec!["stage", "count", "total_ms", "mean_us", "min_us", "max_us"]
    );
}

// rq-ef7c2bb9
#[test]
fn rows_appear_in_documented_order() {
    let dir = tmp_path("row_order");
    let path = write_pair(&dir, 10, 5, 5, 0.0, true, false, 1, 2);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    let stages: Vec<&str> = body
        .lines()
        .skip(1)
        .map(|l| l.split_whitespace().next().unwrap())
        .collect();
    let expected = vec![
        "vv_kick_drift",
        "class_accumulator_memset",
        "jit_composed_pair_force",
        "combine_class_totals",
        "vv_kick",
        "host_to_device_upload",
        "device_to_host_download",
        "trajectory_write",
        "log_write",
        "config_load",
        "init_load",
        "gpu_init",
        "total_runtime",
    ];
    assert_eq!(stages, expected);
}

// rq-44dfc3da
#[test]
fn numeric_columns_have_expected_precision() {
    let dir = tmp_path("precision");
    let path = write_pair(&dir, 5, 0, 0, 0.0, true, false, 1, 2);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    for line in body.lines().skip(1) {
        let cols: Vec<&str> = line.split_whitespace().collect();
        let total_ms = cols[2];
        let mean_us = cols[3];
        let min_us = cols[4];
        let max_us = cols[5];
        let decimals = |s: &str| s.split('.').nth(1).unwrap_or("").len();
        assert_eq!(decimals(total_ms), 3, "total_ms {total_ms:?}");
        assert_eq!(decimals(mean_us), 1, "mean_us {mean_us:?}");
        assert_eq!(decimals(min_us), 1, "min_us {min_us:?}");
        assert_eq!(decimals(max_us), 1, "max_us {max_us:?}");
    }
}

// rq-f4780029
#[test]
fn zero_count_stages_absent() {
    let dir = tmp_path("zero_count_absent");
    let path = write_pair(&dir, 5, 0, 0, 0.0, true, false, 1, 2);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    // With trajectory_every=0 and log_every=0, trajectory_write and log_write
    // never run; their rows must be absent.
    assert!(stage_row(&body, "trajectory_write").is_none());
    assert!(stage_row(&body, "log_write").is_none());
}

// rq-ad403fb6
#[test]
fn trajectory_and_log_remain_bit_identical_with_timings_on() {
    let dir_a = tmp_path("repro_a");
    let dir_b = tmp_path("repro_b");
    let path_a = write_pair(&dir_a, 30, 5, 5, 0.0, true, false, 1, 2);
    let path_b = write_pair(&dir_b, 30, 5, 5, 0.0, true, false, 1, 2);
    run_simulation(&path_a).unwrap();
    run_simulation(&path_b).unwrap();
    let canon_a = std::fs::canonicalize(&dir_a).unwrap();
    let canon_b = std::fs::canonicalize(&dir_b).unwrap();
    assert_eq!(
        std::fs::read(canon_a.join("sim.out.run.xyz")).unwrap(),
        std::fs::read(canon_b.join("sim.out.run.xyz")).unwrap()
    );
    assert_eq!(
        std::fs::read(canon_a.join("sim.out.run.log")).unwrap(),
        std::fs::read(canon_b.join("sim.out.run.log")).unwrap()
    );
}

// rq-1b8fd2a0
#[test]
fn two_runs_produce_same_stage_rows_and_counts() {
    let dir_a = tmp_path("stages_a");
    let dir_b = tmp_path("stages_b");
    let path_a = write_pair(&dir_a, 10, 5, 5, 0.0, true, false, 1, 2);
    let path_b = write_pair(&dir_b, 10, 5, 5, 0.0, true, false, 1, 2);
    run_simulation(&path_a).unwrap();
    run_simulation(&path_b).unwrap();
    let body_a = read_timings(&dir_a);
    let body_b = read_timings(&dir_b);
    let stages_a: Vec<(String, u64)> = body_a
        .lines()
        .skip(1)
        .map(|l| {
            let mut it = l.split_whitespace();
            let name = it.next().unwrap().to_string();
            let count: u64 = it.next().unwrap().parse().unwrap();
            (name, count)
        })
        .collect();
    let stages_b: Vec<(String, u64)> = body_b
        .lines()
        .skip(1)
        .map(|l| {
            let mut it = l.split_whitespace();
            let name = it.next().unwrap().to_string();
            let count: u64 = it.next().unwrap().parse().unwrap();
            (name, count)
        })
        .collect();
    assert_eq!(stages_a, stages_b);
}

// rq-f84e4fa1
#[test]
fn timings_path_eq_trajectory_rejected() {
    let dir = tmp_path("collision_traj_timings");
    let body = format!(
        r#"schema_version = 1
init = "sim.in.xyz"

[simulation]
cuda_graphs_disable = true
seed = 1
temperature = 0.0

[[phase]]
name = "run"
n_steps = 1
dt = 1.0e-15


[phase.integrator]
kind = "velocity-verlet"
lossless = false

[[particle_types]]
name = "Ar"
mass = 1.0

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0

[phase.output]
trajectory_path = "out.dat"
timings_path = "out.dat"
"#
    );
    let path = dir.join("sim.in.toml");
    std::fs::write(&path, body).unwrap();
    write_init(&dir, 0, false);
    let err = run_simulation(&path).unwrap_err();
    match err {
        RunnerError::Config(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-ec5b6cf1
#[test]
fn timings_path_eq_log_rejected() {
    let dir = tmp_path("collision_log_timings");
    let body = format!(
        r#"schema_version = 1
init = "sim.in.xyz"

[simulation]
cuda_graphs_disable = true
seed = 1
temperature = 0.0

[[phase]]
name = "run"
n_steps = 1
dt = 1.0e-15


[phase.integrator]
kind = "velocity-verlet"
lossless = false

[[particle_types]]
name = "Ar"
mass = 1.0

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0

[phase.output]
log_path = "out.dat"
timings_path = "out.dat"
"#
    );
    let path = dir.join("sim.in.toml");
    std::fs::write(&path, body).unwrap();
    write_init(&dir, 0, false);
    let err = run_simulation(&path).unwrap_err();
    match err {
        RunnerError::Config(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-2ebf81fd
#[test]
fn timings_path_eq_init_rejected() {
    let dir = tmp_path("collision_init_timings");
    let body = format!(
        r#"schema_version = 1
init = "particles.dat"

[simulation]
cuda_graphs_disable = true
seed = 1
temperature = 0.0

[[phase]]
name = "run"
n_steps = 1
dt = 1.0e-15


[phase.integrator]
kind = "velocity-verlet"
lossless = false

[[particle_types]]
name = "Ar"
mass = 1.0

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 1.0
epsilon = 1.0
cutoff = 1.0

[phase.output]
timings_path = "particles.dat"
"#
    );
    let path = dir.join("sim.in.toml");
    std::fs::write(&path, body).unwrap();
    let err = run_simulation(&path).unwrap_err();
    match err {
        RunnerError::Config(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-de5ca07a
#[test]
fn write_timings_refuses_overwrite() {
    let dir = tmp_path("write_refuse_overwrite");
    let path = dir.join("run.timings");
    std::fs::write(&path, "existing").unwrap();
    let report = TimingsReport {
        stages: vec![StageStats {
            name: "stage_a".to_string(),
            count: 1,
            total_ns: 1_000_000,
            min_ns: 1_000_000,
            max_ns: 1_000_000,
        }],
    };
    match write_timings_file(&path, &report).unwrap_err() {
        TimingsWriterError::OutputExists { path: p } => assert_eq!(p, path),
        other => panic!("unexpected: {other:?}"),
    }
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "existing");
}

// rq-e93e7ad1
#[test]
fn write_timings_fails_when_parent_missing() {
    let dir = tmp_path("write_no_parent");
    let path = dir.join("missing").join("run.timings");
    let report = TimingsReport { stages: Vec::new() };
    match write_timings_file(&path, &report).unwrap_err() {
        TimingsWriterError::Io(_) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-5c716d48
#[test]
fn write_timings_empty_report_writes_header_only() {
    let dir = tmp_path("write_empty");
    let path = dir.join("run.timings");
    let report = TimingsReport { stages: Vec::new() };
    write_timings_file(&path, &report).unwrap();
    let body = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = body.lines().collect();
    assert_eq!(lines.len(), 1);
    assert!(lines[0].contains("stage") && lines[0].contains("count"));
}

// rq-e946870f
#[test]
fn timings_new_allocates_event_pairs() {
    let gpu = init_device().unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let report = timings.finalize().unwrap();
    assert!(report.stages.is_empty());
}

// rq-79291197
#[test]
fn kernel_start_stop_and_finalize_records_one_sample() {
    let gpu = init_device().unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let state = ParticleState::new(
        vec![0.0, 1.0e-10],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![1.0; 2],
        vec![0.0; 2],
        vec![0u32; 2],
        None,
            None,
    )
    .unwrap();
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut slot_out = heddle_md::gpu::SlotOutputBuffers::new(&gpu.device, 2).unwrap();
    let params = single_type_lj_table(&gpu.device, 1.0e-10, 1.0, 1.0e-9);
    let sim_box = SimulationBox::new(&gpu.device, 1.0e-9, 1.0e-9, 1.0e-9, 0.0, 0.0, 0.0).unwrap();
    timings.kernel_start(KernelStage::LJ_PAIR_FORCE).unwrap();
    lj_pair_force_no_excl(
        &buffers,
        &mut slot_out,
        &sim_box,
        &params,
        heddle_md::forces::AggregateLevel::ForcesOnly,
    ).unwrap();
    timings.kernel_stop(KernelStage::LJ_PAIR_FORCE).unwrap();
    let report = timings.finalize().unwrap();
    let entry = report
        .stages
        .iter()
        .find(|s| s.name == "lj_pair_force")
        .unwrap();
    assert_eq!(entry.count, 1);
}

// rq-56043142
#[test]
fn repeated_kernel_starts_stops_accumulate() {
    let gpu = init_device().unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let state = ParticleState::new(
        vec![0.0, 1.0e-10],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![1.0; 2],
        vec![0.0; 2],
        vec![0u32; 2],
        None,
            None,
    )
    .unwrap();
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut slot_out = heddle_md::gpu::SlotOutputBuffers::new(&gpu.device, 2).unwrap();
    let params = single_type_lj_table(&gpu.device, 1.0e-10, 1.0, 1.0e-9);
    let sim_box = SimulationBox::new(&gpu.device, 1.0e-9, 1.0e-9, 1.0e-9, 0.0, 0.0, 0.0).unwrap();
    for _ in 0..10 {
        timings.kernel_start(KernelStage::LJ_PAIR_FORCE).unwrap();
        lj_pair_force_no_excl(
            &buffers,
            &mut slot_out,
            &sim_box,
            &params,
            heddle_md::forces::AggregateLevel::ForcesOnly,
        ).unwrap();
        timings.kernel_stop(KernelStage::LJ_PAIR_FORCE).unwrap();
    }
    let report = timings.finalize().unwrap();
    let entry = report
        .stages
        .iter()
        .find(|s| s.name == "lj_pair_force")
        .unwrap();
    assert_eq!(entry.count, 10);
    assert!(entry.total_ns > 0);
}

// rq-2cbe0828
#[test]
fn record_host_accumulates_count_total_min_max() {
    let gpu = init_device().unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    timings.record_host(HostStage::CONFIG_LOAD, Duration::from_micros(100));
    timings.record_host(HostStage::CONFIG_LOAD, Duration::from_micros(50));
    timings.record_host(HostStage::CONFIG_LOAD, Duration::from_micros(200));
    let report = timings.finalize().unwrap();
    let entry = report
        .stages
        .iter()
        .find(|s| s.name == "config_load")
        .unwrap();
    assert_eq!(entry.count, 3);
    assert_eq!(entry.total_ns, 350_000);
    assert_eq!(entry.min_ns, 50_000);
    assert_eq!(entry.max_ns, 200_000);
}

#[test] // rq-f232d41b
#[should_panic(expected = "unknown KernelStage")]
fn kernel_start_with_unknown_kernel_stage_panics() {
    let gpu = init_device().unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let unknown = KernelStage::new("not_a_stage");
    let _ = timings.kernel_start(unknown);
}

#[test] // rq-264d2234
#[should_panic(expected = "unknown HostStage")]
fn record_host_with_unknown_host_stage_panics() {
    let gpu = init_device().unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let unknown = HostStage::new("not_a_host_stage");
    timings.record_host(unknown, Duration::from_micros(10));
}

// =====================================================================
// Cell-list and morse-bonded timings rows.
// =====================================================================

fn write_cell_list_pair(
    dir: &Path,
    n_steps: u64,
    r_skin: f64,
    seed: u64,
    n_particles: usize,
    include_velocities: bool,
) -> PathBuf {
    let config = format!(
        r#"schema_version = 1
init = "sim.in.xyz"

[simulation]
cuda_graphs_disable = true
seed = {seed}
temperature = 0.0

[[phase]]
name = "run"
n_steps = {n_steps}
dt = 1.0e-15

[phase.integrator]
kind = "velocity-verlet"
lossless = false

[phase.output]
trajectory_every = 0
log_every = 0

[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 1.0e-9

[neighbor_list]
mode = "cell-list"
r_skin = {r_skin}
"#
    );
    let config_path = dir.join("sim.in.toml");
    std::fs::write(&config_path, config).unwrap();
    // Box has to satisfy the box-too-small check: 3 * (cutoff + r_skin)
    // must fit in every perpendicular width. With cutoff = 1e-9 and the
    // r_skin values we pick, a 6e-9 cube is large enough.
    let mut body = String::new();
    body.push_str(&format!("{n_particles}\n"));
    let props = if include_velocities {
        "species:S:1:pos:R:3:velo:R:3"
    } else {
        "species:S:1:pos:R:3"
    };
    body.push_str(&format!(
        "Lattice=\"6.0e-9 0 0 0 6.0e-9 0 0 0 6.0e-9\" Properties={props}\n"
    ));
    let spacing = if n_particles > 1 {
        4.0e-9 / (n_particles as f64 - 1.0)
    } else {
        0.0
    };
    let half_n = (n_particles as f64 - 1.0) / 2.0;
    for i in 0..n_particles {
        let x = (i as f64 - half_n) * spacing;
        if include_velocities {
            body.push_str(&format!("Ar {x:.9e} 0.0 0.0 0.0 0.0 0.0\n"));
        } else {
            body.push_str(&format!("Ar {x:.9e} 0.0 0.0\n"));
        }
    }
    std::fs::write(dir.join("sim.in.xyz"), body).unwrap();
    config_path
}

// rq-ef918dc6 rq-75746f64
#[test]
fn cell_list_records_neighbor_stages() {
    let dir = tmp_path("cell_list_stages");
    let path = write_cell_list_pair(&dir, 10, 3.0e-10, 1, 4, true);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    // rq-ef918dc6: cell-list adds the four neighbor-list host stages on
    // top of the regular jit_composed_pair_force kernel stage.
    assert!(stage_row(&body, "jit_composed_pair_force").is_some());
    assert!(stage_row(&body, "neighbor_displacement_squared").is_some());
    assert!(stage_row(&body, "neighbor_list_build").is_some());
    assert!(stage_row(&body, "copy_positions_into_reference").is_some());
    assert!(stage_row(&body, "neighbor_list_rebuild").is_some());
    // rq-75746f64: with n_steps=10, neighbor_displacement_squared runs
    // once per loop step (no warm-up displacement check before step 1).
    assert_eq!(stage_count(&body, "neighbor_displacement_squared"), Some(10));
}

// rq-7f2310ac
#[test]
fn neighbor_list_build_and_reference_copy_counts_match_rebuilds() {
    // Whatever r_skin we choose, the runner emits one host-stage event
    // for `neighbor_list_build`, `copy_positions_into_reference`, and
    // `neighbor_list_rebuild` per rebuild (including the warm-up
    // build). The architectural contract is that the three counts are
    // equal — that is the invariant this test exercises. We do not
    // try to engineer "exactly K rebuilds" via a tuned r_skin; doing so
    // would couple the test to particle dynamics that aren't part of the
    // contract.
    let dir = tmp_path("nl_rebuild_counts");
    let path = write_cell_list_pair(&dir, 10, 3.0e-10, 1, 4, true);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    let build = stage_count(&body, "neighbor_list_build").unwrap();
    let copy = stage_count(&body, "copy_positions_into_reference").unwrap();
    let rebuild = stage_count(&body, "neighbor_list_rebuild").unwrap();
    assert!(build >= 1, "expected at least the warm-up build, got {build}");
    assert_eq!(build, copy, "neighbor_list_build vs copy_positions_into_reference");
    assert_eq!(build, rebuild, "neighbor_list_build vs neighbor_list_rebuild");
}

fn write_morse_bonded(
    dir: &Path,
    n_steps: u64,
    seed: u64,
) -> PathBuf {
    let config = format!(
        r#"schema_version = 1
init = "sim.in.xyz"
topology = "sim.in.topology"

[simulation]
cuda_graphs_disable = true
seed = {seed}
temperature = 0.0

[[phase]]
name = "run"
n_steps = {n_steps}
dt = 1.0e-15

[phase.integrator]
kind = "velocity-verlet"
lossless = false

[phase.output]
trajectory_every = 0
log_every = 0

[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 1.0e-9

[[bond_types]]
name = "ArAr"
potential = "morse"
de = 1.65e-21
a = 1.9e10
re = 3.4e-10

[neighbor_list]
mode = "all-pairs"
"#
    );
    let config_path = dir.join("sim.in.toml");
    std::fs::write(&config_path, config).unwrap();
    // Two particles, one bond between them.
    let xyz = "2\nLattice=\"4.0e-9 0 0 0 4.0e-9 0 0 0 4.0e-9\" \
               Properties=species:S:1:pos:R:3:velo:R:3\n\
               Ar -1.5e-10 0.0 0.0 0.0 0.0 0.0\n\
               Ar  1.5e-10 0.0 0.0 0.0 0.0 0.0\n";
    std::fs::write(dir.join("sim.in.xyz"), xyz).unwrap();
    let topo = "[bonds]\n0 1 ArAr\n";
    std::fs::write(dir.join("sim.in.topology"), topo).unwrap();
    config_path
}

// rq-14b8e042
#[test]
fn morse_bonded_records_bond_force_and_reduction_rows() {
    let dir = tmp_path("morse_bonded_rows");
    let path = write_morse_bonded(&dir, 10, 1);
    run_simulation(&path).unwrap();
    let body = read_timings(&dir);
    // One warm-up + ten loop iterations = 11.
    assert_eq!(stage_count(&body, "jit_composed_bonded_force"), Some(11));
    assert_eq!(stage_count(&body, "reduce_bond_forces"), Some(11));
    assert_eq!(stage_count(&body, "combine_class_totals"), Some(11));
}

// Silence unused-import warning when individual tests don't reference these.
#[test]
fn _imports_used() {
    let _ = TrajectoryWriterError::Io(String::new());
    let _ = load_init_state;
}
