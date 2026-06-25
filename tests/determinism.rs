//! End-to-end run-to-run determinism guard for the fast-class pair force.
//!
//! Regression test for the class of bug where the JIT-composed pair
//! kernel accumulates an i-atom's contributions across its packed-tile
//! entries in a non-deterministic (atomic-built) order using a
//! *floating-point* register — making the per-atom force depend on GPU
//! scheduling and so differ run-to-run (commit `b17aee1`, since fixed by
//! the i64 fixed-point i-side accumulator; see
//! `rqm/forces/jit-composed-pair-force.md`, rq-693544f8 / rq-c156295f).
//!
//! Two properties are load-bearing for *reliably* catching this:
//!   1. The configuration is **disordered** (liquid-like). A perfect
//!      crystal has too-regular neighbour packing and stays bit-identical
//!      even on the buggy kernel — it does NOT exercise the bug.
//!   2. The system spans **many atom-blocks** with **multi-entry dense
//!      tiles**, and the run is repeated **several** times. The
//!      non-determinism is a scheduling race; a single pair of runs (or a
//!      tiny system) can coincidentally agree.

use std::path::{Path, PathBuf};

use heddle_md::runner::run_simulation;

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("heddle_determinism_{name}"));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Write a disordered LJ-liquid config: a `side³` lattice perturbed by a
/// deterministic LCG (so the *input* is byte-identical every call, but
/// the neighbour structure is irregular). `spacing` near the LJ minimum
/// keeps the dynamics finite over a few steps.
fn write_disordered(dir: &Path, side: usize, spacing: f64) {
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
    let cfg = r#"schema_version = 1
init = "sim.in.xyz"

[simulation]
cuda_graphs_disable = true
seed = 1
temperature = 300.0

[[phase]]
name = "run"
n_steps = 3
dt = 1.0e-15

[phase.integrator]
kind = "velocity-verlet"
lossless = false

[phase.output]
trajectory_every = 1
log_every = 1

[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 9.0e-10
"#;
    std::fs::write(dir.join("sim.in.toml"), cfg).unwrap();
}

// rq-c156295f
#[test]
fn repeated_runs_of_a_disordered_system_are_byte_identical() {
    let n_runs = 5;
    let side = 12; // 1728 atoms across many atom-blocks
    let spacing = 4.0e-10;
    let mut trajectories = Vec::new();
    for r in 0..n_runs {
        let dir = tmp(&format!("run{r}"));
        write_disordered(&dir, side, spacing);
        run_simulation(&dir.join("sim.in.toml")).unwrap();
        // Compare the full trajectory (positions + velocities every step):
        // any per-atom force divergence shows up by step 1.
        trajectories.push(std::fs::read(dir.join("sim.out.run.xyz")).unwrap());
    }
    for r in 1..n_runs {
        assert!(
            trajectories[0] == trajectories[r],
            "run {r} trajectory differs from run 0 — the pair force is not \
             run-to-run deterministic on this GPU"
        );
    }
}
