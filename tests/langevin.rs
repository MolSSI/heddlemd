use std::path::{Path, PathBuf};

use dynamics::gpu::{
    LennardJonesParameters, PairBuffer, ParticleBuffers, init_device, lan_drift_half,
    lan_ou_step, lj_pair_force, reduce_pair_forces,
};
use dynamics::integrator::Integrator;
use dynamics::io::IntegratorKind;
use dynamics::io::log_output::BOLTZMANN_J_PER_K;
use dynamics::pbc::SimulationBox;
use dynamics::runner::run_simulation;
use dynamics::state::ParticleState;
use dynamics::timings::Timings;

fn tmp_path(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!(
        "dynamics-langevin-{}-{}-{}",
        std::process::id(),
        name,
        nanos
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn one_particle_state() -> ParticleState {
    ParticleState::new(
        vec![1.0_f32],
        vec![2.0_f32],
        vec![3.0_f32],
        vec![0.5_f32],
        vec![-0.25_f32],
        vec![0.125_f32],
        vec![1.0_f32],
        None,
    )
    .unwrap()
}

fn n_particle_state(n: usize) -> ParticleState {
    let pos: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let velos: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();
    ParticleState::new(
        pos.clone(),
        pos.clone(),
        pos,
        velos.clone(),
        velos.clone(),
        velos,
        vec![1.0_f32; n],
        None,
    )
    .unwrap()
}

// rq-662fccc1
#[test]
fn init_device_loads_langevin_module() {
    let device = init_device().unwrap();
    assert!(device.get_func("langevin", "lan_drift_half").is_some());
    assert!(device.get_func("langevin", "lan_ou_step").is_some());
}

// rq-457b5271
#[test]
fn construct_langevin_state_stores_parameters() {
    let device = init_device().unwrap();
    let kind = IntegratorKind::LangevinBaoab {
        friction: 1.0e12,
        temperature: 300.0,
        seed: 42,
    };
    let integrator = Integrator::new(device, 4, &kind).unwrap();
    match integrator {
        Integrator::LangevinBaoab(state) => {
            assert_eq!(state.friction, 1.0e12);
            assert_eq!(state.temperature, 300.0);
            assert_eq!(state.seed, 42);
        }
        _ => panic!("expected LangevinBaoab variant"),
    }
}

// rq-e9994f86
#[test]
fn construct_langevin_with_zero_particles() {
    let device = init_device().unwrap();
    let kind = IntegratorKind::LangevinBaoab {
        friction: 1.0e12,
        temperature: 300.0,
        seed: 42,
    };
    let _ = Integrator::new(device, 0, &kind).unwrap();
}

// rq-358de3e6
#[test]
fn lan_drift_half_advances_positions_by_v_half_dt() {
    let device = init_device().unwrap();
    let state = one_particle_state();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    lan_drift_half(&mut buffers, 0.1).unwrap();
    let mut downloaded = state.clone();
    downloaded.download_from(&buffers).unwrap();
    assert!((downloaded.positions_x[0] - (1.0 + 0.5 * 0.05)).abs() < 1.0e-6);
    assert!((downloaded.positions_y[0] - (2.0 + -0.25 * 0.05)).abs() < 1.0e-6);
    assert!((downloaded.positions_z[0] - (3.0 + 0.125 * 0.05)).abs() < 1.0e-6);
    assert!((downloaded.velocities_x[0] - 0.5).abs() < 1.0e-9);
}

// rq-5e8125ac
#[test]
fn lan_drift_half_leaves_other_arrays_unchanged() {
    let device = init_device().unwrap();
    let state = n_particle_state(4);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    lan_drift_half(&mut buffers, 0.1).unwrap();
    let mut downloaded = state.clone();
    downloaded.download_from(&buffers).unwrap();
    assert_eq!(downloaded.velocities_x, state.velocities_x);
    assert_eq!(downloaded.velocities_y, state.velocities_y);
    assert_eq!(downloaded.velocities_z, state.velocities_z);
    assert_eq!(downloaded.forces_x, state.forces_x);
    assert_eq!(downloaded.masses, state.masses);
    assert_eq!(downloaded.particle_ids, state.particle_ids);
}

// rq-2680918c
#[test]
fn lan_drift_half_on_empty_is_noop() {
    let device = init_device().unwrap();
    let state = ParticleState::new(
        Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device, &state).unwrap();
    lan_drift_half(&mut buffers, 0.1).unwrap();
}

// rq-247e3799
#[test]
fn lan_drift_half_with_dt_zero_leaves_positions_unchanged() {
    let device = init_device().unwrap();
    let state = n_particle_state(4);
    let mut buffers = ParticleBuffers::new(device, &state).unwrap();
    lan_drift_half(&mut buffers, 0.0).unwrap();
    let mut downloaded = state.clone();
    downloaded.download_from(&buffers).unwrap();
    assert_eq!(downloaded.positions_x, state.positions_x);
    assert_eq!(downloaded.positions_y, state.positions_y);
    assert_eq!(downloaded.positions_z, state.positions_z);
}

// rq-41389685
#[test]
fn lan_ou_step_with_alpha_one_kt_zero_is_identity() {
    let device = init_device().unwrap();
    let state = n_particle_state(4);
    let mut buffers = ParticleBuffers::new(device, &state).unwrap();
    lan_ou_step(&mut buffers, 1, 1, 1.0, 0.0).unwrap();
    let mut downloaded = state.clone();
    downloaded.download_from(&buffers).unwrap();
    assert_eq!(downloaded.velocities_x, state.velocities_x);
    assert_eq!(downloaded.velocities_y, state.velocities_y);
    assert_eq!(downloaded.velocities_z, state.velocities_z);
}

// rq-01813ffa
#[test]
fn lan_ou_step_on_empty_is_noop() {
    let device = init_device().unwrap();
    let state = ParticleState::new(
        Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device, &state).unwrap();
    lan_ou_step(&mut buffers, 1, 1, 0.5, 1.0).unwrap();
}

// rq-9922b639
#[test]
fn two_identical_lan_ou_calls_byte_identical() {
    let device = init_device().unwrap();
    let state = n_particle_state(64);
    let mut buffers_a = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut buffers_b = ParticleBuffers::new(device.clone(), &state).unwrap();
    lan_ou_step(&mut buffers_a, 42, 7, 0.5, 1.0e-21).unwrap();
    lan_ou_step(&mut buffers_b, 42, 7, 0.5, 1.0e-21).unwrap();
    let mut state_a = state.clone();
    let mut state_b = state.clone();
    state_a.download_from(&buffers_a).unwrap();
    state_b.download_from(&buffers_b).unwrap();
    for i in 0..64 {
        assert_eq!(state_a.velocities_x[i].to_bits(), state_b.velocities_x[i].to_bits());
        assert_eq!(state_a.velocities_y[i].to_bits(), state_b.velocities_y[i].to_bits());
        assert_eq!(state_a.velocities_z[i].to_bits(), state_b.velocities_z[i].to_bits());
    }
}

// rq-10652c60
#[test]
fn different_seeds_produce_different_velocities() {
    // Use kt = 1.0 in reduced units so the OU perturbation is much larger than
    // an f32 ULP at the input velocity magnitudes; the per-(particle, axis)
    // Philox draws then surface as visible bit differences.
    let device = init_device().unwrap();
    let state = n_particle_state(64);
    let mut buffers_a = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut buffers_b = ParticleBuffers::new(device.clone(), &state).unwrap();
    lan_ou_step(&mut buffers_a, 1, 1, 0.5, 1.0).unwrap();
    lan_ou_step(&mut buffers_b, 2, 1, 0.5, 1.0).unwrap();
    let mut state_a = state.clone();
    let mut state_b = state.clone();
    state_a.download_from(&buffers_a).unwrap();
    state_b.download_from(&buffers_b).unwrap();
    let mut differing = 0;
    for i in 0..64 {
        if state_a.velocities_x[i] != state_b.velocities_x[i] {
            differing += 1;
        }
    }
    assert!(differing as f64 / 64.0 >= 0.9, "only {differing} of 64 differ");
}

// rq-e2f2de4f
#[test]
fn different_step_indices_produce_different_velocities() {
    let device = init_device().unwrap();
    let state = n_particle_state(64);
    let mut buffers_a = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut buffers_b = ParticleBuffers::new(device.clone(), &state).unwrap();
    lan_ou_step(&mut buffers_a, 1, 1, 0.5, 1.0).unwrap();
    lan_ou_step(&mut buffers_b, 1, 2, 0.5, 1.0).unwrap();
    let mut state_a = state.clone();
    let mut state_b = state.clone();
    state_a.download_from(&buffers_a).unwrap();
    state_b.download_from(&buffers_b).unwrap();
    let mut differing = 0;
    for i in 0..64 {
        if state_a.velocities_x[i] != state_b.velocities_x[i] {
            differing += 1;
        }
    }
    assert!(differing as f64 / 64.0 >= 0.9);
}

// rq-50baca8c
#[test]
fn ou_variance_scales_with_predicted_factor() {
    let device = init_device().unwrap();
    let n = 10_000;
    let mass = 6.6335e-26_f32;
    let temperature = 300.0_f64;
    let kt = (BOLTZMANN_J_PER_K * temperature) as f32;
    let alpha = 0.5_f32;
    let state = ParticleState::new(
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![mass; n],
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device, &state).unwrap();
    lan_ou_step(&mut buffers, 42, 1, alpha, kt).unwrap();
    let mut downloaded = state.clone();
    downloaded.download_from(&buffers).unwrap();
    let expected_var = (1.0 - alpha as f64 * alpha as f64) * kt as f64 / mass as f64;
    for v in [
        &downloaded.velocities_x,
        &downloaded.velocities_y,
        &downloaded.velocities_z,
    ] {
        let mean: f64 = v.iter().map(|x| *x as f64).sum::<f64>() / n as f64;
        let var: f64 = v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        let rel_err = (var - expected_var).abs() / expected_var;
        assert!(rel_err < 0.05, "var={var}, expected={expected_var}, rel_err={rel_err}");
    }
}

// rq-e1dd0625
#[test]
fn slot_interface_launches_six_kernel_calls() {
    let device = init_device().unwrap();
    let state = n_particle_state(4);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut pair_buffer = PairBuffer::new(device.clone(), 4, 4).unwrap();
    let neighbor_counts = device.htod_sync_copy(&vec![4u32; 4]).unwrap();
    let params = LennardJonesParameters {
        sigma: 1.0,
        epsilon: 1.0,
        cutoff: 1.0e9,
    };
    let sim_box = SimulationBox::new_orthorhombic(1.0e9, 1.0e9, 1.0e9).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    let mut integrator = Integrator::new(
        device,
        4,
        &IntegratorKind::LangevinBaoab {
            friction: 1.0e12,
            temperature: 300.0,
            seed: 1,
        },
    )
    .unwrap();

    // Warm-up force evaluation
    lj_pair_force(&buffers, &mut pair_buffer, &sim_box, &params).unwrap();
    reduce_pair_forces(&pair_buffer, &neighbor_counts, &mut buffers).unwrap();

    integrator.pre_force_step(&mut buffers, 1.0e-15, 1, &mut timings).unwrap();
    lj_pair_force(&buffers, &mut pair_buffer, &sim_box, &params).unwrap();
    reduce_pair_forces(&pair_buffer, &neighbor_counts, &mut buffers).unwrap();
    integrator.post_force_step(&mut buffers, 1.0e-15, 1, &mut timings).unwrap();

    let report = timings.finalize().unwrap();
    let count = |name: &str| -> u64 {
        report
            .stages
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.count)
            .unwrap_or(0)
    };
    assert_eq!(count("langevin_kick_half"), 2);
    assert_eq!(count("langevin_drift_half"), 2);
    assert_eq!(count("langevin_ou_step"), 1);
}

// rq-6e98222c
#[test]
fn langevin_pre_force_step_on_empty_is_noop() {
    let device = init_device().unwrap();
    let state = ParticleState::new(
        Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    let mut integrator = Integrator::new(
        device,
        0,
        &IntegratorKind::LangevinBaoab {
            friction: 1.0e12,
            temperature: 300.0,
            seed: 1,
        },
    )
    .unwrap();
    integrator
        .pre_force_step(&mut buffers, 1.0e-15, 1, &mut timings)
        .unwrap();
}

fn write_langevin_pair(dir: &Path, friction: f64, temperature: f64, seed: u64) -> PathBuf {
    let cfg = format!(
        r#"schema_version = 1
init = "init.xyz"

[simulation]
seed = 1
n_steps = 5
dt = 1.0e-15
temperature = 300.0

[integrator]
kind = "langevin-baoab"
friction = {friction}
temperature = {temperature}
seed = {seed}

[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 1.0e-9

[output]
trajectory_every = 1
log_every = 1
"#
    );
    let path = dir.join("sim.toml");
    std::fs::write(&path, cfg).unwrap();
    let init_body = format!(
        "4\nLattice=\"4.0e-9 0 0 0 4.0e-9 0 0 0 4.0e-9\" Properties=species:S:1:pos:R:3:velo:R:3\n\
         Ar -1.5e-9 0 0 0 0 0\nAr -5.0e-10 0 0 0 0 0\nAr 5.0e-10 0 0 0 0 0\nAr 1.5e-9 0 0 0 0 0\n"
    );
    std::fs::write(dir.join("init.xyz"), init_body).unwrap();
    path
}

// rq-874dbfec
#[test]
fn two_end_to_end_langevin_runs_byte_identical() {
    let dir_a = tmp_path("langevin_repro_a");
    let dir_b = tmp_path("langevin_repro_b");
    let path_a = write_langevin_pair(&dir_a, 1.0e12, 300.0, 42);
    let path_b = write_langevin_pair(&dir_b, 1.0e12, 300.0, 42);
    run_simulation(&path_a).unwrap();
    run_simulation(&path_b).unwrap();
    let traj_a = std::fs::read(std::fs::canonicalize(&dir_a).unwrap().join("sim-traj.xyz")).unwrap();
    let traj_b = std::fs::read(std::fs::canonicalize(&dir_b).unwrap().join("sim-traj.xyz")).unwrap();
    let log_a = std::fs::read(std::fs::canonicalize(&dir_a).unwrap().join("sim.log")).unwrap();
    let log_b = std::fs::read(std::fs::canonicalize(&dir_b).unwrap().join("sim.log")).unwrap();
    assert_eq!(traj_a, traj_b);
    assert_eq!(log_a, log_b);
}

// rq-2a3f0e9b
#[test]
fn langevin_different_seeds_produce_different_trajectories() {
    let dir_a = tmp_path("langevin_seed_a");
    let dir_b = tmp_path("langevin_seed_b");
    let path_a = write_langevin_pair(&dir_a, 1.0e12, 300.0, 1);
    let path_b = write_langevin_pair(&dir_b, 1.0e12, 300.0, 2);
    run_simulation(&path_a).unwrap();
    run_simulation(&path_b).unwrap();
    let traj_a = std::fs::read(std::fs::canonicalize(&dir_a).unwrap().join("sim-traj.xyz")).unwrap();
    let traj_b = std::fs::read(std::fs::canonicalize(&dir_b).unwrap().join("sim-traj.xyz")).unwrap();
    assert_ne!(traj_a, traj_b);
}

// rq-adc3a32f
// Equilibrium kinetic energy approaches 1.5 * N * k_B * T after many steps.
// Run a small Langevin thermostat with strong friction and check the final log
// row's temperature column lies near the target temperature.
#[test]
fn langevin_equilibrium_kinetic_energy() {
    let dir = tmp_path("langevin_equilibrium");
    let n = 512;
    // Generate a simple 8x8x8 lattice (512 atoms) in a 6.4 nm box.
    let mut body = String::new();
    body.push_str(&format!("{n}\n"));
    body.push_str(
        "Lattice=\"6.4e-9 0 0 0 6.4e-9 0 0 0 6.4e-9\" Properties=species:S:1:pos:R:3:velo:R:3\n",
    );
    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                let x = (i as f64 - 3.5) * 7.0e-10;
                let y = (j as f64 - 3.5) * 7.0e-10;
                let z = (k as f64 - 3.5) * 7.0e-10;
                body.push_str(&format!("Ar {x:.9e} {y:.9e} {z:.9e} 0 0 0\n"));
            }
        }
    }
    std::fs::write(dir.join("init.xyz"), body).unwrap();
    let cfg = r#"schema_version = 1
init = "init.xyz"

[simulation]
seed = 1
n_steps = 1000
dt = 1.0e-15
temperature = 300.0

[integrator]
kind = "langevin-baoab"
friction = 1.0e13
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
cutoff = 8.5e-10

[output]
trajectory_every = 0
log_every = 100
"#;
    let path = dir.join("sim.toml");
    std::fs::write(&path, cfg).unwrap();
    run_simulation(&path).unwrap();
    let log_path = std::fs::canonicalize(&dir).unwrap().join("sim.log");
    let log = std::fs::read_to_string(&log_path).unwrap();
    let last = log.lines().last().unwrap();
    let cols: Vec<&str> = last.split(',').collect();
    let final_t: f64 = cols[3].parse().unwrap();
    assert!(
        (final_t - 300.0).abs() / 300.0 < 0.15,
        "final temperature was {final_t}, expected ~300 K"
    );
}

// rq-1c729f15
#[test]
fn langevin_runner_with_n_zero() {
    let dir = tmp_path("langevin_empty");
    let cfg = r#"schema_version = 1
init = "init.xyz"

[simulation]
seed = 1
n_steps = 3
dt = 1.0e-15
temperature = 300.0

[integrator]
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

[output]
trajectory_every = 1
log_every = 1
"#;
    let path = dir.join("sim.toml");
    std::fs::write(&path, cfg).unwrap();
    std::fs::write(
        dir.join("init.xyz"),
        "0\nLattice=\"1.0e-9 0 0 0 1.0e-9 0 0 0 1.0e-9\" Properties=species:S:1:pos:R:3\n",
    )
    .unwrap();
    let summary = run_simulation(&path).unwrap();
    assert_eq!(summary.n_steps, 3);
    let timings_path = std::fs::canonicalize(&dir).unwrap().join("sim.timings");
    let body = std::fs::read_to_string(&timings_path).unwrap();
    for line in body.lines().skip(1) {
        let stage = line.split_whitespace().next().unwrap();
        assert!(
            !stage.starts_with("langevin_"),
            "expected no langevin_* stage in N=0 run, got {stage}"
        );
    }
}
