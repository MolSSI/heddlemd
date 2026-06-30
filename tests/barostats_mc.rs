// rq-09ac44ea — Monte-Carlo barostat tests. See
// `rqm/integration/mc-barostat.md`.

use heddle_md::forces::{
    AngleList, BondList, ExclusionList, ForceField, MoleculeList, PotentialRegistry,
};
use heddle_md::gpu::{GpuContext, ParticleBuffers, init_device, mc_barostat_scale_molecule_com};
use heddle_md::integrator::{
    Barostat, BarostatPeriodicity, BarostatRegistry, McBarostat,
};
use heddle_md::io::SlotConfig;
use heddle_md::io::config::NeighborListConfig;
use heddle_md::pbc::SimulationBox;
use heddle_md::precision::Real;
use heddle_md::registry::KindedBuilder;
use heddle_md::state::ParticleState;
use heddle_md::timings::Timings;

// ---------- harness ----------

fn mc_params(extra: &str) -> SlotConfig {
    SlotConfig::from_params_str("monte-carlo", extra)
}

fn validate(slot: &SlotConfig) -> Result<(), heddle_md::io::config::ConfigError> {
    BarostatRegistry::with_builtins()
        .lookup("monte-carlo")
        .unwrap()
        .validate_params(&slot.params)
}

fn build_mc(gpu: &GpuContext, n: usize, slot: &SlotConfig) -> Box<dyn Barostat> {
    BarostatRegistry::with_builtins()
        .build_optional(Some(slot), gpu, n, 0)
        .unwrap()
        .unwrap()
}

fn unbox_mc(boxed: Box<dyn Barostat>) -> McBarostat {
    unsafe { *Box::from_raw(Box::into_raw(boxed) as *mut McBarostat) }
}

fn cube(gpu: &GpuContext, l: Real) -> SimulationBox {
    SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap()
}

fn empty_force_field(gpu: &GpuContext, n: usize, sim_box: &SimulationBox) -> ForceField {
    ForceField::new(
        &PotentialRegistry::with_builtins(),
        gpu,
        n,
        sim_box,
        &[],
        &[],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(n),
        &AngleList::empty(0),
        &ExclusionList::empty(n),
        &NeighborListConfig::AllPairs,
    )
    .unwrap()
}

fn make_buffers(
    gpu: &GpuContext,
    px: Vec<Real>,
    py: Vec<Real>,
    pz: Vec<Real>,
    masses: Vec<Real>,
) -> ParticleBuffers {
    let n = px.len();
    let zero = vec![0.0 as Real; n];
    let state = ParticleState::new(
        px,
        py,
        pz,
        zero.clone(),
        zero.clone(),
        zero.clone(),
        masses,
        vec![0.0; n],
        (0..n as u32).collect(),
        None,
        None,
    )
    .unwrap();
    ParticleBuffers::new(gpu, &state).unwrap()
}

// ---------- config / registry ----------

#[test] // rq-f609ea67
fn registry_exposes_monte_carlo() {
    let registry = BarostatRegistry::with_builtins();
    let names: Vec<&str> = registry.builders().iter().map(|b| b.kind_name()).collect();
    assert!(names.contains(&"berendsen"));
    assert!(names.contains(&"c-rescale"));
    assert!(names.contains(&"monte-carlo"));
}

#[test] // rq-26d969aa
fn accept_negative_pressure() {
    let slot = mc_params("pressure = -3.4e-9\ntemperature = 9.4e-4\nseed = 1\n");
    assert!(validate(&slot).is_ok());
}

#[test] // rq-cb831438
fn reject_non_positive_temperature() {
    let slot = mc_params("pressure = 3.4e-9\ntemperature = 0.0\nseed = 1\n");
    assert!(validate(&slot).is_err());
}

#[test] // rq-69ba800a
fn reject_zero_frequency() {
    let slot = mc_params("pressure = 3.4e-9\ntemperature = 9.4e-4\nfrequency = 0\nseed = 1\n");
    assert!(validate(&slot).is_err());
}

#[test] // rq-e8f97894
fn reject_non_positive_volume_step() {
    let slot = mc_params("pressure = 3.4e-9\ntemperature = 9.4e-4\nvolume_step = 0.0\nseed = 1\n");
    assert!(validate(&slot).is_err());
}

#[test] // rq-1dadb8e9
fn missing_pressure_rejected() {
    let slot = mc_params("temperature = 9.4e-4\nseed = 1\n");
    assert!(validate(&slot).is_err());
}

#[test] // rq-4207d7ef
fn missing_seed_rejected() {
    let slot = mc_params("pressure = 3.4e-9\ntemperature = 9.4e-4\n");
    assert!(validate(&slot).is_err());
}

// ---------- construction / periodicity ----------

#[test] // rq-220808ef rq-9d8f0e2a
fn construct_via_registry_defaults() {
    let gpu = init_device().unwrap();
    let slot = mc_params("pressure = 3.4e-9\ntemperature = 9.4e-4\nseed = 42\n");
    let baro = build_mc(&gpu, 12, &slot);
    assert_eq!(baro.periodicity(), BarostatPeriodicity::EveryNSteps(25));
    let mc = unbox_mc(baro);
    assert_eq!(mc.frequency, 25);
    assert_eq!(mc.draw_counter, 0);
    assert_eq!(mc.seed, 42);
}

#[test] // rq-21ad8af9
fn volume_step_defaults_to_one_percent_of_volume() {
    let gpu = init_device().unwrap();
    let l: Real = 20.0;
    let sim_box = cube(&gpu, l);
    let v0 = (l * l * l) as f64;
    let slot = mc_params("pressure = 3.4e-9\ntemperature = 9.4e-4\nseed = 1\n");
    let mut baro = build_mc(&gpu, 8, &slot);
    let molecules = MoleculeList::singletons(8);
    baro.init_run(&sim_box, &molecules).unwrap();
    let mc = unbox_mc(baro);
    assert!((mc.max_volume_step - 0.01 * v0).abs() < 1e-6 * v0);
}

#[test] // rq-98c80507
fn apply_is_noop_every_step() {
    let gpu = init_device().unwrap();
    let sim_box0 = cube(&gpu, 20.0);
    let mut sim_box = cube(&gpu, 20.0);
    let mut buffers = make_buffers(
        &gpu,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![1.0; 4],
    );
    let mut timings = Timings::new(&gpu).unwrap();
    let slot = mc_params("pressure = 3.4e-9\ntemperature = 9.4e-4\nseed = 1\n");
    let mut baro = build_mc(&gpu, 4, &slot);
    let gen_before = sim_box.generation();
    baro.apply(&mut buffers, &mut sim_box, 1.0, &mut timings).unwrap();
    assert_eq!(sim_box.generation(), gen_before);
    assert_eq!(sim_box.lattice(), sim_box0.lattice());
}

#[test] // rq-e9b8f0a6
fn apply_move_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let mut sim_box = cube(&gpu, 20.0);
    let mut buffers = make_buffers(&gpu, vec![], vec![], vec![], vec![]);
    let mut ff = empty_force_field(&gpu, 0, &sim_box);
    let mut timings = Timings::new(&gpu).unwrap();
    let slot = mc_params("pressure = 3.4e-9\ntemperature = 9.4e-4\nseed = 1\n");
    let mut baro = build_mc(&gpu, 0, &slot);
    baro.init_run(&sim_box, &MoleculeList::singletons(0)).unwrap();
    let gen0 = sim_box.generation();
    baro.apply_move(&mut ff, &mut buffers, &mut sim_box, None, 1.0, &mut timings)
        .unwrap();
    assert_eq!(sim_box.generation(), gen0);
    let mc = unbox_mc(baro);
    assert_eq!(mc.draw_counter, 0);
}

#[test] // rq-ebe75818
fn log_column_names_are_the_four_mc_columns() {
    let gpu = init_device().unwrap();
    let slot = mc_params("pressure = 3.4e-9\ntemperature = 9.4e-4\nseed = 1\n");
    let baro = build_mc(&gpu, 4, &slot);
    let names: Vec<&str> = baro.log_column_names().iter().map(|(n, _)| *n).collect();
    assert_eq!(
        names,
        vec!["box_volume", "mc_acceptance", "mc_volume_step", "mc_conserved"]
    );
}

// ---------- the move ----------

#[test] // rq-6a983abc
fn draw_counter_increments_per_attempted_move() {
    let gpu = init_device().unwrap();
    let mut sim_box = cube(&gpu, 20.0);
    let mut buffers = make_buffers(
        &gpu,
        vec![1.0, -1.0, 2.0, -2.0],
        vec![0.0; 4],
        vec![0.0; 4],
        vec![1.0; 4],
    );
    let mut ff = empty_force_field(&gpu, 4, &sim_box);
    let mut timings = Timings::new(&gpu).unwrap();
    let slot = mc_params("pressure = 3.4e-9\ntemperature = 9.4e-4\nseed = 7\n");
    let mut baro = build_mc(&gpu, 4, &slot);
    baro.init_run(&sim_box, &MoleculeList::singletons(4)).unwrap();
    baro.apply_move(&mut ff, &mut buffers, &mut sim_box, None, 1.0, &mut timings)
        .unwrap();
    baro.apply_move(&mut ff, &mut buffers, &mut sim_box, None, 1.0, &mut timings)
        .unwrap();
    let mc = unbox_mc(baro);
    assert_eq!(mc.draw_counter, 2);
    assert_eq!(mc.attempted_moves, 2);
}

#[test] // rq-cdd66a5a
fn velocities_never_modified_by_a_move() {
    let gpu = init_device().unwrap();
    let mut sim_box = cube(&gpu, 20.0);
    let n = 4;
    let zero = vec![0.0 as Real; n];
    let vx = vec![0.3, -0.4, 0.5, -0.6];
    let state = ParticleState::new(
        vec![1.0, -1.0, 2.0, -2.0],
        zero.clone(),
        zero.clone(),
        vx.clone(),
        vec![0.1, 0.2, 0.3, 0.4],
        vec![-0.1, -0.2, -0.3, -0.4],
        vec![1.0; n],
        vec![0.0; n],
        (0..n as u32).collect(),
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let vx_before = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy_before = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let mut ff = empty_force_field(&gpu, n, &sim_box);
    let mut timings = Timings::new(&gpu).unwrap();
    let slot = mc_params("pressure = 3.4e-9\ntemperature = 9.4e-4\nseed = 3\n");
    let mut baro = build_mc(&gpu, n, &slot);
    baro.init_run(&sim_box, &MoleculeList::singletons(n)).unwrap();
    baro.apply_move(&mut ff, &mut buffers, &mut sim_box, None, 1.0, &mut timings)
        .unwrap();
    let vx_after = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy_after = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    assert_eq!(vx_before, vx_after);
    assert_eq!(vy_before, vy_after);
}

// ---------- COM-scale kernel ----------

#[test] // rq-2aad1786
fn rigid_molecule_geometry_invariant_under_scale() {
    let gpu = init_device().unwrap();
    // One 3-atom (water-like) molecule.
    let px = vec![1.0 as Real, 1.6, 0.4];
    let py = vec![2.0 as Real, 2.3, 2.3];
    let pz = vec![0.5 as Real, 0.5, 0.5];
    let masses = vec![16.0 as Real, 1.0, 1.0];
    let mut buffers = make_buffers(&gpu, px.clone(), py.clone(), pz.clone(), masses.clone());
    let offsets = gpu.device.htod_sync_copy(&[0u32, 3]).unwrap();
    let indices = gpu.device.htod_sync_copy(&[0u32, 1, 2]).unwrap();
    let scale: Real = 1.25;

    let m: f64 = masses.iter().map(|&x| x as f64).sum();
    let com = |p: &[Real]| p.iter().zip(&masses).map(|(&x, &mm)| x as f64 * mm as f64).sum::<f64>() / m;
    let (cx0, cy0, cz0) = (com(&px), com(&py), com(&pz));

    mc_barostat_scale_molecule_com(&mut buffers, &offsets, &indices, scale).unwrap();
    let (qx, qy, qz) = buffers.download_positions().unwrap();

    // Internal displacements preserved.
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        assert!((((qx[a] - qx[b]) - (px[a] - px[b])) as f64).abs() < 1e-5);
        assert!((((qy[a] - qy[b]) - (py[a] - py[b])) as f64).abs() < 1e-5);
    }
    // COM scaled by `scale`.
    let (cx1, cy1, cz1) = (com(&qx), com(&qy), com(&qz));
    assert!((cx1 - scale as f64 * cx0).abs() < 1e-4 * cx0.abs().max(1.0));
    assert!((cy1 - scale as f64 * cy0).abs() < 1e-4 * cy0.abs().max(1.0));
    assert!((cz1 - scale as f64 * cz0).abs() < 1e-4 * cz0.abs().max(1.0));
}

#[test] // rq-92eee9d8
fn singleton_molecules_scale_each_atom_about_origin() {
    let gpu = init_device().unwrap();
    let px = vec![1.0 as Real, -2.0, 3.0, -4.0];
    let py = vec![0.5 as Real, 1.5, -2.5, 3.5];
    let pz = vec![0.0 as Real, 0.0, 0.0, 0.0];
    let n = 4;
    let mut buffers = make_buffers(&gpu, px.clone(), py.clone(), pz.clone(), vec![1.0; n]);
    let offsets = gpu.device.htod_sync_copy(&[0u32, 1, 2, 3, 4]).unwrap();
    let indices = gpu.device.htod_sync_copy(&[0u32, 1, 2, 3]).unwrap();
    let scale: Real = 0.8;
    mc_barostat_scale_molecule_com(&mut buffers, &offsets, &indices, scale).unwrap();
    let (qx, qy, _) = buffers.download_positions().unwrap();
    for i in 0..n {
        assert!(((qx[i] - scale * px[i]) as f64).abs() < 1e-5);
        assert!(((qy[i] - scale * py[i]) as f64).abs() < 1e-5);
    }
}

// ---------- end-to-end ----------

fn write_npt_config(dir: &std::path::Path, graphs_disable: bool) {
    // Small disordered LJ argon system with a monte-carlo barostat.
    let side = 6usize;
    let n = side * side * side;
    let spacing = 6.0e-10_f64;
    let l = side as f64 * spacing;
    let c = (side as f64 - 1.0) / 2.0;
    let mut lcg: u64 = 0xBEEF_1234;
    let mut jitter = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((lcg >> 33) as f64 / (1u64 << 31) as f64) - 0.5) * 0.4 * spacing
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
    let cfg = format!(
        r#"schema_version = 1
init = "sim.in.xyz"

[simulation]
cuda_graphs_disable = {graphs_disable}
seed = 1
temperature = 120.0

[[phase]]
name = "run"
n_steps = 12
dt = 1.0e-15

[phase.integrator]
kind = "velocity-verlet"
lossless = false

[phase.thermostat]
kind = "csvr"
temperature = 120.0
tau = 1.0e-13
seed = 5

[phase.barostat]
kind = "monte-carlo"
pressure = 1.0e5
temperature = 120.0
frequency = 4
seed = 9

[phase.output]
trajectory_every = 2
log_every = 2

[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 5.0e-10
"#
    );
    std::fs::write(dir.join("sim.in.toml"), cfg).unwrap();
}

fn tmp(name: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("heddle_mc_{name}"));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

#[test] // rq-656ce38f
fn log_header_includes_mc_columns() {
    let dir = tmp("header");
    write_npt_config(&dir, true);
    heddle_md::runner::run_simulation(&dir.join("sim.in.toml")).unwrap();
    let log = std::fs::read_to_string(dir.join("sim.out.run.log")).unwrap();
    let header = log.lines().next().unwrap();
    assert!(
        header.ends_with("box_volume,mc_acceptance,mc_volume_step,mc_conserved"),
        "header was: {header}"
    );
}

#[test] // rq-5af90224
fn two_runs_with_same_seed_are_byte_identical() {
    let d1 = tmp("det1");
    let d2 = tmp("det2");
    write_npt_config(&d1, false);
    write_npt_config(&d2, false);
    heddle_md::runner::run_simulation(&d1.join("sim.in.toml")).unwrap();
    heddle_md::runner::run_simulation(&d2.join("sim.in.toml")).unwrap();
    let log1 = std::fs::read(d1.join("sim.out.run.log")).unwrap();
    let log2 = std::fs::read(d2.join("sim.out.run.log")).unwrap();
    assert_eq!(log1, log2, "monte-carlo NPT logs are not run-to-run deterministic");
    let traj1 = std::fs::read(d1.join("sim.out.run.xyz")).unwrap();
    let traj2 = std::fs::read(d2.join("sim.out.run.xyz")).unwrap();
    assert_eq!(traj1, traj2);
}

#[test] // rq-0bc3a66e (graph vs non-graph parity for an MC barostat phase)
fn graph_and_non_graph_runs_are_byte_identical() {
    let dg = tmp("graph");
    let dn = tmp("nograph");
    write_npt_config(&dg, false); // graphs enabled
    write_npt_config(&dn, true); // graphs disabled
    heddle_md::runner::run_simulation(&dg.join("sim.in.toml")).unwrap();
    heddle_md::runner::run_simulation(&dn.join("sim.in.toml")).unwrap();
    let lg = std::fs::read(dg.join("sim.out.run.log")).unwrap();
    let ln = std::fs::read(dn.join("sim.out.run.log")).unwrap();
    assert_eq!(lg, ln, "graph-mode and per-step MC barostat runs differ");
}
