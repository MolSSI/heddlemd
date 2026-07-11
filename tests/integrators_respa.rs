//! RESPA multiple-timestep integrator tests.
//! Implements `rqm/integration/respa.md`'s Gherkin scenarios.

use std::path::{Path, PathBuf};

use cudarc::driver::DeviceSlice;
use heddle_md::forces::{
    AggregateLevel, AngleList, BondList, DihedralList, ExclusionList, ForceClass, ForceField,
    PotentialRegistry,
};
use heddle_md::gpu::{GpuContext, ParticleBuffers, init_device};
use heddle_md::integrator::{
    Integrator, IntegratorBuilder, IntegratorRegistry, IntegratorStepExt, KickSource,
    RespaBuilder, RespaIntegrator, RunStepOptions, StepPlan, SubStep, run_step,
};
use heddle_md::io::SlotConfig;
use heddle_md::io::config::{ConfigError, NeighborListConfig, PairInteractionConfig, ParticleTypeConfig, SpmeConfig};
use heddle_md::pbc::SimulationBox;
use heddle_md::precision::Real;
use heddle_md::state::ParticleState;
use heddle_md::timings::Timings;

// =================================================================
// Helpers
// =================================================================

fn box_10(gpu: &GpuContext) -> SimulationBox {
    SimulationBox::new(&gpu.device, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0).unwrap()
}

fn ar_type(charge: f64) -> ParticleTypeConfig {
    ParticleTypeConfig {
        name: "Ar".to_string(),
        mass: 1.0,
        sigma: None,
        epsilon: None,
        charge,
    }
}

fn lj_pair(sigma: f64, epsilon: f64, cutoff: f64) -> PairInteractionConfig {
    PairInteractionConfig::lennard_jones(
        ("Ar".to_string(), "Ar".to_string()),
        sigma,
        epsilon,
        cutoff,
        Some(cutoff),
    )
}

fn respa_kind(n_inner: u32) -> SlotConfig {
    SlotConfig::from_params_str("respa", &format!("n_inner = {n_inner}\n"))
}

fn state_n(n: usize, spacing: Real) -> ParticleState {
    let pos: Vec<Real> = (0..n).map(|i| i as Real * spacing).collect();
    let zero = vec![0.0; n];
    ParticleState::new(
        pos,
        zero.clone(),
        zero.clone(),
        zero.clone(),
        zero.clone(),
        zero,
        vec![1.0; n],
        vec![0.0; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap()
}

#[allow(clippy::too_many_arguments)]
fn build_ff(
    gpu: &GpuContext,
    n: usize,
    pair_interactions: &[PairInteractionConfig],
    particle_types: &[ParticleTypeConfig],
    spme: Option<&SpmeConfig>,
    charges: &[Real],
) -> ForceField {
    ForceField::new(
        &PotentialRegistry::with_builtins(),
        gpu,
        n,
        &box_10(gpu),
        particle_types,
        pair_interactions,
        &[],
        &[],
        &[],
        spme,
        charges,
        &BondList::empty(n),
        &AngleList::empty(0),
        &DihedralList::empty(0),
        &ExclusionList::empty(n),
        &NeighborListConfig::AllPairs,
    )
    .unwrap()
}

fn build_respa(gpu: &GpuContext, n: usize, n_inner: u32) -> Box<dyn Integrator> {
    IntegratorRegistry::with_builtins()
        .build(&respa_kind(n_inner), gpu, n, 0)
        .unwrap()
}

// =================================================================
// Construction and plan shape
// =================================================================

// rq-b141648c
#[test]
fn registry_builds_respa() {
    let gpu = init_device().unwrap();
    let _integrator = build_respa(&gpu, 4, 4);
}

// rq-204ec51c
#[test]
fn plan_is_the_unrolled_inner_loop() {
    let gpu = init_device().unwrap();
    let integrator = build_respa(&gpu, 4, 4);
    let dt: Real = 0.4;
    let delta = dt / 4.0;
    let plan = integrator.plan(dt);
    assert_eq!(plan.steps.len(), 15);
    let outer = |s: &SubStep| {
        matches!(s, SubStep::KickHalf { dt: d, source: KickSource::Class(ForceClass::Slow), .. } if *d == dt)
    };
    assert!(outer(&plan.steps[0]));
    for rep in 0..4 {
        let base = 1 + rep * 3;
        assert!(matches!(
            plan.steps[base],
            SubStep::KickDrift { dt: d, source: KickSource::Class(ForceClass::Fast), .. } if d == delta
        ));
        assert!(matches!(
            plan.steps[base + 1],
            SubStep::ForceEval { class: Some(ForceClass::Fast), .. }
        ));
        assert!(matches!(
            plan.steps[base + 2],
            SubStep::KickHalf { dt: d, source: KickSource::Class(ForceClass::Fast), .. } if d == delta
        ));
    }
    assert!(matches!(
        plan.steps[13],
        SubStep::ForceEval { class: Some(ForceClass::Slow), .. }
    ));
    assert!(outer(&plan.steps[14]));
}

// rq-5c63aa07
#[test]
fn plan_shape_is_pure_in_dt() {
    let gpu = init_device().unwrap();
    let integrator = build_respa(&gpu, 4, 2);
    let a = integrator.plan(0.25);
    let b = integrator.plan(0.25);
    assert_eq!(a.steps.len(), b.steps.len());
    for (x, y) in a.steps.iter().zip(b.steps.iter()) {
        assert_eq!(x.variant_name(), y.variant_name());
    }
}

// rq-b0a520ac
#[test]
fn plan_contains_no_thermostat_markers() {
    let gpu = init_device().unwrap();
    let integrator = build_respa(&gpu, 4, 2);
    assert!(!integrator.plan(0.25).has_thermostat_points());
}

// rq-482921eb
#[test]
fn plan_ends_with_trailing_outer_half_kick() {
    let gpu = init_device().unwrap();
    let integrator = build_respa(&gpu, 4, 4);
    let plan = integrator.plan(0.4);
    match plan.steps.last().expect("plan is non-empty") {
        SubStep::KickHalf { dt, label, source } => {
            assert!(matches!(source, KickSource::Class(ForceClass::Slow)));
            assert_eq!(*label, "respa_outer_kick");
            assert_eq!(*dt, 0.4 as Real);
        }
        other => panic!("expected trailing KickHalf, got {}", other.variant_name()),
    }
}

// =================================================================
// Config validation
// =================================================================

// rq-d62c351e
#[test]
fn reject_n_inner_zero() {
    let err = RespaBuilder
        .validate_params(&toml::from_str("n_inner = 0").unwrap())
        .unwrap_err();
    match err {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "integrator.n_inner"),
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

// rq-d9a51e75
#[test]
fn accept_n_inner_one() {
    RespaBuilder
        .validate_params(&toml::from_str("n_inner = 1").unwrap())
        .unwrap();
}

// rq-7e9e52c4 — RESPA declares itself constraint-incompatible; the
// framework's standard integrator/constraint validation (exercised in
// constraint_framework.rs) rejects the pairing from this predicate.
#[test]
fn respa_does_not_support_constraints() {
    assert!(!RespaBuilder.supports_constraints(&toml::from_str("n_inner = 2").unwrap()));
}

// rq-ecd52ccb
#[test]
fn reject_lossless_mode() {
    let err = RespaBuilder
        .validate_params(&toml::from_str("n_inner = 2\nlossless = true").unwrap())
        .unwrap_err();
    match err {
        ConfigError::InvalidValue { field, .. } => assert_eq!(field, "integrator.lossless"),
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

// =================================================================
// Dynamics (in-process, ext-step driven)
// =================================================================

// rq-19504e09
#[test]
fn fast_only_system_matches_velocity_verlet_at_inner_timestep() {
    let gpu = init_device().unwrap();
    let n = 8;
    let pairs = [lj_pair(1.0, 1.0e-3, 4.0)];
    let types = [ar_type(0.0)];
    let charges = vec![0.0 as Real; n];
    let dt_outer: Real = 0.02;
    let dt_inner: Real = 0.01;

    // RESPA: 5 outer steps of 2 inner steps each.
    let state = state_n(n, 1.2);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = build_ff(&gpu, n, &pairs, &types, None, &charges);
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &sim_box, &mut timings, AggregateLevel::ForcesAndScalars)
        .unwrap();
    let mut respa = RespaIntegrator::new(2);
    for _ in 0..5 {
        respa
            .step(&mut buffers, &mut sim_box, &mut ff, dt_outer, &mut timings)
            .unwrap();
    }
    let rx = gpu.device.dtoh_sync_copy(&buffers.posq).unwrap();
    let rvx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();

    // Velocity Verlet: 10 steps of the inner timestep.
    let state = state_n(n, 1.2);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = build_ff(&gpu, n, &pairs, &types, None, &charges);
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &sim_box, &mut timings, AggregateLevel::ForcesAndScalars)
        .unwrap();
    let mut vv = heddle_md::integrator::VelocityVerletState::new(&gpu, n, false).unwrap();
    for _ in 0..10 {
        vv.step(&mut buffers, &mut sim_box, &mut ff, dt_inner, &mut timings)
            .unwrap();
    }
    let vx = gpu.device.dtoh_sync_copy(&buffers.posq).unwrap();
    let vvx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();

    for i in 0..n {
        let dp = (rx[i].x - vx[i].x).abs();
        assert!(
            dp <= 1.0e-5 * vx[i].x.abs().max(1.0),
            "position {i}: respa {} vs vv {}",
            rx[i].x,
            vx[i].x
        );
        let dv = (rvx[i] - vvx[i]).abs();
        assert!(
            dv <= 1.0e-5 * vvx[i].abs().max(1.0e-3),
            "velocity {i}: respa {} vs vv {}",
            rvx[i],
            vvx[i]
        );
    }
}

// rq-bcec000b — with no slots at all, ForceEval sub-steps are no-ops
// and the manually seeded slow accumulator is preserved, so one outer
// step applies exactly the two analytic slow impulses.
#[test]
fn slow_only_system_applies_pure_outer_impulses() {
    let gpu = init_device().unwrap();
    let n = 1;
    let state = state_n(n, 0.0); // at rest at the origin, m = 1
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = build_ff(&gpu, n, &[], &[ar_type(0.0)], None, &[0.0]);
    let f: Real = 0.25;
    gpu.device
        .htod_sync_copy_into(&vec![f; n], &mut ff.slow_total_forces_x)
        .unwrap();
    let dt: Real = 0.5;
    let mut respa = RespaIntegrator::new(2);
    let mut timings = Timings::new(&gpu).unwrap();
    respa
        .step(&mut buffers, &mut sim_box, &mut ff, dt, &mut timings)
        .unwrap();
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let posq = gpu.device.dtoh_sync_copy(&buffers.posq).unwrap();
    // v = (F/m)·dt (two half-impulses); x = (F/m)·(dt/2)·dt (the inner
    // drifts run at v = F·dt/2).
    let expect_v = f * dt;
    let expect_x = f * dt * 0.5 * dt;
    assert!((vx[0] - expect_v).abs() <= 1.0e-6 * expect_v.abs());
    assert!((posq[0].x - expect_x).abs() <= 1.0e-6 * expect_x.abs());
}

// rq-860c8ff9
#[test]
fn nve_energy_is_conserved() {
    let gpu = init_device().unwrap();
    // 4x4x4 lattice, alternating charges, LJ + SPME.
    let side = 4;
    let n = side * side * side;
    let spacing: Real = 2.5;
    let mut px = Vec::with_capacity(n);
    let mut py = Vec::with_capacity(n);
    let mut pz = Vec::with_capacity(n);
    let mut charges: Vec<Real> = Vec::with_capacity(n);
    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                px.push(i as Real * spacing);
                py.push(j as Real * spacing);
                pz.push(k as Real * spacing);
                charges.push(if (i + j + k) % 2 == 0 { 0.05 } else { -0.05 });
            }
        }
    }
    let state = ParticleState::new(
        px,
        py,
        pz,
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        charges.clone(),
        vec![0u32; n],
        None,
        None,
    )
    .unwrap();
    let spme = SpmeConfig {
        alpha: 0.3,
        r_cut_real: 4.5,
        grid: [16, 16, 16],
        spline_order: 5,
    };
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = build_ff(
        &gpu,
        n,
        &[lj_pair(1.0, 1.0e-4, 4.0)],
        &[ar_type(0.0)],
        Some(&spme),
        &charges,
    );
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &sim_box, &mut timings, AggregateLevel::ForcesAndScalars)
        .unwrap();
    let mut respa = RespaIntegrator::new(4);
    let mut ke_scratch = gpu.device.alloc_zeros::<Real>(1).unwrap();
    let mut pe_scratch = gpu.device.alloc_zeros::<Real>(1).unwrap();
    let energy = |buffers: &mut ParticleBuffers,
                  ke_scratch: &mut cudarc::driver::CudaSlice<Real>,
                  pe_scratch: &mut cudarc::driver::CudaSlice<Real>|
     -> f64 {
        let ke = heddle_md::gpu::compute_kinetic_energy(buffers, ke_scratch).unwrap() as f64;
        let pe =
            heddle_md::gpu::compute_total_potential_energy(buffers, pe_scratch).unwrap() as f64;
        ke + pe
    };
    let e0 = energy(&mut buffers, &mut ke_scratch, &mut pe_scratch);
    let mut max_dev: f64 = 0.0;
    for _ in 0..200 {
        respa
            .step(&mut buffers, &mut sim_box, &mut ff, 0.05, &mut timings)
            .unwrap();
        let e = energy(&mut buffers, &mut ke_scratch, &mut pe_scratch);
        max_dev = max_dev.max(((e - e0) / e0.abs()).abs());
    }
    assert!(
        max_dev < 1.0e-3,
        "relative NVE energy drift {max_dev} exceeds 1e-3 (E0 = {e0})"
    );
}

// rq-f746a29f
#[test]
fn outer_kicks_are_bit_exact_noops_without_slow_slots() {
    let gpu = init_device().unwrap();
    let n = 1;
    // Non-trivial velocity, no potentials at all: the slow accumulator
    // is zero-initialised, so the outer kicks must add exactly 0.0.
    let state = ParticleState::new(
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.37],
        vec![-0.21],
        vec![0.093],
        vec![1.0],
        vec![0.0],
        vec![0u32],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = build_ff(&gpu, n, &[], &[ar_type(0.0)], None, &[0.0]);
    let mut respa = RespaIntegrator::new(2);
    let mut timings = Timings::new(&gpu).unwrap();
    respa
        .step(&mut buffers, &mut sim_box, &mut ff, 0.5, &mut timings)
        .unwrap();
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vz = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
    assert_eq!(vx, vec![0.37 as Real]);
    assert_eq!(vy, vec![-0.21 as Real]);
    assert_eq!(vz, vec![0.093 as Real]);
}

// =================================================================
// Full-runner scenarios
// =================================================================

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("heddle_respa_{name}"));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn write_lattice_init(dir: &Path, side: usize, spacing: f64) {
    let n = side * side * side;
    let l = side as f64 * spacing;
    let mut body = format!("{n}\n");
    body.push_str(&format!(
        "Lattice=\"{l:.6e} 0 0 0 {l:.6e} 0 0 0 {l:.6e}\" Properties=species:S:1:pos:R:3\n"
    ));
    let c = (side as f64 - 1.0) / 2.0;
    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                let px = (i as f64 - c) * spacing;
                let py = (j as f64 - c) * spacing;
                let pz = (k as f64 - c) * spacing;
                body.push_str(&format!("Ar {px:.9e} {py:.9e} {pz:.9e}\n"));
            }
        }
    }
    std::fs::write(dir.join("sim.in.xyz"), body).unwrap();
}

fn respa_lj_config(n_steps: u64, n_inner: u32, extra: &str) -> String {
    format!(
        r#"schema_version = 1
init = "sim.in.xyz"

[simulation]
seed = 1
temperature = 30.0

[[phase]]
name = "run"
n_steps = {n_steps}
dt = 2.0e-15

[phase.integrator]
kind = "respa"
n_inner = {n_inner}

[phase.output]
trajectory_every = 1
log_every = 1

{extra}

[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
kind = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 9.0e-10
"#
    )
}

// rq-c8835814
#[test]
fn two_identical_respa_runs_are_byte_identical() {
    // n_steps > the graph-calibration prefix so the captured unrolled
    // RESPA plan replays for most of the run.
    let mut trajectories = Vec::new();
    for r in 0..2 {
        let dir = tmp(&format!("determinism{r}"));
        write_lattice_init(&dir, 9, 4.4e-10);
        std::fs::write(dir.join("sim.in.toml"), respa_lj_config(12, 2, "")).unwrap();
        heddle_md::runner::run_simulation(&dir.join("sim.in.toml")).unwrap();
        trajectories.push(std::fs::read(dir.join("sim.out.run.xyz")).unwrap());
    }
    assert_eq!(trajectories[0], trajectories[1]);
}

// rq-4bf89376
#[test]
fn reject_respa_with_barostat() {
    let dir = tmp("barostat");
    write_lattice_init(&dir, 9, 4.4e-10);
    let extra = "[phase.barostat]\nkind = \"monte-carlo\"\npressure = 1.0e5\ntemperature = 30.0\nfrequency = 4\nseed = 7\n";
    std::fs::write(dir.join("sim.in.toml"), respa_lj_config(2, 2, extra)).unwrap();
    let err = heddle_md::runner::run_simulation(&dir.join("sim.in.toml")).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("respa") && msg.contains("barostat"),
        "unexpected error: {msg}"
    );
}

// rq-64893fbb
#[test]
fn respa_with_zero_particles_completes() {
    let dir = tmp("n0");
    std::fs::write(
        dir.join("sim.in.xyz"),
        "0\nLattice=\"5e-9 0 0 0 5e-9 0 0 0 5e-9\" Properties=species:S:1:pos:R:3\n",
    )
    .unwrap();
    std::fs::write(dir.join("sim.in.toml"), respa_lj_config(2, 2, "")).unwrap();
    let summary = heddle_md::runner::run_simulation(&dir.join("sim.in.toml")).unwrap();
    assert_eq!(summary.total_n_steps, 2);
}

// rq-c3746bb2
#[test]
fn thermostat_couples_once_per_outer_step() {
    // CSVR paired with RESPA through the full runner: the run completes
    // and the thermostat's log column is emitted once per outer step,
    // demonstrating outer-boundary coupling (the default wrapping
    // topology; RESPA's plan carries no ThermostatHalf markers).
    let dir = tmp("thermostat");
    write_lattice_init(&dir, 9, 4.4e-10);
    let extra =
        "[phase.thermostat]\nkind = \"csvr\"\ntemperature = 30.0\ntau = 1.0e-13\nseed = 11\n";
    std::fs::write(dir.join("sim.in.toml"), respa_lj_config(4, 2, extra)).unwrap();
    let summary = heddle_md::runner::run_simulation(&dir.join("sim.in.toml")).unwrap();
    assert_eq!(summary.total_n_steps, 4);
    // log_every = 1 → one row per outer step plus the step-0 row; the
    // CSVR conserved-quantity column is present, and its values are
    // finite — the thermostat ran once per outer step, not once per
    // inner step (an inner-step cadence would show in the conserved
    // quantity, which is exact only under outer-boundary coupling).
    let log = std::fs::read_to_string(dir.join("sim.out.run.log")).unwrap();
    let header = log.lines().next().unwrap();
    assert!(header.contains("csvr_conserved"), "header: {header}");
    // header + rows for steps 0..=4.
    assert_eq!(log.lines().count(), 1 + 5, "log: {log}");
    let csvr_col = header.split(',').position(|c| c == "csvr_conserved").unwrap();
    for row in log.lines().skip(1) {
        let v: f64 = row.split(',').nth(csvr_col).unwrap().parse().unwrap();
        assert!(v.is_finite());
    }
}

// =================================================================
// Runner thermostat topology (framework.md marker scenarios)
// =================================================================

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use heddle_md::integrator::{
    IntegratorError, Thermostat, ThermostatBuilder, ThermostatError, ThermostatPhase,
};
use heddle_md::registries::Registries;
use heddle_md::registry::KindedBuilder;

#[derive(Debug)]
struct CountingThermostat {
    pre: Arc<AtomicU64>,
    post: Arc<AtomicU64>,
}

impl Thermostat for CountingThermostat {
    fn apply_pre(
        &mut self,
        _b: &mut ParticleBuffers,
        _dt: Real,
        _t: &mut Timings,
    ) -> Result<(), ThermostatError> {
        self.pre.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn apply_post(
        &mut self,
        _b: &mut ParticleBuffers,
        _dt: Real,
        _t: &mut Timings,
    ) -> Result<(), ThermostatError> {
        self.post.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    // This thermostat is self-contained in apply_pre/apply_post and
    // performs its own standalone launches.
}

#[derive(Debug, Clone)]
struct CountingThermostatBuilder {
    pre: Arc<AtomicU64>,
    post: Arc<AtomicU64>,
}

impl KindedBuilder for CountingThermostatBuilder {
    fn kind_name(&self) -> &'static str {
        "counting-thermostat"
    }
}

impl ThermostatBuilder for CountingThermostatBuilder {
    fn validate_params(&self, _params: &toml::Value) -> Result<(), ConfigError> {
        Ok(())
    }
    // Host-side counters mutate between launches.
    fn graph_compatible(&self, _params: &toml::Value) -> bool {
        false
    }
    fn build(
        &self,
        _gpu: &GpuContext,
        _particle_count: usize,
        _n_constraints: usize,
        _params: &toml::Value,
    ) -> Result<Box<dyn Thermostat>, ThermostatError> {
        Ok(Box::new(CountingThermostat {
            pre: self.pre.clone(),
            post: self.post.clone(),
        }))
    }
}

/// Stub integrator whose plan optionally carries ThermostatHalf
/// markers (one Post marker, no Pre marker).
#[derive(Debug)]
struct MarkerStubIntegrator {
    markers: bool,
}

impl Integrator for MarkerStubIntegrator {
    fn plan(&self, dt: Real) -> StepPlan {
        let mut steps = vec![SubStep::Drift { dt, label: "d" }];
        if self.markers {
            steps.push(SubStep::ThermostatHalf { dt, phase: ThermostatPhase::Post });
        }
        StepPlan { steps }
    }
    fn execute(
        &mut self,
        _substep: &SubStep,
        _b: &mut ParticleBuffers,
        _sb: &mut SimulationBox,
        _t: &mut Timings,
    ) -> Result<(), IntegratorError> {
        Ok(())
    }
    // The plan walk executes every sub-step as standalone launches;
    // no post-force kernel is composed for this integrator.
}

#[derive(Debug, Clone)]
struct MarkerStubBuilder {
    markers: bool,
    kind: &'static str,
}

impl KindedBuilder for MarkerStubBuilder {
    fn kind_name(&self) -> &'static str {
        self.kind
    }
}

impl IntegratorBuilder for MarkerStubBuilder {
    fn validate_params(&self, _params: &toml::Value) -> Result<(), ConfigError> {
        Ok(())
    }
    fn build(
        &self,
        _gpu: &GpuContext,
        _particle_count: usize,
        _n_constraints: usize,
        _params: &toml::Value,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        Ok(Box::new(MarkerStubIntegrator { markers: self.markers }))
    }
}

fn marker_topology_config(kind: &str) -> String {
    format!(
        r#"schema_version = 1
init = "sim.in.xyz"

[simulation]
cuda_graphs_disable = true
seed = 1
temperature = 30.0

[[phase]]
name = "run"
n_steps = 3
dt = 2.0e-15

[phase.integrator]
kind = "{kind}"

[phase.thermostat]
kind = "counting-thermostat"

[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
kind = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 9.0e-10
"#
    )
}

fn run_marker_topology(name: &str, markers: bool, kind: &'static str) -> (u64, u64) {
    let dir = tmp(name);
    write_lattice_init(&dir, 9, 4.4e-10);
    std::fs::write(dir.join("sim.in.toml"), marker_topology_config(kind)).unwrap();
    let pre = Arc::new(AtomicU64::new(0));
    let post = Arc::new(AtomicU64::new(0));
    let mut registries = Registries::with_builtins();
    registries.register_integrator(Box::new(MarkerStubBuilder { markers, kind }));
    registries.register_thermostat(Box::new(CountingThermostatBuilder {
        pre: pre.clone(),
        post: post.clone(),
    }));
    heddle_md::runner::run_simulation_with_registries(&dir.join("sim.in.toml"), &registries)
        .unwrap();
    (pre.load(Ordering::SeqCst), post.load(Ordering::SeqCst))
}

// rq-8c0c385b
#[test]
fn marker_bearing_plan_suppresses_default_thermostat_wrapping() {
    // Plan carries one Post marker and no Pre marker: the runner's
    // default wrap is suppressed, so apply_pre never fires and
    // apply_post fires exactly once per step at the marker.
    let (pre, post) = run_marker_topology("marker_plan", true, "marker-stub");
    assert_eq!(pre, 0);
    assert_eq!(post, 3);
}

// rq-177b7289
#[test]
fn marker_free_plan_keeps_default_wrapping() {
    let (pre, post) = run_marker_topology("plain_plan", false, "plain-stub");
    assert_eq!(pre, 3);
    assert_eq!(post, 3);
}
