//! Post-force composed-kernel participation and fallback tests.
//! Implements the suffix-closed subset scenarios in
//! `rqm/integration/jit-composed-post-force.md` and the graph-mode
//! scenario in `rqm/cuda-graphs.md`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::CudaSlice;
use heddle_md::gpu::{GpuContext, ParticleBuffers, compute_kinetic_energy, init_device};
use heddle_md::integrator::{
    Barostat, BarostatBuilder, BarostatError, BarostatRegistry, Integrator, IntegratorBuilder,
    IntegratorError, IntegratorRegistry, KickSource, StepPlan, SubStep, Thermostat,
    ThermostatBuilder, ThermostatError, ThermostatRegistry,
};
use heddle_md::io::SlotConfig;
use heddle_md::io::config::ConfigError;
use heddle_md::pbc::SimulationBox;
use heddle_md::precision::Real;
use heddle_md::registries::Registries;
use heddle_md::registry::KindedBuilder;
use heddle_md::runner::{RunnerError, run_simulation, run_simulation_with_registries};
use heddle_md::timings::Timings;

// =================================================================
// Helpers
// =================================================================

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("heddle_postforce_{name}"));
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
                body.push_str(&format!(
                    "Ar {:.9e} {:.9e} {:.9e}\n",
                    (i as f64 - c) * spacing,
                    (j as f64 - c) * spacing,
                    (k as f64 - c) * spacing
                ));
            }
        }
    }
    std::fs::write(dir.join("sim.in.xyz"), body).unwrap();
}

fn lj_config(n_steps: u64, integrator_toml: &str, extra: &str, graphs: bool) -> String {
    let disable = if graphs { "" } else { "cuda_graphs_disable = true\n" };
    format!(
        r#"schema_version = 1
init = "sim.in.xyz"

[simulation]
{disable}seed = 1
temperature = 30.0

[[phase]]
name = "run"
n_steps = {n_steps}
dt = 2.0e-15

[phase.integrator]
{integrator_toml}

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

/// Parse the timings file into (stage, count) pairs.
fn timings_counts(path: &Path) -> Vec<(String, u64)> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let stage = it.next()?.to_string();
            let count: u64 = it.next()?.parse().ok()?;
            Some((stage, count))
        })
        .collect()
}

fn stage_count(counts: &[(String, u64)], stage: &str) -> u64 {
    counts
        .iter()
        .find(|(s, _)| s == stage)
        .map(|(_, c)| *c)
        .unwrap_or(0)
}

/// A non-participating integrator whose plan is velocity Verlet's
/// shape; execute performs real kicks via the vv kernels, so the
/// dynamics are identical to VV while post_force_per_particle()
/// returns None (the trait default).
#[derive(Debug)]
struct EagerVvIntegrator {
    kick_execs: Arc<AtomicU64>,
}

impl Integrator for EagerVvIntegrator {
    fn plan(&self, dt: Real) -> StepPlan {
        StepPlan {
            steps: vec![
                SubStep::KickDrift { dt, label: "kd", source: KickSource::Total },
                SubStep::ForceEval {
                    class: None,
                    level: Some(heddle_md::forces::AggregateLevel::ForcesOnly),
                },
                SubStep::KickHalf { dt, label: "k", source: KickSource::Total },
            ],
        }
    }
    fn execute(
        &mut self,
        substep: &SubStep,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        _timings: &mut Timings,
    ) -> Result<(), IntegratorError> {
        match substep {
            SubStep::KickDrift { dt, .. } => {
                heddle_md::gpu::vv_kick_drift(buffers, sim_box, *dt)?;
            }
            SubStep::KickHalf { dt, .. } => {
                self.kick_execs.fetch_add(1, Ordering::SeqCst);
                heddle_md::gpu::vv_kick(buffers, *dt)?;
            }
            other => {
                return Err(IntegratorError::UnexpectedSubStep {
                    variant: other.variant_name(),
                });
            }
        }
        Ok(())
    }
    // post_force_per_particle: trait default None — this integrator
    // does not participate in the composed kernel.
}

#[derive(Debug, Clone)]
struct EagerVvBuilder {
    kick_execs: Arc<AtomicU64>,
}

impl KindedBuilder for EagerVvBuilder {
    fn kind_name(&self) -> &'static str {
        "eager-vv"
    }
}

impl IntegratorBuilder for EagerVvBuilder {
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
        Ok(Box::new(EagerVvIntegrator { kick_execs: self.kick_execs.clone() }))
    }
}

/// A non-participating thermostat that performs its full work inside
/// apply_post (counts calls; physics-neutral).
#[derive(Debug)]
struct EagerThermostat {
    post: Arc<AtomicU64>,
}

impl Thermostat for EagerThermostat {
    fn apply_post(
        &mut self,
        _b: &mut ParticleBuffers,
        _dt: Real,
        _t: &mut Timings,
    ) -> Result<(), ThermostatError> {
        self.post.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct EagerThermostatBuilder {
    post: Arc<AtomicU64>,
}

impl KindedBuilder for EagerThermostatBuilder {
    fn kind_name(&self) -> &'static str {
        "eager-thermostat"
    }
}

impl ThermostatBuilder for EagerThermostatBuilder {
    fn validate_params(&self, _params: &toml::Value) -> Result<(), ConfigError> {
        Ok(())
    }
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
        Ok(Box::new(EagerThermostat { post: self.post.clone() }))
    }
}

/// A per-step barostat without a post-force fragment (no-op physics).
#[derive(Debug)]
struct FragmentlessBarostat;

impl Barostat for FragmentlessBarostat {}

#[derive(Debug, Clone)]
struct FragmentlessBarostatBuilder;

impl KindedBuilder for FragmentlessBarostatBuilder {
    fn kind_name(&self) -> &'static str {
        "fragmentless-barostat"
    }
}

impl BarostatBuilder for FragmentlessBarostatBuilder {
    fn validate_params(&self, _params: &toml::Value) -> Result<(), ConfigError> {
        Ok(())
    }
    fn build(
        &self,
        _gpu: &GpuContext,
        _particle_count: usize,
        _n_constraints: usize,
        _params: &toml::Value,
    ) -> Result<Box<dyn Barostat>, BarostatError> {
        Ok(Box::new(FragmentlessBarostat))
    }
}

// =================================================================
// Scenarios
// =================================================================

// rq-dcd0d421 — a non-participating integrator's trailing kick executes
// in the plan walk. With a thermostat present and coupling every step,
// no composed kernel is built at all (the integrator has no fragment and
// the thermostat never participates), so the kick dispatches to execute()
// each step and the composed stage records nothing.
#[test]
fn non_participating_integrator_kick_executes_in_plan_walk() {
    let dir = tmp("eager_integrator");
    write_lattice_init(&dir, 9, 4.4e-10);
    let extra = "[phase.thermostat]\nkind = \"csvr\"\ntemperature = 30.0\ntau = 1.0e-13\nseed = 3\n";
    std::fs::write(
        dir.join("sim.in.toml"),
        lj_config(3, "kind = \"eager-vv\"", extra, false),
    )
    .unwrap();
    let kick_execs = Arc::new(AtomicU64::new(0));
    let mut registries = Registries::with_builtins();
    registries.register_integrator(Box::new(EagerVvBuilder { kick_execs: kick_execs.clone() }));
    run_simulation_with_registries(&dir.join("sim.in.toml"), &registries).unwrap();
    // The trailing KickHalf was dispatched to execute() every step —
    // never skipped in favour of the composed kernel.
    assert_eq!(kick_execs.load(Ordering::SeqCst), 3);
    let counts = timings_counts(&dir.join("sim.out.run.timings"));
    assert_eq!(stage_count(&counts, "jit_composed_post_force"), 0);
    // The thermostat coupled (its rescale ran) every step.
    assert_eq!(stage_count(&counts, "csvr_rescale_velocities"), 3);
}

// rq-1d9788b5 — a per-step barostat configured with an integrator whose
// plan emits no BarostatPoint is rejected at phase setup. The eager-vv
// stub's plan is [KickDrift, ForceEval, KickHalf] with no BarostatPoint,
// so pairing it with a c-rescale barostat trips the placement guard.
#[test]
fn barostat_without_placement_marker_is_rejected() {
    let dir = tmp("barostat_placement_missing");
    write_lattice_init(&dir, 9, 4.4e-10);
    let extra = "[phase.barostat]\nkind = \"c-rescale\"\n\
                 pressure = 1.0\ntemperature = 30.0\ntau = 100.0\n\
                 compressibility = 0.01\nseed = 1\n";
    std::fs::write(
        dir.join("sim.in.toml"),
        lj_config(3, "kind = \"eager-vv\"", extra, false),
    )
    .unwrap();
    let kick_execs = Arc::new(AtomicU64::new(0));
    let mut registries = Registries::with_builtins();
    registries.register_integrator(Box::new(EagerVvBuilder { kick_execs }));
    let err = run_simulation_with_registries(&dir.join("sim.in.toml"), &registries)
        .expect_err("expected BarostatPlacementMissing");
    match err {
        RunnerError::BarostatPlacementMissing { integrator } => {
            assert_eq!(integrator, "eager-vv");
        }
        other => panic!("expected BarostatPlacementMissing, got {other:?}"),
    }
}

// rq-79b8a246
#[test]
fn non_participating_thermostat_forces_integrator_out() {
    let dir = tmp("eager_thermostat");
    write_lattice_init(&dir, 9, 4.4e-10);
    let extra = "[phase.thermostat]\nkind = \"eager-thermostat\"\n";
    std::fs::write(
        dir.join("sim.in.toml"),
        lj_config(3, "kind = \"velocity-verlet\"\nlossless = false", extra, false),
    )
    .unwrap();
    let post = Arc::new(AtomicU64::new(0));
    let mut registries = Registries::with_builtins();
    registries.register_thermostat(Box::new(EagerThermostatBuilder { post: post.clone() }));
    run_simulation_with_registries(&dir.join("sim.in.toml"), &registries).unwrap();
    let counts = timings_counts(&dir.join("sim.out.run.timings"));
    // No composed kernel: the plan walk executed VV's trailing kick.
    assert_eq!(stage_count(&counts, "jit_composed_post_force"), 0);
    assert_eq!(stage_count(&counts, "vv_kick"), 3);
    // The thermostat's own apply_post ran after the walk each step.
    assert_eq!(post.load(Ordering::SeqCst), 3);
}

// rq-7bd422a5 — a non-coupling step keeps kick fusion; a coupling step
// does not. With CSVR at coupling_interval = 4 over 8 steps, the six
// non-coupling steps (1,2,3,5,6,7) launch the composed kernel with the
// integrator's kick fused, and the two coupling steps (4,8) bypass the
// composed kernel: the trailing kick runs in the plan walk (standalone
// vv_kick) and CSVR's apply_post reduces the full-step KE and rescales.
#[test]
fn coupling_interval_fuses_non_coupling_steps_only() {
    let dir = tmp("coupling_interval");
    write_lattice_init(&dir, 9, 4.4e-10);
    let extra = "[phase.thermostat]\nkind = \"csvr\"\ntemperature = 30.0\ntau = 1.0e-13\nseed = 3\n\
                 coupling_interval = 4\n";
    std::fs::write(
        dir.join("sim.in.toml"),
        lj_config(8, "kind = \"velocity-verlet\"\nlossless = false", extra, false),
    )
    .unwrap();
    run_simulation(&dir.join("sim.in.toml")).unwrap();
    let counts = timings_counts(&dir.join("sim.out.run.timings"));
    // 6 non-coupling steps fuse the kick into the composed kernel.
    assert_eq!(stage_count(&counts, "jit_composed_post_force"), 6);
    // 2 coupling steps run the trailing kick standalone in the walk.
    assert_eq!(stage_count(&counts, "vv_kick"), 2);
    // The thermostat couples only on the 2 coupling steps.
    assert_eq!(stage_count(&counts, "csvr_rescale_velocities"), 2);
    assert_eq!(stage_count(&counts, "csvr_sample_and_factor"), 2);
}

// rq-609dc377
#[test]
fn every_builtin_slot_kind_exposes_a_post_force_fragment() {
    let gpu = init_device().unwrap();
    let integrators: &[(&str, &str)] = &[
        ("velocity-verlet", "lossless = false\n"),
        ("langevin-baoab", "friction = 1.0e12\ntemperature = 300.0\nseed = 1\n"),
        (
            "mtk-npt",
            "temperature = 9.5e-4\npressure = 3.4e-9\ntau_t = 4.1e3\ntau_p = 4.1e4\n\
             chain_length = 3\nyoshida_order = 3\nn_resp = 1\n",
        ),
        ("respa", "n_inner = 2\n"),
    ];
    for (kind, params) in integrators {
        let built = IntegratorRegistry::with_builtins()
            .build(&SlotConfig::from_params_str(kind, params), &gpu, 4, 0)
            .unwrap();
        assert!(
            built.post_force_per_particle().is_some(),
            "builtin integrator `{kind}` lost its fused post-force path"
        );
    }
    let thermostats: &[(&str, &str)] = &[
        (
            "nose-hoover-chain",
            "temperature = 9.5e-4\ntau = 4.1e3\nchain_length = 3\nyoshida_order = 3\nn_resp = 1\n",
        ),
        ("csvr", "temperature = 9.5e-4\ntau = 4.1e3\nseed = 1\n"),
        ("andersen", "temperature = 9.5e-4\ncollision_rate = 1.0e-4\nseed = 1\n"),
        ("berendsen", "temperature = 9.5e-4\ntau = 4.1e3\n"),
    ];
    for (kind, params) in thermostats {
        // rq-609dc377 — a thermostat carries no post_force_per_particle
        // accessor: its rescale follows a full-step kinetic-energy
        // reduction (a fusion barrier) and runs eagerly in apply_post, so
        // it is never part of the composed kernel. Confirm each still
        // builds.
        let _built = ThermostatRegistry::with_builtins()
            .build_optional(Some(&SlotConfig::from_params_str(kind, params)), &gpu, 4, 0)
            .unwrap()
            .unwrap();
    }
    // Per-step barostats (the Monte-Carlo barostat is periodic and
    // exempt from the composed ordering).
    let barostats: &[(&str, &str)] = &[
        ("berendsen", "pressure = 3.4e-9\ntau = 4.1e4\ncompressibility = 1.0e5\n"),
        (
            "c-rescale",
            "pressure = 3.4e-9\ntemperature = 9.5e-4\ntau = 4.1e4\ncompressibility = 1.0e5\nseed = 1\n",
        ),
    ];
    for (kind, params) in barostats {
        let built = BarostatRegistry::with_builtins()
            .build_optional(Some(&SlotConfig::from_params_str(kind, params)), &gpu, 4, 0)
            .unwrap()
            .unwrap();
        assert!(
            built.post_force_per_particle().is_some(),
            "builtin barostat `{kind}` lost its fused post-force path"
        );
    }
}

// rq-5e904c5d — a thermostat contributes no composed fragment and always
// runs eagerly, so a fragment-less per-step barostat paired with a
// thermostat has a valid execution order: there is no
// PostForceTopologyUnsatisfiable rejection and the phase runs to
// completion.
#[test] // rq-5e904c5d
fn fragmentless_per_step_barostat_with_builtin_thermostat_runs() {
    let dir = tmp("topology");
    write_lattice_init(&dir, 9, 4.4e-10);
    let extra = "[phase.thermostat]\nkind = \"csvr\"\ntemperature = 30.0\ntau = 1.0e-13\nseed = 3\n\
                 [phase.barostat]\nkind = \"fragmentless-barostat\"\n";
    std::fs::write(
        dir.join("sim.in.toml"),
        lj_config(2, "kind = \"velocity-verlet\"\nlossless = false", extra, false),
    )
    .unwrap();
    let mut registries = Registries::with_builtins();
    registries.register_barostat(Box::new(FragmentlessBarostatBuilder));
    run_simulation_with_registries(&dir.join("sim.in.toml"), &registries)
        .expect("thermostat + fragment-less per-step barostat should run without rejection");
}

// rq-3d84a5b8 — graph eligibility is preserved for a non-participating
// integrator; the standalone trailing kick captures like any other
// launch, so the graph-mode trajectory is byte-identical to the eager
// one.
#[test]
fn non_participating_slot_does_not_affect_graph_eligibility() {
    let mut trajectories = Vec::new();
    for (name, graphs) in [("graphs_on", true), ("graphs_off", false)] {
        let dir = tmp(name);
        write_lattice_init(&dir, 9, 4.4e-10);
        std::fs::write(
            dir.join("sim.in.toml"),
            lj_config(12, "kind = \"eager-vv\"", "", graphs),
        )
        .unwrap();
        let mut registries = Registries::with_builtins();
        registries.register_integrator(Box::new(EagerVvBuilder {
            kick_execs: Arc::new(AtomicU64::new(0)),
        }));
        run_simulation_with_registries(&dir.join("sim.in.toml"), &registries).unwrap();
        trajectories.push(std::fs::read(dir.join("sim.out.run.xyz")).unwrap());
    }
    assert_eq!(
        trajectories[0], trajectories[1],
        "graph-mode trajectory differs from eager for a non-participating integrator"
    );
}

// =================================================================
// Thermostat coupling cadence + full-step kinetic energy
// =================================================================

// A thermostat that only *records* (dt, KE) at each apply_pre / apply_post
// call and never modifies velocities. Because it does not rescale, a run
// using it has the same dynamics as an un-thermostatted VV run, so the KE
// it observes at apply_post is the run's post-trailing-kick (full-step)
// kinetic energy.
#[derive(Debug)]
struct RecordingThermostat {
    pre: Arc<Mutex<Vec<(Real, f64)>>>,
    post: Arc<Mutex<Vec<(Real, f64)>>>,
    scratch: CudaSlice<Real>,
}

impl Thermostat for RecordingThermostat {
    fn apply_pre(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: Real,
        _t: &mut Timings,
    ) -> Result<(), ThermostatError> {
        let ke = compute_kinetic_energy(buffers, &mut self.scratch)
            .map_err(ThermostatError::Gpu)? as f64;
        self.pre.lock().unwrap().push((dt, ke));
        Ok(())
    }
    fn apply_post(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: Real,
        _t: &mut Timings,
    ) -> Result<(), ThermostatError> {
        let ke = compute_kinetic_energy(buffers, &mut self.scratch)
            .map_err(ThermostatError::Gpu)? as f64;
        self.post.lock().unwrap().push((dt, ke));
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct RecordingThermostatBuilder {
    pre: Arc<Mutex<Vec<(Real, f64)>>>,
    post: Arc<Mutex<Vec<(Real, f64)>>>,
}

impl KindedBuilder for RecordingThermostatBuilder {
    fn kind_name(&self) -> &'static str {
        "recording-thermostat"
    }
}

impl ThermostatBuilder for RecordingThermostatBuilder {
    fn validate_params(&self, _params: &toml::Value) -> Result<(), ConfigError> {
        Ok(())
    }
    fn graph_compatible(&self, _params: &toml::Value) -> bool {
        false
    }
    fn build(
        &self,
        gpu: &GpuContext,
        _particle_count: usize,
        _n_constraints: usize,
        _params: &toml::Value,
    ) -> Result<Box<dyn Thermostat>, ThermostatError> {
        let scratch = gpu.device.alloc_zeros::<Real>(1).map_err(|e| {
            ThermostatError::Gpu(heddle_md::gpu::GpuError(e))
        })?;
        Ok(Box::new(RecordingThermostat {
            pre: self.pre.clone(),
            post: self.post.clone(),
            scratch,
        }))
    }
}

fn run_with_recording_thermostat(
    dir_name: &str,
    n_steps: u64,
    coupling_interval: u32,
) -> (Vec<(Real, f64)>, Vec<(Real, f64)>) {
    let dir = tmp(dir_name);
    write_lattice_init(&dir, 9, 4.4e-10);
    let extra = format!(
        "[phase.thermostat]\nkind = \"recording-thermostat\"\ncoupling_interval = {coupling_interval}\n"
    );
    std::fs::write(
        dir.join("sim.in.toml"),
        lj_config(n_steps, "kind = \"velocity-verlet\"\nlossless = false", &extra, false),
    )
    .unwrap();
    let pre = Arc::new(Mutex::new(Vec::new()));
    let post = Arc::new(Mutex::new(Vec::new()));
    let mut registries = Registries::with_builtins();
    registries.register_thermostat(Box::new(RecordingThermostatBuilder {
        pre: pre.clone(),
        post: post.clone(),
    }));
    run_simulation_with_registries(&dir.join("sim.in.toml"), &registries).unwrap();
    let pre_out = pre.lock().unwrap().clone();
    let post_out = post.lock().unwrap().clone();
    (pre_out, post_out)
}

// rq-aa624d38 — apply_post observes the full-step (post-trailing-kick)
// kinetic energy. With coupling_interval = 1 the call order is
// pre[1], post[1], pre[2], post[2], ...; the recorder never rescales, so
// the KE apply_post[k] observed equals the KE apply_pre[k+1] observes at
// the next step boundary (nothing changes the velocities between them).
// If apply_post ran before the trailing kick, it would observe the
// half-step KE and the two would differ.
#[test]
fn thermostat_apply_post_observes_full_step_kinetic_energy() {
    let (pre, post) = run_with_recording_thermostat("fullstep_ke", 3, 1);
    assert_eq!(pre.len(), 3);
    assert_eq!(post.len(), 3);
    // Non-trivial dynamics: the trailing kick changes KE within a step.
    assert!(
        (pre[0].1 - post[0].1).abs() > 0.0,
        "the trailing kick should change KE within the step"
    );
    for k in 0..2 {
        let post_ke = post[k].1;
        let next_pre_ke = pre[k + 1].1;
        let rel = (post_ke - next_pre_ke).abs() / next_pre_ke.abs().max(1.0e-30);
        assert!(
            rel < 1.0e-6,
            "apply_post[{k}] KE {post_ke} != apply_pre[{}] KE {next_pre_ke} \
             (apply_post is not seeing the full-step velocities)",
            k + 1
        );
    }
}

// rq-f4d73396 — the default coupling interval (1) couples every step,
// each call receiving the base dt.
#[test]
fn thermostat_default_interval_couples_every_step() {
    let (pre, post) = run_with_recording_thermostat("couple_every", 4, 1);
    assert_eq!(post.len(), 4, "apply_post should run on every step");
    let base_dt = post[0].0;
    for (dt, _) in post.iter().chain(pre.iter()) {
        assert_eq!(*dt, base_dt, "each coupling call receives the base dt");
    }
}

// rq-6ec9d751 rq-76963898 — with coupling_interval = 2 over 4 steps the
// thermostat couples only on the interval-boundary steps (2, 4), and each
// coupling call receives the effective timestep 2 * base_dt.
#[test]
fn thermostat_interval_gates_coupling_and_scales_dt() {
    // base_dt = the per-step dt used by lj_config (2.0e-15 s in atomic
    // units); read it back from a unit-interval run for comparison.
    let (_, post_unit) = run_with_recording_thermostat("interval_unit", 2, 1);
    let base_dt = post_unit[0].0;

    let (pre, post) = run_with_recording_thermostat("interval_two", 4, 2);
    // 4 steps, interval 2 -> coupling on steps 2 and 4 only.
    assert_eq!(post.len(), 2, "apply_post should run on 2 of 4 steps");
    assert_eq!(pre.len(), 2);
    for (dt, _) in post.iter().chain(pre.iter()) {
        let rel = (*dt as f64 - 2.0 * base_dt as f64).abs() / (2.0 * base_dt as f64);
        assert!(rel < 1.0e-6, "coupling dt {dt} should be 2 * base_dt {base_dt}");
    }
}

// rq-f4d73396 rq-79b8a246 — a built-in thermostat's rescale runs as a
// standalone launch every step (never in the composed kernel), and with
// coupling every step no composed kernel is launched at all.
#[test]
fn builtin_thermostat_rescale_is_standalone_every_step() {
    let dir = tmp("standalone_rescale");
    write_lattice_init(&dir, 9, 4.4e-10);
    let extra = "[phase.thermostat]\nkind = \"csvr\"\ntemperature = 30.0\ntau = 1.0e-13\nseed = 3\n";
    std::fs::write(
        dir.join("sim.in.toml"),
        lj_config(3, "kind = \"velocity-verlet\"\nlossless = false", extra, false),
    )
    .unwrap();
    run_simulation(&dir.join("sim.in.toml")).unwrap();
    let counts = timings_counts(&dir.join("sim.out.run.timings"));
    assert_eq!(stage_count(&counts, "csvr_rescale_velocities"), 3);
    assert_eq!(stage_count(&counts, "vv_kick"), 3);
    assert_eq!(stage_count(&counts, "jit_composed_post_force"), 0);
}

// rq-dce6f4cf rq-49f6bbfb — a thermostatted phase is graph-ineligible and
// runs on the per-step path even when graphs are enabled: the thermostat
// still couples (its standalone rescale runs every step).
#[test]
fn thermostatted_phase_runs_per_step_even_with_graphs_enabled() {
    let dir = tmp("thermostat_graphs_on");
    write_lattice_init(&dir, 9, 4.4e-10);
    let extra = "[phase.thermostat]\nkind = \"csvr\"\ntemperature = 30.0\ntau = 1.0e-13\nseed = 3\n";
    std::fs::write(
        dir.join("sim.in.toml"),
        lj_config(4, "kind = \"velocity-verlet\"\nlossless = false", extra, true),
    )
    .unwrap();
    run_simulation(&dir.join("sim.in.toml")).unwrap();
    let counts = timings_counts(&dir.join("sim.out.run.timings"));
    // Coupling every step: the thermostat's standalone rescale ran each
    // step, which only happens on the per-step path.
    assert_eq!(stage_count(&counts, "csvr_rescale_velocities"), 4);
    assert_eq!(stage_count(&counts, "jit_composed_post_force"), 0);
}

// rq-b85a38d6 — a thermostat + per-step barostat together: every step is
// a coupling step (default interval), so the composed kernel is bypassed
// and the barostat's per-particle position rescale runs eagerly as a
// standalone launch, after the thermostat rescale.
#[test]
fn thermostat_and_per_step_barostat_rescale_eagerly_on_coupling_steps() {
    let dir = tmp("thermostat_barostat_eager");
    write_lattice_init(&dir, 9, 4.4e-10);
    let extra = "[phase.thermostat]\nkind = \"csvr\"\ntemperature = 30.0\ntau = 1.0e-13\nseed = 3\n\
                 [phase.barostat]\nkind = \"c-rescale\"\npressure = 1.0\ntemperature = 30.0\n\
                 tau = 100.0\ncompressibility = 0.01\nseed = 1\n";
    std::fs::write(
        dir.join("sim.in.toml"),
        lj_config(3, "kind = \"velocity-verlet\"\nlossless = false", extra, false),
    )
    .unwrap();
    run_simulation(&dir.join("sim.in.toml")).unwrap();
    let counts = timings_counts(&dir.join("sim.out.run.timings"));
    // Composed kernel bypassed on every (coupling) step.
    assert_eq!(stage_count(&counts, "jit_composed_post_force"), 0);
    // Thermostat rescale and barostat position rescale both ran eagerly
    // every step.
    assert_eq!(stage_count(&counts, "csvr_rescale_velocities"), 3);
    assert_eq!(stage_count(&counts, "c_rescale_barostat_rescale_positions"), 3);
    // The trailing kick ran standalone in the walk.
    assert_eq!(stage_count(&counts, "vv_kick"), 3);
}
