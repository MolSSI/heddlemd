//! Post-force tail and thermostat-coupling integration tests.
//! Covers the plan-declared barostat-placement guard and graph-mode
//! scenarios in `rqm/integration/framework.md` and `rqm/cuda-graphs.md`,
//! plus the full-step kinetic-energy coupling cadence.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::CudaSlice;
use heddle_md::gpu::{GpuContext, ParticleBuffers, compute_kinetic_energy};
use heddle_md::integrator::{
    Integrator, IntegratorBuilder, IntegratorError, KickSource, StepPlan, SubStep, Thermostat,
    ThermostatBuilder, ThermostatError,
};
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

/// A custom integrator whose plan is velocity Verlet's shape; execute
/// performs real kicks via the vv kernels, so its dynamics are identical
/// to the built-in velocity Verlet. Used to exercise the plan-walk and
/// graph paths with a non-built-in integrator.
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

// =================================================================
// Scenarios
// =================================================================

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

// rq-f4d73396 — a built-in thermostat's rescale runs as a standalone
// launch every step, following the trailing velocity-Verlet kick.
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
}

// rq-49f6bbfb — a thermostat with the default coupling_interval (1) couples
// every step, so no non-coupling step exists to capture; the phase runs
// entirely on the per-step launch path even with graphs enabled, and the
// thermostat's standalone rescale runs every step.
#[test]
fn thermostat_coupling_every_step_runs_per_step_with_graphs_enabled() {
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
}

// rq-26c9b8cb rq-6f09d7e3 rq-91c02dd8 — the slot-eligibility table: csvr,
// berendsen, and andersen are cadence-inert (their per-step work fires only
// on coupling steps through the runner) and report graph_compatible == true;
// nose-hoover-chain integrates its Yoshida chain every step and reports
// false. A false thermostat makes its phase graph-ineligible regardless of
// coupling_interval.
#[test]
fn thermostat_slot_graph_compatibility_table() {
    let regs = Registries::with_builtins();
    let empty = toml::Value::Table(Default::default());
    let compat = |kind: &str| -> bool {
        regs.thermostats
            .lookup(kind)
            .expect("builtin thermostat")
            .graph_compatible(&empty)
    };
    assert!(compat("csvr"), "csvr is cadence-inert");
    assert!(compat("berendsen"), "berendsen is cadence-inert");
    assert!(compat("andersen"), "andersen resamples only on coupling steps");
    assert!(
        !compat("nose-hoover-chain"),
        "nose-hoover-chain integrates its chain every step"
    );
}

// rq-9c1eb803 — the hybrid graph path (non-coupling steps replayed from a
// captured graph, coupling steps run whole on the per-step launch path)
// produces a byte-identical trajectory to the fully per-step path for the
// same csvr-thermostatted phase with coupling_interval > 1. n_steps = 20
// exceeds the 8-step graph-timing calibration prefix, so the batched graph
// loop runs (steps 9..20) with coupling break-outs at 10, 15, 20.
#[test]
fn hybrid_graph_matches_per_step_byte_for_byte() {
    let extra = "[phase.thermostat]\nkind = \"csvr\"\ntemperature = 30.0\ntau = 1.0e-13\n\
                 seed = 3\ncoupling_interval = 5\n";
    let run = |graphs: bool, tag: &str| -> Vec<u8> {
        let dir = tmp(tag);
        write_lattice_init(&dir, 9, 4.4e-10);
        std::fs::write(
            dir.join("sim.in.toml"),
            lj_config(20, "kind = \"velocity-verlet\"\nlossless = false", extra, graphs),
        )
        .unwrap();
        run_simulation(&dir.join("sim.in.toml")).unwrap();
        std::fs::read(dir.join("sim.out.run.xyz")).unwrap()
    };
    let hybrid = run(true, "hybrid_bitexact_graph");
    let per_step = run(false, "hybrid_bitexact_perstep");
    assert_eq!(
        hybrid, per_step,
        "hybrid graph-replay trajectory differs from the fully per-step trajectory"
    );
}

// rq-b85a38d6 — a thermostat + per-step barostat together: every step is
// a coupling step (default interval), so the barostat's per-particle
// position rescale runs as a standalone launch, after the thermostat
// rescale.
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
    // Thermostat rescale and barostat position rescale both ran as
    // standalone launches every step.
    assert_eq!(stage_count(&counts, "csvr_rescale_velocities"), 3);
    assert_eq!(stage_count(&counts, "c_rescale_barostat_rescale_positions"), 3);
    // The trailing kick ran standalone in the walk.
    assert_eq!(stage_count(&counts, "vv_kick"), 3);
}
