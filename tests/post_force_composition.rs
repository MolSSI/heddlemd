//! Post-force composed-kernel participation and fallback tests.
//! Implements the suffix-closed subset scenarios in
//! `rqm/integration/jit-composed-post-force.md` and the graph-mode
//! scenario in `rqm/cuda-graphs.md`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use heddle_md::gpu::{GpuContext, ParticleBuffers, init_device};
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

// rq-dcd0d421
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
    // The composed kernel still ran (covering only the thermostat).
    // (The stub's kick launches bypass the timings stages, so the
    // execute-count above is the kick-dispatch evidence.)
    let counts = timings_counts(&dir.join("sim.out.run.timings"));
    assert_eq!(stage_count(&counts, "jit_composed_post_force"), 3);
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

// rq-7bd422a5
#[test]
fn fully_participating_configuration_keeps_full_fusion() {
    let dir = tmp("full_fusion");
    write_lattice_init(&dir, 9, 4.4e-10);
    let extra = "[phase.thermostat]\nkind = \"csvr\"\ntemperature = 30.0\ntau = 1.0e-13\nseed = 3\n";
    std::fs::write(
        dir.join("sim.in.toml"),
        lj_config(3, "kind = \"velocity-verlet\"\nlossless = false", extra, false),
    )
    .unwrap();
    run_simulation(&dir.join("sim.in.toml")).unwrap();
    let counts = timings_counts(&dir.join("sim.out.run.timings"));
    // Composed kernel covers the integrator and the thermostat: the
    // trailing kick is skipped from the plan walk (no standalone
    // vv_kick launches).
    assert_eq!(stage_count(&counts, "jit_composed_post_force"), 3);
    assert_eq!(stage_count(&counts, "vv_kick"), 0);
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
        let built = ThermostatRegistry::with_builtins()
            .build_optional(Some(&SlotConfig::from_params_str(kind, params)), &gpu, 4, 0)
            .unwrap()
            .unwrap();
        assert!(
            built.post_force_per_particle().is_some(),
            "builtin thermostat `{kind}` lost its fused post-force path"
        );
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

// Topology rejection: a fragment-bearing thermostat suffix-excluded by
// a fragment-less per-step barostat has no eager fallback.
#[test] // rq-5e904c5d
fn fragmentless_per_step_barostat_with_builtin_thermostat_is_rejected() {
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
    let err = run_simulation_with_registries(&dir.join("sim.in.toml"), &registries).unwrap_err();
    assert!(
        matches!(err, RunnerError::PostForceTopologyUnsatisfiable),
        "unexpected error: {err:?}"
    );
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
