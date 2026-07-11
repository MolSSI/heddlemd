use heddle_md::integrator::IntegratorStepExt;
use heddle_md::registry::KindedBuilder;
use heddle_md::forces::{BondList, DihedralList, ExclusionList, ForceField, PotentialRegistry};
use heddle_md::gpu::{GpuContext, ParticleBuffers, init_device};
use heddle_md::integrator::{
    Integrator, IntegratorBuilder, IntegratorError, IntegratorRegistry, LangevinBaoabBuilder,
    LangevinBaoabState, VelocityVerletBuilder,
};
use heddle_md::io::SlotConfig;
use heddle_md::io::config::NeighborListConfig;
use heddle_md::pbc::SimulationBox;
use heddle_md::state::ParticleState;
use heddle_md::timings::{KernelStage, Timings};
use heddle_md::precision::Real;

fn small_state(n: usize) -> ParticleState {
    let pos: Vec<Real> = (0..n).map(|i| (i as Real) * 0.1).collect();
    let zero = vec![0.0; n];
    let m = vec![1.0; n];
    ParticleState::new(
        pos,
        zero.clone(),
        zero.clone(),
        zero.clone(),
        zero.clone(),
        zero,
        m,
        vec![0.0; n],
        vec![0u32; n],
        None,
            None,
    )
    .unwrap()
}

fn vv_kind(lossless: bool) -> SlotConfig {
    SlotConfig::from_params_str(
        "velocity-verlet",
        &format!("lossless = {lossless}\n"),
    )
}

fn langevin_kind(seed: u64) -> SlotConfig {
    SlotConfig::from_params_str(
        "langevin-baoab",
        &format!("friction = 1.0e12\ntemperature = 300.0\nseed = {seed}\n"),
    )
}

fn box_10(gpu: &heddle_md::gpu::GpuContext) -> SimulationBox {
    SimulationBox::new(&gpu.device, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0).unwrap()
}

fn empty_force_field(gpu: &GpuContext, n: usize) -> ForceField {
    ForceField::new(
        &PotentialRegistry::with_builtins(),
        gpu,
        n,
        &box_10(&gpu),
        &[],
        &[],
        &[],
        &[],
        &[],
        None,
        &[],
        &BondList::empty(n),
        &heddle_md::forces::AngleList::empty(0),
        &DihedralList::empty(0),
        &ExclusionList::empty(n),
        &NeighborListConfig::AllPairs,
    )
    .unwrap()
}

// rq-444903e2
#[test]
fn construct_vv_lossy_via_registry() {
    let gpu = init_device().unwrap();
    let registry = IntegratorRegistry::with_builtins();
    let _integrator = registry.build(&vv_kind(false), &gpu, 4, 0).unwrap();
}

// rq-7d4c470a
// Lossless mode is only available in the default (f32) build.
#[cfg(not(feature = "f64"))]
#[test]
fn construct_vv_lossless_via_registry() {
    let gpu = init_device().unwrap();
    let registry = IntegratorRegistry::with_builtins();
    let state = small_state(4);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = empty_force_field(&gpu, 4);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integrator = registry.build(&vv_kind(true), &gpu, 4, 0).unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, 0.1, &mut timings)
        .unwrap();
    // The lossless build is observable through the lossless KernelStage labels.
    let report = timings.finalize().unwrap();
    let names: Vec<&str> = report.stages.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"vv_kick_drift_lossless"));
    assert!(names.contains(&"vv_kick_lossless"));
}

// rq-706c4b80
#[test]
fn construct_langevin_via_registry() {
    let gpu = init_device().unwrap();
    let registry = IntegratorRegistry::with_builtins();
    let _integrator = registry.build(&langevin_kind(42), &gpu, 4, 0).unwrap();
}

// rq-b44769f1
#[test]
fn construct_with_particle_count_zero() {
    let gpu = init_device().unwrap();
    let registry = IntegratorRegistry::with_builtins();
    let _integrator = registry.build(&vv_kind(true), &gpu, 0, 0).unwrap();
}

// rq-5711d6ce
#[test]
fn registry_without_matching_builder_reports_unknown_kind() {
    let gpu = init_device().unwrap();
    let registry = IntegratorRegistry::new();
    let err = registry.build(&vv_kind(false), &gpu, 4, 0).unwrap_err();
    match err {
        IntegratorError::UnknownKind(name) => assert_eq!(name, "velocity-verlet"),
        other => panic!("expected UnknownKind, got {other:?}"),
    }
}

#[derive(Debug, Clone)]
struct StubBuilder;

#[derive(Debug)]
struct StubIntegrator;

    impl heddle_md::registry::KindedBuilder for StubBuilder {
    fn kind_name(&self) -> &'static str {
        // Use "velocity-verlet" so a real IntegratorKind can route to us; the
        // built-in builder is added later and therefore shadowed by the stub
        // when the stub is first in the builders Vec.
        "velocity-verlet"
    }    }

impl IntegratorBuilder for StubBuilder {
    fn validate_params(&self, _params: &toml::Value) -> Result<(), heddle_md::io::ConfigError> {
        Ok(())
    }
    fn build(
        &self,
        _gpu: &GpuContext,
        _particle_count: usize,
        _n_constraints: usize,
        _params: &toml::Value,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        Ok(Box::new(StubIntegrator))
    }
}

impl Integrator for StubIntegrator {
    fn plan(&self, _dt: Real) -> heddle_md::integrator::StepPlan {
        heddle_md::integrator::StepPlan::empty()
    }

    fn execute(
        &mut self,
        _substep: &heddle_md::integrator::SubStep,
        _buffers: &mut ParticleBuffers,
        _sim_box: &mut SimulationBox,
        _timings: &mut Timings,
    ) -> Result<(), IntegratorError> {
        Ok(())
    }
}

// rq-0d7ebeb6
#[test]
fn custom_builder_registered_takes_priority_over_builtin() {
    let gpu = init_device().unwrap();
    let mut registry = IntegratorRegistry::new();
    registry.register(Box::new(StubBuilder));
    registry.register(Box::new(VelocityVerletBuilder));
    registry.register(Box::new(LangevinBaoabBuilder));
    // Stub appears first, so velocity-verlet routes to it.
    let state = small_state(4);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = empty_force_field(&gpu, 4);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integrator = registry.build(&vv_kind(false), &gpu, 4, 0).unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, 0.1, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    // Stub launches no kernels, so neither vv_kick_drift nor vv_kick should
    // appear in the timings report.
    let names: Vec<&str> = report.stages.iter().map(|s| s.name.as_str()).collect();
    assert!(!names.contains(&"vv_kick_drift"));
    assert!(!names.contains(&"vv_kick"));
}

// rq-8fd4e3bf
#[test]
fn step_on_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = ParticleState::new(
        Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        vec![0.0; 0],
        Vec::new(), None,
            None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = empty_force_field(&gpu, 0);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&vv_kind(false), &gpu, 0, 0)
        .unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, 0.1, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    assert!(report.stages.is_empty());
}

// rq-d3bd619e
#[test]
fn vv_step_launches_kick_drift_force_and_kick() {
    let gpu = init_device().unwrap();
    // Build a state with non-zero velocities so the drift visibly moves
    // positions during step() even with zero forces.
    let n = 4;
    let mut state = small_state(n);
    state.velocities_x = (0..n).map(|i| 0.5 + i as Real * 0.1).collect();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&vv_kind(false), &gpu, n, 0)
        .unwrap();
    let snap_positions = state.positions_x.clone();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, 0.1, &mut timings)
        .unwrap();
    let mut after = state.clone();
    after.download_from(&buffers).unwrap();
    assert_ne!(after.positions_x, snap_positions);
    let report = timings.finalize().unwrap();
    let kd = report
        .stages
        .iter()
        .find(|s| s.name == "vv_kick_drift")
        .expect("vv_kick_drift launched");
    assert_eq!(kd.count, 1);
    let k = report
        .stages
        .iter()
        .find(|s| s.name == "vv_kick")
        .expect("vv_kick launched");
    assert_eq!(k.count, 1);
}

// rq-17def001
// Lossless mode is only available in the default (f32) build.
#[cfg(not(feature = "f64"))]
#[test]
fn lossless_vv_step_uses_lossless_kernels() {
    let gpu = init_device().unwrap();
    let state = small_state(4);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = empty_force_field(&gpu, 4);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&vv_kind(true), &gpu, 4, 0)
        .unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, 0.1, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    let names: Vec<&str> = report.stages.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"vv_kick_drift_lossless"));
    assert!(names.contains(&"vv_kick_lossless"));
    assert!(!names.contains(&"vv_kick_drift"));
    assert!(!names.contains(&"vv_kick"));
}

// rq-812e88d5
#[test]
fn integrator_owns_force_evaluation_inside_step() {
    // Wire up a real LJ slot so the force pipeline runs and we can confirm
    // KernelStage::LJ_PAIR_FORCE was triggered exactly once per step() call.
    use heddle_md::io::config::{PairInteractionConfig, ParticleTypeConfig};
    let gpu = init_device().unwrap();
    let state = small_state(4);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        4,
        &sim_box,
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, sigma: None, epsilon: None, charge: 0.0 }],
        &[PairInteractionConfig::lennard_jones(("Ar".to_string(), "Ar".to_string()), 1.0, 1.0, 1.0, Some(1.0))],
        &[],
        &[],
        &[],
        None,
        &[],
        &BondList::empty(4),
        &heddle_md::forces::AngleList::empty(0),
        &DihedralList::empty(0),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&vv_kind(false), &gpu, 4, 0)
        .unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, 0.001, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    let count = |name: &str| {
        report
            .stages
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.count)
            .unwrap_or(0)
    };
    assert_eq!(count("jit_composed_pair_force"), 1);
    assert_eq!(count("combine_class_totals"), 1);
}

// rq-009bbbdc
#[test]
fn two_consecutive_langevin_steps_produce_different_velocities() {
    let gpu = init_device().unwrap();
    let state = small_state(2);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = empty_force_field(&gpu, 2);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&langevin_kind(1), &gpu, 2, 0)
        .unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, 1.0e-15, &mut timings)
        .unwrap();
    let mut state_after_first = state.clone();
    state_after_first.download_from(&buffers).unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, 1.0e-15, &mut timings)
        .unwrap();
    let mut state_after_second = state.clone();
    state_after_second.download_from(&buffers).unwrap();
    assert_ne!(state_after_first.velocities_x, state_after_second.velocities_x);
}

// rq-1b0504e7
#[test]
fn two_independent_runs_byte_identical() {
    let gpu = init_device().unwrap();
    let state = small_state(4);

    let mut buffers_a = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut buffers_b = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box_a = box_10(&gpu);
    let mut sim_box_b = box_10(&gpu);
    let mut ff_a = empty_force_field(&gpu, 4);
    let mut ff_b = empty_force_field(&gpu, 4);
    let mut timings_a = Timings::new(&gpu).unwrap();
    let mut timings_b = Timings::new(&gpu).unwrap();
    let mut integrator_a = IntegratorRegistry::with_builtins()
        .build(&vv_kind(false), &gpu, 4, 0)
        .unwrap();
    let mut integrator_b = IntegratorRegistry::with_builtins()
        .build(&vv_kind(false), &gpu, 4, 0)
        .unwrap();

    for _ in 1..=10 {
        integrator_a
            .step(&mut buffers_a, &mut sim_box_a, &mut ff_a, 0.001, &mut timings_a)
            .unwrap();
        integrator_b
            .step(&mut buffers_b, &mut sim_box_b, &mut ff_b, 0.001, &mut timings_b)
            .unwrap();
    }

    let mut state_a = state.clone();
    let mut state_b = state.clone();
    state_a.download_from(&buffers_a).unwrap();
    state_b.download_from(&buffers_b).unwrap();
    assert_eq!(state_a.positions_x, state_b.positions_x);
    assert_eq!(state_a.velocities_x, state_b.velocities_x);
}

// rq-01784049
#[test]
fn langevin_draw_counter_starts_at_zero_and_increments_per_step() {
    let gpu = init_device().unwrap();
    let state = small_state(2);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = empty_force_field(&gpu, 2);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integrator = LangevinBaoabState::new(&gpu, 1.0e12, 300.0, 42, 0).unwrap();
    assert_eq!(integrator.draw_counter, 0);
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, 1.0e-15, &mut timings)
        .unwrap();
    assert_eq!(integrator.draw_counter, 1);
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, 1.0e-15, &mut timings)
        .unwrap();
    assert_eq!(integrator.draw_counter, 2);
}

// rq-e70ee09e
#[test]
fn langevin_states_at_same_draw_counter_and_seed_produce_identical_draws() {
    let gpu = init_device().unwrap();
    let state = small_state(4);
    let mut buffers_a = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut buffers_b = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box_a = box_10(&gpu);
    let mut sim_box_b = box_10(&gpu);
    let mut ff_a = empty_force_field(&gpu, 4);
    let mut ff_b = empty_force_field(&gpu, 4);
    let mut timings_a = Timings::new(&gpu).unwrap();
    let mut timings_b = Timings::new(&gpu).unwrap();
    let mut a = LangevinBaoabState::new(&gpu, 1.0e12, 300.0, 7, 5).unwrap();
    let mut b = LangevinBaoabState::new(&gpu, 1.0e12, 300.0, 7, 5).unwrap();
    a.step(&mut buffers_a, &mut sim_box_a, &mut ff_a, 1.0e-15, &mut timings_a)
        .unwrap();
    b.step(&mut buffers_b, &mut sim_box_b, &mut ff_b, 1.0e-15, &mut timings_b)
        .unwrap();
    let mut state_a = state.clone();
    let mut state_b = state.clone();
    state_a.download_from(&buffers_a).unwrap();
    state_b.download_from(&buffers_b).unwrap();
    assert_eq!(state_a.velocities_x, state_b.velocities_x);
    assert_eq!(state_a.velocities_y, state_b.velocities_y);
    assert_eq!(state_a.velocities_z, state_b.velocities_z);
    assert_eq!(a.draw_counter, 6);
    assert_eq!(b.draw_counter, 6);
}

// Silence the unused-name lint for the imported KernelStage variant set.
#[test]
fn _imports_used() {
    let _ = KernelStage::LANGEVIN_KICK_HALF;
}

// =====================================================================
// Orthogonal-slot framework (Thermostat + Barostat) coverage
// =====================================================================

use heddle_md::integrator::{
    BarostatError, BarostatRegistry, ThermostatError, ThermostatRegistry,
};
// rq-5711d6ce
#[test]
fn empty_integrator_registry_reports_unknown_kind() {
    let gpu = init_device().unwrap();
    let registry = IntegratorRegistry::new();
    let err = registry.build(&vv_kind(false), &gpu, 4, 0).unwrap_err();
    matches!(err, IntegratorError::UnknownKind(ref s) if s == "velocity-verlet");
}

// rq-0d7ebeb6: also covered by `custom_builder_registered_takes_priority_over_builtin` above.

// rq-353da04c — placeholder: NHC construction is covered in tests/integrators_nhc.rs
// rq-69d2c5f5 — placeholder: CSVR construction is covered in tests/integrators_csvr.rs
// rq-dc3c616a — placeholder: Andersen construction is covered in tests/integrators_andersen.rs
// rq-e8252f10 — placeholder: Berendsen construction is covered in tests/integrators_berendsen.rs

// Empty thermostat registry reports UnknownKind.
// rq-6dffb17f
#[test]
fn empty_thermostat_registry_reports_unknown_kind() {
    let gpu = init_device().unwrap();
    let registry = ThermostatRegistry::new();
    let kind = SlotConfig::from_params_str(
        "berendsen",
        "temperature = 300.0\ntau = 1.0e-13\n",
    );
    let err = registry
        .build_optional(Some(&kind), &gpu, 4, 0)
        .unwrap_err();
    matches!(err, ThermostatError::UnknownKind(ref s) if s == "berendsen");
}

// build_optional with None returns Ok(None) without consulting builders.
// rq-fb3f2189
#[test]
fn thermostat_build_optional_none_returns_none() {
    let gpu = init_device().unwrap();
    let registry = ThermostatRegistry::with_builtins();
    let result = registry.build_optional(None, &gpu, 4, 0).unwrap();
    assert!(result.is_none());
}

// BarostatRegistry::with_builtins() exposes the registered barostats.
// rq-386e3288
#[test]
fn barostat_with_builtins_contains_berendsen() {
    let registry = BarostatRegistry::with_builtins();
    assert!(
        registry
            .builders()
            .iter()
            .any(|b| b.kind_name() == "berendsen")
    );
}

// build_optional with None returns Ok(None) on the empty barostat registry.
// rq-82cdabba
#[test]
fn barostat_build_optional_none_returns_none() {
    let gpu = init_device().unwrap();
    let registry = BarostatRegistry::with_builtins();
    let result = registry.build_optional(None, &gpu, 4, 0).unwrap();
    assert!(result.is_none());
}

#[test]
fn barostat_error_unknown_kind_is_constructible() {
    // The empty enum prevents an actual `Some(BarostatKind)` call in safe
    // Rust, but `BarostatError::UnknownKind` is reachable from a custom
    // builder/kind combination if a downstream registers one. The compiler
    // smoke test below just confirms the variant exists with the expected
    // payload shape.
    let err = BarostatError::UnknownKind("hypothetical".to_string());
    assert_eq!(format!("{err}"), "unknown barostat kind `hypothetical`");
}

// VelocityVerlet does not own its thermostat; LangevinBaoab does.
#[test]
fn integrator_kind_owns_thermostat_matrix() {
    let vv = vv_kind(false);
    let lan = langevin_kind(0);
    let registry = IntegratorRegistry::with_builtins();
    let vv_b = registry.lookup(&vv.kind).expect("vv builder registered");
    let lan_b = registry.lookup(&lan.kind).expect("langevin builder registered");
    assert!(!vv_b.owns_thermostat(&vv.params));
    assert!(lan_b.owns_thermostat(&lan.params));
}

// Dispatch loop calls thermostat.apply_pre, integrator.step,
// thermostat.apply_post in that order. We use a recording wrapper that
// timestamps every trait call and asserts the recorded order.
#[derive(Debug, Default)]
struct CallLog {
    events: std::sync::Arc<std::sync::Mutex<Vec<&'static str>>>,
}

impl CallLog {
    fn record(&self, name: &'static str) {
        self.events.lock().unwrap().push(name);
    }
}

#[derive(Debug)]
struct RecordingIntegrator {
    log: CallLog,
}

impl Integrator for RecordingIntegrator {
    fn plan(&self, dt: Real) -> heddle_md::integrator::StepPlan {
        heddle_md::integrator::StepPlan {
            steps: vec![heddle_md::integrator::SubStep::KickDrift { dt, label: "rec", source: heddle_md::integrator::KickSource::Total }],
        }
    }

    fn execute(
        &mut self,
        _substep: &heddle_md::integrator::SubStep,
        _b: &mut ParticleBuffers,
        _sb: &mut SimulationBox,
        _t: &mut Timings,
    ) -> Result<(), IntegratorError> {
        self.log.record("execute");
        Ok(())
    }
}

#[derive(Debug)]
struct RecordingThermostat {
    log: CallLog,
}

impl heddle_md::integrator::Thermostat for RecordingThermostat {
    fn apply_pre(
        &mut self,
        _b: &mut ParticleBuffers,
        _dt: Real,
        _t: &mut Timings,
    ) -> Result<(), ThermostatError> {
        self.log.record("apply_pre");
        Ok(())
    }
    fn apply_post(
        &mut self,
        _b: &mut ParticleBuffers,
        _dt: Real,
        _t: &mut Timings,
    ) -> Result<(), ThermostatError> {
        self.log.record("apply_post");
        Ok(())
    }
}

// rq-0a6a97f6
#[test]
fn dispatch_loop_orders_apply_pre_step_apply_post() {
    let gpu = init_device().unwrap();
    let state = small_state(4);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = empty_force_field(&gpu, 4);
    let mut timings = Timings::new(&gpu).unwrap();
    let log = CallLog::default();
    let log_clone_a = CallLog { events: log.events.clone() };
    let log_clone_b = CallLog { events: log.events.clone() };
    let mut integ: Box<dyn Integrator> = Box::new(RecordingIntegrator { log: log_clone_a });
    let mut therm: Box<dyn heddle_md::integrator::Thermostat> =
        Box::new(RecordingThermostat { log: log_clone_b });

    therm
        .apply_pre(&mut buffers, 1.0e-15, &mut timings)
        .unwrap();
    integ
        .step(&mut buffers, &mut sim_box, &mut ff, 1.0e-15, &mut timings)
        .unwrap();
    therm
        .apply_post(&mut buffers, 1.0e-15, &mut timings)
        .unwrap();

    let events = log.events.lock().unwrap().clone();
    // The RecordingIntegrator's plan has one KickDrift sub-step; the
    // extension method walks the plan and the integrator's execute()
    // records the single "execute" event between the two thermostat
    // calls.
    assert_eq!(events, vec!["apply_pre", "execute", "apply_post"]);
}

// =============================================================================
// Plan/execute trait surface tests (introduced with the step-decomposition
// refactor, see rqm/integration/framework.md).
// =============================================================================

use heddle_md::integrator::{
    Barostat, BarostatPeriodicity, ConstraintError, ConstraintPhase,
    RunStepOptions, run_step, StepPlan, SubStep,
};

/// A configurable stub Integrator whose plan and execute() behaviour
/// is controlled by the test. Records every execute() invocation as
/// ("execute", substep variant name) into the supplied log.
#[derive(Debug)]
struct PlanStub {
    plan: StepPlan,
    log: CallLog,
}

impl Integrator for PlanStub {
    fn plan(&self, _dt: Real) -> StepPlan {
        self.plan.clone()
    }
    fn execute(
        &mut self,
        substep: &SubStep,
        _b: &mut ParticleBuffers,
        _sb: &mut SimulationBox,
        _t: &mut Timings,
    ) -> Result<(), IntegratorError> {
        self.log.record(match substep {
            SubStep::KickHalf { .. } => "exec_kick_half",
            SubStep::Drift { .. } => "exec_drift",
            SubStep::KickDrift { .. } => "exec_kick_drift",
            SubStep::ForceEval { .. } => "exec_force_eval",
            SubStep::ThermostatHalf { .. } => "exec_thermostat_half",
            SubStep::ConstraintPoint { .. } => "exec_constraint_point",
            SubStep::BarostatPoint { .. } => "exec_barostat_point",
            SubStep::Custom { .. } => "exec_custom",
        });
        if matches!(substep, SubStep::ForceEval { .. }) {
            return Err(IntegratorError::UnexpectedSubStep {
                variant: substep.variant_name(),
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
struct RecordingConstraint {
    log: CallLog,
}

impl heddle_md::integrator::Constraint for RecordingConstraint {
    fn apply_before_drift(
        &mut self,
        _b: &mut ParticleBuffers,
        _sb: &SimulationBox,
        _dt: Real,
        _t: &mut Timings,
    ) -> Result<(), ConstraintError> {
        self.log.record("before_drift");
        Ok(())
    }
    fn apply_after_drift(
        &mut self,
        _b: &mut ParticleBuffers,
        _sb: &SimulationBox,
        _dt: Real,
        _t: &mut Timings,
    ) -> Result<(), ConstraintError> {
        self.log.record("after_drift");
        Ok(())
    }
    fn apply_after_kick(
        &mut self,
        _b: &mut ParticleBuffers,
        _sb: &SimulationBox,
        _dt: Real,
        _t: &mut Timings,
    ) -> Result<(), ConstraintError> {
        self.log.record("after_kick");
        Ok(())
    }
}

/// Stub per-step (or periodic) barostat that records the `dt` of each
/// `apply` call. A periodic instance leaves `apply` a no-op (mirroring a
/// real periodic barostat, which couples through `apply_move`), so the
/// `BarostatPoint` marker is inert for it.
#[derive(Debug)]
struct RecordingBarostat {
    applies: std::sync::Arc<std::sync::Mutex<Vec<Real>>>,
    periodic: bool,
}

impl Barostat for RecordingBarostat {
    fn periodicity(&self) -> BarostatPeriodicity {
        if self.periodic {
            BarostatPeriodicity::EveryNSteps(1)
        } else {
            BarostatPeriodicity::EveryStep
        }
    }
    fn apply(
        &mut self,
        _b: &mut ParticleBuffers,
        _sb: &mut SimulationBox,
        dt: Real,
        _t: &mut Timings,
    ) -> Result<(), BarostatError> {
        if !self.periodic {
            self.applies.lock().unwrap().push(dt);
        }
        Ok(())
    }
}

/// Stub constraint that records the `dt` each hook receives, used to
/// verify a `ConstraintPoint` marker's own `dt` reaches the hook.
#[derive(Debug)]
struct DtRecordingConstraint {
    seen: std::sync::Arc<std::sync::Mutex<Vec<Real>>>,
}

impl heddle_md::integrator::Constraint for DtRecordingConstraint {
    fn apply_before_drift(
        &mut self,
        _b: &mut ParticleBuffers,
        _sb: &SimulationBox,
        dt: Real,
        _t: &mut Timings,
    ) -> Result<(), ConstraintError> {
        self.seen.lock().unwrap().push(dt);
        Ok(())
    }
    fn apply_after_drift(
        &mut self,
        _b: &mut ParticleBuffers,
        _sb: &SimulationBox,
        dt: Real,
        _t: &mut Timings,
    ) -> Result<(), ConstraintError> {
        self.seen.lock().unwrap().push(dt);
        Ok(())
    }
    fn apply_after_kick(
        &mut self,
        _b: &mut ParticleBuffers,
        _sb: &SimulationBox,
        dt: Real,
        _t: &mut Timings,
    ) -> Result<(), ConstraintError> {
        self.seen.lock().unwrap().push(dt);
        Ok(())
    }
}

/// Barostat that appends a fixed marker to a shared ordered log on each
/// `apply`, so a test can assert the cross-slot ordering of the
/// post-force tail (kick, barostat, velocity projection).
#[derive(Debug)]
struct OrderRecordingBarostat {
    log: CallLog,
}

impl Barostat for OrderRecordingBarostat {
    fn periodicity(&self) -> BarostatPeriodicity {
        BarostatPeriodicity::EveryStep
    }
    fn apply(
        &mut self,
        _b: &mut ParticleBuffers,
        _sb: &mut SimulationBox,
        _dt: Real,
        _t: &mut Timings,
    ) -> Result<(), BarostatError> {
        self.log.record("barostat_apply");
        Ok(())
    }
}

/// Thin wrapper over `run_step` (with `runner_needs_scalars: true` and
/// no thermostat coupling) for the constraint / barostat stub tests.
/// `run_step` walks the whole plan, so the terminal `BarostatPoint` and
/// the terminal velocity projection are dispatched by this one call —
/// the test needs no separate post-force tail.
// rq-277dbeb2
fn run_timestep(
    integrator: &mut dyn Integrator,
    buffers: &mut ParticleBuffers,
    sim_box: &mut SimulationBox,
    ff: &mut ForceField,
    constraint: Option<&mut dyn heddle_md::integrator::Constraint>,
    barostat: Option<&mut dyn Barostat>,
    dt: Real,
    timings: &mut Timings,
) -> Result<(), heddle_md::integrator::StepError> {
    run_step(
        integrator,
        buffers,
        sim_box,
        ff,
        constraint,
        None,
        barostat,
        dt,
        timings,
        RunStepOptions { runner_needs_scalars: true, ..Default::default() },
    )
}

fn fixture() -> (
    GpuContext,
    ParticleBuffers,
    SimulationBox,
    ForceField,
    Timings,
) {
    let gpu = init_device().unwrap();
    let state = small_state(4);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = box_10(&gpu);
    let ff = empty_force_field(&gpu, 4);
    let timings = Timings::new(&gpu).unwrap();
    (gpu, buffers, sim_box, ff, timings)
}

// rq-94a67d95
#[test]
fn plan_returns_same_shape_across_repeated_calls() {
    let registry = IntegratorRegistry::with_builtins();
    let gpu = init_device().unwrap();
    let integ = registry.build(&vv_kind(false), &gpu, 4, 0).unwrap();
    let p1 = integ.plan(0.1);
    let p2 = integ.plan(0.1);
    assert_eq!(p1.steps.len(), p2.steps.len());
    for (a, b) in p1.steps.iter().zip(p2.steps.iter()) {
        assert_eq!(a.variant_name(), b.variant_name());
    }
}

// rq-4300cafc
#[test]
fn plan_is_pure_does_not_launch_kernels_or_touch_buffers() {
    let registry = IntegratorRegistry::with_builtins();
    let (gpu, buffers, _sim_box, _ff, _timings) = fixture();
    let pre_x = buffers.download_positions().unwrap().0;
    let integ = registry.build(&vv_kind(false), &gpu, 4, 0).unwrap();
    let _plan = integ.plan(0.1);
    let post_x = buffers.download_positions().unwrap().0;
    assert_eq!(pre_x, post_x);
}

// rq-384ed838
#[test]
fn empty_plan_walks_as_a_noop() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan::empty(),
        log: CallLog { events: log.events.clone() },
    };
    stub.step(
        &mut buffers,
        &mut sim_box,
        &mut ff,
        0.1,
        &mut timings,
    )
    .unwrap();
    let events = log.events.lock().unwrap().clone();
    assert!(events.is_empty(), "expected no events, got {:?}", events);
}

// rq-07ead62b
#[test]
fn plan_with_multiple_force_evals_dispatches_each() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![
                SubStep::KickHalf { dt: 0.1, label: "k1", source: heddle_md::integrator::KickSource::Total },
                SubStep::Drift { dt: 0.1, label: "d1" },
                SubStep::ForceEval { class: None, level: Some(heddle_md::forces::AggregateLevel::ForcesAndScalars) },
                SubStep::KickHalf { dt: 0.1, label: "k2", source: heddle_md::integrator::KickSource::Total },
                SubStep::Drift { dt: 0.1, label: "d2" },
                SubStep::ForceEval { class: None, level: Some(heddle_md::forces::AggregateLevel::ForcesAndScalars) },
                SubStep::KickHalf { dt: 0.1, label: "k3", source: heddle_md::integrator::KickSource::Total },
            ],
        },
        log: CallLog { events: log.events.clone() },
    };
    stub.step(
        &mut buffers,
        &mut sim_box,
        &mut ff,
        0.1,
        &mut timings,
    )
    .unwrap();
    // Two ForceEval substeps means two force_field.step calls. The
    // ForceField in fixture() has zero slots so the only thing that
    // gets recorded in stub.log is the non-ForceEval calls. We assert
    // by counting that ForceEval was NEVER routed through execute().
    let events = log.events.lock().unwrap().clone();
    assert_eq!(
        events,
        vec![
            "exec_kick_half",
            "exec_drift",
            // (force_eval handled by runner, not stub.execute)
            "exec_kick_half",
            "exec_drift",
            "exec_kick_half",
        ]
    );
}

// rq-d4d435c8
#[test]
fn execute_with_force_eval_directly_returns_unexpected_substep() {
    let (gpu, mut buffers, mut sim_box, _ff, mut timings) = fixture();
    let _ = gpu;
    let log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan::empty(),
        log: CallLog { events: log.events.clone() },
    };
    let err = stub
        .execute(&SubStep::ForceEval { class: None, level: Some(heddle_md::forces::AggregateLevel::ForcesAndScalars) }, &mut buffers, &mut sim_box, &mut timings)
        .unwrap_err();
    match err {
        IntegratorError::UnexpectedSubStep { variant } => {
            assert_eq!(variant, "ForceEval");
        }
        other => panic!("expected UnexpectedSubStep, got {other:?}"),
    }
}

// A plan with the three velocity-Verlet ConstraintPoint markers.
fn marker_plan_vv() -> StepPlan {
    StepPlan {
        steps: vec![
            SubStep::ConstraintPoint { phase: ConstraintPhase::BeforeDrift, dt: 0.1 },
            SubStep::Drift { dt: 0.1, label: "d" },
            SubStep::ConstraintPoint { phase: ConstraintPhase::AfterDrift, dt: 0.1 },
            SubStep::ForceEval { class: None, level: Some(heddle_md::forces::AggregateLevel::ForcesAndScalars) },
            SubStep::KickHalf { dt: 0.1, label: "k", source: heddle_md::integrator::KickSource::Total },
            SubStep::ConstraintPoint { phase: ConstraintPhase::AfterKick, dt: 0.1 },
        ],
    }
}

// rq-99034e90
#[test]
fn constraint_point_markers_dispatch_to_matching_hooks() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let log = CallLog::default();
    let mut stub = PlanStub {
        plan: marker_plan_vv(),
        log: CallLog { events: log.events.clone() },
    };
    let constraint_log = CallLog::default();
    let mut constraint = RecordingConstraint {
        log: CallLog { events: constraint_log.events.clone() },
    };
    // Canonical order across the walk (before/after_drift) and the
    // post-force tail (the terminal after_kick projection).
    run_timestep(
        &mut stub,
        &mut buffers,
        &mut sim_box,
        &mut ff,
        Some(&mut constraint),
        None,
        0.1,
        &mut timings,
    )
    .unwrap();
    let events = constraint_log.events.lock().unwrap().clone();
    assert_eq!(events, vec!["before_drift", "after_drift", "after_kick"]);
}

// rq-3b42c2ff
#[test]
fn repeated_markers_dispatch_once_each_in_plan_order() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let stub_log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![
                SubStep::KickHalf { dt: 0.1, label: "B", source: heddle_md::integrator::KickSource::Total },
                SubStep::ConstraintPoint { phase: ConstraintPhase::BeforeDrift, dt: 0.1 },
                SubStep::Drift { dt: 0.1, label: "A_pre" },
                SubStep::ConstraintPoint { phase: ConstraintPhase::AfterDrift, dt: 0.1 },
                SubStep::Custom { dt: 0.1, label: "O" },
                SubStep::ConstraintPoint { phase: ConstraintPhase::BeforeDrift, dt: 0.1 },
                SubStep::Drift { dt: 0.1, label: "A_post" },
                SubStep::ConstraintPoint { phase: ConstraintPhase::AfterDrift, dt: 0.1 },
                SubStep::ForceEval { class: None, level: Some(heddle_md::forces::AggregateLevel::ForcesAndScalars) },
                SubStep::KickHalf { dt: 0.1, label: "B", source: heddle_md::integrator::KickSource::Total },
                SubStep::ConstraintPoint { phase: ConstraintPhase::AfterKick, dt: 0.1 },
            ],
        },
        log: CallLog { events: stub_log.events.clone() },
    };
    let constraint_log = CallLog::default();
    let mut constraint = RecordingConstraint {
        log: CallLog { events: constraint_log.events.clone() },
    };
    run_timestep(
        &mut stub,
        &mut buffers,
        &mut sim_box,
        &mut ff,
        Some(&mut constraint),
        None,
        0.1,
        &mut timings,
    )
    .unwrap();
    let events = constraint_log.events.lock().unwrap().clone();
    assert_eq!(
        events,
        vec![
            "before_drift",
            "after_drift",
            "before_drift",
            "after_drift",
            "after_kick",
        ]
    );
}

// rq-a90e4189
#[test]
fn plan_with_no_after_kick_marker_never_fires_after_kick() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let stub_log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![
                SubStep::ConstraintPoint { phase: ConstraintPhase::BeforeDrift, dt: 0.1 },
                SubStep::Drift { dt: 0.1, label: "d" },
                SubStep::ConstraintPoint { phase: ConstraintPhase::AfterDrift, dt: 0.1 },
                SubStep::ForceEval { class: None, level: Some(heddle_md::forces::AggregateLevel::ForcesAndScalars) },
                SubStep::KickHalf { dt: 0.1, label: "k", source: heddle_md::integrator::KickSource::Total },
            ],
        },
        log: CallLog { events: stub_log.events.clone() },
    };
    let constraint_log = CallLog::default();
    let mut constraint = RecordingConstraint {
        log: CallLog { events: constraint_log.events.clone() },
    };
    run_step(
        &mut stub,
        &mut buffers,
        &mut sim_box,
        &mut ff,
        Some(&mut constraint),
        None,
        None,
        0.1,
        &mut timings,
        RunStepOptions { runner_needs_scalars: true, ..Default::default() },
    )
    .unwrap();
    let events = constraint_log.events.lock().unwrap().clone();
    assert!(!events.contains(&"after_kick"));
}

// rq-c3b3ec99
#[test]
fn plan_with_no_markers_fires_no_constraint_hooks() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let stub_log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![SubStep::Custom { dt: 0.1, label: "only" }],
        },
        log: CallLog { events: stub_log.events.clone() },
    };
    let constraint_log = CallLog::default();
    let mut constraint = RecordingConstraint {
        log: CallLog { events: constraint_log.events.clone() },
    };
    run_step(
        &mut stub,
        &mut buffers,
        &mut sim_box,
        &mut ff,
        Some(&mut constraint),
        None,
        None,
        0.1,
        &mut timings,
        RunStepOptions { runner_needs_scalars: true, ..Default::default() },
    )
    .unwrap();
    let events = constraint_log.events.lock().unwrap().clone();
    assert!(events.is_empty(), "expected no constraint events, got {events:?}");
}

// rq-309d8d50
#[test]
fn plan_without_markers_fires_no_hooks_even_with_slot_present() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let stub_log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![
                SubStep::KickDrift { dt: 0.1, label: "kd", source: heddle_md::integrator::KickSource::Total },
                SubStep::ForceEval { class: None, level: Some(heddle_md::forces::AggregateLevel::ForcesAndScalars) },
                SubStep::KickHalf { dt: 0.1, label: "k", source: heddle_md::integrator::KickSource::Total },
            ],
        },
        log: CallLog { events: stub_log.events.clone() },
    };
    let constraint_log = CallLog::default();
    let mut constraint = RecordingConstraint {
        log: CallLog { events: constraint_log.events.clone() },
    };
    run_step(
        &mut stub,
        &mut buffers,
        &mut sim_box,
        &mut ff,
        Some(&mut constraint),
        None,
        None,
        0.1,
        &mut timings,
        RunStepOptions { runner_needs_scalars: true, ..Default::default() },
    )
    .unwrap();
    let events = constraint_log.events.lock().unwrap().clone();
    assert!(events.is_empty(), "expected no constraint events, got {events:?}");
}

// rq-f34598ae
#[test]
fn constraint_point_markers_dispatch_in_the_walk_or_no_op_without_slot() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    // With a slot: the interleaved before/after_drift markers dispatch
    // in the walk; the terminal after_kick fires in the post-force tail.
    let mut stub = PlanStub { plan: marker_plan_vv(), log: CallLog::default() };
    let constraint_log = CallLog::default();
    let mut constraint = RecordingConstraint {
        log: CallLog { events: constraint_log.events.clone() },
    };
    run_timestep(
        &mut stub, &mut buffers, &mut sim_box, &mut ff,
        Some(&mut constraint), None, 0.1, &mut timings,
    )
    .unwrap();
    assert_eq!(
        *constraint_log.events.lock().unwrap(),
        vec!["before_drift", "after_drift", "after_kick"]
    );
    // Same plan, no slot: completes Ok with no dispatch.
    let mut stub2 = PlanStub { plan: marker_plan_vv(), log: CallLog::default() };
    run_timestep(
        &mut stub2, &mut buffers, &mut sim_box, &mut ff,
        None, None, 0.1, &mut timings,
    )
    .unwrap();
}

// rq-2e4f64b0
#[test]
fn constraint_point_markers_are_no_ops_without_a_slot() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let mut stub = PlanStub { plan: marker_plan_vv(), log: CallLog::default() };
    // No constraint slot passed; the markers must be inert.
    run_step(
        &mut stub, &mut buffers, &mut sim_box, &mut ff,
        None, None, None, 0.1, &mut timings,
        RunStepOptions { runner_needs_scalars: true, ..Default::default() },
    )
    .unwrap();
}

// rq-62c54adc
#[test]
fn constraint_point_hook_receives_the_markers_own_dt() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![SubStep::ConstraintPoint {
                phase: ConstraintPhase::AfterKick,
                dt: 0.5,
            }],
        },
        log: CallLog::default(),
    };
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Real>::new()));
    let mut constraint = DtRecordingConstraint { seen: seen.clone() };
    // The terminal AfterKick fires in the post-force tail carrying its
    // own marker dt.
    run_timestep(
        &mut stub, &mut buffers, &mut sim_box, &mut ff,
        Some(&mut constraint), None, 0.1, &mut timings,
    )
    .unwrap();
    assert_eq!(*seen.lock().unwrap(), vec![0.5 as Real]);
}

// rq-195a6215
#[test]
fn one_run_step_call_produces_the_whole_canonical_post_force_order() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    // One shared ordered log records the wrapped thermostat's halves, the
    // trailing kick (via the stub's execute), the barostat apply, and the
    // terminal projection — all from a single run_step call.
    let order = CallLog::default();
    // Plan ends in [KickHalf, BarostatPoint, ConstraintPoint(AfterKick)];
    // the two markers form the trailing post-force run.
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![
                SubStep::ForceEval { class: None, level: Some(heddle_md::forces::AggregateLevel::ForcesAndScalars) },
                SubStep::KickHalf { dt: 0.1, label: "k", source: heddle_md::integrator::KickSource::Total },
                SubStep::BarostatPoint { dt: 0.1 },
                SubStep::ConstraintPoint { phase: ConstraintPhase::AfterKick, dt: 0.1 },
            ],
        },
        log: CallLog { events: order.events.clone() },
    };
    let mut thermostat = RecordingThermostat {
        log: CallLog { events: order.events.clone() },
    };
    let mut barostat = OrderRecordingBarostat {
        log: CallLog { events: order.events.clone() },
    };
    let mut constraint = RecordingConstraint {
        log: CallLog { events: order.events.clone() },
    };
    run_step(
        &mut stub, &mut buffers, &mut sim_box, &mut ff,
        Some(&mut constraint), Some(&mut thermostat), Some(&mut barostat),
        0.1, &mut timings,
        RunStepOptions { coupling_dt: Some(0.1 as Real), runner_needs_scalars: true, ..Default::default() },
    )
    .unwrap();
    // apply_pre (before walk) → trailing kick (main walk) → apply_post
    // (post-force-marker boundary) → barostat.apply → apply_after_kick,
    // all within the one run_step call. RATTLE-last.
    assert_eq!(
        *order.events.lock().unwrap(),
        vec!["apply_pre", "exec_kick_half", "apply_post", "barostat_apply", "after_kick"]
    );
}

// rq-d64bc1c6
#[test]
fn trailing_kick_runs_via_execute_in_the_main_walk() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    // A trailing Total-sourced KickHalf at index k followed only by a
    // post-force marker. run_step's main walk dispatches the kick via
    // integrator.execute (the composer that once absorbed it is gone).
    let stub_log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![
                SubStep::KickHalf { dt: 0.1, label: "k", source: heddle_md::integrator::KickSource::Total },
                SubStep::ConstraintPoint { phase: ConstraintPhase::AfterKick, dt: 0.1 },
            ],
        },
        log: CallLog { events: stub_log.events.clone() },
    };
    run_step(
        &mut stub, &mut buffers, &mut sim_box, &mut ff,
        None, None, None, 0.1, &mut timings,
        RunStepOptions { runner_needs_scalars: true, ..Default::default() },
    )
    .unwrap();
    assert_eq!(*stub_log.events.lock().unwrap(), vec!["exec_kick_half"]);
}

// --- Wrapped-thermostat coupling through run_step.coupling_dt ---

// A velocity-Verlet-shaped plan with no ThermostatHalf markers: the
// default (runner-wrapped) thermostat topology.
fn wrapped_plan() -> StepPlan {
    StepPlan {
        steps: vec![
            SubStep::ForceEval { class: None, level: Some(heddle_md::forces::AggregateLevel::ForcesAndScalars) },
            SubStep::KickHalf { dt: 0.1, label: "k", source: heddle_md::integrator::KickSource::Total },
        ],
    }
}

// rq-b7f9628d
#[test]
fn coupling_dt_some_fires_wrapped_halves_at_canonical_positions() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let order = CallLog::default();
    let mut stub = PlanStub { plan: wrapped_plan(), log: CallLog { events: order.events.clone() } };
    let mut thermostat = RecordingThermostat { log: CallLog { events: order.events.clone() } };
    run_step(
        &mut stub, &mut buffers, &mut sim_box, &mut ff,
        None, Some(&mut thermostat), None, 0.1, &mut timings,
        RunStepOptions { coupling_dt: Some(0.4 as Real), runner_needs_scalars: true, ..Default::default() },
    )
    .unwrap();
    // apply_pre before the walk, then the trailing kick, then apply_post.
    assert_eq!(
        *order.events.lock().unwrap(),
        vec!["apply_pre", "exec_kick_half", "apply_post"]
    );
}

// rq-888952cd
#[test]
fn coupling_dt_none_leaves_wrapped_thermostat_inert() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let order = CallLog::default();
    let mut stub = PlanStub { plan: wrapped_plan(), log: CallLog { events: order.events.clone() } };
    let mut thermostat = RecordingThermostat { log: CallLog { events: order.events.clone() } };
    run_step(
        &mut stub, &mut buffers, &mut sim_box, &mut ff,
        None, Some(&mut thermostat), None, 0.1, &mut timings,
        RunStepOptions { coupling_dt: None, runner_needs_scalars: true, ..Default::default() },
    )
    .unwrap();
    // Neither half fires; only the trailing kick runs.
    assert_eq!(*order.events.lock().unwrap(), vec!["exec_kick_half"]);
}

// rq-b7ef61f1
#[test]
fn run_step_ignores_coupling_dt_when_plan_owns_thermostat() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let order = CallLog::default();
    // Plan owns its thermostat placement via ThermostatHalf markers.
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![
                SubStep::ThermostatHalf { dt: 0.1, phase: heddle_md::integrator::ThermostatPhase::Pre },
                SubStep::KickHalf { dt: 0.1, label: "k", source: heddle_md::integrator::KickSource::Total },
                SubStep::ThermostatHalf { dt: 0.1, phase: heddle_md::integrator::ThermostatPhase::Post },
            ],
        },
        log: CallLog { events: order.events.clone() },
    };
    let mut thermostat = RecordingThermostat { log: CallLog { events: order.events.clone() } };
    run_step(
        &mut stub, &mut buffers, &mut sim_box, &mut ff,
        None, Some(&mut thermostat), None, 0.1, &mut timings,
        RunStepOptions { coupling_dt: Some(0.4 as Real), runner_needs_scalars: true, ..Default::default() },
    )
    .unwrap();
    // The coupling_dt path is suppressed: apply_pre / apply_post fire
    // exactly once each, dispatched at the ThermostatHalf markers (not
    // doubled by a wrapped pre/post).
    assert_eq!(
        *order.events.lock().unwrap(),
        vec!["apply_pre", "exec_kick_half", "apply_post"]
    );
}

// rq-38e9c3b5
#[test]
fn coupling_dt_some_is_a_no_op_without_a_thermostat_slot() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let mut stub = PlanStub { plan: wrapped_plan(), log: CallLog::default() };
    // coupling_dt is Some but thermostat is None: run_step completes with
    // no thermostat method invoked.
    run_step(
        &mut stub, &mut buffers, &mut sim_box, &mut ff,
        None, None, None, 0.1, &mut timings,
        RunStepOptions { coupling_dt: Some(0.4 as Real), runner_needs_scalars: true, ..Default::default() },
    )
    .unwrap();
}

// --- Plan-declared barostat points ---

fn total_kick(dt: Real) -> SubStep {
    SubStep::KickHalf { dt, label: "k", source: heddle_md::integrator::KickSource::Total }
}

// rq-b167a309
#[test]
fn terminal_barostat_point_dispatches_apply_once() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let mut stub = PlanStub {
        plan: StepPlan { steps: vec![total_kick(0.3), SubStep::BarostatPoint { dt: 0.3 }] },
        log: CallLog::default(),
    };
    let applies = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Real>::new()));
    let mut baro = RecordingBarostat { applies: applies.clone(), periodic: false };
    // run_step walks the whole plan, so the terminal BarostatPoint fires
    // in run_step's own post-force tail, after the trailing kick.
    run_timestep(
        &mut stub, &mut buffers, &mut sim_box, &mut ff,
        None, Some(&mut baro), 0.3, &mut timings,
    )
    .unwrap();
    assert_eq!(*applies.lock().unwrap(), vec![0.3 as Real]);
}

// rq-68061953
#[test]
fn barostat_point_is_no_op_without_a_barostat() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let mut stub = PlanStub {
        plan: StepPlan { steps: vec![total_kick(0.1), SubStep::BarostatPoint { dt: 0.1 }] },
        log: CallLog::default(),
    };
    run_step(
        &mut stub, &mut buffers, &mut sim_box, &mut ff,
        None, None, None, 0.1, &mut timings,
        RunStepOptions { runner_needs_scalars: true, ..Default::default() },
    )
    .unwrap();
}

// rq-63fe749a
#[test]
fn barostat_point_is_inert_for_a_periodic_barostat() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let mut stub = PlanStub {
        plan: StepPlan { steps: vec![total_kick(0.1), SubStep::BarostatPoint { dt: 0.1 }] },
        log: CallLog::default(),
    };
    let applies = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Real>::new()));
    let mut baro = RecordingBarostat { applies: applies.clone(), periodic: true };
    run_timestep(
        &mut stub, &mut buffers, &mut sim_box, &mut ff,
        None, Some(&mut baro), 0.1, &mut timings,
    )
    .unwrap();
    // A periodic barostat's apply is the no-op default; the terminal
    // BarostatPoint records nothing (its periodic move fires through
    // apply_move at its own batch cadence, not through apply).
    assert!(applies.lock().unwrap().is_empty());
}

// rq-f9d0621d
#[test]
fn has_barostat_points_and_trailing_start_reflect_the_plan() {
    // A terminal BarostatPoint sits in the post-force tail:
    // trailing_post_force_start < steps.len().
    let terminal = StepPlan {
        steps: vec![total_kick(0.5), SubStep::BarostatPoint { dt: 0.5 }],
    };
    assert!(terminal.has_barostat_points());
    assert!(terminal.trailing_post_force_start() < terminal.steps.len());
    assert!(!terminal.has_interleaved_barostat_point());

    // An interleaved BarostatPoint is not in a trailing marker run:
    // trailing_post_force_start == steps.len().
    let interleaved = StepPlan {
        steps: vec![
            SubStep::BarostatPoint { dt: 0.5 },
            SubStep::ForceEval { class: None, level: Some(heddle_md::forces::AggregateLevel::ForcesOnly) },
            total_kick(0.5),
        ],
    };
    assert!(interleaved.has_barostat_points());
    assert_eq!(interleaved.trailing_post_force_start(), interleaved.steps.len());
    assert!(interleaved.has_interleaved_barostat_point());
}

// rq-e043b064
#[test]
fn interleaved_barostat_point_dispatches_apply_in_the_walk() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![
                SubStep::BarostatPoint { dt: 0.2 },
                SubStep::ForceEval { class: None, level: Some(heddle_md::forces::AggregateLevel::ForcesAndScalars) },
                total_kick(0.2),
            ],
        },
        log: CallLog::default(),
    };
    let applies = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Real>::new()));
    let mut baro = RecordingBarostat { applies: applies.clone(), periodic: false };
    run_step(
        &mut stub, &mut buffers, &mut sim_box, &mut ff,
        None, None, Some(&mut baro), 0.2, &mut timings,
        RunStepOptions { runner_needs_scalars: true, ..Default::default() },
    )
    .unwrap();
    assert_eq!(*applies.lock().unwrap(), vec![0.2 as Real]);
}

// =====================================================================
// Open-registry scenarios (SlotConfig + Builder predicates +
// validate_params + lookup). See rqm/integration/framework.md.
// =====================================================================

// rq-79c53582
#[test]
fn populated_registry_unknown_kind_reports_unknown_kind() {
    let gpu = init_device().unwrap();
    let registry = IntegratorRegistry::with_builtins();
    let slot = SlotConfig::from_params_str("no-such-integrator", "");
    let err = registry.build(&slot, &gpu, 4, 0).unwrap_err();
    match err {
        IntegratorError::UnknownKind(name) => assert_eq!(name, "no-such-integrator"),
        other => panic!("expected UnknownKind, got {other:?}"),
    }
}

// rq-8fbdbc0c
#[test]
fn lookup_returns_builder_for_registered_kind() {
    let registry = IntegratorRegistry::with_builtins();
    let b = registry.lookup("velocity-verlet").expect("registered");
    assert_eq!(b.kind_name(), "velocity-verlet");
}

// rq-e8adaa9c
#[test]
fn lookup_returns_none_for_unregistered_kind() {
    let registry = IntegratorRegistry::with_builtins();
    assert!(registry.lookup("no-such-integrator").is_none());
}

// rq-e9be025b
#[test]
fn velocity_verlet_builder_does_not_own_thermostat_or_barostat() {
    let registry = IntegratorRegistry::with_builtins();
    let b = registry.lookup("velocity-verlet").unwrap();
    let vv = SlotConfig::from_params_str("velocity-verlet", "lossless = false\n");
    assert!(!b.owns_thermostat(&vv.params));
    assert!(!b.owns_barostat(&vv.params));
}

// rq-4dd5d2d0
#[test]
fn langevin_baoab_builder_owns_thermostat_not_barostat() {
    let registry = IntegratorRegistry::with_builtins();
    let b = registry.lookup("langevin-baoab").unwrap();
    let lan = SlotConfig::from_params_str(
        "langevin-baoab",
        "friction = 1.0e12\ntemperature = 300.0\nseed = 0\n",
    );
    assert!(b.owns_thermostat(&lan.params));
    assert!(!b.owns_barostat(&lan.params));
}

// rq-95b66af0
#[test]
fn mtk_npt_builder_owns_thermostat_and_barostat() {
    let registry = IntegratorRegistry::with_builtins();
    let b = registry.lookup("mtk-npt").unwrap();
    let mtk = SlotConfig::from_params_str(
        "mtk-npt",
        "temperature = 85.0\npressure = 1.0e5\ntau_t = 1.0e-13\ntau_p = 1.0e-12\nchain_length = 3\nyoshida_order = 3\nn_resp = 1\n",
    );
    assert!(b.owns_thermostat(&mtk.params));
    assert!(b.owns_barostat(&mtk.params));
}

// rq-7d37c707
#[test]
fn velocity_verlet_supports_constraints_depends_on_lossless() {
    let registry = IntegratorRegistry::with_builtins();
    let b = registry.lookup("velocity-verlet").unwrap();
    let lossy = SlotConfig::from_params_str("velocity-verlet", "lossless = false\n");
    let lossless = SlotConfig::from_params_str("velocity-verlet", "lossless = true\n");
    assert!(b.supports_constraints(&lossy.params));
    assert!(!b.supports_constraints(&lossless.params));
}

// rq-084ba25b
#[test]
fn validate_params_accepts_well_formed_params() {
    let registry = IntegratorRegistry::with_builtins();
    let b = registry.lookup("velocity-verlet").unwrap();
    let p = SlotConfig::from_params_str("velocity-verlet", "lossless = false\n");
    b.validate_params(&p.params).unwrap();
}

// rq-cb52dec0
#[test]
fn validate_params_rejects_out_of_domain_field() {
    let registry = IntegratorRegistry::with_builtins();
    let b = registry.lookup("langevin-baoab").unwrap();
    let p = SlotConfig::from_params_str(
        "langevin-baoab",
        "friction = -1.0\ntemperature = 300.0\nseed = 1\n",
    );
    match b.validate_params(&p.params).unwrap_err() {
        heddle_md::io::ConfigError::InvalidValue { field, .. } => {
            assert_eq!(field, "integrator.friction");
        }
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

// rq-7a076bc9
#[test]
fn validate_params_rejects_unknown_field() {
    let registry = IntegratorRegistry::with_builtins();
    let b = registry.lookup("velocity-verlet").unwrap();
    let p = SlotConfig::from_params_str("velocity-verlet", "lossless = false\njunk = true\n");
    match b.validate_params(&p.params).unwrap_err() {
        heddle_md::io::ConfigError::Parse { .. } => {}
        other => panic!("expected Parse, got {other:?}"),
    }
}

// =====================================================================
// SubStep::ForceEval { class: Option<ForceClass> } scenarios. See the
// `Force classes and per-class evaluation` block in
// rqm/forces/framework.md and the SubStep section in
// rqm/integration/framework.md.
// =====================================================================

use heddle_md::forces::ForceClass;

// rq-751bbb3c
#[test]
fn execute_with_force_eval_some_class_also_returns_unexpected_substep() {
    let (gpu, mut buffers, mut sim_box, _ff, mut timings) = fixture();
    let _ = gpu;
    let log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan::empty(),
        log: CallLog { events: log.events.clone() },
    };
    let err = stub
        .execute(
            &SubStep::ForceEval { class: Some(ForceClass::Fast), level: Some(heddle_md::forces::AggregateLevel::ForcesAndScalars) },
            &mut buffers,
            &mut sim_box,
            &mut timings,
        )
        .unwrap_err();
    match err {
        IntegratorError::UnexpectedSubStep { variant } => {
            assert_eq!(variant, "ForceEval");
        }
        other => panic!("expected UnexpectedSubStep, got {other:?}"),
    }
}

// rq-256287cb
#[test]
fn force_eval_some_fast_class_dispatches_to_step_class_fast() {
    // Walk a plan with one ForceEval{class: Some(Fast)} against a
    // ForceField that has one Fast slot (LJ). step_class(Fast) launches
    // its slot kernels and the accumulator.
    let gpu = init_device().unwrap();
    let state = small_state(2);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = ForceField::new(
        &heddle_md::forces::PotentialRegistry::with_builtins(),
        &gpu,
        2,
        &sim_box,
        &[heddle_md::io::config::ParticleTypeConfig {
            name: "Ar".to_string(),
            mass: 1.0,
            sigma: None,
            epsilon: None,
            charge: 0.0,
        }],
        &[heddle_md::io::config::PairInteractionConfig::lennard_jones(("Ar".to_string(), "Ar".to_string()), 1.0, 1.0, 1.0, Some(1.0))],
        &[],
        &[],
        &[],
        None,
        &[],
        &BondList::empty(2),
        &heddle_md::forces::AngleList::empty(0),
        &DihedralList::empty(0),
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![SubStep::ForceEval { class: Some(ForceClass::Fast), level: Some(heddle_md::forces::AggregateLevel::ForcesAndScalars) }],
        },
        log: CallLog { events: log.events.clone() },
    };
    stub.step( &mut buffers, &mut sim_box, &mut ff, 0.001, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    let count_of = |name: &str| {
        report.stages.iter().find(|s| s.name == name).map(|s| s.count).unwrap_or(0)
    };
    // The Fast LJ slot's kernel fires; the accumulator fires once.
    assert_eq!(count_of("jit_composed_pair_force"), 1);
    assert_eq!(count_of("combine_class_totals"), 1);
}

#[test]
fn force_eval_some_slow_class_on_fast_only_ff_is_noop() {
    // Plan emits ForceEval{class: Some(Slow)} against a Fast-only LJ
    // ForceField. step_class(Slow) finds no Slow slots and returns
    // without launching anything — no LJ kernels, no accumulator.
    let gpu = init_device().unwrap();
    let state = small_state(2);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = ForceField::new(
        &heddle_md::forces::PotentialRegistry::with_builtins(),
        &gpu,
        2,
        &sim_box,
        &[heddle_md::io::config::ParticleTypeConfig {
            name: "Ar".to_string(),
            mass: 1.0,
            sigma: None,
            epsilon: None,
            charge: 0.0,
        }],
        &[heddle_md::io::config::PairInteractionConfig::lennard_jones(("Ar".to_string(), "Ar".to_string()), 1.0, 1.0, 1.0, Some(1.0))],
        &[],
        &[],
        &[],
        None,
        &[],
        &BondList::empty(2),
        &heddle_md::forces::AngleList::empty(0),
        &DihedralList::empty(0),
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![SubStep::ForceEval { class: Some(ForceClass::Slow), level: Some(heddle_md::forces::AggregateLevel::ForcesAndScalars) }],
        },
        log: CallLog { events: log.events.clone() },
    };
    stub.step( &mut buffers, &mut sim_box, &mut ff, 0.001, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    assert!(
        report.stages.iter().all(|s| s.count == 0),
        "ForceEval{{Slow}} on Fast-only ForceField launched kernels: {:?}",
        report.stages,
    );
}

// rq-5855473b
#[test]
fn force_eval_none_class_continues_to_dispatch_to_step() {
    // Plan emits ForceEval{class: None} against a Fast-only LJ ForceField.
    // Runner calls step() which evaluates every slot and runs the
    // combiner — one accumulator launch.
    let gpu = init_device().unwrap();
    let state = small_state(2);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = ForceField::new(
        &heddle_md::forces::PotentialRegistry::with_builtins(),
        &gpu,
        2,
        &sim_box,
        &[heddle_md::io::config::ParticleTypeConfig {
            name: "Ar".to_string(),
            mass: 1.0,
            sigma: None,
            epsilon: None,
            charge: 0.0,
        }],
        &[heddle_md::io::config::PairInteractionConfig::lennard_jones(("Ar".to_string(), "Ar".to_string()), 1.0, 1.0, 1.0, Some(1.0))],
        &[],
        &[],
        &[],
        None,
        &[],
        &BondList::empty(2),
        &heddle_md::forces::AngleList::empty(0),
        &DihedralList::empty(0),
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![SubStep::ForceEval { class: None, level: Some(heddle_md::forces::AggregateLevel::ForcesAndScalars) }],
        },
        log: CallLog { events: log.events.clone() },
    };
    stub.step( &mut buffers, &mut sim_box, &mut ff, 0.001, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    let count_of = |name: &str| {
        report.stages.iter().find(|s| s.name == name).map(|s| s.count).unwrap_or(0)
    };
    assert_eq!(count_of("combine_class_totals"), 1);
    assert_eq!(count_of("jit_composed_pair_force"), 1);
}

// --- resolve_aggregate_level + integrator-plan level preferences -------

// rq-9f551521
#[test]
fn resolve_aggregate_level_upgrades_on_a_logging_step() {
    use heddle_md::forces::AggregateLevel;
    use heddle_md::integrator::resolve_aggregate_level;
    let resolved = resolve_aggregate_level(Some(AggregateLevel::ForcesOnly), true);
    assert!(matches!(resolved, AggregateLevel::ForcesAndScalars));
}

// rq-1ee2ef41
#[test]
fn resolve_aggregate_level_upgrades_on_a_trajectory_frame() {
    use heddle_md::forces::AggregateLevel;
    use heddle_md::integrator::resolve_aggregate_level;
    // None sub-step level + runner_needs_scalars upgrades to ForcesAndScalars.
    let resolved = resolve_aggregate_level(None, true);
    assert!(matches!(resolved, AggregateLevel::ForcesAndScalars));
}

// rq-75a19aca
#[test]
fn resolve_aggregate_level_keeps_forces_and_scalars_when_already_requested() {
    use heddle_md::forces::AggregateLevel;
    use heddle_md::integrator::resolve_aggregate_level;
    let resolved = resolve_aggregate_level(Some(AggregateLevel::ForcesAndScalars), false);
    assert!(matches!(resolved, AggregateLevel::ForcesAndScalars));
}

// rq-5e5f48da
#[test]
fn resolve_aggregate_level_falls_through_to_forces_only_when_no_logging_or_traj() {
    use heddle_md::forces::AggregateLevel;
    use heddle_md::integrator::resolve_aggregate_level;
    let resolved = resolve_aggregate_level(Some(AggregateLevel::ForcesOnly), false);
    assert!(matches!(resolved, AggregateLevel::ForcesOnly));
    // None sub-step + no runner request → defaults to ForcesOnly.
    let resolved = resolve_aggregate_level(None, false);
    assert!(matches!(resolved, AggregateLevel::ForcesOnly));
}

// rq-5a7e597e
#[test]
fn velocity_verlet_plan_requests_forces_only_by_default() {
    use heddle_md::forces::AggregateLevel;
    use heddle_md::integrator::SubStep;
    let gpu = init_device().unwrap();
    let kind = SlotConfig::from_params_str("velocity-verlet", "lossless = false\n");
    let integ = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 4, 0)
        .unwrap();
    let plan = integ.plan(1.0e-15);
    let force_eval_level = plan
        .steps
        .iter()
        .find_map(|s| match s {
            SubStep::ForceEval { level, .. } => Some(*level),
            _ => None,
        })
        .expect("VV plan must contain a ForceEval substep");
    assert_eq!(force_eval_level, Some(AggregateLevel::ForcesOnly));
}

// rq-3a9cb990
#[test]
fn mtk_npt_plan_requests_forces_and_scalars() {
    use heddle_md::forces::AggregateLevel;
    use heddle_md::integrator::SubStep;
    let gpu = init_device().unwrap();
    let kind = SlotConfig::from_params_str(
        "mtk-npt",
        "temperature = 9.51e-4\npressure = 3.4e-9\ntau_t = 4131.0\ntau_p = 41310.0\n",
    );
    let integ = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 4, 0)
        .unwrap();
    let plan = integ.plan(1.0);
    let force_eval_level = plan
        .steps
        .iter()
        .find_map(|s| match s {
            SubStep::ForceEval { level, .. } => Some(*level),
            _ => None,
        })
        .expect("MTK-NPT plan must contain a ForceEval substep");
    assert_eq!(force_eval_level, Some(AggregateLevel::ForcesAndScalars));
}

// =============================================================================
// Class-sourced kicks and plan-declared thermostat points
// (rqm/integration/framework.md, KickSource / ThermostatHalf).
// =============================================================================

use heddle_md::integrator::{KickSource, ThermostatPhase};

fn fill_class_forces(
    gpu: &GpuContext,
    ff: &mut ForceField,
    fast: (Real, Real, Real),
    slow: (Real, Real, Real),
    n: usize,
) {
    gpu.device.htod_sync_copy_into(&vec![fast.0; n], &mut ff.fast_total_forces_x).unwrap();
    gpu.device.htod_sync_copy_into(&vec![fast.1; n], &mut ff.fast_total_forces_y).unwrap();
    gpu.device.htod_sync_copy_into(&vec![fast.2; n], &mut ff.fast_total_forces_z).unwrap();
    gpu.device.htod_sync_copy_into(&vec![slow.0; n], &mut ff.slow_total_forces_x).unwrap();
    gpu.device.htod_sync_copy_into(&vec![slow.1; n], &mut ff.slow_total_forces_y).unwrap();
    gpu.device.htod_sync_copy_into(&vec![slow.2; n], &mut ff.slow_total_forces_z).unwrap();
}

// rq-cac5cc99
#[test]
fn class_sourced_kick_half_reads_class_accumulator_not_combined() {
    let gpu = init_device().unwrap();
    let state = small_state(1);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = empty_force_field(&gpu, 1);
    fill_class_forces(&gpu, &mut ff, (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), 1);
    let log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![SubStep::KickHalf {
                dt: 0.5,
                label: "slow_kick",
                source: KickSource::Class(heddle_md::forces::ForceClass::Slow),
            }],
        },
        log: CallLog { events: log.events.clone() },
    };
    let mut timings = Timings::new(&gpu).unwrap();
    run_step(
        &mut stub,
        &mut buffers,
        &mut sim_box,
        &mut ff,
        None,
        None,
        None,
        0.5,
        &mut timings,
        RunStepOptions { runner_needs_scalars: true, ..Default::default() },
    )
    .unwrap();
    // v_x = (F_slow / m) * dt/2 = 2.0 * 0.25 = 0.5 (not 1.0 * 0.25).
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    assert_eq!(vx, vec![0.5 as Real]);
    // The runner dispatched the kick; the integrator's execute was
    // never consulted.
    assert!(log.events.lock().unwrap().is_empty());
}

// rq-5edc2d8f
#[test]
fn class_sourced_kick_drift_kicks_half_dt_and_drifts_dt() {
    let gpu = init_device().unwrap();
    let state = small_state(1); // at rest at x = 0, m = 1
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = empty_force_field(&gpu, 1);
    fill_class_forces(&gpu, &mut ff, (4.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1);
    let log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![SubStep::KickDrift {
                dt: 0.5,
                label: "fast_kick_drift",
                source: KickSource::Class(heddle_md::forces::ForceClass::Fast),
            }],
        },
        log: CallLog { events: log.events.clone() },
    };
    let mut timings = Timings::new(&gpu).unwrap();
    run_step(
        &mut stub,
        &mut buffers,
        &mut sim_box,
        &mut ff,
        None,
        None,
        None,
        0.5,
        &mut timings,
        RunStepOptions { runner_needs_scalars: true, ..Default::default() },
    )
    .unwrap();
    // v = (4/1) * 0.5/2 = 1.0; x = 0 + v * 0.5 = 0.5.
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    assert_eq!(vx, vec![1.0 as Real]);
    let posq = gpu.device.dtoh_sync_copy(&buffers.posq).unwrap();
    assert_eq!(posq[0].x, 0.5 as Real);
    assert!(log.events.lock().unwrap().is_empty());
}

// rq-0047eebd
#[test]
fn total_sourced_kick_is_dispatched_to_execute() {
    let gpu = init_device().unwrap();
    let state = small_state(1);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = empty_force_field(&gpu, 1);
    let log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![SubStep::KickHalf {
                dt: 0.5,
                label: "k",
                source: KickSource::Total,
            }],
        },
        log: CallLog { events: log.events.clone() },
    };
    let mut timings = Timings::new(&gpu).unwrap();
    run_step(
        &mut stub,
        &mut buffers,
        &mut sim_box,
        &mut ff,
        None,
        None,
        None,
        0.5,
        &mut timings,
        RunStepOptions { runner_needs_scalars: true, ..Default::default() },
    )
    .unwrap();
    assert_eq!(log.events.lock().unwrap().clone(), vec!["exec_kick_half"]);
}

// rq-2eac2f62
#[test]
fn thermostat_half_markers_dispatch_at_marker_positions() {
    let gpu = init_device().unwrap();
    let state = small_state(1);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = empty_force_field(&gpu, 1);
    let log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![
                SubStep::ThermostatHalf { dt: 0.5, phase: ThermostatPhase::Pre },
                SubStep::Drift { dt: 0.5, label: "d" },
                SubStep::ThermostatHalf { dt: 0.5, phase: ThermostatPhase::Post },
            ],
        },
        log: CallLog { events: log.events.clone() },
    };
    let mut therm = RecordingThermostat { log: CallLog { events: log.events.clone() } };
    let mut timings = Timings::new(&gpu).unwrap();
    run_step(
        &mut stub,
        &mut buffers,
        &mut sim_box,
        &mut ff,
        None,
        Some(&mut therm),
        None,
        0.5,
        &mut timings,
        RunStepOptions { runner_needs_scalars: true, ..Default::default() },
    )
    .unwrap();
    assert_eq!(
        log.events.lock().unwrap().clone(),
        vec!["apply_pre", "exec_drift", "apply_post"]
    );
}

// rq-22755bc1
#[test]
fn thermostat_half_is_noop_without_thermostat() {
    let gpu = init_device().unwrap();
    let state = small_state(1);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10(&gpu);
    let mut ff = empty_force_field(&gpu, 1);
    let log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![
                SubStep::ThermostatHalf { dt: 0.5, phase: ThermostatPhase::Pre },
                SubStep::Drift { dt: 0.5, label: "d" },
                SubStep::ThermostatHalf { dt: 0.5, phase: ThermostatPhase::Post },
            ],
        },
        log: CallLog { events: log.events.clone() },
    };
    let mut timings = Timings::new(&gpu).unwrap();
    run_step(
        &mut stub,
        &mut buffers,
        &mut sim_box,
        &mut ff,
        None,
        None,
        None,
        0.5,
        &mut timings,
        RunStepOptions { runner_needs_scalars: true, ..Default::default() },
    )
    .unwrap();
    assert_eq!(log.events.lock().unwrap().clone(), vec!["exec_drift"]);
}

// rq-72bb7b90
#[test]
fn has_thermostat_points_reflects_plan_contents() {
    let no_markers = StepPlan {
        steps: vec![SubStep::Drift { dt: 0.1, label: "d" }],
    };
    assert!(!no_markers.has_thermostat_points());
    let with_marker = StepPlan {
        steps: vec![
            SubStep::Drift { dt: 0.1, label: "d" },
            SubStep::ThermostatHalf { dt: 0.1, phase: ThermostatPhase::Post },
        ],
    };
    assert!(with_marker.has_thermostat_points());
}
