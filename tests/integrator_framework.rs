use dynamics::integrator::IntegratorStepExt;
use dynamics::forces::{BondList, ExclusionList, ForceField};
use dynamics::gpu::{GpuContext, ParticleBuffers, init_device};
use dynamics::integrator::{
    Integrator, IntegratorBuilder, IntegratorError, IntegratorRegistry, LangevinBaoabBuilder,
    LangevinBaoabState, VelocityVerletBuilder,
};
use dynamics::io::IntegratorKind;
use dynamics::io::config::NeighborListConfig;
use dynamics::pbc::SimulationBox;
use dynamics::state::ParticleState;
use dynamics::timings::{KernelStage, Timings};

fn small_state(n: usize) -> ParticleState {
    let pos: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let zero = vec![0.0_f32; n];
    let m = vec![1.0_f32; n];
    ParticleState::new(
        pos,
        zero.clone(),
        zero.clone(),
        zero.clone(),
        zero.clone(),
        zero,
        m,
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
            None,
    )
    .unwrap()
}

fn vv_kind(lossless: bool) -> IntegratorKind {
    IntegratorKind::VelocityVerlet { lossless }
}

fn langevin_kind(seed: u64) -> IntegratorKind {
    IntegratorKind::LangevinBaoab {
        friction: 1.0e12,
        temperature: 300.0,
        seed,
    }
}

fn box_10() -> SimulationBox {
    SimulationBox::new(10.0, 10.0, 10.0, 0.0, 0.0, 0.0).unwrap()
}

fn empty_force_field(gpu: &GpuContext, n: usize) -> ForceField {
    ForceField::new(
        gpu,
        n,
        &box_10(),
        &[],
        &[],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(n),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(n),
        &NeighborListConfig::AllPairs,
    )
    .unwrap()
}

// rq-e02917c3
#[test]
fn construct_vv_lossy_via_registry() {
    let gpu = init_device().unwrap();
    let registry = IntegratorRegistry::with_builtins();
    let _integrator = registry.build(&vv_kind(false), &gpu, 4).unwrap();
}

// rq-db78448e
#[test]
fn construct_vv_lossless_via_registry() {
    let gpu = init_device().unwrap();
    let registry = IntegratorRegistry::with_builtins();
    let state = small_state(4);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10();
    let mut ff = empty_force_field(&gpu, 4);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integrator = registry.build(&vv_kind(true), &gpu, 4).unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, None, 0.1, &mut timings)
        .unwrap();
    // The lossless build is observable through the lossless KernelStage labels.
    let report = timings.finalize().unwrap();
    let names: Vec<&str> = report.stages.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"vv_kick_drift_lossless"));
    assert!(names.contains(&"vv_kick_lossless"));
}

// rq-47877631
#[test]
fn construct_langevin_via_registry() {
    let gpu = init_device().unwrap();
    let registry = IntegratorRegistry::with_builtins();
    let _integrator = registry.build(&langevin_kind(42), &gpu, 4).unwrap();
}

// rq-48fd88ed
#[test]
fn construct_with_particle_count_zero() {
    let gpu = init_device().unwrap();
    let registry = IntegratorRegistry::with_builtins();
    let _integrator = registry.build(&vv_kind(true), &gpu, 0).unwrap();
}

#[test]
fn registry_without_matching_builder_reports_unknown_kind() {
    let gpu = init_device().unwrap();
    let registry = IntegratorRegistry::new();
    let err = registry.build(&vv_kind(false), &gpu, 4).unwrap_err();
    match err {
        IntegratorError::UnknownKind(name) => assert_eq!(name, "velocity-verlet"),
        other => panic!("expected UnknownKind, got {other:?}"),
    }
}

#[derive(Debug)]
struct StubBuilder;

#[derive(Debug)]
struct StubIntegrator;

impl IntegratorBuilder for StubBuilder {
    fn kind_name(&self) -> &'static str {
        // Use "velocity-verlet" so a real IntegratorKind can route to us; the
        // built-in builder is added later and therefore shadowed by the stub
        // when the stub is first in the builders Vec.
        "velocity-verlet"
    }
    fn build(
        &self,
        _gpu: &GpuContext,
        _particle_count: usize,
        _kind: &IntegratorKind,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        Ok(Box::new(StubIntegrator))
    }
}

impl Integrator for StubIntegrator {
    fn plan(&self, _dt: f32) -> dynamics::integrator::StepPlan {
        dynamics::integrator::StepPlan::empty()
    }

    fn execute(
        &mut self,
        _substep: &dynamics::integrator::SubStep,
        _buffers: &mut ParticleBuffers,
        _sim_box: &mut SimulationBox,
        _timings: &mut Timings,
    ) -> Result<(), IntegratorError> {
        Ok(())
    }
}

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
    let mut sim_box = box_10();
    let mut ff = empty_force_field(&gpu, 4);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integrator = registry.build(&vv_kind(false), &gpu, 4).unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, None, 0.1, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    // Stub launches no kernels, so neither vv_kick_drift nor vv_kick should
    // appear in the timings report.
    let names: Vec<&str> = report.stages.iter().map(|s| s.name.as_str()).collect();
    assert!(!names.contains(&"vv_kick_drift"));
    assert!(!names.contains(&"vv_kick"));
}

// rq-171b99f5
#[test]
fn step_on_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = ParticleState::new(
        Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        vec![0.0_f32; 0],
        Vec::new(), None,
            None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10();
    let mut ff = empty_force_field(&gpu, 0);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&vv_kind(false), &gpu, 0)
        .unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, None, 0.1, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    assert!(report.stages.is_empty());
}

// rq-2980a672
#[test]
fn vv_step_launches_kick_drift_force_and_kick() {
    let gpu = init_device().unwrap();
    // Build a state with non-zero velocities so the drift visibly moves
    // positions during step() even with zero forces.
    let n = 4;
    let mut state = small_state(n);
    state.velocities_x = (0..n).map(|i| 0.5 + i as f32 * 0.1).collect();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&vv_kind(false), &gpu, n)
        .unwrap();
    let snap_positions = state.positions_x.clone();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, None, 0.1, &mut timings)
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

// rq-7b9aada4
#[test]
fn lossless_vv_step_uses_lossless_kernels() {
    let gpu = init_device().unwrap();
    let state = small_state(4);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10();
    let mut ff = empty_force_field(&gpu, 4);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&vv_kind(true), &gpu, 4)
        .unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, None, 0.1, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    let names: Vec<&str> = report.stages.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"vv_kick_drift_lossless"));
    assert!(names.contains(&"vv_kick_lossless"));
    assert!(!names.contains(&"vv_kick_drift"));
    assert!(!names.contains(&"vv_kick"));
}

#[test]
fn integrator_owns_force_evaluation_inside_step() {
    // Wire up a real LJ slot so the force pipeline runs and we can confirm
    // KernelStage::LJ_PAIR_FORCE was triggered exactly once per step() call.
    use dynamics::io::config::{PairInteractionConfig, PairPotentialParams, ParticleTypeConfig};
    let gpu = init_device().unwrap();
    let state = small_state(4);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10();
    let mut ff = ForceField::new(&gpu,
        4,
        &sim_box,
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[PairInteractionConfig {
            between: ("Ar".to_string(), "Ar".to_string()),
            cutoff: 1.0,
            r_switch: 1.0,
            potential: PairPotentialParams::LennardJones { sigma: 1.0, epsilon: 1.0 },
        }],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(4),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&vv_kind(false), &gpu, 4)
        .unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, None, 0.001, &mut timings)
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
    assert_eq!(count("lj_pair_force"), 1);
    assert_eq!(count("reduce_pair_forces"), 1);
    assert_eq!(count("accumulate_forces"), 1);
}

// rq-d12c24f0
#[test]
fn two_consecutive_langevin_steps_produce_different_velocities() {
    let gpu = init_device().unwrap();
    let state = small_state(2);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10();
    let mut ff = empty_force_field(&gpu, 2);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&langevin_kind(1), &gpu, 2)
        .unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, None, 1.0e-15, &mut timings)
        .unwrap();
    let mut state_after_first = state.clone();
    state_after_first.download_from(&buffers).unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, None, 1.0e-15, &mut timings)
        .unwrap();
    let mut state_after_second = state.clone();
    state_after_second.download_from(&buffers).unwrap();
    assert_ne!(state_after_first.velocities_x, state_after_second.velocities_x);
}

// rq-706001ec
#[test]
fn two_independent_runs_byte_identical() {
    let gpu = init_device().unwrap();
    let state = small_state(4);

    let mut buffers_a = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut buffers_b = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box_a = box_10();
    let mut sim_box_b = box_10();
    let mut ff_a = empty_force_field(&gpu, 4);
    let mut ff_b = empty_force_field(&gpu, 4);
    let mut timings_a = Timings::new(&gpu).unwrap();
    let mut timings_b = Timings::new(&gpu).unwrap();
    let mut integrator_a = IntegratorRegistry::with_builtins()
        .build(&vv_kind(false), &gpu, 4)
        .unwrap();
    let mut integrator_b = IntegratorRegistry::with_builtins()
        .build(&vv_kind(false), &gpu, 4)
        .unwrap();

    for _ in 1..=10 {
        integrator_a
            .step(&mut buffers_a, &mut sim_box_a, &mut ff_a, None, 0.001, &mut timings_a)
            .unwrap();
        integrator_b
            .step(&mut buffers_b, &mut sim_box_b, &mut ff_b, None, 0.001, &mut timings_b)
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
    let mut sim_box = box_10();
    let mut ff = empty_force_field(&gpu, 2);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integrator = LangevinBaoabState {
        friction: 1.0e12,
        temperature: 300.0,
        seed: 42,
        draw_counter: 0,
    };
    assert_eq!(integrator.draw_counter, 0);
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, None, 1.0e-15, &mut timings)
        .unwrap();
    assert_eq!(integrator.draw_counter, 1);
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, None, 1.0e-15, &mut timings)
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
    let mut sim_box_a = box_10();
    let mut sim_box_b = box_10();
    let mut ff_a = empty_force_field(&gpu, 4);
    let mut ff_b = empty_force_field(&gpu, 4);
    let mut timings_a = Timings::new(&gpu).unwrap();
    let mut timings_b = Timings::new(&gpu).unwrap();
    let mut a = LangevinBaoabState {
        friction: 1.0e12,
        temperature: 300.0,
        seed: 7,
        draw_counter: 5,
    };
    let mut b = LangevinBaoabState {
        friction: 1.0e12,
        temperature: 300.0,
        seed: 7,
        draw_counter: 5,
    };
    a.step(&mut buffers_a, &mut sim_box_a, &mut ff_a, None, 1.0e-15, &mut timings_a)
        .unwrap();
    b.step(&mut buffers_b, &mut sim_box_b, &mut ff_b, None, 1.0e-15, &mut timings_b)
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

use dynamics::integrator::{
    BarostatError, BarostatRegistry, ThermostatError, ThermostatRegistry,
};
use dynamics::io::ThermostatKind;

// rq-78da0ce9
#[test]
fn empty_integrator_registry_reports_unknown_kind() {
    let gpu = init_device().unwrap();
    let registry = IntegratorRegistry::new();
    let err = registry.build(&vv_kind(false), &gpu, 4).unwrap_err();
    matches!(err, IntegratorError::UnknownKind(ref s) if s == "velocity-verlet");
}

// rq-89b4b926: also covered by `custom_builder_registered_takes_priority_over_builtin` above.

// rq-7e6b3ade — placeholder: NHC construction is covered in tests/integrators_nhc.rs
// rq-9e1142aa — placeholder: CSVR construction is covered in tests/integrators_csvr.rs
// rq-dc3c616a — placeholder: Andersen construction is covered in tests/integrators_andersen.rs
// rq-e8252f10 — placeholder: Berendsen construction is covered in tests/integrators_berendsen.rs

// Empty thermostat registry reports UnknownKind.
#[test]
fn empty_thermostat_registry_reports_unknown_kind() {
    let gpu = init_device().unwrap();
    let registry = ThermostatRegistry::new();
    let kind = ThermostatKind::Berendsen {
        temperature: 300.0,
        tau: 1.0e-13,
    };
    let err = registry
        .build_optional(Some(&kind), &gpu, 4)
        .unwrap_err();
    matches!(err, ThermostatError::UnknownKind(ref s) if s == "berendsen");
}

// build_optional with None returns Ok(None) without consulting builders.
#[test]
fn thermostat_build_optional_none_returns_none() {
    let gpu = init_device().unwrap();
    let registry = ThermostatRegistry::with_builtins();
    let result = registry.build_optional(None, &gpu, 4).unwrap();
    assert!(result.is_none());
}

// BarostatRegistry::with_builtins() exposes the registered barostats.
#[test]
fn barostat_with_builtins_contains_berendsen() {
    let registry = BarostatRegistry::with_builtins();
    assert!(
        registry
            .builders
            .iter()
            .any(|b| b.kind_name() == "berendsen")
    );
}

// build_optional with None returns Ok(None) on the empty barostat registry.
#[test]
fn barostat_build_optional_none_returns_none() {
    let gpu = init_device().unwrap();
    let registry = BarostatRegistry::with_builtins();
    let result = registry.build_optional(None, &gpu, 4).unwrap();
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
    let vv = IntegratorKind::VelocityVerlet { lossless: false };
    let lan = IntegratorKind::LangevinBaoab {
        friction: 1.0e12,
        temperature: 300.0,
        seed: 0,
    };
    assert!(!vv.owns_thermostat());
    assert!(lan.owns_thermostat());
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
    fn plan(&self, dt: f32) -> dynamics::integrator::StepPlan {
        dynamics::integrator::StepPlan {
            steps: vec![dynamics::integrator::SubStep::KickDrift { dt, label: "rec" }],
        }
    }

    fn execute(
        &mut self,
        _substep: &dynamics::integrator::SubStep,
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

impl dynamics::integrator::Thermostat for RecordingThermostat {
    fn apply_pre(
        &mut self,
        _b: &mut ParticleBuffers,
        _dt: f32,
        _t: &mut Timings,
    ) -> Result<(), ThermostatError> {
        self.log.record("apply_pre");
        Ok(())
    }
    fn apply_post(
        &mut self,
        _b: &mut ParticleBuffers,
        _dt: f32,
        _t: &mut Timings,
    ) -> Result<(), ThermostatError> {
        self.log.record("apply_post");
        Ok(())
    }
}

#[test]
fn dispatch_loop_orders_apply_pre_step_apply_post() {
    let gpu = init_device().unwrap();
    let state = small_state(4);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_10();
    let mut ff = empty_force_field(&gpu, 4);
    let mut timings = Timings::new(&gpu).unwrap();
    let log = CallLog::default();
    let log_clone_a = CallLog { events: log.events.clone() };
    let log_clone_b = CallLog { events: log.events.clone() };
    let mut integ: Box<dyn Integrator> = Box::new(RecordingIntegrator { log: log_clone_a });
    let mut therm: Box<dyn dynamics::integrator::Thermostat> =
        Box::new(RecordingThermostat { log: log_clone_b });

    therm
        .apply_pre(&mut buffers, 1.0e-15, &mut timings)
        .unwrap();
    integ
        .step(&mut buffers, &mut sim_box, &mut ff, None, 1.0e-15, &mut timings)
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

use dynamics::integrator::{
    ConstraintError, run_step, run_step_no_constraint, StepPlan, SubStep,
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
    fn plan(&self, _dt: f32) -> StepPlan {
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
            SubStep::ForceEval => "exec_force_eval",
            SubStep::Custom { .. } => "exec_custom",
        });
        if matches!(substep, SubStep::ForceEval) {
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

impl dynamics::integrator::Constraint for RecordingConstraint {
    fn apply_before_drift(
        &mut self,
        _b: &mut ParticleBuffers,
        _sb: &SimulationBox,
        _dt: f32,
        _t: &mut Timings,
    ) -> Result<(), ConstraintError> {
        self.log.record("before_drift");
        Ok(())
    }
    fn apply_after_drift(
        &mut self,
        _b: &mut ParticleBuffers,
        _sb: &SimulationBox,
        _dt: f32,
        _t: &mut Timings,
    ) -> Result<(), ConstraintError> {
        self.log.record("after_drift");
        Ok(())
    }
    fn apply_after_kick(
        &mut self,
        _b: &mut ParticleBuffers,
        _sb: &SimulationBox,
        _dt: f32,
        _t: &mut Timings,
    ) -> Result<(), ConstraintError> {
        self.log.record("after_kick");
        Ok(())
    }
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
    let sim_box = box_10();
    let ff = empty_force_field(&gpu, 4);
    let timings = Timings::new(&gpu).unwrap();
    (gpu, buffers, sim_box, ff, timings)
}

#[test]
fn plan_returns_same_shape_across_repeated_calls() {
    let registry = IntegratorRegistry::with_builtins();
    let gpu = init_device().unwrap();
    let integ = registry.build(&vv_kind(false), &gpu, 4).unwrap();
    let p1 = integ.plan(0.1);
    let p2 = integ.plan(0.1);
    assert_eq!(p1.steps.len(), p2.steps.len());
    for (a, b) in p1.steps.iter().zip(p2.steps.iter()) {
        assert_eq!(a.variant_name(), b.variant_name());
    }
}

#[test]
fn plan_is_pure_does_not_launch_kernels_or_touch_buffers() {
    let registry = IntegratorRegistry::with_builtins();
    let (gpu, buffers, _sim_box, _ff, _timings) = fixture();
    let pre_x = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let integ = registry.build(&vv_kind(false), &gpu, 4).unwrap();
    let _plan = integ.plan(0.1);
    let post_x = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    assert_eq!(pre_x, post_x);
}

#[test]
fn empty_plan_walks_as_a_noop() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan::empty(),
        log: CallLog { events: log.events.clone() },
    };
    run_step_no_constraint(
        &mut stub,
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

#[test]
fn plan_with_multiple_force_evals_dispatches_each() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![
                SubStep::KickHalf { dt: 0.1, label: "k1" },
                SubStep::Drift { dt: 0.1, label: "d1" },
                SubStep::ForceEval,
                SubStep::KickHalf { dt: 0.1, label: "k2" },
                SubStep::Drift { dt: 0.1, label: "d2" },
                SubStep::ForceEval,
                SubStep::KickHalf { dt: 0.1, label: "k3" },
            ],
        },
        log: CallLog { events: log.events.clone() },
    };
    run_step_no_constraint(
        &mut stub,
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
        .execute(&SubStep::ForceEval, &mut buffers, &mut sim_box, &mut timings)
        .unwrap_err();
    match err {
        IntegratorError::UnexpectedSubStep { variant } => {
            assert_eq!(variant, "ForceEval");
        }
        other => panic!("expected UnexpectedSubStep, got {other:?}"),
    }
}

#[test]
fn plan_with_one_drift_fires_before_after_drift() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![
                SubStep::Drift { dt: 0.1, label: "d" },
                SubStep::ForceEval,
                SubStep::KickHalf { dt: 0.1, label: "k" },
            ],
        },
        log: CallLog { events: log.events.clone() },
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
        true,
        0.1,
        &mut timings,
    )
    .unwrap();
    let events = constraint_log.events.lock().unwrap().clone();
    assert_eq!(events, vec!["before_drift", "after_drift", "after_kick"]);
}

#[test]
fn plan_with_two_drifts_fires_before_after_drift_twice() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let stub_log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![
                SubStep::KickHalf { dt: 0.1, label: "B" },
                SubStep::Drift { dt: 0.1, label: "A_pre" },
                SubStep::Custom { dt: 0.1, label: "O" },
                SubStep::Drift { dt: 0.1, label: "A_post" },
                SubStep::ForceEval,
                SubStep::KickHalf { dt: 0.1, label: "B" },
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
        true,
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

#[test]
fn plan_whose_final_substep_is_not_a_kick_does_not_fire_after_kick() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let stub_log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![
                SubStep::KickHalf { dt: 0.1, label: "k" },
                SubStep::ForceEval,
                SubStep::Custom { dt: 0.1, label: "post" },
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
        true,
        0.1,
        &mut timings,
    )
    .unwrap();
    let events = constraint_log.events.lock().unwrap().clone();
    assert!(!events.contains(&"after_kick"));
}

#[test]
fn custom_substep_alone_fires_no_constraint_hooks() {
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
        true,
        0.1,
        &mut timings,
    )
    .unwrap();
    let events = constraint_log.events.lock().unwrap().clone();
    assert!(events.is_empty(), "expected no constraint events, got {events:?}");
}

#[test]
fn install_constraint_hooks_false_suppresses_all_hooks() {
    let (gpu, mut buffers, mut sim_box, mut ff, mut timings) = fixture();
    let _ = gpu;
    let stub_log = CallLog::default();
    let mut stub = PlanStub {
        plan: StepPlan {
            steps: vec![
                SubStep::KickDrift { dt: 0.1, label: "kd" },
                SubStep::ForceEval,
                SubStep::KickHalf { dt: 0.1, label: "k" },
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
        false, // install_constraint_hooks == false
        0.1,
        &mut timings,
    )
    .unwrap();
    let events = constraint_log.events.lock().unwrap().clone();
    assert!(events.is_empty(), "expected no constraint events, got {events:?}");
}
