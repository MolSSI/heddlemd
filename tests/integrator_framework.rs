use std::sync::Arc;

use cudarc::driver::CudaDevice;
use dynamics::forces::{BondList, ExclusionList, ForceField};
use dynamics::gpu::{ParticleBuffers, init_device};
use dynamics::integrator::{
    Integrator, IntegratorBuilder, IntegratorError, IntegratorRegistry, LangevinBaoabBuilder,
    VelocityVerletBuilder,
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
        vec![0u32; n],
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
    SimulationBox::new_orthorhombic(10.0, 10.0, 10.0).unwrap()
}

fn empty_force_field(device: Arc<CudaDevice>, n: usize) -> ForceField {
    ForceField::new(
        device,
        n,
        &box_10(),
        &[],
        &[],
        &[],
        &BondList::empty(n),
        &ExclusionList::empty(n),
        &NeighborListConfig::AllPairs,
    )
    .unwrap()
}

// rq-e02917c3
#[test]
fn construct_vv_lossy_via_registry() {
    let device = init_device().unwrap();
    let registry = IntegratorRegistry::with_builtins();
    let _integrator = registry.build(&vv_kind(false), device, 4).unwrap();
}

// rq-db78448e
#[test]
fn construct_vv_lossless_via_registry() {
    let device = init_device().unwrap();
    let registry = IntegratorRegistry::with_builtins();
    let state = small_state(4);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut sim_box = box_10();
    let mut ff = empty_force_field(device.clone(), 4);
    let mut timings = Timings::new(device.clone()).unwrap();
    let mut integrator = registry.build(&vv_kind(true), device, 4).unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, 0.1, 1, &mut timings)
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
    let device = init_device().unwrap();
    let registry = IntegratorRegistry::with_builtins();
    let _integrator = registry.build(&langevin_kind(42), device, 4).unwrap();
}

// rq-48fd88ed
#[test]
fn construct_with_particle_count_zero() {
    let device = init_device().unwrap();
    let registry = IntegratorRegistry::with_builtins();
    let _integrator = registry.build(&vv_kind(true), device, 0).unwrap();
}

#[test]
fn registry_without_matching_builder_reports_unknown_kind() {
    let device = init_device().unwrap();
    let registry = IntegratorRegistry::new();
    let err = registry.build(&vv_kind(false), device, 4).unwrap_err();
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
        _device: Arc<CudaDevice>,
        _particle_count: usize,
        _kind: &IntegratorKind,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        Ok(Box::new(StubIntegrator))
    }
}

impl Integrator for StubIntegrator {
    fn step(
        &mut self,
        _buffers: &mut ParticleBuffers,
        _sim_box: &mut SimulationBox,
        _force_field: &mut ForceField,
        _dt: f32,
        _step_index: u64,
        _timings: &mut Timings,
    ) -> Result<(), IntegratorError> {
        Ok(())
    }
}

#[test]
fn custom_builder_registered_takes_priority_over_builtin() {
    let device = init_device().unwrap();
    let mut registry = IntegratorRegistry::new();
    registry.register(Box::new(StubBuilder));
    registry.register(Box::new(VelocityVerletBuilder));
    registry.register(Box::new(LangevinBaoabBuilder));
    // Stub appears first, so velocity-verlet routes to it.
    let state = small_state(4);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut sim_box = box_10();
    let mut ff = empty_force_field(device.clone(), 4);
    let mut timings = Timings::new(device.clone()).unwrap();
    let mut integrator = registry.build(&vv_kind(false), device, 4).unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, 0.1, 1, &mut timings)
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
    let device = init_device().unwrap();
    let state = ParticleState::new(
        Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        Vec::new(), None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut sim_box = box_10();
    let mut ff = empty_force_field(device.clone(), 0);
    let mut timings = Timings::new(device.clone()).unwrap();
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&vv_kind(false), device, 0)
        .unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, 0.1, 1, &mut timings)
        .unwrap();
    let report = timings.finalize().unwrap();
    assert!(report.stages.is_empty());
}

// rq-2980a672
#[test]
fn vv_step_launches_kick_drift_force_and_kick() {
    let device = init_device().unwrap();
    // Build a state with non-zero velocities so the drift visibly moves
    // positions during step() even with zero forces.
    let n = 4;
    let mut state = small_state(n);
    state.velocities_x = (0..n).map(|i| 0.5 + i as f32 * 0.1).collect();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut sim_box = box_10();
    let mut ff = empty_force_field(device.clone(), n);
    let mut timings = Timings::new(device.clone()).unwrap();
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&vv_kind(false), device.clone(), n)
        .unwrap();
    let snap_positions = state.positions_x.clone();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, 0.1, 1, &mut timings)
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
    let device = init_device().unwrap();
    let state = small_state(4);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut sim_box = box_10();
    let mut ff = empty_force_field(device.clone(), 4);
    let mut timings = Timings::new(device.clone()).unwrap();
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&vv_kind(true), device, 4)
        .unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, 0.1, 1, &mut timings)
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
    // KernelStage::LjPairForce was triggered exactly once per step() call.
    use dynamics::io::config::{PairInteractionConfig, ParticleTypeConfig};
    let device = init_device().unwrap();
    let state = small_state(4);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut sim_box = box_10();
    let mut ff = ForceField::new(
        device.clone(),
        4,
        &sim_box,
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0 }],
        &[PairInteractionConfig {
            between: ("Ar".to_string(), "Ar".to_string()),
            potential: "lennard-jones".to_string(),
            sigma: 1.0,
            epsilon: 1.0,
            cutoff: 1.0,
        }],
        &[],
        &BondList::empty(4),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&vv_kind(false), device, 4)
        .unwrap();
    integrator
        .step(&mut buffers, &mut sim_box, &mut ff, 0.001, 1, &mut timings)
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
fn step_index_propagates_to_langevin() {
    let device = init_device().unwrap();
    let state = small_state(2);
    let mut buffers_a = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut buffers_b = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut sim_box_a = box_10();
    let mut sim_box_b = box_10();
    let mut ff_a = empty_force_field(device.clone(), 2);
    let mut ff_b = empty_force_field(device.clone(), 2);
    let mut timings_a = Timings::new(device.clone()).unwrap();
    let mut timings_b = Timings::new(device.clone()).unwrap();
    let mut integrator_a = IntegratorRegistry::with_builtins()
        .build(&langevin_kind(1), device.clone(), 2)
        .unwrap();
    let mut integrator_b = IntegratorRegistry::with_builtins()
        .build(&langevin_kind(1), device.clone(), 2)
        .unwrap();
    integrator_a
        .step(&mut buffers_a, &mut sim_box_a, &mut ff_a, 1.0e-15, 1, &mut timings_a)
        .unwrap();
    integrator_b
        .step(&mut buffers_b, &mut sim_box_b, &mut ff_b, 1.0e-15, 2, &mut timings_b)
        .unwrap();
    let mut state_a = state.clone();
    let mut state_b = state.clone();
    state_a.download_from(&buffers_a).unwrap();
    state_b.download_from(&buffers_b).unwrap();
    assert_ne!(state_a.velocities_x, state_b.velocities_x);
}

// rq-706001ec
#[test]
fn two_independent_runs_byte_identical() {
    let device = init_device().unwrap();
    let state = small_state(4);

    let mut buffers_a = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut buffers_b = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut sim_box_a = box_10();
    let mut sim_box_b = box_10();
    let mut ff_a = empty_force_field(device.clone(), 4);
    let mut ff_b = empty_force_field(device.clone(), 4);
    let mut timings_a = Timings::new(device.clone()).unwrap();
    let mut timings_b = Timings::new(device.clone()).unwrap();
    let mut integrator_a = IntegratorRegistry::with_builtins()
        .build(&vv_kind(false), device.clone(), 4)
        .unwrap();
    let mut integrator_b = IntegratorRegistry::with_builtins()
        .build(&vv_kind(false), device.clone(), 4)
        .unwrap();

    for step in 1..=10 {
        integrator_a
            .step(&mut buffers_a, &mut sim_box_a, &mut ff_a, 0.001, step, &mut timings_a)
            .unwrap();
        integrator_b
            .step(&mut buffers_b, &mut sim_box_b, &mut ff_b, 0.001, step, &mut timings_b)
            .unwrap();
    }

    let mut state_a = state.clone();
    let mut state_b = state.clone();
    state_a.download_from(&buffers_a).unwrap();
    state_b.download_from(&buffers_b).unwrap();
    assert_eq!(state_a.positions_x, state_b.positions_x);
    assert_eq!(state_a.velocities_x, state_b.velocities_x);
}

// Silence the unused-name lint for the imported KernelStage variant set.
#[test]
fn _imports_used() {
    let _ = KernelStage::LangevinKickHalf;
}
