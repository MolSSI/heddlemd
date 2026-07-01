// rq-3d5f2e98 — Constraint framework dispatch tests: ConstraintList::subset,
// per-builder fan-out, slot firing order, empty-bucket skip, composite
// short-circuit.

use std::sync::{Arc, Mutex};

use cudarc::driver::CudaDevice;

use heddle_md::forces::{ConstraintGroup, ConstraintList, DihedralList, GroupConstraint};
use heddle_md::gpu::{GpuContext, ParticleBuffers, init_device};
use heddle_md::integrator::{
    Constraint, ConstraintBuilder, ConstraintError, ConstraintRegistry,
};
use heddle_md::io::config::{ConfigError, NamedSlotConfig};
use heddle_md::pbc::SimulationBox;
use heddle_md::state::ParticleState;
use heddle_md::timings::Timings;
use heddle_md::precision::Real;

// --- ConstraintList::subset -----------------------------------------------

fn sample_list_three_groups() -> ConstraintList {
    let groups = vec![
        ConstraintGroup {
            atom_offset: 0,
            atom_count: 3,
            constraint_offset: 0,
            constraint_count: 1,
            constraint_type_index: 7,
        },
        ConstraintGroup {
            atom_offset: 3,
            atom_count: 3,
            constraint_offset: 1,
            constraint_count: 1,
            constraint_type_index: 11,
        },
        ConstraintGroup {
            atom_offset: 6,
            atom_count: 3,
            constraint_offset: 2,
            constraint_count: 1,
            constraint_type_index: 13,
        },
    ];
    let group_atoms = vec![12, 14, 17, 90, 91, 92, 50, 51, 52];
    let group_constraints = vec![
        GroupConstraint { local_i: 0, local_j: 1, r0: 1.0 },
        GroupConstraint { local_i: 0, local_j: 2, r0: 2.0 },
        GroupConstraint { local_i: 1, local_j: 2, r0: 3.0 },
    ];
    ConstraintList {
        groups,
        group_atoms,
        group_constraints,
        particle_count: 100,
    }
}

// rq-de35dea7
#[test]
fn subset_preserves_global_atom_indices_and_particle_count() {
    let list = sample_list_three_groups();
    let sub = list.subset(&[1]);
    assert_eq!(sub.particle_count, 100);
    assert_eq!(sub.group_atoms, vec![90, 91, 92]);
    assert_eq!(sub.groups.len(), 1);
    assert_eq!(sub.groups[0].atom_offset, 0);
    assert_eq!(sub.groups[0].atom_count, 3);
    assert_eq!(sub.groups[0].constraint_offset, 0);
    assert_eq!(sub.groups[0].constraint_count, 1);
    assert_eq!(sub.groups[0].constraint_type_index, 11);
}

// rq-721e0fdb rq-744ddd67
#[test]
fn subset_with_empty_indices_yields_empty_list_preserving_particle_count() {
    let list = sample_list_three_groups();
    let sub = list.subset(&[]);
    assert!(sub.is_empty());
    assert!(sub.group_atoms.is_empty());
    assert!(sub.group_constraints.is_empty());
    assert_eq!(sub.particle_count, 100);
}

// rq-3c6753bc
#[test]
fn subset_preserves_supplied_index_order() {
    let mut list = sample_list_three_groups();
    // Replace group_atoms with one element per group for an unambiguous
    // ordering check.
    list.groups = vec![
        ConstraintGroup {
            atom_offset: 0,
            atom_count: 1,
            constraint_offset: 0,
            constraint_count: 0,
            constraint_type_index: 0,
        },
        ConstraintGroup {
            atom_offset: 1,
            atom_count: 1,
            constraint_offset: 0,
            constraint_count: 0,
            constraint_type_index: 0,
        },
        ConstraintGroup {
            atom_offset: 2,
            atom_count: 1,
            constraint_offset: 0,
            constraint_count: 0,
            constraint_type_index: 0,
        },
    ];
    list.group_atoms = vec![10, 20, 30];
    list.group_constraints.clear();

    let sub = list.subset(&[2, 0]);
    assert_eq!(sub.groups.len(), 2);
    assert_eq!(sub.group_atoms, vec![30, 10]);
}

// --- Recording stub builder + slot ----------------------------------------

#[derive(Debug, Default)]
struct Recorder {
    events: Vec<(String, &'static str)>,
    build_received_group_count: Vec<(String, usize)>,
    fail_on_after_drift_for: Option<String>,
}

#[derive(Debug)]
struct RecordingConstraint {
    kind: String,
    recorder: Arc<Mutex<Recorder>>,
    group_count: usize,
}

impl Constraint for RecordingConstraint {
    fn apply_before_drift(
        &mut self,
        _buffers: &mut ParticleBuffers,
        _sim_box: &SimulationBox,
        _dt: Real,
        _timings: &mut Timings,
    ) -> Result<(), ConstraintError> {
        self.recorder
            .lock()
            .unwrap()
            .events
            .push((self.kind.clone(), "before_drift"));
        Ok(())
    }
    fn apply_after_drift(
        &mut self,
        _buffers: &mut ParticleBuffers,
        _sim_box: &SimulationBox,
        _dt: Real,
        _timings: &mut Timings,
    ) -> Result<(), ConstraintError> {
        let mut r = self.recorder.lock().unwrap();
        r.events.push((self.kind.clone(), "after_drift"));
        if r.fail_on_after_drift_for.as_deref() == Some(self.kind.as_str()) {
            return Err(ConstraintError::UnsupportedKind(format!(
                "stub-failure-from-{}",
                self.kind
            )));
        }
        Ok(())
    }
    fn apply_after_kick(
        &mut self,
        _buffers: &mut ParticleBuffers,
        _sim_box: &SimulationBox,
        _dt: Real,
        _timings: &mut Timings,
    ) -> Result<(), ConstraintError> {
        self.recorder
            .lock()
            .unwrap()
            .events
            .push((self.kind.clone(), "after_kick"));
        Ok(())
    }
    fn group_count(&self) -> usize {
        self.group_count
    }
}

#[derive(Debug, Clone)]
struct StubBuilder {
    kind: &'static str,
    recorder: Arc<Mutex<Recorder>>,
}

    impl heddle_md::registry::KindedBuilder for StubBuilder {
    fn kind_name(&self) -> &'static str {
        self.kind
    }    }

impl ConstraintBuilder for StubBuilder {
    fn validate_params(&self, _params: &toml::Value) -> Result<(), ConfigError> {
        Ok(())
    }
    fn expected_atom_count(&self, _params: &toml::Value) -> usize {
        1
    }
    fn validate_group_shape(
        &self,
        _group_index: usize,
        _atoms: &[u32],
        _constraints: &[GroupConstraint],
        _params: &toml::Value,
        _masses: &[Real],
    ) -> Result<(), ConstraintError> {
        Ok(())
    }
    fn build(
        &self,
        _device: Arc<CudaDevice>,
        _gpu: &GpuContext,
        _particle_count: usize,
        list: &ConstraintList,
        _masses: &[Real],
        _constraint_types: &[NamedSlotConfig],
    ) -> Result<Box<dyn Constraint>, ConstraintError> {
        let group_count = list.groups.len();
        self.recorder
            .lock()
            .unwrap()
            .build_received_group_count
            .push((self.kind.to_string(), group_count));
        Ok(Box::new(RecordingConstraint {
            kind: self.kind.to_string(),
            recorder: self.recorder.clone(),
            group_count,
        }))
    }
}

fn stub_type(name: &str, kind: &str) -> NamedSlotConfig {
    NamedSlotConfig::from_params_str(name, kind, "")
}

// `build_optional` calls `device.clone()` on the GpuContext to pass to
// each builder. We need a real device; reuse the global `init_device`.
fn one_atom_state() -> ParticleState {
    ParticleState::new(
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![1.0],
        vec![0.0],
        vec![0u32; 1],
        None,
        None,
    )
    .unwrap()
}

fn big_box(gpu: &heddle_md::gpu::GpuContext) -> SimulationBox {
    SimulationBox::new(&gpu.device, 1.0e4, 1.0e4, 1.0e4, 0.0, 0.0, 0.0).unwrap()
}

fn list_of_kinds(kinds: &[(usize, &str)], constraint_types: &[NamedSlotConfig]) -> ConstraintList {
    // Build a ConstraintList with one group per entry. Each entry is
    // (constraint_type_index_into_constraint_types, kind_label_for_reference).
    // Atoms are sequential.
    let mut groups = Vec::with_capacity(kinds.len());
    let mut group_atoms = Vec::with_capacity(kinds.len());
    for (i, &(ti, _kind)) in kinds.iter().enumerate() {
        groups.push(ConstraintGroup {
            atom_offset: i as u32,
            atom_count: 1,
            constraint_offset: 0,
            constraint_count: 0,
            constraint_type_index: ti as u32,
        });
        group_atoms.push(i as u32);
    }
    // Sanity: each ti must reference a constraint_types entry whose
    // kind matches the label.
    for (ti, kind) in kinds {
        assert_eq!(constraint_types[*ti].kind, *kind);
    }
    ConstraintList {
        groups,
        group_atoms,
        group_constraints: Vec::new(),
        particle_count: kinds.len(),
    }
}

// --- Fan-out scenarios ----------------------------------------------------

// rq-6abcd773
#[test]
fn single_registered_kind_on_single_kind_topology_returns_bare_slot() {
    let gpu = init_device().unwrap();
    let recorder = Arc::new(Mutex::new(Recorder::default()));
    let mut registry = ConstraintRegistry::new();
    registry.register(Box::new(StubBuilder {
        kind: "stub-a",
        recorder: recorder.clone(),
    }));
    let cts = vec![stub_type("A1", "stub-a"), stub_type("A2", "stub-a")];
    let list = list_of_kinds(&[(0, "stub-a"), (1, "stub-a")], &cts);
    let slot = registry
        .build_optional(&list, &gpu, 2, &[1.0, 1.0], &cts)
        .unwrap()
        .expect("non-empty list produces a slot");
    assert_eq!(slot.group_count(), 2);
    // Exactly one build invocation; the bare slot is the builder's
    // direct output.
    let r = recorder.lock().unwrap();
    assert_eq!(r.build_received_group_count, vec![("stub-a".to_string(), 2)]);
}

// rq-135a6e30
#[test]
fn two_registered_kinds_on_mixed_topology_fan_out() {
    let gpu = init_device().unwrap();
    let recorder = Arc::new(Mutex::new(Recorder::default()));
    let mut registry = ConstraintRegistry::new();
    registry.register(Box::new(StubBuilder {
        kind: "stub-a",
        recorder: recorder.clone(),
    }));
    registry.register(Box::new(StubBuilder {
        kind: "stub-b",
        recorder: recorder.clone(),
    }));
    let cts = vec![
        stub_type("A1", "stub-a"),
        stub_type("A2", "stub-a"),
        stub_type("B1", "stub-b"),
    ];
    // Two stub-a groups, one stub-b group.
    let list = list_of_kinds(
        &[(0, "stub-a"), (1, "stub-a"), (2, "stub-b")],
        &cts,
    );
    let mut slot = registry
        .build_optional(&list, &gpu, 3, &[1.0, 1.0, 1.0], &cts)
        .unwrap()
        .expect("non-empty list produces a slot");
    assert_eq!(slot.group_count(), 3);
    // Each builder's build() received the right sub-list size.
    let r = recorder.lock().unwrap();
    assert_eq!(
        r.build_received_group_count,
        vec![("stub-a".to_string(), 2), ("stub-b".to_string(), 1)]
    );
    drop(r);
    // Composite forwards hooks to every inner slot.
    let mut buffers = ParticleBuffers::new(&gpu, &one_atom_state()).unwrap();
    let sb = big_box(&gpu);
    let mut t = Timings::new(&gpu).unwrap();
    slot.apply_before_drift(&mut buffers, &sb, 1.0, &mut t).unwrap();
    let r2 = recorder.lock().unwrap();
    let kinds_called: Vec<&str> = r2
        .events
        .iter()
        .filter(|(_, e)| *e == "before_drift")
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(kinds_called, vec!["stub-a", "stub-b"]);
}

// rq-08ece93d
#[test]
fn slot_firing_order_matches_registration_order_and_reverses_when_registration_reverses() {
    fn run(register_a_first: bool) -> Vec<String> {
        let gpu = init_device().unwrap();
        let recorder = Arc::new(Mutex::new(Recorder::default()));
        let mut registry = ConstraintRegistry::new();
        if register_a_first {
            registry.register(Box::new(StubBuilder {
                kind: "alpha",
                recorder: recorder.clone(),
            }));
            registry.register(Box::new(StubBuilder {
                kind: "beta",
                recorder: recorder.clone(),
            }));
        } else {
            registry.register(Box::new(StubBuilder {
                kind: "beta",
                recorder: recorder.clone(),
            }));
            registry.register(Box::new(StubBuilder {
                kind: "alpha",
                recorder: recorder.clone(),
            }));
        }
        let cts = vec![stub_type("A", "alpha"), stub_type("B", "beta")];
        let list = list_of_kinds(&[(0, "alpha"), (1, "beta")], &cts);
        let mut slot = registry
            .build_optional(&list, &gpu, 2, &[1.0, 1.0], &cts)
            .unwrap()
            .unwrap();
        let mut buffers = ParticleBuffers::new(&gpu, &one_atom_state()).unwrap();
        let sb = big_box(&gpu);
        let mut t = Timings::new(&gpu).unwrap();
        slot.apply_after_drift(&mut buffers, &sb, 1.0, &mut t).unwrap();
        recorder
            .lock()
            .unwrap()
            .events
            .iter()
            .filter(|(_, e)| *e == "after_drift")
            .map(|(k, _)| k.clone())
            .collect()
    }
    assert_eq!(run(true), vec!["alpha", "beta"]);
    assert_eq!(run(false), vec!["beta", "alpha"]);
}

// rq-051a8191
#[test]
fn empty_bucket_for_registered_kind_skips_its_builder() {
    let gpu = init_device().unwrap();
    let recorder = Arc::new(Mutex::new(Recorder::default()));
    let mut registry = ConstraintRegistry::new();
    registry.register(Box::new(StubBuilder {
        kind: "stub-a",
        recorder: recorder.clone(),
    }));
    registry.register(Box::new(StubBuilder {
        kind: "stub-b",
        recorder: recorder.clone(),
    }));
    let cts = vec![stub_type("A", "stub-a"), stub_type("B", "stub-b")];
    // Only stub-a groups; stub-b's bucket is empty.
    let list = list_of_kinds(&[(0, "stub-a"), (0, "stub-a")], &cts);
    let slot = registry
        .build_optional(&list, &gpu, 2, &[1.0, 1.0], &cts)
        .unwrap()
        .unwrap();
    assert_eq!(slot.group_count(), 2);
    let r = recorder.lock().unwrap();
    // Only stub-a's build() was called.
    assert_eq!(r.build_received_group_count, vec![("stub-a".to_string(), 2)]);
}

// rq-82cf45a2
#[test]
fn composite_short_circuits_on_first_inner_error() {
    let gpu = init_device().unwrap();
    let recorder = Arc::new(Mutex::new(Recorder::default()));
    recorder.lock().unwrap().fail_on_after_drift_for = Some("stub-a".to_string());
    let mut registry = ConstraintRegistry::new();
    registry.register(Box::new(StubBuilder {
        kind: "stub-a",
        recorder: recorder.clone(),
    }));
    registry.register(Box::new(StubBuilder {
        kind: "stub-b",
        recorder: recorder.clone(),
    }));
    let cts = vec![stub_type("A", "stub-a"), stub_type("B", "stub-b")];
    let list = list_of_kinds(&[(0, "stub-a"), (1, "stub-b")], &cts);
    let mut slot = registry
        .build_optional(&list, &gpu, 2, &[1.0, 1.0], &cts)
        .unwrap()
        .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &one_atom_state()).unwrap();
    let sb = big_box(&gpu);
    let mut t = Timings::new(&gpu).unwrap();
    let err = slot
        .apply_after_drift(&mut buffers, &sb, 1.0, &mut t)
        .unwrap_err();
    match err {
        ConstraintError::UnsupportedKind(k) => assert_eq!(k, "stub-failure-from-stub-a"),
        other => panic!("expected UnsupportedKind sentinel, got {other:?}"),
    }
    let r = recorder.lock().unwrap();
    let after_drift_kinds: Vec<&str> = r
        .events
        .iter()
        .filter(|(_, e)| *e == "after_drift")
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(after_drift_kinds, vec!["stub-a"]);
}

// --- Convenience trait surface --------------------------------------------
//
// `ConstraintCapableIntegrator` is implemented by `VelocityVerletState`;
// it is *not* implemented by `LangevinBaoabState` or `MtkNptIntegrator`.
// The non-impl is enforced statically: code like
//
//     fn accept<T: heddle_md::integrator::ConstraintCapableIntegrator>(_: &T) {}
//     let lan = heddle_md::integrator::LangevinBaoabState::new(...);
//     accept(&lan);
//
// fails to compile with a trait-bound error. We exercise that compile
// barrier via the type-asserting `accept_constraint_capable` helper
// below, which is only ever called with a `VelocityVerletState`. Adding
// a call with a non-VV state would not type-check.

fn accept_constraint_capable<T: heddle_md::integrator::ConstraintCapableIntegrator>(_: &T) {}

// rq-2e028ba6
#[test]
fn velocity_verlet_state_implements_constraint_capable_integrator() {
    let gpu = init_device().unwrap();
    let state = heddle_md::integrator::VelocityVerletState::new(&gpu, 0, false).unwrap();
    accept_constraint_capable(&state);
}

// rq-a109cbb7
#[test]
fn velocity_verlet_lossless_false_accepts_constraints_now() {
    use heddle_md::integrator::ConstraintCapableIntegrator;
    let gpu = init_device().unwrap();
    let state = heddle_md::integrator::VelocityVerletState::new(&gpu, 0, false).unwrap();
    assert_eq!(state.check_accepts_constraints_now(), Ok(()));
}

// rq-ca6d04ca
// Lossless mode is only available in the default (f32) build.
#[cfg(not(feature = "f64"))]
#[test]
fn velocity_verlet_lossless_true_rejects_constraints_now() {
    use heddle_md::integrator::ConstraintCapableIntegrator;
    let gpu = init_device().unwrap();
    let state = heddle_md::integrator::VelocityVerletState::new(&gpu, 0, true).unwrap();
    assert_eq!(
        state.check_accepts_constraints_now(),
        Err("velocity-Verlet in lossless mode does not yet support constraints"),
    );
}

// rq-a4a2bf11
#[test]
fn integrator_step_ext_step_has_no_constraint_argument() {
    // The non-constraint convenience surface compiles with five
    // arguments and walks the plan without any constraint hooks. We
    // verify by running step() on a lossless-true VV (which would
    // *reject* step_with_constraint) and confirming the call succeeds.
    use heddle_md::forces::{AngleList, BondList, DihedralList, ExclusionList, ForceField, PotentialRegistry};
    use heddle_md::integrator::IntegratorStepExt;
    use heddle_md::io::config::NeighborListConfig;
    use heddle_md::state::ParticleState;
    let gpu = init_device().unwrap();
    let particle_state = ParticleState::new(
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![1.0],
        vec![0.0],
        vec![0u32; 1],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &particle_state).unwrap();
    let sim_box = SimulationBox::new(&gpu.device, 1.0e4, 1.0e4, 1.0e4, 0.0, 0.0, 0.0).unwrap();
    let mut sim_box = sim_box;
    let mut ff = ForceField::new(
        &PotentialRegistry::with_builtins(),
        &gpu,
        1,
        &sim_box,
        &[],
        &[],
        &[],
        &[],
        &[],
        None,
        &[],
        &BondList::empty(1),
        &AngleList::empty(0),
        &DihedralList::empty(0),
        &ExclusionList::empty(1),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    // Lossless VV — would reject step_with_constraint, but step()
    // has no constraint argument and runs cleanly.
    let mut state = heddle_md::integrator::VelocityVerletState::new(&gpu, 1, true).unwrap();
    state
        .step(&mut buffers, &mut sim_box, &mut ff, 1.0, &mut timings)
        .unwrap();
}

// rq-e9706f76
// Lossless mode is only available in the default (f32) build.
#[cfg(not(feature = "f64"))]
#[test]
fn step_with_constraint_short_circuits_on_lossless_velocity_verlet() {
    use heddle_md::forces::{AngleList, BondList, DihedralList, ExclusionList, ForceField, PotentialRegistry};
    use heddle_md::integrator::{IntegratorStepWithConstraintExt, StepError};
    use heddle_md::io::config::NeighborListConfig;
    use heddle_md::state::ParticleState;

    // A recording constraint that panics if any hook fires — proving
    // step_with_constraint short-circuited before the plan walk.
    #[derive(Debug)]
    struct PanickingConstraint;
    impl Constraint for PanickingConstraint {
        fn apply_before_drift(
            &mut self,
            _b: &mut ParticleBuffers,
            _sb: &SimulationBox,
            _dt: Real,
            _t: &mut Timings,
        ) -> Result<(), ConstraintError> {
            panic!("apply_before_drift should not fire for lossless VV");
        }
        fn apply_after_drift(
            &mut self,
            _b: &mut ParticleBuffers,
            _sb: &SimulationBox,
            _dt: Real,
            _t: &mut Timings,
        ) -> Result<(), ConstraintError> {
            panic!("apply_after_drift should not fire for lossless VV");
        }
        fn apply_after_kick(
            &mut self,
            _b: &mut ParticleBuffers,
            _sb: &SimulationBox,
            _dt: Real,
            _t: &mut Timings,
        ) -> Result<(), ConstraintError> {
            panic!("apply_after_kick should not fire for lossless VV");
        }
    }

    let gpu = init_device().unwrap();
    let particle_state = ParticleState::new(
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![1.0],
        vec![0.0],
        vec![0u32; 1],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &particle_state).unwrap();
    let mut sim_box = SimulationBox::new(&gpu.device, 1.0e4, 1.0e4, 1.0e4, 0.0, 0.0, 0.0).unwrap();
    let mut ff = ForceField::new(
        &PotentialRegistry::with_builtins(),
        &gpu,
        1,
        &sim_box,
        &[],
        &[],
        &[],
        &[],
        &[],
        None,
        &[],
        &BondList::empty(1),
        &AngleList::empty(0),
        &DihedralList::empty(0),
        &ExclusionList::empty(1),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut state = heddle_md::integrator::VelocityVerletState::new(&gpu, 1, true).unwrap();
    let mut constraint = PanickingConstraint;
    let err = state
        .step_with_constraint(
            &mut buffers,
            &mut sim_box,
            &mut ff,
            &mut constraint,
            1.0,
            &mut timings,
        )
        .unwrap_err();
    match err {
        StepError::IntegratorRejectsConstraint { reason } => {
            assert_eq!(reason, "velocity-Verlet in lossless mode does not yet support constraints");
        }
        other => panic!("expected IntegratorRejectsConstraint, got {other:?}"),
    }
}

// --- Empty particle_count: all four hooks are no-ops ---------------------

fn empty_particle_state() -> ParticleState {
    ParticleState::new(
        vec![], vec![], vec![],
        vec![], vec![], vec![],
        vec![], vec![], vec![],
        None, None,
    )
    .unwrap()
}

fn empty_shake_slot(
    gpu: &GpuContext,
) -> heddle_md::integrator::shake::ShakeConstraintsState {
    use heddle_md::integrator::shake::ShakeConstraintsState;
    ShakeConstraintsState::new(
        gpu.device.clone(),
        &ConstraintList::empty(0),
        &[],
        &[],
    )
    .unwrap()
}

// rq-03329010
#[test]
fn apply_before_drift_on_empty_state_is_a_noop() {
    let gpu = init_device().unwrap();
    let mut slot = empty_shake_slot(&gpu);
    let mut buffers = ParticleBuffers::new(&gpu, &empty_particle_state()).unwrap();
    let sb = big_box(&gpu);
    let mut t = Timings::new(&gpu).unwrap();
    assert_eq!(buffers.particle_count(), 0);
    use heddle_md::integrator::Constraint;
    slot.apply_before_drift(&mut buffers, &sb, 0.1, &mut t).unwrap();
    let report = t.finalize().unwrap();
    assert!(report.stages.is_empty(), "empty-state apply_before_drift launched: {:?}", report.stages);
}

// rq-129cb281
#[test]
fn apply_after_drift_on_empty_state_is_a_noop() {
    let gpu = init_device().unwrap();
    let mut slot = empty_shake_slot(&gpu);
    let mut buffers = ParticleBuffers::new(&gpu, &empty_particle_state()).unwrap();
    let sb = big_box(&gpu);
    let mut t = Timings::new(&gpu).unwrap();
    use heddle_md::integrator::Constraint;
    slot.apply_after_drift(&mut buffers, &sb, 0.1, &mut t).unwrap();
    let report = t.finalize().unwrap();
    assert!(report.stages.is_empty(), "empty-state apply_after_drift launched: {:?}", report.stages);
}

// rq-375aba37
#[test]
fn apply_after_kick_on_empty_state_is_a_noop() {
    let gpu = init_device().unwrap();
    let mut slot = empty_shake_slot(&gpu);
    let mut buffers = ParticleBuffers::new(&gpu, &empty_particle_state()).unwrap();
    let sb = big_box(&gpu);
    let mut t = Timings::new(&gpu).unwrap();
    use heddle_md::integrator::Constraint;
    slot.apply_after_kick(&mut buffers, &sb, 0.1, &mut t).unwrap();
    let report = t.finalize().unwrap();
    assert!(report.stages.is_empty(), "empty-state apply_after_kick launched: {:?}", report.stages);
}

// rq-833d83a9
#[test]
fn apply_position_projection_only_on_empty_state_is_a_noop() {
    let gpu = init_device().unwrap();
    let mut slot = empty_shake_slot(&gpu);
    let mut buffers = ParticleBuffers::new(&gpu, &empty_particle_state()).unwrap();
    let sb = big_box(&gpu);
    let mut t = Timings::new(&gpu).unwrap();
    use heddle_md::integrator::Constraint;
    slot.apply_position_projection_only(&mut buffers, &sb, &mut t).unwrap();
    let report = t.finalize().unwrap();
    assert!(report.stages.is_empty(), "empty-state apply_position_projection_only launched: {:?}", report.stages);
}

// --- ConstraintBuilder default and byte-equivalence ---------------------

// rq-fd51fccb
#[test]
fn constraint_builder_default_supports_position_projection_only_returns_true() {
    use heddle_md::integrator::shake::ShakeBuilder;
    let b = ShakeBuilder;
    let mut tbl = toml::map::Map::new();
    tbl.insert(
        "kind".to_string(),
        toml::Value::String("shake".to_string()),
    );
    tbl.insert("r0".to_string(), toml::Value::Float(1.0));
    tbl.insert("tolerance".to_string(), toml::Value::Float(1.0e-6));
    tbl.insert("max_iterations".to_string(), toml::Value::Integer(50));
    let params = toml::Value::Table(tbl);
    assert!(b.supports_position_projection_only(&params));
}

// rq-4f88e13c
#[test]
fn two_independently_constructed_constraint_lists_are_byte_identical() {
    fn build(seed: &[(usize, &str)]) -> ConstraintList {
        let mut groups = Vec::with_capacity(seed.len());
        let mut group_atoms = Vec::with_capacity(seed.len());
        for (i, &(ti, _)) in seed.iter().enumerate() {
            groups.push(ConstraintGroup {
                atom_offset: i as u32,
                atom_count: 1,
                constraint_offset: 0,
                constraint_count: 0,
                constraint_type_index: ti as u32,
            });
            group_atoms.push(i as u32);
        }
        ConstraintList {
            groups,
            group_atoms,
            group_constraints: Vec::new(),
            particle_count: seed.len(),
        }
    }
    let cl_a = build(&[(0, "stub-a"), (1, "stub-a"), (2, "stub-b")]);
    let cl_b = build(&[(0, "stub-a"), (1, "stub-a"), (2, "stub-b")]);
    assert_eq!(cl_a.groups.len(), cl_b.groups.len());
    assert_eq!(cl_a.group_atoms, cl_b.group_atoms);
    assert_eq!(cl_a.group_constraints.len(), cl_b.group_constraints.len());
    for (c1, c2) in cl_a.group_constraints.iter().zip(cl_b.group_constraints.iter()) {
        assert_eq!(c1.local_i, c2.local_i);
        assert_eq!(c1.local_j, c2.local_j);
        assert_eq!(c1.r0.to_bits(), c2.r0.to_bits());
    }
    assert_eq!(cl_a.particle_count, cl_b.particle_count);
    for (g1, g2) in cl_a.groups.iter().zip(cl_b.groups.iter()) {
        assert_eq!(g1.atom_offset, g2.atom_offset);
        assert_eq!(g1.atom_count, g2.atom_count);
        assert_eq!(g1.constraint_offset, g2.constraint_offset);
        assert_eq!(g1.constraint_count, g2.constraint_count);
        assert_eq!(g1.constraint_type_index, g2.constraint_type_index);
    }
}

// rq-fe2cb7ee
#[test]
fn empty_constraints_with_non_supporting_integrator_is_permitted() {
    use heddle_md::io::load_config;
    use std::fs;
    let dir = std::env::temp_dir().join(format!(
        "heddlemd-empty-constraints-non-supporting-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("argon.in.xyz"),
        "1\nLattice=\"1.0e-8 0 0 0 1.0e-8 0 0 0 1.0e-8\" \
         Properties=species:S:1:pos:R:3\nAr 0 0 0\n",
    )
    .unwrap();
    let cfg = r#"schema_version = 1
init = "argon.in.xyz"

[simulation]
seed = 1
temperature = 0.0

[[phase]]
name = "run"
n_steps = 1
dt = 1.0e-15

[phase.integrator]
kind = "langevin-baoab"
friction = 1.0e12
temperature = 300.0
seed = 11

[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
potential = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 1.5e-9

[neighbor_list]
mode = "all-pairs"
"#;
    let path = dir.join("argon.in.toml");
    fs::write(&path, cfg).unwrap();
    load_config(&path)
        .expect("empty [constraints] with langevin-baoab should load (no constraint slot)");
}

// =====================================================================
// Framework-level position-projection obligations
//
// Every registered constraint builder that reports
// `supports_position_projection_only(&params) == true` must provide an
// `apply_position_projection_only` that is (1) a bit-exact identity on
// the manifold, (2) a residual-bounded, idempotent retraction, and (3)
// usable as the projection step of a converging steepest-descent
// minimization. The three tests below exercise every such builtin
// (settle, shake) against one of these obligations each. A coverage
// guard forces a newly registered constraint kind to declare a fixture
// here, so the obligations cannot be silently bypassed.
// =====================================================================

// Atomic units. SPC/E water geometry: r_OH = 1.0 Å = 1.88973 a₀;
// r_HH = 1.633 Å = 3.08593 a₀.
const PROJ_R_OH: Real = 1.889_726_1;
const PROJ_R_HH: Real = 3.085_926_4;
const PROJ_M_O: Real = 29_167.43;
const PROJ_M_H: Real = 1_837.47;

// Representative single-water `[[constraint_types]]` entry for each
// supporting kind, expressed in atomic units (matching the internal
// pipeline; the slot constructors consume atomic-unit params directly).
fn settle_water_type() -> NamedSlotConfig {
    NamedSlotConfig::from_params_str(
        "W",
        "settle",
        &format!("d_OH = {}\nd_HH = {}\n", PROJ_R_OH as f64, PROJ_R_HH as f64),
    )
}

fn shake_water_type() -> NamedSlotConfig {
    NamedSlotConfig::from_params_str(
        "W",
        "shake",
        &format!(
            "atoms = 3\nconstraints = [\n  {{ i = 0, j = 1, d = {} }},\n  {{ i = 0, j = 2, d = {} }},\n  {{ i = 1, j = 2, d = {} }},\n]\n",
            PROJ_R_OH as f64, PROJ_R_OH as f64, PROJ_R_HH as f64,
        ),
    )
}

/// `(kind, constraint_types)` fixture for every registered constraint
/// kind. Cross-checked against the registry by
/// [`assert_fixtures_cover_registry`] so a new kind is forced to declare
/// one.
fn projection_fixtures() -> Vec<(&'static str, Vec<NamedSlotConfig>)> {
    vec![
        ("settle", vec![settle_water_type()]),
        ("shake", vec![shake_water_type()]),
    ]
}

/// Every builder in the default registry must have a fixture, so adding a
/// constraint kind without exercising its projection obligations fails.
fn assert_fixtures_cover_registry() {
    let registry = ConstraintRegistry::with_builtins();
    let fixtures = projection_fixtures();
    for b in registry.builders() {
        assert!(
            fixtures.iter().any(|(k, _)| *k == b.kind_name()),
            "registered constraint kind `{}` has no projection fixture; \
             add one to projection_fixtures() (and mark whether it supports \
             position-only projection)",
            b.kind_name()
        );
    }
}

/// One SPC/E water at the exact canonical geometry: atom 0 = O at
/// `(d_oh, 0, 0)`, atoms 1,2 = H at `(0, ∓r_hh/2, 0)`. Satisfies every
/// canonical constraint exactly.
fn rigid_water_state() -> ParticleState {
    let d_oh = (PROJ_R_OH * PROJ_R_OH - (PROJ_R_HH * 0.5) * (PROJ_R_HH * 0.5)).sqrt();
    ParticleState::new(
        vec![d_oh, 0.0, 0.0],
        vec![0.0, -PROJ_R_HH * 0.5, PROJ_R_HH * 0.5],
        vec![0.0, 0.0, 0.0],
        vec![0.0; 3],
        vec![0.0; 3],
        vec![0.0; 3],
        vec![PROJ_M_O, PROJ_M_H, PROJ_M_H],
        vec![0.0; 3],
        vec![0u32; 3],
        None,
        None,
    )
    .unwrap()
}

fn water_constraint_list() -> ConstraintList {
    ConstraintList {
        groups: vec![ConstraintGroup {
            atom_offset: 0,
            atom_count: 3,
            constraint_offset: 0,
            constraint_count: 3,
            constraint_type_index: 0,
        }],
        group_atoms: vec![0, 1, 2],
        group_constraints: vec![
            GroupConstraint { local_i: 0, local_j: 1, r0: PROJ_R_OH },
            GroupConstraint { local_i: 0, local_j: 2, r0: PROJ_R_OH },
            GroupConstraint { local_i: 1, local_j: 2, r0: PROJ_R_HH },
        ],
        particle_count: 3,
    }
}

/// Build the registered slot for `kind` over one rigid water, or `None`
/// if the builder does not support position-only projection.
fn build_supporting_water_slot(
    gpu: &GpuContext,
    kind: &str,
    cts: &[NamedSlotConfig],
) -> Option<Box<dyn Constraint>> {
    let registry = ConstraintRegistry::with_builtins();
    let builder = registry.lookup(kind).expect("fixture kind is registered");
    if !builder.supports_position_projection_only(&cts[0].params) {
        return None;
    }
    let list = water_constraint_list();
    let state = rigid_water_state();
    registry
        .build_optional(&list, gpu, 3, &state.masses, cts)
        .unwrap()
}

/// Min-image-free distance between two atoms, reconstructed in f64 so the
/// assertion error is dominated by the kernel's own f32 convergence, not
/// by host-side f32 round-off.
fn pdist(px: &[Real], py: &[Real], pz: &[Real], a: usize, b: usize) -> f64 {
    let dx = (px[a] - px[b]) as f64;
    let dy = (py[a] - py[b]) as f64;
    let dz = (pz[a] - pz[b]) as f64;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Largest per-atom Euclidean displacement between two position sets.
fn max_atom_displacement(
    ax: &[Real],
    ay: &[Real],
    az: &[Real],
    bx: &[Real],
    by: &[Real],
    bz: &[Real],
) -> f64 {
    (0..ax.len())
        .map(|i| {
            let dx = (ax[i] - bx[i]) as f64;
            let dy = (ay[i] - by[i]) as f64;
            let dz = (az[i] - bz[i]) as f64;
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(0.0, f64::max)
}

fn assert_water_on_manifold(px: &[Real], py: &[Real], pz: &[Real], ctx: &str) {
    let rel = |d: f64, r: f64| (d - r).abs() / r;
    let r_oh = PROJ_R_OH as f64;
    let r_hh = PROJ_R_HH as f64;
    let tol = 1.0e-6;
    assert!(rel(pdist(px, py, pz, 0, 1), r_oh) < tol, "{ctx}: O-H1 off manifold");
    assert!(rel(pdist(px, py, pz, 0, 2), r_oh) < tol, "{ctx}: O-H2 off manifold");
    assert!(rel(pdist(px, py, pz, 1, 2), r_hh) < tol, "{ctx}: H-H off manifold");
}

// rq-7d63f689
#[test]
fn projection_is_bit_exact_identity_on_manifold_for_every_supporting_kind() {
    assert_fixtures_cover_registry();
    let gpu = init_device().unwrap();
    for (kind, cts) in projection_fixtures() {
        let Some(mut slot) = build_supporting_water_slot(&gpu, kind, &cts) else {
            continue;
        };
        let state = rigid_water_state();
        let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
        let sb = big_box(&gpu);
        let mut t = Timings::new(&gpu).unwrap();
        // The water is already at the exact canonical geometry, so the
        // projection must leave every coordinate byte-identical.
        let (bx, by, bz) = buffers.download_positions().unwrap();
        slot.apply_position_projection_only(&mut buffers, &sb, &mut t).unwrap();
        let (ax, ay, az) = buffers.download_positions().unwrap();
        assert_eq!(bx, ax, "{kind}: x perturbed by projecting an on-manifold molecule");
        assert_eq!(by, ay, "{kind}: y perturbed by projecting an on-manifold molecule");
        assert_eq!(bz, az, "{kind}: z perturbed by projecting an on-manifold molecule");
    }
}

// rq-78f1f153
#[test]
fn projection_displacement_is_residual_bounded_and_idempotent_for_every_supporting_kind() {
    assert_fixtures_cover_registry();
    let gpu = init_device().unwrap();
    // Small off-manifold perturbation. Distinct per-atom directions so
    // every constraint is broken, not just one.
    let eps: Real = 2.0e-3;
    let dirs = [(1.0, -0.5, 0.3), (-0.7, 1.0, -0.4), (0.4, -0.8, 1.0)];
    for (kind, cts) in projection_fixtures() {
        let Some(mut slot) = build_supporting_water_slot(&gpu, kind, &cts) else {
            continue;
        };
        let state = rigid_water_state();
        let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
        let sb = big_box(&gpu);
        let mut t = Timings::new(&gpu).unwrap();

        // Perturb off-manifold by O(eps).
        let (mut px, mut py, mut pz) = buffers.download_positions().unwrap();
        for (i, (dx, dy, dz)) in dirs.iter().enumerate() {
            px[i] += eps * dx;
            py[i] += eps * dy;
            pz[i] += eps * dz;
        }
        buffers.upload_positions(&px, &py, &pz).unwrap();
        let (ix, iy, iz) = (px.clone(), py.clone(), pz.clone());

        slot.apply_position_projection_only(&mut buffers, &sb, &mut t).unwrap();
        let (qx, qy, qz) = buffers.download_positions().unwrap();
        assert_water_on_manifold(&qx, &qy, &qz, &format!("{kind} after projection"));

        // Residual-bounded: the move is O(eps), not a constraint-scale
        // step (~r0 ≈ 1.9 a₀, ~1000× eps). A 10× margin keeps the bound
        // discriminating while tolerating mass-weighting and geometry.
        let disp = max_atom_displacement(&ix, &iy, &iz, &qx, &qy, &qz);
        let bound = 10.0 * eps as f64;
        assert!(
            disp < bound,
            "{kind}: projection moved an atom {disp} a₀, exceeding O(eps) bound {bound}"
        );

        // Idempotent: re-projecting an already-on-manifold configuration
        // applies no further displacement.
        slot.apply_position_projection_only(&mut buffers, &sb, &mut t).unwrap();
        let (rx, ry, rz) = buffers.download_positions().unwrap();
        let second = max_atom_displacement(&qx, &qy, &qz, &rx, &ry, &rz);
        assert!(
            second < 1.0e-7,
            "{kind}: second projection moved an atom {second} a₀ (not idempotent)"
        );
    }
}

/// SI-unit `[[constraint_types]]` block for the end-to-end minimization
/// config (water with d_OH = 1.0 Å, d_HH = 1.633 Å).
fn sd_constraint_types_block(kind: &str) -> String {
    match kind {
        "settle" => "[[constraint_types]]\nname = \"W\"\nkind = \"settle\"\n\
                     d_OH = 1.0e-10\nd_HH = 1.633e-10\n"
            .to_string(),
        "shake" => "[[constraint_types]]\nname = \"W\"\nkind = \"shake\"\natoms = 3\n\
                    constraints = [\n  { i = 0, j = 1, d = 1.0e-10 },\n  \
                    { i = 0, j = 2, d = 1.0e-10 },\n  { i = 1, j = 2, d = 1.633e-10 },\n]\n"
            .to_string(),
        other => panic!(
            "no steepest-descent constraint_types block for supporting kind `{other}`; add one"
        ),
    }
}

// rq-a861d13f
#[test]
fn sd_minimization_with_constraint_converges_for_every_supporting_kind() {
    use std::fs;
    let registry = ConstraintRegistry::with_builtins();
    for (kind, cts) in projection_fixtures() {
        // Only kinds that report support participate in minimization.
        if !registry
            .lookup(kind)
            .unwrap()
            .supports_position_projection_only(&cts[0].params)
        {
            continue;
        }

        let dir = std::env::temp_dir().join(format!(
            "heddlemd-proj-min-{}-{}-{}",
            kind,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        if dir.exists() {
            let _ = fs::remove_dir_all(&dir);
        }
        fs::create_dir_all(&dir).unwrap();

        // Two rigid SPC/E waters with their oxygens 3.0 Å apart (a real
        // repulsive O–O force to minimize). Each water starts at the
        // exact canonical geometry, so every trial-step projection starts
        // from an on-manifold molecule — the case the identity property
        // must keep from collapsing the line search.
        let init = "6\n\
            Lattice=\"5.0e-9 0 0 0 5.0e-9 0 0 0 5.0e-9\" Properties=species:S:1:pos:R:3\n\
            O  -1.500000000e-10 0.000000000e0 0.000000000e0\n\
            H  -0.500000000e-10 0.000000000e0 0.000000000e0\n\
            H  -1.833800000e-10 9.426400000e-11 0.000000000e0\n\
            O   1.500000000e-10 0.000000000e0 0.000000000e0\n\
            H   2.500000000e-10 0.000000000e0 0.000000000e0\n\
            H   1.166200000e-10 9.426400000e-11 0.000000000e0\n";
        fs::write(dir.join("water.in.xyz"), init).unwrap();
        fs::write(
            dir.join("water.in.topology"),
            "[constraints]\n0 1 2 W\n3 4 5 W\n",
        )
        .unwrap();

        let cfg = format!(
            r#"schema_version = 1
units = "si"
init = "water.in.xyz"
topology = "water.in.topology"

[simulation]
seed = 1
temperature = 0.0

[[minimization]]
name = "min"

[minimization.algorithm]
kind = "steepest-descent"
initial_step = 1.0e-13
max_step = 1.0e-11
force_tolerance = 1.0e-11
energy_tolerance = 1.0e-6
max_iterations = 500

[minimization.output]
minlog_every = 1
trajectory_every = 0

[[particle_types]]
name = "O"
mass = 2.6566e-26
charge = 0.0

[[particle_types]]
name = "H"
mass = 1.6735e-27
charge = 0.0

[[pair_interactions]]
between = ["O", "O"]
potential = "lennard-jones"
sigma = 3.166e-10
epsilon = 1.080e-21
cutoff = 1.0e-9
r_switch = 1.0e-9

[[pair_interactions]]
between = ["H", "H"]
potential = "lennard-jones"
sigma = 1.0e-10
epsilon = 1.0e-30
cutoff = 1.0e-9
r_switch = 1.0e-9

[[pair_interactions]]
between = ["H", "O"]
potential = "lennard-jones"
sigma = 1.0e-10
epsilon = 1.0e-30
cutoff = 1.0e-9
r_switch = 1.0e-9

{constraints}
[neighbor_list]
mode = "all-pairs"
"#,
            constraints = sd_constraint_types_block(kind),
        );
        let cfg_path = dir.join("water.in.toml");
        fs::write(&cfg_path, cfg).unwrap();

        let summary = match heddle_md::runner::run_simulation(&cfg_path) {
            Ok(s) => s,
            Err(e) => panic!("{kind}: run_simulation failed (line search likely collapsed): {e:?}"),
        };
        assert_eq!(summary.phases.len(), 1, "{kind}: expected one phase");
        let ps = &summary.phases[0];
        assert_eq!(ps.kind, "minimization", "{kind}: wrong phase kind");
        assert!(
            ps.n_steps > 0 && ps.n_steps <= 500,
            "{kind}: iters = {} (out of range)",
            ps.n_steps
        );
        assert!(
            ps.convergence.is_some(),
            "{kind}: minimization did not converge before the step cap"
        );
    }
}

