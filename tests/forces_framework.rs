//! Pluggable potential slot framework tests.
//!
//! Implements `rqm/forces/framework.md`'s Gherkin scenarios:
//! construction, force-evaluation pipeline, trait dispatch, registry,
//! force classes, RESPA dispatch, byte-identity, combiner, neighbor-list
//! ownership, energy/virial aggregation, and AggregateLevel semantics.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use cudarc::driver::DeviceSlice;
use heddle_md::forces::{
    AggregateLevel, AngleList, Bond, BondList, CutoffHandling, ExclusionList, ForceClass,
    ForceField, ForceFieldContext, ForceFieldError, ForceLaunchBuilder, JitParticipant,
    PairForceBindContext, PairForceFragment, PairForcePotential, Potential, PotentialBuildContext,
    PotentialBuilder, PotentialRegistry, SlotOutputView,
};
use heddle_md::gpu::{GpuContext, ParticleBuffers, init_device};
use heddle_md::io::config::{
    BondTypeConfig, NeighborListConfig, PairInteractionConfig, PairPotentialParams,
    ParticleTypeConfig, SpmeConfig,
};
use heddle_md::pbc::SimulationBox;
use heddle_md::precision::Real;
use heddle_md::state::ParticleState;
use heddle_md::timings::{KernelStage, Timings, TimingsReport};

// =================================================================
// Helpers
// =================================================================

fn box_10(gpu: &heddle_md::gpu::GpuContext) -> SimulationBox {
    SimulationBox::new(&gpu.device, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0).unwrap()
}

fn ar_type() -> ParticleTypeConfig {
    ParticleTypeConfig {
        name: "Ar".to_string(),
        mass: 1.0,
        charge: 0.0,
    }
}

fn lj_pair_config() -> PairInteractionConfig {
    PairInteractionConfig {
        between: ("Ar".to_string(), "Ar".to_string()),
        cutoff: 5.0,
        r_switch: 5.0,
        potential: PairPotentialParams::LennardJones {
            sigma: 1.0,
            epsilon: 1.0,
        },
    }
}

fn morse_bond_type() -> BondTypeConfig {
    BondTypeConfig::Morse {
        name: "CC".to_string(),
        de: 1.0,
        a: 2.0,
        re: 1.0,
    }
}

fn state_n(n: usize) -> ParticleState {
    // Particles laid along x at 1.5 spacing — close enough to give a
    // non-trivial LJ force; box is 10 units.
    let pos: Vec<Real> = (0..n).map(|i| i as Real * 1.5).collect();
    ParticleState::new(
        pos,
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        vec![0.0; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap()
}

fn single_bond_list(n: usize) -> BondList {
    let bonds = vec![Bond {
        atom_i: 0,
        atom_j: 1,
        bond_type_index: 0,
    }];
    let mut atom_bond_offsets = vec![0u32; n + 1];
    atom_bond_offsets[1] = 1;
    for i in 2..=n {
        atom_bond_offsets[i] = 2;
    }
    let atom_bond_indices = vec![0u32, 1u32];
    BondList {
        bonds,
        atom_bond_offsets,
        atom_bond_indices,
        particle_count: n,
    }
}

/// Build an LJ-only ForceField.
fn lj_only_force_field(gpu: &GpuContext, n: usize) -> ForceField {
    ForceField::new(
        &PotentialRegistry::with_builtins(),
        gpu,
        n,
        &box_10(&gpu),
        &[ar_type()],
        &[lj_pair_config()],
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

/// Build an LJ + Morse ForceField (single Morse bond between 0 and 1).
fn lj_and_morse_force_field(gpu: &GpuContext, n: usize) -> ForceField {
    ForceField::new(
        &PotentialRegistry::with_builtins(),
        gpu,
        n,
        &box_10(&gpu),
        &[ar_type()],
        &[lj_pair_config()],
        &[morse_bond_type()],
        &[],
        None,
        None,
        &[],
        &single_bond_list(n),
        &AngleList::empty(0),
        &ExclusionList::empty(n),
        &NeighborListConfig::AllPairs,
    )
    .unwrap()
}

/// Build an empty (zero-slot) ForceField.
fn empty_force_field(gpu: &GpuContext, n: usize) -> ForceField {
    ForceField::new(
        &PotentialRegistry::with_builtins(),
        gpu,
        n,
        &box_10(&gpu),
        &[ar_type()],
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

fn morse_only_force_field(gpu: &GpuContext, n: usize) -> ForceField {
    ForceField::new(
        &PotentialRegistry::with_builtins(),
        gpu,
        n,
        &box_10(&gpu),
        &[ar_type()],
        &[],
        &[morse_bond_type()],
        &[],
        None,
        None,
        &[],
        &single_bond_list(n),
        &AngleList::empty(0),
        &ExclusionList::empty(n),
        &NeighborListConfig::AllPairs,
    )
    .unwrap()
}

/// Run step and return the final TimingsReport.
fn step_and_finalize(
    ff: &mut ForceField,
    buffers: &mut ParticleBuffers,
    sim_box: &SimulationBox,
    gpu: &GpuContext,
    level: AggregateLevel,
) -> TimingsReport {
    let mut timings = Timings::new(gpu).unwrap();
    ff.step(buffers, sim_box, &mut timings, level).unwrap();
    timings.finalize().unwrap()
}

fn stage_count(report: &TimingsReport, name: &str) -> u64 {
    report
        .stages
        .iter()
        .find(|s| s.name == name)
        .map(|s| s.count)
        .unwrap_or(0)
}

// =================================================================
// Stub potential: configurable label + class + cutoff. Writes a
// `marker` value into all five output slots so the test can detect
// whether `compute` was called. Each call also increments an
// `Arc<AtomicU32>` so the test can observe the invocation count.
// =================================================================

#[derive(Debug, Clone)]
struct StubPotential {
    label: &'static str,
    class: ForceClass,
    cutoff: Option<Real>,
    marker: Real,
    call_count: Arc<AtomicU32>,
}

impl Potential for StubPotential {
    fn label(&self) -> &'static str {
        self.label
    }

    fn max_cutoff(&self) -> Option<Real> {
        self.cutoff
    }

    fn frequency_class(&self) -> ForceClass {
        self.class
    }

    fn compute(
        &mut self,
        buffers: &ParticleBuffers,
        _sim_box: &SimulationBox,
        mut output: SlotOutputView<'_>,
        _cx: &ForceFieldContext<'_>,
        _timings: &mut Timings,
        level: AggregateLevel,
    ) -> Result<(), ForceFieldError> {
        let dev = &buffers.device;
        let marker = self.marker;
        let add_into = |slice: &mut cudarc::driver::CudaViewMut<Real>| {
            let mut host = dev.dtoh_sync_copy(&*slice).unwrap();
            for v in host.iter_mut() {
                *v += marker;
            }
            dev.htod_sync_copy_into(&host, slice).unwrap();
        };
        add_into(&mut output.force_x);
        add_into(&mut output.force_y);
        add_into(&mut output.force_z);
        if level == AggregateLevel::ForcesAndScalars {
            add_into(&mut output.energy);
            add_into(&mut output.virial);
        }
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn jit_participant(&self) -> Option<JitParticipant<'_>> {
        if self.class == ForceClass::Fast && self.cutoff.is_some() {
            Some(JitParticipant::PairForce(self))
        } else {
            None
        }
    }
}

impl PairForcePotential for StubPotential {
    fn pair_force_fragment(&self) -> PairForceFragment {
        // No-op fragment so the stub can stand in as a pair-force slot.
        // The functor name is `Box::leak`-ed to satisfy the fragment's
        // 'static lifetime.
        let functor_struct_name: &'static str =
            Box::leak(format!("StubFunctor_{}", self.label).into_boxed_str());
        let functor_source = format!(
            r#"
struct {n} {{
    __device__ inline Real cutoff_squared(unsigned int, unsigned int) const {{ return R(0.0); }}
    __device__ inline void evaluate(Real, Real, Real, Real, Real, unsigned int, unsigned int,
                                     Real &factor, Real &energy, Real &virial) const {{
        factor = R(0.0); energy = R(0.0); virial = R(0.0);
    }}
    __device__ inline Real exclusion_scale(unsigned int, unsigned int) const {{ return R(1.0); }}
}};
"#,
            n = functor_struct_name
        );
        PairForceFragment {
            label: self.label,
            functor_struct_name,
            functor_source,
            entry_point_args: String::new(),
            functor_init_source: String::new(),
            cutoff: CutoffHandling::PerPair,
        }
    }

    fn bind_pair_force_args(
        &self,
        _ctx: &PairForceBindContext<'_>,
        _builder: &mut ForceLaunchBuilder,
    ) {
        // No-op functor takes no parameters.
    }
}

#[derive(Debug, Clone)]
struct StubBuilder {
    label: &'static str,
    class: ForceClass,
    cutoff: Option<Real>,
    marker: Real,
    call_count: Arc<AtomicU32>,
    /// When `false`, `build` always returns `Ok(None)` (inactive).
    active: bool,
    /// When `Some(_)`, `build` returns that error instead.
    force_error: Option<&'static str>,
    /// Labels this stub claims to displace. Surfaced through
    /// `PotentialBuilder::displaces()`.
    displaces: &'static [&'static str],
}

impl StubBuilder {
    fn new(label: &'static str) -> Self {
        StubBuilder {
            label,
            class: ForceClass::Fast,
            cutoff: None,
            marker: 0.0,
            call_count: Arc::new(AtomicU32::new(0)),
            active: true,
            force_error: None,
            displaces: &[],
        }
    }
}

impl PotentialBuilder for StubBuilder {
    fn build(
        &self,
        _cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        if let Some(msg) = self.force_error {
            return Err(ForceFieldError::DuplicateLabel(msg));
        }
        if !self.active {
            return Ok(None);
        }
        Ok(Some(Box::new(StubPotential {
            label: self.label,
            class: self.class,
            cutoff: self.cutoff,
            marker: self.marker,
            call_count: self.call_count.clone(),
        })))
    }
    fn displaces(&self) -> &'static [&'static str] {
        self.displaces
    }
}

fn build_with(
    gpu: &GpuContext,
    n: usize,
    registry: &PotentialRegistry,
) -> Result<ForceField, ForceFieldError> {
    ForceField::new(
        registry,
        gpu,
        n,
        &box_10(&gpu),
        &[ar_type()],
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
}

// =================================================================
// Section 1: Construction
// =================================================================

// rq-56c8a238
#[test]
fn construct_force_field_with_lennardjones_only() {
    let gpu = init_device().unwrap();
    let ff = lj_only_force_field(&gpu, 4);
    assert_eq!(ff.slots.len(), 1);
    assert_eq!(ff.slots[0].label(), "lennard_jones");
}

// rq-3de16ce0
#[test]
fn construct_force_field_with_lennardjones_and_morse() {
    let gpu = init_device().unwrap();
    let ff = lj_and_morse_force_field(&gpu, 4);
    assert_eq!(ff.slots.len(), 2);
    assert_eq!(ff.slots[0].label(), "lennard_jones");
    assert_eq!(ff.slots[1].label(), "morse_bonded");
}

// rq-0f34d11b
#[test]
fn bond_types_declared_with_no_bonds_omits_morse_slot() {
    let gpu = init_device().unwrap();
    let n = 4;
    let ff = ForceField::new(
        &PotentialRegistry::with_builtins(),
        &gpu,
        n,
        &box_10(&gpu),
        &[ar_type()],
        &[lj_pair_config()],
        &[morse_bond_type()],
        &[],
        None,
        None,
        &[],
        &BondList::empty(n),
        &AngleList::empty(0),
        &ExclusionList::empty(n),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    assert_eq!(ff.slots.len(), 1);
    assert_eq!(ff.slots[0].label(), "lennard_jones");
}

// rq-60f445b2
#[test]
fn empty_force_field_class_accumulators_have_length_n_zeroed() {
    let gpu = init_device().unwrap();
    let n = 4;
    let ff = empty_force_field(&gpu, n);
    assert_eq!(ff.slots.len(), 0);
    assert_eq!(ff.fast_total_forces_x.len(), n);
    assert_eq!(ff.fast_total_forces_y.len(), n);
    assert_eq!(ff.fast_total_forces_z.len(), n);
    assert_eq!(ff.slow_total_forces_x.len(), n);
    let host = gpu.device.dtoh_sync_copy(&ff.fast_total_forces_x).unwrap();
    assert!(host.iter().all(|&v| v == 0.0));
}

// rq-455db9c2
#[test]
fn class_accumulator_buffers_have_length_particle_count() {
    let gpu = init_device().unwrap();
    let n = 8;
    let ff = lj_and_morse_force_field(&gpu, n);
    // Per-class accumulator buffers are always length N (one entry per
    // particle), regardless of how many slots populate the class.
    assert_eq!(ff.fast_total_forces_x.len(), n);
    assert_eq!(ff.fast_total_forces_y.len(), n);
    assert_eq!(ff.fast_total_forces_z.len(), n);
    assert_eq!(ff.fast_total_potential_energies.len(), n);
    assert_eq!(ff.fast_total_virials.len(), n);
    assert_eq!(ff.slow_total_forces_x.len(), n);
}

// rq-c525ee79
#[test]
fn construct_empty_n0_force_field_with_potentials_configured() {
    let gpu = init_device().unwrap();
    // Two stub slots so the framework's slot count is 2. Particle count
    // is zero, so each per-class buffer should have length 0.
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder::new("a")));
    registry.register(Box::new(StubBuilder::new("b")));
    let ff = build_with(&gpu, 0, &registry).unwrap();
    assert_eq!(ff.slots.len(), 2);
    assert_eq!(ff.fast_total_forces_x.len(), 0);
    assert_eq!(ff.fast_total_forces_y.len(), 0);
    assert_eq!(ff.fast_total_forces_z.len(), 0);
    assert_eq!(ff.fast_total_potential_energies.len(), 0);
    assert_eq!(ff.fast_total_virials.len(), 0);
}

// rq-c170c0b7
#[test]
fn reject_construction_when_two_slots_share_a_label() {
    let gpu = init_device().unwrap();
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder::new("dupe")));
    registry.register(Box::new(StubBuilder::new("dupe")));
    let err = build_with(&gpu, 4, &registry).unwrap_err();
    match err {
        ForceFieldError::DuplicateLabel(label) => assert_eq!(label, "dupe"),
        other => panic!("expected DuplicateLabel, got {:?}", other),
    }
}

// =================================================================
// Section 2: Force evaluation pipeline
// =================================================================

// rq-32e981cc
#[test]
fn step_lennardjones_only_writes_nonzero_forces() {
    let gpu = init_device().unwrap();
    let state = state_n(2);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut ff = lj_only_force_field(&gpu, 2);
    let report = step_and_finalize(&mut ff, &mut buffers, &box_10(&gpu), &gpu, AggregateLevel::ForcesAndScalars);
    let mut downloaded = state.clone();
    downloaded.download_from(&buffers).unwrap();
    assert!(downloaded.forces_x[0].abs() > 0.0);
    assert!((downloaded.forces_x[0] + downloaded.forces_x[1]).abs() < 1e-5);
    assert!(stage_count(&report, "jit_composed_pair_force") >= 1);
    assert_eq!(stage_count(&report, KernelStage::COMBINE_CLASS_TOTALS.name()), 1);
}

// rq-df3a50f6
#[test]
fn step_with_both_slots_sums_lj_and_morse() {
    let gpu = init_device().unwrap();
    // Two particles separated by 1.2 so both LJ and Morse contribute.
    let state = ParticleState::new(
        vec![0.0 as Real, 1.2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![1.0; 2],
        vec![0.0; 2],
        vec![0u32; 2],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut ff = lj_and_morse_force_field(&gpu, 2);
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let mut downloaded = state.clone();
    downloaded.download_from(&buffers).unwrap();
    // Sum of forces == 0 because both LJ and Morse obey Newton's third law.
    let sx = downloaded.forces_x[0] + downloaded.forces_x[1];
    let sy = downloaded.forces_y[0] + downloaded.forces_y[1];
    let sz = downloaded.forces_z[0] + downloaded.forces_z[1];
    assert!(sx.abs() < 1e-5);
    assert!(sy.abs() < 1e-5);
    assert!(sz.abs() < 1e-5);
}

// rq-fc7b1565
#[test]
fn step_with_zero_slots_writes_zeros_to_forces() {
    let gpu = init_device().unwrap();
    let n = 4;
    let mut state = state_n(n);
    // Seed buffers with non-zero forces to confirm the combiner overwrites.
    state.forces_x = vec![1.0 as Real; n];
    state.forces_y = vec![2.0 as Real; n];
    state.forces_z = vec![3.0 as Real; n];
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut ff = empty_force_field(&gpu, n);
    let report = step_and_finalize(&mut ff, &mut buffers, &box_10(&gpu), &gpu, AggregateLevel::ForcesAndScalars);
    let mut downloaded = state.clone();
    downloaded.download_from(&buffers).unwrap();
    assert!(downloaded.forces_x.iter().all(|&v| v == 0.0));
    assert!(downloaded.forces_y.iter().all(|&v| v == 0.0));
    assert!(downloaded.forces_z.iter().all(|&v| v == 0.0));
    assert_eq!(stage_count(&report, KernelStage::COMBINE_CLASS_TOTALS.name()), 1);
    assert_eq!(stage_count(&report, "morse_bond_force"), 0);
}

// rq-de47c1ac
#[test]
fn step_with_n0_launches_no_kernels() {
    let gpu = init_device().unwrap();
    let state = state_n(0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder::new("a")));
    registry.register(Box::new(StubBuilder::new("b")));
    let mut ff = build_with(&gpu, 0, &registry).unwrap();
    let report = step_and_finalize(&mut ff, &mut buffers, &box_10(&gpu), &gpu, AggregateLevel::ForcesAndScalars);
    for s in &report.stages {
        assert_eq!(s.count, 0, "stage {} expected count 0, got {}", s.name, s.count);
    }
}

// rq-7d8485b3
#[test]
fn slot_contributions_sum_into_the_class_accumulator() {
    let gpu = init_device().unwrap();
    let n = 3;
    let mut registry = PotentialRegistry::new();
    let a = StubBuilder {
        marker: 1.0,
        ..StubBuilder::new("a")
    };
    let b = StubBuilder {
        marker: 2.0,
        ..StubBuilder::new("b")
    };
    registry.register(Box::new(a));
    registry.register(Box::new(b));
    let mut ff = build_with(&gpu, n, &registry).unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let fast_x = gpu.device.dtoh_sync_copy(&ff.fast_total_forces_x).unwrap();
    assert_eq!(fast_x.len(), n);
    // Both slots add their marker into the same length-N accumulator,
    // so every entry is the sum of the markers.
    assert!(fast_x.iter().all(|&v| v == 3.0));
}

// =================================================================
// Section 3: Trait extensibility
// =================================================================

// rq-a9642241
#[test]
fn adding_a_new_potential_implementation_does_not_require_framework_edits() {
    let gpu = init_device().unwrap();
    // Start with the built-ins, append a custom slot. The runner builds
    // the ForceField identically; no change to ForceField::new is needed.
    let mut registry = PotentialRegistry::with_builtins();
    let custom = StubBuilder {
        marker: 42.0,
        ..StubBuilder::new("buckingham_stub")
    };
    registry.register(Box::new(custom));
    let n = 4;
    let mut ff = ForceField::new(
        &registry,
        &gpu,
        n,
        &box_10(&gpu),
        &[ar_type()],
        &[lj_pair_config()],
        &[morse_bond_type()],
        &[],
        None,
        None,
        &[],
        &single_bond_list(n),
        &AngleList::empty(0),
        &ExclusionList::empty(n),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    // 3 slots: LJ + Morse + custom.
    assert_eq!(ff.slots.len(), 3);
    assert_eq!(ff.slots[0].label(), "lennard_jones");
    assert_eq!(ff.slots[1].label(), "morse_bonded");
    assert_eq!(ff.slots[2].label(), "buckingham_stub");

    let mut buffers = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    // The custom slot adds 42.0 into the Fast-class accumulator alongside
    // the LJ and Morse contributions, so every entry is at least 42.0 once
    // the stub's contribution has been folded in.
    let fast_x = gpu.device.dtoh_sync_copy(&ff.fast_total_forces_x).unwrap();
    assert_eq!(fast_x.len(), n);
    let mut ff_no_stub = ForceField::new(
        &PotentialRegistry::with_builtins(),
        &gpu,
        n,
        &box_10(&gpu),
        &[ar_type()],
        &[lj_pair_config()],
        &[morse_bond_type()],
        &[],
        None,
        None,
        &[],
        &single_bond_list(n),
        &AngleList::empty(0),
        &ExclusionList::empty(n),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let mut buffers_b = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings_b = Timings::new(&gpu).unwrap();
    ff_no_stub.step(&mut buffers_b, &box_10(&gpu), &mut timings_b, AggregateLevel::ForcesAndScalars).unwrap();
    let baseline = gpu.device.dtoh_sync_copy(&ff_no_stub.fast_total_forces_x).unwrap();
    for i in 0..n {
        assert!((fast_x[i] - baseline[i] - 42.0).abs() < 1e-4);
    }
}

// =================================================================
// Section 4: Registry-driven construction
// =================================================================

// rq-053a026c
#[test]
fn registry_with_builtins_exposes_seven_builders_in_evaluation_order() {
    let r = PotentialRegistry::with_builtins();
    assert_eq!(r.builders().len(), 6);
    let names: Vec<String> = r.builders().iter().map(|b| format!("{:?}", b)).collect();
    assert!(names[0].contains("LennardJones"), "builder 0 = {}", names[0]);
    assert!(names[1].contains("Coulomb"), "builder 1 = {}", names[1]);
    assert!(names[2].contains("SpmeReal"), "builder 2 = {}", names[2]);
    assert!(names[3].contains("SpmeReciprocal"), "builder 3 = {}", names[3]);
    assert!(names[4].contains("MorseBonded"), "builder 4 = {}", names[4]);
    assert!(names[5].contains("HarmonicAngle"), "builder 5 = {}", names[5]);
}

// rq-78ad9477
#[test]
fn registry_new_starts_empty() {
    let r = PotentialRegistry::new();
    assert!(r.builders().is_empty());
}

// rq-51af5f97
#[test]
fn register_appends_a_builder_at_the_end() {
    let mut r = PotentialRegistry::with_builtins();
    r.register(Box::new(StubBuilder::new("custom")));
    assert_eq!(r.builders().len(), 7);
    let last = format!("{:?}", r.builders()[6]);
    assert!(last.contains("custom"), "last builder = {}", last);
}

// rq-b1a132b5
#[test]
fn force_field_new_iterates_registry_in_registration_order() {
    let gpu = init_device().unwrap();
    let ff = lj_and_morse_force_field(&gpu, 4);
    assert_eq!(ff.slots[0].label(), "lennard_jones");
    assert_eq!(ff.slots[1].label(), "morse_bonded");
}

// rq-ccf4dc3f / "Builder Ok(None) is skipped without erroring"
#[test]
fn builder_returning_ok_none_is_skipped_without_erroring() {
    let gpu = init_device().unwrap();
    let mut registry = PotentialRegistry::new();
    let inactive = StubBuilder {
        active: false,
        ..StubBuilder::new("inactive")
    };
    let active = StubBuilder::new("active");
    registry.register(Box::new(inactive));
    registry.register(Box::new(active));
    let ff = build_with(&gpu, 4, &registry).unwrap();
    assert_eq!(ff.slots.len(), 1);
    assert_eq!(ff.slots[0].label(), "active");
}

// "Builder Err short-circuits ForceField::new"
#[test]
fn builder_err_short_circuits_force_field_new() {
    let gpu = init_device().unwrap();
    let mut registry = PotentialRegistry::new();
    let failing = StubBuilder {
        force_error: Some("boom"),
        ..StubBuilder::new("first")
    };
    let after = StubBuilder::new("after");
    registry.register(Box::new(failing));
    registry.register(Box::new(after));
    let err = build_with(&gpu, 4, &registry).unwrap_err();
    matches!(err, ForceFieldError::DuplicateLabel("boom"));
    // The "after" stub never got to register a slot, but its build was
    // never invoked either (we cannot directly observe; the Err path
    // returns immediately from the loop).
}

// "Two builders producing same label fail construction" — same rq as
// section 1's duplicate-label test, but exercised via two stub builders
// with different identity and same label() in the slot.
#[test]
fn two_distinct_builders_producing_same_label_fail_construction() {
    let gpu = init_device().unwrap();
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder::new("shared")));
    registry.register(Box::new(StubBuilder::new("shared")));
    let err = build_with(&gpu, 4, &registry).unwrap_err();
    assert!(matches!(err, ForceFieldError::DuplicateLabel("shared")));
}

#[test]
fn empty_registry_produces_zero_slot_force_field() {
    let gpu = init_device().unwrap();
    let ff = build_with(&gpu, 4, &PotentialRegistry::new()).unwrap();
    assert_eq!(ff.slots.len(), 0);
}

// =================================================================
// Section 5: PotentialBuildContext field exposure
// =================================================================

#[derive(Debug, Clone)]
struct InspectBuilder {
    seen_particle_count: Arc<AtomicU32>,
}

impl PotentialBuilder for InspectBuilder {
    fn build(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        self.seen_particle_count
            .store(cx.particle_count as u32, Ordering::SeqCst);
        // Touch every named field so a missing one fails to compile.
        let _ = cx.gpu;
        let _ = cx.sim_box;
        let _ = cx.particle_types;
        let _ = cx.pair_interactions;
        let _ = cx.bond_types;
        let _ = cx.angle_types;
        let _ = cx.coulomb_config;
        let _ = cx.spme_config;
        let _ = cx.charges;
        let _ = cx.bond_list;
        let _ = cx.angle_list;
        let _ = cx.exclusion_list;
        let _ = cx.neighbor_list_config;
        Ok(None)
    }
}

#[test]
fn build_context_exposes_every_parsed_config_input_by_reference() {
    let gpu = init_device().unwrap();
    let seen = Arc::new(AtomicU32::new(0));
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(InspectBuilder {
        seen_particle_count: seen.clone(),
    }));
    let _ = build_with(&gpu, 7, &registry).unwrap();
    assert_eq!(seen.load(Ordering::SeqCst), 7);
}

// =================================================================
// Section 6: Force class
// =================================================================

#[test]
fn potential_frequency_class_default_returns_fast() {
    // StubPotential uses the trait default — we configure it to Fast and
    // assert that built-in slots also use Fast unless they override.
    let stub = StubPotential {
        label: "stub",
        class: ForceClass::Fast,
        cutoff: None,
        marker: 0.0,
        call_count: Arc::new(AtomicU32::new(0)),
    };
    assert_eq!(stub.frequency_class(), ForceClass::Fast);
}

#[test]
fn builtin_potentials_report_canonical_class() {
    let gpu = init_device().unwrap();
    // LJ → Fast.
    let ff_lj = lj_only_force_field(&gpu, 4);
    assert_eq!(ff_lj.slots[0].frequency_class(), ForceClass::Fast);
    // Morse → Fast.
    let ff_morse = morse_only_force_field(&gpu, 4);
    assert_eq!(ff_morse.slots[0].frequency_class(), ForceClass::Fast);
}

#[test]
fn step_evaluates_every_class_and_produces_total_in_particle_buffers() {
    let gpu = init_device().unwrap();
    let n = 3;
    let fast_count = Arc::new(AtomicU32::new(0));
    let slow_count = Arc::new(AtomicU32::new(0));
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder {
        marker: 1.0,
        call_count: fast_count.clone(),
        ..StubBuilder::new("fast_stub")
    }));
    registry.register(Box::new(StubBuilder {
        class: ForceClass::Slow,
        marker: 10.0,
        call_count: slow_count.clone(),
        ..StubBuilder::new("slow_stub")
    }));
    let mut ff = build_with(&gpu, n, &registry).unwrap();
    let state = state_n(n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    assert_eq!(fast_count.load(Ordering::SeqCst), 1);
    assert_eq!(slow_count.load(Ordering::SeqCst), 1);
    let mut downloaded = state.clone();
    downloaded.download_from(&buffers).unwrap();
    // Each particle's total = 1.0 (fast) + 10.0 (slow) = 11.0.
    assert!(downloaded.forces_x.iter().all(|&v| (v - 11.0).abs() < 1e-5));
}

#[test]
fn step_class_fast_refreshes_only_fast_slots_contributions() {
    let gpu = init_device().unwrap();
    let n = 3;
    let fast_count = Arc::new(AtomicU32::new(0));
    let slow_count = Arc::new(AtomicU32::new(0));
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder {
        marker: 1.0,
        call_count: fast_count.clone(),
        ..StubBuilder::new("fast_stub")
    }));
    registry.register(Box::new(StubBuilder {
        class: ForceClass::Slow,
        marker: 10.0,
        call_count: slow_count.clone(),
        ..StubBuilder::new("slow_stub")
    }));
    let mut ff = build_with(&gpu, n, &registry).unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    // Prime both classes' buffers via a full step first.
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    fast_count.store(0, Ordering::SeqCst);
    slow_count.store(0, Ordering::SeqCst);
    // Now step_class(Fast).
    ff.step_class(ForceClass::Fast, &mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    assert_eq!(fast_count.load(Ordering::SeqCst), 1);
    assert_eq!(slow_count.load(Ordering::SeqCst), 0);
}

#[test]
fn step_class_slow_refreshes_only_slow_slots_contributions() {
    let gpu = init_device().unwrap();
    let n = 3;
    let fast_count = Arc::new(AtomicU32::new(0));
    let slow_count = Arc::new(AtomicU32::new(0));
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder {
        marker: 1.0,
        call_count: fast_count.clone(),
        ..StubBuilder::new("fast_stub")
    }));
    registry.register(Box::new(StubBuilder {
        class: ForceClass::Slow,
        marker: 10.0,
        call_count: slow_count.clone(),
        ..StubBuilder::new("slow_stub")
    }));
    let mut ff = build_with(&gpu, n, &registry).unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    fast_count.store(0, Ordering::SeqCst);
    slow_count.store(0, Ordering::SeqCst);
    ff.step_class(ForceClass::Slow, &mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    assert_eq!(fast_count.load(Ordering::SeqCst), 0);
    assert_eq!(slow_count.load(Ordering::SeqCst), 1);
}

#[test]
fn step_class_slow_on_force_field_with_no_slow_slots_is_noop() {
    let gpu = init_device().unwrap();
    let n = 4;
    let mut ff = lj_only_force_field(&gpu, n);
    let mut state = state_n(n);
    state.forces_x = vec![99.0; n];
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step_class(ForceClass::Slow, &mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let report = timings.finalize().unwrap();
    for s in &report.stages {
        assert_eq!(s.count, 0, "step_class(Slow) on no-slow-slots ForceField launched {} ({}× )", s.name, s.count);
    }
    // forces_x untouched.
    let mut downloaded = state.clone();
    downloaded.download_from(&buffers).unwrap();
    assert!(downloaded.forces_x.iter().all(|&v| v == 99.0));
}

#[test]
fn step_class_fast_on_force_field_with_no_fast_slots_is_noop() {
    let gpu = init_device().unwrap();
    let n = 4;
    let slow_count = Arc::new(AtomicU32::new(0));
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder {
        class: ForceClass::Slow,
        marker: 7.0,
        call_count: slow_count.clone(),
        ..StubBuilder::new("only_slow")
    }));
    let mut ff = build_with(&gpu, n, &registry).unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step_class(ForceClass::Fast, &mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    assert_eq!(slow_count.load(Ordering::SeqCst), 0);
    let report = timings.finalize().unwrap();
    assert_eq!(stage_count(&report, KernelStage::COMBINE_CLASS_TOTALS.name()), 0);
}

#[test]
fn step_class_with_n0_launches_no_kernels() {
    let gpu = init_device().unwrap();
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder::new("a")));
    registry.register(Box::new(StubBuilder::new("b")));
    let mut ff = build_with(&gpu, 0, &registry).unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state_n(0)).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step_class(ForceClass::Fast, &mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let report = timings.finalize().unwrap();
    for s in &report.stages {
        assert_eq!(s.count, 0);
    }
}

#[test]
fn per_class_accumulator_buffers_have_length_n_per_class() {
    let gpu = init_device().unwrap();
    let n = 5;
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder::new("a")));
    registry.register(Box::new(StubBuilder {
        class: ForceClass::Slow,
        ..StubBuilder::new("b")
    }));
    let ff = build_with(&gpu, n, &registry).unwrap();
    assert_eq!(ff.fast_total_forces_x.len(), n);
    assert_eq!(ff.fast_total_potential_energies.len(), n);
    assert_eq!(ff.fast_total_virials.len(), n);
    assert_eq!(ff.slow_total_forces_x.len(), n);
    assert_eq!(ff.slow_total_potential_energies.len(), n);
    assert_eq!(ff.slow_total_virials.len(), n);
}

#[test]
fn per_class_accumulator_buffers_are_zero_initialised() {
    let gpu = init_device().unwrap();
    let n = 5;
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder::new("a")));
    registry.register(Box::new(StubBuilder {
        class: ForceClass::Slow,
        ..StubBuilder::new("b")
    }));
    let ff = build_with(&gpu, n, &registry).unwrap();
    let fast = gpu.device.dtoh_sync_copy(&ff.fast_total_forces_x).unwrap();
    let slow = gpu.device.dtoh_sync_copy(&ff.slow_total_forces_x).unwrap();
    assert!(fast.iter().all(|&v| v == 0.0));
    assert!(slow.iter().all(|&v| v == 0.0));
}

// =================================================================
// Section 7: RESPA / byte-identity
// =================================================================

#[test]
fn two_respa_call_sequences_with_the_same_plan_produce_identical_state() {
    let gpu = init_device().unwrap();
    let n = 3;
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder {
        marker: 1.0,
        ..StubBuilder::new("fast_stub")
    }));
    registry.register(Box::new(StubBuilder {
        class: ForceClass::Slow,
        marker: 10.0,
        ..StubBuilder::new("slow_stub")
    }));

    let run_plan = |ff: &mut ForceField, buffers: &mut ParticleBuffers, timings: &mut Timings| {
        ff.step_class(ForceClass::Slow, buffers, &box_10(&gpu), timings, AggregateLevel::ForcesAndScalars).unwrap();
        for _ in 0..3 {
            ff.step_class(ForceClass::Fast, buffers, &box_10(&gpu), timings, AggregateLevel::ForcesAndScalars).unwrap();
        }
    };

    let mut ff_a = build_with(&gpu, n, &registry).unwrap();
    let mut buffers_a = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings_a = Timings::new(&gpu).unwrap();
    run_plan(&mut ff_a, &mut buffers_a, &mut timings_a);

    let mut ff_b = build_with(&gpu, n, &registry).unwrap();
    let mut buffers_b = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings_b = Timings::new(&gpu).unwrap();
    run_plan(&mut ff_b, &mut buffers_b, &mut timings_b);

    let fx_a = gpu.device.dtoh_sync_copy(&buffers_a.forces_x).unwrap();
    let fx_b = gpu.device.dtoh_sync_copy(&buffers_b.forces_x).unwrap();
    for i in 0..n {
        assert_eq!(fx_a[i].to_bits(), fx_b[i].to_bits());
    }
}

// SubStep::ForceEval { class: None } dispatches to step()  — covered
// indirectly by `tests/integrator_framework.rs::plan_with_multiple_force_evals_dispatches_each`.
// SubStep::ForceEval { class: Some(Fast) } dispatches to step_class(Fast) —
// same. The framework-side guarantee tested above is that step_class
// behaves as documented; the runner's plan walker is exercised in
// integrator_framework.rs.

#[test]
fn two_independent_runs_byte_identical() {
    let gpu = init_device().unwrap();
    let n = 4;
    let state = state_n(n);
    let mut ff_a = lj_and_morse_force_field(&gpu, n);
    let mut buf_a = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut tim_a = Timings::new(&gpu).unwrap();
    ff_a.step(&mut buf_a, &box_10(&gpu), &mut tim_a, AggregateLevel::ForcesAndScalars).unwrap();
    let mut ff_b = lj_and_morse_force_field(&gpu, n);
    let mut buf_b = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut tim_b = Timings::new(&gpu).unwrap();
    ff_b.step(&mut buf_b, &box_10(&gpu), &mut tim_b, AggregateLevel::ForcesAndScalars).unwrap();
    let fx_a = gpu.device.dtoh_sync_copy(&buf_a.forces_x).unwrap();
    let fx_b = gpu.device.dtoh_sync_copy(&buf_b.forces_x).unwrap();
    let e_a = gpu.device.dtoh_sync_copy(&buf_a.potential_energies).unwrap();
    let e_b = gpu.device.dtoh_sync_copy(&buf_b.potential_energies).unwrap();
    for i in 0..n {
        assert_eq!(fx_a[i].to_bits(), fx_b[i].to_bits(), "forces_x[{}] differ", i);
        assert_eq!(e_a[i].to_bits(), e_b[i].to_bits(), "potential_energies[{}] differ", i);
    }
}

// =================================================================
// Section 8: Combiner kernel
// =================================================================

#[test]
fn combiner_sums_slot_rows_in_slot_order() {
    let gpu = init_device().unwrap();
    let n = 3;
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder {
        marker: 1.0,
        ..StubBuilder::new("a")
    }));
    registry.register(Box::new(StubBuilder {
        marker: 2.0,
        ..StubBuilder::new("b")
    }));
    registry.register(Box::new(StubBuilder {
        marker: 4.0,
        ..StubBuilder::new("c")
    }));
    let mut ff = build_with(&gpu, n, &registry).unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let fx = gpu.device.dtoh_sync_copy(&buffers.forces_x).unwrap();
    for v in &fx {
        assert_eq!(*v, 7.0);
    }
}

#[test]
fn combiner_with_num_slots_zero_writes_zeros() {
    let gpu = init_device().unwrap();
    let n = 4;
    let mut state = state_n(n);
    state.forces_x = vec![99.0; n];
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let fx = gpu.device.dtoh_sync_copy(&buffers.forces_x).unwrap();
    assert!(fx.iter().all(|&v| v == 0.0));
}

#[test]
fn combiner_is_a_single_write_per_output_element() {
    // Indirectly: invoking step twice produces the same totals (the
    // combiner overwrites, doesn't accumulate).
    let gpu = init_device().unwrap();
    let n = 3;
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder {
        marker: 5.0,
        ..StubBuilder::new("a")
    }));
    let mut ff = build_with(&gpu, n, &registry).unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let fx = gpu.device.dtoh_sync_copy(&buffers.forces_x).unwrap();
    for v in &fx {
        assert_eq!(*v, 5.0);
    }
}

// =================================================================
// Section 9: Neighbor-list ownership
// =================================================================

fn neighbor_list_is_some(ff: &ForceField) -> bool {
    ff.neighbor_list.is_some()
}

#[test]
fn force_field_with_short_range_potential_owns_a_shared_neighbor_list() {
    let gpu = init_device().unwrap();
    let ff = lj_only_force_field(&gpu, 4);
    assert!(neighbor_list_is_some(&ff));
}

#[test]
fn force_field_with_only_a_bonded_potential_owns_no_neighbor_list() {
    let gpu = init_device().unwrap();
    let ff = morse_only_force_field(&gpu, 2);
    assert!(!neighbor_list_is_some(&ff));
}

#[derive(Debug, Clone)]
struct CutoffInspectingBuilder {
    cutoff: Real,
    saw_neighbor_list: Arc<AtomicU32>,
}

#[derive(Debug)]
struct CutoffInspectingPotential {
    cutoff: Real,
    saw_neighbor_list: Arc<AtomicU32>,
}

impl Potential for CutoffInspectingPotential {
    fn label(&self) -> &'static str {
        "cutoff_inspector"
    }
    fn max_cutoff(&self) -> Option<Real> {
        Some(self.cutoff)
    }
    fn compute(
        &mut self,
        _buffers: &ParticleBuffers,
        _sim_box: &SimulationBox,
        _output: SlotOutputView<'_>,
        _cx: &ForceFieldContext<'_>,
        _timings: &mut Timings,
        _level: AggregateLevel,
    ) -> Result<(), ForceFieldError> {
        // The framework bypasses compute() for fast-class pair-force
        // slots (they participate in the JIT-composed kernel instead).
        // Kept as a no-op for the trait contract.
        Ok(())
    }

    fn jit_participant(&self) -> Option<JitParticipant<'_>> {
        Some(JitParticipant::PairForce(self))
    }
}

impl PairForcePotential for CutoffInspectingPotential {
    fn pair_force_fragment(&self) -> PairForceFragment {
        // A minimal no-op fragment that contributes zero force / energy /
        // virial, letting the test observe the framework's neighbor-list
        // and JIT-composed-kernel state.
        PairForceFragment {
            label: "cutoff_inspector",
            functor_struct_name: "CutoffInspectorFunctor",
            functor_source: r#"
struct CutoffInspectorFunctor {
    __device__ inline Real cutoff_squared(unsigned int, unsigned int) const { return R(0.0); }
    __device__ inline void evaluate(Real, Real, Real, Real, Real, unsigned int, unsigned int,
                                     Real &factor, Real &energy, Real &virial) const {
        factor = R(0.0); energy = R(0.0); virial = R(0.0);
    }
    __device__ inline Real exclusion_scale(unsigned int, unsigned int) const { return R(1.0); }
};
"#
            .to_string(),
            entry_point_args: String::new(),
            functor_init_source: String::new(),
            cutoff: CutoffHandling::PerPair,
        }
    }

    fn bind_pair_force_args(
        &self,
        _ctx: &PairForceBindContext<'_>,
        _builder: &mut ForceLaunchBuilder,
    ) {
        // No-op functor takes no parameters.
    }
}

impl PotentialBuilder for CutoffInspectingBuilder {
    fn build(
        &self,
        _cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        Ok(Some(Box::new(CutoffInspectingPotential {
            cutoff: self.cutoff,
            saw_neighbor_list: self.saw_neighbor_list.clone(),
        })))
    }
}

#[test]
fn force_field_with_pair_force_slot_owns_shared_neighbor_list_and_jit_kernel() {
    let gpu = init_device().unwrap();
    let saw = Arc::new(AtomicU32::new(0));
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(CutoffInspectingBuilder {
        cutoff: 5.0,
        saw_neighbor_list: saw.clone(),
    }));
    let ff = build_with(&gpu, 4, &registry).unwrap();
    // A fast-class pair-force slot is registered, so the framework
    // builds both the shared neighbour list and the JIT-composed
    // pair-force kernel.
    assert!(ff.neighbor_list.is_some(), "shared neighbor list built");
    assert!(ff.jit_composed.is_some(), "JIT-composed kernel built");
}

#[test]
fn max_cutoff_aggregation_determines_neighbor_list_radius() {
    let gpu = init_device().unwrap();
    let saw = Arc::new(AtomicU32::new(0));
    let mut registry = PotentialRegistry::new();
    // Small cutoff first, big cutoff second; the framework should pick the max.
    registry.register(Box::new(CutoffInspectingBuilder {
        cutoff: 1.0,
        saw_neighbor_list: saw.clone(),
    }));
    let big_cutoff = StubBuilder {
        cutoff: Some(2.5),
        ..StubBuilder::new("big_cutoff")
    };
    registry.register(Box::new(big_cutoff));
    let n = 4;
    // Use CellList so we can read back the chosen r_cut from CellListData.
    let ff = ForceField::new(
        &registry,
        &gpu,
        n,
        &box_10(&gpu),
        &[ar_type()],
        &[],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(n),
        &AngleList::empty(0),
        &ExclusionList::empty(n),
        &NeighborListConfig::CellList { r_skin: 0.0 },
    )
    .unwrap();
    let nl = ff.neighbor_list.as_ref().expect("expected a neighbor list");
    let cl = nl.cell_list_data().expect("expected CellList mode");
    assert!((cl.r_cut - 2.5).abs() < 1e-6, "expected r_cut=2.5, got {}", cl.r_cut);
}

#[test]
fn bonded_only_step_launches_no_neighbor_list_kernels() {
    let gpu = init_device().unwrap();
    let n = 2;
    let mut ff = morse_only_force_field(&gpu, n);
    let mut buffers = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let report = timings.finalize().unwrap();
    assert_eq!(stage_count(&report, "neighbor_displacement_check_flag"), 0);
    assert_eq!(stage_count(&report, "neighbor_list_build"), 0);
}

// =================================================================
// Section 10: Energy / virial aggregation
// =================================================================

#[test]
fn force_field_lj_only_populates_energy_and_virial() {
    let gpu = init_device().unwrap();
    let n = 4;
    let mut ff = lj_only_force_field(&gpu, n);
    let mut buffers = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let e = gpu.device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
    let v = gpu.device.dtoh_sync_copy(&buffers.virials).unwrap();
    let total_e: Real = e.iter().sum();
    let total_v: Real = v.iter().sum();
    assert!(total_e.abs() > 0.0, "expected non-zero potential energy");
    assert!(total_v.abs() > 0.0, "expected non-zero virial");
}

#[test]
fn each_class_has_five_flat_accumulator_arrays_of_length_n() {
    let gpu = init_device().unwrap();
    let n = 4;
    let ff = lj_and_morse_force_field(&gpu, n);
    assert_eq!(ff.fast_total_forces_x.len(), n);
    assert_eq!(ff.fast_total_forces_y.len(), n);
    assert_eq!(ff.fast_total_forces_z.len(), n);
    assert_eq!(ff.fast_total_potential_energies.len(), n);
    assert_eq!(ff.fast_total_virials.len(), n);
}

#[test]
fn combiner_sums_slot_energies_and_virials_in_slot_order() {
    let gpu = init_device().unwrap();
    let n = 3;
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder {
        marker: 3.0,
        ..StubBuilder::new("a")
    }));
    registry.register(Box::new(StubBuilder {
        marker: 5.0,
        ..StubBuilder::new("b")
    }));
    let mut ff = build_with(&gpu, n, &registry).unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let e = gpu.device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
    let v = gpu.device.dtoh_sync_copy(&buffers.virials).unwrap();
    for x in &e {
        assert_eq!(*x, 8.0);
    }
    for x in &v {
        assert_eq!(*x, 8.0);
    }
}

#[test]
fn zero_slot_step_writes_zeros_to_energy_and_virial() {
    let gpu = init_device().unwrap();
    let n = 4;
    let mut state = state_n(n);
    state.potential_energies = vec![99.0; n];
    state.virials = vec![77.0; n];
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let e = gpu.device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
    let v = gpu.device.dtoh_sync_copy(&buffers.virials).unwrap();
    assert!(e.iter().all(|&x| x == 0.0));
    assert!(v.iter().all(|&x| x == 0.0));
}

#[test]
fn system_total_potential_energy_equals_sum_of_particle_shares() {
    let gpu = init_device().unwrap();
    let n = 3;
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder {
        marker: 0.5,
        ..StubBuilder::new("a")
    }));
    let mut ff = build_with(&gpu, n, &registry).unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let e = gpu.device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
    let total: Real = e.iter().sum();
    assert!((total - 1.5).abs() < 1e-6, "expected total 1.5, got {}", total);
}

#[test]
fn system_total_scalar_virial_equals_sum_of_particle_shares() {
    let gpu = init_device().unwrap();
    let n = 3;
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder {
        marker: 0.25,
        ..StubBuilder::new("a")
    }));
    let mut ff = build_with(&gpu, n, &registry).unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let v = gpu.device.dtoh_sync_copy(&buffers.virials).unwrap();
    let total: Real = v.iter().sum();
    assert!((total - 0.75).abs() < 1e-6, "expected total 0.75, got {}", total);
}

// =================================================================
// Section 11: AggregateLevel
// =================================================================

#[test]
fn step_forces_only_updates_forces_and_leaves_energies_and_virials_stale() {
    let gpu = init_device().unwrap();
    let n = 3;
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder {
        marker: 1.0,
        ..StubBuilder::new("a")
    }));
    let mut ff = build_with(&gpu, n, &registry).unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    // First, ForcesAndScalars to seed the energy/virial rows.
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let e1 = gpu.device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
    let v1 = gpu.device.dtoh_sync_copy(&buffers.virials).unwrap();
    // Now ForcesOnly; energy/virial rows must remain bit-identical.
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesOnly).unwrap();
    let e2 = gpu.device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
    let v2 = gpu.device.dtoh_sync_copy(&buffers.virials).unwrap();
    for i in 0..n {
        assert_eq!(e1[i].to_bits(), e2[i].to_bits(), "energy[{}] changed across ForcesOnly call", i);
        assert_eq!(v1[i].to_bits(), v2[i].to_bits(), "virial[{}] changed across ForcesOnly call", i);
    }
}

#[test]
fn step_forces_and_scalars_refreshes_energies_and_virials() {
    let gpu = init_device().unwrap();
    let n = 3;
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder {
        marker: 1.0,
        ..StubBuilder::new("a")
    }));
    let mut ff = build_with(&gpu, n, &registry).unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let e = gpu.device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
    let v = gpu.device.dtoh_sync_copy(&buffers.virials).unwrap();
    assert!(e.iter().all(|&x| x == 1.0));
    assert!(v.iter().all(|&x| x == 1.0));
}

#[test]
fn two_runs_with_identical_call_level_sequences_are_byte_identical() {
    let gpu = init_device().unwrap();
    let n = 3;
    let sequence = [
        AggregateLevel::ForcesAndScalars,
        AggregateLevel::ForcesOnly,
        AggregateLevel::ForcesAndScalars,
    ];
    let run = || -> (Vec<Real>, Vec<Real>, Vec<Real>) {
        let mut ff = lj_only_force_field(&gpu, n);
        let mut buffers = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
        let mut timings = Timings::new(&gpu).unwrap();
        for &lvl in &sequence {
            ff.step(&mut buffers, &box_10(&gpu), &mut timings, lvl).unwrap();
        }
        let fx = gpu.device.dtoh_sync_copy(&buffers.forces_x).unwrap();
        let e = gpu.device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
        let v = gpu.device.dtoh_sync_copy(&buffers.virials).unwrap();
        (fx, e, v)
    };
    let (fx_a, e_a, v_a) = run();
    let (fx_b, e_b, v_b) = run();
    for i in 0..n {
        assert_eq!(fx_a[i].to_bits(), fx_b[i].to_bits());
        assert_eq!(e_a[i].to_bits(), e_b[i].to_bits());
        assert_eq!(v_a[i].to_bits(), v_b[i].to_bits());
    }
}

#[test]
fn bonded_slot_honours_forces_only_via_the_level_parameter() {
    // A Morse-only ForceField step(ForcesOnly) following a step(ForcesAndScalars)
    // preserves the energy and virial values produced by the prior
    // ForcesAndScalars call: the bonded slot's compute reads `level` and
    // skips the scalar writes when ForcesOnly, and the class accumulator's
    // energy / virial entries are not zeroed for a ForcesOnly call.
    let gpu = init_device().unwrap();
    let n = 2;
    let state = ParticleState::new(
        vec![0.0 as Real, 1.2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![1.0; 2],
        vec![0.0; 2],
        vec![0u32; 2],
        None,
        None,
    )
    .unwrap();
    let mut ff = morse_only_force_field(&gpu, n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let e_seed = gpu.device.dtoh_sync_copy(&ff.fast_total_potential_energies).unwrap();
    let w_seed = gpu.device.dtoh_sync_copy(&ff.fast_total_virials).unwrap();
    // Morse bond is stretched (r=1.2 vs r_e=1.0) so the seeded values
    // are non-zero.
    assert!(e_seed.iter().any(|&x| x.abs() > 1e-6));
    assert!(w_seed.iter().any(|&x| x.abs() > 1e-6));
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesOnly).unwrap();
    let e_after = gpu.device.dtoh_sync_copy(&ff.fast_total_potential_energies).unwrap();
    let w_after = gpu.device.dtoh_sync_copy(&ff.fast_total_virials).unwrap();
    for i in 0..n {
        assert_eq!(e_seed[i].to_bits(), e_after[i].to_bits(),
            "energy[{}] changed across ForcesOnly call: {} -> {}", i, e_seed[i], e_after[i]);
        assert_eq!(w_seed[i].to_bits(), w_after[i].to_bits(),
            "virial[{}] changed across ForcesOnly call: {} -> {}", i, w_seed[i], w_after[i]);
    }
}

#[test]
fn a_pair_force_slot_honours_forces_only_for_the_slot_energy_virial() {
    // After a ForcesAndScalars then a ForcesOnly, the LJ slot's
    // energy/virial rows (not the combiner output) hold the stale
    // ForcesAndScalars values.
    let gpu = init_device().unwrap();
    let n = 4;
    let mut ff = lj_only_force_field(&gpu, n);
    let mut buffers = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let e_seed = gpu.device.dtoh_sync_copy(&ff.fast_total_potential_energies).unwrap();
    let v_seed = gpu.device.dtoh_sync_copy(&ff.fast_total_virials).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesOnly).unwrap();
    let e_after = gpu.device.dtoh_sync_copy(&ff.fast_total_potential_energies).unwrap();
    let v_after = gpu.device.dtoh_sync_copy(&ff.fast_total_virials).unwrap();
    for i in 0..n {
        assert_eq!(e_seed[i].to_bits(), e_after[i].to_bits());
        assert_eq!(v_seed[i].to_bits(), v_after[i].to_bits());
    }
}

#[test]
fn combiner_always_runs_regardless_of_level() {
    let gpu = init_device().unwrap();
    let n = 4;
    let mut ff = lj_only_force_field(&gpu, n);
    let mut buffers = ParticleBuffers::new(&gpu, &state_n(n)).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &box_10(&gpu), &mut timings, AggregateLevel::ForcesOnly).unwrap();
    let report = timings.finalize().unwrap();
    assert_eq!(stage_count(&report, KernelStage::COMBINE_CLASS_TOTALS.name()), 1);
}

// =================================================================
// Section: composite slot displacement
// =================================================================

fn spme_config_default() -> SpmeConfig {
    SpmeConfig {
        alpha: 0.3,
        r_cut_real: 5.0,
        grid: [16, 16, 16],
        spline_order: 5,
    }
}

fn ar_type_with_charge(charge: Real) -> ParticleTypeConfig {
    ParticleTypeConfig {
        name: "Ar".to_string(),
        mass: 1.0,
        charge: charge as f64,
    }
}

// Build helper that supports passing an SpmeConfig + charges.
fn build_with_spme(
    gpu: &GpuContext,
    n: usize,
    registry: &PotentialRegistry,
    pair_interactions: &[PairInteractionConfig],
    spme_config: Option<&SpmeConfig>,
    charges: &[Real],
) -> Result<ForceField, ForceFieldError> {
    ForceField::new(
        registry,
        gpu,
        n,
        &box_10(&gpu),
        &[ar_type_with_charge(charges.first().copied().unwrap_or(0.0))],
        pair_interactions,
        &[],
        &[],
        None,
        spme_config,
        charges,
        &BondList::empty(n),
        &AngleList::empty(0),
        &ExclusionList::empty(n),
        &NeighborListConfig::AllPairs,
    )
}

#[test]
fn potential_builder_displaces_default_returns_empty_slice() {
    let b = StubBuilder::new("default_stub");
    let d = b.displaces();
    assert!(d.is_empty());
}

// rq-a9a37874 — the LJ + spme_real composed kernel (packed-neighbour pass
// declared with `__launch_bounds__`) compiles via nvrtc and loads; the
// resolved `jit_composed` handle proves the `_f`/`_fev` entry points exist.
#[test]
fn lj_and_spme_real_both_in_slot_list_and_jit_composed_when_both_configured() {
    let gpu = init_device().unwrap();
    let n = 4;
    let charges = vec![1.0 as Real, -1.0, 1.0, -1.0];
    let ff = build_with_spme(
        &gpu,
        n,
        &PotentialRegistry::with_builtins(),
        &[lj_pair_config()],
        Some(&spme_config_default()),
        &charges,
    )
    .unwrap();
    let labels: Vec<&str> = ff.slots.iter().map(|s| s.label()).collect();
    // Under JIT composition both standalone slots are present; their
    // pair-force kernels are fused into the JIT-composed kernel.
    assert!(labels.contains(&"lennard_jones"), "labels = {labels:?}");
    assert!(labels.contains(&"spme_real"), "labels = {labels:?}");
    assert!(labels.contains(&"spme_reciprocal"), "labels = {labels:?}");
    assert!(ff.jit_composed.is_some(), "JIT composed kernel built");
}

#[test]
fn lj_only_configuration_jit_composes_a_single_fragment() {
    let gpu = init_device().unwrap();
    let n = 4;
    let ff = build_with_spme(
        &gpu,
        n,
        &PotentialRegistry::with_builtins(),
        &[lj_pair_config()],
        None,
        &[],
    )
    .unwrap();
    let labels: Vec<&str> = ff.slots.iter().map(|s| s.label()).collect();
    assert!(labels.contains(&"lennard_jones"), "labels = {labels:?}");
    assert!(ff.jit_composed.is_some(), "JIT composed kernel built");
}

#[test]
fn spme_real_only_configuration_jit_composes_a_single_fragment() {
    let gpu = init_device().unwrap();
    let n = 4;
    let charges = vec![1.0 as Real, -1.0, 1.0, -1.0];
    let ff = build_with_spme(
        &gpu,
        n,
        &PotentialRegistry::with_builtins(),
        &[],
        Some(&spme_config_default()),
        &charges,
    )
    .unwrap();
    let labels: Vec<&str> = ff.slots.iter().map(|s| s.label()).collect();
    assert!(labels.contains(&"spme_real"), "labels = {labels:?}");
    assert!(ff.jit_composed.is_some(), "JIT composed kernel built");
}

#[test]
fn displacement_claim_against_unbuilt_label_is_a_no_op() {
    let gpu = init_device().unwrap();
    let mut registry = PotentialRegistry::new();
    let phantom_displacer = StubBuilder {
        displaces: &["never_built"],
        ..StubBuilder::new("phantom_displacer")
    };
    registry.register(Box::new(phantom_displacer));
    let ff = build_with(&gpu, 4, &registry).unwrap();
    assert_eq!(ff.slots.len(), 1);
    assert_eq!(ff.slots[0].label(), "phantom_displacer");
}

#[test]
fn two_builders_claiming_the_same_built_constituent_errors_at_construction() {
    let gpu = init_device().unwrap();
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder::new("target")));
    registry.register(Box::new(StubBuilder {
        displaces: &["target"],
        ..StubBuilder::new("first_claimer")
    }));
    registry.register(Box::new(StubBuilder {
        displaces: &["target"],
        ..StubBuilder::new("second_claimer")
    }));
    let err = build_with(&gpu, 4, &registry).unwrap_err();
    match err {
        ForceFieldError::DisplaceConflict { label, by } => {
            assert_eq!(label, "target");
            assert!(by.contains(&"first_claimer"));
            assert!(by.contains(&"second_claimer"));
        }
        other => panic!("expected DisplaceConflict, got {other:?}"),
    }
}

#[test]
fn two_builders_both_claiming_unbuilt_label_do_not_error() {
    let gpu = init_device().unwrap();
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(StubBuilder {
        displaces: &["nobody_built_this"],
        ..StubBuilder::new("a")
    }));
    registry.register(Box::new(StubBuilder {
        displaces: &["nobody_built_this"],
        ..StubBuilder::new("b")
    }));
    let ff = build_with(&gpu, 4, &registry).unwrap();
    let labels: Vec<&str> = ff.slots.iter().map(|s| s.label()).collect();
    assert!(labels.contains(&"a"));
    assert!(labels.contains(&"b"));
}

#[test]
fn slot_list_has_lj_then_spme_real_then_spme_reciprocal_under_lj_plus_spme() {
    let gpu = init_device().unwrap();
    let n = 4;
    let charges = vec![1.0 as Real, -1.0, 1.0, -1.0];
    let ff = build_with_spme(
        &gpu,
        n,
        &PotentialRegistry::with_builtins(),
        &[lj_pair_config()],
        Some(&spme_config_default()),
        &charges,
    )
    .unwrap();
    let labels: Vec<&str> = ff.slots.iter().map(|s| s.label()).collect();
    assert_eq!(labels, vec!["lennard_jones", "spme_real", "spme_reciprocal"]);
}

#[test]
fn neighbor_list_radius_is_max_of_active_pair_force_cutoffs() {
    let gpu = init_device().unwrap();
    let n = 4;
    let charges = vec![1.0 as Real, -1.0, 1.0, -1.0];
    let mut spme_cfg = spme_config_default();
    spme_cfg.r_cut_real = 7.0;
    let lj_cfg = lj_pair_config();
    let ff = build_with_spme(
        &gpu,
        n,
        &PotentialRegistry::with_builtins(),
        &[lj_cfg],
        Some(&spme_cfg),
        &charges,
    )
    .unwrap();
    // The framework picks max(slot.max_cutoff()) for the neighbor
    // list; the wider cutoff (SPME-real r_cut_real = 7.0) wins.
    let spme_real = ff
        .slots
        .iter()
        .find(|s| s.label() == "spme_real")
        .expect("spme_real slot present");
    let mc = spme_real.max_cutoff().expect("spme_real reports a cutoff");
    assert!((mc - 7.0 as Real).abs() < 1e-6, "max_cutoff = {mc}");
}

#[test]
fn jit_composed_configuration_step_byte_identical_run_to_run() {
    let gpu = init_device().unwrap();
    let n = 4;
    let charges = vec![1.0 as Real, -1.0, 1.0, -1.0];
    let run = || -> (Vec<Real>, Vec<Real>) {
        let mut ff = build_with_spme(
            &gpu,
            n,
            &PotentialRegistry::with_builtins(),
            &[lj_pair_config()],
            Some(&spme_config_default()),
            &charges,
        )
        .unwrap();
        let mut state = state_n(n);
        state.charges = charges.clone();
        let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
        let mut timings = Timings::new(&gpu).unwrap();
        ff.step(
            &mut buffers,
            &box_10(&gpu),
            &mut timings,
            AggregateLevel::ForcesAndScalars,
        )
        .unwrap();
        let fx = gpu.device.dtoh_sync_copy(&buffers.forces_x).unwrap();
        let e = gpu.device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
        (fx, e)
    };
    let a = run();
    let b = run();
    assert_eq!(a.0, b.0, "fused forces_x not byte-identical run-to-run");
    assert_eq!(a.1, b.1, "fused energies not byte-identical run-to-run");
}

#[test]
fn jit_composed_step_emits_one_pair_force_launch_and_zero_per_potential_launches() {
    let gpu = init_device().unwrap();
    let n = 4;
    let charges = vec![1.0 as Real, -1.0, 1.0, -1.0];
    let mut ff = build_with_spme(
        &gpu,
        n,
        &PotentialRegistry::with_builtins(),
        &[lj_pair_config()],
        Some(&spme_config_default()),
        &charges,
    )
    .unwrap();
    let mut state = state_n(n);
    state.charges = charges.clone();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    ff.step(
        &mut buffers,
        &box_10(&gpu),
        &mut timings,
        AggregateLevel::ForcesAndScalars,
    )
    .unwrap();
    let report = timings.finalize().unwrap();
    assert_eq!(
        stage_count(&report, KernelStage::JIT_COMPOSED_PAIR_FORCE.name()),
        1
    );
}
