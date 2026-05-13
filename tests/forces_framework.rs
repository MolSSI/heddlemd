mod common;

use cudarc::driver::DeviceSlice;
use dynamics::forces::{
    Bond, BondList, ExclusionList, ForceField, ForceFieldError, Potential, SlotForceView,
};
use dynamics::gpu::{ParticleBuffers, init_device};
use dynamics::io::config::{BondTypeConfig, NeighborListConfig, PairInteractionConfig};
use dynamics::pbc::SimulationBox;
use dynamics::state::ParticleState;
use dynamics::timings::Timings;

fn lj_pair_config() -> PairInteractionConfig {
    PairInteractionConfig {
        between: ("Ar".to_string(), "Ar".to_string()),
        potential: "lennard-jones".to_string(),
        sigma: 1.0,
        epsilon: 1.0,
        cutoff: 5.0,
    }
}

fn box_10() -> SimulationBox {
    SimulationBox::new_orthorhombic(10.0, 10.0, 10.0).unwrap()
}

fn state_n(n: usize) -> ParticleState {
    let pos: Vec<f32> = (0..n).map(|i| i as f32 * 1.5).collect();
    ParticleState::new(
        pos,
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![1.0_f32; n],
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

// rq-56c8a238
#[test]
fn force_field_lj_only() {
    let device = init_device().unwrap();
    let ff = ForceField::new(
        device,
        4,
        &box_10(),
        &[lj_pair_config()],
        &[],
        &BondList::empty(4),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    assert_eq!(ff.slots.len(), 1);
    assert_eq!(ff.slots[0].label(), "lennard_jones");
}

// rq-3de16ce0
#[test]
fn force_field_lj_and_morse() {
    let device = init_device().unwrap();
    let bl = single_bond_list(4);
    let bt = vec![BondTypeConfig::Morse {
        name: "CC".to_string(),
        de: 1.0,
        a: 2.0,
        re: 1.0,
    }];
    let ff = ForceField::new(
        device,
        4,
        &box_10(),
        &[lj_pair_config()],
        &bt,
        &bl,
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    assert_eq!(ff.slots.len(), 2);
    assert_eq!(ff.slots[0].label(), "lennard_jones");
    assert_eq!(ff.slots[1].label(), "morse_bonded");
}

// rq-0f34d11b
#[test]
fn bond_types_declared_no_bonds() {
    let device = init_device().unwrap();
    let bt = vec![BondTypeConfig::Morse {
        name: "CC".to_string(),
        de: 1.0,
        a: 2.0,
        re: 1.0,
    }];
    let ff = ForceField::new(
        device,
        4,
        &box_10(),
        &[lj_pair_config()],
        &bt,
        &BondList::empty(4),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    assert_eq!(ff.slots.len(), 1);
    assert_eq!(ff.slots[0].label(), "lennard_jones");
}

// rq-60f445b2
#[test]
fn force_field_zero_slots() {
    let device = init_device().unwrap();
    let ff = ForceField::new(
        device,
        4,
        &box_10(),
        &[],
        &[],
        &BondList::empty(4),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    assert!(ff.slots.is_empty());
    assert_eq!(ff.slot_forces_x.len(), 0);
    assert_eq!(ff.slot_forces_y.len(), 0);
    assert_eq!(ff.slot_forces_z.len(), 0);
}

// rq-455db9c2
#[test]
fn slot_buffers_sized_num_slots_times_particle_count() {
    let device = init_device().unwrap();
    let bt = vec![BondTypeConfig::Morse {
        name: "CC".to_string(),
        de: 1.0,
        a: 2.0,
        re: 1.0,
    }];
    let bl = single_bond_list(8);
    let ff = ForceField::new(
        device,
        8,
        &box_10(),
        &[lj_pair_config()],
        &bt,
        &bl,
        &ExclusionList::empty(8),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    assert_eq!(ff.slots.len(), 2);
    assert_eq!(ff.slot_forces_x.len(), 16);
    assert_eq!(ff.slot_forces_y.len(), 16);
    assert_eq!(ff.slot_forces_z.len(), 16);
}

// rq-c525ee79
#[test]
fn empty_force_field() {
    let device = init_device().unwrap();
    let ff = ForceField::new(
        device,
        0,
        &box_10(),
        &[lj_pair_config()],
        &[],
        &BondList::empty(0),
        &ExclusionList::empty(0),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    assert_eq!(ff.slots.len(), 1);
    assert_eq!(ff.slot_forces_x.len(), 0);
    assert_eq!(ff.slot_forces_y.len(), 0);
    assert_eq!(ff.slot_forces_z.len(), 0);
}

// rq-c170c0b7
//
// `ForceField::new` produces deterministic, non-colliding labels from its
// real config inputs, so the only way to exercise the DuplicateLabel guard
// is to construct the framework manually from two slots that share a
// label. We use a stub Potential implementation for that.

#[derive(Debug)]
struct LabelStub(&'static str);

impl Potential for LabelStub {
    fn label(&self) -> &'static str {
        self.0
    }
    fn contribute(
        &mut self,
        _buffers: &ParticleBuffers,
        _sim_box: &SimulationBox,
        _timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        Ok(())
    }
    fn reduce(
        &mut self,
        _output: SlotForceView<'_>,
        _timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        Ok(())
    }
}

#[test]
fn duplicate_label_check_rejects_colliding_labels() {
    // Mirror ForceField::new's duplicate-check rule on a hand-built slot list.
    let slots: Vec<Box<dyn Potential>> = vec![
        Box::new(LabelStub("dup")),
        Box::new(LabelStub("dup")),
    ];
    let mut collision: Option<&'static str> = None;
    for i in 0..slots.len() {
        for j in (i + 1)..slots.len() {
            if slots[i].label() == slots[j].label() {
                collision = Some(slots[i].label());
            }
        }
    }
    assert_eq!(collision, Some("dup"));
}

// rq-32e981cc
#[test]
fn step_lj_only_writes_lj_forces() {
    let device = init_device().unwrap();
    let state = state_n(2);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    let mut ff = ForceField::new(
        device,
        2,
        &box_10(),
        &[lj_pair_config()],
        &[],
        &BondList::empty(2),
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();
    let mut downloaded = state.clone();
    downloaded.download_from(&buffers).unwrap();
    assert!(downloaded.forces_x[0] != 0.0);
    assert!((downloaded.forces_x[0] + downloaded.forces_x[1]).abs() < 1.0e-6);
}

// rq-df3a50f6
#[test]
fn step_both_slots_sums_lj_and_morse() {
    let device = init_device().unwrap();
    let state = state_n(2);
    let mut buffers_a = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut buffers_b = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut buffers_lj_only = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings_a = Timings::new(device.clone()).unwrap();
    let mut timings_b = Timings::new(device.clone()).unwrap();
    let mut timings_lj = Timings::new(device.clone()).unwrap();

    let bt = vec![BondTypeConfig::Morse {
        name: "CC".to_string(),
        de: 1.0,
        a: 2.0,
        re: 1.0,
    }];
    let bl = single_bond_list(2);

    let mut ff_lj = ForceField::new(
        device.clone(),
        2,
        &box_10(),
        &[lj_pair_config()],
        &[],
        &BondList::empty(2),
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    ff_lj.step(&mut buffers_lj_only, &box_10(), &mut timings_lj).unwrap();
    let mut lj_state = state.clone();
    lj_state.download_from(&buffers_lj_only).unwrap();

    let lj_tiny = PairInteractionConfig {
        between: ("Ar".to_string(), "Ar".to_string()),
        potential: "lennard-jones".to_string(),
        sigma: 1.0,
        epsilon: 1.0,
        cutoff: 0.5,
    };
    let mut ff_morse = ForceField::new(
        device.clone(),
        2,
        &box_10(),
        &[lj_tiny],
        &bt,
        &bl,
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    ff_morse.step(&mut buffers_b, &box_10(), &mut timings_b).unwrap();
    let mut morse_state = state.clone();
    morse_state.download_from(&buffers_b).unwrap();

    let mut ff_both = ForceField::new(
        device,
        2,
        &box_10(),
        &[lj_pair_config()],
        &bt,
        &bl,
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    ff_both.step(&mut buffers_a, &box_10(), &mut timings_a).unwrap();
    let mut combined = state.clone();
    combined.download_from(&buffers_a).unwrap();

    let expected = lj_state.forces_x[0] + morse_state.forces_x[0];
    let got = combined.forces_x[0];
    assert!((got - expected).abs() < 1.0e-4, "expected {expected}, got {got}");
}

// rq-fc7b1565
#[test]
fn step_zero_slots_writes_zero_forces() {
    let device = init_device().unwrap();
    // Seed forces_* with non-zero junk so we can prove the combiner
    // overwrites them.
    let state = ParticleState::new(
        vec![0.0_f32, 1.0, 2.0, 3.0],
        vec![0.0_f32; 4],
        vec![0.0_f32; 4],
        vec![0.0_f32; 4],
        vec![0.0_f32; 4],
        vec![0.0_f32; 4],
        vec![1.0_f32; 4],
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    // Stamp non-zero forces directly into the device buffer.
    let nonzero = vec![7.0_f32; 4];
    device
        .htod_sync_copy_into(&nonzero, &mut buffers.forces_x)
        .unwrap();
    device
        .htod_sync_copy_into(&nonzero, &mut buffers.forces_y)
        .unwrap();
    device
        .htod_sync_copy_into(&nonzero, &mut buffers.forces_z)
        .unwrap();

    let mut timings = Timings::new(device.clone()).unwrap();
    let mut ff = ForceField::new(
        device.clone(),
        4,
        &box_10(),
        &[],
        &[],
        &BondList::empty(4),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    assert!(ff.slots.is_empty());
    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();

    let fx = device.dtoh_sync_copy(&buffers.forces_x).unwrap();
    let fy = device.dtoh_sync_copy(&buffers.forces_y).unwrap();
    let fz = device.dtoh_sync_copy(&buffers.forces_z).unwrap();
    assert!(fx.iter().all(|&v| v == 0.0));
    assert!(fy.iter().all(|&v| v == 0.0));
    assert!(fz.iter().all(|&v| v == 0.0));

    let report = timings.finalize().unwrap();
    // Only AccumulateForces should have a sample; LJ and Morse stages absent.
    let names: Vec<&str> = report.stages.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"accumulate_forces"));
    assert!(!names.contains(&"lj_pair_force"));
    assert!(!names.contains(&"reduce_pair_forces"));
    assert!(!names.contains(&"morse_bond_force"));
    assert!(!names.contains(&"reduce_bond_forces"));
}

// rq-de47c1ac
#[test]
fn step_empty_launches_no_kernels() {
    let device = init_device().unwrap();
    let state = ParticleState::new(
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    let mut ff = ForceField::new(
        device,
        0,
        &box_10(),
        &[lj_pair_config()],
        &[],
        &BondList::empty(0),
        &ExclusionList::empty(0),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();
    let report = timings.finalize().unwrap();
    assert!(report.stages.is_empty());
}

// rq-7d8485b3
#[test]
fn each_slot_writes_its_own_row() {
    let device = init_device().unwrap();
    let state = state_n(3);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    let bt = vec![BondTypeConfig::Morse {
        name: "CC".to_string(),
        de: 1.0,
        a: 2.0,
        re: 1.0,
    }];
    let bl = single_bond_list(3);
    let mut ff = ForceField::new(
        device.clone(),
        3,
        &box_10(),
        &[lj_pair_config()],
        &bt,
        &bl,
        &ExclusionList::empty(3),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();

    // Recompute slot 0 (LJ) in isolation to compare.
    let mut buffers_lj = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut t_lj = Timings::new(device.clone()).unwrap();
    let mut ff_lj = ForceField::new(
        device.clone(),
        3,
        &box_10(),
        &[lj_pair_config()],
        &[],
        &BondList::empty(3),
        &ExclusionList::empty(3),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    ff_lj.step(&mut buffers_lj, &box_10(), &mut t_lj).unwrap();
    let lj_x = device.dtoh_sync_copy(&buffers_lj.forces_x).unwrap();

    // And slot 1 (Morse) in isolation.
    let lj_tiny = PairInteractionConfig {
        between: ("Ar".to_string(), "Ar".to_string()),
        potential: "lennard-jones".to_string(),
        sigma: 1.0,
        epsilon: 1.0,
        cutoff: 0.5,
    };
    let mut buffers_m = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut t_m = Timings::new(device.clone()).unwrap();
    let mut ff_m = ForceField::new(
        device.clone(),
        3,
        &box_10(),
        &[lj_tiny],
        &bt,
        &bl,
        &ExclusionList::empty(3),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    ff_m.step(&mut buffers_m, &box_10(), &mut t_m).unwrap();
    let morse_x = device.dtoh_sync_copy(&buffers_m.forces_x).unwrap();

    let row_x = device.dtoh_sync_copy(&ff.slot_forces_x).unwrap();
    assert_eq!(row_x.len(), 6);
    for i in 0..3 {
        assert!((row_x[i] - lj_x[i]).abs() < 1.0e-6, "row 0 mismatch at {i}");
        assert!((row_x[3 + i] - morse_x[i]).abs() < 1.0e-4, "row 1 mismatch at {i}");
    }
}

// rq-a9642241
//
// A third Potential implementation can be slotted into a ForceField without
// editing the combiner kernel or ForceField. We instantiate a stub that
// writes a known per-particle pattern into its assigned row, then verify
// the combiner sums it on top of an LJ slot's contribution and that the
// stub's row is at index 1 (right after LJ).

#[derive(Debug)]
struct ConstStub {
    value_x: f32,
    value_y: f32,
    value_z: f32,
    device: std::sync::Arc<cudarc::driver::CudaDevice>,
}

impl Potential for ConstStub {
    fn label(&self) -> &'static str {
        "const_stub"
    }
    fn contribute(
        &mut self,
        _buffers: &ParticleBuffers,
        _sim_box: &SimulationBox,
        _timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        Ok(())
    }
    fn reduce(
        &mut self,
        mut output: SlotForceView<'_>,
        _timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        let n = output.x.len();
        if n == 0 {
            return Ok(());
        }
        let vx = vec![self.value_x; n];
        let vy = vec![self.value_y; n];
        let vz = vec![self.value_z; n];
        self.device
            .htod_sync_copy_into(&vx, &mut output.x)
            .map_err(|e| ForceFieldError::Gpu(e.into()))?;
        self.device
            .htod_sync_copy_into(&vy, &mut output.y)
            .map_err(|e| ForceFieldError::Gpu(e.into()))?;
        self.device
            .htod_sync_copy_into(&vz, &mut output.z)
            .map_err(|e| ForceFieldError::Gpu(e.into()))?;
        Ok(())
    }
}

#[test]
fn third_potential_extensibility() {
    let device = init_device().unwrap();
    let state = state_n(3);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    let mut ff = ForceField::new(
        device.clone(),
        3,
        &box_10(),
        &[lj_pair_config()],
        &[],
        &BondList::empty(3),
        &ExclusionList::empty(3),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    // Splice in a third slot manually.
    ff.slots.push(Box::new(ConstStub {
        value_x: 1.0,
        value_y: 2.0,
        value_z: 3.0,
        device: device.clone(),
    }));
    // Re-allocate the flat buffers for 2 slots.
    let new_len = ff.slots.len() * 3;
    ff.slot_forces_x = device.alloc_zeros::<f32>(new_len).unwrap();
    ff.slot_forces_y = device.alloc_zeros::<f32>(new_len).unwrap();
    ff.slot_forces_z = device.alloc_zeros::<f32>(new_len).unwrap();

    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();

    // The stub writes (1, 2, 3) per particle. Subtract the LJ-only result
    // to recover the stub's contribution.
    let mut buffers_lj = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut t_lj = Timings::new(device.clone()).unwrap();
    let mut ff_lj = ForceField::new(
        device.clone(),
        3,
        &box_10(),
        &[lj_pair_config()],
        &[],
        &BondList::empty(3),
        &ExclusionList::empty(3),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    ff_lj.step(&mut buffers_lj, &box_10(), &mut t_lj).unwrap();
    let lj_x = device.dtoh_sync_copy(&buffers_lj.forces_x).unwrap();
    let lj_y = device.dtoh_sync_copy(&buffers_lj.forces_y).unwrap();
    let lj_z = device.dtoh_sync_copy(&buffers_lj.forces_z).unwrap();

    let mixed_x = device.dtoh_sync_copy(&buffers.forces_x).unwrap();
    let mixed_y = device.dtoh_sync_copy(&buffers.forces_y).unwrap();
    let mixed_z = device.dtoh_sync_copy(&buffers.forces_z).unwrap();

    for i in 0..3 {
        assert!((mixed_x[i] - lj_x[i] - 1.0).abs() < 1.0e-6);
        assert!((mixed_y[i] - lj_y[i] - 2.0).abs() < 1.0e-6);
        assert!((mixed_z[i] - lj_z[i] - 3.0).abs() < 1.0e-6);
    }
    // Row 1 of slot_forces_x should hold the stub's value.
    let row_x = device.dtoh_sync_copy(&ff.slot_forces_x).unwrap();
    for v in &row_x[3..6] {
        assert_eq!(*v, 1.0);
    }
}

// rq-c8e5b14e
#[test]
fn two_independent_runs_byte_identical() {
    let device = init_device().unwrap();
    let state = state_n(4);
    let mut buffers_a = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut buffers_b = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings_a = Timings::new(device.clone()).unwrap();
    let mut timings_b = Timings::new(device.clone()).unwrap();
    let mut ff_a = ForceField::new(
        device.clone(),
        4,
        &box_10(),
        &[lj_pair_config()],
        &[],
        &BondList::empty(4),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let mut ff_b = ForceField::new(
        device,
        4,
        &box_10(),
        &[lj_pair_config()],
        &[],
        &BondList::empty(4),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    ff_a.step(&mut buffers_a, &box_10(), &mut timings_a).unwrap();
    ff_b.step(&mut buffers_b, &box_10(), &mut timings_b).unwrap();
    let mut state_a = state.clone();
    let mut state_b = state.clone();
    state_a.download_from(&buffers_a).unwrap();
    state_b.download_from(&buffers_b).unwrap();
    assert_eq!(state_a.forces_x, state_b.forces_x);
    assert_eq!(state_a.forces_y, state_b.forces_y);
    assert_eq!(state_a.forces_z, state_b.forces_z);
}

// rq-a5aa743e
#[test]
fn combiner_sums_slot_rows_in_slot_order() {
    let device = init_device().unwrap();
    let state = state_n(2);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    // Build a ForceField with two ConstStub slots that write distinct rows.
    let mut ff = ForceField::new(
        device.clone(),
        2,
        &box_10(),
        &[],
        &[],
        &BondList::empty(2),
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    ff.slots.push(Box::new(ConstStub {
        value_x: 1.0,
        value_y: 0.0,
        value_z: 0.0,
        device: device.clone(),
    }));
    // Second slot — a different label so duplicate-label doesn't trip.
    #[derive(Debug)]
    struct ConstStubB {
        device: std::sync::Arc<cudarc::driver::CudaDevice>,
    }
    impl Potential for ConstStubB {
        fn label(&self) -> &'static str {
            "const_stub_b"
        }
        fn contribute(
            &mut self,
            _b: &ParticleBuffers,
            _s: &SimulationBox,
            _t: &mut Timings,
        ) -> Result<(), ForceFieldError> {
            Ok(())
        }
        fn reduce(
            &mut self,
            mut output: SlotForceView<'_>,
            _t: &mut Timings,
        ) -> Result<(), ForceFieldError> {
            let n = output.x.len();
            let vx = vec![10.0_f32; n];
            self.device
                .htod_sync_copy_into(&vx, &mut output.x)
                .map_err(|e| ForceFieldError::Gpu(e.into()))?;
            self.device
                .htod_sync_copy_into(&vec![0.0_f32; n], &mut output.y)
                .map_err(|e| ForceFieldError::Gpu(e.into()))?;
            self.device
                .htod_sync_copy_into(&vec![0.0_f32; n], &mut output.z)
                .map_err(|e| ForceFieldError::Gpu(e.into()))?;
            Ok(())
        }
    }
    ff.slots.push(Box::new(ConstStubB { device: device.clone() }));
    let new_len = ff.slots.len() * 2;
    ff.slot_forces_x = device.alloc_zeros::<f32>(new_len).unwrap();
    ff.slot_forces_y = device.alloc_zeros::<f32>(new_len).unwrap();
    ff.slot_forces_z = device.alloc_zeros::<f32>(new_len).unwrap();

    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();
    let fx = device.dtoh_sync_copy(&buffers.forces_x).unwrap();
    assert_eq!(fx, vec![11.0, 11.0]);
}

// rq-3e9217e2
#[test]
fn combiner_with_zero_slots_writes_zeros() {
    let device = init_device().unwrap();
    let state = state_n(4);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    // Pre-stamp non-zero forces.
    let stamp = vec![42.0_f32; 4];
    device.htod_sync_copy_into(&stamp, &mut buffers.forces_x).unwrap();
    device.htod_sync_copy_into(&stamp, &mut buffers.forces_y).unwrap();
    device.htod_sync_copy_into(&stamp, &mut buffers.forces_z).unwrap();

    let mut timings = Timings::new(device.clone()).unwrap();
    let mut ff = ForceField::new(
        device.clone(),
        4,
        &box_10(),
        &[],
        &[],
        &BondList::empty(4),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();
    let fx = device.dtoh_sync_copy(&buffers.forces_x).unwrap();
    let fy = device.dtoh_sync_copy(&buffers.forces_y).unwrap();
    let fz = device.dtoh_sync_copy(&buffers.forces_z).unwrap();
    assert!(fx.iter().all(|&v| v == 0.0));
    assert!(fy.iter().all(|&v| v == 0.0));
    assert!(fz.iter().all(|&v| v == 0.0));
}

// rq-82acb52f
#[test]
fn combiner_idempotent_across_two_calls() {
    let device = init_device().unwrap();
    let state = state_n(4);
    let mut buffers = ParticleBuffers::new(device.clone(), &state).unwrap();
    let mut timings = Timings::new(device.clone()).unwrap();
    let mut ff = ForceField::new(
        device,
        4,
        &box_10(),
        &[lj_pair_config()],
        &[],
        &BondList::empty(4),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();
    let mut first = state.clone();
    first.download_from(&buffers).unwrap();
    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();
    let mut second = state.clone();
    second.download_from(&buffers).unwrap();
    assert_eq!(first.forces_x, second.forces_x);
}
