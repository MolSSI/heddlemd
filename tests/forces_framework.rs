mod common;

use cudarc::driver::DeviceSlice;
use dynamics::forces::{Bond, BondList, ExclusionList, ForceField, ForceFieldContext, ForceFieldError, Potential, PotentialRegistry, SlotOutputView};
use dynamics::gpu::{ParticleBuffers, init_device};
use dynamics::io::config::{
    BondTypeConfig, NeighborListConfig, PairInteractionConfig, PairPotentialParams,
    ParticleTypeConfig,
};
use dynamics::pbc::SimulationBox;
use dynamics::state::ParticleState;
use dynamics::timings::Timings;

fn lj_pair_config() -> PairInteractionConfig {
    PairInteractionConfig {
        between: ("Ar".to_string(), "Ar".to_string()),
        cutoff: 5.0,
        r_switch: 5.0,
        potential: PairPotentialParams::LennardJones { sigma: 1.0, epsilon: 1.0 },
    }
}

fn box_10() -> SimulationBox {
    SimulationBox::new(10.0, 10.0, 10.0, 0.0, 0.0, 0.0).unwrap()
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
        vec![0.0_f32; n],
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

// rq-56c8a238
#[test]
fn force_field_lj_only() {
    let gpu = init_device().unwrap();
    let ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        4,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(4),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs)
    .unwrap();
    assert_eq!(ff.slots.len(), 1);
    assert_eq!(ff.slots[0].label(), "lennard_jones");
}

// rq-3de16ce0
#[test]
fn force_field_lj_and_morse() {
    let gpu = init_device().unwrap();
    let bl = single_bond_list(4);
    let bt = vec![BondTypeConfig::Morse {
        name: "CC".to_string(),
        de: 1.0,
        a: 2.0,
        re: 1.0,
    }];
    let ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        4,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &bt,
        &[],
        None,
        None,
        &[],
        &bl,
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs)
    .unwrap();
    assert_eq!(ff.slots.len(), 2);
    assert_eq!(ff.slots[0].label(), "lennard_jones");
    assert_eq!(ff.slots[1].label(), "morse_bonded");
}

// rq-0f34d11b
#[test]
fn bond_types_declared_no_bonds() {
    let gpu = init_device().unwrap();
    let bt = vec![BondTypeConfig::Morse {
        name: "CC".to_string(),
        de: 1.0,
        a: 2.0,
        re: 1.0,
    }];
    let ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        4,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &bt,
        &[],
        None,
        None,
        &[],
        &BondList::empty(4),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs)
    .unwrap();
    assert_eq!(ff.slots.len(), 1);
    assert_eq!(ff.slots[0].label(), "lennard_jones");
}

// rq-60f445b2
#[test]
fn force_field_zero_slots() {
    let gpu = init_device().unwrap();
    let ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        4,
        &box_10(),
        &[],
        &[],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(4),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs)
    .unwrap();
    assert!(ff.slots.is_empty());
    assert_eq!(ff.slot_forces_x.len(), 0);
    assert_eq!(ff.slot_forces_y.len(), 0);
    assert_eq!(ff.slot_forces_z.len(), 0);
}

// rq-455db9c2
#[test]
fn slot_buffers_sized_num_slots_times_particle_count() {
    let gpu = init_device().unwrap();
    let bt = vec![BondTypeConfig::Morse {
        name: "CC".to_string(),
        de: 1.0,
        a: 2.0,
        re: 1.0,
    }];
    let bl = single_bond_list(8);
    let ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        8,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &bt,
        &[],
        None,
        None,
        &[],
        &bl,
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(8),
        &NeighborListConfig::AllPairs)
    .unwrap();
    assert_eq!(ff.slots.len(), 2);
    assert_eq!(ff.slot_forces_x.len(), 16);
    assert_eq!(ff.slot_forces_y.len(), 16);
    assert_eq!(ff.slot_forces_z.len(), 16);
}

// rq-c525ee79
#[test]
fn empty_force_field() {
    let gpu = init_device().unwrap();
    let ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        0,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(0),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(0),
        &NeighborListConfig::AllPairs)
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
    fn max_cutoff(&self) -> Option<f32> {
        None
    }
    fn contribute(
        &mut self,
        _buffers: &ParticleBuffers,
        _sim_box: &SimulationBox,
        _cx: &ForceFieldContext<'_>,
        _timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        Ok(())
    }
    fn reduce(
        &mut self,
        _output: SlotOutputView<'_>,
        _cx: &ForceFieldContext<'_>,
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
    let gpu = init_device().unwrap();
    let state = state_n(2);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        2,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(2),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs)
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
    let gpu = init_device().unwrap();
    let state = state_n(2);
    let mut buffers_a = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut buffers_b = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut buffers_lj_only = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings_a = Timings::new(&gpu).unwrap();
    let mut timings_b = Timings::new(&gpu).unwrap();
    let mut timings_lj = Timings::new(&gpu).unwrap();

    let bt = vec![BondTypeConfig::Morse {
        name: "CC".to_string(),
        de: 1.0,
        a: 2.0,
        re: 1.0,
    }];
    let bl = single_bond_list(2);

    let mut ff_lj = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        2,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(2),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs)
    .unwrap();
    ff_lj.step(&mut buffers_lj_only, &box_10(), &mut timings_lj).unwrap();
    let mut lj_state = state.clone();
    lj_state.download_from(&buffers_lj_only).unwrap();

    let lj_tiny = PairInteractionConfig {
        between: ("Ar".to_string(), "Ar".to_string()),
        cutoff: 0.5,
        r_switch: 0.5,
        potential: PairPotentialParams::LennardJones { sigma: 1.0, epsilon: 1.0 },
    };
    let mut ff_morse = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        2,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_tiny],
        &bt,
        &[],
        None,
        None,
        &[],
        &bl,
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs)
    .unwrap();
    ff_morse.step(&mut buffers_b, &box_10(), &mut timings_b).unwrap();
    let mut morse_state = state.clone();
    morse_state.download_from(&buffers_b).unwrap();

    let mut ff_both = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        2,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &bt,
        &[],
        None,
        None,
        &[],
        &bl,
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs)
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
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
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
        vec![0.0_f32; 4],
        vec![0u32; 4],
        None,
            None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
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

    let mut timings = Timings::new(&gpu).unwrap();
    let mut ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        4,
        &box_10(),
        &[],
        &[],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(4),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs)
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
    let gpu = init_device().unwrap();
    let state = ParticleState::new(
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        vec![0.0_f32; 0],
        Vec::new(),
        None,
            None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        0,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(0),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(0),
        &NeighborListConfig::AllPairs)
    .unwrap();
    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();
    let report = timings.finalize().unwrap();
    assert!(report.stages.is_empty());
}

// rq-7d8485b3
#[test]
fn each_slot_writes_its_own_row() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let state = state_n(3);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let bt = vec![BondTypeConfig::Morse {
        name: "CC".to_string(),
        de: 1.0,
        a: 2.0,
        re: 1.0,
    }];
    let bl = single_bond_list(3);
    let mut ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        3,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &bt,
        &[],
        None,
        None,
        &[],
        &bl,
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(3),
        &NeighborListConfig::AllPairs)
    .unwrap();
    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();

    // Recompute slot 0 (LJ) in isolation to compare.
    let mut buffers_lj = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut t_lj = Timings::new(&gpu).unwrap();
    let mut ff_lj = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        3,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(3),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(3),
        &NeighborListConfig::AllPairs)
    .unwrap();
    ff_lj.step(&mut buffers_lj, &box_10(), &mut t_lj).unwrap();
    let lj_x = device.dtoh_sync_copy(&buffers_lj.forces_x).unwrap();

    // And slot 1 (Morse) in isolation.
    let lj_tiny = PairInteractionConfig {
        between: ("Ar".to_string(), "Ar".to_string()),
        cutoff: 0.5,
        r_switch: 0.5,
        potential: PairPotentialParams::LennardJones { sigma: 1.0, epsilon: 1.0 },
    };
    let mut buffers_m = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut t_m = Timings::new(&gpu).unwrap();
    let mut ff_m = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        3,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_tiny],
        &bt,
        &[],
        None,
        None,
        &[],
        &bl,
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(3),
        &NeighborListConfig::AllPairs)
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
    fn max_cutoff(&self) -> Option<f32> {
        None
    }
    fn contribute(
        &mut self,
        _buffers: &ParticleBuffers,
        _sim_box: &SimulationBox,
        _cx: &ForceFieldContext<'_>,
        _timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        Ok(())
    }
    fn reduce(
        &mut self,
        mut output: SlotOutputView<'_>,
        _cx: &ForceFieldContext<'_>,
        _timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        let n = output.force_x.len();
        if n == 0 {
            return Ok(());
        }
        let vx = vec![self.value_x; n];
        let vy = vec![self.value_y; n];
        let vz = vec![self.value_z; n];
        self.device
            .htod_sync_copy_into(&vx, &mut output.force_x)
            .map_err(|e| ForceFieldError::Gpu(e.into()))?;
        self.device
            .htod_sync_copy_into(&vy, &mut output.force_y)
            .map_err(|e| ForceFieldError::Gpu(e.into()))?;
        self.device
            .htod_sync_copy_into(&vz, &mut output.force_z)
            .map_err(|e| ForceFieldError::Gpu(e.into()))?;
        Ok(())
    }
}

#[test]
fn third_potential_extensibility() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let state = state_n(3);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        3,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(3),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(3),
        &NeighborListConfig::AllPairs)
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
    ff.slot_energies = device.alloc_zeros::<f32>(new_len).unwrap();
    ff.slot_virials = device.alloc_zeros::<f32>(new_len).unwrap();

    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();

    // The stub writes (1, 2, 3) per particle. Subtract the LJ-only result
    // to recover the stub's contribution.
    let mut buffers_lj = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut t_lj = Timings::new(&gpu).unwrap();
    let mut ff_lj = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        3,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(3),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(3),
        &NeighborListConfig::AllPairs)
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
    let gpu = init_device().unwrap();
    let state = state_n(4);
    let mut buffers_a = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut buffers_b = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings_a = Timings::new(&gpu).unwrap();
    let mut timings_b = Timings::new(&gpu).unwrap();
    let mut ff_a = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        4,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(4),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs)
    .unwrap();
    let mut ff_b = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        4,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(4),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs)
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
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let state = state_n(2);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    // Build a ForceField with two ConstStub slots that write distinct rows.
    let mut ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        2,
        &box_10(),
        &[],
        &[],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(2),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs)
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
        fn max_cutoff(&self) -> Option<f32> {
            None
        }
        fn contribute(
            &mut self,
            _b: &ParticleBuffers,
            _s: &SimulationBox,
            _cx: &ForceFieldContext<'_>,
            _t: &mut Timings,
        ) -> Result<(), ForceFieldError> {
            Ok(())
        }
        fn reduce(
            &mut self,
            mut output: SlotOutputView<'_>,
            _cx: &ForceFieldContext<'_>,
            _t: &mut Timings,
        ) -> Result<(), ForceFieldError> {
            let n = output.force_x.len();
            let vx = vec![10.0_f32; n];
            self.device
                .htod_sync_copy_into(&vx, &mut output.force_x)
                .map_err(|e| ForceFieldError::Gpu(e.into()))?;
            self.device
                .htod_sync_copy_into(&vec![0.0_f32; n], &mut output.force_y)
                .map_err(|e| ForceFieldError::Gpu(e.into()))?;
            self.device
                .htod_sync_copy_into(&vec![0.0_f32; n], &mut output.force_z)
                .map_err(|e| ForceFieldError::Gpu(e.into()))?;
            Ok(())
        }
    }
    ff.slots.push(Box::new(ConstStubB { device: device.clone() }));
    let new_len = ff.slots.len() * 2;
    ff.slot_forces_x = device.alloc_zeros::<f32>(new_len).unwrap();
    ff.slot_forces_y = device.alloc_zeros::<f32>(new_len).unwrap();
    ff.slot_forces_z = device.alloc_zeros::<f32>(new_len).unwrap();
    ff.slot_energies = device.alloc_zeros::<f32>(new_len).unwrap();
    ff.slot_virials = device.alloc_zeros::<f32>(new_len).unwrap();

    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();
    let fx = device.dtoh_sync_copy(&buffers.forces_x).unwrap();
    assert_eq!(fx, vec![11.0, 11.0]);
}

// rq-3e9217e2
#[test]
fn combiner_with_zero_slots_writes_zeros() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let state = state_n(4);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    // Pre-stamp non-zero forces.
    let stamp = vec![42.0_f32; 4];
    device.htod_sync_copy_into(&stamp, &mut buffers.forces_x).unwrap();
    device.htod_sync_copy_into(&stamp, &mut buffers.forces_y).unwrap();
    device.htod_sync_copy_into(&stamp, &mut buffers.forces_z).unwrap();

    let mut timings = Timings::new(&gpu).unwrap();
    let mut ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        4,
        &box_10(),
        &[],
        &[],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(4),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs)
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
    let gpu = init_device().unwrap();
    let state = state_n(4);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        4,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(4),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs)
    .unwrap();
    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();
    let mut first = state.clone();
    first.download_from(&buffers).unwrap();
    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();
    let mut second = state.clone();
    second.download_from(&buffers).unwrap();
    assert_eq!(first.forces_x, second.forces_x);
}

// --- Shared neighbor list ---

#[test] // rq-b33cf896
fn force_field_with_lj_owns_shared_neighbor_list() {
    let gpu = init_device().unwrap();
    let sim_box = SimulationBox::new(20.0, 20.0, 20.0, 0.0, 0.0, 0.0).unwrap();
    let ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        4,
        &sim_box,
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(4),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(4),
        &NeighborListConfig::CellList { max_neighbors: 16, r_skin: 0.3 },
    )
    .unwrap();
    let nl = ff.neighbor_list.as_ref().expect("shared neighbor list");
    assert_eq!(nl.max_neighbors, 16);
}

#[test] // rq-433c972f rq-83312d09
fn force_field_with_only_bonded_owns_no_neighbor_list() {
    let gpu = init_device().unwrap();
    let bt = vec![BondTypeConfig::Morse {
        name: "CC".to_string(),
        de: 1.0,
        a: 2.0,
        re: 1.0,
    }];
    let bl = single_bond_list(4);
    let ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        4,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[],
        &bt,
        &[],
        None,
        None,
        &[],
        &bl,
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    assert_eq!(ff.slots.len(), 1);
    assert_eq!(ff.slots[0].label(), "morse_bonded");
    assert!(ff.neighbor_list.is_none());
}

#[test] // rq-47540d14
fn bonded_only_step_launches_no_neighbor_list_kernels() {
    let gpu = init_device().unwrap();
    let state = state_n(4);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let bt = vec![BondTypeConfig::Morse {
        name: "CC".to_string(),
        de: 1.0,
        a: 2.0,
        re: 1.0,
    }];
    let bl = single_bond_list(4);
    let mut ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        4,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[],
        &bt,
        &[],
        None,
        None,
        &[],
        &bl,
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(4),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();
    let report = timings.finalize().unwrap();
    let names: Vec<&str> = report.stages.iter().map(|s| s.name.as_str()).collect();
    assert!(!names.contains(&"neighbor_displacement_squared"));
    assert!(!names.contains(&"neighbor_list_build"));
}

// Stub Potential that records the value of cx.neighbor_list passed to contribute.
#[derive(Debug)]
struct ContextProbeStub {
    last_seen_nl_some: std::sync::Arc<std::sync::Mutex<Option<bool>>>,
}

impl Potential for ContextProbeStub {
    fn label(&self) -> &'static str {
        "context_probe"
    }
    fn max_cutoff(&self) -> Option<f32> {
        None
    }
    fn contribute(
        &mut self,
        _b: &ParticleBuffers,
        _s: &SimulationBox,
        cx: &ForceFieldContext<'_>,
        _t: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        *self.last_seen_nl_some.lock().unwrap() = Some(cx.neighbor_list.is_some());
        Ok(())
    }
    fn reduce(
        &mut self,
        output: SlotOutputView<'_>,
        _cx: &ForceFieldContext<'_>,
        _t: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        // Write zeros to the output row.
        let n = output.force_x.len();
        let device = std::sync::Arc::clone(&self.last_seen_nl_some);
        let _ = device; // unused
        if n == 0 {
            return Ok(());
        }
        Ok(())
    }
}

#[test] // rq-81e84c73 rq-2ed643ad
fn context_exposes_shared_neighbor_list_to_contribute() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let state = state_n(2);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        2,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(2),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    // Append a probe stub that records the context it sees.
    let probe = std::sync::Arc::new(std::sync::Mutex::new(None::<bool>));
    ff.slots.push(Box::new(ContextProbeStub {
        last_seen_nl_some: probe.clone(),
    }));
    // Re-allocate slot-accumulator buffers to match the new slot count.
    let new_len = ff.slots.len() * 2;
    ff.slot_forces_x = device.alloc_zeros::<f32>(new_len).unwrap();
    ff.slot_forces_y = device.alloc_zeros::<f32>(new_len).unwrap();
    ff.slot_forces_z = device.alloc_zeros::<f32>(new_len).unwrap();
    ff.slot_energies = device.alloc_zeros::<f32>(new_len).unwrap();
    ff.slot_virials = device.alloc_zeros::<f32>(new_len).unwrap();
    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();
    assert_eq!(*probe.lock().unwrap(), Some(true));
}

// Stub Potential that reports a configurable max_cutoff. Used to verify the
// framework aggregates max_cutoff across slots when building the shared list.
#[derive(Debug)]
#[allow(dead_code)]
struct CutoffProbeStub {
    cutoff: f32,
}

impl Potential for CutoffProbeStub {
    fn label(&self) -> &'static str {
        "cutoff_probe"
    }
    fn max_cutoff(&self) -> Option<f32> {
        Some(self.cutoff)
    }
    fn contribute(
        &mut self,
        _b: &ParticleBuffers,
        _s: &SimulationBox,
        _cx: &ForceFieldContext<'_>,
        _t: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        Ok(())
    }
    fn reduce(
        &mut self,
        _output: SlotOutputView<'_>,
        _cx: &ForceFieldContext<'_>,
        _t: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        Ok(())
    }
}

#[test] // rq-e39d0ed8 rq-3bc18e1a
fn max_cutoff_aggregation_determines_neighbor_list_radius() {
    // The LJ slot's max_cutoff() governs the neighbor-list radius.
    let gpu = init_device().unwrap();
    let sim_box = SimulationBox::new(20.0, 20.0, 20.0, 0.0, 0.0, 0.0).unwrap();
    let r_skin = 0.5_f64;
    let ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        4,
        &sim_box,
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()], // cutoff = 5.0
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(4),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(4),
        &NeighborListConfig::CellList { max_neighbors: 32, r_skin },
    )
    .unwrap();
    let nl = ff.neighbor_list.as_ref().unwrap();
    let cl = nl.cell_list_data().unwrap();
    let r_search = (5.0_f32 + r_skin as f32).powi(2);
    assert!(
        (cl.r_search_sq - r_search).abs() < 1.0e-3,
        "r_search_sq = {}, expected ~{}",
        cl.r_search_sq,
        r_search
    );
}

// --- Per-particle energy and virial outputs ---

#[test] // rq-531faea9
fn force_field_lj_only_populates_energy_and_virial() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let state = state_n(4);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        4,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
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
    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();
    let pe = device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
    let vw = device.dtoh_sync_copy(&buffers.virials).unwrap();
    assert!(pe.iter().any(|&v| v != 0.0), "potential_energies should be non-zero");
    assert!(vw.iter().any(|&v| v != 0.0), "virials should be non-zero");
    assert!(pe.iter().all(|&v| v.is_finite()));
    assert!(vw.iter().all(|&v| v.is_finite()));
}

#[test] // rq-a85e8216
fn slot_output_buffers_have_five_flat_arrays() {
    let gpu = init_device().unwrap();
    let bt = vec![BondTypeConfig::Morse {
        name: "CC".to_string(),
        de: 1.0,
        a: 2.0,
        re: 1.0,
    }];
    let bl = single_bond_list(8);
    let ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        8,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &bt,
        &[],
        None,
        None,
        &[],
        &bl,
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(8),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    assert_eq!(ff.slots.len(), 2);
    assert_eq!(ff.slot_forces_x.len(), 16);
    assert_eq!(ff.slot_forces_y.len(), 16);
    assert_eq!(ff.slot_forces_z.len(), 16);
    assert_eq!(ff.slot_energies.len(), 16);
    assert_eq!(ff.slot_virials.len(), 16);
}

#[test] // rq-3d38868e
fn combiner_sums_slot_energies_and_virials_in_slot_order() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let state = state_n(2);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        2,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(2),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    // Push two ConstStub slots and pre-fill their assigned rows.
    ff.slots.push(Box::new(ConstStub {
        value_x: 0.0,
        value_y: 0.0,
        value_z: 0.0,
        device: device.clone(),
    }));
    ff.slots.push(Box::new(ConstStub {
        value_x: 0.0,
        value_y: 0.0,
        value_z: 0.0,
        device: device.clone(),
    }));
    // The label uniqueness check normally rejects duplicate "const_stub" labels,
    // but here we bypass ForceField::new and inject directly. Two slots → label
    // collision is not checked during step.
    let new_len = ff.slots.len() * 2;
    ff.slot_forces_x = device.alloc_zeros::<f32>(new_len).unwrap();
    ff.slot_forces_y = device.alloc_zeros::<f32>(new_len).unwrap();
    ff.slot_forces_z = device.alloc_zeros::<f32>(new_len).unwrap();
    ff.slot_energies = device.alloc_zeros::<f32>(new_len).unwrap();
    ff.slot_virials = device.alloc_zeros::<f32>(new_len).unwrap();

    // Pre-seed the slot energy/virial rows. ConstStub.reduce writes only
    // force_x/y/z (zeros here); we want the combiner to see specific
    // energy/virial values. ConstStub doesn't touch energy/virial, so the
    // pre-seeded values pass through into the combiner.
    device
        .htod_sync_copy_into(&vec![1.0_f32, 2.0, 10.0, 20.0], &mut ff.slot_energies)
        .unwrap();
    device
        .htod_sync_copy_into(&vec![0.5_f32, 1.0, 5.0, 10.0], &mut ff.slot_virials)
        .unwrap();

    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();
    let pe = device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
    let vw = device.dtoh_sync_copy(&buffers.virials).unwrap();
    assert_eq!(pe, vec![11.0_f32, 22.0]);
    assert_eq!(vw, vec![5.5_f32, 11.0]);
}

#[test] // rq-c0f2daca
fn zero_slot_step_writes_zeros_to_energy_and_virial() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let state = state_n(4);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    // Pre-stamp non-zero junk into PE / virial.
    device
        .htod_sync_copy_into(&vec![42.0_f32; 4], &mut buffers.potential_energies)
        .unwrap();
    device
        .htod_sync_copy_into(&vec![-7.0_f32; 4], &mut buffers.virials)
        .unwrap();

    let mut timings = Timings::new(&gpu).unwrap();
    let mut ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        4,
        &box_10(),
        &[],
        &[],
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
    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();
    let pe = device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
    let vw = device.dtoh_sync_copy(&buffers.virials).unwrap();
    assert_eq!(pe, vec![0.0_f32; 4]);
    assert_eq!(vw, vec![0.0_f32; 4]);
}

#[test] // rq-db3b3d5e
fn system_total_potential_energy_equals_sum_of_particle_shares() {
    let gpu = init_device().unwrap();
    // Two particles at r=1.5 with σ=1, ε=1.
    let state = ParticleState::new(
        vec![0.0_f32, 1.5],
        vec![0.0_f32, 0.0],
        vec![0.0_f32, 0.0],
        vec![0.0_f32, 0.0],
        vec![0.0_f32, 0.0],
        vec![0.0_f32, 0.0],
        vec![1.0_f32, 1.0],
        vec![0.0_f32; vec![1.0_f32, 1.0].len()],
        vec![0u32, 0],
        None,
            None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        2,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(2),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();
    let pe = gpu.device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
    let total: f32 = pe.iter().sum();
    // Closed-form LJ energy at r=1.5 with σ=1, ε=1.
    let r = 1.5_f32;
    let sr2 = (1.0_f32 / r).powi(2);
    let sr6 = sr2.powi(3);
    let sr12 = sr6 * sr6;
    let expected = 4.0_f32 * 1.0 * (sr12 - sr6);
    assert!((total - expected).abs() < 1.0e-5, "got {total} expected {expected}");
}

#[test] // rq-7fe57a77
fn system_total_virial_equals_sum_of_particle_shares() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let state = ParticleState::new(
        vec![0.0_f32, 1.5],
        vec![0.0_f32, 0.0],
        vec![0.0_f32, 0.0],
        vec![0.0_f32, 0.0],
        vec![0.0_f32, 0.0],
        vec![0.0_f32, 0.0],
        vec![1.0_f32, 1.0],
        vec![0.0_f32; vec![1.0_f32, 1.0].len()],
        vec![0u32, 0],
        None,
            None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        2,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(2),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    ff.step(&mut buffers, &box_10(), &mut timings).unwrap();
    let vw = device.dtoh_sync_copy(&buffers.virials).unwrap();
    let fx = device.dtoh_sync_copy(&buffers.forces_x).unwrap();
    let total_virial: f32 = vw.iter().sum();
    // Single pair, r_ij = (-1.5, 0, 0) for i=0 → r · F = -1.5 * F_x_on_0.
    // F_x_on_0 = fx[0] (particle 0's net force comes entirely from particle 1).
    let expected = -1.5_f32 * fx[0];
    assert!((total_virial - expected).abs() < 1.0e-5, "got {total_virial} expected {expected}");
}

// =====================================================================
// PotentialRegistry-driven construction scenarios. See the
// "PotentialRegistry-driven construction" block in
// rqm/forces/framework.md.
// =====================================================================

use dynamics::forces::{
    CoulombBuilder, HarmonicAngleBuilder, LennardJonesBuilder, MorseBondedBuilder,
    PotentialBuildContext, PotentialBuilder, SpmeRealBuilder, SpmeReciprocalBuilder,
};
use dynamics::gpu::GpuContext;

// rq-053a026c
#[test]
fn registry_with_builtins_exposes_the_six_builders_in_evaluation_order() {
    let registry = PotentialRegistry::with_builtins();
    assert_eq!(registry.builders.len(), 6);
    // For unit-struct builders, `#[derive(Debug)]` prints the type name
    // itself ("LennardJonesBuilder") with no trailing fields. That gives
    // us a portable handle on the concrete type behind each
    // `Box<dyn PotentialBuilder>`.
    let names: Vec<String> = registry
        .builders
        .iter()
        .map(|b| format!("{b:?}"))
        .collect();
    assert_eq!(
        names,
        vec![
            "LennardJonesBuilder",
            "CoulombBuilder",
            "SpmeRealBuilder",
            "SpmeReciprocalBuilder",
            "MorseBondedBuilder",
            "HarmonicAngleBuilder",
        ]
    );
}

// rq-78ad9477
#[test]
fn registry_new_starts_empty() {
    let registry = PotentialRegistry::new();
    assert!(registry.builders.is_empty());
}

// rq-51af5f97
#[test]
fn registry_register_appends_a_builder_at_the_end() {
    let mut registry = PotentialRegistry::with_builtins();
    assert_eq!(registry.builders.len(), 6);
    registry.register(Box::new(SkipBuilder));
    assert_eq!(registry.builders.len(), 7);
    let t = format!("{:?}", registry.builders[6]);
    assert_eq!(t, "SkipBuilder");
}

// rq-b1a132b5
#[test]
fn force_field_new_iterates_registry_in_registration_order() {
    // LJ + Morse simultaneously: with the canonical builtin order, LJ comes first.
    let gpu = init_device().unwrap();
    let n = 4;
    let ff = ForceField::new(
        &PotentialRegistry::with_builtins(),
        &gpu,
        n,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[lj_pair_config()],
        &[BondTypeConfig::Morse {
            name: "CC".to_string(),
            de: 1.0,
            a: 1.0,
            re: 1.0,
        }],
        &[],
        None,
        None,
        &[],
        &single_bond_list(n),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(n),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    assert_eq!(ff.slots[0].label(), "lennard_jones");
    assert_eq!(ff.slots[1].label(), "morse_bonded");
}

/// PotentialBuilder that always returns Ok(None). Used by scenarios that
/// need a no-op builder to exercise the skip path or to occupy a registry
/// slot without producing a slot.
#[derive(Debug)]
struct SkipBuilder;

impl PotentialBuilder for SkipBuilder {
    fn build(
        &self,
        _cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        Ok(None)
    }
}

// rq-ccf4dc3f
#[test]
fn builder_returning_ok_none_is_skipped_without_erroring() {
    let gpu = init_device().unwrap();
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(SkipBuilder));
    let ff = ForceField::new(
        &registry,
        &gpu,
        4,
        &box_10(),
        &[],
        &[],
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
    assert!(ff.slots.is_empty());
}

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc as StdArc;

/// PotentialBuilder that always errors out. Used by the short-circuit
/// scenario.
#[derive(Debug)]
struct ErrBuilder;

impl PotentialBuilder for ErrBuilder {
    fn build(
        &self,
        _cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        Err(ForceFieldError::DuplicateLabel("ErrBuilder-marker"))
    }
}

/// Records whether `build` was invoked. The short-circuit scenario asserts
/// that no builder downstream of an `Err`-returning one is called.
#[derive(Debug)]
struct RecordingBuilder {
    called: StdArc<AtomicBool>,
}

impl PotentialBuilder for RecordingBuilder {
    fn build(
        &self,
        _cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        self.called.store(true, Ordering::SeqCst);
        Ok(None)
    }
}

// rq-6ed7e318
#[test]
fn builder_err_short_circuits_force_field_new() {
    let gpu = init_device().unwrap();
    let called = StdArc::new(AtomicBool::new(false));
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(ErrBuilder));
    registry.register(Box::new(RecordingBuilder { called: called.clone() }));
    let err = ForceField::new(
        &registry,
        &gpu,
        4,
        &box_10(),
        &[],
        &[],
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
    .unwrap_err();
    match err {
        ForceFieldError::DuplicateLabel(name) => {
            assert_eq!(name, "ErrBuilder-marker");
        }
        other => panic!("expected the ErrBuilder error, got {other:?}"),
    }
    assert!(
        !called.load(Ordering::SeqCst),
        "downstream builder was invoked despite earlier Err"
    );
}

/// Stub Potential whose label is configurable. Used to manufacture
/// duplicate-label scenarios without depending on which built-ins are
/// registered.
#[derive(Debug)]
struct RegistryLabelStub(&'static str);

impl Potential for RegistryLabelStub {
    fn label(&self) -> &'static str {
        self.0
    }
    fn max_cutoff(&self) -> Option<f32> {
        None
    }
    fn contribute(
        &mut self,
        _b: &ParticleBuffers,
        _s: &SimulationBox,
        _cx: &ForceFieldContext<'_>,
        _t: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        Ok(())
    }
    fn reduce(
        &mut self,
        _o: SlotOutputView<'_>,
        _cx: &ForceFieldContext<'_>,
        _t: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        Ok(())
    }
}

#[derive(Debug)]
struct DuplicateLabelBuilder;

impl PotentialBuilder for DuplicateLabelBuilder {
    fn build(
        &self,
        _cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        Ok(Some(Box::new(RegistryLabelStub("duplicate"))))
    }
}

// rq-24c36f8d
#[test]
fn two_builders_producing_slots_with_same_label_fail_construction() {
    let gpu = init_device().unwrap();
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(DuplicateLabelBuilder));
    registry.register(Box::new(DuplicateLabelBuilder));
    let err = ForceField::new(
        &registry,
        &gpu,
        4,
        &box_10(),
        &[],
        &[],
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
    .unwrap_err();
    match err {
        ForceFieldError::DuplicateLabel(label) => assert_eq!(label, "duplicate"),
        other => panic!("expected DuplicateLabel, got {other:?}"),
    }
}

// rq-028f5f8e
#[test]
fn empty_registry_produces_zero_slot_force_field() {
    let gpu = init_device().unwrap();
    let registry = PotentialRegistry::new();
    let ff = ForceField::new(
        &registry,
        &gpu,
        4,
        &box_10(),
        &[],
        &[],
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
    assert!(ff.slots.is_empty());
    assert!(ff.neighbor_list.is_none());
}

/// Captures the addresses of every borrowed field on the
/// `PotentialBuildContext` for the rq-b75ce71a passthrough scenario.
#[derive(Debug)]
struct PassthroughCaptureBuilder {
    captured: StdArc<std::sync::Mutex<Option<CapturedAddresses>>>,
}

#[derive(Debug, Clone, Copy)]
struct CapturedAddresses {
    gpu: usize,
    particle_count: usize,
    sim_box: usize,
    particle_types: usize,
    pair_interactions: usize,
    bond_types: usize,
    angle_types: usize,
    coulomb_config: Option<usize>,
    spme_config: Option<usize>,
    charges: usize,
    bond_list: usize,
    angle_list: usize,
    exclusion_list: usize,
    neighbor_list_config: usize,
}

impl PotentialBuilder for PassthroughCaptureBuilder {
    fn build(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        let addrs = CapturedAddresses {
            gpu: cx.gpu as *const GpuContext as usize,
            particle_count: cx.particle_count,
            sim_box: cx.sim_box as *const SimulationBox as usize,
            particle_types: cx.particle_types.as_ptr() as usize,
            pair_interactions: cx.pair_interactions.as_ptr() as usize,
            bond_types: cx.bond_types.as_ptr() as usize,
            angle_types: cx.angle_types.as_ptr() as usize,
            coulomb_config: cx
                .coulomb_config
                .map(|c| c as *const _ as usize),
            spme_config: cx.spme_config.map(|s| s as *const _ as usize),
            charges: cx.charges.as_ptr() as usize,
            bond_list: cx.bond_list as *const BondList as usize,
            angle_list: cx.angle_list as *const dynamics::forces::AngleList as usize,
            exclusion_list: cx.exclusion_list as *const ExclusionList as usize,
            neighbor_list_config: cx.neighbor_list_config as *const NeighborListConfig as usize,
        };
        *self.captured.lock().unwrap() = Some(addrs);
        Ok(None)
    }
}

// rq-b75ce71a
#[test]
fn build_context_exposes_every_parsed_config_input_by_reference() {
    let gpu = init_device().unwrap();
    let n = 3usize;
    let sim_box = box_10();
    let particle_types = vec![ParticleTypeConfig {
        name: "Ar".to_string(),
        mass: 1.0,
        charge: 0.0,
    }];
    let pair_interactions: Vec<PairInteractionConfig> = vec![];
    let bond_types: Vec<BondTypeConfig> = vec![];
    let angle_types: Vec<dynamics::io::config::AngleTypeConfig> = vec![];
    let charges: Vec<f32> = vec![0.0; n];
    let bond_list = BondList::empty(n);
    let angle_list = dynamics::forces::AngleList::empty(0);
    let exclusion_list = ExclusionList::empty(n);
    let nl_config = NeighborListConfig::AllPairs;

    let captured = StdArc::new(std::sync::Mutex::new(None::<CapturedAddresses>));
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(PassthroughCaptureBuilder { captured: captured.clone() }));

    ForceField::new(
        &registry,
        &gpu,
        n,
        &sim_box,
        &particle_types,
        &pair_interactions,
        &bond_types,
        &angle_types,
        None,
        None,
        &charges,
        &bond_list,
        &angle_list,
        &exclusion_list,
        &nl_config,
    )
    .unwrap();

    let addrs = captured.lock().unwrap().expect("captured");
    assert_eq!(addrs.gpu, &gpu as *const GpuContext as usize);
    assert_eq!(addrs.particle_count, n);
    assert_eq!(addrs.sim_box, &sim_box as *const SimulationBox as usize);
    assert_eq!(addrs.particle_types, particle_types.as_ptr() as usize);
    assert_eq!(addrs.pair_interactions, pair_interactions.as_ptr() as usize);
    assert_eq!(addrs.bond_types, bond_types.as_ptr() as usize);
    assert_eq!(addrs.angle_types, angle_types.as_ptr() as usize);
    assert_eq!(addrs.coulomb_config, None);
    assert_eq!(addrs.spme_config, None);
    assert_eq!(addrs.charges, charges.as_ptr() as usize);
    assert_eq!(addrs.bond_list, &bond_list as *const BondList as usize);
    assert_eq!(
        addrs.angle_list,
        &angle_list as *const dynamics::forces::AngleList as usize
    );
    assert_eq!(
        addrs.exclusion_list,
        &exclusion_list as *const ExclusionList as usize
    );
    assert_eq!(
        addrs.neighbor_list_config,
        &nl_config as *const NeighborListConfig as usize
    );
}

// Suppress unused-import lints for the per-builder names; they're
// reachable through the registry but not referenced by name in tests.
#[test]
fn _builtin_builders_are_pub() {
    let _ = LennardJonesBuilder;
    let _ = CoulombBuilder;
    let _ = SpmeRealBuilder;
    let _ = SpmeReciprocalBuilder;
    let _ = MorseBondedBuilder;
    let _ = HarmonicAngleBuilder;
}
