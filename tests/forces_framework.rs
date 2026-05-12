use dynamics::forces::{
    Bond, BondList, ExclusionList, ForceField, PotentialSlot,
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
    assert!(matches!(ff.slots[0], PotentialSlot::LennardJones(_)));
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
    assert!(matches!(ff.slots[0], PotentialSlot::LennardJones(_)));
    assert!(matches!(ff.slots[1], PotentialSlot::MorseBonded(_)));
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
    // Atoms at 0 and 1.5 separation; with sigma=1, cutoff=5, the LJ force is
    // non-zero. forces_x[0] and forces_x[1] should be opposite and non-zero.
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

    // LJ only run.
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

    // Morse only — use a tiny LJ cutoff so the LJ contribution is zero.
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

    // Combined.
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
