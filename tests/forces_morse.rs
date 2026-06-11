use dynamics::forces::{AggregateLevel, BondList, ExclusionList, ForceField, MorseBondedState, PotentialRegistry};
use dynamics::gpu::{
    ParticleBuffers, init_device, morse_bond_force, reduce_bond_forces,
};
use dynamics::io::config::{
    BondTypeConfig, NeighborListConfig, PairInteractionConfig, PairPotentialParams,
    ParticleTypeConfig,
};
use dynamics::pbc::SimulationBox;
use dynamics::state::ParticleState;
use dynamics::timings::Timings;

fn two_particle_state(p0: [f32; 3], p1: [f32; 3]) -> ParticleState {
    ParticleState::new(
        vec![p0[0], p1[0]],
        vec![p0[1], p1[1]],
        vec![p0[2], p1[2]],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0_f32; vec![1.0, 1.0].len()],
        vec![0u32; 2],
        None,
            None,
    )
    .unwrap()
}

fn box_10() -> SimulationBox {
    SimulationBox::new(10.0, 10.0, 10.0, 0.0, 0.0, 0.0).unwrap()
}

fn morse_type(de: f64, a: f64, re: f64) -> BondTypeConfig {
    BondTypeConfig::Morse {
        name: "CC".to_string(),
        de,
        a,
        re,
    }
}

fn single_bond_list(n: usize) -> BondList {
    use dynamics::forces::Bond;
    let bonds = vec![Bond {
        atom_i: 0,
        atom_j: 1,
        bond_type_index: 0,
    }];
    let mut atom_bond_offsets = vec![0u32; n + 1];
    atom_bond_offsets[1] = 1;
    atom_bond_offsets[2] = 2;
    for i in 3..=n {
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

// rq-679282f5
#[test]
fn init_device_loads_morse_module() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    assert!(device.get_func("morse", "morse_bond_force").is_some());
    assert!(device.get_func("morse", "reduce_bond_forces").is_some());
    let _ = gpu.kernels.morse.morse_bond_force.clone();
    let _ = gpu.kernels.morse.reduce_bond_forces.clone();
}

// rq-9f2de58c
#[test]
fn construct_morse_bonded_state() {
    let gpu = init_device().unwrap();
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let state = MorseBondedState::new(&gpu, &bl, &bt).unwrap();
    assert_eq!(state.bond_count, 1);
    assert_eq!(state.particle_count, 2);
}

// rq-2e4e70b4
#[test]
fn equilibrium_distance_produces_zero_force() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let state = two_particle_state([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let mut mb = MorseBondedState::new(&gpu, &bl, &bt).unwrap();
    morse_bond_force(
        &buffers,
        &mb.bonds,
        &mb.bond_de,
        &mb.bond_a,
        &mb.bond_re,
        &box_10(),
        &mut mb.bond_pair_x,
        &mut mb.bond_pair_y,
        &mut mb.bond_pair_z,
        &mut mb.bond_pair_energy,
        &mut mb.bond_pair_virial,
        1,
    )
    .unwrap();
    let bx = device.dtoh_sync_copy(&mb.bond_pair_x).unwrap();
    let by = device.dtoh_sync_copy(&mb.bond_pair_y).unwrap();
    let bz = device.dtoh_sync_copy(&mb.bond_pair_z).unwrap();
    assert!(bx[0].abs() < 1.0e-6, "bx[0] = {}", bx[0]);
    assert!(by[0].abs() < 1.0e-6);
    assert!(bz[0].abs() < 1.0e-6);
    assert!(bx[1].abs() < 1.0e-6);
}

// rq-f79657d2
#[test]
fn compressed_bond_repulsive() {
    let gpu = init_device().unwrap();
    // r = 0.5, re = 1.0 → r < re → repulsive: atom 0 is pushed in -x.
    let state = two_particle_state([0.0, 0.0, 0.0], [0.5, 0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let mut mb = MorseBondedState::new(&gpu, &bl, &bt).unwrap();
    morse_bond_force(
        &buffers,
        &mb.bonds,
        &mb.bond_de,
        &mb.bond_a,
        &mb.bond_re,
        &box_10(),
        &mut mb.bond_pair_x,
        &mut mb.bond_pair_y,
        &mut mb.bond_pair_z,
        &mut mb.bond_pair_energy,
        &mut mb.bond_pair_virial,
        1,
    )
    .unwrap();
    let bx = gpu.device.dtoh_sync_copy(&mb.bond_pair_x).unwrap();
    // dx = r_i - r_j = -0.5 (atom 0 is left of atom 1). Compressed bond is
    // repulsive: atom 0 is pushed away from atom 1, i.e. in -x. So bx[0] < 0.
    assert!(bx[0] < 0.0, "bx[0] = {} should be negative", bx[0]);
    assert!((bx[0] + bx[1]).abs() < 1.0e-6, "Newton's third law: {} + {}", bx[0], bx[1]);
}

// rq-2cb90e10
#[test]
fn stretched_bond_attractive() {
    let gpu = init_device().unwrap();
    // r = 2.0, re = 1.0 → r > re → attractive: atom 0 pulled toward atom 1 (+x).
    let state = two_particle_state([0.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let mut mb = MorseBondedState::new(&gpu, &bl, &bt).unwrap();
    morse_bond_force(
        &buffers,
        &mb.bonds,
        &mb.bond_de,
        &mb.bond_a,
        &mb.bond_re,
        &box_10(),
        &mut mb.bond_pair_x,
        &mut mb.bond_pair_y,
        &mut mb.bond_pair_z,
        &mut mb.bond_pair_energy,
        &mut mb.bond_pair_virial,
        1,
    )
    .unwrap();
    let bx = gpu.device.dtoh_sync_copy(&mb.bond_pair_x).unwrap();
    // Stretched bond → attractive, fmag negative; F on atom 0 = fmag * (-2.0)
    // → positive (toward atom 1 in +x).
    assert!(bx[0] > 0.0, "bx[0] = {}", bx[0]);
    assert!((bx[0] + bx[1]).abs() < 1.0e-6);
}

// rq-d61fa682
#[test]
fn force_magnitude_matches_closed_form() {
    let gpu = init_device().unwrap();
    let r = 1.2_f32;
    let state = two_particle_state([0.0, 0.0, 0.0], [r, 0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let bl = single_bond_list(2);
    let de = 1.0_f64;
    let a = 2.0_f64;
    let re = 1.0_f64;
    let bt = vec![morse_type(de, a, re)];
    let mut mb = MorseBondedState::new(&gpu, &bl, &bt).unwrap();
    morse_bond_force(
        &buffers,
        &mb.bonds,
        &mb.bond_de,
        &mb.bond_a,
        &mb.bond_re,
        &box_10(),
        &mut mb.bond_pair_x,
        &mut mb.bond_pair_y,
        &mut mb.bond_pair_z,
        &mut mb.bond_pair_energy,
        &mut mb.bond_pair_virial,
        1,
    )
    .unwrap();
    let bx = gpu.device.dtoh_sync_copy(&mb.bond_pair_x).unwrap();
    let dr = (r as f64) - re;
    let e = (-a * dr).exp();
    let expected_magnitude = (2.0 * de * a * (1.0 - e) * e) as f32;
    assert!((bx[0].abs() - expected_magnitude.abs()).abs() < 1.0e-5);
}

// rq-4811af60
#[test]
fn r_zero_produces_zero_force() {
    let gpu = init_device().unwrap();
    let state = two_particle_state([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let mut mb = MorseBondedState::new(&gpu, &bl, &bt).unwrap();
    morse_bond_force(
        &buffers,
        &mb.bonds,
        &mb.bond_de,
        &mb.bond_a,
        &mb.bond_re,
        &box_10(),
        &mut mb.bond_pair_x,
        &mut mb.bond_pair_y,
        &mut mb.bond_pair_z,
        &mut mb.bond_pair_energy,
        &mut mb.bond_pair_virial,
        1,
    )
    .unwrap();
    let bx = gpu.device.dtoh_sync_copy(&mb.bond_pair_x).unwrap();
    for v in &bx {
        assert!(v.is_finite() && *v == 0.0);
    }
}

// rq-62e2469f
#[test]
fn morse_bond_force_zero_bonds_is_noop() {
    let gpu = init_device().unwrap();
    let state = two_particle_state([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let bl = BondList::empty(2);
    let bt: Vec<BondTypeConfig> = Vec::new();
    let mut mb = MorseBondedState::new(&gpu, &bl, &bt).unwrap();
    morse_bond_force(
        &buffers,
        &mb.bonds,
        &mb.bond_de,
        &mb.bond_a,
        &mb.bond_re,
        &box_10(),
        &mut mb.bond_pair_x,
        &mut mb.bond_pair_y,
        &mut mb.bond_pair_z,
        &mut mb.bond_pair_energy,
        &mut mb.bond_pair_virial,
        0,
    )
    .unwrap();
}

// rq-1ce4ce5a
#[test]
fn atom_with_two_bonds_sums_contributions() {
    use dynamics::forces::Bond;

    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    // 3 atoms in a row at -1, 0, +1. Bonds 0-1 and 1-2; atom 1 receives forces from both.
    let state = ParticleState::new(
        vec![-1.0_f32, 0.0_f32, 1.0_f32],
        vec![0.0_f32; 3],
        vec![0.0_f32; 3],
        vec![0.0_f32; 3],
        vec![0.0_f32; 3],
        vec![0.0_f32; 3],
        vec![1.0_f32; 3],
        vec![0.0_f32; 3],
        vec![0u32; 3],
        None,
            None,
    )
    .unwrap();
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    // Bonds: (0,1) at slot pair [0,1]; (1,2) at slot pair [2,3].
    let bonds = vec![
        Bond { atom_i: 0, atom_j: 1, bond_type_index: 0 },
        Bond { atom_i: 1, atom_j: 2, bond_type_index: 0 },
    ];
    // Per-atom lookups: atom 0 → [0], atom 1 → [1, 2], atom 2 → [3].
    let atom_bond_offsets = vec![0u32, 1, 3, 4];
    let atom_bond_indices = vec![0u32, 1, 2, 3];
    let bl = BondList {
        bonds,
        atom_bond_offsets,
        atom_bond_indices,
        particle_count: 3,
    };
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let mut mb = MorseBondedState::new(&gpu, &bl, &bt).unwrap();
    morse_bond_force(
        &buffers,
        &mb.bonds,
        &mb.bond_de,
        &mb.bond_a,
        &mb.bond_re,
        &box_10(),
        &mut mb.bond_pair_x,
        &mut mb.bond_pair_y,
        &mut mb.bond_pair_z,
        &mut mb.bond_pair_energy,
        &mut mb.bond_pair_virial,
        2,
    )
    .unwrap();
    let mut acc_x = device.alloc_zeros::<f32>(3).unwrap();
    let mut acc_y = device.alloc_zeros::<f32>(3).unwrap();
    let mut acc_z = device.alloc_zeros::<f32>(3).unwrap();
    let mut acc_e = device.alloc_zeros::<f32>(3).unwrap();
    let mut acc_w = device.alloc_zeros::<f32>(3).unwrap();
    reduce_bond_forces(&gpu.kernels,
        &mb.bond_pair_x,
        &mb.bond_pair_y,
        &mb.bond_pair_z,
        &mb.bond_pair_energy,
        &mb.bond_pair_virial,
        &mb.atom_bond_offsets,
        &mb.atom_bond_indices,
        &mut acc_x.slice_mut(..),
        &mut acc_y.slice_mut(..),
        &mut acc_z.slice_mut(..),
        &mut acc_e.slice_mut(..),
        &mut acc_w.slice_mut(..),
        3,
    )
    .unwrap();
    let ax = device.dtoh_sync_copy(&acc_x).unwrap();
    // Each bond is at equilibrium (r=1.0=re), so force magnitudes are 0;
    // accumulator should be zero for all atoms.
    for v in &ax {
        assert!(v.abs() < 1.0e-6);
    }
}

// rq-1ca90a29
#[test]
fn atom_with_no_bonds_gets_zero_accumulator() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let state = ParticleState::new(
        vec![0.0_f32; 4],
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
    let _ = state;
    // Atom 3 has no bonds. Bond (0,1) only.
    let bl = single_bond_list(4);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let mb = MorseBondedState::new(&gpu, &bl, &bt).unwrap();
    // Reduction without contribution kernel populating bond_pair_* leaves it
    // zeroed by allocation; accumulator should remain zero.
    let mut acc_x = device.alloc_zeros::<f32>(4).unwrap();
    let mut acc_y = device.alloc_zeros::<f32>(4).unwrap();
    let mut acc_z = device.alloc_zeros::<f32>(4).unwrap();
    let mut acc_e = device.alloc_zeros::<f32>(4).unwrap();
    let mut acc_w = device.alloc_zeros::<f32>(4).unwrap();
    reduce_bond_forces(&gpu.kernels,
        &mb.bond_pair_x,
        &mb.bond_pair_y,
        &mb.bond_pair_z,
        &mb.bond_pair_energy,
        &mb.bond_pair_virial,
        &mb.atom_bond_offsets,
        &mb.atom_bond_indices,
        &mut acc_x.slice_mut(..),
        &mut acc_y.slice_mut(..),
        &mut acc_z.slice_mut(..),
        &mut acc_e.slice_mut(..),
        &mut acc_w.slice_mut(..),
        4,
    )
    .unwrap();
    let ax = device.dtoh_sync_copy(&acc_x).unwrap();
    assert_eq!(ax[3], 0.0);
}

// rq-966e43ed (reduce_bond_forces on zero particles is a no-op — IDs from spec)
#[test]
fn reduce_bond_forces_zero_particles_noop() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let bl = BondList::empty(0);
    let bt: Vec<BondTypeConfig> = Vec::new();
    let mb = MorseBondedState::new(&gpu, &bl, &bt).unwrap();
    let mut acc_x = device.alloc_zeros::<f32>(0).unwrap();
    let mut acc_y = device.alloc_zeros::<f32>(0).unwrap();
    let mut acc_z = device.alloc_zeros::<f32>(0).unwrap();
    let mut acc_e = device.alloc_zeros::<f32>(0).unwrap();
    let mut acc_w = device.alloc_zeros::<f32>(0).unwrap();
    reduce_bond_forces(&gpu.kernels,
        &mb.bond_pair_x,
        &mb.bond_pair_y,
        &mb.bond_pair_z,
        &mb.bond_pair_energy,
        &mb.bond_pair_virial,
        &mb.atom_bond_offsets,
        &mb.atom_bond_indices,
        &mut acc_x.slice_mut(..),
        &mut acc_y.slice_mut(..),
        &mut acc_z.slice_mut(..),
        &mut acc_e.slice_mut(..),
        &mut acc_w.slice_mut(..),
        0,
    )
    .unwrap();
}

// End-to-end: through the framework.
// rq-c7af1f28
#[test]
fn diatomic_equilibrium_produces_zero_net_force() {
    let gpu = init_device().unwrap();
    let state = two_particle_state([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    // LJ params with cutoff < bond length so LJ contributes nothing.
    let pair = PairInteractionConfig {
        between: ("Ar".to_string(), "Ar".to_string()),
        cutoff: 0.5,
        r_switch: 0.5,
        potential: PairPotentialParams::LennardJones { sigma: 1.0, epsilon: 1.0 },
    };
    let mut ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        2,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[pair],
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
    ff.step(&mut buffers, &box_10(), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let mut downloaded = state.clone();
    downloaded.download_from(&buffers).unwrap();
    assert!(downloaded.forces_x[0].abs() < 1.0e-6);
    assert!(downloaded.forces_x[1].abs() < 1.0e-6);
}

// rq-6d06e36e
#[test]
fn newtons_third_law_holds_for_combined_force() {
    let gpu = init_device().unwrap();
    // Atoms inside LJ cutoff and within bond range.
    let state = two_particle_state([0.0, 0.0, 0.0], [1.2, 0.0, 0.0]);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let pair = PairInteractionConfig {
        between: ("Ar".to_string(), "Ar".to_string()),
        cutoff: 5.0,
        r_switch: 5.0,
        potential: PairPotentialParams::LennardJones { sigma: 1.0, epsilon: 1.0 },
    };
    let mut ff = ForceField::new(&PotentialRegistry::with_builtins(), &gpu,
        2,
        &box_10(),
        &[ParticleTypeConfig { name: "Ar".to_string(), mass: 1.0, charge: 0.0 }],
        &[pair],
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
    ff.step(&mut buffers, &box_10(), &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let mut downloaded = state.clone();
    downloaded.download_from(&buffers).unwrap();
    let sum_x = downloaded.forces_x[0] + downloaded.forces_x[1];
    let sum_y = downloaded.forces_y[0] + downloaded.forces_y[1];
    let sum_z = downloaded.forces_z[0] + downloaded.forces_z[1];
    assert!(sum_x.abs() < 1.0e-5, "{sum_x}");
    assert!(sum_y.abs() < 1.0e-5);
    assert!(sum_z.abs() < 1.0e-5);
    // Verify the ForceField uses the MorseBonded slot when bonds exist.
    assert_eq!(ff.slots.len(), 2);
    assert_eq!(ff.slots[1].label(), "morse_bonded");
}

// --- Energy and virial outputs ---

#[test] // rq-7ba4f321
fn stretched_bond_energy_matches_closed_form() {
    let gpu = init_device().unwrap();
    let r = 1.5_f32;
    let state = two_particle_state([0.0, 0.0, 0.0], [r, 0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let bl = single_bond_list(2);
    let de = 1.0_f64;
    let a = 2.0_f64;
    let re = 1.0_f64;
    let bt = vec![morse_type(de, a, re)];
    let mut mb = MorseBondedState::new(&gpu, &bl, &bt).unwrap();
    morse_bond_force(
        &buffers,
        &mb.bonds,
        &mb.bond_de,
        &mb.bond_a,
        &mb.bond_re,
        &box_10(),
        &mut mb.bond_pair_x,
        &mut mb.bond_pair_y,
        &mut mb.bond_pair_z,
        &mut mb.bond_pair_energy,
        &mut mb.bond_pair_virial,
        1,
    )
    .unwrap();
    let be = gpu.device.dtoh_sync_copy(&mb.bond_pair_energy).unwrap();
    let dr = (r as f64) - re;
    let e = (-a * dr).exp();
    let one_minus = 1.0 - e;
    let expected = (de * one_minus * one_minus) as f32;
    assert!((be[0] + be[1] - expected).abs() < 1.0e-5, "got {} expected {}", be[0] + be[1], expected);
}

#[test] // rq-ca49d49a
fn stretched_bond_virial_matches_r_dot_f() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let r = 1.5_f32;
    let state = two_particle_state([0.0, 0.0, 0.0], [r, 0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let mut mb = MorseBondedState::new(&gpu, &bl, &bt).unwrap();
    morse_bond_force(
        &buffers,
        &mb.bonds,
        &mb.bond_de,
        &mb.bond_a,
        &mb.bond_re,
        &box_10(),
        &mut mb.bond_pair_x,
        &mut mb.bond_pair_y,
        &mut mb.bond_pair_z,
        &mut mb.bond_pair_energy,
        &mut mb.bond_pair_virial,
        1,
    )
    .unwrap();
    let bx = device.dtoh_sync_copy(&mb.bond_pair_x).unwrap();
    let bv = device.dtoh_sync_copy(&mb.bond_pair_virial).unwrap();
    // dx = r_0 - r_1 = -1.5 (atom 0 at origin, atom 1 at +x).
    // F on atom 0 due to atom 1 = bx[0] (along the dx direction times fmag).
    // r_ij · F_ij = dx * F_x = (-1.5) * bx[0].
    let expected = -1.5_f32 * bx[0];
    let total = bv[0] + bv[1];
    assert!(
        (total - expected).abs() < 1.0e-5,
        "got {total} expected {expected}"
    );
}

#[test] // rq-fe9f2ebe
fn r_zero_produces_zero_energy_and_virial() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let state = two_particle_state([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let mut mb = MorseBondedState::new(&gpu, &bl, &bt).unwrap();
    morse_bond_force(
        &buffers,
        &mb.bonds,
        &mb.bond_de,
        &mb.bond_a,
        &mb.bond_re,
        &box_10(),
        &mut mb.bond_pair_x,
        &mut mb.bond_pair_y,
        &mut mb.bond_pair_z,
        &mut mb.bond_pair_energy,
        &mut mb.bond_pair_virial,
        1,
    )
    .unwrap();
    let be = device.dtoh_sync_copy(&mb.bond_pair_energy).unwrap();
    let bv = device.dtoh_sync_copy(&mb.bond_pair_virial).unwrap();
    for v in be.iter().chain(bv.iter()) {
        assert!(v.is_finite() && *v == 0.0);
    }
}

#[test] // rq-6897ffda
fn bond_reduction_sums_energy_and_virial_alongside_forces() {
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    let r = 1.5_f32;
    let state = two_particle_state([0.0, 0.0, 0.0], [r, 0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let mut mb = MorseBondedState::new(&gpu, &bl, &bt).unwrap();
    morse_bond_force(
        &buffers,
        &mb.bonds,
        &mb.bond_de,
        &mb.bond_a,
        &mb.bond_re,
        &box_10(),
        &mut mb.bond_pair_x,
        &mut mb.bond_pair_y,
        &mut mb.bond_pair_z,
        &mut mb.bond_pair_energy,
        &mut mb.bond_pair_virial,
        1,
    )
    .unwrap();
    let mut acc_x = device.alloc_zeros::<f32>(2).unwrap();
    let mut acc_y = device.alloc_zeros::<f32>(2).unwrap();
    let mut acc_z = device.alloc_zeros::<f32>(2).unwrap();
    let mut acc_e = device.alloc_zeros::<f32>(2).unwrap();
    let mut acc_w = device.alloc_zeros::<f32>(2).unwrap();
    reduce_bond_forces(&gpu.kernels,
        &mb.bond_pair_x,
        &mb.bond_pair_y,
        &mb.bond_pair_z,
        &mb.bond_pair_energy,
        &mb.bond_pair_virial,
        &mb.atom_bond_offsets,
        &mb.atom_bond_indices,
        &mut acc_x.slice_mut(..),
        &mut acc_y.slice_mut(..),
        &mut acc_z.slice_mut(..),
        &mut acc_e.slice_mut(..),
        &mut acc_w.slice_mut(..),
        2,
    )
    .unwrap();
    let acc_e_host = device.dtoh_sync_copy(&acc_e).unwrap();
    let acc_w_host = device.dtoh_sync_copy(&acc_w).unwrap();
    let be = device.dtoh_sync_copy(&mb.bond_pair_energy).unwrap();
    let bv = device.dtoh_sync_copy(&mb.bond_pair_virial).unwrap();
    // Each atom's share equals one half-bond entry.
    assert_eq!(acc_e_host[0], be[0]);
    assert_eq!(acc_e_host[1], be[1]);
    assert_eq!(acc_w_host[0], bv[0]);
    assert_eq!(acc_w_host[1], bv[1]);
}

// --- Reduction kernel direct passthrough ----------------------------------

fn run_reduce_with_buffers(
    gpu: &dynamics::gpu::GpuContext,
    bond_pair_x: &[f32],
    bond_pair_y: &[f32],
    bond_pair_z: &[f32],
    bond_pair_energy: &[f32],
    bond_pair_virial: &[f32],
    atom_bond_offsets: &[u32],
    atom_bond_indices: &[u32],
    particle_count: usize,
) -> Vec<f32> {
    use cudarc::driver::DeviceSlice;
    let device = gpu.device.clone();
    let bx = device.htod_sync_copy(bond_pair_x).unwrap();
    let by = device.htod_sync_copy(bond_pair_y).unwrap();
    let bz = device.htod_sync_copy(bond_pair_z).unwrap();
    let be = device.htod_sync_copy(bond_pair_energy).unwrap();
    let bv = device.htod_sync_copy(bond_pair_virial).unwrap();
    let offsets = device.htod_sync_copy(atom_bond_offsets).unwrap();
    let indices = if atom_bond_indices.is_empty() {
        device.alloc_zeros::<u32>(1).unwrap()
    } else {
        device.htod_sync_copy(atom_bond_indices).unwrap()
    };
    let mut acc_x = device.alloc_zeros::<f32>(particle_count.max(1)).unwrap();
    let mut acc_y = device.alloc_zeros::<f32>(particle_count.max(1)).unwrap();
    let mut acc_z = device.alloc_zeros::<f32>(particle_count.max(1)).unwrap();
    let mut acc_e = device.alloc_zeros::<f32>(particle_count.max(1)).unwrap();
    let mut acc_w = device.alloc_zeros::<f32>(particle_count.max(1)).unwrap();
    let up_x = acc_x.len();
    let up_y = acc_y.len();
    let up_z = acc_z.len();
    let up_e = acc_e.len();
    let up_w = acc_w.len();
    reduce_bond_forces(
        &gpu.kernels,
        &bx, &by, &bz, &be, &bv,
        &offsets, &indices,
        &mut acc_x.slice_mut(0..up_x),
        &mut acc_y.slice_mut(0..up_y),
        &mut acc_z.slice_mut(0..up_z),
        &mut acc_e.slice_mut(0..up_e),
        &mut acc_w.slice_mut(0..up_w),
        particle_count,
    )
    .unwrap();
    device.dtoh_sync_copy(&acc_x).unwrap()
}

// rq-2d4efead
#[test]
fn atom_with_one_bond_receives_the_bonds_force_directly() {
    let gpu = init_device().unwrap();
    let bond_pair_x = vec![2.0_f32, -2.0];
    let zeros = vec![0.0_f32; 2];
    let atom_bond_offsets = vec![0u32, 1, 2];
    let atom_bond_indices = vec![0u32, 1];
    let acc_x = run_reduce_with_buffers(
        &gpu,
        &bond_pair_x, &zeros, &zeros, &zeros, &zeros,
        &atom_bond_offsets, &atom_bond_indices,
        2,
    );
    assert_eq!(acc_x[0], 2.0_f32);
    assert_eq!(acc_x[1], -2.0_f32);
}

// rq-55f89976
#[test]
fn reduction_summation_order_is_sorted_bond_index() {
    let gpu = init_device().unwrap();
    // Atom 0 receives contributions from bonds 0 and 1; slots [0, 2].
    // The reduction walks atom_bond_indices left-to-right, so accumulator
    // = bond_pair_x[0] + bond_pair_x[2].
    let bond_pair_x = vec![1.5_f32, -7.0, 4.0, -0.5];
    let zeros = vec![0.0_f32; 4];
    // 3 particles. atom 0 → [0, 2] (the two slots that name it); atom 1 → [1]; atom 2 → [3].
    let atom_bond_offsets = vec![0u32, 2, 3, 4];
    let atom_bond_indices = vec![0u32, 2, 1, 3];
    let acc_x = run_reduce_with_buffers(
        &gpu,
        &bond_pair_x, &zeros, &zeros, &zeros, &zeros,
        &atom_bond_offsets, &atom_bond_indices,
        3,
    );
    assert_eq!(acc_x[0], 1.5_f32 + 4.0, "atom 0 must sum slots [0, 2] in left-to-right order");
    assert_eq!(acc_x[1], -7.0);
    assert_eq!(acc_x[2], -0.5);
}

// rq-556b7c13
#[test]
fn minimum_image_is_applied_to_bond_displacement() {
    use cudarc::driver::DeviceSlice;
    let gpu = init_device().unwrap();
    let sim_box = box_10(); // L=10
    // p0 = (-4.5, 0, 0), p1 = (4.5, 0, 0). Naive r_ij = -9.0, wraps to +1.0
    // (the periodic image). The bond is at r=1.0 = re for the type
    // re=1.0, so the morse force is exactly zero — confirming the kernel
    // used the wrapped displacement, not r=9.0 which would give a large
    // attractive force.
    let state = two_particle_state([-4.5, 0.0, 0.0], [4.5, 0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];
    let mut mb = MorseBondedState::new(&gpu, &bl, &bt).unwrap();
    morse_bond_force(
        &buffers,
        &mb.bonds,
        &mb.bond_de,
        &mb.bond_a,
        &mb.bond_re,
        &sim_box,
        &mut mb.bond_pair_x,
        &mut mb.bond_pair_y,
        &mut mb.bond_pair_z,
        &mut mb.bond_pair_energy,
        &mut mb.bond_pair_virial,
        1,
    )
    .unwrap();
    let bx = gpu.device.dtoh_sync_copy(&mb.bond_pair_x).unwrap();
    let energy = gpu.device.dtoh_sync_copy(&mb.bond_pair_energy).unwrap();
    let _ = bx.len();
    for v in &bx {
        assert!(v.abs() < 1.0e-5, "wrapped r = 1.0 (= re); force must be ~0, got {v}");
    }
    for e in &energy {
        assert!(e.abs() < 1.0e-5, "wrapped r = 1.0 (= re); energy must be ~0, got {e}");
    }
}

// rq-696caf8e
#[test]
fn two_independent_calls_produce_byte_identical_accumulators() {
    use cudarc::driver::DeviceSlice;
    let gpu = init_device().unwrap();
    let device = gpu.device.clone();
    // Stretched bond so the forces are non-zero and the comparison is
    // meaningful.
    let state = two_particle_state([0.0, 0.0, 0.0], [1.3, 0.0, 0.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let bl = single_bond_list(2);
    let bt = vec![morse_type(1.0, 2.0, 1.0)];

    let run = || -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut mb = MorseBondedState::new(&gpu, &bl, &bt).unwrap();
        morse_bond_force(
            &buffers,
            &mb.bonds,
            &mb.bond_de,
            &mb.bond_a,
            &mb.bond_re,
            &box_10(),
            &mut mb.bond_pair_x,
            &mut mb.bond_pair_y,
            &mut mb.bond_pair_z,
            &mut mb.bond_pair_energy,
            &mut mb.bond_pair_virial,
            1,
        )
        .unwrap();
        let mut acc_x = device.alloc_zeros::<f32>(2).unwrap();
        let mut acc_y = device.alloc_zeros::<f32>(2).unwrap();
        let mut acc_z = device.alloc_zeros::<f32>(2).unwrap();
        let mut acc_e = device.alloc_zeros::<f32>(2).unwrap();
        let mut acc_w = device.alloc_zeros::<f32>(2).unwrap();
        let up_x = acc_x.len();
        let up_y = acc_y.len();
        let up_z = acc_z.len();
        let up_e = acc_e.len();
        let up_w = acc_w.len();
        reduce_bond_forces(
            &gpu.kernels,
            &mb.bond_pair_x, &mb.bond_pair_y, &mb.bond_pair_z,
            &mb.bond_pair_energy, &mb.bond_pair_virial,
            &mb.atom_bond_offsets, &mb.atom_bond_indices,
            &mut acc_x.slice_mut(0..up_x),
            &mut acc_y.slice_mut(0..up_y),
            &mut acc_z.slice_mut(0..up_z),
            &mut acc_e.slice_mut(0..up_e),
            &mut acc_w.slice_mut(0..up_w),
            2,
        )
        .unwrap();
        (
            device.dtoh_sync_copy(&acc_x).unwrap(),
            device.dtoh_sync_copy(&acc_y).unwrap(),
            device.dtoh_sync_copy(&acc_z).unwrap(),
            device.dtoh_sync_copy(&acc_e).unwrap(),
            device.dtoh_sync_copy(&acc_w).unwrap(),
        )
    };
    let a = run();
    let b = run();
    assert_eq!(a.0, b.0, "acc_x not bit-exact");
    assert_eq!(a.1, b.1, "acc_y not bit-exact");
    assert_eq!(a.2, b.2, "acc_z not bit-exact");
    assert_eq!(a.3, b.3, "acc_e not bit-exact");
    assert_eq!(a.4, b.4, "acc_w not bit-exact");
    // The bond is stretched, so accumulators must be non-trivially non-zero.
    assert!(a.0[0].abs() > 0.0, "stretched bond must produce non-zero force");
}
