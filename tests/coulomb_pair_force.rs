// rq-846bdb8b
use std::sync::Arc;

use cudarc::driver::CudaDevice;
use heddle_md::forces::{DeviceExclusionList, Exclusion, ExclusionList, NeighborListState};
use heddle_md::gpu::{
    GpuContext, GpuError, K_COULOMB_F32, PairBuffer, ParticleBuffers, coulomb_pair_force,
    init_device, lj_pair_force,
};
use heddle_md::pbc::SimulationBox;
use heddle_md::state::ParticleState;
use heddle_md::precision::Real;

mod common;

const E_CHARGE: Real = 1.602176634e-19; // elementary charge (C)

fn default_box() -> SimulationBox {
    SimulationBox::new(10.0e-9, 10.0e-9, 10.0e-9, 0.0, 0.0, 0.0).unwrap()
}

fn build_state_with_charges(positions: &[[Real; 3]], charges: &[Real]) -> ParticleState {
    let n = positions.len();
    assert_eq!(n, charges.len());
    let mut px = Vec::with_capacity(n);
    let mut py = Vec::with_capacity(n);
    let mut pz = Vec::with_capacity(n);
    for p in positions {
        px.push(p[0]);
        py.push(p[1]);
        pz.push(p[2]);
    }
    ParticleState::new(
        px,
        py,
        pz,
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        charges.to_vec(),
        vec![0u32; n],
        None,
        None,
    )
    .expect("ParticleState::new")
}

fn empty_excl(device: &Arc<CudaDevice>, n: usize) -> DeviceExclusionList {
    DeviceExclusionList::from_host(device, &ExclusionList::empty(n))
        .expect("empty exclusion buffers")
}

fn trivial_nl(gpu: &GpuContext, sim_box: &SimulationBox, n: usize) -> NeighborListState {
    NeighborListState::new_trivial(gpu, sim_box, n).expect("trivial neighbor list")
}

fn run_coulomb(
    gpu: &GpuContext,
    sim_box: &SimulationBox,
    state: &ParticleState,
    cutoff: Real,
    r_switch: Real,
) -> (PairBuffer, NeighborListState, ParticleBuffers) {
    let n = state.particle_count();
    let particle_buffers = ParticleBuffers::new(gpu, state).expect("buffers");
    let mut pair = PairBuffer::new(gpu, n, n as u32).expect("pair buffer");
    let excl = empty_excl(&gpu.device, n);
    let nl = trivial_nl(gpu, sim_box, n);
    coulomb_pair_force(
        &particle_buffers,
        &mut pair,
        sim_box,
        cutoff,
        r_switch,
        &excl.atom_excl_offsets,
        &excl.atom_excl_partners,
        &excl.atom_excl_coul_scales,
        &nl.neighbor_list,
        &nl.neighbor_counts,
    )
    .expect("coulomb_pair_force");
    (pair, nl, particle_buffers)
}

fn download_pair_forces(pair: &PairBuffer) -> (Vec<Real>, Vec<Real>, Vec<Real>) {
    let device = pair.device.clone();
    let fx = device.dtoh_sync_copy(&pair.pair_forces_x).unwrap();
    let fy = device.dtoh_sync_copy(&pair.pair_forces_y).unwrap();
    let fz = device.dtoh_sync_copy(&pair.pair_forces_z).unwrap();
    (fx, fy, fz)
}

fn download_pair_energies(pair: &PairBuffer) -> Vec<Real> {
    pair.device.dtoh_sync_copy(&pair.pair_energies).unwrap()
}

fn download_pair_virials(pair: &PairBuffer) -> Vec<Real> {
    pair.device.dtoh_sync_copy(&pair.pair_virials).unwrap()
}

// rq-4f23c656
// rq-4d3f63fb
#[test]
fn opposite_sign_charges_attract() {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    let positions = [[0.0, 0.0, 0.0], [3.0e-10, 0.0, 0.0]];
    let charges = [E_CHARGE, -E_CHARGE];
    let state = build_state_with_charges(&positions, &charges);
    let r_cut = 5.0e-9;
    let (pair, _nl, _buf) = run_coulomb(&gpu, &sim_box, &state, r_cut, r_cut);
    let (fx, _fy, _fz) = download_pair_forces(&pair);
    let r: Real = 3.0e-10;
    // Atom 0 at origin, atom 1 at +x. Opposite charges attract atom 0 toward
    // atom 1 (along +x), so fx is positive.
    // fx = factor * dx where dx = positions[0] - positions[1] = -r and
    // factor = k_C * q_i * q_j / r^3 with q_i*q_j = -e²; fx = -k_C·e²·(-r)/r³ > 0.
    let expected = K_COULOMB_F32 * E_CHARGE * (-E_CHARGE) * (-r) / (r * r * r);
    assert!(
        (fx[0 * 2 + 1] - expected).abs() < 1.0e-9 * expected.abs().max(1.0),
        "got fx = {}, expected {}",
        fx[0 * 2 + 1],
        expected
    );
    assert!(fx[0 * 2 + 1] > 0.0, "atom 0 should be pulled toward atom 1 along +x");
}

// rq-82e4f74e
#[test]
fn same_sign_charges_repel_and_obey_newtons_third_law() {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    let positions = [[0.0, 0.0, 0.0], [3.0e-10, 0.0, 0.0]];
    let charges = [E_CHARGE, E_CHARGE];
    let state = build_state_with_charges(&positions, &charges);
    let r_cut = 5.0e-9;
    let (pair, _nl, _buf) = run_coulomb(&gpu, &sim_box, &state, r_cut, r_cut);
    let (fx, _fy, _fz) = download_pair_forces(&pair);
    // Atom 0 at origin, atom 1 at +x. Same-sign repulsion pushes atom 0
    // along -x, so fx is negative.
    assert!(fx[0 * 2 + 1] < 0.0);
    assert_eq!(fx[0 * 2 + 1], -fx[1 * 2 + 0]);
}

// rq-3b7da473
#[test]
fn zero_charges_produce_zero_force() {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    let positions = [[0.0, 0.0, 0.0], [3.0e-10, 0.0, 0.0]];
    let state = build_state_with_charges(&positions, &[0.0, 0.0]);
    let r_cut = 5.0e-9;
    let (pair, _nl, _buf) = run_coulomb(&gpu, &sim_box, &state, r_cut, r_cut);
    let (fx, fy, fz) = download_pair_forces(&pair);
    let energies = download_pair_energies(&pair);
    let virials = download_pair_virials(&pair);
    for slot in 0..fx.len() {
        assert_eq!(fx[slot], 0.0);
        assert_eq!(fy[slot], 0.0);
        assert_eq!(fz[slot], 0.0);
        assert_eq!(energies[slot], 0.0);
        assert_eq!(virials[slot], 0.0);
    }
}

// rq-02a15197
#[test]
fn mixed_charge_magnitudes_scale_linearly() {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    let positions = [[0.0, 0.0, 0.0], [3.0e-10, 0.0, 0.0]];
    let r_cut = 5.0e-9;
    let (pair1, _nl1, _buf1) = run_coulomb(
        &gpu,
        &sim_box,
        &build_state_with_charges(&positions, &[E_CHARGE, E_CHARGE]),
        r_cut,
        r_cut,
    );
    let (pair2, _nl2, _buf2) = run_coulomb(
        &gpu,
        &sim_box,
        &build_state_with_charges(&positions, &[2.0 * E_CHARGE, E_CHARGE]),
        r_cut,
        r_cut,
    );
    let (fx1, _, _) = download_pair_forces(&pair1);
    let (fx2, _, _) = download_pair_forces(&pair2);
    // Force scales linearly with q_i*q_j, so doubling q_i doubles fx.
    assert!((fx2[0 * 2 + 1] - 2.0 * fx1[0 * 2 + 1]).abs() < 1.0e-7);
}

// rq-0896c33a
#[test]
fn pair_beyond_cutoff_contributes_zero() {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    let positions = [[0.0, 0.0, 0.0], [5.0e-10, 0.0, 0.0]];
    let state = build_state_with_charges(&positions, &[E_CHARGE, E_CHARGE]);
    let r_cut = 2.0e-10; // smaller than the 5e-10 separation
    let (pair, _nl, _buf) = run_coulomb(&gpu, &sim_box, &state, r_cut, r_cut);
    let (fx, _, _) = download_pair_forces(&pair);
    let energies = download_pair_energies(&pair);
    let virials = download_pair_virials(&pair);
    assert_eq!(fx[0 * 2 + 1], 0.0);
    assert_eq!(energies[0 * 2 + 1], 0.0);
    assert_eq!(virials[0 * 2 + 1], 0.0);
}

// rq-3df3ed61
#[test]
fn pair_at_exactly_cutoff_contributes_smoothed_zero() {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    let r_cut = 3.0e-10;
    let positions = [[0.0, 0.0, 0.0], [r_cut, 0.0, 0.0]];
    let state = build_state_with_charges(&positions, &[E_CHARGE, E_CHARGE]);
    let (pair, _nl, _buf) = run_coulomb(&gpu, &sim_box, &state, r_cut, 2.7e-10);
    let (fx, _, _) = download_pair_forces(&pair);
    let energies = download_pair_energies(&pair);
    // S(r_c²) = 0 → energy zero.
    assert!(energies[0 * 2 + 1].abs() < 1.0e-25);
    // Force gets the chain-rule kick at the boundary; but evaluating S=0
    // at r=r_cut also leaves the radial multiplier zero only when
    // tau == 1 and chain_coeff == 0. Confirm fx is "near zero" relative
    // to the unsmoothed force scale.
    let unsmoothed = K_COULOMB_F32 * E_CHARGE * E_CHARGE / (r_cut * r_cut);
    assert!(
        fx[0 * 2 + 1].abs() < 1.0e-3 * unsmoothed.abs(),
        "fx = {}, unsmoothed scale = {}",
        fx[0 * 2 + 1],
        unsmoothed
    );
}

// rq-b07abbd4
#[test]
fn pair_inside_inner_plateau_is_unsmoothed() {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    let r_cut = 5.0e-10;
    let r_switch = 4.0e-10;
    let r = 3.5e-10; // < r_switch
    let positions = [[0.0, 0.0, 0.0], [r, 0.0, 0.0]];
    let state = build_state_with_charges(&positions, &[E_CHARGE, E_CHARGE]);
    let (pair, _nl, _buf) = run_coulomb(&gpu, &sim_box, &state, r_cut, r_switch);
    let (fx, _, _) = download_pair_forces(&pair);
    let energies = download_pair_energies(&pair);
    let qq = E_CHARGE * E_CHARGE;
    let expected_energy = 0.5 * K_COULOMB_F32 * qq / r;
    // Atom 0 at origin, atom 1 at +x, same-sign repulsion pushes atom 0 along
    // -x. fx = factor * dx with dx = -r and factor > 0, so fx < 0.
    // Compute in the same order as the kernel (factor * dx, where
    // factor = K * qq / r^3 evaluated via inverse r and r^2) to avoid
    // f32 underflow in the intermediate `K * qq * r` term.
    let inv_r = 1.0 / r;
    let inv_r2 = inv_r * inv_r;
    let factor = K_COULOMB_F32 * qq * inv_r * inv_r2;
    let expected_fx = factor * (-r);
    assert!(
        (energies[0 * 2 + 1] - expected_energy).abs() / expected_energy < 1.0e-5,
        "energies = {}, expected {}",
        energies[0 * 2 + 1],
        expected_energy
    );
    assert!(
        (fx[0 * 2 + 1] - expected_fx).abs() / expected_fx.abs() < 1.0e-5,
        "fx = {}, expected {}",
        fx[0 * 2 + 1],
        expected_fx
    );
}

// rq-d52bcc88
#[test]
fn pair_inside_switching_interval_is_smoothed() {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    let r_cut = 5.0e-10;
    let r_switch = 4.0e-10;
    let r = 4.5e-10; // between r_switch and cutoff
    let positions = [[0.0, 0.0, 0.0], [r, 0.0, 0.0]];
    let state = build_state_with_charges(&positions, &[E_CHARGE, E_CHARGE]);
    let (pair, _nl, _buf) = run_coulomb(&gpu, &sim_box, &state, r_cut, r_switch);
    let (fx, _, _) = download_pair_forces(&pair);
    let energies = download_pair_energies(&pair);
    let qq = E_CHARGE * E_CHARGE;
    let unswitched_energy = 0.5 * K_COULOMB_F32 * qq / r;
    // Same-sign repulsion, atom 1 at +x → atom 0 pushed along -x (fx < 0).
    // Energy is in (0, unswitched_energy) thanks to the smoothing factor.
    assert!(energies[0 * 2 + 1] > 0.0);
    assert!(energies[0 * 2 + 1] < unswitched_energy);
    // fx remains finite and points in the expected (-x) direction. The chain-
    // rule term can substantially boost |fx| in the switching region, so
    // we only assert direction and finiteness here.
    assert!(fx[0 * 2 + 1] < 0.0);
    assert!(fx[0 * 2 + 1].is_finite());
}

// rq-67678030
#[test]
fn switching_interval_equal_to_cutoff_selects_hard_cutoff() {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    let r_cut = 4.0e-10;
    let r = 3.9e-10;
    let positions = [[0.0, 0.0, 0.0], [r, 0.0, 0.0]];
    let state = build_state_with_charges(&positions, &[E_CHARGE, E_CHARGE]);
    let (pair, _nl, _buf) = run_coulomb(&gpu, &sim_box, &state, r_cut, r_cut);
    let energies = download_pair_energies(&pair);
    let expected_energy = 0.5 * K_COULOMB_F32 * E_CHARGE * E_CHARGE / r;
    assert!(
        (energies[0 * 2 + 1] - expected_energy).abs() / expected_energy < 1.0e-5,
        "energy = {}, expected {}",
        energies[0 * 2 + 1],
        expected_energy
    );
}

// rq-ef3083cd
#[test]
fn pair_across_periodic_boundary_uses_minimum_image() {
    let gpu = init_device().unwrap();
    let sim_box = SimulationBox::new(1.0e-9, 10.0e-9, 10.0e-9, 0.0, 0.0, 0.0).unwrap();
    let lx = 1.0e-9;
    let positions = [
        [-lx * 0.5 + 1.0e-10, 0.0, 0.0],
        [lx * 0.5 - 1.0e-10, 0.0, 0.0],
    ];
    let charges = [E_CHARGE, -E_CHARGE];
    let state = build_state_with_charges(&positions, &charges);
    let r_cut = 5.0e-10;
    let (pair, _nl, _buf) = run_coulomb(&gpu, &sim_box, &state, r_cut, r_cut);
    let (fx, _, _) = download_pair_forces(&pair);
    // Minimum-image separation is 0.2e-10 across the +x boundary; attraction is
    // strong (negative on atom 0).
    assert!(fx[0 * 2 + 1] < 0.0, "expected negative attractive fx; got {}", fx[0 * 2 + 1]);
}

// rq-9af1f9dc
#[test]
fn minimum_image_works_for_triclinic_box() {
    let gpu = init_device().unwrap();
    // Triclinic box with non-zero tilts; place a charged pair close together
    // in Cartesian coords so the minimum-image displacement is well within
    // the perpendicular-width-limited cutoff.
    let sim_box =
        SimulationBox::new(10.0e-9, 10.0e-9, 10.0e-9, 2.0e-10, 1.0e-10, -3.0e-10).unwrap();
    let positions = [[0.0, 0.0, 0.0], [3.0e-10, 0.0, 0.0]];
    let charges = [E_CHARGE, E_CHARGE];
    let state = build_state_with_charges(&positions, &charges);
    let r_cut = 5.0e-10;
    let (pair, _nl, _buf) = run_coulomb(&gpu, &sim_box, &state, r_cut, r_cut);
    let (fx, _, _) = download_pair_forces(&pair);
    // Same-sign charges, in-cell separation 3 Å with atom 1 at +x →
    // atom 0 is repelled along -x.
    assert!(fx[0 * 2 + 1] < 0.0);
}

// rq-bf7dfc6d
#[test]
fn self_slot_in_trivial_mode_yields_zero() {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    let positions = [[0.0, 0.0, 0.0]];
    let state = build_state_with_charges(&positions, &[E_CHARGE]);
    let r_cut = 5.0e-10;
    let (pair, _nl, _buf) = run_coulomb(&gpu, &sim_box, &state, r_cut, r_cut);
    let (fx, _, _) = download_pair_forces(&pair);
    let energies = download_pair_energies(&pair);
    let virials = download_pair_virials(&pair);
    assert_eq!(fx[0], 0.0);
    assert_eq!(energies[0], 0.0);
    assert_eq!(virials[0], 0.0);
}

fn run_coulomb_with_excl(
    gpu: &GpuContext,
    sim_box: &SimulationBox,
    state: &ParticleState,
    excl: &ExclusionList,
    cutoff: Real,
    r_switch: Real,
) -> (PairBuffer, ParticleBuffers) {
    let n = state.particle_count();
    let particle_buffers = ParticleBuffers::new(gpu, state).expect("buffers");
    let mut pair = PairBuffer::new(gpu, n, n as u32).expect("pair buffer");
    let device_excl = DeviceExclusionList::from_host(&gpu.device, excl).expect("device excl");
    let nl = trivial_nl(gpu, sim_box, n);
    coulomb_pair_force(
        &particle_buffers,
        &mut pair,
        sim_box,
        cutoff,
        r_switch,
        &device_excl.atom_excl_offsets,
        &device_excl.atom_excl_partners,
        &device_excl.atom_excl_coul_scales,
        &nl.neighbor_list,
        &nl.neighbor_counts,
    )
    .expect("coulomb_pair_force");
    (pair, particle_buffers)
}

fn excl_with_scales(n: usize, atom_i: u32, atom_j: u32, lj: Real, coul: Real) -> ExclusionList {
    let (a, b) = if atom_i < atom_j {
        (atom_i, atom_j)
    } else {
        (atom_j, atom_i)
    };
    let mut atom_excl_offsets = vec![0u32; n + 1];
    atom_excl_offsets[a as usize + 1] += 1;
    atom_excl_offsets[b as usize + 1] += 1;
    for i in 1..=n {
        atom_excl_offsets[i] += atom_excl_offsets[i - 1];
    }
    let mut atom_excl_partners = vec![0u32; 2];
    let mut atom_excl_lj_scales = vec![0.0; 2];
    let mut atom_excl_coul_scales = vec![0.0; 2];
    // Slot for atom a: partner b
    let slot_a = atom_excl_offsets[a as usize] as usize;
    let slot_b = atom_excl_offsets[b as usize] as usize;
    atom_excl_partners[slot_a] = b;
    atom_excl_lj_scales[slot_a] = lj;
    atom_excl_coul_scales[slot_a] = coul;
    atom_excl_partners[slot_b] = a;
    atom_excl_lj_scales[slot_b] = lj;
    atom_excl_coul_scales[slot_b] = coul;
    ExclusionList {
        entries: vec![Exclusion {
            atom_i: a,
            atom_j: b,
            scale_lj: lj,
            scale_coul: coul,
        }],
        atom_excl_offsets,
        atom_excl_partners,
        atom_excl_lj_scales,
        atom_excl_coul_scales,
        particle_count: n,
    }
}

// rq-c4d4608f
#[test]
fn pair_with_coul_exclusion_scale_zero_contributes_nothing() {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    let positions = [[0.0, 0.0, 0.0], [3.0e-10, 0.0, 0.0]];
    let state = build_state_with_charges(&positions, &[E_CHARGE, -E_CHARGE]);
    let r_cut = 5.0e-10;
    let excl = excl_with_scales(2, 0, 1, 1.0, 0.0);
    let (pair, _buf) = run_coulomb_with_excl(&gpu, &sim_box, &state, &excl, r_cut, r_cut);
    let (fx, fy, fz) = download_pair_forces(&pair);
    let energies = download_pair_energies(&pair);
    let virials = download_pair_virials(&pair);
    assert_eq!(fx[0 * 2 + 1], 0.0);
    assert_eq!(fy[0 * 2 + 1], 0.0);
    assert_eq!(fz[0 * 2 + 1], 0.0);
    assert_eq!(energies[0 * 2 + 1], 0.0);
    assert_eq!(virials[0 * 2 + 1], 0.0);
}

// rq-d26e9f9c
#[test]
fn pair_with_coul_exclusion_scale_half_contributes_half() {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    let positions = [[0.0, 0.0, 0.0], [3.0e-10, 0.0, 0.0]];
    let state = build_state_with_charges(&positions, &[E_CHARGE, -E_CHARGE]);
    let r_cut = 5.0e-10;
    let excl_none = ExclusionList::empty(2);
    let excl_half = excl_with_scales(2, 0, 1, 1.0, 0.5);
    let (pair_full, _) = run_coulomb_with_excl(&gpu, &sim_box, &state, &excl_none, r_cut, r_cut);
    let (pair_half, _) = run_coulomb_with_excl(&gpu, &sim_box, &state, &excl_half, r_cut, r_cut);
    let (fx_full, _, _) = download_pair_forces(&pair_full);
    let (fx_half, _, _) = download_pair_forces(&pair_half);
    let e_full = download_pair_energies(&pair_full);
    let e_half = download_pair_energies(&pair_half);
    assert!(
        (fx_half[0 * 2 + 1] - 0.5 * fx_full[0 * 2 + 1]).abs() < 1.0e-15,
        "scaled fx mismatch"
    );
    assert!(
        (e_half[0 * 2 + 1] - 0.5 * e_full[0 * 2 + 1]).abs() < 1.0e-30,
        "scaled energy mismatch"
    );
}

// rq-8c96d3c7
#[test]
fn coulomb_and_lj_exclusions_are_independent() {
    use heddle_md::gpu::LennardJonesParameterTable;
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    let positions = [[0.0, 0.0, 0.0], [3.0e-10, 0.0, 0.0]];
    let n = positions.len();
    let state = build_state_with_charges(&positions, &[E_CHARGE, -E_CHARGE]);
    let particle_buffers = ParticleBuffers::new(&gpu, &state).unwrap();

    // ExclusionList with different LJ and Coulomb scales.
    let excl_host = excl_with_scales(n, 0, 1, 0.5, 0.833);
    let device_excl = DeviceExclusionList::from_host(&gpu.device, &excl_host).unwrap();
    let nl = trivial_nl(&gpu, &sim_box, n);

    // Run Coulomb only first.
    let mut coul_pair = PairBuffer::new(&gpu, n, n as u32).unwrap();
    let r_cut = 5.0e-10;
    coulomb_pair_force(
        &particle_buffers,
        &mut coul_pair,
        &sim_box,
        r_cut,
        r_cut,
        &device_excl.atom_excl_offsets,
        &device_excl.atom_excl_partners,
        &device_excl.atom_excl_coul_scales,
        &nl.neighbor_list,
        &nl.neighbor_counts,
    )
    .unwrap();
    let coul_e_scaled = download_pair_energies(&coul_pair);

    // Run Coulomb with full scale (no exclusion) for the unscaled reference.
    let excl_none = ExclusionList::empty(n);
    let device_excl_none =
        DeviceExclusionList::from_host(&gpu.device, &excl_none).unwrap();
    let mut coul_pair_full = PairBuffer::new(&gpu, n, n as u32).unwrap();
    coulomb_pair_force(
        &particle_buffers,
        &mut coul_pair_full,
        &sim_box,
        r_cut,
        r_cut,
        &device_excl_none.atom_excl_offsets,
        &device_excl_none.atom_excl_partners,
        &device_excl_none.atom_excl_coul_scales,
        &nl.neighbor_list,
        &nl.neighbor_counts,
    )
    .unwrap();
    let coul_e_full = download_pair_energies(&coul_pair_full);
    // Coulomb is scaled by 0.833.
    assert!(
        ((coul_e_scaled[0 * 2 + 1] / coul_e_full[0 * 2 + 1]) - 0.833).abs() < 1.0e-5
    );

    // Run LJ with the same DeviceExclusionList (same partner table); LJ scales
    // by 0.5. Use sigma != r to get a non-zero LJ energy (LJ(r=σ) ≡ 0).
    let sigma = 2.0e-10;
    let epsilon = 1.0e-21;
    let table = LennardJonesParameterTable {
        n_types: 1,
        sigma: gpu.device.htod_sync_copy(&[sigma]).unwrap(),
        epsilon: gpu.device.htod_sync_copy(&[epsilon]).unwrap(),
        cutoff: gpu.device.htod_sync_copy(&[r_cut]).unwrap(),
        switch: gpu.device.htod_sync_copy(&[r_cut]).unwrap(),
    };
    let mut lj_pair = PairBuffer::new(&gpu, n, n as u32).unwrap();
    lj_pair_force(
        &particle_buffers,
        &mut lj_pair,
        &sim_box,
        &table,
        &device_excl.atom_excl_offsets,
        &device_excl.atom_excl_partners,
        &device_excl.atom_excl_lj_scales,
        &nl.neighbor_list,
        &nl.neighbor_counts,
    )
    .unwrap();
    let mut lj_pair_full = PairBuffer::new(&gpu, n, n as u32).unwrap();
    lj_pair_force(
        &particle_buffers,
        &mut lj_pair_full,
        &sim_box,
        &table,
        &device_excl_none.atom_excl_offsets,
        &device_excl_none.atom_excl_partners,
        &device_excl_none.atom_excl_lj_scales,
        &nl.neighbor_list,
        &nl.neighbor_counts,
    )
    .unwrap();
    let lj_e_scaled = download_pair_energies(&lj_pair);
    let lj_e_full = download_pair_energies(&lj_pair_full);
    // LJ is scaled by 0.5.
    assert!(
        ((lj_e_scaled[0 * 2 + 1] / lj_e_full[0 * 2 + 1]) - 0.5).abs() < 1.0e-5
    );
}

// rq-5444c7ae
#[test]
fn pair_energies_carries_half_potential() {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    let r = 3.0e-10;
    let positions = [[0.0, 0.0, 0.0], [r, 0.0, 0.0]];
    let state = build_state_with_charges(&positions, &[E_CHARGE, E_CHARGE]);
    let r_cut = 5.0e-10;
    let (pair, _nl, _buf) = run_coulomb(&gpu, &sim_box, &state, r_cut, r_cut);
    let e = download_pair_energies(&pair);
    let expected = 0.5 * K_COULOMB_F32 * E_CHARGE * E_CHARGE / r;
    assert!((e[0 * 2 + 1] - expected).abs() / expected < 1.0e-5);
    // The j→i slot is also half.
    assert!((e[1 * 2 + 0] - expected).abs() / expected < 1.0e-5);
}

// rq-e412e54a
#[test]
fn pair_virials_carries_half_scalar_virial() {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    let r = 3.0e-10;
    let positions = [[0.0, 0.0, 0.0], [r, 0.0, 0.0]];
    let state = build_state_with_charges(&positions, &[E_CHARGE, E_CHARGE]);
    let r_cut = 5.0e-10;
    let (pair, _nl, _buf) = run_coulomb(&gpu, &sim_box, &state, r_cut, r_cut);
    let w = download_pair_virials(&pair);
    // U = k_C q^2 / r, F = k_C q^2 / r^2, scalar virial = F · r = k_C q^2 / r.
    let scalar = K_COULOMB_F32 * E_CHARGE * E_CHARGE / r;
    let expected_half = 0.5 * scalar;
    assert!((w[0 * 2 + 1] - expected_half).abs() / expected_half < 1.0e-5);
}

// rq-d01b6fb0
#[test]
fn slots_beyond_neighbor_counts_are_zeroed() {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    // To trigger the "slot beyond neighbor_counts[i]" branch we build a
    // custom neighbor list whose `max_neighbors` exceeds each atom's actual
    // neighbor count. Two particles, max_neighbors = 4, neighbor_counts = 1
    // for each.
    let positions = [[0.0, 0.0, 0.0], [3.0e-10, 0.0, 0.0]];
    let n = positions.len();
    let state = build_state_with_charges(&positions, &[E_CHARGE, E_CHARGE]);
    let particle_buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let max_neighbors: u32 = 4;
    let mut pair = PairBuffer::new(&gpu, n, max_neighbors).unwrap();
    let device_excl = empty_excl(&gpu.device, n);

    // Custom neighbor_list/counts: atom 0's partner is 1, atom 1's partner is 0,
    // trailing slots are filler (the kernel won't read past neighbor_counts[i]).
    let host_nl: Vec<u32> = vec![
        1, 0, 0, 0, // atom 0
        0, 0, 0, 0, // atom 1
    ];
    let host_counts: Vec<u32> = vec![1, 1];
    let neighbor_list = gpu.device.htod_sync_copy(&host_nl).unwrap();
    let neighbor_counts = gpu.device.htod_sync_copy(&host_counts).unwrap();

    let r_cut = 1.0e-9;
    coulomb_pair_force(
        &particle_buffers,
        &mut pair,
        &sim_box,
        r_cut,
        r_cut,
        &device_excl.atom_excl_offsets,
        &device_excl.atom_excl_partners,
        &device_excl.atom_excl_coul_scales,
        &neighbor_list,
        &neighbor_counts,
    )
    .unwrap();
    let (fx, fy, fz) = download_pair_forces(&pair);
    let energies = download_pair_energies(&pair);
    let virials = download_pair_virials(&pair);
    for i in 0..n {
        for k in 1..max_neighbors as usize {
            let slot = i * max_neighbors as usize + k;
            assert_eq!(fx[slot], 0.0, "fx[{slot}] should be zero");
            assert_eq!(fy[slot], 0.0, "fy[{slot}] should be zero");
            assert_eq!(fz[slot], 0.0, "fz[{slot}] should be zero");
            assert_eq!(energies[slot], 0.0, "energy[{slot}] should be zero");
            assert_eq!(virials[slot], 0.0, "virial[{slot}] should be zero");
        }
    }
    // The k=0 slot of atom 0 (partner=1) should be non-zero.
    assert!(fx[0].abs() > 0.0);
}

// rq-1a0f3eef
#[test]
fn identical_inputs_produce_byte_identical_outputs() {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    let positions = [[0.0, 0.0, 0.0], [3.0e-10, 0.0, 0.0]];
    let state = build_state_with_charges(&positions, &[E_CHARGE, -E_CHARGE]);
    let r_cut = 5.0e-10;
    let (pair1, _, _) = run_coulomb(&gpu, &sim_box, &state, r_cut, r_cut);
    let (pair2, _, _) = run_coulomb(&gpu, &sim_box, &state, r_cut, r_cut);
    let (fx1, fy1, fz1) = download_pair_forces(&pair1);
    let (fx2, fy2, fz2) = download_pair_forces(&pair2);
    let e1 = download_pair_energies(&pair1);
    let e2 = download_pair_energies(&pair2);
    let w1 = download_pair_virials(&pair1);
    let w2 = download_pair_virials(&pair2);
    assert_eq!(fx1, fx2);
    assert_eq!(fy1, fy2);
    assert_eq!(fz1, fz2);
    assert_eq!(e1, e2);
    assert_eq!(w1, w2);
}

// rq-76a6be2f
#[test]
fn zero_particles_is_noop() -> Result<(), GpuError> {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    let state = ParticleState::new(
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
        None,
    )
    .unwrap();
    let particle_buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut pair = PairBuffer::new(&gpu, 0, 1).unwrap();
    let excl = empty_excl(&gpu.device, 0);
    // Build a trivial neighbor list for 0 particles.
    let nl = NeighborListState::new_trivial(&gpu, &sim_box, 0).unwrap();
    coulomb_pair_force(
        &particle_buffers,
        &mut pair,
        &sim_box,
        1.0e-9,
        1.0e-9,
        &excl.atom_excl_offsets,
        &excl.atom_excl_partners,
        &excl.atom_excl_coul_scales,
        &nl.neighbor_list,
        &nl.neighbor_counts,
    )?;
    Ok(())
}

// rq-ee4ebbda
#[test]
fn neutral_system_produces_zero_contribution() {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    let positions = [[0.0, 0.0, 0.0], [3.0e-10, 0.0, 0.0]];
    let state = build_state_with_charges(&positions, &[0.0, 0.0]);
    let r_cut = 5.0e-10;
    let (pair, _nl, _buf) = run_coulomb(&gpu, &sim_box, &state, r_cut, r_cut);
    let (fx, fy, fz) = download_pair_forces(&pair);
    let energies = download_pair_energies(&pair);
    let virials = download_pair_virials(&pair);
    // The kernel still runs over every pair, producing zeros.
    for slot in 0..fx.len() {
        assert_eq!(fx[slot], 0.0);
        assert_eq!(fy[slot], 0.0);
        assert_eq!(fz[slot], 0.0);
        assert_eq!(energies[slot], 0.0);
        assert_eq!(virials[slot], 0.0);
    }
}

// rq-f652bf7c
#[test]
fn forces_on_pair_members_are_equal_and_opposite_bit_exact() {
    let gpu = init_device().unwrap();
    let sim_box = default_box();
    // Off-axis pair to exercise all three force components.
    let positions = [[0.0, 0.0, 0.0], [3.0e-10, 1.0e-10, -0.5e-10]];
    let state = build_state_with_charges(&positions, &[E_CHARGE, -E_CHARGE]);
    let r_cut = 5.0e-10;
    let (pair, _nl, _buf) = run_coulomb(&gpu, &sim_box, &state, r_cut, r_cut);
    let (fx, fy, fz) = download_pair_forces(&pair);
    assert_eq!(fx[0 * 2 + 1], -fx[1 * 2 + 0]);
    assert_eq!(fy[0 * 2 + 1], -fy[1 * 2 + 0]);
    assert_eq!(fz[0 * 2 + 1], -fz[1 * 2 + 0]);
}
