//! Truncated Coulomb pair-force kernel tests.
//!
//! Implements switching-function and exclusion-scaling scenarios from
//! `rqm/forces/coulomb-pair-force.md`. Per-pair force assertions use a
//! direct closed-form reference; assertions involving switching exercise
//! the CHARMM-style C¹ polynomial branch.

mod common;

use cudarc::driver::CudaSlice;
use heddle_md::forces::{AggregateLevel, DeviceExclusionList};
use heddle_md::gpu::{
    GpuContext, K_COULOMB_F32, ParticleBuffers, SlotOutputBuffers, coulomb_pair_force, init_device,
};
use heddle_md::pbc::SimulationBox;
use heddle_md::precision::Real;
use heddle_md::state::ParticleState;

use common::{empty_exclusions, exclusions_from_entries};

fn box_20() -> SimulationBox {
    SimulationBox::new(20.0, 20.0, 20.0, 0.0, 0.0, 0.0).unwrap()
}

fn two_particles_charged(separation: Real, q0: Real, q1: Real) -> ParticleState {
    ParticleState::new(
        vec![0.0 as Real, separation],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![1.0; 2],
        vec![q0, q1],
        vec![0u32; 2],
        None,
        None,
    )
    .unwrap()
}

fn launch_coulomb(
    gpu: &GpuContext,
    buffers: &ParticleBuffers,
    output: &mut SlotOutputBuffers,
    sim_box: &SimulationBox,
    cutoff: Real,
    r_switch: Real,
    exclusions: &DeviceExclusionList,
    level: AggregateLevel,
) {
    let n = buffers.particle_count();
    let nl_flat: Vec<u32> = (0..n).flat_map(|i| (0..n).map(move |j| j as u32)).collect();
    let counts = vec![n as u32; n];
    let nl = gpu.device.htod_sync_copy(&nl_flat).unwrap();
    let cnt = gpu.device.htod_sync_copy(&counts).unwrap();
    let mut view = output.view();
    coulomb_pair_force(
        buffers,
        &mut view,
        sim_box,
        cutoff,
        r_switch,
        &exclusions.atom_excl_offsets,
        &exclusions.atom_excl_partners,
        &exclusions.atom_excl_coul_scales,
        &nl,
        &cnt,
        n as u32,
        level,
    )
    .unwrap();
}

struct PairResult {
    fx0: Real,
    fy0: Real,
    fz0: Real,
    fx1: Real,
    e0: Real,
    e1: Real,
    w0: Real,
    w1: Real,
}

fn run_pair(
    separation: Real,
    q0: Real,
    q1: Real,
    cutoff: Real,
    r_switch: Real,
    level: AggregateLevel,
    exclusions_input: Option<&[(u32, u32, Real, Real)]>,
) -> PairResult {
    let gpu = init_device().unwrap();
    let state = two_particles_charged(separation, q0, q1);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = box_20();
    let excl = match exclusions_input {
        None => empty_exclusions(&gpu.device, 2),
        Some(entries) => exclusions_from_entries(&gpu.device, 2, entries),
    };
    let mut output = SlotOutputBuffers::new(&gpu.device, 2).unwrap();
    launch_coulomb(&gpu, &buffers, &mut output, &sim_box, cutoff, r_switch, &excl, level);

    let fx = gpu.device.dtoh_sync_copy(&output.force_x).unwrap();
    let fy = gpu.device.dtoh_sync_copy(&output.force_y).unwrap();
    let fz = gpu.device.dtoh_sync_copy(&output.force_z).unwrap();
    let e = gpu.device.dtoh_sync_copy(&output.energy).unwrap();
    let w = gpu.device.dtoh_sync_copy(&output.virial).unwrap();
    PairResult {
        fx0: fx[0],
        fy0: fy[0],
        fz0: fz[0],
        fx1: fx[1],
        e0: e[0],
        e1: e[1],
        w0: w[0],
        w1: w[1],
    }
}

fn closed_form_coulomb_factor(r: Real, q0: Real, q1: Real) -> Real {
    // factor such that fx = factor * dx (dx = r0_x - r1_x = -r for our setup).
    K_COULOMB_F32 * q0 * q1 / (r * r * r)
}

fn closed_form_coulomb_energy(r: Real, q0: Real, q1: Real) -> Real {
    K_COULOMB_F32 * q0 * q1 / r
}

// =================================================================
// Sign conventions: same-sign charges repel, opposite attract.
// =================================================================

#[test]
fn opposite_sign_charges_attract() {
    let res = run_pair(3.0, 1.0, -1.0, 5.0, 5.0, AggregateLevel::ForcesOnly, None);
    // dx = 0 - 3 = -3, factor = k_C * (-1) / 27 < 0
    // fx0 = factor * dx = (-k_C/27) * (-3) > 0 → particle 0 pulled toward +x (toward particle 1).
    assert!(res.fx0 > 0.0, "fx0 = {} (expected > 0 — attraction toward +x)", res.fx0);
}

#[test]
fn same_sign_charges_repel() {
    let res = run_pair(3.0, 1.0, 1.0, 5.0, 5.0, AggregateLevel::ForcesOnly, None);
    // fx0 < 0 → particle 0 pushed toward -x (away from particle 1).
    assert!(res.fx0 < 0.0, "fx0 = {} (expected < 0 — repulsion)", res.fx0);
}

#[test]
fn doubling_one_charge_doubles_the_force() {
    let base = run_pair(3.0, 1.0, 1.0, 5.0, 5.0, AggregateLevel::ForcesOnly, None);
    let doubled = run_pair(3.0, 2.0, 1.0, 5.0, 5.0, AggregateLevel::ForcesOnly, None);
    let expected = 2.0 * base.fx0;
    let rel = ((doubled.fx0 - expected).abs() / expected.abs()) as f64;
    assert!(rel < 1e-5, "doubled fx0 = {} vs expected {} (rel {})", doubled.fx0, expected, rel);
}

// =================================================================
// Closed-form correctness.
// =================================================================

#[test]
fn two_particles_at_fixed_separation_match_closed_form() {
    let r = 3.0;
    let q0 = 1.0;
    let q1 = -1.0;
    let res = run_pair(r, q0, q1, 5.0, 5.0, AggregateLevel::ForcesAndScalars, None);
    let expected_fx0 = closed_form_coulomb_factor(r, q0, q1) * (0.0 - r);
    let rel_f = ((res.fx0 - expected_fx0).abs() / expected_fx0.abs()) as f64;
    assert!(rel_f < 1e-5, "fx0 = {} vs expected {} (rel {})", res.fx0, expected_fx0, rel_f);
    let expected_e = closed_form_coulomb_energy(r, q0, q1);
    let rel_e = ((res.e0 + res.e1 - expected_e).abs() / expected_e.abs()) as f64;
    assert!(rel_e < 1e-5, "e0+e1 = {} vs expected {}", res.e0 + res.e1, expected_e);
}

// =================================================================
// Switching function.
// =================================================================

#[test]
fn pair_inside_inner_plateau_is_unsmoothed() {
    // r = 3 inside r_switch=4, so switching factor S=1.
    let r = 3.0;
    let res = run_pair(r, 1.0, 1.0, 5.0, 4.0, AggregateLevel::ForcesAndScalars, None);
    let expected_fx0 = closed_form_coulomb_factor(r, 1.0, 1.0) * (0.0 - r);
    let rel = ((res.fx0 - expected_fx0).abs() / expected_fx0.abs()) as f64;
    assert!(rel < 1e-5, "fx0 = {} vs expected {} (rel {})", res.fx0, expected_fx0, rel);
}

#[test]
fn pair_inside_switching_interval_is_smoothed() {
    // r = 4.5 between r_switch=4 and cutoff=5.
    let switched = run_pair(4.5, 1.0, 1.0, 5.0, 4.0, AggregateLevel::ForcesAndScalars, None);
    let unmodified = run_pair(4.5, 1.0, 1.0, 5.0, 5.0, AggregateLevel::ForcesAndScalars, None);
    // For Coulomb r⁻² in repulsive case, the smoothed energy is strictly
    // between 0 and the unmodified value.
    assert!(
        switched.e0.abs() < unmodified.e0.abs(),
        "switched |e0| = {} not less than unmodified = {}",
        switched.e0.abs(),
        unmodified.e0.abs()
    );
}

#[test]
fn pair_at_exactly_cutoff_contributes_zero() {
    // S(r_c²) = 0 — smoothing forces both force and energy to zero at cutoff.
    let res = run_pair(5.0, 1.0, 1.0, 5.0, 4.0, AggregateLevel::ForcesAndScalars, None);
    assert_eq!(res.fx0, 0.0);
    assert_eq!(res.fy0, 0.0);
    assert_eq!(res.fz0, 0.0);
    assert_eq!(res.e0, 0.0);
    assert_eq!(res.e1, 0.0);
    assert_eq!(res.w0, 0.0);
    assert_eq!(res.w1, 0.0);
}

#[test]
fn switching_interval_r_switch_eq_cutoff_selects_hard_cutoff() {
    // r_switch = cutoff = 4 → no smoothing inside, hard cutoff at 4.
    let r = 3.9;
    let res = run_pair(r, 1.0, 1.0, 4.0, 4.0, AggregateLevel::ForcesAndScalars, None);
    let expected_e = closed_form_coulomb_energy(r, 1.0, 1.0);
    let rel = ((res.e0 + res.e1 - expected_e).abs() / expected_e.abs()) as f64;
    assert!(rel < 1e-5, "e0+e1 = {} vs expected {}", res.e0 + res.e1, expected_e);
}

#[test]
fn coulomb_force_is_c1_continuous_at_r_switch() {
    // CHARMM-1 switching gives a continuous force at r_switch: just below
    // (in the plateau) factor = unswitched; just above factor = S·unswitched
    // + chain_coeff·energy, where both correction terms vanish at τ = 0.
    // The bound is 5% rather than 1% because the chain-rule term carries
    // f32-precision noise of order a few percent at δr = 1e-3 a₀.
    let f_below = run_pair(4.0 - 1.0e-3, 1.0, -1.0, 5.0, 4.0, AggregateLevel::ForcesOnly, None);
    let f_above = run_pair(4.0 + 1.0e-3, 1.0, -1.0, 5.0, 4.0, AggregateLevel::ForcesOnly, None);
    let diff = (f_below.fx0 - f_above.fx0).abs();
    let bound = 5.0e-2 * f_below.fx0.abs();
    assert!(
        diff <= bound,
        "discontinuity at r_switch: |Δfx0| = {} > 5% of f_below.fx0 ({})",
        diff,
        bound
    );
}

#[test]
fn coulomb_force_decays_toward_zero_near_r_cut() {
    // r very close to r_cut so that both S(τ) and chain_coeff(τ) (each
    // proportional to (1 - τ)) are small enough that the resulting force
    // is well under 1% of the unmodified value at r_switch. At δ = 1e-3
    // a₀ from r_cut the chain-rule term is still a few percent.
    let f_inside = run_pair(5.0 - 1.0e-5, 1.0, -1.0, 5.0, 4.0, AggregateLevel::ForcesOnly, None);
    let unmodified_at_switch = closed_form_coulomb_factor(4.0, 1.0, -1.0) * (0.0 - 4.0);
    let bound = 1.0e-2 * unmodified_at_switch.abs();
    assert!(
        f_inside.fx0.abs() < bound,
        "force just inside cutoff = {} not small enough (bound = {})",
        f_inside.fx0.abs(),
        bound
    );
}

// =================================================================
// Exclusion scaling.
// =================================================================

#[test]
fn pair_with_coulomb_exclusion_scale_zero_contributes_nothing() {
    let res = run_pair(
        3.0,
        1.0,
        -1.0,
        5.0,
        5.0,
        AggregateLevel::ForcesAndScalars,
        Some(&[(0, 1, 1.0, 0.0)]),
    );
    assert_eq!(res.fx0, 0.0);
    assert_eq!(res.fy0, 0.0);
    assert_eq!(res.fz0, 0.0);
    assert_eq!(res.e0, 0.0);
    assert_eq!(res.e1, 0.0);
    assert_eq!(res.w0, 0.0);
}

#[test]
fn pair_with_coulomb_exclusion_scale_half_contributes_half() {
    let scaled = run_pair(
        3.0,
        1.0,
        -1.0,
        5.0,
        5.0,
        AggregateLevel::ForcesAndScalars,
        Some(&[(0, 1, 1.0, 0.5)]),
    );
    let unscaled = run_pair(3.0, 1.0, -1.0, 5.0, 5.0, AggregateLevel::ForcesAndScalars, None);
    assert_eq!(scaled.fx0.to_bits(), (0.5 * unscaled.fx0).to_bits());
    assert_eq!(scaled.e0.to_bits(), (0.5 * unscaled.e0).to_bits());
    assert_eq!(scaled.w0.to_bits(), (0.5 * unscaled.w0).to_bits());
}

#[test]
fn pair_with_coulomb_exclusion_scale_one_matches_no_exclusion() {
    let explicit = run_pair(
        3.0,
        1.0,
        -1.0,
        5.0,
        5.0,
        AggregateLevel::ForcesAndScalars,
        Some(&[(0, 1, 1.0, 1.0)]),
    );
    let implicit = run_pair(3.0, 1.0, -1.0, 5.0, 5.0, AggregateLevel::ForcesAndScalars, None);
    assert_eq!(explicit.fx0.to_bits(), implicit.fx0.to_bits());
    assert_eq!(explicit.e0.to_bits(), implicit.e0.to_bits());
    assert_eq!(explicit.w0.to_bits(), implicit.w0.to_bits());
}

#[test]
fn coulomb_exclusion_only_applies_to_listed_pair() {
    // 3-particle system, exclude only (0, 1) Coulomb. Particle 2's
    // contributions are unaffected.
    let gpu = init_device().unwrap();
    let state = ParticleState::new(
        vec![0.0 as Real, 2.0, 4.0],
        vec![0.0; 3],
        vec![0.0; 3],
        vec![0.0; 3],
        vec![0.0; 3],
        vec![0.0; 3],
        vec![1.0; 3],
        vec![1.0, -1.0, 1.0],
        vec![0u32; 3],
        None,
        None,
    )
    .unwrap();
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = box_20();
    let excl_partial = exclusions_from_entries(&gpu.device, 3, &[(0, 1, 1.0, 0.0)]);
    let excl_empty = empty_exclusions(&gpu.device, 3);

    let mut out_excl = SlotOutputBuffers::new(&gpu.device, 3).unwrap();
    launch_coulomb(&gpu, &buffers, &mut out_excl, &sim_box, 5.0, 5.0, &excl_partial, AggregateLevel::ForcesOnly);
    let mut out_full = SlotOutputBuffers::new(&gpu.device, 3).unwrap();
    launch_coulomb(&gpu, &buffers, &mut out_full, &sim_box, 5.0, 5.0, &excl_empty, AggregateLevel::ForcesOnly);

    let fx_excl = gpu.device.dtoh_sync_copy(&out_excl.force_x).unwrap();
    let fx_full = gpu.device.dtoh_sync_copy(&out_full.force_x).unwrap();

    // Particle 2: not in any exclusion entry → its force is unchanged.
    assert_eq!(fx_excl[2].to_bits(), fx_full[2].to_bits(), "particle 2 must be unaffected");
    // Particle 0: loses contribution from particle 1.
    let f01 = closed_form_coulomb_factor(2.0, 1.0, -1.0) * (0.0 - 2.0);
    let expected_fx0 = fx_full[0] - f01;
    let rel = ((fx_excl[0] - expected_fx0).abs() / expected_fx0.abs()) as f64;
    assert!(
        rel < 1e-4,
        "particle 0 with (0,1) Coulomb excluded: fx0 = {} vs expected {} (rel {})",
        fx_excl[0],
        expected_fx0,
        rel
    );
}

#[test]
fn coulomb_exclusion_with_switching_scales_force_energy_virial_uniformly() {
    let r = 4.5;
    let cutoff = 5.0;
    let r_switch = 4.0;
    let scaled = run_pair(
        r,
        1.0,
        -1.0,
        cutoff,
        r_switch,
        AggregateLevel::ForcesAndScalars,
        Some(&[(0, 1, 1.0, 0.5)]),
    );
    let unscaled = run_pair(r, 1.0, -1.0, cutoff, r_switch, AggregateLevel::ForcesAndScalars, None);
    assert_eq!(scaled.fx0.to_bits(), (0.5 * unscaled.fx0).to_bits());
    assert_eq!(scaled.e0.to_bits(), (0.5 * unscaled.e0).to_bits());
    assert_eq!(scaled.w0.to_bits(), (0.5 * unscaled.w0).to_bits());
}

#[test]
fn coulomb_and_lj_exclusions_are_independent() {
    // The exclusion entry carries scale_lj=0.5 and scale_coul=0.833. The
    // Coulomb launcher reads only scale_coul. With scale_coul=0.833,
    // the Coulomb force is 0.833× the unscaled value.
    let scaled = run_pair(
        3.0,
        1.0,
        -1.0,
        5.0,
        5.0,
        AggregateLevel::ForcesOnly,
        Some(&[(0, 1, 0.5, 0.833)]),
    );
    let unscaled = run_pair(3.0, 1.0, -1.0, 5.0, 5.0, AggregateLevel::ForcesOnly, None);
    let expected = (0.833 as Real) * unscaled.fx0;
    let rel = ((scaled.fx0 - expected).abs() / expected.abs()) as f64;
    assert!(
        rel < 1e-5,
        "Coulomb fx0 = {} vs expected (0.833 × unscaled) {} (rel {})",
        scaled.fx0,
        expected,
        rel
    );
}
