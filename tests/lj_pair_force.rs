//! Lennard-Jones pair-force kernel tests.
//!
//! Implements the switching-function and exclusion-scaling Gherkin
//! scenarios in `rqm/forces/lj-pair-force.md`. Test bodies map 1:1 to
//! the scenarios; the `@rq-…` ID is in a leading comment on each test.

mod common;

use std::sync::Arc;

use cudarc::driver::CudaDevice;
use heddle_md::forces::{
    AggregateLevel, DeviceExclusionList, NeighborListState, SlotOutputView,
};
use heddle_md::gpu::{
    GpuContext, LennardJonesParameterTable, ParticleBuffers, SlotOutputBuffers, init_device,
    lj_pair_force,
};
use heddle_md::pbc::SimulationBox;
use heddle_md::precision::Real;
use heddle_md::state::ParticleState;

use common::{empty_exclusions, exclusions_from_entries, single_type_lj_table_with_switch};

// =================================================================
// Fixtures.
// =================================================================

const SIGMA: Real = 1.0;
const EPSILON: Real = 1.0;

fn box_20() -> SimulationBox {
    SimulationBox::new(20.0, 20.0, 20.0, 0.0, 0.0, 0.0).unwrap()
}

fn two_particles_at(separation: Real) -> ParticleState {
    ParticleState::new(
        vec![0.0, separation],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![1.0; 2],
        vec![0.0; 2], // charges unused for LJ
        vec![0u32; 2],
        None,
        None,
    )
    .unwrap()
}

fn launch_lj(
    gpu: &GpuContext,
    buffers: &ParticleBuffers,
    output: &mut SlotOutputBuffers,
    sim_box: &SimulationBox,
    params: &LennardJonesParameterTable,
    exclusions: &DeviceExclusionList,
    level: AggregateLevel,
) {
    let n = buffers.particle_count();
    let nl_flat: Vec<u32> = (0..n).flat_map(|i| (0..n).map(move |j| j as u32)).collect();
    let counts = vec![n as u32; n];
    let nl = gpu.device.htod_sync_copy(&nl_flat).unwrap();
    let cnt = gpu.device.htod_sync_copy(&counts).unwrap();
    let mut view = output.view();
    lj_pair_force(
        buffers,
        &mut view,
        sim_box,
        params,
        &exclusions.atom_excl_offsets,
        &exclusions.atom_excl_partners,
        &exclusions.atom_excl_lj_scales,
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
    cutoff: Real,
    r_switch: Real,
    level: AggregateLevel,
    exclusions_input: Option<&[(u32, u32, Real, Real)]>,
) -> PairResult {
    let gpu = init_device().unwrap();
    let state = two_particles_at(separation);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = box_20();
    let params = single_type_lj_table_with_switch(&gpu.device, SIGMA, EPSILON, cutoff, r_switch);
    let excl = match exclusions_input {
        None => empty_exclusions(&gpu.device, 2),
        Some(entries) => exclusions_from_entries(&gpu.device, 2, entries),
    };
    let mut output = SlotOutputBuffers::new(&gpu.device, 2).unwrap();
    launch_lj(&gpu, &buffers, &mut output, &sim_box, &params, &excl, level);

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

fn unswitched_lj_factor(r: Real) -> Real {
    let inv_r2 = 1.0 / (r * r);
    let sigma2 = SIGMA * SIGMA;
    let sr2 = sigma2 * inv_r2;
    let sr6 = sr2 * sr2 * sr2;
    let sr12 = sr6 * sr6;
    24.0 * EPSILON * inv_r2 * (2.0 * sr12 - sr6)
}

fn unswitched_lj_energy(r: Real) -> Real {
    let inv_r2 = 1.0 / (r * r);
    let sigma2 = SIGMA * SIGMA;
    let sr2 = sigma2 * inv_r2;
    let sr6 = sr2 * sr2 * sr2;
    let sr12 = sr6 * sr6;
    4.0 * EPSILON * (sr12 - sr6)
}

// =================================================================
// Switching function.
// =================================================================

// rq-0c4f8da8
#[test]
fn pair_inside_r_switch_sees_unmodified_lj_force() {
    let r = 1.5;
    let res = run_pair(r, 5.0, 4.0, AggregateLevel::ForcesAndScalars, None);
    let expected_fx0 = unswitched_lj_factor(r) * (0.0 - r);
    let rel = ((res.fx0 - expected_fx0).abs() / expected_fx0.abs()) as f64;
    assert!(rel < 1e-5, "fx0 = {} vs expected {} (rel {})", res.fx0, expected_fx0, rel);
    let expected_total_energy = unswitched_lj_energy(r);
    let total_e = (res.e0 + res.e1) as f64;
    let rel_e = (total_e - expected_total_energy as f64).abs() / (expected_total_energy as f64).abs();
    assert!(rel_e < 1e-5, "e0+e1 = {} vs expected {}", total_e, expected_total_energy);
}

// rq-38441c15
#[test]
fn pair_exactly_at_r_switch_sees_unmodified_lj_force() {
    let r = 4.0;
    let res = run_pair(r, 5.0, 4.0, AggregateLevel::ForcesAndScalars, None);
    let expected_fx0 = unswitched_lj_factor(r) * (0.0 - r);
    let rel = ((res.fx0 - expected_fx0).abs() / expected_fx0.abs()) as f64;
    assert!(rel < 1e-4, "fx0 = {} vs expected {} (rel {})", res.fx0, expected_fx0, rel);
}

// rq-f93d278e
#[test]
fn pair_at_r_cut_yields_zero_force_and_energy_when_switch_below_cutoff() {
    let res = run_pair(5.0, 5.0, 4.0, AggregateLevel::ForcesAndScalars, None);
    assert_eq!(res.fx0, 0.0);
    assert_eq!(res.fy0, 0.0);
    assert_eq!(res.fz0, 0.0);
    assert_eq!(res.e0, 0.0);
    assert_eq!(res.e1, 0.0);
    assert_eq!(res.w0, 0.0);
    assert_eq!(res.w1, 0.0);
}

// rq-cb85cf61
#[test]
fn pair_near_r_cut_inside_switching_window_has_force_smaller_than_unmodified() {
    // At r = 4.95 (well into the switching window, near r_cut), the
    // switching function dominates the chain-rule correction and the
    // switched force magnitude is strictly less than the unmodified
    // value. (At smaller r within the window — e.g. r = 4.5 — the
    // chain-rule term can push |F_switched| above |F_unmodified| in
    // the LJ attractive tail, so this assertion is positioned where
    // the switching dominates.)
    let switched = run_pair(4.95, 5.0, 4.0, AggregateLevel::ForcesAndScalars, None);
    let unmodified = run_pair(4.95, 5.0, 5.0, AggregateLevel::ForcesAndScalars, None);
    assert!(
        switched.fx0.abs() < unmodified.fx0.abs(),
        "switched |fx0| = {} not less than unmodified |fx0| = {}",
        switched.fx0.abs(),
        unmodified.fx0.abs()
    );
    assert!(
        switched.fx0.signum() == unmodified.fx0.signum() || switched.fx0 == 0.0,
        "switched fx0 sign differs from unmodified"
    );
}

// rq-ae20ddac
#[test]
fn lj_force_is_c1_continuous_at_r_switch() {
    let f_below = run_pair(4.0 - 1.0e-3, 5.0, 4.0, AggregateLevel::ForcesOnly, None);
    let f_above = run_pair(4.0 + 1.0e-3, 5.0, 4.0, AggregateLevel::ForcesOnly, None);
    let diff = (f_below.fx0 - f_above.fx0).abs();
    let bound = 1.0e-2 * f_below.fx0.abs();
    assert!(
        diff <= bound,
        "discontinuity at r_switch: |Δfx0| = {} > 1% of f_below.fx0 = {}",
        diff,
        bound
    );
}

// rq-e5e3443f
#[test]
fn lj_force_decays_toward_zero_at_r_cut() {
    let f_inside = run_pair(5.0 - 1.0e-3, 5.0, 4.0, AggregateLevel::ForcesOnly, None);
    let unmodified_at_switch = unswitched_lj_factor(4.0) * (0.0 - 4.0);
    let bound = 1.0e-2 * unmodified_at_switch.abs();
    assert!(
        f_inside.fx0.abs() < bound,
        "force just inside cutoff = {} not small (bound {} = 1% of unswitched at r_switch)",
        f_inside.fx0.abs(),
        bound
    );
}

// rq-916f99f3
#[test]
fn switch_equals_cutoff_reproduces_hard_cutoff_inside_cutoff() {
    for &r in &[1.5 as Real, 3.0, 4.5] {
        let res = run_pair(r, 5.0, 5.0, AggregateLevel::ForcesAndScalars, None);
        let expected_fx0 = unswitched_lj_factor(r) * (0.0 - r);
        let rel = ((res.fx0 - expected_fx0).abs() / expected_fx0.abs()) as f64;
        assert!(rel < 1e-4, "at r={}, fx0 = {} vs expected {}", r, res.fx0, expected_fx0);
        let expected_total_e = unswitched_lj_energy(r);
        let rel_e = ((res.e0 + res.e1 - expected_total_e).abs() / expected_total_e.abs()) as f64;
        assert!(rel_e < 1e-4, "at r={}, e0+e1 = {} vs expected {}", r, res.e0 + res.e1, expected_total_e);
    }
}

// rq-531afe39
#[test]
fn pair_beyond_r_cut_yields_zero_regardless_of_r_switch() {
    let res = run_pair(6.0, 5.0, 4.0, AggregateLevel::ForcesAndScalars, None);
    assert_eq!(res.fx0, 0.0);
    assert_eq!(res.fy0, 0.0);
    assert_eq!(res.fz0, 0.0);
    assert_eq!(res.e0, 0.0);
    assert_eq!(res.e1, 0.0);
    assert_eq!(res.w0, 0.0);
    assert_eq!(res.w1, 0.0);
}

// rq-d0f489d7
#[test]
fn pair_virial_inside_switching_window_equals_factor_switched_times_r2() {
    let r = 4.5;
    let res = run_pair(r, 5.0, 4.0, AggregateLevel::ForcesAndScalars, None);
    // factor_switched * r² is the same as (fx * dx + fy * dy + fz * dz)
    // computed in the kernel, summed over both particles via the
    // half-sum convention. We reconstruct factor_switched from the
    // measured force: factor_switched = -fx0 / dx (since dx = 0 - r = -r,
    // so fx0 = factor_switched * (-r) ⇒ factor_switched = -fx0 / r).
    let factor_switched = -res.fx0 / r;
    let expected_virial = factor_switched * (r * r);
    let total_w = (res.w0 + res.w1) as f64;
    let rel = (total_w - expected_virial as f64).abs() / (expected_virial as f64).abs();
    assert!(rel < 1e-4, "w0+w1 = {} vs expected {}", total_w, expected_virial);
}

// rq-ef8013be
#[test]
fn newton_third_law_holds_bitwise_across_switching_window() {
    // 3D non-axis-aligned displacement inside the switching window.
    let gpu = init_device().unwrap();
    let state = ParticleState::new(
        vec![0.0 as Real, 4.5],
        vec![0.0 as Real, 0.4],
        vec![0.0 as Real, -0.2],
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
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = box_20();
    let params = single_type_lj_table_with_switch(&gpu.device, SIGMA, EPSILON, 5.0, 4.0);
    let excl = empty_exclusions(&gpu.device, 2);
    let mut output = SlotOutputBuffers::new(&gpu.device, 2).unwrap();
    launch_lj(&gpu, &buffers, &mut output, &sim_box, &params, &excl, AggregateLevel::ForcesOnly);
    let fx = gpu.device.dtoh_sync_copy(&output.force_x).unwrap();
    let fy = gpu.device.dtoh_sync_copy(&output.force_y).unwrap();
    let fz = gpu.device.dtoh_sync_copy(&output.force_z).unwrap();
    assert_eq!(fx[0].to_bits(), (-fx[1]).to_bits(), "fx Newton's third law");
    assert_eq!(fy[0].to_bits(), (-fy[1]).to_bits(), "fy Newton's third law");
    assert_eq!(fz[0].to_bits(), (-fz[1]).to_bits(), "fz Newton's third law");
}

// =================================================================
// Exclusion scaling.
// =================================================================

#[test]
fn empty_exclusion_list_leaves_pair_force_unchanged() {
    let res = run_pair(1.5, 5.0, 5.0, AggregateLevel::ForcesOnly, None);
    let expected_fx0 = unswitched_lj_factor(1.5) * (0.0 - 1.5);
    let rel = ((res.fx0 - expected_fx0).abs() / expected_fx0.abs()) as f64;
    assert!(rel < 1e-5);
}

#[test]
fn full_exclusion_scale_zero_zeros_lj_contribution() {
    let res = run_pair(
        1.5,
        5.0,
        5.0,
        AggregateLevel::ForcesAndScalars,
        Some(&[(0, 1, 0.0, 1.0)]),
    );
    assert_eq!(res.fx0, 0.0);
    assert_eq!(res.fy0, 0.0);
    assert_eq!(res.fz0, 0.0);
    assert_eq!(res.e0, 0.0);
    assert_eq!(res.e1, 0.0);
}

#[test]
fn half_strength_exclusion_halves_lj_contribution() {
    let scaled = run_pair(
        1.5,
        5.0,
        5.0,
        AggregateLevel::ForcesAndScalars,
        Some(&[(0, 1, 0.5, 1.0)]),
    );
    let unscaled = run_pair(1.5, 5.0, 5.0, AggregateLevel::ForcesAndScalars, None);
    let half_unscaled_fx = 0.5 * unscaled.fx0;
    // The exclusion path applies scale to factor, which is then multiplied
    // by dx — same arithmetic as the kernel's unscaled path × 0.5.
    // The result should be bit-for-bit since 0.5 multiplication is exact.
    assert_eq!(scaled.fx0.to_bits(), half_unscaled_fx.to_bits());
    let half_unscaled_e = 0.5 * unscaled.e0;
    assert_eq!(scaled.e0.to_bits(), half_unscaled_e.to_bits());
    let half_unscaled_w = 0.5 * unscaled.w0;
    assert_eq!(scaled.w0.to_bits(), half_unscaled_w.to_bits());
}

#[test]
fn exclusion_scale_one_equivalent_to_no_exclusion() {
    let explicit = run_pair(
        1.5,
        5.0,
        5.0,
        AggregateLevel::ForcesAndScalars,
        Some(&[(0, 1, 1.0, 1.0)]),
    );
    let implicit = run_pair(1.5, 5.0, 5.0, AggregateLevel::ForcesAndScalars, None);
    assert_eq!(explicit.fx0.to_bits(), implicit.fx0.to_bits());
    assert_eq!(explicit.e0.to_bits(), implicit.e0.to_bits());
    assert_eq!(explicit.w0.to_bits(), implicit.w0.to_bits());
}

// rq-95c2f543
#[test]
fn exclusion_scaling_applies_uniformly_to_force_energy_virial() {
    let scaled = run_pair(
        1.5,
        5.0,
        5.0,
        AggregateLevel::ForcesAndScalars,
        Some(&[(0, 1, 0.5, 1.0)]),
    );
    let unscaled = run_pair(1.5, 5.0, 5.0, AggregateLevel::ForcesAndScalars, None);
    // All three quantities are scaled by the same 0.5 factor.
    assert_eq!(scaled.fx0.to_bits(), (0.5 * unscaled.fx0).to_bits());
    assert_eq!(scaled.e0.to_bits(), (0.5 * unscaled.e0).to_bits());
    assert_eq!(scaled.w0.to_bits(), (0.5 * unscaled.w0).to_bits());
}

// rq-fb55af77
#[test]
fn exclusion_scaling_multiplies_switched_force_energy_virial() {
    // Inside the switching window with switch < cutoff. Scale=0.5
    // applied to the post-switching factor, energy, and virial.
    let r = 4.5;
    let cutoff = 5.0;
    let r_switch = 4.0;
    let scaled = run_pair(
        r,
        cutoff,
        r_switch,
        AggregateLevel::ForcesAndScalars,
        Some(&[(0, 1, 0.5, 1.0)]),
    );
    let unscaled = run_pair(r, cutoff, r_switch, AggregateLevel::ForcesAndScalars, None);
    assert_eq!(scaled.fx0.to_bits(), (0.5 * unscaled.fx0).to_bits());
    assert_eq!(scaled.e0.to_bits(), (0.5 * unscaled.e0).to_bits());
    assert_eq!(scaled.w0.to_bits(), (0.5 * unscaled.w0).to_bits());
}

#[test]
fn exclusion_only_applies_to_listed_pair() {
    // 3-particle system: exclude only pair (0, 1). Particle 0's force
    // should reflect the (0, 2) pair only; particle 2's force should
    // reflect contributions from both (2, 0) and (2, 1) unscaled.
    let gpu = init_device().unwrap();
    let state = ParticleState::new(
        vec![0.0 as Real, 1.5, 3.0],
        vec![0.0; 3],
        vec![0.0; 3],
        vec![0.0; 3],
        vec![0.0; 3],
        vec![0.0; 3],
        vec![1.0; 3],
        vec![0.0; 3],
        vec![0u32; 3],
        None,
        None,
    )
    .unwrap();
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = box_20();
    let params = single_type_lj_table_with_switch(&gpu.device, SIGMA, EPSILON, 5.0, 5.0);
    let excl_full = exclusions_from_entries(&gpu.device, 3, &[(0, 1, 0.0, 1.0)]);
    let excl_empty = empty_exclusions(&gpu.device, 3);
    let mut out_excl = SlotOutputBuffers::new(&gpu.device, 3).unwrap();
    launch_lj(&gpu, &buffers, &mut out_excl, &sim_box, &params, &excl_full, AggregateLevel::ForcesOnly);
    let mut out_full = SlotOutputBuffers::new(&gpu.device, 3).unwrap();
    launch_lj(&gpu, &buffers, &mut out_full, &sim_box, &params, &excl_empty, AggregateLevel::ForcesOnly);

    let fx_excl = gpu.device.dtoh_sync_copy(&out_excl.force_x).unwrap();
    let fx_full = gpu.device.dtoh_sync_copy(&out_full.force_x).unwrap();

    // Particle 2 has no excluded pair → identical force.
    assert_eq!(fx_excl[2].to_bits(), fx_full[2].to_bits(), "particle 2 should be unaffected");
    // Particle 0 lost its contribution from particle 1.
    let f01 = unswitched_lj_factor(1.5) * (0.0 - 1.5);
    let expected_fx0 = fx_full[0] - f01;
    let rel = ((fx_excl[0] - expected_fx0).abs() / fx_excl[0].abs()) as f64;
    assert!(
        rel < 1e-4,
        "particle 0 with (0,1) excluded: fx0 = {} vs expected {} (rel {})",
        fx_excl[0],
        expected_fx0,
        rel
    );
}
