//! SPME real-space pair-force kernel tests.
//!
//! Implements switching-equivalent (cutoff) and exclusion-scaling
//! scenarios from `rqm/forces/spme.md`, plus a closed-form erfc match
//! and Newton's third law on an isolated pair.

mod common;

use heddle_md::forces::{AggregateLevel, DeviceExclusionList};
use heddle_md::gpu::{
    GpuContext, K_COULOMB_F32, ParticleBuffers, SlotOutputBuffers, init_device,
    spme_real_pair_force,
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

fn launch_spme_real(
    gpu: &GpuContext,
    buffers: &ParticleBuffers,
    output: &mut SlotOutputBuffers,
    sim_box: &SimulationBox,
    alpha: Real,
    r_cut_real: Real,
    exclusions: &DeviceExclusionList,
    level: AggregateLevel,
) {
    let n = buffers.particle_count();
    let nl_flat: Vec<u32> = (0..n).flat_map(|i| (0..n).map(move |j| j as u32)).collect();
    let counts = vec![n as u32; n];
    let nl = gpu.device.htod_sync_copy(&nl_flat).unwrap();
    let cnt = gpu.device.htod_sync_copy(&counts).unwrap();
    let mut view = output.view();
    spme_real_pair_force(
        buffers,
        &mut view,
        sim_box,
        alpha,
        r_cut_real,
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
    alpha: Real,
    r_cut_real: Real,
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
    launch_spme_real(&gpu, &buffers, &mut output, &sim_box, alpha, r_cut_real, &excl, level);

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

// Abramowitz-Stegun 7.1.26 rational approximation of erfc for x >= 0,
// max error ~1.5e-7 — sufficient for f32-precision GPU comparison.
fn erfc_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    1.0 - sign * y
}

/// Closed-form factor f such that fx = f * dx (dx = pos_i - pos_j).
fn closed_form_spme_factor(r: Real, q0: Real, q1: Real, alpha: Real) -> Real {
    let r_f = r as f64;
    let alpha_f = alpha as f64;
    let inv_r = 1.0 / r_f;
    let inv_r2 = inv_r * inv_r;
    let ar = alpha_f * r_f;
    let erfc_ar = erfc_f64(ar);
    let gauss = (-(ar * ar)).exp();
    let inv_sqrt_pi = 1.0 / std::f64::consts::PI.sqrt();
    let factor = (K_COULOMB_F32 as f64) * (q0 as f64) * (q1 as f64) * inv_r2
        * (erfc_ar * inv_r + 2.0 * alpha_f * inv_sqrt_pi * gauss);
    factor as Real
}

// =================================================================
// Closed-form match.
// =================================================================

#[test]
fn closed_form_erfc_force_matches_kernel() {
    let r = 4.0 as Real;
    let alpha = 0.4 as Real;
    let q0 = 1.0 as Real;
    let q1 = 1.0 as Real;
    let res = run_pair(r, q0, q1, alpha, 6.0, AggregateLevel::ForcesOnly, None);
    let factor = closed_form_spme_factor(r, q0, q1, alpha);
    let expected_fx0 = factor * (0.0 - r);
    let rel = ((res.fx0 - expected_fx0).abs() / expected_fx0.abs()) as f64;
    assert!(
        rel < 1.0e-5,
        "fx0 = {} vs closed-form {} (rel {})",
        res.fx0,
        expected_fx0,
        rel
    );
}

// =================================================================
// Cutoff behaviour.
// =================================================================

#[test]
fn pair_outside_r_cut_real_contributes_zero() {
    // r_cut_real=3, separation=4 → outside cutoff → kernel writes 0.
    let res = run_pair(4.0, 1.0, -1.0, 0.4, 3.0, AggregateLevel::ForcesAndScalars, None);
    assert_eq!(res.fx0, 0.0);
    assert_eq!(res.fy0, 0.0);
    assert_eq!(res.fz0, 0.0);
    assert_eq!(res.e0, 0.0);
    assert_eq!(res.e1, 0.0);
    assert_eq!(res.w0, 0.0);
    assert_eq!(res.w1, 0.0);
}

// =================================================================
// Newton's third law.
// =================================================================

#[test]
fn newton_third_law_holds_bit_exactly_for_isolated_pair() {
    let res = run_pair(3.0, 1.0, -1.0, 0.4, 6.0, AggregateLevel::ForcesOnly, None);
    assert_eq!(
        res.fx0.to_bits(),
        (-res.fx1).to_bits(),
        "fx0 = {} but -fx1 = {} (Newton's third law must be bit-exact for an isolated pair)",
        res.fx0,
        -res.fx1
    );
}

// =================================================================
// Exclusion scaling.
// =================================================================

#[test]
fn excluded_pair_scale_zero_produces_zero_contribution() {
    let res = run_pair(
        3.0,
        1.0,
        -1.0,
        0.4,
        6.0,
        AggregateLevel::ForcesAndScalars,
        Some(&[(0, 1, 1.0, 0.0)]),
    );
    assert_eq!(res.fx0, 0.0);
    assert_eq!(res.fy0, 0.0);
    assert_eq!(res.fz0, 0.0);
    assert_eq!(res.e0, 0.0);
    assert_eq!(res.e1, 0.0);
    assert_eq!(res.w0, 0.0);
    assert_eq!(res.w1, 0.0);
}

#[test]
fn excluded_pair_scale_half_contributes_half() {
    let scaled = run_pair(
        3.0,
        1.0,
        -1.0,
        0.4,
        6.0,
        AggregateLevel::ForcesAndScalars,
        Some(&[(0, 1, 1.0, 0.5)]),
    );
    let unscaled = run_pair(3.0, 1.0, -1.0, 0.4, 6.0, AggregateLevel::ForcesAndScalars, None);
    assert_eq!(scaled.fx0.to_bits(), (0.5 * unscaled.fx0).to_bits());
    assert_eq!(scaled.e0.to_bits(), (0.5 * unscaled.e0).to_bits());
    assert_eq!(scaled.w0.to_bits(), (0.5 * unscaled.w0).to_bits());
}

#[test]
fn excluded_pair_scale_one_reproduces_unscaled() {
    let explicit = run_pair(
        3.0,
        1.0,
        -1.0,
        0.4,
        6.0,
        AggregateLevel::ForcesAndScalars,
        Some(&[(0, 1, 1.0, 1.0)]),
    );
    let implicit = run_pair(3.0, 1.0, -1.0, 0.4, 6.0, AggregateLevel::ForcesAndScalars, None);
    assert_eq!(explicit.fx0.to_bits(), implicit.fx0.to_bits());
    assert_eq!(explicit.e0.to_bits(), implicit.e0.to_bits());
    assert_eq!(explicit.w0.to_bits(), implicit.w0.to_bits());
}

#[test]
fn exclusion_only_affects_listed_pair() {
    // 3 particles in a row at 0, 2, 4; charges (+1, -1, +1); exclude (0,1)
    // entirely. Particle 2 (not in any exclusion entry) is unaffected;
    // particle 0 loses the (0,1) contribution but keeps (0,2).
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
    let alpha = 0.4 as Real;
    let r_cut_real = 6.0 as Real;
    let excl_partial = exclusions_from_entries(&gpu.device, 3, &[(0, 1, 1.0, 0.0)]);
    let excl_empty = empty_exclusions(&gpu.device, 3);

    let mut out_excl = SlotOutputBuffers::new(&gpu.device, 3).unwrap();
    launch_spme_real(
        &gpu,
        &buffers,
        &mut out_excl,
        &sim_box,
        alpha,
        r_cut_real,
        &excl_partial,
        AggregateLevel::ForcesOnly,
    );
    let mut out_full = SlotOutputBuffers::new(&gpu.device, 3).unwrap();
    launch_spme_real(
        &gpu,
        &buffers,
        &mut out_full,
        &sim_box,
        alpha,
        r_cut_real,
        &excl_empty,
        AggregateLevel::ForcesOnly,
    );

    let fx_excl = gpu.device.dtoh_sync_copy(&out_excl.force_x).unwrap();
    let fx_full = gpu.device.dtoh_sync_copy(&out_full.force_x).unwrap();

    // Particle 2: unaffected by the exclusion entry.
    assert_eq!(
        fx_excl[2].to_bits(),
        fx_full[2].to_bits(),
        "particle 2 must be bit-exactly unaffected (fx_excl={}, fx_full={})",
        fx_excl[2],
        fx_full[2]
    );
    // Particle 0: loses contribution from particle 1. Reconstruct expected
    // force on 0 as the closed-form (0, 2) contribution alone (r=4).
    let f02 = closed_form_spme_factor(4.0, 1.0, 1.0, alpha) * (0.0 - 4.0);
    let rel = ((fx_excl[0] - f02).abs() / f02.abs()) as f64;
    assert!(
        rel < 1.0e-4,
        "particle 0 with (0,1) excluded: fx0 = {} vs (0,2)-only closed form {} (rel {})",
        fx_excl[0],
        f02,
        rel
    );
}
