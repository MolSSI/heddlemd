//! rq-1a9f7f9c — SPME excluded-pair reciprocal correction.
//!
//! See `rqm/forces/spme.md`, *Excluded-pair correction*.
//!
//! The reciprocal-space sum is a mesh sum over the charge density: it cannot be
//! told which pairs are excluded, so it carries the smooth `erf(a*r)/r` part of
//! EVERY pair at full strength, modified pairs included. Scaling only the
//! real-space term leaves a modified pair at `s*erfc(a*r)/r + erf(a*r)/r`, which
//! is not `s/r`. `U_excl = (s - 1) * k_C * q_i q_j * erf(a*r)/r` supplies the
//! missing share, and the four Ewald terms then sum to `s/r` exactly.

mod common;

use heddle_md::forces::{
    AggregateLevel, AngleList, BondList, DihedralList, ExclusionList, ForceField,
    PotentialRegistry,
};
use heddle_md::gpu::{GpuContext, ParticleBuffers, init_device};
use heddle_md::io::config::{
    NeighborListConfig, PairInteractionConfig, ParticleTypeConfig, SpmeConfig,
};
use heddle_md::pbc::SimulationBox;
use heddle_md::precision::Real;
use heddle_md::state::ParticleState;
use heddle_md::timings::Timings;

use common::host_exclusions_from_entries;

/// Atomic units throughout: `k_C = 1`, lengths in Bohr.
const K_C: f64 = 1.0;
/// Wide enough that the two charges' interaction with their own periodic images
/// is small, and (crucially) `s`-independent — see `net_pair_energy`.
const BOX: Real = 40.0;

fn charged_type(name: &'static str, charge: f64) -> ParticleTypeConfig {
    ParticleTypeConfig {
        name: name.to_string(),
        mass: 1.0,
        sigma: None,
        epsilon: None,
        charge,
    }
}

/// A zero-epsilon Lennard-Jones table. It contributes exactly nothing, and
/// exists only so that some slot reports a cutoff: the shared neighbour list —
/// and therefore the correction pass — is built only when one does, and the
/// excluded-pair fragment reports none by design.
fn null_lj(cutoff: f64) -> Vec<PairInteractionConfig> {
    vec![
        PairInteractionConfig::lennard_jones(("P", "P"), 1.0, 0.0, cutoff, Some(cutoff)),
        PairInteractionConfig::lennard_jones(("M", "M"), 1.0, 0.0, cutoff, Some(cutoff)),
        PairInteractionConfig::lennard_jones(("P", "M"), 1.0, 0.0, cutoff, Some(cutoff)),
    ]
}

struct Sample {
    energy: f64,
    virial: f64,
    force_x0: f64,
}

/// Evaluate a two-charge system (+q at the origin, -q at `r` along x) carrying a
/// single exclusion of Coulomb scale `scale_coul`, under SPME.
fn eval(
    gpu: &GpuContext,
    r: f64,
    q: f64,
    alpha: f64,
    scale_coul: Real,
    n_exclusions: usize,
    grid: u32,
    cutoff: f64,
) -> Sample {
    let sim_box = SimulationBox::new(&gpu.device, BOX, BOX, BOX, 0.0, 0.0, 0.0).unwrap();
    let charges = vec![q as Real, -q as Real];
    let spme = SpmeConfig {
        alpha,
        r_cut_real: cutoff,
        grid: [grid, grid, grid],
        spline_order: 5,
    };
    let exclusions = if n_exclusions == 0 {
        ExclusionList::empty(2)
    } else {
        host_exclusions_from_entries(2, &[(0, 1, 1.0, scale_coul)])
    };
    let mut ff = ForceField::new(
        &PotentialRegistry::with_builtins(),
        gpu,
        2,
        &sim_box,
        &[charged_type("P", q), charged_type("M", -q)],
        &null_lj(cutoff),
        &[],
        &[],
        &[],
        Some(&spme),
        &charges,
        &BondList::empty(2),
        &AngleList::empty(0),
        &DihedralList::empty(0),
        &exclusions,
        &NeighborListConfig::AllPairs,
    )
    .expect("build SPME ForceField");

    let state = ParticleState::new(
        vec![0.0, r as Real],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 0.0],
        vec![1.0, 1.0],
        charges.clone(),
        vec![0u32, 1u32],
        None,
        None,
    )
    .expect("valid two-particle state");
    let mut buffers = ParticleBuffers::new(gpu, &state).unwrap();
    let mut timings = Timings::new(gpu).unwrap();
    ff.step(
        &mut buffers,
        &sim_box,
        &mut timings,
        AggregateLevel::ForcesAndScalars,
    )
    .expect("force evaluation");

    let e: Vec<Real> = gpu
        .device
        .dtoh_sync_copy(&buffers.potential_energies)
        .unwrap();
    let w: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.virials).unwrap();
    let fx: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.forces_x).unwrap();
    Sample {
        energy: e.iter().map(|&x| x as f64).sum(),
        virial: w.iter().map(|&x| x as f64).sum(),
        force_x0: fx[0] as f64,
    }
}

/// The pair's OWN Coulomb energy, with the (scale-independent) interaction
/// between the charges and their periodic images removed.
///
/// `U_total(s) = s * k_C * q_i q_j / r + U_images`, and `U_images` does not
/// depend on `s` — an image is not an excluded partner. So the difference
/// `U(s=1) - U(s=0)` is exactly the pair's own Coulomb energy at full strength,
/// free of any lattice-sum residue. That is what makes this checkable to f32
/// precision in a finite box.
fn net_pair_energy(gpu: &GpuContext, r: f64, q: f64, alpha: f64, s: Real) -> f64 {
    // The baseline is the FULLY EXCLUDED system, not the s = 0 one. Those are the
    // same thing only if the correction is present; taking `U(s) - U(0)` as the
    // definition would make the s = 0 case `U(0) - U(0) = 0`, which is vacuously
    // true whether or not the correction exists.
    let at_s = eval(gpu, r, q, alpha, s, 1, 64, 5.0);
    let excluded = eval(gpu, r, q, alpha, 0.0, 1, 64, 5.0);
    at_s.energy - excluded.energy
}

// ---- Alpha invariance ---------------------------------------------------
//
// The Ewald splitting is an identity: alpha partitions work between real and
// reciprocal space and cannot move the answer. Omitting U_excl breaks that,
// which makes it the sharpest available check on the correction.

#[test] // rq-c811e838
fn total_electrostatic_energy_is_invariant_under_the_splitting_parameter() {
    let gpu = init_device().unwrap();
    // A fully excluded pair. Its own Coulomb interaction is removed entirely, so
    // what remains is the (alpha-independent) lattice sum over the periodic
    // images. Any alpha dependence here is an uncancelled reciprocal-space term.
    let vals: Vec<f64> = [0.30, 0.35, 0.40, 0.45]
        .iter()
        .map(|&a| eval(&gpu, 2.0, 1.0, a, 0.0, 1, 64, 5.0).energy)
        .collect();
    let spread = vals.iter().cloned().fold(f64::MIN, f64::max)
        - vals.iter().cloned().fold(f64::MAX, f64::min);
    // The bare Coulomb of this pair is 1/2 = 0.5 Ha; the residue must be far
    // below it and must not track alpha.
    assert!(
        spread < 2.0e-3,
        "total energy moved with alpha: {vals:?} (spread {spread:.3e}). \
         The reciprocal sum carries erf(a*r)/r for every pair including the \
         excluded ones; without U_excl that share is never removed and alpha \
         stops cancelling."
    );
}

#[test] // rq-064d2401
fn total_scalar_virial_is_invariant_under_the_splitting_parameter() {
    let gpu = init_device().unwrap();
    let vals: Vec<f64> = [0.30, 0.35, 0.40, 0.45]
        .iter()
        .map(|&a| eval(&gpu, 2.0, 1.0, a, 0.0, 1, 64, 5.0).virial)
        .collect();
    let spread = vals.iter().cloned().fold(f64::MIN, f64::max)
        - vals.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        spread < 2.0e-3,
        "total virial moved with alpha: {vals:?} (spread {spread:.3e}). \
         The correction's virial is what makes the pressure alpha-independent."
    );
}

// ---- The Ewald sum reproduces the bare Coulomb --------------------------

#[test] // rq-a37db23a
fn a_fully_excluded_pair_has_no_net_coulomb_interaction() {
    let gpu = init_device().unwrap();
    let r = 2.0;
    let bare = K_C * (1.0 * -1.0) / r;
    // Measured against a system with NO exclusion entry, which keeps the pair's
    // full Coulomb interaction. Removing the exclusion must remove exactly the
    // bare Coulomb energy -- no more, no less.
    //
    // Without the correction the excluded pair retains the reciprocal sum's
    // erf(a*r)/r share, so this difference collapses to erfc(a*r)/r -- at
    // alpha = 0.35, r = 2 that is erfc(0.7) = 0.32 of the bare value.
    let unexcluded = eval(&gpu, r, 1.0, 0.35, 0.0, 0, 64, 5.0);
    let excluded = eval(&gpu, r, 1.0, 0.35, 0.0, 1, 64, 5.0);
    let removed = unexcluded.energy - excluded.energy;
    assert!(
        (removed - bare).abs() < 2e-3 * bare.abs(),
        "excluding the pair removed {removed:.6e} of Coulomb energy, expected          exactly the bare {bare:.6e}. A short-fall means the reciprocal sum's          erf(a*r)/r share was left behind."
    );
}

#[test] // rq-e5dd8842
fn a_scaled_pair_retains_exactly_its_scaled_coulomb_interaction() {
    let gpu = init_device().unwrap();
    let r = 2.0;
    let bare = K_C * (1.0 * -1.0) / r;
    // The 1-4 Coulomb scale of the OPLS/AMBER convention.
    for s in [0.0_f64, 0.5, 0.8333, 1.0] {
        let net = net_pair_energy(&gpu, r, 1.0, 0.35, s as Real);
        let want = s * bare;
        assert!(
            (net - want).abs() < 2e-3 * bare.abs(),
            "scale {s}: net Coulomb {net:.6e}, expected {want:.6e} (= s * {bare:.6e}). \
             The four Ewald terms must sum to s/r exactly."
        );
    }
}

// ---- The correction is cutoff-free --------------------------------------

#[test] // rq-ba1b0ef5
fn a_modified_pair_beyond_the_cutoff_is_still_corrected() {
    let gpu = init_device().unwrap();
    // The pair sits at r = 8, well beyond the 5.0 real-space cutoff. The mesh
    // sum has no cutoff, so it still carries erf(a*r)/r for this pair at full
    // strength -- and the correction must still remove it, or a residue of
    // (1 - s) * k_C * q_i q_j / r survives: long-ranged, silent, and growing
    // with the charges.
    let r = 8.0;
    let bare = K_C * (1.0 * -1.0) / r;
    let full = eval(&gpu, r, 1.0, 0.35, 1.0, 1, 64, 5.0);
    let zero = eval(&gpu, r, 1.0, 0.35, 0.0, 1, 64, 5.0);
    let net = full.energy - zero.energy;
    assert!(
        (net - bare).abs() < 5e-3 * bare.abs(),
        "beyond the cutoff the pair's net Coulomb is {net:.6e}, expected {bare:.6e}. \
         A cutoff-masked correction would leave the excluded pair's reciprocal \
         share in place."
    );
}

// ---- Degenerate cases ---------------------------------------------------

#[test] // rq-ee1ce29c
fn coincident_excluded_atoms_produce_no_non_finite_value() {
    let gpu = init_device().unwrap();
    // erf(a*r)/r is finite as r -> 0 (it tends to 2a/sqrt(pi)) and the force
    // tends to zero, but the direct expressions are 0/0. The fragment branches
    // to the closed-form limit below EXCL_MIN_ALPHA_R, and does so BEFORE
    // consuming inv_r, so no infinity can reach the outputs.
    let s = eval(&gpu, 0.0, 1.0, 0.35, 0.0, 1, 64, 5.0);
    assert!(
        s.energy.is_finite() && s.virial.is_finite() && s.force_x0.is_finite(),
        "coincident excluded atoms produced a non-finite value: \
         energy {}, virial {}, force {}",
        s.energy,
        s.virial,
        s.force_x0
    );
    assert!(
        s.force_x0.abs() < 1e-6,
        "the force between coincident atoms must vanish, got {}",
        s.force_x0
    );
}

#[test] // rq-a7adc7c9
fn a_system_with_no_exclusions_carries_no_correction() {
    let gpu = init_device().unwrap();
    let r = 2.0;
    // No modified pairs at all: the correction pass walks an empty list, and the
    // pair keeps its full Coulomb interaction.
    let none = eval(&gpu, r, 1.0, 0.35, 0.0, 0, 64, 5.0);
    let unscaled = eval(&gpu, r, 1.0, 0.35, 1.0, 1, 64, 5.0);
    assert!(
        (none.energy - unscaled.energy).abs() < 1e-4,
        "a system with no exclusion entries ({}) must match one whose single \
         entry is unscaled ({})",
        none.energy,
        unscaled.energy
    );
}

#[test] // rq-3fea9eed
fn uncharged_atoms_carry_no_correction() {
    let gpu = init_device().unwrap();
    // qq = 0, so every Ewald term vanishes and so must the correction.
    let s = eval(&gpu, 2.0, 0.0, 0.35, 0.0, 1, 64, 5.0);
    assert!(
        s.energy.abs() < 1e-6 && s.force_x0.abs() < 1e-6,
        "uncharged excluded pair produced energy {} and force {}",
        s.energy,
        s.force_x0
    );
}

#[test] // rq-95bdc1a2
fn the_correction_is_reproducible_run_to_run() {
    let gpu = init_device().unwrap();
    let a = eval(&gpu, 2.0, 1.0, 0.35, 0.0, 1, 64, 5.0);
    let b = eval(&gpu, 2.0, 1.0, 0.35, 0.0, 1, 64, 5.0);
    assert_eq!(a.energy.to_bits(), b.energy.to_bits());
    assert_eq!(a.virial.to_bits(), b.virial.to_bits());
    assert_eq!(a.force_x0.to_bits(), b.force_x0.to_bits());
}
