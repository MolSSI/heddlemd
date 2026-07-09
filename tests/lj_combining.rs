// rq-1adf5954 rq-f5b943c1 — Lennard-Jones combining rules in
// LennardJonesParameterTable::from_config. Each test builds the device
// parameter table from a config and downloads it to check the resolved
// (override-or-combine) per-pair values.

use heddle_md::gpu::{LennardJonesParameterTable, init_device};
use heddle_md::io::config::{
    CombiningRule, LennardJonesConfig, PairInteractionConfig, ParticleTypeConfig,
};
use heddle_md::precision::Real;

fn ptype(name: &str, sigma: f64, epsilon: f64) -> ParticleTypeConfig {
    ParticleTypeConfig {
        name: name.to_string(),
        mass: 1.0,
        sigma: Some(sigma),
        epsilon: Some(epsilon),
        charge: 0.0,
    }
}

fn lb(cutoff: f64) -> LennardJonesConfig {
    LennardJonesConfig {
        combining_rule: CombiningRule::LorentzBerthelot,
        cutoff,
        r_switch: 0.9 * cutoff,
    }
}

fn approx(a: Real, b: Real) -> bool {
    (a - b).abs() <= 1.0e-6 * (1.0 + b.abs())
}

// rq-9c22145d
#[test]
fn from_config_combines_unoverridden_pair_lorentz_berthelot() {
    let gpu = init_device().unwrap();
    let types = [ptype("A", 3.0e-10, 1.0e-21), ptype("B", 4.0e-10, 4.0e-21)];
    let lj = lb(9.0e-10);
    let table =
        LennardJonesParameterTable::from_config(&gpu.device, &types, &[], Some(&lj)).unwrap();
    let n = 2usize;
    let sigma: Vec<Real> = gpu.device.dtoh_sync_copy(&table.sigma).unwrap();
    let epsilon: Vec<Real> = gpu.device.dtoh_sync_copy(&table.epsilon).unwrap();
    let cutoff: Vec<Real> = gpu.device.dtoh_sync_copy(&table.cutoff).unwrap();
    // (A,B) at index 0*n+1: arithmetic mean sigma, geometric mean epsilon.
    assert!(approx(sigma[0 * n + 1], 3.5e-10), "sigma AB = {}", sigma[0 * n + 1]);
    assert!(approx(epsilon[0 * n + 1], 2.0e-21), "eps AB = {}", epsilon[0 * n + 1]);
    assert!(approx(cutoff[0 * n + 1], 9.0e-10));
    // Symmetric table.
    assert!(approx(sigma[1 * n + 0], sigma[0 * n + 1]));
    assert!(approx(epsilon[1 * n + 0], epsilon[0 * n + 1]));
}

// rq-2c0a228b
#[test]
fn from_config_override_beats_combination() {
    let gpu = init_device().unwrap();
    let types = [ptype("A", 3.0e-10, 1.0e-21), ptype("B", 4.0e-10, 4.0e-21)];
    let lj = lb(9.0e-10);
    let over = PairInteractionConfig::lennard_jones(("A", "B"), 2.5e-10, 5.0e-22, 7.0e-10, Some(0.9 * 7.0e-10));
    let over = heddle_md::forces::resolve_lj_pair(&over).unwrap();
    let table =
        LennardJonesParameterTable::from_config(&gpu.device, &types, &[over], Some(&lj)).unwrap();
    let n = 2usize;
    let sigma: Vec<Real> = gpu.device.dtoh_sync_copy(&table.sigma).unwrap();
    let epsilon: Vec<Real> = gpu.device.dtoh_sync_copy(&table.epsilon).unwrap();
    let cutoff: Vec<Real> = gpu.device.dtoh_sync_copy(&table.cutoff).unwrap();
    assert!(approx(sigma[0 * n + 1], 2.5e-10), "override sigma");
    assert!(approx(epsilon[0 * n + 1], 5.0e-22), "override epsilon");
    assert!(approx(cutoff[0 * n + 1], 7.0e-10), "override cutoff");
}

// rq-270597f3
#[test]
fn from_config_self_pair_combines_to_own_params() {
    let gpu = init_device().unwrap();
    let types = [ptype("A", 3.4e-10, 1.65e-21)];
    let lj = lb(9.0e-10);
    let table =
        LennardJonesParameterTable::from_config(&gpu.device, &types, &[], Some(&lj)).unwrap();
    let sigma: Vec<Real> = gpu.device.dtoh_sync_copy(&table.sigma).unwrap();
    let epsilon: Vec<Real> = gpu.device.dtoh_sync_copy(&table.epsilon).unwrap();
    assert!(approx(sigma[0], 3.4e-10));
    assert!(approx(epsilon[0], 1.65e-21));
}

// rq-44481602
#[test]
fn from_config_inert_combined_pair_when_one_epsilon_zero() {
    let gpu = init_device().unwrap();
    let types = [ptype("A", 3.0e-10, 1.0e-21), ptype("H", 1.0e-10, 0.0)];
    let lj = lb(9.0e-10);
    let table =
        LennardJonesParameterTable::from_config(&gpu.device, &types, &[], Some(&lj)).unwrap();
    let n = 2usize;
    let epsilon: Vec<Real> = gpu.device.dtoh_sync_copy(&table.epsilon).unwrap();
    assert_eq!(epsilon[0 * n + 1], 0.0, "sqrt(eps_A * 0) must be exactly 0");
}

// rq-b6723765
#[test]
fn from_config_two_runs_byte_identical() {
    let gpu = init_device().unwrap();
    let types = [ptype("A", 3.0e-10, 1.0e-21), ptype("B", 4.0e-10, 4.0e-21)];
    let lj = lb(9.0e-10);
    let a =
        LennardJonesParameterTable::from_config(&gpu.device, &types, &[], Some(&lj)).unwrap();
    let b =
        LennardJonesParameterTable::from_config(&gpu.device, &types, &[], Some(&lj)).unwrap();
    for (sa, sb) in [
        (&a.sigma, &b.sigma),
        (&a.epsilon, &b.epsilon),
        (&a.cutoff, &b.cutoff),
        (&a.switch, &b.switch),
    ] {
        // All entries are finite (no NaN), so `==` is an exact,
        // bit-for-bit comparison of the two deterministically-built tables.
        let va: Vec<Real> = gpu.device.dtoh_sync_copy(sa).unwrap();
        let vb: Vec<Real> = gpu.device.dtoh_sync_copy(sb).unwrap();
        assert_eq!(va, vb);
    }
}
