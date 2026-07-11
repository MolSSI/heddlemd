//! End-to-end tests: full simulations driven through `run_simulation`,
//! asserting physical and compositional invariants. See
//! `rqm/e2e-testing.md`. The harness lives in `tests/common/e2e.rs`.

mod common;

use common::e2e::{
    assert_energy_drift_bounded, assert_mean_pressure_near, assert_runs_reproducible,
    assert_water_on_manifold, read_last_frame, run_case, Case, SystemBuilder,
};

// ---- Harness ------------------------------------------------------------

#[test] // rq-0a87253a
fn case_creates_unique_empty_directory() {
    let a = Case::new("unique");
    let b = Case::new("unique");
    assert_ne!(a.dir(), b.dir(), "two Cases must have distinct directories");
    for c in [&a, &b] {
        assert!(c.dir().exists(), "case directory must exist");
        assert_eq!(
            std::fs::read_dir(c.dir()).unwrap().count(),
            0,
            "case directory must start empty"
        );
    }
}

#[test] // rq-1fdcfc84
fn system_builder_writes_a_runnable_input_set() {
    let case = Case::new("runnable");
    let cfg = SystemBuilder::argon_lattice(6).n_steps(2).write(&case);
    assert!(case.dir().join("sim.in.xyz").exists());
    assert!(cfg.exists());
    let summary = run_case(&cfg);
    assert_eq!(summary.phases.len(), 1);
    assert_eq!(summary.phases[0].n_steps, 2);
}

// ---- Slot composition (post-force tail) --------------------------------

#[test] // rq-0baf3195
fn constrained_water_with_per_step_barostat_stays_on_manifold() {
    let case = Case::new("water_settle_baro");
    let cfg = SystemBuilder::spce_water()
        .constraints("settle")
        .barostat("c-rescale", 1.0e5)
        .n_steps(200)
        .trajectory_every(1)
        .write(&case);
    run_case(&cfg);
    let (pos, vel) = read_last_frame(&case.dir().join("sim.out.run.xyz"));
    assert_water_on_manifold(&pos, &vel, 1e-4);
}

#[test] // rq-75dc5b88
fn four_slot_run_completes_and_stays_on_manifold() {
    let case = Case::new("water_four_slot");
    let cfg = SystemBuilder::spce_water()
        .thermostat("csvr", 300.0)
        .barostat("c-rescale", 1.0e5)
        .constraints("settle")
        .n_steps(200)
        .trajectory_every(1)
        .write(&case);
    run_case(&cfg);
    let (pos, vel) = read_last_frame(&case.dir().join("sim.out.run.xyz"));
    assert_water_on_manifold(&pos, &vel, 1e-4);
}

// ---- Reproducibility of stochastic / long-range paths ------------------

#[test] // rq-dd4240b5
fn csvr_thermostat_run_is_byte_identical() {
    let builder = SystemBuilder::disordered_lj_liquid(6, 4.0e-10)
        .thermostat("csvr", 120.0)
        .seed(7)
        .n_steps(50);
    assert_runs_reproducible(&builder, 3);
}

#[test] // rq-a9e9f039
fn andersen_thermostat_run_is_byte_identical() {
    let builder = SystemBuilder::disordered_lj_liquid(6, 4.0e-10)
        .thermostat("andersen", 120.0)
        .seed(11)
        .n_steps(50);
    assert_runs_reproducible(&builder, 3);
}

#[test] // rq-9ef78757
fn c_rescale_barostat_run_is_byte_identical() {
    let builder = SystemBuilder::disordered_lj_liquid(6, 4.0e-10)
        .barostat("c-rescale", 1.0e5)
        .seed(13)
        .n_steps(50);
    assert_runs_reproducible(&builder, 3);
}

#[test] // rq-86bf63c3
fn spme_electrostatics_run_is_byte_identical() {
    let builder = SystemBuilder::ionic_lattice(6).with_spme().seed(17).n_steps(50);
    assert_runs_reproducible(&builder, 3);
}

#[test] // rq-87307e49
fn settle_constrained_run_is_byte_identical() {
    let builder = SystemBuilder::spce_water().constraints("settle").seed(3).n_steps(50);
    assert_runs_reproducible(&builder, 3);
}

// ---- Energy conservation -----------------------------------------------

#[test] // rq-d2cd8351
fn nve_velocity_verlet_conserves_total_energy() {
    let case = Case::new("nve_energy");
    let cfg = SystemBuilder::argon_lattice(6)
        .n_steps(400)
        .log_every(10)
        .write(&case);
    let summary = run_case(&cfg);
    let phase = &summary.phases[0];
    // Bound the drift slope relative to the energy scale over the run.
    let e0 = phase.physics.first().unwrap().total_energy.abs().max(1e-30);
    let span = phase.physics.last().unwrap().time - phase.physics.first().unwrap().time;
    // Allow up to 5% of |E0| of drift across the whole trajectory.
    let max_slope = 0.05 * e0 / span.max(1e-30);
    assert_energy_drift_bounded(phase, max_slope);
}

#[test] // rq-4fcc97ab
fn nve_run_with_spme_conserves_total_energy() {
    let case = Case::new("nve_spme_energy");
    let cfg = SystemBuilder::ionic_lattice(6)
        .with_spme()
        .n_steps(400)
        .log_every(10)
        .write(&case);
    let summary = run_case(&cfg);
    let phase = &summary.phases[0];
    let e0 = phase.physics.first().unwrap().total_energy.abs().max(1e-30);
    let span = phase.physics.last().unwrap().time - phase.physics.first().unwrap().time;
    let max_slope = 0.05 * e0 / span.max(1e-30);
    assert_energy_drift_bounded(phase, max_slope);
}

// ---- Pressure control --------------------------------------------------

#[test] // rq-b02a9070
fn npt_run_drives_mean_pressure_to_target() {
    let case = Case::new("npt_pressure");
    let target = 1.0e5;
    let cfg = SystemBuilder::disordered_lj_liquid(6, 3.8e-10)
        .barostat("c-rescale", target)
        .n_steps(600)
        .log_every(10)
        .write(&case);
    let summary = run_case(&cfg);
    // Barostat pressure control is noisy; assert the mean is the right
    // order of magnitude and sign rather than a tight tolerance.
    assert_mean_pressure_near(&summary.phases[0], target, 5.0);
}

#[test] // rq-5214e3b3
fn npt_run_box_volume_responds_to_barostat() {
    let case = Case::new("npt_volume");
    let cfg = SystemBuilder::disordered_lj_liquid(6, 3.8e-10)
        .barostat("c-rescale", 1.0e8)
        .n_steps(400)
        .log_every(10)
        .write(&case);
    let summary = run_case(&cfg);
    let phys = &summary.phases[0].physics;
    let v0 = phys.first().unwrap().volume;
    let vlast = phys.last().unwrap().volume;
    assert!(
        (vlast - v0).abs() > 0.0,
        "box volume did not respond to the barostat (v0={v0:e}, vlast={vlast:e})"
    );
}

#[test] // rq-5a00037b
fn npt_run_with_spme_runs_stably_and_box_responds() {
    let case = Case::new("npt_spme");
    let cfg = SystemBuilder::ionic_lattice(6)
        .with_spme()
        .barostat("c-rescale", 1.0e8)
        .n_steps(200)
        .log_every(10)
        .write(&case);
    let summary = run_case(&cfg);
    let phys = &summary.phases[0].physics;
    // The reciprocal pipeline tracks the box the barostat mutates: the
    // run completes, the volume responds, and no sample goes non-finite.
    let v0 = phys.first().unwrap().volume;
    let vlast = phys.last().unwrap().volume;
    assert!(
        (vlast - v0).abs() > 0.0,
        "box volume did not respond under SPME (v0={v0:e}, vlast={vlast:e})"
    );
    assert!(
        phys.iter().all(|p| p.total_energy.is_finite() && p.pressure.is_finite()),
        "a non-finite energy or pressure sample appeared during the SPME NPT run"
    );
}
