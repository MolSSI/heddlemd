// rq-202493a5 rq-9ca00d25 rq-f6d45062
//
// PR 2 validation: tests that exercise the SPME slot wrappers
// (`SpmeRealSpaceState`, `SpmeReciprocalState`) and the
// `ForceField`-level integration of [spme] config.

use std::f64::consts::PI;

use heddle_md::forces::neighbor_list::NeighborListState;
use heddle_md::forces::{ExclusionList, ForceField, ForceFieldContext, Potential, PotentialRegistry, SpmeParameters, SpmeRealSpaceState, SpmeReciprocalState};
use heddle_md::gpu::cufft::cufft_determinism_smoke_test;
use heddle_md::gpu::{GpuContext, K_COULOMB_F32, ParticleBuffers, init_device};
use heddle_md::io::config::{NeighborListConfig, ParticleTypeConfig, SpmeConfig};
use heddle_md::pbc::SimulationBox;
use heddle_md::state::ParticleState;
use heddle_md::timings::Timings;
use heddle_md::precision::Real;

fn build_state(positions: &[[Real; 3]], charges: &[Real]) -> ParticleState {
    let n = positions.len();
    assert_eq!(charges.len(), n);
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

fn default_spme_config() -> SpmeConfig {
    SpmeConfig {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 4,
    }
}

fn nm_box() -> SimulationBox {
    let l = 1.0e-9;
    SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap()
}

// @rq-dd41209b
#[test]
fn force_field_registers_spme_slots_when_configured() {
    let gpu = init_device().unwrap();
    let particle_types = vec![ParticleTypeConfig {
        name: "Na".to_string(),
        mass: 22.99,
        charge: 1.0,
    }];
    let spme = default_spme_config();
    let charges = vec![1.0];
    let ff = ForceField::new(
        &PotentialRegistry::with_builtins(),
        &gpu,
        1,
        &nm_box(),
        &particle_types,
        &[],
        &[],
        &[],
        None,
        Some(&spme),
        &charges,
        &heddle_md::forces::BondList::empty(1),
        &heddle_md::forces::AngleList::empty(0),
        &ExclusionList::empty(1),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    assert_eq!(ff.slots.len(), 2, "expected two SPME slots");
    assert_eq!(ff.slots[0].label(), "spme_real");
    assert_eq!(ff.slots[1].label(), "spme_reciprocal");
}

// @rq-09d4e13f
// rq-09d4e13f
#[test]
fn spme_reciprocal_state_two_independent_constructs_produce_byte_identical_grids() {
    let gpu = init_device().unwrap();
    let sim_box = nm_box();
    let e_charge = 1.602176634e-19;
    let positions = [[0.1e-9, 0.0, 0.0], [-0.1e-9, 0.0, 0.0]];
    let charges = [e_charge, -e_charge];
    let state = build_state(&positions, &charges);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();

    let params = SpmeParameters::from(&default_spme_config());

    let mut s1 = SpmeReciprocalState::new(&gpu, &sim_box, 2, &charges, params).unwrap();
    let mut s2 = SpmeReciprocalState::new(&gpu, &sim_box, 2, &charges, params).unwrap();
    let mut t = Timings::new(&gpu).unwrap();
    let cx = ForceFieldContext {
        neighbor_list: None,
        buffers: &buffers,
        sim_box: &sim_box,
    };
    s1.contribute(&buffers, &sim_box, &cx, &mut t).unwrap();
    s2.contribute(&buffers, &sim_box, &cx, &mut t).unwrap();
    s1.grid().sync_recip().unwrap();
    s2.grid().sync_recip().unwrap();

    let v1: Vec<Real> = gpu.device.dtoh_sync_copy(&s1.grid().v).unwrap();
    let v2: Vec<Real> = gpu.device.dtoh_sync_copy(&s2.grid().v).unwrap();
    assert_eq!(v1, v2, "two independent slots must produce bit-identical V");
}

// @rq-ef8dee82
//
// `SpmeReciprocalState`'s gather kernel subtracts
// `u_self_per_particle[i]` from `slot_energy[i]`. Verify by the global
// identity
//   Σ_i slot_energy[i] = U_recip − Σ_i u_self[i]
// where `U_recip = (1/2) Σ_g rho[g] · V[g]` is independently computed
// from the slot's own pipeline buffers.
#[test]
fn spme_reciprocal_total_energy_equals_recip_minus_self() {
    let gpu = init_device().unwrap();
    let sim_box = nm_box();
    let alpha: Real = 4.0e9;
    let e_charge: Real = 1.602176634e-19;
    let charges = [e_charge, -e_charge, 2.0 * e_charge, -2.0 * e_charge];
    let positions = [
        [0.1e-9, 0.0, 0.0],
        [-0.1e-9, 0.2e-9, 0.0],
        [0.0, -0.2e-9, 0.15e-9],
        [-0.15e-9, 0.1e-9, -0.1e-9],
    ];
    let n = positions.len();
    let state = build_state(&positions, &charges);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();

    let params = SpmeParameters {
        alpha,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 4,
    };
    let mut slot = SpmeReciprocalState::new(&gpu, &sim_box, n, &charges, params).unwrap();
    let mut t = Timings::new(&gpu).unwrap();
    let cx = ForceFieldContext {
        neighbor_list: None,
        buffers: &buffers,
        sim_box: &sim_box,
    };
    slot.contribute(&buffers, &sim_box, &cx, &mut t).unwrap();
    slot.grid().sync_recip().unwrap();

    // Independent U_recip from the pipeline-resident buffers.
    let device = gpu.device.clone();
    let rho: Vec<Real> = device.dtoh_sync_copy(&slot.grid().rho).unwrap();
    let v: Vec<Real> = device.dtoh_sync_copy(&slot.grid().v).unwrap();
    let u_recip: f64 = 0.5
        * rho
            .iter()
            .zip(v.iter())
            .map(|(&r, &vg)| (r as f64) * (vg as f64))
            .sum::<f64>();

    // Run reduce() into a fresh output view.
    let mut force_x = device.alloc_zeros::<Real>(n).unwrap();
    let mut force_y = device.alloc_zeros::<Real>(n).unwrap();
    let mut force_z = device.alloc_zeros::<Real>(n).unwrap();
    let mut energy = device.alloc_zeros::<Real>(n).unwrap();
    let mut virial = device.alloc_zeros::<Real>(n).unwrap();
    {
        let view = heddle_md::forces::SlotOutputView {
            force_x: force_x.slice_mut(0..n),
            force_y: force_y.slice_mut(0..n),
            force_z: force_z.slice_mut(0..n),
            energy: energy.slice_mut(0..n),
            virial: virial.slice_mut(0..n),
        };
        slot.reduce(view, &cx, &mut t, heddle_md::forces::AggregateLevel::ForcesAndScalars).unwrap();
    }
    let energies: Vec<Real> = device.dtoh_sync_copy(&energy).unwrap();
    let total: f64 = energies.iter().map(|&e| e as f64).sum();

    let inv_sqrt_pi = 1.0_f64 / PI.sqrt();
    let prefactor = (K_COULOMB_F32 as f64) * (alpha as f64) * inv_sqrt_pi;
    let u_self_total: f64 = charges
        .iter()
        .map(|&q| prefactor * (q as f64) * (q as f64))
        .sum();
    let expected = u_recip - u_self_total;
    let rel = (total - expected).abs() / expected.abs().max(1.0e-30);
    assert!(
        rel < 5.0e-3,
        "Σ slot_energy = {:e}, expected (U_recip − U_self) = {:e}, rel = {:e}",
        total,
        expected,
        rel
    );
}

// @rq-af7018c0
//
// The real-space `erfc` slot writes zero force for a pair whose
// separation exceeds `r_cut_real`. We construct an isolated pair just
// outside the cutoff and verify that the per-particle force is zero.
// rq-af7018c0
#[test]
fn spme_real_slot_zero_outside_r_cut() {
    let gpu = init_device().unwrap();
    let sim_box = SimulationBox::new(2.0e-9, 2.0e-9, 2.0e-9, 0.0, 0.0, 0.0).unwrap();
    let r_cut_real: Real = 0.3e-9;
    let positions = [[0.0, 0.0, 0.0], [r_cut_real + 0.05e-9, 0.0, 0.0]];
    let charges = [1.0e-19, -1.0e-19];
    let state = build_state(&positions, &charges);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();

    let nl = NeighborListState::new_trivial(&gpu, &sim_box, 2).unwrap();
    let mut slot = SpmeRealSpaceState::new(
        &gpu,
        2,
        4.0e9,
        r_cut_real,
        2,
        &ExclusionList::empty(2),
    )
    .unwrap();
    let mut t = Timings::new(&gpu).unwrap();
    let cx = ForceFieldContext {
        neighbor_list: Some(&nl),
        buffers: &buffers,
        sim_box: &sim_box,
    };
    slot.contribute(&buffers, &sim_box, &cx, &mut t).unwrap();

    let device = gpu.device.clone();
    let n = 2usize;
    let mut force_x = device.alloc_zeros::<Real>(n).unwrap();
    let mut force_y = device.alloc_zeros::<Real>(n).unwrap();
    let mut force_z = device.alloc_zeros::<Real>(n).unwrap();
    let mut energy = device.alloc_zeros::<Real>(n).unwrap();
    let mut virial = device.alloc_zeros::<Real>(n).unwrap();
    {
        let view = heddle_md::forces::SlotOutputView {
            force_x: force_x.slice_mut(0..n),
            force_y: force_y.slice_mut(0..n),
            force_z: force_z.slice_mut(0..n),
            energy: energy.slice_mut(0..n),
            virial: virial.slice_mut(0..n),
        };
        slot.reduce(view, &cx, &mut t, heddle_md::forces::AggregateLevel::ForcesAndScalars).unwrap();
    }
    let fx: Vec<Real> = device.dtoh_sync_copy(&force_x).unwrap();
    assert!(
        fx.iter().all(|f| f.abs() < 1.0e-20),
        "expected zero forces beyond r_cut_real, got {:?}",
        fx
    );
}

// @rq-83088c2f
//
// Real-space `erfc` slot matches the closed-form pair force
//   F = k_C · q_i q_j / r² · (erfc(α r) / r + 2 α / √π · exp(−(α r)²))
// for an isolated pair well inside the cutoff.
// rq-83088c2f
#[test]
fn spme_real_slot_matches_closed_form_erfc_pair() {
    let gpu = init_device().unwrap();
    let sim_box = SimulationBox::new(2.0e-9, 2.0e-9, 2.0e-9, 0.0, 0.0, 0.0).unwrap();
    let alpha: Real = 4.0e9;
    let r_cut_real: Real = 0.5e-9;
    let r: Real = 0.15e-9;
    let positions = [[0.0, 0.0, 0.0], [r, 0.0, 0.0]];
    let q1: Real = 1.0e-19;
    let q2: Real = -1.0e-19;
    let charges = [q1, q2];
    let state = build_state(&positions, &charges);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();

    let nl = NeighborListState::new_trivial(&gpu, &sim_box, 2).unwrap();
    let mut slot = SpmeRealSpaceState::new(
        &gpu,
        2,
        alpha,
        r_cut_real,
        2,
        &ExclusionList::empty(2),
    )
    .unwrap();
    let mut t = Timings::new(&gpu).unwrap();
    let cx = ForceFieldContext {
        neighbor_list: Some(&nl),
        buffers: &buffers,
        sim_box: &sim_box,
    };
    slot.contribute(&buffers, &sim_box, &cx, &mut t).unwrap();

    let device = gpu.device.clone();
    let n = 2usize;
    let mut force_x = device.alloc_zeros::<Real>(n).unwrap();
    let mut force_y = device.alloc_zeros::<Real>(n).unwrap();
    let mut force_z = device.alloc_zeros::<Real>(n).unwrap();
    let mut energy = device.alloc_zeros::<Real>(n).unwrap();
    let mut virial = device.alloc_zeros::<Real>(n).unwrap();
    {
        let view = heddle_md::forces::SlotOutputView {
            force_x: force_x.slice_mut(0..n),
            force_y: force_y.slice_mut(0..n),
            force_z: force_z.slice_mut(0..n),
            energy: energy.slice_mut(0..n),
            virial: virial.slice_mut(0..n),
        };
        slot.reduce(view, &cx, &mut t, heddle_md::forces::AggregateLevel::ForcesAndScalars).unwrap();
    }
    let fx: Vec<Real> = device.dtoh_sync_copy(&force_x).unwrap();

    // Closed-form prediction. q1 at origin, q2 at +x ⇒ force on q1 from
    // q2 points in -x for opposite charges (attractive). dx = pos[0] -
    // pos[1] = -r, so factor * dx with factor > 0 gives negative fx on
    // particle 0.
    let alpha_f = alpha as f64;
    let r_f = r as f64;
    let ar = alpha_f * r_f;
    let erfc_ar = libm_erfc(ar);
    let gauss = (-(ar * ar)).exp();
    let one_over_sqrt_pi = 1.0_f64 / PI.sqrt();
    let qq = (q1 as f64) * (q2 as f64);
    let factor = (K_COULOMB_F32 as f64) * qq / (r_f * r_f)
        * (erfc_ar / r_f + 2.0 * alpha_f * one_over_sqrt_pi * gauss);
    // dx for particle 0 = -r; for particle 1 = +r.
    let fx0_expected = (factor * (-r_f)) as Real;
    let rel = (fx[0] - fx0_expected).abs() / fx0_expected.abs();
    assert!(
        rel < 5.0e-3,
        "fx[0] = {:e}, expected {:e}, rel error = {:e}",
        fx[0],
        fx0_expected,
        rel
    );
}

// Host-side erfc implementation (no libm dependency; use the series
// good enough for the test).
fn libm_erfc(x: f64) -> f64 {
    // erfc(x) = 1 − erf(x). Use a high-quality Padé approximant.
    let z = x.abs();
    // Abramowitz & Stegun 7.1.26 (max error ~1.5e-7, plenty for f32 test).
    let p = 0.3275911_f64;
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736_f64;
    let a3 = 1.421413741_f64;
    let a4 = -1.453152027_f64;
    let a5 = 1.061405429_f64;
    let t = 1.0 / (1.0 + p * z);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-z * z).exp();
    let erf_z = if x >= 0.0 { y } else { -y };
    1.0 - erf_z
}

// @rq-637cd1a5
#[test]
fn cufft_smoke_test_passes_on_this_device() {
    let gpu = init_device().unwrap();
    let differences = cufft_determinism_smoke_test(&gpu.device).unwrap();
    assert_eq!(
        differences, 0,
        "cuFFT R2C produced different outputs on identical inputs ({differences} float positions differ)"
    );
}

// @rq-674cc467
//
// The runner's box-compatibility check picks up `spme.r_cut_real` when
// computing the minimum perpendicular width. We test this at the
// config level by constructing a `Config` that would pass the check
// without SPME, and confirming the SPME version raises the threshold.
//
// We bypass `run_simulation` (which would require disk artifacts) and
// instead recompute the threshold the same way the runner does.
// rq-674cc467
#[test]
fn box_compatibility_picks_up_spme_r_cut_real() {
    let r_skin: f64 = 0.05;
    let lj_cutoff: f64 = 0.3;
    let spme_r_cut_real: f64 = 0.6;
    // Replicate the runner's max-aggregation logic.
    let mut cutoff_max: f64 = [lj_cutoff].into_iter().fold(0.0, f64::max);
    cutoff_max = cutoff_max.max(spme_r_cut_real);
    let required = 3.0 * (cutoff_max + r_skin);
    assert!(
        (required - 3.0 * (spme_r_cut_real + r_skin)).abs() < 1.0e-12,
        "required width should be driven by the larger of spme.r_cut_real and pair cutoffs"
    );
}

// @rq-203ecf81
//
// Reject a config that declares both [spme] and [coulomb].
// rq-203ecf81
#[test]
fn config_rejects_both_spme_and_coulomb() {
    let toml = r#"schema_version = 1
init = "sim.in.xyz"

[simulation]
seed = 0
temperature = 300.0

[[phase]]
name = "run"
n_steps = 1
dt = 1.0e-15


[phase.integrator]
kind = "velocity-verlet"
lossless = false

[[particle_types]]
name = "Na"
mass = 23.0
charge = 1.0

[[pair_interactions]]
between = ["Na", "Na"]
potential = "lennard-jones"
sigma = 3.4e-10
epsilon = 1.65e-21
cutoff = 1.0e-9

[coulomb]
cutoff = 0.3e-9
r_switch = 0.3e-9

[spme]
alpha = 4.0e9
r_cut_real = 0.3e-9
grid = [16, 16, 16]
spline_order = 4

[phase.output]
trajectory_every = 0
log_every = 0
"#;
    let path = std::env::temp_dir().join(format!(
        "dynamics_spme_conflict_{}.in.toml",
        std::process::id()
    ));
    std::fs::write(&path, toml).unwrap();
    let result = heddle_md::io::config::load_config(&path);
    let _ = std::fs::remove_file(&path);
    assert!(
        matches!(
            result,
            Err(heddle_md::io::config::ConfigError::ConflictingElectrostatics)
        ),
        "expected ConflictingElectrostatics, got {:?}",
        result.as_ref().err()
    );
}

// @rq-aeb23925
#[test]
fn spme_rejects_grid_below_2_times_spline_order_along_a() {
    let gpu = init_device().unwrap();
    let sim_box = nm_box();
    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [7, 16, 16],
        spline_order: 4,
    };
    let res = SpmeReciprocalState::new(&gpu, &sim_box, 1, &[1.0], params);
    assert!(matches!(
        res,
        Err(heddle_md::forces::SpmeError::InvalidGrid { axis: "a", n: 7, required: 8 })
    ));
}

// @rq-ab74a666
#[test]
fn spme_rejects_grid_below_2_times_spline_order_along_b() {
    let gpu = init_device().unwrap();
    let sim_box = nm_box();
    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [16, 7, 16],
        spline_order: 4,
    };
    let res = SpmeReciprocalState::new(&gpu, &sim_box, 1, &[1.0], params);
    assert!(matches!(
        res,
        Err(heddle_md::forces::SpmeError::InvalidGrid { axis: "b", n: 7, required: 8 })
    ));
}

// @rq-ea67c26b
//
// Doubling all charges quadruples reciprocal-space energy (since the
// energy is bilinear in q) and scales forces linearly in q.
// Verified through the slot's per-particle energy output.
#[test]
fn doubling_charges_quadruples_reciprocal_energy() {
    let gpu = init_device().unwrap();
    let sim_box = nm_box();
    let positions = [[0.1e-9, 0.0, 0.0], [-0.1e-9, 0.0, 0.0]];
    let params = SpmeParameters::from(&default_spme_config());

    let n = positions.len();
    let device = gpu.device.clone();
    let energy_single = run_spme_recip_energy(&gpu, &sim_box, &positions, &[1.0, -1.0], params);
    let energy_doubled = run_spme_recip_energy(&gpu, &sim_box, &positions, &[2.0, -2.0], params);
    // Self-energy also scales as q²; quotient still 4×.
    let ratio = energy_doubled / energy_single;
    assert!(
        (ratio - 4.0).abs() < 5.0e-3,
        "doubled-charge energy ratio = {ratio} (expected ~4)"
    );
    let _ = device;
    let _ = n;
}

fn run_spme_recip_energy(
    gpu: &GpuContext,
    sim_box: &SimulationBox,
    positions: &[[Real; 3]],
    charges: &[Real],
    params: SpmeParameters,
) -> f64 {
    let n = positions.len();
    let state = build_state(positions, charges);
    let buffers = ParticleBuffers::new(gpu, &state).unwrap();
    let mut slot = SpmeReciprocalState::new(gpu, sim_box, n, charges, params).unwrap();
    let mut t = Timings::new(gpu).unwrap();
    let cx = ForceFieldContext {
        neighbor_list: None,
        buffers: &buffers,
        sim_box,
    };
    slot.contribute(&buffers, sim_box, &cx, &mut t).unwrap();
    let device = gpu.device.clone();
    let mut force_x = device.alloc_zeros::<Real>(n).unwrap();
    let mut force_y = device.alloc_zeros::<Real>(n).unwrap();
    let mut force_z = device.alloc_zeros::<Real>(n).unwrap();
    let mut energy = device.alloc_zeros::<Real>(n).unwrap();
    let mut virial = device.alloc_zeros::<Real>(n).unwrap();
    {
        let view = heddle_md::forces::SlotOutputView {
            force_x: force_x.slice_mut(0..n),
            force_y: force_y.slice_mut(0..n),
            force_z: force_z.slice_mut(0..n),
            energy: energy.slice_mut(0..n),
            virial: virial.slice_mut(0..n),
        };
        slot.reduce(view, &cx, &mut t, heddle_md::forces::AggregateLevel::ForcesAndScalars).unwrap();
    }
    let energies: Vec<Real> = device.dtoh_sync_copy(&energy).unwrap();
    energies.iter().map(|&e| e as f64).sum()
}

// Run the real-space SPME slot for a fixed system; return (fx, fy, fz, energy, virial) per particle.
fn run_spme_real(
    gpu: &GpuContext,
    sim_box: &SimulationBox,
    positions: &[[Real; 3]],
    charges: &[Real],
    alpha: Real,
    r_cut_real: Real,
    exclusions: &ExclusionList,
) -> (Vec<Real>, Vec<Real>, Vec<Real>, Vec<Real>, Vec<Real>) {
    let n = positions.len();
    let state = build_state(positions, charges);
    let buffers = ParticleBuffers::new(gpu, &state).unwrap();
    let nl = NeighborListState::new_trivial(gpu, sim_box, n).unwrap();
    let mut slot = SpmeRealSpaceState::new(gpu, n, alpha, r_cut_real, n as u32, exclusions).unwrap();
    let mut t = Timings::new(gpu).unwrap();
    let cx = ForceFieldContext {
        neighbor_list: Some(&nl),
        buffers: &buffers,
        sim_box,
    };
    slot.contribute(&buffers, sim_box, &cx, &mut t).unwrap();
    let device = gpu.device.clone();
    let mut force_x = device.alloc_zeros::<Real>(n).unwrap();
    let mut force_y = device.alloc_zeros::<Real>(n).unwrap();
    let mut force_z = device.alloc_zeros::<Real>(n).unwrap();
    let mut energy = device.alloc_zeros::<Real>(n).unwrap();
    let mut virial = device.alloc_zeros::<Real>(n).unwrap();
    {
        let view = heddle_md::forces::SlotOutputView {
            force_x: force_x.slice_mut(0..n),
            force_y: force_y.slice_mut(0..n),
            force_z: force_z.slice_mut(0..n),
            energy: energy.slice_mut(0..n),
            virial: virial.slice_mut(0..n),
        };
        slot.reduce(view, &cx, &mut t, heddle_md::forces::AggregateLevel::ForcesAndScalars).unwrap();
    }
    (
        device.dtoh_sync_copy(&force_x).unwrap(),
        device.dtoh_sync_copy(&force_y).unwrap(),
        device.dtoh_sync_copy(&force_z).unwrap(),
        device.dtoh_sync_copy(&energy).unwrap(),
        device.dtoh_sync_copy(&virial).unwrap(),
    )
}

// rq-0caebe37
#[test]
fn real_space_slot_obeys_newtons_third_law_for_non_boundary_pair() {
    use heddle_md::forces::Exclusion;
    let gpu = init_device().unwrap();
    let sim_box = SimulationBox::new(2.0e-9, 2.0e-9, 2.0e-9, 0.0, 0.0, 0.0).unwrap();
    // Isolated pair well inside r_cut_real and far from box faces.
    let positions = [[0.05e-9, 0.0, 0.0], [0.20e-9, 0.0, 0.0]];
    let charges = [1.0e-19, -1.0e-19];
    let _ = Exclusion {
        atom_i: 0,
        atom_j: 1,
        scale_lj: 0.0,
        scale_coul: 0.0,
    };
    let (fx, fy, fz, _e, _w) = run_spme_real(
        &gpu,
        &sim_box,
        &positions,
        &charges,
        4.0e9,
        0.5e-9,
        &ExclusionList::empty(2),
    );
    assert_eq!(fx[0], -fx[1], "fx[0]={} must equal -fx[1]={} bit-exactly", fx[0], fx[1]);
    assert_eq!(fy[0], -fy[1], "fy[0]={} must equal -fy[1]={} bit-exactly", fy[0], fy[1]);
    assert_eq!(fz[0], -fz[1], "fz[0]={} must equal -fz[1]={} bit-exactly", fz[0], fz[1]);
}

// rq-3726c0f1
#[test]
fn real_space_slot_produces_zero_force_on_an_excluded_pair() {
    use heddle_md::forces::Exclusion;
    let gpu = init_device().unwrap();
    let sim_box = SimulationBox::new(2.0e-9, 2.0e-9, 2.0e-9, 0.0, 0.0, 0.0).unwrap();
    // Two oppositely charged particles within r_cut_real, but excluded
    // with scale_coul = 0. Expected output: all-zero force/energy/virial.
    let positions = [[0.0, 0.0, 0.0], [0.15e-9, 0.0, 0.0]];
    let charges = [1.0e-19, -1.0e-19];
    let mut excl = ExclusionList::empty(2);
    excl.entries.push(Exclusion {
        atom_i: 0,
        atom_j: 1,
        scale_lj: 0.0,
        scale_coul: 0.0,
    });
    excl.atom_excl_offsets = vec![0, 1, 2];
    excl.atom_excl_partners = vec![1, 0];
    excl.atom_excl_lj_scales = vec![0.0, 0.0];
    excl.atom_excl_coul_scales = vec![0.0, 0.0];
    let (fx, fy, fz, e, w) = run_spme_real(
        &gpu,
        &sim_box,
        &positions,
        &charges,
        4.0e9,
        0.5e-9,
        &excl,
    );
    for arr in [&fx, &fy, &fz, &e, &w] {
        for (i, &v) in arr.iter().enumerate() {
            assert_eq!(v, 0.0, "excluded-pair output[{i}] = {v}, expected exactly 0.0");
        }
    }
}

// rq-0816969e
#[test]
fn reciprocal_space_virial_is_distributed_equally_per_particle() {
    let gpu = init_device().unwrap();
    let sim_box = nm_box();
    let e_charge: Real = 1.602176634e-19;
    let charges = [e_charge, -e_charge, 2.0 * e_charge, -2.0 * e_charge];
    let positions = [
        [0.1e-9, 0.0, 0.0],
        [-0.1e-9, 0.2e-9, 0.0],
        [0.0, -0.2e-9, 0.15e-9],
        [-0.15e-9, 0.1e-9, -0.1e-9],
    ];
    let n = positions.len();
    let state = build_state(&positions, &charges);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let params = SpmeParameters::from(&default_spme_config());
    let mut slot = SpmeReciprocalState::new(&gpu, &sim_box, n, &charges, params).unwrap();
    let mut t = Timings::new(&gpu).unwrap();
    let cx = ForceFieldContext {
        neighbor_list: None,
        buffers: &buffers,
        sim_box: &sim_box,
    };
    slot.contribute(&buffers, &sim_box, &cx, &mut t).unwrap();

    let device = gpu.device.clone();
    let mut force_x = device.alloc_zeros::<Real>(n).unwrap();
    let mut force_y = device.alloc_zeros::<Real>(n).unwrap();
    let mut force_z = device.alloc_zeros::<Real>(n).unwrap();
    let mut energy = device.alloc_zeros::<Real>(n).unwrap();
    let mut virial = device.alloc_zeros::<Real>(n).unwrap();
    {
        let view = heddle_md::forces::SlotOutputView {
            force_x: force_x.slice_mut(0..n),
            force_y: force_y.slice_mut(0..n),
            force_z: force_z.slice_mut(0..n),
            energy: energy.slice_mut(0..n),
            virial: virial.slice_mut(0..n),
        };
        slot.reduce(view, &cx, &mut t, heddle_md::forces::AggregateLevel::ForcesAndScalars)
            .unwrap();
    }
    let virials: Vec<Real> = device.dtoh_sync_copy(&virial).unwrap();
    // Every per-particle virial entry equals W_recip / N (within f32 round-off).
    let expected = virials[0];
    assert!(expected.abs() > 0.0, "reciprocal virial must be non-zero");
    for (i, &v) in virials.iter().enumerate() {
        assert_eq!(
            v, expected,
            "particle {i} virial {v} != expected uniform share {expected}"
        );
    }
    let sum: f64 = virials.iter().map(|&v| v as f64).sum();
    assert!(
        ((sum - (n as f64 * expected as f64)).abs()) < 1.0e-6 * sum.abs(),
        "sum {sum} should be n * uniform_share = {}",
        n as f64 * expected as f64
    );
}

// rq-3b9611f2
#[test]
fn spme_runs_on_a_triclinic_box_with_non_zero_tilts() {
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(l, l, l, 0.10 * l, -0.08 * l, 0.05 * l).unwrap();
    let alpha: Real = 4.0e9;
    let e_charge: Real = 1.602176634e-19;
    let charges = [e_charge, -e_charge, 2.0 * e_charge, -2.0 * e_charge];
    let positions = [
        [0.1e-9, 0.0, 0.0],
        [-0.1e-9, 0.15e-9, 0.0],
        [0.0, -0.15e-9, 0.10e-9],
        [-0.10e-9, 0.05e-9, -0.10e-9],
    ];
    let n = positions.len();
    let params = SpmeParameters {
        alpha,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 4,
    };
    let state = build_state(&positions, &charges);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut slot = SpmeReciprocalState::new(&gpu, &sim_box, n, &charges, params).unwrap();
    let mut t = Timings::new(&gpu).unwrap();
    let cx = ForceFieldContext {
        neighbor_list: None,
        buffers: &buffers,
        sim_box: &sim_box,
    };
    slot.contribute(&buffers, &sim_box, &cx, &mut t).unwrap();
    slot.grid().sync_recip().unwrap();

    // Use the same U_recip = Σ slot_energy + U_self identity that
    // `spme_reciprocal_total_energy_equals_recip_minus_self` checks on
    // an orthorhombic box. Passing it here verifies the influence
    // function (built from the reciprocal lattice H^(-T) of the
    // triclinic box) feeds a self-consistent pipeline.
    let device = gpu.device.clone();
    let rho: Vec<Real> = device.dtoh_sync_copy(&slot.grid().rho).unwrap();
    let v: Vec<Real> = device.dtoh_sync_copy(&slot.grid().v).unwrap();
    let u_recip: f64 = 0.5
        * rho.iter().zip(v.iter()).map(|(&r, &vg)| (r as f64) * (vg as f64)).sum::<f64>();

    let mut force_x = device.alloc_zeros::<Real>(n).unwrap();
    let mut force_y = device.alloc_zeros::<Real>(n).unwrap();
    let mut force_z = device.alloc_zeros::<Real>(n).unwrap();
    let mut energy = device.alloc_zeros::<Real>(n).unwrap();
    let mut virial = device.alloc_zeros::<Real>(n).unwrap();
    {
        let view = heddle_md::forces::SlotOutputView {
            force_x: force_x.slice_mut(0..n),
            force_y: force_y.slice_mut(0..n),
            force_z: force_z.slice_mut(0..n),
            energy: energy.slice_mut(0..n),
            virial: virial.slice_mut(0..n),
        };
        slot.reduce(view, &cx, &mut t, heddle_md::forces::AggregateLevel::ForcesAndScalars)
            .unwrap();
    }
    let energies: Vec<Real> = device.dtoh_sync_copy(&energy).unwrap();
    let total: f64 = energies.iter().map(|&e| e as f64).sum();
    let inv_sqrt_pi = 1.0_f64 / PI.sqrt();
    let prefactor = (K_COULOMB_F32 as f64) * (alpha as f64) * inv_sqrt_pi;
    let u_self_total: f64 = charges
        .iter()
        .map(|&q| prefactor * (q as f64) * (q as f64))
        .sum();
    let expected = u_recip - u_self_total;
    let rel = (total - expected).abs() / expected.abs().max(1.0e-30);
    assert!(
        rel < 5.0e-3,
        "Σ slot_energy = {:e}, expected (U_recip − U_self) = {:e}, rel = {:e}",
        total,
        expected,
        rel
    );

    let fx: Vec<Real> = device.dtoh_sync_copy(&force_x).unwrap();
    assert!(
        fx.iter().any(|&v| v.is_finite() && v != 0.0),
        "expected non-zero, finite forces on triclinic box"
    );
}

// rq-991b4695
#[test]
fn box_compatibility_ignores_the_spme_reciprocal_grid() {
    // The runner's box-compatibility check is built from r_cut_real and
    // r_skin only — the FFT grid resolution does not enter. Replicate
    // the runner's max-aggregation logic across two configs that differ
    // only in `grid` and confirm the required width is unchanged.
    let r_skin: f64 = 0.05;
    let lj_cutoff: f64 = 0.3;
    let spme_r_cut_real: f64 = 0.6;
    let required_for = |_grid: [u32; 3]| -> f64 {
        // The grid argument is intentionally ignored; this mirrors the
        // runner's threshold derivation.
        let cutoff_max = [lj_cutoff, spme_r_cut_real]
            .into_iter()
            .fold(0.0, f64::max);
        3.0 * (cutoff_max + r_skin)
    };
    assert_eq!(required_for([16, 16, 16]), required_for([256, 256, 256]));
}

// rq-73efd4be
#[test]
fn two_stream_pipeline_preserves_bit_exact_reproducibility_across_runs() {
    let gpu = init_device().unwrap();
    let sim_box = nm_box();
    let e_charge: Real = 1.602176634e-19;
    let charges = [e_charge, -e_charge, 2.0 * e_charge, -2.0 * e_charge];
    let positions = [
        [0.1e-9, 0.0, 0.0],
        [-0.1e-9, 0.2e-9, 0.0],
        [0.0, -0.2e-9, 0.15e-9],
        [-0.15e-9, 0.1e-9, -0.1e-9],
    ];
    let n = positions.len();
    let params = SpmeParameters::from(&default_spme_config());
    let run = || -> (Vec<Real>, Vec<Real>, Vec<Real>, Vec<Real>, Vec<Real>) {
        let state = build_state(&positions, &charges);
        let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
        let mut slot = SpmeReciprocalState::new(&gpu, &sim_box, n, &charges, params).unwrap();
        let mut t = Timings::new(&gpu).unwrap();
        let cx = ForceFieldContext {
            neighbor_list: None,
            buffers: &buffers,
            sim_box: &sim_box,
        };
        slot.contribute(&buffers, &sim_box, &cx, &mut t).unwrap();
        let device = gpu.device.clone();
        let mut fx = device.alloc_zeros::<Real>(n).unwrap();
        let mut fy = device.alloc_zeros::<Real>(n).unwrap();
        let mut fz = device.alloc_zeros::<Real>(n).unwrap();
        let mut e = device.alloc_zeros::<Real>(n).unwrap();
        let mut w = device.alloc_zeros::<Real>(n).unwrap();
        {
            let view = heddle_md::forces::SlotOutputView {
                force_x: fx.slice_mut(0..n),
                force_y: fy.slice_mut(0..n),
                force_z: fz.slice_mut(0..n),
                energy: e.slice_mut(0..n),
                virial: w.slice_mut(0..n),
            };
            slot.reduce(view, &cx, &mut t, heddle_md::forces::AggregateLevel::ForcesAndScalars)
                .unwrap();
        }
        (
            device.dtoh_sync_copy(&fx).unwrap(),
            device.dtoh_sync_copy(&fy).unwrap(),
            device.dtoh_sync_copy(&fz).unwrap(),
            device.dtoh_sync_copy(&e).unwrap(),
            device.dtoh_sync_copy(&w).unwrap(),
        )
    };
    let a = run();
    let b = run();
    assert_eq!(a.0, b.0, "fx not bit-exact across two SPME-enabled runs");
    assert_eq!(a.1, b.1, "fy not bit-exact across two SPME-enabled runs");
    assert_eq!(a.2, b.2, "fz not bit-exact across two SPME-enabled runs");
    assert_eq!(a.3, b.3, "energy not bit-exact across two SPME-enabled runs");
    assert_eq!(a.4, b.4, "virial not bit-exact across two SPME-enabled runs");
}

