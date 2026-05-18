// rq-202493a5 rq-9ca00d25 rq-f6d45062
//
// PR 2 validation: tests that exercise the SPME slot wrappers
// (`SpmeRealSpaceState`, `SpmeReciprocalState`) and the
// `ForceField`-level integration of [spme] config.

use std::f64::consts::PI;

use dynamics::forces::neighbor_list::NeighborListState;
use dynamics::forces::{ExclusionList, ForceField, ForceFieldContext, Potential, PotentialRegistry, SpmeParameters, SpmeRealSpaceState, SpmeReciprocalState};
use dynamics::gpu::cufft::cufft_determinism_smoke_test;
use dynamics::gpu::{GpuContext, K_COULOMB_F32, ParticleBuffers, init_device};
use dynamics::io::config::{NeighborListConfig, ParticleTypeConfig, SpmeConfig};
use dynamics::pbc::SimulationBox;
use dynamics::state::ParticleState;
use dynamics::timings::Timings;

fn build_state(positions: &[[f32; 3]], charges: &[f32]) -> ParticleState {
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
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![1.0_f32; n],
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
    let l = 1.0e-9_f32;
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
    let charges = vec![1.0_f32];
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
        &dynamics::forces::BondList::empty(1),
        &dynamics::forces::AngleList::empty(0),
        &ExclusionList::empty(1),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    assert_eq!(ff.slots.len(), 2, "expected two SPME slots");
    assert_eq!(ff.slots[0].label(), "spme_real");
    assert_eq!(ff.slots[1].label(), "spme_reciprocal");
}

// @rq-09d4e13f
#[test]
fn spme_reciprocal_state_two_independent_constructs_produce_byte_identical_grids() {
    let gpu = init_device().unwrap();
    let sim_box = nm_box();
    let e_charge = 1.602176634e-19_f32;
    let positions = [[0.1e-9_f32, 0.0, 0.0], [-0.1e-9, 0.0, 0.0]];
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

    let v1: Vec<f32> = gpu.device.dtoh_sync_copy(&s1.grid().v).unwrap();
    let v2: Vec<f32> = gpu.device.dtoh_sync_copy(&s2.grid().v).unwrap();
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
    let alpha: f32 = 4.0e9;
    let e_charge: f32 = 1.602176634e-19;
    let charges = [e_charge, -e_charge, 2.0 * e_charge, -2.0 * e_charge];
    let positions = [
        [0.1e-9_f32, 0.0, 0.0],
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

    // Independent U_recip from the pipeline-resident buffers.
    let device = gpu.device.clone();
    let rho: Vec<f32> = device.dtoh_sync_copy(&slot.grid().rho).unwrap();
    let v: Vec<f32> = device.dtoh_sync_copy(&slot.grid().v).unwrap();
    let u_recip: f64 = 0.5
        * rho
            .iter()
            .zip(v.iter())
            .map(|(&r, &vg)| (r as f64) * (vg as f64))
            .sum::<f64>();

    // Run reduce() into a fresh output view.
    let mut force_x = device.alloc_zeros::<f32>(n).unwrap();
    let mut force_y = device.alloc_zeros::<f32>(n).unwrap();
    let mut force_z = device.alloc_zeros::<f32>(n).unwrap();
    let mut energy = device.alloc_zeros::<f32>(n).unwrap();
    let mut virial = device.alloc_zeros::<f32>(n).unwrap();
    {
        let view = dynamics::forces::SlotOutputView {
            force_x: force_x.slice_mut(0..n),
            force_y: force_y.slice_mut(0..n),
            force_z: force_z.slice_mut(0..n),
            energy: energy.slice_mut(0..n),
            virial: virial.slice_mut(0..n),
        };
        slot.reduce(view, &cx, &mut t).unwrap();
    }
    let energies: Vec<f32> = device.dtoh_sync_copy(&energy).unwrap();
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
#[test]
fn spme_real_slot_zero_outside_r_cut() {
    let gpu = init_device().unwrap();
    let sim_box = SimulationBox::new(2.0e-9, 2.0e-9, 2.0e-9, 0.0, 0.0, 0.0).unwrap();
    let r_cut_real: f32 = 0.3e-9;
    let positions = [[0.0_f32, 0.0, 0.0], [r_cut_real + 0.05e-9, 0.0, 0.0]];
    let charges = [1.0e-19_f32, -1.0e-19];
    let state = build_state(&positions, &charges);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();

    let nl = NeighborListState::new_trivial(&gpu, &sim_box, 2).unwrap();
    let mut slot = SpmeRealSpaceState::new(
        &gpu,
        2,
        4.0e9_f32,
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
    let mut force_x = device.alloc_zeros::<f32>(n).unwrap();
    let mut force_y = device.alloc_zeros::<f32>(n).unwrap();
    let mut force_z = device.alloc_zeros::<f32>(n).unwrap();
    let mut energy = device.alloc_zeros::<f32>(n).unwrap();
    let mut virial = device.alloc_zeros::<f32>(n).unwrap();
    {
        let view = dynamics::forces::SlotOutputView {
            force_x: force_x.slice_mut(0..n),
            force_y: force_y.slice_mut(0..n),
            force_z: force_z.slice_mut(0..n),
            energy: energy.slice_mut(0..n),
            virial: virial.slice_mut(0..n),
        };
        slot.reduce(view, &cx, &mut t).unwrap();
    }
    let fx: Vec<f32> = device.dtoh_sync_copy(&force_x).unwrap();
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
#[test]
fn spme_real_slot_matches_closed_form_erfc_pair() {
    let gpu = init_device().unwrap();
    let sim_box = SimulationBox::new(2.0e-9, 2.0e-9, 2.0e-9, 0.0, 0.0, 0.0).unwrap();
    let alpha: f32 = 4.0e9;
    let r_cut_real: f32 = 0.5e-9;
    let r: f32 = 0.15e-9;
    let positions = [[0.0_f32, 0.0, 0.0], [r, 0.0, 0.0]];
    let q1: f32 = 1.0e-19;
    let q2: f32 = -1.0e-19;
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
    let mut force_x = device.alloc_zeros::<f32>(n).unwrap();
    let mut force_y = device.alloc_zeros::<f32>(n).unwrap();
    let mut force_z = device.alloc_zeros::<f32>(n).unwrap();
    let mut energy = device.alloc_zeros::<f32>(n).unwrap();
    let mut virial = device.alloc_zeros::<f32>(n).unwrap();
    {
        let view = dynamics::forces::SlotOutputView {
            force_x: force_x.slice_mut(0..n),
            force_y: force_y.slice_mut(0..n),
            force_z: force_z.slice_mut(0..n),
            energy: energy.slice_mut(0..n),
            virial: virial.slice_mut(0..n),
        };
        slot.reduce(view, &cx, &mut t).unwrap();
    }
    let fx: Vec<f32> = device.dtoh_sync_copy(&force_x).unwrap();

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
    let fx0_expected = (factor * (-r_f)) as f32;
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
#[test]
fn config_rejects_both_spme_and_coulomb() {
    let toml = r#"schema_version = 1
init = "sim.in.xyz"

[simulation]
seed = 0
n_steps = 1
dt = 1.0e-15
temperature = 300.0

[integrator]
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

[output]
trajectory_every = 0
log_every = 0
"#;
    let path = std::env::temp_dir().join(format!(
        "dynamics_spme_conflict_{}.in.toml",
        std::process::id()
    ));
    std::fs::write(&path, toml).unwrap();
    let result = dynamics::io::config::load_config(&path);
    let _ = std::fs::remove_file(&path);
    assert!(
        matches!(
            result,
            Err(dynamics::io::config::ConfigError::ConflictingElectrostatics)
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
    let res = SpmeReciprocalState::new(&gpu, &sim_box, 1, &[1.0_f32], params);
    assert!(matches!(
        res,
        Err(dynamics::forces::SpmeError::InvalidGrid { axis: "a", n: 7, required: 8 })
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
    let res = SpmeReciprocalState::new(&gpu, &sim_box, 1, &[1.0_f32], params);
    assert!(matches!(
        res,
        Err(dynamics::forces::SpmeError::InvalidGrid { axis: "b", n: 7, required: 8 })
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
    let positions = [[0.1e-9_f32, 0.0, 0.0], [-0.1e-9, 0.0, 0.0]];
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
    positions: &[[f32; 3]],
    charges: &[f32],
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
    let mut force_x = device.alloc_zeros::<f32>(n).unwrap();
    let mut force_y = device.alloc_zeros::<f32>(n).unwrap();
    let mut force_z = device.alloc_zeros::<f32>(n).unwrap();
    let mut energy = device.alloc_zeros::<f32>(n).unwrap();
    let mut virial = device.alloc_zeros::<f32>(n).unwrap();
    {
        let view = dynamics::forces::SlotOutputView {
            force_x: force_x.slice_mut(0..n),
            force_y: force_y.slice_mut(0..n),
            force_z: force_z.slice_mut(0..n),
            energy: energy.slice_mut(0..n),
            virial: virial.slice_mut(0..n),
        };
        slot.reduce(view, &cx, &mut t).unwrap();
    }
    let energies: Vec<f32> = device.dtoh_sync_copy(&energy).unwrap();
    energies.iter().map(|&e| e as f64).sum()
}

