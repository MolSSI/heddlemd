// rq-202493a5 rq-9ca00d25
//
// PR 1 validation: run the SPME reciprocal-space pipeline (spread →
// FFT → multiply → IFFT) and compare its reciprocal-space energy
// (1/2) Σ_g rho[g] · V[g] against an explicit-Ewald brute-force sum on
// the host.

use std::f64::consts::PI;

use dynamics::forces::spme::{SpmeParameters, SpmeReciprocalGrid};
use dynamics::gpu::{K_COULOMB_F32, ParticleBuffers, init_device};
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

/// Explicit-Ewald reciprocal-space energy (truncated to |m_d| <= m_max).
/// Returns U_recip in joules. For an orthorhombic box.
fn explicit_ewald_recip_energy(
    positions: &[[f32; 3]],
    charges: &[f32],
    box_lengths: [f64; 3],
    alpha: f64,
    m_max: i32,
) -> f64 {
    let v_box = box_lengths[0] * box_lengths[1] * box_lengths[2];
    let prefactor = (K_COULOMB_F32 as f64) / (2.0 * v_box);
    let inv_4_alpha2 = 1.0 / (4.0 * alpha * alpha);
    let mut total = 0.0_f64;
    for ma in -m_max..=m_max {
        for mb in -m_max..=m_max {
            for mc in -m_max..=m_max {
                if ma == 0 && mb == 0 && mc == 0 {
                    continue;
                }
                let kx = 2.0 * PI * (ma as f64) / box_lengths[0];
                let ky = 2.0 * PI * (mb as f64) / box_lengths[1];
                let kz = 2.0 * PI * (mc as f64) / box_lengths[2];
                let k2 = kx * kx + ky * ky + kz * kz;
                // ρ̂_true(K) = Σ_i q_i exp(-i K·r_i)
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for (i, q) in charges.iter().enumerate() {
                    let phase = kx * positions[i][0] as f64
                        + ky * positions[i][1] as f64
                        + kz * positions[i][2] as f64;
                    re += (*q as f64) * phase.cos();
                    im -= (*q as f64) * phase.sin();
                }
                let s2 = re * re + im * im;
                total += (4.0 * PI / k2) * (-k2 * inv_4_alpha2).exp() * s2;
            }
        }
    }
    prefactor * total
}

fn spme_energy_from_pipeline(grid: &SpmeReciprocalGrid) -> f64 {
    // (1/2) Σ_g rho[g] · V[g] computed host-side after dtoh.
    let device = grid.device.clone();
    let m = grid.m;
    let rho_host: Vec<f32> = device.dtoh_sync_copy(&grid.rho).expect("dtoh rho");
    let v_host: Vec<f32> = device.dtoh_sync_copy(&grid.v).expect("dtoh V");
    assert_eq!(rho_host.len(), m);
    assert_eq!(v_host.len(), m);
    let mut e = 0.0_f64;
    for i in 0..m {
        e += (rho_host[i] as f64) * (v_host[i] as f64);
    }
    0.5 * e
}

// rq-9ca00d25
#[test]
fn spme_pipeline_matches_explicit_ewald_two_charge_pair() {
    let gpu = init_device().unwrap();
    let l = 1.0e-9_f32;
    let sim_box = SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap();
    let positions = [[0.2e-9_f32, 0.0, 0.0], [-0.2e-9, 0.0, 0.0]];
    let e_charge = 1.602176634e-19_f32;
    let charges = [e_charge, -e_charge];
    let state = build_state(&positions, &charges);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();

    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 4,
    };
    let mut spme = SpmeReciprocalGrid::new(&gpu, &sim_box, 2, params).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    spme.compute(&sim_box, &buffers, &mut timings).unwrap();
    spme.sync_recip().unwrap();

    let energy_spme = spme_energy_from_pipeline(&spme);
    let energy_ref = explicit_ewald_recip_energy(
        &positions,
        &charges,
        [l as f64, l as f64, l as f64],
        params.alpha as f64,
        8,
    );
    let rel = (energy_spme - energy_ref).abs() / energy_ref.abs();
    assert!(
        rel < 5.0e-3,
        "SPME = {:e}, ref = {:e}, rel error = {:e}",
        energy_spme,
        energy_ref,
        rel
    );
}

// rq-2996a545
#[test]
fn spme_pipeline_matches_explicit_ewald_four_charges() {
    let gpu = init_device().unwrap();
    let l = 1.0e-9_f32;
    let sim_box = SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap();
    let positions = [
        [0.1e-9_f32, 0.0, 0.0],
        [-0.1e-9, 0.2e-9, 0.0],
        [0.0, -0.2e-9, 0.15e-9],
        [-0.15e-9, 0.1e-9, -0.1e-9],
    ];
    let e_charge = 1.602176634e-19_f32;
    let charges = [e_charge, -e_charge, 2.0 * e_charge, -2.0 * e_charge];
    let state = build_state(&positions, &charges);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();

    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 4,
    };
    let mut spme =
        SpmeReciprocalGrid::new(&gpu, &sim_box, positions.len(), params).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    spme.compute(&sim_box, &buffers, &mut timings).unwrap();
    spme.sync_recip().unwrap();

    let energy_spme = spme_energy_from_pipeline(&spme);
    let energy_ref = explicit_ewald_recip_energy(
        &positions,
        &charges,
        [l as f64, l as f64, l as f64],
        params.alpha as f64,
        8,
    );
    let rel = (energy_spme - energy_ref).abs() / energy_ref.abs();
    assert!(
        rel < 5.0e-3,
        "SPME = {:e}, ref = {:e}, rel error = {:e}",
        energy_spme,
        energy_ref,
        rel
    );
}

// rq-79291441
#[test]
fn spread_conserves_total_charge() {
    let gpu = init_device().unwrap();
    let l = 1.0e-9_f32;
    let sim_box = SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap();
    let positions = [
        [0.1e-9_f32, 0.05e-9, -0.2e-9],
        [-0.1e-9, 0.2e-9, 0.15e-9],
        [0.25e-9, -0.1e-9, 0.0],
    ];
    // Use a net-non-zero charge configuration so the relative-error
    // assertion is well-defined.
    let charges = [0.3_f32, -0.5, 0.4];
    let total_charge: f64 = charges.iter().map(|&q| q as f64).sum();
    let state = build_state(&positions, &charges);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();

    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 4,
    };
    let mut spme =
        SpmeReciprocalGrid::new(&gpu, &sim_box, positions.len(), params).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    spme.compute(&sim_box, &buffers, &mut timings).unwrap();
    spme.sync_recip().unwrap();

    let rho: Vec<f32> = gpu.device.dtoh_sync_copy(&spme.rho).unwrap();
    let summed: f64 = rho.iter().map(|&v| v as f64).sum();
    let rel = (summed - total_charge).abs() / total_charge.abs();
    assert!(
        rel < 1.0e-4,
        "Σ rho = {}, expected {}, rel error = {:e}",
        summed,
        total_charge,
        rel
    );
}

// rq-e5bf6fea
#[test]
fn k_zero_entry_of_influence_function_is_zero() {
    let gpu = init_device().unwrap();
    let l = 1.0e-9_f32;
    let sim_box = SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap();
    let state = build_state(&[[0.0, 0.0, 0.0]], &[1.0]);
    let _buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 4,
    };
    let spme = SpmeReciprocalGrid::new(&gpu, &sim_box, 1, params).unwrap();
    let g: Vec<f32> = gpu.device.dtoh_sync_copy(&spme.influence_g).unwrap();
    assert_eq!(g[0], 0.0, "influence_G[0] must be zero (tinfoil BC)");
}

// rq-09d4e13f
#[test]
fn identical_inputs_produce_byte_identical_grids() {
    let gpu = init_device().unwrap();
    let l = 1.0e-9_f32;
    let sim_box = SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap();
    let positions = [[0.1e-9_f32, 0.0, 0.0], [-0.1e-9, 0.0, 0.0]];
    let e = 1.602176634e-19_f32;
    let charges = [e, -e];
    let state = build_state(&positions, &charges);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();

    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 4,
    };

    let mut spme1 = SpmeReciprocalGrid::new(&gpu, &sim_box, 2, params).unwrap();
    let mut spme2 = SpmeReciprocalGrid::new(&gpu, &sim_box, 2, params).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    spme1.compute(&sim_box, &buffers, &mut timings).unwrap();
    spme2.compute(&sim_box, &buffers, &mut timings).unwrap();
    spme1.sync_recip().unwrap();
    spme2.sync_recip().unwrap();

    let v1: Vec<f32> = gpu.device.dtoh_sync_copy(&spme1.v).unwrap();
    let v2: Vec<f32> = gpu.device.dtoh_sync_copy(&spme2.v).unwrap();
    assert_eq!(v1, v2);
    let rho1: Vec<f32> = gpu.device.dtoh_sync_copy(&spme1.rho).unwrap();
    let rho2: Vec<f32> = gpu.device.dtoh_sync_copy(&spme2.rho).unwrap();
    assert_eq!(rho1, rho2);
}
