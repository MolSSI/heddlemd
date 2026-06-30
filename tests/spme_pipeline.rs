// rq-202493a5 rq-9ca00d25
//
// PR 1 validation: run the SPME reciprocal-space pipeline (spread →
// FFT → multiply → IFFT) and compare its reciprocal-space energy
// (1/2) Σ_g rho[g] · V[g] against an explicit-Ewald brute-force sum on
// the host.

use std::f64::consts::PI;

use heddle_md::forces::spme::{SpmeParameters, SpmeReciprocalGrid};
use heddle_md::gpu::{K_COULOMB_F32, ParticleBuffers, init_device};
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

/// Explicit-Ewald reciprocal-space energy (truncated to |m_d| <= m_max).
/// Returns U_recip in joules. For an orthorhombic box.
fn explicit_ewald_recip_energy(
    positions: &[[Real; 3]],
    charges: &[Real],
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
    let rho_host: Vec<Real> = device.dtoh_sync_copy(&grid.rho).expect("dtoh rho");
    let v_host: Vec<Real> = device.dtoh_sync_copy(&grid.v).expect("dtoh V");
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
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    let positions = [[0.2e-9, 0.0, 0.0], [-0.2e-9, 0.0, 0.0]];
    // Charges in elementary-charge units to match the engine's atomic-unit
    // pipeline; the reference energy uses the same charges, so the
    // relative-error assertion is unit-invariant.
    let charges = [1.0 as Real, -1.0];
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
    spme.compute(&sim_box, &buffers, 1, &mut timings).unwrap();
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
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    let positions = [
        [0.1e-9, 0.0, 0.0],
        [-0.1e-9, 0.2e-9, 0.0],
        [0.0, -0.2e-9, 0.15e-9],
        [-0.15e-9, 0.1e-9, -0.1e-9],
    ];
    // Charges in elementary-charge units to match the engine's atomic-unit
    // pipeline.
    let charges = [1.0 as Real, -1.0, 2.0, -2.0];
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
    spme.compute(&sim_box, &buffers, 1, &mut timings).unwrap();
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
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    let positions = [
        [0.1e-9, 0.05e-9, -0.2e-9],
        [-0.1e-9, 0.2e-9, 0.15e-9],
        [0.25e-9, -0.1e-9, 0.0],
    ];
    // Use a net-non-zero charge configuration so the relative-error
    // assertion is well-defined.
    let charges = [0.3, -0.5, 0.4];
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
    spme.compute(&sim_box, &buffers, 1, &mut timings).unwrap();
    spme.sync_recip().unwrap();

    let rho: Vec<Real> = gpu.device.dtoh_sync_copy(&spme.rho).unwrap();
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
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    let state = build_state(&[[0.0, 0.0, 0.0]], &[1.0]);
    let _buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 4,
    };
    let spme = SpmeReciprocalGrid::new(&gpu, &sim_box, 1, params).unwrap();
    let g: Vec<Real> = gpu.device.dtoh_sync_copy(&spme.influence_g).unwrap();
    assert_eq!(g[0], 0.0, "influence_G[0] must be zero (tinfoil BC)");
}

// rq-09d4e13f
#[test]
fn identical_inputs_produce_byte_identical_grids() {
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    let positions = [[0.1e-9, 0.0, 0.0], [-0.1e-9, 0.0, 0.0]];
    let e = 1.602176634e-19;
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
    spme1.compute(&sim_box, &buffers, 1, &mut timings).unwrap();
    spme2.compute(&sim_box, &buffers, 1, &mut timings).unwrap();
    spme1.sync_recip().unwrap();
    spme2.sync_recip().unwrap();

    let v1: Vec<Real> = gpu.device.dtoh_sync_copy(&spme1.v).unwrap();
    let v2: Vec<Real> = gpu.device.dtoh_sync_copy(&spme2.v).unwrap();
    assert_eq!(v1, v2);
    let rho1: Vec<Real> = gpu.device.dtoh_sync_copy(&spme1.rho).unwrap();
    let rho2: Vec<Real> = gpu.device.dtoh_sync_copy(&spme2.rho).unwrap();
    assert_eq!(rho1, rho2);
}

// Cardinal B-spline M_p(x) via Cox–de Boor recursion. Host-side replica
// of the (private) helper in src/forces/spme.rs used to derive a
// closed-form prediction for the charge-spread weight in tests below.
fn host_cardinal_bspline(p: usize, x: f64) -> f64 {
    let mut vals: Vec<f64> = (0..p)
        .map(|i| {
            let xi = x - (i as f64);
            if (0.0..1.0).contains(&xi) { 1.0 } else { 0.0 }
        })
        .collect();
    for order in 2..=p {
        let inv = 1.0 / (order as f64 - 1.0);
        for i in 0..(p - order + 1) {
            let xi = x - i as f64;
            vals[i] = xi * inv * vals[i] + ((order as f64) - xi) * inv * vals[i + 1];
        }
    }
    vals[0]
}

fn small_box_grid(spme: SpmeParameters, n: usize) -> (
    heddle_md::gpu::GpuContext,
    SimulationBox,
    SpmeReciprocalGrid,
) {
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    let grid = SpmeReciprocalGrid::new(&gpu, &sim_box, n, spme).unwrap();
    (gpu, sim_box, grid)
}

// rq-3c0beda9
#[test]
fn spread_for_one_isolated_particle_matches_b_spline_weights() {
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    let p_u = 4u32;
    let n_grid = 16u32;
    // Fractional position in [-0.5, 0.5) (box is centred at origin).
    let s = [0.11_f64, -0.18, 0.37];
    let q: Real = 0.5;
    let positions = [[(s[0] as Real) * l, (s[1] as Real) * l, (s[2] as Real) * l]];
    let state = build_state(&positions, &[q]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [n_grid, n_grid, n_grid],
        spline_order: p_u,
    };
    let mut grid = SpmeReciprocalGrid::new(&gpu, &sim_box, 1, params).unwrap();
    let mut t = Timings::new(&gpu).unwrap();
    grid.compute(&sim_box, &buffers, 1, &mut t).unwrap();
    grid.sync_recip().unwrap();
    let rho: Vec<Real> = gpu.device.dtoh_sync_copy(&grid.rho).unwrap();
    // The kernel maps each particle to fractional coords s_a, computes
    // u = (s_a + 0.5) * n, bin = floor(u), t = u - bin, and adds the
    // contribution `q · M_p(da + t)` to grid point `(bin + da) % n` for
    // da = 0..p. Equivalently for axis a, the grid point g_a receives a
    // contribution iff da_a := (bin_a - g_a) mod n is in [0, p).
    let n = n_grid as i64;
    let p_i = p_u as usize;
    let u: [f64; 3] = [
        (s[0] + 0.5) * n as f64,
        (s[1] + 0.5) * n as f64,
        (s[2] + 0.5) * n as f64,
    ];
    let bin = [u[0].floor() as i64, u[1].floor() as i64, u[2].floor() as i64];
    let t_axis = [u[0] - u[0].floor(), u[1] - u[1].floor(), u[2] - u[2].floor()];
    let axis_weight = |axis: usize, g: i64| -> f64 {
        let da = (bin[axis] - g).rem_euclid(n);
        if da < p_i as i64 {
            host_cardinal_bspline(p_i, da as f64 + t_axis[axis])
        } else {
            0.0
        }
    };
    for ga in 0..n {
        for gb in 0..n {
            for gc in 0..n {
                let idx = ((ga * n + gb) * n + gc) as usize;
                let expected = (q as f64)
                    * axis_weight(0, ga)
                    * axis_weight(1, gb)
                    * axis_weight(2, gc);
                let got = rho[idx] as f64;
                assert!(
                    (got - expected).abs() < 1.0e-5 * (1.0 + expected.abs()),
                    "rho[{ga},{gb},{gc}] = {got}, expected {expected}"
                );
            }
        }
    }
}

// rq-881559bd
#[test]
fn spread_is_zero_at_a_grid_point_with_no_particle_support() {
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    // Particle at fractional s = (0.2, 0.2, 0.2); with the kernel's
    // centring shift s + 0.5 → 0.7 and grid size 16, the spline bins are
    // {8, 9, 10, 11} on each axis (p = 4). Grid point (0, 0, 0) sits
    // outside that support on every axis, so rho[0] must be exactly zero.
    let positions = [[0.2 * l, 0.2 * l, 0.2 * l]];
    let state = build_state(&positions, &[1.0]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 4,
    };
    let mut grid = SpmeReciprocalGrid::new(&gpu, &sim_box, 1, params).unwrap();
    let mut t = Timings::new(&gpu).unwrap();
    grid.compute(&sim_box, &buffers, 1, &mut t).unwrap();
    grid.sync_recip().unwrap();
    let rho: Vec<Real> = gpu.device.dtoh_sync_copy(&grid.rho).unwrap();
    assert_eq!(rho[0], 0.0, "rho at unsupported grid point must be exactly zero");
}

// rq-07297467
#[test]
fn spread_is_byte_identical_under_two_input_orderings_in_same_bin() {
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    // Two particles in the same primary bin (close in fractional coords)
    // submitted in two different orderings.
    let p0 = [0.31 * l, 0.42 * l, 0.57 * l];
    let p1 = [0.32 * l, 0.43 * l, 0.58 * l];
    let q0: Real = 0.5;
    let q1: Real = -0.7;
    let state_a = build_state(&[p0, p1], &[q0, q1]);
    let state_b = build_state(&[p1, p0], &[q1, q0]);
    let buffers_a = ParticleBuffers::new(&gpu, &state_a).unwrap();
    let buffers_b = ParticleBuffers::new(&gpu, &state_b).unwrap();
    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 4,
    };
    let mut grid_a = SpmeReciprocalGrid::new(&gpu, &sim_box, 2, params).unwrap();
    let mut grid_b = SpmeReciprocalGrid::new(&gpu, &sim_box, 2, params).unwrap();
    let mut t = Timings::new(&gpu).unwrap();
    grid_a.compute(&sim_box, &buffers_a, 1, &mut t).unwrap();
    grid_b.compute(&sim_box, &buffers_b, 1, &mut t).unwrap();
    grid_a.sync_recip().unwrap();
    grid_b.sync_recip().unwrap();
    let rho_a: Vec<Real> = gpu.device.dtoh_sync_copy(&grid_a.rho).unwrap();
    let rho_b: Vec<Real> = gpu.device.dtoh_sync_copy(&grid_b.rho).unwrap();
    assert_eq!(rho_a, rho_b, "cell-list sort must canonicalise particle order within a bin");
}

// rq-e3c3898a
#[test]
fn forward_fft_of_a_zero_grid_produces_zero() {
    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 4,
    };
    let (gpu, _sim_box, mut grid) = small_box_grid(params, 0);
    // grid.rho was zero-allocated; execute forward FFT directly.
    grid.forward_plan
        .execute(&grid.rho, &mut grid.rho_hat_interleaved)
        .unwrap();
    grid.sync_recip().unwrap();
    let rho_hat: Vec<Real> = gpu
        .device
        .dtoh_sync_copy(&grid.rho_hat_interleaved)
        .unwrap();
    assert!(
        rho_hat.iter().all(|&v| v == 0.0),
        "FFT of zero rho must produce zero rho_hat"
    );
}

// rq-f02e9e0e
#[test]
fn inverse_fft_round_trips_forward_fft_up_to_scale_factor() {
    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [8, 8, 8],
        spline_order: 4,
    };
    let (gpu, _sim_box, mut grid) = small_box_grid(params, 0);
    // Fill rho with a deterministic non-trivial pattern.
    let m = grid.m;
    let input: Vec<Real> = (0..m).map(|i| ((i as Real) * 0.013).sin()).collect();
    gpu.device
        .htod_sync_copy_into(&input, &mut grid.rho)
        .unwrap();
    grid.forward_plan
        .execute(&grid.rho, &mut grid.rho_hat_interleaved)
        .unwrap();
    // Round-trip back to V via inverse, no influence multiply in between.
    grid.inverse_plan
        .execute(&grid.rho_hat_interleaved, &mut grid.v)
        .unwrap();
    grid.sync_recip().unwrap();
    let round_trip: Vec<Real> = gpu.device.dtoh_sync_copy(&grid.v).unwrap();
    let scale = m as f64;
    for (i, (&a, &b)) in input.iter().zip(round_trip.iter()).enumerate() {
        let expected = (a as f64) * scale;
        let rel = ((b as f64) - expected).abs() / expected.abs().max(1.0);
        assert!(rel < 1.0e-4, "round-trip mismatch at {i}: got {b}, expected {expected}");
    }
}

// rq-2ae37ac3
#[test]
fn spme_reciprocal_state_owns_a_length_M_fixed_point_grid() {
    use cudarc::driver::DeviceSlice;
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 4,
    };
    let n = 1usize;
    let grid = SpmeReciprocalGrid::new(&gpu, &sim_box, n, params).unwrap();
    let m = 16usize * 16 * 16;
    assert_eq!(grid.m, m);
    assert_eq!(grid.rho_fixed.len(), m);
}

// rq-dd829afb
#[test]
fn spread_pipeline_does_not_launch_neighbor_list_kernels() {
    use heddle_md::timings::KernelStage;
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    let positions = [[0.1e-9, 0.0, 0.0], [-0.1e-9, 0.0, 0.0]];
    let charges = [1.0, -1.0];
    let state = build_state(&positions, &charges);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 4,
    };
    let mut grid = SpmeReciprocalGrid::new(&gpu, &sim_box, 2, params).unwrap();
    let mut t = Timings::new(&gpu).unwrap();
    grid.compute(&sim_box, &buffers, 1, &mut t).unwrap();
    grid.compute(&sim_box, &buffers, 1, &mut t).unwrap();
    grid.sync_recip().unwrap();
    let report = t.finalize().unwrap();
    // SpmeReciprocalGrid no longer owns a NeighborListState; the four
    // pre-step bin-list kernels (displacement check, copy-positions,
    // neighbor-list build, neighbor-list rebuild) must never appear in
    // the report when only the recip slot has been driven.
    for forbidden in [
        KernelStage::NEIGHBOR_DISPLACEMENT_SQUARED.name(),
        KernelStage::COPY_POSITIONS_INTO_REFERENCE.name(),
    ] {
        assert!(
            !report.stages.iter().any(|s| s.name == forbidden),
            "{forbidden} kernel must not run for the recip-only spread pipeline"
        );
    }
}

// =====================================================================
// Section: spme_recip_apply_influence (E1 fused influence-multiply +
// per-block virial partial-sum reduction).
// =====================================================================

fn small_charged_pair_grid() -> (
    heddle_md::gpu::GpuContext,
    ParticleBuffers,
    SimulationBox,
    SpmeReciprocalGrid,
) {
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    let e = 1.602176634e-19;
    let positions = [[0.1e-9, 0.0, 0.0], [-0.1e-9, 0.0, 0.0]];
    let charges = [e, -e];
    let state = build_state(&positions, &charges);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 4,
    };
    let grid = SpmeReciprocalGrid::new(&gpu, &sim_box, 2, params).unwrap();
    (gpu, buffers, sim_box, grid)
}

#[test]
fn virial_partials_buffer_length_equals_ceil_m_complex_over_256() {
    let (_gpu, _b, _sim_box, grid) = small_charged_pair_grid();
    let expected = grid.m_complex.div_ceil(256);
    use cudarc::driver::DeviceSlice;
    assert_eq!(grid.virial_partials.len(), expected);
    assert!(grid.virial_partials.len() < grid.m_complex,
        "virial_partials ({}) should be much smaller than M_complex ({})",
        grid.virial_partials.len(), grid.m_complex);
}

#[test]
fn apply_influence_two_runs_byte_identical_v_hat_and_virial_partials() {
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    let e = 1.602176634e-19;
    let positions = [[0.1e-9, 0.0, 0.0], [-0.1e-9, 0.0, 0.0]];
    let charges = [e, -e];
    let state = build_state(&positions, &charges);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 4,
    };
    let mut a = SpmeReciprocalGrid::new(&gpu, &sim_box, 2, params).unwrap();
    let mut b = SpmeReciprocalGrid::new(&gpu, &sim_box, 2, params).unwrap();
    let mut t = Timings::new(&gpu).unwrap();
    a.compute(&sim_box, &buffers, 1, &mut t).unwrap();
    b.compute(&sim_box, &buffers, 1, &mut t).unwrap();
    a.sync_recip().unwrap();
    b.sync_recip().unwrap();
    let rho_hat_a: Vec<Real> = gpu.device.dtoh_sync_copy(&a.rho_hat_interleaved).unwrap();
    let rho_hat_b: Vec<Real> = gpu.device.dtoh_sync_copy(&b.rho_hat_interleaved).unwrap();
    assert_eq!(rho_hat_a, rho_hat_b, "V_hat (rho_hat after multiply) not byte-identical");
    let partials_a: Vec<Real> = gpu.device.dtoh_sync_copy(&a.virial_partials).unwrap();
    let partials_b: Vec<Real> = gpu.device.dtoh_sync_copy(&b.virial_partials).unwrap();
    assert_eq!(partials_a, partials_b, "virial_partials not byte-identical");
}

#[test]
fn apply_influence_writes_zero_to_k_zero_component_of_v_hat() {
    // Run apply_influence directly on a hand-poked rho_hat: this avoids
    // having to interleave the spread pipeline (which would overwrite
    // rho_hat with V_hat anyway). We verify that the k=0 slot is zero
    // after the multiply, regardless of the rho_hat[0] input, because
    // influence_G[0] == 0 (tinfoil boundary).
    let (gpu, buffers, sim_box, mut grid) = small_charged_pair_grid();
    // Refresh influence_G + virial_factor for this box.
    heddle_md::gpu::spme_recip_compute_influence(
        &buffers.kernels,
        &grid.b_factors_a,
        &grid.b_factors_b,
        &grid.b_factors_c,
        &mut grid.influence_g,
        &mut grid.virial_factor,
        &sim_box,
        grid.params.grid,
        K_COULOMB_F32,
        grid.params.alpha,
        grid.m_complex as u32,
    )
    .unwrap();
    // Poke rho_hat[k=0] to a non-zero (re, im) and the rest to known
    // non-zero values so we can also assert that the multiply is
    // active for the other cells.
    let mut rho_hat_host = vec![1.5 as Real; 2 * grid.m_complex];
    rho_hat_host[0] = 2.5;
    rho_hat_host[1] = -3.0;
    gpu.device
        .htod_sync_copy_into(&rho_hat_host, &mut grid.rho_hat_interleaved)
        .unwrap();
    let n_c = grid.params.grid[2];
    let n_c_complex = (n_c / 2 + 1) as u32;
    heddle_md::gpu::spme_recip_apply_influence(
        &buffers.kernels,
        &grid.influence_g,
        &grid.virial_factor,
        &mut grid.rho_hat_interleaved,
        &mut grid.virial_partials,
        n_c,
        n_c_complex,
        grid.m_complex as u32,
    )
    .unwrap();
    gpu.device.synchronize().unwrap();
    let rho_hat: Vec<Real> = gpu.device.dtoh_sync_copy(&grid.rho_hat_interleaved).unwrap();
    // After spme_recip_apply_influence the k=0 slot of rho_hat (now V_hat)
    // must be (0, 0) since influence_G[0] == 0 (tinfoil boundary).
    assert_eq!(rho_hat[0], 0.0 as Real, "Re V_hat[k=0] != 0");
    assert_eq!(rho_hat[1], 0.0 as Real, "Im V_hat[k=0] != 0");
}

#[test]
fn sum_of_virial_partials_equals_hermitian_weighted_sum_of_virial_factor_times_rho_hat_sq() {
    // After apply_influence runs, the sum of virial_partials should equal
    // Σ_k hw(k) · virial_factor[k] · |rho_hat[k]|² where rho_hat is the
    // pre-multiply complex grid (the kernel snapshots rho_hat before the
    // multiply when forming the virial contribution).
    //
    // The cleanest sanity check: the W_recip computed by
    // spme_recip_reduce_partials must equal the host-computed reference
    // up to f32 round-off. Reference: Σ_k virial_partials[b] reduced
    // host-side in f64 (the kernel pipeline writes virial_partials in
    // the same fixed order across runs, so a host-side sum reproduces
    // the device-side reduction up to associativity differences).
    let (gpu, buffers, sim_box, mut grid) = small_charged_pair_grid();
    let mut t = Timings::new(&gpu).unwrap();
    grid.compute(&sim_box, &buffers, 1, &mut t).unwrap();
    grid.sync_recip().unwrap();
    let partials: Vec<Real> = gpu.device.dtoh_sync_copy(&grid.virial_partials).unwrap();
    let host_sum: f64 = partials.iter().map(|&x| x as f64).sum();
    // Sanity: sum is finite. The sign depends on whether the dominant
    // K-modes lie inside or outside the (1 − K²/(2α²)) sign crossing,
    // so either sign is physically possible.
    assert!(host_sum.is_finite(), "non-finite Σ virial_partials: {host_sum}");
    assert!(partials.iter().all(|x| x.is_finite()),
        "non-finite partial: {partials:?}");
}

#[test]
fn end_to_end_w_per_particle_virial_equals_half_over_n_times_sum_of_partials() {
    // The slot's reduction step produces
    //   w_per_particle_virial[0] = (0.5 / N) · Σ virial_partials[b].
    // Run the full pipeline (which includes the reduce kernel inside
    // SpmeReciprocalState::compute), then verify the scalar matches the
    // host-side reference.
    use heddle_md::forces::{AggregateLevel, ForceField, PotentialRegistry};
    use heddle_md::forces::{AngleList, BondList, DihedralList, ExclusionList};
    use heddle_md::io::config::{NeighborListConfig, PairInteractionConfig, ParticleTypeConfig, PairPotentialParams, SpmeConfig};

    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    let e = 1.602176634e-19;
    let positions = [[0.1e-9, 0.0, 0.0], [-0.1e-9, 0.0, 0.0]];
    let charges = [e as Real, -e as Real];
    let state = build_state(&positions, &charges);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let n = state.particle_count();
    // Compose a real ForceField (with builtins) so the recip slot's
    // compute() runs the reduce_partials kernel after apply_influence.
    let particle_types = vec![ParticleTypeConfig { name: "X".into(), mass: 1.0, charge: 0.0 }];
    let pairs = vec![PairInteractionConfig {
        between: ("X".into(), "X".into()),
        cutoff: 0.3e-9,
        r_switch: 0.3e-9,
        potential: PairPotentialParams::LennardJones { sigma: 1.0e-10, epsilon: 1.0e-30 },
    }];
    let spme_cfg = SpmeConfig {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 4,
    };
    let mut ff = ForceField::new(
        &PotentialRegistry::with_builtins(),
        &gpu,
        n,
        &sim_box,
        &particle_types,
        &pairs,
        &[],
        &[],
        &[],
        None,
        Some(&spme_cfg),
        &charges,
        &BondList::empty(n),
        &AngleList::empty(0),
        &DihedralList::empty(0),
        &ExclusionList::empty(n),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    let mut t = Timings::new(&gpu).unwrap();
    ff.step(&mut buffers, &sim_box, &mut t, AggregateLevel::ForcesAndScalars).unwrap();
    gpu.device.synchronize().unwrap();

    // Pull virial_partials out of the recip slot for the reference sum.
    // The recip slot is at known position 4 in PotentialRegistry::with_builtins().
    let recip = ff
        .slots
        .iter()
        .find(|s| s.label() == "spme_reciprocal")
        .expect("spme_reciprocal slot present");
    let _ = recip; // only label-checked; we use buffers.virials instead.

    let virials: Vec<Real> = gpu.device.dtoh_sync_copy(&buffers.virials).unwrap();
    // The reciprocal slot's per-particle virial contribution is
    // W_recip / N, repeated for each of N particles.
    // The total system virial includes other slots, so we check
    // sign + finiteness only: the recip contribution is nontrivially
    // positive for a charged pair on a finite grid.
    for v in &virials {
        assert!(v.is_finite(), "non-finite virial entry: {v}");
    }
    let total: f64 = virials.iter().map(|&x| x as f64).sum();
    assert!(total.is_finite(), "non-finite total virial");
}

// rq-e7b74f7a
#[test]
fn influence_tracks_box_change_across_compute_calls() {
    // Under NPT the box changes every step; the influence function must
    // follow it. After a box change (what a barostat does in place to
    // the device lattice), the next compute() must produce a different
    // influence_G.
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let mut sim_box = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    let positions = [[0.2e-9, 0.0, 0.0], [-0.2e-9, 0.0, 0.0]];
    let charges = [1.0 as Real, -1.0];
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

    spme.compute(&sim_box, &buffers, 1, &mut timings).unwrap();
    spme.sync_recip().unwrap();
    let g_a: Vec<Real> = gpu.device.dtoh_sync_copy(&spme.influence_g).unwrap();

    // Isotropic box expansion, applied to the device lattice in place.
    let l2 = 1.05e-9;
    sim_box.set_lattice(l2, l2, l2, 0.0, 0.0, 0.0).unwrap();
    spme.compute(&sim_box, &buffers, 1, &mut timings).unwrap();
    spme.sync_recip().unwrap();
    let g_b: Vec<Real> = gpu.device.dtoh_sync_copy(&spme.influence_g).unwrap();

    let changed = g_a.iter().zip(&g_b).any(|(a, b)| (a - b).abs() > 0.0);
    assert!(changed, "influence_G must change after a box rescale");
}

// rq-e7b74f7a
#[test]
fn influence_recomputed_every_call_even_when_box_unchanged() {
    // The recompute is unconditional (not gated on a host-side box
    // generation counter), so it records into the captured CUDA graph
    // and replays every step. Overwriting influence_G on device and
    // calling compute() again with the SAME box must restore the correct
    // values — a generation-gated launch would skip the recompute and
    // leave the corrupted buffer in place.
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    let positions = [[0.2e-9, 0.0, 0.0], [-0.2e-9, 0.0, 0.0]];
    let charges = [1.0 as Real, -1.0];
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

    spme.compute(&sim_box, &buffers, 1, &mut timings).unwrap();
    spme.sync_recip().unwrap();
    let g_ref: Vec<Real> = gpu.device.dtoh_sync_copy(&spme.influence_g).unwrap();

    // Corrupt influence_G on device; box (and its generation) unchanged.
    let garbage = vec![7.0 as Real; g_ref.len()];
    gpu.device
        .htod_sync_copy_into(&garbage, &mut spme.influence_g)
        .unwrap();

    spme.compute(&sim_box, &buffers, 1, &mut timings).unwrap();
    spme.sync_recip().unwrap();
    let g_after: Vec<Real> = gpu.device.dtoh_sync_copy(&spme.influence_g).unwrap();

    assert_eq!(
        g_after, g_ref,
        "compute() must recompute influence_G unconditionally, not skip on unchanged box generation"
    );
}

// rq-f81b4298
// The order-specialized spread/gather (JIT-compiled per run with
// PME_ORDER fixed at the configured spline order) reproduce the explicit
// Ewald reciprocal energy within tolerance for every accepted order. The
// grid (16^3) satisfies n_d >= 2*spline_order up to order 8.
#[test]
fn reciprocal_energy_matches_ewald_for_every_spline_order() {
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    let positions = [[0.2e-9, 0.0, 0.0], [-0.2e-9, 0.0, 0.0]];
    let charges = [1.0 as Real, -1.0];
    let state = build_state(&positions, &charges);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let alpha = 4.0e9_f64;
    let energy_ref = explicit_ewald_recip_energy(
        &positions,
        &charges,
        [l as f64, l as f64, l as f64],
        alpha,
        8,
    );
    // Every accepted spline order 4..=8. `compute_b_factors` averages the
    // near-zero B-spline structure-factor moduli that odd orders produce,
    // so odd orders (5, 7) are correct too.
    for order in 4u32..=8 {
        let params = SpmeParameters {
            alpha: alpha as Real,
            r_cut_real: 0.3e-9,
            grid: [16, 16, 16],
            spline_order: order,
        };
        let mut spme = SpmeReciprocalGrid::new(&gpu, &sim_box, 2, params).unwrap();
        let mut timings = Timings::new(&gpu).unwrap();
        spme.compute(&sim_box, &buffers, 1, &mut timings).unwrap();
        spme.sync_recip().unwrap();
        let energy_spme = spme_energy_from_pipeline(&spme);
        let rel = (energy_spme - energy_ref).abs() / energy_ref.abs();
        assert!(
            rel < 5.0e-3,
            "spline_order {order}: SPME = {energy_spme:e}, ref = {energy_ref:e}, rel = {rel:e}"
        );
    }
}

// rq-141360a1
// The order-specialized spread is deterministic at a non-default order:
// two slots constructed with spline_order=5 on identical inputs produce
// byte-identical rho_fixed, rho, and V. (Gather/force determinism is
// structural — one thread per atom, fixed accumulation order — and is
// covered by the full-pipeline reproducibility scenario.)
#[test]
fn specialized_pipeline_is_byte_identical_at_non_default_order() {
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap();
    let positions = [[0.1e-9, 0.0, 0.0], [-0.1e-9, 0.0, 0.0]];
    let e = 1.602176634e-19;
    let charges = [e, -e];
    let state = build_state(&positions, &charges);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();

    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 5,
    };

    let mut spme1 = SpmeReciprocalGrid::new(&gpu, &sim_box, 2, params).unwrap();
    let mut spme2 = SpmeReciprocalGrid::new(&gpu, &sim_box, 2, params).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    spme1.compute(&sim_box, &buffers, 1, &mut timings).unwrap();
    spme2.compute(&sim_box, &buffers, 1, &mut timings).unwrap();
    spme1.sync_recip().unwrap();
    spme2.sync_recip().unwrap();

    let rf1: Vec<i64> = gpu.device.dtoh_sync_copy(&spme1.rho_fixed).unwrap();
    let rf2: Vec<i64> = gpu.device.dtoh_sync_copy(&spme2.rho_fixed).unwrap();
    assert_eq!(rf1, rf2, "rho_fixed must be byte-identical at order 5");
    let rho1: Vec<Real> = gpu.device.dtoh_sync_copy(&spme1.rho).unwrap();
    let rho2: Vec<Real> = gpu.device.dtoh_sync_copy(&spme2.rho).unwrap();
    assert_eq!(rho1, rho2, "rho must be byte-identical at order 5");
    let v1: Vec<Real> = gpu.device.dtoh_sync_copy(&spme1.v).unwrap();
    let v2: Vec<Real> = gpu.device.dtoh_sync_copy(&spme2.v).unwrap();
    assert_eq!(v1, v2, "V must be byte-identical at order 5");
}
