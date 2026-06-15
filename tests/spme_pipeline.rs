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
    let sim_box = SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap();
    let positions = [[0.2e-9, 0.0, 0.0], [-0.2e-9, 0.0, 0.0]];
    let e_charge = 1.602176634e-19;
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
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap();
    let positions = [
        [0.1e-9, 0.0, 0.0],
        [-0.1e-9, 0.2e-9, 0.0],
        [0.0, -0.2e-9, 0.15e-9],
        [-0.15e-9, 0.1e-9, -0.1e-9],
    ];
    let e_charge = 1.602176634e-19;
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
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap();
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
    spme.compute(&sim_box, &buffers, &mut timings).unwrap();
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
    let g: Vec<Real> = gpu.device.dtoh_sync_copy(&spme.influence_g).unwrap();
    assert_eq!(g[0], 0.0, "influence_G[0] must be zero (tinfoil BC)");
}

// rq-09d4e13f
#[test]
fn identical_inputs_produce_byte_identical_grids() {
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap();
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
    spme1.compute(&sim_box, &buffers, &mut timings).unwrap();
    spme2.compute(&sim_box, &buffers, &mut timings).unwrap();
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
    let sim_box = SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap();
    let grid = SpmeReciprocalGrid::new(&gpu, &sim_box, n, spme).unwrap();
    (gpu, sim_box, grid)
}

// rq-3c0beda9
#[test]
fn spread_for_one_isolated_particle_matches_b_spline_weights() {
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap();
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
    grid.compute(&sim_box, &buffers, &mut t).unwrap();
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
    let sim_box = SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap();
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
    grid.compute(&sim_box, &buffers, &mut t).unwrap();
    grid.sync_recip().unwrap();
    let rho: Vec<Real> = gpu.device.dtoh_sync_copy(&grid.rho).unwrap();
    assert_eq!(rho[0], 0.0, "rho at unsupported grid point must be exactly zero");
}

// rq-07297467
#[test]
fn spread_is_byte_identical_under_two_input_orderings_in_same_bin() {
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap();
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
    grid_a.compute(&sim_box, &buffers_a, &mut t).unwrap();
    grid_b.compute(&sim_box, &buffers_b, &mut t).unwrap();
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
fn spme_reciprocal_internal_cell_list_uses_one_bin_per_fft_grid_cell() {
    use cudarc::driver::DeviceSlice;
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap();
    let params = SpmeParameters {
        alpha: 4.0e9,
        r_cut_real: 0.3e-9,
        grid: [16, 16, 16],
        spline_order: 4,
    };
    let grid = SpmeReciprocalGrid::new(&gpu, &sim_box, 1, params).unwrap();
    assert!(grid.bin_list.is_bin_only());
    let cl = grid
        .bin_list
        .cell_list_data()
        .expect("CellListOnly should expose cell-list data");
    assert_eq!(cl.n_cells, [16, 16, 16]);
    assert_eq!(grid.bin_list.max_neighbors, 0);
    assert_eq!(grid.bin_list.neighbor_list.len(), 0);
    assert_eq!(grid.bin_list.neighbor_counts.len(), 0);
}

// rq-dd829afb
#[test]
fn bin_only_cell_list_rebuilds_every_step_regardless_of_displacement() {
    use heddle_md::timings::{HostStage, KernelStage};
    let gpu = init_device().unwrap();
    let l = 1.0e-9;
    let sim_box = SimulationBox::new(l, l, l, 0.0, 0.0, 0.0).unwrap();
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
    // Two back-to-back pipeline runs with no position change.
    grid.compute(&sim_box, &buffers, &mut t).unwrap();
    grid.compute(&sim_box, &buffers, &mut t).unwrap();
    grid.sync_recip().unwrap();
    let report = t.finalize().unwrap();
    // The displacement-check + neighbor_list_build kernels must never
    // fire (they are absent in CellListOnly mode); each pre_step call
    // must perform a fresh rebuild (host-side stage).
    for forbidden in [
        KernelStage::NEIGHBOR_DISPLACEMENT_SQUARED.name(),
        KernelStage::NEIGHBOR_LIST_BUILD.name(),
    ] {
        assert!(
            !report.stages.iter().any(|s| s.name == forbidden),
            "{forbidden} kernel must not run in bin-only mode"
        );
    }
    let rebuild = report
        .stages
        .iter()
        .find(|s| s.name == HostStage::NEIGHBOR_LIST_REBUILD.name())
        .expect("neighbor_list_rebuild host stage should be present");
    assert_eq!(rebuild.count, 2, "expected one rebuild per pipeline call");
}
