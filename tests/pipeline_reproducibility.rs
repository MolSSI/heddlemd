mod common;
use common::*;

use dynamics::gpu::{GpuContext, PairBuffer, ParticleBuffers, init_device, vv_kick, vv_kick_drift};
use dynamics::pbc::SimulationBox;
use dynamics::state::ParticleState;

const N: usize = 64;
const BOX_L: f32 = 8.0;
const LATTICE_SPACING: f32 = 2.0;
const LATTICE_ORIGIN: f32 = -3.0;
const DT: f32 = 0.001;
const SIGMA: f32 = 1.0;
const EPSILON: f32 = 1.0;
const CUTOFF: f32 = 2.5;

// rq-8dfac0eb
fn build_initial_state() -> ParticleState {
    let mut positions_x = Vec::with_capacity(N);
    let mut positions_y = Vec::with_capacity(N);
    let mut positions_z = Vec::with_capacity(N);
    for ix in 0..4 {
        for iy in 0..4 {
            for iz in 0..4 {
                let i = ix * 16 + iy * 4 + iz;
                let fi = i as f32;
                let perturb_x = 0.2 * (fi * 0.7 + 0.0).sin();
                let perturb_y = 0.2 * (fi * 0.7 + 1.1).sin();
                let perturb_z = 0.2 * (fi * 0.7 + 2.3).sin();
                positions_x.push(ix as f32 * LATTICE_SPACING + LATTICE_ORIGIN + perturb_x);
                positions_y.push(iy as f32 * LATTICE_SPACING + LATTICE_ORIGIN + perturb_y);
                positions_z.push(iz as f32 * LATTICE_SPACING + LATTICE_ORIGIN + perturb_z);
            }
        }
    }
    ParticleState::new(
        positions_x,
        positions_y,
        positions_z,
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vec![1.0; N],
        vec![0u32; N],
        None,
            None,
    )
    .expect("build_initial_state: ParticleState::new")
}

// rq-6b6180af
fn run_pipeline(gpu: &GpuContext, n_steps: usize) -> ParticleState {
    let state = build_initial_state();
    let mut buffers = ParticleBuffers::new(gpu, &state).unwrap();
    let mut pair = PairBuffer::new(gpu, N, N as u32).unwrap();
    let counts = gpu.device.htod_sync_copy(&vec![N as u32; N]).unwrap();
    let sim_box = SimulationBox::new(BOX_L, BOX_L, BOX_L, 0.0, 0.0, 0.0).unwrap();
    let params = single_type_lj_table(&gpu.device, SIGMA, EPSILON, CUTOFF);

    // Warm-up: populate forces with F(0) before the first kick_drift consumes them.
    lj_pair_force_no_excl(&buffers, &mut pair, &sim_box, &params).unwrap();
    reduce_pair_forces_into_buffers(&pair, &counts, &mut buffers).unwrap();

    for _ in 0..n_steps {
        vv_kick_drift(&mut buffers, &sim_box, DT).unwrap();
        lj_pair_force_no_excl(&buffers, &mut pair, &sim_box, &params).unwrap();
        reduce_pair_forces_into_buffers(&pair, &counts, &mut buffers).unwrap();
        vv_kick(&mut buffers, DT).unwrap();
    }

    let mut downloaded = state.clone();
    downloaded.download_from(&buffers).unwrap();
    downloaded
}

// rq-24a2b5ef
fn assert_states_byte_identical(a: &ParticleState, b: &ParticleState) {
    assert_eq!(a.positions_x, b.positions_x, "positions_x");
    assert_eq!(a.positions_y, b.positions_y, "positions_y");
    assert_eq!(a.positions_z, b.positions_z, "positions_z");
    assert_eq!(a.velocities_x, b.velocities_x, "velocities_x");
    assert_eq!(a.velocities_y, b.velocities_y, "velocities_y");
    assert_eq!(a.velocities_z, b.velocities_z, "velocities_z");
    assert_eq!(a.forces_x, b.forces_x, "forces_x");
    assert_eq!(a.forces_y, b.forces_y, "forces_y");
    assert_eq!(a.forces_z, b.forces_z, "forces_z");
    assert_eq!(a.masses, b.masses, "masses");
    assert_eq!(a.particle_ids, b.particle_ids, "particle_ids");
}

#[test] // rq-b2314952
fn bit_exact_after_single_full_step() {
    let gpu = init_device().expect("init_device");
    let result_a = run_pipeline(&gpu, 1);
    let result_b = run_pipeline(&gpu, 1);
    assert_states_byte_identical(&result_a, &result_b);
}

#[test] // rq-2846ee8b
fn bit_exact_after_100_step_run() {
    let gpu = init_device().expect("init_device");
    let result_a = run_pipeline(&gpu, 100);
    let result_b = run_pipeline(&gpu, 100);
    assert_states_byte_identical(&result_a, &result_b);
}

#[test] // rq-d0a54b3c
fn positions_visibly_evolve_over_100_step_run() {
    let gpu = init_device().expect("init_device");
    let initial = build_initial_state();
    let result = run_pipeline(&gpu, 100);

    let max_disp = (0..N)
        .map(|i| {
            let dx = result.positions_x[i] - initial.positions_x[i];
            let dy = result.positions_y[i] - initial.positions_y[i];
            let dz = result.positions_z[i] - initial.positions_z[i];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(0.0_f32, f32::max);
    assert!(
        max_disp > 0.001,
        "max displacement after 100 steps was {max_disp} (should exceed 0.001)"
    );
}

#[test] // rq-3f46fb2e
fn all_outputs_finite_after_100_step_run() {
    let gpu = init_device().expect("init_device");
    let result = run_pipeline(&gpu, 100);

    let arrays: [&[f32]; 9] = [
        &result.positions_x,
        &result.positions_y,
        &result.positions_z,
        &result.velocities_x,
        &result.velocities_y,
        &result.velocities_z,
        &result.forces_x,
        &result.forces_y,
        &result.forces_z,
    ];
    for (label, arr) in [
        "positions_x",
        "positions_y",
        "positions_z",
        "velocities_x",
        "velocities_y",
        "velocities_z",
        "forces_x",
        "forces_y",
        "forces_z",
    ]
    .into_iter()
    .zip(arrays.into_iter())
    {
        for (i, &v) in arr.iter().enumerate() {
            assert!(v.is_finite(), "{label}[{i}] = {v} is not finite");
        }
    }
}
