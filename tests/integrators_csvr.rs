// rq-891232bf
//
// CSVR (Bussi-Donadio-Parrinello) thermostat integrator tests.

use dynamics::forces::{AngleList, BondList, ExclusionList, ForceField};
use dynamics::gpu::{
    GpuContext, ParticleBuffers, compute_kinetic_energy, init_device, lan_ou_step,
};
use dynamics::integrator::{
    CsvrState, Integrator, IntegratorRegistry, philox_4x32_10, philox_normal,
};
use dynamics::io::IntegratorKind;
use dynamics::io::config::NeighborListConfig;
use dynamics::pbc::SimulationBox;
use dynamics::state::ParticleState;
use dynamics::timings::{KernelStage, Timings};

const KB: f64 = 1.380649e-23;

fn box_large() -> SimulationBox {
    SimulationBox::new(1.0e6, 1.0e6, 1.0e6, 0.0, 0.0, 0.0).unwrap()
}

fn empty_force_field(gpu: &GpuContext, n: usize) -> ForceField {
    ForceField::new(
        gpu,
        n,
        &box_large(),
        &[],
        &[],
        &[],
        &[],
        None,
        None,
        &[],
        &BondList::empty(n),
        &AngleList::empty(0),
        &ExclusionList::empty(n),
        &NeighborListConfig::AllPairs,
    )
    .unwrap()
}

fn csvr_kind(temperature: f64, tau: f64, seed: u64) -> IntegratorKind {
    IntegratorKind::Csvr {
        temperature,
        tau,
        seed,
    }
}

// Atomic-scale particle state: 1 amu masses and ~1000 m/s thermal velocities
// so KE ~ kT and CSVR's rescale factor stays well-behaved.
fn atomic_state(n: usize) -> ParticleState {
    let mass: f32 = 1.66e-27;
    let mut vx: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n / 2 {
        let v = 500.0 * ((i as f32) + 1.0);
        vx.push(v);
        vx.push(-v);
    }
    if vx.len() < n {
        vx.push(0.0);
    }
    let zero = vec![0.0_f32; n];
    ParticleState::new(
        (0..n).map(|i| (i as f32) * 1.0e-10).collect(),
        zero.clone(),
        zero.clone(),
        vx,
        zero.clone(),
        zero,
        vec![mass; n],
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap()
}

// --- Construction ---

#[test]
fn registry_builds_csvr() {
    let gpu = init_device().unwrap();
    let kind = csvr_kind(300.0, 1.0e-13, 42);
    let _integrator = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 4)
        .unwrap();
}

#[test]
fn registry_builds_csvr_particle_count_zero() {
    let gpu = init_device().unwrap();
    let kind = csvr_kind(300.0, 1.0e-13, 1);
    let _integrator = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 0)
        .unwrap();
}

// --- Host-side Philox parity with the device-side kernel ---

#[test]
fn host_philox_matches_device_philox() {
    let gpu = init_device().unwrap();
    // We construct a single particle with v = 0 and call lan_ou_step with
    // alpha = 0 and known seed/draw_counter/kt/m. With alpha = 0, the OU
    // update collapses to v[axis] = sigma * xi[axis] where
    //   sigma = sqrt(kt / m), xi = philox_gaussian(seed, draw, pid, axis).
    // The device-side draw uses f32 throughout (cast at the end); we
    // recompute the same Box-Muller cos branch host-side in f64 and cast
    // to f32 to compare exactly.
    let n = 1usize;
    let mass: f32 = 1.0;
    let kt: f32 = 1.0;
    let seed: u64 = 0x1234_5678_9ABC_DEF0;
    let draw: u64 = 7;
    let state = ParticleState::new(
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![mass; n],
        vec![0.0_f32; n],
        vec![0u32; n], // particle_id = 0
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    lan_ou_step(&mut buffers, seed, draw, 0.0_f32, kt).unwrap();
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vz = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();

    // Host-side reproduction of the device's draw.
    let seed_lo = seed as u32;
    let seed_hi = (seed >> 32) as u32;
    let draw_lo = draw as u32;
    let draw_hi = (draw >> 32) as u32;
    let host_xi = |axis: u32| -> f32 {
        philox_normal(seed_lo, seed_hi, draw_lo, draw_hi, 0, axis) as f32
    };
    let sigma = (kt / mass).sqrt();
    let expected_vx = sigma * host_xi(0);
    let expected_vy = sigma * host_xi(1);
    let expected_vz = sigma * host_xi(2);
    assert_eq!(
        vx[0].to_bits(),
        expected_vx.to_bits(),
        "x: host {expected_vx} vs device {}",
        vx[0]
    );
    assert_eq!(vy[0].to_bits(), expected_vy.to_bits());
    assert_eq!(vz[0].to_bits(), expected_vz.to_bits());
}

#[test]
fn philox_is_pure_function() {
    let a = philox_4x32_10(1, 2, 3, 4, 5, 6);
    let b = philox_4x32_10(1, 2, 3, 4, 5, 6);
    assert_eq!(a, b);
    let c = philox_4x32_10(1, 2, 3, 4, 5, 7);
    assert_ne!(a, c, "different counter must yield different output");
}

// --- Per-step kernel sequence ---

#[test]
fn csvr_step_launches_expected_kernels() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut force_field = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();

    let kind = csvr_kind(300.0, 1.0e-13, 1);
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, n)
        .unwrap();
    force_field
        .step(&mut buffers, &sim_box, &mut timings)
        .unwrap();

    integrator
        .step(
            &mut buffers,
            &mut sim_box,
            &mut force_field,
            1.0e-15,
            &mut timings,
        )
        .unwrap();

    let report = timings.finalize().unwrap();
    let count_for = |stage: KernelStage| -> u64 {
        report
            .stages
            .iter()
            .find(|r| r.name == stage.name())
            .map(|r| r.count)
            .unwrap_or(0)
    };
    assert_eq!(count_for(KernelStage::VV_KICK_DRIFT), 1);
    assert_eq!(count_for(KernelStage::VV_KICK), 1);
    assert_eq!(count_for(KernelStage::KINETIC_ENERGY_REDUCE), 1);
    assert_eq!(count_for(KernelStage::CSVR_RESCALE_VELOCITIES), 1);
}

#[test]
fn csvr_step_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = atomic_state(0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut force_field = empty_force_field(&gpu, 0);
    let mut timings = Timings::new(&gpu).unwrap();
    let kind = csvr_kind(300.0, 1.0e-13, 1);
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 0)
        .unwrap();
    integrator
        .step(
            &mut buffers,
            &mut sim_box,
            &mut force_field,
            1.0e-15,
            &mut timings,
        )
        .unwrap();
}

// --- draw_counter advances ---

#[test]
fn csvr_draw_counter_increments_per_step() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut force_field = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let kind = csvr_kind(300.0, 1.0e-13, 1);
    let boxed = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, n)
        .unwrap();
    let mut state = unsafe { *Box::from_raw(Box::into_raw(boxed) as *mut CsvrState) };
    assert_eq!(state.draw_counter, 0);
    force_field
        .step(&mut buffers, &sim_box, &mut timings)
        .unwrap();
    state
        .step(
            &mut buffers,
            &mut sim_box,
            &mut force_field,
            1.0e-15,
            &mut timings,
        )
        .unwrap();
    assert_eq!(state.draw_counter, 1);
    state
        .step(
            &mut buffers,
            &mut sim_box,
            &mut force_field,
            1.0e-15,
            &mut timings,
        )
        .unwrap();
    assert_eq!(state.draw_counter, 2);
}

// --- Log columns ---

#[test]
fn csvr_log_column_names_returns_csvr_conserved() {
    let gpu = init_device().unwrap();
    let kind = csvr_kind(300.0, 1.0e-13, 1);
    let integrator = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 4)
        .unwrap();
    assert_eq!(integrator.log_column_names(), &["csvr_conserved"]);
}

#[test]
fn csvr_log_column_values_subtracts_cumulative_injection() {
    let gpu = init_device().unwrap();
    let kind = csvr_kind(300.0, 1.0e-13, 1);
    let boxed = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 4)
        .unwrap();
    let mut state = unsafe { *Box::from_raw(Box::into_raw(boxed) as *mut CsvrState) };
    state.cumulative_injection = 1.0e-20;
    let extras = state.log_column_values(2.5e-20, 3.0e-20);
    assert_eq!(extras.len(), 1);
    let expected = 2.5e-20 + 3.0e-20 - 1.0e-20;
    assert!((extras[0] - expected).abs() < 1.0e-30);
}

// --- Cumulative injection updates during step ---

#[test]
fn csvr_cumulative_injection_tracks_kinetic_energy_changes() {
    let gpu = init_device().unwrap();
    let n = 8usize;
    let state = atomic_state(n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut force_field = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let kind = csvr_kind(300.0, 1.0e-13, 1);
    let boxed = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, n)
        .unwrap();
    let mut integ = unsafe { *Box::from_raw(Box::into_raw(boxed) as *mut CsvrState) };
    force_field
        .step(&mut buffers, &sim_box, &mut timings)
        .unwrap();

    // Capture KE before and after the step; the cumulative injection
    // should equal the change in KE (since no forces act in this
    // empty-force-field setup, every KE change comes from the
    // thermostat).
    let mut scratch = gpu.device.alloc_zeros::<f32>(1).unwrap();
    let k_before = compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    integ
        .step(
            &mut buffers,
            &mut sim_box,
            &mut force_field,
            1.0e-15,
            &mut timings,
        )
        .unwrap();
    let k_after = compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    let expected = k_after - k_before;
    let rel = (integ.cumulative_injection - expected).abs() / expected.abs().max(1.0e-30);
    assert!(
        rel < 1.0e-4,
        "cumulative_injection = {}, K_after - K_before = {} (rel {})",
        integ.cumulative_injection,
        expected,
        rel
    );
}

// --- End-to-end determinism + COM preservation ---

#[test]
fn csvr_two_runs_with_identical_inputs_match() {
    let gpu = init_device().unwrap();
    let state = atomic_state(8);

    fn run_once(gpu: &GpuContext, state: &ParticleState) -> Vec<f32> {
        let n = state.particle_count();
        let mut buffers = ParticleBuffers::new(gpu, state).unwrap();
        let mut sim_box = box_large();
        let mut ff = empty_force_field(gpu, n);
        let mut timings = Timings::new(gpu).unwrap();
        let kind = csvr_kind(300.0, 1.0e-13, 42);
        let mut integ = IntegratorRegistry::with_builtins()
            .build(&kind, gpu, n)
            .unwrap();
        ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
        for _ in 0..5 {
            integ
                .step(
                    &mut buffers,
                    &mut sim_box,
                    &mut ff,
                    1.0e-15,
                    &mut timings,
                )
                .unwrap();
        }
        gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap()
    }

    let a = run_once(&gpu, &state);
    let b = run_once(&gpu, &state);
    assert_eq!(a, b);
    for v in &a {
        assert!(v.is_finite(), "NaN/Inf in CSVR trajectory: {a:?}");
    }
}

#[test]
fn csvr_different_seeds_produce_different_trajectories() {
    let gpu = init_device().unwrap();
    let state = atomic_state(8);

    fn run_once(gpu: &GpuContext, state: &ParticleState, seed: u64) -> Vec<f32> {
        let n = state.particle_count();
        let mut buffers = ParticleBuffers::new(gpu, state).unwrap();
        let mut sim_box = box_large();
        let mut ff = empty_force_field(gpu, n);
        let mut timings = Timings::new(gpu).unwrap();
        let kind = csvr_kind(300.0, 1.0e-13, seed);
        let mut integ = IntegratorRegistry::with_builtins()
            .build(&kind, gpu, n)
            .unwrap();
        ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
        for _ in 0..3 {
            integ
                .step(
                    &mut buffers,
                    &mut sim_box,
                    &mut ff,
                    1.0e-15,
                    &mut timings,
                )
                .unwrap();
        }
        gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap()
    }

    let a = run_once(&gpu, &state, 1);
    let b = run_once(&gpu, &state, 2);
    assert_ne!(a, b, "seed should change the trajectory");
}

#[test]
fn csvr_preserves_com_momentum() {
    let gpu = init_device().unwrap();
    let n = 16usize;
    let state = atomic_state(n);
    let mass = 1.66e-27_f32;
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let kind = csvr_kind(300.0, 1.0e-13, 7);
    let mut integ = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, n)
        .unwrap();
    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
    for _ in 0..20 {
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, 1.0e-15, &mut timings)
            .unwrap();
    }
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let p_com: f64 = vx.iter().map(|&v| (mass as f64) * (v as f64)).sum();
    let scale: f32 = vx.iter().map(|v| v.abs()).fold(0.0, f32::max);
    // CSVR is a uniform-scalar rescale and preserves COM momentum
    // exactly in real arithmetic; f32 round-off bounds the residual.
    let tol = (mass as f64) * (scale as f64) * 1.0e-3;
    assert!(
        p_com.abs() < tol,
        "p_com = {p_com} (tol {tol}); v = {vx:?}"
    );
}

#[test]
fn csvr_time_averaged_ke_tracks_k_target() {
    // Equilibrium test: start the system AT the target KE and verify
    // that the time-averaged KE over a long run stays close. CSVR
    // ensures the canonical distribution after the chain mixes; the
    // instantaneous KE has variance K_target² · 2/N_f. For N_f = 93
    // and ~250 averaging samples, the standard error is ~1.5% — a 15%
    // tolerance is generous (~10σ) and won't be flaky.
    let gpu = init_device().unwrap();
    let n = 32usize;
    let mass: f32 = 1.66e-27;
    let temperature = 300.0_f64;
    let kt = KB * temperature;
    let g_dof = (3 * n - 3) as f64;
    let k_target = (g_dof / 2.0) * kt;
    let v_each = ((k_target / ((n as f64) * 0.5 * (mass as f64))) as f64).sqrt() as f32;
    let mut vx: Vec<f32> = Vec::with_capacity(n);
    for _ in 0..n / 2 {
        vx.push(v_each);
        vx.push(-v_each);
    }
    let zero = vec![0.0_f32; n];
    let state = ParticleState::new(
        (0..n).map(|i| (i as f32) * 1.0e-10).collect(),
        zero.clone(),
        zero.clone(),
        vx,
        zero.clone(),
        zero,
        vec![mass; n],
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    // Use τ = 10·dt so the chain mixes briskly within a few hundred
    // steps; otherwise transient bias would dominate the time average.
    let kind = csvr_kind(temperature, 1.0e-14, 11);
    let mut integ = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, n)
        .unwrap();
    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
    let mut scratch = gpu.device.alloc_zeros::<f32>(1).unwrap();
    // Burn-in.
    for _ in 0..100 {
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, 1.0e-15, &mut timings)
            .unwrap();
    }
    // Accumulate.
    let mut sum = 0.0_f64;
    let n_samples = 250;
    for _ in 0..n_samples {
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, 1.0e-15, &mut timings)
            .unwrap();
        sum += compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    }
    let k_avg = sum / (n_samples as f64);
    let rel = (k_avg - k_target).abs() / k_target;
    assert!(
        rel < 0.15,
        "time-averaged K = {k_avg:e}, K_target = {k_target:e}, rel = {rel}"
    );
}
