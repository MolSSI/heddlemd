// rq-5e059f6b
//
// Andersen thermostat integrator tests.

use dynamics::forces::{AngleList, BondList, ExclusionList, ForceField};
use dynamics::gpu::{
    GpuContext, ParticleBuffers, andersen_resample, compute_kinetic_energy, init_device,
};
use dynamics::integrator::{AndersenState, Integrator, IntegratorRegistry};
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

fn andersen_kind(temperature: f64, collision_rate: f64, seed: u64) -> IntegratorKind {
    IntegratorKind::Andersen {
        temperature,
        collision_rate,
        seed,
    }
}

// Atomic-scale particle state: 1 amu masses and a few hundred m/s thermal
// velocities so KE ~ kT.
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
        (0..n as u32).collect(),
        None,
        None,
    )
    .unwrap()
}

// --- Construction ---

#[test]
fn registry_builds_andersen() {
    let gpu = init_device().unwrap();
    let kind = andersen_kind(300.0, 1.0e12, 42);
    let _integrator = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 4)
        .unwrap();
}

#[test]
fn registry_builds_andersen_particle_count_zero() {
    let gpu = init_device().unwrap();
    let kind = andersen_kind(300.0, 1.0e12, 1);
    let _integrator = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 0)
        .unwrap();
}

#[test]
fn registry_builds_andersen_collision_rate_zero() {
    let gpu = init_device().unwrap();
    let kind = andersen_kind(300.0, 0.0, 1);
    let _integrator = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 4)
        .unwrap();
}

// --- andersen_resample kernel ---

#[test]
fn andersen_resample_p_zero_is_identity() {
    let gpu = init_device().unwrap();
    let n = 8usize;
    let state = atomic_state(n);
    let snap_vx: Vec<f32> = state.velocities_x.clone();
    let snap_vy: Vec<f32> = state.velocities_y.clone();
    let snap_vz: Vec<f32> = state.velocities_z.clone();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    andersen_resample(&mut buffers, 1, 1, 0.0, (KB * 300.0) as f32).unwrap();
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vz = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
    assert_eq!(vx, snap_vx);
    assert_eq!(vy, snap_vy);
    assert_eq!(vz, snap_vz);
}

#[test]
fn andersen_resample_p_one_replaces_every_particle() {
    let gpu = init_device().unwrap();
    let n = 1024usize;
    // Start with all velocities at 1000 m/s so the comparison is clean.
    let mass: f32 = 1.66e-27;
    let vx_init = vec![1000.0_f32; n];
    let vy_init = vec![1000.0_f32; n];
    let vz_init = vec![1000.0_f32; n];
    let state = ParticleState::new(
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vx_init.clone(),
        vy_init.clone(),
        vz_init.clone(),
        vec![mass; n],
        vec![0.0_f32; n],
        (0..n as u32).collect(),
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let kt = KB * 300.0;
    andersen_resample(&mut buffers, 42, 1, 1.0, kt as f32).unwrap();
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vz = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
    // Every component must differ from the initial value (with overwhelming probability).
    for i in 0..n {
        assert_ne!(vx[i], vx_init[i], "vx[{i}] unchanged");
        assert_ne!(vy[i], vy_init[i], "vy[{i}] unchanged");
        assert_ne!(vz[i], vz_init[i], "vz[{i}] unchanged");
    }
    // Sample variance per axis should match sigma² = kt/m within ~5%.
    let sigma2_target = (kt / mass as f64) as f64;
    for (label, comp) in [("vx", &vx), ("vy", &vy), ("vz", &vz)] {
        let mean: f64 = comp.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var: f64 = comp.iter().map(|&v| ((v as f64) - mean).powi(2)).sum::<f64>() / n as f64;
        let rel = (var - sigma2_target).abs() / sigma2_target;
        assert!(
            rel < 0.1,
            "{label} variance {var:e} vs expected {sigma2_target:e} (rel {rel})"
        );
    }
}

#[test]
fn andersen_resample_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = atomic_state(0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    andersen_resample(&mut buffers, 1, 1, 0.5, 1.0).unwrap();
}

#[test]
fn andersen_resample_deterministic_across_runs() {
    let gpu = init_device().unwrap();
    let n = 64usize;
    let state = atomic_state(n);
    let mut a = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut b = ParticleBuffers::new(&gpu, &state).unwrap();
    let kt = (KB * 300.0) as f32;
    andersen_resample(&mut a, 7, 3, 1.0, kt).unwrap();
    andersen_resample(&mut b, 7, 3, 1.0, kt).unwrap();
    let va = gpu.device.dtoh_sync_copy(&a.velocities_x).unwrap();
    let vb = gpu.device.dtoh_sync_copy(&b.velocities_x).unwrap();
    assert_eq!(va, vb);
}

#[test]
fn andersen_resample_different_seeds_differ() {
    let gpu = init_device().unwrap();
    let n = 64usize;
    let state = atomic_state(n);
    let mut a = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut b = ParticleBuffers::new(&gpu, &state).unwrap();
    let kt = (KB * 300.0) as f32;
    andersen_resample(&mut a, 1, 1, 1.0, kt).unwrap();
    andersen_resample(&mut b, 2, 1, 1.0, kt).unwrap();
    let va = gpu.device.dtoh_sync_copy(&a.velocities_x).unwrap();
    let vb = gpu.device.dtoh_sync_copy(&b.velocities_x).unwrap();
    // At least 90% of components should differ.
    let differs = va.iter().zip(vb.iter()).filter(|(x, y)| x != y).count();
    assert!(
        differs as f64 / n as f64 > 0.9,
        "{differs} of {n} components differ"
    );
}

// --- Per-step kernel sequence ---

#[test]
fn andersen_step_launches_expected_kernels() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut force_field = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let kind = andersen_kind(300.0, 1.0e12, 1);
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
    assert_eq!(count_for(KernelStage::KINETIC_ENERGY_REDUCE), 2);
    assert_eq!(count_for(KernelStage::ANDERSEN_RESAMPLE), 1);
}

#[test]
fn andersen_step_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = atomic_state(0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut force_field = empty_force_field(&gpu, 0);
    let mut timings = Timings::new(&gpu).unwrap();
    let kind = andersen_kind(300.0, 1.0e12, 1);
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

// --- draw_counter and cumulative_injection ---

fn unbox_andersen(integ: Box<dyn Integrator>) -> AndersenState {
    unsafe { *Box::from_raw(Box::into_raw(integ) as *mut AndersenState) }
}

#[test]
fn andersen_draw_counter_increments_per_step() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut force_field = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let kind = andersen_kind(300.0, 1.0e12, 1);
    let mut integ = unbox_andersen(
        IntegratorRegistry::with_builtins()
            .build(&kind, &gpu, n)
            .unwrap(),
    );
    assert_eq!(integ.draw_counter, 0);
    force_field
        .step(&mut buffers, &sim_box, &mut timings)
        .unwrap();
    integ
        .step(
            &mut buffers,
            &mut sim_box,
            &mut force_field,
            1.0e-15,
            &mut timings,
        )
        .unwrap();
    assert_eq!(integ.draw_counter, 1);
    integ
        .step(
            &mut buffers,
            &mut sim_box,
            &mut force_field,
            1.0e-15,
            &mut timings,
        )
        .unwrap();
    assert_eq!(integ.draw_counter, 2);
}

#[test]
fn andersen_cumulative_injection_tracks_ke_change() {
    let gpu = init_device().unwrap();
    let n = 32usize;
    let state = atomic_state(n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut force_field = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    // Use p = 1 so every particle is resampled and the injection is large
    // enough to verify against an independent KE computation.
    let kind = andersen_kind(300.0, 1.0e16, 1);
    let mut integ = unbox_andersen(
        IntegratorRegistry::with_builtins()
            .build(&kind, &gpu, n)
            .unwrap(),
    );
    force_field
        .step(&mut buffers, &sim_box, &mut timings)
        .unwrap();
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

// --- Log columns ---

#[test]
fn andersen_log_column_names_returns_andersen_conserved() {
    let gpu = init_device().unwrap();
    let kind = andersen_kind(300.0, 1.0e12, 1);
    let integ = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 4)
        .unwrap();
    assert_eq!(integ.log_column_names(), &["andersen_conserved"]);
}

#[test]
fn andersen_log_column_values_subtracts_cumulative_injection() {
    let gpu = init_device().unwrap();
    let kind = andersen_kind(300.0, 1.0e12, 1);
    let mut integ = unbox_andersen(
        IntegratorRegistry::with_builtins()
            .build(&kind, &gpu, 4)
            .unwrap(),
    );
    integ.cumulative_injection = 1.0e-20;
    let extras = integ.log_column_values(2.5e-20, 3.0e-20);
    assert_eq!(extras.len(), 1);
    let expected = 2.5e-20 + 3.0e-20 - 1.0e-20;
    assert!((extras[0] - expected).abs() < 1.0e-30);
}

// --- Determinism + temperature tracking ---

#[test]
fn andersen_two_runs_with_identical_inputs_match() {
    let gpu = init_device().unwrap();
    let state = atomic_state(8);

    fn run_once(gpu: &GpuContext, state: &ParticleState) -> Vec<f32> {
        let n = state.particle_count();
        let mut buffers = ParticleBuffers::new(gpu, state).unwrap();
        let mut sim_box = box_large();
        let mut ff = empty_force_field(gpu, n);
        let mut timings = Timings::new(gpu).unwrap();
        let kind = andersen_kind(300.0, 1.0e12, 42);
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
        assert!(v.is_finite(), "non-finite velocity: {a:?}");
    }
}

#[test]
fn andersen_different_seeds_diverge() {
    let gpu = init_device().unwrap();
    let state = atomic_state(8);

    fn run_once(gpu: &GpuContext, state: &ParticleState, seed: u64) -> Vec<f32> {
        let n = state.particle_count();
        let mut buffers = ParticleBuffers::new(gpu, state).unwrap();
        let mut sim_box = box_large();
        let mut ff = empty_force_field(gpu, n);
        let mut timings = Timings::new(gpu).unwrap();
        // High collision rate so the seed actually matters within 3 steps.
        let kind = andersen_kind(300.0, 1.0e16, seed);
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
    assert_ne!(a, b);
}

#[test]
fn andersen_time_averaged_ke_tracks_target() {
    // Strong-coupling Andersen run; the time-averaged KE should equal
    // (3N/2) k_B T to within ~10% given enough averaging samples.
    let gpu = init_device().unwrap();
    let n = 64usize;
    let mass: f32 = 1.66e-27;
    let temperature = 300.0_f64;
    let kt = KB * temperature;
    let k_target = (3.0 * n as f64 / 2.0) * kt;
    // Initial velocities don't matter for an aggressive Andersen run.
    let zero = vec![0.0_f32; n];
    let state = ParticleState::new(
        (0..n).map(|i| (i as f32) * 1.0e-10).collect(),
        zero.clone(),
        zero.clone(),
        vec![100.0_f32; n],
        zero.clone(),
        vec![0.0_f32; n],
        vec![mass; n],
        vec![0.0_f32; n],
        (0..n as u32).collect(),
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    // collision_rate · dt = 0.5 so half the particles are resampled per step;
    // mixing time is a few steps.
    let kind = andersen_kind(temperature, 5.0e14, 11);
    let mut integ = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, n)
        .unwrap();
    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
    let mut scratch = gpu.device.alloc_zeros::<f32>(1).unwrap();
    // Burn-in.
    for _ in 0..200 {
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, 1.0e-15, &mut timings)
            .unwrap();
    }
    // Accumulate over 500 samples.
    let mut sum = 0.0_f64;
    let n_samples = 500;
    for _ in 0..n_samples {
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, 1.0e-15, &mut timings)
            .unwrap();
        sum += compute_kinetic_energy(&buffers, &mut scratch).unwrap() as f64;
    }
    let k_avg = sum / n_samples as f64;
    let rel = (k_avg - k_target).abs() / k_target;
    assert!(
        rel < 0.15,
        "time-averaged K = {k_avg:e}, K_target = {k_target:e}, rel = {rel}"
    );
}
