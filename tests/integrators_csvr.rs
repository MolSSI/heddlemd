// rq-891232bf
//
// CSVR (Bussi-Donadio-Parrinello) thermostat tests. The thermostat is
// exercised in isolation through its `apply_post` hook.

use heddle_md::forces::{AggregateLevel, AngleList, BondList, ExclusionList, ForceField, PotentialRegistry};
use heddle_md::gpu::{
    GpuContext, ParticleBuffers, compute_kinetic_energy, init_device, lan_ou_step,
};
use heddle_md::integrator::IntegratorStepExt;
use heddle_md::integrator::{
    CsvrThermostat, Thermostat, ThermostatRegistry, philox_4x32_10, philox_normal,
};
use heddle_md::io::SlotConfig;
use heddle_md::io::config::NeighborListConfig;
use heddle_md::pbc::SimulationBox;
use heddle_md::state::ParticleState;
use heddle_md::timings::{KernelStage, Timings};
use heddle_md::precision::Real;

// k_B = 1 inside the engine; this SI value is retained as a reference
// for converting human-readable SI inputs to atomic units.
#[allow(dead_code)]
const KB: f64 = 1.380649e-23;
const LEN_F: f64 = 5.29177210903e-11;
const MASS_F: f64 = 9.1093837015e-31;
const TIME_F: f64 = 2.4188843265857195e-17;
const TEMP_F: f64 = 315775.0248040668;
const VEL_F: f64 = 2187691.2636411153;

fn box_large(gpu: &heddle_md::gpu::GpuContext) -> SimulationBox {
    let l = (1.0e6 / LEN_F) as Real;
    SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap()
}

fn empty_force_field(gpu: &GpuContext, n: usize) -> ForceField {
    ForceField::new(
        &PotentialRegistry::with_builtins(),
        gpu,
        n,
        &box_large(&gpu),
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

fn csvr_kind(temperature: f64, tau: f64, seed: u64) -> SlotConfig {
    // Convert human-readable SI inputs (K, s) to atomic units.
    let temperature = temperature / TEMP_F;
    let tau = tau / TIME_F;
    SlotConfig::from_params_str(
        "csvr",
        &format!("temperature = {temperature:e}\ntau = {tau:e}\nseed = {seed}\n"),
    )
}

fn build_csvr(gpu: &GpuContext, n: usize, slot: &SlotConfig) -> Box<dyn Thermostat> {
    ThermostatRegistry::with_builtins()
        .build_optional(Some(slot), gpu, n, 0)
        .unwrap()
        .unwrap()
}

fn unbox_csvr(boxed: Box<dyn Thermostat>) -> CsvrThermostat {
    unsafe { *Box::from_raw(Box::into_raw(boxed) as *mut CsvrThermostat) }
}

fn atomic_state(n: usize) -> ParticleState {
    // Atomic-unit values: mass in m_e, velocity in Bohr/(hbar/E_h),
    // position in Bohr.
    let mass: Real = (1.66e-27 / MASS_F) as Real;
    let mut vx: Vec<Real> = Vec::with_capacity(n);
    for i in 0..n / 2 {
        let v = (500.0 / VEL_F) as Real * ((i as Real) + 1.0);
        vx.push(v);
        vx.push(-v);
    }
    if vx.len() < n {
        vx.push(0.0);
    }
    let zero = vec![0.0; n];
    ParticleState::new(
        (0..n).map(|i| (i as Real) * (1.0e-10 / LEN_F) as Real).collect(),
        zero.clone(),
        zero.clone(),
        vx,
        zero.clone(),
        zero,
        vec![mass; n],
        vec![0.0; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap()
}

// --- Construction ---

// rq-a6cd03aa
#[test]
fn registry_builds_csvr() {
    let gpu = init_device().unwrap();
    let kind = csvr_kind(300.0, 1.0e-13, 42);
    let therm = unbox_csvr(build_csvr(&gpu, 4, &kind));
    assert_eq!(therm.draw_counter, 0);
    assert_eq!(therm.cumulative_injection, 0.0);
    assert_eq!(therm.g_dof, 9);
    // The csvr_kind helper converts SI inputs; the engine stores
    // kt_target = temperature directly (k_B = 1 in atomic units).
    assert!((therm.kt_target - 300.0 / TEMP_F).abs() < 1.0e-30);
}

// rq-b5089af4
#[test]
fn registry_builds_csvr_particle_count_zero() {
    let gpu = init_device().unwrap();
    let kind = csvr_kind(300.0, 1.0e-13, 1);
    let _therm = build_csvr(&gpu, 0, &kind);
}

// rq-7326a2d5
#[test]
fn registry_builds_csvr_particle_count_one() {
    let gpu = init_device().unwrap();
    let kind = csvr_kind(300.0, 1.0e-13, 1);
    let therm = unbox_csvr(build_csvr(&gpu, 1, &kind));
    assert_eq!(therm.g_dof, 1);
}

// --- Host-side Philox parity with the device-side kernel ---

// rq-11a953dc
#[test]
fn host_philox_matches_device_philox() {
    let gpu = init_device().unwrap();
    let n = 1usize;
    let mass: Real = 1.0;
    let kt: Real = 1.0;
    let seed: u64 = 0x1234_5678_9ABC_DEF0;
    let draw: u64 = 7;
    let state = ParticleState::new(
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![mass; n],
        vec![0.0; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut counter: cudarc::driver::CudaSlice<u64> =
        gpu.device.alloc_zeros::<u64>(1).unwrap();
    gpu.device.htod_sync_copy_into(&[draw], &mut counter).unwrap();
    lan_ou_step(&mut buffers, &mut counter, seed, 0.0, kt).unwrap();
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vz = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();

    let seed_lo = seed as u32;
    let seed_hi = (seed >> 32) as u32;
    let draw_lo = draw as u32;
    let draw_hi = (draw >> 32) as u32;
    let host_xi = |axis: u32| -> Real {
        philox_normal(seed_lo, seed_hi, draw_lo, draw_hi, 0, axis) as Real
    };
    let sigma = (kt / mass).sqrt();
    let expected_vx = sigma * host_xi(0);
    let expected_vy = sigma * host_xi(1);
    let expected_vz = sigma * host_xi(2);
    // Bit-exact equality holds in the default (f32) build because the
    // cast to Real == f32 lops off the precision delta between Rust's
    // f64 cos and CUDA's f64 cos. In the f64 build the kernel keeps
    // double precision and the two implementations differ by at most a
    // few ULPs.
    #[cfg(not(feature = "f64"))]
    {
        assert_eq!(vx[0].to_bits(), expected_vx.to_bits());
        assert_eq!(vy[0].to_bits(), expected_vy.to_bits());
        assert_eq!(vz[0].to_bits(), expected_vz.to_bits());
    }
    #[cfg(feature = "f64")]
    {
        let tol = 4.0 * Real::EPSILON * sigma.max(1.0);
        assert!((vx[0] - expected_vx).abs() <= tol);
        assert!((vy[0] - expected_vy).abs() <= tol);
        assert!((vz[0] - expected_vz).abs() <= tol);
    }
}

// rq-db1298bd
#[test]
fn philox_is_pure_function() {
    let a = philox_4x32_10(1, 2, 3, 4, 5, 6);
    let b = philox_4x32_10(1, 2, 3, 4, 5, 6);
    assert_eq!(a, b);
    let c = philox_4x32_10(1, 2, 3, 4, 5, 7);
    assert_ne!(a, c);
}

// --- Per-step kernel sequence ---

// rq-4e9e09f0
#[test]
fn csvr_apply_post_launches_expected_kernels() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = build_csvr(&gpu, n, &csvr_kind(300.0, 1.0e-13, 1));
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
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
    assert_eq!(count_for(KernelStage::KINETIC_ENERGY_REDUCE), 1);
    // The per-particle velocity rescale is dispatched by the
    // JIT-composed post-force per-particle kernel; the standalone
    // `CSVR_RESCALE_VELOCITIES` stage is not recorded.
    assert_eq!(count_for(KernelStage::CSVR_RESCALE_VELOCITIES), 0);
    assert_eq!(count_for(KernelStage::VV_KICK_DRIFT), 0);
    assert_eq!(count_for(KernelStage::VV_KICK), 0);
}

// rq-5f59fa80
// Multi-block CSVR path (g_dof > SINGLE_BLOCK_CSVR_MAX = 8192): the
// parallel draw + deterministic two-pass reduction must produce a
// byte-identical rescale factor across two independent runs with
// identical inputs.
#[test]
fn csvr_multi_block_sample_is_deterministic() {
    let gpu = init_device().unwrap();
    let n = 4000usize; // g_dof = 3n - 3 = 11997 > 8192 -> multi-block
    let mass = 1.0 as Real;
    // Non-zero velocities so k_old > 0 and the factor depends on the
    // sampled s = Σ xi².
    let vx: Vec<Real> = (0..n).map(|i| 0.001 + (i as Real) * 1.0e-7).collect();
    let make_state = || {
        ParticleState::new(
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vx.clone(),
            vec![0.002 as Real; n],
            vec![0.003 as Real; n],
            vec![mass; n],
            vec![0.0; n],
            vec![0u32; n],
            None,
            None,
        )
        .unwrap()
    };
    let dt = (1.0e-15 / TIME_F) as Real;
    let run = || -> u32 {
        let state = make_state();
        let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
        let mut timings = Timings::new(&gpu).unwrap();
        let mut therm = unbox_csvr(build_csvr(&gpu, n, &csvr_kind(300.0, 1.0e-13, 7)));
        therm.apply_post(&mut buffers, dt, &mut timings).unwrap();
        let f = gpu.device.dtoh_sync_copy(&therm.factor_device).unwrap();
        f[0].to_bits()
    };
    let a = run();
    let b = run();
    assert_eq!(
        a, b,
        "multi-block CSVR factor must be byte-identical across runs"
    );
}

// rq-a2454a72
#[test]
fn csvr_apply_post_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = atomic_state(0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = build_csvr(&gpu, 0, &csvr_kind(300.0, 1.0e-13, 1));
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
        .unwrap();
}

// rq-d1f1b53e
#[test]
fn csvr_apply_pre_is_trait_default_noop() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let snap_vx = state.velocities_x.clone();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = build_csvr(&gpu, n, &csvr_kind(300.0, 1.0e-13, 1));
    therm
        .apply_pre(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
        .unwrap();
    let vx_after = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    assert_eq!(vx_after, snap_vx);
    let report = timings.finalize().unwrap();
    for s in &report.stages {
        assert_eq!(s.count, 0, "apply_pre launched kernel {:?}", s.name);
    }
}

// --- draw_counter advances ---

// rq-1e5dcdc9 rq-b2d5886a
#[test]
fn csvr_draw_counter_increments_per_apply_post() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = unbox_csvr(build_csvr(&gpu, n, &csvr_kind(300.0, 1.0e-13, 1)));
    assert_eq!(therm.draw_counter, 0);
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
        .unwrap();
    therm.flush_pending_injection(&gpu.device).unwrap();
    assert_eq!(therm.draw_counter, 1);
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
        .unwrap();
    therm.flush_pending_injection(&gpu.device).unwrap();
    assert_eq!(therm.draw_counter, 2);
}

// rq-dc95802b
#[test]
fn csvr_two_thermostats_at_same_counter_produce_identical_velocities() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = atomic_state(n);
    let mut buffers_a = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut buffers_b = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings_a = Timings::new(&gpu).unwrap();
    let mut timings_b = Timings::new(&gpu).unwrap();
    let mut therm_a = build_csvr(&gpu, n, &csvr_kind(300.0, 1.0e-13, 7));
    let mut therm_b = build_csvr(&gpu, n, &csvr_kind(300.0, 1.0e-13, 7));
    therm_a
        .apply_post(&mut buffers_a, (1.0e-15 / TIME_F) as Real, &mut timings_a)
        .unwrap();
    therm_b
        .apply_post(&mut buffers_b, (1.0e-15 / TIME_F) as Real, &mut timings_b)
        .unwrap();
    let va = gpu.device.dtoh_sync_copy(&buffers_a.velocities_x).unwrap();
    let vb = gpu.device.dtoh_sync_copy(&buffers_b.velocities_x).unwrap();
    assert_eq!(va, vb);
}

// --- Log columns ---

// rq-2c1bb918 rq-1d25a0db
#[test]
fn csvr_log_column_names_returns_csvr_conserved() {
    let gpu = init_device().unwrap();
    let kind = csvr_kind(300.0, 1.0e-13, 1);
    let therm = build_csvr(&gpu, 4, &kind);
    let names: Vec<&str> = therm.log_column_names().iter().map(|(n, _)| *n).collect();
    assert_eq!(names, vec!["csvr_conserved"]);
}

// rq-ca0b98cb
#[test]
fn csvr_log_column_values_subtracts_cumulative_injection() {
    let gpu = init_device().unwrap();
    let mut therm = unbox_csvr(build_csvr(&gpu, 4, &csvr_kind(300.0, 1.0e-13, 1)));
    therm.cumulative_injection = 1.0e-20;
    let extras = therm.log_column_values(2.5e-20, 3.0e-20);
    assert_eq!(extras.len(), 1);
    let expected: f64 = 2.5e-20 + 3.0e-20 - 1.0e-20;
    assert!((extras[0] - expected).abs() < 1.0e-30);
}

// --- Cumulative injection updates during apply_post ---

// rq-11b0deff
#[test]
fn csvr_cumulative_injection_tracks_kinetic_energy_changes() {
    let gpu = init_device().unwrap();
    let n = 8usize;
    let state = atomic_state(n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    use heddle_md::gpu::rescale_velocities_device_factor;
    let mut therm = unbox_csvr(build_csvr(&gpu, n, &csvr_kind(300.0, 1.0e-13, 1)));
    let mut scratch = gpu.device.alloc_zeros::<Real>(1).unwrap();
    let k_before = compute_kinetic_energy(&mut buffers, &mut scratch).unwrap() as f64;
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
        .unwrap();
    // The composed post-force per-particle kernel applies the rescale
    // in production. Tests that bypass the composed kernel dispatch
    // the standalone equivalent against `factor_device`.
    rescale_velocities_device_factor(&mut buffers, &therm.factor_device).unwrap();
    // Device-side `(k_new - k_old)` accumulator is updated by
    // `apply_post`. Drain it into `therm.cumulative_injection` before
    // reading; the runner does the same before each log row.
    therm.flush_pending_injection(&gpu.device).unwrap();
    let k_after = compute_kinetic_energy(&mut buffers, &mut scratch).unwrap() as f64;
    let expected = k_after - k_before;
    let rel = (therm.cumulative_injection - expected).abs() / expected.abs().max(1.0e-30);
    assert!(rel < 1.0e-4);
}

// --- End-to-end determinism + COM preservation ---

// rq-dc51e1c3
#[test]
fn csvr_two_runs_with_identical_inputs_match() {
    let gpu = init_device().unwrap();
    let state = atomic_state(8);

    fn run_once(gpu: &GpuContext, state: &ParticleState) -> Vec<Real> {
        let n = state.particle_count();
        let mut buffers = ParticleBuffers::new(gpu, state).unwrap();
        let mut timings = Timings::new(gpu).unwrap();
        let mut therm = build_csvr(gpu, n, &csvr_kind(300.0, 1.0e-13, 42));
        for _ in 0..5 {
            therm
                .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
                .unwrap();
        }
        gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap()
    }

    let a = run_once(&gpu, &state);
    let b = run_once(&gpu, &state);
    assert_eq!(a, b);
    for v in &a {
        assert!(v.is_finite());
    }
}

// rq-94a43204
#[test]
fn csvr_different_seeds_produce_different_trajectories() {
    let gpu = init_device().unwrap();
    let state = atomic_state(8);

    fn run_once(gpu: &GpuContext, state: &ParticleState, seed: u64) -> Vec<Real> {
        use heddle_md::gpu::rescale_velocities_device_factor;
        let n = state.particle_count();
        let mut buffers = ParticleBuffers::new(gpu, state).unwrap();
        let mut timings = Timings::new(gpu).unwrap();
        let mut therm = unbox_csvr(build_csvr(gpu, n, &csvr_kind(300.0, 1.0e-13, seed)));
        let dt = (1.0e-15 / TIME_F) as Real;
        for _ in 0..3 {
            therm.apply_post(&mut buffers, dt, &mut timings).unwrap();
            rescale_velocities_device_factor(&mut buffers, &therm.factor_device).unwrap();
        }
        gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap()
    }

    let a = run_once(&gpu, &state, 1);
    let b = run_once(&gpu, &state, 2);
    assert_ne!(a, b);
}

// rq-287e8d41
#[test]
fn csvr_preserves_com_momentum() {
    let gpu = init_device().unwrap();
    let n = 16usize;
    let state = atomic_state(n);
    let mass = 1.66e-27;
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = build_csvr(&gpu, n, &csvr_kind(300.0, 1.0e-13, 7));
    for _ in 0..20 {
        therm
            .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
            .unwrap();
    }
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let p_com: f64 = vx.iter().map(|&v| (mass as f64) * (v as f64)).sum();
    let scale: Real = vx.iter().map(|v| v.abs()).fold(0.0, Real::max);
    let tol = (mass as f64) * (scale as f64) * 1.0e-3;
    assert!(p_com.abs() < tol, "p_com = {p_com} (tol {tol})");
}

// rq-f70f7c1e
#[test]
fn csvr_time_averaged_ke_tracks_k_target() {
    let gpu = init_device().unwrap();
    let n = 32usize;
    // Atomic-unit values: mass in m_e, k_B = 1 so kt = T.
    let mass: Real = (1.66e-27 / MASS_F) as Real;
    let temperature_si = 300.0_f64;
    let temperature_au = temperature_si / TEMP_F;
    let kt = temperature_au;
    let g_dof = (3 * n - 3) as f64;
    let k_target = (g_dof / 2.0) * kt;
    let v_each = ((k_target / ((n as f64) * 0.5 * (mass as f64))) as f64).sqrt() as Real;
    let mut vx: Vec<Real> = Vec::with_capacity(n);
    for _ in 0..n / 2 {
        vx.push(v_each);
        vx.push(-v_each);
    }
    let zero = vec![0.0; n];
    let state = ParticleState::new(
        (0..n).map(|i| (i as Real) * (1.0e-10 / LEN_F) as Real).collect(),
        zero.clone(),
        zero.clone(),
        vx,
        zero.clone(),
        zero,
        vec![mass; n],
        vec![0.0; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large(&gpu);
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integ = heddle_md::integrator::IntegratorRegistry::with_builtins()
        .build(
            &SlotConfig::from_params_str("velocity-verlet", "lossless = false"),
            &gpu,
            n, 0)
        .unwrap();
    let mut therm = build_csvr(&gpu, n, &csvr_kind(temperature_si, 1.0e-14, 11));
    ff.step(&mut buffers, &sim_box, &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    let mut scratch = gpu.device.alloc_zeros::<Real>(1).unwrap();
    for _ in 0..100 {
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, (1.0e-15 / TIME_F) as Real, &mut timings)
            .unwrap();
        therm
            .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
            .unwrap();
    }
    let mut sum = 0.0_f64;
    let n_samples = 250;
    for _ in 0..n_samples {
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, (1.0e-15 / TIME_F) as Real, &mut timings)
            .unwrap();
        therm
            .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
            .unwrap();
        sum += compute_kinetic_energy(&mut buffers, &mut scratch).unwrap() as f64;
    }
    let k_avg = sum / (n_samples as f64);
    let rel = (k_avg - k_target).abs() / k_target;
    assert!(rel < 0.15, "k_avg = {k_avg:e}, target {k_target:e}, rel {rel}");
}

// rq-efea1b70
#[test]
fn csvr_leaves_velocities_unchanged_when_k_zero() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    // All-zero velocities → K = 0.
    let state = ParticleState::new(
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![(1.66e-27 / MASS_F) as Real; n],
        vec![0.0; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut therm = build_csvr(&gpu, n, &csvr_kind(300.0, 1.0e-13, 1));
    therm
        .apply_post(&mut buffers, (1.0e-15 / TIME_F) as Real, &mut timings)
        .unwrap();
    // The GPU-side `csvr_sample_and_factor` kernel writes factor = 1.0
    // when k_old == 0 (the k_old > 0 guard suppresses the cross term and
    // the rescale-factor computation), so the rescale kernel runs but
    // leaves velocities at zero. The host can no longer skip the kernel
    // launch because k_old is never downloaded.
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vz = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
    assert!(vx.iter().all(|&v| v == 0.0));
    assert!(vy.iter().all(|&v| v == 0.0));
    assert!(vz.iter().all(|&v| v == 0.0));
}

// rq-70a46202
#[test]
fn csvr_constructs_for_a_settled_water_system() {
    let gpu = init_device().unwrap();
    let kind = csvr_kind(300.0, 1.0e-13, 1);
    // 24 particles (8 waters) with 24 constraints (3 per water for SETTLE);
    // g_dof = max(1, 3*24 - 24 - 3) = 45.
    let therm = ThermostatRegistry::with_builtins()
        .build_optional(Some(&kind), &gpu, 24, 24)
        .unwrap()
        .unwrap();
    let therm = unbox_csvr(therm);
    assert_eq!(therm.g_dof, 45);
}

// rq-d16be675
#[test]
fn csvr_clamps_g_dof_to_one_for_heavily_constrained_system() {
    let gpu = init_device().unwrap();
    let kind = csvr_kind(300.0, 1.0e-13, 1);
    // 4 particles with 12 constraints (degenerate over-constrained case):
    // 3*4 - 12 - 3 = -3 → clamped to 1.
    let therm = ThermostatRegistry::with_builtins()
        .build_optional(Some(&kind), &gpu, 4, 12)
        .unwrap()
        .unwrap();
    let therm = unbox_csvr(therm);
    assert_eq!(therm.g_dof, 1);
}
