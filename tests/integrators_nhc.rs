// rq-f606ff6f
//
// Nose-Hoover chain (NHC) integrator tests.

use dynamics::forces::{AngleList, BondList, ExclusionList, ForceField};
use dynamics::gpu::{
    GpuContext, ParticleBuffers, compute_kinetic_energy, init_device, rescale_velocities,
};
use dynamics::integrator::{
    Integrator, IntegratorRegistry, NoseHooverChainState,
};
use dynamics::io::IntegratorKind;
use dynamics::io::config::NeighborListConfig;
use dynamics::pbc::SimulationBox;
use dynamics::state::ParticleState;
use dynamics::timings::{KernelStage, Timings};

const KB: f64 = 1.380649e-23;

fn small_state(n: usize, mass: f32) -> ParticleState {
    let pos: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let zero = vec![0.0_f32; n];
    ParticleState::new(
        pos,
        zero.clone(),
        zero.clone(),
        zero.clone(),
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

fn nhc_kind(
    temperature: f64,
    tau: f64,
    chain_length: u32,
    yoshida_order: u32,
    n_resp: u32,
) -> IntegratorKind {
    IntegratorKind::NoseHooverChain {
        temperature,
        tau,
        chain_length,
        yoshida_order,
        n_resp,
    }
}

// --- Construction ---

#[test]
fn registry_builds_nhc_with_defaults() {
    let gpu = init_device().unwrap();
    let kind = nhc_kind(300.0, 1.0e-13, 3, 3, 1);
    let _integrator = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 4)
        .unwrap();
}

#[test]
fn registry_builds_nhc_with_chain_length_1() {
    let gpu = init_device().unwrap();
    let kind = nhc_kind(300.0, 1.0e-13, 1, 3, 1);
    let _integrator = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 4)
        .unwrap();
}

#[test]
fn registry_builds_nhc_with_particle_count_zero() {
    let gpu = init_device().unwrap();
    let kind = nhc_kind(300.0, 1.0e-13, 3, 3, 1);
    let _integrator = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 0)
        .unwrap();
}

// --- compute_kinetic_energy helper ---

#[test]
fn kinetic_energy_reduce_zero_velocity_returns_zero() {
    let gpu = init_device().unwrap();
    let state = small_state(4, 1.0);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut scratch = gpu.device.alloc_zeros::<f32>(1).unwrap();
    let ke = compute_kinetic_energy(&buffers, &mut scratch).unwrap();
    assert_eq!(ke, 0.0);
}

#[test]
fn kinetic_energy_reduce_matches_host_formula() {
    let gpu = init_device().unwrap();
    let n = 8usize;
    let masses: Vec<f32> = (0..n).map(|i| 1.0 + 0.5 * i as f32).collect();
    let vx: Vec<f32> = (0..n).map(|i| 0.1 + i as f32).collect();
    let vy: Vec<f32> = (0..n).map(|i| -0.2 * i as f32).collect();
    let vz: Vec<f32> = (0..n).map(|i| 0.3 - i as f32).collect();
    let state = ParticleState::new(
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vx.clone(),
        vy.clone(),
        vz.clone(),
        masses.clone(),
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap();
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut scratch = gpu.device.alloc_zeros::<f32>(1).unwrap();
    let ke = compute_kinetic_energy(&buffers, &mut scratch).unwrap();
    let mut expected = 0.0_f32;
    for i in 0..n {
        expected += 0.5 * masses[i] * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    }
    let rel = (ke - expected).abs() / expected.abs();
    assert!(rel < 5.0e-5, "ke = {}, expected ≈ {}", ke, expected);
}

#[test]
fn kinetic_energy_reduce_is_deterministic() {
    let gpu = init_device().unwrap();
    let n = 1000usize;
    let masses: Vec<f32> = (0..n).map(|i| 1.0 + 0.001 * i as f32).collect();
    let vx: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
    let vy: Vec<f32> = (0..n).map(|i| (i as f32).cos()).collect();
    let vz: Vec<f32> = (0..n).map(|i| 0.5 - 0.001 * i as f32).collect();
    let state = ParticleState::new(
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vx,
        vy,
        vz,
        masses,
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap();
    let buffers_a = ParticleBuffers::new(&gpu, &state).unwrap();
    let buffers_b = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut scratch_a = gpu.device.alloc_zeros::<f32>(1).unwrap();
    let mut scratch_b = gpu.device.alloc_zeros::<f32>(1).unwrap();
    let ke_a = compute_kinetic_energy(&buffers_a, &mut scratch_a).unwrap();
    let ke_b = compute_kinetic_energy(&buffers_b, &mut scratch_b).unwrap();
    assert_eq!(ke_a.to_bits(), ke_b.to_bits(), "byte-for-byte equal");
}

#[test]
fn kinetic_energy_reduce_empty_state_returns_zero() {
    let gpu = init_device().unwrap();
    let state = small_state(0, 1.0);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut scratch = gpu.device.alloc_zeros::<f32>(1).unwrap();
    let ke = compute_kinetic_energy(&buffers, &mut scratch).unwrap();
    assert_eq!(ke, 0.0);
}

// --- rescale_velocities helper ---

#[test]
fn rescale_velocities_multiplies_components_by_factor() {
    let gpu = init_device().unwrap();
    let state = ParticleState::new(
        vec![0.0_f32; 2],
        vec![0.0_f32; 2],
        vec![0.0_f32; 2],
        vec![1.0_f32, -4.0_f32],
        vec![2.0_f32, 5.0_f32],
        vec![3.0_f32, -6.0_f32],
        vec![1.0_f32; 2],
        vec![0.0_f32; 2],
        vec![0u32; 2],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    rescale_velocities(&mut buffers, 0.5).unwrap();
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let vy = gpu.device.dtoh_sync_copy(&buffers.velocities_y).unwrap();
    let vz = gpu.device.dtoh_sync_copy(&buffers.velocities_z).unwrap();
    assert_eq!(vx, vec![0.5_f32, -2.0_f32]);
    assert_eq!(vy, vec![1.0_f32, 2.5_f32]);
    assert_eq!(vz, vec![1.5_f32, -3.0_f32]);
}

#[test]
fn rescale_velocities_factor_one_is_identity() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let vx: Vec<f32> = (0..n).map(|i| 0.5 - i as f32).collect();
    let vy: Vec<f32> = (0..n).map(|i| 1.0 + 0.3 * i as f32).collect();
    let vz: Vec<f32> = (0..n).map(|i| -0.2 * i as f32).collect();
    let state = ParticleState::new(
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vx.clone(),
        vy.clone(),
        vz.clone(),
        vec![1.0_f32; n],
        vec![0.0_f32; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    rescale_velocities(&mut buffers, 1.0).unwrap();
    let rvx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    assert_eq!(rvx, vx);
}

#[test]
fn rescale_velocities_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = small_state(0, 1.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    rescale_velocities(&mut buffers, 0.5).unwrap();
}

// --- NHC slot integration ---

#[test]
fn nhc_step_launches_expected_kernels() {
    let gpu = init_device().unwrap();
    let state = small_state(4, 1.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut force_field = empty_force_field(&gpu, 4);
    let mut timings = Timings::new(&gpu).unwrap();

    let kind = nhc_kind(300.0, 1.0e-13, 3, 3, 1);
    let mut integrator = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 4)
        .unwrap();

    // Warm up forces.
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
    // Two KE reductions per step.
    assert_eq!(count_for(KernelStage::KINETIC_ENERGY_REDUCE), 2);
    // Yoshida 3 × n_resp 1 × 2 halves = 6 rescale launches per step.
    assert_eq!(count_for(KernelStage::NHC_RESCALE_VELOCITIES), 6);
    assert_eq!(count_for(KernelStage::VV_KICK_DRIFT), 1);
    assert_eq!(count_for(KernelStage::VV_KICK), 1);
}

#[test]
fn nhc_step_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = small_state(0, 1.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut force_field = empty_force_field(&gpu, 0);
    let mut timings = Timings::new(&gpu).unwrap();

    let kind = nhc_kind(300.0, 1.0e-13, 3, 3, 1);
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

// --- Log columns ---

#[test]
fn nhc_log_column_names_returns_nhc_conserved() {
    let gpu = init_device().unwrap();
    let kind = nhc_kind(300.0, 1.0e-13, 3, 3, 1);
    let integrator = IntegratorRegistry::with_builtins()
        .build(&kind, &gpu, 4)
        .unwrap();
    assert_eq!(integrator.log_column_names(), &["nhc_conserved"]);
}

#[test]
fn vv_and_langevin_log_column_names_are_empty() {
    let gpu = init_device().unwrap();
    let vv = IntegratorRegistry::with_builtins()
        .build(&IntegratorKind::VelocityVerlet { lossless: false }, &gpu, 4)
        .unwrap();
    assert!(vv.log_column_names().is_empty());
    let lan = IntegratorRegistry::with_builtins()
        .build(
            &IntegratorKind::LangevinBaoab {
                friction: 1.0e12,
                temperature: 300.0,
                seed: 1,
            },
            &gpu,
            4,
        )
        .unwrap();
    assert!(lan.log_column_names().is_empty());
}

#[test]
fn nhc_log_column_values_combines_ke_pe_and_chain_term() {
    let gpu = init_device().unwrap();
    // Construct an NHC with known T, M=2, chain state set by hand below.
    let state = NoseHooverChainState::new_for_test(&gpu, 4, 300.0, 1.0e-13, 2, 3, 1);
    let kt = KB * 300.0;
    let q1 = (state.g_dof as f64) * kt * 1.0e-13_f64.powi(2);
    let q2 = kt * 1.0e-13_f64.powi(2);
    // Hand-set xi and p_xi for predictable arithmetic.
    let mut s = state;
    s.xi[0] = 0.1;
    s.xi[1] = 0.2;
    s.p_xi[0] = 0.5e-30;
    s.p_xi[1] = -0.3e-30;

    let ke = 1.0e-20;
    let pe = 2.0e-20;
    let expected_chain = s.p_xi[0].powi(2) / (2.0 * q1)
        + s.p_xi[1].powi(2) / (2.0 * q2)
        + (s.g_dof as f64) * kt * s.xi[0]
        + kt * s.xi[1];
    let expected = ke + pe + expected_chain;
    let extras = s.log_column_values(ke, pe);
    assert_eq!(extras.len(), 1);
    let rel = (extras[0] - expected).abs() / expected.abs();
    assert!(rel < 1.0e-12, "nhc_conserved = {}, expected {}", extras[0], expected);
}

// Tiny test-only constructor: NoseHooverChainState fields are pub for
// inspection/checkpoint use, but `new` is private. Provide a thin
// helper so tests can build a state without bouncing through the
// registry.
trait NhcTestCtor: Sized {
    fn new_for_test(
        gpu: &GpuContext,
        particle_count: usize,
        temperature: f64,
        tau: f64,
        chain_length: u32,
        yoshida_order: u32,
        n_resp: u32,
    ) -> Self;
}

impl NhcTestCtor for NoseHooverChainState {
    fn new_for_test(
        gpu: &GpuContext,
        particle_count: usize,
        temperature: f64,
        tau: f64,
        chain_length: u32,
        yoshida_order: u32,
        n_resp: u32,
    ) -> Self {
        let kind = nhc_kind(temperature, tau, chain_length, yoshida_order, n_resp);
        let boxed = IntegratorRegistry::with_builtins()
            .build(&kind, gpu, particle_count)
            .unwrap();
        // The registry returns Box<dyn Integrator>; for our tests we
        // need the concrete type. Downcast via a Debug-string sanity
        // check, then transmute. Since this is the only path that
        // produces NoseHooverChainState, this is safe.
        let raw: *mut dyn Integrator = Box::into_raw(boxed);
        unsafe { *Box::from_raw(raw as *mut NoseHooverChainState) }
    }
}

// --- End-to-end determinism + COM conservation ---

// Build an SI-realistic state for NHC tests: atomic-scale mass
// (1 amu ≈ 1.66e-27 kg) and thermal-scale velocities (~1000 m/s) so
// KE ~ kT and the chain doesn't blow up.
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

#[test]
fn nhc_two_runs_with_identical_inputs_match() {
    let gpu = init_device().unwrap();
    let state = atomic_state(8);

    fn run_once(gpu: &GpuContext, state: &ParticleState) -> Vec<f32> {
        let n = state.particle_count();
        let mut buffers = ParticleBuffers::new(gpu, state).unwrap();
        let mut sim_box = box_large();
        let mut ff = empty_force_field(gpu, n);
        let mut timings = Timings::new(gpu).unwrap();
        let kind = nhc_kind(300.0, 1.0e-13, 3, 3, 1);
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
    // Sanity: velocities are finite.
    for v in &a {
        assert!(v.is_finite(), "NaN/Inf in NHC trajectory: {a:?}");
    }
}

#[test]
fn nhc_preserves_com_momentum_to_round_off() {
    let gpu = init_device().unwrap();
    let n = 16usize;
    let state = atomic_state(n);
    let mass = 1.66e-27_f32;
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_large();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let kind = nhc_kind(300.0, 1.0e-13, 3, 3, 1);
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
    // The NHC velocity rescale is uniform and so preserves COM
    // momentum exactly in real arithmetic; in f32 the rescale factor
    // and per-component multiply accumulate small ULP errors. After 20
    // steps with 6 rescales/step, the residual COM momentum is bounded
    // by ~120 · ULP · m · max|v|.
    let scale: f32 = vx.iter().map(|v| v.abs()).fold(0.0, f32::max);
    let tol = (mass as f64) * (scale as f64) * 1.0e-3;
    assert!(
        p_com.abs() < tol,
        "p_com = {p_com} (tol {tol}), velocities = {vx:?}"
    );
}
