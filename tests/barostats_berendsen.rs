// rq-0d8c8688
//
// Berendsen weak-coupling barostat tests. The barostat is exercised in
// isolation through its `apply` hook; the shared `compute_total_virial`
// and `rescale_positions` helpers are also exercised directly, and
// `SimulationBox::rescale_isotropic` is unit-tested host-side.

use heddle_md::forces::{AggregateLevel, AngleList, BondList, ExclusionList, ForceField, PotentialRegistry};
use heddle_md::gpu::{
    GpuContext, ParticleBuffers, compute_total_virial, init_device, rescale_positions,
};
use heddle_md::integrator::IntegratorStepExt;
use heddle_md::integrator::{
    Barostat, BarostatRegistry, BerendsenBarostat, IntegratorRegistry, ThermostatRegistry,
};
use heddle_md::precision::Real;
#[allow(unused_imports)]
use heddle_md::integrator::Thermostat; // needed for trait-method calls below
use heddle_md::io::config::NeighborListConfig;
use heddle_md::io::SlotConfig;
use heddle_md::pbc::{SimulationBox, SimulationBoxError};
use heddle_md::state::ParticleState;
use heddle_md::timings::{KernelStage, Timings};

const KB: f64 = 1.380649e-23;

fn box_small(gpu: &heddle_md::gpu::GpuContext) -> SimulationBox {
    // A box big enough that the position rescale never wraps anything
    // out of the primary image even after small μ deviations.
    SimulationBox::new(&gpu.device, 1.0e-9, 1.0e-9, 1.0e-9, 0.0, 0.0, 0.0).unwrap()
}

fn empty_force_field(gpu: &GpuContext, n: usize) -> ForceField {
    ForceField::new(
        &PotentialRegistry::with_builtins(),
        gpu,
        n,
        &box_small(&gpu),
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

fn berendsen_kind(pressure: f64, tau: f64, compressibility: f64) -> SlotConfig {
    SlotConfig::from_params_str(
        "berendsen",
        &format!(
            "pressure = {pressure:e}\ntau = {tau:e}\ncompressibility = {compressibility:e}\n"
        ),
    )
}

fn build_berendsen_barostat(
    gpu: &GpuContext,
    n: usize,
    slot: &SlotConfig,
) -> Box<dyn Barostat> {
    BarostatRegistry::with_builtins()
        .build_optional(Some(slot), gpu, n, 0)
        .unwrap()
        .unwrap()
}

fn unbox_berendsen_barostat(boxed: Box<dyn Barostat>) -> BerendsenBarostat {
    unsafe { *Box::from_raw(Box::into_raw(boxed) as *mut BerendsenBarostat) }
}

// Build a state with prescribed positions, velocities, and virials.
fn make_state(
    positions_x: Vec<Real>,
    velocities_x: Vec<Real>,
    masses: Vec<Real>,
    virials: Vec<Real>,
) -> ParticleState {
    let n = positions_x.len();
    let zero = vec![0.0; n];
    let state = ParticleState::new(
        positions_x,
        zero.clone(),
        zero.clone(),
        velocities_x,
        zero.clone(),
        zero.clone(),
        masses,
        vec![0.0; n],
        (0..n as u32).collect(),
        None,
        None,
    )
    .unwrap();
    let mut s = state;
    s.virials = virials;
    s
}

// --- Construction ---

// rq-52b7d30e
#[test]
fn registry_builds_berendsen_barostat() {
    let gpu = init_device().unwrap();
    let kind = berendsen_kind(1.0e5, 1.0e-12, 4.5e-10);
    let baro = unbox_berendsen_barostat(build_berendsen_barostat(&gpu, 4, &kind));
    assert_eq!(baro.most_recent_pressure, 0.0);
    assert_eq!(baro.most_recent_volume, 0.0);
    assert_eq!(baro.pressure, 1.0e5);
    assert_eq!(baro.tau, 1.0e-12);
    assert_eq!(baro.compressibility, 4.5e-10);
}

// rq-abf09acf
#[test]
fn registry_builds_berendsen_barostat_particle_count_zero() {
    let gpu = init_device().unwrap();
    let kind = berendsen_kind(1.0e5, 1.0e-12, 4.5e-10);
    let _baro = build_berendsen_barostat(&gpu, 0, &kind);
}

// rq-05e8f300
#[test]
fn barostat_build_optional_none_returns_none() {
    let gpu = init_device().unwrap();
    let registry = BarostatRegistry::with_builtins();
    let result = registry.build_optional(None, &gpu, 4, 0).unwrap();
    assert!(result.is_none());
}

// rq-909e5bb4
#[test]
fn barostat_with_builtins_exposes_berendsen() {
    let registry = BarostatRegistry::with_builtins();
    assert!(
        registry
            .builders
            .iter()
            .any(|b| b.kind_name() == "berendsen")
    );
}

// --- compute_total_virial helper ---

// rq-cf4d6ab4
#[test]
fn compute_total_virial_zero_virial_returns_zero() {
    let gpu = init_device().unwrap();
    let state = make_state(vec![0.0; 1], vec![0.0; 1], vec![1.0; 1], vec![0.0; 1]);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut scratch = gpu.device.alloc_zeros::<Real>(1).unwrap();
    let w = compute_total_virial(&buffers, &mut scratch).unwrap();
    assert_eq!(w, 0.0);
}

// rq-098fabd1
#[test]
fn compute_total_virial_matches_host_sum() {
    let gpu = init_device().unwrap();
    let virials = vec![1.0, -2.0, 3.0, -4.0];
    let state = make_state(
        vec![0.0; 4],
        vec![0.0; 4],
        vec![1.0; 4],
        virials.clone(),
    );
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut scratch = gpu.device.alloc_zeros::<Real>(1).unwrap();
    let w = compute_total_virial(&buffers, &mut scratch).unwrap();
    let expected: Real = virials.iter().sum();
    assert!((w - expected).abs() < 1.0e-5);
}

// rq-d801f67c
#[test]
fn compute_total_virial_is_deterministic() {
    let gpu = init_device().unwrap();
    let n = 1000usize;
    let virials: Vec<Real> = (0..n).map(|i| 0.5 - 0.001 * i as Real).collect();
    let zero = vec![0.0; n];
    let state = make_state(zero.clone(), zero.clone(), vec![1.0; n], virials);
    let buffers_a = ParticleBuffers::new(&gpu, &state).unwrap();
    let buffers_b = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut scratch_a = gpu.device.alloc_zeros::<Real>(1).unwrap();
    let mut scratch_b = gpu.device.alloc_zeros::<Real>(1).unwrap();
    let wa = compute_total_virial(&buffers_a, &mut scratch_a).unwrap();
    let wb = compute_total_virial(&buffers_b, &mut scratch_b).unwrap();
    assert_eq!(wa.to_bits(), wb.to_bits());
}

// rq-4a328491
#[test]
fn compute_total_virial_empty_state_returns_zero() {
    let gpu = init_device().unwrap();
    let state = make_state(Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut scratch = gpu.device.alloc_zeros::<Real>(1).unwrap();
    let w = compute_total_virial(&buffers, &mut scratch).unwrap();
    assert_eq!(w, 0.0);
}

// --- rescale_positions helper ---

// rq-77292dee
#[test]
fn rescale_positions_multiplies_components() {
    let gpu = init_device().unwrap();
    let state = ParticleState::new(
        vec![1.0, -4.0],
        vec![2.0, 5.0],
        vec![3.0, -6.0],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 2],
        vec![1.0; 2],
        vec![0.0; 2],
        vec![0u32; 2],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    rescale_positions(&mut buffers, 0.5).unwrap();
    let x = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let y = gpu.device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let z = gpu.device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    assert_eq!(x, vec![0.5, -2.0]);
    assert_eq!(y, vec![1.0, 2.5]);
    assert_eq!(z, vec![1.5, -3.0]);
}

// rq-2fc35d61
#[test]
fn rescale_positions_factor_one_is_identity() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let px: Vec<Real> = (0..n).map(|i| 0.5 - i as Real).collect();
    let py: Vec<Real> = (0..n).map(|i| 1.0 + 0.3 * i as Real).collect();
    let pz: Vec<Real> = (0..n).map(|i| -0.2 * i as Real).collect();
    let state = ParticleState::new(
        px.clone(),
        py.clone(),
        pz.clone(),
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        vec![0.0; n],
        vec![0u32; n],
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    rescale_positions(&mut buffers, 1.0).unwrap();
    let rx = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    assert_eq!(rx, px);
}

// rq-00e98375
#[test]
fn rescale_positions_does_not_touch_velocities_forces_masses_or_images() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = ParticleState::new(
        vec![1.0; n],
        vec![2.0; n],
        vec![3.0; n],
        vec![0.5; n],
        vec![-0.5; n],
        vec![0.25; n],
        vec![1.0e-26; n],
        vec![0.0; n],
        (0..n as u32).collect(),
        None,
        None,
    )
    .unwrap();
    let snap_velocities_x = state.velocities_x.clone();
    let snap_masses = state.masses.clone();
    let snap_images_x = state.images_x.clone();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    rescale_positions(&mut buffers, 0.7).unwrap();
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let m = gpu.device.dtoh_sync_copy(&buffers.masses).unwrap();
    let ix = gpu.device.dtoh_sync_copy(&buffers.images_x).unwrap();
    assert_eq!(vx, snap_velocities_x);
    assert_eq!(m, snap_masses);
    assert_eq!(ix, snap_images_x);
}

// rq-64c051d4
#[test]
fn rescale_positions_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = make_state(Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    rescale_positions(&mut buffers, 0.5).unwrap();
}

// --- SimulationBox::rescale_isotropic ---

// rq-af9257bb
#[test]
fn rescale_isotropic_multiplies_all_six_lattice_parameters() {
    let gpu = init_device().unwrap();
    let mut sim_box = SimulationBox::new(&gpu.device, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3).unwrap();
    sim_box.rescale_isotropic(0.5).unwrap();
    let [lx, ly, lz, xy, xz, yz] = sim_box.lattice();
    assert!((lx - 0.5).abs() < 1.0e-6);
    assert!((ly - 1.0).abs() < 1.0e-6);
    assert!((lz - 1.5).abs() < 1.0e-6);
    assert!((xy - 0.05).abs() < 1.0e-6);
    assert!((xz - 0.10).abs() < 1.0e-6);
    assert!((yz - 0.15).abs() < 1.0e-6);
}

// rq-911d9120
#[test]
fn rescale_isotropic_bumps_generation() {
    let gpu = init_device().unwrap();
    let mut sim_box = SimulationBox::new(&gpu.device, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0).unwrap();
    let g = sim_box.generation();
    sim_box.rescale_isotropic(1.01).unwrap();
    assert_eq!(sim_box.generation(), g + 1);
}

// rq-b0b4c220
#[test]
fn rescale_isotropic_rejects_zero_factor() {
    let gpu = init_device().unwrap();
    let mut sim_box = SimulationBox::new(&gpu.device, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0).unwrap();
    let result = sim_box.rescale_isotropic(0.0);
    assert!(matches!(result, Err(SimulationBoxError::NonPositiveDiagonal { .. })));
}

// rq-9ba11e1e
#[test]
fn rescale_isotropic_rejects_nan_factor() {
    let gpu = init_device().unwrap();
    let mut sim_box = SimulationBox::new(&gpu.device, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0).unwrap();
    let result = sim_box.rescale_isotropic(Real::NAN);
    assert!(matches!(result, Err(SimulationBoxError::NonFiniteLatticeValue { .. })));
}

// --- Per-step kernel sequence ---

// rq-92cecd28
#[test]
fn apply_launches_expected_kernel_set() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = make_state(
        (0..n).map(|i| 1.0e-10 * i as Real).collect(),
        vec![500.0; n],
        vec![1.66e-27; n],
        vec![0.0; n],
    );
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small(&gpu);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_berendsen_barostat(&gpu, n, &berendsen_kind(1.0e5, 1.0e-12, 4.5e-10));
    baro.apply(&mut buffers, &mut sim_box, 1.0e-15, &mut timings)
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
    assert_eq!(count_for(KernelStage::VIRIAL_SUM_REDUCE), 1);
    // The per-particle position rescale is dispatched by the
    // JIT-composed post-force per-particle kernel; the standalone
    // stage is not recorded.
    assert_eq!(
        count_for(KernelStage::BERENDSEN_BAROSTAT_RESCALE_POSITIONS),
        0
    );
    // The barostat does not launch the integrator's VV kernels.
    assert_eq!(count_for(KernelStage::VV_KICK_DRIFT), 0);
    assert_eq!(count_for(KernelStage::VV_KICK), 0);
}

// rq-69600add
#[test]
fn apply_on_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = make_state(Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small(&gpu);
    let g_before = sim_box.generation();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_berendsen_barostat(&gpu, 0, &berendsen_kind(1.0e5, 1.0e-12, 4.5e-10));
    baro.apply(&mut buffers, &mut sim_box, 1.0e-15, &mut timings)
        .unwrap();
    assert_eq!(sim_box.generation(), g_before);
}

// --- μ correctness ---

// Helper: build a tiny system whose K and W are exactly known, so we
// can read the post-apply box volume and confirm μ³ matches the
// analytical formula.
fn system_with_pressure(target_pressure_pa: f64) -> (Vec<Real>, Vec<Real>, Vec<Real>, Vec<Real>, f64) {
    // 8 particles at four ±v pairs, all on the x-axis to give COM=0.
    let n = 8;
    let mass: Real = 1.66e-27;
    let v_mag: Real = 500.0;
    let v_squared_sum = (n as f64) * (v_mag as f64).powi(2);
    let k = 0.5 * (mass as f64) * v_squared_sum;
    // For a chosen target P, solve W = 3 V P - 2 K.
    let v = (1.0e-9_f64).powi(3); // box_small volume
    let w_required = 3.0 * v * target_pressure_pa - 2.0 * k;
    // Distribute W evenly across particles.
    let per_particle_virial = (w_required / n as f64) as Real;
    let mut vx: Vec<Real> = Vec::with_capacity(n);
    for _ in 0..n / 2 {
        vx.push(v_mag);
        vx.push(-v_mag);
    }
    let px: Vec<Real> = (0..n).map(|i| 1.0e-10 * (i as Real - 3.5)).collect();
    let masses = vec![mass; n];
    let virials = vec![per_particle_virial; n];
    (px, vx, masses, virials, target_pressure_pa)
}

// rq-a2bb55c6
#[test]
fn mu_equals_one_when_pressure_equals_target() {
    let gpu = init_device().unwrap();
    let p_target = 1.0e6_f64;
    let (px, vx, masses, virials, _) = system_with_pressure(p_target);
    let n = px.len();
    let state = make_state(px.clone(), vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small(&gpu);
    let v_pre = sim_box.volume();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_berendsen_barostat(&gpu, n, &berendsen_kind(p_target, 1.0e-12, 4.5e-10));
    baro.apply(&mut buffers, &mut sim_box, 1.0e-15, &mut timings)
        .unwrap();
    sim_box.flush_from_device().unwrap();
    // μ should be 1.0 to within f32 round-off, so the box stays the same size.
    let v_post = sim_box.volume();
    let rel = ((v_post - v_pre) / v_pre).abs() as f64;
    assert!(rel < 1.0e-3, "v_post = {v_post}, v_pre = {v_pre} (rel {rel})");
    // Positions effectively unchanged.
    let px_post = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    for (a, b) in px_post.iter().zip(px.iter()) {
        let r = (a - b).abs() / b.abs().max(1.0e-12);
        assert!(r < 1.0e-3, "pos drift {a} vs {b}");
    }
}

// rq-c9f9d550
#[test]
fn mu_less_than_one_when_pressure_below_target() {
    // P = P_target / 2 → P_target − P > 0 → μ³ < 1 → contraction.
    let gpu = init_device().unwrap();
    let p_target = 1.0e6_f64;
    let (px, vx, masses, virials, _) = system_with_pressure(p_target / 2.0);
    let n = px.len();
    let state = make_state(px, vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small(&gpu);
    let v_pre = sim_box.volume() as f64;
    let dt = 1.0e-13;
    let tau = 1.0e-12_f64;
    let beta = 4.5e-10_f64;
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_berendsen_barostat(&gpu, n, &berendsen_kind(p_target, tau, beta));
    baro.apply(&mut buffers, &mut sim_box, dt, &mut timings)
        .unwrap();
    sim_box.flush_from_device().unwrap();
    let v_post = sim_box.volume() as f64;
    let mu_cubed_actual = v_post / v_pre;
    let expected_mu_cubed = 1.0 - beta * ((dt as f64) / tau) * (p_target - p_target / 2.0);
    let rel = (mu_cubed_actual - expected_mu_cubed).abs() / expected_mu_cubed.abs();
    assert!(mu_cubed_actual < 1.0, "expected contraction");
    assert!(rel < 1.0e-3, "μ³ actual = {mu_cubed_actual}, expected ≈ {expected_mu_cubed}");
}

// rq-ed3ed814
#[test]
fn mu_greater_than_one_when_pressure_above_target() {
    // P = 2 · P_target → μ³ > 1 → expansion.
    let gpu = init_device().unwrap();
    let p_target = 1.0e6_f64;
    let (px, vx, masses, virials, _) = system_with_pressure(2.0 * p_target);
    let n = px.len();
    let state = make_state(px, vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small(&gpu);
    let v_pre = sim_box.volume() as f64;
    let dt = 1.0e-13;
    let tau = 1.0e-12_f64;
    let beta = 4.5e-10_f64;
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_berendsen_barostat(&gpu, n, &berendsen_kind(p_target, tau, beta));
    baro.apply(&mut buffers, &mut sim_box, dt, &mut timings)
        .unwrap();
    sim_box.flush_from_device().unwrap();
    let v_post = sim_box.volume() as f64;
    let mu_cubed_actual = v_post / v_pre;
    let expected_mu_cubed = 1.0 - beta * ((dt as f64) / tau) * (p_target - 2.0 * p_target);
    let rel = (mu_cubed_actual - expected_mu_cubed).abs() / expected_mu_cubed.abs();
    assert!(mu_cubed_actual > 1.0);
    assert!(rel < 1.0e-3, "μ³ actual = {mu_cubed_actual}, expected ≈ {expected_mu_cubed}");
}

// rq-4dbe4a07
#[test]
fn mu_clamped_to_safety_floor() {
    // Set up so that β · (dt/τ) · (P_target − P) >> 1; μ³ would be
    // negative, gets clamped to MU_MIN³ = 1e-18, so V_post / V_pre = 1e-18.
    let gpu = init_device().unwrap();
    let p_target = 1.0e25_f64; // wildly large target → huge underpressure
    let (px, vx, masses, virials, _) = system_with_pressure(0.0);
    let n = px.len();
    let state = make_state(px, vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small(&gpu);
    let v_pre = sim_box.volume() as f64;
    let dt = 1.0e-12;
    let tau = 1.0e-12_f64;
    let beta = 1.0_f64; // also huge β
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_berendsen_barostat(&gpu, n, &berendsen_kind(p_target, tau, beta));
    baro.apply(&mut buffers, &mut sim_box, dt, &mut timings)
        .unwrap();
    sim_box.flush_from_device().unwrap();
    let v_post = sim_box.volume() as f64;
    let mu_cubed_actual = v_post / v_pre;
    // μ_min^3 = 1e-18; in f32 the cbrt(1e-18) round-trip can be off by
    // factors of ~3, so check the order of magnitude.
    assert!(
        mu_cubed_actual < 1.0e-15,
        "μ³ clamped; got {mu_cubed_actual}"
    );
    assert!(
        mu_cubed_actual > 1.0e-21,
        "μ³ should not be exactly zero; got {mu_cubed_actual}"
    );
}

// --- Fractional-coord and shape invariants ---

// rq-cf183b79
#[test]
fn fractional_coordinates_invariant_under_apply() {
    let gpu = init_device().unwrap();
    let p_target = 1.0e6_f64;
    let (px, vx, masses, virials, _) = system_with_pressure(p_target / 2.0);
    let n = px.len();
    let state = make_state(px.clone(), vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small(&gpu);
    let lx_pre = sim_box.lx();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_berendsen_barostat(&gpu, n, &berendsen_kind(p_target, 1.0e-12, 4.5e-10));
    baro.apply(&mut buffers, &mut sim_box, 1.0e-13, &mut timings)
        .unwrap();
    sim_box.flush_from_device().unwrap();
    let lx_post = sim_box.lx();
    let px_post = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    // Fractional coord (x / lx) must be invariant under uniform scaling.
    for (i, (a, b)) in px_post.iter().zip(px.iter()).enumerate() {
        let f_pre = (b / lx_pre) as f64;
        let f_post = (a / lx_post) as f64;
        let rel = (f_post - f_pre).abs() / f_pre.abs().max(1.0e-12);
        assert!(rel < 1.0e-3, "particle {i} fractional drift {rel}");
    }
}

// rq-16252a37
#[test]
fn triclinic_shape_preserved_under_apply() {
    let gpu = init_device().unwrap();
    let p_target = 1.0e6_f64;
    let (px, vx, masses, virials, _) = system_with_pressure(p_target / 2.0);
    let n = px.len();
    let state = make_state(px, vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    // Triclinic box.
    let mut sim_box =
        SimulationBox::new(&gpu.device, 1.0e-9, 1.0e-9, 1.0e-9, 0.1e-9, 0.2e-9, 0.3e-9).unwrap();
    let [lx_pre, _ly_pre, _lz_pre, xy_pre, xz_pre, yz_pre] = sim_box.lattice();
    let r_xy_pre = (xy_pre / lx_pre) as f64;
    let r_xz_pre = (xz_pre / lx_pre) as f64;
    let r_yz_pre = (yz_pre / lx_pre) as f64;
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_berendsen_barostat(&gpu, n, &berendsen_kind(p_target, 1.0e-12, 4.5e-10));
    baro.apply(&mut buffers, &mut sim_box, 1.0e-13, &mut timings)
        .unwrap();
    let [lx_post, _ly_post, _lz_post, xy_post, xz_post, yz_post] = sim_box.lattice();
    let r_xy_post = (xy_post / lx_post) as f64;
    let r_xz_post = (xz_post / lx_post) as f64;
    let r_yz_post = (yz_post / lx_post) as f64;
    let rel_xy = (r_xy_post - r_xy_pre).abs() / r_xy_pre.abs();
    let rel_xz = (r_xz_post - r_xz_pre).abs() / r_xz_pre.abs();
    let rel_yz = (r_yz_post - r_yz_pre).abs() / r_yz_pre.abs();
    assert!(rel_xy < 1.0e-4);
    assert!(rel_xz < 1.0e-4);
    assert!(rel_yz < 1.0e-4);
}

// --- Box-generation propagation ---

// rq-136f7d15
#[test]
fn generation_advances_after_apply() {
    let gpu = init_device().unwrap();
    let p_target = 1.0e6_f64;
    let (px, vx, masses, virials, _) = system_with_pressure(p_target);
    let n = px.len();
    let state = make_state(px, vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small(&gpu);
    let g_pre = sim_box.generation();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_berendsen_barostat(&gpu, n, &berendsen_kind(p_target, 1.0e-12, 4.5e-10));
    baro.apply(&mut buffers, &mut sim_box, 1.0e-15, &mut timings)
        .unwrap();
    assert_eq!(sim_box.generation(), g_pre + 1);
}

// --- Log columns ---

// rq-7564b1e7 rq-75297a48
#[test]
fn log_column_names_returns_pressure_and_box_volume() {
    let gpu = init_device().unwrap();
    let baro = build_berendsen_barostat(&gpu, 4, &berendsen_kind(1.0e5, 1.0e-12, 4.5e-10));
    let names: Vec<&str> = baro.log_column_names().iter().map(|(n, _)| *n).collect();
    assert_eq!(names, vec!["pressure", "box_volume"]);
}

// rq-24073418
#[test]
fn log_column_values_returns_cached_pressure_and_volume() {
    let gpu = init_device().unwrap();
    let mut baro =
        unbox_berendsen_barostat(build_berendsen_barostat(&gpu, 4, &berendsen_kind(1.0e5, 1.0e-12, 4.5e-10)));
    baro.most_recent_pressure = 1.01e5;
    baro.most_recent_volume = 1.0e-27;
    let extras = baro.log_column_values(0.0, 0.0);
    assert_eq!(extras, vec![1.01e5, 1.0e-27]);
}

// --- Composition with the orthogonal framework ---

// rq-13bf10fc rq-ad67b3da rq-2d579721
#[test]
fn composes_with_velocity_verlet_and_berendsen_thermostat() {
    // Smoke test: full per-step dispatch order
    // (thermostat.apply_pre → integrator.step → thermostat.apply_post → barostat.apply)
    // completes without error for a few steps on a tiny LJ-free system.
    let gpu = init_device().unwrap();
    let n = 8usize;
    let mass: Real = 1.66e-27;
    let v_mag: Real = 500.0;
    let mut vx: Vec<Real> = Vec::with_capacity(n);
    for _ in 0..n / 2 {
        vx.push(v_mag);
        vx.push(-v_mag);
    }
    let state = make_state(
        (0..n).map(|i| 1.0e-10 * (i as Real - 3.5)).collect(),
        vx,
        vec![mass; n],
        vec![0.0; n],
    );
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small(&gpu);
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integ = IntegratorRegistry::with_builtins()
        .build(
            &SlotConfig::from_params_str("velocity-verlet", "lossless = false"),
            &gpu,
            n, 0)
        .unwrap();
    let mut therm = ThermostatRegistry::with_builtins()
        .build_optional(
            Some(&SlotConfig::from_params_str(
                "berendsen",
                "temperature = 85.0\ntau = 1.0e-13\n",
            )),
            &gpu,
            n,
            0,
        )
        .unwrap()
        .unwrap();
    let mut baro = build_berendsen_barostat(&gpu, n, &berendsen_kind(1.0e5, 1.0e-12, 1.0e-9));
    ff.step(&mut buffers, &sim_box, &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    for _ in 0..10 {
        therm
            .apply_pre(&mut buffers, 1.0e-15, &mut timings)
            .unwrap();
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, 1.0e-15, &mut timings)
            .unwrap();
        therm
            .apply_post(&mut buffers, 1.0e-15, &mut timings)
            .unwrap();
        baro.apply(&mut buffers, &mut sim_box, 1.0e-15, &mut timings)
            .unwrap();
    }
    sim_box.flush_from_device().unwrap();
    let v_final = sim_box.volume();
    assert!(v_final.is_finite() && v_final > 0.0);
}

// --- Determinism ---

// rq-3460c38a
#[test]
fn two_runs_byte_identical() {
    let gpu = init_device().unwrap();
    let p_target = 1.0e6_f64;
    let (px, vx, masses, virials, _) = system_with_pressure(p_target / 2.0);

    fn run_once(
        gpu: &GpuContext,
        px: &[Real],
        vx: &[Real],
        masses: &[Real],
        virials: &[Real],
        p_target: f64,
    ) -> (Vec<Real>, [Real; 6]) {
        let n = px.len();
        let state = make_state(
            px.to_vec(),
            vx.to_vec(),
            masses.to_vec(),
            virials.to_vec(),
        );
        let mut buffers = ParticleBuffers::new(gpu, &state).unwrap();
        let mut sim_box = box_small(&gpu);
        let mut timings = Timings::new(gpu).unwrap();
        let mut baro =
            build_berendsen_barostat(gpu, n, &berendsen_kind(p_target, 1.0e-12, 4.5e-10));
        for _ in 0..5 {
            baro.apply(&mut buffers, &mut sim_box, 1.0e-15, &mut timings)
                .unwrap();
        }
        let positions_x = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
        (positions_x, sim_box.lattice())
    }

    let (px_a, lat_a) = run_once(&gpu, &px, &vx, &masses, &virials, p_target);
    let (px_b, lat_b) = run_once(&gpu, &px, &vx, &masses, &virials, p_target);
    assert_eq!(px_a, px_b);
    for i in 0..6 {
        assert_eq!(lat_a[i].to_bits(), lat_b[i].to_bits());
    }
    let _ = KB;
}
