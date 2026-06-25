// rq-11f5dfd1
//
// Stochastic cell-rescaling (C-rescale) barostat tests. The barostat
// is exercised in isolation through its `apply` hook; the shared
// `compute_total_virial` and `rescale_positions` helpers and the
// `SimulationBox::rescale_isotropic` convenience are covered in
// `tests/barostats_berendsen.rs` and not re-tested here.

use heddle_md::forces::{AggregateLevel, AngleList, BondList, ExclusionList, ForceField, PotentialRegistry};
use heddle_md::gpu::{GpuContext, ParticleBuffers, init_device};
use heddle_md::integrator::IntegratorStepExt;
use heddle_md::integrator::{
    Barostat, BarostatRegistry, CRescaleBarostat, IntegratorRegistry, ThermostatRegistry,
    philox_normal,
};
use heddle_md::io::config::NeighborListConfig;
use heddle_md::io::SlotConfig;
use heddle_md::pbc::SimulationBox;
use heddle_md::state::ParticleState;
use heddle_md::timings::{KernelStage, Timings};
use heddle_md::precision::Real;

// k_B = 1 inside the engine (atomic units). The tests below operate
// in atomic units throughout; this SI value of the Boltzmann constant
// is retained only as a reference for converting human-readable SI
// inputs to atomic units.
#[allow(dead_code)]
const KB: f64 = 1.380649e-23;
// SI-per-atomic conversion factors. Divide an SI value by the
// matching factor to express it in atomic units (e.g. `1e-9 / LEN_F`
// converts 1 nm to ~18.9 Bohr).
const LEN_F: f64 = 5.29177210903e-11;
const MASS_F: f64 = 9.1093837015e-31;
const TIME_F: f64 = 2.4188843265857195e-17;
const PRESSURE_F: f64 = 29421015696522.1;
const TEMP_F: f64 = 315775.0248040668;

fn box_small(gpu: &heddle_md::gpu::GpuContext) -> SimulationBox {
    // 1 nm cubic box, expressed in atomic units (~18.9 Bohr per side).
    let l = (1.0e-9 / LEN_F) as Real;
    SimulationBox::new(&gpu.device, l, l, l, 0.0, 0.0, 0.0).unwrap()
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

fn c_rescale_kind(
    pressure: f64,
    temperature: f64,
    tau: f64,
    compressibility: f64,
    seed: u64,
) -> SlotConfig {
    // Accept SI inputs (Pa, K, s, 1/Pa) for human readability and
    // convert to atomic units (E_h/a_0^3, E_h/k_B, hbar/E_h, a_0^3/E_h).
    let pressure = pressure / PRESSURE_F;
    let temperature = temperature / TEMP_F;
    let tau = tau / TIME_F;
    let compressibility = compressibility * PRESSURE_F;
    SlotConfig::from_params_str(
        "c-rescale",
        &format!(
            "pressure = {pressure:e}\ntemperature = {temperature:e}\ntau = {tau:e}\ncompressibility = {compressibility:e}\nseed = {seed}\n"
        ),
    )
}

fn build_c_rescale(
    gpu: &GpuContext,
    n: usize,
    slot: &SlotConfig,
) -> Box<dyn Barostat> {
    BarostatRegistry::with_builtins()
        .build_optional(Some(slot), gpu, n, 0)
        .unwrap()
        .unwrap()
}

fn unbox_c_rescale(boxed: Box<dyn Barostat>) -> CRescaleBarostat {
    unsafe { *Box::from_raw(Box::into_raw(boxed) as *mut CRescaleBarostat) }
}

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

// Helper: build a tiny system whose K and W produce a known instantaneous
// pressure (matches the helper used in tests/barostats_berendsen.rs).
fn system_with_pressure(
    target_pressure_pa: f64,
) -> (Vec<Real>, Vec<Real>, Vec<Real>, Vec<Real>) {
    // All quantities are expressed in atomic units, since the engine
    // operates in atomic units throughout. The SI target pressure
    // input is converted to atomic on the way in.
    let n = 8;
    let mass: Real = (1.66e-27 / MASS_F) as Real;
    let v_mag: Real = (500.0 / 2187691.2636411153_f64) as Real;
    let v_squared_sum = (n as f64) * (v_mag as f64).powi(2);
    let k = 0.5 * (mass as f64) * v_squared_sum;
    let box_edge_au = 1.0e-9 / LEN_F; // 1 nm cubic box, in Bohr
    let v = box_edge_au.powi(3);
    let target_pressure_au = target_pressure_pa / PRESSURE_F;
    let w_required = 3.0 * v * target_pressure_au - 2.0 * k;
    let per_particle_virial = (w_required / n as f64) as Real;
    let mut vx: Vec<Real> = Vec::with_capacity(n);
    for _ in 0..n / 2 {
        vx.push(v_mag);
        vx.push(-v_mag);
    }
    let px: Vec<Real> =
        (0..n).map(|i| (1.0e-10 / LEN_F) as Real * (i as Real - 3.5)).collect();
    (px, vx, vec![mass; n], vec![per_particle_virial; n])
}

// --- Construction ---

// rq-26b98781
#[test]
fn registry_builds_c_rescale() {
    let gpu = init_device().unwrap();
    let kind = c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 42);
    let baro = unbox_c_rescale(build_c_rescale(&gpu, 4, &kind));
    assert_eq!(baro.draw_counter, 0);
    assert_eq!(baro.cumulative_barostat_injection, 0.0);
    // Stored values are in atomic units after `c_rescale_kind` converts
    // its SI inputs.
    assert_eq!(baro.pressure, 1.0e5 / PRESSURE_F);
    assert_eq!(baro.temperature, 85.0 / TEMP_F);
    assert_eq!(baro.tau, 1.0e-12 / TIME_F);
    assert_eq!(baro.compressibility, 4.5e-10 * PRESSURE_F);
    assert_eq!(baro.seed, 42);
}

// rq-f1b57184
#[test]
fn registry_builds_c_rescale_particle_count_zero() {
    let gpu = init_device().unwrap();
    let kind = c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 1);
    let _baro = build_c_rescale(&gpu, 0, &kind);
}

// rq-1e27ee47
#[test]
fn barostat_registry_exposes_berendsen_and_c_rescale() {
    let registry = BarostatRegistry::with_builtins();
    assert!(
        registry
            .builders
            .iter()
            .any(|b| b.kind_name() == "berendsen")
    );
    assert!(
        registry
            .builders
            .iter()
            .any(|b| b.kind_name() == "c-rescale")
    );
}

// --- Per-step kernel sequence ---

// rq-3e57f675
#[test]
fn apply_launches_expected_kernel_set() {
    let gpu = init_device().unwrap();
    let (px, vx, masses, virials) = system_with_pressure(1.0e5);
    let n = px.len();
    let state = make_state(px, vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small(&gpu);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_c_rescale(&gpu, n, &c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 1));
    baro.apply(&mut buffers, &mut sim_box, (1.0e-15 / TIME_F as Real), &mut timings)
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
    // The compute-mu + lattice-rescale scalar kernel is instrumented. rq-5f59fa80
    assert_eq!(count_for(KernelStage::C_RESCALE_COMPUTE_MU), 1);
    // The per-particle position rescale is dispatched by the
    // JIT-composed post-force per-particle kernel via c-rescale's
    // source fragment, not by `apply`. The standalone
    // `C_RESCALE_BAROSTAT_RESCALE_POSITIONS` stage is never
    // recorded.
    assert_eq!(
        count_for(KernelStage::C_RESCALE_BAROSTAT_RESCALE_POSITIONS),
        0
    );
    // The Berendsen-barostat label must not be touched.
    assert_eq!(
        count_for(KernelStage::BERENDSEN_BAROSTAT_RESCALE_POSITIONS),
        0
    );
    assert_eq!(count_for(KernelStage::VV_KICK_DRIFT), 0);
    assert_eq!(count_for(KernelStage::VV_KICK), 0);
}

// rq-4894ae09
#[test]
fn apply_on_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = make_state(Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small(&gpu);
    let g_before = sim_box.generation();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_c_rescale(&gpu, 0, &c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 1));
    baro.apply(&mut buffers, &mut sim_box, (1.0e-15 / TIME_F as Real), &mut timings)
        .unwrap();
    assert_eq!(sim_box.generation(), g_before);
    // Underlying state's draw_counter should be unchanged when n == 0.
    let s = unbox_c_rescale(baro);
    assert_eq!(s.draw_counter, 0);
    assert_eq!(s.cumulative_barostat_injection, 0.0);
}

// --- draw_counter advances ---

// rq-1f2b5320
#[test]
fn draw_counter_starts_at_zero_and_increments_per_apply() {
    let gpu = init_device().unwrap();
    let (px, vx, masses, virials) = system_with_pressure(1.0e5);
    let n = px.len();
    let state = make_state(px, vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small(&gpu);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro =
        unbox_c_rescale(build_c_rescale(&gpu, n, &c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 1)));
    assert_eq!(baro.draw_counter, 0);
    baro.apply(&mut buffers, &mut sim_box, (1.0e-15 / TIME_F as Real), &mut timings)
        .unwrap();
    assert_eq!(baro.draw_counter, 1);
    baro.apply(&mut buffers, &mut sim_box, (1.0e-15 / TIME_F as Real), &mut timings)
        .unwrap();
    assert_eq!(baro.draw_counter, 2);
}

// rq-cedda168
#[test]
fn two_barostats_at_same_seed_and_counter_produce_identical_outputs() {
    let gpu = init_device().unwrap();
    let (px, vx, masses, virials) = system_with_pressure(1.0e5);
    let n = px.len();
    let state = make_state(px, vx, masses, virials);
    let mut buffers_a = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut buffers_b = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box_a = box_small(&gpu);
    let mut sim_box_b = box_small(&gpu);
    let mut timings_a = Timings::new(&gpu).unwrap();
    let mut timings_b = Timings::new(&gpu).unwrap();
    let mut baro_a = build_c_rescale(&gpu, n, &c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 7));
    let mut baro_b = build_c_rescale(&gpu, n, &c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 7));
    baro_a
        .apply(&mut buffers_a, &mut sim_box_a, (1.0e-15 / TIME_F as Real), &mut timings_a)
        .unwrap();
    baro_b
        .apply(&mut buffers_b, &mut sim_box_b, (1.0e-15 / TIME_F as Real), &mut timings_b)
        .unwrap();
    let (px_a, _, _) = buffers_a.download_positions().unwrap();
    let (px_b, _, _) = buffers_b.download_positions().unwrap();
    assert_eq!(px_a, px_b);
    assert_eq!(sim_box_a.lattice(), sim_box_b.lattice());
}

// --- μ correctness with a known Philox draw ---

// rq-9e5579bb
#[test]
fn mu_cubed_matches_analytical_formula_with_known_philox_draw() {
    // P = P_target / 2, so the deterministic drift is negative
    // (contraction). The noise term either adds or subtracts. We can
    // compute the expected post-rescale volume directly by reproducing
    // the host arithmetic.
    let gpu = init_device().unwrap();
    let p_target = 1.0e6_f64;
    let (px, vx, masses, virials) = system_with_pressure(p_target / 2.0);
    let n = px.len();
    let state = make_state(px, vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small(&gpu);
    let v_pre = sim_box.volume() as f64;
    let dt_si = 1.0e-13_f64;
    let dt = (dt_si / TIME_F) as Real;
    let tau = 1.0e-12_f64;
    let beta = 4.5e-10_f64;
    let temperature = 85.0_f64;
    let seed: u64 = 1;
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_c_rescale(&gpu, n, &c_rescale_kind(p_target, temperature, tau, beta, seed));
    baro.apply(&mut buffers, &mut sim_box, dt, &mut timings)
        .unwrap();
    sim_box.flush_from_device().unwrap();
    let v_post = sim_box.volume() as f64;
    let mu_cubed_actual = v_post / v_pre;

    // Reproduce the host computation in the engine's atomic units.
    // `c_rescale_kind` already converts the SI inputs above; the
    // expected formula must therefore use the matching atomic-unit
    // values, and the Boltzmann constant collapses to 1 (kt = T).
    // The device-resident draw counter starts at 0; the kernel reads
    // it, draws Philox, then increments. The first kernel call therefore
    // uses counter = 0.
    let r = philox_normal(seed as u32, (seed >> 32) as u32, 0, 0, 0, 0);
    let temperature_au = temperature / TEMP_F;
    let p_target_au = p_target / PRESSURE_F;
    let p_current_au = (p_target / 2.0) / PRESSURE_F;
    let tau_au = tau / TIME_F;
    let beta_au = beta * PRESSURE_F;
    let dt_au = dt as f64;
    let kt_au = temperature_au;
    let deterministic = -beta_au * (dt_au / tau_au) * (p_target_au - p_current_au);
    let noise_amplitude = (2.0 * beta_au * kt_au * dt_au / (tau_au * v_pre)).sqrt();
    let expected_mu_cubed = 1.0 + deterministic + noise_amplitude * r;
    let rel = (mu_cubed_actual - expected_mu_cubed).abs() / expected_mu_cubed.abs();
    assert!(
        rel < 1.0e-3,
        "μ³ actual = {mu_cubed_actual}, expected ≈ {expected_mu_cubed} (rel {rel})"
    );
}

// rq-ac873434
#[test]
fn temperature_zero_limit_matches_berendsen_barostat() {
    // With temperature → 0 the noise amplitude → 0; the C-rescale
    // formula reduces to the Berendsen barostat formula. We compare the
    // post-rescale volume produced by a C-rescale barostat with
    // temperature = 1e-30 against the analytical Berendsen formula
    // (computed host-side).
    let gpu = init_device().unwrap();
    let p_target = 1.0e6_f64;
    let (px, vx, masses, virials) = system_with_pressure(p_target / 2.0);
    let n = px.len();
    let state = make_state(px, vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small(&gpu);
    let v_pre = sim_box.volume() as f64;
    let dt_si = 1.0e-13_f64;
    let dt = (dt_si / TIME_F) as Real;
    let tau = 1.0e-12_f64;
    let beta = 4.5e-10_f64;
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_c_rescale(&gpu, n, &c_rescale_kind(p_target, 1.0e-30, tau, beta, 1));
    baro.apply(&mut buffers, &mut sim_box, dt, &mut timings)
        .unwrap();
    sim_box.flush_from_device().unwrap();
    let v_post = sim_box.volume() as f64;
    let mu_cubed_actual = v_post / v_pre;
    // Expected Berendsen μ³ = 1 − β·(dt/τ)·(P_target − P) in atomic units
    let p_target_au = p_target / PRESSURE_F;
    let p_current_au = (p_target / 2.0) / PRESSURE_F;
    let tau_au = tau / TIME_F;
    let beta_au = beta * PRESSURE_F;
    let dt_au = dt as f64;
    let expected_mu_cubed =
        1.0 - beta_au * (dt_au / tau_au) * (p_target_au - p_current_au);
    let rel = (mu_cubed_actual - expected_mu_cubed).abs() / expected_mu_cubed.abs();
    assert!(rel < 1.0e-6, "actual {mu_cubed_actual}, expected {expected_mu_cubed}, rel {rel}");
}

// --- Fractional-coord and shape invariants ---

// rq-3b9e9550
#[test]
fn fractional_coordinates_invariant_under_apply() {
    use heddle_md::gpu::rescale_positions_device_factor;
    let gpu = init_device().unwrap();
    let p_target = 1.0e6_f64;
    let (px, vx, masses, virials) = system_with_pressure(p_target / 2.0);
    let n = px.len();
    let state = make_state(px.clone(), vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small(&gpu);
    let lx_pre = sim_box.lx();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = unbox_c_rescale(build_c_rescale(
        &gpu,
        n,
        &c_rescale_kind(p_target, 85.0, 1.0e-12, 4.5e-10, 1),
    ));
    baro.apply(&mut buffers, &mut sim_box, (1.0e-13 / TIME_F as Real), &mut timings)
        .unwrap();
    // The composed post-force per-particle kernel applies the
    // position rescale in production. Tests that bypass the composed
    // kernel dispatch the standalone equivalent against
    // `mu_device` to keep the post-apply state covered.
    rescale_positions_device_factor(&mut buffers, &baro.mu_device).unwrap();
    sim_box.flush_from_device().unwrap();
    let lx_post = sim_box.lx();
    let (px_post, _, _) = buffers.download_positions().unwrap();
    for (i, (a, b)) in px_post.iter().zip(px.iter()).enumerate() {
        let f_pre = (b / lx_pre) as f64;
        let f_post = (a / lx_post) as f64;
        let rel = (f_post - f_pre).abs() / f_pre.abs().max(1.0e-12);
        assert!(rel < 1.0e-3, "particle {i} fractional drift {rel}");
    }
}

// rq-94d30346
#[test]
fn triclinic_shape_preserved_under_apply() {
    let gpu = init_device().unwrap();
    let p_target = 1.0e6_f64;
    let (px, vx, masses, virials) = system_with_pressure(p_target / 2.0);
    let n = px.len();
    let state = make_state(px, vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box =
        SimulationBox::new(&gpu.device, 1.0e-9, 1.0e-9, 1.0e-9, 0.1e-9, 0.2e-9, 0.3e-9).unwrap();
    let [lx_pre, _, _, xy_pre, xz_pre, yz_pre] = sim_box.lattice();
    let r_xy_pre = (xy_pre / lx_pre) as f64;
    let r_xz_pre = (xz_pre / lx_pre) as f64;
    let r_yz_pre = (yz_pre / lx_pre) as f64;
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_c_rescale(&gpu, n, &c_rescale_kind(p_target, 85.0, 1.0e-12, 4.5e-10, 1));
    baro.apply(&mut buffers, &mut sim_box, (1.0e-13 / TIME_F as Real), &mut timings)
        .unwrap();
    let [lx_post, _, _, xy_post, xz_post, yz_post] = sim_box.lattice();
    let r_xy_post = (xy_post / lx_post) as f64;
    let r_xz_post = (xz_post / lx_post) as f64;
    let r_yz_post = (yz_post / lx_post) as f64;
    assert!((r_xy_post - r_xy_pre).abs() / r_xy_pre.abs() < 1.0e-4);
    assert!((r_xz_post - r_xz_pre).abs() / r_xz_pre.abs() < 1.0e-4);
    assert!((r_yz_post - r_yz_pre).abs() / r_yz_pre.abs() < 1.0e-4);
}

// --- Box-generation propagation ---

// rq-9d2d90b3
#[test]
fn generation_advances_after_apply() {
    let gpu = init_device().unwrap();
    let (px, vx, masses, virials) = system_with_pressure(1.0e5);
    let n = px.len();
    let state = make_state(px, vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small(&gpu);
    let g_pre = sim_box.generation();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_c_rescale(&gpu, n, &c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 1));
    baro.apply(&mut buffers, &mut sim_box, (1.0e-15 / TIME_F as Real), &mut timings)
        .unwrap();
    assert_eq!(sim_box.generation(), g_pre + 1);
}

// --- Log columns ---

// rq-e5cb5505 rq-df305128
#[test]
fn log_column_names_returns_pressure_volume_and_conserved() {
    let gpu = init_device().unwrap();
    let baro = build_c_rescale(&gpu, 4, &c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 1));
    let names: Vec<&str> = baro.log_column_names().iter().map(|(n, _)| *n).collect();
    assert_eq!(names, vec!["pressure", "box_volume", "c_rescale_conserved"]);
}

// rq-fb78338b
#[test]
fn log_column_values_combines_pressure_volume_and_cumulative_injection() {
    let gpu = init_device().unwrap();
    // Build the barostat in atomic units (c_rescale_kind converts SI
    // inputs); after construction, override the diagnostic-only
    // fields to known atomic-unit values so the formula composition
    // can be verified in isolation.
    let mut baro =
        unbox_c_rescale(build_c_rescale(&gpu, 4, &c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 1)));
    let pressure_au = baro.pressure; // = 1.0e5 / PRESSURE_F (atomic)
    let volume_au = 1.0e-27 / (LEN_F * LEN_F * LEN_F);
    baro.most_recent_pressure = 1.01e5 / PRESSURE_F;
    baro.most_recent_volume = volume_au;
    baro.cumulative_barostat_injection = 3.0e-22 / (4.359744722207101e-18);
    let ke = 1.5e-20 / 4.359744722207101e-18;
    let pe = 2.0e-20 / 4.359744722207101e-18;
    let extras = baro.log_column_values(ke, pe);
    assert_eq!(extras.len(), 3);
    assert_eq!(extras[0], 1.01e5 / PRESSURE_F);
    assert_eq!(extras[1], volume_au);
    let expected_conserved: f64 = ke + pe + pressure_au * volume_au
        - baro.cumulative_barostat_injection;
    assert!((extras[2] - expected_conserved).abs() < 1.0e-20 * expected_conserved.abs().max(1.0));
}

// --- Composition with the orthogonal framework ---

// rq-0f3f63c8 rq-2d109b3a
#[test]
fn composes_with_velocity_verlet_and_csvr_thermostat() {
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
                "csvr",
                "temperature = 85.0\ntau = 1.0e-13\nseed = 17\n",
            )),
            &gpu,
            n,
            0,
        )
        .unwrap()
        .unwrap();
    let mut baro = build_c_rescale(&gpu, n, &c_rescale_kind(1.0e5, 85.0, 1.0e-12, 1.0e-9, 19));
    ff.step(&mut buffers, &sim_box, &mut timings, AggregateLevel::ForcesAndScalars).unwrap();
    for _ in 0..10 {
        // Order: thermostat.apply_pre (trait default no-op for CSVR)
        // → integrator.step → thermostat.apply_post → barostat.apply
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, (1.0e-15 / TIME_F as Real), &mut timings)
            .unwrap();
        therm
            .apply_post(&mut buffers, (1.0e-15 / TIME_F as Real), &mut timings)
            .unwrap();
        baro.apply(&mut buffers, &mut sim_box, (1.0e-15 / TIME_F as Real), &mut timings)
            .unwrap();
    }
    sim_box.flush_from_device().unwrap();
    let v_final = sim_box.volume();
    assert!(v_final.is_finite() && v_final > 0.0);
}

// --- Determinism ---

// rq-6e0d6cb4
#[test]
fn two_runs_with_same_seed_are_byte_identical() {
    let gpu = init_device().unwrap();
    let p_target = 1.0e6_f64;
    let (px, vx, masses, virials) = system_with_pressure(p_target / 2.0);

    fn run_once(
        gpu: &GpuContext,
        px: &[Real],
        vx: &[Real],
        masses: &[Real],
        virials: &[Real],
        seed: u64,
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
        let mut baro = build_c_rescale(gpu, n, &c_rescale_kind(1.0e6, 85.0, 1.0e-12, 4.5e-10, seed));
        for _ in 0..5 {
            baro.apply(&mut buffers, &mut sim_box, (1.0e-15 / TIME_F as Real), &mut timings)
                .unwrap();
        }
        let (positions_x, _, _) = buffers.download_positions().unwrap();
        (positions_x, sim_box.lattice())
    }

    let (px_a, lat_a) = run_once(&gpu, &px, &vx, &masses, &virials, 42);
    let (px_b, lat_b) = run_once(&gpu, &px, &vx, &masses, &virials, 42);
    assert_eq!(px_a, px_b);
    for i in 0..6 {
        assert_eq!(lat_a[i].to_bits(), lat_b[i].to_bits());
    }
}

// rq-efc07f81
#[test]
fn different_seeds_produce_different_trajectories() {
    let gpu = init_device().unwrap();
    let p_target = 1.0e6_f64;
    let (px, vx, masses, virials) = system_with_pressure(p_target / 2.0);

    fn run_once(
        gpu: &GpuContext,
        px: &[Real],
        vx: &[Real],
        masses: &[Real],
        virials: &[Real],
        seed: u64,
    ) -> Vec<Real> {
        use heddle_md::gpu::rescale_positions_device_factor;
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
        let mut baro = unbox_c_rescale(build_c_rescale(
            gpu,
            n,
            &c_rescale_kind(1.0e6, 85.0, 1.0e-12, 4.5e-10, seed),
        ));
        for _ in 0..5 {
            baro.apply(&mut buffers, &mut sim_box, (1.0e-15 / TIME_F as Real), &mut timings)
                .unwrap();
            rescale_positions_device_factor(&mut buffers, &baro.mu_device).unwrap();
        }
        buffers.download_positions().unwrap().0
    }

    let a = run_once(&gpu, &px, &vx, &masses, &virials, 1);
    let b = run_once(&gpu, &px, &vx, &masses, &virials, 2);
    assert_ne!(a, b);
}
