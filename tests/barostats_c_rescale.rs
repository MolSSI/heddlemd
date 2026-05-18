// rq-11f5dfd1
//
// Stochastic cell-rescaling (C-rescale) barostat tests. The barostat
// is exercised in isolation through its `apply` hook; the shared
// `compute_total_virial` and `rescale_positions` helpers and the
// `SimulationBox::rescale_isotropic` convenience are covered in
// `tests/barostats_berendsen.rs` and not re-tested here.

use dynamics::forces::{AngleList, BondList, ExclusionList, ForceField};
use dynamics::gpu::{GpuContext, ParticleBuffers, init_device};
use dynamics::integrator::IntegratorStepExt;
use dynamics::integrator::{
    Barostat, BarostatRegistry, CRescaleBarostat, IntegratorRegistry, ThermostatRegistry,
    philox_normal,
};
use dynamics::io::config::NeighborListConfig;
use dynamics::io::SlotConfig;
use dynamics::pbc::SimulationBox;
use dynamics::state::ParticleState;
use dynamics::timings::{KernelStage, Timings};

const KB: f64 = 1.380649e-23;

fn box_small() -> SimulationBox {
    SimulationBox::new(1.0e-9, 1.0e-9, 1.0e-9, 0.0, 0.0, 0.0).unwrap()
}

fn empty_force_field(gpu: &GpuContext, n: usize) -> ForceField {
    ForceField::new(
        gpu,
        n,
        &box_small(),
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
        .build_optional(Some(slot), gpu, n)
        .unwrap()
        .unwrap()
}

fn unbox_c_rescale(boxed: Box<dyn Barostat>) -> CRescaleBarostat {
    unsafe { *Box::from_raw(Box::into_raw(boxed) as *mut CRescaleBarostat) }
}

fn make_state(
    positions_x: Vec<f32>,
    velocities_x: Vec<f32>,
    masses: Vec<f32>,
    virials: Vec<f32>,
) -> ParticleState {
    let n = positions_x.len();
    let zero = vec![0.0_f32; n];
    let state = ParticleState::new(
        positions_x,
        zero.clone(),
        zero.clone(),
        velocities_x,
        zero.clone(),
        zero.clone(),
        masses,
        vec![0.0_f32; n],
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
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = 8;
    let mass: f32 = 1.66e-27;
    let v_mag: f32 = 500.0;
    let v_squared_sum = (n as f64) * (v_mag as f64).powi(2);
    let k = 0.5 * (mass as f64) * v_squared_sum;
    let v = (1.0e-9_f64).powi(3);
    let w_required = 3.0 * v * target_pressure_pa - 2.0 * k;
    let per_particle_virial = (w_required / n as f64) as f32;
    let mut vx: Vec<f32> = Vec::with_capacity(n);
    for _ in 0..n / 2 {
        vx.push(v_mag);
        vx.push(-v_mag);
    }
    let px: Vec<f32> = (0..n).map(|i| 1.0e-10 * (i as f32 - 3.5)).collect();
    (px, vx, vec![mass; n], vec![per_particle_virial; n])
}

// --- Construction ---

#[test]
fn registry_builds_c_rescale() {
    let gpu = init_device().unwrap();
    let kind = c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 42);
    let baro = unbox_c_rescale(build_c_rescale(&gpu, 4, &kind));
    assert_eq!(baro.draw_counter, 0);
    assert_eq!(baro.cumulative_barostat_injection, 0.0);
    assert_eq!(baro.pressure, 1.0e5);
    assert_eq!(baro.temperature, 85.0);
    assert_eq!(baro.tau, 1.0e-12);
    assert_eq!(baro.compressibility, 4.5e-10);
    assert_eq!(baro.seed, 42);
}

#[test]
fn registry_builds_c_rescale_particle_count_zero() {
    let gpu = init_device().unwrap();
    let kind = c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 1);
    let _baro = build_c_rescale(&gpu, 0, &kind);
}

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

#[test]
fn apply_launches_expected_kernel_set() {
    let gpu = init_device().unwrap();
    let (px, vx, masses, virials) = system_with_pressure(1.0e5);
    let n = px.len();
    let state = make_state(px, vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_c_rescale(&gpu, n, &c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 1));
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
    assert_eq!(
        count_for(KernelStage::C_RESCALE_BAROSTAT_RESCALE_POSITIONS),
        1
    );
    // The Berendsen-barostat label must not be touched.
    assert_eq!(
        count_for(KernelStage::BERENDSEN_BAROSTAT_RESCALE_POSITIONS),
        0
    );
    assert_eq!(count_for(KernelStage::VV_KICK_DRIFT), 0);
    assert_eq!(count_for(KernelStage::VV_KICK), 0);
}

#[test]
fn apply_on_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = make_state(Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small();
    let g_before = sim_box.generation();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_c_rescale(&gpu, 0, &c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 1));
    baro.apply(&mut buffers, &mut sim_box, 1.0e-15, &mut timings)
        .unwrap();
    assert_eq!(sim_box.generation(), g_before);
    // Underlying state's draw_counter should be unchanged when n == 0.
    let s = unbox_c_rescale(baro);
    assert_eq!(s.draw_counter, 0);
    assert_eq!(s.cumulative_barostat_injection, 0.0);
}

// --- draw_counter advances ---

#[test]
fn draw_counter_starts_at_zero_and_increments_per_apply() {
    let gpu = init_device().unwrap();
    let (px, vx, masses, virials) = system_with_pressure(1.0e5);
    let n = px.len();
    let state = make_state(px, vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro =
        unbox_c_rescale(build_c_rescale(&gpu, n, &c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 1)));
    assert_eq!(baro.draw_counter, 0);
    baro.apply(&mut buffers, &mut sim_box, 1.0e-15, &mut timings)
        .unwrap();
    assert_eq!(baro.draw_counter, 1);
    baro.apply(&mut buffers, &mut sim_box, 1.0e-15, &mut timings)
        .unwrap();
    assert_eq!(baro.draw_counter, 2);
}

#[test]
fn two_barostats_at_same_seed_and_counter_produce_identical_outputs() {
    let gpu = init_device().unwrap();
    let (px, vx, masses, virials) = system_with_pressure(1.0e5);
    let n = px.len();
    let state = make_state(px, vx, masses, virials);
    let mut buffers_a = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut buffers_b = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box_a = box_small();
    let mut sim_box_b = box_small();
    let mut timings_a = Timings::new(&gpu).unwrap();
    let mut timings_b = Timings::new(&gpu).unwrap();
    let mut baro_a = build_c_rescale(&gpu, n, &c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 7));
    let mut baro_b = build_c_rescale(&gpu, n, &c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 7));
    baro_a
        .apply(&mut buffers_a, &mut sim_box_a, 1.0e-15, &mut timings_a)
        .unwrap();
    baro_b
        .apply(&mut buffers_b, &mut sim_box_b, 1.0e-15, &mut timings_b)
        .unwrap();
    let px_a = gpu.device.dtoh_sync_copy(&buffers_a.positions_x).unwrap();
    let px_b = gpu.device.dtoh_sync_copy(&buffers_b.positions_x).unwrap();
    assert_eq!(px_a, px_b);
    assert_eq!(sim_box_a.lattice(), sim_box_b.lattice());
}

// --- μ correctness with a known Philox draw ---

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
    let mut sim_box = box_small();
    let v_pre = sim_box.volume() as f64;
    let dt = 1.0e-13_f32;
    let tau = 1.0e-12_f64;
    let beta = 4.5e-10_f64;
    let temperature = 85.0_f64;
    let seed: u64 = 1;
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_c_rescale(&gpu, n, &c_rescale_kind(p_target, temperature, tau, beta, seed));
    baro.apply(&mut buffers, &mut sim_box, dt, &mut timings)
        .unwrap();
    let v_post = sim_box.volume() as f64;
    let mu_cubed_actual = v_post / v_pre;

    // Reproduce the host computation.
    let r = philox_normal(seed as u32, (seed >> 32) as u32, 1, 0, 0, 0);
    let kt = KB * temperature;
    let deterministic = -beta * ((dt as f64) / tau) * (p_target - p_target / 2.0);
    let noise_amplitude = (2.0 * beta * kt * (dt as f64) / (tau * v_pre)).sqrt();
    let expected_mu_cubed = 1.0 + deterministic + noise_amplitude * r;
    let rel = (mu_cubed_actual - expected_mu_cubed).abs() / expected_mu_cubed.abs();
    assert!(
        rel < 1.0e-3,
        "μ³ actual = {mu_cubed_actual}, expected ≈ {expected_mu_cubed} (rel {rel})"
    );
}

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
    let mut sim_box = box_small();
    let v_pre = sim_box.volume() as f64;
    let dt = 1.0e-13_f32;
    let tau = 1.0e-12_f64;
    let beta = 4.5e-10_f64;
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_c_rescale(&gpu, n, &c_rescale_kind(p_target, 1.0e-30, tau, beta, 1));
    baro.apply(&mut buffers, &mut sim_box, dt, &mut timings)
        .unwrap();
    let v_post = sim_box.volume() as f64;
    let mu_cubed_actual = v_post / v_pre;
    // Expected Berendsen μ³ = 1 − β·(dt/τ)·(P_target − P)
    let expected_mu_cubed = 1.0 - beta * ((dt as f64) / tau) * (p_target - p_target / 2.0);
    let rel = (mu_cubed_actual - expected_mu_cubed).abs() / expected_mu_cubed.abs();
    assert!(rel < 1.0e-6);
}

// --- Fractional-coord and shape invariants ---

#[test]
fn fractional_coordinates_invariant_under_apply() {
    let gpu = init_device().unwrap();
    let p_target = 1.0e6_f64;
    let (px, vx, masses, virials) = system_with_pressure(p_target / 2.0);
    let n = px.len();
    let state = make_state(px.clone(), vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small();
    let lx_pre = sim_box.lx();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_c_rescale(&gpu, n, &c_rescale_kind(p_target, 85.0, 1.0e-12, 4.5e-10, 1));
    baro.apply(&mut buffers, &mut sim_box, 1.0e-13, &mut timings)
        .unwrap();
    let lx_post = sim_box.lx();
    let px_post = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    for (i, (a, b)) in px_post.iter().zip(px.iter()).enumerate() {
        let f_pre = (b / lx_pre) as f64;
        let f_post = (a / lx_post) as f64;
        let rel = (f_post - f_pre).abs() / f_pre.abs().max(1.0e-12);
        assert!(rel < 1.0e-3, "particle {i} fractional drift {rel}");
    }
}

#[test]
fn triclinic_shape_preserved_under_apply() {
    let gpu = init_device().unwrap();
    let p_target = 1.0e6_f64;
    let (px, vx, masses, virials) = system_with_pressure(p_target / 2.0);
    let n = px.len();
    let state = make_state(px, vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box =
        SimulationBox::new(1.0e-9, 1.0e-9, 1.0e-9, 0.1e-9, 0.2e-9, 0.3e-9).unwrap();
    let [lx_pre, _, _, xy_pre, xz_pre, yz_pre] = sim_box.lattice();
    let r_xy_pre = (xy_pre / lx_pre) as f64;
    let r_xz_pre = (xz_pre / lx_pre) as f64;
    let r_yz_pre = (yz_pre / lx_pre) as f64;
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_c_rescale(&gpu, n, &c_rescale_kind(p_target, 85.0, 1.0e-12, 4.5e-10, 1));
    baro.apply(&mut buffers, &mut sim_box, 1.0e-13, &mut timings)
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

#[test]
fn generation_advances_after_apply() {
    let gpu = init_device().unwrap();
    let (px, vx, masses, virials) = system_with_pressure(1.0e5);
    let n = px.len();
    let state = make_state(px, vx, masses, virials);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small();
    let g_pre = sim_box.generation();
    let mut timings = Timings::new(&gpu).unwrap();
    let mut baro = build_c_rescale(&gpu, n, &c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 1));
    baro.apply(&mut buffers, &mut sim_box, 1.0e-15, &mut timings)
        .unwrap();
    assert_eq!(sim_box.generation(), g_pre + 1);
}

// --- Log columns ---

#[test]
fn log_column_names_returns_pressure_volume_and_conserved() {
    let gpu = init_device().unwrap();
    let baro = build_c_rescale(&gpu, 4, &c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 1));
    assert_eq!(
        baro.log_column_names(),
        &["pressure", "box_volume", "c_rescale_conserved"]
    );
}

#[test]
fn log_column_values_combines_pressure_volume_and_cumulative_injection() {
    let gpu = init_device().unwrap();
    let mut baro =
        unbox_c_rescale(build_c_rescale(&gpu, 4, &c_rescale_kind(1.0e5, 85.0, 1.0e-12, 4.5e-10, 1)));
    baro.most_recent_pressure = 1.01e5;
    baro.most_recent_volume = 1.0e-27;
    baro.cumulative_barostat_injection = 3.0e-22;
    let ke = 1.5e-20_f64;
    let pe = 2.0e-20_f64;
    let extras = baro.log_column_values(ke, pe);
    assert_eq!(extras.len(), 3);
    assert_eq!(extras[0], 1.01e5);
    assert_eq!(extras[1], 1.0e-27);
    let expected_conserved: f64 = ke + pe + 1.0e5 * 1.0e-27 - 3.0e-22;
    assert!((extras[2] - expected_conserved).abs() < 1.0e-30);
}

// --- Composition with the orthogonal framework ---

#[test]
fn composes_with_velocity_verlet_and_csvr_thermostat() {
    let gpu = init_device().unwrap();
    let n = 8usize;
    let mass: f32 = 1.66e-27;
    let v_mag: f32 = 500.0;
    let mut vx: Vec<f32> = Vec::with_capacity(n);
    for _ in 0..n / 2 {
        vx.push(v_mag);
        vx.push(-v_mag);
    }
    let state = make_state(
        (0..n).map(|i| 1.0e-10 * (i as f32 - 3.5)).collect(),
        vx,
        vec![mass; n],
        vec![0.0_f32; n],
    );
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integ = IntegratorRegistry::with_builtins()
        .build(
            &SlotConfig::from_params_str("velocity-verlet", "lossless = false"),
            &gpu,
            n,
        )
        .unwrap();
    let mut therm = ThermostatRegistry::with_builtins()
        .build_optional(
            Some(&SlotConfig::from_params_str(
                "csvr",
                "temperature = 85.0\ntau = 1.0e-13\nseed = 17\n",
            )),
            &gpu,
            n,
        )
        .unwrap()
        .unwrap();
    let mut baro = build_c_rescale(&gpu, n, &c_rescale_kind(1.0e5, 85.0, 1.0e-12, 1.0e-9, 19));
    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
    for _ in 0..10 {
        // Order: thermostat.apply_pre (trait default no-op for CSVR)
        // → integrator.step → thermostat.apply_post → barostat.apply
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, None, 1.0e-15, &mut timings)
            .unwrap();
        therm
            .apply_post(&mut buffers, 1.0e-15, &mut timings)
            .unwrap();
        baro.apply(&mut buffers, &mut sim_box, 1.0e-15, &mut timings)
            .unwrap();
    }
    let v_final = sim_box.volume();
    assert!(v_final.is_finite() && v_final > 0.0);
}

// --- Determinism ---

#[test]
fn two_runs_with_same_seed_are_byte_identical() {
    let gpu = init_device().unwrap();
    let p_target = 1.0e6_f64;
    let (px, vx, masses, virials) = system_with_pressure(p_target / 2.0);

    fn run_once(
        gpu: &GpuContext,
        px: &[f32],
        vx: &[f32],
        masses: &[f32],
        virials: &[f32],
        seed: u64,
    ) -> (Vec<f32>, [f32; 6]) {
        let n = px.len();
        let state = make_state(
            px.to_vec(),
            vx.to_vec(),
            masses.to_vec(),
            virials.to_vec(),
        );
        let mut buffers = ParticleBuffers::new(gpu, &state).unwrap();
        let mut sim_box = box_small();
        let mut timings = Timings::new(gpu).unwrap();
        let mut baro = build_c_rescale(gpu, n, &c_rescale_kind(1.0e6, 85.0, 1.0e-12, 4.5e-10, seed));
        for _ in 0..5 {
            baro.apply(&mut buffers, &mut sim_box, 1.0e-15, &mut timings)
                .unwrap();
        }
        let positions_x = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
        (positions_x, sim_box.lattice())
    }

    let (px_a, lat_a) = run_once(&gpu, &px, &vx, &masses, &virials, 42);
    let (px_b, lat_b) = run_once(&gpu, &px, &vx, &masses, &virials, 42);
    assert_eq!(px_a, px_b);
    for i in 0..6 {
        assert_eq!(lat_a[i].to_bits(), lat_b[i].to_bits());
    }
}

#[test]
fn different_seeds_produce_different_trajectories() {
    let gpu = init_device().unwrap();
    let p_target = 1.0e6_f64;
    let (px, vx, masses, virials) = system_with_pressure(p_target / 2.0);

    fn run_once(
        gpu: &GpuContext,
        px: &[f32],
        vx: &[f32],
        masses: &[f32],
        virials: &[f32],
        seed: u64,
    ) -> Vec<f32> {
        let n = px.len();
        let state = make_state(
            px.to_vec(),
            vx.to_vec(),
            masses.to_vec(),
            virials.to_vec(),
        );
        let mut buffers = ParticleBuffers::new(gpu, &state).unwrap();
        let mut sim_box = box_small();
        let mut timings = Timings::new(gpu).unwrap();
        let mut baro = build_c_rescale(gpu, n, &c_rescale_kind(1.0e6, 85.0, 1.0e-12, 4.5e-10, seed));
        for _ in 0..5 {
            baro.apply(&mut buffers, &mut sim_box, 1.0e-15, &mut timings)
                .unwrap();
        }
        gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap()
    }

    let a = run_once(&gpu, &px, &vx, &masses, &virials, 1);
    let b = run_once(&gpu, &px, &vx, &masses, &virials, 2);
    assert_ne!(a, b);
}
