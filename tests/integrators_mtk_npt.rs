// rq-3b6d5001
//
// MTK NPT integrator (isotropic) tests. The integrator is fused
// (owns its thermostat and barostat), so it is exercised through
// IntegratorRegistry::build + Integrator::step. The two new CUDA
// kernels (mtk_velocity_half_kick, mtk_position_drift) and the
// shared nhc_chain_sub_step host helper are also exercised directly.

use dynamics::forces::{AngleList, BondList, ExclusionList, ForceField, PotentialRegistry};
use dynamics::gpu::{
    GpuContext, ParticleBuffers, init_device, mtk_position_drift, mtk_velocity_half_kick,
};
use dynamics::integrator::IntegratorStepExt;
use dynamics::integrator::{
    Integrator, IntegratorRegistry, MtkNptIntegrator, nhc_chain_sub_step,
};
use dynamics::io::SlotConfig;
use dynamics::io::config::NeighborListConfig;
use dynamics::pbc::SimulationBox;
use dynamics::state::ParticleState;
use dynamics::timings::{KernelStage, Timings};

const KB: f64 = 1.380649e-23;

fn box_small() -> SimulationBox {
    SimulationBox::new(1.0e-9, 1.0e-9, 1.0e-9, 0.0, 0.0, 0.0).unwrap()
}

fn empty_force_field(gpu: &GpuContext, n: usize) -> ForceField {
    ForceField::new(
        &PotentialRegistry::with_builtins(),
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

fn mtk_kind(
    temperature: f64,
    pressure: f64,
    tau_t: f64,
    tau_p: f64,
    chain_length: u32,
    yoshida_order: u32,
    n_resp: u32,
) -> SlotConfig {
    SlotConfig::from_params_str(
        "mtk-npt",
        &format!(
            "temperature = {temperature:e}\n\
             pressure = {pressure:e}\n\
             tau_t = {tau_t:e}\n\
             tau_p = {tau_p:e}\n\
             chain_length = {chain_length}\n\
             yoshida_order = {yoshida_order}\n\
             n_resp = {n_resp}\n"
        ),
    )
}

fn build_mtk(gpu: &GpuContext, n: usize, slot: &SlotConfig) -> Box<dyn Integrator> {
    IntegratorRegistry::with_builtins()
        .build(slot, gpu, n)
        .unwrap()
}

fn unbox_mtk(boxed: Box<dyn Integrator>) -> MtkNptIntegrator {
    unsafe { *Box::from_raw(Box::into_raw(boxed) as *mut MtkNptIntegrator) }
}

// Build a state with prescribed positions, velocities, masses, virials.
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

// Convenience: build an N=8 system with symmetric ±v pairs (COM=0)
// and zero virials, suitable for empty_force_field tests.
fn symmetric_state(n: usize, mass: f32, v_mag: f32) -> ParticleState {
    assert!(n.is_multiple_of(2));
    let mut vx: Vec<f32> = Vec::with_capacity(n);
    for _ in 0..n / 2 {
        vx.push(v_mag);
        vx.push(-v_mag);
    }
    make_state(
        (0..n).map(|i| 1.0e-10 * (i as f32 - (n as f32) / 2.0 + 0.5)).collect(),
        vx,
        vec![mass; n],
        vec![0.0_f32; n],
    )
}

// --- Construction ---

#[test]
fn registry_builds_mtk_npt_with_defaults() {
    let gpu = init_device().unwrap();
    let kind = mtk_kind(85.0, 1.0e5, 1.0e-13, 1.0e-12, 3, 3, 1);
    let state = unbox_mtk(build_mtk(&gpu, 4, &kind));
    assert_eq!(state.chain_length, 3);
    assert_eq!(state.xi_part, vec![0.0, 0.0, 0.0]);
    assert_eq!(state.p_xi_part, vec![0.0, 0.0, 0.0]);
    assert_eq!(state.xi_cell, vec![0.0, 0.0, 0.0]);
    assert_eq!(state.p_xi_cell, vec![0.0, 0.0, 0.0]);
    assert_eq!(state.p_eps, 0.0);
    assert_eq!(state.eps, 0.0);
    let g_dof = state.g_dof as f64;
    let expected_w = (g_dof + 3.0) * KB * 85.0 * (1.0e-12_f64).powi(2);
    assert!((state.w_cell - expected_w).abs() / expected_w < 1.0e-10);
    let expected_q1 = g_dof * KB * 85.0 * (1.0e-13_f64).powi(2);
    assert!((state.q_mass_part[0] - expected_q1).abs() / expected_q1 < 1.0e-10);
}

#[test]
fn registry_builds_mtk_npt_with_chain_length_1() {
    let gpu = init_device().unwrap();
    let kind = mtk_kind(85.0, 1.0e5, 1.0e-13, 1.0e-12, 1, 3, 1);
    let state = unbox_mtk(build_mtk(&gpu, 4, &kind));
    assert_eq!(state.xi_part.len(), 1);
    assert_eq!(state.xi_cell.len(), 1);
}

#[test]
fn registry_builds_mtk_npt_with_particle_count_zero() {
    let gpu = init_device().unwrap();
    let kind = mtk_kind(85.0, 1.0e5, 1.0e-13, 1.0e-12, 3, 3, 1);
    let state = unbox_mtk(build_mtk(&gpu, 0, &kind));
    assert_eq!(state.g_dof, 1); // max(1, 3·0 − 3) clamped to 1
}

// --- Ownership flags ---

#[test]
fn mtk_npt_owns_thermostat_and_barostat() {
    use dynamics::integrator::IntegratorRegistry;
    let kind = mtk_kind(85.0, 1.0e5, 1.0e-13, 1.0e-12, 3, 3, 1);
    let registry = IntegratorRegistry::with_builtins();
    let b = registry.lookup(&kind.kind).unwrap();
    assert!(b.owns_thermostat(&kind.params));
    assert!(b.owns_barostat(&kind.params));
}

// --- Per-step kernel sequence ---

#[test]
fn step_launches_expected_kernel_set() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let mass: f32 = 1.66e-27;
    let state = symmetric_state(n, mass, 500.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integ = build_mtk(&gpu, n, &mtk_kind(85.0, 1.0e5, 1.0e-13, 1.0e-12, 3, 3, 1));
    // Warm up the force pipeline so virials are populated before the
    // first step (matches the runner's contract).
    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
    integ
        .step(&mut buffers, &mut sim_box, &mut ff, None, 1.0e-15, &mut timings)
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
    // Three KE reductions per step: pre, post-drift, post-vel-kick-2.
    assert_eq!(count_for(KernelStage::KINETIC_ENERGY_REDUCE), 3);
    // Two virial reductions: pre and post-drift.
    assert_eq!(count_for(KernelStage::VIRIAL_SUM_REDUCE), 2);
    // 6 particle-chain rescales (3 Yoshida × 1 RESP × 2 halves).
    assert_eq!(count_for(KernelStage::MTK_NPT_RESCALE_VELOCITIES), 6);
    // 2 velocity half-kicks.
    assert_eq!(count_for(KernelStage::MTK_NPT_VELOCITY_HALF_KICK), 2);
    // 1 position drift.
    assert_eq!(count_for(KernelStage::MTK_NPT_POSITION_DRIFT), 1);
    // The integrator never uses the plain VV kernels.
    assert_eq!(count_for(KernelStage::VV_KICK_DRIFT), 0);
    assert_eq!(count_for(KernelStage::VV_KICK), 0);
}

#[test]
fn step_on_empty_state_is_noop() {
    let gpu = init_device().unwrap();
    let state = make_state(Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small();
    let mut ff = empty_force_field(&gpu, 0);
    let mut timings = Timings::new(&gpu).unwrap();
    let g_pre = sim_box.generation();
    let mut integ = build_mtk(&gpu, 0, &mtk_kind(85.0, 1.0e5, 1.0e-13, 1.0e-12, 3, 3, 1));
    integ
        .step(&mut buffers, &mut sim_box, &mut ff, None, 1.0e-15, &mut timings)
        .unwrap();
    assert_eq!(sim_box.generation(), g_pre);
    let s = unbox_mtk(integ);
    assert_eq!(s.p_eps, 0.0);
    assert_eq!(s.eps, 0.0);
}

// --- Cell-coupled kernels (identity-mode checks) ---

#[test]
fn mtk_velocity_half_kick_identity_mode_matches_half_vv_kick() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    // m = 1 so F/m = F. Known v, F.
    let state = ParticleState::new(
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![1.0_f32, 2.0, 3.0, 4.0],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![1.0_f32; n],
        vec![0.0_f32; n],
        (0..n as u32).collect(),
        None,
        None,
    )
    .unwrap();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    // Write known forces.
    let fx = vec![10.0_f32, -5.0, 1.0, 0.0];
    let fy = vec![0.0_f32; n];
    let fz = vec![0.0_f32; n];
    gpu.device
        .htod_sync_copy_into(&fx, &mut buffers.forces_x)
        .unwrap();
    gpu.device
        .htod_sync_copy_into(&fy, &mut buffers.forces_y)
        .unwrap();
    gpu.device
        .htod_sync_copy_into(&fz, &mut buffers.forces_z)
        .unwrap();
    // exp_minus_alpha = 1, phi_v_dt_half = 0.5 → v ← v + 0.5·F
    mtk_velocity_half_kick(&mut buffers, 1.0, 0.5).unwrap();
    let vx_post = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let expected: Vec<f32> = (0..n).map(|i| state.velocities_x[i] + 0.5 * fx[i]).collect();
    for (a, b) in vx_post.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1.0e-5, "{a} vs {b}");
    }
}

#[test]
fn mtk_position_drift_identity_mode_matches_plain_drift() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = ParticleState::new(
        vec![1.0_f32, 2.0, -3.0, 0.5],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![0.5_f32, -1.0, 2.0, 0.0],
        vec![0.0_f32; n],
        vec![0.0_f32; n],
        vec![1.0_f32; n],
        vec![0.0_f32; n],
        (0..n as u32).collect(),
        None,
        None,
    )
    .unwrap();
    let snap_px = state.positions_x.clone();
    let snap_vx = state.velocities_x.clone();
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    // exp_b_dt = 1, phi_x_dt = 0.1 → x ← x + 0.1·v
    mtk_position_drift(&mut buffers, 1.0, 0.1).unwrap();
    let px_post = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    for i in 0..n {
        let expected = snap_px[i] + 0.1 * snap_vx[i];
        assert!((px_post[i] - expected).abs() < 1.0e-5);
    }
}

#[test]
fn mtk_kernels_empty_state_are_noops() {
    let gpu = init_device().unwrap();
    let state = make_state(Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    mtk_velocity_half_kick(&mut buffers, 1.0, 1.0).unwrap();
    mtk_position_drift(&mut buffers, 1.0, 1.0).unwrap();
}

// --- Shared chain helper ---

#[test]
fn nhc_chain_sub_step_m_eq_1_runs_without_panicking() {
    let mut xi = vec![0.0_f64; 1];
    let mut p_xi = vec![0.0_f64; 1];
    let q_mass = vec![1.0e-50_f64];
    let factor = nhc_chain_sub_step(
        &mut xi, &mut p_xi, &q_mass, 1.0e-15, 2.0e-20, 100.0, KB * 85.0,
    );
    assert!(factor.is_finite());
    assert!(factor > 0.0);
}

#[test]
fn nhc_chain_sub_step_m_eq_0_is_identity() {
    let mut xi: Vec<f64> = Vec::new();
    let mut p_xi: Vec<f64> = Vec::new();
    let q_mass: Vec<f64> = Vec::new();
    let factor = nhc_chain_sub_step(
        &mut xi, &mut p_xi, &q_mass, 1.0e-15, 2.0e-20, 100.0, KB * 85.0,
    );
    assert_eq!(factor, 1.0);
}

#[test]
fn nhc_chain_sub_step_is_pure_function() {
    // Calling twice with the same inputs (but on independent buffers)
    // produces the same output.
    let xi_orig = vec![1.0e-3_f64, 2.0e-3, 3.0e-3];
    let p_xi_orig = vec![1.0e-25_f64, -2.0e-25, 5.0e-25];
    let q_mass = vec![1.0e-50_f64, 1.0e-52, 1.0e-52];
    let mut xi_a = xi_orig.clone();
    let mut p_xi_a = p_xi_orig.clone();
    let mut xi_b = xi_orig.clone();
    let mut p_xi_b = p_xi_orig.clone();
    let f_a = nhc_chain_sub_step(
        &mut xi_a, &mut p_xi_a, &q_mass, 1.0e-15, 5.0e-20, 9.0, KB * 85.0,
    );
    let f_b = nhc_chain_sub_step(
        &mut xi_b, &mut p_xi_b, &q_mass, 1.0e-15, 5.0e-20, 9.0, KB * 85.0,
    );
    assert_eq!(f_a.to_bits(), f_b.to_bits());
    assert_eq!(xi_a, xi_b);
    assert_eq!(p_xi_a, p_xi_b);
}

// --- Box-generation propagation ---

#[test]
fn generation_advances_every_step() {
    let gpu = init_device().unwrap();
    let n = 4usize;
    let state = symmetric_state(n, 1.66e-27, 500.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integ = build_mtk(&gpu, n, &mtk_kind(85.0, 1.0e5, 1.0e-13, 1.0e-12, 3, 3, 1));
    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
    let g0 = sim_box.generation();
    integ
        .step(&mut buffers, &mut sim_box, &mut ff, None, 1.0e-15, &mut timings)
        .unwrap();
    // step() calls sim_box.rescale_isotropic once, which bumps the
    // generation by 1. (The runner-level force_field.step before the
    // integrator does not bump it.)
    assert!(sim_box.generation() >= g0 + 1);
}

// --- Log columns ---

#[test]
fn log_column_names_returns_pressure_volume_and_conserved() {
    let gpu = init_device().unwrap();
    let kind = mtk_kind(85.0, 1.0e5, 1.0e-13, 1.0e-12, 3, 3, 1);
    let integ = build_mtk(&gpu, 4, &kind);
    assert_eq!(
        integ.log_column_names(),
        &["pressure", "box_volume", "mtk_npt_conserved"]
    );
}

#[test]
fn log_column_values_includes_chain_terms_in_conserved_hamiltonian() {
    let gpu = init_device().unwrap();
    let mut s = unbox_mtk(build_mtk(
        &gpu,
        4,
        &mtk_kind(85.0, 1.0e5, 1.0e-13, 1.0e-12, 2, 3, 1),
    ));
    s.most_recent_pressure = 1.01e5;
    s.most_recent_volume = 1.0e-27;
    // Hand-set chain DOFs.
    s.xi_part[0] = 0.1;
    s.xi_part[1] = 0.2;
    s.p_xi_part[0] = 0.5e-30;
    s.p_xi_part[1] = -0.3e-30;
    s.xi_cell[0] = 0.05;
    s.xi_cell[1] = 0.15;
    s.p_xi_cell[0] = 0.0;
    s.p_xi_cell[1] = 0.0;
    s.p_eps = 2.5e-25;
    let ke = 1.0e-20_f64;
    let pe = 2.0e-20_f64;
    let extras = s.log_column_values(ke, pe);
    assert_eq!(extras.len(), 3);
    assert_eq!(extras[0], 1.01e5);
    assert_eq!(extras[1], 1.0e-27);
    let g_dof = s.g_dof as f64;
    let q_p = &s.q_mass_part;
    let q_c = &s.q_mass_cell;
    let expected_h = ke + pe
        + s.pressure * s.most_recent_volume
        + 0.5 * s.p_eps * s.p_eps / s.w_cell
        + s.p_xi_part[0].powi(2) / (2.0 * q_p[0])
        + s.p_xi_part[1].powi(2) / (2.0 * q_p[1])
        + s.p_xi_cell[0].powi(2) / (2.0 * q_c[0])
        + s.p_xi_cell[1].powi(2) / (2.0 * q_c[1])
        + g_dof * s.kt * s.xi_part[0]
        + s.kt * s.xi_part[1]
        + s.kt * s.xi_cell[0]
        + s.kt * s.xi_cell[1];
    assert!((extras[2] - expected_h).abs() / expected_h.abs() < 1.0e-12);
}

// --- Determinism ---

#[test]
fn two_runs_with_identical_configs_are_byte_identical() {
    let gpu = init_device().unwrap();
    let n = 4usize;

    fn run_once(gpu: &GpuContext, n: usize) -> (Vec<f32>, [f32; 6]) {
        let state = symmetric_state(n, 1.66e-27, 500.0);
        let mut buffers = ParticleBuffers::new(gpu, &state).unwrap();
        let mut sim_box = box_small();
        let mut ff = empty_force_field(gpu, n);
        let mut timings = Timings::new(gpu).unwrap();
        let mut integ = build_mtk(gpu, n, &mtk_kind(85.0, 1.0e5, 1.0e-13, 1.0e-12, 3, 3, 1));
        ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
        for _ in 0..5 {
            integ
                .step(&mut buffers, &mut sim_box, &mut ff, None, 1.0e-15, &mut timings)
                .unwrap();
        }
        let px = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
        (px, sim_box.lattice())
    }

    let (px_a, lat_a) = run_once(&gpu, n);
    let (px_b, lat_b) = run_once(&gpu, n);
    assert_eq!(px_a, px_b);
    for i in 0..6 {
        assert_eq!(lat_a[i].to_bits(), lat_b[i].to_bits());
    }
}

// --- Smoke test of physical correctness ---

// --- Approximate time-reversibility smoke ---

// Runs 50 MTK steps forward on an empty-force-field system, then flips
// every momentum (particle velocities, p_eps, both chains' p_xi), then
// runs 50 more steps. With a symmetric Trotter splitting (which MTTK
// 1996 is) the second pass retraces the first, modulo f32 storage
// round-off in the cell-coupled drift / velocity-rescale kernels.
//
// MTK has NO lossless mode (the rqm doc declares this out of scope: the
// cell-coupled exp() and the chain sinh/exp rescales don't have a clean
// compensated f32+f64 form), so this is a tolerance check, not a
// bit-exact one. Empirically ~100 lossy kernel launches accumulate
// ~100·ULP ≈ 1e-5 relative drift; the assertions below use 1e-3
// relative to leave headroom.
#[test]
fn mtk_approximate_time_reversibility_smoke() {
    let gpu = init_device().unwrap();
    let n = 8usize;
    let mass: f32 = 1.66e-27;
    // Start at the chain's target kinetic energy so the chain isn't
    // strongly injecting/extracting energy (which would amplify f32
    // drift). K_target = (g_dof/2) · k_B · T with g_dof = 3N − 3.
    let temperature = 85.0_f64;
    let g_dof = (3 * n - 3) as f64;
    let k_target = (g_dof / 2.0) * KB * temperature;
    let v_each = (k_target / ((n as f64) * 0.5 * (mass as f64))).sqrt() as f32;
    let state = symmetric_state(n, mass, v_each);

    // Snapshot the initial state (positions, velocities, lattice).
    let snap_px = state.positions_x.clone();
    let snap_vx = state.velocities_x.clone();
    let snap_lattice = box_small().lattice();

    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();

    // Use a sensible dt and moderately stiff couplings.
    let dt = 1.0e-15_f32;
    let mut integ = unbox_mtk(build_mtk(
        &gpu,
        n,
        &mtk_kind(temperature, 1.0e5, 1.0e-13, 1.0e-12, 3, 3, 1),
    ));

    // Warm up the force pipeline so virials are populated.
    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();

    // --- Forward 50 steps ---
    for _ in 0..50 {
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, None, dt, &mut timings)
            .unwrap();
    }

    // --- Flip every momentum ---
    // 1. Particle velocities (on the device).
    for axis_slice in [
        &mut buffers.velocities_x,
        &mut buffers.velocities_y,
        &mut buffers.velocities_z,
    ] {
        let v = gpu.device.dtoh_sync_copy(axis_slice).unwrap();
        let v_flipped: Vec<f32> = v.iter().map(|&x| -x).collect();
        gpu.device
            .htod_sync_copy_into(&v_flipped, axis_slice)
            .unwrap();
    }
    // 2. Cell momentum and both chains' momenta (on the host).
    integ.p_eps = -integ.p_eps;
    for p in integ.p_xi_part.iter_mut() {
        *p = -*p;
    }
    for p in integ.p_xi_cell.iter_mut() {
        *p = -*p;
    }

    // Re-warm the force pipeline at the post-flip state. Forces are
    // unchanged by velocity flip (F depends on positions), but the
    // virial buffer would be stale only if we mutated positions —
    // which we did not — so this is conceptually a no-op for the
    // empty force field. We run it anyway to mirror the runner's
    // per-step contract.
    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();

    // --- Reverse 50 steps ---
    for _ in 0..50 {
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, None, dt, &mut timings)
            .unwrap();
    }

    // --- Compare to the initial state ---
    let px_final = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let vx_final = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();

    // Positions should return to within accumulated f32 round-off.
    // Use absolute tolerance scaled by the largest |x| in the system,
    // since the symmetric_state positions span 0 → 4e-10 m and
    // include a zero crossing where relative tolerance breaks down.
    let max_abs_x: f32 = snap_px.iter().map(|x| x.abs()).fold(0.0, f32::max);
    let pos_tol = max_abs_x * 1.0e-3;
    for (i, (a, b)) in px_final.iter().zip(snap_px.iter()).enumerate() {
        assert!(
            (a - b).abs() < pos_tol,
            "particle {i}: x drifted {} vs snap {} (tol {pos_tol:e})",
            a,
            b
        );
    }

    // Velocities should return to NEGATIVE of the initial values
    // (because we flipped them between the two passes, so the reverse
    // pass leaves them at -v_initial).
    let max_abs_v: f32 = snap_vx.iter().map(|v| v.abs()).fold(0.0, f32::max);
    let vel_tol = max_abs_v * 1.0e-3;
    for (i, (a, b)) in vx_final.iter().zip(snap_vx.iter()).enumerate() {
        let expected = -*b;
        assert!(
            (a - expected).abs() < vel_tol,
            "particle {i}: v drifted {} vs expected {} (tol {vel_tol:e})",
            a,
            expected
        );
    }

    // Lattice should return to within accumulated f32 round-off.
    let lat_final = sim_box.lattice();
    for i in 0..6 {
        let snap = snap_lattice[i];
        let tol = snap.abs().max(1.0e-10) * 1.0e-3;
        assert!(
            (lat_final[i] - snap).abs() < tol,
            "lattice[{i}]: {} vs snap {} (tol {tol:e})",
            lat_final[i],
            snap
        );
    }

    // Cell momentum should also return to near zero (it started at 0).
    assert!(
        integ.p_eps.abs() < 1.0e-30,
        "p_eps drifted to {}",
        integ.p_eps
    );
}

#[test]
fn finite_step_keeps_velocities_and_positions_finite() {
    // No physical-correctness assertion (too short a run; no thermalisation),
    // just verify the integrator produces finite numbers and the box volume
    // stays positive after a handful of steps.
    let gpu = init_device().unwrap();
    let n = 16usize;
    let state = symmetric_state(n, 1.66e-27, 500.0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let mut sim_box = box_small();
    let mut ff = empty_force_field(&gpu, n);
    let mut timings = Timings::new(&gpu).unwrap();
    let mut integ = build_mtk(&gpu, n, &mtk_kind(85.0, 1.0e5, 1.0e-13, 1.0e-12, 3, 3, 1));
    ff.step(&mut buffers, &sim_box, &mut timings).unwrap();
    for _ in 0..50 {
        integ
            .step(&mut buffers, &mut sim_box, &mut ff, None, 1.0e-15, &mut timings)
            .unwrap();
    }
    let v_final = sim_box.volume();
    assert!(v_final.is_finite() && v_final > 0.0);
    let vx = gpu.device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    for v in &vx {
        assert!(v.is_finite(), "non-finite velocity {v}");
    }
    let px = gpu.device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    for p in &px {
        assert!(p.is_finite(), "non-finite position {p}");
    }
}
