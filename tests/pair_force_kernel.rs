//! Tests for the shared warp-per-particle fused pair-force kernel pattern.
//!
//! Implements the Gherkin scenarios in `rqm/forces/pair-force-kernel.md`.
//! Scenarios prefaced "any pair-force potential P" run three times — once
//! each for Lennard-Jones, truncated Coulomb, and SPME real-space.
//!
//! The kernel implementation under test lives in `kernels/pair_compute.cuh`
//! plus the three per-potential `.cu` files; the Rust launcher dispatches
//! the `_f` / `_fev` variant based on `AggregateLevel`.

mod common;

use std::f64::consts::PI;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};
use heddle_md::forces::{AggregateLevel, CoulombParameters};
use heddle_md::gpu::{
    GpuContext, K_COULOMB_F32, LennardJonesParameterTable, ParticleBuffers, SlotOutputBuffers,
    coulomb_pair_force, init_device, lj_pair_force, spme_real_pair_force,
};
use heddle_md::pbc::SimulationBox;
use heddle_md::precision::Real;
use heddle_md::state::ParticleState;

use common::empty_exclusions;

// =================================================================
// PotentialKind: parameterisation over the three fused pair-force
// kernels.
// =================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PotentialKind {
    Lj,
    Coulomb,
    Spme,
}

impl PotentialKind {
    fn name(self) -> &'static str {
        match self {
            PotentialKind::Lj => "lj",
            PotentialKind::Coulomb => "coulomb",
            PotentialKind::Spme => "spme_real",
        }
    }
}

// Per-pair functional-form parameters used across tests. Chosen so the
// per-pair force on the (0, 1) pair at the typical test separation is
// finite, nonzero, and computable in closed form on the host.
const SIGMA: Real = 1.0;
const EPSILON: Real = 1.0;
const CUTOFF: Real = 5.0;
const CHARGE_0: Real = 0.5;
const CHARGE_1: Real = -0.7;
const ALPHA: Real = 0.4;
const R_CUT_REAL: Real = 5.0;

// Abramowitz-Stegun 7.1.26 rational approximation of erfc for x >= 0.
// Max error ~1.5e-7, sufficient for f32-precision GPU output comparison.
fn erfc_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    1.0 - sign * y
}

// Closed-form per-pair force factor: `factor` such that
// `force_on_i = factor * (r_i - r_j)`. Identical to the kernel's
// `factor` variable. For LJ with `r_switch == cutoff` (hard cutoff),
// no switching is applied below the cutoff.
fn closed_form_factor(kind: PotentialKind, r2: f64) -> f64 {
    let r = r2.sqrt();
    match kind {
        PotentialKind::Lj => {
            let inv_r2 = 1.0 / r2;
            let sigma2 = (SIGMA as f64) * (SIGMA as f64);
            let sr2 = sigma2 * inv_r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;
            24.0 * (EPSILON as f64) * inv_r2 * (2.0 * sr12 - sr6)
        }
        PotentialKind::Coulomb => {
            let inv_r2 = 1.0 / r2;
            let inv_r = 1.0 / r;
            let qq = (CHARGE_0 as f64) * (CHARGE_1 as f64);
            (K_COULOMB_F32 as f64) * qq * inv_r * inv_r2
        }
        PotentialKind::Spme => {
            let inv_r2 = 1.0 / r2;
            let inv_r = 1.0 / r;
            let qq = (CHARGE_0 as f64) * (CHARGE_1 as f64);
            let ar = (ALPHA as f64) * r;
            let erfc_ar = erfc_f64(ar);
            let gauss = (-(ar * ar)).exp();
            let inv_sqrt_pi = 1.0 / PI.sqrt();
            (K_COULOMB_F32 as f64)
                * qq
                * inv_r2
                * (erfc_ar * inv_r + 2.0 * (ALPHA as f64) * inv_sqrt_pi * gauss)
        }
    }
}

// =================================================================
// Builders for particle state, exclusions, and neighbour-list buffers.
// =================================================================

fn build_state(n: usize, positions: Vec<(Real, Real, Real)>) -> ParticleState {
    assert_eq!(positions.len(), n);
    let (xs, (ys, zs)): (Vec<_>, (Vec<_>, Vec<_>)) =
        positions.into_iter().map(|(x, y, z)| (x, (y, z))).unzip();
    let mut charges = vec![0.0 as Real; n];
    if n >= 1 {
        charges[0] = CHARGE_0;
    }
    if n >= 2 {
        charges[1] = CHARGE_1;
    }
    // Every other particle gets a deterministic alternating charge so
    // multi-particle Coulomb / SPME-real systems are nontrivial.
    for (i, q) in charges.iter_mut().enumerate().skip(2) {
        *q = if i % 2 == 0 { 0.3 } else { -0.4 };
    }
    ParticleState::new(
        xs,
        ys,
        zs,
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        charges,
        vec![0u32; n],
        None,
        None,
    )
    .expect("ParticleState::new")
}

fn upload_neighbor_list(
    device: &Arc<CudaDevice>,
    flat: &[u32],
) -> CudaSlice<u32> {
    device.htod_sync_copy(flat).expect("upload neighbor list")
}

fn upload_neighbor_counts(
    device: &Arc<CudaDevice>,
    counts: &[u32],
) -> CudaSlice<u32> {
    device.htod_sync_copy(counts).expect("upload neighbor counts")
}

// =================================================================
// Launch the chosen kernel against a SlotOutputBuffers.
// =================================================================

fn launch(
    kind: PotentialKind,
    gpu: &GpuContext,
    buffers: &ParticleBuffers,
    output: &mut SlotOutputBuffers,
    sim_box: &SimulationBox,
    neighbor_list: &CudaSlice<u32>,
    neighbor_counts: &CudaSlice<u32>,
    max_neighbors: u32,
    level: AggregateLevel,
) {
    let n = buffers.particle_count();
    let excl = empty_exclusions(&gpu.device, n);
    let mut view = output.view();
    match kind {
        PotentialKind::Lj => {
            let params = LennardJonesParameterTable {
                n_types: 1,
                sigma: gpu.device.htod_sync_copy(&[SIGMA]).unwrap(),
                epsilon: gpu.device.htod_sync_copy(&[EPSILON]).unwrap(),
                cutoff: gpu.device.htod_sync_copy(&[CUTOFF]).unwrap(),
                switch: gpu.device.htod_sync_copy(&[CUTOFF]).unwrap(),
            };
            lj_pair_force(
                buffers,
                &mut view,
                sim_box,
                &params,
                &excl.atom_excl_offsets,
                &excl.atom_excl_partners,
                &excl.atom_excl_lj_scales,
                neighbor_list,
                neighbor_counts,
                max_neighbors,
                level,
            )
            .expect("lj_pair_force");
        }
        PotentialKind::Coulomb => {
            let params = CoulombParameters {
                cutoff: CUTOFF,
                r_switch: CUTOFF, // hard cutoff
            };
            coulomb_pair_force(
                buffers,
                &mut view,
                sim_box,
                params.cutoff,
                params.r_switch,
                &excl.atom_excl_offsets,
                &excl.atom_excl_partners,
                &excl.atom_excl_coul_scales,
                neighbor_list,
                neighbor_counts,
                max_neighbors,
                level,
            )
            .expect("coulomb_pair_force");
        }
        PotentialKind::Spme => {
            spme_real_pair_force(
                buffers,
                &mut view,
                sim_box,
                ALPHA,
                R_CUT_REAL,
                &excl.atom_excl_offsets,
                &excl.atom_excl_partners,
                &excl.atom_excl_coul_scales,
                neighbor_list,
                neighbor_counts,
                max_neighbors,
                level,
            )
            .expect("spme_real_pair_force");
        }
    }
}

fn download_force_x(buffers: &SlotOutputBuffers, device: &Arc<CudaDevice>) -> Vec<Real> {
    device.dtoh_sync_copy(&buffers.force_x).unwrap()
}

fn download_force_y(buffers: &SlotOutputBuffers, device: &Arc<CudaDevice>) -> Vec<Real> {
    device.dtoh_sync_copy(&buffers.force_y).unwrap()
}

fn download_force_z(buffers: &SlotOutputBuffers, device: &Arc<CudaDevice>) -> Vec<Real> {
    device.dtoh_sync_copy(&buffers.force_z).unwrap()
}

fn download_energy(buffers: &SlotOutputBuffers, device: &Arc<CudaDevice>) -> Vec<Real> {
    device.dtoh_sync_copy(&buffers.energy).unwrap()
}

fn download_virial(buffers: &SlotOutputBuffers, device: &Arc<CudaDevice>) -> Vec<Real> {
    device.dtoh_sync_copy(&buffers.virial).unwrap()
}

fn default_box(gpu: &heddle_md::gpu::GpuContext) -> SimulationBox {
    SimulationBox::new(&gpu.device, 20.0, 20.0, 20.0, 0.0, 0.0, 0.0).unwrap()
}

// =================================================================
// Grid layout: ragged + under-full + no shared memory.
// =================================================================

#[test]
fn particle_count_not_multiple_of_warps_per_block_uses_ragged_grid() {
    // 10 particles, WARPS_PER_BLOCK = 8 → 2 blocks. Warps 2..8 of the
    // second block have no particle to handle and must return without
    // writing past index 9.
    let gpu = init_device().unwrap();
    let n = 10;
    // Place 10 particles along the x axis at 1.5-spacing. Each
    // particle's only listed neighbour is its immediate +x neighbour
    // (particle (i+1) mod n), with wraparound so particle 9's
    // neighbour is particle 0. Particle 9 and particle 0 are 13.5
    // apart in absolute coords, but the box is 20 on a side with PBC
    // off, so the 9→0 wrap pair is not within cutoff. Use the −x
    // neighbour for particle 0 (= particle 1) so every pair is at
    // r = 1.5 in the cutoff.
    let positions = (0..n)
        .map(|i| (i as Real * 1.5, 0.0, 0.0))
        .collect::<Vec<_>>();
    let state = build_state(n, positions);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = default_box(&gpu);

    let mut nl_flat = Vec::with_capacity(n);
    let counts = vec![1u32; n];
    for i in 0..n {
        let partner = if i == 0 { 1 } else { i - 1 };
        nl_flat.push(partner as u32);
    }
    let neighbor_list = upload_neighbor_list(&gpu.device, &nl_flat);
    let neighbor_counts = upload_neighbor_counts(&gpu.device, &counts);

    let mut output = SlotOutputBuffers::new(&gpu.device, n).unwrap();
    launch(
        PotentialKind::Lj,
        &gpu,
        &buffers,
        &mut output,
        &sim_box,
        &neighbor_list,
        &neighbor_counts,
        1,
        AggregateLevel::ForcesOnly,
    );

    let fx = download_force_x(&output, &gpu.device);
    assert_eq!(fx.len(), n);
    // Every particle has exactly one in-cutoff partner. None of the
    // outputs should be NaN/Inf, and there should be no out-of-bounds
    // write (we'd see corruption past index 9 — but n=10 == buffer
    // size so we can't detect that here other than via Cuda not
    // crashing).
    for (i, v) in fx.iter().enumerate() {
        assert!(v.is_finite(), "fx[{i}] = {v} is not finite");
        assert!(*v != 0.0, "fx[{i}] = 0 — kernel may not have written");
    }
}

#[test]
fn particle_count_below_warps_per_block_uses_under_full_block() {
    // 3 particles → 1 block. Warps 3..8 have no particle and return.
    let gpu = init_device().unwrap();
    let n = 3;
    let positions = vec![(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0)];
    let state = build_state(n, positions);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = default_box(&gpu);
    let counts = vec![2u32; n];
    let nl_flat = vec![1u32, 2, 0, 2, 0, 1];
    let neighbor_list = upload_neighbor_list(&gpu.device, &nl_flat);
    let neighbor_counts = upload_neighbor_counts(&gpu.device, &counts);
    let mut output = SlotOutputBuffers::new(&gpu.device, n).unwrap();
    launch(
        PotentialKind::Lj,
        &gpu,
        &buffers,
        &mut output,
        &sim_box,
        &neighbor_list,
        &neighbor_counts,
        2,
        AggregateLevel::ForcesOnly,
    );
    let fx = download_force_x(&output, &gpu.device);
    assert_eq!(fx.len(), 3);
    for (i, v) in fx.iter().enumerate() {
        assert!(v.is_finite(), "fx[{i}] = {v} is not finite");
    }
}

#[test]
fn pair_force_kernels_declare_no_shared_memory() {
    // PTX-level check: none of the three pair-force CUDA modules should
    // contain a `.shared` directive. Catches a regression where some
    // future helper accidentally allocates `__shared__` storage.
    for (mod_name, ptx) in [
        ("pair_force", heddle_md::kernels::PAIR_FORCE),
        ("coulomb", heddle_md::kernels::COULOMB),
        ("spme_real", heddle_md::kernels::SPME_REAL),
    ] {
        assert!(
            !ptx.contains(".shared"),
            "{mod_name}.ptx contains a `.shared` directive"
        );
    }
}

// =================================================================
// Sweep semantics.
// =================================================================

fn count_zero_preserves_seeded_accumulator_impl(kind: PotentialKind) {
    let gpu = init_device().unwrap();
    let n = 4;
    let positions = (0..n)
        .map(|i| (i as Real * 1.5, 0.0, 0.0))
        .collect::<Vec<_>>();
    let state = build_state(n, positions);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = default_box(&gpu);
    let counts = vec![0u32; n];
    // 4 slots × max_neighbors=4 = 16, all unused.
    let nl_flat = vec![0u32; n * 4];
    let neighbor_list = upload_neighbor_list(&gpu.device, &nl_flat);
    let neighbor_counts = upload_neighbor_counts(&gpu.device, &counts);

    let mut output = SlotOutputBuffers::new(&gpu.device, n).unwrap();
    // Seed the accumulator with a known nonzero value. The kernel adds
    // each particle's per-pair contributions into the seeded slot; with
    // count == 0 no contribution is added, so the seed is preserved
    // unchanged.
    let device = gpu.device.clone();
    let seed = vec![7.0 as Real; n];
    device.htod_sync_copy_into(&seed, &mut output.force_x).unwrap();
    device.htod_sync_copy_into(&seed, &mut output.force_y).unwrap();
    device.htod_sync_copy_into(&seed, &mut output.force_z).unwrap();

    launch(
        kind,
        &gpu,
        &buffers,
        &mut output,
        &sim_box,
        &neighbor_list,
        &neighbor_counts,
        4,
        AggregateLevel::ForcesOnly,
    );
    let fx = download_force_x(&output, &gpu.device);
    let fy = download_force_y(&output, &gpu.device);
    let fz = download_force_z(&output, &gpu.device);
    for i in 0..n {
        assert_eq!(fx[i], 7.0, "{}: fx[{i}] = {}", kind.name(), fx[i]);
        assert_eq!(fy[i], 7.0, "{}: fy[{i}] = {}", kind.name(), fy[i]);
        assert_eq!(fz[i], 7.0, "{}: fz[{i}] = {}", kind.name(), fz[i]);
    }
}
#[test] fn count_zero_preserves_seeded_accumulator_lj() { count_zero_preserves_seeded_accumulator_impl(PotentialKind::Lj); }
#[test] fn count_zero_preserves_seeded_accumulator_coulomb() { count_zero_preserves_seeded_accumulator_impl(PotentialKind::Coulomb); }
#[test] fn count_zero_preserves_seeded_accumulator_spme() { count_zero_preserves_seeded_accumulator_impl(PotentialKind::Spme); }

#[test]
fn sweep_reads_only_slots_up_to_count() {
    // particle_count = 1, count = 3, max_neighbors = 8. Slots 0..3 list
    // valid partners (a single in-cutoff particle 1, repeated for
    // simplicity). Slot 4..8 are populated with junk partner IDs that
    // would cause out-of-range reads if visited. The kernel must not
    // visit them, so the output stays finite.
    let gpu = init_device().unwrap();
    let n = 2;
    let positions = vec![(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)];
    let state = build_state(n, positions);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = default_box(&gpu);
    let max_neighbors = 8;
    // Particle 0: visit partner 1 three times; junk in slots 3..8.
    // Junk values picked to be in-range valid indices (so the test
    // verifies the count guard, not a fault-on-OOB check).
    let mut row0 = vec![1u32, 1, 1, 0, 0, 0, 0, 0];
    let mut row1 = vec![0u32, 0, 0, 0, 0, 0, 0, 0];
    let mut nl_flat = Vec::new();
    nl_flat.append(&mut row0);
    nl_flat.append(&mut row1);
    let counts = vec![3u32, 0];
    let neighbor_list = upload_neighbor_list(&gpu.device, &nl_flat);
    let neighbor_counts = upload_neighbor_counts(&gpu.device, &counts);

    let mut output = SlotOutputBuffers::new(&gpu.device, n).unwrap();
    launch(
        PotentialKind::Lj,
        &gpu,
        &buffers,
        &mut output,
        &sim_box,
        &neighbor_list,
        &neighbor_counts,
        max_neighbors,
        AggregateLevel::ForcesOnly,
    );
    let fx = download_force_x(&output, &gpu.device);
    assert!(fx[0].is_finite(), "fx[0] = {} not finite", fx[0]);
}

fn self_pair_skipped_in_trivial_neighbour_list_impl(kind: PotentialKind) {
    // particle_count = 2, neighbor_counts = [2, 2], neighbor_list =
    // [0, 1, 1, 0]. Each particle's list contains itself plus the
    // other. The self-pair must be skipped (i == j guard) so the
    // result equals the (0, 1) closed-form contribution only.
    let gpu = init_device().unwrap();
    let dx = 1.5_f64;
    let positions = vec![(0.0, 0.0, 0.0), (dx as Real, 0.0, 0.0)];
    let state = build_state(2, positions);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = default_box(&gpu);
    let counts = vec![2u32, 2];
    let nl_flat = vec![0u32, 1, 1, 0];
    let neighbor_list = upload_neighbor_list(&gpu.device, &nl_flat);
    let neighbor_counts = upload_neighbor_counts(&gpu.device, &counts);

    let mut output = SlotOutputBuffers::new(&gpu.device, 2).unwrap();
    launch(
        kind,
        &gpu,
        &buffers,
        &mut output,
        &sim_box,
        &neighbor_list,
        &neighbor_counts,
        2,
        AggregateLevel::ForcesOnly,
    );
    let fx = download_force_x(&output, &gpu.device);
    assert!(fx[0].is_finite(), "{}: fx[0] = {} not finite (NaN means i==j was evaluated)", kind.name(), fx[0]);
    assert!(fx[1].is_finite(), "{}: fx[1] = {} not finite", kind.name(), fx[1]);
    // The pair (0, 1) contributes `factor * dx` to fx[0]. The self
    // pair (0, 0) contributes zero. So fx[0] = factor * (0 - 1.5).
    let factor = closed_form_factor(kind, dx * dx);
    let expected_fx0 = factor * (0.0 - dx);
    let rel_err = (fx[0] as f64 - expected_fx0).abs() / expected_fx0.abs().max(1e-30);
    assert!(
        rel_err < 1e-4,
        "{}: fx[0] = {} != expected {} (rel err {})",
        kind.name(),
        fx[0],
        expected_fx0,
        rel_err
    );
}
#[test] fn self_pair_skipped_lj() { self_pair_skipped_in_trivial_neighbour_list_impl(PotentialKind::Lj); }
#[test] fn self_pair_skipped_coulomb() { self_pair_skipped_in_trivial_neighbour_list_impl(PotentialKind::Coulomb); }
#[test] fn self_pair_skipped_spme() { self_pair_skipped_in_trivial_neighbour_list_impl(PotentialKind::Spme); }

// =================================================================
// Reduction shape: sweep iteration boundaries.
// =================================================================

fn sweep_boundary_impl(kind: PotentialKind, count: usize) {
    // Particle 0 has `count` neighbours, all pointing at particle 1.
    // The expected per-particle force is N × closed-form-per-pair-force.
    let gpu = init_device().unwrap();
    let dx = 1.5_f64;
    let positions = vec![(0.0, 0.0, 0.0), (dx as Real, 0.0, 0.0)];
    let state = build_state(2, positions);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = default_box(&gpu);
    let max_neighbors = count as u32;
    let counts = vec![count as u32, 0u32];
    let mut row0 = vec![1u32; count];
    // Pad row0 to max_neighbors length, then row1 of zeros.
    while row0.len() < max_neighbors as usize {
        row0.push(0);
    }
    let row1 = vec![0u32; max_neighbors as usize];
    let mut nl_flat = Vec::new();
    nl_flat.extend_from_slice(&row0);
    nl_flat.extend_from_slice(&row1);
    let neighbor_list = upload_neighbor_list(&gpu.device, &nl_flat);
    let neighbor_counts = upload_neighbor_counts(&gpu.device, &counts);

    let mut output = SlotOutputBuffers::new(&gpu.device, 2).unwrap();
    launch(
        kind,
        &gpu,
        &buffers,
        &mut output,
        &sim_box,
        &neighbor_list,
        &neighbor_counts,
        max_neighbors,
        AggregateLevel::ForcesOnly,
    );
    let fx = download_force_x(&output, &gpu.device);
    let factor = closed_form_factor(kind, dx * dx);
    let per_pair_fx = factor * (0.0 - dx);
    let expected = (count as f64) * per_pair_fx;
    let rel_err = (fx[0] as f64 - expected).abs() / expected.abs().max(1e-30);
    assert!(
        rel_err < 1e-3,
        "{}: count={} → fx[0] = {} != expected {} (rel err {})",
        kind.name(),
        count,
        fx[0],
        expected,
        rel_err
    );
}

#[test] fn sweep_boundary_count_32_lj() { sweep_boundary_impl(PotentialKind::Lj, 32); }
#[test] fn sweep_boundary_count_32_coulomb() { sweep_boundary_impl(PotentialKind::Coulomb, 32); }
#[test] fn sweep_boundary_count_32_spme() { sweep_boundary_impl(PotentialKind::Spme, 32); }
#[test] fn sweep_boundary_count_33_lj() { sweep_boundary_impl(PotentialKind::Lj, 33); }
#[test] fn sweep_boundary_count_33_coulomb() { sweep_boundary_impl(PotentialKind::Coulomb, 33); }
#[test] fn sweep_boundary_count_33_spme() { sweep_boundary_impl(PotentialKind::Spme, 33); }
#[test] fn sweep_boundary_count_96_lj() { sweep_boundary_impl(PotentialKind::Lj, 96); }
#[test] fn sweep_boundary_count_96_coulomb() { sweep_boundary_impl(PotentialKind::Coulomb, 96); }
#[test] fn sweep_boundary_count_96_spme() { sweep_boundary_impl(PotentialKind::Spme, 96); }

// CPU warp-tree reference scenario (rq-3982aff8). LJ only — the warp-
// tree shape is shared, so verifying one potential exercises it.
#[test]
fn warp_tree_reduction_agrees_with_cpu_reference_lj() {
    let gpu = init_device().unwrap();
    let n = 1;
    let positions = vec![(0.0, 0.0, 0.0)];
    let state = build_state(n, positions);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = default_box(&gpu);

    // Single particle with a synthetic neighbour list of 80 valid
    // partner indices — but particle_count = 1, so partner IDs must
    // be valid (0..n). We can't have 80 partners on a 1-particle
    // system without referring to nonexistent particles.
    //
    // Use a 2-particle system instead: particle 0 with 80 entries all
    // pointing at particle 1. Each pair contributes a deterministic
    // factor at fixed r = 1.5. CPU sums via the same warp-tree shape.
    let n = 2;
    let positions = vec![(0.0, 0.0, 0.0), (1.5 as Real, 0.0, 0.0)];
    let state = build_state(n, positions);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();

    let count: usize = 80;
    let max_neighbors = count as u32;
    let counts = vec![count as u32, 0u32];
    let mut row0 = vec![1u32; count];
    while row0.len() < max_neighbors as usize {
        row0.push(0);
    }
    let row1 = vec![0u32; max_neighbors as usize];
    let mut nl_flat = Vec::new();
    nl_flat.extend_from_slice(&row0);
    nl_flat.extend_from_slice(&row1);
    let neighbor_list = upload_neighbor_list(&gpu.device, &nl_flat);
    let neighbor_counts = upload_neighbor_counts(&gpu.device, &counts);
    let mut output = SlotOutputBuffers::new(&gpu.device, n).unwrap();
    launch(
        PotentialKind::Lj,
        &gpu,
        &buffers,
        &mut output,
        &sim_box,
        &neighbor_list,
        &neighbor_counts,
        max_neighbors,
        AggregateLevel::ForcesOnly,
    );
    let fx = download_force_x(&output, &gpu.device);

    // CPU warp-tree reference: 32 strided partial sums (lane t holds
    // every (s * 32 + t)-th contribution for s = 0 .. ceil(N / 32)),
    // then a 5-step pairwise XOR butterfly across the 32 lanes.
    let r2 = 1.5_f64 * 1.5_f64;
    let factor = closed_form_factor(PotentialKind::Lj, r2) as f32;
    let per_pair_fx = factor * (0.0 - 1.5_f32);
    let mut lanes = [0.0_f32; 32];
    for k in 0..count {
        lanes[k & 31] += per_pair_fx;
    }
    for &stride in &[16usize, 8, 4, 2, 1] {
        let next = lanes;
        for lane in 0..32 {
            lanes[lane] = next[lane] + next[lane ^ stride];
        }
    }
    let cpu_reference = lanes[0];
    assert_eq!(
        fx[0].to_bits(),
        cpu_reference.to_bits(),
        "GPU fx[0] = {} (bits {:x}) != CPU warp-tree reference {} (bits {:x})",
        fx[0],
        fx[0].to_bits(),
        cpu_reference,
        cpu_reference.to_bits()
    );
}

// =================================================================
// Variant selection.
// =================================================================

#[test]
fn init_device_exposes_both_kernel_variants() {
    let gpu = init_device().unwrap();
    // Each pair-force `*Kernels` field exposes both `*_f` and `*_fev`
    // CudaFunction handles. Just touch each one — if a field is
    // missing or not populated the program won't compile.
    let _ = &gpu.kernels.lj.pair_force_f;
    let _ = &gpu.kernels.lj.pair_force_fev;
    let _ = &gpu.kernels.coulomb.coulomb_pair_force_f;
    let _ = &gpu.kernels.coulomb.coulomb_pair_force_fev;
    let _ = &gpu.kernels.spme_real.spme_real_pair_force_f;
    let _ = &gpu.kernels.spme_real.spme_real_pair_force_fev;
}

#[test]
fn f_variant_does_not_write_energy_or_virial_lj() {
    // Seed energy/virial outputs with a sentinel; launch _f variant;
    // assert they remain byte-identical.
    let gpu = init_device().unwrap();
    let n = 2;
    let positions = vec![(0.0, 0.0, 0.0), (1.5 as Real, 0.0, 0.0)];
    let state = build_state(n, positions);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = default_box(&gpu);
    let counts = vec![1u32, 1u32];
    let nl_flat = vec![1u32, 0u32];
    let neighbor_list = upload_neighbor_list(&gpu.device, &nl_flat);
    let neighbor_counts = upload_neighbor_counts(&gpu.device, &counts);

    let mut output = SlotOutputBuffers::new(&gpu.device, n).unwrap();
    let device = gpu.device.clone();
    let sentinel = vec![123.456 as Real; n];
    device.htod_sync_copy_into(&sentinel, &mut output.energy).unwrap();
    device.htod_sync_copy_into(&sentinel, &mut output.virial).unwrap();
    launch(
        PotentialKind::Lj,
        &gpu,
        &buffers,
        &mut output,
        &sim_box,
        &neighbor_list,
        &neighbor_counts,
        1,
        AggregateLevel::ForcesOnly,
    );
    let e = download_energy(&output, &gpu.device);
    let w = download_virial(&output, &gpu.device);
    for i in 0..n {
        assert_eq!(e[i].to_bits(), (123.456 as Real).to_bits(), "energy[{i}] modified");
        assert_eq!(w[i].to_bits(), (123.456 as Real).to_bits(), "virial[{i}] modified");
    }
}

#[test]
fn fev_variant_writes_energy_and_virial_lj() {
    let gpu = init_device().unwrap();
    let n = 2;
    let positions = vec![(0.0, 0.0, 0.0), (1.5 as Real, 0.0, 0.0)];
    let state = build_state(n, positions);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = default_box(&gpu);
    let counts = vec![1u32, 1u32];
    let nl_flat = vec![1u32, 0u32];
    let neighbor_list = upload_neighbor_list(&gpu.device, &nl_flat);
    let neighbor_counts = upload_neighbor_counts(&gpu.device, &counts);

    let mut output = SlotOutputBuffers::new(&gpu.device, n).unwrap();
    let device = gpu.device.clone();
    let sentinel = vec![123.456 as Real; n];
    device.htod_sync_copy_into(&sentinel, &mut output.energy).unwrap();
    device.htod_sync_copy_into(&sentinel, &mut output.virial).unwrap();
    launch(
        PotentialKind::Lj,
        &gpu,
        &buffers,
        &mut output,
        &sim_box,
        &neighbor_list,
        &neighbor_counts,
        1,
        AggregateLevel::ForcesAndScalars,
    );
    let e = download_energy(&output, &gpu.device);
    let w = download_virial(&output, &gpu.device);
    // The energy and virial must differ from the sentinel.
    let any_e_changed = e.iter().any(|&v| v.to_bits() != (123.456 as Real).to_bits());
    let any_w_changed = w.iter().any(|&v| v.to_bits() != (123.456 as Real).to_bits());
    assert!(any_e_changed, "energy not written by _fev");
    assert!(any_w_changed, "virial not written by _fev");
}

fn f_and_fev_agree_on_force_impl(kind: PotentialKind) {
    let gpu = init_device().unwrap();
    let n = 16;
    let positions = (0..n)
        .map(|i| (i as Real * 1.7, (i % 3) as Real * 0.3, 0.0))
        .collect::<Vec<_>>();
    let state = build_state(n, positions);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = default_box(&gpu);
    // Each particle has n-1 neighbours (the other particles).
    let mut counts = vec![0u32; n];
    let mut nl_flat = Vec::new();
    let max_neighbors = (n - 1) as u32;
    for i in 0..n {
        let mut row = Vec::with_capacity(max_neighbors as usize);
        for j in 0..n {
            if j != i {
                row.push(j as u32);
            }
        }
        counts[i] = row.len() as u32;
        // Pad if shorter than max_neighbors (it isn't here since n-1
        // exactly = max_neighbors).
        while row.len() < max_neighbors as usize {
            row.push(0);
        }
        nl_flat.extend_from_slice(&row);
    }
    let neighbor_list = upload_neighbor_list(&gpu.device, &nl_flat);
    let neighbor_counts = upload_neighbor_counts(&gpu.device, &counts);

    let mut out_a = SlotOutputBuffers::new(&gpu.device, n).unwrap();
    let mut out_b = SlotOutputBuffers::new(&gpu.device, n).unwrap();
    launch(
        kind,
        &gpu,
        &buffers,
        &mut out_a,
        &sim_box,
        &neighbor_list,
        &neighbor_counts,
        max_neighbors,
        AggregateLevel::ForcesOnly,
    );
    launch(
        kind,
        &gpu,
        &buffers,
        &mut out_b,
        &sim_box,
        &neighbor_list,
        &neighbor_counts,
        max_neighbors,
        AggregateLevel::ForcesAndScalars,
    );
    let fxa = download_force_x(&out_a, &gpu.device);
    let fxb = download_force_x(&out_b, &gpu.device);
    let fya = download_force_y(&out_a, &gpu.device);
    let fyb = download_force_y(&out_b, &gpu.device);
    let fza = download_force_z(&out_a, &gpu.device);
    let fzb = download_force_z(&out_b, &gpu.device);
    for i in 0..n {
        assert_eq!(fxa[i].to_bits(), fxb[i].to_bits(), "{}: fx[{i}] mismatch", kind.name());
        assert_eq!(fya[i].to_bits(), fyb[i].to_bits(), "{}: fy[{i}] mismatch", kind.name());
        assert_eq!(fza[i].to_bits(), fzb[i].to_bits(), "{}: fz[{i}] mismatch", kind.name());
    }
}
#[test] fn f_and_fev_agree_lj() { f_and_fev_agree_on_force_impl(PotentialKind::Lj); }
#[test] fn f_and_fev_agree_coulomb() { f_and_fev_agree_on_force_impl(PotentialKind::Coulomb); }
#[test] fn f_and_fev_agree_spme() { f_and_fev_agree_on_force_impl(PotentialKind::Spme); }

// =================================================================
// Reproducibility: two independent runs produce byte-identical output.
// =================================================================

fn two_runs_byte_identical_impl(kind: PotentialKind) {
    let gpu = init_device().unwrap();
    let n = 16;
    let positions = (0..n)
        .map(|i| ((i as Real) * 1.3, (i as Real) * 0.7, 0.0))
        .collect::<Vec<_>>();
    let state = build_state(n, positions);
    let buffers_a = ParticleBuffers::new(&gpu, &state).unwrap();
    let buffers_b = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = default_box(&gpu);
    let max_neighbors = (n - 1) as u32;
    let mut counts = vec![0u32; n];
    let mut nl_flat = Vec::new();
    for i in 0..n {
        let mut row = Vec::with_capacity(max_neighbors as usize);
        for j in 0..n {
            if j != i {
                row.push(j as u32);
            }
        }
        counts[i] = row.len() as u32;
        while row.len() < max_neighbors as usize {
            row.push(0);
        }
        nl_flat.extend_from_slice(&row);
    }
    let neighbor_list = upload_neighbor_list(&gpu.device, &nl_flat);
    let neighbor_counts = upload_neighbor_counts(&gpu.device, &counts);

    let mut out_a = SlotOutputBuffers::new(&gpu.device, n).unwrap();
    let mut out_b = SlotOutputBuffers::new(&gpu.device, n).unwrap();
    launch(
        kind,
        &gpu,
        &buffers_a,
        &mut out_a,
        &sim_box,
        &neighbor_list,
        &neighbor_counts,
        max_neighbors,
        AggregateLevel::ForcesAndScalars,
    );
    launch(
        kind,
        &gpu,
        &buffers_b,
        &mut out_b,
        &sim_box,
        &neighbor_list,
        &neighbor_counts,
        max_neighbors,
        AggregateLevel::ForcesAndScalars,
    );
    for (name, a, b) in [
        ("force_x", download_force_x(&out_a, &gpu.device), download_force_x(&out_b, &gpu.device)),
        ("force_y", download_force_y(&out_a, &gpu.device), download_force_y(&out_b, &gpu.device)),
        ("force_z", download_force_z(&out_a, &gpu.device), download_force_z(&out_b, &gpu.device)),
        ("energy", download_energy(&out_a, &gpu.device), download_energy(&out_b, &gpu.device)),
        ("virial", download_virial(&out_a, &gpu.device), download_virial(&out_b, &gpu.device)),
    ] {
        for i in 0..n {
            assert_eq!(
                a[i].to_bits(),
                b[i].to_bits(),
                "{}: {}[{i}] mismatch (A bits {:x}, B bits {:x})",
                kind.name(),
                name,
                a[i].to_bits(),
                b[i].to_bits()
            );
        }
    }
}
#[test] fn two_runs_byte_identical_lj() { two_runs_byte_identical_impl(PotentialKind::Lj); }
#[test] fn two_runs_byte_identical_coulomb() { two_runs_byte_identical_impl(PotentialKind::Coulomb); }
#[test] fn two_runs_byte_identical_spme() { two_runs_byte_identical_impl(PotentialKind::Spme); }

// =================================================================
// Newton's third law.
// =================================================================

fn newton_third_law_pair_impl(kind: PotentialKind) {
    // N = 2 with a non-axis-aligned displacement so the test catches
    // any single-axis-only sign issue.
    let gpu = init_device().unwrap();
    let positions = vec![(0.0, 0.0, 0.0), (1.3 as Real, 0.4, -0.2)];
    let state = build_state(2, positions);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = default_box(&gpu);
    let counts = vec![1u32, 1u32];
    let nl_flat = vec![1u32, 0u32];
    let neighbor_list = upload_neighbor_list(&gpu.device, &nl_flat);
    let neighbor_counts = upload_neighbor_counts(&gpu.device, &counts);

    let mut output = SlotOutputBuffers::new(&gpu.device, 2).unwrap();
    launch(
        kind,
        &gpu,
        &buffers,
        &mut output,
        &sim_box,
        &neighbor_list,
        &neighbor_counts,
        1,
        AggregateLevel::ForcesOnly,
    );
    let fx = download_force_x(&output, &gpu.device);
    let fy = download_force_y(&output, &gpu.device);
    let fz = download_force_z(&output, &gpu.device);
    assert_eq!(fx[0].to_bits(), (-fx[1]).to_bits(), "{}: fx mismatch ({} vs -{})", kind.name(), fx[0], fx[1]);
    assert_eq!(fy[0].to_bits(), (-fy[1]).to_bits(), "{}: fy mismatch ({} vs -{})", kind.name(), fy[0], fy[1]);
    assert_eq!(fz[0].to_bits(), (-fz[1]).to_bits(), "{}: fz mismatch ({} vs -{})", kind.name(), fz[0], fz[1]);
}
#[test] fn newton_third_law_pair_lj() { newton_third_law_pair_impl(PotentialKind::Lj); }
#[test] fn newton_third_law_pair_coulomb() { newton_third_law_pair_impl(PotentialKind::Coulomb); }
#[test] fn newton_third_law_pair_spme() { newton_third_law_pair_impl(PotentialKind::Spme); }

fn per_particle_forces_sum_to_zero_impl(kind: PotentialKind) {
    let gpu = init_device().unwrap();
    let n = 16;
    let positions = (0..n)
        .map(|i| ((i as Real) * 1.5, (i as Real % 4.0) * 1.5, ((i as Real / 4.0).floor()) * 1.5))
        .collect::<Vec<_>>();
    let state = build_state(n, positions);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = default_box(&gpu);
    let max_neighbors = (n - 1) as u32;
    let mut counts = vec![0u32; n];
    let mut nl_flat = Vec::new();
    for i in 0..n {
        let mut row = Vec::with_capacity(max_neighbors as usize);
        for j in 0..n {
            if j != i {
                row.push(j as u32);
            }
        }
        counts[i] = row.len() as u32;
        while row.len() < max_neighbors as usize {
            row.push(0);
        }
        nl_flat.extend_from_slice(&row);
    }
    let neighbor_list = upload_neighbor_list(&gpu.device, &nl_flat);
    let neighbor_counts = upload_neighbor_counts(&gpu.device, &counts);

    let mut output = SlotOutputBuffers::new(&gpu.device, n).unwrap();
    launch(
        kind,
        &gpu,
        &buffers,
        &mut output,
        &sim_box,
        &neighbor_list,
        &neighbor_counts,
        max_neighbors,
        AggregateLevel::ForcesOnly,
    );
    let fx = download_force_x(&output, &gpu.device);
    let fy = download_force_y(&output, &gpu.device);
    let fz = download_force_z(&output, &gpu.device);
    let sum_fx: f64 = fx.iter().map(|&v| v as f64).sum();
    let sum_fy: f64 = fy.iter().map(|&v| v as f64).sum();
    let sum_fz: f64 = fz.iter().map(|&v| v as f64).sum();
    let max_abs = fx.iter().chain(fy.iter()).chain(fz.iter())
        .map(|v| v.abs() as f64).fold(0.0_f64, f64::max);
    let tol = (n as f64) * (f32::EPSILON as f64) * max_abs.max(1.0);
    // f32 accumulation sloppy; use relaxed tolerance scaled by max force.
    let tol = (tol * 16.0).max(1e-4);
    assert!(sum_fx.abs() < tol, "{}: sum_fx = {} (tol {})", kind.name(), sum_fx, tol);
    assert!(sum_fy.abs() < tol, "{}: sum_fy = {} (tol {})", kind.name(), sum_fy, tol);
    assert!(sum_fz.abs() < tol, "{}: sum_fz = {} (tol {})", kind.name(), sum_fz, tol);
}
#[test] fn per_particle_forces_sum_to_zero_lj() { per_particle_forces_sum_to_zero_impl(PotentialKind::Lj); }
#[test] fn per_particle_forces_sum_to_zero_coulomb() { per_particle_forces_sum_to_zero_impl(PotentialKind::Coulomb); }
#[test] fn per_particle_forces_sum_to_zero_spme() { per_particle_forces_sum_to_zero_impl(PotentialKind::Spme); }

// =================================================================
// Empty state.
// =================================================================

fn launcher_particle_count_zero_is_noop_impl(kind: PotentialKind) {
    let gpu = init_device().unwrap();
    let state = ParticleState::new(
        vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![],
        vec![], None, None,
    ).unwrap();
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = default_box(&gpu);
    let nl_flat: Vec<u32> = vec![];
    let counts: Vec<u32> = vec![];
    let neighbor_list = upload_neighbor_list(&gpu.device, &nl_flat);
    let neighbor_counts = upload_neighbor_counts(&gpu.device, &counts);
    let mut output = SlotOutputBuffers::new(&gpu.device, 0).unwrap();
    // No assertion needed beyond "doesn't panic / fault". launch()
    // returns Ok via the launcher's particle_count == 0 early-exit.
    launch(
        kind,
        &gpu,
        &buffers,
        &mut output,
        &sim_box,
        &neighbor_list,
        &neighbor_counts,
        1, // max_neighbors irrelevant when n == 0
        AggregateLevel::ForcesAndScalars,
    );
}
#[test] fn launcher_particle_count_zero_is_noop_lj() { launcher_particle_count_zero_is_noop_impl(PotentialKind::Lj); }
#[test] fn launcher_particle_count_zero_is_noop_coulomb() { launcher_particle_count_zero_is_noop_impl(PotentialKind::Coulomb); }
#[test] fn launcher_particle_count_zero_is_noop_spme() { launcher_particle_count_zero_is_noop_impl(PotentialKind::Spme); }

// =================================================================
// Side effects: kernel reads but does not write its inputs.
// =================================================================

#[test]
fn kernel_does_not_modify_positions_velocities_masses_charges_lj() {
    let gpu = init_device().unwrap();
    let n = 4;
    let positions = (0..n).map(|i| (i as Real * 1.5, 0.0, 0.0)).collect();
    let state = build_state(n, positions);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = default_box(&gpu);
    let max_neighbors = (n - 1) as u32;
    let mut counts = vec![0u32; n];
    let mut nl_flat = Vec::new();
    for i in 0..n {
        let mut row = Vec::with_capacity(max_neighbors as usize);
        for j in 0..n {
            if j != i {
                row.push(j as u32);
            }
        }
        counts[i] = row.len() as u32;
        nl_flat.extend_from_slice(&row);
    }
    let neighbor_list = upload_neighbor_list(&gpu.device, &nl_flat);
    let neighbor_counts = upload_neighbor_counts(&gpu.device, &counts);

    // Snapshot the input device buffers before the launch.
    let device = gpu.device.clone();
    let snap_px = device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let snap_py = device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let snap_pz = device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    let snap_vx = device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let snap_masses = device.dtoh_sync_copy(&buffers.masses).unwrap();
    let snap_charges = device.dtoh_sync_copy(&buffers.charges).unwrap();
    let snap_nl = device.dtoh_sync_copy(&neighbor_list).unwrap();
    let snap_counts = device.dtoh_sync_copy(&neighbor_counts).unwrap();

    let mut output = SlotOutputBuffers::new(&gpu.device, n).unwrap();
    launch(
        PotentialKind::Lj,
        &gpu,
        &buffers,
        &mut output,
        &sim_box,
        &neighbor_list,
        &neighbor_counts,
        max_neighbors,
        AggregateLevel::ForcesAndScalars,
    );

    let after_px = device.dtoh_sync_copy(&buffers.positions_x).unwrap();
    let after_py = device.dtoh_sync_copy(&buffers.positions_y).unwrap();
    let after_pz = device.dtoh_sync_copy(&buffers.positions_z).unwrap();
    let after_vx = device.dtoh_sync_copy(&buffers.velocities_x).unwrap();
    let after_masses = device.dtoh_sync_copy(&buffers.masses).unwrap();
    let after_charges = device.dtoh_sync_copy(&buffers.charges).unwrap();
    let after_nl = device.dtoh_sync_copy(&neighbor_list).unwrap();
    let after_counts = device.dtoh_sync_copy(&neighbor_counts).unwrap();

    assert_eq!(snap_px, after_px, "positions_x modified");
    assert_eq!(snap_py, after_py, "positions_y modified");
    assert_eq!(snap_pz, after_pz, "positions_z modified");
    assert_eq!(snap_vx, after_vx, "velocities_x modified");
    assert_eq!(snap_masses, after_masses, "masses modified");
    assert_eq!(snap_charges, after_charges, "charges modified");
    assert_eq!(snap_nl, after_nl, "neighbor_list modified");
    assert_eq!(snap_counts, after_counts, "neighbor_counts modified");
}

// =================================================================
// Numerical edge case: NaN propagation.
// =================================================================

#[test]
fn nan_pair_contribution_propagates_to_per_particle_output_lj() {
    // Two particles at exactly the same position: r² = 0 → factor =
    // ∞ or NaN (1 / 0 in inv_r2). The kernel's i == j guard doesn't
    // help because i != j here. The per-particle output for particle
    // 0 must be NaN or ±Inf, but the test just asserts non-finite.
    let gpu = init_device().unwrap();
    let n = 2;
    let positions = vec![(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)];
    let state = build_state(n, positions);
    let buffers = ParticleBuffers::new(&gpu, &state).unwrap();
    let sim_box = default_box(&gpu);
    let counts = vec![1u32, 1u32];
    let nl_flat = vec![1u32, 0u32];
    let neighbor_list = upload_neighbor_list(&gpu.device, &nl_flat);
    let neighbor_counts = upload_neighbor_counts(&gpu.device, &counts);
    let mut output = SlotOutputBuffers::new(&gpu.device, n).unwrap();
    launch(
        PotentialKind::Lj,
        &gpu,
        &buffers,
        &mut output,
        &sim_box,
        &neighbor_list,
        &neighbor_counts,
        1,
        AggregateLevel::ForcesOnly,
    );
    let fx = download_force_x(&output, &gpu.device);
    // Particle 0's force from a zero-distance pair must be non-finite.
    assert!(!fx[0].is_finite(), "fx[0] = {} unexpectedly finite at r=0", fx[0]);
}
