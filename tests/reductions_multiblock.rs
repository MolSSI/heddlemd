//! Determinism and correctness of the multi-block scalar reductions
//! (the large-`N` path, `n > SINGLE_BLOCK_REDUCE_MAX`). The single-block
//! path is exercised by the thermostat/barostat tests with small systems;
//! these tests cover the two-pass multi-block path that the benchmark
//! sizes take. See `rqm/integration/nose-hoover-chain.md`.

use heddle_md::gpu::{ParticleBuffers, compute_kinetic_energy, compute_total_virial, init_device};
use heddle_md::precision::Real;
use heddle_md::state::ParticleState;

// 10_000 atoms > SINGLE_BLOCK_REDUCE_MAX (8192), so the reductions take
// the deterministic two-pass multi-block path.
const N: usize = 10_000;

fn make_buffers(gpu: &heddle_md::gpu::GpuContext) -> (ParticleBuffers, Vec<Real>, Vec<Real>) {
    // Deterministic pseudo-random velocities/masses/virials (no rng crate).
    let mut vx = vec![0.0 as Real; N];
    let mut vy = vec![0.0 as Real; N];
    let mut vz = vec![0.0 as Real; N];
    let mut mass = vec![0.0 as Real; N];
    let mut virials = vec![0.0 as Real; N];
    for i in 0..N {
        let f = i as Real;
        vx[i] = (f * 0.7).sin();
        vy[i] = (f * 1.3 + 0.4).sin();
        vz[i] = (f * 0.9 + 1.1).sin();
        mass[i] = 1.0 + (f * 0.05).sin().abs();
        virials[i] = (f * 0.31 + 0.2).cos() * 2.0;
    }
    let mut state = ParticleState::new(
        vec![0.0; N],
        vec![0.0; N],
        vec![0.0; N],
        vx.clone(),
        vy.clone(),
        vz.clone(),
        mass.clone(),
        vec![0.0; N],
        vec![0u32; N],
        None,
        None,
    )
    .unwrap();
    state.virials = virials.clone();
    let buffers = ParticleBuffers::new(gpu, &state).unwrap();
    // Host reference kinetic energy (f64 accumulation).
    let ke_ref: f64 = (0..N)
        .map(|i| {
            0.5 * mass[i] as f64
                * ((vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]) as f64)
        })
        .sum();
    let w_ref: f64 = virials.iter().map(|&v| v as f64).sum();
    (buffers, vec![ke_ref as Real], vec![w_ref as Real])
}

// rq-1727d6bd
#[test]
fn multiblock_kinetic_energy_is_deterministic_and_correct() {
    let gpu = init_device().unwrap();
    let (mut buffers, ke_ref, _) = make_buffers(&gpu);
    let mut scratch = gpu.device.alloc_zeros::<Real>(1).unwrap();

    let k1 = compute_kinetic_energy(&mut buffers, &mut scratch).unwrap();
    let k2 = compute_kinetic_energy(&mut buffers, &mut scratch).unwrap();

    // Deterministic: two reductions of the same data are bit-identical.
    assert_eq!(k1.to_bits(), k2.to_bits(), "multi-block KE must be bit-reproducible");
    // Correct: within a small relative tolerance of the f64 reference
    // (the device path is f32, so exact equality is not expected).
    let rel = ((k1 as f64) - ke_ref[0] as f64).abs() / (ke_ref[0] as f64).abs();
    assert!(rel < 1.0e-4, "KE {k1} vs ref {} (rel {rel})", ke_ref[0]);
}

// rq-1727d6bd
#[test]
fn multiblock_virial_is_deterministic_and_correct() {
    let gpu = init_device().unwrap();
    let (mut buffers, _, w_ref) = make_buffers(&gpu);
    let mut scratch = gpu.device.alloc_zeros::<Real>(1).unwrap();

    let w1 = compute_total_virial(&mut buffers, &mut scratch).unwrap();
    let w2 = compute_total_virial(&mut buffers, &mut scratch).unwrap();

    assert_eq!(w1.to_bits(), w2.to_bits(), "multi-block virial must be bit-reproducible");
    let rel = ((w1 as f64) - w_ref[0] as f64).abs() / (w_ref[0] as f64).abs();
    assert!(rel < 1.0e-4, "virial {w1} vs ref {} (rel {rel})", w_ref[0]);
}
