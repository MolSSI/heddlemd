// Behavioural tests for the `gpu_launch!` launch-wrapper macro, exercised
// through the public launch wrappers whose bodies it expands. The macro
// itself is crate-internal (`pub(crate)`), so its per_element and
// single_thread grid strategies are validated here through `vv_kick` (a
// per_element wrapper) and `increment_u64_device` (a single_thread
// wrapper). The single_block strategy and the no-timing invariant are
// covered by crate-internal unit tests in `src/gpu/kernel_macros.rs`.
// See `rqm/build-pipeline.md`.

use heddle_md::gpu::{ParticleBuffers, increment_u64_device, init_device, vv_kick};
use heddle_md::precision::Real;
use heddle_md::state::ParticleState;

// A zero-velocity, unit-mass state with a distinct nonzero x-force on
// every particle, so a per-element velocity kick leaves an observable,
// index-dependent mark on every element (including the last).
fn state_with_forces(n: usize) -> ParticleState {
    let mut state = ParticleState::new(
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        vec![0.0; n],
        vec![0u32; n],
        None,
        None,
    )
    .expect("ParticleState::new");
    state.forces_x = (0..n).map(|i| 1.0 + i as Real).collect();
    state
}

// rq-8bccc91c
#[test]
fn per_element_wrapper_is_noop_on_empty_input() {
    let gpu = init_device().expect("init_device");
    let state = state_with_forces(0);
    let mut buffers = ParticleBuffers::new(&gpu, &state).expect("ParticleBuffers::new");
    assert_eq!(buffers.particle_count(), 0);
    // The per_element empty guard returns Ok(()) without launching. A
    // launch would otherwise compute a zero-block grid, which the driver
    // rejects — so reaching Ok here is the observable no-op.
    vv_kick(&mut buffers, 0.5).expect("vv_kick on empty buffers must return Ok(())");
}

// rq-40f78e04
#[test]
fn per_element_wrapper_processes_last_element_of_non_block_aligned_size() {
    let gpu = init_device().expect("init_device");
    // 1000 is not a multiple of BLOCK_SIZE (256); the grid is
    // ceil(1000 / 256) = 4 blocks and the last element lives in the
    // partially-filled final block.
    let n = 1000usize;
    let state = state_with_forces(n);
    let mut buffers = ParticleBuffers::new(&gpu, &state).expect("ParticleBuffers::new");

    vv_kick(&mut buffers, 0.5).expect("vv_kick");

    let mut snapshot = state_with_forces(n);
    snapshot
        .download_from(&buffers)
        .expect("download_from failed");
    // Every particle started at zero velocity with a nonzero x-force, so
    // every processed element now has a nonzero x-velocity. Checking the
    // final element proves the grid covered the last, non-block-aligned
    // index rather than truncating at a block boundary.
    assert_ne!(
        snapshot.velocities_x[n - 1],
        0.0 as Real,
        "the last element of a non-block-aligned size must be processed"
    );
    assert!(
        snapshot.velocities_x.iter().all(|&v| v != 0.0 as Real),
        "every element must be processed"
    );
}

// rq-1474f94e
#[test]
fn single_thread_wrapper_launches_and_updates_device_output() {
    let gpu = init_device().expect("init_device");
    let state = state_with_forces(1);
    let buffers = ParticleBuffers::new(&gpu, &state).expect("ParticleBuffers::new");

    // A device-resident counter, zero-initialized. The single_thread
    // wrapper has no empty guard, so it always launches its one thread.
    let mut counter = gpu
        .device
        .alloc_zeros::<u64>(1)
        .expect("alloc counter");
    increment_u64_device(&buffers, &mut counter).expect("increment_u64_device");

    let mut host = [0u64; 1];
    gpu.device
        .dtoh_sync_copy_into(&counter, &mut host)
        .expect("download counter");
    assert_eq!(
        host[0], 1,
        "single_thread launch must run once and increment the device counter"
    );
}
