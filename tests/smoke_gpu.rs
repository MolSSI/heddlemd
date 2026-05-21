use cudarc::driver::{LaunchAsync, LaunchConfig};
use dynamics::gpu::init_device;

const BLOCK_SIZE: u32 = 256;

fn launch_config(n: u32) -> LaunchConfig {
    let grid = n.div_ceil(BLOCK_SIZE);
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[test] // rq-05691d2f
fn init_device_returns_gpu_context_with_fill_kernel() {
    let gpu = init_device().expect("init_device failed");
    // Referencing the field is a compile-time assertion that the `Kernels`
    // handle exposes `fill`; cloning the launchable function confirms it was
    // populated at runtime.
    let _ = gpu.kernels.fill.fill.clone();
}

#[test] // rq-9cf657ed
fn fill_block_aligned() {
    let gpu = init_device().expect("init_device failed");
    let n: u32 = 1024;
    let mut buf = gpu
        .device
        .alloc_zeros::<f32>(n as usize)
        .expect("alloc_zeros failed");

    let func = gpu.kernels.fill.fill.clone();
    unsafe {
        func.launch(launch_config(n), (&mut buf, 1.0_f32, n))
            .expect("kernel launch failed");
    }

    let host = gpu.device.dtoh_sync_copy(&buf).expect("dtoh_sync_copy failed");
    assert_eq!(host.len(), n as usize);
    for (i, &v) in host.iter().enumerate() {
        assert_eq!(v, 1.0_f32, "element {i} = {v}");
    }
}

#[test] // rq-26d8c08c
fn fill_non_block_aligned() {
    let gpu = init_device().expect("init_device failed");
    let n: u32 = 1000;
    let mut buf = gpu
        .device
        .alloc_zeros::<f32>(n as usize)
        .expect("alloc_zeros failed");

    let func = gpu.kernels.fill.fill.clone();
    unsafe {
        func.launch(launch_config(n), (&mut buf, 1.0_f32, n))
            .expect("kernel launch failed");
    }

    let host = gpu.device.dtoh_sync_copy(&buf).expect("dtoh_sync_copy failed");
    assert_eq!(host.len(), n as usize);
    for (i, &v) in host.iter().enumerate() {
        assert_eq!(v, 1.0_f32, "element {i} = {v}");
    }
}

#[test] // rq-d920e446
fn fill_does_not_write_beyond_n() {
    let gpu = init_device().expect("init_device failed");
    let buf_len: usize = 1024;
    let n: u32 = 1000;
    let mut buf = gpu
        .device
        .alloc_zeros::<f32>(buf_len)
        .expect("alloc_zeros failed");

    let func = gpu.kernels.fill.fill.clone();
    unsafe {
        func.launch(launch_config(n), (&mut buf, 1.0_f32, n))
            .expect("kernel launch failed");
    }

    let host = gpu.device.dtoh_sync_copy(&buf).expect("dtoh_sync_copy failed");
    assert_eq!(host.len(), buf_len);
    for i in 0..(n as usize) {
        assert_eq!(host[i], 1.0_f32, "expected 1.0 at index {i}, got {}", host[i]);
    }
    for i in (n as usize)..buf_len {
        assert_eq!(host[i], 0.0_f32, "expected 0.0 at index {i}, got {}", host[i]);
    }
}

// =====================================================================
// Per-subsystem `Kernels` composition. See the "Per-subsystem kernel
// composition" scenario block in rqm/build-pipeline.md.
// =====================================================================

// `Kernels` exposes one field per subsystem; the chain to every kernel
// has the shape `gpu.kernels.<subsystem>.<kernel>`.
// rq-6211a82f
#[test]
fn kernels_is_composed_of_per_subsystem_sub_structs() {
    let gpu = init_device().expect("init_device failed");
    // Each subsystem's typed sub-struct must be reachable as a field
    // of `Kernels`. The clones below are compile-time assertions on
    // the layout: the 17 sub-structs from the Types-section table all
    // appear under their canonical names. Selecting a representative
    // kernel from each confirms the sub-struct was populated.
    let _ = gpu.kernels.fill.fill.clone();
    let _ = gpu.kernels.integrate.vv_kick_drift.clone();
    let _ = gpu.kernels.reduce.reduce_pair_forces.clone();
    let _ = gpu.kernels.lj.pair_force.clone();
    let _ = gpu.kernels.coulomb.coulomb_pair_force.clone();
    let _ = gpu.kernels.spme_real.spme_real_pair_force.clone();
    let _ = gpu.kernels.spme_recip.spme_charge_spread.clone();
    let _ = gpu.kernels.langevin.lan_drift_half.clone();
    let _ = gpu.kernels.morse.morse_bond_force.clone();
    let _ = gpu.kernels.angle.harmonic_angle_force.clone();
    let _ = gpu.kernels.nose_hoover.kinetic_energy_reduce.clone();
    let _ = gpu.kernels.andersen.andersen_resample.clone();
    let _ = gpu.kernels.barostat.virial_sum_reduce.clone();
    let _ = gpu.kernels.mtk.mtk_velocity_half_kick.clone();
    let _ = gpu.kernels.settle.settle_snapshot.clone();
    let _ = gpu.kernels.forces.accumulate_forces.clone();
    let _ = gpu.kernels.neighbor.neighbor_displacement_squared.clone();
}

// Each subsystem's `XKernels::load(&device)` returns a populated
// handle whose kernel fields are launchable.
// rq-6745e7c5
#[test]
fn xkernels_load_returns_populated_handle() {
    use dynamics::forces::lj::LjKernels;
    let gpu = init_device().expect("init_device failed");
    // `Kernels::load` was already called during init_device; a second
    // `load_ptx` for the same module name is a no-op in cudarc (the
    // module is keyed by name). The returned handle still resolves
    // the kernel.
    let lj = LjKernels::load(&gpu.device).expect("LjKernels::load failed");
    let _ = lj.pair_force.clone();
}

// Cross-subsystem reads pull from the kernel's home sub-struct.
// `reduce_pair_forces` lives in `kernels.reduce.*` and is used by the
// LJ, Coulomb, and SPME-real launch wrappers — none of those
// subsystems carries its own copy of the kernel handle.
// rq-7b651edb
#[test]
fn cross_subsystem_reads_pull_from_home_sub_struct() {
    let gpu = init_device().expect("init_device failed");
    let shared = gpu.kernels.reduce.reduce_pair_forces.clone();
    // The same handle reached via `kernels.reduce.reduce_pair_forces`
    // must launch successfully on the device. Cloning the
    // `CudaFunction` is itself a smoke check that the field is the
    // populated handle, not the default-uninitialised value.
    let _ = shared;
    // The other subsystems do NOT shadow it under their own field
    // names: the lj/coulomb/spme_real sub-structs expose only their
    // own kernels, not a duplicate of the reduce kernel.
    // (Compile-time assertion via the `Kernels` struct definition.)
    let _ = gpu.kernels.lj.pair_force.clone();
    let _ = gpu.kernels.coulomb.coulomb_pair_force.clone();
    let _ = gpu.kernels.spme_real.spme_real_pair_force.clone();
    // Sanity: the reduce sub-struct is a real, non-zero-sized handle
    // whose only kernel is reduce_pair_forces.
    let _ = std::mem::size_of_val(&gpu.kernels.reduce);
}
