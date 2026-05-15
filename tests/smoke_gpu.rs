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
    let _ = gpu.kernels.fill.clone();
}

#[test] // rq-9cf657ed
fn fill_block_aligned() {
    let gpu = init_device().expect("init_device failed");
    let n: u32 = 1024;
    let mut buf = gpu
        .device
        .alloc_zeros::<f32>(n as usize)
        .expect("alloc_zeros failed");

    let func = gpu.kernels.fill.clone();
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

    let func = gpu.kernels.fill.clone();
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

    let func = gpu.kernels.fill.clone();
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
