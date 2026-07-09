//! Behaviour of the shared `htod_or_empty` upload helper. See
//! `rqm/forces/framework.md`.

use cudarc::driver::DeviceSlice;

use heddle_md::gpu::{htod_or_empty, init_device};
use heddle_md::precision::Real;

#[test] // rq-f92cfcdf
fn htod_or_empty_uploads_non_empty_host_slice() {
    let gpu = init_device().unwrap();
    let data: Vec<Real> = vec![1.0, 2.0, 3.0];
    let slice = htod_or_empty(&gpu.device, &data).unwrap();
    assert_eq!(slice.len(), 3);
    let back = gpu.device.dtoh_sync_copy(&slice).unwrap();
    assert_eq!(back, data);
}

#[test] // rq-737f68cd
fn htod_or_empty_returns_zero_length_buffer_for_empty_host_slice() {
    let gpu = init_device().unwrap();
    let data: Vec<u32> = Vec::new();
    let slice = htod_or_empty(&gpu.device, &data).unwrap();
    assert_eq!(slice.len(), 0);
}
