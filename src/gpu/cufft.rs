// rq-846bdb8b
//! Minimal raw FFI bindings to libcufft, plus safe wrappers around 3D
//! real-to-complex and complex-to-real plans used by the SPME reciprocal
//! pipeline. cudarc 0.13 does not include cuFFT support, so this module
//! exposes just enough surface for `forces::spme` to drive the
//! transforms.

use std::os::raw::{c_int, c_void};
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut};

pub type CufftResult = c_int;
pub type CufftType = c_int;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct CufftHandle(pub c_int);

pub const CUFFT_SUCCESS: CufftResult = 0;
pub const CUFFT_R2C: CufftType = 0x2a;
pub const CUFFT_C2R: CufftType = 0x2c;

unsafe extern "C" {
    fn cufftPlan3d(
        plan: *mut CufftHandle,
        nx: c_int,
        ny: c_int,
        nz: c_int,
        ttype: CufftType,
    ) -> CufftResult;
    fn cufftDestroy(plan: CufftHandle) -> CufftResult;
    fn cufftExecR2C(
        plan: CufftHandle,
        idata: *mut f32,
        odata: *mut c_void,
    ) -> CufftResult;
    fn cufftExecC2R(
        plan: CufftHandle,
        idata: *mut c_void,
        odata: *mut f32,
    ) -> CufftResult;
}

#[derive(Debug, thiserror::Error)]
pub enum CuFftError {
    #[error("cuFFT call failed with status code {0}")]
    Status(c_int),
}

fn check(result: CufftResult) -> Result<(), CuFftError> {
    if result == CUFFT_SUCCESS {
        Ok(())
    } else {
        Err(CuFftError::Status(result))
    }
}

/// 3D real-to-complex forward FFT plan. The output is laid out in
/// Hermitian symmetry: `n_a * n_b * (n_c / 2 + 1)` `cufftComplex`
/// entries.
#[derive(Debug)]
pub struct Plan3dR2C {
    handle: CufftHandle,
    pub n_a: u32,
    pub n_b: u32,
    pub n_c: u32,
    // The device is held by reference to ensure that cuFFT's bound
    // context outlives the plan handle. `_device` is otherwise unused.
    _device: Arc<CudaDevice>,
}

impl Plan3dR2C {
    pub fn new(
        device: &Arc<CudaDevice>,
        n_a: u32,
        n_b: u32,
        n_c: u32,
    ) -> Result<Self, CuFftError> {
        device.bind_to_thread().map_err(|_| CuFftError::Status(-1))?;
        let mut handle = CufftHandle(0);
        let result = unsafe {
            cufftPlan3d(
                &mut handle,
                n_a as c_int,
                n_b as c_int,
                n_c as c_int,
                CUFFT_R2C,
            )
        };
        check(result)?;
        Ok(Plan3dR2C {
            handle,
            n_a,
            n_b,
            n_c,
            _device: device.clone(),
        })
    }

    /// Execute the R2C transform `input → output`. `input` must hold
    /// `n_a * n_b * n_c` `f32` values; `output` must hold
    /// `n_a * n_b * (n_c / 2 + 1)` `cufftComplex` values (laid out as
    /// `2 * M_complex` `f32`s with interleaved real/imag parts).
    pub fn execute(
        &self,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CuFftError> {
        let idata = *input.device_ptr() as *mut f32;
        let odata = *output.device_ptr_mut() as *mut c_void;
        let result = unsafe { cufftExecR2C(self.handle, idata, odata) };
        check(result)
    }
}

impl Drop for Plan3dR2C {
    fn drop(&mut self) {
        unsafe { cufftDestroy(self.handle) };
    }
}

/// 3D complex-to-real inverse FFT plan. cuFFT C2R consumes a
/// Hermitian-symmetric input of `n_a * n_b * (n_c / 2 + 1)` complex
/// values and produces `n_a * n_b * n_c` real values. The transform is
/// unnormalised: `IFFT(FFT(x)) = N · x` where `N = n_a * n_b * n_c`.
#[derive(Debug)]
pub struct Plan3dC2R {
    handle: CufftHandle,
    pub n_a: u32,
    pub n_b: u32,
    pub n_c: u32,
    _device: Arc<CudaDevice>,
}

impl Plan3dC2R {
    pub fn new(
        device: &Arc<CudaDevice>,
        n_a: u32,
        n_b: u32,
        n_c: u32,
    ) -> Result<Self, CuFftError> {
        device.bind_to_thread().map_err(|_| CuFftError::Status(-1))?;
        let mut handle = CufftHandle(0);
        let result = unsafe {
            cufftPlan3d(
                &mut handle,
                n_a as c_int,
                n_b as c_int,
                n_c as c_int,
                CUFFT_C2R,
            )
        };
        check(result)?;
        Ok(Plan3dC2R {
            handle,
            n_a,
            n_b,
            n_c,
            _device: device.clone(),
        })
    }

    pub fn execute(
        &self,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CuFftError> {
        let idata = *input.device_ptr() as *mut c_void;
        let odata = *output.device_ptr_mut() as *mut f32;
        let result = unsafe { cufftExecC2R(self.handle, idata, odata) };
        check(result)
    }
}

impl Drop for Plan3dC2R {
    fn drop(&mut self) {
        unsafe { cufftDestroy(self.handle) };
    }
}
