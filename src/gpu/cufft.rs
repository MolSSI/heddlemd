// rq-846bdb8b
//! Minimal raw FFI bindings to libcufft, plus safe wrappers around 3D
//! real-to-complex and complex-to-real plans used by the SPME reciprocal
//! pipeline. cudarc 0.13 does not include cuFFT support, so this module
//! exposes just enough surface for `forces::spme` to drive the
//! transforms.

use std::os::raw::{c_int, c_void};
use std::sync::Arc;

use cudarc::driver::sys::CUstream;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut};
use crate::precision::Real;

pub type CufftResult = c_int;
pub type CufftType = c_int;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct CufftHandle(pub c_int);

pub const CUFFT_SUCCESS: CufftResult = 0;
pub const CUFFT_R2C: CufftType = 0x2a;
pub const CUFFT_C2R: CufftType = 0x2c;
// rq-d2zforward
pub const CUFFT_D2Z: CufftType = 0x6a;
// rq-z2dinverse
pub const CUFFT_Z2D: CufftType = 0x6c;

#[allow(dead_code)]
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
    fn cufftExecD2Z(
        plan: CufftHandle,
        idata: *mut f64,
        odata: *mut c_void,
    ) -> CufftResult;
    fn cufftExecZ2D(
        plan: CufftHandle,
        idata: *mut c_void,
        odata: *mut f64,
    ) -> CufftResult;
    fn cufftSetStream(plan: CufftHandle, stream: *mut c_void) -> CufftResult;
}

#[cfg(not(feature = "f64"))]
const CUFFT_FWD_TYPE: CufftType = CUFFT_R2C;
#[cfg(feature = "f64")]
const CUFFT_FWD_TYPE: CufftType = CUFFT_D2Z;

#[cfg(not(feature = "f64"))]
const CUFFT_INV_TYPE: CufftType = CUFFT_C2R;
#[cfg(feature = "f64")]
const CUFFT_INV_TYPE: CufftType = CUFFT_Z2D;

// rq-1ad7e751
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
                CUFFT_FWD_TYPE,
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

    // rq-4c21c386
    /// Execute the R2C transform `input → output`. `input` must hold
    /// `n_a * n_b * n_c` `Real` values; `output` must hold
    /// `n_a * n_b * (n_c / 2 + 1)` `RealComplex` values (laid out as
    /// `2 * M_complex` `Real`s with interleaved real/imag parts).
    pub fn execute(
        &self,
        input: &CudaSlice<Real>,
        output: &mut CudaSlice<Real>,
    ) -> Result<(), CuFftError> {
        let odata = *output.device_ptr_mut() as *mut c_void;
        #[cfg(not(feature = "f64"))]
        let result = {
            let idata = *input.device_ptr() as *mut f32;
            unsafe { cufftExecR2C(self.handle, idata, odata) }
        };
        #[cfg(feature = "f64")]
        let result = {
            let idata = *input.device_ptr() as *mut f64;
            unsafe { cufftExecD2Z(self.handle, idata, odata) }
        };
        check(result)
    }

    /// Bind the plan to the given CUDA stream. Subsequent `execute()`
    /// calls run on this stream. Per rqm/forces/spme.md, the SPME
    /// reciprocal plans are bound once at slot construction and never
    /// rebound.
    pub fn set_stream(&self, stream: CUstream) -> Result<(), CuFftError> {
        let result = unsafe { cufftSetStream(self.handle, stream as *mut c_void) };
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
                CUFFT_INV_TYPE,
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
        input: &CudaSlice<Real>,
        output: &mut CudaSlice<Real>,
    ) -> Result<(), CuFftError> {
        let idata = *input.device_ptr() as *mut c_void;
        #[cfg(not(feature = "f64"))]
        let result = {
            let odata = *output.device_ptr_mut() as *mut f32;
            unsafe { cufftExecC2R(self.handle, idata, odata) }
        };
        #[cfg(feature = "f64")]
        let result = {
            let odata = *output.device_ptr_mut() as *mut f64;
            unsafe { cufftExecZ2D(self.handle, idata, odata) }
        };
        check(result)
    }

    /// Bind the plan to the given CUDA stream. Subsequent `execute()`
    /// calls run on this stream.
    pub fn set_stream(&self, stream: CUstream) -> Result<(), CuFftError> {
        let result = unsafe { cufftSetStream(self.handle, stream as *mut c_void) };
        check(result)
    }
}

impl Drop for Plan3dC2R {
    fn drop(&mut self) {
        unsafe { cufftDestroy(self.handle) };
    }
}

// rq-d880c228 rq-637cd1a5 rq-02f4d342
//
// Runs an R2C forward FFT on a fixed input twice on the same device,
// returning the number of float positions that differ bit-for-bit
// between the two runs. cuFFT's plan selection on a given GPU is
// deterministic for a fixed problem size, so the expected return value
// is 0. A non-zero count means cuFFT chose different algorithms
// between calls (driver/library issue or hardware unsupported for the
// determinism guarantee), and the runner refuses to start an SPME run
// in that case.
//
// The probe size (`n_a = n_b = n_c = 16`) is small enough to incur
// negligible setup cost yet large enough to exercise the multi-axis
// plan selector. The input is a fixed deterministic pattern (rather
// than zeros), so any algorithmic divergence produces a numerically
// distinguishable bit pattern.
pub fn cufft_determinism_smoke_test(
    device: &Arc<CudaDevice>,
) -> Result<usize, CuFftError> {
    const N: u32 = 16;
    let m = (N * N * N) as usize;
    let m_complex = (N as usize * N as usize) * (N as usize / 2 + 1);

    let mut input_host = vec![0.0; m];
    for (i, v) in input_host.iter_mut().enumerate() {
        // Mix in a triangular and a frequency pattern so the FFT
        // output spreads across many output cells.
        let x = (i as Real) * 0.123456;
        *v = x.sin() + 0.5 * x.cos();
    }
    let input = device
        .htod_sync_copy(&input_host)
        .map_err(|_| CuFftError::Status(-1))?;

    let plan = Plan3dR2C::new(device, N, N, N)?;
    let mut out_a = device
        .alloc_zeros::<Real>(2 * m_complex)
        .map_err(|_| CuFftError::Status(-1))?;
    let mut out_b = device
        .alloc_zeros::<Real>(2 * m_complex)
        .map_err(|_| CuFftError::Status(-1))?;
    plan.execute(&input, &mut out_a)?;
    plan.execute(&input, &mut out_b)?;

    let host_a: Vec<Real> = device
        .dtoh_sync_copy(&out_a)
        .map_err(|_| CuFftError::Status(-1))?;
    let host_b: Vec<Real> = device
        .dtoh_sync_copy(&out_b)
        .map_err(|_| CuFftError::Status(-1))?;
    let differences = host_a
        .iter()
        .zip(host_b.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    Ok(differences)
}
