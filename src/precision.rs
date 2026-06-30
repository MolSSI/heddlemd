// rq-89c517aa
//! Compile-time precision selector.
//!
//! The canonical floating-point storage and compute type is [`Real`].
//! It resolves to [`f32`] by default and to [`f64`] when the `f64`
//! Cargo feature is on. The selection is purely compile-time: there
//! is no runtime branching on precision anywhere in the engine.

// rq-9edae17f
#[cfg(not(feature = "f64"))]
pub type Real = f32;

#[cfg(feature = "f64")]
pub type Real = f64;

/// Packed 4-tuple of `Real` values used for `posq` (positions
/// interleaved with charges) on the device. Memory layout matches
/// CUDA's `float4` under the default `f32` build and `double4`
/// under `--features f64` (the same `(x, y, z, w)` field order and
/// the same 16- or 32-byte size and natural alignment), so a
/// `CudaSlice<Real4>` can be passed to a kernel parameter typed
/// `const float4*` or `const double4*` without conversion.
///
/// The `.x`, `.y`, `.z` components carry the wrapped position; `.w`
/// carries the per-particle charge.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Real4 {
    pub x: Real,
    pub y: Real,
    pub z: Real,
    pub w: Real,
}

impl Real4 {
    /// Pointwise construction.
    pub const fn new(x: Real, y: Real, z: Real, w: Real) -> Self {
        Real4 { x, y, z, w }
    }

    /// Construction from a 3-tuple position and a separate `w` (the
    /// canonical use case for posq from a SoA particle state).
    pub const fn from_xyzw(x: Real, y: Real, z: Real, w: Real) -> Self {
        Real4 { x, y, z, w }
    }
}

// `Real4` is a POD struct of `Real` fields, so it is valid to
// memset to zero and to copy bit-for-bit between host and device.
unsafe impl cudarc::driver::DeviceRepr for Real4 {}
unsafe impl cudarc::driver::ValidAsZeroBits for Real4 {}

// rq-182dd348
pub const REAL_BYTES: usize = std::mem::size_of::<Real>();

// rq-507c40d1
#[cfg(not(feature = "f64"))]
pub const REAL_IS_F64: bool = false;

#[cfg(feature = "f64")]
pub const REAL_IS_F64: bool = true;

// rq-d4759a9a
#[cfg(not(feature = "f64"))]
pub const REAL_NAME: &str = "f32";

#[cfg(feature = "f64")]
pub const REAL_NAME: &str = "f64";

// rq-f969ffbb
#[cfg(not(feature = "f64"))]
pub const REAL_FMT_DIGITS: usize = 9;

#[cfg(feature = "f64")]
pub const REAL_FMT_DIGITS: usize = 17;

/// Build-dependent CPU-reference tolerance bound used by tests
/// that compare a kernel result against a CPU-computed expected
/// value. Smaller in the `f64` build.
#[cfg(not(feature = "f64"))]
pub const CPU_REFERENCE_TOLERANCE: Real = 1.0e-5;

#[cfg(feature = "f64")]
pub const CPU_REFERENCE_TOLERANCE: Real = 1.0e-13;
