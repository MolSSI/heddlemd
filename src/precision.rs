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
