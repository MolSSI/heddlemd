// rq-9922bfd1
// Compile-time precision selector for CUDA kernels.
//
// Real is float by default and double when nvcc is invoked with
// -DREAL_F64 (build.rs passes this flag iff the f64 Cargo feature
// is on). Every .cu / .cuh under kernels/ includes this header
// directly or transitively and uses Real instead of float and the
// Real_* shim functions instead of precision-suffixed CUDA math
// intrinsics. This file is the only place precision-suffixed
// intrinsic names appear.

#ifndef HEDDLEMD_PRECISION_CUH
#define HEDDLEMD_PRECISION_CUH

#include <math_constants.h>

#ifdef REAL_F64
// rq-0c7e2ed1
typedef double Real;
// rq-928da8e4
typedef double2 Real2;
typedef double4 Real4;
#else
typedef float Real;
typedef float2 Real2;
typedef float4 Real4;
#endif

// rq-4dd1b9e3
// Cast a numeric literal to Real at the call site. Use R(0.5) instead
// of 0.5f or 0.5 anywhere a bare literal would otherwise produce a
// narrowing-conversion warning when Real == float, or unwanted
// promotion to double when Real == float.
#define R(x) (static_cast<Real>(x))

#ifdef REAL_F64

// rq-c3d3dc8e
__device__ __forceinline__ Real Real_sqrt(Real x) { return sqrt(x); }
// rq-24048902
__device__ __forceinline__ Real Real_rsqrt(Real x) { return rsqrt(x); }
// rq-48e7115e
__device__ __forceinline__ Real Real_exp(Real x) { return exp(x); }
// rq-c6364175
__device__ __forceinline__ Real Real_log(Real x) { return log(x); }
// rq-043796fa
__device__ __forceinline__ Real Real_sin(Real x) { return sin(x); }
// rq-23398f89
__device__ __forceinline__ Real Real_cos(Real x) { return cos(x); }
// rq-781ce8fc
__device__ __forceinline__ Real Real_pow(Real x, Real y) { return pow(x, y); }
// rq-13070c27
__device__ __forceinline__ Real Real_fabs(Real x) { return fabs(x); }
// rq-3f77dca9
__device__ __forceinline__ Real Real_floor(Real x) { return floor(x); }
// rq-688a5b6c
__device__ __forceinline__ Real Real_rint(Real x) { return rint(x); }
// rq-02f5d756
__device__ __forceinline__ Real Real_fma(Real x, Real y, Real z) { return fma(x, y, z); }
// rq-7d9d35f7
__device__ __forceinline__ void Real_sincos(Real x, Real *s, Real *c) { sincos(x, s, c); }

#else  // single precision

__device__ __forceinline__ Real Real_sqrt(Real x) { return sqrtf(x); }
__device__ __forceinline__ Real Real_rsqrt(Real x) { return rsqrtf(x); }
__device__ __forceinline__ Real Real_exp(Real x) { return expf(x); }
__device__ __forceinline__ Real Real_log(Real x) { return logf(x); }
__device__ __forceinline__ Real Real_sin(Real x) { return sinf(x); }
__device__ __forceinline__ Real Real_cos(Real x) { return cosf(x); }
__device__ __forceinline__ Real Real_pow(Real x, Real y) { return powf(x, y); }
__device__ __forceinline__ Real Real_fabs(Real x) { return fabsf(x); }
__device__ __forceinline__ Real Real_floor(Real x) { return floorf(x); }
__device__ __forceinline__ Real Real_rint(Real x) { return rintf(x); }
__device__ __forceinline__ Real Real_fma(Real x, Real y, Real z) { return fmaf(x, y, z); }
__device__ __forceinline__ void Real_sincos(Real x, Real *s, Real *c) { sincosf(x, s, c); }

#endif

#endif  // HEDDLEMD_PRECISION_CUH
