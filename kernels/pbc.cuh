// rq-4ca9b179
// Fractional-coordinate wrap helpers shared by every kernel that
// applies periodic boundary conditions. The wrap computes the
// fractional coordinates of the input via back-substitution
// (z-then-y-then-x), picks the integer image triple that brings each
// component into [-1/2, 1/2), and applies the image-vector correction
// directly in Cartesian coordinates. The output has fractional
// coordinates in [-1/2, 1/2)³ and lies inside the primary
// parallelepiped. For an orthorhombic box (xy = xz = yz = 0) the
// algorithm reduces to three independent per-axis wraps that match the
// v0 orthorhombic implementation bit-for-bit.

#ifndef DYNAMICS_KERNELS_PBC_CUH
#define DYNAMICS_KERNELS_PBC_CUH
#include "precision.cuh"

// Cartesian -> fractional coordinates via back-substitution.
__device__ static inline void triclinic_cart_to_frac(
    Real x, Real y, Real z,
    Real lx, Real ly, Real lz,
    Real xy, Real xz, Real yz,
    Real &s_a, Real &s_b, Real &s_c)
{
  s_c = z / lz;
  s_b = (y - s_c * yz) / ly;
  s_a = (x - s_b * xy - s_c * xz) / lx;
}

// In-place minimum-image wrap. Discards image counts.
__device__ static inline void triclinic_min_image(
    Real &dx, Real &dy, Real &dz,
    Real lx, Real ly, Real lz,
    Real xy, Real xz, Real yz)
{
  Real s_a, s_b, s_c;
  triclinic_cart_to_frac(dx, dy, dz, lx, ly, lz, xy, xz, yz, s_a, s_b, s_c);
  Real ka = Real_floor(s_a + R(0.5));
  Real kb = Real_floor(s_b + R(0.5));
  Real kc = Real_floor(s_c + R(0.5));
  dx -= ka * lx + kb * xy + kc * xz;
  dy -= kb * ly + kc * yz;
  dz -= kc * lz;
}

// In-place wrap returning the integer image counts (k_a, k_b, k_c).
__device__ static inline void triclinic_wrap_with_image(
    Real &x, Real &y, Real &z,
    int &k_a, int &k_b, int &k_c,
    Real lx, Real ly, Real lz,
    Real xy, Real xz, Real yz)
{
  Real s_a, s_b, s_c;
  triclinic_cart_to_frac(x, y, z, lx, ly, lz, xy, xz, yz, s_a, s_b, s_c);
  Real ka = Real_floor(s_a + R(0.5));
  Real kb = Real_floor(s_b + R(0.5));
  Real kc = Real_floor(s_c + R(0.5));
  x -= ka * lx + kb * xy + kc * xz;
  y -= kb * ly + kc * yz;
  z -= kc * lz;
  k_a = (int) ka;
  k_b = (int) kb;
  k_c = (int) kc;
}

#endif // DYNAMICS_KERNELS_PBC_CUH
