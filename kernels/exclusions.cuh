// rq-b2f23140

#include "precision.cuh"

#pragma once

__device__ static inline Real exclusion_scale(
    unsigned int i,
    unsigned int j,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const Real *atom_excl_scales)
{
  unsigned int start = atom_excl_offsets[i];
  unsigned int end = atom_excl_offsets[i + 1];
  for (unsigned int m = start; m < end; ++m) {
    if (atom_excl_partners[m] == j) {
      return atom_excl_scales[m];
    }
  }
  return R(1.0);
}
