// rq-b2f23140

#pragma once

__device__ static inline float exclusion_scale(
    unsigned int i,
    unsigned int j,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const float *atom_excl_scales)
{
  unsigned int start = atom_excl_offsets[i];
  unsigned int end = atom_excl_offsets[i + 1];
  for (unsigned int m = start; m < end; ++m) {
    if (atom_excl_partners[m] == j) {
      return atom_excl_scales[m];
    }
  }
  return 1.0f;
}
