// Combiner kernel for the pluggable potential framework. Sums each
// present slot's private accumulator into ParticleBuffers.forces_*.
//
// The bitmask `present_slots_bitmask` selects which slot pointers to read.
// Bit 0 corresponds to slot 0 (Lennard-Jones), bit 1 to slot 1
// (Morse-bonded). Caller passes valid pointers for present slots only; the
// pointers for absent slots are not dereferenced and may be null.

extern "C" __global__ void accumulate_forces(
    const float *slot0_x, const float *slot0_y, const float *slot0_z,
    const float *slot1_x, const float *slot1_y, const float *slot1_z,
    unsigned int n_slots,
    unsigned int present_slots_bitmask,
    float *forces_x,
    float *forces_y,
    float *forces_z,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  (void) n_slots;

  float sx = 0.0f;
  float sy = 0.0f;
  float sz = 0.0f;

  if ((present_slots_bitmask & 1u) != 0u) {
    sx += slot0_x[i];
    sy += slot0_y[i];
    sz += slot0_z[i];
  }
  if ((present_slots_bitmask & 2u) != 0u) {
    sx += slot1_x[i];
    sy += slot1_y[i];
    sz += slot1_z[i];
  }

  forces_x[i] = sx;
  forces_y[i] = sy;
  forces_z[i] = sz;
}
