#include "precision.cuh"

extern "C" __global__ void fill(Real *out, Real value, unsigned int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    out[index] = value;
  }
}
