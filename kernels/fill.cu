extern "C" __global__ void fill(float *out, float value, unsigned int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    out[index] = value;
  }
}
