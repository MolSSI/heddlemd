// rq-4ddab3c7

extern "C" __global__ void lj_pair_force(
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    float *pair_forces_x,
    float *pair_forces_y,
    float *pair_forces_z,
    unsigned int max_neighbors,
    float lx, float ly, float lz,
    float sigma,
    float epsilon,
    float cutoff,
    unsigned int n)
{
  unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || k >= n) {
    return;
  }

  unsigned int slot = i * max_neighbors + k;

  if (i == k) {
    pair_forces_x[slot] = 0.0f;
    pair_forces_y[slot] = 0.0f;
    pair_forces_z[slot] = 0.0f;
    return;
  }

  float dx = positions_x[i] - positions_x[k];
  float dy = positions_y[i] - positions_y[k];
  float dz = positions_z[i] - positions_z[k];

  dx = dx - lx * floorf((dx + lx * 0.5f) / lx);
  dy = dy - ly * floorf((dy + ly * 0.5f) / ly);
  dz = dz - lz * floorf((dz + lz * 0.5f) / lz);

  float r2 = dx * dx + dy * dy + dz * dz;
  if (r2 > cutoff * cutoff) {
    pair_forces_x[slot] = 0.0f;
    pair_forces_y[slot] = 0.0f;
    pair_forces_z[slot] = 0.0f;
    return;
  }

  float inv_r2 = 1.0f / r2;
  float sigma2 = sigma * sigma;
  float sr2 = sigma2 * inv_r2;
  float sr6 = sr2 * sr2 * sr2;
  float sr12 = sr6 * sr6;
  float factor = 24.0f * epsilon * inv_r2 * (2.0f * sr12 - sr6);

  pair_forces_x[slot] = factor * dx;
  pair_forces_y[slot] = factor * dy;
  pair_forces_z[slot] = factor * dz;
}
