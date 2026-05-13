// rq-4ddab3c7

extern "C" __global__ void lj_pair_force(
    const float *positions_x,
    const float *positions_y,
    const float *positions_z,
    const unsigned int *type_indices,
    float *pair_forces_x,
    float *pair_forces_y,
    float *pair_forces_z,
    float *pair_energies,
    float *pair_virials,
    unsigned int max_neighbors,
    float lx, float ly, float lz,
    unsigned int n_types,
    const float *type_sigma,
    const float *type_epsilon,
    const float *type_cutoff,
    const unsigned int *atom_excl_offsets,
    const unsigned int *atom_excl_partners,
    const float *atom_excl_scales,
    const unsigned int *neighbor_list,
    const unsigned int *neighbor_counts,
    unsigned int n)
{
  unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || k >= max_neighbors) {
    return;
  }
  unsigned int slot = i * max_neighbors + k;
  if (k >= neighbor_counts[i]) {
    pair_forces_x[slot] = 0.0f;
    pair_forces_y[slot] = 0.0f;
    pair_forces_z[slot] = 0.0f;
    pair_energies[slot] = 0.0f;
    pair_virials[slot]  = 0.0f;
    return;
  }
  unsigned int j = neighbor_list[slot];

  if (i == j) {
    pair_forces_x[slot] = 0.0f;
    pair_forces_y[slot] = 0.0f;
    pair_forces_z[slot] = 0.0f;
    pair_energies[slot] = 0.0f;
    pair_virials[slot]  = 0.0f;
    return;
  }

  unsigned int ti = type_indices[i];
  unsigned int tj = type_indices[j];
  unsigned int p = ti * n_types + tj;
  float sigma = type_sigma[p];
  float epsilon = type_epsilon[p];
  float cutoff = type_cutoff[p];

  float dx = positions_x[i] - positions_x[j];
  float dy = positions_y[i] - positions_y[j];
  float dz = positions_z[i] - positions_z[j];

  dx = dx - lx * floorf((dx + lx * 0.5f) / lx);
  dy = dy - ly * floorf((dy + ly * 0.5f) / ly);
  dz = dz - lz * floorf((dz + lz * 0.5f) / lz);

  float r2 = dx * dx + dy * dy + dz * dz;
  if (r2 > cutoff * cutoff) {
    pair_forces_x[slot] = 0.0f;
    pair_forces_y[slot] = 0.0f;
    pair_forces_z[slot] = 0.0f;
    pair_energies[slot] = 0.0f;
    pair_virials[slot]  = 0.0f;
    return;
  }

  float inv_r2 = 1.0f / r2;
  float sigma2 = sigma * sigma;
  float sr2 = sigma2 * inv_r2;
  float sr6 = sr2 * sr2 * sr2;
  float sr12 = sr6 * sr6;
  float factor = 24.0f * epsilon * inv_r2 * (2.0f * sr12 - sr6);
  float energy = 4.0f * epsilon * (sr12 - sr6);

  float fx = factor * dx;
  float fy = factor * dy;
  float fz = factor * dz;
  float w = fx * dx + fy * dy + fz * dz;

  // rq-dddcbf07
  unsigned int start = atom_excl_offsets[i];
  unsigned int end = atom_excl_offsets[i + 1];
  float scale = 1.0f;
  for (unsigned int m = start; m < end; ++m) {
    if (atom_excl_partners[m] == j) {
      scale = atom_excl_scales[m];
      break;
    }
  }
  fx *= scale;
  fy *= scale;
  fz *= scale;
  energy *= scale;
  w *= scale;

  pair_forces_x[slot] = fx;
  pair_forces_y[slot] = fy;
  pair_forces_z[slot] = fz;
  pair_energies[slot] = energy * 0.5f;
  pair_virials[slot]  = w * 0.5f;
}
