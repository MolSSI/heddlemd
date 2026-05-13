// rq-0469400b

// Minimum-image wrap of a displacement component into [-L/2, +L/2).
__device__ static inline float min_image(float dx, float lx)
{
  return dx - lx * floorf((dx + lx * 0.5f) / lx);
}

// Per-axis cell index of a position. Wraps the position into the primary
// cell, clamps to [0, n_cells - 1] (handles the +L/2 boundary case).
__device__ static inline unsigned int cell_index_axis(
    float x, float lx, float cell_size, unsigned int n_cells)
{
  float wrapped = x - lx * floorf((x + lx * 0.5f) / lx);
  int idx = (int) floorf((wrapped + lx * 0.5f) / cell_size);
  if (idx < 0) {
    idx = 0;
  }
  if (idx >= (int) n_cells) {
    idx = (int) n_cells - 1;
  }
  return (unsigned int) idx;
}

// rq-884b5cd6
extern "C" __global__ void neighbor_displacement_squared(
    const float *positions_x, const float *positions_y, const float *positions_z,
    const float *reference_x, const float *reference_y, const float *reference_z,
    float lx, float ly, float lz,
    float *disp_sq,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  float dx = min_image(positions_x[i] - reference_x[i], lx);
  float dy = min_image(positions_y[i] - reference_y[i], ly);
  float dz = min_image(positions_z[i] - reference_z[i], lz);
  disp_sq[i] = dx * dx + dy * dy + dz * dz;
}

// rq-a1262872
extern "C" __global__ void neighbor_list_build(
    const float *positions_x, const float *positions_y, const float *positions_z,
    const unsigned int *sorted_particle_ids,
    const unsigned int *cell_offsets,
    float lx, float ly, float lz,
    float cell_size_x, float cell_size_y, float cell_size_z,
    unsigned int n_cells_x, unsigned int n_cells_y, unsigned int n_cells_z,
    float r_search_sq,
    unsigned int max_neighbors,
    unsigned int *neighbor_list,
    unsigned int *neighbor_counts,
    unsigned int *overflow_flag,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  float xi = positions_x[i];
  float yi = positions_y[i];
  float zi = positions_z[i];

  unsigned int cx = cell_index_axis(xi, lx, cell_size_x, n_cells_x);
  unsigned int cy = cell_index_axis(yi, ly, cell_size_y, n_cells_y);
  unsigned int cz = cell_index_axis(zi, lz, cell_size_z, n_cells_z);

  unsigned int *self_list = neighbor_list + (size_t) i * (size_t) max_neighbors;
  unsigned int count = 0;
  unsigned int overflowed = 0;

  // Walk 27 cells in a deterministic order: dx outer, dy middle, dz inner.
  for (int dxc = -1; dxc <= 1; ++dxc) {
    int ncx = (int) cx + dxc;
    while (ncx < 0) { ncx += (int) n_cells_x; }
    while (ncx >= (int) n_cells_x) { ncx -= (int) n_cells_x; }
    for (int dyc = -1; dyc <= 1; ++dyc) {
      int ncy = (int) cy + dyc;
      while (ncy < 0) { ncy += (int) n_cells_y; }
      while (ncy >= (int) n_cells_y) { ncy -= (int) n_cells_y; }
      for (int dzc = -1; dzc <= 1; ++dzc) {
        int ncz = (int) cz + dzc;
        while (ncz < 0) { ncz += (int) n_cells_z; }
        while (ncz >= (int) n_cells_z) { ncz -= (int) n_cells_z; }

        unsigned int c = ((unsigned int) ncx * n_cells_y + (unsigned int) ncy)
                         * n_cells_z + (unsigned int) ncz;
        unsigned int start = cell_offsets[c];
        unsigned int end = cell_offsets[c + 1];
        for (unsigned int s = start; s < end; ++s) {
          unsigned int j = sorted_particle_ids[s];
          if (j == i) {
            continue;
          }
          float ddx = min_image(xi - positions_x[j], lx);
          float ddy = min_image(yi - positions_y[j], ly);
          float ddz = min_image(zi - positions_z[j], lz);
          float r2 = ddx * ddx + ddy * ddy + ddz * ddz;
          if (r2 <= r_search_sq) {
            if (count < max_neighbors) {
              self_list[count] = j;
              count += 1;
            } else {
              overflowed = 1;
            }
          }
        }
      }
    }
  }

  // Insertion sort by partner index.
  for (unsigned int k = 1; k < count; ++k) {
    unsigned int key = self_list[k];
    int pos = (int) k - 1;
    while (pos >= 0 && self_list[pos] > key) {
      self_list[pos + 1] = self_list[pos];
      pos -= 1;
    }
    self_list[pos + 1] = key;
  }

  neighbor_counts[i] = count;
  if (overflowed) {
    atomicOr(overflow_flag, 1u);
  }
}

// rq-344f7af0
extern "C" __global__ void copy_positions_into_reference(
    const float *positions_x, const float *positions_y, const float *positions_z,
    float *reference_x, float *reference_y, float *reference_z,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  reference_x[i] = positions_x[i];
  reference_y[i] = positions_y[i];
  reference_z[i] = positions_z[i];
}

#define SCAN_BLOCK_SIZE 256u

extern "C" __global__ void compute_cell_indices_and_histogram(
    const float *positions_x, const float *positions_y, const float *positions_z,
    float lx, float ly, float lz,
    float cell_size_x, float cell_size_y, float cell_size_z,
    unsigned int n_cells_x, unsigned int n_cells_y, unsigned int n_cells_z,
    unsigned int *cell_indices,
    unsigned int *cell_counts,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  unsigned int cx = cell_index_axis(positions_x[i], lx, cell_size_x, n_cells_x);
  unsigned int cy = cell_index_axis(positions_y[i], ly, cell_size_y, n_cells_y);
  unsigned int cz = cell_index_axis(positions_z[i], lz, cell_size_z, n_cells_z);
  unsigned int c = (cx * n_cells_y + cy) * n_cells_z + cz;
  cell_indices[i] = c;
  atomicAdd(&cell_counts[c], 1u);
}

extern "C" __global__ void prefix_scan_local_blocks(
    const unsigned int *cell_counts,
    unsigned int *cell_offsets,
    unsigned int *scan_block_totals,
    unsigned int n_cells_total)
{
  __shared__ unsigned int temp[2u * SCAN_BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int gid = blockIdx.x * SCAN_BLOCK_SIZE + t;
  unsigned int my_input = (gid < n_cells_total) ? cell_counts[gid] : 0u;
  unsigned int pout = 0u;
  unsigned int pin = 1u;
  temp[pout * SCAN_BLOCK_SIZE + t] = my_input;
  __syncthreads();
  for (unsigned int offset = 1u; offset < SCAN_BLOCK_SIZE; offset *= 2u) {
    pout = 1u - pout;
    pin = 1u - pin;
    if (t >= offset) {
      temp[pout * SCAN_BLOCK_SIZE + t] =
          temp[pin * SCAN_BLOCK_SIZE + t]
          + temp[pin * SCAN_BLOCK_SIZE + t - offset];
    } else {
      temp[pout * SCAN_BLOCK_SIZE + t] = temp[pin * SCAN_BLOCK_SIZE + t];
    }
    __syncthreads();
  }
  unsigned int inclusive = temp[pout * SCAN_BLOCK_SIZE + t];
  unsigned int exclusive = inclusive - my_input;
  if (gid < n_cells_total) {
    cell_offsets[gid] = exclusive;
  }
  if (t == SCAN_BLOCK_SIZE - 1u) {
    scan_block_totals[blockIdx.x] = inclusive;
  }
}

extern "C" __global__ void prefix_scan_block_totals(
    unsigned int *scan_block_totals,
    unsigned int n_blocks)
{
  __shared__ unsigned int temp[2u * SCAN_BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int my_input = (t < n_blocks) ? scan_block_totals[t] : 0u;
  unsigned int pout = 0u;
  unsigned int pin = 1u;
  temp[pout * SCAN_BLOCK_SIZE + t] = my_input;
  __syncthreads();
  for (unsigned int offset = 1u; offset < SCAN_BLOCK_SIZE; offset *= 2u) {
    pout = 1u - pout;
    pin = 1u - pin;
    if (t >= offset) {
      temp[pout * SCAN_BLOCK_SIZE + t] =
          temp[pin * SCAN_BLOCK_SIZE + t]
          + temp[pin * SCAN_BLOCK_SIZE + t - offset];
    } else {
      temp[pout * SCAN_BLOCK_SIZE + t] = temp[pin * SCAN_BLOCK_SIZE + t];
    }
    __syncthreads();
  }
  unsigned int inclusive = temp[pout * SCAN_BLOCK_SIZE + t];
  unsigned int exclusive = inclusive - my_input;
  if (t < n_blocks) {
    scan_block_totals[t] = exclusive;
  }
}

extern "C" __global__ void prefix_scan_apply_block_totals(
    const unsigned int *scan_block_totals,
    unsigned int *cell_offsets,
    unsigned int n_cells_total,
    unsigned int particle_count)
{
  unsigned int t = threadIdx.x;
  unsigned int gid = blockIdx.x * SCAN_BLOCK_SIZE + t;
  if (gid < n_cells_total) {
    cell_offsets[gid] += scan_block_totals[blockIdx.x];
  }
  if (blockIdx.x == 0u && t == 0u) {
    cell_offsets[n_cells_total] = particle_count;
  }
}

extern "C" __global__ void scatter_atoms_into_cells(
    const unsigned int *cell_indices,
    const unsigned int *cell_offsets,
    unsigned int *write_cursors,
    unsigned int *sorted_particle_ids,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  unsigned int c = cell_indices[i];
  unsigned int slot = atomicAdd(&write_cursors[c], 1u);
  sorted_particle_ids[cell_offsets[c] + slot] = i;
}

extern "C" __global__ void sort_cells_by_particle_id(
    const unsigned int *cell_offsets,
    unsigned int *sorted_particle_ids,
    unsigned int n_cells_total)
{
  unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells_total) {
    return;
  }
  unsigned int start = cell_offsets[c];
  unsigned int end = cell_offsets[c + 1u];
  for (unsigned int k = start + 1u; k < end; ++k) {
    unsigned int key = sorted_particle_ids[k];
    int pos = (int) k - 1;
    while (pos >= (int) start && sorted_particle_ids[pos] > key) {
      sorted_particle_ids[pos + 1] = sorted_particle_ids[pos];
      pos -= 1;
    }
    sorted_particle_ids[pos + 1] = key;
  }
}
