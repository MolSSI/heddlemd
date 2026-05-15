// rq-0469400b

#include "pbc.cuh"

// Compute the parallelepiped cell index of a Cartesian position. Wraps
// the position into the primary image, transforms to fractional
// coordinates, and bins each fractional component to [0, n_cells_d - 1]
// (clamping handles the +0.5 boundary case).
__device__ static inline void parallelepiped_cell_indices(
    float x, float y, float z,
    float lx, float ly, float lz, float xy, float xz, float yz,
    unsigned int n_cells_a, unsigned int n_cells_b, unsigned int n_cells_c,
    unsigned int &ca, unsigned int &cb, unsigned int &cc)
{
  int dummy_a, dummy_b, dummy_c;
  triclinic_wrap_with_image(x, y, z, dummy_a, dummy_b, dummy_c,
                            lx, ly, lz, xy, xz, yz);
  float s_a, s_b, s_c;
  triclinic_cart_to_frac(x, y, z, lx, ly, lz, xy, xz, yz, s_a, s_b, s_c);
  int ia = (int) floorf((s_a + 0.5f) * (float) n_cells_a);
  int ib = (int) floorf((s_b + 0.5f) * (float) n_cells_b);
  int ic = (int) floorf((s_c + 0.5f) * (float) n_cells_c);
  if (ia < 0) ia = 0;
  if (ia >= (int) n_cells_a) ia = (int) n_cells_a - 1;
  if (ib < 0) ib = 0;
  if (ib >= (int) n_cells_b) ib = (int) n_cells_b - 1;
  if (ic < 0) ic = 0;
  if (ic >= (int) n_cells_c) ic = (int) n_cells_c - 1;
  ca = (unsigned int) ia;
  cb = (unsigned int) ib;
  cc = (unsigned int) ic;
}

// rq-884b5cd6
extern "C" __global__ void neighbor_displacement_squared(
    const float *positions_x, const float *positions_y, const float *positions_z,
    const float *reference_x, const float *reference_y, const float *reference_z,
    float lx, float ly, float lz, float xy, float xz, float yz,
    float *disp_sq,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  float dx = positions_x[i] - reference_x[i];
  float dy = positions_y[i] - reference_y[i];
  float dz = positions_z[i] - reference_z[i];
  triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);
  disp_sq[i] = dx * dx + dy * dy + dz * dz;
}

// rq-a1262872
extern "C" __global__ void neighbor_list_build(
    const float *positions_x, const float *positions_y, const float *positions_z,
    const unsigned int *sorted_particle_ids,
    const unsigned int *cell_offsets,
    float lx, float ly, float lz, float xy, float xz, float yz,
    unsigned int n_cells_a, unsigned int n_cells_b, unsigned int n_cells_c,
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

  unsigned int ca, cb, cc;
  parallelepiped_cell_indices(xi, yi, zi,
                              lx, ly, lz, xy, xz, yz,
                              n_cells_a, n_cells_b, n_cells_c,
                              ca, cb, cc);

  unsigned int *self_list = neighbor_list + (size_t) i * (size_t) max_neighbors;
  unsigned int count = 0;
  unsigned int overflowed = 0;

  // Walk 27 cells in a deterministic order: a outer, b middle, c inner.
  for (int da = -1; da <= 1; ++da) {
    int nca = (int) ca + da;
    while (nca < 0) { nca += (int) n_cells_a; }
    while (nca >= (int) n_cells_a) { nca -= (int) n_cells_a; }
    for (int db = -1; db <= 1; ++db) {
      int ncb = (int) cb + db;
      while (ncb < 0) { ncb += (int) n_cells_b; }
      while (ncb >= (int) n_cells_b) { ncb -= (int) n_cells_b; }
      for (int dc = -1; dc <= 1; ++dc) {
        int ncc = (int) cc + dc;
        while (ncc < 0) { ncc += (int) n_cells_c; }
        while (ncc >= (int) n_cells_c) { ncc -= (int) n_cells_c; }

        unsigned int c = ((unsigned int) nca * n_cells_b + (unsigned int) ncb)
                         * n_cells_c + (unsigned int) ncc;
        unsigned int start = cell_offsets[c];
        unsigned int end = cell_offsets[c + 1];
        for (unsigned int s = start; s < end; ++s) {
          unsigned int j = sorted_particle_ids[s];
          if (j == i) {
            continue;
          }
          float ddx = xi - positions_x[j];
          float ddy = yi - positions_y[j];
          float ddz = zi - positions_z[j];
          triclinic_min_image(ddx, ddy, ddz, lx, ly, lz, xy, xz, yz);
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
    float lx, float ly, float lz, float xy, float xz, float yz,
    unsigned int n_cells_a, unsigned int n_cells_b, unsigned int n_cells_c,
    unsigned int *cell_indices,
    unsigned int *cell_counts,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  unsigned int ca, cb, cc;
  parallelepiped_cell_indices(positions_x[i], positions_y[i], positions_z[i],
                              lx, ly, lz, xy, xz, yz,
                              n_cells_a, n_cells_b, n_cells_c,
                              ca, cb, cc);
  unsigned int c = (ca * n_cells_b + cb) * n_cells_c + cc;
  cell_indices[i] = c;
  atomicAdd(&cell_counts[c], 1u);
}

// Per-block exclusive Hillis-Steele scan of input[0..len] into
// output[0..len], writing each block's inclusive total to
// block_totals[blockIdx]. Each thread reads its input element into a
// register before any global write, and blocks write disjoint output
// ranges, so `input` may alias `output` — the recursive scan driver
// relies on this to scan each block-totals level of the stack in place.
extern "C" __global__ void prefix_scan_local_blocks(
    const unsigned int *input,
    unsigned int *output,
    unsigned int *block_totals,
    unsigned int len)
{
  __shared__ unsigned int temp[2u * SCAN_BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int gid = blockIdx.x * SCAN_BLOCK_SIZE + t;
  unsigned int my_input = (gid < len) ? input[gid] : 0u;
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
  if (gid < len) {
    output[gid] = exclusive;
  }
  if (t == SCAN_BLOCK_SIZE - 1u) {
    block_totals[blockIdx.x] = inclusive;
  }
}

// Generic add-back: output[gid] += block_offsets[gid / SCAN_BLOCK_SIZE]
// for every gid < len.
extern "C" __global__ void prefix_scan_apply_block_totals(
    const unsigned int *block_offsets,
    unsigned int *output,
    unsigned int len)
{
  unsigned int gid = blockIdx.x * SCAN_BLOCK_SIZE + threadIdx.x;
  if (gid < len) {
    output[gid] += block_offsets[blockIdx.x];
  }
}

// Writes the trailing cell_offsets[n_cells_total] = particle_count
// sentinel slot with a single thread.
extern "C" __global__ void prefix_scan_finalize_offsets(
    unsigned int *cell_offsets,
    unsigned int n_cells_total,
    unsigned int particle_count)
{
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
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
