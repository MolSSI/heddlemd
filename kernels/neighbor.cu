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
//
// One block per home cell. The block's threads cooperate so each of the
// 27 neighbour cells' positions is loaded from global memory exactly
// once per block (tiled through dynamic shared memory in chunks of
// blockDim.x candidates) and amortised across all home-cell atoms. Each
// thread owns one home-cell atom's neighbour list and walks the
// shared-memory candidates in cell-sweep order — `(da, db, dc)` lex
// outer-to-inner, particle-ID ascending within each cell. No trailing
// per-atom sort.
//
// Dynamic shared memory layout (in bytes), set at launch:
//   shared_x : float[blockDim.x]
//   shared_y : float[blockDim.x]
//   shared_z : float[blockDim.x]
//   shared_id: unsigned int[blockDim.x]
// Total = 4 * blockDim.x * sizeof(float).
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
  (void) n;

  extern __shared__ unsigned char smem[];
  float *shared_x = reinterpret_cast<float *>(smem);
  float *shared_y = shared_x + blockDim.x;
  float *shared_z = shared_y + blockDim.x;
  unsigned int *shared_id =
      reinterpret_cast<unsigned int *>(shared_z + blockDim.x);

  unsigned int home_cell = blockIdx.x;
  unsigned int total_cells = n_cells_a * n_cells_b * n_cells_c;
  if (home_cell >= total_cells) {
    return;
  }

  unsigned int home_start = cell_offsets[home_cell];
  unsigned int home_end = cell_offsets[home_cell + 1u];
  unsigned int home_count = home_end - home_start;
  if (home_count == 0u) {
    return;
  }

  // Decode home cell into (ca, cb, cc) using the same row-major
  // convention as the cell-index/histogram kernel:
  //   home_cell = (ca * n_cells_b + cb) * n_cells_c + cc
  unsigned int ca = home_cell / (n_cells_b * n_cells_c);
  unsigned int rem = home_cell - ca * (n_cells_b * n_cells_c);
  unsigned int cb = rem / n_cells_c;
  unsigned int cc = rem - cb * n_cells_c;

  // Iterate home-cell atoms in chunks of blockDim.x so a single block
  // can service arbitrarily dense cells.
  for (unsigned int home_off = 0u; home_off < home_count;
       home_off += blockDim.x) {
    unsigned int thread_atom = home_off + threadIdx.x;
    bool active = (thread_atom < home_count);

    unsigned int i = 0u;
    float xi = 0.0f, yi = 0.0f, zi = 0.0f;
    if (active) {
      i = sorted_particle_ids[home_start + thread_atom];
      xi = positions_x[i];
      yi = positions_y[i];
      zi = positions_z[i];
    }
    unsigned int count = 0u;
    unsigned int overflowed = 0u;

    // 27-cell sweep: a outer, b middle, c inner.
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

          unsigned int c_neigh =
              ((unsigned int) nca * n_cells_b + (unsigned int) ncb)
              * n_cells_c + (unsigned int) ncc;
          unsigned int n_start = cell_offsets[c_neigh];
          unsigned int n_end = cell_offsets[c_neigh + 1u];

          // Stream candidates through shared memory in chunks. Each
          // chunk fits exactly one shared-memory tile of blockDim.x
          // candidates; cells with > blockDim.x atoms span multiple
          // chunks. The outer-to-inner cell order and the in-cell
          // sorted_particle_ids order together pin the neighbour append
          // order.
          for (unsigned int chunk_base = n_start; chunk_base < n_end;
               chunk_base += blockDim.x) {
            unsigned int chunk_size = n_end - chunk_base;
            if (chunk_size > blockDim.x) {
              chunk_size = blockDim.x;
            }

            __syncthreads();
            if (threadIdx.x < chunk_size) {
              unsigned int j = sorted_particle_ids[chunk_base + threadIdx.x];
              shared_id[threadIdx.x] = j;
              shared_x[threadIdx.x] = positions_x[j];
              shared_y[threadIdx.x] = positions_y[j];
              shared_z[threadIdx.x] = positions_z[j];
            }
            __syncthreads();

            if (active) {
              for (unsigned int k = 0u; k < chunk_size; ++k) {
                unsigned int j = shared_id[k];
                if (j == i) {
                  continue;
                }
                float ddx = xi - shared_x[k];
                float ddy = yi - shared_y[k];
                float ddz = zi - shared_z[k];
                triclinic_min_image(ddx, ddy, ddz,
                                    lx, ly, lz, xy, xz, yz);
                float r2 = ddx * ddx + ddy * ddy + ddz * ddz;
                if (r2 <= r_search_sq) {
                  if (count < max_neighbors) {
                    neighbor_list[(size_t) i * (size_t) max_neighbors
                                  + count] = j;
                    count += 1u;
                  } else {
                    overflowed = 1u;
                  }
                }
              }
            }
          }
        }
      }
    }

    if (active) {
      neighbor_counts[i] = count;
      if (overflowed) {
        atomicOr(overflow_flag, 1u);
      }
    }

    // Barrier before the next home_off iteration overwrites shared
    // memory while another wave of threads might still be reading it.
    __syncthreads();
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
