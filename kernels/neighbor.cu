// rq-0469400b

#include "precision.cuh"

#include "pbc.cuh"

// Compute the parallelepiped cell index of a Cartesian position. Wraps
// the position into the primary image, transforms to fractional
// coordinates, and bins each fractional component to [0, n_cells_d - 1]
// (clamping handles the +0.5 boundary case).
__device__ static inline void parallelepiped_cell_indices(
    Real x, Real y, Real z,
    const Real *lattice,
    unsigned int n_cells_a, unsigned int n_cells_b, unsigned int n_cells_c,
    unsigned int &ca, unsigned int &cb, unsigned int &cc)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  int dummy_a, dummy_b, dummy_c;
  triclinic_wrap_with_image(x, y, z, dummy_a, dummy_b, dummy_c,
                            lx, ly, lz, xy, xz, yz);
  Real s_a, s_b, s_c;
  triclinic_cart_to_frac(x, y, z, lx, ly, lz, xy, xz, yz, s_a, s_b, s_c);
  int ia = (int) Real_floor((s_a + R(0.5)) * (Real) n_cells_a);
  int ib = (int) Real_floor((s_b + R(0.5)) * (Real) n_cells_b);
  int ic = (int) Real_floor((s_c + R(0.5)) * (Real) n_cells_c);
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
//
// Device-side displacement check. One thread per atom computes the
// minimum-image displacement from the atom's last-rebuild reference
// position, squares it, and (only when the squared length exceeds the
// host-supplied `threshold_sq = (r_skin / 2)²`) issues
// `atomicOr(disp_rebuild_flag, 1u)`. The flag is therefore set to `1u`
// as soon as any atom on any call exceeds the threshold and stays set
// until the host explicitly clears it via `cudaMemset`. No per-atom
// output buffer is written, so the kernel scales to N threads with a
// single one-word output. See
// `rqm/forces/neighbor-list.md` *Displacement Check*.
extern "C" __global__ void neighbor_displacement_check_flag(
    const Real4 *posq,
    const Real *reference_x, const Real *reference_y, const Real *reference_z,
    const Real *lattice,
    Real threshold_sq,
    unsigned int *neighbor_status,
    unsigned int n)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  Real4 pq = posq[i];
  Real dx = pq.x - reference_x[i];
  Real dy = pq.y - reference_y[i];
  Real dz = pq.z - reference_z[i];
  triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);
  Real d2 = dx * dx + dy * dy + dz * dz;
  if (d2 > threshold_sq) {
    // Bit 0 of the shared status word; bits 1-4 are owned by the
    // packed-neighbour construction (set_neighbor_status_bits).
    atomicOr(neighbor_status, 1u);
  }
}

// rq-344f7af0
extern "C" __global__ void copy_positions_into_reference(
    const Real4 *posq,
    Real *reference_x, Real *reference_y, Real *reference_z,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  Real4 pq = posq[i];
  reference_x[i] = pq.x;
  reference_y[i] = pq.y;
  reference_z[i] = pq.z;
}

#define SCAN_BLOCK_SIZE 256u

extern "C" __global__ void compute_cell_indices_and_histogram(
    const Real4 *posq,
    const Real *lattice,
    unsigned int n_cells_a, unsigned int n_cells_b, unsigned int n_cells_c,
    unsigned int *cell_indices,
    unsigned int *cell_counts,
    unsigned int n)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  unsigned int ca, cb, cc;
  Real4 pq = posq[i];
  parallelepiped_cell_indices(pq.x, pq.y, pq.z,
                              lattice,
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

// rq-67a09135
// Device-sourced variant of the sentinel write: `offsets[n] = count[0]`.
// Used by the i-block offset scan so the trailing total comes from the
// device-resident interaction count rather than a host-read value,
// keeping a steady-state rebuild free of any device-to-host copy.
extern "C" __global__ void prefix_scan_finalize_offsets_dev(
    unsigned int *offsets,
    unsigned int n,
    const unsigned int *count)
{
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    offsets[n] = count[0];
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

// =====================================================================
// Packed-neighbour pair-force architecture
// (rqm/forces/packed-neighbour-pair-force.md)
// =====================================================================

#define TILE_SIZE 32u

// Gathers per-particle posq into the tile-sorted view. One thread per
// atom; block size 256. For partial-block padding lanes (index >=
// particle_count), writes are out-of-range and so this kernel is
// launched only over [0, particle_count).
extern "C" __global__ void scatter_positions_to_tile_order(
    const Real4 *posq,
    const unsigned int *sorted_particle_ids,
    Real4 *tile_sorted_posq,
    unsigned int n)
{
  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  unsigned int pid = sorted_particle_ids[k];
  tile_sorted_posq[k] = posq[pid];
}

// Fills the partial-block padding lanes of the tile-sorted posq with
// +infinity (xyz) and zero (w) so the construction kernel and force
// kernel treat them as infinitely far from every other atom. Called
// once per build after scatter_positions_to_tile_order. One thread
// per padding lane.
extern "C" __global__ void fill_tile_position_padding(
    Real4 *tile_sorted_posq,
    unsigned int n,
    unsigned int padded_n)
{
  unsigned int k = n + blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= padded_n) return;
  Real pos_inf = (Real) 3.4e38;
  Real4 pad;
  pad.x = pos_inf;
  pad.y = pos_inf;
  pad.z = pos_inf;
  pad.w = R(0.0);
  tile_sorted_posq[k] = pad;
}

// Computes per-block axis-aligned bounding boxes. One warp per block.
// block_centre[b] holds the centre (x, y, z) and the maximum
// atom-to-centre distance squared in .w. block_bbox[b] holds the
// per-axis half-extents.
//
// Layout: block_centre is 4 Reals per block (cx, cy, cz, max_disp_sq).
// block_bbox is 3 Reals per block (dx, dy, dz).
extern "C" __global__ void compute_block_bbox(
    const Real4 *tile_sorted_posq,
    const unsigned int *tile_atom_count,
    Real *block_centre,
    Real *block_bbox,
    unsigned int n_blocks)
{
  unsigned int warp_in_block = threadIdx.x / 32u;
  unsigned int lane = threadIdx.x & 31u;
  unsigned int b = blockIdx.x * (blockDim.x / 32u) + warp_in_block;
  if (b >= n_blocks) return;

  unsigned int count = tile_atom_count[b];
  bool active = lane < count;

  unsigned int idx = b * TILE_SIZE + lane;
  Real pos_inf = (Real) 3.4e38;
  Real neg_inf = -pos_inf;
  Real4 pq_lane = active ? tile_sorted_posq[idx] : (Real4){pos_inf, pos_inf, pos_inf, R(0.0)};
  Real px = pq_lane.x;
  Real py = pq_lane.y;
  Real pz = pq_lane.z;
  Real qx = active ? pq_lane.x : neg_inf;
  Real qy = active ? pq_lane.y : neg_inf;
  Real qz = active ? pq_lane.z : neg_inf;

  for (unsigned int off = 16u; off > 0u; off >>= 1) {
    Real ox = __shfl_xor_sync(0xFFFFFFFFu, px, off);
    Real oy = __shfl_xor_sync(0xFFFFFFFFu, py, off);
    Real oz = __shfl_xor_sync(0xFFFFFFFFu, pz, off);
    if (ox < px) px = ox;
    if (oy < py) py = oy;
    if (oz < pz) pz = oz;
    Real mx = __shfl_xor_sync(0xFFFFFFFFu, qx, off);
    Real my = __shfl_xor_sync(0xFFFFFFFFu, qy, off);
    Real mz = __shfl_xor_sync(0xFFFFFFFFu, qz, off);
    if (mx > qx) qx = mx;
    if (my > qy) qy = my;
    if (mz > qz) qz = mz;
  }

  Real cx = R(0.5) * (px + qx);
  Real cy = R(0.5) * (py + qy);
  Real cz = R(0.5) * (pz + qz);
  Real dx = R(0.5) * (qx - px);
  Real dy = R(0.5) * (qy - py);
  Real dz = R(0.5) * (qz - pz);

  // Compute max atom-to-centre distance squared.
  Real disp_sq = R(0.0);
  if (active) {
    Real rx = pq_lane.x - cx;
    Real ry = pq_lane.y - cy;
    Real rz = pq_lane.z - cz;
    disp_sq = rx * rx + ry * ry + rz * rz;
  }
  for (unsigned int off = 16u; off > 0u; off >>= 1) {
    Real o = __shfl_xor_sync(0xFFFFFFFFu, disp_sq, off);
    if (o > disp_sq) disp_sq = o;
  }

  if (lane == 0u) {
    block_centre[b * 4u + 0u] = cx;
    block_centre[b * 4u + 1u] = cy;
    block_centre[b * 4u + 2u] = cz;
    block_centre[b * 4u + 3u] = disp_sq;
    block_bbox[b * 3u + 0u] = dx;
    block_bbox[b * 3u + 1u] = dy;
    block_bbox[b * 3u + 2u] = dz;
  }
}

// Closest-approach squared distance between two AABBs under minimum
// image, with bounding-sphere widening via the .w terms.
__device__ static inline Real packed_block_bbox_dist_sq(
    Real cx_a, Real cy_a, Real cz_a, Real rsq_a,
    Real dx_a, Real dy_a, Real dz_a,
    Real cx_b, Real cy_b, Real cz_b, Real rsq_b,
    Real dx_b, Real dy_b, Real dz_b,
    const Real *lattice)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  Real dx = cx_a - cx_b;
  Real dy = cy_a - cy_b;
  Real dz = cz_a - cz_b;
  triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);
  Real hx = dx_a + dx_b;
  Real hy = dy_a + dy_b;
  Real hz = dz_a + dz_b;
  Real ex = Real_fabs(dx) - hx; if (ex < R(0.0)) ex = R(0.0);
  Real ey = Real_fabs(dy) - hy; if (ey < R(0.0)) ey = R(0.0);
  Real ez = Real_fabs(dz) - hz; if (ez < R(0.0)) ez = R(0.0);
  return ex * ex + ey * ey + ez * ez;
  (void) rsq_a; (void) rsq_b;
}

// Construction kernel: for each i-block, sweep candidate j-blocks via
// AABB pre-filter, then for surviving candidates do per-atom
// refinement. Pack interacting j-atoms into per-warp buffer; flush 32
// at a time to interacting_atoms / interacting_tiles. Each entry
// stores 32 INDIVIDUAL j-atom IDs (original atom IDs), drawn from
// possibly different j-blocks.
//
// One warp per i-block. The warp iterates j_block from i_block (so
// self-block is included; the force kernel skips the diagonal
// self-pair by atom ID).
//
// Shared memory: per-warp staging buffer of 64 atom IDs (BUFFER_SIZE),
// plus the warp's loaded i-block positions for the per-atom refine.

#define PACKED_NL_WARPS_PER_BLOCK 4
#define PACKED_NL_BLOCK_SIZE (PACKED_NL_WARPS_PER_BLOCK * 32)

// rq-bce26a14
// Compile-time threshold for sparse-tile single-pair extraction.
// When a candidate (i-block, j-block) pair produces
// <= MAX_BITS_FOR_PAIRS j-atom hits, every (i_atom, j_atom) hit
// is written individually to single_pair_atoms instead of being
// emitted as a mostly-empty packed entry.
#ifndef MAX_BITS_FOR_PAIRS
#define MAX_BITS_FOR_PAIRS 16
#endif

// rq-a4c22484 rq-560d3be9 rq-c37263fd rq-e1bd63f5
// Each dense (i-block, j-block) tile-pair produces exactly one packed
// entry. The entry carries `interacting_tiles[slot] = i_block` and
// `interacting_j_blocks[slot] = j_block`; all 32 j-atom slots come
// from the single j-block, with lanes that lack `any_hit` (or that
// index past `n_atoms`) padded with the sentinel `n_atoms`. No
// staging buffer is used, no cross-tile-pair aggregation happens,
// and no end-of-sweep tail flush is issued.
extern "C" __global__ void find_blocks_with_interactions(
    const Real4 *tile_sorted_posq,
    const unsigned int *sorted_particle_ids,
    const Real *block_centre,
    const Real *block_bbox,
    const Real *lattice,
    Real r_search_sq,
    unsigned int n_blocks,
    unsigned int n_atoms,
    unsigned int max_entries,
    unsigned int max_single_pairs,
    unsigned int *interacting_tiles,
    unsigned int *interacting_j_blocks,
    unsigned int *interacting_atoms,
    unsigned int *single_pair_atoms,
    unsigned int *interaction_count)
{
  __shared__ Real warp_ix[PACKED_NL_WARPS_PER_BLOCK][TILE_SIZE];
  __shared__ Real warp_iy[PACKED_NL_WARPS_PER_BLOCK][TILE_SIZE];
  __shared__ Real warp_iz[PACKED_NL_WARPS_PER_BLOCK][TILE_SIZE];
  __shared__ unsigned int warp_iid[PACKED_NL_WARPS_PER_BLOCK][TILE_SIZE];

  unsigned int warp_in_block = threadIdx.x / 32u;
  unsigned int lane = threadIdx.x & 31u;
  unsigned int b = blockIdx.x * PACKED_NL_WARPS_PER_BLOCK + warp_in_block;
  if (b >= n_blocks) return;

  // Load i-block atom positions + original IDs into shared (per-warp).
  // sorted_particle_ids is sized to n_atoms, so gate the read for
  // partial-block padding lanes (b * 32 + lane >= n_atoms).
  unsigned int i_slot = b * TILE_SIZE + lane;
  bool i_in_range = i_slot < n_atoms;
  Real4 pq_i = tile_sorted_posq[i_slot];
  Real ix = pq_i.x;
  Real iy = pq_i.y;
  Real iz = pq_i.z;
  unsigned int iid = i_in_range ? sorted_particle_ids[i_slot] : n_atoms;
  warp_ix[warp_in_block][lane] = ix;
  warp_iy[warp_in_block][lane] = iy;
  warp_iz[warp_in_block][lane] = iz;
  warp_iid[warp_in_block][lane] = iid;
  __syncwarp(0xFFFFFFFFu);

  Real cx_i = block_centre[b * 4u + 0u];
  Real cy_i = block_centre[b * 4u + 1u];
  Real cz_i = block_centre[b * 4u + 2u];
  Real rsq_i = block_centre[b * 4u + 3u];
  Real dx_i = block_bbox[b * 3u + 0u];
  Real dy_i = block_bbox[b * 3u + 1u];
  Real dz_i = block_bbox[b * 3u + 2u];

  // Iterate candidate j-blocks. j >= b so each unordered pair is
  // counted once; self-block (b, b) is included.
  for (unsigned int j_base = b; j_base < n_blocks; j_base += 32u) {
    unsigned int j_block = j_base + lane;
    bool j_in_range = j_block < n_blocks;
    bool prune_pass = false;
    if (j_in_range) {
      Real cx_j = block_centre[j_block * 4u + 0u];
      Real cy_j = block_centre[j_block * 4u + 1u];
      Real cz_j = block_centre[j_block * 4u + 2u];
      Real rsq_j = block_centre[j_block * 4u + 3u];
      Real dx_j = block_bbox[j_block * 3u + 0u];
      Real dy_j = block_bbox[j_block * 3u + 1u];
      Real dz_j = block_bbox[j_block * 3u + 2u];
      Real d2 = packed_block_bbox_dist_sq(
          cx_i, cy_i, cz_i, rsq_i, dx_i, dy_i, dz_i,
          cx_j, cy_j, cz_j, rsq_j, dx_j, dy_j, dz_j,
          lattice);
      prune_pass = d2 <= r_search_sq;
    }
    unsigned int prune_ballot = __ballot_sync(0xFFFFFFFFu, prune_pass ? 1u : 0u);

    // For each candidate j-block that passed the bbox prune, do
    // per-atom refinement. Process them sequentially via __ffs.
    while (prune_ballot != 0u) {
      unsigned int bit_pos = (unsigned int) __ffs((int) prune_ballot) - 1u;
      prune_ballot &= prune_ballot - 1u;
      unsigned int jb = j_base + bit_pos;

      // Load j-block's atoms (one per lane).
      unsigned int j_slot = jb * TILE_SIZE + lane;
      bool j_in_range = j_slot < n_atoms;
      Real4 pq_j = tile_sorted_posq[j_slot];
      Real jx = pq_j.x;
      Real jy = pq_j.y;
      Real jz = pq_j.z;
      unsigned int jid = j_in_range ? sorted_particle_ids[j_slot] : n_atoms;

      // Test j-atom (this lane's) against all 32 i-atoms via lane sweep.
      // Per lane: bit `m` of `i_hit_mask` is set when i-atom `m` is in
      // range of this lane's j-atom (and is a distinct atom).
      Real lx_ = lattice[0]; Real ly_ = lattice[1]; Real lz_ = lattice[2];
      Real xy_ = lattice[3]; Real xz_ = lattice[4]; Real yz_ = lattice[5];
      unsigned int i_hit_mask = 0u;
      for (unsigned int m = 0u; m < TILE_SIZE; ++m) {
        Real ax = warp_ix[warp_in_block][m];
        Real ay = warp_iy[warp_in_block][m];
        Real az = warp_iz[warp_in_block][m];
        Real ddx = ax - jx;
        Real ddy = ay - jy;
        Real ddz = az - jz;
        triclinic_min_image(ddx, ddy, ddz, lx_, ly_, lz_, xy_, xz_, yz_);
        Real r2 = ddx * ddx + ddy * ddy + ddz * ddz;
        // Skip self-pair (same original atom ID).
        unsigned int aid = warp_iid[warp_in_block][m];
        if (aid != jid && r2 <= r_search_sq) {
          i_hit_mask |= (1u << m);
        }
      }
      bool any_hit = (i_hit_mask != 0u);
      unsigned int hit_ballot = __ballot_sync(0xFFFFFFFFu, any_hit ? 1u : 0u);
      unsigned int hits = __popc(hit_ballot);

      if (hits > 0u && hits <= MAX_BITS_FOR_PAIRS) {
        // Sparse-tile path: emit one entry per (i_atom, j_atom) hit to
        // single_pair_atoms. Each lane with any_hit iterates the set
        // bits of its i_hit_mask; for each set bit, atomically claim a
        // slot and write the canonical (i, j) atom IDs.
        //
        // For self-block tile-pairs (jb == b), the pair (a, b) with a, b
        // both in the block is discovered twice — once from lane b's
        // sweep of `i_hit_mask` (finding a) and once from lane a's sweep
        // (finding b). Emit only when `aid < jid` so the pair reaches
        // single_pair_atoms exactly once. Non-self-block tile-pairs are
        // naturally single-emit because the outer loop iterates
        // `j_block >= b`.
        bool self_block_tile = (jb == b);
        if (any_hit) {
          unsigned int local_mask = i_hit_mask;
          while (local_mask != 0u) {
            unsigned int m = (unsigned int) __ffs((int) local_mask) - 1u;
            local_mask &= local_mask - 1u;
            unsigned int aid = warp_iid[warp_in_block][m];
            if (self_block_tile && aid >= jid) {
              continue;
            }
            unsigned int slot = atomicAdd(&interaction_count[1], 1u);
            // interaction_count[1] accumulates the true required count even
            // past capacity; entries beyond capacity are not written. The
            // host learns of the overflow from set_neighbor_status_bits.
            if (slot < max_single_pairs) {
              single_pair_atoms[2u * slot] = aid;
              single_pair_atoms[2u * slot + 1u] = jid;
            }
          }
        }
      } else if (hits > MAX_BITS_FOR_PAIRS) {
        // Dense-tile path: emit exactly one packed entry per
        // tile-pair. Every lane's j-atom (with `any_hit` and
        // `j_slot < n_atoms`) becomes a real j-atom of the entry at
        // its own lane index; lanes that don't hit or that index
        // past `n_atoms` write the sentinel. Newton's-3rd is
        // handled by the pair-force kernel via the entry's
        // warp-uniform `self_block = (j_block == i_block)` gate.
        // rq-560d3be9 rq-a4c22484 rq-c37263fd
        unsigned int slot;
        if (lane == 0u) {
          slot = atomicAdd(&interaction_count[0], 1u);
        }
        slot = __shfl_sync(0xFFFFFFFFu, slot, 0);
        // `interaction_count[0]` accumulates the true required count
        // even past capacity; entries beyond capacity are not written
        // and the host learns about the overflow from
        // set_neighbor_status_bits.
        if (slot < max_entries) {
          if (lane == 0u) {
            interacting_tiles[slot] = b;
            interacting_j_blocks[slot] = jb;
          }
          unsigned int emit;
          if (any_hit && j_slot < n_atoms) {
            emit = jid;
          } else {
            emit = n_atoms;
          }
          interacting_atoms[slot * TILE_SIZE + lane] = emit;
        }
      }
      __syncwarp(0xFFFFFFFFu);
    }
  }

  // rq-e1bd63f5 — no tail flush: every dense tile-pair has already
  // emitted its own entry directly. There is no per-warp staging
  // buffer to drain at the end of the j_base sweep.
}
#undef MAX_BITS_FOR_PAIRS

// rq-67a09135 rq-0acba2a0
// Single designated thread compares the live interaction counts against
// their capacities and high-water marks and sets bits 1-4 of the shared
// `neighbor_status` word via atomicOr (bit 0 is owned by the
// displacement-check kernel). Bits: 1 = tiles_high_water,
// 2 = single_pairs_high_water, 3 = tiles_overflow,
// 4 = single_pairs_overflow. Counts are read device-side; nothing is
// copied to the host. See `rqm/forces/packed-neighbour-pair-force.md`
// *Capacity*.
extern "C" __global__ void set_neighbor_status_bits(
    const unsigned int *interaction_count,
    unsigned int tiles_capacity,
    unsigned int single_pairs_capacity,
    unsigned int tiles_high_water_mark,
    unsigned int single_pairs_high_water_mark,
    unsigned int *neighbor_status)
{
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    unsigned int c0 = interaction_count[0];
    unsigned int c1 = interaction_count[1];
    unsigned int bits = 0u;
    if (c0 > tiles_capacity) {
      bits |= (1u << 3);
    } else if (c0 > tiles_high_water_mark) {
      bits |= (1u << 1);
    }
    if (c1 > single_pairs_capacity) {
      bits |= (1u << 4);
    } else if (c1 > single_pairs_high_water_mark) {
      bits |= (1u << 2);
    }
    if (bits != 0u) {
      atomicOr(neighbor_status, bits);
    }
  }
}

// Histogram entries by i-block. For each entry e in 0..entry_count,
// reads interacting_tiles[e] and increments the corresponding
// counter via atomicAdd. One thread per entry; the work is small but
// embarrassingly parallel.
extern "C" __global__ void histogram_entries_by_iblock(
    const unsigned int *interacting_tiles,      // length = entry_count
    const unsigned int *entry_count_ptr,        // length 1; read once
    unsigned int *iblock_count,                 // length = n_blocks
    unsigned int n_blocks)
{
  unsigned int e = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int entry_count = *entry_count_ptr;
  if (e >= entry_count) return;
  unsigned int b = interacting_tiles[e];
  if (b < n_blocks) {
    atomicAdd(&iblock_count[b], 1u);
  }
}

// Scatter entries from the unordered (interacting_tiles,
// interacting_j_blocks, interacting_atoms) layout into the i-block-
// sorted layout. For each entry e, claims a destination slot inside
// its i-block's contiguous range via
// `atomicAdd(&iblock_cursor[b], 1) + iblock_offset[b]`, then copies
// the entry's j-block index into `sorted_interacting_j_blocks` and
// the 32 packed j-atom IDs into `sorted_interacting_atoms`. One warp
// per entry: lane k copies `interacting_atoms[e*32+k]` into the
// destination row; lane 0 additionally writes the entry's j-block
// index. The within-i-block entry order is unstable (atomic-claimed
// slots below). Force-kernel determinism does not depend on it: the
// pair kernel folds each i-atom's per-entry contributions into a
// warp-resident i64 fixed-point accumulator, and integer addition is
// associative regardless of entry order. See
// rqm/forces/jit-composed-pair-force.md (rq-693544f8).
extern "C" __global__ void scatter_entries_by_iblock(
    const unsigned int *interacting_tiles,           // length = entry_count
    const unsigned int *interacting_j_blocks,        // length = entry_count
    const unsigned int *interacting_atoms,           // length = entry_count * 32
    const unsigned int *entry_count_ptr,             // length 1
    const unsigned int *iblock_offset,               // length = n_blocks + 1
    unsigned int *iblock_cursor,                     // length = n_blocks; init zero
    unsigned int *sorted_interacting_j_blocks,       // length = entry_count
    unsigned int *sorted_interacting_atoms,          // length = entry_count * 32
    unsigned int n_blocks)
{
  unsigned int e = blockIdx.x * (blockDim.x / 32u) + (threadIdx.x / 32u);
  unsigned int lane = threadIdx.x & 31u;
  unsigned int entry_count = *entry_count_ptr;
  if (e >= entry_count) return;
  unsigned int b = interacting_tiles[e];
  if (b >= n_blocks) return;
  unsigned int slot;
  if (lane == 0u) {
    slot = atomicAdd(&iblock_cursor[b], 1u) + iblock_offset[b];
    sorted_interacting_j_blocks[slot] = interacting_j_blocks[e];
  }
  slot = __shfl_sync(0xFFFFFFFFu, slot, 0);
  sorted_interacting_atoms[slot * 32u + lane] =
      interacting_atoms[e * 32u + lane];
}

// Converts a fixed-point sum back to Real and writes it into a slot
// of an output buffer. One thread per atom.
extern "C" __global__ void finalize_packed_forces(
    const unsigned long long *fp_fx,
    const unsigned long long *fp_fy,
    const unsigned long long *fp_fz,
    const unsigned long long *fp_e,
    const unsigned long long *fp_w,
    Real *out_fx,
    Real *out_fy,
    Real *out_fz,
    Real *out_e,
    Real *out_w,
    unsigned int n,
    unsigned int write_ev)
{
  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  // Scale 2^48 — must match heddle_jit_real_to_fixed in
  // src/forces/jit_composed.rs. Built by two multiplications to stay
  // numerically stable in f32 intermediates.
  const Real inv_scale = (R(1.0) / (Real) (1u << 24)) / (Real) (1u << 24);
  long long sfx = (long long) fp_fx[k];
  long long sfy = (long long) fp_fy[k];
  long long sfz = (long long) fp_fz[k];
  out_fx[k] += (Real) sfx * inv_scale;
  out_fy[k] += (Real) sfy * inv_scale;
  out_fz[k] += (Real) sfz * inv_scale;
  if (write_ev) {
    long long se = (long long) fp_e[k];
    long long sw = (long long) fp_w[k];
    out_e[k] += (Real) se * inv_scale;
    out_w[k] += (Real) sw * inv_scale;
  }
}
