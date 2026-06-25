use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};

use crate::gpu::{GpuContext, GpuError, Kernels};
use crate::precision::{Real, Real4};
use crate::state::{ParticleState, ParticleStateError, check_len};

// rq-4a8de06c
#[derive(Debug)]
pub struct ParticleBuffers {
    pub device: Arc<CudaDevice>,
    pub kernels: Arc<Kernels>,
    /// Interleaved positions + charges. `posq[i].x/.y/.z` carry the
    /// wrapped position of particle `i`; `posq[i].w` carries its
    /// charge. One coalesced 16- or 32-byte load (depending on the
    /// precision feature) per warp per 32 atoms.
    pub posq: CudaSlice<Real4>,
    pub images_x: CudaSlice<i32>,
    pub images_y: CudaSlice<i32>,
    pub images_z: CudaSlice<i32>,
    pub velocities_x: CudaSlice<Real>,
    pub velocities_y: CudaSlice<Real>,
    pub velocities_z: CudaSlice<Real>,
    pub forces_x: CudaSlice<Real>,
    pub forces_y: CudaSlice<Real>,
    pub forces_z: CudaSlice<Real>,
    pub potential_energies: CudaSlice<Real>,
    pub virials: CudaSlice<Real>,
    pub masses: CudaSlice<Real>,
    pub type_indices: CudaSlice<u32>,
    pub particle_ids: CudaSlice<u32>,
    /// Fixed-length scratch holding one partial sum per block for the
    /// deterministic multi-block scalar reductions (kinetic energy,
    /// virial, potential energy) on the large-`N` path. Length
    /// `REDUCE_PARTIAL_BLOCKS`. See `rqm/integration/nose-hoover-chain.md`.
    /// rq-1727d6bd
    pub reduction_partials: CudaSlice<Real>,
}

/// Number of blocks in pass 1 of the multi-block reduction (and the
/// length of `ParticleBuffers::reduction_partials`). Fixed so the launch
/// dimensions are constant for CUDA-graph capture. rq-1727d6bd
pub const REDUCE_PARTIAL_BLOCKS: u32 = 1024;

/// Build a host-side `Vec<Real4>` of length `n` by interleaving the
/// four scalar SoA arrays.
fn interleave_posq(
    positions_x: &[Real],
    positions_y: &[Real],
    positions_z: &[Real],
    charges: &[Real],
) -> Vec<Real4> {
    let n = positions_x.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(Real4 {
            x: positions_x[i],
            y: positions_y[i],
            z: positions_z[i],
            w: charges[i],
        });
    }
    out
}

/// Split a host-side `Vec<Real4>` into four scalar SoA arrays in
/// place.
pub fn split_posq_into(
    src: &[Real4],
    positions_x: &mut [Real],
    positions_y: &mut [Real],
    positions_z: &mut [Real],
    charges: &mut [Real],
) {
    for (i, p) in src.iter().enumerate() {
        positions_x[i] = p.x;
        positions_y[i] = p.y;
        positions_z[i] = p.z;
        charges[i] = p.w;
    }
}

impl ParticleBuffers {
    // rq-b09032cb
    pub fn new(
        gpu: &GpuContext,
        state: &ParticleState,
    ) -> Result<Self, ParticleStateError> {
        let device = gpu.device.clone();
        let kernels = gpu.kernels.clone();
        let n = state.particle_count();
        check_len("positions_y", n, state.positions_y.len())?;
        check_len("positions_z", n, state.positions_z.len())?;
        check_len("images_x", n, state.images_x.len())?;
        check_len("images_y", n, state.images_y.len())?;
        check_len("images_z", n, state.images_z.len())?;
        check_len("velocities_x", n, state.velocities_x.len())?;
        check_len("velocities_y", n, state.velocities_y.len())?;
        check_len("velocities_z", n, state.velocities_z.len())?;
        check_len("forces_x", n, state.forces_x.len())?;
        check_len("forces_y", n, state.forces_y.len())?;
        check_len("forces_z", n, state.forces_z.len())?;
        check_len("potential_energies", n, state.potential_energies.len())?;
        check_len("virials", n, state.virials.len())?;
        check_len("masses", n, state.masses.len())?;
        check_len("charges", n, state.charges.len())?;
        check_len("type_indices", n, state.type_indices.len())?;
        check_len("particle_ids", n, state.particle_ids.len())?;

        let posq_host = interleave_posq(
            &state.positions_x,
            &state.positions_y,
            &state.positions_z,
            &state.charges,
        );
        let posq = if posq_host.is_empty() {
            // cudarc's htod_sync_copy requires non-empty slices.
            device.alloc_zeros::<Real4>(0).map_err(GpuError::from)?
        } else {
            device.htod_sync_copy(&posq_host).map_err(GpuError::from)?
        };
        let images_x = device.htod_sync_copy(&state.images_x).map_err(GpuError::from)?;
        let images_y = device.htod_sync_copy(&state.images_y).map_err(GpuError::from)?;
        let images_z = device.htod_sync_copy(&state.images_z).map_err(GpuError::from)?;
        let velocities_x = device.htod_sync_copy(&state.velocities_x).map_err(GpuError::from)?;
        let velocities_y = device.htod_sync_copy(&state.velocities_y).map_err(GpuError::from)?;
        let velocities_z = device.htod_sync_copy(&state.velocities_z).map_err(GpuError::from)?;
        let forces_x = device.htod_sync_copy(&state.forces_x).map_err(GpuError::from)?;
        let forces_y = device.htod_sync_copy(&state.forces_y).map_err(GpuError::from)?;
        let forces_z = device.htod_sync_copy(&state.forces_z).map_err(GpuError::from)?;
        let potential_energies = device
            .htod_sync_copy(&state.potential_energies)
            .map_err(GpuError::from)?;
        let virials = device.htod_sync_copy(&state.virials).map_err(GpuError::from)?;
        let masses = device.htod_sync_copy(&state.masses).map_err(GpuError::from)?;
        let type_indices = device.htod_sync_copy(&state.type_indices).map_err(GpuError::from)?;
        let particle_ids = device.htod_sync_copy(&state.particle_ids).map_err(GpuError::from)?;
        let reduction_partials = device
            .alloc_zeros::<Real>(REDUCE_PARTIAL_BLOCKS as usize)
            .map_err(GpuError::from)?;

        Ok(ParticleBuffers {
            device,
            kernels,
            posq,
            images_x,
            images_y,
            images_z,
            velocities_x,
            velocities_y,
            velocities_z,
            forces_x,
            forces_y,
            forces_z,
            potential_energies,
            virials,
            masses,
            type_indices,
            particle_ids,
            reduction_partials,
        })
    }

    // rq-18411920
    pub fn particle_count(&self) -> usize {
        self.posq.len()
    }

    /// Download per-atom positions into three host-side arrays. Test
    /// helper: the device buffer is `posq: CudaSlice<Real4>`, so a
    /// direct dtoh of `positions_x` is no longer possible.
    pub fn download_positions(
        &self,
    ) -> Result<(Vec<Real>, Vec<Real>, Vec<Real>), GpuError> {
        let n = self.particle_count();
        let mut posq_host = vec![Real4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }; n];
        if n > 0 {
            self.device
                .dtoh_sync_copy_into(&self.posq, &mut posq_host)
                .map_err(GpuError::from)?;
        }
        let mut px = vec![0 as Real; n];
        let mut py = vec![0 as Real; n];
        let mut pz = vec![0 as Real; n];
        for (i, p) in posq_host.iter().enumerate() {
            px[i] = p.x;
            py[i] = p.y;
            pz[i] = p.z;
        }
        Ok((px, py, pz))
    }

    /// Upload per-atom positions from three host-side arrays, preserving
    /// the existing per-atom charges in `posq[i].w`.
    pub fn upload_positions(
        &mut self,
        positions_x: &[Real],
        positions_y: &[Real],
        positions_z: &[Real],
    ) -> Result<(), GpuError> {
        let n = self.particle_count();
        assert_eq!(positions_x.len(), n);
        assert_eq!(positions_y.len(), n);
        assert_eq!(positions_z.len(), n);
        if n == 0 {
            return Ok(());
        }
        let mut posq_host = vec![Real4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }; n];
        self.device
            .dtoh_sync_copy_into(&self.posq, &mut posq_host)
            .map_err(GpuError::from)?;
        for (i, p) in posq_host.iter_mut().enumerate() {
            p.x = positions_x[i];
            p.y = positions_y[i];
            p.z = positions_z[i];
        }
        self.device
            .htod_sync_copy_into(&posq_host, &mut self.posq)
            .map_err(GpuError::from)?;
        Ok(())
    }

    /// Download per-atom charges into a host-side array.
    pub fn download_charges(&self) -> Result<Vec<Real>, GpuError> {
        let n = self.particle_count();
        let mut posq_host = vec![Real4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }; n];
        if n > 0 {
            self.device
                .dtoh_sync_copy_into(&self.posq, &mut posq_host)
                .map_err(GpuError::from)?;
        }
        Ok(posq_host.into_iter().map(|p| p.w).collect())
    }

    // rq-179ed985
    pub fn upload(&mut self, state: &ParticleState) -> Result<(), ParticleStateError> {
        let n = self.particle_count();
        check_len("positions_x", n, state.positions_x.len())?;
        check_len("positions_y", n, state.positions_y.len())?;
        check_len("positions_z", n, state.positions_z.len())?;
        check_len("images_x", n, state.images_x.len())?;
        check_len("images_y", n, state.images_y.len())?;
        check_len("images_z", n, state.images_z.len())?;
        check_len("velocities_x", n, state.velocities_x.len())?;
        check_len("velocities_y", n, state.velocities_y.len())?;
        check_len("velocities_z", n, state.velocities_z.len())?;
        check_len("forces_x", n, state.forces_x.len())?;
        check_len("forces_y", n, state.forces_y.len())?;
        check_len("forces_z", n, state.forces_z.len())?;
        check_len("potential_energies", n, state.potential_energies.len())?;
        check_len("virials", n, state.virials.len())?;
        check_len("masses", n, state.masses.len())?;
        check_len("charges", n, state.charges.len())?;
        check_len("type_indices", n, state.type_indices.len())?;
        check_len("particle_ids", n, state.particle_ids.len())?;

        let device = &self.device;
        let posq_host = interleave_posq(
            &state.positions_x,
            &state.positions_y,
            &state.positions_z,
            &state.charges,
        );
        if !posq_host.is_empty() {
            device
                .htod_sync_copy_into(&posq_host, &mut self.posq)
                .map_err(GpuError::from)?;
        }
        device
            .htod_sync_copy_into(&state.images_x, &mut self.images_x)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.images_y, &mut self.images_y)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.images_z, &mut self.images_z)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.velocities_x, &mut self.velocities_x)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.velocities_y, &mut self.velocities_y)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.velocities_z, &mut self.velocities_z)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.forces_x, &mut self.forces_x)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.forces_y, &mut self.forces_y)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.forces_z, &mut self.forces_z)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.potential_energies, &mut self.potential_energies)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.virials, &mut self.virials)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.masses, &mut self.masses)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.type_indices, &mut self.type_indices)
            .map_err(GpuError::from)?;
        device
            .htod_sync_copy_into(&state.particle_ids, &mut self.particle_ids)
            .map_err(GpuError::from)?;
        Ok(())
    }
}
