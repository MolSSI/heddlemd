use std::collections::HashSet;

use crate::gpu::{GpuError, ParticleBuffers};
use crate::precision::Real;

// rq-3766be01
#[derive(Debug, Clone)]
pub struct ParticleState {
    pub positions_x: Vec<Real>,
    pub positions_y: Vec<Real>,
    pub positions_z: Vec<Real>,
    pub images_x: Vec<i32>,
    pub images_y: Vec<i32>,
    pub images_z: Vec<i32>,
    pub velocities_x: Vec<Real>,
    pub velocities_y: Vec<Real>,
    pub velocities_z: Vec<Real>,
    pub forces_x: Vec<Real>,
    pub forces_y: Vec<Real>,
    pub forces_z: Vec<Real>,
    pub potential_energies: Vec<Real>,
    pub virials: Vec<Real>,
    pub masses: Vec<Real>,
    pub charges: Vec<Real>,
    pub type_indices: Vec<u32>,
    pub particle_ids: Vec<u32>,
}

// rq-bec7b519 rq-e1ceb5c0 rq-6cf916af
#[derive(Debug, thiserror::Error)]
pub enum ParticleStateError {
    #[error("length mismatch on array `{array}`: expected {expected}, got {actual}")]
    LengthMismatch {
        array: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("duplicate particle id {0}")]
    DuplicateParticleId(u32),
    #[error("{0}")]
    Gpu(#[from] GpuError),
}

pub(crate) fn check_len(
    array: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), ParticleStateError> {
    if expected == actual {
        Ok(())
    } else {
        Err(ParticleStateError::LengthMismatch {
            array,
            expected,
            actual,
        })
    }
}

impl ParticleState {
    // rq-5e0598cb
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        positions_x: Vec<Real>,
        positions_y: Vec<Real>,
        positions_z: Vec<Real>,
        velocities_x: Vec<Real>,
        velocities_y: Vec<Real>,
        velocities_z: Vec<Real>,
        masses: Vec<Real>,
        charges: Vec<Real>,
        type_indices: Vec<u32>,
        ids: Option<Vec<u32>>,
        images: Option<(Vec<i32>, Vec<i32>, Vec<i32>)>,
    ) -> Result<Self, ParticleStateError> {
        let n = positions_x.len();
        check_len("positions_y", n, positions_y.len())?;
        check_len("positions_z", n, positions_z.len())?;
        check_len("velocities_x", n, velocities_x.len())?;
        check_len("velocities_y", n, velocities_y.len())?;
        check_len("velocities_z", n, velocities_z.len())?;
        check_len("masses", n, masses.len())?;
        check_len("charges", n, charges.len())?;
        check_len("type_indices", n, type_indices.len())?;

        let particle_ids = match ids {
            Some(v) => {
                check_len("particle_ids", n, v.len())?;
                let mut seen: HashSet<u32> = HashSet::with_capacity(n);
                for &id in &v {
                    if !seen.insert(id) {
                        return Err(ParticleStateError::DuplicateParticleId(id));
                    }
                }
                v
            }
            None => (0..n as u32).collect(),
        };

        let (images_x, images_y, images_z) = match images {
            Some((ix, iy, iz)) => {
                check_len("images_x", n, ix.len())?;
                check_len("images_y", n, iy.len())?;
                check_len("images_z", n, iz.len())?;
                (ix, iy, iz)
            }
            None => (vec![0i32; n], vec![0i32; n], vec![0i32; n]),
        };

        Ok(ParticleState {
            positions_x,
            positions_y,
            positions_z,
            images_x,
            images_y,
            images_z,
            velocities_x,
            velocities_y,
            velocities_z,
            forces_x: vec![0.0; n],
            forces_y: vec![0.0; n],
            forces_z: vec![0.0; n],
            potential_energies: vec![0.0; n],
            virials: vec![0.0; n],
            masses,
            charges,
            type_indices,
            particle_ids,
        })
    }

    // rq-ac035b90
    pub fn particle_count(&self) -> usize {
        self.positions_x.len()
    }

    // rq-9a19bfa3
    pub fn download_from(
        &mut self,
        buffers: &ParticleBuffers,
    ) -> Result<(), ParticleStateError> {
        let n = buffers.particle_count();
        check_len("positions_x", n, self.positions_x.len())?;
        check_len("positions_y", n, self.positions_y.len())?;
        check_len("positions_z", n, self.positions_z.len())?;
        check_len("images_x", n, self.images_x.len())?;
        check_len("images_y", n, self.images_y.len())?;
        check_len("images_z", n, self.images_z.len())?;
        check_len("velocities_x", n, self.velocities_x.len())?;
        check_len("velocities_y", n, self.velocities_y.len())?;
        check_len("velocities_z", n, self.velocities_z.len())?;
        check_len("forces_x", n, self.forces_x.len())?;
        check_len("forces_y", n, self.forces_y.len())?;
        check_len("forces_z", n, self.forces_z.len())?;
        check_len("potential_energies", n, self.potential_energies.len())?;
        check_len("virials", n, self.virials.len())?;
        check_len("masses", n, self.masses.len())?;
        check_len("charges", n, self.charges.len())?;
        check_len("type_indices", n, self.type_indices.len())?;
        check_len("particle_ids", n, self.particle_ids.len())?;

        let device = &buffers.device;
        device
            .dtoh_sync_copy_into(&buffers.positions_x, &mut self.positions_x)
            .map_err(GpuError::from)?;
        device
            .dtoh_sync_copy_into(&buffers.positions_y, &mut self.positions_y)
            .map_err(GpuError::from)?;
        device
            .dtoh_sync_copy_into(&buffers.positions_z, &mut self.positions_z)
            .map_err(GpuError::from)?;
        device
            .dtoh_sync_copy_into(&buffers.images_x, &mut self.images_x)
            .map_err(GpuError::from)?;
        device
            .dtoh_sync_copy_into(&buffers.images_y, &mut self.images_y)
            .map_err(GpuError::from)?;
        device
            .dtoh_sync_copy_into(&buffers.images_z, &mut self.images_z)
            .map_err(GpuError::from)?;
        device
            .dtoh_sync_copy_into(&buffers.velocities_x, &mut self.velocities_x)
            .map_err(GpuError::from)?;
        device
            .dtoh_sync_copy_into(&buffers.velocities_y, &mut self.velocities_y)
            .map_err(GpuError::from)?;
        device
            .dtoh_sync_copy_into(&buffers.velocities_z, &mut self.velocities_z)
            .map_err(GpuError::from)?;
        device
            .dtoh_sync_copy_into(&buffers.forces_x, &mut self.forces_x)
            .map_err(GpuError::from)?;
        device
            .dtoh_sync_copy_into(&buffers.forces_y, &mut self.forces_y)
            .map_err(GpuError::from)?;
        device
            .dtoh_sync_copy_into(&buffers.forces_z, &mut self.forces_z)
            .map_err(GpuError::from)?;
        device
            .dtoh_sync_copy_into(&buffers.potential_energies, &mut self.potential_energies)
            .map_err(GpuError::from)?;
        device
            .dtoh_sync_copy_into(&buffers.virials, &mut self.virials)
            .map_err(GpuError::from)?;
        device
            .dtoh_sync_copy_into(&buffers.masses, &mut self.masses)
            .map_err(GpuError::from)?;
        device
            .dtoh_sync_copy_into(&buffers.charges, &mut self.charges)
            .map_err(GpuError::from)?;
        device
            .dtoh_sync_copy_into(&buffers.type_indices, &mut self.type_indices)
            .map_err(GpuError::from)?;
        device
            .dtoh_sync_copy_into(&buffers.particle_ids, &mut self.particle_ids)
            .map_err(GpuError::from)?;
        Ok(())
    }
}
