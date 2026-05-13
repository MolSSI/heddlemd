use std::collections::HashSet;

use crate::gpu::{GpuError, ParticleBuffers};

// rq-3766be01
#[derive(Debug, Clone)]
pub struct ParticleState {
    pub positions_x: Vec<f32>,
    pub positions_y: Vec<f32>,
    pub positions_z: Vec<f32>,
    pub velocities_x: Vec<f32>,
    pub velocities_y: Vec<f32>,
    pub velocities_z: Vec<f32>,
    pub forces_x: Vec<f32>,
    pub forces_y: Vec<f32>,
    pub forces_z: Vec<f32>,
    pub potential_energies: Vec<f32>,
    pub virials: Vec<f32>,
    pub masses: Vec<f32>,
    pub type_indices: Vec<u32>,
    pub particle_ids: Vec<u32>,
}

// rq-bec7b519
#[derive(Debug)]
pub enum ParticleStateError {
    LengthMismatch {
        array: &'static str,
        expected: usize,
        actual: usize,
    },
    DuplicateParticleId(u32),
    Gpu(GpuError),
}

impl std::fmt::Display for ParticleStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParticleStateError::LengthMismatch {
                array,
                expected,
                actual,
            } => write!(
                f,
                "length mismatch on array {array}: expected {expected}, got {actual}"
            ),
            ParticleStateError::DuplicateParticleId(id) => {
                write!(f, "duplicate particle id {id}")
            }
            ParticleStateError::Gpu(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ParticleStateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ParticleStateError::Gpu(e) => Some(e),
            _ => None,
        }
    }
}

impl From<GpuError> for ParticleStateError {
    fn from(e: GpuError) -> Self {
        ParticleStateError::Gpu(e)
    }
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
        positions_x: Vec<f32>,
        positions_y: Vec<f32>,
        positions_z: Vec<f32>,
        velocities_x: Vec<f32>,
        velocities_y: Vec<f32>,
        velocities_z: Vec<f32>,
        masses: Vec<f32>,
        type_indices: Vec<u32>,
        ids: Option<Vec<u32>>,
    ) -> Result<Self, ParticleStateError> {
        let n = positions_x.len();
        check_len("positions_y", n, positions_y.len())?;
        check_len("positions_z", n, positions_z.len())?;
        check_len("velocities_x", n, velocities_x.len())?;
        check_len("velocities_y", n, velocities_y.len())?;
        check_len("velocities_z", n, velocities_z.len())?;
        check_len("masses", n, masses.len())?;
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

        Ok(ParticleState {
            positions_x,
            positions_y,
            positions_z,
            velocities_x,
            velocities_y,
            velocities_z,
            forces_x: vec![0.0; n],
            forces_y: vec![0.0; n],
            forces_z: vec![0.0; n],
            potential_energies: vec![0.0; n],
            virials: vec![0.0; n],
            masses,
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
        check_len("velocities_x", n, self.velocities_x.len())?;
        check_len("velocities_y", n, self.velocities_y.len())?;
        check_len("velocities_z", n, self.velocities_z.len())?;
        check_len("forces_x", n, self.forces_x.len())?;
        check_len("forces_y", n, self.forces_y.len())?;
        check_len("forces_z", n, self.forces_z.len())?;
        check_len("potential_energies", n, self.potential_energies.len())?;
        check_len("virials", n, self.virials.len())?;
        check_len("masses", n, self.masses.len())?;
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
            .dtoh_sync_copy_into(&buffers.type_indices, &mut self.type_indices)
            .map_err(GpuError::from)?;
        device
            .dtoh_sync_copy_into(&buffers.particle_ids, &mut self.particle_ids)
            .map_err(GpuError::from)?;
        Ok(())
    }
}
