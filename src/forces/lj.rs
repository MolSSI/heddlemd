use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};

use crate::gpu::{
    GpuError, LennardJonesParameters, PairBuffer, ParticleBuffers, lj_pair_force,
    reduce_pair_forces,
};
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::bonds::{DeviceExclusionList, ExclusionList};

#[derive(Debug)]
pub struct LennardJonesState {
    #[allow(dead_code)]
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) pair_buffer: PairBuffer,
    pub(crate) neighbor_counts: CudaSlice<u32>,
    pub(crate) params: LennardJonesParameters,
    pub(crate) exclusions: DeviceExclusionList,
    pub(crate) accumulator_x: CudaSlice<f32>,
    pub(crate) accumulator_y: CudaSlice<f32>,
    pub(crate) accumulator_z: CudaSlice<f32>,
    pub(crate) particle_count: usize,
}

impl LennardJonesState {
    pub fn new(
        device: Arc<CudaDevice>,
        particle_count: usize,
        params: LennardJonesParameters,
        exclusion_list: &ExclusionList,
    ) -> Result<Self, GpuError> {
        let max_neighbors = if particle_count == 0 {
            0u32
        } else {
            particle_count as u32
        };
        let pair_buffer = PairBuffer::new(device.clone(), particle_count, max_neighbors)?;
        let neighbor_counts = device
            .htod_sync_copy(&vec![particle_count as u32; particle_count])
            .map_err(GpuError::from)?;
        let exclusions = DeviceExclusionList::from_host(&device, exclusion_list)?;
        let accumulator_x = device.alloc_zeros::<f32>(particle_count).map_err(GpuError::from)?;
        let accumulator_y = device.alloc_zeros::<f32>(particle_count).map_err(GpuError::from)?;
        let accumulator_z = device.alloc_zeros::<f32>(particle_count).map_err(GpuError::from)?;
        Ok(LennardJonesState {
            device,
            pair_buffer,
            neighbor_counts,
            params,
            exclusions,
            accumulator_x,
            accumulator_y,
            accumulator_z,
            particle_count,
        })
    }

    pub(crate) fn contribute(
        &mut self,
        buffers: &ParticleBuffers,
        sim_box: &SimulationBox,
        timings: &mut Timings,
    ) -> Result<(), GpuError> {
        if self.particle_count == 0 {
            return Ok(());
        }
        timings
            .kernel_start(KernelStage::LjPairForce)
            .map_err(map_timings_err)?;
        lj_pair_force(
            buffers,
            &mut self.pair_buffer,
            sim_box,
            &self.params,
            &self.exclusions.atom_excl_offsets,
            &self.exclusions.atom_excl_partners,
            &self.exclusions.atom_excl_scales,
        )?;
        timings
            .kernel_stop(KernelStage::LjPairForce)
            .map_err(map_timings_err)?;
        Ok(())
    }

    pub(crate) fn reduce(&mut self, timings: &mut Timings) -> Result<(), GpuError> {
        if self.particle_count == 0 {
            return Ok(());
        }
        timings
            .kernel_start(KernelStage::ReducePairForces)
            .map_err(map_timings_err)?;
        reduce_pair_forces(
            &self.pair_buffer,
            &self.neighbor_counts,
            &mut self.accumulator_x,
            &mut self.accumulator_y,
            &mut self.accumulator_z,
            self.particle_count,
        )?;
        timings
            .kernel_stop(KernelStage::ReducePairForces)
            .map_err(map_timings_err)?;
        Ok(())
    }

    pub(crate) fn accumulator(&self) -> (&CudaSlice<f32>, &CudaSlice<f32>, &CudaSlice<f32>) {
        (&self.accumulator_x, &self.accumulator_y, &self.accumulator_z)
    }
}

fn map_timings_err(e: crate::timings::TimingsError) -> GpuError {
    match e {
        crate::timings::TimingsError::Gpu(g) => g,
    }
}
