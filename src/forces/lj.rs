// rq-a5a919df rq-d3a14184 rq-d46a89d5
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};

use crate::gpu::{
    GpuError, LennardJonesParameters, PairBuffer, ParticleBuffers, lj_pair_force,
    lj_pair_force_neighbor, reduce_pair_forces,
};
use crate::io::config::NeighborListConfig;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::bonds::{DeviceExclusionList, ExclusionList};
use super::neighbor_list::{NeighborListError, NeighborListState};

#[derive(Debug)]
pub struct LjCommon {
    #[allow(dead_code)]
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) pair_buffer: PairBuffer,
    pub(crate) params: LennardJonesParameters,
    pub(crate) exclusions: DeviceExclusionList,
    pub(crate) accumulator_x: CudaSlice<f32>,
    pub(crate) accumulator_y: CudaSlice<f32>,
    pub(crate) accumulator_z: CudaSlice<f32>,
    pub(crate) particle_count: usize,
}

#[derive(Debug)]
pub enum LennardJonesState {
    AllPairs {
        common: LjCommon,
        neighbor_counts: CudaSlice<u32>,
    },
    CellList {
        common: LjCommon,
        neighbor_list: NeighborListState,
    },
}

impl LennardJonesState {
    pub fn new(
        device: Arc<CudaDevice>,
        particle_count: usize,
        sim_box: SimulationBox,
        params: LennardJonesParameters,
        exclusion_list: &ExclusionList,
        neighbor_list_config: &NeighborListConfig,
    ) -> Result<Self, NeighborListError> {
        match neighbor_list_config {
            NeighborListConfig::AllPairs => {
                let max_neighbors = if particle_count == 0 {
                    0u32
                } else {
                    particle_count as u32
                };
                let pair_buffer =
                    PairBuffer::new(device.clone(), particle_count, max_neighbors)?;
                let neighbor_counts = device
                    .htod_sync_copy(&vec![particle_count as u32; particle_count])
                    .map_err(GpuError::from)?;
                let exclusions = DeviceExclusionList::from_host(&device, exclusion_list)?;
                let accumulator_x =
                    device.alloc_zeros::<f32>(particle_count).map_err(GpuError::from)?;
                let accumulator_y =
                    device.alloc_zeros::<f32>(particle_count).map_err(GpuError::from)?;
                let accumulator_z =
                    device.alloc_zeros::<f32>(particle_count).map_err(GpuError::from)?;
                Ok(LennardJonesState::AllPairs {
                    common: LjCommon {
                        device,
                        pair_buffer,
                        params,
                        exclusions,
                        accumulator_x,
                        accumulator_y,
                        accumulator_z,
                        particle_count,
                    },
                    neighbor_counts,
                })
            }
            NeighborListConfig::CellList {
                max_neighbors,
                r_skin,
            } => {
                let pair_buffer =
                    PairBuffer::new(device.clone(), particle_count, *max_neighbors)?;
                let exclusions = DeviceExclusionList::from_host(&device, exclusion_list)?;
                let accumulator_x =
                    device.alloc_zeros::<f32>(particle_count).map_err(GpuError::from)?;
                let accumulator_y =
                    device.alloc_zeros::<f32>(particle_count).map_err(GpuError::from)?;
                let accumulator_z =
                    device.alloc_zeros::<f32>(particle_count).map_err(GpuError::from)?;
                let neighbor_list = NeighborListState::new(
                    device.clone(),
                    sim_box,
                    particle_count,
                    params.cutoff,
                    *max_neighbors,
                    *r_skin as f32,
                )?;
                Ok(LennardJonesState::CellList {
                    common: LjCommon {
                        device,
                        pair_buffer,
                        params,
                        exclusions,
                        accumulator_x,
                        accumulator_y,
                        accumulator_z,
                        particle_count,
                    },
                    neighbor_list,
                })
            }
        }
    }

    pub(crate) fn contribute(
        &mut self,
        buffers: &ParticleBuffers,
        sim_box: &SimulationBox,
        timings: &mut Timings,
    ) -> Result<(), NeighborListError> {
        match self {
            LennardJonesState::AllPairs { common, .. } => {
                if common.particle_count == 0 {
                    return Ok(());
                }
                timings
                    .kernel_start(KernelStage::LjPairForce)
                    .map_err(map_timings_err)?;
                lj_pair_force(
                    buffers,
                    &mut common.pair_buffer,
                    sim_box,
                    &common.params,
                    &common.exclusions.atom_excl_offsets,
                    &common.exclusions.atom_excl_partners,
                    &common.exclusions.atom_excl_scales,
                )?;
                timings
                    .kernel_stop(KernelStage::LjPairForce)
                    .map_err(map_timings_err)?;
                Ok(())
            }
            LennardJonesState::CellList {
                common,
                neighbor_list,
            } => {
                if common.particle_count == 0 {
                    return Ok(());
                }
                neighbor_list.pre_step(buffers, timings)?;
                timings
                    .kernel_start(KernelStage::LjPairForceNeighbor)
                    .map_err(map_timings_err)?;
                lj_pair_force_neighbor(
                    buffers,
                    &mut common.pair_buffer,
                    &neighbor_list.sim_box,
                    &common.params,
                    &common.exclusions.atom_excl_offsets,
                    &common.exclusions.atom_excl_partners,
                    &common.exclusions.atom_excl_scales,
                    &neighbor_list.neighbor_list,
                    &neighbor_list.neighbor_counts,
                )?;
                timings
                    .kernel_stop(KernelStage::LjPairForceNeighbor)
                    .map_err(map_timings_err)?;
                Ok(())
            }
        }
    }

    pub(crate) fn reduce(&mut self, timings: &mut Timings) -> Result<(), NeighborListError> {
        let (common, neighbor_counts) = match self {
            LennardJonesState::AllPairs { common, neighbor_counts } => (common, &*neighbor_counts),
            LennardJonesState::CellList { common, neighbor_list } => {
                (common, &neighbor_list.neighbor_counts)
            }
        };
        if common.particle_count == 0 {
            return Ok(());
        }
        timings
            .kernel_start(KernelStage::ReducePairForces)
            .map_err(map_timings_err)?;
        reduce_pair_forces(
            &common.pair_buffer,
            neighbor_counts,
            &mut common.accumulator_x,
            &mut common.accumulator_y,
            &mut common.accumulator_z,
            common.particle_count,
        )?;
        timings
            .kernel_stop(KernelStage::ReducePairForces)
            .map_err(map_timings_err)?;
        Ok(())
    }

    pub(crate) fn accumulator(&self) -> (&CudaSlice<f32>, &CudaSlice<f32>, &CudaSlice<f32>) {
        let common = match self {
            LennardJonesState::AllPairs { common, .. } => common,
            LennardJonesState::CellList { common, .. } => common,
        };
        (&common.accumulator_x, &common.accumulator_y, &common.accumulator_z)
    }

    pub fn particle_count(&self) -> usize {
        match self {
            LennardJonesState::AllPairs { common, .. } => common.particle_count,
            LennardJonesState::CellList { common, .. } => common.particle_count,
        }
    }
}

fn map_timings_err(e: crate::timings::TimingsError) -> NeighborListError {
    match e {
        crate::timings::TimingsError::Gpu(g) => NeighborListError::Gpu(g),
    }
}
