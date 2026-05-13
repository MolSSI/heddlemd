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
use super::{ForceFieldError, Potential, SlotForceView};

#[derive(Debug)]
pub struct LjCommon {
    #[allow(dead_code)]
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) pair_buffer: PairBuffer,
    pub(crate) params: LennardJonesParameters,
    pub(crate) exclusions: DeviceExclusionList,
    pub(crate) particle_count: usize,
}

// rq-af2d1628
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
                Ok(LennardJonesState::AllPairs {
                    common: LjCommon {
                        device,
                        pair_buffer,
                        params,
                        exclusions,
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
                        particle_count,
                    },
                    neighbor_list,
                })
            }
        }
    }

    pub fn particle_count(&self) -> usize {
        match self {
            LennardJonesState::AllPairs { common, .. } => common.particle_count,
            LennardJonesState::CellList { common, .. } => common.particle_count,
        }
    }
}

impl Potential for LennardJonesState {
    fn label(&self) -> &'static str {
        "lennard_jones"
    }

    fn contribute(
        &mut self,
        buffers: &ParticleBuffers,
        sim_box: &SimulationBox,
        timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        match self {
            LennardJonesState::AllPairs { common, .. } => {
                if common.particle_count == 0 {
                    return Ok(());
                }
                timings.kernel_start(KernelStage::LjPairForce)?;
                lj_pair_force(
                    buffers,
                    &mut common.pair_buffer,
                    sim_box,
                    &common.params,
                    &common.exclusions.atom_excl_offsets,
                    &common.exclusions.atom_excl_partners,
                    &common.exclusions.atom_excl_scales,
                )?;
                timings.kernel_stop(KernelStage::LjPairForce)?;
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
                timings.kernel_start(KernelStage::LjPairForceNeighbor)?;
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
                timings.kernel_stop(KernelStage::LjPairForceNeighbor)?;
                Ok(())
            }
        }
    }

    fn reduce(
        &mut self,
        mut output: SlotForceView<'_>,
        timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        let (common, neighbor_counts) = match self {
            LennardJonesState::AllPairs { common, neighbor_counts } => (common, &*neighbor_counts),
            LennardJonesState::CellList { common, neighbor_list } => {
                (common, &neighbor_list.neighbor_counts)
            }
        };
        if common.particle_count == 0 {
            return Ok(());
        }
        timings.kernel_start(KernelStage::ReducePairForces)?;
        reduce_pair_forces(
            &common.pair_buffer,
            neighbor_counts,
            &mut output.x,
            &mut output.y,
            &mut output.z,
            common.particle_count,
        )?;
        timings.kernel_stop(KernelStage::ReducePairForces)?;
        Ok(())
    }
}
