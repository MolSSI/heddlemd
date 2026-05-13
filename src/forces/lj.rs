// rq-a5a919df rq-d3a14184
use std::sync::Arc;

use cudarc::driver::CudaDevice;

use crate::gpu::{
    LennardJonesParameterTable, PairBuffer, ParticleBuffers, lj_pair_force, reduce_pair_forces,
};
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::bonds::{DeviceExclusionList, ExclusionList};
use super::neighbor_list::NeighborListError;
use super::{ForceFieldContext, ForceFieldError, Potential, SlotForceView};

// rq-af2d1628
#[derive(Debug)]
pub struct LennardJonesState {
    #[allow(dead_code)]
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) pair_buffer: PairBuffer,
    pub(crate) params: LennardJonesParameterTable,
    pub(crate) exclusions: DeviceExclusionList,
    pub(crate) particle_count: usize,
    pub(crate) max_cutoff: f32,
}

impl LennardJonesState {
    pub fn new(
        device: Arc<CudaDevice>,
        particle_count: usize,
        params: LennardJonesParameterTable,
        max_cutoff: f32,
        max_neighbors: u32,
        exclusion_list: &ExclusionList,
    ) -> Result<Self, NeighborListError> {
        let pair_buffer = PairBuffer::new(device.clone(), particle_count, max_neighbors)?;
        let exclusions = DeviceExclusionList::from_host(&device, exclusion_list)?;
        Ok(LennardJonesState {
            device,
            pair_buffer,
            params,
            exclusions,
            particle_count,
            max_cutoff,
        })
    }

    pub fn particle_count(&self) -> usize {
        self.particle_count
    }
}

impl Potential for LennardJonesState {
    fn label(&self) -> &'static str {
        "lennard_jones"
    }

    fn max_cutoff(&self) -> Option<f32> {
        Some(self.max_cutoff)
    }

    fn contribute(
        &mut self,
        buffers: &ParticleBuffers,
        sim_box: &SimulationBox,
        cx: &ForceFieldContext<'_>,
        timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        if self.particle_count == 0 {
            return Ok(());
        }
        let nl = cx
            .neighbor_list
            .expect("LennardJonesState requires a shared neighbor list");
        timings.kernel_start(KernelStage::LjPairForce)?;
        lj_pair_force(
            buffers,
            &mut self.pair_buffer,
            sim_box,
            &self.params,
            &self.exclusions.atom_excl_offsets,
            &self.exclusions.atom_excl_partners,
            &self.exclusions.atom_excl_scales,
            &nl.neighbor_list,
            &nl.neighbor_counts,
        )?;
        timings.kernel_stop(KernelStage::LjPairForce)?;
        Ok(())
    }

    fn reduce(
        &mut self,
        mut output: SlotForceView<'_>,
        cx: &ForceFieldContext<'_>,
        timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        if self.particle_count == 0 {
            return Ok(());
        }
        let nl = cx
            .neighbor_list
            .expect("LennardJonesState requires a shared neighbor list");
        timings.kernel_start(KernelStage::ReducePairForces)?;
        reduce_pair_forces(
            &self.pair_buffer,
            &nl.neighbor_counts,
            &mut output.x,
            &mut output.y,
            &mut output.z,
            self.particle_count,
        )?;
        timings.kernel_stop(KernelStage::ReducePairForces)?;
        Ok(())
    }
}
