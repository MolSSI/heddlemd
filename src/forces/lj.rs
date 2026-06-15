// rq-a5a919df rq-d3a14184
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction};
use cudarc::nvrtc::Ptx;

use crate::gpu::device::get_func;
use crate::gpu::{
    GpuContext, GpuError, LennardJonesParameterTable, PairBuffer, ParticleBuffers, lj_pair_force,
    reduce_pair_energy_virial, reduce_pair_forces,
};
use crate::kernels;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::topology::{DeviceExclusionList, ExclusionList};
use super::neighbor_list::NeighborListError;
use super::{
    AggregateLevel, ForceFieldContext, ForceFieldError, Potential, PotentialBuildContext,
    PotentialBuilder, SlotOutputView,
};

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
        gpu: &GpuContext,
        particle_count: usize,
        params: LennardJonesParameterTable,
        max_cutoff: f32,
        max_neighbors: u32,
        exclusion_list: &ExclusionList,
    ) -> Result<Self, NeighborListError> {
        let device = gpu.device.clone();
        let pair_buffer = PairBuffer::new(gpu, particle_count, max_neighbors)?;
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
        timings.kernel_start(KernelStage::LJ_PAIR_FORCE)?;
        lj_pair_force(
            buffers,
            &mut self.pair_buffer,
            sim_box,
            &self.params,
            &self.exclusions.atom_excl_offsets,
            &self.exclusions.atom_excl_partners,
            &self.exclusions.atom_excl_lj_scales,
            &nl.neighbor_list,
            &nl.neighbor_counts,
        )?;
        timings.kernel_stop(KernelStage::LJ_PAIR_FORCE)?;
        Ok(())
    }

    fn reduce(
        &mut self,
        mut output: SlotOutputView<'_>,
        cx: &ForceFieldContext<'_>,
        timings: &mut Timings,
        level: AggregateLevel,
    ) -> Result<(), ForceFieldError> {
        if self.particle_count == 0 {
            return Ok(());
        }
        let nl = cx
            .neighbor_list
            .expect("LennardJonesState requires a shared neighbor list");
        timings.kernel_start(KernelStage::REDUCE_PAIR_FORCES)?;
        reduce_pair_forces(
            &self.pair_buffer,
            &nl.neighbor_counts,
            &mut output.force_x,
            &mut output.force_y,
            &mut output.force_z,
            self.particle_count,
        )?;
        timings.kernel_stop(KernelStage::REDUCE_PAIR_FORCES)?;
        if level.includes_scalars() {
            timings.kernel_start(KernelStage::REDUCE_PAIR_ENERGY_VIRIAL)?;
            reduce_pair_energy_virial(
                &self.pair_buffer,
                &nl.neighbor_counts,
                &mut output.energy,
                &mut output.virial,
                self.particle_count,
            )?;
            timings.kernel_stop(KernelStage::REDUCE_PAIR_ENERGY_VIRIAL)?;
        }
        Ok(())
    }
}

// rq-e8550f96
#[derive(Debug, Clone)]
pub struct LennardJonesBuilder;

impl PotentialBuilder for LennardJonesBuilder {
    fn build(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        if cx.pair_interactions.is_empty() {
            return Ok(None);
        }
        let params = LennardJonesParameterTable::from_config(
            &cx.gpu.device,
            cx.particle_types,
            cx.pair_interactions,
        )?;
        let max_cutoff = cx
            .pair_interactions
            .iter()
            .map(|p| p.cutoff as f32)
            .fold(0.0_f32, f32::max);
        let max_neighbors = super::max_neighbors_from(cx.neighbor_list_config, cx.particle_count);
        let state = LennardJonesState::new(
            cx.gpu,
            cx.particle_count,
            params,
            max_cutoff,
            max_neighbors,
            cx.exclusion_list,
        )?;
        Ok(Some(Box::new(state)))
    }

    fn box_clone(&self) -> Box<dyn PotentialBuilder> {
        Box::new(self.clone())
    }
}

// rq-2093594f rq-78d9fd1c
#[derive(Debug, Clone)]
pub struct LjKernels {
    pub pair_force: CudaFunction,
}

impl LjKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::PAIR_FORCE),
            "pair_force",
            &["lj_pair_force"],
        )?;
        Ok(LjKernels {
            pair_force: get_func(device, "pair_force", "lj_pair_force")?,
        })
    }
}

