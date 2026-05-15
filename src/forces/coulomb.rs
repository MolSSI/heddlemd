// rq-846bdb8b
use std::sync::Arc;

use cudarc::driver::CudaDevice;

use crate::gpu::{GpuContext, PairBuffer, ParticleBuffers, coulomb_pair_force, reduce_pair_forces};
use crate::io::config::CoulombConfig;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::bonds::{DeviceExclusionList, ExclusionList};
use super::neighbor_list::NeighborListError;
use super::{ForceFieldContext, ForceFieldError, Potential, SlotOutputView};

// CoulombParameters carries the runtime real-space parameters: the
// cutoff and the inner switching radius. Per-particle charges live on
// `ParticleBuffers`. See rq-bfd7004c.
#[derive(Debug, Clone, Copy)]
pub struct CoulombParameters {
    pub cutoff: f32,
    pub r_switch: f32,
}

impl From<&CoulombConfig> for CoulombParameters {
    fn from(c: &CoulombConfig) -> Self {
        CoulombParameters {
            cutoff: c.cutoff as f32,
            r_switch: c.r_switch as f32,
        }
    }
}

// rq-846bdb8b
#[derive(Debug)]
pub struct CoulombState {
    #[allow(dead_code)]
    pub(crate) device: Arc<CudaDevice>,
    pub(crate) pair_buffer: PairBuffer,
    pub(crate) params: CoulombParameters,
    pub(crate) exclusions: DeviceExclusionList,
    pub(crate) particle_count: usize,
}

impl CoulombState {
    pub fn new(
        gpu: &GpuContext,
        particle_count: usize,
        params: CoulombParameters,
        max_neighbors: u32,
        exclusion_list: &ExclusionList,
    ) -> Result<Self, NeighborListError> {
        let device = gpu.device.clone();
        let pair_buffer = PairBuffer::new(gpu, particle_count, max_neighbors)?;
        let exclusions = DeviceExclusionList::from_host(&device, exclusion_list)?;
        Ok(CoulombState {
            device,
            pair_buffer,
            params,
            exclusions,
            particle_count,
        })
    }

    pub fn particle_count(&self) -> usize {
        self.particle_count
    }
}

impl Potential for CoulombState {
    fn label(&self) -> &'static str {
        "coulomb"
    }

    fn max_cutoff(&self) -> Option<f32> {
        Some(self.params.cutoff)
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
            .expect("CoulombState requires a shared neighbor list");
        timings.kernel_start(KernelStage::COULOMB_PAIR_FORCE)?;
        coulomb_pair_force(
            buffers,
            &mut self.pair_buffer,
            sim_box,
            self.params.cutoff,
            self.params.r_switch,
            &self.exclusions.atom_excl_offsets,
            &self.exclusions.atom_excl_partners,
            &self.exclusions.atom_excl_coul_scales,
            &nl.neighbor_list,
            &nl.neighbor_counts,
        )?;
        timings.kernel_stop(KernelStage::COULOMB_PAIR_FORCE)?;
        Ok(())
    }

    fn reduce(
        &mut self,
        mut output: SlotOutputView<'_>,
        cx: &ForceFieldContext<'_>,
        timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        if self.particle_count == 0 {
            return Ok(());
        }
        let nl = cx
            .neighbor_list
            .expect("CoulombState requires a shared neighbor list");
        timings.kernel_start(KernelStage::REDUCE_PAIR_FORCES)?;
        reduce_pair_forces(
            &self.pair_buffer,
            &nl.neighbor_counts,
            &mut output.force_x,
            &mut output.force_y,
            &mut output.force_z,
            &mut output.energy,
            &mut output.virial,
            self.particle_count,
        )?;
        timings.kernel_stop(KernelStage::REDUCE_PAIR_FORCES)?;
        Ok(())
    }
}
