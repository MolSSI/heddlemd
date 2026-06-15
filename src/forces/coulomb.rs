// rq-846bdb8b
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction};
use cudarc::nvrtc::Ptx;

use crate::gpu::device::get_func;
use crate::gpu::{
    GpuContext, GpuError, PairBuffer, ParticleBuffers, coulomb_pair_force,
    reduce_pair_energy_virial, reduce_pair_forces,
};
use crate::kernels;
use crate::io::config::CoulombConfig;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::topology::{DeviceExclusionList, ExclusionList};
use super::neighbor_list::NeighborListError;
use super::{
    AggregateLevel, ForceFieldContext, ForceFieldError, Potential, PotentialBuildContext,
    PotentialBuilder, SlotOutputView,
};

// CoulombParameters carries the runtime real-space parameters: the
// cutoff and the inner switching radius. Per-particle charges live on
// `ParticleBuffers`. See rq-bfd7004c.
// rq-6bdfdd6d
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

// rq-846bdb8b rq-d340b338
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
        level: AggregateLevel,
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
pub struct CoulombBuilder;

impl PotentialBuilder for CoulombBuilder {
    fn build(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        let Some(coul) = cx.coulomb_config else {
            return Ok(None);
        };
        let params = CoulombParameters::from(coul);
        let max_neighbors = super::max_neighbors_from(cx.neighbor_list_config, cx.particle_count);
        let state = CoulombState::new(
            cx.gpu,
            cx.particle_count,
            params,
            max_neighbors,
            cx.exclusion_list,
        )?;
        Ok(Some(Box::new(state)))
    }

    fn box_clone(&self) -> Box<dyn PotentialBuilder> {
        Box::new(self.clone())
    }
}

// rq-2093594f rq-846bdb8b
#[derive(Debug, Clone)]
pub struct CoulombKernels {
    pub coulomb_pair_force: CudaFunction,
}

impl CoulombKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::COULOMB),
            "coulomb",
            &["coulomb_pair_force"],
        )?;
        Ok(CoulombKernels {
            coulomb_pair_force: get_func(device, "coulomb", "coulomb_pair_force")?,
        })
    }
}
