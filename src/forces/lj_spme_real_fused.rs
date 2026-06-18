// rq-79282483 — Fused LJ + SPME real-space pair-force slot.
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction};
use cudarc::nvrtc::Ptx;

use crate::gpu::device::get_func;
use crate::gpu::{
    GpuContext, GpuError, LennardJonesParameterTable, ParticleBuffers,
    lj_spme_real_fused_pair_force,
};
use crate::kernels;
use crate::pbc::SimulationBox;
use crate::precision::Real;
use crate::timings::{KernelStage, Timings};

use super::neighbor_list::NeighborListError;
use super::spme::SpmeParameters;
use super::topology::{DeviceExclusionList, ExclusionList};
use super::{
    AggregateLevel, ForceFieldContext, ForceFieldError, Potential,
    PotentialBuildContext, PotentialBuilder, SlotOutputView,
};

// rq-0bda299c
#[derive(Debug)]
pub struct LjSpmeRealFusedState {
    #[allow(dead_code)]
    device: Arc<CudaDevice>,
    params: LennardJonesParameterTable,
    exclusions: DeviceExclusionList,
    alpha: Real,
    r_cut_spme_real: Real,
    r_cut_lj: Real,
    particle_count: usize,
    max_neighbors: u32,
}

impl LjSpmeRealFusedState {
    pub fn new(
        gpu: &GpuContext,
        particle_count: usize,
        params: LennardJonesParameterTable,
        r_cut_lj: Real,
        alpha: Real,
        r_cut_spme_real: Real,
        max_neighbors: u32,
        exclusion_list: &ExclusionList,
    ) -> Result<Self, NeighborListError> {
        let device = gpu.device.clone();
        let exclusions = DeviceExclusionList::from_host(&device, exclusion_list)?;
        Ok(LjSpmeRealFusedState {
            device,
            params,
            exclusions,
            alpha,
            r_cut_spme_real,
            r_cut_lj,
            particle_count,
            max_neighbors,
        })
    }
}

impl Potential for LjSpmeRealFusedState {
    fn label(&self) -> &'static str {
        "lj_spme_real_fused"
    }

    fn max_cutoff(&self) -> Option<Real> {
        Some(self.r_cut_lj.max(self.r_cut_spme_real))
    }

    fn compute(
        &mut self,
        buffers: &ParticleBuffers,
        sim_box: &SimulationBox,
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
            .expect("LjSpmeRealFusedState requires a shared neighbor list");
        timings.kernel_start(KernelStage::LJ_SPME_REAL_FUSED_PAIR_FORCE)?;
        lj_spme_real_fused_pair_force(
            buffers,
            &mut output,
            sim_box,
            &self.params,
            self.alpha,
            self.r_cut_spme_real,
            &self.exclusions.atom_excl_offsets,
            &self.exclusions.atom_excl_partners,
            &self.exclusions.atom_excl_lj_scales,
            &self.exclusions.atom_excl_coul_scales,
            &nl.neighbor_list,
            &nl.neighbor_counts,
            self.max_neighbors,
            level,
        )?;
        timings.kernel_stop(KernelStage::LJ_SPME_REAL_FUSED_PAIR_FORCE)?;
        Ok(())
    }
}

// rq-c2a28bda
#[derive(Debug, Clone)]
pub struct LjSpmeRealFusedBuilder;

impl PotentialBuilder for LjSpmeRealFusedBuilder {
    fn build(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        if cx.pair_interactions.is_empty() {
            return Ok(None);
        }
        let Some(spme_cfg) = cx.spme_config else {
            return Ok(None);
        };
        let params = LennardJonesParameterTable::from_config(
            &cx.gpu.device,
            cx.particle_types,
            cx.pair_interactions,
        )?;
        let r_cut_lj = cx
            .pair_interactions
            .iter()
            .map(|p| p.cutoff as Real)
            .fold(0.0, Real::max);
        let spme_params = SpmeParameters::from(spme_cfg);
        let max_neighbors = super::max_neighbors_from(cx.neighbor_list_config, cx.particle_count);
        let state = LjSpmeRealFusedState::new(
            cx.gpu,
            cx.particle_count,
            params,
            r_cut_lj,
            spme_params.alpha,
            spme_params.r_cut_real,
            max_neighbors,
            cx.exclusion_list,
        )?;
        Ok(Some(Box::new(state)))
    }

    fn box_clone(&self) -> Box<dyn PotentialBuilder> {
        Box::new(self.clone())
    }

    fn displaces(&self) -> &'static [&'static str] {
        &["lennard_jones", "spme_real"]
    }
}

// rq-2093594f
#[derive(Debug, Clone)]
pub struct LjSpmeRealFusedKernels {
    pub pair_force_f: CudaFunction,
    pub pair_force_fev: CudaFunction,
}

impl LjSpmeRealFusedKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::LJ_SPME_REAL_FUSED),
            "lj_spme_real_fused",
            &[
                "lj_spme_real_fused_pair_force_f",
                "lj_spme_real_fused_pair_force_fev",
            ],
        )?;
        Ok(LjSpmeRealFusedKernels {
            pair_force_f: get_func(
                device,
                "lj_spme_real_fused",
                "lj_spme_real_fused_pair_force_f",
            )?,
            pair_force_fev: get_func(
                device,
                "lj_spme_real_fused",
                "lj_spme_real_fused_pair_force_fev",
            )?,
        })
    }
}
