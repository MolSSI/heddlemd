pub mod angle;
pub mod coulomb;
pub mod lj;
pub mod morse;
pub mod neighbor_list;
pub mod spme;
pub mod topology;

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, CudaViewMut};

use crate::gpu::{
    GpuContext, GpuError, Kernels, ParticleBuffers, accumulate_forces,
};
use crate::io::config::{
    AngleTypeConfig, BondTypeConfig, CoulombConfig, NeighborListConfig, PairInteractionConfig,
    ParticleTypeConfig, SpmeConfig,
};
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings, TimingsError};

pub use angle::{HarmonicAngleBuilder, HarmonicAngleState};
pub use coulomb::{CoulombBuilder, CoulombParameters, CoulombState};
pub use spme::{
    SpmeError, SpmeParameters, SpmeReciprocalGrid, SpmeReciprocalState, SpmeRealSpaceState,
    SpmeRealBuilder, SpmeReciprocalBuilder,
};
pub use lj::{LennardJonesBuilder, LennardJonesState};
pub use morse::{MorseBondedBuilder, MorseBondedState};
pub use topology::{
    Angle, AngleList, Bond, BondList, ConstraintGroup, ConstraintList,
    DeviceExclusionList, Exclusion, ExclusionList, GroupConstraint, TopologyFileError,
    load_topology_file,
};
pub use neighbor_list::{
    CellListData, NeighborListError, NeighborListMode, NeighborListState,
};

// rq-67ebf3b1
pub trait Potential: std::fmt::Debug + Send {
    fn label(&self) -> &'static str;

    fn max_cutoff(&self) -> Option<f32>;

    fn contribute(
        &mut self,
        buffers: &ParticleBuffers,
        sim_box: &SimulationBox,
        cx: &ForceFieldContext<'_>,
        timings: &mut Timings,
    ) -> Result<(), ForceFieldError>;

    fn reduce(
        &mut self,
        output: SlotOutputView<'_>,
        cx: &ForceFieldContext<'_>,
        timings: &mut Timings,
    ) -> Result<(), ForceFieldError>;
}

// rq-304b191b
pub struct SlotOutputView<'a> {
    pub force_x: CudaViewMut<'a, f32>,
    pub force_y: CudaViewMut<'a, f32>,
    pub force_z: CudaViewMut<'a, f32>,
    pub energy: CudaViewMut<'a, f32>,
    pub virial: CudaViewMut<'a, f32>,
}

// rq-9f7d4b40
pub struct ForceFieldContext<'a> {
    pub neighbor_list: Option<&'a NeighborListState>,
    pub buffers: &'a ParticleBuffers,
    pub sim_box: &'a SimulationBox,
}

// rq-a2e20b02 rq-e1ceb5c0 rq-6cf916af
#[derive(Debug, thiserror::Error)]
pub enum ForceFieldError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("{0}")]
    Timings(#[from] TimingsError),
    #[error("{0}")]
    NeighborList(#[from] NeighborListError),
    #[error("duplicate potential slot label `{0}`")]
    DuplicateLabel(&'static str),
}

// rq-d116af5f
pub struct PotentialBuildContext<'a> {
    pub gpu: &'a GpuContext,
    pub particle_count: usize,
    pub sim_box: &'a SimulationBox,
    pub particle_types: &'a [ParticleTypeConfig],
    pub pair_interactions: &'a [PairInteractionConfig],
    pub bond_types: &'a [BondTypeConfig],
    pub angle_types: &'a [AngleTypeConfig],
    pub coulomb_config: Option<&'a CoulombConfig>,
    pub spme_config: Option<&'a SpmeConfig>,
    pub charges: &'a [f32],
    pub bond_list: &'a BondList,
    pub angle_list: &'a AngleList,
    pub exclusion_list: &'a ExclusionList,
    pub neighbor_list_config: &'a NeighborListConfig,
}

// rq-e8550f96
pub trait PotentialBuilder: std::fmt::Debug + Send + Sync {
    fn build(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError>;
}

// rq-50f0a96a
#[derive(Debug)]
pub struct PotentialRegistry {
    pub builders: Vec<Box<dyn PotentialBuilder>>,
}

impl PotentialRegistry {
    pub fn new() -> Self {
        PotentialRegistry { builders: Vec::new() }
    }

    pub fn with_builtins() -> Self {
        PotentialRegistry {
            builders: vec![
                Box::new(LennardJonesBuilder),
                Box::new(CoulombBuilder),
                Box::new(SpmeRealBuilder),
                Box::new(SpmeReciprocalBuilder),
                Box::new(MorseBondedBuilder),
                Box::new(HarmonicAngleBuilder),
            ],
        }
    }

    pub fn register(&mut self, builder: Box<dyn PotentialBuilder>) {
        self.builders.push(builder);
    }
}

impl Default for PotentialRegistry {
    fn default() -> Self {
        PotentialRegistry::with_builtins()
    }
}

pub(crate) fn max_neighbors_from(cfg: &NeighborListConfig, particle_count: usize) -> u32 {
    match cfg {
        NeighborListConfig::AllPairs => particle_count as u32,
        NeighborListConfig::CellList { max_neighbors, .. } => *max_neighbors,
    }
}

// rq-684a29f1
#[derive(Debug)]
pub struct ForceField {
    pub device: Arc<CudaDevice>,
    pub kernels: Arc<Kernels>,
    pub slots: Vec<Box<dyn Potential>>,
    pub slot_forces_x: CudaSlice<f32>,
    pub slot_forces_y: CudaSlice<f32>,
    pub slot_forces_z: CudaSlice<f32>,
    pub slot_energies: CudaSlice<f32>,
    pub slot_virials: CudaSlice<f32>,
    pub neighbor_list: Option<NeighborListState>,
    particle_count: usize,
}

impl ForceField {
    // rq-79938dbf
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        registry: &PotentialRegistry,
        gpu: &GpuContext,
        particle_count: usize,
        sim_box: &SimulationBox,
        particle_types: &[ParticleTypeConfig],
        pair_interactions: &[PairInteractionConfig],
        bond_types: &[BondTypeConfig],
        angle_types: &[AngleTypeConfig],
        coulomb_config: Option<&CoulombConfig>,
        spme_config: Option<&SpmeConfig>,
        charges: &[f32],
        bond_list: &BondList,
        angle_list: &AngleList,
        exclusion_list: &ExclusionList,
        neighbor_list_config: &NeighborListConfig,
    ) -> Result<Self, ForceFieldError> {
        let device = gpu.device.clone();
        let kernels = gpu.kernels.clone();

        let cx = PotentialBuildContext {
            gpu,
            particle_count,
            sim_box,
            particle_types,
            pair_interactions,
            bond_types,
            angle_types,
            coulomb_config,
            spme_config,
            charges,
            bond_list,
            angle_list,
            exclusion_list,
            neighbor_list_config,
        };

        let mut slots: Vec<Box<dyn Potential>> = Vec::new();
        for builder in &registry.builders {
            if let Some(slot) = builder.build(&cx)? {
                slots.push(slot);
            }
        }

        for i in 0..slots.len() {
            for j in (i + 1)..slots.len() {
                if slots[i].label() == slots[j].label() {
                    return Err(ForceFieldError::DuplicateLabel(slots[i].label()));
                }
            }
        }

        let flat_len = slots.len() * particle_count;
        let slot_forces_x = device.alloc_zeros::<f32>(flat_len).map_err(GpuError::from)?;
        let slot_forces_y = device.alloc_zeros::<f32>(flat_len).map_err(GpuError::from)?;
        let slot_forces_z = device.alloc_zeros::<f32>(flat_len).map_err(GpuError::from)?;
        let slot_energies = device.alloc_zeros::<f32>(flat_len).map_err(GpuError::from)?;
        let slot_virials = device.alloc_zeros::<f32>(flat_len).map_err(GpuError::from)?;

        // Build the shared NeighborListState when any slot reports a cutoff.
        let aggregated_cutoff: Option<f32> = slots
            .iter()
            .filter_map(|s| s.max_cutoff())
            .fold(None::<f32>, |acc, c| Some(acc.map_or(c, |a| a.max(c))));
        let neighbor_list = if let Some(r_cut) = aggregated_cutoff {
            match neighbor_list_config {
                NeighborListConfig::CellList { max_neighbors, r_skin } => Some(
                    NeighborListState::new_cell_list(
                        gpu,
                        sim_box,
                        particle_count,
                        r_cut,
                        *max_neighbors,
                        *r_skin as f32,
                    )?,
                ),
                NeighborListConfig::AllPairs => Some(NeighborListState::new_trivial(
                    gpu,
                    sim_box,
                    particle_count,
                )?),
            }
        } else {
            None
        };

        Ok(ForceField {
            device,
            kernels,
            slots,
            slot_forces_x,
            slot_forces_y,
            slot_forces_z,
            slot_energies,
            slot_virials,
            neighbor_list,
            particle_count,
        })
    }

    // rq-3579df3b
    pub fn step(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        let n = self.particle_count;
        if n == 0 {
            return Ok(());
        }

        // Shared neighbor-list update (no-op in Trivial mode and when absent).
        if let Some(nl) = self.neighbor_list.as_mut() {
            nl.pre_step(sim_box, buffers, timings)?;
        }

        let nl_ref = self.neighbor_list.as_ref();
        for slot in self.slots.iter_mut() {
            let cx = ForceFieldContext {
                neighbor_list: nl_ref,
                buffers: &*buffers,
                sim_box,
            };
            slot.contribute(buffers, sim_box, &cx, timings)?;
        }

        let num_slots = self.slots.len();
        let slots = &mut self.slots;
        let sfx = &mut self.slot_forces_x;
        let sfy = &mut self.slot_forces_y;
        let sfz = &mut self.slot_forces_z;
        let sen = &mut self.slot_energies;
        let svi = &mut self.slot_virials;
        for k in 0..num_slots {
            let start = k * n;
            let end = (k + 1) * n;
            let view = SlotOutputView {
                force_x: sfx.slice_mut(start..end),
                force_y: sfy.slice_mut(start..end),
                force_z: sfz.slice_mut(start..end),
                energy: sen.slice_mut(start..end),
                virial: svi.slice_mut(start..end),
            };
            let cx = ForceFieldContext {
                neighbor_list: nl_ref,
                buffers: &*buffers,
                sim_box,
            };
            slots[k].reduce(view, &cx, timings)?;
        }

        timings.kernel_start(KernelStage::ACCUMULATE_FORCES)?;
        accumulate_forces(
            buffers,
            &self.slot_forces_x,
            &self.slot_forces_y,
            &self.slot_forces_z,
            &self.slot_energies,
            &self.slot_virials,
            num_slots as u32,
        )?;
        timings.kernel_stop(KernelStage::ACCUMULATE_FORCES)?;
        Ok(())
    }
}
