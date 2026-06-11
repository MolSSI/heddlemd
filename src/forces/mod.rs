pub mod angle;
pub mod coulomb;
pub mod lj;
pub mod morse;
pub mod neighbor_list;
pub mod spme;
pub mod topology;

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, CudaViewMut};
use cudarc::nvrtc::Ptx;

use crate::gpu::device::get_func;
use crate::gpu::{
    GpuContext, GpuError, Kernels, ParticleBuffers, accumulate_forces,
};
use crate::kernels;
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

// rq-df6d79a1 rq-c4861786
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ForceClass {
    Fast,
    Slow,
}

// rq-81ac7d6a
/// Selects whether a force-evaluation call aggregates only the three force
/// components, or also the per-particle potential-energy and scalar-virial
/// shares. See `rqm/forces/framework.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateLevel {
    ForcesOnly,
    ForcesAndScalars,
}

impl AggregateLevel {
    pub fn includes_scalars(self) -> bool {
        matches!(self, AggregateLevel::ForcesAndScalars)
    }
}

// rq-67ebf3b1
pub trait Potential: std::fmt::Debug + Send {
    fn label(&self) -> &'static str;

    fn max_cutoff(&self) -> Option<f32>;

    fn frequency_class(&self) -> ForceClass {
        ForceClass::Fast
    }

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
        level: AggregateLevel,
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

// rq-559783fe
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
    pub fast_slot_forces_x: CudaSlice<f32>,
    pub fast_slot_forces_y: CudaSlice<f32>,
    pub fast_slot_forces_z: CudaSlice<f32>,
    pub fast_slot_energies: CudaSlice<f32>,
    pub fast_slot_virials: CudaSlice<f32>,
    pub slow_slot_forces_x: CudaSlice<f32>,
    pub slow_slot_forces_y: CudaSlice<f32>,
    pub slow_slot_forces_z: CudaSlice<f32>,
    pub slow_slot_energies: CudaSlice<f32>,
    pub slow_slot_virials: CudaSlice<f32>,
    pub neighbor_list: Option<NeighborListState>,
    /// For each slot in `slots`, in canonical slot order, the row
    /// index of that slot within its class's slot-output buffers
    /// (i.e. the slot's position among same-class slots, also in
    /// canonical order).
    class_row: Vec<u32>,
    num_fast_slots: usize,
    num_slow_slots: usize,
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

        // Map each slot in canonical order to its row index within its
        // class's slot-output buffers. Counts of slots per class drive
        // the per-class buffer lengths.
        let mut class_row: Vec<u32> = Vec::with_capacity(slots.len());
        let mut num_fast_slots: usize = 0;
        let mut num_slow_slots: usize = 0;
        for slot in &slots {
            match slot.frequency_class() {
                ForceClass::Fast => {
                    class_row.push(num_fast_slots as u32);
                    num_fast_slots += 1;
                }
                ForceClass::Slow => {
                    class_row.push(num_slow_slots as u32);
                    num_slow_slots += 1;
                }
            }
        }

        let fast_len = num_fast_slots * particle_count;
        let slow_len = num_slow_slots * particle_count;
        let fast_slot_forces_x = device.alloc_zeros::<f32>(fast_len).map_err(GpuError::from)?;
        let fast_slot_forces_y = device.alloc_zeros::<f32>(fast_len).map_err(GpuError::from)?;
        let fast_slot_forces_z = device.alloc_zeros::<f32>(fast_len).map_err(GpuError::from)?;
        let fast_slot_energies = device.alloc_zeros::<f32>(fast_len).map_err(GpuError::from)?;
        let fast_slot_virials = device.alloc_zeros::<f32>(fast_len).map_err(GpuError::from)?;
        let slow_slot_forces_x = device.alloc_zeros::<f32>(slow_len).map_err(GpuError::from)?;
        let slow_slot_forces_y = device.alloc_zeros::<f32>(slow_len).map_err(GpuError::from)?;
        let slow_slot_forces_z = device.alloc_zeros::<f32>(slow_len).map_err(GpuError::from)?;
        let slow_slot_energies = device.alloc_zeros::<f32>(slow_len).map_err(GpuError::from)?;
        let slow_slot_virials = device.alloc_zeros::<f32>(slow_len).map_err(GpuError::from)?;

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
            fast_slot_forces_x,
            fast_slot_forces_y,
            fast_slot_forces_z,
            fast_slot_energies,
            fast_slot_virials,
            slow_slot_forces_x,
            slow_slot_forces_y,
            slow_slot_forces_z,
            slow_slot_energies,
            slow_slot_virials,
            neighbor_list,
            class_row,
            num_fast_slots,
            num_slow_slots,
            particle_count,
        })
    }

    // rq-3579df3b
    pub fn step(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        timings: &mut Timings,
        level: AggregateLevel,
    ) -> Result<(), ForceFieldError> {
        self.run(None, buffers, sim_box, timings, level)
    }

    // rq-be1eb548
    pub fn step_class(
        &mut self,
        class: ForceClass,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        timings: &mut Timings,
        level: AggregateLevel,
    ) -> Result<(), ForceFieldError> {
        // No-op when the class has no slots: nothing to recompute and
        // the existing combined total in ParticleBuffers.forces_* is
        // already current.
        let class_count = match class {
            ForceClass::Fast => self.num_fast_slots,
            ForceClass::Slow => self.num_slow_slots,
        };
        if class_count == 0 {
            return Ok(());
        }
        self.run(Some(class), buffers, sim_box, timings, level)
    }

    fn run(
        &mut self,
        class_filter: Option<ForceClass>,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        timings: &mut Timings,
        level: AggregateLevel,
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
            if let Some(c) = class_filter {
                if slot.frequency_class() != c {
                    continue;
                }
            }
            let cx = ForceFieldContext {
                neighbor_list: nl_ref,
                buffers: &*buffers,
                sim_box,
            };
            slot.contribute(buffers, sim_box, &cx, timings)?;
        }

        // Per-slot reductions. Each slot writes into the row of its
        // class's slot-output buffers indexed by class_row[slot_index].
        let class_row = &self.class_row;
        let slots = &mut self.slots;
        let fast_x = &mut self.fast_slot_forces_x;
        let fast_y = &mut self.fast_slot_forces_y;
        let fast_z = &mut self.fast_slot_forces_z;
        let fast_e = &mut self.fast_slot_energies;
        let fast_w = &mut self.fast_slot_virials;
        let slow_x = &mut self.slow_slot_forces_x;
        let slow_y = &mut self.slow_slot_forces_y;
        let slow_z = &mut self.slow_slot_forces_z;
        let slow_e = &mut self.slow_slot_energies;
        let slow_w = &mut self.slow_slot_virials;
        for (i, slot) in slots.iter_mut().enumerate() {
            let slot_class = slot.frequency_class();
            if let Some(c) = class_filter {
                if slot_class != c {
                    continue;
                }
            }
            let row = class_row[i] as usize;
            let start = row * n;
            let end = (row + 1) * n;
            let view = match slot_class {
                ForceClass::Fast => SlotOutputView {
                    force_x: fast_x.slice_mut(start..end),
                    force_y: fast_y.slice_mut(start..end),
                    force_z: fast_z.slice_mut(start..end),
                    energy: fast_e.slice_mut(start..end),
                    virial: fast_w.slice_mut(start..end),
                },
                ForceClass::Slow => SlotOutputView {
                    force_x: slow_x.slice_mut(start..end),
                    force_y: slow_y.slice_mut(start..end),
                    force_z: slow_z.slice_mut(start..end),
                    energy: slow_e.slice_mut(start..end),
                    virial: slow_w.slice_mut(start..end),
                },
            };
            let cx = ForceFieldContext {
                neighbor_list: nl_ref,
                buffers: &*buffers,
                sim_box,
            };
            slot.reduce(view, &cx, timings, level)?;
        }

        timings.kernel_start(KernelStage::ACCUMULATE_FORCES)?;
        accumulate_forces(
            buffers,
            &self.fast_slot_forces_x,
            &self.fast_slot_forces_y,
            &self.fast_slot_forces_z,
            &self.fast_slot_energies,
            &self.fast_slot_virials,
            self.num_fast_slots as u32,
            &self.slow_slot_forces_x,
            &self.slow_slot_forces_y,
            &self.slow_slot_forces_z,
            &self.slow_slot_energies,
            &self.slow_slot_virials,
            self.num_slow_slots as u32,
        )?;
        timings.kernel_stop(KernelStage::ACCUMULATE_FORCES)?;
        Ok(())
    }
}

// rq-2093594f
#[derive(Debug, Clone)]
pub struct ForcesKernels {
    pub accumulate_forces: CudaFunction,
}

impl ForcesKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::FORCES),
            "forces",
            &["accumulate_forces"],
        )?;
        Ok(ForcesKernels {
            accumulate_forces: get_func(device, "forces", "accumulate_forces")?,
        })
    }
}
