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
    GpuContext, GpuError, Kernels, ParticleBuffers, combine_class_totals,
};
use crate::kernels;
use crate::io::config::{
    AngleTypeConfig, BondTypeConfig, CoulombConfig, NeighborListConfig, PairInteractionConfig,
    ParticleTypeConfig, SpmeConfig,
};
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings, TimingsError};
use crate::precision::Real;

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

    fn max_cutoff(&self) -> Option<Real>;

    fn frequency_class(&self) -> ForceClass {
        ForceClass::Fast
    }

    /// `true` iff `compute` consists of pure CUDA kernel launches on
    /// the device's default stream with no host-side state mutation
    /// and no use of secondary streams. Determines whether phases
    /// using this potential run under CUDA graph mode; see
    /// `cuda-graphs.md`. Default `true`. Potentials that launch
    /// kernels on streams other than the default (e.g. SPME
    /// reciprocal's `recip_stream`) override to `false`: work on
    /// uncaptured streams executes immediately and is not part of the
    /// captured graph, so replays would produce stale forces.
    fn graph_compatible(&self) -> bool {
        true
    }

    fn compute(
        &mut self,
        buffers: &ParticleBuffers,
        sim_box: &SimulationBox,
        output: SlotOutputView<'_>,
        cx: &ForceFieldContext<'_>,
        timings: &mut Timings,
        level: AggregateLevel,
    ) -> Result<(), ForceFieldError>;
}

// rq-304b191b
pub struct SlotOutputView<'a> {
    pub force_x: CudaViewMut<'a, Real>,
    pub force_y: CudaViewMut<'a, Real>,
    pub force_z: CudaViewMut<'a, Real>,
    pub energy: CudaViewMut<'a, Real>,
    pub virial: CudaViewMut<'a, Real>,
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
    pub charges: &'a [Real],
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

    fn box_clone(&self) -> Box<dyn PotentialBuilder>;
}

// rq-50f0a96a
#[derive(Debug)]
pub struct PotentialRegistry {
    pub builders: Vec<Box<dyn PotentialBuilder>>,
}

impl Clone for PotentialRegistry {
    fn clone(&self) -> Self {
        PotentialRegistry {
            builders: self.builders.iter().map(|b| b.box_clone()).collect(),
        }
    }
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
    pub fast_total_forces_x: CudaSlice<Real>,
    pub fast_total_forces_y: CudaSlice<Real>,
    pub fast_total_forces_z: CudaSlice<Real>,
    pub fast_total_potential_energies: CudaSlice<Real>,
    pub fast_total_virials: CudaSlice<Real>,
    pub slow_total_forces_x: CudaSlice<Real>,
    pub slow_total_forces_y: CudaSlice<Real>,
    pub slow_total_forces_z: CudaSlice<Real>,
    pub slow_total_potential_energies: CudaSlice<Real>,
    pub slow_total_virials: CudaSlice<Real>,
    pub neighbor_list: Option<NeighborListState>,
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
        charges: &[Real],
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

        // Count slots per class; each class's accumulators are sized
        // particle_count regardless of slot count.
        let mut num_fast_slots: usize = 0;
        let mut num_slow_slots: usize = 0;
        for slot in &slots {
            match slot.frequency_class() {
                ForceClass::Fast => num_fast_slots += 1,
                ForceClass::Slow => num_slow_slots += 1,
            }
        }

        let n = particle_count;
        let fast_total_forces_x = device.alloc_zeros::<Real>(n).map_err(GpuError::from)?;
        let fast_total_forces_y = device.alloc_zeros::<Real>(n).map_err(GpuError::from)?;
        let fast_total_forces_z = device.alloc_zeros::<Real>(n).map_err(GpuError::from)?;
        let fast_total_potential_energies =
            device.alloc_zeros::<Real>(n).map_err(GpuError::from)?;
        let fast_total_virials = device.alloc_zeros::<Real>(n).map_err(GpuError::from)?;
        let slow_total_forces_x = device.alloc_zeros::<Real>(n).map_err(GpuError::from)?;
        let slow_total_forces_y = device.alloc_zeros::<Real>(n).map_err(GpuError::from)?;
        let slow_total_forces_z = device.alloc_zeros::<Real>(n).map_err(GpuError::from)?;
        let slow_total_potential_energies =
            device.alloc_zeros::<Real>(n).map_err(GpuError::from)?;
        let slow_total_virials = device.alloc_zeros::<Real>(n).map_err(GpuError::from)?;

        // Build the shared NeighborListState when any slot reports a cutoff.
        let aggregated_cutoff: Option<Real> = slots
            .iter()
            .filter_map(|s| s.max_cutoff())
            .fold(None::<Real>, |acc, c| Some(acc.map_or(c, |a| a.max(c))));
        let neighbor_list = if let Some(r_cut) = aggregated_cutoff {
            match neighbor_list_config {
                NeighborListConfig::CellList { max_neighbors, r_skin } => Some(
                    NeighborListState::new_cell_list(
                        gpu,
                        sim_box,
                        particle_count,
                        r_cut,
                        *max_neighbors,
                        *r_skin as Real,
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
            fast_total_forces_x,
            fast_total_forces_y,
            fast_total_forces_z,
            fast_total_potential_energies,
            fast_total_virials,
            slow_total_forces_x,
            slow_total_forces_y,
            slow_total_forces_z,
            slow_total_potential_energies,
            slow_total_virials,
            neighbor_list,
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
        self.run(None, buffers, sim_box, timings, level, true)
    }

    /// Same per-slot compute path as `step`, but skips the internal
    /// `NeighborListState::pre_step` call. Used inside CUDA graph
    /// capture and inside the batched-replay loop, where the runner
    /// calls `nl.pre_step` at every batch boundary instead. See
    /// `cuda-graphs.md`.
    pub fn step_no_neighbor_check(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        timings: &mut Timings,
        level: AggregateLevel,
    ) -> Result<(), ForceFieldError> {
        self.run(None, buffers, sim_box, timings, level, false)
    }

    /// Runs `NeighborListState::pre_step` standalone — used by the
    /// CUDA graph batched-replay loop, which moves the per-step
    /// displacement check / rebuild out of `force_field.step` and
    /// into the host loop between graph launches.
    pub fn run_neighbor_pre_step(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        if let Some(nl) = self.neighbor_list.as_mut() {
            nl.pre_step(sim_box, buffers, timings)?;
        }
        Ok(())
    }

    /// `true` iff every potential slot configured in this force
    /// field reports `Potential::graph_compatible == true`. Used by
    /// the runner to decide whether a phase is eligible for CUDA
    /// graph capture; see `cuda-graphs.md`.
    pub fn graph_compatible(&self) -> bool {
        self.slots.iter().all(|slot| slot.graph_compatible())
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
        self.run(Some(class), buffers, sim_box, timings, level, true)
    }

    /// Per-class variant of `step_no_neighbor_check`.
    pub fn step_class_no_neighbor_check(
        &mut self,
        class: ForceClass,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        timings: &mut Timings,
        level: AggregateLevel,
    ) -> Result<(), ForceFieldError> {
        let class_count = match class {
            ForceClass::Fast => self.num_fast_slots,
            ForceClass::Slow => self.num_slow_slots,
        };
        if class_count == 0 {
            return Ok(());
        }
        self.run(Some(class), buffers, sim_box, timings, level, false)
    }

    fn run(
        &mut self,
        class_filter: Option<ForceClass>,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        timings: &mut Timings,
        level: AggregateLevel,
        run_neighbor_pre_step: bool,
    ) -> Result<(), ForceFieldError> {
        let n = self.particle_count;
        if n == 0 {
            return Ok(());
        }

        // Shared neighbor-list update (no-op in Trivial mode and when absent).
        if run_neighbor_pre_step {
            if let Some(nl) = self.neighbor_list.as_mut() {
                nl.pre_step(sim_box, buffers, timings)?;
            }
        }

        let write_scalars = matches!(level, AggregateLevel::ForcesAndScalars);

        // Step 3: zero the class accumulators for each class that
        // will be re-evaluated this call. ForcesOnly zeros only the
        // three force buffers; ForcesAndScalars zeros all five.
        timings.kernel_start(KernelStage::CLASS_ACCUMULATOR_MEMSET)?;
        let evaluating_fast = match class_filter {
            None => self.num_fast_slots > 0,
            Some(c) => c == ForceClass::Fast && self.num_fast_slots > 0,
        };
        let evaluating_slow = match class_filter {
            None => self.num_slow_slots > 0,
            Some(c) => c == ForceClass::Slow && self.num_slow_slots > 0,
        };
        if evaluating_fast {
            self.device
                .memset_zeros(&mut self.fast_total_forces_x)
                .map_err(GpuError::from)?;
            self.device
                .memset_zeros(&mut self.fast_total_forces_y)
                .map_err(GpuError::from)?;
            self.device
                .memset_zeros(&mut self.fast_total_forces_z)
                .map_err(GpuError::from)?;
            if write_scalars {
                self.device
                    .memset_zeros(&mut self.fast_total_potential_energies)
                    .map_err(GpuError::from)?;
                self.device
                    .memset_zeros(&mut self.fast_total_virials)
                    .map_err(GpuError::from)?;
            }
        }
        if evaluating_slow {
            self.device
                .memset_zeros(&mut self.slow_total_forces_x)
                .map_err(GpuError::from)?;
            self.device
                .memset_zeros(&mut self.slow_total_forces_y)
                .map_err(GpuError::from)?;
            self.device
                .memset_zeros(&mut self.slow_total_forces_z)
                .map_err(GpuError::from)?;
            if write_scalars {
                self.device
                    .memset_zeros(&mut self.slow_total_potential_energies)
                    .map_err(GpuError::from)?;
                self.device
                    .memset_zeros(&mut self.slow_total_virials)
                    .map_err(GpuError::from)?;
            }
        }
        timings.kernel_stop(KernelStage::CLASS_ACCUMULATOR_MEMSET)?;

        let nl_ref = self.neighbor_list.as_ref();
        // Step 4: per-slot compute. Each slot ADDs into its class
        // accumulator via the SlotOutputView.
        let slots = &mut self.slots;
        let fast_x = &mut self.fast_total_forces_x;
        let fast_y = &mut self.fast_total_forces_y;
        let fast_z = &mut self.fast_total_forces_z;
        let fast_e = &mut self.fast_total_potential_energies;
        let fast_w = &mut self.fast_total_virials;
        let slow_x = &mut self.slow_total_forces_x;
        let slow_y = &mut self.slow_total_forces_y;
        let slow_z = &mut self.slow_total_forces_z;
        let slow_e = &mut self.slow_total_potential_energies;
        let slow_w = &mut self.slow_total_virials;
        for slot in slots.iter_mut() {
            let slot_class = slot.frequency_class();
            if let Some(c) = class_filter {
                if slot_class != c {
                    continue;
                }
            }
            let view = match slot_class {
                ForceClass::Fast => SlotOutputView {
                    force_x: fast_x.slice_mut(..),
                    force_y: fast_y.slice_mut(..),
                    force_z: fast_z.slice_mut(..),
                    energy: fast_e.slice_mut(..),
                    virial: fast_w.slice_mut(..),
                },
                ForceClass::Slow => SlotOutputView {
                    force_x: slow_x.slice_mut(..),
                    force_y: slow_y.slice_mut(..),
                    force_z: slow_z.slice_mut(..),
                    energy: slow_e.slice_mut(..),
                    virial: slow_w.slice_mut(..),
                },
            };
            let cx = ForceFieldContext {
                neighbor_list: nl_ref,
                buffers: &*buffers,
                sim_box,
            };
            slot.compute(buffers, sim_box, view, &cx, timings, level)?;
        }

        // Step 5: small class-combine kernel sums fast + slow into
        // the ParticleBuffers totals.
        timings.kernel_start(KernelStage::COMBINE_CLASS_TOTALS)?;
        combine_class_totals(
            buffers,
            &self.fast_total_forces_x,
            &self.fast_total_forces_y,
            &self.fast_total_forces_z,
            &self.fast_total_potential_energies,
            &self.fast_total_virials,
            &self.slow_total_forces_x,
            &self.slow_total_forces_y,
            &self.slow_total_forces_z,
            &self.slow_total_potential_energies,
            &self.slow_total_virials,
        )?;
        timings.kernel_stop(KernelStage::COMBINE_CLASS_TOTALS)?;
        Ok(())
    }
}

// rq-2093594f
#[derive(Debug, Clone)]
pub struct ForcesKernels {
    pub combine_class_totals: CudaFunction,
}

impl ForcesKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::FORCES),
            "forces",
            &["combine_class_totals"],
        )?;
        Ok(ForcesKernels {
            combine_class_totals: get_func(device, "forces", "combine_class_totals")?,
        })
    }
}
