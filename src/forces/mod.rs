pub mod angle;
pub mod coulomb;
pub mod jit_composed;
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
use crate::registry::{Builtins, Registry};
use crate::io::config::{
    AngleTypeConfig, BondTypeConfig, CoulombConfig, NeighborListConfig, PairInteractionConfig,
    ParticleTypeConfig, SpmeConfig,
};
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings, TimingsError};
use crate::precision::Real;

pub use angle::{HarmonicAngleBuilder, HarmonicAngleState};
pub use coulomb::{CoulombBuilder, CoulombParameters, CoulombState};
pub use jit_composed::{
    AngleForceFragment, AngleScratchView, ArgKind, BondedForceFragment, BondedScratchView,
    CutoffHandling, ElemTy, ForceLaunchBuilder, ForceLaunchContext, JitComposedAngleForce,
    JitComposedBondedForce, JitComposedPairForce, JitComposedPostForcePerParticle, KernelArg,
    KernelArgBinder, KernelArgSchema, KernelArgType, KernelElem, PairForceBindContext,
    PairForceFragment, PerParticleFragment, PostForceBindContext,
    set_jit_fast_math,
};
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
    CellListData, NeighborListError, NeighborListMode, NeighborListState, PreStepOutcome,
    all_pairs_tile_capacity, default_interacting_tiles_capacity,
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

    /// Declare whether this slot contributes to a JIT-composed kernel
    /// and, if so, in which shape. Returns `None` (the default) from a
    /// slot that runs only through `compute` (the slow-class SPME
    /// reciprocal slot, and any slot not JIT-composed). A participating
    /// slot returns `Some(JitParticipant::PairForce(self))`,
    /// `Some(JitParticipant::Bonded(self))`, or
    /// `Some(JitParticipant::Angle(self))`, implementing the matching
    /// capability trait. Because the return is a single enum, a slot
    /// participates in at most one shape by construction. See
    /// `rqm/forces/jit-composed-pair-force.md` and
    /// `rqm/forces/jit-composed-intramolecular.md`.
    fn jit_participant(&self) -> Option<JitParticipant<'_>> {
        None
    }
}

// rq-8571bd3e
/// Single-shape tag a slot returns from [`Potential::jit_participant`]
/// to declare its JIT-composed contribution. Each variant borrows the
/// slot itself as the matching capability trait object.
pub enum JitParticipant<'a> {
    PairForce(&'a dyn PairForcePotential),
    Bonded(&'a dyn BondedPotential),
    Angle(&'a dyn AnglePotential),
}

// rq-e533174d
/// Capability trait a pair-force slot implements to contribute to the
/// JIT-composed pair-force kernel. Carries both the slot's source
/// fragment and its launch-time argument binding, so a slot cannot
/// provide one without the other.
pub trait PairForcePotential {
    /// The slot's CUDA source fragment, computed from the build inputs
    /// at construction. Collected once per participant at
    /// `ForceField::new`.
    fn pair_force_fragment(&self) -> PairForceFragment;

    /// Push the slot's parameter buffers and scalars onto `builder`
    /// through a `KernelArgBinder` over the slot's schema, in the order
    /// the fragment's `entry_point_args` declares them.
    fn bind_pair_force_args(
        &self,
        ctx: &PairForceBindContext<'_>,
        builder: &mut ForceLaunchBuilder,
    );
}

// rq-d7ddc1ac
/// Capability trait a bonded slot implements, carrying the slot's
/// fragment, its per-bond scratch view, and its argument binding.
pub trait BondedPotential {
    fn bonded_force_fragment(&self) -> BondedForceFragment;
    fn bonded_scratch(&self) -> BondedScratchView<'_>;
    fn bind_bonded_force_args(
        &self,
        ctx: &ForceLaunchContext<'_>,
        builder: &mut ForceLaunchBuilder,
    );
}

// rq-da327920
/// Capability trait an angle slot implements, carrying the slot's
/// fragment, its per-angle scratch view, and its argument binding.
pub trait AnglePotential {
    fn angle_force_fragment(&self) -> AngleForceFragment;
    fn angle_scratch(&self) -> AngleScratchView<'_>;
    fn bind_angle_force_args(
        &self,
        ctx: &ForceLaunchContext<'_>,
        builder: &mut ForceLaunchBuilder,
    );
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
    #[error("multiple built slots claim to displace `{label}`: {by:?}")]
    DisplaceConflict {
        label: &'static str,
        by: Vec<&'static str>,
    },
    #[error("JIT-composed kernel failed to compile: {log}")]
    FragmentCompileFailed { log: String },
    #[error("JIT-composed kernel failed to load: {0}")]
    FragmentLoadFailed(GpuError),
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
//
// `PotentialBuilder` carries no `KindedBuilder` bound: potentials are
// activated compositionally by configuration presence, not selected by
// a `kind` key, so `PotentialRegistry` has no `lookup`. See
// `rqm/registry-framework.md`.
pub trait PotentialBuilder:
    PotentialBuilderClone + std::fmt::Debug + Send + Sync
{
    fn build(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError>;

    fn displaces(&self) -> &'static [&'static str] {
        &[]
    }
}

// rq-50f0a96a
pub type PotentialRegistry = Registry<dyn PotentialBuilder>;

impl Builtins for dyn PotentialBuilder {
    fn builtins() -> Vec<Box<dyn PotentialBuilder>> {
        vec![
            Box::new(LennardJonesBuilder),
            Box::new(CoulombBuilder),
            Box::new(SpmeRealBuilder),
            Box::new(SpmeReciprocalBuilder),
            Box::new(MorseBondedBuilder),
            Box::new(HarmonicAngleBuilder),
        ]
    }
}

crate::registry_builder_clone!(pub PotentialBuilderClone for PotentialBuilder);

pub(crate) fn max_neighbors_from(cfg: &NeighborListConfig, particle_count: usize) -> u32 {
    // The packed-neighbour pair-force pipeline (see
    // `rqm/forces/packed-neighbour-pair-force.md`) sizes its entry
    // list at runtime via overflow-driven growth, so no user-supplied
    // per-atom cap exists for cell-list mode. Per-particle padded
    // structures kept around for legacy callers fall back to a fixed
    // default in cell-list mode and to the all-pairs upper bound in
    // trivial mode.
    match cfg {
        NeighborListConfig::AllPairs => particle_count as u32,
        NeighborListConfig::CellList { .. } => LEGACY_FALLBACK_MAX_NEIGHBORS,
    }
}

pub(crate) const LEGACY_FALLBACK_MAX_NEIGHBORS: u32 = 1024;

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
    /// Fixed-point per-particle accumulators for fast-class pair-force
    /// slots (see `rqm/forces/packed-neighbour-pair-force.md`). Scale
    /// `2^32`; interpreted as `i64` two's-complement.
    pub fast_total_forces_fp_x: CudaSlice<u64>,
    pub fast_total_forces_fp_y: CudaSlice<u64>,
    pub fast_total_forces_fp_z: CudaSlice<u64>,
    pub fast_total_potential_energies_fp: CudaSlice<u64>,
    pub fast_total_virials_fp: CudaSlice<u64>,
    pub neighbor_list: Option<NeighborListState>,
    /// JIT-composed pair-force kernel, built when at least one
    /// fast-class pair-force slot is active. `None` when no
    /// fast-class pair-force slot is configured (zero-slot ForceField,
    /// or ForceField with only bonded / angle / slow slots).
    pub jit_composed: Option<JitComposedPairForce>,
    /// Flat `(atom_i, atom_j)` pairs for every canonical exclusion in
    /// the topology, interleaved as `[i0, j0, i1, j1, …]`. Built once
    /// at `ForceField::new` from `ExclusionList.entries`, never
    /// re-uploaded. Consumed by the per-pair JIT correction kernel
    /// (one thread per pair). Length `0` when the topology has no
    /// exclusions. `excluded_pair_count == excluded_pair_atoms.len() / 2`.
    pub excluded_pair_atoms: CudaSlice<u32>,
    /// Number of canonical exclusion pairs.
    pub excluded_pair_count: u32,
    /// Indices into `slots` of fast-class pair-force slots that
    /// participate in the JIT-composed kernel. The framework bypasses
    /// these slots' `Potential::compute` at step time and instead
    /// launches the composed kernel once with each slot's
    /// `bind_pair_force_args` having pushed its parameters in canonical
    /// slot order.
    jit_slot_indices: Vec<usize>,
    /// Maximum `max_neighbors` across the participating JIT pair-force
    /// slots. The composed kernel reads only one `max_neighbors`
    /// scalar at launch and uses it to compute the per-particle
    /// `neighbor_list` row offset; every JIT pair-force slot in this
    /// codebase resolves the same value from the shared
    /// `NeighborListState`, but the field is cached here to avoid a
    /// downcast at launch time.
    jit_max_neighbors: u32,
    /// JIT-composed bonded module, built when at least one fast-class
    /// bonded slot is active. The per-step pipeline launches one
    /// entry point per slot from this module before the slot's
    /// per-atom reduction.
    pub jit_composed_bonded: Option<JitComposedBondedForce>,
    /// Indices into `slots` of fast-class bonded slots that
    /// participate in the JIT-composed bonded module, in canonical
    /// slot order. The index within this `Vec` matches the entry-
    /// point index used by `JitComposedBondedForce::launch_slot`.
    jit_bonded_slot_indices: Vec<usize>,
    /// JIT-composed angle module, parallel to `jit_composed_bonded`.
    pub jit_composed_angle: Option<JitComposedAngleForce>,
    jit_angle_slot_indices: Vec<usize>,
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

        // For each surviving builder, collect: (slot, displaces). Each
        // slot declares its own JIT participation and carries its own
        // fragment via `Potential::jit_participant`; the builder no
        // longer produces fragments.
        type BuilderEntry = (Box<dyn Potential>, &'static [&'static str]);
        let mut built: Vec<BuilderEntry> = Vec::new();
        for builder in registry.builders() {
            if let Some(slot) = builder.build(&cx)? {
                built.push((slot, builder.displaces()));
            }
        }

        for i in 0..built.len() {
            for j in (i + 1)..built.len() {
                if built[i].0.label() == built[j].0.label() {
                    return Err(ForceFieldError::DuplicateLabel(built[i].0.label()));
                }
            }
        }

        // Displacement resolution: collect every claim against a label that
        // some other built slot carries, error on multi-claim conflicts,
        // and drop displaced constituents.
        let built_labels: Vec<&'static str> =
            built.iter().map(|(slot, _)| slot.label()).collect();
        let mut claimers_per_label: std::collections::HashMap<&'static str, Vec<&'static str>> =
            std::collections::HashMap::new();
        for (slot, displaces) in &built {
            for &target in *displaces {
                if built_labels.contains(&target) {
                    claimers_per_label
                        .entry(target)
                        .or_default()
                        .push(slot.label());
                }
            }
        }
        for (label, claimers) in &claimers_per_label {
            if claimers.len() > 1 {
                return Err(ForceFieldError::DisplaceConflict {
                    label,
                    by: claimers.clone(),
                });
            }
        }
        let displaced: std::collections::HashSet<&'static str> =
            claimers_per_label.keys().copied().collect();
        let mut slots: Vec<Box<dyn Potential>> = Vec::new();
        for (slot, _) in built.into_iter() {
            if displaced.contains(slot.label()) {
                continue;
            }
            slots.push(slot);
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

        // Fixed-point accumulators for the packed-neighbour pair-force path.
        let fast_total_forces_fp_x = device.alloc_zeros::<u64>(n).map_err(GpuError::from)?;
        let fast_total_forces_fp_y = device.alloc_zeros::<u64>(n).map_err(GpuError::from)?;
        let fast_total_forces_fp_z = device.alloc_zeros::<u64>(n).map_err(GpuError::from)?;
        let fast_total_potential_energies_fp =
            device.alloc_zeros::<u64>(n).map_err(GpuError::from)?;
        let fast_total_virials_fp = device.alloc_zeros::<u64>(n).map_err(GpuError::from)?;

        // Build the shared NeighborListState when any slot reports a cutoff.
        let aggregated_cutoff: Option<Real> = slots
            .iter()
            .filter_map(|s| s.max_cutoff())
            .fold(None::<Real>, |acc, c| Some(acc.map_or(c, |a| a.max(c))));
        let neighbor_list = if let Some(r_cut) = aggregated_cutoff {
            match neighbor_list_config {
                NeighborListConfig::CellList { r_skin } => Some(
                    NeighborListState::new_cell_list(
                        gpu,
                        sim_box,
                        particle_count,
                        r_cut,
                        LEGACY_FALLBACK_MAX_NEIGHBORS,
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

        // Collect each shape's participants from `jit_participant`. A
        // slot's single `JitParticipant` variant determines its shape
        // and carries its fragment + binding; a slot that returns `None`
        // is not JIT-composed and runs via `compute`. See
        // `rqm/forces/jit-composed-pair-force.md` and
        // `rqm/forces/jit-composed-intramolecular.md`.
        let mut jit_fragments: Vec<PairForceFragment> = Vec::new();
        let mut jit_slot_indices: Vec<usize> = Vec::new();
        let mut jit_bonded_fragments: Vec<BondedForceFragment> = Vec::new();
        let mut jit_bonded_slot_indices: Vec<usize> = Vec::new();
        let mut jit_angle_fragments: Vec<AngleForceFragment> = Vec::new();
        let mut jit_angle_slot_indices: Vec<usize> = Vec::new();
        for (idx, slot) in slots.iter().enumerate() {
            match slot.jit_participant() {
                Some(JitParticipant::PairForce(p)) => {
                    jit_fragments.push(p.pair_force_fragment());
                    jit_slot_indices.push(idx);
                }
                Some(JitParticipant::Bonded(b)) => {
                    jit_bonded_fragments.push(b.bonded_force_fragment());
                    jit_bonded_slot_indices.push(idx);
                }
                Some(JitParticipant::Angle(a)) => {
                    jit_angle_fragments.push(a.angle_force_fragment());
                    jit_angle_slot_indices.push(idx);
                }
                None => {}
            }
        }
        let jit_composed = if jit_fragments.is_empty() {
            None
        } else {
            // Every slot that contributed a fragment had
            // `max_cutoff().is_some()`, so `aggregated_cutoff` is Some
            // on this branch by construction. The JIT embeds the value
            // as `HEDDLE_JIT_MAX_CUTOFF_SQUARED` for the per-pair prune.
            let jit_max_cutoff = aggregated_cutoff
                .expect("aggregated_cutoff is Some when jit_fragments is non-empty");
            Some(JitComposedPairForce::compile_and_load(
                &device,
                &jit_fragments,
                jit_max_cutoff,
            )?)
        };
        // All fast-class pair-force slots in this codebase resolve
        // their `max_neighbors` from `NeighborListConfig` via
        // `max_neighbors_from(neighbor_list_config, particle_count)`,
        // which yields the same value for every slot. Re-derive it
        // once for the composed-kernel launch arg.
        let jit_max_neighbors: u32 =
            max_neighbors_from(neighbor_list_config, particle_count);

        // JIT compose the fast-class bonded module.
        // See `rqm/forces/jit-composed-intramolecular.md`.
        let jit_composed_bonded = if jit_bonded_fragments.is_empty() {
            None
        } else {
            Some(JitComposedBondedForce::compile_and_load(
                &device,
                &jit_bonded_fragments,
            )?)
        };

        // JIT compose the fast-class angle module.
        let jit_composed_angle = if jit_angle_fragments.is_empty() {
            None
        } else {
            Some(JitComposedAngleForce::compile_and_load(
                &device,
                &jit_angle_fragments,
            )?)
        };

        // Build the per-pair exclusion correction list. ExclusionList
        // stores canonical entries (atom_i < atom_j) with no
        // duplicates; pack them as interleaved (atom_i, atom_j) so the
        // JIT correction kernel reads one pair per thread.
        let mut excluded_pair_flat: Vec<u32> =
            Vec::with_capacity(2 * exclusion_list.entries.len());
        for excl in &exclusion_list.entries {
            excluded_pair_flat.push(excl.atom_i);
            excluded_pair_flat.push(excl.atom_j);
        }
        let excluded_pair_count = exclusion_list.entries.len() as u32;
        let excluded_pair_atoms = if excluded_pair_flat.is_empty() {
            // cudarc's alloc_zeros requires len > 0; use a single
            // zeroed slot so the kernel arg remains valid even with no
            // exclusions (the correction kernel is not launched when
            // count is 0, so the buffer contents are not read).
            device.alloc_zeros::<u32>(1).map_err(GpuError::from)?
        } else {
            device
                .htod_sync_copy(&excluded_pair_flat)
                .map_err(GpuError::from)?
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
            fast_total_forces_fp_x,
            fast_total_forces_fp_y,
            fast_total_forces_fp_z,
            fast_total_potential_energies_fp,
            fast_total_virials_fp,
            neighbor_list,
            jit_composed,
            excluded_pair_atoms,
            excluded_pair_count,
            jit_slot_indices,
            jit_max_neighbors,
            jit_composed_bonded,
            jit_bonded_slot_indices,
            jit_composed_angle,
            jit_angle_slot_indices,
            num_fast_slots,
            num_slow_slots,
            particle_count,
        })
    }

    /// The pair-force capability of the slot at `idx`. The index comes
    /// from `jit_slot_indices`, which `ForceField::new` populates only
    /// with pair-force participants, so the match is exhaustive in
    /// practice.
    fn pair_force_participant(&self, idx: usize) -> &dyn PairForcePotential {
        match self.slots[idx].jit_participant() {
            Some(JitParticipant::PairForce(p)) => p,
            _ => unreachable!("jit_slot_indices holds only pair-force participants"),
        }
    }

    /// The bonded capability of the slot at `idx` (from
    /// `jit_bonded_slot_indices`).
    fn bonded_participant(&self, idx: usize) -> &dyn BondedPotential {
        match self.slots[idx].jit_participant() {
            Some(JitParticipant::Bonded(b)) => b,
            _ => unreachable!("jit_bonded_slot_indices holds only bonded participants"),
        }
    }

    /// The angle capability of the slot at `idx` (from
    /// `jit_angle_slot_indices`).
    fn angle_participant(&self, idx: usize) -> &dyn AnglePotential {
        match self.slots[idx].jit_participant() {
            Some(JitParticipant::Angle(a)) => a,
            _ => unreachable!("jit_angle_slot_indices holds only angle participants"),
        }
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
    ///
    /// Returns `true` when the rebuild reallocated a packed-neighbour
    /// buffer; the batched-replay loop re-captures the phase graph in
    /// that case (see `rqm/cuda-graphs.md`). rq-1217c816
    pub fn run_neighbor_pre_step(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        timings: &mut Timings,
    ) -> Result<bool, ForceFieldError> {
        if let Some(nl) = self.neighbor_list.as_mut() {
            let outcome = nl.pre_step(sim_box, buffers, timings)?;
            return Ok(outcome.reallocated);
        }
        Ok(false)
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

        // Launch the JIT-composed pair-force kernel once for the
        // fast-class pair-force slots when (a) the framework has a
        // composed kernel built, (b) we're evaluating the Fast class,
        // and (c) the participating slot list is non-empty.
        let dispatch_jit = evaluating_fast
            && self.jit_composed.is_some()
            && !self.jit_slot_indices.is_empty();
        if dispatch_jit {
            // Zero the fixed-point Fast-class accumulators.
            self.device
                .memset_zeros(&mut self.fast_total_forces_fp_x)
                .map_err(GpuError::from)?;
            self.device
                .memset_zeros(&mut self.fast_total_forces_fp_y)
                .map_err(GpuError::from)?;
            self.device
                .memset_zeros(&mut self.fast_total_forces_fp_z)
                .map_err(GpuError::from)?;
            self.device
                .memset_zeros(&mut self.fast_total_potential_energies_fp)
                .map_err(GpuError::from)?;
            self.device
                .memset_zeros(&mut self.fast_total_virials_fp)
                .map_err(GpuError::from)?;

            // Refresh the tile-sorted position view for the current
            // step's positions.
            timings.kernel_start(KernelStage::SCATTER_POSITIONS_TO_TILE_ORDER)?;
            {
                let kernels = self.kernels.clone();
                let nl = self
                    .neighbor_list
                    .as_mut()
                    .expect("JIT pair-force kernel requires a shared neighbor list");
                // Split borrow: sorted_particle_ids and packed live on
                // disjoint fields of NeighborListState.
                let sorted_ptr: *const cudarc::driver::CudaSlice<u32> = nl
                    .sorted_particle_ids_for_packed()
                    .expect("packed-neighbour dispatch requires sorted_particle_ids");
                let packed = nl.packed.as_mut().expect("packed data present");
                let sorted_view = unsafe { &*sorted_ptr };
                let n_blocks = packed.n_blocks;
                crate::gpu::scatter_positions_to_tile_order(
                    &kernels,
                    buffers,
                    sorted_view,
                    &mut packed.tile_sorted_posq,
                )?;
                // Refill +∞ padding for partial last block.
                crate::gpu::fill_tile_position_padding(
                    &kernels,
                    &mut packed.tile_sorted_posq,
                    n as u32,
                    n_blocks * 32,
                )?;
            }
            timings.kernel_stop(KernelStage::SCATTER_POSITIONS_TO_TILE_ORDER)?;

            timings.kernel_start(KernelStage::JIT_COMPOSED_PAIR_FORCE)?;
            let nl = self
                .neighbor_list
                .as_ref()
                .expect("JIT pair-force kernel requires a shared neighbor list");
            let sorted_view = nl
                .sorted_particle_ids_for_packed()
                .expect("packed-neighbour dispatch requires sorted_particle_ids");
            let packed = nl.packed.as_ref().expect("packed data present");
            let n_iblocks = packed.n_blocks;
            let bind_ctx = PairForceBindContext {
                buffers: &*buffers,
                sim_box,
                neighbor_list: nl,
            };
            let mut launch_builder = ForceLaunchBuilder::new();
            // Common args, in the order the composer declares them.
            // `block_centre` and `block_bbox` are consumed by the
            // per-block single-periodic-copy fast-path check at the
            // top of the outer loop; the kernel decides per-block at
            // runtime whether to apply `triclinic_wrap_against_center`
            // to pi and pj and skip the per-pair `triclinic_min_image`
            // call. See `rqm/forces/packed-neighbour-pair-force.md`
            // *Single-Periodic-Copy Fast Path*.
            launch_builder.push_device_buffer(&buffers.posq);
            launch_builder.push_device_buffer(&packed.tile_sorted_posq);
            launch_builder.push_device_buffer(&packed.block_centre);
            launch_builder.push_device_buffer(&packed.block_bbox);
            launch_builder.push_device_buffer(sorted_view);
            launch_builder.push_device_buffer(&packed.iblock_offset);
            launch_builder.push_device_buffer(&packed.sorted_interacting_atoms);
            launch_builder.push_scalar(n_iblocks);
            launch_builder.push_device_buffer(sim_box.lattice_device());
            launch_builder.push_device_buffer(&self.fast_total_forces_fp_x);
            launch_builder.push_device_buffer(&self.fast_total_forces_fp_y);
            launch_builder.push_device_buffer(&self.fast_total_forces_fp_z);
            launch_builder.push_device_buffer(&self.fast_total_potential_energies_fp);
            launch_builder.push_device_buffer(&self.fast_total_virials_fp);
            // Per-fragment args in canonical slot order.
            for &slot_idx in &self.jit_slot_indices {
                self.pair_force_participant(slot_idx)
                    .bind_pair_force_args(&bind_ctx, &mut launch_builder);
            }
            // Trailing `n` arg.
            launch_builder.push_scalar(n as u32);

            let jit = self
                .jit_composed
                .as_ref()
                .expect("dispatch_jit implies jit_composed.is_some()");
            unsafe {
                jit.launch(n_iblocks, write_scalars, launch_builder)?;
            }
            timings.kernel_stop(KernelStage::JIT_COMPOSED_PAIR_FORCE)?;

            // Sparse-tile single-pair pass. The neighbour-list builder
            // routes (i-block, j-block) candidates with
            // `n_hits <= MAX_BITS_FOR_PAIRS = 3` into
            // `single_pair_atoms` instead of the packed buffer. The
            // launch covers `single_pairs_capacity` threads
            // unconditionally (so the kernel is captured into the
            // CUDA graph even when the post-capture count is zero);
            // each thread reads the live count from
            // `interaction_count[1]` via a device pointer and returns
            // early past the live boundary. The captured kernel thus
            // tolerates per-rebuild changes to the live count without
            // graph re-capture.
            let packed_opt = self
                .neighbor_list
                .as_ref()
                .and_then(|nl| nl.packed.as_ref());
            if let Some(packed) = packed_opt {
                if packed.single_pairs_capacity > 0 {
                    let mut single_pair_builder = ForceLaunchBuilder::new();
                    single_pair_builder.push_device_buffer(&buffers.posq);
                    single_pair_builder.push_device_buffer(&packed.single_pair_atoms);
                    single_pair_builder.push_device_buffer(&packed.interaction_count);
                    single_pair_builder.push_device_buffer(sim_box.lattice_device());
                    single_pair_builder.push_device_buffer(&self.fast_total_forces_fp_x);
                    single_pair_builder.push_device_buffer(&self.fast_total_forces_fp_y);
                    single_pair_builder.push_device_buffer(&self.fast_total_forces_fp_z);
                    single_pair_builder
                        .push_device_buffer(&self.fast_total_potential_energies_fp);
                    single_pair_builder.push_device_buffer(&self.fast_total_virials_fp);
                    for &slot_idx in &self.jit_slot_indices {
                        self.pair_force_participant(slot_idx)
                            .bind_pair_force_args(&bind_ctx, &mut single_pair_builder);
                    }
                    single_pair_builder.push_scalar(n as u32);
                    let cap = packed.single_pairs_capacity;
                    unsafe {
                        jit.launch_single_pair(cap, write_scalars, single_pair_builder)?;
                    }
                }
            }

            // Per-pair exclusion correction. The main pair-force kernel
            // above added `+1 × evaluate` for every pair (excluded or
            // not). For each excluded pair the correction kernel adds
            // `(scale − 1) × evaluate`, leaving `scale × evaluate` on
            // the fixed-point accumulators. When there are no
            // exclusions, this launch is skipped entirely.
            if self.excluded_pair_count > 0 {
                let mut correction_builder = ForceLaunchBuilder::new();
                // Common args for the correction entry point (order
                // must match emit_correction_entry_point).
                correction_builder.push_device_buffer(&buffers.posq);
                correction_builder.push_device_buffer(&self.excluded_pair_atoms);
                correction_builder.push_scalar(self.excluded_pair_count);
                correction_builder.push_device_buffer(sim_box.lattice_device());
                correction_builder.push_device_buffer(&self.fast_total_forces_fp_x);
                correction_builder.push_device_buffer(&self.fast_total_forces_fp_y);
                correction_builder.push_device_buffer(&self.fast_total_forces_fp_z);
                correction_builder.push_device_buffer(&self.fast_total_potential_energies_fp);
                correction_builder.push_device_buffer(&self.fast_total_virials_fp);
                // Per-fragment args in canonical slot order.
                for &slot_idx in &self.jit_slot_indices {
                    self.pair_force_participant(slot_idx)
                        .bind_pair_force_args(&bind_ctx, &mut correction_builder);
                }
                correction_builder.push_scalar(n as u32);
                unsafe {
                    jit.launch_correction(
                        self.excluded_pair_count,
                        write_scalars,
                        correction_builder,
                    )?;
                }
            }

            // Finalize: convert fixed-point sums to Real and add into
            // the existing fast-class Real accumulator buffers.
            timings.kernel_start(KernelStage::FINALIZE_PACKED_FORCES)?;
            {
                let kernels = self.kernels.clone();
                let mut fx = self.fast_total_forces_x.slice_mut(..);
                let mut fy = self.fast_total_forces_y.slice_mut(..);
                let mut fz = self.fast_total_forces_z.slice_mut(..);
                let mut fe = self.fast_total_potential_energies.slice_mut(..);
                let mut fw = self.fast_total_virials.slice_mut(..);
                crate::gpu::finalize_packed_forces(
                    &kernels,
                    &self.fast_total_forces_fp_x,
                    &self.fast_total_forces_fp_y,
                    &self.fast_total_forces_fp_z,
                    &self.fast_total_potential_energies_fp,
                    &self.fast_total_virials_fp,
                    &mut fx,
                    &mut fy,
                    &mut fz,
                    &mut fe,
                    &mut fw,
                    n as u32,
                    write_scalars,
                )?;
            }
            timings.kernel_stop(KernelStage::FINALIZE_PACKED_FORCES)?;
        }

        let nl_ref = self.neighbor_list.as_ref();

        // Launch the JIT-composed bonded module's per-slot entry
        // points. The composed kernel writes the per-bond contributions
        // into each slot's bond-pair scratch buffer; the slot's
        // `Potential::compute` then runs the universal per-atom
        // reduction kernel which sums those contributions into the
        // Fast-class accumulator.
        let dispatch_bonded = evaluating_fast
            && self.jit_composed_bonded.is_some()
            && !self.jit_bonded_slot_indices.is_empty();
        if dispatch_bonded {
            timings.kernel_start(KernelStage::JIT_COMPOSED_BONDED_FORCE)?;
            let bonded_jit = self
                .jit_composed_bonded
                .as_ref()
                .expect("dispatch_bonded implies jit_composed_bonded.is_some()");
            let bind_ctx = ForceLaunchContext {
                buffers: &*buffers,
                sim_box,
            };
            for (entry_idx, &slot_idx) in self.jit_bonded_slot_indices.iter().enumerate() {
                let scratch = self.bonded_participant(slot_idx).bonded_scratch();
                if scratch.bond_count == 0 {
                    continue;
                }
                let mut launch_builder = ForceLaunchBuilder::new();
                launch_builder.push_device_buffer(&buffers.posq);
                launch_builder.push_device_buffer(scratch.bonds);
                launch_builder.push_device_buffer(sim_box.lattice_device());
                launch_builder.push_device_buffer(scratch.bond_pair_x);
                launch_builder.push_device_buffer(scratch.bond_pair_y);
                launch_builder.push_device_buffer(scratch.bond_pair_z);
                if write_scalars {
                    launch_builder.push_device_buffer(scratch.bond_pair_energy);
                    launch_builder.push_device_buffer(scratch.bond_pair_virial);
                }
                self.bonded_participant(slot_idx)
                    .bind_bonded_force_args(&bind_ctx, &mut launch_builder);
                launch_builder.push_scalar(scratch.bond_count as u32);
                unsafe {
                    bonded_jit.launch_slot(
                        entry_idx,
                        scratch.bond_count as u32,
                        write_scalars,
                        launch_builder,
                    )?;
                }
            }
            timings.kernel_stop(KernelStage::JIT_COMPOSED_BONDED_FORCE)?;
        }

        // Launch the JIT-composed angle module's per-slot entry
        // points. Same pattern as bonded.
        let dispatch_angle = evaluating_fast
            && self.jit_composed_angle.is_some()
            && !self.jit_angle_slot_indices.is_empty();
        if dispatch_angle {
            timings.kernel_start(KernelStage::JIT_COMPOSED_ANGLE_FORCE)?;
            let angle_jit = self
                .jit_composed_angle
                .as_ref()
                .expect("dispatch_angle implies jit_composed_angle.is_some()");
            let bind_ctx = ForceLaunchContext {
                buffers: &*buffers,
                sim_box,
            };
            for (entry_idx, &slot_idx) in self.jit_angle_slot_indices.iter().enumerate() {
                let scratch = self.angle_participant(slot_idx).angle_scratch();
                if scratch.angle_count == 0 {
                    continue;
                }
                let mut launch_builder = ForceLaunchBuilder::new();
                launch_builder.push_device_buffer(&buffers.posq);
                launch_builder.push_device_buffer(scratch.angles);
                launch_builder.push_device_buffer(sim_box.lattice_device());
                launch_builder.push_device_buffer(scratch.angle_triple_x);
                launch_builder.push_device_buffer(scratch.angle_triple_y);
                launch_builder.push_device_buffer(scratch.angle_triple_z);
                if write_scalars {
                    launch_builder.push_device_buffer(scratch.angle_triple_energy);
                    launch_builder.push_device_buffer(scratch.angle_triple_virial);
                }
                self.angle_participant(slot_idx)
                    .bind_angle_force_args(&bind_ctx, &mut launch_builder);
                launch_builder.push_scalar(scratch.angle_count as u32);
                unsafe {
                    angle_jit.launch_slot(
                        entry_idx,
                        scratch.angle_count as u32,
                        write_scalars,
                        launch_builder,
                    )?;
                }
            }
            timings.kernel_stop(KernelStage::JIT_COMPOSED_ANGLE_FORCE)?;
        }

        // Per-slot compute path for slots NOT covered by the JIT
        // composed kernel (every slot whose index is not in
        // jit_slot_indices). The composed kernel already populated
        // the fast-class accumulator for every JIT slot; the remaining
        // slots ADD into their class accumulator via the SlotOutputView.
        let jit_idx_set: std::collections::HashSet<usize> =
            self.jit_slot_indices.iter().copied().collect();
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
        for (idx, slot) in slots.iter_mut().enumerate() {
            if jit_idx_set.contains(&idx) {
                continue;
            }
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

        // Step 6: device-side displacement-check kernel. Sets the
        // cell-list's `disp_rebuild_flag` to `1u` if any atom's
        // min-image displacement from its reference position exceeds
        // `r_skin / 2`. The flag is sticky across replays and is
        // cleared on rebuild. Queued here as the last per-step launch
        // so it lands inside any captured CUDA graph that includes
        // the force-evaluation sequence. See
        // `rqm/forces/neighbor-list.md` *Displacement Check*.
        if let Some(nl) = self.neighbor_list.as_mut() {
            nl.enqueue_displacement_check(sim_box, buffers, timings)?;
        }
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
