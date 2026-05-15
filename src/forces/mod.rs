pub mod bonds;
pub mod coulomb;
pub mod lj;
pub mod morse;
pub mod neighbor_list;
pub mod spme;

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, CudaViewMut};

use crate::gpu::{
    GpuContext, GpuError, Kernels, LennardJonesParameterTable, ParticleBuffers, accumulate_forces,
};
use crate::io::config::{
    BondTypeConfig, CoulombConfig, NeighborListConfig, PairInteractionConfig, ParticleTypeConfig,
};
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings, TimingsError};

pub use bonds::{
    Bond, BondList, BondsFileError, DeviceExclusionList, Exclusion, ExclusionList,
    load_bonds_file,
};
pub use coulomb::{CoulombParameters, CoulombState};
pub use spme::{SpmeError, SpmeParameters, SpmeReciprocalGrid};
pub use lj::LennardJonesState;
pub use morse::MorseBondedState;
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
        gpu: &GpuContext,
        particle_count: usize,
        sim_box: &SimulationBox,
        particle_types: &[ParticleTypeConfig],
        pair_interactions: &[PairInteractionConfig],
        bond_types: &[BondTypeConfig],
        coulomb_config: Option<&CoulombConfig>,
        bond_list: &BondList,
        exclusion_list: &ExclusionList,
        neighbor_list_config: &NeighborListConfig,
    ) -> Result<Self, ForceFieldError> {
        let device = gpu.device.clone();
        let kernels = gpu.kernels.clone();
        let mut slots: Vec<Box<dyn Potential>> = Vec::new();

        let max_neighbors = match neighbor_list_config {
            NeighborListConfig::AllPairs => particle_count as u32,
            NeighborListConfig::CellList { max_neighbors, .. } => *max_neighbors,
        };

        // Slot 0: Lennard-Jones when at least one pair interaction is configured.
        if !pair_interactions.is_empty() {
            let params = LennardJonesParameterTable::from_config(
                &device,
                particle_types,
                pair_interactions,
            )?;
            let max_cutoff = pair_interactions
                .iter()
                .map(|p| p.cutoff as f32)
                .fold(0.0_f32, f32::max);
            let lj_state = LennardJonesState::new(
                gpu,
                particle_count,
                params,
                max_cutoff,
                max_neighbors,
                exclusion_list,
            )?;
            slots.push(Box::new(lj_state));
        }

        // Slot 1: Coulomb when the [coulomb] table is present in the config.
        if let Some(coul) = coulomb_config {
            let params = CoulombParameters::from(coul);
            let coul_state = CoulombState::new(
                gpu,
                particle_count,
                params,
                max_neighbors,
                exclusion_list,
            )?;
            slots.push(Box::new(coul_state));
        }

        // Slot 2: Morse bonded when at least one bond is present.
        if !bond_list.is_empty() {
            let morse_state = MorseBondedState::new(gpu, bond_list, bond_types)?;
            slots.push(Box::new(morse_state));
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
            let cx = ForceFieldContext { neighbor_list: nl_ref };
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
            let cx = ForceFieldContext { neighbor_list: nl_ref };
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
