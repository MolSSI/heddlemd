pub mod bonds;
pub mod lj;
pub mod morse;
pub mod neighbor_list;

use std::sync::Arc;

use cudarc::driver::CudaDevice;

use crate::gpu::{GpuError, ParticleBuffers, accumulate_forces};
use crate::io::config::{BondTypeConfig, NeighborListConfig, PairInteractionConfig};
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings, TimingsError};

pub use bonds::{
    Bond, BondList, BondsFileError, DeviceExclusionList, Exclusion, ExclusionList,
    load_bonds_file,
};
pub use lj::LennardJonesState;
pub use morse::MorseBondedState;
pub use neighbor_list::{NeighborListError, NeighborListState};

#[derive(Debug)]
pub enum PotentialSlot {
    LennardJones(LennardJonesState),
    MorseBonded(MorseBondedState),
}

#[derive(Debug)]
pub enum ForceFieldError {
    Gpu(GpuError),
    Timings(TimingsError),
    NeighborList(NeighborListError),
}

impl std::fmt::Display for ForceFieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForceFieldError::Gpu(e) => write!(f, "Gpu({e})"),
            ForceFieldError::Timings(e) => write!(f, "Timings({e})"),
            ForceFieldError::NeighborList(e) => write!(f, "NeighborList({e})"),
        }
    }
}

impl std::error::Error for ForceFieldError {}

impl From<GpuError> for ForceFieldError {
    fn from(e: GpuError) -> Self {
        ForceFieldError::Gpu(e)
    }
}

impl From<TimingsError> for ForceFieldError {
    fn from(e: TimingsError) -> Self {
        ForceFieldError::Timings(e)
    }
}

impl From<NeighborListError> for ForceFieldError {
    fn from(e: NeighborListError) -> Self {
        ForceFieldError::NeighborList(e)
    }
}

#[derive(Debug)]
pub struct ForceField {
    pub device: Arc<CudaDevice>,
    pub slots: Vec<PotentialSlot>,
}

impl ForceField {
    pub fn new(
        device: Arc<CudaDevice>,
        particle_count: usize,
        sim_box: &SimulationBox,
        pair_interactions: &[PairInteractionConfig],
        bond_types: &[BondTypeConfig],
        bond_list: &BondList,
        exclusion_list: &ExclusionList,
        neighbor_list_config: &NeighborListConfig,
    ) -> Result<Self, ForceFieldError> {
        let mut slots: Vec<PotentialSlot> = Vec::new();

        // Slot 0: Lennard-Jones (always present).
        let pair = pair_interactions
            .first()
            .expect("pair_interactions must contain at least one entry");
        let lj_params = crate::gpu::LennardJonesParameters {
            sigma: pair.sigma as f32,
            epsilon: pair.epsilon as f32,
            cutoff: pair.cutoff as f32,
        };
        let lj_state = LennardJonesState::new(
            device.clone(),
            particle_count,
            *sim_box,
            lj_params,
            exclusion_list,
            neighbor_list_config,
        )?;
        slots.push(PotentialSlot::LennardJones(lj_state));

        // Slot 1: Morse bonded (only when at least one bond exists).
        if !bond_list.is_empty() {
            let morse_state =
                MorseBondedState::new(device.clone(), bond_list, bond_types)?;
            slots.push(PotentialSlot::MorseBonded(morse_state));
        }

        Ok(ForceField { device, slots })
    }

    pub fn step(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &SimulationBox,
        timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }

        // Phase 1: per-slot contribution kernels.
        for slot in self.slots.iter_mut() {
            match slot {
                PotentialSlot::LennardJones(s) => {
                    s.contribute(buffers, sim_box, timings)?;
                }
                PotentialSlot::MorseBonded(s) => {
                    s.contribute(buffers, sim_box, timings)?;
                }
            }
        }

        // Phase 2: per-slot reduction kernels.
        for slot in self.slots.iter_mut() {
            match slot {
                PotentialSlot::LennardJones(s) => {
                    s.reduce(timings)?;
                }
                PotentialSlot::MorseBonded(s) => {
                    s.reduce(timings)?;
                }
            }
        }

        // Phase 3: combiner.
        let mut slot0 = None;
        let mut slot1 = None;
        for (i, slot) in self.slots.iter().enumerate() {
            let acc = match slot {
                PotentialSlot::LennardJones(s) => s.accumulator(),
                PotentialSlot::MorseBonded(s) => s.accumulator(),
            };
            match i {
                0 => slot0 = Some(acc),
                1 => slot1 = Some(acc),
                _ => unreachable!("framework supports at most two slots in v1"),
            }
        }
        timings.kernel_start(KernelStage::AccumulateForces)?;
        accumulate_forces(buffers, slot0, slot1)?;
        timings.kernel_stop(KernelStage::AccumulateForces)?;
        Ok(())
    }
}
