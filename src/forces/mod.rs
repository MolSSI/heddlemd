pub mod bonds;
pub mod lj;
pub mod morse;
pub mod neighbor_list;

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, CudaViewMut};

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

// rq-67ebf3b1
pub trait Potential: std::fmt::Debug + Send {
    fn label(&self) -> &'static str;

    fn contribute(
        &mut self,
        buffers: &ParticleBuffers,
        sim_box: &SimulationBox,
        timings: &mut Timings,
    ) -> Result<(), ForceFieldError>;

    fn reduce(
        &mut self,
        output: SlotForceView<'_>,
        timings: &mut Timings,
    ) -> Result<(), ForceFieldError>;
}

// rq-304b191b
pub struct SlotForceView<'a> {
    pub x: CudaViewMut<'a, f32>,
    pub y: CudaViewMut<'a, f32>,
    pub z: CudaViewMut<'a, f32>,
}

// rq-a2e20b02
#[derive(Debug)]
pub enum ForceFieldError {
    Gpu(GpuError),
    Timings(TimingsError),
    NeighborList(NeighborListError),
    DuplicateLabel(&'static str),
}

impl std::fmt::Display for ForceFieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForceFieldError::Gpu(e) => write!(f, "Gpu({e})"),
            ForceFieldError::Timings(e) => write!(f, "Timings({e})"),
            ForceFieldError::NeighborList(e) => write!(f, "NeighborList({e})"),
            ForceFieldError::DuplicateLabel(l) => write!(f, "DuplicateLabel({l:?})"),
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

// rq-684a29f1
#[derive(Debug)]
pub struct ForceField {
    pub device: Arc<CudaDevice>,
    pub slots: Vec<Box<dyn Potential>>,
    pub slot_forces_x: CudaSlice<f32>,
    pub slot_forces_y: CudaSlice<f32>,
    pub slot_forces_z: CudaSlice<f32>,
    particle_count: usize,
}

impl ForceField {
    // rq-79938dbf
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
        let mut slots: Vec<Box<dyn Potential>> = Vec::new();

        // Slot 0: Lennard-Jones when at least one pair interaction is configured.
        if let Some(pair) = pair_interactions.first() {
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
            slots.push(Box::new(lj_state));
        }

        // Slot 1: Morse bonded when at least one bond is present.
        if !bond_list.is_empty() {
            let morse_state =
                MorseBondedState::new(device.clone(), bond_list, bond_types)?;
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

        Ok(ForceField {
            device,
            slots,
            slot_forces_x,
            slot_forces_y,
            slot_forces_z,
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

        for slot in self.slots.iter_mut() {
            slot.contribute(buffers, sim_box, timings)?;
        }

        let num_slots = self.slots.len();
        let slots = &mut self.slots;
        let sfx = &mut self.slot_forces_x;
        let sfy = &mut self.slot_forces_y;
        let sfz = &mut self.slot_forces_z;
        for k in 0..num_slots {
            let start = k * n;
            let end = (k + 1) * n;
            let view = SlotForceView {
                x: sfx.slice_mut(start..end),
                y: sfy.slice_mut(start..end),
                z: sfz.slice_mut(start..end),
            };
            slots[k].reduce(view, timings)?;
        }

        timings.kernel_start(KernelStage::AccumulateForces)?;
        accumulate_forces(
            buffers,
            &self.slot_forces_x,
            &self.slot_forces_y,
            &self.slot_forces_z,
            num_slots as u32,
        )?;
        timings.kernel_stop(KernelStage::AccumulateForces)?;
        Ok(())
    }
}
