use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};

use crate::gpu::{GpuError, ParticleBuffers, morse_bond_force, reduce_bond_forces};
use crate::io::config::BondTypeConfig;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::bonds::BondList;
use super::{ForceFieldError, Potential, SlotForceView};

// rq-2361f2b8
#[derive(Debug)]
pub struct MorseBondedState {
    pub device: Arc<CudaDevice>,
    pub bonds: CudaSlice<u32>,
    pub atom_bond_offsets: CudaSlice<u32>,
    pub atom_bond_indices: CudaSlice<u32>,
    pub bond_de: CudaSlice<f32>,
    pub bond_a: CudaSlice<f32>,
    pub bond_re: CudaSlice<f32>,
    pub bond_pair_x: CudaSlice<f32>,
    pub bond_pair_y: CudaSlice<f32>,
    pub bond_pair_z: CudaSlice<f32>,
    pub bond_count: usize,
    pub particle_count: usize,
}

impl MorseBondedState {
    pub fn new(
        device: Arc<CudaDevice>,
        bond_list: &BondList,
        bond_types: &[BondTypeConfig],
    ) -> Result<Self, GpuError> {
        let bond_count = bond_list.bonds.len();
        let particle_count = bond_list.particle_count;

        let mut bonds_flat: Vec<u32> = Vec::with_capacity(3 * bond_count);
        for b in &bond_list.bonds {
            bonds_flat.push(b.atom_i);
            bonds_flat.push(b.atom_j);
            bonds_flat.push(b.bond_type_index);
        }

        let mut de_vec: Vec<f32> = Vec::with_capacity(bond_types.len());
        let mut a_vec: Vec<f32> = Vec::with_capacity(bond_types.len());
        let mut re_vec: Vec<f32> = Vec::with_capacity(bond_types.len());
        for bt in bond_types {
            match bt {
                BondTypeConfig::Morse { de, a, re, .. } => {
                    de_vec.push(*de as f32);
                    a_vec.push(*a as f32);
                    re_vec.push(*re as f32);
                }
            }
        }

        let bonds = htod_or_empty_u32(&device, &bonds_flat)?;
        let atom_bond_offsets = htod_or_empty_u32(&device, &bond_list.atom_bond_offsets)?;
        let atom_bond_indices = htod_or_empty_u32(&device, &bond_list.atom_bond_indices)?;
        let bond_de = htod_or_empty_f32(&device, &de_vec)?;
        let bond_a = htod_or_empty_f32(&device, &a_vec)?;
        let bond_re = htod_or_empty_f32(&device, &re_vec)?;

        let bond_pair_len = 2 * bond_count;
        let bond_pair_x = device
            .alloc_zeros::<f32>(bond_pair_len)
            .map_err(GpuError::from)?;
        let bond_pair_y = device
            .alloc_zeros::<f32>(bond_pair_len)
            .map_err(GpuError::from)?;
        let bond_pair_z = device
            .alloc_zeros::<f32>(bond_pair_len)
            .map_err(GpuError::from)?;

        Ok(MorseBondedState {
            device,
            bonds,
            atom_bond_offsets,
            atom_bond_indices,
            bond_de,
            bond_a,
            bond_re,
            bond_pair_x,
            bond_pair_y,
            bond_pair_z,
            bond_count,
            particle_count,
        })
    }
}

impl Potential for MorseBondedState {
    fn label(&self) -> &'static str {
        "morse_bonded"
    }

    fn contribute(
        &mut self,
        buffers: &ParticleBuffers,
        sim_box: &SimulationBox,
        timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        if self.bond_count == 0 {
            return Ok(());
        }
        timings.kernel_start(KernelStage::MorseBondForce)?;
        morse_bond_force(
            buffers,
            &self.bonds,
            &self.bond_de,
            &self.bond_a,
            &self.bond_re,
            sim_box,
            &mut self.bond_pair_x,
            &mut self.bond_pair_y,
            &mut self.bond_pair_z,
            self.bond_count,
        )?;
        timings.kernel_stop(KernelStage::MorseBondForce)?;
        Ok(())
    }

    fn reduce(
        &mut self,
        mut output: SlotForceView<'_>,
        timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        if self.particle_count == 0 {
            return Ok(());
        }
        if self.bond_count == 0 {
            self.device.memset_zeros(&mut output.x).map_err(GpuError::from)?;
            self.device.memset_zeros(&mut output.y).map_err(GpuError::from)?;
            self.device.memset_zeros(&mut output.z).map_err(GpuError::from)?;
            return Ok(());
        }
        timings.kernel_start(KernelStage::ReduceBondForces)?;
        reduce_bond_forces(
            &self.device,
            &self.bond_pair_x,
            &self.bond_pair_y,
            &self.bond_pair_z,
            &self.atom_bond_offsets,
            &self.atom_bond_indices,
            &mut output.x,
            &mut output.y,
            &mut output.z,
            self.particle_count,
        )?;
        timings.kernel_stop(KernelStage::ReduceBondForces)?;
        Ok(())
    }
}

fn htod_or_empty_u32(
    device: &Arc<CudaDevice>,
    data: &[u32],
) -> Result<CudaSlice<u32>, GpuError> {
    if data.is_empty() {
        device.alloc_zeros::<u32>(0).map_err(GpuError::from)
    } else {
        device.htod_sync_copy(data).map_err(GpuError::from)
    }
}

fn htod_or_empty_f32(
    device: &Arc<CudaDevice>,
    data: &[f32],
) -> Result<CudaSlice<f32>, GpuError> {
    if data.is_empty() {
        device.alloc_zeros::<f32>(0).map_err(GpuError::from)
    } else {
        device.htod_sync_copy(data).map_err(GpuError::from)
    }
}
