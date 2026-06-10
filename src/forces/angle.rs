use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice};
use cudarc::nvrtc::Ptx;

use crate::gpu::device::get_func;
use crate::gpu::{
    GpuContext, GpuError, Kernels, ParticleBuffers, harmonic_angle_force, reduce_angle_forces,
};
use crate::kernels;
use crate::io::config::AngleTypeConfig;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::topology::AngleList;
use super::{
    AggregateLevel, ForceFieldError, Potential, PotentialBuildContext, PotentialBuilder,
    SlotOutputView,
};

// rq-21a8063c
#[derive(Debug)]
pub struct HarmonicAngleState {
    pub device: Arc<CudaDevice>,
    pub kernels: Arc<Kernels>,
    pub angles: CudaSlice<u32>,
    pub atom_angle_offsets: CudaSlice<u32>,
    pub atom_angle_indices: CudaSlice<u32>,
    pub angle_k_theta: CudaSlice<f32>,
    pub angle_theta_0: CudaSlice<f32>,
    pub angle_triple_x: CudaSlice<f32>,
    pub angle_triple_y: CudaSlice<f32>,
    pub angle_triple_z: CudaSlice<f32>,
    pub angle_triple_energy: CudaSlice<f32>,
    pub angle_triple_virial: CudaSlice<f32>,
    pub angle_count: usize,
    pub particle_count: usize,
}

impl HarmonicAngleState {
    pub fn new(
        gpu: &GpuContext,
        angle_list: &AngleList,
        angle_types: &[AngleTypeConfig],
    ) -> Result<Self, GpuError> {
        let device = gpu.device.clone();
        let kernels = gpu.kernels.clone();
        let angle_count = angle_list.angles.len();
        let particle_count = angle_list.particle_count;

        // Flatten angles to [atom_i, atom_j, atom_k, type_idx] quadruples.
        let mut angles_flat: Vec<u32> = Vec::with_capacity(4 * angle_count);
        for a in &angle_list.angles {
            angles_flat.push(a.atom_i);
            angles_flat.push(a.atom_j);
            angles_flat.push(a.atom_k);
            angles_flat.push(a.angle_type_index);
        }

        let mut k_vec: Vec<f32> = Vec::with_capacity(angle_types.len());
        let mut theta0_vec: Vec<f32> = Vec::with_capacity(angle_types.len());
        for at in angle_types {
            match at {
                AngleTypeConfig::Harmonic { k_theta, theta_0, .. } => {
                    k_vec.push(*k_theta as f32);
                    theta0_vec.push(*theta_0 as f32);
                }
            }
        }

        let angles = htod_or_empty_u32(&device, &angles_flat)?;
        let atom_angle_offsets = htod_or_empty_u32(&device, &angle_list.atom_angle_offsets)?;
        let atom_angle_indices = htod_or_empty_u32(&device, &angle_list.atom_angle_indices)?;
        let angle_k_theta = htod_or_empty_f32(&device, &k_vec)?;
        let angle_theta_0 = htod_or_empty_f32(&device, &theta0_vec)?;

        let triple_len = 3 * angle_count;
        let angle_triple_x = device.alloc_zeros::<f32>(triple_len).map_err(GpuError::from)?;
        let angle_triple_y = device.alloc_zeros::<f32>(triple_len).map_err(GpuError::from)?;
        let angle_triple_z = device.alloc_zeros::<f32>(triple_len).map_err(GpuError::from)?;
        let angle_triple_energy =
            device.alloc_zeros::<f32>(triple_len).map_err(GpuError::from)?;
        let angle_triple_virial =
            device.alloc_zeros::<f32>(triple_len).map_err(GpuError::from)?;

        Ok(HarmonicAngleState {
            device,
            kernels,
            angles,
            atom_angle_offsets,
            atom_angle_indices,
            angle_k_theta,
            angle_theta_0,
            angle_triple_x,
            angle_triple_y,
            angle_triple_z,
            angle_triple_energy,
            angle_triple_virial,
            angle_count,
            particle_count,
        })
    }
}

impl Potential for HarmonicAngleState {
    fn label(&self) -> &'static str {
        "harmonic_angle"
    }

    fn max_cutoff(&self) -> Option<f32> {
        None
    }

    fn contribute(
        &mut self,
        buffers: &ParticleBuffers,
        sim_box: &SimulationBox,
        _cx: &crate::forces::ForceFieldContext<'_>,
        timings: &mut Timings,
    ) -> Result<(), ForceFieldError> {
        if self.angle_count == 0 {
            return Ok(());
        }
        timings.kernel_start(KernelStage::HARMONIC_ANGLE_FORCE)?;
        harmonic_angle_force(
            buffers,
            &self.angles,
            &self.angle_k_theta,
            &self.angle_theta_0,
            sim_box,
            &mut self.angle_triple_x,
            &mut self.angle_triple_y,
            &mut self.angle_triple_z,
            &mut self.angle_triple_energy,
            &mut self.angle_triple_virial,
            self.angle_count,
        )?;
        timings.kernel_stop(KernelStage::HARMONIC_ANGLE_FORCE)?;
        Ok(())
    }

    fn reduce(
        &mut self,
        mut output: SlotOutputView<'_>,
        _cx: &crate::forces::ForceFieldContext<'_>,
        timings: &mut Timings,
        _level: AggregateLevel,
    ) -> Result<(), ForceFieldError> {
        // Harmonic-angle reduction is a small kernel; it writes all
        // five output rows on every call regardless of `level`.
        if self.particle_count == 0 {
            return Ok(());
        }
        if self.angle_count == 0 {
            self.device.memset_zeros(&mut output.force_x).map_err(GpuError::from)?;
            self.device.memset_zeros(&mut output.force_y).map_err(GpuError::from)?;
            self.device.memset_zeros(&mut output.force_z).map_err(GpuError::from)?;
            self.device.memset_zeros(&mut output.energy).map_err(GpuError::from)?;
            self.device.memset_zeros(&mut output.virial).map_err(GpuError::from)?;
            return Ok(());
        }
        timings.kernel_start(KernelStage::REDUCE_ANGLE_FORCES)?;
        reduce_angle_forces(
            &self.kernels,
            &self.angle_triple_x,
            &self.angle_triple_y,
            &self.angle_triple_z,
            &self.angle_triple_energy,
            &self.angle_triple_virial,
            &self.atom_angle_offsets,
            &self.atom_angle_indices,
            &mut output.force_x,
            &mut output.force_y,
            &mut output.force_z,
            &mut output.energy,
            &mut output.virial,
            self.particle_count,
        )?;
        timings.kernel_stop(KernelStage::REDUCE_ANGLE_FORCES)?;
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

// rq-e8550f96
#[derive(Debug)]
pub struct HarmonicAngleBuilder;

impl PotentialBuilder for HarmonicAngleBuilder {
    fn build(
        &self,
        cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        if cx.angle_list.is_empty() {
            return Ok(None);
        }
        let state = HarmonicAngleState::new(cx.gpu, cx.angle_list, cx.angle_types)?;
        Ok(Some(Box::new(state)))
    }
}

// rq-2093594f
#[derive(Debug, Clone)]
pub struct AngleKernels {
    pub harmonic_angle_force: CudaFunction,
    pub reduce_angle_forces: CudaFunction,
}

impl AngleKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::ANGLE),
            "angle",
            &["harmonic_angle_force", "reduce_angle_forces"],
        )?;
        Ok(AngleKernels {
            harmonic_angle_force: get_func(device, "angle", "harmonic_angle_force")?,
            reduce_angle_forces: get_func(device, "angle", "reduce_angle_forces")?,
        })
    }
}
