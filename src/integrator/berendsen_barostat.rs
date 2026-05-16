// rq-0d8c8688

use cudarc::driver::CudaSlice;

use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, compute_kinetic_energy, compute_total_virial,
    rescale_positions,
};
use crate::io::config::BarostatKind;
use crate::pbc::SimulationBox;
use crate::timings::{KernelStage, Timings};

use super::{Barostat, BarostatBuilder, BarostatError};

// rq-0d8c8688
#[derive(Debug)]
pub struct BerendsenBarostat {
    pub pressure: f64,
    pub tau: f64,
    pub compressibility: f64,
    pub most_recent_pressure: f64,
    pub most_recent_volume: f64,
    ke_scratch: CudaSlice<f32>,
    virial_scratch: CudaSlice<f32>,
}

impl BerendsenBarostat {
    fn new(
        gpu: &GpuContext,
        _particle_count: usize,
        pressure: f64,
        tau: f64,
        compressibility: f64,
    ) -> Result<Self, GpuError> {
        let ke_scratch = gpu.device.alloc_zeros::<f32>(1).map_err(GpuError::from)?;
        let virial_scratch = gpu.device.alloc_zeros::<f32>(1).map_err(GpuError::from)?;
        Ok(BerendsenBarostat {
            pressure,
            tau,
            compressibility,
            most_recent_pressure: 0.0,
            most_recent_volume: 0.0,
            ke_scratch,
            virial_scratch,
        })
    }
}

// Host-side safety floor on μ. Sensible parameters never approach it;
// the floor only triggers under pathological combinations
// (β · dt/τ · (P_target − P) > 1).
const MU_MIN: f64 = 1.0e-6;

impl Barostat for BerendsenBarostat {
    // rq-1179e42f
    fn apply(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), BarostatError> {
        if buffers.particle_count() == 0 {
            // Pre-populate the diagnostic fields so log_column_values
            // still returns finite numbers for an empty run.
            self.most_recent_pressure = 0.0;
            self.most_recent_volume = sim_box.volume() as f64;
            return Ok(());
        }

        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;

        timings.kernel_start(KernelStage::VIRIAL_SUM_REDUCE)?;
        let w = compute_total_virial(buffers, &mut self.virial_scratch)? as f64;
        timings.kernel_stop(KernelStage::VIRIAL_SUM_REDUCE)?;

        let v_pre = sim_box.volume() as f64;
        let pressure = (2.0 * k + w) / (3.0 * v_pre);

        let mu_cubed = 1.0
            - self.compressibility * ((dt as f64) / self.tau) * (self.pressure - pressure);
        let mu_cubed_clamped = mu_cubed.max(MU_MIN * MU_MIN * MU_MIN);
        let mu = mu_cubed_clamped.cbrt();
        let mu_f32 = mu as f32;

        timings.kernel_start(KernelStage::BERENDSEN_BAROSTAT_RESCALE_POSITIONS)?;
        rescale_positions(buffers, mu_f32)?;
        timings.kernel_stop(KernelStage::BERENDSEN_BAROSTAT_RESCALE_POSITIONS)?;

        // Bumps generation; downstream consumers refresh on next force step.
        sim_box
            .rescale_isotropic(mu_f32)
            .map_err(|_| BarostatError::Gpu(GpuError(
                cudarc::driver::DriverError(
                    cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE,
                ),
            )))?;

        self.most_recent_pressure = pressure;
        self.most_recent_volume = sim_box.volume() as f64;
        Ok(())
    }

    // rq-62b44dc9
    fn log_column_names(&self) -> &'static [&'static str] {
        &["pressure", "box_volume"]
    }

    // rq-62b44dc9
    fn log_column_values(
        &self,
        _kinetic_energy: f64,
        _potential_energy: f64,
    ) -> Vec<f64> {
        vec![self.most_recent_pressure, self.most_recent_volume]
    }
}

#[derive(Debug)]
pub struct BerendsenBarostatBuilder;

impl BarostatBuilder for BerendsenBarostatBuilder {
    fn kind_name(&self) -> &'static str {
        "berendsen"
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        kind: &BarostatKind,
    ) -> Result<Box<dyn Barostat>, BarostatError> {
        match kind {
            BarostatKind::Berendsen {
                pressure,
                tau,
                compressibility,
            } => {
                let state = BerendsenBarostat::new(
                    gpu,
                    particle_count,
                    *pressure,
                    *tau,
                    *compressibility,
                )?;
                Ok(Box::new(state))
            }
            other => Err(BarostatError::UnknownKind(other.name().to_string())),
        }
    }
}
