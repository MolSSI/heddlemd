// rq-891232bf

use cudarc::driver::CudaSlice;

use crate::gpu::{
    GpuContext, GpuError, ParticleBuffers, compute_kinetic_energy, rescale_velocities,
};
use crate::io::config::ThermostatKind;
use crate::io::log_output::BOLTZMANN_J_PER_K;
use crate::timings::{KernelStage, Timings};

use super::philox::philox_normal;
use super::{Thermostat, ThermostatBuilder, ThermostatError};

// rq-47d91c7d
#[derive(Debug)]
pub struct CsvrThermostat {
    pub temperature: f64,
    pub tau: f64,
    pub seed: u64,
    pub draw_counter: u64,
    pub g_dof: u32,
    pub kt_target: f64,
    pub cumulative_injection: f64,
    ke_scratch: CudaSlice<f32>,
    most_recent_ke: f64,
}

impl CsvrThermostat {
    fn new(
        gpu: &GpuContext,
        particle_count: usize,
        temperature: f64,
        tau: f64,
        seed: u64,
    ) -> Result<Self, GpuError> {
        let g_dof = ((3 * particle_count) as i64 - 3).max(1) as u32;
        let kt_target = BOLTZMANN_J_PER_K * temperature;
        let ke_scratch = gpu.device.alloc_zeros::<f32>(1).map_err(GpuError::from)?;
        Ok(CsvrThermostat {
            temperature,
            tau,
            seed,
            draw_counter: 0,
            g_dof,
            kt_target,
            cumulative_injection: 0.0,
            ke_scratch,
            most_recent_ke: 0.0,
        })
    }

    fn draw_new_kinetic_energy(&self, k_old: f64, dt: f32) -> f64 {
        let c = (-(dt as f64) / self.tau).exp();
        let nf = self.g_dof as f64;
        let k_target = (nf / 2.0) * self.kt_target;
        let one_minus_c = 1.0 - c;

        let seed_lo = self.seed as u32;
        let seed_hi = (self.seed >> 32) as u32;
        let ctr_lo = self.draw_counter as u32;
        let ctr_hi = (self.draw_counter >> 32) as u32;

        let r = philox_normal(seed_lo, seed_hi, ctr_lo, ctr_hi, 0, 0);
        let mut s = 0.0_f64;
        for sample_index in 1..self.g_dof {
            let xi = philox_normal(seed_lo, seed_hi, ctr_lo, ctr_hi, sample_index, 0);
            s += xi * xi;
        }

        let cross = if k_old > 0.0 {
            2.0 * r * (c * one_minus_c * k_old * k_target / nf).sqrt()
        } else {
            0.0
        };
        let k_new = c * k_old + (k_target / nf) * one_minus_c * (s + r * r) + cross;
        if k_new.is_finite() && k_new > 0.0 {
            k_new
        } else {
            k_old
        }
    }
}

impl Thermostat for CsvrThermostat {
    // rq-7a124d43
    fn apply_post(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), ThermostatError> {
        if buffers.particle_count() == 0 {
            return Ok(());
        }

        timings.kernel_start(KernelStage::KINETIC_ENERGY_REDUCE)?;
        let k_old = compute_kinetic_energy(buffers, &mut self.ke_scratch)? as f64;
        timings.kernel_stop(KernelStage::KINETIC_ENERGY_REDUCE)?;

        self.draw_counter += 1;
        let k_new = self.draw_new_kinetic_energy(k_old, dt);
        self.cumulative_injection += k_new - k_old;
        self.most_recent_ke = k_new;

        if k_old > 0.0 && (k_new - k_old).abs() > 0.0 {
            let factor = (k_new / k_old).sqrt() as f32;
            timings.kernel_start(KernelStage::CSVR_RESCALE_VELOCITIES)?;
            rescale_velocities(buffers, factor)?;
            timings.kernel_stop(KernelStage::CSVR_RESCALE_VELOCITIES)?;
        }

        Ok(())
    }

    // rq-8ee58ec1
    fn log_column_names(&self) -> &'static [&'static str] {
        &["csvr_conserved"]
    }

    // rq-2a5de2ab
    fn log_column_values(
        &self,
        kinetic_energy: f64,
        potential_energy: f64,
    ) -> Vec<f64> {
        vec![kinetic_energy + potential_energy - self.cumulative_injection]
    }
}

// rq-750b828f
#[derive(Debug)]
pub struct CsvrBuilder;

impl ThermostatBuilder for CsvrBuilder {
    fn kind_name(&self) -> &'static str {
        "csvr"
    }

    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        kind: &ThermostatKind,
    ) -> Result<Box<dyn Thermostat>, ThermostatError> {
        match kind {
            ThermostatKind::Csvr {
                temperature,
                tau,
                seed,
            } => {
                let state = CsvrThermostat::new(gpu, particle_count, *temperature, *tau, *seed)?;
                Ok(Box::new(state))
            }
            other => Err(ThermostatError::UnknownKind(other.name().to_string())),
        }
    }
}
